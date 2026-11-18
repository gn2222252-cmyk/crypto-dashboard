"""Historical study "big_buyer": bitcoin spot ETF data covers less than two years. To test the "big buyer flow ->
price" mechanism on bitcoin over a longer span, use the largest single corporate buyer, Strategy (formerly
MicroStrategy, MSTR), whose purchases since August 2020 are publicly disclosed (date of filing / announcement, BTC
bought, average price, period covered). Owner's hypothesis family: buying follows price (they buy more after rises /
when the stock premium is high), sustained buying pushes price, and when buying slows or stops the price gives back.
Related idea: flow-driven price impact + return-chasing feedback loop (van der Beck, Bouchaud & Villamaina,
https://arxiv.org/abs/2405.12768).

Pre-registered tests (verbatim from the owner):

1. Does buying follow price? Probability/size of a purchase announcement in week w vs BTC return and MSTR premium
   (MSTR market cap ÷ BTC holdings value, if computable from public data) over the prior 1–4 weeks.
2. Announcement effect: BTC return over [−5,−1], [0,+1], [+1,+5], [+5,+20] days around announcements, by purchase size
   tercile; vs placebo same-weekday dates without announcements (or random Mondays). Note: buying itself happens during
   the covered period BEFORE the announcement — so also measure BTC return over the covered period vs matched
   non-buying periods.
3. Buying streaks and their end: runs of consecutive weekly purchase announcements (≥3 weeks); BTC return during the
   run vs after the run ends (next 20/60 days) vs same-length price-only up-runs ending (to separate "buyer stopped"
   from ordinary momentum reversal).
4. Tradable rule (announcement-based, act at the close after the announcement is public): hold BTC while Strategy has
   announced a purchase in the last k weeks (k=2 pre-set; also report k=4 as description), else cash; vs holding, vs
   ma50, vs a placebo of shuffled announcement weeks (200 draws), with costs 0.10%/0.30%; also as a filter on ma50.
   Pass = beats ma50 on return÷volatility in both halves (2020-08..2023-06 and 2023-07..now) and placebo share ≤ 10%.
Uncertainty: block bootstrap; count independent events. Split by period (2020–2022 vs 2023–2026). Mention that once
ETFs existed the buyer is a smaller share of flow — compare purchase USD with same-week ETF net flows (description).
Today: date and size of the latest announcement, whether a streak is running — states only.

Details fixed by the author before looking at any result:
- Data: DefiLlama's public digital-asset-treasury page for MSTR (key-less; every row links the 8-K or the company page
  it comes from). strategy.com answers "Access Denied" to the dashboard's honest user agent and SEC EDGAR's document
  archive refuses it as an undeclared automated tool, so neither is used. Two rows whose dates in the compilation are
  wrong against the company's own dated press releases are corrected (OVERRIDES, each with its source); one of them
  would otherwise have been dated five days BEFORE the public knew. Rows with the same announcement date are merged.
- Timing: an announcement dated d (8-Ks come out before the US open, earnings releases after the close, all before
  00:00 UTC of d+1) is acted on at the bitcoin daily close dated d (00:00 UTC of d+1); the position earns from d+1.
  Perturbation: one more day of delay. Event windows in days relative to d: [a,b] = close[d+b] / close[d+a-1] - 1.
- Weeks are Monday–Sunday. Test 1 predictors are measured at the Sunday close before the announcement week (r1 = last
  week, r4 = last four weeks, premium = latest value at that time); a second version ends one week earlier (before the
  buying itself, which happens the week before a Monday announcement). Terciles over all weeks; block bootstrap of
  weeks (block 8, 1,000 draws) for "top tercile minus bottom tercile".
- Size = BTC bought ÷ BTC newly mined in the covered period (900 a day, 450 after the 2024-04-20 halving).
- Placebo for test 2: dates with the same weekday, more than 3 days from any announcement, in the same sub-period,
  1,000 draws. Covered period: periods of at most 14 days, against random windows of the same length that do not touch
  any covered period, same sub-period, 1,000 draws. 90% intervals: block bootstrap over announcements in time order
  (block 4). Independent episodes: announcements more than 20 days apart.
- Test 3: a streak ends when a Monday–Sunday week passes without an announcement; "after" is measured from that
  Sunday's close (when the end is known). Price-only up-runs: at least 3 consecutive weeks with a higher Sunday close,
  measured from the Sunday close of the first week that is not up.
- Test 4: k weeks = 7k days; return÷volatility = mean daily net return ÷ its standard deviation × √365; all rules
  start 2020-08-12. Placebo: the same number of announcement weeks drawn at random from all weeks of the sample
  (announcement on that week's Monday), 200 draws, fixed seed.
- Premium = MSTR close (Yahoo adjusted) × fully diluted realized share count (DefiLlama, ×10 before the 2024-08-08
  split) ÷ (BTC held per announcements known by then × bitcoin close).
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

LLAMA_PAGE = "https://defillama.com/digital-asset-treasury/MSTR"
STORE_FILE = common.STORE / "big_buyer_defillama.json"
ETF_STORE = common.STORE / "etf_flows_btc.json"
PAPER = "https://arxiv.org/abs/2405.12768"
FARSIDE = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
START = "2020-08-11"
HALF = ("2020-08-12", "2023-06-30", "2023-07-01")
SPLIT = "2023-01-01"  # 2020–2022 vs 2023–2026
SPLIT_DAY = "2024-08-08"  # MSTR 10-for-1 stock split
HALVING = "2024-04-20"
WINDOWS = ((-5, -1), (0, 1), (1, 5), (5, 20))
COSTS = (0.001, 0.003)
SIMS, PLACEBO, BOOT = 1000, 200, 1000
SEED = 20200811
YEAR = 365
# (announcement date in the compilation, BTC amount) -> (true public date, period start, period end, source, url)
OVERRIDES = {
    ("2021-02-19", 19452): ("2021-02-24", "2021-02-19", "2021-02-24", "Business Wire 新聞稿 2021-02-24",
                            "https://www.businesswire.com/news/home/20210224005406/en/MicroStrategy-Acquires-Additional-19452-Bitcoins-for-1.026-Billion"),
    ("2021-01-28", 29646): ("2020-12-21", "2020-12-11", "2020-12-21", "The Block 2020-12-21",
                            "https://www.theblock.co/news/ecosystems/2020-12-21-microstrategy-purchases-another-29646-bitcoin-via-convertible-debt-88699"),
}


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v, digits: int = 1) -> str:
    if not _ok(v):
        return "—"
    s = f"{float(v) * 100:+.{digits}f}%"
    return s.replace("-0.0%", "0.0%").replace("+0.0%", "0.0%") if digits == 1 else s


def _sh(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _n(v, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}" if _ok(v) else "—"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _usd(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    sign = "−" if v < 0 else ""
    v = abs(v)
    return sign + (f"{v / 1e8:,.1f} 億美元" if v >= 1e8 else f"{v / 1e6:,.1f} 百萬美元")


# ---------- data ----------

def _llama_fetch() -> dict:
    reply = requests.get(LLAMA_PAGE, headers=common.HEADERS, timeout=60)
    time.sleep(common.PAUSE)
    if reply.status_code in (401, 403, 429, 503):
        raise RuntimeError(f"DefiLlama 回應 HTTP {reply.status_code}；依規定不繞過")
    reply.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', reply.text, re.S)
    if not m:
        raise RuntimeError("DefiLlama 頁面裡找不到資料區塊，格式改了")
    pp = json.loads(m.group(1))["props"]["pageProps"]
    tx = pp.get("transactions") or []
    if len(tx) < 50:
        raise RuntimeError(f"DefiLlama 只有 {len(tx)} 筆交易")
    keep = ("amount", "avg_price", "usd_value", "start_date", "end_date", "report_date", "type", "source_type",
            "source_url", "is_approved", "asset")
    out = {"fetched": common.today_utc(), "transactions": [{k: x.get(k) for k in keep} for x in tx],
           "shares": pp.get("fdChart", {}).get("dataset", {}).get("source", []),
           "mnav": pp.get("mNAVChart", {}).get("dataset", {}).get("source", [])}
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        STORE_FILE.write_text(json.dumps(out), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store is only a fallback
        pass
    return out


def _llama() -> tuple[dict, str]:
    try:
        return _llama_fetch(), ""
    except Exception as exc:  # noqa: BLE001
        if not STORE_FILE.exists():
            raise
        data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
        return data, f"這次 DefiLlama 取不到（{type(exc).__name__}），改用 {data.get('fetched', '上次')} 存下的資料。"


def _records(raw: list) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Purchases merged by announcement date (and sales separately), after the two documented corrections."""
    rows, fixes = [], []
    for x in raw:
        if x.get("asset") != "bitcoin" or not x.get("is_approved", True):
            continue
        rep, a, b = str(x["report_date"])[:10], str(x["start_date"])[:10], str(x["end_date"])[:10]
        amt = float(x["amount"] or 0)
        key = (rep, int(round(amt)))
        if key in OVERRIDES:
            new = OVERRIDES[key]
            fixes.append(f"{int(amt):,} 顆：彙整寫 {rep}，改成 {new[0]}（{new[3]}）")
            rep, a, b = new[0], new[1], new[2]
        b = min(b, rep)  # a typo such as end 2025-02-06 for a 2024-02-06 release
        a = min(a, b)
        usd = float(x["usd_value"] or 0) or amt * float(x["avg_price"] or 0)
        rows.append({"report": pd.Timestamp(rep), "start": pd.Timestamp(a), "end": pd.Timestamp(b), "btc": amt,
                     "usd": usd, "type": x["type"], "src": x.get("source_type") or "", "url": x.get("source_url") or ""})
    f = pd.DataFrame(rows)
    buys = f[f["type"] == "purchase"].groupby("report").agg(start=("start", "min"), end=("end", "max"),
                                                             btc=("btc", "sum"), usd=("usd", "sum"),
                                                             src=("src", "first")).reset_index()
    sales = f[f["type"] == "sale"].groupby("report").agg(start=("start", "min"), end=("end", "max"),
                                                          btc=("btc", "sum"), usd=("usd", "sum")).reset_index()
    return buys.sort_values("report").reset_index(drop=True), sales.sort_values("report").reset_index(drop=True), fixes


def _mined(a, b) -> float:
    days = pd.date_range(pd.Timestamp(a), pd.Timestamp(b), freq="D")
    return float(np.where(days < pd.Timestamp(HALVING), 900.0, 450.0).sum())


def _holdings(buys: pd.DataFrame, sales: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """BTC held as the public knew it on each day (changes on announcement dates)."""
    ch = pd.concat([buys.set_index("report")["btc"], -sales.set_index("report")["btc"]]).groupby(level=0).sum()
    return ch.reindex(index.union(ch.index), fill_value=0.0).cumsum().reindex(index)


def _premium(llama: dict, mstr: pd.Series, btc: pd.Series, held: pd.Series) -> pd.Series:
    sh = llama.get("shares") or []
    if not sh:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r[0], unit="ms").normalize(): float(r[1]) for r in sh if r and _ok(r[1])})
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.where(s.index >= pd.Timestamp(SPLIT_DAY), s * 10.0)
    days = mstr.index
    shares = s.reindex(days.union(s.index)).ffill().reindex(days)
    mcap = mstr * shares
    value = held.reindex(days, method="ffill") * btc.reindex(days, method="ffill")
    prem = (mcap / value.where(value > 0)).dropna()
    return prem[prem.index >= pd.Timestamp(START)]


def _etf() -> pd.Series:
    try:
        d = json.loads(ETF_STORE.read_text(encoding="utf-8"))
        s = pd.Series({pd.Timestamp(k): float(v) for k, v in d.items() if _ok(v)}).sort_index()
        return s * 1e6
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


# ---------- helpers ----------

def _ret(close: pd.Series, a, b) -> float | None:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a < close.index[0] or b > close.index[-1]:
        return None
    return float(close[b] / close[a] - 1)


def _win(close: pd.Series, d, lo: int, hi: int) -> float | None:
    d = pd.Timestamp(d)
    return _ret(close, d + pd.Timedelta(days=lo - 1), d + pd.Timedelta(days=hi))


def _episodes(days: list, gap: int = 20) -> int:
    n, last = 0, None
    for d in sorted(days):
        if last is None or (d - last).days > gap:
            n += 1
        last = d
    return n


def _block_ci(vals: np.ndarray, rng, block: int = 4) -> tuple[float | None, float | None]:
    v = vals[np.isfinite(vals)]
    n = len(v)
    if n < 5:
        return None, None
    k = int(math.ceil(n / block))
    means = []
    for _ in range(BOOT):
        st = rng.integers(0, max(1, n - block + 1), k)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        means.append(v[idx[idx < n]].mean())
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def _period(d) -> int:
    return 0 if pd.Timestamp(d) < pd.Timestamp(SPLIT) else 1


# ---------- test 1 ----------

def _weeks(close: pd.Series, buys: pd.DataFrame, prem: pd.Series) -> pd.DataFrame:
    last_sun = close.index[-1] - pd.Timedelta(days=(close.index[-1].weekday() + 1) % 7)
    first_sun = pd.Timestamp(START) + pd.Timedelta(days=6 - pd.Timestamp(START).weekday())
    suns = pd.date_range(first_sun, last_sun, freq="7D")
    w = pd.DataFrame(index=suns)
    wk = buys["report"] + pd.to_timedelta((6 - buys["report"].dt.weekday), unit="D")
    usd = buys.groupby(wk)["usd"].sum()
    w["usd"] = usd.reindex(suns).fillna(0.0)
    w["buy"] = (w["usd"] > 0).astype(float)
    for lag, tag in ((7, ""), (14, "_pre")):
        e = suns - pd.Timedelta(days=lag)
        c = close.reindex(e).values
        w["r1" + tag] = c / close.reindex(e - pd.Timedelta(days=7)).values - 1
        w["r4" + tag] = c / close.reindex(e - pd.Timedelta(days=28)).values - 1
        w["prem" + tag] = prem.reindex(prem.index.union(e)).ffill().reindex(e).values if len(prem) else np.nan
    return w


def _tercile_table(w: pd.DataFrame, col: str, rng) -> dict:
    x = w[col]
    m = x.notna()
    q1, q2 = np.nanpercentile(x[m], [100 / 3, 200 / 3])
    grp = pd.Series(np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), index=w.index).where(m)
    out = {"cuts": (q1, q2), "rows": []}
    per = pd.Series([_period(d) for d in w.index], index=w.index)
    for g in range(3):
        sel = grp == g
        r = {"n": int(sel.sum()), "p": float(w.loc[sel, "buy"].mean()),
             "p0": float(w.loc[sel & (per == 0), "buy"].mean()) if (sel & (per == 0)).any() else None,
             "p1": float(w.loc[sel & (per == 1), "buy"].mean()) if (sel & (per == 1)).any() else None,
             "size": float(w.loc[sel & (w["buy"] > 0), "usd"].mean()) if (sel & (w["buy"] > 0)).any() else None}
        out["rows"].append(r)
    g = grp.values
    b = w["buy"].values
    ok = ~np.isnan(g)
    diff = float(b[ok & (g == 2)].mean() - b[ok & (g == 0)].mean())
    n, block = len(b), 8
    k = int(math.ceil(n / block))
    sims = []
    for _ in range(BOOT):
        st = rng.integers(0, n - block + 1, k)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        gg, bb = g[idx], b[idx]
        hi, lo = bb[gg == 2], bb[gg == 0]
        if len(hi) and len(lo):
            sims.append(hi.mean() - lo.mean())
    out["diff"] = diff
    out["ci"] = (float(np.percentile(sims, 5)), float(np.percentile(sims, 95)))
    for p in (0, 1):
        s = (per == p).values & ok
        hi, lo = b[s & (g == 2)], b[s & (g == 0)]
        out[f"diff{p}"] = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None
    return out


# ---------- test 2 ----------

def _event_test(close: pd.Series, buys: pd.DataFrame, rng) -> dict:
    last = close.index[-1]
    ann = list(buys["report"])
    pool = pd.date_range(pd.Timestamp(START), last - pd.Timedelta(days=20), freq="D")
    far = np.ones(len(pool), dtype=bool)
    for d in ann:
        far &= np.abs((pool - d).days) > 3
    pool = pool[far]
    ev = buys.copy()
    ev["size"] = [r.btc / _mined(r.start, r.end) for r in ev.itertuples()]
    for lo, hi in WINDOWS:
        ev[f"w{lo}_{hi}"] = [_win(close, d, lo, hi) for d in ev["report"]]
    ev["per"] = [_period(d) for d in ev["report"]]
    q1, q2 = np.percentile(ev["size"], [100 / 3, 200 / 3])
    ev["terc"] = np.where(ev["size"] <= q1, 0, np.where(ev["size"] <= q2, 1, 2))
    # placebo pools by (weekday, period)
    pools = {}
    for d in pool:
        pools.setdefault((d.weekday(), _period(d)), []).append(d)
    pw = {}
    for lo, hi in WINDOWS:
        vals = {d: _win(close, d, lo, hi) for d in pool}
        pw[(lo, hi)] = vals
    placebo = {}
    for lo, hi in WINDOWS:
        col = f"w{lo}_{hi}"
        sub = ev[ev[col].notna()]
        keys = [(d.weekday(), p) for d, p in zip(sub["report"], sub["per"])]
        arrs = {k: np.array([pw[(lo, hi)][d] for d in pools.get(k, []) if _ok(pw[(lo, hi)][d])]) for k in set(keys)}
        sims = np.zeros(SIMS)
        for k in keys:
            a = arrs[k]
            sims += rng.choice(a, SIMS) if len(a) else 0.0
        sims /= max(1, len(keys))
        placebo[(lo, hi)] = {"mean": float(sims.mean()), "ge": float((sims >= sub[col].mean()).mean()),
                             "ci": _block_ci(sub[col].values.astype(float), rng)}
    # covered period vs matched non-buying windows
    cov = ev[((ev["end"] - ev["start"]).dt.days <= 13) & (ev["start"] > close.index[0])].copy()
    cov["len"] = (cov["end"] - cov["start"]).dt.days + 1
    cov["ret"] = [_ret(close, s - pd.Timedelta(days=1), e) for s, e in zip(cov["start"], cov["end"])]
    covered = np.zeros(len(close), dtype=bool)
    for r in buys.itertuples():
        a, b = close.index.searchsorted(r.start), close.index.searchsorted(r.end)
        covered[a:b + 1] = True
    cum = np.concatenate([[0], np.cumsum(covered)])
    starts = {}
    lr = np.log(close.values)
    i0 = int(close.index.searchsorted(pd.Timestamp(START)))
    for L in cov["len"].unique():
        for p in (0, 1):
            cand = []
            for i in range(max(i0, 1), len(close) - L):
                if cum[i + L] - cum[i] == 0 and _period(close.index[i]) == p:
                    cand.append(math.exp(lr[i + L - 1] - lr[i - 1]) - 1)
            starts[(L, p)] = np.array(cand)
    sims = np.zeros(SIMS)
    used = 0
    for L, p in zip(cov["len"], cov["per"]):
        a = starts.get((L, p), np.array([]))
        if len(a):
            sims += rng.choice(a, SIMS)
            used += 1
    sims /= max(1, used)
    cov_res = {"n": len(cov), "mean": float(cov["ret"].mean()), "placebo": float(sims.mean()),
               "ge": float((sims >= cov["ret"].mean()).mean()), "up": float((cov["ret"] > 0).mean()),
               "ci": _block_ci(cov["ret"].values.astype(float), rng),
               "by_per": {p: (int((cov["per"] == p).sum()), float(cov.loc[cov["per"] == p, "ret"].mean())
                              if (cov["per"] == p).any() else None) for p in (0, 1)}}
    return {"ev": ev, "placebo": placebo, "cov": cov_res, "cuts": (q1, q2)}


# ---------- test 3 ----------

def _runs(flag: pd.Series, min_len: int = 3) -> list:
    out, cur = [], []
    for d, v in flag.items():
        if v:
            cur.append(d)
        else:
            if len(cur) >= min_len:
                out.append((cur[0], cur[-1], len(cur), True))
            cur = []
    if len(cur) >= min_len:
        out.append((cur[0], cur[-1], len(cur), False))  # still running
    return out


def _streaks(close: pd.Series, w: pd.DataFrame, buys: pd.DataFrame) -> dict:
    runs = _runs(w["buy"] > 0)
    rows = []
    for first, lastw, n, ended in runs:
        inrun = buys[(buys["report"] >= first - pd.Timedelta(days=6)) & (buys["report"] <= lastw)]
        a = inrun["start"].min() - pd.Timedelta(days=1)
        b = inrun["report"].max()
        known = lastw + pd.Timedelta(days=7)
        rows.append({"first": inrun["report"].min(), "last": b, "weeks": n, "usd": float(inrun["usd"].sum()),
                     "btc": float(inrun["btc"].sum()), "during": _ret(close, a, b), "ended": ended, "known": known,
                     "a20": _ret(close, known, known + pd.Timedelta(days=20)) if ended else None,
                     "a60": _ret(close, known, known + pd.Timedelta(days=60)) if ended else None})
    wk = close.reindex(w.index)
    up = (wk / close.reindex(w.index - pd.Timedelta(days=7)).values - 1) > 0
    pruns = _runs(pd.Series(up.values, index=w.index))
    prow = []
    for first, lastw, n, ended in pruns:
        if not ended:
            continue
        known = lastw + pd.Timedelta(days=7)
        prow.append({"weeks": n, "during": _ret(close, first - pd.Timedelta(days=7), lastw),
                     "a20": _ret(close, known, known + pd.Timedelta(days=20)),
                     "a60": _ret(close, known, known + pd.Timedelta(days=60))})
    base = {}
    sub = close[close.index >= pd.Timestamp(START)]
    for h in (20, 60):
        base[h] = float((sub.shift(-h) / sub - 1).dropna().mean())
    return {"rows": rows, "price": prow, "base": base}


# ---------- test 4 ----------

def _hold_from(index: pd.DatetimeIndex, days, k: int) -> pd.Series:
    """1 at the close of day t if a purchase was announced on a day d with t-7k < d <= t."""
    ev = pd.Series(1.0, index=pd.DatetimeIndex(sorted(set(days))))
    ev = ev.reindex(index.union(ev.index), fill_value=0.0)
    cnt = ev.rolling(f"{7 * k}D").sum().reindex(index)
    return (cnt > 0.5).astype(float)


def _metrics(net: np.ndarray) -> dict:
    eq = np.cumprod(1 + net)
    yrs = len(net) / YEAR
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 and yrs > 0 else None
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * math.sqrt(YEAR)) if sd > 0 else None
    deep = float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min())
    return {"cagr": cagr, "sharpe": sharpe, "deep": deep}


def _evaluate(target: pd.Series, change: pd.Series, keep: np.ndarray, halves: list, cost: float, lag: int = 1) -> dict:
    pos = target.shift(lag).fillna(0.0).values
    net, _ = te._run(pos, change.values, cost)
    out = _metrics(net[keep])
    out["halves"] = [_metrics(net[h])["sharpe"] for h in halves]
    out["time"] = float(pos[keep].mean())
    return out


def _rule_test(close: pd.Series, buys: pd.DataFrame, w: pd.DataFrame, rng) -> dict:
    change = close.pct_change().fillna(0.0)
    idx = close.index
    keep = idx >= pd.Timestamp(HALF[0])
    halves = [keep & (idx <= pd.Timestamp(HALF[1])), idx >= pd.Timestamp(HALF[2])]
    ma = close.rolling(50, min_periods=50).mean()
    ma50 = (close > ma).astype(float)
    days = list(buys["report"])
    res = {"hold": {c: _evaluate(pd.Series(1.0, index=idx), change, keep, halves, c) for c in COSTS},
           "ma50": {c: _evaluate(ma50, change, keep, halves, c) for c in COSTS}}
    for k in (2, 4):
        t = _hold_from(idx, days, k)
        res[f"k{k}"] = {c: _evaluate(t, change, keep, halves, c) for c in COSTS}
        res[f"f{k}"] = {c: _evaluate(t * ma50, change, keep, halves, c) for c in COSTS}
    t2 = _hold_from(idx, days, 2)
    res["k2_lag2"] = _evaluate(t2, change, keep, halves, COSTS[0], lag=2)
    res["f2_lag2"] = _evaluate(t2 * ma50, change, keep, halves, COSTS[0], lag=2)
    # placebo: shuffled announcement weeks
    n_weeks = int((w["buy"] > 0).sum())
    mondays = w.index - pd.Timedelta(days=6)
    sims, sims_f, sims_half = [], [], []
    for _ in range(PLACEBO):
        pick = rng.choice(len(mondays), n_weeks, replace=False)
        t = _hold_from(idx, list(mondays[pick]), 2)
        r = _evaluate(t, change, keep, halves, COSTS[0])
        rf = _evaluate(t * ma50, change, keep, halves, COSTS[0])
        sims.append(r["sharpe"] if _ok(r["sharpe"]) else -9)
        sims_f.append(rf["sharpe"] if _ok(rf["sharpe"]) else -9)
        sims_half.append(r["halves"])
    sims, sims_f = np.array(sims), np.array(sims_f)
    real, real_f = res["k2"][COSTS[0]]["sharpe"], res["f2"][COSTS[0]]["sharpe"]
    res["placebo"] = {"median": float(np.median(sims)), "ge": float((sims >= real).mean()),
                      "median_f": float(np.median(sims_f)), "ge_f": float((sims_f >= real_f).mean()),
                      "time": None}
    k2, m = res["k2"][COSTS[0]], res["ma50"][COSTS[0]]
    beats = [(_ok(a) and _ok(b) and a > b) for a, b in zip(k2["halves"], m["halves"])]
    res["pass"] = bool(all(beats) and res["placebo"]["ge"] <= 0.10)
    res["beats"] = beats
    res["target"] = t2
    return res


def _lookahead(close: pd.Series, buys: pd.DataFrame) -> dict:
    """Announcement never before its period ends; positions on data cut at 2023-06-30 equal the full-data ones."""
    bad = int((buys["report"] < buys["end"]).sum())
    cut = pd.Timestamp("2023-06-30")
    part = close[close.index <= cut]
    full_t = _hold_from(close.index, list(buys["report"]), 2)[close.index <= cut]
    part_t = _hold_from(part.index, list(buys.loc[buys["report"] <= cut, "report"]), 2)
    return {"bad": bad, "same": bool(np.array_equal(full_t.values, part_t.values))}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    problems: list[str] = []
    llama, prob = _llama()
    if prob:
        problems.append(prob)
    buys, sales, fixes = _records(llama["transactions"])
    close = te._daily_close()
    close = close[close.index >= pd.Timestamp("2020-01-01")]
    asof = close.index[-1]
    buys = buys[buys["report"] <= asof].reset_index(drop=True)
    sales = sales[sales["report"] <= asof].reset_index(drop=True)
    held = _holdings(buys, sales, close.index)
    prem = pd.Series(dtype=float)
    try:
        mstr = ta._share_close("MSTR")
        prem = _premium(llama, mstr, close, held)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"MSTR 股價或股數這次取不到（{type(exc).__name__}），溢價部分留空。")
    w = _weeks(close, buys, prem)

    # ---------- test 1 ----------
    t1 = {c: _tercile_table(w, c, rng) for c in ("r1", "r4", "prem", "r4_pre", "prem_pre") if w[c].notna().sum() > 30}
    names = {"r1": "前 1 週比特幣漲跌", "r4": "前 4 週比特幣漲跌", "prem": "MSTR 溢價（市值÷持幣價值）",
             "r4_pre": "前 4 週漲跌（再早一週，買進之前）", "prem_pre": "溢價（再早一週，買進之前）"}
    rows1 = []
    for c in ("r1", "r4", "prem"):
        if c not in t1:
            continue
        x = t1[c]
        q1, q2 = x["cuts"]
        fmt = (lambda v: f"{v:.2f} 倍") if c == "prem" else _p
        labels = (f"低（≤ {fmt(q1)}）", f"中（{fmt(q1)} 到 {fmt(q2)}）", f"高（> {fmt(q2)}）")
        for g, r in enumerate(x["rows"]):
            rows1.append([names[c] if g == 0 else "", labels[g], f"{r['n']}", _sh(r["p"]), _sh(r["p0"]), _sh(r["p1"]),
                          _usd(r["size"])])
        rows1.append(["", "高減低（90% 區間）", "—", f"{x['diff'] * 100:+.0f} 點（{x['ci'][0] * 100:+.0f} 到 {x['ci'][1] * 100:+.0f}）",
                      f"{x['diff0'] * 100:+.0f} 點" if _ok(x["diff0"]) else "—",
                      f"{x['diff1'] * 100:+.0f} 點" if _ok(x["diff1"]) else "—", "—"])
    for c in ("r4_pre", "prem_pre"):
        if c in t1:
            x = t1[c]
            rows1.append([names[c], "高減低（90% 區間）", "—",
                          f"{x['diff'] * 100:+.0f} 點（{x['ci'][0] * 100:+.0f} 到 {x['ci'][1] * 100:+.0f}）",
                          f"{x['diff0'] * 100:+.0f} 點" if _ok(x["diff0"]) else "—",
                          f"{x['diff1'] * 100:+.0f} 點" if _ok(x["diff1"]) else "—", "—"])

    # ---------- test 2 ----------
    et = _event_test(close, buys, rng)
    ev = et["ev"]
    cols = [f"w{lo}_{hi}" for lo, hi in WINDOWS]

    def ev_row(label, sub):
        cells = []
        for c in cols:
            v = sub[c].dropna()
            cells.append(f"{_p(v.mean())}（漲 {_sh((v > 0).mean())}）" if len(v) else "—")
        return [label, f"{len(sub)} 次／{_episodes(list(sub['report']))} 段"] + cells

    rows2 = [ev_row("全部買進公告", ev)]
    q1, q2 = et["cuts"]
    for g, lab in enumerate((f"買得少（≤ 新挖出 {q1 * 100:.0f}%）", f"中（{q1 * 100:.0f}% 到 {q2 * 100:.0f}%）",
                             f"買得多（> 新挖出 {q2 * 100:.0f}%）")):
        rows2.append(ev_row(lab, ev[ev["terc"] == g]))
    rows2.append(ev_row("2020–2022 的公告", ev[ev["per"] == 0]))
    rows2.append(ev_row("2023–2026 的公告", ev[ev["per"] == 1]))
    pl = et["placebo"]
    rows2.append(["對照：同星期幾、沒有公告的日子（隨機 1,000 次）", "—"]
                 + [_p(pl[wd]["mean"]) for wd in WINDOWS])
    rows2.append(["全部公告的 90% 區間（區塊重抽）｜隨機對照平均一樣高或更高的比例", "—"]
                 + [f"{_p(pl[wd]['ci'][0])} 到 {_p(pl[wd]['ci'][1])}｜{_sh(pl[wd]['ge'])}" for wd in WINDOWS])
    cv = et["cov"]
    rows2.append([f"買進期間本身（期間 ≤ 14 天的 {cv['n']} 次，從期間開始前一天收盤到期間最後一天收盤）", f"{cv['n']} 次",
                  f"平均 {_p(cv['mean'])}（漲 {_sh(cv['up'])}）", f"同長度、沒有買進的期間 {_p(cv['placebo'])}",
                  f"90% 區間 {_p(cv['ci'][0])} 到 {_p(cv['ci'][1])}", f"隨機對照一樣高或更高 {_sh(cv['ge'])}"])

    # ---------- test 3 ----------
    st = _streaks(close, w, buys)
    rows3 = []
    for r in st["rows"][-12:]:
        rows3.append([f"{_d(r['first'])} 到 {_d(r['last'])}", f"{r['weeks']} 週", _usd(r["usd"]), _p(r["during"]),
                      (_p(r["a20"]) if _ok(r["a20"]) else "還不到 20 天") if r["ended"] else "還在進行",
                      (_p(r["a60"]) if _ok(r["a60"]) else "還不到 60 天") if r["ended"] else "—"])
    done = [r for r in st["rows"] if r["ended"]]

    def avg(rs, k):
        v = [r[k] for r in rs if _ok(r[k])]
        return (float(np.mean(v)), sum(x > 0 for x in v), len(v)) if v else (None, 0, 0)

    for lab, rs in (("連續買進段：平均", done),
                    ("對照：比特幣連續 3 週以上上漲（不管有沒有買）結束後", st["price"])):
        d_ = avg(rs, "during")
        a20, a60 = avg(rs, "a20"), avg(rs, "a60")
        rows3.append([lab, f"{len(rs)} 段", "—", _p(d_[0]), f"{_p(a20[0])}（{a20[1]}／{a20[2]} 段漲）",
                      f"{_p(a60[0])}（{a60[1]}／{a60[2]} 段漲）"])
    rows3.append(["對照：2020-08 以來任何一天之後", "—", "—", "—", _p(st["base"][20]), _p(st["base"][60])])

    # ---------- test 4 ----------
    rt = _rule_test(close, buys, w, rng)
    rows4 = []

    def r4(label, key, plc="—"):
        a, b = rt[key][COSTS[0]], rt[key][COSTS[1]]
        rows4.append([label, f"{_p(a['cagr'])}／{_p(b['cagr'])}", _n(a["sharpe"]),
                      f"{_n(a['halves'][0])}｜{_n(a['halves'][1])}", _p(a["deep"], 0), _sh(a["time"]), plc])

    r4("一直抱著", "hold")
    r4("收盤高於 50 日均線才持有", "ma50")
    pl4 = rt["placebo"]
    r4("近 2 週有買進公告才持有（事先定的規則）", "k2",
       f"打亂公告週 {PLACEBO} 次：中位 {_n(pl4['median'])}；{_sh(pl4['ge'])} 一樣好或更好")
    r4("近 4 週有買進公告才持有（只描述）", "k4")
    r4("50 日均線＋近 2 週有公告（兩者都要）", "f2",
       f"打亂公告週：中位 {_n(pl4['median_f'])}；{_sh(pl4['ge_f'])} 一樣好或更好")
    r4("50 日均線＋近 4 週有公告（只描述）", "f4")
    for lab, key in (("近 2 週有公告：晚一天才照做", "k2_lag2"), ("50 日均線＋近 2 週：晚一天才照做", "f2_lag2")):
        a = rt[key]
        rows4.append([lab, f"{_p(a['cagr'])}／—", _n(a["sharpe"]), f"{_n(a['halves'][0])}｜{_n(a['halves'][1])}",
                      _p(a["deep"], 0), _sh(a["time"]), "—"])

    look = _lookahead(close, buys)

    # ---------- ETF comparison and size ----------
    etf = _etf()
    etf_cmp = None
    if len(etf):
        rr = []
        for r in buys.itertuples():
            if r.report < pd.Timestamp("2024-01-15") or (r.end - r.start).days > 13:
                continue
            f = etf[(etf.index >= r.start) & (etf.index <= r.end)]
            if len(f):
                rr.append((r.usd, float(f.sum()), float(f.abs().sum())))
        if rr:
            a = np.array(rr)
            etf_cmp = {"n": len(a), "buy": float(a[:, 0].sum()), "net": float(a[:, 1].sum()),
                       "ratio_med": float(np.median(a[:, 0] / np.maximum(a[:, 2], 1.0))),
                       "bigger": int((a[:, 0] > np.abs(a[:, 1])).sum())}
            all_net = float(etf[etf.index >= pd.Timestamp("2024-01-11")].sum())
            all_buy = float(buys.loc[buys["report"] >= pd.Timestamp("2024-01-11"), "usd"].sum())
            etf_cmp.update({"all_net": all_net, "all_buy": all_buy})
    size_per = {p: float(ev.loc[ev["per"] == p, "btc"].sum() / sum(_mined(r.start, r.end) for r in ev[ev["per"] == p].itertuples()))
                for p in (0, 1) if (ev["per"] == p).any()}

    # ---------- now ----------
    lastb = buys.iloc[-1]
    wk_buy = w["buy"] > 0
    streak = 0
    for v in reversed(wk_buy.values):
        if v:
            streak += 1
        else:
            break
    cur_week_sun = asof + pd.Timedelta(days=(6 - asof.weekday()))
    in_current = lastb["report"] > w.index[-1]
    if in_current:
        streak += 1
    running = (asof - lastb["report"]).days <= 8
    now = [
        {"label": "最近一次買進公告", "value": (f"{_d(lastb['report'])}：{lastb['btc']:,.0f} 顆，約 {_usd(lastb['usd'])}，"
                                          f"買進期間 {_d(lastb['start'])} 到 {_d(lastb['end'])}，約等於同期新挖出比特幣的 "
                                          f"{lastb['btc'] / _mined(lastb['start'], lastb['end']) * 100:.0f}%")},
        {"label": "連續每週公告買進", "value": (f"到 {_d(asof)} 為止連續 {streak} 週有買進公告"
                                          + ("，還在進行" if running and streak else "，目前沒有在進行"))},
        {"label": "持幣（依已公布的數字）", "value": f"{held.iloc[-1]:,.0f} 顆"},
    ]
    if len(sales):
        ls = sales.iloc[-1]
        now.append({"label": "最近一次公告賣出", "value": f"{_d(ls['report'])}：{ls['btc']:,.0f} 顆（2020 年以來共 {len(sales)} 次賣出公告）"})
    if len(prem):
        now.append({"label": f"MSTR 溢價（{_d(prem.index[-1])}）", "value": f"市值約是持幣價值的 {prem.iloc[-1]:.2f} 倍"})
    now.append({"label": "近 2 週有公告規則", "value": "持有" if rt["target"].iloc[-1] >= 0.5 else "空手"})
    now.append({"label": "這幾格的意思", "value": "只描述公告與規則目前的狀態，不是買賣建議"})

    # ---------- verdict ----------
    x4 = t1.get("r4")
    xp = t1.get("prem")
    a1 = ev["w1_5"].dropna()
    k2, m50 = rt["k2"][COSTS[0]], rt["ma50"][COSTS[0]]
    ra20 = avg(done, "a20")[0]
    pa20 = avg(st["price"], "a20")[0]
    def clear(x):
        return x and (x["ci"][0] > 0 or x["ci"][1] < 0)

    if x4:
        s1 = (f"{'看得出' if clear(x4) else '看不出明確的'}「漲了才買」：前 4 週漲最多的三分之一週有公告的比例比跌最多的"
              f"{'多' if x4['diff'] >= 0 else '少'} {abs(x4['diff']) * 100:.0f} 點（90% 區間 {x4['ci'][0] * 100:+.0f} 到 "
              f"{x4['ci'][1] * 100:+.0f}）")
    else:
        s1 = "前 4 週漲跌算不出來"
    if xp:
        s1 += (f"，溢價高的週{'多' if xp['diff'] >= 0 else '少'} {abs(xp['diff']) * 100:.0f} 點（{xp['ci'][0] * 100:+.0f} 到 "
               f"{xp['ci'][1] * 100:+.0f}）")
    s1 += "。"
    s2 = (f"{len(ev)} 次公告後 5 天比特幣平均 {_p(a1.mean())}、實際買進那幾天平均 {_p(cv['mean'])}，"
          f"都沒有比沒公告的對照（{_p(pl[(1, 5)]['mean'])}、{_p(cv['placebo'])}）高；連續買 3 週以上停下後 20 天平均 {_p(ra20)}，"
          f"單純連漲結束後 {_p(pa20)}，但只有 {avg(done, 'a20')[2]} 段。")
    s3 = (f"照公告持有（近 2 週）報酬÷波動 {_n(k2['sharpe'])}，50 日均線 {_n(m50['sharpe'])}，"
          f"打亂公告週 {_sh(pl4['ge'])} 一樣好，{'通過' if rt['pass'] else '沒有通過'}事先定的標準。")
    verdict = s1 + s2 + s3

    # ---------- tables ----------
    tables = [
        {"caption": ("檢驗一：是不是漲了才買？以週（週一到週日）為單位，看「這週有沒有買進公告」和公告前（上週日收盤為止）的比特幣漲跌、"
                     "MSTR 溢價的關係。三組是依全部週數排成三等分。「高減低」是最高三分之一的週有公告比例減最低三分之一，"
                     "90% 區間用連續 8 週為一塊重抽 1,000 次。最後兩列改用再早一週的數字（公司真正買進之前），避免把「買進推高價格」算成「漲了才買」。"),
         "columns": ["看什麼", "分組", "週數", "有買進公告的週比例", "2020–2022", "2023–2026", "有公告時平均買多少"],
         "rows": rows1[:16]},
        {"caption": ("檢驗二：公告前後比特幣怎麼走（每格：平均漲跌，括號是上漲次數比例）。第 0 天是公告日（UTC 日線），"
                     "[−5,−1] 是公告前 5 天，[0,+1] 是公告前一天收盤到隔天收盤，[+1,+5] 是公告日收盤之後 5 天，[+5,+20] 是第 5 到第 20 天。"
                     "「段」是相隔超過 20 天才算另一段的獨立次數。買多買少用「買進顆數 ÷ 同期新挖出的比特幣」分三等分。"
                     "最後一列看公司實際買進的那幾天（公告之前）。"),
         "columns": ["分組", "次數／獨立段", "[−5,−1]", "[0,+1]", "[+1,+5]", "[+5,+20]"],
         "rows": rows2[:16]},
        {"caption": ("檢驗三：連續 3 週以上每週都有買進公告的段落。「期間」是第一筆買進期間開始前一天收盤到最後一次公告日收盤；"
                     "「結束後」從第一個沒有公告的週日收盤（那時才知道停了）算起。對照是比特幣週線連續 3 週以上上漲、"
                     "從第一個沒漲的週日收盤算起。"),
         "columns": ["連續買進段（第一次到最後一次公告）", "週數", "這段買了多少", "期間比特幣漲跌", "結束後 20 天", "結束後 60 天"],
         "rows": rows3[:16]},
        {"caption": (f"檢驗四：照公告持有比特幣。公告日的收盤（UTC 0 點）才照做、隔天開始算報酬；{HALF[0]} 起算到 {_d(asof)}。"
                     f"每年報酬分別扣 {COSTS[0] * 100:.2f}%／{COSTS[1] * 100:.2f}% 交易成本；報酬÷波動 = 每日淨報酬平均 ÷ 標準差 × √365（成本 0.10%）；"
                     f"前半 2020-08 到 2023-06、後半 2023-07 到現在。空手沒有利息。通過標準（事先定）：前後半的報酬÷波動都贏 50 日均線，"
                     f"而且打亂公告週的對照一樣好或更好的比例 ≤ 10%。"),
         "columns": ["做法", "每年報酬（0.10%／0.30%）", "報酬÷波動", "前半｜後半", "最深虧損", "持有時間", "隨機對照（報酬÷波動）"],
         "rows": rows4[:16]},
    ]

    n_all = len(buys)
    notes = [
        "事前登記的檢驗（原文，英文）：" + " ".join(line.strip() for line in __doc__.split("Pre-registered tests (verbatim from the owner):")[1]
                                        .split("Details fixed by the author")[0].strip().splitlines()),
        ("事前登記的檢驗（中文）：一、是不是漲了才買——某週有沒有買進公告、買多少，和前 1 到 4 週比特幣漲跌、MSTR 溢價（市值 ÷ 持幣價值）的關係；"
         "二、公告效果——公告前後 [−5,−1]、[0,+1]、[+1,+5]、[+5,+20] 天的比特幣漲跌，依買進規模分三組，和沒有公告的同星期幾日子比；"
         "因為實際買進發生在公告之前的買進期間，也看買進期間本身和同長度沒買的期間比；三、連續買進與結束——連續 3 週以上有公告，"
         "期間漲跌和結束後 20／60 天，和單純連續上漲結束後比，分開「買家停了」和「一般漲多回檔」；四、可交易規則——近 k 週（k=2 事先定，"
         "k=4 只描述）有買進公告才持有，否則空手，和一直抱著、50 日均線、打亂公告週 200 次比，成本 0.10%／0.30%，也當 50 日均線的過濾；"
         "通過 = 前後半（2020-08 到 2023-06、2023-07 到現在）報酬÷波動都贏 50 日均線，且隨機對照一樣好的比例 ≤ 10%。"
         "不確定性用區塊重抽、數獨立事件，分 2020–2022 和 2023–2026 兩段。"),
        (f"資料：DefiLlama 的 Strategy（MSTR）數位資產金庫頁（免登入、免金鑰，每一筆都附上來源 8-K 或公司網頁連結），"
         f"合併同一天公告後共 {n_all} 次買進公告（{_d(buys['report'].iloc[0])} 到 {_d(buys['report'].iloc[-1])}）、{len(sales)} 次賣出公告。"
         "公司自己的 strategy.com 購買紀錄頁對儀表板的固定使用者代理回應「拒絕存取」，SEC EDGAR 的文件庫把它當成「未申報的自動化工具」拒絕，"
         "依規定兩者都不使用、也不繞過。每次執行把抓到的資料存一份，DefiLlama 擋住時用存檔。"),
        "修正的日期（彙整和公司當天的新聞稿不符）：" + ("；".join(fixes) if fixes else "沒有") +
        "。2021-02-19 那筆若照彙整的日期，會比大眾知道早 5 天（偷看未來），所以一定要改；另一筆是彙整寫得太晚。"
        "其他筆另有晚一天的（例如 2020-09-14 的 8-K 被寫成 9-15），晚只會讓結果保守，沒有改。",
        (f"未來資料檢查：公告日早於買進期間結束的筆數 {look['bad']}；把資料截在 2023-06-30 重算持有部位，和用完整資料算到同一天"
         f"{'完全相同' if look['same'] else '不同（請留意）'}；晚一天才照做的結果列在第四張表最後兩列。"),
        (f"買進規模（買進顆數 ÷ 同期新挖出的比特幣）：2020–2022 合計約 {size_per.get(0, float('nan')) * 100:.0f}%、"
         f"2023–2026 合計約 {size_per.get(1, float('nan')) * 100:.0f}%（買進期間長短不一，早期常是幾個月合併公告）。"
         "沒有找到免費、可長期取得的全市場現貨成交量，所以用新挖出的數量當尺度；買得比新挖出多，代表光這一家就吃掉了不只新供給。")
        if size_per else "",
        (f"和現貨 ETF 比（只描述）：2024-01 以後、買進期間 ≤ 14 天的 {etf_cmp['n']} 次，公司買進合計 {_usd(etf_cmp['buy'])}，"
         f"同一段日子美國現貨比特幣 ETF 淨流入合計 {_usd(etf_cmp['net'])}；"
         f"每次買進金額 ÷ 同期 ETF 每日流量絕對值合計，中位數 {etf_cmp['ratio_med']:.2f} 倍；{etf_cmp['bigger']} 次買得比同期 ETF 淨流量（絕對值）多。"
         f"2024-01-11 以來 ETF 淨流入合計 {_usd(etf_cmp['all_net'])}、公司買進合計 {_usd(etf_cmp['all_buy'])}。ETF 出現後，"
         "這家公司只是買盤的一部分，所以 2023–2026 的結果混著 ETF 和其他買家的影響。ETF 流量用儀表板 etf_flows 存下的 Farside 每日數字（唯讀）。")
        if etf_cmp else "ETF 每日流量的存檔這次讀不到，沒有做比較。",
        ("溢價：MSTR 收盤（Yahoo 調整後）× DefiLlama 的完全稀釋股數（2024-08-08 一拆十之前 ×10）÷（當時已公布的持幣 × 比特幣收盤）。"
         "股數更新的時間點不一定是公布當天，可能有幾週的誤差；沒有扣負債和特別股，所以和公司自己公布的 mNAV 不同。"
         "2020–2021 年持幣還少，市值裡有很大一塊是原本的軟體生意，所以那時的「溢價」偏高（5 倍上下），不全是對比特幣的溢價。"
         "2024-08 以後和 DefiLlama 自己算的 mNAV 相差在一成以內；它在拆股前沒有調整股數，所以沒有直接用它的數字。"),
        ("這是一家公司、一段時間：2024-11 以前大多是幾週到幾個月才公告一次，之後幾乎每週一公告，公告頻率的變化本身就和「溢價高、"
         "可以發股票募資」綁在一起，所以檢驗一的「溢價高才買」也可能只是同一段時期的現象，不是每週的反應。"),
        ("連續買進段落很少、彼此相隔不遠，結束後 60 天的視窗互相重疊；連漲對照的段落也重疊。這些平均只能當描述，不能當成穩定的規律。"
         "檢驗二的事件視窗在每週公告的年份會互相重疊（[+5,+20] 會碰到下一次公告），獨立段數遠少於公告次數。"),
        ("觀念來源：van der Beck、Bouchaud 與 Villamaina 的研究認為「被迫或非價格因素的資金流入會推動價格，價格上漲又吸引追價資金」，"
         "形成回饋；這裡用最大的單一企業買家當作一個看得到的資金流，檢查它在比特幣上的痕跡。"),
        "公司在 2026 年有幾次公告賣出比特幣；賣出不算買進公告，規則和連續段落都只看買進。",
        f"隨機對照用固定的亂數種子，資料沒變的話重跑結果一樣。計算時間約 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只整理公開公告和比特幣價格過去的關係、以及目前的狀態，不是買賣建議，也不針對任何人的 MSTR 持股。",
    ]
    notes = [n for n in notes if n] + problems

    sources = [
        {"name": "DefiLlama：Strategy（MSTR）數位資產金庫（每筆買賣附 8-K 連結）", "url": LLAMA_PAGE},
        {"name": OVERRIDES[("2021-02-19", 19452)][3] + "（19,452 顆）", "url": OVERRIDES[("2021-02-19", 19452)][4]},
        {"name": OVERRIDES[("2021-01-28", 29646)][3] + "（29,646 顆）", "url": OVERRIDES[("2021-01-28", 29646)][4]},
        {"name": "SEC EDGAR：Strategy 申報文件列表（各筆 8-K 的原始出處；本研究沒有直接讀取）",
         "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001050446&type=8-K"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        {"name": "比特幣日收盤：幣安現貨日線（2017-08 以前 blockchain.info）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": FARSIDE},
        {"name": "van der Beck, Bouchaud & Villamaina (2024), arXiv:2405.12768", "url": PAPER},
    ]
    return _clean({
        "id": "big_buyer",
        "title": "最大企業買家與比特幣",
        "question": ("Strategy（前 MicroStrategy）2020 年起公開每一次買比特幣：它是不是漲了才買、溢價高才買？公告前後、"
                     "實際買進的那幾天比特幣怎麼走？連續買進停下來之後會不會回吐？照著公告持有比特幣，有沒有贏過 50 日均線和隨機對照？"),
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    })


def _clean(x):
    """No NaN or infinity anywhere; None in shown cells becomes "—"."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return "—" if x is None else x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
