"""Historical study "mstr_premium": MSTR is often held as "leveraged bitcoin". Its premium over the bitcoin it owns
(mNAV) swings from below 1x to above 3x. Practitioner claims: the premium is pro-cyclical and collapses in crashes
(BitMEX https://www.bitmex.com/blog/mstr-mnav-guide ; CoinDesk
https://hotfix.coindesk.com/markets/2024/10/11/microstrategys-nav-premium-hits-highest-since-2021 ; MarketVector
https://redesign.marketvector.com/insights/mvis-insights/did-etf-options-kill-microstrategy); BitMEX argues a low mNAV
can be an entry point. Nobody has tested: does the premium level predict MSTR's return RELATIVE to bitcoin (and
relative to a beta-matched bitcoin position)?

Pre-registered design (verbatim from the owner):

Premium P_t point-in-time (holdings only as announced by t; shares as known by t — if share counts' update dates are
uncertain, lag them 30 days and say so). Sample 2020-09 onward (after the first purchases); note the 2020–21 premium
includes the software business.
1. Forward MSTR-minus-BTC log return over 20/60/120 trading days by premium tercile and by fixed buckets (<1.0,
   1.0–1.5, 1.5–2.5, >2.5); also vs beta-matched BTC (trailing 120-day beta of MSTR on BTC, known at t). Block
   bootstrap (blocks 60 days, 1,000 draws), non-overlapping counts, split 2020-09..2022-12 / 2023-01..2024-12 /
   2025-01..now.
2. Premium change vs BTC: does premium expand when BTC rises (pro-cyclical)? correlation of 20-day premium change with
   20-day BTC return, by period.
3. Rules (act at the close after the premium is known; MSTR trades US hours — use MSTR's own closes; costs 0.10/0.30%):
   S1 hold MSTR when P < its trailing 1-year median, else hold BTC; S2 hold MSTR only when P < 1.5, else BTC; compare
   with always-MSTR, always-BTC, and with ma50 applied to each (ma50 on BTC decides in/out; S1/S2 decide which
   instrument). Placebo: circularly shifted premium series (200 draws).
Pass test (stated now): low-premium minus high-premium forward relative return has the same sign in all three periods
with the full-sample 90% interval excluding zero AND S1 or S2 beats always-MSTR and always-BTC on return÷volatility
AND placebo share ≤ 10%.
Today: current premium, its trailing-year percentile, and S1/S2 states — states only.

Details fixed by the author before looking at any result:
- Data: Strategy's purchase/sale announcements and the fully diluted share count come from the store that the
  big_buyer study saves from DefiLlama's public page (read only, never refetched here; same two date corrections via
  big_buyer._records). MSTR: Yahoo adjusted daily closes (ta._share_close). Bitcoin: the dashboard's daily closes
  (te._daily_close, Binance daily bars dated by their UTC open, so the bar dated t closes at 00:00 UTC of t+1).
- Time grid = MSTR trading days. On trading day t the premium is
  P_t = MSTR close_t × shares known by t ÷ (BTC held per announcements dated ≤ t × bitcoin close dated t−1),
  i.e. bitcoin's 00:00 UTC price of day t, known before the US close (~21:00 UTC). Shares: DefiLlama gives no reliable
  date on which each count became public, so every count is used only 30 calendar days after its date (×10 before the
  10-for-1 split, matching Yahoo's split-adjusted price; DefiLlama's series keeps pre-split counts until it jumps
  tenfold on 2024-09-13, so the ×10 applies up to that jump — the split itself was public on 2024-08-08).
- Forward returns from day t over h trading days: MSTR close_t → close_{t+h}; bitcoin close dated t → dated t+h (the
  first bitcoin close after the decision, about 3 hours after the US close). Relative = log MSTR − log BTC. Beta-matched
  = log MSTR − β_t × log BTC, β_t = OLS slope of daily log MSTR on daily log bitcoin over the 120 trading days ending
  t−1. Primary horizon for the pass test: 60 trading days, plain MSTR-minus-BTC; 20/120 days and beta-matched are
  description.
- Terciles: full sample cut points for the full-sample rows; within-period cut points for the period rows (the premium
  level drifts between eras). "Low minus high" = mean of the bottom tercile minus mean of the top tercile.
  90% interval: moving-block bootstrap of trading days (block 60, 1,000 draws, fixed seed). Non-overlapping count:
  greedy count of days in the group at least h trading days apart.
- Test 2: 20-trading-day log change of P vs 20-day log change of the bitcoin price used in P; Pearson correlation on
  all days (block-bootstrap 90% interval, block 60) and on every 20th day; mean premium change when bitcoin fell more
  than 20% / rose more than 20% in those 20 days.
- Test 3: decision at MSTR close t using P_t; MSTR bought/sold at close_t, bitcoin at the bitcoin close dated t; the
  position earns from t to t+1. S1 median: the 252 trading days ending t (needs a full year, so all rules are compared
  from the first day S1 exists). ma50: bitcoin close dated t−1 above the 50-day average of closes up to t−1; out =
  cash, no interest. Cost per unit traded (a switch MSTR↔BTC trades 2 units). Return÷volatility = mean daily net ÷
  std × √252. Placebo: the premium series (from 2020-09) rolled circularly by a random 63..n−63 trading days, S1 and
  S2 recomputed, 200 draws; share of draws with return÷volatility ≥ the real rule (0.10% cost). For the pass test the
  placebo share of a rule counts only for a rule that beats both always-MSTR and always-BTC. Perturbation: one more day
  of delay. Periods for rules: the same three periods (description).
- Look-ahead check: every input after 2023-06-30 is scrambled (prices, share counts, announcements); premium, beta,
  ma50, S1 and S2 up to that day must be identical.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import big_buyer as bb
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

MSTR_STORE = common.STORE / "mstr_premium_mstr.json"
START = "2020-09-01"
PERIODS = (("2020-09-01", "2022-12-31", "2020-09 到 2022-12"), ("2023-01-01", "2024-12-31", "2023-01 到 2024-12"),
           ("2025-01-01", "2100-01-01", "2025-01 到現在"))
HORIZONS = (20, 60, 120)
PRIMARY = 60
BUCKETS = ((0.0, 1.0, "低於 1.0 倍"), (1.0, 1.5, "1.0 到 1.5 倍"), (1.5, 2.5, "1.5 到 2.5 倍"), (2.5, 1e9, "高於 2.5 倍"))
SHARE_LAG = 30
BETA_DAYS = 120
MED_DAYS = 252
S2_LINE = 1.5
BLOCK, BOOT, PLACEBO = 60, 1000, 200
COSTS = (0.001, 0.003)
YEAR = 252
SEED = 20200901
CUT = "2023-06-30"
SOURCES_EXT = (
    ("BitMEX：MSTR mNAV 指南（溢價會隨行情放大縮小、低 mNAV 可能是進場點的說法）", "https://www.bitmex.com/blog/mstr-mnav-guide"),
    ("CoinDesk 2024-10-11：MicroStrategy 溢價創 2021 年以來新高",
     "https://hotfix.coindesk.com/markets/2024/10/11/microstrategys-nav-premium-hits-highest-since-2021"),
    ("MarketVector：ETF 選擇權是否壓縮了 MicroStrategy 的溢價",
     "https://redesign.marketvector.com/insights/mvis-insights/did-etf-options-kill-microstrategy"),
)

_ok, _n, _d = bb._ok, bb._n, bb._d


def _p(v, digits: int = 1) -> str:
    return bb._p(v, digits)


def _sh(v) -> str:
    return bb._sh(v)


def _lp(v) -> str:
    """A mean log return shown in percent points."""
    return f"{float(v) * 100:+.1f}%" if _ok(v) else "—"


# ---------- data ----------

def _load() -> dict:
    problems: list[str] = []
    if not bb.STORE_FILE.exists():
        raise RuntimeError("big_buyer 研究存下的 DefiLlama 資料不存在，這個研究不自己重抓")
    raw = json.loads(bb.STORE_FILE.read_text(encoding="utf-8"))
    buys, sales, fixes = bb._records(raw["transactions"])
    btc = te._daily_close()
    btc = btc[btc.index >= pd.Timestamp("2019-06-01")]
    try:
        mstr = ta._share_close("MSTR")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            MSTR_STORE.write_text(json.dumps({"fetched": common.today_utc(),
                                              "close": {_d(k): float(v) for k, v in mstr.items()}}), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not MSTR_STORE.exists():
            raise
        d = json.loads(MSTR_STORE.read_text(encoding="utf-8"))
        mstr = pd.Series({pd.Timestamp(k): float(v) for k, v in d["close"].items()}).sort_index()
        problems.append(f"這次 Yahoo 的 MSTR 股價取不到（{type(exc).__name__}），改用 {d.get('fetched')} 存下的資料。")
    shares = pd.Series({pd.Timestamp(r[0], unit="ms").normalize(): float(r[1]) for r in raw.get("shares") or []
                        if r and len(r) > 1 and _ok(r[1])})
    shares = shares[~shares.index.duplicated(keep="last")].sort_index()
    if len(shares) < 100:
        raise RuntimeError(f"股數資料只有 {len(shares)} 筆")
    return {"buys": buys, "sales": sales, "fixes": fixes, "btc": btc, "mstr": mstr, "shares": shares,
            "fetched": raw.get("fetched", "—"), "problems": problems}


def _split_switch(shares: pd.Series) -> pd.Timestamp:
    """First date on which the share series is in post-split units: DefiLlama keeps pre-split counts for a few weeks
    after the 2024-08-08 10-for-1 split, then jumps about tenfold. Falls back to the split day."""
    after = shares[shares.index >= pd.Timestamp(bb.SPLIT_DAY)]
    jump = after[(after / shares.shift(1).reindex(after.index)) > 5]
    return jump.index[0] if len(jump) else pd.Timestamp(bb.SPLIT_DAY)


def _frame(mstr: pd.Series, btc: pd.Series, shares: pd.Series, buys: pd.DataFrame, sales: pd.DataFrame,
           lag: int = SHARE_LAG) -> pd.DataFrame:
    """Everything known at each MSTR close, plus the bitcoin close used for execution / outcomes."""
    m = mstr[(mstr.index >= pd.Timestamp("2020-01-01")) & (mstr.index <= btc.index[-1])].astype(float)
    days = m.index
    s = shares.where(shares.index >= _split_switch(shares), shares * 10.0)
    s.index = s.index + pd.Timedelta(days=lag)
    sh = s.reindex(days.union(s.index)).ffill().reindex(days)
    held = bb._holdings(buys, sales, btc.index).reindex(days)
    b_prev = btc.shift(1).reindex(days)
    b_now = btc.reindex(days)
    value = held * b_prev
    prem = m * sh / value.where(value > 0)
    f = pd.DataFrame({"m": m, "b": b_now, "bp": b_prev, "sh": sh, "held": held, "P": prem})
    lm, lb = np.log(f["m"]).diff(), np.log(f["b"]).diff()
    cov = lm.rolling(BETA_DAYS, min_periods=BETA_DAYS).cov(lb)
    var = lb.rolling(BETA_DAYS, min_periods=BETA_DAYS).var()
    f["beta"] = (cov / var).shift(1)
    ma = btc.rolling(50, min_periods=50).mean()
    f["ma50"] = (btc > ma).astype(float).where(ma.notna()).shift(1).reindex(days)
    # rules (known at t)
    p = f["P"].where(f.index >= pd.Timestamp(START))
    f["med"] = p.rolling(MED_DAYS, min_periods=MED_DAYS).median()
    f["s1"] = (p < f["med"]).astype(float).where(f["med"].notna() & p.notna())
    f["s2"] = (p < S2_LINE).astype(float).where(p.notna())
    return f


# ---------- test 1 ----------

def _forward(f: pd.DataFrame) -> dict:
    out = {}
    lm, lb = np.log(f["m"]), np.log(f["b"])
    for h in HORIZONS:
        fm, fb = lm.shift(-h) - lm, lb.shift(-h) - lb
        out[h] = fm - fb
        out[f"b{h}"] = fm - f["beta"] * fb
    return out


def _nonoverlap(pos: np.ndarray, h: int) -> int:
    n, last = 0, -10 ** 9
    for i in pos:
        if i - last >= h:
            n += 1
            last = i
    return n


def _terciles(x: pd.Series) -> tuple[pd.Series, tuple[float, float]]:
    q1, q2 = np.nanpercentile(x.dropna(), [100 / 3, 200 / 3])
    g = pd.Series(np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), index=x.index).where(x.notna())
    return g, (float(q1), float(q2))


def _boot_diff(y: np.ndarray, g: np.ndarray, rng) -> tuple[float | None, float | None]:
    """90% interval of mean(y | g==0) - mean(y | g==2), moving blocks of BLOCK trading days."""
    n = len(y)
    if n < 2 * BLOCK:
        return None, None
    k = int(math.ceil(n / BLOCK))
    st = rng.integers(0, n - BLOCK + 1, (BOOT, k))
    idx = (st[:, :, None] + np.arange(BLOCK)).reshape(BOOT, -1)[:, :n]
    yy, gg = y[idx], g[idx]
    ok = np.isfinite(yy)
    lo = np.where(ok & (gg == 0), yy, 0.0)
    hi = np.where(ok & (gg == 2), yy, 0.0)
    nlo, nhi = (ok & (gg == 0)).sum(1), (ok & (gg == 2)).sum(1)
    keep = (nlo > 0) & (nhi > 0)
    d = lo.sum(1)[keep] / nlo[keep] - hi.sum(1)[keep] / nhi[keep]
    if len(d) < BOOT // 2:
        return None, None
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


def _group_row(fw: dict, sel: pd.Series) -> dict:
    r = {"n": int(sel.sum())}
    for h in HORIZONS:
        y = fw[h][sel].dropna()
        r[h] = float(y.mean()) if len(y) else None
        r[f"no{h}"] = _nonoverlap(np.flatnonzero(sel.values & fw[h].notna().values), h)
        yb = fw[f"b{h}"][sel].dropna()
        r[f"b{h}"] = float(yb.mean()) if len(yb) else None
        r[f"up{h}"] = float((y > 0).mean()) if len(y) else None
    return r


def _test1(f: pd.DataFrame, fw: dict, rng) -> dict:
    sample = (f.index >= pd.Timestamp(START)) & f["P"].notna().values
    P = f["P"].where(sample)
    g, cuts = _terciles(P)
    res = {"cuts": cuts, "terc": [_group_row(fw, g == k) for k in range(3)], "buckets": []}
    for lo, hi, lab in BUCKETS:
        res["buckets"].append((lab, _group_row(fw, (P >= lo) & (P < hi))))
    sub = np.flatnonzero(sample)
    gv = g.values[sub]
    res["diff"], res["ci"] = {}, {}
    for key in list(HORIZONS) + [f"b{h}" for h in HORIZONS]:
        y = fw[key].values[sub].astype(float)
        res["diff"][key] = float(np.nanmean(y[gv == 0]) - np.nanmean(y[gv == 2]))
        res["ci"][key] = _boot_diff(y, gv, rng)
    res["per"] = []
    for a, b, lab in PERIODS:
        m = sample & (f.index >= pd.Timestamp(a)) & (f.index <= pd.Timestamp(b))
        gp, cp = _terciles(f["P"].where(m))
        rows = [_group_row(fw, gp == k) for k in range(3)]
        idx = np.flatnonzero(m)
        gg = gp.values[idx]
        diff = {}
        for key in list(HORIZONS) + [f"b{h}" for h in HORIZONS]:
            y = fw[key].values[idx].astype(float)
            lo_, hi_ = y[(gg == 0) & np.isfinite(y)], y[(gg == 2) & np.isfinite(y)]
            diff[key] = float(lo_.mean() - hi_.mean()) if len(lo_) and len(hi_) else None
        y = fw[PRIMARY].values[idx].astype(float)
        res["per"].append({"label": lab, "cuts": cp, "rows": rows, "diff": diff, "ci": _boot_diff(y, gg, rng),
                           "days": int(m.sum())})
    signs = [np.sign(x["diff"][PRIMARY]) if _ok(x["diff"][PRIMARY]) else 0 for x in res["per"]]
    ci = res["ci"][PRIMARY]
    res["same_sign"] = bool(all(s == signs[0] and s != 0 for s in signs))
    res["ci_clear"] = bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))
    res["sign_full"] = float(np.sign(res["diff"][PRIMARY]))
    res["pass"] = bool(res["same_sign"] and res["ci_clear"] and signs[0] == res["sign_full"])
    return res


# ---------- test 2 ----------

def _boot_corr(x: np.ndarray, y: np.ndarray, rng) -> tuple[float | None, float | None]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 * BLOCK:
        return None, None
    k = int(math.ceil(n / BLOCK))
    st = rng.integers(0, n - BLOCK + 1, (BOOT, k))
    idx = (st[:, :, None] + np.arange(BLOCK)).reshape(BOOT, -1)[:, :n]
    xx, yy = x[idx], y[idx]
    xx = xx - xx.mean(1, keepdims=True)
    yy = yy - yy.mean(1, keepdims=True)
    c = (xx * yy).sum(1) / np.sqrt((xx ** 2).sum(1) * (yy ** 2).sum(1))
    return float(np.percentile(c, 5)), float(np.percentile(c, 95))


def _test2(f: pd.DataFrame, rng) -> list:
    sample = (f.index >= pd.Timestamp(START)) & f["P"].notna().values
    dp = np.log(f["P"]) - np.log(f["P"].shift(20))
    db = np.log(f["bp"]) - np.log(f["bp"].shift(20))
    lm, lb = np.log(f["m"]).diff(), np.log(f["b"]).diff()
    out = []
    for a, b, lab in ((START, "2100-01-01", "全部（2020-09 起）"),) + PERIODS:
        m = sample & (f.index >= pd.Timestamp(a)) & (f.index <= pd.Timestamp(b)) & dp.notna().values
        x, y = db[m].values, dp[m].values
        every = np.arange(0, len(x), 20)
        c_all = float(np.corrcoef(x, y)[0, 1]) if len(x) > 30 else None
        c_no = float(np.corrcoef(x[every], y[every])[0, 1]) if len(every) > 8 else None
        crash, boom = y[x < math.log(0.8)], y[x > math.log(1.2)]
        mm = m & lm.notna().values & lb.notna().values
        beta = float(np.cov(lm[mm], lb[mm])[0, 1] / np.var(lb[mm], ddof=1)) if mm.sum() > 30 else None
        out.append({"label": lab, "n": int(m.sum()), "n_no": len(every), "c": c_all, "ci": _boot_corr(x, y, rng),
                    "c_no": c_no, "crash": float(np.exp(crash.mean()) - 1) if len(crash) else None,
                    "n_crash": _nonoverlap(np.flatnonzero(x < math.log(0.8)), 20),
                    "boom": float(np.exp(boom.mean()) - 1) if len(boom) else None,
                    "n_boom": _nonoverlap(np.flatnonzero(x > math.log(1.2)), 20), "beta": beta})
    return out


# ---------- test 3 ----------

def _metrics(net: np.ndarray) -> dict:
    eq = np.cumprod(1 + net)
    yrs = len(net) / YEAR
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 and yrs > 0 else None
    sd = net.std(ddof=1)
    return {"cagr": cagr, "sharpe": float(net.mean() / sd * math.sqrt(YEAR)) if sd > 0 else None,
            "deep": float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min())}


class _Book:
    """Daily returns of MSTR and bitcoin on the MSTR trading-day grid inside the evaluation window."""

    def __init__(self, f: pd.DataFrame, begin: pd.Timestamp):
        self.idx = f.index
        self.rm = f["m"].pct_change().fillna(0.0).values
        self.rb = f["b"].pct_change().fillna(0.0).values
        self.keep = self.idx >= begin
        self.k0 = int(np.argmax(self.keep))
        sub = self.idx[self.keep]
        self.periods = [(sub >= pd.Timestamp(a)) & (sub <= pd.Timestamp(b)) for a, b, _ in PERIODS]

    def run(self, wm: np.ndarray, wb: np.ndarray, cost: float, lag: int = 1, full: bool = True) -> dict:
        pm = np.concatenate([np.full(lag, np.nan), wm[:-lag]])[self.k0:]
        pb = np.concatenate([np.full(lag, np.nan), wb[:-lag]])[self.k0:]
        pm, pb = np.nan_to_num(pm, nan=0.0), np.nan_to_num(pb, nan=0.0)
        rm, rb = self.rm[self.k0:], self.rb[self.k0:]
        trade = np.abs(np.diff(np.concatenate([[0.0], pm]))) + np.abs(np.diff(np.concatenate([[0.0], pb])))
        net = pm * rm + pb * rb - cost * trade
        out = _metrics(net)
        if full:
            out["per"] = [_metrics(net[p])["sharpe"] if p.sum() > 30 else None for p in self.periods]
            out["mstr_time"] = float(pm.mean())
            out["out_time"] = float(1 - (pm + pb).mean())
            out["switch"] = float((trade > 0.5).sum() / (len(net) / YEAR))
        return out


def _weights(f: pd.DataFrame, key: str, s1: pd.Series | None = None, s2: pd.Series | None = None):
    one = np.ones(len(f))
    ma = f["ma50"].fillna(0.0).values
    pick = {"btc": np.zeros(len(f)), "mstr": one,
            "s1": (f["s1"] if s1 is None else s1).ffill().fillna(0.0).values,
            "s2": (f["s2"] if s2 is None else s2).ffill().fillna(0.0).values}
    base, filt = key.split("_")[0], key.endswith("_ma")
    wm = pick[base]
    gate = ma if filt else one
    return wm * gate, (1 - wm) * gate


def _test3(f: pd.DataFrame, rng) -> dict:
    begin = f.index[int(np.argmax(f["s1"].notna().values))]
    book = _Book(f, begin)
    keys = ("btc", "mstr", "s1", "s2", "btc_ma", "mstr_ma", "s1_ma", "s2_ma")
    res = {"begin": begin}
    for k in keys:
        wm, wb = _weights(f, k)
        res[k] = {c: book.run(wm, wb, c) for c in COSTS}
        res[k + "_lag2"] = book.run(wm, wb, COSTS[0], lag=2)
    # placebo: circularly shifted premium
    sample = (f.index >= pd.Timestamp(START)) & f["P"].notna().values
    pos = np.flatnonzero(sample)
    pv = f["P"].values[pos]
    n = len(pv)
    sims = {k: [] for k in ("s1", "s2", "s1_ma", "s2_ma")}
    for _ in range(PLACEBO):
        sh = int(rng.integers(63, n - 63))
        p = pd.Series(np.nan, index=f.index)
        p.iloc[pos] = np.roll(pv, sh)
        med = p.rolling(MED_DAYS, min_periods=MED_DAYS).median()
        s1 = (p < med).astype(float).where(med.notna() & p.notna())
        s2 = (p < S2_LINE).astype(float).where(p.notna())
        for k in sims:
            wm, wb = _weights(f, k, s1, s2)
            r = book.run(wm, wb, COSTS[0], full=False)["sharpe"]
            sims[k].append(r if _ok(r) else -9.0)
    res["placebo"] = {}
    for k, v in sims.items():
        v = np.array(v)
        real = res[k][COSTS[0]]["sharpe"]
        res["placebo"][k] = {"median": float(np.median(v)), "ge": float((v >= real).mean()) if _ok(real) else None}
    sm, sb = res["mstr"][COSTS[0]]["sharpe"], res["btc"][COSTS[0]]["sharpe"]
    res["beats"] = {k: bool(_ok(res[k][COSTS[0]]["sharpe"]) and res[k][COSTS[0]]["sharpe"] > sm
                           and res[k][COSTS[0]]["sharpe"] > sb) for k in ("s1", "s2")}
    res["pass_rule"] = bool(any(res["beats"][k] for k in ("s1", "s2")))
    res["pass_placebo"] = bool(any(res["beats"][k] and _ok(res["placebo"][k]["ge"]) and res["placebo"][k]["ge"] <= 0.10
                                   for k in ("s1", "s2")))
    return res


# ---------- look-ahead check ----------

def _lookahead(data: dict) -> bool:
    cut = pd.Timestamp(CUT)
    rng = np.random.default_rng(7)
    f0 = _frame(data["mstr"], data["btc"], data["shares"], data["buys"], data["sales"])

    def scramble(s: pd.Series) -> pd.Series:
        s = s.copy()
        late = s.index > cut
        s[late] = s[late].values * rng.uniform(0.3, 3.0, late.sum())
        return s

    buys = data["buys"].copy()
    late = buys["report"] > cut
    buys.loc[late, "btc"] = buys.loc[late, "btc"].values * rng.uniform(0.1, 5.0, late.sum())
    sales = data["sales"].copy()
    sales.loc[sales["report"] > cut, "btc"] = 99999.0
    f1 = _frame(scramble(data["mstr"]), scramble(data["btc"]), scramble(data["shares"]), buys, sales)
    cols = ["P", "beta", "ma50", "med", "s1", "s2"]
    a, b = f0.loc[f0.index <= cut, cols], f1.loc[f1.index <= cut, cols]
    return bool(a.shape == b.shape and np.allclose(a.fillna(-1).values, b.fillna(-1).values, rtol=1e-12, atol=1e-12))


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    data = _load()
    problems = list(data["problems"])
    f = _frame(data["mstr"], data["btc"], data["shares"], data["buys"], data["sales"])
    f = f[f["b"].notna() & f["bp"].notna()]
    asof = f.index[-1]
    fw = _forward(f)
    t1 = _test1(f, fw, rng)
    t2 = _test2(f, rng)
    t3 = _test3(f, rng)
    look = _lookahead(data)
    f0 = _frame(data["mstr"], data["btc"], data["shares"], data["buys"], data["sales"], lag=0)
    f0 = f0[f0["b"].notna() & f0["bp"].notna()]
    t3_lag0 = {}
    book0 = _Book(f0, t3["begin"])
    for k in ("s1", "s2"):
        wm, wb = _weights(f0, k)
        t3_lag0[k] = book0.run(wm, wb, COSTS[0], full=False)["sharpe"]
    d0_ = _test1(f0, _forward(f0), np.random.default_rng(SEED))["diff"][PRIMARY]

    # ---------- table 1 ----------
    q1, q2 = t1["cuts"]

    def row1(label, rng_lab, r):
        return [label, rng_lab, f"{r['n']:,} 天／{r[f'no{PRIMARY}']} 段", _lp(r[20]),
                f"{_lp(r[60])}（MSTR 贏 {_sh(r['up60'])}）", _lp(r[120]), _lp(r["b60"])]

    rows1 = [row1("溢價低（最低三分之一）", f"≤ {q1:.2f} 倍", t1["terc"][0]),
             row1("溢價中", f"{q1:.2f} 到 {q2:.2f} 倍", t1["terc"][1]),
             row1("溢價高（最高三分之一）", f"> {q2:.2f} 倍", t1["terc"][2])]
    for lab, r in t1["buckets"]:
        rows1.append(row1("固定分組", lab, r) if r["n"] else ["固定分組", lab, "0 天", "—", "—", "—", "—"])

    def cis(key):
        lo, hi = t1["ci"][key]
        return f"{_lp(t1['diff'][key])}（{_lp(lo)} 到 {_lp(hi)}）"

    rows1.append(["低減高（90% 區間）", "—", "—", cis(20), cis(60), cis(120), cis("b60")])

    # ---------- table 2 ----------
    rows2 = []
    for x in t1["per"]:
        c1, c2 = x["cuts"]
        labs = (f"≤ {c1:.2f} 倍", f"{c1:.2f} 到 {c2:.2f} 倍", f"> {c2:.2f} 倍")
        for k, nm in ((0, "溢價低"), (2, "溢價高")):
            r = x["rows"][k]
            rows2.append([x["label"] if k == 0 else "", nm, labs[k], f"{r['n']:,} 天／{r[f'no{PRIMARY}']} 段",
                          _lp(r[20]), _lp(r[60]), _lp(r[120]), _lp(r["b60"])])
        lo, hi = x["ci"]
        rows2.append(["", "低減高", f"60 天的 90% 區間 {_lp(lo)} 到 {_lp(hi)}", f"{x['days']:,} 天",
                      _lp(x["diff"][20]), _lp(x["diff"][60]), _lp(x["diff"][120]), _lp(x["diff"]["b60"])])

    # ---------- table 3 ----------
    rows3 = []
    for x in t2:
        lo, hi = x["ci"]
        rows3.append([x["label"], f"{x['n']:,} 天／{x['n_no']} 點", f"{_n(x['c'])}（{_n(lo)} 到 {_n(hi)}）", _n(x["c_no"]),
                      f"{_p(x['crash'])}（{x['n_crash']} 段）" if _ok(x["crash"]) else f"—（{x['n_crash']} 段）",
                      f"{_p(x['boom'])}（{x['n_boom']} 段）" if _ok(x["boom"]) else f"—（{x['n_boom']} 段）",
                      _n(x["beta"])])

    # ---------- table 4 ----------
    names = {"btc": "一直拿比特幣", "mstr": "一直拿 MSTR", "s1": "S1：溢價低於近一年中位數拿 MSTR，否則比特幣",
             "s2": f"S2：溢價低於 {S2_LINE} 倍才拿 MSTR，否則比特幣", "btc_ma": "比特幣＋50 日均線",
             "mstr_ma": "MSTR＋50 日均線（看比特幣的均線）", "s1_ma": "S1＋50 日均線", "s2_ma": "S2＋50 日均線"}
    rows4 = []
    for k in ("btc", "mstr", "s1", "s2", "btc_ma", "mstr_ma", "s1_ma", "s2_ma"):
        a, b = t3[k][COSTS[0]], t3[k][COSTS[1]]
        pl = t3["placebo"].get(k)
        plc = f"中位 {_n(pl['median'])}；{_sh(pl['ge'])} 一樣好或更好" if pl else "—"
        rows4.append([names[k], f"{_p(a['cagr'])}／{_p(b['cagr'])}", _n(a["sharpe"]),
                      "｜".join(_n(v) for v in a["per"]), _p(a["deep"], 0),
                      f"{_sh(a['mstr_time'])}／{_sh(a['out_time'])}", plc])
    for k in ("s1", "s2", "s1_ma", "s2_ma"):
        a = t3[k + "_lag2"]
        rows4.append([names[k].split("：")[0] + "：晚一天才照做", f"{_p(a['cagr'])}／—", _n(a["sharpe"]),
                      "｜".join(_n(v) for v in a["per"]), _p(a["deep"], 0),
                      f"{_sh(a['mstr_time'])}／{_sh(a['out_time'])}", "—"])

    # ---------- now ----------
    P = f["P"]
    last = P.iloc[-1]
    year = P.iloc[-MED_DAYS:]
    pct = float((year < last).mean() + 0.5 * (year == last).mean())
    now = [
        {"label": f"MSTR 溢價（{_d(asof)} 收盤）", "value": f"股價總值約是它持有的比特幣總值的 {last:.2f} 倍"
                                                      f"（持幣 {f['held'].iloc[-1]:,.0f} 顆、股數用 30 天前的數字）"},
        {"label": "在近一年裡的位置", "value": f"比近 252 個交易日裡 {pct * 100:.0f}% 的日子低或一樣（近一年中位數 {f['med'].iloc[-1]:.2f} 倍）"},
        {"label": "S1 狀態（低於近一年中位數）", "value": "MSTR" if f["s1"].iloc[-1] >= 0.5 else "比特幣"},
        {"label": f"S2 狀態（低於 {S2_LINE} 倍）", "value": "MSTR" if f["s2"].iloc[-1] >= 0.5 else "比特幣"},
        {"label": "比特幣 50 日均線", "value": "在均線上（持有）" if f["ma50"].iloc[-1] >= 0.5 else "在均線下（空手）"},
        {"label": "這幾格的意思", "value": "只描述溢價和規則目前的狀態，不是買賣建議"},
    ]

    # ---------- verdict ----------
    d60, ci60 = t1["diff"][PRIMARY], t1["ci"][PRIMARY]
    lo_r, hi_r = t1["terc"][0][60], t1["terc"][2][60]
    per_s = "、".join(_lp(x["diff"][PRIMARY]) for x in t1["per"])
    s1 = (f"溢價（股價總值÷持有比特幣總值）最低三分之一的日子之後 60 個交易日，MSTR 比比特幣平均{'多' if lo_r >= 0 else '少'} "
          f"{abs(lo_r) * 100:.0f}%，最高三分之一{'多' if hi_r >= 0 else '少'} {abs(hi_r) * 100:.0f}%，低減高 {_lp(d60)}"
          f"（90% 區間 {_lp(ci60[0])} 到 {_lp(ci60[1])}），三段時期分別 {per_s}{'，方向一致' if t1['same_sign'] else '，方向不一致'}。")
    full2 = t2[0]
    lo2, hi2 = full2["ci"]
    if _ok(lo2) and lo2 > 0:
        head = "全期看得出溢價漲時變大、跌時壓扁"
    elif _ok(hi2) and hi2 < 0:
        head = "全期反而是比特幣漲時溢價變小"
    else:
        head = "全期看不出溢價明確地漲時變大、跌時壓扁"
    s2 = (f"{head}（20 天比特幣漲跌和溢價變化的相關 {_n(full2['c'])}，90% 區間 {_n(lo2)} 到 {_n(hi2)}；三段分別 "
          + "、".join(_n(x["c"]) for x in t2[1:])
          + f"），比特幣 20 天跌超過兩成時溢價平均 {_p(full2['crash'], 0)}、漲超過兩成時 {_p(full2['boom'], 0)}。")
    passed = bool(t1["pass"] and t3["pass_rule"] and t3["pass_placebo"])
    g = lambda k: _n(t3[k][COSTS[0]]["sharpe"])  # noqa: E731
    s3 = (f"依溢價切換的報酬÷波動 S1 {g('s1')}、S2 {g('s2')}，一直拿 MSTR {g('mstr')}、一直拿比特幣 {g('btc')}，"
          f"隨機平移溢價後一樣好的比例 {_sh(t3['placebo']['s1']['ge'])}、{_sh(t3['placebo']['s2']['ge'])}，"
          f"{'通過' if passed else '沒有通過'}事先定的標準。")
    verdict = s1 + s2 + s3

    tables = [
        {"caption": (f"檢驗一：溢價高低和之後 MSTR 相對比特幣的表現（{START[:7]} 到 {_d(asof)}，每個美股交易日一筆）。"
                     "每格是「MSTR 對數報酬 − 比特幣對數報酬」的平均，正的代表 MSTR 贏比特幣；"
                     "「對同β比特幣」是 MSTR 減去「β 倍的比特幣」（β 用前 120 個交易日 MSTR 對比特幣的日報酬斜率），"
                     "等於和一個同樣槓桿的比特幣部位比。三等分用全期的分界。「段」是相隔至少 60 個交易日的獨立次數。"
                     "90% 區間用連續 60 個交易日為一塊重抽 1,000 次；120 天的視窗比區塊長，區間會偏窄。"),
         "columns": ["分組", "溢價範圍", "天數／60 天不重疊段", "之後 20 天 MSTR−比特幣", "之後 60 天（MSTR 贏的比例）",
                     "之後 120 天", "60 天，對同β比特幣"],
         "rows": rows1[:16]},
        {"caption": ("檢驗一（分時期）：每段時期自己分三等分（各時期的溢價水準差很多），只列最低和最高三分之一與兩者相減。"
                     "事先定的通過條件之一：60 天的「低減高」三段方向相同，而且全期 90% 區間不含 0。"),
         "columns": ["時期", "分組", "溢價範圍", "天數／60 天不重疊段", "之後 20 天", "之後 60 天", "之後 120 天", "60 天，對同β比特幣"],
         "rows": rows2[:16]},
        {"caption": ("檢驗二：溢價是不是漲時變大、跌時壓扁？比較 20 個交易日的溢價變化和同期比特幣漲跌（都用對數）。"
                     "相關係數 > 0 代表比特幣漲溢價就變大。「每天」用所有重疊的 20 天，括號是區塊重抽 90% 區間；"
                     "「每 20 天一點」不重疊。最後一欄是 MSTR 日報酬對比特幣的 β：β 大於 1 時，比特幣漲溢價自然會變大，兩件事是同一回事。"),
         "columns": ["時期", "天數／不重疊點", "相關（每天，90% 區間）", "相關（每 20 天一點）", "比特幣 20 天跌超過 20% 時溢價變化",
                     "比特幣 20 天漲超過 20% 時溢價變化", "MSTR 對比特幣 β"],
         "rows": rows3},
        {"caption": (f"檢驗三：依溢價在 MSTR 和比特幣之間切換。每天美股收盤時用當天溢價決定，MSTR 在收盤價成交、比特幣在之後的 UTC 0 點成交，"
                     f"隔天開始算報酬；{_d(t3['begin'])}（S1 需要一整年溢價才算得出來）起算到 {_d(asof)}，以交易日計。"
                     f"每年報酬分別扣 {COSTS[0] * 100:.2f}%／{COSTS[1] * 100:.2f}% 成本（換一次要賣一個買一個）；報酬÷波動 = 每日淨報酬平均 ÷ 標準差 × √252"
                     "（成本 0.10%）。三段 = 2020-09 到 2022-12｜2023 到 2024｜2025 起。50 日均線看比特幣、在均線下全部空手、空手沒有利息。"
                     f"隨機對照：把溢價序列整段循環平移 {PLACEBO} 次後重算 S1／S2。"),
         "columns": ["做法", "每年報酬（0.10%／0.30%）", "報酬÷波動", "三段報酬÷波動", "最深虧損", "拿 MSTR／空手的時間",
                     "隨機對照（報酬÷波動）"],
         "rows": rows4[:16]},
    ]

    first_buy = data["buys"]["report"].iloc[0]
    ps = f["P"][f.index >= pd.Timestamp(START)].dropna()
    yearly = {y: (float(v.median()), float(v.min()), float(v.max())) for y, v in ps.groupby(ps.index.year)}
    notes = [
        "事前登記的設計（原文，英文）：" + " ".join(line.strip() for line in __doc__.split("Pre-registered design (verbatim from the owner):")[1]
                                        .split("Details fixed by the author")[0].strip().splitlines()),
        ("事前登記的設計（中文）：溢價 P 只用當時已知的資料（持幣只算已公告的、股數只用當時已知的；股數公布日期不確定，所以全部延後 30 天用）。"
         "從 2020-09 起（第一次買進之後），2020–21 年的溢價包含原本的軟體生意。一、依溢價三等分和固定分組（< 1.0、1.0–1.5、1.5–2.5、> 2.5 倍），"
         "看之後 20／60／120 個交易日 MSTR 減比特幣的對數報酬，也和同β的比特幣部位比；區塊重抽（60 天一塊、1,000 次）、數不重疊段，"
         "分 2020-09 到 2022-12、2023–2024、2025 起三段。二、溢價是不是順著行情（20 天溢價變化和 20 天比特幣漲跌的相關，分時期）。"
         "三、規則：S1 溢價低於近一年中位數拿 MSTR、否則比特幣；S2 溢價低於 1.5 倍拿 MSTR、否則比特幣；和一直拿 MSTR、一直拿比特幣比，"
         "也都加上比特幣 50 日均線（均線決定進出，S1／S2 決定拿哪個）；成本 0.10%／0.30%；隨機對照把溢價序列循環平移 200 次。"
         "通過 = 低溢價減高溢價的之後相對報酬三段方向相同、全期 90% 區間不含 0，而且 S1 或 S2 的報酬÷波動贏過一直拿 MSTR 和一直拿比特幣，"
         "而且隨機對照一樣好的比例 ≤ 10%。作者在看結果前另外定的細節：主要看 60 天、不調β；隨機對照的比例只算贏過兩者的那條規則。"),
        (f"結果：檢驗一 {'通過' if t1['pass'] else '沒有通過'}（三段方向{'相同' if t1['same_sign'] else '不同'}、全期區間"
         f"{'不含 0' if t1['ci_clear'] else '包含 0'}）；檢驗三規則 {'有' if t3['pass_rule'] else '沒有'}贏過兩者"
         f"（S1 {'贏' if t3['beats']['s1'] else '沒贏'}、S2 {'贏' if t3['beats']['s2'] else '沒贏'}）；"
         f"隨機對照條件 {'符合' if t3['pass_placebo'] else '不符合'}。整體{'通過' if passed else '沒有通過'}。"),
        (f"資料：買進／賣出公告和完全稀釋股數直接讀 big_buyer 研究存下的 DefiLlama Strategy 頁面資料（{data['fetched']} 抓的，"
         f"這裡不重抓；同樣套用那兩筆日期修正：{'；'.join(data['fixes']) if data['fixes'] else '無'}），第一次買進公告 {_d(first_buy)}。"
         "MSTR 用 Yahoo 調整後日收盤；比特幣用儀表板的日線（幣安，標日期 t 的 K 棒在 t+1 的 UTC 0 點收盤）。"),
        ("時間點：交易日 t 的溢價 = MSTR 當天收盤 × 股數 ÷（當天以前已公告的持幣 × 當天 UTC 0 點的比特幣價格），"
         "美股約 UTC 21 點收盤，那時這些都已知道。之後報酬：MSTR 從 t 收盤算起，比特幣從決定後的第一個 UTC 0 點收盤算起（晚約 3 小時）。"
         "β 只用到前一天為止的 120 個交易日。50 日均線用前一天的比特幣收盤。"),
        (f"股數延後 30 天：DefiLlama 的股數是每日序列，但看不出每個數字是哪一天公開的（季報、增發公告時間不一），所以一律晚 30 天才用。"
         f"公司 2024 年以後大量增發，晚 30 天會讓股數偏少、溢價略為偏低。2024-08-08 一拆十：DefiLlama 的股數到 "
         f"{_d(_split_switch(data['shares']))} 才換成拆股後的單位，之前的數字一律 ×10（拆股當天就已公開，只是換單位）。不延後時：60 天低減高 {_lp(d0_)}，"
         f"S1／S2 報酬÷波動 {_n(t3_lag0['s1'])}／{_n(t3_lag0['s2'])}（只供對照）。"),
        ("溢價的定義：這裡是「股價總值 ÷ 持有比特幣總值」，沒有加上負債和特別股，所以和公司自己公布的 mNAV（用企業價值）不同；"
         "公司負債和特別股越多，這裡的數字越低。2020–2021 年持幣還少，股價總值裡有一大塊是原本的軟體生意，所以那時的溢價偏高、"
         "不全是對比特幣的溢價，第一段時期的結果要打折看。"),
        ("機械關係：溢價 = MSTR 價格 × 股數 ÷（持幣 × 比特幣價格）。如果 MSTR 平常就以 β 倍的幅度跟著比特幣動，溢價的變化就約是 (β − 1) × 比特幣漲跌，"
         "所以「溢價順著行情」和「MSTR 比比特幣更會漲跌（β > 1）」是同一件事的兩種說法。另外每次公告買進（尤其是發股票換比特幣），"
         "持幣增加會讓溢價往 1 靠近。"),
        (f"未來資料檢查：把 {CUT} 之後的 MSTR 股價、比特幣價格、股數和公告全部打亂重算，到 {CUT} 為止的溢價、β、50 日均線、"
         f"S1、S2 {'完全相同' if look else '不同（請留意）'}；晚一天才照做的結果列在第四張表最後四列。"),
        ("限制：只有一家公司、約六年；溢價在不同年份差很多（每年中位數，最低到最高："
         + "；".join(f"{y} 年 {v[0]:.2f}（{v[1]:.2f} 到 {v[2]:.2f}）" for y, v in yearly.items())
         + "），三等分其實很大一部分是在分「哪一段時期」。60 天和 120 天的視窗彼此重疊，獨立段數遠少於天數；"
         "120 天的區間偏窄。公司的融資方式（可轉債、增發、特別股）一直在變，過去的關係不一定延續。"),
        f"隨機對照與區塊重抽用固定的亂數種子，資料沒變的話重跑結果一樣。計算時間約 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只整理公開資料過去的關係和目前的狀態，不是買賣建議，也不針對任何人的持股。",
    ] + problems

    sources = [{"name": n, "url": u} for n, u in SOURCES_EXT] + [
        {"name": "DefiLlama：Strategy（MSTR）數位資產金庫（買賣公告、完全稀釋股數；經 big_buyer 研究的存檔）", "url": bb.LLAMA_PAGE},
        {"name": bb.OVERRIDES[("2021-02-19", 19452)][3] + "（日期修正）", "url": bb.OVERRIDES[("2021-02-19", 19452)][4]},
        {"name": bb.OVERRIDES[("2021-01-28", 29646)][3] + "（日期修正）", "url": bb.OVERRIDES[("2021-01-28", 29646)][4]},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://query1.finance.yahoo.com/v8/finance/chart/MSTR"},
        {"name": "比特幣日收盤：幣安現貨日線", "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"},
    ]
    return bb._clean({
        "id": "mstr_premium",
        "title": "MSTR 溢價與相對比特幣的表現",
        "question": ("MSTR 常被當成「有槓桿的比特幣」。它的股價總值除以它持有的比特幣總值（溢價）在 1 倍以下到 3 倍以上之間擺動。"
                     "溢價低的時候，之後 MSTR 會不會比比特幣（和同樣槓桿的比特幣）表現好？溢價是不是漲時變大、跌時壓扁？"
                     "依溢價在 MSTR 和比特幣之間切換，有沒有贏過一直拿其中一個和隨機對照？"),
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    })


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
