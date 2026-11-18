"""Historical study "etf_flows": where ETFs are a big share of investment demand (gold, via SPDR Gold Shares, GLD),
how do ETF inflows / outflows relate to later prices? Fix one rule on gold only, then test it unchanged on the US spot
bitcoin and ether ETFs.

Pre-registered design (verbatim from the owner; decided on GOLD only, then not changed):

    Flow signal: 5-day and 20-day summed flow as % of holdings; "extreme" = top/bottom 10% of its own trailing 1-year
    (rolling, point-in-time) distribution.
    Hypotheses, each tested on GLD 2005–2026:
    - H1 reversal: after extreme inflow, next 20 trading days return below normal; after extreme outflow, above normal.
    - H2 continuation: when 20-day flow > 0, next 20 days above normal.
    - H3 lead: regress next 5- and 20-day returns on flow, controlling for the asset's own past 5/20/60-day returns
      (does flow add anything beyond price momentum?). Also the reverse: does past return predict flow (to confirm
      "flows follow price")?
    Use non-overlapping or Newey-West/block-bootstrap uncertainty; report independent-episode counts; split
    2005–2012 / 2013–2019 / 2020–2026.
    Selection rule (stated now): a hypothesis "holds on gold" only if same sign in all three sub-periods AND
    block-bootstrap 90% interval excludes zero. Pick at most one rule (the one meeting this; if none, say none and
    still run the BTC check of H1 as the literature's claim).
    Trading rule from the chosen hypothesis (e.g. H1: hold normally, step to cash for 20 days after extreme inflow; or
    as a filter on ma50), costs 0.10%/0.30%, vs holding and vs ma50, and a placebo with the flow series circularly
    shifted (200 draws).
    Then apply the SAME rule, thresholds defined the same relative way, to BTC and ETH ETF flows (and BTC price),
    report: sign match, effect size, number of independent extreme episodes, ma50 overlay with main_signal_check's
    grid where data length allows (it won't fully — say so). Be explicit that the BTC sample is < 2 years.
    Today: the current flow percentile for GLD/BTC/ETH ETFs and what the rule state is — states only.

Details fixed by the author before any result was computed (none of them were changed afterwards):
- Percentile of a flow = mid-rank of today's value inside the last 252 flow days including today (ties count half),
  needs at least 120 values; extreme inflow = percentile > 90%, extreme outflow = percentile <= 10%.
- Gold flow = daily change of tonnes in the trust; 5/20-day sum divided by tonnes 5/20 days earlier. Spot crypto ETFs
  publish no holdings history in Farside, so their 5/20-day summed USD flow is divided by the trailing 252-day
  (at least 60 days) average absolute daily flow; the same alternative is computed for gold in USD as a check.
  Percentiles are rank based, so the divisor hardly matters.
- Timing: flows dated day t are published after the US close of t (Farside often the next morning Asia time). The
  earliest action is the next close: GLD's next trading-day close; for crypto the daily candle dated t+1 (closes at
  00:00 UTC of t+2). Returns are measured from that close. Perturbation: one more day of delay.
- H1 statistic: mean next-20-day return after extreme outflow minus after extreme inflow (positive = reversal), once
  with the 5-day signal (H1a) and once with the 20-day signal (H1b). H2: mean next-20-day return when the 20-day flow
  is > 0 minus when it is <= 0. H3a/H3b: OLS coefficient of the next 5/20-day return on (flow percentile - 0.5),
  controlling for past 5/20/60-day returns. Reverse: coefficient of the next 20 days' flow percentile on the past
  20-day return, controlling for the current 20-day flow percentile.
- Uncertainty: moving-block bootstrap, block 60 flow days, 1,000 draws, fixed seed. If several hypotheses hold, the
  one with the largest |effect| / bootstrap standard deviation is chosen. H3 maps to the H1-type rule if its
  coefficient is negative and to the H2-type rule if positive.
- H1-type rule: hold; after an extreme inflow, cash for the next 20 flow days. H2-type: hold only while the summed
  flow is > 0. Overlay: the 50-day average rule (close above its 50-day mean, as main_signal_check) AND the rule.
"""
from __future__ import annotations

import io
import json
import math
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import flows2
from dashboard.studies2 import main_signal_check as mc
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

GLD_PAGE = "https://www.spdrgoldshares.com/usa/gld/"
GLD_XLSX = "https://api.spdrgoldshares.com/api/v1/historical-archive?product=gld&exchange=NYSE&lang=en"
GLD_STORE = common.STORE / "etf_flows_gld.json"
WIN, MINP, ABS_MINP = 252, 120, 60
HI, LO = 0.90, 0.10
H = 20
BLOCK, BOOT, SEED = 60, 1000, 20240111
PLACEBO = 200
COSTS = (0.001, 0.003)
PERIODS = (("2005–2012", "2005-01-01", "2012-12-31"), ("2013–2019", "2013-01-01", "2019-12-31"),
           ("2020–2026", "2020-01-01", "2026-12-31"))
HYP = ("H1a", "H1b", "H2", "H3a", "H3b")
HNAME = {"H1a": "H1 反向（5 日流量極端）", "H1b": "H1 反向（20 日流量極端）", "H2": "H2 順向（20 日淨流入 > 0）",
         "H3a": "H3 流量領先（扣掉價格動能，之後 5 天）", "H3b": "H3 流量領先（扣掉價格動能，之後 20 天）",
         "REV": "反過來：過去 20 天漲跌 → 之後 20 天流量"}
CN = {"GLD": "黃金 GLD", "BTC": "比特幣 ETF", "ETH": "以太幣 ETF"}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

DESIGN_EN = __doc__.split("Pre-registered design (verbatim from the owner; decided on GOLD only, then not changed):")[1] \
    .split("Details fixed by the author")[0]
DESIGN_EN = " ".join(line.strip() for line in DESIGN_EN.strip().splitlines())


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pp(v, digits: int = 1) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{digits}f} 點"


def _pc(v, digits: int = 1) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{digits}f}%"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _xlsx_rows(content: bytes) -> list[dict]:
    """Rows of the archive's data sheet as {column letter: text}; read with zipfile + XML (no openpyxl)."""
    z = zipfile.ZipFile(io.BytesIO(content))
    shared = [''.join(t.text or '' for t in si.iter(NS + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
    sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
    sheet = max(sheets, key=lambda n: z.getinfo(n).file_size)  # the disclaimer sheet is tiny
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(NS + "row"):
        out = {}
        for c in row.findall(NS + "c"):
            v = c.find(NS + "v")
            if v is None:
                continue
            col = ''.join(ch for ch in c.get("r", "") if ch.isalpha())
            out[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(out)
    return rows


def _gld_fetch() -> pd.DataFrame:
    reply = requests.get(GLD_XLSX, headers=common.HEADERS, timeout=60)
    time.sleep(common.PAUSE)
    if reply.status_code in (401, 403, 429, 503):
        raise RuntimeError(f"SPDR 回應 HTTP {reply.status_code}；依規定不繞過")
    reply.raise_for_status()
    if not reply.content.startswith(b"PK"):
        raise RuntimeError("SPDR 回傳的不是 XLSX（可能要登入或被檢查頁取代）")
    rows = _xlsx_rows(reply.content)
    head = {v.strip(): k for k, v in rows[0].items()}
    want = {"price": "Closing Price", "ozps": "Ounces of Gold per Share", "nav": "NAV/Share at 10:30am NYT",
            "oz": "Total Ounces of Gold in the Trust", "tonnes": "Tonnes of Gold"}
    cols = {}
    for key, name in want.items():
        hit = [k for h, k in head.items() if h.lower().startswith(name.lower())]
        if not hit:
            raise RuntimeError(f"SPDR 檔案找不到「{name}」欄，格式改了")
        cols[key] = hit[0]
    data = {}
    for r in rows[1:]:
        try:
            day = datetime.strptime(str(r.get(head.get("Date", "A"), "")).strip(), "%d-%b-%Y").strftime("%Y-%m-%d")
            vals = [float(r[cols[k]]) for k in ("price", "tonnes", "oz", "nav", "ozps")]
        except (KeyError, ValueError, TypeError):
            continue  # holidays ("US Holiday") and blank rows
        if all(math.isfinite(x) and x > 0 for x in vals):
            data[day] = [vals[0], vals[1], vals[2], vals[3] / vals[4]]
    if len(data) < 4000:
        raise RuntimeError(f"SPDR 檔案只讀到 {len(data)} 個交易日")
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        GLD_STORE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store is only a fallback
        pass
    return data


def _gld() -> tuple[pd.DataFrame, str]:
    problem = ""
    try:
        data = _gld_fetch()
    except Exception as exc:  # noqa: BLE001
        if not GLD_STORE.exists():
            raise
        data = json.loads(GLD_STORE.read_text(encoding="utf-8"))
        problem = f"這次下載 SPDR 檔案失敗（{type(exc).__name__}），改用上次存下的資料。"
    f = pd.DataFrame.from_dict(data, orient="index", columns=["price", "tonnes", "oz", "usd_oz"])
    f.index = pd.to_datetime(f.index)
    f = f.sort_index()
    f = f[f.index < pd.Timestamp(common.today_utc())]
    return f, problem


def _farside(asset: str) -> tuple[pd.Series, str]:
    problem = ""
    sid = f"etf_flows_{asset.lower()}"
    try:
        frame = flows2._etf_frame(asset)
        total = frame["Total"]
        total = total[total.index < common.today_utc()]  # today's row is not complete yet
        hist = common.accumulate(sid, [[d, float(v)] for d, v in total.items()])
    except Exception as exc:  # noqa: BLE001 - Farside blocked: use what earlier runs stored, never work around
        hist = common.accumulate(sid, [])
        if not hist:
            raise RuntimeError(f"Farside {asset} 取不到資料（{type(exc).__name__}），也沒有存檔")
        problem = f"這次 Farside {asset} 取不到（{type(exc).__name__}），改用上次存下的資料。"
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in hist}).sort_index()
    return s, problem


# ---------- signals ----------

def _rank(x: np.ndarray) -> np.ndarray:
    """Point-in-time mid-rank of each value inside the trailing WIN values (including itself)."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not math.isfinite(x[i]):
            continue
        w = x[max(0, i - WIN + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= MINP:
            out[i] = ((w < x[i]).sum() + 0.5 * (w == x[i]).sum()) / len(w)
    return out


def _flow_signals(flow: pd.Series, holdings: pd.Series | None) -> pd.DataFrame:
    """f5/f20 (relative size) and their percentiles on the flow-day grid."""
    out = pd.DataFrame(index=flow.index)
    scale = flow.abs().rolling(WIN, min_periods=ABS_MINP).mean()
    for n in (5, 20):
        s = flow.rolling(n, min_periods=n).sum()
        out[f"f{n}"] = s / holdings.shift(n) * 100 if holdings is not None else s / scale
        out[f"p{n}"] = _rank(out[f"f{n}"].values.astype(float))
    return out


def _panel(flow: pd.Series, holdings: pd.Series | None, entry: pd.Series, own: pd.Series) -> pd.DataFrame:
    """Everything on the flow-day grid. entry[t] = price at the earliest close one can act on day t's flow;
    own[t] = price at day t's own close (for the reverse test)."""
    p = _flow_signals(flow, holdings)
    e = entry.values.astype(float)
    for h in (5, 20):
        p[f"fwd{h}"] = (pd.Series(e).shift(-h) / pd.Series(e) - 1).values * 100
    for k in (5, 20, 60):
        p[f"past{k}"] = (pd.Series(e) / pd.Series(e).shift(k) - 1).values * 100
    o = pd.Series(own.values.astype(float))
    p["own_past20"] = (o / o.shift(20) - 1).values * 100
    p["own_past5"] = (o / o.shift(5) - 1).values * 100
    p["next_p20"] = p["p20"].shift(-20)
    p["next_f5"] = p["f5"].shift(-5)
    return p


# ---------- statistics ----------

def _stat(key: str, a: dict, idx: np.ndarray) -> float:
    if key in ("H1a", "H1b"):
        pk = "p5" if key == "H1a" else "p20"
        pv, y = a[pk][idx], a["fwd20"][idx]
        m = np.isfinite(pv) & np.isfinite(y)
        hi, lo = m & (pv > HI), m & (pv <= LO)
        return float(y[lo].mean() - y[hi].mean()) if hi.any() and lo.any() else float("nan")
    if key == "H2":
        f, y = a["f20"][idx], a["fwd20"][idx]
        m = np.isfinite(f) & np.isfinite(y)
        pos, neg = m & (f > 0), m & (f <= 0)
        return float(y[pos].mean() - y[neg].mean()) if pos.any() and neg.any() else float("nan")
    if key in ("H3a", "H3b", "REV"):
        if key == "REV":
            y, cols = a["next_p20"][idx] - 0.5, [a["own_past20"][idx] / 10, a["p20"][idx] - 0.5]
        else:
            h = 5 if key == "H3a" else 20
            y = a[f"fwd{h}"][idx]
            cols = [a[f"p{h}"][idx] - 0.5, a["past5"][idx], a["past20"][idx], a["past60"][idx]]
        X = np.column_stack([np.ones(len(y))] + cols)
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < 30:
            return float("nan")
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        return float(beta[1] * (100 if key == "REV" else 1))
    raise KeyError(key)


def _arrays(p: pd.DataFrame) -> dict:
    return {c: p[c].values.astype(float) for c in p.columns}


def _boot(key: str, a: dict, n: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    k = int(math.ceil(n / BLOCK))
    vals = []
    for _ in range(BOOT):
        starts = rng.integers(0, max(1, n - BLOCK + 1), k)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
        idx = idx[idx < n]
        vals.append(_stat(key, a, idx))
    v = np.array(vals)
    v = v[np.isfinite(v)]
    if len(v) < BOOT // 2:
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(v, 5)), float(np.percentile(v, 95)), float(v.std(ddof=1))


def _episodes(flag: np.ndarray, gap: int = H) -> int:
    n, last = 0, -10 ** 9
    for i in np.flatnonzero(flag):
        if i - last > gap:
            n += 1
        last = i
    return n


def _counts(key: str, p: pd.DataFrame) -> str:
    v = p.dropna(subset=["fwd20"])
    if key in ("H1a", "H1b"):
        pv = v["p5" if key == "H1a" else "p20"].values
        return f"流入 {_episodes(pv > HI)}｜流出 {_episodes(pv <= LO)} 段"
    if key == "H2":
        f = v["f20"].values
        runs = int(((f[1:] > 0) != (f[:-1] > 0)).sum()) + 1 if len(f) else 0
        return f"正負換手 {runs} 段"
    h = 5 if key == "H3a" else 20
    return f"約 {len(v) // h} 個不重疊的 {h} 天"


def _test(p: pd.DataFrame, seed: int, keys=HYP + ("REV",)) -> dict:
    a = _arrays(p)
    n = len(p)
    out = {}
    for i, key in enumerate(keys):
        full = _stat(key, a, np.arange(n))
        lo, hi, sd = _boot(key, a, n, seed + i)
        out[key] = {"est": full, "lo": lo, "hi": hi, "sd": sd, "count": _counts(key, p)}
    return out


def _sub(p: pd.DataFrame, keys) -> dict:
    out = {}
    for label, a, b in PERIODS:
        q = p[(p.index >= a) & (p.index <= b)].reset_index(drop=True)
        arr = _arrays(q)
        out[label] = {k: _stat(k, arr, np.arange(len(q))) for k in keys}
    return out


# ---------- trading rules ----------

def _rule_target(kind: str, sig: pd.DataFrame) -> pd.Series:
    """Wanted holding decided on flow day t (1 = hold, 0 = cash; NaN before the signal exists)."""
    if kind.startswith("cash_after_inflow"):
        pk = "p5" if kind.endswith("5") else "p20"
        ext = (sig[pk] > HI).astype(float).where(sig[pk].notna())
        return 1.0 - ext.rolling(H, min_periods=1).max().where(sig[pk].notna())
    fk = "f5" if kind.endswith("5") else "f20"
    return (sig[fk] > 0).astype(float).where(sig[fk].notna())


def _kind_of(choice: str | None, est: float | None) -> str:
    if choice in (None, "H1b"):
        return "cash_after_inflow20"
    if choice == "H1a":
        return "cash_after_inflow5"
    if choice == "H2":
        return "hold_when_positive20"
    w = "5" if choice == "H3a" else "20"
    return ("cash_after_inflow" if (est or 0) < 0 else "hold_when_positive") + w


KIND_TEXT = {"cash_after_inflow20": "平常持有；20 日淨流入衝到過去一年前 10% 之後，空手 20 個交易日",
             "cash_after_inflow5": "平常持有；5 日淨流入衝到過去一年前 10% 之後，空手 20 個交易日",
             "hold_when_positive20": "20 日淨流入為正才持有，否則空手",
             "hold_when_positive5": "5 日淨流入為正才持有，否則空手"}


def _stats(net: np.ndarray, year: int) -> tuple[float, float]:
    g = float(np.prod(1 + net))
    ret = ((g ** (year / len(net)) - 1) * 100) if g > 0 else -100.0
    eq = np.concatenate([[1.0], np.cumprod(1 + net)])
    return ret, float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


def _calendar_target(t: pd.Series, lag: int, days: pd.DatetimeIndex) -> pd.Series:
    """Flow-day decision t placed on the close of calendar day t+lag, carried forward."""
    s = pd.Series(t.values, index=t.index + pd.Timedelta(days=lag))
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(days.union(s.index)).ffill().reindex(days)


def _backtest(close: pd.Series, rule_t: pd.Series, lag: int, year: int, grid_is_close: bool,
              start=None) -> dict:
    """close: the asset's daily closes (GLD trading days or crypto calendar days). rule_t on the flow grid."""
    if grid_is_close:
        rule = pd.Series(rule_t.values, index=close.index).shift(lag)
    else:
        rule = _calendar_target(rule_t, lag, close.index)
    ma = close.rolling(50, min_periods=50).mean()
    ma50 = (close > ma).astype(float).where(ma.notna())
    targets = {"hold": pd.Series(1.0, index=close.index), "ma50": ma50, "rule": rule, "both": ma50 * rule}
    valid = rule.notna() & ma50.notna()
    first = close.index[int(np.argmax(valid.values))]
    if start is not None:
        first = max(first, pd.Timestamp(start))
    keep = close.index > first
    change = close.pct_change().fillna(0.0).values
    out = {"start": first, "end": close.index[-1], "n": int(keep.sum()), "targets": targets}
    for name, tg in targets.items():
        for cost in COSTS:
            net, _ = te._run(tg.shift(1).fillna(0.0).values, change, cost)
            out[(name, cost)] = _stats(net[keep], year)
        out[(name, "time")] = float(tg.shift(1)[keep].mean() * 100)
        out[(name, "net")] = te._run(tg.shift(1).fillna(0.0).values, change, COSTS[0])[0]
    out["keep"] = keep
    return out


def _placebo(close: pd.Series, sig: pd.DataFrame, kind: str, lag: int, year: int, grid_is_close: bool,
             actual: dict, seed: int) -> tuple[float, float]:
    """Share of circular shifts of the flow signal whose rule (and ma50 overlay) did at least as well as the real one."""
    rng = np.random.default_rng(seed)
    base = sig.dropna(subset=["p20", "p5"])
    n = len(base)
    keep = actual["keep"]
    change = close.pct_change().fillna(0.0).values
    hold = _stats(actual[("hold", "net")][keep], year)[0]
    ma50r = _stats(actual[("ma50", "net")][keep], year)[0]
    real_rule = actual[("rule", COSTS[0])][0] - hold
    real_both = actual[("both", COSTS[0])][0] - ma50r
    ge_rule = ge_both = 0
    lo = min(WIN, n // 4)
    for _ in range(PLACEBO):
        k = int(rng.integers(lo, max(lo + 1, n - lo)))
        shifted = base.copy()
        for c in ("p5", "p20", "f5", "f20"):
            shifted[c] = np.roll(base[c].values, k)
        rt = _rule_target(kind, shifted).reindex(sig.index)
        if grid_is_close:
            rule = pd.Series(rt.values, index=close.index).shift(lag)
        else:
            rule = _calendar_target(rt, lag, close.index)
        ma50 = actual["targets"]["ma50"]
        for tg, ref, real, which in ((rule, hold, real_rule, 0), (ma50 * rule, ma50r, real_both, 1)):
            net, _ = te._run(tg.shift(1).fillna(0.0).values, change, COSTS[0])
            ex = _stats(net[keep], year)[0] - ref
            if ex >= real:
                if which == 0:
                    ge_rule += 1
                else:
                    ge_both += 1
    return ge_rule / PLACEBO * 100, ge_both / PLACEBO * 100


def _start_grid(bt: dict, close: pd.Series) -> dict:
    """main_signal_check-style monthly start dates (textbook execution) where the short crypto sample allows."""
    last = close.index[-1] - pd.Timedelta(days=180)
    starts, labels = mc._month_starts(close.index, bt["start"] + pd.Timedelta(days=1), last)
    if len(starts) == 0:
        return {"n": 0}
    res = {}
    for name in ("hold", "ma50", "both"):
        r, dd = mc._suffix(mc._equity(bt[(name, "net")]), starts, mc.YEAR)
        res[name] = (r[:, 0], dd[:, 0])
    return {"n": len(starts), "first": labels[0], "last": labels[-1],
            "beat_ma50": int((res["both"][0] > res["ma50"][0]).sum()),
            "shallow_ma50": int((res["both"][1] > res["ma50"][1]).sum()),
            "beat_hold": int((res["both"][0] > res["hold"][0]).sum()),
            "edge": float(np.median(res["both"][0] - res["ma50"][0]))}


# ---------- the whole computation ----------

def _gold_panel(g: pd.DataFrame, lag: int, usd: bool = False) -> pd.DataFrame:
    flow = (g["oz"].diff() * g["usd_oz"] / 1e6) if usd else g["tonnes"].diff()
    entry = g["price"].shift(-lag)
    p = _panel(flow, None if usd else g["tonnes"], entry, g["price"])
    return p


def _crypto_panel(flow: pd.Series, close: pd.Series, lag: int) -> pd.DataFrame:
    entry = close.reindex(flow.index + pd.Timedelta(days=lag))
    own = close.reindex(flow.index)
    return _panel(flow, None, pd.Series(entry.values, index=flow.index), pd.Series(own.values, index=flow.index))


def _compute() -> dict:
    problems = []
    g, prob = _gld()
    if prob:
        problems.append(prob)
    gp = _gold_panel(g, 1)
    gp_valid = gp[gp["p20"].notna() & gp["p5"].notna()]
    first_gold = gp_valid.index[0]
    gold = _test(gp_valid.reset_index(drop=True), SEED)
    gold_sub = _sub(gp_valid, HYP + ("REV",))
    holds = {}
    for k in HYP:
        x = gold[k]
        signs = [np.sign(gold_sub[lab][k]) for lab, _, _ in PERIODS]
        same = all(_ok(gold_sub[lab][k]) for lab, _, _ in PERIODS) and len(set(signs)) == 1 and signs[0] != 0
        ci = _ok(x["lo"]) and _ok(x["hi"]) and (x["lo"] > 0 or x["hi"] < 0)
        holds[k] = bool(same and ci and np.sign(x["est"]) == signs[0])
    ok = [k for k in HYP if holds[k]]
    choice = max(ok, key=lambda k: abs(gold[k]["est"]) / (gold[k]["sd"] or 1e-9)) if ok else None
    kind = _kind_of(choice, gold[choice]["est"] if choice else None)

    # look-ahead perturbation and alternative normalization on gold
    gp2 = _gold_panel(g, 2)
    gp2 = gp2[gp2["p20"].notna() & gp2["p5"].notna()].reset_index(drop=True)
    gpu = _gold_panel(g, 1, usd=True)
    gpu = gpu[gpu["p20"].notna() & gpu["p5"].notna()]
    a2, au = _arrays(gp2), _arrays(gpu.reset_index(drop=True))
    key = choice or "H1b"
    gold_lag2 = {k: _stat(k, a2, np.arange(len(gp2))) for k in HYP}
    gold_usd = {k: _stat(k, au, np.arange(len(gpu))) for k in HYP}
    same_ext = float(((gpu["p20"] > HI) == (gp.loc[gpu.index, "p20"] > HI)).mean() * 100)

    gsig = _flow_signals(g["tonnes"].diff(), g["tonnes"])
    rule_g = _rule_target(kind, gsig)
    gclose = g["price"]
    bt_g = _backtest(gclose, rule_g, 1, mc.SHARE_YEAR, True)
    bt_g2 = _backtest(gclose, rule_g, 2, mc.SHARE_YEAR, True, start=bt_g["start"])
    plc_g = _placebo(gclose, gsig, kind, 1, mc.SHARE_YEAR, True, bt_g, SEED + 77)
    sub_bt = {}
    for lab, a, b in PERIODS:
        m = (gclose.index >= a) & (gclose.index <= b) & bt_g["keep"]
        sub_bt[lab] = {n: _stats(bt_g[(n, "net")][m], mc.SHARE_YEAR)[0] for n in ("hold", "ma50", "rule", "both")}

    crypto, prices = {}, {}
    try:
        prices["BTC"] = te._daily_close()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"比特幣日線取不到（{type(exc).__name__}）。")
    try:
        prices["ETH"] = ta._crypto_close("ETHUSDT")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"以太幣日線取不到（{type(exc).__name__}）。")
    for asset in ("BTC", "ETH"):
        if asset not in prices:
            continue
        try:
            flow, prob = _farside(asset)
            if prob:
                problems.append(prob)
        except Exception as exc:  # noqa: BLE001 - blocked: stop that part honestly
            problems.append(f"{asset} 的 Farside 每日流量這次取不到：{exc}")
            continue
        close = prices[asset]
        cp = _crypto_panel(flow, close, 1)
        valid = cp[cp["p20"].notna() & cp["p5"].notna()]
        res = _test(valid.reset_index(drop=True), SEED + (100 if asset == "BTC" else 200))
        cp2 = _crypto_panel(flow, close, 2)
        cp2 = cp2[cp2["p20"].notna() & cp2["p5"].notna()].reset_index(drop=True)
        lag2 = {k: _stat(k, _arrays(cp2), np.arange(len(cp2))) for k in HYP}
        sig = _flow_signals(flow, None)
        rt = _rule_target(kind, sig)
        bt = _backtest(close, rt, 1, mc.YEAR, False)
        bt2 = _backtest(close, rt, 2, mc.YEAR, False, start=bt["start"])
        plc = _placebo(close, sig, kind, 1, mc.YEAR, False, bt, SEED + (300 if asset == "BTC" else 400))
        dwf = (float(valid["own_past5"].corr(valid["next_f5"])), float(valid["f5"].corr(valid["fwd5"])))
        crypto[asset] = {"flow": flow, "panel": cp, "valid": valid, "res": res, "lag2": lag2, "sig": sig, "rt": rt,
                         "bt": bt, "bt2": bt2, "plc": plc, "grid": _start_grid(bt, close), "dwf": dwf,
                         "close": close}
    gdwf = (float(gp_valid["own_past5"].corr(gp_valid["next_f5"])), float(gp_valid["f5"].corr(gp_valid["fwd5"])))
    return {"g": g, "gp": gp, "gp_valid": gp_valid, "first_gold": first_gold, "gold": gold, "gold_sub": gold_sub,
            "holds": holds, "choice": choice, "kind": kind, "key": key, "gold_lag2": gold_lag2, "gold_usd": gold_usd,
            "same_ext": same_ext, "gsig": gsig, "rule_g": rule_g, "bt_g": bt_g, "bt_g2": bt_g2, "plc_g": plc_g,
            "sub_bt": sub_bt, "crypto": crypto, "gdwf": gdwf, "problems": problems}


# ---------- presentation ----------

def _unit(k: str) -> str:
    if k == "REV":
        return "（每多漲 10%，之後流量百分位變幾點）"
    if k in ("H3a", "H3b"):
        return "（流量從一年最低到最高，之後報酬差幾點）"
    return ""


def _now(r: dict) -> list[dict]:
    out = []
    kind = r["kind"]

    def state(sig: pd.DataFrame, rt: pd.Series, close: pd.Series, label: str, unit: str) -> dict:
        last = sig.dropna(subset=["p20"]).index[-1]
        x = sig.loc[last]
        tail = (sig["p20"] > HI) if kind.endswith("20") else (sig["p5"] > HI)
        if kind.startswith("cash_after_inflow"):
            hits = np.flatnonzero(tail.values[-H:])
            word = (f"空手期中（{_d(sig.index[-H + hits[-1]])} 衝到前 10%，還剩 {hits[-1] + 1} 個流量日）"
                    if len(hits) else "正常持有（近 20 個流量日沒有衝到前 10%）")
        else:
            word = "持有（淨流入為正）" if rt.iloc[-1] >= 0.5 else "空手（淨流出）"
        ma = close.rolling(50).mean()
        ma_word = "收盤高於 50 日均線" if close.iloc[-1] > ma.iloc[-1] else "收盤低於 50 日均線"
        f20 = f"{x['f20']:+.2f}% 的持有量" if unit == "pct" else f"{x['f20']:+.1f} 倍的一年平均單日流量"
        return {"label": f"{label}（最新流量日 {_d(last)}）",
                "value": (f"20 日淨流入 {f20}，在過去一年排第 {x['p20'] * 100:.0f} 百分位；5 日排第 {x['p5'] * 100:.0f} 百分位"
                          f"。規則狀態：{word}；{ma_word}（{_d(close.index[-1])}）")}

    out.append(state(r["gsig"], r["rule_g"], r["g"]["price"], CN["GLD"], "pct"))
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        if c:
            out.append(state(c["sig"], c["rt"], c["close"], f"美國現貨{CN[asset]}", "x"))
        else:
            out.append({"label": f"美國現貨{CN[asset]}", "value": "這次取不到 Farside 流量資料"})
    out.append({"label": "這幾格的意思", "value": "只描述規則目前停在哪個狀態，不是買賣建議"})
    return out


def study() -> dict:
    r = _compute()
    gold, sub, holds, choice, kind = r["gold"], r["gold_sub"], r["holds"], r["choice"], r["kind"]
    key = r["key"]
    gv = r["gp_valid"]

    rows1 = []
    for k in HYP + ("REV",):
        x = gold[k]
        signs = [sub[lab][k] for lab, _, _ in PERIODS]
        hold_word = ("成立" if holds[k] else "不成立") if k != "REV" else "（不參加挑選）"
        rows1.append([HNAME[k] + _unit(k), _pp(x["est"], 2), f"{_pp(x['lo'], 2)} 到 {_pp(x['hi'], 2)}"]
                     + [_pp(v, 2) for v in signs] + [x["count"], hold_word])
    # rows1 has 9 columns -> merge counts into the verdict column to stay at 8
    rows1 = [row[:6] + [f"{row[6]}；{row[7]}"] for row in rows1]

    rows2 = []
    gsign = {k: np.sign(gold[k]["est"]) for k in HYP + ("REV",)}
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        if not c:
            rows2.append([f"{CN[asset]}：取不到資料", "—", "—", "—", "—", "—", "—", "—"])
            continue
        v = c["valid"]
        span = f"{_d(v.index[0])} 到 {_d(v.dropna(subset=['fwd20']).index[-1])}（{len(v.dropna(subset=['fwd20']))} 天）"
        for k in ("H1a", "H1b", "H2", "H3b", "REV"):
            x = c["res"][k]
            match = "—" if not _ok(x["est"]) else ("一樣" if np.sign(x["est"]) == gsign[k] else "相反")
            rows2.append([f"{CN[asset]}｜{HNAME[k]}{_unit(k)}", span, _pp(x["est"], 2),
                          f"{_pp(x['lo'], 2)} 到 {_pp(x['hi'], 2)}", f"{match}（黃金 {_pp(gold[k]['est'], 2)}）",
                          x["count"], _pp(c["lag2"].get(k), 2) if k != "REV" else "—",
                          "是" if k == key else ""])

    bt = r["bt_g"]
    rows3 = []

    def bt_rows(label: str, b: dict, plc: tuple | None, years_note: str):
        for name, word in (("hold", "一直抱著"), ("ma50", "50 日均線"), ("rule", "流量規則"), ("both", "50 日均線＋流量規則")):
            p = "—"
            if plc and name == "rule":
                p = f"{plc[0]:.0f}% 的假流量做得一樣好或更好"
            if plc and name == "both":
                p = f"{plc[1]:.0f}%（和 50 日均線比）"
            rows3.append([f"{label}｜{word}", years_note, _pc(b[(name, COSTS[0])][0]), _pc(b[(name, COSTS[1])][0]),
                          _pc(b[(name, COSTS[0])][1], 0), f"{b[(name, 'time')]:.0f}%", p])

    bt_rows("黃金 GLD", bt, r["plc_g"], f"{_d(bt['start'])} 到 {_d(bt['end'])}")
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        if c:
            b = c["bt"]
            bt_rows(CN[asset], b, c["plc"], f"{_d(b['start'])} 到 {_d(b['end'])}（{b['n']} 天）")

    rows4 = []
    for lab, _, _ in PERIODS:
        s = r["sub_bt"][lab]
        rows4.append([f"黃金 {lab} 每年報酬：抱著｜流量規則｜50 日均線｜兩者合用",
                      f"{_pc(s['hold'])}｜{_pc(s['rule'])}｜{_pc(s['ma50'])}｜{_pc(s['both'])}", "—", "—"])
    b2 = r["bt_g2"]
    lag_cells = [f"{key}：{_pp(gold[key]['est'], 2)} → {_pp(r['gold_lag2'][key], 2)}；規則每年 "
                 f"{_pc(bt[('rule', COSTS[0])][0])} → {_pc(b2[('rule', COSTS[0])][0])}"]
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        lag_cells.append("—" if not c else
                         f"{key}：{_pp(c['res'][key]['est'], 2)} → {_pp(c['lag2'][key], 2)}；規則每年 "
                         f"{_pc(c['bt'][('rule', COSTS[0])][0])} → {_pc(c['bt2'][('rule', COSTS[0])][0])}")
    rows4.append(["晚一天才做（流量日後第二個收盤）"] + lag_cells)
    rows4.append([f"黃金改用美元流量÷一年平均（和比特幣同一種算法）",
                  f"{key}：{_pp(r['gold_usd'][key], 2)}；20 日極端流入的日子和用噸數算的有 {r['same_ext']:.0f}% 相同", "—", "—"])
    grid_cells = []
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        gr = c["grid"] if c else {"n": 0}
        grid_cells.append("起算日不夠" if not gr["n"] else
                          (f"{_d(gr['first'])[:7]} 到 {_d(gr['last'])[:7]} 共 {gr['n']} 個起算日：合用贏過 50 日均線 {gr['beat_ma50']} 個、"
                           f"虧得較淺 {gr['shallow_ma50']} 個、贏過抱著 {gr['beat_hold']} 個；差距中位數 {_pp(gr['edge'])}"))
    rows4.append(["主訊號檢查那種每月起算日（照課本做）", "黃金不做（規則在黃金上已經是 21 年的完整樣本）"] + grid_cells)
    gd = r["gdwf"]
    dwf_cells = [f"過去 5 天漲跌 vs 之後 5 天流量 {gd[0]:+.2f}；過去 5 天流量 vs 之後 5 天漲跌 {gd[1]:+.2f}"]
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        dwf_cells.append("—" if not c else f"{c['dwf'][0]:+.2f}；{c['dwf'][1]:+.2f}")
    rows4.append(["相關係數（DWF Labs 的比法）"] + dwf_cells)

    btc = r["crypto"].get("BTC")
    eth = r["crypto"].get("ETH")
    gk = gold[key]
    years = f"{_d(gv.index[0])[:4]}–{_d(gv.index[-1])[:4]}"
    rev = gold["REV"]
    rev_ok = _ok(rev["lo"]) and rev["lo"] > 0
    rev_all = all(_ok(sub[lab]["REV"]) and sub[lab]["REV"] > 0 for lab, _, _ in PERIODS)
    rev_text = ("；倒是金價先漲、之後流量才跟著進來（全期明確" + ("，三段都是" if rev_all else "，但不是每一段都明顯") + "），流量多半跟在價格後面"
                if rev_ok else "；過去漲跌對之後流量的影響也不明確")
    if choice:
        g_sent = (f"黃金 ETF（GLD）{years} 年裡，「{HNAME[choice]}」通過事前定的標準（三段時期同方向、區間不含零），"
                  f"效果 {_pp(gk['est'], 2)}{rev_text}。")
    else:
        g_sent = (f"黃金 ETF（GLD）{years} 年裡，淨流入流出之後的金價，反向、順向、扣掉價格動能後的領先都沒有通過事前定的標準"
                  f"（三段同方向且區間不含零），所以沒有選出規則{rev_text}。")
    if btc:
        bx = btc["res"][key]
        what = "同一條規則" if choice else "文獻說的反向效果（H1，20 日流量）"
        if _ok(bx["est"]):
            direction = (f"結果是順向、和文獻相反：極端流出之後 20 天比極端流入之後少 {abs(bx['est']):.1f} 個百分點" if bx["est"] < 0
                         else f"結果也是反向：極端流出之後 20 天比極端流入之後多 {bx['est']:.1f} 個百分點")
        else:
            direction = "極端的日子太少，算不出來"
        same_eth = ""
        if eth and _ok(eth["res"][key]["est"]) and _ok(bx["est"]):
            same_eth = "，以太幣方向相同" if np.sign(eth["res"][key]["est"]) == np.sign(bx["est"]) else "，以太幣方向相反"
        b_sent = (f"把{what}原封不動搬到美國現貨比特幣 ETF，{direction}（90% 區間 {_pp(bx['lo'], 1)} 到 {_pp(bx['hi'], 1)}，"
                  f"{bx['count']}）{same_eth}；但比特幣 ETF 2024-01 才上市，能用的樣本不到兩年、極端只有幾段，只能當參考。")
    else:
        b_sent = "比特幣 ETF 的流量這次取不到，沒有做搬過去的檢查。"
    verdict = g_sent + b_sent

    gb = r["bt_g"]
    tables = [
        {"caption": (f"黃金 ETF（GLD）{_d(gv.index[0])} 到 {_d(gv.index[-1])}：每個假設的效果（單位是之後報酬的百分點），"
                     f"括號外是全期，90% 區間用每段 {BLOCK} 個交易日的區塊重抽 {BOOT:,} 次；後三欄是三段時期各自的效果。"
                     "H1 = 極端淨流出之後減極端淨流入之後的 20 天報酬（正 = 反向）；H2 = 20 日淨流入為正減為負之後的 20 天報酬（正 = 順向）；"
                     "H3 = 扣掉過去 5／20／60 天漲跌之後，流量百分位對之後報酬的回歸係數。成立 = 三段同方向且區間不含零。"
                     "淨流入 = 當天買進 ETF 的錢減去贖回的錢（黃金用信託裡金子的噸數變化）。"),
         "columns": ["假設", "全期效果", "90% 區間", PERIODS[0][0], PERIODS[1][0], PERIODS[2][0], "獨立段數；結論"],
         "rows": rows1},
        {"caption": ("同樣的算法原封不動用在美國現貨比特幣、以太幣 ETF（Farside 每日淨流入）。極端一樣是「過去一年（最少 120 個流量日）"
                     "的前／後 10%」，價格用流量日隔天的日線收盤（UTC 0 點）。樣本不到兩年（以太幣更短），區間很寬。"
                     "「晚一天」是流量日後第二個收盤才做的同一個數字。最後一欄標出搬過去的那一條。"),
         "columns": ["資產｜假設", "樣本", "效果", "90% 區間", "方向和黃金比", "獨立段數", "晚一天才做", "搬過去的規則"],
         "rows": rows2},
        {"caption": (f"交易規則（從黃金選出、原封不動）：{KIND_TEXT[kind]}。流量日 t 的訊號最早在下一個收盤（GLD 隔一個交易日；"
                     f"幣是隔天 UTC 0 點）照做。每次買賣扣 {COSTS[0] * 100:.2f}% 或 {COSTS[1] * 100:.2f}%，空手沒有利息。"
                     f"安慰劑：把流量序列整段循環平移（至少錯開一年）{PLACEBO} 次，看假流量做得一樣好或更好的比例，越低代表越不像運氣。"
                     "黃金一年用 252 個交易日，幣用 365 天。"),
         "columns": ["資產｜做法", "期間", f"每年報酬（成本 {COSTS[0] * 100:.2f}%）", f"每年報酬（成本 {COSTS[1] * 100:.2f}%）",
                     "最深虧損", "持有時間", "安慰劑"],
         "rows": rows3},
        {"caption": ("穩健性：黃金三段時期的每年報酬（成本 0.10%）、晚一天才照做、換成和幣一樣的流量算法、主訊號檢查那種每月起算日"
                     "（幣只有一年多的起算日，遠少於主訊號檢查的 108 個 × 24 個收盤時間）、以及「流量跟著價格」的簡單相關係數。"),
         "columns": ["檢查", "黃金 GLD", "比特幣 ETF", "以太幣 ETF"],
         "rows": rows4},
    ]

    notes = [
        "事前登記的設計（中文）：流量訊號用 5 日與 20 日合計淨流入占持有量的百分比；「極端」= 在自己過去一年（滾動、只用當時已知資料）"
        "的分布裡排前 10% 或後 10%。三個假設都在 GLD 2005–2026 上測：H1 反向（極端流入後 20 個交易日報酬低於平常、極端流出後高於平常）；"
        "H2 順向（20 日流量為正時之後 20 天高於平常）；H3 領先（扣掉自己過去 5／20／60 天漲跌之後，流量還能不能說明之後 5／20 天報酬），"
        "以及反過來（過去漲跌能不能說明之後的流量）。用區塊重抽估不確定性、報告獨立段數、分 2005–2012／2013–2019／2020–2026 三段。"
        "挑選標準：三段同方向且區間不含零才算在黃金上成立，最多挑一條；都沒有就說沒有，仍用文獻的 H1 去測比特幣。"
        "規則、成本、對照、安慰劑、搬到比特幣與以太幣，全部在看結果之前就定好，之後沒有改。",
        "事前登記原文（英文）：" + DESIGN_EN,
        ("作者在看任何結果前另外補定的細節：百分位用過去 252 個流量日（含當天、同值算一半）、最少 120 個；極端流入 = 百分位 > 90%、"
         "極端流出 = ≤ 10%；比特幣、以太幣沒有持有量歷史，流量改除以過去一年平均單日流量（黃金也用這種算法核對過）；區塊長度 60、重抽 1,000 次、"
         "固定亂數種子；有好幾條成立時挑效果÷標準差最大的；H3 係數為負對應「流入極端後空手」，為正對應「淨流入為正才持有」。"),
        (f"這次選出的：{HNAME[choice] if choice else '沒有一條成立；交易規則與比特幣檢查用文獻說的 H1（20 日流量）'}。"
         f"黃金各假設成立與否：" + "、".join(f"{k} {'成立' if holds[k] else '不成立'}" for k in HYP) + "。"),
        ("時間對齊（防偷看未來）：流量日 t 的數字要等美國收盤後才公布（Farside 常在亞洲時間隔天早上才齊），所以最早只能在下一個收盤照做："
         "GLD 是下一個交易日收盤；幣是標為 t+1 的日線（UTC t+2 日 0 點收盤）。報酬從那個收盤算起。另外把所有動作再延後一天重算一次，"
         "數字列在第二、第四張表。"),
        ("GLD 的持有噸數是 SPDR 官方歷史檔每天的「Tonnes of Gold」，當天的申購贖回可能要隔一天才反映在噸數上；這只會讓訊號比實際晚，"
         "不會偷看未來。價格用同一個檔案的 GLD 收盤價（含 0.40% 年費的拖累），不是倫敦金價。"),
        ("文獻：Brown、Davies、Ringgenberg（2021）在美國各種 ETF 上發現流入多的 ETF 之後一個月報酬較低（反向），買流出、賣流入每月賺 1.1–2.0%；"
         "Bitwise 和 DWF Labs 發現比特幣 ETF 的流量跟在價格後面（和過去 5 天相關 0.54、和之後 5 天只有 0.03）；黃金 ETF 持有量的高低點"
         "也常落後金價。本研究是在單一資產的時間序列上看，和論文的跨 ETF 比較不完全一樣。"),
        ("比特幣 ETF 2024-01-11 才上市、以太幣 ETF 2024-07-23 才上市；百分位要先累積 120 個流量日，之後還要留 20 天看報酬，"
         "所以可用的樣本不到兩年，極端只有幾段，區間很寬，不能說「在比特幣上成立或不成立」，只能說方向和大小像不像黃金。"
         "主訊號檢查那種 108 個起算日 × 24 個收盤時間的格子在這裡做不到。"),
        ("Farside 彙整各發行商公布的數字，最新一天可能修正；今天（UTC）那一列還沒收完，不用。每次執行把抓到的每日合計存在本頁自己的存檔，"
         "Farside 擋住時就用存檔，不繞過。GLD 檔案也同樣存一份。"),
        ("試過但沒有用上的第二個黃金持有量來源：iShares IAU 的公開下載連結回傳的是網頁不是資料檔，依規定沒有再找別的方法，只用 GLD。"
         "GLD 約占全球黃金 ETF 持有量的三到四成，不是全部的黃金投資需求（還有金條、金幣、央行），"
         "所以「ETF 是大部分投資需求」只在黃金 ETF 這個子市場成立。"),
        ("比特幣、以太幣的區塊重抽只有 400 到 500 多天、每段 60 天，極端流入只有 2 到 6 段，區間就算不含零也不可靠；"
         f"黃金上換成和幣一樣的流量算法（美元流量÷一年平均），20 日極端的日子有 {r['same_ext']:.0f}% 相同，{key} 從 "
         f"{_pp(gk['est'], 2)} 變成 {_pp(r['gold_usd'][key], 2)}"
         + ("，連方向都變了，表示黃金上的效果本來就接近零。" if np.sign(r["gold_usd"][key]) != np.sign(gk["est"]) else "，方向不變。")),
        "空手時沒有算利息；成本只算買賣手續費，沒有稅和滑價。一直抱著不扣成本。",
        "過去的結果不保證未來一樣。這裡只描述流量和之後價格過去的關係、以及規則目前停在什麼狀態，不是買賣建議。",
    ] + r["problems"]

    sources = [
        {"name": "Brown, Davies & Ringgenberg (2021), ETF Arbitrage, Non-Fundamental Demand, and Return Predictability, Review of Finance",
         "url": "https://ideas.repec.org/a/oup/revfin/v25y2021i4p937-972..html"},
        {"name": "Bitwise：Bitcoin ETP series part 2（流量跟著價格）",
         "url": "https://bitwiseinvestments.eu/blog/crypto-research/bitcoin-etp-series-part-2"},
        {"name": "Cointelegraph 報導 DWF Labs：比特幣 ETF 流量跟著價格走",
         "url": "https://cointelegraph.com/news/bitcoin-etf-flows-follow-price-trends-dwf-labs"},
        {"name": "Zeal LLC：GLD 持有量的高低點落後金價", "url": "https://www.zealllc.com/2025/gldchsrv.htm"},
        {"name": "SPDR Gold Shares（GLD）官方頁面", "url": GLD_PAGE},
        {"name": "SPDR Gold Shares 官方歷史檔（Historical Archive，XLSX：每日收盤價、信託黃金噸數）", "url": GLD_XLSX},
        {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": flows2.FARSIDE["BTC"][0]},
        {"name": "Farside Investors：美國現貨以太幣 ETF 每日淨流入", "url": flows2.FARSIDE["ETH"][0]},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
    ]
    return {
        "id": "etf_flows",
        "title": "ETF 資金流入流出之後的價格",
        "question": ("在 ETF 占投資需求很大一塊的黃金上，ETF 淨流入、淨流出之後的金價是反向、順向，還是流量只是跟在價格後面？"
                     "從黃金挑出的規則原封不動搬到比特幣、以太幣 ETF 上，方向一樣嗎？"),
        "asof": _d(r["g"].index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
