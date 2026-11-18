"""Historical study "vrp_direction": the fear premium in option prices (implied volatility minus realised volatility,
the variance risk premium) - does it point to bitcoin's later direction, which way, and does it help the 50-day rule?

Literature, and which side each rule is on (stated before any result was looked at):
- Bollerslev, Tauchen & Zhou (2009), US stocks: a high variance risk premium comes before HIGHER returns (1-6 months).
  Their side here: V1 (step out when the premium is low) and V3 (step in early when the premium is high).
- Almeida, Grith, Miftachov & Wang (arXiv 2410.15195), bitcoin: Deribit options 2017-07 ... 2022-12, 27-day horizon;
  they sort days into a high- and a low-volatility cluster and find the low-volatility cluster has the higher
  variance premium (0.17 vs 0.12) and the lower return premium (0.62 vs 0.69) - a NEGATIVE link between the two,
  from cluster averages, not from a predictive regression. Their side here: V2 (step out when the premium is high).
- An unreviewed note (bryanvine.github.io) finds DVOL minus realised volatility nearly uncorrelated (+0.12) with later
  spot returns.

Data: Deribit DVOL (BTC and ETH, from 2021-03), daily candle closes; the candle that starts at 00:00 UTC on day t
closes at the end of day t, the same moment as the day-t price close. Prices: the dashboard's long bitcoin daily series
(trend_ensemble._daily_close) and Binance ETHUSDT daily closes. RV30_t = standard deviation of the last 30 daily log
returns (t-29 ... t) x sqrt(365). VRP_t = DVOL_t^2 - RV30_t^2 (in squared volatility percent / 100) and the simple
DVOL_t - RV30_t. The forward 30-day realised volatility (t+1 ... t+30) is used only to describe, never in a rule.
Trailing-year percentile of VRP_t = its rank among the last 365 daily values including day t (at least 90 values).

Pre-registered tests:
1. Predictive regressions of BTC forward 5/20/60-day log returns on VRP_t (and on the simple spread, DVOL level alone
   and RV alone), standardised predictor, Newey-West intervals (lag = horizon); by VRP quintile; split
   2021-03..2022-12 / 2023 / 2024 / 2025-2026 (sign consistency). Same on ETH. The 20-day horizon is the main one.
2. Rules (decided at the close, held from the next day; costs 0.10% textbook, 0.30% ordinary person):
   ma50 = hold while the close is above the 50-day average.
   V1 = ma50, but cash while VRP is in its trailing-year bottom 20% (options cheap relative to realised moves).
   V2 = ma50, but cash while VRP is in its trailing-year top 20%.
   V3 = ma50, or hold anyway while VRP is in its trailing-year top 20% (fear overpriced -> rally start).
   Compared with ma50 and holding on every monthly start from the first day every rule has a value to one year before
   the last day; ordinary person = act one close later, 0.30% cost, 10% of entries missed (robust_trend._missed),
   200 simulations; ETH as second asset; placebo: VRP circularly shifted by a random 90 ... n-90 days, percentile
   recomputed, 200 draws; crash-avoided share (bear markets of 50%+ as crash_timing) and whipsaws.
Pass test (stated before running): the 20-day VRP slope has the same sign in all four periods and its full-sample 90%
interval excludes zero, AND a rule beats ma50 on return / volatility over the whole span with a placebo share of at
most 10%, AND the same on ETH (same slope sign; the passing rule also beats ma50 on ETH).

No look-ahead: a decision at close t uses closes and DVOL up to t only (checked in study() by scrambling everything
after cut dates, and by delaying VRP one extra day). Public, key-less Deribit / Binance / blockchain.info data through
the dashboard's helpers with its honest user agent; the DVOL history is also kept in cache/_store/vrp_direction_*.json.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

ID = "vrp_direction"
DERIBIT = "https://www.deribit.com/api/v2/public"
DVOL_URL = f"{DERIBIT}/get_volatility_index_data"
DVOL_START = datetime(2021, 3, 1, tzinfo=timezone.utc)
DAY_MS = 86_400_000
YEAR = 365
RV_DAYS = 30
PCT_DAYS, PCT_MIN = 365, 90
LOW, HIGH = 0.20, 0.80
HORIZONS = (5, 20, 60)
MAIN_H = 20
Z90 = 1.645
COST, P_COST, P_MISS = 0.001, 0.003, 0.10
SIMS, DRAWS, SEED = 200, 200, 20210301
PLACEBO_BAR = 0.10
PERIODS = (("2021-03–2022-12", "2021-03-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025–2026", "2025-01-01", "2026-12-31"))
RULES = ("hold", "ma50", "v1", "v2", "v3")
VS = ("v1", "v2", "v3")
NAMES = {"hold": "一直抱著", "ma50": "50 日均線（對照）",
         "v1": "V1 均線＋恐懼溢價最低兩成時下車", "v2": "V2 均線＋恐懼溢價最高兩成時下車",
         "v3": "V3 均線＋恐懼溢價最高兩成時提早上車"}
SIDE = {"hold": "—", "ma50": "—", "v1": "美股研究那邊", "v2": "比特幣論文那邊", "v3": "美股研究那邊"}
PRED = {"vrp": "恐懼溢價（隱含²−實際²）", "spread": "隱含−實際（簡單差）", "dvol": "隱含波動（DVOL）", "rv": "過去 30 天實際波動"}


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _f(v, digits: int = 1, sign: bool = True, unit: str = "") -> str:
    if not _ok(v):
        return "—"
    return f"{float(v):{'+' if sign else ''}.{digits}f}{unit}"


def _share(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _dvol(asset: str) -> tuple[pd.Series, str | None]:
    """Daily DVOL candle closes (percent), index = candle day; refreshed from Deribit, kept in the store."""
    path = common.STORE / f"{ID}_dvol_{asset.lower()}.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        stored = {str(k)[:10]: float(v) for k, v in stored.items() if _ok(v)}
    except Exception:  # noqa: BLE001 - a damaged store starts over
        stored = {}
    problem = None
    try:
        start, end, fresh = int(DVOL_START.timestamp() * 1000), int(time.time() * 1000), {}
        for _ in range(10):
            result = common.get_json(DVOL_URL, {"currency": asset, "resolution": "1D", "start_timestamp": start,
                                                "end_timestamp": end}, timeout=30)["result"]
            for row in result["data"]:
                v = float(row[4])
                if math.isfinite(v) and v > 0:
                    fresh[datetime.fromtimestamp(row[0] / 1000, timezone.utc).strftime("%Y-%m-%d")] = v
            end = result.get("continuation")
            if not end or not result["data"]:
                break
        stored.update(fresh)
        common.STORE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(stored.items()))), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - fall back to what is stored
        problem = f"{asset} 的 DVOL 這次從 Deribit 讀不到（{type(exc).__name__}），用之前存下的資料。"
    s = pd.Series(stored, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[s.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    if len(s) < 600:
        raise RuntimeError(f"{asset} DVOL 歷史只有 {len(s)} 天")
    return s, problem


def _frame(close: pd.Series, dvol: pd.Series) -> pd.DataFrame:
    """One row per day from the first DVOL day: close, DVOL, RV30, VRP, forward returns / RV (description only)."""
    lr = np.log(close).diff()
    rv = lr.rolling(RV_DAYS, min_periods=RV_DAYS).std(ddof=1) * math.sqrt(YEAR) * 100
    fwd_rv = lr[::-1].rolling(RV_DAYS, min_periods=RV_DAYS).std(ddof=1)[::-1].shift(-1) * math.sqrt(YEAR) * 100
    ma = close.rolling(50, min_periods=50).mean()
    df = pd.DataFrame({"close": close, "rv": rv, "fwd_rv": fwd_rv, "ma50": (close > ma).astype(float).where(ma.notna())})
    df["dvol"] = dvol.reindex(df.index)
    df = df[df.index >= dvol.index[0]].copy()
    df["dvol"] = df["dvol"].ffill(limit=2)
    df["vrp"] = (df["dvol"] ** 2 - df["rv"] ** 2) / 100
    df["spread"] = df["dvol"] - df["rv"]
    for h in HORIZONS:
        df[f"f{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    return df


def _pct(vrp: pd.Series) -> pd.Series:
    """Trailing-year percentile of each day's VRP among the last 365 values including that day (0..1)."""
    return vrp.rolling(PCT_DAYS, min_periods=PCT_MIN).rank(pct=True)


def _targets(ma50: pd.Series, pct: pd.Series) -> pd.DataFrame:
    """What each rule wants to hold after each close (NaN until both inputs exist)."""
    ok = ma50.notna() & pct.notna()
    m = ma50.fillna(0.0)
    low, high = (pct <= LOW).astype(float), (pct >= HIGH).astype(float)
    out = pd.DataFrame({"hold": 1.0, "ma50": m, "v1": m * (1 - low), "v2": m * (1 - high),
                        "v3": np.maximum(m, high)}, index=ma50.index)
    out.loc[~ok.values] = np.nan
    return out


# ---------- test 1: predictive regressions ----------

def _nw(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float, int]:
    """OLS slope of y on x with a Newey-West standard error; (slope, se, n)."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 40:
        return math.nan, math.nan, n
    X = np.column_stack([np.ones(n), x])
    xtx = np.linalg.inv(X.T @ X)
    b = xtx @ X.T @ y
    u = X * (y - X @ b)[:, None]
    s = u.T @ u
    for k in range(1, min(lag, n - 1) + 1):
        g = u[k:].T @ u[:-k]
        s += (1 - k / (lag + 1)) * (g + g.T)
    v = xtx @ s @ xtx
    return float(b[1]), float(math.sqrt(max(v[1, 1], 0.0))), n


def _regress(df: pd.DataFrame, col: str, h: int) -> dict:
    x = df[col]
    z = (x - x.mean()) / x.std(ddof=1)
    out = {}
    b, se, n = _nw(df[f"f{h}"].values, z.values, h)
    out["all"] = (b, se, n)
    for label, a, e in PERIODS:
        sel = (df.index >= a) & (df.index <= e)
        out[label] = _nw(df[f"f{h}"].values[sel], z.values[sel], h)
    return out


def _quintiles(df: pd.DataFrame) -> list[dict]:
    x = df["vrp"]
    ok = x.notna()
    edges = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    g = np.searchsorted(edges, x[ok].values, side="right")
    sub = df[ok].assign(g=g)
    out = []
    for q in range(5):
        part = sub[sub["g"] == q]
        out.append({"lo": float(part["vrp"].min()), "hi": float(part["vrp"].max()), "n": len(part),
                    "f20": float(part["f20"].mean()), "f60": float(part["f60"].mean()),
                    "up20": float((part["f20"].dropna() > 0).mean()), "fwd_rv": float(part["fwd_rv"].mean()),
                    "dvol": float(part["dvol"].mean())})
    return out


# ---------- test 2: rules ----------

def _net(pos: np.ndarray, change: np.ndarray, cost: float) -> np.ndarray:
    """Daily net return of an all-or-nothing position held during each day (rows = days, any number of columns)."""
    p = np.nan_to_num(pos, nan=0.0)
    trade = np.abs(np.diff(p, axis=0, prepend=np.zeros((1,) + p.shape[1:])))
    ch = change.reshape((-1,) + (1,) * (p.ndim - 1))
    return p * ch - cost * trade


def _from_starts(net: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every start s: yearly return %, return / volatility and deepest loss % from s to the end."""
    e = net.reshape(len(net), -1)
    lg = np.log1p(np.clip(e, -0.999999, None))
    rs = lambda a: np.cumsum(a[::-1], axis=0)[::-1]  # noqa: E731 - suffix sums
    s1, s2, sl = rs(e), rs(e * e), rs(lg)
    n = (len(e) - starts)[:, None]
    mean = s1[starts] / n
    var = np.maximum(s2[starts] / n - mean ** 2, 0.0) * n / np.maximum(n - 1, 1)
    ret = (np.exp(sl[starts] * YEAR / n) - 1) * 100
    vol = np.sqrt(var * YEAR) * 100
    ratio = np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)
    deep = []
    for s in starts:
        eq = np.exp(np.cumsum(lg[s:], axis=0))
        eq = np.vstack([np.ones((1, eq.shape[1])), eq])
        deep.append((eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0) * 100)
    return ret, ratio, np.array(deep)


def _month_starts(index: pd.DatetimeIndex, first: int, last_ts) -> np.ndarray:
    out = []
    for m in pd.date_range(index[first].to_period("M").to_timestamp(), last_ts, freq="MS"):
        i = int(index.searchsorted(max(m, index[first])))
        if i < len(index) and (not out or i > out[-1]):
            out.append(i)
    return np.array(out, dtype=int)


def _rules(df: pd.DataFrame, pct: pd.Series, seed: int, placebo: bool = True) -> dict:
    """Textbook and ordinary-person results of every rule from monthly starts, plus the placebo."""
    target = _targets(df["ma50"], pct)
    valid = target.notna().all(axis=1).values
    first = int(np.argmax(valid))
    change = df["close"].pct_change().fillna(0.0).values
    starts = _month_starts(df.index, first + 1, df.index[-1] - pd.DateOffset(years=1))
    out = {"target": target, "first": first, "starts": starts, "text": {}, "person": {}}
    for k in RULES:
        w = target[k].fillna(0.0).values
        pos = np.concatenate([[0.0], w[:-1]])
        out["text"][k] = _from_starts(_net(pos, change, COST), starts)
        late = np.concatenate([[0.0, 0.0], w[:-2]])
        rng = np.random.default_rng(seed)  # same draws for every rule
        mp = rt._missed(late, P_MISS, rng, SIMS)
        r, q, _ = _from_starts_fast(_net(mp, change, P_COST), starts)
        out["person"][k] = (r, q)
    out["crash_pos"] = {k: target[k].fillna(0.0).values for k in RULES}
    if placebo:
        out["placebo"] = _placebo(df, seed, first)
    return out


def _from_starts_fast(net: np.ndarray, starts: np.ndarray):
    """Like _from_starts without the deepest loss (for the simulations)."""
    e = net.reshape(len(net), -1)
    lg = np.log1p(np.clip(e, -0.999999, None))
    rs = lambda a: np.cumsum(a[::-1], axis=0)[::-1]  # noqa: E731
    s1, s2, sl = rs(e), rs(e * e), rs(lg)
    n = (len(e) - starts)[:, None]
    mean = s1[starts] / n
    var = np.maximum(s2[starts] / n - mean ** 2, 0.0) * n / np.maximum(n - 1, 1)
    ret = (np.exp(sl[starts] * YEAR / n) - 1) * 100
    vol = np.sqrt(var * YEAR) * 100
    return ret, np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0), None


def _placebo(df: pd.DataFrame, seed: int, first: int) -> dict:
    """Return / volatility over the whole span (from the real first valid day) with VRP circularly shifted."""
    vrp = df["vrp"]
    ok = vrp.notna().values
    vals = vrp.values[ok]
    n = len(vals)
    rng = np.random.default_rng(seed + 7)
    change = df["close"].pct_change().fillna(0.0).values
    out = {k: [] for k in VS}
    for _ in range(DRAWS):
        k_shift = int(rng.integers(PCT_MIN, n - PCT_MIN))
        arr = vrp.values.astype(float).copy()
        arr[ok] = np.roll(vals, k_shift)
        shifted = pd.Series(arr, index=vrp.index)
        t =_targets(df["ma50"], _pct(shifted))
        for k in VS:
            w = t[k].fillna(0.0).values
            pos = np.concatenate([[0.0], w[:-1]])
            _, q, _ = _from_starts_fast(_net(pos, change, COST), np.array([first + 1]))
            out[k].append(float(q[0, 0]))
    return out


def _crash_whip(df: pd.DataFrame, res: dict) -> dict:
    c = df["close"].values.astype(float)
    first = res["first"]
    bears = [b for b in ct._bears(df["close"]) if b["top"] >= first]
    out = {"bears": bears, "crash": {}, "whip": {}}
    for k in ("ma50",) + VS:
        legs = [(1.0, res["crash_pos"][k])]
        out["crash"][k] = [ct._crash(c, legs, b) for b in bears]
        out["whip"][k] = ct._whipsaws(c, legs, first)
    return out


# ---------- look-ahead checks ----------

def _lookahead(close: pd.Series, dvol: pd.Series) -> tuple[int, int]:
    """(cut points, rule paths that changed up to the cut) when every close and DVOL after the cut is scrambled."""
    base_df = _frame(close, dvol)
    base = _targets(base_df["ma50"], _pct(base_df["vrp"]))
    rng = np.random.default_rng(11)
    cuts = bad = 0
    for back in (700, 300, 40):
        cut = close.index[-back]
        c2, d2 = close.copy(), dvol.copy()
        after_c, after_d = c2.index > cut, d2.index > cut
        c2[after_c] = c2[after_c].values * np.exp(rng.normal(0, 0.2, after_c.sum()))
        d2[after_d] = d2[after_d].values * np.exp(rng.normal(0, 0.3, after_d.sum()))
        df2 = _frame(c2, d2)
        alt = _targets(df2["ma50"], _pct(df2["vrp"]))
        cuts += 1
        for k in RULES:
            a, b = base[k][base.index <= cut].values, alt[k][alt.index <= cut].values
            if not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)):
                bad += 1
    return cuts, bad


# ---------- the whole computation ----------

def _asset(close: pd.Series, dvol: pd.Series, seed: int) -> dict:
    df = _frame(close, dvol)
    pct = _pct(df["vrp"])
    reg = {(col, h): _regress(df, col, h) for col in PRED for h in HORIZONS}
    res = _rules(df, pct, seed)
    late = _rules(df, pct.shift(1), seed, placebo=False)  # VRP one extra day late
    return {"df": df, "pct": pct, "reg": reg, "quint": _quintiles(df), "rules": res,
            "late": late, "cw": _crash_whip(df, res),
            "late_reg": _regress(df.assign(vrp=df["vrp"].shift(1)), "vrp", MAIN_H)}


def _compute() -> dict:
    problems = []
    btc_close = te._daily_close()
    eth_close = ta._crypto_close("ETHUSDT")
    out = {"problems": problems}
    for key, close, seed in (("BTC", btc_close, SEED), ("ETH", eth_close, SEED + 1)):
        dvol, problem = _dvol(key)
        if problem:
            problems.append(problem)
        out[key] = _asset(close, dvol, seed)
        if key == "BTC":
            out["look"] = _lookahead(close, dvol)
    return out


# ---------- summaries ----------

def _rule_summary(a: dict) -> dict:
    res = a["rules"]
    s = {}
    m_ret, m_q, _ = res["text"]["ma50"]
    pm_r, pm_q = res["person"]["ma50"]
    for k in RULES:
        ret, q, deep = res["text"][k]
        pr, pq = res["person"][k]
        x = {"ret": float(ret[0, 0]), "ratio": float(q[0, 0]), "deep": float(deep[0, 0]),
             "beat": int((q[:, 0] > m_q[:, 0]).sum()), "n": len(res["starts"]),
             "p_beat": float((pq > pm_q).mean()), "p_ratio": float(np.median(pq[0])),
             "time": float(np.nanmean(res["target"][k].values[res["first"]:]))}
        if k in VS:
            pl = np.array(res["placebo"][k])
            x["placebo"] = float((pl >= x["ratio"]).mean())
            x["late_ratio"] = float(a["late"]["text"][k][1][0, 0])
        s[k] = x
    return s


def _passes(btc: dict, eth: dict, sb: dict, se: dict) -> dict:
    r = btc["reg"][("vrp", MAIN_H)]
    b, sd, _ = r["all"]
    signs = [np.sign(r[p[0]][0]) for p in PERIODS if _ok(r[p[0]][0])]
    same_sign = len(signs) == len(PERIODS) and all(s == signs[0] for s in signs)
    ci_ok = _ok(b) and (b - Z90 * sd > 0 or b + Z90 * sd < 0)
    rule_ok = [k for k in VS if sb[k]["ratio"] > sb["ma50"]["ratio"] and sb[k]["placebo"] <= PLACEBO_BAR]
    eb = eth["reg"][("vrp", MAIN_H)]["all"][0]
    eth_sign = _ok(eb) and _ok(b) and np.sign(eb) == np.sign(b)
    eth_rule = [k for k in rule_ok if se[k]["ratio"] > se["ma50"]["ratio"]]
    return {"same_sign": same_sign, "ci_ok": ci_ok, "rule_ok": rule_ok, "eth_sign": eth_sign, "eth_rule": eth_rule,
            "all": same_sign and ci_ok and bool(rule_ok) and eth_sign and bool(eth_rule)}


def _yes(v: bool) -> str:
    return "有" if v else "沒有"


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    btc, eth = r["BTC"], r["ETH"]
    sb, se = _rule_summary(btc), _rule_summary(eth)
    ok = _passes(btc, eth, sb, se)
    df = btc["df"]
    asof = _d(df.index[-1])

    # table 1: regressions
    rows1 = []
    for asset, a in (("比特幣", btc), ("以太幣", eth)):
        cols = list(PRED) if asset == "比特幣" else ["vrp"]
        for col in cols:
            for h in HORIZONS:
                if asset == "比特幣" and col != "vrp" and h != MAIN_H:
                    continue
                x = a["reg"][(col, h)]
                b, sd, n = x["all"]
                rows1.append([f"{asset}：{PRED[col]}", f"{h} 天",
                              f"{_f(b, 2)}（{_f(b - Z90 * sd, 1)} 到 {_f(b + Z90 * sd, 1)}）"]
                             + [_f(x[p[0]][0], 1) for p in PERIODS] + [f"{n:,}"])

    # table 2: quintiles
    rows2 = []
    for q in range(5):
        bq, eq = btc["quint"][q], eth["quint"][q]
        rows2.append([f"第 {q + 1} 組" + ("（最低）" if q == 0 else "（最高）" if q == 4 else ""),
                      f"{bq['lo']:.0f} 到 {bq['hi']:.0f}｜{bq['n']} 天",
                      f"{_f(bq['f20'])}%｜{_share(bq['up20'])}", f"{_f(bq['f60'])}%",
                      f"{bq['dvol']:.0f}%｜{bq['fwd_rv']:.0f}%",
                      f"{_f(eq['f20'])}%", f"{_f(eq['f60'])}%"])

    # table 3: rules
    rows3 = []
    for asset, s in (("比特幣", sb), ("以太幣", se)):
        for k in RULES:
            x = s[k]
            rows3.append([f"{asset}：{NAMES[k]}", f"{x['ratio']:.2f}｜{_f(x['ret'], 0, unit='%')}",
                          _f(x["deep"], 0, sign=False, unit="%"),
                          "—" if k in ("ma50",) else f"{x['n']} 個裡 {x['beat']} 個",
                          "—" if k == "ma50" else _share(x["p_beat"]),
                          _share(x.get("placebo")) if k in VS else "—",
                          _share(x["time"]), SIDE[k]])

    # table 4: crashes and whipsaws
    rows4 = []
    for asset, a in (("比特幣", btc), ("以太幣", eth)):
        cw, c = a["cw"], a["df"]["close"]
        if not cw["bears"]:
            continue
        for k in ("ma50",) + VS:
            parts, catch = [], []
            for b, x in zip(cw["bears"], cw["crash"][k]):
                parts.append(f"{_d(c.index[b['top']])[:7]}→{_d(c.index[b['low']])[:7]}：{_share(x['avoid'])}")
                catch.append(_share(x.get("catch")) if b.get("done") else "未收復")
            w = cw["whip"][k]
            rows4.append([f"{asset}：{NAMES[k]}", "；".join(parts), "；".join(catch), f"{w['exits']:.1f} 次",
                          f"{w['whips']:.1f} 次", f"{w['cost']:.1f} 個百分點"])

    starts = btc["rules"]["starts"]
    first_day = df.index[btc["rules"]["first"]]
    tables = [
        {"caption": ("預測式迴歸：把當天的指標標準化（減平均、除標準差），看之後 5／20／60 天的漲跌（對數報酬，%）平均跟著變多少。"
                     "「每高一個標準差」= 指標比平常高一截時，之後平均多漲（正）或多跌（負）幾個百分點；括號是 90% 區間，用 Newey-West "
                     "方法處理相鄰天數重疊。四段期間的欄位是同一個斜率在各段單獨算的結果，只看正負號是否一致。比特幣其他三個指標只列 20 天（主要的期限）。"),
         "columns": ["指標", "之後幾天", "全期：每高一個標準差（90% 區間）"] + [f"{p[0]}" for p in PERIODS] + ["樣本天數"],
         "rows": rows1},
        {"caption": ("依恐懼溢價（隱含²−實際²，單位：波動率百分點²／100）把每天分成五等份（用全部歷史切，事後回看），之後的平均漲跌。"
                     "「之後 30 天實際晃動」只用來描述選擇權有沒有高估，沒有用在任何規則裡。"),
         "columns": ["恐懼溢價組別", "比特幣：範圍｜天數", "比特幣 20 天後平均｜上漲比例", "比特幣 60 天後平均",
                     "比特幣：當時隱含波動｜之後 30 天實際晃動", "以太幣 20 天後平均", "以太幣 60 天後平均"],
         "rows": rows2},
        {"caption": (f"規則比較（收盤決定、隔天照做、每次買賣扣 {COST * 100:.2f}%）。全期間從 {_d(first_day)}（每條規則都有值的第一天）"
                     f"算到 {asof}。起算日是每個月 1 日，共 {len(starts)} 個（到最後一天前一年為止），比的是「報酬÷波動」有沒有高過 50 日均線。"
                     f"一般人做法：晚一天才照做、每次扣 {P_COST * 100:.2f}%、每次進場有 {P_MISS * 100:.0f}% 機會漏掉，{SIMS} 次模擬 × 每個起算日，"
                     f"和同樣做法的均線比。隨機對照：把恐懼溢價整段錯開隨機天數再算 {DRAWS} 次，「隨機版不比真的差的比例」越低代表越不是運氣。"
                     "最後一欄是事先寫下的文獻方向：美股研究（溢價高→之後漲）支持 V1、V3；比特幣論文（溢價高→報酬溢價低）支持 V2。"),
         "columns": ["規則", "全期間 報酬÷波動｜每年報酬", "全期間最深虧損", "起算日中報酬÷波動贏均線",
                     "一般人做法贏均線的比例", "隨機對照：不比真的差的比例", "持有時間占比", "文獻方向"],
         "rows": rows3},
        {"caption": ("大跌（收盤比前高低 50% 以上，到從低點翻倍為止，和 crash_timing 同一個定義）裡，規則在高點到低點之間避開的跌幅比例，"
                     "以及低點到下一個高點之間接回的漲幅比例；只算規則已經有值以後開始的大跌。「白忙」= 出場後 30 天內又在更高的價格買回，"
                     "成本是每年因此少賺的百分點。避開比例超過 100% 代表規則在那段大跌裡反而賺到（例如中途提早上車碰到反彈）。"),
         "columns": ["規則", "大跌時避開的比例", "收復時接回的比例", "一年出場幾次", "一年白忙幾次", "一年白忙成本"],
         "rows": rows4},
    ]

    # today
    now = []
    for asset, a in (("比特幣", btc), ("以太幣", eth)):
        d, p = a["df"], a["pct"]
        i = d["vrp"].last_valid_index()
        row = d.loc[i]
        rp = (d["dvol"].loc[:i].tail(PCT_DAYS).rank(pct=True).iloc[-1], d["rv"].loc[:i].tail(PCT_DAYS).rank(pct=True).iloc[-1])
        now.append({"label": f"{asset}（{_d(i)} 收盤）：隱含波動 DVOL｜過去 30 天實際波動",
                    "value": f"{row['dvol']:.1f}%（過去一年第 {rp[0] * 100:.0f} 百分位）｜{row['rv']:.1f}%（第 {rp[1] * 100:.0f} 百分位）"})
        now.append({"label": f"{asset}：恐懼溢價（隱含²−實際²）｜簡單差",
                    "value": f"{row['vrp']:.1f}（過去一年第 {p.loc[i] * 100:.0f} 百分位）｜{row['spread']:+.1f} 個百分點"})
        t = a["rules"]["target"].iloc[-1]
        word = lambda v: "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")  # noqa: E731
        now.append({"label": f"{asset}：規則狀態（50 日均線｜V1｜V2｜V3）",
                    "value": f"{word(t['ma50'])}｜{word(t['v1'])}｜{word(t['v2'])}｜{word(t['v3'])}"})
    now.append({"label": "這幾格的意思", "value": "只描述最後一根日線收盤後各指標的位置和規則停在哪個狀態，不是買賣建議"})

    # verdict
    b, sd, _ = btc["reg"][("vrp", MAIN_H)]["all"]
    signs = [btc["reg"][("vrp", MAIN_H)][p[0]][0] for p in PERIODS]
    n_pos = sum(1 for v in signs if _ok(v) and v > 0)
    eb = eth["reg"][("vrp", MAIN_H)]["all"][0]
    dir_word = "平均略漲" if b > 0 else "平均略跌"
    lit = "和美股研究同邊" if b > 0 else "和美股研究相反、和比特幣論文同邊"
    first = (f"恐懼溢價（選擇權預期的晃動比實際晃動多出來的部分）高的時候，比特幣之後 20 天{dir_word}（每高一個標準差 {b:+.1f} 個百分點，"
             f"90% 區間 {b - Z90 * sd:+.1f} 到 {b + Z90 * sd:+.1f}），四段期間 {n_pos} 正 {len(PERIODS) - n_pos} 負、以太幣是 {eb:+.1f}，"
             f"{'方向固定' if ok['same_sign'] and ok['ci_ok'] else '沒有固定方向'}，勉強算{lit}。")
    best = max(VS, key=lambda k: sb[k]["ratio"])
    x = sb[best]
    if x["ratio"] > sb["ma50"]["ratio"]:
        weak = []
        if x["p_beat"] < 0.5:
            weak.append(f"一般人做法只有 {_share(x['p_beat'])} 贏")
        if x["late_ratio"] <= sb["ma50"]["ratio"]:
            weak.append("溢價晚一天才用就輸給均線")
        second = (f"加在 50 日均線上，「{NAMES[best].split(' ', 1)[1]}」（{best.upper()}，{SIDE[best]}）報酬÷波動 {x['ratio']:.2f} "
                  f"贏均線 {sb['ma50']['ratio']:.2f}、隨機對照 {_share(x['placebo'])}"
                  + (f"，但{'、'.join(weak)}" if weak else "")
                  + f"，事先定的檢驗整體{'通過' if ok['all'] else '沒有通過'}。")
    else:
        second = (f"三種加在 50 日均線上的做法報酬÷波動都沒有贏過均線（{sb['ma50']['ratio']:.2f}），"
                  f"事先定的檢驗{'通過' if ok['all'] else '沒有通過'}。")
    third = f"資料只有 {_d(btc['df']['dvol'].first_valid_index())[:7]} 以後約 {len(df) / YEAR:.1f} 年，大跌只有兩三段，結論很不穩。"
    verdict = first + second + third

    look_cuts, look_bad = r["look"]
    lr = btc["late_reg"]["all"]
    notes = [
        "名詞：隱含波動 = 選擇權價格裡大家預期之後會晃多大（Deribit 的 DVOL，30 天期、年化）；實際波動 = 過去 30 天每天漲跌實際晃了多大（年化）；"
        "恐懼溢價 = 預期的晃動比實際晃動多出來的部分。本頁主要用「隱含² − 實際²」（變異數的差，學術上的定義），也列「隱含 − 實際」的簡單差。",
        "事先寫下的檢驗（原文照錄）：1. 比特幣之後 5／20／60 天漲跌對當天恐懼溢價（也單獨對隱含波動、單獨對實際波動）做迴歸，"
        "用 Newey-West 區間；依恐懼溢價分五組；分 2021-03–2022-12／2023／2024／2025–2026 四段看正負號是否一致；以太幣同樣做。"
        "2. 規則（下一個收盤照做，成本 0.10%／0.30%）：V1 = 50 日均線，但恐懼溢價在過去一年最低 20% 時改拿現金；V2 = 50 日均線，"
        "但在最高 20% 時改拿現金；V3 = 跌破 50 日均線時，如果恐懼溢價在最高 20% 就提早回來。和 50 日均線、一直抱著比，"
        "從 2021-04 以後每個月起算、一般人做法、以太幣當第二個資產；隨機對照把恐懼溢價整段錯開（200 次）；大跌避開比例和白忙。"
        "通過條件：20 天斜率四段同號、全期 90% 區間不含 0，而且有一條規則報酬÷波動贏過均線、隨機對照比例 ≤ 10%，而且以太幣方向相同。",
        f"通過檢驗了嗎：四段同號 {_yes(ok['same_sign'])}；全期 90% 區間不含 0 {_yes(ok['ci_ok'])}；"
        f"贏均線且隨機對照 ≤ 10% 的規則：{'、'.join(k.upper() for k in ok['rule_ok']) or '沒有'}；以太幣斜率同號 {_yes(ok['eth_sign'])}；"
        f"以太幣上也贏均線的：{'、'.join(k.upper() for k in ok['eth_rule']) or '沒有'}。整體：{'通過' if ok['all'] else '沒有通過'}。",
        "文獻在說什麼：Bollerslev、Tauchen、Zhou（2009）用美股發現恐懼溢價高之後 1–6 個月報酬較高（正向）。Almeida、Grith、Miftachov、Wang"
        "（arXiv 2410.15195）用 Deribit 比特幣選擇權 2017-07 到 2022-12、27 天期，把日子分成高波動、低波動兩群：低波動群恐懼溢價較高（年化 0.17 對 0.12）、"
        "報酬溢價較低（0.62 對 0.69），所以說兩者是負向關係、和美股相反；但那是兩群的平均比較，論文明說沒有做預測式迴歸。"
        "另一篇未審查的網頁筆記發現 DVOL 減實際波動和之後的漲跌幾乎無關（相關 +0.12）。",
        f"五等份的界線是用全部歷史事後切的；規則用的「過去一年百分位」只用當天以前（含當天）的 365 天，前 {PCT_MIN} 天以後就開始算，"
        "所以 2021 年的百分位是用不到一年的資料。",
        f"偷看未來的檢查：把 {look_cuts} 個切點之後的價格和 DVOL 全部打亂，切點以前的規則狀態有 {look_bad} 條變了（應該是 0）。"
        f"DVOL 用的是當天 UTC 0 點開始那根日線的收盤，也就是和當天價格收盤同一刻。把恐懼溢價再晚一天才用：20 天斜率變成 {_f(lr[0], 2)}"
        f"（原本 {_f(b, 2)}），V1／V2／V3 全期間報酬÷波動變成 {sb['v1']['late_ratio']:.2f}／{sb['v2']['late_ratio']:.2f}／"
        f"{sb['v3']['late_ratio']:.2f}（原本 {sb['v1']['ratio']:.2f}／{sb['v2']['ratio']:.2f}／{sb['v3']['ratio']:.2f}）。",
        "之後 20、60 天的漲跌互相重疊，真正獨立的觀察只有幾十個；四段期間每段只有一兩年，正負號一致也可能是運氣。",
        "相鄰的起算日共用大部分的歷史，所以「幾個裡贏幾個」不是互相獨立的證據；模擬用固定亂數種子，資料沒變重跑結果一樣。",
        "成本很粗略：照課本扣 0.10%、一般人扣 0.30%，現金沒有利息、沒有算稅；一直抱著也從第一天扣一次買進成本。",
        "比特幣價格用儀表板的長期日線（2017 年以後是幣安 BTCUSDT，UTC 0 點收盤），以太幣用幣安 ETHUSDT；DVOL 是 Deribit 的指數，不是實際成交價。",
        "過去的結果不保證未來會一樣。這裡只是整理歷史、說明規則現在停在哪個狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": ID,
        "title": "選擇權裡的恐懼溢價能看出方向嗎？",
        "question": ("選擇權價格裡預期的晃動比實際晃動多出來的部分（恐懼溢價）高或低的時候，比特幣之後有沒有固定的漲跌方向？"
                     "和美股研究同方向還是相反？加在 50 日均線上提早下車或提早上車，有沒有比較好？"),
        "asof": asof,
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Bollerslev, Tauchen & Zhou (2009) Expected Stock Returns and Variance Risk Premia", "url": "https://doi.org/10.1093/rfs/hhp008"},
            {"name": "Almeida, Grith, Miftachov & Wang：Risk Premia in the Bitcoin Market（arXiv 2410.15195）", "url": "https://arxiv.org/abs/2410.15195"},
            {"name": "上面論文的摘要文章（Harbourfront Quant）", "url": "https://harbourfrontquant.substack.com/p/return-and-variance-risk-premia-in"},
            {"name": "Alexander & Imeraj：比特幣的 VIX", "url": "https://sro.sussex.ac.uk/id/eprint/91094/"},
            {"name": "未審查的網頁筆記：DVOL 恐懼溢價與之後漲跌", "url": "https://bryanvine.github.io/alpha-research/paper1.html"},
            {"name": "Deribit 波動率指數 DVOL 日線（公開 API）", "url": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=1D&start_timestamp=1614556800000&end_timestamp=1900000000000"},
            {"name": "Deribit DVOL 頁面", "url": "https://www.deribit.com/statistics/BTC/volatility-index"},
            {"name": "幣安 BTC／ETH 對 USDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017 年以前，均線暖機用）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
