"""Historical study "breadth_timing": does the share of coins that are in an up-trend (breadth) help time bitcoin?

Practitioners: Quant (h)Edge measures crypto breadth as the share of coins above their N-day average (50 days tested
best); SentimenTrader's "simple 3-pronged system for bitcoin" holds bitcoin when it is above its 40-day average AND at
least 40% of cryptocurrencies are above their 200-day average AND at least 40% of traders are bullish (a proprietary
survey we cannot reproduce); the equity classic is Zweig's breadth thrust.

Universe (point-in-time): each day, the 50 non-stable coins with the highest trailing 30-day USD volume on Binance
among coins with at least 60 daily candles there (so their averages exist); later-delisted coins count while they
traded. A top-100 version is a check. Bitcoin itself is never in the breadth.

Pre-registered rules (written verbatim before any result was seen; not changed afterwards):
B1 (SentimenTrader two prongs): hold BTC when BTC close > 40-day average AND breadth200 (share of universe above own
   200-day average; coins with < 200 days use what they have only if >= 120 days, else excluded) >= 40%.
B1s (labelled variant): B1 AND Fear & Greed >= 40 (from 2018-02 only).
B2 (breadth filter on main signal): ma50 AND breadth50 >= 50%.
B3 (breadth divergence exit): ma50, but go to cash when BTC is within 5% of its 60-day high while breadth50 has
   fallen >= 20 points from its own 60-day high (narrow rally) - re-enter on the normal ma50 condition once the
   divergence clears.
B4 (thrust entry, rally starts): from cash, enter BTC when the crypto Zweig thrust fires (10-day EMA of the share of
   universe coins up on the day rises from < 0.40 to > 0.615 within 10 days); exit on the ma50 rule. Also report:
   forward 20/60/120-day BTC returns after each thrust vs all days, and how many days earlier than ma50 it re-entered
   after the 2015, 2019, 2020, 2022-23 bottoms.
Evaluate each vs ma50 and holding on the fair grid (as many starts as breadth data allows - breadth starts when the
universe has >= 20 coins; say the start date), ordinary-person execution, other coins (ETH/SOL/BNB/XRP with the same
breadth series), crash-avoided share with whipsaws, median deepest loss. Placebo: circularly shift the breadth series
(200 draws).
Pass test (stated now): ranking score >= ma50's AND median deepest loss no worse AND placebo share <= 10%. For B4
additionally: re-enters earlier than ma50 on average after bottoms without lower ranking score.

Details fixed with the design (before results):
- Data: the Binance USDT daily store completed by new_listings (cache/_store/survivorship/ plus
  cache/_store/new_listings_daily/), read only (nothing is downloaded into or written to those stores here), with the
  same exclusions: stablecoins (closes stuck near one dollar), leveraged tokens, tokenized stocks, redenominations
  (earlier closes rescaled), renames (pairs of the same asset: on each day only the one with the larger 30-day volume
  can be in the universe). Bitcoin's own closes are the dashboard's long daily series (te._daily_close), as in
  main_signal_check; all daily closes are UTC 00:00.
- breadth50 = share of universe coins whose close is above their own 50-day average. breadth200 as in B1; it starts
  once at least 20 universe coins qualify. "Up on the day" = close above the previous close; the share is
  advancers / (advancers + decliners) (unchanged coins left out); EMA with span 10 (alpha 2/11). A thrust fires on the
  close where the EMA moves above 0.615 while some EMA value of the previous 10 days was below 0.40.
- B3: within 5% = close >= 0.95 x the highest close of the last 60 days (today included); breadth50's 60-day high
  includes today.
- B4: a literal "out at any close below the 50-day average" would close every thrust entry made below that average
  at once and make B4 identical to ma50, so it is read as in rally_timing (this codebase's convention for an early
  entry added to ma50): hold when ma50 holds OR a thrust entry is open; a thrust entry opens at the thrust close;
  while ma50 has not yet taken over, it is closed by a close below the 20-day average; once the close is above the
  50-day average the ma50 rule governs (out at its next close below the 50-day average).
- Fair grid: main_signal_check's pieces unchanged (monthly start dates on UTC 00:00 daily closes up to 2023-12;
  monthly starts 2018-01 ... 2023-12 x 24 closing hours from Binance 1-hour candles; ordinary person: 10 hours later,
  0.25% cost, 10% of entries missed, 200 simulations per clock with main_signal_check's seeds for ma50; other coins:
  monthly starts from the first date every rule has a value to three years ago). All rules and ma50 / holding use
  the same (later) start dates. At any closing hour the breadth used is the last finished UTC day's.
  Ranking score = the lowest of the four "yearly return beats holding" shares; median deepest loss = median over the
  start x clock cells (textbook).
- Placebo: the breadth series (breadth50, breadth200, the up-share EMA and so the thrusts) are shifted circularly
  within their own valid span by the same random offset (at least 60 days from either end), 200 draws, seed 7; the
  statistic is the median, over the UTC 00:00 monthly start dates, of the rule's yearly return minus holding's
  (textbook). Placebo share = share of draws at least as good as the real breadth.
- Bottoms: rally_timing's bear markets (50%+ falls since 2013, ended when the price doubles from the low); entry day =
  first held day on or after the low (rally_timing._entry). Crash-avoided share and whipsaws: crash_timing's measures.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import new_listings as nl
from dashboard.studies2 import popular_rules as pr
from dashboard.studies2 import rally_timing as rlt
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import survivorship as sv
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

BTC = "BTCUSDT"
TOP, TOP_CHECK = 50, 100
MIN_AGE = 60
VOL_DAYS = 30
MIN_COINS = 20
B200_MIN = 120
B1_LINE, FG_LINE, FG_FROM = 0.40, 40.0, "2018-02-01"
B2_LINE = 0.50
NEAR_HIGH, DIV_DROP, DIV_DAYS = 0.05, 0.20, 60
EMA_SPAN, THRUST_LOW, THRUST_HIGH, THRUST_DAYS = 10, 0.40, 0.615, 10
FORWARD = (20, 60, 120)
DRAWS, PLACEBO_SEED, MIN_SHIFT = 200, 7, 60
PLACEBO_BAR = 0.10
BOTTOMS_FROM = "2014-01-01"  # the 2015, 2019 (low 2018-12), 2020, 2021 and 2022-23 bottoms
COST = msc.COST
YEAR = 365
KEYS = ("ma50", "B1", "B1s", "B2", "B3", "B4")
CANDS = ("B1", "B1s", "B2", "B3", "B4")
MSC_KI = {"ma50": 1}  # same random draws as main_signal_check for ma50
OTHER = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
FNG_STORE = common.STORE / "breadth_timing_fng.json"
SUMMARY = common.STORE / "breadth_timing_summary.json"
NAMES = {"ma50": "50 日均線（現行主訊號，對照）",
         "B1": "B1 三條件系統的兩條件：比特幣在 40 日均線上，且四成以上的幣在 200 日均線上",
         "B1s": "B1s（替代版）：B1 再加恐懼與貪婪指數 ≥ 40",
         "B2": "B2 50 日均線，再要求一半以上的幣在 50 日均線上",
         "B3": "B3 50 日均線，但漲勢變窄（比特幣近 60 日高點、廣度卻掉 20 點）先下車",
         "B4": "B4 大量幣同時翻多（廣度爆發）就提早上車，下車照 50 日均線"}
SHORT = {"ma50": "50 日均線", "B1": "B1 兩條件", "B1s": "B1s 加恐懼貪婪", "B2": "B2 廣度濾網",
         "B3": "B3 漲勢變窄下車", "B4": "B4 廣度爆發上車"}

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _share(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v * 100:.0f}%"


def _med(vals) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else None


# ---------- data (read only) ----------

def _load() -> tuple[dict, dict, dict]:
    """Binance USDT daily panel exactly as new_listings builds it, without downloading or writing anything."""
    pairs = nl._read(nl.SURV_DIR / "pairs.json", {})
    if not pairs.get("symbols"):
        raise RuntimeError("存活者偏差研究的交易對清單還沒建立")
    bases = {sv._base(s) for s in pairs["symbols"]}
    pairs["symbols"] = [s for s in pairs["symbols"] if sv._wanted(s, bases)]
    raw, info = nl._daily(pairs, 0.0)  # a deadline in the past: nothing is fetched, nothing is written
    if BTC not in raw:
        raise RuntimeError("比特幣的日線取不到")
    info["pegged"] = [s for s, f in raw.items() if float((f["close"] - 1).abs().median()) < sv.PEG_BAND]
    for s in info["pegged"]:
        raw.pop(s)
    info["stocks"] = [s for s, f in raw.items() if s != BTC and sv._base(s).endswith("B") and len(sv._base(s)) >= 3
                      and f.index[0] >= nl.STOCKS_FROM]
    for s in info["stocks"]:
        raw.pop(s)
    info["redenom"] = []
    for s, f in raw.items():  # same rule as new_listings: a near power-of-ten jump of 20x or more is a change of units
        ratio = (f["close"] / f["close"].shift(1)).values
        n_first = len(sv._cut(f)[0])
        lg = np.log10(np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0))
        hits = np.where((np.abs(lg) >= math.log10(nl.JUMP)) & (np.abs(lg - np.round(lg)) <= nl.POW_TOL)
                        & (np.arange(len(lg)) < n_first))[0]
        if len(hits):
            close = f["close"].values.copy()
            for i in hits:
                close[:i] *= ratio[i]
            raw[s] = f.assign(close=close)
            info["redenom"].append(s)
    p = nl._panel(raw, pairs["status"])
    a = nl._assets(p, nl._read(nl.FIRSTS, {}))
    return p, a, info


def _fng() -> tuple[pd.Series | None, str | None]:
    """alternative.me fear & greed per UTC day; a failed download falls back to this module's stored copy."""
    try:
        s = pr._fear_greed()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            FNG_STORE.write_text(json.dumps({_d(k): float(v) for k, v in s.items()}), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        return s, None
    except Exception as exc:  # noqa: BLE001
        stored = nl._read(FNG_STORE, {})
        if stored:
            s = pd.Series({pd.Timestamp(k): float(v) for k, v in stored.items()}).sort_index()
            return s, f"恐懼與貪婪指數這次抓不到（{type(exc).__name__}），改用之前存下來的資料（到 {max(stored)}）。"
        return None, f"恐懼與貪婪指數這次抓不到（{type(exc).__name__}），B1s 這次沒有算。"


# ---------- breadth ----------

def _breadth(p: dict, a: dict, top: int, P: np.ndarray | None = None, qv: np.ndarray | None = None) -> dict:
    """Point-in-time universe and breadth series on the panel's days (each value uses closes up to that day only)."""
    P = p["P"] if P is None else P
    qv = p["qv"] if qv is None else qv
    coins, days = p["coins"], p["days"]
    T, K = P.shape
    b = coins.index(BTC)
    cs = np.cumsum(np.vstack([np.zeros((1, K)), qv]), axis=0)
    vol30 = cs[1:] - cs[np.maximum(np.arange(1, T + 1) - VOL_DAYS, 0)]
    ok = p["life"] & (p["count"] >= MIN_AGE)
    ok[:, b] = False
    groups: dict[str, list[int]] = {}
    for j, s in enumerate(coins):
        groups.setdefault(a["asset"][s], []).append(j)
    for cols in groups.values():
        if len(cols) < 2:
            continue
        v = np.where(ok[:, cols], vol30[:, cols], -1.0)
        best = np.argmax(v, axis=1)
        keep = np.zeros((T, len(cols)), bool)
        keep[np.arange(T), best] = True
        ok[:, cols] &= keep
    score = np.where(ok, vol30, -np.inf)
    idx = np.argsort(-score, axis=1, kind="stable")[:, :top]
    member = np.zeros((T, K), bool)
    np.put_along_axis(member, idx, True, axis=1)
    member &= ok
    n = member.sum(axis=1)
    frame = pd.DataFrame(P)
    m50 = frame.rolling(50, min_periods=50).mean().values
    m200 = frame.rolling(200, min_periods=B200_MIN).mean().values
    prev = np.vstack([np.full((1, K), np.nan), P[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        v50 = member & ~np.isnan(m50)
        n50 = v50.sum(axis=1)
        b50 = np.where(n >= MIN_COINS, (v50 & (P > m50)).sum(axis=1) / np.maximum(n50, 1), np.nan)
        v200 = member & ~np.isnan(m200)
        n200 = v200.sum(axis=1)
        b200 = np.where(n200 >= MIN_COINS, (v200 & (P > m200)).sum(axis=1) / np.maximum(n200, 1), np.nan)
        live = member & ~np.isnan(prev)
        adv = (live & (P > prev)).sum(axis=1)
        dec = (live & (P < prev)).sum(axis=1)
        up = np.where((n >= MIN_COINS) & (adv + dec > 0), adv / np.maximum(adv + dec, 1), np.nan)
    up_s = pd.Series(up, index=days)
    first = up_s.first_valid_index()
    ema = pd.Series(np.nan, index=days)
    if first is not None:
        ema[first:] = up_s[first:].ffill().ewm(span=EMA_SPAN, adjust=False).mean()
    out = {"b50": pd.Series(b50, index=days), "b200": pd.Series(b200, index=days), "ema": ema,
           "n": pd.Series(n, index=days), "n200": pd.Series(n200, index=days),
           "eligible": pd.Series(ok.sum(axis=1), index=days)}
    out["thr"] = _thrusts(ema)
    return out


def _thrusts(ema: pd.Series) -> pd.Series:
    """1 on the close where the EMA moves above THRUST_HIGH while some EMA of the previous 10 days was below
    THRUST_LOW, else 0; NaN while the EMA has no value."""
    e = ema.values
    prev = np.concatenate([[np.nan], e[:-1]])
    low = pd.Series(e).shift(1).rolling(THRUST_DAYS, min_periods=1).min().values
    with np.errstate(invalid="ignore"):
        fire = (e > THRUST_HIGH) & (prev <= THRUST_HIGH) & (low < THRUST_LOW)
    return pd.Series(np.where(np.isnan(e), np.nan, fire.astype(float)), index=ema.index)


# ---------- rules ----------

def _b4(c: list, ma50: np.ndarray, m20: np.ndarray, thr: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    m, a, t = ma50.tolist(), m20.tolist(), thr.tolist()
    entry = False
    for i, x in enumerate(c):
        if m[i] != m[i] or a[i] != a[i] or t[i] != t[i]:
            continue
        if m[i] > 0.5:
            entry = False  # ma50 has taken over: its own exit applies
            out[i] = 1.0
            continue
        if t[i] > 0.5:
            entry = True
        elif entry and x < a[i]:
            entry = False
        out[i] = 1.0 if entry else 0.0
    return out


def _rules(c, br: dict, keys=KEYS) -> dict[str, np.ndarray]:
    """What each rule wants to hold after each close (1 / 0, NaN while it has no value). br holds the breadth arrays
    already aligned to these closes."""
    c = np.asarray(c, dtype=float)
    s = pd.Series(c)
    m20, m40, m50 = (s.rolling(n, min_periods=n).mean().values for n in (20, 40, 50))
    ma50 = ct._gt(c, m50)
    b50, b200, thr, fg = br["b50"], br["b200"], br["thr"], br["fg"]
    nan = np.nan
    out = {"ma50": ma50}
    with np.errstate(invalid="ignore"):
        b1 = np.where(np.isnan(m40) | np.isnan(b200), nan, ((c > m40) & (b200 >= B1_LINE)).astype(float))
        out["B1"] = b1
        out["B1s"] = np.where(np.isnan(b1) | np.isnan(fg), nan, b1 * (fg >= FG_LINE))
        out["B2"] = np.where(np.isnan(ma50) | np.isnan(b50), nan, ma50 * (b50 >= B2_LINE))
        hi60 = s.rolling(DIV_DAYS, min_periods=DIV_DAYS).max().values
        bh60 = pd.Series(b50).rolling(DIV_DAYS, min_periods=DIV_DAYS).max().values
        div = (c >= (1 - NEAR_HIGH) * hi60) & (b50 <= bh60 - DIV_DROP)
        out["B3"] = np.where(np.isnan(ma50) | np.isnan(hi60) | np.isnan(bh60) | np.isnan(b50), nan, ma50 * ~div)
    if "B4" in keys:
        out["B4"] = _b4(c.tolist(), ma50, m20, thr)
    return {k: out[k] for k in keys}


def _align(series: dict, index) -> dict:
    return {k: series[k].reindex(index).values.astype(float) for k in ("b50", "b200", "thr", "fg")}


# ---------- the fair grid (main_signal_check's pieces) ----------

def _daily(close: pd.Series, series: dict, first, last, year: int = YEAR, keys=KEYS, begin=None) -> dict:
    target = pd.DataFrame(_rules(close.values, _align(series, close.index), keys), index=close.index)
    valid = target.notna().all(axis=1)
    if not valid.any():
        raise RuntimeError("規則沒有值")
    b = close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1)
    b = max(b, pd.Timestamp(first)) if begin is None else max(pd.Timestamp(begin), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, b, last)
    change = close.pct_change().fillna(0.0).values
    out = {"starts": labels, "pos": starts, "target": target, "begin": b}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in keys:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, series: dict, begin) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    first = max(pd.Timestamp(msc.CLOCK_STARTS[0]), pd.Timestamp(begin))
    starts, labels = msc._month_starts(m.dates, first, msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, YEAR)
    hr, hd = hr[:, 0], hd[:, 0]
    pre_dates = pd.date_range(end=b0 - pd.Timedelta(days=2), periods=npre, freq="D")
    wants = []
    for h in range(24):
        seq = price[h::24]
        # value j is the close at clock h on the day after label j; the label's UTC daily breadth is already final
        dates = pre_dates.append(pd.date_range(b0 - pd.Timedelta(days=1), periods=len(seq), freq="D"))
        w = _rules(np.concatenate([pre, seq]), _align(series, dates))
        wants.append({k: np.nan_to_num(w[k][npre:][:days], nan=0.0) for k in KEYS})
    out = {}
    for j, k in enumerate(KEYS):
        ki = MSC_KI.get(k, 10 + j)
        tr, td = np.zeros((len(starts), 24)), np.zeros((len(starts), 24))
        pr_, pdd = np.zeros((len(starts), 24, msc.SIMS)), np.zeros((len(starts), 24, msc.SIMS))
        for h in range(24):
            w = wants[h][k]
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            bb, wl = rt._later(h, w, days)
            rng = np.random.default_rng(msc.SEED + 1000 * ki + h)
            pos = rt._missed(wl, msc.P_MISS, rng, msc.SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, msc.SIMS)), eng._mid(bb, pos, msc.P_COST, None)]), starts, YEAR)
            pr_[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - hr[:, None], pr_ - hr[:, None, None]
        out[k] = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                  "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                  "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
                  "p_edge": float(np.median(pedge)), "p_deep_med": float(np.median(pdd))}
    return {"starts": labels, "rules": out, "hold_deep_med": float(np.median(hd))}


def _summary(daily: dict, clock: dict, other: dict) -> dict:
    s = {}
    for k in KEYS:
        dr, dd = daily[k]
        x = dict(clock["rules"][k])
        x["d_n"], x["d_beat"] = len(dr), int((dr > daily["hold"][0]).sum())
        x["d_edge"] = float(np.median(dr - daily["hold"][0]))
        x["d_shallow"] = int((dd > daily["hold"][1]).sum())
        x["o_each"] = {a: (len(o["starts"]), int((o[k][0] > o["hold"][0]).sum())) for a, o in other.items()}
        x["o_n"] = sum(v[0] for v in x["o_each"].values())
        x["o_beat"] = sum(v[1] for v in x["o_each"].values())
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        s[k] = x
    eligible = [k for k in KEYS if s[k]["loss_ok"] >= msc.LOSS_BAR]
    order = sorted(KEYS, key=lambda k: (k not in eligible, -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "order": order, "eligible": eligible}


# ---------- placebo ----------

def _shifted(series: dict, k: int) -> dict:
    """Breadth series rolled circularly by k days within each one's own valid span (thrusts rebuilt)."""
    out = dict(series)
    for name in ("b50", "b200", "ema"):
        s = series[name]
        v = s.values.copy()
        ok = np.flatnonzero(~np.isnan(v))
        if len(ok) > 2 * MIN_SHIFT:
            lo, hi = ok[0], ok[-1] + 1
            v[lo:hi] = np.roll(v[lo:hi], k % (hi - lo))
        out[name] = pd.Series(v, index=s.index)
    out["thr"] = _thrusts(out["ema"])
    return out


def _edge_stat(close: pd.Series, series: dict, pos: np.ndarray, hold_r: np.ndarray) -> dict:
    target = _rules(close.values, _align(series, close.index), CANDS)
    change = close.pct_change().fillna(0.0).values
    out = {}
    for k in CANDS:
        net, _ = te._run(np.concatenate([[np.nan], target[k][:-1]]), change, COST)
        r, _ = msc._suffix(msc._equity(net), pos, YEAR)
        out[k] = float(np.median(r[:, 0] - hold_r))
    return out


def _placebo(close: pd.Series, series: dict, pos: np.ndarray, hold_r: np.ndarray) -> dict:
    real = _edge_stat(close, series, pos, hold_r)
    span = int((~np.isnan(series["b50"].values)).sum())
    rng = np.random.default_rng(PLACEBO_SEED)
    draws = {k: [] for k in CANDS}
    for _ in range(DRAWS):
        k = int(rng.integers(MIN_SHIFT, max(span - MIN_SHIFT, MIN_SHIFT + 1)))
        got = _edge_stat(close, _shifted(series, k), pos, hold_r)
        for key in CANDS:
            draws[key].append(got[key])
    return {k: {"real": real[k], "share": float(np.mean(np.array(draws[k]) >= real[k] - 1e-12)),
                "med": float(np.median(draws[k]))} for k in CANDS}


# ---------- look-ahead ----------

def _lookahead(p: dict, a: dict, base: dict, close: pd.Series, series: dict) -> tuple[int, int]:
    """(cuts checked, problems): breadth and rule positions before a cut must not change when every price and
    volume after the cut is scrambled."""
    rng = np.random.default_rng(PLACEBO_SEED + 1)
    T = len(p["days"])
    bad = cuts = 0
    for cut in (T - 1500, T - 700, T - 120):
        P, V = p["P"].copy(), p["qv"].copy()
        perm = rng.permutation(np.arange(cut + 1, T))
        P[cut + 1:] = P[perm] * rng.uniform(0.2, 5.0, size=(1, P.shape[1]))
        V[cut + 1:] = V[perm] * rng.uniform(0.2, 5.0, size=(1, V.shape[1]))
        alt = _breadth(p, a, TOP, P, V)
        cuts += 1
        for name in ("b50", "b200", "ema", "thr", "n"):
            x, y = base[name].values[:cut + 1], alt[name].values[:cut + 1]
            if not np.array_equal(np.nan_to_num(x, nan=-9), np.nan_to_num(y, nan=-9)):
                bad += 1
        day = p["days"][cut]
        c2 = close.copy()
        after = c2.index > day
        c2[after] = c2[after].values * np.exp(rng.normal(0, 0.2, int(after.sum())))
        alt["fg"] = series["fg"]
        w1 = _rules(close.values, _align(series, close.index))
        w2 = _rules(c2.values, _align(alt, close.index))
        upto = int(close.index.searchsorted(day, side="right"))
        for k in KEYS:
            if not np.array_equal(np.nan_to_num(w1[k][:upto], nan=-9), np.nan_to_num(w2[k][:upto], nan=-9)):
                bad += 1
    return cuts, bad


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    p, a, info = _load()
    problems = []
    series = _breadth(p, a, TOP)
    series100 = _breadth(p, a, TOP_CHECK)
    fg, fg_problem = _fng()
    if fg_problem:
        problems.append(fg_problem)
    fg = fg if fg is not None else pd.Series(dtype=float)
    fg = fg[fg.index >= pd.Timestamp(FG_FROM)]
    days = p["days"]
    for s_ in (series, series100):
        s_["fg"] = fg.reindex(days).ffill(limit=3)

    close = te._daily_close()
    close = close[close.index <= days[-1]]
    start50 = series["b50"].first_valid_index()
    start200 = series["b200"].first_valid_index()

    daily = _daily(close, series, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    begin = daily["begin"]
    rows_h, problem = cr._hourly(BTC)
    if problem:
        problems.append(problem)
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre, series, begin)

    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for symbol, key in OTHER:
        try:
            oc = ta._crypto_close(symbol)
            oc = oc[oc.index <= days[-1]]
            other[key] = _daily(oc, series, oc.index[0], three)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            other.pop(key, None)
            problems.append(f"{key} 這次算不出來（{type(exc).__name__}），沒有算進去。")
    sm = _summary(daily, clock, other)
    s, order = sm["s"], sm["order"]

    # crash share, whipsaws, bottoms
    c = close.values.astype(float)
    target = daily["target"]
    first_i = int(close.index.searchsorted(begin))
    bears = ct._bears(close)
    covered = [i for i, b in enumerate(bears) if b["done"] and b["top"] >= first_i]
    legs = {k: [(1.0, target[k].values)] for k in KEYS}
    crash = {k: [ct._crash(c, legs[k], bears[i]) for i in covered] for k in KEYS}
    whip = {k: ct._whipsaws(c, legs[k], first_i) for k in KEYS}
    avoid = {k: _med([x["avoid"] for x in crash[k]]) for k in KEYS}
    catch = {k: _med([x.get("catch") for x in crash[k]]) for k in KEYS}

    def differ(k: str) -> float:
        ok = target["ma50"].notna() & target[k].notna()
        return float((target[k][ok] != target["ma50"][ok]).mean())

    rbears = [bear for bear in rlt._bears(close) if bear["low"] >= pd.Timestamp(BOTTOMS_FROM)]
    entries = {}
    for i, bear in enumerate(rbears):
        row = {}
        for k in KEYS:
            if pd.isna(target[k].get(bear["low"], np.nan)):
                row[k] = None
            else:
                row[k] = rlt._entry(close, target[k], bear)
        entries[i] = row
    gaps = []  # B4 entry earlier than ma50, in days, per covered bottom
    for i, bear in enumerate(rbears):
        e4, e0 = entries[i]["B4"], entries[i]["ma50"]
        if e4 and e0:
            gaps.append((e0["day"] - e4["day"]).days)
    b4_earlier = float(np.mean(gaps)) if gaps else None

    # top-100 check and one extra day of delay (UTC 00:00 monthly starts, textbook)
    def quick(ser: dict) -> dict:
        d = _daily(close, ser, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], begin=begin)
        out = {}
        for k in KEYS:
            r = d[k][0] - d["hold"][0]
            out[k] = {"beat": float((r > 0).mean()), "edge": float(np.median(r)), "n": len(r),
                      "deep": float(np.median(d[k][1]))}
        tg = d["target"]
        out["avoid"] = {k: _med([ct._crash(c, [(1.0, tg[k].values)], bears[i])["avoid"] for i in covered]) for k in KEYS}
        return out

    chk100 = quick(series100)
    late = {k: (v.shift(1) if k in ("b50", "b200", "ema", "fg") else v) for k, v in series.items()}
    late["thr"] = _thrusts(late["ema"])
    chk_late = quick(late)

    # placebo
    hold_r = daily["hold"][0]
    plc = _placebo(close, series, daily["pos"], hold_r)

    # thrusts and forward returns
    thr = series["thr"].reindex(close.index)
    t_days = [i for i, v in enumerate(thr.values) if v == 1.0]
    valid_i = np.flatnonzero(~np.isnan(thr.values))
    fwd = {}
    for h in FORWARD:
        f_all = [c[i + h] / c[i] - 1 for i in valid_i if i + h < len(c)]
        f_thr = [c[i + h] / c[i] - 1 for i in t_days if i + h < len(c)]
        fwd[h] = (f_thr, f_all)
    thr100 = series100["thr"].reindex(close.index)
    t100 = [i for i, v in enumerate(thr100.values) if v == 1.0]
    fwd100 = {h: [c[i + h] / c[i] - 1 for i in t100 if i + h < len(c)] for h in FORWARD}

    look = _lookahead(p, a, series, close, series)

    # ma50 reproduces main_signal_check on its own 108 start dates
    ref = msc._daily(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    mine = _daily(close, series, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], keys=("ma50",), begin=msc.DAILY_STARTS[0])
    same_n = min(len(ref["ma50"][0]), len(mine["ma50"][0]))
    repro = bool(same_n == len(ref["ma50"][0]) and np.allclose(ref["ma50"][0][:same_n], mine["ma50"][0][:same_n])
                 and np.allclose(ref["ma50"][1][:same_n], mine["ma50"][1][:same_n]))

    # ---------- pass test ----------
    base = s["ma50"]
    tests = {}
    for k in CANDS:
        t_rank = s[k]["low"] >= base["low"] - 1e-12
        t_loss = s[k]["c_deep_med"] >= base["c_deep_med"] - 1e-12
        t_plc = plc[k]["share"] <= PLACEBO_BAR
        parts = [t_rank, t_loss, t_plc]
        if k == "B4":
            parts.append(b4_earlier is not None and b4_earlier > 0)
        tests[k] = parts
    passed = [k for k in CANDS if all(tests[k])]

    # ---------- tables ----------
    n_d, n_c = s["ma50"]["d_n"], s["ma50"]["c_n"]
    rows1 = []
    for k in KEYS:
        x = s[k]
        rank = order.index(k) + 1
        if k == "ma50":
            cell = "比較的基準（不做隨機錯位）"
        else:
            words = ["分數不輸" if tests[k][0] else "分數輸", "虧損不更深" if tests[k][1] else "虧損更深",
                     f"錯位 {_share(plc[k]['share'])} 不比真的差"]
            if k == "B4":
                words.append("底部平均較早上車" if tests[k][3] else "底部沒有較早上車")
            cell = "、".join(words) + ("→ 及格" if all(tests[k]) else "→ 不及格")
        rows1.append([NAMES[k]] + [_share(v) for v in x["checks"]]
                     + [f"{_share(x['low'])}（第 {rank}／{len(KEYS)}）",
                        f"{_pct(x['c_deep_med'])}｜{_share(x['loss_ok'])}", cell])
    rows1.append(["對照：一直抱著", "—", "—", "—", "—", "—", f"{_pct(clock['hold_deep_med'])}｜—", "—"])

    rows2 = []
    for k in KEYS:
        w = whip[k]
        rows2.append([SHORT[k], f"{_share(avoid[k])}（{'、'.join(_share(x['avoid']) for x in crash[k])}）",
                      _share(catch[k]), f"{w['whips']:.1f} 次｜{w['cost']:.1f}%", f"{w['exits']:.1f} 次",
                      "—" if k == "ma50" else _share(differ(k)),
                      f"{_share(chk100[k]['beat'])}｜{_pp(chk100[k]['edge'])}｜{_share(chk100['avoid'][k])}",
                      f"{_share(chk_late[k]['beat'])}｜{_pp(chk_late[k]['edge'])}"])

    def entry_cell(e) -> str:
        if e is None:
            return "沒有廣度資料"
        if e.get("held"):
            return "低點時已持有"
        return f"{_d(e['day'])}（第 {(e['day'] - bear['low']).days} 天，比低點 {_pct(e['above'])}）"

    rows3 = []
    for i, bear in enumerate(rbears):
        head = f"{_d(bear['low'])} 低點 {bear['low_px']:,.0f}（高點 {_d(bear['peak'])}）"
        rows3.append([head] + [entry_cell(entries[i][k]) for k in KEYS] + [
            "—" if not (entries[i]["B4"] and entries[i]["ma50"]) else
            f"{(entries[i]['ma50']['day'] - entries[i]['B4']['day']).days} 天"])
    # 8 columns max: drop B1s from the bottoms table (it is in the notes)
    rows3 = [[r[0], r[1], r[2], r[4], r[5], r[6], r[7]] for r in rows3]
    rows3.append(["平均提早（B4 比 50 日均線，有資料的底部）", "—", "—", "—", "—", "—",
                  "—" if b4_earlier is None else f"{b4_earlier:.1f} 天（{len(gaps)} 次）"])

    rows4 = []
    for h in FORWARD:
        f_thr, f_all = fwd[h]
        rows4.append([f"之後 {h} 天", f"{len(f_thr)} 次", _pct(_med(f_thr) * 100 if f_thr else None),
                      _pct(float(np.mean(f_thr)) * 100 if f_thr else None),
                      _share(float(np.mean([v > 0 for v in f_thr])) if f_thr else None),
                      f"{_pct(_med(f_all) * 100)}｜{_share(float(np.mean([v > 0 for v in f_all])))}",
                      f"{len(fwd100[h])} 次｜{_pct(_med(fwd100[h]) * 100 if fwd100[h] else None)}"])

    assets = list(other)
    o_range = "、".join(f"{x} {_d(other[x]['starts'][0])[:7]}–{_d(other[x]['starts'][-1])[:7]}" for x in assets)
    tables = [
        {"caption": (f"main_signal_check 的公平檢驗，起算日只能從廣度有值以後開始（所有規則都有值的第一天 {_d(begin)} 之後）："
                     f"比特幣 {n_d} 個起算月（UTC 0 點日線，到 {msc.DAILY_STARTS[1][:7]}）、{n_c:,} 格起算月×收盤時間（照課本做）、"
                     f"同樣格子照一般人方式做（10 小時後才做、成本 {msc.P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {msc.SIMS} 次）、"
                     f"其他幣（{o_range}）。前四欄是報酬贏過同一天起一直抱著的比例；排名分數＝四項裡最低的那一項，"
                     f"最深虧損比抱著淺的比例要至少 {msc.LOSS_BAR * 100:.0f}% 才排名。最深虧損中位數是 {n_c:,} 格的中位數。"
                     f"最後一欄是事先定的及格條件：排名分數不輸 50 日均線、最深虧損中位數不更深、隨機錯位（{DRAWS} 次）裡不比真的差的不超過 "
                     f"{PLACEBO_BAR * 100:.0f}%；B4 另外要在底部平均比 50 日均線早上車。"),
         "columns": ["規則", f"比特幣 {n_d} 個起算月", f"{n_c:,} 格（照課本）", "照一般人方式做", "其他幣",
                     "排名分數（名次）", "最深虧損中位數｜比抱著淺的比例", "及格檢驗"],
         "rows": rows1},
        {"caption": (f"大熊市和被洗（UTC 0 點日線、照課本、不扣成本的等效價算法同 crash_timing）。廣度從 {_d(begin)} 起才有值，"
                     f"所以只算高點在那之後、已經走完的 {len(covered)} 次大熊市（跌 50% 以上、從低點漲一倍算結束）："
                     + "、".join(f"{_d(close.index[bears[i]['top']])}→{_d(close.index[bears[i]['low']])}" for i in covered)
                     + f"。被洗＝下車後 {ct.WHIP_DAYS} 天內又用更高價買回，每年次數和每年多付從 {_d(begin)} 算起。"
                     f"倒數第二欄把前 50 名換成前 100 名（{n_d} 個起算月贏過抱著的比例｜差距中位數｜避開跌幅中位數）；"
                     "最後一欄是廣度故意晚一天才用（起算月贏的比例｜差距中位數），看結果是否靠剛好同一天的資料。"),
         "columns": ["規則", "避開的跌幅：中位數（各次）", "抓到的回升：中位數", "被洗：每年次數｜每年多付", "一年下車幾次",
                     "和 50 日均線持有狀態不同的日子", "前 100 名版本", "廣度晚一天才用"],
         "rows": rows2},
        {"caption": ("每次大熊市（2013 年以來跌 50% 以上、從低點漲一倍才算結束）低點之後，各規則第一次持有的日期"
                     "（低點後第幾天、買在比低點高多少）。2015 年和 2018 年 12 月（2019 年漲勢之前）那兩次還沒有廣度資料：幣安 2017 年 8 月才有 USDT 交易對，"
                     f"到 {_d(start50)} 名單才滿 {MIN_COINS} 個幣。"
                     "最後一欄是 B4 比 50 日均線早幾天上車（負數＝比較晚）。B1s 的日期和 B1 大致相同，寫在附註。"),
         "columns": ["大熊市低點", SHORT["ma50"], SHORT["B1"], SHORT["B2"], SHORT["B3"], SHORT["B4"], "B4 早幾天"],
         "rows": rows3},
        {"caption": (f"廣度爆發（前 50 名幣每天「上漲家數 ÷（上漲＋下跌家數）」的 {EMA_SPAN} 日指數平均，在 {THRUST_DAYS} 天內從低於 "
                     f"{THRUST_LOW} 升到高於 {THRUST_HIGH}，也就是股市的 Zweig 廣度爆發）發生那天收盤之後，比特幣 20／60／120 天的漲跌，"
                     f"和 {_d(series['ema'].first_valid_index())} 起所有日子比。最後一欄是改用前 100 名時的爆發次數和中位數。"
                     "爆發次數很少，而且彼此不獨立，只能當描述。"),
         "columns": ["期間", "爆發次數", "爆發後：中位數", "爆發後：平均", "爆發後上漲的比例", "所有日子：中位數｜上漲比例",
                     "前 100 名：次數｜中位數"],
         "rows": rows4},
    ]

    # ---------- now ----------
    last_day = days[-1]
    br_now = {k: series[k].iloc[-1] for k in ("b50", "b200", "ema")}
    last_thr = [days[i] for i, v in enumerate(series["thr"].values) if v == 1.0]
    now = [{"label": f"最後一根收完的日線（{_d(last_day)}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
           {"label": "廣度（前 50 名幣，不含比特幣）",
            "value": (f"{_share(br_now['b50'])} 在自己的 50 日均線上、{_share(br_now['b200'])} 在 200 日均線上；"
                      f"每天上漲家數比例的 10 日平均 {br_now['ema']:.2f}；名單 {int(series['n'].iloc[-1])} 個幣"
                      f"（符合條件的 {int(series['eligible'].iloc[-1])} 個）")},
           {"label": "最近一次廣度爆發", "value": _d(last_thr[-1]) if last_thr else "沒有"}]
    if len(fg):
        now.append({"label": "恐懼與貪婪指數（B1s 用）", "value": f"{fg.iloc[-1]:.0f}（{_d(fg.index[-1])}）"})
    for k in KEYS:
        v = target[k].iloc[-1]
        now.append({"label": NAMES[k], "value": "—" if pd.isna(v) else ("持有" if v > 0.5 else "現金")})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ---------- verdict ----------
    def score(k: str) -> str:
        return _share(s[k]["low"])

    v1 = (f"用「前 50 大幣有幾成站在均線上」決定比特幣進出（{_d(begin)[:7]} 起才算得出來）：公開的三條件系統兩條件版排名分數 "
          f"{score('B1')}、加恐懼貪婪 {score('B1s')}，50 日均線加廣度濾網 {score('B2')}、漲勢變窄提早下車 {score('B3')}、"
          f"大量幣同時翻多提早上車 {score('B4')}，50 日均線 {score('ma50')}。")
    v2 = (f"最深虧損中位數 50 日均線 {_pct(base['c_deep_med'])}，這四條 "
          + "／".join(_pct(s[k]["c_deep_med"]) for k in ("B1", "B2", "B3", "B4"))
          + f"；隨機錯位 200 次裡不比真的差的比例 " + "／".join(_share(plc[k]["share"]) for k in ("B1", "B2", "B3", "B4"))
          + (f"，B4 在 {len(gaps)} 次底部平均早 {b4_earlier:.0f} 天上車。" if b4_earlier is not None else "。"))
    if passed:
        v3 = f"事先定的及格條件只有{'、'.join(SHORT[k] for k in passed)}通過。"
    else:
        near = max(CANDS, key=lambda k: (sum(tests[k]) / len(tests[k]), s[k]["low"]))
        fail = [w for w, ok in zip(["排名分數", "最深虧損", "隨機錯位", "提早上車"], tests[near]) if not ok]
        v3 = f"事先定的及格條件沒有一條通過，最接近的{SHORT[near]}輸在{'、'.join(fail)}。"
    verdict = v1 + v2 + v3

    # ---------- notes ----------
    yearly = []
    for y in range(days[0].year + 1, last_day.year + 1):
        t = days.searchsorted(pd.Timestamp(f"{y}-01-01"))
        if t < len(days):
            yearly.append(f"{y}-01 {int(series['n'].iloc[t])}／{int(series['eligible'].iloc[t])}"
                          f"（200 日 {int(series['n200'].iloc[t])}）")
    cuts, bad = look
    b1s_cells = "；".join(
        f"{_d(bear['low'])}：" + ("沒有資料" if entries[i]["B1s"] is None else
                                 ("低點時已持有" if entries[i]["B1s"].get("held") else _d(entries[i]["B1s"]["day"])))
        for i, bear in enumerate(rbears))
    notes = [
        ("預先定的設計（看結果之前寫下，照原文）：B1（SentimenTrader 三條件裡能重現的兩條）：比特幣收盤高於 40 日均線，而且 breadth200"
         "（名單裡收盤高於自己 200 日均線的比例；不滿 200 天的幣只有在至少 120 天時用它現有的天數，否則不算）≥ 40% 才持有。"
         "B1s（標明是替代版）：B1 再加恐懼與貪婪指數 ≥ 40（2018-02 起才有）。B2（主訊號加廣度濾網）：50 日均線而且 breadth50 ≥ 50%。"
         "B3（廣度背離出場）：50 日均線，但比特幣離 60 日高點 5% 以內、breadth50 卻比它自己的 60 日高點低 20 點以上（漲勢變窄）時改拿現金，"
         "背離消失後照 50 日均線原本的條件回來。B4（爆發進場）：現金時，前 50 名的 Zweig 廣度爆發（每天上漲幣比例的 10 日指數平均 10 天內從"
         " < 0.40 升到 > 0.615）就買進，下車照 50 日均線；並報爆發後 20／60／120 天比特幣報酬對比所有日子，以及 2015、2019、2020、"
         "2022–23 年底部後比 50 日均線早幾天上車。全部和 50 日均線、一直抱著在公平檢驗上比；隨機錯位對照：把廣度序列整段循環平移（200 次）。"
         "及格條件（事先定）：排名分數 ≥ 50 日均線、最深虧損中位數不更深、隨機錯位不比真的差的比例 ≤ 10%；B4 另外要在底部平均比 50 日均線早上車、"
         "而且排名分數不輸。第三個條件（四成交易者看多）是付費的調查資料，沒辦法重現。"),
        ("B4 的「下車照 50 日均線」怎麼解讀（看結果前定的）：如果照字面「收盤低於 50 日均線就下車」，在 50 日均線下方發生的爆發隔天就會被賣掉，"
         "B4 會和 50 日均線一模一樣，所以採用 rally_timing 對「提早上車」的做法：爆發那天收盤買進；還沒站上 50 日均線之前，"
         "收盤跌破 20 日均線就賣；站上 50 日均線之後就交給 50 日均線（之後第一次收在 50 日均線下就賣）。"),
        (f"名單（每天、只用當天以前的資料）：幣安 USDT 現貨、扣掉穩定幣、槓桿代幣、股票代幣，至少有 {MIN_AGE} 根日線，"
         f"取過去 {VOL_DAYS} 天成交額最大的 {TOP} 個（另做 {TOP_CHECK} 個的版本），不含比特幣；同一個資產換代號的只留成交額大的那個；"
         "後來下市的幣在還有交易的日子照樣算。每年 1 月初的名單大小／符合條件的幣數（200 日均線有值的幣數）："
         + "、".join(yearly) + f"。breadth50 從 {_d(start50)} 起有值（名單滿 {MIN_COINS} 個），breadth200 從 {_d(start200)} 起"
         f"（至少 {MIN_COINS} 個幣有 120 天以上），所有規則都有值的第一天是 {_d(begin - pd.Timedelta(days=1))}。"),
        (f"隨機錯位對照的數字（比特幣 UTC 0 點起算月、照課本，報酬比抱著多的中位數，百分點；真的｜200 次錯位的中位數）："
         + "；".join(f"{SHORT[k]} {_pp(plc[k]['real'])}｜{_pp(plc[k]['med'])}" for k in CANDS)
         + "。錯位保留廣度本身的高低起伏和持續時間，只拆掉它和比特幣價格在時間上的對應，所以分數一樣好就代表廣度的時機沒有幫上忙。"),
        (f"B1s 在各底部第一次持有的日期：{b1s_cells}。廣度爆發的日子（前 50 名）："
         + ("、".join(_d(close.index[i]) for i in t_days) if t_days else "沒有") + "。"),
        (f"不偷看未來的檢查：把 {cuts} 個切點之後的幣價和成交額全部打亂（比特幣收盤也加上雜訊），重算名單、廣度和規則，切點以前"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 處改變，這次結果不可信。")
         + f" 50 日均線用同樣的程式在 main_signal_check 的 108 個起算月上重算，結果{'和它一模一樣' if repro else '和它不一樣（要查）'}。"),
        ("限制：只有幣安一家的 USDT 交易對（2017 年 8 月起），所以 2018 年以前沒有廣度、2015 年的底部和 2017 年底開始的大熊市高點都算不到；"
         "起算月和收盤時間比 main_signal_check 少，相鄰起算月共用大部分歷史，24 個收盤時間只差幾個小時，「幾個裡幾個」不是互相獨立的證據。"
         "其他幣本身也在廣度名單裡。2018–2026 年只有少數幾次大熊市，樣本很少。"),
        (f"成本很粗略：照課本 {COST * 100:.2f}%、一般人做法 {msc.P_COST * 100:.2f}%，沒有滑價、稅和現金利息；一直抱著不扣成本。"
         f"資料：交易對清單和日線來自 survivorship／new_listings 存下來的幣安資料（{len(p['coins'])} 個交易對，只讀不寫），"
         f"另有 {len(info['stale'])} 個還在交易的交易對最近幾天沒有更新、{len(info['missing'])} 個沒有日線"
         f"（由 new_listings 補，這裡不下載）。"),
        f"這次執行用了 {time.time() - t0:.0f} 秒。過去的結果不保證未來會一樣；這裡只描述幾條固定規則過去的表現和現在的狀態，不是買賣建議。",
    ]
    notes += problems

    sources = [
        {"name": "Quant (h)Edge：Checking in on crypto's breadth", "url": "https://quanthedge.substack.com/p/checking-in-on-cryptos-breadth"},
        {"name": "SentimenTrader：A simple 3-pronged system for bitcoin", "url": "https://sentimentrader.com/blog/a-simple-3-pronged-system-for-bitcoin"},
        {"name": "LuxAlgo：Breadth thrusts（Zweig 廣度爆發）", "url": "https://www.luxalgo.com/library/concept/breadth-thrusts/"},
        {"name": "幣安公開歷史資料庫（現貨日線，含已下市交易對）", "url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/"},
        {"name": "幣安現貨 K 線端點說明（日線、1 小時線）", "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
        {"name": "alternative.me 恐懼與貪婪指數", "url": "https://api.alternative.me/fng/?limit=0"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
    ]
    seconds = round(time.time() - t0, 1)
    nl._write(SUMMARY, {"asof": _d(last_day), "seconds": seconds, "begin": _d(begin), "start50": _d(start50),
                        "start200": _d(start200), "passed": passed, "lookahead_bad": bad, "repro": repro,
                        "placebo": {k: plc[k]["share"] for k in CANDS}, "score": {k: s[k]["low"] for k in KEYS},
                        "thrusts": len(t_days)})
    result = {"id": "breadth_timing", "title": "用「多少幣在漲」決定比特幣進出",
              "question": ("廣度（有多少比例的幣處在上漲狀態，例如站在自己的 50 日或 200 日均線上）能不能幫比特幣的進出場？"
                           "公開的三條件系統、加在 50 日均線上的廣度濾網、漲勢變窄提早下車、大量幣同時翻多提早上車，"
                           "在公平檢驗和隨機錯位對照下有沒有勝過 50 日均線？"),
              "asof": _d(last_day), "verdict": verdict, "now": now, "tables": tables, "notes": notes,
              "sources": sources}
    return nl._clean(result)


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
