"""Historical study "reluctant_sellers": when bitcoin is rising and holders COULD sell at a profit, yet few coins are
sent to exchanges ("reluctant sellers"), does the rise tend to continue - and when a rise is met by a surge of coins
sent to exchanges ("distribution"), does it tend to stall? Can either help the main signal ma50 (hold while the daily
close is above its 50-day average)?

History (why this study exists, stated plainly). supply_demand tested "few willing sellers + buying comes in" and
failed. The project owner pointed out that its definition of "few sellers" was wrong: it used "holders' paper profit
is LOW" (MVRV below its two-year median) plus low exchange inflows. That describes holders who are stuck at a loss in
a bear market, not holders who are unwilling to sell. The corrected idea: selling pressure is low when holders COULD
sell at a profit and price is rising, yet they do not send coins to exchanges (RELUCTANT); selling pressure is high
when a price rise is met by a surge of coins sent to exchanges (DISTRIBUTION). This is a SECOND attempt at the same
idea with a definition changed after seeing a failure, so the placebo bar is stricter: 5%, not the usual 10%.
Earlier related results: exchange_reserves (exchange balances) found forward returns opposite to the paper and rules
that lose to ma50; price_impact (seller scarcity measured from price response) also failed.

Data (only free key-less endpoints the project already uses, fixed user agent "btc-research-dashboard"):
  BTC daily closes: the dashboard's (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Coin Metrics community API, BTC FlowInExNtv (BTC sent to exchange-tagged addresses per day) and CapMVRVCur (market
    value / realised value), fetched with supply_demand's loader into this study's own store
    cache/_store/reluctant_sellers_cm.json (fallback only). A value dated D is used from the close of candle D+1.
  Fair exam inputs as supply_demand / price_impact: Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance) and MSTR
    (Yahoo) daily closes through main_signal_check's loaders.

Definitions (fixed before running; on Coin Metrics data dates D):
  INFLOW30(D) = sum of FlowInExNtv over D-29 .. D (all 30 values required).
  MED(D), P80(D) = median and 80th percentile of INFLOW30 over the 365 previous data days D-365 .. D-1 (>= 330 values).
  LOWFLOW = INFLOW30 < MED; HIGHFLOW = INFLOW30 > P80; PROFIT = MVRV > 1. Missing inputs -> undefined.
  On a decision day t (UTC daily close t, held during t+1) the legs of data day t-1 are used.
  above20(t) = close(t) above the mean of the last 20 closes including t; ma50 = main_signal_check's.
  RELUCTANT(t) = above20(t) AND PROFIT(t-1) AND LOWFLOW(t-1).
  DISTRIBUTION(t) = above20(t) AND HIGHFLOW(t-1). (The two cannot both be on.) Undefined legs count as OFF in rules.

Test A - is the idea right? Sample: decision days with LOWFLOW/HIGHFLOW and PROFIT defined and a full 30-day forward
  window. Among days with above20: three groups RELUCTANT, DISTRIBUTION, neither (below-20 days are left out of the
  groups but the bootstrap draws calendar blocks of all sample days). Forward close[t+30]/close[t]-1, close[t+60]/
  close[t]-1 (days without a 60-day window are skipped for that column), and the deepest drop within 30 days
  min(close[t+1..t+30])/close[t]-1 capped at 0. Mean and median per group, 90% intervals from a circular block
  bootstrap of days (60-day blocks, 1,000 draws, fixed seed) inside each period: full, 2012-16, 2017-21, 2022-now
  (clipped to the sample). RELUCTANT minus DISTRIBUTION 30-day mean with its 90% interval in every period.
  Control (price_impact's matching): full period, above20 days; deciles of the trailing 30-day return
  close[t]/close[t-30]-1 (edges from those days); matched difference = RELUCTANT minus non-RELUCTANT mean 30-day
  forward return inside each decile, weighted by RELUCTANT days in it (deciles holding both kinds only); 90%
  interval from the same bootstrap (report).
  Pre-registered pass, all four: full-period 30-day mean RELUCTANT > neither > DISTRIBUTION; RELUCTANT > DISTRIBUTION
  in at least two of the three sub-periods; the full-period RELUCTANT-minus-DISTRIBUTION 90% interval excludes zero;
  the matched control difference is positive.

Test B - early entry: R1 = hold when ma50 is on OR RELUCTANT.
Test C - not shaken out: R2 = hold when ma50 is on OR (ma50 was on at any of the 20 previous closes t-20..t-1 AND
  LOWFLOW AND PROFIT). No 20-day-average condition: the point is to sit through a dip holders are not selling into.
Filter side, report only: X1 = ma50 on AND NOT DISTRIBUTION (it can only avoid, never add).
Fair exam = main_signal_check's, wired as price_impact's _Exam (supply_demand's exam with a chosen first month):
  first month 2015-01 (the flags exist from about 2012, so main_signal_check's own starts are used): BTC daily
  monthly starts 2015-01 .. 2023-12 textbook 0.10%; clock starts 2018-01 .. 2023-12 x 24 closing hours textbook; the
  same cells as an ordinary person (decide at the clock's close, act 10 hours later, 0.25% cost, 10% of entries
  missed, 200 simulations per clock, main_signal_check's seeds and its ma50 draws); other assets ETH SOL BNB XRP
  (Binance) and MSTR (Yahoo) from their first valid month to three years before today, each with its OWN 50-day and
  20-day averages and BTC's legs as a market-wide flag (as supply_demand did): R1 = own ma50 OR (own above20 AND BTC
  PROFIT&LOWFLOW); R2 = own ma50 OR (own ma50 on at any of its 20 previous closes AND BTC PROFIT&LOWFLOW); X1 = own
  ma50 AND NOT (own above20 AND BTC HIGHFLOW). ma50 is re-scored on exactly the same starts; the number of starts is
  reported. On the 24-clock grid a close on UTC date E uses the legs of decision candle E-1; MSTR's close on date D
  uses decision candle D-1; other coins' UTC closes use the same candle's legs.
  Ranking score = worst of the four "beats holding" shares with main_signal_check's eligibility. Median deepest loss
  = median over the clock textbook cells.
  Placebo: the INFLOW leg only (LOWFLOW for R1 and R2, HIGHFLOW for X1), on decision dates from its first defined
  day, circularly shifted by k in [180, n-180] days, 200 draws (the same k for all rules); PROFIT and the price
  conditions stay in place; the whole exam rebuilt with 20 ordinary-person simulations per clock (same seeds) and the
  real rule re-scored the same way; placebo share = share of draws whose ranking score is at least the real rule's.
  Pre-registered pass for R1 and for R2, all three: ranking score >= ma50's; median deepest loss no more than 5
  points worse than ma50's; placebo share <= 5%. Two rules are tried, so one passing alone is weaker evidence.

Report only:
  - days R1 and R2 hold that ma50 does not (from the first defined day), by year, with average next-day BTC return;
  - R1 early entries (R1 turns on while ma50 is off): "right" = ma50 turns on within 30 days at a higher close than
    the entry close; each stint runs to the first close where R1 goes to cash or ma50 turns on; its BTC return minus
    0.10% per trade made in it (supply_demand._entries' arithmetic);
  - R2 and every ma50 exit (ma50 on at t-1, off at t): whipsaw = ma50 turns back on within 20 days at a close higher
    than the exit close, else real decline (unknown if the window is not over). "Sat through" = R2 still holds at t.
    Stint = t to the first close where R2 goes to cash or ma50 turns on; advantage over ma50 = BTC stint return, plus
    2 x 0.10% when the stint ends because ma50 turns on (ma50 sold at t and buys back at the end), else +0 (both sold
    once). Every sat-through real decline is listed (e.g. the 2018 and 2022 bear-market starts);
  - rallies from rally_sources._rallies (+80% / 30%): share of RELUCTANT and DISTRIBUTION days trough -> ma50 date,
    ma50 date -> peak, and the 60 days after the peak;
  - lead/lag: x(t) = log(INFLOW30 of data day t-1 / INFLOW30 of data day t-15) (14-day change known at t);
    correlation of x(t) with BTC's 14-day log return ending at t+k, k = -30..+30 (k > 0: inflow change first,
    k < 0: price first); noise band = 95th percentile of |corr| over 200 circular shifts of x by >= 180 days,
    pooled over the shown k.
Test D (owner's definition, third definition of the idea, added 2026-11-19 after Tests A-C were fixed and run; Tests
  A-C above are left exactly as pre-registered). The owner's correction: "reluctant" and "distribution" must be defined
  by SELLING BEHAVIOUR ONLY, independent of price direction and of whether holders are in profit (holders may refuse
  to sell in a fall because the price is too low, and may dump in a fall because they cannot stand it).
  QUIET = LOWFLOW (INFLOW30 below its trailing-365-day median); HEAVY = HIGHFLOW (INFLOW30 above its trailing-365-day
  80th percentile); MIDDLE = otherwise (data day t-1, as above). No MVRV condition, no price condition.
  Price side: UP = above20(t), DOWN = close at or below its 20-day average.
  Sample: decision days with the inflow legs defined and a full 30-day forward window (same as Test A). Six cells
  (UP/DOWN x QUIET/MIDDLE/HEAVY): days and share, mean and median 30- and 60-day forward return, mean deepest drop
  within 30 days, circular block-bootstrap 90% intervals (60-day blocks, 1,000 draws, fixed seed), full period and
  2012-16 / 2017-21 / 2022-now. Pooled QUIET-minus-HEAVY difference = the equal-weight average of (QUIET mean - HEAVY
  mean) within UP and within DOWN, 30-day forward return, with its bootstrap interval, in every period.
  Control: QUIET versus HEAVY matched on trailing-30-day-return deciles (price_impact's matching, weights = QUIET days
  per decile, deciles holding both kinds only), separately within UP and within DOWN (deciles from that side's days),
  full period, with bootstrap intervals (report).
  Pre-registered reading: supported if within BOTH price sides QUIET's full-period 30-day mean is above HEAVY's, AND
  the full-period pooled difference has a 90% interval excluding zero, AND the pooled difference is positive in at
  least two of the three sub-periods. If DOWN+HEAVY (capitulation) is instead followed by the best returns, that is a
  different idea (washout) and is stated as an observation only, not a result.
  Rule G1 (one rule only) = hold when ma50 is on OR QUIET (whatever the price is doing). Other assets: own ma50 OR
  BTC's QUIET. Same fair exam, same starts, same pass test as R1/R2 (ranking score >= ma50's; median deepest loss no
  more than 5 points worse; placebo share <= 5% with the QUIET leg circularly shifted >= 180 days, 200 draws, 20
  ordinary-person simulations per clock with the real G1 re-scored the same way). Report: extra days by year, time in
  market (share of BTC days held from 2015-01, versus ma50), median and worst deepest loss, calendar-year return of G1
  minus ma50 in 2018 and 2022 (0.10% cost). Report only: G2 = ma50 on AND NOT HEAVY (filter side; placebo shifts the
  HEAVY leg).
Look-ahead check: inputs cut at three dates (Coin Metrics one day earlier); no flag, R1 or R2 position up to the cut
  may change.
Honesty: thresholds (30 / 365 days, median, 80th percentile, MVRV 1, 20-day average, 20-day memory, 60-day blocks)
  were fixed by reasoning before the run; no alternatives tried. Coin Metrics exchange tagging is today's address
  labels applied backwards and improves over time, so old inflows are understated relative to today's vintage; the
  trailing-365-day percentile partly absorbs this but not fully; 2012-16 has fewer tagged exchanges. A pass would
  only make a rule a candidate for the daily shadow log. Not advice.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import rally_sources as rs
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
DAY = pd.Timedelta(days=1)
CM_URL = sd.CM_URL
STORE_CM = common.STORE / "reluctant_sellers_cm.json"

SUM_DAYS, WIN, WIN_MIN, P_HI = 30, 365, 330, 0.80
A20, MEMORY = 20, 20
H = (30, 60)
BLOCK, BOOT = 60, 1000
FIRST_MONTH = msc.DAILY_STARTS[0]
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.05
WHIP_DAYS, RIGHT_DAYS, AFTER_PEAK = 20, 30, 60
KI = {"ma50": 1, "r1": 96, "r2": 97, "x1": 98, "g1": 99, "g2": 100}
LL_K = tuple(range(-30, 31))
LL_SHOW = (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30)
CUTS = ("2016-06-30", "2020-06-30", "2024-06-30")
PERIODS = (("2012–16", "2012-01-01", "2016-12-31"), ("2017–21", "2017-01-01", "2021-12-31"),
           ("2022 起", "2022-01-01", "2100-01-01"))
GROUPS = ("惜售", "出貨", "兩個都不是")
NAME = {"ma50": "50 日均線（基準）",
        "r1": "R1：50 日均線，或「惜售」也持有（提早上車）",
        "r2": "R2：50 日均線，或「均線 20 天內還在＋流入少＋有賺」也持有（不被洗出場）",
        "x1": "X1：50 日均線，但「出貨」時不持有（只會少拿，只描述）",
        "g1": "G1：50 日均線，或「流入少」也持有（不管價格漲跌）",
        "g2": "G2：50 日均線，但「流入多」時不持有（只描述）"}
SHORT = {"ma50": "50 日均線", "r1": "R1", "r2": "R2", "x1": "X1", "g1": "G1", "g2": "G2"}
SIDES = ("價格在上（站上 20 日均線）", "價格在下（在 20 日均線以下）")
FLOWS = ("流入少（惜售）", "流入中間", "流入多（出貨）")

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- data ----------

def _load() -> dict:
    problems = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    raw, n = sd._stored(STORE_CM, sd._cm_fetch)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
    other, more = msc._load_other()
    problems += more
    return {"close": close, "cm": cm, "other": other, "problems": problems}


def _raw(cm: pd.DataFrame) -> dict:
    """Legs on Coin Metrics data dates."""
    inflow = cm["FlowInExNtv"] if "FlowInExNtv" in cm else pd.Series(dtype=float)
    mv = cm["CapMVRVCur"] if "CapMVRVCur" in cm else pd.Series(dtype=float)
    in30 = inflow.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    past = in30.shift(1).rolling(WIN, min_periods=WIN_MIN)
    med, p80 = past.median(), past.quantile(P_HI)
    okf = in30.notna() & med.notna() & p80.notna()
    low = (in30 < med).astype(float).where(okf)
    high = (in30 > p80).astype(float).where(okf)
    prof = (mv > 1).astype(float).where(mv.notna())
    return {"in30": in30, "med": med, "p80": p80, "mv": mv, "low": low, "high": high, "prof": prof}


def _legs(raw: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision-date legs (data day t-1); NaN = undefined."""
    out = pd.DataFrame(index=index)
    for k in ("low", "high", "prof"):
        out[k] = raw[k].reindex(index - DAY).values
    return out


def _flags(close: pd.Series, lg: pd.DataFrame) -> pd.DataFrame:
    a20 = sd._above20(close.values)
    low, high, prof = (lg[k].fillna(0.0).values for k in ("low", "high", "prof"))
    out = pd.DataFrame(index=close.index)
    out["a20"] = a20
    out["lowprof"] = low * prof
    out["high"] = high
    out["quiet"] = low
    out["rel"] = a20 * low * prof
    out["dist"] = a20 * high
    out["defined"] = lg["low"].notna().values & lg["prof"].notna().values
    return out


def _recent(ma: np.ndarray) -> np.ndarray:
    """1 when ma was on at any of the MEMORY previous closes."""
    s = pd.Series(np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0))
    return s.shift(1).rolling(MEMORY, min_periods=1).max().fillna(0.0).values


def _combine(kind: str, ma, a20, rma, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "r1":
        return np.maximum(ma, f * a20)
    if kind == "r2":
        return np.maximum(ma, f * rma)
    if kind == "x1":
        return ma * (1.0 - f * a20)
    if kind == "g1":
        return np.maximum(ma, f)
    if kind == "g2":
        return ma * (1.0 - f)
    raise ValueError(kind)


# ---------- Test A ----------

def _forward(close: pd.Series) -> dict:
    c = close.values.astype(float)
    n = len(c)
    f = pi._forward(close)
    out = {30: f[30], "deep": f["deep"]}
    g = np.full(n, np.nan)
    g[:n - 60] = c[60:] / c[:n - 60] - 1
    out[60] = g
    return out


def _boot_idx(n: int, rng) -> np.ndarray:
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (BOOT, nb))
    return ((st[:, :, None] + np.arange(BLOCK)) % n).reshape(BOOT, -1)[:, :n]


def _group_stats(grp: np.ndarray, fwd: dict, idx: np.ndarray) -> dict:
    """grp 0/1/2 (-1 = left out); per group mean / median with bootstrap intervals; R - D on 30-day means."""
    gb = grp[idx]
    out = {"n": [int((grp == k).sum()) for k in range(3)]}
    bm30 = {}
    for h in (30, 60, "deep"):
        v = fwd[h]
        vb = v[idx]
        for k in range(3):
            m = (grp == k) & np.isfinite(v)
            mean = float(v[m].mean()) if m.any() else None
            med = float(np.median(v[m])) if m.any() else None
            mb = (gb == k) & np.isfinite(vb)
            cnt = mb.sum(axis=1)
            bm = np.where(cnt > 0, np.where(mb, vb, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            with np.errstate(all="ignore"):
                bmed = np.nanmedian(np.where(mb, vb, np.nan), axis=1)
            if h == 30:
                bm30[k] = bm
            out[(h, k)] = {"mean": mean, "med": med, "mean_ci": pi._q(bm), "med_ci": pi._q(bmed)}
    a, b = out[(30, 0)]["mean"], out[(30, 1)]["mean"]
    out["rd"] = (a - b) if _ok(a) and _ok(b) else None
    out["rd_ci"] = pi._q(bm30[0] - bm30[1])
    return out


def _test_a(close: pd.Series, fl: pd.DataFrame) -> dict:
    fwd_all = _forward(close)
    past = (close / close.shift(30) - 1).values
    idx = close.index
    keep = fl["defined"].values & np.isfinite(fwd_all[30])
    start, end = idx[keep][0], idx[keep][-1]
    a20 = fl["a20"].values > 0.5
    grp_all = np.where(~a20, -1, np.where(fl["rel"].values > 0.5, 0, np.where(fl["dist"].values > 0.5, 1, 2)))
    rng = np.random.default_rng(SEED + 701)
    res = {"start": start, "end": end, "per": {}}
    spans = [("全期", start, end)] + [(lab, max(pd.Timestamp(a), start), min(pd.Timestamp(b), end))
                                     for lab, a, b in PERIODS]
    for lab, a, b in spans:
        m = keep & (idx >= a) & (idx <= b)
        if m.sum() < 2 * BLOCK:
            continue
        n = int(m.sum())
        bidx = _boot_idx(n, rng)
        fwd = {h: fwd_all[h][m] for h in fwd_all}
        res["per"][lab] = _group_stats(grp_all[m], fwd, bidx)
        res["per"][lab]["span"] = (idx[m][0], idx[m][-1])
        res["per"][lab]["days"] = n
        if lab == "全期":
            sub = a20[m] & np.isfinite(past[m])
            pv = past[m][sub]
            edges = np.quantile(pv, np.linspace(0, 1, 11))
            dec = np.full(n, -1)
            dec[sub] = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, 9)
            sc = (fl["rel"].values[m] > 0.5).astype(float)
            out = {}
            for key in (30, "deep"):
                v = np.nan_to_num(fwd[key], nan=0.0)
                real = float(pi._matched(v, sc, dec, sub)[0])
                boot = pi._matched(v[bidx], sc[bidx], dec[bidx], sub[bidx])
                out[key] = (real, pi._q(boot))
            res["matched"] = out
            # how much the trailing 30-day return differs between the groups (descriptive)
            res["past"] = [float(np.nanmean(past[m][grp_all[m] == k])) for k in range(3)]
    full = res["per"]["全期"]
    means = [full[(30, k)]["mean"] for k in range(3)]
    order = all(_ok(v) for v in means) and means[0] > means[2] > means[1]
    subs = [lab for lab, _, _ in PERIODS if lab in res["per"]]
    rd_subs = sum(1 for lab in subs if _ok(res["per"][lab]["rd"]) and res["per"][lab]["rd"] > 0)
    lo, hi = full["rd_ci"]
    excl = _ok(lo) and _ok(hi) and (lo > 0 or hi < 0)
    mpos = _ok(res["matched"][30][0]) and res["matched"][30][0] > 0
    res["pass"] = {"order": bool(order), "rd_subs": rd_subs, "n_subs": len(subs), "excl": bool(excl),
                   "matched": bool(mpos), "pass": bool(order and rd_subs >= 2 and excl and mpos)}
    return res


# ---------- Test D (owner's definition) ----------

def _test_d(close: pd.Series, fl: pd.DataFrame) -> dict:
    """Six cells: side (0 UP, 1 DOWN) x flow (0 QUIET, 1 MIDDLE, 2 HEAVY); cell = 3 * side + flow."""
    fwd_all = _forward(close)
    past = (close / close.shift(30) - 1).values
    idx = close.index
    keep = fl["defined"].values & np.isfinite(fwd_all[30])
    start, end = idx[keep][0], idx[keep][-1]
    side = np.where(fl["a20"].values > 0.5, 0, 1)
    flow = np.where(fl["quiet"].values > 0.5, 0, np.where(fl["high"].values > 0.5, 2, 1))
    cell_all = 3 * side + flow
    rng = np.random.default_rng(SEED + 801)
    res = {"start": start, "end": end, "per": {}}
    spans = [("全期", start, end)] + [(lab, max(pd.Timestamp(a), start), min(pd.Timestamp(b), end))
                                     for lab, a, b in PERIODS]
    for lab, a, b in spans:
        m = keep & (idx >= a) & (idx <= b)
        if m.sum() < 2 * BLOCK:
            continue
        n = int(m.sum())
        bidx = _boot_idx(n, rng)
        cell = cell_all[m]
        cb = cell[bidx]
        fwd = {h: fwd_all[h][m] for h in fwd_all}
        out = {"n": [int((cell == k).sum()) for k in range(6)], "days": n, "span": (idx[m][0], idx[m][-1])}
        bm30 = {}
        for h in (30, 60, "deep"):
            v = fwd[h]
            vb = v[bidx]
            for k in range(6):
                mk = (cell == k) & np.isfinite(v)
                mb = (cb == k) & np.isfinite(vb)
                cnt = mb.sum(axis=1)
                bm = np.where(cnt > 0, np.where(mb, vb, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
                with np.errstate(all="ignore"):
                    bmed = np.nanmedian(np.where(mb, vb, np.nan), axis=1)
                if h == 30:
                    bm30[k] = bm
                out[(h, k)] = {"mean": float(v[mk].mean()) if mk.any() else None,
                               "med": float(np.median(v[mk])) if mk.any() else None,
                               "mean_ci": pi._q(bm), "med_ci": pi._q(bmed)}
        diffs = []
        for sd_ in (0, 1):
            q, hv = out[(30, 3 * sd_)]["mean"], out[(30, 3 * sd_ + 2)]["mean"]
            out[f"qh{sd_}"] = (q - hv) if _ok(q) and _ok(hv) else None
            out[f"qh{sd_}_ci"] = pi._q(bm30[3 * sd_] - bm30[3 * sd_ + 2])
            diffs.append(out[f"qh{sd_}"])
        out["pool"] = (diffs[0] + diffs[1]) / 2 if all(_ok(d_) for d_ in diffs) else None
        out["pool_ci"] = pi._q((bm30[0] - bm30[2] + bm30[3] - bm30[5]) / 2)
        if lab == "全期":
            res["matched"] = {}
            for sd_ in (0, 1):
                sub = (side[m] == sd_) & np.isfinite(past[m]) & np.isin(flow[m], (0, 2))
                pv = past[m][(side[m] == sd_) & np.isfinite(past[m])]
                edges = np.quantile(pv, np.linspace(0, 1, 11))
                dec = np.full(n, -1)
                dec[sub] = np.clip(np.searchsorted(edges, past[m][sub], side="right") - 1, 0, 9)
                sc = (flow[m] == 0).astype(float)
                got = {}
                for key in (30, "deep"):
                    v = np.nan_to_num(fwd[key], nan=0.0)
                    got[key] = (float(pi._matched(v, sc, dec, sub)[0]),
                                pi._q(pi._matched(v[bidx], sc[bidx], dec[bidx], sub[bidx])))
                res["matched"][sd_] = got
        res["per"][lab] = out
    full = res["per"]["全期"]
    both = _ok(full["qh0"]) and _ok(full["qh1"]) and full["qh0"] > 0 and full["qh1"] > 0
    lo, hi = full["pool_ci"]
    excl = _ok(lo) and _ok(hi) and (lo > 0 or hi < 0)
    subs = [lab for lab, _, _ in PERIODS if lab in res["per"]]
    pos_subs = sum(1 for lab in subs if _ok(res["per"][lab]["pool"]) and res["per"][lab]["pool"] > 0)
    means = [full[(30, k)]["mean"] for k in range(6)]
    best = int(np.nanargmax([v if _ok(v) else -np.inf for v in means]))
    res["pass"] = {"both": bool(both), "excl": bool(excl), "pos_subs": pos_subs, "n_subs": len(subs),
                   "pass": bool(both and excl and pos_subs >= 2), "best": best}
    return res


def _cal_years(close: pd.Series, pos: np.ndarray) -> dict:
    change = close.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    s = pd.Series(net, index=close.index)
    return {int(y): float(np.prod(1 + g.values) - 1) for y, g in s.groupby(s.index.year)}


# ---------- the fair exam ----------

class _Exam(pi._Exam):
    """price_impact's exam (main_signal_check's four checks on chosen starts) with three ways to combine the flag."""

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first)
        self.rma = _recent(self.ma)
        self.crma = [_recent(x) for x in self.cma]
        for o in self.other.values():
            o["rma"] = _recent(o["ma"])

    def btc_pos(self, flag, kind: str = "ma50") -> np.ndarray:
        return _combine(kind, self.ma, self.a20, self.rma, self._at(flag, self.close.index, 0))

    def summary(self, flag, ki: int, sims: int = SIMS, kind: str = "ma50") -> dict:
        pos = self.btc_pos(flag, kind)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        f = self._at(flag, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = _combine(kind, self.cma[h], self.ca20[h], self.crma[h], f)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            w = _combine(kind, o["ma"], o["a20"], o["rma"], self._at(flag, o["close"].index, o["lag"]))
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_edge": float(np.median(dr - self.dhold[0])),
             "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_starts": n, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


# ---------- report-only pieces ----------

def _extra(close: pd.Series, ma: np.ndarray, pos: np.ndarray, first: pd.Timestamp) -> dict:
    idx = close.index
    ch = close.pct_change().values
    nxt = np.concatenate([ch[1:], [np.nan]])
    a = idx >= first
    mon = np.nan_to_num(ma) > 0.5
    ex = a & (pos > 0.5) & ~mon
    years = {}
    for y in sorted(set(idx[ex].year)):
        m = ex & (idx.year == y)
        years[int(y)] = (int(m.sum()), float(np.nanmean(nxt[m])))
    ok = ex & np.isfinite(nxt)
    allm = a & np.isfinite(nxt)
    return {"n": int(ex.sum()), "years": years, "avg": float(nxt[ok].mean()) if ok.any() else None,
            "avg_all": float(nxt[allm].mean()), "avg_ma_on": float(nxt[allm & mon].mean())}


def _entries(close: pd.Series, ma: np.ndarray, pos: np.ndarray, first: pd.Timestamp) -> list[dict]:
    c, idx = close.values, close.index
    ma = np.nan_to_num(ma)
    n = len(c)
    out = []
    for t in range(max(int(idx.searchsorted(first)), 1), n):
        if not (pos[t] > 0.5 and pos[t - 1] < 0.5 and ma[t] < 0.5):
            continue
        u = next((k for k in range(t + 1, n) if ma[k] > 0.5), None)
        right = u is not None and u - t <= RIGHT_DAYS and c[u] > c[t]
        e = next((k for k in range(t + 1, n) if pos[k] < 0.5 or ma[k] > 0.5), None)
        if e is None:
            ret, done = c[-1] / c[t] - 1 - COST, False
        else:
            ret, done = c[e] / c[t] - 1 - COST * (2 if pos[e] < 0.5 else 1), True
        known = (u is not None and u - t <= RIGHT_DAYS) or n - 1 - t > RIGHT_DAYS
        out.append({"day": idx[t], "right": bool(right), "ret": float(ret), "done": done, "known": known})
    return out


def _exits(close: pd.Series, ma: np.ndarray, pos: np.ndarray, first: pd.Timestamp) -> list[dict]:
    """Every ma50 exit: whipsaw / real decline / unknown, whether R2 sat through, and R2's advantage."""
    c, idx = close.values, close.index
    ma = np.nan_to_num(ma)
    n = len(c)
    out = []
    for t in range(max(int(idx.searchsorted(first)), 1), n):
        if not (ma[t - 1] > 0.5 and ma[t] < 0.5):
            continue
        u = next((k for k in range(t + 1, min(n, t + WHIP_DAYS + 1)) if ma[k] > 0.5), None)
        if u is not None:
            kind = "whip" if c[u] > c[t] else "real"
        else:
            kind = "real" if n - 1 - t >= WHIP_DAYS else "unknown"
        sat = pos[t] > 0.5
        adv = days = None
        done = True
        if sat:
            e = next((k for k in range(t + 1, n) if pos[k] < 0.5 or ma[k] > 0.5), None)
            if e is None:
                adv, days, done = c[-1] / c[t] - 1, n - 1 - t, False
            else:
                adv = c[e] / c[t] - 1 + (2 * COST if ma[e] > 0.5 else 0.0)
                days = e - t
        out.append({"day": idx[t], "kind": kind, "sat": bool(sat), "adv": adv, "days": days, "done": done})
    return out


def _rally_rows(close: pd.Series, fl: pd.DataFrame) -> tuple[list, list]:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    rel, dist, dfd = fl["rel"].values, fl["dist"].values, fl["defined"].values
    n = len(close)
    rows, pool = [], {seg: [0, 0, 0] for seg in ("tm", "mp", "after")}

    def share(a: int, b: int, seg: str) -> str:
        b = min(b, n)
        m = dfd[a:b]
        if b <= a or not m.any():
            return "—"
        k = int(m.sum())
        r_, d_ = int(rel[a:b][m].sum()), int(dist[a:b][m].sum())
        pool[seg][0] += k
        pool[seg][1] += r_
        pool[seg][2] += d_
        return f"{r_ / k * 100:.0f}%｜{d_ / k * 100:.0f}%（{k} 天）"

    for i, r in enumerate(rallies, 1):
        it, im, ip = r["it"], r["im"], r["ip"]
        after = share(ip + 1, ip + 1 + AFTER_PEAK, "after") if r["done"] else "還沒結束"
        rows.append([f"{i}（{_d(r['t'])} → {_d(r['p'])}）" + ("" if r["done"] else "，還沒結束"),
                     f"{r['gain'] * 100:+.0f}%", _d(r["m"]), share(it, im, "tm"), share(im, ip + 1, "mp"), after])
    tot = []
    for seg in ("tm", "mp", "after"):
        k, r_, d_ = pool[seg]
        tot.append(f"{r_ / k * 100:.0f}%｜{d_ / k * 100:.0f}%（{k} 天）" if k else "—")
    rows.append(["全部合計", "", ""] + tot)
    return rows, pool


def _leadlag(raw: dict, close: pd.Series) -> dict:
    in30 = raw["in30"].where(raw["in30"] > 0)
    x_data = np.log(in30 / in30.shift(14))
    x = x_data.reindex(close.index - DAY).values
    lc = np.log(close.values.astype(float))
    r14 = np.full(len(lc), np.nan)
    r14[14:] = lc[14:] - lc[:-14]
    first = int(np.argmax(np.isfinite(x)))
    f, r = x[first:], r14[first:]
    n = len(f)

    def corr(fv, k):
        if k >= 0:
            a, b = fv[:n - k], r[k:]
        else:
            a, b = fv[-k:], r[:n + k]
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    real = {k: corr(f, k) for k in LL_K}
    rng = np.random.default_rng(SEED + 79)
    pool = []
    for s in rng.integers(GAP, n - GAP + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL_SHOW]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n, "from": close.index[first]}


def _lookahead(data: dict, fl: pd.DataFrame, pos: dict) -> tuple[int, int]:
    total = bad = 0
    close = data["close"]
    for cut in CUTS:
        c = pd.Timestamp(cut)
        cm = data["cm"][data["cm"].index <= c - DAY]
        cl = close[close.index <= c]
        fl2 = _flags(cl, _legs(_raw(cm), cl.index))
        cols = ["rel", "dist", "lowprof", "high", "quiet"]
        total += 1
        bad += int(not np.array_equal(fl2[cols].values, fl.loc[cl.index, cols].values))
        ma2 = msc._targets(cl)["ma50"].values.astype(float)
        for kind in ("r1", "r2", "g1"):
            f2 = fl2["quiet"].values if kind == "g1" else fl2["lowprof"].values
            p2 = _combine(kind, ma2, fl2["a20"].values, _recent(ma2), f2)
            total += 1
            bad += int(not np.array_equal(p2, pos[kind][:len(cl)]))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    raw = _raw(data["cm"])
    lg = _legs(raw, close.index)
    fl = _flags(close, lg)
    ma = msc._targets(close)["ma50"].values.astype(float)
    a = _test_a(close, fl)
    dd = _test_d(close, fl)
    fstart = close.index[int(np.argmax(fl["defined"].values))]

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(close, data["other"], price, b0, pre, pd.Timestamp(FIRST_MONTH))
    lowprof = pd.Series(fl["lowprof"].values, index=close.index)
    high = pd.Series(fl["high"].values, index=close.index)
    low_leg = lg["low"].fillna(0.0)
    prof = lg["prof"].fillna(0.0)
    quiet = pd.Series(fl["quiet"].values, index=close.index)
    flags = {"ma50": (None, "ma50"), "r1": (lowprof, "r1"), "r2": (lowprof, "r2"), "x1": (high, "x1"),
             "g1": (quiet, "g1"), "g2": (high, "g2")}
    s = {k: ex.summary(f, KI[k], SIMS, kind) for k, (f, kind) in flags.items()}
    t_exam = time.time() - t0

    small = {k: ex.summary(flags[k][0], KI[k], PL_SIMS, k) for k in ("r1", "r2", "x1", "g1", "g2")}
    lstart = close.index[int(np.argmax(lg["low"].notna().values))]
    n = int((close.index >= lstart).sum())
    prng = np.random.default_rng(SEED + 19)
    hits = {k: 0 for k in small}
    lows = {k: [] for k in small}
    for k in prng.integers(GAP, n - GAP, DRAWS):
        rl = sd._roll(low_leg, lstart, int(k)) * prof
        rh = sd._roll(lg["high"].fillna(0.0), lstart, int(k))
        rq = sd._roll(low_leg, lstart, int(k))
        for key, f in (("r1", rl), ("r2", rl), ("x1", rh), ("g1", rq), ("g2", rh)):
            x = ex.summary(f, KI[key], PL_SIMS, key)
            lows[key].append(x["low"])
            hits[key] += int(sd._score_ge(x, small[key]))
    placebo = {k: hits[k] / DRAWS for k in small}

    pos = {k: ex.btc_pos(f, kind) for k, (f, kind) in flags.items()}
    extra = {k: _extra(close, ma, pos[k], fstart) for k in ("r1", "r2", "g1")}
    cal = {k: _cal_years(close, pos[k]) for k in ("ma50", "g1", "g2")}
    m15 = close.index >= pd.Timestamp(FIRST_MONTH)
    tim = {k: float(np.nanmean(np.nan_to_num(pos[k][m15]))) for k in ("ma50", "g1", "g2", "r1", "r2")}
    entries = _entries(close, ma, pos["r1"], fstart)
    exits = _exits(close, ma, pos["r2"], fstart)
    m = close.index >= pd.Timestamp(FIRST_MONTH)
    differ = {k: int((pos[k][m] != np.nan_to_num(pos["ma50"][m])).sum()) for k in ("r1", "r2", "x1", "g1", "g2")}
    rally, rpool = _rally_rows(close, fl)
    ll = _leadlag(raw, close)
    look = _lookahead(data, fl, pos)
    return {"raw": raw, "lg": lg, "fl": fl, "ma": ma, "a": a, "dd": dd, "cal": cal, "tim": tim, "fstart": fstart, "ex": ex, "s": s, "small": small,
            "placebo": placebo, "placebo_med": {k: float(np.median(v)) for k, v in lows.items()}, "pos": pos,
            "extra": extra, "entries": entries, "exits": exits, "differ": differ, "rally": rally, "rpool": rpool,
            "ll": ll, "look": look, "problem": problem, "secs": time.time() - t0, "secs_exam": t_exam}


# ---------- today ----------

def _now(data: dict, r: dict) -> list[dict]:
    close, fl, raw, ex = data["close"], r["fl"], r["raw"], r["ex"]
    last = close.index[-1]
    word = {1: "持有", 0: "現金"}
    on = lambda v: "是" if v > 0.5 else "否"  # noqa: E731
    st = {k: int(r["pos"][k][-1] > 0.5) for k in ("ma50", "r1", "r2", "x1", "g1", "g2")}
    c = close.values.astype(float)
    m20, m50 = float(c[-20:].mean()), float(c[-50:].mean())
    day = last - DAY

    def get(s):
        v = s.get(day) if day in s.index else None
        return float(v) if _ok(v) else None
    in30, med, p80, mv = get(raw["in30"]), get(raw["med"]), get(raw["p80"]), get(raw["mv"])
    if fl["defined"].iloc[-1]:
        fw = 0 if fl["quiet"].iloc[-1] > 0.5 else 2 if fl["high"].iloc[-1] > 0.5 else 1
        cell_now = f"{SIDES[0 if fl['a20'].iloc[-1] > 0.5 else 1]}＋{FLOWS[fw]}"
    else:
        cell_now = "資料不足"
    out = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜R1｜R2｜G1",
            "value": f"{word[st['ma50']]}｜{word[st['r1']]}｜{word[st['r2']]}｜{word[st['g1']]}"},
           {"label": "X1｜G2（只描述的過濾規則）", "value": f"{word[st['x1']]}｜{word[st['g2']]}"},
           {"label": "測試 D：今天在六格的哪一格", "value": cell_now},
           {"label": "　流入少（惜售：很少人把幣送去交易所賣）", "value": on(fl["quiet"].iloc[-1])},
           {"label": "　流入多（出貨：大量把幣送去交易所賣）", "value": on(fl["high"].iloc[-1])},
           {"label": "較窄的舊版惜售（價格在漲而且有賺的那一種，測試 A–C）", "value": on(fl["rel"].iloc[-1])},
           {"label": "較窄的舊版出貨（價格在漲時大量送幣，測試 A–C）", "value": on(fl["dist"].iloc[-1])},
           {"label": "　收盤 vs 20 日均線、50 日均線", "value": f"{c[-1]:,.0f}｜{m20:,.0f}｜{m50:,.0f}"
                                                         f"（站上 20 日：{on(fl['a20'].iloc[-1])}）"},
           {"label": f"　近 30 天轉進交易所的比特幣（用 {_d(day)} 的資料）",
            "value": (f"{in30:,.0f} 顆；過去一年中位數 {med:,.0f} 顆、第 80 百分位 {p80:,.0f} 顆"
                      if _ok(in30) and _ok(med) and _ok(p80) else "資料不足")},
           {"label": f"　MVRV（市值 ÷ 大家買進成本；> 1 表示平均有賺；用 {_d(day)} 的資料）",
            "value": f"{mv:.2f}" if _ok(mv) else "資料不足"},
           {"label": "　R2 的「均線 20 天內還在」", "value": on(ex.rma[-1])},
           {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；R1、R2、G1 沒有通過考試前不會用"}]
    return out


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, a, ex, fl = data["close"], r["s"], r["a"], r["ex"], r["fl"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    raw = r["raw"]

    # ----- table 0: data and group shares -----
    dfd = fl["defined"].values
    a20 = fl["a20"].values > 0.5
    nd = int(dfd.sum())
    sh = lambda m: f"{int(m.sum()):,} 天（{m.sum() / nd * 100:.0f}%）"  # noqa: E731
    in_first = raw["in30"].first_valid_index()
    rows0 = [["Coin Metrics 轉進交易所（FlowInExNtv）第一天", _d(data["cm"]["FlowInExNtv"].first_valid_index())],
             ["30 天加總和過去一年門檻第一次算得出來（資料日）", _d(raw["low"].first_valid_index())],
             ["MVRV 第一天", _d(raw["mv"].first_valid_index())],
             ["第一個能拿來決定的收盤日", _d(r["fstart"])],
             ["從那天起的天數", f"{nd:,}"],
             ["站上 20 日均線而且「惜售」", sh(dfd & (fl["rel"].values > 0.5))],
             ["站上 20 日均線而且「出貨」", sh(dfd & (fl["dist"].values > 0.5))],
             ["站上 20 日均線、兩個都不是", sh(dfd & a20 & (fl["rel"].values < 0.5) & (fl["dist"].values < 0.5))],
             ["沒站上 20 日均線（不分組）", sh(dfd & ~a20)],
             ["30 天加總第一天（資料日）", _d(in_first) if in_first is not None else "—"]]

    # ----- Test A -----
    def a_rows() -> list:
        out = []
        for lab, x in a["per"].items():
            sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
            for k in range(3):
                c30, c60, cd = x[(30, k)], x[(60, k)], x[("deep", k)]
                out.append([sp if k == 0 else "", GROUPS[k], f"{x['n'][k]:,}",
                            f"{_p(c30['mean'])} {_ci(*c30['mean_ci'])}", f"{_p(c30['med'])} {_ci(*c30['med_ci'])}",
                            f"{_p(c60['mean'])} {_ci(*c60['mean_ci'])}", f"{_p(c60['med'])} {_ci(*c60['med_ci'])}",
                            f"{_p(cd['mean'])} {_ci(*cd['mean_ci'])}"])
        return out

    cols_a = ["期間", "組", "天數", "之後 30 天平均 ［90% 區間］", "之後 30 天中位數 ［90% 區間］",
              "之後 60 天平均 ［90% 區間］", "之後 60 天中位數 ［90% 區間］", "30 天內最深跌幅平均 ［90% 區間］"]
    rows_rd = [[lab, f"{_pt(x['rd'])} {_ci(*x['rd_ci'], f=_pt)}"] for lab, x in a["per"].items()]
    ap = a["pass"]
    full = a["per"]["全期"]
    mt30, mtd = a["matched"][30], a["matched"]["deep"]

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    means = [full[(30, k)]["mean"] for k in range(3)]
    rows_ap = [["全期之後 30 天平均：惜售 > 兩個都不是 > 出貨",
                f"{_p(means[0])}｜{_p(means[2])}｜{_p(means[1])}", yes(ap["order"])],
               ["三段裡至少兩段「惜售 > 出貨」", f"{ap['rd_subs']} 段（共 {ap['n_subs']} 段）", yes(ap["rd_subs"] >= 2)],
               ["全期「惜售 − 出貨」的 90% 區間不含零", f"{_pt(full['rd'])} {_ci(*full['rd_ci'], f=_pt)}", yes(ap["excl"])],
               ["配對過去 30 天漲幅後，惜售 − 不是惜售 > 0",
                f"{_pt(mt30[0])} {_ci(*mt30[1], f=_pt)}（最深跌幅差 {_pt(mtd[0])} {_ci(*mtd[1], f=_pt)}）",
                yes(ap["matched"])],
               ["四條都要過", "", yes(ap["pass"])]]

    # ----- Tests B / C -----
    base = s["ma50"]
    res = {}
    for k in ("r1", "r2"):
        x = s[k]
        res[k] = {"score": sd._score_ge(x, base), "loss": x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK,
                  "plc": r["placebo"][k] <= PLACEBO_BAR}
        res[k]["pass"] = all(res[k].values())
    rows_b = []
    for k in ("ma50", "r1", "r2", "x1"):
        x = s[k]
        ck = x["checks"]
        rows_b.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                       "—" if k == "ma50" else f"{r['differ'][k]:,} 天",
                       "—" if k == "ma50" else f"{_share(r['placebo'][k])}（真的 {_share1(r['small'][k]['low'])}，"
                                               f"錯位中位數 {_share1(r['placebo_med'][k])}）"])
    rows_pass = []
    for k in ("r1", "r2"):
        x = s[k]
        rows_pass += [[SHORT[k], "排名分數不輸 50 日均線", f"{_share1(x['low'])} vs {_share1(base['low'])}",
                       yes(res[k]["score"])],
                      ["", "最深虧損中位數最多比均線深 5 點", f"{_pc(x['c_deep_med'])} vs {_pc(base['c_deep_med'])}",
                       yes(res[k]["loss"])],
                      ["", "隨機錯位比例 ≤ 5%（第二次嘗試，比平常的 10% 嚴）", _share(r["placebo"][k]), yes(res[k]["plc"])],
                      ["", "三條都要過", "", yes(res[k]["pass"])]]
    x1 = s["x1"]

    # ----- report only -----
    rows_x = []
    for k in ("r1", "r2"):
        e = r["extra"][k]
        for y, (nn, v) in e["years"].items():
            rows_x.append([SHORT[k], str(y), f"{nn:,}", _p(v)])
        rows_x.append([SHORT[k], "合計", f"{e['n']:,}", _p(e["avg"])])
    ent = r["entries"]
    right = [x for x in ent if x["right"]]
    wrong = [x for x in ent if not x["right"] and x["known"]]
    unk = [x for x in ent if not x["right"] and not x["known"]]
    avg = lambda v: float(np.mean([x["ret"] for x in v])) if v else None  # noqa: E731
    comp = lambda v: float(np.prod([1 + x["ret"] for x in v]) - 1) if v else None  # noqa: E731
    rows_e = [["對（30 天內均線也上車，而且價格更高）", f"{len(right)}", _p(avg(right)), _p(comp(right))],
              ["錯（其他情況）", f"{len(wrong)}", _p(avg(wrong)), _p(comp(wrong))],
              ["還不知道（不到 30 天）", f"{len(unk)}", _p(avg(unk)), _p(comp(unk))],
              ["合計", f"{len(ent)}", _p(avg(ent)), _p(comp(ent))]]
    exits = r["exits"]
    rows_w = []
    for kind, lab in (("whip", "假跌破（20 天內均線又上車，價格更高）"), ("real", "真的下跌（其他情況）"),
                      ("unknown", "還不知道（不到 20 天）")):
        allk = [x for x in exits if x["kind"] == kind]
        satk = [x for x in allk if x["sat"]]
        advs = [x["adv"] for x in satk if _ok(x["adv"])]
        rows_w.append([lab, f"{len(allk)}", f"{len(satk)}", _p(float(np.mean(advs)) if advs else None),
                       _p(float(np.prod([1 + v for v in advs]) - 1) if advs else None)])
    real_sat = sorted([x for x in exits if x["kind"] == "real" and x["sat"]], key=lambda x: x["day"])
    rows_rl = [[_d(x["day"]), f"{x['days']} 天" + ("" if x["done"] else "（還沒結束）"), _p(x["adv"])]
               for x in real_sat] or [["—", "—", "—"]]
    whip_sat = [x for x in exits if x["kind"] == "whip" and x["sat"]]
    rows_wl = [[_d(x["day"]), f"{x['days']} 天", _p(x["adv"])] for x in sorted(whip_sat, key=lambda x: x["day"])]

    ll = r["ll"]
    band = ll["band"]
    rows_l = [[f"{k:+d} 天" + ("（價格先動）" if k < 0 else "（流入先動）" if k > 0 else "（同一天）"),
               f"{ll['corr'][k]:+.3f}", "是" if abs(ll["corr"][k]) > band else "否"] for k in LL_SHOW]
    neg = {k: v for k, v in ll["corr"].items() if k < 0}
    posk = {k: v for k, v in ll["corr"].items() if k > 0}
    kn = max(neg, key=lambda k: abs(neg[k]))
    kp = max(posk, key=lambda k: abs(posk[k]))
    rows_l.append(["價格先動那邊相關最大的", f"{kn:+d} 天：{neg[kn]:+.3f}", "是" if abs(neg[kn]) > band else "否"])
    rows_l.append(["流入先動那邊相關最大的", f"{kp:+d} 天：{posk[kp]:+.3f}", "是" if abs(posk[kp]) > band else "否"])

    # ----- Test D (owner's definition) -----
    dd = r["dd"]
    dp = dd["pass"]
    dfull = dd["per"]["全期"]
    cells = [f"{('價格在上', '價格在下')[k // 3]}＋{FLOWS[k % 3]}" for k in range(6)]

    def d_rows() -> list:
        out = []
        for lab, x in dd["per"].items():
            sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
            for k in range(6):
                c30, c60, cd = x[(30, k)], x[(60, k)], x[("deep", k)]
                out.append([sp if k == 0 else "", cells[k], f"{x['n'][k]:,}（{x['n'][k] / x['days'] * 100:.0f}%）",
                            f"{_p(c30['mean'])} {_ci(*c30['mean_ci'])}", f"{_p(c30['med'])} {_ci(*c30['med_ci'])}",
                            f"{_p(c60['mean'])} {_ci(*c60['mean_ci'])}", f"{_p(c60['med'])} {_ci(*c60['med_ci'])}",
                            f"{_p(cd['mean'])} {_ci(*cd['mean_ci'])}"])
        return out

    rows_dq = [[lab, f"{_pt(x['qh0'])} {_ci(*x['qh0_ci'], f=_pt)}", f"{_pt(x['qh1'])} {_ci(*x['qh1_ci'], f=_pt)}",
                f"{_pt(x['pool'])} {_ci(*x['pool_ci'], f=_pt)}"] for lab, x in dd["per"].items()]
    dm = dd["matched"]
    rows_dm = [[SIDES[k], f"{_pt(dm[k][30][0])} {_ci(*dm[k][30][1], f=_pt)}",
                f"{_pt(dm[k]['deep'][0])} {_ci(*dm[k]['deep'][1], f=_pt)}"] for k in (0, 1)]
    rows_dp = [["價格在上、在下兩邊，全期之後 30 天平均都是「流入少 > 流入多」",
                f"在上 {_pt(dfull['qh0'])}，在下 {_pt(dfull['qh1'])}", "成立" if dp["both"] else "不成立"],
               ["兩邊平均的差，90% 區間不含零", f"{_pt(dfull['pool'])} {_ci(*dfull['pool_ci'], f=_pt)}",
                "成立" if dp["excl"] else "不成立"],
               ["三段裡至少兩段兩邊平均的差是正的", f"{dp['pos_subs']} 段（共 {dp['n_subs']} 段）",
                "成立" if dp["pos_subs"] >= 2 else "不成立"],
               ["三條都成立＝想法得到支持", "", "支持" if dp["pass"] else "不支持"]]
    base_d = s["ma50"]
    g1 = s["g1"]
    gres = {"score": sd._score_ge(g1, base_d), "loss": g1["c_deep_med"] >= base_d["c_deep_med"] - LOSS_SLACK,
            "plc": r["placebo"]["g1"] <= PLACEBO_BAR}
    gres["pass"] = all(gres.values())
    rows_g = []
    for k in ("ma50", "g1", "g2"):
        x = s[k]
        ck = x["checks"]
        rows_g.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", _share(r["tim"][k]),
                       "—" if k == "ma50" else f"{r['differ'][k]:,} 天",
                       "—" if k == "ma50" else f"{_share(r['placebo'][k])}（真的 {_share1(r['small'][k]['low'])}，"
                                               f"錯位中位數 {_share1(r['placebo_med'][k])}）"])
    rows_gp = [["排名分數不輸 50 日均線", f"{_share1(g1['low'])} vs {_share1(base_d['low'])}", yes(gres["score"])],
               ["最深虧損中位數最多比均線深 5 點", f"{_pc(g1['c_deep_med'])} vs {_pc(base_d['c_deep_med'])}",
                yes(gres["loss"])],
               ["隨機錯位比例 ≤ 5%", _share(r["placebo"]["g1"]), yes(gres["plc"])],
               ["三條都要過", "", yes(gres["pass"])]]
    cal = r["cal"]
    rows_gy = []
    for y in (2018, 2022):
        rows_gy.append([str(y), _p(cal["ma50"].get(y)), _p(cal["g1"].get(y)),
                        _pt(cal["g1"].get(y, 0) - cal["ma50"].get(y, 0)),
                        _p(cal["g2"].get(y)), _pt(cal["g2"].get(y, 0) - cal["ma50"].get(y, 0))])
    eg = r["extra"]["g1"]
    rows_gx = [[str(y), f"{nn:,}", _p(v)] for y, (nn, v) in eg["years"].items()] + \
        [["合計", f"{eg['n']:,}", _p(eg["avg"])]]
    best = dp["best"]
    wash = best == 5
    d_tables = [
        {"caption": ("測試 D（老闆的定義，第三種定義，2026-11-19 在測試 A–C 定好並跑完之後才加）：只看賣的行為，不管價格漲跌、不管有沒有賺。"
                     "惜售＝很少人把幣送去交易所賣（不管價格漲跌、不管有沒有賺）：近 30 天轉進交易所的比特幣少於過去一年中位數。"
                     "出貨＝大量把幣送去交易所賣（可能是趁漲賣，也可能是跌到受不了賣）：高於過去一年第 80 百分位。其他算中間。"
                     "價格只用來分邊：站上 20 日均線算「在上」，否則「在下」。六格的天數、占比、之後 30／60 天漲跌和 30 天內最深跌幅；"
                     f"90% 區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次。"),
         "columns": ["期間", "格子", "天數（占比）", "之後 30 天平均 ［90% 區間］", "之後 30 天中位數 ［90% 區間］",
                     "之後 60 天平均 ［90% 區間］", "之後 60 天中位數 ［90% 區間］", "30 天內最深跌幅平均 ［90% 區間］"],
         "rows": d_rows()},
        {"caption": "測試 D：「流入少 − 流入多」之後 30 天平均漲跌的差（點），價格在上、在下分開算，以及兩邊的平均。正的＝惜售之後比較好。",
         "columns": ["期間", "價格在上：少 − 多 ［90% 區間］", "價格在下：少 − 多 ［90% 區間］", "兩邊平均 ［90% 區間］"],
         "rows": rows_dq},
        {"caption": ("測試 D 的配對（只描述）：同一邊裡，依「過去 30 天漲多少」分十組，在同一組裡比流入少和流入多的日子，"
                     "避免「其實只是價格已經漲過或跌過了」。"),
         "columns": ["價格那一邊", "流入少 − 流入多：之後 30 天 ［90% 區間］", "流入少 − 流入多：30 天內最深跌幅 ［90% 區間］"],
         "rows": rows_dm},
        {"caption": "測試 D 的判讀（事先定好）。", "columns": ["條件", "數字", "結果"], "rows": rows_dp},
        {"caption": (f"規則 G1（只試這一條）：50 日均線在上，或流入少（不管價格怎樣）就持有。公平考試和 R1、R2 完全一樣（同樣的起算日，"
                     f"隨機錯位把「流入少」往後挪 {GAP} 天以上、{DRAWS} 次，門檻 5%）。G2＝均線在上但流入多時不持有（只描述）。"
                     f"「在場時間」＝{FIRST_MONTH[:7]} 起比特幣日線持有的日子占幾成。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "在場時間", "和均線不同的天數", "隨機錯位比例"],
         "rows": rows_g},
        {"caption": "G1 的通過條件（事先定好）。", "columns": ["條件", "數字", "結果"], "rows": rows_gp},
        {"caption": f"只描述：G1、G2 在 2018、2022 兩個熊市年的整年報酬（扣 {COST * 100:.2f}% 成本），和 50 日均線比。",
         "columns": ["年", "50 日均線", "G1", "G1 減均線", "G2", "G2 減均線"], "rows": rows_gy},
        {"caption": f"只描述：G1 比 50 日均線多持有的日子（{_d(r['fstart'])} 起），以及那些日子隔天的平均漲跌。",
         "columns": ["年", "多持有的天數", "隔天平均漲跌"], "rows": rows_gx},
    ]

    tables = d_tables + [
        {"caption": ("資料從哪天開始，以及測試 A–C 每組的日子占幾成。Coin Metrics 某一天的數字，要等到隔一天的日線收盤才拿來用（晚一天，保守）。"
                     "測試 A–C 用的是較窄的舊版（價格在漲而且有賺的那一種）：惜售＝有賺卻不賣：站上 20 日均線、MVRV > 1（平均有賺）、近 30 天轉進交易所的比特幣少於過去一年中位數。"
                     "出貨＝價格一漲就大量送幣去交易所賣：站上 20 日均線、近 30 天轉進交易所的比特幣高於過去一年第 80 百分位。"),
         "columns": ["項目", "數字"], "rows": rows0},
        {"caption": (f"測試 A（較窄的舊版：價格在漲而且有賺的那一種）：這個想法對不對？只看站上 20 日均線的日子，分成三組，看之後 30、60 天比特幣漲跌和 30 天內最深跌幅。"
                     f"90% 區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次（日子彼此相關，所以整塊抽）。"),
         "columns": cols_a, "rows": a_rows()},
        {"caption": "測試 A：「惜售」減「出貨」之後 30 天平均漲跌的差（點）。正的表示惜售之後比較好。",
         "columns": ["期間", "惜售 − 出貨 ［90% 區間］"], "rows": rows_rd},
        {"caption": (f"測試 A 的通過條件（事先定好，四條都要過）。配對：把站上 20 日均線的日子依「過去 30 天漲多少」分成十組，"
                     f"在同一組裡比惜售和不是惜售的日子，避免「其實只是已經漲過了」。三組過去 30 天平均漲幅："
                     f"{'、'.join(f'{GROUPS[k]} {_p(a['past'][k])}' for k in range(3))}。"),
         "columns": ["條件", "數字", "結果"], "rows": rows_ap},
        {"caption": (f"測試 B、C（較窄的舊版定義）：公平考試（main_signal_check 的做法）。比特幣日線起算月 {FIRST_MONTH[:7]} 到 "
                     f"{msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格"
                     f"（{base['c_starts']} 個起算月），一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個收盤時間 {SIMS} 次），其他資產 {'、'.join(ex.other)}（{base['o_n']} 個起算月；各用自己的 50、20 日均線，"
                     "配比特幣的流入和 MVRV 當全市場訊號）。50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"
                     f"「和均線不同的天數」是 {FIRST_MONTH[:7]} 起比特幣日線上持有／現金不一樣的天數。隨機錯位："
                     f"只把流入那條（流入少／流入多）往後挪 {GAP} 天以上（頭尾相接），{DRAWS} 次，MVRV 和價格條件不動，"
                     f"整個考試重算（一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方法重算來比）。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "和均線不同的天數",
                     "隨機錯位比例"],
         "rows": rows_b},
        {"caption": ("R1、R2 的通過條件（事先定好）。隨機錯位的門檻是 5%，比平常的 10% 嚴，因為這是同一個想法第二次試，"
                     "而且定義是看到第一次失敗以後才改的。兩條規則都試了，只有一條過的話證據比較弱。"),
         "columns": ["規則", "條件", "數字", "結果"], "rows": rows_pass},
        {"caption": (f"只描述：R1、R2 比 50 日均線多持有的日子（{_d(r['fstart'])} 起，比特幣日線），以及那些日子隔天的平均漲跌。"
                     f"對照：同期間每天平均 {_p(r['extra']['r1']['avg_all'])}，均線持有的日子平均 {_p(r['extra']['r1']['avg_ma_on'])}。"),
         "columns": ["規則", "年", "多持有的天數", "隔天平均漲跌"], "rows": rows_x},
        {"caption": ("只描述：R1 提早上車（均線還在空手時 R1 轉成持有）對不對。「對」＝30 天內均線也上車，而且那時價格比提早上車時高。"
                     "每次的結果＝從提早上車到 R1 下車或均線上車（先到的那個）之間的漲跌，扣掉買賣成本。"),
         "columns": ["結果", "次數", "平均漲跌", "接起來（相乘）"], "rows": rows_e},
        {"caption": ("只描述：50 日均線每一次下車（前一天在均線上、今天跌破）。假跌破＝20 天內均線又上車，而且價格比下車時高；"
                     "其他的算真的下跌。「R2 坐過去」＝均線下車那天 R2 還繼續持有。R2 多賺＝從那天到 R2 下車或均線重新上車之間比特幣的漲跌"
                     f"（均線重新上車的話，再加上均線多付的一買一賣成本 {2 * COST * 100:.2f}%）；負的表示 R2 多賠。"),
         "columns": ["均線下車的種類", "次數", "R2 坐過去的次數", "R2 坐過去：平均多賺", "R2 坐過去：接起來（相乘）"],
         "rows": rows_w},
        {"caption": "只描述：R2 坐過去的「真的下跌」，每一次（包括 2018、2022 熊市開始那幾次，如果有的話）。",
         "columns": ["均線下車日", "R2 多抱了幾天", "R2 比均線多賺／多賠"], "rows": rows_rl},
        {"caption": "只描述：R2 坐過去的「假跌破」，每一次。",
         "columns": ["均線下車日", "R2 多抱了幾天", "R2 比均線多賺／多賠"], "rows": rows_wl or [["—", "—", "—"]]},
        {"caption": (f"只描述：rally_sources 的大漲段（從低點漲 {rs.RISE * 100:.0f}% 以上、跌回 {rs.FALL * 100:.0f}% 才算結束）裡，"
                     f"「惜售｜出貨」的日子各占幾成：低點到 50 日均線上車那天、上車到高點、高點之後 {AFTER_PEAK} 天。"
                     "如果想法對，前半段應該多惜售，高點附近和之後應該多出貨。"),
         "columns": ["大漲段", "漲幅", "均線上車日", "低點→均線上車：惜售｜出貨", "均線上車→高點：惜售｜出貨",
                     f"高點之後 {AFTER_PEAK} 天：惜售｜出貨"],
         "rows": r["rally"]},
        {"caption": (f"只描述：誰先動？「轉進交易所的比特幣 30 天加總」過去 14 天的變化，和比特幣 14 天漲跌錯開不同天數的相關"
                     f"（{_d(ll['from'])} 起，{ll['n']:,} 天）。負的天數＝價格先動、流入後動；正的天數＝流入先動、價格後動。"
                     f"雜訊帶＝把流入錯開 {GAP} 天以上、{DRAWS} 次得到的 |相關| 第 95 百分位 {band:.3f}；超過才算看得出來。"
                     "相隔不到 14 天的兩段時間有重疊，相關會被重疊拉高。"),
         "columns": ["錯開天數", "相關", "超過雜訊帶"], "rows": rows_l},
    ]

    # ----- verdict -----
    first = (f"較窄的舊版（價格在漲而且有賺的那一種）測試 A（{_d(a['start'])} 起，只看站上 20 日均線的日子）：之後 30 天平均 惜售 {_p(means[0])}（{full['n'][0]:,} 天）、"
             f"兩個都不是 {_p(means[2])}（{full['n'][2]:,} 天）、出貨 {_p(means[1])}（{full['n'][1]:,} 天）；"
             f"惜售減出貨 {_pt(full['rd'])} {_ci(*full['rd_ci'], f=_pt)}，三段裡 {ap['rd_subs']} 段惜售較好；"
             f"配對過去 30 天漲幅後 {_pt(mt30[0])}。事先定的條件{'通過' if ap['pass'] else '沒有通過'}"
             + ("" if ap["pass"] else "（" + "、".join(
                 ([] if ap["order"] else ["全期順序不對"]) + ([] if ap["rd_subs"] >= 2 else ["三段裡惜售較好的不到兩段"])
                 + ([] if ap["excl"] else ["差的區間含零"]) + ([] if ap["matched"] else ["配對後不是正的"])) + "）") + "。")

    def rule_line(k: str) -> str:
        x, q = s[k], res[k]
        fails = ([] if q["score"] else ["排名分數輸均線"]) + ([] if q["loss"] else ["最深虧損比均線深超過 5 點"]) \
            + ([] if q["plc"] else ["隨機錯位超過 5%"])
        return (f"{SHORT[k]} 排名分數 {_share1(x['low'])}、最深虧損中位數 {_pc(x['c_deep_med'])}、隨機錯位 {_share(r['placebo'][k])}，"
                + ("三條都通過" if q["pass"] else f"沒有通過（{'、'.join(fails)}）"))

    passed = [k for k in ("r1", "r2") if res[k]["pass"]]
    second = (f"公平考試（均線排名分數 {_share1(base['low'])}、最深虧損中位數 {_pc(base['c_deep_med'])}）："
              f"{rule_line('r1')}；{rule_line('r2')}。"
              + ("兩條都通過，只能算是可以放進每日影子紀錄觀察的候選。" if len(passed) == 2 else
                 f"只有 {SHORT[passed[0]]} 通過；試了兩條只過一條，證據比較弱，頂多放進每日影子紀錄觀察。" if passed else ""))
    wh = rows_w[0]
    rl = rows_w[1]
    third = (f"只描述：X1（出貨時不持有）排名分數 {_share1(x1['low'])}、最深虧損中位數 {_pc(x1['c_deep_med'])}、隨機錯位 "
             f"{_share(r['placebo']['x1'])}。R1 提早上車 {len(ent)} 次，對 {len(right)} 次、錯 {len(wrong)} 次（錯的平均 {_p(avg(wrong))}）。"
             f"均線下車時 R2 坐過去：假跌破 {wh[2]} 次（平均 {wh[3]}），真的下跌 {rl[2]} 次（平均 {rl[3]}，接起來 {rl[4]}）。")
    dmeans = [dfull[(30, k)]["mean"] for k in range(6)]
    zero = (f"測試 D（老闆的定義：只看賣的行為，{_d(dd['start'])} 起）：之後 30 天平均，價格在上時 流入少 {_p(dmeans[0])}、"
            f"中間 {_p(dmeans[1])}、流入多 {_p(dmeans[2])}；價格在下時 流入少 {_p(dmeans[3])}、中間 {_p(dmeans[4])}、"
            f"流入多 {_p(dmeans[5])}。兩邊平均「少 − 多」{_pt(dfull['pool'])} {_ci(*dfull['pool_ci'], f=_pt)}，"
            f"三段裡 {dp['pos_subs']} 段是正的；配對後 在上 {_pt(dm[0][30][0])}、在下 {_pt(dm[1][30][0])}。"
            f"事先定的判讀：{'支持' if dp['pass'] else '不支持'}這個想法"
            + ("" if dp["pass"] else "（" + "、".join(
                ([] if dp["both"] else ["兩邊不是都「少 > 多」"]) + ([] if dp["excl"] else ["差的區間含零"])
                + ([] if dp["pos_subs"] >= 2 else ["三段裡正的不到兩段"])) + "）") + "。"
            + ("觀察（不算結果）：六格裡之後 30 天最好的是「價格在下＋流入多」（跌到受不了大量賣之後），比較像「洗盤後反彈」，是另一個想法。"
               if wash else f"六格裡之後 30 天平均最好的是「{cells[best]}」。")
            + f"G1（均線或流入少就持有）排名分數 {_share1(g1['low'])}（均線 {_share1(base_d['low'])}）、最深虧損中位數 "
              f"{_pc(g1['c_deep_med'])}、隨機錯位 {_share(r['placebo']['g1'])}，"
            + ("三條都通過，只能算是放進每日影子紀錄觀察的候選（這是第三種定義，證據要打折）。" if gres["pass"] else
               "沒有通過（" + "、".join(([] if gres["score"] else ["排名分數輸均線"])
                                    + ([] if gres["loss"] else ["最深虧損比均線深超過 5 點"])
                                    + ([] if gres["plc"] else ["隨機錯位超過 5%"])) + "）。")
            + f"G2（流入多時不持有，只描述）排名分數 {_share1(s['g2']['low'])}。")
    verdict = zero + first + second + third

    lk_total, lk_bad = r["look"]
    notes = [
        "為什麼有這份研究：supply_demand 測過「賣的人少＋買盤進來」沒過。老闆指出它對「賣的人少」的定義錯了：它用的是"
        "「持有人帳面獲利低（MVRV 低）＋轉進交易所的幣少」，這描述的是熊市裡被套住、賣了會賠的人，不是「不想賣」的人。"
        "改正後的想法：價格在漲、持有人有賺可以賣，卻沒有把幣送去交易所，才是真的賣壓少（惜售）；價格一漲就有大量幣被送去交易所，是賣壓大（出貨）。",
        "這個想法現在試了三種定義：supply_demand（賺得少＋流入少）、測試 A–C（價格在漲＋有賺＋流入少）、測試 D（只看流入，老闆的定義）。"
        "同一個想法換定義試越多次，碰巧有一次過的機會越大，所以任何一次「通過」的證據都要再打折；測試 D 和 G1 是看到前兩次失敗後才定的。",
        "這是同一個想法第二次試，而且定義是看到第一次失敗以後才改的，所以隨機錯位的門檻從 10% 提高到 5%。兩條規則（R1、R2）都試了，"
        "只有一條過的話證據比較弱。",
        "名詞：MVRV＝比特幣市值 ÷ 所有幣上次移動時的價值加總，大約是「現價 ÷ 大家平均買進成本」，> 1 表示平均有賺。"
        "轉進交易所＝Coin Metrics 認得出是交易所的地址收到的比特幣。第 80 百分位＝過去一年裡只有兩成的日子比它高。",
        "門檻（30 天、365 天、中位數、第 80 百分位、MVRV 1、20 日均線、20 天記憶、60 天的區塊）都是跑之前憑道理定的，沒有試過其他數字。",
        "Coin Metrics 認交易所地址用的是今天的標籤套回過去，而且越來越完整，所以越早的流入被低估（和今天的版本比）。"
        "和「過去一年」自己比的百分位能吸收一部分，但不能全部；2012–16 認得出的交易所比較少，那一段要打折看。"
        "這份歷史在 2026 年 4 月整段重算過（flow_filters 發現），當時的人看到的數字可能不一樣。",
        "其他幣和 MSTR 沒有自己的流入資料，用的是比特幣的流入和 MVRV（當成全市場的狀態），配各自的價格均線；MSTR 用前一天的訊號。",
        f"偷看未來的檢查：把資料在 {'、'.join(CUTS)} 截斷（Coin Metrics 再早一天）重算，{lk_total} 次比對裡有 {lk_bad} 次不一致"
        f"{'（應該是 0）' if lk_bad else '，截斷前的訊號和 R1、R2 持有完全一樣'}。",
        "就算通過，也只代表規則可以放進每日影子紀錄觀察；用的是同一段歷史，真正的考驗是之後每天實際記下的結果。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載；考試本身 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    notes += problems

    return {
        "id": "reluctant_sellers",
        "title": "惜售與出貨：有賺卻不賣，價格會不會繼續漲",
        "question": ("價格在漲、持有人有賺可以賣，卻很少把比特幣送去交易所（惜售），之後是不是比較會繼續漲？"
                     "價格一漲就大量送幣去交易所（出貨），之後是不是比較會停？能不能用來比 50 日均線早上車，或在跌破均線時不被洗出場？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群 API：比特幣 FlowInExNtv（轉進交易所）、CapMVRVCur（MVRV）",
             "url": CM_URL + "?assets=btc&metrics=FlowInExNtv,CapMVRVCur&frequency=1d"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
