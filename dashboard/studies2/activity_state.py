"""Historical study "activity_state": is an ACTIVE bitcoin market (many coins changing hands between holders) followed
by better weeks than a QUIET one, beyond what recent price change already says - and does holding during active states
add return to ma50 (hold while the daily close is above its 50-day average)?

Why this study exists (stated plainly; this hypothesis was formed AFTER seeing the numbers below, so the test is built
to be able to fail). Three project studies each expected that HEAVY activity on the selling side would be followed by
WORSE returns, and each measured the opposite (observed after the fact, not expected in advance):
  reluctant_sellers: price above its 20-day average, 30-day forward mean +25.5% after heavy BTC deposits to exchanges
    versus +4.6% after quiet; below the 20-day average +6.4% versus +2.3%.
  old_coin_selling: 60-day forward mean after heavy movement of coins held 155+ days versus quiet: +41.8% vs +17.1%
    (price above its 20-day average), +14.9% vs +9.1% (below).
  force_balance: the BUY-side force had a +11.6 point effect on 30-day forward returns [+3.2, +16.3]; BUY and SELL
    ranks correlate 0.96.
  The coordinator's follow-up (2012-01 .. 2026-09, 60-day forward mean, old coins QUIET | HEAVY within each fifth of
    the past-30-day return): Q1 +6.3 | +11.5; Q2 +5.0 | +16.0; Q3 +13.0 | +25.7; Q4 +22.6 | +28.2; Q5 +29.7 | +48.5;
    among days with a positive past-30-day return: 2012-16 +45.0 | +53.4, 2017-21 +9.5 | +41.9, 2022-now +6.6 | +11.3.
    Reproduced before this docstring was written (the only thing run before it): Q1 +6.3 | +11.5, Q2 +5.0 | +16.0,
    Q3 +13.1 | +25.7, Q4 +22.7 | +28.2, Q5 +29.4 | +48.5; periods +45.0 | +53.4, +9.5 | +41.9, +6.6 | +11.3 (the
    0.1-0.3 point gaps are presumably the OBM download vintage). The study recomputes and shows it.
Hypothesis (stated as a hypothesis): an ACTIVE market state carries information about forward returns beyond what
recent price change carries, and holding during active states adds return to ma50. A signal measured in the opposite
direction from the earlier guess is still a signal: it is neither discarded nor explained away here; it is tested.
What order_flow already tested with Binance volume (2018+): taker-buy share (1/7/30 days), cumulative volume delta,
today's volume relative to its previous 30-day average, volume spikes, OBV, spot-vs-futures taker share; none became a
usable filter (force_balance's summary: no usable forward relation). It did NOT test the 30-day volume LEVEL ranked
against its own trailing year as a slow "state" - that is M4 here.

Data (free, key-less, fixed user agent "btc-research-dashboard"; a refused endpoint is dropped and named, never worked
around):
  Open Bitcoin Metrics (OBM, CC BY 4.0): the copy old_coin_selling stored (cache/_store/old_coin_selling_obm.json,
    parsed numbers only; DATA FILES ONLY - that repository's code is never cloned, installed or run); series
    spent >= 155 days (ge155) and spent < 155 days (lt155), BTC per day. Used with a 7-day lag (old_coin_selling's).
  Coin Metrics community API: btc FlowInExNtv (BTC sent to exchange-tagged addresses per day) and SplyCur (current
    supply); eth FlowInExNtv (checked before writing this: HTTP 200, free). Own stores
    cache/_store/activity_state_cm_btc.json / _eth.json (fallback only). Used with a 1-day lag.
  Binance spot daily klines BTCUSDT and ETHUSDT (quote volume in USDT, base volume), through order_flow's loader and
    store. A daily candle's volume is known at its own close, so lag 0.
  BTC daily closes: te._daily_close (00:00 UTC; blockchain.info before Binance 2017-08). ETH: Binance ETHUSDT closes
    (main_signal_check's loader). Fair exam inputs: Binance 1-hour BTCUSDT grid; ETH SOL BNB XRP (Binance), MSTR (Yahoo).

Measures (fixed now; on data dates D; each a 30-day sum D-29 .. D with all 30 values required):
  M1 old coins moving: old_coin_selling's OLDSELL (ge155 winsorised at its trailing-365-day 99th percentile, summed).
     The series the observation came from (IN-SAMPLE). Lag 7.
  M2 BTC sent to exchanges: FlowInExNtv summed. In-sample for the first observation. Lag 1.
  M3 total on-chain BTC spent: ge155 + lt155 summed (not winsorised). Never looked at. Lag 7.
  M4 Binance spot BTCUSDT quote volume (USD) summed, 2017-08+. Never looked at. Lag 0.
  M5 velocity: M3 divided by SplyCur on day D. Never looked at. Lag 7. (Note: M3 is already counted in coins, so it
     does not rise just because price rose; M5 removes supply growth, which matters mostly before 2016. The "more
     dollars because price is higher" worry applies to M4, so a report-only line M4b = Binance BTC BASE volume (in
     coins) is shown next to M4; it is not part of any pre-registered reading.)
  RANK(D) = share of the 365 previous values D-365 .. D-1 (>= 330 non-missing) below the value (old_coin_selling's
  _trail_rank). HEAVY = RANK > 0.80, QUIET = RANK < 0.50, MIDDLE otherwise. Decision day t (UTC close t, held t+1)
  uses data day t - lag; undefined -> no level (and OFF / unchanged in rules).

Part 1 - is there information beyond price? Sample per measure: decision days from 2012-01-01 (or the measure's
  first defined day) with a level, a full 60-day forward window, past-30 and past-90-day returns and 20-day volatility.
  Forward return close[t+h]/close[t]-1, h = 30, 60, 90 (90 skips days without a window).
  Descriptive tables, HEAVY versus QUIET: mean (30/60/90), median (60), share of days followed by a rise (60), within
  each fifth of the past-30-day return, within each fifth of the past-90-day return (fifth edges from the measure's
  own sample) and separately for ma50 on / off.
  Matched difference (main number): 30 cells = past-30-day-return decile x 20-day-volatility tercile (edges from the
  measure's sample; volatility = std of daily log returns over the 20 days ending at t); inside each cell HEAVY mean
  minus QUIET mean, weighted by HEAVY days (cells holding both kinds only); 90% interval from a circular block
  bootstrap of sample days (60-day blocks, 1,000 draws, fixed seed). For h = 30, 60, 90; full sample and each period
  2012-16 / 2017-21 / 2022-now (cell edges from the full sample, bootstrap inside the period). Calendar years: raw and
  matched 60-day difference per year (no interval) so the reader sees whether 2013, 2017 and 2020 carry everything.
  Placebo: the decision-day rank series, from its first defined day, circularly shifted by k in [180, n-180] days,
  200 draws; HEAVY / QUIET rebuilt; matched 60-day difference recomputed with the same cell edges; placebo share =
  share of draws at least as large as the real one.
  Out-of-sample checks fixed now: (a) M3, M4, M5 (never looked at): same sign and matched interval excluding zero?
  (b) ETH: the same matched test on ETH's own price (Binance, 2017-08+) with ETH's own exchange deposits (Coin Metrics
  eth FlowInExNtv, lag 1) and Binance ETHUSDT quote volume (lag 0), each ranked the same way, cells from ETH's own
  past-30 return and volatility; "same sign" = the matched 60-day difference is positive for every ETH measure that
  could be computed (at least one). (c) 2022-now alone for every measure.
  Pre-registered reading: a measure QUALIFIES if its matched HEAVY-minus-QUIET 60-day difference is positive with a
  90% interval excluding zero, positive in at least two of the three periods (M4 covers only two periods: both),
  and its placebo share <= 5%. "The observation holds up" if M1 qualifies AND at least two of M3/M4/M5 qualify AND
  the ETH check has the same sign. Otherwise "not confirmed outside the data it was first seen in", naming exactly
  which parts held and which did not. A failure means "not seen with this definition and data", never "the idea is
  wrong".

Part 2 - can it be used? Rules fixed now, each for M1 and for the best-supported never-looked-at measure among M3/M4/M5
  (the code picks the largest full-sample matched 60-day difference):
  A1 = hold when ma50 is on OR (HEAVY and the close is above its 20-day average).
  A2 = hold when ma50 is on OR (HEAVY and ma50 was on at any of the 20 previous closes t-20 .. t-1).
  A3 (sizing) = when ma50 is on, hold 100% if HEAVY or MIDDLE (or undefined), 50% if QUIET; 0 when ma50 is off.
  Fair exam = reluctant_sellers' _Exam (main_signal_check's four checks): BTC daily monthly starts from 2015-01 (or the
  first full month after the measure's rank is defined, whichever is later) to 2023-12, 0.10%; clock starts 2018-01
  (or later) .. 2023-12 x 24 closing hours textbook; the same cells as an ordinary person (act 10 hours later, 0.25%
  cost, 10% of entries missed, 200 simulations per clock); ETH SOL BNB XRP MSTR monthly starts to three years before
  today, each with its OWN price and ma50 / 20-day average and BTC's activity flag as a market-wide state (MSTR uses
  the previous day's flag). ma50 is re-scored on exactly the same starts (counts reported). Ranking score = worst of
  the four "beats holding" shares (main_signal_check's eligibility); median deepest loss = median over the clock
  textbook cells.
  Placebo: the measure's decision-day rank series circularly shifted by k in [180, n-180] days, 200 draws (the same k
  for A1, A2, A3 of that measure); HEAVY / QUIET rebuilt; the whole exam rebuilt with 20 ordinary-person simulations
  per clock (same seeds) and the real rule re-scored the same way; placebo share = share of draws whose ranking score
  is at least the real rule's.
  "Supported": A1 / A2 - ranking score >= ma50's AND median deepest loss no more than 5 points worse AND placebo share
  <= 5%. A3 - ranking score >= ma50's AND median deepest loss shallower than ma50's AND placebo share <= 5%.
  Six rule-by-measure combinations are tried: at a 5% placebo bar about 6 x 0.05 = 0.3 would pass that bar by luck if
  they were independent; Holm's step-down at 5% is applied across the six placebo p-values, p = (hits + 1) / (200 + 1).
Report only: days each rule holds that ma50 does not (A3: days it holds only half), by year, with average next-day BTC
  return; calendar-year return and in-year deepest loss of each rule and ma50 in 2018, 2020, 2022, 2023 (0.10% cost);
  lead/lag of each measure: x(t) = log(level(t-lag) / level(t-lag-14)) versus BTC's 14-day log return ending at t+k,
  k = -30 .. +30, noise band = 95th percentile of |corr| over 200 circular shifts of x by >= 180 days pooled over the
  shown k; what the data cannot tell (labelled guess wherever an explanation is offered): whether activity reflects
  new buyers arriving, old holders handing coins to new ones, custody reshuffles, or simply follows price with a delay
  the matching did not fully remove.
Today: a row labelled exactly "比特幣：50 日均線｜A1｜A2" (M1), the same for the chosen measure, A3's exposure, and each
  measure's level today.

Honesty: thresholds (30 / 365 days, 80th / 50th percentiles, lags, 20-day average and memory, 30 matching cells,
60-day blocks) were fixed by reasoning before the run; no alternatives tried. Coins moving are not coins sold (custody
moves, exchange wallet reshuffles count); exchange tagging is today's labels applied backwards; Binance volume
includes the zero-fee period 2022-07-08 .. 2023-03-22 when volume was inflated. The descriptive split and the
hypothesis come from the same history (M1, M2): only M3/M4/M5, ETH and the placebo are genuine checks. A pass would
only make a rule a candidate for the daily shadow log. Not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import old_coin_selling as oc
from dashboard.studies2 import order_flow as ofl
from dashboard.studies2 import reluctant_sellers as rsl
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED, YEAR = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED, msc.YEAR
DAY = pd.Timedelta(days=1)
CM_URL = sd.CM_URL
STORE_CM = {"btc": common.STORE / "activity_state_cm_btc.json", "eth": common.STORE / "activity_state_cm_eth.json"}
CM_METRICS = {"btc": ("FlowInExNtv", "SplyCur"), "eth": ("FlowInExNtv",)}
OBM_URL = oc.OBM_BASE
SUM_DAYS = 30
P_HEAVY, P_QUIET = 0.80, 0.50
H = (30, 60, 90)
MAIN_H = 60
BLOCK, BOOT = 60, 1000
NDEC, NTER = 10, 3
NCELL = NDEC * NTER
SAMPLE_FROM = "2012-01-01"
REPRO_END = "2026-09-30"
FIRST_MONTH = msc.DAILY_STARTS[0]
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.05
LL_K = tuple(range(-30, 31))
LL_SHOW = (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30)
PERIODS = rsl.PERIODS
WATCH_YEARS = (2018, 2020, 2022, 2023)
MEAS = ("m1", "m2", "m3", "m4", "m5")
NEW = ("m3", "m4", "m5")
LAG = {"m1": 7, "m2": 1, "m3": 7, "m4": 0, "m5": 7, "m4b": 0, "e2": 1, "e4": 0}
MNAME = {"m1": "M1 老幣移動（放了 155 天以上的幣被花掉，30 天加總，修剪單日極端值）",
         "m2": "M2 送進交易所的比特幣（30 天加總）",
         "m3": "M3 鏈上被花掉的比特幣總量（老幣＋新幣，30 天加總）",
         "m4": "M4 幣安現貨比特幣成交金額（美元，30 天加總，2017-08 起）",
         "m5": "M5 鏈上周轉率（M3 ÷ 流通量）",
         "m4b": "M4b（只描述）幣安現貨比特幣成交量（以顆計，30 天加總）",
         "e2": "以太幣：送進交易所的以太幣（30 天加總）",
         "e4": "以太幣：幣安 ETHUSDT 成交金額（美元，30 天加總）"}
MSHORT = {"m1": "M1 老幣移動", "m2": "M2 送進交易所", "m3": "M3 鏈上花費總量", "m4": "M4 幣安成交金額", "m5": "M5 周轉率",
          "m4b": "M4b 幣安成交量（顆）", "e2": "以太幣送進交易所", "e4": "以太幣幣安成交金額"}
MUNIT = {"m1": "顆", "m2": "顆", "m3": "顆", "m4": "美元", "m5": "", "m4b": "顆", "e2": "顆以太幣", "e4": "美元"}
INSAMPLE = {"m1": "事後看到的那條（樣本內）", "m2": "第一次觀察用的（樣本內）", "m3": "從沒看過", "m4": "從沒看過",
            "m5": "從沒看過", "m4b": "只描述", "e2": "以太幣檢查", "e4": "以太幣檢查"}
RULES = ("a1", "a2", "a3")
RNAME = {"ma50": "50 日均線（基準）", "a1": "A1：50 日均線，或「活躍＋收盤站上 20 日均線」也持有",
         "a2": "A2：50 日均線，或「活躍＋均線 20 天內還在」也持有（不被洗出場）",
         "a3": "A3：50 日均線在場時，安靜只拿一半（調部位）"}
RSHORT = {"ma50": "50 日均線", "a1": "A1", "a2": "A2", "a3": "A3"}
LEVELS = ("安靜", "中間", "活躍")

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- data ----------

def _cm_fetcher(asset: str):
    metrics = CM_METRICS[asset]

    def fetch() -> dict:
        reply = common.get_json(CM_URL, {"assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
                                         "start_time": "2009-01-03", "page_size": 10000}, timeout=90)
        rows = list(reply.get("data") or [])
        for _ in range(10):
            if not reply.get("next_page_url"):
                break
            reply = common.get_json(reply["next_page_url"], timeout=90)
            rows += list(reply.get("data") or [])
        out = {}
        for row in rows:
            day = str(row.get("time", ""))[:10]
            rec = {}
            for m in metrics:
                try:
                    v = float(row.get(m))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v >= 0:
                    rec[m] = v
            if day and rec:
                out[day] = rec
        if len(out) < 2000:
            raise RuntimeError(f"Coin Metrics {asset} 只回傳 {len(out)} 天")
        return out
    return fetch


def _frame(raw: dict) -> pd.DataFrame:
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    return cm[cm.index < pd.Timestamp(common.today_utc())].asfreq("D")


def _load() -> dict:
    problems = []
    close = te._daily_close()
    try:
        raw = json.loads(oc.STORE_OBM.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"找不到 old_coin_selling 存下來的 OBM 資料（{type(exc).__name__}）") from exc
    obm = {k: oc._series(raw, k) for k in ("ge155", "lt155")}
    cm = {}
    for asset in ("btc", "eth"):
        try:
            got, n = sd._stored(STORE_CM[asset], _cm_fetcher(asset))
            problems += [n] if n else []
            cm[asset] = _frame(got)
        except Exception as exc:  # noqa: BLE001 - a refused endpoint is dropped and named
            problems.append(f"Coin Metrics {asset} 取不到（{type(exc).__name__}），用到它的量測這次沒有算。")
    spot = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        try:
            fr, pr = ofl._daily("spot", sym)
            spot[sym] = fr
            if pr:
                problems.append(pr)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"幣安 {sym} 日線取不到（{type(exc).__name__}），用到它的量測這次沒有算。")
    other, more = msc._load_other()
    problems += more
    return {"close": close, "obm": obm, "obm_meta": raw.get("_meta", {}), "cm": cm, "spot": spot, "other": other,
            "problems": problems}


# ---------- the measures ----------

def _sum30(s: pd.Series) -> pd.Series:
    return s.asfreq("D").rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()


def _levels_data(data: dict) -> dict:
    """Data-date level series (continuous daily index) per measure."""
    obm, cm, spot = data["obm"], data["cm"], data["spot"]
    out = {"m1": oc._measure(obm["ge155"])["old"]}
    ge, lt = obm["ge155"].asfreq("D"), obm["lt155"].asfreq("D")
    days = ge.index.union(lt.index)
    out["m3"] = _sum30(ge.reindex(days) + lt.reindex(days))
    if "btc" in cm and "FlowInExNtv" in cm["btc"]:
        out["m2"] = _sum30(cm["btc"]["FlowInExNtv"])
    if "btc" in cm and "SplyCur" in cm["btc"]:
        sup = cm["btc"]["SplyCur"].reindex(out["m3"].index)
        out["m5"] = out["m3"] / sup.where(sup > 0)
    if "BTCUSDT" in spot:
        out["m4"] = _sum30(spot["BTCUSDT"]["quote"])
        out["m4b"] = _sum30(spot["BTCUSDT"]["volume"])
    if "eth" in cm and "FlowInExNtv" in cm["eth"]:
        out["e2"] = _sum30(cm["eth"]["FlowInExNtv"])
    if "ETHUSDT" in spot:
        out["e4"] = _sum30(spot["ETHUSDT"]["quote"])
    return out


def _rank(level: pd.Series) -> pd.Series:
    level = level.asfreq("D")
    return pd.Series(oc._trail_rank(level.values.astype(float)), index=level.index)


def _at(series: pd.Series, index: pd.DatetimeIndex, lag: int) -> np.ndarray:
    return series.reindex(index - lag * DAY).values.astype(float)


# ---------- Part 1 ----------

def _prep(close: pd.Series, year: int = YEAR) -> dict:
    c = close.values.astype(float)
    n = len(c)
    fwd = {}
    for h in H:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        fwd[h] = f
    lr = np.log(close).diff()
    return {"idx": close.index, "fwd": fwd,
            "p30": (close / close.shift(30) - 1).values, "p90": (close / close.shift(90) - 1).values,
            "vol": lr.rolling(20, min_periods=20).std().values,
            "ma": np.nan_to_num(msc._targets(close, year)["ma50"].values.astype(float))}


def _matched(v, hv, cell, sub) -> np.ndarray:
    """Cell-weighted HEAVY minus QUIET mean (weights = HEAVY days per cell); arrays may be 2-D (draws x days)."""
    v, hv, cell, sub = (np.atleast_2d(x) for x in (v, hv, cell, sub))
    num = np.zeros(v.shape[0])
    wt = np.zeros(v.shape[0])
    for k in range(NCELL):
        a = sub & (cell == k) & hv
        b = sub & (cell == k) & ~hv
        na, nb = a.sum(axis=1), b.sum(axis=1)
        ma = np.where(a, v, 0.0).sum(axis=1) / np.maximum(na, 1)
        mb = np.where(b, v, 0.0).sum(axis=1) / np.maximum(nb, 1)
        use = (na > 0) & (nb > 0)
        num += np.where(use, na * (ma - mb), 0.0)
        wt += np.where(use, na, 0)
    with np.errstate(all="ignore"):
        return np.where(wt > 0, num / np.maximum(wt, 1), np.nan)


def _q(a: np.ndarray) -> tuple:
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)


def _grp(mask: np.ndarray, lev: np.ndarray, P: dict) -> dict:
    out = {}
    for k, code in (("q", 0), ("h", 2)):
        g = mask & (lev == code)
        r = {"n": int(g.sum())}
        for h in H:
            v = P["fwd"][h][g]
            v = v[np.isfinite(v)]
            r[h] = float(v.mean()) if len(v) else None
        v = P["fwd"][MAIN_H][g]
        v = v[np.isfinite(v)]
        r["med"] = float(np.median(v)) if len(v) else None
        r["up"] = float((v > 0).mean()) if len(v) else None
        out[k] = r
    return out


def _part1(P: dict, rank_t: np.ndarray, seed: int, start: str = SAMPLE_FROM, light: bool = False) -> dict | None:
    idx = P["idx"]
    lev = oc._levels(rank_t)
    base = (np.isfinite(P["fwd"][MAIN_H]) & np.isfinite(P["p30"]) & np.isfinite(P["p90"]) & np.isfinite(P["vol"])
            & (idx >= pd.Timestamp(start)))
    keep = base & (lev >= 0)
    if keep.sum() < 4 * BLOCK:
        return None
    rng = np.random.default_rng(seed)
    p30, vol = P["p30"], P["vol"]
    e10 = np.quantile(p30[keep], np.linspace(0, 1, NDEC + 1))
    e3 = np.quantile(vol[keep], [1 / 3, 2 / 3])
    cell = np.full(len(idx), -1)
    okc = np.isfinite(p30) & np.isfinite(vol)
    cell[okc] = (np.clip(np.searchsorted(e10, p30[okc], side="right") - 1, 0, NDEC - 1) * NTER
                 + np.searchsorted(e3, vol[okc], side="right"))
    vals = {h: np.nan_to_num(P["fwd"][h], nan=0.0) for h in H}
    fin = {h: np.isfinite(P["fwd"][h]) for h in H}

    def mdiff(mask: np.ndarray, levv: np.ndarray, h: int, boot: bool) -> tuple:
        pos = np.where(mask)[0]
        if len(pos) < 2 * BLOCK:
            return None, (None, None)
        hv = levv[pos] == 2
        sub = ((levv[pos] == 0) | hv) & fin[h][pos] & (cell[pos] >= 0)
        v, c = vals[h][pos], cell[pos]
        real = float(_matched(v, hv, c, sub)[0])
        if not boot:
            return real, (None, None)
        b = rsl._boot_idx(len(pos), rng)
        return real, _q(_matched(v[b], hv[b], c[b], sub[b]))

    res = {"keep": keep, "n": int(keep.sum()), "start": idx[keep][0], "end": idx[keep][-1],
           "nq": int((keep & (lev == 0)).sum()), "nh": int((keep & (lev == 2)).sum()), "lev": lev}
    res["raw"] = _grp(keep, lev, P)
    res["m"] = {h: mdiff(keep, lev, h, True) for h in ((MAIN_H,) if light else H)}
    res["per"] = {}
    for lab, a, b in PERIODS:
        pm = keep & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        if pm.sum() < 2 * BLOCK:
            continue
        res["per"][lab] = {"m": mdiff(pm, lev, MAIN_H, True), "g": _grp(pm, lev, P), "n": int(pm.sum()),
                           "span": (idx[pm][0], idx[pm][-1])}
    # placebo: rank circularly shifted from its first defined day
    first = int(np.argmax(np.isfinite(rank_t)))
    nr = len(rank_t) - first
    real = res["m"][MAIN_H][0]
    prng = np.random.default_rng(seed + 7)
    got = []
    for k in prng.integers(GAP, nr - GAP + 1, DRAWS):
        r2 = rank_t.copy()
        r2[first:] = np.roll(r2[first:], int(k))
        l2 = oc._levels(r2)
        got.append(mdiff(base & (l2 >= 0), l2, MAIN_H, False)[0])
    got = np.array([g if _ok(g) else -np.inf for g in got])
    res["placebo"] = float((got >= real - 1e-12).mean()) if _ok(real) else None
    res["placebo_med"] = float(np.median(got[np.isfinite(got)])) if np.isfinite(got).any() else None
    if light:
        return res
    # descriptive splits
    q30 = np.quantile(p30[keep], np.linspace(0, 1, 6))
    q90 = np.quantile(P["p90"][keep], np.linspace(0, 1, 6))
    f30 = np.clip(np.searchsorted(q30, p30, side="right") - 1, 0, 4)
    f90 = np.clip(np.searchsorted(q90, np.nan_to_num(P["p90"]), side="right") - 1, 0, 4)
    res["fifths30"] = [(q30[k], q30[k + 1], _grp(keep & (f30 == k), lev, P)) for k in range(5)]
    res["fifths90"] = [(q90[k], q90[k + 1], _grp(keep & (f90 == k), lev, P)) for k in range(5)]
    on = P["ma"] > 0.5
    res["ma"] = [_grp(keep & on, lev, P), _grp(keep & ~on, lev, P)]
    res["years"] = {}
    for y in sorted(set(idx[keep].year)):
        ym = keep & (idx.year == y)
        g = _grp(ym, lev, P)
        raw = (g["h"][MAIN_H] - g["q"][MAIN_H]) if _ok(g["h"][MAIN_H]) and _ok(g["q"][MAIN_H]) else None
        pos = np.where(ym)[0]
        hv = lev[pos] == 2
        sub = ((lev[pos] == 0) | hv) & (cell[pos] >= 0)
        mt = float(_matched(vals[MAIN_H][pos], hv, cell[pos], sub)[0]) if sub.any() else None
        res["years"][int(y)] = {"n": int(ym.sum()), "nq": g["q"]["n"], "nh": g["h"]["n"], "raw": raw,
                                "m": mt if _ok(mt) else None}
    return res


def _qualifies(r: dict | None) -> dict:
    if not r:
        return {"ok": False, "pos": False, "excl": False, "subs": 0, "nsub": 0, "plc": False}
    v, (lo, _) = r["m"][MAIN_H]
    nsub = len(r["per"])
    subs = sum(1 for x in r["per"].values() if _ok(x["m"][0]) and x["m"][0] > 0)
    need = min(2, nsub)
    q = {"pos": _ok(v) and v > 0, "excl": _ok(lo) and lo > 0, "subs": subs, "nsub": nsub,
         "plc": _ok(r["placebo"]) and r["placebo"] <= PLACEBO_BAR}
    q["sub_ok"] = nsub >= 2 and subs >= need
    q["ok"] = bool(q["pos"] and q["excl"] and q["sub_ok"] and q["plc"])
    return q


def _repro(close: pd.Series, rank_m1: np.ndarray) -> dict:
    """The coordinator's tabulation (M1, 2012-01 .. 2026-09, 60-day forward mean, fifths of past-30-day return)."""
    idx = close.index
    c = close.values.astype(float)
    n = len(c)
    f60 = np.full(n, np.nan)
    f60[:n - 60] = c[60:] / c[:n - 60] - 1
    past = (close / close.shift(30) - 1).values
    lev = oc._levels(rank_m1)
    keep = ((lev >= 0) & np.isfinite(f60) & np.isfinite(past) & (idx >= pd.Timestamp(SAMPLE_FROM))
            & (idx <= pd.Timestamp(REPRO_END)))
    q = np.quantile(past[keep], np.linspace(0, 1, 6))
    fi = np.clip(np.searchsorted(q, past, side="right") - 1, 0, 4)
    rows = []
    for k in range(5):
        a = keep & (fi == k)
        rows.append((q[k], q[k + 1], float(f60[a & (lev == 0)].mean()), float(f60[a & (lev == 2)].mean()),
                     int((a & (lev == 0)).sum()), int((a & (lev == 2)).sum())))
    per = []
    for lab, a, b in PERIODS:
        s = keep & (past > 0) & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        per.append((lab, float(f60[s & (lev == 0)].mean()), float(f60[s & (lev == 2)].mean())))
    return {"rows": rows, "per": per}


# ---------- Part 2: rules and exam ----------

def _combine(kind: str, ma, a20, rma, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "a1":
        return np.maximum(ma, f * a20)
    if kind == "a2":
        return np.maximum(ma, f * rma)
    if kind == "a3":
        return ma * (1.0 - 0.5 * f)
    raise ValueError(kind)


class _Exam(rsl._Exam):
    """reluctant_sellers' exam with this study's three ways to combine the activity flag (A3 = fractional sizing)."""

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


def _flags(rank: pd.Series) -> dict:
    r = rank.values
    heavy = pd.Series(np.where(np.isfinite(r), (r > P_HEAVY).astype(float), 0.0), index=rank.index)
    quiet = pd.Series(np.where(np.isfinite(r), (r < P_QUIET).astype(float), 0.0), index=rank.index)
    return {"a1": heavy, "a2": heavy, "a3": quiet}


def _year_stats(close: pd.Series, pos: np.ndarray) -> dict:
    change = close.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    s = pd.Series(net, index=close.index)
    out = {}
    for y, g in s.groupby(s.index.year):
        eq = np.concatenate([[1.0], np.cumprod(1 + g.values)])
        out[int(y)] = (float(eq[-1] - 1), float((eq / np.maximum.accumulate(eq) - 1).min()))
    return out


def _extra_days(close: pd.Series, mask: np.ndarray, first: pd.Timestamp) -> dict:
    idx = close.index
    ch = close.pct_change().values
    nxt = np.concatenate([ch[1:], [np.nan]])
    m = mask & (idx >= first)
    years = {}
    for y in sorted(set(idx[m].year)):
        g = m & (idx.year == y)
        years[int(y)] = (int(g.sum()), float(np.nanmean(nxt[g])) if np.isfinite(nxt[g]).any() else None)
    ok = m & np.isfinite(nxt)
    return {"n": int(m.sum()), "years": years, "avg": float(nxt[ok].mean()) if ok.any() else None}


def _holm(p: dict, alpha: float = 0.05) -> list:
    order = sorted(p, key=lambda k: p[k])
    m = len(order)
    out = []
    for j, k in enumerate(order):
        if p[k] <= alpha / (m - j):
            out.append(k)
        else:
            break
    return out


# ---------- lead/lag ----------

def _leadlag(level: pd.Series, lag: int, close: pd.Series, seed: int) -> dict:
    o = level.where(level > 0)
    x = _at(np.log(o / o.shift(14)), close.index, lag)
    lc = np.log(close.values.astype(float))
    r14 = np.full(len(lc), np.nan)
    r14[14:] = lc[14:] - lc[:-14]
    first = int(np.argmax(np.isfinite(x)))
    f, r = x[first:], r14[first:]
    n = len(f)

    def corr(fv, k):
        a, b = (fv[:n - k], r[k:]) if k >= 0 else (fv[-k:], r[:n + k])
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    real = {k: corr(f, k) for k in LL_K}
    rng = np.random.default_rng(seed)
    pool = []
    for s in rng.integers(GAP, n - GAP + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL_SHOW]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n, "from": close.index[first]}


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    idx = close.index
    lv = _levels_data(data)
    rank_d = {k: _rank(v) for k, v in lv.items()}
    rank_t = {k: _at(rank_d[k], idx, LAG[k]) for k in lv if not k.startswith("e")}
    P = _prep(close)
    p1 = {}
    for i, k in enumerate(MEAS + ("m4b",)):
        if k in rank_t:
            p1[k] = _part1(P, rank_t[k], SEED + 300 + i, light=(k == "m4b"))
    qual = {k: _qualifies(p1.get(k)) for k in MEAS}
    # ETH
    eth = {}
    if "ETH" in data["other"]:
        ce = data["other"]["ETH"][0].astype(float)
        ce = pd.Series(ce.values, index=pd.to_datetime(ce.index)).sort_index()
        ce = ce[~ce.index.duplicated(keep="last")].asfreq("D").ffill()
        PE = _prep(ce)
        for i, k in enumerate(("e2", "e4")):
            if k in rank_d:
                eth[k] = _part1(PE, _at(rank_d[k], ce.index, LAG[k]), SEED + 400 + i, start="2017-01-01", light=True)
                if eth[k]:
                    eth[k]["per"] = {lab: x for lab, x in eth[k]["per"].items()}
    eth_vals = [eth[k]["m"][MAIN_H][0] for k in eth if eth[k] and _ok(eth[k]["m"][MAIN_H][0])]
    eth_same = bool(eth_vals) and all(v > 0 for v in eth_vals)
    n_new = sum(1 for k in NEW if qual[k]["ok"])
    holds = bool(qual["m1"]["ok"] and n_new >= 2 and eth_same)
    repro = _repro(close, rank_t["m1"])
    lead = {k: _leadlag(lv[k], LAG[k], close, SEED + 500 + i) for i, k in enumerate(MEAS) if k in lv}
    t_desc = time.time() - t0

    # Part 2
    cand = [k for k in NEW if p1.get(k) and _ok(p1[k]["m"][MAIN_H][0])]
    best = max(cand, key=lambda k: p1[k]["m"][MAIN_H][0]) if cand else None
    use = ["m1"] + ([best] if best else [])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    exams, base, base_small = {}, {}, {}
    s, small, placebo, pval, plc_med, flags, rser, firsts = {}, {}, {}, {}, {}, {}, {}, {}
    for j, mk in enumerate(use):
        rs_ = pd.Series(rank_t[mk], index=idx)
        rser[mk] = rs_
        r0 = idx[int(np.argmax(np.isfinite(rs_.values)))]
        first = max(pd.Timestamp(FIRST_MONTH), (r0 + pd.offsets.MonthBegin(1)).normalize())
        firsts[mk] = first
        key = str(first.date())
        if key not in exams:
            exams[key] = _Exam(close, data["other"], price, b0, pre, first)
            base[key] = exams[key].summary(None, 1, SIMS, "ma50")
            base_small[key] = exams[key].summary(None, 1, PL_SIMS, "ma50")
        ex = exams[key]
        fl = _flags(rs_)
        flags[mk] = fl
        for i, rule in enumerate(RULES):
            ki = 140 + 10 * j + i
            s[(mk, rule)] = ex.summary(fl[rule], ki, SIMS, rule)
            small[(mk, rule)] = ex.summary(fl[rule], ki, PL_SIMS, rule)
        n = int((idx >= r0).sum())
        prng = np.random.default_rng(SEED + 610 + j)
        hits = {r: 0 for r in RULES}
        lows = {r: [] for r in RULES}
        for k in prng.integers(GAP, n - GAP + 1, DRAWS):
            fk = _flags(sd._roll(rs_, r0, int(k)))
            for i, rule in enumerate(RULES):
                x = ex.summary(fk[rule], 140 + 10 * j + i, PL_SIMS, rule)
                lows[rule].append(x["low"])
                hits[rule] += int(sd._score_ge(x, small[(mk, rule)]))
        for rule in RULES:
            placebo[(mk, rule)] = hits[rule] / DRAWS
            pval[(mk, rule)] = (hits[rule] + 1) / (DRAWS + 1)
            plc_med[(mk, rule)] = float(np.median(lows[rule]))
    holm = _holm(pval)
    res = {}
    for mk in use:
        b = base[str(firsts[mk].date())]
        for rule in RULES:
            x = s[(mk, rule)]
            q = {"score": sd._score_ge(x, b)}
            if rule == "a3":
                q["loss"] = x["c_deep_med"] > b["c_deep_med"] + 1e-12
            else:
                q["loss"] = x["c_deep_med"] >= b["c_deep_med"] - LOSS_SLACK - 1e-12
            q["plc"] = placebo[(mk, rule)] <= PLACEBO_BAR
            q["pass"] = all(q.values())
            q["holm"] = (mk, rule) in holm
            res[(mk, rule)] = q
    # report-only pieces on BTC daily
    ex0 = exams[str(firsts["m1"].date())]
    ma = np.nan_to_num(ex0.ma)
    pos = {"ma50": ma}
    extra, years = {}, {"ma50": _year_stats(close, ma)}
    for mk in use:
        for rule in RULES:
            p_ = ex0.btc_pos(flags[mk][rule], rule)
            pos[(mk, rule)] = p_
            years[(mk, rule)] = _year_stats(close, p_)
            mask = (p_ > ma + 1e-9) if rule != "a3" else (p_ < ma - 1e-9)
            extra[(mk, rule)] = _extra_days(close, mask, firsts[mk])
    return {"lv": lv, "rank_d": rank_d, "rank_t": rank_t, "p1": p1, "qual": qual, "eth": eth, "eth_same": eth_same,
            "n_new": n_new, "holds": holds, "repro": repro, "lead": lead, "best": best, "use": use,
            "exams": exams, "base": base, "firsts": firsts, "s": s, "small": small, "placebo": placebo, "pval": pval,
            "plc_med": plc_med, "holm": holm, "res": res, "pos": pos, "extra": extra, "years": years,
            "problem": problem, "secs": time.time() - t0, "secs_desc": t_desc}


# ---------- report pieces ----------

def _lvl_text(k: str, v) -> str:
    if not _ok(v):
        return "—"
    if k in ("m4", "e4"):
        return f"{v / 1e8:,.0f} 億美元"
    if k == "m5":
        return f"{v * 100:.1f}%（流通量的比例）"
    return f"{v:,.0f} {MUNIT[k]}"


def _qh(g: dict, key, f=_p) -> str:
    return f"{f(g['q'][key])}｜{f(g['h'][key])}"


def _grp_row(label: str, g: dict) -> list:
    return [label, f"{g['q']['n']:,}｜{g['h']['n']:,}", _qh(g, 30), _qh(g, 60), _qh(g, "med"), _qh(g, "up", _share),
            _qh(g, 90)]


def _now(data: dict, r: dict) -> list[dict]:
    close = data["close"]
    last = close.index[-1]
    word = lambda v: "持有" if v > 0.75 else ("持有一半" if v > 0.25 else "現金")  # noqa: E731
    out = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"}]
    p = r["pos"]
    out.append({"label": "比特幣：50 日均線｜A1｜A2",
                "value": f"{word(p['ma50'][-1])}｜{word(p[('m1', 'a1')][-1])}｜{word(p[('m1', 'a2')][-1])}"})
    out.append({"label": "　上面用的活躍量測", "value": MSHORT["m1"]})
    out.append({"label": "　A3（用 M1）在場比例", "value": f"{p[('m1', 'a3')][-1] * 100:.0f}%"})
    b = r["best"]
    if b:
        out.append({"label": f"比特幣：50 日均線｜A1｜A2（改用 {MSHORT[b]}）",
                    "value": f"{word(p['ma50'][-1])}｜{word(p[(b, 'a1')][-1])}｜{word(p[(b, 'a2')][-1])}"})
        out.append({"label": f"　A3（用 {MSHORT[b]}）在場比例", "value": f"{p[(b, 'a3')][-1] * 100:.0f}%"})
    for k in MEAS + ("m4b",):
        if k not in r["lv"]:
            out.append({"label": f"{MSHORT[k]}：今天", "value": "資料取不到"})
            continue
        day = last - LAG[k] * DAY
        lvv = r["lv"][k].get(day)
        rk = r["rank_d"][k].get(day)
        lev = oc._levels(np.array([rk if _ok(rk) else np.nan]))[0]
        out.append({"label": f"{MSHORT[k]}：今天用的資料日 {_d(day)}（晚 {LAG[k]} 天）",
                    "value": (f"{_lvl_text(k, lvv)}；在過去一年排第 {rk * 100:.0f}（0–100）；{LEVELS[lev]}"
                              if _ok(rk) else f"{_lvl_text(k, lvv)}；排名資料不足")})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；規則沒有通過考試前不會用"})
    return out


def study() -> dict:
    t_all = time.time()
    data = _load()
    r = _compute(data)
    close = data["close"]
    p1, qual, eth = r["p1"], r["qual"], r["eth"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    # ----- reproduction -----
    rp = r["repro"]
    coord = [(6.3, 11.5), (5.0, 16.0), (13.0, 25.7), (22.6, 28.2), (29.7, 48.5)]
    coord_p = [(45.0, 53.4), (9.5, 41.9), (6.6, 11.3)]
    rows_rp = [[f"第 {k + 1} 等份（過去 30 天 {_p(lo)} 到 {_p(hi)}）", f"{nq:,}｜{nh:,}", f"{_p(q)}｜{_p(h)}",
                f"{coord[k][0]:+.1f}%｜{coord[k][1]:+.1f}%"] for k, (lo, hi, q, h, nq, nh) in enumerate(rp["rows"])]
    rows_rp += [[f"過去 30 天是漲的日子，{lab}", "—", f"{_p(q)}｜{_p(h)}", f"{coord_p[i][0]:+.1f}%｜{coord_p[i][1]:+.1f}%"]
                for i, (lab, q, h) in enumerate(rp["per"])]
    heavy_wins = sum(1 for x in rp["rows"] if x[3] > x[2])
    max_gap = max([abs(x[2] * 100 - coord[k][0]) for k, x in enumerate(rp["rows"])]
                  + [abs(x[3] * 100 - coord[k][1]) for k, x in enumerate(rp["rows"])]
                  + [abs(x[1] * 100 - coord_p[i][0]) for i, x in enumerate(rp["per"])]
                  + [abs(x[2] * 100 - coord_p[i][1]) for i, x in enumerate(rp["per"])])

    # ----- Part 1 summary -----
    rows_s = []
    for k in MEAS + ("m4b",):
        x = p1.get(k)
        if not x:
            rows_s.append([MNAME[k], INSAMPLE[k]] + ["資料不足"] * 8)
            continue
        per = "；".join(f"{lab} {_pt(v['m'][0])} {_ci(*v['m'][1], f=_pt)}" for lab, v in x["per"].items())
        q = qual.get(k)
        rows_s.append([MNAME[k], INSAMPLE[k], f"{_d(x['start'])} 到 {_d(x['end'])}（{x['n']:,} 天；安靜 {x['nq']:,}、活躍 {x['nh']:,}）",
                       f"{_p(x['raw']['q'][60])}｜{_p(x['raw']['h'][60])}",
                       f"{_pt(x['m'][60][0])} {_ci(*x['m'][60][1], f=_pt)}",
                       (f"{_pt(x['m'][30][0])} {_ci(*x['m'][30][1], f=_pt)}" if 30 in x["m"] else "—"),
                       (f"{_pt(x['m'][90][0])} {_ci(*x['m'][90][1], f=_pt)}" if 90 in x["m"] else "—"),
                       per or "—", f"{_share1(x['placebo'])}（錯位中位數 {_pt(x['placebo_med'])}）",
                       ("—（只描述）" if q is None else ("符合" if q["ok"] else "不符合（" + "、".join(
                           ([] if q["pos"] else ["不是正的"]) + ([] if q["excl"] else ["區間碰到零"])
                           + ([] if q["sub_ok"] else [f"分段正的只有 {q['subs']}／{q['nsub']} 段"])
                           + ([] if q["plc"] else ["錯位比例超過 5%"])) + "）"))])

    # ----- ETH -----
    rows_e = []
    for k in ("e2", "e4"):
        x = eth.get(k)
        if not x:
            rows_e.append([MNAME[k], "資料不足或取不到", "—", "—", "—", "—"])
            continue
        per = "；".join(f"{lab} {_pt(v['m'][0])} {_ci(*v['m'][1], f=_pt)}" for lab, v in x["per"].items())
        rows_e.append([MNAME[k], f"{_d(x['start'])} 到 {_d(x['end'])}（{x['n']:,} 天；安靜 {x['nq']:,}、活躍 {x['nh']:,}）",
                       f"{_p(x['raw']['q'][60])}｜{_p(x['raw']['h'][60])}",
                       f"{_pt(x['m'][60][0])} {_ci(*x['m'][60][1], f=_pt)}", per or "—", _share(x["placebo"])])

    # ----- details per measure -----
    detail_tables = []
    for k in MEAS:
        x = p1.get(k)
        if not x:
            continue
        rows = []
        for j, (lo, hi, g) in enumerate(x["fifths30"]):
            rows.append(_grp_row(f"過去 30 天漲跌第 {j + 1} 等份（{_p(lo)} 到 {_p(hi)}）", g))
        for j, (lo, hi, g) in enumerate(x["fifths90"]):
            rows.append(_grp_row(f"過去 90 天漲跌第 {j + 1} 等份（{_p(lo)} 到 {_p(hi)}）", g))
        rows.append(_grp_row("50 日均線在場（收盤高於 50 日均線）", x["ma"][0]))
        rows.append(_grp_row("50 日均線不在場", x["ma"][1]))
        rows.append(_grp_row("全部", x["raw"]))
        detail_tables.append({
            "caption": (f"第 1 部分（描述）：{MNAME[k]}。每格左邊是「安靜」（排名在過去一年一半以下），右邊是「活躍」（排名在第 80 百分位以上）。"
                        f"分等份的界線用這個量測自己的樣本（{_d(x['start'])} 到 {_d(x['end'])}）。上漲比例＝之後 60 天收在比今天高的日子占幾成。"),
            "columns": ["分組", "天數：安靜｜活躍", "之後 30 天平均", "之後 60 天平均", "之後 60 天中位數", "之後 60 天上漲比例", "之後 90 天平均"],
            "rows": rows})

    # ----- years -----
    all_years = sorted({y for k in MEAS if p1.get(k) for y in p1[k]["years"]})
    rows_y = []
    for y in all_years:
        row = [str(y)]
        for k in MEAS:
            v = p1[k]["years"].get(y) if p1.get(k) else None
            row.append("—" if not v else f"{_pt(v['raw'])}｜{_pt(v['m'])}（活躍 {v['nh']} 天、安靜 {v['nq']} 天）")
        rows_y.append(row)

    # ----- Part 2 -----
    use, s, base, res = r["use"], r["s"], r["base"], r["res"]
    rows_b = []
    seen = set()
    for mk in use:
        key = str(r["firsts"][mk].date())
        b = base[key]
        if key not in seen:
            seen.add(key)
            ck = b["checks"]
            rows_b.append([f"{RNAME['ma50']}（起算 {key[:7]}）", "—", f"{_share(ck[0])}（{b['d_n']} 個）",
                           f"{_share(ck[1])}（{b['c_n']:,} 格）", _share(ck[2]), f"{_share(ck[3])}（{b['o_n']} 個）",
                           _share1(b["low"]) + ("" if b["eligible"] else "（虧損條件未達 95%）"),
                           f"{_pc(b['c_deep_med'])}｜{_pc(b['c_deep_worst'])}", f"{b['c_edge']:+.1f} 點", "—", "—"])
        for rule in RULES:
            x = s[(mk, rule)]
            ck = x["checks"]
            rows_b.append([f"{RNAME[rule]}（起算 {key[:7]}）", MSHORT[mk], f"{_share(ck[0])}（{x['d_n']} 個）",
                           f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                           _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                           f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                           f"{_share(r['placebo'][(mk, rule)])}（真的 {_share1(r['small'][(mk, rule)]['low'])}，"
                           f"錯位中位數 {_share1(r['plc_med'][(mk, rule)])}）",
                           f"{r['pval'][(mk, rule)]:.3f}"])
    rows_pass = []
    for mk in use:
        b = base[str(r["firsts"][mk].date())]
        for rule in RULES:
            q, x = res[(mk, rule)], s[(mk, rule)]
            lossc = "最深虧損中位數比均線淺" if rule == "a3" else "最深虧損中位數最多比均線深 5 點"
            rows_pass += [[f"{RSHORT[rule]}（{MSHORT[mk]}）", "排名分數不輸 50 日均線",
                           f"{_share1(x['low'])} vs {_share1(b['low'])}", yes(q["score"])],
                          ["", lossc, f"{_pc(x['c_deep_med'])} vs {_pc(b['c_deep_med'])}", yes(q["loss"])],
                          ["", "隨機錯位比例 ≤ 5%", _share(r["placebo"][(mk, rule)]), yes(q["plc"])],
                          ["", "三條都要過", "", yes(q["pass"])],
                          ["", "（另看）Holm 校正六個錯位 p 值後還在", f"p = {r['pval'][(mk, rule)]:.3f}", "還在" if q["holm"] else "不在"]]
    rows_x = []
    for mk in use:
        for rule in RULES:
            e = r["extra"][(mk, rule)]
            yrs = "；".join(f"{y}：{n} 天 {_p(a)}" for y, (n, a) in e["years"].items()) or "沒有"
            what = "均線不在場但它持有" if rule != "a3" else "均線在場但它只拿一半"
            rows_x.append([f"{RSHORT[rule]}（{MSHORT[mk]}）", what, f"{e['n']:,}", _p(e["avg"]), yrs])
    rows_w = []
    for y in WATCH_YEARS:
        row = [str(y)]
        mv = r["years"]["ma50"].get(y)
        row.append(f"{_p(mv[0])}｜{_p(mv[1])}" if mv else "—")
        for mk in use:
            for rule in RULES:
                v = r["years"][(mk, rule)].get(y)
                row.append(f"{_p(v[0])}｜{_p(v[1])}" if v else "—")
        rows_w.append(row)
    wcols = ["年", "50 日均線：報酬｜年內最深"] + [f"{RSHORT[rule]}（{MSHORT[mk]}）：報酬｜年內最深"
                                           for mk in use for rule in RULES]
    # lead/lag
    lead = r["lead"]
    lk = [k for k in MEAS if k in lead]
    rows_l = []
    for kk in LL_SHOW:
        rows_l.append([f"{kk:+d} 天" + ("（價格先動）" if kk < 0 else "（量測先動）" if kk > 0 else "（同一天）")]
                      + [f"{lead[k]['corr'][kk]:+.3f}" + ("＊" if abs(lead[k]["corr"][kk]) > lead[k]["band"] else "")
                         for k in lk])
    peak_rows = []
    for side, lab in ((-1, "價格先動那邊最大"), (1, "量測先動那邊最大")):
        row = [lab]
        for k in lk:
            c = {q: v for q, v in lead[k]["corr"].items() if q * side > 0}
            kb = max(c, key=lambda q: abs(c[q]))
            row.append(f"{kb:+d} 天 {c[kb]:+.3f}" + ("＊" if abs(c[kb]) > lead[k]["band"] else ""))
        peak_rows.append(row)
    rows_l += peak_rows
    rows_l.append(["雜訊帶（超過才打＊）"] + [f"{lead[k]['band']:.3f}" for k in lk])

    # ----- tables -----
    hint_m = ("配對＝把日子依「過去 30 天漲跌」分十組、再依「過去 20 天每天漲跌的波動」分三組，共 30 格；在同一格裡比活躍和安靜，"
              "再依活躍天數加權平均。這樣比的是「漲跌和波動都差不多的日子」，避免「其實只是價格已經漲過了」。")
    tables = [
        {"caption": ("重做協調者的表（先做，再寫測試）：M1 老幣移動，2012-01 到 2026-09，依過去 30 天漲跌分五等份，"
                     "之後 60 天平均（安靜｜活躍）。最後一欄是協調者給的數字。"),
         "columns": ["分組", "天數：安靜｜活躍", "這次算的：安靜｜活躍", "協調者的：安靜｜活躍"], "rows": rows_rp},
        {"caption": ("第 1 部分：活躍（排名在過去一年第 80 百分位以上）減安靜（一半以下）之後的漲跌差（點）。" + hint_m +
                     f" 90% 區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次。隨機錯位＝把排名整條往後挪 {GAP} 天以上（頭尾相接）{DRAWS} 次，"
                     "錯位的配對差至少和真的一樣大的比例。事先定的「符合」：60 天配對差是正的、區間不碰零、分段至少兩段正的、錯位 ≤ 5%。"),
         "columns": ["量測", "看過沒有", "樣本", "之後 60 天平均（原始）：安靜｜活躍", "配對差：60 天 ［90% 區間］",
                     "配對差：30 天", "配對差：90 天", "配對差（60 天）分三段", "隨機錯位比例", "事先定的條件"], "rows": rows_s},
        {"caption": ("第 1 部分，樣本外檢查 (b)：以太幣。用以太幣自己的價格，配以太幣自己的「送進交易所」和「幣安成交金額」，"
                     "同樣的排名、同樣的配對（用以太幣自己的過去 30 天漲跌和波動）。事先定的「同方向」＝每一條算得出來的以太幣量測，"
                     "60 天配對差都是正的。"),
         "columns": ["量測", "樣本", "之後 60 天平均（原始）：安靜｜活躍", "配對差：60 天 ［90% 區間］", "分段", "隨機錯位比例"],
         "rows": rows_e},
    ] + detail_tables + [
        {"caption": ("第 1 部分，每個日曆年（只描述）：活躍減安靜之後 60 天平均的差，原始｜配對（年內格子少，配對數字很粗）。"
                     "用來看是不是只有 2013、2017、2020 這幾年在撐。"),
         "columns": ["年"] + [MSHORT[k] for k in MEAS], "rows": rows_y},
        {"caption": (f"第 2 部分：規則的公平考試（main_signal_check 的做法）。M1 用 {_d(r['firsts']['m1'])} 起的起算月；"
                     + (f"{MSHORT[r['best']]} 用 {_d(r['firsts'][r['best']])} 起（量測有排名以後的第一個整月）；" if r["best"] else "")
                     + f"每組起算日都把 50 日均線用完全相同的起算日重算。比特幣日線起算月到 {msc.DAILY_STARTS[1][:7]}，"
                     f"起算月×24 個收盤時間（2018-01 起），一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），"
                     "其他資產 ETH、SOL、BNB、XRP、MSTR 各用自己的價格和 50 日均線，配比特幣的活躍狀態當全市場狀態。"
                     "排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。隨機錯位＝把排名整條挪 180 天以上，200 次，"
                     f"一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方法重算來比。p 值＝（不輸的次數＋1）÷ 201。"),
         "columns": ["規則", "量測", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "隨機錯位比例", "p 值"], "rows": rows_b},
        {"caption": ("第 2 部分的通過條件（事先定好）。六種「規則×量測」一起試：每種在 5% 的錯位門檻下碰巧過的機率約 5%，"
                     "六種加起來大約 0.3 種會碰巧過（如果彼此獨立）；所以另外用 Holm 校正（最小的 p 值要 ≤ 0.05÷6，下一個 ≤ 0.05÷5，依此類推）。"),
         "columns": ["規則", "條件", "數字", "結果"], "rows": rows_pass},
        {"caption": ("第 2 部分（只描述）：A1、A2 在均線不在場時多持有的天數；A3 在均線在場時只拿一半的天數。依年分，"
                     "後面是那些天的「隔天比特幣平均漲跌」。"),
         "columns": ["規則", "哪種天", "天數", "隔天平均漲跌", "每年：天數 隔天平均"], "rows": rows_x},
        {"caption": f"第 2 部分（只描述）：2018、2022 兩段下跌和 2020、2023 兩段上漲的日曆年報酬和年內最深跌幅（比特幣日線，扣 {COST * 100:.2f}% 成本）。",
         "columns": wcols, "rows": rows_w},
        {"caption": ("只描述：誰先動？每個量測「過去 14 天的變化」（當時拿得到的，已扣掉延遲）和比特幣 14 天漲跌錯開不同天數的相關。"
                     "負的天數＝價格先動、量測後動；正的天數＝量測先動。雜訊帶＝把量測錯開 180 天以上 200 次的 |相關| 第 95 百分位，超過打＊。"
                     "注意量測有延遲（M1、M3、M5 晚 7 天，M2 晚 1 天），所以 -7 天附近才是同一段日曆時間。相隔不到 14 天有重疊，相關會被拉高。"),
         "columns": ["錯開天數"] + [MSHORT[k] for k in lk], "rows": rows_l},
    ]

    # ----- verdict -----
    def mline(k: str) -> str:
        x = p1.get(k)
        if not x:
            return f"{MSHORT[k]} 算不出來"
        v, (lo, hi) = x["m"][MAIN_H]
        return (f"{MSHORT[k]} {_pt(v)} {_ci(lo, hi, f=_pt)}、分段正的 {qual[k]['subs']}／{qual[k]['nsub']}、"
                f"錯位 {_share1(x['placebo'])}（{'符合' if qual[k]['ok'] else '不符合'}）")

    held = [MSHORT[k] for k in MEAS if qual[k]["ok"]]
    nheld = [MSHORT[k] for k in MEAS if not qual[k]["ok"]]
    eth_txt = "；".join(f"{MSHORT[k]} {_pt(eth[k]['m'][MAIN_H][0])} {_ci(*eth[k]['m'][MAIN_H][1], f=_pt)}"
                        for k in ("e2", "e4") if eth.get(k)) or "算不出來"
    late = "；".join(f"{MSHORT[k]} {_pt(p1[k]['per']['2022 起']['m'][0])} {_ci(*p1[k]['per']['2022 起']['m'][1], f=_pt)}"
                     for k in MEAS if p1.get(k) and "2022 起" in p1[k]["per"])
    zero = (f"重做協調者的表：五等份裡活躍都比安靜高的有 {heavy_wins}／5 等份，和協調者的數字最多差 {max_gap:.1f} 點（量到的）。")
    one = ("第 1 部分（量到的，60 天，配對過去 30 天漲跌和波動後的「活躍 − 安靜」）：" + "；".join(mline(k) for k in MEAS) + "。"
           f"樣本外 (a) 從沒看過的 M3／M4／M5 符合 {r['n_new']} 條（要 2 條）；(b) 以太幣 {eth_txt}，"
           f"{'同方向' if r['eth_same'] else '不是同方向'}；(c) 只看 2022 起：{late}。"
           + ("事先定的解讀：「這個觀察站得住」。" if r["holds"] else
              "事先定的解讀：「在第一次看到它的資料之外沒有被確認」——"
              + (f"符合的：{'、'.join(held)}；" if held else "五條量測都不符合；")
              + (f"不符合的：{'、'.join(nheld)}；" if nheld else "")
              + ("M1 本身" + ("符合" if qual["m1"]["ok"] else "不符合") + "；")
              + ("以太幣同方向。" if r["eth_same"] else "以太幣不是同方向。")
              + "這只表示用這個定義和這些資料沒有看到，不表示想法是錯的。"))
    passed = [f"{RSHORT[rule]}（{MSHORT[mk]}）" for mk in use for rule in RULES if res[(mk, rule)]["pass"]]
    holm_ok = [f"{RSHORT[rule]}（{MSHORT[mk]}）" for (mk, rule) in r["holm"]]
    b1 = base[str(r["firsts"]["m1"].date())]
    two = (f"第 2 部分（量到的）：用 M1" + (f" 和 {MSHORT[r['best']]}（M3／M4／M5 裡配對差最大的）" if r["best"] else "")
           + f"，A1／A2／A3 共 {len(use) * 3} 種。均線（2015 起）排名分數 {_share1(b1['low'])}、最深虧損中位數 {_pc(b1['c_deep_med'])}。"
           + "；".join(f"{RSHORT[rule]}（{MSHORT[mk]}）排名分數 {_share1(s[(mk, rule)]['low'])}、最深虧損中位數 "
                       f"{_pc(s[(mk, rule)]['c_deep_med'])}、錯位 {_share(r['placebo'][(mk, rule)])}"
                       for mk in use for rule in RULES) + "。"
           + (f"三條都過的：{'、'.join(passed)}" if passed else "沒有一種三條都過")
           + (f"；Holm 校正後還在的：{'、'.join(holm_ok)}。" if holm_ok else "；Holm 校正後一個都不在。"))
    three = ("資料分不出來的（推測，不是量到的）：活躍可能是新買家進場、老持有人把幣交給新持有人、交易所或託管機構搬幣，"
             "也可能只是跟著價格晚幾天動，配對沒有完全扣掉；這份研究不選其中一個。")
    verdict = zero + one + two + three

    notes = [
        "觀察到的，事後才看到，不是事先預期的：三份研究都預期「賣方那邊很活躍」之後比較差，量到的都相反。"
        "reluctant_sellers：價格站上 20 日均線時，大量送進交易所之後 30 天平均 +25.5%，流入少 +4.6%；在 20 日均線以下 +6.4% 對 +2.3%。"
        "old_coin_selling：老幣大量移動之後 60 天平均 +41.8% 對安靜 +17.1%（價格在上）、+14.9% 對 +9.1%（價格在下）。"
        "force_balance：買方力量對之後 30 天的效果 +11.6 點［+3.2，+16.3］，買方和賣方排名相關 0.96。"
        "協調者再依過去 30 天漲跌分五等份，五等份都是活躍比較高。",
        "所以這份研究的假設是事後才定的：「市場很活躍（很多幣換手）本身就帶有價格以外的訊息」。因為是看過資料才定的，"
        "M1、M2 是樣本內（就是看到它的那批資料），真正的檢查只有：從沒看過的 M3、M4、M5，以太幣，2022 年以後，和隨機錯位。"
        "說明文字和所有測試在跑任何結果之前就寫好（只先重做了協調者的表），看到結果後沒有改。",
        "名詞：活躍＝這個量測近 30 天的總量在過去一年裡排第 80 百分位以上（過去一年只有兩成的日子比它高）；安靜＝排在一半以下；"
        "中間＝其他。配對差＝在「過去 30 天漲跌」和「波動」都差不多的日子裡，活躍減安靜之後的平均漲跌。"
        "90% 區間＝重抽 1,000 次的中間九成，整段在零以上才算「排除零」。隨機錯位＝把活躍的日期整條挪開，看亂配也能得到一樣大差距的機率。",
        "M5 的說明：M3 本來就是用「顆」算的，不會因為價格漲就變大；M5 再除以流通量，主要是扣掉早年流通量快速增加的影響。"
        "會因為價格漲就變大的是 M4（美元成交金額），所以另外列了 M4b（同樣的幣安成交量用「顆」算），只描述，不算在事先定的條件裡。",
        "order_flow 已經用幣安的量試過：主動買進比例、累積買賣差、當天量對前 30 天平均、爆量、OBV、現貨比合約，"
        "都沒有變成能用的濾網（force_balance 的整理：看不出能用的前後關係）。它沒有試過「30 天成交金額和自己過去一年比」這種慢的狀態，就是這裡的 M4。",
        "資料：OBM（Open Bitcoin Metrics，CC BY 4.0）用的是 old_coin_selling 存下來的那份（下載時間 "
        f"{str(data['obm_meta'].get('fetched_utc', '—'))[:10]}），只用資料檔，沒有下載或執行它的任何程式；晚 7 天才用。"
        "Coin Metrics 社群版（免費、不用金鑰）：比特幣送進交易所的量和流通量，以太幣送進交易所的量（寫測試前確認過回 HTTP 200）；晚 1 天才用。"
        "幣安現貨日線的成交金額，收盤時就知道，不延遲。",
        "限制：幣動了不等於被賣了，交易所換錢包、託管機構整理也算。交易所地址標記是用今天的標籤往回套，早年的流入會偏少。"
        "幣安 2022-07-08 到 2023-03-22 比特幣免手續費，那段成交量被灌大，M4 會在那段偏「活躍」。"
        "相鄰的日子之後 60 天有大量重疊，天數不是獨立的證據，所以區間用 60 天一塊的區塊重抽。",
        "第 2 部分的其他資產（ETH、SOL、BNB、XRP、MSTR）沒有自己的活躍資料，用比特幣的活躍狀態當全市場狀態，配各自的 50 日均線和 20 日均線；"
        "MSTR 用前一天的狀態。A3 是調部位（有時只拿一半），一般人做法的「漏掉進場」對部位增加的部分照樣套用。",
        "門檻（30／365 天、第 80／50 百分位、延遲天數、20 日均線、20 天記憶、30 個配對格、60 天的區塊）都是跑之前憑道理定的，沒有試過其他數字。"
        "這是「賣壓／買盤／活躍」這一串的第六份研究，試越多次越容易碰巧有一次過；就算通過，也只代表可以放進每日影子紀錄觀察。",
        "過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {(time.time() - t_all) / 60:.1f} 分鐘（第 1 部分 {r['secs_desc'] / 60:.1f} 分鐘，考試與隨機錯位 "
        f"{(r['secs'] - r['secs_desc']) / 60:.1f} 分鐘）。",
    ]
    edge = [MSHORT[k] for k in MEAS if p1.get(k) and _ok(p1[k]["placebo"]) and abs(p1[k]["placebo"] - PLACEBO_BAR) < 1e-9]
    if edge:
        notes.append(f"（量到的）{'、'.join(edge)} 的隨機錯位比例剛好是 5.0%（200 次裡 10 次），正好壓在事先定的門檻上；"
                     "門檻寫的是「≤ 5%」，所以算符合，但這是最弱的那種符合。")
    neg = [MSHORT[k] for k in NEW if p1.get(k) and _ok(p1[k]["m"][MAIN_H][1][1]) and p1[k]["m"][MAIN_H][1][1] < 0]
    neg22 = [MSHORT[k] for k in NEW if p1.get(k) and "2022 起" in p1[k]["per"]
             and _ok(p1[k]["per"]["2022 起"]["m"][1][1]) and p1[k]["per"]["2022 起"]["m"][1][1] < 0]
    if neg or neg22:
        notes.append("（量到的）從沒看過的量測裡，方向相反（活躍之後比較差）而且區間整段在零以下的："
                     + (f"全期 {'、'.join(neg)}；" if neg else "")
                     + (f"2022 年以後 {'、'.join(neg22)}。" if neg22 else "")
                     + "（推測）鏈上花費總量大部分是放不到 155 天的幣，常常是交易所、找零和連續轉帳，和「老持有人交出幣」不是同一件事；"
                     "資料分不出哪個解釋對。")
    notes += problems

    sources = [{"name": f"Open Bitcoin Metrics：{oc.CN_ID[k]}（CSV，CC BY 4.0；用 old_coin_selling 存下來的那份）",
                "url": OBM_URL.format(oc.IDS[k])} for k in ("ge155", "lt155")]
    sources += [
        {"name": "Open Bitcoin Metrics 說明", "url": oc.OBM_REPO},
        {"name": "Coin Metrics 社群版 API：比特幣 FlowInExNtv、SplyCur",
         "url": CM_URL + "?assets=btc&metrics=FlowInExNtv,SplyCur&frequency=1d&start_time=2009-01-03&page_size=10000"},
        {"name": "Coin Metrics 社群版 API：以太幣 FlowInExNtv",
         "url": CM_URL + "?assets=eth&metrics=FlowInExNtv&frequency=1d&start_time=2009-01-03&page_size=10000"},
        {"name": "幣安現貨日線 K 線（BTCUSDT、ETHUSDT，成交金額與成交量）",
         "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]
    return {
        "id": "activity_state",
        "title": "市場活躍度：很多幣在換手的時候，之後是不是比較好",
        "question": ("三份研究都量到「賣方那邊很活躍之後反而比較好」。這是事後才看到的。這份研究把它當成假設來考："
                     "市場很活躍（很多幣在持有人之間換手）時，扣掉最近漲跌的影響後，之後一到三個月是不是還比較好？"
                     "換成從沒看過的量測、換成以太幣、只看 2022 年以後，還看得到嗎？能不能幫 50 日均線多賺？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
