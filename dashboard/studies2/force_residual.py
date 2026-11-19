"""Historical study "force_residual": separate each day's buying and selling into the SHARED part (how busy the market
is, "activity") and the UNMATCHED part (imbalance), at several time scales, and ask whether the unmatched part says
anything about the next 7-60 days - using every day, not sparse cells.

Why this study exists (stated plainly). force_balance measured a BUY force and a SELL force in USD over 30 days
(SELL = BTC deposited to exchanges; BUY = BTC withdrawn from exchanges + net new stablecoins + US spot ETF net
inflow) and found they move almost as one (rank correlation 0.96; 96% of SELL-HIGH days were also BUY-HIGH). Its 3x3
table could not answer the owner's question (how do buying and selling pressure compare?) because the cells that
matter were nearly empty (BUY high & SELL low: 2 days; SELL high & BUY low: 22 days). With BUY held fixed, the SELL
effect was -1.7 points [-6.0, +6.1]. The comparison of the two forces was left unfinished; this study finishes it.
The logic, stated up front: when the market is busy, both forces are large together; that shared part says how busy
the market is, not who is winning. What the owner asks about is the part of selling NOT matched by buying, and the
part of buying NOT matched by selling. So each day is split into ACTIVITY (the shared size) and IMBALANCE (the
difference relative to the size), at windows of 1, 7, 30 and 90 days.
Earlier studies in the supply/demand line (supply_demand, price_impact, reluctant_sellers, old_coin_selling,
force_balance, after_fall_forces; exit_forces is related): this is the SEVENTH, so a pass is only a candidate for the
daily shadow log, and the placebo bar is 5%.

Data (only free key-less endpoints, fixed user agent "btc-research-dashboard"; a refused endpoint is dropped and
named, never worked around; this study's own stores cache/_store/force_residual_*.json are a fallback only):
  BTC daily closes: the dashboard's (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Coin Metrics community API, asset btc: FlowInExUSD, FlowOutExUSD (USD value of BTC sent to / withdrawn from
    exchange-tagged addresses per day), FlowInExNtv, FlowOutExNtv (the same in coins), SplyExNtv (coins held on
    exchanges; used only to rebuild after_fall_forces' S8 for the overlap check).
  Asset eth (replication only, if the community API serves these four flow metrics for eth): FlowInExUSD,
    FlowOutExUSD, FlowInExNtv, FlowOutExNtv; ETH daily closes from Binance via main_signal_check's loader.
  DefiLlama total stablecoin supply (supply_demand._llama_fetch), daily USD change from 2019-07-01 with
    force_balance's coverage-jump rule (_stable_daily); 0 before its first value.
  Farside US spot bitcoin ETF daily net flow, Total column (flows2._etf_frame; fallback this study's store, then
    force_balance's / etf_loop's stores read only); 0 before 2024-01-11; force_balance's calendar-day rule (_etf_daily).
  Fair exam inputs as force_balance: Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance) and MSTR (Yahoo).
  Timing: every flow source is used one day late (data day D is used from the BTC close labelled D+1).

Daily inputs on data day D: buy(D) = FlowOutExUSD + stablecoin USD change + ETF net flow (BTC); for ETH buy(D) =
  FlowOutExUSD only (no stablecoin or ETH-ETF part: the ETH replication is the pure exchange-flow version).
  sell(D) = FlowInExUSD. out(D), in(D) = FlowOutExNtv, FlowInExNtv (coins).
Measures (fixed before running), each at windows W = 1, 7, 30, 90 days; BUY_W, SELL_W, OUT_W, IN_W = sums over
  D-W+1 .. D (all W values needed):
  ACT_W = log(BUY_W + SELL_W) (undefined if the sum is not positive)  ["activity": how big both sides are together].
  NET_W = (OUT_W - IN_W) / (OUT_W + IN_W), in coins, so price level cannot drive it [+1 all withdrawals, -1 all
    deposits: the pure exchange-flow imbalance].
  IMB_W = (BUY_W - SELL_W) / (BUY_W + SELL_W), in USD, with stablecoins and ETF flows on the buy side [the fuller
    imbalance] (undefined if the denominator is not positive).
  RES_W = log(SELL_W) - (a + b log(BUY_W)), with a, b from an ordinary least-squares fit of log SELL_W on log BUY_W
    over the 730 data days D-730 .. D-1 (at least 600 pairs; days with BUY_W <= 0 or SELL_W <= 0 are skipped) - no
    look-ahead. Positive = more selling than that day's buying would predict ("unmatched selling").
  RESB_W = the mirror: log(BUY_W) regressed on log(SELL_W) the same way; positive = unmatched buying.
  dIMB_W = IMB_W(D) - IMB_W(D-W) (is the balance turning?).
  rank = force_balance's trailing rank: share of the previous 365 data days' values (at least 330 present) below
    today's. Fifths (quintiles) of the rank: Q1 rank < 0.2, ..., Q5 rank >= 0.8 ("top fifth"); terciles: < 1/3,
    < 2/3, else. A decision day t (UTC close t, held during t+1) uses data day t-1.
Hypotheses fixed now by logic: forward returns RISE with NET, IMB, RESB and dIMB, and FALL with RES. ACT has no
  direction fixed (busy markets could go either way); it is tested two-sided.

Part 1 - who is winning, using all days. Forward return close[t+h]/close[t]-1, h = 7, 30, 60. For each measure and
  window: mean forward return by quintile, with 90% intervals from a circular block bootstrap of the sample days
  (60-day blocks, 1,000 draws, fixed seeds); top-minus-bottom = mean(Q5) - mean(Q1) (raw); MATCHED top-minus-bottom =
  Q5 versus Q1 inside cells of (decile of the trailing 30-day return close[t]/close[t-30]-1) x (tercile of the 20-day
  volatility = standard deviation of the last 20 daily log returns ending at t) x (ACT_30 tercile), weights = Q5 days
  per cell, cells holding both kinds only (share of Q5 days matched reported); decile and tercile edges of return and
  volatility from all decision days on which the three matching variables are defined. ACT itself is matched on
  return x volatility only. Sample of each measure-window: decision days with its quintile, the three matching
  variables and the forward return defined (raw and matched use the same days). By period 2012-16 / 2017-21 / 2022-now
  (bootstrap inside each period) and separately on days with ma50 on / off (main_signal_check's ma50; bootstrap of
  the full sample with the state mask travelling), report only.
  Placebo for each measure-window-horizon: its quintile series circularly shifted by k in [180, L-180] days over its
  own defined span (L days), 200 draws, prices and matching cells in place; p = (hits + 1) / 201 where a hit is a
  draw at least as far in the hypothesised direction as the real matched difference (ACT: |difference| at least as
  large).
  PRIMARY tests, fixed now, exactly four: NET_30, IMB_30, RES_30, dIMB_30 at the 60-day horizon, matched. A primary
  test "shows an effect" only if all three hold: the matched difference has the hypothesised sign; its 90% interval
  excludes zero; its placebo p survives Holm step-down across the four at family level 5%.
  Everything else (6 measures x 4 windows x 3 horizons) is a labelled map of the surface, report only: one compact
  grid (rows measure x horizon, columns window) of matched top-minus-bottom differences, marked where the 90% interval
  excludes zero; with no effect anywhere about 10% of cells would be marked by luck (cells are strongly correlated, so
  the luck count can swing widely); the number of cells with placebo p <= 0.05 is also counted (luck: about 5%).

Part 2 - the two forces side by side (report). On ONE common sample (days where ACT_30, NET_30, IMB_30, RES_30 and the
  matching variables are defined and the 60-day forward return exists) and the same bootstrap draws: ACTIVITY effect =
  ACT_30 Q5 minus Q1 matched on return x volatility; IMBALANCE effects = NET_30, IMB_30, RES_30 Q5 minus Q1 matched on
  return x volatility x ACT_30 tercile; each with its interval, at 7 / 30 / 60 days, so neither is used to explain
  away the other. Then the 2-D picture: ACT_30 tercile x NET_30 tercile (terciles of trailing rank): days, 60-day
  forward mean [interval] and median, share followed by a rise, mean trailing 30-day return.

Part 3 - around turns (description). Rallies from rally_sources._rallies (+80% / 30%) with its _describe: average rank
  (0-100) of ACT_30, NET_7, NET_30, RES_30 over the 60 days before, the 31 days centred on (t-15 .. t+15), and the 60
  days after each trough and each finished peak, pooled over events; reference band = 5th-95th percentile of the same
  pooled average for 500 sets of the same number of random days. At ma50 exits (whipsaw_anatomy._exits / _label:
  whipsaw = back on within 20 closes at a higher close; real = -15% before recovering): mean rank on the exit day and
  over the 20 closes ending at the exit, whipsaw versus real, with the 5th-95th percentile of the difference when the
  two labels are shuffled 1,000 times.

Part 4 - three rules, fixed now (flags at data day t-1; undefined counts as off):
  N1 = hold when ma50 is on OR (NET_30 in the top fifth AND close above its 20-day average) [holds more: coins leaving
    exchanges on balance].
  N2 = hold when ma50 is on AND NOT (RES_30 in the top fifth) [unmatched selling - can only avoid].
  N3 = hold when ma50 is on OR (ma50 was on at any of the 20 previous closes t-20 .. t-1 AND NET_7 in the top fifth)
    [not shaken out while coins are leaving exchanges].
  Fair exam = main_signal_check's, wired as force_balance's _Exam (first month 2015-01): BTC daily monthly starts
  2015-01 .. 2023-12 textbook 0.10%; clock starts 2018-01 .. 2023-12 x 24 closing hours textbook; the same cells as an
  ordinary person (act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock); ETH SOL BNB XRP
  MSTR with their OWN 50-day / 20-day averages (and own ma50 history for N3) and BTC's flag as a market-wide flag.
  ma50 re-scored on the same starts. Ranking score = worst of the four "beats holding" shares with
  main_signal_check's eligibility. "Supported", all three: ranking score >= ma50's; median deepest loss (clock cells)
  no more than 5 points worse than ma50's (N2: no worse at all); placebo share <= 5% (the rule's flag circularly
  shifted by >= 180 days, 200 draws, the whole exam rebuilt with 20 ordinary-person simulations per clock, the real
  rule re-scored the same way). Holm across the three placebo p-values ((hits + 1) / 201, family 5%); by luck about
  0.15 rules would pass a 5% placebo bar.
Also (report): how N1 relates to force_balance's F1 (BALANCE rank above 0.67 AND above the 20-day average) and
  after_fall_forces' S8 (exchange balance SplyExNtv lower than 30 data days earlier AND 30-day return <= 0): share of
  days each pair agrees, both-on / either-on, rank correlation of NET_30 and BALANCE, and how often "NET_30 > 0"
  agrees with "exchange balance fell over 30 days" (they should nearly coincide by construction). ETH replication of
  the four primary tests (ETH's own measures, matching and placebo; ETH's IMB is USD exchange flows only).
Look-ahead check: all inputs cut at 2016-06-30, 2020-06-30, 2024-06-30 (sources one day earlier); no quintile of the
  six measures used by the primaries and rules, and no rule flag, up to the cut may change.
Honesty: exchange flows use today's address labels applied backwards; internal exchange transfers inflate both forces
  together (which is why the shared part is separated out); OTC and, before 2024, off-exchange buying are invisible;
  stablecoins are not all used to buy BTC; BUY's composition changes in 2019 and 2024; thresholds (windows, 365 / 730
  days, fifths, terciles, 20-day average) were fixed by reasoning before the run, no alternatives tried. User-facing
  statements are labelled 量到的 (measured, with the number), 文獻說的 (source and whether opened) or 推測 (a guess);
  where the data cannot separate two explanations the text says so. Not advice.
Today: "比特幣：50 日均線｜N1｜N3" ("現金｜現金｜現金" style), N2, and today's ACT_30, NET_7, NET_30, IMB_30, RES_30 with
  their ranks in plain words.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import flows2
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import force_balance as fb
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import rally_sources as rs
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
DAY = pd.Timedelta(days=1)
CM_URL = sd.CM_URL
CM_BTC = ("FlowInExUSD", "FlowOutExUSD", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv")
CM_ETH = ("FlowInExUSD", "FlowOutExUSD", "FlowInExNtv", "FlowOutExNtv")
CM_START = {"btc": "2009-01-03", "eth": "2015-07-30"}
LLAMA = sd.LLAMA
FARSIDE_URL = flows2.FARSIDE["BTC"][0]
STORE_CM = {a: common.STORE / f"force_residual_cm_{a}.json" for a in ("btc", "eth")}
STORE_LL = common.STORE / "force_residual_llama.json"
STORE_ETF = common.STORE / "force_residual_etf.json"
ETF_READ_ONLY = (fb.STORE_ETF, fb.ETF_FALLBACK)

WINDOWS = (1, 7, 30, 90)
HORIZONS = (7, 30, 60)
MAIN_H = 60
REG_DAYS, REG_MIN, CENTER = 730, 600, 20.0
TOP = 0.8
BLOCK, BOOT = fb.BLOCK, fb.BOOT
DRAWS, GAP, PL_SIMS = 200, 180, 20
FAMILY, LOSS_SLACK, PLACEBO_BAR = 0.05, 5.0, 0.05
A20, RECENT = 20, 20
FIRST_MONTH = msc.DAILY_STARTS[0]
PERIODS = fb.PERIODS
CUTS = ("2016-06-30", "2020-06-30", "2024-06-30")
TURN_PRE, TURN_HALF, TURN_POST, TURN_RAND = 60, 15, 60, 500
EXIT_BEFORE, SHUFFLE = 20, 1000

MEASURES = ("ACT", "NET", "IMB", "RES", "RESB", "dIMB")
DIRN = {"ACT": 0, "NET": 1, "IMB": 1, "RES": -1, "RESB": 1, "dIMB": 1}
PRIMARY = (("NET", 30), ("IMB", 30), ("RES", 30), ("dIMB", 30))
CN = {"ACT": "活躍度", "NET": "提存差（顆數）", "IMB": "買賣差（美元）", "RES": "多出來的賣方", "RESB": "多出來的買方",
      "dIMB": "買賣差的變化"}
KI = {"ma50": 1, "n1": 161, "n2": 162, "n3": 163}
RULES = ("n1", "n2", "n3")
NAME = {"ma50": "50 日均線（基準）",
        "n1": "N1：50 日均線，或「30 天提存差在過去一年前五分之一＋站上 20 日均線」也持有",
        "n2": "N2：50 日均線，但「30 天多出來的賣方在過去一年前五分之一」時不持有",
        "n3": "N3：50 日均線，或「20 天內均線還在持有＋7 天提存差在前五分之一」繼續持有"}
SHORT = {"ma50": "50 日均線", "n1": "N1", "n2": "N2", "n3": "N3"}
LV3 = ("低", "中", "高")

_d = cr._d
_ok, _p, _pt, _pc, _ci, _share, _share1 = sd._ok, sd._p, sd._pt, sd._pc, sd._ci, sd._share, sd._share1
_r = fb._r


def _cid(lo, hi) -> str:
    return _ci(lo, hi, f=_pt)


def _mname(m: str, w: int) -> str:
    return f"{m}_{w}"


# ---------- data ----------

def _cm_fetch(asset: str) -> dict:
    metrics = CM_BTC if asset == "btc" else CM_ETH
    reply = common.get_json(CM_URL, {"assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
                                     "start_time": CM_START[asset], "page_size": 10000}, timeout=90)
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
            if math.isfinite(v):
                rec[m] = v
        if day and rec:
            out[day] = rec
    if len(out) < 1500:
        raise RuntimeError(f"Coin Metrics {asset} 只回傳 {len(out)} 天")
    return out


def _cm_frame(raw: dict, today: pd.Timestamp) -> pd.DataFrame:
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    return cm[cm.index < today].asfreq("D")


def _etf_load() -> tuple[pd.Series, str | None]:
    """Farside BTC ETF Total net flow per US trading day (USD); this study's store, then others' (read only)."""
    try:
        tot = flows2._etf_frame("BTC")["Total"].astype(float)
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_ETF.write_text(json.dumps({str(k): float(v) for k, v in tot.items()}), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        prob = None
    except Exception as exc:  # noqa: BLE001 - blocked: stored data, never worked around
        prob = f"這次 Farside 取不到（{type(exc).__name__}），用之前存下來的資料。"
        tot = None
        if STORE_ETF.exists():
            tot = pd.Series(json.loads(STORE_ETF.read_text(encoding="utf-8")), dtype=float)
        elif ETF_READ_ONLY[0].exists():
            tot = pd.Series(json.loads(ETF_READ_ONLY[0].read_text(encoding="utf-8")), dtype=float)
        elif ETF_READ_ONLY[1].exists():
            f = pd.read_json(io.StringIO(ETF_READ_ONLY[1].read_text(encoding="utf-8")), orient="split")
            f.index = [str(i)[:10] for i in f.index]
            tot = f["Total"].astype(float)
        if tot is None:
            raise RuntimeError("Farside ETF 流量取不到，也沒有存檔") from exc
    tot.index = pd.to_datetime(tot.index)
    tot = tot[~tot.index.duplicated(keep="last")].sort_index()
    tot = tot[tot.index < pd.Timestamp(common.today_utc())]
    return tot * 1e6, prob


def _load() -> dict:
    problems = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    raw, n = sd._stored(STORE_CM["btc"], lambda: _cm_fetch("btc"))
    problems += [n] if n else []
    cm = _cm_frame(raw, today)
    ll, n = sd._stored(STORE_LL, sd._llama_fetch)
    problems += [n] if n else []
    sup = pd.Series(ll, dtype=float)
    sup.index = pd.to_datetime(sup.index)
    sup = sup.sort_index()
    sup = sup[sup.index < today].asfreq("D").ffill()
    etf, n = _etf_load()
    problems += [n] if n else []
    other, more = msc._load_other()
    problems += more
    cm_eth, eth_note = None, None
    try:
        raw_e, n = sd._stored(STORE_CM["eth"], lambda: _cm_fetch("eth"))
        problems += [n] if n else []
        cm_eth = _cm_frame(raw_e, today)
        if not all(m in cm_eth for m in CM_ETH):
            eth_note = "Coin Metrics 社群 API 沒有回傳以太幣全部四個流量欄位，以太幣重做沒有算。"
            cm_eth = None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        eth_note = f"Coin Metrics 以太幣交易所流量取不到（{type(exc).__name__}），以太幣重做沒有算。"
    if eth_note:
        problems.append(eth_note)
    return {"close": close, "cm": cm, "sup": sup, "etf": etf, "other": other, "cm_eth": cm_eth,
            "problems": problems}


# ---------- the measures ----------

def _resid(y: pd.Series, x: pd.Series) -> pd.Series:
    """y - (a + b x) with a, b fitted by OLS on the REG_DAYS data days before today (no look-ahead)."""
    ok = y.notna() & x.notna()
    xc = (x - CENTER).where(ok, 0.0)
    yc = (y - CENTER).where(ok, 0.0)

    def roll(s):
        return s.rolling(REG_DAYS, min_periods=1).sum().shift(1)

    n = roll(ok.astype(float))
    sx, sy, sxx, sxy = roll(xc), roll(yc), roll(xc * xc), roll(xc * yc)
    den = n * sxx - sx ** 2
    with np.errstate(all="ignore"):
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        r = (y - CENTER) - a - b * (x - CENTER)
    return r.where((n >= REG_MIN) & (den > 1e-9) & ok)


def _daily_buy(cm: pd.DataFrame, sup: pd.Series | None, etf: pd.Series | None) -> tuple[pd.Series, object]:
    idx = cm.index
    buy = cm["FlowOutExUSD"].copy()
    first_stable = None
    if sup is not None:
        st = fb._stable_daily(sup).reindex(idx)
        first_stable = st.first_valid_index()
        if first_stable is None:
            st = pd.Series(0.0, index=idx)
        else:
            st[idx < first_stable] = 0.0
        buy = buy + st
    if etf is not None:
        ed = fb._etf_daily(etf, idx[-1])
        pre = pd.Series(0.0, index=idx[idx < fb.ETF_FROM])
        ed = pd.concat([pre, ed[ed.index >= fb.ETF_FROM]]).reindex(idx)
        buy = buy + ed
    return buy, first_stable


def _series(cm: pd.DataFrame, sup=None, etf=None) -> dict:
    """Data-day raw measures, sums and trailing ranks for every window."""
    buy_d, first_stable = _daily_buy(cm, sup, etf)
    sell_d = cm["FlowInExUSD"]
    raw, rank, sums = {}, {}, {}
    for w in WINDOWS:
        def s(x):
            return x.rolling(w, min_periods=w).sum()
        B, S, O, I = s(buy_d), s(sell_d), s(cm["FlowOutExNtv"]), s(cm["FlowInExNtv"])
        tot, totn = B + S, O + I
        with np.errstate(all="ignore"):
            act = np.log(tot.where(tot > 0))
            net = ((O - I) / totn).where(totn > 0)
            imb = ((B - S) / tot).where(tot > 0)
            lb, ls = np.log(B.where(B > 0)), np.log(S.where(S > 0))
        vals = {"ACT": act, "NET": net, "IMB": imb, "RES": _resid(ls, lb), "RESB": _resid(lb, ls),
                "dIMB": imb - imb.shift(w)}
        for m, v in vals.items():
            raw[(m, w)] = v
            rank[(m, w)] = fb._trail_rank(v)
        sums[w] = {"buy": B, "sell": S, "out": O, "in": I}
    return {"raw": raw, "rank": rank, "sums": sums, "first_stable": first_stable}


def _dec(s: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Decision-day values: data day t-1."""
    return s.reindex(index - DAY).values.astype(float)


def _q5(rank: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(rank), np.clip(np.floor(np.nan_to_num(rank) * 5), 0, 4), -1).astype(int)


def _q3(rank: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(rank)
    return np.where(np.isfinite(rank), (r >= 1 / 3).astype(int) + (r >= 2 / 3).astype(int), -1)


def _frame(close: pd.Series, ser: dict) -> dict:
    """Everything on decision days for one asset: ranks, quintiles, matching cells, forward returns."""
    idx = close.index
    c = close.values.astype(float)
    n = len(c)
    rk = {k: _dec(v, idx) for k, v in ser["rank"].items()}
    q = {k: _q5(v) for k, v in rk.items()}
    past = np.full(n, np.nan)
    past[30:] = c[30:] / c[:-30] - 1
    vol = np.log(close).diff().rolling(20, min_periods=20).std().values
    act = rk[("ACT", 30)]
    base = np.isfinite(past) & np.isfinite(vol) & np.isfinite(act)
    edges = np.quantile(past[base], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, past, side="right") - 1, 0, 9)
    vedge = np.quantile(vol[base], [1 / 3, 2 / 3])
    vt = np.searchsorted(vedge, vol, side="right")
    at = _q3(act)
    cell3 = np.where(base, dec * 9 + vt * 3 + at, -1)
    cell2 = np.where(base, dec * 3 + vt, -1)
    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        fwd[h] = f
    return {"idx": idx, "c": c, "rk": rk, "q": q, "past": past, "vol": vol, "cell3": cell3, "cell2": cell2,
            "fwd": fwd, "a20": sd._above20(c), "ma": msc._targets(close)["ma50"].values.astype(float)}


def _cell(F: dict, m: str) -> tuple[np.ndarray, int]:
    return (F["cell2"], 30) if m == "ACT" else (F["cell3"], 90)


# ---------- statistics ----------

def _gsum(v2: np.ndarray, g2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    v2, g2 = np.atleast_2d(v2), np.atleast_2d(g2)
    R = v2.shape[0]
    ok = (g2 >= 0) & np.isfinite(v2)
    key = (np.arange(R)[:, None] * k + g2)[ok]
    s = np.bincount(key, weights=v2[ok], minlength=R * k).reshape(R, k)
    c = np.bincount(key, minlength=R * k).reshape(R, k).astype(float)
    return s, c


def _means(v2, g2, k):
    s, c = _gsum(v2, g2, k)
    with np.errstate(all="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan), c


def _matched(v2, q2, cell2, K: int, hi: int = 4, lo: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Group hi minus group lo inside matching cells, weighted by hi-days per cell (cells holding both only)."""
    q2, cell2 = np.atleast_2d(q2), np.atleast_2d(cell2)
    st, ct = _gsum(v2, np.where(q2 == hi, cell2, -1), K)
    sc, cc = _gsum(v2, np.where(q2 == lo, cell2, -1), K)
    use = (ct > 0) & (cc > 0)
    with np.errstate(all="ignore"):
        d = np.where(use, ct * (st / np.maximum(ct, 1) - sc / np.maximum(cc, 1)), 0.0).sum(axis=1)
        w = np.where(use, ct, 0.0).sum(axis=1)
        return np.where(w > 0, d / np.maximum(w, 1), np.nan), np.where(ct.sum(axis=1) > 0, w / ct.sum(axis=1), np.nan)


def _analyse(q, v, cell, K, sample, rng, sub=None, groups: int = 5) -> dict | None:
    """Group means, top-minus-bottom (raw and matched) with block-bootstrap intervals over the sample days."""
    P = np.flatnonzero(sample)
    if len(P) < 2 * BLOCK:
        return None
    B = P[fb._boot_idx(len(P), rng)]
    qq = q if sub is None else np.where(sub, q, -1)
    hi = groups - 1
    mr, cnt = _means(v[P], qq[P], groups)
    mb, _ = _means(v[B], qq[B], groups)
    tr, tb = mr[0, hi] - mr[0, 0], mb[:, hi] - mb[:, 0]
    xr, share = _matched(v[P], qq[P], cell[P], K, hi, 0)
    xb, _ = _matched(v[B], qq[B], cell[B], K, hi, 0)
    with np.errstate(all="ignore"):
        med = [float(np.median(v[P][qq[P] == j])) if (qq[P] == j).any() else None for j in range(groups)]
        up = [float((v[P][qq[P] == j] > 0).mean()) if (qq[P] == j).any() else None for j in range(groups)]
    return {"n": [int(x) for x in cnt[0]], "mean": [float(x) if np.isfinite(x) else None for x in mr[0]],
            "mean_ci": [pi._q(mb[:, j]) for j in range(groups)], "med": med, "up": up,
            "raw": float(tr) if np.isfinite(tr) else None, "raw_ci": pi._q(tb),
            "mt": float(xr[0]) if np.isfinite(xr[0]) else None, "mt_ci": pi._q(xb),
            "share": float(share[0]) if np.isfinite(share[0]) else None, "days": int(len(P)),
            "from": int(P[0]), "to": int(P[-1]), "boot": {"mean": mb, "mt": xb}}


def _placebo(q, v, cell, K, base, real, dirn, rng) -> int:
    """Hits among DRAWS circular shifts (>= GAP days) of the quintile series over its defined span."""
    if real is None:
        return DRAWS
    dfn = np.flatnonzero(q >= 0)
    f0 = int(dfn[0])
    L = len(q) - f0
    hits = 0
    for k in rng.integers(GAP, L - GAP + 1, DRAWS):
        qs = q.copy()
        qs[f0:] = np.roll(q[f0:], int(k))
        P = np.flatnonzero(base & (qs >= 0))
        d = _matched(v[P], qs[P], cell[P], K)[0][0]
        if not np.isfinite(d):
            continue
        hits += int(d >= real if dirn > 0 else d <= real if dirn < 0 else abs(d) >= abs(real))
    return hits


def _holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    m = len(keys)
    out, run, stop = {}, 0.0, False
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * p[k]))
        if stop or p[k] > FAMILY / (m - j):
            stop = True
        out[k] = (run, not stop)
    return out


def _excl(ci) -> int:
    lo, hi = ci
    if not (_ok(lo) and _ok(hi)):
        return 0
    return 1 if lo > 0 else -1 if hi < 0 else 0


# ---------- Part 1 ----------

def _part1(F: dict, seed: int, keys=None, horizons=HORIZONS, placebo: bool = True) -> dict:
    out = {}
    keys = keys or [(m, w) for m in MEASURES for w in WINDOWS]
    for i, (m, w) in enumerate(keys):
        q = F["q"][(m, w)]
        cell, K = _cell(F, m)
        for h in horizons:
            v = F["fwd"][h]
            base = np.isfinite(v) & (F["cell3"] >= 0) & (cell >= 0)
            rng = np.random.default_rng(seed + 100 * i + h)
            a = _analyse(q, v, cell, K, base & (q >= 0), rng)
            if a is None:
                continue
            if placebo:
                a["hits"] = _placebo(q, v, cell, K, base, a["mt"], DIRN[m], np.random.default_rng(seed + 7 + 100 * i + h))
                a["p"] = (a["hits"] + 1) / (DRAWS + 1)
            out[(m, w, h)] = a
    return out


def _splits(F: dict, seed: int) -> dict:
    """Primary measures and ACT_30 at the main horizon: by period and by ma50 state (report)."""
    idx = F["idx"]
    on = np.nan_to_num(F["ma"], nan=-1.0)
    out = {}
    for i, (m, w) in enumerate(list(PRIMARY) + [("ACT", 30)]):
        q = F["q"][(m, w)]
        cell, K = _cell(F, m)
        v = F["fwd"][MAIN_H]
        base = np.isfinite(v) & (F["cell3"] >= 0) & (cell >= 0) & (q >= 0)
        for lab, a, b in PERIODS:
            mk = base & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
            out[(m, w, lab)] = _analyse(q, v, cell, K, mk, np.random.default_rng(seed + 50 * i + len(lab)))
        rng = np.random.default_rng(seed + 900 + i)
        out[(m, w, "on")] = _analyse(q, v, cell, K, base, rng, sub=on > 0.5)
        out[(m, w, "off")] = _analyse(q, v, cell, K, base, np.random.default_rng(seed + 950 + i), sub=on == 0)
    return out


# ---------- Part 2 ----------

def _part2(F: dict, seed: int) -> dict:
    keys = [("ACT", 30), ("NET", 30), ("IMB", 30), ("RES", 30)]
    common_ = (F["cell3"] >= 0) & np.isfinite(F["fwd"][MAIN_H])
    for k in keys:
        common_ &= F["q"][k] >= 0
    P = np.flatnonzero(common_)
    B = P[fb._boot_idx(len(P), np.random.default_rng(seed))]
    side = {}
    for h in HORIZONS:
        v = F["fwd"][h]
        for m, w in keys:
            cell, K = _cell(F, m)
            q = F["q"][(m, w)]
            xr, share = _matched(v[P], q[P], cell[P], K)
            xb, _ = _matched(v[B], q[B], cell[B], K)
            mr, _ = _means(v[P], q[P], 5)
            mb, _ = _means(v[B], q[B], 5)
            side[(m, h)] = {"mt": float(xr[0]), "mt_ci": pi._q(xb), "raw": float(mr[0, 4] - mr[0, 0]),
                            "raw_ci": pi._q(mb[:, 4] - mb[:, 0]), "share": float(share[0])}
    # 2-D: ACT_30 tercile x NET_30 tercile
    a3, n3 = _q3(F["rk"][("ACT", 30)]), _q3(F["rk"][("NET", 30)])
    g = np.where((a3 >= 0) & (n3 >= 0), 3 * a3 + n3, -1)
    v = F["fwd"][MAIN_H]
    mr, cnt = _means(v[P], g[P], 9)
    mb, _ = _means(v[B], g[B], 9)
    cells = []
    for j in range(9):
        sel = P[g[P] == j]
        cells.append({"n": int(cnt[0, j]), "mean": float(mr[0, j]) if cnt[0, j] else None, "ci": pi._q(mb[:, j]),
                      "med": float(np.median(v[sel])) if len(sel) else None,
                      "up": float((v[sel] > 0).mean()) if len(sel) else None,
                      "past": float(np.nanmean(F["past"][sel])) if len(sel) else None})
    rkA, rkN, rkR = (F["rk"][k][P] for k in (("ACT", 30), ("NET", 30), ("RES", 30)))
    corr = {"act_net": float(pd.Series(rkA).corr(pd.Series(rkN), method="spearman")),
            "act_res": float(pd.Series(rkA).corr(pd.Series(rkR), method="spearman")),
            "net_res": float(pd.Series(rkN).corr(pd.Series(rkR), method="spearman"))}
    return {"side": side, "cells": cells, "days": len(P), "from": F["idx"][P[0]], "to": F["idx"][P[-1]], "corr": corr}


# ---------- Part 3 ----------

def _win_sums(rank: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """For every centre e: sum and count of finite rank over e+a .. e+b (clipped)."""
    n = len(rank)
    fin = np.isfinite(rank)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, rank, 0.0))])
    cc = np.concatenate([[0], np.cumsum(fin)])
    e = np.arange(n)
    lo, hi = np.clip(e + a, 0, n), np.clip(e + b + 1, 0, n)
    return cs[hi] - cs[lo], (cc[hi] - cc[lo]).astype(float)


TURN_KEYS = (("ACT", 30), ("NET", 7), ("NET", 30), ("RES", 30))
TURN_WINS = (("前 60 天", -TURN_PRE, -1), ("前後 15 天（共 31 天）", -TURN_HALF, TURN_HALF), ("後 60 天", 1, TURN_POST))


def _part3(close: pd.Series, F: dict, seed: int) -> dict:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    troughs = [r["it"] for r in rallies]
    peaks = [r["ip"] for r in rallies if r["done"]]
    rng = np.random.default_rng(seed)
    n = len(close)
    ok_all = np.all([np.isfinite(F["rk"][k]) for k in TURN_KEYS], axis=0)
    first = int(np.argmax(ok_all))
    rows = {}
    for kind, ev in (("低點", troughs), ("高點", peaks)):
        for lab, a, b in TURN_WINS:
            ev_ok = [e for e in ev if e + a >= first]
            cells = []
            for k in TURN_KEYS:
                s, c = _win_sums(F["rk"][k], a, b)
                real = s[ev_ok].sum() / max(c[ev_ok].sum(), 1) if ev_ok else np.nan
                pool = np.arange(max(first - a, 0), n - max(b, 0))
                draws = []
                for _ in range(TURN_RAND):
                    pick = rng.choice(pool, len(ev_ok), replace=False) if len(ev_ok) else []
                    draws.append(s[pick].sum() / max(c[pick].sum(), 1))
                cells.append((float(real), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))))
            rows[(kind, lab)] = {"n": len(ev_ok), "cells": cells}
    avg = [float(np.nanmean(F["rk"][k][first:])) for k in TURN_KEYS]
    # ma50 exits: whipsaw vs real
    c, ma = F["c"], F["ma"]
    ex = [int(e) for e in wa._exits(ma) if e >= first]
    lab = {e: wa._label(c, ma, e) for e in ex}
    whip = [e for e in ex if lab[e] == "whip"]
    real = [e for e in ex if lab[e] == "real"]
    exits = {}
    for k in TURN_KEYS:
        rk = F["rk"][k]
        s20, c20 = _win_sums(rk, -(EXIT_BEFORE - 1), 0)
        for tag, val in (("day", rk), ("pre", np.where(c20 > 0, s20 / np.maximum(c20, 1), np.nan))):
            w_, r_ = val[whip], val[real]
            w_, r_ = w_[np.isfinite(w_)], r_[np.isfinite(r_)]
            both = np.concatenate([w_, r_])
            diffs = []
            for _ in range(SHUFFLE):
                pm = rng.permutation(both)
                diffs.append(pm[:len(w_)].mean() - pm[len(w_):].mean())
            exits[(k, tag)] = {"whip": float(w_.mean()) if len(w_) else None, "real": float(r_.mean()) if len(r_) else None,
                               "nw": len(w_), "nr": len(r_),
                               "diff": float(w_.mean() - r_.mean()) if len(w_) and len(r_) else None,
                               "band": (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)))}
    counts = {t: sum(1 for e in ex if lab[e] == t) for t in ("whip", "real", "other", "unknown")}
    return {"rows": rows, "avg": avg, "n_rally": len(rallies), "n_peak": len(peaks), "exits": exits,
            "counts": counts, "first": close.index[first]}


# ---------- Part 4: the fair exam ----------

def _recent(ma: np.ndarray) -> np.ndarray:
    s = pd.Series(np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0))
    return s.shift(1).rolling(RECENT, min_periods=1).max().fillna(0.0).values


def _combine(kind: str, ma, a20, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "n1":
        return np.maximum(ma, f * a20)
    if kind == "n2":
        return ma * (1.0 - f)
    if kind == "n3":
        return np.maximum(ma, _recent(ma) * f)
    raise ValueError(kind)


class _Exam(fb._Exam):
    """force_balance's exam with this study's three ways to combine the flag."""

    def btc_pos(self, flag, kind: str = "ma50") -> np.ndarray:
        return _combine(kind, self.ma, self.a20, self._at(flag, self.close.index, 0))

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
            w = _combine(kind, self.cma[h], self.ca20[h], f)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            w = _combine(kind, o["ma"], o["a20"], self._at(flag, o["close"].index, o["lag"]))
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


def _flags(F: dict) -> dict:
    idx = F["idx"]
    mk = {"n1": ("NET", 30), "n2": ("RES", 30), "n3": ("NET", 7)}
    out = {}
    for k, key in mk.items():
        r = F["rk"][key]
        out[k] = pd.Series(np.where(np.isfinite(r), (np.nan_to_num(r) >= TOP).astype(float), np.nan), index=idx)
    return out


def _exam(data: dict, F: dict) -> dict:
    close = data["close"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(close, data["other"], price, b0, pre, pd.Timestamp(FIRST_MONTH))
    fl = _flags(F)
    s = {"ma50": ex.summary(None, KI["ma50"], SIMS, "ma50")}
    small, plc, lows = {}, {}, {}
    for k in RULES:
        f = fl[k].fillna(0.0)
        s[k] = ex.summary(f, KI[k], SIMS, k)
        small[k] = ex.summary(f, KI[k], PL_SIMS, k)
        fstart = fl[k].first_valid_index()
        n = int((close.index >= fstart).sum())
        prng = np.random.default_rng(SEED + 31 + KI[k])
        hits, lw = 0, []
        for kk in prng.integers(GAP, n - GAP, DRAWS):
            x = ex.summary(sd._roll(f, fstart, int(kk)), KI[k], PL_SIMS, k)
            lw.append(x["low"])
            hits += int(sd._score_ge(x, small[k]))
        plc[k] = hits
        lows[k] = float(np.median(lw))
    pos = {k: ex.btc_pos(fl[k].fillna(0.0) if k != "ma50" else None, k) for k in ("ma50",) + RULES}
    m15 = close.index >= pd.Timestamp(FIRST_MONTH)
    tim = {k: float(np.mean(pos[k][m15])) for k in pos}
    differ = {k: int((pos[k][m15] != pos["ma50"][m15]).sum()) for k in RULES}
    return {"s": s, "small": small, "plc": plc, "plc_med": lows, "pos": pos, "tim": tim, "differ": differ,
            "flags": fl, "ex": ex, "problem": problem}


# ---------- overlap with F1 and S8 ----------

def _overlap(data: dict, F: dict, fl: dict) -> dict:
    close = data["close"]
    idx = close.index
    lg = fb._legs(fb._forces(data["cm"], data["sup"], data["etf"]), idx)
    a20 = F["a20"]
    rbal = lg["rbal"].values
    n1x = np.where(np.isfinite(F["rk"][("NET", 30)]), fl["n1"].fillna(0).values * a20, np.nan)
    f1x = np.where(np.isfinite(rbal), (np.nan_to_num(rbal) > fb.HI_Q).astype(float) * a20, np.nan)
    bal = data["cm"]["SplyExNtv"] if "SplyExNtv" in data["cm"] else None
    s8 = np.full(len(idx), np.nan)
    s8force = np.full(len(idx), np.nan)
    if bal is not None:
        ch = bal - bal.shift(30)
        fo = _dec((ch < 0).astype(float).where(ch.notna()), idx)
        s8force = fo
        with np.errstate(invalid="ignore"):
            s8 = np.where(np.isfinite(fo) & np.isfinite(F["past"]), (fo > 0.5) & (F["past"] <= 0), np.nan).astype(float)
    m15 = idx >= pd.Timestamp(FIRST_MONTH)

    def pair(a, b):
        ok = m15 & np.isfinite(a) & np.isfinite(b)
        x, y = a[ok] > 0.5, b[ok] > 0.5
        either = int((x | y).sum())
        return {"n": int(ok.sum()), "agree": float((x == y).mean()), "both": int((x & y).sum()), "either": either,
                "a": int(x.sum()), "b": int(y.sum())}

    net30 = _dec(F["raw"][("NET", 30)], idx)
    ok = m15 & np.isfinite(net30) & np.isfinite(s8force)
    sign = float(((net30[ok] > 0) == (s8force[ok] > 0.5)).mean()) if ok.any() else None
    okr = m15 & np.isfinite(F["rk"][("NET", 30)]) & np.isfinite(rbal)
    corr = float(pd.Series(F["rk"][("NET", 30)][okr]).corr(pd.Series(rbal[okr]), method="spearman"))
    pos_n1 = _combine("n1", F["ma"], a20, fl["n1"].fillna(0).values)
    pos_f1 = fb._combine("f1", F["ma"], a20, lg["hi"].values)
    return {"n1_f1": pair(n1x, f1x), "n1_s8": pair(n1x, s8), "f1_s8": pair(f1x, s8), "sign": sign, "sign_n": int(ok.sum()),
            "corr": corr, "pos_diff": int((pos_n1[m15] != pos_f1[m15]).sum()), "m15": int(m15.sum()),
            "has_s8": bal is not None}


# ---------- ETH replication ----------

def _eth(data: dict) -> dict | None:
    cm = data["cm_eth"]
    if cm is None or "ETH" not in data["other"]:
        return None
    close = data["other"]["ETH"][0]
    ser = _series(cm)
    F = _frame(close, ser)
    res = _part1(F, SEED + 7000, keys=list(PRIMARY), horizons=(MAIN_H,))
    return {"res": res, "first": close.index[0], "cm_first": cm["FlowInExNtv"].first_valid_index(), "idx": F["idx"]}


# ---------- look-ahead ----------

def _lookahead(data: dict, F: dict) -> tuple[int, int]:
    keys = list(PRIMARY) + [("NET", 7), ("ACT", 30)]
    total = bad = 0
    fl = _flags(F)
    for cut in CUTS:
        c = pd.Timestamp(cut)
        cm = data["cm"][data["cm"].index <= c - DAY]
        sup = data["sup"][data["sup"].index <= c - DAY]
        etf = data["etf"][data["etf"].index <= c - DAY]
        cl = data["close"][data["close"].index <= c]
        ser = _series(cm, sup, etf)
        nidx = len(cl)
        for k in keys:
            q2 = _q5(_dec(ser["rank"][k], cl.index))
            total += 1
            bad += int(not np.array_equal(q2, F["q"][k][:nidx]))
        for k, key in (("n1", ("NET", 30)), ("n2", ("RES", 30)), ("n3", ("NET", 7))):
            r = _dec(ser["rank"][key], cl.index)
            f2 = np.where(np.isfinite(r), (np.nan_to_num(r) >= TOP).astype(float), np.nan)
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(f2, nan=-1), np.nan_to_num(fl[k].values[:nidx], nan=-1)))
    return total, bad


# ---------- added after seeing the results (description only, not a test) ----------

def _act_check(close: pd.Series, F: dict, ser: dict) -> dict:
    """How much ACT_30 (USD) moves with longer past price change, and with activity counted in coins."""
    idx = close.index
    c = F["c"]
    p180 = np.full(len(c), np.nan)
    p180[180:] = c[180:] / c[:-180] - 1
    s30 = ser["sums"][30]
    coins = np.log((s30["out"] + s30["in"]).where((s30["out"] + s30["in"]) > 0))
    rk_coin = _dec(fb._trail_rank(coins), idx)
    act = F["rk"][("ACT", 30)]

    def sp_(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")), int(ok.sum())
    return {"p180": sp_(act, p180), "p30": sp_(act, F["past"]), "coin": sp_(act, rk_coin)}


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    ser = _series(data["cm"], data["sup"], data["etf"])
    F = _frame(close, ser)
    F["raw"] = ser["raw"]
    p1 = _part1(F, SEED + 5000)
    sp = _splits(F, SEED + 5500)
    p2 = _part2(F, SEED + 6000)
    p3 = _part3(close, F, SEED + 6500)
    eth = _eth(data)
    look = _lookahead(data, F)
    ac = _act_check(close, F, ser)
    t1 = time.time()
    ex = _exam(data, F)
    ov = _overlap(data, F, ex["flags"])
    return {"ser": ser, "F": F, "p1": p1, "sp": sp, "p2": p2, "p3": p3, "eth": eth, "look": look, "x": ex, "ac": ac,
            "ov": ov, "secs": time.time() - t0, "secs_exam": time.time() - t1}


# ---------- today ----------

def _busy(r) -> str:
    if not _ok(r):
        return "資料不足"
    return ("很忙（過去一年排前五分之一）" if r >= 0.8 else "偏忙" if r >= 0.6 else "普通" if r >= 0.4
            else "偏冷清" if r >= 0.2 else "很冷清（過去一年排後五分之一）")


def _lvl(r) -> str:
    if not _ok(r):
        return "資料不足"
    return ("在過去一年裡算很高（前五分之一）" if r >= 0.8 else "偏高" if r >= 0.6 else "中間" if r >= 0.4
            else "偏低" if r >= 0.2 else "很低（後五分之一）")


def _now(data: dict, r: dict) -> list[dict]:
    close, F, x = data["close"], r["F"], r["x"]
    last = close.index[-1]
    word = {1: "持有", 0: "現金"}
    st = {k: int(x["pos"][k][-1] > 0.5) for k in ("ma50",) + RULES}
    raw = r["ser"]["raw"]
    day = last - DAY

    def val(k):
        v = raw[k].get(day, np.nan) if day in raw[k].index else np.nan
        return float(v) if _ok(v) else None

    def rk(k):
        v = F["rk"][k][-1]
        return float(v) if _ok(v) else None

    n7, n30, i30, r30 = val(("NET", 7)), val(("NET", 30)), val(("IMB", 30)), val(("RES", 30))
    s30 = r["ser"]["sums"][30]
    b, s = s30["buy"].get(day, np.nan), s30["sell"].get(day, np.nan)
    c = close.values.astype(float)

    def side(v):
        return "資料不足" if v is None else ("提走比存入多" if v > 0 else "存入比提走多" if v < 0 else "一樣多")

    return [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤；流量用 {_d(day)} 的資料）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
        {"label": "比特幣：50 日均線｜N1｜N3", "value": f"{word[st['ma50']]}｜{word[st['n1']]}｜{word[st['n3']]}"},
        {"label": "N2（只會少拿的過濾規則）", "value": word[st["n2"]]},
        {"label": "活躍度（近 30 天買賣兩邊加起來）｜在過去一年的排名（0–100）",
         "value": (f"{fb._usd(b + s) if _ok(b) and _ok(s) else '—'}｜{_r(rk(('ACT', 30)))}：今天市場{_busy(rk(('ACT', 30)))}")},
        {"label": "7 天提存差（顆數）｜排名",
         "value": f"{n7:+.3f}｜{_r(rk(('NET', 7)))}：近 7 天{side(n7)}，{_lvl(rk(('NET', 7)))}" if n7 is not None else "資料不足"},
        {"label": "30 天提存差（顆數）｜排名",
         "value": f"{n30:+.3f}｜{_r(rk(('NET', 30)))}：近 30 天{side(n30)}，{_lvl(rk(('NET', 30)))}" if n30 is not None else "資料不足"},
        {"label": "30 天買賣差（美元，含穩定幣與 ETF）｜排名",
         "value": (f"{i30:+.3f}｜{_r(rk(('IMB', 30)))}：{'買方' if i30 > 0 else '賣方'}比較大，{_lvl(rk(('IMB', 30)))}"
                   if i30 is not None else "資料不足")},
        {"label": "30 天多出來的賣方｜排名",
         "value": (f"{r30:+.3f}｜{_r(rk(('RES', 30)))}：賣方比「照這樣的買方該有的量」"
                   f"{'多' if r30 > 0 else '少'}，{_lvl(rk(('RES', 30)))}" if r30 is not None else "資料不足")},
        {"label": "這幾格的意思", "value": "只描述最後一根日線收盤後的狀態，不是買賣建議；N1、N2、N3 沒有通過考試前不會用"},
    ]


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, F, p1, sp, p2, p3, x, ov = (data["close"], r["F"], r["p1"], r["sp"], r["p2"], r["p3"], r["x"], r["ov"])
    problems = data["problems"] + ([x["problem"]] if x["problem"] else [])
    idx = F["idx"]

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    def mark(a) -> str:
        return {1: "✱", -1: "✱", 0: ""}[_excl(a["mt_ci"])]

    # ----- primary -----
    prim = {k: p1.get((k[0], k[1], MAIN_H)) for k in PRIMARY}
    holm = _holm({k: a["p"] for k, a in prim.items() if a is not None})
    prim_ok = {}
    for k, a in prim.items():
        if a is None:
            prim_ok[k] = False
            continue
        d = DIRN[k[0]]
        sign_ok = _ok(a["mt"]) and a["mt"] * d > 0
        excl_ok = _excl(a["mt_ci"]) == d
        prim_ok[k] = bool(sign_ok and excl_ok and holm[k][1])
    rows_prim = []
    for k, a in prim.items():
        if a is None:
            rows_prim.append([_mname(*k), "—", "—", "—", "—", "—", "—", "—", "資料不足"])
            continue
        d = DIRN[k[0]]
        rows_prim.append([f"{_mname(*k)}（{CN[k[0]]}，假設：{'越高越好' if d > 0 else '越高越差'}）",
                          f"{a['days']:,}（{_d(idx[a['from']])} 起）", f"{_pt(a['raw'])} {_cid(*a['raw_ci'])}",
                          f"{_pt(a['mt'])} {_cid(*a['mt_ci'])}", _share(a["share"]), f"{a['p']:.3f}",
                          f"{holm[k][0]:.3f}", "是" if (_ok(a["mt"]) and a["mt"] * d > 0) else "否",
                          "看得到" if prim_ok[k] else "沒看到"])

    # quintile detail for primaries + ACT_30
    rows_q = []
    for m, w in list(PRIMARY) + [("ACT", 30)]:
        first = True
        for j in range(5):
            cells = []
            for h in HORIZONS:
                a = p1.get((m, w, h))
                cells.append(f"{_p(a['mean'][j])} {_ci(*a['mean_ci'][j])}" if a else "—")
            a60 = p1.get((m, w, MAIN_H))
            rows_q.append([_mname(m, w) if first else "", f"第 {j + 1} 個五分之一" + ("（最低）" if j == 0 else "（最高）" if j == 4 else ""),
                           f"{a60['n'][j]:,}" if a60 else "—"] + cells
                          + [_share(a60["up"][j]) if a60 else "—"])
            first = False

    # grid
    rows_g = []
    marks, cells_n, plc_marks = {h: 0 for h in HORIZONS}, {h: 0 for h in HORIZONS}, {h: 0 for h in HORIZONS}
    imb_n = [0, 0, 0, 0]  # imbalance cells (not ACT): cells, interval marks, placebo marks, marks in the hypothesised direction
    for m in MEASURES:
        for h in HORIZONS:
            row = [f"{m}（{CN[m]}）" if h == HORIZONS[0] else "", f"之後 {h} 天"]
            for w in WINDOWS:
                a = p1.get((m, w, h))
                if a is None or not _ok(a["mt"]):
                    row.append("—")
                    continue
                mk = mark(a)
                cells_n[h] += 1
                marks[h] += int(bool(mk))
                plc_marks[h] += int(a["p"] <= 0.05)
                if m != "ACT":
                    imb_n[0] += 1
                    imb_n[1] += int(bool(mk))
                    imb_n[2] += int(a["p"] <= 0.05)
                    imb_n[3] += int(bool(mk) and _excl(a["mt_ci"]) == DIRN[m])
                row.append(f"{_pt(a['mt'])}{mk}" + ("◆" if a["p"] <= 0.05 else ""))
            rows_g.append(row)
    tot_cells = sum(cells_n.values())
    tot_marks = sum(marks.values())
    tot_plc = sum(plc_marks.values())
    sign_ok = {}
    for h in HORIZONS:
        good = tot = 0
        for m in MEASURES:
            if DIRN[m] == 0:
                continue
            for w in WINDOWS:
                a = p1.get((m, w, h))
                if a is not None and _ok(a["mt"]):
                    tot += 1
                    good += int(a["mt"] * DIRN[m] > 0)
        sign_ok[h] = (good, tot)

    # splits
    rows_s = []
    for m, w in list(PRIMARY) + [("ACT", 30)]:
        first = True
        for lab in [p[0] for p in PERIODS] + ["on", "off"]:
            a = sp.get((m, w, lab))
            name = {"on": "50 日均線持有的日子", "off": "50 日均線空手的日子"}.get(lab, lab)
            if a is None:
                rows_s.append([_mname(m, w) if first else "", name, "—", "—", "—"])
            else:
                rows_s.append([_mname(m, w) if first else "", name, f"{sum(a['n'][j] for j in (0, 4)):,}",
                               f"{_pt(a['raw'])} {_cid(*a['raw_ci'])}", f"{_pt(a['mt'])} {_cid(*a['mt_ci'])}"])
            first = False

    # ----- Part 2 -----
    side = p2["side"]
    rows_2 = []
    for h in HORIZONS:
        rows_2.append([f"之後 {h} 天"] + [f"{_pt(side[(m, h)]['mt'])} {_cid(*side[(m, h)]['mt_ci'])}"
                                         for m in ("ACT", "NET", "IMB", "RES")])
    rows_2.append(["（沒配對，之後 60 天）"] + [f"{_pt(side[(m, MAIN_H)]['raw'])} {_cid(*side[(m, MAIN_H)]['raw_ci'])}"
                                         for m in ("ACT", "NET", "IMB", "RES")])
    rows_2.append(["配對到的比例（60 天）"] + [_share(side[(m, MAIN_H)]["share"]) for m in ("ACT", "NET", "IMB", "RES")])
    rows_2x = []
    for a in range(3):
        for nn in range(3):
            cc = p2["cells"][3 * a + nn]
            rows_2x.append([f"活躍度{LV3[a]}", f"提存差{LV3[nn]}", f"{cc['n']:,}", f"{_p(cc['mean'])} {_ci(*cc['ci'])}",
                            _p(cc["med"]), _share(cc["up"]), _p(cc["past"])])

    # ----- Part 3 -----
    rows_3 = []
    for (kind, lab), v in p3["rows"].items():
        rows_3.append([f"{kind}{lab}", f"{v['n']}"] + [f"{c0 * 100:.0f}（隨機 {lo * 100:.0f}–{hi * 100:.0f}）"
                                                      for c0, lo, hi in v["cells"]])
    rows_3.append(["所有日子平均", "—"] + [f"{a * 100:.0f}" for a in p3["avg"]])
    out_band = sum(1 for v in p3["rows"].values() for c0, lo, hi in v["cells"] if not (lo <= c0 <= hi))
    n_band = sum(len(v["cells"]) for v in p3["rows"].values())
    rows_3e = []
    for k in TURN_KEYS:
        for tag, nm in (("day", "下車當天"), ("pre", f"下車前 {EXIT_BEFORE} 天平均")):
            e = p3["exits"][(k, tag)]
            rows_3e.append([_mname(*k), nm, f"{_r(e['whip'])}（{e['nw']} 次）", f"{_r(e['real'])}（{e['nr']} 次）",
                            f"{e['diff'] * 100:+.0f}" if _ok(e["diff"]) else "—",
                            f"{e['band'][0] * 100:+.0f} 到 {e['band'][1] * 100:+.0f}",
                            "是" if _ok(e["diff"]) and not (e["band"][0] <= e["diff"] <= e["band"][1]) else "否"])

    # ----- Part 4 -----
    s = x["s"]
    base = s["ma50"]
    p_rule = {k: (x["plc"][k] + 1) / (DRAWS + 1) for k in RULES}
    holm4 = _holm(p_rule)
    res4 = {}
    for k in RULES:
        f = s[k]
        loss = (f["c_deep_med"] >= base["c_deep_med"] - 1e-9) if k == "n2" else \
            (f["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK)
        res4[k] = {"score": sd._score_ge(f, base), "loss": bool(loss), "plc": x["plc"][k] / DRAWS <= PLACEBO_BAR,
                   "holm": holm4[k][1]}
        res4[k]["pass"] = res4[k]["score"] and res4[k]["loss"] and res4[k]["plc"]
    rows_x = []
    for k in ("ma50",) + RULES:
        e = s[k]
        ck = e["checks"]
        rows_x.append([NAME[k], f"{_share(ck[0])}（{e['d_n']} 個）", f"{_share(ck[1])}（{e['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{e['o_n']} 個）",
                       _share1(e["low"]) + ("" if e["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(e['c_deep_med'])}｜{_pc(e['c_deep_worst'])}", f"{e['c_edge']:+.1f} 點",
                       f"{x['tim'][k] * 100:.0f}%", "—" if k == "ma50" else f"{x['differ'][k]:,} 天",
                       "—" if k == "ma50" else (f"{_share(x['plc'][k] / DRAWS)}（真的 {_share1(x['small'][k]['low'])}，"
                                                f"錯位中位數 {_share1(x['plc_med'][k])}）")])
    rows_pass = []
    for k in RULES:
        f = s[k]
        rr = res4[k]
        rows_pass.append([SHORT[k], f"{_share1(f['low'])} vs {_share1(base['low'])}｜{yes(rr['score'])}",
                          f"{_pc(f['c_deep_med'])} vs {_pc(base['c_deep_med'])}｜{yes(rr['loss'])}",
                          f"{_share(x['plc'][k] / DRAWS)}｜{yes(rr['plc'])}",
                          f"p = {p_rule[k]:.3f}，調整後 {holm4[k][0]:.3f}｜{yes(rr['holm'])}",
                          "支持" if rr["pass"] else "不支持"])

    # ----- overlap -----
    def prow(name, pr):
        return [name, f"{pr['n']:,}", _share1(pr["agree"]), f"{pr['a']:,}｜{pr['b']:,}",
                f"{pr['both']:,} / {pr['either']:,}（{_share(pr['both'] / pr['either']) if pr['either'] else '—'}）"]
    rows_o = [prow("N1 的提早訊號 vs F1 的提早訊號", ov["n1_f1"])]
    if ov["has_s8"]:
        rows_o += [prow("N1 的提早訊號 vs S8（弱勢中幣離開交易所）", ov["n1_s8"]),
                   prow("F1 的提早訊號 vs S8", ov["f1_s8"])]
    rows_o2 = [["30 天提存差的排名 vs 力量差（force_balance）的排名，等級相關", f"{ov['corr']:+.2f}"],
               ["「30 天提存差 > 0」和「交易所餘額比 30 天前少」同一天成立或同一天不成立的比例",
                f"{_share1(ov['sign'])}（{ov['sign_n']:,} 天）" if _ok(ov["sign"]) else "—"],
               [f"{FIRST_MONTH[:7]} 起 N1 和 F1 持有狀態不同的天數", f"{ov['pos_diff']:,} 天（共 {ov['m15']:,} 天）"]]

    # ----- ETH -----
    eth = r["eth"]
    rows_eth = []
    if eth:
        for k in PRIMARY:
            a = eth["res"].get((k[0], k[1], MAIN_H))
            b = prim[k]
            if a is None:
                rows_eth.append([_mname(*k), "—", "—", "—", "—", "—"])
                continue
            same = _ok(a["mt"]) and _ok(b["mt"]) and (a["mt"] > 0) == (b["mt"] > 0)
            rows_eth.append([_mname(*k), f"{a['days']:,}（{_d(eth['idx'][a['from']])} 起）",
                             f"{_pt(a['mt'])} {_cid(*a['mt_ci'])}", f"{a['p']:.3f}",
                             f"{_pt(b['mt'])}", "同方向" if same else "方向相反"])

    # ----- data table -----
    ser = r["ser"]
    cm = data["cm"]
    fv = lambda s_: _d(s_.first_valid_index()) if s_ is not None and s_.first_valid_index() is not None else "—"  # noqa: E731
    rows0 = [["Coin Metrics 送進／提出交易所（美元、顆數）第一天", fv(cm["FlowInExUSD"])],
             ["Coin Metrics 最後一天（資料日）", _d(cm["FlowInExUSD"].last_valid_index())],
             ["新增穩定幣第一天（DefiLlama，從 2019-07-01 用起；之前算 0）", _d(ser["first_stable"]) if ser["first_stable"] is not None else "—"],
             ["ETF 淨流入第一天（Farside；之前算 0）", _d(data["etf"].index[0]) if len(data["etf"]) else "—"]]
    for k in [("ACT", 30), ("NET", 30), ("IMB", 30), ("RES", 30), ("dIMB", 30), ("NET", 7), ("RES", 90)]:
        rows0.append([f"{_mname(*k)} 的排名第一次算得出來（資料日）", fv(ser["rank"][k])])
    lk_total, lk_bad = r["look"]

    luck_all = tot_cells * 0.10
    tables = [
        {"caption": ("資料從哪天開始。名詞：買方（美元）＝從交易所提走的比特幣＋新增穩定幣＋美國現貨 ETF 淨流入；賣方（美元）＝送進交易所的比特幣。"
                     "活躍度＝買賣兩邊加起來有多大（取對數）。買賣差＝買方減賣方再除以兩邊的總和（−1 到 +1）。提存差＝同樣的算法但只用"
                     "比特幣顆數（提走減存入，再除以兩者總和），價格高低影響不到它。多出來的賣方＝賣方比「照當天的買方，過去兩年平常該有的賣方」"
                     "多多少（取對數後的差，正的＝賣得比平常多）；多出來的買方是反過來。買賣差的變化＝今天的買賣差減掉 W 天前的。"
                     "每個數字都用 1、7、30、90 天四種長度算。排名＝今天比過去一年多少比例的日子高（0–100）。每個來源某一天的數字，"
                     "都等到隔一天的日線收盤才拿來用。"),
         "columns": ["項目", "數字"], "rows": rows0},
        {"caption": (f"第一部分的四個主要檢定（跑之前就定好只有這四個）：30 天的提存差、買賣差、多出來的賣方、買賣差的變化，看之後 {MAIN_H} 天。"
                     "「前五分之一 − 後五分之一」＝排名最高那五分之一的日子，減排名最低那五分之一的日子，之後的平均漲跌。"
                     "配對＝只在「過去 30 天漲跌差不多（十組）× 最近 20 天波動差不多（三組）× 市場一樣忙（活躍度三組）」的日子裡互相比，"
                     "這樣差別只剩買賣差本身。90% 區間用 60 天一塊的區塊重抽 1,000 次。隨機錯位 p＝把這條數字整條往後挪 180 天以上（頭尾相接）"
                     f"{DRAWS} 次，有多少次配對後的差至少一樣往假設的方向偏。Holm＝四個一起考時的嚴格調整。「看得到」要三條都成立："
                     "方向和假設一樣、區間不含零、Holm 調整後還在 5% 以內。"),
         "columns": ["數字（假設）", "天數", "前五分之一 − 後五分之一（沒配對）［90% 區間］", "配對後 ［90% 區間］",
                     "配對到的比例", "隨機錯位 p", "Holm 調整後 p", "方向和假設一樣", "結果"],
         "rows": rows_prim},
        {"caption": "四個主要數字和 30 天活躍度：依排名分成五組（各約五分之一的日子），之後 7、30、60 天的平均漲跌和 90% 區間，以及之後 60 天是漲的比例。",
         "columns": ["數字", "組別", "天數（60 天那欄）", "之後 7 天平均", "之後 30 天平均", "之後 60 天平均", "之後 60 天上漲比例"],
         "rows": rows_q},
        {"caption": (f"整片地圖（只描述，不算檢定）：每格是配對後「前五分之一 − 後五分之一」的差（點）。✱＝90% 區間不含零；◆＝隨機錯位 p ≤ 0.05。"
                     f"一共 {tot_cells} 格，其中 ✱ 有 {tot_marks} 格、◆ 有 {tot_plc} 格；扣掉活躍度那 {tot_cells - imb_n[0]} 格，"
                     f"買賣差類的 {imb_n[0]} 格裡 ✱ {imb_n[1]} 格（其中方向和假設一樣的 {imb_n[3]} 格）、◆ {imb_n[2]} 格。如果哪裡都沒有效果，光靠運氣大約會有 "
                     f"{luck_all:.0f} 格 ✱（每格約 10%）和 {tot_cells * 0.05:.0f} 格 ◆（每格約 5%）；而且相鄰的格子（同一個數字不同天數、"
                     "不同看多久）高度相關，運氣造成的格數可能比這多很多或少很多。活躍度沒有事先定方向；其他數字的假設方向："
                     "提存差、買賣差、多出來的買方、買賣差的變化越高越好，多出來的賣方越高越差。"),
         "columns": ["數字", "看多久"] + [f"{w} 天的長度" for w in WINDOWS], "rows": rows_g},
        {"caption": "四個主要數字和 30 天活躍度，分年代、分「50 日均線持有／空手」看（之後 60 天，只描述）。天數＝最高和最低五分之一合計。",
         "columns": ["數字", "期間或狀態", "天數", "沒配對的差 ［90% 區間］", "配對後的差 ［90% 區間］"], "rows": rows_s},
        {"caption": (f"第二部分：兩股力量並排。同一批日子（{_d(p2['from'])} 到 {_d(p2['to'])}，{p2['days']:,} 天），同一套重抽。"
                     "活躍度的效果＝活躍度最高五分之一減最低五分之一（配對過去 30 天漲跌和波動）；買賣差的效果＝提存差、買賣差、多出來的賣方"
                     "最高減最低五分之一（再多配對活躍度）。並排放是為了不用其中一個去「解釋掉」另一個。"),
         "columns": ["", "活躍度 ACT_30", "提存差 NET_30", "買賣差 IMB_30", "多出來的賣方 RES_30"], "rows": rows_2},
        {"caption": (f"第二部分的兩維圖：30 天活躍度三組 × 30 天提存差三組（各用過去一年排名的三分之一切，所以每格天數差不多）。"
                     f"之後 {MAIN_H} 天的平均（90% 區間）、中位數、上漲比例，以及這些日子之前 30 天已經漲跌多少。"
                     f"排名之間的等級相關：活躍度和提存差 {p2['corr']['act_net']:+.2f}，活躍度和多出來的賣方 {p2['corr']['act_res']:+.2f}，"
                     f"提存差和多出來的賣方 {p2['corr']['net_res']:+.2f}。"),
         "columns": ["活躍度", "提存差", "天數", f"之後 {MAIN_H} 天平均 ［90% 區間］", "中位數", "上漲比例", "之前 30 天漲跌"],
         "rows": rows_2x},
        {"caption": (f"第三部分（只描述）：rally_sources 的大漲段（{p3['n_rally']} 段，其中 {p3['n_peak']} 段已經結束有高點）"
                     "低點和高點前後，四個數字的平均排名（0–100）。括號是隨機挑同樣多天、算同樣的平均 500 次，5% 到 95% 的範圍；"
                     f"落在範圍外才算和平常不一樣。{n_band} 格裡有 {out_band} 格在範圍外（純運氣約 {n_band * 0.1:.1f} 格）；"
                     "事件很少，而且低點、高點「之後」的格子是事後才知道的，不能拿來事先判斷。"),
         "columns": ["時段", "事件數", "ACT_30 活躍度", "NET_7 提存差", "NET_30 提存差", "RES_30 多出來的賣方"], "rows": rows_3},
        {"caption": (f"第三部分（只描述）：50 日均線下車時（{p3['first'].strftime('%Y-%m-%d')} 起；假跌破 {p3['counts']['whip']} 次、"
                     f"真下跌 {p3['counts']['real']} 次、其他 {p3['counts']['other']} 次、還不知道 {p3['counts']['unknown']} 次）。"
                     "假跌破＝20 天內均線又上車、而且價格更高；真下跌＝先跌到 −15%。把兩種標籤隨機打亂 1,000 次，看差距的 5%–95% 範圍。"),
         "columns": ["數字", "什麼時候", "假跌破平均排名", "真下跌平均排名", "差（假 − 真）", "隨機打亂的範圍", "超出範圍"],
         "rows": rows_3e},
        {"caption": (f"第四部分：公平考試（main_signal_check 的做法，和 force_balance 一樣的接法）。比特幣日線起算月 {FIRST_MONTH[:7]} 到 "
                     f"{msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格，一般人做法（晚 10 小時、成本 "
                     f"{P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），其他資產 {'、'.join(x['ex'].other)}"
                     f"（{base['o_n']} 個起算月；各用自己的均線，配比特幣的訊號當全市場訊號）。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"
                     f"隨機錯位＝只把規則用到的那條訊號往後挪 {GAP} 天以上（頭尾相接）{DRAWS} 次，整個考試重算（一般人做法每個收盤時間 "
                     f"{PL_SIMS} 次，真的規則也這樣重算來比）。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "持有時間", "和均線不同的天數",
                     "隨機錯位比例"],
         "rows": rows_x},
        {"caption": ("三條規則的條件（跑之前定好）：排名分數不輸均線；最深虧損中位數最多比均線深 5 點（N2 只會少拿，所以要求完全不能更深）；"
                     "隨機錯位 ≤ 5%（這是「供需」這條線第七份研究）。三條都過才叫「支持」。Holm 是三條一起考的調整（另外列，供參考）；"
                     "純靠運氣，三條裡大約 0.15 條會過 5% 的隨機錯位門檻。"),
         "columns": ["規則", "排名分數（規則 vs 均線）", "最深虧損中位數（規則 vs 均線）", "隨機錯位", "Holm（三條一起）", "結果"],
         "rows": rows_pass},
        {"caption": (f"N1、force_balance 的 F1、after_fall_forces 的 S8 是不是同一件事（{FIRST_MONTH[:7]} 起，決定日）。"
                     "N1 的提早訊號＝30 天提存差排名前五分之一＋站上 20 日均線；F1＝力量差（美元）排名前三分之一＋站上 20 日均線；"
                     "S8＝交易所餘額比 30 天前少＋過去 30 天沒漲。"),
         "columns": ["比較", "天數", "同一天狀態一樣的比例", "各自亮的天數", "兩個都亮 / 任一個亮"], "rows": rows_o},
        {"caption": "同一件事的另外兩個角度。",
         "columns": ["項目", "數字"], "rows": rows_o2},
    ]
    if rows_eth:
        tables.append({"caption": (f"以太幣重做第一部分的四個主要檢定（只描述）：以太幣自己的交易所流量（Coin Metrics，"
                                   f"資料日 {_d(eth['cm_first'])} 起；買方只有從交易所提走的以太幣，沒有穩定幣和 ETF），"
                                   f"以太幣自己的價格（幣安，{_d(eth['first'])} 起）、自己的配對和隨機錯位。之後 {MAIN_H} 天。"),
                       "columns": ["數字", "天數", "配對後的差 ［90% 區間］", "隨機錯位 p", "比特幣的同一個數字", "和比特幣比"],
                       "rows": rows_eth})

    # ----- verdict -----
    def pline(k):
        a = prim[k]
        if a is None:
            return f"{_mname(*k)} 資料不足"
        return f"{_mname(*k)} {_pt(a['mt'])} {_cid(*a['mt_ci'])}（隨機錯位 p {a['p']:.3f}）"
    n_seen = sum(prim_ok.values())
    act_cells = [p1[("ACT", w, h)] for w in WINDOWS for h in HORIZONS
                 if p1.get(("ACT", w, h)) is not None and _ok(p1[("ACT", w, h)]["mt"])]
    v1 = (f"【量到的】第一部分四個主要檢定（30 天長度、之後 {MAIN_H} 天、配對過去漲跌、波動和市場忙不忙之後，最高五分之一減最低五分之一）："
          + "；".join(pline(k) for k in PRIMARY) + "。"
          + (f"四個裡有 {n_seen} 個三條件都成立（方向對、區間不含零、Holm 後還在 5% 內）："
             + "、".join(_mname(*k) for k in PRIMARY if prim_ok[k]) + "。" if n_seen else
             "四個都沒有同時做到「方向和假設一樣、區間不含零、Holm 調整後在 5% 內」，所以用這個定義、這份資料，沒看到買賣差對之後 60 天有可靠的關係"
             "（不代表這個想法錯，可能漏掉的東西寫在附註）。"))
    v2 = (f"【量到的】整片地圖：買賣差類（提存差、買賣差、多出來的賣方／買方、買賣差的變化）{imb_n[0]} 格裡 {imb_n[1]} 格區間不含零"
          f"（其中 {imb_n[3]} 格方向和假設一樣；純運氣約 {imb_n[0] * 0.1:.0f} 格）、{imb_n[2]} 格隨機錯位 p ≤ 0.05（純運氣約 "
          f"{imb_n[0] * 0.05:.0f} 格）；方向和假設一樣的格子：之後 7 天 {sign_ok[7][0]}/{sign_ok[7][1]}、30 天 "
          f"{sign_ok[30][0]}/{sign_ok[30][1]}、60 天 {sign_ok[60][0]}/{sign_ok[60][1]}（格子彼此高度相關，不是獨立的票）。"
          f"活躍度那 {len(act_cells)} 格裡 {sum(1 for a in act_cells if a['mt'] > 0)} 格是正的（市場越忙之後越漲）、"
          f"{sum(1 for a in act_cells if _excl(a['mt_ci']))} 格區間不含零、{sum(1 for a in act_cells if a['p'] <= 0.05)} 格隨機錯位 p ≤ 0.05。")
    a60, n60, r60, i60 = side[("ACT", MAIN_H)], side[("NET", MAIN_H)], side[("RES", MAIN_H)], side[("IMB", MAIN_H)]
    ac = r["ac"]
    v3 = (f"【量到的】並排（同一批 {p2['days']:,} 天，{_d(p2['from'])} 起，之後 {MAIN_H} 天）：活躍度最高減最低五分之一 "
          f"{_pt(a60['mt'])} {_cid(*a60['mt_ci'])}；提存差 {_pt(n60['mt'])} {_cid(*n60['mt_ci'])}；買賣差 {_pt(i60['mt'])} "
          f"{_cid(*i60['mt_ci'])}；多出來的賣方 {_pt(r60['mt'])} {_cid(*r60['mt_ci'])}。第一部分的 NET_30 從 {_d(idx[prim[('NET', 30)]['from']])} "
          f"起是 {_pt(prim[('NET', 30)]['mt'])}，第二部分從 {_d(p2['from'])} 起是 {_pt(n60['mt'])}：起點只差一年多，符號就翻過來，"
          "兩個區間都含零。【推測】差別可能來自 2012–13 年漲跌特別大的那段日子，這份資料沒辦法確定。"
          f"【量到的，看到結果後加的描述】活躍度（美元）的排名和過去 180 天漲跌的等級相關 {ac['p180'][0]:+.2f}、和過去 30 天漲跌 "
          f"{ac['p30'][0]:+.2f}、和用顆數算的活躍度 {ac['coin'][0]:+.2f}。【推測】所以「市場忙之後比較會漲」和「價格已經漲了好幾個月之後"
          "繼續漲」在這份資料裡綁在一起，配對只拿掉了過去 30 天，分不開這兩個解釋；兩個都列著，不拿其中一個否定另一個。")
    fails = {k: ([] if res4[k]["score"] else ["排名分數輸均線"]) + ([] if res4[k]["loss"] else ["最深虧損比均線深太多"])
             + ([] if res4[k]["plc"] else ["隨機錯位超過 5%"]) for k in RULES}
    v4 = ("【量到的】第四部分：" + "；".join(
        f"{SHORT[k]} 排名分數 {_share1(s[k]['low'])}（均線 {_share1(base['low'])}）、最深虧損中位數 {_pc(s[k]['c_deep_med'])}"
        f"（均線 {_pc(base['c_deep_med'])}）、隨機錯位 {_share(x['plc'][k] / DRAWS)}，"
        + ("三條都過（只能算每日影子紀錄的候選）" if res4[k]["pass"] else f"不支持（{'、'.join(fails[k])}）")
        for k in RULES) + "。")
    v5 = ("【量到的】N1 和 F1 同一天狀態一樣的比例 " + _share1(ov["n1_f1"]["agree"])
          + (f"、和 S8 {_share1(ov['n1_s8']['agree'])}" if ov["has_s8"] else "")
          + (f"；「30 天提存差 > 0」和「交易所餘額 30 天變少」一致 {_share1(ov['sign'])}" if _ok(ov["sign"]) else "") + "。")
    v6 = ("【推測】這份資料分不開的事：買賣差高的日子之後的表現，可能是買方真的比較強，也可能是交易所自己搬錢包、標籤更新造成的假流量；"
          "也可能兩股力量本來就主要反映市場忙不忙。這裡只能說在這個定義下量到多少。")
    outs = [f"{kind}{lab}的 {_mname(*TURN_KEYS[j])} {c0 * 100:.0f}（隨機 {lo * 100:.0f}–{hi * 100:.0f}）"
            for (kind, lab), v in p3["rows"].items() for j, (c0, lo, hi) in enumerate(v["cells"]) if not (lo <= c0 <= hi)]
    v_t = (f"【量到的】第三部分（只描述，{list(p3["rows"].values())[0]["n"]} 次大漲）：{n_band} 格裡 "
           f"{out_band} 格在隨機範圍外（運氣約 {n_band * 0.1:.1f} 格）" + ("：" + "；".join(outs) if outs else "") + "。")
    verdict = v1 + v2 + v3 + v_t + v4 + v5 + v6

    notes = [
        "為什麼有這份研究：force_balance 用美元量了買方和賣方，發現兩者幾乎一起動（排名相關 0.96；賣方高的日子 96% 買方也高），"
        "所以 3×3 表裡真正要看的格子幾乎沒有日子（買方高、賣方低只有 2 天）。這份研究把每天拆成「兩邊共同的大小」（活躍度：市場有多忙）"
        "和「沒被對方抵掉的那一部分」（買賣差、多出來的賣方或買方），用全部的日子、四種時間長度來看。",
        "名詞：活躍度＝買賣兩邊加起來有多大；買賣差＝買方減賣方再除以兩邊的總和；提存差＝提走的顆數減存入的顆數再除以兩者總和"
        "（只用顆數，價格高低影響不到）；多出來的賣方＝賣方比「照當天的買方、過去兩年平常該有的賣方」多多少；多出來的買方反過來；"
        "買賣差的變化＝今天的買賣差減 W 天前的；五分之一＝依過去一年排名切成五組。",
        "怎麼讀「配對」：買賣差高的日子，常常是價格剛漲過或市場正忙的日子。配對就是只拿過去 30 天漲跌、最近波動、市場忙不忙都差不多的日子來比，"
        "剩下的差別才比較像是買賣差本身帶來的。配對不掉的東西（例如新聞、槓桿）還是會混在裡面。",
        "【量到的】一個傾向不能拿來當定律去否定另一個：例如「漲過的會繼續漲」和「買方比較強」在這份資料裡常常同時發生，"
        "所以並排放（第二部分），各自有自己的區間，不拿其中一個去解釋掉另一個。",
        "【文獻說的】常見說法「交易所淨流出（幣被提走）是看漲」來自鏈上資料商的文章與社群；這份研究沒有打開任何特定文獻，"
        "只把它當成要檢查的假設。",
        "老實說的限制：交易所流量是用今天認得的交易所地址套回過去，越早越低估；交易所自己錢包之間搬、交易所之間互轉，"
        "會讓兩股力量一起被灌大——這正是要把「共同的大小」拆出來的原因。場外交易（OTC）看不到；2024 年以前不經過交易所的買盤也看不到；"
        "新增的穩定幣不全是拿來買比特幣。買方的組成在 2019-07（加入穩定幣）和 2024-01（加入 ETF）各變一次。",
        "沒看到效果的意思是「用這個定義、這份資料沒看到」，不是「這個想法錯了」。可能漏掉的：交易所標籤不是當時的、"
        "不同交易所混在一起（現貨所、衍生品所的存入意義不同）、大戶的場外交易、時間長度或門檻選得不對（沒有試其他數字）。",
        "這是「供需」這條線的第七份研究（supply_demand、price_impact、reluctant_sellers、old_coin_selling、force_balance、"
        "after_fall_forces 之後），所以規則的隨機錯位門檻是 5%；就算通過，也只代表可以放進每日影子紀錄觀察。",
        "多出來的賣方用的直線，是用「前一天為止」的兩年資料算的，所以不會偷看未來；它會隨時間慢慢變。",
        "門檻（1／7／30／90 天、365 天排名、730 天的直線、五分之一、三分之一、20 日均線、60 天的區塊）都是跑之前憑道理定的，"
        "沒有試過其他數字；定義寫好之後才跑，看到結果也沒改。",
        "其他幣和 MSTR 沒有用自己的流量，用的是比特幣的訊號（當成全市場的狀態），配各自的均線；MSTR 用前一天的訊號。",
        f"偷看未來的檢查：把所有資料在 {'、'.join(CUTS)} 截斷（來源再早一天）重算，{lk_total} 次比對裡有 {lk_bad} 次不一致"
        f"{'（應該是 0）' if lk_bad else '，截斷前的五組分法和三條規則的訊號完全一樣'}。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載；考試和隨機錯位 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    if not eth:
        notes.append("以太幣重做沒有算（資料取不到，原因見下）。")
    notes += problems

    return {
        "id": "force_residual",
        "title": "買賣兩邊拆開看：市場忙不忙，和誰沒被抵掉",
        "question": ("買方和賣方幾乎總是一起變大變小。把每天拆成「兩邊共同的大小」（市場有多忙）和「沒被對方抵掉的那一部分」（買賣差），"
                     "用全部的日子、1／7／30／90 天四種長度來看：買賣差偏向買方的時候，之後是不是比較會漲？多出來的賣方是不是之後比較差？"
                     "能不能幫 50 日均線？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群 API：比特幣 FlowInExUSD、FlowOutExUSD、FlowInExNtv、FlowOutExNtv、SplyExNtv",
             "url": CM_URL + "?assets=btc&metrics=" + ",".join(CM_BTC) + "&frequency=1d"},
            {"name": "Coin Metrics 社群 API：以太幣 FlowInExUSD、FlowOutExUSD、FlowInExNtv、FlowOutExNtv",
             "url": CM_URL + "?assets=eth&metrics=" + ",".join(CM_ETH) + "&frequency=1d"},
            {"name": "DefiLlama 穩定幣總供給 API", "url": LLAMA},
            {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": FARSIDE_URL},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
