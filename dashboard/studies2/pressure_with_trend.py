"""Historical study "pressure_with_trend": the buy/sell PRESSURE PATTERN of pressure_before_moves combined with a TREND
STATE. The pattern may say "something is happening"; can a trend state say which way?

Why (the owner's next question, step 3 of their plan "1. find the signs; 2. does following them make money on entry /
lose less on exit; 3. if one indicator cannot do it, does adding a parameter do it"): pressure_before_moves found that
"selling and buying composites both rising over the last 30 days versus the 60 before" was the most common pattern
before BOTH big rises (12 of 23) and big falls (14 of 27), so on its own it does not say which way. "What if these
pressure patterns are combined with the 55-day average or the trend signals tested before?"

Ground rules (the owner's, applied to every user-facing sentence): observe first, do not conclude lightly; statements
are labelled 量到的 (measured, with the number and k of N) or 推測 (a guess, marked); where the data cannot separate
two explanations, say so and pick neither; one tendency is never used as a law to dismiss another; "not seen" means
not seen with this definition and this data, and what may be missing is named; individual events are shown.

Data: exactly pressure_before_moves' loaders and composites (Coin Metrics FlowInExNtv / FlowOutExNtv, Open Bitcoin
Metrics spent >= 155 days CSV data file, Binance BTCUSDT 1-hour klines, DefiLlama stablecoin supply, Farside ETF flows,
Coinbase and Bitstamp daily closes; same lags), stored in this study's own cache/_store/pressure_with_trend_*.json with
pressure_before_moves' / earlier stores as read-only fallbacks. Funding and open interest are NOT loaded (the two
composites do not use them). BTC daily closes te._daily_close (00:00 UTC). Fair exam inputs as main_signal_check:
Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance daily) and MSTR (Yahoo). Free, key-less, user agent
"btc-research-dashboard"; a refusal is reported, never worked around; .env is never read.

Definitions (fixed before running):
  Pattern P on day t (knowable at the close of t): dS = mean rank of SELL_C over days t-29..t minus mean over
    t-89..t-30 (at least 20 and 40 values), dB the same for BUY_C; SELL_C / BUY_C exactly pressure_before_moves'
    composites (ranks already lagged to publication). P = BOTH_UP (dS > 0, dB > 0), BOTH_DOWN, SELL_DOWN_BUY_UP,
    SELL_UP_BUY_DOWN; a zero or missing change = FLAT (left out). For an event on day e the pattern is P(e-1), which
    is exactly pressure_before_moves' "prior 30 days versus the 60 before". dS, dB and NET30 (30-day net withdrawn
    share, coins) are kept as continuous values.
  Trend states at the close of t: T1 = ma50 on / off (close above its 50-day average; trend_ensemble's A50, the
    main signal); T2 = close above / below its 55-day average; T3 = the 55/20 breakout rule in / out
    (clock_robustness._breakout: in after a 55-day closing high, out after a 20-day closing low); T4 = close above /
    below its 200-day average; T5 = trend_ensemble's 17-rule blend in thirds (rules long <= 5 / 6-11 / >= 12);
    T6 = the 50-day average itself rising / falling versus 20 days earlier.
  Events: pressure_before_moves' zigzag (rise starts: lows followed by +50% before a 25% fall; fall starts: highs
    followed by -30% before a 25% rebound), listed from 2012-01-01. Random non-event days: pressure_before_moves'
    pool (from 2012-01-01 to 30 days before the last close, at least 60 days from every event of either type).

Part 1 - at the events (observation). One row per event: date, size, pattern P, and each trend state 30 days before,
  10 days before and on the event day. Tally pattern x trend state (state on the event day; the same at -10 days as a
  second view) before rises, before falls, and on random days (shares converted to "how many in N"). Separation =
  balanced accuracy BA = 0.5 x sum over cells of max(share of rises in the cell, share of falls in the cell) (50% =
  no separation, 100% = complete). BA is computed for the trend state alone, the pattern alone, and pattern x state.
  Because adding cells can only raise in-sample BA, two shuffle nulls (2,000 shuffles each, fixed seed): "what the
  pattern adds to trend" = share of shuffles of the PATTERN labels among events (state and type kept) whose
  pattern x state BA >= the real; "what trend adds to the pattern" = the same shuffling the STATE labels. Plain
  counts "BOTH_UP with ma50 off: k rises, m falls" for every pattern and state. Also: how often T1 and T2 differ.
Part 2 - all days. Days from 2012-01-01 with a non-flat pattern and a complete 60-day look-ahead. For every pattern x
  state cell of T1, T3, T4: days; forward 30- and 60-day return mean and median; share with a positive 60-day return
  ("followed by a rise"); deepest drop within 60 days (lowest close of t+1..t+60 versus close t, capped at 0), mean.
  90% intervals = block bootstrap over the day sequence (60-day blocks, 1,000 draws, fixed seed). Within the same
  state, the difference of each pattern versus the other three, matched on 30 strata (past-30-day return decile x
  20-day volatility tercile; quantiles over all analysis days; weights = the pattern's days per stratum), for the 60-
  and 30-day forward return, with bootstrap 90% intervals. By period 2012-16 / 2017-21 / 2022-now (days and mean
  60-day return per cell). Placebo: the pattern series circularly shifted by k (uniform in [180, span-180] days)
  against price and trend, 200 draws; placebo share = share of draws whose matched 60-day difference is at least as
  far from zero in the same direction.
Part 3 - following it. DISCOVERY = analysis days whose 60-day look-ahead ends by 2018-12-31; TEST = decision days from
  2019-01-01. On DISCOVERY only, with T1: ENTRY cell = the pattern with the highest mean 60-day forward return among
  ma50-OFF days with at least 150 days; EXIT cell = the pattern with the lowest mean 60-day forward return among
  ma50-ON days with at least 150 days. Rules (decided at the close, held the next day):
    E1 = hold when ma50 on OR (P = ENTRY AND close above its 20-day average);
    X1 = hold when ma50 on AND NOT (P = EXIT);
    EX = (ma50 on AND NOT P = EXIT) OR (P = ENTRY AND close above its 20-day average).
  The same three with T3 (55/20) in place of ma50 (cells picked again on DISCOVERY for T3), versus plain 55/20.
  Fair exam (main_signal_check's four checks, price_impact._Exam's starts machinery) restricted to starts from
  2019-01: BTC daily monthly starts 2019-01 .. 2023-12 (0.10%); 1-hour clock starts 2019-01 .. 2023-12 x 24 closing
  hours; the ordinary person (10 hours later, 0.25%, 10% missed entries, 200 simulations per clock); ETH SOL BNB XRP
  MSTR from 2019-01 (or their first valid month) to three years before today with their own ma50 / 55/20 and BTC's
  pattern as a market-wide state (supply_demand._Exam._at lags). Ranking score = worst of the four "beats holding"
  shares with main_signal_check's eligibility (supply_demand._score_ge); median and worst deepest loss over the clock
  textbook cells; time in market = share of BTC test days held. Placebo: the pattern series shifted as in Part 2
  (cells NOT re-picked), 200 draws, 20 ordinary-person simulations per clock, the real rule re-scored with 20;
  placebo share = share of draws whose ranking score is at least the real one's.
  "Supported": E1 - ranking >= the baseline's AND median deepest loss no more than 5 points worse AND placebo <= 5%;
  X1 and EX - ranking >= the baseline's AND median deepest loss no worse AND placebo <= 5%. Six rules: Holm at 5% on
  the placebo shares (exit_forces._holm); 6 x 5% = 0.3 rules would pass the placebo bar by luck alone.
  Plain questions on TEST days (BTC daily, 0.10%): early entries of E1 (episodes where E1 holds while ma50 is in cash,
  from cash; ends when ma50 turns on = "took over", or E1 goes to cash): count, share ending higher, share ending
  higher among those ma50 took over, average gain, average loss, and total effect (E1's money / ma50's money - 1 over
  the test period). Early exits of X1 (episodes where ma50 holds and X1 is in cash; ends when X1 holds again or ma50
  goes to cash): count, average fall avoided, average rise missed, net effect, and the same in 2020-21 and 2022.
Part 4 - exploration only, no rule. Within the ENTRY cell (ma50 off, P = ENTRY) and the EXIT cell (ma50 on, P = EXIT):
  for each condition of this fixed list - NET30 rank >= 50 / < 50; old-coin movement (S_old rank) 30-day change
  > 0 / <= 0 (real-time window as P); activity (ACT rank) >= 50 / < 50; T4 up / down; T6 rising / falling; close more
  than 25% below its 365-day high / not; 20-day volatility above / not above the median of its previous 365 values
  (>= 180) - mean 60-day forward return when the condition holds minus when it does not, on DISCOVERY and on TEST, and
  whether the sign agrees. "Most improves" = the largest absolute discovery difference with at least 30 days on each
  side. This is where a next pre-registered test would start.
Limits to state: composites change make-up (before 2017-08 BUY_C is withdrawals alone, SELL_C deposits + old coins);
  exchange labels are today's applied backwards; only 23 + 28 events, many within 90 days of an opposite event; the
  event-day trend state is partly fixed by construction (lows sit below averages, highs above); the test period holds
  few independent cycles. Seeds fixed. Not advice.
Today: "比特幣：50 日均線｜E1｜X1" ("現金｜現金｜現金" style), the same for the 55/20 family, today's pattern in plain
  words, each trend state, and today's cell with its measured Part 2 numbers (a description, not a forecast).
ADDED AFTER SEEING THE RESULTS (definitions, thresholds and the exam unchanged): the X1 verdict sentence split into
early exits and blocked entries; a verdict line on Part 2's multiple comparisons (24 cells, about 1.2 expected below
5% by luck) with the by-period values of "ma50 on + selling down & buying up" and one marked guess; a verdict line
naming Part 4 conditions whose sign agreed in both cells and both halves.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import exit_forces as ef
from dashboard.studies2 import force_balance as fb
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import pressure_before_moves as pb
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

DAY = pd.Timedelta(days=1)
FROM = pd.Timestamp("2012-01-01")
SPLIT = pd.Timestamp("2019-01-01")
DISC_END = pd.Timestamp("2018-12-31")
H = (30, 60)
BLOCK, BOOT = 60, 1000
DRAWS, SHIFT_MIN, SHUF = 200, 180, 2000
SEED = 20261120
MIN_CELL, MIN_SIDE = 150, 30
NSTRAT = 30
COST, SIMS, P_COST, P_MISS = msc.COST, msc.SIMS, msc.P_COST, msc.P_MISS
PL_SIMS = 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.05
PERIODS = (("2012–16", "2012-01-01", "2016-12-31"), ("2017–21", "2017-01-01", "2021-12-31"),
           ("2022 起", "2022-01-01", "2100-01-01"))
STORE_CM = common.STORE / "pressure_with_trend_cm.json"
STORE_LL = common.STORE / "pressure_with_trend_llama.json"
STORE_OBM = common.STORE / "pressure_with_trend_obm.json"

PAT = ("賣壓降＋買壓升", "兩邊都升", "兩邊都降", "賣壓升＋買壓降")
PAT_FLAT = "有一邊持平或缺資料"
TREND = ("t1", "t2", "t3", "t4", "t5", "t6")
TNAME = {"t1": "50 日均線（主訊號）", "t2": "55 日均線", "t3": "55／20 突破規則", "t4": "200 日均線（長趨勢）",
         "t5": "17 條順勢規則看多幾條", "t6": "50 日均線本身的方向"}
TSHORT = {"t1": "50日線", "t2": "55日線", "t3": "55／20", "t4": "200日線", "t5": "17條", "t6": "50日線方向"}
TLEV = {"t1": ("關（收盤在 50 日均線下）", "開（收盤在 50 日均線上）"),
        "t2": ("收盤在 55 日均線下", "收盤在 55 日均線上"),
        "t3": ("55／20 出場中（現金）", "55／20 持有中"),
        "t4": ("收盤在 200 日均線下", "收盤在 200 日均線上"),
        "t5": ("低（看多 ≤5 條）", "中（6–11 條）", "高（≥12 條）"),
        "t6": ("50 日均線比 20 天前低（在降）", "50 日均線比 20 天前高（在升）")}
TMINI = {"t1": ("關", "開"), "t2": ("下", "上"), "t3": ("出", "持"), "t4": ("下", "上"), "t5": ("低", "中", "高"),
         "t6": ("降", "升")}
TYPE_CN = {"rise": "大漲", "fall": "大跌"}
PART2_T = ("t1", "t3", "t4")
FAM = {"t1": ("ma50", "e1", "x1", "ex"), "t3": ("b5520", "e1b", "x1b", "exb")}
RULES = ("e1", "x1", "ex", "e1b", "x1b", "exb")
KI = {"b5520": 0, "ma50": 1, "e1": 150, "x1": 151, "ex": 152, "e1b": 153, "x1b": 154, "exb": 155}
NAME = {"ma50": "50 日均線（基準）", "b5520": "55／20（基準）",
        "e1": "E1：50 日均線，或「進場型態＋站上 20 日均線」也持有",
        "x1": "X1：50 日均線，但出現「出場型態」就先出場",
        "ex": "EX：E1 和 X1 一起",
        "e1b": "E1（55／20 版）：55／20，或「進場型態＋站上 20 日均線」也持有",
        "x1b": "X1（55／20 版）：55／20，但出現「出場型態」就先出場",
        "exb": "EX（55／20 版）：兩個一起"}
SHORT = {"ma50": "50 日均線", "b5520": "55／20", "e1": "E1", "x1": "X1", "ex": "EX", "e1b": "E1（55／20）",
         "x1b": "X1（55／20）", "exb": "EX（55／20）"}
COND = ("net", "old", "act", "t4", "t6", "dd", "vol")
CNAME = {"net": "淨提走比例排名 ≥50（提走偏多）", "old": "老幣移動 30 天在升", "act": "活躍度排名 ≥50（熱鬧）",
         "t4": "收盤在 200 日均線上", "t6": "50 日均線在升", "dd": "離一年最高點跌超過 25%",
         "vol": "20 天波動高於過去一年中位數"}

_ok, _p, _pt, _share, _ci = sd._ok, sd._p, sd._pt, sd._share, sd._ci
_d = pb._d


def _sg(v, nd: int = 1) -> str:
    return f"{float(v):+.{nd}f}" if _ok(v) else "—"


def _pct(v) -> str:
    """already percent"""
    return f"{float(v):+.1f}%" if _ok(v) else "—"


# ---------- data (pressure_before_moves' loaders, this study's stores) ----------

def _obm() -> tuple[pd.Series, str | None]:
    try:
        raw, note = sd._stored(STORE_OBM, pb._obm_fetch)
    except Exception as exc:  # noqa: BLE001 - earlier stores, read only
        for path in (pb.STORE_OBM,):
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                break
        else:
            if not ocs.STORE_OBM.exists():
                raise
            raw = json.loads(ocs.STORE_OBM.read_text(encoding="utf-8"))["ge155"]
        note = f"OBM 這次取不到（{type(exc).__name__}），改讀之前其他研究存下來的資料。"
    s = pd.Series(raw, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), note


def _load() -> dict:
    problems: list[str] = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    try:
        raw, n = sd._stored(STORE_CM, fb._cm_fetch)
    except Exception:  # noqa: BLE001
        raw, n = pb._stored(pb.STORE_CM, fb._cm_fetch, fb.STORE_CM)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
    obm, n = _obm()
    problems += [n] if n else []
    try:
        ll, n = sd._stored(STORE_LL, sd._llama_fetch)
    except Exception:  # noqa: BLE001
        ll, n = pb._stored(pb.STORE_LL, sd._llama_fetch, fb.STORE_LL)
    problems += [n] if n else []
    sup = pd.Series(ll, dtype=float)
    sup.index = pd.to_datetime(sup.index)
    sup = sup.sort_index()
    sup = sup[sup.index < today].asfreq("D").ffill()
    etf, n = fb._etf_load()
    problems += [n] if n else []
    hourly, n = pi._hourly()
    problems += [n] if n else []
    cb, n = sd._coinbase()
    problems += [n] if n else []
    bs, n = sd._bitstamp()
    problems += [n] if n else []
    fund = pd.Series(dtype=float, index=pd.DatetimeIndex([]))  # not used by the composites
    return {"close": close, "cm": cm, "obm": obm, "sup": sup, "etf": etf, "hourly": hourly, "cb": cb, "bs": bs,
            "fund": fund, "oi": None, "problems": problems}


# ---------- series ----------

def _realtime(r: np.ndarray) -> np.ndarray:
    """mean over t-29..t minus mean over t-89..t-30 (at least 20 / 40 values): known at the close of t."""
    s = pd.Series(np.asarray(r, dtype=float))
    a = s.rolling(30, min_periods=20).mean()
    b = s.shift(30).rolling(60, min_periods=40).mean()
    return (a - b).values


def _pat_arr(ds: np.ndarray, db: np.ndarray) -> np.ndarray:
    a = np.sign(np.nan_to_num(ds, nan=0.0))
    b = np.sign(np.nan_to_num(db, nan=0.0))
    out = np.full(len(ds), 4, dtype=int)
    out[(a < 0) & (b > 0)] = 0
    out[(a > 0) & (b > 0)] = 1
    out[(a < 0) & (b < 0)] = 2
    out[(a > 0) & (b < 0)] = 3
    return out


def _states(close: pd.Series) -> pd.DataFrame:
    sig, _ = te._signals(close)
    c = close
    m55 = c.rolling(55, min_periods=55).mean()
    m50 = c.rolling(50, min_periods=50).mean()
    hi55 = c.rolling(55, min_periods=55).max()
    share = sig.mean(axis=1).where(sig.notna().all(axis=1))
    k = (share * sig.shape[1]).round()
    t5 = pd.Series(np.where(k <= 5, 0.0, np.where(k >= 12, 2.0, 1.0)), index=c.index).where(share.notna())
    return pd.DataFrame({
        "t1": sig["A50"],
        "t2": (c > m55).astype(float).where(m55.notna()),
        "t3": pd.Series(cr._breakout(c, 55, 20), index=c.index).where(hi55.notna()),
        "t4": sig["A200"],
        "t5": t5,
        "t6": (m50 > m50.shift(20)).astype(float).where(m50.shift(20).notna()),
    }, index=c.index)


def _forward(c: np.ndarray) -> dict:
    n = len(c)
    out = {}
    for h in H:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        out[h] = f
    low = pd.Series(c[::-1]).rolling(60, min_periods=60).min().values[::-1]
    dd = np.full(n, np.nan)
    dd[:n - 60] = np.minimum(low[1:n - 59] / c[:n - 60] - 1, 0.0)
    out["dd"] = dd
    out["up"] = np.where(np.isfinite(out[60]), (out[60] > 0).astype(float), np.nan)
    return out


def _boot_idx(n: int, rng, boot: int = BOOT) -> np.ndarray:
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (boot, nb))
    return ((st[:, :, None] + np.arange(BLOCK)) % n).reshape(boot, -1)[:, :n]


def _q(a) -> tuple:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)


def _roll_pat(pat: np.ndarray, u: float) -> np.ndarray:
    """Circular shift of the pattern series from its first defined day by k = 180 + u x (span - 360)."""
    out = pat.copy()
    first = int(np.argmax(pat < 4))
    span = len(pat) - first
    k = SHIFT_MIN + int(u * max(1, span - 2 * SHIFT_MIN))
    out[first:] = np.roll(pat[first:], k)
    return out


# ---------- Part 1 ----------

def _ba(typ: np.ndarray, cell: np.ndarray, ncell: int) -> float:
    """balanced accuracy of the best cell -> type assignment (typ 1 = rise, 0 = fall)."""
    nr, nf = (typ == 1).sum(), (typ == 0).sum()
    if nr == 0 or nf == 0:
        return float("nan")
    r = np.bincount(cell[typ == 1], minlength=ncell) / nr
    f = np.bincount(cell[typ == 0], minlength=ncell) / nf
    return float(0.5 * np.maximum(r, f).sum())


def _part1(R: dict) -> dict:
    pat, S, pos, pool = R["pat"], R["S"], R["pos"], R["pool"]
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for lag in (0, 10):
        out[lag] = {}
        for t in TREND:
            st = S[t].values
            nl = len(TLEV[t])
            rows = []
            for typ, key in ((1, "rise"), (0, "fall")):
                for p in pos[key]:
                    q = p - lag
                    pp, sv = pat[p - 1], st[q] if q >= 0 else np.nan
                    if pp < 4 and np.isfinite(sv):
                        rows.append((typ, int(pp), int(sv)))
            a = np.array(rows, dtype=int).reshape(-1, 3)
            typ, pp, sv = a[:, 0], a[:, 1], a[:, 2]
            cnt = {k: np.zeros((4, nl), dtype=int) for k in (1, 0)}
            for ty, x, y in a:
                cnt[ty][x, y] += 1
            # random pool days: pattern P(p-1), state at p - lag
            pq = pool[pool - lag >= 0]
            rp, rs = pat[pq - 1], st[pq - lag]
            okr = (rp < 4) & np.isfinite(rs)
            share = np.zeros((4, nl))
            if okr.any():
                np.add.at(share, (rp[okr], rs[okr].astype(int)), 1.0)
                share /= okr.sum()
            ba_s = _ba(typ, sv, nl)
            ba_p = _ba(typ, pp, 4)
            ba_j = _ba(typ, pp * nl + sv, 4 * nl)
            sp = np.empty(SHUF)
            ss = np.empty(SHUF)
            sb = np.empty(SHUF)
            for i in range(SHUF):
                sp[i] = _ba(typ, rng.permutation(pp) * nl + sv, 4 * nl)
                ss[i] = _ba(typ, pp * nl + rng.permutation(sv), 4 * nl)
                sb[i] = _ba(rng.permutation(typ), sv, nl)
            out[lag][t] = {"cnt": cnt, "share": share, "nr": int((typ == 1).sum()), "nf": int((typ == 0).sum()),
                           "ba_s": ba_s, "ba_p": ba_p, "ba_j": ba_j,
                           "p_padd": float((sp >= ba_j - 1e-12).mean()), "p_tadd": float((ss >= ba_j - 1e-12).mean()),
                           "p_s": float((sb >= ba_s - 1e-12).mean()), "n_pool": int(okr.sum())}
    # pattern alone null (shuffle type labels)
    a = [(1, int(pat[p - 1])) for p in pos["rise"] if pat[p - 1] < 4] + \
        [(0, int(pat[p - 1])) for p in pos["fall"] if pat[p - 1] < 4]
    a = np.array(a, dtype=int)
    bp = _ba(a[:, 0], a[:, 1], 4)
    nul = np.array([_ba(rng.permutation(a[:, 0]), a[:, 1], 4) for _ in range(SHUF)])
    out["pat_alone"] = {"ba": bp, "p": float((nul >= bp - 1e-12).mean())}
    return out


# ---------- Part 2 ----------

def _part2(R: dict) -> dict:
    idx, pat, S, fw = R["idx"], R["pat"], R["S"], R["fw"]
    base = R["base"]  # analysis days regardless of pattern
    strat = R["strat"]
    rng = np.random.default_rng(SEED + 2)
    days = np.flatnonzero(base)
    n = len(days)
    bidx = _boot_idx(n, rng)
    mets = (30, 60, "up", "dd")
    res = {}
    per_mask = {lab: ((idx[days] >= pd.Timestamp(a)) & (idx[days] <= pd.Timestamp(b))) for lab, a, b in PERIODS}
    for t in PART2_T:
        st = S[t].values[days]
        okst = np.isfinite(st)
        sv = np.where(okst, st, 0).astype(int)
        pp = pat[days]
        valid = okst & (pp < 4)
        key = np.where(valid, sv * 4 + np.minimum(pp, 3), 8)  # 8 = left out
        r = {"cells": {}, "match": {}, "period": {}}
        for m in mets:
            v = fw[m][days]
            v0 = np.nan_to_num(v)
            cnt = np.bincount(key, minlength=9)[:8]
            sm = np.bincount(key, weights=v0, minlength=9)[:8]
            # bootstrap
            bm = np.empty((BOOT, 8))
            for b in range(BOOT):
                ib = bidx[b]
                kb = key[ib]
                c_ = np.bincount(kb, minlength=9)[:8]
                s_ = np.bincount(kb, weights=v0[ib], minlength=9)[:8]
                with np.errstate(all="ignore"):
                    bm[b] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
            for c in range(8):
                sel = key == c
                d = r["cells"].setdefault(c, {"n": int(cnt[c])})
                d[m] = {"mean": float(sm[c] / cnt[c]) if cnt[c] else None,
                        "med": float(np.median(v[sel])) if cnt[c] else None, "ci": _q(bm[:, c])}
        # matched differences (60 and 30)
        skey = np.where(valid, (sv * 4 + np.minimum(pp, 3)) * NSTRAT + strat[days], 8 * NSTRAT)
        for h in (60, 30):
            v0 = np.nan_to_num(fw[h][days])
            real = _mtab(skey, v0)
            boots = []
            for b in range(BOOT):
                ib = bidx[b]
                boots.append(_mtab(skey[ib], v0[ib]))
            boots = np.array(boots)
            for c in range(8):
                r["match"].setdefault(c, {})[h] = {"diff": real[c], "ci": _q(boots[:, c]),
                                                    "raw": _rawdiff(key, fw[h][days], c)}
        for lab, pm in per_mask.items():
            v = fw[60][days]
            r["period"][lab] = {c: (int(((key == c) & pm).sum()),
                                    float(v[(key == c) & pm].mean()) if ((key == c) & pm).any() else None)
                                for c in range(8)}
        res[t] = r
    # placebo: shifted pattern, matched 60-day differences
    prng = np.random.default_rng(SEED + 3)
    hits = {t: np.zeros(8) for t in PART2_T}
    for _ in range(DRAWS):
        ps = _roll_pat(pat, float(prng.random()))
        pp = ps[days]
        for t in PART2_T:
            st = S[t].values[days]
            okst = np.isfinite(st)
            sv = np.where(okst, st, 0).astype(int)
            valid = okst & (pp < 4)
            skey = np.where(valid, (sv * 4 + np.minimum(pp, 3)) * NSTRAT + strat[days], 8 * NSTRAT)
            d = _mtab(skey, np.nan_to_num(fw[60][days]))
            for c in range(8):
                real = res[t]["match"][c][60]["diff"]
                if _ok(real) and _ok(d[c]):
                    hits[t][c] += (d[c] >= real) if real > 0 else (d[c] <= real)
    for t in PART2_T:
        for c in range(8):
            res[t]["match"][c]["placebo"] = float(hits[t][c] / DRAWS) if _ok(res[t]["match"][c][60]["diff"]) else None
    return res


def _mtab(skey: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Matched difference (pattern versus the other three within the same state) for the 8 cells."""
    cnt = np.bincount(skey, minlength=8 * NSTRAT + 1)[:8 * NSTRAT].reshape(2, 4, NSTRAT).astype(float)
    sm = np.bincount(skey, weights=v0, minlength=8 * NSTRAT + 1)[:8 * NSTRAT].reshape(2, 4, NSTRAT)
    out = np.full(8, np.nan)
    for s in range(2):
        for p in range(4):
            npk, spk = cnt[s, p], sm[s, p]
            nok = cnt[s].sum(axis=0) - npk
            sok = sm[s].sum(axis=0) - spk
            use = (npk > 0) & (nok > 0)
            if use.any():
                w = npk[use]
                out[s * 4 + p] = float((w * (spk[use] / w - sok[use] / nok[use])).sum() / w.sum())
    return out


def _rawdiff(key: np.ndarray, v: np.ndarray, c: int) -> float | None:
    s = c // 4
    a = key == c
    b = (key // 4 == s) & (key < 8) & ~a
    if not a.any() or not b.any():
        return None
    return float(np.nanmean(v[a]) - np.nanmean(v[b]))


# ---------- Part 3: the exam ----------

class _Exam(pi._Exam):
    """price_impact's exam (main_signal_check's four checks from a chosen first month) for rules built on a base trend
    rule (ma50 or 55/20) with an ENTRY flag (hold when flag and above the 20-day average) and an EXIT flag (cash when
    flag)."""

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first)
        self.bo = np.nan_to_num(np.asarray(msc._targets(close)["b5520"].values, dtype=float))
        npre = len(pre)
        self.cbo = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cbo.append(cr._breakout(pd.Series(c), 55, 20)[npre:][:self.days])
        for key, o in self.other.items():
            o["bo"] = cr._breakout(o["close"], 55, 20)
        self.test = (close.index >= first)

    @staticmethod
    def _mk(base, a20, ent, ext) -> np.ndarray:
        b = np.nan_to_num(np.asarray(base, dtype=float))
        if ext is not None:
            b = b * (1.0 - ext)
        if ent is not None:
            b = np.maximum(b, ent * a20)
        return b

    def btc_pos(self, spec) -> np.ndarray:
        bk, ent, ext = spec
        base = self.ma if bk == "ma50" else self.bo
        return self._mk(base, self.a20, self._at(ent, self.close.index, 0), self._at(ext, self.close.index, 0))

    def summary(self, spec, ki: int, sims: int = SIMS) -> dict:
        bk, ent, ext = spec
        pos = self.btc_pos(spec)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, msc.YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        fe, fx = self._at(ent, self.m.dates, 1), self._at(ext, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            base = self.cma[h] if bk == "ma50" else self.cbo[h]
            w = self._mk(base, self.ca20[h], fe, fx)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, msc.YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(msc.SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, _ = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts,
                               msc.YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            base = o["ma"] if bk == "ma50" else o["bo"]
            w = self._mk(base, o["a20"], self._at(ent, o["close"].index, o["lag"]),
                         self._at(ext, o["close"].index, o["lag"]))
            net2, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net2), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_edge": float(np.median(dr - self.dhold[0])),
             "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values()),
             "time": float(pos[self.test].mean())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


def _flag(pat: np.ndarray, cell, idx) -> pd.Series | None:
    if cell is None:
        return None
    return pd.Series((pat == cell).astype(float), index=idx)


def _specs(pat: np.ndarray, idx, cells: dict) -> dict:
    out = {}
    for t, (bk, e, x, ex) in FAM.items():
        ent, exi = _flag(pat, cells[t]["entry"], idx), _flag(pat, cells[t]["exit"], idx)
        out[bk] = (bk, None, None)
        out[e] = (bk, ent, None)
        out[x] = (bk, None, exi)
        out[ex] = (bk, ent, exi)
    return out


def _pick(R: dict) -> dict:
    idx, pat, S, fw = R["idx"], R["pat"], R["S"], R["fw"]
    disc = R["base"] & (idx + 60 * DAY <= DISC_END) & (pat < 4)
    test = R["base"] & (idx >= SPLIT) & (pat < 4)
    out = {}
    for t in ("t1", "t3"):
        st = S[t].values
        tab = {}
        for s in (0, 1):
            for p in range(4):
                md = disc & (st == s) & (pat == p)
                mt = test & (st == s) & (pat == p)
                tab[(s, p)] = (int(md.sum()), float(fw[60][md].mean()) if md.any() else None,
                               int(mt.sum()), float(fw[60][mt].mean()) if mt.any() else None)
        ent = [p for p in range(4) if tab[(0, p)][0] >= MIN_CELL]
        exi = [p for p in range(4) if tab[(1, p)][0] >= MIN_CELL]
        out[t] = {"tab": tab, "entry": max(ent, key=lambda p: tab[(0, p)][1]) if ent else None,
                  "exit": min(exi, key=lambda p: tab[(1, p)][1]) if exi else None}
    return out


def _episodes_entry(c, base, rule, lo):
    """E-type episodes from index lo: rule holds while base is in cash."""
    eps, kept = [], 0
    n = len(c)
    t = max(lo, 1)
    while t < n:
        if rule[t] > 0.5 and base[t] < 0.5:
            if rule[t - 1] > 0.5:
                kept += 1  # continued after a base exit (not an entry from cash)
                u = t + 1
                while u < n and rule[u] > 0.5 and base[u] < 0.5:
                    u += 1
                t = u
                continue
            u = t + 1
            while u < n and rule[u] > 0.5 and base[u] < 0.5:
                u += 1
            if u >= n:
                eps.append({"d": t, "end": None, "gain": float(c[-1] / c[t] - 1), "how": "open"})
            else:
                eps.append({"d": t, "end": u, "gain": float(c[u] / c[t] - 1), "how": "took" if base[u] > 0.5 else "cash"})
            t = u
        else:
            t += 1
    return eps, kept


def _episodes_exit(c, base, rule, lo):
    """X-type episodes from index lo: base holds while rule is in cash."""
    eps = []
    n = len(c)
    t = max(lo, 1)
    while t < n:
        if base[t] > 0.5 and rule[t] < 0.5:
            kind = "early" if rule[t - 1] > 0.5 else "blocked"
            u = t + 1
            while u < n and base[u] > 0.5 and rule[u] < 0.5:
                u += 1
            if u >= n:
                eps.append({"d": t, "end": None, "move": float(c[-1] / c[t] - 1), "kind": kind, "how": "open"})
            else:
                eps.append({"d": t, "end": u, "move": float(c[u] / c[t] - 1), "kind": kind,
                            "how": "back" if rule[u] > 0.5 else "base_out"})
            t = u
        else:
            t += 1
    return eps


def _ratio(close: pd.Series, pos_a, pos_b, a, b) -> float:
    change = close.pct_change().fillna(0.0).values
    na, _ = te._run(np.concatenate([[np.nan], pos_a[:-1]]), change, COST)
    nb, _ = te._run(np.concatenate([[np.nan], pos_b[:-1]]), change, COST)
    m = (close.index >= pd.Timestamp(a)) & (close.index <= pd.Timestamp(b))
    return float(np.prod(1 + na[m]) / np.prod(1 + nb[m]) - 1)


def _plain(R: dict, ex: _Exam, specs: dict) -> dict:
    close = R["close"]
    c = close.values.astype(float)
    lo = int(close.index.searchsorted(SPLIT))
    end = close.index[-1]
    out = {}
    for t, (bk, e, x, exx) in FAM.items():
        pb_ = ex.btc_pos(specs[bk])
        pe, px_ = ex.btc_pos(specs[e]), ex.btc_pos(specs[x])
        eps, kept = _episodes_entry(c, pb_, pe, lo)
        done = [z for z in eps if z["how"] != "open"]
        g = np.array([z["gain"] for z in done])
        took = [z for z in done if z["how"] == "took"]
        out[(t, "entry")] = {
            "n": len(eps), "done": len(done), "kept": kept, "open": len(eps) - len(done),
            "higher": int((g > 0).sum()), "took": len(took), "took_higher": int(sum(z["gain"] > 0 for z in took)),
            "avg_win": float(g[g > 0].mean()) if (g > 0).any() else None,
            "avg_loss": float(g[g <= 0].mean()) if (g <= 0).any() else None,
            "total": _ratio(close, pe, pb_, SPLIT, end),
            "y2021": _ratio(close, pe, pb_, "2020-01-01", "2021-12-31"), "y2022": _ratio(close, pe, pb_, "2022-01-01",
                                                                                         "2022-12-31"),
            "eps": eps}
        xe = _episodes_exit(c, pb_, px_, lo)
        dn = [z for z in xe if z["how"] != "open"]
        mv = np.array([z["move"] for z in dn])
        out[(t, "exit")] = {
            "n": len(xe), "done": len(dn), "early": sum(z["kind"] == "early" for z in xe),
            "blocked": sum(z["kind"] == "blocked" for z in xe), "open": len(xe) - len(dn),
            "fell": int((mv < 0).sum()), "rose": int((mv > 0).sum()),
            "avoid": float(mv[mv < 0].mean()) if (mv < 0).any() else None,
            "miss": float(mv[mv > 0].mean()) if (mv > 0).any() else None,
            "total": _ratio(close, px_, pb_, SPLIT, end),
            "y2021": _ratio(close, px_, pb_, "2020-01-01", "2021-12-31"),
            "y2022": _ratio(close, px_, pb_, "2022-01-01", "2022-12-31"),
            "n2021": sum(1 for z in xe if 2020 <= close.index[z["d"]].year <= 2021),
            "n2022": sum(1 for z in xe if close.index[z["d"]].year == 2022),
            "eps": xe}
    return out


# ---------- Part 4 ----------

def _conds(R: dict) -> dict:
    rank, S, close = R["rank"], R["S"], R["close"]
    c = close.values.astype(float)
    hi = close.rolling(365, min_periods=365).max().values
    lr = np.log(close).diff()
    vol = lr.rolling(20, min_periods=20).std(ddof=1)
    vmed = vol.shift(1).rolling(365, min_periods=180).median()
    out = {}

    def b(cond, ok):
        return np.where(ok, cond.astype(float), np.nan)

    r = rank["NET30"].values
    out["net"] = b(r >= 50, np.isfinite(r))
    d = _realtime(rank["S_old"].values)
    out["old"] = b(d > 0, np.isfinite(d))
    r = rank["ACT"].values
    out["act"] = b(r >= 50, np.isfinite(r))
    out["t4"] = S["t4"].values
    out["t6"] = S["t6"].values
    out["dd"] = b(c / hi - 1 < -0.25, np.isfinite(hi))
    out["vol"] = b(vol.values > vmed.values, np.isfinite(vol.values) & np.isfinite(vmed.values))
    return out


def _part4(R: dict, cells: dict) -> dict:
    idx, pat, S, fw = R["idx"], R["pat"], R["S"], R["fw"]
    disc = R["base"] & (idx + 60 * DAY <= DISC_END) & (pat < 4)
    test = R["base"] & (idx >= SPLIT) & (pat < 4)
    cond = _conds(R)
    st = S["t1"].values
    out = {}
    for which, s, p in (("entry", 0, cells["t1"]["entry"]), ("exit", 1, cells["t1"]["exit"])):
        if p is None:
            out[which] = None
            continue
        cellm = (st == s) & (pat == p)
        rows = {}
        for k in COND:
            v = cond[k]
            res = {}
            for lab, m in (("disc", disc & cellm), ("test", test & cellm)):
                y = m & (v == 1)
                nn = m & (v == 0)
                my = float(fw[60][y].mean()) if y.any() else None
                mn = float(fw[60][nn].mean()) if nn.any() else None
                res[lab] = {"ny": int(y.sum()), "nn": int(nn.sum()), "my": my, "mn": mn,
                            "diff": (my - mn) if _ok(my) and _ok(mn) else None}
            rows[k] = res
        elig = [k for k in COND if rows[k]["disc"]["ny"] >= MIN_SIDE and rows[k]["disc"]["nn"] >= MIN_SIDE
                and _ok(rows[k]["disc"]["diff"])]
        best = max(elig, key=lambda k: abs(rows[k]["disc"]["diff"])) if elig else None
        out[which] = {"rows": rows, "best": best, "n_disc": int((disc & cellm).sum()),
                      "n_test": int((test & cellm).sum())}
    return out


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    idx = close.index
    n = len(idx)
    raw, rank, _first = pb._pressures(data)
    ds = _realtime(rank["SELL_C"].values)
    db = _realtime(rank["BUY_C"].values)
    pat = _pat_arr(ds, db)
    S = _states(close)
    c = close.values.astype(float)
    fw = _forward(c)
    past = np.full(n, np.nan)
    past[30:] = c[30:] / c[:-30] - 1
    vol = np.log(close).diff().rolling(20, min_periods=20).std(ddof=1).values
    base = (idx >= FROM) & np.isfinite(fw[60]) & np.isfinite(past) & np.isfinite(vol)
    dq = np.quantile(past[base], np.linspace(0, 1, 11))
    vq = np.quantile(vol[base], [0, 1 / 3, 2 / 3, 1])
    dec = np.clip(np.searchsorted(dq, np.nan_to_num(past), side="right") - 1, 0, 9)
    ter = np.clip(np.searchsorted(vq, np.nan_to_num(vol), side="right") - 1, 0, 2)
    strat = dec * 3 + ter
    ev = {"rise": pb._rises(close), "fall": pb._falls(close)}
    pos = {k: np.array([idx.get_loc(e["d"]) for e in ev[k]], dtype=int) for k in ev}
    lo_i = idx.get_loc(FROM)
    cand = np.arange(lo_i, n - 30)
    allp = np.concatenate([pos["rise"], pos["fall"]])
    pool = cand[np.min(np.abs(cand[:, None] - allp[None, :]), axis=1) >= pb.GAP]
    R = {"close": close, "idx": idx, "raw": raw, "rank": rank, "ds": ds, "db": db, "pat": pat, "S": S, "fw": fw,
         "base": base, "strat": strat, "ev": ev, "pos": pos, "pool": pool}
    # T1 vs T2
    both = (idx >= FROM) & S["t1"].notna().values & S["t2"].notna().values
    R["t12"] = (int((S["t1"].values[both] != S["t2"].values[both]).sum()), int(both.sum()))
    R["t12_ev"] = sum(int(S["t1"].values[p] != S["t2"].values[p]) for k in pos for p in pos[k])
    R["p1"] = _part1(R)
    R["p2"] = _part2(R)
    R["t_p12"] = time.time() - t0
    cells = _pick(R)
    R["cells"] = cells
    rows_h, problem = cr._hourly("BTCUSDT")
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    other, probs = msc._load_other()
    R["problems_exam"] = ([problem] if problem else []) + probs
    ex = _Exam(close, other, price, b0, pre, SPLIT)
    specs = _specs(pat, idx, cells)
    s = {k: ex.summary(sp, KI[k], SIMS) for k, sp in specs.items()}
    small = {k: ex.summary(specs[k], KI[k], PL_SIMS) for k in RULES}
    hits = {k: 0 for k in RULES}
    rng = np.random.default_rng(SEED + 9)
    for _ in range(DRAWS):
        ps = _roll_pat(pat, float(rng.random()))
        sp2 = _specs(ps, idx, cells)
        for k in RULES:
            x = ex.summary(sp2[k], KI[k], PL_SIMS)
            hits[k] += int(sd._score_ge(x, small[k]))
    placebo = {k: hits[k] / DRAWS for k in RULES}
    holm = ef._holm(dict(placebo))
    sup = {}
    for t, (bk, e, x, exx) in FAM.items():
        for k in (e, x, exx):
            slack = LOSS_SLACK if k in ("e1", "e1b") else 0.0
            items = {"rank": sd._score_ge(s[k], s[bk]),
                     "loss": s[k]["c_deep_med"] >= s[bk]["c_deep_med"] - slack - 1e-9,
                     "placebo": placebo[k] <= PLACEBO_BAR}
            sup[k] = {"items": items, "bar": all(items.values()), "holm": holm.get(k, False)}
            sup[k]["ok"] = sup[k]["bar"] and sup[k]["holm"]
    R.update({"ex": ex, "specs": specs, "s": s, "small": small, "placebo": placebo, "holm": holm, "sup": sup})
    R["plain"] = _plain(R, ex, specs)
    R["p4"] = _part4(R, cells)
    R["secs"] = time.time() - t0
    return R


# ---------- text helpers ----------

def _cellname(t: str, c: int) -> tuple[str, str]:
    return PAT[c % 4], TLEV[t][c // 4]


def _p1_table(R: dict, t: str, lag: int) -> dict:
    d = R["p1"][lag][t]
    rows = []
    for p in range(4):
        for s in range(len(TLEV[t])):
            sh = d["share"][p, s]
            rows.append([PAT[p], TLEV[t][s], f"{d['cnt'][1][p, s]} / {d['nr']}", f"{d['cnt'][0][p, s]} / {d['nf']}",
                         f"{sh * d['nr']:.1f}", f"{sh * d['nf']:.1f}"])
    when = "事件當天" if lag == 0 else f"事件前 {lag} 天"
    return {"caption": (f"1. 型態 × {TNAME[t]}（{when}的狀態）：大漲前、大跌前各有幾次。型態＝事件前 30 天比再前 60 天，"
                        f"賣方合計、買方合計各自是升還是降。最後兩欄＝隨機非事件日（{d['n_pool']:,} 天）在這一格的比例，"
                        f"換算成「同樣次數會有幾次」。"),
            "columns": ["型態", "趨勢狀態", "大漲前", "大跌前", "隨機日（換成大漲次數）", "隨機日（換成大跌次數）"],
            "rows": rows}


def _ba_table(R: dict, lag: int) -> dict:
    rows = []
    pa = R["p1"]["pat_alone"]
    for t in TREND:
        d = R["p1"][lag][t]
        rows.append([TNAME[t], f"{d['nr']} / {d['nf']}", f"{d['ba_s'] * 100:.0f}%（{_share(d['p_s'])}）",
                     f"{pa['ba'] * 100:.0f}%（{_share(pa['p'])}）", f"{d['ba_j'] * 100:.0f}%",
                     _share(d["p_padd"]), _share(d["p_tadd"])])
    when = "事件當天" if lag == 0 else f"事件前 {lag} 天"
    return {"caption": (f"1. 分得開的程度（趨勢狀態取{when}）：50%＝完全分不出大漲前還是大跌前，100%＝完全分開。"
                        "括號＝把「大漲／大跌」標籤打亂 2,000 次，至少這麼分得開的比例。格子越多，分得開的程度本來就會變高，"
                        "所以「型態多加的」＝把型態打亂、保留趨勢狀態後，至少這麼分得開的比例（越小代表型態真的多加了東西）；"
                        "「趨勢多加的」反過來，打亂趨勢狀態。"),
            "columns": ["趨勢狀態", "事件數 漲/跌", "只看趨勢（打亂標籤）", "只看型態（打亂標籤）", "兩個一起",
                        "型態多加的：打亂型態", "趨勢多加的：打亂趨勢"],
            "rows": rows}


def _event_rows(R: dict, key: str) -> list:
    S, pat, idx = R["S"], R["pat"], R["idx"]
    rows = []
    for e, p in zip(R["ev"][key], R["pos"][key]):
        cells = []
        for t in TREND:
            st = S[t].values
            seq = []
            for q in (p - 30, p - 10, p):
                v = st[q] if q >= 0 else np.nan
                seq.append(TMINI[t][int(v)] if np.isfinite(v) else "—")
            cells.append("→".join(seq))
        pp = pat[p - 1]
        rows.append([_d(e["d"]) + ("" if e["done"] else "（進行中）"), pb._pc(e["size"]),
                     PAT[pp] if pp < 4 else PAT_FLAT] + cells)
    return rows


def _cont_rows(R: dict) -> list:
    rows = []
    c = R["close"].values
    for key in ("rise", "fall"):
        for e, p in zip(R["ev"][key], R["pos"][key]):
            px = c[p - 1] / c[p - 31] - 1 if p >= 31 else np.nan
            rows.append([TYPE_CN[key], _d(e["d"]), _sg(R["ds"][p - 1]), _sg(R["db"][p - 1]),
                         _sg(R["raw"]["NET30"].values[p - 1]), pb._pc(px)])
    return rows


def _p2_table(R: dict, t: str) -> dict:
    r = R["p2"][t]
    rows = []
    for c in range(8):
        d = r["cells"][c]
        pn, sn = _cellname(t, c)
        rows.append([pn, sn, f"{d['n']:,}", f"{_p(d[30]['mean'])}｜{_p(d[30]['med'])}",
                     f"{_p(d[60]['mean'])} {_ci(*d[60]['ci'])}", _p(d[60]["med"]), _share(d["up"]["mean"]),
                     f"{_p(d['dd']['mean'])} {_ci(*d['dd']['ci'])}"])
    return {"caption": (f"2. 所有日子（2012 年起）：型態 × {TNAME[t]}，之後的價格。「之後 60 天是漲的」＝60 天後收盤比當天高的"
                        "比例；「60 天內最深跌幅」＝之後 60 天最低收盤比當天低多少（平均）。方括號＝以 60 天為一段的區塊"
                        "重抽 1,000 次的 90% 範圍。相鄰的日子幾乎重疊，天數多不代表獨立的次數多。"),
            "columns": ["型態", "趨勢狀態", "天數", "之後 30 天：平均｜中位", "之後 60 天平均［90% 範圍］",
                        "之後 60 天中位", "之後 60 天是漲的", "60 天內最深跌幅平均［90% 範圍］"],
            "rows": rows}


def _match_table(R: dict) -> dict:
    rows = []
    for t in PART2_T:
        for c in range(8):
            m = R["p2"][t]["match"][c]
            pn, sn = _cellname(t, c)
            rows.append([TSHORT[t], sn, pn, f"{R['p2'][t]['cells'][c]['n']:,}", _pt(m[60]["raw"]),
                         f"{_pt(m[60]['diff'])} {_ci(*m[60]['ci'], f=_pt)}",
                         f"{_pt(m[30]['diff'])} {_ci(*m[30]['ci'], f=_pt)}", _share(m.get("placebo"))])
    return {"caption": ("2. 同一個趨勢狀態裡，這個型態 減 其他三種型態：之後 60 天報酬差多少。「配對後」＝先把日子按「過去 30 天"
                        "漲跌（十等分）×20 天波動（三等分）」分成 30 組，只在同一組裡比，避免只是因為之前漲跌不同。方括號＝區塊重抽"
                        "的 90% 範圍；「隨機錯位」＝把型態整條往後挪至少 180 天再算 200 次，至少這麼極端的比例（越小越不像巧合）。"),
            "columns": ["趨勢", "狀態", "型態", "天數", "直接相減（60 天）", "配對後（60 天）［90%］",
                        "配對後（30 天）［90%］", "隨機錯位"],
            "rows": rows}


def _period_table(R: dict) -> dict:
    rows = []
    for t in PART2_T:
        for c in range(8):
            pn, sn = _cellname(t, c)
            row = [TSHORT[t], sn, pn]
            for lab, _a, _b in PERIODS:
                nn, mm = R["p2"][t]["period"][lab][c]
                row.append(f"{nn:,} 天｜{_p(mm)}" if nn else "—")
            rows.append(row)
    return {"caption": "2. 分時期：每一格的天數｜之後 60 天平均報酬。看同一格在不同時期是不是同一個方向。",
            "columns": ["趨勢", "狀態", "型態"] + [lab for lab, _a, _b in PERIODS], "rows": rows}


def _pick_table(R: dict) -> dict:
    rows = []
    for t in ("t1", "t3"):
        cl = R["cells"][t]
        for s in (0, 1):
            for p in range(4):
                nd, md, nt, mt = cl["tab"][(s, p)]
                mark = ""
                if s == 0 and cl["entry"] == p:
                    mark = "← 進場型態（挑中）"
                if s == 1 and cl["exit"] == p:
                    mark = "← 出場型態（挑中）"
                rows.append([TSHORT[t], TLEV[t][s], PAT[p], f"{nd:,}", _p(md), f"{nt:,}", _p(mt), mark])
    return {"caption": (f"3. 只用「發現期」（之後 60 天在 2018-12-31 以前結束的日子）挑格子：趨勢關著（現金）時，之後 60 天"
                        f"平均最高、至少 {MIN_CELL} 天的型態＝進場型態；趨勢開著（持有）時，平均最低、至少 {MIN_CELL} 天的"
                        "＝出場型態。旁邊是「檢驗期」（2019 年起）同一格的數字，只是描述。"),
            "columns": ["趨勢", "狀態", "型態", "發現期天數", "發現期 60 天平均", "檢驗期天數", "檢驗期 60 天平均", "挑中"],
            "rows": rows}


def _exam_rows(R: dict) -> list:
    rows = []
    s = R["s"]
    for t, keys in FAM.items():
        for k in keys:
            x = s[k]
            ch = x["checks"]
            if k in RULES:
                sp = R["sup"][k]
                pl = f"{_share(R['placebo'][k])}"
                hv = "通過" if sp["holm"] else "沒通過"
                ok = "達標" if sp["ok"] else ("單項達標、Holm 沒過" if sp["bar"] else "沒達標")
            else:
                pl, hv, ok = "—", "—", "基準"
            rows.append([NAME[k], f"{x['low'] * 100:.0f}%" + ("" if x["eligible"] else "（虧損條件未達）"),
                         "｜".join(f"{v * 100:.0f}%" for v in ch),
                         f"{msc._pct(x['c_deep_med'])}｜{msc._pct(x['c_deep_worst'])}", _share(x["time"]), pl, hv, ok])
    return rows


def _rule_detail(R: dict) -> list:
    rows = []
    s = R["s"]
    for t, (bk, e, x, exx) in FAM.items():
        for k in (e, x, exx):
            it = R["sup"][k]["items"]
            rows.append([NAME[k], f"{s[k]['low'] * 100:.0f}% 對 {s[bk]['low'] * 100:.0f}%｜{'是' if it['rank'] else '否'}",
                         f"{msc._pct(s[k]['c_deep_med'])} 對 {msc._pct(s[bk]['c_deep_med'])}｜{'是' if it['loss'] else '否'}",
                         f"{_share(R['placebo'][k])}｜{'是' if it['placebo'] else '否'}",
                         f"{msc._pp(s[k]['c_edge'])} 對 {msc._pp(s[bk]['c_edge'])}",
                         f"{s[k]['time'] * 100:.0f}% 對 {s[bk]['time'] * 100:.0f}%"])
    return rows


def _plain_rows(R: dict) -> list:
    rows = []
    for t in ("t1", "t3"):
        a = R["plain"][(t, "entry")]
        name = "50 日均線" if t == "t1" else "55／20"
        rows.append([f"{name}：照著進場（E1）", f"{a['n']} 次（{a['open']} 次還沒結束；另有 {a['kept']} 段是基準出場後繼續抱）",
                     f"{a['higher']} / {a['done']} 次結束時比進場高；基準接手的 {a['took']} 次裡 {a['took_higher']} 次較高",
                     f"賺的平均 {_p(a['avg_win'])}｜賠的平均 {_p(a['avg_loss'])}",
                     f"整段 {_p(a['total'])}｜2020–21 {_p(a['y2021'])}｜2022 {_p(a['y2022'])}"])
        b = R["plain"][(t, "exit")]
        rows.append([f"{name}：照著出場（X1）",
                     f"{b['n']} 次（提早出場 {b['early']}、基準剛進場但被擋住 {b['blocked']}；{b['open']} 次還沒結束）",
                     f"出場期間價格下跌 {b['fell']} 次、上漲 {b['rose']} 次",
                     f"躲過的跌幅平均 {_p(b['avoid'])}｜錯過的漲幅平均 {_p(b['miss'])}",
                     f"整段 {_p(b['total'])}｜2020–21 {_p(b['y2021'])}（{b['n2021']} 次）｜2022 {_p(b['y2022'])}（{b['n2022']} 次）"])
    return rows


def _p4_table(R: dict, which: str) -> dict | None:
    d = R["p4"].get(which)
    if not d:
        return None
    rows = []
    for k in COND:
        a, b = d["rows"][k]["disc"], d["rows"][k]["test"]
        same = "—"
        if _ok(a["diff"]) and _ok(b["diff"]):
            same = "同方向" if a["diff"] * b["diff"] > 0 else "相反"
        rows.append([CNAME[k] + ("（發現期分得最開）" if d["best"] == k else ""),
                     f"{a['ny']}｜{a['nn']}", f"{_p(a['my'])}｜{_p(a['mn'])}", _pt(a["diff"]),
                     f"{b['ny']}｜{b['nn']}", f"{_p(b['my'])}｜{_p(b['mn'])}", _pt(b["diff"]), same])
    cell = R["cells"]["t1"][which]
    st = "關" if which == "entry" else "開"
    return {"caption": (f"4. 探索（不是規則、沒有檢驗）：在{'進場' if which == 'entry' else '出場'}格子裡（50 日均線{st}＋"
                        f"「{PAT[cell]}」；發現期 {d['n_disc']} 天、檢驗期 {d['n_test']} 天），再多加一個條件，把格子切成"
                        "「條件成立｜不成立」兩半，看之後 60 天平均報酬差多少。發現期分得最開的（兩邊都至少 30 天），"
                        "在檢驗期是不是同一個方向。這裡是下一次事先登記檢驗的起點。"),
            "columns": ["多加的條件", "發現期天數 是｜否", "發現期 60 天平均 是｜否", "發現期差", "檢驗期天數 是｜否",
                        "檢驗期 60 天平均 是｜否", "檢驗期差", "方向"],
            "rows": rows}


# ---------- verdict ----------

def _within_text(R: dict, t: str, p: int, lag: int = 0) -> str:
    d = R["p1"][lag][t]
    parts = []
    for s in range(len(TLEV[t])):
        parts.append(f"{TMINI[t][s]}：大漲 {d['cnt'][1][p, s]}、大跌 {d['cnt'][0][p, s]}")
    return "；".join(parts)


def _verdict(R: dict) -> tuple[str, list]:
    lines = []
    p1 = R["p1"]
    d1 = p1[0]["t1"]
    lines.append(f"量到的（事件前）：「兩邊都升」型態在 50 日均線{_within_text(R, 't1', 1)}"
                 f"（共 {d1['nr']} 次大漲、{d1['nf']} 次大跌有資料）；55／20 {_within_text(R, 't3', 1)}。")
    better = [t for t in TREND if p1[0][t]["p_padd"] <= 0.05]
    tadd = [t for t in TREND if p1[0][t]["p_tadd"] <= 0.05]
    lines.append("量到的（分得開的程度，事件當天的趨勢）：只看趨勢狀態就分得很開——"
                 + "、".join(f"{TSHORT[t]} {p1[0][t]['ba_s'] * 100:.0f}%" for t in TREND)
                 + f"；只看型態 {p1['pat_alone']['ba'] * 100:.0f}%。"
                 + ("加上型態後，打亂型態也很難做到這麼好的：" + "、".join(TSHORT[t] for t in better) + "。" if better
                    else "加上型態後，沒有一個趨勢狀態比「隨便打亂型態」明顯更好（打亂後至少同樣好的比例都超過 5%）。")
                 + ("反過來，趨勢狀態確實讓型態分得更開：" + "、".join(TSHORT[t] for t in tadd) + "。" if tadd else "")
                 + "注意：大漲起點是低點、大跌起點是高點，事件當天在均線下或上本來就幾乎是定義造成的，這份資料分不開"
                   "「趨勢在指方向」和「定義讓它看起來在指方向」。")
    d10 = p1[10]["t1"]
    lines.append(f"量到的（事件前 10 天的 50 日均線）：分得開的程度只看趨勢 {d10['ba_s'] * 100:.0f}%、加上型態 "
                 f"{d10['ba_j'] * 100:.0f}%（打亂型態至少同樣好的比例 {_share(d10['p_padd'])}）。")
    # Part 2, T1
    sig = []
    for c in range(8):
        m = R["p2"]["t1"]["match"][c]
        lo, hi = m[60]["ci"]
        if _ok(lo) and _ok(hi) and (lo > 0 or hi < 0) and _ok(m.get("placebo")) and m["placebo"] <= 0.05:
            pn, sn = _cellname("t1", c)
            sig.append(f"50 日均線{TMINI['t1'][c // 4]}＋{pn}（配對後 {_pt(m[60]['diff'])}，{R['p2']['t1']['cells'][c]['n']:,} 天，"
                       f"隨機錯位 {_share(m['placebo'])}）")
    lines.append("量到的（所有日子，50 日均線）：同一個均線狀態下，型態之間配對後的 60 天差距，90% 範圍不含 0、而且隨機錯位 ≤5% 的："
                 + ("；".join(sig) if sig else "沒有") + "。相鄰日子重疊、獨立的次數很少，這只是描述。")
    # added after seeing the results: multiple-comparison count and the by-period check of that cell
    per = R["p2"]["t1"]["period"]
    on_sdbu = [per[lab][4][1] for lab, _a, _b in PERIODS]
    lowest = [lab for lab, _a, _b in PERIODS
              if _ok(per[lab][4][1]) and all(_ok(per[lab][c][1]) and per[lab][4][1] <= per[lab][c][1] for c in (5, 6, 7))]
    lines.append(f"看完結果才加的：第 2 部分一共比了 24 格（3 種趨勢×8 格），用 5% 的隨機錯位當門檻，光靠運氣約 1.2 格會過，"
                 f"所以上面這一兩格不能當成已經證實。「50 日均線開＋賣壓降＋買壓升」之後 60 天平均分時期是 "
                 + "、".join(f"{lab} {_p(v)}" for (lab, _a, _b), v in zip(PERIODS, on_sdbu))
                 + f"，在均線開的四種型態裡最低的時期：{'、'.join(lowest) if lowest else '沒有'}。"
                 "推測（沒有檢驗）：均線開著時「賣壓降、買壓升」可能出現在一段上漲已經走完一大段、交易所進出都在退燒的時候，"
                 "也可能只是早年組成（買方只有提走一項）造成的，這份資料分不開。")
    cl = R["cells"]
    lines.append(f"發現期（2018 年以前）挑中的：50 日均線關時的進場型態「{PAT[cl['t1']['entry']] if cl['t1']['entry'] is not None else '沒有'}」、"
                 f"開時的出場型態「{PAT[cl['t1']['exit']] if cl['t1']['exit'] is not None else '沒有'}」；55／20 版本是"
                 f"「{PAT[cl['t3']['entry']] if cl['t3']['entry'] is not None else '沒有'}」和"
                 f"「{PAT[cl['t3']['exit']] if cl['t3']['exit'] is not None else '沒有'}」。")
    ok = [k for k in RULES if R["sup"][k]["ok"]]
    bar = [k for k in RULES if R["sup"][k]["bar"]]
    s = R["s"]
    lines.append(f"量到的（檢驗期 2019 年起的公平考試）：六條規則裡達到事先定的標準的 {len(bar)} 條"
                 + (f"（{'、'.join(SHORT[k] for k in bar)}）" if bar else "")
                 + f"，再做 Holm 多重比較後還算數的 {len(ok)} 條" + (f"（{'、'.join(SHORT[k] for k in ok)}）" if ok else "")
                 + f"；光靠運氣六條裡約 0.3 條會過隨機錯位那一關。排名分數：50 日均線 {s['ma50']['low'] * 100:.0f}%、"
                 + "、".join(f"{SHORT[k]} {s[k]['low'] * 100:.0f}%" for k in ("e1", "x1", "ex"))
                 + f"；55／20 {s['b5520']['low'] * 100:.0f}%、"
                 + "、".join(f"{SHORT[k]} {s[k]['low'] * 100:.0f}%" for k in ("e1b", "x1b", "exb")) + "。")
    a = R["plain"][("t1", "entry")]
    b = R["plain"][("t1", "exit")]
    lines.append(f"照著進場，賺不賺？（50 日均線版，2019 年起）量到的：提早進場 {a['n']} 次，結束時比進場高的 {a['higher']} / {a['done']} 次，"
                 f"賺的平均 {_p(a['avg_win'])}、賠的平均 {_p(a['avg_loss'])}；整段下來比只用 50 日均線 {_p(a['total'])}。")
    lines.append(f"照著出場，少不少賠？（50 日均線版，2019 年起）量到的：比 50 日均線少抱的 {b['n']} 段（{b['early']} 段是提早出場、"
                 f"{b['blocked']} 段是均線剛轉持有但被型態擋住），期間價格跌 {b['fell']} 次、漲 {b['rose']} 次，"
                 f"躲過的跌幅平均 {_p(b['avoid'])}、錯過的漲幅平均 {_p(b['miss'])}；整段 {_p(b['total'])}，"
                 f"2022 年 {_p(b['y2022'])}、2020–21 年 {_p(b['y2021'])}。")
    for which, nm in (("entry", "進場"), ("exit", "出場")):
        d = R["p4"].get(which)
        if d and d["best"]:
            k = d["best"]
            a_, b_ = d["rows"][k]["disc"]["diff"], d["rows"][k]["test"]["diff"]
            same = ("同方向" if _ok(b_) and a_ * b_ > 0 else ("方向相反" if _ok(b_) else "檢驗期沒有資料"))
            lines.append(f"探索（不是檢驗）：{nm}格子裡，發現期最能再切開的條件是「{CNAME[k]}」（差 {_pt(a_)}），"
                         f"檢驗期 {_pt(b_)}，{same}。")
    # added after seeing the results: the conditions whose sign held in both halves of both cells
    both = [k for k in COND if all(R["p4"].get(w) and _ok(R["p4"][w]["rows"][k]["disc"]["diff"])
                                   and _ok(R["p4"][w]["rows"][k]["test"]["diff"])
                                   and R["p4"][w]["rows"][k]["disc"]["diff"] * R["p4"][w]["rows"][k]["test"]["diff"] > 0
                                   for w in ("entry", "exit"))]
    if both:
        lines.append("看完結果才加的：在進場和出場兩個格子裡，發現期和檢驗期都同方向的條件："
                     + "、".join(f"「{CNAME[k]}」" for k in both)
                     + "。這是挑過之後才看到的，只能當下一次事先登記檢驗的候選。")
    lines.append("看不到的：組成會變（2017-08 以前買方只有提走一項）、交易所標籤用今天的往回套、事件只有 23＋28 次而且常常相隔不到 90 天、"
                 "檢驗期只有大約兩個完整循環；沒看到不代表沒有，只是用這個定義和這份資料看不出來。")
    return " ".join(lines), lines


# ---------- today ----------

def _now(R: dict) -> list[dict]:
    close = R["close"]
    last = len(close) - 1
    ex, specs = R["ex"], R["specs"]
    word = lambda v: "持有" if v > 0.5 else "現金"  # noqa: E731
    pos = {k: ex.btc_pos(specs[k])[-1] for k in specs}
    out = [{"label": "最後一根收完的日線", "value": f"{_d(close.index[-1])}，比特幣 {close.iloc[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜E1｜X1", "value": f"{word(pos['ma50'])}｜{word(pos['e1'])}｜{word(pos['x1'])}"},
           {"label": "比特幣：55／20｜E1（55／20）｜X1（55／20）",
            "value": f"{word(pos['b5520'])}｜{word(pos['e1b'])}｜{word(pos['x1b'])}"}]
    dsv, dbv = R["ds"][last], R["db"][last]
    p = int(R["pat"][last])

    def updown(v):
        return ("在升" if v > 0 else "在降") if _ok(v) and v != 0 else "沒變或沒資料"
    out.append({"label": "今天的型態（近 30 天比再前 60 天）",
                "value": f"賣方合計{updown(dsv)}（{_sg(dsv)} 點）、買方合計{updown(dbv)}（{_sg(dbv)} 點）→ "
                         f"{PAT[p] if p < 4 else PAT_FLAT}；淨提走比例（30 天）{_sg(R['raw']['NET30'].values[last])}%"})
    S = R["S"]
    for t in TREND:
        v = S[t].values[last]
        out.append({"label": f"今天：{TNAME[t]}", "value": TLEV[t][int(v)] if np.isfinite(v) else "沒有資料"})
    for t in PART2_T:
        v = S[t].values[last]
        if p < 4 and np.isfinite(v):
            c = int(v) * 4 + p
            d = R["p2"][t]["cells"][c]
            out.append({"label": f"今天所在的格子（{TSHORT[t]}）",
                        "value": f"{TLEV[t][int(v)]}＋{PAT[p]}：2012 年起 {d['n']:,} 天，之後 60 天平均 {_p(d[60]['mean'])}、"
                                 f"中位 {_p(d[60]['med'])}、{_share(d['up']['mean'])} 是漲的、60 天內最深跌幅平均 "
                                 f"{_p(d['dd']['mean'])}（過去的描述，不是預測）"})
    out.append({"label": "這幾格的意思", "value": "只描述規則和型態在最後一根日線收盤後停在哪裡，不是買賣建議"})
    return out


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    R = _compute(data)
    close = data["close"]
    verdict, _ = _verdict(R)
    tables = []
    for key in ("rise", "fall"):
        tables.append({"caption": (f"1. 每一次{TYPE_CN[key]}（{len(R['ev'][key])} 次）：前 30 天的買賣型態，和每個趨勢狀態在"
                                   f"事件前 30 天→前 10 天→當天的樣子。50日線／55日線／200日線：開或上＝收盤在均線上；"
                                   f"55／20：持＝持有中、出＝出場中；17條：低／中／高＝17 條順勢規則裡看多 ≤5／6–11／≥12 條；"
                                   f"50日線方向：升／降＝50 日均線比 20 天前高／低。日期是事後才知道的。"),
                       "columns": ["日期", "幅度", "前 30 天型態"] + [TSHORT[t] for t in TREND],
                       "rows": _event_rows(R, key)})
    tables.append({"caption": ("1. 每一次事件前 30 天的連續數字：賣方合計、買方合計的變化（排名點數，近 30 天平均 減 再前 60 天平均）、"
                               "事件前一天的淨提走比例（30 天，幣數，正＝提走多），以及事件前 30 天價格變化。"),
                   "columns": ["事件", "日期", "Δ賣方合計", "Δ買方合計", "淨提走比例 %", "前 30 天價格"],
                   "rows": _cont_rows(R)})
    tables.append(_ba_table(R, 0))
    tables.append(_ba_table(R, 10))
    for t in TREND:
        tables.append(_p1_table(R, t, 0))
    tables.append(_p1_table(R, "t1", 10))
    for t in PART2_T:
        tables.append(_p2_table(R, t))
    tables.append(_match_table(R))
    tables.append(_period_table(R))
    tables.append(_pick_table(R))
    tables.append({"caption": ("3. 檢驗期（2019 年起算）的公平考試：排名分數＝四項「每年報酬贏過一直抱著」的比例裡最低的一項"
                               "（比特幣每月起算、比特幣起算×24 個收盤時間、照一般人方式做、其他幣），括號註明最深虧損條件沒達到的。"
                               "最深虧損＝24 個收盤時間×起算日每一格的最深虧損，取中位數｜最深的一格。在場時間＝2019 年起比特幣日線"
                               "持有的天數比例。隨機錯位＝把型態整條挪至少 180 天、200 次（一般人模擬減為每個時鐘 20 次，真的規則也"
                               "用 20 次重算），排名分數至少一樣好的比例。Holm＝六條一起做多重比較修正。"),
                   "columns": ["規則", "排名分數", "四項：起算日｜時鐘｜一般人｜其他幣", "最深虧損：中位｜最深", "在場時間",
                               "隨機錯位", "Holm", "結果"],
                   "rows": _exam_rows(R)})
    tables.append({"caption": ("3. 每條規則和自己的基準比（E1 的最深虧損可以差到 5 個百分點以內；X1、EX 不能更差）。"
                               "最後兩欄只是參考：時鐘格的每年報酬比一直抱著多幾個百分點（中位數），以及在場時間。"),
                   "columns": ["規則", "排名分數 對 基準｜夠嗎", "最深虧損中位 對 基準｜夠嗎", "隨機錯位｜≤5%？",
                               "多賺的百分點 對 基準", "在場時間 對 基準"],
                   "rows": _rule_detail(R)})
    tables.append({"caption": ("3. 白話問答（檢驗期 2019 年起，比特幣日線，成本 0.10%）。照著進場＝基準是現金、但型態＋站上 20 日均線"
                               "所以先買；結束＝基準也轉持有（接手）或型態規則先出場。照著出場＝基準持有、但出現出場型態所以先賣；"
                               "結束＝重新買回或基準也出場。整段＝規則的錢 ÷ 基準的錢 −1。"),
                   "columns": ["問題", "次數", "結果", "平均", "整體效果"],
                   "rows": _plain_rows(R)})
    for which in ("entry", "exit"):
        tb = _p4_table(R, which)
        if tb:
            tables.append(tb)
    n_t12, n_all = R["t12"]
    notes = [
        "這份研究在做什麼：pressure_before_moves 發現「賣方、買方合計兩邊都升」在大漲前（12／23）和大跌前（14／27）都最常見，"
        "單看分不出方向。老闆問：把型態和 55 日均線或以前測過的順勢訊號放在一起呢？型態說「有事要發生」，趨勢也許說「往哪邊」。",
        "名詞：賣方合計＝送進交易所的幣、老幣移動、主動賣出、賣出壓價力這幾項排名的平均；買方合計＝提走、新增穩定幣、ETF 流入、"
        "主動買進、推價力、Coinbase 溢價的排名平均。排名 0–100＝比自己過去一年多少 % 的日子高。型態＝近 30 天平均排名比再前 60 天"
        "是升還是降，兩邊各看一次。",
        f"50 日均線和 55 日均線：2012 年起 {n_all:,} 天裡有 {n_t12:,} 天（{n_t12 / max(n_all, 1) * 100:.1f}%）兩者說法不一樣；"
        f"51 次事件當天有 {R['t12_ev']} 次不一樣。所以兩條幾乎可以互換，表裡兩條都列。",
        "事件的型態用「事件前一天收盤時就知道的型態」（近 30 天 比 再前 60 天），和 pressure_before_moves 的算法完全一樣。",
        "重要：大漲起點是低點、大跌起點是高點，事件當天收盤在均線下或上，幾乎是定義造成的；事件前 10 天、30 天也多半如此"
        "（低點前通常在跌、高點前通常在漲）。所以第 1 部分「趨勢很能分開大漲和大跌」不能當成趨勢能預告方向的證據，"
        "要看第 2、3 部分的所有日子和照著做的結果。",
        "第 2 部分的天數很多，但相鄰日子的之後 60 天幾乎一樣，獨立的次數其實只有幾十段；區塊重抽和隨機錯位就是為了這個。",
        f"發現期／檢驗期在跑之前就定好：發現期＝之後 60 天在 2018-12-31 以前結束的日子；檢驗期＝2019-01-01 起的日子。"
        f"格子只用發現期挑，挑的規則（平均最高或最低、至少 {MIN_CELL} 天）也是先定好的。",
        "公平考試和 main_signal_check 一樣四項一起考，但只用 2019 年以後的起算日；其他幣和 MSTR 用自己的 50 日均線或 55／20，"
        "型態用比特幣的（當成整個市場的狀態）。比特幣以外的資產時間對齊照 supply_demand 的做法（MSTR 晚一天）。",
        "隨機錯位：只挪型態，不重挑格子；一般人模擬在錯位時減為每個時鐘 20 次，真的規則也用 20 次重算來比，"
        "200 次錯位的最小解析度是 0.5%，所以 Holm 的第一關（5%÷6≈0.8%）只有完全沒有錯位做得一樣好才會過。",
        "組成會變：2017-08 以前賣方合計只有送進交易所和老幣、買方合計只有提走；送進和提走都跟活躍度一起動，"
        "所以早年「兩邊都升／都降」有一部分只是市場變熱鬧。發現期大部分在這段時間，挑出來的格子可能帶著這個特性。",
        "交易所地址是用今天的標籤往回套，越早年漏得越多；幣安零手續費期間（2022-07-08～2023-04-04）主動買賣、壓價力、推價力拿掉。",
        "事件只有 23 次大漲、28 次大跌（最後一次大跌可能還在進行），很多相隔不到 90 天；檢驗期 2019 年起只有大約兩個完整的多空循環。",
        "第 4 部分是探索：同一批發現期資料挑出最好的條件，會偏樂觀；檢驗期同方向也只是「值得事先登記再測」，不是證據。",
        "所有門檻（30／60 天窗口、150 天、60 天區塊、200 次錯位、2,000 次打亂、5 個百分點）都在跑之前定好，沒有試別的數字。",
        "後來加的（看完結果才加）：判讀裡「24 格多重比較、分時期、一個推測」那一段、「兩個格子都同方向的條件」那一段，"
        "以及照著出場的次數拆成「提早出場／被擋住的進場」；定義、門檻和考試都沒有改。",
        f"這次計算用了 {time.time() - t0:.0f} 秒（第 1、2 部分 {R['t_p12']:.0f} 秒）。",
    ] + [f"資料狀況：{x}" for x in data["problems"] + R["problems_exam"]]
    sources = [
        {"name": "Coin Metrics 社群 API：比特幣 FlowInExNtv、FlowOutExNtv（送進／提出交易所的幣）",
         "url": fb.CM_URL + "?assets=btc&metrics=" + ",".join(fb.CM_METRICS) + "&frequency=1d"},
        {"name": "Open Bitcoin Metrics：放 155 天以上的幣被花掉（只下載 CSV 資料檔）", "url": ocs.OBM_BASE.format(pb.OBM_ID)},
        {"name": "幣安現貨 BTCUSDT 1 小時線（主動買賣量、壓價力，也用來拼 24 個收盤時間）",
         "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"},
        {"name": "DefiLlama 穩定幣總供給 API", "url": sd.LLAMA},
        {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": fb.FARSIDE_URL},
        {"name": "Coinbase Exchange 公開日線 BTC-USD", "url": sd.CB_URL + "?granularity=86400"},
        {"name": "Bitstamp 公開 API：BTC/USD 日線", "url": sd.BS_URL + "?step=86400&limit=1000"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線（ETH、SOL、BNB、XRP 對 USDT）", "url": "https://api.binance.com/api/v3/klines?interval=1d"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        {"name": "幣安公告：零手續費比特幣交易開始", "url": pi.URL_LAUNCH},
        {"name": "幣安公告：零手續費比特幣交易結束", "url": pi.URL_END},
    ]
    return {"id": "pressure_with_trend",
            "title": "買賣壓型態＋趨勢狀態：分得出方向嗎？",
            "question": ("pressure_before_moves 看到「賣方、買方兩邊都升」在大漲前和大跌前都最常見。把這些型態和 50／55 日均線、"
                         "55／20、200 日均線等順勢狀態放在一起，能不能分出接下來往哪邊？照著進場會不會賺、照著出場會不會少賠？"),
            "asof": _d(close.index[-1]),
            "verdict": verdict,
            "now": _now(R), "tables": tables, "notes": notes, "sources": sources}


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
