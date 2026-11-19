"""Historical study "whipsaw_anatomy": at the moment the main signal ma50 (hold bitcoin while the daily close is above
its 50-day average) turns off, is there anything KNOWN AT THAT MOMENT that separates a whipsaw (ma50 soon turns back
on at a higher price) from a real decline?

Why (written before any run). Every filter added on top of ma50 can only avoid losses; to earn MORE than ma50 a rule
must hold on days ma50 does not. reluctant_sellers (rule R2, report table) counted ma50's exits since 2012: 99 were
whipsaws (back on within 20 days at a higher close) and 39 real declines; sitting through whipsaws gained +2.6% each on
average, sitting through real declines cost -11.0% each (2018: -17% to -30% several times). So the prize is clear and
so is the danger. What was already tried (not repeated here as "new"): trend_anatomy (20/10 breakout, trade by trade)
found the whole edge comes from avoiding big falls, that in mid-size back-and-forth stretches 21 of 45 trades were
quick losers (37 such in-and-outs over the whole period, compounding to about -92%) and that 48 of 72 exits were bought
back higher; rule_upgrades tried faster re-entry after an exit (A1-A3), half-and-half blends, a 200-day slope filter,
volatility sizing and stops on 55/20 - none passed; robust_trend showed the trend edge survives clocks and sloppy
execution (it did not try to tell whipsaws apart); mtf_gate timed exits/entries with 4-hour bars inside the daily
gate - more trades, failed; reluctant_sellers R2 sat through exits when exchange inflows were low and holders in profit
- failed (placebo 76%). Here the question is only about the STATE OF PRICE AT THE EXIT CLOSE.

Data (only what the project already loads; public key-less endpoints, fixed user agent "btc-research-dashboard"):
  BTC daily closes: te._daily_close (00:00 UTC; blockchain.info before Binance 2017-08), from the series' start.
  BTC daily volume: Binance spot BTCUSDT 1-day klines, quote volume (USDT), from 2017-08-17, kept in this study's own
    store cache/_store/whipsaw_anatomy_1d_BTCUSDT.csv (topped up, closed candles only). No volume before that.
  Fair exam inputs as main_signal_check / price_impact: Binance 1-hour BTCUSDT grid (clock_robustness), ETH SOL BNB XRP
    Binance daily closes and their own 1-day quote volumes (own stores whipsaw_anatomy_1d_<SYMBOL>.csv), MSTR Yahoo
    adjusted closes (no volume used for MSTR).

1. Exits and labels. ma50 = main_signal_check's (close above the mean of the last 50 closes). Exit at close e = ma50
   on at e-1 and off at e, every exit from the start of the series. Labels (known only later):
   rec = first close after e where ma50 is on AND the close is above the exit close; stop = first close in e+1..e+60
   at or below 0.85 x exit close.
   WHIPSAW = the first close after e where ma50 is on comes within 20 closes (u - e <= 20), is above the exit close
     (reluctant_sellers' definition), and no stop close came before it.
   REAL = a stop close exists (within 60 closes) and comes before rec (or rec never comes).
   OTHER = neither, once known (rec found, or 60 closes have passed); UNKNOWN otherwise (too recent).
   Sitting through (cost/benefit of each type): hold from e until the first close where ma50 is on again or the close is
   at or below 0.85 x exit close, whichever first (no time limit); gain versus ma50 = close(end)/close(e) - 1, plus
   2 x 0.10% when it ends because ma50 is back on (ma50 sold at e and buys back), else +0 (both sold once). Open at the
   end of the data -> measured to the last close. Also shown (report): the 20-day version used by H1 (below).
   Counts by period (discovery / test / full and by calendar year) and the gain of each type.
2. Descriptors measured AT the exit close e (only closes up to e), fixed by reasoning, each oriented so that TRUE
   should mean "more likely a whipsaw":
   (a) close above its 200-day average; (b) the 50-day average above its own value 10 closes earlier; (c) shallow
   break: close above 0.98 x the 50-day average; (d) trend age: ma50 was on for at least the 30 closes e-30..e-1;
   (e) run-up: close at least 30% above the close 90 closes earlier; (f) calm: 20-day realised volatility (std of
   daily log returns over the 20 returns ending at e) below the median of that volatility over the 365 previous
   closes (>= 330 values); (g) not a breakdown: close above the lowest close of e-20..e-1; (h) close at least 0.85 x
   the highest close of e-180..e-1; (i) volume: the exit candle's quote volume below the mean of the 20 previous
   candles' (only where volume exists; else undefined); (j) weekly: the last completed weekly close (weeks end on
   Sunday; on day t the last week whose Sunday is on or before t) above the mean of the last 10 weekly closes.
   Missing history -> undefined (left out of the descriptor's counts; counts as FALSE inside a rule).
   On the exam's other series the same definitions use that series' own closes (rows = trading days for MSTR); for
   the 24 clocks the weekly close is the clock's Sunday close and (i) uses BTC's UTC daily candle of the previous
   date (the last one closed at that clock's close); other coins use their own Binance volume; MSTR (i) undefined.
3. Honest split fixed in advance: DISCOVERY = exits before 2019-01-01, TEST = exits from 2019-01-01 (BTC daily series).
   On DISCOVERY only (UNKNOWN never occurs there): for each descriptor, when TRUE and when FALSE, counts and shares of
   WHIPSAW / REAL / OTHER, and the average sitting-through gain (definition of point 1). The CODE picks the single
   descriptor with the highest average gain when TRUE among those with at least 15 TRUE cases, and separately the best
   PAIR (logical AND of two of the ten; undefined if either is) with at least 10 TRUE cases. Ties: fewer REAL cases.
4. TEST. H1 = hold when ma50 is on, OR (the picked descriptor was TRUE at the most recent ma50 exit e AND t - e <= 19
   (decisions at closes e .. e+19, so at most 20 extra days held) AND every close from e to t is above 0.85 x the exit
   close; once a close is at or below it, cash until ma50 is on again). H2 = the same with the picked pair.
   Reported on TEST exits (BTC daily): whipsaws and real declines sat through, their 20-day stint gains versus ma50
   (stint ends at the first close where H goes to cash or ma50 is on; + 2 x 0.10% when ma50 is on at the end, else +0),
   every real decline sat through listed.
   Fair exam = main_signal_check's four checks wired like price_impact._Exam with first month 2019-01: BTC daily
   monthly starts 2019-01 .. 2023-12 textbook 0.10%; clock starts 2019-01 .. 2023-12 x 24 closing hours textbook; the
   same cells as an ordinary person (act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock,
   main_signal_check's seeds and its ma50 draws); ETH SOL BNB XRP MSTR from max(first valid month, 2019-01) to three
   years before today, each with its OWN ma50, exits and descriptors. ma50 re-scored on exactly the same starts
   (numbers of starts reported). Ranking score = worst of the four "beats holding" shares (main_signal_check's
   eligibility). Median deepest loss = median over the clock textbook cells.
   Placebo: in every series (BTC daily, each clock, each other asset) the rule's descriptor value at each exit is
   replaced by its value at another exit of the same series - a random derangement of that series' exits (no exit
   keeps its own value), 200 draws (fixed seeds; the same permutation for H1 and H2 within a draw); the 20-day window
   and the -15% stop stay. The whole exam is rebuilt with 20 ordinary-person simulations per clock (same seeds) and the
   real rule is re-scored the same way; placebo share = share of draws whose ranking score is at least the real one's.
   Pre-registered pass for H1 and for H2, all three: ranking score >= ma50's on those starts; median deepest loss no
   more than 5 points worse than ma50's; placebo share <= 10%. Two rules tried: one passing alone is weaker evidence.
5. Report only: the full-history descriptor table (all known exits); the same table on TEST exits and whether
   discovery's pick (single and pair) is also the best in the test period (at least 5 TRUE test cases; stability of
   the ranking); calendar years where H1 / H2 gain or lose versus ma50 (BTC daily, 0.10%, positions from the whole
   history; years before 2019 are discovery years); the null rule N0 = "always sit through for 20 days with the -15%
   stop" (no descriptor) on the same exam, and for H1, H2, N0 the share of exam cells whose return beats ma50's (daily
   starts, clock cells, ordinary-person simulations, other-asset starts) with the median difference.
Look-ahead check: closes (and volume) cut at 2019-06-30, 2021-06-30, 2024-06-30; descriptors, H1 and H2 positions up to
   the cut must not change.
Honesty: every threshold (20 / 60 closes, -15%, 2%, 30 closes, 30% / 90 closes, 365-close median, 20 / 180 closes, 10
   weeks, 15 / 10 minimum cases, 2019-01-01 split) was fixed by reasoning before the run; no alternatives were tried.
   Ten descriptors plus 45 pairs were searched on discovery, so the pick is optimistic there by construction; only the
   test period counts. A pass would only make a rule a candidate for the daily shadow log. Not advice.
"""
from __future__ import annotations

import json
import math
import time
from itertools import combinations

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
DAY = pd.Timedelta(days=1)
SPLIT = pd.Timestamp("2019-01-01")
WHIP_DAYS, REAL_DAYS, STOP = 20, 60, 0.85
HOLD_DAYS = 20  # decisions at closes e .. e+19
AGE, RUN_DAYS, RUN_UP, LOW_DAYS, HIGH_DAYS, NEAR_HIGH = 30, 90, 0.30, 20, 180, 0.85
SHALLOW, SLOPE_DAYS, VOL_DAYS, VOL_WIN, VOL_MIN, VAVG, WEEKS = 0.98, 10, 20, 365, 330, 20, 10
MIN_ONE, MIN_PAIR, MIN_TEST = 15, 10, 5
DRAWS, PL_SIMS = 200, 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.10
KI = {"ma50": 1, "h1": 110, "h2": 111, "n0": 112}
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")
VOL_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT"}
FIRST_MS = 1502928000000  # 2017-08-17 00:00 UTC
MAX_PAGES = 12
KEYS = tuple("abcdefghij")
DESC = {"a": "長期趨勢還在：收盤在 200 日均線之上",
        "b": "50 日均線本身還在往上（比 10 天前高）",
        "c": "只跌破一點：收盤離 50 日均線不到 2%",
        "d": "趨勢走了一段：之前已連續持有至少 30 天",
        "e": "之前漲很多：收盤比 90 天前高 30% 以上",
        "f": "波動平靜：20 天波動低於過去一年中位數",
        "g": "不是破底：收盤高於前 20 天最低收盤",
        "h": "離高點不遠：收盤在前 180 天最高收盤的 85% 以上",
        "i": "量沒放大：當天成交金額低於前 20 天平均（2017-08 起才有）",
        "j": "週線還在：上一根收完的週線高於 10 週均線"}
WHY = {"a": "大趨勢還向上，跌破 50 日均線比較像回檔",
       "b": "中期均線還沒轉彎，價格只是暫時掉下來",
       "c": "跌得淺，比較容易站回去",
       "d": "走了一段的趨勢，比剛開始的短波段穩",
       "e": "漲多了以後的回檔，常常只是休息",
       "f": "沒有恐慌，跌破時很平靜",
       "g": "還沒跌破最近的低點，不是崩跌",
       "h": "還在高檔附近，不是已經大跌一段",
       "i": "沒有大家搶著賣的爆量",
       "j": "更長的週線還是多頭"}
KIND = {"whip": "假跌破", "real": "真的下跌", "other": "其他", "unknown": "還不知道"}
NAME = {"ma50": "50 日均線（基準）",
        "h1": "H1：50 日均線，或「下車那天挑到的條件成立」就多抱 20 天（跌 15% 停損）",
        "h2": "H2：50 日均線，或「下車那天挑到的兩個條件都成立」就多抱 20 天（跌 15% 停損）",
        "n0": "N0（對照）：50 日均線，每次下車都多抱 20 天（跌 15% 停損）"}
SHORT = {"ma50": "50 日均線", "h1": "H1", "h2": "H2", "n0": "N0"}

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- data ----------

def _store(symbol: str):
    return common.STORE / f"whipsaw_anatomy_1d_{symbol}.csv"


def _read(symbol: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    try:
        for line in _store(symbol).read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            v = float(parts[1])
            if math.isfinite(v):
                rows[int(parts[0])] = v
    except Exception:  # noqa: BLE001 - no store yet: fetch from the start
        return {}
    return rows


def _write(symbol: str, rows: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{ms},{v!r}" for ms, v in sorted(rows.items()))
        _store(symbol).write_text("ms,quote\n" + body + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run fetches again
        pass


def _volume(symbol: str) -> tuple[pd.Series, str | None]:
    """Quote volume of closed Binance spot 1-day candles, indexed by candle date (UTC)."""
    rows = _read(symbol)
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - 86_400_000:
        start = (max(rows) + 1) if rows else FIRST_MS
        added = 0
        try:
            for _ in range(MAX_PAGES):
                page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000,
                                                      "startTime": start})
                if not page:
                    break
                now_ms = int(time.time() * 1000)
                for r in page:
                    if int(r[6]) < now_ms:  # closed candles only
                        rows[int(r[0])] = float(r[7])
                        added += 1
                if len(page) < 1000:
                    break
                start = int(page[-1][0]) + 1
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = (f"幣安 {symbol} 日成交金額這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
        if added:
            _write(symbol, rows)
    if not rows:
        return pd.Series(dtype=float), problem
    s = pd.Series(rows).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms").normalize()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index < pd.Timestamp(common.today_utc())]
    return s, problem


def _vol_flag(v: pd.Series) -> pd.Series:
    """1 when the candle's quote volume is below the mean of the 20 previous candles (NaN = undefined)."""
    if v.empty:
        return pd.Series(dtype=float)
    v = v.asfreq("D")
    v = v.where(v > 0)
    avg = v.shift(1).rolling(VAVG, min_periods=VAVG).mean()
    return (v < avg).astype(float).where(v.notna() & avg.notna())


def _load() -> dict:
    problems = []
    close = te._daily_close()
    other, more = msc._load_other()
    problems += more
    vols = {}
    for key, sym in VOL_SYMBOLS.items():
        if key != "BTC" and key not in other:
            continue
        v, prob = _volume(sym)
        if prob:
            problems.append(prob)
        vols[key] = v
    return {"close": close, "other": other, "vols": vols, "problems": problems}


# ---------- descriptors, exits, labels ----------

def _f(cond: np.ndarray, ok: np.ndarray) -> np.ndarray:
    return np.where(ok, np.asarray(cond, dtype=float), np.nan)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    out[k:] = a[:len(a) - k]
    return out


def _desc(close: pd.Series, ma: np.ndarray, vflag: np.ndarray | None) -> dict[str, np.ndarray]:
    """Ten 0/1 descriptors at every close (NaN = undefined); only closes up to that close are used."""
    c = close.values.astype(float)
    s = pd.Series(c)
    n = len(c)
    m200 = s.rolling(200, min_periods=200).mean().values
    m50 = s.rolling(50, min_periods=50).mean().values
    out = {"a": _f(c > m200, np.isfinite(m200))}
    m50p = _shift(m50, SLOPE_DAYS)
    out["b"] = _f(m50 > m50p, np.isfinite(m50) & np.isfinite(m50p))
    out["c"] = _f(c > SHALLOW * m50, np.isfinite(m50))
    mav = np.asarray(ma, dtype=float)
    on = np.nan_to_num(mav) > 0.5
    run = np.zeros(n)
    for t in range(n):
        run[t] = run[t - 1] + 1 if (t and on[t]) else (1.0 if on[t] else 0.0)
    prev_run = _shift(run, 1)
    prev_ok = np.isfinite(_shift(mav, 1))
    out["d"] = _f(prev_run >= AGE, prev_ok & np.isfinite(prev_run))
    c90 = _shift(c, RUN_DAYS)
    out["e"] = _f(c / np.where(np.isfinite(c90), c90, 1.0) - 1 >= RUN_UP, np.isfinite(c90))
    lr = np.log(s).diff()
    v20 = lr.rolling(VOL_DAYS, min_periods=VOL_DAYS).std()
    med = v20.shift(1).rolling(VOL_WIN, min_periods=VOL_MIN).median()
    out["f"] = _f((v20 < med).values, (v20.notna() & med.notna()).values)
    low = s.shift(1).rolling(LOW_DAYS, min_periods=LOW_DAYS).min().values
    out["g"] = _f(c > low, np.isfinite(low))
    high = s.shift(1).rolling(HIGH_DAYS, min_periods=HIGH_DAYS).max().values
    out["h"] = _f(c >= NEAR_HIGH * high, np.isfinite(high))
    out["i"] = np.full(n, np.nan) if vflag is None else np.asarray(vflag, dtype=float)
    idx = pd.DatetimeIndex(close.index)
    weekly = close.resample("W-SUN").last()
    wma = weekly.rolling(WEEKS, min_periods=WEEKS).mean()
    wflag = (weekly > wma).astype(float).where(weekly.notna() & wma.notna())
    label = idx - pd.to_timedelta((idx.weekday + 1) % 7, unit="D")
    out["j"] = wflag.reindex(label).values.astype(float)
    return out


def _exits(ma: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(np.asarray(ma, dtype=float)) > 0.5
    return np.where(m[:-1] & ~m[1:])[0] + 1


def _label(c: np.ndarray, ma: np.ndarray, e: int) -> str:
    n = len(c)
    on = np.nan_to_num(ma) > 0.5
    stop_lvl = STOP * c[e]
    u = next((k for k in range(e + 1, n) if on[k]), None)
    s = next((k for k in range(e + 1, min(n, e + REAL_DAYS + 1)) if c[k] <= stop_lvl), None)
    rec = next((k for k in range(e + 1, n) if on[k] and c[k] > c[e]), None)
    if u is not None and u - e <= WHIP_DAYS and c[u] > c[e] and (s is None or s > u):
        return "whip"
    if s is not None and (rec is None or s < rec):
        return "real"
    if rec is not None or n - 1 - e >= REAL_DAYS:
        return "other"
    return "unknown"


def _sit(c: np.ndarray, ma: np.ndarray, e: int, cap: int | None) -> tuple[float, int, bool]:
    """Gain versus ma50 of sitting through exit e (until ma50 on, the -15% stop, or cap closes); (gain, days, done)."""
    n = len(c)
    on = np.nan_to_num(ma) > 0.5
    last = n - 1 if cap is None else min(n - 1, e + cap)
    for k in range(e + 1, last + 1):
        if on[k]:
            return c[k] / c[e] - 1 + 2 * COST, k - e, True
        if c[k] <= STOP * c[e]:
            return c[k] / c[e] - 1, k - e, True
    if cap is not None and e + cap <= n - 1:
        return c[e + cap] / c[e] - 1, cap, True
    return c[-1] / c[e] - 1, n - 1 - e, False


def _hold(c: np.ndarray, ma: np.ndarray, ex: np.ndarray, vals: np.ndarray) -> np.ndarray:
    pos = np.nan_to_num(np.asarray(ma, dtype=float))
    on = pos > 0.5
    pos = pos.copy()
    n = len(c)
    for e, v in zip(ex, vals):
        if not v:
            continue
        lvl = STOP * c[e]
        for t in range(e, min(n, e + HOLD_DAYS)):
            if t > e and on[t]:
                break
            if c[t] <= lvl:
                break
            pos[t] = 1.0
    return pos


def _vals(D: dict, spec) -> np.ndarray:
    """Rule value at each exit: spec None = never (ma50), 'all' = always (N0), tuple of keys = AND (NaN -> False)."""
    n = len(next(iter(D.values())))
    if spec is None:
        return np.zeros(n, dtype=bool)
    if spec == "all":
        return np.ones(n, dtype=bool)
    out = np.ones(n, dtype=bool)
    for k in spec:
        out &= np.nan_to_num(D[k], nan=0.0) > 0.5
    return out


def _derange(n: int, rng) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


# ---------- the exam ----------

class _Exam(pi._Exam):
    """price_impact's exam (main_signal_check's four checks on starts from a chosen month) for 'ma50 OR sit through
    the exit for 20 days when a descriptor held at the exit'; every series has its own exits and descriptors."""

    def __init__(self, close, other, price, b0, pre, first, vflags: dict):
        super().__init__(close, other, price, b0, pre, first)
        self.series = {}
        vb = vflags.get("BTC", pd.Series(dtype=float))
        self.series["btc"] = self._prep(close, self.ma, vb.reindex(close.index).values)
        npre = len(pre)
        for h in range(24):
            cf = np.concatenate([pre, price[h::24]]).astype(float)
            idx = self.m.dates[0] + pd.to_timedelta(np.arange(-npre, len(cf) - npre), unit="D")
            cs = pd.Series(cf, index=idx)
            maf = msc._targets(pd.Series(cf))["ma50"].values.astype(float)
            D = _desc(cs, maf, vb.reindex(idx - DAY).values)
            c = cf[npre:][:self.days]
            ma = self.cma[h]
            ex = _exits(ma)
            self.series[("clock", h)] = {"c": c, "ma": ma, "ex": ex, "D": {k: D[k][npre + ex] for k in KEYS}}
        for key, o in self.other.items():
            vf = vflags.get(key)
            self.series[("o", key)] = self._prep(o["close"], o["ma"],
                                                 None if vf is None or vf.empty else vf.reindex(o["close"].index).values)

    @staticmethod
    def _prep(close: pd.Series, ma: np.ndarray, vflag) -> dict:
        D = _desc(close, ma, vflag)
        ex = _exits(ma)
        return {"c": close.values.astype(float), "ma": np.nan_to_num(np.asarray(ma, dtype=float)), "ex": ex,
                "D": {k: D[k][ex] for k in KEYS}, "full": D}

    def pos(self, key, spec, perm: dict | None = None) -> np.ndarray:
        s = self.series[key]
        v = _vals(s["D"], spec)
        if perm is not None:
            v = v[perm[key]]
        return _hold(s["c"], s["ma"], s["ex"], v)

    def perms(self, rng) -> dict:
        return {k: _derange(len(s["ex"]), rng) for k, s in self.series.items()}

    def summary(self, spec, ki: int, sims: int = SIMS, perm: dict | None = None) -> dict:
        pos = self.pos("btc", spec, perm)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = self.pos(("clock", h), spec, perm)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :] = r
        each, ors = {}, []
        for key, o in self.other.items():
            w = self.pos(("o", key), spec, perm)
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
            ors.append(r[:, 0])
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
        x["arr"] = {"d": dr, "c": tr, "p": pr, "o": np.concatenate(ors) if ors else np.array([])}
        return x


def _vs_ma(x: dict, base: dict) -> list:
    """Share of cells whose yearly return beats ma50's, and the median difference, per check."""
    out = []
    for k in ("d", "c", "p", "o"):
        a, b = x["arr"][k], base["arr"][k]
        if k == "p":
            b = np.broadcast_to(base["arr"]["p"].mean(axis=2, keepdims=True), a.shape) if a.shape != b.shape else b
        if not a.size:
            out.append((None, None))
            continue
        out.append((float((a > b + 1e-9).mean()), float(np.median(a - b))))
    return out


# ---------- descriptive pieces ----------

def _exit_rows(close: pd.Series, ma: np.ndarray, D: dict) -> list[dict]:
    c = close.values.astype(float)
    out = []
    for e in _exits(ma):
        g, days, done = _sit(c, ma, e, None)
        g20, d20, done20 = _sit(c, ma, e, HOLD_DAYS)
        out.append({"i": int(e), "day": close.index[e], "kind": _label(c, ma, e), "gain": g, "days": days,
                    "done": done, "g20": g20, "d20": d20, "done20": done20,
                    "D": {k: D[k][e] for k in KEYS}})
    return out


def _cell(rows: list[dict], spec) -> dict:
    """Counts and gains among known exits for a descriptor spec (tuple of keys); TRUE / FALSE / undefined."""
    known = [r for r in rows if r["kind"] != "unknown"]
    t, f = [], []
    for r in known:
        v = [r["D"][k] for k in spec]
        if any(not _ok(x) for x in v):
            continue
        (t if all(x > 0.5 for x in v) else f).append(r)

    def part(g: list) -> dict:
        n = len(g)
        cnt = {k: sum(1 for r in g if r["kind"] == k) for k in ("whip", "real", "other")}
        return {"n": n, **cnt, "sw": cnt["whip"] / n if n else None, "sr": cnt["real"] / n if n else None,
                "gain": float(np.mean([r["gain"] for r in g])) if n else None,
                "g20": float(np.mean([r["g20"] for r in g])) if n else None}
    return {"t": part(t), "f": part(f)}


def _pick(rows: list[dict], specs: list, min_n: int) -> tuple[list, dict]:
    cells = {s: _cell(rows, s) for s in specs}
    ok = [s for s in specs if cells[s]["t"]["n"] >= min_n]
    ranked = sorted(ok, key=lambda s: (-cells[s]["t"]["gain"], cells[s]["t"]["real"]))
    return ranked, cells


def _cal_years(close: pd.Series, pos: np.ndarray) -> dict:
    change = close.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    s = pd.Series(net, index=close.index)
    return {int(y): float(np.prod(1 + g.values) - 1) for y, g in s.groupby(s.index.year)}


def _stints(close: pd.Series, ma: np.ndarray, pos: np.ndarray, rows: list[dict]) -> list[dict]:
    """For every exit a rule sat through: its stint return versus ma50 (20-day rule stint)."""
    c = close.values.astype(float)
    on = np.nan_to_num(ma) > 0.5
    n = len(c)
    out = []
    for r in rows:
        e = r["i"]
        if pos[e] < 0.5:
            continue
        k = next((t for t in range(e + 1, n) if pos[t] < 0.5 or on[t]), None)
        if k is None:
            out.append({**r, "adv": c[-1] / c[e] - 1, "sd": n - 1 - e, "sdone": False})
        else:
            out.append({**r, "adv": c[k] / c[e] - 1 + (2 * COST if on[k] else 0.0), "sd": k - e, "sdone": True})
    return out


def _lookahead(data: dict, pick: tuple, pair: tuple, full_pos: dict) -> tuple[int, int]:
    total = bad = 0
    close = data["close"]
    vb = data["vols"].get("BTC", pd.Series(dtype=float))
    full_D = _desc(close, msc._targets(close)["ma50"].values.astype(float), _vol_flag(vb).reindex(close.index).values)
    for cut in CUTS:
        cc = pd.Timestamp(cut)
        cl = close[close.index <= cc]
        v2 = vb[vb.index <= cc]
        ma2 = msc._targets(cl)["ma50"].values.astype(float)
        D2 = _desc(cl, ma2, _vol_flag(v2).reindex(cl.index).values)
        n = len(cl)
        for k in KEYS:
            total += 1
            a, b = D2[k], full_D[k][:n]
            bad += int(not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(np.nan_to_num(a), np.nan_to_num(b)))
        ex = _exits(ma2)
        Dx = {k: D2[k][ex] for k in KEYS}
        for name, spec in (("h1", pick), ("h2", pair)):
            p2 = _hold(cl.values.astype(float), ma2, ex, _vals(Dx, spec))
            total += 1
            bad += int(not np.array_equal(p2, full_pos[name][:n]))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    vflags = {k: _vol_flag(v) for k, v in data["vols"].items()}
    rows_h, problem = cr._hourly("BTCUSDT")
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(close, data["other"], price, b0, pre, SPLIT, vflags)
    btc = ex.series["btc"]
    ma = ex.ma
    rows = _exit_rows(close, ma, btc["full"])
    disc = [r for r in rows if r["day"] < SPLIT]
    test = [r for r in rows if r["day"] >= SPLIT]
    singles = [(k,) for k in KEYS]
    pairs = list(combinations(KEYS, 2))
    r1, c1 = _pick(disc, singles, MIN_ONE)
    r2, c2 = _pick(disc, pairs, MIN_PAIR)
    if not r1 or not r2:
        raise RuntimeError("發現期沒有夠多的例子可以挑")
    pick, pair = r1[0], r2[0]
    specs = {"ma50": None, "h1": pick, "h2": pair, "n0": "all"}
    s = {k: ex.summary(sp, KI[k], SIMS) for k, sp in specs.items()}
    t_exam = time.time() - t0

    small = {k: ex.summary(specs[k], KI[k], PL_SIMS) for k in ("h1", "h2")}
    hits = {k: 0 for k in small}
    lows = {k: [] for k in small}
    for d in range(DRAWS):
        perm = ex.perms(np.random.default_rng(SEED + 5000 + d))
        for k in small:
            x = ex.summary(specs[k], KI[k], PL_SIMS, perm)
            lows[k].append(x["low"])
            hits[k] += int(sd._score_ge(x, small[k]))
    placebo = {k: hits[k] / DRAWS for k in small}

    pos = {k: ex.pos("btc", sp) for k, sp in specs.items()}
    cal = {k: _cal_years(close, pos[k]) for k in ("ma50", "h1", "h2", "n0")}
    stints = {k: _stints(close, ma, pos[k], test) for k in ("h1", "h2", "n0")}
    full_cells = {sp: _cell(rows, sp) for sp in singles}
    test_r1, test_c1 = _pick(test, singles, MIN_TEST)
    test_r2, test_c2 = _pick(test, pairs, MIN_TEST)
    vs = {k: _vs_ma(s[k], s["ma50"]) for k in ("h1", "h2", "n0")}
    look = _lookahead(data, pick, pair, pos)
    return {"ex": ex, "rows": rows, "disc": disc, "test": test, "r1": r1, "c1": c1, "r2": r2, "c2": c2,
            "pick": pick, "pair": pair, "specs": specs, "s": s, "small": small, "placebo": placebo,
            "placebo_med": {k: float(np.median(v)) for k, v in lows.items()}, "pos": pos, "cal": cal,
            "stints": stints, "full_cells": full_cells, "test_r1": test_r1, "test_c1": test_c1,
            "test_r2": test_r2, "test_c2": test_c2, "vs": vs, "look": look, "problem": problem,
            "secs": time.time() - t0, "secs_exam": t_exam}


# ---------- text helpers ----------

def _sname(spec) -> str:
    return "＋".join(f"（{k}）" for k in spec)


def _slong(spec) -> str:
    return "，而且".join(f"（{k}）{DESC[k]}" for k in spec)


def _count_rows(rows: list[dict]) -> list:
    out = []
    groups = (("發現期（2019 年以前）", [r for r in rows if r["day"] < SPLIT]),
              ("測試期（2019 年起）", [r for r in rows if r["day"] >= SPLIT]), ("全期", rows))
    for lab, g in groups:
        for k in ("whip", "real", "other", "unknown"):
            sub = [r for r in g if r["kind"] == k]
            if not sub and k == "unknown":
                continue
            gains = [r["gain"] for r in sub]
            g20 = [r["g20"] for r in sub]
            out.append([lab if k == "whip" else "", KIND[k], f"{len(sub)}",
                        _p(float(np.mean(gains))) if gains else "—",
                        _p(float(np.prod([1 + v for v in gains]) - 1)) if gains else "—",
                        f"{float(np.median([r['days'] for r in sub])):.0f} 天" if sub else "—",
                        _p(float(np.mean(g20))) if g20 else "—"])
        out.append([lab, "合計", f"{len(g)}", "", "", "", ""])
    return out


def _year_rows(rows: list[dict]) -> list:
    out = []
    for y in sorted({r["day"].year for r in rows}):
        g = [r for r in rows if r["day"].year == y]
        cnt = {k: sum(1 for r in g if r["kind"] == k) for k in KIND}
        out.append([str(y), f"{len(g)}", f"{cnt['whip']}", f"{cnt['real']}", f"{cnt['other']}",
                    f"{cnt['unknown']}" if cnt["unknown"] else "0"])
    return out


def _cell_row(name: str, c: dict) -> list:
    t, f = c["t"], c["f"]
    return [name, f"{t['n']}（{t['whip']}｜{t['real']}｜{t['other']}）", f"{_share(t['sw'])}｜{_share(t['sr'])}",
            _p(t["gain"]), _p(t["g20"]),
            f"{f['n']}（{f['whip']}｜{f['real']}｜{f['other']}）", f"{_share(f['sw'])}｜{_share(f['sr'])}", _p(f["gain"])]


CELL_COLS = ["條件", "成立的次數（假跌破｜真的下跌｜其他）", "成立時：假跌破｜真的下跌 比例", "成立時：坐過去平均",
             "成立時：只多抱 20 天平均", "不成立的次數（假｜真｜其他）", "不成立時：假跌破｜真的下跌 比例", "不成立時：坐過去平均"]


# ---------- today ----------

def _now(data: dict, r: dict) -> list[dict]:
    close = data["close"]
    c = close.values.astype(float)
    ma = r["ex"].ma
    word = {1: "持有", 0: "現金"}
    st = {k: int(r["pos"][k][-1] > 0.5) for k in ("ma50", "h1", "h2", "n0")}
    full = r["ex"].series["btc"]["full"]
    ex = _exits(ma)
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜H1", "value": f"{word[st['ma50']]}｜{word[st['h1']]}"},
           {"label": "H2｜N0（對照：每次都多抱 20 天）", "value": f"{word[st['h2']]}｜{word[st['n0']]}"},
           {"label": "H1 用的條件（發現期挑的）", "value": _slong(r["pick"])},
           {"label": "H2 用的兩個條件（發現期挑的）", "value": _slong(r["pair"])}]
    if len(ex):
        e = int(ex[-1])
        days = len(c) - 1 - e
        out.append({"label": "最近一次 50 日均線下車",
                    "value": f"{_d(close.index[e])}，收盤 {c[e]:,.0f} 美元（{days} 天前；停損線 {STOP * c[e]:,.0f} 美元）"})
        for k in KEYS:
            v = full[k][e]
            out.append({"label": f"　那天（{k}）{DESC[k]}", "value": ("是" if v > 0.5 else "否") if _ok(v) else "沒有資料"})
        win = days < HOLD_DAYS and not (np.nan_to_num(ma[e + 1:]) > 0.5).any()
        out.append({"label": "多抱 20 天的時間窗還開著嗎", "value": "是" if win else "否（已超過 20 天、或均線已重新上車）"})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；H1、H2 沒有通過考試前不會用"})
    return out


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, ex = data["close"], r["s"], r["ex"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    rows, disc, test = r["rows"], r["disc"], r["test"]
    pick, pair = r["pick"], r["pair"]
    base = s["ma50"]

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    # ----- tables -----
    vfirst = data["vols"].get("BTC", pd.Series(dtype=float))
    rows0 = [["比特幣日線第一天", _d(close.index[0])],
             ["第一次 50 日均線下車", _d(rows[0]["day"]) if rows else "—"],
             ["50 日均線下車總次數", f"{len(rows)}（發現期 {len(disc)}、測試期 {len(test)}）"],
             ["幣安比特幣日成交金額第一天（條件 i）", _d(vfirst.index[0]) if len(vfirst) else "沒有"],
             ["發現期／測試期分界（事先定好）", "2019-01-01 以前下車＝發現期；2019-01-01 起＝測試期"]]
    rows_def = [[f"（{k}）", DESC[k], WHY[k]] for k in KEYS]
    rows_cnt = _count_rows(rows)
    rows_year = _year_rows(rows)
    rows_disc = [_cell_row(f"（{k}）{DESC[k]}", r["c1"][(k,)]) for k in KEYS]
    rank1 = [[f"第 {i + 1} 名", _sname(sp), f"{r['c1'][sp]['t']['n']}", _p(r["c1"][sp]["t"]["gain"]),
              f"{r['c1'][sp]['t']['real']}"] for i, sp in enumerate(r["r1"][:5])]
    rank2 = [[f"第 {i + 1} 名", _sname(sp), f"{r['c2'][sp]['t']['n']}", _p(r["c2"][sp]["t"]["gain"]),
              f"{r['c2'][sp]['t']['real']}"] for i, sp in enumerate(r["r2"][:5])]
    base_gain = float(np.mean([x["gain"] for x in disc])) if disc else None
    base_whip = sum(1 for x in disc if x["kind"] == "whip")
    base_real = sum(1 for x in disc if x["kind"] == "real")

    # test-period sat-through
    rows_t = []
    for k in ("h1", "h2", "n0"):
        st = r["stints"][k]
        for kind in ("whip", "real", "other", "unknown"):
            allk = [x for x in test if x["kind"] == kind]
            satk = [x for x in st if x["kind"] == kind]
            if not allk:
                continue
            advs = [x["adv"] for x in satk]
            rows_t.append([SHORT[k] if kind == "whip" else "", KIND[kind], f"{len(allk)}", f"{len(satk)}",
                           _p(float(np.mean(advs))) if advs else "—",
                           _p(float(np.prod([1 + v for v in advs]) - 1)) if advs else "—"])
        advs = [x["adv"] for x in st]
        rows_t.append([SHORT[k], "合計", f"{len(test)}", f"{len(st)}", _p(float(np.mean(advs))) if advs else "—",
                       _p(float(np.prod([1 + v for v in advs]) - 1)) if advs else "—"])
    rows_tl = []
    for k in ("h1", "h2"):
        for x in sorted(r["stints"][k], key=lambda x: x["day"]):
            rows_tl.append([SHORT[k], _d(x["day"]), KIND[x["kind"]], f"{x['sd']} 天" + ("" if x["sdone"] else "（還沒結束）"),
                            _p(x["adv"])])
    rows_tl = rows_tl or [["—", "—", "—", "—", "—"]]

    # exam
    res = {}
    for k in ("h1", "h2"):
        x = s[k]
        res[k] = {"score": sd._score_ge(x, base), "loss": x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK,
                  "plc": r["placebo"][k] <= PLACEBO_BAR}
        res[k]["pass"] = all(res[k].values())
    rows_b = []
    for k in ("ma50", "h1", "h2", "n0"):
        x = s[k]
        ck = x["checks"]
        rows_b.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                       "—" if k in ("ma50", "n0") else f"{_share(r['placebo'][k])}（真的 {_share1(r['small'][k]['low'])}，"
                                                       f"錯位中位數 {_share1(r['placebo_med'][k])}）"])
    rows_pass = []
    for k in ("h1", "h2"):
        x = s[k]
        rows_pass += [[SHORT[k], "排名分數不輸 50 日均線（同樣的起算日）", f"{_share1(x['low'])} vs {_share1(base['low'])}",
                       yes(res[k]["score"])],
                      ["", "最深虧損中位數最多比均線深 5 點", f"{_pc(x['c_deep_med'])} vs {_pc(base['c_deep_med'])}",
                       yes(res[k]["loss"])],
                      ["", "隨機錯位比例 ≤ 10%", _share(r["placebo"][k]), yes(res[k]["plc"])],
                      ["", "三條都要過", "", yes(res[k]["pass"])]]
    rows_vs = []
    labs = ("日線起算日", "起算日×收盤時間", "一般人做法（模擬）", "其他資產起算日")
    for k in ("h1", "h2", "n0"):
        v = r["vs"][k]
        rows_vs.append([NAME[k]] + [f"{_share(a)}｜{b:+.1f} 點" if _ok(a) else "—" for a, b in v])

    # full history and stability
    rows_full = [_cell_row(f"（{k}）{DESC[k]}", r["full_cells"][(k,)]) for k in KEYS]
    rows_test = [_cell_row(f"（{k}）{DESC[k]}", r["test_c1"][(k,)]) for k in KEYS]
    trank = {sp: i + 1 for i, sp in enumerate(r["test_r1"])}
    trank2 = {sp: i + 1 for i, sp in enumerate(r["test_r2"])}
    best1 = r["test_r1"][0] if r["test_r1"] else None
    best2 = r["test_r2"][0] if r["test_r2"] else None
    rows_stab = [["單一條件", _sname(pick), f"第 {trank[pick]} 名（共 {len(r['test_r1'])} 個夠 {MIN_TEST} 次）"
                  if pick in trank else f"測試期成立不到 {MIN_TEST} 次，不能排", _sname(best1) if best1 else "—",
                  _p(r["test_c1"][pick]["t"]["gain"]), _p(r["test_c1"][best1]["t"]["gain"]) if best1 else "—"],
                 ["兩個條件", _sname(pair), f"第 {trank2[pair]} 名（共 {len(r['test_r2'])} 組夠 {MIN_TEST} 次）"
                  if pair in trank2 else f"測試期成立不到 {MIN_TEST} 次，不能排", _sname(best2) if best2 else "—",
                  _p(r["test_c2"][pair]["t"]["gain"]), _p(r["test_c2"][best2]["t"]["gain"]) if best2 else "—"]]

    # calendar years
    cal = r["cal"]
    rows_cal = []
    for y in sorted(cal["ma50"]):
        if y < 2011:
            continue
        rows_cal.append([str(y) + ("（測試期）" if y >= SPLIT.year else "（發現期）"), _p(cal["ma50"][y]),
                         _pt(cal["h1"][y] - cal["ma50"][y]), _pt(cal["h2"][y] - cal["ma50"][y]),
                         _pt(cal["n0"][y] - cal["ma50"][y])])
    test_years = [y for y in cal["ma50"] if y >= SPLIT.year]
    h1_up = [y for y in test_years if cal["h1"][y] - cal["ma50"][y] > 1e-4]
    h1_dn = [y for y in test_years if cal["h1"][y] - cal["ma50"][y] < -1e-4]

    tables = [
        {"caption": "資料和分界。50 日均線下車＝前一天收盤在 50 日均線上、今天收在下面。", "columns": ["項目", "數字"],
         "rows": rows0},
        {"caption": ("十個「下車當下就知道」的條件（跑之前憑道理定好），每一個都朝「成立＝比較可能是假跌破」的方向寫。"
                     "條件 i 只有 2017-08 起有成交量；條件 j 的週線以週日收盤為一週結束。"),
         "columns": ["代號", "條件（下車那天收盤時就知道）", "為什麼可能代表假跌破"], "rows": rows_def},
        {"caption": (f"每次 50 日均線下車，事後怎麼分類。假跌破＝跌破均線後很快又站回去、而且站回的價格更高：{WHIP_DAYS} 天內均線"
                     f"又上車、價格比下車時高，中間沒有先跌到下車價的 {STOP * 100:.0f}%。真的下跌＝{REAL_DAYS} 天內先跌到下車價的 "
                     f"{STOP * 100:.0f}% 以下（比均線用更高價格重新上車還早）。其他＝兩種都不是。「坐過去」＝下車那天不賣，"
                     f"抱到均線重新上車或跌 15% 停損（先到的那個），和均線比多賺多少（均線重新上車的話加回均線多付的一買一賣 "
                     f"{2 * COST * 100:.2f}%）。「只多抱 20 天」＝H1、H2 的做法，最多抱 20 天。"),
         "columns": ["期間", "種類", "次數", "坐過去：平均多賺", "坐過去：接起來（相乘）", "坐過去：天數中位數",
                     "只多抱 20 天：平均多賺"], "rows": rows_cnt},
        {"caption": "每年的 50 日均線下車次數和種類。", "columns": ["年", "下車次數", "假跌破", "真的下跌", "其他", "還不知道"],
         "rows": rows_year},
        {"caption": (f"發現期（2019 年以前，{len(disc)} 次下車，其中假跌破 {base_whip}、真的下跌 {base_real}；全部坐過去平均 "
                     f"{_p(base_gain)}）：每個條件成立和不成立時，假跌破和真的下跌各占幾成、坐過去平均多賺多少。"
                     "沒有資料的下車不算進那個條件。"),
         "columns": CELL_COLS, "rows": rows_disc},
        {"caption": (f"程式在發現期自動挑：「成立時坐過去平均多賺」最高的單一條件（至少 {MIN_ONE} 次成立）和兩個條件同時成立"
                     f"（至少 {MIN_PAIR} 次）；同分看真的下跌少的。左邊是單一條件前五名，挑到 {_sname(pick)}。"),
         "columns": ["名次", "條件", "成立次數", "成立時坐過去平均", "其中真的下跌"], "rows": rank1},
        {"caption": f"兩個條件同時成立的前五名（45 組裡），挑到 {_sname(pair)}。",
         "columns": ["名次", "兩個條件", "成立次數", "成立時坐過去平均", "其中真的下跌"], "rows": rank2},
        {"caption": (f"測試期（2019 年起，{len(test)} 次下車）：H1、H2 和對照 N0 各坐過去幾次、坐過去那段比均線多賺多少"
                     f"（從下車到規則賣掉或均線重新上車；均線重新上車時加 {2 * COST * 100:.2f}%）。"),
         "columns": ["規則", "種類", "下車次數", "規則坐過去的次數", "坐過去：平均多賺", "坐過去：接起來（相乘）"], "rows": rows_t},
        {"caption": "測試期：H1、H2 坐過去的每一次。", "columns": ["規則", "均線下車日", "種類", "多抱幾天", "比均線多賺／多賠"],
         "rows": rows_tl},
        {"caption": (f"測試期公平考試（main_signal_check 的做法，只用 2019 年起的起算日）。比特幣日線起算月 2019-01 到 "
                     f"{msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格"
                     f"（{base['c_starts']} 個起算月），一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 "
                     f"{SIMS} 次），其他資產 {'、'.join(ex.other)}（{base['o_n']} 個起算月，各用自己的均線、下車和條件）。"
                     "50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。隨機錯位：每條價格線上，"
                     f"把每次下車的條件結果換成另一次下車的（打亂、沒有一次保留自己的），{DRAWS} 次；20 天和 15% 停損不變；"
                     f"一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方法重算來比。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "隨機錯位比例"],
         "rows": rows_b},
        {"caption": "H1、H2 的通過條件（事先定好）。兩條規則都試了，只有一條過的話證據比較弱。",
         "columns": ["規則", "條件", "數字", "結果"], "rows": rows_pass},
        {"caption": ("只描述：和 50 日均線直接比，報酬比均線高的格子占幾成，以及每年報酬差距的中位數（百分點）。"
                     "N0 是「不看任何條件、每次都多抱 20 天」的對照，用來看條件到底有沒有幫上忙。"),
         "columns": ["規則"] + [f"{x}：贏均線｜差距中位數" for x in labs], "rows": rows_vs},
        {"caption": (f"只描述：全期（{len(rows)} 次下車，含發現期）每個條件的表現，看下車當下的狀態哪些有關係。"
                     "這張用到了發現期，不能當作測試結果。"), "columns": CELL_COLS, "rows": rows_full},
        {"caption": f"只描述：同樣的表只用測試期的下車（{len(test)} 次）。次數少，看方向就好。", "columns": CELL_COLS,
         "rows": rows_test},
        {"caption": (f"只描述：發現期挑到的，在測試期還是不是最好（測試期至少 {MIN_TEST} 次成立的才排名）。"),
         "columns": ["", "發現期挑到", "在測試期的名次", "測試期最好的", "挑到的在測試期：坐過去平均", "測試期最好的：坐過去平均"],
         "rows": rows_stab},
        {"caption": ("只描述：比特幣日線每年報酬（扣 0.10% 成本），H1、H2、N0 比 50 日均線多或少幾個百分點。"
                     "2019 年以前是發現期，條件就是在那段挑的，所以那幾年好看是應該的。"),
         "columns": ["年", "50 日均線", "H1 − 均線", "H2 − 均線", "N0 − 均線"], "rows": rows_cal},
    ]

    # ----- verdict -----
    wh = [x for x in rows if x["kind"] == "whip"]
    rl = [x for x in rows if x["kind"] == "real"]
    first = (f"{_d(rows[0]['day'])} 起比特幣 50 日均線下車 {len(rows)} 次：假跌破 {len(wh)} 次（坐過去平均 "
             f"{_p(float(np.mean([x['gain'] for x in wh])))}）、真的下跌 {len(rl)} 次（坐過去平均 "
             f"{_p(float(np.mean([x['gain'] for x in rl])))}）。")
    pc1, pc2 = r["c1"][pick]["t"], r["c2"][pair]["t"]
    second = (f"發現期（2019 年以前）程式挑到的單一條件是{_sname(pick)}{DESC[pick[0]]}（成立 {pc1['n']} 次，假跌破占 "
              f"{_share(pc1['sw'])}、真的下跌占 {_share(pc1['sr'])}，坐過去平均 {_p(pc1['gain'])}），兩個條件是{_sname(pair)}"
              f"（成立 {pc2['n']} 次，坐過去平均 {_p(pc2['gain'])}）。")

    def rule_line(k: str) -> str:
        x, q = s[k], res[k]
        fails = ([] if q["score"] else ["排名分數輸均線"]) + ([] if q["loss"] else ["最深虧損比均線深超過 5 點"]) \
            + ([] if q["plc"] else ["隨機錯位超過 10%"])
        st = r["stints"][k]
        sw = [y for y in st if y["kind"] == "whip"]
        sr = [y for y in st if y["kind"] == "real"]
        return (f"{SHORT[k]} 坐過去假跌破 {len(sw)} 次（平均 {_p(float(np.mean([y['adv'] for y in sw])) if sw else None)}）、"
                f"真的下跌 {len(sr)} 次（平均 {_p(float(np.mean([y['adv'] for y in sr])) if sr else None)}），"
                f"排名分數 {_share1(x['low'])}、最深虧損中位數 {_pc(x['c_deep_med'])}、隨機錯位 {_share(r['placebo'][k])}，"
                + ("三條都通過" if q["pass"] else f"沒有通過（{'、'.join(fails)}）"))

    passed = [k for k in ("h1", "h2") if res[k]["pass"]]
    third = (f"測試期（2019 年起 {len(test)} 次下車；考試只用 2019 年起的起算日，均線排名分數 {_share1(base['low'])}、"
             f"最深虧損中位數 {_pc(base['c_deep_med'])}）：{rule_line('h1')}；{rule_line('h2')}。"
             + ("兩條都通過，只能算是可以放進每日影子紀錄觀察的候選。" if len(passed) == 2 else
                f"只有 {SHORT[passed[0]]} 通過；試了兩條只過一條，證據比較弱，頂多放進每日影子紀錄觀察。" if passed else ""))
    n0 = s["n0"]
    same = [x["i"] for x in r["stints"]["h1"]] == [x["i"] for x in r["stints"]["h2"]]
    fourth = (("H1 和 H2 在比特幣日線測試期坐過去的是完全相同的下車，第二個條件沒有多分出什麼。" if same else "")
              + f"對照 N0（每次都多抱 20 天、不看條件）排名分數 {_share1(n0['low'])}、日線起算日報酬贏均線 "
              f"{_share(r['vs']['n0'][0][0])}。")
    known = [x for x in rows if x["kind"] != "unknown"]
    sws = [r["full_cells"][(k,)]["t"]["sw"] for k in KEYS if _ok(r["full_cells"][(k,)]["t"]["sw"])]
    fifth = (f"全期十個條件成立時，假跌破占 {_share(min(sws))} 到 {_share(max(sws))}（全部下車 "
             f"{_share(sum(1 for x in known if x['kind'] == 'whip') / len(known))}），沒有一個能把真的下跌挑掉；"
             f"發現期挑到的{_sname(pick)}在測試期排第 "
             + (f"{[i for i, sp in enumerate(r['test_r1'], 1) if sp == pick][0]}" if pick in r["test_r1"] else "—")
             + f" 名（共 {len(r['test_r1'])} 個），排名換一段時間就不穩。")
    verdict = first + second + third + fourth + fifth

    lk_total, lk_bad = r["look"]
    notes = [
        "為什麼有這份研究：50 日均線上面再加任何過濾，都只能少賠、不能多賺；想比均線賺更多，就得在均線空手的日子也持有。"
        "reluctant_sellers 數過：均線下車後如果硬抱著，假跌破平均多賺 +2.6%，真的下跌平均多賠 -11.0%（2018 年好幾次 -17% 到 -30%）。"
        "所以問題是：均線下車那一刻，有沒有什麼當下就知道的事，可以分出假跌破和真的下跌？",
        "已經試過、這裡不重複的：trend_anatomy（20／10 突破規則逐筆看）發現優勢全部來自躲掉大跌，中小型來回的日子 45 筆裡 21 筆"
        "是 15 天內就賠錢出場，72 次賣出裡 48 次用更貴的價格買回；rule_upgrades 試過賣出後更快買回（A1–A3）、分兩半、200 日均線"
        "斜率濾網、依波動調整、停損，都沒過；robust_trend 證明順勢的優勢換收盤時間、照一般人做法還在，但沒有分辨假跌破；"
        "mtf_gate 在日線閘門裡用 4 小時線抓進出，交易變多、沒過；reluctant_sellers 的 R2（流入少＋有賺就不下車）沒過，隨機錯位 76%。"
        "這份只看「下車那天收盤時價格本身的狀態」。",
        "名詞：假跌破＝跌破均線後很快又站回去、而且站回的價格更高。真的下跌＝下車後 60 天內先跌到下車價的 85% 以下。"
        "週線＝每週日收盤那一根；10 週均線＝最近 10 根週線收盤的平均。波動＝每天漲跌幅的標準差。",
        "分界（2019-01-01）、十個條件、所有門檻（20／60 天、15%、2%、30 天、90 天 30%、365 天中位數、20／180 天、10 週、至少 15／10 次）"
        "都是跑之前憑道理定的，沒有試過其他數字。十個條件加 45 組兩兩組合在發現期裡挑最好的，挑到的在發現期一定好看；只有測試期算數。",
        "其他資產和 24 個收盤時間各用自己的價格、自己的均線下車和自己的條件；收盤時間的成交量條件用前一根 UTC 日線的比特幣成交金額"
        "（那個收盤時已經收完）；MSTR 沒有用成交量（條件 i 當作不成立），週線用週日以前最後一個交易日。",
        f"偷看未來的檢查：把價格和成交量在 {'、'.join(CUTS)} 截斷重算，{lk_total} 次比對裡有 {lk_bad} 次不一致"
        f"{'（應該是 0）' if lk_bad else '，截斷前的條件和 H1、H2 持有完全一樣'}。",
        "就算通過，也只代表規則可以放進每日影子紀錄觀察；用的是同一段歷史，真正的考驗是之後每天實際記下的結果。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。測試期只有大約七年、一次大熊市（2022），"
        "真的下跌的次數很少。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載；考試本身 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    notes += problems

    return {
        "id": "whipsaw_anatomy",
        "title": "假跌破還是真的下跌：均線下車那一刻分得出來嗎",
        "question": ("50 日均線下車（收盤跌破均線）的那一刻，有沒有什麼當下就知道的事（長期趨勢、跌多深、之前漲多少、波動、成交量、週線）"
                     "可以分出「很快又站回去而且價格更高」的假跌破和真的下跌？如果分得出來，假跌破時多抱 20 天能不能比均線賺更多？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 K 線 API（BTC、ETH、SOL、BNB、XRP 對 USDT 的日線收盤和成交金額、比特幣 1 小時線）",
             "url": "https://api.binance.com/api/v3/klines"},
            {"name": "幣安現貨交易頁（BTC/USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
