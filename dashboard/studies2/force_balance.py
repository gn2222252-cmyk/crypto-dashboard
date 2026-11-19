"""Historical study "force_balance": measure a BUY force and a SELL force for bitcoin separately, in the same units
(US dollars over the last 30 days), and look at them TOGETHER - the 3x3 picture and the balance between them - before
asking whether either one, or the balance, says anything about the next 30-60 days.

Why this study exists (stated plainly). reluctant_sellers (Test A) found that, among days with the close above its
20-day average, days with HEAVY bitcoin deposits to exchanges were followed by the BEST 30-day returns (+25.5% versus
+4.6% for light deposits). One reading is "deposits are not selling pressure". The project owner's better reading:
deposits may well BE selling pressure, but on those days BUYING pressure was even stronger - so looking at one force
alone is meaningless; the two forces must be measured together and compared. This study does that.
Earlier related results (summarised honestly in the notes): exchange_reserves (falling exchange balances were followed
by returns opposite to the published claim; its rules lost to ma50); rally_sources (net exchange outflow and other
money-flow channels "led" rallies no more often than time-shifted copies of themselves); stablecoin_supply (supply
growth follows price rather than leading it); order_flow (taker-buy share had no usable forward relation);
price_impact (seller scarcity measured from price response failed); supply_demand ("few sellers + buying comes in"
failed); reluctant_sellers (above). How this differs: both forces are expressed in the SAME units (USD per 30 days),
each is sized SEPARATELY against its own past year, and they are CROSS-TABULATED (3x3), so "heavy deposits" can be
read conditional on how strong buying was at the same time. This is the fourth study of the supply/demand idea, so
the placebo bar is 5%, not the usual 10%.

Data (only free key-less endpoints, fixed user agent "btc-research-dashboard"; a refused endpoint is dropped and
named, never worked around):
  BTC daily closes: the dashboard's (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Coin Metrics community API, asset btc: FlowInExUSD (USD value of BTC sent to exchange-tagged addresses per day),
    FlowOutExUSD (USD value of BTC withdrawn from them), FlowInExNtv, FlowOutExNtv (the same in coins, report only).
    Store cache/_store/force_balance_cm.json (fallback only). Checked before writing this: these four return 200.
    The same exchange-flow metrics for usdt / usdc / usdt_eth / usdc_eth are NOT free (400 "not supported" or 403),
    so the stablecoin-deposit variant BUY2 cannot be computed and is dropped (stated in the notes).
  DefiLlama total stablecoin supply in USD (supply_demand's loader, own store cache/_store/force_balance_llama.json),
    used from 2019-07-01 as stablecoin_supply does (earlier coverage is thin); daily USD change, set to 0 on days
    whose log growth exceeds 15% in absolute value (stablecoin_supply's coverage-jump rule).
  Farside US spot bitcoin ETF daily net flow, Total column, USD millions (flows2's loader, own store
    cache/_store/force_balance_etf.json; etf_loop's store read as a last fallback), from 2024-01-11. Calendar days
    without trading count as 0; weekdays after Farside's last row are missing (BUY undefined there).
  Fair exam inputs as reluctant_sellers / price_impact: Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance) and
    MSTR (Yahoo) daily closes through main_signal_check's loaders.
  Timing: every source above is used one day late: a value dated D (Coin Metrics day D, DefiLlama point D, Farside
    US trading day D) is used from the BTC close labelled D+1.

Definitions (fixed before running; on data days D, all trailing-30-day sums D-29 .. D in USD, all 30 values needed):
  SELL(D) = sum of FlowInExUSD (USD value of BTC deposited to exchanges).
  BTCOUT(D) = sum of FlowOutExUSD (USD value of BTC withdrawn from exchanges: coins bought and taken away).
  STABLE(D) = sum of the cleaned daily USD change of total stablecoin supply (net new stablecoins); undefined before
    2019-07-31.
  ETF(D) = sum of US spot ETF net flow (0 before 2024-01-11, because the ETFs did not exist; negative when redeeming).
  BUY(D) = BTCOUT + STABLE (only where STABLE is defined; before that BUY is BTC withdrawals only) + ETF.
  rank(X)(D) = share of the previous 365 data days' X values (at least 330 non-missing) that are below X(D).
  HIGH = rank above 0.67, LOW = rank below 0.33, MID otherwise (percentiles, because the dollar scale grows over time).
  BALANCE(D) = (BUY - SELL) / (BUY + |SELL|) (undefined if the denominator is not positive), and its rank.
  A decision day t (UTC close t, held during t+1) uses data day t-1.
  Component first usable dates are reported. "Stablecoin era" = decision days whose whole 365-day BUY rank window
  includes STABLE (first STABLE day + 365 days); earlier years have BUY = BTC withdrawals (+ nothing else) and are
  marked; Test A is shown with them (full period) and without them (stablecoin era only).
  above20(t) = close(t) above the mean of the last 20 closes including t; ma50 = main_signal_check's.

Test A - the 2-D picture (main scientific result). Sample: decision days with both ranks defined and a full 30-day
  forward window. The 3x3 table BUY level (LOW/MID/HIGH) x SELL level (LOW/MID/HIGH): days and share per cell, mean and
  median of close[t+30]/close[t]-1 and close[t+60]/close[t]-1 (days without a 60-day window skipped for that column),
  mean deepest drop within 30 days min(close[t+1..t+30])/close[t]-1 capped at 0; 90% intervals from a circular block
  bootstrap of days (60-day blocks, 1,000 draws, fixed seed) inside each period: full, 2012-16, 2017-21, 2022-now
  (clipped to the sample), plus the stablecoin era (report).
  Owner's hypothesis, fixed now: (i) for a given SELL level, forward returns rise with BUY; (ii) for a given BUY
  level, forward returns FALL as SELL rises; (iii) so BUY HIGH & SELL LOW is the best cell and BUY LOW & SELL HIGH the
  worst.
  Statistics on 30-day means: best-minus-worst = mean(BUY HIGH, SELL LOW) - mean(BUY LOW, SELL HIGH);
  BUY effect = average over the three SELL levels of mean(BUY HIGH, s) - mean(BUY LOW, s);
  SELL effect = average over the three BUY levels of mean(b, SELL HIGH) - mean(b, SELL LOW) (SELL's effect with BUY
  held fixed); each with its bootstrap interval. Also reported: whether (i) and (ii) hold monotonically cell by cell.
  Pre-registered pass, all of: full-period best-minus-worst > 0 with a 90% interval excluding zero; best-minus-worst
  > 0 in at least two of the three sub-periods; full-period BUY effect > 0; full-period SELL effect < 0. If the SELL
  effect is positive even with BUY held fixed, the study says plainly that the data do not support "deposits =
  selling pressure" at this resolution.
  Also (report): correlation of the two ranks, correlation of log BUY and log SELL, the share of SELL-HIGH days that
  are also BUY-HIGH (versus the share of all days that are BUY-HIGH), and the 30-day forward mean of SELL-HIGH days
  split by BUY level - is heavy SELL followed by good returns only because BUY is high at the same time?
  Control (report): BALANCE HIGH minus BALANCE LOW 30-day forward mean (and deepest drop) matched on deciles of the
  trailing 30-day return close[t]/close[t-30]-1 (price_impact's matching, weights = BALANCE-HIGH days per decile,
  deciles holding both kinds only), full period, with the bootstrap interval - so "price already rose" is not the
  signal.

Test B - the balance as a signal (report). Forward returns by BALANCE-rank tercile (LOW/MID/HIGH) in every period, the
  HIGH-minus-LOW 30-day difference with its interval. Lead/lag: x(t) = BALANCE(t-1) - BALANCE(t-15) (its 14-day change
  known at t); correlation of x(t) with BTC's 14-day log return ending at t+k, k = -30..+30 (k > 0: balance first,
  k < 0: price first); noise band = 95th percentile of |corr| over 200 circular shifts of x by >= 180 days, pooled
  over the shown k.

Test C - two fixed rules. F1 = hold when ma50 is on OR (BALANCE rank above 0.67 AND above20) [early entry].
  F2 = hold when ma50 is on AND NOT (BALANCE rank below 0.33) [filter side; report only for the "can it add return"
  question, which it cannot]. Undefined ranks count as OFF.
  Fair exam = main_signal_check's, wired as price_impact's _Exam with first month 2015-01: BTC daily monthly starts
  2015-01 .. 2023-12 textbook 0.10%; clock starts 2018-01 .. 2023-12 x 24 closing hours textbook; the same cells as an
  ordinary person (decide at the clock's close, act 10 hours later, 0.25% cost, 10% of entries missed, 200
  simulations per clock, main_signal_check's seeds and its ma50 draws); ETH SOL BNB XRP (Binance) and MSTR (Yahoo)
  from their first valid month to three years before today, each with its OWN 50-day and 20-day averages and BTC's
  BALANCE flag as a market-wide flag. ma50 re-scored on exactly the same starts; the numbers of starts are reported.
  On the 24-clock grid a close on UTC date E uses the flag of decision candle E-1; MSTR's close on date D uses
  decision candle D-1. Ranking score = worst of the four "beats holding" shares with main_signal_check's eligibility.
  Median deepest loss = median over the clock textbook cells.
  Placebo: the BALANCE-HIGH flag series circularly shifted by k in [180, n-180] days from its first defined day,
  200 draws, the price conditions in place; the whole exam rebuilt with 20 ordinary-person simulations per clock
  (same seeds) and the real F1 re-scored the same way; placebo share = share of draws whose ranking score is at least
  the real F1's.
  Pre-registered pass for F1, all three: ranking score >= ma50's; median deepest loss no more than 5 points worse than
  ma50's; placebo share <= 5% (the fourth study of the supply/demand idea, hence 5%).

Report only: for the rallies from rally_sources._rallies (+80% / 30%), the average BUY rank, SELL rank and BALANCE
  rank in the 30 days before the trough, trough -> ma50 date, ma50 date -> peak, and the 60 days after the peak; the
  same 60 days before / after each major top (a running all-time-high close followed by a fall of at least 50% before
  any higher close) and over the 2018 and 2022 calendar years; F1's early entries (F1 turns on while ma50 is off):
  "right" = ma50 turns on within 30 days at a higher close; each stint runs to the first close where F1 goes to cash
  or ma50 turns on, BTC return minus 0.10% per trade (reluctant_sellers' arithmetic); calendar-year return of F1 minus
  ma50 (0.10% cost).
Look-ahead check: all inputs cut at three dates (sources one day earlier); no rank, flag or F1 position up to the cut
  may change.
Honesty: exchange flows are today's address tags applied backwards (older years understated; some "withdrawals" and
  "deposits" are internal shuffles between an exchange's own wallets or exchange-to-exchange transfers, which inflate
  BOTH forces together - which is why the two are highly correlated and why the balance, not either level, is the
  thing to read); OTC desks and, before 2024, any off-exchange buying are invisible; stablecoin issuance is not all used
  to buy BTC; BUY's composition changes in 2019 (stablecoins added) and 2024 (ETFs added), which shifts its level
  inside the rank window for a year after each; thresholds (30 / 365 days, terciles, 20-day average, 60-day blocks)
  were fixed by reasoning before the run, no alternatives tried. A pass would only make F1 a candidate for the daily
  shadow log. Not advice.
Today: "比特幣：50 日均線｜F1" ("持有｜現金" style), F2, today's BUY and SELL (USD, 30 days) with components, their
  ranks, BALANCE and its rank, and today's 3x3 cell.
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
CM_METRICS = ("FlowInExUSD", "FlowOutExUSD", "FlowInExNtv", "FlowOutExNtv")
LLAMA = sd.LLAMA
FARSIDE_URL = flows2.FARSIDE["BTC"][0]
STORE_CM = common.STORE / "force_balance_cm.json"
STORE_LL = common.STORE / "force_balance_llama.json"
STORE_ETF = common.STORE / "force_balance_etf.json"
ETF_FALLBACK = common.STORE / "etf_loop_btc_frame.json"  # read only
STABLE_FROM, JUMP = "2019-07-01", 0.15
ETF_FROM = pd.Timestamp("2024-01-11")

SUM_DAYS, WIN, WIN_MIN = 30, 365, 330
HI_Q, LO_Q = 0.67, 0.33
A20 = 20
BLOCK, BOOT = 60, 1000
FIRST_MONTH = msc.DAILY_STARTS[0]
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.05
RIGHT_DAYS, AFTER_PEAK, PRE_TROUGH, TOP_WIN, TOP_FALL = 30, 60, 30, 60, 0.50
KI = {"ma50": 1, "f1": 121, "f2": 122}
LL_K = tuple(range(-30, 31))
LL_SHOW = (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30)
CUTS = ("2016-06-30", "2020-06-30", "2024-06-30")
PERIODS = (("2012–16", "2012-01-01", "2016-12-31"), ("2017–21", "2017-01-01", "2021-12-31"),
           ("2022 起", "2022-01-01", "2100-01-01"))
STABLE_ERA = "穩定幣時期（買方含穩定幣）"
LV = ("低", "中", "高")
BEST, WORST = 3 * 2 + 0, 3 * 0 + 2  # cell = 3 * BUY level + SELL level
NAME = {"ma50": "50 日均線（基準）",
        "f1": "F1：50 日均線，或「力量差在過去一年前三分之一＋站上 20 日均線」也持有（提早上車）",
        "f2": "F2：50 日均線，但「力量差在過去一年後三分之一」時不持有（只描述）"}
SHORT = {"ma50": "50 日均線", "f1": "F1", "f2": "F2"}

_d = cr._d
_ok, _p, _pt, _pc, _ci, _share, _share1 = sd._ok, sd._p, sd._pt, sd._pc, sd._ci, sd._share, sd._share1


def _usd(v) -> str:
    """USD -> 億美元 with sign when negative."""
    return f"{float(v) / 1e8:,.0f} 億美元" if _ok(v) else "—"


def _r(v) -> str:
    """rank 0..1 -> percentile text"""
    return f"{float(v) * 100:.0f}" if _ok(v) else "—"


# ---------- data ----------

def _cm_fetch() -> dict:
    reply = common.get_json(CM_URL, {"assets": "btc", "metrics": ",".join(CM_METRICS), "frequency": "1d",
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
        for m in CM_METRICS:
            try:
                v = float(row.get(m))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                rec[m] = v
        if day and rec:
            out[day] = rec
    if len(out) < 3000:
        raise RuntimeError(f"Coin Metrics 只回傳 {len(out)} 天")
    return out


def _etf_load() -> tuple[pd.Series, str | None]:
    """Farside BTC ETF Total net flow per US trading day (USD)."""
    try:
        f = flows2._etf_frame("BTC")
        tot = f["Total"].astype(float)
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_ETF.write_text(json.dumps({str(k): float(v) for k, v in tot.items()}), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        prob = None
    except Exception as exc:  # noqa: BLE001 - blocked: stored data, never worked around
        prob = f"這次 Farside 取不到（{type(exc).__name__}），用之前存下來的資料。"
        if STORE_ETF.exists():
            tot = pd.Series(json.loads(STORE_ETF.read_text(encoding="utf-8")), dtype=float)
        elif ETF_FALLBACK.exists():
            f = pd.read_json(io.StringIO(ETF_FALLBACK.read_text(encoding="utf-8")), orient="split")
            f.index = [str(i)[:10] for i in f.index]
            tot = f["Total"].astype(float)
        else:
            raise RuntimeError("Farside ETF 流量取不到，也沒有存檔") from exc
    tot.index = pd.to_datetime(tot.index)
    tot = tot[~tot.index.duplicated(keep="last")].sort_index()
    tot = tot[tot.index < pd.Timestamp(common.today_utc())]
    return tot * 1e6, prob


def _load() -> dict:
    problems = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    raw, n = sd._stored(STORE_CM, _cm_fetch)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
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
    return {"close": close, "cm": cm, "sup": sup, "etf": etf, "other": other, "problems": problems}


# ---------- the two forces ----------

def _trail_rank(x: pd.Series) -> pd.Series:
    """Share of the previous WIN values (at least WIN_MIN non-missing) below today's value."""
    v = x.values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    if n <= WIN:
        return pd.Series(out, index=x.index)
    win = np.lib.stride_tricks.sliding_window_view(v[:-1], WIN)  # row j = v[j .. j+WIN-1], today = j+WIN
    cur = v[WIN:]
    fin = np.isfinite(win)
    cnt = fin.sum(axis=1)
    below = (np.where(fin, win, np.inf) < cur[:, None]).sum(axis=1)
    ok = np.isfinite(cur) & (cnt >= WIN_MIN)
    out[WIN:] = np.where(ok, below / np.maximum(cnt, 1), np.nan)
    return pd.Series(out, index=x.index)


def _stable_daily(sup: pd.Series) -> pd.Series:
    lv = sup[sup.index >= pd.Timestamp(STABLE_FROM)]
    lv = lv.where(lv > 0)
    d = lv.diff()
    g = np.log(lv).diff()
    return d.where(g.abs() <= JUMP, 0.0).iloc[1:]


def _etf_daily(etf: pd.Series, last: pd.Timestamp) -> pd.Series:
    """Calendar-day ETF flow: 0 before launch and on no-trading days; weekdays after Farside's last row missing."""
    days = pd.date_range(min(ETF_FROM, last), last, freq="D")
    s = etf.reindex(days)
    lr = etf.index[-1] if len(etf) else ETF_FROM
    inside = (days >= ETF_FROM) & (days <= lr)
    s[inside & s.isna().values] = 0.0
    after = days > lr
    s[after & (days.dayofweek >= 5)] = 0.0
    return s


def _forces(cm: pd.DataFrame, sup: pd.Series, etf: pd.Series) -> dict:
    """Data-day series of the two forces, their ranks and the balance."""
    idx = cm.index
    sell = cm["FlowInExUSD"].rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    out_ = cm["FlowOutExUSD"].rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    sell_n = cm["FlowInExNtv"].rolling(SUM_DAYS, min_periods=SUM_DAYS).sum() if "FlowInExNtv" in cm else None
    out_n = cm["FlowOutExNtv"].rolling(SUM_DAYS, min_periods=SUM_DAYS).sum() if "FlowOutExNtv" in cm else None
    sd_ = _stable_daily(sup)
    st30 = sd_.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum().reindex(idx)
    ed = _etf_daily(etf, idx[-1])
    pre = pd.Series(0.0, index=idx[idx < ETF_FROM])
    ed = pd.concat([pre, ed[ed.index >= ETF_FROM]]).reindex(idx)
    e30 = ed.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    first_stable = st30.first_valid_index()
    # before the first STABLE value BUY has no stablecoin part (0); afterwards a missing STABLE makes BUY missing
    st_use = st30.copy() if first_stable is not None else pd.Series(0.0, index=idx)
    if first_stable is not None:
        st_use[idx < first_stable] = 0.0
    buy = out_ + st_use + e30
    den = buy + sell.abs()
    bal = ((buy - sell) / den).where(den > 0)
    return {"sell": sell, "out": out_, "stable": st30, "etf": e30, "buy": buy, "bal": bal,
            "sell_n": sell_n, "out_n": out_n, "first_stable": first_stable,
            "rb": _trail_rank(buy), "rs": _trail_rank(sell), "rbal": _trail_rank(bal)}


def _level(rank: np.ndarray) -> np.ndarray:
    """0 LOW, 1 MID, 2 HIGH, -1 undefined."""
    return np.where(~np.isfinite(rank), -1, np.where(rank > HI_Q, 2, np.where(rank < LO_Q, 0, 1)))


def _legs(fc: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision-date values (data day t-1)."""
    out = pd.DataFrame(index=index)
    for k in ("sell", "out", "stable", "etf", "buy", "bal", "rb", "rs", "rbal"):
        out[k] = fc[k].reindex(index - DAY).values
    out["lb"], out["ls"], out["lbal"] = _level(out["rb"].values), _level(out["rs"].values), _level(out["rbal"].values)
    out["hi"] = (out["lbal"] == 2).astype(float)
    out["lo"] = (out["lbal"] == 0).astype(float)
    out["defined"] = (out["lb"] >= 0) & (out["ls"] >= 0)
    return out


# ---------- Test A / B ----------

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


def _group_stats(grp: np.ndarray, k: int, fwd: dict, idx: np.ndarray) -> dict:
    """Groups 0..k-1 (-1 left out): mean / median with bootstrap intervals; per-draw 30-day means kept."""
    gb = grp[idx]
    out = {"n": [int((grp == j).sum()) for j in range(k)], "bm30": {}}
    for h in (30, 60, "deep"):
        v = fwd[h]
        vb = v[idx]
        for j in range(k):
            m = (grp == j) & np.isfinite(v)
            mb = (gb == j) & np.isfinite(vb)
            cnt = mb.sum(axis=1)
            bm = np.where(cnt > 0, np.where(mb, vb, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            with np.errstate(all="ignore"):
                bmed = np.nanmedian(np.where(mb, vb, np.nan), axis=1)
            if h == 30:
                out["bm30"][j] = bm
            out[(h, j)] = {"mean": float(v[m].mean()) if m.any() else None,
                           "med": float(np.median(v[m])) if m.any() else None,
                           "mean_ci": pi._q(bm), "med_ci": pi._q(bmed)}
    return out


def _effects(st: dict) -> dict:
    m = {j: st[(30, j)]["mean"] for j in range(9)}
    b = st["bm30"]

    def pt(f):
        try:
            v = f(m)
            return float(v) if _ok(v) else None
        except TypeError:
            return None
    bw = lambda x: x[BEST] - x[WORST]  # noqa: E731
    be = lambda x: sum(x[6 + s] - x[0 + s] for s in range(3)) / 3  # noqa: E731
    se = lambda x: sum(x[3 * q + 2] - x[3 * q] for q in range(3)) / 3  # noqa: E731
    out = {"bw": pt(bw), "be": pt(be), "se": pt(se),
           "bw_ci": pi._q(bw(b)), "be_ci": pi._q(be(b)), "se_ci": pi._q(se(b))}
    # monotonic checks: (i) within each SELL level, LOW < MID < HIGH in BUY; (ii) within each BUY level, falling in SELL
    i_ok = sum(1 for s in range(3) if all(_ok(m[3 * q + s]) for q in range(3)) and m[s] < m[3 + s] < m[6 + s])
    ii_ok = sum(1 for q in range(3) if all(_ok(m[3 * q + s]) for s in range(3)) and m[3 * q] > m[3 * q + 1] > m[3 * q + 2])
    out["mono"] = (i_ok, ii_ok)
    vals = [m[j] if _ok(m[j]) else -np.inf for j in range(9)]
    out["best"], out["worst"] = int(np.argmax(vals)), int(np.argmin([v if v != -np.inf else np.inf for v in vals]))
    return out


def _spans(idx, keep, start, end, era_from):
    sp = [("全期", start, end)] + [(lab, max(pd.Timestamp(a), start), min(pd.Timestamp(b), end))
                                  for lab, a, b in PERIODS]
    if era_from is not None:
        sp.append((STABLE_ERA, max(era_from, start), end))
    return sp


def _test_a(close: pd.Series, lg: pd.DataFrame, era_from) -> dict:
    fwd_all = _forward(close)
    past = (close / close.shift(30) - 1).values
    idx = close.index
    keep = lg["defined"].values & np.isfinite(fwd_all[30])
    start, end = idx[keep][0], idx[keep][-1]
    cell_all = np.where(lg["defined"].values, 3 * lg["lb"].values + lg["ls"].values, -1)
    bal_all = lg["lbal"].values
    rng = np.random.default_rng(SEED + 1101)
    res = {"start": start, "end": end, "per": {}, "bal": {}}
    for lab, a, b in _spans(idx, keep, start, end, era_from):
        m = keep & (idx >= a) & (idx <= b)
        if m.sum() < 2 * BLOCK:
            continue
        n = int(m.sum())
        bidx = _boot_idx(n, rng)
        fwd = {h: fwd_all[h][m] for h in fwd_all}
        st = _group_stats(cell_all[m], 9, fwd, bidx)
        st.update(_effects(st))
        st["span"], st["days"] = (idx[m][0], idx[m][-1]), n
        res["per"][lab] = st
        # Test B: BALANCE terciles on the same days
        bg = np.where(lg["rbal"].notna().values[m], bal_all[m], -1)
        bs = _group_stats(bg, 3, fwd, bidx)
        bs["hl"] = (bs[(30, 2)]["mean"] - bs[(30, 0)]["mean"]) if _ok(bs[(30, 2)]["mean"]) and _ok(bs[(30, 0)]["mean"]) \
            else None
        bs["hl_ci"] = pi._q(bs["bm30"][2] - bs["bm30"][0])
        bs["span"] = st["span"]
        res["bal"][lab] = bs
        if lab == "全期":
            rb, rsl = lg["rb"].values[m], lg["rs"].values[m]
            res["corr_rank"] = float(np.corrcoef(rb, rsl)[0, 1])
            lb_, ls_ = np.log(np.abs(lg["buy"].values[m])), np.log(lg["sell"].values[m])
            okl = np.isfinite(lb_) & np.isfinite(ls_) & (lg["buy"].values[m] > 0)
            res["corr_log"] = float(np.corrcoef(lb_[okl], ls_[okl])[0, 1])
            sh = lg["ls"].values[m] == 2
            res["sh_hi_hi"] = float((lg["lb"].values[m][sh] == 2).mean())
            res["sh_buy_hi"] = float((lg["lb"].values[m] == 2).mean())
            res["sell_hi_by_buy"] = [st[(30, 3 * q + 2)] for q in range(3)]
            res["sell_hi_n"] = [st["n"][3 * q + 2] for q in range(3)]
            # sell-only view (one force alone), to connect with reluctant_sellers
            sg = lg["ls"].values[m]
            so = _group_stats(sg, 3, fwd, bidx)
            res["sell_only"] = so
            bo = _group_stats(lg["lb"].values[m], 3, fwd, bidx)
            res["buy_only"] = bo
            # control: BALANCE HIGH vs LOW matched on trailing 30-day return deciles
            pv = past[m]
            okp = np.isfinite(pv)
            edges = np.quantile(pv[okp], np.linspace(0, 1, 11))
            sub = okp & np.isin(bg, (0, 2))
            dec = np.full(n, -1)
            dec[sub] = np.clip(np.searchsorted(edges, pv[sub], side="right") - 1, 0, 9)
            sc = (bg == 2).astype(float)
            got = {}
            for key in (30, "deep"):
                v = np.nan_to_num(fwd[key], nan=0.0)
                got[key] = (float(pi._matched(v, sc, dec, sub)[0]),
                            pi._q(pi._matched(v[bidx], sc[bidx], dec[bidx], sub[bidx])))
            res["matched"] = got
            res["past"] = [float(np.nanmean(pv[bg == j])) for j in range(3)]
            res["past_cell"] = [float(np.nanmean(pv[cell_all[m] == j])) if (cell_all[m] == j).any() else None
                                for j in range(9)]
    full = res["per"]["全期"]
    subs = [lab for lab, _, _ in PERIODS if lab in res["per"]]
    lo, hi = full["bw_ci"]
    excl = _ok(lo) and _ok(hi) and lo > 0
    pos_subs = sum(1 for lab in subs if _ok(res["per"][lab]["bw"]) and res["per"][lab]["bw"] > 0)
    bpos = _ok(full["be"]) and full["be"] > 0
    sneg = _ok(full["se"]) and full["se"] < 0
    bwpos = _ok(full["bw"]) and full["bw"] > 0
    res["pass"] = {"bw": bool(bwpos and excl), "pos_subs": pos_subs, "n_subs": len(subs), "be": bool(bpos),
                   "se": bool(sneg), "pass": bool(bwpos and excl and pos_subs >= 2 and bpos and sneg)}
    return res


def _leadlag(fc: dict, close: pd.Series) -> dict:
    bal = fc["bal"]
    x_data = bal - bal.shift(14)
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
    rng = np.random.default_rng(SEED + 111)
    pool = []
    for s in rng.integers(GAP, n - GAP + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL_SHOW]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n, "from": close.index[first]}


# ---------- the fair exam ----------

def _combine(kind: str, ma, a20, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "f1":
        return np.maximum(ma, f * a20)
    if kind == "f2":
        return ma * (1.0 - f)
    raise ValueError(kind)


class _Exam(pi._Exam):
    """price_impact's exam (main_signal_check's four checks on chosen starts) with two ways to combine the flag."""

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


# ---------- report-only pieces ----------

def _cal_years(close: pd.Series, pos: np.ndarray) -> dict:
    change = close.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    s = pd.Series(net, index=close.index)
    return {int(y): float(np.prod(1 + g.values) - 1) for y, g in s.groupby(s.index.year)}


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
            ret = c[-1] / c[t] - 1 - COST
        else:
            ret = c[e] / c[t] - 1 - COST * (2 if pos[e] < 0.5 else 1)
        known = (u is not None and u - t <= RIGHT_DAYS) or n - 1 - t > RIGHT_DAYS
        out.append({"day": idx[t], "right": bool(right), "ret": float(ret), "known": known})
    return out


def _avg3(lg: pd.DataFrame, a: int, b: int, n: int) -> tuple[str, list]:
    a, b = max(a, 0), min(b, n)
    if b <= a:
        return "—", []
    seg = lg.iloc[a:b]
    vals = [seg[k].dropna() for k in ("rb", "rs", "rbal")]
    if not len(vals[0]) or not len(vals[1]):
        return "—", []
    txt = f"{_r(vals[0].mean())}｜{_r(vals[1].mean())}｜{_r(vals[2].mean()) if len(vals[2]) else '—'}"
    return txt, [list(v.values) for v in vals]


def _rally_rows(close: pd.Series, lg: pd.DataFrame) -> list:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    n = len(close)
    rows, pool = [], {s: [[], [], []] for s in ("pre", "tm", "mp", "after")}

    def cell(a, b, seg):
        txt, v = _avg3(lg, a, b, n)
        for j, x in enumerate(v):
            pool[seg][j] += x
        return txt

    for i, r in enumerate(rallies, 1):
        it, im, ip = r["it"], r["im"], r["ip"]
        rows.append([f"{i}（{_d(r['t'])} → {_d(r['p'])}）" + ("" if r["done"] else "，還沒結束"),
                     f"{r['gain'] * 100:+.0f}%", _d(r["m"]), cell(it - PRE_TROUGH, it, "pre"), cell(it, im, "tm"),
                     cell(im, ip + 1, "mp"), cell(ip + 1, ip + 1 + AFTER_PEAK, "after") if r["done"] else "還沒結束"])
    tot = []
    for seg in ("pre", "tm", "mp", "after"):
        v = pool[seg]
        tot.append(f"{_r(np.mean(v[0]))}｜{_r(np.mean(v[1]))}｜{_r(np.mean(v[2])) if v[2] else '—'}" if v[0] else "—")
    rows.append(["全部合計（每天平均）", "", ""] + tot)
    return rows


def _tops(close: pd.Series) -> list[int]:
    """Running all-time-high closes followed by a >= 50% fall before any higher close."""
    c = close.values.astype(float)
    out, hi, i = [], 0, 1
    for i in range(1, len(c)):
        if c[i] > c[hi]:
            hi = i
        elif c[i] <= c[hi] * (1 - TOP_FALL) and (not out or out[-1] != hi):
            out.append(hi)
    return [k for k in out if close.index[k] >= pd.Timestamp("2012-06-01")]


def _top_rows(close: pd.Series, lg: pd.DataFrame) -> list:
    n = len(close)
    rows = []
    for k in _tops(close):
        rows.append([f"高點 {_d(close.index[k])}（{close.iloc[k]:,.0f} 美元）", _avg3(lg, k - TOP_WIN, k, n)[0],
                     _avg3(lg, k + 1, k + 1 + TOP_WIN, n)[0]])
    for y in (2018, 2022):
        m = np.flatnonzero(close.index.year == y)
        rows.append([f"{y} 年整年的下跌（{close.iloc[m[0]]:,.0f} → {close.iloc[m[-1]]:,.0f} 美元）",
                     _avg3(lg, m[0], m[-1] + 1, n)[0], "—"])
    return rows


def _lookahead(data: dict, lg: pd.DataFrame, pos_f1: np.ndarray, ex: _Exam) -> tuple[int, int]:
    total = bad = 0
    close = data["close"]
    for cut in CUTS:
        c = pd.Timestamp(cut)
        cm = data["cm"][data["cm"].index <= c - DAY]
        sup = data["sup"][data["sup"].index <= c - DAY]
        etf = data["etf"][data["etf"].index <= c - DAY]
        cl = close[close.index <= c]
        lg2 = _legs(_forces(cm, sup, etf), cl.index)
        cols = ["lb", "ls", "lbal"]
        total += 1
        bad += int(not np.array_equal(lg2[cols].values, lg.loc[cl.index, cols].values))
        ma2 = msc._targets(cl)["ma50"].values.astype(float)
        p2 = _combine("f1", ma2, sd._above20(cl.values), lg2["hi"].values)
        total += 1
        bad += int(not np.array_equal(p2, pos_f1[:len(cl)]))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    fc = _forces(data["cm"], data["sup"], data["etf"])
    lg = _legs(fc, close.index)
    era_from = (fc["first_stable"] + pd.Timedelta(days=WIN + 1)) if fc["first_stable"] is not None else None
    a = _test_a(close, lg, era_from)
    ll = _leadlag(fc, close)
    ma = msc._targets(close)["ma50"].values.astype(float)
    fstart = close.index[int(np.argmax(lg["rbal"].notna().values))]
    t_a = time.time() - t0

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(close, data["other"], price, b0, pre, pd.Timestamp(FIRST_MONTH))
    hi = pd.Series(lg["hi"].values, index=close.index)
    lo = pd.Series(lg["lo"].values, index=close.index)
    flags = {"ma50": (None, "ma50"), "f1": (hi, "f1"), "f2": (lo, "f2")}
    s = {k: ex.summary(f, KI[k], SIMS, kind) for k, (f, kind) in flags.items()}
    t_exam = time.time() - t0

    small = ex.summary(hi, KI["f1"], PL_SIMS, "f1")
    n = int((close.index >= fstart).sum())
    prng = np.random.default_rng(SEED + 23)
    hits, lows = 0, []
    for k in prng.integers(GAP, n - GAP, DRAWS):
        x = ex.summary(sd._roll(hi, fstart, int(k)), KI["f1"], PL_SIMS, "f1")
        lows.append(x["low"])
        hits += int(sd._score_ge(x, small))
    placebo = hits / DRAWS

    pos = {k: ex.btc_pos(f, kind) for k, (f, kind) in flags.items()}
    cal = {k: _cal_years(close, pos[k]) for k in ("ma50", "f1", "f2")}
    m15 = close.index >= pd.Timestamp(FIRST_MONTH)
    tim = {k: float(np.nanmean(np.nan_to_num(pos[k][m15]))) for k in ("ma50", "f1", "f2")}
    differ = {k: int((pos[k][m15] != np.nan_to_num(pos["ma50"][m15])).sum()) for k in ("f1", "f2")}
    entries = _entries(close, ma, pos["f1"], fstart)
    rally = _rally_rows(close, lg)
    tops = _top_rows(close, lg)
    look = _lookahead(data, lg, pos["f1"], ex)
    return {"fc": fc, "lg": lg, "era_from": era_from, "a": a, "ll": ll, "ma": ma, "fstart": fstart, "ex": ex,
            "s": s, "small": small, "placebo": placebo, "placebo_med": float(np.median(lows)), "pos": pos,
            "cal": cal, "tim": tim, "differ": differ, "entries": entries, "rally": rally, "tops": tops,
            "look": look, "problem": problem, "secs": time.time() - t0, "secs_exam": t_exam - t_a}


# ---------- today ----------

def _now(data: dict, r: dict) -> list[dict]:
    close, lg, fc = data["close"], r["lg"], r["fc"]
    last = close.index[-1]
    word = {1: "持有", 0: "現金"}
    st = {k: int(r["pos"][k][-1] > 0.5) for k in ("ma50", "f1", "f2")}
    row = lg.iloc[-1]
    day = last - DAY
    c = close.values.astype(float)
    m20, m50 = float(c[-20:].mean()), float(c[-50:].mean())
    cell = (f"買方{LV[int(row['lb'])]}｜賣方{LV[int(row['ls'])]}" if row["lb"] >= 0 and row["ls"] >= 0 else "資料不足")
    full = r["a"]["per"]["全期"]
    cm30 = full[(30, 3 * int(row["lb"]) + int(row["ls"]))]["mean"] if row["lb"] >= 0 and row["ls"] >= 0 else None
    stable_txt = _usd(row["stable"]) if _ok(row["stable"]) else "—"
    return [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
        {"label": "比特幣：50 日均線｜F1", "value": f"{word[st['ma50']]}｜{word[st['f1']]}"},
        {"label": "F2（只描述的過濾規則）", "value": word[st["f2"]]},
        {"label": f"買方力量：近 30 天（用 {_d(day)} 的資料）",
         "value": (f"{_usd(row['buy'])}＝從交易所提走的比特幣 {_usd(row['out'])}＋新增穩定幣 {stable_txt}"
                   f"＋ETF 淨流入 {_usd(row['etf'])}")},
        {"label": "賣方力量：近 30 天送進交易所的比特幣", "value": _usd(row["sell"])},
        {"label": "買方力量在過去一年的排名（0–100，越高越強）", "value": _r(row["rb"])},
        {"label": "賣方力量在過去一年的排名（0–100，越高越強）", "value": _r(row["rs"])},
        {"label": "力量差（買 − 賣）÷（買 + 賣）｜它在過去一年的排名",
         "value": f"{row['bal']:+.3f}｜{_r(row['rbal'])}" if _ok(row["bal"]) else "資料不足"},
        {"label": "今天在 3×3 表的哪一格", "value": cell + (f"（這一格歷史上之後 30 天平均 {_p(cm30)}）" if _ok(cm30) else "")},
        {"label": "收盤｜20 日均線｜50 日均線", "value": f"{c[-1]:,.0f}｜{m20:,.0f}｜{m50:,.0f}"},
        {"label": "這幾格的意思", "value": "只描述最後一根日線收盤後的狀態，不是買賣建議；F1 沒有通過考試前不會用"},
    ]


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, a, ex, lg, fc = data["close"], r["s"], r["a"], r["ex"], r["lg"], r["fc"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    full = a["per"]["全期"]
    ap = a["pass"]

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    # ----- table 0: data -----
    cm = data["cm"]
    fv = lambda x: _d(x.first_valid_index()) if x is not None and x.first_valid_index() is not None else "—"  # noqa: E731
    nd = int(lg["defined"].values.sum())
    rows0 = [["Coin Metrics 送進／提出交易所的美元金額（FlowInExUSD／FlowOutExUSD）第一天", fv(cm["FlowInExUSD"])],
             ["賣方力量、從交易所提走的 30 天加總第一天（資料日）", fv(fc["sell"])],
             ["新增穩定幣的 30 天加總第一天（DefiLlama，從 2019-07-01 用起）", fv(fc["stable"])],
             ["ETF 淨流入第一天（Farside）", _d(data["etf"].index[0]) if len(data["etf"]) else "—"],
             ["兩股力量的排名第一次算得出來（資料日）", fv(fc["rb"].where(fc["rs"].notna()))],
             ["力量差的排名第一次算得出來（資料日）", fv(fc["rbal"])],
             ["第一個能拿來決定的收盤日", _d(a["start"])],
             ["「穩定幣時期」從哪天起（排名的整整一年都含穩定幣）", _d(r["era_from"]) if r["era_from"] is not None else "—"],
             ["測試 A 的天數（全期）", f"{full['days']:,}"],
             ["兩股力量都有定義的天數（含最後 30 天）", f"{nd:,}"],
             ["Coin Metrics 最後一天（資料日）", _d(cm["FlowInExUSD"].last_valid_index())],
             ["穩定幣交易所流入（USDT／USDC 的 FlowInExUSD）", "不是免費的（Coin Metrics 回 400／403），所以 BUY2 沒辦法算，拿掉"]]

    # ----- Test A -----
    rows_a = []
    for lab, x in a["per"].items():
        sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
        first = True
        for q in range(3):
            for sl in range(3):
                j = 3 * q + sl
                c30, c60, cd = x[(30, j)], x[(60, j)], x[("deep", j)]
                rows_a.append([sp if first else "", LV[q], LV[sl], f"{x['n'][j]:,}（{x['n'][j] / x['days'] * 100:.0f}%）",
                               f"{_p(c30['mean'])} {_ci(*c30['mean_ci'])}", f"{_p(c30['med'])} {_ci(*c30['med_ci'])}",
                               f"{_p(c60['mean'])} {_ci(*c60['mean_ci'])}", f"{_p(c60['med'])} {_ci(*c60['med_ci'])}",
                               f"{_p(cd['mean'])} {_ci(*cd['mean_ci'])}"])
                first = False
    cols_a = ["期間", "買方力量", "賣方力量", "天數（占比）", "之後 30 天平均 ［90% 區間］", "之後 30 天中位數 ［90% 區間］",
              "之後 60 天平均 ［90% 區間］", "之後 60 天中位數 ［90% 區間］", "30 天內最深跌幅平均 ［90% 區間］"]
    rows_m = []
    for q in range(3):
        rows_m.append([f"買方{LV[q]}"] + [f"{_p(full[(30, 3 * q + sl)]['mean'])}（{full['n'][3 * q + sl]:,} 天）"
                                         for sl in range(3)])
    rows_eff = []
    for lab, x in a["per"].items():
        rows_eff.append([lab, f"{_pt(x['bw'])} {_ci(*x['bw_ci'], f=_pt)}", f"{_pt(x['be'])} {_ci(*x['be_ci'], f=_pt)}",
                         f"{_pt(x['se'])} {_ci(*x['se_ci'], f=_pt)}", f"{x['mono'][0]} / 3", f"{x['mono'][1]} / 3",
                         f"買方{LV[x['best'] // 3]}＋賣方{LV[x['best'] % 3]}｜買方{LV[x['worst'] // 3]}＋賣方{LV[x['worst'] % 3]}"])
    rows_ap = [["全期「最好格 − 最差格」> 0，而且 90% 區間不含零", f"{_pt(full['bw'])} {_ci(*full['bw_ci'], f=_pt)}", yes(ap["bw"])],
               ["三段裡至少兩段「最好格 − 最差格」> 0", f"{ap['pos_subs']} 段（共 {ap['n_subs']} 段）", yes(ap["pos_subs"] >= 2)],
               ["全期買方效果 > 0（賣方固定時，買方高減買方低）", f"{_pt(full['be'])} {_ci(*full['be_ci'], f=_pt)}", yes(ap["be"])],
               ["全期賣方效果 < 0（買方固定時，賣方高減賣方低）", f"{_pt(full['se'])} {_ci(*full['se_ci'], f=_pt)}", yes(ap["se"])],
               ["四條都要過", "", yes(ap["pass"])]]
    so, bo = a["sell_only"], a["buy_only"]
    rows_c = [["兩股力量排名的相關（全期）", f"{a['corr_rank']:+.2f}"],
              ["兩股力量美元金額（取對數）的相關", f"{a['corr_log']:+.2f}"],
              ["賣方高的日子裡，買方也高的比例", f"{_share(a['sh_hi_hi'])}（所有日子裡買方高的比例 {_share(a['sh_buy_hi'])}）"],
              ["只看賣方一股力量：低｜中｜高，之後 30 天平均",
               "｜".join(f"{_p(so[(30, j)]['mean'])}（{so['n'][j]:,} 天）" for j in range(3))],
              ["只看買方一股力量：低｜中｜高，之後 30 天平均",
               "｜".join(f"{_p(bo[(30, j)]['mean'])}（{bo['n'][j]:,} 天）" for j in range(3))],
              ["賣方高的日子，再依買方低｜中｜高分開，之後 30 天平均",
               "｜".join(f"{_p(x['mean'])}（{nn:,} 天）" for x, nn in zip(a["sell_hi_by_buy"], a["sell_hi_n"]))]]
    mt30, mtd = a["matched"][30], a["matched"]["deep"]
    bf = a["bal"]["全期"]
    rows_ctl = [["力量差高 − 力量差低，之後 30 天平均（沒配對）", f"{_pt(bf['hl'])} {_ci(*bf['hl_ci'], f=_pt)}"],
                ["配對過去 30 天漲幅（十組）後，之後 30 天平均的差", f"{_pt(mt30[0])} {_ci(*mt30[1], f=_pt)}"],
                ["配對後，30 天內最深跌幅的差", f"{_pt(mtd[0])} {_ci(*mtd[1], f=_pt)}"],
                ["力量差低｜中｜高的日子，過去 30 天平均已經漲了", "｜".join(_p(v) for v in a["past"])]]

    # ----- Test B -----
    rows_b = []
    for lab, x in a["bal"].items():
        sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
        for j in range(3):
            c30, c60, cd = x[(30, j)], x[(60, j)], x[("deep", j)]
            rows_b.append([sp if j == 0 else "", LV[j], f"{x['n'][j]:,}", f"{_p(c30['mean'])} {_ci(*c30['mean_ci'])}",
                           f"{_p(c30['med'])} {_ci(*c30['med_ci'])}", f"{_p(c60['mean'])} {_ci(*c60['mean_ci'])}",
                           f"{_p(cd['mean'])} {_ci(*cd['mean_ci'])}",
                           f"{_pt(x['hl'])} {_ci(*x['hl_ci'], f=_pt)}" if j == 2 else ""])
    ll = r["ll"]
    band = ll["band"]
    rows_l = [[f"{k:+d} 天" + ("（價格先動）" if k < 0 else "（力量差先動）" if k > 0 else "（同一天）"),
               f"{ll['corr'][k]:+.3f}", "是" if abs(ll["corr"][k]) > band else "否"] for k in LL_SHOW]
    neg = {k: v for k, v in ll["corr"].items() if k < 0}
    posk = {k: v for k, v in ll["corr"].items() if k > 0}
    kn = max(neg, key=lambda k: abs(neg[k]))
    kp = max(posk, key=lambda k: abs(posk[k]))
    rows_l.append(["價格先動那邊相關最大的", f"{kn:+d} 天：{neg[kn]:+.3f}", "是" if abs(neg[kn]) > band else "否"])
    rows_l.append(["力量差先動那邊相關最大的", f"{kp:+d} 天：{posk[kp]:+.3f}", "是" if abs(posk[kp]) > band else "否"])

    # ----- Test C -----
    base, f1 = s["ma50"], s["f1"]
    res = {"score": sd._score_ge(f1, base), "loss": f1["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK,
           "plc": r["placebo"] <= PLACEBO_BAR}
    res["pass"] = all(res.values())
    rows_x = []
    for k in ("ma50", "f1", "f2"):
        x = s[k]
        ck = x["checks"]
        rows_x.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                       f"{r['tim'][k] * 100:.0f}%", "—" if k == "ma50" else f"{r['differ'][k]:,} 天",
                       "—" if k != "f1" else f"{_share(r['placebo'])}（真的 {_share1(r['small']['low'])}，"
                                             f"錯位中位數 {_share1(r['placebo_med'])}）"])
    rows_pass = [["排名分數不輸 50 日均線", f"{_share1(f1['low'])} vs {_share1(base['low'])}", yes(res["score"])],
                 ["最深虧損中位數最多比均線深 5 點", f"{_pc(f1['c_deep_med'])} vs {_pc(base['c_deep_med'])}", yes(res["loss"])],
                 ["隨機錯位比例 ≤ 5%（同一個想法第四次試，比平常的 10% 嚴）", _share(r["placebo"]), yes(res["plc"])],
                 ["三條都要過", "", yes(res["pass"])]]
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
    cal = r["cal"]
    yrs = [y for y in sorted(cal["ma50"]) if y >= pd.Timestamp(FIRST_MONTH).year]
    rows_y = [[str(y), _p(cal["ma50"][y]), _p(cal["f1"][y]), _pt(cal["f1"][y] - cal["ma50"][y]),
               _p(cal["f2"][y]), _pt(cal["f2"][y] - cal["ma50"][y])] for y in yrs]
    gain_y = [y for y in yrs if cal["f1"][y] - cal["ma50"][y] > 0.0005]
    lose_y = [y for y in yrs if cal["f1"][y] - cal["ma50"][y] < -0.0005]

    tables = [
        {"caption": ("資料從哪天開始。買方力量＝從交易所提走的比特幣（美元）＋新增的穩定幣（美元）＋美國現貨 ETF 淨流入（美元）；"
                     "賣方力量＝送進交易所的比特幣（美元）；兩個都是近 30 天加總。每個來源某一天的數字，都等到隔一天的日線收盤才拿來用。"
                     "2019-07 以前沒有穩定幣資料，那幾年的買方力量只有「從交易所提走的比特幣」；2024-01-11 以前沒有 ETF（算 0）。"),
         "columns": ["項目", "數字"], "rows": rows0},
        {"caption": ("測試 A 的重點：全期 3×3 表，每格是「之後 30 天比特幣平均漲跌（天數）」。橫列是買方力量、直欄是賣方力量，"
                     "高／中／低＝在自己過去一年裡排前三分之一／中間／後三分之一。老闆的假設：同一欄往下（買方越強）應該越好，"
                     "同一列往右（賣方越強）應該越差，左下角（買方高、賣方低）最好，右上角（買方低、賣方高）最差。"),
         "columns": ["", "賣方低", "賣方中", "賣方高"], "rows": rows_m},
        {"caption": (f"測試 A 全部數字：每一期的九格，之後 30、60 天漲跌和 30 天內最深跌幅。90% 區間用 {BLOCK} 天一塊的區塊重抽 "
                     f"{BOOT:,} 次（相鄰日子彼此相關，所以整塊抽）。最後一段「{STABLE_ERA}」是拿掉沒有穩定幣資料那幾年之後的結果（只描述）。"),
         "columns": cols_a, "rows": rows_a},
        {"caption": ("測試 A 的三個數字（之後 30 天平均，點）。最好格 − 最差格＝買方高賣方低 減 買方低賣方高。"
                     "買方效果＝把賣方固定在同一級，買方高減買方低，三級平均。賣方效果＝把買方固定在同一級，賣方高減賣方低，三級平均"
                     "（這就是「送進交易所是不是賣壓」的直接答案：負的才是賣壓）。「順序對的列數」＝三列（或三欄）裡完全照假設的順序排的有幾個。"),
         "columns": ["期間", "最好格 − 最差格 ［90% 區間］", "買方效果 ［90% 區間］", "賣方效果 ［90% 區間］",
                     "(i) 買方越強越好：順序對的欄數", "(ii) 賣方越強越差：順序對的列數", "實際最好格｜最差格"],
         "rows": rows_eff},
        {"caption": "測試 A 的通過條件（跑之前定好，四條都要過）。",
         "columns": ["條件", "數字", "結果"], "rows": rows_ap},
        {"caption": ("賣方大的日子之後漲得好，是不是只因為同時買方也大？很多「送進」「提出」其實是交易所自己錢包之間搬來搬去，"
                     "或交易所之間互轉，會讓兩股力量一起變大，所以兩者高度相關。這張表看兩者有多綁在一起，以及只看一股力量會看到什麼。"),
         "columns": ["項目", "數字"], "rows": rows_c},
        {"caption": ("對照：力量差（買 − 賣）÷（買 + 賣）在過去一年排前三分之一的日子 減 後三分之一的日子。配對＝把日子依"
                     "「過去 30 天已經漲多少」分成十組，在同一組裡比，避免「其實只是價格已經漲過了」。權重是每組力量差高的天數。"),
         "columns": ["項目", "數字"], "rows": rows_ctl},
        {"caption": "測試 B：力量差當訊號。依力量差的排名分三組（低／中／高），之後的漲跌；最後一欄是「高 − 低」之後 30 天平均的差。",
         "columns": ["期間", "力量差", "天數", "之後 30 天平均 ［90% 區間］", "之後 30 天中位數 ［90% 區間］",
                     "之後 60 天平均 ［90% 區間］", "30 天內最深跌幅平均 ［90% 區間］", "高 − 低（30 天）"],
         "rows": rows_b},
        {"caption": (f"測試 B：誰先動？力量差過去 14 天的變化，和比特幣 14 天漲跌錯開不同天數的相關（{_d(ll['from'])} 起，"
                     f"{ll['n']:,} 天）。負的天數＝價格先動；正的天數＝力量差先動。雜訊帶＝把力量差錯開 {GAP} 天以上、{DRAWS} 次"
                     f"得到的 |相關| 第 95 百分位 {band:.3f}；超過才算看得出來。相隔不到 14 天的兩段時間有重疊，相關會被重疊拉高。"),
         "columns": ["錯開天數", "相關", "超過雜訊帶"], "rows": rows_l},
        {"caption": (f"測試 C：公平考試（main_signal_check 的做法）。比特幣日線起算月 {FIRST_MONTH[:7]} 到 "
                     f"{msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格"
                     f"（{base['c_starts']} 個起算月），一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個收盤時間 {SIMS} 次），其他資產 {'、'.join(ex.other)}（{base['o_n']} 個起算月；各用自己的 50、20 日均線，"
                     "配比特幣的力量差當全市場訊號）。50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"
                     f"持有時間＝{FIRST_MONTH[:7]} 起比特幣日線上持有的天數比例。隨機錯位：只把 F1 的「力量差高」那條往後挪 {GAP} 天以上"
                     f"（頭尾相接），{DRAWS} 次，價格條件不動，整個考試重算（一般人做法每個收盤時間只模擬 {PL_SIMS} 次，"
                     "真的 F1 也用同樣方法重算來比）。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "持有時間",
                     "和均線不同的天數", "隨機錯位比例"],
         "rows": rows_x},
        {"caption": "F1 的通過條件（跑之前定好，三條都要過）。F2 只描述：過濾只會少拿，不可能多賺，所以不考。",
         "columns": ["條件", "數字", "結果"], "rows": rows_pass},
        {"caption": ("只描述：F1 提早上車（均線還在空手時 F1 轉成持有）對不對。「對」＝30 天內均線也上車，而且那時價格比提早上車時高。"
                     "每次的結果＝從提早上車到 F1 下車或均線上車（先到的那個）之間的漲跌，扣掉買賣成本。"),
         "columns": ["結果", "次數", "平均漲跌", "接起來（相乘）"], "rows": rows_e},
        {"caption": f"只描述：每個日曆年的報酬（比特幣日線，每次買賣成本 {COST * 100:.2f}%），F1、F2 和 50 日均線比。",
         "columns": ["年", "50 日均線", "F1", "F1 − 均線", "F2", "F2 − 均線"], "rows": rows_y},
        {"caption": (f"只描述：rally_sources 的大漲段（從低點漲 {rs.RISE * 100:.0f}% 以上、跌回 {rs.FALL * 100:.0f}% 才算結束），"
                     f"每一段裡「買方排名｜賣方排名｜力量差排名」的平均（0–100，50 是一年裡的中間）：低點前 {PRE_TROUGH} 天、"
                     f"低點到 50 日均線上車、上車到高點、高點之後 {AFTER_PEAK} 天。如果力量差真的先轉，低點前應該已經偏高，高點附近應該先往下掉。"),
         "columns": ["大漲段", "漲幅", "均線上車日", f"低點前 {PRE_TROUGH} 天", "低點→均線上車", "均線上車→高點",
                     f"高點之後 {AFTER_PEAK} 天"],
         "rows": r["rally"]},
        {"caption": (f"只描述：大高點（創新高之後跌了 {TOP_FALL * 100:.0f}% 以上才再創新高）前後 {TOP_WIN} 天，"
                     "以及 2018、2022 兩年的下跌，「買方排名｜賣方排名｜力量差排名」的平均。"),
         "columns": ["事件", f"之前 {TOP_WIN} 天（2018、2022 為整年）", f"之後 {TOP_WIN} 天"], "rows": r["tops"]},
    ]

    # ----- verdict -----
    se_pos = _ok(full["se"]) and full["se"] > 0
    era = a["per"].get(STABLE_ERA)
    v1 = (f"兩股力量綁得很緊：排名相關 {a['corr_rank']:+.2f}，賣方高的日子有 {_share(a['sh_hi_hi'])} 買方也高（平常 "
          f"{_share(a['sh_buy_hi'])}）。測試 A（{_d(a['start'])} 起，{full['days']:,} 天）：最好格（買方高、賣方低）減最差格"
          f"（買方低、賣方高）之後 30 天 {_pt(full['bw'])} {_ci(*full['bw_ci'], f=_pt)}，三段裡 {ap['pos_subs']} 段是正的；"
          f"買方效果 {_pt(full['be'])} {_ci(*full['be_ci'], f=_pt)}，賣方效果（買方固定）{_pt(full['se'])} "
          f"{_ci(*full['se_ci'], f=_pt)}。事先定的條件{'通過' if ap['pass'] else '沒有通過'}"
          + ("" if ap["pass"] else "（" + "、".join(
              ([] if ap["bw"] else ["最好減最差不是明確為正"]) + ([] if ap["pos_subs"] >= 2 else ["三段裡為正的不到兩段"])
              + ([] if ap["be"] else ["買方效果不是正的"]) + ([] if ap["se"] else ["賣方效果不是負的"])) + "）") + "。")
    if se_pos:
        v1 += "就算把買方固定，賣方越大之後反而越好，所以這份資料在這個精細度上不支持「送進交易所＝賣壓」。"
    if era is not None and not _ok(era["bw"]):
        v1 += (f"只看穩定幣時期（{_d(era['span'][0])} 起）：買方高、賣方低的那一格沒有日子（兩股力量幾乎總是一起高、一起低），"
               "三個數字都算不出來。")
    elif era is not None:
        v1 += (f"只看穩定幣時期（{_d(era['span'][0])} 起）：最好減最差 {_pt(era['bw'])}、買方效果 {_pt(era['be'])}、"
               f"賣方效果 {_pt(era['se'])}。")
    v2 = (f"測試 B：力量差高減低之後 30 天 {_pt(bf['hl'])} {_ci(*bf['hl_ci'], f=_pt)}，配對過去 30 天漲幅後 {_pt(mt30[0])} "
          f"{_ci(*mt30[1], f=_pt)}；誰先動：價格先動那邊最大 {kn:+d} 天 {neg[kn]:+.3f}、力量差先動那邊最大 {kp:+d} 天 "
          f"{posk[kp]:+.3f}（雜訊帶 {band:.3f}）。")
    fails = ([] if res["score"] else ["排名分數輸均線"]) + ([] if res["loss"] else ["最深虧損比均線深超過 5 點"]) \
        + ([] if res["plc"] else ["隨機錯位超過 5%"])
    v3 = (f"測試 C：F1 排名分數 {_share1(f1['low'])}（均線 {_share1(base['low'])}），最深虧損中位數 {_pc(f1['c_deep_med'])}"
          f"（均線 {_pc(base['c_deep_med'])}），隨機錯位 {_share(r['placebo'])}，"
          + ("三條都通過，只能算是可以放進每日影子紀錄觀察的候選。" if res["pass"] else f"沒有通過（{'、'.join(fails)}）。")
          + f"F1 提早上車 {len(ent)} 次，對 {len(right)} 次、錯 {len(wrong)} 次（錯的平均 {_p(avg(wrong))}）。"
          f"F2 排名分數 {_share1(s['f2']['low'])}（只描述）。")
    verdict = v1 + v2 + v3

    lk_total, lk_bad = r["look"]
    notes = [
        "為什麼有這份研究：reluctant_sellers 發現，站上 20 日均線時，大量比特幣送進交易所的日子之後 30 天反而最好"
        "（+25.5%，送得少的只有 +4.6%）。一種讀法是「送進交易所不是賣壓」。老闆的讀法比較好：送進去可能真的是賣壓，"
        "但那些日子買方力量更大；只看一股力量沒有意義，要兩股一起量、一起比。這份研究就是這樣做。",
        "名詞：買方力量＝從交易所提走的比特幣（買了帶走）＋新增的穩定幣（準備拿來買的美元）＋美國現貨 ETF 淨流入（ETF 申購是"
        "不經過交易所存款的買盤；贖回時是負的），都換成美元、近 30 天加總。賣方力量＝送進交易所的比特幣（美元、近 30 天）。"
        "排名＝今天的數字比過去一年裡多少比例的日子高（用排名是因為美元規模一直在長大）。力量差＝（買 − 賣）÷（買 + 賣），"
        "介於 −1 和 +1 之間。",
        "和之前的研究差在哪：之前每份只看一股力量，或用不同單位。這份把兩股力量換成同一個單位（每 30 天多少美元），各自和自己的過去比，"
        "再交叉成 3×3 表，所以可以問「在買方一樣強的日子裡，賣方大是不是比較差」。",
        "之前相關結果（照實寫）：exchange_reserves：交易所餘額減少之後的報酬和論文說的相反，規則輸 50 日均線。"
        "rally_sources：交易所淨流出等管道在大漲前「先亮」的次數，和把訊號隨機挪動時間的假訊號差不多。"
        "stablecoin_supply：穩定幣供給比較像跟著價格走，不是帶動價格。order_flow：主動買進比例沒有可用的領先關係。"
        "price_impact：用價格反應量「賣方稀少」沒過。supply_demand：「賣壓少＋買盤來」沒過。reluctant_sellers：如上，結果和「惜售」想法相反。",
        "這是「供需」這個想法第四次試，所以 F1 的隨機錯位門檻是 5%，比平常的 10% 嚴。",
        "老實說的限制：交易所流量是用今天認得的交易所地址套回過去，越早越低估；很多「送進」「提出」其實是交易所自己錢包之間搬、"
        "或交易所之間互轉，兩股力量會一起被灌大——這正是兩者高度相關的原因，也是為什麼要看力量差，而不是單看哪一股的大小。"
        "場外交易（OTC）看不到；2024 年以前不經過交易所的買盤也看不到。新增的穩定幣不全是拿來買比特幣。",
        "買方力量的組成在 2019-07（加入穩定幣）和 2024-01（加入 ETF）各變一次，之後一年內它的排名會被墊高一點；"
        "所以另外列了「穩定幣時期」的結果。",
        "Coin Metrics 的 USDT、USDC 交易所流入（穩定幣送進交易所＝美元到場準備買）不是免費資料（回 400／403），"
        "原本想做的變體 BUY2 沒辦法算，已拿掉，沒有用別的來源代替。",
        "門檻（30 天、365 天、三分之一、20 日均線、60 天的區塊）都是跑之前憑道理定的，沒有試過其他數字；定義寫好之後才跑，看到結果也沒改。",
        "其他幣和 MSTR 沒有自己的流量資料，用的是比特幣的力量差（當成全市場的狀態），配各自的價格均線；MSTR 用前一天的訊號。",
        f"偷看未來的檢查：把所有資料在 {'、'.join(CUTS)} 截斷（來源再早一天）重算，{lk_total} 次比對裡有 {lk_bad} 次不一致"
        f"{'（應該是 0）' if lk_bad else '，截斷前的高中低分級和 F1 持有完全一樣'}。",
        "就算通過，也只代表 F1 可以放進每日影子紀錄觀察；用的是同一段歷史，真正的考驗是之後每天實際記下的結果。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        (f"F1 比均線多賺的年份：{'、'.join(str(y) for y in gain_y) or '沒有'}；少賺的年份：{'、'.join(str(y) for y in lose_y) or '沒有'}。"),
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載；考試本身 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    notes += problems

    return {
        "id": "force_balance",
        "title": "買方力量和賣方力量一起量：誰比較大",
        "question": ("把「送進交易所的比特幣」（賣方力量）和「從交易所提走的比特幣＋新增穩定幣＋ETF 淨流入」（買方力量）換成同樣的美元單位、"
                     "各自和過去一年比，再一起看：買方強、賣方弱的時候之後是不是比較會漲？在買方一樣強時，賣方大是不是比較差？"
                     "兩者的差能不能讓 50 日均線提早上車？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群 API：比特幣 FlowInExUSD、FlowOutExUSD、FlowInExNtv、FlowOutExNtv（送進／提出交易所）",
             "url": CM_URL + "?assets=btc&metrics=" + ",".join(CM_METRICS) + "&frequency=1d"},
            {"name": "DefiLlama 穩定幣總供給 API", "url": LLAMA},
            {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": FARSIDE_URL},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
