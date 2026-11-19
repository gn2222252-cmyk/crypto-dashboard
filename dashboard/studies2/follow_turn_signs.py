"""Historical study "follow_turn_signs": step 2 of the owner's plan. Step 1 (signs_before_turns) observed which series
sat at an extreme before the 8 bear starts and 8 bull starts that have data. This study asks: if a person FOLLOWS
those signs, does exiting lose less, and does entering early make money?

The central difficulty: step 1 chose the signs by looking at ALL the turns, and there are only about 8 of each. A sign
set chosen that way will always "see" the turns it was chosen on. So the main method is leave-one-turn-out, and the
honest candidate is a strictly forward version that only uses turns already confirmed at the time.

Definitions (written before any result of this study was seen; anything added after seeing results is labelled so).

Data, ranks and turns (all reused from signs_before_turns, imported, not copied):
  * Daily close te._daily_close; the 43 series of signs_before_turns._assemble (each lagged to publication); its
    downloads go to this study's own store cache/_store/follow_turn_signs/ (seeded read-only from the shared store and
    from signs_before_turns' store).
  * Rank = signs_before_turns' rank (share of the trailing 730 values <= today, >= 365 values, 0-100).
    M30(t) = mean rank over days t-29..t (>= 20 values), known at close t. LEV(T) = M30(T-1), the 30-day level before a
    turn exactly as step 1 measured it.
  * Cycle turns = signs_before_turns._turns: bear start (fall scan 50% / 100%), bull start (rise scan 100% / 50%).
    Only turns where at least one series has LEV are used ("turns with data": 8 bear, 8 bull).
  * Confirmation: a bear start at close p is confirmed on the first later close <= 0.5 x p; a bull start at low p on
    the first later close >= 2 x p (the scan has then fixed the turn). FORWARD uses confirmation dates only.

Choosing six signs from a set of turns U of one type (the recipe of signs_before_turns with the owner's ranking):
  * Random-day pool: days at least 605 days after the price history starts and at least 90 days away from every
    cycle turn (bear and bull) in the reference set, up to a cut-off (see below). Expectation p_top / p_bot = share of
    pool days whose LEV is >= 80 / <= 20 (all pool days with a value, >= 100 needed; step 1 used 2,000 draws of the
    same pool - same expectation, less noise).
  * A series qualifies with LEV at >= NREQ of the turns in U (NREQ = 5; FORWARD: min(5, |U|)). Its side = the fifth
    (top >= 80 / bottom <= 20) it was in more often across U (ties: larger excess). Excess = k/N - p_side.
  * The six = the qualifying series with the largest excess (ties: larger N, then key).
Three versions of "chosen":
  * ALL: U = all turns of the type, pool = all cycle turns excluded, no cut-off. In-sample; reference only.
  * LOTO (leave one turn out): for turn j, U = the other turns, pool excludes the other cycle turns only (not j).
    Daily: on day t, the set of the nearest turn of that type within +-365 days; if none is that near, the ALL set.
    LOTO still uses later turns to choose signs; only FORWARD does not.
  * FORWARD: U = the turns of that type confirmed on or before day t (at least 3), pool = days up to the latest of
    those confirmations, excluding days within 90 days of cycle turns (either type) confirmed by then. Re-chosen at
    each confirmation of that type. Undefined before the third confirmation.
  * (Report-only robustness for Part 1, fixed now) LCO, leave a cluster out: like LOTO but every turn of that type
    within 365 days of turn j is also left out (April 2013 has two bear starts 15 days apart).
Daily counts: BEAR_COUNT(t) = number of the six bear signs whose M30(t) is in their side's fifth; BULL_COUNT likewise;
  a sign with no value counts as absent. Undefined (NaN) only where no set exists (FORWARD before 3 confirmations).

Part 1 - did the count see the held-out turns? For each cycle turn T (index of the turn close): with the LOTO set of
  that turn (and the FORWARD daily count; and LCO as one tally line): the count at T, its maximum over T-90..T, the
  first day in T-90..T it reached 4 and 5 (days before T, or "never"). Tally k of N turns reaching 4+ / 5+. Non-turn
  share: days at least 90 days away from every turn of that type, share with the daily count >= 4 / >= 5. False alarms
  on the daily count: an episode = a day with count >= L after at least 90 consecutive defined days below L; followed
  = a turn of that type dated within 0..180 days after the episode day; episodes in the last 180 days = unresolved.

Part 2 - following it. ma = ma50 (main_signal_check: close above its 50-day average), A20 = close above its 20-day
  average (supply_demand._above20). Rules, each with LOTO and FORWARD counts (ALL as reference):
    XB  = ma AND NOT (BEAR_COUNT >= 5);       XB4 = ma AND NOT (BEAR_COUNT >= 4)
    EB  = ma OR (BULL_COUNT >= 4 AND A20)
    SB  = ma x (1 if BEAR_COUNT <= 2, 0.5 if 3-4, 0 if >= 5)
    XE  = (ma OR (BULL_COUNT >= 4 AND A20)) AND NOT (BEAR_COUNT >= 5)   [bear signs override]
  Undefined counts act as 0 (the rule is then ma50). Fair exam = price_impact._Exam (main_signal_check's four checks:
  BTC daily monthly starts 2015-01 .. 2023-12 at 0.10%; 72 starts 2018-01 .. 2023-12 x 24 closing hours; the same as an
  ordinary person, 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock; ETH SOL BNB XRP MSTR
  each with its own ma50 / A20 and BTC's counts as a market-wide state; a clock close on UTC date E and MSTR's date D
  use the counts of E-1 / D-1). ma50 re-scored on the same starts. Ranking score = worst of the four "beats holding"
  shares with main_signal_check's 95% loss eligibility (supply_demand._score_ge to compare). Median / worst deepest
  loss = over the clock textbook cells. Time in market = mean BTC daily position from 2015-01-01. Calendar years =
  BTC daily (0.10%) rule return minus ma50's.
  Placebo (LOTO and FORWARD): BEAR_COUNT and BULL_COUNT together circularly shifted by k (uniform integer in
  [365, span - 365], span = days from the first defined count) against price, 200 draws (fixed seed); exam rebuilt
  with 20 ordinary-person simulations per clock and the real rule re-scored the same way; placebo share = share of
  draws scoring at least the real one (_score_ge). p = (1 + hits) / (1 + draws) for Holm.
  "Supported" (FORWARD only): XB, XB4, SB - ranking >= ma50's AND median deepest loss shallower than ma50's AND
  placebo <= 5%; EB - ranking >= ma50's AND median deepest loss no more than 5 points worse AND placebo <= 5%; XE -
  ranking >= ma50's AND median deepest loss no worse AND placebo <= 5%. Holm at 5% across the five FORWARD rules is
  reported next to it (expected passes by luck = 5 x 5%).
  Plain answers (BTC daily, UTC 00:00, no costs, from the first defined count):
    Exit: a stand-aside spell = a run of days with BEAR_COUNT >= 5 that contains at least one day ma50 was on. Per
      spell: dates, ma50-on days skipped, ma50's return over those skipped days (negative = fall avoided, positive =
      rise missed), price change from the first spell close to the close after it, lowest close inside. Averages and
      net = XB's equity / ma50's equity - 1 over all spells. Per cycle top: max BEAR_COUNT in T-90..T, XB's first
      stand-aside day in T-90..T+60 (days from T, price versus the top close), ma50's first exit at or after T-90
      (days, price), XB minus ma50 return over T-180..T+180. 2018 and 2022 from the calendar-year table.
    Entry: an early entry = a run of days with EB on and ma50 off; from its first close to the close it ends (ma50
      takes over / close below the 20-day average / BULL_COUNT below 4). Per cycle bottom B: ma50's first on-day at or
      after B (price versus the bottom), EB's first early entry in B-90 .. that day, and the early entries in that span.
Part 3 - the tops the signs missed. Groups fixed from step 1: well flagged = 2013-04-09, 2013-04-24, 2013-12-04,
  2017-12-16, 2021-04-13; missed = 2019-06-26, 2021-11-08, 2025-10-06. LEV of every series (plus, if loadable,
  pressure_before_moves' composite ranks SELL_C / BUY_C and NET30, given the same 30-day mean, lagged 1) at each top.
  Lists: series in the same fifth at >= 2 of the 3 missed tops and at <= 1 of the flagged tops with data; series in the
  same fifth at >= half (and >= 2) of the flagged tops with data and at <= 1 of the missed. Median rank difference
  (missed - flagged) for series with data at all 3 missed and >= 3 flagged; the 12 largest absolute differences shown.
  Any reading is a guess (推測).
Part 4 - loose financial conditions before tops versus the NFCI gate. Daily, from where both are defined: NFCI M30 <= 20
  (bottom fifth, looser than usual for 2 years) and valuation M30 >= 80 (MVRV; second table: distance above the 200-day
  average). Groups: both; NFCI low only; valuation high only; neither. Also nfci_filter's gate, done daily: NFCI known
  at t below the mean of the 364 previous daily values (= its last 52 weekly values) ("easier than its own average"),
  on / off, and on AND valuation high. Outcomes: 90- and 180-day forward return, deepest close in t+1..t+180 versus t,
  share of days with a deepest drop beyond -30%. Mean with a 90% circular block bootstrap interval (block 180 days,
  1,000 draws, fixed seed); per period 2011-2016, 2017-2020, 2021-now; number of separate stretches (new stretch after
  30 days out of the group).
Added after the first run (presentation only, no definition or threshold changed; labelled where shown): in the per-top
  table a column "ma50's first exit at or after T" (the "at or after T-90" column often caught a brief exit long before
  the top); a table listing the Part 3 groups; a one-line lead to the verdict; plain wording of the exit averages.
Today: "比特幣：50 日均線｜XB｜EB" with FORWARD states, the FORWARD and LOTO counts and which signs are on. Description,
  not a forecast.
Limits stated in the notes: about 8 turns of each type; April 2013 counted as two bear starts and one bull start; most
signs are price / valuation measures that move together, so six signs are far fewer than six independent clues; several
series are revised or use today's version; LOTO still uses the future to choose signs; FORWARD has very few early turns
to learn from (its first bear set is chosen from three turns, two of them 15 days apart). Not advice.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa
from dashboard.studies2 import attention as at

ID = "follow_turn_signs"
DAY = pd.Timedelta(days=1)
TOP, BOT = sbt.TOP, sbt.BOT
SIX, NREQ, MIN_FWD = 6, 5, 3
START = sbt.WIN_MIN + 240
AWAY = 90
LOOK, NEAR, QUIET, FOLLOW = 90, 365, 90, 180
LEVELS = (4, 5)
DRAWS, PL_SIMS, SHIFT_MIN = 200, 20, 365
SEED = 20261120
COST, SIMS, P_COST, P_MISS = msc.COST, msc.SIMS, msc.P_COST, msc.P_MISS
FIRST = pd.Timestamp(msc.DAILY_STARTS[0])
ALPHA, LOSS_SLACK = 0.05, 5.0
BOOT, BLOCK = 1000, 180
PERIODS = (("2011–2016", "2011-01-01", "2016-12-31"), ("2017–2020", "2017-01-01", "2020-12-31"),
           ("2021–現在", "2021-01-01", "2100-01-01"))
RULES = ("xb", "xb4", "eb", "sb", "xe")
VERS = ("fwd", "loto", "all")
VNAME = {"fwd": "往前（只用當時已確認的轉折）", "loto": "留一（拿掉要考的那次轉折）", "all": "全部（樣本內，只當參考）"}
VSHORT = {"fwd": "往前", "loto": "留一", "all": "全部（參考）"}
RNAME = {"xb": "XB：50 日均線持有中，但空頭徵兆 ≥5 個就出場",
         "xb4": "XB4：同 XB，但 ≥4 個就出場（穩健度）",
         "eb": "EB：50 日均線，或多頭徵兆 ≥4 個而且站上 20 日均線就提早進場",
         "sb": "SB：50 日均線持有中，空頭徵兆 ≤2 個抱 100%、3–4 個抱 50%、≥5 個抱 0%",
         "xe": "XE：XB 和 EB 一起用（空頭徵兆優先）"}
RSHORT = {"xb": "XB", "xb4": "XB4", "eb": "EB", "sb": "SB", "xe": "XE", "ma50": "50 日均線"}
KI = {"ma50": 1, "xb": 140, "xb4": 141, "eb": 142, "sb": 143, "xe": 144}
FLAGGED = ("2013-04-09", "2013-04-24", "2013-12-04", "2017-12-16", "2021-04-13")
MISSED = ("2019-06-26", "2021-11-08", "2025-10-06")
TOPS_NAMED = ("2013-04", "2013-12", "2017-12", "2019-06", "2021-04", "2021-11", "2025-10")
BOTTOMS_NAMED = ("2015-01", "2018-12", "2020-03", "2021-07", "2022-11")
PRESS_KEYS = ("SELL_C", "BUY_C", "NET30")
PRESS_META = {"SELL_C": ("賣方合計（壓力）", "送進交易所、老幣移動、主動賣、賣出壓價力的排名平均"),
              "BUY_C": ("買方合計（壓力）", "提走、新增穩定幣、ETF 流入、主動買、推價力、Coinbase 溢價的排名平均"),
              "NET30": ("淨提走比例（30 天）", "（從交易所提走−送進）÷（提走＋送進）")}

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID

_d = te._d
_ok = sbt._ok


# ---------- small helpers ----------

def _pc(v, nd: int = 0) -> str:
    return f"{v * 100:+.{nd}f}%" if _ok(v) else "—"


def _pp(v, nd: int = 1) -> str:
    return f"{v:+.{nd}f}" if _ok(v) else "—"


def _sh(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


def _px(v) -> str:
    return "—" if not _ok(v) else (f"{v:,.2f}" if v < 100 else f"{v:,.0f}")


def _name(k: str) -> str:
    if k in sbt.META:
        return sbt.META[k][0]
    return PRESS_META.get(k, (k,))[0]


def _sname(sig: dict) -> str:
    return f"{_name(sig['key'])}{'高' if sig['side'] > 0 else '低'}"


# ---------- own store ----------

@contextmanager
def _store():
    """Every loader writes into cache/_store/follow_turn_signs/; the shared store and signs_before_turns' store are
    only read (copied in when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for src_dir in (REAL_STORE / sbt.ID,):
        if src_dir.exists():
            for f in src_dir.glob("*.json"):
                dst = MY_STORE / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(f, dst)
                except Exception:  # noqa: BLE001 - a missing seed only means a download
                    pass
    old = sbt.MY_STORE
    sbt.MY_STORE = MY_STORE
    try:
        with sbt._own_store():
            yield
    finally:
        sbt.MY_STORE = old


def _pressure(close: pd.Series) -> tuple[dict, list[str], dict]:
    """pressure_before_moves' composite ranks (0-100), its stores redirected into this study's folder."""
    try:
        from dashboard.studies2 import pressure_before_moves as pbm
    except Exception as exc:  # noqa: BLE001
        return {}, [f"買賣壓力合計指標載入失敗（{type(exc).__name__}），第 3 部分不含它們。"], {}
    names = ("STORE_CM", "STORE_LL", "STORE_OBM")
    saved = {a: getattr(pbm, a) for a in names}
    try:
        for a, p in saved.items():
            dst = MY_STORE / p.name
            try:
                if p.exists() and (not dst.exists() or p.stat().st_mtime > dst.stat().st_mtime):
                    shutil.copy2(p, dst)
            except Exception:  # noqa: BLE001
                pass
            setattr(pbm, a, dst)
        data = pbm._load()
        _raw, rank, _first = pbm._pressures(data)
        out = {k: rank[k].reindex(close.index).astype(float).values for k in PRESS_KEYS if k in rank}
        src = {"Coin Metrics 社群 API（FlowInExNtv、FlowOutExNtv，買賣壓力）":
               pbm.fb.CM_URL + "?assets=btc&metrics=" + ",".join(pbm.fb.CM_METRICS) + "&frequency=1d",
               "Open Bitcoin Metrics（放 155 天以上的幣被花掉）": pbm.ocs.OBM_BASE.format(pbm.OBM_ID),
               "幣安現貨 BTCUSDT 1 小時線（主動買賣、推壓價力）": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h",
               "Coinbase Exchange 公開日線 BTC-USD": sd.CB_URL + "?granularity=86400",
               "Bitstamp 公開 API：BTC/USD 日線": sd.BS_URL + "?step=86400&limit=1000",
               "Farside 美國現貨比特幣 ETF 每日淨流入（買賣壓力）": pbm.fb.FARSIDE_URL}
        return out, list(data.get("problems") or []), src
    except Exception as exc:  # noqa: BLE001 - optional, reported
        return {}, [f"買賣壓力合計指標這次算不出來（{type(exc).__name__}），第 3 部分不含它們。"], {}
    finally:
        for a, p in saved.items():
            setattr(pbm, a, p)


# ---------- turns, confirmation, choosing signs ----------

def _confirm(c: np.ndarray, t: dict, ty: str) -> int | None:
    i, p = t["i"], t["p"]
    after = c[i + 1:]
    hit = np.flatnonzero(after <= 0.5 * p) if ty == "bear" else np.flatnonzero(after >= 2.0 * p)
    return int(i + 1 + hit[0]) if len(hit) else None


def _pool(n: int, excl: list[int], upto: int | None = None) -> np.ndarray:
    ok = np.zeros(n, bool)
    ok[START:(n if upto is None else min(n, upto + 1))] = True
    for t in excl:
        ok[max(0, t - AWAY + 1):min(n, t + AWAY)] = False
    return np.flatnonzero(ok)


def _choose(lev: dict, U: list[int], pool: np.ndarray, nreq: int) -> list[dict]:
    items = []
    for k in sorted(lev):
        a = lev[k]
        v = a[U]
        v = v[np.isfinite(v)]
        N = len(v)
        if N < nreq or N == 0:
            continue
        rv = a[pool]
        rv = rv[np.isfinite(rv)]
        if len(rv) < 100:
            continue
        top, bot = int((v >= TOP).sum()), int((v <= BOT).sum())
        pt, pb = float((rv >= TOP).mean()), float((rv <= BOT).mean())
        et, eb = top / N - pt, bot / N - pb
        if top > bot or (top == bot and et >= eb):
            side, kk, p, ex = 1, top, pt, et
        else:
            side, kk, p, ex = -1, bot, pb, eb
        items.append({"key": k, "side": side, "k": kk, "N": N, "p": p, "ex": ex})
    items.sort(key=lambda x: (-x["ex"], -x["N"], x["key"]))
    return items[:SIX]


def _count(m30: dict, sset: list[dict], n: int) -> np.ndarray:
    out = np.zeros(n)
    for s in sset:
        a = m30[s["key"]]
        out += (a >= TOP) if s["side"] > 0 else (a <= BOT)  # NaN compares False -> absent
    return out


def _on(m30: dict, sset: list[dict], i: int) -> list[str]:
    on = []
    for s in sset:
        v = m30[s["key"]][i]
        if _ok(v) and ((s["side"] > 0 and v >= TOP) or (s["side"] < 0 and v <= BOT)):
            on.append(_sname(s))
    return on


def _sets(close: pd.Series, lev: dict, m30: dict, turns: dict) -> dict:
    n = len(close)
    c = close.values.astype(float)
    cyc = [t["i"] for ty in ("bear", "bull") for t in turns[ty]]
    S = {}
    for ty in ("bear", "bull"):
        tl = [t for t in turns[ty] if any(_ok(lev[k][t["i"]]) for k in lev)]
        ti = [t["i"] for t in tl]
        conf = [_confirm(c, t, ty) for t in tl]
        all_set = _choose(lev, ti, _pool(n, cyc), NREQ)
        loto, lco = [], []
        for j, t in enumerate(tl):
            U = [x for x in ti if x != t["i"]]
            loto.append(_choose(lev, U, _pool(n, [x for x in cyc if x != t["i"]]), NREQ))
            near = [x for x in ti if abs(x - t["i"]) <= NEAR]
            U2 = [x for x in ti if x not in near]
            lco.append(_choose(lev, U2, _pool(n, [x for x in cyc if x not in near]), NREQ))
        # forward: sets after each confirmation (in confirmation order)
        order = sorted([(cf, j) for j, cf in enumerate(conf) if cf is not None])
        all_conf = []
        for ty2 in ("bear", "bull"):
            for t in turns[ty2]:
                cf = _confirm(c, t, ty2)
                if cf is not None:
                    all_conf.append((cf, t["i"]))
        fwd = []  # (from index, set, turns used)
        for m in range(MIN_FWD, len(order) + 1):
            cf = order[m - 1][0]
            U = [ti[j] for _, j in order[:m]]
            known = [i for cfx, i in all_conf if cfx <= cf]
            fwd.append((cf, _choose(lev, U, _pool(n, known, cf), min(NREQ, m)), [tl[j] for _, j in order[:m]]))
        # daily series
        cnt_all = _count(m30, all_set, n)
        cnt_loto_each = [_count(m30, s, n) for s in loto]
        cnt_lco_each = [_count(m30, s, n) for s in lco]
        loto_daily = cnt_all.copy()
        which = np.full(n, -1)
        idx = np.arange(n)
        if ti:
            dist = np.abs(idx[:, None] - np.array(ti)[None, :])
            nearest = dist.argmin(axis=1)
            close_enough = dist.min(axis=1) <= NEAR
            which = np.where(close_enough, nearest, -1)
            for j in range(len(ti)):
                m = which == j
                loto_daily[m] = cnt_loto_each[j][m]
        fwd_daily = np.full(n, np.nan)
        fwd_which = np.full(n, -1)
        for q, (cf, sset, _) in enumerate(fwd):
            fwd_daily[cf:] = _count(m30, sset, n)[cf:]
            fwd_which[cf:] = q
        S[ty] = {"turns": tl, "ti": ti, "conf": conf, "all": all_set, "loto": loto, "lco": lco, "fwd": fwd,
                 "cnt": {"all": cnt_all, "loto": loto_daily, "fwd": fwd_daily},
                 "loto_each": cnt_loto_each, "lco_each": cnt_lco_each, "which": which, "fwd_which": fwd_which}
    return S


# ---------- Part 1 ----------

def _per_turn(cnt: np.ndarray, T: int) -> dict:
    a = cnt[max(0, T - LOOK):T + 1]
    if not np.isfinite(a).any():
        return {"ok": False}
    out = {"ok": True, "at": cnt[T], "max": float(np.nanmax(a))}
    for L in LEVELS:
        hit = np.flatnonzero(np.nan_to_num(a, nan=-1) >= L)
        out[L] = int(len(a) - 1 - hit[0]) if len(hit) else None
    return out


def _episodes(cnt: np.ndarray, L: int, turn_idx: list[int], n: int) -> dict:
    defined = np.isfinite(cnt)
    above = np.nan_to_num(cnt, nan=-1) >= L
    eps = []
    below_run = 0
    for i in range(n):
        if not defined[i]:
            below_run = 0
            continue
        if above[i]:
            if below_run >= QUIET:
                eps.append(i)
            below_run = 0
        else:
            below_run += 1
    tset = np.array(sorted(turn_idx))
    fol, unres = 0, 0
    for e in eps:
        if e + FOLLOW >= n:
            unres += 1
            continue
        if len(tset) and ((tset >= e) & (tset <= e + FOLLOW)).any():
            fol += 1
    return {"n": len(eps), "fol": fol, "unres": unres, "days": eps}


def _nonturn_share(cnt: np.ndarray, turn_idx: list[int], n: int) -> dict:
    ok = np.isfinite(cnt)
    for t in turn_idx:
        ok[max(0, t - AWAY + 1):min(n, t + AWAY)] = False
    v = cnt[ok]
    return {L: (float((v >= L).mean()) if len(v) else None) for L in LEVELS} | {"days": int(len(v))}


# ---------- rules and exam ----------

def _rule(rule: str, ma, a20, B, U) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float))
    a20 = np.nan_to_num(np.asarray(a20, dtype=float))
    B = np.nan_to_num(np.asarray(B, dtype=float), nan=0.0)
    U = np.nan_to_num(np.asarray(U, dtype=float), nan=0.0)
    on = ma > 0.5
    early = (U >= 4) & (a20 > 0.5)
    if rule == "ma50":
        return on.astype(float)
    if rule == "xb":
        return (on & (B < 5)).astype(float)
    if rule == "xb4":
        return (on & (B < 4)).astype(float)
    if rule == "eb":
        return (on | early).astype(float)
    if rule == "sb":
        return on * np.where(B >= 5, 0.0, np.where(B >= 3, 0.5, 1.0))
    if rule == "xe":
        return ((on | early) & (B < 5)).astype(float)
    raise ValueError(rule)


class _Exam(pi._Exam):
    """main_signal_check's four checks for ma50 and the five sign rules; BTC's daily counts are the market state."""

    @staticmethod
    def _at(s: pd.Series, dates, lag: int) -> np.ndarray:
        return s.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)

    def btc_pos(self, rule, B: pd.Series, U: pd.Series) -> np.ndarray:
        return _rule(rule, self.ma, self.a20, self._at(B, self.close.index, 0), self._at(U, self.close.index, 0))

    def summary(self, rule: str, B: pd.Series, U: pd.Series, ki: int, sims: int = SIMS) -> dict:
        pos = self.btc_pos(rule, B, U)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, msc.YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        b1, u1 = self._at(B, self.m.dates, 1), self._at(U, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = _rule(rule, self.cma[h], self.ca20[h], b1, u1)
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
            w = _rule(rule, o["ma"], o["a20"], self._at(B, o["close"].index, o["lag"]),
                      self._at(U, o["close"].index, o["lag"]))
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        m = self.close.index >= FIRST
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values()),
             "time": float(pos[m].mean())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


def _roll(s: pd.Series, k: int, start: int) -> pd.Series:
    v = s.values.copy()
    v[start:] = np.roll(v[start:], k)
    return pd.Series(v, index=s.index)


def _supported(rule: str, x: dict, base: dict, pl: float | None) -> dict:
    rank_ok = sd._score_ge(x, base)
    if rule in ("xb", "xb4", "sb"):
        loss_ok = x["c_deep_med"] > base["c_deep_med"] + 1e-9
    elif rule == "eb":
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK
    else:
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - 1e-9
    pl_ok = pl is not None and pl <= ALPHA
    return {"rank": rank_ok, "loss": loss_ok, "pl": pl_ok, "ok": rank_ok and loss_ok and pl_ok}


def _holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    out, stop, m = {}, False, len(order)
    for i, k in enumerate(order):
        if not stop and ps[k] <= ALPHA / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out


# ---------- plain answers ----------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _spells(c: np.ndarray, ma: np.ndarray, B: np.ndarray) -> list[dict]:
    n = len(c)
    r1 = np.concatenate([c[1:] / c[:-1] - 1, [np.nan]])
    on = np.nan_to_num(ma) > 0.5
    out = []
    for s, e in _runs(np.nan_to_num(B, nan=0.0) >= 5):
        days = [t for t in range(s, e + 1) if on[t] and t + 1 < n]
        if not any(on[s:e + 1]):
            continue
        miss = float(np.prod(1 + r1[days]) - 1) if days else 0.0
        endc = min(e + 1, n - 1)
        out.append({"s": s, "e": e, "skip": len(days), "miss": miss, "chg": c[endc] / c[s] - 1,
                    "low": float(c[s:endc + 1].min() / c[s] - 1), "open": e + 1 >= n})
    return out


def _early(c: np.ndarray, ma: np.ndarray, a20: np.ndarray, U: np.ndarray) -> list[dict]:
    n = len(c)
    on = np.nan_to_num(ma) > 0.5
    a = np.nan_to_num(a20) > 0.5
    u = np.nan_to_num(U, nan=0.0) >= 4
    eb = u & a & ~on
    out = []
    for s, e in _runs(eb):
        x = e + 1
        if x >= n:
            out.append({"s": s, "x": n - 1, "gain": c[-1] / c[s] - 1, "why": "仍在進行", "open": True})
            continue
        why = "50 日均線接手" if on[x] else ("跌回 20 日均線下" if not a[x] else "多頭徵兆少於 4 個")
        out.append({"s": s, "x": x, "gain": c[x] / c[s] - 1, "why": why, "open": False})
    return out


def _eq(pos: np.ndarray, c: np.ndarray, a: int, b: int) -> float:
    r1 = np.concatenate([c[1:] / c[:-1] - 1, [0.0]])
    a, b = max(0, a), min(len(c) - 1, b)
    return float(np.prod(1 + pos[a:b] * r1[a:b]))


# ---------- Part 4 ----------

def _forward(c: np.ndarray) -> dict:
    n = len(c)
    f90, f180, dd = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    f90[:n - 90] = c[90:] / c[:n - 90] - 1
    f180[:n - 180] = c[180:] / c[:n - 180] - 1
    s = pd.Series(c[::-1])
    fut_min = s.rolling(180, min_periods=180).min().values[::-1]  # min of c[t..t+179]
    nxt = np.concatenate([fut_min[1:], [np.nan]])  # min of c[t+1..t+180]
    dd[:n - 180] = nxt[:n - 180] / c[:n - 180] - 1
    return {"f90": f90, "f180": f180, "dd": dd}


def _boot(vals: dict, masks: dict, valid: np.ndarray, seed: int) -> dict:
    """Circular block bootstrap over the days in `valid`: 90% interval of each group's mean for each outcome."""
    pos = np.flatnonzero(valid)
    m = len(pos)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(m / BLOCK))
    out = {g: {k: [] for k in vals} for g in masks}
    for _ in range(BOOT):
        st = rng.integers(0, m, nb)
        ix = ((st[:, None] + np.arange(BLOCK)[None, :]) % m).ravel()[:m]
        days = pos[ix]
        for g, mk in masks.items():
            sel = days[mk[days]]
            for k, v in vals.items():
                out[g][k].append(float(np.mean(v[sel])) if len(sel) else np.nan)
    return {g: {k: (float(np.nanpercentile(a, 5)), float(np.nanpercentile(a, 95))) for k, a in d.items()}
            for g, d in out.items()}


def _stretches(mask: np.ndarray) -> int:
    n_, gap = 0, 10 ** 9
    for v in mask:
        if v:
            if gap > 30:
                n_ += 1
            gap = 0
        else:
            gap += 1
    return n_


def _part4(close: pd.Series, series: dict, m30: dict) -> dict:
    c = close.values.astype(float)
    n = len(c)
    F = _forward(c)
    out = {}
    nf = m30.get("nfci")
    if nf is None:
        return {}
    nraw = series["nfci"].values.astype(float)
    avg = pd.Series(nraw).rolling(364, min_periods=300).mean().shift(1).values
    gate = np.where(np.isfinite(nraw) & np.isfinite(avg), (nraw < avg).astype(float), np.nan)
    for vk in ("mvrv", "ma200"):
        if vk not in m30:
            continue
        val = m30[vk]
        defined = np.isfinite(nf) & np.isfinite(val)
        valid = defined & np.isfinite(F["f180"]) & np.isfinite(F["dd"])
        lo, hi = nf <= BOT, val >= TOP
        g = {"both": lo & hi & defined, "nfci": lo & ~hi & defined, "val": ~lo & hi & defined,
             "none": ~lo & ~hi & defined}
        if vk == "mvrv":
            gd = np.isfinite(gate) & defined
            g["gate_on"] = (gate > 0.5) & gd
            g["gate_off"] = (gate < 0.5) & gd
            g["gate_val"] = (gate > 0.5) & hi & gd
        vals = {"f90": np.nan_to_num(F["f90"]), "f180": np.nan_to_num(F["f180"]), "dd": np.nan_to_num(F["dd"]),
                "big": (np.nan_to_num(F["dd"]) <= -0.30).astype(float)}
        masks = {k: v & valid for k, v in g.items()}
        ci = _boot(vals, masks, valid, SEED + (7 if vk == "mvrv" else 8))
        res = {}
        idx = close.index
        for k, mk in masks.items():
            sel = np.flatnonzero(mk)
            per = {}
            for lab, a, b in PERIODS:
                pm = mk & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
                ps = np.flatnonzero(pm)
                per[lab] = (len(ps), float(np.mean(F["f180"][ps])) if len(ps) else None,
                            float(np.mean(F["dd"][ps])) if len(ps) else None)
            res[k] = {"n": len(sel), "share": float(mk.sum() / max(1, valid.sum())), "str": _stretches(mk),
                      "f90": float(np.mean(F["f90"][sel])) if len(sel) else None,
                      "f180": float(np.mean(F["f180"][sel])) if len(sel) else None,
                      "dd": float(np.mean(F["dd"][sel])) if len(sel) else None,
                      "big": float(np.mean(F["dd"][sel] <= -0.30)) if len(sel) else None,
                      "ci": ci[k], "per": per}
        first = idx[int(np.argmax(valid))] if valid.any() else None
        out[vk] = {"res": res, "first": first, "days": int(valid.sum())}
    out["gate_now"] = gate[-1] if len(gate) else None
    return out


# ---------- computation ----------

def _compute() -> dict:
    t0 = time.time()
    with _store():
        close = te._daily_close()
        series, dropped, problems, src, extra = sbt._assemble(close)
        press, pprob, psrc = _pressure(close)
    problems += pprob
    t_data = time.time() - t0
    n = len(close)
    c = close.values.astype(float)
    m30, lev = {}, {}
    for k, s in series.items():
        r = at._trail_pct(s.values, sbt.WIN, sbt.WIN_MIN) * 100
        W = sbt._windows(r)
        m30[k], lev[k] = W["m30"], W["lev"]
    pm30, plev = {}, {}
    for k, r in press.items():
        W = sbt._windows(np.asarray(r, dtype=float))
        pm30[k], plev[k] = W["m30"], W["lev"]
    turns = sbt._turns(close)
    S = _sets(close, lev, m30, turns)

    # Part 1
    P1 = {}
    for ty in ("bear", "bull"):
        st = S[ty]
        rows = []
        for j, t in enumerate(st["turns"]):
            T = t["i"]
            fset = None
            q = st["fwd_which"][T]
            if q >= 0:
                fset = st["fwd"][q]
            rows.append({"t": t, "loto": _per_turn(st["loto_each"][j], T), "lco": _per_turn(st["lco_each"][j], T),
                         "fwd": _per_turn(st["cnt"]["fwd"], T) if q >= 0 else {"ok": False},
                         "all": _per_turn(st["cnt"]["all"], T), "fset": fset,
                         "loto_on": _on(m30, st["loto"][j], T), "fwd_on": _on(m30, fset[1], T) if fset else []})
        tall = {}
        for v in ("loto", "lco", "fwd", "all"):
            ok = [r for r in rows if r[v]["ok"]]
            tall[v] = {"N": len(ok), **{L: sum(1 for r in ok if r[v][L] is not None) for L in LEVELS}}
        tidx = [t["i"] for t in turns[ty]]
        alarms, share = {}, {}
        for v in ("loto", "fwd", "all"):
            cnt = st["cnt"][v]
            share[v] = _nonturn_share(cnt, tidx, n)
            alarms[v] = {L: _episodes(cnt, L, tidx, n) for L in LEVELS}
        P1[ty] = {"rows": rows, "tally": tall, "share": share, "alarms": alarms}

    # Part 2 exam
    rows_h, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    other, prob2 = msc._load_other()
    problems += prob2
    ex = _Exam(close, other, price, b0, pre, FIRST)
    idx = close.index
    CS = {v: (pd.Series(S["bear"]["cnt"][v], index=idx), pd.Series(S["bull"]["cnt"][v], index=idx)) for v in VERS}
    zero = pd.Series(0.0, index=idx)
    base = ex.summary("ma50", zero, zero, KI["ma50"], SIMS)
    X = {v: {r: ex.summary(r, *CS[v], KI[r], SIMS) for r in RULES} for v in VERS}
    t_exam = time.time()
    # starts that each version can be scored on (count defined at the start date)
    scored = {}
    for v in VERS:
        B, U = CS[v]
        okd = B.notna() & U.notna()
        scored[v] = {"d": int(okd.reindex(ex.close.index[ex.starts]).fillna(False).sum()),
                     "c": int(okd.reindex(pd.DatetimeIndex(ex.m.dates[ex.cstarts]) - DAY).fillna(False).sum()) * 24,
                     "o": sum(int(okd.reindex(o["close"].index[o["starts"]] - o["lag"] * DAY).fillna(False).sum())
                              for o in ex.other.values()),
                     "first": idx[int(np.argmax(okd.values))]}
    # placebo
    PL, PLM = {}, {}
    rng = np.random.default_rng(SEED)
    for v in ("fwd", "loto"):
        B, U = CS[v]
        start = int(np.argmax((B.notna() & U.notna()).values))
        span = n - start
        small = {r: ex.summary(r, B, U, KI[r], PL_SIMS) for r in RULES}
        hits = {r: 0 for r in RULES}
        lows = {r: [] for r in RULES}
        for _ in range(DRAWS):
            k = int(rng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, span - SHIFT_MIN)))
            Bs, Us = _roll(B, k, start), _roll(U, k, start)
            for r in RULES:
                x = ex.summary(r, Bs, Us, KI[r], PL_SIMS)
                hits[r] += int(sd._score_ge(x, small[r]))
                lows[r].append(x["low"])
        PL[v] = {r: hits[r] / DRAWS for r in RULES}
        PLM[v] = {r: float(np.median(lows[r])) for r in RULES}
    t_pl = time.time()
    ps = {r: (1 + PL["fwd"][r] * DRAWS) / (1 + DRAWS) for r in RULES}
    holm = _holm(ps)
    sup = {r: _supported(r, X["fwd"][r], base, PL["fwd"][r]) for r in RULES}

    # positions on BTC daily and calendar years
    ma = ex.ma
    a20 = ex.a20
    pos = {"ma50": _rule("ma50", ma, a20, np.zeros(n), np.zeros(n))}
    for v in VERS:
        for r in RULES:
            pos[(v, r)] = _rule(r, ma, a20, S["bear"]["cnt"][v], S["bull"]["cnt"][v])
    cal = {k: wa._cal_years(close, p) for k, p in pos.items()}

    # plain answers
    PA = {}
    for v in ("fwd", "loto"):
        B, U = S["bear"]["cnt"][v], S["bull"]["cnt"][v]
        sp = _spells(c, ma, B)
        ea = _early(c, ma, a20, U)
        tops = []
        for t in S["bear"]["turns"]:
            T = t["i"]
            win = B[max(0, T - LOOK):T + 1]
            mx = float(np.nanmax(win)) if np.isfinite(win).any() else None
            on = np.nan_to_num(ma) > 0.5
            out_days = [i for i in range(max(0, T - LOOK), min(n, T + 61)) if on[i] and np.nan_to_num(B[i]) >= 5]
            xb_first = out_days[0] if out_days else None
            ma_exit = next((i for i in range(max(1, T - LOOK), n) if on[i - 1] and not on[i]), None)
            if ma_exit is not None and ma_exit < T - LOOK:
                ma_exit = None
            d = _eq(pos[(v, "xb")], c, T - 180, T + 180) / _eq(pos["ma50"], c, T - 180, T + 180) - 1
            ma_after = next((i for i in range(max(1, T), n) if on[i - 1] and not on[i]), None)
            tops.append({"t": t, "max": mx, "xb": xb_first, "ma": ma_exit, "ma_after": ma_after, "diff": d,
                         "full": T + 180 < n})
        bots = []
        for t in S["bull"]["turns"]:
            Bt = t["i"]
            on = np.nan_to_num(ma) > 0.5
            ma_on = next((i for i in range(Bt, n) if on[i]), None)
            lim = ma_on if ma_on is not None else n
            eps = [e for e in ea if Bt - LOOK <= e["s"] < lim]
            bots.append({"t": t, "ma_on": ma_on, "eps": eps})
        PA[v] = {"spells": sp, "early": ea, "tops": tops, "bots": bots}

    P4 = _part4(close, series, m30)

    # Part 3
    tops_all = {_d(t["date"]): t["i"] for t in turns["bear"]}
    P3 = {"tops": [d for d in FLAGGED + MISSED if d in tops_all], "idx": tops_all, "lev": {**lev, **plev}}
    return {"close": close, "series": series, "dropped": dropped, "problems": problems, "src": {**src, **psrc},
            "m30": m30, "lev": lev, "pm30": pm30, "turns": turns, "S": S, "P1": P1, "ex": ex, "base": base, "X": X,
            "scored": scored, "PL": PL, "PLM": PLM, "ps": ps, "holm": holm, "sup": sup, "cal": cal, "pos": pos,
            "PA": PA, "P4": P4, "P3": P3, "press": press,
            "secs": {"data": t_data, "exam": t_exam - t0, "placebo": t_pl - t_exam, "all": time.time() - t0}}


# ---------- tables ----------

def _sig_text(s: dict) -> str:
    return f"{_sname(s)}（{s['N']} 次裡 {s['k']} 次，隨機約 {s['p'] * s['N']:.1f}）"


def _set_tables(R: dict) -> list[dict]:
    idx = R["close"].index
    out = []
    for ty in ("bear", "bull"):
        st = R["S"][ty]
        rows = [[VSHORT["all"], "—", f"全部 {len(st['ti'])} 次", "、".join(_sig_text(s) for s in st["all"])]]
        for j, t in enumerate(st["turns"]):
            rows.append([VSHORT["loto"], "—", f"拿掉 {_d(t['date'])}", "、".join(_sig_text(s) for s in st["loto"][j])])
        for cf, s, U in st["fwd"]:
            rows.append([VSHORT["fwd"], _d(idx[cf]), "、".join(_d(t["date"]) for t in U),
                         "、".join(_sig_text(x) for x in s)])
        out.append({"caption": f"選出來的六個{sbt.TSHORT[ty]}徵兆（每一種做法）。挑法：在用來挑的那幾次轉折前 30 天，平均名次落在自己"
                               "過去兩年的偏高（≥80）或偏低（≤20）五分之一的次數比例，減掉隨機日子的比例，取多出最多的 6 個。括號＝"
                               "有資料的轉折裡有幾次在那一邊，以及隨機日子平均會有幾次。「往前」那幾列的日期＝從哪一天起才知道"
                               "（又一次轉折被確認的那天），只用到當時已確認的轉折。",
                    "columns": ["做法", "從哪天起用", "用哪些轉折來挑", "六個徵兆"], "rows": rows})
    return out


def _first_txt(v) -> str:
    return "沒有" if v is None else ("當天" if v == 0 else f"前 {v} 天")


def _p1_tables(R: dict) -> list[dict]:
    out = []
    for ty in ("bear", "bull"):
        P = R["P1"][ty]
        rows = []
        for r in P["rows"]:
            for v in ("loto", "fwd"):
                x = r[v]
                if not x["ok"]:
                    rows.append([_d(r["t"]["date"]), VSHORT[v], "—", "—", "—", "—",
                                 "之前確認的轉折不到 3 次，還不能挑" if v == "fwd" else "沒資料"])
                    continue
                on = r["loto_on"] if v == "loto" else r["fwd_on"]
                rows.append([_d(r["t"]["date"]), VSHORT[v], f"{x['at']:.0f}／6", f"{x['max']:.0f}／6",
                             _first_txt(x[4]), _first_txt(x[5]), "、".join(on) or "（沒有）"])
        out.append({"caption": f"第 1 部分・每一次{sbt.TNAME[ty]}：用沒看過這次轉折的徵兆，轉折當天亮了幾個（6 個裡）、前 90 天"
                               "最多亮幾個、第一次到 4 個／5 個是轉折前幾天。「留一」＝拿掉這一次、用其他 7 次挑的 6 個徵兆"
                               "（還是用到之後的轉折）；「往前」＝只用這一天以前已確認的轉折挑的（當時真的做得到）。",
                    "columns": ["轉折日", "做法", "當天亮幾個", "前 90 天最多", "第一次 ≥4", "第一次 ≥5", "當天亮的徵兆"],
                    "rows": rows})
    rows = []
    for ty in ("bear", "bull"):
        P = R["P1"][ty]
        for v in ("loto", "lco", "fwd", "all"):
            t = P["tally"][v]
            if v == "lco":
                rows.append([sbt.TSHORT[ty], "留一群（穩健度）", f"{t['N']} 次裡 {t[4]} 次", f"{t['N']} 次裡 {t[5]} 次",
                             "—", "—", "—", "—"])
                continue
            sh, al = P["share"][v], P["alarms"][v]

            def ep(L, al=al):
                a = al[L]
                return f"{a['n']} 段｜{a['fol']} 段" + (f"（{a['unres']} 段還沒滿 180 天）" if a["unres"] else "")
            rows.append([sbt.TSHORT[ty], VSHORT[v], f"{t['N']} 次裡 {t[4]} 次", f"{t['N']} 次裡 {t[5]} 次",
                         _sh(sh[4]), _sh(sh[5]), ep(4), ep(5)])
    out.append({"caption": "第 1 部分・合計與誤報。「標到」＝轉折前 90 天內（含當天）徵兆數曾到 4 個／5 個的次數。「非轉折日」＝離這類轉折"
                           "90 天以上的日子裡，徵兆數 ≥4／≥5 的比例。「段數」＝徵兆數連續 90 天以上低於門檻後又到門檻，算一段；"
                           "後面那個數＝其中 180 天內真的出現這類轉折的段數。「留一群」＝拿掉一次轉折時，連同 365 天內同類的轉折一起拿掉"
                           "（2013 年 4 月兩次走熊只隔 15 天），只算標到幾次。「全部」用所有轉折挑徵兆，是樣本內，只當參考。",
                "columns": ["類型", "做法", "標到（≥4）", "標到（≥5）", "非轉折日 ≥4", "非轉折日 ≥5", "≥4 段數｜之後有轉折",
                            "≥5 段數｜之後有轉折"], "rows": rows})
    return out


def _exam_tables(R: dict) -> list[dict]:
    X, base, PL, sup, holm = R["X"], R["base"], R["PL"], R["sup"], R["holm"]
    sc = R["scored"]
    out = []
    rows = [["50 日均線（基準）", "—", _sh(base["low"]) + ("" if base["eligible"] else "（虧損門檻未過）"),
             f"{base['c_deep_med']:.0f}%｜{base['c_deep_worst']:.0f}%", _sh(base["time"]), "—", "—", "—"]]
    for r in RULES:
        for v in VERS:
            x = X[v][r]
            pl = PL.get(v, {}).get(r)
            if v == "fwd":
                s = sup[r]
                verdict = ("支持" if s["ok"] else "不支持") + "（" + "、".join(
                    f"{lab}{'✓' if s[k] else '✗'}" for k, lab in (("rank", "排名"), ("loss", "虧損"), ("pl", "安慰劑"))) + "）"
                hl = "通過" if holm.get(r) else "沒通過"
            else:
                verdict, hl = "只報告", "—"
            rows.append([RSHORT[r], VSHORT[v], _sh(x["low"]) + ("" if x["eligible"] else "（虧損門檻未過）"),
                         f"{x['c_deep_med']:.0f}%｜{x['c_deep_worst']:.0f}%", _sh(x["time"]),
                         _sh(pl) if pl is not None else "—", hl, verdict])
    out.append({"caption": "第 2 部分・公平考試（和 50 日均線同一批起算日）。排名分數＝四項檢查（比特幣 108 個起算日、起算日×24 個收盤時間、"
                           "一般人做法、其他幣）裡「每年報酬贏過一直抱著」比例最低的那一項。最深虧損＝起算日×收盤時間各格的最深虧損"
                           "中位數｜最深的一格。持有時間＝2015 年起平均持有比例。安慰劑＝把徵兆數整條錯開 365 天以上（200 次）"
                           "後，分數不輸真的那條的比例（一般人做法每個收盤時間只模擬 20 次，真的那條也用 20 次重算）。只有「往前」"
                           "那一列能被判定支持；Holm＝五條「往前」規則一起校正後是否仍在 5% 以下。"
                           f"可計分的起算日：往前 {sc['fwd']['d']} 個／{sc['fwd']['c']:,} 格／其他幣 {sc['fwd']['o']} 個，"
                           f"留一 {sc['loto']['d']} 個／{sc['loto']['c']:,} 格／{sc['loto']['o']} 個"
                           f"（往前的徵兆數從 {_d(sc['fwd']['first'])} 起才有）。",
                "columns": ["規則", "做法", "排名分數", "最深虧損：中位數｜最深", "持有時間", "安慰劑", "Holm", "判定（往前）"],
                "rows": rows})
    rows = []
    for k, x in [("ma50", base)] + [((r, v), X[v][r]) for r in RULES for v in ("fwd", "loto")]:
        lab = ["50 日均線", "—"] if k == "ma50" else [RSHORT[k[0]], VSHORT[k[1]]]
        rows.append(lab + [_sh(v) for v in x["checks"]] + [f"{x['c_edge']:+.1f}"])
    out.append({"caption": "第 2 部分・四項檢查各自的「每年報酬贏過一直抱著」比例，以及起算日×收盤時間每年報酬比一直抱著多幾個百分點"
                           "（中位數）。其他幣用比特幣的徵兆數當成整個市場的狀態。",
                "columns": ["規則", "做法", "比特幣起算日", "起算日×收盤時間", "一般人做法", "其他幣", "每年多幾點（中位數）"],
                "rows": rows})
    for v in ("fwd", "loto"):
        years = sorted(R["cal"]["ma50"])
        rows = []
        for y in years:
            if y < 2012:
                continue
            m = R["cal"]["ma50"][y]
            rows.append([str(y), _pc(m)] + [f"{(R['cal'][(v, r)][y] - m) * 100:+.1f}" for r in RULES])
        out.append({"caption": f"第 2 部分・每一年（{VNAME[v]}）：50 日均線那一年的報酬，以及每條規則減掉 50 日均線（百分點；比特幣 UTC 0 點"
                               "日線、每次買賣扣 0.10%）。正的＝那一年比 50 日均線好。今年算到最後一根收盤。",
                    "columns": ["年份", "50 日均線"] + [RSHORT[r] for r in RULES], "rows": rows})
    return out


def _pa_tables(R: dict) -> list[dict]:
    idx, c = R["close"].index, R["close"].values.astype(float)
    out = []
    for v in ("fwd", "loto"):
        sp = R["PA"][v]["spells"]
        rows = [[_d(idx[s["s"]]), _d(idx[s["e"]]) + ("（還在）" if s["open"] else ""), str(s["e"] - s["s"] + 1),
                 str(s["skip"]), _pc(s["miss"], 1), _pc(s["chg"], 1), _pc(s["low"], 1)] for s in sp]
        out.append({"caption": f"照著出場，少不少賠？（{VNAME[v]}）每一段「空頭徵兆 ≥5 個、而且期間 50 日均線有幾天本來會持有」的時間。"
                               "「沒抱的天數」＝50 日均線本來持有、XB 沒抱的天數；「那些天的漲跌」＝50 日均線在那些天賺或賠了多少"
                               "（負的＝XB 少賠，正的＝XB 少賺）；「期間價格」＝這段開始的收盤到結束後那天的收盤；「期間最低」＝期間"
                               "最低收盤比開始低多少。不扣成本。",
                    "columns": ["開始", "結束", "天數", "沒抱的天數", "那些天的漲跌", "期間價格", "期間最低"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        for x in R["PA"][v]["tops"]:
            T = x["t"]["i"]
            p = x["t"]["p"]

            def cell(i, T=T, p=p):
                return "沒有" if i is None else f"{_d(idx[i])}（{i - T:+d} 天｜{_pc(c[i] / p - 1)}）"
            rows.append([_d(x["t"]["date"]), VSHORT[v], "—" if x["max"] is None else f"{x['max']:.0f}／6", cell(x["xb"]),
                         cell(x["ma"]), cell(x.get("ma_after")),
                         _pc(x["diff"], 1) + ("" if x["full"] else "（還沒滿 180 天）")])
    out.append({"caption": "照著出場・每一次走熊開始（頭部）：頭部前 90 天空頭徵兆最多亮幾個；XB 在頭部前 90 天到後 60 天之間第一次"
                           "因徵兆而出場（50 日均線還在持有）的日子（離頭部幾天｜價格比頭部收盤）；50 日均線在頭部前 90 天以後第一次"
                           "出場的日子；（看過第一次結果後加的一欄，只是描述）50 日均線在頭部當天或之後第一次出場的日子——前一欄常常"
                           "是頭部前很久、價格還低很多時的一次短暫出場；頭部前後各 180 天，XB 的報酬比 50 日均線多（少）多少（不扣成本）。"
                           "2013 年 4 月有兩次。",
                "columns": ["頭部日", "做法", "前 90 天最多", "XB 第一次因徵兆出場", "50 日均線第一次出場",
                            "50 日均線頭部後第一次出場", "前後 180 天 XB 比均線"],
                "rows": rows})
    for v in ("fwd", "loto"):
        ea = R["PA"][v]["early"]
        rows = [[_d(idx[e["s"]]), _px(c[e["s"]]), _d(idx[e["x"]]), str(e["x"] - e["s"]), _pc(e["gain"], 1), e["why"]]
                for e in ea]
        out.append({"caption": f"照著進場，賺不賺？（{VNAME[v]}）每一次 EB 比 50 日均線早進場：多頭徵兆 ≥4 個而且站上 20 日均線、"
                               "50 日均線還沒持有。從進場收盤算到結束那天收盤（50 日均線接手、跌回 20 日均線下、或多頭徵兆少於 4 個）。"
                               "不扣成本。",
                    "columns": ["進場日", "進場價（美元）", "結束日", "天數", "這段漲跌", "怎麼結束"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        for x in R["PA"][v]["bots"]:
            p = x["t"]["p"]
            mo = x["ma_on"]
            eps = x["eps"]
            first = eps[0] if eps else None
            tot = float(np.prod([1 + e["gain"] for e in eps]) - 1) if eps else None
            rows.append([_d(x["t"]["date"]), VSHORT[v],
                         "—" if mo is None else f"{_d(idx[mo])}（{_pc(c[mo] / p - 1)}）",
                         "沒有" if first is None else f"{_d(idx[first['s']])}（{_pc(c[first['s']] / p - 1)}）",
                         "—" if first is None or mo is None else _pc(c[first["s"]] / c[mo] - 1),
                         f"{len(eps)} 次｜{_pc(tot, 1)}" if eps else "0 次"])
    out.append({"caption": "照著進場・每一次走牛開始（底部）：50 日均線在底部以後第一次持有的日子（價格比底部）；EB 在底部前 90 天到 50 日"
                           "均線進場之間第一次提早進場（價格比底部）；第一次提早進場的價格比 50 日均線進場價高低多少（負的＝比較便宜）；"
                           "那段期間 EB 提早進場幾次、合計賺賠（不扣成本）。",
                "columns": ["底部日", "做法", "50 日均線進場", "EB 第一次提早進場", "比均線進場價", "提早進場次數｜合計"],
                "rows": rows})
    return out


def _p3(R: dict) -> dict:
    lev, P3 = R["P3"]["lev"], R["P3"]
    tops = P3["tops"]
    fl = [d for d in tops if d in FLAGGED]
    mi = [d for d in tops if d in MISSED]
    vals = {k: {d: lev[k][P3["idx"][d]] for d in tops} for k in lev}

    def ext(v, side):
        return _ok(v) and ((side > 0 and v >= TOP) or (side < 0 and v <= BOT))
    only_missed, only_flag, diffs = [], [], []
    for k in lev:
        fm = [vals[k][d] for d in fl if _ok(vals[k][d])]
        mm = [vals[k][d] for d in mi if _ok(vals[k][d])]
        for side in (1, -1):
            em = sum(ext(v, side) for v in mm)
            ef = sum(ext(v, side) for v in fm)
            if em >= 2 and ef <= 1:
                only_missed.append((k, side, em, len(mm), ef, len(fm)))
            if len(fm) and ef >= max(2, math.ceil(len(fm) / 2)) and em <= 1:
                only_flag.append((k, side, ef, len(fm), em, len(mm)))
        if len(mm) == len(mi) and len(fm) >= 3:
            diffs.append((float(np.median(mm) - np.median(fm)), k, mm, fm))
    diffs.sort(key=lambda x: (-abs(x[0]), x[1]))
    return {"fl": fl, "mi": mi, "vals": vals, "only_missed": only_missed, "only_flag": only_flag, "diffs": diffs[:12]}


def _p3_tables(R: dict, Q: dict) -> list[dict]:
    tops = Q["fl"] + Q["mi"]
    rows = []
    for k in list(sbt.META) + list(PRESS_KEYS):
        if k not in Q["vals"]:
            continue
        cells = []
        for d in tops:
            v = Q["vals"][k][d]
            cells.append("—" if not _ok(v) else f"{v:.0f}" + ("高" if v >= TOP else ("低" if v <= BOT else "")))
        if all(x == "—" for x in cells):
            continue
        rows.append([_name(k)] + cells)
    out = [{"caption": "第 3 部分・每一個指標在每次頭部前 30 天的平均名次（0–100，和自己過去兩年比；≥80 標「高」、≤20 標「低」）。"
                       f"前 {len(Q['fl'])} 欄＝第一步標得很清楚的頭部，後 {len(Q['mi'])} 欄＝只標到一半的頭部。最後幾列（如果有）是"
                       "買賣壓力研究的合計排名（它的排名是「過去兩年比它低的比例」，和其他列「≤ 它的比例」差一點點）。",
            "columns": ["指標"] + [d + ("（標到）" if d in FLAGGED else "（漏掉）") for d in tops], "rows": rows}]
    rows = []
    for dlt, k, mm, fm in Q["diffs"]:
        rows.append([_name(k), "、".join(f"{v:.0f}" for v in mm), f"{np.median(fm):.0f}（{len(fm)} 次）", f"{dlt:+.0f}",
                     sbt.META[k][1] if k in sbt.META else PRESS_META[k][1]])
    lrows = []
    for k, s, e2, n2, e1, n1 in Q["only_missed"]:
        lrows.append(["只在漏掉的三次常見", f"{_name(k)}（{'高' if s > 0 else '低'}）", f"{n2} 次裡 {e2} 次",
                     f"{n1} 次裡 {e1} 次" if n1 else "沒資料"])
    for k, s, e1, n1, e2, n2 in Q["only_flag"]:
        lrows.append(["只在標到的頭部常見", f"{_name(k)}（{'高' if s > 0 else '低'}）", f"{n2} 次裡 {e2} 次",
                     f"{n1} 次裡 {e1} 次"])
    out.append({"caption": "第 3 部分・兩種頭部各自常見的極端（事先定的挑法：在漏掉的三次裡至少 2 次在同一個極端、標到的頭部最多 1 次；"
                           "或反過來，在有資料的標到頭部裡至少一半而且至少 2 次、漏掉的三次最多 1 次）。資料少的指標（衍生品、ETF、"
                           "穩定幣）只碰得到最近幾次，次數很少；只是描述，解讀是推測。",
                "columns": ["哪一種", "指標（哪一邊）", "漏掉的三次", "標到的頭部（有資料的）"], "rows": lrows})
    out.append({"caption": "第 3 部分・漏掉的三次和標到的頭部比，名次差最多的 12 個（三次都有資料、標到的至少 3 次有資料）。差距＝漏掉三次的"
                           "中位數減標到幾次的中位數。只是描述，次數太少，任何解讀都是推測。",
                "columns": ["指標", "漏掉三次的名次", "標到的中位數", "差距", "白話說明"], "rows": rows})
    return out


P4_LAB = {"both": "金融條件寬鬆（NFCI 低五分之一）＋估值高五分之一", "nfci": "只有金融條件寬鬆", "val": "只有估值高",
          "none": "兩個都沒有", "gate_on": "NFCI 低於自己 52 週平均（nfci_filter 的閘門開）",
          "gate_off": "NFCI 高於自己 52 週平均（閘門關）", "gate_val": "閘門開＋估值高五分之一"}


def _p4_tables(R: dict) -> list[dict]:
    P4 = R["P4"]
    out = []
    for vk, vname in (("mvrv", "MVRV"), ("ma200", "離 200 日均線")):
        if vk not in P4:
            continue
        res = P4[vk]["res"]
        rows = []
        for g in ("both", "nfci", "val", "none", "gate_on", "gate_off", "gate_val"):
            if g not in res:
                continue
            x = res[g]
            ci = x["ci"]

            def cc(k, x=x, ci=ci):
                return f"{_pc(x[k])}（{_pc(ci[k][0])} 到 {_pc(ci[k][1])}）" if x["n"] else "—"
            rows.append([P4_LAB[g].replace("估值", vname), f"{x['n']:,}（{_sh(x['share'])}｜{x['str']} 段）", cc("f90"),
                         cc("f180"), cc("dd"), f"{_sh(x['big'])}（{_sh(ci['big'][0])} 到 {_sh(ci['big'][1])}）"
                         if x["n"] else "—"])
        out.append({"caption": f"第 4 部分・金融條件寬鬆和估值（{vname}）放在一起看，之後的比特幣：每天依 NFCI（芝加哥聯準銀行金融條件指數，"
                               f"越低越寬鬆）和 {vname} 的 30 天平均名次分組，從 {_d(P4[vk]['first'])} 起、之後還有 180 天資料的 "
                               f"{P4[vk]['days']:,} 天。括號＝區塊自助抽樣（每塊 180 天、1,000 次）的 90% 範圍；相鄰的日子幾乎是同一件事，"
                               "「段數」＝隔 30 天以上才算新的一段，真正獨立的次數接近段數，不是天數。"
                               + ("後三列是 nfci_filter 的閘門（NFCI 低於自己過去 52 週平均才持有）的每天版本，和前四列並排，不互相否定。"
                                  if vk == "mvrv" else ""),
                    "columns": ["組別", "天數（比例｜段數）", "之後 90 天漲跌", "之後 180 天漲跌", "180 天內最深跌幅",
                                "180 天內跌超過 30% 的比例"], "rows": rows})
    if "mvrv" in P4:
        res = P4["mvrv"]["res"]
        rows = []
        for g in ("both", "nfci", "val", "none", "gate_on", "gate_off"):
            x = res[g]
            rows.append([P4_LAB[g].replace("估值", "MVRV")] + [
                ("—" if not x["per"][p][0] else f"{_pc(x['per'][p][1])}｜{_pc(x['per'][p][2])}（{x['per'][p][0]:,} 天）")
                for p, _, _ in PERIODS])
        out.append({"caption": "第 4 部分・分時期（MVRV 版）：每格＝之後 180 天平均漲跌｜180 天內平均最深跌幅（天數）。",
                    "columns": ["組別"] + [p for p, _, _ in PERIODS], "rows": rows})
    return out


# ---------- verdict ----------

def _tally_line(R: dict, ty: str, v: str) -> str:
    P = R["P1"][ty]
    t = P["tally"][v]
    s = f"{VSHORT[v]} {t['N']} 次裡 {t[4]} 次到 4 個、{t[5]} 次到 5 個"
    if v in P["share"]:
        sh, al = P["share"][v], P["alarms"][v]
        s += (f"（非轉折日 {_sh(sh[4])} 的日子 ≥4、{_sh(sh[5])} ≥5；≥4 出現 {al[4]['n']} 段、其中 {al[4]['fol']} 段 180 天內"
              f"真的{sbt.TSHORT[ty]}；≥5 出現 {al[5]['n']} 段、{al[5]['fol']} 段）")
    return s


def _exit_sum(R: dict, v: str) -> tuple:
    sp = [s for s in R["PA"][v]["spells"] if s["skip"]]
    neg = [s["miss"] for s in sp if s["miss"] < 0]
    posi = [s["miss"] for s in sp if s["miss"] > 0]
    net = float(np.prod([1 / (1 + s["miss"]) for s in sp]) - 1) if sp else 0.0
    return len(sp), neg, posi, net


def _entry_sum(R: dict, v: str) -> tuple:
    ea = [e for e in R["PA"][v]["early"] if not e["open"]]
    g = [e["gain"] for e in ea]
    return (len(ea), sum(x > 0 for x in g), float(np.mean(g)) if g else None, float(np.median(g)) if g else None,
            float(np.prod([1 + x for x in g]) - 1) if g else None)


def _tops_line(R: dict, v: str) -> str:
    out = []
    for x in R["PA"][v]["tops"]:
        T = x["t"]["i"]
        if x["max"] is None:
            out.append(f"{_d(x['t']['date'])} 還沒有徵兆組")
            continue
        out.append(f"{_d(x['t']['date'])} 前 90 天最多 {x['max']:.0f} 個、"
                   + ("XB 沒有因徵兆出場" if x["xb"] is None else f"XB 在頭部 {x['xb'] - T:+d} 天出場")
                   + f"、前後半年比均線 {_pc(x['diff'], 1)}")
    return f"{VSHORT[v]}版本在每次頭部：" + "；".join(out) + "。"


def _year_line(R: dict) -> str:
    out = []
    for y in (2018, 2022):
        m = R["cal"]["ma50"].get(y)
        if m is None:
            continue
        out.append(f"{y} 年 50 日均線 {_pc(m)}，XB 往前／留一比它 "
                   f"{(R['cal'][('fwd', 'xb')][y] - m) * 100:+.1f}／{(R['cal'][('loto', 'xb')][y] - m) * 100:+.1f} 個百分點")
    return "；".join(out) + "。"


def _bots_line(R: dict, v: str) -> str:
    c = R["close"].values.astype(float)
    out = []
    for x in R["PA"][v]["bots"]:
        eps = x["eps"]
        if not eps:
            out.append(f"{_d(x['t']['date'])} 沒有提早進場")
            continue
        tot = float(np.prod([1 + e["gain"] for e in eps]) - 1)
        out.append(f"{_d(x['t']['date'])} 提早進場 {len(eps)} 次、合計 {_pc(tot, 1)}"
                   + ("" if x["ma_on"] is None else f"、第一次比均線進場價 {_pc(c[eps[0]['s']] / c[x['ma_on']] - 1)}"))
    return f"{VSHORT[v]}版本在每次底部：" + "；".join(out) + "。"


def _p4_line(R: dict) -> str:
    P4 = R["P4"]
    if "mvrv" not in P4:
        return "第 4 部分算不出來（缺 NFCI 或 MVRV）。"
    r = P4["mvrv"]["res"]

    def g(k):
        x = r[k]
        if not x["n"]:
            return "沒有這種日子"
        return (f"{x['n']:,} 天／{x['str']} 段，之後 180 天平均 {_pc(x['f180'])}（{_pc(x['ci']['f180'][0])} 到 "
                f"{_pc(x['ci']['f180'][1])}），180 天內平均最深 {_pc(x['dd'])}")
    return ("量到的（第 4 部分，MVRV 版）：金融條件寬鬆＋估值高 " + g("both") + "；只有寬鬆 " + g("nfci") + "；只有估值高 "
            + g("val") + "；都沒有 " + g("none") + "。nfci_filter 的閘門（NFCI 低於自己 52 週平均）開 " + g("gate_on")
            + "；關 " + g("gate_off") + "。兩件事並排：寬鬆在頭部前常見（第一步），閘門在寬鬆時才讓持有（nfci_filter）；"
            "相鄰的日子幾乎是同一件事，範圍很寬，這裡不拿其中一個去否定另一個。")


def _verdict(R: dict, Q: dict) -> str:
    X, base, PL, sup, holm = R["X"], R["base"], R["PL"], R["sup"], R["holm"]
    lco = {ty: R["P1"][ty]["tally"]["lco"] for ty in ("bear", "bull")}
    good0 = [r for r in RULES if sup[r]["ok"]]
    ex0 = _exit_sum(R, "fwd")
    lead = (f"一句話（量到的）：照著徵兆做的五條「往前」規則（只用當時已確認的轉折挑徵兆），照事先定的標準被支持的有 {len(good0)} 條；"
            f"出場規則 XB 的排名分數 {_sh(X['fwd']['xb']['low'])}（50 日均線 {_sh(base['low'])}），它站開的 {ex0[0]} 段裡"
            f" {len(ex0[2])} 段之後在漲；提早進場 EB {_sh(X['fwd']['eb']['low'])}。這是用這個定義、這些資料、這麼少的轉折量到的，"
            "不是說徵兆沒有意義。")
    parts = [lead, "(a) 量到的（第 1 部分，沒看過那次轉折的徵兆數）：走熊開始前—" + "；".join(_tally_line(R, "bear", v) for v in
             ("loto", "fwd", "all")) + f"；留一群 {lco['bear']['N']} 次裡 {lco['bear'][4]} 次到 4 個。走牛開始前—"
             + "；".join(_tally_line(R, "bull", v) for v in ("loto", "fwd", "all"))
             + f"；留一群 {lco['bull']['N']} 次裡 {lco['bull'][4]} 次到 4 個。「全部」是用所有轉折挑的，一定看得比較準，只當參考；"
             "每一次的細節在表裡。"]
    good = [r for r in RULES if sup[r]["ok"]]
    hl = [r for r in RULES if holm.get(r)]
    exam = "；".join(f"{RSHORT[r]} 往前／留一 {_sh(X['fwd'][r]['low'])}／{_sh(X['loto'][r]['low'])}，最深虧損中位數 "
                    f"{X['fwd'][r]['c_deep_med']:.0f}%／{X['loto'][r]['c_deep_med']:.0f}%，安慰劑 {_sh(PL['fwd'][r])}／"
                    f"{_sh(PL['loto'][r])}" for r in RULES)
    parts.append(f"(b) 量到的（第 2 部分公平考試）：50 日均線排名分數 {_sh(base['low'])}、最深虧損中位數 {base['c_deep_med']:.0f}%。"
                 f"{exam}。照事先定的標準（只看「往前」），支持的有 {len(good)} 條"
                 + (f"（{'、'.join(RSHORT[r] for r in good)}）" if good else "")
                 + f"；五條一起做 Holm 校正後仍通過的 {len(hl)} 條" + (f"（{'、'.join(RSHORT[r] for r in hl)}）" if hl else "")
                 + "；五條各用 5% 門檻，光靠運氣平均約 0.25 條會過。")
    ex = {v: _exit_sum(R, v) for v in ("fwd", "loto")}
    def _abs(v):
        return "—" if v is None else f"{abs(v) * 100:.1f}%"
    parts.append("(c) 照著出場，少不少賠？（量到的，比特幣日線、不扣成本）" + "；".join(
        f"{VSHORT[v]}：XB 因空頭徵兆 ≥5 個而站開、而且 50 日均線本來有持有的共 {ex[v][0]} 段，其中 {len(ex[v][1])} 段那些天在跌"
        f"（平均少賠 {_abs(np.mean(ex[v][1]) if ex[v][1] else None)}），{len(ex[v][2])} 段在漲（平均少賺 "
        f"{_abs(np.mean(ex[v][2]) if ex[v][2] else None)}），全部合起來 XB 的帳戶是 50 日均線的 {(1 + ex[v][3]) * 100:.0f}%"
        for v in ("fwd", "loto")) + "。" + _tops_line(R, "fwd") + _year_line(R))
    en = {v: _entry_sum(R, v) for v in ("fwd", "loto")}
    parts.append("(d) 照著進場，賺不賺？（量到的，不扣成本）" + "；".join(
        f"{VSHORT[v]}：EB 提早進場 {en[v][0]} 次（已結束的），{en[v][1]} 次結束時比進場高，平均 {_pc(en[v][2], 1)}、"
        f"中位數 {_pc(en[v][3], 1)}、全部連乘 {_pc(en[v][4], 1)}" for v in ("fwd", "loto")) + "。" + _bots_line(R, "fwd"))
    om = "、".join(f"{_name(k)}{'高' if s > 0 else '低'}（漏掉 {n2} 次裡 {e2} 次；"
                   + (f"標到的 {n1} 次裡 {e1} 次）" if n1 else "標到的頭部沒資料）")
                   for k, s, e2, n2, e1, n1 in Q["only_missed"]) or "沒有"
    of_ = "、".join(f"{_name(k)}{'高' if s > 0 else '低'}（標到 {n1} 次裡 {e1} 次；漏掉 {n2} 次裡 {e2} 次）"
                    for k, s, e1, n1, e2, n2 in Q["only_flag"] if n1 >= 3) or "沒有"
    parts.append(f"(e) 第 3 部分（名次是量到的，解讀全是推測）：只在漏掉的三次頭部（{'、'.join(Q['mi'])}）常在極端、標到的頭部很少的："
                 f"{om}。標到的頭部常有（至少 3 次有資料）、漏掉的三次很少的：{of_}。推測：如果後者多半是估值、價格離均線這類「漲過頭」的指標，"
                 "漏掉的三次可能是另一種頭部——價格沒有漲到過去兩年的極端就轉弱；但也可能只是比較基準的問題：2019-06 和 2021-11 "
                 "前面兩年內就有大頭部或大漲，「和自己過去兩年比」本身就被拉高。只有三次，這兩種說法用這些資料分不開。")
    parts.append("(f) " + _p4_line(R))
    parts.append("(g) 看不到的與限制：每一類只有約 8 次轉折，2013 年 4 月算成兩次走熊、一次走牛；「往前」的第一組走熊徵兆只用 3 次轉折"
                 "（其中兩次相隔 15 天）挑；六個徵兆多半是價格或估值類，彼此一起動，不是六個獨立線索；鏈上數字、短期持有人成本、"
                 "OECD 用今天的修正版；「留一」仍用到之後的轉折挑徵兆，只有「往前」是當時做得到的。沒看到的效果只表示用這個定義和"
                 "這些資料沒看到，可能漏掉的是：別種組合方式（例如加權而不是數個數）、更多資產的轉折、當時的原始數字。")
    return "\n".join(parts)


def _now(R: dict) -> list[dict]:
    idx, c = R["close"].index, R["close"].values.astype(float)
    i = len(idx) - 1
    word = {True: "持有", False: "現金"}
    p = R["pos"]
    out = [{"label": f"最新收盤（{_d(idx[i])}，UTC 0 點日線）", "value": f"{c[i]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜XB｜EB",
            "value": f"{word[bool(p['ma50'][i] > 0.5)]}｜{word[bool(p[('fwd', 'xb')][i] > 0.5)]}｜"
                     f"{word[bool(p[('fwd', 'eb')][i] > 0.5)]}"},
           {"label": "SB（往前）持有比例", "value": f"{p[('fwd', 'sb')][i] * 100:.0f}%"}]
    for ty in ("bear", "bull"):
        st = R["S"][ty]
        q = st["fwd_which"][i]
        on_f = _on(R["m30"], st["fwd"][q][1], i) if q >= 0 else []
        j = st["which"][i]
        lset = st["loto"][j] if j >= 0 else st["all"]
        on_l = _on(R["m30"], lset, i)
        fv = st["cnt"]["fwd"][i]
        out.append({"label": f"{sbt.TSHORT[ty]}徵兆數（往前｜留一）",
                    "value": (f"{fv:.0f}／6（{'、'.join(on_f) or '沒有亮的'}）" if _ok(fv) else "—")
                             + f"｜{st['cnt']['loto'][i]:.0f}／6（{'、'.join(on_l) or '沒有亮的'}）"})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根收盤後的狀態和徵兆數，是描述，不是預測，也不是買賣建議"})
    return out


def study() -> dict:
    t0 = time.time()
    R = _compute()
    Q = _p3(R)
    tables = _set_tables(R) + _p1_tables(R) + _exam_tables(R) + _pa_tables(R) + _p3_tables(R, Q) + _p4_tables(R)
    verdict = _verdict(R, Q)
    sec = R["secs"]
    notes = [
        "這是第二步：照著第一步（signs_before_turns）看到的徵兆去做，看看出場是不是少賠、進場是不是賺。第一步的徵兆是看過所有轉折才挑的，"
        "所以這裡主要用「留一」（拿掉要考的那次再挑）和「往前」（只用當時已確認的轉折挑）兩種做法；只有「往前」是當時真的做得到的，"
        "也只有它能被判定「支持」。",
        "名詞：名次＝今天的值在自己過去 730 天裡排第幾（0–100）；徵兆「亮」＝過去 30 天平均名次落在它那一邊的五分之一（≥80 或 ≤20）；"
        "50 日均線＝收盤高於過去 50 天平均才持有（目前的主訊號）；20 日均線同理；最深虧損＝從起算日到現在帳戶從高點掉最多的幅度；"
        "NFCI＝芝加哥聯準銀行金融條件指數，越高代表借錢越難；MVRV＝市值除以每顆幣上次移動時價格加總的成本。",
        "轉折的「確認」：走熊開始要等價格從高點跌掉一半、走牛開始要等價格從低點翻倍才算確認；「往前」只在確認以後才把那次轉折拿來挑徵兆。",
        "「留一」在離某次轉折 365 天以內的日子，用拿掉最近那次轉折挑的徵兆；離所有轉折都超過 365 天的日子，用全部轉折挑的那組"
        "（那些日子沒有要考的轉折）。",
        "公平考試和 main_signal_check 一樣：比特幣 2015-01 到 2023-12 每月起算、2018-01 到 2023-12 每月起算×24 個收盤時間、"
        "一般人做法（晚 10 小時、成本 0.25%、一成進場漏掉、每個收盤時間 200 次模擬）、其他幣（ETH、SOL、BNB、XRP、MSTR）。"
        "其他幣沒有自己的徵兆，用比特幣的徵兆數。安慰劑每次只模擬 20 次，真的那條也用 20 次重算來比。",
        "「照著出場／進場」那幾張表不扣成本，只看價格；考試的數字有扣成本。",
        "限制：每一類只有約 8 次轉折；2013 年 4 月算成兩次走熊、一次走牛；六個徵兆多半是價格或估值類、彼此一起動；鏈上、短期持有人成本、"
        "OECD 用今天的修正版；「往前」能學的早期轉折很少（第一組走熊徵兆只用 3 次、其中兩次相隔 15 天）；衍生品、ETF 等指標歷史短，"
        "早期的徵兆組選不到它們。",
        "所有門檻（6 個、≥4／≥5、90／180／365 天、20 日均線、安慰劑錯開 365 天以上）都在跑之前寫好，看過結果後沒有改。",
        f"這次計算：資料約 {sec['data']:.0f} 秒、考試約 {sec['exam'] - sec['data']:.0f} 秒、安慰劑約 {sec['placebo']:.0f} 秒，"
        f"全部約 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來一樣；這裡只是描述幾條固定規則過去做了什麼，不是買賣建議。",
    ] + [p for p in R["problems"] if p]
    if R["dropped"]:
        notes.append("這次拿掉的指標：" + "、".join(f"{_name(k)}（{v}）" for k, v in R["dropped"].items()))
    src = dict(R["src"])
    src["幣安現貨 BTCUSDT 1 小時線（24 個收盤時間）"] = px.BINANCE_HOSTS[0] + "/api/v3/klines?symbol=BTCUSDT&interval=1h"
    for s in ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"):
        src[f"幣安現貨 {s} 日線"] = px.BINANCE_HOSTS[0] + f"/api/v3/klines?symbol={s}&interval=1d"
    src["Yahoo 日線 MSTR"] = ro.YAHOO.format("MSTR")
    return {"id": ID, "title": "照著轉折徵兆做：出場少賠嗎？進場賺嗎？",
            "question": "第一步看到走熊、走牛之前常出現的徵兆。如果真的照著它們做——空頭徵兆多就出場、多頭徵兆多就提早進場——"
                        "在沒看過那次轉折的情況下，出場有沒有少賠、進場有沒有賺？漏掉的頭部長什麼樣子？寬鬆的金融條件該怎麼看？",
            "asof": _d(R["close"].index[-1]), "verdict": verdict, "now": _now(R), "tables": tables, "notes": notes,
            "sources": [{"name": k, "url": v} for k, v in src.items()]}


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
