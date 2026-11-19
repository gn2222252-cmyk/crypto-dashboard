"""Historical study "signs_with_confirmation": step 3 of the owner's plan ("if one indicator cannot do it, does adding a
parameter do it?"), applied to the bull / bear turn signs.

Background (measured in earlier studies, quoted, not re-measured here): step 1 (signs_before_turns) observed which of
43 series sat at an extreme before bear starts and bull starts. Step 2 (follow_turn_signs) acted on the signs alone:
the forward-chosen bear signs reached 5 of 6 before 2 of 5 usable tops (2017-12, 2021-04) but 55-90 days before the
top, with price still 41%-69% below it; standing aside then missed rises and the account ended at 14% of ma50's.
ma50 (hold while the close is above its 50-day average) left 4-26 days after each top. The measured problem is timing,
so the added parameter here is a PRICE CONFIRMATION: the signs only change how the trend rule behaves; a price event
always makes the move. Earlier price-only attempts (crash_timing, exit_entry_combo) left earlier but bought back too
late; whipsaw_anatomy found 70% of ma50 exits are whipsaws.

Everything below was written before any result of this study was seen. Anything added later is labelled.

Data, ranks, turns, sign sets (reused from follow_turn_signs / signs_before_turns, imported, not copied):
  * Daily close te._daily_close; the 43 series of signs_before_turns._assemble, lagged to publication; downloads go to
    this study's own store cache/_store/signs_with_confirmation/ (seeded read-only from signs_before_turns' and
    follow_turn_signs' stores). Ranks, M30, LEV, turns, confirmation, FORWARD / LOTO / ALL six-sign sets and the daily
    BEAR_COUNT / BULL_COUNT = follow_turn_signs._sets. FORWARD counts exist from the third confirmed turn of each type
    (bear and bull both defined from 2013-12-18); an undefined count counts as 0 (the rule is then plain ma50).
States (known at close t):
  BEAR_ON(t) = BEAR_COUNT >= 4 on any day t-119..t;  BULL_ON(t) = BULL_COUNT >= 4 on any day t-119..t.
  Robustness (FORWARD, exam only, no placebo): ">=5" = count >= 5 with the 120-day memory; "60d" = >= 4 with a
  60-day memory.
Price inputs on each series (BTC daily, each of the 24 BTC clocks, each other asset): ma = ma50 on (main_signal_check
  _targets), a20 = close above its 20-day average (supply_demand._above20), hi20 = close above the highest of the 20
  previous closes, lo30 = lowest of the 30 previous closes. Positions are decided at a close and held from the next.
Rules (the signs alone never move the position; a price event does):
  C1 faster exit when warned: plain ma50 when not BEAR_ON; while BEAR_ON hold only when the close is above its 50-day
     AND its 20-day average. Reading fixed now: "leave as soon as the close is below its 20-day average, come back only
     when the close is back above its 50-day average" - a close above the 50-day but below the 20-day average is still
     a "leave" day, so the combined condition is ma AND a20 (identical to crash_timing's a20_50 while warned).
  C2 trailing stop when warned: plain ma50; while BEAR_ON also leave when the close is <= 85% of the anchor = highest
     close since the later of (the day this BEAR_ON stretch began, the day the current holding began); after such a
     stop stay out until ma is on AND hi20; if BEAR_ON ends while stopped, plain ma50 again. (The anchor restarts at an
     entry so that a re-entry cannot be stopped out again the same day by an old high.)
  C3 slower re-entry when warned: plain ma50 exits; while BEAR_ON an entry needs ma on for 10 consecutive closes
     (a position already held stays while ma is on).
  C4 faster entry when primed: plain ma50; while BULL_ON and ma is off, enter on a close with a20 on, with a stop at
     lo30 of the entry day; leave the early position on a close below that stop; after a stop, a new early entry needs
     a20 to have been off at least once; ma50 takes over once it turns on; if BULL_ON ends while ma is off, cash.
  C5 = C1 and C4 together; when both states are on, BEAR_ON wins (C1's rule applies).
  Twins without sign information (state always on): T1 = ma AND a20 always; T2 = 15% trailing stop always (anchor =
  highest close since the current holding began); T3 = 10-day re-entry delay always; T4 = early entry with stop always.
Fair exam: follow_turn_signs' exam (main_signal_check's four checks: BTC daily, 108 monthly starts 2015-01..2023-12 at
  0.10%; 72 monthly starts 2018-01..2023-12 x 24 closing hours; the ordinary person, 10 hours later, 0.25% cost, 10% of
  entries missed, 200 simulations per clock; ETH SOL BNB XRP MSTR each with its own price inputs and BTC's states as a
  market-wide state, a clock close on UTC date E and MSTR's date D using the states of E-1 / D-1). ma50 re-scored on
  the same starts, and every rule and twin on the same starts (all starts are after 2013-12-18). Ranking score = worst
  of the four "beats holding" shares with main_signal_check's 95% loss eligibility. Median / worst deepest loss over
  the clock textbook cells. Time in market = mean BTC daily position from 2015-01-01; switches per year = position
  changes per year on BTC daily from 2015-01-01. Calendar years = BTC daily (0.10%) rule return minus ma50's.
  Placebo (FORWARD and LOTO, C1-C5 only - the twins have no sign input): BEAR_ON and BULL_ON together circularly
  shifted by k (uniform integer in [365, span - 365], span from the first defined count) against price, 200 draws,
  fixed seed; exam with 20 ordinary-person simulations per clock, the real rule re-scored the same way; placebo share =
  share of draws scoring at least the real one; p = (1 + hits) / (1 + draws).
  "Supported" (FORWARD only): C1, C2, C3 - ranking >= ma50's AND median deepest loss shallower than ma50's AND placebo
  <= 5% AND ranking strictly above its twin's (eligibility first, as _score_ge); C4 - ranking >= ma50's AND median
  deepest loss no more than 5 points worse AND placebo <= 5% AND strictly above T4; C5 - ranking >= ma50's AND median
  deepest loss no worse AND placebo <= 5%. Holm at 5% across the five placebo p-values; expected passes by luck 0.25.
Plain answers (BTC daily, UTC 00:00, no costs; FORWARD with LOTO next to it):
  "照著出場，少不少賠？" For each bear turn T (2013-04 twice): BEAR_ON at T and since when; for ma50, C1, C2, C3 the first
    exit at or after T (days, price vs the top close; or "already out at T" with the last exit before T); the following
    fall = T to the next bull turn's low (the lowest close so far if none, marked unfinished); share of that fall
    avoided = (rule growth - price growth) / (1 - price growth) over T..low (whipsaws included); re-entries during the
    fall; the first entry at or after the low (days, % above the low close). Then every spell with C1 (C2) out and ma50
    in: count, number followed by a fall / a rise (price from the spell's first close to the close after it), average
    fall avoided, average rise missed, net = rule equity / ma50 equity - 1 over all spells (exact: these rules only
    hold when ma50 holds). 2018 and 2022 from the calendar-year table.
  "照著進場，賺不賺？" For each bull turn B: BULL_ON at B; ma50's first on-day at or after B (% above the bottom close);
    C4's first early entry in B-90 .. that day (% vs bottom); C4's growth from its early entries in that window
    (= the gain captured before ma50, ma50 being in cash there); every early entry (C4 on, ma50 off): count, share
    ended higher, average gain of those up, average loss of those stopped, net (product).
Tops without the sign warning (2019-06, 2021-11, 2025-10). Report only: days C1-C3 differ from ma50 within T-180..T+180;
  and three alternative warning states fixed now (M30 = the 30-day mean rank of follow_turn_signs, known at t; "near the
  high" = close >= 85% of the highest close of the last 365 days):
    W1 = Fed net liquidity M30 <= 33.3 AND near the high;  W2 = gold M30 >= 80 AND near the high;
    W3 = open interest M30 >= 80 (where data exists).
  At every bear turn: on at T, days on in T-90..T; ordinary days (>= 90 days from every bear turn, where defined):
  share of days on. No rule; any reading is a guess (推測).
Today: "比特幣：50 日均線｜C1｜C4" with FORWARD states, whether BEAR_ON / BULL_ON are on and since when. Description,
  not a forecast.
Limits stated in the notes: about 8 turns of each type, 5 usable for FORWARD; BEAR_ON (FORWARD) can differ from off
  around very few tops; the signs are mostly valuation measures that move together; revised series; the 120-day
  memory, 4-of-6, 20-day average and 15% stop were fixed by reasoning, no alternatives tried beyond the two robustness
  lines. Not advice.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import follow_turn_signs as fts
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa

ID = "signs_with_confirmation"
DAY = pd.Timedelta(days=1)
MEM, LEVEL = 120, 4
TRAIL, DELAY, HI_N, LO_N = 0.15, 10, 20, 30
NEAR_HI, W_LIQ, W_TOP = 0.85, 100 / 3, 80
LOOK, AWAY = 90, 90
DRAWS, PL_SIMS, SHIFT_MIN, CHUNK = 200, 20, 365, 25
SEED = 20261121
COST, SIMS, P_COST, P_MISS = msc.COST, msc.SIMS, msc.P_COST, msc.P_MISS
FIRST = pd.Timestamp(msc.DAILY_STARTS[0])
ALPHA, LOSS_SLACK = 0.05, 5.0
RULES = ("c1", "c2", "c3", "c4", "c5")
TWINS = ("t1", "t2", "t3", "t4")
TWIN_OF = {"c1": "t1", "c2": "t2", "c3": "t3", "c4": "t4"}
VERS = ("fwd", "loto", "all")
VSHORT = {"fwd": "往前", "loto": "留一", "all": "全部（參考）"}
VNAME = {"fwd": "往前（只用當時已確認的轉折挑徵兆）", "loto": "留一（拿掉要考的那次轉折）", "all": "全部（樣本內，只當參考）"}
RSHORT = {"ma50": "50 日均線", "c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "c5": "C5",
          "t1": "T1", "t2": "T2", "t3": "T3", "t4": "T4"}
RNAME = {"c1": "C1 警告時提早出場：空頭狀態下，收盤跌破 20 日均線就出場，要回到 50 日和 20 日均線之上才回來",
         "c2": "C2 警告時移動停損：空頭狀態下，收盤比狀態開始（或這次上車）以來最高收盤低 15% 就出場，要 50 日均線持有而且創 20 日新高才回來",
         "c3": "C3 警告時晚點買回：空頭狀態下，出場後要連續 10 天收在 50 日均線上才買回",
         "c4": "C4 預備時提早進場：多頭狀態下，50 日均線還沒持有、收盤站上 20 日均線就先進場，停損設在前 30 天最低收盤",
         "c5": "C5＝C1 加 C4（兩個狀態同時亮時，空頭優先）",
         "t1": "T1（C1 的對照）：不看徵兆，永遠用 50 日＋20 日均線",
         "t2": "T2（C2 的對照）：不看徵兆，永遠用 15% 移動停損（從這次上車以來最高收盤算）",
         "t3": "T3（C3 的對照）：不看徵兆，出場後永遠要連續 10 天在 50 日均線上才買回",
         "t4": "T4（C4 的對照）：不看徵兆，永遠在 50 日均線下站上 20 日均線就先進場、帶停損"}
KI = {"ma50": 1, "c1": 150, "c2": 151, "c3": 152, "c4": 153, "c5": 154, "t1": 155, "t2": 156, "t3": 157, "t4": 158}
FLAGGED, MISSED = fts.FLAGGED, fts.MISSED

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID

_d = te._d
_ok = sbt._ok
_pc, _sh, _px = fts._pc, fts._sh, fts._px


# ---------- own store ----------

@contextmanager
def _store():
    """Every loader writes into cache/_store/signs_with_confirmation/; the stores of signs_before_turns and
    follow_turn_signs are only read (copied in when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for src_dir in (REAL_STORE / sbt.ID, REAL_STORE / fts.ID):
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


# ---------- states and price inputs ----------

def _state(cnt: np.ndarray, level: int, mem: int) -> np.ndarray:
    hit = (np.nan_to_num(cnt, nan=0.0) >= level).astype(float)
    return pd.Series(hit).rolling(mem, min_periods=1).max().values


def _feat(c: np.ndarray, ma: np.ndarray, a20: np.ndarray) -> dict:
    c = np.asarray(c, dtype=float)
    s = pd.Series(c)
    hi = s.shift(1).rolling(HI_N, min_periods=HI_N).max().values
    lo = s.shift(1).rolling(LO_N, min_periods=LO_N).min().values
    return {"ma": np.nan_to_num(np.asarray(ma, dtype=float)) > 0.5, "a20": np.nan_to_num(np.asarray(a20, dtype=float)) > 0.5,
            "c": c, "hi20": np.where(np.isfinite(hi), c > hi, False), "lo30": lo}


def _machine(rule: str, F: dict, bear: np.ndarray, bull: np.ndarray) -> np.ndarray:
    """Positions (T, K) of one rule for K parallel series. F arrays and states are (T, K); states are booleans."""
    ma, a20, c, hi20, lo30 = F["ma"], F["a20"], F["c"], F["hi20"], F["lo30"]
    T, K = ma.shape
    if rule == "ma50":
        return ma.astype(float)
    if rule in ("c1", "t1"):
        W = bear if rule == "c1" else np.ones_like(ma)
        return (ma & (a20 | ~W)).astype(float)
    out = np.zeros((T, K))
    if rule in ("c3", "t3"):
        run = np.zeros(K)
        p = np.zeros(K, bool)
        for t in range(T):
            run = np.where(ma[t], run + 1, 0)
            W = bear[t] if rule == "c3" else np.ones(K, bool)
            p = ma[t] & (p | (run >= DELAY) | ~W)
            out[t] = p
        return out
    if rule in ("c2", "t2"):
        p = np.zeros(K, bool)
        latch = np.zeros(K, bool)
        wprev = np.zeros(K, bool)
        anchor = np.array(c[0], dtype=float)
        ones = np.ones(K, bool)
        for t in range(T):
            W = bear[t] if rule == "c2" else ones
            ct = c[t]
            re = latch & ma[t] & hi20[t]
            pn = np.where(latch, re, ma[t])
            entered = pn & ~p
            neww = (W & ~wprev) if rule == "c2" else np.zeros(K, bool)
            anchor = np.where(entered | neww, ct, np.maximum(anchor, ct))
            stop = W & pn & (ct <= (1 - TRAIL) * anchor)
            pn = pn & ~stop
            latch = ((latch & ~re) | stop) & W
            pn = np.where(W, pn, ma[t])
            p, wprev = pn, W
            out[t] = p
        return out
    if rule in ("c4", "t4", "c5"):
        p = np.zeros(K, bool)
        early = np.zeros(K, bool)
        slatch = np.zeros(K, bool)
        stop = np.full(K, np.nan)
        ones, zeros = np.ones(K, bool), np.zeros(K, bool)
        for t in range(T):
            U = ones if rule == "t4" else bull[t]
            m, a, ct = ma[t], a20[t], c[t]
            slatch = slatch & a
            with np.errstate(invalid="ignore"):
                stopped = early & (ct < stop)
            keep = early & ~stopped
            enter = ~early & a & ~slatch
            stop = np.where(enter, lo30[t], stop)
            free = ~m & U
            p = m | (free & (keep | enter))
            early = free & (keep | enter)
            slatch = np.where(free, slatch | stopped, np.where(m, slatch, zeros)) & U
            if rule == "c5":
                B = bear[t]
                p = np.where(B, m & a, p)
                early = early & ~B
                slatch = slatch & ~B
            out[t] = p
        return out
    raise ValueError(rule)


# ---------- the exam ----------

class _Exam(pi._Exam):
    """main_signal_check's four checks for ma50, C1-C5 and T1-T4; BTC's daily states are the market state."""

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first)
        npre = len(pre)
        self.F = _feat(close.values, self.ma, self.a20)
        cols = {k: [] for k in ("ma", "a20", "c", "hi20", "lo30")}
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            f = _feat(c, np.zeros(len(c)), np.zeros(len(c)))
            for k in ("c", "hi20", "lo30"):
                cols[k].append(f[k][npre:][:self.days])
            cols["ma"].append(self.cma[h] > 0.5)
            cols["a20"].append(self.ca20[h] > 0.5)
        self.FC = {k: np.stack(v, axis=1) for k, v in cols.items()}  # (days, 24)
        self.FO = {key: _feat(o["close"].values, o["ma"], o["a20"]) for key, o in self.other.items()}
        idx = close.index
        self.map_d = np.arange(len(idx))
        self.map_c = idx.get_indexer(pd.DatetimeIndex(self.m.dates) - DAY)
        self.map_o = {key: idx.get_indexer(o["close"].index - o["lag"] * DAY) for key, o in self.other.items()}

    @staticmethod
    def _take(S: np.ndarray, m: np.ndarray) -> np.ndarray:
        """States (n, D) picked at index map m (-1 -> off)."""
        out = S[np.clip(m, 0, None)] > 0.5
        out[m < 0] = False
        return out

    def positions(self, rule: str, BE: np.ndarray, BU: np.ndarray) -> dict:
        """BE, BU: (n, D) state arrays on the BTC daily index. Returns daily (n, D), clock (days, 24, D), other."""
        D = BE.shape[1]
        Fd = {k: np.repeat(v[:, None], D, axis=1) for k, v in self.F.items()}
        daily = _machine(rule, Fd, self._take(BE, self.map_d), self._take(BU, self.map_d))
        bc, uc = self._take(BE, self.map_c), self._take(BU, self.map_c)
        clock = np.zeros((self.days, 24, D))
        for a in range(0, D, CHUNK):
            b = min(D, a + CHUNK)
            Fc = {k: np.tile(v, (1, b - a)) for k, v in self.FC.items()}
            w = _machine(rule, Fc, np.repeat(bc[:, a:b], 24, axis=1), np.repeat(uc[:, a:b], 24, axis=1))
            clock[:, :, a:b] = w.reshape(self.days, b - a, 24).transpose(0, 2, 1)
        oth = {}
        for key, f in self.FO.items():
            Fo = {k: np.repeat(v[:, None], D, axis=1) for k, v in f.items()}
            oth[key] = _machine(rule, Fo, self._take(BE, self.map_o[key]), self._take(BU, self.map_o[key]))
        return {"daily": daily, "clock": clock, "other": oth}

    def score(self, pos: np.ndarray, wc: np.ndarray, wo: dict, ki: int, sims: int) -> dict:
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, msc.YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = wc[:, h]
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
            w = wo[key]
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        m = self.close.index >= FIRST
        sw = float(np.abs(np.diff(pos[m])).sum() / (m.sum() / 365.25))
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values()),
             "time": float(pos[m].mean()), "switch": sw}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x

    def summary(self, rule: str, be: np.ndarray, bu: np.ndarray, sims: int = SIMS) -> tuple[dict, np.ndarray]:
        P = self.positions(rule, be[:, None], bu[:, None])
        x = self.score(P["daily"][:, 0], P["clock"][:, :, 0], {k: v[:, 0] for k, v in P["other"].items()},
                       KI[rule], sims)
        return x, P["daily"][:, 0]


def _beats(x: dict, y: dict) -> bool:
    """x strictly above y (eligibility first)."""
    if y["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not y["eligible"]:
        return True
    return x["low"] > y["low"] + 1e-12


def _supported(rule: str, x: dict, base: dict, twin: dict | None, pl: float | None) -> dict:
    rank_ok = sd._score_ge(x, base)
    if rule in ("c1", "c2", "c3"):
        loss_ok = x["c_deep_med"] > base["c_deep_med"] + 1e-9
    elif rule == "c4":
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK
    else:
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - 1e-9
    pl_ok = pl is not None and pl <= ALPHA
    tw_ok = True if twin is None else _beats(x, twin)
    return {"rank": rank_ok, "loss": loss_ok, "pl": pl_ok, "twin": tw_ok, "has_twin": twin is not None,
            "ok": rank_ok and loss_ok and pl_ok and tw_ok}


# ---------- plain answers ----------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    return fts._runs(np.asarray(mask, bool))


def _growth(pos: np.ndarray, c: np.ndarray, a: int, b: int) -> float:
    return fts._eq(pos, c, a, b)


def _exits(pos: np.ndarray) -> np.ndarray:
    return np.flatnonzero((pos[:-1] > 0.5) & (pos[1:] < 0.5)) + 1


def _entries(pos: np.ndarray) -> np.ndarray:
    return np.flatnonzero((pos[:-1] < 0.5) & (pos[1:] > 0.5)) + 1


def _since(state: np.ndarray, i: int) -> int | None:
    """First day of the run of state==on that contains i (None if off at i)."""
    if state[i] < 0.5:
        return None
    j = i
    while j > 0 and state[j - 1] > 0.5:
        j -= 1
    return j


def _top_row(c: np.ndarray, T: int, L: int, pos: np.ndarray) -> dict:
    n = len(c)
    ex = _exits(pos)
    if pos[T] < 0.5:
        prev = ex[ex <= T]
        first = {"already": True, "i": int(prev[-1]) if len(prev) else None}
    else:
        nxt = ex[ex > T]
        first = {"already": False, "i": int(nxt[0]) if len(nxt) else None}
    gp = c[L] / c[T]
    gr = _growth(pos, c, T, L)
    avoided = (gr - gp) / (1 - gp) if gp < 1 else None
    en = _entries(pos)
    during = int(((en > T) & (en <= L)).sum())
    after = en[en >= L]
    if pos[L] > 0.5:
        # holding at the low: the entry that put it in
        before = en[en <= L]
        re = int(before[-1]) if len(before) else None
    else:
        re = int(after[0]) if len(after) else None
    return {"first": first, "avoided": avoided, "during": during, "re": re, "gr": gr}


def _out_spells(c: np.ndarray, pm: np.ndarray, pr: np.ndarray) -> list[dict]:
    n = len(c)
    out = []
    for s, e in _runs((pm > 0.5) & (pr < 0.5)):
        x = min(e + 1, n - 1)
        out.append({"s": s, "e": e, "miss": c[x] / c[s] - 1, "low": float(c[s:x + 1].min() / c[s] - 1),
                    "open": e + 1 >= n})
    return out


def _spell_sum(sp: list[dict]) -> dict:
    fell = [s["miss"] for s in sp if s["miss"] < 0]
    rose = [s["miss"] for s in sp if s["miss"] > 0]
    net = float(np.prod([1 / (1 + s["miss"]) for s in sp]) - 1) if sp else 0.0
    return {"n": len(sp), "fell": fell, "rose": rose, "net": net,
            "avg_fell": float(np.mean(fell)) if fell else None, "avg_rose": float(np.mean(rose)) if rose else None}


def _early(c: np.ndarray, pm: np.ndarray, p4: np.ndarray, bull: np.ndarray) -> list[dict]:
    n = len(c)
    out = []
    for s, e in _runs((p4 > 0.5) & (pm < 0.5)):
        x = e + 1
        if x >= n:
            out.append({"s": s, "x": n - 1, "gain": c[-1] / c[s] - 1, "why": "仍在進行", "open": True})
            continue
        if pm[x] > 0.5:
            why = "50 日均線接手"
        elif bull[x] > 0.5:
            why = "停損"
        else:
            why = "多頭狀態結束"
        out.append({"s": s, "x": x, "gain": c[x] / c[s] - 1, "why": why, "open": False})
    return out


def _early_sum(ea: list[dict]) -> dict:
    done = [e for e in ea if not e["open"]]
    g = [e["gain"] for e in done]
    up = [x for x in g if x > 0]
    st = [e["gain"] for e in done if e["why"] == "停損"]
    return {"n": len(done), "up": len(up), "avg_up": float(np.mean(up)) if up else None,
            "avg_stop": float(np.mean(st)) if st else None, "n_stop": len(st),
            "net": float(np.prod([1 + x for x in g]) - 1) if g else None}


# ---------- computation ----------

def _compute() -> dict:
    t0 = time.time()
    with _store():
        close = te._daily_close()
        series, dropped, problems, src, _extra = sbt._assemble(close)
    t_data = time.time() - t0
    n = len(close)
    idx = close.index
    c = close.values.astype(float)
    m30, lev = {}, {}
    for k, s in series.items():
        r = at._trail_pct(s.values, sbt.WIN, sbt.WIN_MIN) * 100
        W = sbt._windows(r)
        m30[k], lev[k] = W["m30"], W["lev"]
    turns = sbt._turns(close)
    S = fts._sets(close, lev, m30, turns)
    cnt = {v: (S["bear"]["cnt"][v], S["bull"]["cnt"][v]) for v in VERS}
    first_def = {v: int(np.argmax(np.isfinite(cnt[v][0]) & np.isfinite(cnt[v][1]))) for v in VERS}
    ST = {v: (_state(cnt[v][0], LEVEL, MEM), _state(cnt[v][1], LEVEL, MEM)) for v in VERS}
    ROB = {"ge5": (_state(cnt["fwd"][0], 5, MEM), _state(cnt["fwd"][1], 5, MEM)),
           "m60": (_state(cnt["fwd"][0], LEVEL, 60), _state(cnt["fwd"][1], LEVEL, 60))}

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
    zero = np.zeros(n)
    base, pos_ma = ex.summary("ma50", zero, zero)
    X, POS = {}, {"ma50": pos_ma}
    for v in VERS:
        X[v] = {}
        for r in RULES:
            X[v][r], POS[(v, r)] = ex.summary(r, *ST[v])
    TW = {}
    for r in TWINS:
        TW[r], POS[r] = ex.summary(r, zero, zero)
    RB = {k: {r: ex.summary(r, *ST2)[0] for r in RULES} for k, ST2 in ROB.items()}
    t_exam = time.time()

    # placebo: states shifted together, all draws at once per rule
    PL = {}
    rng = np.random.default_rng(SEED)
    for v in ("fwd", "loto"):
        be, bu = ST[v]
        start = first_def[v]
        span = n - start
        ks = [int(rng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, span - SHIFT_MIN))) for _ in range(DRAWS)]
        BE = np.repeat(be[:, None], DRAWS, axis=1)
        BU = np.repeat(bu[:, None], DRAWS, axis=1)
        for j, k in enumerate(ks):
            BE[start:, j] = np.roll(be[start:], k)
            BU[start:, j] = np.roll(bu[start:], k)
        PL[v] = {}
        for r in RULES:
            small, _ = ex.summary(r, be, bu, PL_SIMS)
            P = ex.positions(r, BE, BU)
            hits = 0
            for j in range(DRAWS):
                x = ex.score(P["daily"][:, j], P["clock"][:, :, j], {kk: vv[:, j] for kk, vv in P["other"].items()},
                             KI[r], PL_SIMS)
                hits += int(sd._score_ge(x, small))
            PL[v][r] = hits / DRAWS
    t_pl = time.time()
    ps = {r: (1 + PL["fwd"][r] * DRAWS) / (1 + DRAWS) for r in RULES}
    holm = fts._holm(ps)
    sup = {r: _supported(r, X["fwd"][r], base, TW.get(TWIN_OF.get(r, ""), None) if r in TWIN_OF else None,
                         PL["fwd"][r]) for r in RULES}

    cal = {"ma50": wa._cal_years(close, pos_ma)}
    for k, p in POS.items():
        if k != "ma50":
            cal[k] = wa._cal_years(close, p)

    # plain answers
    bear_t = S["bear"]["turns"]
    bull_t = S["bull"]["turns"]
    bull_i = sorted(t["i"] for t in turns["bull"])
    PA = {}
    for v in ("fwd", "loto"):
        be, bu = ST[v]
        tops = []
        for t in bear_t:
            T = t["i"]
            nx = [b for b in bull_i if b > T]
            if nx:
                L, done = nx[0], True
            else:
                L, done = T + int(np.argmin(c[T:])), False
            row = {"t": t, "L": L, "done": done, "fall": c[L] / c[T] - 1, "on": be[T] > 0.5,
                   "since": _since(be, T), "max": (float(np.nanmax(cnt[v][0][max(0, T - LOOK):T + 1]))
                                                    if np.isfinite(cnt[v][0][max(0, T - LOOK):T + 1]).any() else None)}
            for r in ("ma50", "c1", "c2", "c3"):
                p = pos_ma if r == "ma50" else POS[(v, r)]
                row[r] = _top_row(c, T, L, p)
            for r in ("c1", "c2", "c3"):
                a, b = max(0, T - 180), min(n, T + 181)
                row[r]["diffdays"] = int((np.abs(POS[(v, r)][a:b] - pos_ma[a:b]) > 0.5).sum())
            tops.append(row)
        sp = {r: _out_spells(c, pos_ma, POS[(v, r)]) for r in ("c1", "c2", "c3")}
        ea = _early(c, pos_ma, POS[(v, "c4")], bu)
        bots = []
        for t in bull_t:
            B = t["i"]
            on = pos_ma > 0.5
            ma_on = next((i for i in range(B, n) if on[i]), None)
            lim = ma_on if ma_on is not None else n
            eps = [e for e in ea if B - LOOK <= e["s"] < lim]
            gwin = float(np.prod([1 + e["gain"] for e in eps]) - 1) if eps else 0.0
            w = bu[max(0, B - LOOK):lim]
            bots.append({"t": t, "ma_on": ma_on, "eps": eps, "g": gwin, "on": bu[B] > 0.5,
                         "on_win": int((w > 0.5).sum()), "win": len(w)})
        PA[v] = {"tops": tops, "spells": sp, "sum": {r: _spell_sum(sp[r]) for r in sp}, "early": ea,
                 "esum": _early_sum(ea), "bots": bots}

    # alternative warnings W1-W3
    hi365 = pd.Series(c).rolling(365, min_periods=200).max().values
    near = np.where(np.isfinite(hi365), c >= NEAR_HI * hi365, False)
    WS = {}
    if "netliq" in m30:
        a = m30["netliq"]
        WS["W1"] = np.where(np.isfinite(a), ((a <= W_LIQ) & near).astype(float), np.nan)
    if "gold" in m30:
        a = m30["gold"]
        WS["W2"] = np.where(np.isfinite(a), ((a >= W_TOP) & near).astype(float), np.nan)
    if "oi" in m30:
        a = m30["oi"]
        WS["W3"] = np.where(np.isfinite(a), (a >= W_TOP).astype(float), np.nan)
    bear_all = [t["i"] for t in turns["bear"]]
    ordinary = np.ones(n, bool)
    for T in bear_all:
        ordinary[max(0, T - AWAY + 1):min(n, T + AWAY)] = False
    WR = {}
    for k, a in WS.items():
        dfn = np.isfinite(a)
        o = ordinary & dfn
        WR[k] = {"ord": float(np.nanmean(a[o])) if o.any() else None, "ord_n": int(o.sum()),
                 "first": idx[int(np.argmax(dfn))] if dfn.any() else None}
    return {"close": close, "series": series, "dropped": dropped, "problems": problems, "src": src, "m30": m30,
            "turns": turns, "S": S, "cnt": cnt, "ST": ST, "first_def": first_def, "ex": ex, "base": base, "X": X,
            "TW": TW, "RB": RB, "PL": PL, "ps": ps, "holm": holm, "sup": sup, "POS": POS, "cal": cal, "PA": PA,
            "WS": WS, "WR": WR, "ordinary": ordinary,
            "secs": {"data": t_data, "exam": t_exam - t0, "placebo": t_pl - t_exam, "all": time.time() - t0}}


# ---------- tables ----------

def _rank_cell(x: dict) -> str:
    return _sh(x["low"]) + ("" if x["eligible"] else "（虧損門檻未過）")


def _exam_tables(R: dict) -> list[dict]:
    X, base, PL, sup, holm, TW = R["X"], R["base"], R["PL"], R["sup"], R["holm"], R["TW"]
    out = []
    rows = [["50 日均線（基準）", "—", _rank_cell(base), f"{base['c_deep_med']:.0f}%｜{base['c_deep_worst']:.0f}%",
             _sh(base["time"]), f"{base['switch']:.1f}", "—", "—", "—"]]
    for r in RULES:
        for v in VERS:
            x = X[v][r]
            pl = PL.get(v, {}).get(r)
            if v == "fwd":
                verdict = "支持" if sup[r]["ok"] else "不支持"
                hl = "通過" if holm.get(r) else "沒通過"
            else:
                verdict, hl = "只報告", "—"
            rows.append([RSHORT[r], VSHORT[v], _rank_cell(x), f"{x['c_deep_med']:.0f}%｜{x['c_deep_worst']:.0f}%",
                         _sh(x["time"]), f"{x['switch']:.1f}", _sh(pl) if pl is not None else "—", hl, verdict])
    for r in TWINS:
        x = TW[r]
        rows.append([RSHORT[r], "不看徵兆（對照）", _rank_cell(x), f"{x['c_deep_med']:.0f}%｜{x['c_deep_worst']:.0f}%",
                     _sh(x["time"]), f"{x['switch']:.1f}", "—", "—", "對照"])
    ex = R["ex"]
    out.append({"caption": "公平考試（每條規則都和 50 日均線用同一批起算日：比特幣 2015-01 到 2023-12 每月一個起算日共 "
                           f"{len(ex.starts)} 個；2018-01 到 2023-12 每月×24 個收盤時間共 {len(ex.cstarts) * 24:,} 格；"
                           f"一般人做法；其他幣 {sum(len(o['starts']) for o in ex.other.values())} 個起算日。「往前」的狀態從 "
                           f"{_d(R['close'].index[R['first_def']['fwd']])} 起才有，所有起算日都在那之後）。排名分數＝四項檢查裡"
                           "「每年報酬贏過一直抱著」比例最低的那一項。最深虧損＝起算日×收盤時間各格從起算日到現在帳戶從高點掉最多的"
                           "幅度：中位數｜最深的一格。持有時間＝2015 年起平均持有比例。每年換手＝2015 年起每年進出幾次。安慰劑＝把空頭／"
                           "多頭狀態整條錯開 365 天以上（200 次）後，分數不輸真的那條的比例（越低越好；一般人做法每個收盤時間只模擬 20 次，"
                           "真的那條也用 20 次重算）。Holm＝五條「往前」規則一起校正後是否仍在 5% 以下。T1–T4 是不看徵兆、永遠用同一個"
                           "價格規則的對照。",
                "columns": ["規則", "做法", "排名分數", "最深虧損：中位數｜最深", "持有時間", "每年換手", "安慰劑", "Holm",
                            "判定"], "rows": rows})
    rows = []
    for r in RULES:
        s = sup[r]
        tw = TW.get(TWIN_OF.get(r, ""))
        rows.append([RNAME[r], f"{_sh(X['fwd'][r]['low'])}（均線 {_sh(base['low'])}）{'✓' if s['rank'] else '✗'}",
                     f"{X['fwd'][r]['c_deep_med']:.1f}%（均線 {base['c_deep_med']:.1f}%）{'✓' if s['loss'] else '✗'}",
                     f"{_sh(PL['fwd'][r])} {'✓' if s['pl'] else '✗'}",
                     (f"{_sh(tw['low'])} {'✓' if s['twin'] else '✗'}" if tw is not None else "不需要"),
                     "通過" if holm.get(r) else "沒通過", "支持" if s["ok"] else "不支持"])
    out.append({"caption": "事先定的「支持」標準，逐項看（只看「往前」）。C1–C3：排名不輸均線、最深虧損中位數比均線淺、安慰劑 ≤5%、"
                           "排名高過自己的對照。C4：排名不輸、最深虧損中位數最多深 5 點、安慰劑 ≤5%、高過 T4。C5：排名不輸、最深虧損"
                           "不更深、安慰劑 ≤5%。✓＝這一項過，✗＝沒過。五條各用 5% 門檻，光靠運氣平均約 0.25 條會過。",
                "columns": ["規則", "排名分數", "最深虧損中位數", "安慰劑", "對照的排名", "Holm", "判定"], "rows": rows})
    rows = [[RSHORT[r], f"{_sh(R['X']['fwd'][r]['low'])}｜{R['X']['fwd'][r]['c_deep_med']:.0f}%",
             f"{_sh(R['RB']['ge5'][r]['low'])}｜{R['RB']['ge5'][r]['c_deep_med']:.0f}%",
             f"{_sh(R['RB']['m60'][r]['low'])}｜{R['RB']['m60'][r]['c_deep_med']:.0f}%"] for r in RULES]
    out.append({"caption": "穩健度（只看「往前」，沒做安慰劑）：狀態換一種事先定的寫法，排名分數｜最深虧損中位數。主版本＝6 個徵兆裡 ≥4 個、"
                           "記住 120 天；「≥5」＝要 ≥5 個；「60 天」＝只記 60 天。50 日均線是 "
                           f"{_sh(base['low'])}｜{base['c_deep_med']:.0f}%。",
                "columns": ["規則", "主版本（≥4、120 天）", "≥5、120 天", "≥4、60 天"], "rows": rows})
    rows = []
    for k, x in [("ma50", base)] + [(r, X["fwd"][r]) for r in RULES] + [(r, TW[r]) for r in TWINS]:
        rows.append([RSHORT[k]] + [_sh(v) for v in x["checks"]] + [f"{x['c_edge']:+.1f}"])
    out.append({"caption": "四項檢查各自的「每年報酬贏過一直抱著」比例（C 規則為「往前」），以及起算日×收盤時間每年報酬比一直抱著多幾個"
                           "百分點（中位數）。其他幣用比特幣的狀態當成整個市場的狀態。",
                "columns": ["規則", "比特幣起算日", "起算日×收盤時間", "一般人做法", "其他幣", "每年多幾點（中位數）"],
                "rows": rows})
    years = [y for y in sorted(R["cal"]["ma50"]) if y >= 2013]
    for v in ("fwd", "loto"):
        rows = []
        for y in years:
            m = R["cal"]["ma50"][y]
            rows.append([str(y), _pc(m)] + [f"{(R['cal'][(v, r)][y] - m) * 100:+.1f}" for r in RULES])
        out.append({"caption": f"每一年（{VNAME[v]}）：50 日均線那一年的報酬，以及每條規則減掉 50 日均線（百分點；比特幣 UTC 0 點日線、"
                               "每次買賣扣 0.10%）。正的＝那一年比 50 日均線好。今年算到最後一根收盤。",
                    "columns": ["年份", "50 日均線"] + [RSHORT[r] for r in RULES], "rows": rows})
    rows = []
    for y in years:
        m = R["cal"]["ma50"][y]
        rows.append([str(y), _pc(m)] + [f"{(R['cal'][r][y] - m) * 100:+.1f}" for r in TWINS])
    out.append({"caption": "每一年（不看徵兆的對照 T1–T4）：減掉 50 日均線的百分點。",
                "columns": ["年份", "50 日均線"] + [RSHORT[r] for r in TWINS], "rows": rows})
    return out


def _cell_exit(c: np.ndarray, idx, T: int, x: dict) -> str:
    f = x["first"]
    p = c[T]
    if f["i"] is None:
        return "頭部時已是現金" if f["already"] else "之後沒有出場"
    i = f["i"]
    s = f"{_d(idx[i])}（{i - T:+d} 天｜{_pc(c[i] / p - 1)}）"
    return ("頭部時已是現金，上次出場 " + s) if f["already"] else s


def _cell_fall(c: np.ndarray, idx, T: int, L: int, x: dict) -> str:
    re = x["re"]
    if re is None:
        rtxt = "還沒買回"
    elif re <= L:
        rtxt = f"低點時已在場（{_d(idx[re])} 進）"
    else:
        rtxt = f"{_d(idx[re])} 買回（比低點 {_pc(c[re] / c[L] - 1)}）"
    return f"避開 {_sh(x['avoided'])}｜中途買回 {x['during']} 次｜{rtxt}"


def _pa_tables(R: dict) -> list[dict]:
    idx, c = R["close"].index, R["close"].values.astype(float)
    out = []
    rows = []
    for v in ("fwd", "loto"):
        for x in R["PA"][v]["tops"]:
            T = x["t"]["i"]
            on = ("是（" + _d(idx[x["since"]]) + " 起）") if x["on"] else "否"
            rows.append([_d(x["t"]["date"]), VSHORT[v], on, "—" if x["max"] is None else f"{x['max']:.0f}／6",
                         _cell_exit(c, idx, T, x["ma50"]), _cell_exit(c, idx, T, x["c1"]),
                         _cell_exit(c, idx, T, x["c2"]), _cell_exit(c, idx, T, x["c3"])])
    out.append({"caption": "照著出場，少不少賠？・每一次走熊開始（頭部；2013 年 4 月有兩次）。「空頭狀態」＝過去 120 天內 6 個空頭徵兆曾經"
                           "≥4 個亮。每個規則在頭部當天或之後第一次出場的日子（離頭部幾天｜價格比頭部收盤）；如果頭部當天已經是現金，"
                           "寫上次出場的日子。",
                "columns": ["頭部日", "做法", "頭部時空頭狀態", "前 90 天最多亮幾個", "50 日均線出場", "C1 出場", "C2 出場",
                            "C3 出場"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        for x in R["PA"][v]["tops"]:
            T, L = x["t"]["i"], x["L"]
            low = f"{_d(idx[L])}（{_pc(x['fall'])}）" + ("" if x["done"] else "（還沒確認，用目前最低）")
            rows.append([_d(x["t"]["date"]), VSHORT[v], low] +
                        [_cell_fall(c, idx, T, L, x[r]) for r in ("ma50", "c1", "c2", "c3")])
    out.append({"caption": "照著出場・之後那段跌勢避開多少、買回付出多少。跌勢＝頭部收盤到下一次走牛開始的低點收盤。避開比例＝（規則在"
                           "頭部到低點的帳戶變化 − 價格變化）÷ 價格跌幅，中途被洗也算進去；100%＝全部避開，0%＝和一直抱著一樣。中途"
                           "買回＝頭部到低點之間又進場幾次。買回＝低點當天或之後第一次進場，比低點收盤高多少。不扣成本。",
                "columns": ["頭部日", "做法", "低點（跌幅）", "50 日均線", "C1", "C2", "C3"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        for r in ("c1", "c2", "c3"):
            s = R["PA"][v]["sum"][r]
            rows.append([RSHORT[r], VSHORT[v], str(s["n"]), f"{len(s['fell'])}｜{len(s['rose'])}",
                         _pc(s["avg_fell"], 1), _pc(s["avg_rose"], 1),
                         f"{(1 + s['net']) * 100:.0f}%"])
    out.append({"caption": "照著出場・全部「規則是現金、50 日均線持有」的時段（全期，比特幣日線，不扣成本）。之後跌｜漲＝這段從第一天收盤"
                           "到結束後那天收盤，價格跌了幾段、漲了幾段。平均少賠＝跌的那幾段平均跌幅；平均少賺＝漲的那幾段平均漲幅。"
                           "合計＝把這些時段全部加起來，規則的帳戶是 50 日均線的幾 %（>100% 表示少賠比少賺多）。",
                "columns": ["規則", "做法", "時段數", "之後跌｜漲", "平均少賠（跌幅）", "平均少賺（漲幅）", "合計：規則／均線"],
                "rows": rows})
    for r in ("c1", "c2"):
        sp = R["PA"]["fwd"]["spells"][r]
        rows = [[_d(idx[s["s"]]), _d(idx[s["e"]]) + ("（還在）" if s["open"] else ""), str(s["e"] - s["s"] + 1),
                 _px(c[s["s"]]), _pc(s["miss"], 1), _pc(s["low"], 1)] for s in sp]
        out.append({"caption": f"照著出場・{RSHORT[r]}（往前）每一段「{RSHORT[r]} 是現金、50 日均線持有」的時段。期間價格＝第一天收盤到"
                               "結束後那天收盤（負的＝少賠，正的＝少賺）；期間最低＝期間最低收盤比第一天低多少。不扣成本。",
                    "columns": ["開始", "結束", "天數", "開始價（美元）", "期間價格", "期間最低"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        for x in R["PA"][v]["bots"]:
            p = x["t"]["p"]
            mo = x["ma_on"]
            eps = x["eps"]
            first = eps[0] if eps else None
            rows.append([_d(x["t"]["date"]), VSHORT[v],
                         ("是" if x["on"] else "否") + f"（窗口內 {x['on_win']}／{x['win']} 天）",
                         "—" if mo is None else f"{_d(idx[mo])}（{_pc(c[mo] / p - 1)}）",
                         "沒有" if first is None else f"{_d(idx[first['s']])}（{_pc(c[first['s']] / p - 1)}）",
                         f"{len(eps)} 次｜{sum(e['gain'] > 0 for e in eps)} 次結束時較高" if eps else "0 次",
                         _pc(x["g"], 1) if eps else "—"])
    out.append({"caption": "照著進場，賺不賺？・每一次走牛開始（底部）。「多頭狀態」＝過去 120 天內 6 個多頭徵兆曾經 ≥4 個亮；窗口＝底部前 "
                           "90 天到 50 日均線第一次持有。C4 第一次提早進場的價格比底部收盤高多少；窗口內提早進場的結果合計＝在 50 日均線"
                           "進場前多賺（賠）多少（那段 50 日均線是現金）。不扣成本。",
                "columns": ["底部日", "做法", "底部時多頭狀態", "50 日均線第一次持有", "C4 第一次提早進場", "提早進場次數",
                            "均線進場前多賺"], "rows": rows})
    rows = []
    for v in ("fwd", "loto"):
        s = R["PA"][v]["esum"]
        rows.append([VSHORT[v], str(s["n"]), f"{s['up']} 次（{_sh(s['up'] / s['n']) if s['n'] else '—'}）",
                     _pc(s["avg_up"], 1), f"{s['n_stop']} 次，平均 {_pc(s['avg_stop'], 1)}", _pc(s["net"], 1)])
    out.append({"caption": "照著進場・C4 全部提早進場（C4 持有、50 日均線現金的時段，已結束的），全期，不扣成本。",
                "columns": ["做法", "次數", "結束時比進場高", "這些平均賺", "被停損", "全部連乘"], "rows": rows})
    ea = R["PA"]["fwd"]["early"]
    rows = [[_d(idx[e["s"]]), _px(c[e["s"]]), _d(idx[e["x"]]), str(e["x"] - e["s"]), _pc(e["gain"], 1), e["why"]]
            for e in ea]
    out.append({"caption": "照著進場・C4（往前）每一次提早進場：從進場收盤到結束那天收盤（50 日均線接手、停損、或多頭狀態結束）。不扣成本。",
                "columns": ["進場日", "進場價（美元）", "結束日", "天數", "這段漲跌", "怎麼結束"], "rows": rows})
    return out


def _w_tables(R: dict) -> list[dict]:
    idx, c = R["close"].index, R["close"].values.astype(float)
    WS, WR = R["WS"], R["WR"]
    be = R["ST"]["fwd"][0]
    keys = [k for k in ("W1", "W2", "W3") if k in WS]
    rows = []
    for t in R["turns"]["bear"]:
        T = t["i"]
        d = _d(t["date"])
        grp = "標到" if d in FLAGGED else ("漏掉" if d in MISSED else "—")
        cells = []
        for k in ("W1", "W2", "W3"):
            if k not in WS:
                cells.append("沒資料")
                continue
            a = WS[k]
            w = a[max(0, T - LOOK):T + 1]
            if not np.isfinite(a[T]):
                cells.append("沒資料")
                continue
            cells.append(f"{'亮' if a[T] > 0.5 else '沒亮'}｜前 90 天 {int(np.nansum(w))}／{int(np.isfinite(w).sum())} 天")
        rows.append([d, grp, "亮" if be[T] > 0.5 else "沒亮"] + cells)
    o = R["ordinary"].copy()
    o[:R["first_def"]["fwd"]] = False
    rows.append(["一般日子（離每次頭部 90 天以上）", "—", f"{_sh(float(np.mean(be[o])))} 的日子亮（{int(o.sum()):,} 天）"] +
                [(f"{_sh(WR[k]['ord'])} 的日子亮（{WR[k]['ord_n']:,} 天）" if k in WR and WR[k]["ord"] is not None else "沒資料")
                 for k in ("W1", "W2", "W3")])
    first = "、".join(f"{k} 從 {_d(WR[k]['first'])}" for k in keys if WR[k]["first"] is not None)
    return [{"caption": "空頭徵兆沒標到的頭部：三個另外的警告狀態（事先定，只報告，不做規則）。W1＝聯準會淨流動性 30 天平均名次在過去兩年"
                        "最低三分之一，而且價格離一年最高不到 15%；W2＝黃金 30 天平均名次在最高五分之一，而且價格離一年最高不到 15%；"
                        "W3＝未平倉量 30 天平均名次在最高五分之一。每格＝頭部當天亮不亮｜前 90 天有幾天亮（有資料的天數）。最後一列＝"
                        f"一般日子裡亮的比例。資料起點：{first}。次數極少，任何解讀都是推測。",
             "columns": ["頭部日", "第一步", "空頭狀態（往前）", "W1 淨流動性緊＋近高點", "W2 黃金強＋近高點", "W3 未平倉量高"],
             "rows": rows}]


# ---------- verdict ----------

def _abs(v) -> str:
    return "—" if v is None else f"{abs(v) * 100:.1f}%"


def _verdict(R: dict) -> str:
    X, base, PL, sup, holm, TW = R["X"], R["base"], R["PL"], R["sup"], R["holm"], R["TW"]
    idx, c = R["close"].index, R["close"].values.astype(float)
    good = [r for r in RULES if sup[r]["ok"]]
    hl = [r for r in RULES if holm.get(r)]
    parts = [f"一句話（量到的）：讓徵兆只改變均線規則的做法、由價格決定進出的五條「往前」規則，照事先定的標準被支持的有 {len(good)} 條"
             + (f"（{'、'.join(RSHORT[r] for r in good)}）" if good else "")
             + f"；50 日均線排名分數 {_sh(base['low'])}、最深虧損中位數 {base['c_deep_med']:.0f}%。這是用這個定義、這些資料、"
               "這麼少的轉折量到的，不是說徵兆或價格確認沒有意義。"]
    exam = "；".join(f"{RSHORT[r]} 往前／留一 {_sh(X['fwd'][r]['low'])}／{_sh(X['loto'][r]['low'])}、最深虧損中位數 "
                    f"{X['fwd'][r]['c_deep_med']:.0f}%／{X['loto'][r]['c_deep_med']:.0f}%、安慰劑 {_sh(PL['fwd'][r])}／"
                    f"{_sh(PL['loto'][r])}" + (f"、對照 {RSHORT[TWIN_OF[r]]} {_sh(TW[TWIN_OF[r]]['low'])}"
                                                if r in TWIN_OF else "") for r in RULES)
    parts.append(f"(a) 公平考試（量到的）：{exam}。Holm 校正後仍通過 {len(hl)} 條；五條各用 5% 門檻，光靠運氣平均約 0.25 條會過。"
                 "對照（不看徵兆）：" + "、".join(f"{RSHORT[r]} {_sh(TW[r]['low'])}（最深虧損中位數 {TW[r]['c_deep_med']:.0f}%）"
                                              for r in TWINS) + "。")
    tops = R["PA"]["fwd"]["tops"]
    on_tops = [x for x in tops if x["on"]]
    seg = []
    for x in tops:
        T = x["t"]["i"]

        def e(r, x=x, T=T):
            f = x[r]["first"]
            if f["i"] is None:
                return "已現金" if f["already"] else "沒出場"
            return ("頭部前已出 " if f["already"] else "") + f"{f['i'] - T:+d} 天（{_pc(c[f['i']] / c[T] - 1)}）"
        seg.append(f"{_d(x['t']['date'])}（空頭狀態{'亮' if x['on'] else '沒亮'}）：均線 {e('ma50')}、C1 {e('c1')}、C2 {e('c2')}；"
                   f"避開跌幅 均線 {_sh(x['ma50']['avoided'])}、C1 {_sh(x['c1']['avoided'])}、C2 {_sh(x['c2']['avoided'])}")
    s1, s2 = R["PA"]["fwd"]["sum"]["c1"], R["PA"]["fwd"]["sum"]["c2"]
    l1, l2 = R["PA"]["loto"]["sum"]["c1"], R["PA"]["loto"]["sum"]["c2"]
    yl = []
    for y in (2018, 2022):
        m = R["cal"]["ma50"].get(y)
        if m is not None:
            yl.append(f"{y} 年均線 {_pc(m)}，C1／C2／C3 往前比它 " + "／".join(
                f"{(R['cal'][('fwd', r)][y] - m) * 100:+.1f}" for r in ("c1", "c2", "c3")) + " 點")
    parts.append(f"(b) 照著出場，少不少賠？（量到的，往前，不扣成本）頭部當天空頭狀態亮的有 {len(on_tops)}／{len(tops)} 次。"
                 + "；".join(seg) + "。全部「規則現金、均線持有」的時段："
                 f"C1 {s1['n']} 段（{len(s1['fell'])} 段之後跌、平均少賠 {_abs(s1['avg_fell'])}；{len(s1['rose'])} 段之後漲、平均少賺 "
                 f"{_abs(s1['avg_rose'])}），合計帳戶是均線的 {(1 + s1['net']) * 100:.0f}%（留一 {(1 + l1['net']) * 100:.0f}%）；"
                 f"C2 {s2['n']} 段（{len(s2['fell'])} 跌、{len(s2['rose'])} 漲），合計 {(1 + s2['net']) * 100:.0f}%"
                 f"（留一 {(1 + l2['net']) * 100:.0f}%）。" + "；".join(yl) + "。")
    es, el = R["PA"]["fwd"]["esum"], R["PA"]["loto"]["esum"]
    bl = []
    for x in R["PA"]["fwd"]["bots"]:
        bl.append(f"{_d(x['t']['date'])}（多頭狀態{'亮' if x['on'] else '沒亮'}）提早進場 {len(x['eps'])} 次、均線進場前多賺 "
                  f"{_pc(x['g'], 1) if x['eps'] else '0'}")
    parts.append(f"(c) 照著進場，賺不賺？（量到的，往前，不扣成本）C4 提早進場 {es['n']} 次（已結束），{es['up']} 次結束時比進場高"
                 f"（平均 {_pc(es['avg_up'], 1)}），{es['n_stop']} 次被停損（平均 {_pc(es['avg_stop'], 1)}），全部連乘 {_pc(es['net'], 1)}"
                 f"（留一 {el['n']} 次、連乘 {_pc(el['net'], 1)}）。每次底部：" + "；".join(bl) + "。")
    mt = [x for x in tops if _d(x["t"]["date"]) in MISSED]
    parts.append("(d) 空頭徵兆沒標到的頭部（量到的）：" + "；".join(
        f"{_d(x['t']['date'])} 頭部前後 180 天，C1／C2／C3 和均線不同的天數 {x['c1']['diffdays']}／{x['c2']['diffdays']}／"
        f"{x['c3']['diffdays']}" for x in mt) + "（狀態沒亮時它們就是 50 日均線，所以在這幾次幫不上忙）。三個另外的警告狀態 W1–W3 "
        "見表；只有 3 次，任何「這種頭部有另一種警告」的說法都是推測。")
    parts.append("(e) 看不到的與限制：每一類只有約 8 次轉折、「往前」能用的約 5 次；空頭狀態在頭部亮的次數很少，所以 C1–C3 只在很少幾次"
                 "頭部和均線不同；六個徵兆多半是估值類、彼此一起動；部分資料是今天的修正版；120 天記憶、4／6 個、20 日均線、15% 都是"
                 "事先用道理定的，除了兩條穩健度以外沒試別的數字。沒看到的效果只表示用這個定義和這些資料沒看到。")
    return "\n".join(parts)


def _now(R: dict) -> list[dict]:
    idx, c = R["close"].index, R["close"].values.astype(float)
    i = len(idx) - 1
    word = {True: "持有", False: "現金"}
    P = R["POS"]
    out = [{"label": f"最新收盤（{_d(idx[i])}，UTC 0 點日線）", "value": f"{c[i]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜C1｜C4",
            "value": f"{word[bool(P['ma50'][i] > 0.5)]}｜{word[bool(P[('fwd', 'c1')][i] > 0.5)]}｜"
                     f"{word[bool(P[('fwd', 'c4')][i] > 0.5)]}"}]
    for k, nm, cn in (("bear", "空頭狀態（往前）", 0), ("bull", "多頭狀態（往前）", 1)):
        st = R["ST"]["fwd"][cn]
        cnt = R["cnt"]["fwd"][cn]
        s = _since(st, i)
        hit = np.flatnonzero(np.nan_to_num(cnt, nan=0.0) >= LEVEL)
        last = _d(idx[hit[-1]]) if len(hit) else "從沒"
        val = (f"亮（{_d(idx[s])} 起）" if s is not None else "沒亮") + f"；今天徵兆數 {cnt[i]:.0f}／6；最後一次 ≥4 個是 {last}"
        out.append({"label": nm, "value": val})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根收盤後的狀態，是描述，不是預測，也不是買賣建議"})
    return out


def study() -> dict:
    t0 = time.time()
    R = _compute()
    tables = _exam_tables(R) + _pa_tables(R) + _w_tables(R)
    verdict = _verdict(R)
    sec = R["secs"]
    notes = [
        "這是第三步：第二步量到空頭徵兆常常在頭部前 55–90 天就亮、那時價格還比頭部低 41%–69%，光照徵兆站開會少賺很多。所以這一步"
        "加一個「價格確認」：徵兆只改變 50 日均線規則的出場或進場方式，真正進出一定要價格先動。",
        "名詞：50 日均線規則＝收盤高於過去 50 天平均才持有（目前的主訊號）；20 日均線同理；空頭狀態＝過去 120 天內 6 個空頭徵兆曾經"
        "≥4 個同時亮；多頭狀態同理；徵兆「亮」＝那個指標過去 30 天平均名次落在自己過去兩年偏高或偏低的五分之一；20 日新高＝收盤高於"
        "前 20 天的最高收盤；最深虧損＝帳戶從高點掉最多的幅度。",
        "C1 的讀法（事先定）：「跌破 20 日均線就走、回到 50 日均線上才回來」——收盤在 50 日均線上但還在 20 日均線下，仍然是「該走」的"
        "日子，所以空頭狀態下等於要同時站上兩條均線才持有（和 crash_timing 的 a20_50 一樣）。C2 的最高點從「狀態開始那天」和「這次"
        "上車那天」比較晚的那天算起，免得剛買回就被舊高點觸發停損。",
        "做法：「往前」＝只用當時已確認的轉折挑徵兆（當時真的做得到，只有它能判定支持）；「留一」＝拿掉要考的那次轉折再挑（還是用到"
        "之後的轉折）；「全部」＝看過所有轉折才挑，樣本內，只當參考。",
        "之前量過的（不是這次量的）：crash_timing——50 日均線第一次下車平均已比高點低 35%、買回比低點高 32%，只避開中位數 30% 的跌幅；"
        "更早下車的做法避開更多（最多 69%），但等效買回價比低點高 +128%，排名分數只剩 23%（均線 80%）；20 日均線每年被洗 13.8 次、"
        "50 日 7.3 次。exit_entry_combo——最會躲跌的組合躲掉 83%，但等於比低點高 +221% 才接回，排名分數 16%；25 種組合沒有一種勝過"
        "50 日均線。whipsaw_anatomy——50 日均線 148 次下車裡 104 次（70%）是假跌破。這一步的 T1–T4 就是在檢查：同樣的價格規則"
        "不看徵兆時會怎樣。",
        "公平考試和 main_signal_check 一樣：比特幣 2015-01 到 2023-12 每月起算、2018-01 到 2023-12 每月起算×24 個收盤時間、"
        "一般人做法（晚 10 小時、成本 0.25%、一成進場漏掉、每個收盤時間 200 次模擬）、其他幣（ETH、SOL、BNB、XRP、MSTR，用自己的"
        "價格、比特幣的狀態）。安慰劑每次只模擬 20 次，真的那條也用 20 次重算來比。",
        "「照著出場／進場」那幾張表不扣成本，只看價格；考試的數字有扣成本。",
        "限制：每一類只有約 8 次轉折，「往前」能用的約 5 次；空頭狀態在頭部亮的次數很少，C1–C3 只在很少幾次頭部和均線不同；六個徵兆"
        "多半是估值類、彼此一起動，不是六個獨立線索；鏈上、短期持有人成本、OECD 用今天的修正版；120 天、4／6、20 日均線、15%、"
        "10 天都是事先用道理定的，除了「≥5」和「60 天」兩條穩健度以外沒試別的數字。",
        "所有定義和門檻都在跑之前寫好，看過結果後沒有改。",
        f"這次計算：資料約 {sec['data']:.0f} 秒、考試約 {sec['exam'] - sec['data']:.0f} 秒、安慰劑約 {sec['placebo']:.0f} 秒，"
        f"全部約 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來一樣；這裡只是描述幾條固定規則過去做了什麼，不是買賣建議。",
    ] + [p for p in R["problems"] if p]
    if R["dropped"]:
        notes.append("這次拿掉的指標：" + "、".join(f"{fts._name(k)}（{v}）" for k, v in R["dropped"].items()))
    src = dict(R["src"])
    src["幣安現貨 BTCUSDT 1 小時線（24 個收盤時間）"] = px.BINANCE_HOSTS[0] + "/api/v3/klines?symbol=BTCUSDT&interval=1h"
    for s in ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"):
        src[f"幣安現貨 {s} 日線"] = px.BINANCE_HOSTS[0] + f"/api/v3/klines?symbol={s}&interval=1d"
    src["Yahoo 日線 MSTR"] = ro.YAHOO.format("MSTR")
    return {"id": ID, "title": "徵兆加上價格確認：出場少賠嗎？進場賺嗎？",
            "question": "第二步量到：照轉折徵兆直接進出，時間點太早。如果徵兆不直接下單，而是只改變均線規則的做法——空頭徵兆多時"
                        "跌破 20 日均線就走、或用移動停損、或晚點買回；多頭徵兆多時站上 20 日均線就先進場——在沒看過那次轉折的情況下，"
                        "出場有沒有少賠、進場有沒有賺？而且是徵兆在幫忙，還是價格規則本身？",
            "asof": _d(R["close"].index[-1]), "verdict": verdict, "now": _now(R), "tables": tables, "notes": notes,
            "sources": [{"name": k, "url": v} for k, v in src.items()]}


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
