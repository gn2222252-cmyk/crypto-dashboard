"""Historical study "scale_in": enter ma50's trades at a smaller size and go to full size only once the trade has proved
itself. Does that earn MORE than ma50 AND lose LESS, robustly - and is it the timing of the add-on that matters, or
only that less is held?

Why: the main signal ma50 (hold bitcoin while the daily close is above its 50-day average, otherwise cash) makes its
money from a handful of trades. at_the_signal measured 148 closed entries: 105 fizzled (trade result <= 0, mean -3.6%),
20 were small and 23 were runners (>= +20%, mean +191%, median +54%). As a mere comparison twin with no indicator,
that study scored "enter at HALF size; go to full size once the trade is 10% in profit" (E0): ranking 77% and median
deepest loss -40.2% on starts from 2019, against ma50's 79% and -52.4%. It was never tested as a rule of its own. It
needs no indicator. Its logic: most entries fail, so risk less until an entry proves itself. This study tests it and
is built so that it can fail.

Owner's ground rules (every user-facing sentence): observe first; do not conclude lightly; label statements 量到的
(measured, with the number, k of N) or 推測 (a guess, marked); where the data cannot separate two explanations say so;
never use one tendency as a law to dismiss another; "not seen" = not seen with this definition and this data. "Holy
grail" standard = earns MORE than the main signal AND loses LESS, robustly; both sides are reported separately and
together.

Written BEFORE any run of this file. Nothing is tuned after seeing results; later additions are labelled.

Data (free, key-less; .env never read; loaders REUSED): BTC daily closes te._daily_close (blockchain.info before Binance
2017-08, 00:00 UTC); Binance 1-hour BTCUSDT grid and prehistory (clock_robustness, read only); ETH SOL BNB XRP (Binance)
and MSTR (Yahoo) via main_signal_check._load_other; NFCI via nfci_filter._series with its fallback store redirected to
cache/_store/scale_in/nfci.json (seeded read-only from the shared copy). No other file is written.

Definitions.
  ma = main_signal_check's ma50 state after each close (1 on / 0 off). An ENTRY is a close s with ma on at s and off
  (or undefined) at s-1; the TRADE runs over the closes s .. end-1, end = the first later close with ma off (the exit
  close x = end). All rules below: cash when ma is off; everything sold when ma turns off. A rule gives a SIZE between
  0 and 1 for each close of a trade; the position is ma x size, decided at the close, held from the next day on.
  S10 (pre-registered main rule, already glimpsed in at_the_signal): size 50% from the entry close; 100% from the first
    close t in the trade with close(t) >= 1.10 x close(s) (confirmation); never back to 50% inside the trade.
  Grid (fixed now): start size 33% / 50% / 67% x confirmation +5% / +10% / +15% / +20% (12 settings; 50%/+10% = S10).
  ST: 50%, 100% from the 10th consecutive close above the 50-day average (close s+9, if the trade is still on).
  SH: 50%, 100% from the first close t > s that is a new 20-day closing high (close(t) > every close of t-19 .. t-1).
  SM: 50%, 100% from the first close t >= s where the 50-day average is higher than 10 closes earlier.
  CONST twin of a rule: ma x c, c = the rule's own time-weighted size on BTC 00:00 UTC days with ma on, from 2015-01-01
    (the exam's first start). Same average exposure, no timing.
  RAND (placebo) of a rule with start size f: on each series, each trade is "confirmed" with probability p = the share
    of that series' trades the rule itself confirmed; a confirmed trade goes to 100% from a close drawn uniformly from
    s .. end-1; otherwise it stays at f the whole trade. 200 draws, fixed seeds. Question: does WHEN the second part is
    added matter, or only that less is held?
  H_NFCI (half_gate, unchanged): ma x (100% when the monthly NFCI gate is favorable, 50% when unfavorable or unknown).
  SG (combination, fixed now): ma x min(S10's size, H_NFCI's size).
  Every series in the exam (BTC daily, the 24 closing hours, ETH SOL BNB XRP MSTR) applies the rule to ITS OWN closes,
  50-day average and ma50 entries; the NFCI gate is the same US monthly flag for all (as half_gate).

PART 1 - observe ma50's own trades first (BTC daily from 2010, no rule). Closed trades only (a trade still open is left
  out). RESULT = close(x)/close(s) - 1 - 2 x 0.10%; RUNNER >= +20%, FIZZLE <= 0, SMALL in between (at_the_signal's
  labels). BEST / WORST = highest / lowest close in s..x over close(s) - 1. Distribution of RESULT in fixed bins with
  median BEST, WORST and length. Runners: days from s to the first close >= +5% / +10% / +15% and the lowest close in
  s..that close (deepest dip below the entry price before it). Fizzles: how many were ever >= +5% / +10% (false
  confirmations). Arithmetic of scaling in at 50% for confirmation +3 / +5 / +10 / +15%, described per trade with the
  two parts BOUGHT AND HELD inside the trade (a description, not the exam, which resets the size every close):
  scaled RESULT = 0.5 x (close(x)/close(s) - 1) + 0.5 x (close(x)/close(t) - 1) if confirmed at t, else 0.5 x
  (close(x)/close(s) - 1); cost 0.10% on each part bought and sold. For each level: runners joined at full size (k of
  23) and the mean confirmation price above the entry, the mean result given up per runner (points of the whole
  position), fizzles held at half the whole trade (k of N) and the mean saved per fizzle, and all closed trades
  chained (product of 1 + RESULT) at full size versus scaled. The runners listed one by one. By entry period 2011-16 /
  2017-21 / 2022-now (entries in 2010 are only in the totals).
PART 2 - the rule family above, fixed now.
PART 3 - the exam versus ma50 on the same starts: business_cycle._Exam's arithmetic (main_signal_check's four checks,
  half_gate's line-for-line copy kept per start): BTC 00:00 UTC daily, 108 monthly starts 2015-01 .. 2023-12 (0.10%
  cost); 72 monthly starts 2018-01 .. 2023-12 x 24 closing hours = 1,728 textbook cells (0.10%); the ordinary person on
  the same 1,728 cells: traded 10 hours later, 0.25% cost on EVERY change of size (entry, add-on, exit and the daily
  reset of a part position), and robust_trend._missed: every close where the wanted size rises by more than 10 points
  - the first part AND the add-on alike - is missed with probability 10%, drawn independently per day; what was not
  bought stays not bought until the wanted size has come down by as much (for these rules: until the exit); 200
  simulations per clock; other assets ETH SOL BNB XRP MSTR from their first valid month to three years before today.
  Ranking score = lowest of the four "beats holding" shares, eligible only if the deepest loss is shallower than
  holding's in >= 95% of BTC starts and cells (business_cycle._score_ge). Median / worst deepest loss over the 1,728
  textbook cells. Seeds: ma50 = main_signal_check's; every other rule = half_gate's "m2" stream (so H_NFCI equals
  half_gate's numbers; checked). Reported per rule: the four parts, ranking, median and worst deepest loss, exposure
  (mean size from 2015 on BTC daily), switches per year (closes where the wanted size changes), yearly turnover and
  the cost it pays at 0.10% and 0.25%, calendar-year return minus ma50 for every year (textbook, 0.10%), and return per
  unit of deepest loss = median over the 1,728 textbook cells of yearly return / |deepest loss|.
  S10, pre-registered reading, all four: (1) ranking >= ma50's; (2) median deepest loss strictly shallower than ma50's;
  (3) ranking strictly above its CONST twin's (half_gate._above); (4) placebo share <= 5% = share of the 200 RAND draws
  with ranking at least the real rule's, both scored with 50 ordinary-person simulations per clock. The same four items
  for ST, SH, SM; Holm at 5% across S10, ST, SH, SM on p = (hits + 1) / 201; a rule "passes after Holm" when it passes
  all four AND its Holm-adjusted p <= 0.05. Luck: 4 x 5% = 0.2 rules expected to clear the placebo item by chance.
  Grid: each of the 12 settings scored with the full exam; count of settings with ranking >= ma50's AND strictly
  shallower median deepest loss; "robust" if at least 9 of 12. Each setting's CONST twin shown beside it (report only).
PART 4 - the owner's question in plain counts, each rule versus ma50 on the same starts: of the 108 daily starts, the
  1,728 textbook cells and the 1,728 x 200 ordinary-person runs (simulation i of the rule paired with simulation i of
  ma50; missed entries drawn separately, as half_gate), how many end with MORE money (higher yearly return), with a
  SHALLOWER deepest loss, and with BOTH; start years in which fewer than half of the daily starts end with more money
  / with both. Plain reading fixed now, on the textbook cells: "賺更多又虧更少" if more >= 50% AND shallower >= 50% AND
  both >= 50%; "虧更少、但是靠少賺換來的" if shallower >= 50% and more < 50%; "賺更多、但虧更深" if more >= 50% and
  shallower < 50%; otherwise "兩邊都沒有多數".
PART 5 - SG with the same exam and counts, against S10 alone, H_NFCI alone and SG's CONST twin. Pre-registered: SG
  "adds something" if its ranking >= the higher of S10's and H_NFCI's AND its median deepest loss is strictly
  shallower than both AND its ranking is strictly above its CONST twin's.
PART 6 - honest split: S10 was first seen on starts from 2019. Every Part 3 score (four parts, ranking, losses, S10's
  four items with the placebo re-scored on the same subset) and Part 4 count is reported separately for starts before
  2019-01-01 (daily 2015-2018, textbook / ordinary-person cells 2018 only, other assets' starts in 2018) and from 2019.
  The pre-2019 starts did not enter the earlier glimpse and are the cleaner evidence.
Overall reading fixed now: "站得住" = S10 passes all four AND passes after Holm AND the grid is robust; "部分站得住" =
  S10 passes all four but Holm or the grid does not hold; otherwise "這次沒有達到".
Today: "比特幣：50 日均線｜S10｜SG" (持有 / 半倉 / 現金), the latest ma50 entry and whether it was ever confirmed.
Limits: the extra trades' costs; one asset's history with about 23 runners carrying everything; the other assets use
  their own prices; a rule that holds less will almost always lose less, so the meaningful comparisons are CONST and
  RAND. Not advice.
Added after the first test run (a 3-draw smoke run; no definition, threshold or bar above changed): the pre-registered
  RAND confirms random trades, and the rule's confirmed trades are mostly the long ones, so RAND held much less than
  the rule (BTC mean size 0.34 against 0.49) - an easy placebo. RAND-B is added and REPORTED ONLY (not in any bar or in
  Holm): the very trades the rule confirmed, the add-on close drawn uniformly from s .. end-1, 200 draws, scored the
  same way. It isolates WHEN the add-on happens (and knows which trades later reached the level, so it is a hard
  comparison). The median time-weighted size on ma50-on days of the RAND draws is shown beside the rule's.
Added after the first full run (wording only, no number, rule or bar changed): the verdict now also states S10's
  daily-start counts next to the textbook-cell reading, and the per-trade points are printed without a confusing sign.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import at_the_signal as ats
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import half_gate as hg
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import robust_trend as rt

ID = "scale_in"
DAY = pd.Timedelta(days=1)
FIRST = msc.DAILY_STARTS[0]
SPLIT = pd.Timestamp("2019-01-01")
COST, P_COST, SIMS = bc.COST, bc.P_COST, bc.SIMS
DRAWS, PL_SIMS, PLC_BAR, HOLM_ALPHA = 200, 50, 0.05, 0.05
START_F, CONF_K = 0.5, 0.10
SIZES = (0.33, 0.50, 0.67)
CONFS = (0.05, 0.10, 0.15, 0.20)
OBS_LEVELS = (0.03, 0.05, 0.10, 0.15)
RUN_LEVELS = (0.05, 0.10, 0.15)
ST_DAYS, SH_DAYS, SM_DAYS = 10, 20, 10
ROBUST_N = 9
RUNNER, FIZZLE = ats.RUNNER, ats.FIZZLE
PERIODS = (("2011–16", 2011, 2016), ("2017–21", 2017, 2021), ("2022–今", 2022, 2100))
BINS = ((-1e9, -0.10, "虧 10% 以上"), (-0.10, -0.05, "虧 5%–10%"), (-0.05, 0.0, "虧 0–5%（含打平）"),
        (0.0, 0.20, "賺 0–20%（小賺）"), (0.20, 0.50, "賺 20%–50%"), (0.50, 1.00, "賺 50%–100%"), (1.00, 1e9, "賺 1 倍以上"))
RAND_SEED = 20261120
MAIN = ("S10", "ST", "SH", "SM")
RNAME = {"S10": "S10：先 50%，比進場價高 10% 加滿", "ST": "ST：先 50%，連續 10 天站在均線上加滿",
         "SH": "SH：先 50%，創 20 天收盤新高加滿", "SM": "SM：先 50%，50 日均線比 10 天前高就加滿",
         "SG": "SG：S10 和 H_NFCI 取小的", "H_NFCI": "H_NFCI：NFCI 不利時半倉（half_gate）"}

_pct, _pp, _num, _d, _ok, _share = bc._pct, bc._pp, bc._num, bc._d, bc._ok, bc._share
MY_STORE = common.STORE / ID


# ---------- data ----------

@contextmanager
def _nfci_store():
    """nfci_filter's fallback store redirected into this study's folder (seeded from the shared copy, never written)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    mine, shared = MY_STORE / "nfci.json", nf.STORE_PATH
    try:
        if shared.exists() and (not mine.exists() or shared.stat().st_mtime > mine.stat().st_mtime):
            shutil.copy2(shared, mine)
    except Exception:  # noqa: BLE001 - a missing seed only means no fallback
        pass
    old = nf.STORE_PATH
    nf.STORE_PATH = mine
    try:
        yield
    finally:
        nf.STORE_PATH = old


def _load() -> dict:
    with _nfci_store():
        nfci, where, problems = nf._series()
    other, more = msc._load_other()
    return {"close": bc.te._daily_close(), "nfci": nfci, "where": where, "other": other, "problems": problems + more}


# ---------- the rules ----------

class _Series:
    """One price series with its ma50 trades and the confirmation conditions, on a full history; positions are read on
    the exam's slice."""

    def __init__(self, c: np.ndarray, ma_full: np.ndarray, sl: slice, ma_exam: np.ndarray, gate: np.ndarray):
        self.c = np.asarray(c, dtype=float)
        self.on = np.nan_to_num(np.asarray(ma_full, dtype=float)) > 0.5
        self.sl, self.ma, self.gate = sl, np.asarray(ma_exam, dtype=float), gate
        n = len(self.c)
        self.ent = np.flatnonzero(self.on[1:] & ~self.on[:-1]) + 1
        if n and self.on[0]:
            self.ent = np.concatenate([[0], self.ent])  # a trade already on at the first close: treated as entered there
        offs = np.flatnonzero(~self.on)
        if len(offs):
            j = np.searchsorted(offs, self.ent)
            self.end = np.where(j < len(offs), offs[np.minimum(j, len(offs) - 1)], n)
        else:
            self.end = np.full(len(self.ent), n, dtype=int)
        s = pd.Series(self.c)
        prev = s.shift(1).rolling(SH_DAYS - 1, min_periods=SH_DAYS - 1).max().values
        self.sh = np.nan_to_num(self.c > prev, nan=False) & np.isfinite(prev)
        m50 = s.rolling(50, min_periods=50).mean().values
        m10 = np.concatenate([np.full(SM_DAYS, np.nan), m50[:-SM_DAYS]])
        self.sm = np.isfinite(m50) & np.isfinite(m10) & (m50 > m10)

    def confirm(self, kind: str, k: float) -> np.ndarray:
        """Close index from which the trade is at 100% (-1 = never) for every trade."""
        out = np.full(len(self.ent), -1, dtype=int)
        for i, (s, e) in enumerate(zip(self.ent, self.end)):
            if kind == "scale":
                hit = np.flatnonzero(self.c[s:e] >= (1 + k) * self.c[s])
                t = s + hit[0] if len(hit) else -1
            elif kind == "st":
                t = s + ST_DAYS - 1 if s + ST_DAYS - 1 < e else -1
            elif kind == "sh":
                hit = np.flatnonzero(self.sh[s + 1:e])
                t = s + 1 + hit[0] if len(hit) else -1
            elif kind == "sm":
                hit = np.flatnonzero(self.sm[s:e])
                t = s + hit[0] if len(hit) else -1
            else:
                raise ValueError(kind)
            out[i] = t
        return out

    def rand(self, p: float, rng: np.random.Generator) -> np.ndarray:
        n_ = len(self.ent)
        yes = rng.random(n_) < p
        u = rng.random(n_)
        span = self.end - self.ent
        t = self.ent + np.floor(u * span).astype(int)
        return np.where(yes & (span > 0), t, -1)

    def rand_same(self, real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """RAND-B (added after the first test run): the very trades the rule confirmed, add-on day drawn uniformly."""
        u = rng.random(len(self.ent))
        span = self.end - self.ent
        t = self.ent + np.floor(u * span).astype(int)
        return np.where((real >= 0) & (span > 0), t, -1)

    def size(self, f: float, conf: np.ndarray) -> np.ndarray:
        frac = np.ones(len(self.c))
        for s, e, t in zip(self.ent, self.end, conf):
            frac[s:(t if t >= 0 else e)] = f
        return frac[self.sl]


class _Book:
    """Positions of every rule on every series of the exam."""

    def __init__(self, ex: bc._Exam, price: np.ndarray, pre: np.ndarray, F: pd.Series):
        self.ex, self.F = ex, F
        close = ex.close
        S = {}
        S["btc"] = _Series(close.values, ex.btc_ma, slice(0, len(close)), ex.btc_ma,
                           bc._on_days(F, close.index + DAY))
        npre, days = len(pre), ex.days
        gc = bc._on_days(F, ex.m.dates)
        self.clock_ok = True
        for h in range(24):
            full = np.concatenate([pre, price[h::24]]).astype(float)
            maf = msc._targets(pd.Series(full)).fillna(0.0)["ma50"].values.astype(float)
            sl = slice(npre, npre + days)
            if not np.array_equal(maf[sl], ex.cma[h]):
                self.clock_ok = False
            S[("clock", h)] = _Series(full, maf, sl, ex.cma[h], gc)
        for a, o in ex.other.items():
            S[("o", a)] = _Series(o["close"].values, o["ma"], slice(0, len(o["close"])), o["ma"],
                                  bc._on_days(F, o["close"].index + DAY))
        self.S = S
        self.keys = list(S)
        self._conf = {}

    def conf(self, key, kind: str, k: float = 0.0) -> np.ndarray:
        ck = (key, kind, k)
        if ck not in self._conf:
            self._conf[ck] = self.S[key].confirm(kind, k)
        return self._conf[ck]

    def frac(self, key, spec: tuple) -> np.ndarray:
        s = self.S[key]
        kind = spec[0]
        if kind == "ma50":
            return np.ones(len(s.ma))
        if kind == "const":
            return np.full(len(s.ma), spec[1])
        if kind == "scale":
            return s.size(spec[1], self.conf(key, "scale", spec[2]))
        if kind in ("st", "sh", "sm"):
            return s.size(START_F, self.conf(key, kind))
        if kind == "hn":
            return 0.5 + 0.5 * s.gate
        if kind == "sg":
            return np.minimum(self.frac(key, ("scale", START_F, CONF_K)), 0.5 + 0.5 * s.gate)
        raise ValueError(kind)

    def pos(self, spec: tuple) -> dict:
        return {key: self.S[key].ma * self.frac(key, spec) for key in self.keys}

    def rand_pos(self, spec: tuple, rng: np.random.Generator, same: bool = False) -> dict:
        """RAND twin of a confirmation rule: same start size, same share of trades confirmed per series, random day.
        same=True: RAND-B, the same trades as the rule (added after the first test run)."""
        f = spec[1] if spec[0] == "scale" else START_F
        out = {}
        for key in self.keys:
            s = self.S[key]
            real = self.conf(key, spec[0], spec[2] if spec[0] == "scale" else 0.0)
            if same:
                out[key] = s.ma * s.size(f, s.rand_same(real, rng))
                continue
            p = float((real >= 0).mean()) if len(real) else 0.0
            out[key] = s.ma * s.size(f, s.rand(p, rng))
        return out


SPECS = {"ma50": ("ma50",), "S10": ("scale", START_F, CONF_K), "ST": ("st",), "SH": ("sh",), "SM": ("sm",),
         "H_NFCI": ("hn",), "SG": ("sg",)}


# ---------- the exam (half_gate._exam, positions given per series, per-start arrays kept) ----------

def _exam(ex: bc._Exam, P: dict, ki: int, sims: int = SIMS) -> dict:
    pos = P["btc"]
    net, _ = bc.te._run(np.concatenate([[np.nan], pos[:-1]]), ex.btc_change, COST)
    dr, dd = msc._suffix(msc._equity(net), ex.dstarts, bc.YEAR)
    n = len(ex.cstarts)
    tr, td = np.zeros((n, 24)), np.zeros((n, 24))
    pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
    for h in range(24):
        w = np.nan_to_num(P[("clock", h)], nan=0.0).astype(float)
        r, d_ = msc._suffix(np.concatenate([[1.0], ex.eng._mid(h, w, COST, None)]), ex.cstarts, bc.YEAR)
        tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
        b, wl = rt._later(h, w, ex.days)
        rng = np.random.default_rng(bc.SEED + 1000 * ki + h)
        p = rt._missed(wl, bc.P_MISS, rng, sims)
        r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), ex.eng._mid(b, p, P_COST, None)]), ex.cstarts, bc.YEAR)
        pr[:, h, :], pdd[:, h, :] = r, d_
    oth = {}
    for a, o in ex.other.items():
        p = P[("o", a)]
        net2, _ = bc.te._run(np.concatenate([[np.nan], p[:-1]]), o["change"], COST)
        r, d_ = msc._suffix(msc._equity(net2), o["starts"], o["year"])
        oth[a] = (r[:, 0], d_[:, 0])
    a = ex.close.index >= pd.Timestamp(FIRST)
    pa = np.nan_to_num(pos[a])
    on = np.nan_to_num(ex.btc_ma[a]) > 0.5
    _n, trade = bc.te._run(pos, ex.btc_change, COST)
    yrs = a.sum() / 365.0
    tp = np.nan_to_num(pos)
    sw = int((np.abs(np.diff(tp[a])) > 1e-9).sum())
    return {"arr": {"dr": dr[:, 0], "dd": dd[:, 0], "tr": tr, "td": td, "pr": pr, "pdd": pdd, "o": oth},
            "exp": float(pa.mean()), "exp_on": float(pa[on].mean()) if on.any() else None,
            "sw_y": sw / yrs, "turn_y": float(trade[a].sum() / yrs)}


def _masks(ex: bc._Exam, part: str | None) -> dict:
    def m(labels):
        lab = pd.DatetimeIndex(labels)
        if part is None:
            return np.ones(len(lab), dtype=bool)
        return np.asarray(lab < SPLIT) if part == "pre" else np.asarray(lab >= SPLIT)

    return {"d": m(ex.dlabels), "c": m(ex.clabels), "o": {a: m(o["labels"]) for a, o in ex.other.items()}}


def _score(ex: bc._Exam, X: dict, part: str | None = None) -> dict:
    """main_signal_check's four checks from the kept arrays, on all starts or on one side of the split."""
    A, M = X["arr"], _masks(ex, part)
    md, mc = M["d"], M["c"]
    dr, dd = A["dr"][md], A["dd"][md]
    dh = (ex.dhold[0][md], ex.dhold[1][md])
    tr, td, pr, pdd = A["tr"][mc], A["td"][mc], A["pr"][mc], A["pdd"][mc]
    hr, hd = ex.chold[0][mc], ex.chold[1][mc]
    edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
    o_n = o_beat = 0
    for a, o in ex.other.items():
        mo = M["o"][a]
        r = A["o"][a][0][mo]
        o_n += int(mo.sum())
        o_beat += int((r > o["hold"][0][mo]).sum())
    x = {"d_n": len(dr), "c_n": edge.size, "p_n": pedge.size, "o_n": o_n}
    x["checks"] = [float((dr > dh[0]).mean()) if len(dr) else 0.0, float((edge > 0).mean()) if edge.size else 0.0,
                   float((pedge > 0).mean()) if pedge.size else 0.0, (o_beat / o_n) if o_n else 0.0]
    x["low"] = min(x["checks"])
    ls = [float((dd > dh[1]).mean()) if len(dd) else 0.0, float((td > hd[:, None]).mean()) if td.size else 0.0]
    x["loss_ok"] = min(ls)
    x["eligible"] = x["loss_ok"] >= bc.LOSS_BAR
    x["c_deep_med"] = float(np.median(td)) if td.size else None
    x["c_deep_worst"] = float(td.min()) if td.size else None
    x["d_deep_med"] = float(np.median(dd)) if len(dd) else None
    ok = td < -1e-9
    x["ratio"] = float(np.median(tr[ok] / np.abs(td[ok]))) if ok.any() else None
    x["p_edge"] = float(np.median(pedge)) if pedge.size else None
    for k in ("exp", "exp_on", "sw_y", "turn_y"):
        x[k] = X.get(k)
    return x


def _vs(ex: bc._Exam, X: dict, B: dict, part: str | None = None) -> dict:
    """Part 4 counts of X against ma50 (B) on the same starts."""
    a, b, M = X["arr"], B["arr"], _masks(ex, part)
    md, mc = M["d"], M["c"]
    out = {}
    for lab, r1, d1, r0, d0 in (("d", a["dr"][md], a["dd"][md], b["dr"][md], b["dd"][md]),
                                 ("c", a["tr"][mc], a["td"][mc], b["tr"][mc], b["td"][mc]),
                                 ("p", a["pr"][mc], a["pdd"][mc], b["pr"][mc], b["pdd"][mc])):
        more_, shal = r1 > r0 + 1e-9, d1 > d0 + 1e-9
        out[lab] = (int(more_.sum()), int(shal.sum()), int((more_ & shal).sum()), int(more_.size))
    out["d_more"] = a["dr"] > b["dr"] + 1e-9
    out["d_both"] = out["d_more"] & (a["dd"] > b["dd"] + 1e-9)
    return out


def _reading(v: tuple) -> str:
    m_, sh, b, n = v
    if not n:
        return "—"
    if m_ >= n / 2 and sh >= n / 2 and b >= n / 2:
        return "賺更多又虧更少"
    if sh >= n / 2 and m_ < n / 2:
        return "虧更少、但是靠少賺換來的"
    if m_ >= n / 2 and sh < n / 2:
        return "賺更多、但虧更深"
    return "兩邊都沒有多數"


# ---------- Part 1: ma50's own trades ----------

def _observe(close: pd.Series) -> dict:
    c = close.values.astype(float)
    ma = msc._targets(close)["ma50"].values.astype(float)
    EN = ats._entry_events(close, ma)
    T = []
    for r in EN["rows"]:
        if r["res"] is None:
            continue
        s, x = r["i"], r["x"]
        seg = c[s:x + 1] / c[s] - 1
        t = {"s": s, "x": x, "day": r["day"], "res": r["res"], "best": float(seg.max()), "worst": float(seg.min()),
             "len": x - s, "kind": r["kind"], "reach": {}, "dip": {}}
        for k in sorted(set(OBS_LEVELS) | set(RUN_LEVELS)):
            hit = np.flatnonzero(c[s:x] >= (1 + k) * c[s])  # confirmation only while the trade is on (closes s..x-1)
            if len(hit):
                tt = s + int(hit[0])
                t["reach"][k] = tt
                t["dip"][k] = float(c[s:tt + 1].min() / c[s] - 1)
        T.append(t)
    return {"T": T, "c": c, "open": [r for r in EN["rows"] if r["res"] is None]}


def _scaled(t: dict, c: np.ndarray, k: float, f: float = START_F) -> float:
    s, x = t["s"], t["x"]
    g = f * (c[x] / c[s] - 1) - 2 * COST * f
    tt = t["reach"].get(k)
    if tt is not None:
        g += (1 - f) * (c[x] / c[tt] - 1) - 2 * COST * (1 - f)
    return float(g)


# ---------- computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, nfci = data["close"], data["nfci"]
    months = pd.period_range(nfci.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
    F = hg._flag("nfci", nfci, months)
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam({"close": close, "other": data["other"]}, price, b0, pre)
    book = _Book(ex, price, pre, F)
    KM, K2 = bc.KI["ma50"], bc.KI["m2"]

    # identity checks against the parents
    X = {"ma50": _exam(ex, book.pos(SPECS["ma50"]), KM)}
    chk = ex.summary("ma50", None)
    s_ma = _score(ex, X["ma50"])
    hn_parent = hg._exam(ex, "half", F)
    X["H_NFCI"] = _exam(ex, book.pos(SPECS["H_NFCI"]), K2)
    s_hn = _score(ex, X["H_NFCI"])
    same = (abs(chk["low"] - s_ma["low"]) < 1e-9 and abs(chk["c_deep_med"] - s_ma["c_deep_med"]) < 1e-9
            and abs(hn_parent["low"] - s_hn["low"]) < 1e-9 and abs(hn_parent["c_deep_med"] - s_hn["c_deep_med"]) < 1e-9
            and book.clock_ok)

    for k in ("S10", "ST", "SH", "SM", "SG"):
        X[k] = _exam(ex, book.pos(SPECS[k]), K2)
    S = {k: _score(ex, v) for k, v in X.items()}
    tw = {}
    for k in ("S10", "ST", "SH", "SM", "SG", "H_NFCI"):
        c_ = S[k]["exp_on"]
        tw[k] = _exam(ex, book.pos(("const", c_)), K2)
        tw[k]["c"] = c_
    TS = {k: _score(ex, v) for k, v in tw.items()}
    t_main = time.time() - t0

    # placebo (RAND) for the four confirmation rules, 50 ordinary-person simulations per clock
    small = {k: _exam(ex, book.pos(SPECS[k]), K2, PL_SIMS) for k in MAIN}
    small_s = {k: {p: _score(ex, small[k], p) for p in (None, "pre", "post")} for k in MAIN}
    pl = {}
    for mode, off in (("a", 0), ("b", 500_000)):  # "a" pre-registered; "b" added after the first test run
        hits = {k: {p: 0 for p in (None, "pre", "post")} for k in MAIN}
        lows, deeps, exps = {k: [] for k in MAIN}, {k: [] for k in MAIN}, {k: [] for k in MAIN}
        for j, k in enumerate(MAIN):
            for i in range(DRAWS):
                rng = np.random.default_rng(RAND_SEED + off + 1000 * j + i)
                Xr = _exam(ex, book.rand_pos(SPECS[k], rng, same=(mode == "b")), K2, PL_SIMS)
                exps[k].append(Xr["exp_on"])
                for p in (None, "pre", "post"):
                    sr = _score(ex, Xr, p)
                    hits[k][p] += int(bc._score_ge(sr, small_s[k][p]))
                    if p is None:
                        lows[k].append(sr["low"])
                        deeps[k].append(sr["c_deep_med"])
        pl[mode] = {"plc": {k: {p: hits[k][p] / DRAWS for p in hits[k]} for k in MAIN},
                    "pval": {k: (hits[k][None] + 1) / (DRAWS + 1) for k in MAIN},
                    "rand": {k: (float(np.median(lows[k])), float(np.median(deeps[k])), float(np.median(exps[k])))
                             for k in MAIN}}
    plc, pval, rand = pl["a"]["plc"], pl["a"]["pval"], pl["a"]["rand"]
    holm = hg._holm(pval)
    t_plc = time.time() - t0 - t_main

    # grid with the full exam
    grid = []
    for f in SIZES:
        for kk in CONFS:
            Xg = _exam(ex, book.pos(("scale", f, kk)), K2)
            sg = _score(ex, Xg)
            tg = _score(ex, _exam(ex, book.pos(("const", sg["exp_on"])), K2))
            grid.append({"f": f, "k": kk, "s": sg, "tw": tg,
                         "score_ok": bool(bc._score_ge(sg, S["ma50"])),
                         "loss_ok": sg["c_deep_med"] > S["ma50"]["c_deep_med"] + 1e-12})
    for g in grid:
        g["meets"] = g["score_ok"] and g["loss_ok"]

    # positions on BTC daily, years
    pos = {k: book.pos(SPECS[k])["btc"] for k in SPECS}
    pos["CONST_S10"] = book.pos(("const", tw["S10"]["c"]))["btc"]
    years = {k: bc._years(v, close, FIRST) for k, v in pos.items()}
    conf_btc = book.conf("btc", "scale", CONF_K)
    obs = _observe(close)
    return {"ex": ex, "book": book, "F": F, "X": X, "S": S, "tw": tw, "TS": TS, "same": same, "small_s": small_s,
            "plc": plc, "pval": pval, "holm": holm, "rand": rand, "plb": pl["b"], "grid": grid, "pos": pos, "years": years,
            "conf_btc": conf_btc, "obs": obs, "problem": problem, "t_main": t_main, "t_plc": t_plc,
            "secs": time.time() - t0}


# ---------- the study ----------

def _f(v) -> str:
    """fraction -> signed percent"""
    return _pct(v * 100) if _ok(v) else "—"


def _pts(v) -> str:
    """fraction difference -> signed points"""
    return f"{v * 100:+.1f} 點" if _ok(v) else "—"


def _of(a: int, n: int) -> str:
    return f"{a}/{n}" if n else "—"


def _yn(v: bool) -> str:
    return "過" if v else "不過"


def _rank(x: dict) -> str:
    return _share(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）")


def _four(x: dict) -> str:
    return "／".join(_share(v) for v in x["checks"])


def _trip(v: tuple) -> str:
    m_, sh, b, n = v
    return f"{m_ * 100 / n:.0f}%｜{sh * 100 / n:.0f}%｜{b * 100 / n:.0f}%" if n else "—"


def _times(g: float) -> str:
    return f"{g:,.0f} 倍" if g >= 10 else f"{g:,.2f} 倍"


def _med(v) -> float | None:
    v = [x for x in v if _ok(x)]
    return float(np.median(v)) if v else None


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, S, TS, X = data["close"], r["ex"], r["S"], r["TS"], r["X"]
    base = S["ma50"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    n_d, n_c = base["d_n"], base["c_n"]

    # ----- Part 1 -----
    ob = r["obs"]
    T, c = ob["T"], ob["c"]
    N = len(T)
    run = [t for t in T if t["kind"] == "runner"]
    fiz = [t for t in T if t["kind"] == "fizzle"]
    sml = [t for t in T if t["kind"] == "small"]
    nr, nf_, ns = len(run), len(fiz), len(sml)

    rows_a = []
    for lo, hi, lab in BINS:
        g = [t for t in T if lo < t["res"] <= hi]
        rows_a.append([lab, f"{len(g)}", f"{len(g) / N * 100:.0f}%" if N else "—",
                       _f(_med([t["best"] for t in g])), _f(_med([t["worst"] for t in g])),
                       f"{_med([t['len'] for t in g]):.0f} 天" if g else "—"])
    rows_a.append(["全部", f"{N}", "100%", _f(_med([t["best"] for t in T])), _f(_med([t["worst"] for t in T])),
                   f"{_med([t['len'] for t in T]):.0f} 天"])
    tA = {"caption": (f"第一部分（先觀察，還沒有任何規則）：比特幣日線 {_d(T[0]['day'])} 起，50 日均線做完的 {N} 筆交易"
                      f"（{len(ob['open'])} 筆還沒結束、不算）。結果＝出場收盤÷進場收盤−1，扣來回成本 0.20%。"
                      f"量到的：跑起來（≥ +20%）{nr} 筆、小賺 {ns} 筆、熄火（≤ 0）{nf_} 筆。"
                      "「期間最好／最差」＝從進場那天到出場那天，收盤比進場價最高高多少、最低低多少（各組中位數）。"),
          "columns": ["交易結果", "筆數", "占比", "期間最好時（中位數）", "期間最差時（中位數）", "持有天數（中位數）"],
          "rows": rows_a}

    full_chain = float(np.prod([1 + t["res"] for t in T]))
    rows_b = []
    lv_info = {}
    for k in OBS_LEVELS:
        jr = [t for t in run if k in t["reach"]]
        days_ = _med([t["reach"][k] - t["s"] for t in jr])
        above = float(np.mean([c[t["reach"][k]] / c[t["s"]] - 1 for t in jr])) if jr else None
        give = float(np.mean([t["res"] - _scaled(t, c, k) for t in run])) if run else None
        fc = [t for t in fiz if k in t["reach"]]
        save = float(np.mean([_scaled(t, c, k) - t["res"] for t in fiz])) if fiz else None
        sm_d = float(np.mean([_scaled(t, c, k) - t["res"] for t in sml])) if sml else None
        chain = float(np.prod([1 + _scaled(t, c, k) for t in T]))
        lv_info[k] = {"jr": len(jr), "fc": len(fc), "give": give, "save": save, "above": above, "chain": chain,
                      "small": sm_d, "days": days_}
        rows_b.append([f"+{k * 100:.0f}%", _of(len(jr), nr), f"{days_:.0f} 天" if _ok(days_) else "—", _f(above),
                       _pts(-give) if _ok(give) else "—",
                       f"{_of(len(fc), nf_)}（其餘 {nf_ - len(fc)} 筆整筆只拿一半）", _pts(save), _pts(sm_d),
                       f"{_times(full_chain)}｜{_times(chain)}"])
    tB = {"caption": ("第一部分：分批進場的算術（描述，不是考試）。假設每次進場先買一半、收盤第一次比進場價高某個門檻時再買另一半，"
                      "兩份都抱到均線出場（不每天調回比例；每份買賣各付 0.10%）。每一列：跑起來的交易有幾筆到得了門檻（加碼成功）、"
                      "從進場到門檻的天數中位數、加碼那天的價格平均比進場價高多少（＝少賺的來源）、跑起來的每筆平均少賺幾點（整個部位的百分點）、"
                      "熄火的交易裡有幾筆曾經到過門檻（＝假確認：加滿之後還是賠）、熄火的每筆平均少虧幾點、小賺的每筆平均差幾點，"
                      f"最後一欄是 {N} 筆交易接起來：全部滿倉｜分批（只算持有期間，不含空手期間）。"),
          "columns": ["確認門檻", "跑起來的加碼成功", "到門檻天數（中位數）", "加碼價高於進場（平均）", "跑起來每筆少賺（平均）",
                      "熄火的曾到門檻（假確認）", "熄火每筆少虧（平均）", "小賺每筆差（平均）", "全部接起來：滿倉｜分批"],
          "rows": rows_b}

    rows_c = []
    for t in sorted(run, key=lambda t: t["day"]):
        dd_ = "｜".join(f"{t['reach'][k] - t['s']}" if k in t["reach"] else "—" for k in RUN_LEVELS)
        sc = _scaled(t, c, CONF_K)
        rows_c.append([_d(t["day"]), f"{t['len']}", _f(t["res"]), f"{_f(t['best'])}｜{_f(t['worst'])}", dd_,
                       _f(t["dip"].get(CONF_K)), _f(sc), _pts(sc - t["res"])])
    tC = {"caption": (f"第一部分：{nr} 筆跑起來的交易逐筆列出。「到 +5%｜+10%｜+15% 的天數」＝從進場收盤算起第幾天收盤第一次到那個漲幅；"
                      "「到 +10% 前最深跌破」＝進場到第一次 +10% 之間，收盤最低比進場價低多少（0.0% 表示一路沒跌破進場價）。"
                      "「分批的這筆」＝先一半、+10% 加另一半、抱到出場的結果；最後一欄是比滿倉少幾點。"),
          "columns": ["進場日", "持有天數", "滿倉結果", "期間最好｜最差", "到 +5%｜+10%｜+15% 的天數", "到 +10% 前最深跌破",
                      "分批的這筆（+10% 加滿）", "比滿倉"],
          "rows": rows_c}

    rows_d = []
    per_info = {}
    for lab, y0, y1 in PERIODS:
        g = [t for t in T if y0 <= t["day"].year <= y1]
        gr = [t for t in g if t["kind"] == "runner"]
        gf = [t for t in g if t["kind"] == "fizzle"]
        gs = [t for t in g if t["kind"] == "small"]
        f5 = sum(0.05 in t["reach"] for t in gf)
        f10 = sum(0.10 in t["reach"] for t in gf)
        d10 = _med([t["reach"][0.10] - t["s"] for t in gr if 0.10 in t["reach"]])
        dip = _med([t["dip"][0.10] for t in gr if 0.10 in t["dip"]])
        net = float(np.sum([_scaled(t, c, CONF_K) - t["res"] for t in g])) if g else None
        ch_f = float(np.prod([1 + t["res"] for t in g])) if g else None
        ch_s = float(np.prod([1 + _scaled(t, c, CONF_K) for t in g])) if g else None
        per_info[lab] = (len(g), len(gr), len(gf), f10, ch_f, ch_s)
        rows_d.append([lab, f"{len(g)}", f"{len(gr)}｜{len(gs)}｜{len(gf)}", f"{f5}｜{f10}（共 {len(gf)}）",
                       f"{d10:.0f} 天" if _ok(d10) else "—", _f(dip), _pts(net),
                       f"{_times(ch_f)}｜{_times(ch_s)}" if g else "—"])
    n2010 = sum(t["day"].year < PERIODS[0][1] for t in T)
    tD = {"caption": (f"第一部分（依進場年份分三段）：{n2010} 筆 2010 年進場的交易只算在全期。「熄火曾到 +5%｜+10%」＝熄火的交易裡曾經比進場價高 5%／10% 的筆數；"
                      "「分批合計差」＝這一段每筆「先一半、+10% 加滿」減「滿倉」的點數直接加總（正數＝分批比較好）；最後一欄是這段的交易接起來 滿倉｜分批。"),
          "columns": ["期間", "交易筆數", "跑起來｜小賺｜熄火", "熄火曾到 +5%｜+10%", "跑起來到 +10% 天數（中位數）",
                      "跑起來到 +10% 前最深跌破（中位數）", "分批合計差（+10%）", "接起來：滿倉｜分批"],
          "rows": rows_d}

    # ----- Part 3 bar -----
    tw, plc, holm, pval, rand = r["tw"], r["plc"], r["holm"], r["pval"], r["rand"]
    bar = {}
    for k in MAIN:
        x = S[k]
        q = {"score": bool(bc._score_ge(x, base)), "loss": x["c_deep_med"] > base["c_deep_med"] + 1e-12,
             "twin": hg._above(x, TS[k]), "plc": plc[k][None] <= PLC_BAR + 1e-12}
        q["all"] = all(q.values())
        q["holm"] = q["all"] and holm[k] <= HOLM_ALPHA + 1e-12
        bar[k] = q
    plb = r["plb"]
    rows1 = []
    for k in MAIN:
        q, x, t_ = bar[k], S[k], TS[k]
        rm, rd, re_ = rand[k]
        bm, bd, be = plb["rand"][k]
        rows1.append([RNAME[k], f"{_rank(x)} 對 {_rank(base)} → {_yn(q['score'])}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])} → {_yn(q['loss'])}",
                      f"{_rank(x)} 對 {_rank(t_)}（雙胞胎永遠 {tw[k]['c'] * 100:.1f}%，虧損 {_pct(t_['c_deep_med'])}） → {_yn(q['twin'])}",
                      f"{_share(plc[k][None])}（亂加碼的中位數：排名 {_share(rm)}、虧損 {_pct(rd)}、持有日平均 {re_ * 100:.0f}%；"
                      f"真的 {_share(r['small_s'][k][None]['low'])}、{x['exp_on'] * 100:.0f}%） → {_yn(q['plc'])}",
                      f"{_share(plb['plc'][k][None])}（中位數：排名 {_share(bm)}、虧損 {_pct(bd)}、持有日平均 {be * 100:.0f}%）",
                      f"{pval[k]:.3f} → 校正後 {holm[k]:.3f}",
                      "校正後仍過" if q["holm"] else ("四項都過、校正後不過" if q["all"] else "沒有四項都過")])
    t1 = {"caption": ("第三部分的標準（跑之前寫好，四項都要過）：(1) 排名分數不低於 50 日均線；(2) 最深虧損中位數比均線淺；"
                      "(3) 排名分數比「固定比例雙胞胎」高（雙胞胎＝均線說持有時永遠只放固定比例，比例剛好等於這條規則在均線持有日子的平均持有比例，"
                      "所以兩者平均放的一樣多，只差在什麼時候放）；(4) 隨機錯位（安慰劑）：每筆交易用同樣的機率決定要不要加滿，"
                      f"但加滿的日子在這筆交易裡隨便抽，{DRAWS} 次；排名一樣好或更好的比例 ≤ {PLC_BAR * 100:.0f}% 才算過"
                      f"（一般人做法每格模擬 {PL_SIMS} 次，真的規則也用同樣方式重算一次再比）。四條規則一起用 Holm 校正："
                      f"p＝（一樣好的次數＋1）÷（{DRAWS}＋1），校正後 ≤ {HOLM_ALPHA:.2f} 才算「校正後仍過」。純靠運氣，(4) 這項四條裡平均約 0.2 條會過。"
                      "（4b）是試跑之後才加的（不算進判斷）：試跑時看到 (4) 的亂加碼平均放得比真的規則少很多"
                      "（真的規則加滿的多半是後來大漲的長交易，亂抽的多半是很快出場的短交易），對照太容易贏；"
                      "（4b）改成「和真的規則加滿的是同一批交易，只是加滿的日子在那筆交易裡隨便抽」，只剩「什麼時候加」不一樣。"
                      "（4b）知道哪些交易後來到過門檻，是比較難贏的對照。"),
          "columns": ["規則", "(1) 排名 對 均線", "(2) 虧損中位數 對 均線", "(3) 排名 對 固定比例雙胞胎", "(4) 隨機錯位",
                      "(4b) 同一批交易、加滿日子亂抽（試跑後加）", "p → Holm 校正後", "結果"],
          "rows": rows1}

    def exam_row(name: str, x: dict) -> list:
        return [name, _four(x), _rank(x), f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}",
                f"{x['exp'] * 100:.0f}%（持有日 {x['exp_on'] * 100:.0f}%）" if _ok(x.get("exp_on")) else "—",
                f"{x['sw_y']:.1f}", f"{x['turn_y']:.1f} 倍｜{x['turn_y'] * COST * 100:.2f}%｜{x['turn_y'] * P_COST * 100:.2f}%",
                _num(x["ratio"], 2)]

    rows2 = [exam_row("50 日均線（基準）", base)]
    for k in ("S10", "ST", "SH", "SM", "H_NFCI", "SG"):
        rows2.append(exam_row(RNAME[k], S[k]))
        rows2.append(exam_row(f"　{k} 的固定比例雙胞胎（永遠 {tw[k]['c'] * 100:.1f}%）", TS[k]))
    t2 = {"caption": (f"第三部分：公平考試。四項＝「報酬贏過同一天起一直抱著」的比例：比特幣 UTC 0 點日線 {n_d} 個起算月／"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {COST * 100:.2f}%）／同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、每次改變持有比例都付 {P_COST * 100:.2f}%、每次要加碼的那天都有一成機率漏掉——第一筆和加碼那筆都算，每格 {SIMS} 次）／"
                      "其他資產 ETH、SOL、BNB、XRP、MSTR 用自己的價格和 50 日均線。排名分數＝四項裡最低的，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      f"最深虧損＝從起算日到現在帳戶從高點往下跌最多的幅度，{n_c:,} 格照課本做的中位數｜最差一格。"
                      f"持有比例＝{FIRST[:4]} 年起比特幣日線平均放幾成（括號是均線說持有那些日子的平均）。換手次數＝每年持有比例改變幾次；"
                      "換手量＝每年買賣金額是帳戶的幾倍（含部分持倉每天調回比例的小額買賣）｜照課本 0.10% 每年付掉的成本｜一般人 0.25% 每年付掉的成本。"
                      "每單位虧損的報酬＝每格「每年報酬 ÷ 最深虧損」的中位數。"),
          "columns": ["規則", "四項：日線／課本／一般人／其他資產", "排名分數", "最深虧損：中位數｜最差", "持有比例",
                      "每年換手次數", "每年換手量｜成本 0.10%｜0.25%", "每單位虧損的報酬"],
          "rows": rows2}

    # ----- years -----
    yrs = sorted(r["years"]["ma50"])
    ycols = ("S10", "CONST_S10", "ST", "SH", "SM", "H_NFCI", "SG")
    rows3, diffs = [], {k: {} for k in ycols}
    for y in yrs:
        m = r["years"]["ma50"].get(y)
        cells = []
        for k in ycols:
            v = r["years"][k].get(y)
            if _ok(v) and _ok(m):
                diffs[k][y] = v - m
            cells.append(_pp(v - m) if _ok(v) and _ok(m) else "—")
        rows3.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(m)] + cells)
    rows3.append(["每年平均", "—"] + [_pp(float(np.mean(list(diffs[k].values())))) for k in ycols])
    rows3.append(["贏均線的年數", "—"] + [f"{sum(v > 0.05 for v in diffs[k].values())}/{len(diffs[k])}" for k in ycols])
    t3 = {"caption": (f"第三部分：UTC 0 點日線照課本做（成本 {COST * 100:.2f}%）每個日曆年的報酬，{FIRST[:4]} 年起。第二欄是 50 日均線本身，"
                      "其餘是各規則減 50 日均線（百分點；負數＝那年比均線少賺）。CONST＝S10 的固定比例雙胞胎。"),
          "columns": ["年份", "50 日均線", "S10 減均線", "CONST 減均線", "ST 減均線", "SH 減均線", "SM 減均線", "H_NFCI 減均線", "SG 減均線"],
          "rows": rows3}

    # ----- grid -----
    grid = r["grid"]
    gcount = sum(g["meets"] for g in grid)
    robust = gcount >= ROBUST_N
    rows4 = []
    for g in grid:
        x, t_ = g["s"], g["tw"]
        rows4.append([f"{g['f'] * 100:.0f}%" + ("（S10）" if g["f"] == START_F and g["k"] == CONF_K else ""),
                      f"+{g['k'] * 100:.0f}%", _rank(x), _pct(x["c_deep_med"]), f"{x['exp'] * 100:.0f}%",
                      f"{_rank(t_)}｜{_pct(t_['c_deep_med'])}", _yn(g["score_ok"]), _yn(g["loss_ok"]),
                      "過" if g["meets"] else "不過"])
    t4 = {"caption": (f"第三部分（換設定）：起手比例 33%／50%／67% × 確認門檻 +5%／+10%／+15%／+20%，12 種，全部用完整考試。"
                      f"過＝排名分數不低於均線（{_rank(base)}）而且最深虧損中位數比均線（{_pct(base['c_deep_med'])}）淺。"
                      f"事先寫好：12 種裡至少 {ROBUST_N} 種過才算「穩健」。這次 {gcount}/12 → {'穩健' if robust else '不算穩健'}。"
                      "旁邊附上每種設定自己的固定比例雙胞胎（只報告）。"),
          "columns": ["起手比例", "確認門檻", "排名分數", "最深虧損中位數", "持有比例", "雙胞胎：排名｜虧損", "排名不低於均線",
                      "虧損比均線淺", "兩項都過"],
          "rows": rows4}

    # ----- Part 4 -----
    vs = {k: _vs(ex, X[k], X["ma50"]) for k in ("S10", "ST", "SH", "SM", "H_NFCI", "SG")}
    vs["CONST_S10"] = _vs(ex, tw["S10"], X["ma50"])
    vs["CONST_SG"] = _vs(ex, tw["SG"], X["ma50"])
    dy = np.array([d.year for d in ex.dlabels])
    dyears = sorted(set(dy.tolist()))

    def bad(mask) -> str:
        b = [str(y) for y in dyears if mask[dy == y].sum() < (dy == y).sum() / 2]
        return "、".join(b) if b else "沒有"

    p4 = [("S10", "S10"), ("ST", "ST"), ("SH", "SH"), ("SM", "SM"), ("SG", "SG"), ("H_NFCI", "H_NFCI"),
          ("CONST_S10", "S10 的固定比例雙胞胎"), ("CONST_SG", "SG 的固定比例雙胞胎")]
    rows5 = []
    for key, lab in p4:
        v = vs[key]
        rows5.append([lab, _trip(v["d"]), _trip(v["c"]), _trip(v["p"]), bad(v["d_more"]), bad(v["d_both"]),
                      _reading(v["c"])])
    t5 = {"caption": ("第四部分：老闆問的問題，直接數。每條規則和 50 日均線比，同一個起算點：最後錢比較多的比例｜最深虧損比較淺的比例｜兩個都做到的比例。"
                      f"比特幣日線 {n_d} 個起算月；{n_c:,} 格照課本做；一般人做法 {n_c:,} 格 × {SIMS} 次（規則的第 i 次和均線的第 i 次配對，漏掉哪次是各自抽的）。"
                      "後兩欄：比特幣日線起算月裡，那一年 12 個起算月「錢更多」／「兩個都做到」不到一半的起算年。"
                      "白話讀法照課本格、事先寫好：多、淺、都三個都過半＝賺更多又虧更少；淺過半但多不過半＝虧更少、但是靠少賺換來的。"),
          "columns": ["規則", "日線：多｜淺｜都", "課本格：多｜淺｜都", "一般人：多｜淺｜都", "錢更多不到一半的起算年",
                      "兩個都做到不到一半的起算年", "白話讀法（課本格）"],
          "rows": rows5}

    # ----- Part 5 -----
    sg, s10, hn, tsg = S["SG"], S["S10"], S["H_NFCI"], TS["SG"]
    better = s10 if not bc._score_ge(hn, s10) else hn
    p5a = bool(bc._score_ge(sg, better))
    p5b = sg["c_deep_med"] > max(s10["c_deep_med"], hn["c_deep_med"]) + 1e-12
    p5c = hg._above(sg, tsg)
    adds = p5a and p5b and p5c
    rows6 = [["(a) 排名分數不低於兩者較高的", f"SG {_rank(sg)}；S10 {_rank(s10)}；H_NFCI {_rank(hn)}", _yn(p5a)],
             ["(b) 最深虧損中位數比兩者都淺", f"SG {_pct(sg['c_deep_med'])}；S10 {_pct(s10['c_deep_med'])}；H_NFCI {_pct(hn['c_deep_med'])}", _yn(p5b)],
             ["(c) 排名分數比自己的固定比例雙胞胎高", f"SG {_rank(sg)}；雙胞胎（永遠 {tw['SG']['c'] * 100:.1f}%）{_rank(tsg)}、虧損 {_pct(tsg['c_deep_med'])}", _yn(p5c)],
             ["持有比例", f"SG {sg['exp'] * 100:.0f}%；S10 {s10['exp'] * 100:.0f}%；H_NFCI {hn['exp'] * 100:.0f}%；均線 {base['exp'] * 100:.0f}%", "—"],
             ["課本格：多｜淺｜都（對均線）", f"SG {_trip(vs['SG']['c'])}；S10 {_trip(vs['S10']['c'])}；H_NFCI {_trip(vs['H_NFCI']['c'])}", "—"],
             ["結論（事先寫好：三項都過＝有加分）", "—", "有加分" if adds else "沒有超過兩者較好的那個"]]
    t6 = {"caption": ("第五部分：和半倉條件一起用。H_NFCI（half_gate）是均線持有、但 NFCI 金融情勢比自己一年平均緊時只放一半；"
                      "S10 是剛進場、還沒被證明時只放一半。兩個在不同時候起作用。SG＝每天取兩者中比較小的那個比例。"),
          "columns": ["項目", "數字", "結果"], "rows": rows6}

    # ----- Part 6 -----
    parts = {"pre": "2019 年以前起算", "post": "2019 年起起算"}
    SP = {k: {p: _score(ex, X[k], p) for p in parts} for k in X}
    TSP = {k: {p: _score(ex, tw[k], p) for p in parts} for k in ("S10", "SG")}
    VP = {k: {p: _vs(ex, X[k], X["ma50"], p) for p in parts} for k in ("S10", "ST", "SH", "SM", "SG", "H_NFCI")}
    VP["CONST_S10"] = {p: _vs(ex, tw["S10"], X["ma50"], p) for p in parts}

    def cellx(x) -> str:
        return f"{_rank(x)}（{_four(x)}）"

    rows7 = []
    for key, lab in (("ma50", "50 日均線"), ("S10", "S10"), ("CONST_S10", "S10 的固定比例雙胞胎"), ("ST", "ST"), ("SH", "SH"),
                     ("SM", "SM"), ("H_NFCI", "H_NFCI"), ("SG", "SG")):
        row = [lab]
        for p in parts:
            x = TSP["S10"][p] if key == "CONST_S10" else SP[key][p]
            row += [cellx(x), f"{_pct(x['d_deep_med'])}｜{_pct(x['c_deep_med'])}",
                    _trip(VP[key][p]["c"]) if key in VP else "—"]
        rows7.append(row)
    mpre = _masks(ex, "pre")
    pre_d = int(mpre["d"].sum())
    pre_c = int(mpre["c"].sum()) * 24
    pre_o = sum(int(v.sum()) for v in mpre["o"].values())
    t7 = {"caption": (f"第六部分：誠實地切開。S10 是在 2019 年起的起算日上先被看到的，所以 2019 年以前的起算日是比較乾淨的證據。"
                      f"2019 年以前：日線 {pre_d} 個起算月（2015–2018）、課本格和一般人只有 2018 年的 {pre_c} 格（小時線 2017-08 才有）、"
                      f"其他資產 {pre_o} 個起算月（都在 2018 年）；2019 年起：日線 {n_d - pre_d} 個、課本格 {n_c - pre_c:,} 格。"
                      "每格：排名分數（四項：日線／課本／一般人／其他資產）；最深虧損中位數 日線｜課本格；課本格對均線 多｜淺｜都。"),
          "columns": ["規則", "2019 前：排名（四項）", "2019 前：虧損中位數 日線｜課本格", "2019 前：多｜淺｜都",
                      "2019 起：排名（四項）", "2019 起：虧損中位數 日線｜課本格", "2019 起：多｜淺｜都"],
          "rows": rows7}

    rows8 = []
    split_ok = {}
    for p, lab in (("pre", "2019 年以前起算"), ("post", "2019 年起起算"), (None, "全部")):
        x = S["S10"] if p is None else SP["S10"][p]
        b_ = base if p is None else SP["ma50"][p]
        t_ = TS["S10"] if p is None else TSP["S10"][p]
        q1 = bool(bc._score_ge(x, b_))
        q2 = bool(_ok(x["c_deep_med"]) and x["c_deep_med"] > b_["c_deep_med"] + 1e-12)
        q3 = bool(hg._above(x, t_))
        q4 = bool(plc["S10"][p] <= PLC_BAR + 1e-12)
        split_ok[p] = (q1, q2, q3, q4)
        rows8.append([lab, f"{_rank(x)} 對 {_rank(b_)} → {_yn(q1)}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(b_['c_deep_med'])}（日線 {_pct(x['d_deep_med'])} 對 {_pct(b_['d_deep_med'])}） → {_yn(q2)}",
                      f"{_rank(x)} 對 {_rank(t_)} → {_yn(q3)}", f"{_share(plc['S10'][p])} → {_yn(q4)}",
                      _share(plb["plc"]["S10"][p]),
                      "四項都過" if all(split_ok[p]) else f"過 {sum(split_ok[p])} 項"])
    t8 = {"caption": ("第六部分：S10 的四項標準分開看。隨機錯位那一項在每一段都用同樣 200 次亂加碼、只看那一段的起算點重新評分；"
                      "（4b）是試跑後才加的對照（同一批交易、加滿日子亂抽），只報告。"),
          "columns": ["起算點", "(1) 排名 對 均線", "(2) 虧損中位數 對 均線", "(3) 排名 對 雙胞胎", "(4) 隨機錯位", "(4b)（只報告）", "結果"],
          "rows": rows8}

    # ----- today -----
    last = close.index[-1]
    pm = r["pos"]

    def amount(v) -> str:
        if not _ok(v) or v < 0.25:
            return "現金"
        return "持有" if v >= 0.75 else "半倉"

    sb = r["book"].S["btc"]
    s_last, e_last, t_last = int(sb.ent[-1]), int(sb.end[-1]), int(r["conf_btc"][-1])
    cs = float(close.iloc[s_last])
    ent_txt = f"{_d(close.index[s_last])} 收盤 {cs:,.0f} 美元站上均線"
    ent_txt += (f"；{_d(close.index[e_last])} 跌回均線下、已出場" if e_last < len(close) else "；到最後一根日線仍在均線上")
    ent_txt += (f"；{_d(close.index[t_last])} 收盤 {float(close.iloc[t_last]):,.0f} 美元第一次比進場價高 10%，S10 在那天加滿"
                if t_last >= 0 else "；這段期間收盤從沒有比進場價高 10%，S10 一直只放一半（從沒被確認）")
    cur = (last + DAY).to_period("M") - 1
    gv = r["F"].get(cur, np.nan)
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": "比特幣：50 日均線｜S10｜SG", "value": "｜".join(amount(pm[k][-1]) for k in ("ma50", "S10", "SG"))},
        {"label": "最近一次 50 日均線進場", "value": ent_txt},
        {"label": f"NFCI 條件（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": "有利" if gv == 1 else "不利（H_NFCI、SG 在均線持有時只放一半）" if gv == 0 else "不知道（當作不利）"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態（持有＝100%、半倉＝部分、現金＝0%），不是買賣建議"},
    ]

    # ----- verdict -----
    q = bar["S10"]
    word = ("站得住" if q["holm"] and robust else "部分站得住" if q["all"] else "這次沒有達到")
    passed = [k for k in MAIN if bar[k]["all"]]
    passed_h = [k for k in MAIN if bar[k]["holm"]]
    v10, l10 = vs["S10"], lv_info[CONF_K]
    pre_ok, post_ok = split_ok["pre"], split_ok["post"]
    failed = [n for i, n in (("score", "排名"), ("loss", "虧損"), ("twin", "雙胞胎"), ("plc", "隨機錯位")) if not q[i]]
    verdict = (
        f"{word}（事先寫好的讀法）。先觀察（量到的）：均線做完的 {N} 筆交易裡跑起來 {nr} 筆、熄火 {nf_} 筆；"
        f"以 +10% 當門檻，跑起來的 {l10['jr']}/{nr} 筆都加得到（加碼價平均比進場高 {_f(l10['above'])}，每筆平均少賺 {l10['give'] * 100:.1f} 點），"
        f"熄火的 {l10['fc']}/{nf_} 筆曾經到過 +10%（假確認），其餘整筆只放一半，熄火每筆平均少虧 {l10['save'] * 100:.1f} 點。"
        f"考試（{n_d} 個日線起算月、{n_c:,} 格）：50 日均線排名 {_rank(base)}、最深虧損中位數 {_pct(base['c_deep_med'])}；"
        f"S10 排名 {_rank(S['S10'])}、虧損 {_pct(S['S10']['c_deep_med'])}；平均放一樣多的固定比例雙胞胎（永遠 {tw['S10']['c'] * 100:.0f}%）"
        f"{_rank(TS['S10'])}、{_pct(TS['S10']['c_deep_med'])}；亂加碼對照的排名中位數 {_share(rand['S10'][0])}、隨機錯位 {_share(plc['S10'][None])}"
        f"（試跑後加的、比較難的同一批交易對照：{_share(plb['plc']['S10'][None])}，排名中位數 {_share(plb['rand']['S10'][0])}，不算進判斷）；"
        f"四項過 {4 - len(failed)}/4" + (f"（沒過：{'、'.join(failed)}）" if failed else "")
        + f"。S10、ST、SH、SM 四條裡四項都過 {len(passed)} 條" + (f"（{'、'.join(passed)}）" if passed else "")
        + f"、Holm 校正後 {len(passed_h)} 條；純靠運氣約 0.2 條。換設定：12 種裡 {gcount} 種排名不低於均線又虧得淺 → {'穩健' if robust else '不算穩健'}。"
        f"老闆的問題（課本格）：S10 錢更多 {v10['c'][0] * 100 / v10['c'][3]:.0f}%、虧損更淺 {v10['c'][1] * 100 / v10['c'][3]:.0f}%、"
        f"兩個都做到 {v10['c'][2] * 100 / v10['c'][3]:.0f}% → {_reading(v10['c'])}；一般人做法兩個都做到 {v10['p'][2] * 100 / v10['p'][3]:.0f}%；"
        f"但 2015 年起的 {n_d} 個日線起算月只有 {v10['d'][2]}/{v10['d'][3]} 個兩個都做到（虧損更淺 {v10['d'][1]}/{v10['d'][3]}），"
        f"錢更多不到一半的起算年：{bad(v10['d_more'])}。"
        f"和半倉條件合用的 SG：{'有加分' if adds else '沒有超過兩者較好的那個'}（排名 {_rank(sg)}、虧損 {_pct(sg['c_deep_med'])}）。"
        f"切開看：2019 年以前起算（比較乾淨）S10 四項過 {sum(pre_ok)} 項，2019 年起過 {sum(post_ok)} 項。"
        "推測（不是量到的）：放得比較少本身就會讓虧損變淺、大漲時少賺，所以真正要看的是它有沒有贏過平均放一樣多的雙胞胎和亂加碼的對照。"
        f"整段歷史只有 {nr} 筆跑起來的交易撐起全部獲利，資料量很小。不是買賣建議。")

    notes = [
        ("事先寫死的設計和標準（照原文，跑之前寫好）：S10 = 50% from the entry close, 100% from the first close >= +10% "
         "above the entry close, all out when ma50 turns off. Grid 33/50/67% x +5/+10/+15/+20%. ST = 100% from the 10th "
         "consecutive close above the 50-day average; SH = first new 20-day closing high after entry; SM = 50-day average "
         "above its value 10 closes earlier. CONST = constant share equal to the rule's mean size on ma50-on days; RAND = "
         "same share of trades confirmed per series, add-on day drawn uniformly inside the trade, 200 draws. S10 bar (all "
         "four): ranking >= ma50, strictly shallower median deepest loss, ranking strictly above CONST, placebo <= 5%; Holm "
         "over S10/ST/SH/SM; grid robust if >= 9 of 12 meet ranking >= ma50 AND shallower loss. SG = min(S10, H_NFCI); "
         "adds if ranking >= the better of the two AND shallower loss than both AND above its CONST twin. Overall: 站得住 = "
         "S10 all four + Holm + grid robust; 部分站得住 = all four but not Holm or grid."),
        ("成本和漏掉進場怎麼算（照實說）：照課本是每次改變持有比例都付 0.10%（進場那一份、加碼那一份、出場全部），"
         "照一般人是 0.25%；部分持倉（例如 50%）每天收盤會調回原比例，這個小額買賣也照同樣費率付，固定比例雙胞胎和 H_NFCI 也一樣。"
         "一般人漏掉進場用 robust_trend 的做法：每一個「想要的持有比例一天內增加超過 10 個百分點」的收盤（第一筆進場和加碼那一筆都算）"
         "各自有 10% 機率沒做到，每天獨立抽；沒買到的部分要等到想要的比例降下同樣多（也就是均線出場）才歸零。"
         "所以加碼那一筆漏掉，那筆交易剩下的時間就只有起手的比例；第一筆漏掉、加碼有做到，就只有加碼那一份。"),
        ("已經知道的事（別的研究量到的）：at_the_signal——均線 148 次已結束的進場，熄火 105 次（平均 -3.6%）、跑起來 23 次（平均 +191%、中位數 +54%）；"
         "它的對照版本 E0（就是這裡的 S10，只用 2019 年起的起算日）排名 77%、最深虧損中位數 -40.2%（均線 79%、-52.4%）。"
         "whipsaw_anatomy——均線 148 次下車裡 104 次是假跌破。vol_sizing——2018 年起完全不看漲跌、只放固定比例："
         "一直抱著每年 +18.7%、最深虧 -81.2%；固定只放 66% 是 +17.3%、-64.5%；順勢混合 +31.7%、-48.2%，同波動的固定 58% 是 +16.1%、-58.6%。"
         "也就是「放得少就虧得少」本身是量得到的，這就是為什麼這裡一定要比固定比例雙胞胎。"
         "trend_anatomy（量的是之前的 55／20 規則）：最好的 5 筆交易占全部成長 96.3%。"),
        (f"這次第一部分用的是 {_d(T[0]['day'])} 起的日線、{N} 筆已結束的交易（at_the_signal 跑的時候是 148 筆；"
         "2026-11-19 修正了 2017 年以前 blockchain.info 價格的日期標記，所以筆數和平均可能和那份研究差一點）。"
         "第一部分的「分批算術」是兩份各自買進抱著，考試裡則是每天收盤調回比例，兩者數字不會完全一樣。"),
        ("第四部分怎麼讀：「錢更多」＝從起算日到現在的每年報酬比均線高；「虧損更淺」＝這段期間最深的跌幅比均線淺。"
         "相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"),
        ("限制：(1) 分批進場多了交易，成本已照上面算進去，但真實滑價可能更大；(2) 只有一個資產的歷史，二十幾筆跑起來的交易撐起全部獲利，"
         "少數幾筆交易的加碼時間就能左右結果；(3) 其他資產用自己的價格和自己的均線，是同一種市場行為，不是完全獨立的證據；"
         "(4) 放得比較少的規則幾乎一定虧得比較淺，所以「虧損比均線淺」本身不算證據，真正的比較是固定比例雙胞胎和亂加碼對照；"
         f"(5) 2019 年以前的課本格只有 2018 年起算的 {pre_c} 格（剛好是熊市年），日線起算月才涵蓋 2015–2018。"
         "「沒有看到」只表示用這個定義和這些資料沒有看到，不表示不存在。"),
        (f"重算檢查：考試照 half_gate 的寫法重寫一次以留下每個起算點的數字；50 日均線和 H_NFCI 的結果和 business_cycle／half_gate 的考試"
         f"{'完全一樣' if r['same'] else '不一樣（要查）'}。H_NFCI 的 NFCI 來自{data['where']}。"),
        f"隨機的部分都用固定種子。這次計算用了 {time.time() - t0:.0f} 秒（主要考試 {r['t_main']:.0f} 秒、隨機錯位 {r['t_plc']:.0f} 秒）。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "blockchain.info 長期每日價格（2017-08 以前）", "url": "https://api.blockchain.info/charts/market-price"},
        {"name": "幣安現貨 K 線（BTCUSDT 日線與 1 小時線；ETHUSDT、SOLUSDT、BNBUSDT、XRPUSDT 日線）", "url": "https://api.binance.com/api/v3/klines"},
        {"name": "幣安現貨 K 線備用主機", "url": "https://data-api.binance.vision/api/v3/klines"},
        {"name": "Yahoo Finance 日線：MSTR（調整後收盤）", "url": "https://query1.finance.yahoo.com/v8/finance/chart/MSTR"},
        ({"name": "FRED：NFCI", "url": nf.FRED_URL} if data["where"] == "FRED"
         else {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": nf.CHI_URL}),
    ]

    return {
        "id": ID,
        "title": "分批進場：先放一半，賺 10% 再加滿",
        "question": ("50 日均線的進場大多會失敗（熄火），獲利靠少數幾筆大漲。如果每次進場先只放一半，"
                     "等這筆交易證明自己（例如比進場價高 10%）才加滿，和均線本身比，能不能在大多數起算點「錢更多、最深虧損也更淺」？"
                     "是「什麼時候加碼」有用，還是只因為平均放得比較少？換設定、換加碼方式還在不在？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [tA, tB, tC, tD, t1, t2, t3, t4, t5, t6, t7, t8],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
