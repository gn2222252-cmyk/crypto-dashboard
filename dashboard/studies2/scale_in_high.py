"""Historical study "scale_in_high": verification of SH, the one scale-in rule of scale_in that passed - and an
attempt to break it.

Why: scale_in tested "enter ma50's trades at half size, add the second half once the trade has proved itself". Its
pre-registered main rule S10 (add at +10% above the entry close) did not pass and was fragile (2 of 12 settings). One of
three alternative confirmations, SH = "start at 50%; go to 100% at the first new 20-day closing high after the entry",
passed all four items and Holm there: ranking 86% against ma50's 80%, median deepest loss -48.3% against -53.7%,
constant-exposure twin 78%, placebo 0%; MORE money AND a SHALLOWER deepest loss than ma50 in 74% of 108 daily starts
(69% of textbook cells, 49% for the ordinary person), failing for starts in 2022 and 2023. SH was not the main rule,
its settings were never varied, and it is one of four rules tried, so it may be the lucky one. This study tries to break
it. It REUSES scale_in's exam, costs, missed-entry model, twins, RAND placebos and per-start arrays (imported, not
re-implemented).

Owner's ground rules (every user-facing sentence): do not conclude lightly; 量到的 (measured, number, k of N) or 推測
(a guess, marked); where the data cannot separate two explanations say so; never use one tendency as a law to dismiss
another; "not seen" = not seen with this definition and data. Holy grail = earns MORE than ma50 AND loses LESS,
robustly; "more", "shallower" and "both" are always reported separately. No variant found here is promoted; a neighbour
that looks better than SH is an observation only.

Written BEFORE any run of this file. Nothing is tuned after seeing results; later additions are labelled.

Data (free, key-less, .env never read; loaders REUSED): BTC daily closes (bc.te._daily_close: blockchain.info before
Binance 2017-08, 00:00 UTC); Binance 1-hour BTCUSDT grid and prehistory (clock_robustness, read only); ETH SOL BNB XRP
(Binance) and MSTR (Yahoo) via main_signal_check._load_other; NFCI via nfci_filter._series with its fallback store
redirected to cache/_store/scale_in_high/nfci.json (seeded read-only from the shared copy). Nothing else is written.

Definitions (as scale_in).
  ma50 on/off after each close; an ENTRY is a close s with ma on at s and off at s-1; the TRADE covers closes s..end-1,
  end = first later close with ma off. Size per close of a trade; position = ma x size, decided at the close.
  HIGH(w) at close t: close(t) > every close of t-(w-1) .. t-1 (the close is the highest of the last w closes).
  SH(f, w): size f from the entry close; 100% from the first close t > s inside the trade with HIGH(w); never back.
    SH = SH(50%, 20) exactly as scale_in (the entry close itself never counts: t > s).
  CONST twin: ma x c, c = the rule's mean size on BTC 00:00 UTC days with ma on, from 2015-01-01.
  Exam = scale_in._exam / _score / _vs unchanged: BTC daily 108 monthly starts 2015-01..2023-12 (0.10% cost); 1,728
  textbook cells (72 monthly starts 2018-01..2023-12 x 24 closing hours, 0.10%); the ordinary person on the same cells
  (10 hours late, 0.25% on every change of size, each close where the wanted size rises by > 10 points missed with 10%
  probability, 200 simulations); ETH SOL BNB XRP MSTR. Ranking = lowest of the four "beats holding" shares (eligibility
  95%). Seeds: ma50 = main_signal_check's (KI ma50); every other rule half_gate's "m2" stream, as scale_in.
  "more / shallower / both" (scale_in._vs): higher yearly return / shallower deepest loss / both than ma50 on the same
  start (daily starts; textbook cells; ordinary-person runs paired simulation i with i).

PART 0 - reproduce SH from scale_in on today's data: four parts, ranking, median and worst deepest loss, CONST twin,
  RAND placebo (scale_in's seeds, 200 draws, 50 simulations), RAND-B, p, and the three "both" shares, beside scale_in's
  published numbers (run of 2026-11-18). Holm over scale_in's four rules is not re-run (it needs the other three
  rules' placebos); the bound m x p = 4p is shown. A mismatch stops the reading (said in the verdict).
PART 1 - observe, no rule: ma50's closed BTC daily trades from 2010 (at_the_signal's events; runner >= +20%, fizzle <= 0,
  small between). Per trade: is the entry close already HIGH(20); SH's add-on close t (day d = t - s, price above the
  entry close(t)/close(s) - 1) or never. Cross-table outcome x {add-on on day 1, days 2-10, day 11 or later, never}.
  Runners one by one. Per-trade arithmetic as scale_in Part 1 (two halves bought and held, 0.10% per half each way):
  given up per runner, saved per fizzle, small trades, chained product - for SH and for S10 (+10%) side by side.
PART 2 - grid fixed now: start 33/50/67% x window 10/15/20/30/40/55 (18 settings; 50%/20 = SH), full exam each with
  its own CONST twin. MEETS = ranking >= ma50's AND strictly shallower median deepest loss (textbook cells) AND ranking
  strictly above its own twin (half_gate._above). Reading: "robust" if >= 13 of 18 meet AND >= 4 of the 6 windows meet
  at 50%; "fragile" if < 9 of 18; otherwise "in between". Each setting's daily / ordinary-person "both" shares shown.
  Best and worst neighbours named as observation only.
PART 3 - placebos for SH, 200 draws each where random, every series (BTC daily, 24 clocks, other assets) treated alike;
  the real SH and every draw scored with 50 ordinary-person simulations per clock (as scale_in's placebo):
  (a) RAND-B = scale_in's: the trades SH confirmed get the add-on on a uniformly drawn close of the trade (seeds as
      scale_in, so it reproduces its RAND-B);
  (b) SAME-DELAY: trades of a series sorted by length (end - s), cut into consecutive blocks of 5 (a last block shorter
      than 2 joins the previous one); inside a block, in a random order, each trade receives the waiting time of the
      previous trade ("never" included); add-on at s + that delay if it falls inside the receiving trade, else never;
  (c) PRICE-TWIN (deterministic): add-on at the first close t > s with close(t) >= (1 + X) close(s); X per series is
      set midway between the m-th and (m+1)-th largest best-close-after-entry of its trades, m = trades SH confirmed in
      that series, so the same number of trades get the add-on;
  (d) SHIFT: the HIGH(20) flag series of each series rolled circularly by a whole number of closes drawn uniformly
      from 180 .. n - 180, then SH's confirmation applied to the shifted flag.
  Per placebo: share of draws with ranking at least SH's (business_cycle._score_ge) and share with as many or more
  daily starts "both" (vs ma50). SH BEATS a placebo (pre-registered) when both shares are <= 5% (for (c): when (c)'s
  ranking is not >= SH's AND its daily "both" count is below SH's).
PART 4 - time: scale_in._score / _vs on starts before 2019 and from 2019. By start year: daily starts (12 per year)
  and textbook cells (288 per year) more / shallower / both; a year "fails" when both < half; the failure is labelled
  "earned less" when more < half and "lost more" when shallower < half (both labels when both hold).
  Leave-one-year-out on BTC daily (textbook, 0.10%): net daily returns of SH and ma50 set to 0 in calendar year Y (both
  rules); recompute, from the 2015-01 start, yearly return SH minus ma50 and deepest losses, and the 108 daily starts'
  "both" count. Leave-one-runner-out: the five largest closed runners whose exit is on/after 2015-01-01; for each in
  turn, SH and ma50 positions set to 0 on BTC daily and the 24 clocks on the dates from its entry day to its exit day
  (other assets untouched); ranking of both (50 simulations each), median deepest loss, daily "both".
PART 5 - other assets as their own evidence: per asset its own ma50 / HIGH(20) (already so in the exam). Per asset:
  "beats holding" share SH vs ma50 vs SH's CONST twin, median deepest loss SH vs ma50, more / shallower / both over its
  starts, SH's confirmed share of trades. AGREES = beats-holding share >= ma50's AND strictly shallower median deepest
  loss. How much of SH's 86% rested on the other-assets part: ranking with and without that part, for SH and ma50.
PART 6 - ordinary-person gap, on the 1,728 cells, SH vs ma50 under the SAME setting: (i) textbook (on time, 0.10%, none
  missed); (ii) 10 hours late, 0.10%, none missed ("neither"); (iii) late, 0.25%, none missed ("costs only"); (iv) late,
  0.10%, 10% missed ("missed only"); (v) late, 0.25%, 10% missed (the ordinary person, 200 simulations, must equal
  Part 0). Robustness: late + 10% missed at 0.10% / 0.50%; late + 0.25% at 5% / 20% missed. Cost scan 0.10 .. 0.50%
  (late, 10% missed, 50 simulations; and late, none missed) and on BTC daily starts (0.10 .. 1.00%): the lowest cost at
  which the "both" share is below one half. Missed draws use the exam's seeds (ma50 KI ma50, SH m2).
PART 7 - SHG = ma x min(SH's size, H_NFCI's size) (H_NFCI = half_gate's 100% / 50% by the monthly NFCI gate). Same exam,
  CONST twin and counts. "Adds" (as scale_in Part 5) = ranking >= the better of SH and H_NFCI AND median deepest loss
  strictly shallower than both AND ranking strictly above its CONST twin.
VERDICT fixed now, four items: (1) Part 2 "robust"; (2) SH beats placebos (b), (c) and (d); (3) both time halves have
  ranking >= ma50's AND strictly shallower median deepest loss (textbook cells); (4) >= 3 of 5 other assets AGREE.
  "站得住" = all four; "部分站得住" = two or three; "站不住" = otherwise. The numbers behind each are listed.
Today: "比特幣：50 日均線｜SH" (持有 / 半倉 / 現金), the latest ma50 entry, whether and when it made a 20-day closing
  high, and the close a new 20-day closing high would need at the next close.
Limits: about 23 runners carry everything; one asset's history; SH was picked after four alternatives in one study, so
  even a pass here is in-sample - the real test is the daily shadow log; a rule that holds less nearly always loses
  less, so only twins and placebos mean anything. Not advice.
Runtime: the grid uses the full exam (200 simulations); placebos and leave-one-runner-out use 50 simulations with ma50 /
  SH re-scored the same way; the cost scan uses 50 simulations.
Added after the smoke run (2 placebo draws; no definition, threshold, bar or reading above changed):
  (b2) REPORTED ONLY, not in the verdict: SH's waiting times passed on between ALL trades of a series (one block, no
  length matching), 200 draws. The smoke run showed (b) about as good as SH; (b) knows each trade's length, which is
  future information closely tied to the outcome, so (b2) is shown beside it to see how much of (b) comes from that.
  Wording only: the cost-scan rows list the cost levels at which "both" is below one half instead of "from ... on"
  (the share does not fall monotonically with cost); the grid caption also counts settings whose daily "both" >= 50%.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import half_gate as hg
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import scale_in as si

ID = "scale_in_high"
DAY = pd.Timedelta(days=1)
FIRST = msc.DAILY_STARTS[0]
SPLIT = si.SPLIT
COST, P_COST, SIMS, P_MISS = bc.COST, bc.P_COST, bc.SIMS, bc.P_MISS
DRAWS, PL_SIMS, BAR = 200, 50, 0.05
START_F, WIN = 0.5, 20
SIZES = (0.33, 0.50, 0.67)
WINDOWS = (10, 15, 20, 30, 40, 55)
ROBUST_N, ROBUST_W, FRAGILE_N = 13, 4, 9
BLOCK, MIN_SHIFT = 5, 180
SEED_B, SEED_D = 20261121, 20261122
COSTS_P = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050)
COSTS_D = (0.0010, 0.0025, 0.0050, 0.0075, 0.0100)
N_RUNNERS_OUT = 5
SH_J = si.MAIN.index("SH")  # scale_in's seed slot for SH
# scale_in's published SH numbers (its run of 2026-11-18), for Part 0
PUB = {"sh_rank": "86%", "ma_rank": "80%", "sh_four": "100%／99%／86%／90%", "ma_four": "100%／99%／81%／80%",
       "sh_deep": "-48.3%｜-65.0%", "ma_deep": "-53.7%｜-71.0%", "twin": "78%", "twin_c": "94.0%", "plc": "0%",
       "plb": "0%", "p": "0.005", "d": "74%｜85%｜74%", "c": "69%｜94%｜69%", "p3": "53%｜88%｜49%"}

_pct, _pp, _num, _d, _ok, _share = bc._pct, bc._pp, bc._num, bc._d, bc._ok, bc._share
_f, _pts, _of, _yn, _rank, _four, _trip, _times, _med = (si._f, si._pts, si._of, si._yn, si._rank, si._four, si._trip,
                                                          si._times, si._med)
MY_STORE = common.STORE / ID


# ---------- data ----------

@contextmanager
def _nfci_store():
    """nfci_filter's fallback store redirected into this study's folder (seeded from the shared copy)."""
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


# ---------- confirmation helpers on scale_in's _Series ----------

def _high(c: np.ndarray, w: int) -> np.ndarray:
    """HIGH(w): close above every one of the previous w-1 closes (same arithmetic as scale_in's SH flag)."""
    prev = pd.Series(c).shift(1).rolling(w - 1, min_periods=w - 1).max().values
    return np.nan_to_num(c > prev, nan=False) & np.isfinite(prev)


def _next_true(flag: np.ndarray) -> np.ndarray:
    n = len(flag)
    idx = np.where(flag, np.arange(n), n)
    return np.append(np.minimum.accumulate(idx[::-1])[::-1], n)


def _conf_flag(S, flag: np.ndarray) -> np.ndarray:
    """First close t > s inside each trade with the flag on (-1 = never)."""
    if not len(S.ent):
        return np.zeros(0, dtype=int)
    nxt = _next_true(flag)
    t = nxt[np.minimum(S.ent + 1, len(flag))]
    return np.where(t < S.end, t, -1).astype(int)


def _best_after(S) -> np.ndarray:
    """Best close after the entry inside each trade, over the entry close, minus 1 (-inf if the trade has one close)."""
    out = np.full(len(S.ent), -np.inf)
    for i, (s, e) in enumerate(zip(S.ent, S.end)):
        if e > s + 1:
            out[i] = S.c[s + 1:e].max() / S.c[s] - 1
    return out


def _conf_price(S, x: float) -> np.ndarray:
    out = np.full(len(S.ent), -1, dtype=int)
    for i, (s, e) in enumerate(zip(S.ent, S.end)):
        hit = np.flatnonzero(S.c[s + 1:e] >= (1 + x) * S.c[s])
        if len(hit):
            out[i] = s + 1 + int(hit[0])
    return out


def _same_delay(S, real: np.ndarray, rng: np.random.Generator, block: int | None = BLOCK) -> np.ndarray:
    """Placebo (b): waiting times passed on to another trade of similar length. block=None: (b2, added after the
    smoke run) one block of all trades, no length matching."""
    n = len(S.ent)
    if n < 2:
        return real.copy()
    span = S.end - S.ent
    d = np.where(real >= 0, real - S.ent, -1)
    order = np.argsort(span, kind="stable")
    bl = block or n
    groups = [order[i:i + bl] for i in range(0, n, bl)]
    if len(groups) > 1 and len(groups[-1]) < 2:
        groups[-2] = np.concatenate([groups[-2], groups[-1]])
        groups.pop()
    out = np.full(n, -1, dtype=int)
    for g in groups:
        g = rng.permutation(g)
        for rcv, dn in zip(g, np.roll(g, 1)):
            dd = d[dn]
            if dd >= 1 and S.ent[rcv] + dd < S.end[rcv]:
                out[rcv] = S.ent[rcv] + dd
    return out


class _Lab:
    """Positions of SH-type rules on every series of scale_in's book."""

    def __init__(self, book: si._Book):
        self.book, self.S, self.keys = book, book.S, book.keys
        self._flag, self._conf = {}, {}

    def flag(self, key, w: int) -> np.ndarray:
        if (key, w) not in self._flag:
            self._flag[(key, w)] = _high(self.S[key].c, w)
        return self._flag[(key, w)]

    def conf(self, key, w: int = WIN) -> np.ndarray:
        if (key, w) not in self._conf:
            self._conf[(key, w)] = _conf_flag(self.S[key], self.flag(key, w))
        return self._conf[(key, w)]

    def pos_conf(self, confs: dict, f: float = START_F) -> dict:
        return {k: self.S[k].ma * self.S[k].size(f, confs[k]) for k in self.keys}

    def pos(self, f: float = START_F, w: int = WIN) -> dict:
        return self.pos_conf({k: self.conf(k, w) for k in self.keys}, f)

    def const(self, c: float) -> dict:
        return {k: self.S[k].ma * c for k in self.keys}

    def shg(self) -> dict:
        return {k: self.S[k].ma * np.minimum(self.S[k].size(START_F, self.conf(k)), 0.5 + 0.5 * self.S[k].gate)
                for k in self.keys}

    def same_delay(self, rng, block: int | None = BLOCK) -> dict:
        return self.pos_conf({k: _same_delay(self.S[k], self.conf(k), rng, block) for k in self.keys})

    def shift(self, rng) -> dict:
        confs = {}
        for k in self.keys:
            fl = self.flag(k, WIN)
            n = len(fl)
            sh = int(rng.integers(MIN_SHIFT, max(n - MIN_SHIFT, MIN_SHIFT + 1)))
            confs[k] = _conf_flag(self.S[k], np.roll(fl, sh))
        return self.pos_conf(confs)

    def price_twin(self) -> tuple[dict, dict]:
        confs, xs = {}, {}
        for k in self.keys:
            S = self.S[k]
            m = int((self.conf(k) >= 0).sum())
            g = np.sort(_best_after(S))[::-1]
            if m == 0:
                x = np.inf
            elif m >= len(g) or not np.isfinite(g[m]):
                x = float(g[m - 1]) - 1e-9
            else:
                x = float((g[m - 1] + g[m]) / 2)
            xs[k] = (x, m)
            confs[k] = _conf_price(S, x) if np.isfinite(x) else np.full(len(S.ent), -1, dtype=int)
        return self.pos_conf(confs), {"x": xs, "conf": confs}

    def zero_span(self, P: dict, ex: bc._Exam, d0: pd.Timestamp, d1: pd.Timestamp) -> dict:
        """Positions on BTC daily and the 24 clocks set to 0 on the dates d0..d1 (leave-one-runner-out)."""
        out = dict(P)
        md = (ex.close.index >= d0) & (ex.close.index <= d1)
        out["btc"] = np.where(md, 0.0, P["btc"])
        mc = (ex.m.dates >= d0) & (ex.m.dates <= d1)
        for h in range(24):
            out[("clock", h)] = np.where(mc, 0.0, P[("clock", h)])
        return out


# ---------- ordinary-person variants (Part 6) ----------

def _person(ex: bc._Exam, P: dict, ki: int, sims: int, cost: float, miss: float, late: bool = True) -> dict:
    n = len(ex.cstarts)
    pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
    for h in range(24):
        w = np.nan_to_num(P[("clock", h)], nan=0.0).astype(float)
        b, wl = rt._later(h, w, ex.days) if late else (h, w[:ex.days])
        rng = np.random.default_rng(bc.SEED + 1000 * ki + h)
        p = rt._missed(wl, miss, rng, sims)
        r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), ex.eng._mid(b, p, cost, None)]), ex.cstarts, bc.YEAR)
        pr[:, h, :], pdd[:, h, :] = r, d_
    return {"pr": pr, "pdd": pdd}


def _pvs(ex: bc._Exam, a: dict, b: dict) -> dict:
    more_, shal = a["pr"] > b["pr"] + 1e-9, a["pdd"] > b["pdd"] + 1e-9
    hr = ex.chold[0]
    return {"t": (int(more_.sum()), int(shal.sum()), int((more_ & shal).sum()), int(more_.size)),
            "beat_a": float((a["pr"] > hr[:, None, None]).mean()), "beat_b": float((b["pr"] > hr[:, None, None]).mean())}


def _daily(ex: bc._Exam, pos: np.ndarray, cost: float, zero: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    net, _ = bc.te._run(np.concatenate([[np.nan], pos[:-1]]), ex.btc_change, cost)
    if zero is not None:
        net = np.where(zero, 0.0, net)
    dr, dd = msc._suffix(msc._equity(net), ex.dstarts, bc.YEAR)
    return dr[:, 0], dd[:, 0]


def _cnt(r1, d1, r0, d0) -> tuple:
    m_, s_ = r1 > r0 + 1e-9, d1 > d0 + 1e-9
    return int(m_.sum()), int(s_.sum()), int((m_ & s_).sum()), int(m_.size)


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
    book = si._Book(ex, price, pre, F)
    lab = _Lab(book)
    KM, K2 = bc.KI["ma50"], bc.KI["m2"]
    R = {"ex": ex, "book": book, "lab": lab, "F": F, "problem": problem}

    # ----- Part 0: reproduce -----
    X = {"ma50": si._exam(ex, book.pos(si.SPECS["ma50"]), KM)}
    P_sh = lab.pos()
    P_sh_si = book.pos(si.SPECS["SH"])
    R["sh_same_as_si"] = all(np.array_equal(np.nan_to_num(P_sh[k]), np.nan_to_num(P_sh_si[k])) for k in book.keys)
    X["SH"] = si._exam(ex, P_sh, K2)
    X["H_NFCI"] = si._exam(ex, book.pos(si.SPECS["H_NFCI"]), K2)
    X["SHG"] = si._exam(ex, lab.shg(), K2)
    S = {k: si._score(ex, v) for k, v in X.items()}
    TW, TS = {}, {}
    for k in ("SH", "SHG", "H_NFCI"):
        c_ = S[k]["exp_on"]
        TW[k] = si._exam(ex, lab.const(c_), K2)
        TW[k]["c"] = c_
        TS[k] = si._score(ex, TW[k])
    V = {k: si._vs(ex, X[k], X["ma50"]) for k in ("SH", "SHG", "H_NFCI")}
    V["CONST_SH"] = si._vs(ex, TW["SH"], X["ma50"])
    R.update(X=X, S=S, TW=TW, TS=TS, V=V)
    t_main = time.time() - t0

    # small (50-simulation) versions for placebos
    Xs = {"ma50": si._exam(ex, book.pos(si.SPECS["ma50"]), KM, PL_SIMS), "SH": si._exam(ex, P_sh, K2, PL_SIMS)}
    Ss = {k: si._score(ex, v) for k, v in Xs.items()}
    sh_dboth = V["SH"]["d"][2]

    def run_draws(make, n_draws: int) -> dict:
        hr, hb, lows, deeps, dboth, cboth = 0, 0, [], [], [], []
        for i in range(n_draws):
            Xr = si._exam(ex, make(i), K2, PL_SIMS)
            sr = si._score(ex, Xr)
            vr = si._vs(ex, Xr, Xs["ma50"])
            hr += int(bc._score_ge(sr, Ss["SH"]))
            hb += int(vr["d"][2] >= sh_dboth)
            lows.append(sr["low"])
            deeps.append(sr["c_deep_med"])
            dboth.append(vr["d"][2] / vr["d"][3])
            cboth.append(vr["c"][2] / vr["c"][3])
        return {"rank": hr / n_draws, "both": hb / n_draws, "n": n_draws, "low": float(np.median(lows)),
                "deep": float(np.median(deeps)), "dboth": float(np.median(dboth)), "cboth": float(np.median(cboth)),
                "p": (hr + 1) / (n_draws + 1)}

    PL = {}
    PL["A"] = run_draws(lambda i: book.rand_pos(si.SPECS["SH"], np.random.default_rng(si.RAND_SEED + 1000 * SH_J + i)),
                        DRAWS)
    PL["a"] = run_draws(lambda i: book.rand_pos(si.SPECS["SH"],
                                                np.random.default_rng(si.RAND_SEED + 500_000 + 1000 * SH_J + i),
                                                same=True), DRAWS)
    PL["b"] = run_draws(lambda i: lab.same_delay(np.random.default_rng(SEED_B + i)), DRAWS)
    PL["d"] = run_draws(lambda i: lab.shift(np.random.default_rng(SEED_D + i)), DRAWS)
    # (b2) added after the smoke run, reported only
    PL["b2"] = run_draws(lambda i: lab.same_delay(np.random.default_rng(SEED_B + 700_000 + i), None), DRAWS)
    P_pt, pt_info = lab.price_twin()
    Xpt = si._exam(ex, P_pt, K2, PL_SIMS)
    spt, vpt = si._score(ex, Xpt), si._vs(ex, Xpt, Xs["ma50"])
    PL["c"] = {"rank": float(bc._score_ge(spt, Ss["SH"])), "both": float(vpt["d"][2] >= sh_dboth), "n": 1,
               "low": spt["low"], "deep": spt["c_deep_med"], "dboth": vpt["d"][2] / vpt["d"][3],
               "cboth": vpt["c"][2] / vpt["c"][3], "info": pt_info, "s": spt, "v": vpt}
    R.update(Xs=Xs, Ss=Ss, PL=PL)
    t_plc = time.time() - t0 - t_main

    # ----- Part 2: grid -----
    grid = []
    for f in SIZES:
        for w in WINDOWS:
            if f == START_F and w == WIN:
                Xg, sg, tg = X["SH"], S["SH"], TS["SH"]
            else:
                Xg = si._exam(ex, lab.pos(f, w), K2)
                sg = si._score(ex, Xg)
                tg = si._score(ex, si._exam(ex, lab.const(sg["exp_on"]), K2))
            vg = si._vs(ex, Xg, X["ma50"])
            q = {"score": bool(bc._score_ge(sg, S["ma50"])), "loss": sg["c_deep_med"] > S["ma50"]["c_deep_med"] + 1e-12,
                 "twin": bool(hg._above(sg, tg))}
            grid.append({"f": f, "w": w, "s": sg, "tw": tg, "v": vg, "q": q, "meets": all(q.values())})
    R["grid"] = grid
    t_grid = time.time() - t0 - t_main - t_plc

    # ----- Part 4: leave-one-runner-out -----
    obs = si._observe(close)
    R["obs"] = obs
    runners = sorted([t for t in obs["T"] if t["kind"] == "runner" and close.index[t["x"]] >= pd.Timestamp(FIRST)],
                     key=lambda t: -t["res"])[:N_RUNNERS_OUT]
    lro = []
    P_ma = book.pos(si.SPECS["ma50"])
    for t in runners:
        d0, d1 = close.index[t["s"]], close.index[t["x"]]
        xa = si._exam(ex, lab.zero_span(P_sh, ex, d0, d1), K2, PL_SIMS)
        xb = si._exam(ex, lab.zero_span(P_ma, ex, d0, d1), KM, PL_SIMS)
        lro.append({"t": t, "sa": si._score(ex, xa), "sb": si._score(ex, xb), "v": si._vs(ex, xa, xb)})
    R["lro"] = lro

    # ----- Part 6: ordinary-person decomposition -----
    P_ma_ = P_ma
    var = {}
    for key, (cost, miss, late, sims) in {
            "neither": (COST, 0.0, True, 1), "cost": (P_COST, 0.0, True, 1), "miss": (COST, P_MISS, True, SIMS),
            "both": (P_COST, P_MISS, True, SIMS), "c010": (0.0010, P_MISS, True, SIMS), "c050": (0.0050, P_MISS, True, SIMS),
            "m05": (P_COST, 0.05, True, SIMS), "m20": (P_COST, 0.20, True, SIMS)}.items():
        a = _person(ex, P_sh, K2, sims, cost, miss, late)
        b = _person(ex, P_ma_, KM, sims, cost, miss, late)
        var[key] = _pvs(ex, a, b)
    scan_p, scan_p0 = {}, {}
    for cost in COSTS_P:
        a = _person(ex, P_sh, K2, PL_SIMS, cost, P_MISS)
        b = _person(ex, P_ma_, KM, PL_SIMS, cost, P_MISS)
        scan_p[cost] = _pvs(ex, a, b)
        a = _person(ex, P_sh, K2, 1, cost, 0.0)
        b = _person(ex, P_ma_, KM, 1, cost, 0.0)
        scan_p0[cost] = _pvs(ex, a, b)
    scan_d = {}
    for cost in COSTS_D:
        r1, d1 = _daily(ex, P_sh["btc"], cost)
        r0, d0 = _daily(ex, P_ma_["btc"], cost)
        scan_d[cost] = _cnt(r1, d1, r0, d0)
    R.update(var=var, scan_p=scan_p, scan_p0=scan_p0, scan_d=scan_d)

    # ----- Part 4: leave-one-year-out (BTC daily, textbook) -----
    yrs = ex.close.index.year
    loyo = {}
    for y in [None] + sorted(set(int(v) for v in yrs[ex.close.index >= pd.Timestamp(FIRST)])):
        z = (yrs == y) if y is not None else None
        r1, d1 = _daily(ex, P_sh["btc"], COST, z)
        r0, d0 = _daily(ex, P_ma_["btc"], COST, z)
        loyo[y] = {"diff": float(r1[0] - r0[0]), "d1": float(d1[0]), "d0": float(d0[0]), "cnt": _cnt(r1, d1, r0, d0)}
    R["loyo"] = loyo
    R["years"] = {"SH": bc._years(P_sh["btc"], close, FIRST), "ma50": bc._years(P_ma_["btc"], close, FIRST)}
    R["t"] = (t_main, t_plc, t_grid, time.time() - t0)
    return R


# ---------- formatting ----------

def _amount(v) -> str:
    if not _ok(v) or v < 0.25:
        return "現金"
    return "持有" if v >= 0.75 else "半倉"


def _sc(t: dict, c: np.ndarray, tt: int | None, f: float = START_F) -> float:
    """scale_in Part 1 arithmetic with a given add-on close (None = never)."""
    s, x = t["s"], t["x"]
    g = f * (c[x] / c[s] - 1) - 2 * COST * f
    if tt is not None and tt >= 0:
        g += (1 - f) * (c[x] / c[tt] - 1) - 2 * COST * (1 - f)
    return float(g)


def _pc(v: tuple) -> str:
    m_, sh, b, n = v
    return f"{m_}｜{sh}｜{b}（共 {n}）" if n else "—"


def _half(v: tuple) -> bool:
    return v[3] > 0 and v[2] >= v[3] / 2


def study() -> dict:
    t0 = time.time()
    data = _load()
    R = _compute(data)
    close, ex, S, TS, X, V, TW, PL = (data["close"], R["ex"], R["S"], R["TS"], R["X"], R["V"], R["TW"], R["PL"])
    base, sh = S["ma50"], S["SH"]
    lab = R["lab"]
    problems = data["problems"] + ([R["problem"]] if R["problem"] else [])
    n_d, n_c = base["d_n"], base["c_n"]
    tables = []

    # ----- Part 0 -----
    A_, a_ = PL["A"], PL["a"]
    mine = {"sh_rank": _share(sh["low"]), "ma_rank": _share(base["low"]), "sh_four": _four(sh), "ma_four": _four(base),
            "sh_deep": f"{_pct(sh['c_deep_med'])}｜{_pct(sh['c_deep_worst'])}",
            "ma_deep": f"{_pct(base['c_deep_med'])}｜{_pct(base['c_deep_worst'])}", "twin": _share(TS["SH"]["low"]),
            "twin_c": f"{TW['SH']['c'] * 100:.1f}%", "plc": _share(A_["rank"]), "plb": _share(a_["rank"]),
            "p": f"{A_['p']:.3f}", "d": _trip(V["SH"]["d"]), "c": _trip(V["SH"]["c"]), "p3": _trip(V["SH"]["p"])}
    labels0 = {"sh_rank": "SH 排名分數", "ma_rank": "50 日均線排名分數", "sh_four": "SH 四項：日線／課本／一般人／其他資產",
               "ma_four": "均線四項", "sh_deep": "SH 最深虧損：中位數｜最差一格", "ma_deep": "均線最深虧損：中位數｜最差一格",
               "twin": "SH 的固定比例雙胞胎排名", "twin_c": "雙胞胎的固定比例", "plc": "隨機錯位（亂挑交易、亂挑日子）一樣好的比例",
               "plb": "同一批交易、加滿日子亂抽（RAND-B）一樣好的比例", "p": "p＝（一樣好次數＋1）÷201",
               "d": "日線起算月：多｜淺｜都", "c": "課本格：多｜淺｜都", "p3": "一般人：多｜淺｜都"}
    diff0 = [k for k in PUB if PUB[k] != mine[k]]
    reproduced = not diff0
    rows0 = [[labels0[k], PUB[k], mine[k], "一樣" if PUB[k] == mine[k] else "不一樣"] for k in PUB]
    rows0.append(["Holm 校正（四條規則一起）", "0.020", f"不重跑；上限 4 × p ＝ {min(1.0, 4 * A_['p']):.3f}", "—"])
    tables.append({"caption": ("第零部分：先重現 scale_in 量到的 SH 數字（用一樣的程式、一樣的種子，資料多了最近幾天）。"
                               "SH＝均線進場時先放 50%，進場後第一次出現「20 日收盤新高」（收盤價是最近 20 天裡最高的）就加到 100%，均線出場時全部賣掉。"
                               "多｜淺｜都＝和 50 日均線同一個起算點比：最後錢比較多｜最深虧損比較淺｜兩個都做到的比例。"
                               f"這次{'全部重現' if reproduced else '有 ' + str(len(diff0)) + ' 項不一樣（見最後一欄）'}；"
                               f"另外確認本研究自己算的 SH 持有比例和 scale_in 的 SH {'逐日完全一樣' if R['sh_same_as_si'] else '不一樣（要查）'}。"),
                   "columns": ["項目", "scale_in 公布（2026-11-18）", "這次重算", "比對"], "rows": rows0})

    # ----- Part 1 -----
    ob = R["obs"]
    T, c = ob["T"], ob["c"]
    sb = R["book"].S["btc"]
    conf_sh = lab.conf("btc")
    flag = lab.flag("btc", WIN)
    pos_of = {int(s): i for i, s in enumerate(sb.ent)}
    for t in T:
        i = pos_of.get(int(t["s"]))
        tt = int(conf_sh[i]) if i is not None else -1
        t["sh_t"] = tt if tt >= 0 else None
        t["sh_d"] = (tt - t["s"]) if tt >= 0 else None
        t["sh_above"] = float(c[tt] / c[t["s"]] - 1) if tt >= 0 else None
        t["ent_high"] = bool(flag[t["s"]])
        t["s10_t"] = t["reach"].get(0.10)
    N = len(T)
    groups = (("跑起來（≥ +20%）", "runner"), ("小賺（0 到 +20%）", "small"), ("熄火（≤ 0）", "fizzle"), ("全部", None))

    def bucket(t) -> int:
        d = t["sh_d"]
        return 3 if d is None else (0 if d == 1 else (1 if d <= 10 else 2))

    rows1 = []
    cnt = {}
    for name, kind in groups:
        g = [t for t in T if kind is None or t["kind"] == kind]
        b = [sum(bucket(t) == j for t in g) for j in range(4)]
        cnt[kind] = (len(g), sum(t["ent_high"] for t in g), b)
        rows1.append([name, f"{len(g)}", _of(sum(t["ent_high"] for t in g), len(g)), _of(b[0], len(g)), _of(b[1], len(g)),
                      _of(b[2], len(g)), _of(b[3], len(g)),
                      f"{_med([t['sh_d'] for t in g if t['sh_d'] is not None]):.0f} 天" if any(t["sh_d"] for t in g) else "—",
                      _f(_med([t["sh_above"] for t in g if t["sh_above"] is not None]))])
    nr, ns, nf_ = cnt["runner"][0], cnt["small"][0], cnt["fizzle"][0]
    ent_hi_all = cnt[None][1]
    d1_after_hi = sum(1 for t in T if t["ent_high"] and t["sh_d"] == 1)
    tables.append({"caption": (f"第一部分（先觀察，沒有規則）：比特幣日線 {_d(T[0]['day'])} 起，50 日均線做完的 {N} 筆交易。"
                               "「20 日新高」＝收盤價是最近 20 天裡最高的。照 scale_in 的定義，進場那天本身不算（要進場之後的收盤），"
                               "所以就算進場那天已經是 20 日新高，SH 也至少有一天只放一半。「第 1 天加滿」＝進場隔天的收盤就創新高，從第 2 天起滿倉。"
                               f"量到的：{N} 筆裡 {ent_hi_all} 筆進場那天收盤已經是 20 日新高；這 {ent_hi_all} 筆裡 {d1_after_hi} 筆隔天又創新高、第 1 天就加滿。"),
                   "columns": ["交易結果", "筆數", "進場那天已是 20 日新高", "第 1 天加滿", "第 2–10 天加滿", "第 11 天以後加滿",
                               "從沒加滿（整筆半倉）", "加滿天數（中位數）", "加滿那天比進場價高（中位數）"],
                   "rows": rows1})

    run = [t for t in T if t["kind"] == "runner"]
    fiz = [t for t in T if t["kind"] == "fizzle"]
    sml = [t for t in T if t["kind"] == "small"]
    full_chain = float(np.prod([1 + t["res"] for t in T]))
    cmp_ = {}
    rows2 = []
    for name, key in (("SH：20 日新高加滿", "sh_t"), ("S10：比進場價高 10% 加滿", "s10_t")):
        jr = [t for t in run if t[key] is not None]
        fc = [t for t in fiz if t[key] is not None]
        give = float(np.mean([t["res"] - _sc(t, c, t[key]) for t in run]))
        save = float(np.mean([_sc(t, c, t[key]) - t["res"] for t in fiz]))
        save_fc = float(np.mean([_sc(t, c, t[key]) - t["res"] for t in fc])) if fc else None
        save_nc = float(np.mean([_sc(t, c, t[key]) - t["res"] for t in fiz if t[key] is None])) if len(fc) < len(fiz) else None
        smd = float(np.mean([_sc(t, c, t[key]) - t["res"] for t in sml]))
        chain = float(np.prod([1 + _sc(t, c, t[key]) for t in T]))
        above = float(np.mean([c[t[key]] / c[t["s"]] - 1 for t in jr])) if jr else None
        days_ = _med([t[key] - t["s"] for t in jr])
        cmp_[key] = {"jr": len(jr), "fc": len(fc), "give": give, "save": save, "chain": chain, "above": above,
                     "days": days_, "save_fc": save_fc, "save_nc": save_nc, "small": smd}
        rows2.append([name, f"{_of(len(jr), nr)}（{days_:.0f} 天、高 {_f(above)}）", _pts(-give),
                      f"{_of(len(fc), nf_)} 加滿後還是賠｜{nf_ - len(fc)} 筆整筆半倉", _pts(save),
                      f"{_pts(save_fc)}｜{_pts(save_nc)}", _pts(smd), f"{_times(full_chain)}｜{_times(chain)}"])
    tables.append({"caption": ("第一部分：SH 和 S10 放在同一張表比（描述，不是考試）。假設先買一半、確認那天收盤再買另一半，兩份都抱到均線出場"
                               "（每份買賣各付 0.10%）。「跑起來的加滿」＝23 筆左右的大漲交易裡有幾筆加得到，括號是加滿天數中位數、加滿價平均比進場價高多少；"
                               "「每筆少賺」＝跑起來的交易平均比滿倉少幾點（整個部位的百分點）；熄火的交易裡「加滿後還是賠」＝假確認；"
                               "「每筆少虧」＝熄火的交易平均比滿倉少虧幾點，後一格把熄火分成「加滿過的｜一直半倉的」各自平均；"
                               f"最後一欄是 {N} 筆交易接起來 滿倉｜這個做法（只算持有期間）。"),
                   "columns": ["確認方式", "跑起來的加滿（天數、加滿價）", "跑起來每筆少賺", "熄火的", "熄火每筆少虧",
                               "熄火：加滿過的｜一直半倉的", "小賺每筆差", "接起來：滿倉｜這個做法"],
                   "rows": rows2})

    rows3 = []
    for t in sorted(run, key=lambda t: t["day"]):
        rows3.append([_d(t["day"]), f"{t['len']}", _f(t["res"]), "是" if t["ent_high"] else "否",
                      f"第 {t['sh_d']} 天、{_f(t['sh_above'])}" if t["sh_d"] is not None else "沒加滿",
                      _pts(_sc(t, c, t["sh_t"]) - t["res"]),
                      f"第 {t['s10_t'] - t['s']} 天" if t["s10_t"] is not None else "沒加滿",
                      _pts(_sc(t, c, t["s10_t"]) - t["res"])])
    tables.append({"caption": (f"第一部分：{nr} 筆跑起來的交易逐筆列出。「SH 加滿」＝進場後第幾天收盤第一次創 20 日新高、那天比進場價高多少；"
                               "「比滿倉」＝先一半、確認後加滿、抱到出場，和一開始就滿倉差幾點（負數＝少賺）。"),
                   "columns": ["進場日", "持有天數", "滿倉結果", "進場那天已是 20 日新高", "SH 加滿", "SH 比滿倉", "S10（+10%）加滿",
                               "S10 比滿倉"],
                   "rows": rows3})

    # ----- Part 2 -----
    grid = R["grid"]
    gcount = sum(g["meets"] for g in grid)
    w50 = sum(g["meets"] for g in grid if g["f"] == START_F)
    robust = gcount >= ROBUST_N and w50 >= ROBUST_W
    fragile = gcount < FRAGILE_N
    g_word = "穩健" if robust else ("脆弱" if fragile else "介於中間")
    rows4 = []
    for g in grid:
        x, t_ = g["s"], g["tw"]
        rows4.append([f"{g['f'] * 100:.0f}%｜{g['w']} 天" + ("（SH）" if g["f"] == START_F and g["w"] == WIN else ""),
                      _rank(x), _pct(x["c_deep_med"]), f"{_rank(t_)}｜{_pct(t_['c_deep_med'])}",
                      "／".join(_yn(g["q"][k]) for k in ("score", "loss", "twin")), "過" if g["meets"] else "不過",
                      _trip(g["v"]["d"]), _trip(g["v"]["p"])])
    others = [g for g in grid if not (g["f"] == START_F and g["w"] == WIN)]
    key_ = lambda g: (g["s"]["low"] if g["s"]["eligible"] else -1, g["s"]["c_deep_med"])  # noqa: E731
    best, worst = max(others, key=key_), min(others, key=key_)
    gboth = sum(_half(g["v"]["d"]) for g in grid)
    gboth_p = sum(_half(g["v"]["p"]) for g in grid)

    def _nb(g) -> str:
        return (f"起手 {g['f'] * 100:.0f}%、{g['w']} 天（排名 {_rank(g['s'])}、虧損 {_pct(g['s']['c_deep_med'])}、"
                f"日線兩個都做到 {_share(g['v']['d'][2] / g['v']['d'][3])}）")
    tables.append({"caption": (f"第二部分（是不是剛好選中一個幸運的設定）：起手 33%／50%／67% × 新高天數 10／15／20／30／40／55，18 種，全部用完整考試。"
                               f"三項＝排名不低於均線（{_rank(base)}）／最深虧損中位數比均線（{_pct(base['c_deep_med'])}）淺／排名比自己的固定比例雙胞胎高。"
                               f"事先寫好：至少 {ROBUST_N} 種三項都過、而且起手 50% 的 6 種裡至少 {ROBUST_W} 種過＝穩健；少於 {FRAGILE_N} 種＝脆弱。"
                               f"這次 {gcount}/18、起手 50% 有 {w50}/6 → {g_word}。最後兩欄是和均線比的 多｜淺｜都（日線起算月、一般人做法）；"
                               f"（試跑後加的計數，不算進判斷）日線起算月「兩個都做到」過半的設定 {gboth}/18，一般人做法過半的 {gboth_p}/18。"
                               f"只當觀察、不推薦：排名和虧損看起來最好的鄰居是 {_nb(best)}，最差的是 {_nb(worst)}。"),
                   "columns": ["起手｜新高天數", "排名分數", "最深虧損中位數", "雙胞胎：排名｜虧損", "三項（排名／虧損／雙胞胎）", "三項都過",
                               "日線：多｜淺｜都", "一般人：多｜淺｜都"],
                   "rows": rows4})

    # ----- Part 3 -----
    sh_small = R["Ss"]["SH"]
    beat = {}
    rows5 = []
    pt = PL["c"]
    xb = pt["info"]["x"]["btc"]
    names = {"A": "scale_in 原本的隨機錯位（亂挑同樣多筆交易、亂挑日子）", "a": "(a) RAND-B：同一批交易、加滿日子亂抽",
             "b": "(b) 同樣等待天數、換到長度相近的另一筆交易", "c": f"(c) 價格雙胞胎：比進場價高 {xb[0] * 100:.1f}% 就加滿（比特幣日線）",
             "d": "(d) 新高旗標整串錯開 180 天以上",
             "b2": "(b2) 同樣等待天數、換到任何一筆交易（不看長度；試跑後加，只報告）"}
    for k in ("A", "a", "b", "b2", "c", "d"):
        p = PL[k]
        if k == "c":
            beat[k] = (not p["rank"]) and (not p["both"])
            hit = f"排名 {_share(p['low'])}{'（不低於 SH）' if p['rank'] else '（低於 SH）'}"
            both_ = f"日線兩個都做到 {_share(p['dboth'])}{'（不少於 SH）' if p['both'] else '（少於 SH）'}"
        else:
            beat[k] = p["rank"] <= BAR + 1e-12 and p["both"] <= BAR + 1e-12
            hit, both_ = _share(p["rank"]), _share(p["both"])
        rows5.append([names[k], hit, both_, f"{_share(p['low'])}｜{_pct(p['deep'])}", f"{_share(p['dboth'])}｜{_share(p['cboth'])}",
                      "SH 贏" if beat[k] else "SH 沒贏過"])
    rows5.append(["SH 本身（同樣用每格 50 次模擬重算）", "—", "—", f"{_share(sh_small['low'])}｜{_pct(sh_small['c_deep_med'])}",
                  f"{_share(V['SH']['d'][2] / V['SH']['d'][3])}｜{_share(V['SH']['c'][2] / V['SH']['c'][3])}", "—"])
    pb = beat["b"] and beat["c"] and beat["d"]
    m_btc = xb[1]
    ov = int(((pt["info"]["conf"]["btc"] >= 0) & (lab.conf("btc") >= 0)).sum())
    tables.append({"caption": ("第三部分（更難的安慰劑）：scale_in 自己的安慰劑後來發現太容易贏，這裡換四種更難的對照，每種隨機的抽 200 次，"
                               "每一次都和 SH 用同樣方式考（一般人做法每格 50 次模擬）。第二欄＝排名分數不低於 SH 的抽樣比例；第三欄＝日線起算月"
                               "「錢更多又虧更淺」的個數不少於 SH 的抽樣比例。事先寫好：兩個比例都 ≤ 5% 才算 SH 贏；(c) 只有一種，"
                               "排名比 SH 低而且兩個都做到的比 SH 少才算 SH 贏。"
                               f"(c) 的門檻是每個資產、每個收盤時間各自算，讓加滿的交易筆數和 SH 一樣；比特幣日線 SH 加滿 {m_btc} 筆，"
                               f"價格雙胞胎也加滿 {int((pt['info']['conf']['btc'] >= 0).sum())} 筆，其中 {ov} 筆是同一批。"
                               "最後兩欄是抽樣的中位數。第一列是 scale_in 原本的對照，只為重現，不算進判斷。"
                               "(b2) 是試跑之後才加的、只報告：試跑時 (b) 和 SH 差不多好，而 (b) 照交易長度配對，"
                               "交易長度是事後才知道、又和結果很有關的資訊（長的交易多半是大漲的），所以另外放一個不看長度的版本對照。"),
                   "columns": ["對照", "排名一樣好或更好", "兩個都做到一樣多或更多", "排名｜虧損中位數", "兩個都做到：日線｜課本格", "結果"],
                   "rows": rows5})

    # ----- Part 4 -----
    parts = {"pre": "2019 年以前起算", "post": "2019 年起起算"}
    SP = {k: {p: si._score(ex, X[k], p) for p in parts} for k in ("ma50", "SH")}
    TSP = {p: si._score(ex, TW["SH"], p) for p in parts}
    VP = {p: si._vs(ex, X["SH"], X["ma50"], p) for p in parts}
    half_ok = {}
    rows6 = []
    for p, labp in parts.items():
        x, b_ = SP["SH"][p], SP["ma50"][p]
        q1 = bool(bc._score_ge(x, b_))
        q2 = bool(x["c_deep_med"] > b_["c_deep_med"] + 1e-12)
        half_ok[p] = q1 and q2
        rows6.append([labp, f"{_rank(x)}（{_four(x)}）", f"{_rank(b_)}（{_four(b_)}）", _rank(TSP[p]),
                      f"{_pct(x['c_deep_med'])} 對 {_pct(b_['c_deep_med'])}", _trip(VP[p]["d"]), _trip(VP[p]["c"]),
                      _trip(VP[p]["p"]), "過" if half_ok[p] else "不過"])
    mpre = si._masks(ex, "pre")
    tables.append({"caption": (f"第四部分：時間切開。2019 年以前：日線 {int(mpre['d'].sum())} 個起算月（2015–2018）、課本格只有 2018 年的 "
                               f"{int(mpre['c'].sum()) * 24} 格（小時線 2017-08 才有）；2019 年起：日線 {n_d - int(mpre['d'].sum())} 個。"
                               "排名括號是四項：日線／課本／一般人／其他資產。事先寫好：兩段都要排名不低於均線、而且課本格虧損中位數比均線淺。"),
                   "columns": ["起算點", "SH 排名", "均線排名", "雙胞胎排名", "虧損中位數 SH 對 均線", "日線：多｜淺｜都",
                               "課本格：多｜淺｜都", "一般人：多｜淺｜都", "結果"],
                   "rows": rows6})

    A_sh, A_ma = X["SH"]["arr"], X["ma50"]["arr"]
    dy = np.array([d.year for d in ex.dlabels])
    cy = np.array([d.year for d in ex.clabels])
    rows7, fail_years = [], []
    for y in sorted(set(dy.tolist())):
        md = dy == y
        vd = _cnt(A_sh["dr"][md], A_sh["dd"][md], A_ma["dr"][md], A_ma["dd"][md])
        mc = cy == y
        vc = _cnt(A_sh["tr"][mc], A_sh["td"][mc], A_ma["tr"][mc], A_ma["td"][mc]) if mc.any() else (0, 0, 0, 0)
        why = []
        if vd[2] < vd[3] / 2:
            if vd[0] < vd[3] / 2:
                why.append("賺得比較少")
            if vd[1] < vd[3] / 2:
                why.append("虧得比較深")
            fail_years.append((y, "、".join(why) or "多和淺各自過半但不是同一批"))
        rows7.append([f"{y}", _pc(vd), _pc(vc) if vc[3] else "—（沒有小時線）",
                      ("兩個都做到不到一半：" + ("、".join(why) or "多和淺各自過半但不是同一批")) if vd[2] < vd[3] / 2 else "過半"])
    tables.append({"caption": ("第四部分：依起算年份。每格是 SH 對均線：錢更多｜虧損更淺｜兩個都做到（共幾個）。"
                               "日線每年 12 個起算月；課本格每年 12 個月 × 24 個收盤時間。最後一欄說日線那一年失敗是因為賺得少、還是虧得深。"),
                   "columns": ["起算年", "日線：多｜淺｜都", "課本格：多｜淺｜都", "日線讀法"], "rows": rows7})

    loyo = R["loyo"]
    rows8 = []
    for y, v in loyo.items():
        rows8.append(["不拿掉（原本）" if y is None else f"拿掉 {y} 年", _pp(v["diff"]), f"{_pct(v['d1'])}｜{_pct(v['d0'])}",
                      _pc(v["cnt"])])
    flips = [y for y, v in loyo.items() if y is not None and (v["diff"] <= 0) != (loyo[None]["diff"] <= 0)]
    bothflip = [y for y, v in loyo.items() if y is not None and _half(v["cnt"]) != _half(loyo[None]["cnt"])]
    tables.append({"caption": ("第四部分：一年一年拿掉（比特幣日線照課本做，成本 0.10%）。拿掉＝那一年 SH 和均線每天的報酬都當作 0，其他年照算。"
                               "第二欄是從 2015-01 起算的每年報酬 SH 減均線（百分點）；第三欄是同一起算點的最深虧損 SH｜均線；"
                               f"最後一欄是 {n_d} 個日線起算月 多｜淺｜都。量到的：拿掉後「SH 每年賺得比均線多或少」翻轉的年份："
                               f"{'、'.join(str(y) for y in flips) if flips else '沒有'}；「兩個都做到」是否過半翻轉的年份："
                               f"{'、'.join(str(y) for y in bothflip) if bothflip else '沒有'}。"),
                   "columns": ["拿掉哪一年", "每年報酬 SH 減均線（2015 起）", "最深虧損 SH｜均線（2015 起）", "日線起算月：多｜淺｜都"],
                   "rows": rows8})

    lro = R["lro"]
    sma, ssh = R["Ss"]["ma50"], R["Ss"]["SH"]
    rows9 = [["不拿掉（每格 50 次模擬）", f"{_rank(ssh)} 對 {_rank(sma)}", f"{(ssh['low'] - sma['low']) * 100:+.0f} 點",
              f"{_pct(ssh['c_deep_med'])} 對 {_pct(sma['c_deep_med'])}", _pc(V["SH"]["d"])]]
    lro_keep = 0
    for L in lro:
        t, a, b_ = L["t"], L["sa"], L["sb"]
        ok = bool(bc._score_ge(a, b_)) and a["low"] > b_["low"] + 1e-12
        lro_keep += int(ok)
        rows9.append([f"拿掉 {_d(t['day'])} 進場（{_f(t['res'])}，{t['len']} 天）", f"{_rank(a)} 對 {_rank(b_)}",
                      f"{(a['low'] - b_['low']) * 100:+.0f} 點", f"{_pct(a['c_deep_med'])} 對 {_pct(b_['c_deep_med'])}",
                      _pc(L["v"]["d"])])
    tables.append({"caption": (f"第四部分：一筆一筆拿掉最大的 {N_RUNNERS_OUT} 筆跑起來的交易（只挑 2015 年以後才出場、會影響考試的）。"
                               "拿掉＝那筆交易從進場日到出場日，SH 和均線在比特幣日線和 24 個收盤時間都不持有（一直抱著的對照不變、其他資產不變）。"
                               f"量到的：{lro_keep}/{len(lro)} 次拿掉後 SH 的排名仍然比均線高。"),
                   "columns": ["情況", "排名 SH 對 均線", "排名差", "課本格虧損中位數 SH 對 均線", "日線：多｜淺｜都"],
                   "rows": rows9})

    # ----- Part 5 -----
    rows10, agree = [], 0
    for a, o in ex.other.items():
        hold = o["hold"]
        r1, d1 = A_sh["o"][a]
        r0, d0 = A_ma["o"][a]
        rt_, _dt = TW["SH"]["arr"]["o"][a]
        b1, b0_ = float((r1 > hold[0]).mean()), float((r0 > hold[0]).mean())
        bt = float((rt_ > hold[0]).mean())
        m1, m0 = float(np.median(d1)), float(np.median(d0))
        ok = b1 >= b0_ - 1e-12 and m1 > m0 + 1e-12
        agree += int(ok)
        Sa = R["book"].S[("o", a)]
        cf = lab.conf(("o", a))
        rows10.append([a, f"{len(r1)}", f"{_share(b1)}｜{_share(b0_)}｜{_share(bt)}", f"{_pct(m1)}｜{_pct(m0)}",
                       _pc(_cnt(r1, d1, r0, d0)), _of(int((cf >= 0).sum()), len(Sa.ent)), "一致" if ok else "不一致"])
    nobase = {k: min(S[k]["checks"][:3]) for k in ("SH", "ma50")}
    tables.append({"caption": ("第五部分：其他資產各自當證據。每個資產用自己的價格、自己的 50 日均線、自己的 20 日新高，和自己的均線比。"
                               "贏抱著＝從起算月到三年前，報酬贏過同一天起一直抱著的比例：SH｜均線｜SH 的固定比例雙胞胎。"
                               "事先寫好：贏抱著的比例不低於均線、而且最深虧損中位數比均線淺＝一致。"
                               f"量到的：5 個裡 {agree} 個一致。這一項在 SH 的排名分數裡是「其他資產 {_share(sh['checks'][3])}」"
                               f"（均線 {_share(base['checks'][3])}）；排名分數取四項最低的，SH 最低的是一般人那項 {_share(sh['checks'][2])}，"
                               f"把其他資產那項拿掉，SH 排名仍是 {_share(nobase['SH'])}、均線變 {_share(nobase['ma50'])}。"),
                   "columns": ["資產", "起算月", "贏抱著：SH｜均線｜雙胞胎", "最深虧損中位數 SH｜均線", "SH 對均線 多｜淺｜都",
                               "SH 加滿的交易", "結果"],
                   "rows": rows10})

    # ----- Part 6 -----
    var, scan_p, scan_p0, scan_d = R["var"], R["scan_p"], R["scan_p0"], R["scan_d"]
    vlab = (("textbook", "照課本：準時、0.10%、不漏"), ("neither", "晚 10 小時、0.10%、不漏（兩個都沒有）"),
            ("cost", "晚 10 小時、0.25%、不漏（只有成本）"), ("miss", "晚 10 小時、0.10%、一成漏掉（只有漏掉）"),
            ("both", "晚 10 小時、0.25%、一成漏掉（一般人）"), ("c010", "穩健：成本 0.10%、一成漏掉（和「只有漏掉」同一個設定）"),
            ("c050", "穩健：成本 0.50%、一成漏掉"), ("m05", "穩健：0.25%、5% 漏掉"), ("m20", "穩健：0.25%、20% 漏掉"))
    rows11 = []
    tb = V["SH"]["c"]
    for k, lb in vlab:
        if k == "textbook":
            rows11.append([lb, _trip(tb), f"{_share(sh['checks'][1])}｜{_share(base['checks'][1])}"])
            continue
        v = var[k]
        rows11.append([lb, _trip(v["t"]), f"{_share(v['beat_a'])}｜{_share(v['beat_b'])}"])

    def first_below(sc: dict) -> str:
        low = []
        for cost, v in sc.items():
            t = v["t"] if isinstance(v, dict) else v
            if t[2] < t[3] / 2:
                low.append(f"{cost * 100:.2f}%")
        return ("低於一半的成本：" + "、".join(low)) if low else f"試過的成本（到 {max(sc) * 100:.2f}%）都沒有低於一半"

    scan_txt = "；".join(f"{k * 100:.2f}% → {_share(v['t'][2] / v['t'][3])}" for k, v in scan_p.items())
    scan0_txt = "；".join(f"{k * 100:.2f}% → {_share(v['t'][2] / v['t'][3])}" for k, v in scan_p0.items())
    scand_txt = "；".join(f"{k * 100:.2f}% → {_share(v[2] / v[3])}" for k, v in scan_d.items())
    rows11.append(["成本掃描：晚 10 小時、一成漏掉（每格 50 次）", scan_txt, first_below(scan_p)])
    rows11.append(["成本掃描：晚 10 小時、不漏", scan0_txt, first_below(scan_p0)])
    rows11.append(["成本掃描：比特幣日線起算月（準時、不漏）", scand_txt, first_below(scan_d)])
    tables.append({"caption": (f"第六部分：為什麼一般人做法「兩個都做到」掉到一半以下。同樣 {n_c:,} 格，SH 和均線每次都用同一種做法比。"
                               f"一般人做法有三個差別：收盤後 10 小時才做、每次改變持有比例付 0.25%（照課本 0.10%）、每次要買進的那天有一成機率漏掉"
                               "（SH 的加碼那一筆也算；漏掉加碼，那筆交易剩下的時間就只有一半）。一次只加一個差別來拆。"
                               f"SH 每年改變持有比例 {sh['sw_y']:.1f} 次、均線 {base['sw_y']:.1f} 次。"
                               "第二欄是 SH 對均線 多｜淺｜都；第三欄是報酬贏過一直抱著的比例 SH｜均線。"),
                   "columns": ["做法", "多｜淺｜都 或 兩個都做到的比例", "贏抱著 SH｜均線 或 讀法"], "rows": rows11})

    # ----- Part 7 -----
    sg, hn, tsg = S["SHG"], S["H_NFCI"], TS["SHG"]
    better = sh if bc._score_ge(sh, hn) else hn
    p7a = bool(bc._score_ge(sg, better))
    p7b = sg["c_deep_med"] > max(sh["c_deep_med"], hn["c_deep_med"]) + 1e-12
    p7c = bool(hg._above(sg, tsg))
    adds = p7a and p7b and p7c
    rows12 = [["(a) 排名不低於兩者較高的", f"SHG {_rank(sg)}；SH {_rank(sh)}；H_NFCI {_rank(hn)}", _yn(p7a)],
              ["(b) 最深虧損中位數比兩者都淺", f"SHG {_pct(sg['c_deep_med'])}；SH {_pct(sh['c_deep_med'])}；H_NFCI {_pct(hn['c_deep_med'])}", _yn(p7b)],
              ["(c) 排名比自己的固定比例雙胞胎高", f"SHG {_rank(sg)}；雙胞胎（永遠 {TW['SHG']['c'] * 100:.1f}%）{_rank(tsg)}、虧損 {_pct(tsg['c_deep_med'])}", _yn(p7c)],
              ["持有比例（2015 起比特幣日線）", f"SHG {sg['exp'] * 100:.0f}%；SH {sh['exp'] * 100:.0f}%；H_NFCI {hn['exp'] * 100:.0f}%；均線 {base['exp'] * 100:.0f}%", "—"],
              ["對均線 多｜淺｜都：日線", f"SHG {_trip(V['SHG']['d'])}；SH {_trip(V['SH']['d'])}；H_NFCI {_trip(V['H_NFCI']['d'])}", "—"],
              ["對均線 多｜淺｜都：一般人", f"SHG {_trip(V['SHG']['p'])}；SH {_trip(V['SH']['p'])}；H_NFCI {_trip(V['H_NFCI']['p'])}", "—"],
              ["結論（事先寫好：三項都過＝有加分）", "—", "有加分" if adds else "沒有超過兩者較好的那個"]]
    tables.append({"caption": ("第七部分：和半倉條件一起用。H_NFCI（half_gate）＝均線持有，但美國金融情勢指數 NFCI 比自己一年平均緊時只放一半；"
                               "SHG＝每天取 SH 和 H_NFCI 比較小的比例。scale_in 量過 +10% 版本合用沒有加分。"),
                   "columns": ["項目", "數字", "結果"], "rows": rows12})

    # ----- years (context) -----
    yr = R["years"]
    rows13 = [[f"{y}" + ("（到目前）" if y == max(yr["ma50"]) else ""), _pct(yr["ma50"][y]), _pct(yr["SH"].get(y)),
               _pp(yr["SH"].get(y) - yr["ma50"][y]) if _ok(yr["SH"].get(y)) else "—"] for y in sorted(yr["ma50"])]
    tables.append({"caption": "參考：比特幣 UTC 0 點日線照課本做（0.10%）每個日曆年的報酬，2015 年起。",
                   "columns": ["年份", "50 日均線", "SH", "SH 減均線"], "rows": rows13})

    # ----- today -----
    last = close.index[-1]
    P_now = {"ma50": R["book"].pos(si.SPECS["ma50"])["btc"], "SH": lab.pos()["btc"], "SHG": lab.shg()["btc"]}
    s_last, e_last = int(sb.ent[-1]), int(sb.end[-1])
    t_last = int(conf_sh[-1])
    cs = float(close.iloc[s_last])
    ent_txt = f"{_d(close.index[s_last])} 收盤 {cs:,.0f} 美元站上均線"
    ent_txt += f"（那天{'已經' if flag[s_last] else '不'}是 20 日收盤新高）"
    ent_txt += (f"；{_d(close.index[e_last])} 跌回均線下、已出場" if e_last < len(close) else "；到最後一根日線仍在均線上")
    ent_txt += (f"；{_d(close.index[t_last])} 收盤 {float(close.iloc[t_last]):,.0f} 美元是進場後第一次 20 日收盤新高（比進場價"
                f"{_f(float(close.iloc[t_last]) / cs - 1)}），SH 那天加滿" if t_last >= 0
                else "；進場之後到出場（或到現在）收盤沒有創過 20 日新高，SH 一直只放一半")
    need = float(close.iloc[-(WIN - 1):].max())
    on_now = e_last >= len(close)
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": "比特幣：50 日均線｜SH", "value": "｜".join(_amount(P_now[k][-1]) for k in ("ma50", "SH"))},
        {"label": "比特幣：SHG（SH 和 NFCI 半倉條件取小的）", "value": _amount(P_now["SHG"][-1])},
        {"label": "最近一次 50 日均線進場", "value": ent_txt},
        {"label": "下一根日線要創 20 日收盤新高需要",
         "value": (f"收盤高於 {need:,.0f} 美元（最近 19 根收盤的最高）"
                   + ("；均線目前持有且 SH 還沒加滿，達到就會加滿" if on_now and t_last < 0 else
                      "；SH 已經加滿，這個價格只供參考" if on_now else "；均線目前不持有，這個價格不會讓 SH 買進"))},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態（持有＝100%、半倉＝部分、現金＝0%），不是買賣建議"},
    ]

    # ----- verdict -----
    items = {"grid": robust, "placebo": pb, "halves": half_ok["pre"] and half_ok["post"], "assets": agree >= 3}
    k_ok = sum(items.values())
    word = "站得住" if k_ok == 4 else ("部分站得住" if k_ok >= 2 else "站不住")
    if not reproduced:
        word = "重現不完全，讀法暫停"
    cs_, cf_ = cmp_["sh_t"], cmp_["s10_t"]
    vsh = V["SH"]
    fy = "、".join(f"{y}（{w}）" for y, w in fail_years) or "沒有"
    verdict = (
        f"{word}（事先寫好的讀法：四項全過＝站得住，過兩到三項＝部分站得住）。四項各自量到的："
        f"(1) 換設定 18 種裡 {gcount} 種、起手 50% 的 6 種新高天數裡 {w50} 種三項都過（要 ≥ {ROBUST_N} 且 ≥ {ROBUST_W}）→ {g_word}，{_yn(items['grid'])}；"
        f"(2) 安慰劑（排名一樣好｜兩個都做到一樣多的抽樣比例）：(b) 同樣等待天數換到別筆交易 {_share(PL['b']['rank'])}｜{_share(PL['b']['both'])}、"
        f"(c) 同樣多筆數的價格門檻（比特幣 +{xb[0] * 100:.1f}%）排名 {_share(PL['c']['low'])}、兩個都做到 {_share(PL['c']['dboth'])}、"
        f"(d) 新高旗標錯開 {_share(PL['d']['rank'])}｜{_share(PL['d']['both'])}；另外 (a) 同一批交易亂抽日子 {_share(PL['a']['rank'])}｜{_share(PL['a']['both'])} → {_yn(items['placebo'])}；"
        f"(3) 2019 年以前起算 SH {_rank(SP['SH']['pre'])} 對均線 {_rank(SP['ma50']['pre'])}、虧損 {_pct(SP['SH']['pre']['c_deep_med'])} 對 {_pct(SP['ma50']['pre']['c_deep_med'])}；"
        f"2019 年起 {_rank(SP['SH']['post'])} 對 {_rank(SP['ma50']['post'])}、{_pct(SP['SH']['post']['c_deep_med'])} 對 {_pct(SP['ma50']['post']['c_deep_med'])} → {_yn(items['halves'])}；"
        f"(4) 其他 5 個資產 {agree} 個一致 → {_yn(items['assets'])}。"
        f"重現：scale_in 的 SH 數字{'全部重現' if reproduced else '有 ' + str(len(diff0)) + ' 項不一樣'}（排名 {_rank(sh)} 對均線 {_rank(base)}、虧損中位數 {_pct(sh['c_deep_med'])} 對 {_pct(base['c_deep_med'])}、"
        f"雙胞胎 {_rank(TS['SH'])}）。SH 實際上是什麼（量到的）：{N} 筆交易裡 {cnt[None][2][0]} 筆第 1 天就加滿、"
        f"{cnt[None][2][1]} 筆第 2–10 天、{cnt[None][2][2]} 筆更晚、{cnt[None][2][3]} 筆整筆半倉；跑起來的 {nr} 筆裡 {cs_['jr']} 筆加滿"
        f"（第 1 天 {cnt['runner'][2][0]} 筆），每筆平均少賺 {cs_['give'] * 100:.1f} 點（S10 是 {cf_['give'] * 100:.1f} 點）；"
        f"熄火的 {nf_} 筆裡 {nf_ - cs_['fc']} 筆整筆半倉、{cs_['fc']} 筆加滿後還是賠，每筆平均少虧 {cs_['save'] * 100:.1f} 點（S10 是 {cf_['save'] * 100:.1f} 點）。"
        f"老闆的問題（對均線，多｜淺｜都）：日線起算月 {_trip(vsh['d'])}、課本格 {_trip(vsh['c'])}、一般人 {_trip(vsh['p'])}；"
        f"日線兩個都做到不到一半的起算年：{fy}。一般人那一格掉下來，拆開看：只有成本 {_trip(var['cost']['t'])}、只有漏掉 {_trip(var['miss']['t'])}、"
        f"兩個都沒有 {_trip(var['neither']['t'])}。SHG：{'有加分' if adds else '沒有超過兩者較好的那個'}。"
        f"其他量到的：試跑後加的 (b2)（等待天數換到任何一筆交易、不看長度）{_share(PL['b2']['rank'])}｜{_share(PL['b2']['both'])}；"
        f"換設定裡日線「兩個都做到」過半的只有 {gboth}/18 種；拿掉 2022 年，日線兩個都做到變 {loyo[2022]['cnt'][2]}/{loyo[2022]['cnt'][3]}"
        f"（原本 {loyo[None]['cnt'][2]}/{loyo[None]['cnt'][3]}）；一筆一筆拿掉最大的 {len(lro)} 筆大漲交易，{lro_keep}/{len(lro)} 次 SH 排名仍比均線高。"
        "推測（不是量到的）：放得少本身就會讓虧損變淺，所以只有雙胞胎和安慰劑的比較算數；"
        f"SH 是在 scale_in 看過四條規則之後挑出來的，這裡就算過了也還是同一段歷史，真正的考驗是之後每天的影子紀錄。整段只有 {nr} 筆跑起來的交易。不是買賣建議。")

    notes = [
        ("事先寫死的設計（照原文）：grid 33/50/67% x HIGH 10/15/20/30/40/55; meets = ranking >= ma50 AND strictly "
         "shallower median deepest loss AND ranking strictly above own CONST twin; robust >= 13/18 AND >= 4/6 windows at 50%; "
         "fragile < 9/18. Placebos (b) same waiting time moved to another trade of similar length (blocks of 5 by length), "
         "(c) price threshold giving the same number of confirmed trades per series, (d) HIGH(20) flag rolled >= 180 closes; "
         "SH beats one when share of draws with ranking >= SH AND share with daily 'both' >= SH are both <= 5%. Verdict: "
         "grid robust; beats (b)(c)(d); both halves ranking >= ma50 and shallower; >= 3/5 assets agree. 4 = 站得住, 2-3 = "
         "部分站得住."),
        ("和以前「新高」研究的差別：main_signal_check——55 日新高進、20 日新低出（55／20）換起算日和收盤時間後中位數只比抱著多 +3.8 點，"
         "一般人照做只有 48% 贏過抱著（50 日均線 81%）；rule_upgrades——10 種改良（包括 C「20 日新高先買一半、55 日新高買滿、20 日新低全出」）"
         "沒有一種三關全過；donchian_ensemble——論文的九條通道新高規則 2018 年起每年 +13.5%、最深虧 -19.9%，但平均只持有 19%，"
         "最簡單的 20 日新高進、10 日新低出每年 +29.3%、最深虧 -56.8%。那些都是用新高決定「進不進場」；SH 不同：進出場完全照 50 日均線，"
         "新高只決定均線已經打開的這筆交易放一半還是放滿。所以那些結果不能直接拿來說 SH 會不會有用，反過來也一樣。"),
        ("「20 日新高」的精確定義：收盤價高於前 19 根收盤的最高（也就是最近 20 根收盤裡最高的那根）。"
         "進場那天本身不算，從進場隔天的收盤開始找；找到後一直滿倉到均線出場，中間不會降回一半。"),
        ("安慰劑怎麼做：(a) 和 scale_in 的 RAND-B 一樣，同一批 SH 有加滿的交易，加滿的日子在那筆交易裡隨便抽；"
         "(b) 把每筆交易照長度排好、5 筆一組，組內打亂後每筆拿「另一筆」SH 實際等了幾天（沒加滿也算一種），等那麼多天後加滿，超過這筆交易長度就不加；"
         "(c) 每個資產、每個收盤時間各自找一個漲幅門檻，讓「進場後曾經漲到這個門檻」的交易筆數剛好等於 SH 加滿的筆數；"
         "(d) 把「哪幾天是 20 日新高」整串往後錯開 180 天到（總天數−180 天）之間隨便一個天數（尾巴繞回開頭），再照 SH 的規則加滿。"
         "所有對照和 SH 都用一般人做法每格 50 次模擬重算再比（scale_in 也是這樣）。"),
        ("(b) 怎麼讀：(b) 保留了「等幾天才加滿」，也保留了「交易多長」，只拿掉「加滿那天是不是新高」。如果 (b) 和 SH 一樣好，"
         "這份資料分不開兩種說法：一是新高本身沒有比「這筆交易撐過頭幾天」多帶什麼資訊；二是照長度配對讓對照知道哪些交易後來會拖很長"
         "（長的交易多半是大漲的），對照本身太有利。(b2) 不看長度，只能部分分開這兩種說法，不能完全分開。"),
        ("一筆一筆拿掉大漲交易怎麼做：只拿掉 SH 和均線那段日期的持有，一直抱著的對照沒有拿掉，所以拿掉之後兩條規則贏抱著的比例都會下降；"
         "要看的是 SH 減均線的差還在不在。24 個收盤時間的交易日期和日線不完全一樣，拿掉的是同一段日期。"),
        ("第四部分怎麼讀：「錢更多」＝從起算日到現在的每年報酬比均線高；「虧損更淺」＝這段期間最深的跌幅比均線淺。"
         "相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"),
        ("限制：(1) 大約 23 筆跑起來的交易撐起全部獲利，少數幾筆的加碼時間就能左右結果；(2) 只有一個資產的長歷史，其他資產是同一種市場行為，"
         "不是完全獨立的證據；(3) SH 是 scale_in 看過四條規則之後才被挑出來的，這次就算全部過關也還是用同一段歷史，"
         "真正的考驗是之後每天的影子紀錄；(4) 放得比較少的規則幾乎一定虧得比較淺，所以只有雙胞胎和安慰劑的比較有意義；"
         "(5) 換設定、安慰劑、拿掉大漲交易都是在同一段資料上做，只能說「這樣做沒有把它打破」或「打破了」，不能證明以後有效。"
         "「沒有看到」只表示用這個定義和這些資料沒有看到。"),
        (f"計算方式：考試直接用 scale_in 的 _exam／_score／_vs（成本、漏掉進場、雙胞胎、每個起算點的數字都一樣）；"
         f"安慰劑、拿掉大漲交易、成本掃描用每格 50 次模擬，均線和 SH 也用同樣方式重算。H_NFCI 的 NFCI 來自{data['where']}。"
         f"這次計算用了 {time.time() - t0:.0f} 秒（主要考試 {R['t'][0]:.0f} 秒、安慰劑 {R['t'][1]:.0f} 秒、換設定 {R['t'][2]:.0f} 秒）。"),
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
        "title": "分批進場的驗證：創 20 日新高才加滿（SH）",
        "question": ("scale_in 裡唯一過關的分批方式 SH（均線進場先放一半，第一次創 20 日收盤新高才加滿）是真的有用，還是四條裡剛好運氣好的那條？"
                     "換設定、換更難的安慰劑、切開時間、拿掉大漲的交易、換資產、照一般人方式做，它還能不能「錢比均線多、最深虧損也比均線淺」？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
