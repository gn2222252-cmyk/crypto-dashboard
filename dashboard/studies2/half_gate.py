"""Historical study "half_gate": when ma50 is on and a slow "conditions" gate is unfavorable, hold HALF instead of
nothing. Does that earn at least as often as ma50 AND lose less - and is it the timing of the half days that matters,
or only that less is held?

Why this study exists: gate_design measured its design D2 (= the half rule below with the NFCI and the OECD gate) at
ranking score 84% / 85% (ma50 80%), median deepest loss -35.6% / -36.5% (ma50 -53.7%), placebo 2% / 2%, but labelled it
"fail" because its bar compared D2 with the all-or-nothing gate (+3 points on both indicators), not with ma50. This
study asks the owner's question instead (earn more AND lose less than the main signal) and is built so it can fail:
the half rule must also beat a constant-share twin that holds exactly as much on average, and a placebo at 5%.

Pre-registered (written before this study was run; nothing tuned after seeing results; later additions labelled).
  Signal: ma = main_signal_check's ma50 state after each close (1 on / 0 off). Gate flags are monthly (decided at the
  last 00:00 UTC daily close of month M, held during M+1; unknown = unfavorable), built exactly as in the parents:
    NFCI : nfci_filter N1 flag - latest weekly NFCI at least 6 days old below the average of its last 52 weekly values.
    OECD : business_cycle/oecd_followup - OECD US CLI (value two months old) above the average of its last 10 values.
    HYG  : stress_family "credit" - HYG/IEF (adjusted closes, SPY calendar, carry <= 5 days) above its 252-day average,
           flag of the last SPY day dated on or before the last calendar day of M.
    LQD  : stress_family "quality" - LQD/IEF, same.
    OFR  : nfci_stress sibling - OFR Financial Stress Index below its 252-value average, a value dated x usable once
           x + 2 business days + 1 day <= the last calendar day of M.
  Rules (all: ma off = cash; ma on = 100% when the gate is favorable, 50% when unfavorable):
    H_NFCI, H_OECD, H_HYG, H_LQD, H_OFR; H_VOTE = 100% when at least half of the gates that have a flag that month are
    favorable, else 50% (no gate with a flag = 50%).
  Baselines on the same starts: ma50; D0 = the all-or-nothing version of each gate (ma * g); CONST twin = ma50 at a
  constant share c equal to the rule's own average share on days ma50 is on (BTC 00:00 UTC daily from 2015-01-01), so
  the twin holds exactly as much on average and only the WHEN differs; CONST75 = ma50 at 75% (reference); RAND = the
  half days assigned by a circular shift of the gate (the placebo distribution: its median ranking and loss shown).
  Exam: business_cycle._Exam arithmetic (main_signal_check's four checks: BTC daily 108 monthly starts 2015-01..2023-12;
  72 monthly starts x 24 closing hours = 1,728 textbook cells at 0.10% cost; ordinary person = 10 hours later, 0.25%
  cost, 10% of entries missed, 200 simulations per clock; ETH/SOL/BNB/XRP/MSTR with their own ma50 and the same US
  gate). Ranking score = lowest of the four "beats holding" shares, eligible only if the deepest loss is shallower than
  holding's in >= 95% of BTC starts and cells. The exam is re-implemented here line for line so that per-start arrays
  can be kept (Part 4); ma50 and D0 NFCI are checked to equal _Exam.summary. Seeds: ma50 uses main_signal_check's
  (business_cycle KI "ma50"), every other rule uses the parents' "m2" seeds.
  PART 1 - standard bar versus ma50 (all four): (1) ranking score >= ma50's (business_cycle._score_ge); (2) median
    deepest loss over the 1,728 textbook cells shallower (strictly) than ma50's; (3) placebo share <= 5%: every gate's
    raw series circularly shifted by >= about 6 months (NFCI 26 weeks with nfci_filter's stream SEED+41, OECD 6 months
    with oecd_followup's stream SEED+31, HYG/IEF and LQD/IEF and OFR 126 trading/business days with own streams
    SEED+61/62/63; draw i of every stream; H_VOTE uses all its gates shifted independently), flags rebuilt, rule
    re-scored with 50 ordinary-person simulations per clock and the real rule re-scored the same way, share of 200
    draws with ranking at least as high; (4) ranking score strictly above its constant-share twin's (eligibility as in
    _score_ge). Reported: the four parts, median and worst deepest loss, time-weighted exposure, calendar-year return
    minus ma50 for every year (also averaged with and without 2018 and 2022), and return per unit of deepest loss =
    median over the 1,728 textbook cells of (yearly return / |deepest loss|). Multiple testing: Holm across the six
    rules on placebo p = (hits + 1) / (draws + 1); a rule "passes after Holm" when it passes all four AND its
    Holm-adjusted p <= 0.05. Luck: expected number of rules clearing the placebo part by chance = 6 x 0.05.
  PART 2 - one lucky setting? H_NFCI on nfci_stress's 20-setting grid (window 26/39/52/78/104 weeks x extra delay
    0/1/2/4 weeks); OECD windows 5/10/20 months; HYG, LQD, OFR windows 126/252/504 days (H_VOTE with all windows scaled
    together, report only). Each cell scored with 50 ordinary-person simulations per clock, ma50 re-scored the same
    way. Count of cells with ranking >= ma50's AND shallower median deepest loss. "Robust" if >= 15 of 20 for NFCI AND
    >= 2 of 3 windows for at least two of OECD/HYG/LQD/OFR.
  PART 3 - independent data: does H_OFR / H_HYG / H_LQD repeat H_NFCI (ranking >= ma50's AND shallower median deepest
    loss, full exam)? "Repeats" if at least 2 of the 3 do. Decision frequency (report only): H_NFCI decided weekly
    (nfci_stress's daily lookup of the latest value at least 6 days old) versus monthly.
  PART 4 - the owner's question in plain counts, each rule versus ma50 on the same starts: how many of the 108 BTC
    daily starts, of the 1,728 textbook cells, and of the 1,728 x 200 ordinary-person runs (simulation i of the rule
    paired with simulation i of ma50; their missed entries are drawn separately) end with MORE money (higher yearly
    return), with a SHALLOWER deepest loss, and with BOTH; per start year how many of the 12 daily starts achieve both.
    Report only, no bar.
  PART 5 - EXPLORATORY, added after seeing 2020 (the two weaknesses were found by looking at results, so these cannot
    be judged on 2020): BUF = NFCI gate unfavorable only when NFCI is more than 0.05 above its 52-week average;
    EASING = favorable when below its 52-week average OR lower than 8 weeks earlier. Same half rule, full exam, the 12
    months of 2020, and a version with the 2020 held months forced back to the original gate (= "2020 excluded"), plus
    yearly differences averaged without 2020. No placebo, no bar.
  Overall reading, fixed now: "站得住" if at least one rule passes Part 1 after Holm AND Part 2 is robust AND Part 3
    repeats; "部分站得住" if at least one rule passes Part 1 (before Holm) and at least one of Part 2 / Part 3 holds;
    otherwise "這次沒有達到". Look-ahead check: gate values not yet usable after three month-ends are scrambled; no
    monthly flag decided up to then may change. Today: states only.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import nfci_stress as nfs
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import stress_family as sf

DAY = bc.DAY
FIRST = msc.DAILY_STARTS[0]
DRAWS, PL_SIMS, SIMS = bc.DRAWS, bc.PL_SIMS, bc.SIMS
PLC_BAR, HOLM_ALPHA = 0.05, 0.05
GAP = {"nfci": nf.GAP_W, "oecd": bc.GAP, "hyg": 126, "lqd": 126, "ofr": 126}
SEEDS = {"nfci": bc.SEED + 41, "oecd": bc.SEED + 31, "hyg": bc.SEED + 61, "lqd": bc.SEED + 62, "ofr": bc.SEED + 63}
BASEWIN = {"nfci": 52, "oecd": bc.AVG, "hyg": sf.AVG_D, "lqd": sf.AVG_D, "ofr": 252}
WIN_UNIT = {"nfci": "週", "oecd": "個月", "hyg": "個交易日", "lqd": "個交易日", "ofr": "個營業日"}
SCALES = (0.5, 1.0, 2.0)
ROBUST_N, ROBUST_OTHERS = 15, 2
BUF, EASE_W = 0.05, 8
SKIP_YEARS = (2018, 2022)
GATES = ("nfci", "oecd", "hyg", "lqd", "ofr")
GNAME = {"nfci": "NFCI", "oecd": "OECD", "hyg": "HYG", "lqd": "LQD", "ofr": "OFR"}
GDESC = {"nfci": "芝加哥聯準銀行金融情勢指數 NFCI 低於自己 52 週平均（金融情勢比平常寬鬆）",
         "oecd": "OECD 美國領先指標（兩個月前的數字）高於自己 10 個月平均（景氣往上）",
         "hyg": "HYG÷IEF（高收益債 ETF 除以 7–10 年公債 ETF）高於自己 252 個交易日平均（垃圾債比公債強）",
         "lqd": "LQD÷IEF（投資等級公司債 ETF 除以 7–10 年公債 ETF）高於自己 252 個交易日平均",
         "ofr": "OFR 金融壓力指數（美國財政部金融研究辦公室）低於自己 252 個營業日平均（壓力比平常低）"}
RULES = ("H_NFCI", "H_OECD", "H_HYG", "H_LQD", "H_OFR", "H_VOTE")
RGATE = {"H_NFCI": "nfci", "H_OECD": "oecd", "H_HYG": "hyg", "H_LQD": "lqd", "H_OFR": "ofr", "H_VOTE": "vote"}
YAHOO_SYMS = ("HYG", "IEF", "LQD", "SPY")

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _credit() -> tuple[pd.DataFrame | None, list[str]]:
    """HYG/IEF and LQD/IEF on SPY's trading-day calendar, exactly as stress_family builds them."""
    raw, notes = {}, []
    for sym in YAHOO_SYMS:
        s, note = sf._ticker(sym)
        if note:
            notes.append(note + "。")
        if s is not None:
            raw[sym] = s
    if "SPY" not in raw or "IEF" not in raw:
        return None, notes + ["SPY 或 IEF 取不到，HYG、LQD 兩個條件這次沒有算。"]
    cal = raw["SPY"].index
    panel = {}
    for key, a in (("hyg", "HYG"), ("lqd", "LQD")):
        if a not in raw:
            notes.append(f"{a} 取不到，H_{a} 這次沒有算。")
            continue
        both = pd.concat([raw[a], raw["IEF"]], axis=1, join="inner").dropna()
        v = both.iloc[:, 0] / both.iloc[:, 1]
        panel[key] = v.reindex(cal.union(v.index)).ffill(limit=sf.CARRY).reindex(cal).where(cal >= v.index[0])
    return pd.DataFrame(panel, index=cal), notes


def _load() -> dict:
    nfci, where_n, problems = nf._series()
    oecd, where_o, note = bc._series("oecd")
    if note:
        problems.append(note)
    panel, more = _credit()
    problems += more
    ofr, where_ofr, more = nfs._sibling("ofr")
    problems += more
    other, more = msc._load_other()
    problems += more
    raw = {"nfci": nfci, "oecd": oecd}
    if panel is not None:
        for k in panel:
            raw[k] = panel[k]
    if ofr is not None:
        raw["ofr"] = ofr
    return {"close": bc.te._daily_close(), "raw": raw, "other": other, "problems": problems,
            "where": {"nfci": where_n, "oecd": where_o, "ofr": where_ofr}}


# ---------- gate flags ----------

def _flag(gate: str, s: pd.Series, months: pd.PeriodIndex, win: int | None = None, delay: int = 0) -> pd.Series:
    """Monthly flag (decision month -> 1 favorable / 0 / NaN) exactly as the parent studies build it."""
    w = win or BASEWIN[gate]
    if gate == "nfci":
        return nfs._monthly(nfs._fav(s, w), s.index + nf.SAFE + pd.Timedelta(weeks=delay), months)
    if gate == "oecd":
        avg = s.rolling(w, min_periods=w).mean()
        fav = (s > avg).astype(float).where(avg.notna() & s.notna())
        fav.index = fav.index + bc.LAG["oecd"]
        return fav.reindex(months)
    if gate in ("hyg", "lqd"):
        avg = s.rolling(w, min_periods=w).mean()
        fav = (s > avg).astype(float).where(avg.notna() & s.notna())
        return sf._monthly(fav.to_frame(gate), months)[gate]
    if gate == "ofr":
        return nfs._monthly(nfs._fav(s, w), nfs._usable("ofr", s.index), months)
    raise ValueError(gate)


def _vote(G: pd.DataFrame) -> pd.Series:
    have = G.notna().sum(axis=1)
    return (G.sum(axis=1) >= have / 2).astype(float).where(have > 0)


def _gates(raw: dict, months: pd.PeriodIndex, scale: float = 1.0) -> pd.DataFrame:
    cols = {}
    for g in GATES:
        if g in raw:
            win = None if scale == 1.0 else max(2, int(round(BASEWIN[g] * scale)))
            cols[g] = _flag(g, raw[g], months, win)
    G = pd.DataFrame(cols, index=months)
    G["vote"] = _vote(G[[g for g in GATES if g in G]])
    return G


def _buf_flag(s: pd.Series, months) -> pd.Series:
    avg = s.rolling(52, min_periods=52).mean()
    fav = (s <= avg + BUF).astype(float).where(avg.notna() & s.notna())
    return nfs._monthly(fav, s.index + nf.SAFE, months)


def _ease_flag(s: pd.Series, months) -> pd.Series:
    avg = s.rolling(52, min_periods=52).mean()
    prev = s.shift(EASE_W)
    fav = ((s < avg) | (s < prev)).astype(float).where(avg.notna() & s.notna() & prev.notna())
    return nfs._monthly(fav, s.index + nf.SAFE, months)


# ---------- the exam (business_cycle._Exam.summary, line for line, keeping per-start arrays) ----------

def _macro(F, dates) -> np.ndarray | float:
    if F is None:
        return 0.0
    if isinstance(F, nfs._DailyFlag):
        return F.at(dates)
    return bc._on_days(F, dates)


def _posf(mode: str, ma, g, c: float):
    if mode == "ma50":
        return ma
    if mode == "d0":
        return ma * g
    if mode == "half":
        return ma * (0.5 + 0.5 * np.asarray(g, dtype=float))
    if mode == "const":
        return ma * c
    raise ValueError(mode)


def _btc_pos(ex: bc._Exam, mode: str, F=None, c: float = 1.0) -> np.ndarray:
    return _posf(mode, ex.btc_ma, _macro(F, ex.close.index + DAY), c)


def _exam(ex: bc._Exam, mode: str, F=None, c: float = 1.0, sims: int = SIMS, keep: bool = False) -> dict:
    ki = bc.KI["ma50"] if mode == "ma50" else bc.KI["m2"]
    pos = _btc_pos(ex, mode, F, c)
    net, _ = bc.te._run(np.concatenate([[np.nan], pos[:-1]]), ex.btc_change, bc.COST)
    dr, dd = msc._suffix(msc._equity(net), ex.dstarts, bc.YEAR)
    dr, dd = dr[:, 0], dd[:, 0]
    dh = ex.dhold
    cmac = _macro(F, ex.m.dates)
    n = len(ex.cstarts)
    tr, td = np.zeros((n, 24)), np.zeros((n, 24))
    pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
    for h in range(24):
        w = np.nan_to_num(_posf(mode, ex.cma[h], cmac, c), nan=0.0).astype(float)
        r, d_ = msc._suffix(np.concatenate([[1.0], ex.eng._mid(h, w, bc.COST, None)]), ex.cstarts, bc.YEAR)
        tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
        b, wl = rt._later(h, w, ex.days)
        rng = np.random.default_rng(bc.SEED + 1000 * ki + h)
        p = rt._missed(wl, bc.P_MISS, rng, sims)
        r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), ex.eng._mid(b, p, bc.P_COST, None)]), ex.cstarts, bc.YEAR)
        pr[:, h, :], pdd[:, h, :] = r, d_
    hr, hd = ex.chold
    edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
    each = {}
    for a, o in ex.other.items():
        p = _posf(mode, o["ma"], _macro(F, o["close"].index + DAY), c)
        net2, _ = bc.te._run(np.concatenate([[np.nan], p[:-1]]), o["change"], bc.COST)
        r, d_ = msc._suffix(msc._equity(net2), o["starts"], o["year"])
        each[a] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((d_[:, 0] > o["hold"][1]).sum()),
                   float(np.median(r[:, 0] - o["hold"][0])))
    x = {"d_n": len(dr), "d_beat": int((dr > dh[0]).sum()), "d_shallow": int((dd > dh[1]).sum()),
         "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
         "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
         "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
         "p_edge": float(np.median(pedge)), "p_deep_med": float(np.median(pdd)),
         "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
    x["low"] = min(x["checks"])
    x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
    x["eligible"] = x["loss_ok"] >= bc.LOSS_BAR
    ok = td < -1e-9
    x["ratio"] = float(np.median(tr[ok] / np.abs(td[ok]))) if ok.any() else None
    a = ex.close.index >= pd.Timestamp(FIRST)
    pa = np.nan_to_num(pos[a])
    x["exp"] = float(pa.mean())
    on = np.nan_to_num(ex.btc_ma[a]) > 0.5
    x["exp_on"] = float(pa[on].mean()) if on.any() else None
    if keep:
        x["arr"] = {"dr": dr, "dd": dd, "tr": tr, "td": td, "pr": pr, "pdd": pdd}
    return x


def _above(x: dict, twin: dict) -> bool:
    """Ranking score strictly above the twin's (eligibility first, as in business_cycle._score_ge)."""
    if twin["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not twin["eligible"]:
        return True
    return x["low"] > twin["low"] + 1e-12


def _meets(x: dict, base: dict) -> bool:
    return bool(bc._score_ge(x, base) and x["c_deep_med"] > base["c_deep_med"] + 1e-12)


def _holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def _vs(x: dict, base: dict) -> dict:
    """Part 4 counts of x against ma50 on the same starts."""
    a, b = x["arr"], base["arr"]
    out = {}
    for lab, r1, d1, r0, d0 in (("d", a["dr"], a["dd"], b["dr"], b["dd"]), ("c", a["tr"], a["td"], b["tr"], b["td"]),
                                 ("p", a["pr"], a["pdd"], b["pr"], b["pdd"])):
        more_, shal = r1 > r0 + 1e-9, d1 > d0 + 1e-9
        out[lab] = (int(more_.sum()), int(shal.sum()), int((more_ & shal).sum()), int(more_.size))
    both_d = (a["dr"] > b["dr"] + 1e-9) & (a["dd"] > b["dd"] + 1e-9)
    both_c = ((a["tr"] > b["tr"] + 1e-9) & (a["td"] > b["td"] + 1e-9)).mean(axis=1)
    out["d_mask"], out["c_share"] = both_d, both_c
    return out


# ---------- look-ahead ----------

def _lookahead(raw: dict, months: pd.PeriodIndex) -> tuple[int, int]:
    rng = np.random.default_rng(19)
    total = bad = 0
    G0 = _gates(raw, months)
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        cm_ = c.to_period("M")
        raw2 = {}
        for g, s in raw.items():
            if g == "oecd":
                known = s.index <= cm_ - bc.LAG["oecd"]
            elif g == "nfci":
                known = (s.index + nf.SAFE) <= c
            elif g == "ofr":
                known = nfs._usable("ofr", s.index) <= c
            else:
                known = s.index <= c
            raw2[g] = s.where(known, s * np.exp(rng.normal(0, 0.5, len(s))) + rng.normal(0, 1.0, len(s)))
        G2 = _gates(raw2, months)
        upto = months <= cm_
        for col in G0:
            total += 1
            bad += int(not np.array_equal(G0[col][upto].fillna(-1).values, G2[col][upto].fillna(-1).values))
    return total, bad


# ---------- the computation ----------

def _shifted(raw: dict, i_shift: dict) -> dict:
    return {g: pd.Series(np.roll(s.values, int(i_shift[g])), index=s.index) for g, s in raw.items()}


def _compute(data: dict) -> dict:
    t0 = time.time()
    close, raw = data["close"], data["raw"]
    first = min(s.index[0].to_period("M") if isinstance(s.index, pd.DatetimeIndex) else s.index[0] for s in raw.values())
    months = pd.period_range(first, close.index[-1].to_period("M"), freq="M")
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    G = _gates(raw, months)
    rules = [r for r in RULES if RGATE[r] in G]

    # identity check against business_cycle._Exam.summary
    chk_ma = ex.summary("ma50", None)
    base = _exam(ex, "ma50", keep=True)
    chk_d0 = ex.summary("m2", G["nfci"])
    d0n = _exam(ex, "d0", G["nfci"])
    same = all(abs(chk_ma[k] - base[k]) < 1e-9 for k in ("low", "c_deep_med", "c_deep_worst", "p_edge")) and \
        all(abs(chk_d0[k] - d0n[k]) < 1e-9 for k in ("low", "c_deep_med", "c_deep_worst", "p_edge"))

    # Part 1
    s = {r: _exam(ex, "half", G[RGATE[r]], keep=True) for r in rules}
    d0 = {r: (d0n if r == "H_NFCI" else _exam(ex, "d0", G[RGATE[r]])) for r in rules}
    twin = {r: _exam(ex, "const", None, c=s[r]["exp_on"]) for r in rules}
    c75 = _exam(ex, "const", None, c=0.75, keep=True)
    small = {r: _exam(ex, "half", G[RGATE[r]], sims=PL_SIMS) for r in rules}
    t1 = time.time() - t0

    shifts = {g: np.random.default_rng(SEEDS[g]).integers(GAP[g], len(raw[g]) - GAP[g] + 1, DRAWS) for g in raw}
    hits = {r: 0 for r in rules}
    lows, deeps = {r: [] for r in rules}, {r: [] for r in rules}
    for i in range(DRAWS):
        G2 = _gates(_shifted(raw, {g: shifts[g][i] for g in raw}), months)
        for r in rules:
            x = _exam(ex, "half", G2[RGATE[r]], sims=PL_SIMS)
            lows[r].append(x["low"])
            deeps[r].append(x["c_deep_med"])
            hits[r] += int(bc._score_ge(x, small[r]))
    plc = {r: hits[r] / DRAWS for r in rules}
    pval = {r: (hits[r] + 1) / (DRAWS + 1) for r in rules}
    holm = _holm(pval)
    rand = {r: (float(np.median(lows[r])), float(np.median(deeps[r]))) for r in rules}
    t_plc = time.time() - t0 - t1

    # Part 2
    small_ma = _exam(ex, "ma50", sims=PL_SIMS)
    grid = []
    for w in nfs.WINDOWS:
        for dl in nfs.DELAYS:
            x = _exam(ex, "half", _flag("nfci", raw["nfci"], months, w, dl), sims=PL_SIMS)
            grid.append({"w": w, "dl": dl, "x": x, "meets": _meets(x, small_ma),
                         "score_ok": bool(bc._score_ge(x, small_ma)),
                         "loss_ok": x["c_deep_med"] > small_ma["c_deep_med"] + 1e-12})
    wins = {}
    for g in ("oecd", "hyg", "lqd", "ofr"):
        if g not in raw:
            continue
        wins[g] = []
        for sc in SCALES:
            w = max(2, int(round(BASEWIN[g] * sc)))
            x = _exam(ex, "half", _flag(g, raw[g], months, w), sims=PL_SIMS)
            wins[g].append({"w": w, "x": x, "meets": _meets(x, small_ma)})
    vote_sc = []
    for sc in SCALES:
        x = _exam(ex, "half", _gates(raw, months, sc)["vote"], sims=PL_SIMS)
        vote_sc.append({"sc": sc, "x": x, "meets": _meets(x, small_ma)})

    # Part 3: weekly decisions for H_NFCI
    wk = nfs._DailyFlag(nfs._fav(raw["nfci"], 52), raw["nfci"].index + nf.SAFE)
    weekly = _exam(ex, "half", wk)
    pos_w = _btc_pos(ex, "half", wk)

    # Part 5 (exploratory)
    Fb, Fe = _buf_flag(raw["nfci"], months), _ease_flag(raw["nfci"], months)
    m20 = (months >= pd.Period("2019-12", "M")) & (months <= pd.Period("2020-11", "M"))  # decisions for held 2020
    Fb_x, Fe_x = Fb.copy(), Fe.copy()
    Fb_x[m20], Fe_x[m20] = G["nfci"][m20].values, G["nfci"][m20].values
    ex5 = {"buf": _exam(ex, "half", Fb, keep=True), "ease": _exam(ex, "half", Fe, keep=True),
           "buf_x": _exam(ex, "half", Fb_x), "ease_x": _exam(ex, "half", Fe_x)}
    F5 = {"buf": Fb, "ease": Fe, "buf_x": Fb_x, "ease_x": Fe_x}

    # positions, years
    pos = {"ma50": _btc_pos(ex, "ma50"), **{r: _btc_pos(ex, "half", G[RGATE[r]]) for r in rules},
           **{f"D0_{r}": _btc_pos(ex, "d0", G[RGATE[r]]) for r in rules},
           **{k: _btc_pos(ex, "half", F5[k]) for k in F5}, "weekly": pos_w}
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST), **{k: bc._years(v, close, FIRST) for k, v in pos.items()}}

    def switches(p) -> int:
        a = np.nan_to_num(np.asarray(p, dtype=float)[close.index >= pd.Timestamp(FIRST)])
        return int((np.abs(np.diff(a)) > 0.25).sum())

    sw = {"monthly": switches(pos["H_NFCI"]), "weekly": switches(pos_w), "ma50": switches(pos["ma50"])}

    # Part 4
    vs = {r: _vs(s[r], base) for r in rules}
    vs["c75"] = _vs(c75, base)
    vs["buf"], vs["ease"] = _vs(ex5["buf"], base), _vs(ex5["ease"], base)
    look = _lookahead(raw, months)
    return {"ex": ex, "G": G, "months": months, "rules": rules, "base": base, "same": same, "s": s, "d0": d0,
            "twin": twin, "c75": c75, "small": small, "plc": plc, "pval": pval, "holm": holm, "rand": rand,
            "small_ma": small_ma, "grid": grid, "wins": wins, "vote_sc": vote_sc, "weekly": weekly, "sw": sw,
            "ex5": ex5, "F5": F5, "pos": pos, "years": years, "vs": vs, "look": look, "problem": problem,
            "t1": t1, "t_plc": t_plc, "secs": time.time() - t0}


# ---------- the study ----------

def _st(x: dict) -> str:
    return _share(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）")


def _yn(v: bool) -> str:
    return "過" if v else "不過"


def _of(a: int, n: int) -> str:
    return f"{a:,}/{n:,}（{a / n * 100:.0f}%）" if n else "—"


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, G, base, s, rules = data["close"], r["ex"], r["G"], r["base"], r["s"], r["rules"]
    raw = data["raw"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    n_c, n_d = base["c_n"], base["d_n"]

    # ----- Part 1 bar -----
    bar = {}
    for k in rules:
        x = s[k]
        q = {"score": bool(bc._score_ge(x, base)), "loss": x["c_deep_med"] > base["c_deep_med"] + 1e-12,
             "plc": r["plc"][k] <= PLC_BAR + 1e-12, "twin": _above(x, r["twin"][k])}
        q["all"] = all(q.values())
        q["holm"] = q["all"] and r["holm"][k] <= HOLM_ALPHA + 1e-12
        bar[k] = q
    p1_raw = [k for k in rules if bar[k]["all"]]
    p1_holm = [k for k in rules if bar[k]["holm"]]

    # ----- table 1: exam -----
    def exam_row(name: str, x: dict) -> list:
        ck = x["checks"]
        return [name, _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]), _st(x),
                f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", f"{x['exp'] * 100:.0f}%",
                _num(x["ratio"], 2) if _ok(x["ratio"]) else "—"]

    rows1 = [exam_row("50 日均線（基準）", base)]
    for k in rules:
        rows1.append(exam_row(f"{k}（不利時半倉）", s[k]))
    for k in rules:
        rows1.append(exam_row(f"D0 {GNAME.get(RGATE[k], '投票')}（不利時現金）", r["d0"][k]))
    for k in rules:
        rows1.append(exam_row(f"{k} 的固定比例雙胞胎（均線持有時永遠 {s[k]['exp_on'] * 100:.1f}%）", r["twin"][k]))
    rows1.append(exam_row("CONST75（均線持有時永遠 75%）", r["c75"]))
    t1 = {"caption": (f"第一部分：公平考試（前四欄都是「報酬贏過同一天起一直抱著」的比例）。比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產 ETH、SOL、BNB、XRP、MSTR "
                      "用自己的 50 日均線、同一個美國條件。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      f"最深虧損＝從起算日到現在，資產從高點往下跌最多的幅度；這裡是 {n_c:,} 格照課本做的中位數｜最差一格。"
                      f"持有比例＝{FIRST[:4]} 年起比特幣日線平均放了幾成（半倉算 0.5）。每單位虧損的報酬＝每格「每年報酬 ÷ 最深虧損」的中位數。"
                      "固定比例雙胞胎：均線說持有時永遠只放某個固定比例，比例剛好等於那條半倉規則在均線持有日子的平均持有比例，"
                      "所以兩者平均放的一樣多，差別只在「什麼時候」放半倉。"),
          "columns": ["規則", f"比特幣起算月（{n_d}）", f"起算月×收盤時間（{n_c:,}）", "照一般人方式做", "其他資產起算月",
                      "排名分數", "最深虧損：中位數｜最差", "持有比例", "每單位虧損的報酬"],
          "rows": rows1}

    # ----- table 2: the bar -----
    rows2 = []
    for k in rules:
        q, x, tw = bar[k], s[k], r["twin"][k]
        rm, rd = r["rand"][k]
        rows2.append([k,
                      f"{_share(x['low'])} 對 {_share(base['low'])} → {_yn(q['score'])}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])} → {_yn(q['loss'])}",
                      f"{_share(r['plc'][k])}（亂排的中位數：排名 {_share(rm)}、虧損 {_pct(rd)}；真的 {_share(r['small'][k]['low'])}） → {_yn(q['plc'])}",
                      f"{_share(x['low'])} 對 {_share(tw['low'])}（雙胞胎虧損 {_pct(tw['c_deep_med'])}） → {_yn(q['twin'])}",
                      f"{r['pval'][k]:.3f} → 校正後 {r['holm'][k]:.3f}",
                      "校正後仍過" if q["holm"] else ("四項都過、校正後不過" if q["all"] else "沒有四項都過")])
    t2 = {"caption": ("第一部分的標準（跑之前寫好，四項都要過）：(1) 排名分數不低於 50 日均線；(2) 最深虧損中位數比均線淺；"
                      f"(3) 隨機錯位（安慰劑）：把條件的原始數字整串循環平移至少大約半年（NFCI 26 週、OECD 6 個月、其他 126 個交易日），"
                      f"{DRAWS} 次、每次重算每月條件和整套考試（一般人做法每格只模擬 {PL_SIMS} 次，真的規則也用同樣方式重算一次再比），"
                      f"排名分數一樣好或更好的比例 ≤ {PLC_BAR * 100:.0f}%；這等於「半倉的日子隨便亂放」的對照組（RAND）；"
                      "(4) 排名分數比固定比例雙胞胎高。六條規則一起比，多重比較用 Holm 校正：p＝（一樣好的次數＋1）÷（次數＋1），"
                      f"校正後 ≤ {HOLM_ALPHA:.2f} 才算「校正後仍過」。"),
          "columns": ["規則", "(1) 排名 對 均線", "(2) 虧損中位數 對 均線", "(3) 隨機錯位", "(4) 排名 對 雙胞胎", "p → Holm 校正後", "結果"],
          "rows": rows2}

    # ----- table 3: years -----
    yrs = sorted(r["years"]["ma50"])
    rows3 = []
    diffs = {k: {} for k in rules}
    for y in yrs:
        m = r["years"]["ma50"].get(y)
        cells = []
        for k in rules:
            v = r["years"][k].get(y)
            if _ok(v) and _ok(m):
                diffs[k][y] = v - m
            cells.append(_pp(v - m) if _ok(v) and _ok(m) else "—")
        rows3.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(m)] + cells)
    rows3.append(["每年平均（全部）", "—", "—"] + [_pp(float(np.mean(list(diffs[k].values())))) for k in rules])
    rows3.append(["每年平均（拿掉 2018、2022）", "—", "—"]
                 + [_pp(float(np.mean([v for y, v in diffs[k].items() if y not in SKIP_YEARS]))) for k in rules])
    t3 = {"caption": (f"UTC 0 點日線照課本做（成本 {bc.COST * 100:.2f}%）每個日曆年的報酬，{FIRST[:4]} 年起。前兩欄是一直抱著和 50 日均線本身，"
                      "其餘是各半倉規則減 50 日均線（百分點；負數＝那年比均線少賺）。最後兩列看贏的部分是不是只來自 2018、2022 兩個大跌年。"),
          "columns": ["年份", "一直抱著", "50 日均線"] + [f"{k} 減均線" for k in rules], "rows": rows3}

    # ----- Part 2 -----
    sm = r["small_ma"]
    grid = r["grid"]
    gcount = sum(g["meets"] for g in grid)
    rows4 = []
    for g in grid:
        x = g["x"]
        rows4.append([f"{g['w']} 週", f"{g['dl']} 週", _st(x), _pct(x["c_deep_med"]), _yn(g["score_ok"]),
                      _yn(g["loss_ok"]), "過" if g["meets"] else "不過"])
    rows4.append(["50 日均線（同樣簡化算法的基準）", "—", _st(sm), _pct(sm["c_deep_med"]), "—", "—", "—"])
    t4 = {"caption": ("第二部分（A）：H_NFCI 的兩個設定往旁邊挪，和 nfci_stress 一樣的 20 種：平均的長度 26／39／52／78／104 週 × "
                      "在「至少 6 天前的那週」之外再多等 0／1／2／4 週；52 週、多等 0 週就是 H_NFCI 本身。"
                      f"為了在時間內算完，這裡一般人做法每格只模擬 {PL_SIMS} 次，50 日均線也用同樣方式重算一次再比。"
                      "過＝排名分數不低於均線，而且最深虧損中位數比均線淺。"),
          "columns": ["平均長度", "多等幾週", "排名分數", "最深虧損中位數", "排名不低於均線", "虧損比均線淺", "兩項都過"],
          "rows": rows4}
    rows5, others_ok = [], 0
    for g, lst in r["wins"].items():
        k = sum(c["meets"] for c in lst)
        others_ok += int(k >= 2)
        for i, c in enumerate(lst):
            x = c["x"]
            rows5.append([f"H_{GNAME[g]}", f"{c['w']} {WIN_UNIT[g]}" + ("（原本）" if SCALES[i] == 1.0 else ""),
                          _st(x), _pct(x["c_deep_med"]), "過" if c["meets"] else "不過",
                          f"{k}/3" if i == 0 else ""])
    for c in r["vote_sc"]:
        x = c["x"]
        rows5.append(["H_VOTE（只報告）", f"所有條件的長度 × {c['sc']:g}", _st(x), _pct(x["c_deep_med"]),
                      "過" if c["meets"] else "不過", ""])
    p2_ok = gcount >= ROBUST_N and others_ok >= ROBUST_OTHERS
    t5 = {"caption": ("第二部分（B）：其他條件的平均長度改成原本的一半、原本、兩倍（OECD 5／10／20 個月；HYG、LQD 252 個交易日的一半和兩倍；"
                      f"OFR 同樣）。同樣簡化算法、和同樣簡化的均線比。事先寫好的讀法：NFCI 20 種裡至少 {ROBUST_N} 種過，"
                      f"而且其他四個條件裡至少 {ROBUST_OTHERS} 個是「三種長度至少 2 種過」，才算「穩健」。"
                      f"這次 NFCI {gcount}/20、其他條件達到的有 {others_ok} 個 → {'穩健' if p2_ok else '不算穩健'}。"),
          "columns": ["規則", "平均長度", "排名分數", "最深虧損中位數", "兩項都過", "三種長度裡過的"], "rows": rows5}

    # ----- Part 3 -----
    sib = [k for k in ("H_OFR", "H_HYG", "H_LQD") if k in s]
    rep = {k: _meets(s[k], base) for k in sib}
    p3_ok = sum(rep.values()) >= 2
    rows6 = [["H_NFCI（被比的）", GDESC["nfci"], _st(s["H_NFCI"]), _pct(s["H_NFCI"]["c_deep_med"]),
              "過" if _meets(s["H_NFCI"], base) else "不過"]]
    for k in sib:
        rows6.append([k, GDESC[RGATE[k]], _st(s[k]), _pct(s[k]["c_deep_med"]), "過" if rep[k] else "不過"])
    wkx, sw = r["weekly"], r["sw"]
    rows6.append(["H_NFCI 改成每週決定（只報告）", f"NFCI 一公布就更新（隔天收盤起用）；換手 {sw['weekly']} 次（每月版 {sw['monthly']} 次，均線 {sw['ma50']} 次）",
                  _st(wkx), _pct(wkx["c_deep_med"]), "過" if _meets(wkx, base) else "不過"])
    t6 = {"caption": ("第三部分：換成別人編的、獨立的資料，同樣的半倉做法，效果會不會重現？「過」＝排名分數不低於均線、而且最深虧損中位數比均線淺（完整算法）。"
                      f"事先寫好：OFR、HYG、LQD 三個裡至少 2 個過才算重現。這次 {sum(rep.values())}/{len(sib)} 個 → {'重現' if p3_ok else '沒有重現'}。"
                      f"換手次數是 {FIRST} 起 UTC 0 點日線上持有比例改變的次數。"),
          "columns": ["規則", "條件", "排名分數", "最深虧損中位數", "兩項都過"], "rows": rows6}

    # ----- Part 4 -----
    vs = r["vs"]

    def triple(v) -> str:
        m_, sh, b, n = v
        return f"{m_ * 100 / n:.0f}%｜{sh * 100 / n:.0f}%｜{b * 100 / n:.0f}%"

    dyears = sorted({d.year for d in ex.dlabels})
    dy = np.array([d.year for d in ex.dlabels])
    cy = np.array([d.year for d in ex.clabels])

    def bad_years(v) -> str:
        bad = [str(y) for y in dyears if v["d_mask"][dy == y].sum() < (dy == y).sum() / 2]
        return "、".join(bad) if bad else "沒有"

    p4_keys = [(k, k) for k in rules] + [("c75", "CONST75（對照）"), ("buf", "BUF（探索）"), ("ease", "EASING（探索）")]
    rows7 = []
    for key, lab in p4_keys:
        v = vs[key]
        rows7.append([lab, f"{v['d'][2]}/{v['d'][3]}", triple(v["d"]), f"{v['c'][2]:,}/{v['c'][3]:,}", triple(v["c"]),
                      triple(v["p"]), bad_years(v)])
    t7 = {"caption": ("第四部分：老闆問的問題，直接數。每條規則和 50 日均線比，同一個起算點：最後錢比較多的比例｜最深虧損比較淺的比例｜兩個都做到的比例。"
                      f"比特幣日線 {n_d} 個起算月；{n_c:,} 格照課本做；一般人做法是 {n_c:,} 格 × {bc.SIMS} 次模擬（規則的第 i 次和均線的第 i 次配對，"
                      "兩邊漏掉哪次進場是各自抽的）。最後一欄：比特幣日線起算月裡，那一年 12 個起算月「兩個都做到」不到一半的年份。"
                      "「兩個都做到」的比例很高，就是老闆說的「賺更多又虧更少」。只報告。"),
          "columns": ["規則", f"日線兩個都做到（{n_d}）", "日線：多｜淺｜都", f"課本格兩個都做到（{n_c:,}）", "課本格：多｜淺｜都",
                      "一般人：多｜淺｜都", "做不到一半的起算年"], "rows": rows7}
    rows8 = []
    for y in dyears:
        cells = []
        for k in rules:
            v = vs[k]
            dm = v["d_mask"][dy == y]
            txt = f"{int(dm.sum())}/{len(dm)}"
            if (cy == y).any():
                txt += f"｜{float(v['c_share'][cy == y].mean()) * 100:.0f}%"
            cells.append(txt)
        rows8.append([str(y)] + cells)
    t8 = {"caption": ("第四部分（依起算年）：每一格是「日線起算月裡兩個都做到的個數／該年起算月數」｜「課本格（2018 起才有）兩個都做到的比例」。"
                      "看賺更多又虧更少是集中在哪幾年開始的人身上。"),
          "columns": ["起算年"] + rules, "rows": rows8}

    # ----- Part 5 (exploratory) -----
    ex5 = r["ex5"]
    yb = r["years"]

    def avg_wo2020(k: str) -> str:
        v = [yb[k][y] - yb["ma50"][y] for y in yb["ma50"] if y != 2020 and _ok(yb[k].get(y)) and _ok(yb["ma50"].get(y))]
        return _pp(float(np.mean(v)))

    rows9 = []
    for key, lab in (("H_NFCI", "H_NFCI（原本的條件）"), ("buf", "BUF：高出平均 0.05 以上才算不利"),
                     ("ease", "EASING：低於平均、或比 8 週前低，就算有利"), ("buf_x", "BUF，2020 年改回原本條件（＝拿掉 2020）"),
                     ("ease_x", "EASING，2020 年改回原本條件（＝拿掉 2020）")):
        x = s["H_NFCI"] if key == "H_NFCI" else ex5[key]
        y20 = yb[key].get(2020)
        rows9.append([lab, _st(x), _pct(x["c_deep_med"]), f"{x['exp'] * 100:.0f}%",
                      "過" if _meets(x, base) else "不過",
                      _pp(y20 - yb["ma50"][2020]) if _ok(y20) and _ok(yb["ma50"].get(2020)) else "—",
                      avg_wo2020(key)])
    t9 = {"caption": ("第五部分（探索，看過 2020 的結果之後才加的）：H_NFCI 條件寫法的兩個毛病是看了 2020 年逐月結果才發現的——"
                      "沒有緩衝（2020 年 1 月 NFCI 只比平均高 0.009 就被擋）、大跌後平均跟不上（2020 年 4 到 7 月情勢快速轉好卻還被擋）。"
                      "所以下面兩個寫法是照 2020 的結果設計的，不能用 2020 來評斷它們好不好；「拿掉 2020」那兩列把 2020 年的決定改回原本條件，"
                      "只看其他年份有沒有差。完整算法，沒有隨機錯位、沒有過關標準。最後兩欄：2020 年減均線、和不含 2020 的每年平均減均線（百分點）。"),
          "columns": ["規則", "排名分數", "最深虧損中位數", "持有比例", "排名不低於均線且虧損較淺", "2020 減均線", "不含 2020 每年平均減均線"],
          "rows": rows9}

    me = bc._month_ends(close)
    mret = (me / me.shift(1) - 1)
    rawn = raw["nfci"]
    avg52 = rawn.rolling(52, min_periods=52).mean()
    rows10 = []
    for held in pd.period_range("2020-01", "2020-12", freq="M"):
        dm = held - 1
        wk_ = nf._used_week(rawn, dm)
        prev = rawn.shift(EASE_W).get(wk_) if wk_ is not None else None

        def fw(F) -> str:
            v = F.get(dm, np.nan)
            return "有利" if v == 1 else "不利" if v == 0 else "—"

        rows10.append([held.strftime("%Y-%m"), _d(wk_) if wk_ is not None else "—",
                       f"{_num(rawn.get(wk_), 3)}（{_num(avg52.get(wk_), 3)}）" if wk_ is not None else "—",
                       _num(prev, 3) if _ok(prev) else "—", fw(G["nfci"]), fw(r["F5"]["buf"]), fw(r["F5"]["ease"]),
                       _p(mret.get(held))])
    t10 = {"caption": ("2020 年逐月（探索的來源）：每個持有月用的是前一個月底能用的最新一週 NFCI（至少 6 天前），括號是它自己 52 週平均；"
                       "數字越高＝金融情勢越緊。有利＝持有 100%，不利＝均線持有時只放一半。最後一欄是那個月比特幣（月底收盤到月底收盤）的漲跌。"),
           "columns": ["持有月", "用的那週", "NFCI（52 週平均）", "8 週前的 NFCI", "原本條件", "BUF", "EASING", "比特幣當月"],
           "rows": rows10}

    # ----- today -----
    last = close.index[-1]
    cur = (last + DAY).to_period("M") - 1
    pm = r["pos"]

    def amount(v) -> str:
        if not _ok(v) or v < 0.25:
            return "現金"
        return "持有" if v >= 0.75 else "半倉"

    def gword(g) -> str:
        v = G[g].get(cur, np.nan) if g in G else np.nan
        return "有利" if v == 1 else "不利" if v == 0 else "—"

    vote_pos = pm["H_VOTE"][-1] if "H_VOTE" in pm else np.nan
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": "比特幣：50 日均線｜H_NFCI｜H_VOTE",
         "value": "｜".join(amount(v) for v in (pm["ma50"][-1], pm["H_NFCI"][-1], vote_pos))},
        {"label": f"各條件（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）：NFCI｜OECD｜HYG｜LQD｜OFR",
         "value": "｜".join(gword(g) for g in GATES)},
        {"label": "其他半倉規則：H_OECD｜H_HYG｜H_LQD｜H_OFR",
         "value": "｜".join(amount(pm[k][-1]) if k in pm else "—" for k in ("H_OECD", "H_HYG", "H_LQD", "H_OFR"))},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態（持有＝100%、半倉＝50%、現金＝0%），不是買賣建議"},
    ]

    # ----- verdict -----
    word = ("站得住" if p1_holm and p2_ok and p3_ok else
            "部分站得住" if p1_raw and (p2_ok or p3_ok) else "這次沒有達到")
    vn = vs["H_NFCI"]
    best4 = max(rules, key=lambda k: vs[k]["c"][2])
    holm_luck = len(rules) * PLC_BAR
    verdict = (f"{word}（事先寫好的讀法）。量到的：第一部分六條半倉規則裡 {len(p1_raw)} 條四項都過"
               + (f"（{'、'.join(p1_raw)}）" if p1_raw else "")
               + f"，Holm 校正後仍過 {len(p1_holm)} 條" + (f"（{'、'.join(p1_holm)}）" if p1_holm else "")
               + f"；純靠運氣，隨機錯位那一項六條裡平均約 {holm_luck:.1f} 條會過。"
               f"50 日均線排名 {_share(base['low'])}、虧損中位數 {_pct(base['c_deep_med'])}；"
               + "；".join(f"{k} 排名 {_share(s[k]['low'])}、虧損 {_pct(s[k]['c_deep_med'])}、雙胞胎 {_share(r['twin'][k]['low'])}、"
                          f"隨機錯位 {_share(r['plc'][k])}" for k in rules) + "。"
               f"第二部分：NFCI 20 種設定 {gcount} 種過、其他條件 {others_ok} 個在三種長度裡至少過 2 種 → {'穩健' if p2_ok else '不算穩健'}。"
               f"第三部分：OFR、HYG、LQD {sum(rep.values())}/{len(sib)} 個重現 → {'重現' if p3_ok else '沒有重現'}。"
               f"第四部分（老闆的問題）：H_NFCI 在 {vn['d'][3]} 個日線起算月裡 {vn['d'][2]} 個「錢更多而且最深虧損更淺」，"
               f"{vn['c'][3]:,} 個課本格裡 {vn['c'][2]:,} 個（{vn['c'][2] / vn['c'][3] * 100:.0f}%），一般人做法 {vn['p'][2] / vn['p'][3] * 100:.0f}%；"
               f"六條裡課本格比例最高的是 {best4}（{vs[best4]['c'][2] / vs[best4]['c'][3] * 100:.0f}%）。"
               "第五部分是看了 2020 才加的探索，不算進判斷。推測：贏均線的部分大多來自 2018、2022 少虧（見每年表），"
               "這段歷史裡只有兩個這樣的大跌年，資料沒辦法分辨「條件真的有預警力」和「剛好碰上這兩年」。"
               "就算過關，也只是值得每天記錄、往後再看的候選，不是新的訊號。")

    tot, bad = r["look"]
    notes = [
        ("事先寫死的設計和標準（照原文，跑之前寫好）：Rules: ma off = cash; ma on = 100% when the gate is favorable, 50% when "
         "unfavorable; gates NFCI/OECD/HYG/LQD/OFR exactly as their parents; H_VOTE = 100% when >= half of the gates with a "
         "flag are favorable. Part 1 (all four): ranking >= ma50's; median deepest loss strictly shallower; placebo share "
         "<= 5% (circular shift >= ~6 months, 200 draws, 50 ordinary-person sims, real rule re-scored the same way); ranking "
         "strictly above the constant-share twin (same average exposure on ma50-on days). Holm across six rules on p = "
         "(hits+1)/(draws+1), alpha 0.05. Part 2 robust: >= 15/20 NFCI settings AND >= 2/3 windows for >= 2 other gates. "
         "Part 3 repeats: >= 2 of OFR/HYG/LQD meet ranking >= ma50 AND shallower loss. Part 4 report only. Part 5 "
         "exploratory. Overall: 站得住 = Part 1 after Holm + Part 2 + Part 3; 部分站得住 = Part 1 before Holm + Part 2 or 3."),
        ("為什麼重做：gate_design 的 D2（就是這裡的 H_NFCI、H_OECD）量到排名 84%／85%（均線 80%）、虧損中位數 -35.6%／-36.5%"
         "（均線 -53.7%）、隨機錯位 2%／2%，用專案平常的標準（和均線比）是過的；它被標成「不過」，是因為那份研究的標準是"
         "「比全有全無的版本多 3 個百分點、兩個指標都要」，那是和另一種用法比，不是老闆要問的「比主訊號賺更多又虧更少」。"
         "這份研究改用和均線比的標準，再加上兩個更難的對照（固定比例雙胞胎、5% 的隨機錯位），讓它有機會不過。"),
        ("看了結果才發現的事（照實說）：2020 年逐月檢查時看到條件寫法的兩個毛病——沒有緩衝（2020 年 1 月 NFCI 只比平均高 0.009 就被擋）、"
         "大跌之後一年平均跟不上（2020 年 4 到 7 月情勢快速轉好仍被擋）。這是看過結果之後才注意到的，所以第五部分的 BUF、EASING 只是探索，"
         "不能拿 2020 證明它們比較好，也不算進總判斷。"),
        ("固定比例雙胞胎在問什麼：半倉規則平均放得比均線少，少放本身就會讓虧損變淺、在大漲時少賺。雙胞胎平均放一樣多、但不挑時間；"
         "半倉規則要比它好，才表示「什麼時候放半倉」有用，而不只是「放得比較少」。和 vol_sizing 研究的「永遠只放固定比例」是同一個想法。"),
        ("第四部分怎麼讀：「錢更多」＝從起算日到現在的每年報酬比均線高；「虧損更淺」＝這段期間最深的跌幅比均線淺。"
         "相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"
         "一般人做法兩邊漏掉哪次進場是各自抽的，所以那一欄多了一點隨機。"),
        ("限制：(1) 五個條件都是今天的修正後版本（NFCI、OECD、OFR 都會改過去的數字），可能比當時真的能做到的好看；"
         "(2) 原本兩份研究的優勢主要來自 2018、2022 年站在旁邊，這段歷史只有這兩個大跌年；"
         "(3) 五個條件同屬「金融情勢／景氣」一家（信用利差、波動、資金市場互相重疊），不是五個獨立的測試，"
         "H_VOTE 又是它們的組合；(4) 其他幣、MSTR 用同一個美國條件；(5) 第二部分的 20 種設定彼此高度重疊。"
         "「沒有看到」只表示用這個定義和這些資料沒有看到，不表示不存在。"),
        (f"資料：NFCI 來自{data['where']['nfci']}；OECD 來自{data['where']['oecd']}；OFR 來自{data['where']['ofr'] or '這次拿不到'}；"
         "HYG、IEF、LQD、SPY 來自 Yahoo Finance 公開日線（ETF 用含息調整後收盤）。"
         f"重算檢查：這裡為了留下每個起算點的數字把考試重寫了一次，50 日均線和 D0 NFCI 的結果和 business_cycle 的考試"
         f"{'完全一樣' if r['same'] else '不一樣（要查）'}。偷看未來的檢查：三個月底之後把那時還不能用的條件數字換成亂數，"
         f"切點以前的每月條件 {tot} 組裡 {tot - bad} 組完全沒變" + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        (f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子；"
         "NFCI、OECD 的隨機錯位和 gate_design 用同一串平移量。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒（第一部分完整考試 {r['t1']:.0f} 秒、隨機錯位 {r['t_plc']:.0f} 秒）。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": nf.CHI_URL} if data["where"]["nfci"] != "FRED"
        else {"name": "FRED：NFCI", "url": nf.FRED_URL},
        ({"name": "FRED：OECD 美國綜合領先指標", "url": bc.FRED_CSV.format(bc.FRED_ID["oecd"])} if data["where"]["oecd"] == "FRED"
         else {"name": "OECD 綜合領先指標（SDMX）", "url": bc.OECD_URL}),
        {"name": "OFR 金融壓力指數資料（CSV）", "url": nfs.OFR_URL},
        {"name": "OFR 金融壓力指數說明頁", "url": nfs.OFR_PAGE},
    ] + [{"name": f"Yahoo Finance 公開日線：{sym}", "url": ro.YAHOO.format(sym)} for sym in YAHOO_SYMS] + [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]

    return {
        "id": "half_gate",
        "title": "條件不利時放一半：賺更多又虧更少嗎",
        "question": ("50 日均線說持有、但金融情勢或景氣條件不利時，不是全部換現金，而是只放一半。"
                     "和 50 日均線本身比，這樣能不能在大多數起算點「錢更多、最深虧損也更淺」？"
                     "而且是「什麼時候放一半」有用，還是只因為平均放得比較少？換設定、換資料還在不在？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t2, t1, t3, t4, t5, t6, t7, t8, t9, t10],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
