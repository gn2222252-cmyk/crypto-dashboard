"""Historical study "gate_design": can a different way of USING the two macro gates that passed the fair exam keep
most of their loss protection while giving up less in strong years?

Background: N1 (nfci_filter: ma50 AND Chicago Fed NFCI below its own 52-week average) and O2 (oecd_followup: ma50 AND
OECD US leading indicator above its 10-month average, value two months old) both passed. They sit out most of 2018
and 2022 but give up a lot in strong years (2020, 2015/2016/2023), so their ranking score is only a little above
ma50's. The gate indicators, thresholds, data and monthly timing are taken unchanged from those two studies
(nfci_filter._series/_flags with "avg52"; business_cycle._series("oecd")/_flags with LAG 2); only the way the gate
is combined with ma50 changes.

Pre-registered (written before this study was run; no other designs, no changes after seeing results).
  Notation: ma = main_signal_check's ma50 state after each close (1 on / 0 off); g = the gate state in effect that
  day (decided at the last daily close of the previous month exactly as in the parent studies; unknown = unfavorable,
  as the parents treat unknown as cash). Each single-gate design is run with the NFCI gate and with the OECD gate.
    D0 (reference, = N1 / O2): position = ma * g.
    D1 entry-only: the gate only blocks NEW entries. position(t) = ma(t) AND (position(t-1) OR g(t)): when ma turns
        on while g is unfavorable, stay in cash; once held, keep it until ma itself turns off; if ma is on, flat and
        g turns favorable at a month-end decision, enter then.
    D2 half size: position = ma * (0.5 + 0.5 g) - 50% instead of 0% when ma is on and g is unfavorable.
    D3 OR-logic ("growth-trend timing", Philosophical Economics): hold unless BOTH ma is off AND g is unfavorable,
        position = max(ma, g). Holds more than ma50, so deeper losses are expected; reported as is.
    D4 two-gate vote (one design): cash only when ma is on but BOTH the NFCI and the OECD gate are unfavorable,
        position = ma * max(g_NFCI, g_OECD).
  Scoring: business_cycle._Exam unchanged (main_signal_check's four checks: BTC UTC daily 108 monthly starts
  2015-01..2023-12; 72 monthly starts x 24 closing hours = 1,728 textbook cells; ordinary person = trade 10 hours
  later, 0.25% cost, 10% of entries missed, 200 simulations per clock with the same seeds as the parents (kind "m2");
  other assets ETH, SOL, BNB, XRP, MSTR with their own ma50 and the same US gate; ranking score = lowest of the four
  "beats holding" shares, eligible only when the deepest loss is shallower than holding's in >= 95% of BTC starts and
  cells). The exam's position function is swapped for the design's while scoring; the engine, the ordinary-person
  miss model (rt._missed: a rise of the wanted share by more than 10% is an entry that can be missed) and te._run all
  take fractional positions, so D2 is scored the same way, not approximated.
  Report for each design and gate: ranking score and its four parts, median and worst deepest loss over the 1,728
  cells, average share held on the BTC daily series from 2015-01-01 (time in market; D2's half days count 0.5),
  calendar-year return minus ma50 for every year from 2015, days differing from D0 and what BTC did on them, and each
  gate's favorable-vs-unfavorable next-month BTC return (business_cycle._events; the gate itself is unchanged).
  Bar for calling a design "better than the plain AND-gate" (all four, and for BOTH gates; D4 once, against BOTH D0s):
    (1) ranking score >= D0's (same gate) + 3 percentage points (a design that loses the 95% eligibility while D0
        keeps it fails);
    (2) median deepest loss no worse than ma50's;
    (3) median deepest loss no more than 8 percentage points worse than D0's (same gate);
    (4) placebo share <= 10%: the gate's raw series circularly shifted exactly as in the parent (NFCI weekly by 26 ...
        length-26 weeks, seed SEED+41; OECD monthly by 6 ... length-6 months, seed SEED+31; 200 draws; for D4 the two
        series are shifted independently, draw i of each stream), flags rebuilt, the design re-scored with the full
        ranking score and the ordinary-person check at 50 simulations per clock; the real design re-scored the same
        way; share of draws scoring at least as high (business_cycle._score_ge).
  Seven design-by-gate rules (D1-D3 x 2 + D4) are tried: multiple-testing caveat stated; any winner is only a
  candidate for the daily shadow log, not a new signal. Look-ahead check: prices and gate decisions not yet known are
  scrambled after three month-ends; no position decided up to then may change. Today: states only.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf

LAG_O = bc.LAG["oecd"]
DRAWS, PL_SIMS = bc.DRAWS, bc.PL_SIMS
GAP_N, GAP_O = nf.GAP_W, bc.GAP
SEED_N, SEED_O = bc.SEED + 41, bc.SEED + 31   # the parents' placebo streams
FIRST = msc.DAILY_STARTS[0]
SCORE_GAIN, LOSS_SLACK, PLC_BAR = 0.03, 8.0, 0.10

SINGLE = ("D0", "D1", "D2", "D3")
GATES = ("n", "o")
GNAME = {"n": "NFCI", "o": "OECD"}
DNAME = {"D0": "D0 原本的「而且」（不利就現金）", "D1": "D1 只擋新進場", "D2": "D2 不利時只放一半",
         "D3": "D3 「或」：兩個都壞才現金", "D4": "D4 兩個指標都不利才現金"}
KEYS = [f"{d}{g}" for g in GATES for d in SINGLE] + ["D4"]
TESTED = [k for k in KEYS if not k.startswith("D0")]

PE_URL = "https://www.philosophicaleconomics.com/?p=7644"
NF_URL = "https://blog.thinknewfound.com/2019/10/macro-timing-with-trend-following/"

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


def _label(key: str) -> str:
    return "D4（NFCI＋OECD）" if key == "D4" else f"{key[:2]} {GNAME[key[2]]}"


# ---------- data ----------

def _load() -> dict:
    nfci, where_n, problems = nf._series()
    oecd, where_o, note = bc._series("oecd")
    if note:
        problems.append(note)
    other, more = msc._load_other()
    return {"close": bc.te._daily_close(), "nfci": nfci, "oecd": oecd, "where": {"n": where_n, "o": where_o},
            "other": other, "problems": problems + more, "fred_down": bc._RUN["fred_down"]}


def _gates(nfci: pd.Series, oecd: pd.Series, last: pd.Period) -> pd.DataFrame:
    """Monthly gate flags (decision month -> 1 favorable / 0 / NaN), exactly as the parent studies build them."""
    months = pd.period_range(nfci.index[0].to_period("M"), last, freq="M")
    fn = nf._flags(nfci, "avg52", months)
    fo = bc._flags(oecd, LAG_O)
    return pd.concat({"n": fn, "o": fo}, axis=1).sort_index()


# ---------- the designs ----------

def _combine(design: str, ma, g) -> np.ndarray:
    """Position after each close from the ma50 state and the gate state(s) in effect (g: (days, k), unknown = 0)."""
    ma = np.asarray(ma, dtype=float)
    unknown = ~np.isfinite(ma)
    m = np.nan_to_num(ma, nan=0.0)
    g = np.asarray(g, dtype=float).reshape(len(m), -1)
    gate = g.max(axis=1) if design == "D4" else g[:, 0]
    if design in ("D0", "D4"):
        p = m * gate
    elif design == "D2":
        p = m * (0.5 + 0.5 * gate)
    elif design == "D3":
        p = np.maximum(m, gate)
    elif design == "D1":
        on = m > 0.5
        run = np.cumsum(~on)  # every off day starts a new run; on days share the id of the off day before them
        seen = pd.Series(np.where(on, gate, 0.0)).groupby(run).cummax().values
        p = np.where(on, seen, 0.0)
    else:
        raise ValueError(design)
    return np.where(unknown, np.nan, p)


_CUR = {"design": None}


def _design_pos(kind: str, ma, macro):
    if kind == "ma50":
        return ma
    return _combine(_CUR["design"], ma, macro)


@contextmanager
def _use(design: str):
    """Score a design with business_cycle's exam unchanged: only its position function is swapped (kind "m2" keeps
    the parents' ordinary-person seeds)."""
    old = bc._pos
    bc._pos, _CUR["design"] = _design_pos, design
    try:
        yield
    finally:
        bc._pos, _CUR["design"] = old, None


def _cols(key: str) -> list[str]:
    return ["n", "o"] if key == "D4" else [key[2]]


def _score(ex: bc._Exam, key: str, G: pd.DataFrame, sims: int = bc.SIMS) -> dict:
    with _use(key[:2]):
        return ex.summary("m2", G[_cols(key)], sims)


def _btc_pos(ex: bc._Exam, key: str, G: pd.DataFrame) -> np.ndarray:
    with _use(key[:2]):
        return ex.btc_pos("m2", G[_cols(key)])


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: bc._Exam, G: pd.DataFrame) -> tuple[int, int]:
    rng = np.random.default_rng(17)
    close = data["close"]
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        G2 = G.copy()
        later = G2.index > c.to_period("M")
        G2.loc[later] = rng.integers(0, 2, size=(int(later.sum()), G2.shape[1])).astype(float)
        upto = close.index <= c
        for key in KEYS:
            cols = _cols(key)
            a = _combine(key[:2], ex.btc_ma, bc._on_days(G[cols], close.index + bc.DAY))
            b = _combine(key[:2], ma2, bc._on_days(G2[cols], close.index + bc.DAY))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, nfci, oecd = data["close"], data["nfci"], data["oecd"]
    last = close.index[-1].to_period("M")
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    G = _gates(nfci, oecd, last)

    s = {"ma50": ex.summary("ma50", None)}
    for k in KEYS:
        s[k] = _score(ex, k, G)
    small = {k: _score(ex, k, G, PL_SIMS) for k in KEYS}
    t_full = time.time() - t0

    # placebo: draw i shifts NFCI by the i-th shift of the nfci_filter stream and OECD by the i-th of oecd_followup's
    rn, ro = np.random.default_rng(SEED_N), np.random.default_rng(SEED_O)
    sh_n = rn.integers(GAP_N, len(nfci) - GAP_N + 1, DRAWS)
    sh_o = ro.integers(GAP_O, len(oecd) - GAP_O + 1, DRAWS)
    hits = {k: 0 for k in TESTED}
    lows = {k: [] for k in TESTED}
    for a, b in zip(sh_n, sh_o):
        n2 = pd.Series(np.roll(nfci.values, int(a)), index=nfci.index)
        o2 = pd.Series(np.roll(oecd.values, int(b)), index=oecd.index)
        G2 = _gates(n2, o2, last)
        for k in TESTED:  # D0's placebo is the parents' (same shifts and seeds); not re-run, to stay within time
            x = _score(ex, k, G2, PL_SIMS)
            lows[k].append(x["low"])
            hits[k] += int(bc._score_ge(x, small[k]))
    plc = {k: hits[k] / DRAWS for k in TESTED}
    plc_low = {k: float(np.median(lows[k])) for k in TESTED}

    pos = {"ma50": ex.btc_pos("ma50", None), **{k: _btc_pos(ex, k, G) for k in KEYS}}
    a = close.index >= pd.Timestamp(FIRST)
    tim = {k: float(np.nan_to_num(pos[k][a]).mean()) for k in pos}
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST), **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    # days that differ from D0 of the same gate (D4: from D0 NFCI) and what bitcoin did on them (next-day move)
    nxt = close.pct_change().shift(-1).fillna(0.0).values
    differ = {}
    for k in TESTED:
        ref = np.nan_to_num(pos["D0n" if k == "D4" else f"D0{k[2]}"][a])
        cur = np.nan_to_num(pos[k][a])
        more_ = cur > ref + 1e-9
        differ[k] = (int(more_.sum()), int((cur < ref - 1e-9).sum()), int(a.sum()),
                     float((np.prod(1 + nxt[a][more_]) - 1) * 100) if more_.any() else None)
    rng = np.random.default_rng(bc.SEED + 5)
    ev = {g: bc._events(close, G[g].dropna(), rng) for g in GATES}
    look = _lookahead(data, ex, G)
    return {"ex": ex, "G": G, "s": s, "small": small, "plc": plc, "plc_low": plc_low, "pos": pos, "tim": tim,
            "years": years, "differ": differ, "ev": ev, "look": look, "problem": problem,
            "t_full": t_full, "secs": time.time() - t0}


def _bar(s: dict, plc: dict, key: str) -> dict:
    base = s["ma50"]
    refs = ["D0n", "D0o"] if key == "D4" else [f"D0{key[2]}"]
    x = s[key]

    def gain(ref):
        r = s[ref]
        if r["eligible"] and not x["eligible"]:
            return False
        return x["low"] >= r["low"] + SCORE_GAIN - 1e-12

    q = {"score": all(gain(r) for r in refs),
         "vs_ma": x["c_deep_med"] >= base["c_deep_med"] - 1e-12,
         "vs_d0": all(x["c_deep_med"] >= s[r]["c_deep_med"] - LOSS_SLACK - 1e-12 for r in refs),
         "plc": plc[key] <= PLC_BAR}
    q["all"] = all(q.values())
    return q


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, G = data["close"], r["ex"], r["s"], r["G"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base = s["ma50"]
    n_c, n_d = base["c_n"], base["d_n"]
    bars = {k: _bar(s, r["plc"], k) for k in TESTED}
    design_ok = {d: all(bars[f"{d}{g}"]["all"] for g in GATES) for d in ("D1", "D2", "D3")}
    design_ok["D4"] = bars["D4"]["all"]
    winners = [d for d in ("D1", "D2", "D3", "D4") if design_ok[d]]

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    # ----- table 1: fair exam -----
    rows1 = []
    for k in ["ma50"] + KEYS:
        x = s[k]
        ck = x["checks"]
        rows1.append(["50 日均線（基準）" if k == "ma50" else _label(k),
                      _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]),
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", f"{r['tim'][k] * 100:.0f}%",
                      f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    t1 = {"caption": (f"main_signal_check 的公平考試（前四欄都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產 ETH、SOL、BNB、XRP、MSTR "
                      "用自己的 50 日均線、同一個美國指標。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      f"最深虧損是 {n_c:,} 格照課本做的中位數｜最差一格；持有比例是 {FIRST[:4]} 年起比特幣日線平均放了幾成（D2 的半倉算 0.5）。"
                      "D0 就是原本兩份研究的 N1、O2。"),
          "columns": ["規則", f"比特幣起算月（{n_d}）", f"起算月×收盤時間（{n_c:,}）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損：中位數｜最差", "持有比例", "每年多賺中位數：課本｜一般人"],
          "rows": rows1}

    # ----- table 2: the bar -----
    rows2 = []
    for k in TESTED:
        q, x = bars[k], s[k]
        refs = ["D0n", "D0o"] if k == "D4" else [f"D0{k[2]}"]
        rows2.append([_label(k),
                      f"{_share(x['low'])} 對 " + "、".join(_share(s[c]['low']) for c in refs) + f" → {yes(q['score'])}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])} → {yes(q['vs_ma'])}",
                      f"{_pct(x['c_deep_med'])} 對 " + "、".join(_pct(s[c]['c_deep_med']) for c in refs) + f" → {yes(q['vs_d0'])}",
                      f"{_share(r['plc'][k])}（打亂後中位數 {_share(r['plc_low'][k])}，真的 {_share(r['small'][k]['low'])}） → {yes(q['plc'])}",
                      "四項都過" if q["all"] else "沒有全過"])
    t2 = {"caption": ("事先寫好的標準，四項都要過、而且 NFCI 和 OECD 兩個版本都要過（D4 只有一個版本，要同時對兩個 D0 都過）："
                      f"(1) 排名分數比同一個指標的 D0 高至少 {SCORE_GAIN * 100:.0f} 個百分點；(2) 最深虧損中位數不比 50 日均線深；"
                      f"(3) 最深虧損中位數比 D0 深不超過 {LOSS_SLACK:.0f} 個百分點；(4) 隨機錯位一樣好或更好的比例 ≤ 10%。"
                      f"隨機錯位：NFCI 每週數字循環平移 {GAP_N} 週到「長度 − {GAP_N}」週、OECD 每月數字平移 {GAP_O} 到「長度 − {GAP_O}」個月，"
                      f"{DRAWS} 次（平移量和原本兩份研究完全相同；D4 兩個指標各自平移），重算每月訊號、整套考試重做；這一步一般人做法每個收盤時間只模擬 "
                      f"{PL_SIMS} 次，真的規則也用同樣方式算一次再比（簡化版）。D0 自己的隨機錯位就是原本兩份研究算過的（平移量和種子相同），這裡不重算。"),
          "columns": ["設計", "(1) 排名分數：設計 對 D0", "(2) 最深虧損中位數 對 均線", "(3) 最深虧損中位數 對 D0", "(4) 隨機錯位", "結果"],
          "rows": rows2}

    # ----- table 3: years -----
    yrs = sorted(r["years"]["ma50"])
    rows3 = []
    for y in yrs:
        m = r["years"]["ma50"].get(y)
        cells = []
        for k in KEYS:
            v = r["years"][k].get(y)
            cells.append(_pp(v - m) if _ok(v) and _ok(m) else "—")
        rows3.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(m)] + cells)
    t3 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起。前兩欄是一直抱著和 50 日均線本身的報酬，"
                      "其餘是各設計減 50 日均線（百分點；負數＝那年比均線少賺）。看 2018、2022 少虧的部分留下多少、2020、2023 少賺的部分補回多少。"),
          "columns": ["年份", "一直抱著", "50 日均線"] + [f"{_label(k)} 減均線" for k in KEYS], "rows": rows3}

    # ----- table 4: where they differ + gate itself -----
    rows4 = []
    for k in TESTED:
        more_, less, nn, ret = r["differ"][k]
        rows4.append([_label(k), f"{more_:,} 天", f"{less:,} 天", f"{nn:,} 天", _p(ret / 100) if _ok(ret) else "—"])
    t4 = {"caption": (f"和同一個指標的 D0 比（D4 和 D0 NFCI 比），{FIRST} 起的比特幣日線上，設計多放了幾天、少放了幾天（D2 的半倉也算「多放」），"
                      "以及多放那些天比特幣本身連乘起來漲跌多少（不含成本；正數＝多放的那些天整體是賺的）。"),
          "columns": ["設計", "比 D0 多放的天數", "比 D0 少放的天數", "全部天數", "多放那些天比特幣連乘漲跌"],
          "rows": rows4}
    rows5 = []
    for g in GATES:
        e = r["ev"][g]
        a = e["all"]
        ci = a.get("ci")
        rows5.append([GNAME[g], f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()])
    t5 = {"caption": ("指標本身沒有改，這裡只是重列：月底指標有利 / 不利時，下一個月比特幣報酬（月底收盤到月底收盤）的平均，"
                      f"括號是區塊拔靴法（連續 {bc.BLOCK} 個月一塊、{bc.BOOT:,} 次）的 90% 區間；分段欄括號是（有利｜不利）月數。"),
          "columns": ["指標", "有利月：個數｜平均", "不利月：個數｜平均", "差距（90% 區間）"] + [f"{lab} 差距" for lab, _, _ in bc.PERIODS],
          "rows": rows5}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    def gword(g) -> str:
        v = G[g].get(cur, np.nan)
        return "有利" if v == 1 else "不利" if v == 0 else "—"

    def amount(v) -> str:
        if not _ok(v) or v < 0.25:
            return "現金"
        return "持有" if v >= 0.75 else "半倉"

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"兩個指標（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": f"NFCI {gword('n')}｜OECD {gword('o')}"},
        {"label": "比特幣：50 日均線｜D1 NFCI｜D1 OECD｜D4",
         "value": "｜".join(word(pm[k][-1]) for k in ("ma50", "D1n", "D1o", "D4"))},
        {"label": "其他設計：D0 NFCI｜D2 NFCI｜D3 NFCI｜D0 OECD｜D2 OECD｜D3 OECD",
         "value": "｜".join(amount(pm[k][-1]) for k in ("D0n", "D2n", "D3n", "D0o", "D2o", "D3o"))},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    def brief(k: str) -> str:
        x = s[k]
        return f"{_label(k)} 排名 {_share(x['low'])}、虧損中位數 {_pct(x['c_deep_med'])}、持有 {r['tim'][k] * 100:.0f}%"

    nwin = len(winners)
    verdict = (f"D1 到 D4 四種用法裡，{'沒有一種' if nwin == 0 else f'有 {nwin} 種（' + '、'.join(winners) + '）'}"
               "達到事先寫好的「比原本的而且更好」標準。"
               + "".join(f"{_label(f'{d}{g}')} 單獨四項都過，但另一個指標的版本沒過，所以 {d} 不算過關。"
                         for d in ("D1", "D2", "D3") for g in GATES
                         if bars[f"{d}{g}"]["all"] and not design_ok[d])
               + f"基準：50 日均線排名 {_share(base['low'])}、虧損中位數 {_pct(base['c_deep_med'])}；"
               + "；".join(brief(k) for k in KEYS) + "。"
               + "".join(f"{_label(k)} 沒過的是：{'、'.join(lab for lab, key in (('排名分數沒多 3 點', 'score'), ('虧損比均線深', 'vs_ma'), ('虧損比 D0 深超過 8 點', 'vs_d0'), ('隨機錯位', 'plc')) if not bars[k][key])}。"
                         for k in TESTED if not bars[k]["all"])
               + "一共試了七種設計×指標組合，就算有一種過關也可能是碰巧；過關的只能放進每天的影子紀錄當候選，不是新的訊號。")

    tot, bad = r["look"]
    notes = [
        ("事先寫死的設計和標準（照原文，跑之前寫好）：D0 = ma * g (N1 / O2). D1 entry-only: position(t) = ma(t) AND "
         "(position(t-1) OR g(t)). D2: ma * (0.5 + 0.5 g). D3 OR-logic: max(ma, g). D4: ma * max(g_NFCI, g_OECD). Unknown "
         "gate = unfavorable. Bar (all four, for BOTH gates; D4 against both D0s): ranking score >= D0 + 3 points; median "
         "deepest loss no worse than ma50; median deepest loss no more than 8 points worse than D0; placebo share <= 10% "
         "(parents' circular shifts and seeds, 200 draws, full ranking score with 50 ordinary-person simulations per clock, "
         "real design re-scored the same way)."),
        ("「或」的來源：Philosophical Economics〈In Search of the Perfect Recession Indicator〉（2016-02-21）介紹的「成長＋趨勢」做法："
         "美國失業率高於自己 12 個月平均（景氣轉壞）而且標普 500 低於 10 個月均線時才換現金，其他時候都持有；作者說 1930 到 2016 年"
         "比單純均線每年多約 1.5 個百分點、保留均線的下跌保護，理由是避開景氣好時均線的來回假訊號；但他也說用各國自己的經濟數字去做外國股市，"
         "反而輸給單純均線。Newfound Research（2019-10）重做後發現：失業率版本的夏普值只從 0.62 到 0.66、波動和最大跌幅反而較大，"
         "換一種趨勢做法優勢大致消失，可能只是剛好那一種設定。這裡的 D3 是把同一個想法套到比特幣，用 NFCI、OECD 代替失業率。"),
        ("和原本研究的關係：NFCI、OECD 的資料、門檻、每月決定時間、0 點收盤都直接用 nfci_filter 和 oecd_followup 的程式；考試直接用 business_cycle "
         "的 _Exam，只把「怎麼把指標和均線合起來」換掉，所以 D0 的數字就是 N1、O2 的數字（一般人做法的亂數種子也一樣）。"
         "半倉（D2）也照同一套算：引擎本來就接受 0 到 1 之間的部位，一般人「漏掉進場」是看想要的比例一天增加超過一成。"),
        (f"資料來源：NFCI 來自{data['where']['n']}，OECD 來自{data['where']['o']}。兩個指標都會修正過去的數字，這裡只有今天的版本（原本兩份研究已說明）。"
         f"偷看未來的檢查：三個月底之後把價格和那時還不能用的指標決定換成亂數，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("多重比較：D1、D2、D3 各用兩個指標、加上 D4，一共七種組合都是在同一段歷史上比，而且 N1、O2 本身也是之前挑出來的；"
         "所以就算有一種剛好過關，也只能當作要每天記錄、之後再看的候選。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，"
         f"「幾個裡幾個」不是互相獨立的證據。成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒（完整考試 {r['t_full']:.0f} 秒，其餘是隨機錯位）。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的幾種用法照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "gate_design",
        "title": "金融情勢、景氣指標換個用法",
        "question": ("50 日均線再加上「金融情勢不緊」（NFCI）或「景氣領先指標向上」（OECD）的條件，能避開 2018、2022 的大跌，"
                     "但在 2020、2023 這種大漲年少賺很多。指標本身不改，只改用法：只擋新進場、不利時放一半、兩個都壞才出場、"
                     "兩個指標都不利才出場，能不能保住大部分的少虧、又少放掉一些大漲？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5],
        "notes": notes,
        "sources": [
            {"name": "Philosophical Economics：In Search of the Perfect Recession Indicator（2016-02，成長＋趨勢做法）", "url": PE_URL},
            {"name": "Newfound Research：Macro Timing with Trend Following（2019-10）", "url": NF_URL},
            {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": nf.CHI_URL},
            {"name": "FRED：NFCI（備用）", "url": nf.FRED_URL},
            {"name": "OECD 綜合領先指標（SDMX）", "url": bc.OECD_URL},
            {"name": "FRED：OECD 美國綜合領先指標", "url": bc.FRED_CSV.format(bc.FRED_ID["oecd"])},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
