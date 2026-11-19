"""Historical study "canary_breadth": do two published "canary" gates from tactical asset allocation improve the
50-day moving average rule (ma50, main_signal_check's main signal) for bitcoin?

Ideas under test (nobody has published them on BTC):
  (1) Real-yield canary. Keller & Keuning, Hybrid Asset Allocation (HAA, SSRN 4346906; blog post 2023-02): a single
      canary, TIP (US Treasury inflation-protected bonds). At the close of the last trading day of each month, if
      TIP's 13612U momentum (unweighted average of its 1-, 3-, 6- and 12-month total returns) is <= 0, HAA goes fully
      defensive for the next month. The authors want a canary that reacts to rising yields and rising expected
      inflation.
  (2) Cross-asset trend breadth. Faber, "A Quantitative Approach to Tactical Asset Allocation" (2006): each asset
      class is held only while its month-end price is above its 10-month simple average. Keller & Keuning's
      Defensive Asset Allocation (DAA, SSRN 3212862) uses VWO (emerging-market stocks) and BND (total US bond market)
      as a canary pair with breadth B = 2: both bad -> 100% cash.
  This is ABSOLUTE trend of whole asset classes, decided monthly. cross_market tested DAILY filters on US stocks only
  (QQQ/SPY 50/100/200-day averages, drawdowns), VIX levels, 10-year yield jumps and the dollar index; stress_family
  (another study) tests ratios (HYG/IEF, copper/gold, IWM/SPY ...). No ratio is used here.

Pre-registered (written before this study was run; no other variants, no parameter changes afterwards):
  Data: Yahoo Finance adjusted daily closes (payouts reinvested) through trend_world._fetch (public chart endpoint,
  fixed user agent "btc-research-dashboard", full history), last good download kept in common.STORE
  (canary_breadth_closes.json) as fallback. Tickers: TIP; SPY, EFA, EEM, IEF, GLD, DBC, VNQ; VWO, BND. A ticker that
  cannot be loaded is dropped and reported, never substituted (for B1 the count is then over the available ones,
  threshold still 4).
  Timing: month-end closes = the last US close dated inside calendar month M. The BTC daily bar dated the last day of
  M closes at 00:00 UTC of the 1st of M+1, three to four hours after that US close, so the US close is strictly before
  the decision. Decision at that BTC close (te._daily_close, business_cycle's monthly timing), held during M+1.
  Momentum r_k = month-end close / month-end close k months earlier - 1.
    13612U = (r1 + r3 + r6 + r12) / 4;  13612W = (12 r1 + 4 r3 + 2 r6 + r12) / 4.
  10-month average = mean of the last 10 month-end closes including the current one.
  T1 (main): hold BTC when ma50 is on AND TIP 13612U > 0.
  T2 (robustness, one row, no placebo): ma50 on AND TIP month-end close > its 10-month average.
  B1 (main): hold BTC when ma50 is on AND at least 4 of SPY, EFA, EEM, IEF, GLD, DBC, VNQ have month-end close >
      their own 10-month average.
  B2 (robustness, one row, no placebo): ma50 on AND VWO 13612W > 0 AND BND 13612W > 0.
  Pass test for T1 and B1 separately, all four (exactly nfci_filter's):
    (1) favorable > unfavorable average next-month BTC return in 2011-16, 2017-21 and 2022-now (business_cycle's
        event table, holding months from 2011-07);
    (2) ranking score >= ma50's (main_signal_check's four checks via business_cycle._Exam, incl. the 95% loss bar);
    (3) median deepest loss over the textbook cells no worse than ma50's;
    (4) placebo share <= 10%: the monthly gate series circularly shifted by 12 ... (length - 12) months, 200 draws,
        whole exam re-scored with the ordinary-person check at 50 simulations per clock (the real rule re-scored the
        same way).
  Two main rules are tested: one passing alone is weaker evidence than one rule passing alone would be.
  Report only: calendar-year return minus ma50 from 2015 for T1 and B1; agreement of T1 / B1 monthly flags with
  nfci_filter's N1 flag (share of months equal, phi correlation); who moves first: (a) every flip of BTC's own
  10-month trend (month-end close vs its 10-month average) matched with the nearest same-direction flip of the T1 /
  B1 flag within +/- 6 months, (b) correlation of the monthly change of the breadth count (month t) with BTC's
  monthly log return (month t+k), k = -6 ... +6, from 2011-01. Look-ahead check: scramble prices dated after three
  month-ends; no position up to then may change. Today: states only, row "比特幣：50 日均線｜T1｜B1".
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import trend_world as tw

AVG_M = 10
NEED = 4
GAP_M = 12
DRAWS, PL_SIMS = bc.DRAWS, bc.PL_SIMS
LAGS = range(-6, 7)
WIN = 6
LL_FROM = pd.Period("2011-01", "M")
FIRST = msc.DAILY_STARTS[0]
STORE_PATH = common.STORE / "canary_breadth_closes.json"

TIP = "TIP"
BREADTH = ("SPY", "EFA", "EEM", "IEF", "GLD", "DBC", "VNQ")
DAA = ("VWO", "BND")
ALL = (TIP,) + BREADTH + DAA
GLOSS = {"TIP": "TIP（美國抗通膨公債基金）", "SPY": "SPY（美國大型股）", "EFA": "EFA（美國以外的已開發國家股票）",
         "EEM": "EEM（新興市場股票）", "IEF": "IEF（7 到 10 年期美國公債）", "GLD": "GLD（黃金）",
         "DBC": "DBC（一籃子原物料期貨）", "VNQ": "VNQ（美國不動產投資信託）", "VWO": "VWO（另一檔新興市場股票基金）",
         "BND": "BND（美國整體債券市場）"}
NAMES = {"t1": "T1：50 日均線而且抗通膨公債動能為正（主要）", "t2": "T2：50 日均線而且抗通膨公債高於 10 個月平均（只報告）",
         "b1": "B1：50 日均線而且 7 類資產至少 4 類高於 10 個月平均（主要）",
         "b2": "B2：50 日均線而且新興股市和整體債券動能都為正（只報告）"}
GATE = {"t1": "T1：抗通膨公債 13612U 動能為正（主要）", "t2": "T2：抗通膨公債高於 10 個月平均（只報告）",
        "b1": "B1：7 類資產至少 4 類高於 10 個月平均（主要）", "b2": "B2：新興股市 VWO 和整體債券 BND 的 13612W 動能都為正（只報告）"}
SHORT = {"t1": "T1", "t2": "T2", "b1": "B1", "b2": "B2"}
HAA_URL = "https://indexswingtrader.blogspot.com/2023/02/introducing-hybrid-asset-allocation-haa.html"
CXO_URL = "https://www.cxoadvisory.com/strategic-allocation/multi-class-momentum-portfolio-with-canary-crash-protection/"
FABER_URL = "https://mebfaber.com/timing-model/"

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _read_store() -> dict:
    try:
        x = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except Exception:  # noqa: BLE001 - first run or damaged file
        return {}


def _closes() -> tuple[dict, list[str], dict]:
    stored, fresh, data, notes, info = _read_store(), {}, {}, [], {}
    for sym in ALL:
        try:
            s = tw._fetch(sym)
            fresh[sym] = {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "close": [float(f"{v:.8g}") for v in s.values]}
            info[sym] = "Yahoo"
        except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the others
            old = stored.get(sym)
            if isinstance(old, dict) and old.get("dates"):
                s = pd.Series(old["close"], index=pd.to_datetime(old["dates"]), dtype=float)
                fresh[sym] = old
                info[sym] = "存檔"
                notes.append(f"{GLOSS[sym]}這次從 Yahoo 取不到（{type(exc).__name__}），用上次存下的資料（到 {old['dates'][-1]}）。")
            else:
                notes.append(f"{GLOSS[sym]}這次取不到（{type(exc).__name__}），沒有算進去，也沒有用別的代替。")
                continue
        data[sym] = s
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(json.dumps(fresh, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store is only a fallback
        pass
    return data, notes, info


def _monthly(s: pd.Series) -> pd.Series:
    """Last close dated inside each finished calendar month (PeriodIndex)."""
    m = s.groupby(s.index.to_period("M")).last()
    return m[m.index < pd.Timestamp(common.today_utc()).to_period("M")]


def _load() -> dict:
    daily, problems, info = _closes()
    if TIP not in daily:
        raise RuntimeError("TIP 取不到，主要規則 T1 沒辦法算")
    me = {k: _monthly(v) for k, v in daily.items()}
    other, more = msc._load_other()
    nf_raw, nf_note = None, []
    try:
        from dashboard.studies2 import nfci_filter as nf
        nf_raw, _, nf_note = nf._series()
    except Exception as exc:  # noqa: BLE001 - the NFCI comparison is report only
        nf_note = [f"NFCI 這次取不到（{type(exc).__name__}），和 NFCI 的比較沒有算。"]
    return {"close": bc.te._daily_close(), "daily": daily, "me": me, "info": info, "other": other,
            "nfci": nf_raw, "problems": problems + more + list(nf_note)}


# ---------- the signals ----------

def _ret(m: pd.Series, k: int) -> pd.Series:
    return m / m.shift(k) - 1


def _mom_u(m: pd.Series) -> pd.Series:
    return (_ret(m, 1) + _ret(m, 3) + _ret(m, 6) + _ret(m, 12)) / 4


def _mom_w(m: pd.Series) -> pd.Series:
    return (12 * _ret(m, 1) + 4 * _ret(m, 3) + 2 * _ret(m, 6) + _ret(m, 12)) / 4


def _above(m: pd.Series) -> pd.Series:
    avg = m.rolling(AVG_M, min_periods=AVG_M).mean()
    return (m > avg).astype(float).where(avg.notna())


def _count(me: dict) -> tuple[pd.Series, list[str]]:
    have = [k for k in BREADTH if k in me]
    frame = pd.concat({k: _above(me[k]) for k in have}, axis=1)
    return frame.sum(axis=1).where(frame.notna().all(axis=1)), have


def _all_flags(me: dict) -> dict:
    F = {}
    mu = _mom_u(me[TIP])
    F["t1"] = (mu > 0).astype(float).where(mu.notna())
    F["t2"] = _above(me[TIP])
    cnt, _ = _count(me)
    F["b1"] = (cnt >= NEED).astype(float).where(cnt.notna())
    if all(k in me for k in DAA):
        a, b = _mom_w(me["VWO"]), _mom_w(me["BND"])
        F["b2"] = ((a > 0) & (b > 0)).astype(float).where(a.notna() & b.notna())
    return {k: v.dropna() for k, v in F.items()}


def _shift(F: pd.Series, sh: int) -> pd.Series:
    return pd.Series(np.roll(F.values, int(sh)), index=F.index)


# ---------- report-only pieces ----------

def _agree(a: pd.Series, b: pd.Series, since: pd.Period) -> tuple[float, float, int] | None:
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    x = x[x.index >= since]
    if len(x) < 24:
        return None
    phi = float(np.corrcoef(x["a"], x["b"])[0, 1]) if x["a"].std() > 0 and x["b"].std() > 0 else float("nan")
    return float((x["a"] == x["b"]).mean()), phi, len(x)


def _flips(f: pd.Series) -> list[tuple[pd.Period, int]]:
    d = f.diff()
    return [(p, int(np.sign(v))) for p, v in d.items() if _ok(v) and v != 0]


def _match(btc: list, gate: list) -> dict:
    """For each BTC trend flip, the nearest same-direction gate flip within +/- WIN months (lag = gate - btc)."""
    lags, miss = [], 0
    for p, sgn in btc:
        best = None
        for q, s2 in gate:
            if s2 != sgn:
                continue
            k = (q - p).n
            if abs(k) <= WIN and (best is None or abs(k) < abs(best) or (abs(k) == abs(best) and k < best)):
                best = k
        if best is None:
            miss += 1
        else:
            lags.append(best)
    lags = np.array(lags)
    return {"n": len(btc), "first": int((lags < 0).sum()), "same": int((lags == 0).sum()), "after": int((lags > 0).sum()),
            "miss": miss, "med": float(np.median(lags)) if len(lags) else None}


def _corr_lags(cnt: pd.Series, btc_me: pd.Series) -> dict:
    dc = cnt.diff()
    br = np.log(btc_me).diff()
    both = pd.concat([dc.rename("c"), br.rename("b")], axis=1, join="inner").dropna()
    both = both[both.index >= LL_FROM]
    rows = {}
    for k in LAGS:
        pair = pd.concat([both["c"], both["b"].shift(-k)], axis=1).dropna()
        rows[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return {"corr": rows, "n": len(both), "band": 2 / math.sqrt(len(both)), "first": both.index[0], "last": both.index[-1]}


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: bc._Exam) -> tuple[int, int]:
    rng = np.random.default_rng(17)
    close = data["close"]
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        me2 = {}
        for k, s in data["daily"].items():
            me2[k] = _monthly(s.where(s.index <= c, s * np.exp(rng.normal(0, 0.2, len(s)))))
        F1, F2 = _all_flags(data["me"]), _all_flags(me2)
        upto = close.index <= c
        for k in F1:
            a = bc._pos("m2", ex.btc_ma, bc._on_days(F1[k], close.index + bc.DAY))
            b = bc._pos("m2", ma2, bc._on_days(F2[k], close.index + bc.DAY))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    F = _all_flags(data["me"])
    rng = np.random.default_rng(bc.SEED + 7)
    ev = {k: bc._events(close, F[k], rng) for k in F}
    s = {"ma50": ex.summary("ma50", None), **{k: ex.summary("m2", F[k]) for k in F}}

    small = {"ma50": ex.summary("ma50", None, PL_SIMS)}
    plc = {}
    for k, seed in (("t1", bc.SEED + 51), ("b1", bc.SEED + 52)):
        if k not in F:
            continue
        small[k] = ex.summary("m2", F[k], PL_SIMS)
        prng = np.random.default_rng(seed)
        n = len(F[k])
        hits, beat_ma, lows = 0, 0, []
        for sh in prng.integers(GAP_M, n - GAP_M + 1, DRAWS):
            x = ex.summary("m2", _shift(F[k], sh), PL_SIMS)
            lows.append(x["low"])
            hits += int(bc._score_ge(x, small[k]))
            beat_ma += int(bc._score_ge(x, small["ma50"]))
        plc[k] = {"share": hits / DRAWS, "low": float(np.median(lows)), "ma": beat_ma / DRAWS}

    pos = {"ma50": ex.btc_pos("ma50", None), **{k: ex.btc_pos("m2", F[k]) for k in F}}
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST), **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    a = close.index >= pd.Timestamp(FIRST)
    differ = {k: (int((np.nan_to_num(pos["ma50"][a]) != np.nan_to_num(pos[k][a])).sum()), int(a.sum()))
              for k in ("t1", "b1") if k in pos}

    # agreement with NFCI (report only)
    agree = {}
    if data["nfci"] is not None:
        from dashboard.studies2 import nfci_filter as nf
        months = pd.period_range(data["nfci"].index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
        N = nf._flags(data["nfci"], "avg52", months).dropna()
        since = bc.EVENT_FROM - 1
        for k in ("t1", "b1"):
            if k in F:
                agree[k] = _agree(F[k], N, since)
        agree["nfci_fav"] = float(N[N.index >= since].mean())
    if "t1" in F and "b1" in F:
        agree["t1b1"] = _agree(F["t1"], F["b1"], bc.EVENT_FROM - 1)

    # who moves first (report only)
    btc_me = bc._month_ends(close)
    btc_tr = _above(btc_me).dropna()
    btc_fl = [x for x in _flips(btc_tr) if x[0] >= LL_FROM]
    lead = {k: _match(btc_fl, _flips(F[k])) for k in ("t1", "b1") if k in F}
    cnt, have = _count(data["me"])
    ll = _corr_lags(cnt.dropna(), btc_me)
    look = _lookahead(data, ex)
    return {"ex": ex, "F": F, "ev": ev, "s": s, "small": small, "plc": plc, "pos": pos, "years": years,
            "differ": differ, "agree": agree, "lead": lead, "ll": ll, "cnt": cnt, "have": have, "btc_tr": btc_tr,
            "look": look, "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    me = data["me"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base = s["ma50"]
    n_c, n_d = base["c_n"], base["d_n"]
    mains = [k for k in ("t1", "b1") if k in F]
    keys = [k for k in ("t1", "t2", "b1", "b2") if k in F]

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    q = {}
    for k in mains:
        x = s[k]
        q[k] = {"event": ev[k]["all_three"], "score": bc._score_ge(x, base),
                "loss": x["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["plc"][k]["share"] <= 0.10}
        q[k]["pass"] = all(q[k].values())

    # ----- table 1: event table -----
    rows1 = []
    for k in keys:
        e = ev[k]
        a = e["all"]
        ci = a.get("ci")
        rows1.append([GATE[k],
                      f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()]
                     + ["是" if e["all_three"] else "否"])
    t1 = {"caption": ("每個月最後一根比特幣日線（UTC 0 點收盤，比美股收盤晚三到四小時）決定，用當月最後一個美股收盤價（還原配息）。"
                      "每格是下一個月比特幣報酬（月底收盤到月底收盤）的平均，持有月份 "
                      f"{ev[mains[0]]['first']} 到 {ev[mains[0]]['last']}；差距＝有利減不利，括號是區塊拔靴法（連續 {bc.BLOCK} 個月一塊、"
                      f"{bc.BOOT:,} 次）的 90% 區間；分段欄括號是（有利｜不利）月數。這張表只看閘門本身，不含 50 日均線。"),
          "columns": ["閘門（有利的條件）", "有利月：個數｜平均報酬", "不利月：個數｜平均報酬", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in bc.PERIODS] + ["三段都是有利較好"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    rows2 = []
    for lab, x in [("50 日均線（基準）", base)] + [(NAMES[k], s[k]) for k in keys]:
        ck = x["checks"]
        rows2.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(ex.chold[1])))}｜—", "—"])
    t2 = {"caption": (f"main_signal_check 的公平考試（四項都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產（以太幣 ETH、SOL、BNB、XRP、"
                      "微策略公司股票 MSTR）用自己的 50 日均線、同一個閘門。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      "最後兩欄是照課本｜一般人做法的中位數。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: pass test -----
    rows3 = []
    for k in mains:
        x, pl = s[k], r["plc"][k]
        ps = [p["diff"] for p in ev[k]["periods"].values()]
        rows3 += [[SHORT[k], "有利月比不利月好（三段）", "、".join(_p(v) for v in ps), yes(q[k]["event"])],
                  [SHORT[k], "排名分數不低於 50 日均線", f"{_share(x['low'])} 對 {_share(base['low'])}", yes(q[k]["score"])],
                  [SHORT[k], f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])}",
                   yes(q[k]["loss"])],
                  [SHORT[k], "隨機錯位一樣好或更好的比例 ≤ 10%",
                   f"{_share(pl['share'])}（打亂後排名分數中位數 {_share(pl['low'])}，真的 {_share(r['small'][k]['low'])}；"
                   f"打亂後不輸均線的 {_share(pl['ma'])}）", yes(q[k]["plc"])],
                  [SHORT[k], "結果", "—", "通過" if q[k]["pass"] else "沒有通過"]]
    t3 = {"caption": (f"事先寫好的過關標準，T1、B1 各自四項都要過。隨機錯位：把每月的閘門整串循環平移 {GAP_M} 個月到「長度 − {GAP_M}」個月、"
                      f"{DRAWS} 次，整套考試（含一般人做法）重做；為了在幾分鐘內算完，這一步一般人做法每個收盤時間只模擬 {PL_SIMS} 次（種子相同），"
                      "真的規則也用同樣方式算一次再比。同時考兩條主要規則，只有一條過的話，證據比單獨考一條就過要弱。"),
          "columns": ["規則", "條件", "數字", "結果"], "rows": rows3}

    # ----- table 4: years -----
    rows4 = []
    Y = r["years"]
    yrs = sorted(Y["ma50"])
    for y in yrs:
        a = Y["ma50"].get(y)
        row = [f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(Y["hold"].get(y)), _pct(a)]
        for k in mains:
            b = Y[k].get(y)
            row += [_pct(b), _pp(b - a) if _ok(a) and _ok(b) else "—"]
        tm = F["t1"][(F["t1"].index + 1).year == y] if "t1" in F else pd.Series(dtype=float)
        bm = F["b1"][(F["b1"].index + 1).year == y] if "b1" in F else pd.Series(dtype=float)
        row += [f"{int(tm.sum())}｜{int(bm.sum())}（共 {len(tm)} 個月）"]
        rows4.append(row)
    cols4 = ["年份", "一直抱著", "50 日均線"]
    for k in mains:
        cols4 += [SHORT[k], f"{SHORT[k]} 減均線（百分點）"]
    cols4 += ["閘門開著的月數：T1｜B1"]
    dif = "；".join(f"{SHORT[k]} 和均線持有狀態不同 {r['differ'][k][0]:,} 天／{r['differ'][k][1]:,} 天" for k in r["differ"])
    t4 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看差距來自哪幾年。T1、B1 只會比均線少持有"
                      f"（閘門關的月份，均線持有也改成現金）。最後一欄是那年有幾個月閘門開著（允許持有）。{dif}。"),
          "columns": cols4, "rows": rows4}

    # ----- table 5: agreement -----
    ag = r["agree"]
    rows5 = []
    for k, lab in (("t1", "T1 閘門 對 NFCI 閘門"), ("b1", "B1 閘門 對 NFCI 閘門"), ("t1b1", "T1 閘門 對 B1 閘門")):
        v = ag.get(k)
        if v:
            rows5.append([lab, f"{v[2]} 個月", f"{v[0] * 100:.0f}%", f"{v[1]:+.2f}"])
    fav_share = "、".join(f"{SHORT[k]} {_share(ev[k]['fav_share'])}" for k in keys)
    t5 = {"caption": ("和 nfci_filter 的 N1 閘門（芝加哥聯準銀行金融情勢指數低於自己 52 週平均＝寬鬆）比：每月「有利／不利」一樣的比例，"
                      "和兩串 0／1 的相關係數（1＝完全同步、0＝互不相關）。如果兩個閘門每個月都一樣（相關接近 1），那就是同一個訊號，不是新的證據。"
                      f"事件表期間有利月的比例：{fav_share}" + (f"、NFCI {_share(ag['nfci_fav'])}" if "nfci_fav" in ag else "") + "。"),
          "columns": ["比較", "月數", "同一個方向的比例", "相關係數"], "rows": rows5}

    # ----- table 6: who moves first -----
    rows6 = []
    for k, m in r["lead"].items():
        rows6.append([f"{SHORT[k]} 閘門", f"{m['n']} 次", f"{m['first']} 次", f"{m['same']} 次", f"{m['after']} 次", f"{m['miss']} 次",
                      f"{m['med']:+.0f} 個月" if _ok(m["med"]) else "—"])
    t6 = {"caption": (f"誰先動（一）：比特幣自己的 10 個月趨勢（月底收盤高於或低於自己 10 個月平均）從 {LL_FROM} 起每次翻轉，"
                      f"找前後 {WIN} 個月內同方向、最近的一次閘門翻轉，數閘門比比特幣早、同月、晚。中位數為負＝閘門先動。"),
          "columns": ["閘門", "比特幣翻轉次數", "閘門先翻", "同一個月", "閘門後翻", f"{WIN} 個月內沒有對應", "落後月數中位數"],
          "rows": rows6}
    ll = r["ll"]
    rows7 = []
    for k in LAGS:
        who = f"資產廣度先變，{k} 個月後的比特幣" if k > 0 else (f"比特幣先動，{-k} 個月後的資產廣度" if k < 0 else "同一個月")
        c = ll["corr"][k]
        rows7.append([f"{k:+d}" if k else "0", who, f"{c:+.3f}" + ("＊" if abs(c) > ll["band"] else "")])
    t7 = {"caption": (f"誰先動（二）：7 類資產裡高於 10 個月平均的個數每月的變化（月 t）和比特幣月報酬（對數，月 t+k）的相關係數，"
                      f"{ll['first']} 到 {ll['last']}（{ll['n']} 個月，大約的隨機範圍 ±{ll['band']:.3f}，＊＝超過）。k 為正＝資產廣度先變。"),
          "columns": ["k（月）", "意思", "相關"], "rows": rows7}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    def state(k) -> str:
        v = F[k].get(cur, np.nan) if k in F else np.nan
        return "有利" if v == 1 else "不利" if v == 0 else "—"

    tipm = me[TIP]
    parts = [f"{lab} {_p(_ret(tipm, n).get(cur, np.nan))}" for lab, n in (("1 個月", 1), ("3 個月", 3), ("6 個月", 6), ("12 個月", 12))]
    up = [GLOSS[k].split("（")[0] for k in r["have"] if _above(me[k]).get(cur, np.nan) == 1]
    down = [GLOSS[k].split("（")[0] for k in r["have"] if _above(me[k]).get(cur, np.nan) == 0]
    cn = r["cnt"].get(cur, np.nan)
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"T1 抗通膨公債 TIP 動能（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": f"{state('t1')}；平均 {_p(_mom_u(tipm).get(cur, np.nan), 2)}（{'、'.join(parts)}）；T2（高於 10 個月平均）{state('t2')}"},
        {"label": f"B1 七類資產高於 10 個月平均的個數（{cur.strftime('%Y-%m')} 月底）",
         "value": (f"{int(cn) if _ok(cn) else '—'} 類（至少 {NEED} 類才有利）：{state('b1')}；高於：{'、'.join(up) or '無'}；"
                   f"低於：{'、'.join(down) or '無'}；B2（新興股市 VWO 和整體債券 BND）{state('b2')}")},
        {"label": "比特幣：50 日均線｜T1｜B1",
         "value": "｜".join(word(pm[k][-1]) if k in pm else "—" for k in ("ma50", "t1", "b1"))},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議；下一次決定在這個月最後一根日線收盤"},
    ]

    # ----- verdict -----
    def ylist(k, sign):
        out = []
        for y in yrs:
            a, b = Y["ma50"].get(y), Y[k].get(y)
            if _ok(a) and _ok(b) and sign * (b - a) > 0.5:
                out.append(f"{y}（{b - a:+.0f}）")
        return "、".join(out) or "無"

    parts_v = []
    for k in mains:
        x, a0 = s[k], ev[k]["all"]
        ps = [p["diff"] for p in ev[k]["periods"].values()]
        parts_v.append(
            f"{SHORT[k]}（{NAMES[k].split('：')[1].replace('（主要）', '')}）{'通過' if q[k]['pass'] else '沒有通過'}：有利月下個月比特幣平均多 "
            f"{_p(a0['diff'])}，三段 {'、'.join(_p(v) for v in ps)}（{yes(q[k]['event'])}）；排名分數 {_share(x['low'])} 對均線 "
            f"{_share(base['low'])}（{yes(q[k]['score'])}）；最深虧損中位數 {_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])}"
            f"（{yes(q[k]['loss'])}）；隨機錯位一樣好 {_share(r['plc'][k]['share'])}（{yes(q[k]['plc'])}）。"
            f"比均線好的年份：{ylist(k, 1)}；比均線差的年份：{ylist(k, -1)}。")
    rob = "、".join(f"{SHORT[k]} 排名分數 {_share(s[k]['low'])}、最深虧損中位數 {_pct(s[k]['c_deep_med'])}、差距 {_p(ev[k]['all']['diff'])}"
                   for k in ("t2", "b2") if k in s)
    npass = sum(q[k]["pass"] for k in mains)
    head = ("兩條主要規則都通過事先寫好的標準。" if npass == 2 else
            "兩條主要規則只有一條通過；同時考兩條、只過一條，證據比較弱。" if npass == 1 else "兩條主要規則都沒有通過事先寫好的標準。")
    agt = []
    for k in mains:
        v = ag.get(k)
        if v:
            agt.append(f"{SHORT[k]} 和 NFCI 閘門同方向 {v[0] * 100:.0f}%（相關 {v[1]:+.2f}）")
    verdict = (head + "".join(parts_v) + (f"穩健度（只報告）：{rob}。" if rob else "")
               + ("；".join(agt) + "。" if agt else "")
               + "".join(f"{SHORT[k]} 閘門翻轉比比特幣自己 10 個月趨勢早 {m['first']} 次、同月 {m['same']} 次、晚 {m['after']} 次。"
                         for k, m in r["lead"].items()))

    tot, bad = r["look"]
    hist = "；".join(f"{GLOSS[k].split('（')[0]} {_d(data['daily'][k].index[0])} 起（{data['info'][k]}）" for k in data["daily"])
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好）：T1 = hold BTC when ma50 is on AND TIP 13612U momentum (average of its 1-, "
         "3-, 6- and 12-month total returns at month-end) > 0. B1 = hold when ma50 is on AND at least 4 of SPY, EFA, EEM, IEF, "
         "GLD, DBC, VNQ have month-end close above their own 10-month average. Robustness rows (no placebo): T2 = TIP above its "
         "10-month average; B2 = VWO and BND 13612W (12/4/2/1 weights on 1/3/6/12-month returns) both > 0. Decided at the "
         "month-end BTC daily close using the last US close of that month, held for the next month. Pass test, each main rule: "
         "favorable > unfavorable next-month BTC return in 2011-16, 2017-21, 2022-now AND ranking score >= ma50's AND median "
         "deepest loss no worse AND placebo share (monthly gate circularly shifted >= 12 months, 200 draws, ordinary person "
         "50 simulations per clock) <= 10%."),
        ("兩個想法從哪裡來：（一）Keller 和 Keuning 的「混合資產配置」（HAA，2023 年）只用一個「金絲雀」——TIP 抗通膨公債基金。"
         "每月最後一個交易日收盤算它的 13612U 動能（過去 1、3、6、12 個月總報酬的簡單平均），小於等於 0 就整個轉到防守資產，"
         "作者說要的是對「殖利率上升、預期通膨上升」有反應的金絲雀。（二）Faber 2006 年的戰術配置：每類資產月底價格高於 10 個月平均才持有；"
         "Keller 和 Keuning 的「防守資產配置」（DAA）用新興市場股票 VWO 和整體債券 BND 當一對金絲雀，兩個都轉壞就全部現金。"
         "這些都是給股債組合用的，這裡是第一次拿來當比特幣的閘門。"),
        ("13612W 的權重（1 個月 ×12、3 個月 ×4、6 個月 ×2、12 個月 ×1，再除以 4）是照 Keller 和 Keuning 的 DAA 論文（SSRN 3212862）"
         "的定義事先寫進規則；這次讀到的 CXO Advisory 摘要只寫「1、3、6、12 個月報酬的平均」，沒有寫權重，SSRN 論文頁面拒絕自動讀取，"
         "所以權重沒有從原文再核對一次。Faber 的原論文頁面（SSRN）同樣拒絕自動讀取，改讀他自己網站的說明。"),
        ("和以前的研究有什麼不同：cross_market 測的是「每天」看美股（QQQ、SPY 的 50／100／200 日均線、從高點回落）、恐慌指數 VIX、"
         "十年期殖利率跳動、美元指數；這裡是「每月」看七大類資產（股票、債券、黃金、原物料、不動產）各自的絕對趨勢有幾類向上，"
         "以及抗通膨公債自己的總報酬動能。沒有用任何兩個資產相除的比值（比值類壓力指標是另一個研究 stress_family 的範圍）。"),
        (f"資料：Yahoo Finance 還原配息的日收盤，取每月最後一個交易日：{hist}。DBC 2006 年才有，10 個月平均要到 2006 年底才有值，"
         "事件表從 2011-07 起，所以每一類都有完整資料。時間對齊：美股收盤約 UTC 20–21 點，比特幣月底那根日線 UTC 0 點收，晚三到四小時，"
         f"用的一定是決定之前已經收完的價格。偷看未來的檢查：三個月底之後把比特幣和所有 ETF 價格換成亂數，切點以前的持有狀態 {tot} 組裡 "
         f"{tot - bad} 組完全沒變" + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("讀表的注意：TIP 是美國抗通膨公債，價格和「實質利率」反方向走——實質利率上升時 TIP 跌、動能轉負。2022 年聯準會快速升息，"
         "TIP 動能長時間為負，2023 年比特幣大漲時 T1 閘門有很多月是關的，這一年的代價看第四張表。"
         "「誰先動」的表只是粗略描述：翻轉次數很少，相鄰月份的相關也不是互相獨立，偶爾一兩格超過隨機範圍可能只是巧合。"),
        ("其他幣、MSTR 用的是同一個閘門。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "Keller & Keuning：Hybrid Asset Allocation（HAA）介紹文（TIP 金絲雀、13612U 動能；論文 SSRN 4346906）", "url": HAA_URL},
        {"name": "CXO Advisory：Keller & Keuning 防守資產配置（DAA）摘要（VWO、BND 金絲雀）", "url": CXO_URL},
        {"name": "Meb Faber：10 個月平均的戰術配置模型說明", "url": FABER_URL},
    ]
    for k in data["daily"]:
        sources.append({"name": f"Yahoo Finance 每日還原收盤：{GLOSS[k]}", "url": tw.CHART.format(symbol=k)})
    if data["nfci"] is not None:
        from dashboard.studies2 import nfci_filter as nf
        sources.append({"name": "芝加哥聯準銀行 NFCI 資料（只用來比較）", "url": nf.CHI_URL})
    sources += [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]

    return {
        "id": "canary_breadth",
        "title": "抗通膨公債與資產廣度能不能幫 50 日均線",
        "question": ("股債配置的研究者常用「金絲雀」提早發現危險：抗通膨公債（TIP）的動能轉負，代表實質利率在升；"
                     "股票、債券、黃金、原物料、不動產這幾大類資產多數跌破自己 10 個月平均，代表整體市場在轉弱。"
                     "把這兩種金絲雀每月底看一次，當成比特幣 50 日均線的閘門（不好就不持有），能不能少虧、通過完整的公平考試？"
                     "它們和金融情勢指數是不是同一個訊號？誰先動？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5, t6, t7],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
