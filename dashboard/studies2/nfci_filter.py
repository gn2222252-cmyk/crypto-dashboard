"""Historical study "nfci_filter": does the Chicago Fed National Financial Conditions Index (NFCI) improve the 50-day
moving average rule (ma50, main_signal_check's main signal) for bitcoin?

Mechanism under test: tight financial conditions (high NFCI) come first and risk assets fall after. If that is true,
NFCI changes should lead BTC changes, and holding only while conditions are looser than usual should cut losses.

Pre-registered (written before this study was run; no other variants, no parameter changes afterwards):
  Data: NFCI, weekly, dated the Friday of the week, higher = tighter, zero = average since 1971. Source: the Chicago
  Fed's own key-less CSV (api.data.chicagofed.org nfci-data-series-csv.csv, the file its NFCI page links to); FRED
  fredgraph CSV id NFCI as fallback; the last good download kept in common.STORE as the final fallback. Fixed user
  agent "btc-research-dashboard". Today's vintage only (NFCI is revised every week; stated caveat).
  Point in time: NFCI for the week ending Friday f is released the following Wednesday (f + 5 days); to be safe a
  weekly value is usable only when it is at least 6 days old. Decision at the last daily close of month M (00:00 UTC,
  te._daily_close, exactly business_cycle's monthly timing): the latest weekly value with f + 6 days <= the last
  calendar day of M; held during month M+1.
  N1 (main rule): hold BTC when ma50 is on AND that usable NFCI value is below the average of its own last 52 weekly
  values (including itself). Robustness row (ONE only, no placebo): same timing, NFCI < 0.
  Pass test - all four, exactly like oecd_followup:
    (1) favorable > unfavorable average next-month BTC return in 2011-16, 2017-21 and 2022-now (business_cycle's
        event table, holding months from 2011-07);
    (2) ranking score >= ma50's (main_signal_check's four checks via business_cycle._Exam, incl. the 95% loss bar);
    (3) median deepest loss over the 1,728 textbook cells no worse than ma50's;
    (4) placebo share <= 10%: the weekly NFCI series circularly shifted by 26 ... (length - 26) weeks, 200 draws,
        flags rebuilt (52-week average after shifting) and the whole exam re-scored with the ordinary-person check at
        50 simulations per clock (the real rule re-scored the same way).
  Report only: calendar-year returns and days differing from ma50 (where the edge comes from, from 2015-01-01);
  lead/lag: Pearson correlation of weekly NFCI changes (week t) with weekly BTC log returns (Friday to Friday close,
  week t+k) for k = -8 ... +8, weekly from 2011-01-07, also 2020-now; k > 0 = NFCI moves first, k < 0 = BTC moves
  first; rough band +/- 2/sqrt(n). Look-ahead check: scramble prices and not-yet-usable NFCI values after three
  month-ends; no position decided up to then may change. Today: states only.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc

AVG_W = 52           # weekly values in the average
SAFE = pd.Timedelta(days=6)
GAP_W = 26           # placebo minimum shift in weeks
DRAWS, PL_SIMS = bc.DRAWS, bc.PL_SIMS
LAGS = range(-8, 9)
LL_FROM, LL_RECENT = "2011-01-07", "2020-01-01"
FIRST = msc.DAILY_STARTS[0]

CHI_URL = "https://api.data.chicagofed.org/NFCI/nfci-data-series-csv.csv"
CHI_INDEX = "https://data.chicagofed.org/cfed-drm-chicago/NFCI"
CHI_PAGE = "https://www.chicagofed.org/research/data/nfci/current-data"
FRED_URL = bc.FRED_CSV.format("NFCI")
STORE_PATH = common.STORE / "nfci_filter_nfci.json"

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _chicago() -> dict:
    frame = pd.read_csv(io.StringIO(bc._get(CHI_URL, timeout=(10, 60)).text))
    when = pd.to_datetime(frame["Friday_of_Week"], format="%m/%d/%Y")
    s = pd.to_numeric(frame["NFCI"], errors="coerce")
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(when, s) if math.isfinite(v)}


def _fred() -> dict:
    frame = pd.read_csv(io.StringIO(bc._get(FRED_URL, timeout=(8, 25), tries=2).text))
    s = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return {str(d)[:10]: float(v) for d, v in zip(frame.iloc[:, 0], s) if math.isfinite(v)}


def _series() -> tuple[pd.Series, str, list[str]]:
    notes, days, where = [], None, ""
    for name, fn, label in (("chicago", _chicago, "芝加哥聯準銀行自己的 CSV"), ("fred", _fred, "FRED")):
        try:
            got = fn()
            if len(got) < 1000:
                raise RuntimeError(f"只回傳 {len(got)} 筆")
            days, where = got, label
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            notes.append(f"{label}這次抓不到（{type(exc).__name__}）。")
    if days is None:
        if not STORE_PATH.exists():
            raise RuntimeError("NFCI 兩個來源都抓不到，也沒有存檔")
        days = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        where = "上次存下來的資料"
        notes.append(f"NFCI 改用上次存下來的資料（到 {max(days)}）。")
    else:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_PATH.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    age = (pd.Timestamp(common.today_utc()) - s.index[-1]).days
    if age > 30:
        notes.append(f"NFCI 最新一週是 {_d(s.index[-1])}，已經 {age} 天前，資料可能沒有更新。")
    return s, where, notes


def _load() -> dict:
    s, where, problems = _series()
    other, more = msc._load_other()
    return {"close": bc.te._daily_close(), "nfci": s, "where": where, "other": other, "problems": problems + more}


# ---------- the signal ----------

def _weekly_fav(s: pd.Series, rule: str) -> pd.Series:
    if rule == "zero":
        return (s < 0).astype(float).where(s.notna())
    avg = s.rolling(AVG_W, min_periods=AVG_W).mean()
    return (s < avg).astype(float).where(avg.notna() & s.notna())


def _monthly(fav_w: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Decision month M -> flag from the latest weekly value at least 6 days old at the last calendar day of M."""
    last_day = months.to_timestamp(how="end").normalize()
    pos = np.searchsorted(fav_w.index.values, (last_day - SAFE).values, side="right") - 1
    vals = np.where(pos >= 0, fav_w.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=months)


def _flags(s: pd.Series, rule: str, months: pd.PeriodIndex) -> pd.Series:
    return _monthly(_weekly_fav(s, rule), months)


def _used_week(s: pd.Series, month: pd.Period) -> pd.Timestamp | None:
    cut = month.to_timestamp(how="end").normalize() - SAFE
    k = s.index.searchsorted(cut, side="right") - 1
    return s.index[k] if k >= 0 else None


# ---------- lead / lag ----------

def _leadlag(close: pd.Series, s: pd.Series) -> dict:
    w = np.log(close.resample("W-FRI").last()).diff()
    dn = s.diff()
    both = pd.concat([dn.rename("n"), w.rename("b")], axis=1, join="inner").dropna()
    out = {}
    for lab, since in (("all", LL_FROM), ("recent", LL_RECENT)):
        x = both[both.index >= pd.Timestamp(since)]
        rows = {}
        for k in LAGS:
            pair = pd.concat([x["n"], x["b"].shift(-k)], axis=1).dropna()
            rows[k] = (float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), len(pair))
        out[lab] = {"corr": rows, "n": len(x), "band": 2 / math.sqrt(len(x)), "first": x.index[0], "last": x.index[-1],
                    "ac": [float(x["n"].autocorr(k)) for k in (1, 2, 3)]}
    return out


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: bc._Exam, months) -> tuple[int, int]:
    rng = np.random.default_rng(13)
    close, s = data["close"], data["nfci"]
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        s2 = s.where(s.index + SAFE <= c, s + rng.normal(0, 1.0, len(s)))
        upto = close.index <= c
        for rule in ("avg52", "zero"):
            a = bc._pos("m2", ex.btc_ma, bc._on_days(_flags(s, rule, months), close.index + bc.DAY))
            b = bc._pos("m2", ma2, bc._on_days(_flags(s2, rule, months), close.index + bc.DAY))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, raw = data["close"], data["nfci"]
    months = pd.period_range(raw.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    F = _flags(raw, "avg52", months)
    Fz = _flags(raw, "zero", months)
    rng = np.random.default_rng(bc.SEED + 5)
    ev = bc._events(close, F, rng)
    evz = bc._events(close, Fz, rng)
    s = {"ma50": ex.summary("ma50", None), "n1": ex.summary("m2", F), "nz": ex.summary("m2", Fz)}

    small = {"ma50": ex.summary("ma50", None, PL_SIMS), "n1": ex.summary("m2", F, PL_SIMS)}
    prng = np.random.default_rng(bc.SEED + 41)
    n = len(raw)
    hits, lows, beat_ma = 0, [], 0
    for sh in prng.integers(GAP_W, n - GAP_W + 1, DRAWS):
        s2 = pd.Series(np.roll(raw.values, int(sh)), index=raw.index)
        f2 = _flags(s2, "avg52", months)
        x = ex.summary("m2", f2, PL_SIMS)
        lows.append(x["low"])
        hits += int(bc._score_ge(x, small["n1"]))
        beat_ma += int(bc._score_ge(x, small["ma50"]))

    pos = {"ma50": ex.btc_pos("ma50", None), "n1": ex.btc_pos("m2", F)}
    a = close.index >= pd.Timestamp(FIRST)
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["n1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()), int(((p0 > 0.5) & (p1 < 0.5)).sum()))
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST),
             **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    look = _lookahead(data, ex, months)
    ll = _leadlag(close, raw)
    return {"ex": ex, "F": F, "Fz": Fz, "ev": ev, "evz": evz, "s": s, "small": small, "plc": hits / DRAWS,
            "plc_low": float(np.median(lows)), "plc_ma": beat_ma / DRAWS, "pos": pos, "differ": differ,
            "years": years, "look": look, "ll": ll, "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    raw = data["nfci"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base, n1, nz = s["ma50"], s["n1"], s["nz"]
    n_c, n_d = base["c_n"], base["d_n"]

    q = {"event": ev["all_three"], "score": bc._score_ge(n1, base),
         "loss": n1["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["plc"] <= 0.10}
    q["pass"] = all(q.values())

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    # ----- table 1: event table -----
    rows1 = []
    for lab, e in (("N1：NFCI 低於自己 52 週平均（主要）", ev), ("NFCI 低於 0（穩健度，只報告）", r["evz"])):
        a = e["all"]
        ci = a.get("ci")
        rows1.append([lab, f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()]
                     + ["是" if e["all_three"] else "否"])
    t1 = {"caption": ("每個月最後一根日線（UTC 0 點收盤）決定：用那時已經公布至少 6 天的最新一週 NFCI（金融情勢指數，越高代表借錢越難、市場越緊），"
                      f"低於它自己最近 {AVG_W} 週平均就算「寬鬆、有利」。每格是下一個月比特幣報酬（月底收盤到月底收盤）的平均，持有月份 "
                      f"{ev['first']} 到 {ev['last']}；差距＝有利減不利，括號是區塊拔靴法（連續 {bc.BLOCK} 個月一塊、{bc.BOOT:,} 次）的 90% 區間；"
                      "分段欄括號是（有利｜不利）月數。這張表只看金融情勢本身，不含 50 日均線。"),
          "columns": ["設定", "有利月：個數｜平均報酬", "不利月：個數｜平均報酬", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in bc.PERIODS] + ["三段都是有利較好"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    rows2 = []
    for lab, x in (("50 日均線（基準）", base), ("N1：50 日均線而且 NFCI 低於 52 週平均", n1),
                   ("50 日均線而且 NFCI 低於 0（只報告）", nz)):
        ck = x["checks"]
        rows2.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    t2 = {"caption": (f"main_signal_check 的公平考試（四項都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產（ETH、SOL、BNB、XRP、MSTR）"
                      "用自己的 50 日均線、同一個美國金融情勢訊號。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      "最後兩欄是照課本｜一般人做法的中位數。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: pass test -----
    ps = [p["diff"] for p in ev["periods"].values()]
    rows3 = [["有利月比不利月好（三段）", "、".join(_p(v) for v in ps), yes(q["event"])],
             ["排名分數不低於 50 日均線", f"{_share(n1['low'])} 對 {_share(base['low'])}", yes(q["score"])],
             [f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(n1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(q["loss"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['small']['n1']['low'])}；"
              f"打亂後不輸均線的 {_share(r['plc_ma'])}）", yes(q["plc"])],
             ["結果", "—", "通過" if q["pass"] else "沒有通過"]]
    t3 = {"caption": (f"事先寫好的過關標準，四項都要過。隨機錯位：把每週 NFCI 整串循環平移 {GAP_W} 週到「長度 − {GAP_W}」週、{DRAWS} 次，"
                      f"重新算 52 週平均和每月訊號、整套考試（含一般人做法）重做；為了在十分鐘內算完，這一步一般人做法每個收盤時間只模擬 {PL_SIMS} 次"
                      "（種子相同），真的規則也用同樣方式算一次再比（簡化版）。"),
          "columns": ["條件", "數字", "結果"], "rows": rows3}

    # ----- table 4: years -----
    dd_, nn, out_ = r["differ"]
    rows4 = []
    yrs = sorted(r["years"]["ma50"])
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["n1"].get(y)
        rows4.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    rows4.append([f"和 50 日均線持有狀態不同的天數（{FIRST} 起）", "—", "—", f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—",
                  f"其中均線持有、N1 現金 {out_:,} 天"])
    t4 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看 N1 和 50 日均線的差來自哪幾年；"
                      "N1 只會比均線少持有（金融情勢偏緊時，均線持有也改成現金）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "N1", "N1 減均線（百分點）"], "rows": rows4}

    # ----- table 5: lead / lag -----
    ll = r["ll"]
    la, lr = ll["all"], ll["recent"]
    rows5 = []
    for k in LAGS:
        if k > 0:
            who = f"NFCI 先動，{k} 週後的比特幣"
        elif k < 0:
            who = f"比特幣先動，{-k} 週後的 NFCI"
        else:
            who = "同一週"
        ca, cr_ = la["corr"][k][0], lr["corr"][k][0]
        mark = lambda c, band: f"{c:+.3f}" + ("＊" if abs(c) > band else "")  # noqa: E731
        rows5.append([f"{k:+d}" if k else "0", who, mark(ca, la["band"]), mark(cr_, lr["band"])])
    t5 = {"caption": (f"誰先動：每週 NFCI 的變化（週 t）和比特幣週報酬（週五到週五收盤的對數報酬，週 t+k）的相關係數。"
                      f"k 為正＝NFCI 先變、之後幾週的比特幣；k 為負＝比特幣先變、之後幾週的 NFCI。全期 {_d(la['first'])} 到 {_d(la['last'])}"
                      f"（{la['n']} 週，大約的隨機範圍 ±{la['band']:.3f}）；{LL_RECENT[:4]} 年起 {lr['n']} 週（±{lr['band']:.3f}）。"
                      "＊＝超過這個範圍。NFCI 本身就包含股市波動、利差等市場價格，所以同一週的相關大多只是同時反映，不代表誰造成誰。"),
          "columns": ["k（週）", "意思", "全期相關", f"{LL_RECENT[:4]} 年起相關"], "rows": rows5}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    used = _used_week(raw, cur)
    avg = raw.rolling(AVG_W, min_periods=AVG_W).mean()
    fnow = F.get(cur, np.nan)
    pm = r["pos"]
    latest_fav = bool(raw.iloc[-1] < avg.iloc[-1])

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"NFCI（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": (f"{'寬鬆、有利' if fnow == 1 else '偏緊、不利' if fnow == 0 else '—'}；用 {_d(used) if used is not None else '—'} 那週的 "
                   f"{_num(raw.get(used, np.nan), 3)}，近 {AVG_W} 週平均 {_num(avg.get(used, np.nan), 3)}")},
        {"label": "最新公布的 NFCI",
         "value": (f"{_d(raw.index[-1])} 那週 {_num(raw.iloc[-1], 3)}，近 {AVG_W} 週平均 {_num(avg.iloc[-1], 3)}"
                   f"（{'低於' if latest_fav else '不低於'}平均；{'低於 0，比歷史平均寬鬆' if raw.iloc[-1] < 0 else '不低於 0'}；"
                   "下一次決定看月底時能用的那一週）")},
        {"label": "比特幣：50 日均線｜N1", "value": f"{word(pm['ma50'][-1])}｜{word(pm['n1'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    a0 = ev["all"]
    better = [y for y in yrs if _ok(r["years"]["n1"].get(y)) and _ok(r["years"]["ma50"].get(y))
              and r["years"]["n1"][y] - r["years"]["ma50"][y] > 0.5]
    worse = [y for y in yrs if _ok(r["years"]["n1"].get(y)) and _ok(r["years"]["ma50"].get(y))
             and r["years"]["n1"][y] - r["years"]["ma50"][y] < -0.5]
    c0 = la["corr"][0][0]
    lead_n = [k for k in LAGS if k > 0 and abs(la["corr"][k][0]) > la["band"]]
    lead_b = [k for k in LAGS if k < 0 and abs(la["corr"][k][0]) > la["band"]]
    best_n = max((k for k in LAGS if k > 0), key=lambda k: abs(la["corr"][k][0]))
    best_b = max((k for k in LAGS if k < 0), key=lambda k: abs(la["corr"][k][0]))
    if lead_b and not lead_n:
        mech = "比特幣先動、NFCI 之後才跟上的跡象比較多，和「金融情勢先緊、風險資產後跌」的說法相反。"
    elif lead_n and not lead_b:
        mech = "有 NFCI 先動、比特幣之後才跟的跡象。"
    elif lead_n and lead_b:
        mech = "兩個方向都有幾週超過隨機範圍，看不出明確的先後。"
    else:
        mech = "前後幾週都沒有超過隨機範圍，主要是同一週一起動，看不出誰先。"
    verdict = (f"{'通過' if q['pass'] else '沒有通過'}事先寫好的標準。N1（50 日均線而且 NFCI 低於自己 52 週平均）："
               f"金融情勢寬鬆月下個月比特幣平均比偏緊月多 {_p(a0['diff'])}，三段 {'、'.join(_p(v) for v in ps)}（{yes(q['event'])}）；"
               f"排名分數 {_share(n1['low'])} 對均線 {_share(base['low'])}（{yes(q['score'])}）；最深虧損中位數 {_pct(n1['c_deep_med'])} "
               f"對 {_pct(base['c_deep_med'])}（{yes(q['loss'])}）；隨機錯位一樣好 {_share(r['plc'])}（{yes(q['plc'])}）。"
               f"穩健度（NFCI 低於 0）排名分數 {_share(nz['low'])}、差距 {_p(r['evz']['all']['diff'])}。"
               + (f"比均線好的年份：{'、'.join(str(y) for y in better)}" if better else "沒有一年明顯比均線好")
               + (f"；比均線差的年份：{'、'.join(str(y) for y in worse)}。" if worse else "。")
               + f"誰先動：同一週相關 {c0:+.2f}（NFCI 變緊的那週比特幣多半跌）；NFCI 先動最明顯的是 {best_n} 週後 "
               f"{la['corr'][best_n][0]:+.3f}，比特幣先動最明顯的是 {-best_b} 週後 {la['corr'][best_b][0]:+.3f}（隨機範圍 ±{la['band']:.3f}）。"
               + mech + "NFCI 每週都會修正，這裡只有今天的版本。")

    tot, bad = r["look"]
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好）：N1 = hold BTC when ma50 is on AND the latest NFCI value at least 6 days "
         "old at the month-end decision is below the average of its own last 52 weekly values; decided at the last daily close "
         "of the month, held for the next month. Robustness (one row, no placebo): NFCI < 0. Pass test: favorable > unfavorable "
         "next-month BTC return in 2011-16, 2017-21, 2022-now AND ranking score >= ma50's AND median deepest loss no worse AND "
         "placebo share (weekly series circularly shifted >= 26 weeks, 200 draws, full ranking score incl. ordinary person "
         "with 50 simulations per clock) <= 10%. Lead/lag of weekly changes at -8..+8 weeks: report only."),
        ("NFCI 是什麼：芝加哥聯準銀行把 105 個美國貨幣市場、債市、股市和銀行體系的數字合成一個每週指數，0 是 1971 年以來的平均，"
         "正數代表比平均緊（借錢難、風險溢價高），負數代表比平均鬆。這裡用「低於自己最近 52 週平均」看的是方向（比去年這段期間鬆或緊），"
         "「低於 0」看的是和長期平均比。"),
        ("修正的問題：NFCI 每週發布時都會用新資料重新估計，過去的數字也會跟著改（芝加哥聯準銀行自己有公布修正表）。"
         "這裡用的是今天的版本，當時真的看到的數字可能不同，所以結果可能比當時真的能做到的好看一些；沒有取得免金鑰的歷史版本來核對。"),
        (f"資料來源：{data['where']}（{_d(raw.index[0])} 到 {_d(raw.index[-1])}，{len(raw):,} 週）。芝加哥聯準銀行舊網址 "
         "www.chicagofed.org/-/media/publications/nfci/nfci-data-series-csv.csv 只更新到 2026 年 4 月，所以改用它 NFCI 頁面現在連的 "
         "api.data.chicagofed.org 檔案；FRED 是備用。時間對齊：每週 NFCI 以那週星期五標日期、下週三公布，這裡保守地只用至少 6 天前的那週。"
         f"偷看未來的檢查：三個月底之後把價格和那時還不能用的 NFCI 換成亂數，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("「誰先動」怎麼讀：NFCI 裡面本來就有股市波動、信用利差這些每天的市場價格，所以和比特幣同一週一起動是預料中的；"
         "要支持「金融情勢先緊、比特幣後跌」，應該看到 k 為正（NFCI 先變）的相關明顯為負。相鄰幾週的相關不是互相獨立的，"
         "±2/√週數只是粗略的範圍，17 個 k 裡偶爾有一兩個超過也可能是巧合。"
         f"另外，NFCI 每週的變化本身非常平滑：這週和下週變化的相關高達 {la['ac'][0]:+.2f}（隔兩週 {la['ac'][1]:+.2f}、三週 {la['ac'][2]:+.2f}），"
         "所以同一週的負相關會原封不動地往前後各攤開兩三週，k 為正和為負的格子看起來差不多，這張表本身分不出誰先誰後。"),
        ("其他幣、MSTR 用的是同一個美國金融情勢訊號。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的一條規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "nfci_filter",
        "title": "美國金融情勢指數能不能幫 50 日均線",
        "question": ("有人說：美國的金融情勢先變緊（借錢變難、風險溢價變高），股票、比特幣這類風險資產才跟著跌。"
                     "芝加哥聯準銀行每週公布的金融情勢指數（NFCI）如果比自己過去一年的平均還緊，就不持有比特幣，"
                     "能不能讓 50 日均線少虧、通過完整的公平考試？而且到底是金融情勢先動，還是比特幣先動？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5],
        "notes": notes,
        "sources": [
            {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": CHI_URL},
            {"name": "芝加哥聯準銀行 NFCI 頁面", "url": CHI_PAGE},
            {"name": "FRED：NFCI（備用）", "url": FRED_URL},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
