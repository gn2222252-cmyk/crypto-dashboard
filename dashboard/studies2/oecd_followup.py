"""Historical study "oecd_followup": the one lead business_cycle found only AFTER looking at its results, tested
alone with a rule and a pass test written down before this run.

Where the lead came from: in business_cycle (pre-registered main series = Philadelphia Fed survey, which failed),
the robustness row "ma50 AND OECD US leading indicator above its 10-month average" scored a ranking score of 81% vs
ma50's 80% with a shallower median deepest loss (-29.0% vs -53.7%) and favorable > unfavorable in all three periods,
but it had no placebo and was noticed after the fact. This cannot be truly out-of-sample: it is the same history and
the exam numbers are already known; the new information here is the placebo. The real test is the daily record from
now on (the `now` row "比特幣：50 日均線｜O2" is there so it can be logged every day).

Pre-registered (fixed before this run; no other variants, no parameter changes):
  O2 rule: hold BTC when main_signal_check's ma50 is on AND the OECD US composite leading indicator (amplitude
  adjusted) is above the average of its last 10 monthly values (including itself), using at each month-end the value
  for the month two months earlier, decided at the last daily close of the month (00:00 UTC, te._daily_close) and
  held for the next month - exactly business_cycle's M2 OECD (its loaders, cache, _flags, _Exam, seeds).
  Pass test - all four:
    (1) favorable > unfavorable average next-month BTC return in 2011-16, 2017-21 and 2022-now (business_cycle's
        event table);
    (2) ranking score >= ma50's on the same starts (main_signal_check's arithmetic incl. the 95% loss eligibility);
    (3) median deepest loss over the 1,728 textbook cells no worse than ma50's;
    (4) placebo share <= 10%: the monthly OECD series circularly shifted by 6 ... (length - 6) months, 200 draws,
        the whole exam rebuilt with the full ranking score including the ordinary-person check; to stay within about
        ten minutes the ordinary-person part uses 50 simulations per clock in the placebo (same seeds) and the real
        rule is re-scored the same way for this comparison (the reduced version, stated).
  Report only (not part of the test): days on which O2 differs from ma50 and calendar-year returns (which years the
  edge comes from), both on the UTC daily series from 2015-01-01; the same rule with ONE EXTRA month of lag (value
  three months earlier); the OECD revision caveat (the indicator is revised every month and only today's vintage is
  available key-less). Today: states only.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc

LAG = bc.LAG["oecd"]
DRAWS, GAP, PL_SIMS = bc.DRAWS, bc.GAP, bc.PL_SIMS
FIRST = msc.DAILY_STARTS[0]

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


def _load() -> dict:
    problems = []
    series, where, note = bc._series("oecd")
    if note:
        problems.append(note)
    other, more = msc._load_other()
    problems += more
    return {"close": bc.te._daily_close(), "series": {"oecd": series}, "where": where, "other": other,
            "problems": problems, "fred_down": bc._RUN["fred_down"]}


def _compute(data: dict) -> dict:
    t0 = time.time()
    close, raw = data["close"], data["series"]["oecd"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    F = bc._flags(raw, LAG)
    F_lag = bc._flags(raw, LAG + 1)
    rng = np.random.default_rng(bc.SEED + 5)
    ev = bc._events(close, F, rng)
    ev_lag = bc._events(close, F_lag, rng)
    s = {"ma50": ex.summary("ma50", None), "o2": ex.summary("m2", F), "o2_lag": ex.summary("m2", F_lag)}

    small = {"ma50": ex.summary("ma50", None, PL_SIMS), "o2": ex.summary("m2", F, PL_SIMS)}
    prng = np.random.default_rng(bc.SEED + 31)
    n = len(raw)
    hits, lows, beat_ma = 0, [], 0
    for sh in prng.integers(GAP, n - GAP + 1, DRAWS):
        s2 = pd.Series(np.roll(raw.values, int(sh)), index=raw.index)
        x = ex.summary("m2", bc._flags(s2, LAG), PL_SIMS)
        lows.append(x["low"])
        hits += int(bc._score_ge(x, small["o2"]))
        beat_ma += int(bc._score_ge(x, small["ma50"]))

    pos = {"ma50": ex.btc_pos("ma50", None), "o2": ex.btc_pos("m2", F)}
    a = close.index >= pd.Timestamp(FIRST)
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["o2"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()), int(((p0 > 0.5) & (p1 < 0.5)).sum()))
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST),
             **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    look = bc._lookahead(data, ex)
    return {"ex": ex, "F": F, "ev": ev, "ev_lag": ev_lag, "s": s, "small": small, "plc": hits / DRAWS,
            "plc_low": float(np.median(lows)), "plc_ma": beat_ma / DRAWS, "pos": pos, "differ": differ,
            "years": years, "look": look, "problem": problem, "secs": time.time() - t0}


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base, o2, ol = s["ma50"], s["o2"], s["o2_lag"]
    n_c, n_d = base["c_n"], base["d_n"]

    q = {"event": ev["all_three"], "score": bc._score_ge(o2, base),
         "loss": o2["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["plc"] <= 0.10}
    q["pass"] = all(q.values())

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    # ----- table 1: event table -----
    rows1 = []
    for lab, e in (("O2 的景氣條件（兩個月前的數字）", ev), ("多等一個月（三個月前的數字，只報告）", r["ev_lag"])):
        a = e["all"]
        ci = a.get("ci")
        rows1.append([lab, f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()]
                     + ["是" if e["all_three"] else "否"])
    t1 = {"caption": (f"每個月最後一根日線（UTC 0 點收盤）決定：OECD 美國領先指標（振幅調整）兩個月前的數字高於它自己最近 {bc.AVG} 個月平均，"
                      f"就算景氣「有利」。每格是下一個月比特幣報酬（月底收盤到月底收盤）的平均，持有月份 {ev['first']} 到 {ev['last']}；"
                      f"差距＝有利減不利，括號是區塊拔靴法（連續 {bc.BLOCK} 個月一塊、{bc.BOOT:,} 次）的 90% 區間；分段欄括號是（有利｜不利）月數。"
                      "這張表只看景氣條件本身，不含 50 日均線。"),
          "columns": ["設定", "有利月：個數｜平均報酬", "不利月：個數｜平均報酬", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in bc.PERIODS] + ["三段都是有利較好"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    rows2 = []
    for lab, x in (("50 日均線（基準）", base), ("O2：50 日均線而且 OECD 領先指標有利", o2),
                   ("O2 多等一個月（只報告）", ol)):
        ck = x["checks"]
        rows2.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    t2 = {"caption": (f"main_signal_check 的公平考試（四項都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產（ETH、SOL、BNB、XRP、MSTR）"
                      "用自己的 50 日均線、同一個美國景氣訊號。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      "最後兩欄是照課本｜一般人做法的中位數。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: pass test -----
    ps = [p["diff"] for p in ev["periods"].values()]
    rows3 = [["有利月比不利月好（三段）", "、".join(_p(v) for v in ps), yes(q["event"])],
             ["排名分數不低於 50 日均線", f"{_share(o2['low'])} 對 {_share(base['low'])}", yes(q["score"])],
             [f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(o2['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(q["loss"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['small']['o2']['low'])}；"
              f"打亂後不輸均線的 {_share(r['plc_ma'])}）", yes(q["plc"])],
             ["結果", "—", "通過" if q["pass"] else "沒有通過"]]
    t3 = {"caption": (f"事先寫好的過關標準，四項都要過。隨機錯位：把 OECD 領先指標的每月數字整串循環平移 {GAP} 個月到「長度 − {GAP}」個月、"
                      f"{DRAWS} 次，整套考試（含一般人做法）重做；為了在十分鐘內算完，這一步一般人做法每個收盤時間只模擬 {PL_SIMS} 次"
                      "（種子相同），真的規則也用同樣方式算一次再比（簡化版）。"),
          "columns": ["條件", "數字", "結果"], "rows": rows3}

    # ----- table 4: years -----
    dd_, nn, out_ = r["differ"]
    rows4 = []
    yrs = sorted(r["years"]["ma50"])
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["o2"].get(y)
        rows4.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    rows4.append([f"和 50 日均線持有狀態不同的天數（{FIRST} 起）", "—", "—", f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—",
                  f"其中均線持有、O2 現金 {out_:,} 天"])
    t4 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看 O2 和 50 日均線的差來自哪幾年；"
                      "O2 只會比均線少持有（景氣不利時均線持有也改成現金）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "O2", "O2 減均線（百分點）"], "rows": rows4}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    raw = data["series"]["oecd"].dropna()
    avg = raw.rolling(bc.AVG, min_periods=bc.AVG).mean()
    used = cur - LAG
    fnow = F.get(cur, np.nan)
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"OECD 美國領先指標（{cur.strftime('%Y-%m')} 月底的決定，用 {used.strftime('%Y-%m')} 的數字，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": (f"{'有利' if fnow == 1 else '不利' if fnow == 0 else '—'}；{used.strftime('%Y-%m')} 是 {_num(raw.get(used, np.nan), 2)}，"
                   f"近 {bc.AVG} 個月平均 {_num(avg.get(used, np.nan), 2)}；最新公布的是 {raw.index[-1].strftime('%Y-%m')} 的 {_num(raw.iloc[-1], 2)}")},
        {"label": "比特幣：50 日均線｜O2", "value": f"{word(pm['ma50'][-1])}｜{word(pm['o2'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    a0 = ev["all"]
    better = [y for y in yrs if _ok(r["years"]["o2"].get(y)) and _ok(r["years"]["ma50"].get(y))
              and r["years"]["o2"][y] - r["years"]["ma50"][y] > 0.5]
    verdict = (f"{'通過' if q['pass'] else '沒有通過'}事先寫好的標準。O2（50 日均線而且 OECD 美國領先指標兩個月前的數字高於 10 個月平均）："
               f"景氣有利月下個月比特幣平均比不利月多 {_p(a0['diff'])}，三段 {'、'.join(_p(v) for v in ps)}（{yes(q['event'])}）；"
               f"排名分數 {_share(o2['low'])} 對均線 {_share(base['low'])}（{yes(q['score'])}）；最深虧損中位數 {_pct(o2['c_deep_med'])} "
               f"對 {_pct(base['c_deep_med'])}（{yes(q['loss'])}）；隨機錯位一樣好 {_share(r['plc'])}（{yes(q['plc'])}）。"
               f"多等一個月時排名分數 {_share(ol['low'])}、差距 {_p(r['ev_lag']['all']['diff'])}。"
               + (f"領先均線的年份只有 {'、'.join(str(y) for y in better)}（其餘年份持平或落後），"
                  "少虧主要靠避開這幾年的下跌。" if better else "") +
               "這條線索是看了結果才挑出來的、OECD 數字每月都會修正，即使通過也只能當候選，要看之後每天的實際紀錄。")

    tot, bad = r["look"]
    notes = [
        ("事先寫死的規則和標準（照原文）：O2 rule: hold BTC when ma50 is on AND the OECD US composite leading indicator "
         "(amplitude adjusted) is above the average of its last 10 monthly values, using at each month-end the value for the "
         "month two months earlier, held for the next month. Pass test: favorable > unfavorable in 2011-16, 2017-21, 2022-now "
         "AND ranking score >= ma50's AND median deepest loss no worse AND placebo share (circular shift >= 6 months, 200 draws, "
         "full ranking score incl. ordinary person with 50 simulations per clock) <= 10%. Extra month of lag: report only."),
        ("線索從哪裡來：business_cycle 研究事先登記的主要指標（費城製造業調查）沒有通過，OECD 領先指標加均線只是其中一列穩健度檢查，"
         "排名分數 81% 對均線 80%、最深虧損較淺，但沒有做隨機錯位、是看了結果才注意到的。這份研究只測這一條、不改任何參數；"
         "考試的數字本來就已經看過，真正新的只有隨機錯位，所以仍然不是樣本外。"),
        ("OECD 修正的問題：領先指標是把多個數列濾波後組成的，每個月公布時都會改寫前面幾個月（甚至更久）的數字，2023 年 OECD 也改過計算方法；"
         "這裡用的是今天的版本，當時真的看到的數字可能不同，而免金鑰的歷史版本（ALFRED）這次連不上，沒有辦法核對。"
         "所以這裡的結果可能比當時真的能做到的好看；多等一個月的敏感度只能部分反映這件事。"),
        (f"資料來源：{data['where']}。" + (f"FRED 這次對儀表板固定的 user agent 沒有回應（{data['fred_down']}），改用 OECD 自己的公開資料庫。"
                                         if data["fred_down"] else "")
         + "時間對齊：OECD 大約在下個月 10 日前後公布上個月，這裡保守地在月底只用兩個月前的數字。"
         f"偷看未來的檢查：三個月底之後把價格和那時還不能用的數字換成亂數，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("其他幣、MSTR 用的是同一個美國景氣訊號，所以第四項是在問「美國景氣能不能幫其他幣」。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，"
         f"「幾個裡幾個」不是互相獨立的證據。成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的一條規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "oecd_followup",
        "title": "OECD 領先指標加均線的事先登記複驗",
        "question": ("上一份景氣研究看了結果才注意到：50 日均線再加上「OECD 美國領先指標高於 10 個月平均」，排名分數略高於只用均線。"
                     "事先把規則和過關標準寫死、加上隨機錯位後，這條線索站得住嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4],
        "notes": notes,
        "sources": [
            {"name": "OECD 綜合領先指標（SDMX）", "url": bc.OECD_URL},
            {"name": "FRED：OECD 美國綜合領先指標", "url": bc.FRED_CSV.format(bc.FRED_ID["oecd"])},
            {"name": "SentimenTrader：Using manufacturing data as an input for Bitcoin（2025-03）",
             "url": "https://sentimentrader.com/blog/using-manufacturing-data-as-an-input-for-bitcoin"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
