"""Historical study "main_signal_check": the page's main signal (55-day closing high in / 20-day closing low out) was
quoted from 2018-01-01, two weeks after the 2017 top. Which single on/off trend rule holds up when the start date,
the daily closing hour, ordinary-person execution and the asset are all changed at the same time - and what honest
numbers should the page quote?

Candidates, fixed before running (no tuning): 55/20 breakout; close above the 50-day average; "thresh" (all in when
at least 9 of the 17 rules of trend_ensemble say long, else cash); close above the 100-day average; 60-day return
positive; and for reference the fractional 17-rule blend and the blend x 40% volatility target.

1. Bitcoin, the dashboard's daily closes (00:00 UTC, te._daily_close), 108 monthly start dates 2015-01 ... 2023-12,
   textbook execution (decide at the close, hold the next day, 0.10% of the amount traded).
2. Bitcoin, 72 monthly start dates 2018-01 ... 2023-12 x 24 closing hours, daily bars rebuilt from Binance 1-hour
   candles exactly as clock_robustness does (hourly data starts 2017-08, so clock tests start in 2018).
3. The same 72 x 24 cells, ordinary person: decide at the clock's close, trade 10 hours later (the shape of "check at
   22:00 Taiwan, act at 08:00"), 0.25% cost, 10% of entries missed at random (robust_trend._missed), 200 simulations
   per clock (fixed seed).
4. ETH, SOL, BNB, XRP (Binance daily) and MSTR (Yahoo, from 2020-08-11), monthly start dates from the first date on
   which every rule has a value to three years before today, textbook execution.

Every result of a start date is measured from that date to the last finished day; positions come from the full
history (the rules are already warm on the start date). No look-ahead: a signal uses closes up to and including the
decision close; random draws belong to days. Only public key-less endpoints through the dashboard's helpers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import signals as sig
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST = 0.001
P_COST, P_MISS = 0.0025, 0.10
SIMS = 200
SEED = 20150101
YEAR = 365
SHARE_YEAR = 252
DAILY_STARTS = ("2015-01-01", "2023-12-01")
CLOCK_STARTS = ("2018-01-01", "2023-12-01")
PERSON_CLOCK = rt.PERSON_CLOCK  # 14:00 UTC = 22:00 Taiwan
LOSS_BAR = 0.95
VOL_DAYS, VOL_TARGET = 30, 0.40

CANDS = ("b5520", "ma50", "thresh", "ma100", "mom60", "blend", "vol40")
ONOFF = ("b5520", "ma50", "thresh", "ma100", "mom60")
SHORT = {"b5520": "55／20", "ma50": "50 日均線", "thresh": "17 條過半", "ma100": "100 日均線", "mom60": "60 天漲跌",
         "blend": "17 條混合（參考）", "vol40": "混合×波動 40%（參考）"}
NAMES = {"b5520": "55 日新高進、20 日新低出（55／20，目前主訊號）", "ma50": "收盤高於 50 日均線才持有",
         "thresh": "17 條規則至少 9 條看多就全部持有", "ma100": "收盤高於 100 日均線才持有",
         "mom60": "過去 60 天是漲的才持有", "blend": "17 條混合，看多幾成持有幾成（參考）",
         "vol40": "17 條混合 × 波動調整 40%（參考）"}
OTHER = (("ETHUSDT", "ETH", None), ("SOLUSDT", "SOL", None), ("BNBUSDT", "BNB", None), ("XRPUSDT", "XRP", None),
         ("MSTR", "MSTR", "2020-08-11"))

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(x: float) -> str:
    return "—" if not _ok(x) else f"{x * 100:.0f}%"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- rules ----------

def _targets(close: pd.Series, year: int = YEAR) -> pd.DataFrame:
    """What each candidate wants to hold after each close (NaN while a rule has no value yet)."""
    signal, _ = te._signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    vol = close.pct_change().rolling(VOL_DAYS, min_periods=VOL_DAYS).std(ddof=1) * math.sqrt(year)
    scale = (VOL_TARGET / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
    high = close.rolling(55, min_periods=55).max()
    breakout = pd.Series(cr._breakout(close, 55, 20), index=close.index).where(high.notna())
    return pd.DataFrame({"b5520": breakout, "ma50": signal["A50"], "thresh": (share >= 0.5).astype(float).where(share.notna()),
                         "ma100": signal["A100"], "mom60": signal["D60"], "blend": share, "vol40": share * scale},
                        index=close.index)


# ---------- measuring from many start dates at once ----------

def _suffix(equity: np.ndarray, starts: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray]:
    """equity[d] = money before day d (equity[0] = 1, last row = after the last day); one column per path.
    For each start s: yearly return % from s to the end and deepest loss % inside that span. Shapes (starts, paths).

    Deepest loss of the span s..end = min(deepest of s+1..end, min(equity[s..end]) / equity[s] - 1)."""
    e = equity.reshape(len(equity), -1)
    n = len(e) - 1
    future_low = np.minimum.accumulate(e[::-1], axis=0)[::-1]
    deep = np.minimum.accumulate((future_low / e - 1)[::-1], axis=0)[::-1]
    days = (n - starts)[:, None]
    ret = ((e[-1][None, :] / e[starts]) ** (year / days) - 1) * 100
    return ret, deep[starts] * 100


def _equity(net: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], np.cumprod(1 + net)])


def _month_starts(index: pd.DatetimeIndex, first, last) -> tuple[np.ndarray, list]:
    """Positions in index of the first day on or after the 1st of every month from first to last."""
    pos, labels = [], []
    for month in pd.date_range(pd.Timestamp(first).to_period("M").to_timestamp(), pd.Timestamp(last), freq="MS"):
        day = max(month, pd.Timestamp(first))
        i = int(index.searchsorted(day))
        if i < len(index) - 30 and (not pos or i > pos[-1]):
            pos.append(i)
            labels.append(index[i])
    return np.array(pos, dtype=int), labels


def _daily(close: pd.Series, year: int, first, last) -> dict:
    """Textbook runs on one daily series from monthly start dates: ret/deep per start for every candidate and holding."""
    target = _targets(close, year)
    change = close.pct_change().fillna(0.0).values
    valid = target.notna().all(axis=1)
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = _month_starts(close.index, begin, last)
    out = {"starts": labels, "target": target}
    out["hold"] = _suffix(_equity(change), starts, year)
    for k in CANDS:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = _suffix(_equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    out["hold"] = (out["hold"][0][:, 0], out["hold"][1][:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray) -> dict:
    """72 start dates x 24 clocks, textbook and ordinary person."""
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = _month_starts(m.dates, CLOCK_STARTS[0], CLOCK_STARTS[1])
    hold_r, hold_d = _suffix(_equity(m.change), starts, YEAR)
    hold_r, hold_d = hold_r[:, 0], hold_d[:, 0]
    text = {k: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for k in CANDS}
    person = {k: (np.zeros((len(starts), 24, SIMS)), np.zeros((len(starts), 24, SIMS))) for k in CANDS}
    for h in range(24):
        closes = pd.Series(np.concatenate([pre, price[h::24]]))
        target = _targets(closes).fillna(0.0)
        for ki, k in enumerate(CANDS):
            w = target[k].values[npre:][:days].astype(float)
            r, dd = _suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            mid = eng._mid(b, pos, P_COST, None)
            r, dd = _suffix(np.vstack([np.ones((1, SIMS)), mid]), starts, YEAR)
            person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
    return {"starts": labels, "hold": (hold_r, hold_d), "text": text, "person": person, "last": m.dates[-1]}


# ---------- the whole computation ----------

def _load_other() -> tuple[dict, list[str]]:
    data, problems = {}, []
    for symbol, key, since in OTHER:
        try:
            close = ta._crypto_close(symbol) if symbol.endswith("USDT") else ta._share_close(symbol)
            data[key] = (close, since, SHARE_YEAR if since else YEAR)
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            problems.append(f"{key} 這次取不到資料（{type(exc).__name__}），沒有算進去。")
    return data, problems


def _trade_stats(close: pd.Series, target: pd.DataFrame) -> dict:
    """Trades per year, win rate, average win / loss on the daily series from the first start date (binary rules);
    yearly turnover for the fractional ones."""
    out = {}
    for k in CANDS:
        rec = sig._record(close, target[k], pd.Timestamp(DAILY_STARTS[0]), YEAR, k in ONOFF)
        out[k] = rec
    return out


def _compute() -> dict:
    close = te._daily_close()
    daily = _daily(close, YEAR, DAILY_STARTS[0], DAILY_STARTS[1])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre)
    other_data, problems = _load_other()
    if problem:
        problems.insert(0, problem)
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key, (c, since, year) in other_data.items():
        try:
            other[key] = _daily(c, year, since or c.index[0], three)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
        except Exception as exc:  # noqa: BLE001
            other.pop(key, None)
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")
    trades = _trade_stats(close, daily["target"])
    check = {k: sig._record(close, daily["target"][k], pd.Timestamp("2018-01-01"), YEAR, False) for k in CANDS}
    check["hold"] = check["b5520"]
    return {"close": close, "daily": daily, "clock": clock, "other": other, "problems": problems, "trades": trades,
            "check": check}


# ---------- summaries ----------

def _summary(r: dict) -> dict:
    d, c, o = r["daily"], r["clock"], r["other"]
    hr, hd = c["hold"]
    s = {}
    for k in CANDS:
        dr, dd = d[k]
        edge_d = dr - d["hold"][0]
        tr, td = c["text"][k]
        edge = tr - hr[:, None]
        pr, pd_ = c["person"][k]
        pedge = pr - hr[:, None, None]
        n_other = sum(len(o[a]["starts"]) for a in o)
        beat_other = sum(int((o[a][k][0] > o[a]["hold"][0]).sum()) for a in o)
        shallow_other = sum(int((o[a][k][1] > o[a]["hold"][1]).sum()) for a in o)
        s[k] = {
            "d_n": len(dr), "d_beat": int((edge_d > 0).sum()), "d_edge": float(np.median(edge_d)),
            "d_shallow": int((dd > d["hold"][1]).sum()), "d_deep": (float(dd.max()), float(dd.min())),
            "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
            "c_p10": float(np.percentile(edge, 10)), "c_p90": float(np.percentile(edge, 90)),
            "c_min": float(edge.min()), "c_max": float(edge.max()),
            "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
            "c_deep_worst": float(td.min()), "c_deep_best": float(td.max()),
            "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
            "p_p10": float(np.percentile(pedge, 10)), "p_shallow": int((pd_ > hd[:, None, None]).sum()),
            "p_deep_med": float(np.median(pd_)), "p_deep_p10": float(np.percentile(pd_, 10)),
            "p_clock_n": pedge[:, PERSON_CLOCK, :].size, "p_clock_beat": int((pedge[:, PERSON_CLOCK, :] > 0).sum()),
            "o_n": n_other, "o_beat": beat_other, "o_shallow": shallow_other,
            "o_each": {a: (len(o[a]["starts"]), int((o[a][k][0] > o[a]["hold"][0]).sum()),
                           int((o[a][k][1] > o[a]["hold"][1]).sum())) for a in o},
        }
        x = s[k]
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
    eligible = [k for k in CANDS if s[k]["loss_ok"] >= LOSS_BAR]
    order = sorted(CANDS, key=lambda k: (k not in eligible, -s[k]["low"], -s[k]["c_edge"]))
    loss_order = sorted(CANDS, key=lambda k: -s[k]["c_deep_med"])
    return {"s": s, "eligible": eligible, "order": order, "winner": order[0] if eligible else None,
            "loss_winner": loss_order[0], "hold_deep": (float(hd.max()), float(hd.min())),
            "hold_deep_med": float(np.median(hd))}


def _quote(k: str, x: dict, hold_deep: tuple) -> str:
    return (f"每年報酬比一直抱著多（中位數）{x['c_edge']:+.1f} 個百分點，起算日／收盤時間不同時在 {x['c_p10']:+.1f} 到 "
            f"{x['c_p90']:+.1f} 之間；最深虧損約 {x['c_deep_best']:.0f}% 到 {x['c_deep_worst']:.0f}%"
            f"（一直抱著 {hold_deep[0]:.0f}% 到 {hold_deep[1]:.0f}%）")


# ---------- today ----------

def _blend_count(close: pd.Series, signal: pd.DataFrame, x: float) -> int:
    """How many of the 17 rules would be long if the next close were x."""
    flips = sig._flip_prices(close, signal)
    n = 0
    for col, (level, long) in flips.items():
        if col[0] == "C":
            n += (x > level) if long else (x >= level)  # long stays long unless x <= low; cash turns long at x >= high
        else:
            n += x > level
    return int(n)


def _now(close: pd.Series, target: pd.DataFrame) -> list[dict]:
    c = close.values
    last = float(c[-1])
    signal, _ = te._signals(close)
    flips = sig._flip_prices(close, signal)
    long_n = sum(1 for _, (lv, lg) in flips.items() if lg)

    def word(v) -> str:
        if not _ok(v):
            return "—"
        return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"

    def level(lv: float, long: bool) -> str:
        return (f"收盤 {'≤' if long else '≥'} {lv:,.0f} 美元（離最後收盤 {(lv / last - 1) * 100:+.1f}%）轉成"
                f"{'現金' if long else '持有'}")

    t = target.iloc[-1]
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"}]
    lv = float(c[-19:].min() if t["b5520"] >= 0.5 else c[-54:].max())
    out.append({"label": SHORT["b5520"], "value": f"{word(t['b5520'])}；{level(lv, t['b5520'] >= 0.5)}"})
    for k, n in (("ma50", 50), ("ma100", 100)):
        lv = float(c[-(n - 1):].sum() / (n - 1))
        out.append({"label": SHORT[k], "value": f"{word(t[k])}；{level(lv, t[k] >= 0.5)}"})
    lv = float(c[-60])
    out.append({"label": SHORT["mom60"], "value": f"{word(t['mom60'])}；{level(lv, t['mom60'] >= 0.5)}"})
    levels = sorted(v[0] for v in flips.values())
    is_long = long_n >= 9
    cut = None
    if is_long:
        for lvl in reversed(levels):
            if _blend_count(close, signal, lvl * (1 - 1e-9)) < 9:
                cut = lvl
                break
    else:
        for lvl in levels:
            if _blend_count(close, signal, lvl * (1 + 1e-9)) >= 9:
                cut = lvl
                break
    thresh_text = f"17 條裡 {long_n} 條看多，{word(t['thresh'])}"
    if cut is not None:
        thresh_text += f"；大約在{level(cut, is_long)}"
    out.append({"label": SHORT["thresh"], "value": thresh_text})
    out.append({"label": "17 條混合／混合×波動 40%（參考）", "value": f"{word(t['blend'])}／{word(t['vol40'])}"})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態、收盤到哪裡會翻轉，不是買賣建議"})
    return out


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    sm = _summary(r)
    s, d, c, o = sm["s"], r["daily"], r["clock"], r["other"]
    close = r["close"]
    n_d, n_starts = len(d["starts"]), len(c["starts"])
    n_c = n_starts * 24
    tr = r["trades"]

    def trade_cells(k: str) -> list:
        x = tr[k]
        if k in ONOFF:
            rate = f"{x['wins'] / x['trades'] * 100:.0f}%" if x.get("trades") else "—"
            win = _pct(x.get("avg_win")) if x.get("avg_win") is not None else "—"
            loss = _pct(x.get("avg_loss")) if x.get("avg_loss") is not None else "—"
            return [f"{x['trades_per_year']:.1f} 次｜{rate}", f"{win}｜{loss}"]
        return [f"一年換手 {x['turnover']:.1f} 次（分批進出）", "—"]

    rows1 = []
    for k in CANDS:
        x = s[k]
        rows1.append([NAMES[k], f"{_of(x['d_n'], x['d_beat'])}｜{_pp(x['d_edge'])}", _of(x["c_n"], x["c_beat"]),
                      f"{_pp(x['c_edge'])}｜{_pp(x['c_p10'])}", _of(x["c_n"], x["c_shallow"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}"] + trade_cells(k))
    hd = c["hold"][1]
    rows1.append(["對照：一直抱著不動", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜{_pct(float(hd.min()))}", "—", "—"])

    rows2 = []
    for k in CANDS:
        x = s[k]
        rows2.append([NAMES[k], f"{_of(x['p_n'], x['p_beat'])}（{_share(x['p_beat'] / x['p_n'])}）",
                      f"{_pp(x['p_edge'])}｜{_pp(x['p_p10'])}",
                      f"{_share(x['p_shallow'] / x['p_n'])}", f"{_pct(x['p_deep_med'])}｜{_pct(x['p_deep_p10'])}",
                      f"{_of(x['p_clock_n'], x['p_clock_beat'])}（{_share(x['p_clock_beat'] / x['p_clock_n'])}）",
                      _pp(x["c_edge"])])

    assets = list(o)
    rows3 = [["起算日範圍（幾個）"] + [f"{_d(o[a]['starts'][0])[:7]} 到 {_d(o[a]['starts'][-1])[:7]}（{len(o[a]['starts'])} 個）"
                                for a in assets] + [f"共 {sum(len(o[a]['starts']) for a in assets)} 個"]]
    rows3.append(["一直抱著：每年報酬中位數｜最深虧損中位數"]
                 + [f"{_pct(float(np.median(o[a]['hold'][0])))}｜{_pct(float(np.median(o[a]['hold'][1])))}" for a in assets]
                 + ["—"])
    for k in CANDS:
        x = s[k]
        rows3.append([NAMES[k]] + [f"報酬 {_of(x['o_each'][a][0], x['o_each'][a][1])}｜虧損 {x['o_each'][a][2]} 個" for a in assets]
                     + [f"報酬 {_of(x['o_n'], x['o_beat'])}｜虧損 {x['o_shallow']} 個"])

    rows4 = []
    for rank, k in enumerate(sm["order"], 1):
        x = s[k]
        ok = x["loss_ok"] >= LOSS_BAR
        rows4.append([NAMES[k]] + [_share(v) for v in x["checks"]]
                     + [_share(x["low"]), f"{_share(x['loss_ok'])}{'' if ok else '（未達 95%）'}",
                        f"第 {rank}" if ok else "不列入排名"])

    tables = [
        {"caption": (f"比特幣、照課本做（收盤決定、隔天照做、每次買賣扣 {COST * 100:.2f}%）。第二欄是用儀表板的日線（UTC 0 點收盤）"
                     f"從 {DAILY_STARTS[0][:7]} 到 {DAILY_STARTS[1][:7]} 每個月 1 日起算（{n_d} 個起算日），算到 {_d(close.index[-1])}，"
                     "「報酬贏」= 每年報酬比同一天起一直抱著高，後面的數字是贏多少個百分點的中位數。第三到第六欄改用 1 小時線拼出 24 個收盤時間，"
                     f"1 小時線 2017-08 才有，所以起算日只能從 {CLOCK_STARTS[0][:7]} 到 {CLOCK_STARTS[1][:7]}（{n_starts} 個 × 24 個收盤時間 = "
                     f"{n_c:,} 格）。「最差一成」是 {n_c:,} 格裡排在後 10% 的那一格。最後兩欄是 UTC 0 點日線從 {DAILY_STARTS[0]} 算起的每筆交易"
                     "（已結束的），兩種混合是分批進出，改列換手次數。"),
         "columns": ["規則", f"起算日 {n_d} 個：報酬贏過一直抱著｜差距中位數", f"起算日×收盤時間 {n_c:,} 格：報酬贏的",
                     "差距（百分點）：中位數｜最差一成", "最深虧損比一直抱著淺的格", "最深虧損：中位數｜最深的一格",
                     "一年交易幾次｜勝率", "平均每筆賺｜平均每筆賠"],
         "rows": rows1},
        {"caption": (f"同樣 {n_c:,} 格，改成一般人的做法：收盤後 10 小時才照做（台灣晚上 10 點看、隔天早上 8 點做的樣子）、每次買賣扣 "
                     f"{P_COST * 100:.2f}%、每次要進場時有 {P_MISS * 100:.0f}% 的機會漏掉（漏掉就等下一次）。每個收盤時間各模擬 {SIMS} 次，"
                     f"所以每條規則共 {n_c * SIMS:,} 次；和同一天起一直抱著（不扣成本）比。倒數第二欄只看「台灣晚上 10 點看」那一個收盤時間。"
                     f"模擬次數比 robust_trend 的 1,000 次少，是為了讓整頁在幾分鐘內算完，比例的誤差約 1 個百分點以內。"),
         "columns": ["規則", "報酬贏過一直抱著的模擬", "差距（百分點）：中位數｜最差一成", "最深虧損比一直抱著淺的比例",
                     "最深虧損：中位數｜最差一成", "只看台灣晚上 10 點那個時鐘：報酬贏的", "對照：照課本做的差距中位數"],
         "rows": rows2},
        {"caption": ("換資產：同樣的規則原封不動用在其他幣（幣安現貨日線，UTC 0 點收盤）和 MSTR（Yahoo 調整後收盤，只用 2020-08-11 公司開始買比特幣以後），"
                     "從每條規則都有值的第一個月起、每個月 1 日起算，到三年前為止，照課本做、成本 0.10%。每格是「報酬贏過一直抱著的起算日個數｜"
                     "最深虧損比一直抱著淺的個數」，分母是第一列的起算日個數。"),
         "columns": ["規則"] + assets + ["合計"],
         "rows": rows3},
        {"caption": ("排名規則（看結果之前就定好）：四項檢查各算「報酬贏過一直抱著」的比例——比特幣 108 個起算日、比特幣起算日×收盤時間、"
                     "照一般人方式做、其他資產——取四項裡最低的那一項比大小，最低的那項最高的就是「最可靠」；但前提是最深虧損比一直抱著淺的比例，"
                     f"在比特幣兩組（起算日、起算日×收盤時間）裡較低的那組也至少 {LOSS_BAR * 100:.0f}%。同分時比起算日×收盤時間的差距中位數。"),
         "columns": ["規則", f"比特幣起算日（{n_d} 個）", f"比特幣起算日×收盤時間（{n_c:,} 格）", "照一般人方式做",
                     "其他資產", "四項裡最低", "最深虧損比抱著淺（兩組裡較低）", "排名"],
         "rows": rows4},
    ]

    win = sm["winner"]
    lw = sm["loss_winner"]
    b = s["b5520"]
    chk = r["check"]["b5520"]
    hold_deep = sm["hold_deep"]
    quotes = {k: _quote(k, s[k], hold_deep) for k in ([win] if win else []) + (["b5520"] if win != "b5520" else [])}

    lw_worst = max(CANDS, key=lambda k: s[k]["c_deep_worst"])
    if win:
        w = s[win]
        first = (f"換起算日、換收盤時間、照一般人方式做、換幣四項一起考，最可靠的是「{NAMES[win].split('（')[0]}」：最差的一項也有 "
                 f"{_share(w['low'])} 的情況報酬贏過一直抱著，比特幣 {n_c:,} 格每年多 {w['c_edge']:.1f} 個百分點（中間八成 "
                 f"{w['c_p10']:+.1f} 到 {w['c_p90']:+.1f}），一般人照做 {_share(w['p_beat'] / w['p_n'])} 贏，但最深虧損 "
                 f"{w['c_deep_best']:.0f}% 到 {w['c_deep_worst']:.0f}%。")
    else:
        first = f"七條規則沒有一條在比特幣兩組裡都有 {LOSS_BAR * 100:.0f}% 以上的情況虧得比一直抱著淺，所以沒有選出最可靠的一條。"
    quoted = chk["ret"] - chk["hold_ret"]
    second = (f"55／20 原本引用 2018-01-01 起每年 {_pct(chk['ret'])}（比抱著多 {quoted:.1f} 點）、最深 {_pct(chk['deep'])}，"
              f"換起算日和收盤時間後中位數只多 {b['c_edge']:+.1f} 點（{n_c:,} 格裡 {b['c_beat']:,} 格贏），一般人照做只有 "
              f"{_share(b['p_beat'] / b['p_n'])} 贏，原數字被好起點美化了約 {quoted - b['c_edge']:.0f} 點。")
    if lw_worst == lw:
        third = f"只看少虧，「{SHORT[lw]}」最深虧損最淺（中位數 {_pct(s[lw]['c_deep_med'])}，抱著 {_pct(sm['hold_deep_med'])}）。"
    else:
        third = (f"只看少虧，「{SHORT[lw]}」最深虧損中位數最淺（{_pct(s[lw]['c_deep_med'])}，抱著 {_pct(sm['hold_deep_med'])}），"
                 f"最深的一格則是「{SHORT[lw_worst]}」最淺（{_pct(s[lw_worst]['c_deep_worst'])}）。")
    verdict = first + second + third

    notes = []
    for k, q in quotes.items():
        notes.append(f"頁面可以引用的誠實數字（{SHORT[k]}，比特幣 {CLOCK_STARTS[0][:7]}–{CLOCK_STARTS[1][:7]} 起算 × 24 個收盤時間、照課本做）："
                     f"「{q}」。報酬差距的範圍是 {n_c:,} 格裡中間八成（最差、最好的一格分別是 {s[k]['c_min']:+.1f} 和 {s[k]['c_max']:+.1f} 個百分點），"
                     f"最深虧損的範圍是全部 {n_c:,} 格從最淺到最深；照一般人方式做，差距中位數是 {s[k]['p_edge']:+.1f}、"
                     f"報酬贏的模擬占 {_share(s[k]['p_beat'] / s[k]['p_n'])}。")
    notes += [
        f"核對：用儀表板日線、UTC 0 點、從 2018-01-01 起照課本做，55／20 每年 {_pct(chk['ret'])}、最深 {_pct(chk['deep'])}；"
        f"50 日均線 {_pct(r['check']['ma50']['ret'])}、{_pct(r['check']['ma50']['deep'])}；17 條過半 "
        f"{_pct(r['check']['thresh']['ret'])}、{_pct(r['check']['thresh']['deep'])}；一直抱著 {_pct(chk['hold_ret'])}、{_pct(chk['hold_deep'])}。",
        f"只看「少虧」怎麼排：{n_c:,} 格最深虧損中位數由淺到深是 "
        + "、".join(f"{SHORT[k]} {_pct(s[k]['c_deep_med'])}" for k in sorted(CANDS, key=lambda k: -s[k]['c_deep_med']))
        + f"；一直抱著 {_pct(sm['hold_deep_med'])}。",
        "七條候選規則是在知道「順勢規則在比特幣上表現不錯」之後才挑的，雖然這次沒有調任何參數，事後才知道的成分還在，數字可能比真正事先能做到的好看。",
        "1 小時線從 2017-08 才有，所以換收盤時間、照一般人方式做的檢查只能從 2018-01 起算；2015–2017 起算的 108 個起算日只用 UTC 0 點日線"
        "（2017-08 以前是 blockchain.info 的每日價格）。比特幣在每個收盤時間的序列前面接上 2017-08 以前的日線暖機。",
        "相鄰的起算日共用大部分的歷史（例如 2018-01 和 2018-02 起算，後面八年多完全一樣），24 個收盤時間也只差幾個小時，所以這些「幾個裡幾個」"
        "不是互相獨立的證據，只能說明結果對起點和時間有多敏感；越晚的起算日涵蓋的年數越短，也越容易被最近一兩年左右。",
        "成本很粗略：照課本做扣 0.10%、一般人做法扣 0.25%，成交價用整點收盤價、沒有另外算滑價；現金沒有利息，也沒有算稅。"
        "一直抱著不扣成本。股票（MSTR）一年用 252 個交易日換算。",
        "其他資產是今天還在的大幣和一檔股票，有倖存者偏誤；各自的起算日範圍不同，彼此的數字不能直接比大小。",
        f"隨機模擬用固定的亂數種子，資料沒變的話重跑結果一樣。每條規則 {n_c * SIMS:,} 次模擬是 {n_c:,} 格各 {SIMS} 次，格與格之間高度相關。",
        "過去的結果不保證未來會一樣。這裡只是比較幾條固定規則過去在不同條件下做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "main_signal_check",
        "title": "主訊號重新公平挑一次",
        "question": ("換起算日、換收盤時間、照一般人的方式做、換到其他幣，全部一起考的時候，哪一條單一的進出場順勢規則最可靠？"
                     "頁面應該引用什麼樣的誠實數字？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(close, d["target"]),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
