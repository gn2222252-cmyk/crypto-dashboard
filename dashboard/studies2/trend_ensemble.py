"""Historical study "trend_ensemble": if picking the best trend parameter is luck, does holding a blend of many trend
rules - and scaling the amount held by how wild the market is - do better or more steadily than one hand-picked rule?

Seventeen daily bitcoin trend rules fixed in advance (close above an N-day average, N-day-high entry with an
N/2-day-low exit, L-day return positive). Variants: the share of rules that are long (blend), the blend with a 50%
threshold, the blend scaled by min(1, target / 30-day realised volatility), a long/short blend, and the benchmarks
buy-and-hold, two single rules and "half bitcoin, half cash, rebalanced monthly".

No look-ahead: every signal (and the volatility used for sizing) uses data up to and including the close of day t and
decides the position held during day t+1. Costs are charged on the amount actually traded, including the small trades
that keep a fractional position at its target after the price has moved. Only closed daily candles from public,
key-less endpoints through common.get_json.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic

COST = 0.001  # 0.10% of the amount traded
YEAR = 365  # bitcoin trades every day
EARLY = ("2013-01-01", "2017-12-31")
MAIN_START = "2018-01-01"
MA_DAYS = (10, 20, 30, 50, 75, 100, 150, 200)
CHANNEL = (10, 20, 50, 100)
PAST = (14, 30, 60, 90, 180)
VOL_DAYS = 30
TARGETS = (0.40, 0.60)
COSTS = (0.0, 0.001, 0.003, 0.005)
SIMS = 1000
SEED = 20180101
FAMILIES = {"A": "均線類", "C": "新高新低類", "D": "過去漲跌類"}

NAMES = {
    "hold": "一直抱著不動",
    "c20": "單一規則：創 20 日新高進場、10 日新低出場",
    "ma50": "單一規則：收盤高於 50 日均線才持有",
    "half": "一半比特幣一半現金，每月調回一半",
    "blend": "混合：照看多的規則比例持有",
    "thresh": "混合過半：過半規則看多就全部持有",
    "vol40": "混合 × 波動調整（目標 40%）",
    "vol60": "混合 × 波動調整（目標 60%）",
    "ls": "混合多空：看多比例低於一半就做空",
}
ORDER = ("hold", "c20", "ma50", "half", "blend", "thresh", "vol40", "vol60", "ls")


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed daily closes since 2010 on a DatetimeIndex with no missing day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


# ---------- the rules ----------

def _signals(close: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Signal at each day's close (1 = long, 0 = not long, NaN = not enough history yet) and each rule's family."""
    cols, family = {}, {}
    for n in MA_DAYS:
        mean = close.rolling(n, min_periods=n).mean()
        cols[f"A{n}"], family[f"A{n}"] = (close > mean).astype(float).where(mean.notna()), "A"
    for n in CHANNEL:
        m = n // 2
        high = close.rolling(n, min_periods=n).max()
        low = close.rolling(m, min_periods=m).min()
        state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
        cols[f"C{n}"] = state.where(high.notna()).ffill().fillna(0.0).where(high.notna())  # cash until a first high
        family[f"C{n}"] = "C"
    for n in PAST:
        past = close / close.shift(n) - 1
        cols[f"D{n}"], family[f"D{n}"] = (past > 0).astype(float).where(past.notna()), "D"
    return pd.DataFrame(cols), family


def _targets(close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """What each variant wants to hold, decided at each day's close (before any shift)."""
    signal, family = _signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))  # only once every rule has enough history
    vol = close.pct_change().rolling(VOL_DAYS, min_periods=VOL_DAYS).std(ddof=1) * math.sqrt(YEAR)
    out = pd.DataFrame(index=close.index)
    out["c20"] = signal["C20"]
    out["ma50"] = signal["A50"]
    out["blend"] = share
    out["thresh"] = (share >= 0.5).astype(float).where(share.notna())
    for target in TARGETS:
        scale = (target / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
        out[f"vol{int(target * 100)}"] = share * scale
    out["ls"] = 2 * share - 1
    for fam in FAMILIES:
        keep = [k for k in signal.columns if family[k] != fam]
        out[f"drop{fam}"] = signal[keep].mean(axis=1).where(signal.notna().all(axis=1))
    return out, signal, vol, family


# ---------- simulating ----------

def _run(position: np.ndarray, change: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Daily net return and amount traded for a position held during each day and reset to its target every close.

    After a day's move a position of w has drifted to w(1+r)/(1+wr); the trade is the gap to the next target."""
    position = np.nan_to_num(position, nan=0.0)
    change = np.nan_to_num(change, nan=0.0)
    base = 1 + position * change
    drift = position * (1 + change) / np.where(np.abs(base) > 1e-9, base, 1.0)
    trade = np.abs(position - np.concatenate([[0.0], drift[:-1]]))  # starts from cash
    return position * change - cost * trade, trade


def _half(index: pd.DatetimeIndex, change: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half bitcoin, half cash, put back to 50/50 at the close before the first day of every month."""
    n = len(index)
    net, trade, weight = np.zeros(n), np.zeros(n), np.zeros(n)
    first = index.day == 1
    w = 0.0
    for i in range(n):
        if first[i] or i == 0:
            trade[i] = abs(0.5 - w)
            w = 0.5
        r = change[i] if math.isfinite(change[i]) else 0.0
        weight[i] = w
        net[i] = w * r - cost * trade[i]
        w = w * (1 + r) / (1 + w * r)
    return net, trade, weight


def _score_parts(net: np.ndarray) -> tuple[float | None, float, float]:
    """(yearly compound return in %, yearly volatility in %, return ÷ volatility) of a run of daily net returns."""
    n = len(net)
    if n < 2:
        return None, 0.0, 0.0
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    return ret, vol, (ret / vol if vol > 1e-9 else 0.0)


def _measure(net: pd.Series, position: pd.Series, trade: pd.Series) -> dict:
    values = net.values
    ret, vol, ratio = _score_parts(values)
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    peak = np.maximum.accumulate(equity)
    deepest = float((equity / peak - 1).min() * 100)
    longest, since, open_end = 0, 0, False
    for i in range(1, len(equity)):
        if equity[i] < peak[i] - 1e-12:
            if i - since > longest:
                longest, open_end = i - since, i == len(equity) - 1
            elif i == len(equity) - 1 and i - since == longest:
                open_end = True
        else:
            since = i
    by_year = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    worst = min(by_year, key=lambda y: by_year[y])
    return {"days": len(values), "ret": ret, "vol": vol, "ratio": ratio, "deepest": deepest, "under": int(longest),
            "under_open": bool(open_end), "years": by_year, "worst": worst, "share": float(position.mean() * 100),
            "turnover": float(trade.sum() / (len(values) / YEAR))}


def _random_check(change: np.ndarray, share: float, cycle: float, cost: float) -> dict:
    """Random all-in / all-out schedules holding bitcoin about `share` of the days, in blocks averaging
    share*cycle days in and (1-share)*cycle days out. Returns the ratio of each schedule."""
    rng = np.random.default_rng(SEED)
    n = len(change)
    stay_in = 1 - 1 / max(share * cycle, 1.0001)
    stay_out = 1 - 1 / max((1 - share) * cycle, 1.0001)
    state = rng.random(SIMS) < share
    pos = np.zeros((n, SIMS))
    draws = rng.random((n, SIMS))
    for i in range(n):
        pos[i] = state
        state = np.where(state, draws[i] < stay_in, draws[i] >= stay_out)
    trade = np.abs(np.diff(pos, axis=0, prepend=0.0))
    net = pos * np.nan_to_num(change)[:, None] - cost * trade
    growth = np.prod(1 + net, axis=0)
    ret = (growth ** (YEAR / n) - 1) * 100
    vol = np.std(net, axis=0, ddof=1) * math.sqrt(YEAR) * 100
    ratio = np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)
    return {"ratio": ratio, "ret": ret, "held": float(pos.mean() * 100)}


def _compute(close: pd.Series) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    target, signal, vol, family = _targets(close)
    change = close.pct_change()
    last = close.index[-1]
    index = close.index

    def series(key: str, delay: int = 1, cost: float = COST) -> tuple[pd.Series, pd.Series, pd.Series]:
        if key == "hold":
            return change.fillna(0.0), pd.Series(1.0, index=index), pd.Series(0.0, index=index)
        if key == "half":
            net, trade, weight = _half(index, change.values, cost)
            return pd.Series(net, index=index), pd.Series(weight, index=index), pd.Series(trade, index=index)
        pos = target[key].shift(delay)  # decided at the close of day t, held during day t+delay
        net, trade = _run(pos.values, change.values, cost)
        return pd.Series(net, index=index), pos.fillna(0.0), pd.Series(trade, index=index)

    def measure(parts, start: str, end: str | None = None) -> dict:
        rows = (index >= pd.Timestamp(start)) & (index <= (pd.Timestamp(end) if end else last))
        return _measure(parts[0][rows], parts[1][rows], parts[2][rows])

    runs = {k: series(k) for k in ORDER}
    main = {k: measure(runs[k], MAIN_START) for k in ORDER}
    early = {k: measure(runs[k], *EARLY) for k in ORDER}
    for table in (main, early):
        years = sorted(table["hold"]["years"])
        for k in ORDER:
            table[k]["beat"] = sum(1 for y in years if table[k]["years"][y] > table["hold"]["years"][y])
            table[k]["n_years"] = len(years)

    robust = {
        "cost": {c: measure(series("blend", cost=c), MAIN_START) for c in COSTS},
        "delay": {d: measure(series("blend", delay=d), MAIN_START) for d in (1, 2, 3)},
        "drop": {f: measure(series(f"drop{f}"), MAIN_START) for f in FAMILIES},
    }

    rows = index >= pd.Timestamp(MAIN_START)
    thresh = runs["thresh"][1][rows].values
    entries = int(((thresh[1:] == 1) & (thresh[:-1] == 0)).sum()) + int(thresh[0] == 1)
    cycle = len(thresh) / max(entries, 1)
    rnd = _random_check(change[rows].values, main["blend"]["share"] / 100, cycle, COST)

    positions = pd.DataFrame({k: runs[k][1] for k in ORDER if k not in ("hold", "half")})
    return {"last": last, "close": close, "main": main, "early": early, "robust": robust, "positions": positions,
            "target": target, "signal": signal, "vol": vol, "family": family, "cycle": cycle,
            "random": {"median": float(np.median(rnd["ratio"])), "held": rnd["held"],
                       "better": float((rnd["ratio"] > main["blend"]["ratio"]).mean() * 100),
                       "better_ret": float((rnd["ret"] > main["blend"]["ret"]).mean() * 100)}}


# ---------- the study ----------

def _cmp(a: float, b: float, tol: float, more: str, less: str) -> str:
    if abs(a - b) <= tol:
        return "差不多"
    return more if a > b else less


def study() -> dict:
    close = _daily_close()
    r = _compute(close)
    main, early, last = r["main"], r["early"], r["last"]
    years = sorted(main["hold"]["years"])
    n = len(years)
    current = years[-1]
    partial = last < pd.Timestamp(f"{current}-12-31")
    n_rules = r["signal"].shape[1]

    def year_name(y: int) -> str:
        return f"{y}（到 {last.strftime('%m-%d')}）" if y == current and partial else str(y)

    def under(m: dict) -> str:
        return f"{m['under']:,} 天" + ("（期末還沒回到）" if m["under_open"] else "")

    def held(key: str, m: dict) -> str:
        if key == "ls":
            return f"淨 {m['share']:+.0f}%｜{m['turnover']:.1f} 次"
        return f"{m['share']:.0f}%｜{m['turnover']:.1f} 次"

    # ----- table 1: all variants over the main period -----
    summary_rows = []
    for k in ORDER:
        m = main[k]
        wins = "—" if k == "hold" else f"{m['n_years']} 年裡 {m['beat']} 年"
        summary_rows.append([NAMES[k], _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]),
                             _pct(m["deepest"]), under(m), f"{m['worst']} 年 {_pct(m['years'][m['worst']])}｜{wins}",
                             held(k, m)])

    # ----- table 2: year by year -----
    year_keys = ("hold", "c20", "blend", "thresh", "vol40", "vol60", "half")
    year_rows = [[year_name(y)] + [_pct(main[k]["years"][y]) for k in year_keys] for y in years]
    year_rows.append(["最差的一年"] + [_pct(main[k]["years"][main[k]["worst"]]) for k in year_keys])
    year_rows.append(["賺得比一直抱著多的年數", "—"] + [f"{n} 年裡 {main[k]['beat']} 年" for k in year_keys[1:]])

    # ----- table 3: robustness of the blend -----
    def robust_row(name: str, m: dict) -> list:
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['turnover']:.1f} 次"]

    rb = r["robust"]
    counts = {f: sum(1 for k in r["family"] if r["family"][k] == f) for f in FAMILIES}
    robust_rows = [robust_row("原本的設定（成本 0.10%、隔天照做）", main["blend"])]
    robust_rows += [robust_row(f"成本改成 {c * 100:.2f}%", rb["cost"][c]) for c in COSTS if c != COST]
    robust_rows += [robust_row("晚一天才照做（訊號後第 2 天）", rb["delay"][2]),
                    robust_row("晚兩天才照做（訊號後第 3 天）", rb["delay"][3])]
    robust_rows += [robust_row(f"拿掉{FAMILIES[f]}（剩 {n_rules - counts[f]} 條）", rb["drop"][f]) for f in FAMILIES]
    rnd = r["random"]
    robust_rows.append([f"對照：{SIMS:,} 個隨機進出時間表的中間值", "—", "—", _num(rnd["median"]), "—", "—"])

    # ----- table 4: 2013-2017 -----
    early_years = len(early["hold"]["years"])
    early_rows = []
    for k in ORDER:
        m = early[k]
        wins = "—" if k == "hold" else f"{early_years} 年裡 {m['beat']} 年"
        early_rows.append([NAMES[k], _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]),
                           _pct(m["deepest"]), under(m), wins, held(k, m)])

    span = f"{MAIN_START[:7]} 到 {_d(last)}"
    tables = [
        {"caption": (f"總表（{span}，{n} 個年度{'，最後一年還沒過完' if partial else ''}）。每種做法都是當天收盤決定、隔天才照做，"
                     f"每次買賣扣交易金額的 {COST * 100:.2f}%，現金沒有利息；一直抱著沒有扣成本。最後一欄是"
                     "「平均持有比特幣的比例｜一年換手幾次」（換手 1 次 = 一年內買賣的總金額等於全部資金一次）。"),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最久多久沒回到高點",
                     "最差的一年｜賺得比一直抱著多的年數", "平均持有｜一年換手幾次"],
         "rows": summary_rows},
        {"caption": "逐年報酬（扣成本後）。每一年都是同一套規則一路做下來的結果，沒有每年重新挑選。",
         "columns": ["年度", "一直抱著", "單一規則 20 日新高／10 日新低", "混合", "混合過半", "混合×波動 40%", "混合×波動 60%",
                     "一半一半每月調回"],
         "rows": year_rows},
        {"caption": (f"「混合」經不經得起改動（只看 {span}）：改成本、晚幾天才照做、整類規則拿掉。最後一列是運氣對照："
                     f"{SIMS:,} 個隨機的「全進／全出」時間表，持有比特幣的天數比例和混合差不多（平均 {rnd['held']:.0f}%）、"
                     f"每進出一輪平均約 {r['cycle']:.0f} 天，其中 {rnd['better']:.0f}% 的報酬÷波動比混合高。"),
         "columns": ["改了什麼", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "一年換手幾次"],
         "rows": robust_rows},
        {"caption": (f"對照期間 {EARLY[0][:4]}–{EARLY[1][:4]}（比特幣早年漲得異常多的 {early_years} 年），同樣的做法、同樣的成本。"
                     "這段期間的價格來源是 blockchain.info 的每日價格。"),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最久多久沒回到高點", "賺得比一直抱著多的年數",
                     "平均持有｜一年換手幾次"],
         "rows": early_rows},
    ]

    # ----- today -----
    state = r["signal"].iloc[-1]
    want = r["target"].iloc[-1]
    long_now = int((state == 1).sum())
    vol_now = float(r["vol"].iloc[-1] * 100)
    by_family = "、".join(f"{FAMILIES[f]} {counts[f]} 條裡 {int(sum(state[k] == 1 for k in state.index if r['family'][k] == f))} 條"
                          for f in FAMILIES)

    def stake(v: float) -> str:
        if not _ok(v):
            return "—"
        if v < -0.0005:
            return f"做空 {abs(v) * 100:.0f}%"
        return f"持有比特幣 {v * 100:.0f}%" if v > 0.0005 else "全部現金"

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"{n_rules} 條順勢規則裡現在看多的", "value": f"{long_now} 條（{long_now / n_rules * 100:.0f}%）：{by_family}"},
        {"label": f"最近 {VOL_DAYS} 天的波動（換算成一年）", "value": _pct(vol_now, signed=False)},
        {"label": "照規則下一天的狀態：混合", "value": stake(want["blend"])},
        {"label": "照規則下一天的狀態：混合過半", "value": stake(want["thresh"])},
        {"label": "照規則下一天的狀態：混合×波動調整", "value": f"目標 40%：{stake(want['vol40'])}；目標 60%：{stake(want['vol60'])}"},
        {"label": "照規則下一天的狀態：混合多空", "value": stake(want["ls"])},
        {"label": "照規則下一天的狀態：兩條單一規則",
         "value": f"20 日新高／10 日新低：{stake(want['c20'])}；50 日均線：{stake(want['ma50'])}"},
    ]

    # ----- verdict -----
    b, c, h, half, ls = main["blend"], main["c20"], main["hold"], main["half"], main["ls"]
    v40, v60, d2, d3 = main["vol40"], main["vol60"], rb["delay"][2], rb["delay"][3]
    m50 = main["ma50"]
    vol_words = []
    for name, m in (("40%", v40), ("60%", v60)):
        vol_words.append(f"目標 {name} 是每年 {_pct(m['ret'])}、報酬÷波動 {_num(m['ratio'])}、最深虧 {_pct(m['deepest'])}")
    best_vol = max(v40["ratio"], v60["ratio"])
    worst_vol = min(v40["ratio"], v60["ratio"])
    if worst_vol > b["ratio"] + 0.03:
        vol_help = "兩種目標的報酬÷波動都比沒調整的混合高"
    elif best_vol < b["ratio"] - 0.03:
        vol_help = "兩種目標的報酬÷波動都比沒調整的混合低，主要只是持有得更少"
    else:
        vol_help = "報酬÷波動和沒調整的混合差不多，主要只是持有得更少、起伏和報酬一起變小"
    if ls["ratio"] < b["ratio"] - 0.03 and ls["ret"] < b["ret"]:
        ls_help = "報酬和報酬÷波動都比只做多的混合差，做空沒有幫上忙"
    elif ls["ratio"] > b["ratio"] + 0.03 and ls["ret"] > b["ret"]:
        ls_help = "報酬和報酬÷波動都比只做多的混合好"
    else:
        ls_help = (f"每年報酬比只做多的混合{'高' if ls['ret'] > b['ret'] else '低'}、"
                   f"報酬÷波動{_cmp(ls['ratio'], b['ratio'], 0.03, '比較高', '比較低')}")
    verdict = (
        f"{years[0]} 到 {_d(last)} 這 {n} 個年度，把 {n_rules} 條事先定好的順勢規則混在一起、照「看多的比例」持有比特幣，"
        f"扣成本後每年報酬 {_pct(b['ret'])}、報酬÷波動 {_num(b['ratio'])}、最深曾經虧 {_pct(b['deepest'])}、"
        f"{n} 年裡 {b['beat']} 年賺得比一直抱著多；單一規則「創 20 日新高進場、10 日新低出場」是 {_pct(c['ret'])}、"
        f"{_num(c['ratio'])}、{_pct(c['deepest'])}、{n} 年裡 {c['beat']} 年，一直抱著是 {_pct(h['ret'])}、{_num(h['ratio'])}、"
        f"{_pct(h['deepest'])}，最簡單的「一半比特幣一半現金」是 {_pct(half['ret'])}、{_num(half['ratio'])}、"
        f"{_pct(half['deepest'])}——也就是混合的每年報酬比這條單一規則{_cmp(b['ret'], c['ret'], 0.5, '高', '低')}、"
        f"報酬÷波動{_cmp(b['ratio'], c['ratio'], 0.03, '比較高', '比較低')}、最深虧損"
        f"{_cmp(b['deepest'], c['deepest'], 1.0, '比較淺', '比較深')}，報酬÷波動比一半一半"
        f"{_cmp(b['ratio'], half['ratio'], 0.03, '高', '低')}；但另一條單一規則「收盤高於 50 日均線才持有」是 {_pct(m50['ret'])}、"
        f"{_num(m50['ratio'])}、{_pct(m50['deepest'])}，報酬÷波動和混合比是{_cmp(m50['ratio'], b['ratio'], 0.03, '比較高', '比較低')}，"
        + ("所以混合並沒有贏過每一條單一規則。" if m50["ratio"] > b["ratio"] + 0.03 else "。") +
        f"再照市場波動調整持有量：{'；'.join(vol_words)}，{vol_help}；加上做空的版本是每年 {_pct(ls['ret'])}、"
        f"報酬÷波動 {_num(ls['ratio'])}、最深虧 {_pct(ls['deepest'])}，{ls_help}（還沒算做空要付的費用）。"
        f"混合如果晚一天才照做，每年報酬從 {_pct(b['ret'])} 變成 {_pct(d2['ret'])}"
        f"（{'少' if d2['ret'] < b['ret'] else '多'} {abs(d2['ret'] - b['ret']):.1f} 個百分點），晚兩天是 {_pct(d3['ret'])}；"
        f"{SIMS:,} 個持有天數比例相近的隨機進出時間表裡，有 {rnd['better']:.0f}% 的報酬÷波動比混合高。"
    )

    notes = [
        f"{n_rules} 條規則是：收盤高於 N 日均線算看多（N = {'、'.join(map(str, MA_DAYS))}）；創 N 日新高後算看多、跌破 N/2 日新低後"
        f"不算（N = {'、'.join(map(str, CHANNEL))}）；過去 L 天是漲的算看多（L = {'、'.join(map(str, PAST))}）。"
        "每條一票、票票等值，沒有挑選、沒有調整參數。",
        "各種做法一句話：「混合」＝看多的規則占幾成，就把幾成的錢放在比特幣、其餘放現金；「混合過半」＝至少一半的規則看多就全部持有，"
        "否則全部現金；「混合×波動調整」＝混合的比例再乘上「目標波動 ÷ 最近 30 天的波動」（最多乘 1，不借錢放大），市場越劇烈拿得越少；"
        "「混合多空」＝部位是「看多比例 × 2 − 1」，全部看多時滿倉、全部不看多時滿倉做空、剛好一半時空手；"
        "「一半一半」＝不看任何訊號，一半比特幣一半現金，每月 1 日調回各半。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「每年波動」是每天漲跌的起伏換算成一年；「報酬÷波動」是兩者相除，"
        "越高代表每承受一分起伏換到的報酬越多；「最深曾經虧多少」是從最高點往下掉最多的幅度；「最久多久沒回到高點」是從一個高點算起、"
        "到資金重新回到那個高點為止最長的天數（期末還沒回到的算到期末）；「最差的一年」是日曆年裡報酬最低的那一年"
        + ("，還沒過完的今年也算一年。" if partial else "。"),
        "所有訊號和用來調整的波動，都只用到當天收盤為止已經知道的資料，隔天才持有對應的部位；只用已經收完的日線。"
        "持有比例不是 0 或 100% 的做法，每天收盤都調回目標比例，這些小額調整也照樣算進成本和換手次數。",
        "這組規則是用今天的知識定的：現在大家都知道順勢規則在比特幣上表現過不錯，才會拿這幾類規則來混合，"
        "所以即使沒有挑參數，整個研究還是留有一些事後才知道的成分，表裡的數字可能比真正事先能做到的好看。",
        f"只測了比特幣一種資產，主要期間只有 {n} 個年度，而且這些年並不是彼此獨立的（大致就是兩三輪大漲大跌），"
        f"「{n} 年裡幾年」這種數字多一年少一年就會差很多，做法之間的小差距不能當成確定的結論。",
        f"成本只用一個粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%。實際的手續費、買賣價差和滑價會隨交易所、金額和年代不同"
        "（第三張表有換成其他成本的結果）。持有現金假設沒有利息，也沒有算稅。",
        "做空的版本只扣了和買賣一樣的成本，沒有算做空實際要付的資金費率或借幣利息，也沒有算被強制平倉的風險，"
        "所以它的數字比實際能做到的好看；2013–2017 年其實也沒有方便做空的管道，那一段的做空數字只是算術上的對照。",
        f"運氣對照的做法：產生 {SIMS:,} 個隨機的「全部持有／全部現金」時間表，持有天數的比例和混合的平均持有比例相同，"
        f"每一輪進出的平均長度取自「混合過半」實際的進出次數（平均約 {r['cycle']:.0f} 天一輪），同樣扣成本；"
        f"其中 {rnd['better']:.0f}% 的報酬÷波動比混合高、{rnd['better_ret']:.0f}% 的每年報酬比混合高。這只是很簡單的檢查，"
        "隨機時間表是全進全出、混合是部分持有，兩者不完全是同一種東西。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）。"
        "blockchain.info 標示的日期大約比幣安晚一天，整段內部順序一致、不會用到未來的價格，但 2017-08 接縫那一天等於合併了兩天的漲跌；"
        "新高新低用的是收盤價，不是盤中最高最低價。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼，不是買賣建議。",
    ]

    return {
        "id": "trend_ensemble",
        "title": "不挑參數，把很多順勢規則混在一起",
        "question": "既然挑到哪一組參數最好多半是運氣，那乾脆把很多條順勢規則混在一起、再依市場劇烈程度調整持有多少，會不會比只挑一條規則更好或更穩？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2010 年起）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
