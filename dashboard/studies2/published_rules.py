"""Historical study "published_rules": do the few bitcoin rules that published research found useful still work
on data after the papers' sample periods?

Four rules, each retested on the paper's own period (as far as our data reaches) and on the later period:

1. Moving-average trend (Detzel, Liu, Strauss, Zhou, Zhu, Financial Management 2021; sample 2010-2017): hold
   bitcoin on the day after the close is above its N-day simple moving average, otherwise cash.
2. New-high breakout (Quantpedia MAX rule; sample 2015-11 to 2022-01): hold bitcoin on the day after the close is
   the highest close of the last N days, otherwise cash. Also the opposite (N-day low).
3. Weekly momentum (Liu & Tsyvinski, Review of Financial Studies 2021; sample 2011 to May 2018): weeks sorted into
   five groups by their own return, and what the next week did.
4. Carry as crash warning (Schmeling, Schrimpf, Todorov, BIS Working Paper 1087): days when the 7-day average
   perpetual funding rate is in its top tenth, ranked against its own earlier history only.

No look-ahead: a signal is computed from closes up to and including day t and decides the position held from the
close of day t to the close of day t+1. Thresholds of rule 4 use only days before the day being ranked. Only closed
candles are used. Everything comes from public, key-less endpoints through common.get_json (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import price as px

FAPI = "https://fapi.binance.com"
COST = 0.001  # 0.10% of the position every time the rule switches between bitcoin and cash
YEAR = 365  # bitcoin trades every day
MA_DAYS = (5, 10, 20, 50, 100)
BREAK_DAYS = (10, 20, 50)
MA_PAPER_END = "2017-12-31"  # Detzel et al.: 2010-2017
BREAK_PAPER_START = "2015-11-01"  # Quantpedia: first signal 2015-11
BREAK_PAPER_END = "2022-01-31"
WEEK_PAPER_START = "2011-01-01"  # Liu & Tsyvinski: 2011-01 to 2018-05
WEEK_PAPER_END = "2018-05-31"
CARRY_SINCE = "2019-09-01"
CARRY_PAPER_END = "2022-12-31"
CARRY_MIN_DAYS = 365
CARRY_TOP = 0.90
HORIZONS = (7, 14, 28)
DROP = 0.10
GROUPS = ("跌最多的 1/5", "偏弱的 1/5", "中間的 1/5", "偏強的 1/5", "漲最多的 1/5")


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    digits = 0 if abs(v) >= 100 else 1
    text = f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"
    return text.replace("+0.0%", "0.0%").replace("-0.0%", "0.0%") if abs(v) < 0.05 else text


def _num(v, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}" if _ok(v) else "—"


def _times(v) -> str:
    return f"100 次裡 {float(v):.0f} 次" if _ok(v) else "—"


def _span(values, fmt) -> str:
    """Lowest ~ highest of several variants, so that no single lucky variant is picked."""
    values = [float(v) for v in values if _ok(v)]
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} ~ {fmt(hi)}"


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


def _funding_history() -> pd.Series:
    """Every Binance BTCUSDT funding settlement since 2019-09, in % per settlement, DatetimeIndex (UTC, naive)."""
    out, start = [], int(pd.Timestamp(CARRY_SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(25):
        page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1000, "startTime": start})
        if not page:
            break
        out += page
        start = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    if len(out) < 1000:
        raise RuntimeError(f"幣安資金費率歷史只回傳 {len(out)} 筆")
    series = pd.Series([float(p["fundingRate"]) * 100 for p in out],
                       index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
    return series[~series.index.duplicated()].sort_index()


def _spot_candles() -> pd.DataFrame:
    """Binance spot BTCUSDT daily low and close since 2019-09, finished days only."""
    rows, start = [], int(pd.Timestamp(CARRY_SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(8):
        page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + px.DAY_MS
        if len(page) < 1000:
            break
    frame = pd.DataFrame([[r[0], float(r[3]), float(r[4])] for r in rows],
                         columns=["ms", "low", "close"]).drop_duplicates("ms").sort_values("ms")
    frame.index = pd.to_datetime(frame["ms"], unit="ms")
    frame = frame[frame.index < pd.Timestamp(common.today_utc())]  # drop the candle that is still open
    if len(frame) < 1000:
        raise RuntimeError(f"幣安現貨日線只回傳 {len(frame)} 筆")
    return frame[["low", "close"]]


# ---------- rules 1 and 2: in the market or in cash ----------

def _positions(close: pd.Series) -> dict:
    """Position (1 = bitcoin, 0 = cash) held during each day, decided by the signal at the previous day's close."""
    out = {"hold": pd.Series(1.0, index=close.index)}
    signals = {}
    for n in MA_DAYS:
        average = close.rolling(n, min_periods=n).mean()
        signals[f"ma{n}"] = (close > average).where(average.notna())
    for n in BREAK_DAYS:
        high = close.rolling(n, min_periods=n).max()
        low = close.rolling(n, min_periods=n).min()
        signals[f"high{n}"] = (close >= high).where(high.notna())
        signals[f"low{n}"] = (close <= low).where(low.notna())
    for key, signal in signals.items():
        out[key] = signal.astype(float).shift(1)  # signal at the close of day t -> position for day t+1
    out["_signals"] = signals
    return out


def _measure(close: pd.Series, position: pd.Series, start, end) -> dict | None:
    """Results of one position series between two dates (both included), with and without the switching cost."""
    change = close.pct_change()
    turn = position.diff().abs().fillna(0.0)
    window = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end)) & position.notna() & change.notna()
    pos, ret, turn = position[window], change[window], turn[window]
    n = int(len(pos))
    if n < 60:
        return None
    gross = pos * ret
    net = gross - COST * turn

    def yearly(series) -> float | None:
        growth = float((1 + series).prod())
        return (growth ** (YEAR / n) - 1) * 100 if growth > 0 else None

    equity = np.concatenate([[1.0], (1 + net).cumprod().values])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    vol = float(net.std(ddof=1) * math.sqrt(YEAR) * 100)
    ret_net = yearly(net)
    return {"days": n, "ret_gross": yearly(gross), "ret_net": ret_net, "vol": vol,
            "ratio": ret_net / vol if ret_net is not None and vol > 0 else None,
            "deepest": deepest, "share": float(pos.mean() * 100), "switches": int(round(float(turn.sum())))}


def _strategy_row(period: str, name: str, m: dict | None) -> list:
    if not m:
        return [period, name] + ["—"] * 6
    return [period, name, _pct(m["ret_gross"]), _pct(m["ret_net"]), _pct(m["vol"], signed=False), _num(m["ratio"]),
            _pct(m["deepest"]), f"{m['share']:.0f}%／{m['switches']} 次"]


def _range_row(period: str, name: str, ms: list) -> list:
    ms = [m for m in ms if m]
    if not ms:
        return [period, name] + ["—"] * 6
    return [period, name, _span([m["ret_gross"] for m in ms], _pct), _span([m["ret_net"] for m in ms], _pct),
            _span([m["vol"] for m in ms], lambda v: _pct(v, signed=False)), _span([m["ratio"] for m in ms], _num),
            _span([m["deepest"] for m in ms], _pct),
            _span([m["share"] for m in ms], lambda v: f"{v:.0f}%") + "／"
            + _span([m["switches"] for m in ms], lambda v: f"{v:.0f}") + " 次"]


STRATEGY_COLUMNS = ["期間", "做法", "每年報酬（不算成本）", "每年報酬（扣成本）", "每年波動", "報酬÷波動",
                    "最深曾經虧多少", "持有比特幣的天數比例／切換次數"]


# ---------- rule 3: weekly momentum ----------

def _weekly(close: pd.Series) -> pd.DataFrame:
    """Finished Monday-to-Sunday weeks: this week's return and the next week's return (in %)."""
    count = close.resample("W-SUN").count()
    week = close.resample("W-SUN").last()[count == 7]  # whole weeks only; the running week is dropped
    change = (week / week.shift(1) - 1) * 100
    consecutive = pd.Series(week.index, index=week.index).diff() == pd.Timedelta(days=7)
    change = change.where(consecutive)
    frame = pd.DataFrame({"this": change, "next": change.shift(-1)})
    frame["next"] = frame["next"].where(consecutive.shift(-1, fill_value=False))
    return frame.dropna(subset=["this"])


def _week_groups(frame: pd.DataFrame, start, end) -> dict | None:
    """Five equal groups by this week's return (edges from the same period), and what the following week did."""
    part = frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))].dropna()
    # the following week must also end inside the period
    part = part[part.index + pd.Timedelta(days=7) <= pd.Timestamp(end)]
    if len(part) < 50:
        return None
    edges = [float(part["this"].quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
    group = np.searchsorted(edges, part["this"].values, side="left")
    rows = []
    for g in range(5):
        chunk = part[group == g]
        rows.append({"n": int(len(chunk)), "lo": float(chunk["this"].min()), "hi": float(chunk["this"].max()),
                     "mean": float(chunk["next"].mean()), "median": float(chunk["next"].median()),
                     "up": float((chunk["next"] > 0).mean() * 100)})
    everything = {"n": int(len(part)), "lo": float(part["this"].min()), "hi": float(part["this"].max()),
                  "mean": float(part["next"].mean()), "median": float(part["next"].median()),
                  "up": float((part["next"] > 0).mean() * 100)}
    return {"groups": rows, "all": everything, "edges": edges, "first": _d(part.index[0]), "last": _d(part.index[-1])}


def _week_rows(period: str, result: dict | None) -> list:
    if not result:
        return [[period, "資料不足", "—", "—", "—", "—", "—"]]
    rows = []
    for label, g in list(zip(GROUPS, result["groups"])) + [("全部的週（對照）", result["all"])]:
        rows.append([period, label, f"{_pct(g['lo'])} ~ {_pct(g['hi'])}", f"{g['n']} 週", _pct(g["mean"]),
                     _pct(g["median"]), _times(g["up"])])
    return rows


# ---------- rule 4: funding in its top tenth ----------

def _carry(funding: pd.Series, candles: pd.DataFrame) -> dict:
    daily = funding.resample("1D").mean()
    daily = daily[daily.index < pd.Timestamp(common.today_utc())]  # finished days only
    avg7 = daily.rolling(7, min_periods=7).mean().round(6).dropna()
    avg7 = avg7[avg7.index.isin(candles.index)]
    # the line a day has to beat: the 90th percentile of all EARLIER days (shift keeps the day itself out)
    line = avg7.expanding(min_periods=CARRY_MIN_DAYS).quantile(CARRY_TOP).shift(1).round(6)
    ranked = line.notna()
    hot = (avg7 > line) & ranked
    close, low = candles["close"], candles["low"]
    worst = {}
    for h in HORIZONS:
        # lowest low of days D+1 .. D+h against the close of day D, as a positive "largest drop" in %
        ahead = low[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        worst[h] = (-(ahead / close - 1) * 100).clip(lower=0).reindex(avg7.index)
    paper_end = pd.Timestamp(CARRY_PAPER_END)
    periods = [("論文當時", ranked & (avg7.index <= paper_end)), ("之後", ranked & (avg7.index > paper_end))]
    rows, summary = [], {}
    for period, inside in periods:
        span = avg7.index[inside]
        label = f"{period}（{_d(span[0])[:7]} ~ {_d(span[-1])[:7]}）" if len(span) else period
        for kind, mask in (("費率在前一成的日子", inside & hot), ("全部日子（對照）", inside)):
            days = int(mask.sum())
            full = mask.reindex(pd.date_range(avg7.index[0], avg7.index[-1], freq="D"), fill_value=False)
            episodes = int((full & ~full.shift(1, fill_value=False)).sum())
            for h in HORIZONS:
                values = worst[h][mask].dropna()
                stats = None
                if len(values):
                    stats = {"n": int(len(values)), "median": float(values.median()), "mean": float(values.mean()),
                             "drop": float((values >= DROP * 100).mean() * 100)}
                summary[(period, kind[:2], h)] = stats
                first = h == HORIZONS[0]
                rows.append([label if first and kind.startswith("費率") else "", kind if first else "",
                             (f"{days} 天／{episodes} 段" if kind.startswith("費率") else f"{days} 天") if first else "",
                             f"之後 {h} 天",
                             _pct(-stats["median"]) if stats else "—", _pct(-stats["mean"]) if stats else "—",
                             _times(stats["drop"]) if stats else "—"])
            summary[(period, kind[:2], "days")] = days
            summary[(period, kind[:2], "episodes")] = episodes
    last_day = avg7.index[-1]
    return {"rows": rows, "summary": summary, "avg7": float(avg7.iloc[-1]),
            "line": float(line.iloc[-1]) if _ok(line.iloc[-1]) else None, "hot_now": bool(hot.iloc[-1]),
            "last": _d(last_day), "first_ranked": _d(avg7.index[ranked][0]) if ranked.any() else None,
            "first": _d(avg7.index[0])}


# ---------- the study ----------

def study() -> dict:
    close = _daily_close()
    last = close.index[-1]
    pos = _positions(close)
    signals = pos["_signals"]

    # ----- rule 1: moving averages -----
    ma_start = close.index[max(MA_DAYS)]  # the first day every moving average already has a position
    ma_periods = [("論文當時", ma_start, pd.Timestamp(MA_PAPER_END)),
                  ("之後", pd.Timestamp(MA_PAPER_END) + pd.Timedelta(days=1), last),
                  ("2022 年以來", pd.Timestamp("2022-01-01"), last),
                  ("最近 2 年", last - pd.Timedelta(days=2 * YEAR - 1), last)]
    ma = {name: {key: _measure(close, pos[key], a, b) for key in ["hold"] + [f"ma{n}" for n in MA_DAYS]}
          for name, a, b in ma_periods}
    span = {name: f"{name}（{_d(a)[:7]} ~ {_d(b)[:7]}）" for name, a, b in ma_periods}
    ma_rows = []
    for name in ("論文當時", "之後"):
        ma_rows.append(_strategy_row(span[name], "一直抱著不動", ma[name]["hold"]))
        for n in MA_DAYS:
            ma_rows.append(_strategy_row("", f"{n} 日均線", ma[name][f"ma{n}"]))
    for name in ("2022 年以來", "最近 2 年"):
        ma_rows.append(_strategy_row(span[name], "一直抱著不動", ma[name]["hold"]))
        ma_rows.append(_range_row("", "5 種均線（最低 ~ 最高）", [ma[name][f"ma{n}"] for n in MA_DAYS]))

    def beat(table: dict, keys: list, field: str, higher: bool = True) -> int:
        base = table["hold"][field] if table.get("hold") else None
        count = 0
        for key in keys:
            value = table[key][field] if table.get(key) else None
            if _ok(base) and _ok(value) and ((value > base) if higher else (value < base)):
                count += 1
        return count

    ma_keys = [f"ma{n}" for n in MA_DAYS]

    # ----- rule 2: new highs and new lows -----
    br_periods = [("論文當時", pd.Timestamp(BREAK_PAPER_START), pd.Timestamp(BREAK_PAPER_END)),
                  ("之後", pd.Timestamp(BREAK_PAPER_END) + pd.Timedelta(days=1), last),
                  ("2022 年以來", pd.Timestamp("2022-01-01"), last),
                  ("最近 2 年", last - pd.Timedelta(days=2 * YEAR - 1), last)]
    br_keys = [f"high{n}" for n in BREAK_DAYS]
    low_keys = [f"low{n}" for n in BREAK_DAYS]
    br = {name: {key: _measure(close, pos[key], a, b) for key in ["hold"] + br_keys + low_keys}
          for name, a, b in br_periods}
    br_span = {name: f"{name}（{_d(a)[:7]} ~ {_d(b)[:7]}）" for name, a, b in br_periods}
    br_rows = []
    for name in ("論文當時", "之後"):
        br_rows.append(_strategy_row(br_span[name], "一直抱著不動", br[name]["hold"]))
        for n in BREAK_DAYS:
            br_rows.append(_strategy_row("", f"創 {n} 日新高才持有", br[name][f"high{n}"]))
        br_rows.append(_range_row("", "反過來：創新低才持有（3 種）", [br[name][k] for k in low_keys]))
    for name in ("2022 年以來", "最近 2 年"):
        br_rows.append(_strategy_row(br_span[name], "一直抱著不動", br[name]["hold"]))
        br_rows.append(_range_row("", "創新高才持有（3 種）", [br[name][k] for k in br_keys]))
        br_rows.append(_range_row("", "反過來：創新低才持有（3 種）", [br[name][k] for k in low_keys]))

    # ----- rule 3: weekly momentum -----
    weeks = _weekly(close)
    wk_paper = _week_groups(weeks, WEEK_PAPER_START, WEEK_PAPER_END)
    wk_after = _week_groups(weeks, pd.Timestamp(WEEK_PAPER_END) + pd.Timedelta(days=1), weeks.index[-1])
    wk_rows = (_week_rows(f"論文當時（{WEEK_PAPER_START[:7]} ~ {WEEK_PAPER_END[:7]}）", wk_paper)
               + _week_rows(f"之後（2018-06 ~ {_d(weeks.index[-1])[:7]}）", wk_after))
    for index, row in enumerate(wk_rows):
        if index % 6:
            row[0] = ""
    last_week = float(weeks["this"].iloc[-1])
    last_week_end = weeks.index[-1]
    week_group = int(np.searchsorted(wk_after["edges"], last_week, side="left")) if wk_after else None

    # ----- rule 4: funding -----
    notes_extra: list[str] = []
    carry = None
    try:
        carry = _carry(_funding_history(), _spot_candles())
    except Exception as exc:  # noqa: BLE001 - the three price rules still stand without the funding data
        notes_extra.append(f"幣安的資金費率或日線資料這次取不到（{type(exc).__name__}），第四張表留空，沒有改用其他方法繞過。")

    carry_columns = ["期間", "哪些日子", "天數／段數", "往後看", "途中最大跌幅（中位數）", "途中最大跌幅（平均）", "曾跌 10% 以上"]
    carry_rows = carry["rows"] if carry else [["這次取不到資料", "—", "—", "—", "—", "—", "—"]]

    tables = [
        {"caption": (f"規則一「均線順勢」（Detzel 等人，樣本 2010–2017）：收盤價高於過去 N 天的平均收盤價，隔天就持有比特幣，"
                     f"否則隔天持有現金；成本以每次切換 {COST * 100:.2f}% 計算。"),
         "columns": STRATEGY_COLUMNS, "rows": ma_rows},
        {"caption": (f"規則二「創新高突破」（Quantpedia，樣本 2015-11 到 2022-01）：今天的收盤價是最近 N 天裡最高的，隔天就持有比特幣，"
                     f"否則持有現金；「反過來」是收盤價創 N 天新低才持有。成本同樣以每次切換 {COST * 100:.2f}% 計算。"),
         "columns": STRATEGY_COLUMNS, "rows": br_rows},
        {"caption": ("規則三「週動能」（Liu 與 Tsyvinski，樣本 2011 到 2018-05）：把每一週（週一到週日）依當週漲跌幅由低到高平分成五組，"
                     "看下一週的表現；如果動能存在，漲最多那組的下一週應該最好。"),
         "columns": ["期間", "這一週屬於哪一組", "這一週的漲跌幅範圍", "週數", "下一週平均漲跌", "下一週漲跌中位數", "下一週上漲"],
         "rows": wk_rows},
        {"caption": ("規則四「資金費率過高是下跌警訊」（BIS 工作論文 1087）：幣安比特幣永續合約的 7 天平均資金費率，"
                     "高過它自己先前所有日子的九成（至少先累積 365 天）時，之後 7／14／28 天途中的最大跌幅有沒有比平常大。"),
         "columns": carry_columns, "rows": carry_rows},
    ]

    # ----- where each rule stands today -----
    last_close = float(close.iloc[-1])
    ma_now = [(n, bool(signals[f"ma{n}"].iloc[-1] == 1)) for n in MA_DAYS]
    high_now = [(n, bool(signals[f"high{n}"].iloc[-1] == 1)) for n in BREAK_DAYS]
    low_now = [(n, bool(signals[f"low{n}"].iloc[-1] == 1)) for n in BREAK_DAYS]
    inside = "持有比特幣"
    outside = "持有現金"
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {last_close:,.0f} 美元"},
        {"label": "規則一：均線順勢，照規則下一天的狀態",
         "value": f"5 種裡 {sum(v for _, v in ma_now)} 種持有比特幣："
                  + "、".join(f"{n} 日{inside if v else outside}" for n, v in ma_now)},
        {"label": "規則二：創新高突破，照規則下一天的狀態",
         "value": "、".join(f"{n} 日{inside if v else outside}" for n, v in high_now)
                  + "；反過來的創新低規則：" + "、".join(f"{n} 日{inside if v else outside}" for n, v in low_now)},
        {"label": f"規則三：上一個完整的週（到 {_d(last_week_end)}）",
         "value": (f"漲跌 {_pct(last_week)}，屬於「{GROUPS[week_group]}」（用 2018-06 以後的分組界線）"
                   if week_group is not None else f"漲跌 {_pct(last_week)}")},
    ]
    if carry:
        if carry["line"] is None:
            state = "還沒有足夠的歷史可以排名"
        else:
            state = (f"門檻是 {carry['line']:+.4f}%，" + ("在前一成之內" if carry["hot_now"] else "不在前一成"))
        now.append({"label": f"規則四：7 天平均資金費率（{carry['last']}）",
                    "value": f"{carry['avg7']:+.4f}%（每 8 小時），{state}"})

    # ----- verdict -----
    after, paper = ma["之後"], ma["論文當時"]
    n_ret = beat(after, ma_keys, "ret_net")
    n_ratio = beat(after, ma_keys, "ratio")
    n_deep = beat(after, ma_keys, "deepest")  # a drawdown closer to zero is "higher"
    n_ret_paper = beat(paper, ma_keys, "ret_net")
    b_after = br["之後"]
    n_high = beat(b_after, br_keys, "ret_net")
    n_high_ratio = beat(b_after, br_keys, "ratio")
    n_low = beat(b_after, low_keys, "ret_net")
    alive, faded = [], []
    (alive if n_ret * 2 > len(ma_keys) else faded).append("均線順勢")
    (alive if n_high * 2 > len(br_keys) else faded).append("創新高突破")
    week_clause = ""
    if wk_paper and wk_after:
        top_p, top_a, bot_a = wk_paper["groups"][4], wk_after["groups"][4], wk_after["groups"][0]
        week_alive = top_a["mean"] > wk_after["all"]["mean"] and top_a["mean"] > bot_a["mean"]
        (alive if week_alive else faded).append("週動能")
        week_clause = (f"週動能：漲最多那組的下一週平均從論文期間的 {_pct(top_p['mean'])} 降到 {_pct(top_a['mean'])}"
                       f"（全部的週 {_pct(wk_after['all']['mean'])}）")
    carry_clause = ""
    if carry:
        s = carry["summary"]
        hot_stats, base_stats = s.get(("之後", "費率", 14)), s.get(("之後", "全部", 14))
        hot_days = s.get(("之後", "費率", "days"), 0)
        if hot_stats and base_stats:
            (alive if hot_stats["drop"] > base_stats["drop"] else faded).append("資金費率警訊")
            carry_clause = (f"資金費率警訊：2023 年以後費率在前一成的日子只有 {hot_days} 天，之後 14 天 {_times(hot_stats['drop'])}"
                            f"曾跌 10% 以上（全部日子 {_times(base_stats['drop'])}）")
        else:
            carry_clause = f"資金費率警訊：2023 年以後費率在前一成的日子只有 {hot_days} 天，無法比較"
    head = "四條規則拿論文樣本之後的資料重測，"
    if alive and faded:
        head += f"還看得到效果的是{'、'.join(alive)}（都比論文當時弱），{'、'.join(faded)}已經賺得比一直抱著少。"
    elif alive:
        head += "四條都還看得到效果，但都比論文當時弱。"
    else:
        head += "四條都已經看不到優勢。"
    details = [
        f"均線順勢：2018 年以後 5 種均線扣成本每年 {_span([after[k]['ret_net'] for k in ma_keys], _pct)}，"
        f"一直抱著 {_pct(after['hold']['ret_net'])}，{n_ret} 種賺得比較多（論文期間是 {n_ret_paper} 種），"
        f"最深虧損 {_span([after[k]['deepest'] for k in ma_keys], _pct)}（一直抱著 {_pct(after['hold']['deepest'])}）",
        f"創新高突破：2022-02 以後扣成本每年 {_span([b_after[k]['ret_net'] for k in br_keys], _pct)}，"
        f"一直抱著 {_pct(b_after['hold']['ret_net'])}，3 種裡 {n_high} 種賺得比較多，"
        f"最深虧損 {_span([b_after[k]['deepest'] for k in br_keys], _pct)}（一直抱著 {_pct(b_after['hold']['deepest'])}）",
    ] + [clause for clause in (week_clause, carry_clause) if clause]
    verdict = head + "；".join(details) + "。"

    carry_note, week_note = "", ""
    if carry:
        cs = carry["summary"]
        carry_note = (f"2021 年初的費率非常高，把門檻墊高了，所以 2023 年以後落在前一成的日子只有 {cs.get(('之後', '費率', 'days'), 0)} 天、"
                      f"{cs.get(('之後', '費率', 'episodes'), 0)} 段，例子太少，那幾列的數字只能當參考。")
    if wk_after:
        second = wk_after["groups"][3]
        if second["mean"] < wk_after["all"]["mean"]:
            week_note = (f"2018-06 以後各組並不是越強下一週越好：「{GROUPS[3]}」的下一週平均是 {_pct(second['mean'])}，"
                         f"比全部的週（{_pct(wk_after['all']['mean'])}）還低，只有漲最多那一組比較好。")
    notes = [
        "每條規則都是用當天的收盤價決定「下一天」要不要持有，當天的漲跌不會算進自己的訊號裡；只用已經收完的日線。",
        "「每年報酬」是把整段期間的總成果換算成每年的複利；「每年波動」是每天漲跌的起伏換算成一年；"
        "「報酬÷波動」就是前兩個數字相除，越高代表每承受一分起伏換到的報酬越多；「最深曾經虧多少」是從最高點往下掉最多的幅度。",
        f"規則比一直抱著賺得少的地方，表裡照實列出：2018 年以後 5 種均線裡有 {len(ma_keys) - n_ret} 種扣成本後的每年報酬低於一直抱著，"
        f"2022-02 以後 3 種創新高規則裡有 {len(br_keys) - n_high} 種低於一直抱著（報酬÷波動則分別有 {n_ratio} 種和 {n_high_ratio} 種比較高，"
        f"反過來的創新低規則有 {n_low} 種贏過一直抱著）。規則有不少時間持有現金，所以起伏和虧損通常比較小，但這不等於賺得比較多。",
        "早期的數字被比特幣剛出現時的巨大漲幅灌水：2010 到 2013 年價格從不到 1 美元漲到上千美元，任何有持有的做法每年報酬都高得不像話，"
        "而且當時幾乎沒有地方能照這個價格買賣。2010 年的價格只記到美分（例如連續很多天都是 0.07 美元），那一段特別粗糙。",
        "這裡一共試了 5 種均線、3 種新高、3 種新低、4 段期間，試得越多，表現最好的那一個越可能只是運氣；"
        "所以後兩段期間只列「最低 ~ 最高」，不挑單一最好的來講。",
        f"成本只用一個粗略的數字：每次切換扣 {COST * 100:.2f}%。實際的手續費、買賣價差和滑價會隨交易所、金額和年代不同，早年高得多。",
        "持有現金的日子假設沒有任何利息，也沒有算稅；一直抱著不動的那一列沒有成本。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均，不是單一交易所的收盤價），之後是幣安現貨日線"
        "（UTC 0 點收盤）。blockchain.info 標示的日期大約比幣安晚一天（它某一天的價格接近幣安前一天的收盤），整段內部順序一致，"
        "不會用到未來的價格，但 2017-08 接縫那一天等於合併了兩天的漲跌，而且 2017-08 以前的「一週」實際上大約是週日到週六。"
        "論文用的資料來源和我們不同，所以「論文當時」的數字不會和論文一模一樣。",
        "規則二的原始文章用的是每天 0 點的開盤價，試了 10 到 50 天共五種天數，樣本從 2015-11 開始；這裡用收盤價、三種天數重做。"
        "「之後」從 2022-02 起算，和「2022 年以來」只差一個月，所以兩列很接近。",
        "規則三的五組界線是用同一段期間自己的資料切的（和論文的做法相同），等於事後才知道界線在哪裡；每組只有幾十到一百多週，"
        "平均值很容易被少數幾個暴漲的週拉高，所以同時列出中位數。" + week_note,
        "規則四的原始論文用的是期貨價格高出現貨多少（套利利差），這裡用幣安永續合約的資金費率代替，只是相近而不是同一個數字；"
        "資料從 2019-09 才有，扣掉先累積的 365 天後能排名的日子從 2020-09 開始。門檻只用當天以前的日子來算。"
        "相鄰的日子往後看的幾乎是同一段行情，實際獨立的例子遠少於天數；「途中最大跌幅」是之後那幾天的最低價和當天收盤價相比。"
        + carry_note,
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼，不是買賣建議。",
    ] + notes_extra

    return {
        "id": "published_rules",
        "title": "論文裡有效的規則，後來還有效嗎",
        "question": "研究論文說對比特幣有用的四條規則，拿論文樣本結束之後的資料重新檢查，還站得住腳嗎？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Detzel, Liu, Strauss, Zhou, Zhu (2021)〈Learning and predictability via technical analysis: "
                     "Evidence from bitcoin and stocks with hard-to-value fundamentals〉Financial Management",
             "url": "https://ideas.repec.org/a/bla/finmgt/v50y2021i1p107-137.html"},
            {"name": "Quantpedia〈Revisiting Trend-following and Mean-reversion Strategies in Bitcoin〉",
             "url": "https://quantpedia.com/revisiting-trend-following-and-mean-reversion-strategies-in-bitcoin/"},
            {"name": "Liu, Tsyvinski〈Risks and Returns of Cryptocurrency〉Review of Financial Studies 2021（NBER 工作論文 24877）",
             "url": "https://www.nber.org/system/files/working_papers/w24877/w24877.pdf"},
            {"name": "Schmeling, Schrimpf, Todorov〈Crypto carry〉BIS Working Paper 1087",
             "url": "https://www.bis.org/publ/work1087.htm"},
            {"name": "blockchain.info 每日比特幣價格（2010 年起）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安 U 本位永續合約資金費率歷史",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
