"""Historical study "trend_anatomy": what following the 20-day-high / 10-day-low rule on bitcoin was actually like,
trade by trade, and where its advantage over buy-and-hold came from.

The rule (the walk-forward pick of studies2/walkforward.py): hold bitcoin from the day after the daily close is the
highest of the last 20 closes; go to cash from the day after the close is the lowest of the last 10 closes.

No look-ahead in the rule: the signal at the close of day t decides the position held during day t+1, and every
switch costs 0.10%. The big falls and big rises used in the "where does the edge come from" part ARE found with
hindsight (a zigzag over the whole price history); they only sort the calendar for explaining, the rule never sees
them. Only closed daily candles from public, key-less endpoints.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic

COST = 0.001  # 0.10% of the amount moved every time the rule switches between bitcoin and cash
YEAR = 365
ENTRY, EXIT = 20, 10
PERIODS = (("2013-01-01", "2013 年起"), ("2018-01-01", "2018 年起"))
MAIN = "2018-01-01"
GRID_ENTRY = (10, 15, 20, 30, 40, 55)
GRID_EXIT = (5, 10, 15, 20)
BEST = 5
WHIPSAW_DAYS = 15
SWING = 0.20  # a swing ends when the close has moved 20% back from its extreme (down 20% = up 25%)
BIG_FALL = 0.30  # a down swing of 30% or more, peak close to trough close
BIG_RISE = 1.00  # an up swing of 100% or more, trough close to peak close
MISSED = 0.20
LONG_SPELL = 30  # under-water spells at least this long count as "long"
LAST_TRADES = 8
DEEP = 20  # "well under water": this many percent or more below the account's own previous high
FALL, RISE, MID_UP, MID_DOWN = "大跌", "大漲", "其餘：中小型的上漲段", "其餘：中小型的下跌段"
KINDS = (FALL, RISE, MID_UP, MID_DOWN)


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


def _usd(v) -> str:
    if not _ok(v):
        return "—"
    return f"{float(v):,.0f}" if float(v) >= 1000 else f"{float(v):,.2f}"


def _total(growth) -> str:
    """A growth factor as plain words: +35.0%, −62.1%, or 'N 倍' once it is too large for a percentage to read well."""
    if not _ok(growth):
        return "—"
    growth = float(growth)
    if growth >= 11:
        return "變成 " + _times(growth)
    if growth < 0.1:
        left = growth * 100
        return "只剩不到 0.01%" if left < 0.01 else (f"只剩 {left:.2f}%" if left < 1 else f"只剩 {left:.1f}%")
    return _pct((growth - 1) * 100)


def _times(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if v >= 1e8:
        return f"{v / 1e8:,.1f} 億倍"
    if v >= 1e4:
        return f"{v / 1e4:,.0f} 萬倍" if v >= 1e5 else f"{v / 1e4:,.1f} 萬倍"
    if v >= 100:
        return f"{v:,.0f} 倍"
    if v >= 10:
        return f"{v:,.1f} 倍"
    return f"{v:.2f} 倍" if v >= 0.1 else f"{v:.2g} 倍"


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


def _cross_check(close: pd.Series) -> str | None:
    """Compare the latest closed candles with Binance spot klines fetched directly; text of a problem, or None."""
    try:
        from dashboard.collectors import price as px
        page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 8})
        today = pd.Timestamp(common.today_utc())
        worst = 0.0
        seen = 0
        for row in page:
            day = pd.Timestamp(int(row[0]), unit="ms").normalize()
            if day >= today or day not in close.index:
                continue
            seen += 1
            worst = max(worst, abs(float(row[4]) / float(close[day]) - 1) * 100)
        if seen and worst > 0.5:
            return f"最近幾根日線和幣安直接查到的收盤價最多差 {worst:.2f}%，這次的數字請保守看待。"
    except Exception:  # noqa: BLE001 - the check is optional; the study stands on the main series
        return None
    return None


# ---------- the rule ----------

def _signal(close: pd.Series, entry: int, exit_: int) -> pd.Series:
    """1 = hold bitcoin tomorrow, 0 = cash; decided at each day's close. Cash until the first new high."""
    high = close.rolling(entry, min_periods=entry).max()
    low = close.rolling(exit_, min_periods=exit_).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0)


def _run(close: pd.Series, signal: pd.Series, start: str) -> dict:
    """Position, daily net return and buy-and-hold return from `start` on; the rule starts the period from cash."""
    position = signal.shift(1).fillna(0.0)
    change = close.pct_change().fillna(0.0)
    keep = close.index >= pd.Timestamp(start)
    pos, chg = position[keep], change[keep]
    turn = pos.diff().abs()
    turn.iloc[0] = pos.iloc[0]
    net = pos * chg - COST * turn
    return {"pos": pos, "chg": chg, "net": net, "first": int(np.argmax(keep))}


def _parts(net: np.ndarray) -> tuple[float | None, float, float]:
    """(yearly compound return in %, yearly volatility in %, return ÷ volatility)."""
    n = len(net)
    if n < 2:
        return None, 0.0, 0.0
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    return ret, vol, (ret / vol if vol > 1e-9 else 0.0)


def _deepest(net: np.ndarray) -> float:
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    return float((equity / np.maximum.accumulate(equity) - 1).min() * 100)


def _measure(net: pd.Series) -> dict:
    ret, vol, ratio = _parts(net.values)
    return {"ret": ret, "vol": vol, "ratio": ratio, "deepest": _deepest(net.values),
            "growth": float(np.prod(1 + net.values))}


def _trades(close: pd.Series, signal: pd.Series, run: dict) -> list[dict]:
    """Every stay in bitcoin inside the period. Entry price = the close that gave the signal (the day before the
    first held day); exit price = the close that gave the exit signal (the last held day). Result after both costs."""
    pos, net = run["pos"].values, run["net"].values
    days = run["pos"].index
    prices = close.values
    offset = run["first"]  # position of the period's first day in the full series
    n = len(pos)
    out = []
    i = 0
    while i < n:
        if pos[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and pos[j + 1] == 1:
            j += 1
        still_open = j == n - 1 and float(signal.iloc[-1]) == 1
        growth = float(np.prod(1 + net[i:j + 1]))
        if not still_open:
            growth *= 1 - COST  # the exit switch (charged on the following day in the daily account)
        entry_at = offset + i - 1
        out.append({"i": i, "j": j, "open": still_open,
                    "entry_day": close.index[entry_at], "entry": float(prices[entry_at]),
                    "exit_day": days[j], "exit": float(prices[offset + j]),
                    "ret": (growth - 1) * 100, "log": math.log(growth), "held": j - i + 1,
                    "cut": i == 0})  # already in when the period starts: counted from the period's first day
        i = j + 1
    return out


def _trade_stats(trades: list[dict], run: dict) -> dict:
    closed = [t for t in trades if not t["open"]]
    wins = [t for t in closed if t["ret"] > 0]
    losses = [t for t in closed if t["ret"] <= 0]
    streak = longest = 0
    for t in closed:
        streak = streak + 1 if t["ret"] <= 0 else 0
        longest = max(longest, streak)
    best = sorted(closed, key=lambda t: -t["ret"])[:BEST]
    net = run["net"]
    total_log = float(np.log1p(net.values).sum())
    without = net.values.copy()
    for t in best:
        without[t["i"]:min(t["j"] + 2, len(without))] = 0.0  # those days become cash, their two costs disappear
    rule, hold = _measure(net), _measure(run["chg"])

    def mean(items, key):
        return float(np.mean([t[key] for t in items])) if items else None

    def median(items, key):
        return float(np.median([t[key] for t in items])) if items else None

    return {
        "n": len(closed), "wins": len(wins), "losses": len(losses), "open": [t for t in trades if t["open"]],
        "win_mean": mean(wins, "ret"), "win_median": median(wins, "ret"),
        "loss_mean": mean(losses, "ret"), "loss_median": median(losses, "ret"),
        "top": max(closed, key=lambda t: t["ret"]) if closed else None,
        "bottom": min(closed, key=lambda t: t["ret"]) if closed else None,
        "win_days": mean(wins, "held"), "loss_days": mean(losses, "held"), "streak": longest,
        "whipsaw": sum(1 for t in losses if t["held"] <= WHIPSAW_DAYS),
        "whipsaw_cost": float(np.prod([1 + t["ret"] / 100 for t in losses if t["held"] <= WHIPSAW_DAYS])),
        "best": best, "best_share": (sum(t["log"] for t in best) / total_log * 100) if total_log > 1e-9 else None,
        "without": _parts(without)[0], "rule": rule, "hold": hold, "share": float(run["pos"].mean() * 100),
        "first": run["pos"].index[0], "last": run["pos"].index[-1],
    }


# ---------- market types, found with hindsight ----------

def _swings(close: pd.Series) -> list[dict]:
    """Zigzag over the closes: a swing ends at its extreme once the close has come back SWING from it."""
    prices = close.values
    up_back = 1 / (1 - SWING) - 1
    legs = []
    start, extreme, rising = 0, 0, True
    for i in range(1, len(prices)):
        p = prices[i]
        if rising:
            if p >= prices[extreme]:
                extreme = i
            elif p <= prices[extreme] * (1 - SWING):
                legs.append({"a": start, "b": extreme, "up": True, "done": True})
                start, extreme, rising = extreme, i, False
        else:
            if p <= prices[extreme]:
                extreme = i
            elif p >= prices[extreme] * (1 + up_back):
                legs.append({"a": start, "b": extreme, "up": False, "done": True})
                start, extreme, rising = extreme, i, True
    legs.append({"a": start, "b": extreme, "up": rising, "done": False})  # its extreme may still move
    if extreme < len(prices) - 1:
        legs.append({"a": extreme, "b": len(prices) - 1, "up": not rising, "done": False, "tail": True})
    for leg in legs:
        leg["move"] = float(prices[leg["b"]] / prices[leg["a"]] - 1)
        leg["from"], leg["to"] = close.index[leg["a"]], close.index[leg["b"]]
        leg["p0"], leg["p1"] = float(prices[leg["a"]]), float(prices[leg["b"]])
        if not leg["up"] and leg["move"] <= -BIG_FALL:
            leg["kind"] = FALL
        elif leg["up"] and leg["move"] >= BIG_RISE:
            leg["kind"] = RISE
        else:
            leg["kind"] = MID_UP if leg["up"] else MID_DOWN
    return [leg for leg in legs if leg["b"] > leg["a"]]


def _kinds(close: pd.Series, legs: list[dict]) -> pd.Series:
    """The market type of every day (a swing from pivot a to pivot b owns the daily moves of days a+1 … b)."""
    kind = np.array([MID_UP] * len(close), dtype=object)
    for leg in legs:
        kind[leg["a"] + 1:leg["b"] + 1] = leg["kind"]
    return pd.Series(kind, index=close.index)


def _by_kind(run: dict, kinds: pd.Series, trades: list[dict], legs: list[dict]) -> dict:
    pos, net, chg = run["pos"], run["net"], run["chg"]
    kind = kinds.reindex(pos.index).values
    first = pos.index[0]
    out = {}
    for name in KINDS:
        mask = kind == name
        days = int(mask.sum())
        hold = float(np.prod(1 + chg.values[mask])) if days else 1.0
        rule = float(np.prod(1 + net.values[mask])) if days else 1.0
        started = [t for t in trades if kind[t["i"]] == name and not t["open"]]
        n_legs = sum(1 for leg in legs if leg["kind"] == name and leg["to"] >= first)
        out[name] = {
            "per_hold": hold ** (1 / n_legs) if n_legs else None, "per_rule": rule ** (1 / n_legs) if n_legs else None,
            "days": days, "day_share": days / len(pos) * 100, "hold": hold, "rule": rule,
            "ahead": rule / hold if hold > 0 else None, "in": float(pos.values[mask].mean() * 100) if days else None,
            "trades": len(started), "lost": sum(1 for t in started if t["ret"] <= 0),
            "whipsaw": sum(1 for t in started if t["ret"] <= 0 and t["held"] <= WHIPSAW_DAYS),
            "legs": n_legs,
        }
    return out


# ---------- what it costs to follow ----------

def _under_water(net: pd.Series) -> dict:
    equity = np.cumprod(1 + net.values)
    below = equity < np.maximum.accumulate(equity) * (1 - 1e-12)
    spells, i, n = [], 0, len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        spells.append({"days": j - i + 1, "from": net.index[i], "to": net.index[j], "over": j < n - 1})
        i = j + 1
    long_ = [s["days"] for s in spells if s["days"] >= LONG_SPELL]
    deep = equity <= np.maximum.accumulate(equity) * (1 - DEEP / 100)
    return {"share": float(below.mean() * 100), "deep": float(deep.mean() * 100), "longest": max(spells, key=lambda s: s["days"]) if spells else None,
            "long_n": len(long_), "long_median": float(np.median(long_)) if long_ else None,
            "now": spells[-1] if spells and not spells[-1]["over"] else None}


def _worst_year(net: pd.Series) -> dict | None:
    if len(net) <= YEAR:
        return None
    logs = np.concatenate([[0.0], np.cumsum(np.log1p(net.values))])
    window = logs[YEAR:] - logs[:-YEAR]
    k = int(np.argmin(window))
    return {"ret": float((math.exp(window[k]) - 1) * 100), "from": net.index[k], "to": net.index[k + YEAR - 1]}


def _gaps(close: pd.Series, trades: list[dict]) -> dict:
    """Every stay in cash between an exit and the next entry: the price it sold at against the price it bought back."""
    rows = []
    for before, after in zip(trades[:-1], trades[1:]):
        if before["open"] or after["cut"]:
            continue
        low = float(close[before["exit_day"]:after["entry_day"]].min())
        rows.append({"diff": (after["entry"] / before["exit"] - 1) * 100, "from_low": (after["entry"] / low - 1) * 100})
    if not rows:
        return {"n": 0}
    diffs = [g["diff"] for g in rows]
    return {"n": len(rows), "higher": sum(1 for v in diffs if v > 0), "mean": float(np.mean(diffs)),
            "median": float(np.median(diffs)), "missed": sum(1 for g in rows if g["from_low"] >= MISSED * 100),
            "missed_higher": sum(1 for g in rows if g["diff"] >= MISSED * 100)}


# ---------- everything that depends on the data ----------

def _compute(close: pd.Series) -> dict:
    signal = _signal(close, ENTRY, EXIT)
    legs = _swings(close)
    kinds = _kinds(close, legs)
    periods = {}
    for start, _name in PERIODS:
        run = _run(close, signal, start)
        trades = _trades(close, signal, run)
        periods[start] = {"run": run, "trades": trades, "stats": _trade_stats(trades, run),
                          "kinds": _by_kind(run, kinds, trades, legs), "gaps": _gaps(close, trades),
                          "water": _under_water(run["net"]), "hold_water": _under_water(run["chg"]),
                          "worst": _worst_year(run["net"]), "hold_worst": _worst_year(run["chg"])}

    grid = {}
    for entry in GRID_ENTRY:
        for exit_ in GRID_EXIT:
            grid[(entry, exit_)] = _measure(_run(close, _signal(close, entry, exit_), MAIN)["net"])

    # today
    last = close.index[-1]
    held = float(signal.iloc[-1]) == 1
    flips = signal[signal != signal.shift(1)]
    since = flips.index[-1]
    up_line = float(close.iloc[-(ENTRY - 1):].max())  # tomorrow's close at or above this is a 20-close high
    down_line = float(close.iloc[-(EXIT - 1):].min())  # tomorrow's close at or below this is a 10-close low
    return {"signal": signal, "legs": legs, "periods": periods, "grid": grid, "last": last, "held": held,
            "since": since, "since_price": float(close[since]), "up_line": up_line, "down_line": down_line,
            "price": float(close.iloc[-1])}


# ---------- the study ----------

def study() -> dict:
    close = _daily_close()
    r = _compute(close)
    last, price = r["last"], r["price"]
    main = r["periods"][MAIN]
    s, hold = main["stats"], main["stats"]["hold"]
    span = {start: f"{_d(p['stats']['first'])[:7]} 到 {_d(last)}" for start, p in r["periods"].items()}

    # ----- table 1: the trade list, summarised -----
    def trade_text(t: dict | None) -> str:
        if not t:
            return "—"
        return f"{_pct(t['ret'])}（{_d(t['entry_day'])} 進、{_d(t['exit_day'])} 出，抱 {t['held']} 天）"

    def column(start: str) -> list[str]:
        st = r["periods"][start]["stats"]
        n = st["n"]
        open_text = "，另有 1 筆還沒結束" if st["open"] else ""
        return [
            span[start],
            f"{n} 筆{open_text}",
            f"{n} 筆裡 {st['wins']} 筆賺、{st['losses']} 筆賠" if n else "—",
            f"{_pct(st['win_mean'])} ／ {_pct(st['win_median'])}",
            f"{_pct(st['loss_mean'])} ／ {_pct(st['loss_median'])}",
            trade_text(st["top"]),
            trade_text(st["bottom"]),
            f"{_num(st['win_days'], 0)} 天 ／ {_num(st['loss_days'], 0)} 天",
            f"{st['streak']} 筆",
f"{st['losses']} 筆賠的裡面 {st['whipsaw']} 筆，這幾筆接起來共 {_pct((st['whipsaw_cost'] - 1) * 100)}",
            (f"{_pct(st['best_share'], signed=False)}" if _ok(st["best_share"]) else "—"),
            f"{_pct(st['rule']['ret'])} ／ {_pct(st['without'])} ／ {_pct(st['hold']['ret'])}",
            f"{_pct(st['rule']['deepest'])} ／ {_pct(st['hold']['deepest'])}",
            f"{_num(st['rule']['ratio'])} ／ {_num(st['hold']['ratio'])}",
            f"{st['share']:.0f}%",
        ]

    names = ["期間", "已經結束的交易", "賺錢的有幾筆", "賺的那幾筆：平均 ／ 中間值", "賠的那幾筆：平均 ／ 中間值",
             "賺最多的一筆", "賠最多的一筆", "平均抱幾天：賺的 ／ 賠的", "最多連續賠幾筆",
             f"{WHIPSAW_DAYS} 天內就賠錢出場（短進短出）", f"最好的 {BEST} 筆佔全部成長的比例",
             f"每年報酬：規則 ／ 拿掉最好 {BEST} 筆 ／ 一直抱著", "最深曾經虧多少：規則 ／ 一直抱著",
             "報酬÷波動：規則 ／ 一直抱著", "持有比特幣的天數比例"]
    cols = [column(start) for start, _ in PERIODS]
    summary_rows = [[name] + [c[k] for c in cols] for k, name in enumerate(names)]

    # ----- table 2: where the edge comes from -----
    kind_rows = []
    for start, name in PERIODS:
        p = r["periods"][start]
        for kind in KINDS:
            k = p["kinds"][kind]
            kind_rows.append([name, kind, f"{k['legs']} 段，{k['day_share']:.0f}% 的日子", _total(k["per_hold"]),
                              _total(k["per_rule"]), _times(k["ahead"]), (f"{k['in']:.0f}%" if _ok(k["in"]) else "—"),
                              f"{k['trades']} 筆裡 {k['whipsaw']} 筆" if k["trades"] else "—"])
        st = p["stats"]
        kind_rows.append([name, "全部合計", "100% 的日子", "整段 " + _total(st["hold"]["growth"]),
                          "整段 " + _total(st["rule"]["growth"]),
                          _times(st["rule"]["growth"] / st["hold"]["growth"]), f"{st['share']:.0f}%",
                          f"{st['n']} 筆裡 {st['whipsaw']} 筆"])

    # ----- table 3: the grid around 20/10 -----
    grid = r["grid"]
    beat = sum(1 for m in grid.values() if m["ratio"] > hold["ratio"])
    grid_rows = []
    for entry in GRID_ENTRY:
        row = [f"創 {entry} 日新高進場"]
        for exit_ in GRID_EXIT:
            m = grid[(entry, exit_)]
            mark = " ★" if (entry, exit_) == (ENTRY, EXIT) else ""
            row.append(f"{_num(m['ratio'])} ／ {_pct(m['deepest'])}{mark}")
        grid_rows.append(row)
    ratios = sorted((m["ratio"] for m in grid.values()), reverse=True)
    place = ratios.index(grid[(ENTRY, EXIT)]["ratio"]) + 1

    # ----- table 4: the last trades -----
    trades = r["periods"][PERIODS[0][0]]["trades"]
    trade_rows = []
    for t in trades[-LAST_TRADES:]:
        if t["open"]:
            trade_rows.append([_d(t["entry_day"]), _usd(t["entry"]), "還沒結束", f"（{_d(last)} 收盤 {_usd(price)}）",
                               f"{_pct(t['ret'])}（到目前）", f"{t['held']} 天"])
        else:
            trade_rows.append([_d(t["entry_day"]), _usd(t["entry"]), _d(t["exit_day"]), _usd(t["exit"]),
                               _pct(t["ret"]), f"{t['held']} 天"])

    tables = [
        {"caption": (f"規則：收盤創 {ENTRY} 天新高，隔天起持有比特幣；收盤創 {EXIT} 天新低，隔天起持有現金。把每一次「進場到出場」當成一筆交易，"
                     f"結果已扣進出各 {COST * 100:.2f}% 成本。「中間值」是把每筆由小排到大、排在正中間的那一筆。"
                     f"「最好的 {BEST} 筆佔全部成長的比例」用複利算，超過 100% 代表其餘的交易合起來是賠的。"),
         "columns": ["項目"] + [name for _, name in PERIODS], "rows": summary_rows},
        {"caption": (f"優勢從哪裡來：事後回頭把行情切成一段一段（收盤從高點回跌 {SWING * 100:.0f}% 才算一段上漲結束，"
                     f"從低點回升 {(1 / (1 - SWING) - 1) * 100:.0f}% 才算一段下跌結束），高點到低點跌 {BIG_FALL * 100:.0f}% 以上的算「大跌」，"
                     f"低點到高點漲 {BIG_RISE * 100:.0f}% 以上的算「大漲」，沒到門檻的上漲段和下跌段就是其餘（一般說的盤整、來回）。"
                     "「平均每一段」是把同一類每一段的結果用複利平均（不滿一段的頭尾也算一段）；"
                     "「規則的錢是一直抱著的幾倍」是把那一類的日子全部接起來之後兩者的差距，大於 1 代表規則領先，小於 1 代表落後，"
                     "四類相乘就是全部合計那一列（接起來算複利，所以數字很極端）。"
                     f"最後一欄是在那一類日子進場的交易裡，{WHIPSAW_DAYS} 天內就賠錢出場的筆數。"),
         "columns": ["期間", "行情類型", "段數與天數", "比特幣平均每一段", "規則平均每一段", "全部接起來，規則的錢是一直抱著的幾倍",
                     "規則持有比特幣的天數比例", "短進短出賠錢"],
         "rows": kind_rows},
        {"caption": (f"換成鄰近的參數會怎樣（{span[MAIN]}）。每格是「報酬÷波動 ／ 最深曾經虧多少」，★ 是本頁的 {ENTRY}／{EXIT}。"
                     f"同期間一直抱著是 {_num(hold['ratio'])} ／ {_pct(hold['deepest'])}。"
                     f"{len(grid)} 格裡有 {beat} 格的報酬÷波動高過一直抱著；{len(grid)} 格最低 {_num(ratios[-1])}、最高 {_num(ratios[0])}，"
                     f"{ENTRY}／{EXIT} 排第 {place}。"),
         "columns": ["進場 ＼ 出場"] + [f"創 {x} 日新低出場" for x in GRID_EXIT], "rows": grid_rows},
        {"caption": f"最近 {len(trade_rows)} 筆交易（價格是發出訊號那一天的收盤價，美元；結果已扣成本）。",
         "columns": ["進場訊號日", "進場價", "出場訊號日", "出場價", "結果", "抱了幾天"], "rows": trade_rows},
    ]

    # ----- now -----
    held = r["held"]
    line = r["down_line"] if held else r["up_line"]
    away = (line / price - 1) * 100
    if held and line >= price:
        flip_note = "（最後一根收盤已經是這 9 天最低，下一根只要不高於它就會轉成現金）"
    elif not held and line <= price:
        flip_note = "（最後一根收盤已經是這 19 天最高，下一根只要不低於它就會轉成持有）"
    else:
        flip_note = ""
    water, hold_water = main["water"], main["hold_water"]
    worst, hold_worst = main["worst"], main["hold_worst"]

    def spell(w: dict) -> str:
        longest = w["longest"]
        if not longest:
            return "—"
        tail = "" if longest["over"] else "，到現在還沒回到"
        return f"{longest['days']} 天（{_d(longest['from'])} 到 {_d(longest['to'])}{tail}）"

    def gap_text(g: dict, key: str) -> str:
        if not g.get("n"):
            return "—"
        if key == "higher":
            return (f"{g['n']} 次裡 {g['higher']} 次買回來比賣掉貴；買回價和賣出價平均相差 {_pct(g['mean'])}"
                    f"（中間值 {_pct(g['median'])}）")
        return (f"{g['n']} 次裡 {g['missed']} 次（其中買回價比當初賣出價高 {MISSED * 100:.0f}% 以上的有 {g['missed_higher']} 次）")

    both = lambda key: "；".join(f"{name}：{gap_text(r['periods'][start]['gaps'], key)}" for start, name in PERIODS)  # noqa: E731
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {_usd(price)} 美元"},
        {"label": "規則下一天的狀態", "value": ("持有比特幣" if held else "持有現金")
         + f"（{_d(r['since'])} 收盤 {_usd(r['since_price'])} 美元發出訊號，隔天起）"},
        {"label": ("會讓規則轉成現金的收盤價" if held else "會讓規則轉成持有比特幣的收盤價"),
         "value": (f"下一根日線收在 {_usd(line)} 美元或以{'下' if held else '上'}"
                   f"（離最後收盤 {_pct(away)}）{flip_note}")},
        {"label": f"規則的帳戶低於自己先前最高點的日子（{PERIODS[1][1]}）",
         "value": (f"每 100 天裡約 {water['share']:.0f} 天（一直抱著約 {hold_water['share']:.0f} 天）；"
                   f"其中比先前最高點低 {DEEP}% 以上的約 {water['deep']:.0f} 天（一直抱著約 {hold_water['deep']:.0f} 天）")},
        {"label": "最久一次多久才回到先前最高點", "value": f"{spell(water)}；一直抱著最久 {spell(hold_water)}"},
        {"label": f"持續 {LONG_SPELL} 天以上的「低於前高」", "value": (
            f"{water['long_n']} 次，中間值 {_num(water['long_median'], 0)} 天" if water["long_n"] else "—")},
        {"label": "最差的連續 12 個月", "value": (
            f"規則 {_pct(worst['ret'])}（{_d(worst['from'])} 到 {_d(worst['to'])}）；"
            f"一直抱著 {_pct(hold_worst['ret'])}（{_d(hold_worst['from'])} 到 {_d(hold_worst['to'])}）"
            if worst and hold_worst else "—")},
        {"label": "賣掉之後再買回來的價格", "value": both("higher")},
        {"label": f"持有現金期間，比特幣從那段的最低收盤漲了 {MISSED * 100:.0f}% 以上規則才買回", "value": both("missed")},
    ]
    if water["now"]:
        now.insert(6, {"label": "目前這一次低於前高已經", "value": f"{water['now']['days']} 天（從 {_d(water['now']['from'])} 起）"})

    # ----- verdict -----
    kinds = main["kinds"]
    fall, rise, mid_up, mid_down = (kinds[k] for k in KINDS)
    mid_hold, mid_rule = mid_up["hold"] * mid_down["hold"], mid_up["rule"] * mid_down["rule"]
    mid_days = mid_up["day_share"] + mid_down["day_share"]
    mid_trades, mid_whip = mid_up["trades"] + mid_down["trades"], mid_up["whipsaw"] + mid_down["whipsaw"]
    mid_ahead = mid_rule / mid_hold if mid_hold > 0 else None
    if _ok(fall["ahead"]) and fall["ahead"] > 1 and _ok(rise["ahead"]) and rise["ahead"] <= 1:
        source = "優勢全部來自躲掉大跌，不是抓到大漲"
    elif _ok(fall["ahead"]) and fall["ahead"] > 1:
        source = ("優勢主要來自躲掉大跌，大漲時也領先" if fall["ahead"] >= rise["ahead"] else "優勢主要來自大漲，躲掉大跌的幫助比較小")
    elif _ok(rise["ahead"]) and rise["ahead"] > 1:
        source = "優勢來自大漲，大跌並沒有躲掉"
    else:
        source = "大跌和大漲時都沒有領先"
    if _ok(mid_ahead) and mid_ahead < 1:
        mid_word = "落後得最多" if _ok(rise["ahead"]) and mid_ahead < rise["ahead"] else "也落後"
    else:
        mid_word = "沒有落後"
    win_loss = (f"賺的平均 {_pct(s['win_mean'])}、賠的平均 {_pct(s['loss_mean'])}"
                if _ok(s["win_mean"]) and _ok(s["loss_mean"]) else "")
    share_text = (f"最好的 {BEST} 筆就佔了全部成長的 {_pct(s['best_share'], signed=False)}，" if _ok(s["best_share"]) else "")
    verdict = (
        f"{span[MAIN]}，這條規則做完的 {s['n']} 筆交易裡只有 {s['wins']} 筆賺（{win_loss}），最多連賠 {s['streak']} 筆；"
        f"{share_text}拿掉這 {BEST} 筆，每年報酬會從 {_pct(s['rule']['ret'])} 變成 {_pct(s['without'])}"
        f"（一直抱著 {_pct(hold['ret'])}）。"
        f"{source}：{fall['legs']} 段大跌比特幣平均每段 {_total(fall['per_hold'])}、規則平均 {_total(fall['per_rule'])}"
        f"（那些日子規則只有 {fall['in']:.0f}% 的時間在場）；{rise['legs']} 段大漲比特幣平均每段 {_total(rise['per_hold'])}、"
        f"規則只拿到 {_total(rise['per_rule'])}。"
        f"其餘中小型來回的日子（佔 {mid_days:.0f}%）規則{mid_word}：全部接起來比特幣{_total(mid_hold)}、規則 {_total(mid_rule)}，"
        f"在這類日子進場的 {mid_trades} 筆裡有 {mid_whip} 筆是 {WHIPSAW_DAYS} 天內就賠錢出場"
        f"（全期這種短進短出共 {s['whipsaw']} 筆，接起來 {_pct((s['whipsaw_cost'] - 1) * 100)}）。"
        f"跟著做的代價：規則的帳戶最久 {water['longest']['days'] if water['longest'] else '—'} 天沒有回到自己先前的最高點，"
        f"賣掉之後 {main['gaps'].get('n', 0)} 次裡 {main['gaps'].get('higher', 0)} 次是用更貴的價格買回來。"
    )

    # ----- notes -----
    def leg_list(kind: str) -> str:
        first = pd.Timestamp(PERIODS[0][0])
        items = [f"{_d(leg['from'])}→{_d(leg['to'])} {_pct(leg['move'] * 100)}" + ("（還沒確定結束）" if not leg["done"] else "")
                 for leg in r["legs"] if leg["kind"] == kind and leg["to"] >= first]
        return f"{len(items)} 段：" + "；".join(items) if items else "沒有"

    problems = [text for text in [_cross_check(close)] if text]
    cut = [name for start, name in PERIODS if r["periods"][start]["trades"] and r["periods"][start]["trades"][0]["cut"]]
    notes = [
        f"規則只用收盤價：當天收盤是含當天在內最近 {ENTRY} 個收盤裡最高的，隔天起持有比特幣；是最近 {EXIT} 個收盤裡最低的，隔天起持有現金；"
        f"其他日子維持原狀。每次切換扣 {COST * 100:.2f}% 成本。只用已經收完的日線。"
        "「每年報酬」是把整段總成果換算成每年的複利；「報酬÷波動」是每年報酬除以每年的起伏，越高代表每承受一分起伏換到的報酬越多；"
        "「最深曾經虧多少」是從最高點往下掉最多的幅度。",
        "大跌、大漲和盤整是事後回頭看才分得出來的（要等行情走完才知道高點和低點在哪），這裡只拿來解釋規則的優勢從哪裡來；"
        "規則本身沒有用到這個分類，當下也沒有人能確定自己正處在哪一類。分界的數字（回跌 "
        f"{SWING * 100:.0f}% 算一段結束、跌 {BIG_FALL * 100:.0f}% 算大跌、漲 {BIG_RISE * 100:.0f}% 算大漲）是人為訂的，換一組數字各類的天數和合計都會變，"
        "比特幣所謂的盤整其實是幅度不小的來回（漲幾成、跌兩成多），所以「其餘」分成上漲段和下跌段兩列；最後一段行情還沒走完，分類之後可能改變。"
        "把同一類的日子接在一起算複利，數字極端是正常的，重點是規則和一直抱著的相對差距。",
        f"2013 年以來的大跌（收盤高點→收盤低點）{leg_list(FALL)}。",
        f"2013 年以來的大漲（收盤低點→收盤高點）{leg_list(RISE)}。",
        f"只測了比特幣一種資產，交易筆數也不多（{PERIODS[1][1]}只有 {s['n']} 筆），而且成果集中在少數幾筆大賺的交易上——"
        "少遇到一兩次大行情，結果就會差很多；同一條規則換到別的資產不一定一樣。",
        f"{ENTRY}／{EXIT} 這一組是「每年只用當時以前的資料挑參數」那個測試挑出來的，不是這一頁回頭挑的；"
        "但當初要測哪幾類規則、每類試哪些參數，是用今天的知識選的，所以整件事還是留有一些事後才知道的成分，數字可能比真正事先能做到的好看。"
        "第三張表的鄰近參數則全部是事後回頭算的，只用來看結果是不是只靠某一格。",
        f"成本只用一個粗略的數字（每次切換 {COST * 100:.2f}%），實際的手續費、買賣價差和滑價會隨交易所、金額和年代不同；"
        "持有現金假設沒有任何利息，沒有算稅；一直抱著沒有扣成本。實際照做是在收盤後才知道訊號，成交價不會剛好等於那一根的收盤價。",
        "每個期間都從「手上是現金」開始算"
        + (f"；{'、'.join(cut)}開始那天規則本來就在場內，那一筆從期間第一天算起，進場價用前一天的收盤。" if cut else "。")
        + "還沒結束的那一筆不算進「幾筆裡幾筆賺」，但算在每年報酬裡。"
        f"「低於先前最高點」是每天收盤時規則的帳戶比它自己之前的最高紀錄低；「最差的連續 12 個月」是任選連續 {YEAR} 天裡最差的一段。",
        "「會讓規則轉向的收盤價」是照定義換算的：下一根的收盤要成為含它自己在內 "
        f"{EXIT} 個收盤裡最低（或 {ENTRY} 個收盤裡最高），所以門檻是前面 {EXIT - 1}（或 {ENTRY - 1}）個收盤的最低（最高）。這只是描述規則的機械狀態。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）。"
        "blockchain.info 標示的日期大約比幣安晚一天，整段內部順序一致、不會用到未來的價格，但 2017-08 以前的交易日期可能差一天，"
        "接縫那一天等於合併了兩天的漲跌。用的是收盤價的新高新低，不是盤中最高最低價。",
        "過去的結果不保證未來會一樣。這裡只是整理這條規則過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "trend_anatomy",
        "title": "順勢規則一筆一筆看：優勢從哪裡來、跟著做要忍受什麼",
        "question": f"「創 {ENTRY} 天新高進場、創 {EXIT} 天新低出場」這條規則，實際跟著做是什麼感覺？它是常常賺，還是靠少數幾筆？優勢是躲掉大跌還是抓到大漲？",
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
