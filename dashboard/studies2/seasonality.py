"""Historical study "seasonality": does bitcoin really do better at certain hours, on certain weekdays, around the
turn of the month, in certain months ("Uptober") or on weekends?

Every calendar pattern is measured on the whole sample, on its first and second half separately and year by year; a
pattern only counts when it points the same way in both halves and in most years. Then the same question is asked
the honest way round: at the start of each year from 2020 the best session / weekday / month by the data BEFORE that
year is picked and held during the year only (with and without a 0.10% cost per switch), and a shuffle test shows how
good the best of 24 hours / 7 weekdays / 12 months looks by luck alone.

Data: Binance BTCUSDT 1-hour spot candles since 2017-08 (kept in cache/_store/seasonality_hourly.csv so later runs
only fetch the new hours) for hours, sessions, weekdays and the turn of the month; the long daily series of
collectors.classic (blockchain.info before 2017-08, Binance after) for the month-of-the-year test. Only closed candles
and finished UTC days are used. Everything comes from public, key-less endpoints through common.get_json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import price as px

STORE_PATH = common.STORE / "seasonality_hourly.csv"
HOUR_MS = 3_600_000
FIRST_MS = 1_502_928_000_000  # 2017-08-17 00:00 UTC, the day Binance listed BTCUSDT
MAX_PAGES = 130
MIN_HOURS = 50_000
COST = 0.001  # 0.10% of the amount moved at every switch between bitcoin and cash
YEAR = 365
FIRST_PICK_YEAR = 2020
MONTHS_SINCE = 2011  # month-of-the-year test: full calendar months from this year on
SIMS = 500
SEED = 20170817
MOST = 2 / 3  # "most years"

TW = 8  # Taiwan is UTC+8
SESSIONS = ("亞洲時段（台灣早上 8 點–下午 4 點）", "歐洲時段（台灣下午 4 點–半夜 12 點）",
            "美洲時段（台灣半夜 12 點–早上 8 點）")
SESSION_SHORT = ("亞洲時段", "歐洲時段", "美洲時段")
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
MONTH_NAMES = tuple(f"{m} 月" for m in range(1, 13))


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, digits: int = 1, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.5 * 10 ** -digits:
        return f"{0:.{digits}f}%"
    if abs(v) >= 100:
        digits = 0
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _pp(v, digits: int = 1) -> str:
    """A difference between two percentages, in percentage points."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.5 * 10 ** -digits:
        return f"{0:.{digits}f}"
    if abs(v) >= 100:
        digits = 0
    return f"{v:+,.{digits}f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _tw_hour(h: int) -> str:
    """UTC hour h (the hour starting at h:00) as Taiwan clock time."""
    t = (h + TW) % 24
    part = "半夜" if t == 0 else "凌晨" if t < 6 else "早上" if t < 12 else "中午" if t == 12 else "下午" if t < 18 else "晚上"
    return f"台灣{part} {12 if t % 12 == 0 else t % 12} 點"


def _hour_label(h: int) -> str:
    return f"{_tw_hour(h)}起的一小時"


# ---------- data ----------

def _read_store() -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    try:
        for line in STORE_PATH.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            o, c = float(parts[1]), float(parts[2])
            if o > 0 and c > 0 and math.isfinite(o) and math.isfinite(c):
                rows[int(parts[0])] = (o, c)
    except Exception:  # noqa: BLE001 - first run or a damaged file: start over
        return {}
    return rows


def _write_store(rows: dict[int, tuple[float, float]]) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        text = "ms,open,close\n" + "\n".join(f"{ms},{o!r},{c!r}" for ms, (o, c) in sorted(rows.items())) + "\n"
        STORE_PATH.write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run simply fetches again
        pass


def _hourly() -> tuple[pd.DataFrame, str | None]:
    """Closed BTCUSDT 1-hour candles (open time, open, close), oldest first, and a problem text if the fetch failed."""
    rows = _read_store()
    start = max(rows) if rows else FIRST_MS  # the last stored hour is fetched again
    problem, added = None, 0
    try:
        for page_no in range(MAX_PAGES):
            page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
                                                  "startTime": start})
            if not page:
                break
            now_ms = int(time.time() * 1000)
            for r in page:
                if int(r[6]) < now_ms and float(r[1]) > 0 and float(r[4]) > 0:  # closed candles only
                    rows[int(r[0])] = (float(r[1]), float(r[4]))
                    added += 1
            if len(page) < 1000:
                break
            start = int(page[-1][0]) + 1
            if page_no % 20 == 19:
                _write_store(rows)  # keep what a long first run has fetched so far
    except Exception as exc:  # noqa: BLE001 - a stored history is still usable
        problem = f"幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料，沒有改用其他方法繞過。"
    if added:
        _write_store(rows)
    if len(rows) < MIN_HOURS:
        raise RuntimeError(f"幣安 1 小時線只有 {len(rows)} 筆")
    frame = pd.DataFrame([(ms, o, c) for ms, (o, c) in sorted(rows.items())], columns=["ms", "open", "close"])
    return frame, problem


def _monthly_returns() -> pd.DataFrame:
    """Year × month table of full-calendar-month returns in % from the long daily series (NaN = no such month)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[(close.index < pd.Timestamp(common.today_utc())) & (close > 0)]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    period = close.index.to_period("M")
    last = close.groupby(period).last()
    change = (last / last.shift(1) - 1) * 100
    final = close.index[-1]
    complete = final.to_period("M") if (final + pd.Timedelta(days=1)).month != final.month else final.to_period("M") - 1
    change = change[(change.index.year >= MONTHS_SINCE) & (change.index <= complete)].dropna()
    table = pd.DataFrame({"y": change.index.year, "m": change.index.month, "r": change.values})
    return table.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))


# ---------- measuring ----------

def _mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _judge(e1, e2, positive: int, years: int) -> tuple[str, bool]:
    """(text, counts): same sign in both halves and the same direction in most years."""
    if not (_ok(e1) and _ok(e2)) or years == 0:
        return "—", False
    if e1 * e2 <= 0:
        return "前後相反", False
    up = e1 > 0
    agree = positive if up else years - positive
    if agree / years >= MOST:
        return ("一致偏高" if up else "一致偏低"), True
    return "兩半同向，逐年不穩", False


def _annual(log_sum: float, days: float) -> float | None:
    """Compound yearly return in % of a strategy whose log returns over `days` days sum to log_sum."""
    if days <= 0:
        return None
    return float(math.expm1(log_sum * YEAR / days) * 100)


def _compute(frame: pd.DataFrame, months: pd.DataFrame) -> dict:
    """Everything that depends on the data; study() only formats it."""
    ts = pd.to_datetime(frame["ms"].values, unit="ms")
    lr = np.log(frame["close"].values / frame["open"].values)
    first = ts[0].normalize() + (pd.Timedelta(days=1) if ts[0].hour else pd.Timedelta(0))
    end = pd.Timestamp(common.today_utc())  # exclusive: finished UTC days only
    keep = (ts >= first) & (ts < end)
    ts, lr = ts[keep], lr[keep]
    days = pd.date_range(first, end - pd.Timedelta(days=1), freq="D")
    n = len(days)
    grid = np.full((n, 24), np.nan)
    grid[(ts.normalize() - first).days.values, ts.hour.values] = lr
    have = np.isfinite(grid)
    filled = np.where(have, grid, 0.0)
    missing_hours = int((~have).sum())
    day_log = filled.sum(axis=1)
    day_ok = have.any(axis=1)
    day_ret = np.where(day_ok, np.expm1(day_log) * 100, np.nan)  # % per UTC day
    hour_ret = np.expm1(grid) * 100  # % per hour, NaN where the candle is missing
    session_log = np.stack([filled[:, 8 * s:8 * s + 8].sum(axis=1) for s in range(3)], axis=1)
    year = days.year.values
    month = days.month.values
    weekday = days.weekday.values
    dom = days.day.values
    to_end = days.days_in_month.values - dom
    turn = (dom <= 3) | (to_end <= 1)  # first three and last two calendar days of the month
    weekend = weekday >= 5
    years = sorted(set(year.tolist()))
    half = n // 2
    whole = np.ones(n, dtype=bool)
    h1 = np.arange(n) < half
    h2 = ~h1
    year_masks = [year == y for y in years]

    def row(name: str, level, excess, digits: int) -> dict:
        """level(mask) and excess(mask) -> numbers for the whole sample, both halves and every year."""
        by_year = [excess(m) for m in year_masks]
        by_year = [v for v in by_year if _ok(v)]
        positive = sum(1 for v in by_year if v > 0)
        e1, e2 = excess(h1), excess(h2)
        text, counts = _judge(e1, e2, positive, len(by_year))
        return {"name": name, "whole": level(whole), "h1": level(h1), "h2": level(h2), "e": excess(whole), "e1": e1,
                "e2": e2, "positive": positive, "years": len(by_year), "judge": text, "counts": counts,
                "digits": digits}

    # ----- sessions: yearly return if bitcoin is held only during that session, every day -----
    def session_level(s: int):
        return lambda m: _annual(float(session_log[m, s].sum()), int(m.sum()))

    def session_excess(s: int):
        def f(m):
            levels = [_annual(float(session_log[m, k].sum()), int(m.sum())) for k in range(3)]
            return None if None in levels else levels[s] - float(np.mean(levels))
        return f

    sessions = [row(SESSIONS[s], session_level(s), session_excess(s), 1) for s in range(3)]

    # ----- hours: average hourly return -----
    def hour_level(h: int):
        return lambda m: _mean(hour_ret[m, h])

    def hour_excess(h: int):
        def f(m):
            a, b = _mean(hour_ret[m, h]), _mean(hour_ret[m, :])
            return None if a is None or b is None else a - b
        return f

    hours = [row(_hour_label(h), hour_level(h), hour_excess(h), 3) for h in range(24)]

    # ----- days: a group of days against all days, or against the other days -----
    def day_level(member):
        return lambda m: _mean(day_ret[m & member])

    def day_excess(member, other):
        def f(m):
            a, b = _mean(day_ret[m & member]), _mean(day_ret[m & other])
            return None if a is None or b is None else a - b
        return f

    weekdays = [row(WEEKDAYS[k], day_level(weekday == k), day_excess(weekday == k, whole), 2) for k in range(7)]
    pairs = [row("週末（六、日）", day_level(weekend), day_excess(weekend, ~weekend), 2),
             row("平日（一到五）", day_level(~weekend), day_excess(~weekend, weekend), 2),
             row("月底 2 天＋月初 3 天", day_level(turn), day_excess(turn, ~turn), 2),
             row("每個月其他的日子", day_level(~turn), day_excess(~turn, turn), 2)]

    # ----- months of the year (long daily series) -----
    m_years = [int(y) for y in months.index]
    m_half = len(m_years) // 2
    first_years, second_years = m_years[:m_half], m_years[m_half:]
    values = months.values  # (years, 12), NaN where the month is not there
    all_median = float(np.nanmedian(values))
    med1, med2 = float(np.nanmedian(values[:m_half])), float(np.nanmedian(values[m_half:]))
    year_median = np.nanmedian(values, axis=1)
    month_rows = []
    for m in range(12):
        col = values[:, m]
        okc = np.isfinite(col)
        a, b = col[:m_half], col[m_half:]
        ma = float(np.nanmedian(a)) if np.isfinite(a).any() else None
        mb = float(np.nanmedian(b)) if np.isfinite(b).any() else None
        beat = int((col[okc] > year_median[okc]).sum())
        text, counts = _judge(None if ma is None else ma - med1, None if mb is None else mb - med2, beat,
                              int(okc.sum()))
        month_rows.append({"name": MONTH_NAMES[m], "n": int(okc.sum()), "up": int((col[okc] > 0).sum()),
                           "median": float(np.median(col[okc])), "mean": float(col[okc].mean()),
                           "m1": ma, "m2": mb, "beat": beat, "judge": text, "counts": counts})

    # ----- the pick-in-advance test -----
    last_year = years[-1]
    test_years = [y for y in range(FIRST_PICK_YEAR, last_year + 1)]
    month_values = {y: months.loc[y].values if y in months.index else np.full(12, np.nan) for y in m_years}

    def pick_session(before: int) -> int:
        return int(np.argmax(session_log[year < before].sum(axis=0)))

    def pick_weekday(before: int) -> int:
        m = year < before
        return int(np.argmax([_mean(day_ret[m & (weekday == k)]) for k in range(7)]))

    def pick_month(before: int) -> int:
        past = np.array([month_values[y] for y in m_years if y < before])
        return int(np.nanargmax(np.nanmedian(past, axis=0)))

    picks = {y: {"session": pick_session(y), "weekday": pick_weekday(y), "month": pick_month(y)}
             for y in test_years + [last_year + 1]}
    test = year >= FIRST_PICK_YEAR
    t_days = int(test.sum())
    t_year = year[test]
    flat = filled[test].reshape(-1)  # hourly log returns of the test period, in time order
    flat_year = np.repeat(t_year, 24)
    flat_session = np.tile(np.repeat(np.arange(3), 8), t_days)
    flat_weekday = np.repeat(weekday[test], 24)
    flat_month = np.repeat(month[test] - 1, 24)

    def run(position: np.ndarray) -> dict:
        """Hold bitcoin during the hours where position is 1. Starts from cash; every change is one switch."""
        position = position.astype(float)
        moves = np.abs(np.diff(np.concatenate([[0.0], position])))
        gross_log = flat * position
        total, switches = float(gross_log.sum()), float(moves.sum())
        by_year, by_year_net = {}, {}
        for y in test_years:
            m = flat_year == y
            g, s = float(gross_log[m].sum()), float(moves[m].sum())
            by_year[y] = math.expm1(g) * 100
            by_year_net[y] = math.expm1(g + s * math.log(1 - COST)) * 100
        if switches > 0 and total > 0:
            breakeven = (1 - math.exp(-total / switches)) * 100  # the cost per switch that leaves exactly nothing
        else:
            breakeven = None
        return {"gross": _annual(total, t_days), "net": _annual(total + switches * math.log(1 - COST), t_days),
                "switches": switches * YEAR / t_days, "breakeven": breakeven, "share": float(position.mean() * 100),
                "years": by_year, "years_net": by_year_net, "log": total}

    hold = run(np.ones(len(flat)))
    kinds = {}
    for kind, labels, flat_kind in (("session", SESSION_SHORT, flat_session), ("weekday", WEEKDAYS, flat_weekday),
                                    ("month", MONTH_NAMES, flat_month)):
        chosen = np.array([picks[y][kind] for y in flat_year])
        pick = run(flat_kind == chosen)
        each = [run(flat_kind == k) for k in range(len(labels))]
        average = {key: float(np.mean([e[key] for e in each])) for key in ("gross", "net", "switches", "share")}
        costs = [e["breakeven"] for e in each]
        average["breakeven"] = float(np.mean(costs)) if all(_ok(c) for c in costs) else None
        average["years"] = {y: float(np.mean([e["years"][y] for e in each])) for y in test_years}
        pick["beat_average"] = sum(1 for y in test_years if pick["years"][y] > average["years"][y])
        pick["beat_hold_net"] = sum(1 for y in test_years if pick["years_net"][y] > hold["years"][y])
        pick["rank"] = {y: 1 + sum(1 for e in each if e["years"][y] > each[picks[y][kind]]["years"][y])
                        for y in test_years}
        kinds[kind] = {"pick": pick, "average": average, "labels": labels, "count": len(labels)}

    # ----- luck: how good does the best of many look when nothing is real? -----
    rng = np.random.default_rng(SEED)

    def session_year(sums: np.ndarray) -> np.ndarray:
        return np.expm1(sums * YEAR / n) * 100

    real_hour = np.nanmean(hour_ret, axis=0)
    real_session = session_year(session_log.sum(axis=0))
    sim_hour_max, sim_hour_min, sim_session_max = np.empty(SIMS), np.empty(SIMS), np.empty(SIMS)
    for i in range(SIMS):
        mixed = rng.permuted(grid, axis=1)  # each day keeps its own 24 hourly returns, in a random order
        means = np.nanmean(np.expm1(mixed) * 100, axis=0)
        sim_hour_max[i], sim_hour_min[i] = means.max(), means.min()
        mixed = np.where(np.isfinite(mixed), mixed, 0.0)
        sim_session_max[i] = session_year(mixed.reshape(n, 3, 8).sum(axis=(0, 2))).max()

    monday = int(np.argmax(weekday == 0))
    n_weeks = (n - monday) // 7
    weeks = day_ret[monday:monday + 7 * n_weeks].reshape(n_weeks, 7)  # full Monday-to-Sunday weeks
    real_week = np.nanmean(weeks, axis=0)
    sim_week_max, sim_week_min = np.empty(SIMS), np.empty(SIMS)
    for i in range(SIMS):
        means = np.nanmean(rng.permuted(weeks, axis=1), axis=0)
        sim_week_max[i], sim_week_min[i] = means.max(), means.min()

    real_month = np.nanmedian(values, axis=0)
    sim_month_max, sim_month_min = np.empty(SIMS), np.empty(SIMS)
    for i in range(SIMS):
        mixed = values.copy()
        for r in range(len(mixed)):  # each year keeps its own monthly returns, dealt out to random months
            there = np.isfinite(mixed[r])
            mixed[r, there] = rng.permutation(mixed[r, there])
        medians = np.nanmedian(mixed, axis=0)
        sim_month_max[i], sim_month_min[i] = medians.max(), medians.min()

    def luck(name: str, which: str, real: float, sims: np.ndarray, high: bool, digits: int) -> dict:
        as_extreme = int((sims >= real).sum() if high else (sims <= real).sum())
        return {"name": name, "which": which, "real": float(real), "median": float(np.median(sims)),
                "edge": float(np.percentile(sims, 95 if high else 5)), "count": as_extreme, "high": high,
                "digits": digits}

    hb, hw = int(np.argmax(real_hour)), int(np.argmin(real_hour))
    sb = int(np.argmax(real_session))
    wb, ww = int(np.argmax(real_week)), int(np.argmin(real_week))
    mb, mw = int(np.argmax(real_month)), int(np.argmin(real_month))
    lucks = [
        luck("24 個小時裡平均最高的", _hour_label(hb), real_hour[hb], sim_hour_max, True, 3),
        luck("24 個小時裡平均最低的", _hour_label(hw), real_hour[hw], sim_hour_min, False, 3),
        luck("3 個時段裡每年報酬最高的", SESSION_SHORT[sb], real_session[sb], sim_session_max, True, 1),
        luck("7 個星期幾裡平均最高的", WEEKDAYS[wb], real_week[wb], sim_week_max, True, 2),
        luck("7 個星期幾裡平均最低的", WEEKDAYS[ww], real_week[ww], sim_week_min, False, 2),
        luck("12 個月份裡中位數最高的", MONTH_NAMES[mb], real_month[mb], sim_month_max, True, 1),
        luck("12 個月份裡中位數最低的", MONTH_NAMES[mw], real_month[mw], sim_month_min, False, 1),
    ]

    return {
        "first": days[0], "last": days[-1], "n": n, "half_day": days[half], "years": years,
        "missing_hours": missing_hours, "last_close": float(frame["close"].values[keep][-1]),
        "hold_all": _annual(float(day_log.sum()), n),
        "sessions": sessions, "hours": hours, "weekdays": weekdays, "pairs": pairs,
        "hour_best": hb, "hour_worst": hw,
        "months": month_rows, "m_years": m_years, "first_years": first_years, "second_years": second_years,
        "all_median": all_median, "month_n": int(np.isfinite(values).sum()),
        "test_years": test_years, "picks": picks, "hold": hold, "kinds": kinds, "t_days": t_days,
        "lucks": lucks, "n_weeks": n_weeks,
    }


# ---------- the study ----------

def study() -> dict:
    frame, problem = _hourly()
    months = _monthly_returns()
    r = _compute(frame, months)
    last, years, test_years = r["last"], r["years"], r["test_years"]
    current = years[-1]
    ny, nt = len(years), len(test_years)
    partial = last < pd.Timestamp(f"{current}-12-31")

    # ----- table 1: sessions, weekdays, weekend, turn of the month, best and worst hour -----
    def pattern_row(kind: str, x: dict) -> list:
        d = x["digits"]
        return [f"{kind}：{x['name']}", _pct(x["whole"], d), _pct(x["h1"], d), _pct(x["h2"], d), _pp(x["e1"], d),
                _pp(x["e2"], d), f"{x['years']} 年裡 {x['positive']} 年", x["judge"]]

    best_hour, worst_hour = r["hours"][r["hour_best"]], r["hours"][r["hour_worst"]]
    pattern_rows = ([pattern_row("時段·每年", x) for x in r["sessions"]]
                    + [pattern_row("星期·每天", x) for x in r["weekdays"]]
                    + [pattern_row("每天", x) for x in r["pairs"]]
                    + [pattern_row("最高的小時·每小時", best_hour), pattern_row("最低的小時·每小時", worst_hour)])
    half_text = f"前半段到 {_d(r['half_day'] - pd.Timedelta(days=1))}、後半段從 {_d(r['half_day'])} 起"

    # ----- table 2: months -----
    month_rows = [[m["name"], f"{m['n']} 年裡 {m['up']} 年", _pct(m["median"]), _pct(m["mean"]), _pct(m["m1"]),
                   _pct(m["m2"]), f"{m['n']} 年裡 {m['beat']} 年", m["judge"]] for m in r["months"]]
    fy, sy = r["first_years"], r["second_years"]

    # ----- table 3: pick in advance -----
    hold, kinds = r["hold"], r["kinds"]
    kind_name = {"session": "時段", "weekday": "星期幾", "month": "月份"}

    def cost_text(v) -> str:
        return f"{v:.2f}%" if _ok(v) else "不扣成本就沒有賺"

    def times(v: float) -> str:
        return f"{v:,.0f} 次" if v >= 10 else f"{v:.1f} 次"

    pick_rows = [["一直抱著不動", _pct(hold["gross"]), _pct(hold["gross"]), "0 次", "—", "—", "—", "100%"]]
    for kind in ("session", "weekday", "month"):
        k = kinds[kind]
        p, a = k["pick"], k["average"]
        pick_rows.append([f"{kind_name[kind]}：每年年初事先挑最好的一個，只在那時持有", _pct(p["gross"]), _pct(p["net"]),
                          times(p["switches"]), cost_text(p["breakeven"]), f"{nt} 年裡 {p['beat_average']} 年",
                          f"{nt} 年裡 {p['beat_hold_net']} 年", f"{p['share']:.0f}%"])
        pick_rows.append([f"{kind_name[kind]}：{k['count']} 個各自照做的平均", _pct(a["gross"]), _pct(a["net"]),
                          times(a["switches"]), cost_text(a["breakeven"]), "—", "—", f"{a['share']:.0f}%"])

    # ----- table 4: luck -----
    def luck_view(x: dict) -> str:
        share = x["count"] / SIMS
        if share >= 0.10:
            return "運氣就做得到"
        if share >= 0.05:
            return "只比運氣好一點"
        return "比單靠運氣更極端"

    def luck_row(x: dict) -> list:
        d = x["digits"]
        return [x["name"], x["which"], _pct(x["real"], d), _pct(x["median"], d), _pct(x["edge"], d),
                f"{SIMS} 次裡 {x['count']} 次", luck_view(x)]

    luck_rows = [luck_row(x) for x in r["lucks"]]

    span = f"{_d(r['first'])} 到 {_d(last)}"
    tables = [
        {"caption": (f"時段、星期幾、週末、月底月初，以及 24 個小時裡平均最高和最低的那一個（幣安 1 小時線，{span}；{half_text}）。"
                     "「時段」列是每天只在那 8 小時持有比特幣、其他時間抱現金，換算成每年的報酬（不扣成本）；「星期」「每天」列是那種日子"
                     "平均一天漲跌多少；「小時」列是那一個小時平均漲跌多少。「比對照多幾個百分點」的對照：時段是三個時段的平均，"
                     "星期幾和小時是全部日子／全部小時的平均，週末對平日、月底月初對其他日子則是互相比。前後兩半同方向、"
                     f"而且至少三分之二的年份也同方向才算「一致」。一天是指台灣時間早上 8 點到隔天早上 8 點；"
                     f"{years[0]} 年只有 8 月中以後{f'、{current} 年只到 {last.strftime(chr(37) + 'm-' + chr(37) + 'd')}' if partial else ''}。"),
         "columns": ["項目", "整段", "前半段", "後半段", "前半段比對照多幾個百分點", "後半段比對照多幾個百分點", "比對照高的年數",
                     "前後一致嗎"],
         "rows": pattern_rows},
        {"caption": (f"一年十二個月（長期日線，{r['m_years'][0]}-01 起的完整月份，共 {r['month_n']} 個月；全部月份的報酬中位數是 "
                     f"{_pct(r['all_median'])}）。每個月份只有 {min(m['n'] for m in r['months'])}–{max(m['n'] for m in r['months'])} "
                     f"個觀察值。前半段是 {fy[0]}–{fy[-1]} 年、後半段是 {sy[0]}–{sy[-1]} 年；「前後一致」是指那個月份的中位數在前後兩半"
                     "都高於（或都低於）同一半全部月份的中位數，而且至少三分之二的年份它也排在當年十二個月的上半（或下半）。"),
         "columns": ["月份", "上漲的年數", "報酬中位數", "報酬平均", "前半段中位數", "後半段中位數", "贏過當年一半月份的年數",
                     "前後一致嗎"],
         "rows": month_rows},
        {"caption": (f"事先挑的測試（{FIRST_PICK_YEAR}-01-01 到 {_d(last)}，共 {nt} 個測試年）。每年 1 月 1 日只用那天以前的資料，"
                     "挑出過去報酬最高的時段、平均最高的星期幾、中位數最高的月份，那一年就只在那個時段／那一天／那個月持有比特幣，"
                     f"其他時間抱現金（沒有利息）。成本是每次進出各扣 {COST * 100:.2f}%：只抱一個時段等於每天進出各一次。"
                     "「成本到多少就把報酬吃光」是每次進出的成本高到這個數字時，整段剛好不賺不賠。"),
         "columns": ["做法", "每年報酬（不扣成本）", f"每年報酬（每次進出扣 {COST * 100:.2f}%）", "每年進出幾次", "成本到多少就把報酬吃光",
                     "不扣成本贏過同類平均的年數", "扣成本後贏過一直抱著的年數", "持有比特幣的時間比例"],
         "rows": pick_rows},
        {"caption": (f"運氣檢查：把每一天裡 24 個小時的漲跌隨機對調（每天的總漲跌不變）、把每一週裡 7 天的漲跌隨機對調、"
                     f"把每一年裡各個月的漲跌隨機對調，各做 {SIMS} 次。這樣任何小時、星期幾、月份都沒有真本事，"
                     "再看其中「最好（或最差）的那一個」能有多極端，和實際的最好（最差）比。"
                     "「九成五的情況不會超過」是打亂後有 95% 的次數到不了的那條線。"),
         "columns": ["檢查什麼", "實際是哪一個", "實際數字", "打亂後的中間值", "九成五的情況不會超過", "打亂後不輸實際的次數", "看法"],
         "rows": luck_rows},
    ]

    # ----- today -----
    picks = r["picks"]
    now_pick, next_pick = picks[current], picks[current + 1]

    def names(p: dict) -> str:
        return f"{SESSION_SHORT[p['session']]}、{WEEKDAYS[p['weekday']]}、{MONTH_NAMES[p['month']]}"

    now = [
        {"label": f"最後一個收完的 UTC 日（{_d(last)}，台灣時間隔天早上 8 點收）", "value": f"收在 {r['last_close']:,.0f} 美元"},
        {"label": f"{current} 年年初照程序挑到的（只用到 {current - 1} 年底的資料）", "value": names(now_pick)},
        {"label": f"如果今天就重挑（用到 {_d(last)} 的資料）", "value": names(next_pick)},
        {"label": f"{current} 年挑到的時段對應的台灣時間", "value": SESSIONS[now_pick["session"]]},
        {"label": "這幾格的意思", "value": "只是描述這套挑選程序目前停在哪裡，不代表那個時段、那一天或那個月接下來會比較好"},
    ]

    # ----- verdict -----
    hour_same = sum(1 for x in r["hours"] if _ok(x["e1"]) and _ok(x["e2"]) and x["e1"] * x["e2"] > 0)
    hour_count = sum(1 for x in r["hours"] if x["counts"])
    solid = ([f"{SESSION_SHORT[i]}" for i, x in enumerate(r["sessions"]) if x["counts"]]
             + [x["name"] for x in r["weekdays"] if x["counts"]]
             + [x["name"] for x in r["pairs"][::2] if x["counts"]]
             + [x["name"] for x in r["months"] if x["counts"]])
    tested = 3 + 7 + 2 + 12
    lucks = r["lucks"]
    by_luck = [x for x in lucks if x["count"] / SIMS < 0.05]
    sp, sa = kinds["session"]["pick"], kinds["session"]["average"]
    wp, mp = kinds["weekday"]["pick"], kinds["month"]["pick"]
    if solid:
        part1 = (f"三個時段、七個星期幾、週末、月底月初和十二個月份共 {tested} 項裡，只有 {len(solid)} 項通過「前後兩半同方向、"
                 f"而且三分之二以上的年份也同方向」（{'、'.join(solid)}），其餘不是前後相反就是逐年不穩")
    else:
        part1 = (f"三個時段、七個星期幾、週末、月底月初和十二個月份共 {tested} 項裡，沒有任何一項通過「前後兩半同方向、"
                 "而且三分之二以上的年份也同方向」")
    best_m = max(r["months"], key=lambda m: m["median"])
    oct_m = r["months"][9]
    part1 += (f"；常被提到的 10 月是 {oct_m['n']} 年裡 {oct_m['up']} 年上漲、中位數 {_pct(oct_m['median'])}"
              + (f"（中位數最高的是 {best_m['name']}：{best_m['n']} 年裡 {best_m['up']} 年上漲、{_pct(best_m['median'])}）"
                 if best_m is not oct_m else "，是十二個月裡中位數最高的")
              + f"，但每個月份只有大約 {oct_m['n']} 個觀察值。")
    lh, lw, lm = lucks[0], lucks[3], lucks[5]
    part2 = (f"運氣檢查：實際平均最高的小時（{lh['which']}，每小時 {_pct(lh['real'], 3)}）在打亂 {SIMS} 次裡有 {lh['count']} 次"
             f"被純運氣追平或超過，最高的星期幾（{lw['which']}，每天 {_pct(lw['real'], 2)}）是 {lw['count']} 次，"
             f"最高的月份（{lm['which']}）是 {lm['count']} 次"
             + ("——" + "、".join(f"「{x['name']}」" for x in by_luck) + f"比單靠運氣更極端（{SIMS} 次裡不到 {int(SIMS * 0.05)} 次）"
                if by_luck else "——沒有一項比單靠運氣更極端")
             + f"；24 個小時裡有 {hour_same} 個前後兩半同方向（純運氣也會有一半左右），{hour_count} 個連逐年也一致。")
    part3 = (f"事先挑的測試（{FIRST_PICK_YEAR}–{current}，{nt} 個測試年）：每年年初挑過去最好的時段、只在那 8 小時持有，不扣成本每年 "
             f"{_pct(sp['gross'])}（三個時段平均 {_pct(sa['gross'])}、一直抱著 {_pct(hold['gross'])}），{nt} 年裡 {sp['beat_average']} 年"
             f"贏過三個時段的平均；每次進出扣 {COST * 100:.2f}% 之後變成每年 {_pct(sp['net'])}，"
             + (f"成本只要到每次 {sp['breakeven']:.2f}% 就把報酬全部吃光" if _ok(sp["breakeven"]) else "而且不扣成本就已經沒有賺")
             + f"；挑星期幾是每年 {_pct(wp['gross'])}（扣成本後 {_pct(wp['net'])}）、挑月份是每年 {_pct(mp['gross'])}"
             f"（扣成本後 {_pct(mp['net'])}），扣成本後贏過一直抱著的年數分別是 {nt} 年裡 {sp['beat_hold_net']}、{wp['beat_hold_net']}、"
             f"{mp['beat_hold_net']} 年。")
    verdict = part1 + part2 + part3

    # ----- notes -----
    def pick_list(kind: str, labels) -> str:
        p = kinds[kind]["pick"]
        return "、".join(f"{y} {labels[picks[y][kind]]}（當年在 {kinds[kind]['count']} 個裡排第 {p['rank'][y]}）"
                         for y in test_years)

    notes = [
        f"一次試了很多東西：24 個小時、3 個時段、7 個星期幾、週末、月底月初、12 個月份。試得越多，「最好的那一個」光靠運氣就會很好看，"
        "所以不能只看排第一的那個有多突出，要看第四張表的運氣檢查，以及它在前後兩半、每一年是不是都同方向。",
        f"月份的資料非常少：每個月份一年只出現一次，{r['m_years'][0]} 年以來每個月份只有大約 {r['months'][0]['n']} 個觀察值，"
        "「幾年裡幾年上漲」多一年少一年就會差很多；而且比特幣早年漲得異常多，早年的月份報酬會把中位數和平均都拉高。",
        "交易所和時區的定義會影響結果：這裡的小時、時段、星期幾、月底月初用的是幣安現貨 BTCUSDT，一天從 UTC 0 點（台灣時間早上 8 點）"
        "算到隔天同一時間，所以「星期一」其實是台灣星期一早上 8 點到星期二早上 8 點；換一家交易所、換一種切日方式，"
        "或把三個時段的界線前後挪一兩個小時，數字都會不同。三個時段是照 UTC 0–8、8–16、16–24 點硬切的，不等於各地股市的開盤時間。",
        f"成本遠比 {COST * 100:.2f}% 高：{COST * 100:.2f}% 是一個偏樂觀的粗略數字。一般人每次進出要付的手續費加買賣價差多半更高，"
        f"而只抱一個時段每年要進出大約 {kinds['session']['pick']['switches']:,.0f} 次，成本乘上這個次數就是每年被扣掉的部分；"
        "表裡「不扣成本」的數字實際上拿不到。持有現金假設沒有利息，也沒有算稅。",
        f"事先挑的每一年挑到什麼——時段：{pick_list('session', SESSION_SHORT)}。星期幾：{pick_list('weekday', WEEKDAYS)}。"
        f"月份：{pick_list('month', MONTH_NAMES)}。時段和星期幾只能用幣安 2017-08 以後的資料來挑，"
        f"所以 {FIRST_PICK_YEAR} 年那次只有兩年多的資料；月份用 {r['m_years'][0]} 年以後的月報酬中位數來挑。"
        "挑到的月份如果當年還沒到，那一年就是整年抱現金。",
        f"小時的漲跌用的是每根 1 小時線自己的「收盤 ÷ 開盤」，幣安早年停機維護造成的缺口（整段共缺 {r['missing_hours']} 個小時）"
        f"不算進任何一個小時，所以這裡的「一直抱著」（整段每年 {_pct(r['hold_all'])}）和用日線算的會有一點出入。"
        "只用已經收完的小時和已經過完的 UTC 日。",
        "月份用的是長期日線：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均，標示的日期和幣安大約差一天），之後是幣安現貨日線。"
        "差一天對整個月的報酬影響很小，但對星期幾和月底月初影響很大，所以那兩項沒有往 2017 年以前延伸，只用幣安的資料。",
        "運氣檢查的做法有它的限制：把一天裡的小時互相對調，等於假設每個小時的起伏一樣大，實際上美洲時段的起伏比較大，"
        f"所以這個檢查是約略的；只做了 {SIMS} 次，「{SIMS} 次裡幾次」本身也有幾次的誤差。平均值很容易被少數幾個暴漲暴跌的小時或日子拉動。",
        "已發表的研究：Baur、Cahill、Godfrey、Liu（Finance Research Letters，2019）用 2011–2017 年多家交易所的資料，結論是比特幣的"
        "時段、星期幾、月份報酬型態並不穩定、不會一直持續；Quantpedia（2022–2024）則報告過比特幣在一天裡特定幾個小時報酬較高的型態。"
        "這裡是用幣安的資料自己重算，定義和他們不完全相同，數字不能直接對照。",
        "過去的結果不保證未來會一樣。這裡只是整理這些日曆型態過去的樣子，不是買賣建議。",
    ]
    if problem:
        notes.append(problem)

    return {
        "id": "seasonality",
        "title": "幾點、星期幾、哪個月比較會漲？日曆型態的檢驗",
        "question": ("常聽到比特幣在某些時段、某幾天、月底月初、10 月或週末表現比較好。用全部的資料檢查，"
                     "再看型態被看見「之後」的那些年，這些說法有哪些是真的？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 BTCUSDT 1 小時線（2017-08 起）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2010 年起，月份統計用）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Baur, Cahill, Godfrey, Liu (2019)〈Bitcoin time-of-day, day-of-week and month-of-year effects in "
                     "returns and trading volume〉Finance Research Letters",
             "url": "https://doi.org/10.1016/j.frl.2019.04.023"},
            {"name": "Quantpedia〈Are There Seasonal Intraday or Overnight Anomalies in Bitcoin?〉",
             "url": "https://quantpedia.com/are-there-seasonal-intraday-or-overnight-anomalies-in-bitcoin/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
