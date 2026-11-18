"""Historical study "vol_breakout": Larry Williams' volatility breakout as Korean retail traders and many GitHub bots
run it on bitcoin - "buy when the price rises above today's open plus half of yesterday's range, sell at the next
day's open" - does it beat the 50-day average after costs, and does the breakout timing itself matter?

Rule (fixed before running, k = 0.5 as published). Each day starts at a chosen hour (the day boundary).
range = yesterday's high - low; target = today's open + k x range; if the price trades above the target during the
day, buy; sell at the next day's open (= this day's last hourly close). Two fills: "at target" (the published
backtests: a stop order at the target; the hour's open if that is already higher) and the conservative "first hourly
close above target" (if that is only the day's last hour, no trade).
Variants (all stated now, no tuning):
  base   k = 0.5
  ma5    k = 0.5, only on days whose open is above the 5-day average of the previous closes ("이동평균 필터")
  noise  k = average over the previous 20 days of 1 - |open - close| / (high - low) (noise-adaptive k)
  size   k = 0.5, position = 2% / (yesterday's range / today's open), at most 1
  all3   noise k + ma5 filter + size
  base_ma50 / all3_ma50   the same, but only on days on which the 50-day average rule is on
                          (yesterday's close above the average of the previous 50 closes)
The "best" variant is the one with the highest return / volatility on bitcoin, day boundary 00:00 UTC (= 09:00 KST,
Upbit's daily candle), fill at target, 0.10% per side.

Pre-registered tests:
1. On BTC hourly Binance data from 2017-08: each variant with fills at target and at the first hourly close above
   target; costs 0.10% and 0.30% per side; the break-even cost (per side) at which the yearly return falls to zero
   and at which return / volatility falls to the 50-day average's (at 0.10%).
2. Robustness: all 24 day-boundary hours; monthly start dates 2018-01 ... 2023-12 (as main_signal_check) x 24 hours;
   other coins ETH/SOL/BNB/XRP (00:00 UTC); periods 2017-08..2019, 2020..2022, 2023..now (00:00 UTC).
3. Compared with holding and the 50-day average (ma50, 0.10% per change) on return / volatility, yearly return and
   deepest loss; breakout only on days ma50 is on as variants.
4. Placebo: on the same days, with the same position size, the entry is moved to a random hour (drawn from the
   strategy's own entry hours), exit unchanged; 200 draws (fixed seed). A second placebo (reported, not part of the
   pass test) picks random days among the days the variant's filters allow, as many as it traded, because entering
   earlier on a day already known to break out later borrows information from that day.
Pass test (stated now): at 0.10% per side, the best pre-stated variant beats ma50 on return / volatility in >= 80% of
the 24 hours and in all three periods, beats >= 90% of placebo draws, and holds on >= 3 of 4 other coins.

No look-ahead: the target uses only the previous day and today's open, the filters only earlier closes (and today's
open); a fill is never before the hour in which the price reached the target; the exit is the day's last close.
Data: Binance spot 1-hour candles (open/high/low/close) kept in this study's own stores
cache/_store/vol_breakout_hourly_<SYMBOL>.csv (first run fetches the whole history, later runs only new hours).
Only closed candles, only public key-less endpoints through the dashboard's own fetch helpers.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
MAX_PAGES = 140
MIN_HOURS = 40_000
YEAR = 365
K = 0.5
NOISE_DAYS = 20
MA_SHORT, MA_LONG = 5, 50
VOL_TARGET = 0.02
COST, COST_HIGH = 0.001, 0.003
DRAWS = 200
SEED = 20170817
BE_MAX = 0.02
WARM = MA_LONG + 1  # first measured day: every feature has a value
STARTS = ("2018-01-01", "2023-12-01")
PERIODS = (("2017-08 到 2019", None, "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"),
           ("2023 到最新", "2023-01-01", None))
HOURS_BAR, PLACEBO_BAR, COINS_BAR = 0.80, 0.90, 3
COINS = ("ETH", "SOL", "BNB", "XRP")

VARIANTS = {
    "base": {"k": "fix", "ma5": False, "size": False, "ma50": False},
    "ma5": {"k": "fix", "ma5": True, "size": False, "ma50": False},
    "noise": {"k": "noise", "ma5": False, "size": False, "ma50": False},
    "size": {"k": "fix", "ma5": False, "size": True, "ma50": False},
    "all3": {"k": "noise", "ma5": True, "size": True, "ma50": False},
    "base_ma50": {"k": "fix", "ma5": False, "size": False, "ma50": True},
    "all3_ma50": {"k": "noise", "ma5": True, "size": True, "ma50": True},
}
NAMES = {
    "base": "基本版（k=0.5）",
    "ma5": "＋開盤高於 5 日均線才做",
    "noise": "k 改用 20 天平均雜訊比",
    "size": "＋部位＝2%÷前一天振幅%",
    "all3": "三個附加條件全加",
    "base_ma50": "基本版＋50 日均線多頭才做",
    "all3_ma50": "三個全加＋50 日均線多頭才做",
}
FILLS = {"tgt": "目標價成交", "close": "第一個收盤站上才買"}

_pct, _pp, _num, _d = cr._pct, cr._pp, cr._num, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(x) -> str:
    return "—" if not _ok(x) else f"{float(x) * 100:.0f}%"


def _cost_text(c) -> str:
    if not _ok(c):
        return "—"
    return f"{float(c) * 100:.3f}".rstrip("0").rstrip(".") + "%"


# ---------- data ----------

def _store(symbol: str):
    return common.STORE / f"vol_breakout_hourly_{symbol}.csv"


def _read(symbol: str) -> dict[int, tuple]:
    rows: dict[int, tuple] = {}
    try:
        for line in _store(symbol).read_text(encoding="utf-8").splitlines():
            p = line.split(",")
            if len(p) != 5 or not p[0].isdigit():
                continue
            v = tuple(float(x) for x in p[1:])
            if all(x > 0 and math.isfinite(x) for x in v):
                rows[int(p[0])] = v
    except Exception:  # noqa: BLE001 - no file or a damaged one: fetch again
        return {}
    return rows


def _write(symbol: str, rows: dict[int, tuple]) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        text = "ms,open,high,low,close\n" + "\n".join(
            f"{ms},{o!r},{h!r},{lo!r},{c!r}" for ms, (o, h, lo, c) in sorted(rows.items())) + "\n"
        _store(symbol).write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run fetches again
        pass


def _fetch(symbol: str, rows: dict[int, tuple]) -> int:
    from dashboard.collectors import price as px

    start = max(rows) + 1 if rows else 0
    added = 0
    try:
        for page_no in range(MAX_PAGES):
            page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 1000,
                                                  "startTime": start})
            if not page:
                break
            now_ms = int(time.time() * 1000)
            for r in page:
                v = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                if int(r[6]) < now_ms and min(v) > 0:  # closed candles only
                    rows[int(r[0])] = v
                    added += 1
            if len(page) < 1000:
                break
            start = int(page[-1][0]) + 1
            if page_no % 20 == 19:
                _write(symbol, rows)
    finally:
        if added:
            _write(symbol, rows)
    return added


def _hourly(symbol: str) -> tuple[dict[int, tuple], str | None]:
    rows = _read(symbol)
    problem = None
    now_ms = int(time.time() * 1000)
    if not rows or max(rows) < now_ms - 2 * HOUR_MS:
        try:
            _fetch(symbol, rows)
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = (f"{symbol[:-4]} 的幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    return rows, problem


class _Grid:
    """Hourly open/high/low/close from the first UTC midnight on; index i = candle opening at b0 + i hours.
    A missing hour is a flat candle at the last close."""

    def __init__(self, rows: dict[int, tuple]):
        first = min(rows)
        self.b0 = -(-first // DAY_MS) * DAY_MS
        n = (max(rows) - self.b0) // HOUR_MS + 1
        a = np.full((n, 4), np.nan)
        for ms, v in rows.items():
            i = (ms - self.b0) // HOUR_MS
            if 0 <= i < n:
                a[i] = v
        self.missing = int(np.isnan(a[:, 3]).sum())
        c = pd.Series(a[:, 3]).ffill().bfill().values
        for j in range(3):
            a[:, j] = np.where(np.isnan(a[:, j]), c, a[:, j])
        a[:, 3] = c
        self.o, self.h, self.l, self.c = a.T
        self.n = n
        self.start = pd.Timestamp(self.b0, unit="ms")

    def days(self, hb: int) -> dict:
        """Daily bars whose day starts at hour hb (UTC); only finished days."""
        nd = (self.n - hb) // 24
        sl = slice(hb, hb + 24 * nd)
        O, H, L, C = (x[sl].reshape(nd, 24) for x in (self.o, self.h, self.l, self.c))
        dates = self.start + pd.Timedelta(hours=hb) + pd.to_timedelta(np.arange(nd), unit="D")
        return {"O": O, "H": H, "L": L, "C": C, "op": O[:, 0], "hi": H.max(1), "lo": L.min(1), "cl": C[:, -1],
                "dates": dates, "n": nd}


# ---------- the rule ----------

def _features(d: dict) -> dict:
    op, hi, lo, cl = d["op"], d["hi"], d["lo"], d["cl"]
    rng = hi - lo
    prev_rng = np.concatenate([[np.nan], rng[:-1]])
    span = np.where(rng > 0, rng, np.nan)
    noise = 1 - np.abs(op - cl) / span
    knoise = pd.Series(noise).rolling(NOISE_DAYS, min_periods=NOISE_DAYS // 2).mean().shift(1).values
    c = pd.Series(cl)
    ma5 = c.rolling(MA_SHORT, min_periods=MA_SHORT).mean().shift(1).values
    ma50 = c.rolling(MA_LONG, min_periods=MA_LONG).mean().shift(1).values
    prev_cl = c.shift(1).values
    size = np.minimum(1.0, VOL_TARGET / np.where(prev_rng > 0, prev_rng / op, np.nan))
    return {"prev_rng": prev_rng, "knoise": knoise, "ma5_ok": op > ma5, "ma50_on": prev_cl > ma50,
            "size": np.nan_to_num(size, nan=1.0), "day_ret": cl / op - 1}


def _rule(d: dict, f: dict, var: str, fill: str) -> dict:
    """Which days trade, the entry hour (the hour from whose start the position is held, 0..23; the fill happens
    inside hour 'cross' for fill at target), entry price, size. Exit at the day's last close."""
    v = VARIANTS[var]
    k = np.full(d["n"], K) if v["k"] == "fix" else f["knoise"]
    target = d["op"] + k * f["prev_rng"]
    allowed = np.isfinite(target).copy()
    if v["ma5"]:
        allowed &= f["ma5_ok"]
    if v["ma50"]:
        allowed &= f["ma50_on"]
    allowed[:WARM] = False
    t = np.where(np.isfinite(target), target, np.inf)[:, None]
    rows = np.arange(d["n"])
    if fill == "tgt":
        hit = d["H"] >= t
        any_hit = hit.any(1)
        j = hit.argmax(1)
        entry = np.maximum(target, d["O"][rows, j])
        start_hour = j
    else:
        hit = d["C"][:, :23] > t
        any_hit = hit.any(1)
        j = hit.argmax(1)
        entry = d["C"][rows, j]
        start_hour = j + 1
    trade = allowed & any_hit
    size = f["size"] if v["size"] else np.ones(d["n"])
    return {"trade": trade, "allowed": allowed, "hour": start_hour, "cross": j, "entry": entry, "size": size,
            "target": target}


def _net(d: dict, r: dict, cost: float) -> np.ndarray:
    x = d["cl"] / np.where(r["trade"], r["entry"], 1.0)
    s = r["size"]
    return np.where(r["trade"], s * (x - 1) - cost * s * (1 + x), 0.0)


def _ma50_net(d: dict, f: dict, cost: float) -> np.ndarray:
    w = np.nan_to_num(f["ma50_on"].astype(float))
    w[:WARM] = 0.0
    prev = np.concatenate([[0.0], w[:-1]])
    return w * f["day_ret"] - cost * np.abs(w - prev)


def _stats(net: np.ndarray) -> dict:
    return cr._stats(np.asarray(net, dtype=float))


def _suffix_ratio(net: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Return / volatility from each start to the end (shape (starts,))."""
    x = np.asarray(net, dtype=float)
    lg = np.log1p(x)
    rev = lambda a: np.cumsum(a[::-1])[::-1]  # noqa: E731
    s_lg, s1, s2 = rev(lg), rev(x), rev(x * x)
    n = (len(x) - starts).astype(float)
    ret = (np.exp(s_lg[starts] * YEAR / n) - 1) * 100
    var = (s2[starts] - s1[starts] ** 2 / n) / (n - 1)
    vol = np.sqrt(np.maximum(var, 0)) * math.sqrt(YEAR) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1), 0.0)


def _bisect(fn, lo: float = 0.0, hi: float = BE_MAX):
    """Smallest cost in [lo, hi] at which fn(cost) <= 0 (fn falls with cost); None if fn(lo) <= 0 or fn(hi) > 0."""
    if fn(lo) <= 0:
        return 0.0
    if fn(hi) > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if fn(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- the whole computation ----------

def _coin_run(g: _Grid, hb: int) -> dict:
    d = g.days(hb)
    f = _features(d)
    w = slice(WARM, None)
    out = {"d": d, "f": f, "hold": _stats(f["day_ret"][w]), "ma50": _stats(_ma50_net(d, f, COST)[w]),
           "ma50_net": _ma50_net(d, f, COST), "v": {}}
    for var in VARIANTS:
        for fill in FILLS:
            r = _rule(d, f, var, fill)
            n1, n3 = _net(d, r, COST), _net(d, r, COST_HIGH)
            s1, s3 = _stats(n1[w]), _stats(n3[w])
            years = (d["n"] - WARM) / YEAR
            out["v"][(var, fill)] = {"r": r, "net": n1, "s1": s1, "s3": s3,
                                     "tpy": float(r["trade"][w].sum() / years)}
    return out


def _placebo(run: dict, var: str, fill: str) -> dict:
    d, r = run["d"], run["v"][(var, fill)]["r"]
    w = slice(WARM, None)
    actual = run["v"][(var, fill)]["s1"]["ratio"]
    days = np.where(r["trade"])[0]
    days = days[days >= WARM]
    hours = r["hour"][days]
    rng = np.random.default_rng(SEED)
    same, rand = [], []
    allowed = np.where(r["allowed"])[0]
    for _ in range(DRAWS):
        for kind, chosen in (("same", days), ("rand", np.sort(rng.choice(allowed, size=len(days), replace=False)))):
            hh = rng.choice(hours, size=len(chosen), replace=True)
            entry = np.where(hh < 24, d["O"][chosen, np.minimum(hh, 23)], d["cl"][chosen])
            x = d["cl"][chosen] / entry
            s = r["size"][chosen]
            net = np.zeros(d["n"])
            net[chosen] = s * (x - 1) - COST * s * (1 + x)
            (same if kind == "same" else rand).append(_stats(net[w])["ratio"])
    same, rand = np.array(same), np.array(rand)
    return {"actual": actual, "same": same, "rand": rand, "beat_same": float((actual > same).mean()),
            "beat_rand": float((actual > rand).mean())}


def _periods(run: dict, net: np.ndarray) -> list[dict]:
    dates = run["d"]["dates"]
    out = []
    for name, a, b in PERIODS:
        m = np.arange(len(dates)) >= WARM
        if a:
            m &= dates >= pd.Timestamp(a)
        if b:
            m &= dates <= pd.Timestamp(b) + pd.Timedelta(hours=23)
        out.append({"name": name, "s": _stats(net[m]), "hold": _stats(run["f"]["day_ret"][m]),
                    "ma50": _stats(run["ma50_net"][m]), "from": _d(dates[m][0]), "to": _d(dates[m][-1])})
    return out


def _month_starts(dates: pd.DatetimeIndex) -> np.ndarray:
    pos = []
    for month in pd.date_range(STARTS[0], STARTS[1], freq="MS"):
        i = int(dates.searchsorted(month))
        if WARM <= i < len(dates) - 30 and (not pos or i > pos[-1]):
            pos.append(i)
    return np.array(pos, dtype=int)


def _break_even(run: dict, var: str, fill: str) -> dict:
    d, f = run["d"], run["f"]
    r = run["v"][(var, fill)]["r"]
    w = slice(WARM, None)
    bar = run["ma50"]["ratio"]
    zero = _bisect(lambda c: _stats(_net(d, r, c)[w])["ret"])
    vs = _bisect(lambda c: _stats(_net(d, r, c)[w])["ratio"] - bar)
    return {"zero": zero, "ma50": vs}


def _check(runs24: list[dict], coins: dict, var: str, fill: str) -> dict:
    hours_beat = [runs24[h]["v"][(var, fill)]["s1"]["ratio"] > runs24[h]["ma50"]["ratio"] for h in range(24)]
    per = _periods(runs24[0], runs24[0]["v"][(var, fill)]["net"])
    per_beat = [p["s"]["ratio"] > p["ma50"]["ratio"] for p in per]
    coin_beat = {c: coins[c]["v"][(var, fill)]["s1"]["ratio"] > coins[c]["ma50"]["ratio"] for c in coins}
    starts = _month_starts(runs24[0]["d"]["dates"])
    cells = beat = 0
    for h in range(24):
        st = _month_starts(runs24[h]["d"]["dates"])
        a = _suffix_ratio(runs24[h]["v"][(var, fill)]["net"], st)
        b = _suffix_ratio(runs24[h]["ma50_net"], st)
        cells += len(st)
        beat += int((a > b).sum())
    return {"hours": int(sum(hours_beat)), "hours_list": hours_beat, "periods": per, "per_beat": int(sum(per_beat)),
            "coins": coin_beat, "coin_beat": int(sum(coin_beat.values())), "cells": cells, "cells_beat": beat,
            "n_starts": len(starts),
            "ratios24": np.array([runs24[h]["v"][(var, fill)]["s1"]["ratio"] for h in range(24)]),
            "ma50_24": np.array([runs24[h]["ma50"]["ratio"] for h in range(24)])}


def _compute() -> dict:
    problems = []
    rows, problem = _hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    if len(rows) < MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    g = _Grid(rows)
    runs24 = [_coin_run(g, h) for h in range(24)]
    coins = {}
    for c in COINS:
        try:
            cr_rows, p = _hourly(f"{c}USDT")
            if p:
                problems.append(p)
            if len(cr_rows) < 24 * 400:
                raise RuntimeError("資料太短")
            coins[c] = _coin_run(_Grid(cr_rows), 0)
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{c} 這次算不出來（{type(exc).__name__}），沒有算進去。")
    main = runs24[0]
    best = max(VARIANTS, key=lambda v: main["v"][(v, "tgt")]["s1"]["ratio"])
    checks = {(v, fl): _check(runs24, coins, v, fl) for v in VARIANTS for fl in FILLS}
    placebo = {(v, fl): _placebo(main, v, fl) for v in VARIANTS for fl in FILLS}
    be = {(v, fl): _break_even(main, v, fl) for v in VARIANTS for fl in FILLS}
    return {"g": g, "rows": rows, "runs24": runs24, "coins": coins, "best": best, "checks": checks,
            "placebo": placebo, "be": be, "problems": problems}


def _passed(c: dict, p: dict) -> dict:
    out = {"hours": c["hours"] / 24 >= HOURS_BAR, "periods": c["per_beat"] == len(PERIODS),
           "placebo": p["beat_same"] >= PLACEBO_BAR, "coins": c["coin_beat"] >= COINS_BAR}
    out["all"] = all(out.values())
    return out


# ---------- today ----------

def _now(r: dict) -> list[dict]:
    rows = r["rows"]
    last = max(rows)
    today = (last // DAY_MS) * DAY_MS
    d0 = r["runs24"][0]["d"]
    f0 = r["runs24"][0]["f"]
    out = []
    today_rows = [rows[ms] for ms in sorted(rows) if ms >= today]
    yday = pd.Timestamp(today - DAY_MS, unit="ms")
    if d0["dates"][-1] != yday or not today_rows:
        out.append({"label": "今天的目標價", "value": "今天 UTC 0 點以後的 1 小時線還沒有拿到，這次不列"})
        return out
    op = today_rows[0][0]
    hi_y, lo_y, cl_y = float(d0["hi"][-1]), float(d0["lo"][-1]), float(d0["cl"][-1])
    rng = hi_y - lo_y
    noise = 1 - np.abs(d0["op"] - d0["cl"]) / np.where(d0["hi"] > d0["lo"], d0["hi"] - d0["lo"], np.nan)
    kn = float(np.nanmean(noise[-NOISE_DAYS:]))
    ma5 = float(d0["cl"][-MA_SHORT:].mean())
    ma50 = float(d0["cl"][-MA_LONG:].mean())
    size = min(1.0, VOL_TARGET / (rng / op)) if rng > 0 else 1.0
    high_now = max(x[1] for x in today_rows)
    t05, tn = op + K * rng, op + kn * rng
    day = _d(pd.Timestamp(today, unit="ms"))
    asof = pd.Timestamp(last + HOUR_MS, unit="ms").strftime("%Y-%m-%d %H:%M")
    out += [
        {"label": f"比特幣 {day} 的開盤（UTC 0 點＝韓國早上 9 點）", "value": f"{op:,.0f} 美元"},
        {"label": f"前一天（{_d(yday)}）的振幅", "value": f"最高 {hi_y:,.0f} − 最低 {lo_y:,.0f} ＝ {rng:,.0f} 美元（開盤的 {rng / op * 100:.1f}%）"},
        {"label": "k＝0.5 的目標價", "value": f"{t05:,.0f} 美元（開盤 ＋ 0.5 × 振幅，比開盤高 {(t05 / op - 1) * 100:.1f}%）"},
        {"label": "k 用 20 天平均雜訊比時", "value": f"k ≈ {kn:.2f}，目標價 {tn:,.0f} 美元"},
        {"label": "5 日均線條件", "value": f"前 5 天收盤平均 {ma5:,.0f}，開盤{'高於' if op > ma5 else '不高於'}它（{'會' if op > ma5 else '不會'}做）"},
        {"label": "50 日均線條件", "value": f"前一天收盤 {cl_y:,.0f}、前 50 天收盤平均 {ma50:,.0f}，{'多頭（會做）' if cl_y > ma50 else '不是多頭（不做）'}"},
        {"label": "部位大小版本", "value": f"2% ÷ {rng / op * 100:.1f}% ＝ 投入 {size * 100:.0f}%"},
        {"label": f"到 {asof} UTC 為止今天的最高價", "value": f"{high_now:,.0f} 美元（{'已經' if high_now >= t05 else '還沒'}碰到 k＝0.5 的目標價）"},
        {"label": "這幾格的意思", "value": "只描述這條規則今天的價位怎麼算，不是買賣建議"},
    ]
    return out


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    runs24, coins, best = r["runs24"], r["coins"], r["best"]
    main = runs24[0]
    d0 = main["d"]
    first, last = _d(d0["dates"][WARM]), _d(d0["dates"][-1])
    hold, ma50 = main["hold"], main["ma50"]

    # table 1: BTC 00:00 UTC
    rows1 = []
    for var in VARIANTS:
        for fill in FILLS:
            x = main["v"][(var, fill)]
            rows1.append([NAMES[var], FILLS[fill], f"{x['tpy']:.0f} 天", _num(x["s1"]["ratio"]), _pct(x["s1"]["ret"]),
                          _pct(x["s1"]["deepest"]), _num(x["s3"]["ratio"]), _pct(x["s3"]["ret"])])
    rows1.append(["對照：50 日均線", "收盤決定、隔天照做", "—", _num(ma50["ratio"]), _pct(ma50["ret"]),
                  _pct(ma50["deepest"]), "—", "—"])
    rows1.append(["對照：一直抱著", "—", "—", _num(hold["ratio"]), _pct(hold["ret"]), _pct(hold["deepest"]), "—", "—"])

    # table 2: the pass test for the best variant, both fills
    cb, cc = r["checks"][(best, "tgt")], r["checks"][(best, "close")]
    pb, pc = r["placebo"][(best, "tgt")], r["placebo"][(best, "close")]
    ok_b, ok_c = _passed(cb, pb), _passed(cc, pc)
    yn = lambda ok: "及格" if ok else "不及格"  # noqa: E731
    rows2 = [
        ["24 個開盤時間：報酬÷波動贏 50 日均線", f"≥ {math.ceil(HOURS_BAR * 24 - 1e-9)} 個（80%）", f"{cb['hours']} 個", f"{cc['hours']} 個",
         f"突破 {_num(cb['ratios24'].min())}～{_num(cb['ratios24'].max())}；均線 {_num(cb['ma50_24'].min())}～{_num(cb['ma50_24'].max())}",
         f"{yn(ok_b['hours'])}／{yn(ok_c['hours'])}"],
    ]
    for i, p in enumerate(cb["periods"]):
        q = cc["periods"][i]
        rows2.append([f"時段 {p['name']}（{p['from'][:7]}～{p['to'][:7]}）：報酬÷波動", "贏 50 日均線",
                      _num(p["s"]["ratio"]), _num(q["s"]["ratio"]),
                      f"均線 {_num(p['ma50']['ratio'])}；抱著 {_num(p['hold']['ratio'])}",
                      f"{'贏' if p['s']['ratio'] > p['ma50']['ratio'] else '輸'}／{'贏' if q['s']['ratio'] > q['ma50']['ratio'] else '輸'}"])
    rows2.append(["同一批交易日、進場改成隨機時間（200 次）：贏過的比例", f"≥ {PLACEBO_BAR * 100:.0f}%",
                  _share(pb["beat_same"]), _share(pc["beat_same"]),
                  f"隨機版中位數 {_num(float(np.median(pb['same'])))}（目標價成交版本）",
                  f"{yn(ok_b['placebo'])}／{yn(ok_c['placebo'])}"])
    rows2.append(["參考：隨機挑同樣多天、隨機時間進場（200 次）：贏過的比例", "不列入及格", _share(pb["beat_rand"]),
                  _share(pc["beat_rand"]), f"隨機版中位數 {_num(float(np.median(pb['rand'])))}", "—"])
    rows2.append(["其他幣 ETH／SOL／BNB／XRP：報酬÷波動贏 50 日均線", f"≥ {COINS_BAR} 個",
                  f"{cb['coin_beat']} 個（{'、'.join(c for c, v in cb['coins'].items() if v) or '沒有'}）",
                  f"{cc['coin_beat']} 個（{'、'.join(c for c, v in cc['coins'].items() if v) or '沒有'}）",
                  f"共算了 {len(coins)} 個幣", f"{yn(ok_b['coins'])}／{yn(ok_c['coins'])}"])
    rows2.append([f"參考：起算日（{STARTS[0][:7]}～{STARTS[1][:7]} 每月）× 24 個開盤時間：報酬÷波動贏 50 日均線", "不列入及格",
                  f"{cb['cells']:,} 格裡 {cb['cells_beat']:,}（{_share(cb['cells_beat'] / max(cb['cells'], 1))}）",
                  f"{cc['cells']:,} 格裡 {cc['cells_beat']:,}（{_share(cc['cells_beat'] / max(cc['cells'], 1))}）", "—", "—"])
    beb, bec = r["be"][(best, "tgt")], r["be"][(best, "close")]
    rows2.append(["損益兩平的手續費（每邊）：每年報酬降到 0｜報酬÷波動降到 50 日均線", "—",
                  f"{_cost_text(beb['zero'])}｜{_cost_text(beb['ma50'])}", f"{_cost_text(bec['zero'])}｜{_cost_text(bec['ma50'])}",
                  "0 = 不扣成本也比不過", "—"])
    rows2.append(["整體", "四項都及格", yn(ok_b["all"]), yn(ok_c["all"]), "—", "—"])

    # table 3: every variant through the same checks (fill at target, 0.10%)
    rows3 = []
    for var in VARIANTS:
        c, p, b = r["checks"][(var, "tgt")], r["placebo"][(var, "tgt")], r["be"][(var, "tgt")]
        c2 = r["checks"][(var, "close")]
        rows3.append([NAMES[var] + ("（事先定的「最好」）" if var == best else ""), f"{c['hours']}／24（{c2['hours']}／24）",
                      f"{c['per_beat']}／3（{c2['per_beat']}／3）", f"{c['coin_beat']}／{len(coins)}（{c2['coin_beat']}／{len(coins)}）",
                      f"{_share(p['beat_same'])}｜{_share(p['beat_rand'])}", f"{_cost_text(b['zero'])}｜{_cost_text(b['ma50'])}",
                      yn(_passed(c, p)["all"])])

    # table 4: coins
    rows4 = []
    for name, run in [("BTC", main)] + list(coins.items()):
        xb, xc = run["v"][(best, "tgt")], run["v"][(best, "close")]
        rows4.append([name, f"{_d(run['d']['dates'][WARM])[:7]} 起", _num(run["hold"]["ratio"]), _num(run["ma50"]["ratio"]),
                      _num(xb["s1"]["ratio"]), _num(xc["s1"]["ratio"]), _num(xb["s3"]["ratio"]),
                      f"{_pct(xb['s1']['deepest'])}｜{_pct(run['ma50']['deepest'])}"])

    tables = [
        {"caption": (f"比特幣，一天從 UTC 0 點（韓國早上 9 點）算起，{first} 到 {last}。目標價＝當天開盤＋k×前一天振幅（振幅＝一天裡最高價減最低價），"
                     "漲破就買、隔天開盤賣。「目標價成交」是網路上常見回測的算法（剛好在目標價買到）；「第一個收盤站上才買」是保守算法"
                     "（等某個整點收盤高於目標價才用那個收盤價買）。手續費每邊（買一次、賣一次各扣一次）。報酬÷波動越高越好；50 日均線每次換手扣 0.10%。"),
         "columns": ["做法", "成交方式", "一年交易幾天", "報酬÷波動（0.10%）", "每年報酬（0.10%）", "最深虧損（0.10%）",
                     "報酬÷波動（0.30%）", "每年報酬（0.30%）"],
         "rows": rows1},
        {"caption": (f"事先定的及格檢查，用事先定的「最好」做法：{NAMES[best]}（UTC 0 點、目標價成交、0.10% 時報酬÷波動最高的那個）。"
                     "每格兩個版本：目標價成交／第一個收盤站上才買，手續費每邊 0.10%。「隨機時間」是同樣的交易日、同樣的部位，只把進場時間換成隨機"
                     "（從這個做法自己的進場時間裡抽），出場不變；這個比法會讓隨機版偷看到「今天後來會漲破」，對隨機版有利，所以另外列了隨機挑日子的版本當參考。"),
         "columns": ["檢查", "及格線", "目標價成交", "第一個收盤站上才買", "對照數字", "結果（兩個版本）"],
         "rows": rows2},
        {"caption": ("七個事先定的做法都走一遍同樣的檢查（手續費每邊 0.10%，括號裡是「第一個收盤站上才買」的結果）。"
                     "隨機時間一欄是「贏過同一批日子隨機時間進場的比例｜贏過隨機挑日子的比例」；兩平手續費是「每年報酬降到 0｜報酬÷波動降到 50 日均線」的每邊手續費。"
                     "最後一欄只看目標價成交的四項及格檢查。"),
         "columns": ["做法", "24 個開盤時間贏均線", "三個時段贏均線", "其他幣贏均線", "贏過隨機時間的比例", "兩平手續費（每邊）", "及格"],
         "rows": rows3},
        {"caption": (f"換幣：{NAMES[best]}，一天從 UTC 0 點算起，幣安現貨 1 小時線拼成日線，從每個幣有足夠資料的第一天算到 {last}。"
                     "數字是報酬÷波動；最後一欄是最深虧損（突破做法目標價成交 0.10%｜50 日均線）。"),
         "columns": ["幣", "期間", "一直抱著", "50 日均線", "突破：目標價成交 0.10%", "突破：收盤站上 0.10%", "突破：目標價成交 0.30%",
                     "最深虧損：突破｜均線"],
         "rows": rows4},
    ]

    # verdict
    xb = main["v"][(best, "tgt")]
    xc = main["v"][(best, "close")]
    base = main["v"][("base", "tgt")]
    s1 = (f"「漲破開盤價加前一天振幅一半就買、隔天開盤賣」在比特幣（{first[:7]} 起、韓國早上 9 點開盤）一年約交易 {base['tpy']:.0f} 天，"
          f"基本版每邊扣 0.10% 時報酬÷波動 {_num(base['s1']['ratio'])}、扣 0.30% 時 {_num(base['s3']['ratio'])}；"
          f"事先定的最好做法「{NAMES[best]}」是 {_num(xb['s1']['ratio'])}（保守成交 {_num(xc['s1']['ratio'])}），"
          f"50 日均線 {_num(ma50['ratio'])}、一直抱著 {_num(hold['ratio'])}，每邊手續費超過約 {_cost_text(beb['ma50'])} 就不如均線。")
    s2 = (f"換開盤時間 24 個裡 {cb['hours']} 個贏均線、三個時段贏 {cb['per_beat']} 個、四個其他幣贏 {cb['coin_beat']} 個"
          f"（保守成交：{cc['hours']}、{cc['per_beat']}、{cc['coin_beat']}）。")
    s3 = (f"突破的時間點本身：同一批日子改隨機時間進場，突破版只贏 {_share(pb['beat_same'])} 的抽樣（這個比法偏袒隨機版），"
          f"改成隨機挑日子則贏 {_share(pb['beat_rand'])}；事先定的及格檢查{'通過' if ok_b['all'] else '沒有通過'}"
          f"{'' if ok_b['all'] == ok_c['all'] else '（保守成交' + ('通過' if ok_c['all'] else '沒有通過') + '）'}。")
    verdict = s1 + s2 + s3

    notes = [
        "事先寫好的檢查（照抄）：1. 比特幣幣安 1 小時線（2017-08 起），每個做法都用「目標價成交」和「第一個整點收盤高於目標價才買」兩種算，"
        "手續費每邊 0.10% 和 0.30%，並算出兩平的手續費。2. 穩不穩：24 個開盤時間（韓國用的早上 9 點＝UTC 0 點）、每月起算日（同 main_signal_check）、"
        "其他幣 ETH／SOL／BNB／XRP、三個時段 2017-08～2019、2020～2022、2023～最新。3. 和一直抱著、50 日均線比報酬÷波動、每年報酬、最深虧損；"
        "「只在 50 日均線多頭的日子做突破」當作一個做法。4. 對照：同一批日子、同樣部位，把突破進場換成隨機時間，抽 200 次。"
        f"及格（事先定）：每邊 0.10% 時，最好的做法在 24 個開盤時間裡至少 80% 報酬÷波動贏 50 日均線、三個時段都贏、贏過至少 90% 的隨機抽樣、四個其他幣至少 3 個也贏。",
        "七個做法（事先定、沒有調參數）：基本版 k＝0.5；加「開盤高於前 5 天收盤平均才做」；k 改成前 20 天「1 − |開盤−收盤|÷振幅」的平均（雜訊比）；"
        "部位＝2%÷前一天振幅占開盤的百分比（最多 100%）；三個全加；基本版和三個全加再各加「前一天收盤高於前 50 天收盤平均才做」。"
        "「最好的做法」事先定為：比特幣、UTC 0 點、目標價成交、0.10% 時報酬÷波動最高的那一個——這是從七個裡挑最好的，本身就偏樂觀。",
        f"量測期間從每個開盤時間的第 {WARM + 1} 天起（50 日均線和雜訊比都有值以後），比特幣大約是 {first} 起，所以「2017-08～2019」這段實際從 {first[:7]} 開始。",
        "沒有偷看未來：目標價只用前一天的最高、最低和今天開盤；5 日、50 日均線和雜訊比只用前一天以前的收盤；「目標價成交」只在某個小時的最高價碰到目標價時才成交，"
        "價格用目標價（若那個小時一開盤就已經高於目標價，用那個小時的開盤價）；保守版用那個整點收盤價，若只有最後一個小時收在目標價之上就不做；"
        "賣出是當天最後一個整點收盤（＝隔天開盤）。",
        "「目標價成交」假設剛好在目標價買到，沒有滑價；實際用市價單或停損買單追突破通常會買得更貴，保守版比較接近一般人看到突破才下單的情況。"
        "1 小時線只看得到每小時的開高低收，同一個小時內先漲還是先跌看不到。",
        "手續費「每邊」是買一次扣一次、賣一次再扣一次，所以每筆來回是兩倍。幣安現貨一般費率約 0.10%、韓國 Upbit 約 0.05%，但追突破的滑價常常更多；"
        "這個做法幾乎天天進出，一年來回上百次，手續費和滑價是成敗關鍵。現金沒有利息，也沒有算稅。",
        "隨機時間對照的限制：同一批日子是「事後知道當天有漲破」的日子，隨機版若在漲破前就進場，等於偷看了當天後來會漲，所以這個比法偏袒隨機版；"
        "另外列的「隨機挑日子」版本（在做法的條件允許的日子裡隨機挑同樣多天、同樣的進場時間分布）沒有偷看，但它同時也拿掉了「挑日子」的效果。",
        f"24 個開盤時間只差幾個小時、相鄰的起算日共用大部分歷史，所以這些「幾個裡幾個」不是互相獨立的證據。隨機抽樣用固定的亂數種子（{SEED}），資料沒變的話重跑結果一樣。",
        "其他幣是今天還在的大幣，有倖存者偏誤；各自的期間不同，彼此的數字不能直接比大小。",
        "來源核對：SC4RECOIN 的 GitHub 範例用前一天振幅、k＝0.6、當天結束平倉，README 自己寫 2020 年 ETH 永續的回測報酬輸給一直抱著、但虧損淺很多；"
        "TradingView「Volatility Breakout Daily [Backtest]」寫的是漲破當天的波動線就買、隔天開盤賣，k 可調；MQL5 的 Larry Williams 系列用前一根的振幅、"
        "倍數預設 0.5，另外加了波段結構條件；pyupbit 的 README 本身只是 Upbit API 套件、沒有寫這條規則，用 pyupbit 寫的熱門範例"
        "（youtube-jocoding/pyupbit-autotrade）用的是 k＝0.5、漲破就市價買、日K結束前賣，回測版本假設剛好在目標價買到、當天收盤賣，另有 15 日均線版本。"
        "5 日均線過濾、雜訊比 k、2% 部位這三個附加條件是韓國散戶圈常見的說法（源自 Larry Williams 的書和韓國的書籍、部落格），這裡沒有找到可以直接引用的公開原文，照題目的定義測。",
        "過去的結果不保證未來會一樣。這裡只是比較一條固定規則過去在不同條件下做了什麼、今天的價位怎麼算，不是買賣建議。",
    ]
    if r["g"].missing:
        notes.append(f"比特幣 1 小時線有 {r['g'].missing} 個小時缺資料（幣安維修等），用前一小時的收盤補平。")
    notes += r["problems"]

    return {
        "id": "vol_breakout",
        "title": "韓國散戶愛用的波動突破",
        "question": ("「漲破開盤價加前一天振幅一半就買、隔天開盤賣」（Larry Williams 的波動突破，韓國散戶和 GitHub 自動交易程式很常用），"
                     "扣掉手續費後有沒有贏過 50 日均線？換開盤時間、換時段、換幣穩不穩？突破的時間點本身有沒有用？"),
        "asof": last,
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 1 小時線 API（BTC、ETH、SOL、BNB、XRP 對 USDT）",
             "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000"},
            {"name": "幣安 BTC／USDT 現貨", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "SC4RECOIN／simple-crypto-breakout-strategy（GitHub）",
             "url": "https://github.com/SC4RECOIN/simple-crypto-breakout-strategy"},
            {"name": "TradingView：Volatility Breakout Daily [Backtest]", "url": "https://il.tradingview.com/script/Yz4phkcH"},
            {"name": "MQL5：Larry Williams 波動突破系列", "url": "https://www.mql5.com/en/articles/21003"},
            {"name": "pyupbit（Upbit API 套件）", "url": "https://github.com/sharebook-kr/pyupbit"},
            {"name": "youtube-jocoding／pyupbit-autotrade（用 pyupbit 寫的波動突破範例）",
             "url": "https://github.com/youtube-jocoding/pyupbit-autotrade"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
