"""Historical study "clock_robustness": is the "which hour is the daily close" problem that trend_practical found for
the single 20/10 breakout rule shared by every trend variant the dashboard relies on - and by other coins?

Daily bars are rebuilt from 1-hour candles so that the day ends at each of the 24 hours (UTC). On every one of those
24 "clocks" the same rules are run: the signal is decided at the bar's close and is the position held from that
moment for the next 24 hours; 0.10% of the amount traded is charged (fractional positions are put back to their
target at every close, as trend_ensemble._run does). Whatever the clock, profit and loss is always measured on the
same cut (UTC midnight to UTC midnight) so the 24 versions and buy-and-hold are comparable.

1. bitcoin, nine rules, across the 24 clocks;
2. the "staggered" remedy: the money split into 24 (4, 2) equal parts, each on its own clock;
3. the staggered version by sub-period and calendar year against buy-and-hold;
4. ETH, SOL, BNB, XRP: the 20/10 rule and the 17-rule blend across the 24 clocks;
5. why the clock matters (trades that differ, the handful of trades behind the 00:00 UTC result, signal clock
   against execution clock);
6. the size of the clock luck (standard deviation across clocks) against the rule's average edge;
7. today's state of the 24 parts.

Data: Binance spot 1-hour candles. Bitcoin's store is the one seasonality.py keeps (read only, topped up in memory);
the other coins get their own files under cache/_store (clock_robustness_hourly_<SYMBOL>.csv). For bitcoin the
daily closes before 2017-08 (the dashboard's long daily series) are put in front of every clock's series only so
that the long look-backs are already warm on 2018-01-01. Only closed candles and finished UTC days, only public
key-less endpoints through common.get_json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common

SHARED_STORE = common.STORE / "seasonality_hourly.csv"  # kept by seasonality.py - read only
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
MAX_PAGES = 130
MIN_HOURS = 50_000

COST = 0.001  # 0.10% of the amount traded
YEAR = 365
MAIN_START = "2018-01-01"
TW = 8  # Taiwan is UTC+8
WARM = 201  # days a coin without earlier history needs before every one of the 17 rules has a value on all clocks
MIN_DAYS = 2 * YEAR
VOL_DAYS, VOL_TARGET = 30, 0.40
LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)  # the nine channels of donchian_ensemble
COINS = ("ETH", "SOL", "BNB", "XRP")
SPANS = (("2018–2019", "2018-01-01", "2019-12-31"), ("2020–2021", "2020-01-01", "2021-12-31"),
         ("2022–2023", "2022-01-01", "2023-12-31"), ("2024 到最新", "2024-01-01", None))
MOST = 18  # "most clocks": at least 18 of 24

RULES = ("c20", "c55", "ma50", "ma100", "ma200", "mom60", "blend", "vol40", "don")
NAMES = {
    "c20": "創 20 日新高進場、10 日新低出場",
    "c55": "創 55 日新高進場、20 日新低出場",
    "ma50": "收盤高於 50 日均線才持有",
    "ma100": "收盤高於 100 日均線才持有",
    "ma200": "收盤高於 200 日均線才持有",
    "mom60": "過去 60 天是漲的才持有",
    "blend": "17 條順勢規則混合（照看多的比例持有）",
    "vol40": "17 條混合 × 波動調整（目標 40%）",
    "don": "九條新高規則＋移動停損（不調波動）",
}
SHORT = {"c20": "20 日新高／10 日新低", "c55": "55 日新高／20 日新低", "ma50": "50 日均線", "ma100": "100 日均線",
         "ma200": "200 日均線", "mom60": "60 天漲跌", "blend": "17 條混合", "vol40": "混合×波動 40%",
         "don": "九條新高＋移動停損"}


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


def _pp(v) -> str:
    """A difference between two percentages, in percentage points."""
    if not _ok(v):
        return "—"
    v = float(v)
    return "0.0" if abs(v) < 0.05 else f"{v:+,.1f}"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _tw(hour_utc: int) -> str:
    """UTC hour as Taiwan clock time."""
    t = (hour_utc + TW) % 24
    part = "半夜" if t == 0 else "凌晨" if t < 6 else "早上" if t < 12 else "中午" if t == 12 else "下午" if t < 18 else "晚上"
    return f"台灣{part} {12 if t % 12 == 0 else t % 12} 點"


def _tri(values, fmt) -> str:
    return f"{fmt(float(np.min(values)))}／{fmt(float(np.median(values)))}／{fmt(float(np.max(values)))}"


def _of24(n: int) -> str:
    return f"24 個時鐘裡 {int(n)} 個"


# ---------- data ----------

def _own_store(symbol: str):
    return common.STORE / f"clock_robustness_hourly_{symbol}.csv"


def _read(path) -> dict[int, float]:
    """{candle open time in ms: close} from a store file (ms,close or ms,open,close); empty if missing or damaged."""
    rows: dict[int, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            c = float(parts[-1])
            if c > 0 and math.isfinite(c):
                rows[int(parts[0])] = c
    except Exception:  # noqa: BLE001 - no file or a damaged one
        return {}
    return rows


def _write(symbol: str, rows: dict[int, float]) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        text = "ms,close\n" + "\n".join(f"{ms},{c!r}" for ms, c in sorted(rows.items())) + "\n"
        _own_store(symbol).write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run simply fetches again
        pass


def _fetch(symbol: str, rows: dict[int, float], keep: bool) -> int:
    """Add closed 1-hour candles from the last stored hour on (in place). Returns how many were added."""
    from dashboard.collectors import price as px

    start = max(rows) if rows else 0  # 0 = from the coin's first candle
    added = 0
    try:
        for page_no in range(MAX_PAGES):
            page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 1000,
                                                  "startTime": start})
            if not page:
                break
            now_ms = int(time.time() * 1000)
            for r in page:
                if int(r[6]) < now_ms and float(r[4]) > 0:  # closed candles only
                    rows[int(r[0])] = float(r[4])
                    added += 1
            if len(page) < 1000:
                break
            start = int(page[-1][0]) + 1
            if keep and page_no % 20 == 19:
                _write(symbol, rows)  # keep what a long first run has fetched so far
    finally:
        if keep and added:
            _write(symbol, rows)
    return added


def _hourly(symbol: str) -> tuple[dict[int, float], str | None]:
    """{candle open time in ms: close} of closed 1-hour candles, and a problem text if an update failed.

    Bitcoin: the store of seasonality.py is read and never written (missing hours are fetched into memory only);
    only when it is not there does this module keep its own bitcoin file. Other coins always use their own file."""
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    rows, shared = {}, False
    if symbol == "BTCUSDT":
        rows = _read(SHARED_STORE)
        shared = len(rows) >= MIN_HOURS
    if not shared:
        rows = _read(_own_store(symbol))
    problem = None
    if not rows or max(rows) < today_ms - HOUR_MS:  # yesterday's last hour is not there yet
        try:
            _fetch(symbol, rows, keep=not shared)
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = (f"{symbol[:-4]} 的幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    return rows, problem


def _grid(rows: dict[int, float]) -> tuple[np.ndarray, pd.Timestamp, int]:
    """Price at every full hour from the first UTC midnight to the last one: P[k] is the price at B0 + k hours
    (= the close of the candle that opened one hour earlier). A missing hour repeats the last known price.
    Returns (P, B0, number of missing hours); len(P) = 24 * days + 1."""
    first = min(rows) + HOUR_MS
    b0 = -(-first // DAY_MS) * DAY_MS
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    end = min(today_ms, ((max(rows) + HOUR_MS) // DAY_MS) * DAY_MS)  # finished UTC days only
    k = (end - b0) // HOUR_MS + 1
    if k < 24 * 200:
        raise RuntimeError("1 小時線太短")
    price = np.full(k, np.nan)
    for ms, c in rows.items():
        i = (ms + HOUR_MS - b0) // HOUR_MS
        if 0 <= i < k:
            price[i] = c
    missing = int(np.isnan(price).sum())
    filled = pd.Series(price).ffill().bfill().values
    return filled, pd.Timestamp(b0, unit="ms"), missing


def _prehistory(b0: pd.Timestamp) -> np.ndarray:
    """Bitcoin daily closes before the hourly data (the dashboard's long daily series), oldest first; empty if the
    series cannot be had. The last one is the close of the day before the first hourly candle's day."""
    try:
        from dashboard.collectors import classic

        close = classic._long_close()
        close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
        close = close[(close > 0) & (close.index < b0 - pd.Timedelta(days=1))]
        if len(close) < 400:
            return np.array([])
        return close.asfreq("D").ffill().values
    except Exception:  # noqa: BLE001 - without it the long look-backs simply start later
        return np.array([])


# ---------- the rules ----------

def _breakout(s: pd.Series, n_in: int, n_out: int) -> np.ndarray:
    """1 = long from a close that is the highest of the last n_in closes until one that is the lowest of n_out."""
    high = s.rolling(n_in, min_periods=n_in).max()
    low = s.rolling(n_out, min_periods=n_out).min()
    ready = high.notna().values
    raw = np.where(ready & (s >= high).values, 1.0, np.where(ready & (s <= low).values, 0.0, np.nan))
    return pd.Series(raw).ffill().fillna(0.0).values


def _wants(closes: np.ndarray, keys) -> dict[str, np.ndarray]:
    """What each rule wants to hold after each close of one clock's daily series (0 while a rule has no value)."""
    from dashboard.studies2 import trend_ensemble as te

    s = pd.Series(np.asarray(closes, dtype=float))
    signal, _ = te._signals(s)  # the 17 rules exactly as trend_ensemble builds them
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    out = {"c20": signal["C20"].fillna(0.0).values, "blend": share.fillna(0.0).values}
    if "c55" in keys:
        out["c55"] = _breakout(s, 55, 20)
    for n in (50, 100, 200):
        if f"ma{n}" in keys:
            out[f"ma{n}"] = signal[f"A{n}"].fillna(0.0).values
    if "mom60" in keys:
        out["mom60"] = signal["D60"].fillna(0.0).values
    if "vol40" in keys:
        vol = s.pct_change().rolling(VOL_DAYS, min_periods=VOL_DAYS).std(ddof=1) * math.sqrt(YEAR)
        scale = (VOL_TARGET / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
        out["vol40"] = (share * scale).fillna(0.0).values
    if "don" in keys:
        from dashboard.studies2 import donchian_ensemble as de

        out["don"] = np.mean([de._channel(s.values, n, True)[0] for n in LOOKBACKS], axis=0)
    return {k: out[k] for k in keys}


def _stats(net: np.ndarray) -> dict:
    """Yearly compound return %, yearly volatility %, return / volatility and deepest loss % of daily net returns."""
    n = len(net)
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100) if n > 1 else 0.0
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"ret": float(ret), "vol": vol, "ratio": float(ret / vol) if vol > 1e-9 else 0.0, "deepest": deepest}


class _Market:
    """One coin's hourly price grid and the engine that turns one clock's positions into UTC-day net returns."""

    def __init__(self, price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray | None = None):
        self.price, self.b0, self.k = price, b0, len(price)
        self.days = (self.k - 1) // 24
        self.dates = pd.date_range(b0, periods=self.days, freq="D")  # day j runs from boundary j to boundary j+1
        self.pad = np.concatenate([price, np.full(24, price[-1])])  # flat after the end: an unfinished last period
        self.bound = price[::24]
        self.change = self.bound[1:] / self.bound[:-1] - 1  # buy-and-hold's return of UTC day j
        self.pre = np.asarray(pre if pre is not None else [], dtype=float)
        start = max(int((pd.Timestamp(MAIN_START) - b0).days), 0)
        self.w0 = start if len(self.pre) else max(start, WARM)  # first day that is measured
        if self.days - self.w0 < MIN_DAYS:
            raise RuntimeError("1 小時線涵蓋的期間不夠")
        self.years = (self.days - self.w0) / YEAR

    def wants(self, h: int, keys) -> dict[str, np.ndarray]:
        """Each rule's wanted position after every close of clock h; index j is the close at boundary h + 24j
        (for h = 0 there is one more value than days: the close at the very end, used only for today's state)."""
        got = _wants(np.concatenate([self.pre, self.price[h::24]]), keys)
        return {k: v[len(self.pre):] for k, v in got.items()}

    def run(self, h: int, want: np.ndarray, cost: float = COST) -> dict:
        """want[j] is held from boundary h + 24j for 24 hours and then put back to want[j+1].

        Like trend_ensemble._run, one period's result is 1 + w * move - cost * trade, and the trade is the gap
        between the new target and what the old position drifted to. Equity is followed hour by hour and then
        read at every UTC midnight, so every clock is measured on the same days."""
        days = self.days
        w = np.asarray(want, dtype=float)[:days]
        seg = self.pad[h:h + 24 * days + 1]
        path = seg[1:].reshape(days, 24) / seg[:-1:24][:, None] - 1
        r = path[:, -1]
        base = 1 + w * r
        drift = w * (1 + r) / np.where(np.abs(base) > 1e-9, base, 1.0)
        trade = np.abs(w - np.concatenate([[0.0], drift[:-1]]))  # starts from cash
        factor = 1 + w[:, None] * path - cost * trade[:, None]
        before = np.concatenate([[1.0], np.cumprod(factor[:, -1])[:-1]])
        equity = np.ones(h + 24 * days + 1)
        equity[h + 1:] = (before[:, None] * factor).ravel()
        mid = equity[:self.k:24]
        net = mid[1:] / mid[:-1] - 1
        out = _stats(net[self.w0:])
        out.update(net=net, trade=trade, want=w, turnover=float(trade[self.w0:].sum() / self.years),
                   share=float(w[self.w0:].mean() * 100))
        return out

    def hold(self) -> dict:
        out = _stats(self.change[self.w0:])
        out.update(net=self.change, turnover=0.0, share=100.0)
        return out

    def stagger(self, runs: list[dict]) -> dict:
        """The money split into equal parts on the first measured day, each part following one clock, never
        moved between the parts afterwards."""
        w = self.w0
        eq = np.array([np.concatenate([[1.0], np.cumprod(1 + x["net"][w:])]) for x in runs])
        port = eq.mean(axis=0)
        net = port[1:] / port[:-1] - 1
        traded = (np.array([x["trade"][w:] for x in runs]) * eq[:, :-1]).mean(axis=0) / port[:-1]
        out = _stats(net)
        out.update(net_main=net, turnover=float(traded.sum() / self.years))
        return out


def _across(runs: list[dict], hold: dict) -> dict:
    """Summary of one rule over the 24 clocks."""
    ret = np.array([x["ret"] for x in runs])
    ratio = np.array([x["ratio"] for x in runs])
    deep = np.array([x["deepest"] for x in runs])
    order = sorted(range(len(runs)), key=lambda h: -ret[h])
    return {"ret": ret, "ratio": ratio, "deep": deep, "beat": int((ret > hold["ret"]).sum()),
            "beat_ratio": int((ratio > hold["ratio"]).sum()), "shallow": int((deep > hold["deepest"]).sum()),
            "rank0": order.index(0) + 1, "std": float(np.std(ret, ddof=1)), "mean": float(ret.mean()),
            "median": float(np.median(ret)), "order": order}


def _by_period(m: _Market, net_main: np.ndarray) -> dict:
    """Yearly return and deepest loss inside each span, and total return / worst fall inside each calendar year."""
    dates = m.dates[m.w0:]
    series = pd.Series(net_main, index=dates)
    spans = {}
    for name, start, end in SPANS:
        part = series[(dates >= pd.Timestamp(start)) & (dates <= (pd.Timestamp(end) if end else dates[-1]))]
        spans[name] = _stats(part.values) if len(part) > 30 else None
    years = {}
    for y, part in series.groupby(series.index.year):
        equity = np.concatenate([[1.0], np.cumprod(1 + part.values)])
        years[int(y)] = (float((equity[-1] - 1) * 100), float((equity / np.maximum.accumulate(equity) - 1).min() * 100))
    return {"spans": spans, "years": years}


def _trades(want: np.ndarray, h: int, first_day: int) -> list[tuple[int, int | None]]:
    """(entry hour, exit hour or None while still open) of a 0/1 position on clock h, entries from first_day on."""
    held = want > 0
    before = np.concatenate([[False], held[:-1]])
    entries = np.where(held & ~before)[0]
    exits = np.where(~held & before)[0]
    out = []
    for e in entries:
        if e < first_day:
            continue
        later = exits[exits > e]
        out.append((h + 24 * int(e), h + 24 * int(later[0]) if len(later) else None))
    return out


def _diagnose(m: _Market, wants: list[np.ndarray], runs: list[dict], across: dict) -> dict:
    """Why the clock matters for the 20/10 rule."""
    w0, days = m.w0, m.days
    trades = [_trades(wants[h][:days], h, w0) for h in range(24)]

    def overlap(step: int) -> tuple[float, float]:
        both, entry = [], []
        for h in range(24):
            other = trades[(h + step) % 24]
            pairs, starts = set(other), {e for e, _ in other}
            mine = trades[h]
            if not mine:
                continue
            both.append(np.mean([(e + step, None if x is None else x + step) in pairs for e, x in mine]))
            entry.append(np.mean([e + step in starts for e, _ in mine]))
        return float(np.mean(both) * 100), float(np.mean(entry) * 100)

    hourly = []  # hourly[h][k]: what clock h holds during the hour that starts at boundary k
    for h in range(24):
        pos = np.zeros(24 * days)
        pos[h:] = np.repeat(wants[h][:days], 24)[: 24 * days - h]
        hourly.append(pos[24 * w0:])

    def differ(step: int) -> float:
        return float(np.mean([np.mean(hourly[h] != hourly[(h + step) % 24]) for h in range(24)]) * 100)

    # ----- the handful of trades behind clock 0's result -----
    ret = across["ret"]
    middle = across["order"][12]  # the clock ranked 13th of 24
    eq0 = np.concatenate([[1.0], np.cumprod(1 + runs[0]["net"])])
    eqm = np.concatenate([[1.0], np.cumprod(1 + runs[middle]["net"])])
    held0 = wants[0][:days] > 0
    entries = [j for j in range(w0 + 1, days) if held0[j] and not held0[j - 1]]
    edges = [w0] + entries + [days]
    mid_trades = trades[middle]
    windows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        g0, gm = eq0[b] / eq0[a], eqm[b] / eqm[a]
        inside = np.where(~held0[a:b])[0]
        out_day = a + int(inside[0]) if len(inside) else None
        item = {"start": a, "end": b, "exit": out_day, "g0": float((g0 - 1) * 100), "gm": float((gm - 1) * 100),
                "diff": float(math.log(g0) - math.log(gm)), "opened": a > w0,
                "lag_in": None, "lag_out": None,
                "mid_entries": sum(1 for e, _ in mid_trades if 24 * a <= e < 24 * b)}
        if item["opened"] and mid_trades:
            near = min(mid_trades, key=lambda t: abs(t[0] - 24 * a))
            if abs(near[0] - 24 * a) <= 24 * 10:
                item["lag_in"] = near[0] - 24 * a
                if out_day is not None and near[1] is not None:
                    item["lag_out"] = near[1] - 24 * out_day
        windows.append(item)
    total = float(math.log(eq0[days] / eq0[w0]) - math.log(eqm[days] / eqm[w0]))
    top = sorted(windows, key=lambda x: -x["diff"])[:5]
    top_sum = float(sum(x["diff"] for x in top))
    growth = eq0[days] / eq0[w0] / math.exp(top_sum)
    without = float((growth ** (1 / m.years) - 1) * 100)

    # ----- signal clock a, execution clock b -----
    grid = np.zeros((24, 24))
    deep = np.zeros((24, 24))
    for a in range(24):
        wa = wants[a][:days]
        late = np.concatenate([[0.0], wa[:-1]])
        for b in range(24):
            x = runs[a] if a == b else m.run(b, wa if a <= b else late)
            grid[a, b], deep[a, b] = x["ret"], x["deepest"]
    grand = grid.mean()
    ss = float(((grid - grand) ** 2).sum())
    by_signal = float(24 * ((grid.mean(axis=1) - grand) ** 2).sum() / ss * 100) if ss > 0 else 0.0
    by_exec = float(24 * ((grid.mean(axis=0) - grand) ** 2).sum() / ss * 100) if ss > 0 else 0.0
    worst, best = int(np.argmin(ret)), int(np.argmax(ret))
    others = []
    for h in (worst, middle, 12, best):
        if h != 0 and h not in others:
            others.append(h)
    pairs = [(a, b) for h in others[:3] for a, b in ((0, h), (h, 0))]
    return {"both1": overlap(1), "both12": overlap(12), "differ1": differ(1), "differ12": differ(12),
            "counts": [len(t) for t in trades], "middle": middle, "windows": windows, "top": top, "total": total,
            "top_sum": top_sum, "without": without, "grid": grid, "deep": deep, "by_signal": by_signal,
            "by_exec": by_exec, "pairs": pairs, "grow0": float(eq0[days] / eq0[w0]),
            "growm": float(eqm[days] / eqm[w0])}


def _coin(m: _Market, keys, full: bool) -> dict:
    """All clocks of one coin for the given rules, plus staggered versions and buy-and-hold."""
    hold = m.hold()
    wants = {k: [] for k in keys}
    runs = {k: [] for k in keys}
    for h in range(24):
        got = m.wants(h, keys)
        for k in keys:
            wants[k].append(got[k])
            runs[k].append(m.run(h, got[k]))
    across = {k: _across(runs[k], hold) for k in keys}
    stag = {k: {24: m.stagger(runs[k])} for k in keys}
    if full:
        for k in keys:
            for parts in (4, 2):
                step = 24 // parts
                sets = [m.stagger([runs[k][h] for h in range(off, 24, step)]) for off in range(step)]
                stag[k][parts] = sets[0]
                stag[k][parts]["range"] = (min(x["ret"] for x in sets), max(x["ret"] for x in sets))
                stag[k][parts]["deep_range"] = (min(x["deepest"] for x in sets), max(x["deepest"] for x in sets))
    now = {k: [float(wants[k][h][-1]) for h in range(24)] for k in keys}
    return {"m": m, "hold": hold, "wants": wants, "runs": runs, "across": across, "stag": stag, "now": now}


def _daily_check(r: dict) -> str | None:
    """Compare the 00:00 UTC clock with the same rules on the dashboard's own daily series (trend_ensemble)."""
    try:
        from dashboard.studies2 import trend_ensemble as te

        close = te._daily_close()
        target = te._targets(close)[0]
        m = r["m"]
        last = m.dates[-1]
        parts = []
        for key in ("c20", "blend"):
            net, _ = te._run(target[key].shift(1).values, close.pct_change().values, COST)
            net = pd.Series(net, index=close.index)
            net = net[(net.index >= m.dates[m.w0]) & (net.index <= last)]
            ref, mine = _stats(net.values), r["runs"][key][0]
            parts.append(f"「{SHORT[key]}」日線版每年 {_pct(ref['ret'])}、最深虧 {_pct(ref['deepest'])}，"
                         f"這裡 {_pct(mine['ret'])}、{_pct(mine['deepest'])}")
        return ("核對：台灣早上 8 點（UTC 0 點）這個時鐘用 1 小時線拼出來的結果，和儀表板原本用日線算的同一條規則比——"
                + "；".join(parts) + "。兩邊對得上，表示換時鐘的差異不是算法不同造成的。")
    except Exception:  # noqa: BLE001 - the cross-check is optional
        return None


def _compute() -> dict:
    rows, problem = _hourly("BTCUSDT")
    if len(rows) < MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, missing = _grid(rows)
    pre = _prehistory(b0)
    m = _Market(price, b0, pre)
    r = _coin(m, RULES, full=True)
    r.update(problem=problem, missing=missing, warm=len(pre) > 0, last=m.dates[-1], last_price=float(price[-1]))
    r["diag"] = _diagnose(m, r["wants"]["c20"], r["runs"]["c20"], r["across"]["c20"])
    r["hold_periods"] = _by_period(m, r["hold"]["net"][m.w0:])
    r["periods"] = {k: _by_period(m, r["stag"][k][24]["net_main"]) for k in RULES}
    r["rebalanced"] = {k: _stats(np.mean([x["net"][m.w0:] for x in r["runs"][k]], axis=0)) for k in ("c20", "blend")}

    coins, problems = {}, []
    for coin in COINS:
        try:
            crow, cproblem = _hourly(f"{coin}USDT")
            if cproblem:
                problems.append(cproblem)
            if len(crow) < 24 * (WARM + MIN_DAYS):
                raise RuntimeError("太短")
            cprice, cb0, _ = _grid(crow)
            cm = _Market(cprice, cb0)
            if cm.dates[-1] != m.dates[-1] and not cproblem:
                problems.append(f"{coin} 的 1 小時線只到 {_d(cm.dates[-1])}。")
            coins[coin] = _coin(cm, ("c20", "blend"), full=False)
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{coin} 的 1 小時線這次拿不到或太短（{type(exc).__name__}），沒有算進去。")
    r.update(coins=coins, coin_problems=problems)
    return r


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    m, hold, across, stag, diag = r["m"], r["hold"], r["across"], r["stag"], r["diag"]
    last = r["last"]
    start = _d(m.dates[m.w0])
    span = f"{start} 到 {_d(last)}"
    hp = r["hold_periods"]
    years = sorted(hp["years"])
    n_years = len(years)
    partial = last < pd.Timestamp(f"{years[-1]}-12-31")

    # ----- table 1: nine rules across the 24 clocks -----
    rows1 = []
    for k in RULES:
        a = across[k]
        rows1.append([NAMES[k], _tri(a["ret"], _pct), _of24(a["beat"]),
                      f"{a['std']:.1f}｜{_pp(a['mean'] - hold['ret'])}", _tri(a["ratio"], _num), _of24(a["beat_ratio"]),
                      f"{_pct(a['deep'].min())} ～ {_pct(a['deep'].max())}｜{_of24(a['shallow'])}",
                      f"{_pct(r['runs'][k][0]['ret'])}｜第 {a['rank0']}"])
    rows1.append(["對照：一直抱著不動", _pct(hold["ret"]), "—", "—", _num(hold["ratio"]), "—", _pct(hold["deepest"]), "—"])

    # ----- table 2: staggered versions and sub-periods -----
    def win_years(per: dict) -> str:
        more = sum(1 for y in years if per["years"][y][0] > hp["years"][y][0])
        shallow = sum(1 for y in years if per["years"][y][1] > hp["years"][y][1] + 1e-9)
        return f"{n_years} 年裡 {more} 年｜{n_years} 年裡 {shallow} 年"

    def span_cells(per: dict) -> list:
        return [f"{_pct(per['spans'][name]['ret'])}｜{_pct(per['spans'][name]['deepest'])}" if per["spans"][name] else "—"
                for name, _, _ in SPANS]

    def row2(name: str, x: dict, per: dict, hold_row: bool = False) -> list:
        return ([name, f"{_pct(x['ret'])}｜{_pct(x['vol'], signed=False)}｜{_num(x['ratio'])}",
                 f"{_pct(x['deepest'])}｜{'—' if hold_row else format(x['turnover'], '.1f') + ' 次'}"]
                + span_cells(per) + ["—" if hold_row else win_years(per)])

    rows2 = [row2("一直抱著不動", hold, hp, hold_row=True)]
    for k in RULES:
        rows2.append(row2(f"分成 24 份：{SHORT[k]}", stag[k][24], r["periods"][k]))
    for k in ("c20", "blend"):
        for parts in (4, 2):
            hours = "台灣" + "、".join(_tw(h)[2:] for h in range(0, 24, 24 // parts))
            x = stag[k][parts]
            rows2.append(row2(f"分成 {parts} 份（{hours}）：{SHORT[k]}", x, _by_period(m, x["net_main"])))
    base0 = r["runs"]["c20"][0]
    rows2.append(row2(f"對照：只用一個時鐘（{_tw(0)}）：{SHORT['c20']}", base0, _by_period(m, base0["net"][m.w0:])))

    # ----- table 3: other coins -----
    def coin_rows(name: str, c: dict) -> list:
        out = []
        for k in ("c20", "blend"):
            a, h, s = c["across"][k], c["hold"], c["stag"][k][24]
            out.append([f"{name}（{_d(c['m'].dates[c['m'].w0])} 起）：{SHORT[k]}",
                        f"{_pct(h['ret'])}｜{_num(h['ratio'])}｜{_pct(h['deepest'])}", _tri(a["ret"], _pct),
                        _of24(a["beat"]), _of24(a["beat_ratio"]),
                        f"{_pct(a['deep'].min())} ～ {_pct(a['deep'].max())}｜{_of24(a['shallow'])}",
                        f"{_pct(c['runs'][k][0]['ret'])}｜第 {a['rank0']}",
                        f"{_pct(s['ret'])}｜{_num(s['ratio'])}｜{_pct(s['deepest'])}"])
        return out

    rows3 = coin_rows("比特幣", r)
    for coin in COINS:
        if coin in r["coins"]:
            rows3 += coin_rows(coin, r["coins"][coin])

    # ----- table 4: diagnosis -----
    a20 = across["c20"]
    mid = diag["middle"]
    run_mid = r["runs"]["c20"][mid]

    def lag(hours) -> str:
        if hours is None:
            return "—"
        if hours == 0:
            return "同一時刻"
        size = abs(hours)
        text = f" {size} 小時" if size < 12 else f"約 {max(1, round(size / 24))} 天"
        return ("晚" if hours > 0 else "早") + text

    rows4 = [
        ["只差 1 小時的兩個時鐘，交易有多像", f"進場、出場都在同一天的交易：平均 {diag['both1'][0]:.0f}%",
         f"差 12 小時的兩個時鐘：{diag['both12'][0]:.0f}%",
         f"其餘 {100 - diag['both1'][0]:.0f}% 的交易至少有一邊差了一天以上，或只出現在其中一個時鐘",
         f"只看進場：{diag['both1'][1]:.0f}% 在同一天（差 12 小時的是 {diag['both12'][1]:.0f}%）；"
         f"24 個時鐘的進場次數從 {min(diag['counts'])} 到 {max(diag['counts'])} 次"],
        ["兩個時鐘「一個持有、一個現金」的時間", f"只差 1 小時：{diag['differ1']:.1f}% 的時間",
         f"差 12 小時：{diag['differ12']:.1f}% 的時間", "—", "同一條規則、同一段歷史，只是收盤時刻不同"],
    ]
    for i, x in enumerate(diag["top"], 1):
        end = m.dates[x["exit"]] if x["exit"] is not None else None
        when = f"{_d(m.dates[x['start']])} 進場" if x["opened"] else f"期初 {_d(m.dates[x['start']])} 起"
        if not x["opened"] and x["exit"] == x["start"]:
            end = None
        if x["opened"] or end is not None:
            when += f"、{_d(end)} 出場" if end is not None else "、到期末還沒出場"
        note = (f"中位時鐘最接近的那筆交易：進場{lag(x['lag_in'])}、出場{lag(x['lag_out'])}" if x["lag_in"] is not None
                else "中位時鐘在前後 10 天內沒有對應的進場")
        note += f"；這一段裡{_tw(0)}的時鐘進場 1 次、中位時鐘進場 {x['mid_entries']} 次" if x["opened"] else ""
        rows4.append([f"{_tw(0)}的時鐘贏最多的第 {i} 段：{when}（算到下一次進場前）", f"這一段 {_pct(x['g0'])}",
                      f"中位時鐘同一段 {_pct(x['gm'])}", f"{(math.exp(x['diff']) - 1) * 100:+.1f}%（相對多出）", note])
    share5 = diag["top_sum"] / diag["total"] * 100 if abs(diag["total"]) > 1e-9 else None
    rows4.append([f"這五段合計（{_tw(0)}的時鐘共 {len(diag['windows'])} 段）",
                  f"期末資金是期初的 {diag['grow0']:.1f} 倍（每年 {_pct(base0['ret'])}）",
                  f"中位時鐘（{_tw(mid)}）{diag['growm']:.1f} 倍（每年 {_pct(run_mid['ret'])}）",
                  (f"五段占全部差距的 {share5:.0f}%" if _ok(share5) else "—"),
                  f"把這五段換成中位時鐘的成績，{_tw(0)}的時鐘會變成每年 {_pct(diag['without'])}"])
    for a, b in diag["pairs"]:
        x, pa, pb = diag["grid"][a, b], diag["grid"][a, a], diag["grid"][b, b]
        nearer = "比較接近訊號那個時鐘" if abs(x - pa) < abs(x - pb) else "比較接近成交那個時鐘"
        if abs(abs(x - pa) - abs(x - pb)) < 1.0:
            nearer = "兩邊差不多遠"
        rows4.append([f"訊號看{_tw(a)}的收盤，等到{_tw(b)}才用當時的價格成交（晚 {(b - a) % 24} 小時）",
                      f"每年 {_pct(x)}｜最深虧 {_pct(diag['deep'][a, b])}",
                      f"訊號那個時鐘自己做 {_pct(pa)}；成交那個時鐘自己做 {_pct(pb)}",
                      f"離訊號時鐘 {_pp(x - pa)}、離成交時鐘 {_pp(x - pb)}", nearer])
    rest = max(0.0, 100 - diag["by_signal"] - diag["by_exec"])
    rows4.append(["24 種訊號時鐘 × 24 種成交時鐘全部跑一遍（576 種）", f"每年報酬的差異有 {diag['by_signal']:.0f}% 跟著訊號時鐘走",
                  f"{diag['by_exec']:.0f}% 跟著成交時鐘走", f"其餘 {rest:.0f}% 是兩者的搭配（包含等待的時間長短）",
                  f"576 種的每年報酬從 {_pct(diag['grid'].min())} 到 {_pct(diag['grid'].max())}"])

    tables = [
        {"caption": (f"比特幣九種順勢規則，各用 24 個收盤時刻（時鐘）跑一次（{span}，成本 {COST * 100:.2f}%）。"
                     "每一欄的「最低／中位／最高」是 24 個時鐘的結果。「標準差」是 24 個時鐘每年報酬的離散程度（百分點），"
                     f"「平均多幾點」是 24 個時鐘的平均每年報酬減掉一直抱著的 {_pct(hold['ret'])}：標準差比平均多出來的還大，"
                     "就表示單看一個時鐘的報酬優勢分不出是規則有用還是時鐘運氣。最後一欄是儀表板原本用的時鐘（UTC 0 點＝"
                     f"{_tw(0)}）的每年報酬和它在 24 個裡的排名（第 1 最高）。"),
         "columns": ["規則", "每年報酬：最低／中位／最高", "報酬贏過一直抱著的時鐘", "時鐘之間的標準差｜平均比一直抱著多幾點",
                     "報酬÷波動：最低／中位／最高", "報酬÷波動贏過一直抱著的時鐘", "最深虧損：最深～最淺｜比一直抱著淺的時鐘",
                     f"{_tw(0)}的時鐘：每年報酬｜排名"],
         "rows": rows1},
        {"caption": (f"把錢平分到不同時鐘的「分散版」和一直抱著比（比特幣，{span}）。分成 24 份＝{start} 那天把錢分成 24 等份，"
                     "每一份照同一條規則、但各用一個不同的收盤時刻，之後各做各的、不互相調撥；這樣就沒有「剛好選到哪個時鐘」的運氣。"
                     "四個期間的格子是「那段期間的每年報酬｜那段期間內最深虧損」。最後一欄是逐年比較："
                     f"{n_years} 個日曆年{'（最後一年還沒過完）' if partial else ''}裡有幾年報酬比一直抱著高｜有幾年年內最大跌幅比一直抱著淺。"
                     "換手 1 次＝一年內買賣的總金額等於全部資金一次。"),
         "columns": ["做法", "每年報酬｜每年波動｜報酬÷波動", "最深虧損｜一年換手", SPANS[0][0], SPANS[1][0], SPANS[2][0],
                     f"2024 到 {_d(last)}", "報酬贏過一直抱著的年數｜跌得比較淺的年數"],
         "rows": rows2},
        {"caption": ("其他幣也做同樣的事：「創 20 日新高進場、10 日新低出場」和「17 條混合」各用 24 個時鐘跑一次，和各自的一直抱著比"
                     f"（幣安現貨 1 小時線，成本 {COST * 100:.2f}%）。比特幣以外的幣沒有更早的價格可以暖機，所以從上架滿 {WARM} 天"
                     "（而且不早於 2018 年）才開始算，每種幣的起算日不同，彼此的數字不能直接比大小。"
                     "最後一欄是平分到 24 個時鐘的分散版。"),
         "columns": ["幣（起算日）與規則", "一直抱著：每年報酬｜報酬÷波動｜最深虧", "24 個時鐘每年報酬：最低／中位／最高",
                     "報酬贏過一直抱著的時鐘", "報酬÷波動贏過一直抱著的時鐘", "最深虧損：最深～最淺｜比一直抱著淺的時鐘",
                     f"{_tw(0)}的時鐘：每年報酬｜排名", "分成 24 份：每年報酬｜報酬÷波動｜最深虧"],
         "rows": rows3},
        {"caption": (f"時鐘為什麼有差（比特幣「{SHORT['c20']}」，{span}）。前兩列看不同時鐘的交易有多不一樣；中間幾列把"
                     f"{_tw(0)}的時鐘的歷史切成一段一段（每一段從一次進場到下一次進場前），找出它比「中位時鐘」"
                     f"（24 個裡每年報酬排第 13 的{_tw(mid)}）多賺最多的五段；最後幾列把「看哪個時鐘的訊號」和「在哪個時刻成交」拆開："
                     "訊號照 A 時鐘的收盤判斷，但等到下一個 B 時刻才用當時的價格成交。"),
         "columns": ["看什麼", "結果", "拿來比的", "差距", "補充"],
         "rows": rows4},
    ]

    # ----- today -----
    long20 = int(sum(v > 0 for v in r["now"]["c20"]))
    blend_now = np.array(r["now"]["blend"]) * 100
    now = [
        {"label": f"最後一個收完的日子（{_d(last)}，台灣時間隔天早上 8 點收）", "value": f"比特幣 {r['last_price']:,.0f} 美元"},
        {"label": f"「{SHORT['c20']}」分成 24 份：目前是持有狀態的",
         "value": f"{_of24(long20)}（等於整體持有 {long20 / 24 * 100:.0f}%）"},
        {"label": "「17 條混合」分成 24 份：目前平均持有比例",
         "value": f"{blend_now.mean():.0f}%（24 個時鐘各自在 {blend_now.min():.0f}% ～ {blend_now.max():.0f}% 之間）"},
        {"label": f"只看{_tw(0)}這一個時鐘", "value": ("20 日新高規則：" + ("持有" if r["now"]["c20"][0] > 0 else "現金")
                                                + f"；17 條混合：持有 {r['now']['blend'][0] * 100:.0f}%")},
        {"label": "這幾格的意思", "value": "只是描述規則在各個時鐘下目前停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    keep = [k for k in RULES if across[k]["beat"] >= MOST]
    lost = [k for k in RULES if across[k]["beat"] < MOST]
    every_shallow = all(across[k]["shallow"] == 24 for k in RULES)
    few_shallow = min(across[k]["shallow"] for k in RULES)

    def tag(k: str) -> str:
        a = across[k]
        return (f"「{SHORT[k]}」（24 個時鐘每年 {_pct(a['ret'].min())} 到 {_pct(a['ret'].max())}，{_of24(a['beat'])}贏過一直抱著，"
                f"時鐘之間的標準差 {a['std']:.1f} 個百分點、平均比一直抱著{'多' if a['mean'] >= hold['ret'] else '少'} "
                f"{abs(a['mean'] - hold['ret']):.1f}）")

    loss = (f"九種規則的最深虧損則是 24 個時鐘全部比一直抱著的 {_pct(hold['deepest'])} 淺" if every_shallow
            else f"最深虧損方面，九種規則各自在 24 個時鐘裡至少 {few_shallow} 個比一直抱著的 {_pct(hold['deepest'])} 淺")
    head = f"比特幣 {start[:4]} 年到 {_d(last)}（一直抱著每年 {_pct(hold['ret'])}），把日線的收盤時刻換成 24 個整點各跑一次："
    if lost and keep:
        low = min(across[k]["beat"] for k in keep)
        part1 = (head + f"時鐘問題集中在 {len(lost)} 種規則——" + "、".join(tag(k) for k in lost)
                 + f"，它們的報酬贏不贏要看時鐘，只剩「虧得比較淺」站得住；其餘 {len(keep)} 種（"
                 + "、".join(SHORT[k] for k in keep) + f"）在 24 個時鐘裡"
                 + ("全部" if low == 24 else f"至少 {low} 個") + f"報酬都贏過一直抱著，{loss}。")
    elif lost:
        part1 = (head + f"九種規則沒有一種能在至少 {MOST} 個時鐘贏過一直抱著的報酬（"
                 + "、".join(f"{SHORT[k]} {across[k]['beat']} 個" for k in lost) + f"），{loss}。")
    else:
        low = min(across[k]["beat"] for k in RULES)
        part1 = (head + "九種規則在 24 個時鐘裡" + ("全部" if low == 24 else f"至少 {low} 個")
                 + f"報酬都贏過一直抱著，{loss}。")
    s20, sbl = stag["c20"][24], stag["blend"][24]
    stag_beat = [k for k in RULES if stag[k][24]["ret"] > hold["ret"]]
    stag_rets = [stag[k][24]["ret"] for k in RULES]
    stag_deeps = [stag[k][24]["deepest"] for k in RULES]
    won = [sum(1 for y in years if r["periods"][k]["years"][y][0] > hp["years"][y][0]) for k in RULES]
    part2 = (f"把錢平分到 24 個時鐘、拿掉時鐘運氣之後，20 日新高規則是每年 {_pct(s20['ret'])}、報酬÷波動 {_num(s20['ratio'])}、"
             f"最深虧 {_pct(s20['deepest'])}（只用{_tw(0)}一個時鐘是 {_pct(base0['ret'])}、{_pct(base0['deepest'])}），"
             f"17 條混合是 {_pct(sbl['ret'])}、{_num(sbl['ratio'])}、{_pct(sbl['deepest'])}，一直抱著是 {_pct(hold['ret'])}、"
             f"{_num(hold['ratio'])}、{_pct(hold['deepest'])}；九種規則的分散版每年 {_pct(min(stag_rets))} 到 {_pct(max(stag_rets))}"
             f"（{len(stag_beat)} 種高於一直抱著）、最深虧 {_pct(min(stag_deeps))} 到 {_pct(max(stag_deeps))}，"
             f"但逐年看 {n_years} 年裡只有 {min(won)} 到 {max(won)} 年報酬贏過一直抱著，多出來的報酬集中在少數幾年。")
    if r["coins"]:
        def counts(k: str, field: str) -> str:
            return "、".join(f"{coin} {c['across'][k][field]} 個" for coin, c in r["coins"].items())

        versus = "、".join(f"{coin} {_pct(c['stag']['c20'][24]['ret'])} 對 {_pct(c['hold']['ret'])}"
                           for coin, c in r["coins"].items())
        part3 = (f"其他幣（起算日各不相同）24 個時鐘裡報酬贏過各自一直抱著的：20 日新高規則是 {counts('c20', 'beat')}，"
                 f"17 條混合是 {counts('blend', 'beat')}；最深虧損比較淺的：20 日新高規則是 {counts('c20', 'shallow')}；"
                 f"20 日新高規則分散版對一直抱著的每年報酬是 {versus}。")
    else:
        part3 = "其他幣的 1 小時線這次拿不到，沒有比較。"
    verdict = part1 + part2 + part3

    # ----- notes -----
    def offs(k: str, parts: int) -> str:
        lo, hi = stag[k][parts]["range"]
        return f"{_pct(lo)} ～ {_pct(hi)}"

    reb = r["rebalanced"]
    notes = [
        "做法：用幣安 1 小時線把日線重新拼 24 次，一天分別結束在每一個整點（UTC 0 點＝台灣早上 8 點，依此類推）。每個時鐘各自用自己的"
        "日收盤算訊號，在那個收盤時刻決定、從那一刻起持有 24 小時；每次買賣扣交易金額的 0.10%，持有比例不是 0 或 100% 的規則每天"
        "收盤調回目標比例、這些小額調整也算成本；現金沒有利息；一直抱著沒有扣成本。不管哪個時鐘，損益都用同一種切法"
        "（台灣早上 8 點到隔天早上 8 點）來算，才能互相比、也才能和一直抱著比。",
        "九種規則：20 日新高進場／10 日新低出場；55 日新高進場／20 日新低出場；收盤高於 50、100、200 日均線才持有；過去 60 天是漲的才持有；"
        "17 條混合＝和「不挑參數，把很多順勢規則混在一起」那一頁完全相同的 17 條規則，看多的占幾成就持有幾成；"
        "混合×波動調整＝再乘上「40% ÷ 最近 30 天的波動」（最多乘 1）；九條新高規則＋移動停損＝論文〈Catching Crypto Trends〉的九個天數"
        "（5 到 360 日），創新高進場、跌破只會往上移的通道中線停損，九條平均，這裡不做波動調整。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「報酬÷波動」是每年報酬除以每天漲跌換算成一年的起伏；"
        "「最深虧損」是資金從最高點往下掉最多的幅度（用每天台灣早上 8 點的資金算，盤中會更深）。",
        (f"1 小時線從 2017-08 才有，所以結果只能從 {start} 算起，更早的年代沒辦法換時鐘。比特幣為了讓 200 日均線、360 日通道這類"
         "長天數的規則在 2018 年初就有值，在每個時鐘的序列前面接上儀表板原本的日線（2017-08 以前，UTC 0 點收盤）當暖機；"
         "2018 年頭幾個月長天數規則的訊號還帶著這段「不是那個時鐘」的舊資料，影響很小但不是零。其他幣沒有更早的資料，"
         f"要等上架滿 {WARM} 天才開始算。" if r["warm"] else
         f"1 小時線從 2017-08 才有，這次也拿不到更早的日線來暖機，所以 200 日均線、17 條混合、360 日通道這些長天數的規則在 "
         f"{start[:4]} 年初還沒有值、先當成現金，最早幾個月的結果因此偏向這些規則（剛好躲過 2018 年初的下跌），要保守看。"),
        f"時鐘運氣有多大：以 20 日新高規則為例，24 個時鐘的每年報酬標準差是 {a20['std']:.1f} 個百分點，而 24 個時鐘平均只比一直抱著多 "
        f"{_pp(a20['mean'] - hold['ret'])} 個百分點；17 條混合的標準差是 {across['blend']['std']:.1f}、平均多 "
        f"{_pp(across['blend']['mean'] - hold['ret'])}。混合類規則的時鐘標準差比較小，是因為它本來就把很多條規則平均掉了。",
        f"分散版的細節：{start} 當天把資金分成等份，各份之後不互相調撥，所以賺得多的那幾份後來占的比重會變大；"
        f"如果改成每天把各份調回一樣大（不另外算調撥成本），20 日新高規則是每年 {_pct(reb['c20']['ret'])}、最深虧 {_pct(reb['c20']['deepest'])}，"
        f"17 條混合是 {_pct(reb['blend']['ret'])}、{_pct(reb['blend']['deepest'])}。分成 4 份或 2 份時還是有「選哪幾個時鐘」的問題："
        f"20 日新高規則分 4 份的 6 種選法每年報酬 {offs('c20', 4)}，分 2 份的 12 種選法 {offs('c20', 2)}；"
        f"17 條混合分別是 {offs('blend', 4)} 和 {offs('blend', 2)}。表裡列的是包含台灣早上 8 點的那一種。",
        "24 個時鐘不是 24 次獨立的檢驗：它們用的是同一段價格，只差幾個小時，結果高度相關。「24 個時鐘裡幾個贏」只能說明結果對收盤時刻"
        "有多敏感，不能當成 24 次獨立的證據；相鄰時鐘之間的差距主要來自少數幾筆關鍵交易的進出剛好落在哪一天。",
        f"年數很少：只有 {n_years} 個日曆年{'（最後一年還沒過完，也算一年）' if partial else ''}，大致就是兩三輪大漲大跌，"
        "而且各年不是彼此獨立的；「幾年裡幾年」多一年少一年就會差很多，兩年一段的分段數字更是只有一兩輪行情。",
        "只用一家交易所（幣安現貨）的價格；其他交易所同一時刻的價格會有些微差異，早年差異更大。"
        "成本很粗略：一律扣交易金額的 0.10%，成交價用整點的收盤價、沒有另外算滑價；實際的手續費、價差會隨交易所、金額、年代和幣種不同，"
        "小幣和早年會貴得多。現金假設沒有利息，也沒有算稅。",
        f"「訊號看 A、成交在 B」的測試一定會多出等待的時間（最多 23 小時），所以它同時包含了「換成交價格」和「晚一點才做」兩件事；"
        f"576 種組合的拆解只是粗略的算術，用來看差異主要跟著哪一邊走。比特幣的 1 小時線整段缺 {r['missing']:,} 個小時"
        "（幣安早年停機維護），缺的小時沿用前一個小時的價格。",
        "這一頁的檢查是事後、已經知道這段歷史之後才設計的：要比哪些規則、分幾份、怎麼拆解，都是這次才決定的，不是事先登記好的。"
        "其他幣的名單（ETH、SOL、BNB、XRP）是今天還在的大幣，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則在不同收盤時刻下過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    check = _daily_check(r)
    if check:
        notes.insert(3, check)
    if r["problem"]:
        notes.append(r["problem"])
    notes += r["coin_problems"]

    return {
        "id": "clock_robustness",
        "title": "換一個收盤時刻，順勢規則還贏嗎？",
        "question": ("「創 20 日新高進場、10 日新低出場」的報酬優勢，換個收盤時刻就不見了一大半。這是那一條規則的問題，"
                     "還是儀表板用到的所有順勢規則、連同其他幣，都只是剛好選對了時鐘？把錢分散到 24 個時鐘之後，真正剩下多少？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安公開行情 API（K 線）",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "blockchain.info 每日比特幣價格（只用來暖機，2017-08 以前）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
