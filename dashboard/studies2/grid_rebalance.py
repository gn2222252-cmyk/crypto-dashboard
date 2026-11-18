"""Historical study "grid_rebalance": does grid trading or rebalancing between bitcoin and cash earn something extra
from bitcoin's swings - and what happens in big rallies and crashes?

These rules do not predict direction. They buy lower and sell higher around a level:

1. Constant mix: X% bitcoin (25 / 50 / 75), the rest cash, put back to X% monthly, weekly, or whenever the bitcoin
   share has drifted 5 / 10 / 20 points away (checked every hour or every UTC midnight). The rebalancing premium is
   measured against buy-and-hold of the SAME average exposure (X% bitcoin bought once, never rebalanced).
2. Exchange-style spot grids: N grid cells (10 / 20 / 50) with equally spaced lines from the start price -r to +r
   (r = 30% / 50%), the same quote amount per cell. A cell buys at its lower line when the price comes down to it and
   sells at its upper line when the price comes up to it; at the start every cell above the price (and the cell the
   price is in) is bought at market, as the exchange bots do. Fixed range, re-centred every quarter, or "trailing":
   re-centred whenever the price leaves the range.
3. Constant mix with a trend filter: 50% bitcoin (10-point band, daily check) only while bitcoin's daily close is
   above its 50-day average, all cash otherwise.

Prices: Binance spot 1-hour closes (only closed candles, finished UTC days). Fills are evaluated on hourly closes:
a grid order fills at its line price when an hourly close is at or beyond the line (moves inside an hour that come
back before the close are missed). Every fill pays the fee on the amount traded (0.10%; 0.05% / 0.02% as maker-fee
cases), also the first purchase. Cash earns nothing; a second version adds the 3-month US T-bill rate
(robust_trend._tbill) to the cash share of every day. Results are read at UTC midnights.

Start dates: the first day of every month 2018-01 ... 2024-12 (later for coins listed later), each run to the last
finished day and over fixed one-year windows. Bitcoin's hourly store is the one seasonality.py keeps and the other
coins' are the ones clock_robustness keeps; both are only read (topped up in memory). Only when neither exists does this
module keep its own file (grid_rebalance_hourly_<SYMBOL>.csv). Only public key-less endpoints through the dashboard's
helpers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import trend_ensemble as te

YEAR = 365
FEE = 0.001
FEES = (0.0002, 0.0005)
FIRST, LAST = "2018-01-01", "2024-12-01"
WARM_DAYS = 60
MA = 50
COINS = (("BTCUSDT", "比特幣"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
NOW_DAYS = 30

# constant mix: (key, weight, check, band); check "month"/"week" = rebalance every time, "hour"/"day" = only on drift
CM = (("m50", 0.50, "month", 0.0), ("w50", 0.50, "week", 0.0),
      ("h50_5", 0.50, "hour", 0.05), ("h50_10", 0.50, "hour", 0.10), ("h50_20", 0.50, "hour", 0.20),
      ("d50_5", 0.50, "day", 0.05), ("d50_10", 0.50, "day", 0.10), ("d50_20", 0.50, "day", 0.20),
      ("m25", 0.25, "month", 0.0), ("h25_10", 0.25, "hour", 0.10),
      ("m75", 0.75, "month", 0.0), ("h75_10", 0.75, "hour", 0.10))
TF = ("tf50", 0.50, "day", 0.10)
# grids: (key, half-width, cells, mode)
GRIDS = tuple((f"g{int(r * 100)}_{n}_{mode}", r, n, mode)
              for mode in ("fixed", "quarter") for r in (0.30, 0.50) for n in (10, 20, 50)) + \
    (("g30_20_trail", 0.30, 20, "trail"), ("g50_20_trail", 0.50, 20, "trail"))
DENSE = "g30_50_quarter"  # the grid that trades most: used for the cost cases
CHECK = {"month": "每月", "week": "每週", "hour": "每小時檢查", "day": "每天檢查"}
MODE = {"fixed": "區間固定", "quarter": "每季重設區間", "trail": "跌破或突破就重設"}

_pct, _pp, _num, _d = cr._pct, cr._pp, cr._num, cr._d


def _cm_name(key: str, coin: str = "比特幣") -> str:
    spec = next(x for x in CM + (TF,) if x[0] == key)
    _, w, check, band = spec
    if key == "tf50":
        return "50/50＋趨勢濾網（高於 50 日均線才持有，偏離 10 點就調）"
    head = f"{int(w * 100)}% {coin}"
    return f"{head}・{CHECK[check]}調回" if band == 0 else f"{head}・{CHECK[check]}，偏離 {int(band * 100)} 點才調"


def _grid_name(key: str) -> str:
    _, r, n, mode = next(x for x in GRIDS if x[0] == key)
    return f"±{int(r * 100)}%・{n} 格・{MODE[mode]}"


def _name(key: str, coin: str = "比特幣") -> str:
    return _grid_name(key) if key.startswith("g") else _cm_name(key, coin)


def _frac(k: int, n: int) -> str:
    return f"{k}／{n}" if n else "—"


# ---------- data ----------

def _own_store(symbol: str):
    return common.STORE / f"grid_rebalance_hourly_{symbol}.csv"


def _hourly(symbol: str) -> tuple[dict, str | None]:
    """Closed 1-hour closes {open ms: close}. Reads the stores other modules keep (never writes them); tops up in
    memory. Keeps its own file only when no other store exists."""
    paths = ([cr.SHARED_STORE] if symbol == "BTCUSDT" else []) + [cr._own_store(symbol), _own_store(symbol)]
    rows, src = {}, None
    for path in paths:
        got = cr._read(path)
        if len(got) > len(rows):
            rows, src = got, path
    problem = None
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    if not rows or max(rows) < today_ms - cr.HOUR_MS:
        try:
            added = cr._fetch(symbol, rows, keep=False)
            if added and (src is None or src == _own_store(symbol)):
                try:
                    common.STORE.mkdir(parents=True, exist_ok=True)
                    text = "ms,close\n" + "\n".join(f"{ms},{c!r}" for ms, c in sorted(rows.items())) + "\n"
                    _own_store(symbol).write_text(text, encoding="utf-8")
                except Exception:  # noqa: BLE001 - next run fetches again
                    pass
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = (f"{symbol[:-4]} 的幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    return rows, problem


class _Coin:
    """Hourly price grid of one coin plus the daily things every run needs."""

    def __init__(self, price: np.ndarray, b0: pd.Timestamp, rate: pd.Series | None):
        self.P = np.asarray(price, dtype=float)
        self.b0 = b0
        self.days = (len(self.P) - 1) // 24
        self.dates = pd.date_range(b0, periods=self.days + 1, freq="D")  # boundary j = hour 24j
        self.D = self.P[::24][: self.days + 1]
        self.change = self.D[1:] / self.D[:-1] - 1  # return of day j
        self.month = np.asarray(self.dates.day == 1)
        self.week = np.asarray(self.dates.weekday == 0)
        self.quarter = self.month & np.isin(self.dates.month, (1, 4, 7, 10))
        self.quarter_days = np.flatnonzero(self.quarter)
        mean = pd.Series(self.D).rolling(MA, min_periods=MA).mean().values
        self.sig = np.where(np.isfinite(mean) & (self.D > np.nan_to_num(mean, nan=np.inf)), 1.0, 0.0)  # close j only
        if rate is not None:
            d = self.dates[: self.days]
            self.rf = rate.reindex(rate.index.union(d)).ffill().reindex(d).fillna(0.0).values / 100 / YEAR
        else:
            self.rf = np.zeros(self.days)

    def starts(self, first: str = FIRST, last: str = LAST) -> list[int]:
        begin = max(pd.Timestamp(first), self.dates[0] + pd.Timedelta(days=WARM_DAYS))
        out = []
        for month in pd.date_range(begin.to_period("M").to_timestamp(), pd.Timestamp(last), freq="MS"):
            if month < begin:
                continue
            j = int((month - self.b0).days)
            if 0 <= j < self.days - YEAR:
                out.append(j)
        return out


def _first(mask, lo: int, hi: int) -> int:
    """First index in [lo, hi) where mask(a, b) (a boolean array for indices a..b-1) is true, else hi."""
    size = 256
    while lo < hi:
        b = min(hi, lo + size)
        m = mask(lo, b)
        if m.any():
            return lo + int(np.argmax(m))
        lo, size = b, size * 4
    return hi


# ---------- engines ----------

def _finish(c: _Coin, sd: int, ed: int, ev_h: list, ev_q: list, ev_c: list, fills: int, drag: float,
            extra: dict | None = None) -> dict:
    """Equity at the UTC midnights sd..ed from the list of holdings changes, and the summary of the run."""
    hb = 24 * np.arange(sd, ed + 1)
    idx = np.searchsorted(np.asarray(ev_h), hb, side="right") - 1
    q, cash = np.asarray(ev_q)[idx], np.asarray(ev_c)[idx]
    equity = cash + q * c.P[hb]
    share = q * c.P[hb] / equity
    return _summary(c, sd, ed, equity, share, fills, drag, extra)


def _summary(c: _Coin, sd: int, ed: int, equity: np.ndarray, share: np.ndarray, fills: int, drag: float,
             extra: dict | None = None) -> dict:
    full = equity.copy()
    full[0] = 1.0  # the first day's return carries the fee of the first purchase
    net = full[1:] / full[:-1] - 1
    net_rf = net + (1 - share[:-1]) * c.rf[sd:ed]
    years = (ed - sd) / YEAR
    st, st_rf = cr._stats(net), cr._stats(net_rf)
    out = {"ret": st["ret"], "deep": st["deepest"], "ratio": st["ratio"], "ret_rf": st_rf["ret"],
           "total": float((np.prod(1 + net) - 1) * 100), "share": float(share[:-1].mean()),
           "end_share": float(min(1.0, max(0.0, share[-1]))), "fills": fills / years, "count": fills,
           "drag": drag / years * 100}
    out.update(extra or {})
    return out


def _cm(c: _Coin, sd: int, ed: int, w: float, check: str, band: float, fee: float, filt: bool = False) -> dict:
    """Constant mix. Periodic checks rebalance every time; drift checks only when the share leaves w ± band
    (or the trend filter's target changes)."""
    hs, he = 24 * sd, 24 * ed
    if check in ("month", "week"):
        flag = c.month if check == "month" else c.week
        checks = 24 * np.unique(np.concatenate([[sd], sd + np.flatnonzero(flag[sd:ed])]))
    elif check == "hour":
        checks = np.arange(hs, he)
    else:
        checks = 24 * np.arange(sd, ed)
    X = c.P[checks]
    W = (w * c.sig[checks // 24]) if filt else np.full(len(checks), w)
    ev_h, ev_q, ev_c = [hs], [0.0], [1.0]
    q, cash, fills, drag = 0.0, 1.0, 0, 0.0
    n, k = len(checks), 0
    periodic = band == 0
    while k < n:
        price, wt = float(X[k]), float(W[k])
        equity = cash + q * price
        trade = wt * equity - q * price
        if abs(trade) > 1e-9 * equity:
            f = fee * abs(trade)
            q = wt * equity / price
            cash = equity - q * price - f
            fills += 1
            drag += f / equity
            ev_h.append(int(checks[k]))
            ev_q.append(q)
            ev_c.append(cash)
        if periodic:
            k += 1
            continue
        up = (wt + band) * (1 - wt) / (wt * (1 - wt - band)) if 0 < wt and wt + band < 1 else math.inf
        dn = (wt - band) * (1 - wt) / (wt * (1 - wt + band)) if 0 < wt < 1 and wt - band > 0 else 0.0
        anchor = price  # the share is exactly wt at this price (just traded to it, or already there)

        def mask(a, b, anchor=anchor, wt=wt, up=up, dn=dn):
            r = X[a:b] / anchor
            return (r >= up) | (r <= dn) | (W[a:b] != wt)

        k = _first(mask, k + 1, n)
    return _finish(c, sd, ed, ev_h, ev_q, ev_c, fills, drag)


def _grid(c: _Coin, sd: int, ed: int, r: float, cells: int, mode: str, fee: float) -> dict:
    """Exchange-style spot grid (see the module text)."""
    P, N = c.P, cells
    hs, he = 24 * sd, 24 * ed
    ev_h, ev_q, ev_c = [hs], [0.0], [1.0]
    q, cash, fills, drag, outside, recentres = 0.0, 1.0, 0, 0.0, 0, 0
    a = hs
    while a < he:
        centre = float(P[a])
        lines = np.linspace(centre * (1 - r), centre * (1 + r), N + 1)
        lo_line, hi_line = lines[0], lines[-1]
        if mode == "quarter":
            later = c.quarter_days[c.quarter_days > a // 24]
            b = min(he, 24 * int(later[0])) if len(later) else he
        elif mode == "trail":
            b = _first(lambda x, y: (P[x:y] > hi_line) | (P[x:y] < lo_line), a + 1, he + 1)
            b = min(b, he)
        else:
            b = he
        lower = lines[:-1]
        inv = 1 / lower
        cs_inv = np.concatenate([[0.0], np.cumsum(inv)])
        cs_up = np.concatenate([[0.0], np.cumsum(lines[1:] * inv)])
        # starting position of this range: every cell above the price, and the one it is in, holds coin
        equity = cash + q * centre
        C = int(N - np.searchsorted(lines[1:], centre, side="right"))
        S = float(cs_inv[N] - cs_inv[N - C])
        Q = equity / ((N - C) + centre * S)
        target = Q * S
        f = fee * abs(target - q) * centre
        if abs(target - q) * centre > 1e-9 * equity:
            fills += 1
            drag += f / equity
        q, cash = target, equity - target * centre - f
        ev_h.append(a)
        ev_q.append(q)
        ev_c.append(cash)
        seg = P[a + 1:b + 1]
        A = N - np.searchsorted(lower, seg, side="left")  # cells that must hold coin (price at or below their line)
        B = N - np.searchsorted(lines[1:], seg, side="right")  # cells that may still hold coin
        outside += int(((seg > hi_line) | (seg < lo_line)).sum())
        changed = np.flatnonzero((np.diff(A, prepend=-1) != 0) | (np.diff(B, prepend=-1) != 0))
        for i, lo_c, hi_c in zip(changed.tolist(), A[changed].tolist(), B[changed].tolist()):
            if C < lo_c:
                new = lo_c
            elif C > hi_c:
                new = hi_c
            else:
                continue
            if new > C:  # buy cells N-new .. N-C-1 at their lower lines
                spent = Q * (new - C)
                q += Q * (cs_inv[N - C] - cs_inv[N - new])
                cash -= spent
                f = fee * spent
            else:  # sell cells N-C .. N-new-1 at their upper lines
                got = Q * (cs_up[N - new] - cs_up[N - C])
                q -= Q * (cs_inv[N - new] - cs_inv[N - C])
                cash += got
                f = fee * got
            cash -= f
            drag += f / (cash + q * float(seg[i]))
            fills += abs(new - C)
            C = new
            ev_h.append(a + 1 + i)
            ev_q.append(q)
            ev_c.append(cash)
        if b < he:
            recentres += 1
        a = b
    return _finish(c, sd, ed, ev_h, ev_q, ev_c, fills, drag,
                   {"outside": outside / max(1, he - hs), "recentres": recentres})


def _hold(c: _Coin, sd: int, ed: int, w: float, fee: float) -> dict:
    """w bitcoin bought once at sd, never touched."""
    rel = c.D[sd:ed + 1] / c.D[sd]
    coin = w * (1 - fee) * rel
    equity = (1 - w) + coin
    return _summary(c, sd, ed, equity, coin / equity, 1, w * fee)


def _ma50(c: _Coin, sd: int, ed: int, fee: float) -> dict:
    pos = c.sig[sd:ed]
    net, trade = te._run(pos, c.change[sd:ed], fee)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    share = np.concatenate([pos, [pos[-1]]])
    return _summary(c, sd, ed, equity, share, int((trade > 1e-9).sum()), float(fee * trade.sum()))


def _run(c: _Coin, key: str, sd: int, ed: int, fee: float) -> dict:
    if key == "tf50":
        return _cm(c, sd, ed, 0.5, "day", 0.10, fee, filt=True)
    if key.startswith("g"):
        _, r, n, mode = next(x for x in GRIDS if x[0] == key)
        return _grid(c, sd, ed, r, n, mode, fee)
    _, w, check, band = next(x for x in CM if x[0] == key)
    return _cm(c, sd, ed, w, check, band, fee)


# ---------- across start dates ----------

def _bench(c: _Coin, starts: list[int], fee: float) -> dict:
    out = {}
    for span in ("now", "one"):
        rows = []
        for sd in starts:
            ed = c.days if span == "now" else sd + YEAR
            rows.append({"hold": _hold(c, sd, ed, 1.0, fee), "ma50": _ma50(c, sd, ed, fee)})
        out[span] = rows
    return out


def _sweep(c: _Coin, key: str, starts: list[int], bench: dict, fee: float) -> dict:
    """One rule from every start date: to the last day and over one-year windows, with the same-exposure hold."""
    res = {}
    for span in ("now", "one"):
        rows = []
        for i, sd in enumerate(starts):
            ed = c.days if span == "now" else sd + YEAR
            x = _run(c, key, sd, ed, fee)
            same = _hold(c, sd, ed, x["share"], fee)
            b = bench[span][i]
            x.update(prem=x["ret"] - same["ret"], prem_rf=x["ret_rf"] - same["ret_rf"],
                     beat_hold=x["ret"] > b["hold"]["ret"], beat_ma=x["ret"] > b["ma50"]["ret"],
                     beat_same=x["ret"] > same["ret"])
            rows.append(x)
        res[span] = rows
    now, one = res["now"], res["one"]

    def med(rows, field):
        return float(np.median([x[field] for x in rows]))

    return {"n": len(now), "ret": med(now, "ret"), "deep": med(now, "deep"), "ratio": med(now, "ratio"),
            "prem": med(now, "prem"), "prem_rf": med(now, "prem_rf"), "win": sum(x["beat_same"] for x in now),
            "beat_hold": sum(x["beat_hold"] for x in now), "beat_ma": sum(x["beat_ma"] for x in now),
            "prem1": med(one, "prem"), "win1": sum(x["beat_same"] for x in one), "n1": len(one),
            "fills": med(now, "fills"), "drag": med(now, "drag"), "share": med(now, "share"),
            "outside": med(now, "outside") if "outside" in now[0] else None,
            "first": now[0], "rows": now}


def _bench_summary(bench: dict, field: str) -> dict:
    now = [b[field] for b in bench["now"]]
    return {"ret": float(np.median([x["ret"] for x in now])), "deep": float(np.median([x["deep"] for x in now])),
            "ratio": float(np.median([x["ratio"] for x in now])), "fills": float(np.median([x["fills"] for x in now])),
            "drag": float(np.median([x["drag"] for x in now]))}


# ---------- episodes ----------

def _episodes(c: _Coin) -> list[tuple[str, int, int]]:
    """Big rallies (from the low to the later high) and crashes (from the high to the later low) inside fixed
    calendar windows; the turning points are read from the closes afterwards, only to describe what happened."""
    out = []
    spec = (("2018 大跌", "2018-01-01", "2018-12-31", "down"), ("2020–21 大漲", "2020-01-01", "2021-12-31", "up"),
            ("2022 大跌", "2021-11-01", "2022-12-31", "down"), ("2023–24 大漲", "2023-01-01", "2024-12-31", "up"),
            ("2024 年中盤整", "2024-03-14", "2024-10-31", "flat"), ("2025–26 大跌", "2025-01-01", None, "down"))
    for name, a, b, kind in spec:
        i = int((pd.Timestamp(a) - c.b0).days)
        j = c.days if b is None else min(c.days, int((pd.Timestamp(b) - c.b0).days))
        if i < 0 or j - i < 60:
            continue
        part = c.D[i:j + 1]
        if kind == "flat":
            s, e = i, j
        elif kind == "up":
            s = i + int(np.argmin(part))
            e = s + int(np.argmax(c.D[s:j + 1]))
        else:
            s = i + int(np.argmax(part))
            e = s + int(np.argmin(c.D[s:j + 1]))
        if e - s >= 30:
            out.append((name, s, e))
    return out


# ---------- the whole computation ----------

def _load(symbol: str, rate) -> tuple[_Coin, str | None]:
    rows, problem = _hourly(symbol)
    if len(rows) < 24 * (WARM_DAYS + 2 * YEAR):
        raise RuntimeError(f"{symbol} 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    return _Coin(price, b0, rate), problem


def _compute(btc: _Coin, others: dict) -> dict:
    starts = btc.starts()
    bench = _bench(btc, starts, FEE)
    keys = [x[0] for x in CM] + ["tf50"] + [x[0] for x in GRIDS]
    res = {k: _sweep(btc, k, starts, bench, FEE) for k in keys}
    pure = [x[0] for x in CM]
    best_cm = max(pure, key=lambda k: res[k]["prem"])
    best_grid = max((x[0] for x in GRIDS), key=lambda k: res[k]["prem"])
    fees = {}
    for fee in FEES:
        b = _bench(btc, starts, fee)
        fees[fee] = {k: _sweep(btc, k, starts, b, fee) for k in (best_cm, best_grid, DENSE)}
    coin_res = {}
    for name, c in others.items():
        cs = c.starts()
        if len(cs) < 6:
            continue
        b = _bench(c, cs, FEE)
        coin_res[name] = {"start": c.dates[cs[0]], "hold": _bench_summary(b, "hold"),
                          "sweep": {k: _sweep(c, k, cs, b, FEE) for k in (best_cm, best_grid)}}
    show = ("hold", "half", "m50", "h50_10", "g30_20_fixed", "g30_20_quarter", "g30_20_trail")
    episodes = []
    for name, s, e in _episodes(btc):
        row = {}
        for k in show:
            if k == "hold":
                row[k] = _hold(btc, s, e, 1.0, FEE)
            elif k == "half":
                row[k] = _hold(btc, s, e, 0.5, FEE)
            else:
                row[k] = _run(btc, k, s, e, FEE)
        episodes.append((name, s, e, row))
    sd = btc.days - NOW_DAYS
    today = {"cm": _cm(btc, sd, btc.days, 0.5, "hour", 0.10, FEE), "grid": _grid(btc, sd, btc.days, 0.30, 20, "fixed", FEE),
             "hold": _hold(btc, sd, btc.days, 1.0, FEE), "start_price": float(btc.D[sd]), "start": btc.dates[sd]}
    return {"starts": starts, "bench": bench, "res": res, "best_cm": best_cm, "best_grid": best_grid, "fees": fees,
            "coins": coin_res, "episodes": episodes, "show": show, "today": today}


# ---------- the study ----------

def study() -> dict:
    from dashboard.studies2 import robust_trend as rt

    rate, rate_source, rate_problem = rt._tbill()
    problems = [rate_problem] if rate_problem else []
    btc, problem = _load("BTCUSDT", rate)
    if problem:
        problems.append(problem)
    others = {}
    for symbol, name in COINS[1:]:
        try:
            c, problem = _load(symbol, rate)
            if problem:
                problems.append(problem)
            others[name] = c
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{name} 的 1 小時線這次拿不到或太短（{type(exc).__name__}），沒有算進去。")
    r = _compute(btc, others)
    res, bench = r["res"], r["bench"]
    last = btc.dates[btc.days]
    n = len(r["starts"])
    hold = _bench_summary(bench, "hold")
    ma = _bench_summary(bench, "ma50")
    half_now = [_hold(btc, sd, btc.days, 0.5, FEE) for sd in r["starts"]]
    half = {f: float(np.median([x[f] for x in half_now])) for f in ("ret", "deep", "ratio")}
    first_lbl, last_lbl = _d(btc.dates[r["starts"][0]])[:7], _d(btc.dates[r["starts"][-1]])[:7]

    def cells(x: dict, grid: bool) -> list:
        return [f"{_pct(x['ret'])}｜{_pct(x['deep'])}", _num(x["ratio"]),
                f"{_pp(x['prem'])}｜{_frac(x['win'], x['n'])}",
                f"{_frac(x['beat_hold'], x['n'])}｜{_frac(x['beat_ma'], x['n'])}",
                f"{_pp(x['prem1'])}｜{_frac(x['win1'], x['n1'])}",
                (f"{x['share'] * 100:.0f}%｜{x['outside'] * 100:.0f}%" if grid else _pp(x["prem_rf"])),
                f"{x['fills']:.0f} 次｜{x['drag']:.2f}%"]

    def ref(name: str, x: dict, fills: str = "—") -> list:
        return [name, f"{_pct(x['ret'])}｜{_pct(x['deep'])}", _num(x["ratio"]), "—", "—", "—", "—", fills]

    # ----- table 1: constant mix -----
    rows1 = [[_cm_name(k)] + cells(res[k], False) for k in [x[0] for x in CM] + ["tf50"]]
    rows1.append(ref("對照：100% 一直抱著", hold))
    rows1.append(ref("對照：50% 比特幣買一次就不動", half))
    rows1.append(ref("對照：收盤高於 50 日均線才全部持有", ma, f"{ma['fills']:.0f} 次｜{ma['drag']:.2f}%"))

    # ----- table 2: grids -----
    rows2 = [[_grid_name(x[0])] + cells(res[x[0]], True) for x in GRIDS]
    rows2.append(ref("對照：100% 一直抱著", hold))
    rows2.append(ref("對照：50% 比特幣買一次就不動", half))

    # ----- table 3: episodes -----
    rows3 = []
    for name, s, e, row in r["episodes"]:
        line = [f"{name}（{_d(btc.dates[s])} → {_d(btc.dates[e])}）"]
        for k in r["show"]:
            x = row[k]
            cell = f"{_pct(x['total'])}｜{_pct(x['deep'])}"
            if k.startswith("g"):
                cell += f"｜結束持幣 {x['end_share'] * 100:.0f}%"
            line.append(cell)
        rows3.append(line)

    # ----- table 4: other coins, costs, interest -----
    bc, bg = r["best_cm"], r["best_grid"]

    def row4(label: str, x: dict, h: dict) -> list:
        return [label, str(x["n"]),f"{_pct(x['ret'])}｜{_pct(x['deep'])}",
                f"{_pct(h['ret'])}｜{_pct(h['deep'])}", f"{_pp(x['prem'])}｜{_frac(x['win'], x['n'])}",
                f"{_frac(x['beat_hold'], x['n'])}｜{_frac(x['beat_ma'], x['n'])}",
                f"{_pp(x['prem1'])}｜{_frac(x['win1'], x['n1'])}", f"{x['fills']:.0f} 次｜{x['drag']:.2f}%"]

    rows4 = []
    for k in (bc, bg):
        rows4.append(row4(f"比特幣（{first_lbl} 起）：{_name(k)}", res[k], hold))
    for name, cr_ in r["coins"].items():
        for k in (bc, bg):
            rows4.append(row4(f"{name}（{_d(cr_['start'])[:7]} 起）：{_name(k, '幣')}", cr_["sweep"][k], cr_["hold"]))
    for fee, keys4 in ((0.0002, (bc, bg, DENSE)), (0.0005, (DENSE,))):
        for k in keys4:
            rows4.append(row4(f"比特幣，每筆手續費 {fee * 100:.2f}%：{_name(k)}", r["fees"][fee][k], hold))
    rows4 = rows4[:16]

    tables = [
        {"caption": (f"固定比例再平衡（比特幣，{first_lbl} 到 {last_lbl} 每個月 1 號各開始一次、共 {n} 個起點，算到 {_d(last)}；"
                     f"每筆手續費 {FEE * 100:.2f}%，現金沒有利息）。格子裡是 {n} 個起點的中位數。「比同比例一直抱著多」＝"
                     "每年報酬減掉「同樣平均持有比例的比特幣買一次就不動」的每年報酬（百分點），這就是再平衡多賺（或少賺）的部分；"
                     "「贏的起點」是幾個起點這個差距大於 0。一年期＝每個起點只算 12 個月。手續費拖累＝每年付掉的手續費占資金的比例。"),
         "columns": ["做法", "每年報酬｜最深虧損", "報酬÷波動", "比同比例一直抱著多（點）｜贏的起點",
                     "贏過 100% 一直抱著｜贏過 50 日均線", "一年期：比同比例抱著多（點）｜贏的比例",
                     "現金領國庫券利息時：比同比例抱著多（點）", "每年成交次數｜手續費拖累"],
         "rows": rows1},
        {"caption": (f"交易所式現貨網格（比特幣，同樣 {n} 個起點、每筆 {FEE * 100:.2f}%）。區間是開始那一刻價格的上下 30% 或 50%，"
                     "平均切成 10／20／50 格，每格放一樣多的錢：價格跌到某條線就買一格、漲到上一條線就把那一格賣掉；"
                     "開始時區間上半部（價格以上）的格子先用市價買好幣。用每小時收盤價判斷有沒有碰到線。"
                     "「比同平均持有量抱著多」的對照是：用這個網格整段平均持幣的比例，在開始那天買一次比特幣、之後不動。"
                     "「平均持幣｜區間外」是整段平均有幾成資金在比特幣、以及價格跑出區間外（網格停擺）的時間比例（中位數）。"),
         "columns": ["網格設定", "每年報酬｜最深虧損", "報酬÷波動", "比同平均持有量抱著多（點）｜贏的起點",
                     "贏過 100% 一直抱著｜贏過 50 日均線", "一年期：比同持有量抱著多（點）｜贏的比例",
                     "平均持幣｜價格在區間外的時間", "每年成交次數｜手續費拖累"],
         "rows": rows2},
        {"caption": ("大漲、大跌、盤整時各做法的表現（比特幣，每一段都在那段的起點重新開始；大漲段從窗口裡的最低點算到之後的最高點、"
                     "大跌段從最高點算到之後的最低點，是事後回頭找的轉折點，只用來描述發生過什麼）。"
                     "格子是「整段報酬｜段內最深虧損」，網格另外列結束時有幾成資金在比特幣（0% 就是已經全部賣光）。"
                     "50/50 每小時檢查是偏離 10 點才調；三種網格都是 ±30%、20 格。"),
         "columns": ["期間", "100% 一直抱著", "50% 買一次不動", "50/50 每月調回", "50/50 偏離 10 點才調",
                     "網格・區間固定", "網格・每季重設", "網格・跌破或突破就重設"],
         "rows": rows3},
        {"caption": (f"在比特幣上看起來最好的兩種設定（再平衡溢酬中位數最高的：「{_name(bc)}」和「{_name(bg)}」），"
                     f"原封不動套到其他幣（固定比例時「比特幣」換成那個幣），以及手續費改成掛單價 0.02%、0.05% 的情況"
                     f"（另外加上交易最多的「{_name(DENSE)}」，它在 0.10% 的結果見上面網格表）。其他幣從上架滿 60 天後的第一個月 1 號"
                     f"（不早於 {first_lbl}）開始，到 {last_lbl} 每月一個起點；各幣起算日不同，數字不能直接互比。格子都是起點的中位數。"),
         "columns": ["幣（起算）與做法", "起點數", "每年報酬｜最深虧損", "100% 一直抱著：每年報酬｜最深虧損",
                     "比同持有量抱著多（點）｜贏的起點", "贏過 100% 一直抱著｜贏過 50 日均線",
                     "一年期：比同持有量抱著多（點）｜贏的比例", "每年成交次數｜手續費拖累"],
         "rows": rows4},
    ]

    # ----- today -----
    t = r["today"]
    g, cm = t["grid"], t["cm"]
    lo, hi = t["start_price"] * 0.7, t["start_price"] * 1.3
    price_now = float(btc.D[-1])
    where = "區間內" if lo <= price_now <= hi else ("已經跌破區間下緣（網格停擺、全部是幣）" if price_now < lo
                                                  else "已經突破區間上緣（網格停擺、全部是現金）")
    now = [
        {"label": f"最後一個收完的日子（{_d(last)}）", "value": f"比特幣 {price_now:,.0f} 美元（30 天前 {t['start_price']:,.0f}，"
                                                      f"漲跌 {_pct(t['hold']['total'])}）"},
        {"label": f"30 天前（{_d(t['start'])}）開始的 50/50、偏離 10 點才調", "value":
            f"現在約 {cm['end_share'] * 100:.0f}% 在比特幣；這 30 天 {_pct(cm['total'])}，成交 {cm['count']} 次"},
        {"label": "30 天前開始的 ±30%、20 格網格", "value":
            f"區間 {lo:,.0f} ～ {hi:,.0f} 美元，價格{where}；現在約 {g['end_share'] * 100:.0f}% 在比特幣，"
            f"這 30 天 {_pct(g['total'])}，成交 {g['count']} 筆（含開始時買進的 1 筆）"},
        {"label": "這幾格的意思", "value": "只是描述這兩種機械做法如果 30 天前開始、現在會停在什麼狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    rc, rg = res[bc], res[bg]
    half_keys = [x[0] for x in CM if x[1] == 0.5]
    fixed_keys = [x[0] for x in GRIDS if x[3] == "fixed"]
    moving_keys = [x[0] for x in GRIDS if x[3] != "fixed"]

    def span(keys, field="prem") -> str:
        vals = [res[k][field] for k in keys]
        return f"{_pp(min(vals))}～{_pp(max(vals))}"

    eps = {name: row for name, _, _, row in r["episodes"]}
    ups = [eps[k] for k in eps if "漲" in k]
    downs = [eps[k] for k in eps if "跌" in k]
    up_hold = "、".join(_pct(x["hold"]["total"]) for x in ups)
    up_grid = "、".join(_pct(x["g30_20_fixed"]["total"]) for x in ups)
    down_hold = "、".join(_pct(x["hold"]["total"]) for x in downs)
    down_grid = "、".join(_pct(x["g30_20_fixed"]["total"]) for x in downs)
    c02 = r["fees"][0.0002]
    fee_gain = [c02[k]["prem"] - res[k]["prem"] for k in (bc, bg, DENSE)]
    outside = min(res[k]["outside"] for k in fixed_keys) * 100
    s1 = (f"{first_lbl}～{last_lbl} 每月起算到今天，50/50 再平衡比「50% 買了不動」每年只差 {span(half_keys)} 點，"
          f"最好的固定比例（{int(next(x[1] for x in CM if x[0] == bc) * 100)}%）也只多 {_pp(rc['prem'])} 點；"
          f"網格只有區間固定的多 {span(fixed_keys)} 點，但那是價格很快跑出區間、至少 {outside:.0f}% 的時間停擺，"
          f"每季重設或跟著移動的網格則是 {span(moving_keys)} 點。")
    s2 = (f"大漲時 ±30% 網格很快賣光，只賺 {up_grid}（比特幣 {up_hold}）；大跌時一路買滿，"
          f"跌 {down_grid}，接近全抱的 {down_hold}。")
    s3 = (f"溢酬本來就在幾點以內，手續費從 0.10% 降到 0.02% 只改變 {min(fee_gain):.1f}～{max(fee_gain):.1f} 點，"
          f"最常交易的網格每年付掉 {res[DENSE]['drag']:.1f}%。")
    verdict = s1 + s2 + s3

    rate_text = {"fred": "FRED DTB3", "treasury": "美國財政部每日國庫券利率表", "none": "（這次取不到，以 0% 計）"}[rate_source]
    notes = [
        "做法：價格用幣安現貨 1 小時收盤價；每筆成交（包括第一次買進）扣成交金額的手續費，主要用 0.10%（一般吃單），"
        "另外看 0.05%、0.02%（掛單、優惠費率）；沒有算滑價和稅。資金每天台灣早上 8 點（UTC 0 點）記一次，最深虧損用這個時刻的資金算，盤中會更深。",
        "網格只看每小時的收盤價：一小時內碰到線又彈回去的成交會漏掉，所以真實網格（尤其格子很密的）成交次數會比這裡多，"
        "每格賺的小價差和手續費都會更多；反過來，成交價這裡假設剛好在線上，實際掛單也是這樣，但市價單、跳空會有差異。",
        "交易所的網格機器人細節各不相同（幣安、Pionex、OKX 的等差／等比格子、每格放相同幣量或相同金額、啟動時怎麼買、"
        "網格利潤要不要再投入、往上追的「無限網格」怎麼移），這裡只是其中一種常見設定：等差格子、每格相同金額、利潤留在現金。"
        "「跌破或突破就重設」是把整個區間移到新價格附近重新排（兩個方向都移），和交易所只往上移的版本不完全一樣。",
        "「同樣平均持有量」的對照：用每種做法整段平均的持幣比例（再平衡大約就是目標比例，例如 50%），在開始那天買一次比特幣就不動。"
        "買了不動的那一份在大漲後占比會變大，所以兩者的平均曝險不會完全相同；這個差距就是再平衡溢酬本身的一部分，不是另外的偏誤。",
        (f"區間固定的網格要小心解讀：從 {first_lbl} 起算的起點，價格大多在一兩年內跑出區間（中位數 "
         f"{min(res[k]['outside'] for k in fixed_keys) * 100:.0f}%～{max(res[k]['outside'] for k in fixed_keys) * 100:.0f}% "
         "的時間在區間外），跑上去就全部變現金、跑下去就全部變幣，之後不再交易；它「比同持有量抱著多」主要是因為整段平均只有約 "
         f"{np.median([res[k]['share'] for k in fixed_keys]) * 100:.0f}% 的資金在幣上（中位數）、"
         "而且持幣集中在剛開始那段，不是網格的價差利潤。一年期的數字比較能看出網格在區間內的表現。"
         f"現金有國庫券利息時，最好看的網格比同持有量抱著 {_pp(rg['prem_rf'])} 點、每季重設的 ±30% 20 格 {_pp(res['g30_20_quarter']['prem_rf'])} 點。"),
        f"現金有利息的版本：沒有買比特幣的那部分資金，每天加上美國三個月國庫券利率（{rate_text}）÷365；"
        "實際放在交易所的美元或穩定幣不一定領得到這個利率。",
        f"起點：{first_lbl} 到 {last_lbl} 每個月 1 號（UTC）開始一次，各自算到 {_d(last)}，另外每個起點只算 12 個月的「一年期」。"
        "相鄰起點的時間大部分重疊，不是獨立的檢驗；「幾個起點贏」只說明結果對開始時點有多敏感。",
        "大漲大跌的次數很少：2018 年以來大致只有兩輪大漲、三輪大跌，網格和再平衡的結果主要取決於這幾段趨勢，"
        "下一輪的長短和幅度不同，結果就會不同。",
        "最好看的設定是看完比特幣的結果才挑的（挑溢酬中位數最高的），再原封不動套到其他幣；其他幣的名單是今天還在的大幣，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只描述這些機械做法過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += problems

    sources = [
        {"name": "幣安現貨 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "幣安公開行情 API（K 線）",
         "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
        {"name": "幣安現貨網格交易說明", "url": "https://www.binance.com/zh-TC/support/faq/what-is-spot-grid-trading-and-how-does-it-work-d5f441e8ab544a5b98241e00efb3a4ab"},
    ]
    if rate_source == "fred":
        sources.append({"name": "FRED：美國三個月國庫券利率（DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"})
    elif rate_source == "treasury":
        sources.append({"name": "美國財政部每日國庫券利率",
                        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates"})

    return {
        "id": "grid_rebalance",
        "title": "網格交易和固定比例再平衡，有多賺嗎？",
        "question": ("不猜方向、只在價格區間裡「跌一格買一點、漲一格賣一點」的網格交易，以及固定比特幣／現金比例、偏離就調回的再平衡，"
                     "到底有沒有從比特幣的上下震盪多賺到東西？遇到大漲、大跌時又會怎樣？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
