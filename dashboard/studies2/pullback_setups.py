"""Historical study "pullback_setups": do the best-known "buy the pullback inside an uptrend" setups - Linda Bradford
Raschke's "Holy Grail" and Larry Connors' RSI(2) among them - make money on bitcoin, and do they improve the page's
50-day-average rule?

Eleven long-or-cash setups on daily bars, parameters as published and fixed before running (no tuning). Every one
is decided at a day's close; the trade is done at that close (Connors buys "at the close"; for the others "the next
daily candle" of a coin that trades around the clock opens at that same price), and the position is held from there.
  A  Holy Grail (Raschke, Street Smarts 1995): a "touch" day has its low at or below the 20-day exponential average,
     ADX(14) above 30, ADX rising up to the day before (it may turn down on the pullback, as the book expects) and
     +DI above -DI (uptrend). Buy when a later close is above the touch day's high (the daily-close version of the
     buy stop). Exit at the first close at or above the recent swing high (highest high of the 20 days before entry)
     or below the touch day's low (the protective stop).
  A2 the same without "ADX rising" (added after A turned out to trade about once in five years; reported as such).
  B  Connors RSI(2) (Short Term Trading Strategies That Work, 2008): close above the 200-day average and RSI(2)
     (Wilder) below 10 / below 5 -> buy at the close. Exit at the first close above the 5-day average / with RSI(2)
     above 70. Four versions.
  C  Buy the dip in an uptrend: close above the 50-day average and the close at least 5% / 10% below one of the last
     three closes -> buy, sell at the close 5 / 10 days later. Four versions.
  D  Internal bar strength (close - low) / (high - low) below 0.2 with the close above the 200-day average -> buy;
     exit at the first close above the previous day's high.
  E  Pullback to the 20-day average while the 50-day average is rising: close above the rising 50-day average and the
     low at or below the 20-day average -> buy; exit at the first close below the 50-day average or after 10 days.
Each setup is used (i) alone (cash otherwise) and (ii) as an overlay on "close above the 50-day average" (ma50): the
setup may only add exposure while ma50 is out; the position is max(ma50, setup), never above 100%.

Fair grid of main_signal_check, reused: bitcoin UTC-midnight daily bars from monthly start dates 2018-01 ... 2023-12
(highs and lows exist from 2017-08); 72 start dates x 24 closing hours rebuilt from Binance 1-hour candles (a clock's
daily high / low = highest / lowest of its 24 hourly closes); the ordinary person (trade 10 hours after the close,
0.25% cost, 10% of entries missed, 200 simulations per clock); other assets ETH, SOL, BNB, XRP (Binance), MSTR (from
2020-08-11), the S&P 500 and the Taiwan weighted index (Yahoo, from 2000). Ranking score = the lowest of the four
"beats holding on yearly return" shares, exactly as main_signal_check (its other-asset check uses the same five
assets; the two stock indices are reported beside it).

Pre-stated tests. An overlay improves ma50 only if its ranking score is at least 79%, its median deepest loss on the
clock grid is no worse than ma50's and its median yearly edge over holding is higher than ma50's. A stand-alone setup
"works" if it beats holding on return / volatility in at least 2/3 of the bitcoin start x clock cells and on at least
2/3 of the other assets (an asset counts when it wins in more than half of that asset's start dates).

No look-ahead: every value uses bars up to and including the decision close (checked by changing all later prices);
random draws belong to days. Only public key-less endpoints through the dashboard's helpers; stores of other studies
are read, never written.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.collectors import signals as sig
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import retail_indicators as ri
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST = 0.001
P_COST, P_MISS = msc.P_COST, msc.P_MISS
SIMS = msc.SIMS
SEED = 20180601
YEAR, SHARE_YEAR = 365, 252
STARTS = ("2018-01-01", "2023-12-01")  # highs and lows of bitcoin exist from 2017-08
HALVES = (("2018–2021", "2018-01-01", "2021-12-31"), ("2022–至今", "2022-01-01", None))
SCORE_BAR = 0.79
TWO_THIRDS = 2 / 3
DAY_MS = 86_400_000
KLINES = common.STORE / "donchian_ensemble_klines.json"  # kept by donchian_ensemble - read only
STOCK_FROM = "2000-01-01"

SETUPS = ("A", "A2", "B10c", "B10r", "B5c", "B5r", "C5h5", "C5h10", "C10h5", "C10h10", "D", "E")
OVER = tuple(f"o{k}" for k in SETUPS)
KEYS = SETUPS + ("ma50",) + OVER
NAMES = {
    "A": "A 聖杯（ADX>30 且還在升、拉回 20 日指數均線、突破高點買）",
    "A2": "A2 聖杯寬鬆版（只要 ADX>30，事後加的）",
    "B10c": "B RSI(2)<10，收盤高於 5 日均線賣",
    "B10r": "B RSI(2)<10，RSI(2)>70 賣",
    "B5c": "B RSI(2)<5，收盤高於 5 日均線賣",
    "B5r": "B RSI(2)<5，RSI(2)>70 賣",
    "C5h5": "C 1–3 天跌 5% 買、抱 5 天",
    "C5h10": "C 1–3 天跌 5% 買、抱 10 天",
    "C10h5": "C 1–3 天跌 10% 買、抱 5 天",
    "C10h10": "C 1–3 天跌 10% 買、抱 10 天",
    "D": "D 收在當天低檔（IBS<0.2）買、收過前一天高點賣",
    "E": "E 拉回 20 日均線（50 日均線上升中）",
    "ma50": "50 日均線（收盤高於才持有）",
    "hold": "一直抱著",
}
SHORT = {"A": "聖杯", "A2": "聖杯寬鬆版", "B10c": "RSI2<10／5日線", "B10r": "RSI2<10／>70", "B5c": "RSI2<5／5日線", "B5r": "RSI2<5／>70",
         "C5h5": "跌5%抱5天", "C5h10": "跌5%抱10天", "C10h5": "跌10%抱5天", "C10h10": "跌10%抱10天", "D": "IBS",
         "E": "拉回20日線", "ma50": "50 日均線", "hold": "一直抱著"}
for _k in SETUPS:
    NAMES[f"o{_k}"] = f"50 日均線＋{SHORT[_k]}"
    SHORT[f"o{_k}"] = f"50日線＋{SHORT[_k]}"
CRYPTO_OTHER = (("ETHUSDT", "ETH", None), ("SOLUSDT", "SOL", None), ("BNBUSDT", "BNB", None), ("XRPUSDT", "XRP", None))
SHARES_OTHER = (("MSTR", "MSTR", "2020-08-11"), ("^GSPC", "標普 500", STOCK_FROM), ("^TWII", "台灣加權", STOCK_FROM))
RANK_ASSETS = ("ETH", "SOL", "BNB", "XRP", "MSTR")  # the same five as main_signal_check

_pct, _pp, _d = cr._pct, cr._pp, cr._d
_ok = msc._ok


def _share(x) -> str:
    return "—" if not _ok(x) else f"{x * 100:.0f}%"


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if obj is None:
        return "—"
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else "—"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------- indicators ----------

def _indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    h, lo, c = (pd.Series(np.asarray(x, dtype=float)) for x in (high, low, close))
    prev = c.shift(1)
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    up, dn = h - h.shift(1), lo.shift(1) - lo
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    w = dict(alpha=1 / 14, adjust=False, min_periods=14)
    atr = tr.ewm(**w).mean()
    pdi = 100 * pdm.ewm(**w).mean() / atr.where(atr > 0)
    mdi = 100 * mdm.ewm(**w).mean() / atr.where(atr > 0)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where(pdi + mdi > 0)
    adx = dx.ewm(**w).mean().where(dx.notna().rolling(14, min_periods=14).sum() >= 14)
    d = c.diff()
    ag = d.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    al = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    rsi2 = (100 - 100 / (1 + ag / al.where(al > 0))).where(al > 0, 100.0).where(ag.notna())
    rng = h - lo
    return {
        "c": c.values, "h": h.values, "l": lo.values, "adx": adx.values, "pdi": pdi.values, "mdi": mdi.values,
        "ema20": c.ewm(span=20, adjust=False, min_periods=20).mean().values,
        "sma5": c.rolling(5, min_periods=5).mean().values, "sma20": c.rolling(20, min_periods=20).mean().values,
        "sma50": c.rolling(50, min_periods=50).mean().values, "sma200": c.rolling(200, min_periods=200).mean().values,
        "rsi2": rsi2.values, "ibs": ((c - lo) / rng.where(rng > 0)).values,
        "drop3": (c / pd.concat([c.shift(1), c.shift(2), c.shift(3)], axis=1).max(axis=1) - 1).values,
        "swing": h.rolling(20, min_periods=20).max().values,
    }


def _gt(a, b) -> bool:
    return a == a and b == b and a > b  # False whenever a value is missing


# ---------- the setups (position decided at each close: 1 long, 0 cash) ----------

def _positions(high, low, close) -> dict[str, np.ndarray]:
    x = _indicators(high, low, close)
    c, h, lo = x["c"], x["h"], x["l"]
    n = len(c)
    out = {}

    # A Holy Grail (A: ADX still rising up to the day before the touch; A2: ADX above 30 is enough)
    for key, rising in (("A", True), ("A2", False)):
        pos = np.zeros(n)
        touch = np.zeros(n, dtype=bool)
        for t in range(2, n):
            touch[t] = (lo[t] <= x["ema20"][t] and _gt(x["adx"][t], 30) and _gt(x["pdi"][t], x["mdi"][t])
                        and (not rising or _gt(x["adx"][t - 1], x["adx"][t - 2])))
        inside, target, stop = False, 0.0, 0.0
        for t in range(1, n):
            if inside:
                if c[t] >= target or c[t] < stop:
                    inside = False
            elif touch[t - 1] and c[t] > h[t - 1] and x["swing"][t - 1] == x["swing"][t - 1]:
                inside, target, stop = True, x["swing"][t - 1], lo[t - 1]
            pos[t] = inside
        out[key] = pos

    # B Connors RSI(2)
    for thr in (10, 5):
        for ex in ("c", "r"):
            pos = np.zeros(n)
            inside = False
            for t in range(n):
                if inside:
                    if (ex == "c" and _gt(c[t], x["sma5"][t])) or (ex == "r" and _gt(x["rsi2"][t], 70)):
                        inside = False
                elif _gt(c[t], x["sma200"][t]) and _gt(thr, x["rsi2"][t]):
                    inside = True
                pos[t] = inside
            out[f"B{thr}{ex}"] = pos

    # C dips in an uptrend, fixed holding time
    for drop in (5, 10):
        for days in (5, 10):
            pos = np.zeros(n)
            left = 0
            for t in range(n):
                if left > 0:
                    left -= 1
                elif _gt(c[t], x["sma50"][t]) and _gt(-drop / 100 + 1e-12, x["drop3"][t]):
                    left = days
                pos[t] = left > 0
            out[f"C{drop}h{days}"] = pos

    # D internal bar strength
    pos = np.zeros(n)
    inside = False
    for t in range(1, n):
        if inside:
            if c[t] > h[t - 1]:
                inside = False
        elif _gt(c[t], x["sma200"][t]) and _gt(0.2, x["ibs"][t]):
            inside = True
        pos[t] = inside
    out["D"] = pos

    # E pullback to the 20-day average while the 50-day average rises
    pos = np.zeros(n)
    left = 0
    for t in range(1, n):
        if left > 0:
            left -= 1
            if left == 0 or not _gt(c[t], x["sma50"][t]):
                left = 0
        elif (_gt(c[t], x["sma50"][t]) and _gt(x["sma50"][t], x["sma50"][t - 1])
              and x["sma20"][t] == x["sma20"][t] and lo[t] <= x["sma20"][t]):
            left = 10
        pos[t] = left > 0
    out["E"] = pos

    ma50 = np.where(np.isnan(x["sma50"]), 0.0, (c > np.nan_to_num(x["sma50"], nan=np.inf)).astype(float))
    out["ma50"] = ma50
    for k in SETUPS:
        out[f"o{k}"] = np.maximum(ma50, out[k])
    return out


# ---------- measuring from many start dates ----------

def _ratio(equity: np.ndarray, starts: np.ndarray, year: int) -> np.ndarray:
    """Return / volatility (yearly %, as clock_robustness._stats) from each start to the end; shape (starts, paths)."""
    e = equity.reshape(len(equity), -1)
    net = e[1:] / e[:-1] - 1
    n = len(net)
    s1 = np.cumsum(net[::-1], axis=0)[::-1]
    s2 = np.cumsum((net * net)[::-1], axis=0)[::-1]
    m = (n - starts)[:, None].astype(float)
    a, b = s1[starts], s2[starts]
    vol = np.sqrt(np.maximum((b - a * a / m) / (m - 1), 0.0)) * math.sqrt(year) * 100
    growth = e[-1][None, :] / e[starts]
    ret = np.where(growth > 0, np.abs(growth) ** (year / m) - 1, -1.0) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)


def _daily_grid(frame: pd.DataFrame, year: int, first, last) -> dict:
    """Textbook runs (decide at the close, hold the next day, 0.10%) from monthly start dates."""
    pos = _positions(frame["high"].values, frame["low"].values, frame["close"].values)
    change = frame["close"].pct_change().fillna(0.0).values
    begin = max(frame.index[min(260, len(frame) - 1)], pd.Timestamp(first))
    starts, labels = msc._month_starts(frame.index, begin, last)
    if not len(starts):
        raise RuntimeError("沒有起算日")
    eq = msc._equity(change)
    r, dd = msc._suffix(eq, starts, year)
    out = {"starts": labels, "hold": (r[:, 0], dd[:, 0], _ratio(eq, starts, year)[:, 0]), "pos": pos}
    for k in KEYS:
        net, _ = te._run(np.concatenate([[0.0], pos[k][:-1]]), change, COST)
        eq = msc._equity(net)
        r, dd = msc._suffix(eq, starts, year)
        out[k] = (r[:, 0], dd[:, 0], _ratio(eq, starts, year)[:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, STARTS[0], STARTS[1])
    heq = msc._equity(m.change)
    hr, hd = msc._suffix(heq, starts, YEAR)
    hq = _ratio(heq, starts, YEAR)
    hmax = pd.Series(price).rolling(24, min_periods=1).max().values
    hmin = pd.Series(price).rolling(24, min_periods=1).min().values
    ns = len(starts)
    text = {k: (np.zeros((ns, 24)), np.zeros((ns, 24)), np.zeros((ns, 24))) for k in KEYS}
    person = {k: np.zeros((ns, 24, SIMS), dtype=np.float32) for k in KEYS}
    for h in range(24):
        pos = _positions(np.concatenate([pre, hmax[h::24]]), np.concatenate([pre, hmin[h::24]]),
                         np.concatenate([pre, price[h::24]]))
        for ki, k in enumerate(KEYS):
            w = pos[k][npre:][:days].astype(float)
            eq = np.concatenate([[1.0], eng._mid(h, w, COST, None)])
            r, dd = msc._suffix(eq, starts, YEAR)
            text[k][0][:, h], text[k][1][:, h], text[k][2][:, h] = r[:, 0], dd[:, 0], _ratio(eq, starts, YEAR)[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            mid = eng._mid(b, rt._missed(wl, P_MISS, rng, SIMS), P_COST, None)
            r, _ = msc._suffix(np.vstack([np.ones((1, SIMS)), mid]), starts, YEAR)
            person[k][:, h, :] = r
    return {"starts": labels, "hold": (hr[:, 0], hd[:, 0], hq[:, 0]), "text": text, "person": person}


# ---------- data ----------

def _stored_klines(symbol: str) -> pd.DataFrame:
    """Binance daily high / low / close from donchian_ensemble's store (read only), topped up in memory."""
    try:
        stored = json.loads(KLINES.read_text(encoding="utf-8")).get(symbol, [])
    except Exception:  # noqa: BLE001
        stored = []
    rows = {int(r[0]): r for r in stored if isinstance(r, list) and len(r) == 4}
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    if not rows or max(rows) < today_ms - DAY_MS:
        start = max(rows) - 3 * DAY_MS if rows else 0
        for _ in range(12):
            page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
            if not page:
                break
            now_ms = time.time() * 1000
            for r in page:
                if int(r[6]) < now_ms:
                    rows[int(r[0])] = [int(r[0]), float(r[2]), float(r[3]), float(r[4])]
            start = page[-1][0] + 1
            if len(page) < 1000:
                break
    clean = [rows[ms] for ms in sorted(rows) if ms < today_ms and rows[ms][3] > 0]
    if len(clean) < 400:
        raise RuntimeError(f"{symbol} 日線只有 {len(clean)} 筆")
    f = pd.DataFrame(clean, columns=["ms", "high", "low", "close"])
    f.index = pd.to_datetime(f["ms"], unit="ms").dt.normalize()
    f = f[~f.index.duplicated(keep="last")].drop(columns="ms").asfreq("D")
    gap = f["close"].isna()
    f["close"] = f["close"].ffill()
    f.loc[gap, "high"] = f.loc[gap, "close"]
    f.loc[gap, "low"] = f.loc[gap, "close"]
    return f


def _btc_frame() -> pd.DataFrame:
    """The dashboard's long daily closes with Binance highs and lows where the two closes agree (from 2017-08)."""
    close = te._daily_close()
    frame = pd.DataFrame({"close": close})
    k = _stored_klines("BTCUSDT").reindex(frame.index)
    agree = (k["close"] / frame["close"] - 1).abs() < 0.02
    frame["high"] = np.where(agree, np.fmax(k["high"], frame["close"]), frame["close"])
    frame["low"] = np.where(agree, np.fmin(k["low"], frame["close"]), frame["close"])
    return frame


def _load_other() -> tuple[dict, list[str]]:
    data, problems = {}, []
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    for symbol, key, since in CRYPTO_OTHER + SHARES_OTHER:
        try:
            if symbol.endswith("USDT"):
                frame, year = _stored_klines(symbol), YEAR
            else:
                frame, _ = ri._yahoo(symbol)
                for day, r in frame.attrs.get("repairs", []):
                    if day < pd.Timestamp(since) - pd.Timedelta(days=800):
                        continue  # before the part used here
                    problems.append(f"{key} 的 Yahoo 價格在 {_d(day)} 跳了 {(r - 1) * 100:+.0f}%（拆股只調整了部分歷史），"
                                    "之前的價格乘上同一比例接起來。")
                frame = frame[frame.index >= pd.Timestamp(since) - pd.Timedelta(days=800)]
                frame, year = frame[["high", "low", "close"]], SHARE_YEAR
            data[key] = _daily_grid(frame, year, since or frame.index[0], three)
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            problems.append(f"{key} 這次取不到或算不出來（{type(exc).__name__}），沒有算進去。")
    return data, problems


# ---------- checks ----------

def _lookahead(frame: pd.DataFrame) -> tuple[int, int]:
    """Change every price after a cut and recompute: positions up to the cut must not move. (cuts, mismatches)."""
    base = _positions(frame["high"].values, frame["low"].values, frame["close"].values)
    rng = np.random.default_rng(7)
    n = len(frame)
    cuts = sorted(rng.integers(n - 2500, n - 5, size=8))
    bad = 0
    for cut in cuts:
        f = frame.copy()
        shock = np.exp(np.cumsum(rng.normal(0, 0.04, n - cut - 1)))
        f.iloc[cut + 1:, f.columns.get_loc("close")] *= shock
        f.iloc[cut + 1:, f.columns.get_loc("high")] *= shock * 1.03
        f.iloc[cut + 1:, f.columns.get_loc("low")] *= shock * 0.97
        got = _positions(f["high"].values, f["low"].values, f["close"].values)
        bad += sum(int(not np.array_equal(base[k][:cut + 1], got[k][:cut + 1])) for k in KEYS)
    return len(cuts), bad


def _halves(frame: pd.DataFrame, pos: dict) -> dict:
    change = frame["close"].pct_change().fillna(0.0)
    out = {}
    for k in KEYS + ("hold",):
        if k == "hold":
            net = change.values
        else:
            net, _ = te._run(np.concatenate([[0.0], pos[k][:-1]]), change.values, COST)
        net = pd.Series(net, index=frame.index)
        out[k] = []
        for _, a, b in HALVES:
            part = net[(net.index >= pd.Timestamp(a)) & ((net.index <= pd.Timestamp(b)) if b else True)]
            out[k].append(cr._stats(part.values))
    return out


# ---------- the whole computation ----------

def _compute() -> dict:
    frame = _btc_frame()
    daily = _daily_grid(frame, YEAR, STARTS[0], STARTS[1])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre)
    other, problems = _load_other()
    if problem:
        problems.insert(0, problem)
    pos = daily["pos"]
    trades = {k: sig._record(frame["close"], pd.Series(pos[k], index=frame.index), pd.Timestamp(STARTS[0]), YEAR, True)
              for k in KEYS}
    trades["hold"] = sig._record(frame["close"], pd.Series(1.0, index=frame.index), pd.Timestamp(STARTS[0]), YEAR, False)
    return {"frame": frame, "daily": daily, "clock": clock, "other": other, "problems": problems, "trades": trades,
            "halves": _halves(frame, pos), "look": _lookahead(frame)}


def _summary(r: dict) -> dict:
    d, c, o = r["daily"], r["clock"], r["other"]
    hr, hd, hq = c["hold"]
    s = {}
    for k in KEYS:
        dr = d[k][0]
        tr, td, tq = c["text"][k]
        edge = tr - hr[:, None]
        pedge = c["person"][k] - hr[:, None, None]
        rank = [a for a in RANK_ASSETS if a in o]
        o_n = sum(len(o[a]["starts"]) for a in rank)
        o_beat = sum(int((o[a][k][0] > o[a]["hold"][0]).sum()) for a in rank)
        x = {"d_n": len(dr), "d_beat": int((dr > d["hold"][0]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "c_qbeat": int((tq > hq[:, None]).sum()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "o_n": o_n, "o_beat": o_beat,
             "assets": {a: (len(o[a]["starts"]), int((o[a][k][2] > o[a]["hold"][2]).sum()),
                            int((o[a][k][0] > o[a]["hold"][0]).sum()),
                            float(np.median(o[a][k][0] - o[a]["hold"][0])),
                            float(np.median(o[a][k][1])), float(np.median(o[a][k][2] - o[a]["hold"][2])))
                        for a in o}}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (o_beat / o_n) if o_n else 0.0]
        x["low"] = min(x["checks"])
        x["q_share"] = x["c_qbeat"] / x["c_n"]
        x["asset_wins"] = sum(1 for a in o if x["assets"][a][1] * 2 > x["assets"][a][0])
        x["asset_n"] = len(o)
        x["works"] = bool(x["q_share"] >= TWO_THIRDS - 1e-12 and x["asset_n"]
                          and x["asset_wins"] >= math.ceil(TWO_THIRDS * x["asset_n"] - 1e-9))
        s[k] = x
    base = s["ma50"]
    for k in OVER:
        x = s[k]
        x["pass_score"] = x["low"] >= SCORE_BAR
        x["pass_deep"] = x["c_deep_med"] >= base["c_deep_med"] - 1e-9
        x["pass_edge"] = x["c_edge"] > base["c_edge"]
        x["improves"] = x["pass_score"] and x["pass_deep"] and x["pass_edge"]
    return {"s": s, "hold_deep_med": float(np.median(hd)), "hold_q_med": float(np.median(hq))}


# ---------- today ----------

def _now(frame: pd.DataFrame, pos: dict) -> list[dict]:
    x = _indicators(frame["high"].values, frame["low"].values, frame["close"].values)
    i = len(frame) - 1
    c = x["c"][i]

    def state(k: str) -> str:
        p = pos[k]
        if p[i] < 0.5:
            return "沒有持有"
        j = i
        while j > 0 and p[j - 1] > 0.5:
            j -= 1
        return f"持有中（{_d(frame.index[j])} 收盤進場）"

    def rel(a, b) -> str:
        return "—" if not (_ok(a) and _ok(b)) else f"{(a / b - 1) * 100:+.1f}%"

    def num(v, digits=1) -> str:
        return "—" if not _ok(v) else f"{v:.{digits}f}"

    out = [{"label": f"最後一根收完的日線（{_d(frame.index[-1])}，UTC 0 點收盤）",
            "value": f"比特幣 {c:,.0f} 美元；最高 {x['h'][i]:,.0f}、最低 {x['l'][i]:,.0f}"},
           {"label": "50 日均線規則", "value": f"收盤比 50 日均線 {rel(c, x['sma50'][i])}，{'持有' if pos['ma50'][i] > 0.5 else '現金'}"},
           {"label": "A 聖杯", "value": (f"ADX（趨勢強度指標）{num(x['adx'][i])}、+DI {num(x['pdi'][i])}／-DI {num(x['mdi'][i])}、"
                                        f"收盤比 20 日指數均線 {rel(c, x['ema20'][i])}；聖杯 {state('A')}；寬鬆版 {state('A2')}")},
           {"label": "B RSI(2)", "value": (f"RSI(2) {num(x['rsi2'][i])}、收盤比 200 日均線 {rel(c, x['sma200'][i])}；"
                                          + "；".join(f"{SHORT[k]} {state(k)}" for k in ("B10c", "B10r", "B5c", "B5r")))},
           {"label": "C 漲勢中跌深買", "value": (f"收盤比前三天最高收盤 {num(x['drop3'][i] * 100)}%；"
                                            + "；".join(f"{SHORT[k]} {state(k)}" for k in ("C5h5", "C5h10", "C10h5", "C10h10")))},
           {"label": "D IBS", "value": f"IBS {num(x['ibs'][i], 2)}；{state('D')}"},
           {"label": "E 拉回 20 日均線", "value": (f"50 日均線{'上升' if _gt(x['sma50'][i], x['sma50'][i - 1]) else '沒有上升'}、"
                                             f"最低價比 20 日均線 {rel(x['l'][i], x['sma20'][i])}；{state('E')}")},
           {"label": "這幾格的意思", "value": "只描述各規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"}]
    return out


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    sm = _summary(r)
    s, c, o, tr, hv = sm["s"], r["clock"], r["other"], r["trades"], r["halves"]
    frame = r["frame"]
    n_d, n_c = s["ma50"]["d_n"], s["ma50"]["c_n"]
    assets = list(o)

    def pf(x: dict):
        if not x.get("trades") or x.get("avg_win") is None:
            return None
        losses = x["trades"] - x["wins"]
        if not losses or x.get("avg_loss") is None or abs(x["avg_loss"]) < 1e-9:
            return None
        return x["wins"] * x["avg_win"] / (losses * abs(x["avg_loss"]))

    def trow(k: str) -> list:
        x = tr[k]
        if k == "hold":
            rate, aw, f = "—", "—", "—"
        else:
            rate = f"{x['trades_per_year']:.1f} 次｜{x['wins'] / x['trades'] * 100:.0f}%" if x.get("trades") else "0 次｜—"
            aw = f"{_pct(x.get('avg_win'))}｜{_pct(x.get('avg_loss'))}"
            f = "—" if pf(x) is None else f"{pf(x):.2f}"
        halves = "｜".join(f"{_pct(st['ret'])}" for st in hv[k])
        q = "｜".join(f"{st['ratio']:.2f}" for st in hv[k])
        return [NAMES[k], rate, aw, f, f"{_pct(x['ret'])}｜{_pct(x['deep'])}", f"{x['share']:.0f}%", halves, q]

    rows1 = [trow(k) for k in SETUPS + ("ma50", "hold")]

    rows2 = []
    for k in SETUPS + ("ma50",):
        x = s[k]
        a_rank = "、".join(a for a in assets if x["assets"][a][1] * 2 > x["assets"][a][0]) or "沒有"
        rows2.append([NAMES[k], f"{x['c_qbeat']:,}／{x['c_n']:,}（{_share(x['q_share'])}）", _pp(x["c_edge"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}",
                      f"{x['asset_wins']}／{x['asset_n']}：{a_rank}", _share(x["low"]),
                      "過關" if x["works"] else "沒過"])
    rows2.append(["對照：一直抱著", "—", "—", f"{_pct(sm['hold_deep_med'])}｜{_pct(float(c['hold'][1].min()))}", "—", "—", "—"])

    rows3 = []
    for k in ("ma50",) + OVER:
        x = s[k]
        ok = "—" if k == "ma50" else ("會變好" if x["improves"] else "沒變好（" + "、".join(
            w for w, p in (("分數", x["pass_score"]), ("虧損", x["pass_deep"]), ("報酬", x["pass_edge"])) if not p) + "不過）")
        rows3.append([NAMES[k], "｜".join(_share(v) for v in x["checks"]), f"{x['low'] * 100:.1f}%", _pp(x["c_edge"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", _pp(x["p_edge"]),
                      f"{tr[k]['trades_per_year']:.1f} 次｜{tr[k]['share']:.0f}%", ok])

    stock = [a for a in ("標普 500", "台灣加權") if a in o]
    rows4 = []
    for k in ("ma50",) + OVER:
        x = s[k]
        row = [NAMES[k]] + [f"{_pct(st['ret'])}、{_pct(st['deepest'])}" for st in hv[k]]
        for a in stock:
            n, _, beat, edge, deep, _ = x["assets"][a]
            base = s["ma50"]["assets"][a]
            row.append(f"{_pp(edge)}｜{_pct(deep)}" + ("" if k == "ma50" else f"（50 日線 {_pp(base[3])}｜{_pct(base[4])}）"))
        while len(row) < 5:
            row.append("—")
        rows4.append(row)
    rows4.append(["對照：一直抱著"] + [f"{_pct(st['ret'])}、{_pct(st['deepest'])}" for st in hv["hold"]]
                 + [f"0.0｜{_pct(float(np.median(o[a]['hold'][1])))}" for a in stock] + ["—"] * (2 - len(stock)))

    tables = [
        {"caption": (f"比特幣、單獨用（沒訊號時拿現金），儀表板日線（UTC 0 點收盤），從 {STARTS[0]} 起算到 {_d(frame.index[-1])}，"
                     f"收盤決定、收盤成交，每次買賣扣 {COST * 100:.2f}%。交易只算已結束的。「獲利因子」= 賺錢交易合計 ÷ 賠錢交易合計，"
                     "大於 1 才是整體有賺；「持有時間比例」是有部位的天數占比。報酬÷波動越高越好。"),
         "columns": ["規則", "一年交易幾次｜勝率", "平均每筆賺｜平均每筆賠", "獲利因子", "每年報酬｜最深虧損", "持有時間比例",
                     f"每年報酬：{HALVES[0][0]}｜{HALVES[1][0]}", f"報酬÷波動：{HALVES[0][0]}｜{HALVES[1][0]}"],
         "rows": rows1},
        {"caption": (f"單獨用的公平檢查：比特幣 {len(c['starts'])} 個起算日（{STARTS[0][:7]} 到 {STARTS[1][:7]}，每月 1 日）× 24 個收盤時間"
                     f" = {n_c:,} 格（日線由 1 小時線拼出，最高、最低用那 24 個整點價格），照課本做、成本 {COST * 100:.2f}%；"
                     f"其他資產 {len(assets)} 個（{'、'.join(assets)}），每個資產有一半以上的起算日報酬÷波動贏過一直抱著才算一個。"
                     "事先定好的過關標準：比特幣格子和其他資產都至少三分之二贏過一直抱著（看報酬÷波動）。"
                     "「排名分數」是 main_signal_check 的四項（比特幣起算日、起算日×收盤時間、一般人做法、五個其他資產，看每年報酬贏的比例）裡最低的一項。"),
         "columns": ["規則", "報酬÷波動贏過一直抱著的格", "每年報酬比一直抱著多（中位數，百分點）", "最深虧損：中位數｜最深的一格",
                     "其他資產贏的個數", "排名分數", "單獨用過關？"],
         "rows": rows2},
        {"caption": ("加在 50 日均線上：50 日均線說持有就全部持有；50 日均線說現金時，拉回買法有訊號才持有（最多 100%）。"
                     f"第二欄是四項各自報酬贏過一直抱著的比例：比特幣 {n_d} 個起算日｜{n_c:,} 格｜一般人做法（收盤後 10 小時才做、"
                     f"成本 {P_COST * 100:.2f}%、{P_MISS * 100:.0f}% 的進場漏掉，每個收盤時間 {SIMS} 次）｜五個其他資產。"
                     f"事先定好的標準：排名分數至少 {SCORE_BAR * 100:.0f}%、最深虧損中位數不比 50 日均線深、每年報酬多出的中位數比 50 日均線高，三項都過才算變好。"),
         "columns": ["規則", "四項贏的比例", "排名分數（四項最低）", "每年報酬比抱著多（中位數）", "最深虧損：中位數｜最深的一格",
                     "一般人做法：比抱著多（中位數）", "一年交易幾次｜持有時間比例", "有沒有比 50 日均線好"],
         "rows": rows3},
        {"caption": ("加在 50 日均線上，分前後兩段看（比特幣 UTC 0 點日線，照課本做），再看兩個股票指數（Yahoo 日線，2000 年起每月起算到三年前，"
                     "照課本做、成本 0.10%）。股票欄是「每年報酬比一直抱著多的中位數（百分點）｜最深虧損中位數」，括號內是只用 50 日均線的同樣數字。"),
         "columns": ["規則", f"比特幣 {HALVES[0][0]}：每年報酬、最深虧損", f"比特幣 {HALVES[1][0]}：每年報酬、最深虧損"]
                    + [f"{a}：比抱著多｜最深虧損" for a in stock] + ["—"] * (2 - len(stock)),
         "rows": rows4},
    ]

    # verdict
    def rng_text(keys, fn, fmt) -> str:
        vals = [fn(k) for k in keys if fn(k) is not None]
        return "—" if not vals else f"{fmt(min(vals))}–{fmt(max(vals))}"

    bkeys = ("B10c", "B10r", "B5c", "B5r")
    works = [k for k in SETUPS if s[k]["works"]]
    better = [k for k in OVER if s[k]["improves"]]
    first = (f"比特幣 2018 年起照原書做：「聖杯」一年只出手 {tr['A']['trades_per_year']:.1f} 次（寬鬆版 "
             f"{tr['A2']['trades_per_year']:.1f} 次、每年 {_pct(tr['A2']['ret'])}），RSI(2) 勝率 "
             f"{rng_text(bkeys, lambda k: tr[k]['wins'] / tr[k]['trades'] * 100 if tr[k].get('trades') else None, lambda v: f'{v:.0f}%')}、"
             f"賺賠比（獲利因子）{rng_text(bkeys, lambda k: pf(tr[k]), lambda v: f'{v:.1f}')}，但大多時間空手，每年只有 "
             f"{rng_text(bkeys, lambda k: tr[k]['ret'], _pct)}，一直抱著是 {_pct(tr['hold']['ret'])}。")
    if works:
        second = f"換起算日、收盤時間、其他幣和標普、台股一起考，單獨用過關的有：{'、'.join(SHORT[k] for k in works)}。"
    else:
        best = max(SETUPS, key=lambda k: (s[k]["q_share"], s[k]["asset_wins"]))
        second = (f"換起算日、收盤時間、其他幣和標普、台股一起考，12 種沒有一種單獨用過關（最接近的「{SHORT[best]}」在比特幣 "
                  f"{_share(s[best]['q_share'])} 的格子贏，但 {s[best]['asset_n']} 個其他資產只贏 {s[best]['asset_wins']} 個）。")
    if better:
        bs = [s[k] for k in better]
        third = (f"加在 50 日均線上，只有{'、'.join('「' + SHORT[k[1:]] + '」' for k in better)}剛好通過三項事先標準（分數 "
                 + "／".join(f"{b['low'] * 100:.1f}%" for b in bs) + "，比 50 日均線每年多 "
                 + "／".join(f"{b['c_edge'] - s['ma50']['c_edge']:.1f}" for b in bs) + " 點、最深虧損淺 "
                 + "／".join(f"{b['c_deep_med'] - s['ma50']['c_deep_med']:.1f}" for b in bs)
                 + (" 點，一般人照做則沒有多賺），其餘沒有變好。" if all(b["p_edge"] <= s["ma50"]["p_edge"] for b in bs)
                    else " 點），其餘沒有變好。"))
    else:
        bo = max(OVER, key=lambda k: s[k]["c_edge"])
        third = (f"加在 50 日均線上，沒有一種通過三項事先標準（最好的「{SHORT[bo[1:]]}」分數 {s[bo]['low'] * 100:.1f}%、"
                 f"報酬多 {s[bo]['c_edge'] - s['ma50']['c_edge']:+.1f} 點）。")
    verdict = first + second + third

    look_n, look_bad = r["look"]
    notes = [
        "參數全部照原書或題目先寫好的數字，沒有調過：聖杯 ADX(14)>30、20 日指數均線；RSI(2) 用 Wilder 平滑，門檻 10 和 5、出場收盤高於 5 日均線或 RSI(2)>70；"
        "跌深買 5%／10%、抱 5／10 天；IBS<0.2；拉回 20 日均線出場是收盤跌破 50 日均線或滿 10 天（這條原本沒有標準出場，看結果前就定好）。",
        "聖杯的細節：觸碰日要最低價碰到 20 日指數均線、ADX>30、ADX 到前一天為止在上升（書上說拉回時 ADX 常常轉頭向下，所以觸碰當天不要求上升）、+DI>-DI；"
        "之後收盤突破觸碰日的最高價才買（原書是盤中掛突破單，日線只能看收盤），收盤到達進場前 20 天的最高價（最近的波段高點）就賣，收盤跌破觸碰日最低價停損，沒有另外做移動停損。"
        f"照這樣寫，比特幣 2018 年起只出手 {tr['A']['trades']} 次（ADX 還在上升、碰到均線時，多半是 -DI 大於 +DI 的下跌趨勢），"
        "所以看完結果後另外加了「寬鬆版」：不要求 ADX 上升，其他一樣；這一條是事後加的，數字要打折看。",
        "聖杯和「拉回 20 日均線」的進場幾乎都落在 50 日均線本來就持有的時候（拉回 20 日均線那條更是跌破 50 日均線就出場），"
        "所以加在 50 日均線上幾乎沒有改變任何部位；真正會多出部位的是 RSI(2)、跌深買和 IBS。",
        f"50 日均線自己在這裡的排名分數是 {s['ma50']['low'] * 100:.1f}%，本身就略低於 {SCORE_BAR * 100:.0f}% 的門檻；"
        "通過門檻的加法都是剛好超過；照一般人方式做時的報酬差距中位數請和 50 日均線比（第三張表倒數第二欄）。"
        "兩個股票指數上加 RSI(2) 的變化見第四張表（股票是這些買法原本的出處）。",
        "日線只看得到收盤，盤中碰到再拉回的觸發都漏掉了；成交價一律用決定那天的收盤價。比特幣 24 小時交易，「隔天開盤」就是下一根日線的開始，價格等於這根收盤，"
        "所以收盤成交和隔天開盤成交在比特幣上是同一件事；股票指數則用收盤成交，沒有算隔夜跳空，這對 RSI(2)（原書就是收盤買）沒影響，對其他幾種略為樂觀。",
        f"成本：照課本做每次買賣扣 {COST * 100:.2f}%，一般人做法扣 {P_COST * 100:.2f}%；沒有算滑價、稅和現金利息。短線買法一年交易十幾到幾十次，成本吃掉的比例比順勢規則大。",
        "比特幣 2017-08 以前沒有可靠的最高、最低價（只有每天一個價格），所以起算日從 2018-01 開始；24 個收盤時間的日線最高、最低用 1 小時收盤價拼，比真正的盤中高低窄一點，"
        "IBS 和聖杯的觸碰會略受影響。2017-08 以前的日線只用來讓 200 日均線等先有值。",
        f"沒有偷看未來的檢查：隨機挑 {look_n} 個日子，把之後的價格全部亂改再重算，{len(KEYS)} 條規則在那天以前的部位"
        + ("完全沒變。" if look_bad == 0 else f"有 {look_bad} 次變了，結果不可靠。"),
        "排名分數照 main_signal_check：四項各算「每年報酬贏過一直抱著」的比例，取最低的一項；其他資產那項用同樣的五個（ETH、SOL、BNB、XRP、MSTR），標普和台股另外列。"
        f"79% 的門檻是看結果前給定的。這裡 50 日均線因為比特幣起算日只有 {n_d} 個（2018 年起）、一般人做法模擬的亂數不同，分數會和 main_signal_check 的表略有差異。",
        "相鄰起算日、相鄰收盤時間共用大部分的歷史，「幾格裡幾格」不是互相獨立的證據；其他資產是今天還在的，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    result = {
        "id": "pullback_setups",
        "title": "上漲途中買拉回",
        "question": ("有名的「上漲途中買拉回」買法——包括 Raschke 的「聖杯」和 Connors 的 RSI(2)——在比特幣上有沒有賺？"
                     "加在 50 日均線規則上會不會變好？"),
        "asof": _d(frame.index[-1]),
        "verdict": verdict,
        "now": _now(frame, r["daily"]["pos"]),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Linda Bradford Raschke & Laurence Connors（1995），Street Smarts：「聖杯」（Holy Grail）買法", "url": ""},
            {"name": "Larry Connors & Cesar Alvarez（2008），Short Term Trading Strategies That Work：RSI(2)", "url": ""},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR、標普 500 ^GSPC、台灣加權 ^TWII 日線", "url": "https://finance.yahoo.com/quote/%5EGSPC/history"},
        ],
    }
    return _clean(result)


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
