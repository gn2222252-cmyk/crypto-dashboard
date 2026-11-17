"""Historical study "ewma_forecast": does a continuous, volatility-scaled trend forecast with a no-trade buffer beat
the dashboard's simple on/off breakout rule - for bitcoin alone (long or cash) and for a basket of large coins,
especially from 2024 on?

The forecast follows the README of the open project OctopusTakopi/crypto-trend-following (a long-short system on
Binance perpetuals), adapted here to long-or-cash on spot closes:

  p = log(close);  three speeds (S, L) = (16, 48), (32, 96), (64, 192) days
  y_k = (EWMA_S(p) - EWMA_L(p)) / (sigma_d * sqrt(L))
        sigma_d = max(0.7 * sd of the last 60 daily log returns + 0.3 * sd of all history (at most 10 years),
                      20% a year / sqrt(365))
  z_k = y_k / (trailing 365-day standard deviation of y_k, at least 60 values)
  z   = average of the available z_k, clipped to [-6, 6];  response f = tanh(z)
  buffer: do nothing while the held share is within b = 10% of a full allocation of the target, otherwise trade to
          the near edge of that band

Variants (no leverage): A = clip(f, 0, 1) with the buffer; B = A x min(1, target / 30-day realised volatility) for
targets 40% and 25%; each speed alone; the plain sign version (in when z > 0); a long/short version clip(f, -1, 1);
benchmarks rebuilt as in trend_ensemble (buy-and-hold, 20/10 closing breakout, blend of 17 rules, blend x 40%).
Basket: sixteen large coins (Binance spot, survivor-biased list), equal weight across the coins that have a forecast.

No look-ahead: everything decided at the close of day t uses data up to that close and sets the share held during
day t+1. study() really runs a perturbation check (prices after a date are changed; the shares held up to the next
day must stay identical) and reports it. Costs: 0.10% of the amount traded. Only closed daily candles from public,
key-less endpoints through the dashboard's helpers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import price as px

COST = 0.001  # 0.10% of the amount traded
YEAR = 365
DAY_MS = 86_400_000
MAIN_START = "2018-01-01"
PICK_FROM = "2013-01-01"  # scoring data for the pick-in-advance test starts here
PICK_FIRST = 2020
PAIRS = ((16, 48), (32, 96), (64, 192))
VOL_SHORT = 60  # days of log returns in the short volatility estimate
VOL_LONG_CAP = 3650  # the long estimate uses all history, at most ten years
VOL_SHORT_WEIGHT = 0.70
VOL_FLOOR = 0.20  # yearly
RESCALE_DAYS = 365
RESCALE_MIN = 60
Z_CAP = 6.0
BUFFER = 0.10
BUFFERS = (0.0, 0.05, 0.10, 0.20)
COSTS = (0.001, 0.003, 0.005)
SIZE_VOL_DAYS = 30
TARGETS = (0.40, 0.25)
MIN_BARS = 120  # a coin joins the basket only after this many daily bars (the project's own history gate)
MA_DAYS = (10, 20, 30, 50, 75, 100, 150, 200)
CHANNEL = (10, 20, 50, 100)
PAST = (14, 30, 60, 90, 180)
COINS = ("BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "LTC", "LINK", "TRX", "DOT", "AVAX", "BCH", "XLM", "ATOM",
         "ETC")
STORE_FILE = "ewma_forecast_klines.json"
SEED_FILE = "donchian_ensemble_klines.json"  # read only, to avoid downloading the whole history again
CHECK_DATES = ("2018-12-15", "2020-03-12", "2021-11-10", "2023-06-30", "2025-03-19")
SEED = 20240101
PERIODS = (("all", MAIN_START, None), ("p1", "2018-01-01", "2021-12-31"), ("p2", "2022-01-01", "2023-12-31"),
           ("recent", "2024-01-01", None))

ORDER = ("hold", "c20", "blend", "blend40", "A", "B40", "B25", "p48", "p96", "p192", "sign", "ls")
ONOFF = ("c20", "sign")
TREND = ("c20", "blend", "blend40", "A", "B40", "B25", "p48", "p96", "p192", "sign")  # long-or-cash trend variants
PICKS = ("c20", "blend", "A", "B40", "sign")
NAMES = {
    "hold": "一直抱著不動",
    "c20": "簡單進出：創 20 日新高進場、10 日新低出場",
    "blend": "17 條規則混合",
    "blend40": "17 條規則混合 × 波動調整（目標 40%）",
    "A": "A 連續加減碼（三組快慢線平均＋緩衝區）",
    "B40": "B 連續加減碼 × 波動調整（目標 40%）",
    "B25": "B 連續加減碼 × 波動調整（目標 25%）",
    "p48": "只用 16／48 日這一組",
    "p96": "只用 32／96 日這一組",
    "p192": "只用 64／192 日這一組",
    "sign": "只看正負：趨勢強度大於 0 全部持有，否則全部現金",
    "ls": "A 的多空版：趨勢強度是負的就做空",
}
SHORT = {"hold": "一直抱著", "c20": "20/10 簡單進出", "blend": "17 條混合", "blend40": "混合×波動 40%", "A": "A 連續加減碼",
         "B40": "B 加波動調整 40%", "sign": "只看正負"}


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


def _usd(v) -> str:
    return f"{float(v):,.0f} 美元" if _ok(v) else "—"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _cmp(a: float, b: float, tol: float, more: str, less: str) -> str:
    if abs(a - b) <= tol:
        return "差不多"
    return more if a > b else less


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed daily bitcoin closes since 2010 with no missing day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


def _read_json(path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a damaged store file starts over
        return {}


def _coin_closes() -> tuple[dict, list[str]]:
    """Finished Binance spot daily closes per coin from its first listed day, kept under common.STORE so a later run
    only asks for the newest days. The first run starts from the candles donchian_ensemble already stored."""
    path = common.STORE / STORE_FILE
    stored = _read_json(path)
    seed = None
    today = common.today_utc()
    closes, problems = {}, []
    for coin in COINS:
        symbol = f"{coin}USDT"
        days = {str(k)[:10]: float(v) for k, v in (stored.get(symbol) or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(v) and v > 0} if isinstance(stored.get(symbol), dict) else {}
        if not days:
            if seed is None:
                seed = _read_json(common.STORE / SEED_FILE)
            for row in seed.get(symbol) or []:
                if isinstance(row, list) and len(row) == 4 and isinstance(row[3], (int, float)) and row[3] > 0:
                    days[pd.Timestamp(int(row[0]), unit="ms").strftime("%Y-%m-%d")] = float(row[3])
        try:
            start = int(pd.Timestamp(max(days)).value // 1_000_000) - 3 * DAY_MS if days else 0
            for _ in range(12):
                page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000,
                                                      "startTime": start})
                if not page:
                    break
                for r in page:
                    days[pd.Timestamp(int(r[0]), unit="ms").strftime("%Y-%m-%d")] = float(r[4])
                start = page[-1][0] + 1
                if len(page) < 1000:
                    break
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            if not days:
                problems.append(f"{coin} 這次取不到資料（{type(exc).__name__}），沒有算進籃子。")
                continue
            problems.append(f"{coin} 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的日線。")
        days = {day: v for day, v in days.items() if day < today and v > 0}  # never today's unfinished candle
        if len(days) < 400:
            problems.append(f"{coin} 的日線只有 {len(days)} 筆，沒有算進籃子。")
            continue
        stored[symbol] = dict(sorted(days.items()))
        series = pd.Series(stored[symbol], dtype=float)
        series.index = pd.to_datetime(series.index)
        closes[coin] = series.sort_index().asfreq("D").ffill()
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return closes, problems


# ---------- the forecast and the rules ----------

def _forecast(close: pd.Series) -> pd.DataFrame:
    """z of each speed and the combined z at each day's close (NaN until there is enough history)."""
    logp = np.log(close)
    ret = logp.diff()
    short = ret.rolling(VOL_SHORT, min_periods=VOL_SHORT).std(ddof=1)
    long = ret.rolling(VOL_LONG_CAP, min_periods=VOL_SHORT).std(ddof=1)
    sigma = (VOL_SHORT_WEIGHT * short + (1 - VOL_SHORT_WEIGHT) * long).clip(lower=VOL_FLOOR / math.sqrt(YEAR))
    count = np.arange(1, len(close) + 1)
    cols = {}
    for fast, slow in PAIRS:
        gap = logp.ewm(span=fast, adjust=False).mean() - logp.ewm(span=slow, adjust=False).mean()
        y = (gap / (sigma * math.sqrt(slow))).where(count >= slow)
        sd = y.rolling(RESCALE_DAYS, min_periods=RESCALE_MIN).std(ddof=1)
        cols[f"z{slow}"] = y / sd.where(sd > 1e-12)
    frame = pd.DataFrame(cols)
    frame["z"] = frame.mean(axis=1, skipna=True).clip(-Z_CAP, Z_CAP)
    frame["sigma"] = sigma * math.sqrt(YEAR)
    return frame


def _channel(close: pd.Series, n: int) -> pd.Series:
    m = n // 2
    high = close.rolling(n, min_periods=n).max()
    low = close.rolling(m, min_periods=m).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0).where(high.notna())  # cash until a first high


def _blend(close: pd.Series) -> pd.Series:
    """Share of the 17 rules of trend_ensemble that are long (NaN until every rule has enough history)."""
    cols = {}
    for n in MA_DAYS:
        mean = close.rolling(n, min_periods=n).mean()
        cols[f"A{n}"] = (close > mean).astype(float).where(mean.notna())
    for n in CHANNEL:
        cols[f"C{n}"] = _channel(close, n)
    for n in PAST:
        past = close / close.shift(n) - 1
        cols[f"D{n}"] = (past > 0).astype(float).where(past.notna())
    frame = pd.DataFrame(cols)
    return frame.mean(axis=1).where(frame.notna().all(axis=1))


def _targets(close: pd.Series, full: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(target, band, forecast): what each variant wants to hold, decided at each day's close (before any shift),
    and the half-width of its no-trade band (0 = always go to the target)."""
    fc = _forecast(close)
    z = fc["z"]
    target, band = pd.DataFrame(index=close.index), pd.DataFrame(index=close.index)
    target["c20"], band["c20"] = _channel(close, 20), 0.0
    target["A"], band["A"] = np.tanh(z).clip(0.0, 1.0), BUFFER
    target["sign"], band["sign"] = (z > 0).astype(float).where(z.notna()), 0.0
    if not full:
        return target, band, fc
    vol = close.pct_change().rolling(SIZE_VOL_DAYS, min_periods=SIZE_VOL_DAYS).std(ddof=1) * math.sqrt(YEAR)
    fc["vol30"] = vol
    share = _blend(close)
    target["blend"], band["blend"] = share, 0.0
    for goal in TARGETS:
        scale = (goal / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
        key = f"B{int(goal * 100)}"
        target[key], band[key] = target["A"] * scale, (BUFFER * scale).fillna(0.0)
        fc[f"scale{int(goal * 100)}"] = scale
        if goal == TARGETS[0]:
            target["blend40"], band["blend40"] = share * scale, 0.0
    for _, slow in PAIRS:
        target[f"p{slow}"], band[f"p{slow}"] = np.tanh(fc[f"z{slow}"]).clip(0.0, 1.0), BUFFER
    target["ls"], band["ls"] = np.tanh(z), BUFFER
    return target, band, fc


# ---------- simulating ----------

def _step(cur: float, want: float, band: float, short: bool) -> float:
    """The share held next, given the share held now (after the price moved), the target and the band."""
    if abs(want - cur) <= band:
        new = cur
    elif want == 0.0 and not short:
        new = 0.0  # long-or-cash: a target of zero means all cash, not "the edge of the band"
    else:
        new = want - band if cur < want else want + band  # trade to the near edge only
    return min(1.0, max(-1.0 if short else 0.0, new))


def _sim(target, band, change, cost: float, delay: int = 1, short: bool = False) -> dict:
    """Daily net return, share held and amount traded. The target decided at the close of day t is acted on for
    day t+delay; between trades the held share drifts with the price: w -> w(1+r)/(1+wr)."""
    n = len(change)
    tgt = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0)
    bnd = np.nan_to_num(np.broadcast_to(np.asarray(band, dtype=float), (n,)), nan=0.0)
    chg = np.nan_to_num(np.asarray(change, dtype=float), nan=0.0)
    net, held, trade = np.zeros(n), np.zeros(n), np.zeros(n)
    cur = 0.0  # starts from cash
    for i in range(n):
        j = i - delay
        new = _step(cur, tgt[j], bnd[j], short) if j >= 0 else cur
        moved, r = abs(new - cur), chg[i]
        held[i], trade[i], net[i] = new, moved, new * r - cost * moved
        base = 1 + new * r
        cur = new * (1 + r) / base if abs(base) > 1e-9 else new
    return {"net": net, "held": held, "trade": trade, "next": _step(cur, tgt[-1], bnd[-1], short)}


def _score(net: np.ndarray) -> tuple[float | None, float, float]:
    n = len(net)
    if n < 2:
        return None, 0.0, 0.0
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    return ret, vol, (ret / vol if vol > 1e-9 else 0.0)


def _measure(net: pd.Series, held: pd.Series, trade: pd.Series) -> dict:
    values = net.values
    ret, vol, ratio = _score(values)
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
    year_ratio = {int(y): _score(chunk.values)[2] for y, chunk in net.groupby(net.index.year)}
    return {"days": len(values), "ret": ret, "vol": vol, "ratio": ratio, "deepest": deepest, "under": int(longest),
            "under_open": bool(open_end), "years": by_year, "year_ratio": year_ratio,
            "share": float(held.mean() * 100), "turnover": float(trade.sum() / (len(values) / YEAR)),
            "growth": float(np.prod(1 + values))}


def _trades(run: dict, rows: np.ndarray) -> tuple[int, int]:
    """(round trips entered inside `rows`, how many of them lost money after costs) for an all-in / all-out rule.
    A trip still open at the end is counted with its result so far."""
    held, net = run["held"], run["net"]
    count = lost = 0
    inside, growth, counted = False, 1.0, False
    for i in range(len(held)):
        if held[i] > 0.5:
            if not inside:
                inside, growth, counted = True, 1.0, bool(rows[i])
            growth *= 1 + net[i]
        elif inside:
            growth *= 1 + net[i]  # the selling cost lands on the first day out
            inside = False
            if counted:
                count, lost = count + 1, lost + (growth < 1)
    if inside and counted:
        count, lost = count + 1, lost + (growth < 1)
    return count, int(lost)


def _btc_runs(close: pd.Series, keys=ORDER, cost: float = COST, delay: int = 1, buffer: float | None = None) -> dict:
    """Simulate the bitcoin variants over the whole history. buffer overrides A's band (robustness)."""
    target, band, fc = _targets(close)
    change = close.pct_change().values
    runs = {}
    for key in keys:
        if key == "hold":
            chg = np.nan_to_num(change, nan=0.0)
            runs[key] = {"net": chg, "held": np.ones(len(chg)), "trade": np.zeros(len(chg)), "next": 1.0}
            continue
        b = band[key].values if buffer is None or key != "A" else buffer
        runs[key] = _sim(target[key].values, b, change, cost, delay, short=(key == "ls"))
    runs["_target"], runs["_fc"] = target, fc
    return runs


def _basket(closes: dict) -> dict:
    """Equal weight across the coins that have a forecast on that day; each coin's sleeve follows its own rule and
    pays its own costs. Capital is put back to equal sleeves every day without cost (a simplification)."""
    index = pd.date_range(min(c.index[0] for c in closes.values()), max(c.index[-1] for c in closes.values()))
    methods = ("hold", "c20", "A", "sign")
    net = {m: pd.DataFrame(index=index, columns=list(closes), dtype=float) for m in methods}
    held = {m: pd.DataFrame(index=index, columns=list(closes), dtype=float) for m in methods}
    trade = {m: pd.DataFrame(index=index, columns=list(closes), dtype=float) for m in methods}
    ready_from, states, coin_runs = {}, {}, {}
    for coin, close in closes.items():
        target, band, fc = _targets(close, full=False)
        bars = np.arange(1, len(close) + 1)
        usable = fc["z"].notna() & (bars >= MIN_BARS)  # the decision made at this close may be used
        change = close.pct_change()
        active = usable.shift(1, fill_value=False) & change.notna()  # days on which the coin is in the basket
        if not active.any():
            continue
        ready_from[coin] = active.idxmax()
        runs = {}
        for m in ("c20", "A", "sign"):
            runs[m] = _sim(target[m].where(usable).values, band[m].values, change.values, COST)
        chg = change.fillna(0.0).values
        runs["hold"] = {"net": chg, "held": np.ones(len(chg)), "trade": np.zeros(len(chg))}
        for m in methods:
            for store, field in ((net, "net"), (held, "held"), (trade, "trade")):
                store[m].loc[close.index, coin] = np.where(active.values, runs[m][field], np.nan)
        coin_runs[coin] = {m: pd.Series(runs[m]["net"], index=close.index)[active] for m in methods}
        states[coin] = {"z": float(fc["z"].iloc[-1]) if _ok(fc["z"].iloc[-1]) else None,
                        "A": runs["A"]["next"], "c20": runs["c20"]["next"]}
    count = net["hold"].notna().sum(axis=1)
    out = {"count": count, "ready_from": ready_from, "states": states, "coin_runs": coin_runs, "series": {},
           "held_frame": held}
    for m in methods:
        out["series"][m] = (net[m].mean(axis=1).fillna(0.0), held[m].mean(axis=1).fillna(0.0),
                            trade[m].mean(axis=1).fillna(0.0))
    return out


# ---------- checks ----------

def _perturb(close: pd.Series, date: str, rng) -> pd.Series:
    changed = close.copy()
    after = changed.index > pd.Timestamp(date)
    changed[after] = changed[after].values * np.exp(np.cumsum(rng.normal(0.0, 0.06, int(after.sum()))))
    return changed


def _lookahead(close: pd.Series, runs: dict, closes: dict) -> dict:
    """Change every price after a date and recompute: the shares held up to the day after that date must not move."""
    rng = np.random.default_rng(SEED)
    keys = [k for k in ORDER if k != "hold"]
    tests = failures = 0
    for date in CHECK_DATES:
        if pd.Timestamp(date) >= close.index[-2]:
            continue
        again = _btc_runs(_perturb(close, date, rng), keys)
        rows = close.index <= pd.Timestamp(date) + pd.Timedelta(days=1)
        for key in keys:
            tests += 1
            if not np.allclose(runs[key]["held"][rows], again[key]["held"][rows], atol=1e-12, rtol=0.0):
                failures += 1
    sample = {c: closes[c] for c in ("ETH", "SOL") if c in closes}
    if sample:
        base = _basket(sample)
        for date in CHECK_DATES[2:4]:
            again = _basket({c: _perturb(s, date, rng) for c, s in sample.items()})
            limit = pd.Timestamp(date) + pd.Timedelta(days=1)
            for m in ("c20", "A", "sign"):
                tests += 1
                a = base["held_frame"][m][base["held_frame"][m].index <= limit].fillna(-9.0).values
                b = again["held_frame"][m][again["held_frame"][m].index <= limit].fillna(-9.0).values
                if a.shape != b.shape or not np.allclose(a, b, atol=1e-12, rtol=0.0):
                    failures += 1
    return {"tests": tests, "failures": failures, "dates": len(CHECK_DATES)}


def _zero_cross(close: pd.Series, z_now: float) -> dict | None:
    """The next daily close at which the combined forecast would change sign (searched between -60% and +150%)."""
    if not _ok(z_now) or z_now == 0:
        return None
    last = float(close.iloc[-1])
    day = close.index[-1] + pd.Timedelta(days=1)

    def z_at(move: float) -> float:
        trial = pd.concat([close, pd.Series([last * (1 + move)], index=[day])])
        return float(_forecast(trial)["z"].iloc[-1])

    lo, hi = (0.0, 1.5) if z_now < 0 else (-0.6, 0.0)
    far = hi if z_now < 0 else lo
    if (z_at(far) > 0) == (z_now > 0):
        return {"reachable": False, "limit": far * 100}
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if z_at(mid) > 0:
            hi = mid
        else:
            lo = mid
    move = 0.5 * (lo + hi)
    return {"reachable": True, "move": move * 100, "price": last * (1 + move)}


# ---------- everything that depends on the data ----------

def _compute(close: pd.Series, closes: dict) -> dict:
    index, last = close.index, close.index[-1]
    runs = _btc_runs(close)

    def parts(run: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
        return (pd.Series(run["net"], index=index), pd.Series(run["held"], index=index),
                pd.Series(run["trade"], index=index))

    def measure(p, start: str, end: str | None = None) -> dict:
        rows = (p[0].index >= pd.Timestamp(start)) & (p[0].index <= (pd.Timestamp(end) if end else last))
        return _measure(p[0][rows], p[1][rows], p[2][rows])

    series = {k: parts(runs[k]) for k in ORDER}
    periods = {name: {k: measure(series[k], start, end) for k in ORDER} for name, start, end in PERIODS}
    main = periods["all"]
    years = sorted(main["hold"]["years"])
    for k in ORDER:
        main[k]["beat"] = sum(1 for y in years if main[k]["years"][y] > main["hold"]["years"][y])
    rows = np.asarray(index >= pd.Timestamp(MAIN_START))
    recent_rows = np.asarray(index >= pd.Timestamp(PERIODS[3][1]))
    trades = {k: _trades(runs[k], rows) for k in ONOFF}
    trades_recent = {k: _trades(runs[k], recent_rows) for k in ONOFF}

    robust = {"buffer": {b: measure(parts(_btc_runs(close, ("A",), buffer=b)["A"]), MAIN_START) for b in BUFFERS},
              "cost": {c: measure(parts(_btc_runs(close, ("A",), cost=c)["A"]), MAIN_START) for c in COSTS},
              "delay": measure(parts(_btc_runs(close, ("A",), delay=2)["A"]), MAIN_START),
              "c20_delay": measure(parts(_btc_runs(close, ("c20",), delay=2)["c20"]), MAIN_START)}

    # pick in advance: every year the variant with the best return / volatility on the data before that year
    chained = pd.Series(0.0, index=index)
    chain_held, chain_trade = pd.Series(0.0, index=index), pd.Series(0.0, index=index)
    picks, previous = [], None
    for year in range(PICK_FIRST, last.year + 1):
        before = (index >= pd.Timestamp(PICK_FROM)) & (index < pd.Timestamp(f"{year}-01-01"))
        scores = {k: _score(runs[k]["net"][before])[2] for k in PICKS}
        choice = max(PICKS, key=lambda k: scores[k])
        this = np.asarray(index.year == year)
        if not this.any():
            continue
        chained[this] = series[choice][0][this]
        chain_held[this], chain_trade[this] = series[choice][1][this], series[choice][2][this]
        first = int(np.argmax(this))
        if previous is not None and previous != choice:  # moving from the old variant's holding to the new one's
            switch = abs(runs[choice]["held"][first] - runs[previous]["held"][first])
            chained.iloc[first] -= COST * switch
            chain_trade.iloc[first] += switch
        picks.append((year, choice, scores[choice]))
        previous = choice
    pick_start = f"{PICK_FIRST}-01-01"
    pick = {"picked": measure((chained, chain_held, chain_trade), pick_start),
            "fixed": {k: measure(series[k], pick_start) for k in PICKS + ("hold",)}, "picks": picks}

    basket = None
    if closes:
        b = _basket(closes)
        blast = b["series"]["hold"][0].index[-1]

        def bmeasure(m: str, start: str, end: str | None) -> dict:
            net, held, trade = b["series"][m]
            keep = (net.index >= pd.Timestamp(start)) & (net.index <= (pd.Timestamp(end) if end else blast))
            return _measure(net[keep], held[keep], trade[keep])

        table = {name: {m: bmeasure(m, start, end) for m in b["series"]} for name, start, end in PERIODS}
        wins = {}
        for name, start, end in (("all", MAIN_START, None), ("recent", PERIODS[3][1], None)):
            used = better_c20 = better_hold = 0
            for coin, nets in b["coin_runs"].items():
                cut = {m: s[(s.index >= pd.Timestamp(start))] for m, s in nets.items()}
                if len(cut["A"]) < YEAR:
                    continue
                used += 1
                ratio = {m: _score(s.values)[2] for m, s in cut.items()}
                better_c20 += ratio["A"] > ratio["c20"]
                better_hold += ratio["A"] > ratio["hold"]
            wins[name] = {"n": used, "c20": int(better_c20), "hold": int(better_hold)}
        count = b["count"]
        basket = {"table": table, "wins": wins, "last": blast, "coins": len(b["ready_from"]),
                  "start_count": int(count[count.index >= pd.Timestamp(MAIN_START)].iloc[0]),
                  "end_count": int(count.iloc[-1]), "states": b["states"],
                  "recent_count": int(count[count.index >= pd.Timestamp(PERIODS[3][1])].iloc[0])}

    check = _lookahead(close, runs, closes)
    fc = runs["_fc"]
    return {"last": last, "close": close, "runs": runs, "periods": periods, "years": years, "trades": trades,
            "trades_recent": trades_recent, "robust": robust, "pick": pick, "basket": basket, "check": check,
            "fc": fc, "target": runs["_target"], "cross": _zero_cross(close, float(fc["z"].iloc[-1]))}


# ---------- the study ----------

def study() -> dict:
    close = _daily_close()
    try:
        closes, problems = _coin_closes()
    except Exception as exc:  # noqa: BLE001 - the bitcoin part still stands without the basket
        closes, problems = {}, [f"籃子的資料這次取不到（{type(exc).__name__}），只有比特幣的部分。"]
    r = _compute(close, closes)
    if r["check"]["failures"]:
        raise RuntimeError(f"偷看未來的檢查沒有通過：{r['check']['failures']} / {r['check']['tests']}")
    last, periods, years = r["last"], r["periods"], r["years"]
    main, recent = periods["all"], periods["recent"]
    n = len(years)
    current = years[-1]
    partial = last < pd.Timestamp(f"{current}-12-31")
    recent_label = f"2024-01 到 {_d(last)}"

    def under(m: dict) -> str:
        return f"{m['under']:,} 天" + ("（還沒回到）" if m["under_open"] else "")

    def held(key: str, m: dict) -> str:
        share = f"淨 {m['share']:+.0f}%" if key == "ls" else f"{m['share']:.0f}%"
        return f"{share}｜{m['turnover']:.1f} 次"

    def trade_text(key: str) -> str:
        if key not in r["trades"]:
            return "—"
        count, lost = r["trades"][key]
        return f"{count} 次｜{lost / count * 100:.0f}% 賠" if count else "0 次"

    # ----- table 1: bitcoin variants over the main period -----
    rows1 = []
    for k in ORDER:
        m = main[k]
        rows1.append([NAMES[k], _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]),
                      f"{_pct(m['deepest'])}｜{under(m)}", "—" if k == "hold" else f"{n} 年裡 {m['beat']} 年",
                      held(k, m), trade_text(k)])

    # ----- table 2: by period and by year -----
    keys2 = ("hold", "c20", "blend", "blend40", "A", "B40", "sign")

    def cell(m: dict) -> str:
        return f"{_pct(m['ret'])}｜{_num(m['ratio'])}"

    rows2 = [[label] + [cell(periods[name][k]) for k in keys2]
             for name, label in (("all", f"2018-01 到 {_d(last)}"), ("p1", "2018–2021"), ("p2", "2022–2023"),
                                 ("recent", recent_label))]
    for y in years:
        label = f"{y}（到 {last.strftime('%m-%d')}）" if y == current and partial else str(y)
        rows2.append([label] + [f"{_pct(main[k]['years'][y])}｜{_num(main[k]['year_ratio'][y])}" for k in keys2])
    rows2 = rows2[:16]

    # ----- table 3: basket -----
    tables = [
        {"caption": (f"比特幣，只做多或抱現金（最後一列例外），2018-01 到 {_d(last)}，{n} 個年度"
                     f"{'（最後一年還沒過完）' if partial else ''}。每種做法都是當天收盤決定、隔天才照做，每次買賣扣交易金額的 "
                     f"{COST * 100:.2f}%，現金沒有利息，不借錢放大。「平均持有｜一年換手」＝平均放幾成在比特幣｜一年內買賣的總金額是全部資金的幾倍。"
                     "最後一欄只有「全進全出」的做法才有：進出幾輪｜其中幾成是賠的。"),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少｜最久沒回到高點", "賺得比一直抱著多的年數",
                     "平均持有｜一年換手", "進出次數｜賠的比例"],
         "rows": rows1},
        {"caption": ("分段和逐年（比特幣）。每格是「報酬｜報酬÷波動」：分段的報酬換算成每年，年度列是那一年實際的漲跌"
                     f"{'（今年只算到最後一根日線）' if partial else ''}，報酬÷波動一律換算成一年。同一套規則一路做下來，沒有分段重新調整。"),
         "columns": ["期間"] + [SHORT[k] for k in keys2],
         "rows": rows2},
    ]
    bk = r["basket"]
    basket_names = {"hold": "一直抱著整個籃子", "c20": "每種幣各自 20/10 簡單進出", "A": "每種幣各自 A 連續加減碼",
                    "sign": "每種幣各自只看正負"}
    if bk:
        rows3 = []
        for name, label in (("all", f"2018-01 到 {_d(bk['last'])}"), ("p1", "2018–2021"), ("p2", "2022–2023"),
                            ("recent", f"2024-01 到 {_d(bk['last'])}")):
            for m in ("hold", "c20", "A", "sign"):
                v = bk["table"][name][m]
                rows3.append([basket_names[m], label, _pct(v["ret"]), _pct(v["vol"], signed=False), _num(v["ratio"]),
                              _pct(v["deepest"]), f"{v['share']:.0f}%", f"{v['turnover']:.1f} 次"])
        w_all, w_new = bk["wins"]["all"], bk["wins"]["recent"]
        tables.append({
            "caption": (f"籃子：{bk['coins']} 種現在的大幣（幣安現貨），每種幣從它有足夠資料那天起加入，資金平均分給當天在籃子裡的幣，"
                        f"每種幣照自己的訊號決定持有幾成、其餘是現金。2018 年初籃子裡只有 {bk['start_count']} 種幣，2024 年初 "
                        f"{bk['recent_count']} 種，現在 {bk['end_count']} 種。這份名單是用今天的排名挑的（倖存者），已經歸零或下市的幣不在裡面。"
                        f"逐一看每種幣：整段期間 {w_all['n']} 種裡 {w_all['c20']} 種的 A 報酬÷波動高於 20/10、{w_all['hold']} 種高於一直抱著；"
                        f"2024 年以後 {w_new['n']} 種裡 {w_new['c20']} 種高於 20/10、{w_new['hold']} 種高於一直抱著。"),
            "columns": ["做法", "期間", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "平均持有", "一年換手幾次"],
            "rows": rows3[:16]})

    # ----- table 4: robustness and pick-in-advance -----
    rb, pk = r["robust"], r["pick"]

    def rrow(name: str, m: dict) -> list:
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['turnover']:.1f} 次"]

    rows4 = [rrow("A 原本的設定（緩衝區 10 個百分點、成本 0.10%、隔天照做）", main["A"])]
    rows4 += [rrow(f"A：緩衝區改成 {b * 100:.0f} 個百分點" + ("（沒有緩衝區，每天調到目標）" if b == 0 else ""), rb["buffer"][b])
              for b in BUFFERS if b != BUFFER]
    rows4 += [rrow(f"A：成本改成 {c * 100:.2f}%", rb["cost"][c]) for c in COSTS if c != COST]
    rows4 += [rrow("A：晚一天才照做", rb["delay"]), rrow("對照：20/10 簡單進出晚一天才照做", rb["c20_delay"]),
              rrow("對照：20/10 簡單進出原本的設定", main["c20"])]
    pick_span = f"{PICK_FIRST}-01 到 {_d(last)}"
    rows4 += [rrow(f"每年事先挑一種做法（{pick_span}）", pk["picked"]),
              rrow(f"一直用 20/10 簡單進出（{pick_span}）", pk["fixed"]["c20"]),
              rrow(f"一直用 A 連續加減碼（{pick_span}）", pk["fixed"]["A"]),
              rrow(f"一直抱著（{pick_span}）", pk["fixed"]["hold"])]
    pick_words = "、".join(f"{y} 年用「{SHORT[k]}」" for y, k, _ in pk["picks"])
    tables.append({
        "caption": (f"上半：A 經不經得起改動（比特幣，2018-01 到 {_d(last)}）。下半：事先挑選的測試——從 {PICK_FIRST} 年起，每年年初只看那一年"
                    f"以前（{PICK_FROM[:4]} 年起）的資料，在「20/10 簡單進出、17 條混合、A、B（目標 40%）、只看正負」五種裡挑報酬÷波動最高的，"
                    f"那一年就照它做，一年一年接起來。實際挑到的是：{pick_words}。"),
        "columns": ["項目", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "一年換手幾次"],
        "rows": rows4[:16]})

    # ----- today -----
    fc, runs = r["fc"], r["runs"]
    z_now = float(fc["z"].iloc[-1])

    def stake(v) -> str:
        if not _ok(v):
            return "—"
        if v < -0.005:
            return f"做空 {abs(v) * 100:.0f}%"
        return f"持有比特幣 {v * 100:.0f}%" if v > 0.005 else "全部現金"

    cross = r["cross"]
    if cross is None:
        cross_text = "—"
    elif not cross["reachable"]:
        cross_text = (f"下一根日線就算{'漲' if z_now < 0 else '跌'} {abs(cross['limit']):.0f}% 也還不會"
                      f"{'轉正' if z_now < 0 else '轉負'}")
    else:
        cross_text = (f"下一根日線收盤要{'高於' if z_now < 0 else '低於'}約 {_usd(cross['price'])}"
                      f"（比最後收盤 {_pct(cross['move'])}），趨勢強度才會{'轉正' if z_now < 0 else '轉負'}；這個價位每天都會變")
    pair_text = "、".join(f"{fast}／{slow} 日 {_num(float(fc[f'z{slow}'].iloc[-1]))}" for fast, slow in PAIRS)
    c20_now = runs["c20"]["next"]
    flip = float(close.iloc[-9:].min()) if c20_now > 0.5 else float(close.iloc[-19:].max())
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {_usd(float(close.iloc[-1]))}"},
        {"label": "現在的趨勢強度（三組平均；正的代表快線在慢線上面）",
         "value": f"{_num(z_now)}（{pair_text}）；換算成 A 想持有的比例是 {max(0.0, math.tanh(z_now)) * 100:.0f}%"},
        {"label": "趨勢強度要變號，價格得走到哪", "value": cross_text},
        {"label": f"最近 {SIZE_VOL_DAYS} 天的波動（換算成一年）",
         "value": f"{_pct(float(fc['vol30'].iloc[-1]) * 100, signed=False)}；波動調整的倍數：目標 40% 是 "
                  f"{_num(float(fc['scale40'].iloc[-1]))}、目標 25% 是 {_num(float(fc['scale25'].iloc[-1]))}"},
        {"label": "照規則下一天的狀態：連續加減碼",
         "value": f"A：{stake(runs['A']['next'])}；B 目標 40%：{stake(runs['B40']['next'])}；B 目標 25%：{stake(runs['B25']['next'])}"},
        {"label": "照規則下一天的狀態：單組與只看正負",
         "value": "；".join(f"{fast}／{slow} 日：{stake(runs[f'p{slow}']['next'])}" for fast, slow in PAIRS)
                  + f"；只看正負：{stake(runs['sign']['next'])}"},
        {"label": "照規則下一天的狀態：多空版", "value": stake(runs["ls"]["next"])},
        {"label": "照規則下一天的狀態：對照的規則",
         "value": f"20/10 簡單進出：{stake(c20_now)}（收盤{' ≤ ' if c20_now > 0.5 else ' ≥ '}{_usd(flip)} 才會"
                  f"{'出場' if c20_now > 0.5 else '進場'}）；17 條混合：{stake(runs['blend']['next'])}；"
                  f"混合×波動 40%：{stake(runs['blend40']['next'])}"},
    ]
    if bk:
        states = bk["states"]
        positive = sum(1 for s in states.values() if _ok(s["z"]) and s["z"] > 0)
        in_c20 = sum(1 for s in states.values() if s["c20"] > 0.5)
        now.append({"label": f"籃子裡 {len(states)} 種幣現在的狀態",
                    "value": f"趨勢強度是正的有 {positive} 種；20/10 簡單進出是持有的有 {in_c20} 種；"
                             f"A 平均持有 {np.mean([s['A'] for s in states.values()]) * 100:.0f}%"})

    # ----- verdict -----
    a, c, h, s, ls = main["A"], main["c20"], main["hold"], main["sign"], main["ls"]
    tc, tl = r["trades"]["c20"]
    ts, tsl = r["trades"]["sign"]
    better_recent = [k for k in TREND if recent[k]["ratio"] > recent["hold"]["ratio"]]
    richer_recent = [k for k in TREND if recent[k]["growth"] > recent["hold"]["growth"]]
    if a["ratio"] > c["ratio"] + 0.03 and a["ret"] > c["ret"]:
        who = "連續加減碼的報酬和報酬÷波動都比簡單進出高"
    elif a["ratio"] < c["ratio"] - 0.03 and a["ret"] < c["ret"]:
        who = "連續加減碼的報酬和報酬÷波動都比簡單進出低，沒有贏"
    else:
        who = (f"連續加減碼的每年報酬比簡單進出{_cmp(a['ret'], c['ret'], 0.5, '高', '低')}、"
               f"報酬÷波動{_cmp(a['ratio'], c['ratio'], 0.03, '比較高', '比較低')}")
    verdict = (
        f"比特幣只做多或抱現金，{years[0]} 到 {_d(last)}：照趨勢強度連續加減碼（A）扣成本後每年 {_pct(a['ret'])}、報酬÷波動 "
        f"{_num(a['ratio'])}、最深曾經虧 {_pct(a['deepest'])}；簡單的「創 20 日新高進場、10 日新低出場」是 {_pct(c['ret'])}、"
        f"{_num(c['ratio'])}、{_pct(c['deepest'])}；一直抱著是 {_pct(h['ret'])}、{_num(h['ratio'])}、{_pct(h['deepest'])}——{who}。"
        f"進出次數：簡單進出 {tc} 輪（{tl / max(tc, 1) * 100:.0f}% 是賠的）、一年換手 {c['turnover']:.1f} 次；只看趨勢強度正負的版本 "
        f"{ts} 輪（{tsl / max(ts, 1) * 100:.0f}% 是賠的）、每年 {_pct(s['ret'])}、報酬÷波動 {_num(s['ratio'])}；A 是分批加減，一年換手 "
        f"{a['turnover']:.1f} 次；加上做空的版本每年 {_pct(ls['ret'])}、報酬÷波動 {_num(ls['ratio'])}（沒算做空的費用）。"
        f"2024 年以後（{recent_label}）：一直抱著每年 {_pct(recent['hold']['ret'])}、報酬÷波動 {_num(recent['hold']['ratio'])}，"
        f"簡單進出 {_pct(recent['c20']['ret'])}／{_num(recent['c20']['ratio'])}，17 條混合 {_pct(recent['blend']['ret'])}／"
        f"{_num(recent['blend']['ratio'])}，A {_pct(recent['A']['ret'])}／{_num(recent['A']['ratio'])}，只看正負 "
        f"{_pct(recent['sign']['ret'])}／{_num(recent['sign']['ratio'])}；{len(TREND)} 種只做多的順勢做法裡 {len(better_recent)} 種"
        f"報酬÷波動高於一直抱著、{len(richer_recent)} 種最後的錢比一直抱著多。"
    )
    if bk:
        ba, bn = bk["table"]["all"], bk["table"]["recent"]
        verdict += (
            f"{bk['coins']} 種大幣的籃子（名單是倖存者）：整段期間 A 每年 {_pct(ba['A']['ret'])}、報酬÷波動 {_num(ba['A']['ratio'])}，"
            f"各自簡單進出 {_pct(ba['c20']['ret'])}／{_num(ba['c20']['ratio'])}，抱著整個籃子 {_pct(ba['hold']['ret'])}／"
            f"{_num(ba['hold']['ratio'])}；2024 年以後依序是 {_pct(bn['A']['ret'])}／{_num(bn['A']['ratio'])}、"
            f"{_pct(bn['c20']['ret'])}／{_num(bn['c20']['ratio'])}、{_pct(bn['hold']['ret'])}／{_num(bn['hold']['ratio'])}。")

    # ----- notes -----
    check = r["check"]
    notes = [
        "「趨勢強度」＝快慢兩條平均線的差距除以近期波動：把價格取對數後各算一條反應快、一條反應慢的指數平均線（16／48、32／96、64／192 日"
        "三組），快線減慢線，再除以「每天的波動 × 慢線天數的平方根」；每一組再除以它自己過去 365 天的起伏（至少 60 天）讓三組大小可以相比，"
        "三組取平均。正的代表最近的價格在較長期的平均之上，越大代表趨勢相對於波動越明顯。「每天的波動」用的是：最近 60 天每日漲跌"
        "（對數）的標準差占七成、全部歷史（最多十年）的標準差占三成，而且不低於年化 20%；全部只用到當天收盤為止的資料。",
        "各種做法：A＝把趨勢強度用 tanh 壓到 −1 到 1 之間，負的當成 0，得到 0 到 100% 的目標持有比例；「緩衝區」＝差距不夠大就不調整——"
        "手上的比例和目標差不到 10 個百分點就不動，超過才調，而且只調到緩衝區的邊緣，不是一次調到目標；目標是 0 的時候例外，直接全部換成現金。"
        "B＝A 再乘上「目標波動 ÷ 最近 30 天的波動」（最多乘 1），緩衝區也跟著等比例縮小。單組＝只用三組裡的一組。只看正負＝趨勢強度大於 0 就"
        "全部持有、否則全部現金。多空版＝負的不當成 0，照樣做空，最多 100%。",
        "公式是從一個開源的多空系統（OctopusTakopi/crypto-trend-following，對象是幣安 700 多個永續合約、可以做空、整個組合有波動目標）"
        "改成「只做多或抱現金、單一資產、現貨收盤價」，所以這裡的結果不能和它公布的數字相比。它自己公布的是：整個組合扣成本後報酬÷波動 "
        "1.04（2020-01 到 2026-07），其中 2024 年以後 1.54；只看正負的版本 1.15；單一合約的中間值只有 0.10。這些是它的說法，本頁沒有重做。"
        "和它不同的地方：它用隔天開盤價成交並計入資金費，這裡用收盤價、沒有資金費；它沒有說明指數平均線的算法細節，這裡用一般的定義"
        "（權重 2 ÷（天數 + 1））；目標是 0 就全部出場是本頁為了「只做多或抱現金」加的。",
        "對照的規則照本頁其他研究的寫法重做：20/10 簡單進出＝收盤創 20 日新高後持有、創 10 日新低後換成現金；17 條混合＝17 條簡單的順勢規則"
        "裡幾成說持有就持有幾成；混合×波動 40%＝再乘上「40% ÷ 最近 30 天的波動」（最多乘 1）。",
        f"偷看未來的檢查：把某一天之後的價格全部改掉再重算，那一天和隔天以前的持有比例必須完全一樣。這次實際跑了 {check['tests']} 組"
        f"（比特幣的每種做法 × {check['dates']} 個日期，加上籃子裡兩種幣），有 {check['failures']} 組不一樣。",
        "時間點：訊號用每天 UTC 0 點的收盤價算，並假設隔天整天就照新的比例持有，等於假設剛好能用收盤價成交；實際上會有幾分鐘到幾小時的落差"
        "（第四張表有晚一天才照做的結果）。新高新低用的是收盤價，不是盤中的最高最低價。",
        f"成本只用一個粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%。實際的手續費、買賣價差和滑價會隨交易所、金額、幣種和年代不同，"
        "小幣早年尤其貴。現金假設沒有利息，也沒有算稅。多空版沒有算做空要付的資金費率或借幣利息，也沒有算被強制平倉的風險，數字比實際能做到的好看。",
        "籃子的名單是今天還活著、還排在前面的幣（倖存者），已經歸零或下市的幣不在裡面：一直抱著的數字因此偏好看，而規則在一路跌到歸零的幣上"
        "本來可以早早出場的例子也被漏掉了，兩邊都偏。籃子假設每天把資金免費調回各幣等分（只算每種幣自己進出的成本），新幣加入時的調整也沒算成本；"
        f"每種幣要有至少 {MIN_BARS} 根日線、算得出趨勢強度才加入，三種做法用的是同一批幣、同一段日子。比特幣在籃子裡用幣安 2017-08 起的資料。",
        f"獨立的年份很少：主要期間只有 {n} 個年度，大致就是兩三輪大漲大跌，各種幣又是一起漲一起跌；2024 年以後更只有不到三年。"
        "做法之間的小差距、「幾年裡幾年」這類數字，多一年少一年就可能翻過來，不能當成確定的結論。事先挑選的測試在年初換做法時，"
        "只多扣了兩種做法當天持有比例差距的成本。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「每年波動」是每天漲跌的起伏換算成一年；「報酬÷波動」是兩者相除；"
        "「最深曾經虧多少」是從最高點往下掉最多的幅度；「最久沒回到高點」是資金從一個高點到重新回到那個高點最長的天數。"
        "比特幣的價格 2017-08 以前是 blockchain.info 的每日價格，之後是幣安現貨日線。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "ewma_forecast",
        "title": "照趨勢強度連續加減碼，比簡單的進出規則好嗎",
        "question": "專業順勢交易常用的做法——照「趨勢強度」連續加減碼、再加一個緩衝區避免小幅來回調整——有沒有比本頁簡單的"
                    "「創新高進場、創新低出場」更好？只看比特幣、只做多或抱現金時如何，換成一籃子大幣又如何，特別是 2024 年以後？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "OctopusTakopi/crypto-trend-following（公式的出處，開源專案的說明文件）",
             "url": "https://github.com/OctopusTakopi/crypto-trend-following"},
            {"name": "blockchain.info 每日比特幣價格（2010 年起）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線（BTC 與其他 15 種幣對 USDT）", "url": "https://www.binance.com/zh-TC/markets/overview"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
