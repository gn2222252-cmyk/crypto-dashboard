"""Historical study "vol_sizing": how much of what trend-following achieves on bitcoin can be had just by sizing the
position to volatility - or by simply holding a smaller fixed amount - with no market timing at all?

1. Fixed share: always 25 / 40 / 50 / 60 / 75% bitcoin, put back to the share on the first day of every month.
2. Volatility sizing alone: share = min(1, target / recent yearly volatility). Recent volatility is the 30-day
   realised one (also 60-day, an exponentially weighted one and a HAR forecast whose coefficients are re-estimated
   every day on an expanding window of past data only). The position is only traded when the wanted share is more
   than 10% (relative) away from the share actually held (one variant trades every day).
3. Trend only (trend_ensemble's 17-rule blend) and trend x volatility sizing (40% and 30% targets), exactly as
   trend_ensemble computes them (reset to target every close).
4. The fair comparison: for every timing approach the FIXED share with the same yearly volatility over the period
   (and, separately, the same deepest loss) and what that fixed share returned.
5. What each signal predicts: days sorted by 30-day volatility (ranked against earlier history only) and by the
   blend's share held; the next 30 days' return, how often they lost money, the worst fall, the realised volatility.
6. Eight other coins from their first Binance day: volatility sizing at a 40% target against holding, against the
   same-volatility fixed share and against the blend.
7. A luck check: the share series circularly shifted 1,000 times.

No look-ahead: everything decided at the close of day t is the position during day t+1. Costs are 0.10% of the
amount traded. Only closed daily candles from public, key-less endpoints. The other coins' closes are kept in
cache/_store/vol_sizing_klines.json (first filled from the candles ewma_forecast / donchian_ensemble already stored).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import trend_ensemble as te

COST = 0.001
YEAR = 365
FULL_START = "2014-01-01"
MAIN_START = "2018-01-01"
PERIODS = (("full", FULL_START), ("main", MAIN_START), ("p22", "2022-01-01"), ("p24", "2024-01-01"))
FIXED = (0.25, 0.40, 0.50, 0.60, 0.75)
TARGETS = (0.20, 0.30, 0.40, 0.60, 0.80)
MAIN_TARGET = 0.40
BAND = 0.10  # trade only when the wanted share is more than 10% (relative) away from the share held
LAMBDA = 0.94  # exponentially weighted variance: yesterday's estimate keeps 94% of its weight
HAR_START = "2012-01-01"
HAR_MIN = 365  # past days needed before the first forecast
HAR_FLOOR = 0.10  # a forecast is never below 10% of the average variance seen so far
AHEAD = 30
SHARES = np.linspace(0.0, 1.0, 201)  # the fixed shares tried when matching risk
SIMS = 1000
SEED = 20180101
COINS = (("ETH", "以太幣"), ("BNB", "幣安幣"), ("XRP", "瑞波幣"), ("ADA", "艾達幣"), ("DOGE", "狗狗幣"),
         ("SOL", "索拉納"), ("LTC", "萊特幣"), ("LINK", "Chainlink"))
STORE_FILE = "vol_sizing_klines.json"
SEEDS = (("ewma_forecast_klines.json", "dict"), ("donchian_ensemble_klines.json", "rows"))  # read only
DAY_MS = 86_400_000

_pct, _num, _d, _ok = te._pct, te._num, te._d, te._ok


def _vkey(target: float) -> str:
    return f"v{int(round(target * 100))}"


VOL_KEYS = tuple(_vkey(t) for t in TARGETS)
MAIN_KEY = _vkey(MAIN_TARGET)
KEYS = VOL_KEYS + (MAIN_KEY + "d", "w60", "ew", "har", "blend", "bv40", "bv30")
DAILY = (MAIN_KEY + "d", "blend", "bv40", "bv30")  # reset to target every close (as trend_ensemble does)
NAMES = {
    "hold": "一直抱著不動",
    MAIN_KEY + "d": "依波動調整（目標 40%，30 天波動），每天都調",
    "w60": "依波動調整（目標 40%，改看 60 天波動）",
    "ew": "依波動調整（目標 40%，近期加重的波動）",
    "har": "依波動調整（目標 40%，用 HAR 預測的明天波動）",
    "blend": "只看順勢：17 條規則混合",
    "bv40": "順勢混合 × 依波動調整（目標 40%）",
    "bv30": "順勢混合 × 依波動調整（目標 30%）",
}
for _t in TARGETS:
    NAMES[_vkey(_t)] = f"依波動調整（目標 {_t * 100:.0f}%，30 天波動）"


# ---------- data ----------

def _read_json(path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a damaged store file starts over
        return {}


def _seed(symbol: str, cache: dict) -> dict:
    """Closes another study already stored for this symbol (read only), so the history is not downloaded again."""
    for name, kind in SEEDS:
        if name not in cache:
            cache[name] = _read_json(common.STORE / name)
        got = cache[name].get(symbol)
        days = {}
        if kind == "dict" and isinstance(got, dict):
            days = {str(k)[:10]: float(v) for k, v in got.items() if _ok(v) and v > 0}
        elif kind == "rows" and isinstance(got, list):
            for row in got:
                if isinstance(row, list) and len(row) == 4 and _ok(row[3]) and row[3] > 0:
                    days[pd.Timestamp(int(row[0]), unit="ms").strftime("%Y-%m-%d")] = float(row[3])
        if days:
            return days
    return {}


def _coin_closes() -> tuple[dict, list[str]]:
    """Finished Binance spot daily closes per coin from its first listed day; a later run only asks for new days."""
    path = common.STORE / STORE_FILE
    stored = _read_json(path)
    today = common.today_utc()
    closes, problems, cache = {}, [], {}
    for coin, _ in COINS:
        symbol = f"{coin}USDT"
        got = stored.get(symbol)
        days = {str(k)[:10]: float(v) for k, v in got.items() if _ok(v) and v > 0} if isinstance(got, dict) else {}
        if not days:
            days = _seed(symbol, cache)
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
                problems.append(f"{coin} 這次取不到資料（{type(exc).__name__}），沒有算進其他幣的比較。")
                continue
            problems.append(f"{coin} 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的日線。")
        days = {day: v for day, v in days.items() if day < today and v > 0}  # never today's unfinished candle
        if len(days) < 600:
            problems.append(f"{coin} 的日線只有 {len(days)} 筆，沒有算進其他幣的比較。")
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


def _tbill() -> tuple[pd.Series | None, str, str | None]:
    """3-month US Treasury bill rate the way robust_trend gets it (FRED, then the Treasury's own tables, then 0%)."""
    try:
        from dashboard.studies2 import robust_trend as rt
        return rt._tbill()
    except Exception as exc:  # noqa: BLE001 - the study still runs with 0% interest
        return None, "none", f"美國三個月國庫券利率這次拿不到（{type(exc).__name__}），「現金有利息」的幾列用 0% 計算，因此和沒有利息相同。"


# ---------- volatility forecasts ----------

def _har(close: pd.Series) -> pd.DataFrame:
    """HAR forecast of the next day's variance (and of the next 5 days' average), made at each day's close.

    Variance proxy v = squared daily log return (in percent squared). Regressors known at the close of day t: v of
    day t, the average of the last 7 days, the average of the last 30 days. The coefficients used on day t come from
    least squares on every earlier day whose outcome is already known on day t (an expanding window)."""
    c = close[close.index >= pd.Timestamp(HAR_START)]
    v = (np.log(c).diff() * 100) ** 2
    week, month = v.rolling(7, min_periods=7).mean(), v.rolling(30, min_periods=30).mean()
    past = v.expanding(min_periods=30).mean()
    n = len(c)
    x = np.column_stack([np.ones(n), v.values, week.values, month.values])
    out = pd.DataFrame({"naive": month, "past": past}, index=c.index)
    for h in (1, 5):
        y = v.rolling(h, min_periods=h).mean().shift(-h).values  # average of days t+1 .. t+h
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        xo, yo = np.where(ok[:, None], x, 0.0), np.where(ok, y, 0.0)
        a = np.cumsum(xo[:, :, None] * xo[:, None, :], axis=0)
        b = np.cumsum(xo * yo[:, None], axis=0)
        count = np.cumsum(ok)
        beta = np.full((n, 4), np.nan)
        for t in range(h, n):
            s = t - h  # the last day whose outcome is fully known at the close of day t
            if count[s] >= HAR_MIN:
                try:
                    beta[t] = np.linalg.solve(a[s], b[s])
                except np.linalg.LinAlgError:
                    pass
        raw = (beta * x).sum(axis=1)
        out[f"f{h}"] = np.maximum(raw, HAR_FLOOR * past.values)
        out[f"y{h}"] = y
        if h == 1:
            for i in range(4):
                out[f"b{i}"] = beta[:, i]
    return out.reindex(close.index)


def _har_score(har: pd.DataFrame, start: str) -> dict:
    """Out-of-sample fit from `start` on: against the naive 'same as the last 30 days', and each against 'the
    average of everything so far'."""
    part = har[har.index >= pd.Timestamp(start)]
    out = {}
    for h in (1, 5):
        rows = part[[f"f{h}", f"y{h}", "naive", "past"]].dropna()
        y = rows[f"y{h}"].values
        sse = float(((y - rows[f"f{h}"].values) ** 2).sum())
        naive = float(((y - rows["naive"].values) ** 2).sum())
        base = float(((y - rows["past"].values) ** 2).sum())
        out[h] = {"n": len(rows), "vs_naive": (1 - sse / naive) * 100 if naive > 0 else None,
                  "har": (1 - sse / base) * 100 if base > 0 else None,
                  "naive": (1 - naive / base) * 100 if base > 0 else None}
    return out


def _vols(close: pd.Series, with_har: bool = True) -> pd.DataFrame:
    """Yearly volatility estimates known at each day's close (fractions, 0.6 = 60%)."""
    change = close.pct_change()
    out = pd.DataFrame(index=close.index)
    out["d30"] = change.rolling(30, min_periods=30).std(ddof=1) * math.sqrt(YEAR)
    out["d60"] = change.rolling(60, min_periods=60).std(ddof=1) * math.sqrt(YEAR)
    out["ew"] = np.sqrt((change ** 2).ewm(alpha=1 - LAMBDA, min_periods=30).mean() * YEAR)
    if with_har:
        out["har"] = np.sqrt(_har(close)["f1"] * YEAR) / 100
    return out


def _size(vol: pd.Series, target: float) -> pd.Series:
    """min(1, target / volatility), exactly as trend_ensemble scales its blend."""
    return (target / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())


def _wants(close: pd.Series, with_har: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The share every approach wants, decided at each day's close (before the one-day delay), and the volatilities."""
    vols = _vols(close, with_har)
    blend = te._targets(close)[0]["blend"]
    out = pd.DataFrame(index=close.index)
    for t in TARGETS:
        out[_vkey(t)] = _size(vols["d30"], t)
    out[MAIN_KEY + "d"] = out[MAIN_KEY]
    out["w60"] = _size(vols["d60"], MAIN_TARGET)
    out["ew"] = _size(vols["ew"], MAIN_TARGET)
    if with_har:
        out["har"] = _size(vols["har"], MAIN_TARGET)
    out["blend"] = blend
    out["bv40"] = blend * _size(vols["d30"], 0.40)
    out["bv30"] = blend * _size(vols["d30"], 0.30)
    return out, vols


# ---------- simulating ----------

def _sim(target, change: np.ndarray, band=0.0, rebalance: np.ndarray | None = None, rf: np.ndarray | None = None,
         cost: float = COST) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily net return, amount traded and share held for every column of `target` (the share wanted DURING each
    day, i.e. already delayed by a day). Between trades the share drifts with the price. A column is traded to its
    target when the gap is more than band x the share held (band 0 = every close), or - if `rebalance` is given -
    only on the flagged days. Idle cash earns rf (a daily rate) when given. Starts from cash."""
    target = np.nan_to_num(np.asarray(target, dtype=float), nan=0.0)
    if target.ndim == 1:
        target = target[:, None]
    change = np.nan_to_num(np.asarray(change, dtype=float), nan=0.0)
    n, m = target.shape
    band = np.broadcast_to(np.asarray(band, dtype=float), (m,))
    net, trade, held = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))
    w = np.zeros(m)
    for i in range(n):
        t = target[i]
        if rebalance is not None:
            new = t if (i == 0 or rebalance[i]) else w
        else:
            new = np.where(np.abs(t - w) > band * w, t, w)
        r = change[i]
        idle = (1 - new) * rf[i] if rf is not None else 0.0
        moved = np.abs(new - w)
        net[i], trade[i], held[i] = new * r + idle - cost * moved, moved, new
        w = new * (1 + r) / (1 + new * r + idle)
    return net, trade, held


def _stats(net: np.ndarray) -> dict:
    """Yearly compound return %, yearly volatility %, return / volatility, deepest loss % of every column."""
    net = np.asarray(net, dtype=float)
    flat = net.ndim == 1
    if flat:
        net = net[:, None]
    n = len(net)
    growth = np.prod(1 + net, axis=0)
    ret = np.where(growth > 0, np.abs(growth) ** (YEAR / max(n, 1)) - 1, -1.0) * 100
    vol = np.std(net, axis=0, ddof=1) * math.sqrt(YEAR) * 100 if n > 1 else np.zeros(net.shape[1])
    ratio = np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)
    equity = np.vstack([np.ones(net.shape[1]), np.cumprod(1 + net, axis=0)])
    deepest = (equity / np.maximum.accumulate(equity, axis=0) - 1).min(axis=0) * 100
    out = {"ret": ret, "vol": vol, "ratio": ratio, "deepest": deepest}
    return {k: float(v[0]) for k, v in out.items()} if flat else out


def _one(net: np.ndarray, trade: np.ndarray | None = None, held: np.ndarray | None = None) -> dict:
    out = _stats(net)
    out["days"] = len(net)
    out["share"] = float(np.mean(held) * 100) if held is not None else 100.0
    out["turnover"] = float(np.sum(trade) / (len(net) / YEAR)) if trade is not None and len(net) else 0.0
    return out


def _match(value: float, by: str, grid: dict) -> dict:
    """The fixed share whose yearly volatility (by='vol') or deepest loss (by='deepest') equals `value`, and what
    that fixed share did (interpolated between the shares tried)."""
    x = grid["vol"] if by == "vol" else -grid["deepest"]
    x = np.maximum.accumulate(x)
    share = float(np.interp(value if by == "vol" else -value, x, SHARES))
    out = {k: float(np.interp(share, SHARES, grid[k])) for k in ("ret", "vol", "deepest")}
    out["ratio"] = out["ret"] / out["vol"] if out["vol"] > 1e-9 else 0.0
    out["share"] = share * 100
    return out


def _fixed_grid(index: pd.DatetimeIndex, change: np.ndarray, rf: np.ndarray | None = None):
    """Every fixed share in SHARES, put back to the share at the close before the first day of each month."""
    target = np.tile(SHARES, (len(index), 1))
    return _sim(target, change, rebalance=np.asarray(index.day == 1), rf=rf)


def _never(change: np.ndarray, share: float) -> tuple[dict, float]:
    """`share` in bitcoin at the start and never touched again: its statistics and the share it ended with."""
    coin = share * (1 - COST) * np.cumprod(1 + np.nan_to_num(change))
    equity = np.concatenate([[1.0], coin + (1 - share)])
    return _stats(equity[1:] / equity[:-1] - 1), float(coin[-1] / equity[-1] * 100)


# ---------- what the signals predict ----------

def _ahead(close: pd.Series) -> pd.DataFrame:
    """What happened in the AHEAD days after each day's close (NaN where those days have not all happened yet)."""
    change = close.pct_change()
    out = pd.DataFrame(index=close.index)
    out["ret"] = (close.shift(-AHEAD) / close - 1) * 100
    low = close[::-1].rolling(AHEAD, min_periods=AHEAD).min()[::-1].shift(-1)  # lowest close of days t+1 .. t+AHEAD
    out["fall"] = ((low / close - 1) * 100).clip(upper=0.0)
    out["vol"] = change[::-1].rolling(AHEAD, min_periods=AHEAD).std(ddof=1)[::-1].shift(-1) * math.sqrt(YEAR) * 100
    return out


def _rank(vol: pd.Series, start: str) -> pd.Series:
    """Where each day's volatility stands among all EARLIER days since `start` (0 = lowest so far, 1 = highest)."""
    part = vol[vol.index >= pd.Timestamp(start)]
    values = part.values
    out = np.full(len(values), np.nan)
    for i in range(HAR_MIN, len(values)):
        if np.isfinite(values[i]):
            earlier = values[:i]
            earlier = earlier[np.isfinite(earlier)]
            if len(earlier) >= HAR_MIN:
                out[i] = float((earlier < values[i]).mean())
    return pd.Series(out, index=part.index).reindex(vol.index)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 10:
        return None
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _groups(score: pd.Series, ahead: pd.DataFrame, starts: dict) -> dict:
    """Days split in five by `score` (0..1): what followed, for each start date; plus rank correlations on
    non-overlapping days (every AHEAD-th day)."""
    out = {}
    for name, start in starts.items():
        frame = ahead.assign(score=score)
        frame = frame[frame.index >= pd.Timestamp(start)].dropna()
        bucket = np.minimum((frame["score"].values * 5).astype(int), 4)
        rows = []
        for g in range(5):
            part = frame[bucket == g]
            rows.append({"n": len(part),
                         "ret": float(part["ret"].mean()) if len(part) else None,
                         "lost": float((part["ret"] < 0).mean() * 100) if len(part) else None,
                         "fall": float(part["fall"].mean()) if len(part) else None,
                         "vol": float(part["vol"].mean()) if len(part) else None})
        apart = frame.iloc[::AHEAD]
        out[name] = {"rows": rows, "n": len(frame), "apart": len(apart),
                     "rho_ret": _spearman(apart["score"].values, apart["ret"].values),
                     "rho_vol": _spearman(apart["score"].values, apart["vol"].values),
                     "rho_fall": _spearman(apart["score"].values, apart["fall"].values)}
    return out


# ---------- luck check ----------

def _shifted(position: np.ndarray, change: np.ndarray, band: float) -> dict:
    """The share series moved along the calendar by SIMS random circular shifts: same shares, same persistence, same
    average, but no longer lined up with what the market did. Column 0 is the real one."""
    n = len(position)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(60, n - 60, SIMS)
    cols = np.column_stack([position] + [np.roll(position, int(k)) for k in shifts])
    st = _stats(_sim(cols, change, band=band)[0])
    real = {k: float(v[0]) for k, v in st.items()}
    return {"real": real,
            "ratio_ge": int((st["ratio"][1:] >= real["ratio"]).sum()),
            "ret_ge": int((st["ret"][1:] >= real["ret"]).sum()),
            "shallow_ge": int((st["deepest"][1:] >= real["deepest"]).sum()),
            "ratio_med": float(np.median(st["ratio"][1:])), "deepest_med": float(np.median(st["deepest"][1:])),
            "ret_med": float(np.median(st["ret"][1:]))}


# ---------- the computation (kept apart from the formatting so look-ahead can be tested) ----------

def _coin(close: pd.Series) -> dict | None:
    """One other coin from the first day every approach has a value: hold, volatility sizing, the blend, and the
    fixed shares with the same volatility."""
    wants, _ = _wants(close, with_har=False)
    pos = wants[[MAIN_KEY, "blend"]].shift(1)
    ready = pos.notna().all(axis=1)
    if ready.sum() < 400:
        return None
    first = ready.idxmax()
    keep = close.index >= first
    change = close.pct_change().values[keep]
    net, trade, held = _sim(pos.values[keep], change, band=np.array([BAND, 0.0]))
    grid = _stats(_fixed_grid(close.index[keep], change)[0])
    out = {"first": first, "hold": _one(change), "vol": _one(net[:, 0], trade[:, 0], held[:, 0]),
           "blend": _one(net[:, 1], trade[:, 1], held[:, 1])}
    out["vol_fixed"] = _match(out["vol"]["vol"], "vol", grid)
    out["blend_fixed"] = _match(out["blend"]["vol"], "vol", grid)
    return out


def _compute(close: pd.Series, coins: dict, rate: pd.Series | None, sims: bool = True) -> dict:
    wants, vols = _wants(close)
    har = _har(close)
    change_all = close.pct_change()
    keep = close.index >= pd.Timestamp(FULL_START)
    index = close.index[keep]
    change = change_all.values[keep]
    last = index[-1]
    position = wants.shift(1)[keep]  # decided at the close of day t, wanted during day t+1
    bands = np.array([0.0 if k in DAILY else BAND for k in KEYS])
    net, trade, held = _sim(position[list(KEYS)].values, change, band=bands)
    gnet, gtrade, gheld = _fixed_grid(index, change)
    col = {k: i for i, k in enumerate(KEYS)}
    fixed_col = {f: int(round(f * (len(SHARES) - 1))) for f in FIXED}

    periods = {}
    for name, start in PERIODS:
        rows = np.asarray(index >= pd.Timestamp(start))
        if rows.sum() < 60:
            continue
        grid = _stats(gnet[rows])
        p = {"start": index[rows][0], "hold": _one(change[rows]), "runs": {}, "fixed": {}, "same_vol": {},
             "same_deep": {}, "years": float(rows.sum() / YEAR)}
        for k in KEYS:
            i = col[k]
            p["runs"][k] = _one(net[rows, i], trade[rows, i], held[rows, i])
            p["same_vol"][k] = _match(p["runs"][k]["vol"], "vol", grid)
            p["same_deep"][k] = _match(p["runs"][k]["deepest"], "deepest", grid)
        for f, i in fixed_col.items():
            p["fixed"][f] = _one(gnet[rows, i], gtrade[rows, i], gheld[rows, i])
        p["never"], p["never_end"] = _never(change[rows], 0.5)
        periods[name] = p

    main_rows = np.asarray(index >= pd.Timestamp(MAIN_START))

    # ----- cash earns the Treasury bill rate (main period only) -----
    interest = None
    if rate is not None:
        rf = rate.reindex(rate.index.union(index)).ffill().reindex(index).fillna(0.0).values / 100 / YEAR
        keys = (MAIN_KEY, "blend", "bv40")
        inet, itrade, iheld = _sim(position[list(keys)].values, change, band=np.array([bands[col[k]] for k in keys]),
                                   rf=rf)
        igrid = _stats(_fixed_grid(index, change, rf)[0][main_rows])
        interest = {"mean": float(rf[main_rows].mean() * YEAR * 100), "last": rate.index[-1], "runs": {},
                    "same_vol": {}, "same_deep": {}}
        for i, k in enumerate(keys):
            interest["runs"][k] = _one(inet[main_rows, i], itrade[main_rows, i], iheld[main_rows, i])
            interest["same_vol"][k] = _match(interest["runs"][k]["vol"], "vol", igrid)
            interest["same_deep"][k] = _match(interest["runs"][k]["deepest"], "deepest", igrid)

    # ----- what each signal predicts -----
    ahead = _ahead(close)
    rank = _rank(vols["d30"], HAR_START)
    starts = {"full": FULL_START, "main": MAIN_START}
    predict = {"vol": _groups(rank, ahead, starts), "trend": _groups(wants["blend"], ahead, starts)}

    # ----- luck check (main period) -----
    luck = None
    if sims:
        luck = {"vol": _shifted(position[MAIN_KEY].values[main_rows], change[main_rows], BAND),
                "blend": _shifted(position["blend"].values[main_rows], change[main_rows], 0.0)}

    # ----- other coins -----
    others = {}
    for coin, _ in COINS:
        if coin in coins:
            got = _coin(coins[coin])
            if got:
                others[coin] = got

    return {"last": last, "close": close, "wants": wants, "vols": vols, "har": har, "rank": rank,
            "periods": periods, "interest": interest, "predict": predict, "luck": luck, "others": others,
            "har_score": {"main": _har_score(har, MAIN_START), "full": _har_score(har, FULL_START)},
            "position": position}


# ---------- the study ----------

def _three(m: dict | None) -> str:
    return "—" if not m else f"{_pct(m['ret'])}｜{_num(m['ratio'])}｜{_pct(m['deepest'])}"


def _fixed_cell(m: dict) -> str:
    return f"拿 {m['share']:.0f}% → {_pct(m['ret'])}"


def _of(n: int, m: int) -> str:
    return f"{n} 個裡 {m} 個"


def _share(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def study() -> dict:
    close = te._daily_close()
    coins, problems = _coin_closes()
    rate, rate_source, rate_problem = _tbill()
    r = _compute(close, coins, rate)
    last, periods = r["last"], r["periods"]
    main = periods["main"]
    hold, runs = main["hold"], main["runs"]
    span = f"{MAIN_START[:4]} 年到 {_d(last)}"

    # ----- table 1: every timing approach with its same-risk fixed share -----
    def row1(name: str, m: dict, sv: dict | None, sd: dict | None) -> list:
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['share']:.0f}%｜{m['turnover']:.1f} 次", _fixed_cell(sv) if sv else "—", _fixed_cell(sd) if sd else "—"]

    rows1 = [row1(NAMES["hold"], hold, None, None)]
    rows1 += [row1(NAMES[k], runs[k], main["same_vol"][k], main["same_deep"][k]) for k in KEYS]
    inter = r["interest"]
    if inter:
        for k in (MAIN_KEY, "blend", "bv40"):
            rows1.append(row1("現金有利息：" + NAMES[k].split("（")[0].replace("只看順勢：", "")
                              + ("（目標 40%）" if k != "blend" else ""),
                              inter["runs"][k], inter["same_vol"][k], inter["same_deep"][k]))

    # ----- table 2: by period -----
    names2 = [("full", f"{FULL_START[:4]} 年起"), ("main", f"{MAIN_START[:4]} 年起"), ("p22", "2022 年起"), ("p24", "2024 年起")]
    names2 = [(k, n) for k, n in names2 if k in periods]

    def row2(name: str, pick) -> list:
        return [name] + [_three(pick(periods[k])) for k, _ in names2]

    def fixed_same(p: dict, key: str) -> dict:
        return p["same_vol"][key]

    rows2 = [row2(NAMES["hold"], lambda p: p["hold"])]
    rows2 += [row2(f"固定只拿 {f * 100:.0f}%，每月調回", lambda p, f=f: p["fixed"][f]) for f in FIXED]
    rows2.append(row2("一開始放 50%，之後都不調回", lambda p: p["never"]))
    rows2 += [row2(NAMES[k], lambda p, k=k: p["runs"][k]) for k in ("v30", MAIN_KEY)]
    rows2.append(row2("　和上一列同波動的固定比例", lambda p: fixed_same(p, MAIN_KEY)))
    rows2.append(row2(NAMES["har"], lambda p: p["runs"]["har"]))
    rows2.append(row2(NAMES["blend"], lambda p: p["runs"]["blend"]))
    rows2.append(row2("　和上一列同波動的固定比例", lambda p: fixed_same(p, "blend")))
    rows2.append(row2(NAMES["bv40"], lambda p: p["runs"]["bv40"]))
    rows2.append(row2("　和上一列同波動的固定比例", lambda p: fixed_same(p, "bv40")))
    rows2.append(row2(NAMES["bv30"], lambda p: p["runs"]["bv30"]))

    # ----- table 3: what each signal predicts -----
    pv, pt = r["predict"]["vol"], r["predict"]["trend"]
    vol_names = ["波動最低的五分之一", "波動次低", "波動中間", "波動次高", "波動最高的五分之一"]
    trend_names = ["順勢混合想拿 0–20%", "順勢混合想拿 20–40%", "順勢混合想拿 40–60%", "順勢混合想拿 60–80%", "順勢混合想拿 80–100%"]

    def row3(name: str, full: dict, mainp: dict) -> list:
        return [name, f"{full['n']:,} 天", _pct(full["ret"]), f"{_pct(full['lost'], signed=False)}", _pct(full["fall"]),
                _pct(full["vol"], signed=False), _pct(mainp["ret"]), _pct(mainp["lost"], signed=False)]

    rows3 = [row3("依最近 30 天波動分：" + vol_names[g], pv["full"]["rows"][g], pv["main"]["rows"][g]) for g in range(5)]
    rows3 += [row3("依" + trend_names[g], pt["full"]["rows"][g], pt["main"]["rows"][g]) for g in range(5)]

    # ----- table 4: other coins -----
    others = r["others"]
    name_of = dict(COINS)
    rows4 = []
    for coin, _ in COINS:
        if coin not in others:
            continue
        o = others[coin]
        rows4.append([f"{name_of[coin]} {coin}（{_d(o['first'])} 起）", _three(o["hold"]), _three(o["vol"]),
                      f"拿 {o['vol_fixed']['share']:.0f}%：{_three(o['vol_fixed'])}", _three(o["blend"]),
                      f"拿 {o['blend_fixed']['share']:.0f}%：{_three(o['blend_fixed'])}"])
    n_c = len(others)

    def count(test) -> int:
        return sum(1 for o in others.values() if test(o))

    tally = {
        "vol_ratio": count(lambda o: o["vol"]["ratio"] > o["hold"]["ratio"]),
        "vol_ret": count(lambda o: o["vol"]["ret"] > o["hold"]["ret"]),
        "vol_deep": count(lambda o: o["vol"]["deepest"] > o["hold"]["deepest"]),
        "fix_ratio": count(lambda o: o["vol_fixed"]["ratio"] > o["hold"]["ratio"]),
        "fix_ret": count(lambda o: o["vol_fixed"]["ret"] > o["hold"]["ret"]),
        "fix_deep": count(lambda o: o["vol_fixed"]["deepest"] > o["hold"]["deepest"]),
        "blend_ratio": count(lambda o: o["blend"]["ratio"] > o["hold"]["ratio"]),
        "blend_ret": count(lambda o: o["blend"]["ret"] > o["hold"]["ret"]),
        "blend_deep": count(lambda o: o["blend"]["deepest"] > o["hold"]["deepest"]),
        "bfix_ratio": count(lambda o: o["blend_fixed"]["ratio"] > o["hold"]["ratio"]),
        "bfix_ret": count(lambda o: o["blend_fixed"]["ret"] > o["hold"]["ret"]),
        "bfix_deep": count(lambda o: o["blend_fixed"]["deepest"] > o["hold"]["deepest"]),
        "vol_vs_fix": count(lambda o: o["vol"]["ret"] > o["vol_fixed"]["ret"]),
        "vol_vs_fix_deep": count(lambda o: o["vol"]["deepest"] > o["vol_fixed"]["deepest"]),
        "blend_vs_fix": count(lambda o: o["blend"]["ret"] > o["blend_fixed"]["ret"]),
        "blend_vs_fix_deep": count(lambda o: o["blend"]["deepest"] > o["blend_fixed"]["deepest"]),
        "vol_vs_blend": count(lambda o: o["vol"]["ratio"] > o["blend"]["ratio"]),
    }
    if n_c:
        rows4.append(["報酬÷波動贏過一直抱著", "—", _of(n_c, tally["vol_ratio"]), _of(n_c, tally["fix_ratio"]),
                      _of(n_c, tally["blend_ratio"]), _of(n_c, tally["bfix_ratio"])])
        rows4.append(["每年報酬贏過一直抱著", "—", _of(n_c, tally["vol_ret"]), _of(n_c, tally["fix_ret"]),
                      _of(n_c, tally["blend_ret"]), _of(n_c, tally["bfix_ret"])])
        rows4.append(["最深虧損比一直抱著淺", "—", _of(n_c, tally["vol_deep"]), _of(n_c, tally["fix_deep"]),
                      _of(n_c, tally["blend_deep"]), _of(n_c, tally["bfix_deep"])])
        rows4.append(["和自己同波動的固定比例比：每年報酬較高｜最深虧損較淺", "—",
                      f"{_of(n_c, tally['vol_vs_fix'])}｜{_of(n_c, tally['vol_vs_fix_deep'])}", "—",
                      f"{_of(n_c, tally['blend_vs_fix'])}｜{_of(n_c, tally['blend_vs_fix_deep'])}", "—"])

    luck = r["luck"]
    hs = r["har_score"]["main"]
    v, b, bv = runs[MAIN_KEY], runs["blend"], runs["bv40"]
    sv_v, sv_b, sv_bv = main["same_vol"][MAIN_KEY], main["same_vol"]["blend"], main["same_vol"]["bv40"]
    sd_v, sd_b = main["same_deep"][MAIN_KEY], main["same_deep"]["blend"]
    never_gain = main["fixed"][0.5]["ret"] - main["never"]["ret"]

    rate_text = (f"「現金有利息」的三列，沒有買比特幣的錢領美國三個月國庫券利率（這段期間平均一年 {inter['mean']:.1f}%），"
                 "拿來比的固定比例也一樣領利息。" if inter else "這次拿不到國庫券利率，所以沒有「現金有利息」的幾列。")
    luck_text = ""
    if luck:
        lv, lb = luck["vol"], luck["blend"]
        luck_text = (f"運氣對照：把「依波動調整（目標 40%）」每天的持有比例沿著日曆隨機平移 {SIMS:,} 次（比例本身、平均、連續性都不變，"
                     f"只是不再對準行情），{SIMS:,} 次裡 {lv['ratio_ge']:,} 次的報酬÷波動不輸真的、{lv['shallow_ge']:,} 次的最深虧損不比真的深；"
                     f"順勢混合同樣平移，{SIMS:,} 次裡 {lb['ratio_ge']:,} 次、{lb['shallow_ge']:,} 次。")

    tables = [
        {"caption": (f"比特幣，{span}。每種做法都是當天收盤決定、隔天才照做，買賣扣交易金額的 {COST * 100:.2f}%，不借錢（最多拿 100%），"
                     "現金沒有利息；一直抱著不扣成本。「依波動調整」＝持有比例 = 目標波動 ÷ 最近的波動（最多 100%），完全不看漲跌，"
                     f"想拿的比例和手上的比例差超過一成（相對）才調；順勢的三列每天調回。最後兩欄是公平對照：找一個「固定只拿幾成、每月調回」"
                     "的比例，讓它這段期間的每年波動（或最深虧損）和這個做法一樣，再看它每年賺多少——比這個數字多出來的，才是「看時機」帶來的。"
                     + rate_text + luck_text),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "平均持有｜一年換手幾次",
                     "同樣波動的固定比例 → 每年報酬", "同樣最深虧損的固定比例 → 每年報酬"],
         "rows": rows1},
        {"caption": ("分期間看（比特幣）。每格是「每年報酬｜報酬÷波動｜最深曾經虧多少」，每個期間都算到 " + _d(last)
                     + "。「同波動的固定比例」是在那個期間裡另外找的，所以每個期間拿的成數不一樣。"
                     f"「一開始放 50%、之後都不調回」用來看每月調回本身的影響：{MAIN_START[:4]} 年起，每月調回的固定 50% 每年報酬比不調回的"
                     f"{'多' if never_gain >= 0 else '少'} {abs(never_gain):.1f} 個百分點，但不調回的那一份期末已經變成 {main['never_end']:.0f}% 是比特幣，"
                     "兩者承受的風險並不一樣。2017-08 以前的價格來自 blockchain.info。"),
         "columns": ["做法"] + [n for _, n in names2],
         "rows": rows2},
        {"caption": (f"波動和順勢各自預告了什麼（比特幣）。前五列：每天把最近 30 天的波動和「在那之前」的全部歷史（{HAR_START[:4]} 年起）比，"
                     "看它排在歷史的哪五分之一；後五列：依 17 條順勢規則當天想拿的比例分。然後看之後 30 天發生了什麼。"
                     f"第 2–6 欄是 {FULL_START[:4]} 年起、最後兩欄是 {MAIN_START[:4]} 年起。每天都算一次，所以 30 天的窗口互相重疊，"
                     f"真正不重疊的窗口 {FULL_START[:4]} 年起只有 {pv['full']['apart']} 個。用不重疊的日子算排名相關（+1 完全同向、0 沒關係）："
                     f"波動排名對之後 30 天報酬 {_num(pv['full']['rho_ret'])}、對之後 30 天的實際波動 {_num(pv['full']['rho_vol'])}；"
                     f"順勢比例對之後報酬 {_num(pt['full']['rho_ret'])}、對之後的實際波動 {_num(pt['full']['rho_vol'])}。"),
         "columns": ["那一天的狀態", "天數", "之後 30 天平均報酬", "之後 30 天是虧的比例", "30 天內平均最深跌到", "之後 30 天的實際波動",
                     f"{MAIN_START[:4]} 年起：平均報酬", f"{MAIN_START[:4]} 年起：虧的比例"],
         "rows": rows3},
        {"caption": (f"其他 {n_c} 種幣（幣安現貨日線，各自從上架後規則都算得出來的第一天起，到 {_d(last)}），同一套做法原封不動。"
                     "每格是「每年報酬｜報酬÷波動｜最深曾經虧多少」；「同波動的固定比例」是每種幣各自找的、每月調回。"
                     "這份名單是今天還活著的大幣，已經消失或跌到沒人理的幣不在裡面，所以整張表偏樂觀。"),
         "columns": ["幣（起算日）", "一直抱著", "依波動調整（目標 40%）", "和它同波動的固定比例", "17 條順勢混合", "和混合同波動的固定比例"],
         "rows": rows4},
    ]

    # ----- today -----
    want, vols, har = r["wants"].iloc[-1], r["vols"].iloc[-1], r["har"].iloc[-1]
    rank_now = r["rank"].iloc[-1]
    group_now = vol_names[min(int(rank_now * 5), 4)] if _ok(rank_now) else "—"
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": "比特幣最近 30 天的波動（換算成一年）",
         "value": f"{_pct(vols['d30'] * 100, signed=False)}，比 {HAR_START[:4]} 年以來 {_share(rank_now)} 的日子高（{group_now}）"},
        {"label": "其他算法的波動", "value": (f"60 天 {_pct(vols['d60'] * 100, signed=False)}；近期加重 {_pct(vols['ew'] * 100, signed=False)}；"
                                       f"HAR 預測的明天 {_pct(vols['har'] * 100, signed=False)}（換算成一年）")},
        {"label": "「依波動調整」下一天想拿的比例（30 天波動）",
         "value": "；".join(f"目標 {t * 100:.0f}%：{_share(want[_vkey(t)])}" for t in TARGETS)},
        {"label": "「依波動調整（目標 40%）」換其他波動算法",
         "value": f"60 天：{_share(want['w60'])}；近期加重：{_share(want['ew'])}；HAR 預測：{_share(want['har'])}"},
        {"label": "17 條順勢混合下一天想拿的比例", "value": _share(want["blend"])},
        {"label": "順勢混合 × 依波動調整下一天想拿的比例", "value": f"目標 40%：{_share(want['bv40'])}；目標 30%：{_share(want['bv30'])}"},
        {"label": "這幾格的意思", "value": "只是描述各種規則目前停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    def tenths(mine: float, theirs: float) -> str:
        cut_mine, cut_theirs = mine - hold["deepest"], theirs - hold["deepest"]
        if cut_theirs <= 0.5:
            return "—"
        x = max(0.0, cut_mine / cut_theirs) * 10
        return f"{x:.0f} 成" if x < 10.5 else "超過十成"

    def versus(a: float, c: float, tol: float = 1.0) -> str:
        return "差不多" if abs(a - c) <= tol else ("多" if a > c else "少")

    top, low = pv["full"]["rows"][4], pv["full"]["rows"][0]
    if top["ret"] < low["ret"] - 1 and top["lost"] > low["lost"] + 3:
        tell = "波動高之後確實比較常跌"
    elif top["ret"] > low["ret"] + 1 and top["lost"] < low["lost"] - 3:
        tell = "波動高之後反而沒有比較常跌"
    else:
        tell = "波動高低看不出之後漲跌的方向"
    t_top, t_low = pt["full"]["rows"][4], pt["full"]["rows"][0]
    part1 = (f"比特幣 {span}：一直抱著每年 {_pct(hold['ret'])}、最深虧 {_pct(hold['deepest'])}，順勢混合 {_pct(b['ret'])}、"
             f"{_pct(b['deepest'])}；完全不看漲跌、只依波動調整持有多少（目標 40%）是 {_pct(v['ret'])}、{_pct(v['deepest'])}，"
             f"虧損縮小的幅度約是順勢的 {tenths(v['deepest'], b['deepest'])}。")
    part2 = (f"但波動相同的「固定只拿 {sv_v['share']:.0f}%」就有 {_pct(sv_v['ret'])}、{_pct(sv_v['deepest'])}，依波動調整每年報酬"
             f"{versus(v['ret'], sv_v['ret'])}；順勢對它同波動的固定 {sv_b['share']:.0f}% 是 {_pct(b['ret'])} 對 {_pct(sv_b['ret'])}、"
             f"最深虧 {_pct(b['deepest'])} 對 {_pct(sv_b['deepest'])}。")
    part3 = (f"波動最高的五分之一日子，之後 30 天平均 {_pct(top['ret'])}、{top['lost']:.0f}% 是虧的（最低那組 {_pct(low['ret'])}、"
             f"{low['lost']:.0f}%），{tell}；順勢想拿最少時是 {_pct(t_low['ret'])}、{t_low['lost']:.0f}%，最多時 {_pct(t_top['ret'])}、"
             f"{t_top['lost']:.0f}%"
             + (f"。其他 {n_c} 種幣依波動調整的報酬÷波動 {_of(n_c, tally['vol_ratio'])}贏過抱著、{_of(n_c, tally['vol_vs_fix'])}贏過同波動的固定比例。"
                if n_c else "。"))
    verdict = part1 + part2 + part3

    # ----- notes -----
    f1, f5 = hs[1], hs[5]
    notes = [
        "做法一句話：「固定只拿幾成」＝不看任何訊號，永遠拿固定比例的比特幣、其餘現金，每月 1 日調回；「依波動調整持有多少」＝"
        "持有比例 = 目標波動 ÷ 最近的波動（最多 100%、不借錢），市場越劇烈拿得越少，完全不判斷漲跌；「順勢混合」＝「不挑參數，把很多順勢規則混在一起」"
        "那一頁的 17 條規則，看多的占幾成就拿幾成；「順勢混合 × 依波動調整」＝兩個比例相乘。",
        f"波動的四種算法：最近 30 天每日漲跌的標準差（主要版本）、最近 60 天、近期加重（每往前一天權重乘 {LAMBDA}）、HAR 預測。"
        "HAR 是用「昨天、最近 7 天平均、最近 30 天平均」的每日漲跌平方去預測明天的漲跌平方，係數每天只用當天以前的資料重新估計"
        f"（{HAR_START[:4]} 年起、累積滿 {HAR_MIN} 天才開始預測；預測值不低於歷史平均的 {HAR_FLOOR * 100:.0f}%）。",
        f"HAR 預測準不準（{MAIN_START[:4]} 年起、全部是事後才看到答案的預測，{f1['n']:,} 天）：預測「明天的漲跌平方」，和最笨的「就當成和最近 30 天一樣」比，"
        f"誤差平方和{'少' if (f1['vs_naive'] or 0) >= 0 else '多'} {abs(f1['vs_naive'] or 0):.1f}%；和「一律猜歷史平均」比，HAR 解釋了 {_pct(f1['har'])}、"
        f"最近 30 天平均解釋了 {_pct(f1['naive'])}。改成預測「接下來 5 天的平均」：比最近 30 天平均{'少' if (f5['vs_naive'] or 0) >= 0 else '多'} "
        f"{abs(f5['vs_naive'] or 0):.1f}%，HAR {_pct(f5['har'])}、最近 30 天平均 {_pct(f5['naive'])}。每日漲跌平方本身雜訊極大，"
        "這裡的數字不能直接和用盤中資料算波動的研究（例如公開研究報告的 R² 約 0.27）相比。",
        "「同樣波動／同樣最深虧損的固定比例」怎麼找：把 0% 到 100% 每隔 0.5% 的固定比例各跑一次（每月調回、同樣扣成本），"
        "用內插找出這段期間每年波動（或最深虧損）和該做法相同的那個比例，再讀出它的每年報酬。這個比例是事後才知道的，"
        "事前並不知道該拿幾成才會剛好一樣，所以這個對照對固定比例是有利的。",
        "目標波動 20%、30%、40%、60%、80% 這幾個數字，以及用 30 天來算波動，都是已經知道比特幣大概有多劇烈之後才選的（後見之明）；"
        "目標定得越低，其實就只是平均拿得越少。表裡把五個目標都列出來，沒有只挑好看的。",
        "「報酬÷波動」會偏袒任何「拿得比較少」的做法：只要這段期間有大跌，少拿就會讓這個比值變好看，並不代表有判斷時機的本事。"
        "所以這一頁才另外用「同樣波動的固定比例」來比。",
        f"只有比特幣一種資產、主要期間不到九個年度（{FULL_START[:4]} 年起也只有十幾年），大致就是三四輪大漲大跌，彼此不獨立；"
        "2022 年起、2024 年起的期間更短。做法之間一兩個百分點的差距不能當成確定的結論。第三張表的 30 天窗口互相重疊，也沒有做顯著性檢定。",
        "其他幣的名單是今天還在、而且夠大的幣（倖存者偏差）：當年同樣有名、後來歸零或下架的幣不在裡面，對「一直抱著」和所有做法都偏樂觀。"
        "每種幣的起算日是上架後約 200 天（等最長的順勢規則有值），各幣期間長短不同。",
        f"成本很粗略：一律按交易金額的 {COST * 100:.2f}% 扣，實際的手續費、價差和滑價隨交易所、金額、年代不同，沒有算稅。"
        "「依波動調整」差超過一成才調，換手比每天調少；每天調的版本列在第一張表供對照。"
        + (f"現金利息用美國三個月國庫券的貼現率除以 365 當每天的利息（資料到 {_d(inter['last'])}，假日沿用前一天），"
           "實際上放在交易所的美元或穩定幣不一定領得到。" if inter else ""),
        f"運氣對照的做法：把 {MAIN_START[:4]} 年起每天的持有比例整串往後平移隨機天數（尾巴接回開頭），重複 {SIMS:,} 次、同樣扣成本。"
        "這樣保留了「平均拿多少、多久變一次」，只拿掉「拿得少的日子剛好對上大跌」這件事；如果很多次平移的結果都不輸真的，"
        "代表成果主要來自平均拿得少，而不是時機。固定的亂數種子，資料沒變重跑結果相同。",
        "所有訊號、波動和 HAR 係數都只用到當天收盤為止的資料，隔天才持有對應的比例；只用已經收完的日線（UTC 0 點收盤），"
        "沒有換其他收盤時刻，所以帶有「剛好用這個時鐘」的運氣（見「拿掉時鐘運氣」那一頁）。",
        "過去的結果不保證未來會一樣。這一頁只是整理這些做法過去做了什麼、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += [p for p in problems + [rate_problem] if p]

    sources = [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線（BTC 與其他 8 種幣對 USDT）", "url": "https://www.binance.com/zh-TC/markets/overview"},
    ]
    if rate_source == "treasury":
        sources.append({"name": "美國財政部：每日國庫券利率（13 週）",
                        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates"})
    elif rate_source == "fred":
        sources.append({"name": "FRED：美國三個月國庫券利率（DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"})
    sources += [
        {"name": "公開研究：f1devbin／Crypto-Research-Labs（只依波動調整持有量的事先登記研究）",
         "url": "https://github.com/f1devbin/Crypto-Research-Labs"},
        {"name": "HAR 模型原始論文：Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility",
         "url": "https://doi.org/10.1093/jjfinec/nbp001"},
    ]

    return {
        "id": "vol_sizing",
        "title": "不看漲跌，只控制拿多少：依波動調整、或固定少拿一點，能做到順勢的幾成？",
        "question": ("順勢規則最站得住的好處是「虧得比較淺」。這個好處有多少其實只是「拿得比較少」？"
                     "如果完全不判斷漲跌，只在市場劇烈時少拿一點，或乾脆永遠只拿固定幾成，能做到順勢的幾成？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
