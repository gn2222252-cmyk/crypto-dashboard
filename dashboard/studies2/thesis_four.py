"""Historical study "thesis_four": the predictors that Sanhueza (2026, PhD thesis, Universidad Diego Portales, chapter 1
"A Comprehensive Look at Bitcoin Return Predictability") reports as the only ones beating the random walk at every
horizon out of sample - tested as plain hold-or-cash rules on the dashboard's fair exam, and on data after the
thesis ends.

What the thesis says (read before running; pages of the PDF):
- Daily data Sep 2014 - Apr 2025, 141 predictors, targets the log return of the next 1 / 7 / 14 / 28 days (Eq. 1).
  Out of sample: recursive (expanding) OLS r(t,t+h) = a + b x(t) against the random walk with drift (b = 0, Eq. 3),
  P/R = 1 reported, Campbell-Thompson style OOS R2 = 1 - MSPE(model) / MSPE(benchmark), Clark-West test.
- Text (p. 3, 25, 43): "67 variables" in-sample, "29" out of sample at one horizon at least, "just 4 outperformed the
  random walk across all horizons". The thesis never lists the four by name.
- Table 3 (recursive, P/R = 1, benchmark random walk), the rows with a positive OOS R2 AND a Clark-West star at all
  four horizons (1 / 7 / 14 / 28 days, in %):
    Volatility KCHI      0.32*** 0.75*** 0.73*** 1.27***
    Trend EMA Slow       0.10*   0.70*** 0.94*** 0.14***
    Trend Ichimoku Base  0.08*   0.38**  0.82*** 0.58***
    Trend ADX            0.11*   0.26*** 0.65*** 0.22**
    Trend ADX+           0.20**  0.71*** 0.79*** 0.57***
    (and the composite "Trends Factor" 0.13** 0.37*** 0.52*** 0.70***, a PCA of all significant trend indicators)
  That is five single predictors, not four; with the 5% level instead of 10% only KCHI and ADX+ remain, against the
  AR(4) benchmark (Table 6) only KCHI (and Volume ADI). Which one of the five the text leaves out cannot be told, so
  all five are tested (the factor is not: rebuilding it needs every significant trend variable).
- Table 1 definitions (computed with the Python "ta" library; each predictor is a 0/1 signal):
    KCHI: 1 when the close is above the upper Keltner band ("original" version: 20-day average of (4H - 2L + C) / 3).
    EMA Slow: 1 when the 12-day EMA is above the 26-day EMA (ta's ema_fast / ema_slow).
    Ichimoku Base: 1 when the conversion line (9-day (high max + low min) / 2) is above the base line (26 days).
    ADX: 1 when ADX(14) > 25 and DI+ > DI-.   ADX+: 1 when DI+(14) > DI-(14).
- Trading (Table 4, Anatolyev-Gerko): long when the one-step forecast is positive, short when negative, no cost.
  P/R = 1: KCHI 0.41 (Sharpe 0.64), EMA Slow 0.52 (0.80), Ichimoku Base 0.12 (0.18), ADX 0.45 (0.70), ADX+ 0.50
  (0.78); buy & hold in the same table 0.45 (0.70).

Pre-registered design (fixed before running):
P = each of the five predictors above, built exactly as Table 1 says, on daily UTC bars. x(t) = the 0/1 signal at
close t. Forecast at close t = expanding OLS of the next-day log return on x (pairs from 2014-09-01 whose target is
already known at close t, at least 100 pairs, forecasts from 2014-12-15), i.e. the thesis' Eq. 3 with h = 1,
point in time.
Rule (a) "P": hold BTC from close t to the next close when the forecast > 0, else cash (the thesis' sign rule made
long-or-cash, since the exam has no shorting).
Rule (b) "P + ma50": hold when the close is above its 50-day average AND the forecast > 0.
Exam = main_signal_check's, copied as blockspace / ls_ratio / lead_followup do: 108 monthly starts 2015-01..2023-12
(textbook, 0.10%), 72 starts x 24 clocks (textbook) and the same cells as an ordinary person (decide at the clock's
close, act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock, main_signal_check's seed and
its ma50 draws), other coins ETH SOL BNB XRP (each coin's own bars, own predictor and own regression; starts from the
first month every rule has a value, to three years before today). On each clock the predictor and the regression
use that clock's own daily bars (high / low = highest / lowest hourly close of the 24 hours). Ranking score = the
lowest of the four "beats holding" shares. Median deepest loss = median over the 1,728 textbook clock cells.
Placebo: the 0/1 signal circularly shifted by 180 days or more (100 draws), regression refit, ranking score
recomputed with 20 ordinary-person simulations per clock (the real rule scored the same way); placebo share = share
of draws scoring at least as high.
Pass test (stated now): a rule passes when its ranking score >= ma50's AND its median deepest loss is no worse than
ma50's AND its placebo share <= 10%. All three are needed.
Also reported (not part of the pass test): OOS R2 (Campbell-Thompson: 1 - MSPE(forecast) / MSPE(expanding mean of
the same target)) at 1 / 7 / 14 / 28 days in 2015-2019, 2020-2022, 2023-now, with 90% block-bootstrap intervals;
the thesis' own window (P/R = 1 inside Sep 2014 - Apr 2025) to see whether this construction reproduces Table 3 and
Table 4; and the period after the thesis' sample (2025-05-01 onward = data the thesis never saw): OOS R2 and the
rules' return against holding.

Data: Bitstamp BTC/USD daily OHLC (public, key-less) up to 2017-08-16, Binance BTCUSDT daily OHLC from 2017-08-17;
other coins Binance daily OHLC; returns of the exam come from the dashboard's daily closes and Binance 1-hour
closes exactly as in main_signal_check. Only public key-less endpoints through common.get_json (fixed user agent).

Found during the first smoke run, before any exam result was looked at (a data fix, not a design change): the
dashboard's pre-2017 daily close labelled each value by the UTC midnight it was taken at (= the previous day's close),
so before 2017-08 the "next-day" target was the same day's return (look-ahead). Fixed at the source on 2026-11-19
(classic._long_close now labels by the day the candle covers, like Bitstamp and Binance).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

THESIS = "https://doctorados.udp.cl/cms/wp-content/uploads/2026/06/Tesis_Doctoral_Aliro_Sanhueza.pdf"
BS_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BS_FROM, BS_TO = "2014-06-01", "2017-08-16"
COINS = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
TRAIN_FROM, OOS_FROM, MIN_PAIRS = "2014-09-01", "2014-12-15", 100
FULL_FROM = "2015-01-01"
THESIS_FROM, THESIS_END, AFTER_FROM = "2014-09-17", "2025-04-30", "2025-05-01"
PERIODS = (("2015 到 2019", "2015-01-01", "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"),
           ("2023 到今天", "2023-01-01", "2100-01-01"))
HORIZONS = (1, 7, 14, 28)
BLOCK = {1: 30, 7: 60, 14: 60, 28: 120}
PLACEBO, PLACEBO_GAP, PL_SIMS = 100, 180, 20
PRED = ("kchi", "ema", "ichi", "adx", "adxp")
PNAME = {"kchi": "KCHI（收盤高於肯特納通道上緣）", "ema": "EMA Slow（12 日指數均線高於 26 日）",
         "ichi": "一目均衡表基準線（轉換線高於基準線）", "adx": "ADX（趨勢強度 > 25 且上漲力道大於下跌力道）",
         "adxp": "ADX+（上漲力道 DI+ 大於下跌力道 DI−）"}
PSHORT = {"kchi": "KCHI", "ema": "EMA Slow", "ichi": "一目基準線", "adx": "ADX", "adxp": "ADX+"}
TABLE3 = {"kchi": (0.32, 0.75, 0.73, 1.27), "ema": (0.10, 0.70, 0.94, 0.14), "ichi": (0.08, 0.38, 0.82, 0.58),
          "adx": (0.11, 0.26, 0.65, 0.22), "adxp": (0.20, 0.71, 0.79, 0.57)}
TABLE4 = {"kchi": (0.41, 0.64), "ema": (0.52, 0.80), "ichi": (0.12, 0.18), "adx": (0.45, 0.70), "adxp": (0.50, 0.78)}
TESTED = tuple(p for p in PRED) + tuple(f"{p}_ma" for p in PRED)
RULES = ("ma50",) + TESTED
KI = {"ma50": 1, **{k: 81 + i for i, k in enumerate(TESTED)}}  # ma50 uses main_signal_check's random draws
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _name(k: str) -> str:
    if k == "ma50":
        return "50 日均線（基準）"
    if k.endswith("_ma"):
        return f"{PSHORT[k[:-3]]} 預測會漲＋50 日均線之上才持有"
    return f"{PSHORT[k]} 預測會漲才持有"


def _short(k: str) -> str:
    if k == "ma50":
        return "50 日均線"
    return f"{PSHORT[k[:-3]]}＋均線" if k.endswith("_ma") else PSHORT[k]


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return (f"{v * 100:.0f}%" if abs(v * 100) >= 0.5 else "0%") if _ok(v) else "—"


def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def _pt(v) -> str:
    return _pp(v * 100) if _ok(v) else "—"


def _r2(v) -> str:
    return f"{v * 100:+.2f}%" if _ok(v) else "—"


def _ci(lo, hi, f) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _store_path(name: str):
    return common.STORE / f"thesis_four_{name}.json"


def _save(name: str, data) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        _store_path(name).write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run simply fetches again
        pass


def _load_saved(name: str):
    try:
        return json.loads(_store_path(name).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _frame(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["high", "low", "close"]).astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _bitstamp() -> tuple[pd.DataFrame, str | None]:
    """BTC/USD daily OHLC 2014-06 .. 2017-08-16 (a closed history: kept once fetched)."""
    saved = _load_saved("bitstamp")
    if saved and max(saved) >= BS_TO and min(saved) <= "2014-06-02":
        return _frame(saved), None
    try:
        rows, start = {}, int(pd.Timestamp(BS_FROM).timestamp())
        stop = int(pd.Timestamp(BS_TO).timestamp())
        for _ in range(6):
            page = common.get_json(BS_URL, {"step": 86400, "limit": 1000, "start": start}, timeout=60)["data"]["ohlc"]
            if not page:
                break
            for p in page:
                ts = int(p["timestamp"])
                if ts <= stop:
                    rows[pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")] = [float(p["high"]), float(p["low"]),
                                                                            float(p["close"])]
            start = int(page[-1]["timestamp"]) + 86400
            if start > stop:
                break
        if len(rows) < 1100:
            raise RuntimeError(f"Bitstamp 只回傳 {len(rows)} 天")
        _save("bitstamp", rows)
        return _frame(rows), None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        if saved:
            return _frame(saved), f"Bitstamp 這次取不到（{type(exc).__name__}），用之前存下來的資料。"
        raise RuntimeError(f"Bitstamp 日線取不到（{type(exc).__name__}）") from exc


def _binance_ohlc(symbol: str) -> tuple[pd.DataFrame, str | None]:
    """Finished Binance spot daily OHLC from the first listed day (one row per calendar day, gaps carried)."""
    from dashboard.collectors import price as px

    try:
        rows, start = {}, 0
        for _ in range(12):
            page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
            if not page:
                break
            for r in page:
                rows[pd.Timestamp(int(r[0]), unit="ms").strftime("%Y-%m-%d")] = [float(r[2]), float(r[3]), float(r[4])]
            start = int(page[-1][0]) + 1
            if len(page) < 1000:
                break
        if len(rows) < 400:
            raise RuntimeError(f"幣安 {symbol} 日線只有 {len(rows)} 筆")
        _save(f"binance_{symbol}", rows)
        got, problem = rows, None
    except Exception as exc:  # noqa: BLE001
        got = _load_saved(f"binance_{symbol}")
        if not got:
            raise
        problem = f"幣安 {symbol} 日線這次取不到（{type(exc).__name__}），用之前存下來的資料。"
    df = _frame(got)
    df = df[(df.index < pd.Timestamp(common.today_utc())) & (df["close"] > 0)]
    return df.asfreq("D").ffill(), problem


# ---------- the five predictors (Table 1) ----------

def _wilder_sum(x: np.ndarray, n: int, first: int) -> np.ndarray:
    """Wilder's running sum: the plain sum of x[first-n+1 .. first], then s = s - s / n + x."""
    out = np.full(len(x), np.nan)
    if first >= len(x):
        return out
    s = float(np.sum(x[first - n + 1:first + 1]))
    out[first] = s
    for i in range(first + 1, len(x)):
        s = s - s / n + x[i]
        out[i] = s
    return out


def _signals(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    """0/1 signal of each predictor at every close (NaN while it has no value); only bars up to that close.
    Leading days without a bar are skipped (NaN)."""
    h, l, c = (np.asarray(a, dtype=float) for a in (h, l, c))
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    first = int(np.argmax(ok)) if ok.any() else len(c)
    if first > 0:
        inner = _signals(h[first:], l[first:], c[first:]) if first < len(c) else {p: np.array([]) for p in PRED}
        return {p: np.concatenate([np.full(first, np.nan), v]) for p, v in inner.items()}
    H, L, C = pd.Series(h), pd.Series(l), pd.Series(c)
    n = len(c)

    def binary(cond, known):
        return np.where(known, cond.astype(float), np.nan)

    hb = ((4 * H - 2 * L + C) / 3).rolling(20, min_periods=20).mean().values
    out = {"kchi": binary(c > hb, np.isfinite(hb))}
    e12 = C.ewm(span=12, adjust=False, min_periods=12).mean().values
    e26 = C.ewm(span=26, adjust=False, min_periods=26).mean().values
    out["ema"] = binary(e12 > e26, np.isfinite(e26) & np.isfinite(e12))
    conv = 0.5 * (H.rolling(9, min_periods=9).max() + L.rolling(9, min_periods=9).min()).values
    base = 0.5 * (H.rolling(26, min_periods=26).max() + L.rolling(26, min_periods=26).min()).values
    out["ichi"] = binary(conv > base, np.isfinite(base) & np.isfinite(conv))
    # Wilder's ADX(14)
    k = 14
    tr, pdm, ndm = np.zeros(n), np.zeros(n), np.zeros(n)
    if n > 1:
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
        up, down = h[1:] - h[:-1], l[:-1] - l[1:]
        pdm[1:] = np.where((up > down) & (up > 0), up, 0.0)
        ndm[1:] = np.where((down > up) & (down > 0), down, 0.0)
    s_tr, s_p, s_n = (_wilder_sum(a, k, k) for a in (tr, pdm, ndm))
    with np.errstate(all="ignore"):
        dip, din = 100 * s_p / s_tr, 100 * s_n / s_tr
        dx = 100 * np.abs(dip - din) / (dip + din)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    adx = np.full(n, np.nan)
    a0 = 2 * k - 1
    if n > a0:
        adx[a0] = float(np.mean(dx[k:a0 + 1]))
        for i in range(a0 + 1, n):
            adx[i] = (adx[i - 1] * (k - 1) + dx[i]) / k
    known_di = np.isfinite(dip) & np.isfinite(din)
    out["adxp"] = binary(dip > din, known_di)
    out["adx"] = binary((adx > 25) & (dip > din), known_di & np.isfinite(adx))
    return out


# ---------- expanding-window OLS (the thesis' Eq. 3) ----------

def _fit(dates: pd.DatetimeIndex, c: np.ndarray, x: np.ndarray, h: int, train_from, oos_from,
         oos_to=None) -> dict:
    """x[i] = predictor at close i. Target y[i] = log(c[i+h]/c[i]). The forecast at close i uses only pairs s <= i-h
    (their targets are known at close i). Benchmark = expanding mean of the same pairs (b = 0)."""
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(c)
    y = np.full(n, np.nan)
    if n > h:
        y[:n - h] = np.log(c[h:] / c[:-h])
    train = dates >= pd.Timestamp(train_from)
    if oos_to is not None:
        train &= dates <= pd.Timestamp(oos_to)
    valid = np.asarray(train) & np.isfinite(x) & np.isfinite(y)
    xv, yv = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
    s1, sx, sy = np.cumsum(valid), np.cumsum(xv), np.cumsum(yv)
    sxx, sxy = np.cumsum(xv * xv), np.cumsum(xv * yv)
    i = np.arange(n)
    j = i - h
    use = (j >= 0) & (dates >= pd.Timestamp(oos_from)) & np.isfinite(x)
    if oos_to is not None:
        use &= dates <= pd.Timestamp(oos_to)
    jj = np.clip(j, 0, None)
    k = s1[jj].astype(float)
    use &= k >= MIN_PAIRS
    with np.errstate(all="ignore"):
        mx, my = sx[jj] / k, sy[jj] / k
        vxx = sxx[jj] / k - mx * mx
        beta = np.where(vxx > 1e-12, (sxy[jj] / k - mx * my) / vxx, 0.0)
        f = my + beta * (x - mx)
    return {"f": np.where(use, f, np.nan), "b": np.where(use, my, np.nan), "y": y}


def _state(dates, c, x) -> np.ndarray:
    fit = _fit(dates, c, x, 1, TRAIN_FROM, OOS_FROM)
    return np.where(np.isnan(fit["f"]), np.nan, (fit["f"] > 0).astype(float))


def _roll(x: np.ndarray, k: int) -> np.ndarray:
    """Circular shift of the defined part of a signal by k days (the placebo)."""
    out = np.array(x, dtype=float)
    ok = np.flatnonzero(np.isfinite(out))
    if len(ok):
        out[ok] = np.roll(out[ok], k % len(ok))
    return out


def _r2_fn(d):
    return 1 - d["a"].sum(1) / d["b"].sum(1)


def _oos(fit: dict, idx: pd.DatetimeIndex, a: str, b: str, h: int) -> dict:
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)) & np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    em, eb = (fit["y"] - fit["f"])[m] ** 2, (fit["y"] - fit["b"])[m] ** 2
    return {"n": int(m.sum()), "r2": at._boot({"a": em, "b": eb}, _r2_fn, BLOCK[h])}


def _r2_plain(fit: dict, m: np.ndarray) -> float | None:
    m = m & np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    if m.sum() < 30:
        return None
    return float(1 - np.sum((fit["y"] - fit["f"])[m] ** 2) / np.sum((fit["y"] - fit["b"])[m] ** 2))


def _thesis_window(close: pd.Series, x: np.ndarray) -> dict:
    """The thesis' own exercise: Sep 2014 - Apr 2025, the first half only to estimate (P/R = 1), recursive OLS;
    OOS R2 at 1 / 7 / 14 / 28 days and the long-or-short rule (no cost, yearly = mean daily log return x 365)."""
    w = (close.index >= pd.Timestamp(THESIS_FROM)) & (close.index <= pd.Timestamp(THESIS_END))
    idx, c, xs = close.index[w], close.values[w], x[w]
    split = idx[len(idx) // 2]
    out = {"from": split, "r2": {}}
    for hz in HORIZONS:
        fit = _fit(idx, c, xs, hz, THESIS_FROM, split)
        out["r2"][hz] = _r2_plain(fit, np.ones(len(idx), dtype=bool))
    fit = _fit(idx, c, xs, 1, THESIS_FROM, split)
    m = np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    s = np.where(fit["f"][m] > 0, 1.0, -1.0) * fit["y"][m]
    bh = fit["y"][m]

    def ann(v):
        return float(v.mean() * YEAR), float(v.mean() / v.std(ddof=1) * math.sqrt(YEAR))
    out["strat"], out["hold"], out["long"] = ann(s), ann(bh), float(np.mean(fit["f"][m] > 0))
    return out


# ---------- rules ----------

def _ma(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return ct._gt(c, ct._mean(c, 50))


def _rules(ma: np.ndarray, st: dict) -> dict[str, np.ndarray]:
    out = {"ma50": ma}
    for p in PRED:
        out[p] = st[p]
        out[f"{p}_ma"] = np.where(np.isnan(ma) | np.isnan(st[p]), np.nan,
                                  np.nan_to_num(ma) * np.nan_to_num(st[p]))
    return out


def _grid_daily(close: pd.Series, pos: dict, year: int, first, last, keys=RULES) -> dict:
    """Textbook runs from monthly start dates (main_signal_check._daily with given positions)."""
    change = close.pct_change().fillna(0.0).values
    known = np.ones(len(close), dtype=bool)
    for k in keys:
        known &= ~np.isnan(pos[k])
    if not known.any():
        raise RuntimeError("規則沒有值")
    begin = max(close.index[int(np.argmax(known))] + DAY, pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in keys:
        net, _ = te._run(np.concatenate([[np.nan], pos[k][:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    """The 24 closing hours of the hourly grid (main_signal_check's). On each clock the predictors and the
    regression use that clock's own daily bars (high / low = highest / lowest hourly close of its 24 hours)."""

    def __init__(self, price, b0, pre, pre_ohlc: pd.DataFrame):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]])
            self.ma.append(np.nan_to_num(_ma(c)[npre:][:self.days], nan=0.0))
        ps = pd.Series(price)
        hi, lo = ps.rolling(25, min_periods=1).max().values, ps.rolling(25, min_periods=1).min().values
        po = pre_ohlc[pre_ohlc.index < b0 - DAY]
        self.npre = len(po)
        self.series = []
        for h in range(24):
            c = np.concatenate([po["close"].values, price[h::24]])
            hh = np.maximum(np.concatenate([po["high"].values, hi[h::24]]), c)
            ll = np.minimum(np.concatenate([po["low"].values, lo[h::24]]), c)
            dates = po.index.append(pd.date_range(b0 - DAY, periods=len(c) - self.npre, freq="D"))
            self.series.append({"c": c, "hh": hh, "ll": ll, "dates": dates, "sig": _signals(hh, ll, c)})

    def states(self, shift: dict | None = None) -> dict:
        """{predictor: [24 arrays of days]} of 'forecast > 0'; shift = {predictor: days} for the placebo."""
        out = {p: [] for p in PRED}
        for s in self.series:
            for p in PRED:
                x = s["sig"][p] if not shift else _roll(s["sig"][p], shift[p])
                st = _state(s["dates"], s["c"], x)[self.npre:][:self.days]
                out[p].append(np.nan_to_num(st, nan=0.0))
        return out

    def want(self, k: str, h: int, st: dict) -> np.ndarray:
        if k == "ma50":
            return self.ma[h]
        if k.endswith("_ma"):
            return self.ma[h] * st[k[:-3]][h]
        return st[k][h]

    def full(self, k: str, st: dict, sims: int = SIMS, seed_add: int = 0) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = self.want(k, h, st)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[k] + h + seed_add)
            pos = rt._missed(wl, P_MISS, rng, sims)
            r, _dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, P_COST, None)]), self.starts,
                                 YEAR)
            pr[:, h, :] = r
        edge, pedge = tr - self.hr[:, None], pr - self.hr[:, None, None]
        return {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                "c_deep_worst": float(td.min()),
                "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge))}


def _coin_pos(cd: dict, shift: dict | None = None) -> dict:
    st = {}
    for p in PRED:
        x = cd["sig"][p] if not shift else _roll(cd["sig"][p], shift[p])
        st[p] = _state(cd["close"].index, cd["close"].values, x)
    return _rules(_ma(cd["close"].values), st)


def _score(x: dict) -> None:
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
    x["low"] = min(x["checks"])


def _evaluate(k: str, close: pd.Series, pos: dict, clock: _Clock, cst: dict, coins: dict, coin_pos: dict,
              sims: int, seed_add: int = 0) -> dict:
    d = _grid_daily(close, pos, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], keys=(k,))
    x = clock.full(k, cst, sims, seed_add)
    dr, dd = d[k]
    x["d_n"], x["d_beat"], x["d_shallow"] = len(dr), int((dr > d["hold"][0]).sum()), int((dd > d["hold"][1]).sum())
    x["d_edge"] = float(np.median(dr - d["hold"][0]))
    x["d_deep_med"] = float(np.median(dd))
    o_n = o_beat = 0
    x["o_each"] = {}
    for a, cd in coins.items():
        try:
            g = _grid_daily(cd["close"], coin_pos[a], YEAR, cd["first"], _others_last(), keys=(k,))
        except Exception:  # noqa: BLE001
            continue
        n_a, b_a = len(g[k][0]), int((g[k][0] > g["hold"][0]).sum())
        x["o_each"][a] = (n_a, b_a)
        o_n, o_beat = o_n + n_a, o_beat + b_a
    x["o_n"], x["o_beat"] = o_n, o_beat
    _score(x)
    return x


def _others_last() -> pd.Timestamp:
    return pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)


# ---------- after the thesis ----------

def _after(close: pd.Series, pos: dict) -> dict:
    """From 2025-05-01 to the last close, textbook (0.10%): growth of each rule and of holding."""
    i0 = int(close.index.searchsorted(pd.Timestamp(AFTER_FROM)))
    change = close.pct_change().fillna(0.0).values
    out = {"hold": float(close.iloc[-1] / close.iloc[i0 - 1] - 1), "days": len(close) - i0}
    hold_eq = np.cumprod(1 + change[i0:])
    out["hold_deep"] = float((hold_eq / np.maximum.accumulate(hold_eq) - 1).min())
    for k in RULES:
        net, _ = te._run(np.concatenate([[np.nan], pos[k][:-1]]), change, COST)
        eq = np.cumprod(1 + net[i0:])
        out[k] = (float(eq[-1] - 1), float((eq / np.maximum.accumulate(eq) - 1).min()),
                  float(np.nanmean(np.nan_to_num(pos[k][i0 - 1:-1]))))
    return out


# ---------- look-ahead check ----------

def _lookahead(close: pd.Series, ohlc: pd.DataFrame, clock: _Clock) -> tuple[int, int]:
    """Scramble everything after a cut close: states up to the cut must not move (daily bars and two clocks)."""
    rng = np.random.default_rng(11)
    base = _daily_states(close, ohlc)
    bad = cuts = 0
    for cut in (len(close) - 2500, len(close) - 900, len(close) - 60):
        t = close.index[cut]
        c2 = close.copy()
        c2.iloc[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(close) - cut - 1))
        o2 = ohlc.copy()
        later = o2.index > t
        f = np.exp(rng.normal(0, 0.2, int(later.sum())))
        o2.loc[later, "close"] *= f
        o2.loc[later, "high"] *= f * 1.05
        o2.loc[later, "low"] *= f * 0.95
        alt = _daily_states(c2, o2)
        cuts += 1
        for p in PRED:
            if not np.array_equal(np.nan_to_num(base[p][:cut + 1], nan=-1), np.nan_to_num(alt[p][:cut + 1], nan=-1)):
                bad += 1
    for h in (0, 14):
        s = clock.series[h]
        n = len(s["c"])
        for cut in (n - 900, n - 60):
            b1 = {p: _state(s["dates"], s["c"], s["sig"][p]) for p in PRED}
            c3 = s["c"].copy()
            c3[cut + 1:] *= np.exp(rng.normal(0, 0.2, n - cut - 1))
            hi, lo = s["hh"].copy(), s["ll"].copy()
            hi[cut + 1:], lo[cut + 1:] = c3[cut + 1:] * 1.05, c3[cut + 1:] * 0.95
            sig2 = _signals(hi, lo, c3)
            cuts += 1
            for p in PRED:
                b2 = _state(s["dates"], c3, sig2[p])
                if not np.array_equal(np.nan_to_num(b1[p][:cut + 1], nan=-1), np.nan_to_num(b2[:cut + 1], nan=-1)):
                    bad += 1
    return cuts, bad


def _daily_states(close: pd.Series, ohlc: pd.DataFrame) -> dict:
    o = ohlc.reindex(close.index)
    sig = _signals(o["high"].values, o["low"].values, o["close"].values)
    return {p: _state(close.index, close.values, sig[p]) for p in PRED} | {"sig": sig}


# ---------- the computation ----------

def _compute() -> dict:
    problems = []
    bs, prob = _bitstamp()
    if prob:
        problems.append(prob)
    bn, prob = _binance_ohlc("BTCUSDT")
    if prob:
        problems.append(prob)
    # Bitstamp, Binance and the dashboard's daily close (classic._long_close, relabelled 2026-11-19) all label a
    # bar by the day it covers (checked below with the day-by-day return correlation).
    ohlc = pd.concat([bs[bs.index < bn.index[0]], bn])
    ohlc = ohlc[~ohlc.index.duplicated(keep="last")].sort_index().asfreq("D").ffill()
    ohlc = ohlc[ohlc.index < pd.Timestamp(common.today_utc())]
    close = te._daily_close()
    ds = _daily_states(close, ohlc)
    sig = ds["sig"]
    ma = _ma(close.values)
    pos = _rules(ma, ds)

    # OOS R2 at four horizons by period, and the thesis window
    oos, thesis = {}, {}
    for p in PRED:
        oos[p] = {}
        for hz in HORIZONS:
            fit = _fit(close.index, close.values, sig[p], hz, TRAIN_FROM, OOS_FROM)
            oos[p][hz] = {"全部": _oos(fit, close.index, FULL_FROM, "2100-01-01", hz),
                          "之後": _oos(fit, close.index, AFTER_FROM, "2100-01-01", hz)}
            for name, a, b in PERIODS:
                oos[p][hz][name] = _oos(fit, close.index, a, b, hz)
        thesis[p] = _thesis_window(close, sig[p])
    after = _after(close, pos)

    # fair grid
    rows, prob = cr._hourly("BTCUSDT")
    if prob:
        problems.insert(0, prob)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _Clock(price, b0, pre, ohlc)
    cst = clock.states()
    coins = {}
    for symbol, key in COINS:
        try:
            df, prob = _binance_ohlc(symbol)
            if prob:
                problems.append(prob)
            c = df["close"]
            cd = {"close": c, "sig": _signals(df["high"].values, df["low"].values, c.values)}
            cp = _coin_pos(cd)
            valid = msc._targets(c, YEAR).notna().all(axis=1).values
            known = np.ones(len(c), dtype=bool)
            for k in RULES:
                known &= ~np.isnan(cp[k])
            if not valid.any() or not known.any():
                continue
            cd["first"] = max(c.index[int(np.argmax(valid))], c.index[int(np.argmax(known))]) + DAY
            if cd["first"] >= _others_last():
                continue
            coins[key] = cd
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{key} 這次取不到資料（{type(exc).__name__}），沒有算進去。")
    coin_pos = {a: _coin_pos(cd) for a, cd in coins.items()}
    s = {k: _evaluate(k, close, pos, clock, cst, coins, coin_pos, SIMS) for k in RULES}

    # placebo: the 0/1 signal circularly shifted by >= 180 days, regression refit
    real20 = {k: _evaluate(k, close, pos, clock, cst, coins, coin_pos, PL_SIMS)["low"] for k in TESTED}
    rng = np.random.default_rng(SEED + 4)
    n_def = int(np.isfinite(sig["ema"]).sum())
    plac = {k: [] for k in TESTED}
    for i in range(PLACEBO):
        shift = {p: int(rng.integers(PLACEBO_GAP, n_def - PLACEBO_GAP)) for p in PRED}
        sp = {p: _state(close.index, close.values, _roll(sig[p], shift[p])) for p in PRED}
        pp = _rules(ma, sp)
        cstp = clock.states(shift)
        cpp = {a: _coin_pos(cd, shift) for a, cd in coins.items()}
        for k in TESTED:
            plac[k].append(_evaluate(k, close, pp, clock, cstp, coins, cpp, PL_SIMS, 7919 * (i + 1))["low"])
    placebo = {k: float(np.mean(np.array(v) >= real20[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    # crashes and whipsaws since 2015
    c = close.values.astype(float)
    first = int(close.index.searchsorted(pd.Timestamp(FULL_FROM)))
    bears = [b for b in ct._bears(close) if b["top"] >= first]
    crash = {k: [ct._crash(c, [(1.0, pos[k])], b) for b in bears] for k in RULES}
    whip = {k: ct._whipsaws(c, [(1.0, pos[k])], first) for k in RULES}
    held = {k: float(np.nanmean(np.nan_to_num(pos[k][first:]))) for k in RULES}
    flips = {p: float(np.mean(np.abs(np.diff(np.nan_to_num(ds[p][first:]))))) * YEAR for p in PRED}

    look = _lookahead(close, ohlc, clock)
    agree = {}
    o = ohlc.reindex(close.index)
    both = (close.index >= pd.Timestamp(FULL_FROM)) & np.isfinite(o["close"].values)
    agree["px"] = float(np.median(np.abs(o["close"].values[both] / close.values[both] - 1)))
    ra, rb = np.log(o["close"]).diff(), np.log(close).diff()
    for key, a, b in (("old", FULL_FROM, "2017-08-10"), ("new", "2017-08-25", "2100-01-01")):
        j = pd.concat([ra, rb], axis=1).dropna()
        j = j[(j.index >= pd.Timestamp(a)) & (j.index <= pd.Timestamp(b))]
        agree[key] = float(j.corr().iloc[0, 1])
    return {"close": close, "ohlc": ohlc, "sig": sig, "ds": ds, "pos": pos, "oos": oos, "thesis": thesis,
            "after": after, "s": s, "clock": clock, "real20": real20, "placebo": placebo, "placebo_med": placebo_med,
            "bears": bears, "crash": crash, "whip": whip, "held": held, "flips": flips, "look": look,
            "problems": problems, "coins": list(coins), "agree": agree,
            "coin_first": {a: cd["first"] for a, cd in coins.items()}}


# ---------- the study ----------

def study(r: dict | None = None) -> dict:
    r = r or _compute()
    close, s, oos, th, af = r["close"], r["s"], r["oos"], r["thesis"], r["after"]
    pnames = [p[0] for p in PERIODS]
    m = s["ma50"]

    # ----- pass test -----
    rule_ok = {}
    for k in TESTED:
        x = s[k]
        rule_ok[k] = {"score": x["low"] >= m["low"] - 1e-12, "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12,
                      "placebo": r["placebo"][k] <= 0.10}
        rule_ok[k]["all"] = all(rule_ok[k].values())
    passed = [k for k in TESTED if rule_ok[k]["all"]]

    # ----- table 1: what the thesis reported vs this construction in the thesis window -----
    rows1 = []
    for p in PRED:
        t = th[p]
        mine = "／".join(_r2(t["r2"][hz]) for hz in HORIZONS)
        paper = "／".join(f"{v:+.2f}%" for v in TABLE3[p])
        rows1.append([PNAME[p], paper, mine,
                      f"每年 {TABLE4[p][0] * 100:.0f}%（夏普 {TABLE4[p][1]:.2f}）",
                      f"每年 {_p(t['strat'][0])}（夏普 {t['strat'][1]:.2f}）；抱著 {_p(t['hold'][0])}（{t['hold'][1]:.2f}）",
                      _share(t["long"])])

    # ----- table 2: OOS R2 by period -----
    rows2 = []
    for p in PRED:
        for hz in HORIZONS:
            o = oos[p][hz]
            full = o["全部"]["r2"]
            rows2.append([f"{PSHORT[p]}，之後 {hz} 天", f"{_r2(full[0])} {_ci(full[1], full[2], _r2)}"]
                         + [_r2(o[n]["r2"][0]) for n in pnames]
                         + [f"{_r2(o['之後']['r2'][0])} {_ci(o['之後']['r2'][1], o['之後']['r2'][2], _r2)}"])

    # ----- table 3: rules on the fair grid -----
    done = [i for i, b in enumerate(r["bears"]) if b["done"]]

    def avoid(k):
        vals = [r["crash"][k][i]["avoid"] for i in done]
        return float(np.median(vals)) if vals else None

    order = sorted(RULES, key=lambda k: -s[k]["low"])
    rows3 = []
    for k in RULES:
        x = s[k]
        rows3.append([_name(k), _of(x["d_n"], x["d_beat"]), f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}", _of(x["o_n"], x["o_beat"]),
                      f"{_share(x['low'])}（第 {order.index(k) + 1}）｜{_pct(x['c_deep_med'])}",
                      f"{_share(r['held'][k])}｜{_share(avoid(k))}｜{r['whip'][k]['whips']:.1f} 次",
                      "—（基準）" if k == "ma50" else _share(r["placebo"][k])])
    hd = r["clock"].hd
    rows3.append(["對照：一直抱著不動", "—", "—", "—", "—", f"—｜{_pct(float(np.median(hd)))}", "100%｜0%｜0 次", "—"])

    # ----- table 4: pass test -----
    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows4 = []
    for k in TESTED:
        rows4.append([_name(k), f"{_share(s[k]['low'])} vs {_share(m['low'])}（{yes(rule_ok[k]['score'])}）",
                      f"{_pct(s[k]['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(rule_ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（{yes(rule_ok[k]['placebo'])}）",
                      "通過" if rule_ok[k]["all"] else "沒有通過"])

    # ----- table 5: after the thesis -----
    rows5 = [["一直抱著", _p(af["hold"]), _p(af["hold_deep"]), "100%"]]
    for k in RULES:
        g, dd, hl = af[k]
        rows5.append([_name(k), _p(g), _p(dd), _share(hl)])

    n_d, n_c = m["d_n"], m["c_n"]
    tables = [
        {"caption": ("一、先核對：照論文的做法（2014-09-17 到 2025-04-30，前一半只拿來估計、後一半每天用到當時為止的資料重估一次，"
                     "P/R = 1），這裡算出來的樣本外 R²（之後 1／7／14／28 天）和論文表 3 比；再照論文表 4 的交易練習：預測為正做多、"
                     "為負放空、不扣成本，年報酬＝日對數報酬平均×365。論文表 4 同一欄一直抱著是每年 45%（夏普 0.70）。"
                     "價格來源不同（論文沒寫哪一家，這裡 2017-08 以前用 Bitstamp、之後用幣安），指標細節也可能不同，只能對個大概。"),
         "columns": ["預測變數", "論文表 3：樣本外 R²（1／7／14／28 天）", "這裡算的（同樣 1／7／14／28 天）",
                     "論文表 4：做多／放空", "這裡算的：做多／放空｜同期間抱著", "預測為正（看多）的比例"],
         "rows": rows1},
        {"caption": ("二、樣本外 R²（Campbell–Thompson：1 −「模型預測誤差平方和」÷「過去平均報酬當預測的誤差平方和」，"
                     "大於 0 才代表比只猜過去平均好）。2014-09 起的資料、每天重估，2015 起才計分；方括號是 90% 區間"
                     "（區塊重抽 1,000 次）。最後一欄是論文樣本結束後（2025-05-01 起），論文完全沒看過的新資料。"
                     "多天的報酬天天重疊，真正獨立的樣本遠少於天數。"),
         "columns": ["預測變數、預測幾天", "2015 起全部［90% 區間］"] + pnames + ["2025-05 以後（新資料）［90% 區間］"],
         "rows": rows2},
        {"caption": (f"三、規則和 50 日均線同樣條件考（main_signal_check 的公平考法）：比特幣 {n_d} 個起算月（2015-01 到 2023-12，UTC 0 點收盤，"
                     f"成本 {COST * 100:.2f}%）、{n_c:,} 格起算月×收盤時間（照課本做）、同樣 {n_c:,} 格照一般人方式做（10 小時後才做、成本 "
                     f"{P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），以及其他幣（{'、'.join(r['coins'])}，各用自己的 K 線、"
                     "自己的指標和自己的迴歸）。前四項是報酬贏過同一天起一直抱著的比例（後面是每年多賺的中位數，百分點）；"
                     f"排名分數＝四項裡最低的；最深虧損＝{n_c:,} 格的中位數。持有｜熊市避開｜被洗：2015 起持有天數比例、"
                     "走完的大熊市（跌 50% 以上）避開跌幅的中位數、下車後 30 天內用更高價買回的每年次數。"
                     f"安慰劑＝把指標的 0/1 訊號整串循環平移至少 {PLACEBO_GAP} 天（{PLACEBO} 次）、重估迴歸再算排名分數，一樣好或更好的比例。"
                     "50 日均線的其他幣只算這四個幣（main_signal_check 還有 MSTR），所以它的分數可能和那份研究差一點。"),
         "columns": ["規則", "比特幣起算月：贏的", f"{n_c:,} 格照課本：贏的｜多賺", "一般人做法：贏的｜多賺", "其他幣起算月：贏的",
                     "排名分數（名次）｜最深虧損中位數", "持有｜熊市避開｜每年被洗", "安慰劑：打亂後一樣好的比例"],
         "rows": rows3},
        {"caption": ("四、事先寫下的通過條件逐條對：排名分數不低於 50 日均線；最深虧損中位數不比 50 日均線深；安慰劑比例 ≤ 10%。"
                     "三條都過才算通過。"),
         "columns": ["規則", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "安慰劑比例", "結論"],
         "rows": rows4},
        {"caption": (f"五、論文樣本結束後（{AFTER_FROM} 到 {_d(close.index[-1])}，{af['days']} 天）：照課本做（UTC 0 點收盤、隔天照做、"
                     f"成本 {COST * 100:.2f}%），規則的累積報酬、期間最深虧損、持有天數比例。只有一年多、一段行情，只能看方向，不能當證據。"),
         "columns": ["規則", "累積報酬", "期間最深虧損", "持有天數比例"],
         "rows": rows5},
    ]

    # ----- today -----
    pos, ds, sig = r["pos"], r["ds"], r["sig"]
    t = close.index[-1]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    now = [{"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"}]
    for p in PRED:
        fit = _fit(close.index, close.values, sig[p], 1, TRAIN_FROM, OOS_FROM)
        f = fit["f"][-1]
        sv = sig[p][-1]
        now.append({"label": f"{PSHORT[p]}：訊號｜模型預測明天報酬",
                    "value": (f"{'亮（1）' if sv >= 0.5 else '沒亮（0）'}｜{f * 100:+.3f}%" if _ok(sv) and _ok(f) else "—")})
    for k in RULES:
        now.append({"label": _name(k), "value": word(pos[k][-1])})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    best = max(TESTED, key=lambda k: s[k]["low"])
    r2_after = {p: oos[p][1]["之後"]["r2"][0] for p in PRED}
    r2_after28 = {p: oos[p][28]["之後"]["r2"][0] for p in PRED}
    pos_periods = {p: sum(1 for n in pnames for hz in HORIZONS if _ok(oos[p][hz][n]["r2"][0]) and oos[p][hz][n]["r2"][0] > 0)
                   for p in PRED}
    first = ("事先定的通過條件" + ("有規則過了：" + "、".join(_short(k) for k in passed) if passed else "沒有一條規則過")
             + f"。論文說「只有 4 個預測變數在每個天數都贏過隨機漫步」，但沒點名；它的表 3 其實有 5 個單一變數符合（KCHI、EMA Slow、"
             f"一目基準線、ADX、ADX+），這裡五個都測。")
    second = (f"公平考試的排名分數：50 日均線 {_share(m['low'])}，五個預測單獨用 "
              + "、".join(f"{PSHORT[p]} {_share(s[p]['low'])}" for p in PRED)
              + "，加上均線 " + "、".join(f"{PSHORT[p]} {_share(s[p + '_ma']['low'])}" for p in PRED)
              + f"；最好的是「{_short(best)}」（最深虧損中位數 {_pct(s[best]['c_deep_med'])}，50 日均線 {_pct(m['c_deep_med'])}，"
              f"安慰劑比例 {_share(r['placebo'][best])}）。")
    third = ("樣本外 R²（12 格＝三段期間×四種天數）為正的格數："
             + "、".join(f"{PSHORT[p]} {pos_periods[p]}" for p in PRED)
             + "；論文樣本結束後的新資料，明天報酬的 R² " + "、".join(f"{PSHORT[p]} {_r2(r2_after[p])}" for p in PRED)
             + "，28 天 " + "、".join(f"{PSHORT[p]} {_r2(r2_after28[p])}" for p in PRED) + "。")
    fourth = (f"論文樣本結束後（{AFTER_FROM} 起）一直抱著 {_p(af['hold'])}、50 日均線 {_p(af['ma50'][0])}，五個預測單獨用 "
              + "、".join(f"{PSHORT[p]} {_p(af[p][0])}" for p in PRED) + "；只有一年多，只能看方向。")
    verdict = first + second + third + fourth

    # ----- notes -----
    cuts, bad = r["look"]
    bear_txt = "；".join(
        f"{_d(close.index[b['top']])} 起" + ("" if b["done"] else "（還在進行，沒算進中位數）") + "："
        + "、".join(f"{_short(k)} {_share(r['crash'][k][i]['avoid'])}" for k in RULES)
        for i, b in enumerate(r["bears"])) or "沒有"
    notes = [
        ("事先寫下的設計（照原文）：P = each of the five predictors (KCHI, EMA Slow, Ichimoku Base, ADX, ADX+), built exactly "
         "as Table 1 says, on daily UTC bars; forecast = expanding OLS of the next-day log return on the 0/1 signal (pairs "
         "from 2014-09-01, at least 100, point in time). Rule (a): hold when the forecast > 0, else cash. Rule (b): hold "
         "when above the 50-day average AND forecast > 0. Exam = main_signal_check's (108 monthly starts, 72 x 24 clocks "
         "textbook and ordinary person 0.25% / 10 h later / 10% missed, other coins ETH SOL BNB XRP), ranking score = "
         "the lowest of the four beats-holding shares, median deepest loss over the 1,728 textbook cells, placebo = "
         f"signal circularly shifted >= 180 days ({PLACEBO} draws). Pass test: ranking score >= ma50's AND median deepest "
         "loss no worse AND placebo share <= 10%. Also: OOS R2 (Campbell-Thompson) at 1/7/14/28 days in 2015-2019, "
         "2020-2022, 2023-now, the thesis window (P/R = 1), and 2025-05 onward as new data."),
        ("論文實際寫的（Sanhueza 2026，第 3、6–8、25–29、43 頁）：2014-09 到 2025-04 每天一筆、141 個預測變數；67 個樣本內顯著、29 個"
         "至少一個天數樣本外顯著、「只有 4 個在每個天數都贏過隨機漫步」，但文中沒有列出是哪 4 個。表 3（遞迴、P/R = 1、和隨機漫步加漂移比）"
         "在 1／7／14／28 天都為正又都有 Clark–West 星號的單一變數有 5 個：KCHI、EMA Slow、一目基準線、ADX、ADX+（另外還有 PCA 合成的"
         "「趨勢因子」）。改用 5% 顯著水準只剩 KCHI 和 ADX+；改和 AR(4) 比（表 6）只剩 KCHI 和 Volume ADI。哪一種算法得到「4」，"
         "論文沒說，所以五個都測，也就多了一些「碰巧過關」的機會。趨勢因子要用所有顯著的趨勢指標做主成分，這裡沒有重建。"),
        ("指標怎麼算（論文表 1，用 Python 的 ta 套件，每個都是 0/1 訊號）：KCHI＝收盤高於肯特納通道上緣（原始版：(4×最高−2×最低+收盤)÷3 "
         "的 20 日平均）；EMA Slow＝12 日指數均線高於 26 日；一目基準線＝轉換線（9 日最高與最低的中點）高於基準線（26 日）；"
         "ADX＝14 日 ADX > 25 且 DI+ > DI−；ADX+＝DI+ > DI−。ADX 這裡用 Wilder 教科書的平滑法，ta 套件的寫法有些小差異；"
         "ta 套件「一次算全部指標」的函式裡肯特納通道可能用 10 日，試過 10 日，論文期間的 R² 一樣對不上論文。"
         "迴歸是論文的式 3：報酬＝a＋b×訊號，訊號只有 0 或 1，所以「預測為正」其實就是「到目前為止，訊號是這個狀態的日子，"
         "隔天平均報酬是正的」。比特幣過去平均是漲的，所以多數時候兩種狀態都預測為正，規則常常就等於一直抱著；"
         "2015 起持有天數比例見第三表。"),
        (f"每年翻轉次數（訊號 0/1 換邊，2015 起）：" + "、".join(f"{PSHORT[p]} {r['flips'][p]:.1f} 次" for p in PRED)
         + "（是訊號本身，不是規則；規則只有在預測正負改變時才換手）。"),
        ("通過條件：" + ("、".join(_short(k) for k in passed) + " 通過" if passed else "沒有一條規則通過")
         + f"。安慰劑用每個收盤時間 {PL_SIMS} 次一般人模擬重算排名分數（真的規則也用同樣方法算："
         + "、".join(f"{_short(k)} {_share(r['real20'][k])}" for k in TESTED) + "），打亂後排名分數的中位數："
         + "、".join(f"{_short(k)} {_share(r['placebo_med'][k])}" for k in TESTED) + "。"),
        ("不偷看未來：指標只用到當天收盤為止的 K 線；迴歸每天重估，只用報酬已經實現的那些天；換收盤時間時，每個收盤時間用自己的日 K"
         f"（最高／最低是那 24 小時裡最高／最低的整點收盤價，會比真正的最高最低窄一點）。檢查：把 {cuts} 個切點之後的價格隨機打亂再算一次，"
         "切點以前的狀態" + ("完全沒變。" if bad == 0 else f"有 {bad} 處變了，這次結果不可信。")),
        (f"資料：比特幣日 K（最高、最低、收盤）2017-08-16 以前用 Bitstamp BTC/USD、之後用幣安 BTCUSDT；考試的報酬用儀表板的日線和幣安 1 小時線，"
         f"和 main_signal_check 一樣。儀表板 2017-08 以前的日線（blockchain.info）是用「UTC 0 點取的價」標日期，等於前一天的收盤，"
         "所以 Bitstamp 的日 K 往後挪一天對齊；不挪的話，2017 以前的「明天報酬」其實是當天報酬，會偷看到未來。對齊後每天報酬的相關係數："
         f"2015 到 2017-08 是 {r['agree']['old']:.3f}、之後 {r['agree']['new']:.3f}；收盤價差距中位數 {r['agree']['px'] * 100:.2f}%。"
         "其他幣用幣安日 K，起算月從每條規則都有值的那個月起：" + "、".join(f"{a} {_d(d)[:7]}" for a, d in r["coin_first"].items()) + "。"),
        (f"2015 年以後的大熊市避開的跌幅（算進中途被洗）：{bear_txt}。大熊市只有寥寥幾次，差幾個百分點不代表真的比較好。"),
        ("相鄰起算日共用大部分的歷史，24 個收盤時間只差幾個小時，這些「幾個裡幾個」不是互相獨立的證據。十條規則一起看，"
         "多重比較會讓「碰巧過關」的機會變大。其他幣是今天還在的大幣，有倖存者偏誤。"),
        "過去的結果不保證未來會一樣。這裡只比較論文挑出的指標過去的表現、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "thesis_four",
        "title": "博士論文說「每個天數都贏過隨機漫步」的技術指標",
        "question": ("一篇 2026 年的博士論文測了 141 個預測比特幣報酬的變數，說只有少數幾個在樣本外每個天數都贏過隨機漫步。"
                     "照論文的方法把它們做成「預測會漲才持有」的規則，在公平考試裡能不能贏過 50 日均線和隨機對照？論文樣本結束後還成立嗎？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Sanhueza (2026), 博士論文 第一章 A Comprehensive Look at Bitcoin Return Predictability, Universidad Diego Portales",
             "url": THESIS},
            {"name": "Bitstamp 公開 API：BTC/USD 日 K（2017-08 以前）", "url": "https://www.bitstamp.net/api/#ohlc_data"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前，考試用的日線）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
