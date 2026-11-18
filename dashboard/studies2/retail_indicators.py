"""Historical study "retail_indicators": do the indicators most taught to Taiwanese and Chinese-language retail
traders - KD, MACD, RSI, Bollinger bands, moving-average alignment, the 60/240-day lines, the 50/200 cross and the
price-and-volume breakout - work on bitcoin and Taiwan stocks when tested the dashboard's way?

Fourteen long-or-cash rules with the textbook parameters, fixed in advance (no tuning):
  K1 KD(9,3,3): buy on a golden cross with D below 20, sell on a death cross with D above 80
  K2 KD(9,3,3): hold while K is above D (any golden cross buys, any death cross sells)
  M1 MACD(12,26,9): hold while MACD (DIF) is above its signal line
  M2 the same, only while MACD is also above zero
  R1 RSI(14): buy when RSI is below 30, sell when it is above 70
  R2 RSI(14): buy when RSI rises back above 30, sell when it falls back below 70
  B1 Bollinger(20, 2): buy at or below the lower band, sell at or above the middle band
  B2 Bollinger(20, 2): buy at or below the lower band, sell at or above the upper band
  B3 Bollinger(20, 2) breakout: buy on a close above the upper band, sell on a close below the middle band
  A1 moving-average alignment: hold while the 5 > 10 > 20 > 60-day averages
  A2 hold while the close is above the 60-day average (the "quarter line")
  A3 hold while the close is above the 240-day average (the "year line")
  G1 hold while the 50-day average is above the 200-day average
  V1 buy on a close at a 20-day high with volume above 1.5x the average of the previous 20 days; sell on a 10-day low
Benchmarks: buy-and-hold, the dashboard's verified 55-day-high / 20-day-low rule and the 17-rule trend blend.

Markets: bitcoin (the dashboard's long daily close series, with Binance highs, lows and volume from 2017-08), ether
(Binance), the Taiwan weighted index ^TWII, 0050.TW, 2330.TW and the S&P 500 (Yahoo Finance daily bars).

No look-ahead: every indicator uses data up to and including the close of day t and decides the position held
during day t+1; the yearly choice uses returns before 1 January of the test year only. Costs: 0.10% of the amount
moved (0.30% in the take-apart check). Only finished daily bars from public, key-less endpoints with the dashboard's
fixed user agent.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import trend_ensemble as te

COST = 0.001
COST_HIGH = 0.003
CRYPTO_YEAR = 365
SHARE_YEAR = 252
BTC_MAIN = "2018-01-01"  # Binance highs, lows and volume exist from 2017-08; everything has a value by then
BTC_TRAIN = "2014-01-01"
FIRST_TEST = {"crypto": 2019, "shares": 2005}
MIN_TRAIN_YEARS = 1
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"

MARKETS = (  # key, name, short, kind, loader
    ("BTC", "比特幣", "比特幣", "crypto", "btc"),
    ("ETH", "以太幣", "以太幣", "crypto", "ETHUSDT"),
    ("TWII", "台灣加權指數", "加權", "shares", "^TWII"),
    ("0050", "元大台灣50（0050）", "0050", "shares", "0050.TW"),
    ("2330", "台積電（2330）", "2330", "shares", "2330.TW"),
    ("SPX", "美國標普 500 指數", "標普", "shares", "^GSPC"),
)
CLOSE_HOUR_UTC = {"^TWII": 6.0, "0050.TW": 6.0, "2330.TW": 6.0, "^GSPC": 22.0}  # a bar is finished after this

RULES = ("K1", "K2", "M1", "M2", "R1", "R2", "B1", "B2", "B3", "A1", "A2", "A3", "G1", "V1")
BENCH = ("T55", "BL")
NAMES = {
    "K1": "KD 低檔黃金交叉買、高檔死亡交叉賣",
    "K2": "KD 黃金交叉買、死亡交叉賣",
    "M1": "MACD 黃金交叉買、死亡交叉賣",
    "M2": "MACD 交叉，只在零軸之上持有",
    "R1": "RSI 超賣（<30）買、超買（>70）賣",
    "R2": "RSI 回升過 30 買、跌回 70 以下賣",
    "B1": "布林通道下緣買、中線賣",
    "B2": "布林通道下緣買、上緣賣",
    "B3": "布林通道突破上緣買、跌破中線賣",
    "A1": "均線多頭排列（5>10>20>60）",
    "A2": "站上季線（60 日均線）",
    "A3": "站上年線（240 日均線）",
    "G1": "50 日均線在 200 日之上（黃金交叉）",
    "V1": "價量齊揚：20 日新高且量 >1.5 倍、10 日新低賣",
    "T55": "對照：55 日新高進、20 日新低出（已驗證）",
    "BL": "對照：17 條順勢規則混合（已驗證）",
    "HOLD": "一直抱著",
}
SHORT = {"K1": "KD 低檔交叉", "K2": "KD 交叉", "M1": "MACD 交叉", "M2": "MACD 零軸上", "R1": "RSI 超賣超買",
         "R2": "RSI 回升", "B1": "布林下緣→中線", "B2": "布林下緣→上緣", "B3": "布林突破", "A1": "均線多頭排列",
         "A2": "季線", "A3": "年線", "G1": "50/200 黃金交叉", "V1": "價量齊揚", "T55": "55／20", "BL": "17 條混合"}
NO_VOLUME = tuple(k for k in RULES if k != "V1")  # rules that the hourly clock check can rebuild


# ---------- formatting ----------

_pct, _num, _d, _ok = te._pct, te._num, te._d, te._ok


def _clean(obj):
    """None -> "—" and no NaN / infinity anywhere."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if obj is None:
        return "—"
    if isinstance(obj, float) and not math.isfinite(obj):
        return "—"
    return obj


# ---------- data ----------

def _binance_ohlcv(symbol: str) -> pd.DataFrame:
    """Finished Binance spot daily candles (high, low, close, volume) from the first listed day, one row per day."""
    rows, start = [], 0
    for _ in range(12):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    now_ms = time.time() * 1000
    rows = [r for r in rows if int(r[6]) < now_ms]  # closed candles only
    frame = pd.DataFrame({"high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                          "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]},
                         index=pd.to_datetime([r[0] for r in rows], unit="ms").normalize())
    frame = frame[~frame.index.duplicated()].sort_index()
    frame = frame[frame["close"] > 0]
    if len(frame) < 400:
        raise RuntimeError(f"幣安 {symbol} 日線只有 {len(frame)} 筆")
    full = frame.asfreq("D")
    gap = full["close"].isna()
    full["close"] = full["close"].ffill()
    full.loc[gap, "high"] = full.loc[gap, "close"]  # a missing day repeats the last close and has no volume
    full.loc[gap, "low"] = full.loc[gap, "close"]
    return full


def _btc() -> pd.DataFrame:
    """The dashboard's long bitcoin close series, with Binance highs, lows and volume where they exist (from 2017-08).
    Before that a day's high and low are its close and volume is missing."""
    close = te._daily_close()
    close = close[close.index >= pd.Timestamp(BTC_TRAIN) - pd.Timedelta(days=400)]
    frame = pd.DataFrame({"close": close})
    try:
        b = _binance_ohlcv("BTCUSDT").reindex(frame.index)
        frame["high"], frame["low"], frame["volume"] = b["high"], b["low"], b["volume"]
    except Exception:  # noqa: BLE001 - without Binance only the close-based rules keep a value
        frame["high"] = frame["low"] = frame["volume"] = np.nan
    frame["high"] = frame[["high", "close"]].max(axis=1)
    frame["low"] = frame[["low", "close"]].min(axis=1)
    return frame


def _yahoo(symbol: str) -> tuple[pd.DataFrame, bool]:
    """Every daily bar Yahoo Finance has (high, low, close, volume), finished bars only. For shares with an adjusted
    close (0050, 2330: splits and dividends), all prices are scaled by adjclose / close. Returns (frame, adjusted)."""
    reply = requests.get(YAHOO + symbol, params={"period1": 0, "period2": int(time.time()), "interval": "1d"},
                         headers=common.HEADERS, timeout=30)
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({"high": quote["high"], "low": quote["low"], "close": quote["close"],
                          "volume": quote["volume"]},
                         index=pd.to_datetime(result["timestamp"], unit="s").normalize()).astype(float)
    adjusted = False
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None and not symbol.startswith("^"):
        factor = pd.Series(adj, index=frame.index, dtype=float) / frame["close"]
        for col in ("high", "low", "close"):
            frame[col] = frame[col] * factor
        adjusted = True
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    finished = frame.index + pd.Timedelta(hours=CLOSE_HOUR_UTC.get(symbol, 22.0)) <= pd.Timestamp(now)
    frame = frame[finished]
    frame["high"] = frame[["high", "close"]].max(axis=1)
    frame["low"] = frame[["low", "close"]].min(axis=1)
    frame.loc[frame["volume"] <= 0, "volume"] = np.nan  # a zero volume is a missing number, not a quiet day
    # Yahoo sometimes applies a share split to only part of the history (0050's 2025 1-to-4 split reaches back to
    # 2014 only). A one-day move beyond these limits cannot be real for these markets: everything before it is put on
    # the later scale, so that day's own move becomes zero.
    repairs = []
    ratio = (frame["close"] / frame["close"].shift(1)).values
    for i in np.where((ratio < 0.55) | (ratio > 1.8))[0]:
        r = float(frame["close"].iloc[i] / frame["close"].iloc[i - 1])
        before = frame.index < frame.index[i]
        for col in ("high", "low", "close"):
            frame.loc[before, col] = frame.loc[before, col] * r
        frame.loc[before, "volume"] = frame.loc[before, "volume"] / r
        repairs.append((frame.index[i], r))
    if len(frame) < 1000:
        raise RuntimeError(f"Yahoo {symbol} 日線只有 {len(frame)} 筆")
    frame.attrs["repairs"] = repairs
    return frame, adjusted


def _load() -> tuple[dict, list[str], dict]:
    data, problems, adjusted = {}, [], {}
    for key, name, _, kind, loader in MARKETS:
        try:
            if loader == "btc":
                data[key] = _btc()
            elif loader.endswith("USDT"):
                data[key] = _binance_ohlcv(loader)
            else:
                data[key], adjusted[key] = _yahoo(loader)
                for day, r in data[key].attrs.get("repairs", []):
                    problems.append(f"{name}在 Yahoo 的價格於 {_d(day)} 有一天跳了 {(r - 1) * 100:+.0f}%（拆股只調整了一部分歷史），"
                                    "這裡把之前的價格乘上同一個比例接起來，那一天的漲跌算成 0。")
        except Exception as exc:  # noqa: BLE001 - one market failing must not stop the others
            problems.append(f"{name}這次取不到資料（{type(exc).__name__}），沒有算進去。")
    return data, problems, adjusted


# ---------- the indicators ----------

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _kd(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Taiwanese KD(9,3,3): RSV over n days, K = 2/3 K + 1/3 RSV, D = 2/3 D + 1/3 K, both starting at 50."""
    hh = pd.Series(high).rolling(n, min_periods=n).max().values
    ll = pd.Series(low).rolling(n, min_periods=n).min().values
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    kp = dp = 50.0
    for i in range(len(close)):
        if not math.isfinite(hh[i]):
            continue
        span = hh[i] - ll[i]
        rsv = (close[i] - ll[i]) / span * 100 if span > 1e-12 else 50.0
        kp = kp * 2 / 3 + rsv / 3
        dp = dp * 2 / 3 + kp / 3
        k[i], d[i] = kp, dp
    return k, d


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.where(down > 1e-12)
    return (100 - 100 / (1 + rs)).where(down > 1e-12, 100.0).where(up.notna())


def _state(buy, sell, ready) -> pd.Series:
    """1 from a buy day until a sell day, 0 otherwise; cash until the first buy; NaN until the indicator is ready."""
    buy, sell, ready = (np.asarray(x, dtype=bool) for x in (buy, sell, ready))
    raw = pd.Series(np.where(ready & buy, 1.0, np.where(ready & sell, 0.0, np.nan)))
    out = raw.ffill().fillna(0.0)
    return out.where(ready)


def _signals(frame: pd.DataFrame, rules=RULES + BENCH) -> pd.DataFrame:
    """What each rule wants to hold after each close (1 = all in, 0 = cash, fractions for the blend; NaN = not ready).
    Only data up to and including that close is used."""
    c = frame["close"].astype(float).reset_index(drop=True)
    h = frame["high"].astype(float).reset_index(drop=True)
    lo = frame["low"].astype(float).reset_index(drop=True)
    n = len(c)
    idx = np.arange(n)
    out: dict[str, pd.Series] = {}
    ma = {m: c.rolling(m, min_periods=m).mean() for m in (5, 10, 20, 50, 60, 200, 240)}

    if {"K1", "K2"} & set(rules):
        k, d = _kd(h.values, lo.values, c.values)
        kp, dpv = np.roll(k, 1), np.roll(d, 1)
        ready = np.isfinite(k) & (idx >= 30)
        ok_prev = np.isfinite(kp) & (idx >= 1)
        golden = ok_prev & (k > d) & (kp <= dpv)
        death = ok_prev & (k < d) & (kp >= dpv)
        out["K1"] = _state(golden & (d < 20), death & (d > 80), ready)
        out["K2"] = pd.Series(np.where(k > d, 1.0, 0.0)).where(ready)
    if {"M1", "M2"} & set(rules):
        dif = _ema(c, 12) - _ema(c, 26)
        sig = _ema(dif, 9)
        ready = idx >= 100
        out["M1"] = (dif > sig).astype(float).where(ready)
        out["M2"] = ((dif > sig) & (dif > 0)).astype(float).where(ready)
    if {"R1", "R2"} & set(rules):
        rsi = _rsi(c)
        prev = rsi.shift(1)
        ready = rsi.notna().values & (idx >= 50)
        out["R1"] = _state((rsi < 30).values, (rsi > 70).values, ready)
        out["R2"] = _state(((prev < 30) & (rsi >= 30)).values, ((prev > 70) & (rsi <= 70)).values, ready)
    if {"B1", "B2", "B3"} & set(rules):
        mid = ma[20]
        sd = c.rolling(20, min_periods=20).std(ddof=0)
        upper, lower = mid + 2 * sd, mid - 2 * sd
        ready = mid.notna().values
        out["B1"] = _state((c <= lower).values, (c >= mid).values, ready)
        out["B2"] = _state((c <= lower).values, (c >= upper).values, ready)
        out["B3"] = _state((c > upper).values, (c < mid).values, ready)
    if "A1" in rules:
        out["A1"] = ((ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60])).astype(float).where(ma[60].notna())
    if "A2" in rules:
        out["A2"] = (c > ma[60]).astype(float).where(ma[60].notna())
    if "A3" in rules:
        out["A3"] = (c > ma[240]).astype(float).where(ma[240].notna())
    if "G1" in rules:
        out["G1"] = (ma[50] > ma[200]).astype(float).where(ma[200].notna())
    if "V1" in rules:
        vol = frame["volume"].astype(float).reset_index(drop=True)
        avg = vol.shift(1).rolling(20, min_periods=15).mean()
        high20 = c.rolling(20, min_periods=20).max()
        low10 = c.rolling(10, min_periods=10).min()
        ready = (avg.notna() & high20.notna()).values
        # once volume history exists the rule stays available; a later day without volume simply cannot buy
        ready = pd.Series(ready).cummax().values
        buy = ((c >= high20) & vol.notna() & (vol > 1.5 * avg)).values
        out["V1"] = _state(buy, (c <= low10).values, ready)
    if "T55" in rules:
        high55 = c.rolling(55, min_periods=55).max()
        low20 = c.rolling(20, min_periods=20).min()
        out["T55"] = _state((c >= high55).values, (c <= low20).values, high55.notna().values)
    if "BL" in rules:
        sig17, _ = te._signals(c)
        out["BL"] = sig17.mean(axis=1).where(sig17.notna().all(axis=1))
    result = pd.DataFrame({k: out[k].values for k in rules}, index=frame.index)
    return result


# ---------- measuring ----------

def _stats(net: np.ndarray, year: int) -> dict:
    n = len(net)
    if n < 2:
        return {"ret": None, "vol": None, "ratio": None, "deepest": None, "growth": None}
    growth = float(np.prod(1 + net))
    ret = (growth ** (year / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(year) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"ret": float(ret), "vol": vol, "ratio": float(ret / vol) if vol > 1e-9 else 0.0, "deepest": deepest,
            "growth": growth}


def _trades(pos: np.ndarray, net: np.ndarray) -> list[float]:
    """Results in % of every closed all-in trade: from the first held day to the day it was sold (the selling cost is
    charged on that day)."""
    out, i = [], 0
    while i < len(pos):
        if pos[i] > 0.5 and (i == 0 or pos[i - 1] < 0.5):
            j = i
            while j + 1 < len(pos) and pos[j + 1] > 0.5:
                j += 1
            if j + 1 < len(pos):
                out.append((float(np.prod(1 + net[i:j + 2])) - 1) * 100)
            i = j + 1
        else:
            i += 1
    return out


def _nets(close: pd.Series, signal: pd.DataFrame, cost: float = COST, delay: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily net returns and positions of every rule: the signal at the close of row t is held during row t+delay."""
    change = close.pct_change()
    pos = signal.shift(delay)
    nets = {}
    for col in signal.columns:
        net, _ = te._run(pos[col].values, change.values, cost)
        nets[col] = pd.Series(net, index=close.index).where(pos[col].notna())
    return pd.DataFrame(nets), pos


def _market(key: str, kind: str, frame: pd.DataFrame) -> dict:
    """Everything for one market; study() only formats it."""
    year = CRYPTO_YEAR if kind == "crypto" else SHARE_YEAR
    close = frame["close"]
    signal = _signals(frame)
    change = close.pct_change()
    nets, pos = _nets(close, signal)
    ready = pos.notna().all(axis=1) & change.notna()
    if key == "BTC":
        ready &= close.index >= pd.Timestamp(BTC_MAIN)
    if not ready.any():
        raise RuntimeError("規則還沒有足夠的資料")
    first = ready.idxmax()
    rows = close.index >= first
    mid = close.index[rows][int(rows.sum() // 2)]
    out = {"key": key, "kind": kind, "year": year, "first": first, "last": close.index[-1], "mid": mid,
           "close": close, "signal": signal, "whole": {}, "halves": {}, "trades": {}, "share": {}}
    hold = change[rows].fillna(0.0).values
    out["whole"]["HOLD"] = _stats(hold, year)
    in_first = (close.index >= first) & (close.index < mid)
    in_second = close.index >= mid
    out["halves"]["HOLD"] = (_stats(change[in_first].values, year), _stats(change[in_second].values, year))
    for col in RULES + BENCH:
        net = nets[col][rows].values
        out["whole"][col] = _stats(net, year)
        out["halves"][col] = (_stats(nets[col][in_first].values, year), _stats(nets[col][in_second].values, year))
        out["share"][col] = float(pos[col][rows].mean() * 100)
        if col != "BL":
            out["trades"][col] = _trades(pos[col][rows].values, net)
    out["walk"] = _walk(nets, change, kind, year, key)
    return out


def _spearman(a: list, b: list) -> float | None:
    if len(a) < 5:
        return None
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    v = float(np.corrcoef(ra, rb)[0, 1])
    return v if math.isfinite(v) else None


def _walk(nets: pd.DataFrame, change: pd.Series, kind: str, year: int, key: str) -> dict:
    """Each test year: score every indicator rule (return / volatility) on the data before 1 January, hold the best
    one during the year, chain the years. Also the rank correlation of the scores before vs inside the year."""
    days = nets.index
    train_start = pd.Timestamp(BTC_TRAIN) if key == "BTC" else days[0]
    first_day = {k: nets[k][nets.index >= train_start].first_valid_index() for k in RULES}
    last = days[-1]
    chosen, held, hold, t55, bl, picks, rhos, beat = [], [], [], [], [], [], [], 0
    years = []
    for y in range(FIRST_TEST[kind], last.year + 1):
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        if a > last:
            break
        eligible = [k for k in RULES if first_day[k] is not None and (a - first_day[k]).days >= 365 * MIN_TRAIN_YEARS]
        if not eligible:
            continue
        before = {k: te._score_parts(nets[k][(days >= first_day[k]) & (days < a)].values)[2] for k in eligible}
        inside = (days >= a) & (days < b)
        if inside.sum() < 2 or nets.loc[inside, eligible].isna().any().any():
            continue
        best = max(eligible, key=lambda k: before[k])
        picks.append((y, best))
        part = nets.loc[inside, best].values
        chosen.append(part)
        h = change[inside].fillna(0.0).values
        hold.append(h)
        t55.append(nets.loc[inside, "T55"].fillna(0.0).values)
        bl.append(nets.loc[inside, "BL"].fillna(0.0).values)
        beat += float(np.prod(1 + part)) > float(np.prod(1 + h))
        years.append(y)
        if inside.sum() >= 30:
            after = {k: te._score_parts(nets.loc[inside, k].values)[2] for k in eligible}
            rho = _spearman([before[k] for k in eligible], [after[k] for k in eligible])
            if rho is not None:
                rhos.append(rho)
    if not chosen:
        return {"years": []}
    cat = np.concatenate
    counts: dict[str, int] = {}
    for _, k in picks:
        counts[k] = counts.get(k, 0) + 1
    return {"years": years, "picks": picks, "counts": counts, "beat": int(beat),
            "chosen": _stats(cat(chosen), year), "hold": _stats(cat(hold), year), "T55": _stats(cat(t55), year),
            "BL": _stats(cat(bl), year), "rho": float(np.mean(rhos)) if rhos else None, "n_rho": len(rhos)}


# ---------- step 4: bitcoin on 24 closing hours, higher cost, one day late ----------

def _clock_wants(m: "cr._Market", h: int, keys) -> dict[str, np.ndarray]:
    """Each rule's wanted position after every close of clock h. A day's high / low are the highest / lowest hourly
    price inside that 24-hour day (before the hourly data: the close)."""
    closes = m.price[h::24]
    hi = np.empty(len(closes))
    lo = np.empty(len(closes))
    hi[0] = lo[0] = closes[0]
    for j in range(1, len(closes)):
        seg = m.price[h + 24 * (j - 1):h + 24 * j + 1]
        hi[j], lo[j] = seg.max(), seg.min()
    pre = m.pre
    frame = pd.DataFrame({"close": np.concatenate([pre, closes]), "high": np.concatenate([pre, hi]),
                          "low": np.concatenate([pre, lo]), "volume": np.nan})
    sig = _signals(frame, tuple(keys))
    return {k: sig[k].fillna(0.0).values[len(pre):] for k in keys}


def _clocks(keys) -> dict:
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    m = cr._Market(price, b0, cr._prehistory(b0))
    hold = m.hold()
    runs = {k: [] for k in keys}
    for h in range(24):
        wants = _clock_wants(m, h, keys)
        for k in keys:
            runs[k].append(m.run(h, wants[k]))
    out = {"hold": hold, "start": m.dates[m.w0], "problem": problem}
    for k in keys:
        ret = np.array([x["ret"] for x in runs[k]])
        ratio = np.array([x["ratio"] for x in runs[k]])
        deep = np.array([x["deepest"] for x in runs[k]])
        out[k] = {"ret": (float(ret.min()), float(ret.max())), "ratio": (float(ratio.min()), float(ratio.max())),
                  "deep": (float(deep.min()), float(deep.max())), "beat": int((ret > hold["ret"]).sum()),
                  "beat_ratio": int((ratio > hold["ratio"]).sum())}
    return out


def _takeapart(frame: pd.DataFrame, keys) -> dict:
    """Bitcoin daily from BTC_MAIN: cost 0.30%, and acting one day late."""
    close = frame["close"]
    signal = _signals(frame, tuple(keys))
    rows = close.index >= pd.Timestamp(BTC_MAIN)
    out = {}
    for name, cost, delay in (("base", COST, 1), ("cost", COST_HIGH, 1), ("late", COST, 2)):
        nets, _ = _nets(close, signal, cost, delay)
        out[name] = {k: _stats(nets[k][rows].values, CRYPTO_YEAR) for k in keys}
    return out


# ---------- the study ----------

def _range(pair, fmt) -> str:
    return f"{fmt(pair[0])} 到 {fmt(pair[1])}"


def study() -> dict:
    data, problems, adjusted = _load()
    if "BTC" not in data or "TWII" not in data:
        raise RuntimeError("比特幣或加權指數取不到資料")
    res = {}
    for key, name, short, kind, _ in MARKETS:
        if key in data:
            try:
                res[key] = _market(key, kind, data[key])
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}資料不夠長（{type(exc).__name__}），沒有算進去。")
    btc, tw = res["BTC"], res["TWII"]
    names = {k: n for k, n, *_ in MARKETS}
    shorts = {k: s for k, _, s, *_ in MARKETS}
    keys_used = [k for k, *_ in MARKETS if k in res]
    n_mk = len(keys_used)

    # ----- the three best-looking indicator rules on bitcoin (that the hourly check can rebuild) -----
    top3 = sorted(NO_VOLUME, key=lambda k: -btc["whole"][k]["ratio"])[:3]
    clock_keys = tuple(top3) + BENCH
    clock_problem = None
    try:
        clocks = _clocks(clock_keys)
    except Exception as exc:  # noqa: BLE001 - the rest of the study still stands
        clocks, clock_problem = None, f"比特幣 1 小時線這次取不到（{type(exc).__name__}），24 個收盤時間的檢查沒有做。"
    apart = _takeapart(data["BTC"], clock_keys)

    def win_text(trades: list[float]) -> str:
        if not trades:
            return "沒有完成的交易"
        won = [t for t in trades if t > 0]
        lost = [t for t in trades if t <= 0]
        return (f"{len(trades)} 筆裡 {len(won)} 筆賺｜{_pct(np.mean(won) if won else None)}／"
                f"{_pct(np.mean(lost) if lost else None)}")

    def win_rate(trades: list[float]) -> float | None:
        return len([t for t in trades if t > 0]) / len(trades) * 100 if trades else None

    # ----- table 1: bitcoin and the Taiwan index, every rule -----
    t1 = []
    for k in RULES + BENCH:
        row = [NAMES[k]]
        for r in (btc, tw):
            m = r["whole"][k]
            row += [f"{_pct(m['ret'])}｜{_num(m['ratio'])}｜{_pct(m['deepest'])}",
                    win_text(r["trades"][k]) if k != "BL" else f"平均持有 {r['share'][k]:.0f}%"]
        halves = []
        for r in (btc, tw):
            a, b = r["halves"][k]
            halves.append(f"{_num(a['ratio'])}／{_num(b['ratio'])}")
        row.append("；".join(halves))
        row.insert(1, f"{btc['share'][k]:.0f}%｜{tw['share'][k]:.0f}%")
        t1.append(row)
    hb, ht = btc["whole"]["HOLD"], tw["whole"]["HOLD"]
    hbh, hth = btc["halves"]["HOLD"], tw["halves"]["HOLD"]
    cap1 = (f"每條規則用課本上的參數、一個數字都沒調，收盤決定、隔天照做，每次買賣扣 {COST * 100:.2f}%。"
            f"比特幣 {_d(btc['first'])} 到 {_d(btc['last'])}，一直抱著是每年 {_pct(hb['ret'])}｜報酬÷波動 {_num(hb['ratio'])}｜"
            f"最深虧 {_pct(hb['deepest'])}；台灣加權指數 {_d(tw['first'])} 到 {_d(tw['last'])}（不含股利），一直抱著是 "
            f"{_pct(ht['ret'])}｜{_num(ht['ratio'])}｜{_pct(ht['deepest'])}。「賺的筆數」是每一次買進到賣出（扣成本）算一筆，"
            f"後面是賺的那些平均賺多少／賠的那些平均賠多少。最後一欄是前半段／後半段的報酬÷波動（比特幣以 {_d(btc['mid'])} 分開，"
            f"一直抱著是 {_num(hbh[0]['ratio'])}／{_num(hbh[1]['ratio'])}；加權以 {_d(tw['mid'])} 分開，一直抱著是 "
            f"{_num(hth[0]['ratio'])}／{_num(hth[1]['ratio'])}）。")
    table1 = {"caption": cap1,
              "columns": ["規則", "平均持有｜比特幣｜加權", "比特幣：每年報酬｜報酬÷波動｜最深虧", "比特幣：賺的筆數｜平均賺／平均賠",
                          "加權：每年報酬｜報酬÷波動｜最深虧", "加權：賺的筆數｜平均賺／平均賠", "前／後半報酬÷波動（比特幣；加權）"],
              "rows": t1}

    # ----- table 2: the same rules on every market -----
    board = {}
    t2 = []
    for k in RULES + BENCH:
        ret_w = [mk for mk in keys_used if res[mk]["whole"][k]["ret"] > res[mk]["whole"]["HOLD"]["ret"]]
        rat_w = [mk for mk in keys_used if res[mk]["whole"][k]["ratio"] > res[mk]["whole"]["HOLD"]["ratio"]]
        deep_w = [mk for mk in keys_used if res[mk]["whole"][k]["deepest"] > res[mk]["whole"]["HOLD"]["deepest"]]
        t55_w = [mk for mk in keys_used if res[mk]["whole"][k]["ratio"] > res[mk]["whole"]["T55"]["ratio"]]
        all_trades = sum((res[mk]["trades"].get(k, []) for mk in keys_used), [])
        board[k] = {"ret": len(ret_w), "ratio": len(rat_w), "deep": len(deep_w), "t55": len(t55_w),
                    "wins": win_rate(all_trades), "n_trades": len(all_trades),
                    "won": len([t for t in all_trades if t > 0])}
        wins = (f"{len(all_trades)} 筆裡 {len([t for t in all_trades if t > 0])} 筆賺" if k != "BL" else "—")
        ratios = "／".join(_num(res[mk]["whole"][k]["ratio"]) for mk in keys_used)
        t2.append([NAMES[k], f"{n_mk} 個裡 {len(ret_w)} 個", f"{n_mk} 個裡 {len(rat_w)} 個",
                   f"{n_mk} 個裡 {len(deep_w)} 個", "—" if k == "T55" else f"{n_mk} 個裡 {len(t55_w)} 個", wins, ratios])
    hold_ratios = "／".join(_num(res[mk]["whole"]["HOLD"]["ratio"]) for mk in keys_used)
    span2 = "；".join(f"{shorts[mk]} {_d(res[mk]['first'])[:7]} 起" for mk in keys_used)
    adj_txt = "、".join(shorts[k] for k, v in adjusted.items() if v)
    table2 = {"caption": (f"同樣的規則原封不動用在 {n_mk} 個市場（{span2}），和同一段日子一直抱著比。"
                          + (f"{adj_txt} 用的是還原拆股和配息的價格；" if adj_txt else "")
                          + "加權、標普是價格指數，不含股利。最後一欄依序是 " + "／".join(shorts[mk] for mk in keys_used)
                          + f" 的報酬÷波動，一直抱著依序是 {hold_ratios}。幣一年算 365 天，股票算 252 個交易日。"),
              "columns": ["規則", "每年報酬贏過抱著", "報酬÷波動贏過抱著", "最深虧損比抱著淺", "報酬÷波動贏過 55／20",
                          "所有市場合計的交易", "報酬÷波動（各市場）"],
              "rows": t2}

    # ----- table 3: chosen in advance each year -----
    t3 = []
    for mk in keys_used:
        w = res[mk]["walk"]
        if not w["years"]:
            t3.append([names[mk], "—", "—", "—", "—", "—", "—", "—"])
            continue
        top = sorted(w["counts"].items(), key=lambda kv: -kv[1])[:2]
        often = "、".join(f"{SHORT[k]} {v} 年" for k, v in top)
        yrs = w["years"]

        def trio(m):
            return f"{_pct(m['ret'])}｜{_num(m['ratio'])}｜{_pct(m['deepest'])}"
        t3.append([f"{names[mk]}（{yrs[0]}–{yrs[-1]}，{len(yrs)} 年）", trio(w["chosen"]), trio(w["hold"]),
                   trio(w["T55"]), trio(w["BL"]), f"{len(yrs)} 年裡 {w['beat']} 年", often,
                   _num(w["rho"]) if w["rho"] is not None else "—"])
    table3 = {"caption": ("事先挑、之後再驗：每一年 1 月 1 日，只用之前的資料，從 14 條指標規則裡挑「報酬÷波動」最高的一條，整年照它做，"
                          "再把每一年接起來。比特幣從 2014 年起的資料開始挑（2017-08 以前沒有最高最低價和成交量，KD 用收盤價代替、"
                          "價量齊揚那時還不能用），其他市場從有資料起。每格是「每年報酬｜報酬÷波動｜最深虧」。最後一欄是挑之前的排名"
                          "和那一年實際排名的相關係數（各年平均；1 = 排名完全一樣，0 = 沒關係，負的 = 反過來）。"),
              "columns": ["市場（測試年份）", "事先挑的指標規則", "一直抱著", "55／20（已驗證）", "17 條混合（已驗證）",
                          "事先挑的報酬贏過抱著", "最常被挑到", "前後排名相關"],
              "rows": t3}

    # ----- table 4: bitcoin take-apart -----
    t4 = []
    base, hi_cost, late = apart["base"], apart["cost"], apart["late"]
    for k in clock_keys:
        row = [NAMES[k], f"{_pct(base[k]['ret'])}｜{_num(base[k]['ratio'])}｜{_pct(base[k]['deepest'])}"]
        if clocks:
            c = clocks[k]
            row += [_range(c["ret"], _pct), f"24 個裡 {c['beat']} 個", _range(c["ratio"], _num), _range(c["deep"], _pct)]
        else:
            row += ["—"] * 4
        row += [_pct(hi_cost[k]["ret"]), _pct(late[k]["ret"])]
        t4.append(row)
    if clocks:
        hh = clocks["hold"]
        t4.append([NAMES["HOLD"], f"{_pct(hb['ret'])}｜{_num(hb['ratio'])}｜{_pct(hb['deepest'])}", _pct(hh["ret"]), "—",
                   _num(hh["ratio"]), _pct(hh["deepest"]), _pct(hb["ret"]), _pct(hb["ret"])])
    table4 = {"caption": (f"拆開檢查（只有比特幣，{BTC_MAIN[:4]} 年起）：在比特幣上報酬÷波動最高的三條指標規則"
                          f"（價量齊揚需要每小時成交量，不在候選內），加上兩條已驗證的順勢規則。把每天的收盤時間換成 24 個整點"
                          f"（用幣安 1 小時線重組日線，最高最低價取那一天裡每小時價格的最高最低）；"
                          f"另外照原本的 UTC 0 點收盤，把成本加到 {COST_HIGH * 100:.2f}%、或晚一天才照做。"
                          + (f"24 個時間從 {_d(clocks['start'])} 起算。" if clocks else "")),
              "columns": ["規則", "原本：每年報酬｜報酬÷波動｜最深虧", "24 個收盤時間的每年報酬", "每年報酬贏過抱著的收盤時間",
                          "24 個收盤時間的報酬÷波動", "24 個收盤時間的最深虧", f"成本 {COST_HIGH * 100:.2f}% 的每年報酬",
                          "晚一天照做的每年報酬"],
              "rows": t4}

    # ----- now -----
    def st(v) -> str:
        if not _ok(v):
            return "—"
        if 0.001 < v < 0.999:
            return f"持有 {v * 100:.0f}%"
        return "持有" if v >= 0.5 else "現金"

    sb, stw = btc["signal"].iloc[-1], tw["signal"].iloc[-1]
    now = [{"label": f"最後一根收完的日線", "value": f"比特幣 {_d(btc['last'])} 收盤 {float(btc['close'].iloc[-1]):,.0f} 美元；"
                                                     f"加權 {_d(tw['last'])} 收盤 {float(tw['close'].iloc[-1]):,.0f} 點"}]
    for k in RULES + BENCH:
        now.append({"label": f"照規則下一天的狀態：{NAMES[k]}", "value": f"比特幣 {st(sb[k])}；加權 {st(stw[k])}"})

    # ----- verdict -----
    def rv(r, k):
        return _num(r["whole"][k]["ratio"])
    main6 = (("K1", "KD 黃金交叉"), ("M1", "MACD"), ("R1", "RSI 超買超賣"), ("B1", "布林通道"), ("A1", "均線多頭排列"),
             ("V1", "價量齊揚"))
    parts = "、".join(f"{n} {rv(btc, k)}／{rv(tw, k)}" for k, n in main6)

    def beaters(r) -> str:
        won = [SHORT[k] for k in RULES if r["whole"][k]["ratio"] > r["whole"]["T55"]["ratio"]]
        return "、".join(won) if won else "沒有一條"

    verdict = (f"報酬÷波動（比特幣／加權）：{parts}；一直抱著 {_num(hb['ratio'])}／{_num(ht['ratio'])}，"
               f"已驗證的 55／20 {rv(btc, 'T55')}／{rv(tw, 'T55')}。"
               f"贏過 55／20 的：比特幣{beaters(btc)}、加權{beaters(tw)}")
    wb, wt = btc["walk"], tw["walk"]
    if wb["years"] and wt["years"]:
        verdict += (f"；每年事先挑最好的指標，比特幣每年 {_pct(wb['chosen']['ret'])}（抱著 {_pct(wb['hold']['ret'])}），"
                    f"加權 {_pct(wt['chosen']['ret'])}（{_pct(wt['hold']['ret'])}）")
    verdict += "。"
    # the mean-reversion rules: high win rate, but how did they do overall?
    mr = [k for k in ("R1", "B1", "K1") if board[k]["n_trades"]]
    mr = sorted(mr, key=lambda k: -(board[k]["wins"] or 0))[:2]
    if mr:
        bits = "、".join(f"{SHORT[k]} {board[k]['n_trades']} 筆裡 {board[k]['won']} 筆賺" for k in mr)
        if all(board[k]["ret"] == 0 for k in mr):
            tail = f"但每年報酬在 {n_mk} 個市場裡一個都沒贏過一直抱著"
        else:
            tail = "每年報酬贏過一直抱著的市場：" + "、".join(f"{SHORT[k]} {n_mk} 個裡 {board[k]['ret']} 個" for k in mr)
        verdict += f"勝率高不等於賺得多：{n_mk} 個市場合計 {bits}，{tail}。"

    notes = [
        "參數全部用課本和看盤軟體的預設值，事先定好、沒有調：KD(9,3,3)（K、D 從 50 開始，K = 2/3 前一天 K + 1/3 RSV）、"
        "MACD(12,26,9)、RSI(14，Wilder 平滑)、布林通道(20 日均線 ± 2 個標準差)、均線 5／10／20／60、季線 60 日、年線 240 日、"
        "50／200 日均線、價量齊揚（收盤創 20 日新高、成交量大於前 20 天平均的 1.5 倍買進，收盤創 10 日新低賣出）。"
        "KD 低檔黃金交叉指 K 由下往上穿過 D、而且 D 在 20 以下；高檔死亡交叉指 K 由上往下穿過 D、而且 D 在 80 以上。"
        "只要條件沒出現，部位就不動；所以低檔進場後如果一直沒有高檔死亡交叉，就會一直抱著。",
        "一次測了 14 條指標規則、6 個市場，總會有幾條剛好看起來不錯；單一條、單一市場的好成績要打折。"
        "這些市場彼此也有關係（加權、0050、台積電幾乎一起動，比特幣和以太幣也是），「幾個裡幾個」不是互相獨立的證據。",
        "加權指數和標普 500 是價格指數，不含股利，一直抱著的數字比實際拿到的少（台股一年約少 3–4%）；"
        "0050 和台積電用 Yahoo 還原拆股和配息的價格。現金沒有算利息，也沒有算稅（台股賣出還有 0.1%～0.3% 證交稅）。",
        f"成本只用一個粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%（第四張表有 {COST_HIGH * 100:.2f}% 的結果）。"
        "實際手續費、價差和滑價依券商、交易所和金額不同；股票假設剛好用收盤價成交。",
        "比特幣 2017-08 以前用 blockchain.info 的每日價格（只有收盤價），之後用幣安現貨日線（有最高最低價和成交量，UTC 0 點收盤）。"
        "所以第一、二張表的比特幣從 2018 年起算；事先挑選從 2014 年起的資料開始，那段 KD 只能用收盤價代替最高最低價。"
        "加權指數在 Yahoo 上 2003 年以前沒有成交量，所以從 2003 年起才能比；成交量是 0 的日子當成缺資料。",
        "勝率是把每一次「買進到賣出」算一筆、扣成本之後有沒有賺；還沒賣出的那一筆不算。勝率高不代表賺錢："
        "賺的時候賺得少、賠的時候賠得多，或者大漲時常常不在場上，整體就可能輸。",
        "每個指標都只用當天收盤為止已經知道的價格和成交量，隔天才持有；事先挑選只用 1 月 1 日以前的報酬。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼，不是買賣建議。",
    ]
    if clocks and clocks.get("problem"):
        notes.append(clocks["problem"])
    if clock_problem:
        notes.append(clock_problem)
    notes += problems

    asof = max(r["last"] for r in res.values())
    result = {
        "id": "retail_indicators",
        "title": "台灣散戶最常用的技術指標，真的有用嗎",
        "question": ("KD、MACD、RSI、布林通道、均線多頭排列、季線年線、價量齊揚，這些台灣和中文圈最常教的指標，"
                     "用課本參數、事先定好、換市場、換收盤時間來檢查，在比特幣和台股上有沒有贏過一直抱著和已驗證的順勢規則？"),
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": [table1, table2, table3, table4],
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTCUSDT、ETHUSDT）", "url": "https://www.binance.com/zh-TC/markets/overview"},
            {"name": "Yahoo Finance 台灣加權指數 ^TWII", "url": "https://finance.yahoo.com/quote/%5ETWII/history"},
            {"name": "Yahoo Finance 0050.TW", "url": "https://finance.yahoo.com/quote/0050.TW/history"},
            {"name": "Yahoo Finance 2330.TW", "url": "https://finance.yahoo.com/quote/2330.TW/history"},
            {"name": "Yahoo Finance 標普 500 ^GSPC", "url": "https://finance.yahoo.com/quote/%5EGSPC/history"},
        ],
    }
    return _clean(result)


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
