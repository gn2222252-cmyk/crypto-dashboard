"""Historical study "coinbase_premium": does the Coinbase premium (bitcoin on Coinbase in US dollars vs the
international price on Binance) tell anything about bitcoin's next move, are "a deep discount marks a bottom" and
"a positive premium confirms a rally" true as stated, and does the premium help the 50-day-average rule?

Claims tested exactly as published:
  * CoinDesk 2024-07-01: deeply negative Coinbase premium readings (Nov 2022, Aug 2023) foreshadowed local bottoms;
    a FalconX analyst calls the premium a reliable confirming and sometimes leading indicator.
  * TradingView "Coinbase Bitcoin Premium Index": negative = Americans selling, money leaving.
  * CryptoQuant / BGeometrics chart of the Coinbase premium (Coinbase USD vs Binance USDT).

Pre-registered tests (written before any result was seen; not changed afterwards):
Premium C_t (7-day average and daily), at the same 00:00 UTC timestamp for both exchanges (hourly if needed; check
timestamp alignment as kimchi_premium did).
1. Level: forward 7/30/90-day BTC returns by C quintile; split 2017-08..2019, 2020..2022, 2023..now (ETF era from
   2024 as description); block bootstrap.
2. Bottom claim: C in its trailing-1-year bottom 5% -> forward 30/90-day returns vs all days; independent episodes.
3. Confirmation claim: C > 0 (7-day avg) vs <= 0 -> forward 30-day returns, and as an entry confirmation: rule
   CB1 = ma50 AND C7 > 0.
4. Rule CB2 (bottom entry): when below ma50, enter early if C crosses from its bottom 5% back above its trailing-year
   median; exit on close < 20-day average or when ma50 takes over (as rally_timing does).
Evaluate CB1, CB2 vs ma50 on the fair grid (data from 2017-08 - as many starts as allowed), ordinary-person
execution, ETH (Coinbase ETH-USD vs Binance ETHUSDT), placebo with circularly shifted premium (200 draws),
crash-avoided share with whipsaws, re-entry days after bottoms.
Pass test (stated now): an effect with the same sign in all three periods and full-sample 90% interval excluding
zero, AND a rule beats ma50 on the ranking score with placebo share <= 10%, AND same direction on ETH.
Today: current premium (raw and USDT-adjusted), percentile, rule states - states only.

Details fixed with the design (before results):
  * C = Coinbase BTC-USD daily close / Binance BTCUSDT daily close - 1 (the practitioners' index: US dollars vs
    tether). Both daily candles open at 00:00 UTC and close at 00:00 UTC of the next day (Coinbase candle times are
    checked to sit exactly on a UTC midnight), so both prices are taken at the same instant. C7 = 7-day mean of C
    (at least 5 of 7 days).
  * USDT adjustment: no free daily USDT/USD series reaches back to 2017 (Coinbase USDT-USD starts 2021-05, Kraken's
    public OHLC keeps 720 days, Bitstamp has no USDT/USD history). The adjusted version therefore replaces Binance by
    Bitstamp BTC/USD (a dollar market, daily candles at the same UTC midnights): C_usd = Coinbase / Bitstamp - 1.
    That is the same as dividing Binance's price by the USDT/USD rate implied by Bitstamp. From 2021-05 it is
    checked against Coinbase's own USDT-USD close.
  * Primary series: raw C (as published). Tests 1 and 2 use the daily C; test 3 and both rules use C7. Checks: the
    other variant, C_usd and ETH are reported beside it.
  * Bottom 5% / median: of the 365 values before day t (at least 330), so nothing from day t or later.
  * Primary horizon for the pass test: 30 days. Level effect = correlation of C with the 30-day forward return;
    bottom effect = mean 30-day forward return on bottom days minus the mean over all days on which the bottom line
    exists; confirmation effect = mean 30-day forward return when C7 > 0 minus when C7 <= 0. "Same sign in all three
    periods" = same sign as the full-sample estimate. Block bootstrap: circular blocks of max(30, horizon) days, 1,000
    draws, 90% percentile interval. Circular shift of the signal days (1,000 draws) as a second check. Independent
    episodes: at least 30 days after the previous counted one.
  * CB2: armed on a day when C7 is in its trailing-year bottom 5%; the first later day with C7 above its
    trailing-year median is the trigger (and disarms). A trigger while ma50 is off opens an early position held until
    a close below the 20-day average or until ma50 holds (rally_timing._latch). Before C7 has a year of history CB2
    equals ma50.
  * Fair grid (main_signal_check's pieces, its seeds for ma50): monthly start dates 2017-09 ... 2023-12 on the UTC
    daily closes (textbook, 0.10%); 72 start months 2018-01 ... 2023-12 x 24 closing hours = 1,728 cells, textbook
    and ordinary person (decide at the clock, trade 10 hours later, 0.25%, 10% of entries missed, 200 simulations
    per clock); a decision at hour h of date D uses the premium of the UTC day D-1 (the last finished one). Fourth
    check (other asset) = ETH only, with its own premium, monthly starts from its first valid month to three years
    ago. Ranking score = the lowest of the four "beats holding" shares (eligible when the deepest loss is shallower
    than holding's in at least 95% of the bitcoin cells); "beats ma50" = ranked above ma50 in main_signal_check's
    order. Placebo: C rolled circularly by 180 ... (length - 180) days (200 draws), both rules rebuilt, share of draws
    with return / volatility from 2017-09-01 at least as high as the real rule. Crashes: crash_timing's 50%+ bear
    markets with a top after 2017-08-17: share avoided, whipsaws, days from the low to the first entry.
  * ETH "same direction": the passing effect has the same sign on ETH, and the passing rule's return / volatility on
    ETH (from 2017-09-01) beats ETH's ma50.
  * Look-ahead checks: candle timestamps; every price after three cut dates scrambled -> no decision up to the cut
    may change; one extra day of delay for the signals and the rules.
Added after the first results were seen (description only, never part of the pass test): a placebo for CB1 built
on the dollar-to-dollar premium, because that check version looked better than the primary one.

Data (public, key-less, the dashboard's fixed user agent): Coinbase Exchange public candles (BTC-USD through
kimchi_premium's loader and store, ETH-USD, USDT-USD), Binance daily closes (the dashboard's long daily series;
ETHUSDT through trend_assets), Bitstamp public OHLC (BTC/USD, ETH/USD).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import liquidity_lead as ll
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import rally_timing as rl
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
H = (7, 30, 90)
MAIN_H = 30
Q_LOW, WIN, WIN_MIN = 0.05, 365, 330
GAP = 30
BOOT, DRAWS = 1000, 200
FIRST = "2017-08-17"  # first Binance BTCUSDT day
RSTART = "2017-09-01"
PERIODS = (("2017-08–2019", FIRST, "2019-12-31"), ("2020–2022", "2020-01-01", "2022-12-31"),
           ("2023 起", "2023-01-01", None))
ETF_ERA = ("2024 起（現貨 ETF 之後，只描述）", "2024-01-11", None)
DAILY_GRID = ("2017-09-01", "2023-12-01")
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")
KEYS = ("ma50", "cb1", "cb2")
SEED_KI = {"ma50": 1, "cb1": 60, "cb2": 61}  # ma50 uses main_signal_check's draws
NAME = {"ma50": "50 日均線（現行主訊號）", "cb1": "CB1：50 日均線，而且溢價 7 日平均 > 0 才持有",
        "cb2": "CB2：50 日均線，加上「折價從一年最低 5% 回到一年中位數以上」提早上車"}
SHORT = {"ma50": "50 日均線", "cb1": "CB1", "cb2": "CB2"}

CB_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
BS_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"

SOURCES = [
    {"name": "CoinDesk 2024-07-01：Coinbase 溢價深度轉負（2022-11、2023-08）預告了局部底部；FalconX 分析師稱它是可靠的確認、有時領先的指標",
     "url": "https://coindesk.com/markets/2024/07/01/bitcoins-bottom-in-price-could-be-near-or-already-in-the-coinbase-premium-index-suggests"},
    {"name": "TradingView：Coinbase Bitcoin Premium Index（負值＝美國人在賣、資金流出）",
     "url": "https://www.tradingview.com/script/kqijNHFP-Coinbase-Bitcoin-Premium-Index/"},
    {"name": "BGeometrics（CryptoQuant 式）Coinbase 溢價圖", "url": "https://charts.bgeometrics.com/coinbase_premium.html"},
    {"name": "Coinbase Exchange 公開日線（BTC-USD、ETH-USD、USDT-USD）", "url": CB_URL.format(product="BTC-USD") + "?granularity=86400"},
    {"name": "幣安現貨日線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
    {"name": "Bitstamp 公開 API：BTC/USD、ETH/USD 日線", "url": "https://www.bitstamp.net/api/#ohlc_data"},
]

_ok, _num, _d = te._ok, te._num, te._d


def _pct(v, signed: bool = True) -> str:
    """v is a fraction."""
    return te._pct(v * 100, signed) if _ok(v) else "—"


def _pp(v) -> str:
    return f"{v * 100:+.1f}" if _ok(v) else "—"


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


# ---------- data ----------

def _candles_cb(product: str, first: str) -> tuple[dict, str]:
    """{YYYY-MM-DD: [close, candle start in s]} of finished Coinbase daily candles, kept in our own store."""
    path = common.STORE / f"coinbase_premium_cb_{product.lower()}.json"
    stored = km._read(path)
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - pd.Timedelta(days=5)) if stored else pd.Timestamp(first)
    try:
        while start < today:
            end = min(start + pd.Timedelta(days=299), today)
            page = km._get(CB_URL.format(product=product),
                           {"granularity": 86400, "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                            "end": end.strftime("%Y-%m-%dT00:00:00Z")}).json()
            for row in page:
                day = pd.Timestamp(int(row[0]), unit="s")
                if day < today and float(row[4]) > 0:
                    stored[_d(day)] = [float(row[4]), int(row[0])]
            start = end + pd.Timedelta(days=1)
        km._write(path, stored)
        return stored, ""
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        return stored, f"Coinbase {product} 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料（到 {max(stored)}）。"


def _candles_bs(pair: str) -> tuple[dict, str]:
    """{YYYY-MM-DD: [close, candle start in s]} of finished Bitstamp daily candles from 2017-07, own store."""
    path = common.STORE / f"coinbase_premium_bitstamp_{pair}.json"
    stored = km._read(path)
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - pd.Timedelta(days=5)) if stored else pd.Timestamp("2017-07-01")
    try:
        for _ in range(12):
            page = km._get(BS_URL.format(pair=pair), {"step": 86400, "limit": 1000,
                                                      "start": int(start.timestamp())}).json()["data"]["ohlc"]
            if not page:
                break
            for row in page:
                day = pd.Timestamp(int(row["timestamp"]), unit="s")
                if day < today and float(row["close"]) > 0:
                    stored[_d(day)] = [float(row["close"]), int(row["timestamp"])]
            nxt = pd.Timestamp(int(page[-1]["timestamp"]), unit="s") + pd.Timedelta(days=1)
            if nxt <= start or nxt >= today:
                break
            start = nxt
        km._write(path, stored)
        return stored, ""
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        return stored, f"Bitstamp {pair} 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料（到 {max(stored)}）。"


def _btc_check(stored: dict) -> dict:
    """One fresh page of Coinbase BTC-USD candles: are the candle times UTC midnights, and does kimchi_premium's
    store hold the same closes?"""
    today = pd.Timestamp(common.today_utc())
    page = km._get(CB_URL.format(product="BTC-USD"),
                   {"granularity": 86400, "start": (today - pd.Timedelta(days=290)).strftime("%Y-%m-%dT00:00:00Z"),
                    "end": today.strftime("%Y-%m-%dT00:00:00Z")}).json()
    rows = [r for r in page if pd.Timestamp(int(r[0]), unit="s") < today]
    mid = sum(1 for r in rows if int(r[0]) % 86400 == 0)
    same = sum(1 for r in rows if abs(stored.get(_d(pd.Timestamp(int(r[0]), unit="s")), -1) - float(r[4])) < 1e-6)
    return {"n": len(rows), "mid": mid, "same": same}


def _series(d: dict, days: pd.DatetimeIndex, i: int = 0) -> pd.Series:
    s = pd.Series({pd.Timestamp(k): (v[i] if isinstance(v, list) else v) for k, v in d.items()}).sort_index()
    return s.reindex(days)


def _load() -> dict:
    problems = []
    close = te._daily_close()
    days = pd.date_range(FIRST, close.index[-1], freq="D")
    cb, n = km._coinbase()
    problems += [n] if n else []
    try:
        chk = _btc_check(cb)
    except Exception as exc:  # noqa: BLE001 - the check is optional
        chk, _ = None, problems.append(f"Coinbase BTC-USD 時間檢查這次沒有做成（{type(exc).__name__}）。")
    bs, n = _candles_bs("btcusd")
    problems += [n] if n else []
    eth_cb, n = _candles_cb("ETH-USD", "2017-08-01")
    problems += [n] if n else []
    eth_bs, n = _candles_bs("ethusd")
    problems += [n] if n else []
    try:
        usdt, n = _candles_cb("USDT-USD", "2021-04-01")
        problems += [n] if n else []
    except Exception as exc:  # noqa: BLE001 - only used for a check
        usdt = {}
        problems.append(f"Coinbase USDT-USD 這次取不到（{type(exc).__name__}），USDT 校正的核對沒有做。")
    eth = ta._crypto_close("ETHUSDT")
    eth_days = pd.date_range(FIRST, eth.index[-1], freq="D")
    return {"close": close, "days": days, "cb": _series(cb, days), "bs": _series(bs, days),
            "bs_ts": _series(bs, days, 1), "eth": eth, "eth_days": eth_days, "eth_cb": _series(eth_cb, eth_days),
            "eth_cb_ts": _series(eth_cb, eth_days, 1), "eth_bs": _series(eth_bs, eth_days),
            "usdt": _series(usdt, days) if usdt else pd.Series(np.nan, index=days), "chk": chk, "problems": problems}


# ---------- premium, flags, rules ----------

def _premium(price: pd.Series, ref: pd.Series) -> pd.Series:
    return price / ref - 1


def _flags(c: pd.Series) -> pd.DataFrame:
    """All signal pieces of one daily premium series (each uses values up to its own day only)."""
    c7 = c.rolling(7, min_periods=5).mean()
    lo = c.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(Q_LOW)
    lo7 = c7.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(Q_LOW)
    med7 = c7.shift(1).rolling(WIN, min_periods=WIN_MIN).median()
    bottom7 = ((c7 < lo7) & lo7.notna()).values
    above = ((c7 > med7) & med7.notna()).values
    trig = np.zeros(len(c), dtype=bool)
    armed = np.zeros(len(c), dtype=bool)
    on = False
    for i in range(len(c)):
        if bottom7[i]:
            on = True
        elif on and above[i]:
            trig[i], on = True, False
        armed[i] = on
    return pd.DataFrame({"c": c, "c7": c7, "lo": lo, "lo7": lo7, "med7": med7,
                         "bottom": (c < lo) & lo.notna(), "bottom7": bottom7, "pos7": (c7 > 0) & c7.notna(),
                         "def_b": lo.notna() & c.notna(), "def_b7": lo7.notna() & c7.notna(),
                         "def_p": c7.notna(), "trig": trig, "armed": armed}, index=c.index)


def _rules_on(c: np.ndarray, pos7: np.ndarray, trig: np.ndarray) -> dict:
    """ma50, CB1, CB2 after each close of one close series; pos7 / trig are already the values a decision on that
    close may use (0 / False where unknown)."""
    s = pd.Series(c)
    m50 = s.rolling(50, min_periods=50).mean()
    m20 = s.rolling(20, min_periods=20).mean()
    ma = (s > m50).astype(float).where(m50.notna()).values
    ready = (m50.notna() & m20.notna()).values
    cb2 = rl._latch(ready, np.nan_to_num(ma, nan=0.0), trig, c, m20.values)
    return {"ma50": ma, "cb1": ma * pos7.astype(float), "cb2": cb2}


def _targets(close: pd.Series, fl: pd.DataFrame, extra: int = 0) -> pd.DataFrame:
    """Daily (UTC close) targets; extra = days of additional delay before the premium may be used."""
    idx = close.index
    pos7 = fl["pos7"].astype(float).reindex(idx).shift(extra).fillna(0.0).values
    trig = fl["trig"].astype(float).reindex(idx).shift(extra).fillna(0.0).values > 0.5
    out = _rules_on(close.values.astype(float), pos7, trig)
    return pd.DataFrame(out, index=idx)


# ---------- statistics ----------

def _diff(flag: np.ndarray, r: np.ndarray) -> float | None:
    if flag.sum() == 0 or (~flag).sum() == 0:
        return None
    return float(r[flag].mean() - r[~flag].mean())


def _boot_diff(flag: np.ndarray, r: np.ndarray, h: int, rng) -> tuple:
    idx = km._blocks(len(r), max(30, h), BOOT, rng)
    f, x = flag[idx], r[idx]
    c1, c0 = f.sum(axis=1), (~f).sum(axis=1)
    good = (c1 > 0) & (c0 > 0)
    d = (f * x).sum(axis=1)[good] / c1[good] - ((~f) * x).sum(axis=1)[good] / c0[good]
    return (float(np.percentile(d, 5)), float(np.percentile(d, 95))) if len(d) > 50 else (None, None)


def _shift_diff(flag: np.ndarray, r: np.ndarray, real, sign: int, rng) -> float | None:
    if real is None or len(flag) < 400:
        return None
    hits = 0
    for s in rng.integers(180, len(flag) - 180, BOOT):
        e = _diff(np.roll(flag, int(s)), r)
        hits += int(e is not None and sign * e >= sign * real)
    return hits / BOOT


def _effect(kind: str, x: pd.Series, valid: pd.Series, fwd: dict, rng, sign: int = 1, ci: bool = True) -> dict:
    """kind 'corr' (x = premium), 'excess' (x = flag vs all valid days) or 'diff' (x = flag vs the other days)."""
    out = {"kind": kind}
    for h in H:
        f = fwd[h].reindex(x.index)
        ok = (f.notna() & valid.reindex(x.index).fillna(False).astype(bool) & x.notna()).values
        idx = x.index[ok]
        r = f.values[ok].astype(float)
        if kind == "corr":
            xv = x.values[ok].astype(float)
            est = float(np.corrcoef(xv, r)[0, 1]) if len(r) > 60 else None

            def part(m, xv=xv, r=r):
                return float(np.corrcoef(xv[m], r[m])[0, 1]) if m.sum() > 60 else None
        else:
            fl = x.values[ok].astype(bool)
            est = km._excess(fl, r) if kind == "excess" else _diff(fl, r)

            def part(m, fl=fl, r=r):
                if m.sum() <= 60:
                    return None
                return km._excess(fl[m], r[m]) if kind == "excess" else _diff(fl[m], r[m])
        y = {"est": est, "n": int(ok.sum()), "all": float(r.mean()) if len(r) else None}
        if kind != "corr":
            y["mean"] = float(r[fl].mean()) if fl.sum() else None
            y["win"] = float((r[fl] > 0).mean()) if fl.sum() else None
            y["mean0"] = float(r[~fl].mean()) if (~fl).sum() else None
            y["days"] = int(fl.sum())
        if h == MAIN_H:
            y["per"] = [part(km._span(idx, a, b)) for _, a, b in PERIODS]
            y["etf"] = part(km._span(idx, ETF_ERA[1], ETF_ERA[2]))
            if ci:
                if kind == "corr":
                    y["ci"] = km._boot_corr(xv, r, h, rng)
                    y["shift"] = None
                elif kind == "excess":
                    y["ci"] = km._boot_excess(fl, r, h, rng)
                    y["shift"] = km._shift_share(fl, r, est, sign, rng)
                else:
                    y["ci"] = _boot_diff(fl, r, h, rng)
                    y["shift"] = _shift_diff(fl, r, est, sign, rng)
        out[h] = y
    if kind != "corr":
        flag = x.fillna(False).astype(bool) & valid.reindex(x.index).fillna(False).astype(bool)
        dates = list(flag.index[flag.values])
        out["indep"] = km._independent(dates)
        eps, last = [], None
        for d in dates:
            if last is None or (d - last).days >= GAP:
                eps.append(d)
                last = d
        out["episodes"] = eps
    return out


def _passes(e: dict) -> dict:
    m = e[MAIN_H]
    est, ci = m["est"], m.get("ci", (None, None))
    if not _ok(est):
        return {"pass": False, "same": 0, "excl": False, "sign": 0}
    s = 1 if est > 0 else -1
    same = sum(1 for v in m["per"] if _ok(v) and (1 if v > 0 else -1) == s)
    excl = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
    return {"pass": same == 3 and excl, "same": same, "excl": excl, "sign": s}


# ---------- fair grid ----------

def _daily_grid(close: pd.Series, target: pd.DataFrame, first, last) -> dict:
    valid = target["ma50"].notna()
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    change = close.pct_change().fillna(0.0).values
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out = {"starts": labels, "hold": (hr[:, 0], hd[:, 0])}
    for k in KEYS:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, fl: pd.DataFrame) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, YEAR)
    hr, hd = hr[:, 0], hd[:, 0]
    # a decision at hour h of date D knows the UTC daily premium up to the candle labelled D - 1
    known = fl[["pos7", "trig"]].astype(float).reindex(m.dates - pd.Timedelta(days=1)).fillna(0.0)
    text = {k: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for k in KEYS}
    person = {k: (np.zeros((len(starts), 24, SIMS)), np.zeros((len(starts), 24, SIMS))) for k in KEYS}
    for h in range(24):
        closes = np.concatenate([pre, price[h::24]]).astype(float)
        pos7 = np.zeros(len(closes))
        trig = np.zeros(len(closes), dtype=bool)
        pos7[npre:npre + days] = known["pos7"].values
        trig[npre:npre + days] = known["trig"].values > 0.5
        rules = _rules_on(closes, pos7, trig)
        for k in KEYS:
            w = np.nan_to_num(rules[k][npre:][:days], nan=0.0).astype(float)
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * SEED_KI[k] + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]), starts, YEAR)
            person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
    return {"starts": labels, "hold": (hr, hd), "text": text, "person": person}


def _summary(d: dict, c: dict, o: dict) -> dict:
    hr, hd = c["hold"]
    s = {}
    for k in KEYS:
        dr, dd = d[k]
        tr, td = c["text"][k]
        edge = tr - hr[:, None]
        pr, pdd = c["person"][k]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > d["hold"][0]).sum()), "d_shallow": int((dd > d["hold"][1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "o_n": len(o[k][0]) if o else 0, "o_beat": int((o[k][0] > o["hold"][0]).sum()) if o else 0}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        s[k] = x
    order = sorted(KEYS, key=lambda k: (not s[k]["eligible"], -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "order": order, "hold_deep_med": float(np.median(hd))}


def _ratio(target: pd.Series, change: pd.Series, delay: int = 1) -> float | None:
    m = ro._measure(ll._window(ll._net(target, change, COST, delay=delay), RSTART, None), YEAR)
    return m["ratio"] if m else None


# ---------- the whole computation ----------

def _compute(data: dict) -> dict:
    close, days = data["close"], data["days"]
    bn = close.reindex(days)
    raw = _premium(data["cb"], bn)
    usd = _premium(data["cb"], data["bs"])
    fl = _flags(raw)
    flu = _flags(usd)
    eth, ed = data["eth"], data["eth_days"]
    eraw = _premium(data["eth_cb"], eth.reindex(ed))
    fle = _flags(eraw)
    rng = np.random.default_rng(SEED)
    fwd = {h: (close.shift(-h) / close - 1).reindex(days) for h in H}
    efwd = {h: (eth.shift(-h) / eth - 1).reindex(ed) for h in H}

    # ----- test 1: level -----
    ok = raw.notna()
    edges = np.nanpercentile(raw.values, [0, 20, 40, 60, 80, 100])
    q = pd.Series(np.clip(np.searchsorted(edges[1:-1], raw.values, side="right"), 0, 4), index=days).where(ok)
    quint = []
    for i in range(5):
        m = q == i
        row = {"days": int(m.sum())}
        for h in H:
            f = fwd[h][m].dropna()
            row[h] = float(f.mean()) if len(f) else None
        f = fwd[MAIN_H][m].dropna()
        row["win"] = float((f > 0).mean()) if len(f) else None
        quint.append(row)
    allrow = {h: float(fwd[h][ok].dropna().mean()) for h in H}
    allrow["win"] = float((fwd[MAIN_H][ok].dropna() > 0).mean())
    allrow["days"] = int(ok.sum())
    one = pd.Series(True, index=days)
    eff = {
        "level": _effect("corr", raw, one, fwd, rng),
        "level7": _effect("corr", fl["c7"], one, fwd, rng),
        "level_usd": _effect("corr", usd, one, fwd, rng),
        "bottom": _effect("excess", fl["bottom"], fl["def_b"], fwd, rng, 1),
        "bottom7": _effect("excess", pd.Series(fl["bottom7"].values, index=days), fl["def_b7"], fwd, rng, 1),
        "bottom_usd": _effect("excess", flu["bottom"], flu["def_b"], fwd, rng, 1),
        "conf": _effect("diff", fl["pos7"], fl["def_p"], fwd, rng, 1),
        "conf_raw": _effect("diff", (raw > 0) & raw.notna(), raw.notna(), fwd, rng, 1),
        "conf_usd": _effect("diff", flu["pos7"], flu["def_p"], fwd, rng, 1),
    }
    one_e = pd.Series(True, index=ed)
    eeff = {"level": _effect("corr", eraw, one_e, efwd, rng),
            "bottom": _effect("excess", fle["bottom"], fle["def_b"], efwd, rng, 1),
            "conf": _effect("diff", fle["pos7"], fle["def_p"], efwd, rng, 1)}
    late = {"bottom": _effect("excess", fl["bottom"].shift(1, fill_value=False), fl["def_b"], fwd, rng, ci=False),
            "conf": _effect("diff", fl["pos7"].shift(1, fill_value=False), fl["def_p"].shift(1, fill_value=False),
                            fwd, rng, ci=False)}

    # ----- rules -----
    idx = close.index
    change = close.pct_change().fillna(0.0)
    T = _targets(close, fl)
    TU = _targets(close, flu)
    ech = eth.pct_change().fillna(0.0)
    TE = _targets(eth, fle)
    ratio = {k: _ratio(T[k], change) for k in KEYS}
    ratio_usd = {k: _ratio(TU[k], change) for k in KEYS}
    ratio_eth = {k: _ratio(TE[k], ech) for k in KEYS}
    ratio_late = {k: _ratio(_targets(close, fl, extra=1)[k], change, delay=2) for k in KEYS}
    hold_ratio = _ratio(pd.Series(1.0, index=idx), change)

    daily = _daily_grid(close, T, DAILY_GRID[0], DAILY_GRID[1])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre, fl)
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    try:
        other = _daily_grid(eth, TE, RSTART, three)
    except Exception:  # noqa: BLE001
        other = {}
    sm = _summary(daily, clock, other)

    # placebo: roll the premium
    real = {k: ratio[k] for k in ("cb1", "cb2")}
    better = {"cb1": 0, "cb2": 0}
    prng = np.random.default_rng(SEED + 2)
    n = len(days)
    for s in prng.integers(180, n - 180, DRAWS):
        f2 = _flags(pd.Series(np.roll(raw.values, int(s)), index=days))
        t2 = _targets(close, f2)
        for k in better:
            v = _ratio(t2[k], change)
            better[k] += int(v is not None and real[k] is not None and v >= real[k])
    placebo = {k: better[k] / DRAWS for k in better}
    # added after the first results were seen (not part of the pass test): the same placebo for CB1 built on the
    # dollar-to-dollar premium
    real_u, hits = _ratio(TU["cb1"], change), 0
    urng = np.random.default_rng(SEED + 3)
    for s in urng.integers(180, n - 180, DRAWS):
        t2 = _targets(close, _flags(pd.Series(np.roll(usd.values, int(s)), index=days)))
        v = _ratio(t2["cb1"], change)
        hits += int(v is not None and real_u is not None and v >= real_u)
    placebo_usd = hits / DRAWS

    # crashes, whipsaws, re-entry
    c_arr = close.values
    first = int(idx.searchsorted(pd.Timestamp(RSTART)))
    bears = [b for b in ct._bears(close) if idx[b["top"]] >= pd.Timestamp(FIRST)]
    crash = []
    for b in bears:
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for k in KEYS:
            row[k] = ct._crash(c_arr, [(1.0, T[k].values)], b)["avoid"]
            _, en = ct._first_moves(c_arr, T[k].values, b)
            row[k + "_in"] = (en - b["low"]) if en is not None else None
        crash.append(row)
    whip = {k: ct._whipsaws(c_arr, [(1.0, T[k].values)], first) for k in KEYS}
    after = idx >= pd.Timestamp(RSTART)
    diff_days = {"cb1": int(((T["ma50"] >= 0.5) & (T["cb1"] < 0.5))[after].sum()),
                 "cb2": int(((T["ma50"] < 0.5) & (T["cb2"] >= 0.5))[after].sum())}
    trig_n = int(fl["trig"].sum())

    return {"close": close, "days": days, "raw": raw, "usd": usd, "fl": fl, "flu": flu, "fle": fle, "eraw": eraw,
            "edges": edges, "quint": quint, "allrow": allrow, "eff": eff, "eeff": eeff, "late": late, "T": T,
            "TE": TE, "ratio": ratio, "ratio_usd": ratio_usd, "ratio_eth": ratio_eth, "ratio_late": ratio_late,
            "hold_ratio": hold_ratio, "daily": daily, "clock": clock, "other": other, "sm": sm,
            "placebo": placebo, "placebo_usd": placebo_usd, "crash": crash, "whip": whip, "diff_days": diff_days, "trig_n": trig_n,
            "problem": problem}


# ---------- look-ahead checks ----------

def _lookahead(data: dict, r: dict) -> list[str]:
    out = []
    chk = data["chk"]
    ets = data["eth_cb_ts"].dropna().astype(np.int64)
    bts = data["bs_ts"].dropna().astype(np.int64)

    def midnight(s: pd.Series) -> tuple[int, int]:
        start = (s.index.values.astype("datetime64[s]").astype(np.int64))
        return int((s.values == start).sum()), len(s)

    e_ok, e_n = midnight(ets)
    b_ok, b_n = midnight(bts)
    btc = (f"Coinbase BTC-USD 最近 {chk['n']} 根裡 {chk['mid']} 根的開始時間正好是 UTC 0 點、{chk['same']} 根收盤價和 kimchi_premium 存的一樣；"
           if chk else "")
    out.append("偷看未來的檢查一（時間對齊）：" + btc + f"Coinbase ETH-USD {e_n:,} 根裡 {e_ok:,} 根、Bitstamp BTC/USD {b_n:,} 根裡 {b_ok:,} 根"
               "的開始時間正好是那一天的 UTC 0 點；幣安日線也是 UTC 0 點開、隔天 0 點收，所以兩邊價格是同一刻的收盤。")
    rng = np.random.default_rng(11)
    total = same = 0
    close, days = data["close"], data["days"]
    for cut in CUTS:
        c = pd.Timestamp(cut)
        noisy_close = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        cb = data["cb"]
        noisy_cb = cb.where(cb.index <= c, cb * np.exp(rng.normal(0, 0.2, len(cb))))
        f2 = _flags(_premium(noisy_cb, noisy_close.reindex(days)))
        t2 = _targets(noisy_close, f2)
        m = close.index <= c
        for k in KEYS:
            total += 1
            same += int(np.array_equal(np.nan_to_num(r["T"][k].values[m], nan=-1), np.nan_to_num(t2[k].values[m], nan=-1)))
        md = days <= c
        for k in ("bottom", "bottom7", "pos7", "trig", "armed"):
            total += 1
            same += int(np.array_equal(r["fl"][k].values[md].astype(bool), f2[k].values[md].astype(bool)))
    out.append(f"偷看未來的檢查二：在 {'、'.join(CUTS)} 三個切點，把切點之後的比特幣收盤價和 Coinbase 價格全部換成亂數再算一次，"
               f"切點以前每一天的訊號和進出狀態 {total} 組裡 {same} 組完全沒變" + ("。" if same == total else "（有變動，結果不能用）。"))
    e, lt = r["eff"], r["late"]
    rb = "、".join(f"{SHORT[k]} {_num(r['ratio'][k])} → {_num(r['ratio_late'][k])}" for k in KEYS)
    out.append(f"偷看未來的檢查三（再晚一天）：訊號晚一天才用，「一年最低 5%」之後 30 天比全部多 {_pp(e['bottom'][MAIN_H]['est'])} → "
               f"{_pp(lt['bottom'][MAIN_H]['est'])} 個百分點，「7 日平均為正」比非正多 {_pp(e['conf'][MAIN_H]['est'])} → "
               f"{_pp(lt['conf'][MAIN_H]['est'])}；規則的溢價晚一天、買賣再晚一天，{RSTART} 起報酬÷波動 {rb}。")
    return out


def _usd_note(eff: dict, r: dict, rank: dict) -> str:
    """The dollar-to-dollar check in words (description only, not part of the pass test)."""
    parts = []
    for k, label in (("level_usd", "溢價高低的相關"), ("bottom_usd", "一年最低 5% 的效果"), ("conf_usd", "7 日平均為正的效果")):
        m = eff[k][MAIN_H]
        p = _passes(eff[k])
        ci = m.get("ci", (None, None))
        val = _num(m["est"]) if k == "level_usd" else _pp(m["est"])
        rng_t = (f"{_num(ci[0])} 到 {_num(ci[1])}" if k == "level_usd" else f"{_pp(ci[0])} 到 {_pp(ci[1])}") if ci[0] is not None else "—"
        parts.append(f"{label} {val}（{rng_t}，三段同方向 {p['same']}／3）")
    ru = r["ratio_usd"]
    return ("美元對美元版（Coinbase ÷ Bitstamp）的結果：" + "；".join(parts)
            + f"。用它做的 CB1 在 {RSTART} 起報酬÷波動 {_num(ru['cb1'])}（50 日均線 {_num(ru['ma50'])}），把它的溢價整串循環平移 {DRAWS} 次，"
            f"一樣好或更好的比例 {_share(r['placebo_usd'])}。這一版是事先定的核對，但「拿它做規則再做隨機錯位」是看到結果後才加的，"
            "也沒有跑公平考試和一般人做法，不算進過關標準，只能當下一次要事先寫好再測的線索。")


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    days, raw, usd, fl, flu = r["days"], r["raw"], r["usd"], r["fl"], r["flu"]
    eff, eeff = r["eff"], r["eeff"]
    pnames = [p[0] for p in PERIODS]

    def ci_text(m) -> str:
        ci = m.get("ci", (None, None))
        return f"（{_pp(ci[0])} 到 {_pp(ci[1])}）" if ci and ci[0] is not None else ""

    def cc(e) -> str:
        m = e[MAIN_H]
        ci = m.get("ci", (None, None))
        return f"{_num(m['est'])}（{_num(ci[0])} 到 {_num(ci[1])}）" if ci and ci[0] is not None else _num(m["est"])

    # ----- table 1: level -----
    rows1 = []
    e = r["edges"]
    for i, q in enumerate(r["quint"]):
        rows1.append([f"第 {i + 1} 組（{'最低' if i == 0 else '最高' if i == 4 else '中間'}）", f"{_pct(e[i])} 到 {_pct(e[i + 1])}",
                      f"{q['days']:,}"] + [_pct(q[h]) for h in H] + [_share(q["win"])])
    a = r["allrow"]
    rows1.append(["全部日子", "—", f"{a['days']:,}"] + [_pct(a[h]) for h in H] + [_share(a["win"])])
    lv = eff["level"]
    for j, name in enumerate(pnames):
        rows1.append([f"相關係數：{name}", "—", "—", "—", _num(lv[MAIN_H]["per"][j]), "—", "—"])
    rows1.append([f"相關係數：{ETF_ERA[0]}", "—", "—", "—", _num(lv[MAIN_H]["etf"]), "—", "—"])
    rows1.append(["相關係數：全部（90% 區間）", "—", "—", _num(lv[7]["est"]), cc(lv), _num(lv[90]["est"]), "—"])
    rows1.append(["對照：7 日平均溢價的相關（全部）", "—", "—", _num(eff["level7"][7]["est"]), cc(eff["level7"]),
                  _num(eff["level7"][90]["est"]), "—"])
    rows1.append(["對照：美元對美元版（Coinbase ÷ Bitstamp）", "—", "—", _num(eff["level_usd"][7]["est"]), cc(eff["level_usd"]),
                  _num(eff["level_usd"][90]["est"]), "—"])
    rows1.append(["對照：以太幣（Coinbase ETH-USD ÷ 幣安 ETHUSDT）", "—", "—", _num(eeff["level"][7]["est"]), cc(eeff["level"]),
                  _num(eeff["level"][90]["est"]), "—"])
    t1 = {"caption": ("檢查一：溢價高低和之後的比特幣。Coinbase 溢價 ＝ Coinbase 比特幣美元價 ÷ 幣安比特幣 USDT（泰達幣）價 − 1，兩邊都是每天 UTC 0 點"
                      f"（台灣早上 8 點）同一刻收盤。{_d(days[0])} 到 {_d(days[-1])} 每天一個樣本，按當天溢價分五組（分界用全期間，是事後分的），"
                      "看之後 7／30／90 天比特幣平均漲跌。下半是溢價和之後漲跌的相關係數（+1 完全同向、0 沒關係、−1 完全反向），各段只列 30 天；"
                      "90% 區間是 30 天一段（90 天那欄 90 天）整段重抽 1,000 次。之後的報酬天天重疊，獨立的樣本大約只有天數 ÷ 天數期間。"),
          "columns": ["溢價分組", "溢價範圍", "天數", "之後 7 天平均", "之後 30 天平均", "之後 90 天平均", "30 天上漲比例"],
          "rows": rows1}

    # ----- table 2: the claims -----
    names = {"bottom": "溢價在過去一年最低 5%（每日）", "bottom7": "溢價 7 日平均在過去一年最低 5%",
             "bottom_usd": "美元對美元版，在過去一年最低 5%", "conf": "溢價 7 日平均 > 0（對 ≤ 0）",
             "conf_raw": "每日溢價 > 0（對 ≤ 0）", "conf_usd": "美元對美元版 7 日平均 > 0（對 ≤ 0）"}

    def row2(label: str, x: dict, unit: str = "天") -> list:
        m = x[MAIN_H]
        sh = m.get("shift")
        return [label, f"{m['days']:,} {unit}｜{x['indep']} 段",
                f"{_pct(m['mean'])}｜{_share(m['win'])}（其他日子 {_pct(m['mean0'])}）", _pct(x[90]["mean"]),
                _pp(m["est"]) + ci_text(m), _share(sh), "／".join(_pp(v) for v in m["per"])]

    rows2 = [row2(names[k], eff[k]) for k in ("bottom", "bottom7", "bottom_usd")]
    rows2.append(row2("以太幣：溢價在過去一年最低 5%", eeff["bottom"]))
    rows2 += [row2(names[k], eff[k]) for k in ("conf", "conf_raw", "conf_usd")]
    rows2.append(row2("以太幣：溢價 7 日平均 > 0（對 ≤ 0）", eeff["conf"]))
    b0 = eff["bottom"]
    rows2.append(["全部日子（對照，有一年資料之後）", f"{b0[MAIN_H]['n']:,} 天", f"{_pct(b0[MAIN_H]['all'])}｜—", _pct(b0[90]["all"]),
                  "—", "—", "—"])
    t2 = {"caption": ("檢查二、三：照網路說法原樣的訊號，訊號當天收盤起算之後的比特幣漲跌。「一年最低 5%」的門檻只用前 365 天（至少 330 天）的溢價，"
                      "所以 2018-07 以後才有值；「段」是和上一次至少隔 30 天才算新的一段。最低 5% 那幾列的「效果」＝訊號日之後 30 天平均 − 全部日子平均；"
                      "「> 0」那幾列的「效果」＝ 溢價為正的日子之後 30 天平均 − 溢價 ≤ 0 的日子（百分點）。括號是 30 天一段整段重抽 1,000 次的 90% 區間；"
                      "「隨機錯開一樣好」是把訊號日整串循環平移隨機天數 1,000 次，往說法的方向（之後比較好）一樣好或更好的比例；"
                      f"最後一欄依序是 {'、'.join(pnames)} 三段各自的效果。"),
          "columns": ["訊號", "出現天數｜獨立段數", "之後 30 天：平均｜上漲比例", "之後 90 天平均", "30 天效果（90% 區間）",
                      "隨機錯開一樣好", "三段：30 天效果"],
          "rows": rows2}

    # ----- table 3: fair grid -----
    sm, plc = r["sm"], r["placebo"]
    s = sm["s"]
    rank = {k: i for i, k in enumerate(sm["order"], 1)}
    dn, cn = s["ma50"]["d_n"], s["ma50"]["c_n"]
    on = s["ma50"]["o_n"]
    rows3 = []
    for k in KEYS:
        x = s[k]
        ck = x["checks"]
        rows3.append([NAME[k], _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]) if x["o_n"] else "沒有資料",
                      f"{_share(x['low'])}（第 {rank[k]}）" + ("" if x["eligible"] else "，虧損條件未達"),
                      f"{te._pct(x['c_deep_med'])}｜{cr._pp(x['c_edge'])}", _share(plc[k]) if k in plc else "—"])
    ost = r["other"]["starts"] if r["other"] else []
    t3 = {"caption": (f"檢查四：main_signal_check 的公平考試。比特幣 UTC 0 點日線 {DAILY_GRID[0][:7]} 到 {DAILY_GRID[1][:7]} 每月 1 日起算（{dn} 個，"
                      f"溢價 2017-08 才有，所以不能從 2015 起）；72 個起算月 × 24 個收盤時間（{cn:,} 格），收盤時間 h 的決定只用前一個 UTC 日的溢價；"
                      f"一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次）；第四項只用以太幣自己的 Coinbase 溢價"
                      + (f"（{_d(ost[0])[:7]} 到 {_d(ost[-1])[:7]}，{on} 個起算月）" if ost else "") +
                      "。各格是「報酬贏過一直抱著」的比例；排名看四項裡最低的那項（前提：最深虧損比一直抱著淺的比例至少 95%）。"
                      f"「隨機錯位一樣好」是把溢價整串循環平移隨機天數（{DRAWS} 次）重新做規則，{RSTART} 起報酬÷波動和真的一樣好或更好的比例。"),
          "columns": ["規則", f"起算月（{dn}）", f"起算×時間（{cn:,}）", "一般人照做", "以太幣起算月", "四項最低（排名）",
                      "最深虧損中位數｜每年多賺（中位數）", "隨機錯位一樣好"],
          "rows": rows3}

    # ----- table 4: crashes and more -----
    rows4 = []
    for b in r["crash"]:
        rows4.append([f"{_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）"), _pct(b["drop"])]
                     + [f"{_share(b[k])}｜" + (f"{b[k + '_in']} 天" if b[k + "_in"] is not None else "—") for k in KEYS])
    w = r["whip"]
    rows4.append(["一年下車幾次｜其中 30 天內更高價買回", "—"] + [f"{w[k]['exits']:.1f}｜{w[k]['whips']:.1f}" for k in KEYS])
    rows4.append([f"{RSTART} 起報酬÷波動（比特幣；一直抱著 {_num(r['hold_ratio'])}）", "—"] + [_num(r["ratio"][k]) for k in KEYS])
    rows4.append(["同上，溢價改用美元對美元版", "—"] + [_num(r["ratio_usd"][k]) for k in KEYS])
    rows4.append(["同上，以太幣用自己的溢價", "—"] + [_num(r["ratio_eth"][k]) for k in KEYS])
    rows4.append(["和均線不同的天數（CB1：均線持有它現金；CB2：均線現金它持有）", "—", "—",
                  f"{r['diff_days']['cb1']:,}", f"{r['diff_days']['cb2']:,}（觸發 {r['trig_n']} 次）"])
    t4 = {"caption": ("大跌時躲掉多少、低點後多久上車：比特幣從高點收盤跌 50% 以上的段落（crash_timing 的定義，高點在 2017-08 以後）。每格是"
                      "「這段跌幅裡躲掉的比例｜低點之後幾天第一次上車」（躲掉比例用規則自己在這段的漲跌算，中途反覆進出也算；0 天 ＝ 低點當天已在車上）。"
                      f"下車次數從 {RSTART} 算；「更高價買回」是下車後 30 天內又在更高的收盤價買回。照課本做，成本 {COST * 100:.2f}%。"),
          "columns": ["大跌段（高點 → 低點）", "跌幅", "50 日均線", "CB1", "CB2"], "rows": rows4}
    tables = [t1, t2, t3, t4]

    # ----- pass test -----
    pe = {"level": _passes(eff["level"]), "bottom": _passes(eff["bottom"]), "conf": _passes(eff["conf"])}
    eth_same = {k: pe[k]["pass"] and _ok(eeff[k][MAIN_H]["est"]) and (1 if eeff[k][MAIN_H]["est"] > 0 else -1) == pe[k]["sign"]
                for k in pe}
    beats = {k: rank[k] < rank["ma50"] for k in ("cb1", "cb2")}
    rule_ok = {k: beats[k] and plc[k] <= 0.10 for k in beats}
    eth_rule = {k: rule_ok[k] and _ok(r["ratio_eth"][k]) and _ok(r["ratio_eth"]["ma50"])
                and r["ratio_eth"][k] > r["ratio_eth"]["ma50"] for k in beats}
    passed = any(eth_same.values()) and any(eth_rule.values())

    # ----- today -----
    last = days[-1]
    rl_ = raw.dropna()
    ul = usd.dropna()

    def pctile(series: pd.Series) -> str:
        s_ = series.dropna()
        if len(s_) < WIN_MIN + 1:
            return "—"
        prior = s_.iloc[-WIN - 1:-1]
        return f"{(prior < s_.iloc[-1]).mean() * 100:.0f}%"

    T = r["T"]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    lo, med = fl["lo"].iloc[-1], fl["med7"].iloc[-1]
    armed = bool(fl["armed"].iloc[-1])
    trig_dates = list(fl.index[fl["trig"].values])
    er = r["eraw"].dropna()
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(r['close'].iloc[-1]):,.0f} 美元（幣安 USDT）；Coinbase {float(data['cb'].dropna().iloc[-1]):,.0f} 美元"},
        {"label": "Coinbase 溢價（對幣安，原始版）",
         "value": f"{_pct(rl_.iloc[-1])}（{_d(rl_.index[-1])}）；7 日平均 {_pct(fl['c7'].iloc[-1])}；在過去一年由低到高排第 {pctile(raw)} 的位置"},
        {"label": "美元對美元版（Coinbase ÷ Bitstamp，校正 USDT）",
         "value": f"{_pct(ul.iloc[-1])}（{_d(ul.index[-1])}）；7 日平均 {_pct(flu['c7'].iloc[-1])}；過去一年排第 {pctile(usd)}"},
        {"label": "一年最低 5% 的門檻｜7 日平均的一年中位數",
         "value": f"{_pct(lo)}｜{_pct(med)}；今天在最低 5%：{'是' if bool(fl['bottom'].iloc[-1]) else '否'}"},
        {"label": "CB2 的「折價後回升」",
         "value": ("已武裝（7 日平均曾落在一年最低 5%，還沒回到中位數以上）" if armed else "沒有武裝")
                  + (f"；最近一次觸發在 {_d(trig_dates[-1])}" if trig_dates else "")},
        {"label": "50 日均線｜CB1｜CB2", "value": f"{word(T['ma50'].iloc[-1])}｜{word(T['cb1'].iloc[-1])}｜{word(T['cb2'].iloc[-1])}"},
        {"label": "以太幣 Coinbase 溢價", "value": f"{_pct(er.iloc[-1])}（{_d(er.index[-1])}）；7 日平均 {_pct(r['fle']['c7'].iloc[-1])}"},
        {"label": "這幾格的意思", "value": "只描述溢價和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    m30 = lv[MAIN_H]
    bm, cm_ = eff["bottom"][MAIN_H], eff["conf"][MAIN_H]

    def inside(m) -> str:
        ci = m.get("ci", (None, None))
        return "區間含零" if ci[0] is None or ci[0] <= 0 <= ci[1] else "區間不含零"

    if pe["level"]["pass"]:
        lvl_word = "溢價越高之後越強，三段一致" if m30["est"] > 0 else "溢價越高之後越弱，三段一致"
    elif pe["level"]["same"] == 3:
        lvl_word = "三段都同方向但很弱，全期間區間含零，看不出方向"
    else:
        lvl_word = "方向不穩定"
    bpass = pe["bottom"]["pass"]
    bword = ("反而比平常少" if _ok(bm["est"]) and bm["est"] < 0 else "比平常多") + f" {abs(bm['est']) * 100:.1f} 個百分點"
    s1 = (f"2017-08 起 Coinbase 比幣安貴或便宜，和之後 30 天比特幣漲跌的相關 {_num(m30['est'])}"
          f"（三段 {'、'.join(_num(v) for v in m30['per'])}），{lvl_word}。")
    s2 = (f"「極度折價（一年最低 5%）是底部」照原樣測 {eff['bottom']['indep']} 段，之後 30 天{bword}"
          + ("（三段一致、區間不含零，和說法相反）" if bpass and bm["est"] < 0 else f"（{inside(bm)}）")
          + f"；「溢價為正才確認漲勢」為正減非正 {_pp(cm_['est'])} 個百分點（{inside(cm_)}）。")
    s3 = (f"加在 50 日均線上沒有幫助：公平考試四項最低 均線 {_share(s['ma50']['low'])}、CB1 {_share(s['cb1']['low'])}、"
          f"CB2 {_share(s['cb2']['low'])}（隨機錯位 {_share(plc['cb1'])}、{_share(plc['cb2'])}），事先定的標準"
          + ("通過。" if passed else "沒有通過。"))
    if any(rank[k] < rank["ma50"] for k in ("cb1", "cb2")):
        s3 = s3.replace("加在 50 日均線上沒有幫助：", "加在 50 日均線上：")
    verdict = s1 + s2 + s3

    # ----- notes -----
    def etext(k) -> str:
        x = pe[k]
        return f"三段同方向 {x['same']}／3、全期間 90% 區間{'不含' if x['excl'] else '含'}零 → {'過' if x['pass'] else '不過'}"

    both = pd.concat([raw, usd], axis=1).dropna()
    bn, bs, ut = r["close"].reindex(days), data["bs"], data["usdt"]
    implied = (bn / bs).where(ut.notna())
    cmp_ = pd.concat([implied, ut], axis=1).dropna()
    adj = (data["cb"] / (bn * ut) - 1)
    cmp2 = pd.concat([adj, usd], axis=1).dropna()
    eps = eff["bottom"]["episodes"]
    eps7 = eff["bottom7"]["episodes"]
    notes = [
        "事先寫下的設計（看結果之前，之後沒有改）：Premium C_t (7-day average and daily), at the same 00:00 UTC timestamp for both "
        "exchanges (hourly if needed; check timestamp alignment as kimchi_premium did). 1. Level: forward 7/30/90-day BTC returns "
        "by C quintile; split 2017-08..2019, 2020..2022, 2023..now (ETF era from 2024 as description); block bootstrap. 2. Bottom "
        "claim: C in its trailing-1-year bottom 5% → forward 30/90-day returns vs all days; independent episodes. 3. Confirmation "
        "claim: C > 0 (7-day avg) vs ≤ 0 → forward 30-day returns, and as an entry confirmation: rule CB1 = ma50 AND C7 > 0. "
        "4. Rule CB2 (bottom entry): when below ma50, enter early if C crosses from its bottom 5% back above its trailing-year "
        "median; exit on close < 20-day average or when ma50 takes over (as rally_timing does). Evaluate CB1, CB2 vs ma50 on the "
        "fair grid (data from 2017-08 — as many starts as allowed), ordinary-person execution, ETH (Coinbase ETH-USD vs Binance "
        "ETHUSDT), placebo with circularly shifted premium (200 draws), crash-avoided share with whipsaws, re-entry days after "
        "bottoms. Pass test (stated now): an effect with the same sign in all three periods and full-sample 90% interval "
        "excluding zero, AND a rule beats ma50 on the ranking score with placebo share ≤ 10%, AND same direction on ETH.",
        "上面這段的白話和事先補定的細節：主要看 30 天。溢價高低用相關係數；「最低 5%」用訊號日之後 30 天平均 − 全部日子平均；「為正」用為正的日子 − "
        "非正的日子。三段方向都要和全期間一樣、全期間 90% 區間不含零，而且 CB1 或 CB2 在公平考試的排名要排在 50 日均線前面、隨機錯位一樣好的比例 ≤ 10%，"
        "以太幣上效果同方向、規則的報酬÷波動也贏以太幣自己的 50 日均線，才算通過。CB1、CB2 都用 7 日平均（避開每天的雜訊）；"
        "CB2 在 7 日平均落到一年最低 5% 時「武裝」，之後第一次回到一年中位數以上就是訊號，均線現金時才會提早上車，跌破 20 日均線就下車，均線接手後照均線。",
        f"過關檢查明細｜溢價高低：{etext('level')}；最低 5%：{etext('bottom')}"
        + ("（但方向是之後比較差，和「底部」的說法相反）" if pe["bottom"]["pass"] and pe["bottom"]["sign"] < 0 else "")
        + f"；7 日平均為正：{etext('conf')}；"
        f"以太幣同方向（只看過關的效果）：{'有' if any(eth_same.values()) else '沒有'}｜CB1 排名{'在' if beats['cb1'] else '不在'}均線前、"
        f"隨機錯位 {_share(plc['cb1'])}；CB2 排名{'在' if beats['cb2'] else '不在'}均線前、隨機錯位 {_share(plc['cb2'])}；"
        f"以太幣報酬÷波動 均線 {_num(r['ratio_eth']['ma50'])}、CB1 {_num(r['ratio_eth']['cb1'])}、CB2 {_num(r['ratio_eth']['cb2'])}。"
        f"整體：{'通過' if passed else '沒有通過'}。",
        "「一年最低 5%」（每日）各段的第一天：" + ("、".join(_d(x) for x in eps) if eps else "沒有") + "；7 日平均版："
        + ("、".join(_d(x) for x in eps7) if eps7 else "沒有") + "。CoinDesk 舉的 2022-11、2023-08 是事後挑出來的兩次；這裡把每一次都算進去。",
        "USDT 的問題：原始版拿美元比泰達幣（USDT），泰達幣自己偏離 1 美元時溢價會跟著動（例如 2018-10 泰達幣跌到約 0.97 美元，"
        "幣安的泰達幣報價變高，溢價就會變成負的幾個百分點，那正是「一年最低 5%」最早的兩段）。"
        "免費的 USDT／美元日資料到不了 2017：Coinbase USDT-USD 2021-05 才有，Kraken 公開 API 只給最近 720 天，Bitstamp 沒有 USDT 美元的歷史。"
        f"所以校正版改用 Bitstamp 的比特幣美元價當國際價格（等於用 Bitstamp 推出來的泰達幣匯率校正）。兩版每天的相關 {_num(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 3)}、"
        f"差距中位數 {float((both.iloc[:, 0] - both.iloc[:, 1]).median() * 100):+.2f} 個百分點。"
        + (f" 2021-05 起用 Coinbase 自己的 USDT-USD 收盤核對：Bitstamp 推出來的泰達幣價和它的差距中位數 "
           f"{float((cmp_.iloc[:, 0] - cmp_.iloc[:, 1]).abs().median() * 100):.2f} 個百分點，兩種校正後溢價的相關 "
           f"{_num(float(cmp2.iloc[:, 0].corr(cmp2.iloc[:, 1])), 3)}（{len(cmp2):,} 天）。" if len(cmp2) > 100 else "")
        + " 兩種校正彼此相關不高，代表每天收盤價差只有零點幾個百分點時，Bitstamp 自己的雜訊就很大，美元對美元版只能當粗略的核對。"
        "Bitstamp 也不是美國以外的交易所（在盧森堡，美國人也能用），所以美元對美元版比較像「Coinbase 比另一家美元交易所貴多少」。",
        _usd_note(eff, r, rank),
        f"資料：Coinbase 比特幣日線（kimchi_premium 的存檔）{FIRST} 起缺 {int(data['cb'].isna().sum())} 天；Bitstamp 缺 {int(data['bs'].isna().sum())} 天；"
        f"以太幣 Coinbase 缺 {int(data['eth_cb'].isna().sum())} 天。缺的那天溢價不算，7 日平均要 7 天裡至少 5 天有值。幣安 BTCUSDT 從 2017-08-17 才有，"
        "所以全部檢查從那天開始；「一年最低 5%」和 CB2 要先累積一年，2018-07 以後才有值（之前 CB2 和 50 日均線一模一樣）。",
        "公平考試只能從 2017-09 起算（main_signal_check 從 2015 起），起算月和 2018 年初的大跌高度重疊，相鄰起算日共用大部分歷史、24 個收盤時間也只差幾小時，"
        "這些「幾個裡幾個」不是互相獨立的證據。其他資產只有以太幣（只有它有 Coinbase 和幣安兩邊從 2017 起的日線），比 main_signal_check 的四個幣加 MSTR 少。",
        "CoinDesk 和 TradingView 沒有公開精確的門檻和平均方式（CryptoQuant 的指數用的是分鐘級價差的平均），這裡用日收盤、門檻用自己過去一年的 5%，"
        "數字可能和他們的圖不同。分五組用的是全期間分界，事後才知道。",
    ]
    notes += _lookahead(data, r)
    notes += [
        f"成本很粗略：照課本做每次買賣扣 {COST * 100:.2f}%、一般人做法 {P_COST * 100:.2f}%；現金沒有利息；一直抱著沒有扣成本；沒有算稅。"
        "隨機的部分都用固定的亂數種子，資料沒變的話重跑結果一樣。",
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把流行的說法照原樣測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "coinbase_premium",
        "title": "Coinbase 溢價能看出比特幣方向嗎",
        "question": ("美國 Coinbase 的比特幣比國際交易所貴或便宜（常被當成美國買盤強弱），能不能看出之後的方向？"
                     "「極度折價是底部」「溢價為正才確認漲勢」照原樣測結果如何？加在 50 日均線上有沒有幫助？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
