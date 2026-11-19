"""Historical study "rally_sources": when bitcoin rose a long way, where did the money come from - and did any
money-flow channel turn up at the START of the big rallies BEFORE the main signal ma50 (hold while the daily close is
above its 50-day average) turned on, reliably and without many false alarms?

Why: ma50 and every filter tried on top of it so far can only AVOID losses; nothing found gets in earlier than ma50 or
catches the upside it misses. rally_timing tried PRICE-ONLY early entries (and capitulation markers) after 50%+ bear
markets; crash_timing tried faster exits/re-entries on averages. This study is different: (a) rallies are defined by
the RISE (+80%), not by a preceding crash, (b) the candidates are money-flow channels (who is putting money in, or
buying more eagerly), not price patterns, (c) one channel is picked on rallies before 2021 and tested only on later
ones.

Pre-registered design (written before any result was seen; not changed afterwards):
1. Rallies, mechanical, on the dashboard's daily closes (te._daily_close, 00:00 UTC). The scan starts at
   2012-11-01; only rallies whose trough is on or after 2013-01-01 are listed. State: a candidate trough T (lowest
   close since the previous rally's peak) and a running high H since T. A new lower close resets T and H. When
   H / T - 1 >= 80% the rally qualifies; it then runs until a close <= 70% of its running high (a 30% fall), and its
   peak P is that running high; the next candidate trough starts at the close that ended it. If, before qualifying,
   a close falls 30% below H (a failed attempt), T is kept (it is still the lowest close since the last peak) and H
   restarts from that close. A qualified rally still running today is listed as unfinished (peak = high so far).
   ma50 date M = first close on or after T with ma50 on. Reported: T, M, P dates, gain T->M (what ma50 misses),
   gain T->P, duration. The 80% / 30% numbers are not tuned; one robustness line with 50% / 25% (count only).
2. Channels, each a daily "turned up" flag at each BTC close, same recipe: the channel's 14-day change is above the
   80th percentile of its own 14-day changes over the previous 365 closes (at least 180 of them). Every value is
   the one known at that close (publication lag below); "first usable" = first close with a flag value.
     stable   stablecoin supply (DefiLlama total, stablecoin_supply's loader and cleaning, from 2019-07-01):
              14-day change = sum of cleaned daily log growth. Lag 1 day.
     exbal    exchange BTC balance FALLING (Coin Metrics community SplyExNtv via flow_filters._fetch_cm):
              change = -(log balance today - log balance 14 days ago). Lag 1 day.
     netout   net exchange outflow (FlowOutExNtv - FlowInExNtv, same loader): 14-day sum. Lag 1 day.
     cbp      Coinbase premium (Coinbase BTC-USD / Binance BTCUSDT - 1, kimchi_premium's loader): 7-day average
              (at least 5 days) > 0 today while it was <= 0 on at least one of the previous 14 closes. Lag 0 (both
              candles close at the same 00:00 UTC instant).
     kimchi   Korean premium rising (Upbit KRW / (Binance x Fed H.10 USD/KRW) - 1, kimchi_premium's loader): 7-day
              average, 14-day change in points. Lag 0 (noon New York FX, of that day or earlier).
     taker    taker-buy share of Binance spot BTCUSDT volume (order_flow's loader): 14-day share, its 14-day change.
              Lag 0.
     volume   Binance spot BTCUSDT quote volume (USDT), 14-day sum, log 14-day change. Lag 0.
     oi       Binance USDT-M futures open interest in coins (data.binance.vision metrics via ls_ratio / oi_flush):
              log 14-day change; the row stamped D 00:00 is used from the close labelled D (one day late, as
              oi_flush). Gaps carried at most 3 days.
     funding  Binance BTCUSDT funding: daily mean of settlements, 7-day average (at least 5 days) > 0 today while it
              was < 0 on at least one of the previous 14 closes ("turning from negative to positive"). Lag 0.
     etf      US spot bitcoin ETF net flow (Farside total via etf_loop's loader, from 2024-01-11): 14-day sum over
              calendar days (no-trading days = 0). Flows dated t used from the close labelled t+1.
     adr      active addresses (Coin Metrics AdrActCnt via network_growth's loader): 7-day mean, log 14-day change.
              Lag 1 day.
     wiki     English Wikipedia "Bitcoin" page views (attention's loader, from 2015-07): 7-day mean, log 14-day
              change. Lag 1 day.
   A channel whose loader fails is dropped and named, never replaced by another source.
3. Anatomy. For each rally and channel: the first close in [T - 30 days, M] with the flag on, as days before (+) M;
   if none, the first in (M, P] as a negative number; else "never before the peak". A channel is "available" for a
   rally if its flag is known at T - 30 days. "Led" = turned up strictly before M. Also the channel's change from T
   to M and from M to P (its own units). Per channel: rallies led / available, median lead, rallies where it never
   turned up before the peak, and a chance baseline: the same "led" count with the channel's flag series circularly
   shifted by 180 ... (n - 180) days, 200 shifts (median, and the share of shifts leading at least as many rallies).
   "Sources of the rise" per rally (data-day values, description only): stablecoin supply (USD), exchange balance
   (BTC), ETF cumulative net flow (USD), open interest (BTC and USD at the close) at T and at P.
   Exploratory, report-only, NOT part of any test (added at the coordinator's request before the run): per rally,
   the dollar inflow through direct-inflow channels with data (stablecoin supply change in USD; ETF net flow in
   USD from 2024; realised cap change = Coin Metrics CapMrktCurUSD / CapMVRVCur via collectors.onchain2, the
   holders' total cost basis), the change in bitcoin market value (Coin Metrics CapMrktCurUSD = price x circulating
   supply), and "market value gained per dollar of measured inflow" (stablecoin + ETF; and separately per dollar of
   realised cap), for T->P, T->M and M->P. Missing data -> empty cell, never estimated.
4. False alarms, per channel, over its whole span: every close where the flag switches on (off yesterday, on today)
   while ma50 is off. "Followed by a rally" = a qualifying rally trough within [t, t + 60 days], OR ma50 turns on
   within the next 60 closes and stays on at least 30 consecutive closes. Share followed; false-alarm share = the
   rest. Mean and median BTC return over the next 30 / 60 days versus all ma50-off closes of the same span; random
   comparison: 1,000 random samples of the same size from those ma50-off closes, share with a mean at least as high.
5. Honest split. DISCOVERY = rallies with T < 2021-01-01; TEST = T >= 2021-01-01. On DISCOVERY only the code picks
   the channel with the highest led / available among channels available in at least 3 discovery rallies (if none
   has 3, at least 1); ties -> lower false-alarm share counted on switch-ons before 2021-01-01; then the order of
   the list above. If the picked channel has no flag values from 2021-01-01 on, the best runner-up is used and
   marked. Rule E1 = hold when ma50 is on OR (the picked channel's flag switched on at one of the last 30 closes,
   today included, AND the close is above its 20-day average). Scored on TEST only: test rallies led (E1 holding at
   some close in [T, M)), gain E1 captured from T to M (its own net growth with 0.10% costs) vs T->M, false entries
   (E1 holding while ma50 is off, entered on or after 2021-01-01, ending without ma50 taking over) and their net
   result. Fair exam (main_signal_check's pieces, as stablecoin_supply): monthly starts 2021-01 ... 2023-12 on the
   UTC daily closes (textbook 0.10%); the same months x 24 closing hours (textbook; ordinary person: 10 hours later,
   0.25%, 10% of entries missed, 200 simulations per clock, ma50 with main_signal_check's seeds); ETH, SOL, BNB, XRP
   with their own 50/20-day averages and bitcoin's channel flag, starts 2021-01 ... three years before today.
   Ranking score = lowest of the four "beats holding" shares; ma50 re-scored on the same starts. On the clock grid a
   close on UTC date E uses the daily flag decided at the close that ended E-1. Median deepest loss = median over
   the start x clock textbook cells. Placebo: the picked channel's flag series circularly shifted by 180 ...
   (n - 180) days, 200 draws, E1 rebuilt, scored without the ordinary-person check (the real E1 scored the same
   way); share of draws with a higher score, or the same score and a median yearly edge at least as large.
   PASS = E1 ranking score >= ma50's AND E1 median deepest loss no more than 5 points deeper AND placebo <= 10%.
6. Caveat stated in the notes: some channels ARE direct net inflows (stablecoin net issuance, US spot ETF net
   creations, realised cap); others (Coinbase premium, taker-buy share, funding, open interest, attention) measure
   buying eagerness or leverage, not money in. Price is set by marginal trades, so the rise in market value is
   usually several times the net inflow and the multiple varies; in one trade the buyer's money goes to the seller,
   so stock measures matter, not volume; and an inflow measured during a rise may be following price.
Today: "比特幣：50 日均線｜E1" and one row per channel (flag on now, last switch-on).
Only public key-less endpoints through the project's own loaders (fixed user agent). Description, not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import onchain2
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import etf_loop as el
from dashboard.studies2 import flow_filters as ff
from dashboard.studies2 import funding_positioning as fp
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import network_growth as ng
from dashboard.studies2 import oi_flush as oif
from dashboard.studies2 import order_flow as of
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import stablecoin_supply as ss
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
DAY = pd.Timedelta(days=1)
SCAN_FROM, LIST_FROM = "2012-11-01", "2013-01-01"
RISE, FALL = 0.80, 0.30
RISE_ALT, FALL_ALT = 0.50, 0.25
CH_DAYS, WIN, WIN_MIN, TOP_Q = 14, 365, 180, 0.80
CROSS_LOOK = 14
PRE = 30  # days before the trough in the anatomy window
FA_DAYS, STAY = 60, 30
RECENT = 30
SPLIT = "2021-01-01"
MIN_AVAIL = 3
TEST_FIRST, TEST_LAST = "2021-01-01", msc.DAILY_STARTS[1]
DRAWS, GAP, RAND = 200, 180, 1000
LOSS_GAP = 5.0
KI = {"ma50": 1, "e1": 95}
OTHER = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))

CHANNELS = ("stable", "exbal", "netout", "cbp", "kimchi", "taker", "volume", "oi", "funding", "etf", "adr", "wiki")
CN = {"stable": "穩定幣供給增加", "exbal": "交易所比特幣餘額減少", "netout": "交易所淨流出",
      "cbp": "Coinbase 溢價轉正", "kimchi": "韓國溢價上升", "taker": "主動買進比例上升", "volume": "現貨成交量放大",
      "oi": "合約未平倉量增加", "funding": "資金費率由負轉正", "etf": "美國現貨 ETF 淨流入", "adr": "活躍地址增加",
      "wiki": "維基百科瀏覽量增加"}
EXPLAIN = {
    "stable": "新的美元換成穩定幣（USDT、USDC 等）的總量，DefiLlama，2019-07 起，晚 1 天知道",
    "exbal": "放在交易所的比特幣變少，表示幣被提走、多半是買來放著，Coin Metrics，晚 1 天",
    "netout": "流出交易所的幣比流入多，14 天加總，Coin Metrics，晚 1 天",
    "cbp": "美國 Coinbase 的價格比幣安貴（7 天平均由負轉正），表示美國買家比較急，收盤當下就知道",
    "kimchi": "韓國 Upbit 比國際價貴的幅度變大，表示韓國散戶追價，收盤當下就知道",
    "taker": "幣安現貨成交裡，買方主動吃單的比例變高，收盤當下就知道",
    "volume": "幣安現貨 14 天成交金額變大，收盤當下就知道",
    "oi": "幣安合約還沒平掉的部位（顆數）變多，表示槓桿資金進來，晚 1 天",
    "funding": "合約做多的人從收錢變成付錢給做空的人（7 天平均由負轉正），收盤當下就知道",
    "etf": "新錢透過美國現貨 ETF 買比特幣，2024-01 起，晚 1 天",
    "adr": "每天有動用過的比特幣地址數變多，Coin Metrics，晚 1 天",
    "wiki": "英文維基「Bitcoin」頁的瀏覽人數變多，代表關注度，晚 1 天",
}
LAG_CN = {"stable": "1 天", "exbal": "1 天", "netout": "1 天", "cbp": "0 天", "kimchi": "0 天", "taker": "0 天",
          "volume": "0 天", "oi": "1 天", "funding": "0 天", "etf": "1 天", "adr": "1 天", "wiki": "1 天"}

_d = cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v) -> str:
    return f"{float(v) * 100:+.0f}%" if _ok(v) else "—"


def _share(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _num(v, d: int = 1) -> str:
    return f"{float(v):,.{d}f}" if _ok(v) else "—"


def _bn(v) -> str:
    return f"{float(v) / 1e8:+,.0f} 億美元" if _ok(v) else ""


def _x(v) -> str:
    return f"{float(v):.1f} 倍" if _ok(v) else ""


def _at(series: pd.Series, dates, lag: int) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


# ---------- data ----------

def _top(ch: pd.Series) -> pd.Series:
    q = ch.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(TOP_Q)
    return (ch > q).astype(float).where(q.notna() & ch.notna())


def _cross(x: pd.Series, strict_neg: bool) -> pd.Series:
    """x > 0 today while it was <= 0 (or < 0) on one of the previous CROSS_LOOK closes."""
    prev_min = x.shift(1).rolling(CROSS_LOOK, min_periods=CROSS_LOOK).min()
    was = (prev_min < 0) if strict_neg else (prev_min <= 0)
    return ((x > 0) & was).astype(float).where(x.notna() & prev_min.notna())


def _on_close(s: pd.Series, idx: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Daily data series -> values known at each close label (data day = label - lag)."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.Series(_at(s, idx, lag), index=idx)


def _channels(close: pd.Series) -> tuple[dict, dict, list[str], dict]:
    """Per channel: {'lvl', 'ch', 'flag'} on the close index (values known at each close); raw data-day levels for
    the description table; problems; dropped channels with the reason."""
    idx = close.index
    out, raw, problems, dropped = {}, {}, [], {}

    def drop(k, exc):
        dropped[k] = f"{type(exc).__name__}: {str(exc)[:80]}"

    # stablecoin supply
    try:
        sup, pr = ss._load()
        problems += pr
        g, _ = ss._growth(sup)
        lvl = _on_close(sup["all"], idx, 1)
        ch = _on_close(g["all"], idx, 1).rolling(CH_DAYS, min_periods=CH_DAYS).sum()
        lvl[idx < pd.Timestamp(ss.START) + DAY] = np.nan
        out["stable"] = {"lvl": lvl, "ch": ch, "flag": _top(ch)}
        raw["stable"] = sup["all"][sup.index >= pd.Timestamp(ss.START)]
    except Exception as exc:  # noqa: BLE001
        drop("stable", exc)
    # exchange balance and net flow
    try:
        cm = ff._fetch_cm()
        bal = cm["balance"].where(cm["balance"] > 0).asfreq("D").ffill(limit=3)
        lb = _on_close(np.log(bal), idx, 1)
        ch = -(lb - lb.shift(CH_DAYS))
        out["exbal"] = {"lvl": _on_close(bal, idx, 1), "ch": ch, "flag": _top(ch)}
        raw["exbal"] = bal
        net = (cm["outflow"] - cm["inflow"]).asfreq("D")
        nk = _on_close(net, idx, 1)
        ch = nk.rolling(CH_DAYS, min_periods=CH_DAYS).sum()
        out["netout"] = {"lvl": nk.fillna(0.0).cumsum().where(nk.notna().cummax()), "ch": ch, "flag": _top(ch)}
    except Exception as exc:  # noqa: BLE001
        drop("exbal", exc)
        drop("netout", exc)
    # Coinbase and Korean premiums
    try:
        kd = km._load()
        problems += kd["problems"]
        pm = km._premiums(kd)
        c7 = _on_close(pm["cbp"].rolling(7, min_periods=5).mean(), idx, 0)
        out["cbp"] = {"lvl": c7, "ch": c7 - c7.shift(CH_DAYS), "flag": _cross(c7, strict_neg=False)}
        k7 = _on_close(pm["kp"].rolling(7, min_periods=5).mean(), idx, 0)
        ch = k7 - k7.shift(CH_DAYS)
        out["kimchi"] = {"lvl": k7, "ch": ch, "flag": _top(ch)}
    except Exception as exc:  # noqa: BLE001
        drop("cbp", exc)
        drop("kimchi", exc)
    # taker share and volume
    try:
        bars, pr = of._daily("spot", "BTCUSDT")
        if pr:
            problems.append(pr)
        vol = bars["volume"].where(bars["volume"] > 0)
        sh = bars["tb"].rolling(CH_DAYS, min_periods=CH_DAYS).sum() / vol.rolling(CH_DAYS, min_periods=CH_DAYS).sum()
        shk = _on_close(sh, idx, 0)
        ch = shk - shk.shift(CH_DAYS)
        out["taker"] = {"lvl": shk, "ch": ch, "flag": _top(ch)}
        q14 = _on_close(bars["quote"].where(bars["quote"] > 0).rolling(CH_DAYS, min_periods=CH_DAYS).sum(), idx, 0)
        ch = np.log(q14 / q14.shift(CH_DAYS))
        out["volume"] = {"lvl": q14, "ch": ch, "flag": _top(ch)}
    except Exception as exc:  # noqa: BLE001
        drop("taker", exc)
        drop("volume", exc)
    # open interest
    try:
        rows, pr = lr._fetch("BTCUSDT", budget_s=300)
        if pr:
            problems.append(pr)
        oi = oif._oi(rows, idx[-1]).ffill(limit=3)
        ok = _on_close(oi, idx, 0)  # stamp D used from the close labelled D
        ch = np.log(ok / ok.shift(CH_DAYS))
        out["oi"] = {"lvl": ok, "ch": ch, "flag": _top(ch)}
        raw["oi"] = oi
    except Exception as exc:  # noqa: BLE001
        drop("oi", exc)
    # funding
    try:
        fr = fp._funding_history()
        day = fr.groupby(fr.index.normalize()).mean().asfreq("D")
        f7 = _on_close(day.rolling(7, min_periods=5).mean(), idx, 0)
        out["funding"] = {"lvl": f7, "ch": f7 - f7.shift(CH_DAYS), "flag": _cross(f7, strict_neg=True)}
    except Exception as exc:  # noqa: BLE001
        drop("funding", exc)
    # ETF
    try:
        fr_, pr = el._farside("BTC")
        if pr:
            problems.append(pr)
        tot = fr_["Total"].astype(float) * 1e6
        tot = tot[~tot.index.duplicated(keep="last")].sort_index()
        days = pd.date_range(tot.index[0], idx[-1], freq="D")
        flow = tot.reindex(days).fillna(0.0)
        fk = _on_close(flow, idx, 1)
        ch = fk.rolling(CH_DAYS, min_periods=CH_DAYS).sum()
        out["etf"] = {"lvl": fk.fillna(0.0).cumsum().where(fk.notna().cummax()), "ch": ch, "flag": _top(ch)}
        raw["etf"] = flow.cumsum()
    except Exception as exc:  # noqa: BLE001
        drop("etf", exc)
    # active addresses
    try:
        oc, pr = ng._onchain("btc")
        problems += pr
        a7 = oc["AdrActCnt"].where(oc["AdrActCnt"] > 0).rolling(7, min_periods=7).mean()
        ak = _on_close(a7, idx, 1)
        ch = np.log(ak / ak.shift(CH_DAYS))
        out["adr"] = {"lvl": ak, "ch": ch, "flag": _top(ch)}
    except Exception as exc:  # noqa: BLE001
        drop("adr", exc)
    # Wikipedia
    try:
        v, pr = at._views("btc")
        if pr:
            problems.append(pr)
        w7 = v.where(v > 0).rolling(7, min_periods=7).mean()
        wk = _on_close(w7, idx, 1)
        ch = np.log(wk / wk.shift(CH_DAYS))
        out["wiki"] = {"lvl": wk, "ch": ch, "flag": _top(ch)}
    except Exception as exc:  # noqa: BLE001
        drop("wiki", exc)
    # market value and realised cap (exploratory table only)
    try:
        f = onchain2._cm("BTC")
        f.index = pd.to_datetime(f.index)
        raw["mcap"] = f["CapMrktCurUSD"].where(f["CapMrktCurUSD"] > 0)
        raw["rcap"] = (f["CapMrktCurUSD"] / f["CapMVRVCur"]).where(f["CapMVRVCur"] > 0)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Coin Metrics 市值／實現市值這次取不到（{type(exc).__name__}），探索表的市值欄留空。")
    return out, raw, problems, dropped


# ---------- rallies ----------

def _rallies(close: pd.Series, rise: float, fall: float) -> list[dict]:
    c = close[close.index >= pd.Timestamp(SCAN_FROM)]
    v, ix = c.values.astype(float), c.index
    out, lo, hi, armed = [], 0, 0, False
    for i in range(1, len(v)):
        x = v[i]
        if not armed:
            if x < v[lo]:
                lo = hi = i
            elif x > v[hi]:
                hi = i
            if v[hi] / v[lo] - 1 >= rise:
                armed = True
            elif x <= v[hi] * (1 - fall):
                hi = i  # failed attempt: keep the trough, restart the running high
        else:
            if x > v[hi]:
                hi = i
            elif x <= v[hi] * (1 - fall):
                out.append({"t": ix[lo], "p": ix[hi], "done": True})
                armed, lo, hi = False, i, i
    if armed:
        out.append({"t": ix[lo], "p": ix[hi], "done": False})
    return [r for r in out if r["t"] >= pd.Timestamp(LIST_FROM)]


def _ma50(close: pd.Series) -> np.ndarray:
    c = close.values.astype(float)
    return ct._gt(c, ct._mean(c, 50))


def _describe(rallies: list[dict], close: pd.Series, ma: np.ndarray) -> None:
    idx = close.index
    for r in rallies:
        it, ip = idx.get_loc(r["t"]), idx.get_loc(r["p"])
        on = np.flatnonzero(np.nan_to_num(ma[it:], nan=0.0) > 0.5)
        im = it + int(on[0]) if len(on) else ip
        r.update({"m": idx[im], "it": it, "im": im, "ip": ip, "pt": float(close.iloc[it]),
                  "pm": float(close.iloc[im]), "pp": float(close.iloc[ip])})
        r["miss"] = r["pm"] / r["pt"] - 1
        r["gain"] = r["pp"] / r["pt"] - 1
        r["days"] = int((r["p"] - r["t"]).days)
        r["lead_days"] = int((r["m"] - r["t"]).days)


# ---------- anatomy ----------

def _first_on(flag: np.ndarray, a: int, b: int) -> int | None:
    seg = np.nan_to_num(flag[a:b + 1], nan=0.0) > 0.5
    return a + int(np.argmax(seg)) if seg.any() else None


def _lead(flag: np.ndarray, r: dict, idx: pd.DatetimeIndex) -> tuple[bool, float | None]:
    """(available, lead days; None = never before the peak)."""
    w0 = int(idx.searchsorted(r["t"] - PRE * DAY))
    if not np.isfinite(flag[w0]):
        return False, None
    j = _first_on(flag, w0, r["im"])
    if j is None:
        j = _first_on(flag, r["im"] + 1, r["ip"]) if r["ip"] > r["im"] else None
    return True, (None if j is None else float((r["m"] - idx[j]).days))


def _led_count(flag: np.ndarray, rallies: list[dict], idx) -> tuple[int, int]:
    n = led = 0
    for r in rallies:
        av, ld = _lead(flag, r, idx)
        if av:
            n += 1
            led += int(ld is not None and ld > 0)
    return led, n


def _shifted(flag: np.ndarray, k: int) -> np.ndarray:
    out = flag.copy()
    known = np.flatnonzero(np.isfinite(flag))
    if len(known):
        a, b = known[0], known[-1] + 1
        out[a:b] = np.roll(flag[a:b], k)
    return out


def _span(flag: np.ndarray) -> int:
    known = np.flatnonzero(np.isfinite(flag))
    return int(known[-1] - known[0] + 1) if len(known) else 0


def _chg(k: str, lvl: np.ndarray, a: int, b: int) -> str:
    x, y = lvl[a], lvl[b]
    if not (_ok(x) and _ok(y)):
        return "—"
    if k in ("stable", "exbal", "volume", "oi", "adr", "wiki"):
        return _p(y / x - 1) if x > 0 else "—"
    if k == "netout":
        return f"{y - x:+,.0f} 顆"
    if k == "etf":
        return f"{(y - x) / 1e8:+,.0f} 億美元"
    if k in ("cbp", "kimchi", "taker"):
        return f"{(y - x) * 100:+.2f} 點"
    return f"{(y - x):+.4f} 點"  # funding, % per 8 hours


# ---------- false alarms ----------

def _switch_on(flag: np.ndarray) -> np.ndarray:
    f = np.nan_to_num(flag, nan=-1.0)
    prev = np.concatenate([[-1.0], f[:-1]])
    return (f > 0.5) & (prev == 0.0)


def _false_alarms(flag: np.ndarray, ma: np.ndarray, close: pd.Series, troughs: list, rng,
                  before: pd.Timestamp | None = None) -> dict:
    idx = close.index
    c = close.values.astype(float)
    n = len(c)
    m = np.nan_to_num(ma, nan=0.0) > 0.5
    son = _switch_on(flag) & ~m & ~np.isnan(ma)
    if before is not None:
        son &= idx < before
    tpos = np.array([idx.get_loc(t) for t in troughs], dtype=int)
    hits = []
    ev = np.flatnonzero(son)
    for i in ev:
        hit = bool(((tpos >= i) & (tpos <= i + FA_DAYS)).any())
        if not hit:
            for j in range(i + 1, min(n, i + FA_DAYS + 1)):
                if m[j] and not m[j - 1]:
                    run = 0
                    while j + run < n and m[j + run]:
                        run += 1
                    if run >= STAY:
                        hit = True
                        break
        hits.append(hit)
    out = {"n": len(ev), "hit": int(sum(hits))}
    out["fa"] = (1 - out["hit"] / out["n"]) if out["n"] else None
    known = np.isfinite(flag)
    base_mask = known & ~m & ~np.isnan(ma)
    if before is not None:
        base_mask &= idx < before
    for h in (30, 60):
        fwd = np.full(n, np.nan)
        fwd[:-h] = c[h:] / c[:-h] - 1
        e = ev[np.isfinite(fwd[ev])]
        b = np.flatnonzero(base_mask & np.isfinite(fwd))
        out[f"mean{h}"] = float(fwd[e].mean()) if len(e) else None
        out[f"med{h}"] = float(np.median(fwd[e])) if len(e) else None
        out[f"bmean{h}"] = float(fwd[b].mean()) if len(b) else None
        out[f"bmed{h}"] = float(np.median(fwd[b])) if len(b) else None
        if len(e) and len(b) > len(e):
            draws = np.array([fwd[rng.choice(b, len(e), replace=False)].mean() for _ in range(RAND)])
            out[f"rand{h}"] = float((draws >= out[f"mean{h}"]).mean())
        else:
            out[f"rand{h}"] = None
    return out


# ---------- E1 and the exam ----------

def _recent(flag: np.ndarray) -> np.ndarray:
    s = pd.Series(_switch_on(flag).astype(float))
    return s.rolling(RECENT, min_periods=1).max().values


def _e1(ma: np.ndarray, up20: np.ndarray, rec: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(ma), np.nan, np.maximum(np.nan_to_num(ma), np.nan_to_num(up20) * rec))


def _grid_daily(close: pd.Series, pos: dict, first, last) -> dict:
    starts, labels = msc._month_starts(close.index, first, last)
    change = close.pct_change().fillna(0.0).values
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out = {"starts": labels, "hold": (hr[:, 0], hd[:, 0])}
    for k, w in pos.items():
        net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    """main_signal_check's 24 closing hours, with ma50 and the 20-day average per hour."""

    def __init__(self, first, last):
        rows, self.problem = cr._hourly("BTCUSDT")
        if len(rows) < cr.MIN_HOURS:
            raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
        price, b0, _ = cr._grid(rows)
        pre = cr._prehistory(b0)
        if not len(pre):
            raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, first, last)
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma, self.up = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.ma.append(np.nan_to_num(ct._gt(c, ct._mean(c, 50))[npre:][:self.days], nan=0.0))
            self.up.append(np.nan_to_num(ct._gt(c, ct._mean(c, 20))[npre:][:self.days], nan=0.0))

    def rec(self, idx: pd.DatetimeIndex, rec: np.ndarray) -> np.ndarray:
        return np.nan_to_num(_at(pd.Series(rec, index=idx), self.m.dates, 1), nan=0.0)

    def run(self, k: str, rec: np.ndarray, person: bool) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, SIMS)) if person else None
        for h in range(24):
            w = self.ma[h] if k == "ma50" else np.maximum(self.ma[h], self.up[h] * rec)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            if person:
                b, wl = rt._later(h, w, self.days)
                rng = np.random.default_rng(SEED + 1000 * KI[k] + h)
                pos = rt._missed(wl, P_MISS, rng, SIMS)
                r, _ = msc._suffix(np.vstack([np.ones((1, SIMS)), self.eng._mid(b, pos, P_COST, None)]),
                                   self.starts, YEAR)
                pr[:, h, :] = r
        edge = tr - self.hr[:, None]
        out = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
               "c_deep_med": float(np.median(td))}
        if person:
            pe = pr - self.hr[:, None, None]
            out.update({"p_n": pe.size, "p_beat": int((pe > 0).sum()), "p_edge": float(np.median(pe))})
        return out


def _others_last() -> pd.Timestamp:
    return pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)


def _coin_pos(c: pd.Series, btc_idx: pd.DatetimeIndex, rec: np.ndarray) -> dict:
    v = c.values.astype(float)
    ma = ct._gt(v, ct._mean(v, 50))
    up = ct._gt(v, ct._mean(v, 20))
    r = np.nan_to_num(_at(pd.Series(rec, index=btc_idx), c.index, 0), nan=0.0)
    return {"ma50": ma, "e1": _e1(ma, up, r)}


def _exam(close, ma, up20, rec, clock, coins, keys, person) -> dict:
    pos = {"ma50": ma, "e1": _e1(ma, up20, rec)}
    d = _grid_daily(close, {k: pos[k] for k in keys}, TEST_FIRST, TEST_LAST)
    crec = clock.rec(close.index, rec)
    og = {}
    for a, c in coins.items():
        cp = _coin_pos(c, close.index, rec)
        og[a] = _grid_daily(c, {k: cp[k] for k in keys}, TEST_FIRST, _others_last())
    out = {"n_daily": len(d["starts"]), "n_clock": len(clock.starts)}
    for k in keys:
        x = clock.run(k, crec, person)
        dr, _dd = d[k]
        x["d_n"], x["d_beat"] = len(dr), int((dr > d["hold"][0]).sum())
        o_n = sum(len(g[k][0]) for g in og.values())
        o_b = sum(int((g[k][0] > g["hold"][0]).sum()) for g in og.values())
        x["o_n"], x["o_beat"] = o_n, o_b
        checks = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"]]
        if person:
            checks.append(x["p_beat"] / x["p_n"])
        checks.append(o_b / o_n if o_n else 0.0)
        x["checks"], x["low"] = checks, min(checks)
        out[k] = x
    return out


def _episodes(e1: np.ndarray, ma: np.ndarray, net: np.ndarray, idx: pd.DatetimeIndex, since) -> list[dict]:
    """Early holdings: E1 on while ma50 off; each ends when E1 goes to cash (false) or ma50 takes over."""
    e = np.nan_to_num(e1, nan=0.0) > 0.5
    m = np.nan_to_num(ma, nan=0.0) > 0.5
    out, i, n = [], 0, len(e)
    while i < n:
        if e[i] and not m[i] and (i == 0 or not e[i - 1] or m[i - 1]) and idx[i] >= pd.Timestamp(since):
            j = i
            while j + 1 < n and e[j + 1] and not m[j + 1]:
                j += 1
            took = j + 1 < n and m[j + 1]
            g = float(np.prod(1 + net[i + 1:j + 2]) - 1)  # held the day after each decision close
            out.append({"in": idx[i], "out": idx[min(j + 1, n - 1)], "handed": took, "ret": g})
            i = j + 1
        else:
            i += 1
    return out


# ---------- the computation ----------

def _compute() -> dict:
    t0 = time.time()
    close = te._daily_close()
    close = close[close.index < pd.Timestamp(common.today_utc())]
    idx = close.index
    ma = _ma50(close)
    c = close.values.astype(float)
    up20 = ct._gt(c, ct._mean(c, 20))
    ch, raw, problems, dropped = _channels(close)
    keys = [k for k in CHANNELS if k in ch]
    flags = {k: ch[k]["flag"].values.astype(float) for k in keys}

    rallies = _rallies(close, RISE, FALL)
    _describe(rallies, close, ma)
    alt = _rallies(close, RISE_ALT, FALL_ALT)
    troughs = [r["t"] for r in rallies]
    rng = np.random.default_rng(SEED + 401)

    # anatomy
    anat = {k: [_lead(flags[k], r, idx) for r in rallies] for k in keys}
    per = {}
    for k in keys:
        av = [a for a in anat[k] if a[0]]
        leads = [ld for _, ld in av if ld is not None and ld > 0]
        led, n = len(leads), len(av)
        n_known = _span(flags[k])
        base = []
        if n and n_known > 2 * GAP:
            for _ in range(DRAWS):
                base.append(_led_count(_shifted(flags[k], int(rng.integers(GAP, n_known - GAP))), rallies, idx)[0])
        per[k] = {"n": n, "led": led, "med": float(np.median(leads)) if leads else None,
                  "never": sum(1 for _, ld in av if ld is None),
                  "base_med": float(np.median(base)) if base else None,
                  "base_share": float(np.mean(np.array(base) >= led)) if base else None}

    # false alarms
    fa = {k: _false_alarms(flags[k], ma, close, troughs, rng) for k in keys}
    fa_disc = {k: _false_alarms(flags[k], ma, close, troughs, rng, before=pd.Timestamp(SPLIT)) for k in keys}

    # discovery pick
    disc = [r for r in rallies if r["t"] < pd.Timestamp(SPLIT)]
    test = [r for r in rallies if r["t"] >= pd.Timestamp(SPLIT)]
    dsc = {}
    for k in keys:
        led, n = _led_count(flags[k], disc, idx)
        f = fa_disc[k]["fa"]
        dsc[k] = {"led": led, "n": n, "rate": led / n if n else None, "fa": f}
    pool = [k for k in keys if dsc[k]["n"] >= MIN_AVAIL] or [k for k in keys if dsc[k]["n"] >= 1]
    order = sorted(pool, key=lambda k: (-dsc[k]["rate"], dsc[k]["fa"] if _ok(dsc[k]["fa"]) else 1.0,
                                        CHANNELS.index(k)))
    picked = order[0] if order else None
    used, runner = picked, False
    if picked is not None and not np.isfinite(flags[picked][idx >= pd.Timestamp(SPLIT)]).any():
        nxt = [k for k in order[1:] if np.isfinite(flags[k][idx >= pd.Timestamp(SPLIT)]).any()]
        used, runner = (nxt[0] if nxt else None), True
    min_used = MIN_AVAIL if any(dsc[k]["n"] >= MIN_AVAIL for k in keys) else 1

    res = {"close": close, "ma": ma, "ch": ch, "raw": raw, "keys": keys, "flags": flags, "rallies": rallies,
           "alt": alt, "anat": anat, "per": per, "fa": fa, "fa_disc": fa_disc, "dsc": dsc, "order": order,
           "picked": picked, "used": used, "runner": runner, "min_used": min_used, "problems": problems,
           "dropped": dropped, "disc": disc, "test": test, "up20": up20}
    if used is None:
        res["exam"] = None
        return res

    rec = _recent(flags[used])
    e1 = _e1(ma, up20, rec)
    change = close.pct_change().fillna(0.0).values
    net_e1, _ = te._run(np.concatenate([[np.nan], e1[:-1]]), change, COST)
    tr = []
    for r in test:
        early = (np.nan_to_num(e1[r["it"]:r["im"]], nan=0.0) > 0.5).any()
        g = float(np.prod(1 + net_e1[r["it"] + 1:r["im"] + 1]) - 1)
        tr.append({"led": bool(early), "cap": g})
    eps = _episodes(e1, ma, net_e1, idx, SPLIT)
    res.update({"e1": e1, "rec": rec, "test_rows": tr, "eps": eps})

    clock = _Clock(TEST_FIRST, TEST_LAST)
    if clock.problem:
        problems.append(clock.problem)
    coins = {}
    for sym, key in OTHER:
        try:
            coins[key] = ta._crypto_close(sym)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key} 現貨日線這次取不到（{type(exc).__name__}），沒有算進去。")
    exam = _exam(close, ma, up20, rec, clock, coins, ("ma50", "e1"), True)
    real = _exam(close, ma, up20, rec, clock, coins, ("e1",), False)["e1"]
    n_known = _span(flags[used])
    prng = np.random.default_rng(SEED + 402)
    better = 0
    for _ in range(DRAWS):
        fl = _shifted(flags[used], int(prng.integers(GAP, n_known - GAP)))
        x = _exam(close, ma, up20, _recent(fl), clock, coins, ("e1",), False)["e1"]
        better += int(x["low"] > real["low"] + 1e-12 or (abs(x["low"] - real["low"]) <= 1e-12
                                                          and x["c_edge"] >= real["c_edge"]))
    exam["placebo"] = better / DRAWS
    exam["real_np"] = real
    exam["clock_first"] = clock.labels[0] if clock.labels else None
    exam["seconds"] = time.time() - t0
    res["exam"] = exam
    return res


# ---------- the study ----------

def _lead_cell(a: tuple) -> str:
    av, ld = a
    if not av:
        return "沒資料"
    if ld is None:
        return "高點前沒出現"
    return f"早 {ld:.0f} 天" if ld > 0 else ("同一天" if ld == 0 else f"晚 {-ld:.0f} 天")


ETF_LAUNCH = pd.Timestamp("2024-01-11")  # first trading day of the US spot bitcoin ETFs


def _val(s: pd.Series | None, t: pd.Timestamp, etf: bool = False) -> float | None:
    if s is None:
        return None
    if etf and t < ETF_LAUNCH:
        return 0.0  # the funds did not exist yet: cumulative net flow is truly zero, not missing
    s = s.dropna()
    s = s[s.index <= t]
    if not len(s) or (t - s.index[-1]).days > 5:
        return None
    return float(s.iloc[-1])


def study() -> dict:
    r = _compute()
    close, ma, idx = r["close"], r["ma"], r["close"].index
    keys, flags, rallies = r["keys"], r["flags"], r["rallies"]
    raw = r["raw"]

    # table 1: rallies
    rows1 = []
    for i, x in enumerate(rallies, 1):
        part = "發現期" if x["t"] < pd.Timestamp(SPLIT) else "測試期"
        rows1.append([f"{i}（{part}）", _d(x["t"]), f"{x['pt']:,.0f}", _d(x["m"]), f"{x['lead_days']} 天",
                      _p(x["miss"]), f"{_d(x['p'])}{'' if x['done'] else '（還沒結束）'}", _p(x["gain"]),
                      f"{x['days']} 天", _share(math.log1p(x["miss"]) / math.log1p(x["gain"]) if x["gain"] > 0 else None)])
    # table 2: channel list
    rows2 = []
    for k in CHANNELS:
        if k in keys:
            f = flags[k]
            known = np.flatnonzero(np.isfinite(f))
            first = _d(idx[known[0]]) if len(known) else "—"
            on = float(np.nanmean(f)) if len(known) else None
            rows2.append([CN[k], EXPLAIN[k], first, LAG_CN[k], _share(on)])
        else:
            rows2.append([CN[k], EXPLAIN[k], "取不到，已刪除", LAG_CN[k], "—"])
    # table 3: anatomy per rally (lead)
    cols3 = ["管道"] + [f"{i}：{_d(x['t'])[:7]}" for i, x in enumerate(rallies, 1)]
    rows3 = []
    for k in keys:
        cells = []
        for j, x in enumerate(rallies):
            av, ld = r["anat"][k][j]
            if not av:
                cells.append("沒資料")
                continue
            lv = r["ch"][k]["lvl"].values.astype(float)
            cells.append(f"{_lead_cell((av, ld))}｜低點→均線 {_chg(k, lv, x['it'], x['im'])}｜均線→高點 "
                         f"{_chg(k, lv, x['im'], x['ip'])}")
        rows3.append([CN[k]] + cells)
    # table 4: per channel lead summary + false alarms
    rows4 = []
    for k in keys:
        p, f = r["per"][k], r["fa"][k]
        rows4.append([CN[k], f"{p['n']} 次裡 {p['led']} 次" if p["n"] else "沒資料",
                      f"{p['med']:.0f} 天" if _ok(p["med"]) else "—", f"{p['never']} 次" if p["n"] else "—",
                      (f"{p['base_med']:.0f} 次｜{_share(p['base_share'])}" if _ok(p["base_med"]) else "—"),
                      f"{f['n']} 次", _share(f["fa"]),
                      f"{_p(f['mean30'])}／{_p(f['med30'])}（{_p(f['bmean30'])}／{_p(f['bmed30'])}）",
                      f"{_p(f['mean60'])}／{_p(f['med60'])}（{_p(f['bmean60'])}／{_p(f['bmed60'])}）",
                      f"{_share(f['rand30'])}｜{_share(f['rand60'])}"])
    # table 5: sources of the rise (levels)
    rows5 = []
    for i, x in enumerate(rallies, 1):
        t, pk = x["t"], x["p"]
        cells = [f"{i}：{_d(t)} → {_d(pk)}"]
        a, b = _val(raw.get("stable"), t), _val(raw.get("stable"), pk)
        cells.append(f"{a / 1e8:,.0f} → {b / 1e8:,.0f} 億美元（{_p(b / a - 1)}）" if _ok(a) and _ok(b) else "—")
        a, b = _val(raw.get("exbal"), t), _val(raw.get("exbal"), pk)
        cells.append(f"{a / 1e4:,.0f} → {b / 1e4:,.0f} 萬顆（{b - a:+,.0f} 顆）" if _ok(a) and _ok(b) else "—")
        a, b = _val(raw.get("etf"), t, True), _val(raw.get("etf"), pk, True)
        cells.append(f"{a / 1e8:,.0f} → {b / 1e8:,.0f} 億美元（{(b - a) / 1e8:+,.0f} 億）" if _ok(a) and _ok(b) else "—")
        a, b = _val(raw.get("oi"), t + DAY), _val(raw.get("oi"), pk + DAY)
        if _ok(a) and _ok(b):
            ua, ub = a * x["pt"], b * x["pp"]
            cells.append(f"{a / 1e4:,.1f} → {b / 1e4:,.1f} 萬顆；{ua / 1e8:,.0f} → {ub / 1e8:,.0f} 億美元")
        else:
            cells.append("—")
        rows5.append(cells)
    # table 6: exploratory inflow vs market value
    rows6 = []

    def seg(a_t, b_t):
        mc = (_val(raw.get("mcap"), b_t), _val(raw.get("mcap"), a_t))
        dm = mc[0] - mc[1] if _ok(mc[0]) and _ok(mc[1]) else None
        st = (_val(raw.get("stable"), a_t), _val(raw.get("stable"), b_t))
        ds = st[1] - st[0] if _ok(st[0]) and _ok(st[1]) else None
        et = (_val(raw.get("etf"), a_t, True), _val(raw.get("etf"), b_t, True))
        de = et[1] - et[0] if _ok(et[0]) and _ok(et[1]) and b_t >= ETF_LAUNCH else None
        rc = (_val(raw.get("rcap"), a_t), _val(raw.get("rcap"), b_t))
        dr = rc[1] - rc[0] if _ok(rc[0]) and _ok(rc[1]) else None
        flow = None if ds is None and de is None else (ds or 0.0) + (de or 0.0)
        ratio = dm / flow if _ok(dm) and _ok(flow) and flow > 0 else None
        rratio = dm / dr if _ok(dm) and _ok(dr) and dr > 0 else None
        return dm, ds, de, dr, ratio, rratio

    for i, x in enumerate(rallies, 1):
        dm, ds, de, dr, ratio, rratio = seg(x["t"], x["p"])
        _, _, _, _, r1, q1 = seg(x["t"], x["m"])
        _, _, _, _, r2, q2 = seg(x["m"], x["p"])
        rows6.append([f"{i}：{_d(x['t'])} → {_d(x['p'])}", _bn(dm), _bn(ds), _bn(de), _bn(dr),
                      _x(ratio), f"{_x(r1)}｜{_x(r2)}" if (r1 or r2) else "", _x(rratio),
                      f"{_x(q1)}｜{_x(q2)}" if (q1 or q2) else ""])

    # discovery and test tables
    dsc = r["dsc"]
    rows7 = []
    for k in keys:
        x = dsc[k]
        rank = (r["order"].index(k) + 1) if k in r["order"] else None
        rows7.append([CN[k], f"{x['n']} 次裡 {x['led']} 次" if x["n"] else "沒資料", _share(x["rate"]),
                      _share(x["fa"]), f"第 {rank}" if rank else f"不列入（少於 {r['min_used']} 次）"])

    used, picked = r["used"], r["picked"]
    tables_extra, verdict_test, now_e1 = [], "", "—"
    ex = r.get("exam")
    pass_ = None
    if ex is not None:
        rows8 = []
        for x, tr in zip(r["test"], r["test_rows"]):
            rows8.append([_d(x["t"]), _d(x["m"]), _p(x["miss"]), "有" if tr["led"] else "沒有", _p(tr["cap"])])
        false = [e for e in r["eps"] if not e["handed"]]
        good = [e for e in r["eps"] if e["handed"]]
        fr = [e["ret"] for e in false]
        m5, e5 = ex["ma50"], ex["e1"]
        loss_ok = e5["c_deep_med"] >= m5["c_deep_med"] - LOSS_GAP
        pass_ = bool(e5["low"] >= m5["low"] - 1e-12 and loss_ok and ex["placebo"] <= 0.10)
        rows9 = []
        for k, lab in (("ma50", "50 日均線（基準）"), ("e1", f"E1：50 日均線，或「{CN[used]}」30 天內亮起且站上 20 日均線")):
            x = ex[k]
            rows9.append([lab, f"{x['d_n']} 個裡 {x['d_beat']} 個", f"{x['c_n']:,} 格裡 {x['c_beat']:,} 個",
                          f"{x['p_n']:,} 次裡 {x['p_beat']:,} 次", f"{x['o_n']} 個裡 {x['o_beat']} 個",
                          _share(x["low"]), f"{x['c_edge']:+.1f} 點", f"{x['c_deep_med']:.0f}%"])
        tables_extra = [
            {"caption": (f"測試期（低點在 {SPLIT} 以後）的大漲：E1 有沒有比 50 日均線早上車、在均線亮之前賺到多少（含 0.10% 成本）。"
                         f"另外，{SPLIT} 以後 E1 在均線沒亮時先上車共 {len(r['eps'])} 次：{len(good)} 次後來交給均線接手，"
                         f"{len(false)} 次沒等到均線就下車（假進場），假進場的報酬中位數 "
                         f"{_p(float(np.median(fr))) if fr else '—'}、合計（連乘）{_p(float(np.prod(1 + np.array(fr)) - 1)) if fr else '—'}。"),
             "columns": ["低點日", "50 日均線亮的日子", "均線錯過的漲幅", "E1 有沒有先上車", "E1 在均線亮之前賺到"],
             "rows": rows8},
            {"caption": (f"公平考試，只用 {TEST_FIRST[:7]} 起的起算日：比特幣日線 {ex['n_daily']} 個月初起算、"
                         f"{ex['n_clock']} 個月 × 24 個收盤時間（照課本做 0.10%；一般人做法 10 小時後、0.25%、漏掉 10%、"
                         f"每個時間 {SIMS} 次），其他幣 ETH、SOL、BNB、XRP（用自己的 50／20 日均線、比特幣的管道訊號）起算到三年前。"
                         f"排名分數 = 四項「報酬贏過一直抱著」比例裡最低的一項。50 日均線用同樣的起算日重算。"),
             "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他幣：贏抱著",
                         "排名分數", "每年比抱著多（中位數）", "最深虧損中位數"],
             "rows": rows9},
        ]
        e_now = r["e1"][-1]
        now_e1 = f"{'持有' if ma[-1] > 0.5 else '現金'}｜{'持有' if e_now > 0.5 else '現金'}"
        tag = "（發現期第一名沒有測試期資料，改用第二名）" if r["runner"] else ""
        verdict_test = (f"發現期（低點在 2021 年以前的 {len(r['disc'])} 次大漲）程式挑出的是「{CN[picked]}」{tag}"
                        f"{'' if used == picked else f'，實際測試用「{CN[used]}」'}：發現期 "
                        f"{dsc[used]['n']} 次裡 {dsc[used]['led']} 次比均線早。測試期 {len(r['test'])} 次大漲，E1 先上車 "
                        f"{sum(t['led'] for t in r['test_rows'])} 次；公平考試排名分數 E1 {_share(e5['low'])}、均線 {_share(m5['low'])}，"
                        f"最深虧損中位數 {e5['c_deep_med']:.0f}% 對 {m5['c_deep_med']:.0f}%，隨機對照 {_share(ex['placebo'])}，"
                        f"{'通過' if pass_ else '沒有通過'}事先定好的標準。")
    # verdict
    led_any = [k for k in keys if r["per"][k]["n"] and r["per"][k]["led"] / r["per"][k]["n"] >= 0.5]
    chance = [k for k in keys if _ok(r["per"][k]["base_share"]) and r["per"][k]["base_share"] > 0.10]
    med_miss = float(np.median([x["miss"] for x in rallies])) if rallies else None
    first = (f"2013 年以來用「漲 80% 以上、中途沒跌 30%」機械式找出 {len(rallies)} 次大漲；50 日均線在每次大漲裡平均晚 "
             f"{np.median([x['lead_days'] for x in rallies]):.0f} 天（中位數）才亮，錯過從低點起的漲幅中位數 {_p(med_miss)}。")
    second = (f"{len(keys)} 個管道裡，至少一半大漲都比均線早亮的有 {len(led_any)} 個，但把訊號隨機挪動時間後也能「領先」差不多次數的有 "
              f"{len(chance)} 個——因為這些訊號本來就常亮，低點前後 30～60 天裡亮一次很容易。")
    verdict = first + second + (verdict_test or "沒有任何管道在發現期有資料，沒有做測試。")

    # now
    now = [{"label": "比特幣：50 日均線｜E1", "value": now_e1},
           {"label": "最後一根收完的日線", "value": f"{_d(idx[-1])}，{close.iloc[-1]:,.0f} 美元"}]
    if used:
        now.append({"label": "E1 用的管道", "value": CN[used]})
    for k in keys:
        f = flags[k]
        son = np.flatnonzero(_switch_on(f))
        last = _d(idx[son[-1]]) if len(son) else "—"
        state = "亮" if f[-1] > 0.5 else ("沒亮" if np.isfinite(f[-1]) else "沒有最新資料")
        now.append({"label": f"管道：{CN[k]}", "value": f"{state}；最近一次亮起 {last}"})

    alt_n = len(r["alt"])
    notes = [
        "這裡的大漲是用固定規則從日收盤價找出來的，不是挑出來的；80% 與 30% 沒有調過。"
        f"改用「漲 50%、跌 25%」只當參考：會找出 {alt_n} 次。",
        "每個管道「亮起」用同一個做法：14 天變化高於自己過去一年 14 天變化的前 20%（至少 180 天資料）；"
        "Coinbase 溢價和資金費率則是 7 天平均由負轉正。每個數字只用當天收盤時已經公布的資料（延遲見第二張表）。",
        "「早幾天」是從低點前 30 天到 50 日均線亮的那天之間，第一次亮起的日子；如果那段都沒亮，就看均線亮了以後到高點之前。"
        "因為很多訊號一年裡有兩成的日子會亮，低點附近隨便都會亮一次，所以另外把訊號整段挪動時間 200 次，看隨機的情況會「領先」幾次。",
        "假警報：均線沒亮時訊號亮起，接下來 60 天裡沒有出現大漲的低點、50 日均線也沒有亮起並撐 30 天以上，就算假警報。"
        "括號裡是同一段期間所有「均線沒亮的日子」的平均／中位數；最後一欄是隨機抽同樣多天、平均報酬至少一樣好的比例（越小越不像運氣）。",
        "有些管道確實是「真的有錢進來」：穩定幣淨發行（新的美元換成穩定幣）、美國現貨 ETF 淨申購（新錢透過基金買比特幣）、"
        "實現市值（所有持有人買進成本的總和）。其他管道——Coinbase 溢價、主動買進比例、資金費率、未平倉量、關注度——"
        "量的是買方有多急、槓桿有多大，不是有多少錢進來。",
        "兩個真正的限制：（1）價格由最後成交的那幾筆決定，所以市值增加通常是淨流入的好幾倍，倍數還會隨持有人願不願意賣而變；"
        "（2）每一筆交易買方的錢都進了賣方口袋，所以要看的是存量（穩定幣供給、ETF 持有、實現市值）有沒有變大，而不是成交量。"
        "另外，漲的時候量到的流入，可能只是跟著價格走（這個專案已經發現穩定幣供給就是這樣）。",
        "現貨 ETF 在 2024-01-11 才開始交易，之前的累計淨流入照實記為 0（不是缺資料）；ETF 管道要先累積半年以上資料才有訊號，"
        "所以沒有任何一次大漲的低點落在它能用的期間。"
        + ("到目前為止它每次亮起時，50 日均線都已經亮著——資金是跟著價格走的。" if "etf" in r["fa"] and r["fa"]["etf"]["n"] == 0 else ""),
        "「市值增加 ÷ 量到的流入」那張表是探索用的，不屬於任何及格標準；市值用 Coin Metrics 的 CapMrktCurUSD（價格 × 流通量），"
        "實現市值 = CapMrktCurUSD ÷ CapMVRVCur。沒有資料的格子留空，不去估。",
        "交易所餘額與流出入來自 Coin Metrics 社群版：它用今天的交易所地址標籤往回套整段歷史（2026-04 一次重算），"
        "不是當時的人看得到的數字，早年的結果要打折扣。成交量、主動買進比例只來自幣安一家，2017-08 以後才有。",
        "和 rally_timing（跌深後用價格型態提早上車）、crash_timing（更快出場、更快回來）不同：這裡的大漲不需要前面先大跌，"
        "候選訊號是資金管道，不是價格型態，而且只在 2021 年以前挑一個、2021 年以後才考。",
        f"測試期只有幾次大漲，公平考試的起算日也只有 {TEST_FIRST[:7]} 到 {TEST_LAST[:7]}，證據很薄；相鄰起算日共用大部分歷史，不是獨立的證據。",
        "過去的結果不保證未來。這裡只描述規則在歷史上做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    if r["dropped"]:
        notes.append("取不到而刪除的管道：" + "、".join(f"{CN[k]}（{v}）" for k, v in r["dropped"].items()) + "。")
    notes += r["problems"]

    tables = [
        {"caption": (f"2013 年以來的大漲（低點後漲 {RISE * 100:.0f}% 以上、中途沒有從高點跌 {FALL * 100:.0f}%）。"
                     "「50 日均線亮的日子」是低點後第一次收盤站上 50 日均線；「均線錯過的漲幅」是從低點到那天已經漲了多少。"),
         "columns": ["編號", "低點日", "低點價（美元）", "50 日均線亮的日子", "低點到均線亮隔幾天", "均線錯過的漲幅",
                     "高點日", "低點到高點", "低點到高點幾天", "錯過的占整段漲幅（以對數算）"],
         "rows": rows1},
        {"caption": "十二個資金管道：每個在講什麼、第一天能用的日子、晚幾天才知道、歷史上亮起的日子占多少。",
         "columns": ["管道", "白話說明", "第一天有訊號", "晚幾天知道", "亮起的日子占比"],
         "rows": rows2},
        {"caption": ("每次大漲、每個管道：比 50 日均線早幾天亮起（從低點前 30 天算起），以及這個管道從低點到均線亮、"
                     "從均線亮到高點各變了多少。百分比是水準變化，「點」是百分點變化。"),
         "columns": cols3, "rows": rows3},
        {"caption": ("每個管道的總結。左半：大漲裡比均線早亮幾次、早幾天（中位數）、到高點都沒亮幾次，以及把訊號隨機挪動時間後"
                     "「早亮」次數的中位數｜隨機挪動也能早亮一樣多次的比例。右半：均線沒亮時訊號亮起的全部次數、假警報比例、"
                     "接下來 30／60 天比特幣的平均／中位數（括號是所有均線沒亮日子的平均／中位數）、隨機抽樣做得一樣好的比例。"),
         "columns": ["管道", "大漲裡比均線早亮", "早幾天（中位數）", "到高點都沒亮", "隨機挪動：早亮次數｜一樣多的比例",
                     "均線沒亮時亮起", "假警報比例", "之後 30 天：平均／中位數（基準）", "之後 60 天：平均／中位數（基準）",
                     "隨機抽樣做得一樣好：30 天｜60 天"],
         "rows": rows4},
        {"caption": ("每次大漲時，各個資金存量從低點到高點的變化（用資料當天的數字，只是描述）：穩定幣總供給、交易所比特幣餘額、"
                     "美國現貨 ETF 累計淨流入（2024 起）、幣安合約未平倉量（顆數，以及乘上當時價格的美元）。"),
         "columns": ["大漲", "穩定幣供給", "交易所餘額", "現貨 ETF 累計淨流入", "合約未平倉量"],
         "rows": rows5},
        {"caption": ("探索用（不屬於任何及格標準）：比特幣市值增加了多少，和「直接量得到的流入」比。量得到的流入 = 穩定幣供給增加 + "
                     "現貨 ETF 淨流入（有資料的才算）。「每一塊錢流入換到幾塊錢市值」也分成低點到均線亮｜均線亮到高點兩段，"
                     "看早期的錢是不是推得比較多。空白 = 沒有資料，沒有估。"),
         "columns": ["大漲", "比特幣市值增加", "穩定幣供給增加", "現貨 ETF 淨流入", "實現市值增加",
                     "市值增加 ÷ 量到的流入", "前段｜後段", "市值增加 ÷ 實現市值增加", "前段｜後段"],
         "rows": rows6},
        {"caption": (f"發現期（低點在 {SPLIT} 以前的 {len(r['disc'])} 次大漲）的挑選：程式依「早亮次數 ÷ 有資料的次數」排序，"
                     f"至少要有 {r['min_used']} 次有資料，同分看 {SPLIT} 以前的假警報比例（低的優先）。"),
         "columns": ["管道", "發現期：比均線早亮", "比例", f"假警報比例（{SPLIT[:4]} 以前）", "排名"],
         "rows": rows7},
    ] + tables_extra
    if ex is not None:
        notes.insert(0, (f"事先定好的及格標準：E1 排名分數不低於 50 日均線（{_share(ex['e1']['low'])} 對 {_share(ex['ma50']['low'])}）、"
                         f"最深虧損中位數不比均線深超過 {LOSS_GAP:.0f} 點（{ex['e1']['c_deep_med']:.0f}% 對 {ex['ma50']['c_deep_med']:.0f}%）、"
                         f"隨機對照（把訊號挪動 180 天以上，{DRAWS} 次）做得一樣好的比例不超過 10%（{_share(ex['placebo'])}）。"
                         f"結果：{'通過' if pass_ else '沒有通過'}。"))

    return {
        "id": "rally_sources",
        "title": "大漲的錢從哪裡來",
        "question": ("比特幣大漲時一定有錢從某處進來。在歷次大漲剛開始時，有沒有哪個資金管道比 50 日均線更早亮起，"
                     "而且很少在沒有大漲時亂亮？"),
        "asof": _d(idx[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": _sources(r),
    }


def _sources(r: dict) -> list[dict]:
    src = [
        {"name": "比特幣日收盤：幣安現貨 BTCUSDT 日線（2017-08 以前 blockchain.info）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
    ]
    k = set(r["keys"])
    if "stable" in k:
        src.append({"name": "DefiLlama 穩定幣總供給 API", "url": ss.LLAMA})
    if k & {"exbal", "netout"}:
        src.append({"name": "Coin Metrics 社群版 API（SplyExNtv、FlowInExNtv、FlowOutExNtv）", "url": ff.CM_URL})
    if "adr" in k:
        src.append({"name": "Coin Metrics 社群版 API（AdrActCnt）", "url": ng.CM_URL})
    if "mcap" in r["raw"]:
        src.append({"name": "Coin Metrics 社群版 API（CapMrktCurUSD、CapMVRVCur）", "url": onchain2.CM_URL})
    if k & {"cbp", "kimchi"}:
        src += [{"name": "Coinbase Exchange 公開日線 BTC-USD", "url": km.COINBASE_URL},
                {"name": "Upbit 公開日線 KRW-BTC", "url": km.UPBIT_URL},
                {"name": "美國聯準會 H.10 匯率（美元對韓元）", "url": km.FED_ZIP}]
    if k & {"taker", "volume"}:
        src.append({"name": "幣安現貨 K 線 API（成交量與主動買進量）",
                    "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"})
    if "oi" in k:
        src.append({"name": "幣安合約每日 metrics 檔（未平倉量）", "url": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/"})
    if "funding" in k:
        src.append({"name": "幣安合約資金費率歷史 API", "url": fp.FAPI + "/fapi/v1/fundingRate"})
    if "etf" in k:
        src.append({"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": "https://farside.co.uk/bitcoin-etf-flow-all-data/"})
    if "wiki" in k:
        src.append({"name": "維基媒體瀏覽量 API（英文維基 Bitcoin）",
                    "url": "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Bitcoin/daily/20150701/"})
    if r.get("exam"):
        src.append({"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"})
    return src


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
