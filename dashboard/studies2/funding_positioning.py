"""Historical study "funding_positioning": is funding negative now, what followed negative funding before, and
what the long/short account ratio can and cannot say.

Three questions, all answered from public, key-less endpoints through common.get_json (fixed user agent):

1. Latest settled and predicted next funding for BTC, ETH, SOL on Binance and BTC on Bybit and OKX, as % per
   8 hours with exact signs, plus a plain-language zone.
2. Binance BTCUSDT funding since 2019-09: 7-day average per day, bucketed, with what bitcoin did 14 and 30 days
   later (Binance spot daily candles).
3. Binance long/short ratios for BTCUSDT: share of accounts long (all / top accounts), share of top accounts'
   position size that is long, the implied average short account size, and the 30-day range.

Everything is descriptive and in-sample: overlapping windows, few separate episodes in the extreme bucket, one
exchange for the ratios. A source that blocks us is reported as "取不到", never worked around.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard import common

FAPI = "https://fapi.binance.com"
SPOT_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")  # both are Binance's own public hosts
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com"
SINCE = "2019-09-01"
HORIZONS = (14, 30)
DROP = 0.10  # "fell 10% or more at some point"
MISSING = "取不到"
CN = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "索拉納"}
# buckets of the 7-day average funding (% per 8 hours); values are rounded to 6 decimals before comparing
BUCKETS = (
    ("≤ -0.01%（明顯為負）", lambda v: v <= -0.01),
    ("-0.01% ~ 0（略為負）", lambda v: -0.01 < v < 0),
    ("0 ~ +0.005%", lambda v: 0 <= v < 0.005),
    ("+0.005% ~ +0.01%", lambda v: 0.005 <= v <= 0.01),
    ("> +0.01%（偏熱）", lambda v: v > 0.01),
)


# ---------- formatting ----------

def _rate(v) -> str:
    """Funding in % with an exact sign: a tiny non-zero rate is never shown as 0."""
    if v is None or not math.isfinite(v):
        return MISSING
    if v == 0:
        return "0.0000%"
    for digits in (4, 6, 8):
        if round(v, digits) != 0:
            return f"{v:+.{digits}f}%"
    return f"{v:+.2e}%"


def _pct(v, digits: int = 1) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v:+.{digits}f}%"


def _share(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v:.1f}%"


def _times(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"100 次裡 {v:.0f} 次"


def _zone(rate, avg7=None) -> str:
    """Plain-language zone from the latest settled rate (% per 8h); the 7-day average decides the panic level."""
    if rate is None:
        return MISSING
    if avg7 is not None and round(avg7, 6) <= -0.01:
        return "恐慌等級"
    rate = round(rate, 6)
    if rate <= -0.01:
        return "明顯偏空"
    if rate <= -0.005:
        return "偏空"
    if rate < 0.005:
        return "中性（略為負值）" if rate < 0 else ("中性（略為正值）" if rate > 0 else "中性")
    return "正常偏多"


def _day(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


# ---------- data ----------

def _per8h(times_ms: list) -> float:
    """Factor that turns one settlement's rate into a per-8-hours rate (1.0 when funding settles every 8 hours)."""
    if len(times_ms) < 2:
        return 1.0
    hours = round(abs(times_ms[-1] - times_ms[-2]) / 3_600_000)
    return 8 / hours if hours in (1, 2, 4, 8) else 1.0


def _binance_now(symbol: str) -> dict:
    """Latest settled, predicted next and 7-day average funding of one Binance perpetual (% per 8 hours)."""
    hist = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 100})
    hist = sorted(hist, key=lambda p: p["fundingTime"])
    k = _per8h([p["fundingTime"] for p in hist])
    last_ms = hist[-1]["fundingTime"]
    week = [float(p["fundingRate"]) for p in hist if p["fundingTime"] > last_ms - 7 * 86_400_000 + 60_000]
    nxt = common.get_json(FAPI + "/fapi/v1/premiumIndex", {"symbol": symbol})
    return {"settled": float(hist[-1]["fundingRate"]) * 100 * k, "predicted": float(nxt["lastFundingRate"]) * 100 * k,
            "avg7": float(np.mean(week)) * 100 * k if week else None}


def _bybit_now() -> dict:
    hist = common.get_json(BYBIT + "/v5/market/funding/history",
                           {"category": "linear", "symbol": "BTCUSDT", "limit": 8}, timeout=15)
    if hist.get("retCode") != 0:
        raise RuntimeError(f"Bybit 回報錯誤：{hist.get('retMsg')}")
    rows = sorted(hist["result"]["list"], key=lambda p: int(p["fundingRateTimestamp"]))
    k = _per8h([int(p["fundingRateTimestamp"]) for p in rows])
    tick = common.get_json(BYBIT + "/v5/market/tickers", {"category": "linear", "symbol": "BTCUSDT"}, timeout=15)
    if tick.get("retCode") != 0:
        raise RuntimeError(f"Bybit 回報錯誤：{tick.get('retMsg')}")
    return {"settled": float(rows[-1]["fundingRate"]) * 100 * k,
            "predicted": float(tick["result"]["list"][0]["fundingRate"]) * 100 * k, "avg7": None}


def _okx_now() -> dict:
    hist = common.get_json(OKX + "/api/v5/public/funding-rate-history",
                           {"instId": "BTC-USDT-SWAP", "limit": 8}, timeout=15)
    if hist.get("code") != "0":
        raise RuntimeError(f"OKX 回報錯誤：{hist.get('msg')}")
    rows = sorted(hist["data"], key=lambda p: int(p["fundingTime"]))
    k = _per8h([int(p["fundingTime"]) for p in rows])
    settled = rows[-1].get("realizedRate") or rows[-1]["fundingRate"]
    cur = common.get_json(OKX + "/api/v5/public/funding-rate", {"instId": "BTC-USDT-SWAP"}, timeout=15)
    if cur.get("code") != "0":
        raise RuntimeError(f"OKX 回報錯誤：{cur.get('msg')}")
    return {"settled": float(settled) * 100 * k, "predicted": float(cur["data"][0]["fundingRate"]) * 100 * k,
            "avg7": None}


def _funding_history() -> pd.Series:
    """Every Binance BTCUSDT settlement since 2019-09, in % per settlement (8 hours), DatetimeIndex (UTC, naive)."""
    out, start = [], int(pd.Timestamp(SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(20):
        page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1000, "startTime": start})
        if not page:
            break
        out += page
        start = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    if len(out) < 1000:
        raise RuntimeError(f"幣安資金費率歷史只回傳 {len(out)} 筆")
    series = pd.Series([float(p["fundingRate"]) * 100 for p in out],
                       index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
    return series[~series.index.duplicated()].sort_index()


def _spot(path: str, params: dict):
    last = None
    for host in SPOT_HOSTS:
        try:
            return common.get_json(host + path, params, timeout=15)
        except Exception as exc:  # noqa: BLE001 - blocked or down: try Binance's other public host
            last = exc
    raise last


def _klines() -> pd.DataFrame:
    """Binance spot BTCUSDT daily open, low, close since 2019-09; the last row is today's unfinished candle."""
    rows, start = [], int(pd.Timestamp(SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(8):
        page = _spot("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 86_400_000
        if len(page) < 1000:
            break
    frame = pd.DataFrame([[r[0], float(r[1]), float(r[3]), float(r[4])] for r in rows],
                         columns=["ms", "open", "low", "close"]).drop_duplicates("ms").sort_values("ms")
    frame.index = pd.to_datetime(frame["ms"], unit="ms")
    if len(frame) < 500:
        raise RuntimeError(f"幣安現貨日線只回傳 {len(frame)} 筆")
    return frame[["open", "low", "close"]]


# ---------- part 2: what followed ----------

def _history_table(funding: pd.Series, candles: pd.DataFrame) -> dict:
    daily = funding.resample("1D").mean()
    avg7 = daily.rolling(7, min_periods=7).mean().round(6).dropna()
    closed = candles.iloc[:-1]  # finished days only
    avg7 = avg7[avg7.index.isin(closed.index)]
    close = closed["close"].reindex(avg7.index)
    low = closed["low"]
    fwd = {}
    for h in HORIZONS:
        change = closed["close"].shift(-h) / closed["close"] - 1
        # lowest low of the next h days (days D+1 .. D+h), against the close of day D
        worst = low[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / closed["close"] - 1
        fwd[h] = (change.reindex(avg7.index), worst.reindex(avg7.index))
    rows, summary = [], {}
    groups = [(label, avg7.map(test).astype(bool)) for label, test in BUCKETS]
    groups.append(("全部日子（對照）", pd.Series(True, index=avg7.index)))
    for label, mask in groups:
        days = int(mask.sum())
        # an episode is a run of consecutive calendar days inside the bucket
        inside = mask.reindex(pd.date_range(avg7.index[0], avg7.index[-1], freq="D"), fill_value=False)
        episodes = int((inside & ~inside.shift(1, fill_value=False)).sum())
        summary[label] = {"days": days, "episodes": episodes}
        for h in HORIZONS:
            change, worst = fwd[h]
            ok = mask & change.notna() & worst.notna()
            n = int(ok.sum())
            if n == 0:
                cells = ["—", "—", "—"]
                stats = None
            else:
                c, w = change[ok], worst[ok]
                stats = {"n": n, "median": float(c.median() * 100), "up": float((c > 0).mean() * 100),
                         "drop": float((w <= -DROP).mean() * 100)}
                cells = [_pct(stats["median"]), _times(stats["up"]), _times(stats["drop"])]
            summary[label][h] = stats
            rows.append([label if h == HORIZONS[0] else "", (f"{days} 天" if label.startswith("全部") else f"{days} 天／{episodes} 段") if h == HORIZONS[0] else "",
                         f"{h} 天後"] + cells)
    return {"rows": rows, "summary": summary, "first": avg7.index[0].strftime("%Y-%m-%d"),
            "last": avg7.index[-1].strftime("%Y-%m-%d"), "n_days": int(len(avg7)), "close": close}


def _bucket(v: float) -> str:
    v = round(v, 6)
    return next(label for label, test in BUCKETS if test(v))


# ---------- part 3: positioning ----------

def _ratio(endpoint: str, period: str, limit: int) -> list:
    rows = common.get_json(f"{FAPI}/futures/data/{endpoint}", {"symbol": "BTCUSDT", "period": period, "limit": limit})
    if not rows:
        raise RuntimeError(f"幣安 {endpoint} 沒有資料")
    return sorted(rows, key=lambda r: r["timestamp"])


def _positioning(candles: pd.DataFrame) -> dict:
    all_acc = _ratio("globalLongShortAccountRatio", "4h", 1)[-1]
    top_acc = _ratio("topLongShortAccountRatio", "4h", 1)[-1]
    top_pos = _ratio("topLongShortPositionRatio", "4h", 1)[-1]
    oi = _ratio("openInterestHist", "4h", 1)[-1]
    month = _ratio("globalLongShortAccountRatio", "1d", 30)
    long_all, short_all = float(all_acc["longAccount"]) * 100, float(all_acc["shortAccount"]) * 100
    shares = [float(r["longAccount"]) * 100 for r in month]
    first_day = pd.to_datetime(int(month[0]["timestamp"]), unit="ms").normalize()
    start_price = float(candles["open"].get(first_day, np.nan))
    if not math.isfinite(start_price):
        start_price = float(candles["close"].asof(first_day))
    last_price = float(candles["close"].iloc[-1])
    return {"long_all": long_all, "short_all": short_all,
            "size_ratio": long_all / short_all if short_all > 0 else None,
            "long_top_acc": float(top_acc["longAccount"]) * 100, "long_top_pos": float(top_pos["longAccount"]) * 100,
            "oi_btc": float(oi["sumOpenInterest"]), "oi_usd": float(oi["sumOpenInterestValue"]),
            "asof_ms": int(all_acc["timestamp"]),
            "month_min": min(shares), "month_max": max(shares), "month_n": len(shares),
            "month_from": _day(int(month[0]["timestamp"])), "month_to": _day(int(month[-1]["timestamp"])),
            "month_price": (last_price / start_price - 1) * 100 if math.isfinite(start_price) else None}


# ---------- the study ----------

def study() -> dict:
    notes: list[str] = []
    # part 1: funding right now
    venues = [("幣安", "BTC", lambda: _binance_now("BTCUSDT")), ("幣安", "ETH", lambda: _binance_now("ETHUSDT")),
              ("幣安", "SOL", lambda: _binance_now("SOLUSDT")), ("Bybit", "BTC", _bybit_now), ("OKX", "BTC", _okx_now)]
    now_rows, readings = [], {}
    for venue, asset, fetch in venues:
        try:
            r = fetch()
            readings[(venue, asset)] = r
            now_rows.append([f"{venue} {CN[asset]}", _rate(r["settled"]), _rate(r["predicted"]),
                             _rate(r["avg7"]) if r["avg7"] is not None else "—", _zone(r["settled"], r["avg7"])])
        except Exception as exc:  # noqa: BLE001 - one venue failing must not fail the study
            if (venue, asset) == ("幣安", "BTC"):
                raise
            now_rows.append([f"{venue} {CN[asset]}", MISSING, MISSING, MISSING, MISSING])
            notes.append(f"{venue} 的{CN[asset]}資金費率這次取不到（{type(exc).__name__}），該列留空，沒有改用其他方法繞過。")
    btc = readings[("幣安", "BTC")]

    # part 2: history
    funding = _funding_history()
    candles = _klines()
    hist = _history_table(funding, candles)
    today_bucket = _bucket(btc["avg7"])
    recent = funding[funding.index >= funding.index[-1].normalize() - pd.Timedelta(days=29)]  # 30 calendar days
    neg_settle, n_settle = int((recent < 0).sum()), int(len(recent))
    by_day = recent.groupby(recent.index.normalize()).min()
    neg_days, n_days = int((by_day < 0).sum()), int(len(by_day))
    extreme = hist["summary"][BUCKETS[0][0]]
    mild = hist["summary"][BUCKETS[1][0]]
    base = hist["summary"]["全部日子（對照）"]

    # part 3: positioning (one exchange; if it fails the funding parts still stand)
    pos = None
    try:
        pos = _positioning(candles)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"幣安的多空帳戶比資料這次取不到（{type(exc).__name__}），第三部分留空。")

    tables = [
        {"caption": "現在的資金費率（每 8 小時；正值＝做多的人付錢，負值＝做空的人付錢）",
         "columns": ["市場", "最近一次結算", "預測下一次", "7 天平均", "區間"], "rows": now_rows},
        {"caption": (f"幣安比特幣 7 天平均資金費率落在各區間之後，比特幣的表現"
                     f"（{hist['first']} 到 {hist['last']}，共 {hist['n_days']} 天）"),
         "columns": ["7 天平均費率", "天數／段數", "之後", "漲跌中位數", "結束時比較高", "途中曾跌 10% 以上"],
         "rows": hist["rows"]},
    ]
    now = [
        {"label": "幣安比特幣：最近一次結算的資金費率", "value": f"{_rate(btc['settled'])}（每 8 小時）"},
        {"label": "幣安比特幣：預測下一次資金費率", "value": f"{_rate(btc['predicted'])}（每 8 小時）"},
        {"label": "幣安比特幣：7 天平均資金費率", "value": f"{_rate(btc['avg7'])}，落在「{today_bucket}」區間"},
        {"label": "目前區間", "value": _zone(btc["settled"], btc["avg7"])},
        {"label": "最近 30 天結算為負的天數",
         "value": f"{n_days} 天裡有 {neg_days} 天至少一次為負（{n_settle} 次結算裡 {neg_settle} 次為負）"},
    ]
    if pos:
        tables.append({
            "caption": f"幣安比特幣合約的多空比（{_day(pos['asof_ms'])}，4 小時資料）",
            "columns": ["項目", "數值", "怎麼讀"],
            "rows": [
                ["所有帳戶裡做多的比例", _share(pos["long_all"]), "算人頭：每個有倉位的帳戶算一票，不管倉位大小"],
                ["大戶帳戶裡做多的比例", _share(pos["long_top_acc"]), "保證金前 20% 的帳戶，一樣是算人頭"],
                ["大戶倉位裡做多的比例", _share(pos["long_top_pos"]), "同一批大戶帳戶，改用倉位大小來算"],
                ["平均一個做空帳戶的大小", f"約為做多帳戶的 {pos['size_ratio']:.1f} 倍" if pos["size_ratio"] else "—",
                 "多單總量一定等於空單總量，所以做空的人少，平均每人就比較大"],
                ["未平倉合約", f"{pos['oi_btc']:,.0f} BTC（約 {pos['oi_usd'] / 1e8:,.1f} 億美元）", "多空兩邊各有這麼多"],
                [f"近 {pos['month_n']} 天做多帳戶比例的範圍", f"{_share(pos['month_min'])} ~ {_share(pos['month_max'])}",
                 f"{pos['month_from']} 到 {pos['month_to']}，幣安只公布 30 天"],
                ["同一段期間比特幣價格變化", _pct(pos["month_price"]), "現貨價格，只是放在一起對照，不代表因果"],
            ]})
        now += [
            {"label": "幣安所有帳戶裡做多的比例（算人頭）", "value": _share(pos["long_all"])},
            {"label": "幣安大戶帳戶裡做多的比例（算人頭）", "value": _share(pos["long_top_acc"])},
            {"label": "幣安大戶倉位裡做多的比例（算倉位大小）", "value": _share(pos["long_top_pos"])},
            {"label": "平均一個做空帳戶的大小",
             "value": f"約為做多帳戶的 {pos['size_ratio']:.1f} 倍" if pos["size_ratio"] else "—"},
        ]

    sign = "負的" if btc["settled"] < 0 else ("正的" if btc["settled"] > 0 else "剛好是 0")
    verdict = (f"幣安比特幣最近一次結算的資金費率是 {_rate(btc['settled'])}（{sign}），預測下一次 {_rate(btc['predicted'])}，"
               f"7 天平均 {_rate(btc['avg7'])}，落在「{today_bucket}」區間；")
    e30, m30, b30 = extreme.get(30), mild.get(30), base.get(30)
    if e30 and m30 and b30:
        verdict += (f"歷史上 7 天平均 ≤ -0.01% 只有 {extreme['days']} 天、{extreme['episodes']} 段，之後 30 天 {_times(e30['up'])}比較高"
                    f"（略為負的區間是 {_times(m30['up'])}，全部日子是 {_times(b30['up'])}）。")
    if pos:
        verdict += (f"幣安有 {_share(pos['long_all'])} 的帳戶做多，但這是算人頭不是算錢，"
                    f"大戶的倉位有 {_share(pos['long_top_pos'])} 在多方，這份資料無法證實「大戶在做空」。")

    notes = [
        "資金費率是永續合約多空雙方定期互付的費用：合約比現貨貴時做多的人付錢給做空的人，反過來合約比現貨便宜時就是負的，由做空的人付錢。",
        "區間的分法：最近一次結算 ≥ +0.005% 是正常偏多，-0.005% 到 +0.005% 之間是中性，≤ -0.005% 是偏空，≤ -0.01% 是明顯偏空；"
        "7 天平均 ≤ -0.01% 才算恐慌等級。Bybit 和 OKX 只取最近 8 次結算，所以沒有 7 天平均。",
        "幣安在市場平靜時的資金費率剛好是 +0.01%，所以 7 天平均剛好等於 +0.01% 的日子很多，這些日子歸在「+0.005% ~ +0.01%」區間。",
        "歷史表的每一天都往後看 14 天和 30 天，相鄰的日子看的幾乎是同一段行情，視窗互相重疊，實際獨立的例子遠少於天數。",
        f"最極端的區間（7 天平均 ≤ -0.01%）只有 {extreme['days']} 天、分成 {extreme['episodes']} 段，例子很少，數字很容易被一兩段行情左右。",
        "區間的門檻是看過同一段歷史之後才定的（樣本內），沒有用另一段資料驗證過，也沒有算交易成本。",
        "「途中曾跌 10% 以上」是指之後那段期間內任何一天的最低價，比起算日那天的收盤價低 10% 以上；最近不滿 14 或 30 天的日子不列入計算。",
        "多空帳戶比只有幣安一家的資料，而且幣安只公布最近 30 天，沒有辦法做長期的歷史比較，其他交易所的情況可能不同。",
        "多空帳戶比算的是帳戶數不是金額；每一張多單的對面一定有一張空單，所以做多的帳戶多，只代表做空的帳戶比較少、平均比較大。",
        "誰拿著那些空單並沒有公開；造市商、避險的人和套利基金常常是空單的持有者，他們不是在賭價格下跌，所以不能從這份資料說「大戶在做空」。",
        "過去的結果不保證未來會一樣，這裡只是整理歷史，不是買賣建議。",
    ] + notes

    return {
        "id": "funding_positioning",
        "title": "資金費率與多空比",
        "question": "資金費率現在是不是負的、歷史上轉負之後比特幣怎麼走，以及「多數帳戶做多」能不能說明大戶在做空？",
        "asof": common.today_utc(),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安 U 本位永續合約：資金費率歷史與預測費率",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "幣安合約交易數據：多空帳戶比、大戶多空比、未平倉合約",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/trading-data"},
            {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Bybit BTCUSDT 永續合約資金費率", "url": BYBIT + "/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=8"},
            {"name": "OKX BTC-USDT 永續合約資金費率", "url": OKX + "/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=8"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
