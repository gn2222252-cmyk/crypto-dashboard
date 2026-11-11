"""Altcoin breadth from Binance spot daily closes of a fixed list of large coins. See dashboard/SPEC2.md.

Usage: python -m dashboard.collectors.breadth
The list is fixed here (large, long-listed USDT pairs; no stablecoins, no wrapped coins) and is not tuned.
Only closed daily candles are used.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from dashboard import common

GROUP = "breadth"
NAME = "幣安現貨公開日線（固定清單）"
URL = "https://www.binance.com/zh-TC/markets/overview"
HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")
COINS = ("ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK", "DOT", "LTC", "BCH", "UNI", "ATOM",
         "ETC", "XLM", "NEAR", "FIL", "AAVE", "ALGO", "ICP", "HBAR", "VET", "SAND", "MANA", "EOS", "XTZ", "THETA",
         "GRT", "MKR", "RUNE", "INJ", "APT", "ARB", "OP", "SUI", "SHIB", "FTM", "EGLD", "CRV")
DAYS = 2000
MIN_COINS = 15


def closes(symbol: str) -> pd.Series:
    """Closed daily candles, oldest first, up to DAYS of them."""
    rows, end = [], None
    while len(rows) < DAYS:
        params = {"symbol": symbol + "USDT", "interval": "1d", "limit": 1000}
        if end is not None:
            params["endTime"] = end
        batch = None
        for host in HOSTS:
            try:
                batch = common.get_json(host + "/api/v3/klines", params)
                break
            except Exception:
                continue
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < 1000:
            break
    now = time.time() * 1000
    data = {pd.Timestamp(r[0], unit="ms").strftime("%Y-%m-%d"): float(r[4]) for r in rows if r[6] < now}
    return pd.Series(data, dtype="float64").sort_index()


def _hist(series: pd.Series, digits: int = 2) -> list:
    clean = series.dropna().tail(common.HISTORY_POINTS)
    return [[d, round(float(v), digits)] for d, v in clean.items()]


def collect() -> list[dict]:
    try:
        btc = closes("BTC")
        frame = {}
        for coin in COINS:
            try:
                s = closes(coin)
                if len(s) >= 250:
                    frame[coin] = s
            except Exception:
                continue
        if len(frame) < MIN_COINS:
            raise RuntimeError(f"只拿到 {len(frame)} 個幣的日線，不足 {MIN_COINS} 個")
        px = pd.DataFrame(frame).reindex(btc.index)
    except Exception as err:
        return [common.failed(id="market_altseason_90d", group=GROUP, asset="MARKET", name="山寨幣季節指數（90 天）",
                              source=NAME, source_url=URL, error=repr(err))]
    asof = f"{btc.index[-1]}T23:59:59Z"
    listed = px.notna()
    rows = []

    def add(id, name, series, display, rule, note, reading="none", digits=2):
        rows.append(common.indicator(id=id, group=GROUP, asset="MARKET", name=name, value=round(float(series.iloc[-1]), digits),
                                     unit="%", display=display, asof=asof, source=NAME, source_url=URL, evidence="C",
                                     reading=reading, rule=rule, note=note, history=_hist(series, digits)))

    def share(flag: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
        n = valid.sum(axis=1)
        return (flag & valid).sum(axis=1).div(n).where(n >= MIN_COINS) * 100

    try:
        r90, b90 = px / px.shift(90) - 1, btc / btc.shift(90) - 1
        ok = r90.notna()
        season = share(r90.gt(b90, axis=0), ok).dropna()
        v = float(season.iloc[-1])
        label = "山寨季" if v >= 75 else "比特幣季" if v <= 25 else "介於兩者之間"
        add("market_altseason_90d", "山寨幣季節指數（90 天）", season,
            f"{v:.0f}%（{label}；{int(ok.iloc[-1].sum())} 個幣裡有 {int((r90.iloc[-1] > b90.iloc[-1]).sum())} 個贏過比特幣）",
            "過去 90 天漲幅贏過比特幣的幣種比例。75% 以上通稱山寨季，25% 以下通稱比特幣季；這是通行定義，本頁沒有驗證。",
            f"用固定的 {len(frame)} 個主流幣計算（不是全市場前 50 名）。山寨季常出現在行情後段、投機氣氛最濃的時候。")
    except Exception:
        pass
    for window, id, name in ((200, "market_pct_above_ma200", "站上 200 日均線的幣種比例"),
                             (50, "market_pct_above_ma50", "站上 50 日均線的幣種比例")):
        try:
            ma = px.rolling(window).mean()
            series = share(px > ma, ma.notna()).dropna()
            v = float(series.iloc[-1])
            add(id, name, series, f"{v:.0f}%（{int(ma.iloc[-1].notna().sum())} 個幣）",
                "沒有通行的固定門檻，不給方向。比例很低代表普遍走弱，很高代表普遍走強。",
                f"固定清單裡收盤價高於自己 {window} 日均線的比例，用來看行情是全面性的，還是只有少數幣在撐。")
        except Exception:
            continue
    try:
        r30, b30 = px / px.shift(30) - 1, btc / btc.shift(30) - 1
        for days, r, b in ((30, r30, b30), (90, r90, b90)):
            rel = ((1 + r.mean(axis=1)) / (1 + b) - 1).where(r.notna().sum(axis=1) >= MIN_COINS).dropna() * 100
            v = float(rel.iloc[-1])
            add(f"market_alt_vs_btc_{days}d", f"山寨幣平均相對比特幣的表現（{days} 天）", rel,
                f"{v:+.1f}%（{'山寨幣較強' if v > 0 else '比特幣較強'}）",
                "沒有通行的固定門檻，不給方向。",
                f"固定清單等權平均的 {days} 天漲幅，相對比特幣同期間漲幅的差距。正值代表資金往風險較高的幣移動。")
    except Exception:
        pass
    try:
        hi = px >= px.rolling(30).max()
        lo = px <= px.rolling(30).min()
        net = ((hi & listed).sum(axis=1) - (lo & listed).sum(axis=1)).astype(float).iloc[30:]
        v = float(net.iloc[-1])
        rows.append(common.indicator(
            id="market_new_high_low_30d", group=GROUP, asset="MARKET", name="創 30 天新高減創新低的幣數",
            value=v, unit="個", display=f"{v:+.0f} 個（新高 {int((hi & listed).iloc[-1].sum())}、新低 {int((lo & listed).iloc[-1].sum())}）",
            asof=asof, source=NAME, source_url=URL, evidence="C", reading="none",
            rule="沒有通行的固定門檻，不給方向。", note="固定清單裡當天收盤創 30 天新高的幣數，減去創 30 天新低的幣數。",
            history=_hist(net, 0)))
    except Exception:
        pass
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    out = collect()
    common.save("breadth", out)
    for row in out:
        print(row["id"], row["asof"], row["display"] or row["error"], f"[歷史 {len(row['history'])} 筆]")
