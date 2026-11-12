"""Price collector (group "price"): BTC, ETH, SOL spot price, trend and cycle indicators.

Sources (all public, no key): Binance spot klines (api.binance.com, then data-api.binance.vision), Kraken OHLC as
fallback, CoinGecko /global for bitcoin dominance. See dashboard/SPEC2.md for the contract.
Histories are about four years of daily points; directional readings use closed daily candles only.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard import common

GROUP = "price"
ASSETS = ("BTC", "ETH", "SOL")
CN = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "索拉納"}
BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
BINANCE_NAME = "幣安現貨公開日線"
KRAKEN_NAME = "Kraken 現貨公開日線（備援）"
KRAKEN_PAIR = {"BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD", "ETHBTC": "ETHXBT"}
DAY_MS = 86_400_000


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _binance(path: str, params: dict):
    """Try both Binance public spot hosts; raise the last error if both fail."""
    last = None
    for host in BINANCE_HOSTS:
        try:
            return common.get_json(host + path, params, timeout=15)
        except Exception as exc:  # blocked or down: try the next host
            last = exc
    raise last


def _binance_klines(symbol: str, interval: str, pages: int) -> pd.DataFrame:
    rows, end = [], None
    for _ in range(pages):
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end is not None:
            params["endTime"] = end
        page = _binance("/api/v3/klines", params)
        if not page:
            break
        rows = page + rows
        end = page[0][0] - 1
        if len(page) < 1000:
            break
    frame = pd.DataFrame([[r[0], float(r[2]), float(r[4])] for r in rows], columns=["ms", "high", "close"])
    return frame.drop_duplicates("ms").sort_values("ms").reset_index(drop=True)


def _kraken_klines(symbol: str, minutes: int) -> pd.DataFrame:
    reply = common.get_json("https://api.kraken.com/0/public/OHLC",
                            {"pair": KRAKEN_PAIR[symbol], "interval": minutes}, timeout=15)
    if reply.get("error"):
        raise RuntimeError(f"Kraken 回報錯誤：{reply['error']}")
    key = next(k for k in reply["result"] if k != "last")
    return pd.DataFrame([[r[0] * 1000, float(r[2]), float(r[4])] for r in reply["result"][key]],
                        columns=["ms", "high", "close"])


def _load(symbol: str, pages: int = 1) -> dict:
    """Daily and weekly candles for one pair, Binance first and Kraken as fallback."""
    try:
        daily = _binance_klines(symbol, "1d", pages)
        weekly = _binance_klines(symbol, "1w", 1)
        source = BINANCE_NAME
        pair = symbol[:-4] + "_USDT" if symbol.endswith("USDT") else "ETH_BTC"
        url = f"https://www.binance.com/zh-TC/trade/{pair}"
    except Exception as first:
        try:
            daily = _kraken_klines(symbol, 1440)
            weekly = _kraken_klines(symbol, 10080)
            source = KRAKEN_NAME
            url = "https://www.kraken.com/prices"
        except Exception as second:
            raise RuntimeError(f"幣安失敗（{first!r}）；Kraken 也失敗（{second!r}）") from second
    if daily.empty:
        raise RuntimeError("日線資料是空的")
    close = pd.Series(daily["close"].values, index=[_day(m) for m in daily["ms"]])
    # the last daily candle is still open, so its "close" is the live price
    asof = min(daily["ms"].iloc[-1] + DAY_MS, datetime.now(timezone.utc).timestamp() * 1000)
    return {"close": close, "daily": daily, "weekly": weekly, "source": source, "url": url, "asof": _iso(asof)}


def _hist(series: pd.Series, digits: int = 4) -> list:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna().tail(common.HISTORY_POINTS)
    return [[d, round(float(v), digits)] for d, v in clean.items()]


def _need(series: pd.Series, n: int, what: str) -> None:
    if len(series) < n:
        raise RuntimeError(f"日線只有 {len(series)} 筆，算{what}至少要 {n} 筆")


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's relative strength index, 0-100."""
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down)


def _money(v: float) -> str:
    return f"{v:,.0f} 美元" if v >= 1000 else f"{v:,.2f} 美元"


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _add(rows: list, *, id: str, asset: str, name: str, source: str, url: str, build) -> None:
    """Run one indicator builder inside its own try/except so one failure never drags others down."""
    try:
        fields = build()
        fields.setdefault("source", source)
        fields.setdefault("source_url", url)
        rows.append(common.indicator(id=id, group=GROUP, asset=asset, name=name, **fields))
    except Exception as exc:
        rows.append(common.failed(id=id, group=GROUP, asset=asset, name=name, source=source, source_url=url,
                                  error=f"{type(exc).__name__}: {exc}"))


def _asset_rows(rows: list, asset: str, data: dict | None, error: Exception | None) -> None:
    a, cn = asset.lower(), CN[asset]
    source = data["source"] if data else BINANCE_NAME
    url = data["url"] if data else f"https://www.binance.com/zh-TC/trade/{asset}_USDT"
    ticker_cache: dict = {}

    def get() -> dict:
        if data is None:
            raise RuntimeError(f"抓不到日線：{error}")
        return data

    def change(days: int) -> float:
        close = get()["close"]
        _need(close, days + 1, f" {days} 天漲跌幅")
        return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100

    def day_change() -> tuple:
        """True rolling 24-hour change from the Binance ticker; fall back to change since yesterday's close."""
        if "v" not in ticker_cache:
            try:
                t = _binance("/api/v3/ticker/24hr", {"symbol": f"{asset}USDT"})
                ticker_cache["v"] = (float(t["priceChangePercent"]), "幣安現貨 24 小時行情", "滾動 24 小時的漲跌幅。")
            except Exception:
                ticker_cache["v"] = None
        if ticker_cache["v"] is None:
            return change(1), get()["source"], "抓不到滾動 24 小時行情，改用「相對昨天世界協調時間 0 點收盤」的漲跌幅。"
        return ticker_cache["v"]

    def price() -> dict:
        d = get()
        last = float(d["close"].iloc[-1])
        parts = []
        for label, fn in (("24 小時", lambda: day_change()[0]), ("7 天", lambda: change(7)),
                          ("30 天", lambda: change(30))):
            try:
                parts.append(f"{label} {_pct(fn())}")
            except Exception:
                parts.append(f"{label} 無資料")
        return dict(value=last, unit="美元", display=f"{_money(last)}（{'／'.join(parts)}）", asof=d["asof"],
                    evidence="C", reading="none", rule="",
                    note=f"{cn}對美元穩定幣（泰達幣）的現貨成交價，只是報價，本身不代表方向。",
                    history=_hist(d["close"], 2))

    def change_24h() -> dict:
        v, src, note = day_change()
        return dict(value=round(v, 3), unit="%", display=_pct(v), asof=common.now_utc(), evidence="C",
                    reading="none", rule="", note=note + "短期漲跌雜訊很大，僅供參考。", source=src)

    def change_n(days: int):
        def build() -> dict:
            d = get()
            v = change(days)
            hist = (d["close"] / d["close"].shift(days) - 1) * 100
            return dict(value=round(v, 3), unit="%", display=_pct(v), asof=d["asof"], evidence="C", reading="none",
                        rule="", note=f"現價相對 {days} 天前日線收盤的漲跌幅，只是描述最近走勢，不是預測。",
                        history=_hist(hist, 2))
        return build

    def vs_ma200() -> dict:
        d = get()
        close = d["close"].iloc[:-1]  # closed daily candles only
        _need(close, 200, " 200 日均線")
        ma = close.rolling(200).mean()
        gap = (close / ma - 1) * 100
        v = float(gap.iloc[-1])
        return dict(value=round(v, 3), unit="%", asof=f"{close.index[-1]}T23:59:59Z", evidence="C",
                    display=f"{'高於' if v >= 0 else '低於'} 200 日均線 {abs(v):.1f}%（均線 {_money(float(ma.iloc[-1]))}）",
                    reading="bullish" if v > 0 else "bearish",
                    rule="最近一根已收盤日線高於 200 日均線＝偏多，低於＝偏空。這是市場通用的慣例門檻，沒有驗證過。",
                    note="200 日均線是過去 200 天收盤價的平均，常被當成長期趨勢的分界；在盤整時會反覆給出假訊號。",
                    history=_hist(gap, 2))

    def drawdown() -> dict:
        d = get()
        highs = pd.concat([d["weekly"][["ms", "high"]], d["daily"][["ms", "high"]]], ignore_index=True)
        best = highs.loc[highs["high"].idxmax()]
        top, top_day = float(best["high"]), _day(best["ms"])
        last = float(d["close"].iloc[-1])
        v = (last / top - 1) * 100
        first = _day(highs["ms"].min())
        # history: each day against the highest price seen up to that day (not against today's record)
        early = float(d["weekly"]["high"][d["weekly"]["ms"] < d["daily"]["ms"].iloc[0]].max())
        running = pd.Series(d["daily"]["high"].values, index=d["close"].index).cummax()
        if early == early:  # weekly candles reach further back than the daily ones (fallback source)
            running = running.clip(lower=early)
        return dict(value=round(v, 3), unit="%", asof=d["asof"], evidence="C", reading="none", rule="",
                    display=f"{v:.1f}%（最高價 {_money(top)}，{top_day[:4]} 年 {int(top_day[5:7])} 月）",
                    note=f"現價比歷史最高價低多少；走勢圖裡每一天都是和「到那天為止」的最高價相比。"
                         f"最高價取自這個來源從 {first} 起的資料，更早的價格不在內；跌得深不代表便宜，也可能繼續跌。",
                    history=_hist((d["close"] / running - 1) * 100, 2))

    def volatility() -> dict:
        d = get()
        closed = d["close"].iloc[:-1]  # drop the still-open candle
        _need(closed, 31, " 30 日波動率")
        vol = np.log(closed / closed.shift(1)).rolling(30).std() * np.sqrt(365) * 100
        v = float(vol.iloc[-1])
        return dict(value=round(v, 3), unit="%", display=f"年化 {v:.1f}%", asof=f"{closed.index[-1]}T23:59:59Z",
                    evidence="C", reading="none", rule="",
                    note="用最近 30 根已收盤日線的每日漲跌幅算出的波動程度，再換算成一年（乘以 365 的平方根）。"
                         "只代表價格晃動多大，不分漲跌方向。",
                    history=_hist(vol, 2))

    def rsi(weekly: bool):
        def build() -> dict:
            d = get()
            close = d["close"].iloc[:-1]  # closed daily candles only
            step = 7 if weekly else 1
            _need(close, 15 * step + 1, " RSI")
            if weekly:  # one weekly series per weekday phase, merged, so there is a value for every day
                series = pd.concat([_rsi(close.iloc[k::7]) for k in range(7)]).sort_index().dropna()
            else:
                series = _rsi(close).dropna()
            v = float(series.iloc[-1])
            label = "超買" if v >= 70 else "超賣" if v <= 30 else "中間區"
            unit = "週" if weekly else "日"
            extra = ("這裡的「週」是每 7 天取一次收盤，每天都往回算一次，所以每天都有值，和看盤軟體固定週日收盤的週線會差一點。"
                     if weekly else "日線的超買超賣很常出現，強勢行情裡可以連續好幾週都在 70 以上。")
            return dict(value=round(v, 2), unit="", display=f"{v:.1f}（{label}；70 以上超買、30 以下超賣）",
                        asof=f"{close.index[-1]}T23:59:59Z", evidence="C", reading="none",
                        rule="70 以上通稱超買、30 以下通稱超賣。這是通行慣例，本頁沒有驗證過，所以只標示狀態、不給方向。",
                        note=f"相對強弱指標（RSI）：最近 14 {unit}裡上漲幅度占全部漲跌幅度的比重，0 到 100。"
                             f"數值高代表最近漲得多又急，數值低代表跌得多又急。{extra}",
                        history=_hist(series, 2))
        return build

    for suffix, name, build in (
            ("rsi_14d", f"{cn}日線 RSI（相對強弱指標，14 天）", rsi(False)),
            ("rsi_14w", f"{cn}週線 RSI（相對強弱指標，14 週）", rsi(True)),
            ("price", f"{cn}現價", price),
            ("change_24h", f"{cn} 24 小時漲跌幅", change_24h),
            ("change_7d", f"{cn} 7 天漲跌幅", change_n(7)),
            ("change_30d", f"{cn} 30 天漲跌幅", change_n(30)),
            ("vs_ma200", f"{cn}相對 200 日均線", vs_ma200),
            ("ath_drawdown", f"{cn}距歷史最高價的跌幅", drawdown),
            ("realized_vol_30d", f"{cn} 30 日已實現波動率（年化）", volatility)):
        _add(rows, id=f"{a}_{suffix}", asset=asset, name=name, source=source, url=url, build=build)


def _btc_rows(rows: list, data: dict | None, error: Exception | None) -> None:
    source = data["source"] if data else BINANCE_NAME
    url = data["url"] if data else "https://www.binance.com/zh-TC/trade/BTC_USDT"

    def get() -> dict:
        if data is None:
            raise RuntimeError(f"抓不到日線：{error}")
        return data

    def cross() -> dict:
        d = get()
        close = d["close"].iloc[:-1]  # closed daily candles only
        _need(close, 200, " 200 日均線")
        ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
        ratio = ((ma50 / ma200 - 1) * 100).dropna()
        v = float(ratio.iloc[-1])
        golden = v > 0
        sign = ratio > 0
        flips = sign[sign != sign.shift(1)]
        since = f"，{flips.index[-1]} 起" if len(flips) > 1 else ""
        return dict(value=round(v, 3), unit="%", asof=f"{close.index[-1]}T23:59:59Z", evidence="C",
                    display=f"{'黃金交叉' if golden else '死亡交叉'}狀態{since}（50 日均線比 200 日均線"
                            f"{'高' if golden else '低'} {abs(v):.1f}%）",
                    reading="bullish" if golden else "bearish",
                    rule="50 日均線在 200 日均線之上（黃金交叉）＝偏多，在之下（死亡交叉）＝偏空。市場通用慣例，沒有驗證過。",
                    note="比較短期（50 日）與長期（200 日）平均價。訊號出現得很慢，常在行情走了一大段之後才翻轉。",
                    history=_hist(ratio, 2))

    def week200() -> dict:
        d = get()
        close = d["close"].iloc[:-1]  # closed daily candles only
        _need(close, 1400, " 200 週均線（1400 天）")
        ma = close.rolling(1400).mean()
        mult = close / ma
        v = float(mult.iloc[-1])
        return dict(value=round(v, 4), unit="倍", asof=f"{close.index[-1]}T23:59:59Z", evidence="C", reading="none",
                    display=f"{v:.2f} 倍（200 週均線約 {_money(float(ma.iloc[-1]))}）",
                    rule="沒有事前固定的門檻，不給方向。常見說法是歷次熊市低點多在 1 倍附近，僅供參考。",
                    note="現價除以 200 週均線（用最近 1400 天日線收盤的平均來算）。倍數越高代表離長期平均價越遠；"
                         "歷史上只有幾次週期可以對照。",
                    history=_hist(mult, 4))

    def pi_cycle() -> dict:
        d = get()
        close = d["close"].iloc[:-1]  # closed daily candles only
        _need(close, 350, " 350 日均線")
        ma111, ma350x2 = close.rolling(111).mean(), close.rolling(350).mean() * 2
        ratio = ma111 / ma350x2
        v = float(ratio.iloc[-1])
        hit = v >= 1
        return dict(value=round(v, 4), unit="倍", asof=f"{close.index[-1]}T23:59:59Z", evidence="A",
                    display=f"{v:.3f}（{'已觸發' if hit else '未觸發'}；111 日均線 {_money(float(ma111.iloc[-1]))}，"
                            f"2 倍 350 日均線 {_money(float(ma350x2.iloc[-1]))}）",
                    reading="bearish" if hit else "neutral",
                    rule="用已收盤日線計算：111 日均線 ÷（2 × 350 日均線）≥ 1 視為達到門檻＝偏空，否則中性。"
                         "這是「週期保險」門檻之一：證據很弱（歷史上只有少數幾次），只當警示，不是驗證過的策略。",
                    note="Pi 週期頂部指標：短期均線追上兩倍的長期均線時，過去幾次都很接近週期高點。"
                         "歷史上只觸發過少數幾次，樣本很少；沒觸發不代表安全。",
                    history=_hist(ratio, 4))

    for suffix, name, build in (
            ("ma50_ma200_cross", "比特幣 50 日與 200 日均線（黃金交叉／死亡交叉）", cross),
            ("ma200w_multiple", "比特幣 200 週均線倍數", week200),
            ("pi_cycle_top", "比特幣 Pi 週期頂部指標", pi_cycle)):
        _add(rows, id=f"btc_{suffix}", asset="BTC", name=name, source=source, url=url, build=build)


def _eth_btc(rows: list) -> None:
    cache: dict = {}

    def load() -> dict:
        if "d" not in cache:
            try:
                cache["d"] = (_load("ETHBTC", pages=2), None)
                d = cache["d"][0]
                print(f"  ETHBTC 日線 {len(d['close'])} 筆，最新 {d['close'].index[-1]}（{d['source']}）")
            except Exception as exc:
                cache["d"] = (None, exc)
        d, error = cache["d"]
        if error is not None:
            raise error
        return d

    def vs_ma200() -> dict:
        d = load()
        close = d["close"].iloc[:-1]  # closed daily candles only
        _need(close, 200, " 200 日均線")
        ma = close.rolling(200).mean()
        gap = (close / ma - 1) * 100
        v = float(gap.iloc[-1])
        return dict(value=round(v, 3), unit="%", asof=f"{close.index[-1]}T23:59:59Z", evidence="C", reading="none",
                    display=f"{'高於' if v >= 0 else '低於'} 200 日均線 {abs(v):.1f}%（比值 {float(close.iloc[-1]):.5f}，"
                            f"均線 {float(ma.iloc[-1]):.5f}）",
                    rule="沒有事前固定的門檻，不給方向。",
                    note="以太幣／比特幣比值（一顆以太幣值多少顆比特幣）比自己的 200 日均線高或低多少，用已收盤日線計算。"
                         "高於均線代表以太幣這段時間比比特幣強；它不代表整體市場會漲或跌。",
                    history=_hist(gap, 2), source=d["source"], source_url=d["url"])

    _add(rows, id="eth_btc_vs_ma200", asset="ETH", name="以太幣／比特幣比值相對 200 日均線", source=BINANCE_NAME,
         url="https://www.binance.com/zh-TC/trade/ETH_BTC", build=vs_ma200)

    def build() -> dict:
        d = load()
        close = d["close"]
        _need(close, 200, " 200 日均線")
        ma = close.rolling(200).mean()
        v, m = float(close.iloc[-1]), float(ma.iloc[-1])
        gap = (v / m - 1) * 100
        return dict(value=round(v, 6), unit="比特幣", asof=d["asof"], evidence="C", reading="none",
                    display=f"{v:.5f}（200 日均線 {m:.5f}，{'高於' if gap >= 0 else '低於'}均線 {abs(gap):.1f}%）",
                    rule="沒有事前固定的門檻，不給方向。",
                    note="一顆以太幣值多少顆比特幣。比值上升代表以太幣表現比比特幣強，常被用來觀察資金是否往其他幣種擴散；"
                         "它不代表整體市場會漲或跌。",
                    history=_hist(close, 6), source=d["source"], source_url=d["url"])
    _add(rows, id="eth_btc_ratio", asset="ETH", name="以太幣／比特幣比值與 200 日均線", source=BINANCE_NAME,
         url="https://www.binance.com/zh-TC/trade/ETH_BTC", build=build)


def _dominance(rows: list) -> None:
    def build() -> dict:
        g = common.get_json("https://api.coingecko.com/api/v3/global", timeout=15)["data"]
        v = float(g["market_cap_percentage"]["btc"])
        eth = float(g["market_cap_percentage"].get("eth", float("nan")))
        return dict(value=round(v, 3), unit="%", asof=_iso(g["updated_at"] * 1000), evidence="C", reading="none",
                    display=f"{v:.1f}%（以太幣 {eth:.1f}%）", rule="沒有事前固定的門檻，不給方向。",
                    note="比特幣市值占全部加密貨幣總市值的比例。占比上升通常表示資金偏向比特幣、對小幣較保守；"
                         "免費來源沒有歷史數列，走勢圖是本頁每次更新時自己記下來的，所以一開始很短。",
                    history=common.accumulate("btc_dominance", [[common.today_utc(), round(v, 3)]]))
    _add(rows, id="btc_dominance", asset="MARKET", name="比特幣市值占比", source="CoinGecko 全球市場公開資料",
         url="https://www.coingecko.com/zh-tw/global-charts", build=build)


def collect() -> list[dict]:
    rows: list[dict] = []
    loaded: dict[str, tuple] = {}
    for asset in ASSETS:
        try:
            data = _load(f"{asset}USDT", pages=4)  # 4 x 1000 daily candles covers the whole listing history
            loaded[asset] = (data, None)
            print(f"  {asset} 日線 {len(data['close'])} 筆，最新 {data['close'].index[-1]}（{data['source']}）")
        except Exception as exc:
            loaded[asset] = (None, exc)
            print(f"  {asset} 日線抓取失敗：{exc}")
    for asset in ASSETS:
        _asset_rows(rows, asset, *loaded[asset])
        if asset == "BTC":
            _btc_rows(rows, *loaded[asset])
    _eth_btc(rows)
    _dominance(rows)
    return rows


if __name__ == "__main__":
    rows = collect()
    common.save("price", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"])
