"""Classic cycle models for bitcoin (group "cycle"): rainbow chart, power law, AHR999, 2-year MA multiplier,
golden ratio multiplier, bull market support band, stock-to-flow.

Sources (public, no key): blockchain.info charts API for the daily price since 2010 and the coin supply, Binance
spot daily closes (through the price collector) for 2017 onward. Everything uses closed daily candles.

These are widely quoted community models, not validated strategies. The rainbow and power-law lines are fitted
here on the whole history (hindsight), so the bands differ slightly from other websites and move a little as new
data arrives. No model here gives a direction: every reading is "none".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import onchain2 as oc
from dashboard.collectors import price as px

GROUP = "cycle"
GENESIS = "2009-01-03"
BCI_PRICE = "https://api.blockchain.info/charts/market-price"
SOURCE = "blockchain.info 每日價格（2010 年起）＋幣安現貨日線"
PAGE = "https://www.blockchain.com/explorer/charts/market-price"
RAINBOW_PAGE = "https://www.blockchaincenter.net/bitcoin-rainbow-chart/"
BANDS_SINCE = "2013-01-01"  # the band edges span the lowest and highest deviation seen since this day
BAND_NAMES = ("跳樓大拍賣", "買進", "累積", "還算便宜", "續抱", "是泡沫嗎？", "追高情緒升溫", "該賣了", "極度泡沫")
NO_RULE = "沒有事前固定的門檻，不給方向。"
HINDSIGHT = "模型線是用整段歷史事後擬合的，每次更新會微幅移動；只有四次週期可以對照，不能當成預測。"
_memo: dict = {}


def _long_close() -> pd.Series:
    """Daily closes since 2010 (closed candles only), index YYYY-MM-DD. Binance replaces blockchain.info from 2017."""
    if "close" not in _memo:
        reply = common.get_json(BCI_PRICE, {"timespan": "all", "format": "json", "sampled": "false"}, timeout=60)
        # blockchain.info stamps each price at the UTC midnight it was taken, i.e. the close of the day before;
        # relabel it by the day the candle covers, the Binance convention used everywhere else (checked against
        # Bitstamp daily bars: day-by-day return correlation 0.99 after this shift, ~0 without it). Fixed 2026-11-19.
        old = pd.Series({(datetime.fromtimestamp(p["x"], timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"):
                         float(p["y"]) for p in reply["values"] if p["y"] > 0}).sort_index()
        if len(old) < 3000:
            raise RuntimeError(f"blockchain.info 長期價格只有 {len(old)} 筆")
        today = common.today_utc()
        old = old[old.index < today]
        try:
            recent = px._load("BTCUSDT", pages=4)["close"].iloc[:-1]  # drop the candle that is still open
            close = pd.concat([old[old.index < recent.index[0]], recent])
        except Exception:  # noqa: BLE001 - blockchain.info alone still covers the whole history
            close = old
        _memo["close"] = close[~close.index.duplicated(keep="last")].sort_index()
    return _memo["close"]


def _age(index) -> np.ndarray:
    return (pd.to_datetime(index) - pd.Timestamp(GENESIS)).days.values.astype(float)


def _fit() -> dict:
    """log10(price) = a + b * log10(days since the first block), least squares on the whole history."""
    if "fit" not in _memo:
        close = _long_close()
        x, y = np.log10(_age(close.index)), np.log10(close.values)
        b, a = np.polyfit(x, y, 1)
        model = pd.Series(10 ** (a + b * x), index=close.index)
        resid = pd.Series(y - (a + b * x), index=close.index)
        since = resid[resid.index >= BANDS_SINCE]
        lo, hi = float(since.min()), float(since.max())
        _memo["fit"] = {"a": float(a), "b": float(b), "model": model, "resid": resid, "lo": lo, "hi": hi,
                        "edges": [lo + (hi - lo) * i / 9 for i in range(10)]}
    return _memo["fit"]


def _asof(close: pd.Series) -> str:
    return f"{close.index[-1]}T23:59:59Z"


def _money(v: float) -> str:
    return f"{v:,.0f} 美元"


def _add(rows: list, *, id: str, name: str, build, source: str = SOURCE, url: str = PAGE) -> None:
    try:
        fields = build()
        fields.setdefault("source", source)
        fields.setdefault("source_url", url)
        rows.append(common.indicator(id=id, group=GROUP, asset="BTC", name=name, evidence="C", reading="none",
                                     **fields))
    except Exception as exc:  # noqa: BLE001 - one model failing never drags the others down
        rows.append(common.failed(id=id, group=GROUP, asset="BTC", name=name, source=source, source_url=url,
                                  error=f"{type(exc).__name__}: {exc}"))


def _rainbow() -> dict:
    close, fit = _long_close(), _fit()
    pos = (fit["resid"] - fit["lo"]) / (fit["hi"] - fit["lo"]) * 100
    v = float(pos.iloc[-1])
    band = int(min(8, max(0, v // (100 / 9))))
    model = float(fit["model"].iloc[-1])
    low, high = model * 10 ** fit["edges"][band], model * 10 ** fit["edges"][band + 1]
    return dict(value=round(v, 2), unit="%", asof=_asof(close),
                display=f"由下往上第 {band + 1} 帶（共 9 帶）「{BAND_NAMES[band]}」；這一帶今天的價格範圍 "
                        f"{low:,.0f}–{high:,.0f} 美元，收盤 {_money(float(close.iloc[-1]))}",
                rule=NO_RULE + "色帶名稱是原圖作者取的玩笑話，不是本頁的建議。",
                note="彩虹圖：把比特幣的長期價格在對數座標上擬合成一條成長曲線，再往上下各畫幾條平行的色帶，"
                     "看現價落在哪一帶。數值是現價在整道彩虹裡的位置（0 是最底、100 是最頂）。"
                     f"色帶上下緣取 {BANDS_SINCE[:4]} 年以來離曲線最遠的低點與高點，等分成 9 帶；"
                     "這是本頁自己擬合的版本，色帶位置和原版網站會有些出入。" + HINDSIGHT,
                history=px._hist(pos, 2), source_url=RAINBOW_PAGE)


def _power_law() -> dict:
    close, fit = _long_close(), _fit()
    ratio = close / fit["model"]
    v, model = float(ratio.iloc[-1]), float(fit["model"].iloc[-1])
    floor = float(10 ** fit["lo"])
    return dict(value=round(v, 4), unit="倍", asof=_asof(close),
                display=f"{v:.2f} 倍（模型價 {_money(model)}；{BANDS_SINCE[:4]} 年以來最低到過 {floor:.2f} 倍，"
                        f"換算今天約 {_money(model * floor)}）",
                rule=NO_RULE,
                note="冪律模型：假設比特幣價格大致跟「從創世區塊算起的天數」的某個次方成正比"
                     f"（本頁擬合出來約 {fit['b']:.2f} 次方）。倍數是現價除以模型價，1 倍代表剛好在長期趨勢線上。"
                     "和彩虹圖用的是同一條擬合線，只是換一種方式表達。" + HINDSIGHT,
                history=px._hist(ratio, 4))


def _ahr999() -> dict:
    close = _long_close()
    px._need(close, 200, " AHR999（200 天）")
    cost = np.exp(np.log(close).rolling(200).mean())  # 200-day geometric mean: the cost of buying a little daily
    fitted = pd.Series(10 ** (5.84 * np.log10(_age(close.index)) - 17.01), index=close.index)
    ahr = (close / cost) * (close / fitted)
    v = float(ahr.iloc[-1])
    zone = "低於 0.45，原作者稱為「抄底區」" if v < 0.45 else \
        "在 0.45 到 1.2 之間，原作者稱為「定投區」" if v < 1.2 else "高於 1.2，原作者認為已不便宜"
    recent = ahr.tail(common.HISTORY_POINTS)
    return dict(value=round(v, 4), unit="", asof=_asof(close),
                display=f"{v:.2f}（{zone}；200 日定投成本 {_money(float(cost.iloc[-1]))}，"
                        f"原作者的擬合價 {_money(float(fitted.iloc[-1]))}）",
                rule="原作者（微博用戶 ahr999）的分界：低於 0.45、0.45 到 1.2、高於 1.2。" + oc.UNVERIFIED + "不給方向。",
                note="AHR999 指數（中文圈常說的九神指數）：（現價 ÷ 近 200 天每天定額買進的平均成本）×（現價 ÷ 依幣齡"
                     "擬合的價格）。兩個比值都小，代表現價同時低於近期成本和長期趨勢。"
                     f"最近約四年裡，低於 0.45 的有 {(recent < 0.45).sum()} 天，高於 1.2 的有 {(recent >= 1.2).sum()} 天。"
                     "擬合價的公式是 2018 年定下來的，近兩次週期價格都低於它，所以這個指數近年偏低。",
                history=px._hist(ahr, 4), source_url="https://www.coinglass.com/zh-TW/pro/i/ahr999")


def _ma2y() -> dict:
    close = _long_close()
    px._need(close, 730, " 2 年均線（730 天）")
    ma = close.rolling(730).mean()
    mult = close / ma
    v, m = float(mult.iloc[-1]), float(ma.iloc[-1])
    return dict(value=round(v, 4), unit="倍", asof=_asof(close),
                display=f"{v:.2f} 倍（2 年均線 {_money(m)}，5 倍線 {_money(m * 5)}）",
                rule="原作者的說法：跌破 2 年均線是歷史上的低檔區，漲過 5 倍線是高檔區。" + oc.UNVERIFIED + "不給方向。",
                note="2 年均線乘數：現價除以過去 730 天的平均價。歷次熊市低點都在 1 倍以下，"
                     "2013 與 2017 年高點碰過 5 倍線，2021 年高點沒有碰到。",
                history=px._hist(mult, 4))


def _golden() -> dict:
    close = _long_close()
    px._need(close, 350, " 350 日均線")
    ma = close.rolling(350).mean()
    mult = close / ma
    v, m = float(mult.iloc[-1]), float(ma.iloc[-1])
    return dict(value=round(v, 4), unit="倍", asof=_asof(close),
                display=f"{v:.2f} 倍（350 日均線 {_money(m)}；1.6 倍 {_money(m * 1.6)}、2 倍 {_money(m * 2)}、"
                        f"3 倍 {_money(m * 3)}）",
                rule=NO_RULE + "常見說法是 1.6 倍、2 倍、3 倍是上漲途中的壓力位，僅供參考。",
                note="黃金比例乘數：現價除以 350 日均線。Pi 週期頂部指標用的也是這條均線的 2 倍。"
                     "低於 1 倍代表價格在近一年平均價之下。",
                history=px._hist(mult, 4))


def _support_band() -> dict:
    close = _long_close()
    days = pd.Series(close.values, index=pd.to_datetime(close.index))
    weekly = days.resample("W-SUN").last().dropna()
    if weekly.index[-1] > days.index[-1]:
        weekly = weekly.iloc[:-1]  # the current week has not closed yet
    px._need(weekly, 30, "牛市支撐帶（至少 30 週）")
    sma, ema = weekly.rolling(20).mean(), weekly.ewm(span=21, adjust=False, min_periods=21).mean()
    lower, upper = np.minimum(sma, ema), np.maximum(sma, ema)
    gap = ((weekly / ((sma + ema) / 2) - 1) * 100).dropna()
    last = float(close.iloc[-1])
    lo, hi = float(lower.iloc[-1]), float(upper.iloc[-1])
    where = "在支撐帶之上" if last > hi else "在支撐帶之下" if last < lo else "在支撐帶裡面"
    v = (last / ((lo + hi) / 2) - 1) * 100
    history = [[d.strftime("%Y-%m-%d"), round(float(g), 2)] for d, g in gap.items()]
    return dict(value=round(v, 3), unit="%", asof=_asof(close),
                display=f"{where}（{'高' if v >= 0 else '低'}於帶中央 {abs(v):.1f}%；20 週均線 "
                        f"{_money(float(sma.iloc[-1]))}，21 週指數均線 {_money(float(ema.iloc[-1]))}）",
                rule=NO_RULE + "常見說法是牛市期間價格多半守在這條帶之上，跌破後常變成壓力，僅供參考。",
                note="牛市支撐帶：20 週簡單均線和 21 週指數均線之間的區域，用已收盤的週線計算。"
                     "走勢圖是每週一個點，數值是週收盤比帶中央高或低多少。",
                history=history)


def _stock_to_flow() -> dict:
    close = _long_close()
    supply = oc._bci("total-bitcoins")
    flow = supply - supply.shift(365)
    sf = (supply / flow).replace([np.inf, -np.inf], np.nan).dropna()
    model = np.exp(-1.84) * sf ** 3.36
    ratio = (close.reindex(model.index) / model).dropna()
    if ratio.empty:
        raise RuntimeError("供給量和價格的日期對不起來")
    day = ratio.index[-1]
    v = float(ratio.iloc[-1])
    return dict(value=round(v, 4), unit="倍", asof=oc._asof(day),
                display=f"{v:.2f} 倍（模型價 {_money(float(model[day]))}；存量 {supply[day] / 1e4:,.0f} 萬顆，"
                        f"近一年新增 {flow[day] / 1e4:,.1f} 萬顆，存量流量比 {float(sf[day]):.0f}）",
                rule=NO_RULE + "這個模型在 2021 年之後已經大幅失準，放在這裡只是因為它很有名。",
                note="存量流量比模型（S2F，PlanB 2019 年版）：用「現有的幣 ÷ 一年新挖出的幣」估價格，"
                     "公式是 exp(−1.84) × 比值的 3.36 次方。每次減半比值就翻倍，模型價跟著跳升十倍左右，"
                     "但實際價格遠遠沒有跟上，倍數越小代表偏離越大。",
                history=px._hist(ratio, 4), source="blockchain.info 供給量與每日價格",
                source_url="https://www.blockchain.com/explorer/charts/total-bitcoins")


def _write_chart() -> None:
    """Data for the rainbow chart section: fit, band edges and a weekly price line."""
    path = common.CACHE / "_classic.json"
    try:
        close, fit = _long_close(), _fit()
        line = close[close.index >= "2011-01-01"]
        weekly = line.iloc[::7]
        if weekly.index[-1] != line.index[-1]:
            weekly = pd.concat([weekly, line.iloc[-1:]])
        pos = (float(fit["resid"].iloc[-1]) - fit["lo"]) / (fit["hi"] - fit["lo"])
        band = int(min(8, max(0, pos * 9 // 1)))
        model = float(fit["model"].iloc[-1])
        payload = {"collected_utc": common.now_utc(),
                   "rainbow": {"genesis": GENESIS, "a": fit["a"], "b": fit["b"],
                               "edges": [round(e, 5) for e in fit["edges"]], "bands": list(BAND_NAMES),
                               "bands_since": BANDS_SINCE, "fit_from": close.index[0], "asof": close.index[-1],
                               "close": float(close.iloc[-1]), "band_now": band, "model_now": round(model, 2),
                               "ranges_now": [[round(model * 10 ** fit["edges"][i], 0),
                                               round(model * 10 ** fit["edges"][i + 1], 0)] for i in range(9)],
                               "price": [[d, round(float(v), 2)] for d, v in weekly.items()]}}
        common.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - the chart section is optional; keep the last good file
        print(f"  彩虹圖資料沒有更新：{type(exc).__name__}: {exc}")


def collect() -> list[dict]:
    rows: list[dict] = []
    for id, name, build in (
            ("btc_rainbow_position", "比特幣彩虹圖位置", _rainbow),
            ("btc_power_law_ratio", "比特幣冪律模型倍數", _power_law),
            ("btc_ahr999", "比特幣 AHR999 指數（九神指數）", _ahr999),
            ("btc_ma2y_multiple", "比特幣 2 年均線乘數", _ma2y),
            ("btc_golden_ratio_multiple", "比特幣黃金比例乘數（350 日均線倍數）", _golden),
            ("btc_bull_support_band", "比特幣牛市支撐帶（20 週均線／21 週指數均線）", _support_band),
            ("btc_stock_to_flow", "比特幣存量流量比模型（S2F）偏離倍數", _stock_to_flow)):
        _add(rows, id=id, name=name, build=build)
    _write_chart()
    return rows


if __name__ == "__main__":
    rows = collect()
    common.save("classic", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"])
