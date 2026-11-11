"""總體經濟與傳統市場收集器（group="macro"）。規格見 dashboard/SPEC2.md；歷史抓約 4 到 5 年。

來源（全部免費、公開、不需金鑰，固定 User-Agent）：
- 美國財政部官網：每日公債殖利率曲線（名目與實質）。
- DBnomics（法國公益經濟資料庫，轉載聯準會理事會的公開統計）：廣義美元指數（H.10）、聯準會總資產（H.4.1）、M2（H.6）。
- 紐約聯邦準備銀行公開 API：逆回購每日餘額。美國財政部 Fiscal Data API：財政部帳戶每日餘額。
- 芝加哥選擇權交易所（Cboe）公開檔：VIX、標普 500 指數每日收盤。Nasdaq 官網公開 API：那斯達克 100 指數每日收盤。
- 幣安現貨公開日線：比特幣（BTCUSDT）與黃金代幣 PAXG（PAXGUSDT，一枚對應一盎司實體黃金）。
- 聯準會官網行事曆（calendar.json）：下一次利率決策會議日期。
- 聖路易聯準銀行 FRED：只用來抓高收益債利差（別處沒有免費來源）。實測 FRED 對本專案的固定 User-Agent 不回應
  （連線掛住到逾時），依規定不偽裝、不繞過，所以這個指標抓不到時就記為失敗。
- stooq.com 現在要先通過瀏覽器 JavaScript 驗證，依規定不繞過，所以沒有使用。
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common

GROUP = "macro"
NO_RULE = "沒有事前固定的門檻，不給方向。"
START = (datetime.now(timezone.utc) - timedelta(days=1900)).strftime("%Y-%m-%d")  # 多抓一些給 200 日均線暖機
TIMEOUT = 25
CORR_WINDOW = 63  # 約 90 個日曆天的美股交易日數

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all")
TREASURY_PAGE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
DBNOMICS_URL = "https://api.db.nomics.world/v22/series/FED/{code}"
DBNOMICS_PAGE = "https://db.nomics.world/FED/{code}"
NYFED_RRP = "https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json"
TGA_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CBOE_SPX = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_SPX.json"
NASDAQ_URL = "https://api.nasdaq.com/api/quote/NDX/historical"
BINANCE_KLINES = ("https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines")
FED_CALENDAR = "https://www.federalreserve.gov/json/calendar.json"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_cache: dict = {}


def _once(key: str, build):
    """同一次執行每個來源只抓一次；失敗也記住，讓用到它的指標各自報同一個錯，不重抓。"""
    if key not in _cache:
        try:
            _cache[key] = (build(), None)
        except Exception as error:  # noqa: BLE001
            _cache[key] = (None, error)
    value, error = _cache[key]
    if error is not None:
        raise error
    return value


def _get(url: str, params: dict | None = None, timeout=TIMEOUT) -> requests.Response:
    reply = requests.get(url, params=params, headers=common.HEADERS, timeout=timeout)
    reply.raise_for_status()
    return reply


def _csv(url: str, params: dict | None = None, timeout=TIMEOUT) -> pd.DataFrame:
    text = _get(url, params, timeout).content.decode("utf-8-sig", errors="replace")
    if text.lstrip()[:1] == "<":
        raise RuntimeError("來源回傳網頁而不是 CSV（可能是機器人檢查）；依規定不繞過")
    return pd.read_csv(io.StringIO(text))


def _clip(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= START]


def _history(series: pd.Series, digits: int = 2) -> list:
    return [[str(day)[:10], round(float(value), digits)] for day, value in series.dropna().items()]


def _asof(day) -> str:
    return f"{str(day)[:10]}T00:00:00Z"


def _ago(series: pd.Series, days: int) -> pd.Series:
    """每個日期往回 days 個日曆天時（當天沒有資料就取更早最近一筆）的數值；索引是 YYYY-MM-DD 字串。"""
    stamps = pd.to_datetime(series.index)
    dated = pd.Series(series.values, index=stamps)
    return pd.Series(dated.asof(stamps - pd.Timedelta(days=days)).values, index=series.index)


def _signed(value: float | None, digits: int = 1, unit: str = "%") -> str:
    return "無資料" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}{unit}"


# ---------- 原始資料 ----------

def _treasury(kind: str) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        frames = []
        for year in range(int(START[:4]), datetime.now(timezone.utc).year + 1):
            frame = _csv(TREASURY_URL.format(year=year),
                         {"type": kind, "field_tdr_date_value": str(year), "page": "", "_format": "csv"})
            if "Date" in frame.columns and len(frame):
                frames.append(frame)
        if not frames:
            raise ValueError("美國財政部沒有回傳殖利率資料")
        table = pd.concat(frames)
        table.index = pd.to_datetime(table["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
        table.columns = [str(column).strip().upper() for column in table.columns]
        return table.sort_index()
    return _once(f"treasury:{kind}", build)


def _nominal(column: str) -> pd.Series:
    return _clip(_treasury("daily_treasury_yield_curve")[column])


def _real10() -> pd.Series:
    return _clip(_treasury("daily_treasury_real_yield_curve")["10 YR"])


def _dbnomics(code: str) -> pd.Series:
    def build() -> pd.Series:
        doc = common.get_json(DBNOMICS_URL.format(code=code), {"observations": "1"}, timeout=TIMEOUT)["series"]["docs"][0]
        days = [period if len(period) == 10 else f"{period}-01" for period in doc["period"]]
        series = pd.to_numeric(pd.Series(doc["value"], index=days), errors="coerce").dropna().sort_index()
        if series.empty:
            raise ValueError(f"DBnomics 沒有回傳 {code} 的資料")
        return series
    return _once(f"dbnomics:{code}", build)


def _dollar() -> pd.Series:
    return _clip(_dbnomics("H10/JRXWTFB_N.B"))


def _rrp() -> pd.Series:
    """聯準會隔夜逆回購每日承作金額（美元）。"""
    def build() -> pd.Series:
        reply = common.get_json(NYFED_RRP, {"startDate": START, "endDate": common.today_utc()}, timeout=TIMEOUT)
        points = {op["operationDate"]: float(op.get("totalAmtAccepted") or 0) for op in reply["repo"]["operations"]}
        if not points:
            raise ValueError("紐約聯準銀行沒有回傳逆回購資料")
        return pd.Series(points).sort_index()
    return _once("rrp", build)


def _tga() -> pd.Series:
    """財政部在聯準會的帳戶（TGA）每日收盤餘額（百萬美元）。欄位名稱在 2022 年改過，兩種都接。"""
    def build() -> pd.Series:
        reply = common.get_json(TGA_URL, {"filter": f"record_date:gte:{START}", "sort": "record_date",
                                          "page[size]": 10000,
                                          "fields": "record_date,account_type,close_today_bal,open_today_bal"},
                                timeout=TIMEOUT)
        points = {}
        for row in reply["data"]:
            kind = row["account_type"]
            if kind == "Treasury General Account (TGA) Closing Balance":
                raw = row["open_today_bal"]
            elif kind in ("Federal Reserve Account", "Treasury General Account (TGA)"):
                raw = row["close_today_bal"]
            else:
                continue
            try:
                points[row["record_date"]] = float(raw)
            except (TypeError, ValueError):
                continue
        if not points:
            raise ValueError("美國財政部沒有回傳帳戶餘額")
        return pd.Series(points).sort_index()
    return _once("tga", build)


def _net_liquidity() -> pd.DataFrame:
    """每週三：聯準會總資產 − 逆回購 − 財政部帳戶，單位兆美元。"""
    def build() -> pd.DataFrame:
        assets = _clip(_dbnomics("H41/RESPPA_N.WW"))  # 百萬美元
        stamps = pd.to_datetime(assets.index)

        def on_wednesdays(series: pd.Series) -> np.ndarray:
            return pd.Series(series.values, index=pd.to_datetime(series.index)).asof(stamps).values

        frame = pd.DataFrame({"assets": assets.values / 1e6, "rrp": on_wednesdays(_rrp()) / 1e12,
                              "tga": on_wednesdays(_tga()) / 1e6}, index=assets.index).dropna()
        frame["net"] = frame["assets"] - frame["rrp"] - frame["tga"]
        if frame.empty:
            raise ValueError("淨流動性的三個來源對不上日期")
        return frame
    return _once("net_liquidity", build)


def _vix() -> pd.Series:
    def build() -> pd.Series:
        table = _csv(CBOE_VIX)
        return pd.Series(table["CLOSE"].values,
                         index=pd.to_datetime(table["DATE"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d"))
    return _clip(_once("vix", build))


def _sp500() -> pd.Series:
    def build() -> pd.Series:
        rows = common.get_json(CBOE_SPX, timeout=TIMEOUT)["data"]
        return pd.Series({row["date"]: float(row["close"]) for row in rows if float(row["close"]) > 0})
    return _clip(_once("sp500", build))


def _nasdaq() -> pd.Series:
    def build() -> pd.Series:
        reply = common.get_json(NASDAQ_URL, {"assetclass": "index", "fromdate": START, "todate": common.today_utc(),
                                             "limit": "9999"}, timeout=TIMEOUT)
        rows = ((reply.get("data") or {}).get("tradesTable") or {}).get("rows") or []
        points = {datetime.strptime(row["date"], "%m/%d/%Y").strftime("%Y-%m-%d"): float(row["close"].replace(",", ""))
                  for row in rows}
        if not points:
            raise ValueError("Nasdaq 沒有回傳指數資料")
        return pd.Series(points)
    return _clip(_once("nasdaq", build))


def _binance_close(symbol: str) -> pd.Series:
    """幣安現貨日線收盤價（最新一根是還沒走完的當日），抓兩頁共約 2000 天。"""
    def build() -> pd.Series:
        last_error: Exception | None = None
        for base in BINANCE_KLINES:
            try:
                points: dict = {}
                end = None
                for _ in range(2):
                    params = {"symbol": symbol, "interval": "1d", "limit": 1000}
                    if end is not None:
                        params["endTime"] = end
                    page = common.get_json(base, params, timeout=TIMEOUT)
                    if not page:
                        break
                    for candle in page:
                        day = datetime.fromtimestamp(candle[0] / 1000, timezone.utc).strftime("%Y-%m-%d")
                        points[day] = float(candle[4])
                    end = page[0][0] - 1
                    if len(page) < 1000:
                        break
                if points:
                    return pd.Series(points).sort_index()
            except Exception as error:  # noqa: BLE001 換下一個幣安公開網域
                last_error = error
        raise last_error or ValueError(f"幣安沒有回傳 {symbol} 的日線")
    return _clip(_once(f"binance:{symbol}", build))


# ---------- 指標 ----------

def _dollar_index() -> list[dict]:
    series = _dollar()
    change = ((series / _ago(series, 30) - 1) * 100).dropna()
    value, move = float(series.iloc[-1]), float(change.iloc[-1])
    page = DBNOMICS_PAGE.format(code="H10/JRXWTFB_N.B")
    why = ("比特幣和多數加密貨幣用美元計價，全球資金也多半用美元借貸；美元走強通常代表全球資金變緊、大家不想冒險，"
           "加密貨幣容易承壓，美元走弱時則相反。")
    limit = "限制：這是歷史上常見的關係，不是每一段都成立；聯準會每週才公布一次，數字會晚幾天到一週。"
    return [
        common.indicator(
            id="macro_dollar_index", group=GROUP, asset="MARKET", name="美元指數（聯準會廣義美元指數）",
            value=round(value, 3), unit="點", display=f"{value:.1f} 點（30 天 {_signed(move)}）",
            asof=_asof(series.index[-1]), source="聯準會理事會 H.10 統計（經 DBnomics 轉載）", source_url=page,
            evidence="C", reading="none", rule=NO_RULE,
            note=("美元對一籃子主要貿易夥伴貨幣的強弱，數字越高代表美元越強。" + why + limit
                  + "它和新聞常講的 DXY 美元指數（只含 6 種貨幣）走勢接近但數字不同。"),
            history=_history(series, 3)),
        common.indicator(
            id="macro_dollar_index_change_30d", group=GROUP, asset="MARKET", name="美元指數 30 天變化",
            value=round(move, 3), unit="%", display=_signed(move, 2), asof=_asof(change.index[-1]),
            source="聯準會理事會 H.10 統計（經 DBnomics 轉載，自行換算）", source_url=page, evidence="C",
            reading="none", rule="正數是美元變強，負數是變弱。" + NO_RULE,
            note="廣義美元指數和 30 天前相比漲跌多少。" + why + limit, history=_history(change, 3)),
    ]


def _yields() -> list[dict]:
    ten, two = _nominal("10 YR"), _nominal("2 YR")
    curve = (ten - two).dropna()
    value, spread = float(ten.iloc[-1]), float(curve.iloc[-1])
    month = float(ten.iloc[-1] - _ago(ten, 30).iloc[-1])
    return [
        common.indicator(
            id="macro_us10y_yield", group=GROUP, asset="MARKET", name="美國十年期公債殖利率",
            value=round(value, 2), unit="%", display=f"{value:.2f}%（30 天 {_signed(month, 2, ' 個百分點')}）",
            asof=_asof(ten.index[-1]), source="美國財政部每日殖利率曲線", source_url=TREASURY_PAGE, evidence="C",
            reading="none", rule=NO_RULE,
            note=("把錢借給美國政府十年可以拿到的年利率，是全世界資金的「無風險報酬」基準。殖利率越高，"
                  "放在公債就能穩穩領息，資金越沒有理由去買不配息、波動大的加密貨幣；殖利率下降時則相反。"
                  "限制：殖利率上升有時是因為景氣好（對風險資產不一定是壞事），要搭配實質利率一起看；只有美國交易日有資料。"),
            history=_history(ten, 2)),
        common.indicator(
            id="macro_yield_curve_10y_2y", group=GROUP, asset="MARKET", name="殖利率曲線（十年期減二年期）",
            value=round(spread, 2), unit="個百分點",
            display=f"{spread:+.2f} 個百分點（{'倒掛' if spread < 0 else '正常，長天期較高'}）",
            asof=_asof(curve.index[-1]), source="美國財政部每日殖利率曲線（自行相減）", source_url=TREASURY_PAGE,
            evidence="C", reading="none",
            rule="小於 0 稱為「倒掛」，傳統上被當成景氣衰退的預警；這是景氣訊號，不是加密貨幣的買賣門檻，所以不給方向。",
            note=("十年期公債殖利率減二年期。正常情況長天期利率較高；變成負數（倒掛）代表市場認為現在利率太緊、之後得降息，"
                  "歷史上常出現在衰退之前。和加密貨幣的關係是間接的：它反映景氣與降息預期，而降息與放水通常對加密貨幣有利、"
                  "衰退恐慌則不利。限制：從倒掛到衰退常隔一兩年，而且倒掛「結束」的那段時間反而常是市場最動盪的時候。"),
            history=_history(curve, 2)),
    ]


def _real_yield() -> dict:
    series = _real10()
    value = float(series.iloc[-1])
    month = float(series.iloc[-1] - _ago(series, 30).iloc[-1])
    return common.indicator(
        id="macro_us10y_real_yield", group=GROUP, asset="MARKET", name="美國十年期實質利率（抗通膨公債殖利率）",
        value=round(value, 2), unit="%", display=f"{value:.2f}%（30 天 {_signed(month, 2, ' 個百分點')}）",
        asof=_asof(series.index[-1]), source="美國財政部每日實質殖利率曲線", source_url=TREASURY_PAGE, evidence="C",
        reading="none", rule=NO_RULE,
        note=("扣掉通貨膨脹之後，把錢借給美國政府十年真正賺到的利率（取自抗通膨公債）。它是持有黃金、比特幣這類"
              "「不會生利息的資產」的機會成本：實質利率越高，抱著不生息的資產越吃虧；實質利率很低或是負的時候，"
              "黃金與比特幣歷史上表現較好（例如 2020 到 2021 年）。限制：這個關係在 2023 年之後明顯變弱，不能單獨使用。"),
        history=_history(series, 2))


def _liquidity() -> list[dict]:
    frame = _net_liquidity()
    net = frame["net"]
    last = frame.iloc[-1]
    change = ((net / net.shift(13) - 1) * 100).dropna()
    delta = float(net.iloc[-1] - net.iloc[-14]) if len(net) > 13 else None
    page = DBNOMICS_PAGE.format(code="H41/RESPPA_N.WW")
    source = "聯準會 H.4.1 統計（經 DBnomics）、紐約聯準銀行逆回購、美國財政部每日報表（自行計算）"
    why = ("聯準會的資產負債表是它印出來放進金融體系的錢；其中停在逆回購（金融機構把錢存回聯準會）和財政部帳戶"
           "（政府還沒花出去的錢）的部分並沒有在市場上流動，扣掉之後才是真正在金融體系裡的錢。"
           "這個數字在 2020 年之後和比特幣、美股的大方向常常同步：水位上升時風險資產容易漲，下降時容易跌。")
    limit = ("限制：這是市場上流行的簡化算法，不是官方指標；每週三才有一個數字，會晚幾天；"
             "逆回購餘額降到很低之後，這個指標主要只剩總資產與財政部帳戶在動，參考價值可能下降。")
    rows = [common.indicator(
        id="macro_fed_net_liquidity", group=GROUP, asset="MARKET", name="聯準會淨流動性",
        value=round(float(last["net"]), 4), unit="兆美元",
        display=(f"{last['net']:.2f} 兆美元（總資產 {last['assets']:.2f} − 逆回購 {last['rrp']:.2f} − "
                 f"財政部帳戶 {last['tga']:.2f}）"),
        asof=_asof(frame.index[-1]), source=source, source_url=page, evidence="C", reading="none", rule=NO_RULE,
        note="聯準會總資產減掉逆回購、再減掉財政部帳戶餘額，每週三一個數字。" + why + limit,
        history=_history(net, 4))]
    if change.empty or delta is None:
        raise ValueError("淨流動性歷史不足 13 週")
    move = float(change.iloc[-1])
    rows.append(common.indicator(
        id="macro_fed_net_liquidity_change_13w", group=GROUP, asset="MARKET", name="聯準會淨流動性 13 週變化",
        value=round(move, 3), unit="%",
        display=f"{move:+.2f}%（13 週{'增加' if delta >= 0 else '減少'} {abs(delta) * 1e4:,.0f} 億美元）",
        asof=_asof(change.index[-1]), source=source, source_url=page, evidence="C", reading="none",
        rule="正數是金融體系裡的錢變多，負數是變少。" + NO_RULE,
        note="聯準會淨流動性和 13 週（約一季）前相比變化多少，看的是水位的方向而不是高低。" + why + limit,
        history=_history(change, 3)))
    return rows


def _m2() -> dict:
    series = _dbnomics("H6_H6_M2/M2.M")  # 十億美元，月資料
    growth = ((series / series.shift(12) - 1) * 100).dropna()
    growth = growth[growth.index >= "2015-01-01"]
    value = float(growth.iloc[-1])
    return common.indicator(
        id="macro_m2_yoy", group=GROUP, asset="MARKET", name="美國 M2 貨幣供給年增率",
        value=round(value, 3), unit="%",
        display=f"{value:+.2f}%（{growth.index[-1][:7]} 的數字，M2 共 {float(series.iloc[-1]) / 1000:.2f} 兆美元）",
        asof=_asof(growth.index[-1]), source="聯準會理事會 H.6 統計（經 DBnomics 轉載，自行換算）",
        source_url=DBNOMICS_PAGE.format(code="H6_H6_M2/M2.M"), evidence="C", reading="none",
        rule="正數是錢變多，負數是錢變少。" + NO_RULE,
        note=("M2 是美國民眾和企業手上的現金、活存、定存與貨幣基金的總和，可以想成「社會上總共有多少錢」。"
              "年增率是和一年前相比多了百分之幾。錢增加得快的時期（例如 2020 到 2021 年）資產價格包括比特幣容易上漲，"
              "2022 年 M2 罕見地縮減，加密貨幣也大跌。限制：每月才公布一次、會晚一個多月，是很慢的背景指標，不能用來抓短期時機。"),
        history=_history(growth, 3))


def _vix_indicator() -> dict:
    series = _vix()
    value = float(series.iloc[-1])
    zone = "高於 30，市場恐慌" if value >= 30 else "低於 15，市場很平靜" if value < 15 else "介於 15 到 30 的一般範圍"
    return common.indicator(
        id="macro_vix", group=GROUP, asset="MARKET", name="VIX 恐慌指數（美股預期波動）",
        value=round(value, 2), unit="點", display=f"{value:.1f}（{zone}）", asof=_asof(series.index[-1]),
        source="芝加哥選擇權交易所（Cboe）", source_url="https://www.cboe.com/tradable_products/vix/vix_historical_data/",
        evidence="C", reading="none",
        rule="慣例上高於 30 算恐慌、低於 15 算過度平靜；這是描述美股情緒的慣例，不是加密貨幣的買賣門檻，所以不給方向。",
        note=("從標普 500 選擇權價格推算出來、市場預期未來 30 天美股會有多大的波動，俗稱恐慌指數。"
              "美股恐慌時投資人會先賣掉風險最高的資產換現金，加密貨幣常跟著大跌（例如 2020 年 3 月）；"
              "VIX 很低時代表大家願意冒險。限制：加密貨幣自己出事時（交易所倒閉等）VIX 不會有反應。"),
        history=_history(series, 2))


def _high_yield() -> dict:
    """ICE 美銀高收益債利差只有 FRED 免費提供；FRED 不回應本專案的固定 User-Agent 時就記失敗，不繞過。"""
    table = _csv(FRED_CSV, {"id": "BAMLH0A0HYM2", "cosd": START}, timeout=(5, 8))
    series = _clip(pd.Series(pd.to_numeric(table.iloc[:, 1], errors="coerce").values,
                             index=table.iloc[:, 0].astype(str).str[:10]))
    if series.empty:
        raise ValueError("FRED 沒有回傳高收益債利差")
    value = float(series.iloc[-1])
    return common.indicator(
        id="macro_high_yield_spread", group=GROUP, asset="MARKET", name="高收益債利差",
        value=round(value, 2), unit="個百分點", display=f"{value:.2f} 個百分點", asof=_asof(series.index[-1]),
        source="聖路易聯準銀行 FRED（ICE 美銀高收益債指數）", source_url="https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
        evidence="C", reading="none", rule=NO_RULE,
        note=("信用較差的公司借錢要比美國政府多付多少利息。利差擴大代表市場擔心倒帳、不願意借錢給高風險對象，"
              "這種時候加密貨幣這類高風險資產通常也被拋售；利差很窄代表市場樂觀、資金寬鬆。限制：只有美國交易日有資料。"),
        history=_history(series, 2))


def _vs_ma200(id: str, name: str, label: str, series: pd.Series, source: str, url: str) -> dict:
    ratio = ((series / series.rolling(200).mean() - 1) * 100).dropna()
    if ratio.empty:
        raise ValueError(f"{label}資料不足 200 個交易日")
    value = float(ratio.iloc[-1])
    return common.indicator(
        id=id, group=GROUP, asset="MARKET", name=name, value=round(value, 3), unit="%",
        display=f"{value:+.1f}%（指數 {float(series.iloc[-1]):,.0f} 點，在 200 日均線之{'上' if value >= 0 else '下'}）",
        asof=_asof(ratio.index[-1]), source=source, source_url=url, evidence="C", reading="none",
        rule="正數是指數在 200 日均線之上（多頭慣例），負數在之下。這是美股的趨勢慣例，不是加密貨幣的買賣門檻，所以不給方向。",
        note=(f"{label}收盤價比自己過去 200 個交易日的平均高或低多少，用來看美股的大趨勢與是否漲過頭。"
              "比特幣從 2020 年起和美國科技股常常同漲同跌，美股站穩長期均線時加密貨幣比較容易有行情，"
              "美股跌破時加密貨幣很少能獨自走強。限制：兩者的連動時強時弱（見「比特幣與那斯達克的相關係數」）；只有美國交易日有資料。"),
        history=_history(ratio, 3))


def _nasdaq_ma() -> dict:
    return _vs_ma200("macro_nasdaq_vs_ma200", "那斯達克 100 指數相對 200 日均線", "那斯達克 100 指數（美國大型科技股）",
                     _nasdaq(), "Nasdaq 官網", "https://www.nasdaq.com/market-activity/index/ndx/historical")


def _sp500_ma() -> dict:
    return _vs_ma200("macro_sp500_vs_ma200", "標普 500 指數相對 200 日均線", "標普 500 指數（美國 500 家大公司）",
                     _sp500(), "芝加哥選擇權交易所（Cboe）", "https://www.cboe.com/us/indices/dashboard/spx/")


GOLD_SOURCE = "幣安現貨公開日線（黃金代幣 PAXG）"
GOLD_URL = "https://www.binance.com/zh-TC/trade/PAXG_USDT"
GOLD_LIMIT = "限制：這裡用的是黃金代幣 PAXG（一枚對應一盎司實體黃金）在幣安的價格，和倫敦金價通常差不到 1%，但市場恐慌時偏離會變大。"


def _gold() -> dict:
    series = _binance_close("PAXGUSDT")
    value = float(series.iloc[-1])
    month = (value / float(_ago(series, 30).iloc[-1]) - 1) * 100
    year = (value / float(_ago(series, 365).iloc[-1]) - 1) * 100
    return common.indicator(
        id="macro_gold_price", group=GROUP, asset="MARKET", name="黃金價格（每盎司）",
        value=round(value, 2), unit="美元", display=f"{value:,.0f} 美元（30 天 {_signed(month)}、1 年 {_signed(year)}）",
        asof=_asof(series.index[-1]), source=GOLD_SOURCE, source_url=GOLD_URL, evidence="C", reading="none",
        rule=NO_RULE,
        note=("黃金是傳統的避險與抗通膨資產，比特幣常被稱為「數位黃金」。黃金上漲代表資金在找美元以外的保值工具，"
              "這股需求有時也會流向比特幣；但恐慌時黃金漲、比特幣跌的情況也很常見，兩者不是穩定同向。" + GOLD_LIMIT),
        history=_history(series, 2))


def _btc_gold() -> dict:
    frame = pd.DataFrame({"btc": _binance_close("BTCUSDT"), "gold": _binance_close("PAXGUSDT")}).dropna()
    ratio = frame["btc"] / frame["gold"]
    value = float(ratio.iloc[-1])
    quarter = (value / float(_ago(ratio, 90).iloc[-1]) - 1) * 100
    return common.indicator(
        id="btc_gold_ratio", group=GROUP, asset="BTC", name="比特幣／黃金比值",
        value=round(value, 3), unit="盎司", display=f"1 顆比特幣 = {value:.1f} 盎司黃金（90 天 {_signed(quarter)}）",
        asof=_asof(ratio.index[-1]), source=GOLD_SOURCE + "與比特幣日線（自行相除）", source_url=GOLD_URL,
        evidence="C", reading="none", rule=NO_RULE,
        note=("一顆比特幣可以換幾盎司黃金。用黃金而不是美元來量比特幣，可以把「美元本身變薄」的影響去掉："
              "比值上升代表比特幣跑贏黃金、資金偏好風險較高的那一邊；比值下降代表資金寧願躲在黃金。" + GOLD_LIMIT),
        history=_history(ratio, 3))


def _correlation(id: str, name: str, other: pd.Series, label: str, source: str, url: str, note: str) -> dict:
    frame = pd.DataFrame({"btc": _binance_close("BTCUSDT"), "other": other}).dropna()
    returns = frame.pct_change().dropna()
    corr = returns["btc"].rolling(CORR_WINDOW).corr(returns["other"]).dropna()
    if corr.empty:
        raise ValueError("資料不足以計算相關係數")
    value = float(corr.iloc[-1])
    strength = "幾乎無關" if abs(value) < 0.2 else ("同向" if value > 0 else "反向") + ("、明顯" if abs(value) >= 0.5 else "但不強")
    return common.indicator(
        id=id, group=GROUP, asset="BTC", name=name, value=round(value, 3), unit="相關係數",
        display=f"{value:+.2f}（{strength}）", asof=_asof(corr.index[-1]), source=source, source_url=url,
        evidence="C", reading="none",
        rule="+1 是完全同漲同跌，0 是互不相干，−1 是完全相反。相關係數只描述連動程度，不代表漲跌方向，所以不給方向。",
        note=(f"比特幣和{label}最近約 90 天（{CORR_WINDOW} 個美國交易日）每日漲跌幅的相關係數。" + note
              + "限制：相關係數會變，過去連動不代表之後也連動；比特幣取每天世界標準時間午夜的收盤價，"
              "和美國收盤時間差幾個小時；週末沒有對應資料。"),
        history=_history(corr, 3))


def _corr_nasdaq() -> dict:
    return _correlation(
        "btc_nasdaq_corr_90d", "比特幣與那斯達克的 90 天相關係數", _nasdaq(), "那斯達克 100 指數",
        "幣安現貨日線與 Nasdaq 官網（自行計算）", "https://www.nasdaq.com/market-activity/index/ndx/historical",
        "數字高代表比特幣這陣子被當成科技股一類的風險資產在交易，美股的利多利空會直接傳過來，"
        "這時上面幾個美股與利率指標比較有參考價值；數字接近 0 代表比特幣在走自己的行情。")


def _corr_dollar() -> dict:
    return _correlation(
        "btc_dollar_corr_90d", "比特幣與美元指數的 90 天相關係數", _dollar(), "聯準會廣義美元指數",
        "幣安現貨日線與聯準會 H.10 統計（經 DBnomics，自行計算）", DBNOMICS_PAGE.format(code="H10/JRXWTFB_N.B"),
        "歷史上多數時間是負的（美元漲、比特幣跌）。負得越多代表比特幣這陣子越受美元強弱左右，"
        "這時美元指數比較有參考價值；接近 0 或轉正代表這層關係暫時失效。")


def _fomc() -> dict:
    events = json.loads(_get(FED_CALENDAR).content.decode("utf-8-sig"))["events"]
    today = datetime.now(timezone.utc).date()
    meetings = []
    for event in events:
        if event.get("type") != "FOMC" or event.get("title") != "FOMC Meeting":
            continue
        numbers = re.findall(r"\d+", str(event.get("days") or event.get("day") or ""))
        if not numbers or not event.get("month"):
            continue
        try:
            day = datetime.strptime(f"{event['month']}-{int(numbers[-1]):02d}", "%Y-%m-%d").date()
        except ValueError:
            continue
        if day >= today:
            meetings.append(day)
    if not meetings:
        raise ValueError("聯準會行事曆裡找不到未來的利率決策會議")
    day = min(meetings)
    wait = (day - today).days
    when = "今天" if wait == 0 else f"還有 {wait} 天"
    return common.indicator(
        id="macro_fomc_next", group=GROUP, asset="MARKET", name="下一次聯準會利率決策",
        value=wait, unit="天", display=f"{day.isoformat()}（{when}；美東時間下午 2 點公布，約台灣時間隔天凌晨 2 到 3 點）",
        asof=common.now_utc(), source="聯準會官網行事曆",
        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", evidence="C", reading="none",
        rule="這是行事曆，不是訊號，不給方向。",
        note=("聯準會的利率決策會議（聯邦公開市場委員會，一年 8 次）決定美元利率。升息或降息、以及會後主席記者會的口風，"
              "會直接影響美元、美債與所有風險資產，加密貨幣在公布前後幾個小時常出現大波動。"
              "這裡顯示會議最後一天（公布決策的那天）。限制：行事曆只列到聯準會已公告的會議；日期偶爾會調整。"),
        history=[])


# ---------- 組裝 ----------

def _run(rows: list, build, failures: list[dict]) -> None:
    """每個指標各自 try/except；失敗時用 failures 裡的說明留空位。"""
    try:
        result = build()
        rows.extend(result if isinstance(result, list) else [result])
    except Exception as error:  # noqa: BLE001 一個來源壞掉不能拖垮其他指標
        for meta in failures:
            rows.append(common.failed(group=GROUP, error=f"{type(error).__name__}: {error}", **meta))


def _meta(id: str, name: str, source: str, url: str, asset: str = "MARKET") -> dict:
    return dict(id=id, asset=asset, name=name, source=source, source_url=url)


def collect() -> list[dict]:
    _cache.clear()
    rows: list[dict] = []
    h10 = DBNOMICS_PAGE.format(code="H10/JRXWTFB_N.B")
    h41 = DBNOMICS_PAGE.format(code="H41/RESPPA_N.WW")
    ndx = "https://www.nasdaq.com/market-activity/index/ndx/historical"
    fed = "聯準會理事會 H.10 統計（經 DBnomics）"
    trio = "聯準會、紐約聯準銀行、美國財政部"
    _run(rows, _dollar_index, [_meta("macro_dollar_index", "美元指數（聯準會廣義美元指數）", fed, h10),
                               _meta("macro_dollar_index_change_30d", "美元指數 30 天變化", fed, h10)])
    _run(rows, _yields, [_meta("macro_us10y_yield", "美國十年期公債殖利率", "美國財政部", TREASURY_PAGE),
                         _meta("macro_yield_curve_10y_2y", "殖利率曲線（十年期減二年期）", "美國財政部", TREASURY_PAGE)])
    _run(rows, _real_yield, [_meta("macro_us10y_real_yield", "美國十年期實質利率（抗通膨公債殖利率）", "美國財政部",
                                   TREASURY_PAGE)])
    _run(rows, _liquidity, [_meta("macro_fed_net_liquidity", "聯準會淨流動性", trio, h41),
                            _meta("macro_fed_net_liquidity_change_13w", "聯準會淨流動性 13 週變化", trio, h41)])
    _run(rows, _m2, [_meta("macro_m2_yoy", "美國 M2 貨幣供給年增率", "聯準會理事會 H.6 統計（經 DBnomics）",
                           DBNOMICS_PAGE.format(code="H6_H6_M2/M2.M"))])
    _run(rows, _vix_indicator, [_meta("macro_vix", "VIX 恐慌指數（美股預期波動）", "芝加哥選擇權交易所（Cboe）",
                                      "https://www.cboe.com/tradable_products/vix/vix_historical_data/")])
    _run(rows, _high_yield, [_meta("macro_high_yield_spread", "高收益債利差", "聖路易聯準銀行 FRED",
                                   "https://fred.stlouisfed.org/series/BAMLH0A0HYM2")])
    _run(rows, _nasdaq_ma, [_meta("macro_nasdaq_vs_ma200", "那斯達克 100 指數相對 200 日均線", "Nasdaq 官網", ndx)])
    _run(rows, _sp500_ma, [_meta("macro_sp500_vs_ma200", "標普 500 指數相對 200 日均線", "芝加哥選擇權交易所（Cboe）",
                                 "https://www.cboe.com/us/indices/dashboard/spx/")])
    _run(rows, _gold, [_meta("macro_gold_price", "黃金價格（每盎司）", GOLD_SOURCE, GOLD_URL)])
    _run(rows, _btc_gold, [_meta("btc_gold_ratio", "比特幣／黃金比值", GOLD_SOURCE, GOLD_URL, "BTC")])
    _run(rows, _corr_nasdaq, [_meta("btc_nasdaq_corr_90d", "比特幣與那斯達克的 90 天相關係數",
                                    "幣安現貨日線與 Nasdaq 官網", ndx, "BTC")])
    _run(rows, _corr_dollar, [_meta("btc_dollar_corr_90d", "比特幣與美元指數的 90 天相關係數",
                                    "幣安現貨日線與聯準會 H.10 統計", h10, "BTC")])
    _run(rows, _fomc, [_meta("macro_fomc_next", "下一次聯準會利率決策", "聯準會官網行事曆",
                             "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")])
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = collect()
    common.save("macro", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
