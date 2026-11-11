"""資金流向收集器（group="flows"）：「主力進出」的免費替代資料。規格見 dashboard/SPEC2.md；歷史抓約 4 年。

來源（全部免費、公開、不需金鑰）：
- DefiLlama：穩定幣供給（stablecoins.llama.fi）、DeFi 鎖倉總值（api.llama.fi）。
  DefiLlama 的 ETF 端點（etfs.llama.fi）要付費方案（回傳 402），所以不用。
- Farside Investors 公開表格頁：美國現貨比特幣／以太幣 ETF 每日淨流入。
- 美國商品期貨交易委員會（CFTC）公開資料 API：CME 比特幣期貨持倉報告（TFF，只含期貨）。
- CoinGecko 公開 API：全市場總市值。
- Coinbase Exchange 與幣安的公開報價端點：Coinbase 溢價。
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from dashboard import common

GROUP = "flows"
NO_RULE = "沒有事前固定的門檻，不給方向。"
LLAMA_STABLE = "https://stablecoins.llama.fi/stablecoincharts/all"
LLAMA_STABLE_PAGE = "https://defillama.com/stablecoins"
FARSIDE = {"BTC": ("https://farside.co.uk/bitcoin-etf-flow-all-data/", "比特幣"),
           "ETH": ("https://farside.co.uk/ethereum-etf-flow-all-data/", "以太幣")}
CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CFTC_PAGE = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
CFTC_MARKET = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
CFTC_CONTRACT_BTC = 5  # CME 標準比特幣期貨一口是 5 顆


def _ts_day(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), timezone.utc).strftime("%Y-%m-%d")


def _asof(day: str) -> str:
    return f"{day}T00:00:00Z"


def _history(series: pd.Series, digits: int = 2) -> list:
    series = series.dropna()
    return [[str(day), round(float(value), digits)] for day, value in series.items()]


def _change(series: pd.Series, days: int) -> float | None:
    if len(series) <= days or not series.iloc[-1 - days]:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-1 - days]) - 1) * 100


def _pct(value: float | None) -> str:
    return "無資料" if value is None else f"{value:+.1f}%"


def _yi(usd: float) -> str:
    return f"{usd / 1e8:,.1f} 億美元"


# ---------- 穩定幣（DefiLlama） ----------

_stable_cache: dict = {}


def _stable_series(stablecoin: int | None) -> pd.Series:
    """每日流通量（美元，約 4 年）。stablecoin=None 是全部穩定幣合計，1 是 USDT，2 是 USDC。同一次執行只抓一次。"""
    if stablecoin not in _stable_cache:
        _stable_cache[stablecoin] = _stable_fetch(stablecoin)
    return _stable_cache[stablecoin]


def _stable_fetch(stablecoin: int | None) -> pd.Series:
    reply = common.get_json(LLAMA_STABLE, {"stablecoin": stablecoin} if stablecoin else None, timeout=40)
    points = {}
    for point in reply[-(common.HISTORY_POINTS + 40):]:
        total = sum(float(v) for v in (point.get("totalCirculatingUSD") or {}).values() if v is not None)
        if total > 0:
            points[_ts_day(point["date"])] = total
    series = pd.Series(points).sort_index()
    if series.empty:
        raise ValueError("DefiLlama 沒有回傳穩定幣資料")
    return series


def _stablecoin(id: str, name: str, stablecoin: int | None, what: str) -> dict:
    series = _stable_series(stablecoin)
    value = float(series.iloc[-1])
    change7, change30 = _change(series, 7), _change(series, 30)
    return common.indicator(
        id=id, group=GROUP, asset="MARKET", name=name, value=round(value), unit="美元",
        display=f"{_yi(value)}（7 天 {_pct(change7)}、30 天 {_pct(change30)}）", asof=_asof(series.index[-1]),
        source="DefiLlama", source_url=LLAMA_STABLE_PAGE, evidence="C", reading="none", rule=NO_RULE,
        note=(f"{what}穩定幣是價值釘住美元等法定貨幣的代幣，可以想成停在幣圈裡的現金；供給增加代表有新錢換成代幣進場，"
              "減少代表有人贖回離場。它代表的是整個幣圈的資金水位，不是哪一個大戶。"
              "限制：錢進來不一定拿去買比特幣，也可能只是放著領利息；資料大約每天更新一次，會晚幾小時到一天。"),
        history=_history(series, 0))


def _stable_change() -> dict:
    series = _stable_series(None)
    change = ((series / series.shift(30) - 1) * 100).dropna()
    if change.empty:
        raise ValueError("穩定幣供給資料不足 30 天")
    value = float(change.iloc[-1])
    delta = float(series.iloc[-1] - series.iloc[-31])
    return common.indicator(
        id="market_stablecoin_supply_change_30d", group=GROUP, asset="MARKET", name="穩定幣總供給 30 天變化",
        value=round(value, 3), unit="%",
        display=f"{value:+.2f}%（30 天{'增加' if delta >= 0 else '減少'} {_yi(abs(delta))}）",
        asof=_asof(change.index[-1]), source="DefiLlama（自行換算）", source_url=LLAMA_STABLE_PAGE, evidence="C",
        reading="none", rule="正數是供給增加，負數是減少。" + NO_RULE,
        note=("全部穩定幣的流通量和 30 天前相比變化多少。增加得快代表有新錢換成代幣進到幣圈，減少代表資金贖回離場；"
              "它是整個幣圈的資金水位，不是哪一個大戶。限制：錢進來不一定拿去買幣，也可能只是放著領利息。"),
        history=_history(change, 3))


# ---------- 美國現貨 ETF（Farside 公開表格頁） ----------

def _number(text) -> float | None:
    """Farside 表格的數字：括號是負數，「-」是沒有資料。"""
    text = str(text).strip().replace(",", "").replace("*", "")
    if not text or text in ("-", "nan"):
        return None
    negative = text.startswith("(") and text.endswith(")")
    try:
        value = float(text.strip("()"))
    except ValueError:
        return None
    return -value if negative else value


def _etf_table(asset: str) -> tuple[pd.Series, str]:
    """回傳（每日合計淨流入，單位百萬美元；索引 YYYY-MM-DD）與頁面網址。被擋就直接丟例外，不繞過。"""
    url = FARSIDE[asset][0]
    reply = requests.get(url, headers=common.HEADERS, timeout=40)
    if reply.status_code in (403, 429, 503):
        raise RuntimeError(f"Farside 回應 HTTP {reply.status_code}，可能是機器人檢查；依規定不繞過")
    reply.raise_for_status()
    tables = pd.read_html(io.StringIO(reply.text))
    best = None
    for table in tables:
        if table.shape[1] < 3:
            continue
        days = pd.to_datetime(table.iloc[:, 0].astype(str).str.strip(), format="%d %b %Y", errors="coerce")
        if days.notna().sum() > 5 and (best is None or days.notna().sum() > best[0].notna().sum()):
            best = (days, table)
    if best is None:
        raise RuntimeError("Farside 頁面裡找不到每日流量表格（頁面格式改了，或被機器人檢查頁取代）")
    days, table = best
    last = table.columns[-1]
    if "total" not in " ".join(map(str, last if isinstance(last, tuple) else (last,))).lower():
        raise RuntimeError("Farside 表格最後一欄不是合計（Total），格式改了")
    total = table.iloc[:, -1].map(_number)
    series = pd.Series(total.values, index=days.dt.strftime("%Y-%m-%d").values)
    series = series[days.notna().values].dropna().sort_index()
    if series.empty:
        raise RuntimeError("Farside 表格沒有可用的合計數字")
    return series.astype(float), url


def _flow_words(million: float) -> str:
    return f"{'淨流入' if million >= 0 else '淨流出'} {abs(million) / 100:,.2f} 億美元"


def _etf(asset: str) -> list[dict]:
    daily, url = _etf_table(asset)
    label = FARSIDE[asset][1]
    key = asset.lower()
    who = (f"這是美國現貨{label} ETF（在股票市場買賣、背後真的持有{label}的基金）全部檔數加總的資金進出，"
           "代表用券商帳戶買幣的美國散戶、投資顧問與機構，是目前最接近「法人買盤」的免費資料。")
    limit = ("限制：Farside 是彙整各發行商自己公布的數字，美國收盤後幾小時到隔天才齊全，最新一天可能還會修正；"
             "只有美股交易日有資料；ETF 流入也可能是套利部位（同時放空期貨），不等於單純看多。")
    rows = []
    sums = {1: daily, 5: daily.rolling(5).sum(), 30: daily.rolling(30).sum()}
    names = {1: "單日淨流入", 5: "近 5 個交易日淨流入合計", 30: "近 30 個交易日淨流入合計"}
    ids = {1: "daily", 5: "5d", 30: "30d"}
    for window, series in sums.items():
        series = series.dropna()
        if series.empty:
            raise RuntimeError(f"Farside 資料不足 {window} 個交易日")
        value = float(series.iloc[-1])
        rows.append(common.indicator(
            id=f"{key}_etf_flow_{ids[window]}", group=GROUP, asset=asset, name=f"美國現貨{label} ETF {names[window]}",
            value=round(value, 1), unit="百萬美元", display=f"{_flow_words(value)}（截至 {series.index[-1]}）",
            asof=_asof(series.index[-1]), source="Farside Investors", source_url=url, evidence="C", reading="none",
            rule="正數是淨流入，負數是淨流出。" + NO_RULE, note=who + limit, history=_history(series, 1)))
    return rows


# ---------- CME 比特幣期貨持倉（CFTC） ----------

def _cftc() -> list[dict]:
    reply = common.get_json(CFTC_URL, {"$where": f"market_and_exchange_names = '{CFTC_MARKET}'",
                                       "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 215})  # weekly reports, about four years
    if len(reply) < 2:
        raise ValueError("CFTC 沒有回傳足夠的持倉報告")
    frame = pd.DataFrame(reply)
    frame.index = frame["report_date_as_yyyy_mm_dd"].str[:10]
    frame = frame[~frame.index.duplicated()].sort_index()
    number = lambda column: pd.to_numeric(frame[column], errors="coerce")  # noqa: E731
    asset_mgr = number("asset_mgr_positions_long") - number("asset_mgr_positions_short")
    leveraged = number("lev_money_positions_long") - number("lev_money_positions_short")
    open_interest = number("open_interest_all")
    timing = ("這是每週二收盤的部位，美國商品期貨交易委員會（CFTC）在週五公布，所以看到時已經晚 3 天以上。"
              "只算芝加哥商品交易所（CME）的標準比特幣期貨（一口 5 顆），不含微型合約與選擇權。")

    def row(id: str, name: str, series: pd.Series, note: str, signed: bool) -> dict:
        series = series.dropna()
        value, previous = float(series.iloc[-1]), float(series.iloc[-2])
        delta = value - previous
        if signed:
            side = "淨多" if value >= 0 else "淨空"
            text = f"{side} {abs(value):,.0f} 口（約 {abs(value) * CFTC_CONTRACT_BTC:,.0f} 顆）"
        else:
            text = f"{value:,.0f} 口（約 {value * CFTC_CONTRACT_BTC:,.0f} 顆）"
        return common.indicator(
            id=id, group=GROUP, asset="BTC", name=name, value=round(value), unit="口",
            display=f"{text}，比前一週 {delta:+,.0f} 口（{series.index[-1]} 週二）", asof=_asof(series.index[-1]),
            source="美國商品期貨交易委員會（CFTC）持倉報告", source_url=CFTC_PAGE, evidence="C", reading="none",
            rule=("正數是淨多單，負數是淨空單。" if signed else "") + NO_RULE, note=note + timing,
            history=_history(series, 0))

    return [
        row("btc_cme_asset_manager_net", "CME 比特幣期貨：資產管理機構淨部位", asset_mgr,
            "資產管理機構指退休基金、共同基金、保險公司這類長線法人，他們的多單減空單。淨多增加常被解讀為長線法人加碼。"
            "限制：很多法人現在改用現貨 ETF，期貨部位只是一部分。", True),
        row("btc_cme_leveraged_funds_net", "CME 比特幣期貨：槓桿基金淨部位", leveraged,
            "槓桿基金主要是對沖基金。他們常常「買現貨或 ETF、同時放空期貨」賺價差，"
            "所以淨空單很大通常是套利，不代表看跌。限制：單看這個數字無法分辨套利與真的看空。", True),
        row("btc_cme_open_interest", "CME 比特幣期貨：未平倉量", open_interest,
            "未平倉量是還沒結清的合約總數，代表美國法人在期貨市場的參與規模，不分多空。"
            "限制：量變大只代表參與者變多，不代表方向。", False),
    ]


# ---------- 全市場總市值（CoinGecko）與 DeFi 鎖倉（DefiLlama） ----------

def _total_market_cap() -> dict:
    data = common.get_json("https://api.coingecko.com/api/v3/global")["data"]
    value = float(data["total_market_cap"]["usd"])
    change = data.get("market_cap_change_percentage_24h_usd")
    share = (data.get("market_cap_percentage") or {}).get("btc")
    extra = "" if share is None else f"，比特幣占 {float(share):.1f}%"
    updated = data.get("updated_at")
    asof = datetime.fromtimestamp(updated, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if updated else common.now_utc()
    return common.indicator(
        id="market_total_market_cap", group=GROUP, asset="MARKET", name="加密貨幣全市場總市值", value=round(value),
        unit="美元", display=f"{value / 1e12:,.2f} 兆美元（24 小時 {_pct(None if change is None else float(change))}{extra}）",
        asof=asof, source="CoinGecko", source_url="https://www.coingecko.com/zh-tw/global-charts", evidence="C",
        reading="none", rule=NO_RULE,
        note=("所有加密貨幣的價格乘以流通量加總，代表整個幣圈的規模，看的是所有持有人，不是特定主力。"
              "限制：小幣的價格與流通量常不可靠，總數會高估；免費端點只有當下數字，"
              "走勢圖是本頁每次更新時自己記下來的；約每 10 分鐘更新。"),
        history=common.accumulate("market_total_market_cap", [[asof[:10], round(value)]]))


def _defi_tvl() -> dict:
    reply = common.get_json("https://api.llama.fi/v2/historicalChainTvl", timeout=40)
    series = pd.Series({_ts_day(point["date"]): float(point["tvl"])
                        for point in reply[-(common.HISTORY_POINTS + 40):]}).sort_index()
    if series.empty:
        raise ValueError("DefiLlama 沒有回傳鎖倉資料")
    value = float(series.iloc[-1])
    change7, change30 = _change(series, 7), _change(series, 30)
    return common.indicator(
        id="market_defi_tvl", group=GROUP, asset="MARKET", name="DeFi 鎖倉總值（所有公鏈合計）", value=round(value),
        unit="美元", display=f"{_yi(value)}（7 天 {_pct(change7)}、30 天 {_pct(change30)}）",
        asof=_asof(series.index[-1]), source="DefiLlama", source_url="https://defillama.com/", evidence="C",
        reading="none", rule=NO_RULE,
        note=("DeFi 是「去中心化金融」，指不經過公司、直接用鏈上程式做借貸與交易；鎖倉總值是存在這些程式裡的資產總額，"
              "代表鏈上使用者願意放多少錢進去。限制：以美元計價，幣價漲跌就會讓數字跟著動，不一定是真的有錢進出；"
              "每天更新一次。"),
        history=_history(series, 0))


# ---------- Coinbase 溢價 ----------

def _coinbase_premium() -> dict:
    coinbase = float(common.get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker")["price"])
    binance = float(common.get_json("https://api.binance.com/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])
    asof = common.now_utc()
    value = (coinbase / binance - 1) * 100
    history = []
    try:  # 歷史走勢用兩邊的日線收盤價算；抓不到也不影響當下數值
        candles = common.get_json("https://api.exchange.coinbase.com/products/BTC-USD/candles", {"granularity": 86400})
        klines = common.get_json("https://api.binance.com/api/v3/klines",
                                 {"symbol": "BTCUSDT", "interval": "1d", "limit": 400})
        left = pd.Series({_ts_day(c[0]): float(c[4]) for c in candles})
        right = pd.Series({_ts_day(k[0] / 1000): float(k[4]) for k in klines})
        both = pd.concat([left, right], axis=1, keys=["coinbase", "binance"]).dropna().sort_index()
        both = both[both.index < asof[:10]]  # 今天這根還沒收完，不放進歷史
        history = _history((both["coinbase"] / both["binance"] - 1) * 100, 4)
    except Exception:  # noqa: BLE001
        history = []
    return common.indicator(
        id="btc_coinbase_premium", group=GROUP, asset="BTC", name="Coinbase 溢價（對幣安的價差）", value=round(value, 4),
        unit="%", display=f"{value:+.3f}%（Coinbase {coinbase:,.0f} 美元、幣安 {binance:,.0f} 泰達幣）", asof=asof,
        source="Coinbase Exchange 與幣安公開報價", source_url="https://www.coinbase.com/price/bitcoin", evidence="C",
        reading="none", rule="正數是 Coinbase 比較貴。" + NO_RULE,
        note=("Coinbase 是美國人與美國機構最常用的交易所，幣安以美國以外的用戶為主。Coinbase 比較貴常被當成美國買盤較強的訊號，"
              "代表的是美國現貨買家，即時資料幾乎沒有延遲。限制：幣安是用泰達幣（USDT）報價，"
              "泰達幣本身對美元有小幅偏離，會混進這個價差；兩邊報價相差不到一秒但不是同一瞬間；歷史走勢用每日收盤價計算。"),
        history=history)


# ---------- 組裝 ----------

def _run(rows: list, build, failures: list[dict]) -> None:
    """每個指標各自 try/except；失敗時用 failures 裡的說明留空位。"""
    try:
        result = build()
        rows.extend(result if isinstance(result, list) else [result])
    except Exception as error:  # noqa: BLE001 一個來源壞掉不能拖垮其他指標
        for meta in failures:
            rows.append(common.failed(group=GROUP, error=f"{type(error).__name__}: {error}", **meta))


def collect() -> list[dict]:
    _stable_cache.clear()
    rows: list[dict] = []
    _run(rows, _stable_change, [dict(id="market_stablecoin_supply_change_30d", asset="MARKET",
                                     name="穩定幣總供給 30 天變化", source="DefiLlama",
                                     source_url=LLAMA_STABLE_PAGE)])
    stables = (("market_stablecoin_supply", "穩定幣總供給", None, "全部穩定幣加總。"),
               ("market_usdt_supply", "泰達幣（USDT）供給", 1, "泰達幣（USDT）是最大的美元穩定幣，主要在美國以外的交易所使用。"),
               ("market_usdc_supply", "USDC 供給", 2, "USDC 是美國公司 Circle 發行的美元穩定幣，美國機構用得比較多。"))
    for id, name, stablecoin, what in stables:
        _run(rows, lambda id=id, name=name, stablecoin=stablecoin, what=what: _stablecoin(id, name, stablecoin, what),
             [dict(id=id, asset="MARKET", name=name, source="DefiLlama", source_url=LLAMA_STABLE_PAGE)])
    for asset, (url, label) in FARSIDE.items():
        key = asset.lower()
        _run(rows, lambda asset=asset: _etf(asset),
             [dict(id=f"{key}_etf_flow_{suffix}", asset=asset, name=f"美國現貨{label} ETF {name}",
                   source="Farside Investors", source_url=url)
              for suffix, name in (("daily", "單日淨流入"), ("5d", "近 5 個交易日淨流入合計"),
                                   ("30d", "近 30 個交易日淨流入合計"))])
    _run(rows, _cftc,
         [dict(id=id, asset="BTC", name=name, source="美國商品期貨交易委員會（CFTC）持倉報告", source_url=CFTC_PAGE)
          for id, name in (("btc_cme_asset_manager_net", "CME 比特幣期貨：資產管理機構淨部位"),
                           ("btc_cme_leveraged_funds_net", "CME 比特幣期貨：槓桿基金淨部位"),
                           ("btc_cme_open_interest", "CME 比特幣期貨：未平倉量"))])
    _run(rows, _total_market_cap, [dict(id="market_total_market_cap", asset="MARKET", name="加密貨幣全市場總市值",
                                        source="CoinGecko", source_url="https://www.coingecko.com/zh-tw/global-charts")])
    _run(rows, _defi_tvl, [dict(id="market_defi_tvl", asset="MARKET", name="DeFi 鎖倉總值（所有公鏈合計）",
                                source="DefiLlama", source_url="https://defillama.com/")])
    _run(rows, _coinbase_premium, [dict(id="btc_coinbase_premium", asset="BTC", name="Coinbase 溢價（對幣安的價差）",
                                        source="Coinbase Exchange 與幣安公開報價",
                                        source_url="https://www.coinbase.com/price/bitcoin")])
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = collect()
    common.save("flows", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
