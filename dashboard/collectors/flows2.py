"""大戶與機構資金收集器（group="flows"，預測市場用 "sentiment"；存成 cache/flows2.json）。規格見 dashboard/SPEC2.md。

和 flows.py 不重複（那邊已有：ETF 每日／5 日／30 日淨流入合計、CME 持倉、穩定幣總供給與 30 天變化、Coinbase 溢價、DeFi 鎖倉）。
這裡補上：
- CoinGecko 免金鑰端點：上市公司持有的比特幣與以太幣（合計、最大一家、其餘合計）。來源只有目前數值，每天存一筆。
- Farside 公開表格：美國現貨 ETF 自上市以來的累計淨流入、各發行商近 30 個交易日淨流入。
- DefiLlama：泰達幣（USDT）與 USDC 近 7 天供給變化（鑄造減銷毀）、以太坊／波場／Solana 三條鏈上的穩定幣存量。
- CoinMetrics 社群版：礦工每日收入、普爾倍數（Puell Multiple）、每單位算力收入。
- Polymarket 公開 API：成交量最大的「今年會碰到什麼價位」市場的機率（只當情緒參考）。

全部免費、公開、不需金鑰，只用固定 User-Agent。被擋就記為失敗。
試過但拿不到：持有 1,000 顆以上的地址數與大額轉帳（CoinMetrics 社群版回 403、Blockchair 要金鑰、
BitInfoCharts 有機器人檢查）、交易所的穩定幣儲備（CoinMetrics 社群版沒有開放 USDT／USDC 的 SplyExNtv；
DefiLlama 的交易所儲備一家就 40 多 MB）、礦工持有與轉出（CoinMetrics 社群版回 403）、
bitcointreasuries.net（沒有公開 API，只有網頁）。
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from dashboard import common

GROUP = "flows"
NO_RULE = "沒有事前固定的門檻，不給方向。"
SNAPSHOT = "來源只給目前數值，走勢圖是本頁每次更新時自己記下來的，所以一開始很短。"
COINGECKO = "https://api.coingecko.com/api/v3/companies/public_treasury"
COINGECKO_PAGE = {"BTC": "https://www.coingecko.com/zh-tw/treasuries/bitcoin",
                  "ETH": "https://www.coingecko.com/zh-tw/treasuries/ethereum"}
FARSIDE = {"BTC": ("https://farside.co.uk/bitcoin-etf-flow-all-data/", "比特幣"),
           "ETH": ("https://farside.co.uk/ethereum-etf-flow-all-data/", "以太幣")}
ISSUERS = {"BTC": (("IBIT", "貝萊德 IBIT"), ("FBTC", "富達 FBTC"), ("GBTC", "灰度 GBTC")),
           "ETH": (("ETHA", "貝萊德 ETHA"), ("FETH", "富達 FETH"), ("ETHE", "灰度 ETHE"))}
ISSUER_NOTE = {
    "IBIT": "貝萊德是全球最大的資產管理公司，IBIT 是規模最大的現貨比特幣 ETF，最能代表美國大型機構與投資顧問的買賣。",
    "FBTC": "富達的客戶以美國散戶與退休帳戶為主，FBTC 是規模第二大的現貨比特幣 ETF。",
    "GBTC": "灰度 GBTC 是由舊的信託轉成的 ETF、手續費最高，上市以來長期被贖回；它流出多半是老持有人換到便宜的 ETF 或獲利了結。",
    "ETHA": "貝萊德 ETHA 是規模最大的現貨以太幣 ETF，最能代表美國大型機構與投資顧問的買賣。",
    "FETH": "富達的客戶以美國散戶與退休帳戶為主，FETH 是主要的現貨以太幣 ETF 之一。",
    "ETHE": "灰度 ETHE 是由舊的信託轉成的 ETF、手續費最高，上市以來長期被贖回。"}
LLAMA = "https://stablecoins.llama.fi/stablecoincharts"
LLAMA_PAGE = "https://defillama.com/stablecoins"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_NAME = "CoinMetrics 社群版（自行換算）"
CM_PAGE = "https://charts.coinmetrics.io/crypto-data/"
POLY = "https://gamma-api.polymarket.com/events"
POLY_NAME = "Polymarket 預測市場公開資料"
ASSET_CN = {"BTC": "比特幣", "ETH": "以太幣"}
POLY_COIN = {"BTC": ("bitcoin", "Bitcoin"), "ETH": ("ethereum", "Ethereum")}
PUELL_HOT, PUELL_COLD = 4.0, 0.5  # conventional lines from the indicator's author (David Puell, 2019)


def _ts_day(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), timezone.utc).strftime("%Y-%m-%d")


def _asof(day: str) -> str:
    return f"{day}T00:00:00Z"


def _history(series: pd.Series, digits: int = 2) -> list:
    series = series.dropna()
    return [[str(day), round(float(value), digits)] for day, value in series.items()]


def _yi(usd: float) -> str:
    return f"{usd / 1e8:,.1f} 億美元"


def _signed_yi(usd: float) -> str:
    return f"{'增加' if usd >= 0 else '減少'} {abs(usd) / 1e8:,.1f} 億美元"


def _flow_words(million: float) -> str:
    return f"{'淨流入' if million >= 0 else '淨流出'} {abs(million) / 100:,.2f} 億美元"


# ---------- 上市公司持幣（CoinGecko） ----------

def _treasury(asset: str) -> list[dict]:
    coin = {"BTC": "bitcoin", "ETH": "ethereum"}[asset]
    reply = common.get_json(f"{COINGECKO}/{coin}")
    companies = sorted(reply.get("companies") or [], key=lambda c: -float(c.get("total_holdings") or 0))
    total = float(reply.get("total_holdings") or 0)
    if not companies or total <= 0:
        raise ValueError("CoinGecko 沒有回傳上市公司持幣資料")
    top = companies[0]
    top_amount = float(top["total_holdings"])
    others = total - top_amount
    share = sum(float(c.get("percentage_of_total_supply") or 0) for c in companies)
    a, cn, today, page = asset.lower(), ASSET_CN[asset], common.today_utc(), COINGECKO_PAGE[asset]
    limit = ("限制：CoinGecko 依各公司公告與申報整理，公司買了之後幾天到幾週才會反映；只含有公開揭露的上市公司，"
             "不含 ETF、政府、未上市公司；名單增減也會讓合計跳動。" + SNAPSHOT)

    def row(suffix: str, name: str, value: float, display: str, note: str) -> dict:
        id = f"{a}_treasury_{suffix}"
        return common.indicator(
            id=id, group=GROUP, asset=asset, name=name, value=round(value, 2), unit="顆", display=display,
            asof=common.now_utc(), source="CoinGecko", source_url=page, evidence="C", reading="none", rule=NO_RULE,
            note=note + limit, history=common.accumulate(id, [[today, round(value, 2)]]))

    return [
        row("public_total", f"上市公司持有的{cn}合計", total,
            f"{total:,.0f} 顆（{len(companies)} 家公司，約占總供給 {share:.1f}%）",
            f"把{cn}放在公司資產負債表上的上市公司，持有量加總。持續增加代表企業財庫這群買家還在買，"
            "減少代表有公司開始賣（通常是股價低於持幣價值、被迫還債的時候）。"),
        row("top_holder", f"持有最多{cn}的上市公司", top_amount,
            f"{top_amount:,.0f} 顆（{top['name']}，占上市公司合計 {top_amount / total * 100:.0f}%）",
            f"單一最大的企業買家目前是 {top['name']}。這類公司靠發行股票與債券籌錢買{cn}，買進速度放慢或停止，"
            "代表這條最大的企業買盤在降溫。萬一第一名換人，走勢圖會跳。"),
        row("others", f"其他上市公司持有的{cn}（不含最大一家）", others,
            f"{others:,.0f} 顆（{len(companies) - 1} 家公司）",
            f"扣掉最大一家之後，其他上市公司合計持有的{cn}。用來看企業買盤是只靠一家撐著，還是普遍都在買。"),
    ]


# ---------- 美國現貨 ETF（Farside 公開表格頁） ----------

def _number(text) -> float | None:
    text = str(text).strip().replace(",", "").replace("*", "")
    if not text or text in ("-", "nan"):
        return None
    negative = text.startswith("(") and text.endswith(")")
    try:
        value = float(text.strip("()"))
    except ValueError:
        return None
    return -value if negative else value


def _etf_frame(asset: str) -> pd.DataFrame:
    """每日各檔 ETF 淨流入（百萬美元；索引 YYYY-MM-DD，欄位是代號與 Total）。被擋就直接丟例外，不繞過。"""
    url = FARSIDE[asset][0]
    reply = requests.get(url, headers=common.HEADERS, timeout=40)
    time.sleep(common.PAUSE)
    if reply.status_code in (403, 429, 503):
        raise RuntimeError(f"Farside 回應 HTTP {reply.status_code}，可能是機器人檢查；依規定不繞過")
    reply.raise_for_status()
    best = None
    for table in pd.read_html(io.StringIO(reply.text)):
        if table.shape[1] < 3:
            continue
        days = pd.to_datetime(table.iloc[:, 0].astype(str).str.strip(), format="%d %b %Y", errors="coerce")
        if days.notna().sum() > 5 and (best is None or days.notna().sum() > best[0].notna().sum()):
            best = (days, table)
    if best is None:
        raise RuntimeError("Farside 頁面裡找不到每日流量表格（頁面格式改了，或被機器人檢查頁取代）")
    days, table = best
    columns = [str(c[0] if isinstance(c, tuple) else c).strip() for c in table.columns]
    if columns[-1].lower() != "total":
        raise RuntimeError("Farside 表格最後一欄不是合計（Total），格式改了")
    frame = table.iloc[:, 1:].apply(lambda column: column.map(_number))
    frame.columns = columns[1:]
    frame.index = days.dt.strftime("%Y-%m-%d").values
    frame = frame[days.notna().values].sort_index()
    frame = frame[frame["Total"].notna()]
    if frame.empty:
        raise RuntimeError("Farside 表格沒有可用的合計數字")
    return frame.astype(float)


def _etf(asset: str) -> list[dict]:
    frame = _etf_frame(asset)
    url, label = FARSIDE[asset]
    a = asset.lower()
    last = frame.index[-1]
    limit = ("限制：Farside 彙整各發行商自己公布的數字，美國收盤後幾小時到隔天才齊全，最新一天可能還會修正；只有美股交易日有資料；"
             "ETF 流入也可能是套利部位（同時放空期貨），不等於單純看多。")
    cumulative = frame["Total"].cumsum()
    value = float(cumulative.iloc[-1])
    peak = float(cumulative.max())
    rows = [common.indicator(
        id=f"{a}_etf_cumulative_flow", group=GROUP, asset=asset, name=f"美國現貨{label} ETF 累計淨流入（上市以來）",
        value=round(value, 1), unit="百萬美元",
        display=f"{value / 100:,.1f} 億美元（最高曾到 {peak / 100:,.1f} 億美元；截至 {last}）", asof=_asof(last),
        source="Farside Investors（自行累加）", source_url=url, evidence="C", reading="none", rule=NO_RULE,
        note=(f"從美國現貨{label} ETF 上市第一天起，把每天的淨流入一路加起來。線往上代表透過券商帳戶買幣的美國散戶、投資顧問與機構"
              "整體還在加碼，線走平或往下代表這群買家停手或撤出。這是投入的美元金額，不是持有的顆數，也不含幣價漲跌。" + limit),
        history=_history(cumulative, 1))]
    total = float(frame["Total"].rolling(30).sum().iloc[-1])
    for ticker, name in ISSUERS[asset]:
        id = f"{a}_etf_{ticker.lower()}_flow_30d"
        if ticker not in frame.columns:
            rows.append(common.failed(id=id, group=GROUP, asset=asset, name=f"{name} 近 30 個交易日淨流入",
                                      source="Farside Investors", source_url=url,
                                      error=f"Farside 表格裡找不到 {ticker} 這一欄"))
            continue
        series = frame[ticker].fillna(0.0).rolling(30).sum().dropna()
        value = float(series.iloc[-1])
        rows.append(common.indicator(
            id=id, group=GROUP, asset=asset, name=f"{name} 近 30 個交易日淨流入", value=round(value, 1), unit="百萬美元",
            display=f"{_flow_words(value)}（同期間全部 ETF 合計{_flow_words(total)}；截至 {last}）", asof=_asof(last),
            source="Farside Investors（自行加總）", source_url=url, evidence="C", reading="none",
            rule="正數是淨流入，負數是淨流出。" + NO_RULE, note=ISSUER_NOTE[ticker] + limit,
            history=_history(series, 1)))
    return rows


# ---------- 穩定幣鑄造與銷毀、各鏈分布（DefiLlama） ----------

def _llama(path: str, stablecoin: int | None = None, usd_only: bool = False) -> pd.Series:
    reply = common.get_json(f"{LLAMA}/{path}", {"stablecoin": stablecoin} if stablecoin else None, timeout=40)
    points = {}
    for point in reply[-(common.HISTORY_POINTS + 40):]:
        amounts = point.get("totalCirculatingUSD") or {}
        total = float(amounts.get("peggedUSD") or 0) if usd_only else sum(float(v) for v in amounts.values() if v)
        if total > 0:
            points[_ts_day(point["date"])] = total
    series = pd.Series(points).sort_index()
    if series.empty:
        raise ValueError("DefiLlama 沒有回傳穩定幣資料")
    return series


def _stable_change(id: str, label: str, stablecoin: int, who: str) -> dict:
    series = _llama("all", stablecoin)
    change = series.diff(7).dropna()
    if change.empty:
        raise ValueError("穩定幣供給資料不足 7 天")
    value = float(change.iloc[-1])
    month = float(series.iloc[-1] - series.iloc[-31]) if len(series) > 31 else None
    extra = "" if month is None else f"、30 天{_signed_yi(month)}"
    return common.indicator(
        id=id, group=GROUP, asset="MARKET", name=f"{label}近 7 天淨鑄造（供給變化）", value=round(value), unit="美元",
        display=f"7 天{_signed_yi(value)}（目前流通 {_yi(float(series.iloc[-1]))}{extra}）",
        asof=_asof(change.index[-1]), source="DefiLlama（自行換算）", source_url=LLAMA_PAGE, evidence="C",
        reading="none", rule="正數是鑄造多於銷毀（新錢進場），負數是銷毀多於鑄造（贖回離場）。" + NO_RULE,
        note=(f"{label}流通量和 7 天前相比增減多少美元，等於這 7 天新鑄造的減掉被銷毀（贖回）的。{who}"
              "大額鑄造常被當成有新資金準備進場，大額銷毀則是資金換回美元離開。"
              "限制：鑄造出來不一定馬上拿去買幣；發行商有時先鑄造備用；資料大約每天更新一次。"),
        history=_history(change, 0))


def _chain(id: str, chain: str, label: str, what: str, total: pd.Series | None) -> dict:
    series = _llama(chain, usd_only=True)
    value = float(series.iloc[-1])
    week = float(series.iloc[-1] - series.iloc[-8]) if len(series) > 8 else None
    share = ""
    if total is not None and series.index[-1] in total.index and total[series.index[-1]] > 0:
        share = f"，占全部美元穩定幣 {value / float(total[series.index[-1]]) * 100:.0f}%"
    week_words = "" if week is None else f"，7 天{_signed_yi(week)}"
    return common.indicator(
        id=id, group=GROUP, asset="MARKET", name=f"{label}鏈上的穩定幣存量", value=round(value), unit="美元",
        display=f"{_yi(value)}{share}{week_words}", asof=_asof(series.index[-1]), source="DefiLlama",
        source_url=f"https://defillama.com/stablecoins/{chain}", evidence="C", reading="none", rule=NO_RULE,
        note=(f"發行在{label}這條鏈上的美元穩定幣總額。{what}"
              "哪條鏈的存量增加，代表資金正往那個生態移動。限制：跨鏈搬移會讓一條鏈增加、另一條減少，不代表有新錢進來；"
              "資料大約每天更新一次。"),
        history=_history(series, 0))


# ---------- 礦工（CoinMetrics 社群版） ----------

_miner_cache: dict = {}


def _miner_frame() -> pd.DataFrame:
    if "frame" not in _miner_cache:
        start = (datetime.now(timezone.utc) - timedelta(days=common.HISTORY_POINTS + 410)).strftime("%Y-%m-%d")
        reply = common.get_json(CM_URL, {"assets": "btc", "metrics": "IssTotUSD,FeeTotNtv,PriceUSD,HashRate",
                                         "frequency": "1d", "start_time": start, "page_size": 10000}, timeout=60)
        data = list(reply["data"])
        for _ in range(5):
            if not reply.get("next_page_url"):
                break
            reply = common.get_json(reply["next_page_url"], timeout=60)
            data += reply["data"]
        frame = pd.DataFrame(data)
        if frame.empty:
            raise ValueError("CoinMetrics 沒有回傳資料")
        frame.index = frame["time"].str[:10]
        _miner_cache["frame"] = frame.drop(columns=["asset", "time"]).apply(pd.to_numeric, errors="coerce").sort_index()
    return _miner_cache["frame"]


def _miner_revenue() -> dict:
    frame = _miner_frame()
    fee_usd = frame["FeeTotNtv"] * frame["PriceUSD"]
    series = (frame["IssTotUSD"] + fee_usd).dropna().rolling(7).mean().dropna()
    value = float(series.iloc[-1])
    fees = float(fee_usd.dropna().rolling(7).mean().iloc[-1])
    return common.indicator(
        id="btc_miner_revenue_7d", group=GROUP, asset="BTC", name="比特幣礦工每日收入（7 日平均）", value=round(value),
        unit="美元／天", display=f"{value / 1e4:,.0f} 萬美元／天（其中手續費占 {fees / value * 100:.1f}%）",
        asof=_asof(series.index[-1]), source=CM_NAME, source_url=CM_PAGE, evidence="C", reading="none", rule=NO_RULE,
        note=("全體礦工每天挖到的新幣加上手續費，換算成美元。礦工要付電費，是市場上固定的賣方；收入掉得深，"
              "體質差的礦工會被迫賣出庫存或關機，收入很高時礦工壓力小。限制：這是收入不是賣出量，"
              "看不出礦工實際賣了多少；CoinMetrics 社群版資料通常晚 1 天。"),
        history=_history(series, 0))


def _puell() -> dict:
    issuance = _miner_frame()["IssTotUSD"].dropna()
    series = (issuance.rolling(7).mean() / issuance.rolling(365).mean()).dropna()
    if series.empty:
        raise ValueError("新幣發行金額不足 365 天，算不出普爾倍數")
    value = float(series.iloc[-1])
    reading = "bearish" if value >= PUELL_HOT else "bullish" if value <= PUELL_COLD else "neutral"
    return common.indicator(
        id="btc_puell_multiple", group=GROUP, asset="BTC", name="普爾倍數（礦工近 7 日收入 ÷ 過去一年平均）",
        value=round(value, 3), unit="倍", display=f"{value:.2f} 倍", asof=_asof(series.index[-1]), source=CM_NAME,
        source_url=CM_PAGE, evidence="C", reading=reading,
        rule=(f"≥ {PUELL_HOT:g} 視為過熱＝偏空，≤ {PUELL_COLD:g} 視為過冷＝偏多，其餘中性。這是指標作者 David Puell 2019 年提出時的慣例值，"
              "沒有在本專案驗證過；每次減半後高點一次比一次低，4 倍的門檻近幾年已經很難碰到。"),
        note=("最近 7 天平均每天新挖出的比特幣價值，除以過去 365 天的平均（原始定義用單日，單日出塊數忽多忽少，所以本頁取 7 日平均）。倍數高代表礦工現在賺得比平常多很多（歷史上出現在週期高點附近），"
              "倍數低代表礦工收入遠低於平常（歷史上出現在熊市底部、礦工投降的時候）。只算新發行的幣，不含手續費。"
              "限制：減半當天會瞬間掉一半，那是制度造成的，不是市場訊號；CoinMetrics 社群版資料通常晚 1 天。"),
        history=_history(series, 3))


def _hashprice() -> dict:
    frame = _miner_frame()
    revenue = frame["IssTotUSD"] + frame["FeeTotNtv"] * frame["PriceUSD"]
    series = (revenue / (frame["HashRate"] / 1000)).dropna().rolling(7).mean().dropna()  # HashRate is in TH/s
    value = float(series.iloc[-1])
    return common.indicator(
        id="btc_hashprice_7d", group=GROUP, asset="BTC", name="礦工每單位算力的收入（7 日平均）", value=round(value, 2),
        unit="美元／PH／天", display=f"{value:.1f} 美元（每 1 PH/s 算力每天）", asof=_asof(series.index[-1]),
        source=CM_NAME, source_url=CM_PAGE, evidence="C", reading="none", rule=NO_RULE,
        note=("礦工每投入一單位算力（1 PH/s，大約 5 到 10 台新型礦機）一天能賺多少美元，業界叫 hashprice。"
              "它同時反映幣價、手續費與挖礦競爭；跌到歷史低檔代表礦工利潤被壓到極限，容易出現關機與賣幣，"
              "這種「礦工投降」歷史上多出現在熊市後段。限制：長期趨勢本來就向下（礦機越來越有效率），"
              "要看的是相對近一兩年的位置；算力是由出塊速度估計的，單日會亂跳，所以取 7 日平均。"),
        history=_history(series, 2))


# ---------- 預測市場（Polymarket） ----------

def _poly_event(asset: str) -> dict:
    slug_coin, title_coin = POLY_COIN[asset]
    year = datetime.now(timezone.utc).year
    events = common.get_json(POLY, {"slug": f"what-price-will-{slug_coin}-hit-before-{year + 1}"}, timeout=30)
    events = [e for e in events if not e.get("closed")]
    if not events:  # the slug changes from year to year, so fall back to the biggest matching event by volume
        listed = common.get_json(POLY, {"closed": "false", "limit": 30, "order": "volume", "ascending": "false",
                                        "tag_slug": slug_coin}, timeout=40)
        pattern = re.compile(rf"^What price will {title_coin} hit in \d{{4}}\?$")
        events = [e for e in listed if pattern.match(str(e.get("title", "")))]
    if not events:
        raise RuntimeError("Polymarket 找不到「今年會碰到什麼價位」的市場")
    return events[0]


def _polymarket(asset: str) -> list[dict]:
    event = _poly_event(asset)
    spot = float(common.get_json(f"https://api.exchange.coinbase.com/products/{asset}-USD/ticker")["price"])
    a, cn, today = asset.lower(), ASSET_CN[asset], common.today_utc()
    url = f"https://polymarket.com/event/{event['slug']}"
    end = str(event.get("endDate") or "")[:10]
    ups, downs = [], []
    for m in event.get("markets") or []:
        title = str(m.get("groupItemTitle") or "")
        if m.get("closed") or not m.get("active") or not m.get("outcomePrices") or title[:1] not in ("↑", "↓"):
            continue
        try:
            strike = float(re.sub(r"[^\d.]", "", title))
            prob = float(json.loads(m["outcomePrices"])[0]) * 100
        except (ValueError, IndexError, TypeError):
            continue
        item = {"strike": strike, "prob": prob, "volume": float(m.get("volume24hr") or 0)}
        if title[0] == "↑" and strike > spot:
            ups.append(item)
        elif title[0] == "↓" and strike < spot:
            downs.append(item)
    if not ups or not downs:
        raise RuntimeError("Polymarket 的價位市場裡沒有可用的上漲或下跌題目")
    total_volume = float(event.get("volume") or 0)
    limit = (f"這是 Polymarket 上用真錢下注的機率，題目是「{end} 前{cn}會不會碰到某個價位」（碰到一次就算）。"
             f"整個市場累計成交約 {total_volume / 1e4:,.0f} 萬美元，參與者是加密圈散戶與少數大戶，只當情緒參考。"
             "限制：越接近到期，機率自然越低；美國用戶受限；冷門價位的報價很粗；每年換一個新市場時數字會跳。")

    def top(items: list) -> dict:
        live = [i for i in items if 5 <= i["prob"] <= 95] or items
        return max(live, key=lambda i: i["volume"])

    rows = []
    for key, items, arrow in (("up", ups, "漲到"), ("down", downs, "跌到")):
        pick = top(items)
        id = f"{a}_polymarket_{key}"
        rows.append(common.indicator(
            id=id, group="sentiment", asset=asset,
            name=f"預測市場：{cn}成交最多的{'上漲' if key == 'up' else '下跌'}題目機率", value=round(pick["prob"], 2), unit="%",
            display=f"{pick['prob']:.1f}%（{end} 前{arrow} {pick['strike']:,.0f} 美元；現價 {spot:,.0f}）",
            asof=common.now_utc(), source=POLY_NAME, source_url=url, evidence="C", reading="none", rule=NO_RULE,
            note=(f"在{'高' if key == 'up' else '低'}於現價的價位題目裡，挑近 24 小時成交最多的一題（機率要在 5% 到 95% 之間），顯示它目前的機率。"
                  "注意：被挑中的價位會隨行情改變，所以走勢圖前後不一定是同一題，只能看大概。" + limit + SNAPSHOT),
            history=common.accumulate(id, [[today, round(pick["prob"], 2)]])))
    return rows


# ---------- 組裝 ----------

def _run(rows: list, build, failures: list[dict]) -> None:
    """每個指標各自 try/except；失敗時用 failures 裡的說明留空位。"""
    try:
        result = build()
        rows.extend(result if isinstance(result, list) else [result])
    except Exception as error:  # noqa: BLE001 一個來源壞掉不能拖垮其他指標
        for meta in failures:
            meta = dict(meta)
            rows.append(common.failed(group=meta.pop("group", GROUP), error=f"{type(error).__name__}: {error}", **meta))


def collect() -> list[dict]:
    _miner_cache.clear()
    rows: list[dict] = []
    for asset, cn in ASSET_CN.items():
        a = asset.lower()
        _run(rows, lambda asset=asset: _treasury(asset),
             [dict(id=f"{a}_treasury_{suffix}", asset=asset, name=name, source="CoinGecko",
                   source_url=COINGECKO_PAGE[asset])
              for suffix, name in (("public_total", f"上市公司持有的{cn}合計"), ("top_holder", f"持有最多{cn}的上市公司"),
                                   ("others", f"其他上市公司持有的{cn}（不含最大一家）"))])
    for asset, (url, label) in FARSIDE.items():
        a = asset.lower()
        _run(rows, lambda asset=asset: _etf(asset),
             [dict(id=f"{a}_etf_cumulative_flow", asset=asset, name=f"美國現貨{label} ETF 累計淨流入（上市以來）",
                   source="Farside Investors", source_url=url)]
             + [dict(id=f"{a}_etf_{ticker.lower()}_flow_30d", asset=asset, name=f"{name} 近 30 個交易日淨流入",
                     source="Farside Investors", source_url=url) for ticker, name in ISSUERS[asset]])
    stables = (("market_usdt_supply_change_7d", "泰達幣（USDT）", 1, "泰達幣主要在美國以外的交易所使用，是幣圈最主要的交易資金。"),
               ("market_usdc_supply_change_7d", "USDC", 2, "USDC 由美國公司 Circle 發行，美國機構與鏈上金融用得比較多。"))
    for id, label, stablecoin, who in stables:
        _run(rows, lambda id=id, label=label, stablecoin=stablecoin, who=who: _stable_change(id, label, stablecoin, who),
             [dict(id=id, asset="MARKET", name=f"{label}近 7 天淨鑄造（供給變化）", source="DefiLlama",
                   source_url=LLAMA_PAGE)])
    try:
        total = _llama("all", usd_only=True)
    except Exception:  # noqa: BLE001 - the share of the total is optional
        total = None
    chains = (("market_stablecoin_ethereum", "Ethereum", "以太坊", "以太坊是鏈上金融（借貸、去中心化交易）的主要場所，機構資金多在這裡。"),
              ("market_stablecoin_tron", "Tron", "波場", "波場幾乎都是泰達幣，主要用在亞洲、新興市場的轉帳與場外交易。"),
              ("market_stablecoin_solana", "Solana", "Solana", "Solana 上以 USDC 為主，和迷因幣與鏈上交易的熱度連動。"))
    for id, chain, label, what in chains:
        _run(rows, lambda id=id, chain=chain, label=label, what=what: _chain(id, chain, label, what, total),
             [dict(id=id, asset="MARKET", name=f"{label}鏈上的穩定幣存量", source="DefiLlama",
                   source_url=f"https://defillama.com/stablecoins/{chain}")])
    for build, id, name in ((_miner_revenue, "btc_miner_revenue_7d", "比特幣礦工每日收入（7 日平均）"),
                            (_puell, "btc_puell_multiple", "普爾倍數（礦工近 7 日收入 ÷ 過去一年平均）"),
                            (_hashprice, "btc_hashprice_7d", "礦工每單位算力的收入（7 日平均）")):
        _run(rows, build, [dict(id=id, asset="BTC", name=name, source=CM_NAME, source_url=CM_PAGE)])
    for asset, cn in ASSET_CN.items():
        a = asset.lower()
        _run(rows, lambda asset=asset: _polymarket(asset),
             [dict(id=f"{a}_polymarket_{key}", group="sentiment", asset=asset, name=name, source=POLY_NAME,
                   source_url="https://polymarket.com/crypto")
              for key, name in (("up", f"預測市場：{cn}成交最多的上漲題目機率"),
                                ("down", f"預測市場：{cn}成交最多的下跌題目機率"))])
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    started = time.time()
    rows = collect()
    common.save("flows2", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
    print(f"共 {len(rows)} 個指標，失敗 {sum(1 for r in rows if r['error'])} 個，耗時 {time.time() - started:.0f} 秒")
