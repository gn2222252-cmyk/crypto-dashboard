"""進階鏈上指標收集器（group="onchain"，存成 cache/onchain2.json）。規格見 dashboard/SPEC2.md。

來源（全部免費、公開、不需金鑰）：
- CoinMetrics 社群版 API：市值、MVRV（可拿到 2010 年起的完整歷史，所以 MVRV Z 分數與 NUPL 自己算，當天就有）、
  以太幣流通量／手續費／交易筆數。社群版沒有開放：SplyAct1yr、TxTfrValAdjUSD、TxTfrValUSD、RevUSD、AdrBalUSD1MCnt、
  CapRealUSD（回傳 403）；Solana 只有價格。
- blockchain.info charts API：礦工收入（Puell 倍數）、算力（算力帶）、鏈上轉帳金額（NVT 訊號）。
- BGeometrics（bitcoin-data.com）免費 API：SOPR、短期持有者 MVRV、長期持有者供給、Reserve Risk、一年以上未動供給、
  以太幣質押量。免費版每小時 10 次、每天 15 次，而且多數指標不給最近 7 天，所以每個端點一天只抓一次，
  結果存在 cache/_store/bg_<端點>.json；當天抓不到（額度用完）就用上次存的，並在說明裡註明。
- ultrasound.money：以太幣信標鏈目前的質押餘額（只有當下值，補在 BGeometrics 歷史的最後面）。
- DefiLlama：以太坊與 Solana 的鎖倉金額、去中心化交易所成交量、鏈上手續費、Solana 上的穩定幣。
拿不到的：beaconcha.in（要金鑰，401）、DefiLlama 的質押利率端點（付費，402）、Solana 活躍地址（沒有免費免金鑰來源）。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from dashboard import common

GROUP = "onchain"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_NAME = "CoinMetrics 社群版"
CM_PAGE = "https://charts.coinmetrics.io/crypto-data/"
CM_DELAY = "CoinMetrics 社群版資料通常晚 1 天。"
CM_SPEC = {"BTC": ("2010-07-18", ("CapMrktCurUSD", "CapMVRVCur", "PriceUSD")),
           "ETH": ("2015-08-08", ("CapMrktCurUSD", "CapMVRVCur", "PriceUSD", "SplyCur", "FeeTotNtv", "TxCnt"))}
BCI_URL = "https://api.blockchain.info/charts/"
BCI_NAME = "blockchain.info"
BCI_PAGE = "https://www.blockchain.com/explorer/charts/"
BG_URL = "https://bitcoin-data.com/v1/"
BG_NAME = "BGeometrics（bitcoin-data.com 免費版）"
BG_PAGE = "https://charts.bgeometrics.com/"
LLAMA = "https://api.llama.fi"
LLAMA_NAME = "DefiLlama"
ASSET_NAMES = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "Solana"}
UNVERIFIED = "門檻來自指標原作者或市場通行用法，本頁沒有自己驗證過。"
NO_RULE = "沒有事前固定的門檻，不給方向。"
_memo: dict = {}


# ---------- 小工具 ----------

def _day(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), timezone.utc).strftime("%Y-%m-%d")


def _asof(day: str) -> str:
    return f"{day}T00:00:00Z"


def _history(series: pd.Series, digits: int = 4) -> list:
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


def _cached(key: str, build):
    """同一次執行裡同一個來源只抓一次；抓失敗也記住，後面的指標直接得到同一個錯誤，不重複連線。"""
    if key not in _memo:
        try:
            _memo[key] = (build(), None)
        except Exception as error:  # noqa: BLE001
            _memo[key] = (None, error)
    value, error = _memo[key]
    if error is not None:
        raise error
    return value


def _cm(asset: str) -> pd.DataFrame:
    """CoinMetrics 完整日資料（MVRV Z 分數需要從頭算的市值標準差），索引是 YYYY-MM-DD 字串。"""
    def build() -> pd.DataFrame:
        start, metrics = CM_SPEC[asset]
        reply = common.get_json(CM_URL, {"assets": asset.lower(), "metrics": ",".join(metrics), "frequency": "1d",
                                         "start_time": start, "page_size": 10000}, timeout=60)
        data = list(reply["data"])
        for _ in range(3):
            if not reply.get("next_page_url"):
                break
            reply = common.get_json(reply["next_page_url"], timeout=60)
            data += reply["data"]
        frame = pd.DataFrame(data)
        if frame.empty:
            raise ValueError("CoinMetrics 沒有回傳資料")
        frame.index = frame["time"].str[:10]
        return frame.drop(columns=["asset", "time"]).apply(pd.to_numeric, errors="coerce").sort_index()
    return _cached(f"cm_{asset}", build)


def _bci(chart: str) -> pd.Series:
    """blockchain.info 的日資料（5 年），索引是 YYYY-MM-DD 字串。"""
    def build() -> pd.Series:
        reply = common.get_json(BCI_URL + chart, {"timespan": "5years", "format": "json", "sampled": "false"},
                                timeout=40)
        series = pd.Series({_day(point["x"]): float(point["y"]) for point in reply["values"]}).sort_index()
        if len(series) < 400:
            raise ValueError(f"blockchain.info {chart} 只回傳 {len(series)} 筆")
        return series
    return _cached(f"bci_{chart}", build)


def _bg(endpoint: str) -> tuple[pd.Series, str]:
    """BGeometrics 的一個端點：一天只抓一次，存在 _store/bg_<端點>.json。回傳（數列, 補充說明）。

    免費額度很小（每小時 10 次、每天 15 次）。今天已經抓過就直接讀存檔；今天抓不到就退回上次的存檔並註明；
    連存檔都沒有才算失敗。
    """
    def build() -> tuple[pd.Series, str]:
        store_id = "bg_" + endpoint.replace("-", "_")
        path = common.STORE / f"{store_id}.json"
        fresh = path.exists() and _day(path.stat().st_mtime) == common.today_utc()
        extra = ""
        if not fresh:
            try:
                reply = common.get_json(BG_URL + endpoint, timeout=40)
                points = []
                for row in reply:
                    for key, value in row.items():
                        if key in ("d", "unixTs", "theDay") or value is None:
                            continue
                        try:
                            points.append([str(row["d"])[:10], float(value)])
                        except (TypeError, ValueError):
                            continue
                        break
                if len(points) < 30:
                    raise ValueError(f"只回傳 {len(points)} 筆")
                common.accumulate(store_id, points)
            except Exception as error:  # noqa: BLE001
                if not path.exists():
                    raise ValueError(f"BGeometrics {endpoint} 抓不到，也沒有先前的存檔（{type(error).__name__}: "
                                     f"{str(error)[:120]}）") from error
                extra = "今天向來源更新失敗（可能是免費額度用完），這裡顯示的是上次存下來的資料。"
        stored = json.loads(path.read_text(encoding="utf-8"))
        series = pd.Series({str(day): float(value) for day, value in stored.items()}).sort_index()
        if series.empty:
            raise ValueError(f"BGeometrics {endpoint} 的存檔是空的")
        lag = (datetime.now(timezone.utc) - datetime.fromisoformat(series.index[-1] + "T00:00:00+00:00")).days
        delay = f"這個指標在來源的免費版不提供最近 7 天，所以數值晚約 {lag} 天。" if lag > 3 else ""
        return series.iloc[-common.HISTORY_POINTS - 60:], delay + "每天只向來源抓一次。" + extra
    return _cached(f"bg_{endpoint}", build)


def _llama_chart(key: str, url: str, params: dict | None = None) -> pd.Series:
    """DefiLlama 的每日數列（totalDataChart：[[秒, 數值], ...]）。今天還沒走完的那一天不算。"""
    def build() -> pd.Series:
        reply = common.get_json(url, params, timeout=40)
        series = pd.Series({_day(point[0]): float(point[1]) for point in reply["totalDataChart"]}).sort_index()
        series = series[series.index < common.today_utc()]
        if len(series) < 30:
            raise ValueError(f"DefiLlama 只回傳 {len(series)} 筆")
        return series
    return _cached(key, build)


def _llama_tvl(chain: str) -> pd.Series:
    def build() -> pd.Series:
        reply = common.get_json(f"{LLAMA}/v2/historicalChainTvl/{chain}", timeout=40)
        series = pd.Series({_day(point["date"]): float(point["tvl"]) for point in reply}).sort_index()
        if len(series) < 30:
            raise ValueError(f"DefiLlama 只回傳 {len(series)} 筆")
        return series
    return _cached(f"tvl_{chain}", build)


# ---------- 比特幣與以太幣：用 CoinMetrics 自己算的估值指標 ----------

def _mvrv_z(asset: str) -> dict:
    frame = _cm(asset)[["CapMrktCurUSD", "CapMVRVCur"]].dropna()
    realized = frame["CapMrktCurUSD"] / frame["CapMVRVCur"]
    z = ((frame["CapMrktCurUSD"] - realized) / frame["CapMrktCurUSD"].expanding(min_periods=365).std()).dropna()
    value = float(z.iloc[-1])
    label = ASSET_NAMES[asset]
    if asset == "BTC":
        reading = "bearish" if value > 7 else "bullish" if value < 0 else "neutral"
        rule = ("高於 7 視為過熱＝偏空，低於 0 視為過冷＝偏多，中間不給方向。" + UNVERIFIED +
                "（出處：指標作者 Murad Mahmudov、David Puell 與各圖表網站通行的紅綠區。）"
                "注意：近兩輪高點都沒有碰到 7，門檻可能已經過時。")
    else:
        reading, rule = "none", "7 與 0 的門檻是比特幣的慣例；以太幣沒有事前固定的門檻，所以不給方向。"
    return common.indicator(
        id=f"{asset.lower()}_mvrv_z", group=GROUP, asset=asset, name=f"{label} MVRV Z 分數（市值偏離成本的程度）",
        value=round(value, 3), unit="分", display=f"{value:.2f}", asof=_asof(z.index[-1]),
        source=CM_NAME + "（自行計算）", source_url=CM_PAGE, evidence="C", reading=reading, rule=rule,
        note=("算法：（總市值 − 已實現市值）÷ 總市值從有資料以來的標準差。已實現市值是每顆幣用上次在鏈上移動時的價格"
              "計算的總成本。分數越高代表市值比全體成本高出越多、帳面獲利越大。"
              "限制：和 MVRV 是同一份資訊的另一種寫法，不是獨立證據；標準差會隨歷史變長而改變。" + CM_DELAY),
        history=_history(z, 3))


def _nupl(asset: str) -> dict:
    series = (1 - 1 / _cm(asset)["CapMVRVCur"].dropna()).dropna()
    value = float(series.iloc[-1])
    label = ASSET_NAMES[asset]
    if asset == "BTC":
        reading = "bearish" if value > 0.75 else "bullish" if value < 0 else "neutral"
        rule = ("高於 0.75（俗稱「狂喜區」）視為過熱＝偏空，低於 0（全體帳面賠錢，俗稱「投降區」）視為過冷＝偏多，"
                "中間不給方向。" + UNVERIFIED + "（出處：Adamant Capital 2019 年的報告與各圖表網站通行的分區。）")
    else:
        reading, rule = "none", "0.75 與 0 的門檻是比特幣的慣例；以太幣沒有事前固定的門檻，所以不給方向。"
    return common.indicator(
        id=f"{asset.lower()}_nupl", group=GROUP, asset=asset, name=f"{label} NUPL（全體未實現淨損益占市值比例）",
        value=round(value, 4), unit="比例", display=f"{value:.2f}（全體帳面{'獲利' if value >= 0 else '虧損'}約占市值 {abs(value) * 100:.0f}%）",
        asof=_asof(series.index[-1]), source=CM_NAME + "（自行計算）", source_url=CM_PAGE, evidence="C",
        reading=reading, rule=rule,
        note=("NUPL 是「未實現淨損益」的英文縮寫。算法：（總市值 − 已實現市值）÷ 總市值，也等於 1 − 1 ÷ MVRV。"
              "正數代表全體持有人帳面上有賺，越接近 1 賺越多；負數代表全體帳面賠錢。"
              "限制：和 MVRV 是同一份資訊，不是獨立證據。" + CM_DELAY),
        history=_history(series, 4))


# ---------- 比特幣：blockchain.info 自己算的指標 ----------

def _puell() -> dict:
    revenue = _bci("miners-revenue")
    puell = (revenue / revenue.rolling(365).mean()).dropna()
    value = float(puell.iloc[-1])
    reading = "bearish" if value > 4 else "bullish" if value < 0.5 else "neutral"
    return common.indicator(
        id="btc_puell_multiple", group=GROUP, asset="BTC", name="比特幣 Puell 倍數（礦工今日收入 ÷ 一年平均）",
        value=round(value, 3), unit="倍",
        display=f"{value:.2f} 倍（礦工當日收入 {revenue.iloc[-1] / 1e4:,.0f} 萬美元）", asof=_asof(puell.index[-1]),
        source=BCI_NAME + "（自行計算）", source_url=BCI_PAGE + "miners-revenue", evidence="C", reading=reading,
        rule=("高於 4 視為過熱＝偏空，低於 0.5 視為過冷＝偏多，中間不給方向。" + UNVERIFIED +
              "（出處：指標作者 David Puell 與各圖表網站通行的紅綠區。）注意：每次減半後礦工收入直接砍半，"
              "近兩輪高點都沒有碰到 4。"),
        note=("礦工每天的收入（新發的幣加手續費，換成美元）除以過去 365 天的平均。倍數高代表礦工現在賺得比平常多很多、"
              "有動機賣幣；倍數低代表礦工收入吃緊。限制：單日數值跳動大；減半當下會突然掉一半。"),
        history=_history(puell, 3))


def _hash_ribbons() -> dict:
    rate = _bci("hash-rate") / 1e6  # TH/s → EH/s
    fast, slow = rate.rolling(30).mean(), rate.rolling(60).mean()
    gap = ((fast / slow - 1) * 100).dropna()
    value = float(gap.iloc[-1])
    below = gap < 0
    streak = int((below[::-1] != below.iloc[-1]).to_numpy().argmax()) if (below != below.iloc[-1]).any() else len(gap)
    state = (f"30 日均線低於 60 日均線已 {streak} 天（俗稱礦工投降期）" if below.iloc[-1]
             else f"30 日均線高於 60 日均線已 {streak} 天（算力成長中）")
    return common.indicator(
        id="btc_hash_ribbons", group=GROUP, asset="BTC", name="比特幣算力帶（算力 30 日均線相對 60 日均線）",
        value=round(value, 2), unit="%",
        display=f"{value:+.1f}%：{state}；30 日 {fast.iloc[-1]:,.0f}、60 日 {slow.iloc[-1]:,.0f} EH/s",
        asof=_asof(gap.index[-1]), source=BCI_NAME + "（自行計算）", source_url=BCI_PAGE + "hash-rate", evidence="C",
        reading="none",
        rule=("數值是 30 日均線比 60 日均線高（正）或低（負）多少。指標作者 Charles Edwards 的說法：30 日跌破 60 日代表"
              "礦工關機投降，之後重新站回 60 日之上是他定義的買進訊號。這個訊號還要搭配價格動能判斷，"
              "本頁沒有驗證，所以不給方向。"),
        note=("算力是全網礦機的運算量（EH/s 是每秒一百京次）。短均線掉到長均線下面，代表最近有不少礦機關機，"
              "通常發生在幣價大跌或減半後礦工虧本的時候。限制：算力是從出塊速度反推的估計值；歷史上訊號次數很少。"),
        history=_history(gap, 2))


def _transfer_volume() -> dict:
    volume = _bci("estimated-transaction-volume-usd")
    average = volume.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    return common.indicator(
        id="btc_transfer_volume_7d", group=GROUP, asset="BTC", name="比特幣鏈上轉帳金額（7 日平均）",
        value=round(value), unit="美元／日", display=f"{_yi(value)}／日（30 天變化 {_pct(_change(average, 30))}）",
        asof=_asof(average.index[-1]), source=BCI_NAME, source_url=BCI_PAGE + "estimated-transaction-volume-usd",
        evidence="C", reading="none", rule=NO_RULE,
        note=("每天在比特幣鏈上轉移的金額（blockchain.info 的估計值，已經扣掉找零），取 7 天平均，用來看鏈上實際使用量。"
              "限制：扣找零的方法是估計；交易所內部搬錢包也會算進去；不同資料商的數字可以差好幾倍。"),
        history=_history(average, 0))


def _nvt_signal() -> dict:
    volume = _bci("estimated-transaction-volume-usd").rolling(90).mean()
    cap = _cm("BTC")["CapMrktCurUSD"].dropna()
    nvt = (cap / volume.reindex(cap.index)).dropna()
    value = float(nvt.iloc[-1])
    return common.indicator(
        id="btc_nvt_signal", group=GROUP, asset="BTC", name="比特幣 NVT 訊號（市值 ÷ 鏈上轉帳金額 90 日平均）",
        value=round(value, 2), unit="倍", display=f"{value:.0f} 倍", asof=_asof(nvt.index[-1]),
        source=f"{CM_NAME}（市值）＋{BCI_NAME}（轉帳金額），自行計算", source_url=BCI_PAGE + "estimated-transaction-volume-usd",
        evidence="C", reading="none",
        rule=("NVT 訊號由 Dmitry Kalichkin 提出（改良自 Willy Woo 的 NVT）。他當年提過高於 150 偏貴、低於 45 偏便宜，"
              "但那是 2018 年以前的資料，而且數值高低完全取決於轉帳金額用哪一家的估計，所以這裡不套用門檻、不給方向，"
              "只看它在自己歷史裡的相對位置。"),
        note=("NVT 是「市值 ÷ 鏈上轉帳金額」，可以想成鏈上版的本益比：市值相對於實際轉帳使用量越高，代表價格跑在使用量前面。"
              "這裡的轉帳金額取 90 天平均讓它平滑。限制：越來越多交易在交易所、ETF、閃電網路裡完成而不上鏈，"
              "長期會讓這個比值越來越高。"),
        history=_history(nvt, 2))


# ---------- 比特幣：BGeometrics 的指標 ----------

def _sopr() -> dict:
    series, extra = _bg("sopr")
    average = series.rolling(7).mean().dropna()
    value, last = float(average.iloc[-1]), float(series.iloc[-1])
    return common.indicator(
        id="btc_sopr_7d", group=GROUP, asset="BTC", name="比特幣 SOPR（已花費輸出獲利率，7 日平均）",
        value=round(value, 4), unit="倍",
        display=f"{value:.3f}（當日 {last:.3f}；{'高於 1：鏈上移動的幣平均是賺錢賣' if value >= 1 else '低於 1：鏈上移動的幣平均是賠錢賣'}）",
        asof=_asof(average.index[-1]), source=BG_NAME, source_url=BG_PAGE, evidence="C", reading="none",
        rule=("1 是分界線：高於 1 代表當天在鏈上移動的幣平均是賺錢出手，低於 1 代表平均是認賠出手。"
              "指標作者 Renato Shirakashi 的觀察是多頭時跌回 1 附近常有支撐、空頭時彈回 1 附近常有壓力，"
              "但沒有通行的「過熱」固定門檻，所以不給方向。"),
        note=("SOPR 是「已花費輸出獲利率」的英文縮寫：把當天每一筆被花掉的幣，用「花掉時的價格 ÷ 當初取得時的價格」加總平均。"
              "限制：交易所內部搬移也會被算進去。" + extra),
        history=_history(average, 4))


def _sth() -> list[dict]:
    series, extra = _bg("sth-mvrv")
    value = float(series.iloc[-1])
    rows = [common.indicator(
        id="btc_sth_mvrv", group=GROUP, asset="BTC", name="比特幣短期持有者 MVRV（現價 ÷ 近期買家的平均成本）",
        value=round(value, 3), unit="倍",
        display=f"{value:.2f} 倍（近期買家平均{'獲利' if value >= 1 else '虧損'} {abs(value - 1) * 100:.0f}%）",
        asof=_asof(series.index[-1]), source=BG_NAME, source_url=BG_PAGE, evidence="C", reading="none",
        rule=("1 是分界線：高於 1 代表最近 155 天內買的人平均有賺，低於 1 代表平均套牢。"
              "沒有通行的固定過熱門檻，所以不給方向。"),
        note=("「短期持有者」指持有不到 155 天的幣（Glassnode 的通行定義）。這個比值看的是最近進場的人賺還是賠："
              "賺很多時容易有獲利賣壓，普遍套牢時容易恐慌。限制：155 天是統計切出來的分界，不是真的知道誰是誰。"
              + extra),
        history=_history(series, 3))]
    price = _cm("BTC")["PriceUSD"].dropna()
    cost = (price.reindex(series.index) / series).dropna()
    if cost.empty:
        raise ValueError("短期持有者成本價算不出來（價格與 MVRV 日期對不上）")
    cost_value, now = float(cost.iloc[-1]), float(price.iloc[-1])
    gap = (now / cost_value - 1) * 100
    rows.append(common.indicator(
        id="btc_sth_cost_price", group=GROUP, asset="BTC", name="比特幣短期持有者成本價（近期買家的平均買進價）",
        value=round(cost_value, 2), unit="美元",
        display=(f"{cost_value:,.0f} 美元（{cost.index[-1]} 的估計；最新價格 {now:,.0f} 美元，"
                 f"{'高於' if gap >= 0 else '低於'}它 {abs(gap):.1f}%）"),
        asof=_asof(cost.index[-1]), source=f"{BG_NAME}＋{CM_NAME}（價格 ÷ 短期持有者 MVRV，自行換算）",
        source_url=BG_PAGE, evidence="C", reading="none",
        rule="沒有事前固定的門檻，不給方向。市場上常把這條線當成多頭時的支撐、空頭時的壓力來看，本頁沒有驗證。",
        note=("持有不到 155 天的幣的平均取得價格。現價在它上方代表近期買家整體有賺，在下方代表整體套牢。"
              "限制：和短期持有者 MVRV 是同一份資訊。" + extra),
        history=_history(cost, 2)))
    return rows


def _lth_supply() -> dict:
    series, extra = _bg("long-term-hodler-supply-btc")
    value = float(series.iloc[-1])
    delta = value - float(series.iloc[-31]) if len(series) > 31 else None
    change = (series.diff(30) / series.shift(30) * 100).dropna()
    word = "無資料" if delta is None else f"{'增加' if delta >= 0 else '減少'} {abs(delta) / 1e4:,.1f} 萬顆"
    return common.indicator(
        id="btc_lth_supply_change_30d", group=GROUP, asset="BTC", name="比特幣長期持有者持有量的 30 天變化",
        value=round(float(change.iloc[-1]), 3), unit="%",
        display=f"{float(change.iloc[-1]):+.2f}%（30 天{word}；目前共 {value / 1e4:,.0f} 萬顆）",
        asof=_asof(series.index[-1]), source=BG_NAME, source_url=BG_PAGE, evidence="C", reading="none",
        rule=("正數代表長期持有者手上的幣變多（囤幣），負數代表變少（出貨）。通行的說法是長期持有者在多頭後段大量出貨、"
              "在空頭慢慢吸回，但沒有固定門檻，所以不給方向。"),
        note=("「長期持有者」指持有超過 155 天沒動的幣（Glassnode 的通行定義）。圖上畫的是持有量 30 天變化的百分比。"
              "限制：ETF 或交易所換託管錢包會讓老幣看起來「被賣掉」；幣只要放滿 155 天就會自動算進來。"
              + extra),
        history=_history(change, 3))


def _reserve_risk() -> dict:
    series, extra = _bg("reserve-risk")
    value = float(series.iloc[-1])
    return common.indicator(
        id="btc_reserve_risk", group=GROUP, asset="BTC", name="比特幣 Reserve Risk（價格相對長期持有者的信心）",
        value=round(value, 6), unit="比值", display=f"{value:.4f}", asof=_asof(series.index[-1]), source=BG_NAME,
        source_url=BG_PAGE, evidence="C", reading="none",
        rule=("指標作者 Hans Hauge 的用法：數值低代表長期持有者信心高而價格低（報酬風險比較好），數值高則相反。"
              "圖表網站常畫的紅綠區（約 0.02 以上、0.002 以下）會隨資料商的算法不同而位移，所以這裡不套用門檻、不給方向，"
              "只看它在自己歷史裡的相對位置。"),
        note=("算法是價格除以「持有者忍住沒賣所累積的機會成本」（用幣齡銷毀量推算）。老幣大量移動、價格又高時數值會衝高。"
              "限制：定義複雜、各家算法不同；和 MVRV 一類指標高度相關。" + extra),
        history=_history(series, 6))


def _hodl_1y() -> dict:
    series, extra = _bg("hodl-one-year")
    if float(series.max()) <= 1.5:  # 來源給的是 0–1 的比例時換成百分比
        series = series * 100
    value = float(series.iloc[-1])
    delta = value - float(series.iloc[-91]) if len(series) > 91 else None
    word = "無資料" if delta is None else f"{delta:+.1f} 個百分點"
    return common.indicator(
        id="btc_supply_dormant_1y_pct", group=GROUP, asset="BTC", name="比特幣超過 1 年沒動過的供給比例",
        value=round(value, 2), unit="%", display=f"{value:.1f}%（90 天變化 {word}）", asof=_asof(series.index[-1]),
        source=BG_NAME, source_url=BG_PAGE, evidence="C", reading="none",
        rule=("通行的說法：比例上升代表大家抱著不賣（多半發生在空頭與盤整），比例從高點明顯下滑代表老幣開始出貨"
              "（多半發生在多頭後段）。沒有固定門檻，所以不給方向。"),
        note=("流通的比特幣裡，最近一次在鏈上移動已經是一年多以前的比例。"
              "限制：遺失的幣永遠算在裡面；ETF 與交易所換錢包會讓老幣「看起來動了」。" + extra),
        history=_history(series, 2))


# ---------- 以太幣 ----------

def _eth_staked() -> dict:
    series, extra, source, beacon = pd.Series(dtype=float), "", BG_NAME, ""
    try:
        series, extra = _bg("eth-staked")
    except Exception:  # noqa: BLE001 BGeometrics 拿不到時退回 ultrasound.money 每天累積的值
        pass
    try:  # ultrasound.money 只有當下值：每天存一筆當備援，並顯示在文字裡（口徑和上面差約 1%，所以不接在同一條線上）
        parts = common.get_json("https://ultrasound.money/api/v2/fees/eth-supply-parts", timeout=30)
        now = float(parts["beaconBalancesSum"]) / 1e9
        stored = common.accumulate("eth_staked_ultrasound", [[common.today_utc(), now]])
        beacon = f"；ultrasound.money 今天的信標鏈餘額 {now / 1e4:,.0f} 萬枚"
        if series.empty:
            series = pd.Series({day: value for day, value in stored})
            source, beacon = "ultrasound.money（每天累積）", ""
            extra = "BGeometrics 抓不到，改用 ultrasound.money 每天累積的信標鏈餘額，歷史從開始累積那天算起。"
    except Exception:  # noqa: BLE001
        if series.empty:
            raise
    value = float(series.iloc[-1])
    share = ""
    try:
        share = f"，約占流通量 {value / float(_cm('ETH')['SplyCur'].dropna().iloc[-1]) * 100:.1f}%"
    except Exception:  # noqa: BLE001
        pass
    return common.indicator(
        id="eth_staked", group=GROUP, asset="ETH", name="以太幣質押總量", value=round(value), unit="ETH",
        display=f"{value / 1e4:,.0f} 萬枚{share}（30 天變化 {_pct(_change(series, 30))}{beacon}）",
        asof=_asof(series.index[-1]), source=source, source_url="https://ultrasound.money/", evidence="C",
        reading="none", rule=NO_RULE,
        note=("鎖在以太坊信標鏈上當驗證保證金的以太幣數量。質押越多，市面上能隨時賣的幣越少，也代表持有人願意長期鎖住。"
              "限制：其中很大一部分是透過 Lido 等流動質押，拿到的憑證仍然可以賣；質押量和價格方向沒有穩定關係。" + extra),
        history=_history(series, 0))


def _eth_issuance() -> dict:
    supply = _cm("ETH")["SplyCur"].dropna()
    net = supply.diff(30).dropna()
    rate = (net / supply.shift(30) * 365 / 30 * 100).dropna()
    value, coins = float(rate.iloc[-1]), float(net.iloc[-1])
    word = "通膨（供給增加）" if coins >= 0 else "通縮（供給減少）"
    return common.indicator(
        id="eth_net_issuance_30d", group=GROUP, asset="ETH", name="以太幣近 30 天淨發行（年化供給變化率）",
        value=round(value, 3), unit="%／年",
        display=f"年化 {value:+.2f}%：30 天淨{'增加' if coins >= 0 else '減少'} {abs(coins) / 1e4:,.2f} 萬枚，目前是{word}",
        asof=_asof(rate.index[-1]), source=CM_NAME + "（流通量 30 天差額，自行計算）", source_url="https://ultrasound.money/",
        evidence="C", reading="none",
        rule="正數是通膨、負數是通縮。沒有事前固定的門檻，不給方向。",
        note=("以太幣每天發新幣給驗證者，同時把每筆交易的基本手續費銷毀；鏈上越忙銷毀越多。兩者相減就是淨發行。"
              "負值（通縮）代表鏈上使用熱絡，正值代表使用冷清。"
              "限制：2024 年 3 月升級後大量交易移到第二層網路，主鏈銷毀量結構性下降，和更早的數字不能直接比。" + CM_DELAY),
        history=_history(rate, 3))


def _eth_avg_fee() -> dict:
    frame = _cm("ETH")[["FeeTotNtv", "PriceUSD", "TxCnt"]].dropna()
    fee = (frame["FeeTotNtv"] * frame["PriceUSD"] / frame["TxCnt"]).rolling(7).mean().dropna()
    value = float(fee.iloc[-1])
    return common.indicator(
        id="eth_avg_tx_fee_7d", group=GROUP, asset="ETH", name="以太坊平均每筆交易手續費（gas 費，7 日平均）",
        value=round(value, 4), unit="美元／筆", display=f"{value:,.2f} 美元／筆（30 天變化 {_pct(_change(fee, 30))}）",
        asof=_asof(fee.index[-1]), source=CM_NAME + "（手續費總額 × 價格 ÷ 交易筆數）", source_url=CM_PAGE,
        evidence="C", reading="none", rule=NO_RULE,
        note=("在以太坊主鏈上發一筆交易平均要付多少錢（gas 費就是以太坊的手續費）。費用高代表大家搶著用鏈、鏈上很熱。"
              "限制：2024 年 3 月升級後主鏈費用結構性變低，和更早的數字不能直接比；單一熱門活動就能讓它暴衝。" + CM_DELAY),
        history=_history(fee, 4))


# ---------- DefiLlama：以太坊與 Solana ----------

def _tvl(asset: str, chain: str) -> dict:
    series = _llama_tvl(chain)
    value = float(series.iloc[-1])
    label = "以太坊" if asset == "ETH" else "Solana"
    return common.indicator(
        id=f"{asset.lower()}_tvl", group=GROUP, asset=asset, name=f"{label}鏈上鎖倉金額（TVL）", value=round(value),
        unit="美元", display=f"{_yi(value)}（30 天變化 {_pct(_change(series, 30))}）", asof=_asof(series.index[-1]),
        source=LLAMA_NAME, source_url=f"https://defillama.com/chain/{chain}", evidence="C", reading="none",
        rule=NO_RULE,
        note=("TVL 是「鎖倉總價值」的英文縮寫：存放在這條鏈上各種去中心化金融協議（借貸、交易池等）裡的資產總值，"
              "用來看有多少錢留在這條鏈上運作。限制：以美元計價，幣價漲跌本身就會讓它跟著漲跌；"
              "同一筆錢層層抵押可能被重複計算（這裡用 DefiLlama 預設口徑，不含流動質押與重複抵押）。"),
        history=_history(series, 0))


def _dex_volume(asset: str, chain: str) -> dict:
    series = _llama_chart(f"dex_{chain}", f"{LLAMA}/overview/dexs/{chain}",
                          {"excludeTotalDataChart": "false", "excludeTotalDataChartBreakdown": "true",
                           "dataType": "dailyVolume"})
    average = series.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    label = "以太坊" if asset == "ETH" else "Solana"
    return common.indicator(
        id=f"{asset.lower()}_dex_volume_7d", group=GROUP, asset=asset, name=f"{label}去中心化交易所成交量（7 日平均）",
        value=round(value), unit="美元／日", display=f"{_yi(value)}／日（30 天變化 {_pct(_change(average, 30))}）",
        asof=_asof(average.index[-1]), source=LLAMA_NAME, source_url=f"https://defillama.com/dexs/chains/{chain.lower()}",
        evidence="C", reading="none", rule=NO_RULE,
        note=(f"{label}上所有去中心化交易所（不透過中心化公司、直接在鏈上撮合）每天的成交金額，取 7 天平均，"
              "用來看鏈上投機與交易熱度。限制：迷因幣熱潮與刷量機器人會讓數字暴增；早期資料涵蓋的交易所較少。"),
        history=_history(average, 0))


def _chain_fees(asset: str, slug: str) -> dict:
    series = _llama_chart(f"fees_{slug}", f"{LLAMA}/summary/fees/{slug}", {"dataType": "dailyFees"})
    average = series.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    return common.indicator(
        id=f"{asset.lower()}_chain_fees_7d", group=GROUP, asset=asset, name="Solana 鏈上手續費（7 日平均）",
        value=round(value), unit="美元／日", display=f"{value / 1e4:,.1f} 萬美元／日（30 天變化 {_pct(_change(average, 30))}）",
        asof=_asof(average.index[-1]), source=LLAMA_NAME, source_url=f"https://defillama.com/fees/{slug}",
        evidence="C", reading="none", rule=NO_RULE,
        note=("使用者每天付給 Solana 網路的交易手續費（基本費加上搶先處理的優先費），取 7 天平均，用來看鏈上擁擠程度。"
              "限制：不含各應用自己另外收的費用；迷因幣熱潮時會暴增，和價格方向沒有穩定關係。"),
        history=_history(average, 0))


def _sol_stablecoins() -> dict:
    reply = common.get_json("https://stablecoins.llama.fi/stablecoincharts/Solana", timeout=40)
    series = pd.Series({_day(point["date"]): float(point["totalCirculatingUSD"]["peggedUSD"])
                        for point in reply if point.get("totalCirculatingUSD", {}).get("peggedUSD") is not None})
    series = series.sort_index()
    if len(series) < 30:
        raise ValueError(f"DefiLlama 只回傳 {len(series)} 筆")
    value = float(series.iloc[-1])
    change = (series.pct_change(30) * 100).dropna()
    return common.indicator(
        id="sol_stablecoin_supply", group=GROUP, asset="SOL", name="Solana 鏈上的美元穩定幣總量", value=round(value),
        unit="美元", display=f"{_yi(value)}（30 天變化 {_pct(float(change.iloc[-1]) if len(change) else None)}）",
        asof=_asof(series.index[-1]), source=LLAMA_NAME, source_url="https://defillama.com/stablecoins/Solana",
        evidence="C", reading="none", rule=NO_RULE,
        note=("在 Solana 上流通的美元穩定幣（USDC、USDT 等）總額，可以看成停在這條鏈上、隨時能買幣的資金。"
              "增加代表錢流進這條鏈。限制：穩定幣也用在支付與套利，不一定會拿去買 SOL。"),
        history=_history(series, 0))


# ---------- 組裝 ----------

def _run(rows: list, build, failures: list[dict]) -> None:
    """每個指標各自 try/except；失敗時用 failures 裡的說明留空位。"""
    try:
        result = build()
        rows.extend(result if isinstance(result, list) else [result])
    except Exception as error:  # noqa: BLE001 一個來源壞掉不能拖垮其他指標
        for meta in failures:
            rows.append(common.failed(group=GROUP, error=f"{type(error).__name__}: {error}", **meta))


def _meta(id: str, asset: str, name: str, source: str, url: str) -> dict:
    return dict(id=id, asset=asset, name=name, source=source, source_url=url)


def collect() -> list[dict]:
    _memo.clear()
    rows: list[dict] = []
    bci, llama = BCI_PAGE, "https://defillama.com/"
    plan = [
        (lambda: _mvrv_z("BTC"), [_meta("btc_mvrv_z", "BTC", "比特幣 MVRV Z 分數（市值偏離成本的程度）", CM_NAME, CM_PAGE)]),
        (lambda: _nupl("BTC"), [_meta("btc_nupl", "BTC", "比特幣 NUPL（全體未實現淨損益占市值比例）", CM_NAME, CM_PAGE)]),
        (_sopr, [_meta("btc_sopr_7d", "BTC", "比特幣 SOPR（已花費輸出獲利率，7 日平均）", BG_NAME, BG_PAGE)]),
        (_sth, [_meta("btc_sth_mvrv", "BTC", "比特幣短期持有者 MVRV（現價 ÷ 近期買家的平均成本）", BG_NAME, BG_PAGE),
                _meta("btc_sth_cost_price", "BTC", "比特幣短期持有者成本價（近期買家的平均買進價）", BG_NAME, BG_PAGE)]),
        (_lth_supply, [_meta("btc_lth_supply_change_30d", "BTC", "比特幣長期持有者持有量的 30 天變化", BG_NAME, BG_PAGE)]),
        (_hodl_1y, [_meta("btc_supply_dormant_1y_pct", "BTC", "比特幣超過 1 年沒動過的供給比例", BG_NAME, BG_PAGE)]),
        (_reserve_risk, [_meta("btc_reserve_risk", "BTC", "比特幣 Reserve Risk（價格相對長期持有者的信心）", BG_NAME,
                               BG_PAGE)]),
        (_puell, [_meta("btc_puell_multiple", "BTC", "比特幣 Puell 倍數（礦工今日收入 ÷ 一年平均）", BCI_NAME, bci)]),
        (_hash_ribbons, [_meta("btc_hash_ribbons", "BTC", "比特幣算力帶（算力 30 日均線相對 60 日均線）", BCI_NAME, bci)]),
        (_transfer_volume, [_meta("btc_transfer_volume_7d", "BTC", "比特幣鏈上轉帳金額（7 日平均）", BCI_NAME, bci)]),
        (_nvt_signal, [_meta("btc_nvt_signal", "BTC", "比特幣 NVT 訊號（市值 ÷ 鏈上轉帳金額 90 日平均）", BCI_NAME, bci)]),
        (lambda: _mvrv_z("ETH"), [_meta("eth_mvrv_z", "ETH", "以太幣 MVRV Z 分數（市值偏離成本的程度）", CM_NAME, CM_PAGE)]),
        (lambda: _nupl("ETH"), [_meta("eth_nupl", "ETH", "以太幣 NUPL（全體未實現淨損益占市值比例）", CM_NAME, CM_PAGE)]),
        (_eth_staked, [_meta("eth_staked", "ETH", "以太幣質押總量", BG_NAME, "https://ultrasound.money/")]),
        (_eth_issuance, [_meta("eth_net_issuance_30d", "ETH", "以太幣近 30 天淨發行（年化供給變化率）", CM_NAME,
                               "https://ultrasound.money/")]),
        (_eth_avg_fee, [_meta("eth_avg_tx_fee_7d", "ETH", "以太坊平均每筆交易手續費（gas 費，7 日平均）", CM_NAME,
                              CM_PAGE)]),
        (lambda: _tvl("ETH", "Ethereum"), [_meta("eth_tvl", "ETH", "以太坊鏈上鎖倉金額（TVL）", LLAMA_NAME, llama)]),
        (lambda: _dex_volume("ETH", "Ethereum"),
         [_meta("eth_dex_volume_7d", "ETH", "以太坊去中心化交易所成交量（7 日平均）", LLAMA_NAME, llama)]),
        (lambda: _tvl("SOL", "Solana"), [_meta("sol_tvl", "SOL", "Solana 鏈上鎖倉金額（TVL）", LLAMA_NAME, llama)]),
        (lambda: _dex_volume("SOL", "Solana"),
         [_meta("sol_dex_volume_7d", "SOL", "Solana 去中心化交易所成交量（7 日平均）", LLAMA_NAME, llama)]),
        (lambda: _chain_fees("SOL", "solana"),
         [_meta("sol_chain_fees_7d", "SOL", "Solana 鏈上手續費（7 日平均）", LLAMA_NAME, llama)]),
        (_sol_stablecoins, [_meta("sol_stablecoin_supply", "SOL", "Solana 鏈上的美元穩定幣總量", LLAMA_NAME, llama)]),
    ]
    for build, failures in plan:
        _run(rows, build, failures)
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = collect()
    common.save("onchain2", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
