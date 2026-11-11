"""鏈上資料收集器（group="onchain"）。規格見 dashboard/SPEC2.md；歷史抓約 4 年日資料。

來源（全部免費、公開、不需金鑰）：
- CoinMetrics 社群版 API：市值、MVRV、流通量、活躍地址、交易所流入流出與餘額、手續費（原生幣）。
  社群版沒有開放：CapRealUSD、FeeTotUSD、NVTAdj、NVTAdj90、DiffLast、RevUSD（回傳 403）。
- mempool.space 公開 API：算力、難度調整、建議手續費、區塊高度（減半計算）。
- blockchain.info charts API：算力與手續費的備援。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from dashboard import common

GROUP = "onchain"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_NAME = "CoinMetrics 社群版"
CM_PAGE = "https://charts.coinmetrics.io/crypto-data/"
CM_METRICS = ("CapMVRVCur", "CapMrktCurUSD", "SplyCur", "PriceUSD", "AdrActCnt", "FlowInExUSD", "FlowOutExUSD",
              "SplyExNtv", "FeeTotNtv")
CM_DELAY = "CoinMetrics 社群版資料通常晚 1 天。"
MEMPOOL = "https://mempool.space/api"
MEMPOOL_NAME = "mempool.space"
HALVING_INTERVAL = 210_000
ASSET_NAMES = {"BTC": "比特幣", "ETH": "以太幣"}
_frames: dict[str, pd.DataFrame] = {}


def _cm(asset: str) -> pd.DataFrame:
    """CoinMetrics 日資料（約 4 年，多抓 40 天給滾動平均用），索引是 YYYY-MM-DD 字串，同一次執行只抓一次。"""
    if asset not in _frames:
        start = (datetime.now(timezone.utc) - timedelta(days=common.HISTORY_POINTS + 40)).strftime("%Y-%m-%d")
        reply = common.get_json(CM_URL, {"assets": asset.lower(), "metrics": ",".join(CM_METRICS), "frequency": "1d",
                                         "start_time": start, "page_size": 10000}, timeout=60)
        data = list(reply["data"])
        for _ in range(5):  # 一頁放不下時照來源給的下一頁網址往下翻
            if not reply.get("next_page_url"):
                break
            reply = common.get_json(reply["next_page_url"], timeout=60)
            data += reply["data"]
        frame = pd.DataFrame(data)
        if frame.empty:
            raise ValueError("CoinMetrics 沒有回傳資料")
        frame.index = frame["time"].str[:10]
        frame = frame.drop(columns=["asset", "time"]).apply(pd.to_numeric, errors="coerce").sort_index()
        _frames[asset] = frame
    return _frames[asset]


def _series(asset: str, column: str) -> pd.Series:
    frame = _cm(asset)
    if column not in frame.columns:
        raise ValueError(f"CoinMetrics 社群版沒有 {column}")
    series = frame[column].dropna()
    if series.empty:
        raise ValueError(f"CoinMetrics {column} 沒有數值")
    return series


def _asof(day: str) -> str:
    return f"{day}T00:00:00Z"


def _history(series: pd.Series, digits: int = 4) -> list:
    series = series.dropna()
    return [[str(day), round(float(value), digits)] for day, value in series.items()]


def _change(series: pd.Series, days: int) -> float | None:
    """與 days 筆之前相比的變化百分比（日資料）。"""
    if len(series) <= days or not series.iloc[-1 - days]:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-1 - days]) - 1) * 100


def _pct(value: float | None) -> str:
    return "無資料" if value is None else f"{value:+.1f}%"


def _yi(usd: float) -> str:
    """美元金額轉成「億美元」字串。"""
    return f"{usd / 1e8:,.1f} 億美元"


def _ts_day(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d")


def _ts_iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- CoinMetrics 指標（BTC 與 ETH 共用） ----------

def _mvrv(asset: str) -> dict:
    series = _series(asset, "CapMVRVCur")
    value = float(series.iloc[-1])
    label = ASSET_NAMES[asset]
    if asset == "BTC":
        reading = "bearish" if value >= 3.7 else "neutral"
        evidence = "A"
        rule = ("≥ 3.7 視為過熱＝偏空，其餘中性（低於 1.0 在原規則裡只是空手後的回場條件，這裡不給方向）。"
                "這是「週期保險」門檻之一：證據很弱（歷史上只有少數幾次），只當警示，不是驗證過的策略。")
    else:
        reading, evidence = "none", "C"
        rule = "3.7 的門檻只用在比特幣；以太幣沒有事前固定的門檻，所以不給方向。"
    return common.indicator(
        id=f"{asset.lower()}_mvrv", group=GROUP, asset=asset, name=f"{label} MVRV（市值 ÷ 已實現市值）",
        value=round(value, 3), unit="倍", display=f"{value:.2f} 倍", asof=_asof(series.index[-1]),
        source=CM_NAME, source_url=CM_PAGE, evidence=evidence, reading=reading, rule=rule,
        note=("MVRV 是「現在的總市值」除以「已實現市值」（每一顆幣用它上次在鏈上移動時的價格計算，"
              "可以想成全體持有人的平均成本）。大於 1 代表平均有賺，越高越多人想獲利了結。"
              "限制：交易所內部的買賣不上鏈，成本只是估計。" + CM_DELAY),
        history=_history(series))


def _realized_price(asset: str) -> dict:
    frame = _cm(asset)[["CapMrktCurUSD", "CapMVRVCur", "SplyCur", "PriceUSD"]].dropna()
    if frame.empty:
        raise ValueError("CoinMetrics 缺少計算已實現價格需要的欄位")
    realized = frame["CapMrktCurUSD"] / frame["CapMVRVCur"] / frame["SplyCur"]
    value, price = float(realized.iloc[-1]), float(frame["PriceUSD"].iloc[-1])
    gap = (price / value - 1) * 100
    label = ASSET_NAMES[asset]
    return common.indicator(
        id=f"{asset.lower()}_realized_price", group=GROUP, asset=asset, name=f"{label}已實現價格（全體平均成本）",
        value=round(value, 2), unit="美元",
        display=f"{value:,.0f} 美元（現價 {price:,.0f} 美元，{'高於' if gap >= 0 else '低於'}平均成本 {abs(gap):.1f}%）",
        asof=_asof(frame.index[-1]), source=CM_NAME + "（自行換算）", source_url=CM_PAGE, evidence="C", reading="none",
        rule="算法：市值 ÷ MVRV ÷ 流通量。沒有事前固定的門檻，不給方向（方向判讀看 MVRV 那一列）。",
        note=("已實現價格是全體持有人的估計平均成本。現價在它上方代表整體有賺，在它下方代表整體賠錢。"
              "限制：和 MVRV 是同一份資訊的另一種寫法，不是獨立證據。" + CM_DELAY),
        history=_history(realized, 2))


def _active_addresses(asset: str) -> dict:
    series = _series(asset, "AdrActCnt")
    average = series.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    change = _change(average, 30)
    label = ASSET_NAMES[asset]
    return common.indicator(
        id=f"{asset.lower()}_active_addresses_7d", group=GROUP, asset=asset, name=f"{label}活躍地址數（7 日平均）",
        value=round(value), unit="個／日", display=f"{value / 1e4:,.1f} 萬個／日（30 天變化 {_pct(change)}）",
        asof=_asof(average.index[-1]), source=CM_NAME, source_url=CM_PAGE, evidence="C", reading="none",
        rule="沒有事前固定的門檻，不給方向。",
        note=("每天有收幣或發幣的不重複地址數，取 7 天平均，用來看鏈上使用熱度。"
              "限制：一個人可以有很多地址，交易所也會合併轉帳，所以不等於使用人數。" + CM_DELAY),
        history=_history(average, 0))


def _exchange_netflow(asset: str) -> dict:
    frame = _cm(asset)[["FlowInExUSD", "FlowOutExUSD"]].dropna()
    if len(frame) < 7:
        raise ValueError("CoinMetrics 交易所流量資料不足 7 天")
    net = (frame["FlowInExUSD"] - frame["FlowOutExUSD"]).rolling(7).sum().dropna()
    value = float(net.iloc[-1])
    inflow, outflow = float(frame["FlowInExUSD"].iloc[-7:].sum()), float(frame["FlowOutExUSD"].iloc[-7:].sum())
    label = ASSET_NAMES[asset]
    word = "淨流入交易所" if value >= 0 else "淨流出交易所"
    return common.indicator(
        id=f"{asset.lower()}_exchange_netflow_7d", group=GROUP, asset=asset, name=f"{label}交易所淨流入（7 日合計）",
        value=round(value), unit="美元",
        display=f"{word} {_yi(abs(value))}（流入 {_yi(inflow)}、流出 {_yi(outflow)}）",
        asof=_asof(net.index[-1]), source=CM_NAME, source_url=CM_PAGE, evidence="C", reading="none",
        rule="正數是淨流入交易所，負數是淨流出。沒有事前固定的門檻，不給方向。",
        note=("最近 7 天轉進交易所的金額減掉轉出的金額。流出通常被解讀為有人把幣提走囤起來，流入被解讀為準備賣，"
              "但證據弱：交易所內部搬錢包、託管與 ETF 調度都會算進去，而且只涵蓋 CoinMetrics 認得出來的交易所地址。"
              + CM_DELAY),
        history=_history(net, 0))


def _exchange_balance(asset: str) -> dict:
    series = _series(asset, "SplyExNtv")
    value = float(series.iloc[-1])
    change7, change30 = _change(series, 7), _change(series, 30)
    label = ASSET_NAMES[asset]
    unit = "顆" if asset == "BTC" else "枚"
    share = ""
    try:
        supply = float(_series(asset, "SplyCur").iloc[-1])
        share = f"，占流通量 {value / supply * 100:.1f}%"
    except Exception:
        pass
    return common.indicator(
        id=f"{asset.lower()}_exchange_balance", group=GROUP, asset=asset, name=f"{label}交易所餘額",
        value=round(value), unit=asset,
        display=f"{value / 1e4:,.1f} 萬{unit}{share}（7 天 {_pct(change7)}、30 天 {_pct(change30)}）",
        asof=_asof(series.index[-1]), source=CM_NAME, source_url=CM_PAGE, evidence="C", reading="none",
        rule="沒有事前固定的門檻，不給方向。",
        note=("放在已知交易所地址裡的幣量。餘額下降通常被解讀為囤幣、賣壓變小，但證據弱："
              "只算 CoinMetrics 認得出來的地址，交易所換錢包或 ETF 託管搬移都會讓數字跳動。" + CM_DELAY),
        history=_history(series, 0))


def _fees(asset: str) -> dict:
    label = ASSET_NAMES[asset]
    source, url, extra = CM_NAME + "（原生幣手續費 × 當日價格）", CM_PAGE, CM_DELAY
    try:
        frame = _cm(asset)[["FeeTotNtv", "PriceUSD"]].dropna()
        if len(frame) < 7:
            raise ValueError("CoinMetrics 手續費資料不足 7 天")
        daily = frame["FeeTotNtv"] * frame["PriceUSD"]
    except Exception:
        if asset != "BTC":
            raise
        reply = common.get_json("https://api.blockchain.info/charts/transaction-fees-usd",
                                {"timespan": f"{common.HISTORY_POINTS + 10}days", "format": "json",
                                 "sampled": "false"})
        daily = pd.Series({_ts_day(point["x"]): float(point["y"]) for point in reply["values"]}).sort_index()
        source, url = "blockchain.info（備援）", "https://www.blockchain.com/explorer/charts/transaction-fees-usd"
        extra = "CoinMetrics 這次抓不到，改用 blockchain.info。"
    average = daily.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    change = _change(average, 30)
    who = "付給礦工" if asset == "BTC" else "付給驗證者，其中一部分會被銷毀"
    return common.indicator(
        id=f"{asset.lower()}_fees_usd_7d", group=GROUP, asset=asset, name=f"{label}每日手續費總額（7 日平均）",
        value=round(value), unit="美元／日", display=f"{value / 1e4:,.1f} 萬美元／日（30 天變化 {_pct(change)}）",
        asof=_asof(average.index[-1]), source=source, source_url=url, evidence="C", reading="none",
        rule="沒有事前固定的門檻，不給方向。",
        note=(f"全網使用者每天付出的手續費（{who}），用來看鏈上擁擠程度。"
              "限制：手續費高低受銘文、空投等短期活動影響很大，和價格方向沒有穩定關係。" + extra),
        history=_history(average, 0))


def _nvt_placeholder() -> dict:
    return common.failed(
        id="btc_nvt", group=GROUP, asset="BTC", name="比特幣 NVT（市值 ÷ 鏈上轉帳金額）", source=CM_NAME,
        source_url=CM_PAGE,
        error="CoinMetrics 社群版沒有開放 NVT（NVTAdj、NVTAdj90 回傳 403），目前沒有其他免費且不需金鑰的來源，所以留白。")


# ---------- mempool.space 指標（只有 BTC） ----------

def _hashrate() -> dict:
    note = ("全網礦機每秒的運算量，1 EH/s 是每秒一百京（10 的 18 次方）次。算力高代表礦工願意投入、網路較安全。"
            "限制：算力是從出塊速度反推的估計值，短期會亂跳；和價格的先後關係沒有可靠證據。")
    try:
        reply = common.get_json(f"{MEMPOOL}/v1/mining/hashrate/3y")
        daily = pd.Series({_ts_day(point["timestamp"]): float(point["avgHashrate"]) / 1e18
                           for point in reply["hashrates"]}).sort_index()
        source, url = MEMPOOL_NAME, "https://mempool.space/zh/graphs/mining/hashrate-difficulty"
    except Exception:
        reply = common.get_json("https://api.blockchain.info/charts/hash-rate",
                                {"timespan": f"{common.HISTORY_POINTS + 10}days", "format": "json",
                                 "sampled": "false"})
        daily = pd.Series({_ts_day(point["x"]): float(point["y"]) / 1e6 for point in reply["values"]}).sort_index()
        source, url = "blockchain.info（備援）", "https://www.blockchain.com/explorer/charts/hash-rate"
        note += "mempool.space 這次抓不到，改用 blockchain.info。"
    average = daily.rolling(7).mean().dropna()
    value = float(average.iloc[-1])
    change = _change(average, 30)
    return common.indicator(
        id="btc_hashrate_7d", group=GROUP, asset="BTC", name="比特幣全網算力（7 日平均）", value=round(value, 1),
        unit="EH/s", display=f"{value:,.0f} EH/s（30 天變化 {_pct(change)}）", asof=_asof(average.index[-1]),
        source=source, source_url=url, evidence="C", reading="none", rule="沒有事前固定的門檻，不給方向。",
        note=note, history=_history(average, 1))


def _difficulty() -> dict:
    reply = common.get_json(f"{MEMPOOL}/v1/difficulty-adjustment")
    change, previous = float(reply["difficultyChange"]), float(reply["previousRetarget"])
    remaining = int(reply["remainingBlocks"])
    when = _ts_day(reply["estimatedRetargetDate"] / 1000)
    return common.indicator(
        id="btc_difficulty_adjustment", group=GROUP, asset="BTC", name="比特幣挖礦難度：下次調整預估",
        value=round(change, 2), unit="%",
        display=(f"預估 {change:+.1f}%（約 {when}，還剩 {remaining:,} 個區塊；本期已走 {float(reply['progressPercent']):.0f}%；"
                 f"上次調整 {previous:+.1f}%）"),
        asof=common.now_utc(), source=MEMPOOL_NAME,
        source_url="https://mempool.space/zh/graphs/mining/hashrate-difficulty",
        evidence="C", reading="none", rule="沒有事前固定的門檻，不給方向。",
        note=("比特幣每 2,016 個區塊（約兩週）自動調整一次挖礦難度，讓出塊維持約 10 分鐘一塊。"
              "預估調降代表最近有礦機關機，調升代表算力增加。限制：預估值在調整前會一直變動。"))


def _recommended_fee() -> dict:
    reply = common.get_json(f"{MEMPOOL}/v1/fees/recommended")
    fast, hour, low = reply["fastestFee"], reply["hourFee"], reply["economyFee"]
    return common.indicator(
        id="btc_fee_rate", group=GROUP, asset="BTC", name="比特幣目前建議手續費", value=fast, unit="聰／虛擬位元組",
        display=f"最快 {fast}、一小時內 {hour}、省錢 {low}（單位：聰／虛擬位元組）", asof=common.now_utc(),
        source=MEMPOOL_NAME, source_url="https://mempool.space/zh/", evidence="C", reading="none",
        rule="沒有事前固定的門檻，不給方向。",
        note=("現在發一筆比特幣交易，想在下一個區塊、一小時內、或不趕時間各要付多少。"
              "「聰」是一億分之一顆比特幣，「虛擬位元組」是交易占用的區塊空間。數字高代表鏈上正在塞車。"
              "限制：這是當下快照，幾分鐘內就可能改變。"))


def _halving() -> list[dict]:
    tip = common.get_json(f"{MEMPOOL}/v1/blocks")[0]
    height, tip_time = int(tip["height"]), float(tip["timestamp"])
    last_height = height // HALVING_INTERVAL * HALVING_INTERVAL
    next_height = last_height + HALVING_INTERVAL
    last_time = float(common.get_json(f"{MEMPOOL}/v1/blocks/{last_height}")[0]["timestamp"])
    since = (tip_time - last_time) / 86400
    remaining = next_height - height
    until = remaining * 10 / 1440
    next_day = _ts_day(tip_time + remaining * 600)
    reward = 50 / 2 ** (height // HALVING_INTERVAL)
    url = "https://mempool.space/zh/"
    note = ("比特幣每 210,000 個區塊（約四年）把新幣發行量砍半。很多人用「減半後第幾天」對照過去幾輪的漲跌，"
            "但過去只有四次減半，樣本太少，只能當背景資訊。")
    return [
        common.indicator(
            id="btc_halving_days_since", group=GROUP, asset="BTC", name="距離上次比特幣減半", value=round(since),
            unit="天", display=(f"{since:,.0f} 天（上次減半 {_ts_day(last_time)}，區塊高度 {last_height:,}；"
                               f"目前高度 {height:,}，每個區塊獎勵 {reward:g} 顆）"),
            asof=_ts_iso(tip_time), source=MEMPOOL_NAME + "（用區塊高度計算）", source_url=url, evidence="C",
            reading="none", rule="沒有事前固定的門檻，不給方向。", note=note),
        common.indicator(
            id="btc_halving_next_estimate", group=GROUP, asset="BTC", name="下次比特幣減半估計", value=round(until),
            unit="天", display=(f"約 {next_day}（還剩 {remaining:,} 個區塊，約 {until:,.0f} 天；"
                               f"區塊高度 {next_height:,}，獎勵降到 {reward / 2:g} 顆）"),
            asof=_ts_iso(tip_time), source=MEMPOOL_NAME + "（用區塊高度計算）", source_url=url, evidence="C",
            reading="none", rule="沒有事前固定的門檻，不給方向。",
            note=note + "估計方式：剩餘區塊數 × 每塊 10 分鐘，實際日期會隨算力快慢前後移動幾週。"),
    ]


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
    _frames.clear()
    rows: list[dict] = []
    builders = (("mvrv", " MVRV（市值 ÷ 已實現市值）", _mvrv),
                ("realized_price", "已實現價格（全體平均成本）", _realized_price),
                ("active_addresses_7d", "活躍地址數（7 日平均）", _active_addresses),
                ("exchange_netflow_7d", "交易所淨流入（7 日合計）", _exchange_netflow),
                ("exchange_balance", "交易所餘額", _exchange_balance),
                ("fees_usd_7d", "每日手續費總額（7 日平均）", _fees))
    for asset in ("BTC", "ETH"):
        for key, name, build in builders:
            _run(rows, lambda build=build, asset=asset: build(asset),
                 [dict(id=f"{asset.lower()}_{key}", asset=asset, name=ASSET_NAMES[asset] + name, source=CM_NAME,
                       source_url=CM_PAGE)])
    rows.append(_nvt_placeholder())
    page = "https://mempool.space/zh/"

    def meta(id: str, name: str) -> dict:
        return dict(id=id, asset="BTC", name=name, source=MEMPOOL_NAME, source_url=page)

    _run(rows, _hashrate, [meta("btc_hashrate_7d", "比特幣全網算力（7 日平均）")])
    _run(rows, _difficulty, [meta("btc_difficulty_adjustment", "比特幣挖礦難度：下次調整預估")])
    _run(rows, _recommended_fee, [meta("btc_fee_rate", "比特幣目前建議手續費")])
    _run(rows, _halving, [meta("btc_halving_days_since", "距離上次比特幣減半"),
                          meta("btc_halving_next_estimate", "下次比特幣減半估計")])
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = collect()
    common.save("onchain", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
