"""Sentiment collector: Fear and Greed index, CoinGecko trending searches, Wikipedia pageviews.

Run: .venv/Scripts/python.exe -m dashboard.collectors.sentiment   (see dashboard/SPEC.md)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from dashboard import common

GROUP = "sentiment"

FNG_API = "https://api.alternative.me/fng/"
FNG_PAGE = "https://alternative.me/crypto/fear-and-greed-index/"
FNG_FEAR, FNG_GREED = 25, 75  # the index's own "extreme" bands, not fitted to data
FNG_LABELS = {"Extreme Fear": "極度恐懼", "Fear": "恐懼", "Neutral": "中性", "Greed": "貪婪",
              "Extreme Greed": "極度貪婪"}

TRENDING_API = "https://api.coingecko.com/api/v3/search/trending"
TRENDING_PAGE = "https://www.coingecko.com/en/highlights/trending-crypto"

WIKI_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/Bitcoin/daily/{start}/{end}")
WIKI_PAGE = "https://pageviews.wmcloud.org/?project=en.wikipedia.org&pages=Bitcoin&range=latest-90"
WIKI_SHORT, WIKI_LONG = 7, 90


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def fear_greed() -> dict:
    meta = dict(id="market_fear_greed", group=GROUP, asset="MARKET", name="恐懼與貪婪指數",
                source="alternative.me", source_url=FNG_PAGE)
    try:
        rows = common.get_json(FNG_API, {"limit": 0})["data"]  # limit=0 returns the whole history
        history = sorted([_day(int(r["timestamp"])), int(r["value"])] for r in rows)
        latest = max(rows, key=lambda r: int(r["timestamp"]))
        value = int(latest["value"])
        label = FNG_LABELS.get(latest.get("value_classification", ""), "")
        reading = "bullish" if value <= FNG_FEAR else "bearish" if value >= FNG_GREED else "neutral"
        return common.indicator(
            **meta, value=value, unit="分（0 到 100）",
            display=f"{value} 分" + (f"（{label}）" if label else ""),
            asof=datetime.fromtimestamp(int(latest["timestamp"]), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            evidence="C", reading=reading,
            rule=(f"≤ {FNG_FEAR} 分是極度恐懼，當作偏多（反向指標：大家最怕的時候，往往離低點不遠）；"
                  f"≥ {FNG_GREED} 分是極度貪婪，當作偏空；其餘算中性。"
                  "門檻是這個指數自己的分級，不是看資料調出來的。"),
            note=("把波動、成交量、社群熱度、比特幣市值占比等合成 0 到 100 分，分數越低代表市場越害怕。"
                  "限制：配方由網站自訂、不完全公開，而且極度恐懼可以持續很久；單獨使用的證據很弱，只能當參考。"),
            history=history)
    except Exception as exc:  # noqa: BLE001 - one bad source must not take the others down
        return common.failed(**meta, error=f"{type(exc).__name__}: {exc}")


def trending() -> dict:
    meta = dict(id="market_trending_search", group=GROUP, asset="MARKET", name="熱門搜尋幣種（CoinGecko）",
                source="CoinGecko", source_url=TRENDING_PAGE)
    try:
        coins = common.get_json(TRENDING_API)["coins"]
        names = [f'{c["item"]["name"]}（{str(c["item"]["symbol"]).upper()}）' for c in coins][:10]
        if not names:
            raise ValueError("回傳的名單是空的")
        return common.indicator(
            **meta, value=None, unit="", display="、".join(names), asof=common.now_utc(),
            evidence="C", reading="none",
            rule="只列名單，不做偏多或偏空的判讀：沒有事前固定、說得出門檻的規則。",
            note=("過去 24 小時在 CoinGecko 網站上最多人搜尋的幣，由熱到冷排列。"
                  "可以看出散戶現在在注意什麼；榜上若都是沒聽過的小幣，通常代表投機氣氛較濃。"
                  "限制：只反映這一個網站的使用者，也沒有歷史數列。"),
            history=[])
    except Exception as exc:  # noqa: BLE001
        return common.failed(**meta, error=f"{type(exc).__name__}: {exc}")


def wikipedia() -> dict:
    meta = dict(id="btc_wikipedia_attention", group=GROUP, asset="BTC", name="維基百科「Bitcoin」條目瀏覽量（散戶關注度）",
                source="Wikimedia 瀏覽量統計", source_url=WIKI_PAGE)
    try:
        today = datetime.now(timezone.utc)
        start = today - timedelta(days=common.HISTORY_POINTS + WIKI_LONG + 10)
        items = common.get_json(WIKI_API.format(start=start.strftime("%Y%m%d"), end=today.strftime("%Y%m%d")))["items"]
        views = pd.Series({pd.Timestamp(i["timestamp"][:8]): float(i["views"]) for i in items}).sort_index()
        if len(views) < WIKI_LONG:
            raise ValueError(f"只有 {len(views)} 天資料，不夠算 {WIKI_LONG} 日平均")
        short = views.rolling(WIKI_SHORT).mean()
        ratio = (short / views.rolling(WIKI_LONG).mean()).dropna()
        value = round(float(ratio.iloc[-1]), 3)
        history = [[d.strftime("%Y-%m-%d"), round(float(v), 3)] for d, v in ratio.items()]
        return common.indicator(
            **meta, value=value, unit="倍",
            display=f"{value:.2f} 倍（近 {WIKI_SHORT} 日平均每天約 {short.iloc[-1]:,.0f} 次）",
            asof=ratio.index[-1].strftime("%Y-%m-%dT00:00:00Z"),
            evidence="C", reading="none",
            rule=(f"數值是近 {WIKI_SHORT} 日平均瀏覽量除以近 {WIKI_LONG} 日平均。大於 1 代表關注度升溫，小於 1 代表降溫。"
                  "不做偏多或偏空的判讀：沒有公認、事前固定的門檻。"),
            note=("英文維基百科「Bitcoin」條目每天被真人點開的次數（已排除機器人），當作一般大眾關注度的替代資料。"
                  "限制：關注度暴增可能出現在大漲也可能出現在大跌，本身不分方向；資料通常慢一到兩天。"),
            history=history)
    except Exception as exc:  # noqa: BLE001
        return common.failed(**meta, error=f"{type(exc).__name__}: {exc}")


def collect() -> list[dict]:
    return [fear_greed(), trending(), wikipedia()]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")  # a coin name the console cannot encode must not crash the run
    rows = collect()
    common.save("sentiment", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f'（歷史 {len(r["history"])} 筆）')
