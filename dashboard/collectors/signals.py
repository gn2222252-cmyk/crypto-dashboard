"""The trend rules that held up in the dashboard's studies, turned into daily signals with their track record.

Written to cache/_signals.json and shown on the "順勢訊號" tab. For each asset and each rule:
- what the rule wants to hold for the next day, since when, and the close that would change that;
- its record from the start of the comparison window: yearly return, deepest loss, years beating holding,
  trades won (for on/off rules), and how much of each big fall (30% or more) it took.

These are descriptions of what fixed rules did and what state they are in now. They are not forecasts and not
advice. The rules are the ones the studies under dashboard/studies2 tested; the robustness notes below summarise
those studies (clock_robustness, robust_trend, loss_control, trend_assets) and must be revisited if they change.

Sources (public, no key): blockchain.info and Binance daily closes, Yahoo Finance for MSTR.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

PATH = common.CACHE / "_signals.json"
COST = 0.001
FALL = 0.30  # a "big fall": the close drops this much from a peak before rebounding as much from the low
CHART_DAYS = 3 * 365

ASSETS = (  # key, name, loader, comparison start, days per year
    ("BTC", "比特幣", "btc", "2018-01-01", 365),
    ("ETH", "以太幣", "ETHUSDT", None, 365),
    ("SOL", "索拉納", "SOLUSDT", None, 365),
    ("MSTR", "MSTR（Strategy 公司股票）", "MSTR", "2020-08-11", 252),
)

# Each rule's explanation, how to read it, and the checks it went through. "verified": True means it held up in
# the dashboard's studies (see the study ids in brackets); the marks are "pass", "weak" or "fail". These are our
# readings of the study results and must be revisited when a study changes.
USE_COMMON = [
    "每天台灣早上 8 點以後（加密貨幣日線收盤後；美股是美股收盤後）看一次這一頁，不用盯盤。",
    "看「隔天」那一欄：規則說持有幾成，就是它的狀態；它沒說要做的日子什麼都不做。",
    "「什麼時候會改變」那一欄告訴你收盤要到哪個價位，狀態才會翻轉。盤中碰到不算，要看收盤。",
    "要有效，就要每一次都照做，包括連續賠了幾次之後的下一次。獲利集中在少數幾筆，跳過的那一次可能剛好是它。",
    "它們穩定做到的是大跌時少虧；報酬能不能多，要看哪一條、從什麼時候開始做。下面每條都寫了換起算日、換收盤時間之後的範圍。",
]
# Numbers below come from the studies named in brackets. "honest" is the range over 72 start months (2018-01 to
# 2023-12) x 24 closing hours from main_signal_check; the single-start record on the page is shown next to it.
RULES = (
    {"id": "ma50", "name": "收盤高於 50 日均線才持有", "short": "50 日均線", "verified": True,
     "how": "收盤價高於最近 50 天收盤的平均就全部持有，低於就全部空手。",
     "steps": ["每天收盤後看一次：收盤價高於最近 50 天收盤的平均 → 隔天全部持有（一次全部買，不分批）。",
               "收盤價低於最近 50 天收盤的平均 → 隔天全部賣出。",
               "價格在均線附近來回時會反覆進出、常常小賠，這是正常的，照做就好。",
               "手續費要低（每次 0.1% 左右）；手續費到 1% 優勢就沒了。"],
     "use": ["只有兩種狀態：全部持有或全部空手。比特幣一年大約進出 10 次。",
             "勝率很低（比特幣約四次只有一次賺），完全靠少數幾段大漲把小賠全部賺回來。",
             "它最深的虧損比其他順勢規則深（一般約 −50% 到 −60%，最差 −71%），但仍比一直抱著（−77% 到 −83%）淺。"],
     "honest": "換 72 個起算月 × 24 個收盤時間共 1,728 種情況：每年比一直抱著多（中位數）+16.9 個百分點，中間八成在 +7.6 到 +24.0；最深虧損 −26% 到 −71%（一直抱著 −53% 到 −81%）。照一般人方式做，八成情況仍贏過一直抱著。",
     "why": "它是四種檢驗一起考、最差那一項仍然最好的規則：108 個起算月全部贏過一直抱著；1,728 種起算日 × 收盤時間的情況裡 99% 贏；照一般人方式做（晚看、隔天做、手續費較高、偶爾漏做）80% 贏；換到以太幣、索拉納、幣安幣、瑞波幣、MSTR，79% 的起算點贏。勝率低是因為它很早就跟上漲、很早就離開下跌，所以常被洗出來小賠，但每一段大漲都在場。",
     "edge": "在各種起算日、收盤時間、執行方式下，報酬都最穩定地贏過一直抱著。",
     "weak": "最深虧損較深（中位數約 −54%）；勝率只有兩到三成，要忍受連續小賠；對手續費敏感；它是 17 條規則裡事後看最好的一條，這一點仍帶有事後挑選的成分；在股票等傳統市場上報酬多半輸給一直抱著；瑞波幣上也不好。",
     "tests": [
         ["換起算日（108 個起算月）", "108 個全部報酬贏過一直抱著，每年多（中位數）+16.8 個百分點（main_signal_check）", "pass"],
         ["換起算日 × 換收盤時間（1,728 種）", "99% 贏過一直抱著；每年多 +16.9（中間八成 +7.6 到 +24.0）（main_signal_check）", "pass"],
         ["照一般人方式做", "晚上 10 點看、隔天早上做、手續費 0.25%、漏掉一成：80% 的模擬贏過一直抱著（main_signal_check）", "pass"],
         ["換成別的幣和 MSTR", "271 個起算點裡 214 個（79%）報酬贏過一直抱著；瑞波幣 61 個裡只有 18 個（main_signal_check）", "pass"],
         ["虧損比一直抱著淺", "1,728 種情況全部較淺，但中位數 −54%、最差 −71%", "pass"],
         ["加回後來下市的幣", "比特幣跌破 50 日均線就把整籃幣換成現金，加回下市的幣後每年 +36.1%，整籃抱著 −0.3%（survivorship）", "pass"],
         ["手續費 1%", "每年剩 +17.3%，輸給一直抱著（robust_trend）", "weak"],
         ["換到傳統市場（14 個）", "12 個虧損較淺，只有 4 個最後的錢比抱著多（trend_world）", "weak"]]},
    {"id": "b5520", "name": "55 日新高進、20 日新低出", "short": "55／20", "verified": True,
     "how": "收盤是最近 55 天最高就全部持有；持有中收盤是最近 20 天最低就全部賣出。其他日子不動。",
     "steps": ["空手時：收盤價是最近 55 天收盤裡最高的 → 隔天一次全部買進。",
               "持有時：收盤價是最近 20 天收盤裡最低的 → 隔天一次全部賣出。",
               "其他日子什麼都不做。"],
     "use": ["只有兩種狀態：全部持有或全部空手。一年大約進出 3 次。",
             "它最擅長的是少虧：各種起算日、收盤時間下，最深虧損中位數約 −35%，是所有規則裡最淺的。",
             "但它的報酬優勢不可靠：換起算日之後，108 個起算月只有 58 個贏過一直抱著。"],
     "honest": "換 1,728 種起算日 × 收盤時間：每年比一直抱著多（中位數）只有 +3.2 個百分點，中間八成在 −7.9 到 +11.7；最深虧損 −23% 到 −60%（一直抱著 −53% 到 −81%）。照一般人方式做，只有 45% 的情況贏過一直抱著。之前頁面引用的「+33.6% 對 +19.1%、最深 −35.7%」是從 2018-01-01 起算，剛好在 2017 年高點後兩週，數字被美化了約 11 個百分點。",
     "why": "它站得住的只有「少虧」這一點：所有起算日、所有收盤時間、所有模擬裡，最深虧損都比一直抱著淺，而且是各規則裡最淺的。報酬上它和一直抱著差不多。",
     "edge": "最深虧損最淺（中位數 −35%）；一年只動約 3 次。",
     "weak": "報酬優勢不可靠（中位數只多 3 個百分點，一般人照做多半輸）；用在 MSTR 上 40 個起算點只有 2 個贏。",
     "tests": [
         ["換起算日（108 個起算月）", "58 個報酬贏過一直抱著，中位數只多 +0.6 個百分點；最深虧損 −36% 到 −55%（main_signal_check）", "weak"],
         ["換起算日 × 換收盤時間（1,728 種）", "66% 贏；中位數 +3.2；最深虧損中位數 −34.6%，各規則最淺（main_signal_check）", "weak"],
         ["照一般人方式做", "45% 贏過一直抱著（main_signal_check）", "fail"],
         ["虧損比一直抱著淺", "所有情況全部較淺，而且是各規則裡最淺的", "pass"],
         ["和 10 種改良版比", "沒有一種改良在事先挑、換資產、換收盤時間三關都贏過原版（rule_upgrades）", "pass"],
         ["放進多市場組合", "加進 10 個傳統市場的順勢組合，風險調整後從 0.91 升到 1.27（trend_portfolio）", "pass"]]},
    {"id": "blend", "name": "17 條順勢規則混合", "short": "混合", "verified": True,
     "how": "同時看 17 條簡單的順勢規則，有幾條說「持有」，就持有幾成（例如 6 條說持有就持有約 35%）。這 17 條是：收盤高於 10、20、30、50、75、100、150、200 日均線（8 條）；創 10、20、50、100 日新高進場，跌破一半天數的新低出場（4 條）；收盤高於 14、30、60、90、180 天前的價格（5 條）。每天收盤看一次。",
     "use": ["隔天持有的比例 = 17 條裡說持有的條數 ÷ 17。比例變動不到一成可以不調整。",
             "簡化版：17 條裡 9 條以上說持有就全部持有，否則全部空手。簡化版在 108 個起算月裡 104 個贏過一直抱著，比分批的原版更穩。"],
     "honest": "換 1,728 種起算日 × 收盤時間：每年多（中位數）+5.0 個百分點，中間八成 −3.2 到 +13；最深虧損 −26% 到 −56%。照一般人方式做只有 42% 贏過一直抱著。簡化版（9 條以上全買）：+14.3 個百分點、一般人照做 69% 贏、最深 −28% 到 −62%。",
     "why": "17 條一起看，不用挑參數，各條離場時間不同，所以虧損比多數單條規則淺。但分批調整很頻繁，一般人照做時優勢容易被吃掉。",
     "edge": "不用挑參數；虧損比多數單條淺。",
     "weak": "照一般人方式做多半輸給一直抱著；之前說它「最穩」只對在換收盤時間這一項，換起算日和執行方式後並不是最穩。",
     "tests": [
         ["換起算日（108 個起算月）", "94 個贏，中位數 +5.3（main_signal_check）", "pass"],
         ["換起算日 × 換收盤時間", "82% 贏，中位數 +5.0（main_signal_check）", "pass"],
         ["照一般人方式做", "42% 贏過一直抱著（main_signal_check）", "fail"],
         ["簡化版（9 條以上全買）照一般人方式做", "69% 贏，中位數 +5.6（main_signal_check）", "pass"],
         ["加回後來下市的幣", "388 個幣裡 355 個照混合最後的錢較多（survivorship）", "pass"]]},
    {"id": "vol40", "name": "混合，再依波動調整持有多少", "short": "混合＋波動", "verified": True,
     "how": "先照「17 條混合」決定幾成，市場很激烈時再少拿一點：持有比例乘上「40% ÷ 最近 30 天的年化波動」（最多乘 1，不借錢）。",
     "use": ["適合最怕大虧的人：最差的一種情況下最深也只虧 −42%。",
             "代價是報酬多半輸給一直抱著。"],
     "honest": "換 1,728 種起算日 × 收盤時間：每年比一直抱著少（中位數）1.8 個百分點；只有 42% 贏；最深虧損 −22% 到 −42%，最差的情況是各規則裡最淺的。",
     "why": "它站得住的只有少虧，而且最差情況最淺；報酬是用來換低風險的。",
     "edge": "最差情況下虧損最淺。",
     "weak": "報酬多半輸給一直抱著（108 個起算月只有 28 個贏）。",
     "tests": [
         ["換起算日（108 個起算月）", "28 個報酬贏，中位數 −5.6（main_signal_check）", "fail"],
         ["虧損比一直抱著淺", "所有情況全部較淺，最差的一格 −41.6%，各規則裡最淺（main_signal_check）", "pass"]]},
    {"id": "c2010", "name": "20 日新高進、10 日新低出", "short": "20／10", "verified": False,
     "how": "收盤是最近 20 天最高就持有；持有中收盤是最近 10 天最低就賣出。",
     "use": [], "why": "", "tests": [], "steps": [], "honest": "",
     "edge": "虧損比一直抱著淺。",
     "weak": "報酬優勢看運氣：換一個收盤時間，24 個裡只有 12 個贏；照一般人方式做會輸給一直抱著。"},
)
# Rows: name, claim, what was tested, why it is not used, category ("無效" = the data shows it does not work;
# "證據不足" = looked positive or mixed but too few cases to tell; "已收回" = recommended earlier, withdrawn).
FAILED = (
    ["55 日新高進、20 日新低出 當主訊號", "之前當成主訊號", "換起算日、換收盤時間、照一般人方式做", "之前引用的數字從 2017 年高點後兩週起算，被美化約 11 個百分點；換起算日後報酬優勢只有 +3 點，現在只當「最少虧」的參考", "已收回"],
    ["20 日新高進、10 日新低出（單一一條）", "事先挑參數時九年都挑到它", "換 24 個收盤時間、照一般人方式做", "24 個時間裡只有 12 個報酬贏過一直抱著；一般人照做 1,000 次有 718 次輸", "已收回"],
    ["機器學習預測漲跌", "準確率五成五以上", "74 個指標、每季重訓、嚴格不偷看未來", "猜對 100 次裡 48 到 51 次，和永遠猜漲差不多；最有信心的時候反而最不準", "無效"],
    ["KD 低檔黃金交叉、RSI 超賣、布林下緣買", "跌深就會反彈，勝率高", "6 個市場", "勝率很高，但報酬在 6 個市場一個都沒贏過一直抱著；KD 低檔交叉和布林下緣在比特幣上是賠錢的", "無效"],
    ["主動買賣力道、累積買賣差（CVD）", "主動買盤多就會漲、價漲量縮會跌", "8 年資料、14 種分組、換幣", "看不出之後漲跌；「價漲但買賣差減」之後反而漲更多", "無效"],
    ["時段、星期、月份規律", "某些時間比較會漲", "24 項，前後兩半和逐年檢查", "扣掉交易成本後全部不剩", "無效"],
    ["隔夜與特定時段策略", "公開的隔夜策略每年賺三成以上", "照原文重做、看發表之後", "發表之後每年 −2.6%；賺的其實是「創新高才持有」", "無效"],
    ["81 個指標偏到極端", "指標極端就會反轉", "每個指標偏高、偏低之後 30 天", "可檢驗的 53 種裡顯著 2 種，純靠運氣也會有約 3 種", "無效"],
    ["以太幣／比特幣換著拿", "看比值趨勢決定拿哪個", "事先挑、換其他幣、換收盤時間", "八年只贏兩年；24 個時間裡只有 4 個贏", "無效"],
    ["4 小時線等較快的順勢", "看短一點的 K 線反應比較快", "29 種設定、事先挑、換幣", "事先挑 8 年只贏日線一年；越快越被手續費吃掉", "無效"],
    ["賣出買權（備兌）、買賣權保險", "收權利金或買保險擋大跌", "模型估價、和「一半比特幣一半現金」比", "賣買權等於只拿一半比特幣，和一半一半幾乎一樣；買保險一年保費約 36%，最深虧損幾乎沒變", "無效"],
    ["資金費率套利", "買現貨空合約收費率", "2020 年至今", "最近一年約 +3.5%，低於美元定存", "無效"],
    ["恐懼指數很低就買", "別人恐懼時貪婪", "照規則買賣、和一直抱著比", "2022 年以來每年約 −8%，一直抱著 +7.2%", "無效"],
    ["週期頂部指標（固定門檻、滾動門檻）", "指標過熱就是頂", "照原本門檻、改成滾動門檻、換起算年", "從 2018 年起算看起來贏，從 2014 或 2016 年起算全部輸；2025 年固定門檻一條都沒觸發", "證據不足"],
    ["康波週期（及其他長週期）", "照 50 年長波決定持股或持債", "1871 年以來美股", "只有約 3 次長波，統計上分不出和隨機；照它進出每年 +4.6%，一直持股 +7.1%", "證據不足"],
    ["期貨溢價過高時出場", "期貨比現貨貴太多就快崩了", "13 段高溢價、加在順勢上、換到以太幣", "比特幣上有幫助（風險調整後 0.95 → 1.25），但只有 13 段、2023 年後 3 段，以太幣上反而變差", "證據不足"],
    ["資金費率很負就買、費率過熱就不買", "空單擁擠會軋空、過熱會跌", "44 組參數、前後兩半", "前半段排第 1、2，後半段排第 28、34；極端值只有 4 段", "證據不足"],
    ["OBV 濾網", "成交量確認趨勢", "加在順勢上、24 個收盤時間、換幣", "加在 55／20 上事先挑有小贏，但 81% 的時間和 50 日均線是同一件事，換幣 4 個只贏 2 個", "證據不足"],
    ["殖利率急升時不持有", "利率衝擊會打壓風險資產", "19 種總經濾網、24 個收盤時間、換幣", "比特幣上 4 個起算年都小贏、24 個時間全贏，但索拉納上大輸，真正的利率衝擊只有 4 到 6 段", "證據不足"],
    ["礦工投降訊號（Hash Ribbons）", "礦工撐不住就是底", "28 次訊號、和同年隨機日期比", "和同年隨機日期比一年後贏 58%（2019 年後 82%），但獨立訊號約 24 次，不顯著", "證據不足"],
    ["預測市場偏向", "群眾覺得要跌就會跌", "13 個月", "機率本身報得準，但只有 13 個月，偏向有沒有用分不出來", "證據不足"],
    ["交易所資金流入", "大量搬進交易所就是要賣", "重現別人的研究、檢查資料", "歷史資料是事後重算的，和每天即時公布的對不上，結果不能信", "證據不足"],
    ["多幣順勢籃子、挑最強的幣", "分散到很多幣更好", "包含下市幣的幣單、事先挑、換倉日", "7 種籃子沒有一種三關都過；挑強這一步沒有加分", "無效"],
)
PRIMARY = "ma50"  # chosen by main_signal_check's pre-stated rule: best worst-case share beating holding across four checks
EXTRA = {
    "ma50": {"entry": "收盤高於 50 日均線全部持有，低於全部賣",
             "clock": "+16.9（中間八成 +7.6 到 +24.0）", "ordinary": "80% 贏",
             "fit": "想要報酬最穩定贏過一直抱著；能忍受連續小賠和較深的虧損"},
    "b5520": {"entry": "創 55 日新高全部買、跌破 20 日新低全部賣",
              "clock": "+3.2（中間八成 −7.9 到 +11.7）", "ordinary": "45% 贏",
              "fit": "最在意少虧、不在意報酬多不多"},
    "blend": {"entry": "17 條裡有幾條說持有就持有幾成；簡化版 9 條以上全買",
              "clock": "+5.0（簡化版 +14.3）", "ordinary": "42% 贏（簡化版 69%）",
              "fit": "不想挑參數；建議用簡化版"},
    "vol40": {"entry": "先照 17 條算比例，市場越劇烈拿越少",
              "clock": "−1.8", "ordinary": "20% 贏",
              "fit": "最怕大虧，接受報酬輸給一直抱著"},
    "c2010": {"entry": "創 20 日新高全部買、跌破 10 日新低全部賣",
              "clock": "+11.8% 到 +30.6%（24 個裡只有 12 個贏）", "ordinary": "1,000 次裡 718 次輸",
              "fit": "已收回", "retracted": True},
}
for _rule in RULES:
    _rule.update(EXTRA.get(_rule["id"], {}))
RULE_IDS = tuple(r["id"] for r in RULES)


def _f(v, digits: int = 1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _load(loader: str) -> pd.Series:
    if loader == "btc":
        return te._daily_close()
    if loader.endswith("USDT"):
        return ta._crypto_close(loader)
    return ta._share_close(loader)


def _breakout(close: pd.Series, n: int, m: int) -> pd.Series:
    high = close.rolling(n, min_periods=n).max()
    low = close.rolling(m, min_periods=m).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0).where(high.notna())


def _targets(close: pd.Series, year: int) -> pd.DataFrame:
    """What each rule wants to hold, decided at each close (1 = all in, 0 = cash, fractions for the blends)."""
    signal, _ = te._signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    vol = close.pct_change().rolling(30, min_periods=30).std(ddof=1) * math.sqrt(year)
    scale = (0.40 / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
    mean50 = close.rolling(50, min_periods=50).mean()
    return pd.DataFrame({"blend": share, "vol40": share * scale, "b5520": _breakout(close, 55, 20),
                         "ma50": (close > mean50).astype(float).where(mean50.notna()),
                         "c2010": signal["C20"]}), signal


def _part_name(col: str) -> str:
    """Plain name of one of the 17 rules inside the blend."""
    kind, n = col[0], int(col[1:])
    if kind == "A":
        return f"收盤高於 {n} 日均線"
    if kind == "C":
        return f"創 {n} 日新高進場、跌破 {n // 2} 日新低出場"
    return f"收盤高於 {n} 天前的價格"


def _flip_prices(close: pd.Series, signal: pd.DataFrame) -> dict:
    """For each simple rule: the next close that would switch it, given its current state."""
    c = close.values
    out = {}
    for col in signal.columns:
        state = signal[col].iloc[-1]
        if not np.isfinite(state):
            continue
        kind, n = col[0], int(col[1:])
        if kind == "A":  # close > mean of the last n closes including the new one  <=>  close > sum(last n-1) / (n-1)
            level = c[-(n - 1):].sum() / (n - 1)
        elif kind == "C":
            level = c[-(n - 1):].max() if state < 0.5 else c[-(n // 2 - 1):].min()
        else:  # close above the close n days before the new day
            level = c[-n]
        out[col] = (float(level), bool(state >= 0.5))
    return out


def _falls(close: pd.Series) -> list[tuple]:
    """(peak day, low day) of every fall of FALL or more, ended by a rebound of FALL from the low."""
    out, peak_i, low_i, falling = [], 0, 0, False
    v = close.values
    for i in range(1, len(v)):
        if not falling:
            if v[i] > v[peak_i]:
                peak_i = i
            elif v[i] <= v[peak_i] * (1 - FALL):
                falling, low_i = True, i
        else:
            if v[i] < v[low_i]:
                low_i = i
            elif v[i] >= v[low_i] * (1 + FALL):
                out.append((close.index[peak_i], close.index[low_i]))
                falling, peak_i = False, i
    if falling:
        out.append((close.index[peak_i], close.index[low_i]))
    return out


def _record(close: pd.Series, target: pd.Series, start: pd.Timestamp, year: int, binary: bool) -> dict:
    change = close.pct_change().fillna(0.0)
    position = target.shift(1).fillna(0.0)
    net, trade = te._run(position.values, change.values, COST)
    net = pd.Series(net, index=close.index)[close.index >= start]
    hold = change[close.index >= start]
    pos = position[close.index >= start]

    def stats(r: pd.Series) -> tuple:
        growth = float(np.prod(1 + r.values))
        ret = (growth ** (year / len(r)) - 1) * 100 if growth > 0 else -100.0
        eq = np.concatenate([[1.0], np.cumprod(1 + r.values)])
        return ret, float((eq / np.maximum.accumulate(eq) - 1).min() * 100)

    ret, deep = stats(net)
    hret, hdeep = stats(hold)
    years_rule = {y: float(np.prod(1 + g.values) - 1) for y, g in net.groupby(net.index.year)}
    years_hold = {y: float(np.prod(1 + g.values) - 1) for y, g in hold.groupby(hold.index.year)}
    beat = sum(years_rule[y] > years_hold[y] for y in years_rule)
    falls, taken = [], []
    for peak, low in _falls(close[close.index >= start]):
        span = (net.index > peak) & (net.index <= low)
        if span.sum() < 2:
            continue
        falls.append(float(close[low] / close[peak] - 1) * 100)
        taken.append(float(np.prod(1 + net[span].values) - 1) * 100)
    rec = {"start": start.strftime("%Y-%m-%d"), "ret": _f(ret), "deep": _f(deep), "hold_ret": _f(hret),
           "hold_deep": _f(hdeep), "years": len(years_rule), "beat": int(beat), "share": _f(pos.mean() * 100, 0),
           "falls": len(falls), "fall_avg": _f(np.mean(falls)) if falls else None,
           "fall_rule": _f(np.mean(taken)) if taken else None}
    if binary:
        p = pos.values
        wins = trades = 0
        results = []
        i = 0
        while i < len(p):
            if p[i] > 0.5 and (i == 0 or p[i - 1] < 0.5):
                j = i
                while j + 1 < len(p) and p[j + 1] > 0.5:
                    j += 1
                if j + 1 < len(p):  # closed trades only
                    trades += 1
                    gain = float(np.prod(1 + net.values[i:j + 1])) - 1
                    wins += gain > 0
                    results.append(gain * 100)
                i = j + 1
            else:
                i += 1
        rec["trades"], rec["wins"] = int(trades), int(wins)
        won = [g for g in results if g > 0]
        lost = [g for g in results if g <= 0]
        rec["avg_win"] = _f(np.mean(won)) if won else None
        rec["avg_loss"] = _f(np.mean(lost)) if lost else None
        rec["best"] = _f(max(results)) if results else None
        top5 = sorted(results, reverse=True)[:5]
        rec["best5_sum"] = _f(sum(top5)) if top5 else None
        rec["trades_per_year"] = _f(trades / (len(p) / year), 1)
    else:
        rec["turnover"] = _f(trade[-len(net):].sum() / (len(net) / year), 1)
    return rec


def _since(target: pd.Series) -> str | None:
    """The first day of the current run of the same state (all in / cash / partly)."""
    t = target.dropna()
    if t.empty:
        return None
    state = np.where(t.values >= 0.999, 2, np.where(t.values <= 0.001, 0, 1))
    i = len(state) - 1
    while i > 0 and state[i - 1] == state[-1]:
        i -= 1
    return t.index[i].strftime("%Y-%m-%d")


def _asset(key: str, name: str, loader: str, start: str | None, year: int) -> dict:
    close = _load(loader)
    targets, signal = _targets(close, year)
    first = targets["blend"].first_valid_index()
    begin = max(pd.Timestamp(start) if start else first, first)
    flips = _flip_prices(close, signal)
    last = float(close.iloc[-1])
    rules = {}
    for rule in RULE_IDS:
        t = targets[rule]
        now = _f(t.iloc[-1] * 100, 0)
        item = {"share": now, "since": _since(t), "record": _record(close, t, begin, year, rule in ("b5520", "ma50", "c2010"))}
        if rule == "ma50":
            level = float(close.values[-49:].sum() / 49)
            item["flip"] = {"level": level, "dir": "below" if t.iloc[-1] >= 0.5 else "above",
                            "pct": _f((level / last - 1) * 100)}
        elif rule in ("b5520", "c2010"):
            n, m = (55, 20) if rule == "b5520" else (20, 10)
            long = t.iloc[-1] >= 0.5
            level = float(close.values[-(m - 1):].min() if long else close.values[-(n - 1):].max())
            item["flip"] = {"level": level, "dir": "below" if long else "above", "pct": _f((level / last - 1) * 100)}
        rules[rule] = item
    long_rules = sum(1 for v in flips.values() if v[1])
    parts = sorted(({"rule": col, "name": _part_name(col), "long": is_long, "level": _f(lv, 4),
                     "pct": _f((lv / last - 1) * 100)} for col, (lv, is_long) in flips.items()),
                   key=lambda p: abs(p["pct"]) if p["pct"] is not None else 1e9)
    hist = close[close.index >= close.index[-1] - pd.Timedelta(days=CHART_DAYS)]
    blend = targets["blend"].reindex(hist.index)
    return {"key": key, "name": name, "asof": close.index[-1].strftime("%Y-%m-%d"), "price": _f(last, 4),
            "rules": rules, "blend_long": long_rules, "blend_total": len(flips),
            "parts": parts, "nearest": parts[:3],
            "history": {"price": [[d.strftime("%Y-%m-%d"), _f(v, 4)] for d, v in hist.items()],
                        "blend": [[d.strftime("%Y-%m-%d"), _f(v * 100, 0)] for d, v in blend.items() if np.isfinite(v)]}}


LOG = common.STORE / "signals_live_log.json"
LOG_START = "2026-11-17"  # rules frozen on 2026-11-18 (first logged close: 2026-11-17); results come only from this append-only log


def _live_log(assets: list[dict]) -> dict:
    """Append today's frozen-rule states and closes to the log (never rewrite past days) and score the log.

    Each day stores, per asset, the last close and what each rule wanted to hold for the next day. The live record
    of a rule is the compounded return of holding its logged share from one logged day to the next (no costs), next
    to holding the asset over the same days. Nothing here is recomputed from history."""
    try:
        log = json.loads(LOG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file starts a new log
        log = {}
    for a in assets:
        day = a.get("asof")
        if not day or day < LOG_START:
            continue
        book = log.setdefault(a["key"], {})
        if day not in book:  # append only
            book[day] = {"price": a.get("price"),
                         "shares": {r: (a.get("rules", {}).get(r) or {}).get("share") for r in RULE_IDS}}
    common.STORE.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    out = {"start": LOG_START, "assets": {}}
    for key, book in log.items():
        days = sorted(book)
        rules = {}
        for r in RULE_IDS:
            growth = 1.0
            for prev, cur in zip(days, days[1:]):
                p0, p1 = book[prev].get("price"), book[cur].get("price")
                w = (book[prev].get("shares") or {}).get(r)
                if p0 and p1 and w is not None:
                    growth *= 1 + (w / 100) * (p1 / p0 - 1)
            rules[r] = _f((growth - 1) * 100, 2)
        first, last = book[days[0]].get("price"), book[days[-1]].get("price")
        out["assets"][key] = {"days": len(days), "first": days[0], "last": days[-1], "rules": rules,
                              "hold": _f((last / first - 1) * 100, 2) if first and last else None}
    return out


def collect() -> list[dict]:
    try:
        previous = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        previous = {}
    old = {a.get("key"): a for a in previous.get("assets", []) if isinstance(a, dict)}
    payload = {"collected_utc": common.now_utc(), "rules": list(RULES), "use": USE_COMMON, "primary": PRIMARY,
               "failed": [list(row) for row in FAILED], "assets": [], "errors": {}}
    for key, name, loader, start, year in ASSETS:
        try:
            item = _asset(key, name, loader, start, year)
            json.dumps(item, allow_nan=False)
            payload["assets"].append(item)
        except Exception as exc:  # noqa: BLE001 - one asset failing keeps its previous result
            payload["errors"][key] = f"{type(exc).__name__}: {exc}"[:200]
            if key in old:
                payload["assets"].append(old[key])
            print(f"  訊號 {key} 這次沒有更新：{payload['errors'][key]}")
    payload["live"] = _live_log(payload["assets"])
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for a in data["assets"]:
        print(a["key"], a["price"], a["blend_long"], "/", a["blend_total"], a["nearest"])
        for r, v in a["rules"].items():
            print("  ", r, v["share"], v["since"], v.get("flip"), json.dumps(v["record"], ensure_ascii=False))
    print("errors", data["errors"])
