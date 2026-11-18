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
    "要有效，就要每一次都照做，包括連續賠了幾次之後的下一次。研究裡獲利集中在少數幾筆，跳過的那一次可能剛好是它。",
    "它的長處是大跌時少虧，不是猜中漲跌。用之前先想清楚：能不能接受六成左右的進出是小賠、買回常常比賣出時貴。",
]
RULES = (
    {"id": "blend", "name": "17 條順勢規則混合", "short": "混合", "verified": True,
     "how": "同時看 17 條簡單的順勢規則，有幾條說「持有」，就持有幾成（例如 6 條說持有就持有約 35%）。這 17 條是：收盤高於 10、20、30、50、75、100、150、200 日均線（8 條）；創 10、20、50、100 日新高進場，跌破一半天數的新低出場（4 條）；收盤高於 14、30、60、90、180 天前的價格（5 條）。每天收盤看一次。",
     "use": ["隔天持有的比例 = 17 條裡說持有的條數 ÷ 17。例如 6 條說持有，就是約 35%。",
             "比例變動不到一成可以不調整，研究裡這樣做差別很小。",
             "最接近改變的三條規則和價位列在狀態表裡，可以預先知道收盤到哪裡比例會變。"],
     "why": "它不需要挑參數：17 條一起看，任何一條剛好運氣好或不好都被平均掉。所以換收盤時間、晚幾天做、偶爾漏做，結果都只差一點點。和「固定只拿同樣風險的比例」比，每年多約 15 個百分點，表示它不只是因為拿得少才虧得少，而是真的比較常在下跌時離開。",
     "edge": "研究裡最穩的一種：各種干擾之下結果變化最小。",
     "weak": "分批進出，一年調整約 20 次；大漲時只拿到一部分；2024 年 ETF 上市以後報酬優勢很小，只剩虧得比較淺。",
     "tests": [
         ["有沒有人研究過", "學術界獨立研究（2014–2025 年、141 種指標）：樣本外只有順勢類穩定有效，多條合起來比單條好", "pass"],
         ["事先決定、之後再驗", "一律用混合，2019 年起每年 +45.4%，一直抱著 +43.8%；最深虧 −48% 對 −77%（robust_trend）", "pass"],
         ["換成別的資產", "原封不動用在 10 個資產：10 個風險調整後較好、10 個虧損較淺、10 個最後的錢較多（trend_assets）", "pass"],
         ["換收盤時間", "24 個收盤時間全部贏過一直抱著，每年 +28.9% 到 +34.4%（clock_robustness）", "pass"],
         ["加回後來下市、歸零的幣", "每月依當時成交額選前 20 名、包含後來下市的幣：照混合每年 +20.3%，整籃抱著 −0.3%；下市幣全算歸零仍 +17.8%；388 個幣裡 355 個最後的錢較多（survivorship）", "pass"],
         ["照一般人方式做", "晚上看、隔天做、手續費 0.25%、漏掉一成：1,000 次模擬只有 1 次輸給一直抱著（robust_trend）", "pass"],
         ["和「少拿一點」比", "和同樣波動的固定 58% 比，每年 +31.7% 對 +16.3%（vol_sizing）", "pass"],
         ["2024 年以後", "每年 +17.7% 對 +12.3%，但重新抽樣 2,000 次只有 1,124 次贏，報酬優勢分不出來；虧得淺仍在（−25% 對 −53%）", "weak"],
         ["換到傳統市場（美股、台股、黃金、債券等 14 個）", "14 個全部虧損較淺，但最後的錢比抱著多的只有 3 個；台股加權 +6.4% 對 +5.8%，0050 +13.8% 對 +20.9%（trend_world）", "weak"]]},
    {"id": "vol40", "name": "混合，再依波動調整持有多少", "short": "混合＋波動", "verified": True,
     "how": "先照「17 條混合」決定幾成（同樣那 17 條），市場很激烈時再少拿一點：持有比例乘上「40% ÷ 最近 30 天的年化波動」（最多乘 1，不借錢）。例如混合說持有 50%、最近波動是 60%，就持有 50% × 40 ÷ 60 ≈ 33%。",
     "use": ["先算混合的比例，再乘上波動調整的倍數；狀態表的「隔天」已經算好。",
             "適合最怕大虧的人：研究裡最深虧損最淺。",
             "代價是平均只拿約 37%，大漲時賺得比較少。"],
     "why": "混合規則本身會在下跌時離開；這一層再讓它在劇烈震盪時拿少一點。大跌通常伴隨劇烈震盪，所以兩層疊起來虧損最淺。所有檢驗的結果和混合規則一致，只是報酬換成更低的風險。",
     "edge": "研究裡虧損最淺的一種，最深虧損約一直抱著的一半以下。",
     "weak": "拿得更少，所以報酬比單純混合低；照一般人方式做，報酬只比一直抱著多約 2 個百分點。",
     "tests": [
         ["有沒有人研究過", "依波動調整部位是業界常見做法；學術研究也支持順勢合起來用", "pass"],
         ["事先決定、之後再驗", "公平地每年事先挑，8 年裡 6 年挑到它；2019 年起最深虧 −38%，一直抱著 −77%（robust_trend）", "pass"],
         ["換收盤時間", "24 個收盤時間全部贏過一直抱著，最深虧損 −32% 到 −42%（clock_robustness）", "pass"],
         ["照一般人方式做", "1,000 次模擬裡 62 次輸給一直抱著；最深虧約 −40%（robust_trend）", "pass"],
         ["和「少拿一點」比", "和同樣波動的固定 40% 比，每年 +26.4% 對 +12.7%（vol_sizing）", "pass"],
         ["2024 年以後", "虧得淺仍在；報酬優勢小", "weak"],
         ["換到傳統市場（14 個）", "14 個全部虧損較淺，但最後的錢比抱著多的只有 3 個（trend_world）", "weak"]]},
    {"id": "b5520", "name": "55 日新高進、20 日新低出", "short": "55／20", "verified": True,
     "how": "收盤是最近 55 天最高就全部持有；持有中收盤是最近 20 天最低就全部賣出。其他日子不動。",
     "use": ["只有兩種狀態：全部持有或全部空手。一年大約進出 3 到 7 次。",
             "空手時看「收盤 ≥ 某價位 → 轉成持有」；持有時看「收盤 ≤ 某價位 → 轉成空手」。",
             "最常見的失敗是買回時比賣出時貴而不想買，研究裡這樣做會把優勢全部吃掉。"],
     "why": "它要價格明確創新高才進場、明確轉弱才離開，大部分的大跌它都在場外。換 24 個收盤時間全部贏，表示不是剛好選到幸運的時間；在同樣的虧損控制下，它留下的報酬比停損、少拿一點、買保險都多。",
     "edge": "進出次數少；把最深虧損壓在約 −40% 時，留下的報酬最多（每年約 +33.6%）。",
     "weak": "交易筆數少，一兩筆的好壞影響很大；2024 年以後報酬沒有贏過一直抱著；用在 MSTR 上報酬不如一直抱著。",
     "tests": [
         ["有沒有人研究過", "海龜交易法的系統二就是這個進出規則；突破類規則有學術支持", "pass"],
         ["事先決定、之後再驗", "一律用它，2019 年起每年 +44.7% 對一直抱著 +43.8%，最深虧 −31% 對 −77%（robust_trend）", "pass"],
         ["換收盤時間", "24 個收盤時間全部贏過一直抱著，每年 +27.5% 到 +38.8%（clock_robustness）", "pass"],
         ["照一般人方式做", "1,000 次模擬裡 144 次輸給一直抱著（robust_trend）", "pass"],
         ["和停損、少拿、買保險比", "最深虧損控制在 −40% 左右時，每年 +33.6%；少拿一點 +11.3%、停損 +27.8%（loss_control）", "pass"],
         ["2024 年以後", "每年 +10.9% 對 +12.3%，報酬沒贏；虧得淺仍在", "weak"],
         ["換到傳統市場（美股、台股、黃金、債券等 14 個）", "14 個全部虧損較淺（大跌時平均 −8% 對 −32%），但最後的錢比抱著多的只有 1 個；股票 9 個報酬全輸（trend_world）", "weak"]]},
    {"id": "ma50", "name": "收盤高於 50 日均線才持有", "short": "50 日均線", "verified": True,
     "how": "收盤價高於最近 50 天收盤的平均就持有，低於就空手。",
     "use": ["只有兩種狀態：全部持有或全部空手。一年大約進出 10 次。",
             "價格在均線附近來回時會反覆進出，手續費要低（每次 0.1% 以內）才划算。",
             "勝率很低（比特幣 89 筆裡只有 20 筆賺），完全靠少數幾段大漲。"],
     "why": "它是最簡單的順勢規則，24 個收盤時間全部贏過一直抱著，換資產 10 個裡 9 個風險調整後較好。但它是事後看起來特別好的一條，所以報酬數字要打折。",
     "edge": "歷史報酬最高的一條。",
     "weak": "最深虧損較深（約 −60%）；對手續費和收盤時間比較敏感；勝率只有兩成多。",
     "tests": [
         ["有沒有人研究過", "均線順勢規則的論文（2010–2017）在論文之後的八年仍然有效（published_rules）", "pass"],
         ["換成別的資產", "10 個資產裡 9 個風險調整後較好、10 個虧損較淺、10 個最後的錢較多（trend_assets）", "pass"],
         ["換收盤時間", "24 個收盤時間全部贏過一直抱著，每年 +34.3% 到 +46.6%（clock_robustness）", "pass"],
         ["當成整個市場的濾網", "比特幣跌破 50 日均線就把整籃幣換成現金，加回下市的幣後每年 +36.1%，整籃抱著 −0.3%（survivorship）", "pass"],
         ["照一般人方式做", "1,000 次模擬裡 118 次輸給一直抱著；最深虧約 −63%（robust_trend）", "pass"],
         ["手續費 1%", "每年剩 +17.3%，輸給一直抱著（robust_trend）", "weak"],
         ["2024 年以後", "報酬優勢小", "weak"],
         ["換到傳統市場（14 個）", "12 個虧損較淺，4 個最後的錢比抱著多（那斯達克、台灣加權、恆生、美元）（trend_world）", "weak"]]},
    {"id": "c2010", "name": "20 日新高進、10 日新低出", "short": "20／10", "verified": False,
     "how": "收盤是最近 20 天最高就持有；持有中收盤是最近 10 天最低就賣出。",
     "use": [], "why": "", "tests": [],
     "edge": "虧損比一直抱著淺。",
     "weak": "報酬優勢看運氣：換一個收盤時間，24 個裡只有 12 個贏；照一般人方式做會輸給一直抱著。"},
)
FAILED = (
    ["20 日新高進、10 日新低出（之前推薦過，已收回）", "事先挑參數時九年都挑到它", "換 24 個收盤時間、照一般人方式做",
     "24 個時間裡只有 12 個報酬贏過一直抱著；一般人照做 1,000 次有 718 次輸。只有「虧得淺」站得住"],
    ["恐懼指數很低就買", "別人恐懼時貪婪", "照規則買賣、和一直抱著比", "2022 年以來每年 −13.3%，一直抱著 +7.2%"],
    ["資金費率很負就買", "空單擁擠就會軋空", "照規則買賣、歷史分組", "極端值只有 4 段；照規則做 2023 年以來每年 +1.2%，一直抱著 +42.2%"],
    ["礦工投降訊號（Hash Ribbons）", "礦工撐不住就是底", "28 次訊號一年後的結果", "一年後中位數 +99%，但大跌期間隨便挑一天是 +122%，沒有比較好"],
    ["週期頂部指標（MVRV、Pi 週期、梅耶倍數、Puell 等）", "指標過熱就是頂", "照原本門檻、和改成滾動門檻", "2025 年這一輪固定門檻一條都沒觸發；改成滾動門檻又賣得太早"],
    ["選擇權偏斜", "保險變貴就要跌", "按偏斜分五組看之後漲跌", "最怕跌的那組之後並沒有跌更多，逐年 3 年對 3 年錯"],
    ["預測市場偏向", "群眾覺得要跌就會跌", "24 個月、295 個價位", "機率本身報得準，但偏向哪邊和之後漲跌沒關係"],
    ["81 個指標偏到極端", "指標極端就會反轉", "每個指標偏高、偏低之後 30 天", "162 種組合沒有一個每年方向一致"],
    ["時段、星期、月份規律", "某些時間比較會漲", "24 項，前後兩半和逐年檢查", "扣掉交易成本後全部不剩"],
    ["隔夜與特定時段策略", "公開的隔夜策略每年賺三成以上", "照原文重做、看發表之後", "發表之後每年 −2.6%；賺的其實是「創新高才持有」"],
    ["挑最近最強的幣", "強者恆強", "23 種大幣、事先挑參數", "挑強的幾乎沒加分；有用的是大盤跌破均線就抱現金"],
    ["以太幣／比特幣換著拿", "看比值趨勢決定拿哪個", "事先挑、換其他幣、換收盤時間", "八年只贏兩年；24 個時間裡只有 4 個贏"],
    ["停損加回補", "跌 20% 就停損", "5 種停損 × 4 種回補", "結果全看回補方式；最常見的做法最深虧 −86%，比一直抱著還深"],
    ["買賣權保險", "買保險擋大跌", "每月買跌 10% 的賣權（模型估價）", "一年保費約 36%，最深虧損幾乎沒變（−76% 對 −77%）"],
    ["交易所資金流入", "大量搬進交易所就是要賣", "重現別人的研究、檢查資料", "歷史資料是事後重算的，和每天即時公布的對不上，結果不能信"],
    ["期貨溢價過高時減碼", "期貨比現貨貴太多就快崩了", "13 段高溢價、加在順勢上、換到以太幣", "高溢價後確實較常大跌，但 2023 年後只有 3 段；對以太幣反而變差，事先挑不出穩定設定"],
    ["美股、恐慌指數、相關性、利率濾網", "美股不好比特幣也會跌", "19 種濾網加在順勢上", "多數沒多出東西；最好的「殖利率急升」幅度小、換到索拉納變差"],
    ["只依波動調整、不擇時", "波動大就少拿", "和固定少拿一點比", "效果和固定少拿差不多，沒有多出東西"],
)
PRIMARY = "b5520"  # the one rule shown first: the simplest on/off rule that passed every check
# Plain comparison fields (bitcoin, 2018 on; from clock_robustness, robust_trend and trend_ensemble).
EXTRA = {
    "blend": {"entry": "17 條裡有幾條說持有就持有幾成，每條約 6%；自然分批進出",
              "clock": "+28.9% 到 +34.4%（差 5.5 個百分點，最窄）", "ordinary": "1,000 次裡 1 次",
              "fit": "不想猜、想要最穩的人；能接受每天依比例小幅調整"},
    "vol40": {"entry": "先照 17 條算比例，市場越劇烈拿越少",
              "clock": "+24.2% 到 +28.1%", "ordinary": "1,000 次裡 62 次",
              "fit": "最怕大虧的人；接受報酬較低"},
    "b5520": {"entry": "創 55 日新高全部買、跌破 20 日新低全部賣",
              "clock": "+27.5% 到 +38.8%（24 個全贏）", "ordinary": "1,000 次裡 144 次",
              "fit": "想要一條簡單、一年只動幾次的規則"},
    "ma50": {"entry": "收盤高於 50 日均線全部持有，低於全部賣",
             "clock": "+34.3% 到 +46.6%（24 個全贏）", "ordinary": "1,000 次裡 118 次",
             "fit": "能忍受連續小賠、手續費很低的人"},
    "c2010": {"entry": "創 20 日新高全部買、跌破 10 日新低全部賣",
              "clock": "+11.8% 到 +30.6%（24 個裡只有 12 個贏）", "ordinary": "1,000 次裡 718 次",
              "fit": "已收回，不建議當依據", "retracted": True},
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
