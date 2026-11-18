"""Second set of historical studies, written to cache/_studies2.json and shown on the "history" tab.

Each study lives in its own module under dashboard/studies2/ and exposes study() -> dict with one shared shape, so
the page needs a single renderer for all of them:

    {"id", "title", "question", "asof", "verdict",
     "now": [{"label", "value"}], "tables": [{"caption", "columns", "rows"}], "notes": [...], "sources": [{"name", "url"}]}

Every string is ready to show (Traditional Chinese, already formatted). These are not indicators: collect() returns
no rows. A study that fails keeps its previous result and the failure is recorded under "errors". Everything is
descriptive and in-sample; none of it is a forecast or advice.
"""
from __future__ import annotations

import importlib
import json

from dashboard import common

PATH = common.CACHE / "_studies2.json"
STUDIES = ("main_signal_check", "crash_timing", "rally_timing", "exit_entry_combo", "pullback_setups", "grid_rebalance", "forced_flows", "trend_portfolio", "order_flow", "ml_honest", "coin_basket", "intraday_trend", "trend_world", "rule_upgrades", "retail_indicators", "long_cycles", "survivorship", "loss_control", "robust_trend", "walkforward", "clock_robustness", "trend_practical", "vol_sizing", "rotation", "option_selling", "pair_rotation", "cross_market", "basis_filter", "trend_anatomy", "trend_ensemble", "trend_assets",
           "leveraged_trend", "ewma_forecast", "cross_momentum", "donchian_ensemble", "published_rules",
           "overnight", "flow_filters", "popular_rules", "hash_ribbons", "carry", "seasonality",
           "funding_positioning", "indicator_scan", "touch_calibration", "polymarket_odds", "episodes",
           "skew_history")  # page order
REQUIRED = ("id", "title", "question", "asof", "verdict", "tables", "notes", "sources")
# Our own reading of each study, shown as one line in the summary table at the top of the history tab.
# "pass": held up; "mixed": partly; "unclear": positive or mixed but too few cases to tell (not proven useless);
# "fail": the data shows it does not work. Reviewed by an independent audit on 2026-11-18.
# This is a human judgement written after reading the results, so it must be revisited when results change.
GRADES = {
    "walkforward": ("mixed", "事先挑到的是「20 日新高進、10 日新低出」，之後九年虧得比一直抱著淺；但這一條的報酬優勢換個收盤時間就不穩"),
    "clock_robustness": ("pass", "把收盤時間換成二十四個整點：九種順勢做法裡七種的報酬全部仍贏過一直抱著，九種的虧損全部較淺；不穩的是「20 日新高」和「200 日均線」兩種"),
    "rule_upgrades": ("unclear", "10 種改良 55／20 的做法（更快買回、分兩半、加濾網、依波動調整、停損）沒有一種在事先挑、換資產、換收盤時間三關都贏過原版，所以維持原版"),
    "retail_indicators": ("fail", "台灣常見的 KD、RSI、布林通道勝率很高卻贏不了一直抱著；MACD、均線多頭排列不如已驗證的順勢規則；季線（60 日均線）表現接近 50 日均線，本質就是順勢"),
    "long_cycles": ("unclear", "康波等長週期：156 年美股資料只有約 3 次長波，頻譜檢查 16 項 0 項顯著，轉折點多半事後 10 到 40 年才被寫出來；照康波進出 5 種版本都輸給一直持股"),
    "order_flow": ("fail", "主動買賣力道、累積買賣差、成交量看不出之後漲跌；常說的「價漲量縮背離」之後反而漲更多；唯一有點幫助的 OBV 濾網其實就是另一個順勢條件，事先挑也沒贏"),
    "ml_honest": ("fail", "把所有指標交給機器學習、嚴格不偷看未來：猜下週漲跌只對 100 次裡 48 到 51 次，和永遠猜漲（52 次）差不多；拿來交易沒有贏過 17 條順勢混合"),
    "main_signal_check": ("pass", "公平重選主訊號：換起算日、換收盤時間、照一般人方式做、換幣四項一起考，50 日均線最可靠；之前引用的 55／20 數字因起算日被美化約 11 個百分點"),
    "crash_timing": ("fail", "更早下車：55／20 和「10 日低出」能避開六到七成跌幅（50 日均線只有三成），但買回太晚，公平檢驗分數大降；沒有一種同時更早下車又不吃虧"),
    "rally_timing": ("unclear", "更早上車：跌深後站上 20 日均線最接近低點，但每次只早約 5 個百分點、假反彈多 3 次；「均線轉平才上車」在比特幣上最深虧損最淺，換幣卻不穩"),
    "exit_entry_combo": ("fail", "5 種下車 × 5 種上車共 25 種組合，50 日均線本身仍排第一；越會躲跌的下車方式，買回時就越貴，兩者一直互相抵銷"),
    "pullback_setups": ("unclear", "Raschke「聖杯」、Connors RSI(2) 等拉回買法：勝率七成左右，但單獨用大多時間空手、每年只有 +4% 到 +6%；把 RSI(2)<5 加在 50 日均線上剛好通過三項標準，但差距不到 1 個百分點，分不出是不是運氣"),
    "grid_rebalance": ("fail", "網格交易和固定比例再平衡：和「同樣持有量但不調整」比幾乎沒多賺；大漲時網格很快賣光（比特幣漲 13 倍只賺 12%），大跌時一路買滿，虧損接近全抱"),
    "forced_flows": ("unclear", "被迫的買賣：爆倉大跌後只比隨機日好一點且後半段消失；代幣解鎖前 30 天明顯偏弱但只有 9 件；月結算前一週偏弱在三個幣都出現，但避開那週仍輸給一直抱著；芝商所缺口回補不比隨機價位多"),
    "trend_portfolio": ("mixed", "比特幣、美股、台股、黃金、債券等多市場順勢組合：最深虧損只有 −5% 到 −9%、風險調整後明顯較好，但每年報酬低於六成股四成債；加入比特幣順勢（平均只佔 2%）是最有幫助的一塊"),
    "coin_basket": ("fail", "包含下市幣的真實幣單上，7 種幣籃（風險平均、CTREND、避開高波動）沒有一種三關都贏過只用比特幣照混合規則；事先挑的結果反而輸很多"),
    "intraday_trend": ("fail", "4 小時、1 小時、12 小時線的順勢：事後看有幾種好看，但每年事先挑的結果 8 年只贏日線 55／20 一年；越快的版本越被手續費吃掉"),
    "trend_world": ("mixed", "同樣規則用在美股、台股、日港股、黃金、債券、原油等 14 個市場：虧損變淺的有 12 到 14 個，但最後的錢比抱著多的只有 1 到 4 個；股票上報酬多半輸給抱著"),
    "survivorship": ("pass", "把後來下市、歸零的幣加回去重測：各幣照混合規則每年 +20.3%，整籃抱著 −0.3%；就算下市幣全部算歸零仍有 +17.8%。388 個幣裡 355 個照規則最後的錢較多"),
    "basis_filter": ("unclear", "期貨溢價每年 20% 以上之後確實比較常大跌，但只有 13 段、2023 年後只有 3 段；加在順勢上對比特幣略有幫助，對以太幣反而變差"),
    "loss_control": ("pass", "要把最深虧損壓在同樣程度，順勢做法每一種目標都留下最多報酬；停損加回補的結果全看回補方式，買賣權保險一年要花三成多保費、幾乎沒擋住二〇二二年的慢跌"),
    "pair_rotation": ("fail", "照以太幣／比特幣比值的趨勢換著拿，沒有贏過兩個各自照順勢規則各拿一半；事先挑的結果八年只贏兩年"),
    "cross_market": ("unclear", "美股趨勢、恐慌指數、相關性、利率、美元加在比特幣順勢上，多數沒有多出東西；殖利率急升時減碼最穩但幅度小，搬到索拉納反而變差"),
    "robust_trend": ("pass", "拿掉收盤時間的運氣、再加上一般人的執行方式：六種順勢做法裡五種報酬仍高於一直抱著，虧損全部較淺；二〇二四年以後報酬優勢和零分不出來，虧得淺仍在"),
    "rotation": ("fail", "順勢規則空手時錢放短期公債、那斯達克或黃金，差別很小；週期指標改用滾動門檻後恢復出訊號，但賣得太早；固定比例的組合不比擇時差"),
    "vol_sizing": ("pass", "只依波動調整持有多少，效果和固定少拿一點差不多；順勢則比同樣風險的固定比例每年多十五個百分點，表示順勢訊號本身帶有方向的資訊"),
    "option_selling": ("fail", "持有比特幣同時每月賣出買權，模型估算是賺得略多、虧得較淺，但大漲時幾乎賺不到，而且估價稍有偏差結果就不成立"),
    "trend_practical": ("mixed", "一般人晚一點看、隔天才做、偶爾漏訊號，「20 日新高」這條的報酬優勢大致就沒了；虧得比較淺仍然成立"),
    "leveraged_trend": ("mixed", "兩倍槓桿基金抱著不動幾乎必輸，實際耗損每年兩到四成；照規則持有比較好，但虧損比不加槓桿深很多"),
    "ewma_forecast": ("fail", "照趨勢強度連續加減碼，沒有贏過簡單的進出規則；二〇二四年以後所有順勢做法的優勢都變小"),
    "overnight": ("fail", "隔夜和特定時段的公開策略，發表之後或扣成本後都不剩；賺的其實是創新高才持有"),
    "flow_filters": ("unclear", "交易所流量過濾的歷史資料是事後重算的，和每天即時公布的數字對不上，回測結果不能信"),
    "trend_anatomy": ("pass", "優勢全部來自躲掉大跌；六成的交易是賠的，獲利集中在少數幾筆"),
    "trend_ensemble": ("pass", "十七條順勢規則混合：二十四個收盤時間全部贏過一直抱著，虧損也淺；依波動調整後更穩，做空沒有幫助"),
    "trend_assets": ("pass", "規則原封不動搬到別的資產，十個裡九個最後的錢比較多"),
    "cross_momentum": ("mixed", "挑最強的幣幾乎沒加分；有用的是大盤跌破均線就抱現金"),
    "donchian_ensemble": ("fail", "業界論文的系統和海龜法重做得出來，但沒有比簡單規則多出東西"),
    "published_rules": ("mixed", "均線順勢在論文之後仍有效但變弱；創新高隔天才持有已失效"),
    "popular_rules": ("unclear", "固定門檻的賣出訊號在最近一輪一條都沒觸發；多數規則交易次數太少無法驗證"),
    "hash_ribbons": ("unclear", "礦工投降後買進，沒有比大跌期間隨便挑一天買好"),
    "carry": ("fail", "買現貨同時空合約收資金費，以前每年約一成，最近一年低於美元定存"),
    "seasonality": ("fail", "時段、星期、月份的規律，扣掉交易成本後不剩"),
    "funding_positioning": ("unclear", "資金費率極端負值之後多半上漲，但只有四段例子；多空比算的是人頭不是錢"),
    "indicator_scan": ("fail", "八十一個指標偏到極端，沒有一個每年方向一致"),
    "touch_calibration": ("unclear", "選擇權估的碰到機率大致準，上漲那邊高估約三分之一；準不代表能賺"),
    "polymarket_odds": ("unclear", "預測市場的機率報得準，但它偏向哪邊對之後的方向沒有幫助"),
    "episodes": ("unclear", "急跌之後漲跌各半；和現在最像的幾次也沒有一致方向"),
    "skew_history": ("fail", "跌的保險變貴，之後並沒有跌更多"),
}


def validate(result) -> list[str]:
    """Problems with one study's result (empty list = well formed)."""
    if not isinstance(result, dict):
        return ["結果不是物件"]
    problems = [f"缺少 {key}" for key in REQUIRED if key not in result]
    for key in ("title", "question", "verdict"):
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            problems.append(f"{key} 應該是非空字串")
    for index, row in enumerate(result.get("now") or []):
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or "value" not in row:
            problems.append(f"now[{index}] 格式不對")
    tables = result.get("tables")
    if not isinstance(tables, list):
        problems.append("tables 應該是陣列")
        tables = []
    for index, table in enumerate(tables):
        columns = table.get("columns") if isinstance(table, dict) else None
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(columns, list) or not columns or not isinstance(rows, list):
            problems.append(f"tables[{index}] 缺少 columns 或 rows")
            continue
        if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
            problems.append(f"tables[{index}] 有一列的欄數和表頭不一樣")
    for key in ("notes", "sources"):
        if key in result and not isinstance(result[key], list):
            problems.append(f"{key} 應該是陣列")
    return problems


def collect() -> list[dict]:
    try:
        previous = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        previous = {}
    old = {item.get("id"): item for item in previous.get("studies", []) if isinstance(item, dict)}
    payload = {"collected_utc": common.now_utc(), "errors": {}, "studies": []}
    for name in STUDIES:
        try:
            result = importlib.import_module(f"dashboard.studies2.{name}").study()
            problems = validate(result)
            if problems:
                raise ValueError("；".join(problems[:3]))
            json.dumps(result, allow_nan=False)  # NaN or infinity would break the page
            result["id"] = name
            grade = GRADES.get(name)
            result["grade"], result["grade_note"] = grade if grade else (None, None)
            result["computed_utc"] = payload["collected_utc"]
            payload["studies"].append(result)
        except Exception as exc:  # noqa: BLE001 - one study failing keeps its previous result
            payload["errors"][name] = f"{type(exc).__name__}: {exc}"[:200]
            if name in old:
                payload["studies"].append(old[name])
            print(f"  統計 {name} 這次沒有更新：{payload['errors'][name]}")
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for item in data["studies"]:
        print(item["id"], "->", item["verdict"])
    print("errors:", data["errors"])
