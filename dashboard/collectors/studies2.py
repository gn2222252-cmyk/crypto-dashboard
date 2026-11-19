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
STUDIES = ("main_signal_check", "risk_flags", "etf_flows", "unlock_avoid", "etf_loop", "big_buyer", "liquidity_lead", "new_listings", "breadth_timing", "macro_days", "mstr_premium", "vrp_direction", "kimchi_premium", "vol_breakout", "power_law", "shelton_blend", "attention", "coinbase_premium", "lead_followup", "jump_skew", "blockspace", "pre_holiday", "ls_ratio", "oi_flush", "stablecoin_supply", "thesis_four", "network_growth", "sth_cost", "yen_carry", "cycle_aware", "gold_lead", "puell", "exchange_reserves", "mtf_gate", "gold_dual", "business_cycle", "oecd_followup", "nfci_filter", "dollar_shock", "adx_breakout", "nfci_stress", "gate_design", "stress_family", "economy_family", "net_liquidity", "canary_breadth", "rally_sources", "supply_demand", "funding_warning", "price_impact", "reluctant_sellers", "force_balance", "whipsaw_anatomy", "old_coin_selling", "crash_timing", "rally_timing", "exit_entry_combo", "pullback_setups", "grid_rebalance", "forced_flows", "trend_portfolio", "order_flow", "ml_honest", "coin_basket", "intraday_trend", "trend_world", "rule_upgrades", "retail_indicators", "long_cycles", "survivorship", "loss_control", "robust_trend", "walkforward", "clock_robustness", "trend_practical", "vol_sizing", "rotation", "option_selling", "pair_rotation", "cross_market", "basis_filter", "trend_anatomy", "trend_ensemble", "trend_assets",
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
    "risk_flags": ("fail", "期貨溢價過高、資金費率過熱、殖利率急升、結算週、均線下彎五個警訊事先組成分數加在 50 日均線上：3 個才空手幾乎不出手、排名分數 67%（均線 79%）；2 個就空手躲掉較多大跌但少賺更多，排名 44%；隨機錯位對照也做得一樣好"),
    "etf_flows": ("unclear", "黃金 ETF 22 年：流入流出之後的金價，反向、順向、領先都沒過事先標準，資金多半跟在價格後面；比特幣 ETF 方向是流出後續弱（和學術說的反向相反），但不到兩年、極端只有幾段"),
    "unlock_avoid": ("unclear", "77 個幣 772 次 2% 以上解鎖：解鎖前 30 天比比特幣差 4.2 點，但同幣隨便挑日子也差 2.2 點；扣掉該幣平常就落後的部分只剩 -0.3 點；避開解鎖前 30 天的好處多半是山寨幣平常就弱，不是解鎖本身"),
    "etf_loop": ("fail", "把「小漲→連續流入→推高價格→停止後回吐」拆成七段：大漲後資金確實連續湧入、而且有慣性（比特幣比黃金明顯），但資金流規則沒贏過只看價格的同類規則，隨機對照 45% 一樣好，比特幣上遠輸 50 日均線"),
    "big_buyer": ("fail", "Strategy 121 次買幣公告：看不出漲了才買，公告前後比特幣沒有比沒公告的週好；連續買停下後 60 天平均 -6.7%（單純連漲結束後 +17.1%）但只有 11 段；照公告持有報酬÷波動 0.42，50 日均線 1.24，打亂公告週 70% 一樣好"),
    "liquidity_lead": ("fail", "網路流行的「比特幣落後全球 M2 約 10 週」照原樣測：相關 2013–19 年 0.08、2020–23 年 0.42、2024 年起 -0.59（流行之後反而反方向）；最佳落後天數 0 到 147 天亂飄；拿來進出沒贏過 50 日均線，淨流動性版也一樣"),
    "new_listings": ("mixed", "幣安新上架 436 個幣：上架後 12 個月比比特幣差（中位數）57%、比老幣差 31%，三個時段方向一致，避開上架未滿 6／12 個月的幣贏過 200 次隨機避開；但少數暴漲的幣讓「新幣組合減老幣組合」的區間都含零，事先標準只過一半；只和挑山寨幣有關"),
    "breadth_timing": ("unclear", "用「前 50 大幣有幾成站在均線上」決定比特幣進出（2019 年起）：公開的三條件系統把最深虧損從 -52% 砍到 -28%、隨機錯位只有 9% 一樣好，但一般人做法下只有 48% 贏過持有；大量幣同時翻多提早上車只在 2020 年早了 14 天；只有 3 次熊市"),
    "macro_days": ("fail", "聯準會開會 110 次、CPI 公布 166 次：「會前漲、會後跌」三段方向都不一致、區間都含零；開會後或 CPI 前後避開，沒贏過 50 日均線也沒贏過隨機日期；同期那斯達克的會前上漲也不明顯"),
    "mstr_premium": ("unclear", "MSTR 溢價最低三分之一之後 60 天，MSTR 比比特幣平均多 6%、最高三分之一少 1%，三段方向一致但區間含零；溢價不一定漲時變大、跌時壓扁；依溢價在 MSTR 和比特幣之間切換沒贏過一直拿 MSTR，隨機平移 64–79% 一樣好"),
    "vrp_direction": ("fail", "選擇權恐懼溢價（隱含波動減實際波動）高低，之後比特幣 20 天四段 2 正 2 負、以太幣方向相反；「溢價最高兩成時提早上車」理想做法贏均線，但一般人做法只 38% 贏、晚一天就輸；資料只有 2021 年以後"),
    "kimchi_premium": ("unclear", "韓國溢價越高之後越弱（最高兩成之後 30 天 +0.4%，平常 +4.7%，三段同方向），但主要來自 2017–18、2021 兩次泡沫，2023 起很弱；網路流傳的「轉正會漲」「過熱＋美國溢價為負會回檔」照原樣測區間都含零；加在均線上一般人做法只 19% 贏"),
    "vol_breakout": ("fail", "變動性突破（漲破開盤＋前一天振幅一半就買、隔天開盤賣）：基本版每邊扣 0.10% 報酬÷波動 0.61、扣 0.30% 變負；最好的版本 1.23 仍輸 50 日均線 1.33，換開盤時間 24 個只贏 7 個、其他幣只贏 1 個；每邊手續費超過約 0.08% 就不如均線"),
    "power_law": ("fail", "只用當時資料畫冪次法則趨勢線：便宜一個標準差以上一年後中位數 +124%、貴兩個標準差 -80%，但只來自少數年份；照它加減碼的四條規則沒有一條穩定贏過 50 日均線，和均線合用只差 35 天；起點前後移 1–3 年，斜率從 3.1 變到 8.2，結論跟著變"),
    "shelton_blend": ("unclear", "重做 Shelton（2024）：單一模型（存量流量、梅特卡夫、技術面、情緒）樣本外都失效有重現；但「混合後進出勝過持有」重現不出來（0.32 對 0.35），論文發表後混合預測一直說持有、輸 50 日均線；論文全文取不到，規則是推測的"),
    "attention": ("fail", "維基百科「比特幣」頁面瀏覽量：只看得出價格先動、關注跟著來，關注變化和之後 30 天漲跌相關只有 +0.04；論文說的「漲勢中關注上升推更高」用 50 日均線定義不成立；關注暴衝 2023 年前後方向相反；兩條規則排名 41%、32%（均線 79%）"),
    "coinbase_premium": ("unclear", "Coinbase 比幣安貴或便宜：「極度折價是底部」照原樣測反而之後 30 天少 9.3 點（部分是 USDT 脫鉤）；「溢價為正確認漲勢」原版區間含零；改用美元對美元的溢價為正時 +5.9 點、三段一致，但這是看過結果才加的，要另外事先登記再測"),
    "lead_followup": ("fail", "兩條事後線索寫死規則後重考：美元對美元 Coinbase 溢價為正才持有，最深虧損中位數 -30%（均線 -54%）、大熊市躲掉 61–89%，但上漲年少賺很多，一般人做法只 24% 贏過持有；短線跌破 7 日均線且關注下降就下車，被洗出場太多、隨機錯位 86% 一樣好"),
    "jump_skew": ("fail", "一週內價格突然跳動和偏態高，論文說下週較弱：比特幣三段方向不一致，四個其他幣全部反方向；拿來在 50 日均線上提早下車排名 50%（均線 75%），隨機對照 31–35% 一樣好；論文本身是比較不同幣，不是比特幣擇時"),
    "blockspace": ("unclear", "手續費收入 z 分數為正時之後 90 天多 +35.9 點、三段同方向，但 2023 年起縮到 +8.2、只有約 10 段；和 50 日均線合用最深虧損中位數 -25%（均線 -54%），但排名 52%（均線 80%）、隨機錯位 14%；博士論文的出塊數預測樣本外 R² 為負，晚一天用就沒有優勢"),
    "pre_holiday": ("fail", "美股假日前一天＋創 10 日新高買進、假日後賣（Quantpedia）：扣成本每次 +2.22%（2026-11-19 修正舊價格日期後，區間 +0.07% 到 +4.18% 勉強不含零）但只有 25 次、幾乎全來自 2013–2017，三段方向不一致；同星期幾也創新高的平常日子一樣好（隨機 59% 不輸），其實是順勢不是假日；加在均線上略差"),
    "ls_ratio": ("unclear", "幣安合約散戶多空比越高、或散戶比大戶越偏多，之後 30 天越弱（每高一組 -1.8／-2.1 點，前後半同方向，四個其他幣全部同方向）；但資料只有約 5 年，加在 50 日均線上 LS2 排名 80%（均線 79%）隨機錯位 24% 一樣好、LS1 晚一天就變差"),
    "oi_flush": ("fail", "合約未平倉量 3 天大跌（清槓桿）之後，比特幣 14 天反而比平常弱 3 點（區間不含零），28 次只有 1 次 14 天內漲 20%，CryptoQuant 的反彈說法不成立；30 天接近 K33 說的偏弱但不顯著；清洗後提早上車、或槓桿堆高時下車，排名 48%、62%（均線 79%），隨機錯位 80%、60% 一樣好；資料只有約 5 年"),
    "stablecoin_supply": ("fail", "穩定幣供給大多是跟著價格走：比特幣先漲、約一週後供給才增加（相關 +0.26，三段都同方向）；反過來用供給預測比特幣，相關只有 +0.02、樣本外沒有用；增發爆量的好成績幾乎全來自 2020 年兩次；供給為正才持有排名 38%、爆量提早上車 77%（均線 78%），隨機錯位 46%、15%"),
    "thesis_four": ("fail", "博士論文說 141 個預測因子裡只有少數長期樣本外一直贏（實際是 5 個趨勢／波動技術指標）：做成持有規則沒有一條贏過 50 日均線（最好的 EMA 交叉 56%、加均線的 ADX 75%，均線 81%），多數隨機錯位也一樣好；論文樣本之後的新資料，7–28 天的預測全部變負"),
    "network_growth": ("fail", "2026 期刊論文說網路參與（活躍地址、算力）能預測報酬（論文全文被擋、沒讀到，照摘要重建）：每週成長訊號三段期間方向不一致、單一指標樣本外都是負的；拿來擇時從沒贏過一直抱著（排名 0%），在均線持有時反而略差，沒有幫 50 日均線加分"),
    "sth_cost": ("fail", "短期持有者成本線（近 5 個月買家平均成本）幾乎就是一條 87 日均線（89% 的日子狀態相同）：站上才持有最深虧損較淺（-38%，均線 -54%）但少賺；均線外加「站回成本線提早上車」每年多約 1.7 點，但隨機錯位 19% 一樣好；成本比 < 0.8 的「買點」和 > 1.3 的「過熱」都不準；歷史數值會被改寫"),
    "yen_carry": ("fail", "2013 年起 27 次日圓急升（5 天漲幅在 3 年內最極端 2%）後，比特幣 30 天表現和平常差不多（-0.8 點，兩段方向相反），也沒有比美股跌更多；急升後先拿現金 30 天，最深虧損較淺（-44%，均線 -54%）但排名 65%（均線 80%），隨機錯位 42% 一樣好"),
    "cycle_aware": ("fail", "CoinDesk／10x 的「週期感知」策略只公布成績（Sharpe 1.22、最深 -44%）沒公布規則，無法驗證；照描述做的 10 訊號版和部落格的 20EMA＋50MA 都比抱著少虧（最深虧損中位數 -35%、-42%，均線 -54%）、也不是運氣，但一般人做法和其他幣常輸給持有，排名 50%、57%，遠低於 50 日均線 79%"),
    "gold_lead": ("fail", "「黃金領先比特幣 4–7 個月」只在 2018–21 年成立，2013–17、2022 年後和說法公布後都是反方向；滾動兩年看最佳落後週數從 0 到 32 週亂飄；「黃金漲了、比特幣還沒跟上」之後 13 週反而較差；兩條補漲規則排名 80%、22%（均線 81%），隨機錯位 20%、50% 一樣好"),
    "puell": ("fail", "礦工收入倍數（Puell）：綠區（< 0.5）10 次之後一年中位數 +132%，和任何一個深跌日（+122%）差不多，等於只是「價格跌很多」；紅區（> 4）只有 4 次、3 次在 2013 年太早喊頂，2021 年兩個頂部都沒進紅區；「算力高＋Puell 低會反彈」不成立；兩條補強均線的規則都沒通過"),
    "exchange_reserves": ("fail", "交易所比特幣餘額（Coin Metrics 免費版）：餘額大減之後並沒有比較會漲（30 天 -5.2 點，2013–17 甚至相反），只有「餘額大增之後波動變大」成立（+11 點、三段一致）；兩條規則排名 12%、72%（均線 80%），隨機錯位 34%、49%；整段歷史在 2026 年 4 月用今天的交易所地址重算過，回測看到的不是當時看得到的資料"),
    "mtf_gate": ("fail", "日線准許、4 小時線抓進出時機：比隨機進出好（隨機對照 4–6%），但在場時間從 52% 降到 20–35%、每年多 30 次交易，排名 0%、14%（均線 82%），一般人一天看三次全部輸給均線；Quantpedia 的每小時版本在幣安資料上重做不出來，扣 0.10% 成本就每年虧 25–29%"),
    "gold_dual": ("fail", "比特幣對黃金雙動能（Quantpedia 2026）原版重做得出來（每年 +67.5%、最深 -48%），但成績主要來自「比特幣自己在跌就不拿」加上近年黃金大漲；「比特幣要先贏黃金」沒有比只看比特幣自己漲跌更好；只拿比特幣的版本排名 37%（均線 80%），隨機錯位黃金 82% 一樣好；波動上限版照原文重做不出原文數字"),
    "business_cycle": ("fail", "製造業景氣（費城聯儲調查，代替拿不到的 ISM）高於 10 個月平均的「有利月」：三段方向不一致，「不利月等於零報酬」不成立；均線＋景氣排名 41%（均線 80%），隨機錯位 14%；唯一接近的是 OECD 領先指標＋均線（排名 81%、最深虧損中位數 -29%），但這是看過結果才挑出來的，要另外事先登記再測"),
    "oecd_followup": ("mixed", "OECD 領先指標高於 10 個月平均＋50 日均線，規則寫死後重考，四項標準都過（排名 81% 對均線 80%、最深虧損中位數 -29% 對 -54%、隨機錯位 1%、三段同方向）；但這是看過結果才挑的線索、只贏 1 個百分點、多等一個月就掉到 68%、優勢集中在 2018／2022／2025 少虧、OECD 數字每月修正；2026-11-19 起放進影子紀錄看實際表現"),
    "nfci_filter": ("mixed", "（壓力測試後由通過降為好壞參半：少虧在 20 種鄰近設定都在，但贏均線只有 11 種、換 OFR 指數沒重現）50 日均線＋芝加哥聯準會金融情勢指數 NFCI 低於自己 52 週平均才持有：事先寫好的四項全過（排名 83% 對均線 80%、最深虧損中位數 -37% 對 -54%、隨機錯位 1%、三段寬鬆月都比較好）；但好處集中在 2018、2022 兩個空頭年，2020、2023 大漲年少賺很多，NFCI 每週修正，也看不出金融情勢先動，已加入影子紀錄"),
    "dollar_shock": ("unclear", "美元指數 3 天跌 2% 以上：2011 年起只發生 3 次，事件後比特幣沒有比平常好；加進均線排名 82% 對 80% 但最深虧損更深；美元和比特幣同一天一起動，不是美元先動"),
    "adx_breakout": ("fail", "只在盤整（ADX < 25）時做 15 日突破：閘門確實讓普通突破變好（排名 61% 對 14%、虧損變淺、隨機錯位 4%），但排名仍輸 50 日均線 80%，2017 年後兩段都落後"),
    "nfci_stress": ("fail", "N1 的壓力測試：最深虧損在 20 種鄰近設定全部變淺（-28% 到 -37%，均線 -54%），但排名贏均線的只有 11 種（門檻 15）、N1 的 52 週資料晚 1–4 週就不贏；換 OFR 金融壓力指數排名 67% 沒重現（另兩個指數抓不到）；拿掉 2018、2022 平均每年少 31.5 點。少虧是真的，多賺不可靠"),
    "gate_design": ("fail", "同樣的 NFCI／OECD 閘門換四種用法（只擋新進場、半倉、兩個都壞才下車、兩閘門投票）：沒有一種在兩個指標上都比原本好 3 點以上；半倉只是把少虧和少賺各打對折，『或』的做法虧損明顯變深"),
    "stress_family": ("mixed", "10 個市場壓力指標用同一條規則整批考：2 個達標（高收益債、公司債對公債），運氣平均 0.08 個（打亂後 0.5% 不少於 2 個）；但兩個都是信用類、和 NFCI 相關 +0.36／+0.55，像同一個訊號；過半數投票的組合排名 54% 不及格"),
    "economy_family": ("fail", "9 個美國經濟指標（失業救濟金、活動指數、消費者信心、建築許可、利率方向等）用同一條規則全部沒達標（0／9），組合排名 36% 對均線 80%、隨機錯位 65%；NFCI、OECD 的成功沒有延伸到其他經濟指標"),
    "net_liquidity": ("fail", "聯準會淨流動性（資產負債表－財政部帳戶－逆回購）和比特幣大多同一週一起動，『領先 13 週』不成立；當均線濾網排名 61% 對 80%、隨機錯位 14%；國庫券發行量方向對但很弱"),
    "canary_breadth": ("fail", "抗通膨公債動能、七類資產有幾類在 10 個月均線上當閘門：排名 68%／66% 對均線 80%，隨機錯位約兩成一樣好；2022 少虧約 40 點但 2023 少賺約 130 點，資產廣度多半跟在比特幣後面"),
    "rally_sources": ("fail", "2013 年起 12 段大漲，均線中位數晚 26 天、錯過 +29%；12 個資金管道在大漲前常會亮，但把時間隨機挪動也一樣會亮；發現期挑出的維基瀏覽量在 2021 年後讓均線變差（排名 56% 對 77%、隨機錯位 79%）。線索：起漲初期每一塊錢流入推動的市值比後期大"),
    "supply_demand": ("fail", "『賣壓少（流入交易所少、帳面獲利低）＋買盤來（穩定幣成長或 Coinbase 溢價）』同時成立時，之後 30 天 +0.7%，反而輸給只有買盤來的 +9.3%（交互作用 -6.6 點，區間含零）；當提早進場規則排名 80.2% 對均線 80.4%、隨機錯位 22%。這樣定義的賣壓少其實多半是熊市"),
    "funding_warning": ("fail", "資金費率偏熱多半只是剛漲完（熱的日子過去 30 天平均已漲 29%），之後沒有比平常差；加在均線上排名 39% 對 77%、隨機錯位 89%。BitMEX 資料被擋（403，沒有繞過），只測到 2019-09 以後"),
    "price_impact": ("fail", "直接量『買盤每一塊錢推高多少、賣盤每一塊錢壓低多少』的比值當作賣方多寡：比值高的日子之後 30 天 +3.5%，全部日子 +4.0%，配對過去漲幅後也沒有差別；當提早進場規則排名 68.9% 對均線 80.4%、隨機錯位 90%；它和價格一起動，不領先。只有幣安 2017-08 以後的資料"),
    "reluctant_sellers": ("fail", "只看賣方一邊（送進交易所的比特幣），試了三種定義：流入少之後 30 天反而比流入多漲得少（價格在上 +4.6% 對 +25.5%，價格在下 +2.3% 對 +6.4%）。這只說明賣方比平常活躍時買方通常更活躍，不能說賣方贏；單看一邊沒有意義。三條『多抱』規則都沒過"),
    "force_balance": ("mixed", "買方（提走的比特幣＋新增穩定幣＋ETF 流入）和賣方（送進交易所的比特幣）一起量：兩者幾乎同步（相關 0.96），買方固定後賣方的影響 -1.7 點、分不出來，原本的九格假設驗不出來；但『買賣差額偏買方且站上 20 日均線就提早進場』排名 82.6% 對均線 80.4%、虧損一樣、隨機錯位 2%，三條都過。這是賣壓想法的第五份研究、提早進場那幾段本身合計是賠的，只放影子紀錄"),
    "whipsaw_anatomy": ("fail", "均線下車 148 次：七成是假跌破（坐過去 +4.8%）、兩成是真下跌（坐過去 -18.4%）；下車當下的十種價格狀態都分不出兩者（假跌破比例都在 62%–82%）；發現期挑到的條件在測試期排第 7，規則排名 72.8% 對均線 78.9%"),
    "old_coin_selling": ("fail", "放了 155 天以上的老幣大量移動（Open Bitcoin Metrics，和 Blockchair 抽 20 天核對相符）之後 60 天，事先預期比較差，量到的是比較好（老幣動得少減動得多 -15.2 點，區間 -27.8 到 -2.7，三段都不是正的）；老幣多半在價格上漲時和之後才動；25 次大量移動裡 15 次之後 90 天內沒有見頂；兩條規則都輸均線。資料分不出這些幣是賣掉還是只是搬動"),
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
