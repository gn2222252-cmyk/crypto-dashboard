"""Historical study "sentiment_turns": what did SENTIMENT (fear and greed, attention, crowd positioning, protection
prices, regional eagerness) look like around the turns - and if a person follows it, does entering make money and
does exiting lose less; and if one measure cannot do it, does ONE added condition do it?

Why: the owner asked "what about sentiment?". In about 114 studies the Fear & Greed index was only an ingredient
(collectors/studies.py "washout", popular_rules, indicator_scan, episodes, rally_timing, walkforward, breadth_timing,
shelton_blend, ml_honest) and it was not among the 43 series signs_before_turns laid out before bull and bear starts.
This study does the owner's three steps for sentiment: (1) OBSERVE anchored on the turns, (2) FOLLOW it, (3) ADD ONE
parameter.

Owner's ground rules (every user-facing sentence): observe first; do not conclude lightly; label 量到的 (measured,
number, k of N), 文獻說的 (source, and whether it was opened) or 推測 (a guess, marked); where the data cannot separate two
explanations, say so and pick neither; one tendency is never a law that dismisses another; "not seen" = not seen with
this definition and data; the popular idea "be greedy when others are fearful" is tested in its stated direction and
the measured direction is reported whatever it is; individual turns are shown, not only averages.

Written BEFORE any run of this file. Anything added after seeing results is labelled "added after the first run".

Data (free, key-less, fixed user agent "btc-research-dashboard" via dashboard.common; .env never read; a refusal is
reported and the measure dropped, never substituted). Every download goes to cache/_store/sentiment_turns/ (seeded
read-only from the shared store and the stores of signs_before_turns, follow_turn_signs and at_the_signal, exactly as
at_the_signal does: signs_before_turns' module store is pointed at this folder while its loaders run).
  BTC daily close te._daily_close (candle D closes at D+1 00:00 UTC; "day t" = candle t).
  F&G = alternative.me Crypto Fear & Greed index, https://api.alternative.me/fng/?limit=0 (daily from 2018-02-01).
    The value stamped D (00:00 UTC of D) is published at about 00:00 UTC of D, built from D-1's data; it is used from
    the close of candle D (D+1 00:00 UTC): about one day of margin, no look-ahead. Gaps carried at most 3 days.
  From signs_before_turns._assemble (its lags kept): WIKI = English Wikipedia "Bitcoin" views 7-day mean (lag 1);
    FUND = Binance BTCUSDT funding 7-day mean; LSR = Binance retail (all accounts) long/short ratio 7-day mean;
    LSD = log(LSR) - log(top-trader position ratio) (retail minus top traders; at_the_signal's lsd); BASIS = Deribit
    quarterly futures annualised basis; SKEW = Deribit 25-delta-like skew (puts over calls, skew_history store, every
    4 days); DVOL = Deribit implied volatility; KIMCHI = Upbit (Korea) premium 7-day mean; CBP = Coinbase premium
    7-day mean. NFCI and the 200-day average distance are used only as Part 4 conditions.
  Old coins moving (Part 4 only): pressure_before_moves._obm -> old_coin_selling._measure(...)["old"] (winsorised
    30-day sum of coins held 155+ days that moved), lag 7 (old_coin_selling.LAG).
  Left out, with reasons stated on the page: open interest (size, not direction), ETF and stablecoin flows (money flows,
    covered by pressure studies), VIX (US stock fear, covered by stress_family), CoinGecko trending (no history),
    Google Trends / social counts (not loaded by the project; key-less history not available to it).
PROXY (price-only "sentiment"), full history: ranks (share of the trailing 365 values <= today, >= 180 values, x100;
  attention._trail_pct) of r30 = close/close(t-30) - 1, r90 = close/close(t-90) - 1, -rv30 (30-day realised vol of
  daily log returns, INVERTED: calm = greedy) and dd = close / max(close over 365 days) - 1; PROXY = mean of the four
  ranks (all four needed), 0-100.
How much of F&G is price: on days both exist, Pearson correlation of levels and of 30-day changes; R^2 of an OLS of
  F&G on [1, the four ranks] over all overlapping days (description). RESIDUAL = F&G minus its fit from an EXPANDING
  OLS (coefficients from overlapping days strictly before t, at least 365 of them; ridge 1e-6), so it is point in time;
  its out-of-sample R^2 is reported too. RESIDUAL is "sentiment beyond what price already says".
Every measure is used as LEVEL, RANK (trailing 365 days, as above) and CHANGE30 = level(t) - level(t-30).
Greed orientation g (sign meaning "more greed / more eagerness"): +1 for F&G, PROXY, RESIDUAL, FUND, LSR, LSD, BASIS,
  KIMCHI, CBP; -1 for SKEW (more paid for protection = fear); WIKI and DVOL +1 but stated as "more attention / more
  expected movement", not greed.

Events
  (a) CYCLE turns = signs_before_turns._turns: bear start (fall 50% before a 100% rebound), bull start (rise 100%
      before a 50% fall). (b) pressure_before_moves._falls (>= 30% before a 25% rebound; the 28 big-fall starts) and
      ._rises (>= 50% before a 25% fall; the 23 big-rise starts), listed from 2012. (c) ma50 (main_signal_check: hold
      while the close is above its 50-day average) EXITS labelled by whipsaw_anatomy._label (whip = false breakdown,
      real = real decline; other / unknown not used) and ENTRIES by at_the_signal._entry_events (runner = trade result
      >= +20%, fizzle <= 0; small / open not used). Turn dates are only known afterwards.
PART 1 - OBSERVE (no rule). Per event type and measure, at the event close T (values known at that close):
  * per event: level and rank at T, T-30, T-90; change into the event = level(T) - level(T-30). Cycle turns: every
    event shown; swing events: a summary and the individual events in a second table; ma50: groups, and the real
    exits and runner entries since 2018 one by one.
  * tally: N events with a rank at T; k in the top fifth (rank >= 80) and k in the bottom fifth (<= 20); expected =
    N x share of random days in that fifth. Random days (turns): days at least 90 days from every event of the same
    scale (signs_before_turns._pool), with a value; ma50 events: all days with a value (stated).
  * path: mean rank in 12 bins of 10 days (-90..-81 ... +20..+29); per event a bin needs >= 5 values; random band =
    10th-90th percentile of 2,000 means of n random pool days' 10-day mean rank (n = events with data in that bin).
  * bull versus bear (and big fall versus big rise): median rank at T, difference, 5-95% range over 2,000 label
    shuffles; "same before both" = both types in the same extreme fifth more often than random by >= 1 event.
  * ORDER: for tops (bear, big fall) the day of the maximum of g x level in T-90..T+30 (>= 60 values), for bottoms
    (bull, big rise) the minimum; offset in days (negative = before the price turn). Per cycle turn and the median;
    swing: counts before (< -3), with (-3..+3), after (> +3) and the median. Price's own extreme is day 0 by
    construction.
  * plain sentences: F&G's named band on day T and on any day in T-30..T ("極度貪婪" >= 76 before tops, "極度恐懼"
    <= 24 before bottoms, and the opposite band too), with the random-day expectation; the same for PROXY (frequency
    matched thresholds, below) and RESIDUAL (fifths).
PART 2 - all days. Forward 30/60/90-day BTC return, share of 60-day returns > 0, median deepest drop within 60 days
  (lowest close t+1..t+60 / close t - 1), by fifth of each measure's rank and by F&G's named bands (0-24 extreme fear,
  25-44 fear, 45-55 neutral, 56-75 greed, 76-100 extreme greed); 90% interval of the mean 60-day return per group from
  a circular block bootstrap (60-day blocks, 1,000 draws, fixed seed).
  Extreme versus rest: label = 1 extreme, 0 rest. Extremes: F&G <= 24 / >= 76; PROXY <= q_fear / >= q_greed where the
  thresholds are the PROXY quantiles that give the same share of days as F&G's bands over the F&G days before
  2022-01-01 (frequency-matched); RESIDUAL and all others: rank <= 20 / >= 80. Measured with reversed_direction.
  _measure (horizon 60 days; matched effect M180 = 90 cells: past-30-day return decile x past-180-day return tercile
  x 20-day volatility tercile; 90% block bootstrap interval; halves of the sample; periods 2018-19, 2020-21, 2022-23,
  2024-now; placebo = label circularly shifted >= 180 days, 200 draws). PROXY is measured both on F&G's days and on
  its full history (extra period 2011-2017). Also M180 at 30 and 90 days, and M180 on ma50-on days and on ma50-off days
  (no placebo). Reading (reversed_direction._classify, fixed): HOLDS = same sign in both halves AND interval excludes
  zero AND placebo <= 5%; PARTLY = same sign in both halves otherwise; NOT REPEATED = sign differs (or a half has no
  extreme day). Extreme fear and extreme greed are reported separately; they need not mirror each other.
PART 3 - follow it, honest split. DISCOVERY = days before 2022-01-01; the code computes, for F&G, RESIDUAL and PROXY,
  the sign of M180 (60 days) of extreme fear and of extreme greed on discovery days only. TEST = monthly starts from
  2022-01 to 12 months before today. Rules (ma = ma50, A20 = close above its 20-day average):
    greed followed by WORSE: G_EXIT = ma AND NOT greed ("w"); G_HALF = ma x (1 - 0.5 greed) ("w50").
    greed followed by better (measured direction): G_ENTRY = ma OR (greed AND A20) ("b"); G_HALF = ma OR 0.5 x (greed
      AND A20) ("b50").
    fear followed by BETTER: F_ENTRY = ma OR ENTRY(fear) ("bx"); ENTRY = fear on any of the last 30 closes AND BTC A20,
      with a stop: on entry the stop = lowest close of the 30 closes before; a close <= stop ends it and it cannot
      restart until the condition has been off. (A close below the lowest of the prior 30 closes is also below the
      20-day average, so the stop mostly coincides with A20; how often the stop ended a spell is reported.)
    fear followed by worse (measured direction): F_EXIT = ma AND NOT fear ("w"); F_HALF ("w50").
  Each for F&G and for RESIDUAL. PROXY twin = the same rule form with PROXY's extremes (price only). Constant-exposure
  twin (rules that hold less) = ma x c, c = the rule's mean position on ma50-on days from 2022-01-01 (half_gate).
  Fair exam = reversed_direction._Exam (main_signal_check's four checks: BTC daily monthly starts, 1-hour clocks x 24
  closing hours textbook and ordinary person 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per
  clock; ETH SOL BNB XRP MSTR with their own ma50 / A20 and BTC's daily state as a market-wide state, lag 1 on
  clocks and MSTR) with starts from 2022-01; ma50 re-scored on exactly these starts. Ranking = worst of four "beats
  holding" shares with main_signal_check's eligibility (supply_demand._score_ge); median deepest loss = median over
  the clock textbook cells. Placebo: the rule's daily state circularly shifted (supply_demand._roll, k in [180,
  n-180] from its first defined day), 200 draws, 20 simulations per clock, real rule re-scored the same way.
  "Supported": hold-more rules (b, b50, bx) - ranking >= ma50's AND median deepest loss no more than 5 points worse
  AND placebo <= 5% AND ranking above the PROXY twin's; hold-less rules (w, w50) - ranking >= ma50's AND median
  deepest loss shallower AND placebo <= 5% AND ranking above the PROXY twin's AND above the constant twin's ("above" =
  half_gate._above). Holm at 5% over all candidate rules (F&G, RESIDUAL, Part 4; twins are references) on p = (hits +
  1) / 201; expected passes of the placebo bar by luck = rules x 0.05.
  Plain answers (BTC daily from 2022-01-01, decisions at the close, no costs): ENTRIES = runs of closes where the rule
  holds more than ma50: count, share that ended higher, average gain of those, average loss of the others, net
  (product). EXITS = runs where it holds less: count, how often price then fell / rose over the run, average fall
  avoided, average rise missed, net (rule equity / ma50 equity - 1 over the runs). 2022 calendar year (rule minus
  ma50) and the 2025 bear-start top (the cycle turn dated in 2025): first day the rule held less than ma50 in T-90..
  T+60, and rule minus ma50 over T-60..T+120.
PART 4 - add ONE parameter (EXPLORATION, discovery days only, then one check on test days). For extreme fear and for
  extreme greed (F&G bands) separately, each condition and each of its two sides: label 1 = extreme AND side, 0 = not
  extreme, 2 = extreme AND other side, undefined where the condition is undefined; M180 (60 days) on discovery and on
  test days (no bootstrap, no placebo). Conditions (fixed list): ma50 on / off; close above / below its 200-day
  average; FUND below zero / at or above; LSR rank > 50 high / low; old coins moving rising / falling (30-day sum above
  / below its value 30 days earlier); NFCI favorable / unfavorable (known value below / above the mean of its known
  values over the previous 364 days, at_the_signal's daily version); F&G rising / falling over the last 10 days;
  drawdown from the 365-day high deeper than 30% / not. 8 conditions x 2 sides x 2 extremes = 32 combinations looked
  at. Sharpening = dir x (M180 of the combination - M180 of the extreme alone), dir = the discovery sign of the
  extreme alone. Candidates need >= 30 discovery days on, and must be able to change ma50 (an entry rule with "ma50
  on" or an exit rule with "ma50 off" cannot). The best one for fear and the best one for greed become ONE rule each,
  in the form Part 3 uses for that direction (no half version), with the PROXY twin (PROXY extreme AND the same
  side) and, if it holds less, the constant twin; scored with the same bars and included in Holm.
PART 5 - limits: F&G from 2018 (about two cycles, four bear and four bull starts at most); built partly from price and
  volatility (measured above); recipe set by the provider and changed (the page, opened 2026-11-19, lists volatility
  25%, momentum / volume 25%, social media 15%, surveys 15% "currently paused", dominance 10%, Google Trends 10%);
  Wikipedia usage changed; one venue for positioning and options, short histories; the test period from 2022 holds
  about one and a half cycles.
Today: a row "比特幣：50 日均線｜情緒規則" with ma50's state and the best supported sentiment rule's state if any
  (持有 / 半倉 / 現金), F&G value and band, PROXY and RESIDUAL in plain words, each other measure's rank. Description,
  not a forecast. Seeds fixed. Not advice.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import at_the_signal as ats
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import follow_turn_signs as fts
from dashboard.studies2 import half_gate as hg
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import pressure_before_moves as pbm
from dashboard.studies2 import reversed_direction as rd
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa

ID = "sentiment_turns"
DAY = pd.Timedelta(days=1)
SEED = 20261123
SPLIT = pd.Timestamp("2022-01-01")
FNG_URL = "https://api.alternative.me/fng/?limit=0"
FNG_PAGE = "https://alternative.me/crypto/fear-and-greed-index/"
FEAR_MAX, GREED_MIN = 24, 76
TOP, BOT = 80.0, 20.0
RWIN, RMIN = 365, 180
AWAY, DRAWS1, SHUF = 90, 2000, 2000
BINS = tuple(range(-90, 30, 10))
H_MAIN = 60
BOOT, BLOCK = 1000, 60
COST, P_COST, P_MISS, SIMS = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS
PL_SIMS, PL_DRAWS, SHIFT_MIN = 20, 200, 180
PLACEBO_BAR, LOSS_SLACK, ALPHA, MIN_DISC = 0.05, 5.0, 0.05, 30
PERIODS = (("2018–2019", "2018-01-01", "2019-12-31"), ("2020–2021", "2020-01-01", "2021-12-31"),
           ("2022–2023", "2022-01-01", "2023-12-31"), ("2024 起", "2024-01-01", "2100-01-01"))
PERIODS_FULL = (("2011–2017", "2011-01-01", "2017-12-31"),) + PERIODS

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID

# key: (short name, plain gloss, greed sign, format, lag text)
MEAS = {
    "fg": ("恐懼貪婪指數", "alternative.me 把波動、漲跌動能與成交量、社群討論、搜尋熱度、比特幣市占合成的 0–100 分數，越低越怕",
           1, "x0", "約 1 天"),
    "proxy": ("價格版情緒（PROXY）", "只用價格做的 0–100 分數：30 天漲跌、90 天漲跌、波動（越平靜分數越高）、離一年最高的距離，"
              "各自和過去一年比的名次平均", 1, "x0", "0 天"),
    "resid": ("價格以外的情緒（殘差）", "恐懼貪婪指數減掉「光看價格那四項會給的分數」；正＝比價格顯示的更貪，負＝更怕", 1, "+1", "約 1 天"),
    "wiki": ("維基瀏覽量", "英文維基「Bitcoin」頁每天瀏覽人數（7 天平均），代表大眾關注度（不分恐懼或貪婪）", 1, "big", "1 天"),
    "fund": ("資金費率", "幣安永續合約做多的人付給做空的人的費率（7 天平均）；越高＝做多越擠", 1, "fund", "0 天"),
    "lsr": ("散戶多空比", "幣安所有帳戶裡做多帳戶數除以做空帳戶數（7 天平均）", 1, "x2", "0 天"),
    "lsd": ("散戶減大戶", "散戶多空比和大戶多空比的對數差；越高＝散戶比大戶更偏多", 1, "x2", "0 天"),
    "basis": ("期貨基差", "Deribit 季度期貨比現貨貴多少（年化 %）；越高＝越願意付錢做多", 1, "pp", "0 天"),
    "skew": ("選擇權偏斜", "賣權比買權貴多少（波動點數）；越高＝越多人花錢買跌的保險（偏怕）", -1, "pp", "0 天"),
    "dvol": ("隱含波動", "Deribit 選擇權反映的未來 30 天預期波動（不分恐懼或貪婪）", 1, "pp", "0 天"),
    "kimchi": ("韓國溢價", "韓國 Upbit 價格換成美元後比國際價格貴多少（7 天平均）；越高＝韓國散戶越急著買", 1, "pct2", "0 天"),
    "cbp": ("Coinbase 溢價", "美國 Coinbase 價格比國際價格貴多少（7 天平均）", 1, "pct2", "0 天"),
}
KEYS = tuple(MEAS)
TRIO = ("fg", "proxy", "resid")
OTHERS = tuple(k for k in KEYS if k not in TRIO)
NOT_ORIENTED = ("wiki", "dvol")
ETYPE = {"bear": "走熊開始", "bull": "走牛開始", "fall": "大跌開始", "rise": "大漲開始",
         "whip": "50 日均線假跌破出場", "real": "50 日均線出場後真的下跌", "runner": "50 日均線進場後大漲（≥ +20%）",
         "fizzle": "50 日均線進場後沒漲（≤ 0）"}
ESHORT = {"bear": "走熊", "bull": "走牛", "fall": "大跌", "rise": "大漲", "whip": "假跌破", "real": "真下跌",
          "runner": "進場大漲", "fizzle": "進場沒漲"}
TOPS, BOTTOMS = ("bear", "fall"), ("bull", "rise")
CLS = {"站得住": "站得住（兩半同向、區間不含 0、錯位 ≤ 5%）", "部分": "部分（兩半同向，但區間含 0 或錯位 > 5%）",
       "沒有重複": "沒有重複（前後兩半方向不同）", "算不出": "算不出（資料太少）"}
KIND_CN = {"w": "均線開著而且沒有這個極端才持有（出場版）", "w50": "均線開著時，有這個極端就只拿一半（半倉版）",
           "b": "均線開著，或有這個極端而且站上 20 日均線，就持有（進場版）",
           "b50": "均線開著全倉；均線關著但有這個極端而且站上 20 日均線時拿一半（半倉進場版）",
           "bx": "均線開著，或最近 30 天出現過這個極端而且站上 20 日均線就先進場（停損＝進場前 30 天最低收盤）",
           "const": "均線開著時固定只拿 c 成（持有量相同的對照）"}
HOLD_LESS = ("w", "w50")
COND = {
    "ma": ("50 日均線", "開著", "關著"), "m200": ("200 日均線", "價格在上面", "價格在下面"),
    "fund": ("資金費率", "低於 0（做空的人付錢）", "≥ 0"), "lsr": ("散戶多空比", "偏高（過去一年中位數以上）", "偏低"),
    "old": ("老幣移動", "在增加（30 天合計比 30 天前多）", "在減少"),
    "nfci": ("金融條件 NFCI", "寬鬆（低於自己過去一年平均）", "偏緊"),
    "fgchg": ("恐懼貪婪 10 天變化", "在升（比 10 天前高）", "在降"),
    "dd30": ("離一年最高", "跌超過 30%", "跌不到 30%"),
}

PRIOR = (
    "之前的研究存下來的恐懼貪婪結果（量到的，照它們存的數字）：collectors/studies.py「洗盤」（2019-09-16 起 1,084 天）：很怕而且資金費率已經是負的 "
    "96 天（11 段），30 天後比較高 85%、中位數 +13.6%；很怕但資金費率還是正的 217 天，54%、+0.8%；不算很怕 771 天，49%、-0.3%。"
    "popular_rules：每月定投「恐懼貪婪 ≤25 買兩倍、≥75 不買」2018-02 起 106 個月 3.75 倍，每月固定買 3.83 倍。indicator_scan（整體 fail）："
    "恐懼貪婪偏高那一成，18 次分開發生只有 6 次贏過平常、30 天中位數 -3.8%（平常 +2.1%）、前後兩半同方向；偏低那一成 10 段 211 天，"
    "30 天中位數 +1.4%、100 次裡 58 次比較高（平常 57 次），前後兩半方向相反。walkforward：「恐懼時買進」家族（2026 年初挑到「≤15 之後持有 30 天」）"
    "2022–2026 每年 -13.8%，同期一直抱著 +6.6%，5 年裡 2 年贏。rally_timing：含恐懼貪婪 ≤15 的恐慌訊號＋站上 20 日均線上車，排名分數 59%，"
    "三個條件 ✓✗✓、沒有勝過 50 日均線。breadth_timing：B1 再加恐懼貪婪 ≥40，排名 53%、錯位 8%，不及格。shelton_blend：恐懼貪婪 30 天平均"
    "預測下個月，樣本外 R² -6.9%（52 個月）／-8.0%（28 個月），比猜歷史平均差。ml_honest：價格＋資金費率＋恐懼貪婪猜下週漲跌 100 次對 48–52 次，"
    "和永遠猜漲（52）差不多。episodes 只把它當描述欄。稽核檔（_audit_part1／2）沒有恐懼貪婪的紀錄；裡面的「恐懼」是選擇權偏斜（最怕的那一組大跌比中間少，"
    "38 對 44／100，被列為最弱的反向項目）。這些研究各自用了不同門檻和期間，都沒有以轉折為中心看，也沒有把價格的部分拿掉。")

_d = te._d
_ok = sbt._ok
_fmt = sbt._fmt


def _pc(v, nd: int = 1) -> str:
    return f"{float(v) * 100:+.{nd}f}%" if _ok(v) else "—"


def _pt(v, nd: int = 1) -> str:
    return f"{float(v) * 100:+.{nd}f} 點" if _ok(v) else "—"


def _sh(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _r0(v) -> str:
    return f"{float(v):.0f}" if _ok(v) else "—"


def _lv(k: str, v) -> str:
    if not _ok(v):
        return "—"
    if k == "resid":
        return f"{float(v):+.1f}"
    return _fmt(v, MEAS[k][3])


def _band(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return ("極度恐懼" if v <= 24 else "恐懼" if v <= 44 else "中性" if v <= 55 else "貪婪" if v <= 75 else "極度貪婪")


# ---------- own store ----------

@contextmanager
def _store():
    """Loaders write into cache/_store/sentiment_turns/; the shared store and the stores of signs_before_turns,
    follow_turn_signs and at_the_signal are only read (copied in when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for src_dir in (REAL_STORE / sbt.ID, REAL_STORE / fts.ID, REAL_STORE / ats.ID):
        if src_dir.exists():
            for f in src_dir.glob("*.json"):
                dst = MY_STORE / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(f, dst)
                except Exception:  # noqa: BLE001 - a missing seed only means a download
                    pass
    old_sbt, old_obm = sbt.MY_STORE, pbm.STORE_OBM
    sbt.MY_STORE = MY_STORE
    pbm.STORE_OBM = MY_STORE / old_obm.name
    try:
        with sbt._own_store():
            yield
    finally:
        sbt.MY_STORE = old_sbt
        pbm.STORE_OBM = old_obm


def _fng() -> tuple[pd.Series, str | None]:
    path = MY_STORE / "fng.json"
    try:
        rows = common.get_json(FNG_URL.split("?")[0], {"limit": 0})["data"]
        got = {pd.Timestamp(int(r["timestamp"]), unit="s").strftime("%Y-%m-%d"): float(r["value"]) for r in rows}
        if len(got) < 1000:
            raise RuntimeError("恐懼貪婪指數歷史太短")
        try:
            path.write_text(json.dumps(dict(sorted(got.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        note = None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        if not path.exists():
            raise
        got = json.loads(path.read_text(encoding="utf-8"))
        note = f"恐懼貪婪指數這次取不到（{type(exc).__name__}），用之前存下來的資料。"
    return sbt._ser(got), note


def _load() -> dict:
    t0 = time.time()
    P: list[str] = []
    out: dict = {"problems": P, "dropped": {}}
    with _store():
        close = te._daily_close()
        series, dropped, probs, src, _ = sbt._assemble(close)
        P += probs
        out["dropped"].update(dropped)
        try:
            fg, note = _fng()
            if note:
                P.append(note)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"恐懼貪婪指數取不到：{type(exc).__name__}") from exc
        try:
            obm, note = pbm._obm()
            if note:
                P.append(note)
            old = ocs._measure(obm)["old"]
        except Exception as exc:  # noqa: BLE001
            old = None
            P.append(f"老幣移動資料這次取不到（{type(exc).__name__}），第四部分不含這個條件。")
        rows, problem = cr._hourly("BTCUSDT")
        if problem:
            P.append(problem)
        if len(rows) < cr.MIN_HOURS:
            raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
        price, b0, _ = cr._grid(rows)
        grid = (price, b0, cr._prehistory(b0))
    other, more = msc._load_other()
    P += more
    out.update({"close": close, "series": series, "src": src, "fg_raw": fg, "old": old, "grid": grid,
                "other": other, "load_s": time.time() - t0})
    return out


# ---------- measures ----------

def _rank(x: np.ndarray) -> np.ndarray:
    return at._trail_pct(np.asarray(x, dtype=float), RWIN, RMIN) * 100


def _expanding_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit of y on X using only rows strictly before t (>= 365 valid rows); returns (fit, valid-before count)."""
    n, k = X.shape
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    fit = np.full(n, np.nan)
    xtx = np.zeros((k, k))
    xty = np.zeros(k)
    cnt = 0
    for t in range(n):
        if cnt >= 365 and np.isfinite(X[t]).all():
            beta = np.linalg.solve(xtx + 1e-6 * np.eye(k), xty)
            fit[t] = X[t] @ beta
        if ok[t]:
            xtx += np.outer(X[t], X[t])
            xty += X[t] * y[t]
            cnt += 1
    return fit, ok


def _build(d: dict) -> dict:
    close = d["close"]
    idx = close.index
    c = close.values.astype(float)
    n = len(c)
    S = d["series"]
    L = {}
    L["fg"] = sbt._known(d["fg_raw"], idx, 0, 3).values
    cs = pd.Series(c, index=idx)
    r30 = (cs / cs.shift(30) - 1).values
    r90 = (cs / cs.shift(90) - 1).values
    rv = (np.log(cs).diff().rolling(30, min_periods=30).std(ddof=1) * math.sqrt(365)).values
    dd = (cs / cs.rolling(365, min_periods=365).max() - 1).values
    parts = {"r30": _rank(r30), "r90": _rank(r90), "calm": _rank(-rv), "dd": _rank(dd)}
    PX = np.column_stack([parts[k] for k in ("r30", "r90", "calm", "dd")])
    L["proxy"] = np.where(np.isfinite(PX).all(axis=1), PX.mean(axis=1), np.nan)
    X = np.column_stack([np.ones(n), PX])
    fit, ok = _expanding_fit(L["fg"], X)
    L["resid"] = L["fg"] - fit
    # full-sample description
    beta, *_ = np.linalg.lstsq(X[ok], L["fg"][ok], rcond=None)
    yhat = X[ok] @ beta
    y = L["fg"][ok]
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    okr = np.isfinite(L["resid"]) & np.isfinite(L["fg"])
    yo = L["fg"][okr]
    r2_oos = 1 - np.sum(L["resid"][okr] ** 2) / np.sum((yo - yo.mean()) ** 2)
    both = np.isfinite(L["fg"]) & np.isfinite(L["proxy"])
    corr_lev = float(np.corrcoef(L["fg"][both], L["proxy"][both])[0, 1])
    fgc = L["fg"] - np.concatenate([np.full(30, np.nan), L["fg"][:-30]])
    pxc = L["proxy"] - np.concatenate([np.full(30, np.nan), L["proxy"][:-30]])
    bc_ = np.isfinite(fgc) & np.isfinite(pxc)
    corr_chg = float(np.corrcoef(fgc[bc_], pxc[bc_])[0, 1])
    single = {}
    for j, kk in enumerate(("r30", "r90", "calm", "dd")):
        m = np.isfinite(L["fg"]) & np.isfinite(parts[kk])
        single[kk] = float(np.corrcoef(L["fg"][m], parts[kk][m])[0, 1])
    fitinfo = {"beta": beta, "r2": float(r2), "r2_oos": float(r2_oos), "n": int(ok.sum()), "corr_lev": corr_lev,
               "corr_chg": corr_chg, "single": single, "resid_first": idx[int(np.argmax(np.isfinite(L["resid"])))],
               "n_chg": int(bc_.sum())}
    # other measures (signs_before_turns' lags)
    have = {"wiki": "wiki", "fund": "funding", "lsr": "ls_r", "basis": "basis", "skew": "skew", "dvol": "dvol",
            "kimchi": "kimchi", "cbp": "cbp"}
    dropped = {}
    for k, sk_ in have.items():
        if sk_ in S:
            L[k] = S[sk_].reindex(idx).values.astype(float)
        else:
            dropped[k] = d["dropped"].get(sk_, "沒有組出來")
    if "ls_r" in S and "ls_t" in S:
        a, b = S["ls_r"].reindex(idx).values, S["ls_t"].reindex(idx).values
        with np.errstate(all="ignore"):
            L["lsd"] = np.where((a > 0) & (b > 0), np.log(a) - np.log(b), np.nan)
    else:
        dropped["lsd"] = "多空比資料沒有組出來"
    keys = [k for k in KEYS if k in L]
    R = {k: _rank(L[k]) for k in keys}
    C30 = {k: L[k] - np.concatenate([np.full(30, np.nan), L[k][:-30]]) for k in keys}
    # conditions for Part 4
    ma = np.nan_to_num(msc._targets(close)["ma50"].values.astype(float))
    m200 = cs.rolling(200, min_periods=200).mean().values
    cond = {"ma": np.where(True, (ma > 0.5).astype(float), np.nan),
            "m200": np.where(np.isfinite(m200), (c > m200).astype(float), np.nan)}
    if "fund" in L:
        cond["fund"] = np.where(np.isfinite(L["fund"]), (L["fund"] < 0).astype(float), np.nan)
    if "lsr" in L:
        cond["lsr"] = np.where(np.isfinite(R["lsr"]), (R["lsr"] > 50).astype(float), np.nan)
    if d.get("old") is not None:
        o = d["old"].asfreq("D")
        ch = (o - o.shift(30))
        known = ch.reindex(idx - ocs.LAG * DAY).values.astype(float)
        cond["old"] = np.where(np.isfinite(known), (known > 0).astype(float), np.nan)
    if "nfci" in S:
        nv = S["nfci"].reindex(idx).values.astype(float)
        mean364 = pd.Series(nv).shift(1).rolling(364, min_periods=300).mean().values
        cond["nfci"] = np.where(np.isfinite(nv) & np.isfinite(mean364), (nv < mean364).astype(float), np.nan)
    fg10 = L["fg"] - np.concatenate([np.full(10, np.nan), L["fg"][:-10]])
    cond["fgchg"] = np.where(np.isfinite(fg10), (fg10 > 0).astype(float), np.nan)
    cond["dd30"] = np.where(np.isfinite(dd), (dd < -0.30).astype(float), np.nan)
    a20 = sd._above20(c)
    return {"idx": idx, "c": c, "n": n, "L": L, "R": R, "C30": C30, "keys": keys, "parts": parts, "fit": fitinfo,
            "dropped": dropped, "ma": ma, "a20": a20, "cond": cond, "dd": dd}


def _extremes(B: dict) -> dict:
    """{key: (fear-side bool array, greed-side bool array, defined)} - fear side = low g x value."""
    L, R, idx = B["L"], B["R"], B["idx"]
    out = {}
    fg = L["fg"]
    dfg = np.isfinite(fg)
    out["fg"] = (dfg & (fg <= FEAR_MAX), dfg & (fg >= GREED_MIN), dfg)
    disc = (idx < SPLIT) & dfg & np.isfinite(L["proxy"])
    sf = float(np.mean(fg[disc] <= FEAR_MAX))
    sg = float(np.mean(fg[disc] >= GREED_MIN))
    qf = float(np.quantile(L["proxy"][disc], sf))
    qg = float(np.quantile(L["proxy"][disc], 1 - sg))
    px = L["proxy"]
    dpx = np.isfinite(px)
    out["proxy"] = (dpx & (px <= qf), dpx & (px >= qg), dpx)
    B["pq"] = (qf, qg, sf, sg)
    for k in B["keys"]:
        if k in ("fg", "proxy"):
            continue
        r = R[k]
        dk = np.isfinite(r)
        lo, hi = dk & (r <= BOT), dk & (r >= TOP)
        out[k] = (hi, lo, dk) if MEAS[k][2] < 0 else (lo, hi, dk)
    return out


# ---------- events ----------

def _events(B: dict, close: pd.Series) -> dict:
    idx, c, ma = B["idx"], B["c"], B["ma"]
    turns = sbt._turns(close)
    E = {"bear": [t["i"] for t in turns["bear"]], "bull": [t["i"] for t in turns["bull"]]}
    pos = {d_: i for i, d_ in enumerate(idx)}
    E["fall"] = [pos[e["d"]] for e in pbm._falls(close) if e["d"] in pos]
    E["rise"] = [pos[e["d"]] for e in pbm._rises(close) if e["d"] in pos]
    ex = wa._exits(ma)
    lab = [wa._label(c, ma, int(e)) for e in ex]
    E["whip"] = [int(e) for e, l_ in zip(ex, lab) if l_ == "whip"]
    E["real"] = [int(e) for e, l_ in zip(ex, lab) if l_ == "real"]
    ent = ats._entry_events(close, ma)
    E["runner"] = [r["i"] for r in ent["rows"] if r["kind"] == "runner"]
    E["fizzle"] = [r["i"] for r in ent["rows"] if r["kind"] == "fizzle"]
    info = {"turns": turns, "entries": {r["i"]: r for r in ent["rows"]},
            "falls": pbm._falls(close), "rises": pbm._rises(close)}
    return {"E": E, "info": info}


def _pools(B: dict, E: dict) -> dict:
    n = B["n"]
    cyc = sbt._pool(n, E["bear"] + E["bull"], 0)
    sw = sbt._pool(n, E["fall"] + E["rise"], 0)
    allp = np.arange(n)
    return {"bear": cyc, "bull": cyc, "fall": sw, "rise": sw, "whip": allp, "real": allp, "runner": allp,
            "fizzle": allp}


def _m10(r: np.ndarray) -> np.ndarray:
    """Mean of r over t..t+9 (>= 5 values)."""
    s = pd.Series(r[::-1]).rolling(10, min_periods=5).mean().values[::-1]
    return s


# ---------- Part 1 ----------

def _part1(B: dict, EV: dict) -> dict:
    E, R, L, n = EV["E"], B["R"], B["L"], B["n"]
    pools = _pools(B, E)
    rng = np.random.default_rng(SEED)
    out = {"tally": {}, "path": {}, "contrast": {}, "order": {}, "plain": {}}
    M10 = {k: _m10(R[k]) for k in B["keys"]}
    for ty, ev in E.items():
        pool = pools[ty]
        for k in B["keys"]:
            r = R[k]
            vals = np.array([r[i] for i in ev], dtype=float)
            have = np.isfinite(vals)
            N = int(have.sum())
            rp = r[pool]
            rp = rp[np.isfinite(rp)]
            pt = float(np.mean(rp >= TOP)) if len(rp) >= 100 else None
            pb = float(np.mean(rp <= BOT)) if len(rp) >= 100 else None
            ch = np.array([r[i] - r[i - 30] if i >= 30 else np.nan for i in ev], dtype=float)
            out["tally"][(ty, k)] = {"N": N, "top": int((vals[have] >= TOP).sum()), "bot": int((vals[have] <= BOT).sum()),
                                     "pt": pt, "pb": pb, "med": float(np.median(vals[have])) if N else None,
                                     "chg": float(np.nanmedian(ch)) if np.isfinite(ch).any() else None}
            # path
            m10 = M10[k]
            pv = m10[pool]
            pv = pv[np.isfinite(pv)]
            row = []
            for b in BINS:
                xs = np.array([m10[i + b] if 0 <= i + b < n else np.nan for i in ev], dtype=float)
                xs = xs[np.isfinite(xs)]
                if len(xs) == 0 or len(pv) < 100:
                    row.append(None)
                    continue
                draws = rng.choice(pv, size=(DRAWS1, len(xs)), replace=True).mean(axis=1)
                lo, hi = np.percentile(draws, [10, 90])
                row.append({"m": float(xs.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(xs))})
            out["path"][(ty, k)] = row
    crng = np.random.default_rng(SEED + 1)
    for a, b in (("bear", "bull"), ("fall", "rise"), ("real", "whip"), ("runner", "fizzle")):
        for k in B["keys"]:
            va = np.array([R[k][i] for i in E[a]], dtype=float)
            vb = np.array([R[k][i] for i in E[b]], dtype=float)
            va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            if len(va) < 2 or len(vb) < 2:
                out["contrast"][(a, k)] = None
                continue
            obs = float(np.median(va) - np.median(vb))
            pool_ = np.concatenate([va, vb])
            sims = np.empty(SHUF)
            for s in range(SHUF):
                p = crng.permutation(pool_)
                sims[s] = np.median(p[:len(va)]) - np.median(p[len(va):])
            lo, hi = np.percentile(sims, [5, 95])
            out["contrast"][(a, k)] = {"na": len(va), "nb": len(vb), "ma": float(np.median(va)),
                                       "mb": float(np.median(vb)), "diff": obs, "lo": float(lo), "hi": float(hi),
                                       "share": float(np.mean(np.abs(sims) >= abs(obs) - 1e-12))}
    # order
    for ty in TOPS + BOTTOMS:
        for k in B["keys"]:
            g = MEAS[k][2]
            x = g * L[k]
            offs = []
            for i in E[ty]:
                a_, b_ = i - 90, i + 30
                if a_ < 0 or b_ >= n:
                    offs.append(None)
                    continue
                w = x[a_:b_ + 1]
                if np.isfinite(w).sum() < 60:
                    offs.append(None)
                    continue
                j = int(np.nanargmax(w)) if ty in TOPS else int(np.nanargmin(w))
                offs.append(j - 90)
            out["order"][(ty, k)] = offs
    # plain F&G band sentences
    fg = L["fg"]
    gr = np.where(np.isfinite(fg), (fg >= GREED_MIN).astype(float), np.nan)
    fe = np.where(np.isfinite(fg), (fg <= FEAR_MAX).astype(float), np.nan)

    def any30(f):
        return pd.Series(f).rolling(31, min_periods=20).max().values
    ag, af = any30(gr), any30(fe)
    for ty in E:
        pool = pools[ty]
        res = {}
        for nm, day, anyw in (("greed", gr, ag), ("fear", fe, af)):
            vals = np.array([day[i] for i in E[ty]], dtype=float)
            va = np.array([anyw[i] for i in E[ty]], dtype=float)
            have = np.isfinite(vals)
            pdv, pav = day[pool], anyw[pool]
            res[nm] = {"N": int(have.sum()), "k": int(np.nansum(vals[have])), "ka": int(np.nansum(va[have])),
                       "p": float(np.nanmean(pdv)) if np.isfinite(pdv).sum() > 100 else None,
                       "pa": float(np.nanmean(pav)) if np.isfinite(pav).sum() > 100 else None}
        out["plain"][ty] = res
    out["pools"] = pools
    return out


def _side_count(B, X, ty, E, pools, key, side) -> dict:
    """Events of type ty at which the fixed extreme (Part 2 / 3 definition) of `key` was on, versus random pool days."""
    f, g, dfn = X[key]
    arr = f if side == "fear" else g
    vals = [bool(arr[i]) for i in E[ty] if dfn[i]]
    pool = pools[ty]
    pp = pool[dfn[pool]]
    p = float(arr[pp].mean()) if len(pp) > 100 else None
    return {"N": len(vals), "k": int(sum(vals)), "p": p}


# ---------- Part 2 ----------

def _deep60(c: np.ndarray) -> np.ndarray:
    n = len(c)
    out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(c, 60)
    mins = w.min(axis=1)
    out[:n - 60] = mins[1:n - 59] / c[:n - 60] - 1
    return out


def _group_table(F: dict, grp: np.ndarray, ngroups: int, seed: int, c: np.ndarray) -> list[dict]:
    v30, v60, v90 = F["fwd"][30], F["fwd"][60], F["fwd"][90]
    d60 = _deep60(c)
    S = (grp >= 0) & np.isfinite(v60)
    pos = np.flatnonzero(S)
    m = len(pos)
    rows = []
    if m < 2 * BLOCK:
        return rows
    rng = np.random.default_rng(seed)
    nb = -(-m // BLOCK)
    st = rng.integers(0, m, (BOOT, nb))
    bi = pos[((st[:, :, None] + np.arange(BLOCK)) % m).reshape(BOOT, -1)[:, :m]]
    gb = grp[bi]
    vb = v60[bi]
    for g_ in range(ngroups):
        on = S & (grp == g_)
        if on.sum() == 0:
            rows.append(None)
            continue
        mask = gb == g_
        with np.errstate(all="ignore"):
            bm = np.where(mask, vb, 0).sum(1) / mask.sum(1)
        lo, hi = rd._q(bm)
        rows.append({"n": int(on.sum()), "f30": float(np.nanmean(v30[on])), "f60": float(np.nanmean(v60[on])),
                     "lo": lo, "hi": hi, "f90": float(np.nanmean(v90[on & np.isfinite(v90)])) if
                     (on & np.isfinite(v90)).any() else None, "up": float(np.mean(v60[on] > 0)),
                     "deep": float(np.nanmedian(d60[on])) if np.isfinite(d60[on]).any() else None})
    return rows


def _label(ext: np.ndarray, dfn: np.ndarray) -> np.ndarray:
    return np.where(dfn, ext.astype(float), np.nan)


def _part2(B: dict, X: dict, F: dict) -> dict:
    idx, c = B["idx"], B["c"]
    out = {"fifths": {}, "bands": None, "cls": {}}
    for j, k in enumerate(B["keys"]):
        r = B["R"][k]
        grp = np.where(np.isfinite(r), np.digitize(np.nan_to_num(r), [20.0, 40.0, 60.0, 80.0]), -1)
        out["fifths"][k] = _group_table(F, grp, 5, SEED + 10 + j, c)
    fg = B["L"]["fg"]
    bands = np.full(len(fg), -1)
    ok = np.isfinite(fg)
    bands[ok] = np.select([fg[ok] <= 24, fg[ok] <= 44, fg[ok] <= 55, fg[ok] <= 75], [0, 1, 2, 3], 4)
    out["bands"] = _group_table(F, bands, 5, SEED + 9, c)
    ma_on = B["ma"] > 0.5
    allp = np.ones(len(idx), bool)
    fgdays = np.isfinite(fg)
    items = [(k, k, allp, PERIODS) for k in B["keys"]]
    items.append(("proxy_full", "proxy", allp, PERIODS_FULL))
    for j, (name, k, pop0, per) in enumerate(items):
        fear, greed, dfn = X[k]
        pop = pop0 & fgdays if name == "proxy" else pop0
        for s, arr in (("fear", fear), ("greed", greed)):
            lab = _label(arr, dfn)
            seed = SEED + 100 + 10 * j + (0 if s == "fear" else 5)
            r = rd._measure(F, lab, "rest", H_MAIN, pop, per, seed, placebo=True, boot=True)
            res = {"main": r, "cls": rd._classify(r)}
            for h in (30, 90):
                x = rd._measure(F, lab, "rest", h, pop, per, seed, placebo=False, boot=False)
                res[f"h{h}"] = x["m3"] if x else None
            for nm, pm in (("on", ma_on), ("off", ~ma_on)):
                x = rd._measure(F, lab, "rest", H_MAIN, pop & pm, per, seed, placebo=False, boot=True)
                res[nm] = x
            out["cls"][(name, s)] = res
    return out


# ---------- Part 3 / 4: rules and the exam ----------

def _combine(kind: str, ma, a20, f, c: float = 1.0) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "const":
        return ma * c
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "b":
        return np.maximum(ma, f * a20)
    if kind == "b50":
        return np.maximum(ma, 0.5 * f * a20)
    if kind == "bx":
        return np.maximum(ma, f)
    if kind == "w":
        return ma * (1.0 - f)
    if kind == "w50":
        return ma * (1.0 - 0.5 * f)
    raise ValueError(kind)


class _Exam(rd._Exam):
    """reversed_direction's exam (main_signal_check's four checks on starts from 2022-01) with this study's ways to
    combine a daily state with ma50 (b, b50, bx, w, w50, const)."""

    def btc_pos(self, flag, kind: str = "ma50", c: float = 1.0) -> np.ndarray:
        return _combine(kind, self.ma, self.a20, self._at(flag, self.close.index, 0), c)

    def summary(self, flag, ki: int, sims: int = SIMS, kind: str = "ma50", c: float = 1.0) -> dict:  # noqa: PLR0913
        pos = self.btc_pos(flag, kind, c)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, msc.YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        f = self._at(flag, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = _combine(kind, self.cma[h], self.ca20[h], f, c)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, msc.YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(msc.SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts,
                                msc.YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            w = _combine(kind, o["ma"], o["a20"], self._at(flag, o["close"].index, o["lag"]), c)
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_shallow": int((ddd > self.dhold[1]).sum()),
             "d_edge": float(np.median(dr - self.dhold[0])),
             "c_n": edge.size, "c_starts": n, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


def _entry_state(ext: np.ndarray, c: np.ndarray, a20: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ENTRY: extreme on any of the last 30 closes AND close above its 20-day average, with a stop at the lowest close
    of the 30 closes before the entry; after a stop it cannot restart until the condition has been off.
    Returns (state 0/1, stopped-on-this-close flags)."""
    n = len(c)
    recent = pd.Series(ext.astype(float)).rolling(30, min_periods=1).max().values > 0.5
    cond = recent & (a20 > 0.5)
    st, stopped = np.zeros(n), np.zeros(n, bool)
    on, lock, stop = False, False, np.nan
    for t in range(n):
        if not cond[t]:
            on, lock = False, False
            continue
        if on:
            if c[t] <= stop:
                on, lock = False, True
                stopped[t] = True
            else:
                st[t] = 1.0
        elif not lock:
            stop = float(np.min(c[max(0, t - 30):t])) if t > 0 else -np.inf
            if c[t] > stop:
                on = True
                st[t] = 1.0
    return st, stopped


def _flag_for(kind: str, ext: np.ndarray, B: dict) -> tuple[pd.Series, np.ndarray | None]:
    if kind == "bx":
        st, stopped = _entry_state(ext, B["c"], B["a20"])
        return pd.Series(st, index=B["idx"]), stopped
    return pd.Series(ext.astype(float), index=B["idx"]), None


def _disc_sign(F: dict, lab: np.ndarray, B: dict, h: int = H_MAIN, pop_mask=None) -> tuple[float | None, int]:
    pop = (B["idx"] < SPLIT) if pop_mask is None else pop_mask
    r = rd._measure(F, lab, "rest", h, pop, (), SEED, placebo=False, boot=False)
    if not r:
        return None, 0
    return r["m3"], r["n_on"]


def _spells(B: dict, pos: np.ndarray, first: pd.Timestamp, stopped: np.ndarray | None = None) -> dict:
    """Runs where the rule holds more (entries) / less (exits) than ma50, BTC daily from `first`, no costs."""
    idx, c, ma = B["idx"], B["c"], B["ma"]
    diff = pos - ma
    a0 = int(idx.searchsorted(first))
    n = len(c)
    out = {"ent": [], "ex": []}
    for sign, key in ((1, "ent"), (-1, "ex")):
        m = (sign * diff > 1e-9)
        m[:a0] = False
        t = a0
        while t < n:
            if not m[t]:
                t += 1
                continue
            s = t
            while t < n and m[t]:
                t += 1
            e = min(t, n - 1)
            size = float(np.mean(np.abs(diff[s:t])))
            ret = c[e] / c[s] - 1
            how = ""
            if key == "ent":
                if t >= n:
                    how = "還在進行"
                elif ma[t] > 0.5:
                    how = "50 日均線接手"
                elif stopped is not None and stopped[t]:
                    how = "停損"
                else:
                    how = "條件結束"
            out[key].append({"s": idx[s], "e": idx[e], "days": int(t - s), "ret": float(ret), "size": size,
                             "open": t >= n, "how": how})
    return out


def _spell_sum(sp: list[dict], entry: bool) -> dict:
    if not sp:
        return {"n": 0}
    rets = np.array([x["ret"] * x["size"] for x in sp])
    up = rets > 0
    net = float(np.prod(1 + rets) - 1) if entry else float(np.prod(1 - rets) - 1)
    return {"n": len(sp), "up": int(up.sum()), "down": int((~up).sum()),
            "avg_up": float(rets[up].mean()) if up.any() else None,
            "avg_down": float(rets[~up].mean()) if (~up).any() else None, "net": net,
            "stops": int(sum(1 for x in sp if x["how"] == "停損"))}


def _year_diff(B: dict, pos: np.ndarray, year: int) -> float:
    idx, c, ma = B["idx"], B["c"], B["ma"]
    ch = np.concatenate([[0.0], c[1:] / c[:-1] - 1])
    sel = np.flatnonzero(idx.year == year)
    sel = sel[sel >= 1]
    r_rule = np.prod(1 + pos[sel - 1] * ch[sel]) - 1
    r_ma = np.prod(1 + ma[sel - 1] * ch[sel]) - 1
    return float(r_rule - r_ma)


def _window_diff(B: dict, pos: np.ndarray, T: int, a: int, b: int) -> float | None:
    c, ma = B["c"], B["ma"]
    n = len(c)
    s, e = max(1, T + a), min(n - 1, T + b)
    if e <= s:
        return None
    ch = c[1:] / c[:-1] - 1
    rr = np.prod(1 + pos[s - 1:e] * ch[s - 1:e]) - 1
    rm = np.prod(1 + ma[s - 1:e] * ch[s - 1:e]) - 1
    return float(rr - rm)


def _part3(B: dict, X: dict, F: dict, d: dict, EV: dict) -> dict:
    idx = B["idx"]
    price, b0, pre = d["grid"]
    today = pd.Timestamp(common.today_utc())
    last = (today - pd.DateOffset(months=12)).to_period("M").to_timestamp()
    ex = _Exam(d["close"], d["other"], price, b0, pre, SPLIT, last)
    base = ex.summary(None, 1, SIMS, "ma50")
    base_small = ex.summary(None, 1, PL_SIMS, "ma50")
    disc = {}
    for k in ("fg", "resid", "proxy"):
        fear, greed, dfn = X[k]
        disc[(k, "fear")] = _disc_sign(F, _label(fear, dfn), B)
        disc[(k, "greed")] = _disc_sign(F, _label(greed, dfn), B)
    rules = []  # each: dict(name, src, side, kind, ext, cand)
    for k in ("fg", "resid"):
        fear, greed, dfn = X[k]
        gs, _ = disc[(k, "greed")]
        fs, _ = disc[(k, "fear")]
        if gs is not None:
            kinds = ("w", "w50") if gs < 0 else ("b", "b50")
            for kd in kinds:
                rules.append({"id": f"{k}_greed_{kd}", "src": k, "side": "greed", "kind": kd, "ext": greed,
                              "twin_ext": X["proxy"][1], "dir": gs})
        if fs is not None:
            kinds = ("bx",) if fs > 0 else ("w", "w50")
            for kd in kinds:
                rules.append({"id": f"{k}_fear_{kd}", "src": k, "side": "fear", "kind": kd, "ext": fear,
                              "twin_ext": X["proxy"][0], "dir": fs})
    return {"ex": ex, "base": base, "base_small": base_small, "disc": disc, "rules": rules, "last": last}


def _score_rules(B: dict, P3: dict, rules: list[dict], ki0: int) -> None:
    ex, base, base_small = P3["ex"], P3["base"], P3["base_small"]
    idx = B["idx"]
    after = idx >= SPLIT
    on_after = after & (B["ma"] > 0.5)
    for j, r in enumerate(rules):
        ki = ki0 + j
        flag, stopped = _flag_for(r["kind"], r["ext"], B)
        r["flag"], r["stopped"] = flag, stopped
        x = ex.summary(flag, ki, SIMS, r["kind"])
        pos = ex.btc_pos(flag, r["kind"])
        r["x"], r["pos"] = x, pos
        r["differ"] = int((np.abs(pos[after] - B["ma"][after]) > 1e-9).sum())
        r["time"] = float(pos[after].mean())
        tflag, _ = _flag_for(r["kind"], r["twin_ext"], B)
        r["twin"] = ex.summary(tflag, ki, SIMS, r["kind"])
        r["twin_pos"] = ex.btc_pos(tflag, r["kind"])
        if r["kind"] in HOLD_LESS:
            cc = float(pos[on_after].mean()) if on_after.any() else 1.0
            r["c"] = cc
            r["const"] = ex.summary(None, ki, SIMS, "const", cc)
        else:
            r["const"] = None
        if r["differ"] == 0:
            r["plc"], r["p"] = None, None
            continue
        small = ex.summary(flag, ki, PL_SIMS, r["kind"])
        s0 = r["s0"]
        nn = int((idx >= s0).sum())
        prng = np.random.default_rng(SEED + 7000 + ki)
        hits = 0
        for k_ in prng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, nn - SHIFT_MIN), PL_DRAWS):
            rf = sd._roll(flag, s0, int(k_))
            hits += int(sd._score_ge(ex.summary(rf, ki, PL_SIMS, r["kind"]), small))
        r["plc"] = hits / PL_DRAWS
        r["p"] = (hits + 1) / (PL_DRAWS + 1)
    for r in rules:
        x = r["x"]
        score = bool(sd._score_ge(x, base))
        if r["kind"] in HOLD_LESS:
            loss = x["c_deep_med"] > base["c_deep_med"] + 1e-12
            const_ok = bool(hg._above(x, r["const"]))
        else:
            loss = x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK - 1e-12
            const_ok = True
        twin_ok = bool(hg._above(x, r["twin"]))
        plc_ok = r["plc"] is not None and r["plc"] <= PLACEBO_BAR
        r.update({"score": score, "loss": bool(loss), "twin_ok": twin_ok, "const_ok": const_ok, "plc_ok": plc_ok,
                  "pass": bool(score and loss and plc_ok and twin_ok and const_ok and r["differ"] > 0)})
        r["sp"] = _spells(B, r["pos"], SPLIT, r["stopped"])
        r["y2022"] = _year_diff(B, r["pos"], 2022)


# ---------- Part 4 ----------

def _part4(B: dict, X: dict, F: dict, P3: dict) -> dict:
    idx = B["idx"]
    disc_pop = idx < SPLIT
    test_pop = idx >= SPLIT
    fear, greed, dfn = X["fg"]
    out = {"rows": {}, "best": {}, "looked": 0}
    for side, arr in (("fear", fear), ("greed", greed)):
        dir_, _ = P3["disc"][("fg", side)]
        base_d, nb = _disc_sign(F, _label(arr, dfn), B)
        base_t, nbt = _disc_sign(F, _label(arr, dfn), B, pop_mask=test_pop)
        rows = [{"cond": None, "sv": None, "d": base_d, "nd": nb, "t": base_t, "nt": nbt}]
        best = None
        for ck, cv in B["cond"].items():
            for sv in (1.0, 0.0):
                out["looked"] += 1
                okc = np.isfinite(cv)
                lab = np.where(dfn & okc, np.where(arr & (cv == sv), 1.0, np.where(arr, 2.0, 0.0)), np.nan)
                ed, nd = _disc_sign(F, lab, B, pop_mask=disc_pop)
                et, nt = _disc_sign(F, lab, B, pop_mask=test_pop)
                entry = (dir_ is not None and ((side == "fear" and dir_ > 0) or (side == "greed" and dir_ > 0)))
                frozen = (ck == "ma" and ((entry and sv == 1.0) or (not entry and sv == 0.0)))
                row = {"cond": ck, "sv": sv, "d": ed, "nd": nd, "t": et, "nt": nt, "frozen": frozen}
                if _ok(ed) and _ok(base_d) and dir_ is not None:
                    row["sharp"] = (1 if dir_ > 0 else -1) * (ed - base_d)
                    if nd >= MIN_DISC and not frozen and (best is None or row["sharp"] > best["sharp"]):
                        best = row
                rows.append(row)
        out["rows"][side] = rows
        out["best"][side] = best
        out.setdefault("dir", {})[side] = dir_
    rules = []
    for side, arr, parr in (("fear", fear, X["proxy"][0]), ("greed", greed, X["proxy"][1])):
        b = out["best"].get(side)
        dir_ = out["dir"].get(side)
        if not b or dir_ is None:
            continue
        cv = B["cond"][b["cond"]]
        m = np.isfinite(cv) & (cv == b["sv"])
        if side == "fear":
            kind = "bx" if dir_ > 0 else "w"
        else:
            kind = "b" if dir_ > 0 else "w"
        rules.append({"id": f"p4_{side}_{b['cond']}", "src": "fg+", "side": side, "kind": kind, "ext": arr & m,
                      "twin_ext": parr & m, "dir": dir_, "cond": b["cond"], "sv": b["sv"]})
    out["rules"] = rules
    return out


# ---------- text helpers ----------

def _rname(r: dict) -> str:
    src = {"fg": "恐懼貪婪", "resid": "殘差", "fg+": "恐懼貪婪＋一個條件"}[r["src"]]
    side = "極度恐懼" if r["side"] == "fear" else "極度貪婪"
    if r["src"] == "resid":
        side = "殘差最低五分之一（比價格更怕）" if r["side"] == "fear" else "殘差最高五分之一（比價格更貪）"
    extra = ""
    if r.get("cond"):
        nm, a, b = COND[r["cond"]]
        extra = f"＋{nm}{a if r['sv'] == 1.0 else b}"
    short = {"w": "出場", "w50": "半倉", "b": "進場", "b50": "半倉進場", "bx": "提早進場"}[r["kind"]]
    return f"{src}｜{side}{extra}｜{short}"


def _yn(v) -> str:
    return "✓" if v else "✗"


def _exam_cells(x: dict) -> str:
    return f"{x['low'] * 100:.0f}%（{'、'.join(f'{v * 100:.0f}' for v in x['checks'])}）" + ("" if x["eligible"] else "，虧損條件沒過")


# ---------- tables ----------

def _cell(B, k, i) -> str:
    if i < 0 or i >= B["n"]:
        return "—"
    v, r = B["L"][k][i], B["R"][k][i]
    if not _ok(v):
        return "沒資料"
    return f"{_lv(k, v)}（{_r0(r)}）"


def _event_rows(B: dict, ev: list[int], ty: str) -> list[list]:
    rows = []
    idx, c = B["idx"], B["c"]
    for i in ev:
        fg = B["L"]["fg"][i]
        rows.append([ESHORT[ty], _d(idx[i]), f"{c[i]:,.0f}",
                     f"{_cell(B, 'fg', i)}｜{_cell(B, 'fg', i - 30)}｜{_cell(B, 'fg', i - 90)}"
                     + (f"｜{_band(fg)}" if _ok(fg) else ""),
                     _pt(None) if not _ok(fg) or i < 30 or not _ok(B["L"]["fg"][i - 30]) else
                     f"{fg - B['L']['fg'][i - 30]:+.0f} 分",
                     f"{_cell(B, 'proxy', i)}｜{_cell(B, 'proxy', i - 30)}｜{_cell(B, 'proxy', i - 90)}",
                     f"{_cell(B, 'resid', i)}｜{_cell(B, 'resid', i - 30)}｜{_cell(B, 'resid', i - 90)}"])
    return rows


def _other_rows(B: dict, ev: list[int], ty: str, keys: tuple) -> list[list]:
    rows = []
    for i in ev:
        cells = []
        for k in keys:
            if k not in B["L"]:
                cells.append("拿掉了")
                continue
            r0, r30 = B["R"][k][i], B["R"][k][i - 30] if i >= 30 else np.nan
            cells.append("沒資料" if not _ok(B["L"][k][i]) else
                         f"{_lv(k, B['L'][k][i])}（{_r0(r0)}；30 天前 {_r0(r30)}）")
        rows.append([ESHORT[ty], _d(B["idx"][i])] + cells)
    return rows


def _tables(B, X, EV, P1, P2, P3, P4, rules_all) -> list[dict]:
    T = []
    E = EV["E"]
    idx = B["idx"]
    fit = B["fit"]
    keys = B["keys"]
    # 0. measures
    rows = []
    for k in KEYS:
        if k in B["L"]:
            f = np.flatnonzero(np.isfinite(B["R"][k]))
            rows.append([MEAS[k][0], MEAS[k][1], MEAS[k][4], _d(idx[f[0]]) if len(f) else "—", "用了"])
        else:
            rows.append([MEAS[k][0], MEAS[k][1], MEAS[k][4], "—", f"拿掉：{B['dropped'].get(k, '')}"[:80]])
    for nm, why in (("未平倉量", "是部位大小，不分多空方向"), ("ETF、穩定幣流量", "是資金流向，買賣壓力的研究已經看過"),
                    ("VIX 美股恐慌指數", "是美股的恐懼，stress_family 看過"), ("CoinGecko 熱門搜尋", "沒有歷史資料"),
                    ("Google 搜尋熱度、社群貼文數", "專案沒有載入，也沒有免金鑰的歷史資料")):
        rows.append([nm, "—", "—", "—", f"沒用：{why}"])
    T.append({"caption": "用到的情緒指標（名次＝和自己過去 365 天比，0–100，至少要先累積 180 天）。每個值都只用當時已經公布的資料。",
              "columns": ["指標", "白話說明", "晚幾天才知道", "名次從哪天起", "有沒有用"], "rows": rows})
    # 1. how much F&G is price
    b = fit["beta"]
    rows = [["恐懼貪婪和價格版情緒（PROXY）的相關（水準）", f"{fit['corr_lev']:+.2f}", f"{fit['n']:,} 天"],
            ["30 天變化的相關", f"{fit['corr_chg']:+.2f}", f"{fit['n_chg']:,} 天"],
            ["用四個價格名次回歸，能解釋恐懼貪婪變動的比例（R²，全期一次配）", f"{fit['r2'] * 100:.0f}%", f"{fit['n']:,} 天"],
            ["同一個回歸只用以前的資料逐日配，解釋的比例（樣本外 R²）", f"{fit['r2_oos'] * 100:.0f}%", f"殘差從 {_d(fit['resid_first'])} 起"],
            ["係數：30 天漲跌名次｜90 天漲跌名次｜平靜名次｜離一年最高名次（每 1 名次的分數）",
             f"{b[1]:+.2f}｜{b[2]:+.2f}｜{b[3]:+.2f}｜{b[4]:+.2f}", "全期一次配"]]
    rows += [[f"恐懼貪婪和單一價格名次的相關：{nm}", f"{fit['single'][kk]:+.2f}", ""] for kk, nm in
           (("r30", "30 天漲跌"), ("r90", "90 天漲跌"), ("calm", "平靜（波動越小越高）"), ("dd", "離一年最高"))]
    T.append({"caption": "量到的：恐懼貪婪指數有多少只是在重述價格？PROXY＝只用價格做的 0–100 分數；殘差＝恐懼貪婪減掉「用以前的資料配出來、"
                         "光看價格會給的分數」，就是「價格以外的情緒」。",
              "columns": ["項目", "數字", "樣本"], "rows": rows})
    # 2. cycle turn lists with trio
    for ty in ("bear", "bull"):
        T.append({"caption": f"第一部分・{ETYPE[ty]}（之後跌 50% 以上／漲 100% 以上，事後才知道）每一次：每格＝數值（名次），"
                             "依序是轉折當天｜30 天前｜90 天前；恐懼貪婪另附當天的分級。名次 ≥80 是高五分之一、≤20 是低五分之一。",
                  "columns": ["類型", "轉折日", "收盤（美元）", "恐懼貪婪：當天｜-30｜-90", "恐懼貪婪 30 天變化",
                              "價格版情緒 PROXY：當天｜-30｜-90", "殘差：當天｜-30｜-90"],
                  "rows": _event_rows(B, E[ty], ty)})
    half = (len(OTHERS) + 1) // 2
    for ty in ("bear", "bull"):
        for part, ks in enumerate((OTHERS[:half], OTHERS[half:])):
            evs = [i for i in E[ty] if any(k in B["L"] and _ok(B["L"][k][i]) for k in ks)]
            T.append({"caption": f"第一部分・{ETYPE[ty]}，其他情緒指標（第 {part + 1} 組）：轉折當天的數值（名次；30 天前的名次）。"
                                 "只列至少一個指標有資料的轉折。",
                      "columns": ["類型", "轉折日"] + [MEAS[k][0] for k in ks], "rows": _other_rows(B, evs, ty, ks)})
    # 3. swing events individual (trio)
    for ty in ("fall", "rise"):
        T.append({"caption": f"第一部分・{ETYPE[ty]}（較小尺度，pressure_before_moves 的定義）每一次：格式同上。",
                  "columns": ["類型", "轉折日", "收盤（美元）", "恐懼貪婪：當天｜-30｜-90", "恐懼貪婪 30 天變化",
                              "價格版情緒 PROXY：當天｜-30｜-90", "殘差：當天｜-30｜-90"],
                  "rows": _event_rows(B, E[ty], ty)})
    # 4. tally
    for grp in (("bear", "bull"), ("fall", "rise"), ("real", "whip"), ("runner", "fizzle")):
        rows = []
        for ty in grp:
            for k in keys:
                t = P1["tally"][(ty, k)]
                if not t["N"]:
                    rows.append([ETYPE[ty], MEAS[k][0], "0", "—", "—", "—", "—"])
                    continue
                rows.append([ETYPE[ty], MEAS[k][0], str(t["N"]),
                             f"{t['top']}（隨機 {t['pt'] * t['N']:.1f}）" if t["pt"] is not None else str(t["top"]),
                             f"{t['bot']}（隨機 {t['pb'] * t['N']:.1f}）" if t["pb"] is not None else str(t["bot"]),
                             _r0(t["med"]), f"{t['chg']:+.0f}" if _ok(t["chg"]) else "—"])
        pooltxt = ("隨機日子＝離同一尺度任何轉折 90 天以上的日子" if grp[0] in ("bear", "fall")
                   else "隨機日子＝所有有資料的日子（50 日均線的出場、進場本身很頻繁，沒有排除）")
        T.append({"caption": f"第一部分・算次數：事件當天名次在高五分之一（≥80）、低五分之一（≤20）的次數，括號是同樣次數的隨機日子預期會有幾次；"
                             f"{pooltxt}。最後兩欄＝當天名次的中位數、進入事件前 30 天名次變化的中位數。",
                  "columns": ["事件", "指標", "有資料次數", "在高五分之一", "在低五分之一", "名次中位數", "30 天名次變化中位數"],
                  "rows": rows})
    # 5. path: trio
    for k in TRIO:
        rows = []
        for bi, b_ in enumerate(BINS):
            row = [f"{b_:+d} 到 {b_ + 9:+d} 天"]
            for ty in ("bear", "bull", "fall", "rise"):
                p = P1["path"][(ty, k)][bi]
                if not p:
                    row.append("—")
                    continue
                mk = "↑" if p["m"] > p["hi"] else ("↓" if p["m"] < p["lo"] else "")
                row.append(f"{p['m']:.0f}{mk}（{p['lo']:.0f}–{p['hi']:.0f}；{p['n']} 次）")
            rows.append(row)
        T.append({"caption": f"第一部分・{MEAS[k][0]}的平均名次路徑（轉折當天＝0，每 10 天一格）。括號＝隨機日子同樣次數平均的 10%–90% 範圍；"
                             "↑／↓＝高於／低於這個範圍。",
                  "columns": ["相對轉折的天數", "走熊", "走牛", "大跌", "大漲"], "rows": rows})
    # 6. path: others, coarse
    pick = (0, 3, 6, 8, 9, 11)
    for ty in ("bear", "bull", "fall", "rise"):
        rows = []
        for k in OTHERS:
            if k not in B["L"]:
                continue
            row = [MEAS[k][0]]
            for bi in pick:
                p = P1["path"][(ty, k)][bi]
                if not p:
                    row.append("—")
                    continue
                mk = "↑" if p["m"] > p["hi"] else ("↓" if p["m"] < p["lo"] else "")
                row.append(f"{p['m']:.0f}{mk}（{p['n']}）")
            rows.append(row)
        T.append({"caption": f"第一部分・{ETYPE[ty]}前後，其他指標的平均名次（括號＝有資料次數；↑／↓＝超出隨機日子 10%–90% 範圍）。"
                             "只挑六格顯示，完整十二格的規則同上。",
                  "columns": ["指標"] + [f"{BINS[bi]:+d}～{BINS[bi] + 9:+d}" for bi in pick], "rows": rows})
    # 7. contrast
    for a, b_, nm in (("bear", "bull", "走熊對走牛"), ("fall", "rise", "大跌對大漲"), ("real", "whip", "真下跌對假跌破"),
                      ("runner", "fizzle", "進場大漲對進場沒漲")):
        rows = []
        for k in keys:
            ct = P1["contrast"][(a, k)]
            if ct is None:
                rows.append([MEAS[k][0], "—", "—", "資料太少", "—"])
                continue
            out_ = not (ct["lo"] <= ct["diff"] <= ct["hi"])
            rows.append([MEAS[k][0], f"{ct['ma']:.0f}（{ct['na']} 次）", f"{ct['mb']:.0f}（{ct['nb']} 次）",
                         f"{ct['diff']:+.0f}［{ct['lo']:+.0f}，{ct['hi']:+.0f}］", ("超出洗牌範圍" if out_ else "在範圍內")
                         + f"（{_sh(ct['share'])}）"])
        T.append({"caption": f"第一部分・{nm}：事件當天名次的中位數和差距；方括號＝把兩類標籤洗牌 2,000 次的 5%–95% 範圍，"
                             "最後一欄括號＝洗牌差距至少這麼大的比例。",
                  "columns": ["指標", ESHORT[a], ESHORT[b_], "差距［洗牌範圍］", "分得出來嗎"], "rows": rows})
    # 8. order
    rows = []
    for k in keys:
        row = [MEAS[k][0]]
        for ty in ("bear", "bull"):
            offs = P1["order"][(ty, k)]
            vals = [o for o in offs if o is not None]
            row.append("、".join(f"{o:+d}" if o is not None else "—" for o in offs) or "—")
            row.append(f"{np.median(vals):+.0f}" if vals else "—")
        rows.append(row)
    T.append({"caption": "第一部分・順序（走熊／走牛）：轉折前 90 天到後 30 天裡，指標最「貪」（走熊前）或最「怕」（走牛前）的那一天，"
                         "離價格轉折幾天（負＝比價格早，正＝比價格晚；價格自己的極端就是 0）。偏斜反過來算（越低越貪）；"
                         "維基瀏覽量、隱含波動看最高／最低，不代表貪或怕。每次轉折依日期排列。",
              "columns": ["指標", "走熊：每次", "走熊中位數", "走牛：每次", "走牛中位數"], "rows": rows})
    rows = []
    for k in keys:
        row = [MEAS[k][0]]
        for ty in ("fall", "rise"):
            vals = [o for o in P1["order"][(ty, k)] if o is not None]
            if not vals:
                row += ["—", "—"]
                continue
            v = np.array(vals)
            row.append(f"早 {(v < -3).sum()}｜同時 {(np.abs(v) <= 3).sum()}｜晚 {(v > 3).sum()}")
            row.append(f"{np.median(v):+.0f}")
        rows.append(row)
    T.append({"caption": "第一部分・順序（大跌／大漲）：同上，只列「早（多於 3 天前）／同時（±3 天）／晚」的次數和中位數。",
              "columns": ["指標", "大跌：早｜同時｜晚", "大跌中位數", "大漲：早｜同時｜晚", "大漲中位數"], "rows": rows})
    # 9. ma50 individual events since 2018
    rows = []
    since = pd.Timestamp("2018-02-01")
    for ty in ("real", "runner"):
        for i in E[ty]:
            if idx[i] < since:
                continue
            extra = ""
            if ty == "runner" and i in EV["info"]["entries"]:
                extra = _pc(EV["info"]["entries"][i]["res"], 0)
            elif ty == "real":
                after = B["c"][i + 1:i + 61]
                extra = _pc(after.min() / B["c"][i] - 1, 0) if len(after) else "—"
            rows.append([ETYPE[ty], _d(idx[i]), extra, _cell(B, "fg", i) + "｜" + _band(B["L"]["fg"][i]),
                         _cell(B, "proxy", i), _cell(B, "resid", i)])
    T.append({"caption": "第一部分・2018-02 起 50 日均線「出場後真的下跌」和「進場後大漲」的每一次（第三欄：出場＝之後 60 天最低點，"
                         "進場＝這筆交易的結果）；每格＝數值（名次）。",
              "columns": ["事件", "日期", "之後", "恐懼貪婪", "價格版情緒 PROXY", "殘差"], "rows": rows})
    # 10. plain band counts
    rows = []
    for ty in ("bear", "fall", "real", "bull", "rise", "runner", "whip", "fizzle"):
        p = P1["plain"][ty]
        for nm, lab in (("greed", "極度貪婪（≥76）"), ("fear", "極度恐懼（≤24）")):
            x = p[nm]
            if not x["N"]:
                continue
            rows.append([ETYPE[ty], lab, str(x["N"]), f"{x['k']}（隨機 {x['p'] * x['N']:.1f}）" if x["p"] is not None
                         else str(x["k"]), f"{x['ka']}（隨機 {x['pa'] * x['N']:.1f}）" if x["pa"] is not None else str(x["ka"])])
    T.append({"caption": "第一部分・白話次數：事件當天、以及事件前 30 天內任何一天，恐懼貪婪指數落在它自己的「極度貪婪」或「極度恐懼」分級的次數；"
                         "括號＝同樣次數的隨機日子預期有幾次。",
              "columns": ["事件", "分級", "有資料次數", "當天在這一級", "前 30 天內出現過"], "rows": rows})
    # 11. Part 2 fifths
    half_k = (len(keys) + 1) // 2
    for part, ks in enumerate((keys[:half_k], keys[half_k:])):
        rows = []
        for k in ks:
            for g_, x in enumerate(P2["fifths"][k]):
                if not x:
                    continue
                rows.append([MEAS[k][0], f"第 {g_ + 1} 組（名次 {g_ * 20} 到未滿 {g_ * 20 + 20}）" if g_ < 4 else "第 5 組（名次 ≥80）", f"{x['n']:,}",
                             _pc(x["f30"]), f"{_pc(x['f60'])}［{_pc(x['lo'], 0)}，{_pc(x['hi'], 0)}］", _pc(x["f90"]),
                             _sh(x["up"]), _pc(x["deep"], 0)])
        T.append({"caption": f"第二部分・所有日子依名次分五組（第 {part + 1} 部分）：之後 30／60／90 天比特幣平均漲跌、60 天後比較高的比例、"
                             "60 天內最深跌幅的中位數。方括號＝60 天平均的 90% 區間（60 天一塊的區塊重抽 1,000 次）。相鄰日子高度重疊，"
                             "天數不是獨立次數。",
                  "columns": ["指標", "組", "天數", "30 天", "60 天［90% 區間］", "90 天", "60 天後比較高", "60 天內最深跌幅"],
                  "rows": rows})
    rows = []
    for g_, (nm, rng_) in enumerate((("極度恐懼", "0–24"), ("恐懼", "25–44"), ("中性", "45–55"), ("貪婪", "56–75"),
                                     ("極度貪婪", "76–100"))):
        x = P2["bands"][g_]
        if not x:
            continue
        rows.append([f"{nm}（{rng_}）", f"{x['n']:,}", _pc(x["f30"]), f"{_pc(x['f60'])}［{_pc(x['lo'], 0)}，{_pc(x['hi'], 0)}］",
                     _pc(x["f90"]), _sh(x["up"]), _pc(x["deep"], 0)])
    T.append({"caption": "第二部分・恐懼貪婪指數依它自己的五個分級：之後比特幣的表現（格式同上）。",
              "columns": ["分級", "天數", "30 天", "60 天［90% 區間］", "90 天", "60 天後比較高", "60 天內最深跌幅"],
              "rows": rows})
    # 12. Part 2 classification
    for s, sname in (("fear", "偏恐懼的極端"), ("greed", "偏貪婪的極端")):
        rows = []
        for name in list(keys) + ["proxy_full"]:
            res = P2["cls"].get((name, s))
            if not res:
                continue
            r = res["main"]
            k = "proxy" if name == "proxy_full" else name
            lab = MEAS[k][0] + ("（全期 2011 起）" if name == "proxy_full" else "（恐懼貪婪有資料的日子）" if name == "proxy" else "")
            if not r:
                rows.append([lab, "—", "—", "—", "—", "—", CLS["算不出"]])
                continue
            h0, h1 = r["halves"][0]["m3"], r["halves"][1]["m3"]
            rows.append([lab, f"{r['n_on']:,}／{r['n']:,}", f"{_pt(r['m3'])}［{_pt(r['ci3'][0], 0)}，{_pt(r['ci3'][1], 0)}］",
                         f"{_pt(h0, 0)}｜{_pt(h1, 0)}", _sh(r.get("placebo")),
                         f"{_pt(res['h30'], 0)}｜{_pt(res['h90'], 0)}", CLS[res["cls"]]])
        T.append({"caption": f"第二部分・{sname}之後 60 天，比「其餘日子」多漲幾個百分點（配對：過去 30 天漲跌十組×過去 180 天漲跌三組×"
                             "20 天波動三組，同一格裡比）。恐懼貪婪用它自己的分級（≤24／≥76）；PROXY 用和恐懼貪婪分級同樣天數比例的門檻；"
                             "其他用名次 ≤20／≥80（偏斜反過來：高偏斜＝偏恐懼）。正＝之後比較好。",
                  "columns": ["指標", "極端天數／全部", "配對後差［90% 區間］", "前半｜後半", "錯位對照", "30 天｜90 天",
                              "判讀（事先定的）"], "rows": rows})
        rows = []
        for name in list(keys) + ["proxy_full"]:
            res = P2["cls"].get((name, s))
            if not res or not res["main"]:
                continue
            k = "proxy" if name == "proxy_full" else name
            lab = MEAS[k][0] + ("（全期）" if name == "proxy_full" else "")
            per = "；".join(f"{p['lab']} {_pt(p['m3'], 0)}（{p['n_on']} 天）" for p in res["main"]["per"])
            on, off = res["on"], res["off"]
            rows.append([lab, per or "—", (f"{_pt(on['m3'], 0)}（{on['n_on']} 天）" if on else "—"),
                         (f"{_pt(off['m3'], 0)}（{off['n_on']} 天）" if off else "—")])
        T.append({"caption": f"第二部分・{sname}：分期間，以及只看 50 日均線開著／關著的日子（同樣的配對差，60 天）。",
                  "columns": ["指標", "各期間", "均線開著", "均線關著"], "rows": rows})
    # 13. Part 3 discovery
    rows = []
    for k in ("fg", "resid", "proxy"):
        for s in ("fear", "greed"):
            v, n_on = P3["disc"][(k, s)]
            rows.append([MEAS[k][0], "極度恐懼／最低一組" if s == "fear" else "極度貪婪／最高一組", f"{n_on:,}", _pt(v),
                         "之後比較好" if _ok(v) and v > 0 else ("之後比較差" if _ok(v) else "算不出")])
    qf, qg, sf_, sg_ = B["pq"]
    T.append({"caption": f"第三部分・方向只用 2022 年以前的日子決定（60 天、配對後差）。PROXY 的門檻：≤{qf:.0f} 分算「怕」、≥{qg:.0f} 分算「貪」"
                         f"（2022 年以前恐懼貪婪 ≤24 的日子占 {sf_ * 100:.0f}%、≥76 占 {sg_ * 100:.0f}%，PROXY 取同樣比例）。",
              "columns": ["指標", "極端", "2022 前的極端天數", "配對後差", "量到的方向"], "rows": rows})
    # 14. exam
    base = P3["base"]
    rows = [["50 日均線（基準）", _exam_cells(base), f"{base['c_deep_med']:.1f}%", "—", "—", "—", "—", "—"]]
    for r in rules_all:
        x = r["x"]
        rows.append([_rname(r), _exam_cells(x), f"{x['c_deep_med']:.1f}%", _sh(r["plc"]) if r["plc"] is not None else "沒有改變均線",
                     f"{_exam_cells(r['twin'])}｜{r['twin']['c_deep_med']:.1f}%",
                     (f"c={r['c']:.2f}：{_exam_cells(r['const'])}" if r["const"] else "不需要"),
                     f"{_yn(r['score'])}{_yn(r['loss'])}{_yn(r['plc_ok'])}{_yn(r['twin_ok'])}{_yn(r['const_ok'])}",
                     ("通過" if r["pass"] else "沒通過") + ("，Holm 後仍在" if r.get("holm") else "")])
    ex = P3["ex"]
    T.append({"caption": f"第三部分・公平考試（只用 2022-01 起的起算日：比特幣日線 {base['d_n']} 個每月起算日、1 小時線 {base['c_starts']} 個起算日×24 個收盤時間、"
                         f"一般人照做、其他幣 {base['o_n']} 個；到 {_d(P3['last'])}）。第二欄＝四項「贏過一直抱著」比例裡最低的（排名分數），括號是四項；"
                         "第三欄＝收盤時間格的最深虧損中位數。PROXY 雙胞胎＝同樣的規則改用只看價格的 PROXY；固定持有量雙胞胎＝均線開著時固定拿 c 成。"
                         "條件五個依序：排名不輸均線、虧損條件、錯位對照 ≤5%、贏 PROXY 雙胞胎、贏固定持有量雙胞胎（持有比均線多的規則不需要最後一項）。",
              "columns": ["規則", "排名分數（四項）", "最深虧損中位數", "錯位對照", "PROXY 雙胞胎（排名｜虧損）", "固定持有量雙胞胎",
                          "五個條件", "結果"], "rows": rows})
    rows = []
    for r in rules_all:
        rows.append([_rname(r), KIND_CN[r["kind"]], f"{r['differ']:,} 天", _sh(r["time"]), _pt(r["y2022"])])
    T.append({"caption": "第三部分・規則白話：每條規則是什麼、2022 年起和 50 日均線持有不同的天數、平均持有比例、2022 整年比 50 日均線多賺幾點"
                         "（比特幣日線，不算成本）。",
              "columns": ["規則", "做法", "和均線不同的天數", "平均持有", "2022 年比均線"], "rows": rows})
    # 15. spells
    rows = []
    for r in rules_all:
        for key in ("ent", "ex"):
            for s_ in r["sp"][key][:25]:
                rows.append([_rname(r), "多拿（進場）" if key == "ent" else "少拿（出場）", _d(s_["s"]), _d(s_["e"]),
                             f"{s_['days']}", _pc(s_["ret"]), (s_["how"] or ("還在進行" if s_["open"] else "—"))])
    T.append({"caption": "第三部分・每一段和 50 日均線不同的期間（2022 起，比特幣日線）：從第一天收盤到恢復成和均線一樣的那天收盤，比特幣漲跌多少"
                         "（多拿的段落＝規則賺或賠的；少拿的段落＝規則躲掉的跌或錯過的漲）。每條規則每種最多列 25 段。",
              "columns": ["規則", "種類", "開始", "結束", "天數", "這段比特幣漲跌", "怎麼結束"], "rows": rows})
    # 16. Part 4
    for side in ("fear", "greed"):
        rows = []
        for row in P4["rows"].get(side, []):
            if row["cond"] is None:
                nm = "只看極端本身（基準）"
            else:
                c_, a_, b2 = COND[row["cond"]]
                nm = f"{c_}：{a_ if row['sv'] == 1.0 else b2}" + ("（不會改變均線，不列入）" if row.get("frozen") else "")
            same = (_ok(row["d"]) and _ok(row["t"]) and row["d"] * row["t"] > 0)
            rows.append([nm, f"{_pt(row['d'])}（{row['nd']} 天）", f"{_pt(row['t'])}（{row['nt']} 天）",
                         ("同向" if same else "不同向") if _ok(row["d"]) and _ok(row["t"]) else "—",
                         f"{row['sharp'] * 100:+.1f}" if _ok(row.get("sharp")) else "—"])
        b_ = P4["best"].get(side)
        T.append({"caption": f"第四部分（探索）・{'極度恐懼' if side == 'fear' else '極度貪婪'}再加一個條件：「極端＋條件」那些日子之後 60 天比"
                             "「沒有極端的日子」多漲幾點（配對後），2022 前（探索）和 2022 起（檢查）各算一次；最後一欄＝在探索期間的方向上比"
                             "「只看極端」多分開幾點。" + (f"探索期間最好的是：{COND[b_['cond']][0]}{COND[b_['cond']][1] if b_['sv'] == 1.0 else COND[b_['cond']][2]}。"
                                                    if b_ else "沒有合格的組合（探索期間至少 30 天）。"),
                  "columns": ["加上的條件", "2022 前（探索）", "2022 起（檢查）", "方向", "比只看極端多分開"], "rows": rows})
    return T


# ---------- verdict / now ----------

def _plain_sentence(P1: dict, ty: str, nm: str) -> str:
    x = P1["plain"][ty][nm]
    lab = "極度貪婪" if nm == "greed" else "極度恐懼"
    if not x["N"]:
        return f"{ESHORT[ty]}：沒有恐懼貪婪資料"
    exp = f"{x['p'] * x['N']:.1f}" if x["p"] is not None else "—"
    exa = f"{x['pa'] * x['N']:.1f}" if x["pa"] is not None else "—"
    return (f"在 {x['N']} 次{ESHORT[ty]}前（當天），恐懼貪婪指數有 {x['k']} 次在『{lab}』（隨機日子預期 {exp} 次）；"
            f"前 30 天內出現過的有 {x['ka']} 次（隨機 {exa} 次）")


def _verdict(B, P1, P2, P3, P4, rules_all, sentences_px) -> str:
    fit = B["fit"]
    parts = []
    parts.append(f"量到的（恐懼貪婪有多少是價格）：恐懼貪婪和只用價格做的 PROXY 相關 {fit['corr_lev']:+.2f}，30 天變化相關 {fit['corr_chg']:+.2f}；"
                 f"四個價格名次解釋了 {fit['r2'] * 100:.0f}% 的變動（只用以前資料逐日配的樣本外 {fit['r2_oos'] * 100:.0f}%）。")
    s = [_plain_sentence(P1, "bear", "greed"), _plain_sentence(P1, "bull", "fear"),
         _plain_sentence(P1, "fall", "greed"), _plain_sentence(P1, "rise", "fear"),
         _plain_sentence(P1, "real", "greed"), _plain_sentence(P1, "runner", "fear")]
    parts.append("量到的（第一部分，轉折前）：" + "；".join(s) + "。" + sentences_px)
    # order summary for fg / proxy
    om = []
    for k in ("fg", "proxy", "resid"):
        for ty in ("bear", "bull"):
            v = [o for o in P1["order"][(ty, k)] if o is not None]
            if v:
                om.append(f"{MEAS[k][0]}在{ESHORT[ty]}：{len(v)} 次裡 {sum(1 for o in v if o < -3)} 次比價格早、"
                          f"{sum(1 for o in v if abs(o) <= 3)} 次同時、{sum(1 for o in v if o > 3)} 次晚（中位數 {np.median(v):+.0f} 天）")
    parts.append("量到的（順序）：" + "；".join(om) + "。")
    cl = []
    for k in ("fg", "resid", "proxy", "proxy_full"):
        for sd_ in ("fear", "greed"):
            res = P2["cls"].get((k, sd_))
            if not res or not res["main"]:
                continue
            r = res["main"]
            kk = "proxy" if k == "proxy_full" else k
            cl.append(f"{MEAS[kk][0]}{'（全期）' if k == 'proxy_full' else ''}{'怕' if sd_ == 'fear' else '貪'}的極端 {_pt(r['m3'])}"
                      f"（{res['cls']}）")
    parts.append("量到的（第二部分，所有日子，60 天配對後差，正＝之後比較好）：" + "；".join(cl) + "。")
    others_hold = [f"{MEAS[k][0]}{'怕' if sd_ == 'fear' else '貪'}側 {_pt(P2['cls'][(k, sd_)]['main']['m3'])}"
                   for k in OTHERS for sd_ in ("fear", "greed")
                   if (k, sd_) in P2["cls"] and P2["cls"][(k, sd_)]["cls"] == "站得住"]
    parts.append("其他情緒指標判讀為「站得住」的：" + ("、".join(others_hold) if others_hold else "沒有") + "。")
    n_rules = len([r for r in rules_all if r["plc"] is not None])
    passed = [_rname(r) for r in rules_all if r["pass"]]
    holm = [_rname(r) for r in rules_all if r.get("holm")]
    parts.append(f"量到的（第三、四部分，只在 2022 起的起算日考）：考了 {len(rules_all)} 條規則（{n_rules} 條有改變均線、做了錯位對照），"
                 f"事先定的條件全部通過的 {len(passed)} 條（{'、'.join(passed) if passed else '無'}）；Holm 校正後還在的 {len(holm)} 條；"
                 f"光靠運氣，錯位這一關大約會有 {n_rules * 0.05:.1f} 條過。")
    parts.append("分不出來的：恐懼貪婪本身有一部分就是價格和波動，轉折前的「貪」有多少是「價格已經漲很多」，要看 PROXY 和殘差那兩列；"
                 "次數很少（走熊、走牛各只有約 4 次有恐懼貪婪資料），沒看到＝用這些定義和資料沒看到。")
    return "\n".join(parts)


def study() -> dict:
    t0 = time.time()
    d = _load()
    B = _build(d)
    X = _extremes(B)
    close = d["close"]
    EV = _events(B, close)
    F = rd._feats(close)
    P1 = _part1(B, EV)
    P2 = _part2(B, X, F)
    P3 = _part3(B, X, F, d, EV)
    P4 = _part4(B, X, F, P3)
    rules_all = P3["rules"] + P4["rules"]
    first_def = {"fg": B["idx"][int(np.argmax(np.isfinite(B["L"]["fg"])))],
                 "resid": B["idx"][int(np.argmax(np.isfinite(B["L"]["resid"])))]}
    for r in rules_all:
        r["s0"] = first_def["resid"] if r["src"] == "resid" else first_def["fg"]
    _score_rules(B, P3, P3["rules"], 400)
    _score_rules(B, P3, P4["rules"], 450)
    pv = {r["id"]: r["p"] for r in rules_all if r["p"] is not None}
    adj = hg._holm(pv) if pv else {}
    for r in rules_all:
        r["holm"] = bool(r["pass"] and r["id"] in adj and adj[r["id"]] <= ALPHA)
    # PROXY / RESID plain counts with the fixed extremes
    pools = P1["pools"]
    E = EV["E"]
    px = []
    for ty, side in (("bear", "greed"), ("bull", "fear"), ("fall", "greed"), ("rise", "fear")):
        for k in ("proxy", "resid"):
            x = _side_count(B, X, ty, E, pools, k, side)
            if x["N"]:
                px.append(f"{MEAS[k][0]}在 {x['N']} 次{ESHORT[ty]}當天有 {x['k']} 次在{'貪' if side == 'greed' else '怕'}的極端"
                          f"（隨機 {x['p'] * x['N']:.1f}）" if x["p"] is not None else "")
    sentences_px = "對照：" + "；".join(p for p in px if p) + "。"
    tables = _tables(B, X, EV, P1, P2, P3, P4, rules_all)
    # plain answers
    ans = []
    for r in rules_all:
        se = _spell_sum(r["sp"]["ent"], True)
        sx = _spell_sum(r["sp"]["ex"], False)
        ans.append([_rname(r),
                    (f"{se['n']} 次；{se['up']} 次結束時比較高；賺的平均 {_pc(se['avg_up'])}、賠的平均 {_pc(se['avg_down'])}"
                     f"（停損 {se['stops']} 次）；合計 {_pc(se['net'])}" if se["n"] else "沒有多拿的段落"),
                    (f"{sx['n']} 段；那段期間比特幣跌 {sx['down']} 次、漲 {sx['up']} 次；躲掉的跌平均 {_pc(sx['avg_down'])}、"
                     f"錯過的漲平均 {_pc(sx['avg_up'])}；合計比均線 {_pc(sx['net'])}" if sx["n"] else "沒有少拿的段落")])
    # 2025 top
    top25 = [t for t in EV["info"]["turns"]["bear"] if t["date"].year == 2025]
    T25 = top25[0]["i"] if top25 else None
    rows25 = []
    if T25 is not None:
        for r in rules_all:
            lo, hi = max(0, T25 - 90), min(B["n"] - 1, T25 + 60)
            less = np.flatnonzero(r["pos"][lo:hi + 1] < B["ma"][lo:hi + 1] - 1e-9)
            first = (f"{_d(B['idx'][lo + less[0]])}（{lo + less[0] - T25:+d} 天）" if len(less) else "沒有")
            rows25.append([_rname(r), first, _pt(_window_diff(B, r["pos"], T25, -60, 120))])
        tables.append({"caption": f"第三部分・2025 年的頂部（走熊開始 {_d(B['idx'][T25])}，收盤 {B['c'][T25]:,.0f} 美元）：規則第一次比 50 日均線少拿的日子"
                                  "（頂部前 90 天到後 60 天內），以及頂部前 60 天到後 120 天比均線多賺幾點（比特幣日線、不算成本）。",
                       "columns": ["規則", "第一次少拿", "頂部 -60～+120 天比均線"], "rows": rows25})
    tables.append({"caption": "第三部分・白話回答（2022 起，比特幣日線，收盤決定、不算成本）：「照著進場，賺不賺？」看多拿的段落；"
                              "「照著出場，少不少賠？」看少拿的段落（「跌」＝那段期間比特幣跌了，也就是出場躲掉了）。",
                   "columns": ["規則", "照著進場，賺不賺？", "照著出場，少不少賠？"], "rows": ans})
    # today
    idx = B["idx"]
    i = len(idx) - 1
    L, R = B["L"], B["R"]
    fg_i = i if _ok(L["fg"][i]) else int(np.flatnonzero(np.isfinite(L["fg"]))[-1])
    ma_state = "持有" if B["ma"][i] > 0.5 else "現金"
    best = [r for r in rules_all if r["pass"]]
    if best:
        best.sort(key=lambda r: (-r["x"]["low"], -r["x"]["c_deep_med"]))
        b = best[0]
        p = b["pos"][i]
        rule_state = f"{_rname(b)}：{'持有' if p > 0.99 else ('半倉' if p > 0.01 else '現金')}"
    else:
        rule_state = "沒有通過事先條件的情緒規則（下面各規則的狀態只是描述）"
    rs = "；".join(f"{_rname(r)}：{'持有' if r['pos'][i] > 0.99 else ('半倉' if r['pos'][i] > 0.01 else '現金')}"
                   for r in rules_all)
    res_i = L["resid"][fg_i]
    now = [{"label": "比特幣：50 日均線｜情緒規則", "value": f"50 日均線：{ma_state}｜{rule_state}"},
           {"label": "各情緒規則今天的狀態（描述，不是建議）", "value": rs or "—"},
           {"label": f"恐懼貪婪指數（{_d(idx[fg_i])}）", "value": f"{L['fg'][fg_i]:.0f} 分（{_band(L['fg'][fg_i])}；過去一年名次 {_r0(R['fg'][fg_i])}）"},
           {"label": "價格版情緒 PROXY", "value": f"{L['proxy'][i]:.0f} 分（只看價格：{'很怕' if L['proxy'][i] <= B['pq'][0] else '很貪' if L['proxy'][i] >= B['pq'][1] else '不在極端'}）"},
           {"label": "殘差（今天的情緒比價格本身顯示的更怕還是更貪）",
            "value": (f"{res_i:+.1f} 分：比價格本身顯示的{'更怕' if res_i < 0 else '更貪'}（過去一年名次 {_r0(R['resid'][fg_i])}）"
                      if _ok(res_i) else "—")}]
    for k in OTHERS:
        if k not in L:
            continue
        f = np.flatnonzero(np.isfinite(L[k]))
        if not len(f):
            continue
        j = f[-1]
        now.append({"label": f"{MEAS[k][0]}（{_d(idx[j])}）", "value": f"{_lv(k, L[k][j])}，過去一年名次 {_r0(R[k][j])}（描述，不是預測）"})
    verdict = _verdict(B, P1, P2, P3, P4, rules_all, sentences_px)
    # added after the first run (presentation only): plain answers and the direction note in the verdict
    g_dir = P3["disc"][("fg", "greed")][0]
    f_dir = P3["disc"][("fg", "fear")][0]
    lines = []
    for r in rules_all:
        se, sx = _spell_sum(r["sp"]["ent"], True), _spell_sum(r["sp"]["ex"], False)
        if se["n"]:
            lines.append(f"{_rname(r)}：進場 {se['n']} 次、{se['up']} 次結束時比較高、合計 {_pc(se['net'])}")
        if sx["n"]:
            lines.append(f"{_rname(r)}：出場 {sx['n']} 段、躲掉的跌 {sx['down']} 次、合計比均線 {_pc(sx['net'])}")
    has_exit = any(r["kind"] in HOLD_LESS for r in rules_all)
    gtxt = ("極度貪婪之後比較好，所以照事先寫好的規定沒有「貪婪時出場」的規則可考，改考量到的方向（貪婪時也持有）；但極度貪婪幾乎只出現在"
            "50 日均線已經開著的時候，這條規則在 2022 起只和均線差 "
            + f"{next((r['differ'] for r in rules_all if r['id'] == 'fg_greed_b'), 0)} 天，等於沒有改變均線。"
            if _ok(g_dir) and g_dir > 0 else "極度貪婪之後比較差，所以考的是「貪婪時出場」。")
    ftxt = ("極度恐懼之後比較好，所以考的是「恐懼後站上 20 日均線就提早進場」。" if _ok(f_dir) and f_dir > 0
            else "極度恐懼之後比較差，所以考的是量到的方向「恐懼時不持有」。")
    verdict += ("\n量到的（方向，只用 2022 前）：恐懼貪婪極度貪婪之後 60 天 " + _pt(g_dir) + "、極度恐懼之後 " + _pt(f_dir)
                + "（配對後，正＝比較好）。" + gtxt + ftxt
                + "\n白話回答（2022 起，比特幣日線、不算成本）：照著進場，賺不賺？" + ("；".join(lines) if lines else "沒有進場段落")
                + "。照著出場，少不少賠？" + ("見上面出場的段落。" if has_exit else
                                          "量到的方向沒有產生出場規則，所以這次沒有出場的答案；2025-10 的頂部前後，這些規則都沒有比均線少拿。"))
    src = [{"name": "alternative.me 加密貨幣恐懼與貪婪指數 API（全歷史）", "url": FNG_URL},
           {"name": "alternative.me 恐懼與貪婪指數說明頁（成分與權重，2026-11-19 打開過）", "url": FNG_PAGE}]
    src += [{"name": k, "url": v} for k, v in d["src"].items()]
    src += [{"name": "Open Bitcoin Metrics（放 155 天以上的幣被花掉，第四部分條件）", "url": pbm.ocs.OBM_BASE.format(pbm.OBM_ID)},
            {"name": "幣安 BTCUSDT 1 小時線（公平考試的 24 個收盤時間）",
             "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"},
            {"name": "幣安 ETHUSDT／SOLUSDT／BNBUSDT／XRPUSDT 日線（其他幣）", "url": "https://api.binance.com/api/v3/klines?interval=1d"},
            {"name": "Yahoo MSTR 日線（其他資產）", "url": "https://query1.finance.yahoo.com/v8/finance/chart/MSTR"}]
    notes = [
        PRIOR,
        "這是觀察＋規則研究。第一部分只描述轉折前後的樣子；轉折日期只有事後才知道。第三部分的方向和門檻只用 2022 年以前的日子決定，"
        "規則只在 2022-01 起的起算日考；第四部分是探索，看過 32 個組合才挑一個，結果要打折看。",
        "恐懼貪婪指數（文獻說的，說明頁 2026-11-19 有打開）：alternative.me 寫的成分是波動 25%、漲跌動能與成交量 25%、社群（X 上的比特幣標籤）15%、"
        "問卷 15%（頁面寫「目前暫停」）、比特幣市占 10%、Google 搜尋趨勢 10%；問卷暫停後實際權重怎麼分，頁面沒有寫。配方由網站自訂、改過，"
        "過去的分數用的可能不是今天的配方。",
        "時間對齊：恐懼貪婪標記為 D 日的數字大約在 D 日 00:00 UTC 公布（用 D-1 的資料），這裡從 D 日那根日線收盤（D+1 00:00 UTC）才用，"
        "大約多留一天。其他指標沿用 signs_before_turns 的延遲（維基晚 1 天，衍生品、溢價當天）。",
        "殘差是用「以前的資料」逐日配的回歸算的，不偷看未來；它要先累積 365 天重疊的資料，所以 2019 年起才有。",
        f"限制：恐懼貪婪 2018-02 才有（大約兩個循環、走熊走牛各約 4 次）；維基的使用方式隨時間改變（搜尋引擎、AI 摘要）；"
        "資金費率、多空比只有幣安，基差、偏斜、隱含波動只有 Deribit，而且 2019–2021 年才開始；2022 年起的考試期只有約一個半循環。",
        "很多天是同一段行情裡的連續日子，「天數」不是獨立次數；區間用 60 天一塊的區塊重抽，錯位對照把整條標籤平移 180 天以上。",
        f"這次總共算了 {(time.time() - t0) / 60:.1f} 分鐘（載入資料 {d['load_s'] / 60:.1f} 分鐘）。所有下載存進 dashboard/cache/_store/sentiment_turns/，"
        "共用的存檔只讀。",
    ] + d["problems"]
    return {"id": ID, "title": "情緒在轉折前長什麼樣子？照著它進出有沒有用？（恐懼貪婪、關注度、散戶部位、選擇權、溢價）",
            "question": "走熊、走牛、大跌、大漲之前，恐懼貪婪指數和其他情緒指標長什麼樣子？它們比價格早、同時還是晚？"
                        "照著「別人恐懼時貪婪」或量到的方向進出，進場賺不賺、出場少不少賠？加一個條件會不會比較好？",
            "asof": _d(idx[-1]), "verdict": verdict, "now": now, "tables": tables, "notes": notes, "sources": src}
