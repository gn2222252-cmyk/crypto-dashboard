"""Historical study "at_the_signal": put every variable with a measured effect onto the main signal's own events.

The main signal ma50 (hold bitcoin while the daily close is above its 50-day average) produces two kinds of events
known in real time: EXITS (ma50 turns off; whipsaw_anatomy / exit_forces: most are false breakdowns, back in within
20 days at a higher price, a minority are real declines) and ENTRIES (ma50 turns on; some become big rises, most
fizzle). Several non-price variables were MEASURED in earlier studies to carry information about forward returns, but
each was tested only as a stand-alone rule over all days and dropped when that rule did not beat ma50. Only
exit_forces looked at such variables AT the exit (nine flow measures); whipsaw_anatomy showed ten price descriptors
cannot separate false from real exits; nobody looked at entries. Four studies (crash_timing, rally_timing,
exit_entry_combo, follow_turn_signs) showed that leaving or entering earlier than ma50 is paid back at the other end,
which argues for using information AT ma50's own events, not instead of them. Observation first, then rules.

Owner's ground rules (applied to every user-facing sentence): observe first; do not conclude lightly; label
statements 量到的 (measured, with the number, k of N) or 推測 (a guess, marked); where the data cannot separate two
explanations say so; never use one tendency as a law to dismiss another; "not seen" = not seen with this definition
and this data, and say what may be missing; a result opposite to the expected direction is a finding and is reported
in its measured direction.

Written BEFORE any run of this file. Anything added after seeing results is labelled "added after the first run".

Data (free, key-less, fixed user agent "btc-research-dashboard"; .env never read; loaders REUSED, not copied; every
download of this study goes to cache/_store/at_the_signal/, seeded read-only from the shared store and from the
stores of signs_before_turns and follow_turn_signs, exactly as follow_turn_signs does):
  BTC daily closes te._daily_close (00:00 UTC; blockchain.info before Binance 2017-08).
  signs_before_turns._assemble (43 series, each already lagged to publication; used here: MVRV, distance from the
    200-day average, distance from the 365-day high, active addresses, funding, basis, retail long/short ratio, top
    trader ratio, Korean premium, Wikipedia views, NFCI, HYG/IEF, VIX, dollar index).
  pressure_before_moves._load / _pressures (stores redirected into this study's folder as follow_turn_signs does):
    ACT (exchange deposits + withdrawals, 30 days, lag 1), S_old (coins held 155+ days moving, 30 days, lag 7),
    NET7 / NET30 ((withdrawn - deposited) / (withdrawn + deposited), lag 1), B_stb (net new stablecoins, 30 days,
    lag 1), B_etf (US spot ETF net flow, 30 days, lag 1), B_cb (Coinbase / Bitstamp - 1, USD versus USD, 7-day mean),
    OI (Binance open interest 30-day change), S_take + B_take (Binance spot quote volume, 7 days; zero-fee period
    missing), its Coinbase daily closes and its Binance 1-hour bars (daily quote volume for descriptor (i)).
  Coin Metrics SplyExNtv (coins on exchange-tagged addresses) via exit_forces._cm_fetch, lag 1.
  blockspace._onchain fee revenue in USD -> blockspace._zscore (30-day mean versus its 730-day mean / std), lag 1.
  follow_turn_signs._sets on signs_before_turns' ranks -> BEAR_COUNT / BULL_COUNT, FORWARD version (signs chosen only
    from turns already confirmed at the time; undefined before the third confirmation).
  Fair exam: Binance 1-hour BTCUSDT grid (clock_robustness), ETH SOL BNB XRP (Binance) and MSTR (Yahoo) through
    main_signal_check._load_other.

Variables (value known at the close of day t), grouped:
  ACTIVITY: act (total exchange traffic), old (old coins moving), feez (fee z-score), adr (active addresses), vol
    (Binance spot volume).
  FLOW BALANCE: net7, net30 (NET), bal30 (exchange balance change over 30 days: SplyExNtv(t-1)/SplyExNtv(t-31) - 1),
    stb (stablecoin growth), etf (ETF flow).
  PREMIUMS: cbusd (Coinbase premium USD-vs-USD), cbdisc (Coinbase extreme discount, BINARY: on any of the last 7 days
    the raw daily premium Coinbase / Binance close - 1 was below the 5th percentile of its previous 365 days, >= 330
    values - coinbase_premium's "bottom"), kimchi (Korean premium).
  LEVERAGE AND POSITIONING: fund (funding 7-day mean), oi30 (open interest 30-day change), basis, lsr (retail long /
    short ratio), lsd (log retail ratio - log top-trader ratio).
  ATTENTION: wiki (Wikipedia views, 7-day mean).
  CONDITIONS: nfci (NFCI known at t minus the mean of its known values over the previous 364 days, >= 300), hyg (HYG/
    IEF), vix, dxy.
  VALUATION AND STRETCH: mvrv, ma200 (distance from the 200-day average), hi365 (distance from the 365-day high), bear
    (BEAR_COUNT), bull (BULL_COUNT).
  VOLATILITY: rv20 (std of the last 20 daily log returns x sqrt(365)).
Two forms per variable (binary cbdisc: the flag only):
  RANK = mid-rank 0-100 of today's value among the previous 365 days' values (t-365 .. t-1, at least 180 present):
    100 x (number below + 0.5 x number equal) / number present. Split: HIGH = rank > 50 (above its own past-year
    median, known in real time), LOW otherwise.
  CHANGE = RANK(t) - RANK(t-30). Split: RISING = change > 0, otherwise FALLING.

Events (BTC daily):
  EXITS = whipsaw_anatomy._exits of ma50 (main_signal_check's). Labels WHIPSAW / REAL / OTHER = whipsaw_anatomy._label.
    Outcome PAYOFF = exit_forces._payoff: stay in from close e to the first of ma50 on again (+2 x 0.10%), the hard
    stop (close <= 0.92 x exit close or below the lowest close of e-20..e-1), or close e+60; minus cash (0). Exits whose
    span is still open are left out of every statistic.
  ENTRIES = closes s where ma50 turns on (off at s-1, on at s). Trade = s to the next ma50 exit x. RESULT = c(x)/c(s)
    - 1 - 2 x 0.10%; PEAK = highest close in s..x / c(s) - 1; LENGTH = x - s days. Labels: RUNNER = RESULT >= +20%,
    FIZZLE = RESULT <= 0, SMALL = in between. A trade still open is left out.
    Primary entry statistic = log(1 + RESULT) (so a few ten-fold trades do not decide the mean); mean / median RESULT
    shown beside it.
  Clusters: events sorted by date, a new cluster when the gap to the previous event of the same type > 30 days.

Part 1 - EXITS. For each variable and form: exits with the variable defined and PAYOFF known, split HIGH / LOW (RISING /
  FALLING; binary: on / off). Per half: N, WHIPSAW share, REAL share, mean and median PAYOFF. DIFF = mean PAYOFF(HIGH) -
  mean PAYOFF(LOW), 90% cluster-bootstrap interval (2,000 draws, fixed seed). CONTROLLED = OLS of PAYOFF on [1, HIGH,
  whipsaw_anatomy's ten price descriptors a..j, FALL10 = close(e)/close(e-10) - 1] (missing descriptor values filled
  with the sample mean plus a 0/1 missing indicator, constant columns dropped; exit_forces._design's recipe); the HIGH
  coefficient with a 90% cluster-bootstrap interval (1,000 draws) and its bootstrap standard error. Computed only with
  at least 20 exits and at least 3 in each half. Tables sorted by |CONTROLLED|. Placebo: the variable's daily HIGH/LOW
  series circularly shifted by k (uniform integer in [180, span - 180], span = its first to last defined day) against
  the fixed event dates, 200 draws (fixed seed), CONTROLLED recomputed; placebo share = share of draws with
  |CONTROLLED| >= the real |CONTROLLED| (two-sided).
  REAL-decline exits listed one by one: date, the fall that followed (lowest close in e+1..e+60 / close(e) - 1),
  PAYOFF, and each group's composite reading = mean RANK of the group's continuous variables defined at the exit
  (BEAR_COUNT / BULL_COUNT shown as raw counts beside valuation). Next to them: the median composite over WHIPSAW exits
  and over REAL exits (description).
Part 2 - ENTRIES. The same per variable and form with RUNNER / FIZZLE shares, mean and median RESULT, DIFF in mean
  log(1 + RESULT), CONTROLLED with the entry price descriptors: close / 50-day average - 1, 30-day and 90-day past
  return, close / 365-day high - 1, 20-day realised volatility, close above the 200-day average (0/1). The 15 largest
  trades listed with each group's composite at entry, with the median composite over RUNNER and FIZZLE entries.
Part 3 - multiple testing. Cells = variable x form x event type with a CONTROLLED estimate. A cell "passes" when its
  CONTROLLED 90% interval excludes zero AND placebo share <= 5%. Luck count: for each cell, each placebo draw d is
  treated as if it were real: it "passes" when |coef_d| > 1.645 x the real bootstrap standard error (normal stand-in
  for the interval) and its own placebo share among the other 199 draws <= 5%; luck count = sum over cells of the
  share of draws that pass (with 5% x cells as the simple bound). Benjamini-Hochberg at 10% over all cells using p =
  2 x (1 - Phi(|CONTROLLED| / bootstrap SE)) (a normal approximation; the placebo p cannot reach BH's first threshold
  with 200 draws: its smallest value 1/201 is above 0.10 / cells once there are more than 20 cells - stated, not
  worked around). Survivors listed.
  Honest split: DISCOVERY = events before 2019-01-01, TEST = from 2019-01-01. On DISCOVERY only the CONTROLLED
  coefficient is computed per cell (cells need >= 8 discovery events in each half and >= 20 in total). The code ranks
  cells by |CONTROLLED| on discovery and takes the top five for exits and the top five for entries. For each: the
  discovery coefficient and interval, the TEST coefficient and interval (>= 5 test events in each half, >= 15 total),
  whether the sign is the same and whether the test interval excludes zero.
Part 4 - following it (only what DISCOVERY picked; the #1 exit cell and the #1 entry cell).
  XR = hold when ma50 is on OR (the last ma50 exit e was within the last 60 closes - decisions e..e+59 - AND the top
    discovery exit cell was on its "stay" side at e (the side with the higher discovery coefficient; undefined =
    leave) AND no close after e hit the hard stop): exit_forces._Exam / _hold, unchanged. Twin X0 = always stay with
    the stop (exit_forces' V0).
  ER = plain ma50, except that an entry is taken at HALF size (0.5) when the top discovery entry cell was on its
    "fizzle" side at the entry close (the side with the lower discovery coefficient; undefined = full size), going to
    full size from the first close at least 10% above the entry close, until ma50 turns off. Twin E0 = every entry at
    half size, full at +10%.
  Other series in the exam (24 closing hours, ETH SOL BNB XRP MSTR) use their own ma50, exits, entries and stop with
  BTC's daily flag as a market-wide state (a clock close on UTC date E and MSTR's date D use the flag of E-1 / D-1),
  exit_forces' wiring.
  Fair exam = exit_forces._Exam (main_signal_check's four checks) with first month 2019-01: BTC daily monthly starts
  2019-01 .. 2023-12 (0.10%); clock starts 2019-01 .. 2023-12 x 24 closing hours; ordinary person (10 hours later,
  0.25%, 10% of entries missed, 200 simulations per clock); other assets from max(first valid month, 2019-01) to three
  years before today. ma50 re-scored on the same starts. Ranking score = worst of the four "beats holding" shares
  with main_signal_check's eligibility (supply_demand._score_ge); median deepest loss = median over the clock textbook
  cells. Placebo: the rule's daily flag circularly shifted (exit_forces._roll, k in [180, span - 180]) 200 draws, exam
  rebuilt with 20 ordinary-person simulations per clock, the real rule re-scored the same way; placebo share = share
  of draws scoring at least the real one; p = (1 + hits) / (1 + 200).
  "Supported": ranking >= ma50's AND median deepest loss no more than 5 points worse than ma50's (XR) / no worse (ER)
  AND placebo <= 5% AND ranking above its twin's; Holm at 5% over the two rules on the placebo p.
Part 5 - what cannot be seen: coverage per variable (first day with a rank; exits and entries with data; discovery
  events with data); today's exchange labels applied backwards (Coin Metrics), revised series (MVRV, NFCI), single
  venues (Binance funding / OI / long-short / volume, Deribit basis, one Wikipedia page), short histories (ETF from
  2024, open interest and long/short from 2020-09); ETH's own exits not studied here.
Today: "比特幣：50 日均線｜XR｜ER" (持有 / 半倉 / 現金), the most recent exit and entry with what the top discovery
  cells said there. Description, not forecast.
Honesty: thresholds (365-day rank, median split, 30-day change, 20% / 0% labels, 60 days / 8% / 20-day stop, half
  size, +10%, 2019 split, minimum counts) fixed by reasoning before the run; no alternatives tried. About 30 variables
  x 2 forms x 2 event types are searched, so single cells that look good are expected by luck; Part 3 says how many.
  Events cluster in choppy stretches. Not advice.
Added after the first run (units only, no definition or threshold changed): the exam's median deepest loss is already
  in percent; the first run displayed it x100 and compared XR's 5-point slack as 0.05. Fixed to the pre-written 5
  percentage points; no rule's verdict changed. Sources list rebuilt with the exact endpoint URLs.
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
from dashboard.studies2 import attention as at
from dashboard.studies2 import blockspace as bs
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import exit_forces as xf
from dashboard.studies2 import follow_turn_signs as fts
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa

ID = "at_the_signal"
DAY = pd.Timedelta(days=1)
COST, SIMS, SEED = msc.COST, msc.SIMS, 20261121
SPLIT = pd.Timestamp("2019-01-01")
RANK_WIN, RANK_MIN, CHG_DAYS = 365, 180, 30
RUNNER, FIZZLE = 0.20, 0.0
BOOT, BOOT_REG, DRAWS, SHIFT_MIN, PL_SIMS = 2000, 1000, 200, 180, 20
MIN_CTRL, MIN_SIDE = 20, 3
MIN_DISC_SIDE, MIN_DISC, MIN_TEST_SIDE, MIN_TEST = 8, 20, 5, 15
TOP_K, BH_Q, PLACEBO_BAR, ALPHA, LOSS_SLACK = 5, 0.10, 0.05, 0.05, 5.0  # deepest loss is in percent
HALF_SIZE, FULL_AT = 0.5, 1.10
NFCI_WIN, NFCI_MIN = 364, 300
DISC_Q, DISC_WIN, DISC_MIN, DISC_DAYS = 0.05, 365, 330, 7
KI = {"ma50": 1, "xr": 150, "x0": 151, "er": 152, "e0": 153}
FIRST_MONTH = SPLIT

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID

GROUPS = ("act", "flow", "prem", "lev", "att", "cond", "val", "vol")
GNAME = {"act": "活躍度", "flow": "買賣流向", "prem": "溢價", "lev": "槓桿與部位", "att": "關注度", "cond": "金融環境",
         "val": "估值與拉伸", "vol": "波動"}
# key: (group, short name, plain gloss, data and delay)
VARS = {
    "act": ("act", "交易所總流量", "送進＋提出交易所的比特幣，30 天加總；越高代表交易所越熱鬧", "Coin Metrics，晚 1 天"),
    "old": ("act", "老幣移動", "放 155 天以上的幣被花掉的量，30 天（修剪單日極端值）；動了不一定是賣", "Open Bitcoin Metrics，晚 7 天"),
    "feez": ("act", "手續費 z 分數", "礦工手續費收入（美元，30 天平均）比過去兩年平常高幾個標準差", "Coin Metrics，晚 1 天"),
    "adr": ("act", "活躍地址", "每天有動用的比特幣地址數，7 天平均", "Coin Metrics，晚 1 天"),
    "vol": ("act", "現貨成交金額", "幣安 BTCUSDT 現貨 7 天成交金額", "幣安 1 小時線，2017-08 起；零手續費期間拿掉"),
    "net7": ("flow", "淨提走比例（7 天）", "（從交易所提走−送進）÷（提走＋送進）；越高代表幣離開交易所越多", "Coin Metrics，晚 1 天"),
    "net30": ("flow", "淨提走比例（30 天）", "同上，30 天", "Coin Metrics，晚 1 天"),
    "bal30": ("flow", "交易所餘額變化", "交易所手上的比特幣比 30 天前多或少（%）", "Coin Metrics，晚 1 天"),
    "stb": ("flow", "穩定幣增加", "美元穩定幣總量 30 天淨增加（場邊的錢）", "DefiLlama，2019-07 起，晚 1 天"),
    "etf": ("flow", "ETF 淨流入", "美國現貨比特幣 ETF 30 天淨流入", "Farside，2024-01 起，晚 1 天"),
    "cbusd": ("prem", "Coinbase 溢價（美元對美元）", "Coinbase 比 Bitstamp 貴多少，7 天平均；兩邊都用美元", "Coinbase＋Bitstamp，2017-08 起"),
    "cbdisc": ("prem", "Coinbase 極端折價（是／否）", "近 7 天內有一天 Coinbase 比幣安便宜的程度，落在過去一年最低 5%", "Coinbase＋幣安，2017-08 起"),
    "kimchi": ("prem", "韓國溢價", "韓國 Upbit 換成美元後比國際價格貴多少，7 天平均", "Upbit＋聯準會匯率"),
    "fund": ("lev", "資金費率", "幣安永續合約 7 天平均；越高代表做多的人越擠", "幣安，2019-09 起"),
    "oi30": ("lev", "未平倉量 30 天變化", "幣安 BTCUSDT 合約還沒平掉的部位比 30 天前多多少", "幣安存檔，2020-09 起"),
    "basis": ("lev", "期貨基差", "Deribit 季度期貨比現貨貴多少（年化）；越高代表願意多付錢做多", "Deribit＋Bitstamp"),
    "lsr": ("lev", "散戶多空比", "幣安所有帳戶裡做多帳戶數÷做空帳戶數，7 天平均", "幣安存檔，2020-09 起"),
    "lsd": ("lev", "散戶比大戶偏多", "散戶多空比和大戶多空比的對數差；越高代表散戶比大戶更偏多", "幣安存檔，2020-09 起"),
    "wiki": ("att", "維基瀏覽量", "英文維基「Bitcoin」頁每天瀏覽人數，7 天平均（關注度）", "維基媒體，2015-07 起，晚 1 天"),
    "nfci": ("cond", "金融條件比平常緊", "NFCI 減它過去一年的平均；越高代表借錢比平常難", "芝加哥聯準銀行，晚約 6 天，會修正"),
    "hyg": ("cond", "垃圾債／公債", "高收益債 ETF 除以 7–10 年公債 ETF；越高代表市場越敢冒信用風險", "Yahoo"),
    "vix": ("cond", "VIX 恐慌指數", "美股選擇權反映的未來 30 天預期波動", "Yahoo"),
    "dxy": ("cond", "美元指數", "美元對一籃子主要貨幣的強弱", "Yahoo"),
    "mvrv": ("val", "MVRV", "市值除以「每顆幣上次移動時價格」加總的成本；越高代表持有人帳面賺越多", "Coin Metrics，晚 1 天，會修正"),
    "ma200": ("val", "離 200 日均線", "價格比 200 日平均高多少", "價格"),
    "hi365": ("val", "離一年最高", "價格比過去 365 天最高收盤低多少（0 = 正在創新高）", "價格"),
    "bear": ("val", "空頭徵兆數", "follow_turn_signs 的六個空頭徵兆有幾個在極端（只用當時已確認的轉折挑徵兆）", "多個來源"),
    "bull": ("val", "多頭徵兆數", "同上，六個多頭徵兆", "多個來源"),
    "rv20": ("vol", "20 天實際波動", "過去 20 天每日漲跌的年化波動", "價格"),
}
KEYS = tuple(VARS)
BINARY = ("cbdisc",)
COUNTS = ("bear", "bull")
FORMS = ("rank", "chg")
FNAME = {"rank": "排名", "chg": "30 天變化"}
SIDE_WORD = {"rank": ("高", "低"), "chg": ("在升", "在降"), "bin": ("有", "沒有")}
KIND_EX = {"whip": "假跌破", "real": "真的下跌", "other": "其他"}

_d = te._d
_ok, _p, _pt, _ci, _share = sd._ok, sd._p, sd._pt, sd._ci, sd._share


# ---------- own store ----------

@contextmanager
def _store():
    """Loaders write into cache/_store/at_the_signal/; the shared store and the stores of signs_before_turns and
    follow_turn_signs are only read (copied in when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    seeds = [REAL_STORE / sbt.ID, REAL_STORE / fts.ID]
    for src_dir in seeds:
        if src_dir.exists():
            for f in src_dir.glob("*.json"):
                dst = MY_STORE / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(f, dst)
                except Exception:  # noqa: BLE001 - a missing seed only means a download
                    pass
    for name in ("exit_forces_cm_btc.json", "blockspace_cm.json"):
        f, dst = REAL_STORE / name, MY_STORE / name
        try:
            if f.exists() and (not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(f, dst)
        except Exception:  # noqa: BLE001
            pass
    old = sbt.MY_STORE
    sbt.MY_STORE = MY_STORE
    try:
        with sbt._own_store():
            yield
    finally:
        sbt.MY_STORE = old


def _pressure() -> tuple[dict | None, pd.DataFrame | None, list[str]]:
    """pressure_before_moves' raw daily series, its stores redirected into this study's folder."""
    from dashboard.studies2 import pressure_before_moves as pbm
    names = ("STORE_CM", "STORE_LL", "STORE_OBM")
    saved = {a: getattr(pbm, a) for a in names}
    try:
        for a, p in saved.items():
            dst = MY_STORE / p.name
            try:
                if p.exists() and (not dst.exists() or p.stat().st_mtime > dst.stat().st_mtime):
                    shutil.copy2(p, dst)
            except Exception:  # noqa: BLE001
                pass
            setattr(pbm, a, dst)
        data = pbm._load()
        raw, _rank, _first = pbm._pressures(data)
        return data, raw, list(data.get("problems") or [])
    except Exception as exc:  # noqa: BLE001 - reported, the variables are then missing
        return None, None, [f"買賣壓力那組資料這次算不出來（{type(exc).__name__}），相關變數沒有算。"]
    finally:
        for a, p in saved.items():
            setattr(pbm, a, p)


# ---------- data ----------

def _lagged(s: pd.Series, idx: pd.DatetimeIndex, lag: int) -> pd.Series:
    return pd.Series(s.reindex(idx - lag * DAY).values.astype(float), index=idx)


def _load() -> dict:
    t0 = time.time()
    problems, dropped = [], {}
    with _store():
        close = te._daily_close()
        series, sdrop, probs, src, _extra = sbt._assemble(close)
        problems += probs
        dropped.update(sdrop)
        pdata, praw, pprob = _pressure()
        problems += pprob
        try:
            got, pr = xf._stored("cm_btc.json", lambda: xf._cm_fetch("btc"))
            if pr:
                problems.append(pr)
            cm = pd.DataFrame.from_dict(got, orient="index")
            cm.index = pd.to_datetime(cm.index)
            bal = cm["SplyExNtv"].sort_index().asfreq("D")
        except Exception as exc:  # noqa: BLE001
            bal = None
            dropped["bal30"] = f"{type(exc).__name__}"
        try:
            ob, pr = bs._onchain()
            problems += pr
            fee = (ob["FeeTotNtv"] * ob["PriceUSD"]).where(ob["FeeTotNtv"] > 0)
            feez = bs._zscore(fee)
        except Exception as exc:  # noqa: BLE001
            feez = None
            dropped["feez"] = f"{type(exc).__name__}"
    other, more = msc._load_other()
    problems += more
    return {"close": close, "series": series, "pdata": pdata, "praw": praw, "bal": bal, "feez": feez,
            "other": other, "problems": problems, "dropped": dropped, "src": src, "secs": time.time() - t0}


def _variables(data: dict) -> tuple[dict, list[str]]:
    """{key: daily series on close.index, value known at that close}."""
    close = data["close"]
    idx = close.index
    S, P = data["series"], data["praw"]
    V, notes = {}, []

    def take(k, s):
        if s is None:
            return
        if isinstance(s, pd.Series):
            s = s.reindex(idx)
        s = pd.Series(np.asarray(s, dtype=float), index=idx)
        s = s.where(np.isfinite(s))
        if s.notna().sum() > RANK_MIN + 30:
            V[k] = s

    if P is not None:
        take("act", P["ACT"])
        take("old", P["S_old"])
        take("vol", P["S_take"] + P["B_take"])
        take("net7", P["NET7"])
        take("net30", P["NET30"])
        take("stb", P["B_stb"])
        take("etf", P["B_etf"])
        take("cbusd", P["B_cb"])
        take("oi30", P["OI"])
    if data["feez"] is not None:
        take("feez", _lagged(data["feez"], idx, 1))
    if data["bal"] is not None:
        b = data["bal"]
        take("bal30", _lagged(b / b.shift(30) - 1, idx, 1))
    for k, sk in (("adr", "adr"), ("kimchi", "kimchi"), ("fund", "funding"), ("basis", "basis"), ("lsr", "ls_r"),
                  ("wiki", "wiki"), ("hyg", "hyg_ief"), ("vix", "vix"), ("dxy", "dxy"), ("mvrv", "mvrv"),
                  ("ma200", "ma200"), ("hi365", "hi365")):
        if sk in S:
            take(k, S[sk])
    if "ls_r" in S and "ls_t" in S:
        with np.errstate(all="ignore"):
            take("lsd", np.log(S["ls_r"].where(S["ls_r"] > 0)) - np.log(S["ls_t"].where(S["ls_t"] > 0)))
    if "nfci" in S:
        n_ = S["nfci"]
        take("nfci", n_ - n_.shift(1).rolling(NFCI_WIN, min_periods=NFCI_MIN).mean())
    lr_ = np.log(close).diff()
    take("rv20", lr_.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(365))
    # Coinbase extreme discount (binary), raw Coinbase / dashboard close (Binance from 2017-08)
    pd_ = data["pdata"]
    if pd_ is not None and pd_.get("cb"):
        cbs = pd.Series({pd.Timestamp(k): (v[0] if isinstance(v, list) else v) for k, v in pd_["cb"].items()},
                        dtype=float).sort_index()
        cbs = cbs[~cbs.index.duplicated(keep="last")]
        first = pd.Timestamp(sd.FIRST)
        raw = (cbs.reindex(idx) / close - 1).where((cbs.reindex(idx) > 0) & (idx >= first))
        lo = raw.shift(1).rolling(DISC_WIN, min_periods=DISC_MIN).quantile(DISC_Q)
        bottom = (raw < lo).astype(float).where(raw.notna() & lo.notna())
        flag = bottom.rolling(DISC_DAYS, min_periods=1).max().where(lo.notna())
        V["cbdisc"] = flag
    # BEAR_COUNT / BULL_COUNT (FORWARD) from signs_before_turns' ranks
    try:
        m30, lev = {}, {}
        for k, s in S.items():
            r = at._trail_pct(s.values, sbt.WIN, sbt.WIN_MIN) * 100
            W = sbt._windows(r)
            m30[k], lev[k] = W["m30"], W["lev"]
        turns = sbt._turns(close)
        sets = fts._sets(close, lev, m30, turns)
        V["bear"] = pd.Series(sets["bear"]["cnt"]["fwd"], index=idx, dtype=float)
        V["bull"] = pd.Series(sets["bull"]["cnt"]["fwd"], index=idx, dtype=float)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"空頭／多頭徵兆數這次算不出來（{type(exc).__name__}）。")
    return V, notes


def _rank(x: pd.Series) -> pd.Series:
    """Mid-rank 0-100 of today's value among the previous RANK_WIN values (>= RANK_MIN present)."""
    v = x.values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    if n > RANK_WIN:
        win = np.lib.stride_tricks.sliding_window_view(v[:-1], RANK_WIN)
        cur = v[RANK_WIN:]
        fin = np.isfinite(win)
        cnt = fin.sum(axis=1)
        below = (np.where(fin, win, np.inf) < cur[:, None]).sum(axis=1)
        eq = (win == cur[:, None]).sum(axis=1)
        ok = np.isfinite(cur) & (cnt >= RANK_MIN)
        out[RANK_WIN:] = np.where(ok, 100.0 * (below + 0.5 * eq) / np.maximum(cnt, 1), np.nan)
    for i in range(RANK_MIN, min(RANK_WIN, n)):
        if np.isfinite(v[i]):
            prev = v[:i][np.isfinite(v[:i])]
            if len(prev) >= RANK_MIN:
                out[i] = 100.0 * (float((prev < v[i]).sum()) + 0.5 * float((prev == v[i]).sum())) / len(prev)
    return pd.Series(out, index=x.index)


def _forms(V: dict) -> tuple[dict, dict, dict]:
    """ranks, changes and the daily 1/0 side series per cell (key, form)."""
    R, C, side = {}, {}, {}
    for k, s in V.items():
        if k in BINARY:
            side[(k, "bin")] = s
            continue
        r = _rank(s)
        R[k] = r
        C[k] = r - r.shift(CHG_DAYS)
        side[(k, "rank")] = (r > 50).astype(float).where(r.notna())
        side[(k, "chg")] = (C[k] > 0).astype(float).where(C[k].notna())
    return R, C, side


# ---------- events ----------

def _exit_events(close: pd.Series, vflag: pd.Series) -> dict:
    c = close.values.astype(float)
    rows = xf._events(close, vflag, pd.DataFrame())
    out = []
    for r in rows:
        e = r["i"]
        fall10 = c[e] / c[e - 10] - 1 if e >= 10 else float("nan")
        after = c[e + 1:e + 61]
        out.append({**r, "fall10": fall10, "after": (float(after.min() / c[e] - 1) if len(after) else float("nan"))})
    n = len(out)
    X = np.full((n, len(wa.KEYS) + 1), np.nan)
    for j, r in enumerate(out):
        for q, k in enumerate(wa.KEYS):
            X[j, q] = r["D"][k]
        X[j, -1] = r["fall10"]
    known = np.array([r["pay"] is not None and r["kind"] != "unknown" for r in out])
    y = np.array([r["pay"] if r["pay"] is not None else np.nan for r in out], dtype=float)
    return {"rows": out, "pos": np.array([r["i"] for r in out], dtype=int),
            "day": pd.DatetimeIndex([r["day"] for r in out]), "known": known, "y": y, "ydisp": y,
            "lab": np.array([r["kind"] for r in out]), "X": X, "type": "exit"}


def _entry_events(close: pd.Series, ma: np.ndarray) -> dict:
    c = close.values.astype(float)
    s_ = pd.Series(c)
    m50 = s_.rolling(50, min_periods=50).mean().values
    m200 = s_.rolling(200, min_periods=200).mean().values
    hi = s_.rolling(365, min_periods=365).max().values
    rv = np.log(s_).diff().rolling(20, min_periods=20).std(ddof=1).values * math.sqrt(365)
    on = np.nan_to_num(np.asarray(ma, dtype=float)) > 0.5
    ent = np.where(~on[:-1] & on[1:])[0] + 1
    n = len(c)
    rows = []
    for s in ent:
        off = np.flatnonzero(~on[s + 1:])
        x = int(s + 1 + off[0]) if len(off) else None
        if x is not None:
            res = c[x] / c[s] - 1 - 2 * COST
            peak = float(c[s:x + 1].max() / c[s] - 1)
            lab = "runner" if res >= RUNNER else ("fizzle" if res <= FIZZLE else "small")
        else:
            res, peak, lab = None, float(c[s:].max() / c[s] - 1), "open"
        desc = [c[s] / m50[s] - 1 if np.isfinite(m50[s]) else np.nan,
                c[s] / c[s - 30] - 1 if s >= 30 else np.nan,
                c[s] / c[s - 90] - 1 if s >= 90 else np.nan,
                c[s] / hi[s] - 1 if np.isfinite(hi[s]) else np.nan,
                rv[s],
                float(c[s] > m200[s]) if np.isfinite(m200[s]) else np.nan]
        rows.append({"i": int(s), "day": close.index[s], "x": x, "res": res, "peak": peak,
                     "len": (x - s) if x is not None else n - 1 - s, "kind": lab, "desc": desc})
    known = np.array([r["res"] is not None for r in rows])
    yd = np.array([r["res"] if r["res"] is not None else np.nan for r in rows], dtype=float)
    with np.errstate(all="ignore"):
        y = np.log1p(yd)
    return {"rows": rows, "pos": np.array([r["i"] for r in rows], dtype=int),
            "day": pd.DatetimeIndex([r["day"] for r in rows]), "known": known, "y": y, "ydisp": yd,
            "lab": np.array([r["kind"] for r in rows]), "X": np.array([r["desc"] for r in rows], dtype=float),
            "type": "entry"}


# ---------- statistics ----------

def _clusters(days: pd.DatetimeIndex) -> np.ndarray:
    return xf._clusters(list(days))


def _boot_sets(cid: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    return xf._boot_sets(cid, draws, seed)


def _controls(Xc: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(Xc.shape[1]):
        v = Xc[:, j]
        miss = ~np.isfinite(v)
        fill = float(np.nanmean(v)) if (~miss).any() else 0.0
        cols.append(np.where(miss, fill, v))
        if miss.any() and not miss.all():
            cols.append(miss.astype(float))
    M = np.column_stack(cols) if cols else np.zeros((Xc.shape[0], 0))
    keep = [j for j in range(M.shape[1]) if np.std(M[:, j]) > 0]
    return M[:, keep]


def _coef(y: np.ndarray, hi: np.ndarray, Z: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), hi.astype(float), Z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def _shares(ev: dict, m: np.ndarray) -> tuple:
    lab = ev["lab"][m]
    if not len(lab):
        return None, None
    if ev["type"] == "exit":
        return float((lab == "whip").mean()), float((lab == "real").mean())
    return float((lab == "runner").mean()), float((lab == "fizzle").mean())


def _cell(ev: dict, side: pd.Series, seed: int, sel: np.ndarray | None = None, placebo: bool = True,
          min_side: int = MIN_SIDE, min_n: int = MIN_CTRL) -> dict:
    vals = side.values[ev["pos"]]
    base = ev["known"] & np.isfinite(vals)
    if sel is not None:
        base &= sel
    ix = np.flatnonzero(base)
    n = len(ix)
    hi = vals[ix] > 0.5
    res = {"n": n, "nh": int(hi.sum()), "nl": int((~hi).sum())}
    if n == 0:
        return res
    y, yd = ev["y"][ix], ev["ydisp"][ix]

    def grp(m):
        k = int(m.sum())
        a, b = _shares(ev, ix[m]) if k else (None, None)
        return {"n": k, "a": a, "b": b, "mean": float(np.mean(yd[m])) if k else None,
                "med": float(np.median(yd[m])) if k else None, "ym": float(np.mean(y[m])) if k else None}

    res["hi"], res["lo"] = grp(hi), grp(~hi)
    days = ev["day"][ix]
    cid = _clusters(days)
    res["clusters"] = int(len(np.unique(cid)))
    if res["nh"] < min_side or res["nl"] < min_side:
        return res
    res["diff"] = float(y[hi].mean() - y[~hi].mean())
    sets = _boot_sets(cid, BOOT, seed)
    ds = []
    for s in sets:
        h_ = hi[s]
        if h_.any() and (~h_).any():
            ds.append(float(y[s][h_].mean() - y[s][~h_].mean()))
    res["d_ci"] = (float(np.percentile(ds, 5)), float(np.percentile(ds, 95))) if ds else (None, None)
    if n < min_n:
        return res
    Z = _controls(ev["X"][ix])
    res["coef"] = _coef(y, hi, Z)
    cs = []
    for s in sets[:BOOT_REG]:
        h_ = hi[s]
        if h_.sum() >= 2 and (~h_).sum() >= 2:
            cs.append(_coef(y[s], h_, Z[s]))
    if len(cs) >= 100:
        res["c_ci"] = (float(np.percentile(cs, 5)), float(np.percentile(cs, 95)))
        res["se"] = float(np.std(cs, ddof=1))
        if res["se"] > 0:
            z = abs(res["coef"]) / res["se"]
            res["p_bh"] = float(math.erfc(z / math.sqrt(2)))
    if not placebo:
        return res
    # placebo: circular shift of the daily side series against the fixed event dates
    s_all = side
    valid = s_all.notna().values
    if not valid.any():
        return res
    first = int(np.argmax(valid))
    last = len(valid) - 1 - int(np.argmax(valid[::-1]))
    span = s_all.values[first:last + 1]
    m = len(span)
    if m < 2 * SHIFT_MIN + 10:
        return res
    known_ix = np.flatnonzero(ev["known"] & (sel if sel is not None else True))
    at_ = ev["pos"][known_ix] - first
    inside = (at_ >= 0) & (at_ < m)
    rng = np.random.default_rng(seed + 77)
    draws = []
    for _ in range(DRAWS):
        k = int(rng.integers(SHIFT_MIN, m - SHIFT_MIN))
        rolled = np.roll(span, k)
        v = np.full(len(known_ix), np.nan)
        v[inside] = rolled[at_[inside]]
        ok = np.isfinite(v)
        jj = known_ix[ok]
        h2 = v[ok] > 0.5
        if len(jj) < min_n or h2.sum() < min_side or (~h2).sum() < min_side:
            draws.append(0.0)
            continue
        draws.append(abs(_coef(ev["y"][jj], h2, _controls(ev["X"][jj]))))
    draws = np.array(draws)
    res["pl_draws"] = draws
    res["placebo"] = float(np.mean(draws >= abs(res["coef"]) - 1e-12))
    return res


def _passes(a: dict) -> bool:
    ci = a.get("c_ci")
    return bool(ci and _ok(ci[0]) and _ok(ci[1]) and (ci[0] > 0 or ci[1] < 0)
                and _ok(a.get("placebo")) and a["placebo"] <= PLACEBO_BAR)


def _luck(a: dict) -> float | None:
    d = a.get("pl_draws")
    if d is None or not _ok(a.get("se")) or len(d) < 20:
        return None
    hits = 0
    for j in range(len(d)):
        others = np.delete(d, j)
        if d[j] > 1.645 * a["se"] and float(np.mean(others >= d[j] - 1e-12)) <= PLACEBO_BAR:
            hits += 1
    return hits / len(d)


def _bh(ps: dict, q: float) -> list:
    items = sorted((p, k) for k, p in ps.items() if _ok(p))
    m = len(items)
    cut = 0
    for i, (p, _k) in enumerate(items, 1):
        if p <= q * i / m:
            cut = i
    return [k for _p_, k in items[:cut]]


# ---------- the exam ----------

class _Exam(xf._Exam):
    """exit_forces' exam; spec None = ma50, ("x", flag | "all") = sit through exits (XR / X0),
    ("e", flag | "all") = half-size entries (ER / E0)."""

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first)
        self.emeta = {}
        s = self.series["btc"]
        self.emeta["btc"] = self._em(s["ma"], close.index, 0)
        for h in range(24):
            s = self.series[("clock", h)]
            self.emeta[("clock", h)] = self._em(s["ma"], pd.DatetimeIndex(self.m.dates[:len(s["ma"])]), 1)
        for key, o in self.other.items():
            s = self.series[("o", key)]
            self.emeta[("o", key)] = self._em(s["ma"], o["close"].index, o["lag"])

    @staticmethod
    def _em(ma, dates, lag) -> dict:
        on = np.nan_to_num(np.asarray(ma, dtype=float)) > 0.5
        en = np.where(~on[:-1] & on[1:])[0] + 1
        return {"en": en, "dates": pd.DatetimeIndex(dates[en]) - lag * DAY}

    def pos(self, key, spec, perm=None) -> np.ndarray:
        if isinstance(spec, tuple):
            kind, flag = spec
            if kind == "x":
                return super().pos(key, flag)
            return self._half(key, flag)
        return super().pos(key, spec)

    def _half(self, key, flag) -> np.ndarray:
        s, em = self.series[key], self.emeta[key]
        c = np.asarray(s["c"], dtype=float)
        pos = np.nan_to_num(np.asarray(s["ma"], dtype=float)).copy()
        on = pos > 0.5
        n = min(len(c), len(pos))
        if isinstance(flag, str):
            vals = np.ones(len(em["en"]), dtype=bool)
        else:
            vals = flag.reindex(em["dates"]).fillna(0.0).values > 0.5
        for s0, v in zip(em["en"], vals):
            if not v or s0 >= n:
                continue
            t = s0
            while t < n and on[t]:
                if c[t] >= FULL_AT * c[s0]:
                    break
                pos[t] = HALF_SIZE
                t += 1
        return pos


# ---------- computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    c = close.values.astype(float)
    V, vnotes = _variables(data)
    R, C, side = _forms(V)
    ma = msc._targets(close)["ma50"].values.astype(float)
    # descriptor (i): Binance daily quote volume from pressure_before_moves' 1-hour bars (24 hours present)
    vflag = pd.Series(dtype=float)
    if data["pdata"] is not None and data["pdata"].get("hourly"):
        from dashboard.studies2 import pressure_before_moves as pbm
        td = pbm._taker_daily(data["pdata"]["hourly"])
        q = td["q"].where(td["h"] >= 24)
        vflag = wa._vol_flag(q)
    EX = _exit_events(close, vflag)
    EN = _entry_events(close, ma)
    EV = {"exit": EX, "entry": EN}
    cells = [(k, f) for k in KEYS for f in (("bin",) if k in BINARY else FORMS) if (k, f) in side]

    A, D, T = {}, {}, {}
    for et, ev in EV.items():
        disc = np.asarray(ev["day"] < SPLIT)
        test = ~disc
        for j, cell in enumerate(cells):
            sd_ = SEED + 1000 * (et == "entry") + 7 * j
            A[(et,) + cell] = _cell(ev, side[cell], sd_)
            D[(et,) + cell] = _cell(ev, side[cell], sd_ + 3, sel=disc, placebo=False,
                                    min_side=MIN_DISC_SIDE, min_n=MIN_DISC)
            T[(et,) + cell] = _cell(ev, side[cell], sd_ + 5, sel=test, placebo=False,
                                    min_side=MIN_TEST_SIDE, min_n=MIN_TEST)
    t_cells = time.time() - t0

    # Part 3
    tested = [k for k, a in A.items() if _ok(a.get("coef")) and a.get("c_ci")]
    passing = [k for k in tested if _passes(A[k])]
    luck = [x for x in (_luck(A[k]) for k in tested) if x is not None]
    bh = _bh({k: A[k].get("p_bh") for k in tested}, BH_Q)
    pick = {}
    for et in EV:
        cand = [k for k, a in D.items() if k[0] == et and _ok(a.get("coef"))]
        cand.sort(key=lambda k: (-abs(D[k]["coef"]), k))
        pick[et] = cand[:TOP_K]

    # Part 4
    rows_h, problem = cr._hourly("BTCUSDT")
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    ex = _Exam(close, data["other"], price, b0, pre, FIRST_MONTH)
    flags = {}
    if pick["exit"]:
        k = pick["exit"][0]
        stay = 1.0 if D[k]["coef"] > 0 else 0.0
        sv = side[k[1:]]
        flags["xr"] = (sv == stay).astype(float).where(sv.notna())
        flags["xr_side"] = stay
    if pick["entry"]:
        k = pick["entry"][0]
        weak = 0.0 if D[k]["coef"] > 0 else 1.0
        sv = side[k[1:]]
        flags["er"] = (sv == weak).astype(float).where(sv.notna())
        flags["er_side"] = weak
    specs = {"ma50": None, "x0": ("x", "all"), "e0": ("e", "all")}
    if "xr" in flags:
        specs["xr"] = ("x", flags["xr"])
    if "er" in flags:
        specs["er"] = ("e", flags["er"])
    s = {k: ex.summary(sp, KI[k], SIMS) for k, sp in specs.items()}
    t_exam = time.time()
    small, hits = {}, {}
    rng = np.random.default_rng(SEED + 9000)
    rules = [k for k in ("xr", "er") if k in specs]
    for k in rules:
        small[k] = ex.summary(specs[k], KI[k], PL_SIMS)
        hits[k] = 0
    for _ in range(DRAWS):
        u = float(rng.random())
        for k in rules:
            kind, fl = specs[k]
            x = ex.summary((kind, xf._roll(fl, u)), KI[k], PL_SIMS)
            hits[k] += int(sd._score_ge(x, small[k]))
    placebo = {k: hits[k] / DRAWS for k in rules}
    pval = {k: (1 + hits[k]) / (1 + DRAWS) for k in rules}
    holm = xf._holm(pval) if pval else {}
    support = {}
    base = s["ma50"]
    for k in rules:
        x = s[k]
        twin = s["x0" if k == "xr" else "e0"]
        slack = LOSS_SLACK if k == "xr" else 0.0
        items = {"rank": sd._score_ge(x, base),
                 "loss": x["c_deep_med"] >= base["c_deep_med"] - slack - 1e-12,
                 "placebo": placebo[k] <= PLACEBO_BAR,
                 "twin": sd._score_ge(x, twin) and not (x["low"] <= twin["low"] + 1e-12 and x["eligible"] == twin["eligible"])}
        support[k] = {"items": items, "ok": all(items.values()), "holm": bool(holm.get(k, False))}
    pos = {k: ex.pos("btc", sp) for k, sp in specs.items()}
    return {"V": V, "R": R, "C": C, "side": side, "EX": EX, "EN": EN, "cells": cells, "A": A, "D": D, "T": T,
            "tested": tested, "passing": passing, "luck": luck, "bh": bh, "pick": pick, "flags": flags,
            "s": s, "placebo": placebo, "pval": pval, "holm": holm, "support": support, "pos": pos, "ex": ex,
            "problem": problem, "vnotes": vnotes, "secs": time.time() - t0, "secs_cells": t_cells,
            "secs_exam": time.time() - t_exam}


# ---------- report helpers ----------

def _cname(cell: tuple) -> str:
    k, f = cell[-2], cell[-1]
    base = VARS[k][1]
    return base if f == "bin" else f"{base}（{FNAME[f]}）"


def _sw(cell: tuple, hi: bool) -> str:
    f = cell[-1]
    w = SIDE_WORD["bin" if f == "bin" else f]
    return w[0] if hi else w[1]


def _pp(v) -> str:
    """log difference -> points"""
    return f"{float(v) * 100:+.1f} 點" if _ok(v) else "—"


def _pcs(v) -> str:
    """already in percent"""
    return f"{float(v):.1f}%" if _ok(v) else "—"


def _pts(v) -> str:
    """difference already in percent points"""
    return f"{float(v):+.1f} 點" if _ok(v) else "—"


def _r0(v) -> str:
    return f"{float(v):.0f}" if _ok(v) else "—"


def _cell_row(cell: tuple, a: dict, et: str) -> list:
    name = _cname(cell)
    hw, lw = _sw(cell, True), _sw(cell, False)
    if a.get("n", 0) == 0 or "hi" not in a:
        return [name, "0", "—", "—", "—", "—", "—", "—", "—"]
    h, l = a["hi"], a["lo"]
    nn = f"{hw} {h['n']}｜{lw} {l['n']}（{a['clusters']} 群）"
    sh = lambda g: f"{_share(g['a'])}｜{_share(g['b'])}" if g["n"] else "—"  # noqa: E731
    mm = lambda g: f"{_p(g['mean'])}｜{_p(g['med'])}" if g["n"] else "—"  # noqa: E731
    f = _pt if et == "exit" else _pp
    diff = f"{f(a.get('diff'))} {_ci(*a['d_ci'], f=f)}" if _ok(a.get("diff")) else "資料不夠"
    ctrl = (f"{f(a.get('coef'))} {_ci(*a['c_ci'], f=f)}" if _ok(a.get("coef")) and a.get("c_ci")
            else ("資料不夠" if not _ok(a.get("coef")) else f(a.get("coef"))))
    pl = _share(a["placebo"]) if _ok(a.get("placebo")) else "—"
    return [name, nn, sh(h), sh(l), mm(h), mm(l), diff, ctrl, pl]


def _sort_key(a: dict):
    return -abs(a["coef"]) if _ok(a.get("coef")) else 1.0


def _composite(R: dict, V: dict, i: int) -> dict:
    out = {}
    for g in GROUPS:
        vals = [R[k].values[i] for k in KEYS if VARS[k][0] == g and k in R and k not in COUNTS
                and _ok(R[k].values[i])]
        out[g] = float(np.mean(vals)) if vals else None
    for k in COUNTS:
        out[k] = float(V[k].values[i]) if k in V and _ok(V[k].values[i]) else None
    return out


def _comp_cells(cp: dict) -> list:
    cells = []
    for g in GROUPS:
        v = _r0(cp[g])
        if g == "val" and (cp.get("bear") is not None or cp.get("bull") is not None):
            v += f"；空 {_r0(cp.get('bear'))}／多 {_r0(cp.get('bull'))}"
        cells.append(v)
    return cells


def _comp_median(comps: list) -> dict:
    out = {}
    for g in list(GROUPS) + list(COUNTS):
        v = [c[g] for c in comps if c.get(g) is not None]
        out[g] = float(np.median(v)) if v else None
    return out


def _say(cell: tuple, i: int, side: dict, R: dict, C: dict, V: dict) -> str:
    k, f = cell[-2], cell[-1]
    sv = side[(k, f)].values[i]
    if not _ok(sv):
        return "無資料"
    word = _sw(cell, sv > 0.5)
    if f == "rank":
        return f"{word}（排名 {R[k].values[i]:.0f}）"
    if f == "chg":
        return f"{word}（排名 30 天變化 {C[k].values[i]:+.0f}）"
    return word


def _exam_row(name: str, x: dict, base: dict, pl=None, holm=None, ok=None) -> list:
    ch = x["checks"]
    return [name, _share(x["low"]) + ("" if x["eligible"] else "（跌幅不合格）"),
            f"{_share(ch[0])}｜{_share(ch[1])}", f"{_share(ch[2])}｜{_share(ch[3])}",
            _pcs(x["c_deep_med"]), _pts(x["c_deep_med"] - base["c_deep_med"]),
            _share(pl) if _ok(pl) else "—", "—" if holm is None else ("通過" if holm else "沒通過"),
            "—" if ok is None else ("支持" if ok else "沒有看到")]


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close = data["close"]
    c = close.values.astype(float)
    V, R, C, side, A, D, T = r["V"], r["R"], r["C"], r["side"], r["A"], r["D"], r["T"]
    EX, EN = r["EX"], r["EN"]
    problems = data["problems"] + r["vnotes"] + ([r["problem"]] if r["problem"] else [])
    cells = r["cells"]
    tables = []

    # ----- definitions -----
    first_rank = {}
    for k in KEYS:
        s = R.get(k) if k not in BINARY else V.get(k)
        fv = s.first_valid_index() if s is not None else None
        first_rank[k] = _d(fv) if fv is not None else "—"
    tables.append({"title": "用到的變數（每個都換算成「當天已經知道」的值）",
                   "columns": ["變數", "組別", "白話意思", "資料來源與延遲", "最早有排名的日子"],
                   "rows": [[VARS[k][1], GNAME[VARS[k][0]], VARS[k][2], VARS[k][3], first_rank[k]] for k in KEYS]})

    # ----- event overview -----
    exk = [rw for rw, kn in zip(EX["rows"], EX["known"]) if kn]
    enk = [rw for rw, kn in zip(EN["rows"], EN["known"]) if kn]
    rows_ev = []
    for lab in ("whip", "real", "other"):
        g = [x["pay"] for x in exk if x["kind"] == lab]
        rows_ev.append(["下車", KIND_EX[lab], f"{len(g)}", _share(len(g) / len(exk)) if exk else "—",
                        _p(np.mean(g)) if g else "—", _p(np.median(g)) if g else "—"])
    allp = [x["pay"] for x in exk]
    rows_ev.append(["下車", "全部（每次都不賣、用停損）", f"{len(exk)}", "100%", _p(np.mean(allp)), _p(np.median(allp))])
    for lab, nm in (("runner", "大漲（這段賺 ≥ 20%）"), ("small", "小賺（0～20%）"), ("fizzle", "沒漲起來（≤ 0）")):
        g = [x["res"] for x in enk if x["kind"] == lab]
        rows_ev.append(["上車", nm, f"{len(g)}", _share(len(g) / len(enk)) if enk else "—",
                        _p(np.mean(g)) if g else "—", _p(np.median(g)) if g else "—"])
    alle = [x["res"] for x in enk]
    rows_ev.append(["上車", "全部", f"{len(enk)}", "100%", _p(np.mean(alle)), _p(np.median(alle))])
    tables.append({"title": "事件總覽：50 日均線的下車與上車（2010 年起）",
                   "columns": ["事件", "種類", "次數", "占比", "平均（下車：多抱的得失；上車：這段的結果）", "中位數"],
                   "rows": rows_ev})

    # ----- Part 1 / Part 2 tables -----
    cols1 = ["變數", "有資料的下車：高｜低（群）", "高：假跌破｜真的下跌", "低：假跌破｜真的下跌", "高：平均｜中位數",
             "低：平均｜中位數", "差（高−低）［90% 區間］", "扣掉價格後的差［90% 區間］", "隨機錯位"]
    cols2 = ["變數", "有資料的上車：高｜低（群）", "高：大漲｜沒漲起來", "低：大漲｜沒漲起來", "高：平均｜中位數",
             "低：平均｜中位數", "差（高−低，對數）［90% 區間］", "扣掉價格後的差［90% 區間］", "隨機錯位"]
    for et, cols, word in (("exit", cols1, "下車"), ("entry", cols2, "上車")):
        for form, fword in (("rank", "排名高（高於自己過去一年的中位數）對排名低"),
                            ("chg", "排名在升（比 30 天前高）對在降")):
            cl = [cell for cell in cells if cell[1] == form or (form == "rank" and cell[1] == "bin")]
            cl.sort(key=lambda cell: _sort_key(A[(et,) + cell]))
            tables.append({"title": f"第 {1 if et == 'exit' else 2} 部分：{word}那一刻，{fword}（依扣掉價格後的差排序）",
                           "columns": cols, "rows": [_cell_row(cell, A[(et,) + cell], et) for cell in cl]})

    # ----- real declines one by one -----
    comps_ex = {int(rw["i"]): _composite(R, V, rw["i"]) for rw in exk}
    real_rows = []
    for rw in exk:
        if rw["kind"] != "real":
            continue
        real_rows.append([f"{_d(rw['day'])}（之後最低 {_p(rw['after'])}，多抱 {_p(rw['pay'])}）"]
                         + _comp_cells(comps_ex[int(rw["i"])]))
    for lab in ("real", "whip"):
        med = _comp_median([comps_ex[int(rw["i"])] for rw in exk if rw["kind"] == lab])
        n_ = sum(1 for rw in exk if rw["kind"] == lab)
        real_rows.append([f"（對照）全部{KIND_EX[lab]}的中位數，{n_} 次"] + _comp_cells(med))
    tables.append({"title": "真的下跌的下車，一次一次看：每組變數當天的排名平均（0–100，50 = 跟過去一年的中位數一樣）",
                   "columns": ["下車日（之後怎麼走）"] + [GNAME[g] for g in GROUPS], "rows": real_rows})

    # ----- largest trades -----
    comps_en = {int(rw["i"]): _composite(R, V, rw["i"]) for rw in enk}
    big = sorted(enk, key=lambda x: -x["res"])[:15]
    big_rows = [[f"{_d(rw['day'])}（{_p(rw['res'])}，最高 {_p(rw['peak'])}，{rw['len']} 天）"]
                + _comp_cells(comps_en[int(rw["i"])]) for rw in big]
    for lab, nm in (("runner", "大漲"), ("fizzle", "沒漲起來")):
        med = _comp_median([comps_en[int(rw["i"])] for rw in enk if rw["kind"] == lab])
        n_ = sum(1 for rw in enk if rw["kind"] == lab)
        big_rows.append([f"（對照）全部{nm}的中位數，{n_} 次"] + _comp_cells(med))
    tables.append({"title": "最大的 15 段上車，一段一段看：上車那天每組變數的排名平均",
                   "columns": ["上車日（這段結果，最高，天數）"] + [GNAME[g] for g in GROUPS], "rows": big_rows})

    # ----- Part 3 -----
    ntest = len(r["tested"])
    luck_sum = float(np.sum(r["luck"])) if r["luck"] else None
    ex_t = [k for k in r["tested"] if k[0] == "exit"]
    en_t = [k for k in r["tested"] if k[0] == "entry"]
    p3 = [["有算出「扣掉價格後的差」的格子", f"{ntest}（下車 {len(ex_t)}、上車 {len(en_t)}）"],
          ["區間不含 0 而且隨機錯位 ≤ 5% 的格子", f"{len(r['passing'])}"],
          ["純靠運氣預期會有幾格（用隨機錯位本身估）", f"約 {luck_sum:.1f}" if _ok(luck_sum) else "—"],
          ["簡單上限：5% × 格子數", f"{0.05 * ntest:.1f}"],
          ["Benjamini-Hochberg（10%）後還留下的格子", f"{len(r['bh'])}"]]
    tables.append({"title": "第 3 部分：格子這麼多，有幾格是運氣？", "columns": ["項目", "數字"], "rows": p3})
    surv_rows = []
    for k in sorted(set(r["passing"]) | set(r["bh"]), key=lambda k: -abs(A[k]["coef"])):
        a = A[k]
        f = _pt if k[0] == "exit" else _pp
        surv_rows.append(["下車" if k[0] == "exit" else "上車", _cname(k), f"{a['n']}（{a['clusters']} 群）",
                          f"{f(a['coef'])} {_ci(*a['c_ci'], f=f)}", _share(a.get("placebo")),
                          f"{a['p_bh']:.4f}" if _ok(a.get("p_bh")) else "—", "是" if k in r["bh"] else "否"])
    if surv_rows:
        tables.append({"title": "通過「區間不含 0、隨機錯位 ≤ 5%」或 Benjamini-Hochberg 的格子",
                       "columns": ["事件", "變數", "N（群）", "扣掉價格後的差［90% 區間］", "隨機錯位", "p（常態近似）",
                                   "過 BH？"], "rows": surv_rows})

    split_rows = {}
    for et in ("exit", "entry"):
        f = _pt if et == "exit" else _pp
        rows_s = []
        for j, k in enumerate(r["pick"][et], 1):
            d, t = D[k], T[k]
            same = (_ok(t.get("coef")) and np.sign(t["coef"]) == np.sign(d["coef"]))
            excl = bool(t.get("c_ci") and _ok(t["c_ci"][0]) and (t["c_ci"][0] > 0 or t["c_ci"][1] < 0))
            rows_s.append([f"{j}. {_cname(k)}", f"{d['n']}（{d['nh']}｜{d['nl']}）",
                           f"{f(d['coef'])} {_ci(*d['c_ci'], f=f)}" if d.get("c_ci") else f(d.get("coef")),
                           f"{t['n']}（{t['nh']}｜{t['nl']}）",
                           (f"{f(t['coef'])} {_ci(*t['c_ci'], f=f)}" if t.get("c_ci") else f(t.get("coef")))
                           if _ok(t.get("coef")) else "資料不夠",
                           ("一樣" if same else "相反") if _ok(t.get("coef")) else "—",
                           ("不含 0" if excl else "含 0") if _ok(t.get("coef")) else "—"])
        split_rows[et] = rows_s
        tables.append({"title": f"第 3 部分：誠實切分（{'下車' if et == 'exit' else '上車'}）：2019 年以前挑出前五名，2019 年起再看一次",
                       "columns": ["2019 前挑出的格子", "2019 前 N（高｜低）", "2019 前：扣掉價格後的差［90%］",
                                   "2019 起 N（高｜低）", "2019 起：扣掉價格後的差［90%］", "方向", "2019 起區間"],
                       "rows": rows_s})

    # ----- Part 4 -----
    s_, base = r["s"], r["s"]["ma50"]
    fl = r["flags"]
    xr_desc = er_desc = "沒有選出變數"
    if r["pick"]["exit"]:
        k = r["pick"]["exit"][0]
        xr_desc = f"{_cname(k)} 在「{_sw(k, fl['xr_side'] > 0.5)}」那一邊時不賣"
    if r["pick"]["entry"]:
        k = r["pick"]["entry"][0]
        er_desc = f"{_cname(k)} 在「{_sw(k, fl['er_side'] > 0.5)}」那一邊時半倉上車"
    ex_rows = [_exam_row("50 日均線（基準）", base, base)]
    for k, nm in (("xr", f"XR：{xr_desc}，最多再抱 60 天（有停損）"), ("x0", "X0（對照）：每次下車都不賣，同樣停損"),
                  ("er", f"ER：{er_desc}，賺到 10% 才加滿"), ("e0", "E0（對照）：每次上車都先半倉，賺到 10% 才加滿")):
        if k not in s_:
            continue
        sp = r["support"].get(k)
        ex_rows.append(_exam_row(nm, s_[k], base, r["placebo"].get(k), sp["holm"] if sp else None,
                                 sp["ok"] if sp else None))
    tables.append({"title": "第 4 部分：照著做（只用 2019 年以前挑出的第一名），公平考試 2019 年起的起點",
                   "columns": ["規則", "排名分數", "贏持有：每月起點｜收盤時間", "贏持有：一般人｜其他資產",
                               "收盤時間中位最深跌幅", "比 50 日均線", "隨機錯位", "Holm（兩條）", "達標？"],
                   "rows": ex_rows})
    lab_items = {"rank": "排名分數 ≥ 50 日均線", "loss": "最深跌幅條件", "placebo": "隨機錯位 ≤ 5%", "twin": "比對照好"}
    bar_rows = [[("XR" if k == "xr" else "ER")] + [("過" if r["support"][k]["items"][x] else "沒過") for x in lab_items]
                for k in r["support"]]
    if bar_rows:
        tables.append({"title": "第 4 部分：每個條件有沒有過", "columns": ["規則"] + list(lab_items.values()),
                       "rows": bar_rows})

    # ----- Part 5 coverage -----
    cov = []
    exd, end_ = np.asarray(EX["day"] < SPLIT), np.asarray(EN["day"] < SPLIT)
    for k in KEYS:
        cell = (k, "bin") if k in BINARY else (k, "rank")
        if cell not in side:
            cov.append([VARS[k][1], "沒有資料", "0", "0", "0｜0"])
            continue
        sv = side[cell]
        hx = np.isfinite(sv.values[EX["pos"]]) & EX["known"]
        hn = np.isfinite(sv.values[EN["pos"]]) & EN["known"]
        cov.append([VARS[k][1], first_rank[k], f"{int(hx.sum())}", f"{int(hn.sum())}",
                    f"{int((hx & exd).sum())}｜{int((hn & end_).sum())}"])
    tables.append({"title": "第 5 部分：看得到多少（每個變數有資料的事件數）",
                   "columns": ["變數", "最早有排名", "下車（已知結果）", "上車（已知結果）", "2019 前：下車｜上車"],
                   "rows": cov})

    # ----- now -----
    word = lambda v: "持有" if v > 0.75 else ("半倉" if v > 0.25 else "現金")  # noqa: E731
    pos = r["pos"]
    now = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜XR｜ER",
            "value": f"{word(pos['ma50'][-1])}｜{word(pos['xr'][-1]) if 'xr' in pos else '—'}｜"
                     f"{word(pos['er'][-1]) if 'er' in pos else '—'}"}]
    if EX["rows"]:
        last = EX["rows"][-1]
        i = last["i"]
        now.append({"label": "最近一次 50 日均線下車",
                    "value": f"{_d(last['day'])}，收盤 {c[i]:,.0f} 美元（{len(c) - 1 - i} 天前）"
                             + ("；這次多抱的得失還不知道" if last["pay"] is None else f"；多抱的得失 {_p(last['pay'])}")})
        for k in r["pick"]["exit"]:
            now.append({"label": f"　那天 {_cname(k)}", "value": _say(k, i, side, R, C, V)})
    if EN["rows"]:
        last = EN["rows"][-1]
        i = last["i"]
        now.append({"label": "最近一次 50 日均線上車",
                    "value": f"{_d(last['day'])}，收盤 {c[i]:,.0f} 美元"
                             + ("（這段還沒結束）" if last["res"] is None else f"；這段結果 {_p(last['res'])}")})
        for k in r["pick"]["entry"]:
            now.append({"label": f"　那天 {_cname(k)}", "value": _say(k, i, side, R, C, V)})
    now.append({"label": "這幾格的意思", "value": "只描述最後一根日線收盤後的狀態，不是預測，也不是買賣建議"})

    # ----- verdict -----
    nex, nen = len(exk), len(enk)
    nreal = sum(1 for x in exk if x["kind"] == "real")
    nwhip = sum(1 for x in exk if x["kind"] == "whip")
    nrun = sum(1 for x in enk if x["kind"] == "runner")
    nfiz = sum(1 for x in enk if x["kind"] == "fizzle")

    def top_line(et: str, m: int = 3) -> str:
        ks = [k for k in r["tested"] if k[0] == et]
        ks.sort(key=lambda k: -abs(A[k]["coef"]))
        f = _pt if et == "exit" else _pp
        out = []
        for k in ks[:m]:
            a = A[k]
            out.append(f"{_cname(k)}「{_sw(k, True)}」比「{_sw(k, False)}」{f(a['coef'])}"
                       f"（{_ci(*a['c_ci'], f=f)}，隨機錯位 {_share(a.get('placebo'))}，{a['n']} 次）")
        return "；".join(out) if out else "沒有可以算的格子"

    def split_line(et: str) -> str:
        rows_s = split_rows.get(et) or []
        same = sum(1 for x in rows_s if x[5] == "一樣")
        excl = sum(1 for x in rows_s if x[5] == "一樣" and x[6] == "不含 0")
        have = sum(1 for x in rows_s if x[5] in ("一樣", "相反"))
        return f"前五名裡 {have} 個在 2019 年起有資料，{same} 個方向一樣，{excl} 個方向一樣且區間不含 0"

    def rule_line(k: str) -> str:
        if k not in r["support"]:
            return f"{'XR' if k == 'xr' else 'ER'} 沒有建出來（2019 年前沒有選出變數）"
        x, sp = s_[k], r["support"][k]
        tw = s_["x0" if k == "xr" else "e0"]
        return (f"{'XR' if k == 'xr' else 'ER'} 排名分數 {_share(x['low'])}（50 日均線 {_share(base['low'])}、對照 "
                f"{_share(tw['low'])}），中位最深跌幅 {_pcs(x['c_deep_med'])}（50 日均線 {_pcs(base['c_deep_med'])}），"
                f"隨機錯位 {_share(r['placebo'][k])} → {'支持' if sp['ok'] else '沒有看到'}")

    verdict = (
        f"量到的：50 日均線有 {nex} 次已知結果的下車（假跌破 {nwhip}、真的下跌 {nreal}）和 {nen} 段已知結果的上車"
        f"（大漲 {nrun}、沒漲起來 {nfiz}）。把 {len(KEYS)} 個變數放到這些事件上，每個看「排名」和「30 天變化」兩種。"
        f"下車時差最大的（扣掉價格後）：{top_line('exit')}。上車時差最大的：{top_line('entry')}。"
        f"運氣檢查：{ntest} 格裡 {len(r['passing'])} 格區間不含 0 且隨機錯位 ≤ 5%，純靠運氣預期約 "
        f"{(luck_sum if _ok(luck_sum) else 0.05 * ntest):.1f} 格；Benjamini-Hochberg 後留下 {len(r['bh'])} 格。"
        f"誠實切分：下車{split_line('exit')}；上車{split_line('entry')}。"
        f"照著做（2019 年起考）：{rule_line('xr')}；{rule_line('er')}。"
        "看不到的：很多變數 2017–2020 年以後才有，2019 年以前挑不到它們；交易所地址是今天的標籤往回套；槓桿和成交量只有幣安。"
        "這些是描述，不是預測，也不是買賣建議。"
    )

    notes = [
        "怎麼讀：每個變數在事件當天換算成「排名」——它在自己過去一年裡排第幾（0–100）；排名 > 50 叫「高」。"
        "「30 天變化」是排名比 30 天前高（在升）或低（在降）。只用當天以前已經公布的資料。",
        "下車的得失（多抱）：下車那天不賣，抱到 50 日均線重新上車、停損（比下車價低 8%，或跌破前 20 天最低收盤）或 60 天，"
        "先到的那個；減掉拿現金（0）。上車的結果：從上車那天收盤抱到下一次 50 日均線下車，扣兩次 0.10% 成本。",
        "上車那邊的「差」用對數報酬算（log(1+結果)），因為少數幾段十倍行情會把普通平均拉走；表上的平均和中位數仍是一般報酬。",
        "「扣掉價格後的差」：把下車那天的十個價格描述（whipsaw_anatomy）和下車前 10 天跌了多少一起放進迴歸，看變數高低還多出多少；"
        "上車用離 50 日均線多遠、前 30 天和 90 天漲跌、離一年最高多遠、20 天波動、是否在 200 日均線上。迴歸不是配對，事件少時不穩。",
        "區間是整群重抽：30 天內接連發生的事件算同一群，因為它們是同一段行情，不是獨立的試驗。",
        "隨機錯位：把變數的每日高低序列整條往後挪 180 天以上（頭尾接起來），事件日期不動，重算 200 次；"
        "看有多少次的差（取絕對值）不比真的小。這個比例越小，越不像巧合。",
        "Benjamini-Hochberg 用的是「扣掉價格後的差 ÷ 重抽標準誤」的常態近似 p 值；隨機錯位只有 200 次，最小的 p（1/201）"
        "本身就過不了 BH 第一關，所以沒有拿它來做 BH（直接說明，不另外繞過）。",
        "誠實切分：只用 2019 年以前的事件把格子排名、挑出前五名，再看 2019 年起同一格的差。2019 年以前沒有資料的變數"
        "（資金費率、未平倉量、多空比、ETF 等）不可能被挑到——這是資料長度的限制，不代表它們沒用。",
        "第 4 部分只照第一名做規則，考試只用 2019 年起的起點；其他資產（ETH、SOL、BNB、XRP、MSTR）用自己的 50 日均線和"
        "比特幣當天的變數讀數當作整個市場的狀態；24 個收盤時間和 MSTR 用前一天的讀數。",
        "推測：如果某個變數只在一段時期有效（例如 ETF 上市前後），全期平均會把它沖淡；這裡沒有逐期切開看，只切了 2019 年。",
        "限制：Coin Metrics 交易所地址是今天的標籤往回套，越早越不準；MVRV、NFCI 事後會修正；資金費率、未平倉量、多空比、"
        "成交量只有幣安，基差只有 Deribit；空頭／多頭徵兆數本身是從價格轉折挑出來的；以太幣自己的下車／上車這裡沒有看。",
        f"這次資料下載約 {data['secs'] / 60:.1f} 分鐘，計算約 {r['secs'] / 60:.1f} 分鐘"
        f"（格子 {r['secs_cells'] / 60:.1f} 分鐘，考試與隨機錯位 {r['secs_exam'] / 60:.1f} 分鐘）。",
        "就算有規則達標，也只代表可以放進每日影子紀錄觀察；過去的結果不保證未來，這裡不是買賣建議。",
        "第一次執行後的修正（只改單位和顯示，定義與門檻都沒動）：最深跌幅本來就是百分比，第一次被再乘 100 顯示，"
        "XR 的「最多差 5 點」也誤寫成 0.05 點；已改回事先寫好的 5 個百分點。這個修正沒有改變任何一條規則的判定。",
    ]
    if data["dropped"]:
        notes.append("這次沒有組出來的序列：" + "、".join(f"{k}（{v}）" for k, v in data["dropped"].items()) + "。")
    notes += problems

    nfci_src = next(((k, v) for k, v in (data.get("src") or {}).items() if k.startswith("NFCI")),
                    ("NFCI（芝加哥聯準銀行）", "https://api.data.chicagofed.org/NFCI/nfci-data-series-csv.csv"))
    sources = [
        {"name": "blockchain.info 長期每日價格（2017-08 以前）", "url": "https://api.blockchain.info/charts/market-price"},
        {"name": "幣安現貨 K 線（BTCUSDT 日線、1 小時線；ETH SOL BNB XRP 日線）", "url": "https://api.binance.com/api/v3/klines"},
        {"name": "Coin Metrics 社群 API（FlowInExNtv、FlowOutExNtv、SplyExNtv、CapMVRVCur、AdrActCnt、FeeTotNtv、PriceUSD）",
         "url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"},
        {"name": "Open Bitcoin Metrics：放 155 天以上的老幣被花掉（CSV 資料檔）",
         "url": "https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/obm_spent_value_ge155d_btc_daily/obm_spent_value_ge155d_btc_daily.csv"},
        {"name": "DefiLlama 穩定幣總量", "url": "https://stablecoins.llama.fi/stablecoincharts/all"},
        {"name": "Farside 美國現貨比特幣 ETF 每日流量", "url": "https://farside.co.uk/bitcoin-etf-flow-all-data/"},
        {"name": "Coinbase Exchange 日線 BTC-USD", "url": "https://api.exchange.coinbase.com/products/BTC-USD/candles"},
        {"name": "Bitstamp BTC/USD 日線", "url": "https://www.bitstamp.net/api/v2/ohlc/btcusd/"},
        {"name": "Upbit 日線（韓國溢價）", "url": "https://api.upbit.com/v1/candles/days"},
        {"name": "聯準會 H.10 匯率（美元兌韓元）", "url": "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&filetype=zip"},
        {"name": "幣安 U 本位合約資金費率", "url": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT"},
        {"name": "幣安合約每日資料存檔（未平倉量、多空比）",
         "url": "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip"},
        {"name": "Deribit 季度期貨小時線（基差）", "url": "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"},
        {"name": "維基媒體瀏覽量 API（英文 Bitcoin 頁）",
         "url": "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Bitcoin/daily"},
        {"name": nfci_src[0], "url": nfci_src[1]},
        {"name": "Yahoo Finance 日線（HYG、IEF、^VIX、DX-Y.NYB、MSTR）", "url": "https://query1.finance.yahoo.com/v8/finance/chart/"},
    ]
    return {
        "id": ID,
        "title": "在訊號那一刻看：50 日均線下車和上車時，其他變數分得出好壞嗎",
        "question": ("50 日均線每次下車（收盤跌破）和上車（收盤站上）都是當下就知道的事件。把過去量到有作用的變數——"
                     "交易所活躍度、老幣移動、手續費、買賣流向、Coinbase 和韓國溢價、資金費率、未平倉量、基差、多空比、"
                     "維基瀏覽量、金融環境、估值、空頭／多頭徵兆、波動——放到每一次下車和上車上，"
                     "能不能分出假跌破和真的下跌、大漲和沒漲起來？先觀察，再看照著做會不會比 50 日均線好。"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
