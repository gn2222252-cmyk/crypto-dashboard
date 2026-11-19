"""Historical study "reversed_direction": ten results that came out OPPOSITE to their study's hypothesis, each tested
again in the direction that was actually MEASURED, with a design that can fail.

Why (stated plainly). An audit of about 100 abandoned results (dashboard/cache/_audit_part1.json, _audit_part2.json,
their "reversed" lists) found that whenever a study measured an effect opposite to its hypothesis, it was filed as
"fail" and dropped. A result opposite to the guess is still a result. None of the ten below was ever tested in its
measured direction. All ten were NOTICED on the full history, so re-scoring the same history proves nothing; the
design below therefore relies on halves of each sample, matching on more price history, a placebo, another coin, and
- for rules - a direction fixed on the first half and scored only on the second half.

The ten reversed findings (parent study in dashboard/studies2/; each parent's loaders and exact variable definitions
are reused; the parent's stored number is reproduced first):
  R1 reluctant_sellers: heavy BTC deposits to exchanges (30-day FlowInExNtv above its trailing-year 80th percentile,
     data day t-1) -> BETTER 30-day returns than quiet (below the trailing-year median). Parent: quiet minus heavy
     -12.5 points (Test D pooled); narrow version matched on past 30-day return -9.4.
  R2 old_coin_selling / activity_state: heavy movement of coins held 155+ days (OLDSELL rank > 0.80, data day t-7)
     -> BETTER 60-day returns than quiet (rank < 0.50). Parent: quiet minus heavy -15.2 (pooled); activity_state M1
     matched heavy minus quiet +20.1.
  R3 pressure_with_trend: pattern "selling pressure down and buying pressure up" (pressure_before_moves' composites,
     30 days versus the 60 before) -> WORSE 60-day returns than the other three patterns in the same ma50 state.
     Parent (30-stratum matched): -30.8 with ma50 on (R3a), -12.4 with ma50 off (R3b).
  R4 after_fall_forces: ma50 off and close <= 0.75 x the highest close of the previous 365 days -> WORSE 60-day returns
     than all days. Parent: -15.9.
  R5 oi_flush: open-interest build-up with flat price (episode starts) -> +10.2 points over 30 days versus all days
     (R5a, measured "better"); a flush (episode starts) -> -3.0 over 14 days (R5b, measured "worse").
  R6 coinbase_premium: Coinbase premium (Coinbase BTC-USD / Binance BTCUSDT - 1) in its trailing-year bottom 5% ->
     -9.3 points over 30 days versus all days (measured "worse").
  R7 attention: English Wikipedia "Bitcoin" page views in their trailing-2-year top 5% (views to t-1) -> +27.8
     over 90 days versus all days (R7, full sample, "better"); -12.2 from 2023 (R7b, "worse").
  R8 order_flow: price up over 30 days while cumulative taker buy-minus-sell volume fell over 30 days -> +7.6% versus
     +2.9% when the delta also rose (30 days; measured "better").
  R9 funding_warning: Binance funding turning "hot" (7-day average above its trailing-year 90th percentile; episode
     starts) -> 30-day return no worse than all days; 2022-now +7.4% versus +2.1% (measured "better").
  R10 activity_state: from 2022, total on-chain spending (M3, R10a) and turnover (M5, R10b) HEAVY versus QUIET ->
     WORSE 60-day returns (-14.7, -15.3, 30-cell matched).
  Reproduced = this run's value within max(2 points, 15% of the stored value) of every stored number of that R. An R
  that is not reproduced is said so and dropped from Parts 1-4.

Common measurement (fixed before running; all forward returns from the decision close t: close[t+h]/close[t]-1).
  Each R is a daily label on decision dates: 1 = signal on, 0 = comparison group, 2 = neither (defined), undefined =
  missing; plus a population (price / date restriction that stays in place) and a horizon h. Comparison = all defined
  days ("vs all days": R4 R5 R6 R7 R9) or the parent's comparison group (R1 R2 quiet, R3 the other patterns, R8 price up
  and delta up, R10 quiet). Effect = mean(on) - mean(comparison) ("on minus comparison"; > 0 = followed by better).
  Sample S = population days with the label defined, a full h-day window, and the three matching variables.
  Matching variables: past-30-day return close[t]/close[t-30]-1, past-180-day return close[t]/close[t-180]-1, 20-day
  volatility (std of daily log returns over the 20 days ending at t). Edges from S. Matched effect = per cell
  mean(on) - mean(comparison), weighted by on-days per cell, cells holding both only. Two versions: M30 = 30 cells
  (past-30 decile x volatility tercile, the parents' matching) and M180 = 90 cells (past-30 decile x past-180 tercile x
  volatility tercile, NEW: several of these may be "price had risen for months").
  90% interval: circular block bootstrap over the S days in date order (60-day blocks, 1,000 draws, fixed seed).
  Periods: the parent's three periods (none for R5 and R8: halves only). Halves: S split at its middle day; M180 in
  each half. Placebo: the label series (from its first defined day) circularly shifted by k in [180, n-180] days, 200
  draws, population and cell edges kept; share of draws whose M180 is at least as far from zero in the measured
  direction as the real one.
Part 1 reading (fixed): HOLDS = same sign of M180 in both halves AND the M180 interval excludes zero AND placebo <= 5%;
  PARTLY = same sign in both halves but interval includes zero or placebo above 5%; NOT REPEATED = the sign differs
  between halves (or a half has no signal day). For each R: M30 versus M180 (what the 180-day match did) and the mean
  past-180-day return of on-days versus comparison days.
Part 2 - ETH, its own price (Binance ETHUSDT) and its own series where the data exist: R1 Coin Metrics eth
  FlowInExNtv; R4 price only; R5 Binance ETHUSDT open interest (data.binance.vision metrics); R6 Coinbase ETH-USD
  versus Binance ETHUSDT; R7 English Wikipedia "Ethereum"; R8 Binance ETHUSDT taker volume; R9 Binance ETHUSDT
  funding. R2, R3, R10 have no ETH series in the project. The same measurement (raw, M180 with interval); "same sign"
  = ETH's M180 has the sign of BTC's measured direction.
Part 3 - a rule in the measured direction, honest split. For every R kept: the middle day of its Part 1 sample S
  splits it; the DIRECTION is the sign of the M180 effect on the FIRST half only (the code computes it); the threshold
  is the parent's own. Rules (same form for all, nothing tuned): "followed by better" -> B = hold when ma50 is on OR
  (signal on AND close above its 20-day average); "followed by worse" -> W = hold when ma50 is on AND NOT signal, and
  the softer W50 = ma50 on: 100%, 50% when the signal is on. The signal used by a rule is the parent's DAILY state
  (R5a: build-up flag days; R5b: a flush within the last 14 days; R9: hot days; the others: their label = 1); missing
  = off. A rule that cannot change ma50 by construction is not scored (W when the R was measured on ma50-off days -
  R3b, R4 - and B when measured on ma50-on days - R3a).
  Fair exam = main_signal_check's four checks (reluctant_sellers' _Exam wiring): BTC daily monthly starts from the
  first month after the split to 12 months before today (later than main_signal_check's 2023-12, because these
  second halves are late; ma50 is re-scored on exactly the same starts), textbook 0.10%; 1-hour clock starts from
  max(split, 2018-01) to the same last start x 24 closing hours, textbook and ordinary person (act 10 hours later,
  0.25%, 10% of entries missed, 200 simulations per clock, main_signal_check's seeds); ETH SOL BNB XRP (Binance) and
  MSTR (Yahoo) with their own ma50 / 20-day average and BTC's signal as a market-wide state, starts from the split
  to the same last start. Number of starts reported; a rule with fewer than 24 BTC daily starts is dropped and named.
  Ranking score = worst of the four "beats holding" shares (main_signal_check's eligibility); median deepest loss =
  median over the clock textbook cells. Placebo: the rule's daily state circularly shifted by k in [180, n-180] days
  (from its first defined day), 200 draws (same k for W and W50), 20 ordinary-person simulations per clock, the real
  rule re-scored the same way; share of draws whose ranking score is at least the real one's.
  "Supported": B - ranking >= ma50's AND median deepest loss no more than 5 points worse AND placebo <= 5%; W / W50 -
  ranking >= ma50's AND median deepest loss no worse AND placebo <= 5%. Holm at 5% across all rules scored with
  p = (hits + 1) / (200 + 1); the number expected to pass the 5% placebo bar by luck = rules x 0.05.
Part 4 - one story? Correlation of the daily states (pairwise over overlapping defined days). Counts: on each day,
  how many "better" signals are on (R1, R2, R5a build-up, R7 before 2023, R8) and how many "worse" ones (R3 pattern
  in either ma50 state, R4, R5b, R6, R7 from 2023, R10a, R10b); missing = off; sample = days on which at least three
  of that list are defined; forward 60-day return per count 0 / 1 / 2 / 3+, and M180 of each count versus count 0
  with its interval. Any reading is a guess (推測). The audit's guess examined, not assumed: "busy participation is
  followed by better returns; crowded positioning and quiet 'textbook bullish' set-ups by worse".
Part 5 - what the data cannot tell, per R (labels applied backwards, one venue, short history, definition changes,
  the 2022-24 break when ETFs and custodians moved flows off-chain).
Today: "比特幣：50 日均線｜反向訊號" with ma50's state and how many "better" and "worse" signals are on (description).

Data: free key-less endpoints only, fixed user agent "btc-research-dashboard", through the parents' loaders; every
loader writes into this study's own folder dashboard/cache/_store/reversed_direction/ (seeded from the shared store
when newer); the shared store is only read. OBM data files are read from old_coin_selling's stored copy (read only;
that repository's code is never cloned or run). Seeds fixed. Not advice.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import activity_state as asx
from dashboard.studies2 import after_fall_forces as aff
from dashboard.studies2 import attention as att
from dashboard.studies2 import carry as cy
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import coinbase_premium as cp
from dashboard.studies2 import force_balance as fb
from dashboard.studies2 import funding_warning as fw
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import oi_flush as ofl
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import order_flow as of
from dashboard.studies2 import pressure_before_moves as pb
from dashboard.studies2 import pressure_with_trend as pw
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import reluctant_sellers as rsl
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

ID = "reversed_direction"
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
DAY = pd.Timedelta(days=1)
BLOCK, BOOT, DRAWS, SHIFT_MIN, PL_SIMS = 60, 1000, 200, 180, 20
PLACEBO_BAR, LOSS_SLACK, MIN_STARTS, ALPHA = 0.05, 5.0, 24, 0.05
TOL_PT, TOL_REL = 0.02, 0.15
CLOCK0 = pd.Timestamp(msc.CLOCK_STARTS[0])
H_ALL = (14, 30, 60, 90)

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID
SEED_FILES = ("reluctant_sellers_cm.json", "activity_state_cm_btc.json", "activity_state_cm_eth.json",
              "pressure_with_trend_cm.json", "pressure_with_trend_llama.json", "pressure_with_trend_obm.json",
              "pressure_before_moves_cm.json", "pressure_before_moves_llama.json", "pressure_before_moves_obm.json",
              "force_balance_cm.json", "force_balance_llama.json", "force_balance_etf.json",
              "price_impact_1h_BTCUSDT.csv", "kimchi_premium_coinbase.json", "coinbase_premium_cb_eth-usd.json",
              "coinbase_premium_bitstamp_btcusd.json", "ls_ratio_btcusdt.json", "ls_ratio_ethusdt.json",
              "attention_btc.json", "attention_eth.json", "order_flow_spot_1d_BTCUSDT.csv",
              "order_flow_spot_1d_ETHUSDT.csv", "etf_loop_btc_frame.json", "old_coin_selling_obm.json",
              "supply_demand_cm.json", "supply_demand_llama.json")
REMAP = ((km, "STORE_UP"), (km, "STORE_CB"), (km, "STORE_FX"), (pi, "STORE"), (fb, "STORE_CM"), (fb, "STORE_LL"),
         (fb, "STORE_ETF"), (pb, "STORE_CM"), (pb, "STORE_LL"), (pb, "STORE_OBM"), (pw, "STORE_CM"),
         (pw, "STORE_LL"), (pw, "STORE_OBM"), (rsl, "STORE_CM"), (ocs, "STORE_OBM"), (ocs, "STORE_CDD"),
         (fw, "STORE_PATH"), (sd, "STORE_CM"), (sd, "STORE_LL"), (sd, "BS_STORE"))
FUND_FALLBACK = {"BTCUSDT": REAL_STORE / "funding_warning_binance.json"}

KEYS = ("R1", "R2", "R3a", "R3b", "R4", "R5a", "R5b", "R6", "R7", "R7b", "R8", "R9", "R10a", "R10b")
BETTER = ("R1", "R2", "R5a", "R7pre", "R8")
WORSE = ("R3", "R4", "R5b", "R6", "R7post", "R10a", "R10b")
P_RSL = rsl.PERIODS
P_CB = (("2017-08–2019", "2017-08-17", "2019-12-31"), ("2020–2022", "2020-01-01", "2022-12-31"),
        ("2023 起", "2023-01-01", "2100-01-01"))
P_ATT = (("2015-07 到 2019", "2015-07-01", "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"))
P_ATT_B = (("2023 到今天", "2023-01-01", "2100-01-01"),)
P_FW = (("2020–21", "2020-01-01", "2021-12-31"), ("2022–今", "2022-01-01", "2100-01-01"))
P_R10 = (("2022 起", "2022-01-01", "2100-01-01"),)

# key: (parent, short plain name of the signal, comparison group, horizon, measured direction (+1 better / -1 worse))
META = {
    "R1": ("reluctant_sellers", "大量比特幣轉進交易所（近 30 天轉進量高於過去一年第 80 百分位）",
           "轉進量少的日子（低於過去一年中位數）", 30, 1),
    "R2": ("old_coin_selling／activity_state", "放了 155 天以上的老幣大量移動（30 天合計在過去一年排前 20%）",
           "老幣安靜的日子（排在後 50%）", 60, 1),
    "R3a": ("pressure_with_trend", "「賣壓降＋買壓升」（課本上的看漲型態），50 日均線開著",
            "均線同樣開著的其他三種型態", 60, -1),
    "R3b": ("pressure_with_trend", "「賣壓降＋買壓升」，50 日均線關著", "均線同樣關著的其他三種型態", 60, -1),
    "R4": ("after_fall_forces", "50 日均線關、收盤比過去一年最高收盤低 25% 以上", "所有日子", 60, -1),
    "R5a": ("oi_flush", "合約未平倉量堆高、價格不動（開始那天）", "所有日子", 30, 1),
    "R5b": ("oi_flush", "清槓桿：未平倉量 3 天大跌、價格也跌（開始那天）", "所有日子", 14, -1),
    "R6": ("coinbase_premium", "Coinbase 價格比幣安低很多（折價落在過去一年最低 5%）", "所有日子", 30, -1),
    "R7": ("attention", "維基百科「Bitcoin」瀏覽量爆量（過去兩年最高 5%），全期", "所有日子", 90, 1),
    "R7b": ("attention", "維基百科瀏覽量爆量，只看 2023 起", "所有日子", 90, -1),
    "R8": ("order_flow", "價格 30 天漲、但主動買減主動賣的累積量 30 天在減（背離）", "價格漲、買賣差也增的日子", 30, 1),
    "R9": ("funding_warning", "永續合約資金費率轉為偏熱（7 天平均高於過去一年第 90 百分位，開始那天）", "所有日子", 30, 1),
    "R10a": ("activity_state", "鏈上花費總量大（M3，30 天合計排前 20%），2022 起", "安靜的日子（後 50%）", 60, -1),
    "R10b": ("activity_state", "周轉率高（M5＝M3 除以總供給），2022 起", "安靜的日子（後 50%）", 60, -1),
}
STATE_NOTE = {"R5a": "規則用每天的「堆高」旗標（不是只用開始那天）", "R5b": "規則用「過去 14 天內有清槓桿」",
              "R9": "規則用每天的「偏熱」旗標（不是只用開始那天）"}
LIMITS = {
    "R1": "交易所地址標籤是今天的標籤套回過去，早年轉進量被低估；送進交易所不等於賣（也可能是借貸、做市、換錢包）。"
          "2022–24 年 ETF 和託管機構把大量買賣移到鏈下，之後的轉進量和以前不是同一種東西。資料分不出「熱鬧（參與者多）」和「真的在賣」。",
    "R2": "老幣移動不等於賣出（託管搬家、交易所整理錢包、Mt. Gox 還款都算）；只有一份資料來源（Open Bitcoin Metrics）。"
          "分不出是老手在賣給新買家，還是單純搬家。",
    "R3a": "賣壓、買壓是好幾條鏈上和交易所資料拼成的分數，2017-08 以前買方只有「提出交易所」一項，組成隨時間改變；"
           "24 格裡挑出來的一格，本來就可能靠運氣。分不出「型態本身不好」和「均線開著時漲多了」。",
    "R3b": "同 R3a；均線關著的日子本來就少、又集中在幾次熊市。",
    "R4": "只用價格，資料沒有標籤問題；但 2012 年以來只有十幾次大跌，許多天是同一次大跌裡的連續日子，"
          "不是獨立的幾千次。分不出「跌深的再跌」和「熊市本來就長」。",
    "R5a": "未平倉量只有幣安一家、2020-09 起（扣掉一年暖機只剩約五年），堆高的「開始那天」只有十次左右；"
           "分不出「槓桿多＝擁擠」還是「槓桿多＝很多人真的想買」。",
    "R5b": "同 R5a；清槓桿約二十幾次，14 天很短。",
    "R6": "Coinbase 美元對幣安 USDT：極端折價常常是 USDT 本身脫鉤或交易所出事（2022–23），"
          "母研究換成美元對美元時這個效果幾乎消失（-0.3），所以量到的可能主要是穩定幣壓力，不是美國買盤。",
    "R7": "只有英文維基、2015-07 起；瀏覽量受搜尋引擎、AI 摘要和手機 App 改版影響，2023 起的下降可能有一部分是定義改變（大家不再點進維基）。",
    "R7b": "同 R7；2023 起只有不到四年、幾次爆量。",
    "R8": "只有幣安現貨；2022-07 到 2023-03 幣安零手續費，成交量被灌大；背離的定義（30 天）只是一種。",
    "R9": "只有幣安、2019-09 起；偏熱的「開始那天」只有二十幾次；資金費率跟著價格走（母研究量到價格領先），"
          "分不出「偏熱＝擁擠」和「偏熱＝趨勢強」。",
    "R10a": "鏈上花費包含交易所內部搬幣、找零；2022 起 ETF 和託管把買賣移到鏈下，鏈上活躍度的意義變了；只有約四年。",
    "R10b": "同 R10a；周轉率只是把 M3 除以總供給。",
}

_ok, _p, _pt, _share, _share1 = sd._ok, sd._p, sd._pt, sd._share, sd._share1
_d = cr._d


def _ci(lo, hi, f=_pt) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _num(v, nd: int = 2) -> str:
    return f"{float(v):+.{nd}f}" if _ok(v) else "—"


# ---------- own store ----------

@contextmanager
def _own_store():
    """Loaders write into dashboard/cache/_store/reversed_direction/; the shared store is only read (seeds copied in
    when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        src, dst = REAL_STORE / name, MY_STORE / name
        try:
            if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        except Exception:  # noqa: BLE001 - a missing seed only means the loader must download
            pass
    saved = [(m, a, getattr(m, a)) for m, a in REMAP]
    saved_as = asx.STORE_CM
    old = common.STORE
    common.STORE = MY_STORE
    for m, a, p in saved:
        setattr(m, a, MY_STORE / Path(p).name)
    asx.STORE_CM = {k: MY_STORE / Path(p).name for k, p in saved_as.items()}
    try:
        yield
    finally:
        common.STORE = old
        for m, a, p in saved:
            setattr(m, a, p)
        asx.STORE_CM = saved_as


def _funding(symbol: str) -> tuple[pd.Series, str | None]:
    path = MY_STORE / f"funding_{symbol.lower()}.json"
    try:
        s = cy._funding(symbol)
        try:
            path.write_text(json.dumps({str(int(t.value // 1_000_000)): float(v) for t, v in s.items()}),
                            encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return s, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        for p in (path, FUND_FALLBACK.get(symbol)):
            if p is not None and p.exists():
                got = json.loads(p.read_text(encoding="utf-8"))
                s = pd.Series({pd.Timestamp(int(k), unit="ms"): float(v) for k, v in got.items()}).sort_index()
                return s, f"幣安 {symbol} 資金費率這次取不到（{type(exc).__name__}），用之前存下來的資料。"
        raise


def _try(problems: list, label: str, fn):
    try:
        out = fn()
        if isinstance(out, tuple) and len(out) == 2 and (out[1] is None or isinstance(out[1], str)):
            if out[1]:
                problems.append(out[1])
            return out[0]
        return out
    except Exception as exc:  # noqa: BLE001 - one source failing drops only what needs it
        problems.append(f"{label}這次取不到（{type(exc).__name__}），用到它的部分沒有算。")
        return None


def _load() -> dict:
    P: list[str] = []
    d: dict = {"problems": P}
    d["close"] = te._daily_close()
    other, more = msc._load_other()
    P += more
    d["other"] = other
    eth = other["ETH"][0].astype(float) if "ETH" in other else None
    if eth is not None:
        eth = pd.Series(eth.values, index=pd.to_datetime(eth.index)).sort_index()
        eth = eth[~eth.index.duplicated(keep="last")].asfreq("D").ffill()
    d["eth"] = eth
    d["cm_rsl"] = _try(P, "Coin Metrics 比特幣轉進交易所（R1）", lambda: sd._stored(rsl.STORE_CM, sd._cm_fetch))
    for asset in ("btc", "eth"):
        d[f"cm_{asset}"] = _try(P, f"Coin Metrics {asset}（R1／R10）",
                                lambda a=asset: sd._stored(asx.STORE_CM[a], asx._cm_fetcher(a)))
    try:
        d["obm"] = json.loads((REAL_STORE / "old_coin_selling_obm.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        d["obm"] = None
        P.append(f"old_coin_selling 存下來的 OBM 資料讀不到（{type(exc).__name__}），R2、R10 沒有算。")
    d["pw"] = _try(P, "買賣壓分數的資料（R3）", pw._load)
    if d["pw"] is not None:
        P += d["pw"].get("problems", [])
    for sym in ("BTCUSDT", "ETHUSDT"):
        d[f"oi_{sym}"] = _try(P, f"幣安 {sym} 未平倉量（R5）", lambda s=sym: lr._fetch(s, budget_s=240))
        d[f"bars_{sym}"] = _try(P, f"幣安 {sym} 日線成交量（R8）", lambda s=sym: of._daily("spot", s))
        d[f"fund_{sym}"] = _try(P, f"幣安 {sym} 資金費率（R9）", lambda s=sym: _funding(s))
    d["cb"] = _try(P, "Coinbase BTC-USD（R6）", km._coinbase)
    d["cb_eth"] = _try(P, "Coinbase ETH-USD（R6 的 ETH）", lambda: cp._candles_cb("ETH-USD", "2017-08-01"))
    d["views_btc"] = _try(P, "維基百科「Bitcoin」瀏覽量（R7）", lambda: att._views("btc"))
    d["views_eth"] = _try(P, "維基百科「Ethereum」瀏覽量（R7 的 ETH）", lambda: att._views("eth"))
    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        P.append(problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    d["grid"] = (price, b0, cr._prehistory(b0))
    return d


# ---------- features and the common measurement ----------

def _feats(close: pd.Series) -> dict:
    c = close.values.astype(float)
    n = len(c)
    fwd = {}
    for h in H_ALL:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        fwd[h] = f
    p30, p180 = np.full(n, np.nan), np.full(n, np.nan)
    p30[30:] = c[30:] / c[:-30] - 1
    p180[180:] = c[180:] / c[:-180] - 1
    vol = np.log(close).diff().rolling(20, min_periods=20).std(ddof=1).values
    return {"idx": close.index, "fwd": fwd, "p30": p30, "p180": p180, "vol": vol,
            "base": np.isfinite(p30) & np.isfinite(p180) & np.isfinite(vol)}


def _cells(F: dict, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """90-cell (past-30 decile x past-180 tercile x vol tercile) and 30-cell codes with edges from S; -1 elsewhere."""
    ok = F["base"]
    e30 = np.quantile(F["p30"][S], np.linspace(0, 1, 11))
    e180 = np.quantile(F["p180"][S], [1 / 3, 2 / 3])
    ev = np.quantile(F["vol"][S], [1 / 3, 2 / 3])
    dec = np.clip(np.searchsorted(e30, np.where(ok, F["p30"], 0.0), side="right") - 1, 0, 9)
    t180 = np.searchsorted(e180, np.where(ok, F["p180"], 0.0), side="right")
    tv = np.searchsorted(ev, np.where(ok, F["vol"], 0.0), side="right")
    return np.where(ok, dec * 9 + t180 * 3 + tv, -1), np.where(ok, dec * 3 + tv, -1)


def _md(v, on, ref, cell, nc) -> float:
    a, b = on & (cell >= 0), ref & (cell >= 0)
    na = np.bincount(cell[a], minlength=nc)
    nb = np.bincount(cell[b], minlength=nc)
    sa = np.bincount(cell[a], weights=v[a], minlength=nc)
    sb = np.bincount(cell[b], weights=v[b], minlength=nc)
    use = (na > 0) & (nb > 0)
    if not use.any():
        return float("nan")
    return float((na[use] * (sa[use] / na[use] - sb[use] / nb[use])).sum() / na[use].sum())


def _md_boot(v, on, ref, cell, nc) -> np.ndarray:
    """Rows = bootstrap draws (2-D inputs)."""
    rows = v.shape[0]
    key = np.where(cell >= 0, cell, 0) + nc * np.arange(rows)[:, None]
    a, b = on & (cell >= 0), ref & (cell >= 0)
    size = nc * rows
    na = np.bincount(key[a], minlength=size).reshape(rows, nc)
    nb = np.bincount(key[b], minlength=size).reshape(rows, nc)
    sa = np.bincount(key[a], weights=v[a], minlength=size).reshape(rows, nc)
    sb = np.bincount(key[b], weights=v[b], minlength=size).reshape(rows, nc)
    use = (na > 0) & (nb > 0)
    with np.errstate(all="ignore"):
        dd = np.where(use, sa / np.maximum(na, 1) - sb / np.maximum(nb, 1), 0.0)
        w = np.where(use, na, 0)
        return np.where(w.sum(axis=1) > 0, (w * dd).sum(axis=1) / np.maximum(w.sum(axis=1), 1), np.nan)


def _q(a) -> tuple:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)


def _masks(lab: np.ndarray, mode: str, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    on = S & (lab == 1)
    ref = S & (np.isfinite(lab) if mode == "all" else (lab == 0))
    return on, ref


def _raw(v, on, ref) -> float:
    return float(v[on].mean() - v[ref].mean()) if on.any() and ref.any() else float("nan")


def _measure(F: dict, lab: np.ndarray, mode: str, h: int, pop: np.ndarray, periods, seed: int,
             placebo: bool = True, boot: bool = True) -> dict | None:
    idx = F["idx"]
    v_all = F["fwd"][h]
    S = pop & np.isfinite(lab) & np.isfinite(v_all) & F["base"]
    if S.sum() < 2 * BLOCK:
        return None
    on, ref = _masks(lab, mode, S)
    if on.sum() < 2 or ref.sum() < 2:
        return None
    c3, c2 = _cells(F, S)
    v = np.nan_to_num(v_all)
    out = {"n": int(S.sum()), "n_on": int(on.sum()), "n_ref": int(ref.sum()), "start": idx[S][0], "end": idx[S][-1],
           "mean_on": float(v[on].mean()), "mean_ref": float(v[ref].mean()), "raw": _raw(v, on, ref),
           "m3": _md(v, on, ref, c3, 90), "m2": _md(v, on, ref, c2, 30),
           "p180_on": float(F["p180"][on].mean()), "p180_ref": float(F["p180"][ref].mean()),
           "p30_on": float(F["p30"][on].mean()), "p30_ref": float(F["p30"][ref].mean())}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(S)
    if boot:
        m = len(pos)
        nb = -(-m // BLOCK)
        st = rng.integers(0, m, (BOOT, nb))
        bi = pos[((st[:, :, None] + np.arange(BLOCK)) % m).reshape(BOOT, -1)[:, :m]]
        vb, onb, refb = v[bi], on[bi], ref[bi]
        out["ci3"] = _q(_md_boot(vb, onb, refb, c3[bi], 90))
        out["ci2"] = _q(_md_boot(vb, onb, refb, c2[bi], 30))
        with np.errstate(all="ignore"):
            rb = (np.where(onb, vb, 0).sum(1) / onb.sum(1)) - (np.where(refb, vb, 0).sum(1) / refb.sum(1))
        out["ciraw"] = _q(rb)
    # halves
    cut = idx[pos[len(pos) // 2]]
    out["cut"] = cut
    out["halves"] = []
    for part in (idx < cut, idx >= cut):
        o, r = on & part, ref & part
        out["halves"].append({"n_on": int(o.sum()), "raw": _raw(v, o, r), "m3": _md(v, o, r, c3, 90),
                              "span": (idx[S & part][0], idx[S & part][-1]) if (S & part).any() else (None, None)})
    out["per"] = []
    for lab_p, a, b in periods:
        part = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        o, r = on & part, ref & part
        if not (S & part).any():
            continue
        out["per"].append({"lab": lab_p, "n": int((S & part).sum()), "n_on": int(o.sum()), "raw": _raw(v, o, r),
                           "m3": _md(v, o, r, c3, 90)})
    if placebo and _ok(out["m3"]):
        first = int(np.argmax(np.isfinite(lab)))
        seg = lab[first:]
        n = len(seg)
        prng = np.random.default_rng(seed + 7)
        real, sgn = out["m3"], (1.0 if out["m3"] > 0 else -1.0)
        got = []
        for k in prng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, n - SHIFT_MIN), DRAWS):
            l2 = lab.copy()
            l2[first:] = np.roll(seg, int(k))
            S2 = pop & np.isfinite(l2) & np.isfinite(v_all) & F["base"]
            o2, r2 = _masks(l2, mode, S2)
            got.append(_md(v, o2, r2, c3, 90))
        got = np.array(got, dtype=float)
        out["placebo"] = float(np.mean(np.where(np.isfinite(got), sgn * got >= sgn * real - 1e-12, False)))
        out["placebo_med"] = float(np.nanmedian(got)) if np.isfinite(got).any() else None
    else:
        out["placebo"] = None
    return out


def _classify(r: dict | None) -> str:
    if not r:
        return "算不出"
    a, b = r["halves"][0]["m3"], r["halves"][1]["m3"]
    if not (_ok(a) and _ok(b)) or a * b <= 0:
        return "沒有重複"
    lo, hi = r.get("ci3", (None, None))
    excl = _ok(lo) and _ok(hi) and (lo > 0 or hi < 0)
    plc = _ok(r.get("placebo")) and r["placebo"] <= PLACEBO_BAR
    return "站得住" if excl and plc else "部分"


# ---------- labels from the parents' definitions ----------

def _code(on, ref, defined) -> np.ndarray:
    """1 on, 0 comparison, 2 neither, NaN undefined."""
    lab = np.where(np.asarray(on, dtype=bool), 1.0, np.where(np.asarray(ref, dtype=bool), 0.0, 2.0))
    return np.where(np.asarray(defined, dtype=bool), lab, np.nan)


def _close_tol(stored: float, got) -> bool:
    return _ok(got) and abs(got - stored) <= max(TOL_PT, TOL_REL * abs(stored)) + 1e-12


def _cm_frame(raw: dict) -> pd.DataFrame:
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    return cm[cm.index < pd.Timestamp(common.today_utc())].asfreq("D")


def _build(d: dict) -> dict:
    """Per R: lab, mode, h, pop, periods, state (daily rule state on BTC decision dates), repro rows, ETH spec."""
    close = d["close"]
    idx = close.index
    n = len(idx)
    eth = d["eth"]
    ma = np.nan_to_num(msc._targets(close)["ma50"].values.astype(float))
    R: dict = {}
    fails: dict = {}

    def add(key, lab, mode, pop, periods, state, repro, eth_spec=None):
        R[key] = {"lab": np.asarray(lab, dtype=float), "mode": mode, "h": META[key][3], "pop": np.asarray(pop, bool),
                  "periods": periods, "state": state, "repro": repro, "eth": eth_spec}

    every = np.ones(n, dtype=bool)

    # R1
    try:
        cm = _cm_frame(d["cm_rsl"])
        raw = rsl._raw(cm)
        lg = rsl._legs(raw, idx)
        fl = rsl._flags(close, lg)
        dd = rsl._test_d(close, fl)
        aa = rsl._test_a(close, fl)
        rep = [("少減多：之後 30 天，價格在上下兩邊平均（測試 D）", -0.125, dd["per"]["全期"]["pool"]),
               ("較窄舊版：惜售減出貨，配對過去 30 天後（測試 A）", -0.094, aa["matched"][30][0])]
        defined = lg["low"].notna().values
        lab = _code(fl["high"].values > 0.5, fl["quiet"].values > 0.5, defined)
        state = pd.Series(np.where(defined, fl["high"].values, np.nan), index=idx)
        espec = None
        if eth is not None and d.get("cm_eth") is not None:
            ce = _cm_frame(d["cm_eth"])
            er = rsl._raw(ce)
            el = rsl._legs(er, eth.index)
            edef = el["low"].notna().values
            espec = {"lab": _code(el["high"].values > 0.5, el["low"].values > 0.5, edef),
                     "pop": np.ones(len(eth), bool), "data": "Coin Metrics eth FlowInExNtv（轉進交易所的以太幣）"}
        add("R1", lab, "rest", every, P_RSL, state, rep, espec)
    except Exception as exc:  # noqa: BLE001
        fails["R1"] = type(exc).__name__

    # R2 and R10
    try:
        obm = {k: ocs._series(d["obm"], k) for k in ("ge155", "lt155")}
        a20 = sd._above20(close.values)
        m = ocs._measure(obm["ge155"])
        rank_m1 = ocs._on(m["rank"], idx)
        ta_ = ocs._test_a(close, rank_m1, a20, SEED + 901, light=True)
        data_as = {"obm": obm, "cm": {}, "spot": {}}
        if d.get("cm_btc") is not None:
            data_as["cm"]["btc"] = asx._frame(d["cm_btc"])
        lv = asx._levels_data(data_as)
        rank_t = {k: asx._at(asx._rank(lv[k]), idx, asx.LAG[k]) for k in ("m1", "m3", "m5") if k in lv}
        Pz = asx._prep(close)
        p1 = asx._part1(Pz, rank_t["m1"], SEED + 300, light=True)
        rep = [("安靜減大量：之後 60 天，價格在上下兩邊平均（old_coin_selling）", -0.152, ta_["per"]["全期"]["pool"]),
               ("大量減安靜：配對過去 30 天漲幅×波動（activity_state M1）", 0.201, p1["m"][60][0])]
        lev = ocs._levels(rank_m1)
        defined = lev >= 0
        lab = _code(lev == 2, lev == 0, defined)
        state = pd.Series(np.where(defined, (lev == 2).astype(float), np.nan), index=idx)
        add("R2", lab, "rest", idx >= pd.Timestamp(ocs.SAMPLE_FROM), P_RSL, state, rep)
        for key, mk, seed, stored in (("R10a", "m3", 302, -0.147), ("R10b", "m5", 304, -0.153)):
            if mk not in rank_t:
                fails[key] = "沒有資料"
                continue
            p = asx._part1(Pz, rank_t[mk], SEED + seed, light=True)
            got = p["per"].get("2022 起", {}).get("m", (None,))[0]
            lv_ = ocs._levels(rank_t[mk])
            dfn = lv_ >= 0
            add(key, _code(lv_ == 2, lv_ == 0, dfn), "rest", idx >= pd.Timestamp("2022-01-01"), P_R10,
                pd.Series(np.where(dfn, (lv_ == 2).astype(float), np.nan), index=idx),
                [("大量減安靜：2022 起，配對過去 30 天漲幅×波動（activity_state）", stored, got)])
    except Exception as exc:  # noqa: BLE001
        for k in ("R2", "R10a", "R10b"):
            fails.setdefault(k, type(exc).__name__)

    # R3
    try:
        pdata = d["pw"]
        _raw_p, rank, _ = pb._pressures(pdata)
        pat = pw._pat_arr(pw._realtime(rank["SELL_C"].values), pw._realtime(rank["BUY_C"].values))
        St = pw._states(close)
        c = close.values.astype(float)
        fwp = pw._forward(c)
        past = np.full(n, np.nan)
        past[30:] = c[30:] / c[:-30] - 1
        vol = np.log(close).diff().rolling(20, min_periods=20).std(ddof=1).values
        base = (idx >= pw.FROM) & np.isfinite(fwp[60]) & np.isfinite(past) & np.isfinite(vol)
        dq = np.quantile(past[base], np.linspace(0, 1, 11))
        vq = np.quantile(vol[base], [0, 1 / 3, 2 / 3, 1])
        dec = np.clip(np.searchsorted(dq, np.nan_to_num(past), side="right") - 1, 0, 9)
        ter = np.clip(np.searchsorted(vq, np.nan_to_num(vol), side="right") - 1, 0, 2)
        strat = dec * 3 + ter
        days = np.flatnonzero(base)
        st = St["t1"].values[days]
        okst = np.isfinite(st)
        sv = np.where(okst, st, 0).astype(int)
        pp = pat[days]
        valid = okst & (pp < 4)
        skey = np.where(valid, (sv * 4 + np.minimum(pp, 3)) * pw.NSTRAT + strat[days], 8 * pw.NSTRAT)
        real = pw._mtab(skey, np.nan_to_num(fwp[60][days]))
        first = int(np.argmax(pat < 4))
        defined = np.arange(n) >= first
        lab = np.where(defined, np.where(pat == 0, 1.0, np.where(pat < 4, 0.0, 2.0)), np.nan)
        state = pd.Series(np.where(defined, (pat == 0).astype(float), np.nan), index=idx)
        t1 = St["t1"].values
        add("R3a", lab, "rest", (idx >= pw.FROM) & (t1 == 1), P_RSL, state,
            [("型態減其他型態，均線開著，配對 30 格", -0.308, real[4])])
        add("R3b", lab, "rest", (idx >= pw.FROM) & (t1 == 0), P_RSL, state,
            [("型態減其他型態，均線關著，配對 30 格", -0.124, real[0])])
    except Exception as exc:  # noqa: BLE001
        fails["R3a"] = fails["R3b"] = type(exc).__name__

    # R4
    try:
        f = aff._price(close)
        fwd60 = aff._forward(f["c"])[60]
        okm = np.isfinite(fwd60)
        got = float(fwd60[f["pop"] & okm].mean() - fwd60[f["allday"] & okm].mean())
        lab = np.where(f["allday"], f["pop"].astype(float), np.nan)
        state = pd.Series(lab, index=idx)
        espec = None
        if eth is not None:
            fe = aff._price(eth)
            espec = {"lab": np.where(fe["allday"], fe["pop"].astype(float), np.nan), "pop": np.ones(len(eth), bool),
                     "data": "只用以太幣價格"}
        add("R4", lab, "all", every, P_RSL, state, [("跌深的日子減所有日子，之後 60 天", -0.159, got)], espec)
    except Exception as exc:  # noqa: BLE001
        fails["R4"] = type(exc).__name__

    # R5
    try:
        rows_b = d["oi_BTCUSDT"]
        oi = ofl._oi(rows_b, idx[-1])
        flo = ofl._flags(oi, close)
        ev = ofl._events(flo, close, np.random.default_rng(SEED), ci=False)

        def r5(fl_, cl, kind, known):
            k = fl_[known].reindex(cl.index).fillna(0.0).values > 0.5
            stt = ofl._starts(fl_[kind].reindex(cl.index).fillna(0.0)).values > 0.5
            return np.where(k, stt.astype(float), np.nan)

        eflo = None
        if eth is not None and d.get("oi_ETHUSDT"):
            eflo = ofl._flags(ofl._oi(d["oi_ETHUSDT"], eth.index[-1]), eth)
        for key, kind, known, hh, stored, sk in (("R5a", "build", "kb", 30, 0.102, "build"),
                                                 ("R5b", "flush", "kf", 14, -0.030, "win")):
            kn = flo[known].reindex(idx).fillna(0.0).values > 0.5
            stv = flo[sk].reindex(idx).values
            state = pd.Series(np.where(kn, np.nan_to_num(stv), np.nan), index=idx)
            espec = None
            if eflo is not None:
                espec = {"lab": r5(eflo, eth, kind, known), "pop": np.ones(len(eth), bool),
                         "data": "幣安 ETHUSDT 合約未平倉量（data.binance.vision）"}
            add(key, r5(flo, close, kind, known), "all", every, (), state,
                [(f"開始那天減所有日子，之後 {hh} 天", stored, ev[kind][hh]["ex"])], espec)
    except Exception as exc:  # noqa: BLE001
        fails["R5a"] = fails["R5b"] = type(exc).__name__

    # R6
    try:
        days = pd.date_range(cp.FIRST, idx[-1], freq="D")
        cbs = cp._series(d["cb"], days)
        raw6 = cp._premium(cbs, close.reindex(days))
        f6 = cp._flags(raw6)
        fwd30 = (close.shift(-30) / close - 1).reindex(days)
        ok6 = (fwd30.notna() & f6["def_b"] & raw6.notna()).values
        got = km._excess(f6["bottom"].values[ok6].astype(bool), fwd30.values[ok6].astype(float))
        b6 = f6["bottom"].astype(float).where(f6["def_b"].astype(bool)).reindex(idx)
        espec = None
        if eth is not None and d.get("cb_eth"):
            ed = pd.date_range(cp.FIRST, eth.index[-1], freq="D")
            fe = cp._flags(cp._premium(cp._series(d["cb_eth"], ed), eth.reindex(ed)))
            be = fe["bottom"].astype(float).where(fe["def_b"].astype(bool)).reindex(eth.index)
            espec = {"lab": be.values.astype(float), "pop": np.ones(len(eth), bool),
                     "data": "Coinbase ETH-USD 對幣安 ETHUSDT"}
        add("R6", b6.values.astype(float), "all", every, P_CB, b6, [("極端折價減所有日子，之後 30 天", -0.093, got)],
            espec)
    except Exception as exc:  # noqa: BLE001
        fails["R6"] = type(exc).__name__

    # R7
    try:
        feat = att._features(d["views_btc"])
        df = att._frame(close, feat)
        fr = att._froth(df)
        fr23 = att._froth(att._subset(df, "2023-01-01", "2100-01-01"))
        sp = df["spike"].reindex(idx).values.astype(float)
        state = pd.Series(sp, index=idx)
        espec = None
        if eth is not None and d.get("views_eth") is not None:
            dfe = att._frame(eth, att._features(d["views_eth"]))
            espec = {"lab": dfe["spike"].reindex(eth.index).values.astype(float), "pop": np.ones(len(eth), bool),
                     "data": "英文維基「Ethereum」瀏覽量"}
        add("R7", sp, "all", every, P_ATT + P_ATT_B, state, [("爆量減所有日子，之後 90 天，全期", 0.278, fr["d90"][0])],
            espec)
        add("R7b", sp, "all", idx >= pd.Timestamp("2023-01-01"), P_ATT_B, state,
            [("爆量減所有日子，之後 90 天，2023 起", -0.122, fr23["d90"][0])], espec and dict(espec, pop=(
                eth.index >= pd.Timestamp("2023-01-01"))))
    except Exception as exc:  # noqa: BLE001
        fails["R7"] = fails["R7b"] = type(exc).__name__

    # R8
    try:
        a8 = of._asset(close, d["bars_BTCUSDT"], None)
        pr = a8["pred"]["div_bear"]
        f8 = a8["f"]
        defined = (f8["div_bear"].notna()).values
        lab = _code(f8["div_bear"].values == 1, f8["conf_up"].values == 1, defined)
        state = pd.Series(np.where(defined, f8["div_bear"].values, np.nan), index=idx)
        espec = None
        if eth is not None and d.get("bars_ETHUSDT") is not None:
            fe = of._features(eth, d["bars_ETHUSDT"], None)
            de = fe["div_bear"].notna().values
            espec = {"lab": _code(fe["div_bear"].values == 1, fe["conf_up"].values == 1, de),
                     "pop": eth.index >= pd.Timestamp(of.MAIN_START), "data": "幣安 ETHUSDT 現貨主動買賣量"}
        add("R8", lab, "rest", idx >= a8["start"], (), state,
            [("背離之後 30 天平均", 0.076, pr["high"]["f30"] / 100), ("價漲、買賣差也增之後 30 天平均", 0.029,
                                                               pr["low"]["f30"] / 100)], espec)
    except Exception as exc:  # noqa: BLE001
        fails["R8"] = type(exc).__name__

    # R9
    try:
        def hot_on(rate, cl):
            sig = fw._signal(rate)
            tns = rate.index.values.astype("datetime64[ns]").astype(np.int64)
            return pd.Series(fw._take(sig["hot"].values, fw._idx(tns, cl.index + DAY)), index=cl.index)

        def r9lab(hot):
            eps = set(fw._episodes(hot))
            st_ = np.array([1.0 if t in eps else 0.0 for t in hot.index])
            return np.where(hot.notna().values, st_, np.nan)

        hot_b = hot_on(d["fund_BTCUSDT"], close)
        evf = fw._events(close, hot_b, np.random.default_rng(SEED + 9))
        p22 = evf["periods"]["2022–今"]
        espec = None
        if eth is not None and d.get("fund_ETHUSDT") is not None:
            he = hot_on(d["fund_ETHUSDT"], eth)
            espec = {"lab": r9lab(he), "pop": np.ones(len(eth), bool), "data": "幣安 ETHUSDT 永續合約資金費率"}
        add("R9", r9lab(hot_b), "all", every, P_FW, hot_b,
            [("偏熱開始之後 30 天平均，2022 起", 0.074, p22["ep_r30"]), ("所有日子之後 30 天平均，2022 起", 0.021,
                                                             p22["all_r30"])], espec)
    except Exception as exc:  # noqa: BLE001
        fails["R9"] = type(exc).__name__
    return {"R": R, "fails": fails, "ma": ma}


# ---------- Part 3: the exam ----------

def _combine(kind: str, ma, a20, rma, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if kind == "ma50" or f is None:
        return ma
    f = np.asarray(f, dtype=float)
    if kind == "b":
        return np.maximum(ma, f * a20)
    if kind == "w":
        return ma * (1.0 - f)
    if kind == "w50":
        return ma * (1.0 - 0.5 * f)
    raise ValueError(kind)


class _Exam(rsl._Exam):
    """reluctant_sellers' exam (main_signal_check's four checks) with starts from `first` to `last` and this study's
    three ways to combine the signal (B, W, W50)."""

    def __init__(self, close, other, price, b0, pre, first: pd.Timestamp, last: pd.Timestamp):  # noqa: PLR0913
        self.close = close
        self.ma = msc._targets(close)["ma50"].values.astype(float)
        self.a20 = sd._above20(close.values)
        self.change = close.pct_change().fillna(0.0).values
        self.starts, self.labels = msc._month_starts(close.index, first, last)
        hr, hd = msc._suffix(msc._equity(self.change), self.starts, YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, max(first, CLOCK0), last)
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma, self.ca20 = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(np.nan_to_num(msc._targets(pd.Series(c))["ma50"].values[npre:][:self.days].astype(float)))
            self.ca20.append(sd._above20(c)[npre:][:self.days])
        self.other = {}
        for key, (c, since, year) in other.items():
            tg = msc._targets(c, year)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, pd.Timestamp(since or c.index[0]), first)
            starts, labels = msc._month_starts(c.index, begin, last)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "a20": sd._above20(c.values),
                               "change": change, "starts": starts, "labels": labels, "year": year,
                               "lag": 1 if since else 0, "hold": (hr[:, 0], hd[:, 0])}
        self.rma = rsl._recent(self.ma)
        self.crma = [rsl._recent(x) for x in self.cma]
        for o in self.other.values():
            o["rma"] = rsl._recent(o["ma"])

    def btc_pos(self, flag, kind: str = "ma50") -> np.ndarray:
        return _combine(kind, self.ma, self.a20, self.rma, self._at(flag, self.close.index, 0))

    def summary(self, flag, ki: int, sims: int = SIMS, kind: str = "ma50") -> dict:
        pos = self.btc_pos(flag, kind)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        f = self._at(flag, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = _combine(kind, self.cma[h], self.ca20[h], self.crma[h], f)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            w = _combine(kind, o["ma"], o["a20"], o["rma"], self._at(flag, o["close"].index, o["lag"]))
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_starts": n, "c_beat": int((edge > 0).sum()),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


def _holm(p: dict) -> list:
    order = sorted(p, key=lambda k: p[k])
    m = len(order)
    out = []
    for j, k in enumerate(order):
        if p[k] <= ALPHA / (m - j):
            out.append(k)
        else:
            break
    return out


def _part3(d: dict, B: dict, M1: dict) -> dict:
    close = d["close"]
    idx = close.index
    price, b0, pre = d["grid"]
    today = pd.Timestamp(common.today_utc())
    last = (today - pd.DateOffset(months=12)).to_period("M").to_timestamp()
    plan, exams, base, base_small = {}, {}, {}, {}
    for i, key in enumerate(KEYS):
        r = M1.get(key)
        if key not in B["R"] or not r:
            continue
        spec = B["R"][key]
        F = M1["_F"]
        lab, pop = spec["lab"], spec["pop"]
        v_all = F["fwd"][spec["h"]]
        S = pop & np.isfinite(lab) & np.isfinite(v_all) & F["base"]
        on, ref = _masks(lab, spec["mode"], S)
        c3, _ = _cells(F, S)
        first_half = idx < r["cut"]
        dir1 = _md(np.nan_to_num(v_all), on & first_half, ref & first_half, c3, 90)
        item = {"dir1": dir1, "cut": r["cut"]}
        plan[key] = item
        if not _ok(dir1):
            item["skip"] = "前半算不出方向"
            continue
        kinds = ("b",) if dir1 > 0 else ("w", "w50")
        item["kinds"] = kinds
        if (key == "R3a" and dir1 > 0) or (key in ("R3b", "R4") and dir1 < 0):
            item["skip"] = ("這個訊號只在 50 日均線關著的日子量到，「均線開著且沒有訊號才持有」不會改變均線的持有"
                            if dir1 < 0 else "這個訊號只在 50 日均線開著的日子量到，「均線開著或訊號開著」不會改變均線的持有")
            continue
        first = (r["cut"] + pd.offsets.MonthBegin(1)).normalize()
        starts, _ = msc._month_starts(idx, first, last)
        item["first"], item["n_starts"] = first, len(starts)
        if len(starts) < MIN_STARTS:
            item["skip"] = f"後半只有 {len(starts)} 個每月起算日（少於 {MIN_STARTS} 個）"
            continue
        fk = str(first.date())
        if fk not in exams:
            exams[fk] = _Exam(close, d["other"], price, b0, pre, first, last)
            base[fk] = exams[fk].summary(None, 1, SIMS, "ma50")
            base_small[fk] = exams[fk].summary(None, 1, PL_SIMS, "ma50")
        item["fk"] = fk
        item["ki"] = 300 + 3 * i
    res, pval = {}, {}
    for key, item in plan.items():
        if "fk" not in item:
            continue
        ex = exams[item["fk"]]
        state = B["R"][key]["state"]
        flag = state.fillna(0.0)
        s0 = state.first_valid_index()
        nn = int((idx >= s0).sum())
        full = {k: ex.summary(flag, item["ki"] + j, SIMS, k) for j, k in enumerate(item["kinds"])}
        small = {k: ex.summary(flag, item["ki"] + j, PL_SIMS, k) for j, k in enumerate(item["kinds"])}
        prng = np.random.default_rng(SEED + 7000 + item["ki"])
        hits = {k: 0 for k in item["kinds"]}
        for k_ in prng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, nn - SHIFT_MIN), DRAWS):
            rf = sd._roll(flag, s0, int(k_))
            for j, k in enumerate(item["kinds"]):
                x = ex.summary(rf, item["ki"] + j, PL_SIMS, k)
                hits[k] += int(sd._score_ge(x, small[k]))
        b = base[item["fk"]]
        pos_ma = np.nan_to_num(ex.ma)
        m_after = idx >= item["first"]
        for k in item["kinds"]:
            x = full[k]
            plc = hits[k] / DRAWS
            pval[(key, k)] = (hits[k] + 1) / (DRAWS + 1)
            if k == "b":
                loss = x["c_deep_med"] >= b["c_deep_med"] - LOSS_SLACK - 1e-12
            else:
                loss = x["c_deep_med"] >= b["c_deep_med"] - 1e-12
            score = sd._score_ge(x, b)
            pos = ex.btc_pos(flag, k)
            res[(key, k)] = {"x": x, "plc": plc, "score": score, "loss": loss, "plc_ok": plc <= PLACEBO_BAR,
                             "pass": bool(score and loss and plc <= PLACEBO_BAR),
                             "differ": int((np.abs(pos[m_after] - pos_ma[m_after]) > 1e-9).sum()),
                             "time": float(pos[m_after].mean()), "time_ma": float(pos_ma[m_after].mean())}
    holm = _holm(pval)
    for k in res:
        res[k]["holm"] = k in holm
    return {"plan": plan, "res": res, "base": base, "exams": exams, "last": last, "holm": holm, "pval": pval}


# ---------- Part 4 ----------

COL_KEYS = {"R1": ("R1",), "R2": ("R2",), "R3": ("R3a", "R3b"), "R4": ("R4",), "R5a": ("R5a",), "R5b": ("R5b",),
            "R6": ("R6",), "R7": ("R7", "R7b"), "R8": ("R8",), "R9": ("R9",), "R10a": ("R10a",), "R10b": ("R10b",)}
COUNT_COL = {"R7pre": ("R7", "R7"), "R7post": ("R7", "R7b")}


def _states(B: dict, idx, kept: set) -> pd.DataFrame:
    """Daily states of the kept signals (R3 = the pattern in either ma50 state, R7 = spikes)."""
    R = B["R"]
    cols = {}
    for col, keys in COL_KEYS.items():
        use = [k for k in keys if k in R and k in kept]
        if use:
            cols[col] = R[use[0]]["state"].reindex(idx).astype(float)
    return pd.DataFrame(cols, index=idx)


def _count_frame(st: pd.DataFrame, which: tuple, idx, kept: set) -> pd.DataFrame:
    out = {}
    pre = idx < pd.Timestamp("2023-01-01")
    for k in which:
        col, need = COUNT_COL.get(k, (k, k))
        if col not in st or (need not in kept and not any(x in kept for x in COL_KEYS.get(need, (need,)))):
            continue
        s = st[col]
        if k == "R7pre":
            s = s.where(pre, np.nan)
        elif k == "R7post":
            s = s.where(~pre, np.nan)
        out[k] = s
    return pd.DataFrame(out, index=idx)


def _part4(B: dict, F: dict, idx, kept: set) -> dict:
    st = _states(B, idx, kept)
    out = {"corr": st.corr(min_periods=200), "count": {}, "st": st}
    for nm, which in (("好", BETTER), ("差", WORSE)):
        cs = _count_frame(st, which, idx, kept)
        if cs.empty:
            continue
        defined = cs.notna().sum(axis=1).values
        cnt = cs.fillna(0.0).sum(axis=1).values
        pop = defined >= 3
        bucket = np.minimum(cnt, 3)
        v = F["fwd"][60]
        ok = pop & np.isfinite(v) & F["base"]
        rows = []
        for k in range(4):
            m = ok & (bucket == k)
            row = {"k": k, "n": int(m.sum()), "mean": float(v[m].mean()) if m.any() else None}
            if k > 0 and m.sum() >= 20 and (ok & (bucket == 0)).sum() >= 20:
                lab = np.where(pop, np.where(bucket == k, 1.0, np.where(bucket == 0, 0.0, 2.0)), np.nan)
                r = _measure(F, lab, "rest", 60, pop, (), SEED + 4000 + 10 * k + (0 if nm == "好" else 5),
                             placebo=False)
                if r:
                    row.update({"m3": r["m3"], "ci3": r["ci3"], "m2": r["m2"]})
            rows.append(row)
        out["count"][nm] = {"rows": rows, "which": [c.replace("pre", "（2023 以前）").replace("post", "（2023 起）")
                                                  for c in cs.columns],
                            "first": idx[ok][0] if ok.any() else None, "last": idx[ok][-1] if ok.any() else None}
    return out


# ---------- the computation ----------

def _compute(d: dict) -> dict:
    t0 = time.time()
    close = d["close"]
    idx = close.index
    B = _build(d)
    F = _feats(close)
    repro, kept = {}, set()
    for key in KEYS:
        if key not in B["R"]:
            continue
        rows = B["R"][key]["repro"]
        okr = all(_close_tol(s, g) for _, s, g in rows)
        repro[key] = okr
        if okr:
            kept.add(key)
    M1 = {"_F": F}
    for i, key in enumerate(KEYS):
        if key not in kept:
            continue
        spec = B["R"][key]
        M1[key] = _measure(F, spec["lab"], spec["mode"], spec["h"], spec["pop"], spec["periods"], SEED + 100 * (i + 1))
    t1 = time.time() - t0
    # Part 2: ETH
    eth = d["eth"]
    M2 = {}
    if eth is not None:
        FE = _feats(eth)
        for i, key in enumerate(KEYS):
            if key not in kept or not B["R"][key].get("eth"):
                continue
            es = B["R"][key]["eth"]
            M2[key] = _measure(FE, es["lab"], B["R"][key]["mode"], B["R"][key]["h"], np.asarray(es["pop"], bool), (),
                               SEED + 2000 + 10 * i, placebo=False)
            if M2[key] is not None:
                M2[key]["data"] = es["data"]
    # Part 3
    P3 = _part3(d, B, {k: v for k, v in M1.items()})
    t3 = time.time() - t0
    # Part 4
    P4 = _part4(B, F, idx, kept)
    return {"B": B, "F": F, "repro": repro, "kept": kept, "M1": M1, "M2": M2, "P3": P3, "P4": P4,
            "secs": time.time() - t0, "secs1": t1, "secs3": t3}


# ---------- report ----------

def _dir(v) -> str:
    return "—" if not _ok(v) else ("之後較好" if v > 0 else "之後較差")


def _repro_rows(r: dict) -> list:
    out = []
    B = r["B"]
    for key in KEYS:
        if key not in B["R"]:
            out.append([key, META[key][0], META[key][1], "—", "—", f"資料取不到（{B['fails'].get(key, '—')}），沒有算"])
            continue
        for j, (desc, stored, got) in enumerate(B["R"][key]["repro"]):
            out.append([key if j == 0 else "", META[key][0] if j == 0 else "", desc, _fmt_repro(desc, stored),
                        _fmt_repro(desc, got),
                        ("重現了" if r["repro"][key] else "沒有重現，這一項後面不算") if j == 0 else ""])
    return out


def _fmt_repro(desc: str, v) -> str:
    if not _ok(v):
        return "—"
    return _p(v) if "平均" in desc and "減" not in desc and "兩邊" not in desc else _pt(v)


def _p1_rows(r: dict) -> list:
    rows = []
    for key in KEYS:
        x = r["M1"].get(key)
        if key not in r["kept"]:
            continue
        if not x:
            rows.append([key, "—", "資料不夠", "—", "—", "—", "—", "算不出"])
            continue
        h = META[key][3]
        hv = "｜".join(_pt(z["m3"]) for z in x["halves"])
        same = "和母研究同方向" if _ok(x["m3"]) and x["m3"] * META[key][4] > 0 else "和母研究方向相反"
        rows.append([key, f"{_dir(x['m3'])}（{same}；之後 {h} 天）",
                     f"{_pt(x['raw'])}（{x['n_on']:,} 天 vs {x['n_ref']:,} 天）",
                     f"{_pt(x['m2'])} {_ci(*x['ci2'])}", f"{_pt(x['m3'])} {_ci(*x['ci3'])}", hv,
                     _share(x["placebo"]) if _ok(x.get("placebo")) else "—", _classify(x)])
    return rows


def _period_rows(r: dict) -> list:
    rows = []
    for key in KEYS:
        x = r["M1"].get(key)
        if key not in r["kept"] or not x:
            continue
        first = True
        for z in x["halves"]:
            a, b = z["span"]
            lab = f"前半 {_d(a)}–{_d(b)}" if first else f"後半 {_d(a)}–{_d(b)}"
            rows.append([key if first else "", lab, f"{z['n_on']:,}", _pt(z["raw"]), _pt(z["m3"])])
            first = False
        for z in x["per"]:
            rows.append(["", z["lab"], f"{z['n_on']:,}", _pt(z["raw"]), _pt(z["m3"])])
    return rows


def _m180_rows(r: dict) -> list:
    rows = []
    for key in KEYS:
        x = r["M1"].get(key)
        if key not in r["kept"] or not x:
            continue
        ch = (x["m3"] - x["m2"]) if _ok(x["m3"]) and _ok(x["m2"]) else None
        if not _ok(ch):
            say = "—"
        elif abs(ch) < 0.02:
            say = "幾乎沒變（差不到 2 點）"
        elif x["m2"] * x["m3"] < 0:
            say = "方向翻了"
        elif abs(x["m3"]) < abs(x["m2"]):
            say = f"變小 {abs(ch) * 100:.1f} 點"
        else:
            say = f"變大 {abs(ch) * 100:.1f} 點"
        rows.append([key, f"{_p(x['p180_on'])}｜{_p(x['p180_ref'])}", f"{_p(x['p30_on'])}｜{_p(x['p30_ref'])}",
                     f"{_pt(x['m2'])} → {_pt(x['m3'])}", say])
    return rows


def _eth_rows(r: dict) -> list:
    rows = []
    for key in KEYS:
        if key not in r["kept"]:
            continue
        x = r["M2"].get(key)
        bx = r["M1"].get(key)
        if not x:
            why = "專案裡沒有以太幣的這種資料" if not r["B"]["R"][key].get("eth") else "以太幣資料不夠，算不出"
            rows.append([key, why, "—", "—", "—", "—"])
            continue
        want = bx["m3"] if bx and _ok(bx["m3"]) else META[key][4]
        same = "一樣" if _ok(x["m3"]) and x["m3"] * want > 0 else "不一樣"
        rows.append([key, x["data"], f"{x['n']:,}｜{x['n_on']:,}", _pt(x["raw"]), f"{_pt(x['m3'])} {_ci(*x['ci3'])}",
                     same])
    return rows


KIND_CN = {"b": "B（均線開，或訊號開＋站上 20 日均線）", "w": "W（均線開且沒有訊號）", "w50": "W50（均線開；訊號開時只拿一半）"}


def _p3_rows(r: dict) -> list:
    P3 = r["P3"]
    rows = []
    for key in KEYS:
        if key not in P3["plan"]:
            continue
        it = P3["plan"][key]
        dtxt = f"{_dir(it['dir1'])}（前半配對差 {_pt(it['dir1'])}）"
        if "skip" in it:
            rows.append([key, dtxt, _d(it["first"]) + " 起" if "first" in it else "—",
                         str(it["n_starts"]) if "n_starts" in it else "—", "—", "—", "—", f"不考：{it['skip']}"])
            continue
        b = P3["base"][it["fk"]]
        for k in it["kinds"]:
            z = P3["res"][(key, k)]
            x = z["x"]
            verdict = "支持" if z["pass"] else "不支持"
            why = []
            if not z["score"]:
                why.append("排名輸均線")
            if not z["loss"]:
                why.append("虧損較深" if k == "b" else "虧損沒有比較淺")
            if not z["plc_ok"]:
                why.append("隨機錯位超過 5%")
            rows.append([key, f"{dtxt} → {KIND_CN[k]}", f"{_d(it['first'])} 起",
                         f"{x['d_n']}（時鐘 {x['c_starts']}）",
                         f"{_share1(x['low'])}（均線 {_share1(b['low'])}）",
                         f"{x['c_deep_med']:.1f}%（均線 {b['c_deep_med']:.1f}%）",
                         f"{_share(z['plc'])}｜Holm {'過' if z['holm'] else '沒過'}",
                         verdict + ("" if z["pass"] else f"（{'、'.join(why)}）")])
    return rows


def _corr_table(r: dict, cols: list) -> dict | None:
    c = r["P4"]["corr"]
    cols = [x for x in cols if x in c.columns]
    if not cols:
        return None
    rows = []
    for a in c.index:
        rows.append([a] + [("—" if a == b else (f"{c.loc[a, b]:+.2f}" if _ok(c.loc[a, b]) else "—")) for b in cols])
    return {"columns": ["訊號"] + cols, "rows": rows}


def _count_rows(r: dict) -> list:
    rows = []
    for nm, x in r["P4"]["count"].items():
        first = True
        for z in x["rows"]:
            k = "3 個以上" if z["k"] == 3 else f"{z['k']} 個"
            label = (f"「之後較{nm}」那一組（{'、'.join(x['which'])}；{_d(x['first'])} 到 {_d(x['last'])}）"
                     if first else "")
            rows.append([label, k, f"{z['n']:,}", _p(z["mean"]),
                         ("比較基準" if z["k"] == 0 else
                          (f"{_pt(z.get('m3'))} {_ci(*z.get('ci3', (None, None)))}" if _ok(z.get("m3")) else "天數太少"))])
            first = False
    return rows


def _now(d: dict, r: dict) -> list[dict]:
    close = d["close"]
    idx = close.index
    t = idx[-1]
    ma = r["B"]["ma"][-1]
    st = r["P4"]["st"]
    late = t >= pd.Timestamp("2023-01-01")
    which_b = [k for k in BETTER if k != "R7pre" or not late]
    which_w = [k for k in WORSE if k != "R7post" or late]
    on, unk, used = {"好": [], "差": []}, [], {"好": [], "差": []}
    for nm, which in (("好", which_b), ("差", which_w)):
        cs = _count_frame(st, tuple(which), idx, r["kept"])
        for k in cs.columns:
            name = k.replace("pre", "").replace("post", "")
            used[nm].append(name)
            v = cs[k].iloc[-1]
            if not _ok(v):
                unk.append(name)
            elif v > 0.5:
                on[nm].append(name)
    val = (f"{'持有' if ma > 0.5 else '現金'}｜「之後較好」的訊號今天開著 {len(on['好'])} 個"
           f"{('（' + '、'.join(on['好']) + '）') if on['好'] else ''}，「之後較差」的開著 {len(on['差'])} 個"
           f"{('（' + '、'.join(on['差']) + '）') if on['差'] else ''}")
    rows = [{"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"},
            {"label": "比特幣：50 日均線｜反向訊號", "value": val},
            {"label": "　今天算進去的訊號",
             "value": f"「之後較好」那組 {len(used['好'])} 個（{'、'.join(used['好']) or '無'}）；"
                      f"「之後較差」那組 {len(used['差'])} 個（{'、'.join(used['差']) or '無'}）"}]
    if unk:
        rows.append({"label": "　今天資料還沒出來或算不出（當作沒開）", "value": "、".join(unk)})
    rows.append({"label": "這幾格的意思", "value": "只描述最後一根日線收盤時哪些訊號開著，不是預測，也不是買賣建議；"
                                            "第三部分沒有規則被支持之前，這些訊號都不拿來操作"})
    return rows


def _verdict(r: dict) -> str:
    kept = [k for k in KEYS if k in r["kept"]]
    dropped = [k for k in KEYS if k not in r["kept"]]
    cls = {k: _classify(r["M1"].get(k)) for k in kept}
    holds = [k for k in kept if cls[k] == "站得住"]
    part = [k for k in kept if cls[k] == "部分"]
    notr = [k for k in kept if cls[k] in ("沒有重複", "算不出")]
    flip = [k for k in kept if r["M1"].get(k) and _ok(r["M1"][k]["m3"]) and r["M1"][k]["m3"] * META[k][4] < 0]
    s = [f"量到的：母研究的數字 {len(KEYS)} 個裡 {len(kept)} 個這次重算得出來"
         + (f"（{'、'.join(dropped)} 沒有重現或取不到資料，後面不算）" if dropped else "") + "。"]
    s.append(f"第一部分（事先定好的判讀）：站得住 {len(holds)} 個（{'、'.join(holds) or '無'}）；"
             f"部分 {len(part)} 個（{'、'.join(part) or '無'}）；前後兩半方向不同 {len(notr)} 個（{'、'.join(notr) or '無'}）。"
             + (f"配對再加上「過去 180 天漲幅」後，{'、'.join(flip)} 的方向和母研究相反。" if flip else ""))
    shrink = [k for k in kept if r["M1"].get(k) and _ok(r["M1"][k]["m3"]) and _ok(r["M1"][k]["m2"])
              and abs(r["M1"][k]["m3"]) < abs(r["M1"][k]["m2"]) - 0.02]
    if shrink:
        s.append(f"加配 180 天漲幅讓 {'、'.join(shrink)} 的差變小超過 2 點：這幾個有一部分是「之前已經漲了好幾個月」。")
    eth = [(k, x) for k, x in r["M2"].items() if x and _ok(x["m3"])]
    if eth:
        same = [k for k, x in eth if r["M1"].get(k) and _ok(r["M1"][k]["m3"]) and x["m3"] * r["M1"][k]["m3"] > 0]
        s.append(f"以太幣用自己的資料：{len(eth)} 個能算的裡 {len(same)} 個方向一樣（{'、'.join(same) or '無'}）。")
    res = r["P3"]["res"]
    n_rules = len(res)
    passed = [f"{k}-{kk.upper()}" for (k, kk), z in res.items() if z["pass"]]
    holm = [f"{k}-{kk.upper()}" for (k, kk) in r["P3"]["holm"]]
    s.append(f"第三部分（方向只用前半定、只在後半起算日考）：考了 {n_rules} 條規則，事先定的條件通過 {len(passed)} 條"
             f"（{'、'.join(passed) or '無'}）；Holm 校正後還在的 {len(holm)} 條（{'、'.join(holm) or '無'}）；"
             f"光靠運氣，隨機錯位這一關大約會有 {n_rules * PLACEBO_BAR:.1f} 條過。")
    s.append("推測：這些反過來的結果多半不是一個故事；能說的只有上面量到的數字，資料分不出「熱鬧＝之後好」和「之前漲多了＝之後好」的地方，下面每一項都寫了。"
             "沒看到＝用這些定義和這些資料沒看到。")
    return "".join(s)


def study() -> dict:
    with _own_store():
        data = _load()
        r = _compute(data)
    close = data["close"]
    P3 = r["P3"]
    tables = [
        {"caption": ("先重算母研究的數字（同樣的資料讀法、同樣的定義）。「重現了」＝這次的數字和母研究存的差不超過 2 點或 15%"
                     "（取大的）。沒有重現的，後面都不算。點＝百分點。"),
         "columns": ["編號", "母研究", "量的是什麼", "母研究存的數字", "這次重算", "結果"],
         "rows": _repro_rows(r)},
        {"caption": ("第一部分：每個反過來的結果，照「量到的方向」重新量一次。差＝訊號開的日子之後的平均漲跌，減掉比較組"
                     "（每一項的比較組寫在下一張表）。配對＝只拿「過去 30 天漲幅、近 20 天波動」差不多的日子互比（母研究的做法）；"
                     "再加 180 天＝另外要求「過去 180 天漲幅」也差不多（新加的，用來看是不是「之前已經漲了好幾個月」）。"
                     "［ ］是 90% 區間（60 天一塊的區塊重抽 1,000 次）。前半｜後半＝把樣本照日期切成兩半各算一次（加 180 天的配對）。"
                     "隨機錯位＝把訊號整段往後挪至少 180 天、做 200 次，假訊號至少一樣好的比例。判讀事先定好：站得住＝兩半同方向、"
                     "區間不含 0、隨機錯位 ≤ 5%；部分＝兩半同方向但其他沒全過；沒有重複＝兩半方向不同。"),
         "columns": ["編號", "量到的方向", "原始差（訊號天數 vs 比較天數）", "配對差 ［90% 區間］", "再加 180 天的配對差 ［90% 區間］",
                     "前半｜後半", "隨機錯位", "判讀"],
         "rows": _p1_rows(r)},
        {"caption": "每一項在量什麼、和誰比、看之後幾天（訊號都只用當天收盤前就知道的資料）。",
         "columns": ["編號", "母研究", "訊號", "比較組", "看之後幾天"],
         "rows": [[k, META[k][0], META[k][1], META[k][2] + (f"；{STATE_NOTE[k]}" if k in STATE_NOTE else ""),
                   f"{META[k][3]} 天"] for k in KEYS]},
        {"caption": "第一部分：前後兩半和母研究的分段（原始差與加 180 天的配對差）。R5、R8 的母研究沒有分段，只看前後兩半。",
         "columns": ["編號", "期間", "訊號天數", "原始差", "再加 180 天的配對差"],
         "rows": _period_rows(r)},
        {"caption": ("第一部分：加配「過去 180 天漲幅」做了什麼。前兩欄是訊號日和比較日之前的平均漲幅（量到的）；"
                     "如果訊號日之前漲得多、而加配之後差變小，表示原本的差有一部分只是「之前漲了好幾個月」。"),
         "columns": ["編號", "過去 180 天平均漲幅：訊號日｜比較日", "過去 30 天：訊號日｜比較日", "配對差：只配 30 天 → 再加 180 天",
                     "加配之後"],
         "rows": _m180_rows(r)},
        {"caption": ("第二部分：換成以太幣，用以太幣自己的價格和自己的資料，同樣的量法（加 180 天的配對）。"
                     "「一樣」＝以太幣的配對差和比特幣量到的方向相同。"),
         "columns": ["編號", "以太幣的資料", "樣本天數｜訊號天數", "原始差", "配對差 ［90% 區間］", "方向和比特幣一樣嗎"],
         "rows": _eth_rows(r)},
        {"caption": (f"第三部分：照量到的方向做成規則，誠實切開：方向只用樣本前半定（程式算前半的配對差，正的用 B、負的用 W 和 W50），"
                     f"門檻用母研究自己的，只在後半的起算日考試。公平考試＝主訊號檢查的四關（比特幣日線每月起算、24 個收盤時間、"
                     f"一般人做法：晚 10 小時、成本 0.25%、一成進場漏掉、每個收盤時間模擬 {SIMS} 次、其他幣），起算日到 "
                     f"{_d(P3['last'])}（比主訊號檢查的 2023-12 晚，因為後半都在後面；50 日均線用同樣的起算日重考）。"
                     f"排名分數＝四關「贏過一直抱著」比例裡最低的；最深虧損中位數＝24 個收盤時間各起算日的中位數。"
                     f"隨機錯位：規則的訊號整段平移至少 180 天、{DRAWS} 次、一般人做法每個收盤時間只模擬 {PL_SIMS} 次。"
                     f"支持＝排名不輸均線、虧損（B 最多深 5 點；W、W50 不能更深）、隨機錯位 ≤ 5%。少於 {MIN_STARTS} 個起算日的不考。"),
         "columns": ["編號", "前半的方向 → 規則", "起算", "日線起算日數", "排名分數", "最深虧損中位數", "隨機錯位｜Holm", "判定"],
         "rows": _p3_rows(r)},
    ]
    t = _corr_table(r, ["R1", "R2", "R5a", "R7", "R8", "R9"])
    if t:
        tables.append({"caption": ("第四部分：這些訊號是不是同一件事？每天開／關的相關係數（兩個都有資料的日子；+1＝完全同時開，0＝無關）。"
                                   "這一張是和「之後較好」那幾個（加上 R9）的相關。"), **t})
    t = _corr_table(r, ["R3", "R4", "R5b", "R6", "R10a", "R10b"])
    if t:
        tables.append({"caption": "第四部分：和「之後較差」那幾個的相關。", **t})
    tables.append({"caption": ("第四部分：同一天開著幾個「之後較好」的訊號（或幾個「之後較差」的），之後 60 天的平均漲跌，"
                               "以及和「一個都沒開」的日子用加 180 天配對比的差（推測用，不是規則）。資料還沒有的訊號算「沒開」；"
                               "只看至少三個訊號有資料的日子。"),
                   "columns": ["組", "開著幾個", "天數", "之後 60 天平均", "配對差（vs 0 個）［90% 區間］"],
                   "rows": _count_rows(r)})
    tables.append({"caption": "第五部分：每一項資料分不出來、或可能誤導的地方。",
                   "columns": ["編號", "資料分不出來的事"],
                   "rows": [[k, LIMITS[k]] for k in KEYS]})
    notes = [
        "為什麼做這個：稽核約 100 個被放棄的結果時發現，只要量到的方向和假設相反，就被記成「失敗」丟掉。和猜測相反的結果還是結果；"
        "這十個從來沒有照量到的方向考過。",
        "誠實的問題：這十個都是在整段歷史上「看到」的，所以在同一段歷史上重算證明不了什麼。這裡能算證據的只有：前後兩半是否一致、"
        "加配 180 天漲幅後還在不在、隨機錯位、以太幣，以及第三部分「前半定方向、後半考」。",
        "量到的 vs 推測：表裡的數字都是量到的；凡是解釋（例如「熱鬧代表參與者多」）都是推測，資料分不出來的地方寫在第五部分。"
        "一種傾向（例如「漲過的會繼續漲」）不拿來當定律去否定另一種；兩種都放在表裡用數字並排比。",
        "R5、R9 的第一部分量的是母研究的「開始那天」（次數少：R5 約十次、R9 約二十幾次），第三部分規則用的是每天的旗標。",
        "R7、R10 的反向只出現在一段時間（R7 2023 起反過來；R10 只看 2022 起）：第一部分照母研究的範圍量；"
        "第三部分的前後兩半也在同一個範圍裡切，所以後半的起算日可能不夠而不考。",
        "第三部分的起算日到今天往前一年（主訊號檢查只到 2023-12），因為這些樣本的後半都在近幾年；50 日均線用同樣起算日重考，比較是公平的，"
        "但最後幾年的起算日只量到一兩年，比較吵。",
        f"資料：所有下載都存進 dashboard/cache/_store/{ID}/（共用的存檔只讀）。OBM 的老幣資料讀的是 old_coin_selling 存下來的那一份。"
        f"這次總共算了 {r['secs'] / 60:.1f} 分鐘。",
    ] + [f"資料提醒：{p}" for p in data["problems"]]
    for k, why in r["B"]["fails"].items():
        notes.append(f"{k} 這次算不出來（{why}），沒有算進去。")
    sources = [
        {"name": "Coin Metrics 社群 API：比特幣 FlowInExNtv、CapMVRVCur、SplyCur、FlowOutExNtv；以太幣 FlowInExNtv",
         "url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"},
        {"name": "Open Bitcoin Metrics（CC BY 4.0）：放了 155 天以上／不到 155 天的幣被花掉（只讀資料檔）",
         "url": "https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/obm_spent_value_ge155d_btc_daily/obm_spent_value_ge155d_btc_daily.csv"},
        {"name": "Open Bitcoin Metrics：不到 155 天的幣被花掉",
         "url": "https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/obm_spent_value_lt155d_btc_daily/obm_spent_value_lt155d_btc_daily.csv"},
        {"name": "DefiLlama 穩定幣總供給（買壓分數用）", "url": "https://stablecoins.llama.fi/stablecoincharts/all"},
        {"name": "Farside 美國現貨比特幣 ETF 每日淨流入（買壓分數用）", "url": "https://farside.co.uk/bitcoin-etf-flow-all-data/"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT，含主動買進量）", "url": "https://api.binance.com/api/v3/klines"},
        {"name": "幣安公開歷史資料：U 本位合約每日 metrics 檔（未平倉量，BTCUSDT、ETHUSDT）",
         "url": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/"},
        {"name": "幣安永續合約資金費率（BTCUSDT、ETHUSDT）", "url": "https://fapi.binance.com/fapi/v1/fundingRate"},
        {"name": "Coinbase Exchange 公開日線（BTC-USD、ETH-USD）", "url": "https://api.exchange.coinbase.com/products/BTC-USD/candles"},
        {"name": "Bitstamp 公開日線 BTC/USD（買壓分數的美元溢價，只讀存檔）", "url": "https://www.bitstamp.net/api/v2/ohlc/btcusd/"},
        {"name": "維基媒體瀏覽量 API：英文維基「Bitcoin」「Ethereum」",
         "url": "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Bitcoin/daily/20150701/" + common.today_utc().replace("-", "")},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]
    return {
        "id": ID,
        "title": "反過來的結果：照量到的方向再考一次",
        "question": ("十個被當成「失敗」丟掉的結果，量到的方向都和原本的猜測相反（例如大量轉進交易所之後反而漲得多）。"
                     "照量到的方向重新量：前後兩半一不一致、多配對「過去 180 天漲幅」後還在不在、換以太幣一不一樣、"
                     "用前半定方向的規則在後半能不能贏 50 日均線？"),
        "asof": _d(close.index[-1]),
        "verdict": _verdict(r),
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }
