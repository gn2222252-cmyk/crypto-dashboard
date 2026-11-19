"""Historical study "move_size": many indicators were only ever asked "will price go UP?" and were dropped when the
answer was no. Do some of them say HOW BIG the next move will be (in either direction) rather than WHICH WAY - and, if
so, can that size information help the trend rule ma50 (hold BTC while the close is above its 50-day average)?

Why (written before any run). An audit of about 100 abandoned results (cache/_audit_part1.json, _audit_part2.json,
"unasked_questions" and "connections") found numbers that look like SIZE information: futures basis >= 20%/yr was
followed by a >= 10% drop within 30 days 77% of the time versus 46% on all days, yet the blocked high-basis days still
rose +77% in total; funding in its top tenth 47% versus 20%; open interest high before 4 of 4 big falls; funding low
before 4 of 5 big rises; exchange balance rising followed by bigger moves both ways; exchange traffic ranks 87 around
tops against 42-75 on ordinary days and is also high before rises; "buying and selling pressure both up" was the most
common pattern before BOTH big rises (12 of 23) and big falls (14 of 27); low volatility says a move is coming, not its
direction. An indicator that predicts size but not direction cannot beat ma50 as an entry/exit rule, but it might tell
a trend follower when the next signal matters most, how much to hold, or how tight to keep a stop. Never tested.

Ground rules (the owner's): observe first; do not conclude lightly; statements are labelled 量到的 (measured: number,
k of N) or 推測 (a guess, marked); where the data cannot separate two explanations, say so; one tendency is never used
as a law to dismiss another; "not seen" = not seen with this definition and this data; a result opposite to the
expected direction is a finding and is reported as measured.

Everything below was fixed BEFORE any result was seen. Anything added after seeing results is labelled
"ADDED AFTER SEEING THE RESULTS".

Data (free, key-less, user agent "btc-research-dashboard" through the dashboard's helpers; .env never read; a refused
endpoint is named, never worked around). Loaders are REUSED; they write only into this study's own folder
cache/_store/move_size/ (signs_before_turns' approach: the shared store is only read; files a loader may fall back to
are copied in first when newer; module store constants are pointed at the own folder for the run).
  BTC daily closes: te._daily_close (00:00 UTC; blockchain.info before Binance 2017-08). Day t = candle labelled t.
  pressure_before_moves._load / _pressures (Coin Metrics FlowInExNtv / FlowOutExNtv lag 1; Open Bitcoin Metrics spent
    >= 155 days lag 7; Binance spot 1-hour klines (quote, taker-buy quote; zero-fee days 2022-07-08..2023-04-04 set
    missing); DefiLlama stablecoins lag 1; Farside ETF lag 1; Coinbase / Bitstamp premium; Binance funding;
    Binance futures metrics archive (open interest, long/short ratios)).
  Coin Metrics SplyExNtv (coins on exchange-tagged addresses; exit_forces._cm_fetch), data day D used at day D+1.
  Coin Metrics FeeTotNtv x PriceUSD (blockspace._onchain), 7-day mean, lag 1.
  Deribit quarterly futures + Bitstamp spot -> annualised basis (basis_filter._deribit / _bitstamp / _basis), lag 0.
  Deribit DVOL (vrp_direction._dvol), lag 0.
  Exam inputs: Binance 1-hour BTCUSDT grid (clock_robustness), ETH SOL BNB XRP (Binance daily) and MSTR (Yahoo,
    from 2020-08-11) through main_signal_check._load_other.

Indicators (18; each lagged to publication as above; RANK = pressure_before_moves._trail_rank: share of the previous
365 days' values (at least 180 present) below today's value, x 100):
  LEVERAGE  basis = Deribit annualised basis; fund = Binance funding 7-day mean (per year); fund_abs = |fund|;
            oi = log open interest (coins; stamp D+1 00:00 matched with candle D); oi_ch = OI 30-day change
            (pressure_before_moves' OI).
  ACTIVITY  act = exchange traffic (withdrawn + deposited, 30 days, coins); old = old coins moving (>= 155 days,
            winsorised 30-day sum); spot = Binance spot quote volume, 7 days (taker buy + taker sell); fee = fees in
            USD, 7-day mean.
  PRESSURE  D30(X) = mean rank of X over days -30..-1 minus over -90..-31 (pressure_before_moves._deltas) for the
            composites SELL_C and BUY_C. p_sum = D30(SELL_C) + D30(BUY_C) (high = "both up", low = "both down");
            p_abs = |D30(SELL_C)| + |D30(BUY_C)| (size of the two changes).
  BALANCE   bal_abs = |log(SplyExNtv / SplyExNtv 30 days earlier)|; net_abs = |NET30| (net withdrawn share).
  QUIET     rv20 = 20-day realised volatility (std of daily log returns x sqrt 365; LOW = quiet); range30 = (highest
            - lowest close of the last 30 closes) / close (LOW = narrow).
  OPTIONS   dvol = Deribit DVOL; vrp = DVOL - 30-day realised volatility (points).
  POSITIONING ls_dist = |retail long/short ratio (7-day mean) - its median over the previous 365 days (>= 180)|.
  An indicator whose loader fails is dropped and named, never substituted.

Outcomes for day t (H = 30 and 60 closes after t; complete windows only; in %):
  UPSIZE = max(close t+1..t+H) / close t - 1; DOWNSIZE = 1 - min(close t+1..t+H) / close t;
  SIZE = max(UPSIZE, DOWNSIZE); RET = close t+H / close t - 1; RV = std of the H daily log returns x sqrt 365.
  Direction skill is about RET; size skill is about SIZE (and RV).

Part 1 - size versus direction, per indicator (days t from 2012-01-01 with a rank and a complete window):
  Fifths by rank (<= 20, 20-40, 40-60, 60-80, > 80): mean SIZE, UPSIZE, DOWNSIZE, RET, RV per fifth.
  Raw difference = top fifth minus bottom fifth. MATCHED difference (the key control): cells = decile of past 20-day
  realised volatility (rv20 level) x decile of past 30-day return (close t / close t-30 - 1); deciles fixed once over
  all days 2012-01-01 .. last day with a 60-day window; per cell (top mean - bottom mean) where both are present,
  averaged with weights = number of top-fifth days in the cell. Coverage = share of top days in usable cells.
  Intervals: moving block bootstrap of the indicator's sample days in time order, blocks of 60 consecutive sample
  days, 400 draws (fixed seed); 90% interval = 5th-95th percentile.
  Placebo (H = 30): the rank series circularly shifted within its defined span by k uniform in [180, span - 180]
  days, 200 draws (fixed seed); two-sided p = (1 + #|placebo matched diff| >= |real|) / 201, for SIZE and for RET.
  FAMILY = matched SIZE difference at 30 days, 18 indicators; Benjamini-Hochberg at q = 10% on its placebo p.
  "Survivor" = BH survivor. Luck: with 18 tests, about 1.8 would show p <= 0.10 by chance alone; BH keeps the
  expected share of false survivors at or below 10%.
  "Survives the volatility match" = the matched SIZE30 90% interval excludes zero (raw shown next to it).
  By period for survivors: matched SIZE30 difference on days in 2012-2018, 2019-2022, 2023-now (point values).
  Persistence check (description): correlation of past 30-day size (max |close t-k / close t - 1|, k = 1..30) with
  forward SIZE30, and of rv20 with forward RV30, all days from 2012.

Part 2 - anchored on events (description): events = pressure_before_moves' big-rise starts (>= 50% before a 25% fall)
  and big-fall starts (>= 30% before a 25% rebound) from 2012; ma50 exits labelled by whipsaw_anatomy._label
  (whip = false exit, real = real decline); ma50 entries (off at u-1, on at u) labelled now: FIZZLED = ma50 off again
  within 20 closes at a close below the entry close; RAN = a close >= 1.15 x entry close before ma50 turns off;
  otherwise other. For each surviving size indicator (if there is none: the three with the smallest Part 1 size
  placebo p, clearly labelled "not survivors"): mean rank over days -30..-1 (at least 20 present) per event, averaged
  over the events with data; random band = 10th-90th percentile of the mean of n random days (n = events with data;
  2,000 draws, fixed seed) from all days from 2012 with a value. Question: high before BOTH big rises and big falls,
  and low before false exits / fizzled entries?

Part 3 - size information with the trend rule (rules fixed now; scored on starts from 2019 only):
  S = the best size indicator chosen on days BEFORE 2019-01-01: among the indicators except rv20 (rv20 is the twin)
  with >= 730 sample days whose 30-day window ends before 2019-01-01, the largest |matched SIZE30 difference| on those
  days; the code picks. S is oriented so that HIGH = bigger moves expected (S = 100 - rank if the pre-2019 difference
  is negative). Top third = S > 66.7, bottom third = S < 33.3; a missing S = middle (rule acts as ma50).
  The TWIN of each rule uses the past 20-day realised volatility rank (rv20 rank; high = big) in place of S.
  S and the twin are BTC's, used as a market-wide state on every series of the exam (BTC daily: day t; 24 clocks:
  a clock close on UTC date E uses candle E-1; ETH SOL BNB XRP: same UTC date; MSTR date D uses candle D-1).
   Z1 stop tightness: ma50, but when S is in its top third at an ma50 entry, also leave when the close falls 10% below
      the highest close since that entry; after such a stop wait for ma50's next entry.
   Z2 patience when quiet: when S is in its bottom third at an ma50 exit, keep holding until the close has been at or
      below the 50-day average for 5 consecutive closes (the exit close counts as the first); ma50 back on cancels it.
   Z3 size: when ma50 is on, hold 60% if S is in its top third, else 100%.
   Z3b mirror: when ma50 is on, hold 60% if S is in its bottom third, else 100%. (Both, because the audit's numbers
      do not say which way is right.)
   Z4 re-entry: after an ma50 exit with S in its top third, stay in cash for that close and the next 9 closes (10
      closes), whatever ma50 says; then follow ma50.
  Fair exam versus ma50 = whipsaw_anatomy._Exam.summary (main_signal_check's four checks) on price_impact._Exam built
  from first month 2019-01: BTC daily monthly starts 2019-01 .. 2023-12 (0.10%); the same starts x 24 closing hours
  (textbook); the same as an ordinary person (10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per
  clock); ETH SOL BNB XRP MSTR each with its own ma50, starts from 2019-01 (or later) to three years before today.
  Ranking score = worst of the four "beats holding" shares with main_signal_check's eligibility (sd._score_ge);
  median deepest loss = median over the clock textbook cells. A rule and its twin share the random seeds.
  Placebo per rule: S circularly shifted (exit_forces._roll: k in [180, span - 180] from S's first defined day), the
  whole exam rebuilt with 20 ordinary-person simulations per clock, 200 draws; the real rule re-scored the same way;
  placebo share = share of draws ranked at least as high; Holm p = (hits + 1) / 201.
  "Supported", all of: ranking score >= ma50's (sd._score_ge); median deepest loss no worse than ma50's (Z2: no more
  than 5 points worse); placebo share <= 5%; better than its twin (higher ranking score, or equal score and a higher
  median clock edge). Holm at 5% over the five rules reported next to it; by luck about 5 x 5% = 0.25 rules would
  pass the placebo bar.

Part 4 - what cannot be seen: coverage per indicator (basis, funding, open interest, options and long/short are short
  and from one venue); 30- and 60-day windows overlap, so day counts are not independent cases (about n / 30
  separate windows); SIZE and RV are strongly persistent, so much apparent skill is persistence - which is why the
  volatility match and the volatility twin are the bars.
Today: a row labelled exactly "比特幣：50 日均線｜Z1｜Z3" ("現金｜現金｜現金" style) and where each surviving size
  indicator stands, in plain words, marked as description, not forecast. Seeds fixed. Not advice.
ADDED AFTER SEEING THE RESULTS (no definition, threshold or rule changed): a verdict sentence that, for every indicator
  whose matched SIZE30 interval excludes zero, splits the size difference into its matched UPSIZE, DOWNSIZE and RET
  parts and says whether it comes from one side (then it is direction, not size); and the short name "交易所進出量"
  (was "交易所進出量（活躍度）", shown twice with the group).
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
from dashboard.collectors import classic
from dashboard.collectors import flows2
from dashboard.collectors import price as px
from dashboard.studies2 import basis_filter as bf
from dashboard.studies2 import blockspace as bs
from dashboard.studies2 import carry as cy
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import exit_forces as xf
from dashboard.studies2 import force_balance as fb
from dashboard.studies2 import funding_warning as fw
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import oi_flush as oo
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import pressure_before_moves as pbm
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import vrp_direction as vd
from dashboard.studies2 import whipsaw_anatomy as wa

ID = "move_size"
DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2012-01-01")
SPLIT = pd.Timestamp("2019-01-01")
PERIODS = (("2012–2018", "2012-01-01", "2018-12-31"), ("2019–2022", "2019-01-01", "2022-12-31"),
           ("2023 起", "2023-01-01", "2100-01-01"))
HORIZONS = (30, 60)
OUTS = ("size", "up", "down", "ret", "rv")
OUT_CN = {"size": "最大波動幅度", "up": "最大漲幅", "down": "最大跌幅", "ret": "報酬", "rv": "實際波動"}
BOOT, BLOCK, PL_DRAWS, SHIFT_MIN, BAND_DRAWS = 400, 60, 200, 180, 2000
BH_Q = 0.10
SEED = 20261120
MIN_SAMPLE, MIN_GROUP, MIN_PRE = 365, 30, 730
TOP3, BOT3 = 200.0 / 3.0, 100.0 / 3.0
Z1_STOP, Z2_DAYS, Z3_LOW, Z4_DAYS = 0.90, 5, 0.60, 10
ENTRY_DAYS, ENTRY_RUN = 20, 1.15
EXAM_DRAWS, PL_SIMS = 200, 20
LOSS_SLACK_Z2, PLACEBO_BAR, HOLM_ALPHA = 5.0, 0.05, 0.05
RULES = ("z1", "z2", "z3", "z3b", "z4")
KI = {"ma50": 1, "z1": 150, "z2": 151, "z3": 152, "z3b": 153, "z4": 154}
RNAME = {"ma50": "50 日均線（基準）",
         "z1": "Z1 收緊停損：進場時 S 在高的三分之一，就加一條「從進場後最高收盤跌 10% 就走」",
         "z2": "Z2 安靜時多等：下車時 S 在低的三分之一，要連續 5 天收在 50 日均線之下才走",
         "z3": "Z3 調部位：均線持有時，S 在高的三分之一只拿 60%，其他拿 100%",
         "z3b": "Z3b 反過來：均線持有時，S 在低的三分之一只拿 60%，其他拿 100%",
         "z4": "Z4 不急著回來：下車時 S 在高的三分之一，之後 10 天不進場"}
RSHORT = {"ma50": "50 日均線", "z1": "Z1", "z2": "Z2", "z3": "Z3", "z3b": "Z3b", "z4": "Z4"}

# key: (short name, plain gloss, group, lag text)
META = {
    "basis": ("期貨溢價", "Deribit 季度期貨比現貨貴多少（換算成一年）；越高代表越多人願意付錢做多", "槓桿", "0 天"),
    "fund": ("資金費率", "幣安永續合約做多的人付給做空的人的費率（7 天平均，換算成一年）", "槓桿", "0 天"),
    "fund_abs": ("資金費率離 0 多遠", "資金費率的絕對值：不管是做多還是做空擠，擠得越厲害越大", "槓桿", "0 天"),
    "oi": ("未平倉量", "幣安 BTCUSDT 合約還沒平掉的部位（顆數）", "槓桿", "0 天"),
    "oi_ch": ("未平倉量 30 天變化", "未平倉量比 30 天前多或少幾 %", "槓桿", "0 天"),
    "act": ("交易所進出量", "30 天內送進和提出交易所的幣加起來", "活躍度", "1 天"),
    "old": ("老幣移動", "放了 155 天以上的幣，30 天內被動用的量", "活躍度", "7 天"),
    "spot": ("現貨成交量", "幣安現貨 BTCUSDT 7 天成交金額", "活躍度", "0 天"),
    "fee": ("手續費", "每天付給礦工的手續費（美元，7 天平均）", "活躍度", "1 天"),
    "p_sum": ("買賣壓一起升", "賣方合計和買方合計近 30 天比之前 60 天的變化加起來；高 = 兩邊都升，低 = 兩邊都降", "買賣壓", "1–7 天"),
    "p_abs": ("買賣壓變化大小", "賣方合計和買方合計變化的絕對值加起來；不管方向，變得越多越大", "買賣壓", "1–7 天"),
    "bal_abs": ("交易所存量變化大小", "交易所地址上的幣比 30 天前多或少多少（取絕對值）", "存量", "1 天"),
    "net_abs": ("淨提走比例偏離", "30 天（提走−送進）÷（提走＋送進）的絕對值；不管方向，越偏一邊越大", "存量", "1 天"),
    "rv20": ("過去 20 天波動", "過去 20 天每日漲跌的年化波動；低 = 安靜", "安靜", "0 天"),
    "range30": ("過去 30 天區間寬度", "過去 30 天最高收盤減最低收盤，除以今天收盤；低 = 很窄", "安靜", "0 天"),
    "dvol": ("隱含波動", "Deribit 選擇權反映的未來 30 天預期波動（DVOL）", "選擇權", "0 天"),
    "vrp": ("隱含減實際波動", "隱含波動減過去 30 天實際波動；高 = 選擇權比實際波動貴", "選擇權", "0 天"),
    "ls_dist": ("散戶多空比偏離", "幣安散戶多空比（7 天平均）離它過去一年中位數多遠（不分方向）", "部位", "0 天"),
}
KEYS = tuple(META)
SEED_FILES = ("pressure_before_moves_cm.json", "force_balance_cm.json", "pressure_before_moves_llama.json",
              "force_balance_llama.json", "pressure_before_moves_obm.json", "old_coin_selling_obm.json",
              "force_balance_etf.json", "etf_loop_btc_frame.json", "price_impact_1h_BTCUSDT.csv",
              "coinbase_premium_bitstamp_btcusd.json", "kimchi_premium_coinbase.json", "kimchi_premium_upbit.json",
              "kimchi_premium_usdkrw.json", "funding_warning_binance.json", "ls_ratio_btcusdt.json",
              "basis_filter_contracts.json", "blockspace_cm.json", "vrp_direction_dvol_btc.json",
              "exit_forces_cm_btc.json", "supply_demand_cm.json", "supply_demand_llama.json")
REMAP = ((pbm, "STORE_CM"), (pbm, "STORE_LL"), (pbm, "STORE_OBM"), (fb, "STORE_CM"), (fb, "STORE_LL"),
         (fb, "STORE_ETF"), (fb, "ETF_FALLBACK"), (ocs, "STORE_OBM"), (ocs, "STORE_CDD"), (pi, "STORE"),
         (sd, "STORE_CM"), (sd, "STORE_LL"), (sd, "BS_STORE"), (km, "STORE_CB"), (km, "STORE_UP"),
         (km, "STORE_FX"), (fw, "STORE_PATH"), (bf, "STORE_FILE"))
REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _pp(v, nd: int = 1) -> str:
    return f"{float(v):+.{nd}f}" if _ok(v) else "—"


def _ci(lo, hi, nd: int = 1) -> str:
    return f"{float(lo):+.{nd}f}～{float(hi):+.{nd}f}" if _ok(lo) and _ok(hi) else "—"


def _sh(x) -> str:
    return f"{float(x) * 100:.0f}%" if _ok(x) else "—"


# ---------- own store ----------

@contextmanager
def _own_store():
    """Loaders write into cache/_store/move_size/; the shared store is only read (seed copies when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        src, dst = REAL_STORE / name, MY_STORE / name
        try:
            if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        except Exception:  # noqa: BLE001 - a missing seed only means the loader must download
            pass
    saved = [(m, a, getattr(m, a)) for m, a in REMAP]
    old = common.STORE
    common.STORE = MY_STORE
    for m, a, p in saved:
        setattr(m, a, MY_STORE / Path(p).name)
    try:
        yield
    finally:
        common.STORE = old
        for m, a, p in saved:
            setattr(m, a, p)


# ---------- data ----------

def _load() -> dict:
    """Close, raw indicator series known at each close, problems, dropped indicators, events inputs, exam inputs."""
    problems: list[str] = []
    dropped: dict = {}
    with _own_store():
        data = pbm._load()
        problems += data["problems"]
        close = data["close"]
        idx = close.index
        raw_p, rank_p, _ = pbm._pressures(data)
        raw: dict = {}
        c = close.values.astype(float)
        lc = np.log(close)
        r1 = lc.diff()
        raw["rv20"] = r1.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(365)
        rv30 = r1.rolling(30, min_periods=30).std(ddof=1) * math.sqrt(365)
        raw["range30"] = (close.rolling(30, min_periods=30).max() - close.rolling(30, min_periods=30).min()) / close
        # pressure_before_moves pieces
        raw["fund"] = raw_p["FUND"]
        raw["fund_abs"] = raw_p["FUND"].abs()
        raw["oi_ch"] = raw_p["OI"]
        raw["act"] = raw_p["ACT"]
        raw["old"] = raw_p["S_old"]
        raw["spot"] = raw_p["S_take"] + raw_p["B_take"]
        raw["net_abs"] = raw_p["NET30"].abs()
        ds, _ = pbm._deltas(rank_p["SELL_C"].values)
        db, _ = pbm._deltas(rank_p["BUY_C"].values)
        raw["p_sum"] = pd.Series(ds + db, index=idx)
        raw["p_abs"] = pd.Series(np.abs(ds) + np.abs(db), index=idx)
        # open interest level and retail long/short distance from the archive rows
        rows = data.get("oi")
        if rows:
            o = oo._oi(rows)
            with np.errstate(all="ignore"):
                raw["oi"] = pd.Series(np.log(o.where(o > 0)).reindex(idx + DAY).values.astype(float), index=idx)
            keys = sorted(rows)
            st = pd.to_datetime(keys)
            ls = pd.Series([rows[k][0] if rows[k][0] is not None else np.nan for k in keys], index=st, dtype=float)
            ls7 = ls.asfreq("D").rolling(7, min_periods=5).mean()
            med = ls7.shift(1).rolling(365, min_periods=180).median()
            raw["ls_dist"] = sbt._known((ls7 - med).abs(), idx, 0, 3)
        else:
            dropped["oi"] = dropped["ls_dist"] = "幣安期貨歷史資料庫這次取不到"
        # exchange balance
        try:
            got, prob = xf._stored("cm_btc.json", lambda: xf._cm_fetch("btc"))
            if prob:
                problems.append(prob)
            f = pd.DataFrame.from_dict(got, orient="index")
            f.index = pd.to_datetime(f.index)
            sp = f["SplyExNtv"].astype(float).sort_index().asfreq("D")
            sp = sp.where(sp > 0)
            raw["bal_abs"] = pbm._lag(np.log(sp / sp.shift(30)).abs(), idx, 1)
        except Exception as exc:  # noqa: BLE001
            dropped["bal_abs"] = f"{type(exc).__name__}"
        # fees
        try:
            ob, pr = bs._onchain()
            problems += pr
            fee = (ob["FeeTotNtv"] * ob["PriceUSD"]).where(ob["FeeTotNtv"] > 0).rolling(7, min_periods=7).mean()
            raw["fee"] = sbt._known(fee, idx, 1, 3)
        except Exception as exc:  # noqa: BLE001
            dropped["fee"] = f"{type(exc).__name__}"
        # basis
        try:
            store = bf._read_store()
            spot = bf._bitstamp("btcusd")
            contracts = bf._deribit("BTC", store)
            basis, _ = bf._basis(contracts, spot["close"])
            raw["basis"] = sbt._known(basis, idx, 0, 3)
        except Exception as exc:  # noqa: BLE001
            dropped["basis"] = f"{type(exc).__name__}"
        # DVOL, VRP
        try:
            dv, pr = vd._dvol("BTC")
            if pr:
                problems.append(pr)
            dvk = sbt._known(dv, idx, 0, 3)
            raw["dvol"] = dvk
            raw["vrp"] = dvk - rv30 * 100
        except Exception as exc:  # noqa: BLE001
            dropped["dvol"] = dropped["vrp"] = f"{type(exc).__name__}"
        # exam inputs
        rows_h, prob = cr._hourly("BTCUSDT")
        if prob:
            problems.append(prob)
        other, more = msc._load_other()
        problems += more
    raw = {k: pd.Series(np.asarray(raw[k], dtype=float), index=idx).where(lambda s: np.isfinite(s))
           for k in KEYS if k in raw}
    for k in KEYS:
        if k not in raw and k not in dropped:
            dropped[k] = "沒有組出來"
    rank = {k: pbm._trail_rank(raw[k]) for k in raw}
    return {"close": close, "raw": raw, "rank": rank, "rank_p": rank_p, "problems": problems, "dropped": dropped,
            "rows_h": rows_h, "other": other}


# ---------- outcomes and controls ----------

def _outcomes(c: np.ndarray) -> dict:
    n = len(c)
    lr_ = np.diff(np.log(c), prepend=np.nan)
    out = {}
    for H in HORIZONS:
        m = n - H
        Y = np.full((n, len(OUTS)), np.nan)
        w = np.lib.stride_tricks.sliding_window_view(c[1:], H)[:m]
        up = w.max(axis=1) / c[:m] - 1
        dn = 1 - w.min(axis=1) / c[:m]
        Y[:m, 0] = np.maximum(up, dn)
        Y[:m, 1] = up
        Y[:m, 2] = dn
        Y[:m, 3] = c[H:] / c[:m] - 1
        rw = np.lib.stride_tricks.sliding_window_view(lr_[1:], H)[:m]
        Y[:m, 4] = rw.std(axis=1, ddof=1) * math.sqrt(365)
        out[H] = Y * 100
    return out


def _cells(close: pd.Series, rv20: pd.Series, Y60: np.ndarray) -> np.ndarray:
    """Decile of past 20-day volatility x decile of past 30-day return; edges from all days 2012+ with a 60-day window."""
    past = (close / close.shift(30) - 1).values
    v = rv20.values
    base = (close.index >= START) & np.isfinite(past) & np.isfinite(v) & np.isfinite(Y60[:, 0])
    qs = np.linspace(0.1, 0.9, 9)
    ev, ep = np.quantile(v[base], qs), np.quantile(past[base], qs)
    cell = np.searchsorted(ev, v, side="right") * 10 + np.searchsorted(ep, past, side="right")
    return np.where(np.isfinite(past) & np.isfinite(v), cell, -1)


# ---------- part 1 ----------

def _diffs(G: np.ndarray, C: np.ndarray, Y: np.ndarray, idx: np.ndarray | None = None) -> tuple:
    """(raw diff per outcome, matched diff per outcome, coverage) for top (G=1) minus bottom (G=0)."""
    if idx is not None:
        G, C, Y = G[idx], C[idx], Y[idx]
    m = G < 2
    key = C[m] * 2 + G[m]
    cnt = np.bincount(key, minlength=200).reshape(100, 2).astype(float)
    ok = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
    w = np.where(ok, cnt[:, 1], 0.0)
    raw, mat = np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan)
    tb = cnt.sum(axis=0)
    for j in range(Y.shape[1]):
        s = np.bincount(key, weights=Y[m, j], minlength=200).reshape(100, 2)
        if tb[0] > 0 and tb[1] > 0:
            raw[j] = s[:, 1].sum() / tb[1] - s[:, 0].sum() / tb[0]
        if w.sum() > 0:
            with np.errstate(all="ignore"):
                dc = s[:, 1] / cnt[:, 1] - s[:, 0] / cnt[:, 0]
            mat[j] = float(np.sum(w[ok] * dc[ok]) / w.sum())
    cov = float(w.sum() / tb[1]) if tb[1] > 0 else np.nan
    return raw, mat, cov


def _groups(r: np.ndarray) -> np.ndarray:
    return np.where(r > 80, 1, np.where(r <= 20, 0, 2))


def _blocks(n: int, rng) -> np.ndarray:
    if n <= BLOCK:
        return rng.integers(0, n, n)
    k = -(-n // BLOCK)
    st = rng.integers(0, n - BLOCK + 1, k)
    return (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]


def _roll_rank(r: np.ndarray, u: float) -> np.ndarray:
    fin = np.where(np.isfinite(r))[0]
    out = r.copy()
    if not len(fin):
        return out
    a, b = fin[0], fin[-1] + 1
    span = b - a
    k = SHIFT_MIN + int(u * max(1, span - 2 * SHIFT_MIN))
    out[a:b] = np.roll(r[a:b], k)
    return out


def _sample(r: np.ndarray, Y: np.ndarray, cell: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.where(base & np.isfinite(r) & (cell >= 0) & np.all(np.isfinite(Y), axis=1))[0]


def _part1(close: pd.Series, rank: dict, Ys: dict, cell: np.ndarray) -> dict:
    base = np.asarray(close.index >= START)
    years = close.index
    rng = np.random.default_rng(SEED)
    us = rng.random(PL_DRAWS)
    R = {}
    for ki, k in enumerate(rank):
        r = rank[k].values.astype(float)
        res = {"h": {}}
        for H in HORIZONS:
            Y = Ys[H]
            s = _sample(r, Y, cell, base)
            if len(s) < MIN_SAMPLE:
                res = None
                break
            G, C, YY, rr = _groups(r[s]), cell[s], Y[s], r[s]
            if (G == 1).sum() < MIN_GROUP or (G == 0).sum() < MIN_GROUP:
                res = None
                break
            fifth = np.clip(np.ceil(rr / 20.0) - 1, 0, 4).astype(int)
            fm = np.array([[YY[fifth == q, j].mean() if (fifth == q).any() else np.nan for j in range(len(OUTS))]
                           for q in range(5)])
            raw, mat, cov = _diffs(G, C, YY)
            brng = np.random.default_rng(SEED + 31 * ki + H)
            braw, bmat = np.zeros((BOOT, len(OUTS))), np.zeros((BOOT, len(OUTS)))
            for b in range(BOOT):
                ii = _blocks(len(s), brng)
                braw[b], bmat[b], _ = _diffs(G, C, YY, ii)
            per = {}
            for name, a, z in PERIODS:
                pm = (years[s] >= pd.Timestamp(a)) & (years[s] <= pd.Timestamp(z))
                if pm.sum() >= 180 and (G[pm] == 1).sum() >= 20 and (G[pm] == 0).sum() >= 20:
                    per[name] = (float(_diffs(G[pm], C[pm], YY[pm])[1][0]), int(pm.sum()))
                else:
                    per[name] = (np.nan, int(pm.sum()))
            res["h"][H] = {"n": len(s), "first": years[s[0]], "fm": fm, "raw": raw, "mat": mat, "cov": cov,
                           "raw_ci": np.nanpercentile(braw, [5, 95], axis=0),
                           "mat_ci": np.nanpercentile(bmat, [5, 95], axis=0), "per": per,
                           "ntop": int((G == 1).sum()), "nbot": int((G == 0).sum())}
        if res is None:
            R[k] = None
            continue
        # placebo (30 days): matched SIZE and RET
        Y = Ys[30]
        real = res["h"][30]["mat"]
        hs = hr = 0
        for u in us:
            rr = _roll_rank(r, float(u))
            s = _sample(rr, Y, cell, base)
            G = _groups(rr[s])
            if (G == 1).sum() == 0 or (G == 0).sum() == 0:
                continue
            _, mat, _ = _diffs(G, cell[s], Y[s][:, [0, 3]])
            hs += int(_ok(mat[0]) and abs(mat[0]) >= abs(real[0]) - 1e-12)
            hr += int(_ok(mat[1]) and abs(mat[1]) >= abs(real[3]) - 1e-12)
        res["p_size"] = (1 + hs) / (1 + PL_DRAWS)
        res["p_ret"] = (1 + hr) / (1 + PL_DRAWS)
        # pre-2019 discovery statistic (30-day windows ending before 2019-01-01)
        pre = base & (np.arange(len(close)) + 30 < int(close.index.searchsorted(SPLIT)))
        s = _sample(r, Y, cell, pre)
        if len(s) >= MIN_PRE:
            G = _groups(r[s])
            if (G == 1).sum() >= MIN_GROUP and (G == 0).sum() >= MIN_GROUP:
                res["pre"] = (float(_diffs(G, cell[s], Y[s][:, [0]])[1][0]), len(s))
        R[k] = res
    keys = [k for k in R if R[k] is not None]
    m = len(keys)
    order = sorted(keys, key=lambda k: R[k]["p_size"])
    cut = 0
    for i, k in enumerate(order, 1):
        if R[k]["p_size"] <= i / m * BH_Q:
            cut = i
    surv = order[:cut]
    return {"R": R, "keys": keys, "order": order, "surv": surv, "m": m}


def _persistence(close: pd.Series, rv20: pd.Series, Ys: dict) -> dict:
    c = close.values.astype(float)
    n = len(c)
    past = np.full(n, np.nan)
    w = np.lib.stride_tricks.sliding_window_view(c, 31)
    past[30:] = np.max(np.abs(w[:, :30] / w[:, 30:31] - 1), axis=1) * 100
    base = np.asarray(close.index >= START)
    y = Ys[30]
    m1 = base & np.isfinite(past) & np.isfinite(y[:, 0])
    m2 = base & np.isfinite(rv20.values) & np.isfinite(y[:, 4])
    return {"size": float(np.corrcoef(past[m1], y[m1, 0])[0, 1]),
            "rv": float(np.corrcoef(rv20.values[m2], y[m2, 4])[0, 1])}


# ---------- part 2 ----------

def _entries(c: np.ndarray, ma: np.ndarray) -> list[tuple[int, str]]:
    on = np.nan_to_num(ma) > 0.5
    n = len(c)
    out = []
    for u in np.where(~on[:-1] & on[1:])[0] + 1:
        off = next((k for k in range(u + 1, n) if not on[k]), None)
        run = next((k for k in range(u + 1, n if off is None else off) if c[k] >= ENTRY_RUN * c[u]), None)
        if run is not None:
            lab = "ran"
        elif off is not None and off - u <= ENTRY_DAYS and c[off] < c[u]:
            lab = "fizzle"
        elif off is not None:
            lab = "other"
        else:
            lab = "unknown"
        out.append((int(u), lab))
    return out


def _part2(close: pd.Series, rank: dict, keys: list) -> dict:
    c = close.values.astype(float)
    idx = close.index
    ma = msc._targets(close)["ma50"].values.astype(float)
    ev = {"rise": [idx.get_loc(e["d"]) for e in pbm._rises(close)],
          "fall": [idx.get_loc(e["d"]) for e in pbm._falls(close)]}
    ex = wa._exits(ma)
    labs = [wa._label(c, ma, int(e)) for e in ex]
    ev["whip"] = [int(e) for e, l in zip(ex, labs) if l == "whip"]
    ev["real"] = [int(e) for e, l in zip(ex, labs) if l == "real"]
    en = _entries(c, ma)
    ev["fizzle"] = [u for u, l in en if l == "fizzle"]
    ev["ran"] = [u for u, l in en if l == "ran"]
    base = np.asarray(idx >= START)
    out = {"ev": ev, "n_exits": len(ex), "n_entries": len(en), "cells": {}}
    rng = np.random.default_rng(SEED + 500)
    for k in keys:
        m30 = rank[k].shift(1).rolling(30, min_periods=20).mean().values
        pool = m30[base & np.isfinite(m30)]
        for t, pos in ev.items():
            v = np.array([m30[p] for p in pos if p >= 0 and base[p] and np.isfinite(m30[p])])
            if len(v) < 3 or len(pool) < 365:
                out["cells"][(k, t)] = None
                continue
            draws = rng.choice(pool, size=(BAND_DRAWS, len(v)), replace=True).mean(axis=1)
            lo, hi = np.percentile(draws, [10, 90])
            out["cells"][(k, t)] = {"mean": float(v.mean()), "n": len(v), "lo": float(lo), "hi": float(hi),
                                    "high": int((v >= 60).sum())}
    return out


# ---------- part 3: rules and exam ----------

def _rule(rule: str, c: np.ndarray, ma: np.ndarray, v: np.ndarray) -> np.ndarray:
    on = np.nan_to_num(np.asarray(ma, dtype=float)) > 0.5
    pos = on.astype(float)
    top = np.nan_to_num(v, nan=50.0) > TOP3
    bot = np.nan_to_num(v, nan=50.0) < BOT3
    n = len(c)
    if rule == "z3":
        return pos * np.where(top, Z3_LOW, 1.0)
    if rule == "z3b":
        return pos * np.where(bot, Z3_LOW, 1.0)
    if rule == "z1":
        tight = stopped = False
        peak = 0.0
        for i in range(n):
            if on[i]:
                if i == 0 or not on[i - 1]:
                    tight, stopped, peak = bool(top[i]), False, c[i]
                peak = max(peak, c[i])
                if stopped:
                    pos[i] = 0.0
                elif tight and c[i] <= Z1_STOP * peak:
                    stopped = True
                    pos[i] = 0.0
        return pos
    if rule == "z2":
        patient, off = False, 0
        for i in range(n):
            if on[i]:
                patient, off = False, 0
                continue
            if i > 0 and on[i - 1]:
                patient, off = bool(bot[i]), 0
            off += 1
            if patient and off < Z2_DAYS:
                pos[i] = 1.0
        return pos
    if rule == "z4":
        until = -1
        for i in range(n):
            if i > 0 and on[i - 1] and not on[i] and top[i]:
                until = i + Z4_DAYS - 1
            if i <= until:
                pos[i] = 0.0
        return pos
    raise ValueError(rule)


class _Exam(pi._Exam):
    """price_impact's exam on starts from 2019-01 with whipsaw_anatomy's summary; spec = None (ma50) or
    (rule, BTC state series on BTC daily dates)."""

    summary = wa._Exam.summary

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first)
        npre = len(pre)
        self.series = {"btc": {"c": close.values.astype(float), "ma": np.nan_to_num(self.ma), "dates": close.index}}
        dates = self.m.dates[:self.days] - DAY
        for h in range(24):
            cf = np.concatenate([pre, price[h::24]]).astype(float)
            self.series[("clock", h)] = {"c": cf[npre:][:self.days], "ma": self.cma[h], "dates": dates}
        for key, o in self.other.items():
            self.series[("o", key)] = {"c": o["close"].values.astype(float), "ma": np.nan_to_num(o["ma"]),
                                       "dates": o["close"].index - o["lag"] * DAY}

    def pos(self, key, spec, perm=None) -> np.ndarray:
        s = self.series[key]
        if spec is None:
            return np.asarray(s["ma"], dtype=float).copy()
        rule, state = spec
        return _rule(rule, s["c"], s["ma"], state.reindex(s["dates"]).values.astype(float))


def _better(x: dict, y: dict) -> bool:
    if x["eligible"] != y["eligible"]:
        return x["eligible"]
    if x["low"] > y["low"] + 1e-12:
        return True
    return abs(x["low"] - y["low"]) <= 1e-12 and x["c_edge"] > y["c_edge"]


def _holm(ps: dict, alpha: float) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    out, m, go = {}, len(order), True
    for i, k in enumerate(order):
        go = go and ps[k] <= alpha / (m - i)
        out[k] = go
    return out


def _part3(close: pd.Series, rank: dict, P1: dict, rows_h: dict, other: dict) -> dict:
    R = P1["R"]
    cands = {k: R[k]["pre"] for k in P1["keys"] if k != "rv20" and "pre" in R[k] and _ok(R[k]["pre"][0])}
    if not cands:
        return {"cands": cands, "pick": None}
    pick = max(cands, key=lambda k: abs(cands[k][0]))
    sign = 1.0 if cands[pick][0] >= 0 else -1.0
    S = rank[pick] if sign > 0 else 100.0 - rank[pick]
    twin = rank["rv20"]
    t0 = time.time()
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    ex = _Exam(close, other, price, b0, pre, SPLIT)
    s = {"ma50": ex.summary(None, KI["ma50"], wa.SIMS)}
    tw = {}
    for k in RULES:
        s[k] = ex.summary((k, S), KI[k], wa.SIMS)
        tw[k] = ex.summary((k, twin), KI[k], wa.SIMS)
    t_exam = time.time() - t0
    small = {k: ex.summary((k, S), KI[k], PL_SIMS) for k in RULES}
    hits = {k: 0 for k in RULES}
    rng = np.random.default_rng(SEED + 9000)
    for _ in range(EXAM_DRAWS):
        u = float(rng.random())
        Sr = xf._roll(S, u)
        for k in RULES:
            x = ex.summary((k, Sr), KI[k], PL_SIMS)
            hits[k] += int(sd._score_ge(x, small[k]))
    placebo = {k: hits[k] / EXAM_DRAWS for k in RULES}
    pholm = {k: (hits[k] + 1) / (EXAM_DRAWS + 1) for k in RULES}
    holm = _holm(pholm, HOLM_ALPHA)
    base = s["ma50"]
    sup, parts = {}, {}
    for k in RULES:
        x = s[k]
        slack = LOSS_SLACK_Z2 if k == "z2" else 0.0
        a = sd._score_ge(x, base)
        b = x["c_deep_med"] >= base["c_deep_med"] - slack - 1e-9
        c_ = placebo[k] <= PLACEBO_BAR
        d = _better(x, tw[k])
        parts[k] = (a, b, c_, d)
        sup[k] = a and b and c_ and d
    pos = {"ma50": ex.pos("btc", None), **{k: ex.pos("btc", (k, S)) for k in RULES}}
    return {"cands": cands, "pick": pick, "sign": sign, "S": S, "s": s, "tw": tw, "placebo": placebo,
            "pholm": pholm, "holm": holm, "sup": sup, "parts": parts, "pos": pos, "ex": ex,
            "secs": time.time() - t0, "secs_exam": t_exam}


# ---------- text helpers ----------

def _name(k: str) -> str:
    return META[k][0]


def _today_word(r: float, sign: float) -> str:
    """Plain words for today's rank given the measured direction of the matched size difference."""
    if not _ok(r):
        return "今天沒有值"
    pos = "高" if r > 80 else "偏高" if r > 60 else "中間" if r >= 40 else "偏低" if r >= 20 else "低"
    if pos == "中間":
        size = "落在中間，過去這種位置之後的波動大小也在中間"
    else:
        big = (r > 50) == (sign > 0)
        size = f"過去這種位置之後 30 天的波動通常比較{'大' if big else '小'}"
    return f"排名 {r:.0f}（比過去一年 {r:.0f}% 的日子高，屬於{pos}）；{size}（描述過去，不是預測）"


def _state(v: float) -> str:
    if not _ok(v) or v <= 0.001:
        return "現金"
    return "持有" if v >= 0.999 else f"持有 {v * 100:.0f}%"


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    D = _load()
    close = D["close"]
    rank, raw = D["rank"], D["raw"]
    Ys = _outcomes(close.values.astype(float))
    cell = _cells(close, raw["rv20"], Ys[60])
    P1 = _part1(close, rank, Ys, cell)
    R, surv = P1["R"], P1["surv"]
    for k in KEYS:
        if k in rank and R.get(k) is None and k not in D["dropped"]:
            D["dropped"][k] = "有資料的日子不夠（至少 365 天、最高最低五分之一各 30 天）"
    persist = _persistence(close, raw["rv20"], Ys)
    show = surv if surv else P1["order"][:3]
    P2 = _part2(close, rank, show)
    P3 = _part3(close, rank, P1, D["rows_h"], D["other"])
    secs = time.time() - t0

    keys = P1["keys"]
    tables = []

    # Table 1: summary
    rows = []
    for k in sorted(keys, key=lambda k: R[k]["p_size"]):
        h = R[k]["h"][30]
        h6 = R[k]["h"][60]
        ms, mr = h["mat"][0], h["mat"][3]
        sci, rci = h["mat_ci"][:, 0], h["mat_ci"][:, 3]
        surv_v = sci[0] > 0 or sci[1] < 0
        rows.append([
            f"{_name(k)}（{META[k][2]}）",
            f"{_d(h['first'])[:7]} 起｜{h['n']:,} 天",
            f"{_pp(mr)}（{_ci(rci[0], rci[1])}）｜p {R[k]['p_ret']:.2f}",
            f"{_pp(h['raw'][0])}（{_ci(h['raw_ci'][0, 0], h['raw_ci'][1, 0])}）",
            f"{_pp(ms)}（{_ci(sci[0], sci[1])}）",
            f"{_pp(h6['mat'][0])}（{_ci(h6['mat_ci'][0, 0], h6['mat_ci'][1, 0])}）",
            "撐過" if surv_v else "沒撐過",
            f"{R[k]['p_size']:.3f}｜{'通過' if k in surv else '—'}",
        ])
    tables.append({
        "caption": ("第一部分總表：每個指標「說方向」和「說大小」的本事並排。做法：把每天的指標和它自己過去 365 天比排名（0–100），"
                    "取最高五分之一（排名 > 80）減最低五分之一（≤ 20）的日子，看之後 30 天的表現差多少（百分點）。"
                    "「配對後」＝只在「過去 20 天波動」和「過去 30 天漲跌」都落在同一個十等分格子裡的日子之間比，"
                    "因為波動會延續：一個指標如果只是跟著現在的波動走，配對後就應該剩不多。括號是 90% 區間（把日子按時間切成 60 天一塊、"
                    f"重抽 {BOOT} 次）。p ＝ 把指標整串錯開至少 180 天、{PL_DRAWS} 次，配對後差距一樣大或更大的比例（不分正負）。"
                    "「最大波動幅度」＝之後 30 天內收盤離今天收盤最遠的距離，不分漲跌。按 p 由小到大排。"),
        "columns": ["指標（類別）", "有資料（起｜天數）", "方向：配對後 30 天報酬差（區間）｜p", "大小：沒配對的差（區間）",
                    "大小：配對後的差（區間）", "大小：配對後 60 天", "大小在配對後撐過嗎", "大小 p｜多重檢定"],
        "rows": rows})

    # Table 2: all five outcomes, matched
    rows = []
    for k in sorted(keys, key=lambda k: R[k]["p_size"]):
        h = R[k]["h"][30]
        rows.append([_name(k)] + [f"{_pp(h['raw'][j])} → {_pp(h['mat'][j])}（{_ci(h['mat_ci'][0, j], h['mat_ci'][1, j])}）"
                                  for j in range(len(OUTS))] + [_sh(h["cov"])])
    tables.append({
        "caption": ("五種結果都列出來（之後 30 天，最高五分之一減最低五分之一，百分點）：每格「沒配對 → 配對後（配對後的 90% 區間）」。"
                    "最大漲幅＝之後 30 天內最高收盤比今天高多少；最大跌幅＝最低收盤比今天低多少（正數）；報酬＝30 天後的漲跌；"
                    "實際波動＝之後 30 天每日漲跌的年化波動。最後一欄是配對時用得上的最高五分之一日子比例（格子裡也要有最低五分之一的日子才能比）。"),
        "columns": ["指標"] + [OUT_CN[o] for o in OUTS] + ["配對涵蓋"],
        "rows": rows})

    # Table 3: fifths
    rows = []
    for k in sorted(keys, key=lambda k: R[k]["p_size"]):
        fm = R[k]["h"][30]["fm"]
        rows.append([_name(k)] + [f"{fm[q, 0]:.1f}｜{fm[q, 3]:+.1f}" if _ok(fm[q, 0]) else "—" for q in range(5)])
    tables.append({
        "caption": "按指標排名分五等分（沒配對）：每格是之後 30 天「最大波動幅度平均｜報酬平均」（%）。第一欄最低五分之一，最後一欄最高。",
        "columns": ["指標", "最低 1/5", "第 2", "第 3", "第 4", "最高 1/5"],
        "rows": rows})

    # Table 4: periods for survivors (or the top three)
    rows = []
    for k in show:
        per = R[k]["h"][30]["per"]
        rows.append([_name(k) + ("" if k in surv else "（不是多重檢定通過的，只是 p 最小的前三個）")]
                    + [f"{_pp(per[p][0])}（{per[p][1]:,} 天）" for p, _, _ in PERIODS])
    tables.append({
        "caption": "分期間看（配對後 30 天最大波動幅度差，最高五分之一減最低五分之一，百分點；括號是那段期間有資料的天數）。",
        "columns": ["指標"] + [p for p, _, _ in PERIODS],
        "rows": rows})

    # Table 5: event anchored
    EVN = (("rise", "大漲起點"), ("fall", "大跌起點"), ("whip", "均線假跌破"), ("real", "均線真的下跌"),
           ("fizzle", "進場後很快失敗"), ("ran", "進場後走出 +15%"))
    rows = []
    for k in show:
        row = [_name(k)]
        for t, _ in EVN:
            x = P2["cells"].get((k, t))
            if not x:
                row.append("事件太少")
                continue
            tag = "高於隨機" if x["mean"] > x["hi"] else "低於隨機" if x["mean"] < x["lo"] else "在隨機範圍內"
            row.append(f"{x['mean']:.0f}（{x['n']} 次）｜隨機 {x['lo']:.0f}–{x['hi']:.0f}｜{tag}")
        rows.append(row)
    ev = P2["ev"]
    tables.append({
        "caption": ("第二部分（只是描述）：事件發生前 30 天，指標的平均排名（0–100，50 是普通）。大漲起點＝之後漲 50% 以上才回檔 25%"
                    f"（{len(ev['rise'])} 次）；大跌起點＝之後跌 30% 以上才反彈 25%（{len(ev['fall'])} 次）；均線下車共 {P2['n_exits']} 次，"
                    f"假跌破＝20 天內又回到均線之上而且價格更高（{len(ev['whip'])} 次），真的下跌＝先跌 15% 才回來（{len(ev['real'])} 次）；"
                    f"均線進場共 {P2['n_entries']} 次，很快失敗＝20 天內又跌回均線下、而且價格比進場低（{len(ev['fizzle'])} 次），"
                    f"走出行情＝跌回均線下之前曾漲 15%（{len(ev['ran'])} 次）。括號是有指標資料的次數；隨機＝隨便抽同樣多天、{BAND_DRAWS} 次的中間八成。"),
        "columns": ["指標"] + [n for _, n in EVN],
        "rows": rows})

    # Table 6: discovery
    if P3.get("pick"):
        rows = [[_name(k), f"{P3['cands'][k][1]:,}", _pp(P3["cands"][k][0]), "選中" if k == P3["pick"] else ""]
                for k in sorted(P3["cands"], key=lambda k: -abs(P3["cands"][k][0]))]
        tables.append({
            "caption": ("第三部分先選 S：只用 2019 年以前的日子（之後 30 天的窗口也要在 2019 年前結束），算配對後最大波動幅度差，"
                        "絕對值最大的那個當 S（過去 20 天波動不算，因為它是對照組）。差是負的就把排名反過來，讓「S 高」永遠代表「預期波動大」。"),
            "columns": ["指標", "2019 年前的天數", "配對後大小差（百分點）", "選擇"],
            "rows": rows})

        s, tw = P3["s"], P3["tw"]
        base = s["ma50"]

        def cells(x):
            return [f"{_sh(x['low'])}{'' if x['eligible'] else '（虧損條件沒過）'}",
                    "／".join(_sh(v) for v in x["checks"]), f"{x['c_edge']:+.1f}", f"{x['c_deep_med']:.1f}%"]
        rows = [[RNAME["ma50"]] + cells(base) + ["—", "—"]]
        for k in RULES:
            a, b, c_, d = P3["parts"][k]
            rows.append([RNAME[k]] + cells(s[k]) + [
                f"{_sh(P3['placebo'][k])}｜Holm {'過' if P3['holm'][k] else '沒過'}",
                ("支持" if P3["sup"][k] else "不支持") + f"（排名{'✓' if a else '✗'} 虧損{'✓' if b else '✗'} "
                f"錯位{'✓' if c_ else '✗'} 贏對照{'✓' if d else '✗'}）"])
            rows.append([f"　└ 對照：同一條規則改用過去 20 天波動排名"] + cells(tw[k]) + ["—", "—"])
        ex = P3["ex"]
        tables.append({
            "caption": (f"第三部分公平考試（只算 2019 年以後的起算日）。S ＝ {_name(P3['pick'])}"
                        f"{'' if P3['sign'] > 0 else '（排名反過來）'}。比特幣日線 {len(ex.starts)} 個起算月、"
                        f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間、一般人做法（晚 10 小時、成本 0.25%、一成進場漏掉、每個收盤時間模擬 "
                        f"{wa.SIMS} 次）、其他資產（ETH SOL BNB XRP MSTR 用各自的 50 日均線，S 用比特幣的）。「排名分數」＝四項「報酬贏過一直抱著」"
                        f"比例裡最低的一項；四項依序列出。「差距」＝起算日×收盤時間格每年報酬比一直抱著多幾個百分點（中位數）。錯位＝把 S 整串錯開、"
                        f"{EXAM_DRAWS} 次整套重考，排名一樣好或更好的比例；Holm 是五條一起看的多重檢定（5%）。支持＝排名不輸均線、最深虧損不比均線深"
                        f"（Z2 允許深 5 點）、錯位 ≤ 5%、而且贏過「改用過去 20 天波動」的對照。"),
            "columns": ["規則", "排名分數", "四項：日線／收盤時間／一般人／其他資產", "差距（百分點）", "最深虧損中位數",
                        "錯位｜Holm", "結論"],
            "rows": rows})

    # Table 7: coverage
    rows = []
    for k in KEYS:
        if k in D["dropped"] or R.get(k) is None:
            rows.append([_name(k), META[k][1], META[k][3], "沒有算", D["dropped"].get(k, "—")])
            continue
        h = R[k]["h"][30]
        rows.append([_name(k), META[k][1], META[k][3], f"{_d(h['first'])}｜{h['n']:,} 天（約 {h['n'] // 30} 個不重疊的 30 天）",
                     f"最高 1/5 {h['ntop']:,} 天、最低 1/5 {h['nbot']:,} 天"])
    tables.append({
        "caption": "第四部分：每個指標是什麼、晚幾天才知道、從哪天起有排名、有多少天（相鄰的日子看的是幾乎同一段行情，真正獨立的大約是天數 ÷ 30）。",
        "columns": ["指標", "意思", "晚幾天才用", "從哪天起｜天數", "最高／最低五分之一"],
        "rows": rows})

    # ---------- verdict ----------
    n_surv = len(surv)
    size_ok = [k for k in keys if (R[k]["h"][30]["mat_ci"][0, 0] > 0 or R[k]["h"][30]["mat_ci"][1, 0] < 0)]
    dir_ok = [k for k in keys if (R[k]["h"][30]["mat_ci"][0, 3] > 0 or R[k]["h"][30]["mat_ci"][1, 3] < 0)]
    raw_ok = [k for k in keys if (R[k]["h"][30]["raw_ci"][0, 0] > 0 or R[k]["h"][30]["raw_ci"][1, 0] < 0)]
    v = [f"量到的：{P1['m']} 個指標裡，沒配對時「之後 30 天最大波動幅度」最高和最低五分之一的差，90% 區間不含 0 的有 {len(raw_ok)} 個；"
         f"只在過去波動和過去漲跌相同的日子之間比（配對後）剩 {len(size_ok)} 個；配對後報酬（方向）區間不含 0 的有 {len(dir_ok)} 個。"]
    if n_surv:
        v.append(f"錯位檢查加多重檢定（10%）後，大小的本事留下 {n_surv} 個：" + "、".join(
            f"{_name(k)} {_pp(R[k]['h'][30]['mat'][0])} 點（p {R[k]['p_size']:.3f}）" for k in surv)
            + f"；{P1['m']} 個裡純靠運氣，p ≤ 0.10 的大約會有 {P1['m'] * 0.1:.1f} 個。")
    else:
        v.append(f"錯位檢查加多重檢定（10%）後，沒有一個指標的大小本事留下來（純靠運氣 p ≤ 0.10 的大約 {P1['m'] * 0.1:.1f} 個）。"
                 "以這些定義和資料，沒看到「比現在的波動多說了什麼」的大小訊息。")
    # ADDED AFTER SEEING THE RESULTS: which side the size difference comes from
    parts = []
    for k in size_ok:
        h = R[k]["h"][30]
        mu, md, mr = h["mat"][1], h["mat"][2], h["mat"][3]
        uci, dci = h["mat_ci"][:, 1], h["mat_ci"][:, 2]
        one = ("漲的那邊" if (uci[0] > 0 and dci[1] < 0) else "跌的那邊" if (dci[0] > 0 and uci[1] < 0)
               else "兩邊都有" if (uci[0] > 0 and dci[0] > 0) else "分不清是哪一邊")
        parts.append(f"{_name(k)}（大小 {_pp(h['mat'][0])}：最大漲幅 {_pp(mu)}、最大跌幅 {_pp(md)}、報酬 {_pp(mr)}，來自{one}）")
    if parts:
        v.append("看完結果才加的拆解（量到的，配對後）：" + "；".join(parts)
                 + "。大小的差如果只來自一邊，它其實是在說方向，不是「不分漲跌的大小」。")
    v.append(f"量到的：波動會延續——過去 30 天最大幅度和之後 30 天最大幅度的相關 {persist['size']:+.2f}，過去 20 天波動和之後 30 天波動的相關 "
             f"{persist['rv']:+.2f}，所以沒配對看到的「大小本事」大多可以只是延續。")
    if P3.get("pick"):
        good = [k for k in RULES if P3["sup"][k]]
        v.append(f"用 2019 年前的資料選出的 S 是「{_name(P3['pick'])}」。五條規則 2019 年後的公平考試："
                 + "；".join(f"{RSHORT[k]} 排名 {_sh(P3['s'][k]['low'])}（均線 {_sh(P3['s']['ma50']['low'])}，"
                             f"波動對照 {_sh(P3['tw'][k]['low'])}）、錯位 {_sh(P3['placebo'][k])}" for k in RULES)
                 + "。" + (f"支持的：{'、'.join(RSHORT[k] for k in good)}。" if good else "沒有一條同時過四個標準。")
                 + "五條裡純靠運氣過錯位 5% 的大約 0.25 條。")
    v.append("推測：如果大小訊息大多只是「波動會延續」，那它最多能幫忙決定拿多少，不會讓順勢規則抓得更準；這一點這份資料分不開，"
             "因為槓桿、選擇權這些指標只有 2019–2021 年以後、單一交易所的資料。")
    verdict = "".join(v)

    # ---------- now ----------
    now = []
    if P3.get("pick"):
        pos = P3["pos"]
        now.append({"label": "比特幣：50 日均線｜Z1｜Z3",
                    "value": f"{_state(pos['ma50'][-1])}｜{_state(pos['z1'][-1])}｜{_state(pos['z3'][-1])}"})
        sv = P3["S"].iloc[-1]
        tv = rank["rv20"].iloc[-1]
        now.append({"label": f"S（{_name(P3['pick'])}）今天的位置",
                    "value": (f"{sv:.0f}（高的三分之一 > 67、低的三分之一 < 33）；過去 20 天波動排名 {tv:.0f}" if _ok(sv)
                              else "今天沒有值")})
    for k in show:
        r_ = rank[k].iloc[-1]
        sign = 1.0 if R[k]["h"][30]["mat"][0] >= 0 else -1.0
        tag = "" if k in surv else "（不是通過多重檢定的）"
        now.append({"label": f"{_name(k)}{tag}", "value": _today_word(float(r_) if _ok(r_) else np.nan, sign)})
    now.append({"label": "最後一根收完的日線", "value": f"{_d(close.index[-1])}，比特幣 {close.iloc[-1]:,.0f} 美元"})

    # ---------- notes ----------
    notes = [
        "為什麼做這個：之前約 100 個被放棄的結果只問了「之後會不會漲」。有幾個留下的數字比較像在說「之後會動多大」：期貨溢價每年 20% 以上之後 30 天內"
        "曾跌 10% 以上是 77%（全部日子 46%），但被擋掉的日子合計還漲了 +77%；資金費率最高一成 47% 對 20%；未平倉量在 4 次大跌前 4 次都高；"
        "交易所進出量在頂部附近排名 87、平常 42–75，在大漲前也高；「買賣壓一起升」是大漲前（23 次裡 12 次）和大跌前（27 次裡 14 次）最常見的樣子。",
        "量到的：配對的格子是「過去 20 天波動十等分 × 過去 30 天漲跌十等分」，十等分的切點用 2012 年起全部日子定一次；指標只在同一格裡比。"
        "配對只能拿掉這兩個已知的東西，拿不掉更長期的狀態（例如半年動能、牛熊）。",
        f"量到的：相鄰的日子看的是幾乎同一段 30 天（60 天）行情，所以天數不是獨立的例子；90% 區間用 60 天一塊的重抽，已經把這點算進去一部分，"
        f"但 2019 年後才有的指標（資金費率、未平倉量、選擇權、多空比）真正獨立的 30 天窗口只有大約 70–90 個。",
        "期貨溢價只有 Deribit、資金費率／未平倉量／多空比只有幣安（2019-09／2020-09 以後）、隱含波動只有 Deribit DVOL（2021 起）；"
        "交易所進出和存量是用 Coin Metrics 今天的交易所地址標籤往回套的，早年漏得多；老幣移動不一定是賣；現貨成交量在 2022-07～2023-04 幣安零手續費期間被拿掉。",
        "「過去 20 天波動」（安靜）本身就是配對用的變數，所以它配對後的差只剩格子裡面的細微差別，這一列主要是給其他指標當參照。",
        "第三部分的 S 和對照組都用比特幣的狀態，套到其他幣和 MSTR 上是把它當成整個市場的狀態；其他幣自己的指標沒有用。",
        f"第三部分錯位檢查為了在時間內算完，一般人做法每個收盤時間只模擬 {PL_SIMS} 次（真的規則也用同樣方式再算一次比）。"
        f"這次整份研究算了約 {secs / 60:.1f} 分鐘（其中公平考試 {P3.get('secs', 0) / 60:.1f} 分鐘）。",
        "方向和定義都是看結果之前寫好的；沒有調任何門檻。過去的結果不保證未來；這裡只是描述，不是買賣建議。",
    ]
    try:
        allr = json.loads((common.CACHE / "_studies2.json").read_text(encoding="utf-8"))
        vs = next((x for x in allr.get("studies", []) if isinstance(x, dict) and x.get("id") == "vol_sizing"), None)
        if vs and vs.get("verdict"):
            notes.append("對照之前的 vol_sizing 研究（只依過去波動決定拿多少，完全不看漲跌）存下的結論：「" + vs["verdict"] + "」"
                         "這份研究的 Z3／Z3b 要贏的就是同一個想法（改用過去 20 天波動排名的對照組）。")
    except Exception:  # noqa: BLE001 - the note is optional
        pass
    if D["dropped"]:
        notes.append("沒有算進去的指標：" + "；".join(f"{_name(k)}（{why}）" for k, why in D["dropped"].items()))
    notes += D["problems"]

    sources = [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": classic.BCI_PRICE},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP）", "url": px.BINANCE_HOSTS[0] + "/api/v3/klines"},
        {"name": "Coin Metrics 社群 API（FlowInExNtv、FlowOutExNtv、SplyExNtv、FeeTotNtv、PriceUSD）", "url": sd.CM_URL},
        {"name": "Open Bitcoin Metrics（放 155 天以上的幣被花掉）", "url": ocs.OBM_BASE.format(pbm.OBM_ID)},
        {"name": "DefiLlama 穩定幣總量", "url": sd.LLAMA},
        {"name": "Farside 美國現貨比特幣 ETF 每日流量", "url": flows2.FARSIDE["BTC"][0]},
        {"name": "Coinbase BTC-USD 日線", "url": km.COINBASE_URL},
        {"name": "Bitstamp BTC/USD 日線", "url": bf.BITSTAMP.format(pair="btcusd")},
        {"name": "幣安 U 本位合約資金費率", "url": cy.FAPI + "/fapi/v1/fundingRate?symbol=BTCUSDT"},
        {"name": "幣安期貨歷史資料庫（每日 metrics：未平倉量、多空比）", "url": lr.ARCHIVE.format(s="BTCUSDT", d="YYYY-MM-DD")},
        {"name": "Deribit 季度期貨小時線", "url": bf.DERIBIT},
        {"name": "Deribit DVOL 波動指數", "url": vd.DVOL_URL + "?currency=BTC&resolution=1D"},
        {"name": "Yahoo Finance：MSTR 日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]
    return {
        "id": ID,
        "title": "指標說的是「動多大」還是「往哪走」",
        "question": ("很多指標只被問過「之後會不會漲」，答案是不會就被丟掉。它們會不會其實在說「之後會動多大」（不分漲跌）？"
                     "如果會，這種大小訊息能不能幫 50 日均線決定停損多緊、拿多少、要不要等？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
