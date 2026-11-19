"""Historical study "exit_forces": at the moment the main signal ma50 (hold bitcoin while the daily close is above its
50-day average) turns off, does the state of the BUYING AND SELLING FORCES tell a whipsaw from a real decline?

Why (written before any run). whipsaw_anatomy counted ma50's 148 exits since 2010: 70% were whipsaws (ma50 back on
within 20 days at a higher close; sitting through gained +4.8% on average) and 18% were real declines (-15% within 60
days; sitting through cost -18.4%), and ten price-only descriptors measured at the exit could not tell the two apart.
The owner's idea: who is selling at that moment, whether selling is growing or fading, and whether buyers are absorbing
it may tell them apart where price alone cannot. Ground rules followed here: no force idea is presumed naive; each
parameter below is a hypothesis with its logic, tested alone as well as in one fixed combination; a failed test is
written as "not seen with this definition and this data" with what the definition may be missing (open leads), never
as a refutation; and the evidence bar is strict: a parameter must add something BEYOND price (the ten price
descriptors), and "do not exit + a stop" must beat the same stop used WITHOUT force information (the stop alone might
do the work).

Data (free, key-less, fixed user agent "btc-research-dashboard"; a refused endpoint is named, never worked around;
every download goes to this study's own stores cache/_store/exit_forces_*; other studies' stores are only READ, except
the shared fair-exam loaders clock_robustness._hourly and main_signal_check._load_other that every study uses):
  BTC daily closes: te._daily_close (00:00 UTC; blockchain.info before Binance 2017-08). Exits, labels, price
    descriptors: whipsaw_anatomy's (_exits, _label, _desc), descriptor (i) from this study's own Binance daily store.
  Coin Metrics community API, asset btc (and eth if free): FlowInExNtv (coins sent to exchange-tagged addresses per
    day), SplyExNtv (coins held on them). Data day D is used from the close of D+1.
  Open Bitcoin Metrics CSV data files only (never cloned, installed or run): obm_spent_value_ge365d_btc_daily and
    obm_spent_value_ge155d_btc_daily (the second for a report-only robustness line), parsed as dates and numbers only.
    Data day D is used from the close of D+7 (old_coin_selling's lag).
  Binance spot 1-day klines BTCUSDT (and ETHUSDT): quote volume and taker-buy quote volume (own store).
  Binance spot 1-hour klines BTCUSDT (and ETHUSDT): close, quote, taker-buy quote; price_impact's _measure (own store;
    seeded read-only from price_impact's store).
  Coinbase Exchange daily candles BTC-USD (and ETH-USD), close at the same 00:00 UTC as Binance (own store).
  Binance USDT-margined BTCUSDT funding settlements from 2019-09 (carry._funding; own store as fallback).
  Binance futures daily metrics archive (data.binance.vision) BTCUSDT: open interest at 00:00 UTC, from 2020-09; read
    from ls_ratio's store (read only) and days it lacks fetched into this study's own store.
  DefiLlama total stablecoin supply (supply_demand._llama_fetch; own store), used from 2019-07-01; point D used from
    the close of D+1.
  Fair exam inputs as whipsaw_anatomy: Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance) and MSTR (Yahoo).

1. Events and outcome. Events = every close e where BTC's ma50 turns off (whipsaw_anatomy._exits). Labels WHIPSAW /
   REAL / OTHER / UNKNOWN = whipsaw_anatomy._label (description only).
   Hard stop for exit e: the first close t > e with close <= 0.92 x close(e) OR close below the lowest close of
   e-20 .. e-1 (if fewer than 20 earlier closes, the 8% part only).
   PAYOFF(e) = return of staying in from close e to the first of: a close where ma50 is on again (hand over; + 2 x
   0.10% because ma50 sold at e and buys back there, whipsaw_anatomy's arithmetic), the stop close, or close e+60;
   MINUS the return of cash over the same span (taken as 0). Exits whose span has not ended are UNKNOWN for PAYOFF and
   left out of every statistic (listed in "today").
2. Force parameters at the exit, all known at the exit close (decision candle D = the exit date), each giving "stay"
   (expect whipsaw) or "leave" (expect real decline) or undefined (missing history -> left out of that parameter's
   counts; inside a rule undefined counts as "leave"). Direction fixed by logic:
   Tier 1 (main family, Holm over 3):
   X1 panic deposits: S10 = 10-day sum of FlowInExNtv (data days D-10 .. D-1); z = (S10 - mean) / std of S10 over the
      365 previous data days (>= 330 values). STAY when z >= 1 (forced, liquidity-driven selling tends to reverse).
   X2 old holders selling: O14 = 14-day sum of OBM spent >= 365d (data days ending D-7); LEAVE when O14 is above the
      80th percentile of its 365 previous values (>= 330) (long holders distributing); otherwise STAY.
   X3 inventory building: LEAVE when SplyExNtv(D-1) > SplyExNtv(D-31) (exchanges' coin balance higher than 30 days
      earlier: coins being stocked up for sale); otherwise STAY.
   Tier 2 (second family, Holm over 3; Binance-based, from 2017-08):
   X4 buyers absorbing: share5 = sum taker-buy quote / sum quote over candles D-4 .. D; STAY when share5 >= the median
      of its 365 previous values (>= 180).
   X5 selling does not move price much: price_impact's DOWN impact (price lost per dollar of net selling, 14 days of
      hourly bars to the close of D); STAY when below the median of its 365 previous values (>= 180).
   For BTC, X4 and X5 are undefined on decision days 2022-07-08 .. 2023-04-04 and those days are left out of their
      medians (Binance zero-fee BTC trading distorted taker volume; price_impact's ZF_DROP).
   X6 US buyers: Coinbase premium C = Coinbase close / Binance close - 1 (candle D); STAY when its 7-day mean (>= 5 of
      7 days) > 0.
   Tier 3 (2019-20+, descriptive only, no claims):
   X7 funding: daily mean of the settlements in (D 00:00, D+1 00:00]; f7 = 7-day mean (>= 6 days). STAY when f7 <= 0;
      also reported: HOT = f7 > 0 and above the median of its 365 previous values (>= 180) (longs not yet flushed).
   X8 open interest: STAY when OI at 00:00 of D (= the close of candle D-1) < OI at 00:00 of D-5 (one day of lag kept
      because the archive is published later).
   X9 stablecoin supply: daily log change of total supply (changes beyond +-15% set to 0, stablecoin_supply's coverage
      rule), 30-day sum over data days D-30 .. D-1; STAY when > 0.
   Combination C1 (fixed now): STAY when X1 says stay AND X2 does not say leave AND X3 does not say leave (all three
      defined; otherwise leave).
3. Per parameter (BTC exits with the parameter defined and PAYOFF known): counts, STAY share, WHIPSAW / REAL shares
   and mean / median PAYOFF for STAY and for LEAVE; DIFF = mean PAYOFF(STAY) - mean PAYOFF(LEAVE).
   Clusters: exits sorted by date, a new cluster starts when an exit comes more than 30 days after the previous one.
   Cluster bootstrap (2,000 draws, fixed seed): 90% intervals for mean PAYOFF(STAY) and DIFF; one-sided p = share of
   draws with mean PAYOFF(STAY) <= 0.
   Beyond price: OLS of PAYOFF on [1, STAY, the ten price descriptors a..j] (a descriptor's missing values filled with
   its sample mean plus a 0/1 missing indicator when it has any); the STAY coefficient with a 90% cluster-bootstrap
   interval (1,000 draws). Regression, not matching (N too small for exact matching on ten descriptors).
   Halves: DIFF on exits before 2020-01-01 and from 2020-01-01.
   Placebo: the parameter's daily STAY series circularly shifted by k (uniform in [180, n-180] days over its defined
   span) against the fixed exit dates, 200 draws (fixed seed); placebo share = share of draws whose DIFF >= the real.
   Pre-registered "supported", all of: STAY group >= 30% of the exits; its REAL share <= 8%; its mean PAYOFF > 0 with
   the 90% interval excluding zero; DIFF > 0 in both halves; the STAY coefficient after the price control > 0 with
   its 90% interval excluding zero; placebo share <= 5%; and the one-sided p rejected by Holm at 5% within its tier
   (Tier 1: X1-X3; Tier 2: X4-X6). Tier 3 descriptive. C1: the same bar (a single pre-specified test, no Holm).
   Also (report): Tier 1 restricted to exits from 2017-01-01 (exchange labels are applied backwards, early exits are
   least reliable); X2 rebuilt from spent >= 155d (robustness line).
4. More data: ETH's own ma50 exits (ETH closes from Binance) with ETH's own Coin Metrics flows (X1, X3) if the community
   API serves them for eth (200, not 403), Binance ETHUSDT taker data (X4, X5) and Coinbase ETH-USD (X6); no X2 for
   ETH. Reported separately (30-day clusters) and pooled with BTC (clusters = calendar weeks, because BTC and ETH exits
   share dates; price control adds an asset dummy). Description only; no Holm.
5. Rules. V1 = hold when ma50 is on OR (the most recent ma50 exit e was within the last 60 closes - decisions at closes
   e .. e+59 - AND C1 said STAY at e AND no close after e has hit the hard stop). V0 (stop-only twin) = the same with
   STAY replaced by "always". S1, S2, S3 = the same with STAY = X1 stay / X2 not leave / X3 not leave.
   Fair exam = whipsaw_anatomy's _Exam (main_signal_check's four checks) with first month 2015-01: BTC daily monthly
   starts 2015-01 .. 2023-12 (0.10%); clock starts 2018-01 .. 2023-12 x 24 closing hours; the same as an ordinary
   person (10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock, main_signal_check's seeds);
   ETH SOL BNB XRP MSTR from their first valid month to three years before today, each with its OWN ma50, exits and
   stop, and BTC's force state as a market-wide state (other coins' UTC closes use the same candle; MSTR's date D and
   a clock close on UTC date E use candle D-1 / E-1). ma50 re-scored on the same starts (numbers reported). Ranking
   score = worst of the four "beats holding" shares with main_signal_check's eligibility; median deepest loss = median
   over the clock textbook cells.
   Placebo for V1 and S1-S3: the rule's daily STAY series circularly shifted by k in [180, n-180] days (from its first
   defined day), 200 draws; the whole exam rebuilt with 20 ordinary-person simulations per clock (same seeds) and the
   real rule re-scored the same way; placebo share = share of draws whose ranking score is at least the real one's.
   "Supported" for a rule, all of: ranking score >= ma50's; median deepest loss no more than 5 points worse than
   ma50's; placebo share <= 5%; ranking score above V0's. Definitions fixed before any run, so the whole history is
   scored once; the exam restricted to starts from 2020-01 is reported as a stability line.
6. Report only: calendar years where V1 gains or loses versus ma50 (BTC daily, 0.10%); every BTC exit in 2018 and 2022
   with what each parameter said; exits where X1, X2, X3 all said stay / all said leave / disagreed; open leads.
Look-ahead check: all raw inputs cut at 2019-06-30, 2021-06-30, 2024-06-30 (data day / candle <= cut); no parameter
   value on a decision day <= cut may change.
Honesty: Coin Metrics applies today's exchange address labels backwards; OBM is one author's 2026 dataset and coins
   moving are not always coins sold; taker, funding and open-interest data are Binance only; exits cluster in choppy
   stretches, so they are not 148 independent trials; thresholds (z >= 1, 80th percentile, 30 / 365 days, medians,
   8% / 20-day stop, 60 days) were fixed by reasoning before the run, no alternatives tried. A supported rule would
   only become a candidate for the daily shadow log. Not advice.
Today: "比特幣：50 日均線｜V1｜V0" ("現金｜現金｜現金" style), the most recent ma50 exit, what each parameter said
   there, and whether the 60-day window or the stop has ended.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import carry as cy
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import order_flow as of
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import whipsaw_anatomy as wa

COST, SIMS, SEED = msc.COST, msc.SIMS, msc.SEED
DAY = pd.Timedelta(days=1)
WINDOW, STOP_PCT, LOW_DAYS = 60, 0.92, 20
Z_DAYS, ZWIN, ZMIN, Z_BAR = 10, 365, 330, 1.0
OLD_DAYS, OLD_LAG, OLD_PCT = 14, 7, 0.80
BAL_DAYS = 30
TAKER_DAYS, MED_WIN, MED_MIN = 5, 365, 180
PREM_DAYS, FUND_DAYS, OI_DAYS, STABLE_DAYS = 7, 7, 5, 30
STABLE_FROM, JUMP = pd.Timestamp("2019-07-01"), 0.15
CLUSTER_GAP = 30
BOOT, BOOT_REG, DRAWS, SHIFT_MIN, PL_SIMS = 2000, 1000, 200, 180, 20
HALF, LATE = pd.Timestamp("2020-01-01"), pd.Timestamp("2017-01-01")
BAR_STAY, BAR_REAL, PLACEBO_BAR, ALPHA, LOSS_SLACK = 0.30, 0.08, 0.05, 0.05, 5.0
FIRST_MONTH, STABLE_MONTH = pd.Timestamp(msc.DAILY_STARTS[0]), pd.Timestamp("2020-01-01")
ZF = (pd.Timestamp(pi.ZF_DROP[0]), pd.Timestamp(pi.ZF_DROP[1]))
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")
KI = {"ma50": 1, "v1": 130, "v0": 131, "s1": 132, "s2": 133, "s3": 134}

CM_URL = sd.CM_URL
CM_METRICS = ("FlowInExNtv", "SplyExNtv")
CM_START = {"btc": "2009-01-03", "eth": "2015-07-30"}
OBM_BASE = "https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/{0}/{0}.csv"
OBM_IDS = {"ge365": "obm_spent_value_ge365d_btc_daily", "ge155": "obm_spent_value_ge155d_btc_daily"}
CB_URL = "https://api.exchange.coinbase.com/products/{}/candles"
FUND_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINES_URL = "https://api.binance.com/api/v3/klines"
OI_FIRST = pd.Timestamp(lr.FIRST_DAY["BTCUSDT"])
FIRST_1H = {"BTCUSDT": pi.FIRST_MS, "ETHUSDT": 1502942400000}
HOUR_MS = 3_600_000

P1, P2, P3 = ("x1", "x2", "x3"), ("x4", "x5", "x6"), ("x7", "x8", "x9")
PARAMS = P1 + P2 + P3
TIER = {**{k: 1 for k in P1}, **{k: 2 for k in P2}, **{k: 3 for k in P3}}
PNAME = {"x1": "X1 恐慌存入", "x2": "X2 老手在賣", "x3": "X3 交易所在囤貨", "x4": "X4 買方在接",
         "x5": "X5 賣壓推不動價格", "x6": "X6 美國買盤", "x7": "X7 資金費率", "x8": "X8 未平倉量", "x9": "X9 穩定幣",
         "c1": "C1 組合"}
PDEF = {
    "x1": "前 10 天存進交易所的比特幣，比過去一年的平常水準高至少 1 個標準差 → 說「留」",
    "x2": "前 14 天放了 1 年以上的老幣被花掉的量，高過過去一年的八成日子 → 說「走」，否則「留」",
    "x3": "交易所手上的比特幣比 30 天前多 → 說「走」，否則「留」",
    "x4": "前 5 天主動買進（吃賣單買）的成交金額占比，不低於過去一年的中位數 → 說「留」",
    "x5": "前 14 天每 1 美元淨賣出讓價格掉多少，低於過去一年的中位數 → 說「留」",
    "x6": "Coinbase（美國）價格比幣安高，7 天平均溢價 > 0 → 說「留」",
    "x7": "永續合約資金費率 7 天平均 ≤ 0（做多的人不再付錢）→ 說「留」",
    "x8": "合約未平倉量比 5 天前少（槓桿已經在退）→ 說「留」",
    "x9": "穩定幣總量 30 天在增加（場外有新錢）→ 說「留」",
    "c1": "X1 說「留」，而且 X2、X3 都沒有說「走」→ 說「留」",
}
PWHY = {
    "x1": "被迫、急著要現金的賣，常常賣完就反彈",
    "x2": "老手在出貨，賣壓會延續",
    "x3": "交易所在累積待賣的幣，賣壓還在後面",
    "x4": "有人在接賣單，賣壓被吸收",
    "x5": "賣了很多價格卻掉不多，代表下面有買盤",
    "x6": "美國的買家（含機構）還願意付比較高的價格",
    "x7": "做多的人已經洗掉，沒有一大群人等著被迫平倉",
    "x8": "槓桿已經退場，被迫賣的人變少",
    "x9": "場外有新錢進來，之後有買盤",
    "c1": "恐慌賣壓出現，但老手沒在出貨、交易所沒在囤貨",
}
PDATA = {"x1": "Coin Metrics（2011 年起）", "x2": "Open Bitcoin Metrics（2010 年起）", "x3": "Coin Metrics（2011 年起）",
         "x4": "幣安現貨（2018 年起）", "x5": "幣安 1 小時線（2018 年起）", "x6": "Coinbase＋幣安（2017-08 起）",
         "x7": "幣安永續合約（2019-09 起）", "x8": "幣安合約存檔（2020-09 起）", "x9": "DefiLlama（2019-07 起）",
         "c1": "X1＋X2＋X3"}
KIND = wa.KIND
NAME = {"ma50": "50 日均線（基準）",
        "v1": "V1：50 日均線；下車那天組合 C1 說「留」就不賣，最多再抱 60 天（有停損）",
        "v0": "V0（只有停損的對照）：50 日均線；每次下車都不賣，最多再抱 60 天（同樣的停損）",
        "s1": "S1：同 V1，但只看 X1 恐慌存入", "s2": "S2：同 V1，但只看 X2 老手在賣",
        "s3": "S3：同 V1，但只看 X3 交易所在囤貨"}
SHORT = {"ma50": "50 日均線", "v1": "V1", "v0": "V0", "s1": "S1", "s2": "S2", "s3": "S3"}
RULE_FLAG = {"v1": "c1", "s1": "x1", "s2": "x2", "s3": "x3"}

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- small stores ----------

def _path(name: str):
    return common.STORE / f"exit_forces_{name}"


def _jread(name: str):
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no store yet
        return None


def _jwrite(name: str, data) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        _path(name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store only saves time
        pass


def _stored(name: str, fetch):
    """Fresh download; on failure this study's earlier stored copy (named in the notes)."""
    try:
        got = fetch()
        _jwrite(name, got)
        return got, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        got = _jread(name)
        if got is None:
            raise RuntimeError(f"{name} 取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        return got, f"{name} 這次取不到（{type(exc).__name__}），用之前存下來的資料。"


def _csv_read(name: str, width: int) -> dict:
    rows = {}
    try:
        for line in _path(name).read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != width + 1 or not parts[0].isdigit():
                continue
            v = [float(x) for x in parts[1:]]
            if all(math.isfinite(x) for x in v):
                rows[int(parts[0])] = v
    except Exception:  # noqa: BLE001
        return {}
    return rows


def _csv_write(name: str, rows: dict, cols) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{ms}," + ",".join(repr(x) for x in v) for ms, v in sorted(rows.items()))
        _path(name).write_text("ms," + ",".join(cols) + "\n" + body + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# ---------- downloads ----------

def _cm_fetch(asset: str) -> dict:
    reply = common.get_json(CM_URL, {"assets": asset, "metrics": ",".join(CM_METRICS), "frequency": "1d",
                                     "start_time": CM_START[asset], "page_size": 10000}, timeout=90)
    rows = list(reply.get("data") or [])
    for _ in range(10):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=90)
        rows += list(reply.get("data") or [])
    out = {}
    for row in rows:
        day = str(row.get("time", ""))[:10]
        rec = {}
        for m in CM_METRICS:
            try:
                v = float(row.get(m))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v >= 0:
                rec[m] = v
        if day and rec:
            out[day] = rec
    if len(out) < 1500:
        raise RuntimeError(f"Coin Metrics {asset} 只回傳 {len(out)} 天")
    return out


def _cm_status(asset: str) -> int:
    """HTTP status of a one-row request (200 = free; 403 / 400 = not in the community tier)."""
    try:
        r = requests.get(CM_URL, params={"assets": asset, "metrics": ",".join(CM_METRICS), "frequency": "1d",
                                         "page_size": 1}, headers=common.HEADERS, timeout=30)
        time.sleep(common.PAUSE)
        return int(r.status_code)
    except Exception:  # noqa: BLE001
        return 0


def _obm_parse(text: str) -> dict:
    """Untrusted CSV text -> {date: number}; only dates and finite non-negative numbers are kept."""
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if "date" not in frame.columns or "value" not in frame.columns:
        raise RuntimeError("OBM 欄位不對")
    value = pd.to_numeric(frame["value"], errors="coerce")
    day = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    good = day.notna() & value.notna() & np.isfinite(value.fillna(0.0)) & (value >= 0) & (value <= 2.2e7)
    out = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(day[good], value[good])}
    if len(out) < 3000:
        raise RuntimeError(f"OBM 只有 {len(out)} 筆可用")
    return out


def _obm_fetch() -> dict:
    out = {}
    for key, sid in OBM_IDS.items():
        reply = requests.get(OBM_BASE.format(sid), headers=common.HEADERS, timeout=90)
        reply.raise_for_status()
        if len(reply.content) > 20_000_000:
            raise RuntimeError(f"{sid} 檔案太大")
        out[key] = _obm_parse(reply.text)
        time.sleep(common.PAUSE)
    return out


def _daily_klines(symbol: str) -> tuple[pd.DataFrame, str | None]:
    """Binance spot 1-day klines (of.DAILY_COLS), own store, topped up with closed candles only."""
    name = f"1d_{symbol}.csv"
    rows = _csv_read(name, len(of.DAILY_COLS))
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - 86_400_000:
        try:
            if of._fetch("spot", symbol, "1d", rows, of.DAILY_COLS):
                _csv_write(name, rows, of.DAILY_COLS)
        except Exception as exc:  # noqa: BLE001
            problem = (f"幣安 {symbol} 日線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    if len(rows) < 400:
        raise RuntimeError(f"{symbol} 日線只有 {len(rows)} 筆")
    f = pd.DataFrame([rows[k] for k in sorted(rows)], columns=of.DAILY_COLS,
                     index=pd.to_datetime(sorted(rows), unit="ms").normalize())
    f = f[~f.index.duplicated(keep="last")]
    f = f[f.index < pd.Timestamp(common.today_utc())]
    return f.asfreq("D"), problem


def _hourly(symbol: str) -> tuple[dict, str | None]:
    """{open ms: [close, quote, taker-buy quote]} of closed 1-hour bars, own store (BTC seeded from price_impact's)."""
    name = f"1h_{symbol}.csv"
    rows = _csv_read(name, 3)
    if not rows and symbol == "BTCUSDT":
        rows = pi._read()  # read only
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - HOUR_MS:
        start = (max(rows) + 1) if rows else FIRST_1H[symbol]
        added = 0
        try:
            for _ in range(150):
                page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 1000,
                                                      "startTime": start})
                if not page:
                    break
                now_ms = int(time.time() * 1000)
                for r in page:
                    if int(r[6]) < now_ms:
                        rows[int(r[0])] = [float(r[4]), float(r[7]), float(r[10])]
                        added += 1
                if len(page) < 1000:
                    break
                start = int(page[-1][0]) + 1
                time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            problem = (f"幣安 {symbol} 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
        if added:
            _csv_write(name, rows, ("close", "quote", "tbq"))
    return rows, problem


def _coinbase(product: str) -> tuple[pd.Series, str | None]:
    """Finished Coinbase daily closes from 2017-08-01, own store."""
    name = f"cb_{product.lower()}.json"
    stored = _jread(name) or {}
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - pd.Timedelta(days=5)) if stored else pd.Timestamp("2017-08-01")
    problem = None
    try:
        while start < today:
            end = min(start + pd.Timedelta(days=299), today)
            page = km._get(CB_URL.format(product), {"granularity": 86400,
                                                    "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                                                    "end": end.strftime("%Y-%m-%dT00:00:00Z")}).json()
            for row in page:
                day = pd.Timestamp(int(row[0]), unit="s")
                if day < today and float(row[4]) > 0 and int(row[0]) % 86400 == 0:
                    stored[_d(day)] = float(row[4])
            start = end + pd.Timedelta(days=1)
        _jwrite(name, stored)
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        problem = f"Coinbase {product} 這次沒有更新成功（{type(exc).__name__}），用之前存下來的資料（到 {max(stored)}）。"
    s = pd.Series(stored, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), problem


def _funding() -> tuple[pd.Series, str | None]:
    try:
        s = cy._funding("BTCUSDT")
        _jwrite("funding.json", {str(int(t.value // 1_000_000)): float(v) for t, v in s.items()})
        return s, None
    except Exception as exc:  # noqa: BLE001
        got = _jread("funding.json")
        if not got:
            raise
        s = pd.Series({pd.Timestamp(int(k), unit="ms"): float(v) for k, v in got.items()}).sort_index()
        return s, f"幣安資金費率這次取不到（{type(exc).__name__}），用之前存下來的資料。"


def _oi(budget_s: float = 240) -> tuple[pd.Series, str | None]:
    """BTCUSDT open interest (coins) at 00:00 UTC per file date: ls_ratio's store (read only) + own store."""
    base = lr._read_store("BTCUSDT").get("rows", {})
    own = _jread("oi.json") or {}
    have = {d: r[4] for d, r in base.items() if r and len(r) > 4 and r[4] is not None}
    have.update({d: v for d, v in own.items() if v is not None})
    today = pd.Timestamp(common.today_utc())
    todo = [d for d in pd.date_range(OI_FIRST, today - DAY, freq="D") if d.strftime("%Y-%m-%d") not in have]
    todo = [d for d in todo if d >= today - pd.Timedelta(days=400)] if len(have) > 500 else todo
    problem, t0, added = None, time.time(), 0
    try:
        for d in todo:
            if time.time() - t0 > budget_s:
                problem = "幣安未平倉量存檔這次下載超過時間上限，先停住，下次接著補。"
                break
            day = d.strftime("%Y-%m-%d")
            reply = requests.get(lr.ARCHIVE.format(s="BTCUSDT", d=day), headers=common.HEADERS, timeout=30)
            if reply.status_code == 404:
                continue
            reply.raise_for_status()
            row = lr._midnight_row(reply.content, day)
            if row is not None and row[4] is not None:
                own[day] = float(row[4])
                have[day] = float(row[4])
                added += 1
            time.sleep(lr.DL_PAUSE)
    except Exception as exc:  # noqa: BLE001
        problem = f"幣安未平倉量存檔這次沒有全部補齊（{type(exc).__name__}），用已有的資料。"
    if added:
        _jwrite("oi.json", own)
    s = pd.Series(have, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), problem


def _load() -> dict:
    problems = []
    close = te._daily_close()
    other, more = msc._load_other()
    problems += more
    raw = {"btc": {}, "eth": {}}
    cm_status = {a: _cm_status(a) for a in ("btc", "eth")}
    for asset in ("btc", "eth"):
        if cm_status[asset] != 200 and asset == "eth":
            continue
        try:
            got, prob = _stored(f"cm_{asset}.json", lambda a=asset: _cm_fetch(a))
            if prob:
                problems.append(prob)
            f = pd.DataFrame.from_dict(got, orient="index")
            f.index = pd.to_datetime(f.index)
            raw[asset]["cm"] = f.sort_index()
        except Exception as exc:  # noqa: BLE001
            if asset == "btc":
                raise
            problems.append(f"Coin Metrics 以太幣交易所資料取不到（{type(exc).__name__}）。")
    got, prob = _stored("obm.json", _obm_fetch)
    if prob:
        problems.append(prob)
    for key in OBM_IDS:
        s = pd.Series(got[key], dtype=float)
        s.index = pd.to_datetime(s.index)
        raw["btc"][f"obm_{key}"] = s.sort_index()
    for asset, sym, prod in (("btc", "BTCUSDT", "BTC-USD"), ("eth", "ETHUSDT", "ETH-USD")):
        try:
            k, prob = _daily_klines(sym)
            problems += [prob] if prob else []
            raw[asset]["kl"] = k
            h, prob = _hourly(sym)
            problems += [prob] if prob else []
            raw[asset]["hourly"] = h
            c, prob = _coinbase(prod)
            problems += [prob] if prob else []
            raw[asset]["cb"] = c
        except Exception as exc:  # noqa: BLE001
            if asset == "btc":
                raise
            problems.append(f"以太幣的幣安／Coinbase 資料取不到（{type(exc).__name__}）。")
    try:
        raw["btc"]["funding"], prob = _funding()
        problems += [prob] if prob else []
    except Exception as exc:  # noqa: BLE001
        problems.append(f"幣安資金費率取不到（{type(exc).__name__}），X7 沒有算。")
    try:
        raw["btc"]["oi"], prob = _oi()
        problems += [prob] if prob else []
    except Exception as exc:  # noqa: BLE001
        problems.append(f"幣安未平倉量取不到（{type(exc).__name__}），X8 沒有算。")
    try:
        got, prob = _stored("llama.json", sd._llama_fetch)
        problems += [prob] if prob else []
        s = pd.Series(got, dtype=float)
        s.index = pd.to_datetime(s.index)
        raw["btc"]["llama"] = s.sort_index()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"DefiLlama 穩定幣總量取不到（{type(exc).__name__}），X9 沒有算。")
    eth_close = other["ETH"][0] if "ETH" in other else None
    return {"close": close, "other": other, "raw": raw, "eth_close": eth_close, "cm_status": cm_status,
            "problems": problems}


# ---------- force parameters ----------

def _trail_med(x: pd.Series) -> pd.Series:
    return x.shift(1).rolling(MED_WIN, min_periods=MED_MIN).median()


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    return cond.astype(float).where(valid)


def _forces(raw: dict, index: pd.DatetimeIndex, zero_fee: bool) -> pd.DataFrame:
    """Daily parameter values on decision candles (index). 1 = says stay, 0 = says leave, NaN = undefined."""
    cal = pd.date_range(min(index[0], pd.Timestamp("2009-01-03")), index[-1], freq="D")
    out = pd.DataFrame(index=cal)
    cm = raw.get("cm")
    if cm is not None and "FlowInExNtv" in cm:
        fin = cm["FlowInExNtv"].reindex(cal)
        s10 = fin.rolling(Z_DAYS, min_periods=Z_DAYS).sum()
        mu = s10.shift(1).rolling(ZWIN, min_periods=ZMIN).mean()
        sdv = s10.shift(1).rolling(ZWIN, min_periods=ZMIN).std()
        z = ((s10 - mu) / sdv.where(sdv > 0)).shift(1)
        out["z1"] = z
        out["x1"] = _flag(z >= Z_BAR, z.notna())
    if cm is not None and "SplyExNtv" in cm:
        bal = cm["SplyExNtv"].reindex(cal)
        d30 = (bal / bal.shift(BAL_DAYS) - 1).shift(1)
        out["bal30"] = d30
        out["x3"] = _flag(~(d30 > 0), d30.notna())
    for key, col in (("obm_ge365", "x2"), ("obm_ge155", "x2b")):
        if key in raw:
            s = raw[key].reindex(cal)
            o14 = s.rolling(OLD_DAYS, min_periods=OLD_DAYS).sum()
            thr = o14.shift(1).rolling(ZWIN, min_periods=ZMIN).quantile(OLD_PCT)
            lag_o, lag_t = o14.shift(OLD_LAG), thr.shift(OLD_LAG)
            if col == "x2":
                out["o14_rank"] = (lag_o / lag_t)
            out[col] = _flag(~(lag_o > lag_t), lag_o.notna() & lag_t.notna())
    kl = raw.get("kl")
    if kl is not None:
        k = kl.reindex(cal)
        sh5 = (k["tbq"].rolling(TAKER_DAYS, min_periods=TAKER_DAYS).sum()
               / k["quote"].where(k["quote"] > 0).rolling(TAKER_DAYS, min_periods=TAKER_DAYS).sum())
        if zero_fee:
            sh5 = sh5.where(~((cal >= ZF[0]) & (cal <= ZF[1])))
        med = _trail_med(sh5)
        out["share5"] = sh5
        out["x4"] = _flag(sh5 >= med, sh5.notna() & med.notna())
    hourly = raw.get("hourly")
    if hourly:
        f = pi._measure(hourly, cal)
        down = f["down"]
        if zero_fee:
            down = down.where(~((cal >= ZF[0]) & (cal <= ZF[1])))
        med = _trail_med(down)
        out["down"] = down
        out["x5"] = _flag(down < med, down.notna() & med.notna())
    cb, kl_close = raw.get("cb"), (kl["close"] if kl is not None else None)
    if cb is not None and kl_close is not None:
        prem = (cb.reindex(cal) / kl_close.reindex(cal) - 1)
        p7 = prem.rolling(PREM_DAYS, min_periods=5).mean()
        out["prem7"] = p7
        out["x6"] = _flag(p7 > 0, p7.notna())
    fund = raw.get("funding")
    if fund is not None and len(fund):
        cand = (fund.index - pd.Timedelta(milliseconds=1)).normalize()
        daily = fund.groupby(cand).mean().reindex(cal)
        f7 = daily.rolling(FUND_DAYS, min_periods=6).mean()
        med = _trail_med(f7)
        out["f7"] = f7
        out["x7"] = _flag(f7 <= 0, f7.notna())
        out["x7hot"] = _flag((f7 > 0) & (f7 > med), f7.notna() & med.notna())
    oi = raw.get("oi")
    if oi is not None and len(oi):
        o = oi.reindex(cal)
        ch = o / o.shift(OI_DAYS) - 1
        out["oi5"] = ch
        out["x8"] = _flag(ch < 0, ch.notna())
    ll = raw.get("llama")
    if ll is not None and len(ll):
        s = ll[ll.index >= STABLE_FROM].reindex(cal)
        lg = np.log(s).diff()
        lg = lg.where(~(lg.abs() > JUMP), 0.0).where(lg.notna())
        g30 = lg.rolling(STABLE_DAYS, min_periods=STABLE_DAYS).sum().shift(1)
        out["st30"] = g30
        out["x9"] = _flag(g30 > 0, g30.notna())
    for col in PARAMS:
        if col not in out:
            out[col] = np.nan
    x1, x2, x3 = out["x1"], out["x2"], out["x3"]
    alld = x1.notna() & x2.notna() & x3.notna()
    out["c1"] = _flag((x1 > 0.5) & (x2 > 0.5) & (x3 > 0.5), alld)
    return out.reindex(index)


def _cut_raw(raw: dict, cut: pd.Timestamp) -> dict:
    out = {}
    for k, v in raw.items():
        if k == "hourly":
            lim = int((cut + DAY).value // 1_000_000)
            out[k] = {ms: r for ms, r in v.items() if ms < lim}
        elif k == "funding":
            out[k] = v[v.index <= cut + DAY]
        elif isinstance(v, (pd.Series, pd.DataFrame)):
            out[k] = v[v.index <= cut]
        else:
            out[k] = v
    return out


def _lookahead(raw: dict, index: pd.DatetimeIndex, full: pd.DataFrame) -> tuple[int, int]:
    total = bad = 0
    cols = list(PARAMS) + ["c1"]
    for cut in CUTS:
        cc = pd.Timestamp(cut)
        idx = index[index <= cc]
        part = _forces(_cut_raw(raw, cc), idx, True)
        for c in cols:
            a, b = part[c].values, full[c].reindex(idx).values
            total += 1
            same = np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))
            bad += int(not same)
    return total, bad


# ---------- events and PAYOFF ----------

def _low20(c: np.ndarray, e: int) -> float:
    return float(np.min(c[e - LOW_DAYS:e])) if e >= LOW_DAYS else float("nan")


def _payoff(c: np.ndarray, ma: np.ndarray, e: int) -> tuple[float | None, int, str]:
    """(PAYOFF, days, how it ended) of staying in from exit e; None while the span is still open."""
    n = len(c)
    on = np.nan_to_num(ma) > 0.5
    low = _low20(c, e)
    for t in range(e + 1, min(n, e + WINDOW + 1)):
        if on[t]:
            return c[t] / c[e] - 1 + 2 * COST, t - e, "on"
        if c[t] <= STOP_PCT * c[e] or (math.isfinite(low) and c[t] < low):
            return c[t] / c[e] - 1, t - e, "stop"
    if e + WINDOW <= n - 1:
        return c[e + WINDOW] / c[e] - 1, WINDOW, "time"
    return None, n - 1 - e, "open"


def _events(close: pd.Series, vflag: pd.Series, F: pd.DataFrame) -> list[dict]:
    c = close.values.astype(float)
    ma = msc._targets(close)["ma50"].values.astype(float)
    D = wa._desc(close, ma, vflag.reindex(close.index).values if len(vflag) else None)
    rows = []
    for e in wa._exits(ma):
        pay, days, how = _payoff(c, ma, e)
        day = close.index[e]
        fv = F.loc[day] if day in F.index else None
        rows.append({"i": int(e), "day": day, "kind": wa._label(c, ma, e), "pay": pay, "days": days, "how": how,
                     "D": {k: float(D[k][e]) for k in wa.KEYS},
                     "X": {k: (float(fv[k]) if fv is not None and _ok(fv[k]) else float("nan"))
                           for k in list(PARAMS) + ["c1", "x2b", "x7hot"] if fv is not None and k in F.columns}})
    return rows


def _clusters(days: list, gap: int = CLUSTER_GAP) -> np.ndarray:
    order = np.argsort(np.array(days, dtype="datetime64[ns]"))
    cid = np.zeros(len(days), dtype=int)
    cur, prev = 0, None
    for j in order:
        if prev is not None and (days[j] - prev).days > gap:
            cur += 1
        cid[j] = cur
        prev = days[j]
    return cid


def _week_clusters(days: list) -> np.ndarray:
    keys = [(d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d") for d in days]
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([uniq[k] for k in keys], dtype=int)


def _boot_sets(cid: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    groups = [np.where(cid == g)[0] for g in np.unique(cid)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def _mean(v: np.ndarray) -> float | None:
    return float(np.mean(v)) if len(v) else None


def _ols_coef(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def _design(rows: list[dict], stay: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    cols = [np.ones(len(rows)), stay.astype(float)]
    if extra is not None:
        cols.append(extra.astype(float))
    for k in wa.KEYS:
        v = np.array([r["D"][k] for r in rows], dtype=float)
        miss = ~np.isfinite(v)
        fill = float(np.nanmean(v)) if (~miss).any() else 0.0
        cols.append(np.where(miss, fill, v))
        if miss.any() and not miss.all():
            cols.append(miss.astype(float))
    X = np.column_stack(cols)
    keep = [0, 1] + [j for j in range(2, X.shape[1]) if np.std(X[:, j]) > 0]
    return X[:, keep]


def _analyse(rows: list[dict], key: str, cid_fn=_clusters, seed: int = SEED, extra_dummy: bool = False,
             daily: pd.Series | None = None, light: bool = False) -> dict:
    """Stay / leave statistics for one parameter over known-PAYOFF exits where it is defined."""
    sub = [r for r in rows if r["pay"] is not None and _ok(r["X"].get(key))]
    n = len(sub)
    if n < 4:
        return {"n": n}
    y = np.array([r["pay"] for r in sub])
    st = np.array([r["X"][key] > 0.5 for r in sub])
    kinds = np.array([r["kind"] for r in sub])

    def grp(m):
        k = int(m.sum())
        return {"n": k, "whip": float((kinds[m] == "whip").mean()) if k else None,
                "real": float((kinds[m] == "real").mean()) if k else None, "nreal": int((kinds[m] == "real").sum()),
                "mean": _mean(y[m]), "med": float(np.median(y[m])) if k else None}

    res = {"n": n, "stay": grp(st), "leave": grp(~st), "share": float(st.mean())}
    res["diff"] = (res["stay"]["mean"] - res["leave"]["mean"]) if st.any() and (~st).any() else None
    days = [r["day"] for r in sub]
    cid = cid_fn(days)
    res["clusters"] = int(len(np.unique(cid)))
    sets = _boot_sets(cid, BOOT, seed)
    ms, ds = [], []
    for ix in sets:
        s_, y_ = st[ix], y[ix]
        ms.append(float(y_[s_].mean()) if s_.any() else np.nan)
        ds.append(float(y_[s_].mean() - y_[~s_].mean()) if s_.any() and (~s_).any() else np.nan)
    ms, ds = np.array(ms), np.array(ds)
    res["m_ci"] = (float(np.nanpercentile(ms, 5)), float(np.nanpercentile(ms, 95))) if np.isfinite(ms).any() else (None, None)
    res["d_ci"] = (float(np.nanpercentile(ds, 5)), float(np.nanpercentile(ds, 95))) if np.isfinite(ds).any() else (None, None)
    res["p"] = float(np.mean(~(ms > 0))) if st.any() else 1.0
    # beyond price
    dummy = np.array([1.0 if r.get("asset") == "eth" else 0.0 for r in sub]) if extra_dummy else None
    X = _design(sub, st, dummy)
    if st.any() and (~st).any():
        res["coef"] = _ols_coef(y, X)
        cs = []
        for ix in sets[:BOOT_REG]:
            if st[ix].any() and (~st[ix]).any():
                cs.append(_ols_coef(y[ix], X[ix]))
        res["c_ci"] = (float(np.percentile(cs, 5)), float(np.percentile(cs, 95))) if cs else (None, None)
    else:
        res["coef"], res["c_ci"] = None, (None, None)
    # halves
    early = np.array([d < HALF for d in days])
    for lab, m in (("h1", early), ("h2", ~early)):
        a, b = st & m, ~st & m
        res[lab] = {"n": int(m.sum()), "ns": int(a.sum()),
                    "diff": float(y[a].mean() - y[b].mean()) if a.any() and b.any() else None}
    # from 2017
    late = np.array([d >= LATE for d in days])
    a, b = st & late, ~st & late
    res["late"] = {"n": int(late.sum()), "ns": int(a.sum()), "stay": _mean(y[a]), "leave": _mean(y[b]),
                   "real": float((kinds[a] == "real").mean()) if a.any() else None,
                   "diff": float(y[a].mean() - y[b].mean()) if a.any() and b.any() else None}
    # placebo: circular shift of the daily series against fixed exit dates
    res["placebo"] = None
    if daily is not None and not light and res["diff"] is not None:
        s = daily.copy()
        valid = s.notna()
        first, last = s.index[int(np.argmax(valid.values))], s.index[len(s) - 1 - int(np.argmax(valid.values[::-1]))]
        span = s[(s.index >= first) & (s.index <= last)]
        m = len(span)
        rng = np.random.default_rng(seed + 77)
        at = np.array([span.index.get_loc(d) if first <= d <= last else -1 for d in days])
        hits = 0
        for _ in range(DRAWS):
            k = int(rng.integers(SHIFT_MIN, max(SHIFT_MIN + 1, m - SHIFT_MIN)))
            vals = np.roll(span.values, k)
            v = np.array([vals[j] if j >= 0 else np.nan for j in at])
            ok = np.isfinite(v)
            s_ = v > 0.5
            a, b = ok & s_, ok & ~s_
            dd = float(y[a].mean() - y[b].mean()) if a.any() and b.any() else -np.inf
            hits += int(dd >= res["diff"] - 1e-12)
        res["placebo"] = hits / DRAWS
    return res


def _bar(a: dict) -> dict:
    if a.get("n", 0) < 4 or a.get("diff") is None:
        return {"ok": False, "items": {}}
    it = {"stay": a["share"] >= BAR_STAY,
          "real": a["stay"]["real"] is not None and a["stay"]["real"] <= BAR_REAL,
          "mean": _ok(a["m_ci"][0]) and a["stay"]["mean"] > 0 and a["m_ci"][0] > 0,
          "halves": all(_ok(a[h]["diff"]) and a[h]["diff"] > 0 for h in ("h1", "h2")),
          "price": _ok(a["coef"]) and a["coef"] > 0 and _ok(a["c_ci"][0]) and a["c_ci"][0] > 0,
          "placebo": _ok(a["placebo"]) and a["placebo"] <= PLACEBO_BAR}
    return {"ok": all(it.values()), "items": it}


def _holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    out, stop = {}, False
    m = len(order)
    for i, k in enumerate(order):
        if not stop and ps[k] <= ALPHA / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out


# ---------- the exam ----------

class _Exam(wa._Exam):
    """whipsaw_anatomy's exam (main_signal_check's four checks) for 'ma50 OR sit through the exit for up to 60 days
    with the hard stop when the daily STAY flag (spec) was on at the exit'; spec None = ma50, 'all' = V0."""

    def __init__(self, close, other, price, b0, pre, first):
        super().__init__(close, other, price, b0, pre, first, {})
        self.meta = {}
        s = self.series["btc"]
        self.meta["btc"] = self._m(s["c"], s["ex"], close.index[s["ex"]], 0, 0, s["c"])
        npre = len(pre)
        for h in range(24):
            s = self.series[("clock", h)]
            cf = np.concatenate([pre, price[h::24]]).astype(float)
            self.meta[("clock", h)] = self._m(s["c"], s["ex"], self.m.dates[s["ex"]], 1, npre, cf)
        for key, o in self.other.items():
            s = self.series[("o", key)]
            self.meta[("o", key)] = self._m(s["c"], s["ex"], o["close"].index[s["ex"]], o["lag"], 0, s["c"])

    @staticmethod
    def _m(c, ex, dates, lag, off, full):
        lows = np.array([_low20(full, int(e) + off) for e in ex])
        return {"dates": pd.DatetimeIndex(dates) - lag * DAY, "lows": lows}

    def pos(self, key, spec, perm=None) -> np.ndarray:
        s, mt = self.series[key], self.meta[key]
        if spec is None:
            v = np.zeros(len(s["ex"]), dtype=bool)
        elif isinstance(spec, str):
            v = np.ones(len(s["ex"]), dtype=bool)
        else:
            v = spec.reindex(mt["dates"]).fillna(0.0).values > 0.5
        return _hold(s["c"], s["ma"], s["ex"], v, mt["lows"])


def _hold(c: np.ndarray, ma: np.ndarray, ex: np.ndarray, vals: np.ndarray, lows: np.ndarray) -> np.ndarray:
    pos = np.nan_to_num(np.asarray(ma, dtype=float)).copy()
    on = pos > 0.5
    n = len(c)
    for e, v, low in zip(ex, vals, lows):
        if not v:
            continue
        for t in range(e, min(n, e + WINDOW)):
            if t > e and (on[t] or c[t] <= STOP_PCT * c[e] or (math.isfinite(low) and c[t] < low)):
                break
            pos[t] = 1.0
    return pos


def _roll(flag: pd.Series, u: float) -> pd.Series:
    """Circular shift of the series from its first defined day by k = 180 + u x (span - 360) days (u in [0, 1))."""
    out = flag.copy()
    valid = out.notna().values
    start = out.index[int(np.argmax(valid))]
    m = out.index >= start
    span = int(m.sum())
    k = SHIFT_MIN + int(u * max(1, span - 2 * SHIFT_MIN))
    out[m] = np.roll(out[m].values, k)
    return out


# ---------- report helpers ----------

def _say(v) -> str:
    return ("留" if v > 0.5 else "走") if _ok(v) else "無資料"


def _cal_years(close: pd.Series, pos: np.ndarray) -> dict:
    return wa._cal_years(close, pos)


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    rb = data["raw"]["btc"]
    F = _forces(rb, close.index, True)
    vflag = wa._vol_flag(rb["kl"]["quote"]) if "kl" in rb else pd.Series(dtype=float)
    rows = _events(close, vflag, F)
    for r in rows:
        r["asset"] = "btc"
    known = [r for r in rows if r["pay"] is not None]

    # per parameter (BTC)
    an = {}
    for i, k in enumerate(list(PARAMS) + ["c1"]):
        an[k] = _analyse(rows, k, seed=SEED + 10 * i, daily=F[k])
    an["x2b"] = _analyse(rows, "x2b", seed=SEED + 200, daily=F["x2b"])
    an["x7hot"] = _analyse(rows, "x7hot", seed=SEED + 210, light=True)
    holm = {}
    for fam in (P1, P2):
        ps = {k: an[k]["p"] for k in fam if an[k].get("n", 0) >= 4}
        holm.update(_holm(ps))
    bars = {k: _bar(an[k]) for k in list(PARAMS) + ["c1"]}
    support = {k: bars[k]["ok"] and holm.get(k, False) for k in P1 + P2}
    support["c1"] = bars["c1"]["ok"]

    # ETH
    eth = {"status": data["cm_status"].get("eth"), "rows": [], "an": {}, "pool": {}}
    re = data["raw"]["eth"]
    if data["eth_close"] is not None and ("kl" in re or "cm" in re):
        ec = data["eth_close"]
        FE = _forces(re, ec.index, False)
        ev = wa._vol_flag(re["kl"]["quote"]) if "kl" in re else pd.Series(dtype=float)
        erows = _events(ec, ev, FE)
        for r in erows:
            r["asset"] = "eth"
        eth["rows"] = erows
        for i, k in enumerate(("x1", "x3", "x4", "x5", "x6")):
            eth["an"][k] = _analyse(erows, k, seed=SEED + 300 + i, daily=FE[k])
            eth["pool"][k] = _analyse(rows + erows, k, cid_fn=_week_clusters, seed=SEED + 400 + i,
                                      extra_dummy=True, light=True)
        eth["first"] = ec.index[0]

    # exam
    rows_h, problem = cr._hourly("BTCUSDT")
    if len(rows_h) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows_h)} 筆")
    price, b0, _ = cr._grid(rows_h)
    pre = cr._prehistory(b0)
    ex = _Exam(close, data["other"], price, b0, pre, FIRST_MONTH)
    specs = {"ma50": None, "v0": "all", **{k: F[f] for k, f in RULE_FLAG.items()}}
    s = {k: ex.summary(sp, KI[k], SIMS) for k, sp in specs.items()}
    t_exam = time.time() - t0
    ex2 = _Exam(close, data["other"], price, b0, pre, STABLE_MONTH)
    s2 = {k: ex2.summary(sp, KI[k], SIMS) for k, sp in specs.items()}

    small = {k: ex.summary(specs[k], KI[k], PL_SIMS) for k in RULE_FLAG}
    hits = {k: 0 for k in RULE_FLAG}
    lows = {k: [] for k in RULE_FLAG}
    rng = np.random.default_rng(SEED + 9000)
    for d in range(DRAWS):
        kk = float(rng.random())
        for k in RULE_FLAG:
            x = ex.summary(_roll(specs[k], kk), KI[k], PL_SIMS)
            lows[k].append(x["low"])
            hits[k] += int(sd._score_ge(x, small[k]))
    placebo = {k: hits[k] / DRAWS for k in RULE_FLAG}

    pos = {k: ex.pos("btc", sp) for k, sp in specs.items()}
    cal = {k: _cal_years(close, pos[k]) for k in pos}
    look = _lookahead(rb, close.index, F)
    return {"F": F, "rows": rows, "known": known, "an": an, "holm": holm, "bars": bars, "support": support,
            "eth": eth, "ex": ex, "s": s, "s2": s2, "small": small, "placebo": placebo,
            "placebo_med": {k: float(np.median(v)) for k, v in lows.items()}, "pos": pos, "cal": cal,
            "look": look, "problem": problem, "secs": time.time() - t0, "secs_exam": t_exam}


# ---------- tables ----------

PCOLS = ["參數", "有資料的下車（群）", "說「留」的占比", "留：假跌破｜真的下跌", "留：平均｜中位數",
         "留的平均 90% 區間", "走：假跌破｜真的下跌", "走：平均｜中位數", "差（留−走）與 90% 區間",
         "扣掉價格後的差（係數）與 90% 區間", "差：2020 前｜2020 起", "隨機錯位", "Holm（同一層）", "達標？"]


def _prow(name: str, a: dict, holm=None, ok=None) -> list:
    if a.get("n", 0) < 4 or a.get("diff") is None:
        return [name, f"{a.get('n', 0)}", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "資料不夠"]
    st, lv = a["stay"], a["leave"]
    hv = "—" if holm is None else ("通過" if holm else "沒通過")
    okv = "—" if ok is None else ("達標" if ok else "沒達標")
    return [name, f"{a['n']}（{a['clusters']} 群）", f"{_share(a['share'])}（{st['n']} 次）",
            f"{_share(st['whip'])}｜{_share(st['real'])}", f"{_p(st['mean'])}｜{_p(st['med'])}", _ci(*a["m_ci"]),
            f"{_share(lv['whip'])}｜{_share(lv['real'])}", f"{_p(lv['mean'])}｜{_p(lv['med'])}",
            f"{_pt(a['diff'])} {_ci(*a['d_ci'], f=_pt)}", f"{_pt(a['coef'])} {_ci(*a['c_ci'], f=_pt)}",
            f"{_pt(a['h1']['diff'])}｜{_pt(a['h2']['diff'])}",
            _share(a["placebo"]) if _ok(a.get("placebo")) else "—", hv, okv]


def _now(data: dict, r: dict) -> list[dict]:
    close = data["close"]
    c = close.values.astype(float)
    word = {1: "持有", 0: "現金"}
    st = {k: int(r["pos"][k][-1] > 0.5) for k in ("ma50", "v1", "v0", "s1")}
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜V1｜V0", "value": f"{word[st['ma50']]}｜{word[st['v1']]}｜{word[st['v0']]}"},
           {"label": "比特幣：50 日均線｜S1", "value": f"{word[st['ma50']]}｜{word[st['s1']]}"}]
    rows = r["rows"]
    if rows:
        last = rows[-1]
        e = last["i"]
        ma = r["ex"].ma
        low = _low20(c, e)
        stop_lvl = max(STOP_PCT * c[e], low if math.isfinite(low) else 0.0)
        days = len(c) - 1 - e
        out.append({"label": "最近一次 50 日均線下車",
                    "value": f"{_d(last['day'])}，收盤 {c[e]:,.0f} 美元（{days} 天前）；停損線：收盤跌破 "
                             f"{STOP_PCT * c[e]:,.0f}（下車價 −8%）或 {low:,.0f}（前 20 天最低收盤），先到的那個"})
        for k in list(PARAMS) + ["c1"]:
            v = last["X"].get(k, float("nan"))
            out.append({"label": f"　那天 {PNAME[k]}", "value": _say(v)})
        on_again = bool((np.nan_to_num(ma[e + 1:]) > 0.5).any())
        hit = any(c[t] <= STOP_PCT * c[e] or (math.isfinite(low) and c[t] < low) for t in range(e + 1, len(c)))
        if on_again:
            state = "已結束：50 日均線已經重新上車"
        elif hit:
            state = "已結束：停損已經觸發"
        elif days >= WINDOW:
            state = "已結束：60 天時間窗已過"
        else:
            state = f"還開著：第 {days} 天，停損沒有觸發（最近收盤離較高那條停損線 {c[-1] / stop_lvl - 1:+.1%}）"
        out.append({"label": "60 天時間窗或停損結束了嗎", "value": state})
        if last["pay"] is None:
            out.append({"label": "這次下車的得失", "value": "還不知道（還在 60 天裡），沒有算進任何統計"})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；V1 沒有達標前不會用"})
    return out


def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, s2, an = data["close"], r["s"], r["s2"], r["an"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    rows, known = r["rows"], r["known"]
    F = r["F"]
    base = s["ma50"]

    def first_day(k):
        v = F[k].dropna()
        return _d(v.index[0]) if len(v) else "—"

    # ----- definitions -----
    rows_def = [[PNAME[k], f"第 {TIER[k]} 層" if k in TIER else "組合", PDEF[k], PWHY[k], PDATA[k], first_day(k)]
                for k in list(PARAMS) + ["c1"]]
    kinds = {k: [x for x in known if x["kind"] == k] for k in ("whip", "real", "other")}
    rows_base = [[KIND[k], f"{len(v)}", _share(len(v) / len(known)) if known else "—",
                  _p(_mean(np.array([x["pay"] for x in v]))) if v else "—",
                  _p(float(np.median([x["pay"] for x in v]))) if v else "—"] for k, v in kinds.items()]
    allpay = np.array([x["pay"] for x in known])
    rows_base.append(["全部（＝每次都不賣，V0 的做法）", f"{len(known)}", "100%", _p(_mean(allpay)), _p(float(np.median(allpay)))])
    hows = {h: sum(1 for x in known if x["how"] == h) for h in ("on", "stop", "time")}

    rows_p = [_prow(PNAME[k], an[k], r["holm"].get(k), r["support"][k]) for k in P1]
    rows_p += [_prow(PNAME[k], an[k], r["holm"].get(k), r["support"][k]) for k in P2]
    rows_p += [_prow(PNAME[k], an[k]) for k in P3]
    rows_p.append(_prow(PNAME["c1"], an["c1"], None, r["support"]["c1"]))
    rows_rob = [_prow("X2 換成放 155 天以上的老幣（穩健性）", an["x2b"]),
                _prow("X7 資金費率明顯偏高（說「走」的那組當作『留』來看，方向相反）", an["x7hot"])]

    rows_bar = []
    lab = {"stay": "說「留」至少 30%", "real": "留的那組真的下跌 ≤ 8%", "mean": "留的平均 > 0 且區間不含 0",
           "halves": "前後兩半方向一樣", "price": "扣掉價格後還在（係數 > 0 且區間不含 0）", "placebo": "隨機錯位 ≤ 5%"}
    for k in list(P1) + list(P2) + ["c1"]:
        it = r["bars"][k]["items"]
        rows_bar.append([PNAME[k]] + [("過" if it.get(x) else "沒過") if it else "—" for x in lab]
                        + ["—" if k == "c1" else ("過" if r["holm"].get(k) else "沒過"),
                           "支持" if r["support"][k] else "沒有看到"])

    rows_late = []
    for k in list(P1) + ["c1"]:
        L = an[k].get("late")
        if not L:
            continue
        rows_late.append([PNAME[k], f"{L['n']}", f"{L['ns']}", _p(L["stay"]), _p(L["leave"]), _pt(L["diff"]),
                          _share(L["real"])])

    # ETH
    eth = r["eth"]
    rows_eth = []
    if eth["an"]:
        for k, a in eth["an"].items():
            rows_eth.append(_prow(f"以太幣 {PNAME[k]}", a))
        for k, a in eth["pool"].items():
            rows_eth.append(_prow(f"比特幣＋以太幣合併 {PNAME[k]}", a))

    # exam
    res = {}
    for k in RULE_FLAG:
        x = s[k]
        res[k] = {"score": sd._score_ge(x, base), "loss": x["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK,
                  "plc": r["placebo"][k] <= PLACEBO_BAR, "v0": x["low"] > s["v0"]["low"] + 1e-12}
        res[k]["pass"] = all(res[k].values())

    def exam_rows(S, with_pl):
        out = []
        for k in ("ma50", "v0", "v1", "s1", "s2", "s3"):
            x = S[k]
            ck = x["checks"]
            out.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）",
                        _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                        _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                        f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點"]
                       + ([("—" if k not in RULE_FLAG else f"{_share(r['placebo'][k])}（真的 {_share1(r['small'][k]['low'])}，"
                            f"錯位中位數 {_share1(r['placebo_med'][k])}）")] if with_pl else []))
        return out

    def yes(v):
        return "過" if v else "沒過"

    rows_pass = []
    for k in RULE_FLAG:
        x = s[k]
        rows_pass.append([SHORT[k], f"{_share1(x['low'])} vs {_share1(base['low'])}｜{yes(res[k]['score'])}",
                          f"{_pc(x['c_deep_med'])} vs {_pc(base['c_deep_med'])}｜{yes(res[k]['loss'])}",
                          f"{_share(r['placebo'][k])}｜{yes(res[k]['plc'])}",
                          f"{_share1(x['low'])} vs V0 {_share1(s['v0']['low'])}｜{yes(res[k]['v0'])}",
                          "支持" if res[k]["pass"] else "沒有看到"])

    # per other asset (added after the run, description only)
    assets = list(base["o_each"])
    rows_asset = []
    for k in ("ma50", "v0", "v1", "s1", "s2", "s3"):
        rows_asset.append([SHORT[k]] + [f"{s[k]['o_each'][a][1]}／{s[k]['o_each'][a][0]}" for a in assets])
    gain_asset = {a: s["s1"]["o_each"][a][1] - base["o_each"][a][1] for a in assets}
    top_asset = max(gain_asset, key=gain_asset.get) if assets else None

    # calendar years
    cal = r["cal"]
    rows_cal = []
    for y in sorted(cal["ma50"]):
        if y < 2011:
            continue
        rows_cal.append([str(y), _p(cal["ma50"][y]), _pt(cal["v1"][y] - cal["ma50"][y]), _pt(cal["v0"][y] - cal["ma50"][y]),
                         _pt(cal["s1"][y] - cal["ma50"][y]), _pt(cal["s2"][y] - cal["ma50"][y]),
                         _pt(cal["s3"][y] - cal["ma50"][y])])
    up = [y for y in cal["v1"] if y >= 2011 and cal["v1"][y] - cal["ma50"][y] > 1e-4]
    dn = [y for y in cal["v1"] if y >= 2011 and cal["v1"][y] - cal["ma50"][y] < -1e-4]

    # 2018 and 2022
    rows_yr = []
    for x in rows:
        if x["day"].year not in (2018, 2022):
            continue
        rows_yr.append([_d(x["day"]), KIND[x["kind"]], _p(x["pay"]) if x["pay"] is not None else "未定"]
                       + [_say(x["X"].get(k, float("nan"))) for k in list(PARAMS) + ["c1"]])

    # agreement
    def agree(x):
        v = [x["X"].get(k, float("nan")) for k in P1]
        if not all(_ok(t) for t in v):
            return None
        if all(t > 0.5 for t in v):
            return "三個都說留"
        if all(t < 0.5 for t in v):
            return "三個都說走"
        return "意見不一"
    rows_ag = []
    for g in ("三個都說留", "三個都說走", "意見不一"):
        sub = [x for x in known if agree(x) == g]
        pays = np.array([x["pay"] for x in sub])
        rows_ag.append([g, f"{len(sub)}", _share(np.mean([x["kind"] == "whip" for x in sub])) if sub else "—",
                        _share(np.mean([x["kind"] == "real" for x in sub])) if sub else "—",
                        _p(_mean(pays)) if sub else "—", _p(float(np.median(pays))) if sub else "—"])

    # open leads
    LEADS = {
        "x1": "只看存入交易所的量，沒分是誰存的（散戶、礦工、大戶）、存了以後有沒有真的賣；交易所地址標籤是現在回頭套用的，早年漏很多。",
        "x2": "老幣移動不一定是賣（換錢包、託管搬家，例如 Mt. Gox 受託人）；只用一個作者的資料；14 天和八成門檻可能太粗。",
        "x3": "交易所餘額會因為交易所自己搬錢包而跳動；只有 Coin Metrics 認得的交易所；沒有把 ETF 託管和場外交易算進去。",
        "x4": "只有幣安現貨；沒有其他交易所、合約市場的主動買賣；5 天的日線看不到盤中誰在接（小時或逐筆細節）；也沒有掛單簿深度。",
        "x5": "價格衝擊只用幣安的主動買賣量估；沒有掛單簿深度、沒有其他交易所；14 天可能太長或太短。",
        "x6": "Coinbase 溢價也受美元／USDT 匯價影響；2024 年起 ETF 買盤大多不經過 Coinbase 現貨價格。",
        "x7": "只有幣安的資金費率；沒有其他交易所、沒有爆倉（被迫平倉）量的歷史；次數太少。",
        "x8": "只有幣安的未平倉量，而且 2020-09 起才有；沒有爆倉資料；次數太少。",
        "x9": "穩定幣增加不等於拿來買比特幣；DefiLlama 早年涵蓋不完整；次數少。",
        "c1": "三個參數各自的缺點都帶進來；「三個同時」讓說留的次數很少。",
    }
    rows_leads = []
    for k in list(PARAMS) + ["c1"]:
        if r["support"].get(k):
            continue
        if TIER.get(k) == 3:
            what = "次數太少，只描述"
        else:
            it = r["bars"][k]["items"]
            got = [lab[x] for x in lab if it.get(x)]
            miss = [lab[x] for x in lab if it and not it.get(x)]
            if k in P1 + P2 and not r["holm"].get(k):
                miss.append("Holm")
            what = ("這個定義、這份資料沒有看到" if len(got) <= 2 else "部分看到，但沒有全部達標") + \
                   (f"（過了：{'、'.join(got)}；沒過：{'、'.join(miss)}）" if got else "")
        rows_leads.append([PNAME[k], what, LEADS[k]])

    look_total, look_bad = r["look"]
    tables = [
        {"caption": ("九個「買賣力量」參數和一個組合（跑之前憑道理定好方向）。每個參數在 50 日均線下車那天收盤時就知道，"
                     "說「留」＝預期是假跌破（跌破均線後很快又站回去、而且站回的價格更高），說「走」＝預期是真的下跌。"
                     "第 1 層是主要的檢定，第 2 層是第二組，第 3 層資料太短只描述。"),
         "columns": ["參數", "層", "什麼時候說「留」", "背後的想法", "資料", "第一天有值"], "rows": rows_def},
        {"caption": (f"比特幣 50 日均線下車 {len(rows)} 次（{_d(rows[0]['day'])} 起），已經知道結果的 {len(known)} 次。"
                     "「多抱的得失」＝下車那天不賣，繼續抱到 50 日均線重新上車（交還給均線）、碰到停損、或滿 60 天（先到的那個），"
                     "比賣掉拿現金多賺或多賠多少。停損＝收盤比下車價低 8%，或收盤跌破下車前 20 天的最低收盤，先碰到哪條就賣。"
                     f"結束方式：均線重新上車 {hows['on']} 次、停損 {hows['stop']} 次、滿 60 天 {hows['time']} 次。"
                     "真的下跌＝下車後 60 天內先跌到下車價的 85% 以下（whipsaw_anatomy 的分類）。"),
         "columns": ["種類", "次數", "占比", "多抱的得失：平均", "中位數"], "rows": rows_base},
        {"caption": ("每個參數：下車那天說「留」和說「走」的兩組，各自的假跌破／真的下跌比例和多抱的得失。"
                     "區間＝把 30 天內接連發生的下車當成同一群、整群重抽 2,000 次的 90% 區間（下車常擠在盤整期，不是互相獨立）。"
                     "「扣掉價格後的差」＝把十個只看價格的描述（whipsaw_anatomy 的 a 到 j：長期趨勢、跌多深、之前漲多少、波動、"
                     "成交量、週線等）一起放進迴歸，看這個參數在價格已經知道的情況下，還能多說明多少——如果它只是在重複價格"
                     "已經告訴我們的事，這個數字會接近 0。隨機錯位＝把參數的每日序列整段平移至少 180 天（200 次），"
                     "看「亂對日子」也能有一樣大的差的比例。Holm＝同一層三個參數一起檢定時，把門檻調嚴，避免試多了碰巧過關。"),
         "columns": PCOLS, "rows": rows_p},
        {"caption": "事先定好的「支持」門檻，每一條都要過（第 1、2 層還要過 Holm；組合只有一個，不用 Holm）。",
         "columns": ["參數"] + list(lab.values()) + ["Holm", "結果"], "rows": rows_bar},
        {"caption": ("只描述：穩健性。X2 改用放 155 天以上的老幣；X7 改看「資金費率明顯偏高」（這組本來預期是真的下跌，"
                     "所以表裡的「留」其實是偏高那組，差是正的才代表跟預期相反）。"),
         "columns": PCOLS, "rows": rows_rob},
        {"caption": ("只描述：只看 2017 年起的下車（交易所地址標籤是現在回頭套用的，越早越不準）。"),
         "columns": ["參數", "有資料的下車", "說留的次數", "留：平均得失", "走：平均得失", "差（留−走）", "留的那組真的下跌"],
         "rows": rows_late},
    ]
    if rows_eth:
        tables.append({"caption": (f"更多資料（只描述）：以太幣自己的 50 日均線下車（以太幣日線 {_d(eth['first'])} 起，"
                                   f"{len(eth['rows'])} 次），用以太幣自己的交易所存入和餘額（Coin Metrics）、幣安 ETHUSDT 主動買賣、"
                                   "Coinbase ETH-USD 溢價；以太幣沒有 X2。合併那幾列把比特幣和以太幣放在一起，同一週的下車當成同一群，"
                                   "迴歸多放一個「是哪個幣」。"), "columns": PCOLS, "rows": rows_eth})
    tables += [
        {"caption": (f"公平考試（main_signal_check 的做法）。比特幣日線起算月 {msc.DAILY_STARTS[0][:7]} 到 "
                     f"{msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格"
                     f"（{base['c_starts']} 個起算月），一般人做法（晚 10 小時、成本 0.25%、一成進場漏掉、每個收盤時間 {SIMS} 次），"
                     f"其他資產 {'、'.join(r['ex'].other)}（{base['o_n']} 個起算月，各用自己的均線、下車和停損，參數用比特幣的，"
                     "當作整個市場的狀態）。50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"
                     f"隨機錯位：規則的每日「留」序列整段平移至少 180 天，{DRAWS} 次，一般人做法每個收盤時間只模擬 {PL_SIMS} 次，"
                     "真的規則也用同樣方法重算來比。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "隨機錯位比例"],
         "rows": exam_rows(s, True)},
        {"caption": ("規則的「支持」門檻（事先定好，四條都要過）。最後一條最重要：V0 是「每次都不賣、用一樣的停損」，完全不看買賣力量。"
                     "如果規則贏不了 V0，代表賺到的是停損本身的功勞，不是買賣力量的功勞。"),
         "columns": ["規則", "排名分數不輸均線", "最深虧損最多深 5 點", "隨機錯位 ≤ 5%", "排名分數高過 V0", "結果"],
         "rows": rows_pass},
        {"caption": ("只描述（跑完以後加的說明表）：其他資產每一個的「一年報酬贏過一直抱著」起算月數。"
                     "用來看規則比均線多出來的部分集中在哪裡。"),
         "columns": ["規則"] + assets, "rows": rows_asset},
        {"caption": (f"穩定度（只描述）：同樣的考試，只用 2020 年起的起算日（日線 {s2['ma50']['d_n']} 個、收盤時間 "
                     f"{s2['ma50']['c_n']:,} 格、其他資產 {s2['ma50']['o_n']} 個）。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）"],
         "rows": exam_rows(s2, False)},
        {"caption": "只描述：比特幣日線每年報酬（扣 0.10% 成本），各規則比 50 日均線多或少幾個百分點。",
         "columns": ["年", "50 日均線", "V1 − 均線", "V0 − 均線", "S1 − 均線", "S2 − 均線", "S3 − 均線"], "rows": rows_cal},
        {"caption": "只描述：2018 和 2022 兩次大跌年裡每一次 50 日均線下車，每個參數當時怎麼說（留／走／無資料）。",
         "columns": ["下車日", "事後種類", "多抱的得失"] + [PNAME[k] for k in list(PARAMS) + ["c1"]], "rows": rows_yr},
        {"caption": "只描述：第 1 層三個參數（X1、X2、X3）意見一致和不一致的下車。",
         "columns": ["三個參數", "下車次數", "假跌破", "真的下跌", "多抱的得失：平均", "中位數"], "rows": rows_ag},
        {"caption": ("沒有看到的，不代表想法錯：只代表「這個定義、這份資料」沒看到。下面是每個參數可能漏掉的東西，"
                     "也就是之後可以追的線索。"), "columns": ["參數", "這次的結果", "可能漏掉什麼（之後可以追）"],
         "rows": rows_leads or [["—", "—", "—"]]},
    ]

    # ----- verdict -----
    wh, rl = kinds["whip"], kinds["real"]
    first = (f"{_d(rows[0]['day'])} 起比特幣 50 日均線下車 {len(rows)} 次（已知結果 {len(known)} 次）：假跌破 {len(wh)} 次、"
             f"真的下跌 {len(rl)} 次；每次都不賣、最多多抱 60 天並用停損，平均得失 {_p(_mean(allpay))}。")
    sup = [k for k in list(P1) + list(P2) + ["c1"] if r["support"].get(k)]

    def short_line(k):
        a = an[k]
        if a.get("diff") is None:
            return f"{PNAME[k]}資料不夠"
        return (f"{PNAME[k]}說留 {_share(a['share'])}、留的那組真的下跌 {_share(a['stay']['real'])}、"
                f"差 {_pt(a['diff'])}、扣掉價格後 {_pt(a['coef'])}、隨機錯位 {_share(a['placebo']) if _ok(a.get('placebo')) else '—'}")
    second = ("第 1 層：" + "；".join(short_line(k) for k in P1) + "。第 2 層：" + "；".join(short_line(k) for k in P2)
              + f"。組合 C1：{short_line('c1')}。")
    if sup:
        third = f"達到事先定好全部門檻的參數：{'、'.join(PNAME[k] for k in sup)}。"
    else:
        cand = [k for k in list(P1) + list(P2) if r["bars"][k]["items"]]
        best = max(cand, key=lambda k: sum(r["bars"][k]["items"].values()) + int(bool(r["holm"].get(k)))) if cand else None
        third = "沒有一個參數、也不是組合，達到事先定好的全部門檻。"
        if best is not None:
            it = r["bars"][best]["items"]
            a = an[best]
            third += (f"最接近的是{PNAME[best]}：過了 {sum(it.values()) + int(bool(r['holm'].get(best)))} 條（共 7 條），"
                      f"沒過的是{'、'.join(lab[x] for x in lab if not it[x])}"
                      f"（說留只有 {_share(a['share'])}、{a['stay']['n']} 次，其中 2017 年起只有 {a['late']['ns']} 次；"
                      f"留的那組真的下跌 {_share(a['stay']['real'])}）。")
        third += ("其他參數的差都接近 0。這只代表「這個定義、這份資料」沒有看到，不代表買賣力量的想法錯了；"
                  "可能漏掉的東西列在最後一張表，當作之後可以追的線索。")
    rule_pass = [k for k in RULE_FLAG if res[k]["pass"]]
    v1, v0 = s["v1"], s["v0"]
    fourth = (f"規則考試（{base['d_n']} 個日線起算日、{base['c_n']:,} 格收盤時間、其他資產 {base['o_n']} 個起算月）："
              f"50 日均線排名分數 {_share1(base['low'])}；只有停損、不看買賣力量的對照 V0 {_share1(v0['low'])}"
              + ("（每次都不賣、只靠停損反而比均線差，所以停損本身沒有做到這件事）；" if v0["low"] < base["low"]
                 else "（只靠停損就不輸均線，所以規則還得贏過它）；")
              +
              f"V1 {_share1(v1['low'])}（隨機錯位 {_share(r['placebo']['v1'])}）、S1 {_share1(s['s1']['low'])}"
              f"（隨機錯位 {_share(r['placebo']['s1'])}）、S2 {_share1(s['s2']['low'])}、S3 {_share1(s['s3']['low'])}。")
    if rule_pass:
        fourth += f"規則門檻四條都過的：{'、'.join(SHORT[k] for k in rule_pass)}。"
        if "s1" in rule_pass and top_asset is not None:
            fourth += (f"但要小心：S1 在比特幣只坐過 {an['x1']['stay']['n']} 次下車（2017 年起 {an['x1']['late']['ns']} 次），"
                       f"比均線多出來的主要在 {top_asset}（贏抱著的起算月 {base['o_each'][top_asset][1]} → "
                       f"{s['s1']['o_each'][top_asset][1]}／{base['o_each'][top_asset][0]}），"
                       f"2020 年起的起算日也一樣領先（{_share1(s2['s1']['low'])} 對均線 {_share1(s2['ma50']['low'])}）。"
                       "次數這麼少，只能算候選，放進每日影子紀錄觀察，不是可以直接用的規則。")
    else:
        fourth += ("沒有規則同時贏過 50 日均線和 V0 並通過隨機錯位；加了買賣力量的「不賣」，"
                   "沒有比「每次都不賣、用一樣的停損」更好，或沒有比均線好。")
    eth_line = ""
    if eth["an"]:
        dirs = [k for k in eth["an"] if _ok(eth["an"][k].get("diff")) and _ok(an[k].get("diff"))
                and np.sign(eth["an"][k]["diff"]) == np.sign(an[k]["diff"])]
        opp = [k for k in eth["an"] if k not in dirs and _ok(eth["an"][k].get("diff"))]
        eth_line = (f"以太幣自己的下車 {len(eth['rows'])} 次（用以太幣自己的資料，只描述）：方向跟比特幣一樣的有"
                    f"{'、'.join(PNAME[k] for k in dirs) or '無'}，相反的有{'、'.join(PNAME[k] for k in opp) or '無'}"
                    + (f"（以太幣 X1 差 {_pt(eth['an']['x1']['diff'])}）" if "x1" in opp else "") + "。")
    elif eth["status"] != 200:
        eth_line = f"以太幣的 Coin Metrics 交易所資料不是免費的（回應 {eth['status']}），沒有做以太幣。"
    verdict = first + second + third + fourth + eth_line

    lead_names = "、".join(PNAME[k] for k in list(PARAMS) + ["c1"] if not r["support"].get(k))
    notes = [
        "為什麼有這份研究：whipsaw_anatomy 發現，50 日均線下車那一刻，十個只看價格的描述分不出假跌破和真的下跌。"
        "老闆的想法：價格分不出來，但當下的買賣力量（誰在賣、賣壓在變大還是變小、有沒有人在接）也許分得出來。這份就是測這個。",
        "名詞：假跌破＝跌破均線後很快又站回去、而且站回的價格更高（20 天內 50 日均線重新上車，價格比下車時高）。"
        "真的下跌＝下車後 60 天內先跌到下車價的 85% 以下。多抱的得失＝下車那天不賣、最多再抱 60 天（有停損），比拿現金多賺或多賠多少；"
        "現金這裡當作 0 報酬。主動買進＝直接吃掉別人掛的賣單來買。資金費率＝永續合約裡做多的人付給做空的人的費用，正的代表做多的人多。"
        "未平倉量＝合約市場還沒結清的部位總量。",
        "兩個最重要的比較：第一，「扣掉價格後」——把十個價格描述一起放進去，看買賣力量還能不能多說明一點。"
        "如果某個參數只是換個方式說「價格跌得深」，它就沒有多帶來什麼。第二，「贏過 V0」——V0 是每次下車都不賣、用一樣的停損，"
        "完全不看買賣力量。停損本身可能就有用；只有贏過 V0，才能說是買賣力量的功勞，而不是停損的功勞。",
        "所有定義、門檻、方向（z ≥ 1、八成、30／365 天、中位數、8% 和 20 天最低收盤的停損、60 天、30% 和 8% 的門檻、5% 隨機錯位）"
        "都是跑之前定的；跑完沒有改。整段歷史只評一次，2020 年起的考試只當穩定度參考。",
        "資料的限制：Coin Metrics 的交易所存入和餘額，是用現在認得的交易所地址回頭套到過去，越早越不準（所以另外列了 2017 年起）；"
        "Open Bitcoin Metrics 是一位作者在 2026 年做的資料，老幣移動不一定是賣；主動買賣、資金費率、未平倉量都只有幣安；"
        "幣安 2022-07-08 到 2023-03-22 比特幣零手續費期間的主動買賣量失真，那段的 X4、X5 不算。",
        "下車常常擠在盤整期，連續幾次下車其實是同一段行情，所以不是 148 次獨立的試驗；區間都是整群重抽算的。",
        "考試裡其他資產（ETH、SOL、BNB、XRP、MSTR）用比特幣的買賣力量當作整個市場的狀態；24 個收盤時間和 MSTR 用前一天 UTC 收盤時的值。",
        f"偷看未來的檢查：把所有原始資料在 {'、'.join(CUTS)} 截斷重算，{look_total} 次比對裡有 {look_bad} 次不一致"
        f"{'（應該是 0）' if look_bad else '，截斷前每天的參數值完全一樣'}。",
        f"這次沒有達標、列為之後線索的：{lead_names or '無'}。寫法是「這個定義、這份資料沒有看到」，不是「這個想法錯了」。",
        "就算達標，也只代表規則可以放進每日影子紀錄觀察；過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載；第一次考試 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    if eth["status"] != 200:
        notes.append(f"Coin Metrics 社群版對以太幣的交易所資料回應 {eth['status']}（不是 200），所以以太幣沒有 X1、X3。")
    notes += problems

    sources = [
        {"name": "Coin Metrics 社群 API：比特幣、以太幣交易所存入 FlowInExNtv、交易所餘額 SplyExNtv", "url": CM_URL},
        {"name": "Open Bitcoin Metrics：放 1 年以上的老幣被花掉（CSV 資料檔）", "url": OBM_BASE.format(OBM_IDS["ge365"])},
        {"name": "Open Bitcoin Metrics：放 155 天以上的老幣被花掉（CSV 資料檔）", "url": OBM_BASE.format(OBM_IDS["ge155"])},
        {"name": "幣安現貨 K 線 API（BTC、ETH 日線和 1 小時線的成交金額、主動買進金額；SOL、BNB、XRP 日線）", "url": KLINES_URL},
        {"name": "Coinbase Exchange 日線（BTC-USD、ETH-USD）", "url": CB_URL.format("BTC-USD")},
        {"name": "幣安 U 本位永續合約資金費率（BTCUSDT）", "url": FUND_URL},
        {"name": "幣安合約每日資料存檔（BTCUSDT 未平倉量）", "url": lr.ARCHIVE.format(s="BTCUSDT", d="YYYY-MM-DD")},
        {"name": "DefiLlama 穩定幣總量", "url": sd.LLAMA},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]
    return {
        "id": "exit_forces",
        "title": "均線下車那一刻的買賣力量：分得出假跌破嗎",
        "question": ("50 日均線下車（收盤跌破均線）的那一刻，當下的買賣力量——有沒有恐慌存入交易所、老手有沒有在賣、交易所有沒有在囤貨、"
                     "有沒有人在接賣單、賣壓推不推得動價格、美國買盤、資金費率、槓桿、穩定幣——能不能分出假跌破和真的下跌？"
                     "如果能，下車那天不賣、多抱一段（有停損），會不會比 50 日均線好，而且比「每次都不賣、用一樣的停損」更好？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
