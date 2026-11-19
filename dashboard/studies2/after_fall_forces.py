"""Historical study "after_fall_forces": after a big fall, do measures of the buying and selling forces turn unusual
BEFORE price turns - and can a holder use them to get in earlier than the main signal ma50 (hold while the daily
close is above its 50-day average), with a stop fixed in advance?

Why (the owner's idea, stated as a hypothesis to test fairly). ma50 gets into rallies a median 26 days after the low
and misses a median +29% (rally_sources). Price-only early entries after deep falls were tried before: rally_timing
("deep fall, then close above the 20-day average" got in closest to the low, a median 5 points cheaper than ma50, but
with more false rallies, and no early rule beat ma50 on its three pre-set conditions) and episodes (after "14-day fall
of 12% or more" since 2021, roughly as many further falls as rises). This study asks whether FORCE measures add
anything beyond price. Each popular claim below is a hypothesis with its logic; none is presumed true or false.

Ground rules (the owner's). A single signal can be useful alone if it is logical, so each signal is tested alone AND
inside a sequence. A failed test means "not seen with this definition and this data"; what the definition may miss
is listed as an open lead; nothing is called refuted. Every result is compared with (1) a price-only twin and (2)
days that look the same on price but lack the signal, because many force anomalies happen right after big falls, and
big falls may be followed by above-average returns on their own (the baseline table measures exactly that).
User-facing statements are labelled 量到的 (measured here, with the number), 文獻說的 (with the source and whether it
was opened) or 推測 (a guess); where the data cannot tell two explanations apart, the text says so.

Data (free, key-less, fixed user agent "btc-research-dashboard"; a refusal is reported, never worked around; this
study's own stores are cache/_store/after_fall_forces_*.json and are only a fallback):
  BTC daily closes: the dashboard's (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Coin Metrics community API, asset btc: FlowInExNtv (BTC sent to exchanges), SplyExNtv (BTC held on exchanges),
    SplyCur (coins in existence). Data day D is used from the close of candle D+1 (lag 1).
  Open Bitcoin Metrics CSV (DATA FILE ONLY, parsed by old_coin_selling._parse into dates and numbers; the repository
    is never cloned, installed or run): obm_spent_value_ge365d_btc_daily (BTC spent that had not moved for >= 1 year).
    Data day D used from the close of D+7 (lag 7, old_coin_selling's). If the download fails: this study's store,
    then old_coin_selling's store (read only).
  Bitstamp BTC/USD daily OHLC (volume in BTC), from 2011-08-18, lag 0 (candle closes at 00:00 UTC).
  Binance BTCUSDT perpetual funding settlements (carry._funding, 2019-09+); at the close of candle D the settlements
    timestamped before D+1 00:00 are known.
  Binance spot BTCUSDT daily klines (quote volume, taker-buy quote volume), 2017-08-17+, lag 0.
  Coinbase premium = Coinbase BTC-USD / Bitstamp BTC/USD daily close - 1 (supply_demand's loaders, read-only stores),
    7-day mean with at least 5 of 7 days, lag 0.
  DefiLlama total stablecoin supply (supply_demand._llama_fetch), used from 2019-07-01, lag 1.
  Fair exam inputs: Binance 1-hour BTCUSDT grid (clock_robustness), ETH SOL BNB XRP (Binance) and MSTR (Yahoo) daily.

Population (no hindsight troughs; fixed): every decision day t on which ma50 is OFF (main_signal_check's ma50 = 0)
  and close(t) <= 0.75 x the highest close of t-365 .. t-1 (all 365 needed). Outcomes from close(t): return to t+30,
  t+60, t+90; deepest further drop within h days = min(close[t+1..t+h]) / close(t) - 1 capped at 0 (h = 30/60/90;
  the signal tables use h = 60). Drawdown episodes: population days in date order; a gap of more than 60 calendar
  days without a population day starts a new episode. N and the number of episodes are reported.
  Intervals: circular block bootstrap of calendar days (60-day blocks, 1,000 draws, fixed seed) over the span from
  the first population day to the end; masks travel with the days.

Single signals (direction and logic fixed here; "force part" and "price part"; z-scores use the trailing window
  EXCLUDING the current value; trailing percentiles likewise; at least 80 of 90 / 330 of 365 values required):
  S1 panic deposits. Logic: forced sellers rush coins to exchanges; once forced selling is done it may reverse.
     Force: FlowInExNtv of data day d at least 2 sd above its trailing-90-day mean. Price: day d was a down day
     (close d < close d-1). Lag 1 (both refer to day d = t-1).
  S2 old holders giving up. Logic: long holders capitulating at lows hand coins to new holders.
     Force: 7-day sum of 1y+ coins spent at least 2 sd above its trailing-365-day mean. No price part. Lag 7.
  S3 volume shock. Logic: a selling climax exhausts sellers. Force: Bitstamp volume z >= 2 (trailing 90).
     Price: daily log return z <= -2 (trailing 90). Lag 0.
  S4 crowded shorts. Logic: very negative funding means shorts pay to stay short; they may be squeezed.
     Force: 7-day average funding (21 settlements) at or below the 5th percentile of its trailing 365 daily values.
  S5 selling dried up. Logic: when deposits fall away and no new low is made, sellers are exhausted.
     Force: 20-day average FlowInExNtv below the 30th percentile of its trailing 365 values (lag 1). Price: no new
     closing low for 10 days = min(close t-9..t) > min(close t-364..t-10).
  S6 old holders stopped selling. Logic: long holders refuse to sell at low prices.
     Force: 30-day sum of 1y+ coins spent below the 30th percentile of its trailing 365 values. Lag 7.
  S7 buyers absorbing. Logic: aggressive buying keeps coming while price does not rise; someone absorbs the selling.
     Force: 14-day taker-buy share of Binance spot quote volume above its trailing-365 median. Price: 14-day return
     <= 0. Lag 0 (2017-08+).
  S8 coins leaving exchanges during weakness. Logic: buyers withdraw coins to hold them while price is weak.
     Force: SplyExNtv lower than 30 data days earlier (lag 1). Price: 30-day return <= 0.
  S9 US buyers. Logic: a premium on Coinbase means US buyers pay up. Force: 7-day mean Coinbase premium > 0.
     Price: 14-day return <= 0. Lag 0.
  S10 dry powder. Logic: idle stablecoins relative to BTC's market value can be spent on BTC; this should affect HOW
     FAR a rally goes more than WHEN it starts, so its main horizon is 90 days. Force: stablecoin supply(d) /
     (SplyCur(d) x close(d)) above the 80th percentile of its trailing 365 values. Lag 1.
  A signal is "defined" on t when its force part (and price part) can be computed; "fires" = force AND price.
  Main horizon: 60 days (S10: 90 days).
  Per signal, on population days where it is defined: days and episodes it fired in; mean forward 30/60/90 and
  deepest drop (60) for signal days versus the rest of the population (difference with bootstrap interval); the
  price-only twin = population days where the price part alone holds (signals without a price part: the population
  itself); MATCHED difference = signal days versus signal-absent population days in the same cell of (decile of the
  drawdown from the 365-day high) x (decile of the trailing 30-day return) x (decile of the 20-day volatility of daily
  log returns), decile edges from all population days; weights = signal days per cell; cells holding both kinds only;
  the share of signal days matched is reported; bootstrap interval as above. SHIFT test: the force part (on its own
  defined span) circularly shifted by k in [180, L-180] days, 200 draws, price part and population in place; share of
  draws whose signal-day mean (main horizon) is at least the real one. By period 2012-16 / 2017-21 / 2022-now where
  data exist: main-horizon difference versus the rest and matched difference. One plain table: "what this signal adds
  beyond 'price already fell this much'".

Sequence Q1 (state machine, real time, fixed): ARMED when S1 or S2 or S3 fires on a population day (each new firing
  extends the window to 60 days after it; the stop reference is the lowest close since the first arming of the spell);
  EXHAUSTED when S5 fires while armed; FIRE on the first day both are true (the machine then resets). Price-only twin
  Q0: armed by a daily return z <= -2 on a population day; exhausted (latched) when "no new closing low for 10 days"
  holds while armed; FIRE on the first day, while armed and exhausted, that the close is above its 20-day average.
  States reported: 未啟動 (idle), 已啟動 (armed), 已耗盡 (exhausted, not yet fired).
Rules (same shape): hold when ma50 is on OR the early position is open. The early position opens at a FIRE close
  while ma50 is off; stop = the lowest close since arming; it ends at the first close below the stop (stopped) or
  when ma50 turns on (handed over). Single-signal rules R(Sk): trigger = Sk fired on a population day within the last
  10 days (t-9..t) AND close above its 10-day average (mean of the last 10 closes incl. t), while ma50 is off; stop =
  the lowest close of the 30 closes before entry; same exits; re-entry allowed on a later trigger day.
  Reference only (owner's addition): R0 "no force information" = the same rule with trigger = population day AND
  close above its 10-day average. Not a candidate rule.
Fair exam: main_signal_check's (price_impact's _Exam wiring) with the rule combined as max(ma50, early position):
  BTC daily monthly starts from max(2015-01, first month after the rule's inputs are all defined) to 2023-12, textbook
  0.10%; clock starts from max(that month, 2018-01) to 2023-12 x 24 closing hours textbook, and the same cells as an
  ordinary person (act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock); ETH SOL BNB XRP
  MSTR with their OWN ma50 combined with BTC's early-position flag (a market-wide flag), starts to three years before
  today. The early position is computed on BTC UTC closes; on clock h a close on date E uses the flag of UTC candle
  E-1; MSTR uses candle D-1. ma50 and Q0 are re-scored on exactly the same starts as each rule. Ranking score = worst
  of the four "beats holding" shares with main_signal_check's eligibility.
  Pre-registered "supported", all four: ranking score >= ma50's; median deepest loss (clock cells) no more than 5
  points worse than ma50's; placebo share <= 5% (force part(s) circularly shifted by >= 180 days, 200 draws, the
  whole exam rebuilt with 20 ordinary-person simulations per clock and the real rule re-scored the same way; for Q1
  the S1, S2, S3 and S5 force parts are shifted together by the same k); ranking score strictly above Q0's on the same
  starts (for Q1: Q0 on Q1's starts).
  Multiple testing: 1 sequence + 10 singles = 11 rules; with a 5% placebo bar about 0.55 rules pass that bar by luck.
  Holm step-down across the 11 placebo p-values, p = (hits + 1) / (200 + 1), family level 5%. Any survivor is only a
  candidate for the daily shadow log.
Report only: early entries per rule (count, share handed over to ma50 at a higher close, average gain to the handover,
  average loss when stopped, 0.10% per trade); the rallies of rally_sources._rallies (+80% / 30%): for each, which
  signals fired (any day, not only population days) in the 60 days before the trough, between trough and ma50 entry,
  or not at all, plus Q1 / Q0 FIRE dates - rallies with no signal are expected (for example news-driven starts) and
  are not counted as failures; open leads for every signal that is not supported.
Baseline (owner's addition, report only): for the population, for all days with a 365-day high, and for deeper
  thresholds 40% and 55%: over 30/60/90 days the share followed by a rise / a fall, mean, median, mean when up, mean
  when down, 10th and 90th percentiles, deepest further drop (mean, median, worst); population-minus-all differences
  (mean, share up) with block-bootstrap 90% intervals, full period and by period; and every drawdown episode (start,
  deepest point, population days, mean 60-day forward return) so the reader sees whether one or two carry the average.
Honesty: Coin Metrics exchange flows use today's address labels applied backwards (older panics understated); OBM is
  one author's 2026 dataset and coins moving are not always coins sold; funding and taker data are Binance only;
  volume is Bitstamp only; the population has many days but only a handful of separate drawdown episodes; thresholds
  (2 sd, 5th / 30th / 80th percentiles, 10 / 20 / 30 / 60 / 90 / 365 days) were fixed by reasoning before the run and
  no alternatives were tried. Not advice.
Today: "比特幣：50 日均線｜Q1｜Q0" ("現金｜現金｜現金" style), whether today is in the population, the Q1 / Q0 machine
  states and each signal's on/off today.
"""
from __future__ import annotations

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
from dashboard.studies2 import funding_warning as fw
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import rally_sources as rs
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
DAY = pd.Timedelta(days=1)

CM_URL = sd.CM_URL
CM_METRICS = ("FlowInExNtv", "SplyExNtv", "SplyCur")
OBM_ID = ocs.IDS["ge365"]
OBM_URL = ocs.OBM_BASE.format(OBM_ID)
BS_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
FUND_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT"
SPOT_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
STORE_CM = common.STORE / "after_fall_forces_cm.json"
STORE_OBM = common.STORE / "after_fall_forces_obm.json"
STORE_BS = common.STORE / "after_fall_forces_bitstamp.json"
STORE_FUND = common.STORE / "after_fall_forces_funding.json"
STORE_SPOT = common.STORE / "after_fall_forces_spot1d.json"
STORE_LL = common.STORE / "after_fall_forces_llama.json"
BS_FIRST = 1313625600  # 2011-08-18
SPOT_FIRST_MS = 1502928000000  # 2017-08-17

DD, DEEPER = 0.25, (0.40, 0.55)
H = (30, 60, 90)
MAIN_H = 60
EP_GAP = 60
BLOCK, BOOT = 60, 1000
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK, PLC_BAR, FAMILY = 5.0, 0.05, 0.05
ARM_WIN, RECENT, STOP_DAYS = 60, 10, 30
FIRST_MONTH = msc.DAILY_STARTS[0]
PERIODS = (("2012–16", "2012-01-01", "2016-12-31"), ("2017–21", "2017-01-01", "2021-12-31"),
           ("2022 起", "2022-01-01", "2100-01-01"))
SIG = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
RULES = ("Q1",) + SIG
KI = {"ma50": 1, "Q1": 140, "Q0": 141, "R0": 142, **{k: 150 + i for i, k in enumerate(SIG)}}
CN = {"S1": "S1 恐慌轉進交易所", "S2": "S2 老手認賠（1 年以上的幣大量移動）", "S3": "S3 爆量急跌",
      "S4": "S4 空單擁擠（資金費率極低）", "S5": "S5 賣壓枯竭", "S6": "S6 老手不賣了",
      "S7": "S7 買方在接", "S8": "S8 弱勢中幣離開交易所", "S9": "S9 美國買家（Coinbase 溢價）",
      "S10": "S10 場邊資金（穩定幣相對比特幣市值）"}
LOGIC = {
    "S1": "推測的道理：被迫賣的人急著把幣轉進交易所；被迫賣完以後，賣壓可能反轉。量法：當天轉進交易所的比特幣比過去 90 天平均高 2 個標準差以上，而且那天是跌的。",
    "S2": "推測的道理：放了很久的老手在低點撐不住、把幣交給新買家。量法：放 1 年以上的幣被花掉，7 天合計比過去 365 天平均高 2 個標準差以上。",
    "S3": "推測的道理：爆量急跌是賣方一次出清（賣壓高潮）。量法：Bitstamp 成交量比過去 90 天高 2 個標準差以上，而且當天跌幅超過 2 個標準差。",
    "S4": "推測的道理：資金費率很負＝做空的人付錢也要空，空單擠在一起，之後可能被迫回補。量法：7 天平均資金費率落在過去一年最低 5%。",
    "S5": "推測的道理：轉進交易所的幣變少、價格也不再破底，代表想賣的人賣完了。量法：20 天平均轉進量低於過去一年第 30 百分位，而且 10 天沒有創新低。",
    "S6": "推測的道理：老手在低價不肯賣。量法：1 年以上的幣被花掉，30 天合計低於過去一年第 30 百分位。",
    "S7": "推測的道理：價格沒漲，主動買的比例卻偏高，代表有人在下面接賣壓。量法：14 天主動買比例高於過去一年中位數，而且 14 天是跌或平。",
    "S8": "推測的道理：價格弱的時候還有人把幣從交易所提走自己保管，是在囤。量法：交易所餘額比 30 天前少，而且 30 天是跌或平。",
    "S9": "推測的道理：Coinbase 比 Bitstamp 貴，代表美國買家願意多付錢。量法：溢價 7 天平均 > 0，而且 14 天是跌或平。",
    "S10": "推測的道理：穩定幣是場邊等著買的錢；相對比特幣市值越多，之後能推多高的空間越大（比較影響漲多遠，不太影響何時開始，所以主要看 90 天）。量法：穩定幣總量 ÷ 比特幣市值落在過去一年最高 20%。",
}
LEADS = {
    "S1": "交易所地址標籤是今天的標籤往回套，越早的恐慌量得越少；只看鏈上轉帳，看不到交易所內部、場外和合約的強制平倉；用的是日資料，恐慌常在幾小時內發生（可以試小時資料、強平資料、標籤時點正確的資料）。",
    "S2": "OBM 是一位作者 2026 年整理的資料；幣移動不一定是賣（交易所整理錢包、託管搬家都算）；可能要分年齡層（155 天、2 年、5 年）或用花掉時的成本（SOPR）來看是不是認賠賣。",
    "S3": "只用 Bitstamp 一家的成交量，它的市占率這些年一直變小；應該看全市場成交量、合約強平量、或日內（小時）的賣壓高潮。",
    "S4": "只用幣安一家的資金費率（2019-09 起，只經過少數幾次大跌）；應該配合未平倉量、其他交易所（BitMEX 等）的費率、以及日內費率變化。",
    "S5": "交易所標籤往回套的問題同 S1；也可能要看委託簿深度（掛賣單變少）而不是轉帳量；『10 天不破底』這個門檻是事先定的，沒有試別的。",
    "S6": "同 S2：幣移動不等於賣；也可能要分年齡層，或看長期持有者的總持有量有沒有增加。",
    "S7": "只用幣安現貨，2022-07 到 2023-03 幣安免手續費期間主動成交被灌大；應該也看合約的主動買賣、其他交易所、以及委託簿。",
    "S8": "交易所餘額會因為標籤更新、交易所整理錢包、ETF 託管搬家而改變；需要時點正確的標籤，或扣掉已知的錢包整理。",
    "S9": "只比較 Coinbase 和 Bitstamp 的日收盤；美國盤中時段的溢價、2024 年以後的 ETF 申購可能更直接。",
    "S10": "DefiLlama 早年涵蓋不完整（2019-07 起才用）；穩定幣放在交易所上的量才比較像『準備要買』，那個資料免費版拿不到。",
    "Q1": "序列由 S1／S2／S3 和 S5 組成，各自的限制都在；60 天的等待窗、『啟動時最低收盤』當停損都是事先定的。",
}

_d = cr._d
_ok, _p, _pt, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._share, sd._share1, sd._pc


def _ci(lo, hi, f=_pt) -> str:
    return sd._ci(lo, hi, f)


# ---------- data ----------

def _cm_fetch() -> dict:
    reply = common.get_json(CM_URL, {"assets": "btc", "metrics": ",".join(CM_METRICS), "frequency": "1d",
                                     "start_time": "2009-01-03", "page_size": 10000}, timeout=90)
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
            if math.isfinite(v):
                rec[m] = v
        if day and rec:
            out[day] = rec
    if len(out) < 3000:
        raise RuntimeError(f"Coin Metrics 只回傳 {len(out)} 天")
    return out


def _obm_fetch() -> dict:
    reply = requests.get(OBM_URL, headers=common.HEADERS, timeout=90)
    reply.raise_for_status()
    if len(reply.content) > 20_000_000:
        raise RuntimeError("OBM 檔案太大")
    got, _ = ocs._parse("ge365", reply.text)
    return got


def _obm() -> tuple[pd.Series, str | None]:
    prob = None
    try:
        raw = _obm_fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_OBM.write_text(json.dumps(raw), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            raw = json.loads(STORE_OBM.read_text(encoding="utf-8"))
            prob = f"OBM 這次下載失敗（{type(exc).__name__}），用本研究之前存下來的資料。"
        except Exception:  # noqa: BLE001
            raw = json.loads(ocs.STORE_OBM.read_text(encoding="utf-8"))["ge365"]
            prob = f"OBM 這次下載失敗（{type(exc).__name__}），用 old_coin_selling 存下來的資料（只讀）。"
    s = pd.Series(raw, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), prob


def _bs_fetch() -> dict:
    out, start = {}, BS_FIRST
    today = pd.Timestamp(common.today_utc())
    for _ in range(20):
        page = km._get(BS_URL, {"step": 86400, "limit": 1000, "start": start}).json()["data"]["ohlc"]
        if not page:
            break
        for row in page:
            day = pd.Timestamp(int(row["timestamp"]), unit="s")
            if day < today:
                out[_d(day)] = float(row["volume"])
        nxt = int(page[-1]["timestamp"]) + 86400
        if nxt <= start or len(page) < 1000:
            break
        start = nxt
    if len(out) < 4000:
        raise RuntimeError(f"Bitstamp 只有 {len(out)} 天")
    return out


def _fund_fetch() -> dict:
    s = cy._funding("BTCUSDT")
    return {str(int(t.value // 1_000_000)): float(v) for t, v in s.items()}


def _funding() -> tuple[pd.Series, str | None]:
    prob = None
    try:
        raw = _fund_fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_FUND.write_text(json.dumps(raw), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        path = STORE_FUND if STORE_FUND.exists() else fw.STORE_PATH  # funding_warning's store: read only
        raw = json.loads(path.read_text(encoding="utf-8"))
        prob = f"幣安資金費率這次取不到（{type(exc).__name__}），用之前存下來的資料。"
    s = pd.Series({pd.Timestamp(int(k), unit="ms"): float(v) for k, v in raw.items()}).sort_index()
    return s, prob


def _spot_fetch() -> dict:
    out, start = {}, SPOT_FIRST_MS
    now_ms = int(time.time() * 1000)
    for _ in range(20):
        page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000,
                                              "startTime": start})
        if not page:
            break
        for r in page:
            if int(r[6]) < now_ms:
                out[_d(pd.Timestamp(int(r[0]), unit="ms"))] = [float(r[7]), float(r[10])]
        if len(page) < 1000:
            break
        start = int(page[-1][0]) + 1
    if len(out) < 2000:
        raise RuntimeError(f"幣安日線只有 {len(out)} 天")
    return out


def _load() -> dict:
    problems = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    raw, n = sd._stored(STORE_CM, _cm_fetch)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
    obm, n = _obm()
    problems += [n] if n else []
    obm = obm[obm.index < today].asfreq("D")
    bs, n = sd._stored(STORE_BS, _bs_fetch)
    problems += [n] if n else []
    vol = pd.Series(bs, dtype=float)
    vol.index = pd.to_datetime(vol.index)
    vol = vol.sort_index()
    fund, n = _funding()
    problems += [n] if n else []
    spot, n = sd._stored(STORE_SPOT, _spot_fetch)
    problems += [n] if n else []
    sp = pd.DataFrame.from_dict(spot, orient="index", columns=["quote", "tbq"])
    sp.index = pd.to_datetime(sp.index)
    sp = sp.sort_index()
    ll, n = sd._stored(STORE_LL, sd._llama_fetch)
    problems += [n] if n else []
    sup = pd.Series(ll, dtype=float)
    sup.index = pd.to_datetime(sup.index)
    sup = sup.sort_index()
    sup = sup[sup.index < today].asfreq("D").ffill()
    cb, n = sd._coinbase()
    problems += [n] if n else []
    bst, n = sd._bitstamp()
    problems += [n] if n else []
    days = pd.date_range(sd.FIRST, close.index[-1], freq="D")
    cbs = pd.Series({pd.Timestamp(k): v for k, v in cb.items()}).sort_index().reindex(days)
    bss = pd.Series({pd.Timestamp(k): v for k, v in bst.items()}).sort_index().reindex(days)
    prem = cbs / bss - 1
    return {"close": close, "cm": cm, "obm": obm, "vol": vol, "fund": fund, "spot": sp, "sup": sup, "prem": prem,
            "problems": problems}


# ---------- features and signals (all on decision dates) ----------

def _ztrail(x: pd.Series, win: int, minp: int) -> pd.Series:
    m = x.shift(1).rolling(win, min_periods=minp).mean()
    s = x.shift(1).rolling(win, min_periods=minp).std()
    return (x - m) / s.where(s > 0)


def _qtrail(x: pd.Series, q: float, win: int = 365, minp: int = 330) -> pd.Series:
    return x.shift(1).rolling(win, min_periods=minp).quantile(q)


def _flag(cond: pd.Series, *parts) -> pd.Series:
    ok = pd.concat(parts, axis=1).notna().all(axis=1)
    return cond.astype(float).where(ok)


def _lag(s: pd.Series, idx: pd.DatetimeIndex, lag: int) -> np.ndarray:
    return s.reindex(idx - lag * DAY).values.astype(float)


def _price(close: pd.Series) -> dict:
    c = close.values.astype(float)
    cs = pd.Series(c)
    r = np.log(cs).diff()
    zr = ((r - r.shift(1).rolling(90, min_periods=80).mean()) / r.shift(1).rolling(90, min_periods=80).std()).values
    hi = cs.shift(1).rolling(365, min_periods=365).max().values
    dd = c / hi - 1
    ma = msc._targets(close)["ma50"].values.astype(float)
    off = ma == 0  # NaN (not yet defined) is neither on nor off
    m10, m20 = cs.rolling(10, min_periods=10).mean().values, cs.rolling(20, min_periods=20).mean().values
    low_new = cs.rolling(10, min_periods=10).min()
    low_old = cs.shift(10).rolling(355, min_periods=355).min()
    nonew = (low_new > low_old).values & low_old.notna().values
    out = {"c": c, "zr": zr, "dd": dd, "ma": np.nan_to_num(ma, nan=0.0), "ma_raw": ma, "off": off,
           "a10": c > np.where(np.isnan(m10), np.inf, m10), "a20": c > np.where(np.isnan(m20), np.inf, m20),
           "low30": cs.shift(1).rolling(STOP_DAYS, min_periods=STOP_DAYS).min().values,
           "nonew": nonew, "ret14": (cs / cs.shift(14) - 1).values, "ret30": (cs / cs.shift(30) - 1).values,
           "vol20": r.rolling(20, min_periods=20).std().values,
           "down1": np.concatenate([[False, False], c[1:-1] < c[:-2]])}
    with np.errstate(invalid="ignore"):
        for thr, key in ((DD, "pop"), (DEEPER[0], "pop40"), (DEEPER[1], "pop55")):
            out[key] = off & np.isfinite(dd) & (dd <= -thr)
        out["allday"] = np.isfinite(dd)
    return out


def _signals(data: dict, close: pd.Series, f: dict) -> dict:
    """{k: {"force": 0/1/NaN on decision dates, "price": bool array or None}}"""
    idx = close.index
    cm, obm = data["cm"], data["obm"]
    out = {}
    inflow = cm["FlowInExNtv"]
    z = _ztrail(inflow, 90, 80)
    out["S1"] = {"force": _lag(_flag(z >= 2, z), idx, 1), "price": f["down1"]}
    x7 = obm.rolling(7, min_periods=7).sum()
    z = _ztrail(x7, 365, 330)
    out["S2"] = {"force": _lag(_flag(z >= 2, z), idx, 7), "price": None}
    vol = data["vol"].reindex(idx)
    z = _ztrail(vol, 90, 80)
    with np.errstate(invalid="ignore"):
        out["S3"] = {"force": _flag(z >= 2, z).values.astype(float), "price": f["zr"] <= -2}
    fund = data["fund"]
    f7 = fund.rolling(21, min_periods=21).mean()
    daily = f7.groupby(f7.index.floor("D")).last().dropna()  # settlements timestamped on day D (before D+1 00:00)
    daily = daily.reindex(pd.date_range(daily.index[0], idx[-1], freq="D"))
    q = _qtrail(daily, 0.05)
    out["S4"] = {"force": _flag(daily <= q, daily, q).reindex(idx).values.astype(float), "price": None}
    a20 = inflow.rolling(20, min_periods=20).mean()
    q = _qtrail(a20, 0.30)
    out["S5"] = {"force": _lag(_flag(a20 < q, a20, q), idx, 1), "price": f["nonew"]}
    x30 = obm.rolling(30, min_periods=30).sum()
    q = _qtrail(x30, 0.30)
    out["S6"] = {"force": _lag(_flag(x30 < q, x30, q), idx, 7), "price": None}
    sp = data["spot"].reindex(pd.date_range(data["spot"].index[0], idx[-1], freq="D"))
    share = sp["tbq"].rolling(14, min_periods=14).sum() / sp["quote"].rolling(14, min_periods=14).sum()
    med = _qtrail(share, 0.50)
    with np.errstate(invalid="ignore"):
        out["S7"] = {"force": _flag(share > med, share, med).reindex(idx).values.astype(float),
                     "price": f["ret14"] <= 0}
        bal = cm["SplyExNtv"]
        ch = bal - bal.shift(30)
        out["S8"] = {"force": _lag(_flag(ch < 0, ch), idx, 1), "price": f["ret30"] <= 0}
        p7 = data["prem"].rolling(7, min_periods=5).mean()
        out["S9"] = {"force": _flag(p7 > 0, p7).reindex(idx).values.astype(float), "price": f["ret14"] <= 0}
    sup = data["sup"][data["sup"].index >= pd.Timestamp(sd.STABLE_FROM)]
    mcap = cm["SplyCur"] * close.reindex(cm.index)
    ratio = (sup.reindex(cm.index) / mcap).where(mcap > 0)
    q = _qtrail(ratio, 0.80)
    out["S10"] = {"force": _lag(_flag(ratio > q, ratio, q), idx, 1), "price": None}
    for k, s in out.items():
        fo = s["force"]
        pr = s["price"] if s["price"] is not None else np.ones(len(idx), dtype=bool)
        s["dfn"] = np.isfinite(fo)
        s["sig"] = s["dfn"] & (np.nan_to_num(fo) > 0.5) & pr
        s["f0"] = int(np.argmax(s["dfn"])) if s["dfn"].any() else len(idx)
    return out


def _sig_from(force: np.ndarray, price, f0: int, k: int) -> np.ndarray:
    """Signal with the force part circularly shifted by k days over its own defined span (price part in place)."""
    fo = force.copy()
    fo[f0:] = np.roll(fo[f0:], k)
    pr = price if price is not None else True
    return np.isfinite(fo) & (np.nan_to_num(fo) > 0.5) & pr


# ---------- outcomes, bootstrap, matching ----------

def _forward(c: np.ndarray) -> dict:
    n = len(c)
    out = {}
    for h in H:
        fwd = np.full(n, np.nan)
        fwd[:n - h] = c[h:] / c[:n - h] - 1
        low = pd.Series(c[::-1]).rolling(h, min_periods=h).min().values[::-1]
        deep = np.full(n, np.nan)
        deep[:n - h] = np.minimum(low[1:n - h + 1] / c[:n - h] - 1, 0.0)
        out[h], out[("deep", h)] = fwd, deep
    return out


def _bidx(m: int, rng) -> np.ndarray:
    nb = -(-m // BLOCK)
    st = rng.integers(0, m, (BOOT, nb))
    return ((st[:, :, None] + np.arange(BLOCK)) % m).reshape(BOOT, -1)[:, :m]


def _q90(a) -> tuple:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)


def _mean(v, m):
    m = m & np.isfinite(v)
    return float(v[m].mean()) if m.any() else None


def _bdiff(v, a, b, bi) -> np.ndarray:
    """Per-draw mean(v | a) - mean(v | b) for bootstrap index bi (draws x days)."""
    vb = v[bi]
    fin = np.isfinite(vb)
    vb = np.where(fin, vb, 0.0)
    ab, bb = a[bi] & fin, b[bi] & fin
    na, nb = ab.sum(axis=1), bb.sum(axis=1)
    with np.errstate(all="ignore"):
        return np.where((na > 0) & (nb > 0), (vb * ab).sum(axis=1) / np.maximum(na, 1)
                        - (vb * bb).sum(axis=1) / np.maximum(nb, 1), np.nan)


def _matched(v, s, r, cell, bi=None):
    """Cell-weighted mean(v | s) - mean(v | r); s, r already restricted to matchable days. Returns (diff, coverage)
    or the per-draw diffs when bi is given."""
    nc = 1000
    if bi is None:
        ms, mr = s & np.isfinite(v), r & np.isfinite(v)
        ns = np.bincount(cell[ms], minlength=nc)
        nr = np.bincount(cell[mr], minlength=nc)
        ss = np.bincount(cell[ms], weights=v[ms], minlength=nc)
        sr = np.bincount(cell[mr], weights=v[mr], minlength=nc)
        use = (ns > 0) & (nr > 0)
        if not use.any():
            return None, 0.0
        d = (ss[use] / ns[use] - sr[use] / nr[use])
        return float((ns[use] * d).sum() / ns[use].sum()), float(ns[use].sum() / max(ns.sum(), 1))
    vb = v[bi]
    fin = np.isfinite(vb)
    key = cell[bi] + nc * np.arange(bi.shape[0])[:, None]
    ms, mr = s[bi] & fin, r[bi] & fin
    size = nc * bi.shape[0]
    ns = np.bincount(key[ms], minlength=size).reshape(-1, nc)
    nr = np.bincount(key[mr], minlength=size).reshape(-1, nc)
    ss = np.bincount(key[ms], weights=vb[ms], minlength=size).reshape(-1, nc)
    sr = np.bincount(key[mr], weights=vb[mr], minlength=size).reshape(-1, nc)
    use = (ns > 0) & (nr > 0)
    with np.errstate(all="ignore"):
        d = np.where(use, ss / np.maximum(ns, 1) - sr / np.maximum(nr, 1), 0.0)
        w = np.where(use, ns, 0)
        return np.where(w.sum(axis=1) > 0, (w * d).sum(axis=1) / np.maximum(w.sum(axis=1), 1), np.nan)


def _cells(f: dict) -> np.ndarray:
    """Decile cell (0..999) of drawdown x 30-day return x 20-day volatility; -1 where undefined or not population."""
    pop = f["pop"]
    xs = [f["dd"], f["ret30"], f["vol20"]]
    ok = pop & np.all([np.isfinite(x) for x in xs], axis=0)
    cell = np.zeros(len(pop), dtype=int)
    for j, x in enumerate(xs):
        edges = np.quantile(x[ok], np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(edges, np.where(ok, x, 0.0), side="right") - 1, 0, 9)
        cell += dec * (10 ** (2 - j))
    return np.where(ok, cell, -1)


def _episodes(pop: np.ndarray) -> np.ndarray:
    ep = np.full(len(pop), -1)
    k, last = -1, -10 ** 9
    for t in np.flatnonzero(pop):
        if t - last > EP_GAP:
            k += 1
        ep[t], last = k, t
    return ep


# ---------- single-signal statistics ----------

def _sig_stats(k: str, s: dict, f: dict, fwd: dict, cell: np.ndarray, ep: np.ndarray, idx, p0: int, bi,
               rng) -> dict:
    hm = 90 if k == "S10" else MAIN_H
    pop = f["pop"] & s["dfn"]
    sig = pop & s["sig"]
    rest = pop & ~s["sig"]
    sl = slice(p0, None)
    out = {"h": hm, "n": int(sig.sum()), "n_pop": int(pop.sum()), "eps": len(set(ep[sig])),
           "eps_pop": len(set(ep[pop])), "from": idx[s["f0"]] if s["f0"] < len(idx) else None}
    for key in list(H) + [("deep", MAIN_H)]:
        v = fwd[key]
        ms, mr = _mean(v, sig), _mean(v, rest)
        diff = (ms - mr) if _ok(ms) and _ok(mr) else None
        out[key] = {"s": ms, "r": mr, "diff": diff,
                    "ci": _q90(_bdiff(v[sl], sig[sl], rest[sl], bi)) if diff is not None else (None, None)}
    v = fwd[hm]
    if s["price"] is not None:
        twin = pop & s["price"]
        out["twin"] = (_mean(v, twin), int(twin.sum()))
    else:
        out["twin"] = (_mean(v, pop), int(pop.sum()))
    mok = cell >= 0
    md, cov = _matched(v, sig & mok, rest & mok, np.where(mok, cell, 0))
    out["m"] = md
    out["cov"] = cov
    if md is not None:
        out["m_ci"] = _q90(_matched(v[sl], (sig & mok)[sl], (rest & mok)[sl], np.where(mok, cell, 0)[sl], bi))
    else:
        out["m_ci"] = (None, None)
    # shift test
    real = out[hm]["s"]
    L = len(idx) - s["f0"]
    if _ok(real) and L > 2 * GAP + 10:
        hits, vals = 0, []
        for kk in rng.integers(GAP, L - GAP, DRAWS):
            sh = pop & _sig_from(s["force"], s["price"], s["f0"], int(kk))
            mv = _mean(v, sh)
            if _ok(mv):
                vals.append(mv)
                hits += int(mv >= real)
        out["shift"] = hits / DRAWS
        out["shift_med"] = float(np.median(vals)) if vals else None
    else:
        out["shift"], out["shift_med"] = None, None
    out["per"] = {}
    for lab, a, b in PERIODS:
        m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        si, re = sig & m, rest & m
        if not (si & np.isfinite(v)).any():
            continue
        ms, mr = _mean(v, si), _mean(v, re)
        md2, _ = _matched(v, si & mok, re & mok, np.where(mok, cell, 0))
        out["per"][lab] = (int((si & np.isfinite(v)).sum()), (ms - mr) if _ok(ms) and _ok(mr) else None, md2)
    return out


# ---------- the rules ----------

def _close_pos(c, ma, t, hold, stop, rec, entry):
    """Exit checks at close t; returns (hold, ended)."""
    if not hold:
        return False, False
    if ma[t] > 0.5:
        rec.append({"e": entry, "x": t, "how": "ma"})
        return False, True
    if c[t] < stop:
        rec.append({"e": entry, "x": t, "how": "stop"})
        return False, True
    return True, False


def _single(f: dict, trig: np.ndarray) -> tuple[np.ndarray, list]:
    c, ma, low30 = f["c"], f["ma"], f["low30"]
    n = len(c)
    E = np.zeros(n)
    rec, hold, stop, entry = [], False, 0.0, -1
    for t in range(n):
        hold, ended = _close_pos(c, ma, t, hold, stop, rec, entry)
        if not hold and not ended and ma[t] < 0.5 and trig[t] and np.isfinite(low30[t]):
            hold, stop, entry = True, low30[t], t
        E[t] = 1.0 if hold else 0.0
    if hold:
        rec.append({"e": entry, "x": None, "how": None})
    return E, rec


def _trigger(f: dict, sig: np.ndarray) -> np.ndarray:
    fired = (f["pop"] & sig).astype(float)
    recent = pd.Series(fired).rolling(RECENT, min_periods=1).max().values > 0.5
    return recent & f["a10"]


def _machine(f: dict, arm: np.ndarray, exh: np.ndarray, extra) -> tuple[np.ndarray, list, list, int]:
    """Q1 (extra None: fire when armed and exhausted) or Q0 (exhaustion latched, fire when extra[t])."""
    c, ma = f["c"], f["ma"]
    n = len(c)
    E = np.zeros(n)
    rec, fires = [], []
    hold, stop, entry = False, 0.0, -1
    state, until, alow = 0, -1, 0.0
    for t in range(n):
        hold, ended = _close_pos(c, ma, t, hold, stop, rec, entry)
        if state > 0 and t > until:
            state = 0
        if arm[t]:
            if state == 0:
                state, alow = 1, c[t]
            until = t + ARM_WIN
        if state > 0:
            alow = min(alow, c[t])
        if state == 1 and exh[t]:
            state = 2
        if state == 2 and (extra is None or extra[t]):
            fires.append(t)
            state = 0
            if not hold and not ended and ma[t] < 0.5:
                hold, stop, entry = True, alow, t
        E[t] = 1.0 if hold else 0.0
    if hold:
        rec.append({"e": entry, "x": None, "how": None})
    return E, rec, fires, state


def _q1(f: dict, sigs: dict) -> tuple:
    arm = f["pop"] & (sigs["S1"] | sigs["S2"] | sigs["S3"])
    return _machine(f, arm, sigs["S5"], None)


def _q0(f: dict) -> tuple:
    with np.errstate(invalid="ignore"):
        arm = f["pop"] & (f["zr"] <= -2)
    return _machine(f, arm, f["nonew"], f["a20"])


def _entry_stats(c: np.ndarray, rec: list) -> dict:
    done = [r for r in rec if r["x"] is not None]
    ma_ = [r for r in done if r["how"] == "ma"]
    st = [r for r in done if r["how"] == "stop"]
    g = [c[r["x"]] / c[r["e"]] - 1 - COST for r in ma_]
    lo = [c[r["x"]] / c[r["e"]] - 1 - 2 * COST for r in st]
    return {"n": len(rec), "ma": len(ma_), "higher": sum(1 for r in ma_ if c[r["x"]] > c[r["e"]]),
            "stop": len(st), "open": len(rec) - len(done), "gain": float(np.mean(g)) if g else None,
            "loss": float(np.mean(lo)) if lo else None}


# ---------- the fair exam ----------

def _or(ma, f) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    return ma if f is None else np.maximum(ma, np.asarray(f, dtype=float))


class _Exam(pi._Exam):
    """price_impact's exam with the rule max(ma50, early-position flag)."""

    def btc_pos(self, flag) -> np.ndarray:
        return _or(self.ma, self._at(flag, self.close.index, 0))

    def summary(self, flag, ki: int, sims: int = SIMS) -> dict:
        pos = self.btc_pos(flag)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        f = self._at(flag, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        for h in range(24):
            w = _or(self.cma[h], f)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, _ = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :] = r
        each = {}
        for key, o in self.other.items():
            w = _or(o["ma"], self._at(flag, o["close"].index, o["lag"]))
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_edge": float(np.median(edge)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


def _above(x: dict, base: dict) -> bool:
    """Strictly better ranking score (eligibility first)."""
    if x["eligible"] != base["eligible"]:
        return bool(x["eligible"])
    return x["low"] > base["low"] + 1e-12


def _first_month(day) -> pd.Timestamp:
    day = pd.Timestamp(day)
    m = day if day.day == 1 else day + pd.offsets.MonthBegin(1)
    return max(m, pd.Timestamp(FIRST_MONTH))


def _holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    m = len(keys)
    adj, run, out = {}, 0.0, {}
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * p[k]))
        adj[k] = run
    stop = False
    for j, k in enumerate(keys):
        if stop or p[k] > FAMILY / (m - j):
            stop = True
            out[k] = (adj[k], False)
        else:
            out[k] = (adj[k], True)
    return out


# ---------- baseline (owner's addition) ----------

def _base_row(v: np.ndarray, deep: np.ndarray, m: np.ndarray) -> dict | None:
    ok = m & np.isfinite(v)
    if ok.sum() < 5:
        return None
    x = v[ok]
    dp = deep[ok]
    up, dn = x[x > 0], x[x <= 0]
    return {"n": int(ok.sum()), "up": float((x > 0).mean()), "down": float((x <= 0).mean()), "mean": float(x.mean()),
            "med": float(np.median(x)), "mu": float(up.mean()) if len(up) else None,
            "md": float(dn.mean()) if len(dn) else None, "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "dmean": float(dp.mean()), "dmed": float(np.median(dp)),
            "dworst": float(dp.min())}


def _baseline(f: dict, fwd: dict, idx, ep: np.ndarray) -> dict:
    a0 = int(np.argmax(f["allday"]))
    sl = slice(a0, None)
    rng = np.random.default_rng(SEED + 811)
    bi = _bidx(len(idx) - a0, rng)
    groups = (("母體（跌 25% 以上）", f["pop"]), (f"跌 {int(DEEPER[0] * 100)}% 以上", f["pop40"]),
              (f"跌 {int(DEEPER[1] * 100)}% 以上", f["pop55"]), ("所有日子", f["allday"]))
    out = {"rows": [], "diff": [], "eps": []}
    spans = [("全期", None, None)] + list(PERIODS)
    for lab, a, b in spans:
        per = np.ones(len(idx), dtype=bool) if a is None else ((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))
        for gname, g in groups:
            if a is not None and gname not in ("母體（跌 25% 以上）", "所有日子"):
                continue
            for h in H:
                row = _base_row(fwd[h], fwd[("deep", h)], g & per)
                if row:
                    out["rows"].append((lab, gname, h, row))
        for h in H:
            v = fwd[h]
            pm, am = f["pop"] & per, f["allday"] & per
            mp, ma_ = _mean(v, pm), _mean(v, am)
            if not (_ok(mp) and _ok(ma_)):
                continue
            ci = _q90(_bdiff(v[sl], pm[sl], am[sl], bi))
            up = np.where(np.isfinite(v), (v > 0).astype(float), np.nan)
            upd = _mean(up, pm) - _mean(up, am)
            uci = _q90(_bdiff(up[sl], pm[sl], am[sl], bi))
            out["diff"].append((lab, h, mp - ma_, ci, upd, uci))
    v60 = fwd[60]
    for k in sorted(set(ep[ep >= 0])):
        m = ep == k
        days = np.flatnonzero(m)
        j = days[int(np.argmin(f["dd"][days]))]
        out["eps"].append((idx[days[0]], idx[days[-1]], idx[j], float(f["dd"][j]), int(m.sum()), _mean(v60, m),
                           int((m & np.isfinite(v60)).sum())))
    return out


# ---------- rallies ----------

def _rally_rows(close: pd.Series, f: dict, sigs: dict, s_def: dict, fires: dict) -> list:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    idx = close.index
    rows = []
    for r in rallies:
        it, im = r["it"], r["im"]
        before, mid, none, nodata = [], [], [], []
        for k in SIG:
            if not s_def[k][max(0, it - 60):im + 1].any():
                nodata.append(k)
                continue
            b = sigs[k][max(0, it - 60):it].any()
            m = sigs[k][it:im + 1].any()
            if b:
                before.append(k)
            if m:
                mid.append(k)
            if not b and not m:
                none.append(k)
        fq = []
        for q in ("Q1", "Q0"):
            hit = [t for t in fires[q] if it - 60 <= t <= im]
            fq.append(f"{q} {_d(idx[hit[0]])}" if hit else f"{q} 沒有")
        rows.append([f"{_d(r['t'])} → {_d(r['p'])}" + ("" if r["done"] else "（還沒結束）"),
                     f"{_d(r['m'])}（晚 {r['lead_days']} 天，少 {_p(r['miss'])}）",
                     "、".join(before) or "沒有", "、".join(mid) or "沒有", "、".join(none) or "—",
                     "、".join(nodata) or "—", "；".join(fq)])
    return rows


# ---------- computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    idx = close.index
    n = len(idx)
    f = _price(close)
    sg = _signals(data, close, f)
    sigs = {k: sg[k]["sig"] for k in SIG}
    fwd = _forward(f["c"])
    cell = _cells(f)
    ep = _episodes(f["pop"])
    p0 = int(np.argmax(f["pop"]))
    rng = np.random.default_rng(SEED + 701)
    bi = _bidx(n - p0, rng)
    stats = {k: _sig_stats(k, sg[k], f, fwd, cell, ep, idx, p0, bi, rng) for k in SIG}
    base = _baseline(f, fwd, idx, ep)

    # rules
    E, rec, fires, state = {}, {}, {}, {}
    E["Q1"], rec["Q1"], fires["Q1"], state["Q1"] = _q1(f, sigs)
    E["Q0"], rec["Q0"], fires["Q0"], state["Q0"] = _q0(f)
    E["R0"], rec["R0"] = _single(f, f["pop"] & f["a10"])
    for k in SIG:
        E[k], rec[k] = _single(f, _trigger(f, sigs[k]))
    entries = {k: _entry_stats(f["c"], rec[k]) for k in E}

    # exam
    hrows, problem = cr._hourly("BTCUSDT")
    problems = [problem] if problem else []
    price, b0, _ = cr._grid(hrows)
    pre = cr._prehistory(b0)
    other, more = msc._load_other()
    problems += more
    starts = {"Q0": idx[0], "R0": idx[0]}
    for k in SIG:
        starts[k] = idx[sg[k]["f0"]]
    starts["Q1"] = max(idx[sg[k]["f0"]] for k in ("S1", "S2", "S3", "S5"))
    exams, base_scores = {}, {}
    ser = {k: pd.Series(v, index=idx) for k, v in E.items()}
    scores = {}
    for k in ("Q0", "R0") + RULES:
        fm = _first_month(starts[k])
        if fm not in exams:
            exams[fm] = _Exam(close, other, price, b0, pre, fm)
            ex = exams[fm]
            base_scores[fm] = {"ma50": ex.summary(None, KI["ma50"]), "Q0": ex.summary(ser["Q0"], KI["Q0"])}
        ex = exams[fm]
        x = base_scores[fm]["Q0"] if k == "Q0" else ex.summary(ser[k], KI[k])
        scores[k] = {"x": x, "ma50": base_scores[fm]["ma50"], "Q0": base_scores[fm]["Q0"], "first": fm}
    t_exam = time.time()

    # placebo for the 11 rules
    plc = {}
    for k in RULES:
        ex = exams[scores[k]["first"]]
        real_small = ex.summary(ser[k], KI[k], PL_SIMS)
        prng = np.random.default_rng(SEED + 31 * KI[k])
        if k == "Q1":
            parts = ("S1", "S2", "S3", "S5")
            L = n - max(sg[p]["f0"] for p in parts)
        else:
            parts = (k,)
            L = n - sg[k]["f0"]
        hits = 0
        lows = []
        for kk in prng.integers(GAP, L - GAP, DRAWS):
            sh = dict(sigs)
            for p in parts:
                sh[p] = _sig_from(sg[p]["force"], sg[p]["price"], sg[p]["f0"], int(kk))
            if k == "Q1":
                Es = _q1(f, sh)[0]
            else:
                Es = _single(f, _trigger(f, sh[k]))[0]
            x = ex.summary(pd.Series(Es, index=idx), KI[k], PL_SIMS)
            lows.append(x["low"])
            hits += int(sd._score_ge(x, real_small))
        plc[k] = {"share": hits / DRAWS, "p": (hits + 1) / (DRAWS + 1), "med": float(np.median(lows)),
                  "real_small": real_small["low"]}
    holm = _holm({k: plc[k]["p"] for k in RULES})
    rally = _rally_rows(close, f, sigs, {k: sg[k]["dfn"] for k in SIG}, fires)
    return {"close": close, "f": f, "sg": sg, "sigs": sigs, "stats": stats, "base": base, "E": E, "rec": rec,
            "fires": fires, "state": state, "entries": entries, "scores": scores, "plc": plc, "holm": holm,
            "rally": rally, "ep": ep, "problems": data["problems"] + problems, "secs": time.time() - t0,
            "secs_exam": t_exam - t0}


# ---------- the study ----------

def _verdict_rule(k: str, r: dict) -> dict:
    s = r["scores"][k]
    x, ma, q0 = s["x"], s["ma50"], s["Q0"]
    c1 = sd._score_ge(x, ma)
    c2 = x["c_deep_med"] >= ma["c_deep_med"] - LOSS_SLACK
    c3 = r["plc"][k]["share"] <= PLC_BAR
    c4 = _above(x, q0)
    return {"c": (c1, c2, c3, c4), "ok": c1 and c2 and c3 and c4}


def study() -> dict:
    data = _load()
    r = _compute(data)
    close, f, st, base, idx = r["close"], r["f"], r["stats"], r["base"], r["close"].index
    last = idx[-1]
    pop = f["pop"]
    ep = r["ep"]
    n_pop = int(pop.sum())
    n_eps = len(set(ep[pop]))
    n_pop90 = int((pop & np.isfinite(_forward(f["c"])[90])).sum())

    # ----- baseline tables -----
    def brow(lab, g, h, x):
        return [lab, g, f"{h} 天", f"{x['n']:,}", f"{_share(x['up'])}｜{_share(x['down'])}",
                f"{_p(x['mean'])}｜{_p(x['med'])}", f"{_p(x['mu'])}｜{_p(x['md'])}", f"{_p(x['p10'])}｜{_p(x['p90'])}",
                f"{_p(x['dmean'])}｜{_p(x['dmed'])}｜{_p(x['dworst'])}"]
    rows_base = [brow(*row) for row in base["rows"]]
    rows_bdiff = [[lab, f"{h} 天", f"{_pt(d)} {_ci(*ci)}", f"{_pt(u)} {_ci(*uci)}"]
                  for lab, h, d, ci, u, uci in base["diff"]]
    rows_eps = [[f"{i + 1}", f"{_d(a)} 到 {_d(b)}", f"{_d(j)}（離一年高點 {_p(dd)}）", f"{nd:,}",
                 f"{_p(m60)}（{n60:,} 天有 60 天後的資料）" if _ok(m60) else "—（還沒滿 60 天）"]
                for i, (a, b, j, dd, nd, m60, n60) in enumerate(base["eps"])]

    # ----- single-signal tables -----
    rows_sig, rows_add = [], []
    for k in SIG:
        x = st[k]
        hm = x["h"]

        def cmp(key):
            c = x[key]
            return f"{_p(c['s'])}｜{_p(c['r'])}；差 {_pt(c['diff'])} {_ci(*c['ci'])}"
        twin = x["twin"]
        per = "；".join(f"{lab} {_pt(d)}（配對 {_pt(m)}，{nd} 天）" for lab, (nd, d, m) in x["per"].items()) or "—"
        rows_sig.append([CN[k], _d(x["from"]) if x["from"] is not None else "—",
                         f"{x['n']:,} 天｜{x['eps']} 段（母體內有資料 {x['n_pop']:,} 天、{x['eps_pop']} 段）",
                         cmp(30), cmp(60), cmp(90), cmp(("deep", MAIN_H)),
                         f"{_p(twin[0])}（{twin[1]:,} 天）",
                         f"{_pt(x['m'])} {_ci(*x['m_ci'])}（配到 {_share(x['cov'])}）",
                         (f"{_share(x['shift'])}（錯開後中位數 {_p(x['shift_med'])}）" if _ok(x["shift"]) else "—"),
                         per])
        lo, hi = x["m_ci"]
        if x["n"] == 0:
            plain = "量到的：在母體裡一次都沒亮，無法判斷。"
        elif _ok(lo) and lo > 0 and _ok(x["shift"]) and x["shift"] <= 0.05:
            plain = (f"量到的：比「一樣大跌、但沒有這個訊號」的日子多 {_pt(x['m'])}（區間不含 0），錯開時間也很少一樣好"
                     f"（{_share(x['shift'])}）。注意：只有 {x['eps']} 段下跌。")
        elif _ok(x["m"]) and x["m"] > 0:
            plain = (f"量到的：方向是多 {_pt(x['m'])}，但區間 {_ci(lo, hi)} 含 0 或錯開時間也常一樣好"
                     f"（{_share(x['shift'])}）；用這個定義、這份資料分不出是訊號還是「本來就跌很多」。")
        elif _ok(x["m"]):
            plain = (f"量到的：配對後差 {_pt(x['m'])}（沒有比較多）；用這個定義、這份資料沒有看到它比「已經跌這麼多」多告訴什麼。"
                     "不代表想法錯，可能缺的東西見「待查線索」。")
        else:
            plain = "量到的：配不到對照日，無法判斷。"
        if x["n"] and x["cov"] < 0.2:
            plain += f"注意：有訊號的日子只有 {_share(x['cov'])} 配得到對照日，配對的數字不可靠。"
        rows_add.append([CN[k], f"{x['h']} 天", f"{_pt(x[hm]['diff'])}", f"{_pt(x['m'])} {_ci(lo, hi)}",
                         _share(x["shift"]) if _ok(x["shift"]) else "—", plain])

    # ----- exam -----
    rows_exam, verd = [], {}
    for k in RULES:
        s = r["scores"][k]
        x, ma, q0 = s["x"], s["ma50"], s["Q0"]
        v = _verdict_rule(k, r)
        verd[k] = v
        hp, hs = r["holm"][k]
        yn = lambda b: "✓" if b else "✗"  # noqa: E731
        rows_exam.append([("Q1 力量序列" if k == "Q1" else f"R({k})：{CN[k]}"),
                          f"{_d(s['first'])[:7]} 起：日線 {x['d_n']} 個｜收盤時間 {x['c_n']:,} 格｜其他資產 {x['o_n']} 個",
                          f"{_share1(x['low'])}｜{_share1(ma['low'])}｜{_share1(q0['low'])}",
                          f"{_pc(x['c_deep_med'])}｜{_pc(ma['c_deep_med'])}",
                          f"{_share(r['plc'][k]['share'])}（p={r['plc'][k]['p']:.3f}）",
                          f"{hp:.3f}" + ("（站得住）" if hs else "（站不住）"),
                          " ".join(f"{lab}{yn(b)}" for lab, b in zip(("分數", "虧損", "安慰劑", "贏Q0"), v["c"])),
                          "有支持（只能進每日影子紀錄觀察）" if v["ok"] else "這個定義、這份資料沒有看到"])
    rows_ref = []
    for k, name in (("Q0", "Q0 只看價格的雙胞胎"), ("R0", "R0「沒有力量資訊」的參考（不是候選規則）")):
        s = r["scores"][k]
        x, ma = s["x"], s["ma50"]
        rows_ref.append([name, f"{_d(s['first'])[:7]} 起：日線 {x['d_n']} 個｜收盤時間 {x['c_n']:,} 格｜其他資產 {x['o_n']} 個",
                         f"{_share1(x['low'])}｜{_share1(ma['low'])}",
                         " / ".join(_share(c) for c in x["checks"]),
                         f"{_pc(x['c_deep_med'])}｜{_pc(ma['c_deep_med'])}", f"{x['c_edge']:+.1f} 點"])

    # ----- entries -----
    rows_ent = []
    for k in ("Q1", "Q0", "R0") + SIG:
        e = r["entries"][k]
        name = {"Q1": "Q1 力量序列", "Q0": "Q0 價格雙胞胎", "R0": "R0 沒有力量資訊（參考）"}.get(k, f"R({k})")
        rows_ent.append([name, f"{e['n']}", (f"{e['higher']}／{e['ma']}（{_share(e['higher'] / e['ma'])}）" if e["ma"] else "—"),
                         _p(e["gain"]), f"{e['stop']} 次，平均 {_p(e['loss'])}" if e["stop"] else "0 次", f"{e['open']}"])

    n_pass = sum(1 for k in RULES if verd[k]["ok"])
    plc_pass = sum(1 for k in RULES if r["plc"][k]["share"] <= PLC_BAR)
    survivors = [k for k in RULES if r["holm"][k][1]]
    leads = [[("Q1 力量序列" if k == "Q1" else CN[k]), LEADS[k]] for k in RULES if not verd[k]["ok"]]

    # ----- verdict -----
    full_d = {(lab, h): (d, ci) for lab, h, d, ci, _, _ in base["diff"]}
    d60, ci60 = full_d.get(("全期", 60), (None, (None, None)))
    bp = [x for lab, g, h, x in base["rows"] if lab == "全期" and g.startswith("母體") and h == 60]
    q1 = r["scores"]["Q1"]
    q0own, r0own = r["scores"]["Q0"], r["scores"]["R0"]
    eps_m = [m for (_, _, _, _, _, m, _) in base["eps"] if _ok(m)]
    verdict = (
        f"量到的：母體（50 日均線空手、收盤比過去一年最高收盤低 25% 以上）有 {n_pop:,} 天，分屬 {n_eps} 段下跌"
        f"（日子多，但獨立的下跌只有這幾段）。這些日子之後 60 天平均 {_p(bp[0]['mean']) if bp else '—'}、中位數 "
        f"{_p(bp[0]['med']) if bp else '—'}；減掉所有日子的平均，差 {_pt(d60)} {_ci(*ci60)}；各段的 60 天平均從 "
        f"{_p(min(eps_m)) if eps_m else '—'} 到 {_p(max(eps_m)) if eps_m else '—'}，差很多。"
        f"量到的：11 條規則（1 個序列＋10 個單一訊號）裡，{n_pass} 條四個事先條件都過；單看安慰劑 5% 這條有 {plc_pass} 條過"
        f"（純靠運氣預期約 0.55 條）；Holm 校正後站得住的：{'、'.join(survivors) if survivors else '沒有'}。"
        f"Q1 排名分數 {_share1(q1['x']['low'])}，同起算日的 50 日均線 {_share1(q1['ma50']['low'])}、Q0 {_share1(q1['Q0']['low'])}。"
        f"參考：只看價格的 Q0 {_share1(q0own['x']['low'])}、沒有力量資訊的 R0 {_share1(r0own['x']['low'])}，"
        f"均線 {_share1(q0own['ma50']['low'])}（2015 起）。"
        "沒有通過的訊號，讀法是「用這個定義、這份資料沒有看到它比『一樣大跌但沒訊號』的日子多告訴什麼」，不是想法錯了；"
        "可能缺的參數、交易所、頻率列在「待查線索」。"
        + "".join(f"量到的：{k} 四個條件都過，但安慰劑 {_share(r['plc'][k]['share'])}（門檻 5%）、Holm 校正後 p={r['holm'][k][0]:.2f}"
                  f"（{'站得住' if r['holm'][k][1] else '站不住'}）；"
                  + (f"在訊號那一層，配對後差 {_pt(st[k]['m'])} {_ci(*st[k]['m_ci'])}"
                     + ("，區間含 0" if not (_ok(st[k]['m_ci'][0]) and st[k]['m_ci'][0] > 0) else "")
                     + "。" if k in st else "")
                  + "推測：可能是真的多一點資訊，也可能是 11 條裡碰巧過關的那一條，這份資料分不出來；只能放進每日影子紀錄觀察。"
                  for k in RULES if verd[k]["ok"])
    )

    # ----- today -----
    word = {1: "持有", 0: "現金"}
    E = r["E"]
    ma_now = int(f["ma"][-1] > 0.5)
    stn = {0: "未啟動", 1: "已啟動", 2: "已耗盡"}
    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元，"
            f"離過去一年最高收盤 {_p(f['dd'][-1])}"},
           {"label": "比特幣：50 日均線｜Q1｜Q0",
            "value": f"{word[ma_now]}｜{word[max(ma_now, int(E['Q1'][-1] > 0.5))]}｜{word[max(ma_now, int(E['Q0'][-1] > 0.5))]}"},
           {"label": "今天在不在母體（均線空手且比一年高點低 25% 以上）", "value": "在" if pop[-1] else "不在"},
           {"label": "Q1 狀態機", "value": stn[r["state"]["Q1"]] + ("（提早持有中）" if E["Q1"][-1] > 0.5 else "")},
           {"label": "Q0 狀態機", "value": stn[r["state"]["Q0"]] + ("（提早持有中）" if E["Q0"][-1] > 0.5 else "")}]
    for k in SIG:
        s = r["sg"][k]
        v = "還沒有今天的資料" if not s["dfn"][-1] else ("亮" if s["sig"][-1] else "沒亮")
        now.append({"label": f"今天 {CN[k]}", "value": v})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；沒有通過的規則不會用"})

    first_pop = idx[int(np.argmax(pop))]
    tables = [
        {"caption": ("基準（只描述，沒有任何訊號）：大跌之後本來會怎樣。母體＝50 日均線空手、收盤比過去 365 天最高收盤低 25% 以上的日子；"
                     f"另外列更深的 40%、55%（同樣要均線空手），以及「所有日子」（從 {_d(idx[int(np.argmax(f['allday']))])} 起）。"
                     "每格：「上漲的比例｜下跌的比例」、「平均｜中位數」、「漲的那些平均｜跌的那些平均」、「第 10｜第 90 百分位」、"
                     "「期間內再跌最深：平均｜中位數｜最差」。"),
         "columns": ["期間", "日子", "之後", "天數", "漲｜跌的比例", "平均｜中位數", "漲時平均｜跌時平均", "第 10｜90 百分位",
                     "再跌最深：平均｜中位數｜最差"], "rows": rows_base},
        {"caption": (f"基準：母體減所有日子的差，90% 區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次（日子相連，不能當獨立）。"),
         "columns": ["期間", "之後", "平均報酬差 ［90% 區間］", "上漲比例差 ［90% 區間］"], "rows": rows_bdiff},
        {"caption": (f"基準：每一段下跌（母體日子按日期排，中間超過 {EP_GAP} 天沒有母體日子就算新的一段）。"
                     "看平均是不是被一兩段撐起來。"),
         "columns": ["段", "母體日子的範圍", "最深的一天", "母體天數", "這段日子之後 60 天平均"], "rows": rows_eps},
        {"caption": ("十個單一訊號：每個的道理和量法（推測的道理是要測的假設，不是已知事實）。"),
         "columns": ["訊號", "道理與量法"], "rows": [[CN[k], LOGIC[k]] for k in SIG]},
        {"caption": ("單一訊號（只在母體、而且這個訊號算得出來的日子裡比）：每格「有訊號｜母體其他天；差 ［90% 區間］」。"
                     "「價格雙胞胎」＝只看訊號裡的價格條件（例如『當天是跌的』）的母體日子，沒有價格條件的訊號就是母體本身。"
                     "「配對」＝同一格（離一年高點跌多深、過去 30 天漲跌、20 天波動，各分十等份）裡有訊號和沒訊號的日子比。"
                     f"「錯開時間」＝把訊號的力量部分整條挪 {GAP} 天以上（頭尾相接）{DRAWS} 次，價格條件不動。主要期間 60 天（S10 是 90 天）。"),
         "columns": ["訊號", "資料從", "亮的天數｜段數", "之後 30 天", "之後 60 天", "之後 90 天", "60 天內再跌最深",
                     "價格雙胞胎（主要期間）", "配對後差（主要期間）［90%］", "錯開時間也一樣好的比例", "分期：主要期間差"],
         "rows": rows_sig},
        {"caption": ("白話：這個訊號比「價格已經跌這麼多」多告訴了什麼？常聽到「大跌後本來就比較容易反彈」（推測，不是這裡證明的；"
                     f"本研究量到的：母體之後 60 天平均減所有日子是 {_pt(d60)} {_ci(*ci60)}，見基準表）。不管大跌後本身是好是壞，"
                     "資金費率變成負的，多半也發生在大跌後；所以要拿「一樣大跌、但資金費率沒有變負」的日子來比，"
                     "才知道資金費率有沒有多告訴我們什麼。下表每個訊號都這樣比。"),
         "columns": ["訊號", "主要期間", "比母體其他天多", "比一樣大跌但沒訊號的日子多（配對）［90%］", "錯開時間也一樣好", "白話"],
         "rows": rows_add},
        {"caption": ("公平考試（main_signal_check 的做法）：規則＝50 日均線持有，或「提早上車的部位還沒被停損、均線還沒接手」也持有。"
                     "每條規則從自己的資料算得出來的那個月起算（最早 2015-01），50 日均線和 Q0 用完全相同的起算日重算。"
                     f"一般人做法：晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次。"
                     f"隨機錯位：力量部分挪 {GAP} 天以上 {DRAWS} 次，整個考試重算（一般人做法每個收盤時間 {PL_SIMS} 次，真規則也這樣重算來比）；"
                     "Q1 的 S1、S2、S3、S5 一起挪同樣天數。事先定的「有支持」要四個都過：排名分數不輸均線、最深虧損中位數最多深 5 點、"
                     f"安慰劑 ≤ 5%、排名分數比 Q0 高。11 條、5% 的門檻，純靠運氣預期約 0.55 條會過安慰劑這關；Holm 校正控制整組 5%。"),
         "columns": ["規則", "起算", "排名分數：規則｜均線｜Q0", "最深虧損中位數：規則｜均線", "錯位不輸的比例", "Holm 校正後 p",
                     "四個條件", "讀法"], "rows": rows_exam},
        {"caption": ("參考（不算在 11 條裡）：只看價格的 Q0，以及「沒有力量資訊」的 R0（母體日子＋站上 10 日均線就進、"
                     "停損＝前 30 天最低收盤、均線亮就交棒）。和 50 日均線用同樣的起算日（2015-01 起）。"
                     "四項：日線起算日／起算日×收盤時間／一般人做法／其他資產，贏過一直抱著的比例。"),
         "columns": ["規則", "起算", "排名分數：規則｜均線", "四項", "最深虧損中位數：規則｜均線", "每年比抱著多（中位數）"],
         "rows": rows_ref},
        {"caption": ("只描述：每條規則提早上車的結果（2011 年起全部歷史，比特幣日線）。「交棒」＝持有中 50 日均線亮了，由均線接手；"
                     "「交棒時價格更高」＝交棒那天收盤比進場高。賺賠扣每次 0.10% 成本。"),
         "columns": ["規則", "提早上車次數", "交棒時價格更高（次／交棒次數）", "交棒前平均賺", "被停損", "還在持有"],
         "rows": rows_ent},
        {"caption": ("只描述：rally_sources 的大漲（低點後漲 80% 以上、中途沒跌 30%）。每段：低點前 60 天亮過哪些訊號、"
                     "低點到均線亮之間亮過哪些、都沒亮的、還沒有資料的（這裡算任何一天，不限母體）。也列 Q1、Q0 在低點前 60 天到均線亮之間"
                     "第一次發動的日子。有些大漲沒有任何訊號，例如新聞帶動的開始；這不算失敗，只是這些力量量不到。"),
         "columns": ["大漲（低點 → 高點）", "均線亮（比低點晚幾天、少賺多少）", "低點前 60 天亮過", "低點到均線亮之間亮過",
                     "都沒亮", "還沒有資料", "Q1／Q0 發動"], "rows": r["rally"]},
        {"caption": ("待查線索：沒有被支持的訊號，用這個定義、這份資料沒有看到；它可能缺的東西（推測，沒有測）。"),
         "columns": ["訊號", "可能缺的東西"], "rows": leads or [["—", "—"]]},
    ]

    notes = [
        f"量到的：母體從 {_d(first_pop)} 開始，共 {n_pop:,} 天（其中 {n_pop90:,} 天有完整 90 天後的資料），分屬 {n_eps} 段下跌。"
        "相鄰日子的未來報酬大部分重疊，天數多不代表證據多。",
        "標記方式：「量到的」＝本研究算出來的數字；「推測」＝道理或猜測，沒有被這裡的數字證明；本研究沒有引用、也沒有打開任何文獻來支持這些說法，"
        "所以沒有「文獻說的」這一類。",
        "rally_timing（量到的，該研究）：2013 年以來 7 次腰斬的空頭，50 日均線上車價中位數比低點高 38%；「跌深站上 20 日均線」最接近低點，"
        "上車價中位數比均線低 5 個百分點，但假反彈較多；沒有一種提早做法通過它事先定的三個條件。episodes（量到的，該研究）："
        "2021 年以來「14 天跌 12% 以上」19 次，滿 30 天的 18 次裡 8 次漲超過 5%、7 次再跌超過 5%。",
        "Coin Metrics 的交易所流量用的是今天的地址標籤往回套，越早的恐慌量得越少；資料也會被改寫，這裡只用今天的版本。",
        "OBM 是一位作者 2026 年整理的資料（CC BY 4.0，只下載資料檔，沒有下載或執行它的程式）；幣被花掉不一定是被賣掉。",
        "資金費率和主動買賣只用幣安；成交量只用 Bitstamp；Coinbase 溢價是 Coinbase 和 Bitstamp 日收盤的比。",
        "提早上車部位用比特幣 UTC 0 點的日收盤算；在其他收盤時間和其他資產上，它當成全市場的旗子（各資產自己的 50 日均線加上它）。",
        "門檻（2 個標準差、第 5／30／80 百分位、10／20／30／60／90／365 天）都是跑之前憑道理定的，沒有試其他數字。",
        "就算通過，也只代表可以放進每日影子紀錄觀察；用的是同一段歷史。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（考試前 {r['secs_exam'] / 60:.1f} 分鐘）。",
    ]
    notes += r["problems"]

    return {
        "id": "after_fall_forces",
        "title": "大跌之後的買賣力量：能不能比均線早上車",
        "question": ("大跌之後、大漲之前，買方和賣方的力量會不會先出現異常（恐慌轉進交易所、老手認賠、爆量、空單擁擠、賣壓枯竭、"
                     "買方在接、場邊資金）？這些訊號單獨或照順序出現時，比「價格已經跌這麼多」多告訴我們什麼？能不能比 50 日均線早上車，"
                     "而且事先定好停損？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群版 API（FlowInExNtv、SplyExNtv、SplyCur）",
             "url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=FlowInExNtv,SplyExNtv,SplyCur&frequency=1d"},
            {"name": "Open Bitcoin Metrics 資料檔：放 1 年以上的幣被花掉（只下載資料，CC BY 4.0）", "url": OBM_URL},
            {"name": "Open Bitcoin Metrics 專案頁（授權與說明）", "url": ocs.OBM_REPO},
            {"name": "Bitstamp BTC/USD 日線（含成交量）", "url": "https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000"},
            {"name": "幣安 BTCUSDT 永續合約資金費率", "url": FUND_URL},
            {"name": "幣安現貨 BTCUSDT 日線（成交金額、主動買進金額）", "url": SPOT_URL},
            {"name": "Coinbase BTC-USD 日線", "url": km.COINBASE_URL},
            {"name": "DefiLlama 穩定幣總供給", "url": sd.LLAMA},
            {"name": "幣安現貨 BTCUSDT 1 小時線（公平考試）", "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"},
            {"name": "幣安現貨日線（ETH、SOL、BNB、XRP 對 USDT）", "url": "https://api.binance.com/api/v3/klines?interval=1d"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
