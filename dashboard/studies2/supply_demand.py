"""Historical study "supply_demand": does bitcoin rise a lot when FEW holders want to sell (supply tight) AND buying
starts to come in (demand on) at the same time - and can that get a holder in EARLIER than the main signal ma50?

Hypothesis (reasoned from how prices form, written before any test): a price moves most when the sellers who would
meet new buyers are scarce and new buyers actually arrive. Either side alone can be harmless (few sellers but nobody
buying; buyers arriving but meeting plenty of supply). Earlier studies tested each side ALONE and every one failed:
  supply side - exchange_reserves (30-day exchange-balance falls: forward returns opposite to the paper, rules lose
    to ma50), sth_cost (short-term-holder cost basis: rules lose to ma50, mostly a disguised 87-day average), puell
    (miner-selling pressure: rare events, rules lose to ma50, placebo > 10%);
  demand side - stablecoin_supply (supply growth FOLLOWS price more than it leads; S1/S2 fail), etf_flows (flows
    follow price; short sample), coinbase_premium (premium > 0 adds +3.2 points over 30 days with an interval through
    zero; the ma50 filter fails the fair exam; lead_followup's pre-registered L1 also failed, placebo 8% but ranking
    score 25% vs 80%).
Nobody tested the INTERACTION. Practical goal: ma50 (hold while the daily close is above its 50-day average) and all
filters on it can only avoid losses; this is an attempt to get in EARLIER than ma50 and catch upside ma50 misses.
rally_timing tried earlier entries from PRICE ONLY (e.g. "deep fall, then close above the 20-day average") around
50%-bear-market lows; here the early entry needs supply AND demand conditions and is not limited to bear lows.

Definitions (fixed before running; each uses only values known at the BTC daily close; the decision at the close of
UTC candle D is held during D+1):
  Prices: the dashboard's daily closes (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Coin Metrics community API (free, key-less; FlowInExNtv, CapMVRVCur): a daily value dated D is used from the close
    of candle D+1 (one full day late; the provider publishes D a few hours after D ends).
  DefiLlama total stablecoin supply (/stablecoincharts/all, sum of every peg): value dated D used from the close of
    D+1, as stablecoin_supply. stablecoin_supply's cleaning is kept: growth only from 2019-07-01 (coverage before is
    incomplete: USDT on Omni missing), daily log changes beyond +-15% are coverage jumps and set to zero; 30-day
    growth = sum of 30 cleaned daily log changes (> 0 exactly when supply grew).
  Coinbase premium: coinbase_premium's dollar-to-dollar version (Coinbase BTC-USD / Bitstamp BTC/USD daily close - 1,
    both candles close at 00:00 UTC), 7-day mean with at least 5 of 7 days; lead_followup's convention: at the close
    of candle D the 7-day mean through D is known. Series from 2017-08-17.
  TIGHT = both (1) t-flow: the sum of FlowInExNtv (BTC sent to exchanges) over the last 30 days is below the median of
    that 30-day sum over the 365 previous days (at least 330 values); (2) t-mvrv: CapMVRVCur (market value / what
    holders paid) below its median over the 730 previous days (at least 660 values).
  DEMAND = at least one of (1) d-stable: 30-day stablecoin growth > 0 AND above its median over the 365 previous days
    (at least 330); (2) d-premium: Coinbase premium 7-day mean > 0.
  A leg whose value is missing or whose series does not yet exist counts as OFF. First usable dates (decision dates)
  are reported in the first table; d-stable cannot be on before about 2020-08 (2019-07 + 30 + 330 days + 1),
  d-premium from about 2017-08-22.
  JOINT = TIGHT and DEMAND. ma50 = close above its 50-day average (main_signal_check). above20 = close above the
  average of the last 20 closes including this one.

Test A - the interaction itself (main scientific result). Sample: every UTC decision day from the first day on which
  both TIGHT legs and the premium leg exist (so DEMAND is measurable; earlier days would all be "no demand") to the
  last day with a full forward window. Four cells: tight & demand, tight only, demand only, neither. Forward returns
  close[t+30]/close[t] - 1 and close[t+60]/close[t] - 1 from the decision close t. Per cell: days, mean and median
  30- and 60-day forward return, 90% intervals from a circular block bootstrap of days (blocks of 60 days, 1,000
  draws, fixed seed) inside each period. Periods: full; "2015-19" (which, given the data, starts with the sample,
  about 2017-08); 2020-22; 2023-now. Interaction = (tight&demand - tight only) - (demand only - neither) on the mean
  30-day forward return, same bootstrap. Pre-registered pass: tight&demand has the highest mean 30-day forward return
  of the four cells in the full period AND in at least two of the three sub-periods AND the full-period interaction's
  90% interval excludes zero. Report only: the same table on days when ma50 is OFF (cells on other days blanked, the
  bootstrap still draws calendar blocks).

Test B - the rule. S1 = hold when ma50 is on OR (JOINT and above20). Fair exam = main_signal_check's, wired as
  lead_followup does: monthly UTC starts 2017-09 .. 2023-12 (the first month JOINT can be on; 76 starts, textbook
  0.10%), 72 starts (2018-01 .. 2023-12) x 24 closing hours textbook, the same cells as an ordinary person (decide at
  the clock's close, act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock,
  main_signal_check's seed and its ma50 draws), other assets ETH SOL BNB XRP (Binance) and MSTR (Yahoo, from
  2020-08-11) with their own 50- and 20-day averages and BTC's JOINT as a market-wide flag, starts from
  max(main_signal_check's first month, 2017-09) to three years before today. ma50 is re-scored on exactly the same
  starts. On the 24-clock grid a close on UTC date E uses JOINT decided at the UTC close of candle E-1; MSTR's close
  on date D uses JOINT of candle D-1; other coins' UTC closes use JOINT of the same candle. Ranking score = worst of
  the four "beats holding" shares with main_signal_check's eligibility (deepest loss shallower than holding in >= 95%
  of BTC starts and cells). Median deepest loss = median over the 1,728 textbook cells.
  Placebo: JOINT (on its decision dates, from the Test A sample start) circularly shifted by 180 ... (n - 180) days,
  200 draws, the whole exam rebuilt; to stay within minutes the ordinary-person part uses 20 simulations per clock
  (same seeds) and the real S1 is re-scored the same way; placebo share = share of draws whose ranking score is at
  least the real rule's (main_signal_check's order).
  Pre-registered pass, all three: ranking score >= ma50's; median deepest loss no more than 5 points worse than
  ma50's; placebo share <= 10%.
Report only: S1's extra days held versus ma50 (UTC series from 2017-09-01; count, by year, average next-day BTC return
  on those days); calendar-year return of S1 minus ma50 from 2018 (0.10% cost); early entries (S1 turns on while ma50
  is off): "right" = ma50 turns on within 30 days at a higher close than the entry close, else "wrong"; each entry's
  stint runs to the first close where S1 goes to cash or ma50 turns on, its BTC return minus 0.10% per trade made in
  it; each leg alone in place of the pair (T: ma50 OR (TIGHT and above20); D: ma50 OR (DEMAND and above20)) on the
  same exam (no placebo); who moves first between each demand leg and price: corr(leg over the past 30 days, BTC log
  return over the next 30) versus corr(BTC log return over the past 30 days, leg over the next 30), daily overlapping
  windows, 90% noise band from 200 circular shifts (>= 180 days) of the leg; "leg over 30 days" = mean daily premium,
  or the cleaned 30-day log growth of stablecoin supply.
Look-ahead check: inputs cut at three dates (Coin Metrics and DefiLlama one day earlier than prices and premium);
  no flag or S1 position up to the cut may change.
Honesty: the thresholds (medians, 30 / 365 / 730 days, the 20-day average, 60-day blocks) were fixed by reasoning
  before the run and no alternatives were tried. Coin Metrics MVRV and exchange flows are revised and exchange
  address tagging improves over time; only today's vintage is used (flow_filters found the whole history rewritten in
  April 2026). A pass would only make S1 a candidate for the daily shadow log.
Only free key-less endpoints the project already uses, through common.get_json / kimchi_premium._get (fixed user
agent "btc-research-dashboard"). Coinbase and Bitstamp candles are read from the stores of kimchi_premium and
coinbase_premium (read only; missing recent days fetched into memory, never written); this study keeps its own
stores supply_demand_cm.json and supply_demand_llama.json. Not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import coinbase_premium as cp
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = ("FlowInExNtv", "CapMVRVCur")
LLAMA = "https://stablecoins.llama.fi/stablecoincharts/all"
CB_URL = km.COINBASE_URL
BS_URL = cp.BS_URL.format(pair="btcusd")
STORE_CM = common.STORE / "supply_demand_cm.json"
STORE_LL = common.STORE / "supply_demand_llama.json"
BS_STORE = common.STORE / "coinbase_premium_bitstamp_btcusd.json"

FIRST = cp.FIRST  # 2017-08-17
STABLE_FROM, JUMP = "2019-07-01", 0.15
SUM_DAYS, FLOW_WIN, FLOW_MIN = 30, 365, 330
MVRV_WIN, MVRV_MIN = 730, 660
G_DAYS, G_WIN, G_MIN = 30, 365, 330
C7_DAYS, C7_MIN = 7, 5
A20 = 20
H = (30, 60)
BLOCK, BOOT = 60, 1000
GRID = ("2017-09-01", msc.DAILY_STARTS[1])
DRAWS, GAP, PL_SIMS = 200, 180, 20
LL_DRAWS = 200
LOSS_SLACK = 5.0
KI = {"ma50": 1, "s1": 90, "t": 91, "d": 92}  # ma50 = main_signal_check's draws
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")
PERIODS = (("2015–19", "2015-01-01", "2019-12-31"), ("2020–22", "2020-01-01", "2022-12-31"),
           ("2023 起", "2023-01-01", "2100-01-01"))
CELLS = ("賣壓少＋買盤來", "只有賣壓少", "只有買盤來", "兩個都沒有")
NAME = {"ma50": "50 日均線（基準）",
        "s1": "S1：50 日均線，或「賣壓少＋買盤來＋站上 20 日均線」也持有",
        "t": "只用賣壓少：50 日均線，或「賣壓少＋站上 20 日均線」",
        "d": "只用買盤來：50 日均線，或「買盤來＋站上 20 日均線」"}
SHORT = {"ma50": "50 日均線", "s1": "S1", "t": "只賣壓少", "d": "只買盤來"}

_d = cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v) -> str:
    """fraction -> signed percent"""
    return f"{float(v) * 100:+.1f}%" if _ok(v) else "—"


def _pt(v) -> str:
    """fraction difference -> signed points"""
    return f"{float(v) * 100:+.1f} 點" if _ok(v) else "—"


def _pc(v) -> str:
    """already percent"""
    return f"{float(v):.1f}%" if _ok(v) else "—"


def _share(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _share1(v) -> str:
    return f"{float(v) * 100:.1f}%" if _ok(v) else "—"


def _ci(lo, hi, f=_p) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _stored(path, fetch) -> tuple[dict, str | None]:
    """Fetch fresh data; on failure use what an earlier run of this study stored."""
    try:
        got = fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        return got, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"{path.name} 取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        return got, f"{path.stem} 這次取不到（{type(exc).__name__}），用之前存下來的資料。"


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


def _llama_fetch() -> dict:
    reply = common.get_json(LLAMA, timeout=60)
    pts = {}
    for p in reply:
        tot = sum(float(v) for v in (p.get("totalCirculatingUSD") or {}).values() if v is not None)
        if tot > 0:
            pts[pd.Timestamp(int(p["date"]), unit="s").strftime("%Y-%m-%d")] = tot
    if len(pts) < 1500:
        raise RuntimeError(f"DefiLlama 只有 {len(pts)} 天")
    return pts


def _coinbase() -> tuple[dict, str | None]:
    """{day: close} of Coinbase BTC-USD daily candles: kimchi_premium's store read only, missing days in memory."""
    stored = dict(km._read(km.STORE_CB))
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - 5 * DAY) if stored else pd.Timestamp("2017-08-01")
    try:
        while start < today:
            end = min(start + pd.Timedelta(days=299), today)
            page = km._get(CB_URL, {"granularity": 86400, "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                                    "end": end.strftime("%Y-%m-%dT00:00:00Z")}).json()
            for row in page:
                day = pd.Timestamp(int(row[0]), unit="s")
                if day < today and float(row[4]) > 0:
                    stored[_d(day)] = float(row[4])
            start = end + DAY
        return stored, None
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        return stored, f"Coinbase 這次沒有更新成功（{type(exc).__name__}），用之前存下來的資料（到 {max(stored)}）。"


def _bitstamp() -> tuple[dict, str | None]:
    """{day: close} of Bitstamp BTC/USD daily candles: coinbase_premium's store read only, missing days in memory."""
    stored = {k: (v[0] if isinstance(v, list) else v) for k, v in km._read(BS_STORE).items()}
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - 5 * DAY) if stored else pd.Timestamp("2017-07-01")
    try:
        for _ in range(12):
            page = km._get(BS_URL, {"step": 86400, "limit": 1000,
                                    "start": int(start.timestamp())}).json()["data"]["ohlc"]
            if not page:
                break
            for row in page:
                day = pd.Timestamp(int(row["timestamp"]), unit="s")
                if day < today and float(row["close"]) > 0:
                    stored[_d(day)] = float(row["close"])
            nxt = pd.Timestamp(int(page[-1]["timestamp"]), unit="s") + DAY
            if nxt <= start or nxt >= today:
                break
            start = nxt
        return stored, None
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        return stored, f"Bitstamp 這次沒有更新成功（{type(exc).__name__}），用之前存下來的資料（到 {max(stored)}）。"


def _load() -> dict:
    problems = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    cm_raw, n = _stored(STORE_CM, _cm_fetch)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(cm_raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
    ll_raw, n = _stored(STORE_LL, _llama_fetch)
    problems += [n] if n else []
    sup = pd.Series(ll_raw, dtype=float)
    sup.index = pd.to_datetime(sup.index)
    sup = sup.sort_index()
    sup = sup[sup.index < today].asfreq("D").ffill()
    cb, n = _coinbase()
    problems += [n] if n else []
    bs, n = _bitstamp()
    problems += [n] if n else []
    days = pd.date_range(FIRST, close.index[-1], freq="D")
    cbs = pd.Series({pd.Timestamp(k): v for k, v in cb.items()}).sort_index().reindex(days)
    bss = pd.Series({pd.Timestamp(k): v for k, v in bs.items()}).sort_index().reindex(days)
    prem = cbs / bss - 1
    other, more = msc._load_other()
    problems += more
    return {"close": close, "cm": cm, "sup": sup, "prem": prem, "other": other, "problems": problems}


# ---------- legs (all on their own data dates) ----------

def _stable_dl(sup: pd.Series) -> pd.Series:
    lv = sup[sup.index >= pd.Timestamp(STABLE_FROM)]
    g = np.log(lv.where(lv > 0)).diff().iloc[1:]
    return g.where(g.abs() <= JUMP, 0.0)


def _raw_legs(cm: pd.DataFrame, sup: pd.Series, prem: pd.Series) -> dict:
    """Each leg on its data dates: (flag 1/0/NaN, value, threshold)."""
    inflow = cm["FlowInExNtv"] if "FlowInExNtv" in cm else pd.Series(dtype=float)
    in30 = inflow.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    in_med = in30.shift(1).rolling(FLOW_WIN, min_periods=FLOW_MIN).median()
    mv = cm["CapMVRVCur"] if "CapMVRVCur" in cm else pd.Series(dtype=float)
    mv_med = mv.shift(1).rolling(MVRV_WIN, min_periods=MVRV_MIN).median()
    dl = _stable_dl(sup)
    g30 = dl.rolling(G_DAYS, min_periods=G_DAYS).sum()
    g_med = g30.shift(1).rolling(G_WIN, min_periods=G_MIN).median()
    c7 = prem.rolling(C7_DAYS, min_periods=C7_MIN).mean()

    def flag(cond, *parts):
        ok = pd.concat(parts, axis=1).notna().all(axis=1)
        return cond.astype(float).where(ok)

    return {"t_flow": (flag(in30 < in_med, in30, in_med), in30, in_med),
            "t_mvrv": (flag(mv < mv_med, mv, mv_med), mv, mv_med),
            "d_stable": (flag((g30 > 0) & (g30 > g_med), g30, g_med), g30, g_med),
            "d_prem": (flag(c7 > 0, c7), c7, None)}


LEG_LAG = {"t_flow": 1, "t_mvrv": 1, "d_stable": 1, "d_prem": 0}
LEG_NAME = {"t_flow": "近 30 天轉進交易所的比特幣少於過去一年中位數", "t_mvrv": "MVRV 低於過去兩年中位數",
            "d_stable": "穩定幣 30 天有增加，而且增幅高於過去一年中位數", "d_prem": "Coinbase 溢價 7 日平均 > 0"}


def _legs(raw: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Flags on UTC decision dates (missing = NaN in *_def, 0 in the flag)."""
    out = pd.DataFrame(index=index)
    for k, (f, _, _) in raw.items():
        v = f.reindex(index - LEG_LAG[k] * DAY).values
        out[k + "_def"] = ~np.isnan(v)
        out[k] = np.nan_to_num(v, nan=0.0)
    out["tight"] = out["t_flow"] * out["t_mvrv"]
    out["demand"] = np.maximum(out["d_stable"], out["d_prem"])
    out["joint"] = out["tight"] * out["demand"]
    return out


def _above20(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    m = pd.Series(c).rolling(A20, min_periods=A20).mean().values
    return np.where(np.isnan(m), 0.0, (c > m).astype(float))


def _pos(ma: np.ndarray, a20: np.ndarray, flag) -> np.ndarray:
    ma = np.nan_to_num(np.asarray(ma, dtype=float), nan=0.0)
    if flag is None:
        return ma
    return np.maximum(ma, np.asarray(flag, dtype=float) * a20)


# ---------- Test A ----------

def _boot_idx(n: int, rng) -> np.ndarray:
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (BOOT, nb))
    return ((st[:, :, None] + np.arange(BLOCK)) % n).reshape(BOOT, -1)[:, :n]


def _cells_stats(cell: np.ndarray, fwd: dict, rng) -> dict:
    """cell in 0..3 (-1 = left out). Means, medians, bootstrap intervals, interaction on 30-day means."""
    n = len(cell)
    idx = _boot_idx(n, rng)
    ci = cell[idx]
    out = {"n": [int((cell == k).sum()) for k in range(4)]}
    bmeans30 = []
    for h in H:
        r = fwd[h]
        rb = r[idx]
        for k in range(4):
            m = (cell == k) & ~np.isnan(r)
            mean = float(r[m].mean()) if m.any() else None
            med = float(np.median(r[m])) if m.any() else None
            mb = (ci == k) & ~np.isnan(rb)
            cnt = mb.sum(axis=1)
            bm = np.where(cnt > 0, np.where(mb, rb, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            with np.errstate(all="ignore"):
                bmed = np.nanmedian(np.where(mb, rb, np.nan), axis=1) if mb.any() else np.full(BOOT, np.nan)
            if h == 30:
                bmeans30.append(bm)

            def q(a):
                a = a[~np.isnan(a)]
                return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)
            out[(h, k)] = {"mean": mean, "med": med, "mean_ci": q(bm), "med_ci": q(bmed)}
    means = [out[(30, k)]["mean"] for k in range(4)]
    if all(_ok(v) for v in means):
        out["inter"] = (means[0] - means[1]) - (means[2] - means[3])
        bi = (bmeans30[0] - bmeans30[1]) - (bmeans30[2] - bmeans30[3])
        bi = bi[~np.isnan(bi)]
        out["inter_ci"] = (float(np.percentile(bi, 5)), float(np.percentile(bi, 95))) if len(bi) > 50 else (None, None)
    else:
        out["inter"], out["inter_ci"] = None, (None, None)
    out["top"] = (int(np.nanargmax([v if _ok(v) else -np.inf for v in means]) == 0)
                  if any(_ok(v) for v in means) else 0)
    return out


def _test_a(close: pd.Series, lg: pd.DataFrame, ma: np.ndarray, start: pd.Timestamp) -> dict:
    c = close.values
    n = len(c)
    fwd_all = {}
    for h in H:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        fwd_all[h] = f
    cell_all = np.where(lg["tight"].values > 0, np.where(lg["demand"].values > 0, 0, 1),
                        np.where(lg["demand"].values > 0, 2, 3))
    keep = (close.index >= start) & ~np.isnan(fwd_all[30])
    rng = np.random.default_rng(SEED + 501)
    res = {"full": {}, "off": {}}
    spans = [("全期", start, close.index[-1])] + [(lab, max(pd.Timestamp(a), start), pd.Timestamp(b))
                                                for lab, a, b in PERIODS]
    for lab, a, b in spans:
        m = keep & (close.index >= a) & (close.index <= b)
        if m.sum() < 2 * BLOCK:
            continue
        cell = cell_all[m]
        fwd = {h: fwd_all[h][m] for h in H}
        res["full"][lab] = _cells_stats(cell, fwd, rng)
        res["full"][lab]["span"] = (close.index[m][0], close.index[m][-1])
        off = np.where(np.nan_to_num(ma[m], nan=1.0) < 0.5, cell, -1)
        res["off"][lab] = _cells_stats(off, fwd, rng)
        res["off"][lab]["span"] = res["full"][lab]["span"]
    full = res["full"]["全期"]
    subs = [lab for lab, _, _ in PERIODS if lab in res["full"]]
    top_full = bool(full["top"])
    top_subs = sum(int(res["full"][lab]["top"]) for lab in subs)
    lo, hi = full["inter_ci"]
    excl = _ok(lo) and _ok(hi) and (lo > 0 or hi < 0)
    res["pass"] = {"top_full": top_full, "top_subs": top_subs, "n_subs": len(subs), "excl": excl,
                   "pass": top_full and top_subs >= 2 and excl}
    return res


# ---------- the fair exam ----------

class _Exam:
    """main_signal_check's four checks for ma50 and 'ma50 OR (flag and above20)' with everything that does not depend
    on the flag precomputed once (the placebo rebuilds only the flag)."""

    def __init__(self, close: pd.Series, other: dict, price, b0, pre):
        self.close = close
        self.ma = msc._targets(close)["ma50"].values.astype(float)
        self.a20 = _above20(close.values)
        self.change = close.pct_change().fillna(0.0).values
        self.starts, self.labels = msc._month_starts(close.index, *GRID)
        hr, hd = msc._suffix(msc._equity(self.change), self.starts, YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma, self.ca20 = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(np.nan_to_num(msc._targets(pd.Series(c))["ma50"].values[npre:][:self.days].astype(float)))
            self.ca20.append(_above20(c)[npre:][:self.days])
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, (c, since, year) in other.items():
            tg = msc._targets(c, year)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, pd.Timestamp(since or c.index[0]),
                        pd.Timestamp(GRID[0]))
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "a20": _above20(c.values),
                               "change": change, "starts": starts, "labels": labels, "year": year,
                               "lag": 1 if since else 0, "hold": (hr[:, 0], hd[:, 0])}

    @staticmethod
    def _at(flag: pd.Series | None, dates, lag: int):
        if flag is None:
            return None
        return flag.reindex(pd.DatetimeIndex(dates) - lag * DAY).fillna(0.0).values.astype(float)

    def btc_pos(self, flag: pd.Series | None) -> np.ndarray:
        return _pos(self.ma, self.a20, self._at(flag, self.close.index, 0))

    def summary(self, flag: pd.Series | None, ki: int, sims: int = SIMS) -> dict:
        pos = self.btc_pos(flag)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.change, COST)
        r, dd = msc._suffix(msc._equity(net), self.starts, YEAR)
        dr, ddd = r[:, 0], dd[:, 0]
        f = self._at(flag, self.m.dates, 1)
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = _pos(self.cma[h], self.ca20[h], f)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pp = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pp, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        each = {}
        for key, o in self.other.items():
            w = _pos(o["ma"], o["a20"], self._at(flag, o["close"].index, o["lag"]))
            net, _ = te._run(np.concatenate([[np.nan], w[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((dd[:, 0] > o["hold"][1]).sum()))
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_edge": float(np.median(dr - self.dhold[0])),
             "d_shallow": int((ddd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= LOSS_BAR
        return x


def _score_ge(x: dict, base: dict) -> bool:
    if base["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not base["eligible"]:
        return True
    return x["low"] >= base["low"] - 1e-12


def _roll(flag: pd.Series, start: pd.Timestamp, k: int) -> pd.Series:
    out = flag.copy()
    m = out.index >= start
    out[m] = np.roll(out[m].values, k)
    return out


# ---------- report-only pieces ----------

def _extra(close: pd.Series, ma: np.ndarray, s1: np.ndarray) -> dict:
    idx = close.index
    ch = close.pct_change().values
    nxt = np.concatenate([ch[1:], [np.nan]])
    a = (idx >= pd.Timestamp(GRID[0]))
    ex = a & (s1 > 0.5) & (np.nan_to_num(ma) < 0.5)
    years = {}
    for y in sorted(set(idx[ex].year)):
        m = ex & (idx.year == y)
        years[int(y)] = (int(m.sum()), float(np.nanmean(nxt[m])))
    ok = ex & ~np.isnan(nxt)
    allm = a & ~np.isnan(nxt)
    return {"n": int(ex.sum()), "years": years, "avg": float(nxt[ok].mean()) if ok.any() else None,
            "avg_all": float(nxt[allm].mean()), "avg_ma_on": float(nxt[allm & (np.nan_to_num(ma) > 0.5)].mean())}


def _years(close: pd.Series, pos: np.ndarray) -> dict:
    change = close.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    s = pd.Series(net, index=close.index)
    s = s[s.index >= pd.Timestamp("2018-01-01")]
    return {int(y): float(np.prod(1 + g.values) - 1) for y, g in s.groupby(s.index.year)}


def _entries(close: pd.Series, ma: np.ndarray, s1: np.ndarray) -> list[dict]:
    c = close.values
    idx = close.index
    ma = np.nan_to_num(ma)
    n = len(c)
    first = int(idx.searchsorted(pd.Timestamp(GRID[0])))
    out = []
    for t in range(max(first, 1), n):
        if not (s1[t] > 0.5 and s1[t - 1] < 0.5 and ma[t] < 0.5):
            continue
        u = next((k for k in range(t + 1, n) if ma[k] > 0.5), None)
        right = u is not None and u - t <= 30 and c[u] > c[t]
        e = next((k for k in range(t + 1, n) if s1[k] < 0.5 or ma[k] > 0.5), None)
        if e is None:
            ret, done = c[-1] / c[t] - 1 - COST, False
        else:
            exit_trade = s1[e] < 0.5
            ret, done = c[e] / c[t] - 1 - COST * (2 if exit_trade else 1), True
        out.append({"day": idx[t], "right": bool(right), "ret": float(ret), "done": done,
                    "known": u is not None and (u - t <= 30 or n - 1 - t > 30)})
    return out


def _leadlag(x_daily: pd.Series, close: pd.Series, kind: str, rng) -> dict:
    """kind 'mean' (premium) or 'sum' (stablecoin log growth): the leg over 30-day windows vs BTC log returns."""
    x = x_daily.dropna()
    xp = x.rolling(30, min_periods=25).mean() if kind == "mean" else x.rolling(30, min_periods=30).sum()
    lc = np.log(close)
    rpast = lc - lc.shift(30)
    rnext = lc.shift(-30) - lc
    df = pd.DataFrame({"xp": xp, "xn": xp.shift(-30), "rp": rpast, "rn": rnext}).dropna()
    if len(df) < 400:
        return {}
    xpv, xnv, rpv, rnv = (df[c].values for c in ("xp", "xn", "rp", "rn"))

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    lead, follow, same = corr(xpv, rnv), corr(rpv, xnv), corr(xpv, rpv)
    n = len(df)
    bl, bf = [], []
    for k in rng.integers(GAP, n - GAP, LL_DRAWS):
        bl.append(corr(np.roll(xpv, k), rnv))
        bf.append(corr(rpv, np.roll(xnv, k)))
    return {"lead": lead, "follow": follow, "same": same, "n": n, "from": df.index[0], "to": df.index[-1],
            "lead_band": (float(np.percentile(bl, 5)), float(np.percentile(bl, 95))),
            "follow_band": (float(np.percentile(bf, 5)), float(np.percentile(bf, 95)))}


def _lookahead(data: dict, lg: pd.DataFrame, s1: np.ndarray) -> tuple[int, int]:
    total = bad = 0
    close = data["close"]
    for cut in CUTS:
        c = pd.Timestamp(cut)
        cm = data["cm"][data["cm"].index <= c - DAY]
        sup = data["sup"][data["sup"].index <= c - DAY]
        prem = data["prem"][data["prem"].index <= c]
        cl = close[close.index <= c]
        lg2 = _legs(_raw_legs(cm, sup, prem), cl.index)
        cols = ["t_flow", "t_mvrv", "d_stable", "d_prem", "joint"]
        total += 1
        bad += int(not np.array_equal(lg2[cols].values, lg.loc[cl.index, cols].values))
        ma2 = msc._targets(cl)["ma50"].values.astype(float)
        p2 = _pos(ma2, _above20(cl.values), lg2["joint"].values)
        total += 1
        bad += int(not np.array_equal(p2, s1[:len(cl)]))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    raw = _raw_legs(data["cm"], data["sup"], data["prem"])
    lg = _legs(raw, close.index)
    joint = lg["joint"]
    ok = lg["t_flow_def"] & lg["t_mvrv_def"] & lg["d_prem_def"]
    start = close.index[int(np.argmax(ok.values))]
    ma = msc._targets(close)["ma50"].values.astype(float)
    a = _test_a(close, lg, ma, start)

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(close, data["other"], price, b0, pre)
    flags = {"ma50": None, "s1": joint, "t": lg["tight"], "d": lg["demand"]}
    s = {k: ex.summary(flags[k], KI[k]) for k in flags}
    t_exam = time.time() - t0

    real_small = ex.summary(joint, KI["s1"], PL_SIMS)
    prng = np.random.default_rng(SEED + 9)
    n = int((joint.index >= start).sum())
    hits, lows = 0, []
    for k in prng.integers(GAP, n - GAP, DRAWS):
        x = ex.summary(_roll(joint, start, int(k)), KI["s1"], PL_SIMS)
        lows.append(x["low"])
        hits += int(_score_ge(x, real_small))
    placebo = hits / DRAWS

    pos = {k: ex.btc_pos(flags[k]) for k in flags}
    extra = _extra(close, ma, pos["s1"])
    years = {k: _years(close, pos[k]) for k in ("ma50", "s1")}
    hold_years = _years(close, np.ones(len(close)))
    entries = _entries(close, ma, pos["s1"])
    differ = {}
    m = close.index >= pd.Timestamp(GRID[0])
    for k in ("s1", "t", "d"):
        differ[k] = int((pos[k][m] != np.nan_to_num(pos["ma50"][m])).sum())
    lrng = np.random.default_rng(SEED + 77)
    ll = {"d_prem": _leadlag(data["prem"], close, "mean", lrng),
          "d_stable": _leadlag(_stable_dl(data["sup"]), close, "sum", lrng)}
    look = _lookahead(data, lg, pos["s1"])
    return {"raw": raw, "lg": lg, "start": start, "a": a, "ex": ex, "s": s, "real_small": real_small,
            "placebo": placebo, "placebo_med": float(np.median(lows)), "pos": pos, "extra": extra, "years": years,
            "hold_years": hold_years, "entries": entries, "differ": differ, "ll": ll, "look": look,
            "problem": problem, "secs": time.time() - t0, "secs_exam": t_exam, "ma": ma}


# ---------- today ----------

def _now(data: dict, r: dict) -> list[dict]:
    close, lg, raw = data["close"], r["lg"], r["raw"]
    last = close.index[-1]
    t = lg.iloc[-1]
    word = {1: "持有", 0: "現金"}
    ma_now, s1_now = int(r["pos"]["ma50"][-1] > 0.5), int(r["pos"]["s1"][-1] > 0.5)
    on = lambda v: "是" if v > 0.5 else "否"  # noqa: E731
    out = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜S1", "value": f"{word[ma_now]}｜{word[s1_now]}"},
           {"label": "賣壓少＋買盤來（兩個都成立）", "value": on(t["joint"])},
           {"label": "賣壓少（兩條都要成立）", "value": on(t["tight"])},
           {"label": "買盤來（任一條成立）", "value": on(t["demand"])}]
    for k in ("t_flow", "t_mvrv", "d_stable", "d_prem"):
        f, v, thr = raw[k]
        day = last - LEG_LAG[k] * DAY
        vv = v.get(day) if day in v.index else None
        tt = thr.get(day) if (thr is not None and day in thr.index) else None
        if k == "t_flow":
            detail = f"30 天 {vv:,.0f} 顆，中位數 {tt:,.0f} 顆" if _ok(vv) and _ok(tt) else "資料不足"
        elif k == "t_mvrv":
            detail = f"MVRV {vv:.2f}，中位數 {tt:.2f}" if _ok(vv) and _ok(tt) else "資料不足"
        elif k == "d_stable":
            detail = (f"30 天 {(math.exp(vv) - 1) * 100:+.2f}%，中位數 {(math.exp(tt) - 1) * 100:+.2f}%"
                      if _ok(vv) and _ok(tt) else "資料不足")
        else:
            detail = f"7 日平均 {vv * 100:+.3f}%" if _ok(vv) else "資料不足"
        out.append({"label": f"　{LEG_NAME[k]}", "value": f"{on(t[k])}（用 {_d(day)} 的資料：{detail}）"})
    if s1_now and not ma_now:
        a20 = _above20(close.values)[-1]
        out.append({"label": "注意", "value": f"S1 現在因為「賣壓少＋買盤來」而提早持有（站上 20 日均線：{on(a20)}）"})
    out.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；S1 沒有通過考試前不會用"})
    return out


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, a, ex = data["close"], r["s"], r["a"], r["ex"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    raw, lg = r["raw"], r["lg"]

    # ----- table 0: data dates -----
    rows0 = []
    for k in ("t_flow", "t_mvrv", "d_stable", "d_prem"):
        f, v, _ = raw[k]
        v = v.dropna()
        fd = f.dropna()
        first_use = lg.index[lg[k + "_def"].values]
        rows0.append([LEG_NAME[k], _d(v.index[0]) if len(v) else "—", _d(fd.index[0]) if len(fd) else "—",
                      _d(first_use[0]) if len(first_use) else "—",
                      "隔天收盤才用（晚一天）" if LEG_LAG[k] else "當天收盤就用", _d(fd.index[-1]) if len(fd) else "—",
                      _share(float(lg.loc[lg.index >= r["start"], k].mean()))])
    jm = lg.index >= r["start"]
    rows0.append(["賣壓少＋買盤來（兩個都成立）", "—", "—", _d(r["start"]), "—", _d(lg.index[-1]),
                  _share(float(lg.loc[jm, "joint"].mean()))])

    # ----- Test A tables -----
    def a_rows(part: dict) -> list:
        out = []
        for lab, x in part.items():
            sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
            for k in range(4):
                c30, c60 = x[(30, k)], x[(60, k)]
                out.append([sp if k == 0 else "", CELLS[k], f"{x['n'][k]:,}",
                            f"{_p(c30['mean'])} {_ci(*c30['mean_ci'])}", f"{_p(c30['med'])} {_ci(*c30['med_ci'])}",
                            f"{_p(c60['mean'])} {_ci(*c60['mean_ci'])}", f"{_p(c60['med'])} {_ci(*c60['med_ci'])}"])
        return out

    cols_a = ["期間", "格子", "天數", "之後 30 天平均 ［90% 區間］", "之後 30 天中位數 ［90% 區間］",
              "之後 60 天平均 ［90% 區間］", "之後 60 天中位數 ［90% 區間］"]
    rows_i = []
    for part, lab_p in ((a["full"], "全部日子"), (a["off"], "只看 50 日均線空手的日子")):
        for lab, x in part.items():
            means = [x[(30, k)]["mean"] for k in range(4)]
            best = CELLS[int(np.nanargmax([v if _ok(v) else -np.inf for v in means]))] if any(_ok(v) for v in means) else "—"
            rows_i.append([lab_p, lab, f"{_pt(x['inter'])} {_ci(*x['inter_ci'], f=_pt)}", best])
    ap = a["pass"]

    # ----- Test B -----
    base = s["ma50"]
    x1 = s["s1"]
    score_ok = _score_ge(x1, base)
    loss_ok = x1["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK
    plc_ok = r["placebo"] <= 0.10
    b_pass = score_ok and loss_ok and plc_ok

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    rows_b = []
    for k in ("ma50", "s1", "t", "d"):
        x = s[k]
        ck = x["checks"]
        rows_b.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                       "—" if k == "ma50" else f"{r['differ'][k]:,} 天",
                       _share(r["placebo"]) if k == "s1" else "—"])
    rows_pass = [["排名分數不輸 50 日均線", f"S1 {_share1(x1['low'])}，均線 {_share1(base['low'])}", yes(score_ok)],
                 ["最深虧損中位數最多比均線深 5 點", f"S1 {_pc(x1['c_deep_med'])}，均線 {_pc(base['c_deep_med'])}", yes(loss_ok)],
                 ["隨機錯位（安慰劑）比例 ≤ 10%", f"{_share(r['placebo'])}（錯位的排名分數中位數 {_share(r['placebo_med'])}；"
                  f"真的 S1 同樣用每個收盤時間 {PL_SIMS} 次模擬算是 {_share(r['real_small']['low'])}）", yes(plc_ok)],
                 ["三條都要過", "", yes(b_pass)]]

    # ----- report only -----
    e = r["extra"]
    rows_x = [[str(y), f"{n:,}", _p(v)] for y, (n, v) in e["years"].items()]
    rows_x.append(["合計", f"{e['n']:,}", _p(e["avg"])])
    yrs = r["years"]
    rows_y = []
    for y in sorted(yrs["ma50"]):
        lab = f"{y}" + ("（到今天）" if y == close.index[-1].year else "")
        rows_y.append([lab, _p(r["hold_years"].get(y)), _p(yrs["ma50"][y]), _p(yrs["s1"].get(y)),
                       _pt(yrs["s1"].get(y, 0) - yrs["ma50"][y])])
    ent = r["entries"]
    right = [x for x in ent if x["right"]]
    wrong = [x for x in ent if not x["right"]]
    avg = lambda v: float(np.mean([x["ret"] for x in v])) if v else None  # noqa: E731
    rows_e = [["對（30 天內均線也上車，而且價格更高）", f"{len(right)}", _p(avg(right))],
              ["錯（其他情況）", f"{len(wrong)}", _p(avg(wrong))],
              ["合計", f"{len(ent)}", _p(avg(ent))]]
    rows_e_list = [[_d(x["day"]), "對" if x["right"] else "錯", _p(x["ret"]) + ("" if x["done"] else "（還沒結束）")]
                   for x in ent]
    rows_l = []
    for k, lab in (("d_prem", "Coinbase 溢價（30 天平均）"), ("d_stable", "穩定幣總量（30 天成長）")):
        q = r["ll"].get(k) or {}
        if not q:
            rows_l.append([lab, "資料不足", "—", "—", "—"])
            continue
        rows_l.append([lab, f"{_d(q['from'])} 到 {_d(q['to'])}（{q['n']:,} 天）",
                       f"{q['lead']:+.2f} ［雜訊 {q['lead_band'][0]:+.2f} 到 {q['lead_band'][1]:+.2f}］",
                       f"{q['follow']:+.2f} ［雜訊 {q['follow_band'][0]:+.2f} 到 {q['follow_band'][1]:+.2f}］",
                       f"{q['same']:+.2f}"])

    full = a["full"]["全期"]
    tables = [
        {"caption": ("每條資料從哪天開始。Coin Metrics 和 DefiLlama 某一天的數字，要等到隔一天的日線收盤才拿來用（晚一天，保守）；"
                     "Coinbase 溢價用兩家交易所當天 UTC 0 點收盤價算，收盤當下就知道。還沒有資料的那條算「不成立」。"
                     "最後一欄是從測試 A 開始那天起，這條成立的日子占幾成。"),
         "columns": ["條件", "原始資料第一天", "條件第一次算得出來", "第一次能拿來決定（收盤日）", "用法", "最後一天", "成立的日子"],
         "rows": rows0},
        {"caption": (f"測試 A：每天依「賣壓少」「買盤來」分成四格，看之後 30、60 天比特幣漲跌。從 {_d(r['start'])} 起（兩條賣壓條件和"
                     "溢價都有資料的第一天；更早的日子買盤一定算「沒有」，放進來會失真），所以「2015–19」實際上從那天開始。"
                     f"90% 區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次（日子彼此相關，所以整塊抽）。"),
         "columns": cols_a, "rows": a_rows(a["full"])},
        {"caption": ("交互作用＝（賣壓少＋買盤來 − 只有賣壓少）−（只有買盤來 − 兩個都沒有），用之後 30 天的平均漲跌算。"
                     "正的表示兩個一起出現時，比各自效果加起來還多。最後一欄是四格裡之後 30 天平均最高的那格。"
                     "下半部只看 50 日均線空手的日子（提早上車真正有用的日子，只描述，不算在通過條件裡）。"),
         "columns": ["日子", "期間", "交互作用（點）［90% 區間］", "30 天平均最高的格子"], "rows": rows_i},
        {"caption": "測試 A 只看 50 日均線空手的日子：四格之後 30、60 天的漲跌（只描述）。",
         "columns": cols_a, "rows": a_rows(a["off"])},
        {"caption": (f"測試 B：公平考試（main_signal_check 的做法）。比特幣日線起算月 {GRID[0][:7]} 到 {GRID[1][:7]}"
                     f"（{x1['d_n']} 個，S1 最早能和均線不同的月份），起算月×24 個收盤時間 {x1['c_n']:,} 格，一般人做法"
                     f"（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），其他資產 "
                     f"{'、'.join(ex.other)}（各用自己的 50、20 日均線，配比特幣的「賣壓少＋買盤來」當全市場訊號）。50 日均線用完全相同的起算日重算。"
                     "排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。「和均線不同的天數」是 2017-09 起比特幣日線上持有／現金不一樣的天數。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "和均線不同的天數", "隨機錯位比例"],
         "rows": rows_b},
        {"caption": (f"測試 B 的通過條件（事先定好）。隨機錯位：把「賣壓少＋買盤來」整條訊號往後挪 {GAP} 天以上（頭尾相接），"
                     f"{DRAWS} 次，整個考試重算（一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的 S1 也用同樣方法重算來比）。"),
         "columns": ["條件", "數字", "結果"], "rows": rows_pass},
        {"caption": (f"只描述：S1 比 50 日均線多持有的日子（{GRID[0]} 起，比特幣日線），以及那些日子隔天的平均漲跌。對照：同期間每天平均 "
                     f"{_p(e['avg_all'])}，均線持有的日子平均 {_p(e['avg_ma_on'])}。"),
         "columns": ["年", "多持有的天數", "隔天平均漲跌"], "rows": rows_x},
        {"caption": f"只描述：每年報酬（扣 {COST * 100:.2f}% 成本），2018 起。",
         "columns": ["年", "一直抱著", "50 日均線", "S1", "S1 減均線"], "rows": rows_y},
        {"caption": ("只描述：S1 提早上車（均線還在空手時 S1 轉成持有）對不對。「對」＝30 天內均線也上車，而且那時價格比提早上車時高。"
                     "每次的結果＝從提早上車到 S1 下車或均線上車（先到的那個）之間的漲跌，扣掉買賣成本。"),
         "columns": ["結果", "次數", "平均漲跌"], "rows": rows_e},
        {"caption": "只描述：每一次提早上車。", "columns": ["日期", "對／錯", "漲跌"], "rows": rows_e_list or [["—", "—", "—"]]},
        {"caption": ("只描述：誰先動？「領先」＝過去 30 天的買盤數字和之後 30 天比特幣漲跌的相關；「跟隨」＝過去 30 天比特幣漲跌和之後 "
                     f"30 天買盤數字的相關。雜訊帶＝把買盤數字錯開 {GAP} 天以上、{LL_DRAWS} 次得到的 90% 範圍；落在帶子外才算看得出來。"),
         "columns": ["買盤數字", "期間", "領先（買盤→價格）", "跟隨（價格→買盤）", "同一段 30 天的相關"], "rows": rows_l},
    ]

    # ----- verdict -----
    td_full = full[(30, 0)]["mean"]
    others_full = "、".join(f"{CELLS[k]} {_p(full[(30, k)]['mean'])}" for k in (1, 2, 3))
    first = (f"測試 A（{_d(r['start'])} 起）：「賣壓少＋買盤來」的日子之後 30 天平均 {_p(td_full)}（{full['n'][0]:,} 天），"
             f"{others_full}；交互作用 {_pt(full['inter'])} {_ci(*full['inter_ci'], f=_pt)}；"
             f"三段裡有 {ap['top_subs']} 段是這格最高。事先定的條件{'通過' if ap['pass'] else '沒有通過'}"
             + ("" if ap["pass"] else "（" + "、".join(
                 ([] if ap["top_full"] else ["全期不是最高"]) + ([] if ap["top_subs"] >= 2 else ["三段裡最高的不到兩段"])
                 + ([] if ap["excl"] else ["交互作用的區間含零"])) + "）") + "。")
    fails = ([] if score_ok else ["排名分數輸均線"]) + ([] if loss_ok else ["最深虧損比均線深超過 5 點"]) \
        + ([] if plc_ok else ["隨機錯位超過 10%"])
    second = (f"測試 B：S1 排名分數 {_share1(x1['low'])}（均線 {_share1(base['low'])}），最深虧損中位數 {_pc(x1['c_deep_med'])}"
              f"（均線 {_pc(base['c_deep_med'])}），隨機錯位 {_share(r['placebo'])}，"
              + ("三條都通過，只能算是可以放進每日影子紀錄觀察的候選。" if b_pass else f"沒有通過（{'、'.join(fails)}）。"))
    third = (f"S1 比均線多持有 {e['n']:,} 天，隔天平均 {_p(e['avg'])}；提早上車 {len(ent)} 次，對 {len(right)} 次、錯 {len(wrong)} 次"
             f"（錯的平均 {_p(avg(wrong))}）。單用賣壓少 {_share(s['t']['low'])}、單用買盤來 {_share(s['d']['low'])}。")
    verdict = first + second + third

    lk_total, lk_bad = r["look"]
    notes = [
        "門檻（中位數、30／365／730 天、20 日均線、60 天的區塊）都是跑之前憑道理定的，沒有試過其他數字；這是一次就定的考試。",
        "Coin Metrics 的 MVRV 和交易所流入會被事後修正，交易所地址的辨認也越來越完整；這裡用的是今天下載的版本，"
        "當時的人看到的數字可能不一樣（flow_filters 發現它在 2026 年 4 月把整段歷史重算過）。所以歷史結果可能比當時真的做得到的好看。",
        f"穩定幣資料 {STABLE_FROM} 以前不完整（缺 Omni 上的 USDT），所以那以前穩定幣這條一律算不成立；單日變動超過 ±15% 當成"
        "資料涵蓋範圍改變、不算成長（和 stablecoin_supply 一樣）。",
        f"Coinbase 溢價從 {FIRST} 才有（Bitstamp、Coinbase 日線都是 UTC 0 點收盤），所以在那以前「買盤來」一定不成立、S1 就等於 50 日均線；"
        "公平考試的日線起算月因此從 2017-09 開始，均線也在同樣的起算月重算。",
        "其他幣和 MSTR 沒有自己的賣壓、買盤資料，用的是比特幣的訊號（當成全市場的狀態）；MSTR 用前一天的訊號（美股收盤在 UTC 0 點之前）。",
        "和 rally_timing 不同：那邊只用價格（例如大跌後站上 20 日均線）在空頭低點附近試著早上車；這裡要求賣壓和買盤同時成立，而且不限空頭低點。",
        f"偷看未來的檢查：把資料在 {'、'.join(CUTS)} 截斷（Coin Metrics、DefiLlama 再早一天）重算，{lk_total} 次比對裡有 {lk_bad} 次不一致"
        f"{'（應該是 0）' if lk_bad else '，截斷前的訊號和持有完全一樣'}。",
        "就算通過，也只代表 S1 可以放進每日影子紀錄觀察；用的是同一段歷史，真正的考驗是之後每天實際記下的結果。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘（不含下載）。",
    ]
    notes += problems

    return {
        "id": "supply_demand",
        "title": "賣壓少＋買盤來：能不能比 50 日均線早上車",
        "question": ("願意在現價賣的人少（賣壓少），同時買盤開始進來（買盤來），兩個一起出現時比特幣會不會漲特別多？"
                     "用這個在 50 日均線還沒上車前提早持有，能不能多賺到均線錯過的上漲？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群 API：比特幣 FlowInExNtv（轉進交易所）、CapMVRVCur（MVRV）",
             "url": CM_URL + "?assets=btc&metrics=FlowInExNtv,CapMVRVCur&frequency=1d"},
            {"name": "DefiLlama 穩定幣總量", "url": LLAMA},
            {"name": "Coinbase Exchange 公開日線 BTC-USD", "url": CB_URL + "?granularity=86400"},
            {"name": "Bitstamp 公開 API：BTC/USD 日線", "url": BS_URL + "?step=86400&limit=1000"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
