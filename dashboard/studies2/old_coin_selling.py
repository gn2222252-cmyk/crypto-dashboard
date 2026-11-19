"""Historical study "old_coin_selling": when coins that had not moved for a long time are spent (long-held coins
leaving their holders), is bitcoin's next one to three months worse - read together with what price is doing and with
whether buyers absorbed those coins? Can it help the main signal ma50 (hold while the daily close is above its 50-day
average)?

Why this study exists (stated plainly). The project has been trying to measure SELLING PRESSURE. Its first measure,
"BTC sent to exchanges", turned out to track how busy the market is: heavy exchange inflows were followed by the BEST
returns (reluctant_sellers, Test D), the opposite of a selling-pressure reading. A scout found a more direct free
measure: coins that had not moved for 155+ days being spent. This is the fifth study of the supply/demand idea
(supply_demand, price_impact, reluctant_sellers A-C, reluctant_sellers D, this one), so any single pass is weak
evidence. The owner's standing point is built in: a selling measure is never read alone - it is read together with
price (Test A's price side) and with whether buyers absorbed it (section 3).

Data (plain HTTP GET of the CSV data files only, fixed user agent "btc-research-dashboard"; the repository is never
cloned and none of its code is downloaded, imported or run; each CSV is treated as untrusted data and parsed by pandas
into dates and numbers only, columns and ranges validated, and the parsed copy stored in
cache/_store/old_coin_selling_obm.json, used only when a later download fails):
  Open Bitcoin Metrics (OBM, CC BY 4.0), daily metrics built from a full node, history from 2009:
    https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/<id>/<id>.csv for
    obm_spent_value_ge155d_btc_daily (BTC spent that had been unmoved for >= 155 days), obm_spent_value_ge365d_btc_daily
    (>= 365 days), obm_spent_value_lt155d_btc_daily (< 155 days), obm_spent_value_age_band_btc_daily (spent value by
    age band), obm_cdd_btcxdays_daily (coin days destroyed), obm_dormancy_days_daily (dormancy). Definitions are cited
    from the repository README and the paper's abstract (arXiv:2607.03124).
  BTC daily closes: the dashboard's (te._daily_close, 00:00 UTC; blockchain.info before Binance 2017-08).
  Buy side (section 3 only): total stablecoin supply (DefiLlama, stablecoin_supply._load) and US spot bitcoin ETF net
    flow (Farside, etf_loop._farside, from 2024-01-11; USD millions x 1e6), as rally_sources loads them.
  Fair exam inputs as reluctant_sellers: Binance 1-hour BTCUSDT grid, ETH SOL BNB XRP (Binance) and MSTR (Yahoo)
    daily closes through main_signal_check's loaders.

0. Data validation (reported first). Per metric: first and last date, missing calendar days, non-numeric or negative
  values (dropped and counted). Publication lag = today minus the last row date. LAG USED, fixed now: a value for data
  day D is used from the BTC daily close of D+7 (the scout saw about 5 days; 7 leaves margin), i.e. decision day t
  uses data day t-7. If the observed lag ever exceeds 7 days the latest decision days simply have no value (stated).
  Sanity checks: (a) spent >=155d + spent <155d versus the sum of the age bands (same source, should agree) and versus
  Coin Metrics community TxTfrValAdjNtv (adjusted on-chain transfer volume, BTC; order of magnitude only, report);
  (b) the 10 largest one-day ratios of spent >=155d to its trailing-365-day median, with dates - known custody moves
  that were NOT sales should appear (e.g. the Mt. Gox trustee wallet moves of 2024-05-28); (c) CDD cross-check against
  Blockchair's daily block dumps (cdd_total per block, summed per UTC day) on 20 days drawn with a fixed seed from
  2012-01-01 .. 60 days before today; one polite GET per day with a pause, stop at the first error or non-gzip reply
  and report how many were checked; the comparison is done once and stored in cache/_store/old_coin_selling_cdd.json
  (later runs reuse it, they do not download again). Agreement = median and worst absolute relative difference.

1. Measure (fixed now; on OBM data dates D).
  S(D) = spent >= 155d BTC on day D. P99(D) = 99th percentile of S over D-365 .. D-1 (>= 330 values).
  W(D) = min(S(D), P99(D)) (winsorised; undefined while P99 is undefined).
  RAW30(D) = sum of S over D-29 .. D; OLDSELL(D) = sum of W over D-29 .. D (all 30 values required).
  RANK(D) = share of the 365 previous OLDSELL values D-365 .. D-1 (>= 330 non-missing) that are below OLDSELL(D).
  HEAVY = RANK > 0.80; QUIET = RANK < 0.50; MIDDLE otherwise. Decision day t uses RANK(t-7); undefined -> no cell
  (and OFF in rules). Robustness line (report only): the same built from spent >= 365d.
  above20(t) = close(t) above the mean of the last 20 closes including t: UP; otherwise DOWN. ma50 =
  main_signal_check's.

2. Test A - does old-coin selling matter, read with price? Sample: decision days from 2012-01-01 with a cell and a full
  60-day forward window (the 90-day column skips days without one). Six cells UP/DOWN x QUIET/MIDDLE/HEAVY: days and
  share; mean and median of close[t+h]/close[t]-1 for h = 30, 60, 90; mean deepest drop within 60 days
  min(close[t+1..t+60])/close[t]-1 capped at 0; 90% intervals from a circular block bootstrap of sample days (60-day
  blocks, 1,000 draws, fixed seed) inside each period: full, 2012-16, 2017-21, 2022-now.
  Hypothesis fixed now (direction by logic): HEAVY is followed by LOWER returns than QUIET - on the UP side
  "distribution into strength", on the DOWN side "long holders giving up".
  D_up, D_down = QUIET mean minus HEAVY mean, 60-day forward return, within UP and within DOWN; pooled = (D_up +
  D_down) / 2 with its bootstrap interval.
  Matched control: within each side, deciles of the trailing 30-day return close[t]/close[t-30]-1 (edges from that
  side's sample days); QUIET-minus-HEAVY 60-day difference inside each decile weighted by QUIET days (deciles holding
  both kinds only; price_impact._matched); matched pooled = average of the two sides.
  Pre-registered pass, all four: D_up > 0 and D_down > 0 (full period); full-period pooled 90% interval entirely above
  zero; pooled > 0 in at least two of the three sub-periods; matched pooled > 0.
  The 1y+ robustness line repeats D_up, D_down and pooled (full period, 60 days) with spent >= 365d (report only).

3. Absorption (the owner's two forces; report only, fixed now). Among HEAVY sample days: ABSORBED = close(t) above
  close(t-30) (buyers took the old coins and price still rose over the same 30 days), NOT ABSORBED = otherwise.
  Forward 30/60/90-day mean and median with bootstrap intervals for HEAVY+ABSORBED, HEAVY+NOT ABSORBED and QUIET
  (all), full period. Level view: OLD_USD = sum over the same 30 data days D-29 .. D (D = t-7) of W(d) x close(d);
  BUY_USD = total stablecoin supply(D) minus supply(D-30) plus the sum of US spot ETF net flow over D-29 .. D (0 before
  the launch); RATIO = BUY_USD / OLD_USD. Sample = decision days whose 30-day buy window lies inside the stablecoin
  data (from 2019-07-01); terciles of RATIO from the sample itself (descriptive, uses the whole sample to set the
  cut-offs); forward 30/60/90-day mean per tercile; also the 2024-ETF-era subsample alone. Few independent periods;
  description only.

4. Tops and bottoms (report only). Rallies from rally_sources._rallies (+80% / -30%, from 2013), described with
  rally_sources._describe. For each rally: mean decision-day RANK in the 60 days before the peak, the 31 days around it
  (peak +-15) and the 60 days after; the same around the trough. Peaks preceded by HEAVY = at least one HEAVY decision
  day in peak-60 .. peak (finished rallies). HEAVY episodes = a HEAVY decision day with no HEAVY day in the 30 days
  before it; false alarm = no finished-rally peak within 90 days after the episode start (unknown if fewer than 90
  days have passed). Base rate: share of all sample days followed by a peak within 90 days.

5. Rules (two, fixed). T1 = hold when ma50 is on AND NOT HEAVY (exit side; can only avoid, never add). T2 = hold when
  ma50 is on OR (ma50 was on at any of the 20 previous closes t-20 .. t-1 AND QUIET) (do not get shaken out when long
  holders are not selling). Other assets: own ma50 with BTC's HEAVY / QUIET as a market-wide flag.
  Fair exam = reluctant_sellers._Exam (main_signal_check's four checks): first month 2015-01 (BTC daily monthly starts
  2015-01 .. 2023-12, 0.10%); clock starts 2018-01 .. 2023-12 x 24 closing hours textbook; the same cells as an
  ordinary person (act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock); ETH SOL BNB XRP
  MSTR monthly starts to three years before today. ma50 is re-scored on exactly the same starts (counts reported).
  Ranking score = worst of the four "beats holding" shares (main_signal_check's eligibility). Median deepest loss =
  median over the clock textbook cells.
  Placebo: the decision-day RANK series, from its first defined day, circularly shifted by k in [180, n-180] days, 200
  draws (same k for both rules), HEAVY and QUIET rebuilt from the shifted rank; the whole exam rebuilt with 20
  ordinary-person simulations per clock (same seeds) and the real rule re-scored the same way; placebo share = share of
  draws whose ranking score is at least the real rule's.
  Pre-registered pass: T1 - ranking score >= ma50's, median deepest loss no worse than ma50's, placebo <= 5%.
  T2 - ranking score >= ma50's, median deepest loss no more than 5 points worse, placebo <= 5%.
  Two rules are tried in the fifth study of the same idea: one pass alone is weak evidence.

6. Report only. Lead/lag: x(t) = log(OLDSELL(t-7) / OLDSELL(t-21)) (14-day change known at t); correlation with BTC's
  14-day log return ending at t+k, k = -30 .. +30 (k > 0: old coins move first; k < 0: price first); noise band = 95th
  percentile of |corr| over 200 circular shifts of x by >= 180 days pooled over the shown k. Calendar-year returns
  (0.10% cost) of T1 and T2 minus ma50, every year from 2012.

Today: the rule states after the last daily close, the latest OLDSELL (winsorised and raw), its rank and cell, today's
Test A cell, and the date of the latest OBM data.

Honesty: coins moving are not coins sold - custody changes, exchange wallet migrations, ETF custodians reorganising,
lost-coin recoveries all count as old coins spent; there is no entity adjustment (moves between one owner's own
addresses count). The winsorising removes only the worst single days. The dataset is a single author's v0.x release
published in 2026; its values may change with the indexer definition or release version, so this is today's vintage
(on-chain history itself does not get revised). Thresholds (155 days, 30 / 365 days, 80th / 50th / 99th percentiles,
7-day lag, 20-day average and memory, 60-day blocks) were fixed by reasoning before the run; no alternatives tried.
A pass would only make a rule a candidate for the daily shadow log. Not advice.
"""
from __future__ import annotations

import gzip
import io
import json
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import etf_loop as el
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import rally_sources as rs
from dashboard.studies2 import reluctant_sellers as rsl
from dashboard.studies2 import stablecoin_supply as ss
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

COST, SIMS, SEED = msc.COST, msc.SIMS, msc.SEED
DAY = pd.Timedelta(days=1)
OBM_BASE = "https://raw.githubusercontent.com/diegorllanos/open-bitcoin-metrics/main/metrics/{0}/{0}.csv"
OBM_REPO = "https://github.com/diegorllanos/open-bitcoin-metrics"
OBM_PAPER = "https://arxiv.org/abs/2607.03124"
OBM_ZENODO = "https://doi.org/10.5281/zenodo.21156871"
BC_URL = "https://gz.blockchair.com/bitcoin/blocks/blockchair_bitcoin_blocks_{}.tsv.gz"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
IDS = {"ge155": "obm_spent_value_ge155d_btc_daily", "ge365": "obm_spent_value_ge365d_btc_daily",
       "lt155": "obm_spent_value_lt155d_btc_daily", "band": "obm_spent_value_age_band_btc_daily",
       "cdd": "obm_cdd_btcxdays_daily", "dorm": "obm_dormancy_days_daily"}
CN_ID = {"ge155": "放了 155 天以上的幣被花掉（BTC）", "ge365": "放了 1 年以上的幣被花掉（BTC）",
         "lt155": "放不到 155 天的幣被花掉（BTC）", "band": "各年齡層被花掉的幣（BTC，12 層加總）",
         "cdd": "幣天數銷毀 CDD（BTC×天）", "dorm": "平均沉睡天數（天）"}
# Range checks: a coin older than 155 days can be spent only once a day, so those series cannot exceed the supply;
# young coins can be re-spent many times a day (chains of transactions), so their bound is only a sanity bound.
MAXV = {"ge155": 2.2e7, "ge365": 2.2e7, "lt155": 1e9, "band": 1e9, "cdd": 1e12, "dorm": 1e5}
STORE_OBM = common.STORE / "old_coin_selling_obm.json"
STORE_CDD = common.STORE / "old_coin_selling_cdd.json"

LAG = 7
SUM_DAYS, WIN, WIN_MIN, P_WIN = 30, 365, 330, 0.99
P_HEAVY, P_QUIET = 0.80, 0.50
H = (30, 60, 90)
MAIN_H = 60
BLOCK = 60
SAMPLE_FROM = "2012-01-01"
FIRST_MONTH = msc.DAILY_STARTS[0]
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK, PLACEBO_BAR = 5.0, 0.05
KI = {"ma50": 1, "t1": 101, "t2": 102}
BEFORE, AROUND, AFTER = 60, 15, 60
EP_GAP, ALARM = 30, 90
N_CHECK = 20
LL_K = tuple(range(-30, 31))
LL_SHOW = (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30)
PERIODS = rsl.PERIODS
SIDES = ("價格在上", "價格在下")
LEVELS = ("老幣安靜", "老幣中間", "老幣大量移動")
NAME = {"ma50": "50 日均線（基準）", "t1": "T1：50 日均線，但老幣大量移動時不持有（只會少拿）",
        "t2": "T2：50 日均線，或「均線 20 天內還在＋老幣安靜」也持有（不被洗出場）"}
SHORT = {"ma50": "50 日均線", "t1": "T1", "t2": "T2"}

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- data ----------

def _parse(key: str, text: str) -> tuple[dict, dict]:
    """Untrusted CSV text -> {date: number}; only dates and numbers are kept, everything else is counted."""
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if "date" not in frame.columns:
        raise RuntimeError(f"{IDS[key]} 沒有 date 欄")
    if key == "band":
        cols = [c for c in frame.columns if c.startswith("spent_value_") and c.endswith("_btc")]
        if len(cols) < 10:
            raise RuntimeError("年齡層欄位不足")
        vals = pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in cols})
        value = vals.sum(axis=1, min_count=len(cols))
    else:
        if "value" not in frame.columns:
            raise RuntimeError(f"{IDS[key]} 沒有 value 欄")
        value = pd.to_numeric(frame["value"], errors="coerce")
    day = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    fin = value.notna() & np.isfinite(value.fillna(0.0))
    good = day.notna() & fin & (value >= 0) & (value <= MAXV[key])
    nan_rows = int((day.notna() & ~fin).sum())
    out = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(day[good], value[good])}
    if len(out) < 3000:
        raise RuntimeError(f"{IDS[key]} 只有 {len(out)} 筆可用")
    pick = lambda c: sorted(set(frame[c]))[:3] if c in frame.columns else []  # noqa: E731
    meta = {"rows": int(len(frame)), "kept": len(out), "nan": nan_rows,
            "bad": int(len(frame)) - len(out) - nan_rows, "unit": pick("unit"), "version": pick("release_version"),
            "series_id": pick("series_id"), "columns": int(len(frame.columns))}
    return out, meta


def _obm_fetch() -> dict:
    out = {"_meta": {}}
    for key, sid in IDS.items():
        reply = requests.get(OBM_BASE.format(sid), headers=common.HEADERS, timeout=90)
        reply.raise_for_status()
        if len(reply.content) > 20_000_000:
            raise RuntimeError(f"{sid} 檔案太大")
        out[key], out["_meta"][key] = _parse(key, reply.text)
        time.sleep(common.PAUSE)
    out["_meta"]["fetched_utc"] = common.now_utc()
    return out


def _series(raw: dict, key: str) -> pd.Series:
    s = pd.Series(raw[key], dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _cdd_check() -> dict:
    """Blockchair daily block dumps on 20 seeded days (done once; stored and reused, never re-downloaded)."""
    try:
        got = json.loads(STORE_CDD.read_text(encoding="utf-8"))
        if got.get("days") is not None:
            return got
    except Exception:  # noqa: BLE001 - not done yet
        pass
    last = pd.Timestamp(common.today_utc()) - pd.Timedelta(days=60)
    pool = pd.date_range(SAMPLE_FROM, last, freq="D")
    rng = np.random.default_rng(SEED + 55)
    days = sorted(pool[rng.choice(len(pool), N_CHECK, replace=False)])
    res, stop = {}, None
    for day in days:
        url = BC_URL.format(day.strftime("%Y%m%d"))
        try:
            reply = requests.get(url, headers=common.HEADERS, timeout=60)
            if reply.status_code != 200:
                stop = f"{_d(day)}：HTTP {reply.status_code}"
                break
            if reply.content[:2] != b"\x1f\x8b" or len(reply.content) > 5_000_000:
                stop = f"{_d(day)}：回傳的不是壓縮檔"
                break
            tab = pd.read_csv(io.StringIO(gzip.decompress(reply.content).decode("utf-8")), sep="\t", dtype=str)
            cdd = pd.to_numeric(tab["cdd_total"], errors="coerce")
            inp = pd.to_numeric(tab["input_total"], errors="coerce")
            res[_d(day)] = {"blocks": int(len(tab)), "cdd": float(cdd.sum()), "input_btc": float(inp.sum()) / 1e8,
                            "bad": int(cdd.isna().sum() + inp.isna().sum())}
        except Exception as exc:  # noqa: BLE001 - stop at any error, never retry or work around
            stop = f"{_d(day)}：{type(exc).__name__}"
            break
        time.sleep(3.0)
    out = {"days": res, "planned": [_d(x) for x in days], "stop": stop, "checked_utc": common.now_utc()}
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        STORE_CDD.write_text(json.dumps(out), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return out


def _buy_side() -> tuple[dict, list]:
    out, problems = {}, []
    try:
        sup, pr = ss._load()
        problems += pr
        out["stable"] = sup["all"].astype(float)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"穩定幣資料取不到（{type(exc).__name__}），買方那一段少了穩定幣。")
    try:
        fr, pr = el._farside("BTC")
        if pr:
            problems.append(pr)
        tot = fr["Total"].astype(float) * 1e6
        out["etf"] = tot[~tot.index.duplicated(keep="last")].sort_index()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ETF 流量取不到（{type(exc).__name__}），買方那一段少了 ETF。")
    return out, problems


def _load() -> dict:
    problems = []
    close = te._daily_close()
    raw, n = sd._stored(STORE_OBM, _obm_fetch)
    problems += [n] if n else []
    obm = {k: _series(raw, k) for k in IDS}
    other, more = msc._load_other()
    problems += more
    buy, more = _buy_side()
    problems += more
    return {"close": close, "obm": obm, "meta": raw.get("_meta", {}), "other": other, "buy": buy,
            "cdd": _cdd_check(), "problems": problems}


# ---------- the measure ----------

def _trail_rank(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        prev = v[max(0, i - WIN):i]
        prev = prev[np.isfinite(prev)]
        if len(prev) >= WIN_MIN:
            out[i] = float((prev < v[i]).mean())
    return out


def _measure(s: pd.Series) -> dict:
    """On a continuous daily data-date index: S, P99, W, RAW30, OLDSELL, RANK (and the trailing median)."""
    s = s.asfreq("D")
    p99 = s.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(P_WIN)
    w = s.where(s <= p99, p99).where(p99.notna() & s.notna())
    raw30 = s.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    old = w.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    rank = pd.Series(_trail_rank(old.values), index=old.index)
    med = s.shift(1).rolling(WIN, min_periods=WIN_MIN).median()
    return {"s": s, "p99": p99, "w": w, "raw30": raw30, "old": old, "rank": rank, "med": med}


def _on(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Data-date series -> value used at each decision close (data day t - LAG)."""
    return series.reindex(index - LAG * DAY).values.astype(float)


def _levels(rank: np.ndarray) -> np.ndarray:
    """0 QUIET, 1 MIDDLE, 2 HEAVY, -1 undefined."""
    return np.where(~np.isfinite(rank), -1, np.where(rank > P_HEAVY, 2, np.where(rank < P_QUIET, 0, 1)))


# ---------- statistics ----------

def _forward(close: pd.Series) -> dict:
    c = close.values.astype(float)
    n = len(c)
    out = {}
    for h in H:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        out[h] = f
    low = pd.Series(c[::-1]).rolling(MAIN_H, min_periods=MAIN_H).min().values[::-1]
    deep = np.full(n, np.nan)
    deep[:n - MAIN_H] = np.minimum(low[1:n - MAIN_H + 1] / c[:n - MAIN_H] - 1, 0.0)
    out["deep"] = deep
    return out


def _gstat(mask: np.ndarray, v: np.ndarray, bidx: np.ndarray, med: bool = True) -> tuple[dict, np.ndarray]:
    m = mask & np.isfinite(v)
    vb = v[bidx]
    mb = mask[bidx] & np.isfinite(vb)
    cnt = mb.sum(axis=1)
    bm = np.where(cnt > 0, np.where(mb, vb, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    out = {"n": int(m.sum()), "mean": float(v[m].mean()) if m.any() else None,
           "med": float(np.median(v[m])) if m.any() else None, "mean_ci": pi._q(bm), "med_ci": (None, None)}
    if med:
        with np.errstate(all="ignore"):
            out["med_ci"] = pi._q(np.nanmedian(np.where(mb, vb, np.nan), axis=1))
    return out, bm


def _test_a(close: pd.Series, rank: np.ndarray, a20: np.ndarray, seed: int, light: bool = False) -> dict:
    """Six cells: cell = 3 * side (0 UP, 1 DOWN) + level (0 QUIET, 1 MIDDLE, 2 HEAVY)."""
    fwd_all = _forward(close)
    past = (close / close.shift(30) - 1).values
    idx = close.index
    lev_all = _levels(rank)
    keep = (lev_all >= 0) & np.isfinite(fwd_all[MAIN_H]) & (idx >= pd.Timestamp(SAMPLE_FROM))
    start, end = idx[keep][0], idx[keep][-1]
    side_all = np.where(a20 > 0.5, 0, 1)
    cell_all = 3 * side_all + lev_all
    rng = np.random.default_rng(seed)
    res = {"start": start, "end": end, "per": {}, "keep": keep}
    spans = [("全期", start, end)] + [(lab, max(pd.Timestamp(a), start), min(pd.Timestamp(b), end))
                                     for lab, a, b in PERIODS]
    for lab, a, b in spans:
        if light and lab != "全期":
            continue
        m = keep & (idx >= a) & (idx <= b)
        if m.sum() < 2 * BLOCK:
            continue
        n = int(m.sum())
        bidx = rsl._boot_idx(n, rng)
        cell = cell_all[m]
        fwd = {h: fwd_all[h][m] for h in fwd_all}
        out = {"days": n, "span": (idx[m][0], idx[m][-1]), "n": [int((cell == k).sum()) for k in range(6)]}
        bmain = {}
        for h in ([MAIN_H] if light else list(H) + ["deep"]):
            for k in range(6):
                out[(h, k)], bm = _gstat(cell == k, fwd[h], bidx, med=not light)
                if h == MAIN_H:
                    bmain[k] = bm
        for s_ in (0, 1):
            q, hv = out[(MAIN_H, 3 * s_)]["mean"], out[(MAIN_H, 3 * s_ + 2)]["mean"]
            out[f"d{s_}"] = (q - hv) if _ok(q) and _ok(hv) else None
            out[f"d{s_}_ci"] = pi._q(bmain[3 * s_] - bmain[3 * s_ + 2])
        out["pool"] = (out["d0"] + out["d1"]) / 2 if _ok(out["d0"]) and _ok(out["d1"]) else None
        out["pool_ci"] = pi._q((bmain[0] - bmain[2] + bmain[3] - bmain[5]) / 2)
        if lab == "全期" and not light:
            res["matched"] = {}
            mb_all = []
            for s_ in (0, 1):
                sidem = side_all[m] == s_
                pv = past[m][sidem & np.isfinite(past[m])]
                edges = np.quantile(pv, np.linspace(0, 1, 11))
                sub = sidem & np.isfinite(past[m]) & np.isin(lev_all[m], (0, 2))
                dec = np.full(n, -1)
                dec[sub] = np.clip(np.searchsorted(edges, past[m][sub], side="right") - 1, 0, 9)
                sc = (lev_all[m] == 0).astype(float)
                got = {}
                for key in (MAIN_H, "deep"):
                    v = np.nan_to_num(fwd[key], nan=0.0)
                    boot = pi._matched(v[bidx], sc[bidx], dec[bidx], sub[bidx])
                    got[key] = (float(pi._matched(v, sc, dec, sub)[0]), pi._q(boot))
                    if key == MAIN_H:
                        mb_all.append(boot)
                res["matched"][s_] = got
            a0, a1 = res["matched"][0][MAIN_H][0], res["matched"][1][MAIN_H][0]
            res["matched"]["pool"] = ((a0 + a1) / 2 if _ok(a0) and _ok(a1) else None,
                                      pi._q((mb_all[0] + mb_all[1]) / 2))
            # section 3: absorption among HEAVY days
            heavy = lev_all[m] == 2
            pm = past[m]
            groups = {"abs": heavy & (pm > 0), "not": heavy & ~(pm > 0), "quiet": lev_all[m] == 0}
            res["absorb"] = {g: {h: _gstat(mk, fwd[h], bidx)[0] for h in H} for g, mk in groups.items()}
            res["absorb_past"] = {g: float(np.nanmean(pm[mk])) if mk.any() else None for g, mk in groups.items()}
        res["per"][lab] = out
    if light:
        return res
    full = res["per"]["全期"]
    both = _ok(full["d0"]) and _ok(full["d1"]) and full["d0"] > 0 and full["d1"] > 0
    lo = full["pool_ci"][0]
    above = _ok(lo) and lo > 0
    subs = [lab for lab, _, _ in PERIODS if lab in res["per"]]
    pos_subs = sum(1 for lab in subs if _ok(res["per"][lab]["pool"]) and res["per"][lab]["pool"] > 0)
    mp = res["matched"]["pool"][0]
    mpos = _ok(mp) and mp > 0
    res["pass"] = {"both": bool(both), "above": bool(above), "pos_subs": pos_subs, "n_subs": len(subs),
                   "matched": bool(mpos), "pass": bool(both and above and pos_subs >= 2 and mpos)}
    return res


# ---------- section 3: level view ----------

def _level_view(close: pd.Series, m: dict, buy: dict, keep: np.ndarray) -> dict:
    if "stable" not in buy:
        return {}
    days = pd.date_range(m["w"].index[0], close.index[-1], freq="D")
    cl = close.reindex(days)
    old_usd = (m["w"].reindex(days) * cl).rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    sup = buy["stable"].reindex(days)
    sup = sup.where(sup.index >= pd.Timestamp(ss.START))
    stable30 = sup - sup.shift(SUM_DAYS)
    etf = buy.get("etf", pd.Series(dtype=float))
    flow = etf.reindex(days).fillna(0.0)
    if len(etf):
        flow[days > etf.index[-1]] = np.nan
    etf30 = flow.rolling(SUM_DAYS, min_periods=SUM_DAYS).sum()
    buy30 = stable30 + etf30
    ratio = buy30 / old_usd.where(old_usd > 0)
    idx = close.index
    r_t, o_t, b_t = (_on(x, idx) for x in (ratio, old_usd, buy30))
    fwd = _forward(close)
    win_start = idx - (LAG + SUM_DAYS - 1) * DAY
    era = {"stable": win_start >= pd.Timestamp(ss.START) + SUM_DAYS * DAY, "etf": win_start >= rs.ETF_LAUNCH}
    out = {}
    for name, cond in era.items():
        mk = keep & cond & np.isfinite(r_t)
        if mk.sum() < 90:
            continue
        cuts = np.quantile(r_t[mk], [1 / 3, 2 / 3])
        terc = np.where(r_t <= cuts[0], 0, np.where(r_t <= cuts[1], 1, 2))
        rows = []
        for k in range(3):
            g = mk & (terc == k)
            row = {"n": int(g.sum()), "lo": float(np.min(r_t[g])), "hi": float(np.max(r_t[g])),
                   "old": float(np.median(o_t[g])), "buy": float(np.median(b_t[g]))}
            for h in H:
                v = fwd[h][g]
                v = v[np.isfinite(v)]
                row[h] = (float(v.mean()) if len(v) else None, int(len(v)))
            rows.append(row)
        out[name] = {"rows": rows, "from": idx[mk][0], "to": idx[mk][-1], "n": int(mk.sum())}
    out["now"] = (o_t[-1], b_t[-1], r_t[-1])
    return out


# ---------- section 4: tops and bottoms ----------

def _tops(close: pd.Series, rank: np.ndarray, lev: np.ndarray) -> dict:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    n = len(close)
    heavy = lev == 2

    def mr(a: int, b: int) -> str:
        a, b = max(a, 0), min(b, n)
        seg = rank[a:b]
        seg = seg[np.isfinite(seg)]
        return f"{seg.mean() * 100:.0f}" if len(seg) else "—"

    rows, peaks, pre_heavy, done = [], [], 0, 0
    for i, r in enumerate(rallies, 1):
        it, ip = r["it"], r["ip"]
        ph = None
        if r["done"]:
            done += 1
            peaks.append(ip)
            ph = bool(heavy[max(ip - BEFORE, 0):ip + 1].any())
            pre_heavy += int(ph)
        rows.append([f"{i}（{_d(r['t'])} → {_d(r['p'])}）" + ("" if r["done"] else "，還沒結束"),
                     f"{r['gain'] * 100:+.0f}%",
                     f"{mr(it - BEFORE, it)}｜{mr(it - AROUND, it + AROUND + 1)}｜{mr(it + 1, it + 1 + AFTER)}",
                     f"{mr(ip - BEFORE, ip)}｜{mr(ip - AROUND, ip + AROUND + 1)}｜"
                     + (mr(ip + 1, ip + 1 + AFTER) if r["done"] else "—"),
                     "—" if ph is None else ("有" if ph else "沒有")])
    idx = close.index
    first = int(idx.searchsorted(pd.Timestamp(rs.LIST_FROM)))
    pk = np.array(peaks, dtype=int)
    eps, last_heavy = [], -10 ** 9
    for t in range(first, n):
        if heavy[t]:
            if t - last_heavy > EP_GAP:
                hit = bool(((pk >= t) & (pk <= t + ALARM)).any())
                eps.append({"day": idx[t], "hit": hit, "known": hit or t + ALARM <= n - 1})
            last_heavy = t
    base_days = [t for t in range(first, n - ALARM) if np.isfinite(rank[t])]
    base = float(np.mean([((pk >= t) & (pk <= t + ALARM)).any() for t in base_days])) if base_days else None
    return {"rows": rows, "done": done, "pre_heavy": pre_heavy, "eps": eps, "base": base}


# ---------- rules and lead/lag ----------

def _rule_flags(rank: pd.Series) -> dict:
    """T1 = reluctant_sellers' 'g2' combination (ma50 AND NOT flag) with HEAVY; T2 = its 'r2' (ma50 OR recent ma50
    AND flag) with QUIET."""
    r = rank.values
    heavy = pd.Series(np.where(np.isfinite(r), (r > P_HEAVY).astype(float), 0.0), index=rank.index)
    quiet = pd.Series(np.where(np.isfinite(r), (r < P_QUIET).astype(float), 0.0), index=rank.index)
    return {"ma50": (None, "ma50"), "t1": (heavy, "g2"), "t2": (quiet, "r2")}


def _leadlag(old: pd.Series, close: pd.Series) -> dict:
    o = old.where(old > 0)
    x = _on(np.log(o / o.shift(14)), close.index)
    lc = np.log(close.values.astype(float))
    r14 = np.full(len(lc), np.nan)
    r14[14:] = lc[14:] - lc[:-14]
    first = int(np.argmax(np.isfinite(x)))
    f, r = x[first:], r14[first:]
    n = len(f)

    def corr(fv, k):
        a, b = (fv[:n - k], r[k:]) if k >= 0 else (fv[-k:], r[:n + k])
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    real = {k: corr(f, k) for k in LL_K}
    rng = np.random.default_rng(SEED + 83)
    pool = []
    for s in rng.integers(GAP, n - GAP + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL_SHOW]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n, "from": close.index[first]}


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, obm = data["close"], data["obm"]
    idx = close.index
    m = _measure(obm["ge155"])
    m365 = _measure(obm["ge365"])
    rank_t = pd.Series(_on(m["rank"], idx), index=idx)
    lev = _levels(rank_t.values)
    a20 = sd._above20(close.values)
    a = _test_a(close, rank_t.values, a20, SEED + 901)
    a365 = _test_a(close, _on(m365["rank"], idx), a20, SEED + 902, light=True)
    level = _level_view(close, m, data["buy"], a["keep"])
    tops = _tops(close, rank_t.values, lev)
    ll = _leadlag(m["old"], close)
    t_desc = time.time() - t0

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = rsl._Exam(close, data["other"], price, b0, pre, pd.Timestamp(FIRST_MONTH))
    flags = _rule_flags(rank_t)
    s = {k: ex.summary(f, KI[k], SIMS, kind) for k, (f, kind) in flags.items()}
    small = {k: ex.summary(flags[k][0], KI[k], PL_SIMS, flags[k][1]) for k in ("t1", "t2")}
    rstart = idx[int(np.argmax(np.isfinite(rank_t.values)))]
    n = int((idx >= rstart).sum())
    prng = np.random.default_rng(SEED + 29)
    hits = {k: 0 for k in small}
    lows = {k: [] for k in small}
    for k in prng.integers(GAP, n - GAP, DRAWS):
        fl = _rule_flags(sd._roll(rank_t, rstart, int(k)))
        for key in ("t1", "t2"):
            x = ex.summary(fl[key][0], KI[key], PL_SIMS, fl[key][1])
            lows[key].append(x["low"])
            hits[key] += int(sd._score_ge(x, small[key]))
    placebo = {k: hits[k] / DRAWS for k in small}
    pos = {k: ex.btc_pos(f, kind) for k, (f, kind) in flags.items()}
    cal = {k: rsl._cal_years(close, pos[k]) for k in pos}
    m15 = idx >= pd.Timestamp(FIRST_MONTH)
    tim = {k: float(np.nanmean(np.nan_to_num(pos[k][m15]))) for k in pos}
    differ = {k: int((pos[k][m15] != np.nan_to_num(pos["ma50"][m15])).sum()) for k in ("t1", "t2")}
    return {"m": m, "m365": m365, "rank_t": rank_t, "lev": lev, "a20": a20, "a": a, "a365": a365, "level": level,
            "tops": tops, "ll": ll, "ex": ex, "s": s, "small": small, "placebo": placebo,
            "placebo_med": {k: float(np.median(v)) for k, v in lows.items()}, "pos": pos, "cal": cal, "tim": tim,
            "differ": differ, "problem": problem, "secs": time.time() - t0, "secs_desc": t_desc, "rstart": rstart}


# ---------- report pieces ----------

def _usd(v) -> str:
    return f"{float(v) / 1e8:,.0f} 億美元" if _ok(v) else "—"


def _rel(a: float, b: float) -> float | None:
    return abs(a / b - 1) if b else None


def _validation(data: dict, m: dict) -> tuple[list, list, dict]:
    obm, meta = data["obm"], data["meta"]
    today = pd.Timestamp(common.today_utc())
    rows = []
    for k in IDS:
        s = obm[k]
        span = pd.date_range(s.index[0], s.index[-1], freq="D")
        mt = meta.get(k, {})
        rows.append([CN_ID[k], f"{_d(s.index[0])} 到 {_d(s.index[-1])}", f"{len(s):,}",
                     f"{len(span) - len(s):,}", f"{mt.get('bad', '—')}｜{mt.get('nan', '—')}",
                     "、".join(mt.get("version", [])) or "—"])
    last = obm["ge155"].index[-1]
    lag_seen = int((today - last).days)
    tot = (obm["ge155"] + obm["lt155"]).dropna()
    band = obm["band"].reindex(tot.index)
    ok = band > 1.0
    diff = ((tot[ok] / band[ok]) - 1).abs()
    recent = tot[tot.index >= pd.Timestamp("2024-01-01")]
    share = (obm["ge155"].reindex(recent.index) / recent).median()
    ck = data["cdd"]
    cdd_rel, inp_rel = [], []
    for day, x in (ck.get("days") or {}).items():
        t = pd.Timestamp(day)
        if t in obm["cdd"].index and x["cdd"] > 0:
            cdd_rel.append(abs(obm["cdd"][t] / x["cdd"] - 1))
        if t in tot.index and x["input_btc"] > 0:
            inp_rel.append(abs(tot[t] / x["input_btc"] - 1))
    cdd_old = [v for (day, _), v in zip((ck.get("days") or {}).items(), cdd_rel) if day < "2019-01-01"]
    info = {"lag": lag_seen, "last": last, "n_check": len(ck.get("days") or {}), "planned": len(ck.get("planned") or []),
            "stop": ck.get("stop"), "cdd_med": float(np.median(cdd_rel)) if cdd_rel else None,
            "cdd_max": float(np.max(cdd_rel)) if cdd_rel else None,
            "inp_max": float(np.max(inp_rel)) if inp_rel else None, "tot_max": float(diff.max()),
            "checked": ck.get("checked_utc", "")[:10], "cdd_old_max": float(np.max(cdd_old)) if cdd_old else None}
    rows2 = [["最後一筆資料日期", f"{_d(last)}（今天 {_d(today)}，晚了 {lag_seen} 天）"],
             ["採用的延遲", f"資料日 D 的數字，要等到 D+{LAG} 那天的比特幣日線收盤才拿來用（看到的延遲約 {lag_seen} 天，多留一點餘裕）"],
             ["放了 155 天以上＋不到 155 天，和 12 個年齡層加總比",
              f"最大差距 {diff.max() * 100:.4f}%（{int(ok.sum()):,} 天）"],
             ["每天被花掉的幣總量（2024 起的中位數）",
              f"{recent.median():,.0f} 顆，其中放了 155 天以上的占 {share * 100:.1f}%（中位數）；大部分是當天或幾天內又轉手的零錢和找零"],
             ["和 Coin Metrics 社群版的鏈上轉帳量比", "社群版不提供這個欄位（TxTfrValAdjNtv、TxTfrValNtv 都回 HTTP 403），沒有比；改用下一列 Blockchair 的區塊資料"],
             [f"Blockchair 每日區塊檔交叉核對（{info['checked']} 抓一次存下，固定亂數挑 {info['planned']} 天）",
              (f"抓到 {info['n_check']} 天" + (f"，在 {info['stop']} 停下" if info["stop"] else "，全部成功")
               + (f"。CDD：差距中位數 {info['cdd_med'] * 100:.2f}%、最大 {info['cdd_max'] * 100:.2f}%"
                  f"（2019 年以前的天最大 {info['cdd_old_max'] * 100:.2f}%，之後幾乎完全一樣）"
                  if info["cdd_med"] is not None else "")
               + (f"；當天被花掉的幣總量（Blockchair 的 input_total）最大差距 {info['inp_max'] * 100:.4f}%"
                  if info["inp_max"] is not None else ""))]]
    # biggest one-day spikes of old coins
    ratio = (m["s"] / m["med"]).where(m["med"] > 0)
    top = ratio.dropna().sort_values(ascending=False).head(10)
    rows3 = []
    for t, v in top.items():
        p99 = m["p99"].get(t)
        rows3.append([_d(t), f"{m['s'][t]:,.0f} 顆", f"{m['med'][t]:,.0f} 顆", f"{v:,.0f} 倍",
                      f"{p99:,.0f} 顆" if _ok(p99) else "—"])
    # added after seeing that the pre-registered top 10 are all 2011 (tiny medians): the same list from 2013 (report)
    rows4 = []
    for t, v in ratio[ratio.index >= pd.Timestamp("2013-01-01")].dropna().sort_values(ascending=False).head(10).items():
        p99 = m["p99"].get(t)
        rows4.append([_d(t), f"{m['s'][t]:,.0f} 顆", f"{m['med'][t]:,.0f} 顆", f"{v:,.0f} 倍",
                      f"{p99:,.0f} 顆" if _ok(p99) else "—"])
    gox = pd.Timestamp("2024-05-28")
    if gox in ratio.index and _ok(ratio[gox]):
        r365 = ratio[ratio.index.year == 2024].rank(ascending=False)[gox]
        info["gox"] = (float(m["s"][gox]), float(ratio[gox]), int(r365), float(m["p99"][gox]))
    return rows, rows2, {"info": info, "spikes": rows3, "spikes13": rows4}


def _cells_rows(a: dict) -> list:
    out = []
    names = [f"{SIDES[k // 3]}＋{LEVELS[k % 3]}" for k in range(6)]
    for lab, x in a["per"].items():
        sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
        for k in range(6):
            row = [sp if k == 0 else "", names[k], f"{x['n'][k]:,}（{x['n'][k] / x['days'] * 100:.0f}%）"]
            for h in H:
                c = x[(h, k)]
                row.append(f"{_p(c['mean'])} {_ci(*c['mean_ci'])}｜{_p(c['med'])} {_ci(*c['med_ci'])}")
            c = x[("deep", k)]
            row.append(f"{_p(c['mean'])} {_ci(*c['mean_ci'])}")
            out.append(row)
    return out


def _now(data: dict, r: dict) -> list[dict]:
    close, m = data["close"], r["m"]
    last = close.index[-1]
    word = {1: "持有", 0: "現金"}
    st = {k: int(r["pos"][k][-1] > 0.5) for k in ("ma50", "t1", "t2")}
    day = last - LAG * DAY
    get = lambda s: float(s[day]) if day in s.index and _ok(s[day]) else None  # noqa: E731
    old, raw30, rk = get(m["old"]), get(m["raw30"]), get(m["rank"])
    lev = r["lev"][-1]
    c = close.values.astype(float)
    side = 0 if r["a20"][-1] > 0.5 else 1
    latest = data["obm"]["ge155"].index[-1]
    lr = m["rank"].dropna()
    lv = r["level"].get("now", (None, None, None))
    return [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {c[-1]:,.0f} 美元"},
        {"label": "比特幣：50 日均線｜T1｜T2", "value": f"{word[st['ma50']]}｜{word[st['t1']]}｜{word[st['t2']]}"},
        {"label": f"今天用的老幣資料（資料日 {_d(day)}，晚 {LAG} 天才用）：近 30 天老幣移動量",
         "value": (f"{old:,.0f} 顆（每天最多算到過去一年的第 99 百分位）；不修剪的話 {raw30:,.0f} 顆"
                   if _ok(old) and _ok(raw30) else "資料不足")},
        {"label": "　在過去一年裡的排名（0–100，越高表示越多老幣在動）",
         "value": f"{rk * 100:.0f}" if _ok(rk) else "資料不足"},
        {"label": "　今天在六格的哪一格",
         "value": f"{SIDES[side]}（收盤 {c[-1]:,.0f}，20 日均線 {c[-20:].mean():,.0f}）＋{LEVELS[lev]}" if lev >= 0 else "資料不足"},
        {"label": "　OBM 最新一筆資料的日期｜最新算得出排名的資料日",
         "value": f"{_d(latest)}｜{_d(lr.index[-1])}（排名 {lr.iloc[-1] * 100:.0f}）"},
        {"label": "　近 30 天老幣移動（美元）｜同期買方（穩定幣增加＋ETF 淨流入）｜比值",
         "value": f"{_usd(lv[0])}｜{_usd(lv[1])}｜{lv[2]:.2f}" if _ok(lv[2]) else "資料不足"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；T1、T2 沒有通過考試前不會用"},
    ]


# ---------- the study ----------

def study() -> dict:
    t_all = time.time()
    data = _load()
    r = _compute(data)
    close, s, a, ex, m = data["close"], r["s"], r["a"], r["ex"], r["m"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    vrows, vrows2, vx = _validation(data, m)
    info = vx["info"]

    def yes(v: bool) -> str:
        return "通過" if v else "沒通過"

    # ----- Test A -----
    ap, full = a["pass"], a["per"]["全期"]
    rows_d = [[lab, f"{_pt(x['d0'])} {_ci(*x['d0_ci'], f=_pt)}", f"{_pt(x['d1'])} {_ci(*x['d1_ci'], f=_pt)}",
               f"{_pt(x['pool'])} {_ci(*x['pool_ci'], f=_pt)}"] for lab, x in a["per"].items()]
    f365 = r["a365"]["per"]["全期"]
    rows_d.append(["（穩健性，只描述）改用放了 1 年以上的幣，全期", f"{_pt(f365['d0'])} {_ci(*f365['d0_ci'], f=_pt)}",
                   f"{_pt(f365['d1'])} {_ci(*f365['d1_ci'], f=_pt)}", f"{_pt(f365['pool'])} {_ci(*f365['pool_ci'], f=_pt)}"])
    mt = a["matched"]
    rows_m = [[f"{SIDES[k]}（站上 20 日均線）" if k == 0 else f"{SIDES[k]}（在 20 日均線或以下）",
               f"{_pt(mt[k][MAIN_H][0])} {_ci(*mt[k][MAIN_H][1], f=_pt)}",
               f"{_pt(mt[k]['deep'][0])} {_ci(*mt[k]['deep'][1], f=_pt)}"] for k in (0, 1)]
    rows_m.append(["兩邊平均", f"{_pt(mt['pool'][0])} {_ci(*mt['pool'][1], f=_pt)}", "—"])
    rows_ap = [["全期之後 60 天平均：兩邊都是「老幣安靜 > 老幣大量移動」",
                f"在上 {_pt(full['d0'])}，在下 {_pt(full['d1'])}", yes(ap["both"])],
               ["兩邊平均的差，90% 區間整段在零以上", f"{_pt(full['pool'])} {_ci(*full['pool_ci'], f=_pt)}", yes(ap["above"])],
               ["三段裡至少兩段兩邊平均的差是正的", f"{ap['pos_subs']} 段（共 {ap['n_subs']} 段）", yes(ap["pos_subs"] >= 2)],
               ["配對過去 30 天漲跌後，兩邊平均的差是正的", _pt(mt["pool"][0]), yes(ap["matched"])],
               ["四條都要過", "", yes(ap["pass"])]]

    # ----- absorption -----
    ab = a["absorb"]
    glab = {"abs": "老幣大量移動，同 30 天價格還是漲（買方接住了）", "not": "老幣大量移動，同 30 天價格跌（沒接住）",
            "quiet": "對照：老幣安靜（全部）"}
    rows_ab = []
    for g in ("abs", "not", "quiet"):
        row = [glab[g], f"{ab[g][30]['n']:,}", _p(a["absorb_past"][g])]
        for h in H:
            c = ab[g][h]
            row.append(f"{_p(c['mean'])} {_ci(*c['mean_ci'])}｜{_p(c['med'])}")
        rows_ab.append(row)
    lvl = r["level"]
    rows_lv = []
    for era, lab in (("stable", "穩定幣有資料以後"), ("etf", "只看 ETF 上市以後")):
        if era not in lvl:
            rows_lv.append([lab, "資料不夠", "", "", "", "", "", ""])
            continue
        e = lvl[era]
        for k, row in enumerate(e["rows"]):
            rows_lv.append([f"{lab}（{_d(e['from'])} 到 {_d(e['to'])}）" if k == 0 else "",
                            ("買方相對少", "中間", "買方相對多")[k] + f"（比值 {row['lo']:.2f} 到 {row['hi']:.2f}）",
                            f"{row['n']:,}", _usd(row["old"]), _usd(row["buy"])]
                           + [f"{_p(row[h][0])}（{row[h][1]:,} 天）" for h in H])

    # ----- tops -----
    tp = r["tops"]
    eps = tp["eps"]
    known = [e for e in eps if e["known"]]
    false = [e for e in known if not e["hit"]]
    rows_ep = [[_d(e["day"]), "90 天內有大漲段的高點" if e["hit"] else ("假警報" if e["known"] else "還不知道（不到 90 天）")]
               for e in eps]

    # ----- rules -----
    base = s["ma50"]
    res = {}
    for k in ("t1", "t2"):
        x = s[k]
        slack = 0.0 if k == "t1" else LOSS_SLACK
        res[k] = {"score": sd._score_ge(x, base), "loss": x["c_deep_med"] >= base["c_deep_med"] - slack - 1e-12,
                  "plc": r["placebo"][k] <= PLACEBO_BAR}
        res[k]["pass"] = all(res[k].values())
    rows_b = []
    for k in ("ma50", "t1", "t2"):
        x = s[k]
        ck = x["checks"]
        rows_b.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點", _share(r["tim"][k]),
                       "—" if k == "ma50" else f"{r['differ'][k]:,} 天",
                       "—" if k == "ma50" else f"{_share(r['placebo'][k])}（真的 {_share1(r['small'][k]['low'])}，"
                                               f"錯位中位數 {_share1(r['placebo_med'][k])}）"])
    rows_pass = []
    for k in ("t1", "t2"):
        x = s[k]
        rows_pass += [[SHORT[k], "排名分數不輸 50 日均線", f"{_share1(x['low'])} vs {_share1(base['low'])}", yes(res[k]["score"])],
                      ["", "最深虧損中位數不比均線深" if k == "t1" else "最深虧損中位數最多比均線深 5 點",
                       f"{_pc(x['c_deep_med'])} vs {_pc(base['c_deep_med'])}", yes(res[k]["loss"])],
                      ["", "隨機錯位比例 ≤ 5%", _share(r["placebo"][k]), yes(res[k]["plc"])],
                      ["", "三條都要過", "", yes(res[k]["pass"])]]
    cal = r["cal"]
    years = [y for y in sorted(cal["ma50"]) if y >= 2012]
    rows_y = [[str(y), _p(cal["ma50"][y]), _p(cal["t1"].get(y)), _pt(cal["t1"].get(y, 0) - cal["ma50"][y]),
               _p(cal["t2"].get(y)), _pt(cal["t2"].get(y, 0) - cal["ma50"][y])] for y in years]
    win1 = sum(1 for y in years if cal["t1"].get(y, 0) - cal["ma50"][y] > 1e-9)
    lose1 = sum(1 for y in years if cal["t1"].get(y, 0) - cal["ma50"][y] < -1e-9)
    win2 = sum(1 for y in years if cal["t2"].get(y, 0) - cal["ma50"][y] > 1e-9)
    lose2 = sum(1 for y in years if cal["t2"].get(y, 0) - cal["ma50"][y] < -1e-9)

    # ----- lead/lag -----
    ll = r["ll"]
    band = ll["band"]
    rows_l = [[f"{k:+d} 天" + ("（價格先動）" if k < 0 else "（老幣先動）" if k > 0 else "（同一天）"),
               f"{ll['corr'][k]:+.3f}", "是" if abs(ll["corr"][k]) > band else "否"] for k in LL_SHOW]
    neg = {k: v for k, v in ll["corr"].items() if k < 0}
    posk = {k: v for k, v in ll["corr"].items() if k > 0}
    kn = max(neg, key=lambda k: abs(neg[k]))
    kp = max(posk, key=lambda k: abs(posk[k]))
    rows_l.append(["價格先動那邊相關最大的", f"{kn:+d} 天：{neg[kn]:+.3f}", "是" if abs(neg[kn]) > band else "否"])
    rows_l.append(["老幣先動那邊相關最大的", f"{kp:+d} 天：{posk[kp]:+.3f}", "是" if abs(posk[kp]) > band else "否"])

    hcols = [f"之後 {h} 天：平均 ［90% 區間］｜中位數 ［90% 區間］" for h in H]
    tables = [
        {"caption": ("資料檢查（第 0 步）：Open Bitcoin Metrics（OBM）每個 CSV 檔的日期範圍、筆數、中間缺幾天、被丟掉的列"
                     "（不是數字、負數或超出合理範圍｜空值）。只用一般的 HTTP 下載 CSV 檔，沒有下載或執行它的任何程式，"
                     "讀進來只留日期和數字。"),
         "columns": ["資料", "日期範圍", "筆數", "中間缺的天數", "丟掉的列：不合理｜空值", "版本"], "rows": vrows},
        {"caption": "資料檢查：延遲、總量對帳、和獨立來源交叉核對。", "columns": ["項目", "結果"], "rows": vrows2},
        {"caption": ("資料檢查：放了 155 天以上的幣「單日」被花掉最多的 10 天（和過去一年每天的中位數比幾倍）。這種一天幾十萬顆的跳動，"
                     "多半是交易所或託管機構整理錢包、破產案受託人搬幣，不是真的在賣，所以下面的量測每天最多只算到過去一年的第 99 百分位"
                     "（最後一欄），再加 30 天。"),
         "columns": ["日期", "當天老幣移動", "過去一年每天中位數", "倍數", "那天的修剪上限（第 99 百分位）"], "rows": vx["spikes"]},
        {"caption": ("上一張表的 10 天全在 2011 年（那時每天的中位數只有幾顆，倍數很容易很大），所以看到結果後加了這張（只描述）："
                     "同樣的排法，只看 2013 年以後。這些日子是不是搬幣、是誰搬的，資料本身看不出來；Mt. Gox 受託人 2024-05-28 的搬幣見附註。"),
         "columns": ["日期", "當天老幣移動", "過去一年每天中位數", "倍數", "那天的修剪上限（第 99 百分位）"], "rows": vx["spikes13"]},
        {"caption": (f"測試 A：老幣移動要和價格一起看。老幣＝放了 155 天以上沒動過的比特幣；老幣移動＝這些幣被花掉或轉走，"
                     f"可能是長期持有的人在賣。量測＝近 30 天老幣移動量（每天最多算到過去一年第 99 百分位），和自己過去一年比排名："
                     f"高於第 80 百分位＝老幣大量移動，低於中位數＝老幣安靜，其他＝中間。價格那一邊：收盤站上 20 日均線＝在上，否則＝在下。"
                     f"用 {_d(a['start'])} 起的收盤日（資料晚 {LAG} 天才用）。90% 區間用 {BLOCK} 天一塊的區塊重抽 1,000 次。"
                     "最後一欄是之後 60 天內最深跌幅的平均。"),
         "columns": ["期間", "格子", "天數（占比）"] + hcols + ["60 天內最深跌幅平均 ［90% 區間］"], "rows": _cells_rows(a)},
        {"caption": ("測試 A：「老幣安靜」減「老幣大量移動」之後 60 天平均漲跌的差（點）。事先定的方向：正的＝老幣大量移動之後比較差"
                     "（價格在上時叫「趁強出貨」，價格在下時叫「長期持有的人放棄」）。最後一列只描述：改用放了 1 年以上的幣。"),
         "columns": ["期間", "價格在上：安靜 − 大量 ［90% 區間］", "價格在下：安靜 − 大量 ［90% 區間］", "兩邊平均 ［90% 區間］"],
         "rows": rows_d},
        {"caption": ("測試 A 的配對：同一邊裡依「過去 30 天漲跌」分十組，在同一組裡比老幣安靜和老幣大量移動的日子，"
                     "避免「其實只是價格已經漲過或跌過了」。"),
         "columns": ["價格那一邊", "安靜 − 大量：之後 60 天 ［90% 區間］", "安靜 − 大量：60 天內最深跌幅 ［90% 區間］"],
         "rows": rows_m},
        {"caption": "測試 A 的通過條件（事先定好，四條都要過）。", "columns": ["條件", "數字", "結果"], "rows": rows_ap},
        {"caption": ("第 3 步（老闆的兩股力量，只描述）：老幣大量移動的日子，再看同一段 30 天價格是漲還是跌——漲＝有買方把這些幣接走了，"
                     "跌＝沒接住。和老幣安靜的日子比，全期。"),
         "columns": ["組", "天數", "過去 30 天平均漲跌"] + [f"之後 {h} 天：平均 ［90% 區間］｜中位數" for h in H], "rows": rows_ab},
        {"caption": ("第 3 步，用美元比大小（只描述）：近 30 天老幣移動量 × 當天價格，和同一段 30 天的買方（穩定幣總供給增加＋美國現貨比特幣 ETF 淨流入，"
                     "ETF 2024-01 起）比；比值＝買方 ÷ 老幣移動。依比值分三等份（分界用整段樣本自己的，所以只是描述），看之後漲跌。"
                     "穩定幣和 ETF 只有最近幾年，獨立的時段很少。"),
         "columns": ["期間", "三等份（比值範圍）", "天數", "老幣移動（美元，中位數）", "買方（美元，中位數）"]
                    + [f"之後 {h} 天平均" for h in H], "rows": rows_lv},
        {"caption": (f"第 4 步（只描述）：rally_sources 的大漲段（從低點漲 {rs.RISE * 100:.0f}% 以上、跌回 {rs.FALL * 100:.0f}% 才算結束）。"
                     f"每格是老幣排名的平均（0–100，用當時拿得到的資料）：前 {BEFORE} 天｜前後 {AROUND} 天｜之後 {AFTER} 天。"
                     f"最後一欄：高點前 {BEFORE} 天內有沒有出現過老幣大量移動。"),
         "columns": ["大漲段（低點 → 高點）", "漲幅", "低點附近：前｜前後｜之後", "高點附近：前｜前後｜之後", "高點前出現大量移動"],
         "rows": tp["rows"]},
        {"caption": (f"第 4 步：每一段「老幣大量移動」的開始日（前 {EP_GAP} 天都沒有大量移動才算新的一段），之後 {ALARM} 天內有沒有大漲段的高點。"
                     f"對照：隨便挑一天，之後 {ALARM} 天內剛好有高點的比例是 {_share(tp['base'])}。"),
         "columns": ["開始日", "結果"], "rows": rows_ep or [["—", "—"]]},
        {"caption": (f"第 5 步：兩條規則的公平考試（main_signal_check 的做法，用 reluctant_sellers 的考試程式）。比特幣日線起算月 "
                     f"{FIRST_MONTH[:7]} 到 {msc.DAILY_STARTS[1][:7]}（{base['d_n']} 個），起算月×24 個收盤時間 {base['c_n']:,} 格"
                     f"（{base['c_starts']} 個起算月），一般人做法（晚 10 小時、成本 {msc.P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個收盤時間 {SIMS} 次），其他資產 {'、'.join(ex.other)}（{base['o_n']} 個起算月；各用自己的 50 日均線，"
                     "配比特幣的老幣訊號當全市場狀態）。50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"
                     f"「在場時間」＝{FIRST_MONTH[:7]} 起比特幣日線持有的日子占幾成。隨機錯位：把老幣排名整條往後挪 {GAP} 天以上（頭尾相接），"
                     f"{DRAWS} 次，重新分出大量／安靜，整個考試重算（一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方法重算來比）。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "在場時間", "和均線不同的天數",
                     "隨機錯位比例"], "rows": rows_b},
        {"caption": ("T1、T2 的通過條件（事先定好）。這是「賣壓／買盤」這個想法的第五份研究，而且試了兩條規則，"
                     "就算有一條過，證據也很弱。"), "columns": ["規則", "條件", "數字", "結果"], "rows": rows_pass},
        {"caption": f"第 6 步（只描述）：每個日曆年的報酬（比特幣日線，扣 {COST * 100:.2f}% 成本），T1、T2 和 50 日均線比。",
         "columns": ["年", "50 日均線", "T1", "T1 減均線", "T2", "T2 減均線"], "rows": rows_y},
        {"caption": (f"第 6 步（只描述）：誰先動？「近 30 天老幣移動量」過去 14 天的變化（當時拿得到的），和比特幣 14 天漲跌錯開不同天數的相關"
                     f"（{_d(ll['from'])} 起，{ll['n']:,} 天）。負的天數＝價格先動、老幣後動；正的天數＝老幣先動、價格後動。"
                     f"雜訊帶＝把老幣那條錯開 {GAP} 天以上、{DRAWS} 次得到的 |相關| 第 95 百分位 {band:.3f}；超過才算看得出來。"
                     f"注意老幣資料晚 {LAG} 天才用，所以「-{LAG} 天」才是兩邊同一段日曆時間；+{LAG} 天以上才是老幣真的走在價格前面。"
                     "相隔不到 14 天的兩段時間有重疊，相關會被拉高。"),
         "columns": ["錯開天數", "相關", "超過雜訊帶"], "rows": rows_l},
    ]

    # ----- verdict -----
    means = [full[(MAIN_H, k)]["mean"] for k in range(6)]
    fails_a = ([] if ap["both"] else ["兩邊不是都「安靜 > 大量」"]) + ([] if ap["above"] else ["兩邊平均的區間沒有整段在零以上"]) \
        + ([] if ap["pos_subs"] >= 2 else ["三段裡正的不到兩段"]) + ([] if ap["matched"] else ["配對後不是正的"])
    zero = (f"測試 A（{_d(a['start'])} 起）：之後 60 天平均，價格在上時 老幣安靜 {_p(means[0])}、中間 {_p(means[1])}、"
            f"大量移動 {_p(means[2])}；價格在下時 安靜 {_p(means[3])}、中間 {_p(means[4])}、大量移動 {_p(means[5])}。"
            f"兩邊平均「安靜 − 大量」{_pt(full['pool'])} {_ci(*full['pool_ci'], f=_pt)}，三段裡 {ap['pos_subs']} 段是正的，"
            f"配對過去 30 天漲跌後 {_pt(mt['pool'][0])}。事先定的條件{'通過' if ap['pass'] else '沒有通過'}"
            + ("" if ap["pass"] else "（" + "、".join(fails_a) + "）") + "。"
            + ("方向和事先猜的相反：老幣大量移動之後平均反而比較好（區間整段在零以下），和「送進交易所的比特幣」那份一樣，"
               "比較像在量市場有多熱、價格已經漲了多少，不像賣壓。"
               if _ok(full["pool_ci"][1]) and full["pool_ci"][1] < 0 else ""))
    one = (f"買方有沒有接住（只描述）：老幣大量移動而價格同時在漲的日子，之後 60 天平均 {_p(ab['abs'][60]['mean'])}；"
           f"價格同時在跌的 {_p(ab['not'][60]['mean'])}；老幣安靜 {_p(ab['quiet'][60]['mean'])}。")

    def rule_line(k: str) -> str:
        x, q = s[k], res[k]
        fails = ([] if q["score"] else ["排名分數輸均線"]) + ([] if q["loss"] else ["最深虧損條件沒過"]) \
            + ([] if q["plc"] else ["隨機錯位超過 5%"])
        return (f"{SHORT[k]} 排名分數 {_share1(x['low'])}、最深虧損中位數 {_pc(x['c_deep_med'])}、隨機錯位 {_share(r['placebo'][k])}，"
                + ("三條都通過" if q["pass"] else f"沒有通過（{'、'.join(fails)}）"))

    passed = [k for k in ("t1", "t2") if res[k]["pass"]]
    two = (f"公平考試（均線排名分數 {_share1(base['low'])}、最深虧損中位數 {_pc(base['c_deep_med'])}）：{rule_line('t1')}；"
           f"{rule_line('t2')}。"
           + ("" if not passed else f"通過的 {'、'.join(SHORT[k] for k in passed)} 也只是第五份研究、兩條裡的結果，證據很弱，頂多放進每日影子紀錄觀察。"))
    three = (f"大漲段 {tp['done']} 段已結束的高點裡，{tp['pre_heavy']} 段在高點前 {BEFORE} 天內出現過老幣大量移動；"
             f"老幣大量移動共 {len(eps)} 段，已知結果的 {len(known)} 段裡 {len(false)} 段是假警報（{ALARM} 天內沒有高點；"
             f"隨便一天碰到高點的機率約 {_share(tp['base'])}）。")
    four = (f"每個日曆年（{years[0]}–{years[-1]}）：T1 比均線好 {win1} 年、差 {lose1} 年；T2 好 {win2} 年、差 {lose2} 年。"
            f"誰先動：價格先動那邊最大相關 {kn:+d} 天 {neg[kn]:+.2f}（{'超過' if abs(neg[kn]) > band else '沒超過'}雜訊帶 {band:.2f}），"
            f"老幣先動那邊最大 {kp:+d} 天 {posk[kp]:+.2f}（{'超過' if abs(posk[kp]) > band else '沒超過'}）。")
    verdict = zero + one + two + three + four

    notes = [
        "為什麼有這份研究：專案一直想量「賣壓」。第一個量法「送進交易所的比特幣」後來發現比較像在量市場有多熱鬧"
        "（reluctant_sellers：大量流入之後反而漲最多）。這次換一個比較直接的量法：放了很久沒動的幣（老幣）被花掉，"
        "代表長期持有的人把幣交出去了。老闆的原則：賣的力量不能單獨看，要和「有沒有人買走」一起看，所以有第 3 步。",
        "名詞：老幣＝放了 155 天以上沒動過的比特幣；老幣移動＝這些幣被花掉或轉走，可能是長期持有的人在賣。"
        "CDD（幣天數銷毀）＝每顆被花掉的幣 × 它放了幾天，加起來。第 80 百分位＝過去一年裡只有兩成的日子比它高。",
        "最重要的限制：幣動了不等於幣被賣了。交易所換錢包、託管機構（包括 ETF 的託管）重新整理、破產案受託人搬幣、"
        "很久以前遺失的幣被找回，都會算成老幣移動。OBM 沒有把同一個人的不同地址合併，自己轉給自己也算。"
        "每天修剪到第 99 百分位只能壓掉最誇張的單日，壓不掉連續好幾天的搬幣。",
        (f"已知的搬幣：2024-05-28 Mt. Gox 受託人搬幣那天，老幣移動 {info['gox'][0]:,.0f} 顆，是過去一年每天中位數的 {info['gox'][1]:,.0f} 倍"
         f"（2024 年第 {info['gox'][2]} 大），量測裡那天只算到 {info['gox'][3]:,.0f} 顆。" if "gox" in info else
         "2024-05-28 Mt. Gox 受託人搬幣那天在資料裡找不到明顯的跳動。"),
        "資料來源：OBM 是單一作者 2026 年發表的 v0.1.0 版，用自己的全節點算出來，資料授權 CC BY 4.0。作者說數字可能隨索引定義或版本改變，"
        "這裡用的是今天下載的版本；鏈上的歷史本身不會被改寫。Blockchair 交叉核對顯示 2019 年以後的 CDD 幾乎完全一樣，"
        "更早的差到約 2%（兩邊算「放了幾天」的方式可能不同）。核對的日子：" + "、".join(data["cdd"].get("days") or {}) + "。",
        f"延遲：OBM 最新資料比今天晚 {info['lag']} 天，所以資料日 D 的數字要到 D+{LAG} 的收盤才用。這讓訊號慢一週，"
        "也是規則表現的一部分。",
        "門檻（155 天、30／365 天、第 80／50／99 百分位、晚 7 天、20 日均線、20 天記憶、60 天的區塊）都是跑之前憑道理定的，沒有試過其他數字。"
        "說明文字和測試在執行任何程式之前就寫好，看到結果後沒有改。",
        "這是「賣壓／買盤」這個想法的第五份研究（supply_demand、price_impact、reluctant_sellers 的兩種定義、這一份），"
        "試越多次越容易碰巧有一次過，所以任何「通過」都要再打折；就算通過，也只代表規則可以放進每日影子紀錄觀察。",
        "其他幣和 MSTR 沒有自己的老幣資料，用的是比特幣的老幣訊號（當成全市場狀態），配各自的 50 日均線；MSTR 用前一天的訊號。",
        "相鄰起算日、相鄰收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {(time.time() - t_all) / 60:.1f} 分鐘（含下載；考試與隨機錯位 {(r['secs'] - r['secs_desc']) / 60:.1f} 分鐘）。",
    ]
    notes += problems

    obm_sources = [{"name": f"Open Bitcoin Metrics：{CN_ID[k]}（CSV，CC BY 4.0）", "url": OBM_BASE.format(sid)}
                   for k, sid in IDS.items()]
    return {
        "id": "old_coin_selling",
        "title": "老幣移動：放很久的比特幣被花掉，之後會不會比較差",
        "question": ("放了 155 天以上沒動過的比特幣大量被花掉（可能是長期持有的人在賣）時，配合價格在漲或在跌、"
                     "以及有沒有買方把這些幣接走，之後一到三個月的比特幣是不是比較差？能不能幫 50 日均線少虧或不被洗出場？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": obm_sources + [
            {"name": "Open Bitcoin Metrics 說明（定義引用）", "url": OBM_REPO},
            {"name": "OBM 論文 arXiv:2607.03124（摘要）", "url": OBM_PAPER},
            {"name": "OBM v0.1.0 Zenodo 存檔", "url": OBM_ZENODO},
            {"name": f"Blockchair 每日區塊檔（交叉核對 {info['n_check']} 天，網址的日期依天替換）", "url": BC_URL.format("YYYYMMDD")},
            {"name": "DefiLlama 穩定幣總供給 API", "url": ss.LLAMA},
            {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": "https://farside.co.uk/bitcoin-etf-flow-all-data/"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
