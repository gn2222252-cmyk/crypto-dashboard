"""Historical study "puell": the Puell Multiple (David Puell, 2019) = miners' daily issuance revenue in USD divided by
its 365-day average. Popular reading (Cointelegraph and many dashboards): the "red band" (> 4, some use > 3.5) marks
cycle tops, the "green band" (< 0.5) marks bottoms; "high hash rate with a low Puell historically precedes recovery".
Miner outflows are often read as miner selling (CryptoQuant warns that outflows are not sales). No public rule-based
backtest with stated costs was found. The dashboard's popular_rules study already runs the plain latch "buy < 0.5,
sell > 4" on blockchain.info revenue (fees included); this study asks the two questions that one does not: are the
band ENTRIES followed by unusual returns compared with ordinary and deep-drawdown days, and does Puell improve the
dashboard's main signal (close > 50-day average, ma50) as a top filter or a bottom catcher?

Data (checked before running): CoinMetrics community API (free, no key) serves IssTotNtv, IssTotUSD, PriceUSD and
HashRate for btc. Issuance revenue = IssTotUSD (new coins only, fees excluded - David Puell's definition); when
IssTotUSD is missing on a day, IssTotNtv x PriceUSD. Bitcoin closes: trend_ensemble._daily_close().

Pre-registered design (fixed before running; no other variants):
Puell(D) = revenue(D) / mean(revenue over D-364 .. D), full 365-day windows only. On-chain day D is used only from
    the daily UTC close of D+1 (on the 24-clock grid a close on UTC date E uses Puell(E-2)).
Band states: green = Puell < 0.5; red = Puell > 4; also red' = Puell > 3.5 (the looser level) for the event table.
    Point-in-time percentile versions: the percentile of today's Puell among all Puell values up to today
    (expanding, at least 730 values); low = below the 5th percentile, high = above the 95th.
Event = a band ENTRY: the first day in the band after at least 30 days in a row outside it. Forward 90 / 180 / 365-day
    returns from the decision close (D+1); compared with (a) every day with a Puell value, (b) every day whose close is
    at least 30% below the highest close so far (hash_ribbons' comparison), (c) every day inside the band. Event counts
    are tiny (a handful per band) - stated as such; no p-values are claimed from them.
Halvings (2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20) cut issuance in half overnight, so Puell drops ~50% at each
    halving and stays depressed for about a year while the 365-day average still contains pre-halving days. Reported:
    the overnight drop at each halving; the share of green-band days / entries that fall within 365 days after a
    halving vs the share of all days in those windows; and a halving-neutral version (each day's revenue rescaled to
    today's block subsidy by IssTotNtv -> blocks; i.e. revenue x subsidy_today / subsidy_that_day) with the same bands,
    as a description only (not a rule).
"High hash rate + low Puell" (description only): Puell < 0.5 (and the looser Puell < 1) AND the 30-day average hash
    rate within 5% of its highest 30-day average of the last 365 days; forward 90/180/365 vs the same comparisons.
Rules:
P1 (top avoidance) = ma50, but cash while Puell > 4.
P2 (bottom catching) = ma50 OR "Puell entered < 0.5 within the last 180 days" (a crossing from >= 0.5 to < 0.5
    starts / restarts a 180-day hold timer).
Exam: main_signal_check's machinery as sth_cost / ls_ratio / lead_followup use it. (1) 108 monthly starts
    2015-01 ... 2023-12 on the dashboard's daily UTC closes, textbook (0.10%); (2) 72 starts x 24 closing hours
    (Binance 1-hour grid), textbook; (3) ordinary person: 10 hours later, 0.25%, 10% of entries missed, 200 simulations
    per clock, main_signal_check's seed (ma50 uses its exact random draws); (4) no Puell for other coins, so period
    splits 2012-17 / 2018-21 / 2022-now (monthly starts inside each, measured to the period end, last start six months
    before it); check = the lowest of the three "beats holding" shares. Ranking score = lowest of the four; median
    deepest loss = median over the 1,728 textbook cells.
Placebo: the Puell series circularly shifted by >= 180 days, 200 draws, whole exam rebuilt (20 ordinary-person
    simulations per clock; the real rule scored the same way); placebo share = draws scoring at least as high.
Pass test (stated now): ranking score >= ma50's (and eligible when ma50 is) AND median deepest loss no worse than
    ma50's AND placebo share <= 10%.
Only public key-less endpoints with the dashboard's fixed user agent (common.get_json / common.HEADERS).
"""
from __future__ import annotations

import bisect
import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import blockspace as bs
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import sth_cost as sc
from dashboard.studies2 import trend_ensemble as te

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = "IssTotNtv,IssTotUSD,PriceUSD,HashRate"
SRC_CT = ("https://cointelegraph.com/news/bitcoin-miner-sell-off-fears-as-puell-multiple-nears-red-zone-last-seen-at-"
          "2017-peak")
SRC_CQ = ("https://cryptoquant.com/insights/quicktake/6af18eefcd57dd300eaa1edf-Bitcoin-miners-moved-726-BTC-as-price-"
          "slipped-toward-74K-The-question-is-where-t")

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
LOW, HIGH, HIGH2 = 0.5, 4.0, 3.5
PLO, PHI, PMIN = 0.05, 0.95, 730
GAPIN = 30          # days outside a band before a new entry counts
HOLD = 180          # P2 hold after entering the green band
HORIZONS = (90, 180, 365)
DEEP = -0.30
HALVINGS = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")
PERIODS = sc.PERIODS
LAST_GAP = sc.LAST_GAP
PLACEBO, GAP, PL_SIMS = 200, 180, 20
RULES = ("ma50", "p1", "p2")
TESTED = ("p1", "p2")
KI = {"ma50": 1, "p1": 91, "p2": 92}  # ma50 = main_signal_check's draws
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d
_ok, _share, _p, _of = sc._ok, sc._share, sc._p, sc._of


# ---------- data ----------

def _cm_fetch() -> dict:
    reply = common.get_json(CM_URL, {"assets": "btc", "metrics": CM_METRICS, "frequency": "1d",
                                     "start_time": "2009-01-03", "page_size": 10000}, timeout=60)
    data = list(reply["data"])
    for _ in range(5):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=60)
        data += reply["data"]
    out = {}
    for item in data:
        row = {}
        for m in CM_METRICS.split(","):
            try:
                row[m] = float(item[m]) if item.get(m) is not None else None
            except (TypeError, ValueError):
                row[m] = None
        out[item["time"][:10]] = row
    if len(out) < 5000:
        raise RuntimeError(f"CoinMetrics 只回傳 {len(out)} 天")
    return out


def _onchain() -> tuple[pd.DataFrame, list]:
    got, problem = bs._stored("puell_cm", _cm_fetch)
    df = pd.DataFrame.from_dict(got, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index < pd.Timestamp(common.today_utc())]  # never today's unfinished day
    return df.asfreq("D"), [problem] if problem else []


def _subsidy(dates: pd.DatetimeIndex) -> np.ndarray:
    n = np.searchsorted(np.array(pd.to_datetime(list(HALVINGS))), dates.values, side="right")
    return 50.0 / (2.0 ** n)


def _puell(rev: pd.Series) -> pd.Series:
    return rev / rev.rolling(YEAR, min_periods=YEAR).mean()


def _pctile(p: pd.Series) -> pd.Series:
    """Point-in-time percentile of each value among all values up to and including that day."""
    seen: list[float] = []
    out = np.full(len(p), np.nan)
    for i, v in enumerate(p.values):
        if not math.isfinite(v):
            continue
        bisect.insort(seen, v)
        if len(seen) >= PMIN:
            out[i] = bisect.bisect_left(seen, v) / len(seen)
    return pd.Series(out, index=p.index)


def _at(series: pd.Series, dates, lag: int) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


# ---------- rules ----------

def _timer(p: pd.Series) -> pd.Series:
    """1 on on-chain days within HOLD days after a crossing into the green band (0 otherwise, NaN before Puell)."""
    v = p.values
    prev = np.concatenate([[np.nan], v[:-1]])
    cross = (v < LOW) & (prev >= LOW)
    out = np.full(len(v), np.nan)
    last = None
    for i in range(len(v)):
        if cross[i]:
            last = i
        if math.isfinite(v[i]):
            out[i] = 1.0 if last is not None and i - last < HOLD else 0.0
    return pd.Series(out, index=p.index)


def _rules(ma: np.ndarray, pu: np.ndarray, tm: np.ndarray, keys=RULES) -> dict[str, np.ndarray]:
    """Positions after each close; pu / tm are the Puell value and P2 timer usable at that close."""
    out = {}
    if "ma50" in keys:
        out["ma50"] = ma
    if "p1" in keys:
        out["p1"] = np.where(np.isnan(ma) | np.isnan(pu), np.nan, ma * (pu <= HIGH))
    if "p2" in keys:
        out["p2"] = np.where(np.isnan(ma) | np.isnan(tm), np.nan, np.maximum(np.nan_to_num(ma), np.nan_to_num(tm)))
    return out


# ---------- the exam (sth_cost's machinery) ----------

class _Clock:
    def __init__(self, price, b0, pre):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, _ = msc._month_starts(self.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]])
            self.ma.append(ct._gt(c, sc._sma(c, 50))[npre:][:self.days])

    def wants(self, p: pd.Series, tm: pd.Series, keys) -> list[dict]:
        pu, t = _at(p, self.m.dates, 2), _at(tm, self.m.dates, 2)
        return [{k: np.nan_to_num(v, nan=0.0) for k, v in _rules(self.ma[h], pu, t, keys).items()} for h in range(24)]

    def full(self, k: str, wants: list[dict], sims: int, seed_add: int = 0) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = wants[h][k]
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[k] + h + seed_add)
            pos = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, P_COST, None)]), self.starts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - self.hr[:, None], pr - self.hr[:, None, None]
        return {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                "c_deep_worst": float(td.min()),
                "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
                "p_deep_med": float(np.median(pdd))}


def _evaluate(k: str, close: pd.Series, pos: dict, clock: _Clock, wants: list[dict], sims: int,
              seed_add: int = 0) -> dict:
    dr, dd, hr, hd = sc._grid(close, pos[k], msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    x = clock.full(k, wants, sims, seed_add)
    x.update({"d_n": len(dr), "d_beat": int((dr > hr).sum()), "d_shallow": int((dd > hd).sum()),
              "d_edge": float(np.median(dr - hr)), "d_deep_med": float(np.median(dd))})
    last_day = close.index[-1]
    per = []
    for name, a, b in PERIODS:
        end = pd.Timestamp(b) if b else last_day
        r_, d_, h_, hd_ = sc._grid(close, pos[k], a, end - LAST_GAP * DAY, end)
        per.append({"name": name, "n": len(r_), "beat": int((r_ > h_).sum()), "edge": float(np.median(r_ - h_)),
                    "deep": float(np.median(d_)), "hold_deep": float(np.median(hd_))})
    x["per"] = per
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   min(p["beat"] / p["n"] for p in per)]
    x["low"] = min(x["checks"])
    x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
    return x


# ---------- events ----------

def _entries(mask: np.ndarray) -> list[int]:
    out, outside = [], GAPIN  # the first in-band day of the record counts only after GAPIN days outside
    started = False
    for i, m in enumerate(mask):
        if m is None:
            continue
        if m:
            if started and outside >= GAPIN:
                out.append(i)
            outside = 0
        else:
            outside += 1
        started = True
    return out


def _forward(close: pd.Series) -> dict:
    out = {n: (close.shift(-n) / close - 1).values for n in HORIZONS}
    out["dd"] = (close / close.cummax() - 1).values
    return out


def _med(v) -> float | None:
    v = [float(x) for x in v if _ok(x)]
    return float(np.median(v)) if v else None


def _wins(v) -> str:
    v = [float(x) for x in v if _ok(x)]
    return f"{len(v)} 次裡 {sum(1 for x in v if x > 0)} 次" if v else "—"


def _group(fwd: dict, idx: np.ndarray) -> dict:
    out = {"n": int(len(idx))}
    for n in HORIZONS:
        f = fwd[n][idx]
        f = f[np.isfinite(f)]
        out[n] = (float(np.median(f)) if len(f) else None, float(np.mean(f > 0)) if len(f) else None, len(f))
    return out


def _band_study(close: pd.Series, state: pd.Series, valid: np.ndarray, fwd: dict, halv: np.ndarray) -> dict:
    """state: boolean per decision day (NaN = unknown). Returns entry events and group stats."""
    s = state.values
    mask = [None if not valid[i] else bool(s[i]) for i in range(len(s))]
    ent = _entries(mask)
    inband = np.where(np.array([m is True for m in mask]))[0]
    ev = []
    for i in ent:
        ev.append({"i": i, "date": close.index[i], "price": float(close.iloc[i]), "dd": float(fwd["dd"][i]),
                   **{n: (float(fwd[n][i]) if math.isfinite(fwd[n][i]) else None) for n in HORIZONS},
                   "halv": bool(halv[i])})
    return {"events": ev, "inband": _group(fwd, inband), "days": len(inband),
            "halv_share": float(np.mean(halv[inband])) if len(inband) else None}


# ---------- computation ----------

def _lookahead(close: pd.Series, p: pd.Series, ma: np.ndarray, clock: _Clock) -> tuple[int, int]:
    base = _rules(ma, _at(p, close.index, 1), _at(_timer(p), close.index, 1))
    w0 = clock.wants(p, _timer(p), TESTED)
    rng = np.random.default_rng(11)
    cuts = bad = 0
    for cut in (len(close) - 2500, len(close) - 900, len(close) - 60):
        t = close.index[cut]
        p2 = p.copy()
        later = p2.index >= t
        p2[later] = p2[later].values * np.exp(rng.normal(0, 1.0, int(later.sum())))
        alt = _rules(ma, _at(p2, close.index, 1), _at(_timer(p2), close.index, 1))
        cuts += 1
        for k in TESTED:
            if not np.array_equal(np.nan_to_num(base[k][:cut + 1], nan=-1), np.nan_to_num(alt[k][:cut + 1], nan=-1)):
                bad += 1
        j = int(clock.m.dates.searchsorted(t))
        if 0 < j < clock.days:
            e = clock.m.dates[j]
            p3 = p.copy()
            p3[p3.index >= e - DAY] *= np.exp(rng.normal(0, 1.0, int((p3.index >= e - DAY).sum())))
            w1 = clock.wants(p3, _timer(p3), TESTED)
            cuts += 1
            for h in (0, msc.PERSON_CLOCK):
                for k in TESTED:
                    if not np.array_equal(w0[h][k][:j + 1], w1[h][k][:j + 1]):
                        bad += 1
    return cuts, bad


def _compute() -> dict:
    df, problems = _onchain()
    rev = df["IssTotUSD"].copy()
    alt = df["IssTotNtv"] * df["PriceUSD"]
    filled = int((rev.isna() & alt.notna()).sum())
    rev = rev.fillna(alt)
    rev = rev[rev > 0].asfreq("D")
    first = rev.first_valid_index()
    rev = rev[rev.index >= first]
    p = _puell(rev).dropna()
    p = p.asfreq("D")
    cross_check = float((df["IssTotUSD"] / alt).dropna().sub(1).abs().median())

    # halving-neutral revenue: today's subsidy / that day's subsidy
    sub = _subsidy(rev.index)
    rev_n = rev * (_subsidy(pd.DatetimeIndex([rev.index[-1]]))[0] / sub)
    pn = _puell(rev_n).dropna().asfreq("D")

    close = te._daily_close()
    c = close.values.astype(float)
    ma = ct._gt(c, sc._sma(c, 50))
    tm = _timer(p)
    pu_t, tm_t = _at(p, close.index, 1), _at(tm, close.index, 1)
    pos = _rules(ma, pu_t, tm_t)

    # halving drops
    drops = []
    for h in HALVINGS:
        d = pd.Timestamp(h)
        a, b = p.get(d - DAY), p.get(d + DAY)
        if _ok(a) and _ok(b):
            drops.append((h, float(a), float(b), float(b / a - 1)))

    # events (decision day = on-chain day + 1)
    fwd = _forward(close)
    valid = np.isfinite(pu_t)
    hv = np.zeros(len(close), dtype=bool)
    for h in HALVINGS:
        d = pd.Timestamp(h)
        hv |= (close.index >= d) & (close.index < d + YEAR * DAY)
    pct_t = _at(_pctile(p), close.index, 1)
    pn_t = _at(pn, close.index, 1)
    hash30 = df["HashRate"].rolling(30, min_periods=30).mean()
    near_top = hash30 >= 0.95 * hash30.rolling(YEAR, min_periods=YEAR).max()
    near_t = _at(near_top.astype(float).where(hash30.rolling(YEAR, min_periods=YEAR).max().notna()), close.index, 1)

    def st(x):
        return pd.Series(x, index=close.index)

    bands = {
        "green": _band_study(close, st(pu_t < LOW), valid, fwd, hv),
        "red": _band_study(close, st(pu_t > HIGH), valid, fwd, hv),
        "red35": _band_study(close, st(pu_t > HIGH2), valid, fwd, hv),
        "plo": _band_study(close, st(pct_t < PLO), np.isfinite(pct_t), fwd, hv),
        "phi": _band_study(close, st(pct_t > PHI), np.isfinite(pct_t), fwd, hv),
        "ngreen": _band_study(close, st(pn_t < LOW), np.isfinite(pn_t), fwd, hv),
        "nred": _band_study(close, st(pn_t > HIGH), np.isfinite(pn_t), fwd, hv),
        "hash05": _band_study(close, st((pu_t < LOW) & (near_t > 0.5)), valid & np.isfinite(near_t), fwd, hv),
        "hash1": _band_study(close, st((pu_t < 1.0) & (near_t > 0.5)), valid & np.isfinite(near_t), fwd, hv),
    }
    all_idx = np.where(valid)[0]
    deep_idx = all_idx[fwd["dd"][all_idx] <= DEEP]
    groups = {"all": _group(fwd, all_idx), "deep": _group(fwd, deep_idx)}
    hv_all = float(np.mean(hv[all_idx]))

    # the exam
    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.insert(0, problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _Clock(price, b0, pre)
    wants = clock.wants(p, tm, RULES)
    s = {k: _evaluate(k, close, pos, clock, wants, SIMS) for k in RULES}

    real20 = {k: _evaluate(k, close, pos, clock, wants, PL_SIMS)["low"] for k in TESTED}
    rng = np.random.default_rng(SEED + 365)
    plac = {k: [] for k in TESTED}
    pv = p.dropna()
    for i in range(PLACEBO):
        sh = pd.Series(np.roll(pv.values, int(rng.integers(GAP, len(pv) - GAP))), index=pv.index).asfreq("D")
        tsh = _timer(sh)
        pp = _rules(ma, _at(sh, close.index, 1), _at(tsh, close.index, 1), keys=TESTED)
        wp = clock.wants(sh, tsh, TESTED)
        for k in TESTED:
            plac[k].append(_evaluate(k, close, pp, clock, wp, PL_SIMS, 7919 * (i + 1))["low"])
    placebo = {k: float(np.mean(np.array(v) >= real20[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    f15 = int(close.index.searchsorted(pd.Timestamp("2015-01-01")))
    differ = {k: float(np.mean(np.nan_to_num(pos[k][f15:]) != np.nan_to_num(pos["ma50"][f15:]))) for k in RULES}
    held = {k: float(np.nanmean(pos[k][f15:])) for k in RULES}
    red15 = int(np.sum(pu_t[f15:] > HIGH))
    look = _lookahead(close, p, ma, clock)
    return {"close": close, "p": p, "pn": pn, "pu_t": pu_t, "pct_t": pct_t, "pn_t": pn_t, "tm_t": tm_t, "pos": pos,
            "bands": bands, "groups": groups, "hv_all": hv_all, "drops": drops, "s": s, "clock": clock,
            "real20": real20, "placebo": placebo, "placebo_med": placebo_med, "differ": differ, "held": held,
            "red15": red15, "look": look, "problems": problems, "filled": filled, "cross": cross_check,
            "first": p.dropna().index[0], "last": p.dropna().index[-1], "max_p": (p.idxmax(), float(p.max())),
            "near_t": near_t}


# ---------- today ----------

def _now(r: dict) -> list[dict]:
    close, pos = r["close"], r["pos"]
    pu, pct, pn = r["pu_t"][-1], r["pct_t"][-1], r["pn_t"][-1]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    band = "—" if not _ok(pu) else ("綠區（< 0.5）" if pu < LOW else "紅區（> 4）" if pu > HIGH else
                                    "3.5 到 4 之間" if pu > HIGH2 else "中間（0.5 到 3.5）")
    return [
        {"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"},
        {"label": f"這個收盤能用的普爾倍數（CoinMetrics {_d(close.index[-1] - DAY)} 的值）",
         "value": f"{pu:.2f}，落在{band}" if _ok(pu) else "—"},
        {"label": "它在自己全部歷史裡的百分位（當時看得到的資料）", "value": f"{pct * 100:.0f}%" if _ok(pct) else "—"},
        {"label": "扣掉減半影響的版本（收入換算成現在的區塊獎勵）", "value": f"{pn:.2f}" if _ok(pn) else "—"},
        {"label": "50 日均線（主訊號）", "value": word(pos["ma50"][-1])},
        {"label": "P1：50 日均線，但普爾倍數 > 4 時拿現金", "value": word(pos["p1"][-1])},
        {"label": f"P2：50 日均線，或跌進 0.5 以下後抱 {HOLD} 天",
         "value": word(pos["p2"][-1]) + ("（綠區計時中）" if _ok(r["tm_t"][-1]) and r["tm_t"][-1] > 0.5 else "")},
        {"label": "這幾格的意思", "value": "只描述規則和指標在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]


# ---------- the study ----------

NAMES = {"ma50": "50 日均線（基準，主訊號）", "p1": f"P1：50 日均線，但普爾倍數 > {HIGH:g} 時拿現金",
         "p2": f"P2：50 日均線，或普爾倍數跌進 {LOW} 以下後抱 {HOLD} 天"}
BAND_NAMES = {"green": f"綠區：普爾倍數 < {LOW}", "red": f"紅區：普爾倍數 > {HIGH:g}", "red35": f"較鬆的紅區：> {HIGH2}",
              "plo": "百分位版：低於當時歷史第 5 百分位", "phi": "百分位版：高於當時歷史第 95 百分位",
              "ngreen": f"扣掉減半影響後 < {LOW}", "nred": f"扣掉減半影響後 > {HIGH:g}",
              "hash05": f"算力近一年高點附近＋普爾倍數 < {LOW}", "hash1": "算力近一年高點附近＋普爾倍數 < 1"}
SHORT = {"green": f"綠區 < {LOW}", "red": f"紅區 > {HIGH:g}", "red35": f"紅區 > {HIGH2}", "plo": "低於第 5 百分位",
         "phi": "高於第 95 百分位"}


def study() -> dict:
    r = _compute()
    s, close, b, g = r["s"], r["close"], r["bands"], r["groups"]
    m = s["ma50"]
    ma_elig = m["loss_ok"] >= LOSS_BAR
    ok = {}
    for k in TESTED:
        x = s[k]
        ok[k] = {"score": x["low"] >= m["low"] - 1e-12 and (x["loss_ok"] >= LOSS_BAR or not ma_elig),
                 "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10}
        ok[k]["all"] = all(ok[k].values())
    passed = [k for k in TESTED if ok[k]["all"]]
    n_d, n_c, n_p = m["d_n"], m["c_n"], m["p_n"]
    pnames = [p[0] for p in PERIODS]

    def gcell(x: dict, n: int) -> str:
        med, win, cnt = x[n]
        return f"{_p(med)}（{_share(win)} 上漲）" if cnt else "—"

    # table 1: band entries vs comparisons
    rows1 = []
    for key, name in BAND_NAMES.items():
        x = b[key]
        ev = x["events"]
        rows1.append([name, f"{len(ev)} 次（{x['days']:,} 天）"] +
                     [f"{_p(_med(e[n] for e in ev))}（{_wins(e[n] for e in ev)}上漲）" for n in HORIZONS] +
                     [gcell(x["inband"], 365)])
    rows1.append([f"對照：每一天（{_d(close.index[np.isfinite(r['pu_t'])][0])} 起）", f"{g['all']['n']:,} 天"]
                 + [gcell(g["all"], n) for n in HORIZONS] + ["—"])
    rows1.append(["對照：比歷史最高價低 30% 以上的每一天", f"{g['deep']['n']:,} 天"]
                 + [gcell(g["deep"], n) for n in HORIZONS] + ["—"])

    # table 2: every entry of the fixed bands
    rows2 = []
    for key in ("green", "red", "red35", "plo", "phi"):
        for e in b[key]["events"]:
            rows2.append([SHORT[key],
                          _d(e["date"]), f"{e['price']:,.0f}", _p(e["dd"]) if e["dd"] < -0.0005 else "正在創新高",
                          " ／ ".join(_p(e[n]) for n in HORIZONS), "是" if e["halv"] else "否"])

    # table 3: halving
    rows3 = [[h, f"{a:.2f}", f"{bb:.2f}", _p(dd)] for h, a, bb, dd in r["drops"]]

    # table 4: exam
    rows4 = []
    for k in RULES:
        x = s[k]
        per = "｜".join(_share(q["beat"] / q["n"]) for q in x["per"])
        rows4.append([NAMES[k], f"{_share(x['checks'][0])}｜{_pp(x['d_edge'])}", f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}", per, _share(x["low"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", _share(x["loss_ok"]),
                      f"{_share(r['held'][k])}｜{_share(r['differ'][k])}"])
    rows4.append(["對照：一直抱著不動", "—", "—", "—", "—", "—",
                  f"{_pct(float(np.median(r['clock'].hd)))}｜{_pct(float(r['clock'].hd.min()))}", "—", "100%｜—"])

    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows5 = []
    for k in TESTED:
        x = s[k]
        rows5.append([NAMES[k], f"{_share(x['low'])} vs {_share(m['low'])}（{yes(ok[k]['score'])}）",
                      f"{_pct(x['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（亂移後中位數 {_share(r['placebo_med'][k])}，真的 {_share(r['real20'][k])}）"
                      f"（{yes(ok[k]['placebo'])}）", "通過" if ok[k]["all"] else "沒有通過"])

    # verdict
    gr, rd = b["green"], b["red"]
    ga, gd = g["all"], g["deep"]
    gre = gr["events"]
    if passed:
        first = (f"有通過：{'、'.join(NAMES[k].split('：')[0] for k in passed)}。"
                 f"{NAMES[passed[0]]}的排名分數 {_share(s[passed[0]]['low'])}（50 日均線 {_share(m['low'])}），"
                 f"亂移對照 {_share(r['placebo'][passed[0]])}。")
    else:
        why = []
        for k in TESTED:
            miss = [w for w, v in (("排名分數輸 50 日均線", ok[k]["score"]), ("最深虧損比 50 日均線深", ok[k]["deep"]),
                                   ("亂移對照超過 10%", ok[k]["placebo"])) if not v]
            why.append(f"{NAMES[k].split('：')[0]}（{'、'.join(miss)}）")
        first = "兩條普爾倍數規則都沒有通過事先定好的考試：" + "；".join(why) + "。"
    second = (f"P1 在 2015 年起只有 {r['red15']} 天碰到紅區，和 50 日均線只有 {_share(r['differ']['p1'])} 的日子不同；"
              f"P2 有 {_share(r['differ']['p2'])} 的日子不同，排名分數 {_share(s['p2']['low'])}（50 日均線 {_share(m['low'])}），"
              f"最深虧損中位數 {_pct(s['p2']['c_deep_med'])}（50 日均線 {_pct(m['c_deep_med'])}）。")
    third = (f"事件本身很少：綠區（< {LOW}）新進場只有 {len(gre)} 次，之後 365 天中位數 {_p(_med(e[365] for e in gre))}"
             f"（{_wins(e[365] for e in gre)}上漲）；每一天是 {_p(ga[365][0])}、比高點低 30% 以上的日子是 {_p(gd[365][0])}。"
             f"紅區（> {HIGH:g}）只有 {len(rd['events'])} 次，365 天中位數 {_p(_med(e[365] for e in rd['events']))}。")
    dr = [d for _, _, _, d in r["drops"]]
    hs = gr["halv_share"]
    more = _ok(hs) and hs > r["hv_all"] * 1.25
    pushed = [h[:4] for h, a, bb, _ in r["drops"] if a >= LOW > bb]
    fourth = (f"減半那天普爾倍數一夜少 {-max(dr) * 100:.0f}% 到 {-min(dr) * 100:.0f}%，但綠區的日子只有 {_share(hs)} 落在減半後一年內"
              f"（全部日子是 {_share(r['hv_all'])}），"
              + ("綠區有一大部分是減半造成的。" if more else
                 "所以綠區大多來自價格大跌，不是減半造成的"
                 + (f"；只有 {'、'.join(pushed)} 年那次減半直接把它推進 {LOW} 以下。" if pushed else "。"))) if dr else ""
    red_in = rd["inband"][365][0]
    rev_ = rd["events"]
    yrs = "、".join(str(e["date"].year) for e in rev_)
    up = sum(1 for e in rev_ if _ok(e[365]) and e[365] > 0)
    p21 = float(r["p"][(r["p"].index >= "2021-01-01") & (r["p"].index < "2022-01-01")].max())
    fifth = (f"「在紅區裡的每一天」之後 365 天中位數 {_p(red_in)}，但進紅區那一刻（{yrs}）之後一年有 {up} 次還在上漲，"
             f"太早喊頂；2021 年兩次頂部時普爾倍數最高只有 {p21:.2f}，根本沒進紅區。")
    verdict = first + third + fifth + second + fourth

    look_c, look_b = r["look"]
    mx = r["max_p"]
    notes = [
        f"資料：CoinMetrics 社群版免費 API（不用金鑰）的 IssTotUSD（新發行的幣乘價格，不含手續費，David Puell 的原始定義）；"
        f"缺的日子用 IssTotNtv × PriceUSD 補（補了 {r['filled']} 天，兩者差距中位數 {r['cross'] * 100:.2f}%）。"
        f"普爾倍數從 {_d(r['first'])} 起有值，最高 {mx[1]:.1f}（{_d(mx[0])}）。",
        f"時間規則：第 D 天的鏈上數字最早在 D+1 天 UTC 0 點收盤才用；24 個收盤時間的格子裡，日期 E 的收盤只用到 E−2 天的值。"
        f"打亂切點之後的普爾倍數再算一次，切點以前的決定不能變：{look_c} 次檢查有 {look_b} 次變了。",
        f"「新進場」＝在區外連續待滿 {GAPIN} 天以後第一次進區；之後 90／180／365 天的報酬從決定那天收盤算起。"
        "事件只有幾次，每次之後的報酬又彼此重疊，數字只能當描述，不能當統計證據。",
        "百分位版：每天的普爾倍數在「到那天為止」全部歷史裡排第幾（至少要有 730 天），不用到未來的資料。",
        "扣掉減半影響的版本：把每天的收入乘上「現在的區塊獎勵 ÷ 當天的區塊獎勵」，等於把過去的發行量都換成今天的標準，"
        "減半那天就不會一夜少一半。這只是描述，不是事先定好要考的規則。",
        "算力條件：30 天平均算力在過去 365 天最高的 30 天平均的 95% 以內，資料也是 CoinMetrics 社群版。",
        "礦工把幣轉出錢包常被解讀成賣幣，但 CryptoQuant 自己也提醒轉出不等於賣出；這裡只用發行收入，沒有用礦工錢包流量。",
        "儀表板的 popular_rules 研究已經測過「低於 0.5 買、高於 4 賣」的鎖定規則（用含手續費的 blockchain.info 礦工收入），"
        "這裡測的是不同的問題：進區之後的報酬，以及拿它來補強 50 日均線。",
        "第四項檢查：普爾倍數只有比特幣有，沒有其他幣可以換，所以改用分段：2012–17、2018–21、2022 到現在，"
        "每段裡每個月 1 日起算、只算到該段結束，取三段裡「報酬贏過一直抱著」比例最低的那段。",
        f"排名分數和 main_signal_check 一樣是四項裡最低的那項；50 日均線用同樣的計算和亂數。亂移對照：把普爾倍數整條往後挪 {GAP} 天以上"
        f"（頭尾相接），重抽 {PLACEBO} 次、整個考試重算；一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方式算一次再比。"
        "如果規則和 50 日均線幾乎一樣，亂移後也幾乎一樣，亂移對照就會很高。",
        "成本很粗略：照課本做扣 0.10%、一般人做法扣 0.25%，沒有另外算滑價；現金沒有利息。一直抱著不扣成本。",
        "過去的結果不保證未來會一樣。這裡只是比較幾條固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    tables = [
        {"caption": ("一、進區之後比特幣漲跌（中位數，括號是上漲的次數或比例）。「新進場」是在區外待滿 30 天後第一次進區；"
                     "最右欄是「在區內的每一天」之後 365 天。對照組照 hash_ribbons 的做法：每一天、比歷史最高價低 30% 以上的每一天。"),
         "columns": ["區間", "新進場次數（區內天數）", "之後 90 天", "之後 180 天", "之後 365 天", "區內每一天之後 365 天"],
         "rows": rows1},
        {"caption": "二、每一次新進場（固定門檻與百分位版）。報酬依序是之後 90／180／365 天；「減半後一年內」表示那天在某次減半後 365 天內。",
         "columns": ["區間", "決定日", "收盤（美元）", "離歷史高點", "之後 90／180／365 天", "減半後一年內"],
         "rows": rows2},
        {"caption": "三、減半前後一天的普爾倍數：新發行量一夜減半，普爾倍數跟著掉一半左右，之後約一年都偏低（365 天平均裡還有減半前的日子）。",
         "columns": ["減半日", "前一天", "後一天", "變化"], "rows": rows3},
        {"caption": (f"四、公平考試（main_signal_check 的做法）。每格「報酬贏過一直抱著的比例｜每年多幾個百分點（中位數）」："
                     f"比特幣 UTC 0 點日線 {msc.DAILY_STARTS[0][:7]} 到 {msc.DAILY_STARTS[1][:7]} 每月 1 日起算（{n_d} 個）、"
                     f"起算日 × 24 個收盤時間照課本做（{n_c:,} 格）、一般人做法（晚 10 小時、0.25%、漏掉 1 成進場，{n_p:,} 次模擬），"
                     "以及三段期間各自的比例（取代換幣）。排名分數＝四項裡最低的。最右欄從 2015 起算。"),
         "columns": ["規則", f"日線起算日 {n_d} 個", f"起算日×收盤時間 {n_c:,} 格", "一般人做法", "三段期間：" + "｜".join(pnames),
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "虧得比抱著淺", "持有天數｜和 50 日均線不同的日子"],
         "rows": rows4},
        {"caption": "五、事先定好的通過條件：排名分數不低於 50 日均線、最深虧損中位數不比 50 日均線深、亂移對照不超過 10%。三項都過才算通過。",
         "columns": ["規則", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "亂移對照（≥ 真的比例）", "結果"],
         "rows": rows5},
    ]
    return {
        "id": "puell",
        "title": "普爾倍數：礦工收入的紅區、綠區有用嗎？",
        "question": ("普爾倍數（礦工每天新發行收入 ÷ 過去 365 天平均）常被說「高於 4 是頂部紅區、低於 0.5 是底部綠區」。"
                     "進區之後比特幣真的比平常好或差嗎？拿它來補強 50 日均線（紅區避開、綠區提早進場）能通過公平考試嗎？減半對它有多大影響？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "CoinMetrics 社群版 API：比特幣每日發行收入、價格、算力",
             "url": CM_URL + "?assets=btc&metrics=IssTotNtv,IssTotUSD,PriceUSD,HashRate&frequency=1d"},
            {"name": "Cointelegraph：普爾倍數接近紅區（上次在 2017 年頂部）", "url": SRC_CT},
            {"name": "CryptoQuant：礦工轉出 726 BTC，轉出不等於賣出", "url": SRC_CQ},
            {"name": "幣安現貨 BTC/USDT 日線與 1 小時線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
