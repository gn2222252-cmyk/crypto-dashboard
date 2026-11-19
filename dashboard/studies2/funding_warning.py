"""Historical study "funding_warning": for a spot-only holder, does "perpetual funding unusually high" work as a
warning that improves the dashboard's main signal ma50 (hold BTC while the daily close is above its 50-day average)?

Claim checked (BIS Working Paper 1087, Schmeling, Schrimpf & Todorov, "Crypto carry", April 2023, read on
2026-11-19): the BIS page says "a high crypto carry predicts future price crashes"; the paper (section 3.4, Table 8)
regresses one-month-ahead crash measures on the 1-month dated-futures basis (Skew data, March 2019 - January 2022,
OKEx and CME): a higher basis predicts more negative option-implied skew, more negative realised skew (significant
only for OKEx) and lower spot returns, strongest at 3 months; Table 6: a 10% rise in carry predicts short-futures
("sell") liquidations rising by 44% of open interest over the next month, and does NOT predict long liquidations. The
paper uses dated futures, not perpetual funding, and states no trading rule; the rule below is ours.

How this differs from the earlier studies: funding_positioning.py describes Binance funding buckets (negative funding,
no rule, no exam); basis_filter.py uses the dated-futures basis on the 17-rule blend; carry.py measures the carry
trade itself. Here the warning is perpetual funding, used as a daily gate on ma50, scored on the full fair exam.

Data (fixed before running):
  Planned main series: BitMEX XBTUSD perpetual funding from 2016 (public REST /api/v1/funding). On 2026-11-19 the
  first plain request with the fixed user agent "btc-research-dashboard" was answered "403 ... Request blocked" by
  CloudFront. Per the brief, BitMEX is therefore NOT used at all (no retries, no other host, no other user agent).
  Fallback = the existing project loader carry._funding("BTCUSDT"): every Binance BTCUSDT perpetual settlement since
  the contract began (2019-09), rate per 8 hours. A good download is stored in dashboard/cache/_store/
  funding_warning_binance.json; a failed download falls back to that stored copy. The longer 2016-19 history that was
  the point of this study is therefore NOT obtained; the 2016-19 period has no funding data and no episodes. The
  BitMEX-vs-Binance correlation cross-check cannot be computed and is reported as such.
  Prices: the dashboard's daily closes (te._daily_close, label D = the day ending at D+1 00:00 UTC), Binance 1-hour
  bars for the clock grid, other assets as in main_signal_check.

Signal (fixed before running):
  F = average funding rate of the latest 21 settlements (7 days of 8-hour prints), x 1095 = per year. F is computed
  at each settlement. Hot = F above the 90th percentile of the F values at all settlements within the trailing 365
  days (including the current one; expanding until 365 days exist; undefined until the window spans 180 days).
  Point in time: at a decision time t, use the latest settlement strictly before t (a print stamped exactly t is not
  used). Daily close D -> t = D+1 00:00 UTC; clock h of the hourly grid -> t = that close; MSTR (US close) -> t =
  D 20:00 UTC; other coins as BTC. Where hot is undefined, W1 = ma50 (no gate) - but no start before the first
  day with a defined flag is scored.
  W1 (main): hold when ma50 is on AND not hot, decided at each daily close. Robustness, one row: hold when ma50 is on
  AND F (same timing) is not above 30% a year.

Event study (fixed before running): day D "turns hot" when hot at D and not hot at D-1; episodes at least 14 days
  apart (a turn within 14 days of the previous episode start is skipped). For each episode: BTC close-to-close return
  over the next 7 / 14 / 30 days and the deepest drop within 30 days (lowest close of D+1..D+30 vs close D, capped
  at 0). Compared with all days of the same period (days with 30 days of future); random-day comparison: 5,000
  random samples of the same size drawn without replacement from those days; share of samples whose mean is at
  least as bad (<=) as the episodes' mean. Periods by episode day: 2016-19 / 2020-21 / 2022-now; episodes counted
  per period.

Fair exam: main_signal_check's four checks through business_cycle._Exam's plumbing (as nfci_filter), but the gate is
  daily: BTC daily textbook starts, 72 x 24 clock cells (textbook) and the same cells as an ordinary person (decide at
  the clock close, act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock), other assets
  ETH / SOL / BNB / XRP / MSTR with their own ma50 and the SAME BTC funding gate. Ranking score = lowest of the four
  "beats holding" shares (eligibility: deepest loss shallower than holding in >= 95% of BTC starts and cells).
  Starts before the first day with a defined hot flag cannot be scored; those starts are dropped from EVERY part of
  the exam (BTC daily, clock grid, each other asset) and ma50 is re-scored on exactly the same remaining starts; the
  number of starts remaining is reported.

Pass test (all four, fixed before running):
  (1) W1 ranking score >= ma50's on the same starts;
  (2) W1 median deepest loss over the clock cells no worse than ma50's;
  (3) placebo share <= 10%: the settlement-level hot flag circularly shifted (within its defined part) by at least
      540 settlements (180 days) and at most length - 540, 200 draws, whole exam re-scored with the ordinary person at
      50 simulations per clock (the real W1 re-scored the same way); share of draws scoring at least as high;
  (4) in every period with at least 5 episodes, the episodes' mean 30-day return is below the all-days mean of that
      period; if no period has 5 episodes, (4) is not met.
Mechanism (report only): correlation of daily F (at each daily decision) with BTC's 7-day log return ending k days
  later, k = -30..+30 (k > 0: funding first); noise band = 95th percentile of |corr| over 200 circular shifts of F by
  >= 180 days, pooled over all k. How much "hot" is just "price ran up": correlation of the daily hot flag with BTC's
  trailing 30-day return; hot days vs not-hot days in the same decile of trailing 30-day return (decile-weighted
  difference of the next-30-day return and deepest drop), with 200 circular shifts of the hot flag as a noise check.
Also reported: time in market and switches per year, calendar-year returns (from the first scored day), look-ahead
  scramble check, today's state row "比特幣：50 日均線｜W1".
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import carry as cy
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

N_AVG = 21                  # settlements in F (7 days of 8-hour prints)
PER_YEAR = 3 * 365          # 8-hour prints per year
PCT = 0.90
TRAIL = "365D"
MIN_SPAN = pd.Timedelta(days=180)
ABS_HOT = 0.30              # robustness: F above 30% a year
GAP_EP = 14                 # days between episodes
HORIZ = (7, 14, 30)
RAND = 5000
DRAWS, PL_SIMS = 200, 50
GAP_PRINTS = 540            # 180 days of 8-hour prints
LL = range(-30, 31)
SHIFT_DAYS = 180
MIN_EP = 5
PERIODS = (("2016–19", 2016, 2019), ("2020–21", 2020, 2021), ("2022–今", 2022, 2100))
KI = {"ma50": 1, "w1": 91, "w30": 92}
COST, P_COST, P_MISS, SIMS, SEED, YEAR = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED, msc.YEAR
DAY, HOUR = pd.Timedelta(days=1), pd.Timedelta(hours=1)

BIS_PAGE = "https://www.bis.org/publ/work1087.htm"
BIS_PDF = "https://www.bis.org/publications/working-paper-1087-crypto-carry.pdf"
BITMEX_URL = "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT"
STORE_PATH = common.STORE / "funding_warning_binance.json"

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _funding() -> tuple[pd.Series, str, list[str]]:
    """Binance BTCUSDT settlements (fraction per 8 hours), via carry's loader; the stored copy as fallback."""
    notes = []
    try:
        s = cy._funding("BTCUSDT")
        where = "幣安 BTCUSDT 永續合約（這次下載）"
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE_PATH.write_text(json.dumps({str(int(t.value // 1_000_000)): float(v) for t, v in s.items()}),
                                  encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not STORE_PATH.exists():
            raise
        got = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        s = pd.Series({pd.Timestamp(int(k), unit="ms"): float(v) for k, v in got.items()}).sort_index()
        where = "幣安 BTCUSDT 永續合約（上次存下來的資料）"
        notes.append(f"幣安資金費率這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {_d(s.index[-1])}）。")
    return s, where, notes


def _signal(rate: pd.Series) -> pd.DataFrame:
    """Per settlement: F (per year), its trailing 90th percentile and the hot flag (NaN = undefined)."""
    F = rate.rolling(N_AVG, min_periods=N_AVG).mean() * PER_YEAR
    Fv = F.dropna()
    thr = Fv.rolling(TRAIL, min_periods=1).quantile(PCT)
    thr = thr.where(Fv.index - Fv.index[0] >= MIN_SPAN)
    hot = (Fv > thr).astype(float).where(thr.notna())
    out = pd.DataFrame({"rate": rate, "F": F})
    out["thr"] = thr.reindex(out.index)
    out["hot"] = hot.reindex(out.index)
    out["abs"] = (F > ABS_HOT).astype(float).where(F.notna())
    return out


def _idx(times_ns: np.ndarray, at) -> np.ndarray:
    """Index of the latest settlement strictly before each decision time (-1 = none)."""
    return np.searchsorted(times_ns, pd.DatetimeIndex(at).values.astype("datetime64[ns]").astype(np.int64),
                           side="left") - 1


def _take(vals: np.ndarray, ix: np.ndarray) -> np.ndarray:
    out = np.where(ix >= 0, vals[np.clip(ix, 0, None)], np.nan)
    return out


def _gate(flag: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """1 = allowed (not hot or unknown), 0 = hot."""
    h = _take(flag, ix)
    return np.where(h == 1.0, 0.0, 1.0)


# ---------- the exam ----------

class _Exam(bc._Exam):
    """business_cycle's exam with a DAILY gate and only the starts that the funding flag can score."""

    def __init__(self, data: dict, price, b0, pre, tns: np.ndarray, first: pd.Timestamp):
        super().__init__(data, price, b0, pre)
        self.first = first
        keep = np.array([pd.Timestamp(x) >= first for x in self.dlabels])
        self.d_all = len(self.dstarts)
        self.dstarts, self.dlabels = self.dstarts[keep], [x for x, k in zip(self.dlabels, keep) if k]
        self.dhold = (self.dhold[0][keep], self.dhold[1][keep])
        keep = np.array([pd.Timestamp(x) >= first for x in self.clabels])
        self.c_all = len(self.cstarts)
        self.cstarts, self.clabels = self.cstarts[keep], [x for x, k in zip(self.clabels, keep) if k]
        self.chold = (self.chold[0][keep], self.chold[1][keep])
        self.o_all = {}
        for a, o in list(self.other.items()):
            keep = np.array([pd.Timestamp(x) >= first for x in o["labels"]])
            self.o_all[a] = len(o["starts"])
            if not keep.any():
                del self.other[a]
                continue
            o["starts"], o["labels"] = o["starts"][keep], [x for x, k in zip(o["labels"], keep) if k]
            o["hold"] = (o["hold"][0][keep], o["hold"][1][keep])
        # where each decision time looks in the settlement series
        self.ix_d = _idx(tns, self.close.index + DAY)
        self.ix_c = [_idx(tns, self.m.dates + h * HOUR) for h in range(24)]
        self.ix_o = {a: _idx(tns, o["close"].index + (20 * HOUR if a == "MSTR" else DAY))
                     for a, o in self.other.items()}

    def pos_btc(self, kind: str, flag) -> np.ndarray:
        return self.btc_ma if kind == "ma50" else self.btc_ma * _gate(flag, self.ix_d)

    def score(self, kind: str, flag, sims: int = SIMS) -> dict:
        pos = self.pos_btc(kind, flag)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        dr, dd = msc._suffix(msc._equity(net), self.dstarts, YEAR)
        dr, dd = dr[:, 0], dd[:, 0]
        dh = self.dhold
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            g = 1.0 if kind == "ma50" else _gate(flag, self.ix_c[h])
            w = np.nan_to_num(self.cma[h] * g, nan=0.0).astype(float)
            r, d_ = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            p = rt._missed(wl, P_MISS, rng, sims)
            r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, p, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, d_
        hr, hd = self.chold
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        each = {}
        for a, o in self.other.items():
            p = o["ma"] if kind == "ma50" else o["ma"] * _gate(flag, self.ix_o[a])
            net, _ = te._run(np.concatenate([[np.nan], p[:-1]]), o["change"], COST)
            r, d_ = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[a] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((d_[:, 0] > o["hold"][1]).sum()),
                       float(np.median(r[:, 0] - o["hold"][0])))
        x = {"d_n": len(dr), "d_beat": int((dr > dh[0]).sum()), "d_edge": float(np.median(dr - dh[0])),
             "d_shallow": int((dd > dh[1]).sum()), "d_deep_med": float(np.median(dd)),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        return x


# ---------- event study ----------

def _forward(close: pd.Series) -> pd.DataFrame:
    c = close.values
    n = len(c)
    out = {}
    for h in HORIZ:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        out[f"r{h}"] = f
    low = pd.Series(c[::-1]).rolling(30, min_periods=30).min().values[::-1]  # min of c[i..i+29]
    deep = np.full(n, np.nan)
    deep[:n - 30] = np.minimum(low[1:n - 29] / c[:n - 30] - 1, 0.0)
    out["deep"] = deep
    return pd.DataFrame(out, index=close.index)


def _episodes(hot_d: pd.Series) -> list[pd.Timestamp]:
    h = hot_d.fillna(0.0).values
    turn = np.flatnonzero((h[1:] == 1.0) & (h[:-1] == 0.0) & hot_d.notna().values[:-1]) + 1
    eps, last = [], None
    for i in turn:
        d = hot_d.index[i]
        if last is None or (d - last).days >= GAP_EP:
            eps.append(d)
            last = d
    return eps


def _events(close: pd.Series, hot_d: pd.Series, rng) -> dict:
    fw = _forward(close)
    eps = _episodes(hot_d)
    out = {}
    for lab, a, b in PERIODS:
        span = fw[(fw.index.year >= a) & (fw.index.year <= b)].dropna()
        e = [d for d in eps if a <= d.year <= b and d in span.index]
        x = {"n": len(e), "n_all_turns": sum(1 for d in eps if a <= d.year <= b), "days": len(span)}
        for col in ("r7", "r14", "r30", "deep"):
            x[f"all_{col}"] = float(span[col].mean()) if len(span) else None
            x[f"ep_{col}"] = float(span.loc[e, col].mean()) if e else None
            x[f"epmed_{col}"] = float(span.loc[e, col].median()) if e else None
        for col in ("r30", "deep"):
            if e and len(span) > len(e):
                v = span[col].values
                draws = np.array([v[rng.choice(len(v), len(e), replace=False)].mean() for _ in range(RAND)])
                x[f"rand_{col}"] = float(np.mean(draws <= x[f"ep_{col}"] + 1e-15))
            else:
                x[f"rand_{col}"] = None
        out[lab] = x
    tested = [p for p in out.values() if p["n"] >= MIN_EP]
    ok = bool(tested) and all(p["ep_r30"] < p["all_r30"] for p in tested)
    return {"periods": out, "eps": eps, "tested": len(tested), "ok": ok}


# ---------- mechanism ----------

def _leadlag(close: pd.Series, Fd: pd.Series) -> dict:
    r7 = np.log(close / close.shift(7))
    both = pd.concat([Fd.rename("f"), r7.rename("r")], axis=1)
    both = both[both["f"].notna()]
    f, r = both["f"].values, both["r"].values
    n = len(f)

    def corr(fv: np.ndarray, k: int) -> float:
        if k >= 0:
            a, b = fv[:n - k], r[k:]
        else:
            a, b = fv[-k:], r[:n + k]
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    real = {k: corr(f, k) for k in LL}
    rng = np.random.default_rng(SEED + 77)
    pool = []
    for s in rng.integers(SHIFT_DAYS, n - SHIFT_DAYS + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n,
            "first": both.index[0], "last": both.index[-1]}


def _runup(close: pd.Series, hot_d: pd.Series) -> dict:
    fw = _forward(close)
    r30b = close / close.shift(30) - 1
    df = pd.DataFrame({"hot": hot_d, "past": r30b, "f30": fw["r30"], "deep": fw["deep"]}).dropna()
    corr = float(np.corrcoef(df["hot"], df["past"])[0, 1])
    edges = np.quantile(df["past"], np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, df["past"].values, side="right") - 1, 0, 9)

    def matched(h: np.ndarray) -> tuple[float, float, int]:
        num_r = num_d = w = 0.0
        for k in range(10):
            m = b == k
            hh, nn = m & (h == 1), m & (h == 0)
            if hh.sum() and nn.sum():
                c = hh.sum()
                num_r += c * (df["f30"].values[hh].mean() - df["f30"].values[nn].mean())
                num_d += c * (df["deep"].values[hh].mean() - df["deep"].values[nn].mean())
                w += c
        return (num_r / w if w else np.nan), (num_d / w if w else np.nan), int(w)

    h = df["hot"].values
    mr, md, used = matched(h)
    top2 = float(np.mean(b[h == 1] >= 8)) if (h == 1).any() else np.nan
    rng = np.random.default_rng(SEED + 78)
    n = len(h)
    sims = []
    for s in rng.integers(SHIFT_DAYS, n - SHIFT_DAYS + 1, DRAWS):
        sims.append(matched(np.roll(h, int(s))))
    sims = np.array([[x[0], x[1]] for x in sims])
    return {"corr": corr, "matched_r": mr, "matched_d": md, "used": used, "hot_days": int((h == 1).sum()),
            "days": n, "top2": top2,
            "raw_r": float(df["f30"][h == 1].mean() - df["f30"][h == 0].mean()),
            "raw_d": float(df["deep"][h == 1].mean() - df["deep"][h == 0].mean()),
            "p_r": float(np.mean(sims[:, 0] <= mr)), "p_d": float(np.mean(sims[:, 1] <= md)),
            "past_hot": float(df["past"][h == 1].mean()), "past_all": float(df["past"].mean())}


# ---------- look-ahead ----------

def _lookahead(close: pd.Series, rate: pd.Series) -> tuple[int, int]:
    rng = np.random.default_rng(13)
    total = bad = 0
    tns = rate.index.values.astype("datetime64[ns]").astype(np.int64)
    ma = msc._targets(close)["ma50"].values.astype(float)
    sig = _signal(rate)
    ix = _idx(tns, close.index + DAY)
    for cut in ("2021-03-31", "2022-11-15", "2025-01-31"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        r2 = rate.where(rate.index < c + DAY, rate + rng.normal(0, 0.001, len(rate)))
        sig2 = _signal(r2)
        upto = close.index <= c
        for col in ("hot", "abs"):
            a = ma * _gate(sig[col].values, ix)
            b = ma2 * _gate(sig2[col].values, ix)
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _switches(pos: np.ndarray, idx: pd.DatetimeIndex, first) -> tuple[float, float]:
    p = np.nan_to_num(pos[idx >= first]) > 0.5
    entries = int(np.sum(p[1:] & ~p[:-1]))
    years = len(p) / 365
    return float(p.mean() * 100), entries / years


def _compute() -> dict:
    t0 = time.time()
    rate, where, problems = _funding()
    sig = _signal(rate)
    tns = rate.index.values.astype("datetime64[ns]").astype(np.int64)
    hot, absf = sig["hot"].values, sig["abs"].values
    close = te._daily_close()
    other, more = msc._load_other()
    problems += more
    ix_d = _idx(tns, close.index + DAY)
    hot_d = pd.Series(_take(hot, ix_d), index=close.index)
    F_d = pd.Series(_take(sig["F"].values, ix_d), index=close.index)
    first = hot_d.first_valid_index()
    if first is None:
        raise RuntimeError("資金費率不夠長，算不出「偏熱」")

    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    ex = _Exam({"close": close, "other": other}, price, b0, pre, tns, first + DAY)

    rng = np.random.default_rng(SEED + 5)
    ev = _events(close, hot_d, rng)
    s = {"ma50": ex.score("ma50", None), "w1": ex.score("w1", hot), "w30": ex.score("w30", absf)}
    small = {"ma50": ex.score("ma50", None, PL_SIMS), "w1": ex.score("w1", hot, PL_SIMS)}

    defined = np.flatnonzero(np.isfinite(hot))
    lo, hi = defined[0], defined[-1] + 1
    seg = hot[lo:hi]
    n = len(seg)
    prng = np.random.default_rng(SEED + 41)
    hits = beat_ma = 0
    lows = []
    for sh in prng.integers(GAP_PRINTS, n - GAP_PRINTS + 1, DRAWS):
        h2 = hot.copy()
        h2[lo:hi] = np.roll(seg, int(sh))
        x = ex.score("w1", h2, PL_SIMS)
        lows.append(x["low"])
        hits += int(bc._score_ge(x, small["w1"]))
        beat_ma += int(bc._score_ge(x, small["ma50"]))

    pos = {"ma50": ex.pos_btc("ma50", None), "w1": ex.pos_btc("w1", hot), "w30": ex.pos_btc("w30", absf)}
    f0 = first + DAY
    years = {"hold": bc._years(np.ones(len(close)), close, _d(f0)), **{k: bc._years(pos[k], close, _d(f0)) for k in pos}}
    a = close.index >= f0
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["w1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()))
    sw = {k: _switches(pos[k], close.index, f0) for k in pos}
    return {"rate": rate, "sig": sig, "where": where, "problems": problems, "close": close, "hot_d": hot_d,
            "F_d": F_d, "first": first, "ex": ex, "ev": ev, "s": s, "small": small, "plc": hits / DRAWS,
            "plc_low": float(np.median(lows)), "plc_ma": beat_ma / DRAWS, "pos": pos, "years": years,
            "differ": differ, "sw": sw, "ll": _leadlag(close, F_d), "run": _runup(close, hot_d),
            "look": _lookahead(close, rate), "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    r = _compute()
    close, ex, s, ev, sig, rate = r["close"], r["ex"], r["s"], r["ev"], r["sig"], r["rate"]
    base, w1, w30 = s["ma50"], s["w1"], s["w30"]
    first = r["first"]

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    q = {"score": bc._score_ge(w1, base), "loss": w1["c_deep_med"] >= base["c_deep_med"] - 1e-12,
         "plc": r["plc"] <= 0.10, "event": ev["ok"]}
    q["pass"] = all(q.values())

    # ----- table 1: event study -----
    rows1 = []
    for lab, p in ev["periods"].items():
        if p["n"] == 0:
            rows1.append([lab, "0 段" + ("（這段沒有資金費率資料）" if lab == PERIODS[0][0] else ""), "—", "—", "—", "—", "—",
                          f"{_p(p['all_r30'])}｜{_p(p['all_deep'])}"])
            continue
        rows1.append([lab, f"{p['n']} 段",
                      f"{_p(p['ep_r7'])}（{_p(p['all_r7'])}）", f"{_p(p['ep_r14'])}（{_p(p['all_r14'])}）",
                      f"{_p(p['ep_r30'])}（{_p(p['all_r30'])}）；中位數 {_p(p['epmed_r30'])}",
                      f"{_p(p['ep_deep'])}（{_p(p['all_deep'])}）；中位數 {_p(p['epmed_deep'])}",
                      f"30 天報酬 {_share(p['rand_r30'])}｜最深跌幅 {_share(p['rand_deep'])}",
                      f"{_p(p['all_r30'])}｜{_p(p['all_deep'])}"])
    t1 = {"caption": ("事件研究：資金費率 7 天平均（年化）從「不熱」變成「偏熱」（高於自己過去一年的第 90 百分位）的那一天，之後比特幣怎麼走。"
                      f"兩段之間至少隔 {GAP_EP} 天。格子是這些段的平均，括號是同一段期間所有日子的平均。"
                      f"「隨機抽樣」＝從同期所有日子隨機抽一樣多天、抽 {RAND:,} 次，平均一樣差或更差的比例（越小越不像巧合）。"
                      "最深跌幅＝之後 30 天內最低收盤比當天收盤低多少（以收盤價計）。"),
          "columns": ["期間", "偏熱段數", "7 天後（全部日子）", "14 天後（全部日子）", "30 天後（全部日子）",
                      "30 天內最深跌幅（全部日子）", "隨機抽樣一樣差的比例", "全部日子：30 天｜最深跌幅"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    rows2 = []
    for lab, x in (("50 日均線（基準，同樣的起算點）", base), ("W1：50 日均線而且資金費率不偏熱（主要）", w1),
                   ("50 日均線而且 7 天平均費率不高於年化 30%（穩健度，只報告）", w30)):
        ck = x["checks"]
        rows2.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(ex.chold[1])))}｜—", "—"])
    o_txt = "、".join(f"{a} {len(o['starts'])}/{ex.o_all[a]}" for a, o in ex.other.items())
    t2 = {"caption": (f"main_signal_check 的公平考試，但只算資金費率「偏熱」訊號已經存在之後的起算點（{_d(first + DAY)} 起）。"
                      f"比特幣日線起算月剩 {base['d_n']} 個（原本 {ex.d_all} 個）；收盤時間格剩 {len(ex.cstarts)} 個起算月"
                      f"（原本 {ex.c_all} 個）× 24 ＝ {base['c_n']:,} 格；其他資產剩下的起算月：{o_txt}。"
                      "50 日均線也只在這些起算點上重新計分。四項都是「報酬贏過同一天起一直抱著」的比例；"
                      f"一般人做法＝收盤後 10 小時才做、成本 {P_COST * 100:.2f}%、一成進場漏掉、每格 {SIMS} 次。"
                      "其他幣、MSTR 用自己的 50 日均線、同一個比特幣資金費率訊號。最後兩欄是照課本｜一般人做法的中位數。"),
          "columns": ["規則", f"比特幣起算月（{base['d_n']} 個）", f"起算月×收盤時間（{base['c_n']:,} 格）", "照一般人方式做",
                      "其他資產起算月", "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: pass test -----
    tested = [f"{lab} {_p(p['ep_r30'])} 對 {_p(p['all_r30'])}" for lab, p in ev["periods"].items() if p["n"] >= MIN_EP]
    rows3 = [["排名分數不低於 50 日均線（同樣的起算點）", f"{_share(w1['low'])} 對 {_share(base['low'])}", yes(q["score"])],
             [f"{base['c_n']:,} 格最深虧損中位數不比均線深", f"{_pct(w1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(q["loss"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['small']['w1']['low'])}；"
              f"打亂後不輸均線的 {_share(r['plc_ma'])}）", yes(q["plc"])],
             [f"有 {MIN_EP} 段以上的期間，偏熱後 30 天平均都比平常差",
              "；".join(tested) if tested else f"沒有任何期間有 {MIN_EP} 段", yes(q["event"])],
             ["結果", "—", "通過" if q["pass"] else "沒有通過"]]
    t3 = {"caption": ("事先寫好的過關標準，四項都要過。隨機錯位：把每次結算的「偏熱」旗子整串循環平移至少 "
                      f"{GAP_PRINTS} 次結算（180 天）、{DRAWS} 次，整套考試重做；為了在時間內算完，這一步一般人做法每個收盤時間只模擬 "
                      f"{PL_SIMS} 次，真的 W1 也用同樣方式算一次再比。"),
          "columns": ["條件", "數字", "結果"], "rows": rows3}

    # ----- table 4: mechanism, lead / lag -----
    ll = r["ll"]
    rows4 = []
    for k in (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30):
        c = ll["corr"][k]
        who = (f"資金費率在前，{k} 天後結束的 7 天報酬" if k > 0 else
               f"7 天報酬在前（{-k} 天前結束），之後的資金費率" if k < 0 else "同一天結束的 7 天報酬")
        rows4.append([f"{k:+d}" if k else "0", who, f"{c:+.3f}" + ("＊" if abs(c) > ll["band"] else "")])
    best_pos = max((k for k in LL if k > 0), key=lambda k: abs(ll["corr"][k]))
    best_neg = max((k for k in LL if k < 0), key=lambda k: abs(ll["corr"][k]))
    rows4.append(["—", "k 為正裡最強的一格", f"k = {best_pos}：{ll['corr'][best_pos]:+.3f}"])
    rows4.append(["—", "k 為負裡最強的一格", f"k = {best_neg}：{ll['corr'][best_neg]:+.3f}"])
    t4 = {"caption": (f"誰先動：每天收盤時的資金費率 7 天平均，和比特幣 7 天報酬（k 天後結束）的相關係數，{_d(ll['first'])} 到 {_d(ll['last'])}"
                      f"（{ll['n']:,} 天）。k 為正＝費率先、報酬後（費率能不能預告）；k 為負＝報酬先、費率後（費率是不是跟著漲跌）。"
                      f"隨機範圍：把費率整串錯開至少 {SHIFT_DAYS} 天、{DRAWS} 次，所有 k 的相關絕對值的第 95 百分位 ±{ll['band']:.3f}；＊＝超過。"
                      "表裡只列部分 k；-30 到 +30 都有算。k 是 1 到 6 時，那段 7 天報酬還包含費率讀數之前的幾天，"
                      "不算真正的「費率在前」，要看 k ≥ 7。"),
          "columns": ["k（天）", "意思", "相關"], "rows": rows4}

    # ----- table 5: mechanism, run-up -----
    ru = r["run"]
    rows5 = [["「偏熱」和過去 30 天漲幅的相關", f"{ru['corr']:+.2f}", "越接近 1，偏熱越只是「最近漲很多」"],
             ["偏熱日子過去 30 天平均漲幅（全部日子）", f"{_p(ru['past_hot'])}（{_p(ru['past_all'])}）", "—"],
             ["偏熱日子落在漲幅最高兩成的比例", _share(ru["top2"]), "漲幅按全部日子分成十等分"],
             ["之後 30 天報酬：偏熱減不偏熱（直接比）", _pp(ru["raw_r"] * 100), "—"],
             ["之後 30 天報酬：偏熱減「漲幅一樣、但不偏熱」", _pp(ru["matched_r"] * 100),
              f"錯位 {DRAWS} 次裡一樣低或更低的比例 {_share(ru['p_r'])}"],
             ["之後 30 天最深跌幅：偏熱減「漲幅一樣、但不偏熱」", _pp(ru["matched_d"] * 100),
              f"錯位 {DRAWS} 次裡一樣深或更深的比例 {_share(ru['p_d'])}"]]
    t5 = {"caption": (f"偏熱是不是只是「價格最近漲很多」：{ru['days']:,} 天裡偏熱 {ru['hot_days']:,} 天。把日子按過去 30 天漲幅分成十等分，"
                      "在同一等分裡比偏熱和不偏熱的日子之後 30 天的表現，再依偏熱日子的分布加權。錯位＝把偏熱旗子整串平移至少 "
                      f"{SHIFT_DAYS} 天重算，看真的差距是不是比隨便亂放還特別。相鄰日子的 30 天視窗大量重疊，天數不是獨立的例子。"),
          "columns": ["項目", "數字", "說明"], "rows": rows5}

    # ----- table 6: years, time in market -----
    yrs = sorted(r["years"]["ma50"])
    rows6 = []
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["w1"].get(y)
        rows6.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    dd_, nn = r["differ"]
    sw = r["sw"]
    rows6.append(["在場時間｜一年進場次數", "100%｜—", f"{sw['ma50'][0]:.0f}%｜{sw['ma50'][1]:.1f} 次",
                  f"{sw['w1'][0]:.0f}%｜{sw['w1'][1]:.1f} 次", f"穩健度那條：{sw['w30'][0]:.0f}%｜{sw['w30'][1]:.1f} 次"])
    rows6.append(["和 50 日均線持有狀態不同的天數", "—", "—", f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—", "—"])
    t6 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，從 {_d(first + DAY)}（偏熱訊號開始有值）起，"
                      "看 W1 和 50 日均線的差來自哪幾年。W1 只會比均線少持有（偏熱時均線持有也改成現金）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "W1", "W1 減均線（百分點）"], "rows": rows6}

    # ----- today -----
    last = close.index[-1]
    F_now = r["F_d"].iloc[-1]
    k = _idx(rate.index.values.astype("datetime64[ns]").astype(np.int64), [last + DAY])[0]
    thr_now = sig["thr"].values[k] if k >= 0 else np.nan
    hot_now = r["hot_d"].iloc[-1]
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": "資金費率 7 天平均（年化，幣安）",
         "value": (f"{_p(F_now)}；過去一年的第 90 百分位 {_p(thr_now)}；"
                   f"{'偏熱' if hot_now == 1 else '不偏熱' if hot_now == 0 else '—'}（用 {_d(rate.index[k])} UTC 那次結算以前的 21 次）")},
        {"label": "比特幣：50 日均線｜W1", "value": f"{word(pm['ma50'][-1])}｜{word(pm['w1'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    better = [y for y in yrs if _ok(r["years"]["w1"].get(y)) and _ok(r["years"]["ma50"].get(y))
              and r["years"]["w1"][y] - r["years"]["ma50"][y] > 0.5]
    worse = [y for y in yrs if _ok(r["years"]["w1"].get(y)) and _ok(r["years"]["ma50"].get(y))
             and r["years"]["w1"][y] - r["years"]["ma50"][y] < -0.5]
    eps_txt = "、".join(f"{lab} {p['n']} 段" for lab, p in ev["periods"].items())
    # k = 1..6: the 7-day return still overlaps days before the funding reading, so only k >= 7 is a clean lead
    lead_pos = [k for k in LL if k >= 7 and abs(ll["corr"][k]) > ll["band"]]
    lead_neg = [k for k in LL if k < 0 and abs(ll["corr"][k]) > ll["band"]]
    if lead_neg and not lead_pos:
        mech = ("費率主要是跟在上漲後面（報酬先、費率後，相關最高 "
                f"{ll['corr'][best_neg]:+.2f}）；費率在前、7 天以上之後的報酬都在隨機範圍內，看不到費率先預告下跌。")
    elif lead_pos and not lead_neg:
        mech = "有費率先動、之後報酬才變的跡象。"
    elif lead_pos and lead_neg:
        mech = "前後兩個方向都有超過隨機範圍的格子。"
    else:
        mech = "前後都沒有超過隨機範圍。"
    verdict = (f"{'通過' if q['pass'] else '沒有通過'}事先寫好的標準。原本想用的 BitMEX 2016 年起的資金費率被網站擋下（403），"
               f"所以只能用幣安 {_d(rate.index[0])} 起的資料，「偏熱」從 {_d(first + DAY)} 才有值，2017–18 那一輪完全沒有測到，"
               f"歷史還是很短。偏熱段數：{eps_txt}。W1（50 日均線而且資金費率不偏熱）："
               f"排名分數 {_share(w1['low'])} 對均線 {_share(base['low'])}（{yes(q['score'])}）；最深虧損中位數 "
               f"{_pct(w1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}（{yes(q['loss'])}）；隨機錯位一樣好 {_share(r['plc'])}"
               f"（{yes(q['plc'])}）；偏熱後 30 天比平常差（{yes(q['event'])}）。"
               f"穩健度（年化 30%）排名分數 {_share(w30['low'])}。"
               + (f"比均線好的年份：{'、'.join(str(y) for y in better)}" if better else "沒有一年明顯比均線好")
               + (f"；比均線差的年份：{'、'.join(str(y) for y in worse)}。" if worse else "。")
               + f"機制：偏熱和過去 30 天漲幅的相關 {ru['corr']:+.2f}；和漲幅一樣但不偏熱的日子比，之後 30 天報酬差 "
               f"{_pp(ru['matched_r'] * 100)} 點。" + mech)

    tot, bad = r["look"]
    notes = [
        ("資金費率是什麼：永續合約多方每 8 小時付給空方的費用（負的時候反過來），越高代表做多的人越擠。"
         "這裡把最近 21 次（7 天）的平均乘上一年 1,095 次，換成「一年要付多少」。只持有現貨、不碰合約的人不用付這筆錢，"
         "這裡只是拿它當「市場有多擠」的溫度計。"),
        ("BIS 論文說了什麼（2026-11-19 讀的）：BIS 第 1087 號工作論文《Crypto carry》（Schmeling、Schrimpf、Todorov，2023 年 4 月）"
         "的頁面寫「a high crypto carry predicts future price crashes」。論文用的是有到期日的期貨基差（1 個月），資料 2019-03 到 2022-01，"
         "第 8 表：基差高，之後 1 到 3 個月的選擇權偏斜更負、實際報酬偏斜更負（只有 OKEx 顯著）、現貨報酬更低（3 個月最明顯）；"
         "第 6 表：基差上升 10%，下個月被強制平倉的「空單」增加相當於未平倉量的 44%，但「多單」的強制平倉沒有被預測到。"
         "論文沒有用永續合約資金費率，也沒有給交易規則，這裡的 W1 是這份研究自己事先定的。"),
        ("資料怎麼拿的：原本計畫用 BitMEX XBTUSD 永續合約 2016 年起的資金費率（公開 API，不用金鑰）。"
         "2026-11-19 第一次用固定的 user agent「btc-research-dashboard」去要，就被 CloudFront 回「403 Request blocked」。"
         "照事先的約定，被擋就停：沒有重試、沒有換網址、沒有換 user agent、沒有繞過，BitMEX 完全沒有用到，"
         "所以也算不出 BitMEX 和幣安的相關係數（原本要拿來互相核對）。"
         f"改用專案原本 carry.py 的讀取方式：{r['where']}，{_d(rate.index[0])} 到 {_d(rate.index[-1])}，共 {len(rate):,} 次結算；"
         "下載成功時存一份在 dashboard/cache/_store/funding_warning_binance.json，下次抓不到時用。"),
        ("和以前幾份研究的差別：funding_positioning 只描述幣安資金費率各區間之後的走勢，沒有規則、沒有考試；"
         "basis_filter 用到期期貨的基差去調整 17 條規則混合；carry 研究的是收資金費率的套利本身。"
         "這份把「資金費率偏熱」當成每天的閘門放在 50 日均線上，用完整的公平考試測。原本新的地方是更長的歷史，但這次拿不到。"),
        ("事先寫死的規則和標準（照原文，跑之前寫好）：F = mean of the latest 21 settlements x 1095; hot = F above the 90th "
         "percentile of F at all settlements in the trailing 365 days (expanding, needs 180 days); decided at each daily "
         "close with settlements strictly before that close. W1 = ma50 AND not hot. Robustness: F > 30% a year. Pass test: "
         "ranking score >= ma50 on the same starts AND median deepest loss no worse AND placebo (hot flag circularly shifted "
         ">= 180 days, 200 draws) <= 10% AND, in every period with >= 5 episodes, mean 30-day return after turning hot "
         "below the all-days mean."),
        (f"偷看未來的檢查：在三個日期之後把價格和那時還沒結算的資金費率換成亂數，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。") +
         "每天的決定只用收盤時間「之前」已經結算的費率（剛好在收盤那一刻結算的那次不算）。"),
        ("限制：偏熱的段數很少，而且全部在 2020 年以後；相鄰起算月、相鄰收盤時間大部分歷史相同，「幾個裡幾個」不是互相獨立的證據。"
         "幣安在市場平靜時費率常常固定在每 8 小時 0.01%（年化約 11%），這時第 90 百分位可能剛好等於這個值，必須「高於」才算偏熱。"
         f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的一條規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + r["problems"]

    return {
        "id": "funding_warning",
        "title": "資金費率偏熱能不能幫 50 日均線躲崩盤",
        "question": ("國際清算銀行（BIS）的研究說：期貨比現貨貴很多（做多的人很擠）的時候，之後容易崩跌。"
                     "只買現貨、不碰合約的人，如果在永續合約資金費率「比自己過去一年都高」的時候先不持有比特幣，"
                     "能不能讓 50 日均線少虧、通過完整的公平考試？費率高是在下跌之前出現，還是只是跟著上漲一起出現？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5, t6],
        "notes": notes,
        "sources": [
            {"name": "BIS 第 1087 號工作論文《Crypto carry》頁面", "url": BIS_PAGE},
            {"name": "BIS 第 1087 號工作論文 PDF", "url": BIS_PDF},
            {"name": "BitMEX XBTUSD 資金費率 API（試過，被 403 擋下，沒有使用）", "url": BITMEX_URL},
            {"name": "幣安 BTCUSDT 永續合約資金費率歷史 API", "url": BINANCE_URL},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
