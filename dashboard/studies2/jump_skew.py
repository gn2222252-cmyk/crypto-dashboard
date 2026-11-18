"""Historical study "jump_skew": when bitcoin "jumped" a lot inside last week (sudden large moves within minutes), or
when last week's small moves were lopsided (realized skewness), is next week weaker - and does sitting out such
weeks on top of the 50-day rule (ma50, the page's main signal) help?

What the papers find (read before writing this design):
- Lee & Wang, "Variance Decomposition and Cryptocurrency Return Prediction", JFQA 60(4) 2025, 1859-1890 (the brief
  called it "Jumps and cryptocurrency returns"): CROSS-SECTION of 100 Coinbase coins, Oct 2015 - Jun 2023, 15-minute
  mid-quote log returns. Each week coins are sorted into terciles on variances estimated from the previous month
  (weeks w-3..w); coins with high total variance earn lower returns the next week (high-minus-low -3.7% a week,
  equal weight). Split by sign of jumps, the effect comes from positive-jump variance (H-L -3.6% a week EW, -2.3% VW)
  and jump-robust variance. Their positive jump variance is NOT RV minus bipower variation: it is
      JV+_w = sum over 15-minute returns r_j in the window of r_j^2 * 1(|T_j| > zeta) * 1(r_j > 0),
  with T_j the Lee-Mykland (2008) statistic (return over local volatility from bipower variation of the previous
  K-1 returns, K = 156 at 15 minutes), zeta the Gumbel rejection level at 5%, after removing the time-of-day
  volatility pattern. This study follows that definition (details below). It is a cross-sectional result; the
  paper has no bitcoin-only time-series test.
- Amaya, Christoffersen, Jacobs & Vasquez, JFE 118 (2015) 135-167: US stocks, five-minute returns. Daily
  RDSkew = sqrt(N) * sum r^3 / RDVar^(3/2), RDKurt = N * sum r^4 / RDVar^2, RDVar = sum r^2 (N returns a day); the
  weekly measure is the average of the daily ones. Lowest-minus-highest skewness decile earns about 19 basis points
  the next week (t = 3.7); kurtosis weaker, volatility none. Cross-sectional.
- Ahmed & Al Mafrachi, IREF 72 (2021) 483-499: intraday BTC, ETH, XRP; quantile regressions; moments up to the
  fifth order matter, skewness and hyper-skewness predict returns in and out of sample. The public abstract does not
  give the sign or the horizon.
- Atance & Serna, QREF 96 (2024): GARCH-type CONDITIONAL skewness (not realized skewness) predicts bitcoin returns;
  the relation is negative in 2018-2020 and positive in 2021-2022, i.e. the sign flips by subperiod.

Pre-registered design (written before any result was looked at; not changed afterwards):
Data: Binance spot 5-minute klines (data.binance.vision monthly / daily files) for BTC, ETH, SOL, BNB, XRP from
2017-08. Weeks run Monday 00:00 UTC to the next Monday 00:00 UTC.
Weekly statistics:
- RV = sum of squared 5-minute log returns in the week.
- RSK (Amaya): daily RDSkew from the 288 five-minute returns of each UTC day, weekly RSK = average of the week's
  daily values (days with at least 144 returns; a week needs at least 5 such days). RKT likewise (reference only).
- PJV (Lee & Wang): 15-minute log returns (last 5-minute close of each quarter hour). Time-of-day pattern removed
  with each quarter hour's mean absolute return over the previous 90 days (past only, at least 20 days), divided by
  the average over all quarter hours. Lee-Mykland statistic L_i = r_i / sigma_i with sigma_i^2 = mean of
  |r_j||r_(j-1)| over the K-2 = 154 products ending at i-1 (K = 156; at least 100 valid). A jump when
  (|L_i| - C_n)/S_n > -log(-log(0.95)) with c = sqrt(2/pi), n = 672 (quarter hours in a week),
  C_n = sqrt(2 log n)/c - (log pi + log log n)/(2c sqrt(2 log n)), S_n = 1/(c sqrt(2 log n)). PJV = sum of the
  squared raw returns of the positive jumps in the week. (Reference: PJV over four weeks, the paper's window.)
- Point in time: a week's statistic is compared with the previous up-to-104 weeks only (at least 52), mid-rank
  percentile p = (number below + half the number equal) / count; quintile = min(5, floor(5p) + 1); "top 20%" = p >= 0.8.
Sign rule fixed in advance from the literature: high RSK -> lower next-week return (Amaya); high PJV -> lower
next-week return (Lee & Wang).
1. Next-week return (Monday 00:00 close to the next Monday 00:00 close) by quintile of RSK and of PJV; slope of an
   ordinary least-squares line of next-week return (%) on the quintile number, 90% interval from a circular block
   bootstrap (blocks of 8 weeks, 2,000 draws, fixed seed); periods 2017-08..2020, 2021..2023, 2024..now (by the
   statistic's week).
2. Rule J1: ma50, but cash for the whole next week when last week's PJV is in its trailing top 20%. Rule S1: same with
   RSK. The weekly flag is known at Monday 00:00 UTC and is used by every decision close from then until the next
   Monday 00:00 (a decision close at time T uses the week that ended at the last Monday 00:00 on or before T).
3. main_signal_check's fair grid with its own code pieces: bitcoin monthly start dates on the dashboard's daily
   closes (textbook, 0.10%), monthly start dates x 24 closing hours (textbook), the same cells ordinary-person
   (decide at the clock, trade 10 hours later, 0.25%, 10% of entries missed, 200 simulations per clock, the seed
   main_signal_check uses for ma50, so ma50, J1 and S1 share their random draws), and ETH/SOL/BNB/XRP textbook
   with their own weekly statistics (no MSTR: it has no 5-minute data). Start dates only where every rule has a full
   104-week history (bitcoin from 2019, up to 2023-12; other coins up to three years before today). Ranking score
   = the lowest of the four "beats holding" shares (main_signal_check's rule). Placebo: the weekly statistic is
   circularly shifted by a random 52..(n-52) weeks (each coin its own shift), flags and ranking score recomputed,
   200 draws; for speed the placebo and the rule it is compared with both use 20 ordinary-person simulations per
   clock instead of 200. Bear markets of 50%+ since 2017-08 (crash_timing's definition): share of the crash avoided
   and whipsaws.
Pass test (stated now): the predicted sign holds in all three periods and the full-sample 90% interval of the slope
excludes zero, AND the rule's ranking score >= ma50's with a median deepest loss (start x clock cells) no worse,
AND the placebo share (placebo score >= the rule's) <= 10%, AND the slope has the predicted sign on >= 3 of the 4
other coins.
Today: last finished week's RSK, PJV, their percentiles and the rule states - states only, no advice.

Public key-less files only (data.binance.vision), honest user agent. The 5-minute data are reduced to per-day sums
and 15-minute closes in cache/_store/jump_skew_<SYMBOL>.npz; only new days are downloaded on later runs.
"""
from __future__ import annotations

import io
import json
import math
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
OTHER = ("ETH", "SOL", "BNB", "XRP")
URL_M = "https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
URL_D = "https://data.binance.vision/data/spot/daily/klines/{s}/5m/{s}-5m-{d}.zip"
FIRST_MONTH = "2017-08"
FIVE_MS, Q_MS, DAY_MS = 300_000, 900_000, 86_400_000
WEEK_MS = 7 * DAY_MS
SLOTS = 96

MIN_DAY_RETURNS, MIN_DAYS = 144, 5
PERIOD_DAYS, PERIOD_MIN = 90, 20
K = 156
LM_MIN = 100
LM_N = 672
ALPHA = 0.05
WIN, WIN_MIN = 104, 52
TOP = 0.8
BLOCK, BOOT = 8, 2000
BOOT_SEED = 20170814
PLACEBO, PL_SIMS, PL_SEED = 200, 20, 20171201
PERIODS = (("2017-08–2020", None, "2021-01-01"), ("2021–2023", "2021-01-01", "2024-01-01"), ("2024–今", "2024-01-01", None))
STATS = ("RSK", "PJV")
RULES = ("ma50", "J1", "S1")
RULE_STAT = {"J1": "PJV", "S1": "RSK"}
NAMES = {"ma50": "50 日均線（現行主訊號）", "J1": "J1：50 日均線＋上週往上跳動前 20% 就空手一週",
         "S1": "S1：50 日均線＋上週偏態前 20% 就空手一週"}
SHORT = {"ma50": "50 日均線", "J1": "J1（跳動）", "S1": "S1（偏態）"}
STAT_NAME = {"RSK": "偏態（RSK）", "PJV": "往上跳動（PJV）", "RV": "總波動（RV）", "RKT": "峰態（RKT）",
             "PJV4": "往上跳動 4 週窗（PJV，論文的窗）"}
MSC_KI = 1  # main_signal_check's random draws for ma50 (CANDS index 1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def _num(v, digits: int = 2) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{digits}f}"


# ---------- data: 5-minute klines -> per-day sums and 15-minute closes ----------

def _path(symbol: str):
    return common.STORE / f"jump_skew_{symbol}.npz"


def _load_store(symbol: str) -> dict:
    """days: (n, 6) rows [day ms, returns, sum r^2, sum r^3, sum r^4, last close]; q0 + c15: quarter-hour closes."""
    try:
        with np.load(_path(symbol), allow_pickle=False) as z:
            return {"days": z["days"].astype(float), "q0": int(z["q0"]), "c15": z["c15"].astype(float),
                    "last": z["last"].astype(float)}
    except Exception:  # noqa: BLE001 - no store yet, or a damaged one: start over
        return {"days": np.zeros((0, 6)), "q0": 0, "c15": np.zeros(0), "last": np.array([-1.0, np.nan])}


def _save_store(symbol: str, st: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, days=st["days"], q0=np.int64(st["q0"]), c15=st["c15"], last=st["last"])
        _path(symbol).write_bytes(buf.getvalue())
    except Exception:  # noqa: BLE001 - without the store the next run simply downloads again
        pass


def _get_zip(session: requests.Session, url: str) -> np.ndarray | None:
    """(open ms, close) rows of one kline zip; None when the file is not there (404)."""
    reply = session.get(url, headers=common.HEADERS, timeout=60)
    if reply.status_code == 404:
        return None
    reply.raise_for_status()
    time.sleep(0.1)
    with zipfile.ZipFile(io.BytesIO(reply.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    rows = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) > 5 and parts[0].strip().isdigit():
            rows.append((float(parts[0]), float(parts[4])))
    if not rows:
        return np.zeros((0, 2))
    a = np.array(rows, dtype=float)
    a[:, 0] = np.where(a[:, 0] > 1e14, np.floor(a[:, 0] / 1000), a[:, 0])  # 2025+ files use microseconds
    return a[np.argsort(a[:, 0], kind="stable")]


def _ingest(st: dict, rows: np.ndarray) -> None:
    """Add whole UTC days of 5-minute candles (in time order) to the store."""
    rows = rows[(rows[:, 0] > st["last"][0]) & (rows[:, 1] > 0)]
    if not len(rows):
        return
    t, c = rows[:, 0].astype(np.int64), rows[:, 1]
    prev = np.concatenate([[st["last"][1]], c[:-1]])
    r = np.log(c / prev)  # a missing candle: the next return covers the gap (Amaya: no price -> return zero)
    day = t // DAY_MS
    starts = np.flatnonzero(np.diff(day, prepend=day[0] - 1))
    ends = np.append(starts[1:], len(t)) - 1
    ok = np.isfinite(r)
    rr = np.where(ok, r, 0.0)
    n = np.add.reduceat(ok.astype(float), starts)
    sums = [np.add.reduceat(rr ** p, starts) for p in (2, 3, 4)]
    block = np.column_stack([day[starts].astype(float) * DAY_MS, n, *sums, c[ends]])
    st["days"] = np.vstack([st["days"], block])
    tq = t + FIVE_MS
    on = tq % Q_MS == 0
    if not st["c15"].size:
        st["q0"] = int(day[0] * DAY_MS)
    idx = (tq[on] - st["q0"]) // Q_MS
    need = int(idx.max()) + 1 if len(idx) else 0
    if need > st["c15"].size:
        st["c15"] = np.concatenate([st["c15"], np.full(need - st["c15"].size, np.nan)])
    st["c15"][idx] = c[on]
    st["last"] = np.array([float(t[-1]), float(c[-1])])


def _update(symbol: str) -> tuple[dict, str | None, int]:
    """Bring one coin's store up to yesterday (whole UTC days only). Returns (store, problem text, files fetched).

    A month that is fully in the past comes as one monthly file; otherwise day files. A day file that is not there
    yet stops the update (tried again next run); a day missing for more than three days is a gap and is skipped."""
    st = _load_store(symbol)
    today = pd.Timestamp(common.today_utc())
    files, problem = 0, None
    session = requests.Session()
    if st["last"][0] > 0:
        day = pd.Timestamp(int(st["last"][0]), unit="ms").normalize() + pd.Timedelta(days=1)
    else:
        day = pd.Timestamp(FIRST_MONTH + "-01")
    try:
        since_save = 0
        while day < today:
            month = day.to_period("M").to_timestamp()
            m_end = month + pd.offsets.MonthBegin(1)
            if day == month and m_end <= today:
                got = _get_zip(session, URL_M.format(s=symbol, m=month.strftime("%Y-%m")))
                files += 1
                if got is not None:
                    _ingest(st, got)
                    _save_store(symbol, st)
                    day = m_end
                    continue
                if not len(st["days"]):
                    day = m_end  # not listed yet
                    continue
            one = _get_zip(session, URL_D.format(s=symbol, d=day.strftime("%Y-%m-%d")))
            files += 1
            if one is None:
                if day >= today - pd.Timedelta(days=3) or not len(st["days"]):
                    break
            else:
                _ingest(st, one)
                since_save += 1
                if since_save >= 10:
                    _save_store(symbol, st)
                    since_save = 0
            day += pd.Timedelta(days=1)
        _save_store(symbol, st)
    except Exception as exc:  # noqa: BLE001 - a stored history is still usable
        problem = (f"{symbol[:-4]} 的幣安 5 分鐘線這次沒有更新完（{type(exc).__name__}），"
                   + ("用的是之前存下來的資料，" if len(st["days"]) else "") + "沒有改用其他方法繞過。")
        _save_store(symbol, st)
    return st, problem, files


# ---------- weekly statistics ----------

def _lm_positive(st: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per UTC day: sum of squared positive-jump returns, negative-jump returns, number of jumps (Lee-Mykland)."""
    c = st["c15"]
    if c.size < SLOTS * 30:
        raise RuntimeError("15 分鐘收盤太少")
    pad = (-c.size) % SLOTS
    c = np.concatenate([c, np.full(pad, np.nan)])
    r = np.diff(np.log(c), prepend=np.nan)
    mat = np.abs(r).reshape(-1, SLOTS)
    slot = pd.DataFrame(mat).rolling(PERIOD_DAYS, min_periods=PERIOD_MIN).mean().shift(1).values  # past days only
    level = np.nanmean(slot, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = slot / level
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0).ravel()
    ra = r / f
    bp = pd.Series(np.abs(ra) * np.abs(np.concatenate([[np.nan], ra[:-1]])))
    sig2 = bp.rolling(K - 2, min_periods=LM_MIN).mean().shift(1).values  # products ending at i-1
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = ra / np.sqrt(sig2)
    cc = math.sqrt(2 / math.pi)
    ln = math.log(LM_N)
    cn = math.sqrt(2 * ln) / cc - (math.log(math.pi) + math.log(ln)) / (2 * cc * math.sqrt(2 * ln))
    sn = 1 / (cc * math.sqrt(2 * ln))
    beta = -math.log(-math.log(1 - ALPHA))
    jump = np.isfinite(stat) & ((np.abs(stat) - cn) / sn > beta)
    r0 = np.nan_to_num(r)
    pos = np.where(jump & (r0 > 0), r0 ** 2, 0.0).reshape(-1, SLOTS)
    neg = np.where(jump & (r0 < 0), r0 ** 2, 0.0).reshape(-1, SLOTS)
    # return index i belongs to the quarter hour that ENDS at q0 + i*15min: shift by one so it sits in its day
    idx = pd.to_datetime(st["q0"] + np.arange(pos.shape[0]) * DAY_MS, unit="ms")
    flat_pos = np.concatenate([pos.ravel()[1:], [0.0]]).reshape(-1, SLOTS)
    flat_neg = np.concatenate([neg.ravel()[1:], [0.0]]).reshape(-1, SLOTS)
    flat_n = np.concatenate([jump.ravel()[1:], [False]]).reshape(-1, SLOTS)
    return (pd.Series(flat_pos.sum(axis=1), index=idx), pd.Series(flat_neg.sum(axis=1), index=idx),
            pd.Series(flat_n.sum(axis=1).astype(float), index=idx))


def _weekly(st: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Weekly statistics (index = the week's Monday) and the daily close (index = UTC day, close at its end)."""
    d = st["days"]
    if len(d) < 60:
        raise RuntimeError("5 分鐘線的天數太少")
    days = pd.DataFrame(d[:, 1:], columns=["n", "s2", "s3", "s4", "close"],
                        index=pd.to_datetime(d[:, 0].astype(np.int64), unit="ms"))
    days = days[~days.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(days.index[0], days.index[-1], freq="D")
    days = days.reindex(full)
    good = days["n"] >= MIN_DAY_RETURNS
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.sqrt(days["n"]) * days["s3"] / days["s2"] ** 1.5
        kurt = days["n"] * days["s4"] / days["s2"] ** 2
    days["skew"] = skew.where(good & (days["s2"] > 0))
    days["kurt"] = kurt.where(good & (days["s2"] > 0))
    pj, nj, cnt = _lm_positive(st)
    days["pjv"] = pj.reindex(days.index)
    days["njv"] = nj.reindex(days.index)
    days["jumps"] = cnt.reindex(days.index)
    monday = days.index - pd.to_timedelta(days.index.weekday, unit="D")
    g = days.groupby(monday)
    w = pd.DataFrame({"RV": g["s2"].sum(min_count=1), "RSK": g["skew"].mean(), "RKT": g["kurt"].mean(),
                      "PJV": g["pjv"].sum(min_count=1), "NJV": g["njv"].sum(min_count=1),
                      "jumps": g["jumps"].sum(min_count=1), "valid": g["skew"].count(), "ndays": g["n"].size()})
    w = w[w["ndays"] == 7]  # whole weeks only (the running week is left out)
    w = w.asfreq("7D")
    bad = w["valid"].fillna(0) < MIN_DAYS
    w.loc[bad, ["RV", "RSK", "RKT", "PJV", "NJV"]] = np.nan
    w["PJV4"] = w["PJV"].rolling(4, min_periods=4).sum()
    close = days["close"].ffill()
    sunday = w.index + pd.Timedelta(days=6)
    c_now = close.reindex(sunday).values
    c_next = close.reindex(sunday + pd.Timedelta(days=7)).values
    w["next"] = (c_next / c_now - 1) * 100  # next week's return in %, Monday 00:00 -> Monday 00:00
    return w, close


def _prank(x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value among the previous up-to-WIN values (at least WIN_MIN); NaN otherwise."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not math.isfinite(x[i]):
            continue
        past = x[max(0, i - WIN):i]
        past = past[np.isfinite(past)]
        if len(past) >= WIN_MIN:
            out[i] = (np.sum(past < x[i]) + 0.5 * np.sum(past == x[i])) / len(past)
    return out


def _quintile(p: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(p), np.minimum(5, np.floor(np.nan_to_num(p) * 5) + 1), np.nan)


def _full_window(p: pd.Series, x: pd.Series) -> pd.Timestamp | None:
    """First Monday whose statistic has a full WIN-week history behind it."""
    cnt = x.notna().astype(int).rolling(WIN, min_periods=1).sum().shift(1)
    ok = p.notna() & (cnt >= WIN)
    return ok.idxmax() if ok.any() else None


# ---------- test 1: next-week return by quintile ----------

def _slope(q: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(q) & np.isfinite(y)
    if ok.sum() < 20 or np.var(q[ok]) <= 0:
        return None
    return float(np.polyfit(q[ok], y[ok], 1)[0])


def _boot(q: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    ok = np.isfinite(q) & np.isfinite(y)
    q, y = q[ok], y[ok]
    n = len(q)
    if n < 3 * BLOCK:
        return None, None
    rng = np.random.default_rng(seed)
    nb = -(-n // BLOCK)
    out = []
    for _ in range(BOOT):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        qq, yy = q[idx], y[idx]
        if np.var(qq) > 0:
            out.append(np.polyfit(qq, yy, 1)[0])
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def _period_mask(index: pd.DatetimeIndex, a, b) -> np.ndarray:
    m = np.ones(len(index), dtype=bool)
    if a:
        m &= index >= pd.Timestamp(a)
    if b:
        m &= index < pd.Timestamp(b)
    return m


def _test1(w: pd.DataFrame, keys=("RSK", "PJV", "PJV4", "RV", "RKT"), boot: bool = True) -> dict:
    out = {}
    y = w["next"].values
    for k in keys:
        p = _prank(w[k].values)
        q = _quintile(p)
        x = {"p": p, "q": q, "slope": _slope(q, y), "n": int((np.isfinite(q) & np.isfinite(y)).sum()),
             "ci": _boot(q, y, BOOT_SEED + len(k)) if boot else (None, None), "per": [], "mean": {}}
        for _, a, b in PERIODS:
            m = _period_mask(w.index, a, b)
            x["per"].append((_slope(q[m], y[m]), int((np.isfinite(q[m]) & np.isfinite(y[m])).sum())))
        for j in range(1, 6):
            row = []
            for m in [np.ones(len(w), dtype=bool)] + [_period_mask(w.index, a, b) for _, a, b in PERIODS]:
                sel = m & (q == j) & np.isfinite(y)
                row.append((float(np.mean(y[sel])) if sel.any() else None, int(sel.sum())))
            x["mean"][j] = row
        out[k] = x
    return out


# ---------- flags on any decision time ----------

def _flag_by_monday(w: pd.DataFrame, p: np.ndarray) -> tuple[int, np.ndarray]:
    """(first Monday in ms, F) with F[k] = 1 when the week that ended at Monday k (= week k-1) was in the top 20%."""
    flag = np.where(np.isfinite(p) & (p >= TOP), 1.0, 0.0)
    w0 = int(w.index[0].value // 1_000_000)
    return w0, np.concatenate([[0.0], flag])


def _flags_at(t_ms: np.ndarray, w0: int, F: np.ndarray) -> np.ndarray:
    """Flag in force at decision times t (ms): the week that ended at the last Monday 00:00 on or before t."""
    t_ms = np.asarray(t_ms, dtype=np.int64)
    weekday = ((t_ms // DAY_MS) + 3) % 7  # 1970-01-01 was a Thursday
    monday = (t_ms // DAY_MS - weekday) * DAY_MS
    k = (monday - w0) // WEEK_MS
    k = np.clip(k, -1, len(F) - 1)
    return np.where(k >= 0, F[np.maximum(k, 0)], 0.0)


def _check_times(t_ms: np.ndarray, w0: int, F: np.ndarray) -> int:
    """How many decision times would use a week that had not ended yet (should be 0)."""
    t_ms = np.asarray(t_ms, dtype=np.int64)
    weekday = ((t_ms // DAY_MS) + 3) % 7
    monday = (t_ms // DAY_MS - weekday) * DAY_MS
    k = np.clip((monday - w0) // WEEK_MS, -1, len(F) - 1)
    week_end = w0 + k * WEEK_MS  # the week used is k-1, it ended at Monday k
    return int(((k >= 1) & (week_end > t_ms)).sum())


# ---------- the fair grid ----------

class _Grid:
    """main_signal_check's bitcoin checks (daily textbook, 24 clocks textbook, ordinary person) and other coins."""

    def __init__(self, close: pd.Series, price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, first, other: dict):
        self.close = close
        self.change = close.pct_change().fillna(0.0).values
        self.ma50_d = te._signals(close)[0]["A50"].values
        self.t_d = ((close.index + pd.Timedelta(days=1)).values.astype("datetime64[ms]").astype(np.int64))
        self.d_starts, self.d_labels = msc._month_starts(close.index, first, msc.DAILY_STARTS[1])
        self.d_hold = msc._suffix(msc._equity(self.change), self.d_starts, msc.YEAR)
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        days, npre = self.m.days, len(pre)
        self.c_starts, self.c_labels = msc._month_starts(self.m.dates, first, msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.c_starts, msc.YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        b0ms = int(b0.value // 1_000_000)
        self.ma50_h, self.t_h = [], []
        for h in range(24):
            closes = pd.Series(np.concatenate([pre, price[h::24]]))
            a50 = te._signals(closes)[0]["A50"].fillna(0.0).values[npre:][:days]
            self.ma50_h.append(a50)
            self.t_h.append(b0ms + (h + 24 * np.arange(days, dtype=np.int64)) * 3_600_000)
        self.other = other  # key -> dict(close, change, ma50, t, starts, hold)

    def run(self, flags: dict | None, sims: int, seed_base: int = msc.SEED) -> dict:
        """flags: None (ma50) or {"BTC": (w0, F), "ETH": (w0, F), ...}. Returns the four checks and loss numbers."""
        def over(key, t):
            if flags is None or key not in flags:
                return 1.0
            w0, F = flags[key]
            return 1.0 - _flags_at(t, w0, F)

        x = {}
        tgt = np.where(np.isnan(self.ma50_d), np.nan, self.ma50_d * over("BTC", self.t_d))
        net, _ = te._run(pd.Series(tgt).shift(1).values, self.change, msc.COST)
        r, dd = msc._suffix(msc._equity(net), self.d_starts, msc.YEAR)
        x["d_n"], x["d_beat"] = len(r), int((r[:, 0] > self.d_hold[0][:, 0]).sum())
        x["d_shallow"] = int((dd[:, 0] > self.d_hold[1][:, 0]).sum())
        ns, days = len(self.c_starts), self.m.days
        tr, td = np.zeros((ns, 24)), np.zeros((ns, 24))
        pr, pdd = np.zeros((ns, 24, sims)), np.zeros((ns, 24, sims))
        for h in range(24):
            w = self.ma50_h[h] * over("BTC", self.t_h[h])
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, msc.COST, None)]), self.c_starts, msc.YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(seed_base + 1000 * MSC_KI + h)
            pos = rt._missed(wl, msc.P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, msc.P_COST, None)]),
                                self.c_starts, msc.YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - self.hr[:, None], pr - self.hr[:, None, None]
        x.update(c_n=edge.size, c_beat=int((edge > 0).sum()), c_edge=float(np.median(edge)),
                 c_shallow=int((td > self.hd[:, None]).sum()), c_deep_med=float(np.median(td)),
                 c_deep_worst=float(td.min()), p_n=pedge.size, p_beat=int((pedge > 0).sum()),
                 p_edge=float(np.median(pedge)), p_deep_med=float(np.median(pdd)))
        x["o_each"] = {}
        for key, o in self.other.items():
            tgt = np.where(np.isnan(o["ma50"]), np.nan, o["ma50"] * over(key, o["t"]))
            net, _ = te._run(pd.Series(tgt).shift(1).values, o["change"], msc.COST)
            r, _ = msc._suffix(msc._equity(net), o["starts"], msc.YEAR)
            x["o_each"][key] = (len(r), int((r[:, 0] > o["hold"][:, 0]).sum()))
        x["o_n"] = sum(v[0] for v in x["o_each"].values())
        x["o_beat"] = sum(v[1] for v in x["o_each"].values())
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        return x


def _shifted(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The statistic circularly shifted by 52..(n-52) weeks inside its own valid stretch."""
    ok = np.flatnonzero(np.isfinite(x))
    out = x.copy()
    if len(ok) < 2 * WIN_MIN + 2:
        return out
    a, b = ok[0], ok[-1] + 1
    seg = x[a:b]
    out[a:b] = np.roll(seg, int(rng.integers(WIN_MIN, len(seg) - WIN_MIN)))
    return out


# ---------- look-ahead check ----------

def _lookahead(st: dict) -> tuple[int, int]:
    """(cut points, weekly rows or flags before the cut that changed) when all 5-minute-derived data after the cut
    are scrambled."""
    base, _ = _weekly(st)
    pb = {k: _prank(base[k].values) for k in STATS}
    rng = np.random.default_rng(11)
    bad = 0
    cuts = (len(st["days"]) - 700, len(st["days"]) - 300, len(st["days"]) - 40)
    for cut in cuts:
        alt = {"days": st["days"].copy(), "q0": st["q0"], "c15": st["c15"].copy(), "last": st["last"]}
        cut_ms = alt["days"][cut, 0]
        tail = alt["days"][cut:, 1:5]
        alt["days"][cut:, 1:5] = tail[rng.permutation(len(tail))] * rng.uniform(0.5, 2.0, (len(tail), 1))
        qi = int((cut_ms - st["q0"]) // Q_MS) + 1  # c15[qi - 1] closes the day before the cut
        alt["c15"][qi:] *= np.exp(rng.normal(0, 0.05, alt["c15"].size - qi))
        w2, _ = _weekly(alt)
        cut_ts = pd.Timestamp(int(cut_ms), unit="ms")
        before = base.index[base.index + pd.Timedelta(days=7) <= cut_ts]
        for k in STATS + ("RV", "RKT"):
            a = base.loc[before, k].values
            b = w2.reindex(before)[k].values
            if not np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9), rtol=1e-12, atol=0):
                bad += 1
        for k in STATS:
            p2 = _prank(w2[k].values)
            n = len(before)
            if not np.array_equal(np.nan_to_num(pb[k][:n], nan=-1), np.nan_to_num(p2[:n], nan=-1)):
                bad += 1
    return len(cuts), bad


# ---------- the whole computation ----------

def _month_ceil(ts: pd.Timestamp) -> pd.Timestamp:
    m = ts.to_period("M").to_timestamp()
    return m if m == ts.normalize() else m + pd.offsets.MonthBegin(1)


def _rule_flags(weekly: dict, stat: str, p: dict | None = None) -> dict:
    return {key: _flag_by_monday(w, (p or {}).get(key, _prank(w[stat].values))) for key, (w, _) in weekly.items()}


def _compute() -> dict:
    t_start = time.time()
    weekly, problems, files, stores = {}, [], 0, {}
    for symbol in SYMBOLS:
        st, problem, n = _update(symbol)
        files += n
        if problem:
            problems.append(problem)
        try:
            weekly[symbol[:-4]] = _weekly(st)
            stores[symbol[:-4]] = st
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            if symbol == "BTCUSDT":
                raise
            problems.append(f"{symbol[:-4]} 的週統計算不出來（{type(exc).__name__}），沒有算進去。")
    t_data = time.time() - t_start
    w_btc, c_btc = weekly["BTC"]
    t1 = _test1(w_btc)
    t1_other = {k: _test1(weekly[k][0], keys=STATS, boot=False) for k in OTHER if k in weekly}

    close = te._daily_close()
    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.insert(0, problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")

    def first_of(w: pd.DataFrame, pr: dict) -> pd.Timestamp | None:
        got = [_full_window(pd.Series(pr[k], index=w.index), w[k]) for k in STATS]
        if any(g is None for g in got):
            return None
        return _month_ceil(max(got) + pd.Timedelta(days=7))  # flag of week w is used from Monday w+1

    p_all = {key: {k: _prank(w[k].values) for k in STATS} for key, (w, _) in weekly.items()}
    first = first_of(w_btc, p_all["BTC"])
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key in OTHER:
        if key not in weekly:
            continue
        w, oc = weekly[key]
        oc = oc[oc.index < pd.Timestamp(common.today_utc())].dropna()
        f0 = first_of(w, p_all[key])
        if f0 is None or f0 >= three:
            problems.append(f"{key} 的週統計還不滿兩年就到三年前了，沒有換幣的起算日。")
            continue
        change = oc.pct_change().fillna(0.0).values
        starts, labels = msc._month_starts(oc.index, f0, three)
        if not len(starts):
            continue
        other[key] = {"close": oc, "change": change, "ma50": te._signals(oc)[0]["A50"].values,
                      "t": (oc.index + pd.Timedelta(days=1)).values.astype("datetime64[ms]").astype(np.int64),
                      "starts": starts, "labels": labels,
                      "hold": msc._suffix(msc._equity(change), starts, msc.YEAR)[0]}
    grid = _Grid(close, price, b0, pre, first, other)
    flags = {"ma50": None}
    for r_ in ("J1", "S1"):
        flags[r_] = {key: _flag_by_monday(weekly[key][0], p_all[key][RULE_STAT[r_]]) for key in weekly}
    res = {r_: grid.run(flags[r_], msc.SIMS) for r_ in RULES}
    t_grid = time.time() - t_start

    # placebo
    pl = {}
    rng = np.random.default_rng(PL_SEED)
    for r_ in ("J1", "S1"):
        stat = RULE_STAT[r_]
        actual = grid.run(flags[r_], PL_SIMS)["low"]
        scores = []
        for _ in range(PLACEBO):
            fl = {}
            for key, (w, _) in weekly.items():
                fl[key] = _flag_by_monday(w, _prank(_shifted(w[stat].values, rng)))
            scores.append(grid.run(fl, PL_SIMS)["low"])
        scores = np.array(scores)
        pl[r_] = {"actual": actual, "share": float(np.mean(scores >= actual - 1e-12)),
                  "p50": float(np.median(scores)), "p90": float(np.percentile(scores, 90))}
    t_placebo = time.time() - t_start

    # ma50 on main_signal_check's own grid (2018-01 .. 2023-12) must give its numbers
    import copy
    g2 = copy.copy(grid)
    g2.c_starts, _ = msc._month_starts(grid.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(grid.m.change), g2.c_starts, msc.YEAR)
    g2.hr, g2.hd = hr[:, 0], hd[:, 0]
    g2.other = {}
    rep = g2.run(None, msc.SIMS)

    # daily positions for crashes, whipsaws and today
    c = close.values.astype(float)
    pos = {}
    for r_ in RULES:
        over = 1.0 if flags[r_] is None else 1.0 - _flags_at(grid.t_d, *flags[r_]["BTC"])
        pos[r_] = np.where(np.isnan(grid.ma50_d), np.nan, grid.ma50_d * over)
    bears = [b for b in ct._bears(close) if close.index[b["top"]] >= pd.Timestamp("2017-08-01")]
    crash = {r_: [ct._crash(c, [(1.0, pos[r_])], b) for b in bears] for r_ in RULES}
    i_first = int(close.index.searchsorted(first))
    whip = {r_: ct._whipsaws(c, [(1.0, pos[r_])], i_first) for r_ in RULES}
    differ = {r_: float(np.mean(pos[r_][i_first:] != pos["ma50"][i_first:])) for r_ in RULES}
    flagged = {r_: float(np.mean(1.0 - _flags_at(grid.t_d[i_first:], *flags[r_]["BTC"]) < 0.5)) for r_ in ("J1", "S1")}

    # look-ahead checks
    look = _lookahead(stores["BTC"])
    early = 0
    for r_ in ("J1", "S1"):
        for key, (w0, F) in flags[r_].items():
            ts = grid.t_d if key == "BTC" else other[key]["t"] if key in other else None
            if ts is not None:
                early += _check_times(ts, w0, F)
        for h in range(24):
            early += _check_times(grid.t_h[h], *flags[r_]["BTC"])
    noisy = c.copy()
    cut = len(c) - 400
    noisy[cut + 1:] *= np.exp(np.random.default_rng(5).normal(0, 0.2, len(c) - cut - 1))
    a50 = te._signals(pd.Series(noisy, index=close.index))[0]["A50"].values
    alt = np.where(np.isnan(a50), np.nan, a50 * (1.0 - _flags_at(grid.t_d, *flags["J1"]["BTC"])))
    pos_change = int(not np.array_equal(np.nan_to_num(alt[:cut + 1], nan=-1), np.nan_to_num(pos["J1"][:cut + 1], nan=-1)))

    return {"weekly": weekly, "t1": t1, "t1_other": t1_other, "close": close, "grid": grid, "res": res, "pl": pl,
            "rep": rep, "pos": pos, "bears": bears, "crash": crash, "whip": whip, "differ": differ,
            "flagged": flagged, "flags": flags, "p_all": p_all, "first": first, "other": other,
            "look": look, "early": early, "pos_change": pos_change, "problems": problems, "files": files,
            "times": (t_data, t_grid, t_placebo, time.time() - t_start)}



# ---------- the study ----------

def _slope_cell(x) -> str:
    return "—" if x is None or not _ok(x) else f"{x:+.2f}"


def study() -> dict:
    r = _compute()
    t1, w_btc = r["t1"], r["weekly"]["BTC"][0]
    res, pl, close = r["res"], r["pl"], r["close"]
    base = res["ma50"]

    # pass test
    sign_ok, coins_neg = {}, {}
    for k in STATS:
        x = t1[k]
        per_ok = all(s is not None and s < 0 for s, _ in x["per"])
        sign_ok[k] = per_ok and x["ci"][1] is not None and x["ci"][1] < 0
        coins_neg[k] = sum(1 for o in r["t1_other"].values() if o[k]["slope"] is not None and o[k]["slope"] < 0)
    verdicts = {}
    for r_ in ("J1", "S1"):
        k = RULE_STAT[r_]
        x = res[r_]
        verdicts[r_] = {"sign": sign_ok[k],
                        "score": x["low"] >= base["low"] - 1e-12 and x["c_deep_med"] >= base["c_deep_med"] - 1e-12,
                        "placebo": pl[r_]["share"] <= 0.10, "coins": coins_neg[k] >= 3}

    # table 1: next-week return by quintile
    rows1 = []
    for k in STATS:
        x = t1[k]
        for j in range(1, 6):
            tag = "（最低）" if j == 1 else "（最高）" if j == 5 else ""
            cells = ["—" if mean is None else f"{_pct(mean)}（{n} 週）" for mean, n in x["mean"][j]]
            rows1.append([STAT_NAME[k], f"第 {j} 組{tag}"] + cells)
        lo, hi = x["ci"]
        rows1.append([STAT_NAME[k], "斜率：每高一組，下週報酬差幾個百分點",
                      f"{_slope_cell(x['slope'])}（90% 區間 {_slope_cell(lo)} 到 {_slope_cell(hi)}）"]
                     + [f"{_slope_cell(s)}（{n} 週）" for s, n in x["per"]])
    for k in ("PJV4", "RV", "RKT"):
        x = t1[k]
        lo, hi = x["ci"]
        rows1.append([STAT_NAME[k] + "（參考）", "斜率",
                      f"{_slope_cell(x['slope'])}（90% 區間 {_slope_cell(lo)} 到 {_slope_cell(hi)}）"]
                     + [f"{_slope_cell(s)}（{n} 週）" for s, n in x["per"]])

    # table 2: fair grid
    g = r["grid"]
    n_d, n_c = base["d_n"], base["c_n"]
    rows2 = []
    for r_ in RULES:
        x = res[r_]
        plc = "—（基準）" if r_ == "ma50" else (f"{_share(pl[r_]['share'])}（安慰劑分數中位數 {_share(pl[r_]['p50'])}，"
                                               f"這條 {_share(pl[r_]['actual'])}）")
        rows2.append([NAMES[r_]] + [_share(v) for v in x["checks"]]
                     + [_share(x["low"]), f"{_pct(x['c_deep_med'])}｜{_share(x['loss_ok'])}", plc])
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(g.hd)))}｜—", "—"])

    # table 3: crashes and whipsaws
    bears = r["bears"][:4]
    rows3 = []
    for r_ in RULES:
        cells = []
        for i, b in enumerate(bears):
            c_ = r["crash"][r_][i]
            cells.append(f"避開 {_share(c_['avoid'])}" + (f"｜回升抓到 {_share(c_['catch'])}" if "catch" in c_ else ""))
        w = r["whip"][r_]
        extra = "—" if r_ == "ma50" else (f"{_share(r['flagged'][r_])} 的日子被週統計擋下；和 50 日均線持有狀態不同 "
                                          f"{_share(r['differ'][r_])}")
        rows3.append([NAMES[r_]] + cells + [f"{w['whips']:.1f} 次｜{w['cost']:.1f}%", f"{w['exits']:.1f} 次", extra])
    bear_cols = []
    for b in bears:
        t, lo = close.index[b["top"]], close.index[b["low"]]
        bear_cols.append(f"{_d(t)[:7]} 高點 → {_d(lo)[:7]} 低點" + ("" if b["done"] else "（還沒走完）"))

    # table 4: other coins
    rows4 = []
    for key in OTHER:
        if key not in r["weekly"]:
            continue
        w = r["weekly"][key][0]
        o = r["t1_other"].get(key)
        ok_weeks = int(np.isfinite(r["p_all"][key]["PJV"]).sum())
        each = [res[r_]["o_each"].get(key) for r_ in RULES]
        rows4.append([key, f"{_d(w.index[0])[:7]} 起，{ok_weeks} 週有分組",
                      _slope_cell(o["RSK"]["slope"]) if o else "—", _slope_cell(o["PJV"]["slope"]) if o else "—"]
                     + [("—" if e is None else f"{e[0]} 個裡 {e[1]} 個") for e in each])

    q_first = w_btc.index[int(np.argmax(np.isfinite(t1["RSK"]["q"])))]
    tables = [
        {"caption": ("比特幣每週（週一 0 點到下週一 0 點，UTC）用 5 分鐘報酬算偏態、用 15 分鐘報酬抓「往上跳動」，"
                     f"每週和自己前面最多 {WIN} 週（至少 {WIN_MIN} 週）比，分成五組，看「下一週」的漲跌平均。"
                     "斜率是把五組連成一條線，每高一組下週報酬差幾個百分點；學術論文預測是負的（越高下週越弱）。"
                     f"90% 區間用 {BLOCK} 週一塊的區塊重抽 {BOOT:,} 次。第一個時段因為要先累積 {WIN_MIN} 週，實際從 "
                     f"{_d(q_first)[:7]} 起。"),
         "columns": ["統計", "組別", "全期", PERIODS[0][0], PERIODS[1][0], PERIODS[2][0]],
         "rows": rows1},
        {"caption": (f"main_signal_check 的公平檢驗：比特幣每月 1 日起算（{n_d} 個，UTC 0 點日線，照課本做）、起算月×24 個收盤時間"
                     f"（{n_c:,} 格，照課本做）、同樣的格子照一般人方式做（10 小時後才做、成本 {msc.P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個時鐘 {msc.SIMS} 次模擬，和 50 日均線用同一組亂數）、其他幣（照課本做）。起算月從 {_d(r['first'])[:7]} "
                     f"（比特幣週統計滿 {WIN} 週）到 2023-12。前四欄是報酬贏過同一天起一直抱著的比例，排名分數＝四項裡最低的。"
                     f"安慰劑：把週統計整段循環錯開隨機週數，重算 {PLACEBO} 次，看有幾成分數不輸真正的規則"
                     f"（為了速度這一欄的一般人模擬每個時鐘 {PL_SIMS} 次）。"),
         "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他幣",
                     "排名分數（四項最低）", "最深虧損中位數｜比抱著淺的比例", "安慰劑分數不輸它的比例"],
         "rows": rows2},
        {"caption": ("比特幣 2017-08 以後每一次跌 50% 以上的大熊市（crash_timing 的定義，UTC 0 點日線）：照規則實際剩下多少，換算成避開了"
                     "高點到低點的幾成、抓到低點到下一個高點的幾成（中途被洗都算進去）。週統計要先累積一年，2018-08 以前 J1、S1 和 50 日均線一樣。"
                     f"被洗＝下車後 30 天內又用更高的價買回，從 {_d(r['first'])} 算起每年幾次、每年多付多少。"),
         "columns": ["規則"] + bear_cols + ["被洗：每年次數｜每年多付", "一年下車幾次", "和 50 日均線差在哪"],
         "rows": rows3},
        {"caption": ("換幣：每個幣用自己的 5 分鐘線算自己的週統計，看斜率方向（負＝和論文同方向），再把同樣的規則照課本做，"
                     "每月 1 日起算（週統計滿兩年起到三年前），報酬贏過一直抱著的起算月個數。"),
         "columns": ["幣", "資料", "偏態斜率", "往上跳動斜率", "50 日均線贏抱著", "J1 贏抱著", "S1 贏抱著"],
         "rows": rows4},
    ]

    # today
    last_w = w_btc.index[-1]
    p_now = {k: r["p_all"]["BTC"][k][-1] for k in STATS}
    pjv_now = w_btc["PJV"].iloc[-1]
    now = [{"label": f"最後一個走完的週（{_d(last_w)} 到 {_d(last_w + pd.Timedelta(days=6))}，UTC）",
            "value": (f"偏態 {_num(w_btc['RSK'].iloc[-1])}（在前面 {WIN} 週裡的百分位 {_share(p_now['RSK'])}）；"
                      f"往上跳動 {('—' if not _ok(pjv_now) else f'{pjv_now * 1e4:.2f}')}（平方報酬 × 10,000，百分位 {_share(p_now['PJV'])}，"
                      f"這週抓到 {int(np.nan_to_num(w_btc['jumps'].iloc[-1]))} 次跳動）")}]
    last_t = int((close.index[-1] + pd.Timedelta(days=1)).value // 1_000_000)
    ma_now = r["pos"]["ma50"][-1]
    now.append({"label": f"50 日均線（{_d(close.index[-1])} 收盤）", "value": "持有" if ma_now >= 0.5 else "現金"})
    for r_ in ("J1", "S1"):
        f = _flags_at(np.array([last_t]), *r["flags"][r_]["BTC"])[0]
        state = r["pos"][r_][-1]
        now.append({"label": NAMES[r_],
                    "value": (f"上週{'在' if f else '不在'}前 20%，這週{'被擋下（空手）' if f else '不擋'}；規則狀態："
                              f"{'持有' if state >= 0.5 else '現金'}")})
    now.append({"label": "這幾格的意思", "value": "只描述週統計和規則現在停在哪個狀態，不是買賣建議"})

    # verdict
    s1, s2 = t1["RSK"], t1["PJV"]

    def weak(x) -> str:
        lo, hi = x["ci"]
        if (x["slope"] or 0) < 0:
            return "比較弱" if hi is not None and hi < 0 else "略弱但不明顯"
        return "反而比較強" if lo is not None and lo > 0 else "反而略強但不明顯"

    def signs(x) -> str:
        return "、".join(f"{name} {_slope_cell(s)}" for (name, _, _), (s, _) in zip(PERIODS, x["per"]))

    ok_names = [STAT_NAME[k].split("（")[0] for k in STATS if sign_ok[k]]
    first_s = (f"比特幣上週偏態高時，下週平均{weak(s1)}（每高一組 {_slope_cell(s1['slope'])} 個百分點，90% 區間 "
               f"{_slope_cell(s1['ci'][0])} 到 {_slope_cell(s1['ci'][1])}）；往上跳動多時下週{weak(s2)}"
               f"（{_slope_cell(s2['slope'])}，{_slope_cell(s2['ci'][0])} 到 {_slope_cell(s2['ci'][1])}）。")
    second_s = (f"三個時段的斜率：偏態 {signs(s1)}；跳動 {signs(s2)}，"
                + (f"{'、'.join(ok_names)}三段都和論文同方向且區間不含零。" if ok_names else "沒有一個在三段都和論文同方向、區間又不含零。"))
    win = [r_ for r_ in ("J1", "S1") if all(verdicts[r_].values())]
    third_s = (f"拿來在 50 日均線上提早下車，排名分數 J1 {_share(res['J1']['low'])}、S1 {_share(res['S1']['low'])}，"
               f"50 日均線 {_share(base['low'])}，安慰劑不輸的比例 {_share(pl['J1']['share'])}／{_share(pl['S1']['share'])}，"
               + ("事先定的四項條件全過：" + "、".join(win) + "。" if win else "事先定的四項條件沒有一條規則全過。"))
    verdict = first_s + second_s + third_s

    t_data, t_grid, t_placebo, t_all = r["times"]
    cuts, bad = r["look"]
    rep = r["rep"]

    def mark(v: bool) -> str:
        return "過" if v else "沒過"

    notes = [
        ("事先定好的通過條件（看結果前寫下）：預測方向在三個時段都成立且全期斜率 90% 區間不含零；規則排名分數不低於 50 日均線、"
         "最深虧損中位數不更深；安慰劑分數不輸它的比例 ≤ 10%；另外四個幣至少三個斜率同方向。結果："
         + "；".join(f"{SHORT[r_]}：方向 {mark(v['sign'])}、分數與虧損 {mark(v['score'])}、安慰劑 {mark(v['placebo'])}、"
                     f"換幣 {mark(v['coins'])}（{coins_neg[RULE_STAT[r_]]}/4 個幣同方向）" for r_, v in verdicts.items()) + "。"),
        ("論文實際發現：Lee & Wang（JFQA 2025，正式名稱是 Variance Decomposition and Cryptocurrency Return Prediction）用 "
         "Coinbase 100 個幣 2015-10 到 2023-06 的 15 分鐘報酬，每週把幣按前一個月的波動分三組，波動高的那組下週平均少 3.7%，"
         "拆開後主要來自「往上跳動」（高減低每週 −3.6%）和平常的波動；這是幣和幣之間比較，不是比特幣自己前後比。"
         "Amaya 等（JFE 2015）是美股：偏態最低一成減最高一成的股票下週多 0.19%。Ahmed & Al Mafrachi（2021）說比特幣、以太幣、瑞波幣的"
         "偏態有預測力，但公開摘要沒有寫方向。Atance & Serna（2024）用模型估的條件偏態，2018–2020 是負的、2021–2022 變成正的。"),
        ("跳動的算法照 Lee & Wang：15 分鐘報酬先除掉一天裡固定時段的波動差異（只用過去 90 天），再用前 154 組相鄰報酬乘積估當下的正常波動，"
         "超過 5% 顯著水準（Gumbel 臨界值，n＝一週 672 個 15 分鐘）的算跳動，往上的跳動平方加總就是「往上跳動」。"
         "論文用 15 分鐘的買賣中間價，這裡用幣安成交價（5 分鐘線取每 15 分鐘最後一根），時段波動也只用簡化的除法。"
         "論文用 4 週的窗，這裡事先定用 1 週，4 週的結果列在第一表當參考。偏態照 Amaya：每天 288 個 5 分鐘報酬算一天的偏態，一週取平均。"),
        (f"核對 50 日均線：用 main_signal_check 自己的網格（2018-01 到 2023-12 起算 × 24 個收盤時間），這裡算出照課本做 "
         f"{rep['c_n']:,} 格裡 {rep['c_beat']:,} 格報酬贏、照一般人方式做 {rep['p_n']:,} 次裡 {rep['p_beat']:,} 次贏"
         "（main_signal_check 2026-11-17 那次是 1,728 格裡 1,714 格、345,600 次裡 275,236 次；多一天資料會差一點點）。"),
        (f"不偷看未來的檢查：(1) 把 5 分鐘資料在 {cuts} 個切點之後全部打亂重算，切點前的週統計和百分位"
         + ("完全沒變" if bad == 0 else f"有 {bad} 項變了（結果不可信）")
         + f"；(2) 所有決策時間用到的週統計都已在決策前結束（違反 {r['early']} 次）；(3) 把日線最後 400 天打亂，J1 前面的持有狀態"
         + ("沒變。" if r["pos_change"] == 0 else "變了（結果不可信）。")),
        (f"參考列（不在事先定的規則裡，看完結果也不拿來當規則）：峰態斜率 {_slope_cell(t1['RKT']['slope'])}"
         f"（90% 區間 {_slope_cell(t1['RKT']['ci'][0])} 到 {_slope_cell(t1['RKT']['ci'][1])}，三段 {signs(t1['RKT'])}），"
         f"總波動 {_slope_cell(t1['RV']['slope'])}，4 週窗的往上跳動 {_slope_cell(t1['PJV4']['slope'])}。這幾列是順便算的，"
         "看了好幾個統計才挑出一個顯著的，很可能只是巧合；要當真得另外事先登記再驗。"),
        ("比特幣的週數不多，五組在一個時段裡每組只有二三十週，一兩週的大漲大跌就能左右平均；"
         "相鄰起算月和 24 個收盤時間高度重疊，「幾個裡幾個」不是互相獨立的證據。"),
        ("成本粗略：照課本 0.10%、一般人 0.25%，沒有滑價、稅和現金利息。幣安 5 分鐘線偶爾缺幾根（交易所維修），缺的那段報酬併進下一根。"
         "其他幣是今天還在的大幣，有倖存者偏誤；MSTR 沒有 5 分鐘資料，所以換資產只用四個幣。"),
        (f"執行時間：這次下載 {r['files']} 個檔案，資料 {t_data:.0f} 秒、公平檢驗 {t_grid - t_data:.0f} 秒、安慰劑 "
         f"{t_placebo - t_grid:.0f} 秒，共 {t_all:.0f} 秒。第一次要下載五個幣約 500 個 5 分鐘月檔（每個約 0.5 MB，約 5 分鐘），"
         "之後只下載新的日檔。"),
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去的表現和現在的狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "jump_skew",
        "title": "一週內的跳動和偏態",
        "question": ("比特幣上一週裡「突然跳」的程度（很短時間內大漲或大跌）和漲跌的不對稱（偏態）偏高時，下一週是不是比較弱？"
                     "學術論文說的方向在不同時段是否一致？拿來在 50 日均線上提早下車，有沒有贏過均線和隨機對照？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Lee & Wang (2025), Variance Decomposition and Cryptocurrency Return Prediction, JFQA 60(4) 1859–1890",
             "url": "https://ideas.repec.org/a/cup/jfinqa/v60y2025i4p1859-1890_9.html"},
            {"name": "Amaya, Christoffersen, Jacobs & Vasquez (2015), Does realized skewness predict the cross-section of equity returns? JFE 118",
             "url": "https://econpapers.repec.org/RePEc:eee:jfinec:v:118:y:2015:i:1:p:135-167"},
            {"name": "Ahmed & Al Mafrachi (2021), Do higher-order realized moments matter for cryptocurrency returns? IREF 72",
             "url": "https://econpapers.repec.org/RePEc:eee:reveco:v:72:y:2021:i:c:p:483-499"},
            {"name": "Atance & Serna (2024), Time-varying expected returns, conditional skewness and Bitcoin return predictability, QREF 96",
             "url": "https://ideas.repec.org/a/eee/quaeco/v96y2024ics1062976924000747.html"},
            {"name": "幣安公開歷史資料：現貨 5 分鐘線月檔／日檔（BTC、ETH、SOL、BNB、XRP 對 USDT）",
             "url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/5m/"},
            {"name": "幣安現貨 1 小時線（24 個收盤時間）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前，儀表板長期日線）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
