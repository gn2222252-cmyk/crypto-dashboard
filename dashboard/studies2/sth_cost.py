"""Historical study "sth_cost": the short-term-holder (STH) realized price - the average on-chain cost basis of coins
that last moved within about 155 days - is the most quoted "support / resistance" line among on-chain analysts
(CryptoSlate, Bitcoin Magazine, Axel Adler; a Bitcoin Magazine Pro substack post backtested "hold when price > STH
realized price, 0.10% fees" without publishing the numbers). Does "hold BTC when close > STH cost basis" beat or add
to the dashboard's main signal (close > 50-day average, ma50)? Is it more than a ~155-day moving average in disguise?
Does the popular "reclaim" entry or the STH-MVRV extremes add anything?

Data (found before running): the dashboard's own collector reads STH-MVRV from BGeometrics (bitcoin-data.com), but
the free plan only serves the last four years (stated in its API description), so it cannot test 2012-2021.
Checkonchain's public chart pages (charts.checkonchain.com, no key, no login) embed the full daily series "STH Cost
Basis" from 2010-08-31 in the chart file served from charts-cdn.checkonchain.com; that is the source used here. The
BGeometrics store is only used as a cross-check for the overlapping years. Archived copies of the same chart in the
Wayback Machine (2024-11-22 and 2026-02-27) show that Checkonchain revises history a little (the study measures how
much and whether the hold / cash state flips because of it), so the test uses today's vintage and says so.

Pre-registered design (fixed before running; no other variants):
H1: hold BTC when close > STH cost basis, else cash.
H2: ma50 OR reclaim: when ma50 is off and the close crosses from at-or-below to above the STH cost basis, enter;
    a reclaim position stays until ma50 has turned on (from then on ma50 governs and its exit ends the trade) - or,
    while ma50 is still off, until the close falls back below the STH cost basis (failed reclaim). When ma50 is on,
    hold (plain ma50).
H3: ma50, but cash when STH-MVRV (close / STH cost basis) > 1.3 (fixed overheated level).
Plain moving-average twins: L = the simple-average length (20 ... 400 days) whose line is closest to the STH cost
    basis on 2012-01-01 ... 2019-12-31 only (smallest mean |log(SMA_L / STH cost)|). T1 = close > SMA_L; T2 = H2 with
    SMA_L in place of the STH cost; T3 = ma50 but cash when close / SMA_L > q, q chosen on the same training years so
    that it is exceeded on the same share of days as STH-MVRV > 1.3.
STH-MVRV extremes (description, not a rule): forward 30 / 90-day returns when STH-MVRV < 0.8 (capitulation, "buy
    zone"), 0.8-1, 1-1.3, > 1.3 ("overheated"), with 90% block-bootstrap intervals, full sample and by period.
Exam: main_signal_check's machinery. (1) 108 monthly starts 2015-01 ... 2023-12 on the dashboard's daily UTC closes,
    textbook execution (0.10%); (2) 72 starts x 24 closing hours (Binance 1-hour grid), textbook; (3) the same cells,
    ordinary person: 10 hours later, 0.25%, 10% of entries missed, 200 simulations per clock, main_signal_check's seed
    (ma50 uses its exact random draws); (4) there is no free STH series for other coins, so the 4th check is replaced
    by period splits: monthly starts inside 2012-17, 2018-21 and 2022-now (each measured only to the end of its period,
    last start six months before that end), the check = the lowest of the three "beats holding" shares.
    Ranking score = the lowest of the four "beats holding" shares (main_signal_check); median deepest loss = median
    over the 1,728 textbook cells.
Placebo: the STH-MVRV series (close / cost on the same day) circularly shifted by 180 days or more, cost rebuilt as
    close / shifted ratio, 200 draws, the whole exam rebuilt (20 ordinary-person simulations per clock; the real rule
    scored the same way for this comparison); placebo share = draws scoring at least as high.
Pass test (stated now): ranking score >= ma50's (and eligible when ma50 is) AND median deepest loss no worse than
    ma50's AND placebo share <= 10% AND beats its plain-MA twin (higher ranking score; a tie is broken by the median
    yearly edge over holding on the 1,728 cells).
Timing: Checkonchain's value for on-chain day D is used only from the daily close of D+1 (UTC closes); on the
    24-clock grid a close on UTC date E uses the value of E-2. Never today's unfinished day. Only public key-less
    endpoints with the dashboard's fixed user agent (common.HEADERS).
"""
from __future__ import annotations

import base64
import json
import math

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COC_PAGE = ("https://charts.checkonchain.com/btconchain/pricing/quantiles_realizedprice_sth/"
            "quantiles_realizedprice_sth_light.html")
COC_DATA = COC_PAGE.replace("https://charts.", "https://charts-cdn.")
COC_TRACE = "STH Cost Basis"
WB_URL = ("https://web.archive.org/web/20241122171034id_/"
          "https://charts.checkonchain.com/btconchain/unrealised/mvrv_sth/mvrv_sth_light.html")
WB_TRACE = "STH Realised Price"
WB_DAY = "2024-11-22"
BG_PAGE = "https://charts.bgeometrics.com/"
SRC_CS = ("https://cryptoslate.com/bitcoins-short-term-holder-realized-price-remains-a-critical-support-amid-market-"
          "volatility/")
SRC_BM = "https://bitcoinmagazine.com/markets/analyzing-long-term-short-term-bitcoin-holder-cost-bases"
SRC_BMP = "https://bmpro.substack.com/p/backtest-signals-to-hedge-your-hodl-strategy"

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
HOT, COLD = 1.3, 0.8
TRAIN = ("2012-01-01", "2019-12-31")
LENGTHS = range(20, 401)
PERIODS = (("2012 到 2017", "2012-01-01", "2017-12-31"), ("2018 到 2021", "2018-01-01", "2021-12-31"),
           ("2022 到現在", "2022-01-01", None))
LAST_GAP = 182
PLACEBO, GAP, PL_SIMS = 200, 180, 20
BLOCK = {30: 60, 90: 120}
RULES = ("ma50", "h1", "h2", "h3", "t1", "t2", "t3")
TESTED = ("h1", "h2", "h3")
TWIN = {"h1": "t1", "h2": "t2", "h3": "t3"}
KI = {"ma50": 1, "h1": 81, "h2": 82, "h3": 83, "t1": 84, "t2": 85, "t3": 86}  # ma50 = main_signal_check's draws
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return (f"{v * 100:.0f}%" if abs(v * 100) >= 0.5 else "0%") if _ok(v) else "—"


def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def _pt(v) -> str:
    return _pp(v * 100) if _ok(v) else "—"


def _ci(lo, hi, f) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _plotly(text: str, name: str) -> dict:
    """One named trace of a Plotly chart file -> {YYYY-MM-DD: value} (handles base64 'bdata' arrays)."""
    start = text.find("Plotly.newPlot(")
    if start < 0:
        raise RuntimeError("圖表檔裡找不到資料")
    data, _ = json.JSONDecoder().raw_decode(text[text.find("[", start):])
    for tr in data:
        if tr.get("name") != name:
            continue
        y = tr["y"]
        arr = (np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"]) if isinstance(y, dict)
               else np.array(y, dtype=float))
        out = {}
        for x, v in zip(tr["x"], arr.astype(float)):
            if math.isfinite(v) and v > 0:
                out[str(x)[:10]] = float(v)
        return out
    raise RuntimeError(f"圖表檔裡沒有「{name}」")


def _fetch_text(url: str) -> str:
    reply = requests.get(url, headers=common.HEADERS, timeout=180)
    reply.raise_for_status()
    return reply.text


def _coc() -> tuple[pd.Series, list, dict]:
    """Today's Checkonchain vintage; compares it with the vintage stored by the previous run (revisions)."""
    path = common.STORE / "sth_cost_coc.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        old = {}
    problems, rev = [], {}
    try:
        got = _plotly(_fetch_text(COC_DATA), COC_TRACE)
        if len(got) < 5000:
            raise RuntimeError(f"只有 {len(got)} 天")
        if old.get("data") and old.get("day") != common.today_utc():
            both = [(old["data"][k], v) for k, v in got.items()
                    if k in old["data"] and k < (pd.Timestamp(old["day"]) - 3 * DAY).strftime("%Y-%m-%d")]
            if both:
                diff = np.array([abs(b / a - 1) for a, b in both])
                rev = {"day": old["day"], "n": len(both), "max": float(diff.max()), "big": float(np.mean(diff > 0.01))}
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            if old.get("day") != common.today_utc():
                path.write_text(json.dumps({"day": common.today_utc(), "data": got,
                                            "previous_day": old.get("day")}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        if not old.get("data"):
            raise RuntimeError(f"Checkonchain 取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        got = old["data"]
        problems.append(f"Checkonchain 這次取不到（{type(exc).__name__}），用 {old.get('day')} 存下來的資料。")
    s = pd.Series(got, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[s.index < pd.Timestamp(common.today_utc())]  # never today's unfinished day
    return s.asfreq("D").ffill(limit=3), problems, rev


def _wayback() -> pd.Series | None:
    """The archived 2024-11-22 vintage (immutable; fetched once and stored)."""
    path = common.STORE / "sth_cost_wb20241122.json"
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        try:
            got = _plotly(_fetch_text(WB_URL), WB_TRACE)
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None
    s = pd.Series(got, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _bg(close: pd.Series) -> pd.Series | None:
    """The dashboard's own BGeometrics STH-MVRV store -> implied cost basis (close / ratio), check only."""
    try:
        d = json.loads((common.STORE / "bg_sth_mvrv.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return (close.reindex(s.index) / s).dropna()


def _at(series: pd.Series, dates, lag: int) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


# ---------- rules ----------

def _sma(c: np.ndarray, n: int) -> np.ndarray:
    return ct._mean(np.asarray(c, dtype=float), n)


def _reclaim(c: np.ndarray, line: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """H2 / T2: ma50, plus an early entry when the close crosses back above the line while ma50 is off."""
    n = len(c)
    out = np.full(n, np.nan)
    cl, ll, ml = c.tolist(), line.tolist(), ma.tolist()
    rec = False
    prev_below = None
    for i in range(n):
        li, mi = ll[i], ml[i]
        if li != li or mi != mi:
            prev_below = None if li != li else cl[i] <= li
            rec = False
            continue
        below = cl[i] <= li
        if mi >= 0.5:
            rec = False
            out[i] = 1.0
        elif rec:
            if cl[i] < li:
                rec = False
                out[i] = 0.0
            else:
                out[i] = 1.0
        elif prev_below is True and not below:
            rec = True
            out[i] = 1.0
        else:
            out[i] = 0.0
        prev_below = below
    return out


def _rules(c: np.ndarray, ma: np.ndarray, cost: np.ndarray, sma: np.ndarray, q: float,
           keys=RULES) -> dict[str, np.ndarray]:
    """Positions wanted after each close. cost / sma are the lines usable at that close."""
    out = {}
    ratio = c / cost
    if "ma50" in keys:
        out["ma50"] = ma
    if "h1" in keys:
        out["h1"] = ct._gt(c, cost)
    if "h2" in keys:
        out["h2"] = _reclaim(c, cost, ma)
    if "h3" in keys:
        out["h3"] = np.where(np.isnan(ma) | np.isnan(ratio), np.nan, ma * (ratio <= HOT))
    if "t1" in keys:
        out["t1"] = ct._gt(c, sma)
    if "t2" in keys:
        out["t2"] = _reclaim(c, sma, ma)
    if "t3" in keys:
        r2 = c / sma
        out["t3"] = np.where(np.isnan(ma) | np.isnan(r2), np.nan, ma * (r2 <= q))
    return out


# ---------- the exam ----------

def _grid(close: pd.Series, pos: np.ndarray, first, last, end=None) -> tuple:
    """Textbook runs from monthly start dates; returns (rule ret, rule deep, hold ret, hold deep)."""
    if end is not None:
        keep = close.index <= pd.Timestamp(end)
        close, pos = close[keep], pos[keep]
    change = close.pct_change().fillna(0.0).values
    known = ~np.isnan(pos)
    begin = max(close.index[int(np.argmax(known))] + DAY, pd.Timestamp(first))
    starts, _ = msc._month_starts(close.index, begin, last)
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
    r, dd = msc._suffix(msc._equity(net), starts, YEAR)
    return r[:, 0], dd[:, 0], hr[:, 0], hd[:, 0]


class _Clock:
    """main_signal_check's 24 closing hours with every rule's line per hour."""

    def __init__(self, price, b0, pre, L: int):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.c, self.ma, self.sma = [], [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]])
            self.c.append(c[npre:][:self.days])
            self.ma.append(ct._gt(c, _sma(c, 50))[npre:][:self.days])
            self.sma.append(_sma(c, L)[npre:][:self.days])

    def wants(self, cost: pd.Series, q: float, keys) -> list[dict]:
        line = _at(cost, self.m.dates, 2)
        return [{k: np.nan_to_num(v, nan=0.0) for k, v in
                 _rules(self.c[h], self.ma[h], line, self.sma[h], q, keys).items()} for h in range(24)]

    def full(self, k: str, wants: list[dict], sims: int, seed_add: int = 0) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, sims))
        pdd = np.zeros((n, 24, sims))
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
                "c_p10": float(np.percentile(edge, 10)),
                "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                "c_deep_worst": float(td.min()),
                "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
                "p_deep_med": float(np.median(pdd))}


def _evaluate(k: str, close: pd.Series, pos: dict, clock: _Clock, wants: list[dict], sims: int,
              seed_add: int = 0) -> dict:
    dr, dd, hr, hd = _grid(close, pos[k], msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    x = clock.full(k, wants, sims, seed_add)
    x.update({"d_n": len(dr), "d_beat": int((dr > hr).sum()), "d_shallow": int((dd > hd).sum()),
              "d_edge": float(np.median(dr - hr)), "d_deep_med": float(np.median(dd))})
    last_day = close.index[-1]
    per = []
    for name, a, b in PERIODS:
        end = pd.Timestamp(b) if b else last_day
        r_, d_, h_, hd_ = _grid(close, pos[k], a, end - LAST_GAP * DAY, end)
        per.append({"name": name, "n": len(r_), "beat": int((r_ > h_).sum()), "edge": float(np.median(r_ - h_)),
                    "deep": float(np.median(d_)), "hold_deep": float(np.median(hd_)),
                    "first": None, "end": end})
    x["per"] = per
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   min(p["beat"] / p["n"] for p in per)]
    x["low"] = min(x["checks"])
    x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
    return x


# ---------- descriptions ----------

def _twin(close: pd.Series, cost: pd.Series) -> tuple[int, dict]:
    c = close.values.astype(float)
    line = cost.reindex(close.index).values
    tr = (close.index >= pd.Timestamp(TRAIN[0])) & (close.index <= pd.Timestamp(TRAIN[1])) & np.isfinite(line)
    fit = {}
    for n in LENGTHS:
        s = _sma(c, n)
        m = tr & np.isfinite(s)
        fit[n] = float(np.mean(np.abs(np.log(s[m] / line[m]))))
    best = min(fit, key=fit.get)
    return best, fit


def _buckets(close: pd.Series, ratio_t: np.ndarray) -> dict:
    df = pd.DataFrame({"r": ratio_t}, index=close.index)
    for h in (30, 90):
        df[f"f{h}"] = close.shift(-h) / close - 1
    df = df[df["r"].notna() & (df.index >= pd.Timestamp(PERIODS[0][1]))]
    masks = {f"< {COLD}（套牢投降區）": lambda r: r < COLD, f"{COLD} 到 1": lambda r: (r >= COLD) & (r < 1),
             f"1 到 {HOT}": lambda r: (r >= 1) & (r <= HOT), f"> {HOT}（過熱區）": lambda r: r > HOT,
             "全部日子": lambda r: r == r}
    out = {}
    for name, f in masks.items():
        row = {}
        for pname, a, b in (("全部", PERIODS[0][1], None),) + PERIODS:
            sub = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b or "2100-01-01"))]
            m = f(sub["r"]).values.astype(bool)
            mv = m.astype(float)
            cell = {"days": int(m.sum()), "runs": at._runs(m)}
            for h in (30, 90):
                cell[f"f{h}"] = (at._boot({"y": sub[f"f{h}"].values, "m": mv}, at._cell_fn, BLOCK[h])
                                 if pname == "全部" else
                                 (float(np.nanmean(sub[f"f{h}"].values[m])) if m.sum() >= 20 else None,))
            row[pname] = cell
        out[name] = row
    return out


def _lookahead(close: pd.Series, cost: pd.Series, ma: np.ndarray, sma: np.ndarray, q: float, clock: _Clock,
               L: int) -> tuple:
    """Scramble closes after a cut and cost values from the cut day on: decisions up to the cut must not move."""
    base = _rules(close.values, ma, _at(cost, close.index, 1), sma, q)
    rng = np.random.default_rng(11)
    bad = cuts = 0
    w0 = clock.wants(cost, q, RULES)
    for cut in (len(close) - 2500, len(close) - 900, len(close) - 60):
        t = close.index[cut]
        c2 = close.copy()
        c2.iloc[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(close) - cut - 1))
        k2 = cost.copy()
        later = k2.index >= t
        k2[later] = k2[later].values * np.exp(rng.normal(0, 0.5, int(later.sum())))
        v = c2.values.astype(float)
        alt = _rules(v, ct._gt(v, _sma(v, 50)), _at(k2, close.index, 1), _sma(v, L), q)
        cuts += 1
        for k in RULES:
            if not np.array_equal(np.nan_to_num(base[k][:cut + 1], nan=-1), np.nan_to_num(alt[k][:cut + 1], nan=-1)):
                bad += 1
        j = int(clock.m.dates.searchsorted(t))
        if 0 < j < clock.days:
            e = clock.m.dates[j]
            k3 = cost.copy()
            k3[k3.index >= e - DAY] *= np.exp(rng.normal(0, 0.5, int((k3.index >= e - DAY).sum())))
            w1 = clock.wants(k3, q, RULES)
            cuts += 1
            for h in (0, msc.PERSON_CLOCK):
                for k in TESTED:
                    if not np.array_equal(w0[h][k][:j + 1], w1[h][k][:j + 1]):
                        bad += 1
    return cuts, bad


# ---------- the computation ----------

def _compute() -> dict:
    cost, problems, rev = _coc()
    close = te._daily_close()
    c = close.values.astype(float)
    ma = ct._gt(c, _sma(c, 50))
    L, fit = _twin(close, cost)
    sma = _sma(c, L)
    cost_t = _at(cost, close.index, 1)
    ratio_t = c / cost_t
    tr = (close.index >= pd.Timestamp(TRAIN[0])) & (close.index <= pd.Timestamp(TRAIN[1]))
    hot_share = float(np.mean(ratio_t[tr & np.isfinite(ratio_t)] > HOT))
    r2 = (c / sma)[tr & np.isfinite(sma)]
    q = float(np.quantile(r2, 1 - hot_share))
    pos = _rules(c, ma, cost_t, sma, q)

    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.insert(0, problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _Clock(price, b0, pre, L)
    wants = clock.wants(cost, q, RULES)
    s = {k: _evaluate(k, close, pos, clock, wants, SIMS) for k in RULES}

    # placebo: STH-MVRV (close / cost, same day) circularly shifted, cost rebuilt
    real20 = {k: _evaluate(k, close, pos, clock, wants, PL_SIMS)["low"] for k in TESTED}
    rr = (close.reindex(cost.index) / cost).dropna()
    rng = np.random.default_rng(SEED + 155)
    plac = {k: [] for k in TESTED}
    for i in range(PLACEBO):
        sh = pd.Series(np.roll(rr.values, int(rng.integers(GAP, len(rr) - GAP))), index=rr.index)
        cost_s = close.reindex(sh.index) / sh
        pp = _rules(c, ma, _at(cost_s, close.index, 1), sma, q, keys=TESTED)
        wp = clock.wants(cost_s, q, TESTED)
        for k in TESTED:
            plac[k].append(_evaluate(k, close, pp, clock, wp, PL_SIMS, 7919 * (i + 1))["low"])
    placebo = {k: float(np.mean(np.array(v) >= real20[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    # in disguise? agreement of H1 with T1 and of the two lines
    first = int(close.index.searchsorted(pd.Timestamp(PERIODS[0][1])))
    m = np.isfinite(pos["h1"]) & np.isfinite(pos["t1"])
    m[:first] = False
    agree = float(np.mean(pos["h1"][m] == pos["t1"][m]))
    gap_line = np.abs(sma / cost_t - 1)[m]
    agree_per = []
    for name, a, b in PERIODS:
        mm = m & (close.index >= pd.Timestamp(a)) & (close.index <= pd.Timestamp(b or "2100-01-01"))
        agree_per.append((name, float(np.mean(pos["h1"][mm] == pos["t1"][mm]))))
    first15 = int(close.index.searchsorted(pd.Timestamp("2015-01-01")))
    held = {k: float(np.nanmean(pos[k][first15:])) for k in RULES}
    differ = {k: float(np.mean(np.nan_to_num(pos[k][first15:], nan=0) != np.nan_to_num(pos["ma50"][first15:], nan=0)))
              for k in RULES}
    trades = {k: float(np.sum(np.diff(np.nan_to_num(pos[k][first15:])) > 0.5) / ((len(c) - first15) / YEAR))
              for k in RULES}

    # revisions: archived 2024-11-22 vintage vs today's
    wb = _wayback()
    wbx = None
    if wb is not None and len(wb):
        both = pd.concat([wb, cost], axis=1, sort=True).dropna()
        both = both[(both.index >= pd.Timestamp("2015-01-01")) & (both.index < pd.Timestamp(WB_DAY) - 3 * DAY)]
        rel = (both.iloc[:, 1] / both.iloc[:, 0] - 1).abs()
        ct_old = _at(wb, close.index, 1)
        mm = (close.index >= pd.Timestamp("2015-01-01")) & (close.index < pd.Timestamp(WB_DAY)) \
            & np.isfinite(ct_old) & np.isfinite(cost_t)
        flips = float(np.mean((c[mm] > ct_old[mm]) != (c[mm] > cost_t[mm])))
        tail = rel[rel.index >= pd.Timestamp(WB_DAY) - 33 * DAY]
        wbx = {"n": len(rel), "med": float(rel.median()), "max": float(rel.max()), "big": float(np.mean(rel > 0.01)),
               "flips": flips, "tail_max": float(tail.max()) if len(tail) else None}
    else:
        problems.append("Wayback Machine 的 2024-11-22 舊版這次取不到，沒有做歷史修正的比對。")

    # cross-check: BGeometrics (dashboard collector) implied cost
    bg = _bg(close)
    bgx = None
    if bg is not None and len(bg) > 100:
        both = pd.concat([bg, cost], axis=1, sort=True).dropna()
        rel = (both.iloc[:, 0] / both.iloc[:, 1] - 1)
        cc = close.reindex(both.index).values
        bgx = {"n": len(both), "from": both.index[0], "med": float(rel.median()), "abs": float(rel.abs().median()),
               "agree": float(np.mean((cc > both.iloc[:, 0].values) == (cc > both.iloc[:, 1].values)))}

    buckets = _buckets(close, ratio_t)
    look = _lookahead(close, cost, ma, sma, q, clock, L)
    return {"close": close, "cost": cost, "cost_t": cost_t, "ratio_t": ratio_t, "L": L, "fit": fit, "q": q,
            "hot_share": hot_share, "pos": pos, "s": s, "clock": clock, "real20": real20, "placebo": placebo,
            "placebo_med": placebo_med, "agree": agree, "agree_per": agree_per, "gap_line": float(np.median(gap_line)),
            "held": held, "differ": differ, "trades": trades, "wb": wbx, "bg": bgx, "rev": rev, "buckets": buckets, "look": look,
            "problems": problems, "sma": sma, "ma": ma}


# ---------- today ----------

def _now(r: dict) -> list[dict]:
    close, pos = r["close"], r["pos"]
    c = float(close.iloc[-1])
    cost = float(r["cost_t"][-1])
    sma = float(r["sma"][-1])

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    m50 = float(close.iloc[-50:].mean())
    return [
        {"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {c:,.0f} 美元"},
        {"label": f"這個收盤能用的短期持有者成本價（Checkonchain {_d(close.index[-1] - DAY)} 的值）",
         "value": f"{cost:,.0f} 美元（收盤{'高於' if c > cost else '低於'}它 {abs(c / cost - 1) * 100:.1f}%；"
                  f"短期持有者 MVRV {c / cost:.2f}）"},
        {"label": "H1：收盤高於短期持有者成本價才持有", "value": word(pos["h1"][-1])},
        {"label": "50 日均線（主訊號）", "value": f"{word(pos['ma50'][-1])}（50 日平均 {m50:,.0f} 美元）"},
        {"label": "H2：50 日均線＋站回成本價提早進場", "value": word(pos["h2"][-1])},
        {"label": f"H3：50 日均線，但短期持有者 MVRV > {HOT} 時改拿現金", "value": word(pos["h3"][-1])},
        {"label": f"對照：{r['L']} 日均線（和成本價最像的普通均線）",
         "value": f"{word(pos['t1'][-1])}（{r['L']} 日平均 {sma:,.0f} 美元）"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]


# ---------- the study ----------

def _beats_twin(x: dict, t: dict) -> bool:
    return (x["low"], x["c_edge"]) > (t["low"], t["c_edge"])


NAMES = {"ma50": "50 日均線（基準，主訊號）", "h1": "H1：收盤高於短期持有者成本價才持有",
         "h2": "H2：50 日均線＋站回成本價時提早進場", "h3": f"H3：50 日均線，但短期持有者 MVRV > {HOT} 時拿現金"}


def study() -> dict:
    r = _compute()
    s, close, L = r["s"], r["close"], r["L"]
    names = dict(NAMES)
    names.update({"t1": f"T1：收盤高於 {L} 日均線（H1 的普通均線雙胞胎）", "t2": f"T2：50 日均線＋站回 {L} 日均線時提早進場",
                  "t3": f"T3：50 日均線，但收盤 ÷ {L} 日均線 > {r['q']:.2f} 時拿現金"})
    m = s["ma50"]
    ma_elig = m["loss_ok"] >= LOSS_BAR
    ok = {}
    for k in TESTED:
        x, t = s[k], s[TWIN[k]]
        ok[k] = {"score": x["low"] >= m["low"] - 1e-12 and (x["loss_ok"] >= LOSS_BAR or not ma_elig),
                 "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10,
                 "twin": _beats_twin(x, t)}
        ok[k]["all"] = all(ok[k].values())
    passed = [k for k in TESTED if ok[k]["all"]]
    n_d, n_c, n_p = m["d_n"], m["c_n"], m["p_n"]
    pnames = [p[0] for p in PERIODS]

    # table 1: the exam
    rows1 = []
    for k in RULES:
        x = s[k]
        per = "｜".join(_share(p["beat"] / p["n"]) for p in x["per"])
        rows1.append([names[k], f"{_share(x['checks'][0])}｜{_pp(x['d_edge'])}", f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}", per, _share(x["low"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", _share(x["loss_ok"]),
                      f"{_share(r['held'][k])}｜{r['trades'][k]:.1f} 次"])
    rows1.append(["對照：一直抱著不動", "—", "—", "—", "—", "—",
                  f"{_pct(float(np.median(r['clock'].hd)))}｜{_pct(float(r['clock'].hd.min()))}", "—", "100%｜0 次"])

    # table 2: periods
    rows2 = []
    for k in RULES:
        rows2.append([names[k]] + [f"{_of(p['n'], p['beat'])}｜{_pp(p['edge'])}｜{_pct(p['deep'])}（抱著 {_pct(p['hold_deep'])}）"
                                   for p in s[k]["per"]])

    # table 3: pass test
    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows3 = []
    for k in TESTED:
        x, t = s[k], s[TWIN[k]]
        rows3.append([names[k], f"{_share(x['low'])} vs {_share(m['low'])}（{yes(ok[k]['score'])}）",
                      f"{_pct(x['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（亂移後中位數 {_share(r['placebo_med'][k])}，真的 {_share(r['real20'][k])}）"
                      f"（{yes(ok[k]['placebo'])}）",
                      f"{_share(x['low'])}／{_pp(x['c_edge'])} vs {_share(t['low'])}／{_pp(t['c_edge'])}（{yes(ok[k]['twin'])}）",
                      "通過" if ok[k]["all"] else "沒有通過"])

    # table 4: STH-MVRV buckets
    b = r["buckets"]
    rows4 = []
    for name, row in b.items():
        a = row["全部"]
        rows4.append([name, f"{a['days']:,} 天｜{a['runs']} 段",
                      f"{_p(a['f30'][0])} {_ci(a['f30'][1], a['f30'][2], _p)}",
                      f"{_p(a['f90'][0])} {_ci(a['f90'][1], a['f90'][2], _p)}",
                      "｜".join(f"{_p(row[p]['f90'][0])}（{row[p]['days']} 天）" for p in pnames)])

    h1, t1 = s["h1"], s["t1"]
    best = max(TESTED, key=lambda k: (s[k]["low"], s[k]["c_edge"]))
    if passed:
        first = (f"有通過：{'、'.join(names[k].split('：')[0] for k in passed)}。"
                 f"{names[passed[0]]}的排名分數 {_share(s[passed[0]]['low'])}（50 日均線 {_share(m['low'])}），"
                 f"最深虧損中位數 {_pct(s[passed[0]]['c_deep_med'])}（50 日均線 {_pct(m['c_deep_med'])}），"
                 f"亂移對照 {_share(r['placebo'][passed[0]])}，也贏過 {L} 日均線雙胞胎。")
    else:
        why = []
        for k in TESTED:
            miss = [w for w, v in (("排名分數輸 50 日均線", ok[k]["score"]), ("最深虧損比 50 日均線深", ok[k]["deep"]),
                                   ("亂移對照超過 10%", ok[k]["placebo"]), (f"沒贏 {L} 日均線雙胞胎", ok[k]["twin"])) if not v]
            why.append(f"{names[k].split('：')[0]}（{'、'.join(miss)}）")
        first = ("三條短期持有者成本價規則都沒有通過事先定好的考試：" + "；".join(why) + "。")
    second = (f"「收盤高於成本價就持有」（H1）排名分數 {_share(h1['low'])}，50 日均線 {_share(m['low'])}；"
              f"{n_c:,} 格每年比抱著多 {h1['c_edge']:+.1f} 個百分點（50 日均線 {m['c_edge']:+.1f}），"
              f"最深虧損中位數 {_pct(h1['c_deep_med'])}（50 日均線 {_pct(m['c_deep_med'])}）。")
    third = (f"它很像一條普通的 {L} 日均線：只用 2019 年以前挑出的 {L} 日均線，和成本價的距離中位數只有 {r['gap_line'] * 100:.1f}%，"
             f"2012 年起 {_share(r['agree'])} 的日子兩者給出同樣的持有／現金，"
             f"{L} 日均線規則的排名分數 {_share(t1['low'])}、每年多 {t1['c_edge']:+.1f} 點。")
    h2 = s["h2"]
    fourth = (f"H2 和 50 日均線只有 {_share(r['differ']['h2'])} 的日子不同（2015 起），"
              f"{n_c:,} 格每年多 {h2['c_edge'] - m['c_edge']:+.1f} 點，但最深的一格 {_pct(h2['c_deep_worst'])}"
              f"（50 日均線 {_pct(m['c_deep_worst'])}）。")
    cold, hot = b[f"< {COLD}（套牢投降區）"]["全部"], b[f"> {HOT}（過熱區）"]["全部"]
    allb = b["全部日子"]["全部"]
    fifth = (f"短期持有者 MVRV 的極端值不像傳說那樣好用：低於 {COLD} 之後 90 天平均 {_p(cold['f90'][0])}、高於 {HOT} 之後 "
             f"{_p(hot['f90'][0])}，全部日子是 {_p(allb['f90'][0])}。")
    verdict = first + fourth + second + third + fifth

    look_c, look_b = r["look"]
    notes = [
        f"資料：Checkonchain 公開圖表檔裡的「STH Cost Basis」，{_d(r['cost'].dropna().index[0])} 起每天一筆，"
        f"本次用到 {_d(r['cost'].dropna().index[-1])}；不用金鑰、不用登入，用儀表板固定的 User-Agent 抓。"
        "儀表板自己的收集器用 BGeometrics，但它的免費版只給最近 4 年（它的 API 說明寫的），不夠測 2012–2021，所以這裡只拿來對照。",
        f"時間規則：成本價第 D 天的值，最早在 D+1 天的 UTC 0 點收盤才用；24 個收盤時間的格子裡，日期 E 的收盤只用到 E−2 天的值。"
        f"打亂切點之後的價格和成本價再算一次，切點以前的決定不能變：{look_c} 次檢查有 {look_b} 次變了。",
    ]
    if r["wb"]:
        w = r["wb"]
        notes.append(f"歷史會被改：拿 Wayback Machine 存的 {WB_DAY} 舊版和今天的版本比，2015 年起 {w['n']:,} 天裡成本價差距中位數 "
                     f"{w['med'] * 100:.2f}%、最大 {w['max'] * 100:.1f}%，差超過 1% 的日子 {_share(w['big'])}；"
                     f"用舊版算 H1，有 {w['flips'] * 100:.1f}% 的日子持有／現金會不一樣。舊版最後一個月的值後來最多改了 "
                     f"{(w['tail_max'] or 0) * 100:.1f}%。所以回測用的是「今天的版本」，當時的人看到的數字和這裡略有不同。")
    if r["rev"]:
        v = r["rev"]
        notes.append(f"和上次執行（{v['day']}）存下的版本比：{v['n']:,} 天裡最大差 {v['max'] * 100:.2f}%，差超過 1% 的日子 {_share(v['big'])}。")
    if r["bg"]:
        g = r["bg"]
        notes.append(f"對照 BGeometrics（儀表板收集器的來源，{_d(g['from'])} 起 {g['n']:,} 天）：它換算出的成本價比 Checkonchain "
                     f"中位數 {g['med'] * 100:+.1f}%（絕對差中位數 {g['abs'] * 100:.1f}%），收盤在線上／線下的判斷 {_share(g['agree'])} 的日子一樣。"
                     "兩家對「155 天」和交易所錢包的處理不同，所以數字不會完全一樣。")
    notes += [
        f"{L} 日均線是在 {TRAIN[0][:4]}–{TRAIN[1][:4]} 這段，從 20 到 400 天裡挑和成本價平均距離最小的（{L} 天的平均對數距離 "
        f"{r['fit'][L] * 100:.1f}%，50 天是 {r['fit'][50] * 100:.1f}%、155 天是 {r['fit'][155] * 100:.1f}%）；2020 年以後的資料沒有拿來挑。"
        f"T3 的門檻 {r['q']:.2f} 也只用這段：讓「收盤 ÷ {L} 日均線」超過它的日子比例，等於短期持有者 MVRV 超過 {HOT} 的比例（{_share(r['hot_share'])}）。",
        "兩條線一致的比例分段：" + "、".join(f"{n} {_share(v)}" for n, v in r["agree_per"]) + "。",
        "第四項檢查：成本價只有比特幣有免費的長歷史，沒有其他幣可以換，所以改用分段：2012–17、2018–21、2022 到現在，"
        f"每段裡每個月 1 日起算、只算到該段結束（最後一個起算日在結束前約半年），取三段裡「報酬贏過一直抱著」比例最低的那段。",
        f"排名分數和 main_signal_check 一樣是四項裡最低的那項；50 日均線用的是同樣的計算和亂數，所以第一到三項和 main_signal_check 的 50 日均線一致。"
        f"亂移對照：把短期持有者 MVRV 整條往後挪 {GAP} 天以上（頭尾相接），再用收盤價換回成本價，重抽 {PLACEBO} 次、整個考試重算；"
        f"為了在幾分鐘內算完，一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方式算一次再比。",
        f"H2 的「站回」是我照常見說法定的：50 日均線沒在持有時，收盤從成本價以下（含）站到上方就進場；之後 50 日均線轉成持有就交給它管，"
        "如果 50 日均線還沒轉多、收盤又跌回成本價下方就出場。",
        "STH-MVRV 分組是描述，不是規則：之後的報酬天天重疊，真正獨立的段落遠少於天數；方括號是區塊重抽的 90% 區間。",
        "成本很粗略：照課本做扣 0.10%、一般人做法扣 0.25%，沒有另外算滑價；現金沒有利息。一直抱著不扣成本。",
        "過去的結果不保證未來會一樣。這裡只是比較幾條固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    tables = [
        {"caption": (f"一、公平考試（main_signal_check 的做法）。每格「報酬贏過一直抱著的比例｜每年多幾個百分點（中位數）」："
                     f"比特幣 UTC 0 點日線 {msc.DAILY_STARTS[0][:7]} 到 {msc.DAILY_STARTS[1][:7]} 每月 1 日起算（{n_d} 個）、"
                     f"起算日 × 24 個收盤時間照課本做（{n_c:,} 格）、一般人做法（晚 10 小時、0.25%、漏掉 1 成進場，{n_p:,} 次模擬），"
                     "以及三段期間各自的比例（取代換幣）。排名分數＝四項裡最低的；"
                     f"「虧得比抱著淺」是兩組比特幣檢查裡較低的那組，至少 {LOSS_BAR * 100:.0f}% 才列入排名。持有天數和一年進場次數從 2015 起算。"),
         "columns": ["規則", f"日線起算日 {n_d} 個", f"起算日×收盤時間 {n_c:,} 格", "一般人做法", "三段期間：" + "｜".join(pnames),
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "虧得比抱著淺", "持有天數｜一年進場"],
         "rows": rows1},
        {"caption": ("二、分段看（照課本做、UTC 0 點日線）：每段裡每月 1 日起算、只算到該段結束。每格是「報酬贏過抱著的起算日｜每年多幾點（中位數）｜"
                     "最深虧損中位數（同段一直抱著）」。"),
         "columns": ["規則"] + pnames, "rows": rows2},
        {"caption": ("三、事先定好的通過條件：排名分數不低於 50 日均線、最深虧損中位數不比 50 日均線深、亂移對照不超過 10%、"
                     "而且要贏過同樣做法但把成本價換成普通均線的雙胞胎（先比排名分數，同分比每年多幾點）。四項都過才算通過。"),
         "columns": ["規則", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "亂移對照（≥ 真的比例）", "排名分數／每年多幾點 vs 雙胞胎", "結果"],
         "rows": rows3},
        {"caption": (f"四、短期持有者 MVRV（收盤 ÷ 前一天的成本價）落在哪一區，之後 30／90 天比特幣平均漲跌（2012 起）。"
                     f"常見說法：低於 {COLD} 是近期買家投降、可以買的區域；高於 {HOT} 是過熱。方括號是 90% 區間，段數＝連續落在同一區的段落數。"),
         "columns": ["區間", "天數｜段數", "之後 30 天平均", "之後 90 天平均", "之後 90 天：" + "｜".join(pnames)],
         "rows": rows4},
    ]
    return {
        "id": "sth_cost",
        "title": "短期持有者成本價：比 50 日均線好嗎？",
        "question": ("鏈上分析最常引用的「短期持有者成本價」（最近約 155 天內動過的幣的平均買進價），拿來當「價格在線上才持有」，"
                     "能不能贏過或補強儀表板的 50 日均線？它是不是只是一條換了名字的普通均線？「站回成本價提早進場」和短期持有者 MVRV 的極端值有沒有用？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Checkonchain：Short-Term Holder Cost Basis（含 STH Cost Basis 每日數列的公開圖表）", "url": COC_PAGE},
            {"name": f"Wayback Machine：Checkonchain STH-MVRV 圖表的 {WB_DAY} 存檔（比對歷史修正）", "url": WB_URL.replace("id_/", "/")},
            {"name": "BGeometrics（儀表板收集器的短期持有者 MVRV 來源，對照用）", "url": BG_PAGE},
            {"name": "CryptoSlate：短期持有者成本價是重要支撐", "url": SRC_CS},
            {"name": "Bitcoin Magazine：長期與短期持有者成本價分析", "url": SRC_BM},
            {"name": "Bitcoin Magazine Pro（Substack）：用訊號避險的回測", "url": SRC_BMP},
            {"name": "幣安現貨 BTC/USDT 日線與 1 小時線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
