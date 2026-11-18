"""Historical study "lead_followup": two leads that earlier studies found only AFTER looking at their results, now
tested with rules written down before this run, on main_signal_check's full fair exam.

Where the leads came from (both seen after the fact, so they could not count there):
  (L1) coinbase_premium: the dollar-to-dollar Coinbase premium (Coinbase BTC-USD / Bitstamp BTC/USD - 1), 7-day
       average > 0 -> next 30 days +5.9 points vs <= 0, same sign in 2017-19, 2020-22, 2023+; the rule CB1$ = ma50
       AND C7$ > 0 scored return / volatility 1.25 vs ma50 1.16 with a 2% placebo - but it was never run on the fair
       grid, the ordinary-person execution or other coins.
  (L2) attention: Kristoufek's own trend definition (price above its 7-day average) x Wikipedia attention rising ->
       next 30 days +6.1 points interaction, all three periods positive.

This cannot be truly out-of-sample: it is the same history. The bar is therefore the full fair exam plus other coins,
and the real test is the dashboard's daily record from now on.

Pre-registered rules (fixed before running; no other variants):
L1 rule: hold BTC when ma50 is on AND the 7-day average of the dollar-to-dollar Coinbase premium (known by the
previous UTC day) > 0.
L2 rule: hold BTC when ma50 is on, except step to cash when close < its 7-day average AND 7-day attention change < 0
(attention falling in a short-term dip); re-enter when that condition clears and ma50 is on.
Evaluate exactly as main_signal_check: 1,728 start x clock cells (as many starts as data allows - L1 from 2017-09,
L2 from 2016-09; state the counts), ordinary-person execution with main_signal_check's settings and seed, other
coins: ETH with its own Coinbase/Bitstamp premium (L1) and Ethereum article (L2); SOL/BNB/XRP/MSTR use BTC's signal
as a market-wide filter (state this). Ranking score per main_signal_check; median deepest loss; crash-avoided share
with whipsaws; placebo with circularly shifted signal (200 draws) on the ranking score, not just return / volatility.
Pass test (stated now): ranking score >= ma50's on the same starts AND median deepest loss no worse AND placebo share
<= 10%. If a rule passes, also report how many days it differs from ma50 and in which years its edge comes from.
Today: each rule's state - states only.

Details fixed with the design (before results):
  * Premium: exactly coinbase_premium's dollar-to-dollar version and loaders (Coinbase BTC-USD daily candles through
    kimchi_premium's store, Bitstamp BTC/USD; both candles open and close at UTC midnight). C7$ = 7-day mean, at least
    5 of 7 days; unknown = not > 0. On the UTC daily series the decision at the close of candle D (00:00 UTC of D+1)
    uses C7$ through candle D, the UTC day that has just finished (coinbase_premium's convention, the one the lead
    was found with); on the 24 closing hours a close on UTC date E uses C7$ through E-1; for MSTR (US close) a close
    on date D uses C7$ through D-1.
  * Attention: attention's definition, ΔA = log 7-day mean of daily user page views minus the same 7 days earlier
    (en "Bitcoin"; en "Ethereum" for ETH). Views of a day are published after it ends: the UTC daily decision at D
    uses views to D-1, a close on UTC date E on the clock grid uses views to E-2 (attention's conservative rule),
    MSTR uses views to D-1. Missing attention = not falling (plain ma50). "Close < its 7-day average": the average
    of the last 7 closes including this one, on the same close series as the rule.
  * Other coins: ETH uses its own premium (Coinbase ETH-USD / Bitstamp ETH/USD) for L1 and its own article for L2.
    SOL, BNB, XRP and MSTR use their own 50-day and 7-day averages but BTC's premium (L1) and BTC's article (L2) as a
    market-wide filter. Start months: main_signal_check's (first month in which all its rules have a value, MSTR
    from 2020-08-11, to three years ago), and not before both signals exist; ma50 on exactly the same starts.
  * Ranking score and eligibility: main_signal_check._summary's arithmetic (lowest of the four "beats holding" shares;
    eligible when the deepest loss is shallower than holding's in at least 95% of the bitcoin starts and cells).
    "Ranking score >= ma50's" also requires the rule to be eligible whenever ma50 is. ma50 is rebuilt with
    main_signal_check._targets and the same random seeds, so on the same starts it is main_signal_check's ma50.
    "Median deepest loss" = median over the 1,728 textbook cells (main_signal_check's loss ordering).
  * Placebo: the premium series (L1; ETH's premium shifted by the same number of days) or the page-view series (L2;
    Ethereum's views by the same number) rolled circularly by 180 ... (length - 180) days, 200 draws, the whole exam
    rebuilt; share of draws whose ranking score is at least the real rule's. To keep the run within minutes the
    ordinary-person part of the placebo uses 50 simulations per clock (same seeds) and the real rule is scored the
    same way for this comparison. Return / volatility from 2017-09-01 is shifted the same way (description only).
  * Crashes: crash_timing's bear markets (50% fall, doubling ends it) with the top after 2017-08-17; avoided share,
    whipsaws from 2017-09-01. Days that differ from ma50 and calendar-year returns from each rule's first start.
  * Look-ahead check: closes, premium and views after three cut dates scrambled -> no decision up to the cut may
    change, on the UTC daily series and on the clock grid's flags.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import attention as att
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import coinbase_premium as cp
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import liquidity_lead as ll
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
FIRST = cp.FIRST            # first Binance BTCUSDT day; the premium exists from here
RSTART = cp.RSTART          # 2017-09-01
GRID = {"l1": ("2017-09-01", msc.DAILY_STARTS[1]), "l2": ("2016-09-01", msc.DAILY_STARTS[1])}
DRAWS, GAP, PL_SIMS = 200, 180, 50
KI = {"ma50": 1, "l1": 70, "l2": 71}  # ma50 = main_signal_check's draws
# days between the flag's own date and the decision that may use it
LAG = {"l1": {"utc": 0, "clock": 1, "share": 1}, "l2": {"utc": 1, "clock": 2, "share": 1}}
RULES = ("l1", "l2")
NAME = {"ma50": "50 日均線（基準）",
        "l1": "L1：50 日均線，而且美元對美元 Coinbase 溢價 7 日平均 > 0 才持有",
        "l2": "L2：50 日均線，但收盤低於 7 日均線而且關注度一週在降時先下車"}
SHORT = {"ma50": "50 日均線", "l1": "L1 溢價為正", "l2": "L2 短線跌破＋關注下降"}
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")

_pct, _pp, _d, _num = cr._pct, cr._pp, cr._d, te._num


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _load() -> dict:
    problems = []
    close = te._daily_close()
    days = pd.date_range(FIRST, close.index[-1], freq="D")
    cb, n = km._coinbase()
    problems += [n] if n else []
    bs, n = cp._candles_bs("btcusd")
    problems += [n] if n else []
    eth_cb, n = cp._candles_cb("ETH-USD", "2017-08-01")
    problems += [n] if n else []
    eth_bs, n = cp._candles_bs("ethusd")
    problems += [n] if n else []
    other, more = msc._load_other()
    problems += more
    if "ETH" not in other:
        raise RuntimeError("拿不到以太幣日線")
    eth = other["ETH"][0]
    edays = pd.date_range(FIRST, eth.index[-1], freq="D")
    vb, n = att._views("btc")
    problems += [n] if n else []
    try:
        ve, n = att._views("eth")
        problems += [n] if n else []
    except Exception as exc:  # noqa: BLE001 - ETH's L2 then falls back to plain ma50, said in the notes
        ve = None
        problems.append(f"英文維基「Ethereum」瀏覽量這次取不到（{type(exc).__name__}），以太幣的 L2 只剩 50 日均線。")
    usd = cp._premium(cp._series(cb, days), cp._series(bs, days))
    eusd = cp._premium(cp._series(eth_cb, edays), cp._series(eth_bs, edays))
    return {"close": close, "days": days, "usd": usd, "eusd": eusd, "vb": vb, "ve": ve, "other": other,
            "cb": cp._series(cb, days), "bs": cp._series(bs, days), "problems": problems}


# ---------- signals ----------

def _c7(prem: pd.Series) -> pd.Series:
    return prem.rolling(7, min_periods=5).mean()


def _l1_flag(prem: pd.Series) -> pd.Series:
    """1 on UTC days whose 7-day average premium is > 0 (unknown = 0)."""
    return (_c7(prem) > 0).astype(float)


def _dA(v: pd.Series) -> pd.Series:
    """attention._features' ΔA: log 7-day mean views minus the same 7 days earlier (indexed by the views' day)."""
    a = np.log(v.rolling(7, min_periods=7).mean().clip(lower=1.0))
    return a - a.shift(7)


def _l2_flag(v: pd.Series | None) -> pd.Series:
    """1 on view days whose 7-day attention change is < 0 (unknown = 0)."""
    if v is None:
        return pd.Series(dtype=float)
    d = _dA(v)
    return (d < 0).astype(float).where(d.notna(), 0.0)


def _usable(flag: pd.Series, dates, lag: int) -> np.ndarray:
    if not len(flag):
        return np.zeros(len(dates))
    return flag.reindex(pd.DatetimeIndex(dates) - lag * DAY).fillna(0.0).values.astype(float)


def _pos(kind: str, ma: np.ndarray, dip: np.ndarray, flag: np.ndarray) -> np.ndarray:
    if kind == "ma50":
        return ma
    if kind == "l1":
        return ma * flag
    return ma * (1 - dip * flag)


def _dip(c: np.ndarray) -> np.ndarray:
    m7 = ct._mean(np.asarray(c, dtype=float), 7)
    return np.where(np.isnan(m7), 0.0, (c < m7).astype(float))


# ---------- the exam ----------

class _Exam:
    """main_signal_check's four checks for ma50, L1 and L2 with everything that does not depend on the filter
    precomputed once (the placebo rebuilds only the filters)."""

    def __init__(self, data: dict, price, b0, pre):
        close = data["close"]
        self.close = close
        self.btc_ma = msc._targets(close)["ma50"].values.astype(float)
        self.btc_dip = _dip(close.values)
        self.btc_change = close.pct_change().fillna(0.0).values
        self.daily = {}
        for k in ("l1", "l2"):
            starts, labels = msc._month_starts(close.index, *GRID[k])
            hr, hd = msc._suffix(msc._equity(self.btc_change), starts, YEAR)
            self.daily[k] = {"starts": starts, "labels": labels, "hold": (hr[:, 0], hd[:, 0])}
        # clock grid
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma, self.cdip = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(msc._targets(pd.Series(c)).fillna(0.0)["ma50"].values[npre:][:self.days].astype(float))
            self.cdip.append(_dip(c)[npre:][:self.days])
        # other assets
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, (c, since, year) in data["other"].items():
            tg = msc._targets(c, year)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, pd.Timestamp(since or c.index[0]),
                        pd.Timestamp(FIRST) + 7 * DAY)
            if key == "ETH":
                e7 = _c7(data["eusd"]).dropna()
                begin = max(begin, e7.index[0] + DAY) if len(e7) else begin
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "dip": _dip(c.values),
                               "change": change, "starts": starts, "labels": labels, "year": year,
                               "share": bool(since), "hold": (hr[:, 0], hd[:, 0])}

    # --- pieces ---
    def btc_pos(self, kind: str, flags: dict) -> np.ndarray:
        f = _usable(flags["btc"], self.close.index, LAG[kind]["utc"]) if kind != "ma50" else 0.0
        return _pos(kind, self.btc_ma, self.btc_dip, f)

    def daily_part(self, kind: str, flags: dict, grid: str) -> tuple:
        pos = self.btc_pos(kind, flags)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        r, dd = msc._suffix(msc._equity(net), self.daily[grid]["starts"], YEAR)
        return r[:, 0], dd[:, 0]

    def clock_part(self, kind: str, flags: dict, sims: int) -> dict:
        f = _usable(flags["btc"], self.m.dates, LAG[kind]["clock"]) if kind != "ma50" else 0.0
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = np.nan_to_num(_pos(kind, self.cma[h], self.cdip[h], f), nan=0.0).astype(float)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            pos = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        return {"text": (tr, td), "person": (pr, pdd)}

    def other_part(self, kind: str, flags: dict) -> dict:
        out = {}
        for key, o in self.other.items():
            if kind == "ma50":
                f = 0.0
            else:
                src = flags["eth"] if key == "ETH" else flags["btc"]
                f = _usable(src, o["close"].index, LAG[kind]["share" if o["share"] else "utc"])
            pos = _pos(kind, o["ma"], o["dip"], f)
            net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), o["change"], COST)
            r, dd = msc._suffix(msc._equity(net), o["starts"], o["year"])
            out[key] = (r[:, 0], dd[:, 0])
        return out

    # --- main_signal_check._summary's arithmetic for one rule ---
    def summary(self, kind: str, flags: dict, grid: str, sims: int = SIMS) -> dict:
        dr, dd = self.daily_part(kind, flags, grid)
        dh = self.daily[grid]["hold"]
        c = self.clock_part(kind, flags, sims)
        hr, hd = self.chold
        tr, td = c["text"]
        pr, pdd = c["person"]
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        o = self.other_part(kind, flags)
        each = {a: (len(o[a][0]), int((o[a][0] > self.other[a]["hold"][0]).sum()),
                    int((o[a][1] > self.other[a]["hold"][1]).sum()),
                    float(np.median(o[a][0] - self.other[a]["hold"][0]))) for a in o}
        x = {"d_n": len(dr), "d_beat": int((dr > dh[0]).sum()), "d_edge": float(np.median(dr - dh[0])),
             "d_shallow": int((dd > dh[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_p10": float(np.percentile(edge, 10)), "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= LOSS_BAR
        return x


def _score_ge(x: dict, base: dict) -> bool:
    """Ranked at least as high as base in main_signal_check's order (eligibility first, then the lowest share)."""
    if base["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not base["eligible"]:
        return True
    return x["low"] >= base["low"] - 1e-12


def _ratio(pos: np.ndarray, close: pd.Series) -> float | None:
    return cp._ratio(pd.Series(pos, index=close.index), close.pct_change().fillna(0.0))


def _years(pos: np.ndarray, close: pd.Series, first: str) -> dict:
    net = ll._window(ll._net(pd.Series(pos, index=close.index), close.pct_change().fillna(0.0), COST), first, None)
    m = ro._measure(net, YEAR)
    return m["years"] if m else {}


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: _Exam, flags: dict) -> tuple[int, int]:
    rng = np.random.default_rng(11)
    close = data["close"]
    total = bad = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        u = data["usd"]
        u2 = u.where(u.index <= c, u + rng.normal(0, 0.02, len(u)))
        v = data["vb"]
        v2 = v.where(v.index < c, v * np.exp(rng.normal(0, 1.0, len(v))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        dip2 = _dip(c2.values)
        m = close.index <= c
        for kind, fl, fl2 in (("l1", flags["l1"]["btc"], _l1_flag(u2)), ("l2", flags["l2"]["btc"], _l2_flag(v2))):
            a = _pos(kind, ex.btc_ma, ex.btc_dip, _usable(fl, close.index, LAG[kind]["utc"]))
            b = _pos(kind, ma2, dip2, _usable(fl2, close.index, LAG[kind]["utc"]))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[m], nan=-1), np.nan_to_num(b[m], nan=-1)))
            # clock grid: a close on UTC date E; scramble what is not yet known there
            j = int(ex.m.dates.searchsorted(c))
            known_from = c - (LAG[kind]["clock"] - 1) * DAY
            if kind == "l1":
                fl3 = _l1_flag(u.where(u.index < known_from, u + rng.normal(0, 0.02, len(u))))
            else:
                fl3 = _l2_flag(v.where(v.index < known_from, v * np.exp(rng.normal(0, 1.0, len(v)))))
            x0 = _usable(fl, ex.m.dates, LAG[kind]["clock"])[:j + 1]
            x1 = _usable(fl3, ex.m.dates, LAG[kind]["clock"])[:j + 1]
            total += 1
            bad += int(not np.array_equal(x0, x1))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    flags = {"l1": {"btc": _l1_flag(data["usd"]), "eth": _l1_flag(data["eusd"])},
             "l2": {"btc": _l2_flag(data["vb"]), "eth": _l2_flag(data["ve"])}}
    none = {"btc": pd.Series(dtype=float), "eth": pd.Series(dtype=float)}
    s = {"ma50_l1": ex.summary("ma50", none, "l1"), "ma50_l2": ex.summary("ma50", none, "l2"),
         "l1": ex.summary("l1", flags["l1"], "l1"), "l2": ex.summary("l2", flags["l2"], "l2")}

    # positions on the UTC daily series
    pos = {"ma50": ex.btc_pos("ma50", none), "l1": ex.btc_pos("l1", flags["l1"]), "l2": ex.btc_pos("l2", flags["l2"])}
    ratio = {k: _ratio(pos[k], close) for k in pos}
    hold_ratio = _ratio(np.ones(len(close)), close)

    # placebo
    real_small = {k: ex.summary(k, flags[k], k, PL_SIMS) for k in RULES}
    plc, plc_ratio, plc_low = {}, {}, {}
    for k, seed in (("l1", SEED + 7), ("l2", SEED + 8)):
        prng = np.random.default_rng(seed)
        src_b = data["usd"] if k == "l1" else data["vb"]
        src_e = data["eusd"] if k == "l1" else data["ve"]
        n = len(src_b)
        hits = hits_r = 0
        lows = []
        for sh in prng.integers(GAP, n - GAP, DRAWS):
            sb = pd.Series(np.roll(src_b.values, int(sh)), index=src_b.index)
            se = pd.Series(np.roll(src_e.values, int(sh) % len(src_e)), index=src_e.index) if src_e is not None else None
            if k == "l1":
                f2 = {"btc": _l1_flag(sb), "eth": _l1_flag(se)}
            else:
                f2 = {"btc": _l2_flag(sb), "eth": _l2_flag(se)}
            x = ex.summary(k, f2, k, PL_SIMS)
            lows.append(x["low"])
            hits += int(_score_ge(x, real_small[k]))
            rr = _ratio(ex.btc_pos(k, f2), close)
            hits_r += int(rr is not None and ratio[k] is not None and rr >= ratio[k])
        plc[k], plc_ratio[k], plc_low[k] = hits / DRAWS, hits_r / DRAWS, float(np.median(lows))

    # crashes, whipsaws, days differing, calendar years
    c_arr = close.values
    idx = close.index
    bears = [b for b in ct._bears(close) if idx[b["top"]] >= pd.Timestamp(FIRST)]
    crash = []
    for b in bears:
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for k in pos:
            row[k] = ct._crash(c_arr, [(1.0, pos[k])], b)["avoid"]
        crash.append(row)
    first = int(idx.searchsorted(pd.Timestamp(RSTART)))
    whip = {k: ct._whipsaws(c_arr, [(1.0, pos[k])], first) for k in pos}
    differ = {}
    for k in RULES:
        a = idx >= pd.Timestamp(GRID[k][0])
        p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos[k][a])
        differ[k] = (int((p0 != p1).sum()), int(a.sum()))
    years = {"hold": _years(np.ones(len(close)), close, GRID["l2"][0]),
             **{k: _years(pos[k], close, GRID["l2"][0]) for k in pos}}

    look = _lookahead(data, ex, flags)
    return {"ex": ex, "s": s, "flags": flags, "pos": pos, "ratio": ratio, "hold_ratio": hold_ratio, "plc": plc,
            "plc_ratio": plc_ratio, "plc_low": plc_low, "real_small": real_small, "crash": crash, "whip": whip,
            "differ": differ, "years": years, "look": look, "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s = data["close"], r["ex"], r["s"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base = {"l1": s["ma50_l1"], "l2": s["ma50_l2"]}
    n_c = s["l1"]["c_n"]
    n_starts = len(ex.cstarts)

    # ----- pass test -----
    res = {}
    for k in RULES:
        x, b = s[k], base[k]
        score_ok = _score_ge(x, b)
        loss_ok = x["c_deep_med"] >= b["c_deep_med"] - 1e-12
        plc_ok = r["plc"][k] <= 0.10
        res[k] = {"score": score_ok, "loss": loss_ok, "plc": plc_ok, "pass": score_ok and loss_ok and plc_ok}

    # ----- table 1: the fair exam -----
    rows1 = []
    for key, label in (("ma50_l1", f"{NAME['ma50']}，起算月同 L1"), ("l1", NAME["l1"]),
                       ("ma50_l2", f"{NAME['ma50']}，起算月同 L2"), ("l2", NAME["l2"])):
        x = s[key]
        ck = x["checks"]
        rows1.append([label, f"{_share(ck[0])}（{x['d_n']} 個）", _share(ck[1]), _share(ck[2]),
                      f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}",
                      f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows1.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    d1, d2 = ex.daily["l1"]["labels"], ex.daily["l2"]["labels"]
    t1 = {"caption": (f"main_signal_check 的公平考試，四項都是「報酬贏過同一天起一直抱著」的比例。第一項：比特幣 UTC 0 點日線每月 1 日起算，"
                      f"L1 從 {_d(d1[0])[:7]} 到 {_d(d1[-1])[:7]}（{len(d1)} 個，溢價 2017-08 才有）、L2 從 {_d(d2[0])[:7]} 到 "
                      f"{_d(d2[-1])[:7]}（{len(d2)} 個），50 日均線各用同一組起算月重算；第二項：{n_starts} 個起算月 × 24 個收盤時間 "
                      f"＝ {n_c:,} 格照課本做（成本 {COST * 100:.2f}%）；第三項：同樣 {n_c:,} 格照一般人方式做（收盤後 10 小時才做、成本 "
                      f"{P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次，亂數種子同 main_signal_check）；第四項：其他資產的起算月合計。"
                      f"排名分數＝四項裡最低的那項；前提是最深虧損比一直抱著淺的比例（比特幣兩組裡較低的）至少 {LOSS_BAR * 100:.0f}%。"
                      "最後兩欄是 1,728 格照課本｜一般人做法的中位數；每年多賺是比一直抱著多幾個百分點。"),
          "columns": ["規則", "比特幣起算月", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows1}

    # ----- table 2: other assets -----
    rows2 = []
    for a, o in ex.other.items():
        m = s["ma50_l1"]["o_each"].get(a)
        e1, e2 = s["l1"]["o_each"].get(a), s["l2"]["o_each"].get(a)
        hd_a = float(np.median(o["hold"][1]))
        src = "自己的溢價／自己的維基頁面" if a == "ETH" else "比特幣的溢價／比特幣的維基頁面"
        rows2.append([a, f"{_d(o['labels'][0])[:7]} 到 {_d(o['labels'][-1])[:7]}（{len(o['starts'])} 個）", src,
                      f"{m[1]} 個｜{_pp(m[3])}" if m else "—", f"{e1[1]} 個｜{_pp(e1[3])}" if e1 else "—",
                      f"{e2[1]} 個｜{_pp(e2[3])}" if e2 else "—",
                      f"{m[2]}｜{e1[2]}｜{e2[2]} 個（抱著中位數 {_pct(hd_a)}）" if m and e1 and e2 else "—"])
    rows2.append(["合計", f"{s['l1']['o_n']} 個", "—", f"{s['ma50_l1']['o_beat']} 個", f"{s['l1']['o_beat']} 個",
                  f"{s['l2']['o_beat']} 個", "—"])
    t2 = {"caption": ("第四項的明細：規則原封不動用在其他資產（幣安現貨日線 UTC 0 點收盤；MSTR 用 Yahoo 調整後收盤、2020-08-11 起、一年 252 天），"
                      "起算月和 main_signal_check 一樣（它全部規則都有值的第一個月起，到三年前），照課本做、成本 0.10%。以太幣用自己的 Coinbase÷Bitstamp "
                      "溢價和英文維基「Ethereum」；SOL、BNB、XRP、MSTR 沒有自己的對應資料，用比特幣的溢價和比特幣頁面當「整個市場」的過濾，"
                      "50 日均線和 7 日均線則用它自己的價格。各格是「報酬贏過一直抱著的起算月個數｜每年多賺中位數（百分點）」；"
                      "最後一欄是最深虧損比一直抱著淺的起算月個數（50 日均線｜L1｜L2）。"),
          "columns": ["資產", "起算月", "L1／L2 用的訊號", "50 日均線", "L1", "L2", "虧損比抱著淺：均線｜L1｜L2"],
          "rows": rows2}

    # ----- table 3: pass test and placebo -----
    rows3 = []
    for k in RULES:
        x, b, q = s[k], base[k], res[k]
        dd_, nn = r["differ"][k]
        rows3.append([NAME[k], f"{_share(x['low'])} 對 {_share(b['low'])} → {'過' if q['score'] else '不過'}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(b['c_deep_med'])} → {'過' if q['loss'] else '不過'}",
                      f"{_share(r['plc'][k])}（打亂後排名分數中位數 {_share(r['plc_low'][k])}，真的 {_share(r['real_small'][k]['low'])}）"
                      f" → {'過' if q['plc'] else '不過'}",
                      f"{_num(r['ratio'][k])}｜均線 {_num(r['ratio']['ma50'])}｜抱著 {_num(r['hold_ratio'])}；打亂後一樣好 {_share(r['plc_ratio'][k])}",
                      f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—",
                      "通過" if q["pass"] else "沒有通過"])
    t3 = {"caption": (f"事先寫好的過關標準：排名分數不低於同一組起算月的 50 日均線、{n_c:,} 格最深虧損中位數不比它深、隨機錯位一樣好的比例 ≤ 10%，"
                      f"三項都要過。隨機錯位：把溢價（L1，以太幣的溢價錯開同樣天數）或維基瀏覽量（L2）整串循環平移 {GAP} 天到「長度 − {GAP}」天，"
                      f"{DRAWS} 次，整套考試重做，排名分數和真的一樣好或更好的比例；為了在幾分鐘內算完，這一步的一般人做法每個收盤時間只模擬 "
                      f"{PL_SIMS} 次（種子相同），真的規則也用同樣方式算一次再比。報酬÷波動是 UTC 0 點日線 {RSTART} 起、照課本做，只是描述，"
                      "不算過關標準。最後一欄前是和 50 日均線持有狀態不同的天數（各自第一個起算月起）。"),
          "columns": ["規則", "排名分數：規則 對 均線", "最深虧損中位數：規則 對 均線", "隨機錯位一樣好（排名分數）",
                      f"報酬÷波動（{RSTART} 起）", "和均線不同的天數", "結果"],
          "rows": rows3}

    # ----- table 4: crashes, whipsaws, calendar years -----
    rows4 = []
    for b in r["crash"]:
        rows4.append([f"大跌 {_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）"), _pct(b["drop"] * 100)]
                     + [_share(b[k]) for k in ("ma50", "l1", "l2")])
    w = r["whip"]
    rows4.append([f"一年下車幾次｜其中 30 天內更高價買回（{RSTART[:7]} 起）", "—"]
                 + [f"{w[k]['exits']:.1f}｜{w[k]['whips']:.1f}" for k in ("ma50", "l1", "l2")])
    yrs = sorted(y for y in r["years"]["ma50"] if y >= 2017)
    room = max(16 - len(rows4), 0)
    for y in (yrs[-room:] if room else []):
        rows4.append([f"{y} 年報酬（{y} 年{'到目前' if y == yrs[-1] else '全年'}）", _pct(r["years"]["hold"].get(y))]
                     + [_pct(r["years"][k].get(y)) for k in ("ma50", "l1", "l2")])
    t4 = {"caption": ("上半：比特幣從高點收盤跌 50% 以上的段落（crash_timing 的定義，高點在 2017-08-17 以後），每格是規則躲掉這段跌幅的比例"
                      "（用規則自己在這段的漲跌算，中途反覆進出也算）；下車次數從 2017-09 算，「更高價買回」＝下車後 30 天內用更高的收盤價買回。"
                      "下半：UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，看規則和 50 日均線的差來自哪幾年；第二欄是那一年一直抱著。"),
          "columns": ["段落／年份", "跌幅／一直抱著", "50 日均線", "L1", "L2"], "rows": rows4}
    tables = [t1, t2, t3, t4]

    # ----- today -----
    flags = r["flags"]
    last = close.index[-1]
    c7 = _c7(data["usd"])
    c7k = c7.dropna()
    dA = _dA(data["vb"])
    use_day = last - DAY
    da_now = dA.get(use_day, np.nan) if use_day in dA.index else np.nan
    m7_now = float(close.iloc[-7:].mean())
    m50_now = float(close.iloc[-50:].mean())
    pos = r["pos"]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    e = ex.other.get("ETH")
    eth_txt = "—"
    if e is not None:
        ec = e["close"]
        e1 = _pos("l1", e["ma"], e["dip"], _usable(flags["l1"]["eth"], ec.index, LAG["l1"]["utc"]))
        e2 = _pos("l2", e["ma"], e["dip"], _usable(flags["l2"]["eth"], ec.index, LAG["l2"]["utc"]))
        eth_txt = f"50 日均線 {word(e['ma'][-1])}｜L1 {word(e1[-1])}｜L2 {word(e2[-1])}（{_d(ec.index[-1])} 收盤）"
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {m50_now:,.0f}、7 日均線 {m7_now:,.0f}"},
        {"label": "美元對美元 Coinbase 溢價 7 日平均（Coinbase ÷ Bitstamp − 1）",
         "value": (f"{float(c7k.iloc[-1]) * 100:+.3f}%（到 {_d(c7k.index[-1])}）；L1 的條件「> 0」{'成立' if float(c7k.iloc[-1]) > 0 else '不成立'}"
                   if len(c7k) else "—")},
        {"label": f"英文維基「Bitcoin」關注度一週變化（瀏覽量到 {_d(use_day)}）",
         "value": (f"{_pct((math.exp(da_now) - 1) * 100)}；收盤{'低於' if float(close.iloc[-1]) < m7_now else '不低於'} 7 日均線，"
                   f"L2 的「短線跌破而且關注下降」{'成立' if (float(close.iloc[-1]) < m7_now and da_now < 0) else '不成立'}")
         if _ok(da_now) else "—"},
        {"label": "比特幣：50 日均線｜L1｜L2", "value": f"{word(pos['ma50'][-1])}｜{word(pos['l1'][-1])}｜{word(pos['l2'][-1])}"},
        {"label": "以太幣（自己的溢價和維基頁面）", "value": eth_txt},
        {"label": "這幾格的意思", "value": "只描述兩條規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    def one(k: str) -> str:
        x, b = s[k], base[k]
        return (f"{'L1' if k == 'l1' else 'L2'} 排名分數 {_share(x['low'])}（均線 {_share(b['low'])}）、最深虧損中位數 "
                f"{_pct(x['c_deep_med'])}（{_pct(b['c_deep_med'])}）、隨機錯位 {_share(r['plc'][k])}，{'通過' if res[k]['pass'] else '沒有通過'}")
    verdict = (f"之前看了結果才發現兩條線索：美元對美元的 Coinbase 溢價 7 日平均為正才持有（L1）、短線跌破 7 日均線而且維基關注下降就先下車（L2）。"
               f"這次先把規則寫死，再用換起算日和收盤時間、一般人做法、其他幣、隨機錯位的完整公平考試測：{one('l1')}；{one('l2')}。"
               "用的仍是同一段歷史，不是真正的新資料，真正的考驗要看之後每天實際記下的結果。")
    if len(verdict) > 350:
        verdict = (f"之前看結果才發現的兩條線索（溢價為正才持有 L1、短線跌破且關注下降先下車 L2），事先寫死規則後做完整公平考試：{one('l1')}；"
                   f"{one('l2')}。仍是同一段歷史，真正的考驗要看之後每天的實際紀錄。")

    # ----- notes -----
    tot, bad = r["look"]
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好，沒有其他版本）：L1 rule: hold BTC when ma50 is on AND the 7-day average of the "
         "dollar-to-dollar Coinbase premium (known by the previous UTC day) > 0. L2 rule: hold BTC when ma50 is on, except step to "
         "cash when close < its 7-day average AND 7-day attention change < 0 (attention falling in a short-term dip); re-enter when "
         "that condition clears and ma50 is on. Evaluate exactly as main_signal_check: 1,728 start×clock cells (as many starts as "
         "data allows — L1 from 2017-09, L2 from 2016-09), ordinary-person execution with main_signal_check's settings and seed, "
         "other coins: ETH with its own Coinbase/Bitstamp premium (L1) and Ethereum article (L2); SOL/BNB/XRP/MSTR use BTC's "
         "signal as a market-wide filter. Ranking score per main_signal_check; median deepest loss; crash-avoided share with "
         "whipsaws; placebo with circularly shifted signal (200 draws) on the ranking score. Pass test (stated now): ranking "
         "score ≥ ma50's on the same starts AND median deepest loss no worse AND placebo share ≤ 10%."),
        ("線索從哪裡來：coinbase_premium 研究的美元對美元版「7 日平均為正」之後 30 天比非正多約 5.9 個百分點、三段同方向，CB1$ 報酬÷波動 1.25 對均線 1.16、"
         "隨機錯位 2%；attention 研究用論文的 7 日均線當趨勢時，不對稱約 +6.1 個百分點、三段都為正。兩者都是看了結果才注意到的，"
         "所以當時不能算數；這份研究把它們變成規則、先寫好標準再考。"),
        ("這不是真正的樣本外：資料還是同一段歷史，線索本來就是從這段歷史裡挑出來的，所以就算通過也會偏樂觀；這裡只能把考試做完整（換起算日、"
         "收盤時間、一般人做法、其他幣、隨機錯位）。真正的檢驗是從今天起每天把狀態記下來、幾年後再看。"),
        ("時間對齊：UTC 0 點日線在第 D 天收盤（D+1 的 0 點）做決定時，L1 用到第 D 天剛收完的溢價（兩家交易所同一刻收盤，coinbase_premium 找到線索時的做法）；"
         "換收盤時間時，落在 UTC 日期 E 的收盤只用到 E−1 的溢價；MSTR（美股收盤）用到前一天。L2 的瀏覽量要等那天結束才公布，UTC 0 點日線用到前一天、"
         "換收盤時間用到 E−2、MSTR 用到前一天。"
         f"偷看未來的檢查：在 {'、'.join(CUTS)} 三個切點之後把價格、溢價、瀏覽量換成亂數再算，切點以前的持有狀態和旗號 {tot} 組裡 "
         f"{tot - bad} 組完全沒變" + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        (f"50 日均線用 main_signal_check 自己的 _targets 和同樣的亂數種子重算，所以 {n_c:,} 格和一般人做法的數字就是它那份研究的 50 日均線；"
         "比特幣起算月因為溢價 2017-08、瀏覽量 2015-07 才有，只能分別從 2017-09、2016-09 起（main_signal_check 從 2015-01），所以第一項和它不同，"
         "表上兩條規則各自和同一組起算月的均線比。"),
        (f"SOL、BNB、XRP、MSTR 用比特幣的訊號當整個市場的過濾（它們沒有自己的 Coinbase÷Bitstamp 長歷史，維基頁面也不一樣），"
         "所以第四項大半其實是在問「比特幣的溢價／關注度能不能幫其他幣」，不是各自的訊號。"),
        ("Bitstamp 每天收盤價和 Coinbase 只差零點幾個百分點，Bitstamp 自己的雜訊很大（coinbase_premium 研究的核對），"
         "美元對美元溢價只能當粗略的指標；瀏覽量長期一路往下（搜尋摘要、手機 App、AI 助手），所以 L2 只用一週變化。"),
        ("相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據；大跌段只有寥寥幾次。"
         f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%，現金沒有利息、一直抱著不扣成本、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的兩條規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "lead_followup",
        "title": "兩條事後線索的事先登記複驗",
        "question": ("之前兩份研究看了結果才發現的線索——美元對美元的 Coinbase 溢價為正才持有、短線跌破 7 日均線而且維基關注下降就先下車——"
                     "事先把規則和過關標準寫死後，用完整的公平考試（換起算日、換收盤時間、一般人做法、其他幣、隨機錯位）再測一次，還站得住嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coinbase Exchange 公開日線（BTC-USD、ETH-USD）", "url": cp.CB_URL.format(product="BTC-USD") + "?granularity=86400"},
            {"name": "Bitstamp 公開 API：BTC/USD、ETH/USD 日線", "url": "https://www.bitstamp.net/api/#ohlc_data"},
            {"name": "維基媒體每日瀏覽量 API（英文「Bitcoin」、「Ethereum」）",
             "url": "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Bitcoin/daily/20150701/"
                    + common.today_utc().replace("-", "")},
            {"name": "Kristoufek (2013), BitCoin meets Google Trends and Wikipedia, Scientific Reports 3:3415",
             "url": "https://www.nature.com/articles/srep03415"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
