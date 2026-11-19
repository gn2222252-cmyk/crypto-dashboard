"""Historical study "gold_dual": dual momentum between gold (GLD) and bitcoin - hold bitcoin only when it has risen
AND beaten gold over the last few weeks, gold in the opposite case, otherwise cash.

The claim (Quantpedia, 2026, "Dual Momentum Allocation Between Physical Gold and Bitcoin (Digital Gold)"): every
Wednesday close hold bitcoin if its X-week return is above GLD's and above 0; hold GLD if GLD's X-week return is above
bitcoin's and above 0; otherwise cash; never both. Composite = equal mix of X = 4, 8 and 12 weeks. Volatility cap:
weight = min(1, 20% / annualised 12-week volatility of the asset held; weekly returns x sqrt(52)). Reported for
2018-12-31 ... 2026-04, no costs: composite 64.73% a year, max drawdown -49.40% (8-week alone 79.91%, -43.94%;
bitcoin held 46.66%, -77.49%; GLD 18.80%, -20.71%); vol-capped composite 12.01%, -12.27% (8-week 14.36%, -10.78%).
Novelty here: rotation only parked idle money in gold and gold_lead tested gold LEADING bitcoin (failed); the relative
condition "bitcoin must beat gold" has not been tested on this dashboard.

Pre-registered design (written before any result was seen; not changed afterwards):
  Data: GLD adjusted daily closes from Yahoo's public chart endpoint through gold_lead's loader and its stored copy
  (from 2010); bitcoin = te._daily_close() (00:00 UTC candles; pre-2017-08 closes relabelled on 2026-11-19 to the day
  the candle covers; no extra shift here). Point in time: the US close of day D is usable from the bitcoin candle
  labelled D, which closes at 00:00 UTC of D+1, three to four hours later (cross_market's convention). The decision
  is taken at the close of the bitcoin candle labelled Wednesday with GLD's Wednesday close; the new holding starts
  with Thursday (bitcoin's Thursday candle, GLD's Thursday close over Wednesday's). Quantpedia priced bitcoin at
  GLD's closing time instead - a 3-4 hour difference, stated. X-week return = close on the decision day over the
  close 7X calendar days earlier (GLD: last US close on or before that day).
  D1 = the composite as published (bitcoin / GLD / cash, each X a third), held from Wednesday to Wednesday with the
     weights drifting in between, re-set at each decision; costs 0 / 0.10% / 0.25% of the amount traded. Measured on
     the replication window 2019-01-01 ... 2026-04-30 and on the full available history (first day all three X and
     the 12-week volatility exist); also (description) from 2015-01-01 (the fair exam's first start) and after the
     publication (2026-05-01 to now, only a few months). The 8-week variant alone and buy-and-hold bitcoin / GLD /
     50-50 (re-set weekly) are reported as checks against the article's numbers.
  D2 = bitcoin-only version: bitcoin's share = the share of the three X for which the composite holds bitcoin (0, a
     third, two thirds or all), the rest cash. Compared with ma50 on the fair exam.
  D3 = D1 with the 20% volatility cap (per asset held, same weekly 12-week volatility, decided on the same Wednesday);
     compared with the cap alone on held bitcoin (no momentum) and with a fixed bitcoin share of the same volatility
     over the replication window (vol_sizing's fair comparison).
  Mechanism: A = absolute momentum alone, bitcoin-only (bitcoin's share = share of X with bitcoin's X-week return
  > 0). D2 minus A is exactly what "bitcoin must beat gold" adds. Description: the two-asset absolute version (bitcoin
  if its return > 0, else GLD if GLD's > 0, else cash) against D1, and for each X the weeks in which the relative
  condition vetoed bitcoin (bitcoin up but gold up more) with bitcoin's and GLD's average next-week return.
  Fair exam for ma50, D2 and A exactly as main_signal_check, as ls_ratio / lead_followup / gold_lead do: 108 monthly
  starts 2015-01 ... 2023-12 on the UTC daily closes (textbook, 0.10%), 72 monthly starts 2018-01 ... 2023-12 x 24
  closing hours (1,728 cells) textbook and ordinary person (decide at the clock, trade 10 hours later, 0.25%, 10% of
  entries missed, 200 simulations per clock, main_signal_check's seeds for ma50); on a clock the decision is the close
  falling on a UTC Wednesday, a close at 23:00 UTC or later reads GLD of that date, earlier clocks the date before.
  Other assets ETH / SOL / BNB / XRP (Binance daily) and MSTR (Yahoo, from 2020-08-11), each against GLD with its own
  X-week returns and its own ma50, monthly starts from the first month every rule has a value to three years ago.
  Positions are re-set to target every close in the exam (main_signal_check's engine), so the thirds pay a tiny daily
  re-set cost there. Ranking score = main_signal_check._summary's lowest of the four "beats holding" shares
  (eligible when the deepest loss is shallower than holding's in >= 95% of the bitcoin starts and cells); median
  deepest loss over the 1,728 textbook cells. Placebo: GLD's daily log returns rolled circularly by 180 ...
  (length - 180) days, 200 draws, GLD rebuilt and the whole exam redone for D2 (for time the ordinary-person part of
  the placebo uses 50 simulations per clock and the real D2 is scored the same way for this comparison).
  Pass test (stated now): D2's ranking score >= ma50's (eligibility first) AND D2's median deepest loss no worse
  than ma50's AND placebo share <= 10% AND relative momentum adds over absolute momentum alone, which means: D2's
  ranking score >= A's AND D2's median deepest loss no worse than A's AND D2 strictly better than A on at least one
  of ranking score, median deepest loss, or median yearly edge over holding on the 1,728 textbook cells.
  Look-ahead check: bitcoin and GLD after three cut dates scrambled -> no D1 / D2 / D3 weight up to the cut may
  change. Today: the last Wednesday's X-week returns, choices and weights - states only.

Only public, key-less sources with the dashboard's fixed user agent; a failed download falls back to the last stored
copy and says so. Descriptive and in-sample; not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import gold_lead as gl
from dashboard.studies2 import lead_followup as lf
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, LOSS_BAR = msc.YEAR, msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
XS = (4, 8, 12)
WED = 2
VOL_W, VOL_CAP = 12, 0.20
COSTS = (0.0, 0.001, 0.0025)
REP = ("2019-01-01", "2026-04-30")
FROM15 = "2015-01-01"
AFTER = "2026-05-01"
DRAWS, GAP, PL_SIMS = 200, 180, 50
KI = {"ma50": 1, "d2": 90, "abs": 91}  # ma50 = main_signal_check's draws
CLOCK_LAG_HOUR = gl.CLOCK_LAG_HOUR
CUTS = ("2016-06-30", "2020-03-15", "2024-06-30")
PAGE = "https://quantpedia.com/dual-momentum-allocation-between-physical-gold-and-bitcoin-digital-gold/"
QUOTED = {"d1": (64.73, -49.40), "d1_8": (79.91, -43.94), "d3": (12.01, -12.27), "d3_8": (14.36, -10.78),
          "btc": (46.66, -77.49), "gld": (18.80, -20.71), "half": (38.65, -50.99)}
NAME = {"ma50": "50 日均線（基準）",
        "d2": "D2：雙動能只做比特幣（比特幣 X 週漲而且贏黃金才持有，4／8／12 週各三分之一）",
        "abs": "A：只看比特幣自己 X 週是漲的（不和黃金比，4／8／12 週各三分之一）"}
SHORT = {"ma50": "50 日均線", "d2": "D2 雙動能（只做比特幣）", "abs": "A 只看自己漲跌"}
EXAM = ("ma50", "d2", "abs")

_pct, _pp, _d, _num = cr._pct, cr._pp, cr._d, te._num
_ok, _share = lf._ok, lf._share


# ---------- data ----------

def _load() -> dict:
    problems = []
    days, note = gl._stored("gld", gl._fetch_gld, 3000)
    if note:
        problems.append(f"GLD 日線{note}。")
    gld = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    close = te._daily_close()
    other, more = msc._load_other()
    problems += more
    return {"gld": gld, "close": close, "other": other, "problems": problems}


# ---------- signals ----------

def _asof(s: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    """Value of s at the last index on or before each t (NaN before the first)."""
    i = s.index.searchsorted(t, side="right") - 1
    v = s.values[np.clip(i, 0, None)].astype(float)
    v[i < 0] = np.nan
    return v


def _ret(s: pd.Series, dates: pd.DatetimeIndex, lag: int, x: int) -> np.ndarray:
    return _asof(s, dates - lag * DAY) / _asof(s, dates - (lag + 7 * x) * DAY) - 1


def _vol(s: pd.Series, dates: pd.DatetimeIndex, lag: int) -> np.ndarray:
    """Annualised volatility of the last 12 weekly returns (sqrt 52), as of each date."""
    lv = np.vstack([_asof(s, dates - (lag + 7 * k) * DAY) for k in range(VOL_W + 1)])
    r = lv[:-1] / lv[1:] - 1
    return np.std(r, axis=0, ddof=1) * math.sqrt(52)


def _vol12(s: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Spread of the 12 overlapping 12-week returns ending in the last 12 weeks, x sqrt(52) (post-hoc reading)."""
    r = np.vstack([_ret(s, dates - 7 * k * DAY, 0, VOL_W) for k in range(VOL_W)])
    return np.std(r, axis=0, ddof=1) * math.sqrt(52)


def _combo(b: dict, g: dict | None, kind: str) -> np.ndarray:
    """Share of the three X for which `kind` holds its asset (NaN while a needed return is unknown)."""
    parts = []
    for x in XS:
        bx = b[x]
        gx = g[x] if g is not None else None
        with np.errstate(invalid="ignore"):
            if kind == "abs":
                v, ok = bx > 0, ~np.isnan(bx)
            elif kind == "d2":
                v, ok = (bx > gx) & (bx > 0), ~np.isnan(bx) & ~np.isnan(gx)
            elif kind == "gld":
                v, ok = (gx > bx) & (gx > 0), ~np.isnan(bx) & ~np.isnan(gx)
            elif kind == "abs_g":  # two-asset absolute version: GLD only when bitcoin is not up
                v, ok = (bx <= 0) & (gx > 0), ~np.isnan(bx) & ~np.isnan(gx)
            else:
                raise ValueError(kind)
        parts.append(np.where(ok, v.astype(float), np.nan))
    return np.mean(parts, axis=0)


def _weekly(w: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Target after each close: the value decided on the latest Wednesday on or before it."""
    dec = np.asarray(dates.weekday == WED)
    return pd.Series(np.where(dec, w, np.nan)).ffill().values


class _Series:
    """One price series on its own dates, with the asset's X-week returns precomputed (gold enters later)."""

    def __init__(self, s: pd.Series, dates: pd.DatetimeIndex, lag):
        self.dates = dates
        self.lag = lag  # days GLD's close is read back (int or per-date array not needed: one per series)
        self.b = {x: _ret(s, dates, 0, x) for x in XS}

    def gold(self, gld: pd.Series) -> dict:
        return {x: _ret(gld, self.dates, self.lag, x) for x in XS}

    def target(self, kind: str, gld: pd.Series | None) -> np.ndarray:
        g = self.gold(gld) if kind != "abs" else None
        return _weekly(_combo(self.b, g, kind), self.dates)


# ---------- D1 / D3 simulation (weekly re-set, drift in between) ----------

def _sim(tb: np.ndarray, tg: np.ndarray, rb: np.ndarray, rg: np.ndarray, dec: np.ndarray, cost: float) -> np.ndarray:
    n = len(rb)
    net = np.zeros(n)
    hb = hg = 0.0
    for i in range(n):
        r = hb * rb[i] + hg * rg[i]
        base = 1 + r
        if abs(base) > 1e-12:
            hb, hg = hb * (1 + rb[i]) / base, hg * (1 + rg[i]) / base
        out = r
        if dec[i] and math.isfinite(tb[i]) and math.isfinite(tg[i]):
            out -= cost * (abs(tb[i] - hb) + abs(tg[i] - hg))
            hb, hg = tb[i], tg[i]
        net[i] = out
    return net


def _measure(net: np.ndarray, index: pd.DatetimeIndex, a, b=None) -> dict | None:
    s = pd.Series(net, index=index)
    m = (s.index >= pd.Timestamp(a)) & ((s.index <= pd.Timestamp(b)) if b else True)
    return ro._measure(s[m], YEAR)


def _d1(data: dict) -> dict:
    """Weights and daily net returns of D1, D3, the 8-week variants, the mechanism variants and the benchmarks."""
    close, gld = data["close"], data["gld"]
    idx = close.index
    ser = _Series(close, idx, 0)
    g = ser.gold(gld)
    dec = np.asarray(idx.weekday == WED)
    rb = close.pct_change().fillna(0.0).values
    gcal = pd.Series(_asof(gld, idx), index=idx)
    rg = gcal.pct_change().fillna(0.0).values
    capb = np.minimum(1.0, VOL_CAP / _vol(close, idx, 0))
    capg = np.minimum(1.0, VOL_CAP / _vol(gld, idx, 0))
    b8, g8 = {x: ser.b[8] for x in XS}, {x: g[8] for x in XS}
    w = {"d1": (_combo(ser.b, g, "d2"), _combo(ser.b, g, "gld")),
         "d1_8": (_combo(b8, g8, "d2"), _combo(b8, g8, "gld")),
         "abs2": (_combo(ser.b, None, "abs"), _combo(ser.b, g, "abs_g")),
         "d2": (_combo(ser.b, g, "d2"), np.zeros(len(idx))),
         "abs": (_combo(ser.b, None, "abs"), np.zeros(len(idx)))}
    w["d3"] = (w["d1"][0] * capb, w["d1"][1] * capg)
    w["d3_8"] = (w["d1_8"][0] * capb, w["d1_8"][1] * capg)
    # found AFTER seeing D3 miss the article (description only): the "12-week standard deviation" read as the spread
    # of the overlapping 12-week returns over the last 12 weeks, x sqrt(52)
    altb = np.minimum(1.0, VOL_CAP / _vol12(close, idx))
    altg = np.minimum(1.0, VOL_CAP / _vol12(gld, idx))
    w["d3alt"] = (w["d1"][0] * altb, w["d1"][1] * altg)
    w["cap"] = (capb, np.zeros(len(idx)))
    w["btc"] = (np.ones(len(idx)), np.zeros(len(idx)))
    w["gld"] = (np.zeros(len(idx)), np.ones(len(idx)))
    w["half"] = (np.full(len(idx), 0.5), np.full(len(idx), 0.5))
    valid = np.isfinite(w["d3"][0]) & np.isfinite(w["d3"][1])
    first = idx[int(np.argmax(valid & dec))] + DAY
    net = {}
    for k, (tb, tg) in w.items():
        for c in COSTS:
            net[(k, c)] = _sim(tb, tg, rb, rg, dec, c)
    ma = msc._targets(close)["ma50"].values.astype(float)
    net[("ma50", COST)], _ = te._run(np.concatenate([[np.nan], ma[:-1]]), rb, COST)
    return {"idx": idx, "w": w, "net": net, "first": first, "b": ser.b, "g": g, "dec": dec, "rb": rb, "rg": rg,
            "capb": capb, "capg": capg, "gcal": gcal}


def _vetoes(r: dict, a: str) -> dict:
    """For each X, weeks in which bitcoin was up but gold up more (the relative condition said no), from date a."""
    idx, dec = r["idx"], r["dec"]
    close_b = np.cumprod(1 + r["rb"])
    close_g = np.cumprod(1 + r["rg"])
    nb = np.full(len(idx), np.nan)
    ng = np.full(len(idx), np.nan)
    nb[:-7] = close_b[7:] / close_b[:-7] - 1  # next 7 days from the close of day i (Thursday ... Wednesday)
    ng[:-7] = close_g[7:] / close_g[:-7] - 1
    m0 = dec & np.asarray(idx >= pd.Timestamp(a)) & ~np.isnan(nb)
    out = {}
    for x in XS:
        b, g = r["b"][x], r["g"][x]
        with np.errstate(invalid="ignore"):
            veto = m0 & (b > 0) & (g >= b)
            keep = m0 & (b > 0) & (b > g)
            off = m0 & (b <= 0)
        out[x] = {"n": int(m0.sum()), "veto": int(veto.sum()), "keep": int(keep.sum()), "off": int(off.sum()),
                  "veto_b": float(np.mean(nb[veto]) * 100) if veto.any() else None,
                  "veto_g": float(np.mean(ng[veto]) * 100) if veto.any() else None,
                  "keep_b": float(np.mean(nb[keep]) * 100) if keep.any() else None,
                  "off_b": float(np.mean(nb[off]) * 100) if off.any() else None}
    return out


# ---------- the fair exam ----------

class _Exam:
    """main_signal_check's four checks for ma50, D2 and A (gold_lead's arithmetic)."""

    def __init__(self, data: dict, price, b0, pre):
        close = data["close"]
        self.close = close
        self.btc = _Series(close, close.index, 0)
        self.btc_ma = msc._targets(close)["ma50"].values.astype(float)
        self.btc_change = close.pct_change().fillna(0.0).values
        self.dstarts, self.dlabels = msc._month_starts(close.index, *msc.DAILY_STARTS)
        hr, hd = msc._suffix(msc._equity(self.btc_change), self.dstarts, YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, *msc.CLOCK_STARTS)
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma, self.cser = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(msc._targets(pd.Series(c)).fillna(0.0)["ma50"].values[npre:][:self.days].astype(float))
            full = pd.Series(c, index=pd.date_range(b0 - npre * DAY, periods=len(c), freq="D"))
            self.cser.append(_Series(full, m.dates, 0 if h >= CLOCK_LAG_HOUR else 1))
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, (c, since, year) in data["other"].items():
            tg = msc._targets(c, year)
            ser = _Series(c, c.index, 0)
            valid = tg.notna().all(axis=1).values & ~np.isnan(_combo(ser.b, ser.gold(data["gld"]), "d2"))
            if not valid.any():
                continue
            begin = max(c.index[int(np.argmax(valid))] + 7 * DAY, pd.Timestamp(since or c.index[0]))
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ser": ser, "ma": tg["ma50"].values.astype(float), "change": change,
                               "starts": starts, "labels": labels, "year": year, "hold": (hr[:, 0], hd[:, 0])}
        self.cache = {}

    def _btc_pos(self, kind: str, gld) -> np.ndarray:
        return self.btc_ma if kind == "ma50" else self.btc.target(kind, gld)

    def summary(self, kind: str, gld: pd.Series, sims: int = SIMS) -> dict:
        pos = self._btc_pos(kind, gld)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        r, dd = msc._suffix(msc._equity(net), self.dstarts, YEAR)
        dr, dd = r[:, 0], dd[:, 0]
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = self.cma[h] if kind == "ma50" else self.cser[h].target(kind, gld)[:self.days]
            w = np.nan_to_num(w, nan=0.0).astype(float)
            r, d2 = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], d2[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            p = rt._missed(wl, P_MISS, rng, sims)
            r, d2 = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, p, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, d2
        each = {}
        for key, o in self.other.items():
            po = o["ma"] if kind == "ma50" else o["ser"].target(kind, gld)
            net, _ = te._run(np.concatenate([[np.nan], po[:-1]]), o["change"], COST)
            r, d2 = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((d2[:, 0] > o["hold"][1]).sum()),
                         float(np.median(r[:, 0] - o["hold"][0])))
        hr, hd = self.chold
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_edge": float(np.median(dr - self.dhold[0])),
             "d_shallow": int((dd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= LOSS_BAR
        return x


def _adds(x: dict, a: dict) -> bool:
    """Pre-registered: D2 >= A on ranking score and median deepest loss, strictly better on at least one of ranking
    score, median deepest loss or median yearly edge on the 1,728 textbook cells."""
    if not lf._score_ge(x, a) or x["c_deep_med"] < a["c_deep_med"] - 1e-12:
        return False
    strict = (x["low"] > a["low"] + 1e-12 or (x["eligible"] and not a["eligible"])
              or x["c_deep_med"] > a["c_deep_med"] + 1e-12 or x["c_edge"] > a["c_edge"] + 1e-12)
    return bool(strict)


# ---------- the computation ----------

def _lookahead(data: dict) -> tuple[int, int]:
    rng = np.random.default_rng(11)
    close, gld = data["close"], data["gld"]
    base = _d1w(close, gld)
    total = bad = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        g2 = gld.where(gld.index <= c, gld * np.exp(rng.normal(0, 0.2, len(gld))))
        alt = _d1w(c2, g2)
        m = np.asarray(close.index <= c)
        for k in base:
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(base[k][m], nan=-1), np.nan_to_num(alt[k][m], nan=-1)))
    return total, bad


def _d1w(close: pd.Series, gld: pd.Series) -> dict:
    idx = close.index
    ser = _Series(close, idx, 0)
    g = ser.gold(gld)
    capb = np.minimum(1.0, VOL_CAP / _vol(close, idx, 0))
    capg = np.minimum(1.0, VOL_CAP / _vol(gld, idx, 0))
    wb, wg = _combo(ser.b, g, "d2"), _combo(ser.b, g, "gld")
    return {"d1b": _weekly(wb, idx), "d1g": _weekly(wg, idx), "d3b": _weekly(wb * capb, idx),
            "d3g": _weekly(wg * capg, idx)}


def _compute(data: dict) -> dict:
    t0 = time.time()
    r1 = _d1(data)
    veto = {"rep": _vetoes(r1, REP[0]), "15": _vetoes(r1, FROM15)}
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    gld = data["gld"]
    s = {k: ex.summary(k, gld) for k in EXAM}
    t1 = time.time()
    real_small = ex.summary("d2", gld, PL_SIMS)
    prng = np.random.default_rng(SEED + 12)
    n = len(gld)
    hits, lows = 0, []
    for sh in prng.integers(GAP, n - GAP, DRAWS):
        x = ex.summary("d2", gl._rebuilt(gld, int(sh)), PL_SIMS)
        lows.append(x["low"])
        hits += int(lf._score_ge(x, real_small))
    t2 = time.time()
    look = _lookahead(data)
    return {"r1": r1, "veto": veto, "ex": ex, "s": s, "plc": hits / DRAWS, "plc_low": float(np.median(lows)),
            "real_small": real_small, "look": look, "problem": problem, "secs": (t1 - t0, t2 - t1, time.time() - t0)}


# ---------- the study ----------

def _m3(m: dict | None) -> str:
    return "—" if not m else f"{_pct(m['ret'])}｜{m['vol']:.0f}%｜{_pct(m['deepest'])}"


def _m2(m: dict | None) -> str:
    return "—" if not m else f"{_pct(m['ret'])}｜{_pct(m['deepest'])}"


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, gld = data["close"], data["gld"]
    r1, ex, s = r["r1"], r["ex"], r["s"]
    idx, net = r1["idx"], r1["net"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    first = r1["first"]
    last = idx[-1]
    after_ok = last >= pd.Timestamp(AFTER) + 20 * DAY

    def meas(k: str, c: float, a, b=None):
        return _measure(net[(k, c)], idx, a, b)

    # ----- D3's fair comparison: fixed bitcoin share with the same volatility over the replication window -----
    d3m = meas("d3", 0.0, *REP)
    btcm = meas("btc", 0.0, *REP)
    fix = min(1.0, d3m["vol"] / btcm["vol"]) if d3m and btcm and btcm["vol"] > 0 else None
    fixm = {}
    if fix is not None:
        n_ = len(idx)
        for c in (0.0, COST):
            fnet = _sim(np.full(n_, fix), np.zeros(n_), r1["rb"], r1["rg"], r1["dec"], c)
            fixm[c] = {"rep": _measure(fnet, idx, *REP), "all": _measure(fnet, idx, first),
                       "15": _measure(fnet, idx, FROM15), "after": _measure(fnet, idx, AFTER) if after_ok else None}

    # ----- table 1: replication -----
    def cost_name(c: float) -> str:
        return "不扣成本" if c == 0 else f"扣 {c * 100:.2f}%"

    rows1 = []
    spec = [("d1", c, "D1：雙動能組合（比特幣／黃金／現金，4／8／12 週各三分之一）", "d1") for c in COSTS]
    spec += [("d1_8", 0.0, "D1 只用 8 週", "d1_8")]
    spec += [("d3", c, "D3：D1 加 20% 波動上限", "d3") for c in COSTS]
    spec += [("d3_8", 0.0, "D3 只用 8 週", "d3_8"),
             ("d3alt", 0.0, "D3 的另一種讀法（事後試的）：「12 週標準差」當成近 12 週裡 12 週報酬的離散程度", None),
             ("cap", 0.0, "對照：一直抱比特幣，只加同樣 20% 波動上限（不看動能）", None),
             ("btc", 0.0, "對照：一直抱比特幣", "btc"), ("gld", 0.0, "對照：一直抱黃金基金 GLD", "gld"),
             ("half", 0.0, "對照：比特幣、GLD 各半（每週三調回）", "half")]
    for k, c, label, q in spec:
        quoted = f"{QUOTED[q][0]:+.1f}%｜{QUOTED[q][1]:.1f}%" if q and c == 0 else "—"
        rows1.append([f"{label}（{cost_name(c)}）", quoted, _m3(meas(k, c, *REP)), _m3(meas(k, c, first)),
                      _m3(meas(k, c, FROM15)), _m2(meas(k, c, AFTER)) if after_ok else "—"])
    if fix is not None:
        rows1.append([f"對照：固定只拿 {fix * 100:.0f}% 比特幣（和 D3 同波動，每週三調回，不扣成本）", "—",
                      _m3(fixm[0.0]["rep"]), _m3(fixm[0.0]["all"]), _m3(fixm[0.0]["15"]),
                      _m2(fixm[0.0]["after"]) if after_ok else "—"])
    t1 = {"caption": (f"重做原文：每週三比特幣日線（UTC 0 點收盤，約在 GLD 美股收盤後 3–4 小時）收盤時決定，用同一天 GLD 的收盤，週四起照新比例持有，"
                      f"一週內比例隨漲跌漂移、下週三再調回。原文的比特幣價格取在 GLD 收盤那一刻，這裡晚 3–4 小時。各格是「每年報酬｜波動｜最深虧損」，"
                      f"一年用 365 天換算、現金沒有利息。第三欄是原文的樣本 {REP[0]} 到 {REP[1]}；全部歷史從 {_d(first)} 起（三種週數和 12 週波動都有值）；"
                      f"最後一欄是原文發表後 {AFTER} 起到 {_d(last)}（只有幾個月，只是描述）。"),
          "columns": ["規則", "原文（每年報酬｜最深虧損）", f"{REP[0][:7]} 到 {REP[1][:7]}", f"全部歷史（{_d(first)[:7]} 起）",
                      f"{FROM15[:4]} 起", f"發表後（{AFTER[:7]} 起）：報酬｜最深"],
          "rows": rows1}

    # ----- table 2: mechanism (whole strategies) -----
    rows2 = []
    for k, label in (("d1", "D1：雙動能（比特幣要漲而且贏黃金；黃金同理）"), ("abs2", "D1 拿掉「要贏對方」：比特幣漲就持有，不然黃金漲就持有黃金"),
                     ("d2", "D2：雙動能只做比特幣（其餘現金）"), ("abs", "A：比特幣漲就持有（不和黃金比，其餘現金）"),
                     ("ma50", "50 日均線（每天看，對照）"), ("btc", "一直抱比特幣")):
        rows2.append([label] + [_m3(meas(k, COST, a, b)) for a, b in (REP, (FROM15, None), (first, None))])
    t2 = {"caption": ("關鍵機制：把「比特幣一定要贏黃金」這個條件拿掉，看結果差多少。都扣 0.10%，週三決定（50 日均線每天決定）。"
                      "D2 和 A 唯一的差別就是比黃金這一步；D1 和第二列也是。各格「每年報酬｜波動｜最深虧損」。"),
          "columns": ["規則", f"{REP[0][:7]} 到 {REP[1][:7]}", f"{FROM15[:4]} 起", f"全部歷史（{_d(first)[:7]} 起）"],
          "rows": rows2}

    # ----- table 3: the weeks the relative condition vetoed bitcoin -----
    rows3 = []
    for per, label in (("15", f"{FROM15[:4]} 起"), ("rep", f"{REP[0][:7]} 起")):
        for x in XS:
            v = r["veto"][per][x]
            rows3.append([f"{label}，{x} 週", f"{v['n']} 週", f"{v['keep']} 週｜{_pct(v['keep_b'])}",
                          f"{v['veto']} 週（{v['veto'] / v['n'] * 100:.0f}%）" if v["n"] else "—",
                          f"{_pct(v['veto_b'])}｜{_pct(v['veto_g'])}", f"{v['off']} 週｜{_pct(v['off_b'])}"])
    t3 = {"caption": ("「要贏黃金」這一步實際擋掉了哪些週：比特幣 X 週是漲的、但黃金漲更多，所以雙動能不拿比特幣（只看自己漲跌的 A 會拿）。"
                      "後面是下一週（週四到下週三）的平均漲跌。如果被擋掉的那些週比特幣平均還是漲得比黃金多，這一步就是在扣分。"),
          "columns": ["期間，週數", "週三決定次數", "比特幣漲而且贏黃金：週數｜下週比特幣平均", "比特幣漲但輸黃金（被擋掉）",
                      "被擋掉那週：下週比特幣｜下週黃金平均", "比特幣沒漲：週數｜下週比特幣平均"],
          "rows": rows3}

    # ----- table 4: the fair exam -----
    rows4 = []
    for k in EXAM:
        x = s[k]
        ck = x["checks"]
        rows4.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", _share(ck[1]), _share(ck[2]),
                      f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows4.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    n_c = s["d2"]["c_n"]
    t4 = {"caption": (f"main_signal_check 的公平考試，四項都是「報酬贏過同一天起一直抱著」的比例：比特幣 UTC 0 點日線 {msc.DAILY_STARTS[0][:7]} 到 "
                      f"{msc.DAILY_STARTS[1][:7]} 每月 1 日起算（{len(ex.dstarts)} 個，照課本、成本 {COST * 100:.2f}%）；{len(ex.cstarts)} 個起算月 × 24 個收盤時間"
                      f"＝{n_c:,} 格照課本做；同樣 {n_c:,} 格照一般人方式做（收盤後 10 小時才做、成本 {P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次，"
                      "亂數種子同 main_signal_check）；其他資產（ETH、SOL、BNB、XRP、MSTR 各自和 GLD 比）的起算月合計。換收盤時間時，決定日是落在 UTC 週三的那次收盤，"
                      f"{CLOCK_LAG_HOUR}:00 UTC 以後的收盤用當天 GLD，更早的用前一天。排名分數＝四項裡最低的那項；前提是最深虧損比一直抱著淺的比例至少 "
                      f"{LOSS_BAR * 100:.0f}%。考試引擎每天把持有比例調回目標，所以三分之一、三分之二的部位每天有很小的調整成本。"),
          "columns": ["規則", "比特幣起算月", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows4}

    rows5 = []
    for a, o in ex.other.items():
        e = [s[k]["o_each"].get(a) for k in EXAM]
        rows5.append([a, f"{_d(o['labels'][0])[:7]} 到 {_d(o['labels'][-1])[:7]}（{len(o['starts'])} 個）"]
                     + [f"{v[1]} 個｜{_pp(v[3])}｜{v[2]} 個" if v else "—" for v in e])
    t5 = {"caption": ("第四項的明細：同樣的規則用在其他資產，各自的 X 週報酬和 GLD 比（MSTR 用 Yahoo 調整後收盤、2020-08-11 起、一年 252 天；"
                      "幣用幣安日線 UTC 0 點收盤），照課本做、成本 0.10%，起算月到三年前為止。各格「報酬贏過一直抱著的起算月個數｜每年多賺中位數｜"
                      "最深虧損比抱著淺的個數」。"),
          "columns": ["資產", "起算月"] + [SHORT[k] for k in EXAM], "rows": rows5}

    # ----- pass test -----
    x, b, a = s["d2"], s["ma50"], s["abs"]
    res = {"score": lf._score_ge(x, b), "loss": x["c_deep_med"] >= b["c_deep_med"] - 1e-12, "plc": r["plc"] <= 0.10,
           "adds": _adds(x, a)}
    res["pass"] = all(res.values())
    yes = {True: "過", False: "不過"}
    rows6 = [
        ["排名分數 ≥ 50 日均線", f"{_share(x['low'])} 對 {_share(b['low'])}", yes[res["score"]]],
        ["最深虧損中位數不比 50 日均線深", f"{_pct(x['c_deep_med'])} 對 {_pct(b['c_deep_med'])}", yes[res["loss"]]],
        [f"隨機錯位 GLD（{DRAWS} 次）一樣好的比例 ≤ 10%",
         f"{_share(r['plc'])}（錯位後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['real_small']['low'])}）", yes[res["plc"]]],
        ["「要贏黃金」比只看自己漲跌有加分（D2 對 A）",
         f"排名分數 {_share(x['low'])} 對 {_share(a['low'])}；最深虧損中位數 {_pct(x['c_deep_med'])} 對 {_pct(a['c_deep_med'])}；"
         f"每年多賺中位數 {_pp(x['c_edge'])} 對 {_pp(a['c_edge'])}", yes[res["adds"]]],
        ["結果", "四項都要過", "通過" if res["pass"] else "沒有通過"],
    ]
    t6 = {"caption": ("事先寫好的過關標準（D2）。隨機錯位：把 GLD 每天的漲跌整串循環平移 180 天到「長度 − 180」天、重建 GLD 後整套考試重做，"
                      f"看排名分數和真的一樣好或更好的比例；為了在幾分鐘內算完，這一步的一般人做法每個收盤時間只模擬 {PL_SIMS} 次（種子相同），"
                      "真的 D2 也用同樣方式算一次再比。因為 D2 只在「和黃金比」這一步用到黃金，錯位測的就是「真的黃金」有沒有比亂排的黃金好。"
                      "「有加分」＝排名分數和最深虧損中位數都不輸 A，而且排名分數、最深虧損中位數、每年多賺中位數至少一項比 A 好。"),
          "columns": ["標準", "數字", "結果"], "rows": rows6}
    tables = [t1, t2, t3, t4, t5, t6]

    # ----- today -----
    decs = np.where(r1["dec"])[0]
    j = int(decs[-1])
    dday = idx[j]
    nxt = dday + 7 * DAY
    now = [{"label": f"最近一次週三決定（比特幣 {_d(dday)} 的日線，UTC 0 點收盤；GLD 用 {_d(gld.index[gld.index <= dday][-1])} 收盤）",
            "value": f"比特幣 {float(close.iloc[j]):,.0f} 美元、GLD {float(_asof(gld, pd.DatetimeIndex([dday]))[0]):,.2f} 美元；下一次是 {_d(nxt)}"}]
    for xw in XS:
        bx, gx = float(r1["b"][xw][j]), float(r1["g"][xw][j])
        pick = "比特幣" if (bx > gx and bx > 0) else "黃金" if (gx > bx and gx > 0) else "現金"
        now.append({"label": f"{xw} 週漲跌：比特幣｜GLD → 雙動能選", "value": f"{_pct(bx * 100)}｜{_pct(gx * 100)} → {pick}"})
    wb, wg = float(r1["w"]["d1"][0][j]), float(r1["w"]["d1"][1][j])
    vb, vg = float(r1["w"]["d3"][0][j]), float(r1["w"]["d3"][1][j])
    now += [
        {"label": "D1 雙動能組合：比特幣｜黃金｜現金", "value": f"{wb * 100:.0f}%｜{wg * 100:.0f}%｜{(1 - wb - wg) * 100:.0f}%"},
        {"label": "D3 加 20% 波動上限：比特幣｜黃金｜現金",
         "value": f"{vb * 100:.0f}%｜{vg * 100:.0f}%｜{(1 - vb - vg) * 100:.0f}%（12 週年化波動：比特幣 "
                  f"{float(_vol(close, pd.DatetimeIndex([dday]), 0)[0]) * 100:.0f}%、"
                  f"GLD {float(_vol(gld, pd.DatetimeIndex([dday]), 0)[0]) * 100:.0f}%）"},
        {"label": "D2（只做比特幣）｜A（只看比特幣自己漲跌）｜50 日均線",
         "value": f"持有 {wb * 100:.0f}%｜持有 {float(r1['w']['abs'][0][j]) * 100:.0f}%｜"
                  f"{'持有' if ex.btc_ma[-1] >= 0.5 else '現金'}（50 日均線看到 {_d(last)}）"},
        {"label": "這幾格的意思", "value": "只描述規則在最近一次週三收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    rep0 = meas("d1", 0.0, *REP)
    rep3 = meas("d3", 0.0, *REP)
    d1ok = rep0 is not None and abs(rep0["ret"] - QUOTED["d1"][0]) <= 5 and abs(rep0["deepest"] - QUOTED["d1"][1]) <= 5
    d3ok = rep3 is not None and abs(rep3["ret"] - QUOTED["d3"][0]) <= 5 and abs(rep3["deepest"] - QUOTED["d3"][1]) <= 5
    verdict = (f"{'不加波動上限的版本重做得出來' if d1ok else '不加波動上限的版本和原文差得多'}："
               f"{REP[0][:7]} 到 {REP[1][:7]} 不扣成本，雙動能組合每年 {_pct(rep0['ret'])}、最深 {_pct(rep0['deepest'])}"
               f"（原文 +64.7%、-49.4%）；照原文文字做的 20% 波動上限{'也對得上' if d3ok else '對不上'}"
               f"（{_pct(rep3['ret'])}、{_pct(rep3['deepest'])}，原文 +12.0%、-12.3%）。"
               f"關鍵的「比特幣要贏黃金」這一步{'有' if res['adds'] else '沒有'}加分：只做比特幣時，雙動能 D2 排名分數 {_share(x['low'])}、"
               f"只看比特幣自己漲跌的 A {_share(a['low'])}，最深虧損中位數 {_pct(x['c_deep_med'])} 對 {_pct(a['c_deep_med'])}；"
               f"把黃金隨機錯位後 {_share(r['plc'])} 一樣好。D2 排名分數也遠低於 50 日均線的 {_share(b['low'])}，"
               f"{'通過' if res['pass'] else '沒有通過'}事先寫好的標準。")

    # ----- notes -----
    tot, bad = r["look"]
    vs = [v for v in r["veto"]["15"].values() if v["veto"]]
    notes = [
        ("事先寫死的設計（跑之前寫好，沒有其他版本）：D1 = composite as published (BTC / GLD / cash, X = 4, 8, 12 weeks, "
         "Wednesday), replication 2019-01 … 2026-04 and full history, costs 0 / 0.10% / 0.25%; D2 = BTC-only version; D3 = D1 with "
         "the 20% vol cap; A = absolute momentum alone (the mechanism question). Fair exam for D2 as main_signal_check, other "
         "assets ETH / SOL / BNB / XRP / MSTR each vs GLD, placebo = GLD returns circularly shifted ≥ 180 days, 200 draws. "
         "Pass: ranking ≥ ma50's AND median deepest loss no worse AND placebo ≤ 10% AND relative momentum adds over A."),
        ("時間對齊：GLD 週三美股收盤（台灣週四清晨）後 3–4 小時，比特幣 UTC 0 點的日線才收，所以比特幣週三那根日線收盤時已經知道 GLD 週三收盤"
         "（cross_market 的慣例）；新的比例從週四開始。原文把比特幣價格取在 GLD 收盤那一刻，和這裡差 3–4 小時。"
         f"偷看未來的檢查：在 {'、'.join(CUTS)} 三個切點之後把比特幣和 GLD 換成亂數再算，切點以前 D1、D3 的比例 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("「要贏黃金」擋掉的週：" + "；".join(
            f"{xw} 週 {r['veto']['15'][xw]['veto']} 次，之後一週比特幣平均 {_pct(r['veto']['15'][xw]['veto_b'])}、黃金 {_pct(r['veto']['15'][xw]['veto_g'])}"
            for xw in XS if r["veto"]["15"][xw]["veto"]) + f"（{FROM15[:4]} 起）。" if vs else "「要贏黃金」在 2015 年後沒有擋掉任何一週。"),
        (f"波動上限（D3）：照原文文字（選到的資產近 12 週每週漲跌的標準差 × √52，持有 = min(1, 20% ÷ 波動)）算，原文樣本的年化波動約 "
         f"{d3m['vol']:.0f}%，原文卻是 8.77%，報酬和最深虧損也差很多。看到對不上之後才試了另一種讀法（把「12 週標準差」當成近 12 週裡 12 週報酬的離散程度），"
         f"波動就和原文差不多，但報酬仍不同，見第一張表那一列；這是事後試的，不能當成重做成功。和 vol_sizing 的「同波動固定比例」比：固定只拿 "
         f"{fix * 100:.0f}% 比特幣和 D3 同波動，原文樣本每年 {_pct(fixm[0.0]['rep']['ret'])}、最深 {_pct(fixm[0.0]['rep']['deepest'])}，"
         f"D3 是 {_pct(d3m['ret'])}、{_pct(d3m['deepest'])}；只加波動上限、不看動能的比特幣是 {_pct(meas('cap', 0.0, *REP)['ret'])}、"
         f"{_pct(meas('cap', 0.0, *REP)['deepest'])}。所以 D3 的好處不只來自拿得少，雙動能（跌時不拿、可換到黃金）在這段時間也有貢獻。"
         if d3m and fix is not None else "波動上限的比較這次算不出來。"),
        (f"兩種「要贏對方」的效果不一樣：有黃金可以換時（D1 對第二張表第二列），{REP[0][:4]}–{REP[1][:4]} 每年 {_pct(meas('d1', COST, *REP)['ret'])} 對 "
         f"{_pct(meas('abs2', COST, *REP)['ret'])}，2015 起 {_pct(meas('d1', COST, FROM15)['ret'])} 對 {_pct(meas('abs2', COST, FROM15)['ret'])}，"
         "有一點加分；只做比特幣、其餘放現金時（D2 對 A），比黃金這一步反而少賺。也就是說，原文的好成績主要來自「比特幣跌時不拿」（絕對動能）"
         "和「黃金這幾年漲很多」，不是「比特幣要贏黃金」。"),
        ("D1 的部位一週只調一次，中間隨漲跌漂移；公平考試用 main_signal_check 的引擎，每天把比例調回目標，三分之一／三分之二的部位有很小的每日調整成本。"
         "其他資產都是今天還在的大幣和一檔股票，有倖存者偏誤。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"),
        (f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%；現金沒有利息、沒有算稅；一直抱著不扣成本。全部歷史從 2010 年起，"
         "早年比特幣漲幅極大，所以全部歷史的每年報酬會很高，只能看相對大小。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒（公平考試 {r['secs'][0]:.0f} 秒、隨機錯位 {r['secs'][1]:.0f} 秒）。",
        "過去的結果不保證未來會一樣。這裡只是重做一篇公開研究、把規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "gold_dual",
        "title": "黃金與比特幣的雙動能輪動",
        "question": ("每週三比較比特幣和黃金基金 GLD 過去 4、8、12 週的漲跌：比特幣漲而且漲贏黃金才拿比特幣、反過來拿黃金、都沒漲就現金。"
                     "公開研究說這樣每年 +64.7%、最深 -49.4%。重做得出來嗎？「比特幣一定要贏黃金」這一步，真的比只看比特幣自己漲跌有用嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Quantpedia：Dual Momentum Allocation Between Physical Gold and Bitcoin (Digital Gold)", "url": PAGE},
            {"name": "Yahoo Finance：GLD（SPDR 黃金基金）調整後日收盤", "url": "https://finance.yahoo.com/quote/GLD/history"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
