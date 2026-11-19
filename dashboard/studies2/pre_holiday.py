"""Historical study "pre_holiday": does bitcoin drift up around US stock-market (NYSE) holidays, as Quantpedia's
"Surprisingly Profitable Pre-Holiday Drift Signal for Bitcoin" (Cyril Dujava, 8 Sep 2025) says - especially when it
is at a short-term closing high? Equity background: Lakonishok & Smidt (1988) and Ariel (1990), the pre-holiday
effect in US stocks.

The claim as published (read once, recorded here): buy at the close of the day before a US holiday (D-1), hold through
the holiday, sell at the close of the day after (D+1). Unfiltered this "tends to be flat for most of the time".
Filter: only when the close is above the highest close of the preceding N trading days, N = 5, 10, 20; the filtered
versions do better, with "Sharpe ratios over 0.6 and Calmar ratio over 0.7" (the text ties the best numbers to the
5-day variant; the results table is an image). Data: a bitcoin futures proxy 2018-2021, then the BITO ETF, January
2018 - June 2025, in-sample, no trading costs mentioned. Holidays: a "consensus U.S. calendar" (Time and Date).

Pre-registered design (written before running, kept verbatim):
Map "close of the day before the holiday" to BTC's 00:00 UTC close nearest to the US close (state the mapping; also
test the 21:00 UTC hourly close on Binance from 2017-08 as the closer match).
H0 unfiltered: long from close D-1 to close D+1 around each NYSE holiday.
H1 Quantpedia filter: same, only when close D-1 > max of prior N closes, N in {5, 10, 20} - N=10 is the primary
(pre-stated); others description.
Tests: event returns vs same-weekday non-holiday windows (placebo, block bootstrap), split 2013-2017 (before the
Quantpedia sample), 2018-mid-2025 (their sample), mid-2025-now (after publication); costs 0.10/0.30% per trade; as an
overlay (add exposure only on those days when ma50 is out - i.e. "ma50 OR H1") vs ma50 on the fair grid; other coins
(ETH/SOL/BNB/XRP) with the same dates.
Pass test (stated now): H1 primary has the same sign in all three periods (especially before 2018 and after
mid-2025) with full-sample 90% interval excluding zero after 0.10% cost AND the overlay's ranking score >= ma50's AND
placebo share <= 10%.
Today: next NYSE holiday and whether H1's condition would be on at the current price - states only.

How the tests are carried out (fixed with the design, no tuning afterwards):
- Holidays: full NYSE closures 2013-2026, hard-coded below from NYSE's calendar pages and NYSE/ICE holiday
  announcements, including the two one-day closures for national days of mourning (2018-12-05, 2025-01-09).
  D-1 = the last NYSE trading day before the closure, D+1 = the first one after it (a Monday holiday: Friday to
  Tuesday). Early-close days (13:00 New York) are a separate flag: an event whose D-1 closed early.
- Mapping: the dashboard's daily close labelled D-1 (te._daily_close: the price at 00:00 UTC at the end of that UTC
  day) is the 00:00 UTC close nearest the 16:00 New York close of D-1 (3-4 hours later, never earlier). Exit = the
  close labelled D+1. Second version: Binance hourly price at 21:00 UTC on D-1 and D+1 (16:00 New York in winter,
  one hour after the close in summer), from 2017-08.
- Filter: the entry close > the highest of the N closes before it on the same clock (24/7 calendar days, not NYSE
  trading days).
- Costs: 0.10% (and 0.30%) of the amount moved at each buy and each sell, so one round trip costs about 0.20%
  (0.60%). The pass test uses 0.10%.
- Placebo: for each event, the same window (same weekday, same number of days) moved 1-26 weeks earlier or later,
  where neither the window nor 3 days around it touches any holiday window. For H1 the placebo windows must also meet
  the filter at their own entry close (so it asks whether the holiday adds anything to "an N-day high"). Excess =
  event return minus the average of its placebo windows; 90% interval: moving-block bootstrap (blocks of 4 events,
  2,000 draws, macro_days._boot). Placebo share = share of 2,000 draws (each event replaced by one of its own placebo
  windows at random, fixed seed) whose average return is at least the real average.
- "Same sign in all three periods" uses the average return per trade after 0.10% cost in each period; the 90%
  interval is the whole sample's (bootstrap of the after-cost trade returns).
- Overlay: ma50 (close above its 50-day average, main_signal_check) OR the H1 (N=10) holding window, i.e. holding
  from the entry close to the exit close whenever H1 is on. Scored exactly as main_signal_check's ranking: bitcoin
  108 monthly start dates 2015-01..2023-12 (daily, textbook); 72 start dates x 24 closing hours (hourly, textbook);
  the same cells for an ordinary person (trade 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations,
  the same seed as main_signal_check's ma50 so both rules see the same missed days); ETH/SOL/BNB/XRP/MSTR. Score =
  the lowest of the four "beats holding" shares; loss gate as there. On a clock h the entry is the first close of
  that clock at or after 21:00 UTC on D-1, the exit the same on D+1.
- Other coins: Binance daily closes, same dates and mapping as bitcoin's 00:00 UTC version.
Descriptive and in-sample; not advice. Only public, key-less sources through the dashboard's helpers.
"""
from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import macro_days as md
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, COST_HIGH = 0.001, 0.003
NS = (5, 10, 20)
PRIMARY = 10
DRAWS = 2000
SEED = 20250908
WEEKS = 26
CLEAR_DAYS = 3
YEAR = 365
NY_CLOSE_UTC = 21  # 16:00 New York in winter = 21:00 UTC
PERIODS = (("2013–2017（研究期間以前）", "2013-01-01", "2017-12-31"),
           ("2018-01～2025-06（研究期間）", "2018-01-01", "2025-06-30"),
           ("2025-07 以後（研究期間以後）", "2025-07-01", "2099-12-31"))
COINS = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
QP_URL = "https://quantpedia.com/surprisingly-profitable-pre-holiday-drift-signal-for-bitcoin/"
NYSE_URL = "https://www.nyse.com/markets/hours-calendars"

# Full NYSE closures (MMDD). Generated from NYSE's holiday rules and checked against: nyse.com/markets/hours-calendars
# (2026 list, seen 2026-11-19), the archived NYSE calendar page of January 2017 (2016-2018), ICE/NYSE announcements of
# the 2015 and 2018-2019 calendars, QuantConnect's NYSE table (2013-2014). New Year's Day on a Saturday (2022) is not
# observed. Juneteenth from 2022. 1205 in 2018 and 0109 in 2025: national days of mourning (Presidents Bush, Carter).
HOLIDAY_TEXT = {
    2013: "0101 0121 0218 0329 0527 0704 0902 1128 1225",
    2014: "0101 0120 0217 0418 0526 0704 0901 1127 1225",
    2015: "0101 0119 0216 0403 0525 0703 0907 1126 1225",
    2016: "0101 0118 0215 0325 0530 0704 0905 1124 1226",
    2017: "0102 0116 0220 0414 0529 0704 0904 1123 1225",
    2018: "0101 0115 0219 0330 0528 0704 0903 1122 1205 1225",
    2019: "0101 0121 0218 0419 0527 0704 0902 1128 1225",
    2020: "0101 0120 0217 0410 0525 0703 0907 1126 1225",
    2021: "0101 0118 0215 0402 0531 0705 0906 1125 1224",
    2022: "0117 0221 0415 0530 0620 0704 0905 1124 1226",
    2023: "0102 0116 0220 0407 0529 0619 0704 0904 1123 1225",
    2024: "0101 0115 0219 0329 0527 0619 0704 0902 1128 1225",
    2025: "0101 0109 0120 0217 0418 0526 0619 0704 0901 1127 1225",
    2026: "0101 0119 0216 0403 0525 0619 0703 0907 1126 1225",
}
# Early closes at 13:00 New York (day before Independence Day, day after Thanksgiving, Christmas Eve when trading).
EARLY_TEXT = {
    2013: "0703 1129 1224", 2014: "0703 1128 1224", 2015: "1127 1224", 2016: "1125", 2017: "0703 1124",
    2018: "0703 1123 1224", 2019: "0703 1129 1224", 2020: "1127 1224", 2021: "1126", 2022: "1125",
    2023: "0703 1124", 2024: "0703 1129 1224", 2025: "0703 1128 1224", 2026: "1127 1224",
}
HOLIDAY_SOURCES = [
    {"name": "NYSE 交易時間與休市日曆（2026–2028 休市日、提早收盤日）", "url": NYSE_URL},
    {"name": "NYSE 休市日曆網頁 2017-01 存檔（UNT 網頁典藏，2016–2018）",
     "url": "https://webarchive.library.unt.edu/web/20170117004636/https://www.nyse.com/markets/hours-calendars"},
    {"name": "ICE／NYSE：公布 2015 年休市與提早收盤日",
     "url": "https://ir.theice.com/press/news-details/2013/NYSE-Euronext-Announces-2015-Holiday-Calendar-and-Early-Closings/default.aspx"},
    {"name": "ICE／NYSE：公布 2018、2019 年休市與提早收盤日",
     "url": "https://ir.theice.com/press/news-details/2016/NYSE-Group-announces-2018-and-2019-holiday-and-early-closings-calendar/default.aspx"},
    {"name": "QuantConnect：美股交易時間與休市表（依 NYSE 日曆，2013–2014 對照）",
     "url": "https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/market-hours"},
]


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v, digits: int = 2) -> str:
    """v is a fraction."""
    if not _ok(v):
        return "—"
    text = f"{float(v) * 100:+.{digits}f}%"
    return text.replace("-", "+", 1) if float(text[:-1]) == 0 else text


def _share(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _ci(ci) -> str:
    return "—" if not ci else f"{_p(ci[0])}～{_p(ci[1])}"


def _share1(v) -> str:
    return f"{float(v) * 100:.1f}%" if _ok(v) else "—"


def _num(v, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}" if _ok(v) else "—"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- calendar ----------

def _days(text: dict) -> list[date]:
    return sorted(date(y, int(m[:2]), int(m[2:])) for y, s in text.items() for m in s.split())


def _calendar() -> tuple[list[dict], set]:
    """Holiday events: holiday date(s), entry day (D-1) and exit day (D+1) as NYSE trading days."""
    closed = set(_days(HOLIDAY_TEXT))
    early = set(_days(EARLY_TEXT))

    def trading(d: date) -> bool:
        return d.weekday() < 5 and d not in closed

    events = []
    for h in sorted(closed):
        if events and events[-1]["entry"] < h < events[-1]["exit"]:
            continue  # part of the same closure
        e = h - timedelta(days=1)
        while not trading(e):
            e -= timedelta(days=1)
        x = h + timedelta(days=1)
        while not trading(x):
            x += timedelta(days=1)
        events.append({"holiday": h, "entry": e, "exit": x, "early": e in early, "year": h.year})
    return events, early


def _blocked(events: list[dict]) -> set:
    out = set()
    for ev in events:
        d = ev["entry"] - timedelta(days=CLEAR_DAYS)
        while d <= ev["exit"] + timedelta(days=CLEAR_DAYS):
            out.add(d)
            d += timedelta(days=1)
    return out


def _period(d: date) -> int:
    s = d.isoformat()
    for i, (_, a, b) in enumerate(PERIODS):
        if a <= s <= b:
            return i
    return -1


# ---------- one price series ----------

class _Series:
    """Closes on a date index (bitcoin labelled by UTC day; the 21:00 UTC clock labelled by its date)."""

    def __init__(self, close: pd.Series):
        self.c = close.astype(float)
        self.pos = {ts.date(): i for i, ts in enumerate(self.c.index)}
        self.v = self.c.values
        self.prior = {n: self.c.shift(1).rolling(n, min_periods=n).max().values for n in NS}

    def ret(self, e: date, x: date) -> float | None:
        i, j = self.pos.get(e), self.pos.get(x)
        if i is None or j is None:
            return None
        return float(self.v[j] / self.v[i] - 1)

    def on(self, e: date, n: int | None) -> bool | None:
        if n is None:
            return True
        i = self.pos.get(e)
        if i is None or not _ok(self.prior[n][i]):
            return None
        return bool(self.v[i] > self.prior[n][i])


def _net(r, cost: float):
    return (1 + np.asarray(r)) * (1 - cost) ** 2 - 1


def _event_study(s: _Series, events: list[dict], blocked: set, n: int | None, rng) -> dict:
    """Trades of one variant, their placebo windows, intervals and periods."""
    rows = []
    for ev in events:
        on = s.on(ev["entry"], n)
        r = s.ret(ev["entry"], ev["exit"])
        if on is None or r is None or not on:
            continue
        cands, cands_on = [], []
        for k in list(range(-WEEKS, 0)) + list(range(1, WEEKS + 1)):
            e2, x2 = ev["entry"] + timedelta(days=7 * k), ev["exit"] + timedelta(days=7 * k)
            span = [e2 + timedelta(days=i) for i in range((x2 - e2).days + 1)]
            if any(d in blocked for d in span):
                continue
            r2, on2 = s.ret(e2, x2), s.on(e2, n)
            if r2 is None or on2 is None:
                continue
            cands.append(r2)
            if on2:
                cands_on.append(r2)
        pool = cands_on if (n is None or len(cands_on) >= 3) else cands
        rows.append({"ev": ev, "r": r, "pool": np.array(pool), "fallback": n is not None and pool is cands})
    out = {"n": len(rows), "rows": rows}
    if len(rows) < 2:
        return out
    r = np.array([x["r"] for x in rows])
    net, net3 = _net(r, COST), _net(r, COST_HIGH)
    with_pool = [x for x in rows if len(x["pool"])]
    plc = np.array([x["pool"].mean() for x in with_pool])
    ex = np.array([x["r"] for x in with_pool]) - plc
    real = float(np.mean([x["r"] for x in with_pool])) if with_pool else None
    if with_pool:
        sums = np.zeros(DRAWS)
        for x in with_pool:
            sums += x["pool"][rng.integers(0, len(x["pool"]), size=DRAWS)]
        draws = sums / len(with_pool)
        plc_share = float((draws >= real).mean())
    else:
        plc_share = None
    out.update({"mean": float(r.mean()), "up": float((net > 0).mean()), "net": float(net.mean()),
                "net_ci": md._boot(net, rng), "net3": float(net3.mean()), "plc": float(plc.mean()) if len(plc) else None,
                "excess": float(ex.mean()) if len(ex) else None, "ex_ci": md._boot(ex, rng) if len(ex) else None,
                "plc_share": plc_share, "fallbacks": sum(x["fallback"] for x in rows),
                "early": [float(_net(x["r"], COST)) for x in rows if x["ev"]["early"]]})
    per = []
    for i in range(len(PERIODS)):
        sub = [x for x in rows if _period(x["ev"]["holiday"]) == i]
        nn = _net(np.array([x["r"] for x in sub]), COST) if sub else np.array([])
        exs = [x["r"] - x["pool"].mean() for x in sub if len(x["pool"])]
        per.append({"n": len(sub), "net": float(nn.mean()) if len(nn) else None, "ci": md._boot(nn, rng),
                    "up": float((nn > 0).mean()) if len(nn) else None,
                    "excess": float(np.mean(exs)) if exs else None})
    out["periods"] = per
    return out


def _sharpe(s: _Series, events: list[dict], n: int | None, first: str, last: str) -> float | None:
    """Yearly return / volatility of holding only in the windows (cash otherwise), no cost, on the same calendar
    days as the series (365 a year) - the measure Quantpedia quotes, on bitcoin itself."""
    w = _window(s.c.index, events, s, n)
    change = s.c.pct_change().fillna(0.0).values
    net, _ = te._run(np.concatenate([[0.0], w[:-1]]), change, 0.0)
    net = pd.Series(net, index=s.c.index)[first:last]
    if len(net) < 100 or net.std(ddof=1) <= 0:
        return None
    return float(net.mean() / net.std(ddof=1) * math.sqrt(YEAR))


def _window(index: pd.DatetimeIndex, events: list[dict], s: _Series, n: int | None) -> np.ndarray:
    """1 after the closes labelled from the entry day up to (not including) the exit day when the rule is on."""
    w = np.zeros(len(index))
    pos = {ts.date(): i for i, ts in enumerate(index)}
    for ev in events:
        i, j = pos.get(ev["entry"]), pos.get(ev["exit"])
        if i is None or j is None:
            if i is not None and s.on(ev["entry"], n):
                w[i:] = 1.0  # still open at the end of the data
            continue
        if s.on(ev["entry"], n):
            w[i:j] = 1.0
    return w


# ---------- overlay on main_signal_check's fair grid ----------

def _daily_grid(close: pd.Series, year: int, first, last, events: list[dict]) -> dict:
    """msc._daily for ma50 and ma50 OR H1(N=10) only."""
    target = msc._targets(close, year)
    s = _Series(close)
    win = _window(close.index, events, s, PRIMARY)
    over = np.maximum(target["ma50"].values, win)
    over = pd.Series(over, index=close.index).where(target["ma50"].notna())
    change = close.pct_change().fillna(0.0).values
    valid = target.notna().all(axis=1)
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k, t in (("ma50", target["ma50"]), ("over", over)):
        net, trade = te._run(t.shift(1).values, change, msc.COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
        out[k + "_days"] = float(np.nan_to_num(t.shift(1).values)[starts[0]:].mean())
    return out


def _clock_window(h: int, closes: np.ndarray, npre: int, b0: pd.Timestamp, days: int, events: list[dict]) -> np.ndarray:
    w = np.zeros(days)
    base = b0.date()
    prior = pd.Series(closes).shift(1).rolling(PRIMARY, min_periods=PRIMARY).max().values

    def j_of(d: date) -> int:
        return (d - base).days + (0 if h >= NY_CLOSE_UTC else 1)

    for ev in events:
        je, jx = j_of(ev["entry"]), j_of(ev["exit"])
        idx = npre + je
        if je < 0 or idx >= len(closes) or not _ok(prior[idx]) or not closes[idx] > prior[idx]:
            continue
        w[max(je, 0):min(jx, days)] = 1.0
    return w


def _clock_grid(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, events: list[dict]) -> dict:
    """msc._clock for ma50 and ma50 OR H1 only (same engine, cost, waiting time, missed entries and seeds)."""
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, msc.YEAR)
    keys = ("ma50", "over")
    text = {k: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for k in keys}
    person = {k: (np.zeros((len(starts), 24, msc.SIMS)), np.zeros((len(starts), 24, msc.SIMS))) for k in keys}
    ki = msc.CANDS.index("ma50")
    for h in range(24):
        closes = np.concatenate([pre, price[h::24]])
        target = msc._targets(pd.Series(closes)).fillna(0.0)
        w_ma = target["ma50"].values[npre:][:days].astype(float)
        w_ov = np.maximum(w_ma, _clock_window(h, closes, npre, b0, days, events))
        for k, w in (("ma50", w_ma), ("over", w_ov)):
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, msc.COST, None)]), starts, msc.YEAR)
            text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(msc.SEED + 1000 * ki + h)  # main_signal_check's ma50 seed, for both rules
            pos = rt._missed(wl, msc.P_MISS, rng, msc.SIMS)
            mid = eng._mid(b, pos, msc.P_COST, None)
            r, dd = msc._suffix(np.vstack([np.ones((1, msc.SIMS)), mid]), starts, msc.YEAR)
            person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
    return {"starts": labels, "hold": (hr[:, 0], hd[:, 0]), "text": text, "person": person}


def _score(d: dict, c: dict, o: dict) -> dict:
    """main_signal_check's ranking numbers for ma50 and the overlay."""
    hr, hd = c["hold"]
    out = {}
    for k in ("ma50", "over"):
        dr, dd = d[k]
        tr, td = c["text"][k]
        pr, _ = c["person"][k]
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        n_o = sum(len(o[a]["starts"]) for a in o)
        beat_o = sum(int((o[a][k][0] > o[a]["hold"][0]).sum()) for a in o)
        x = {"d": float((dr > d["hold"][0]).mean()), "c": float((edge > 0).mean()), "p": float((pedge > 0).mean()),
             "o": beat_o / n_o if n_o else 0.0, "c_edge": float(np.median(edge)), "p_edge": float(np.median(pedge)),
             "c_deep": float(np.median(td)),
             "loss_ok": min(float((dd > d["hold"][1]).mean()), float((td > hd[:, None]).mean())),
             "vs_ma50": float((tr > c["text"]["ma50"][0]).mean()), "vs_edge": float(np.median(tr - c["text"]["ma50"][0])),
             "days": d[k + "_days"]}
        x["low"] = min(x["d"], x["c"], x["p"], x["o"])
        out[k] = x
    return out


# ---------- checks ----------

def _lookahead(close: pd.Series, events: list[dict]) -> bool:
    """Windows and overlay targets from data cut at 2022-06-30 equal the full-data ones up to that day."""
    cut = pd.Timestamp("2022-06-30")
    full = np.maximum(msc._targets(close)["ma50"].fillna(0.0).values, _window(close.index, events, _Series(close), PRIMARY))
    short = close[close.index <= cut]
    part = np.maximum(msc._targets(short)["ma50"].fillna(0.0).values,
                      _window(short.index, [e for e in events if e["entry"] <= cut.date()], _Series(short), PRIMARY))
    n = len(short) - 3  # the last few days of the cut series may hold a window that is still open
    return bool(np.array_equal(full[:n], part[:n]))


# ---------- the study ----------

def _compute() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    events, _ = _calendar()
    today = date.fromisoformat(common.today_utc())
    blocked = _blocked(events)
    problems = []
    close = te._daily_close()
    last_day = close.index[-1].date()
    done = [e for e in events if e["exit"] <= last_day]
    btc = _Series(close)
    daily = {n: _event_study(btc, done, blocked, n, rng) for n in (None,) + NS}
    sharpe = {n: _sharpe(btc, done, n, "2018-01-01", "2025-06-30") for n in (None,) + NS}

    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    price, b0, _ = cr._grid(rows)
    c21 = pd.Series(price[NY_CLOSE_UTC::24], index=pd.date_range(b0, periods=len(price[NY_CLOSE_UTC::24]), freq="D"))
    s21 = _Series(c21)
    done21 = [e for e in done if e["exit"] <= c21.index[-1].date()]
    hourly = {n: _event_study(s21, done21, blocked, n, rng) for n in (None, PRIMARY)}
    sharpe21 = {n: _sharpe(s21, done21, n, "2018-01-01", "2025-06-30") for n in (None, PRIMARY)}

    coins, coin_close = {}, {}
    for symbol, key in COINS:
        try:
            cc = ta._crypto_close(symbol)
            coin_close[key] = cc
            sc = _Series(cc)
            dn = [e for e in done if e["exit"] <= cc.index[-1].date()]
            coins[key] = {"h0": _event_study(sc, dn, blocked, None, rng), "h1": _event_study(sc, dn, blocked, PRIMARY, rng),
                          "first": cc.index[0]}
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{key} 這次取不到資料或算不出來（{type(exc).__name__}），沒有算進去。")

    # overlay on the fair grid
    dgrid = _daily_grid(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], events)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    cgrid = _clock_grid(price, b0, pre, events)
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key, cc in coin_close.items():
        try:
            other[key] = _daily_grid(cc, YEAR, cc.index[0], three, events)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key} 在公平格子上算不出來（{type(exc).__name__}）。")
    try:
        mstr = ta._share_close("MSTR")
        other["MSTR"] = _daily_grid(mstr, msc.SHARE_YEAR, "2020-08-11", three, events)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"MSTR 這次取不到（{type(exc).__name__}），排名的「其他資產」一項少了它，和 main_signal_check 不完全一樣。")
    score = _score(dgrid, cgrid, other)
    look = _lookahead(close, events)
    return {"events": events, "done": done, "today": today, "close": close, "daily": daily, "sharpe": sharpe,
            "hourly": hourly, "sharpe21": sharpe21, "c21": c21, "coins": coins, "score": score, "look": look,
            "problems": problems, "n_cells": len(cgrid["starts"]) * 24, "n_daily": len(dgrid["starts"]),
            "other_keys": list(other), "seconds": time.time() - t0}


def study() -> dict:
    r = _compute()
    D, H, sc = r["daily"], r["hourly"], r["score"]
    close = r["close"]
    p1 = D[PRIMARY]
    per = p1["periods"]

    # ----- pass test -----
    signs = [np.sign(p["net"]) for p in per if _ok(p["net"])]
    same = len(signs) == len(PERIODS) and len(set(signs)) == 1 and signs[0] != 0
    ci = p1.get("net_ci")
    ci_clear = bool(ci) and (ci[0] > 0 or ci[1] < 0)
    ci_sign_ok = ci_clear and same and np.sign(ci[0]) == signs[0]
    over_ok = sc["over"]["low"] >= sc["ma50"]["low"]
    plc_ok = _ok(p1.get("plc_share")) and p1["plc_share"] <= 0.10
    passed = bool(ci_sign_ok and over_ok and plc_ok)

    # ----- table 1: bitcoin event results -----
    def row1(name: str, x: dict, sh) -> list:
        if x["n"] < 2:
            return [name, str(x["n"]), "—", "—", "—", "—", "—", "—"]
        return [name, f"{x['n']}", f"{_p(x['mean'])}｜{_share(x['up'])}", f"{_p(x['net'])}｜{_ci(x['net_ci'])}",
                _p(x["net3"]), _num(sh), f"{_p(x['excess'])}｜{_ci(x['ex_ci'])}", _share(x["plc_share"])]

    rows1 = [row1("不篩選（H0），UTC 0 點收盤", D[None], r["sharpe"][None])]
    for n in NS:
        rows1.append(row1(f"只在創 {n} 日收盤新高時（H1{'，主要' if n == PRIMARY else '，參考'}），UTC 0 點收盤", D[n], r["sharpe"][n]))
    rows1.append(row1("不篩選，美股收盤時間 UTC 21 點（2017-08 起）", H[None], r["sharpe21"][None]))
    rows1.append(row1(f"創 {PRIMARY} 日新高才做，UTC 21 點（2017-08 起）", H[PRIMARY], r["sharpe21"][PRIMARY]))

    # ----- table 2: periods -----
    rows2 = []
    for i, (name, _, _) in enumerate(PERIODS):
        a, b, c = D[None]["periods"][i], per[i], H[PRIMARY]["periods"][i] if "periods" in H[PRIMARY] else {"n": 0, "net": None}
        others = "／".join(_p(D[n]["periods"][i]["net"]) for n in NS if n != PRIMARY)
        rows2.append([name, f"{a['n']} 次｜{_p(a['net'])}", f"{b['n']} 次｜{_p(b['net'])}", _ci(b["ci"]), _share(b["up"]),
                      _p(b["excess"]), f"{c['n']} 次｜{_p(c['net'])}", others])
    rows2.append(["全部", f"{D[None]['n']} 次｜{_p(D[None].get('net'))}", f"{p1['n']} 次｜{_p(p1.get('net'))}", _ci(p1.get("net_ci")),
                  _share(p1.get("up")), _p(p1.get("excess")), f"{H[PRIMARY]['n']} 次｜{_p(H[PRIMARY].get('net'))}",
                  "／".join(_p(D[n].get("net")) for n in NS if n != PRIMARY)])

    # ----- table 3: overlay -----
    names = {"ma50": "50 日均線（main_signal_check）", "over": f"50 日均線，或假日前創 {PRIMARY} 日新高時加買"}
    rows3 = []
    for k in ("ma50", "over"):
        x = sc[k]
        rows3.append([names[k], _share(x["d"]), f"{_share(x['c'])}｜{x['c_edge']:+.1f} 點", f"{_share(x['p'])}｜{x['p_edge']:+.1f} 點",
                      _share(x["o"]), _share1(x["low"]), f"{_share(x['loss_ok'])}｜{x['c_deep']:.0f}%",
                      "—" if k == "ma50" else f"{_share(x['vs_ma50'])}｜{x['vs_edge']:+.2f} 點"])

    # ----- table 4: other coins -----
    rows4 = []
    for key, x in r["coins"].items():
        a, b = x["h0"], x["h1"]
        rows4.append([key, f"{_d(x['first'])[:7]} 起", f"{a['n']} 次｜{_p(a.get('net'))}", f"{b['n']} 次｜{_p(b.get('net'))}",
                      _ci(b.get("net_ci")), _share(b.get("up")), f"{_p(b.get('excess'))}｜{_share(b.get('plc_share'))}"])
    btc18 = [x for x in p1["rows"] if x["ev"]["holiday"] >= date(2018, 1, 1)]
    if btc18:
        nn = _net(np.array([x["r"] for x in btc18]), COST)
        rows4.append(["對照：比特幣同一做法（2018 起）", "2018-01 起", f"{sum(1 for x in D[None]['rows'] if x['ev']['holiday'] >= date(2018, 1, 1))} 次｜"
                      f"{_p(float(_net(np.array([x['r'] for x in D[None]['rows'] if x['ev']['holiday'] >= date(2018, 1, 1)]), COST).mean()))}",
                      f"{len(btc18)} 次｜{_p(float(nn.mean()))}", "—", _share(float((nn > 0).mean())), "—"])

    n_c, n_d = r["n_cells"], r["n_daily"]
    tables = [
        {"caption": ("比特幣在美股（NYSE）休市日前後的表現：假日前最後一個交易日的收盤買、假日後第一個交易日的收盤賣（週一放假就是週五買、週二賣）。"
                     "「UTC 0 點收盤」是儀表板日線，在美股收盤後 3–4 小時；「UTC 21 點」是幣安 1 小時線，最接近美股收盤。成本是每次買或賣各扣 0.10%（或 0.30%），"
                     "一來一回約 0.20%（0.60%）。90% 區間是區塊重抽 2,000 次。夏普值只算 2018-01～2025-06（Quantpedia 的研究期間）、不扣成本、一年 365 天，"
                     "其他時間拿現金。對照是同一星期幾、同樣天數、但不碰假日的時段（前後 26 週內），H1 的對照也要同樣創新高；"
                     "最後一欄是把每次換成隨機一個對照時段、抽 2,000 次，平均報酬不輸真的比例（越小越像真的有假日效果）。"),
         "columns": ["做法", "次數", "平均每次（未扣成本）｜扣成本後賺的比例", "扣 0.10% 後平均｜90% 區間", "扣 0.30% 後平均",
                     "研究期間夏普值（未扣）", "比對照時段多（平均｜90% 區間）", "對照抽樣不輸真的比例"],
         "rows": rows1},
        {"caption": (f"分三段看（依假日日期分）：Quantpedia 研究期間以前、研究期間、研究期間以後。報酬都是扣 0.10%（一來一回約 0.20%）後的平均每次。"
                     f"主要檢驗是「創 {PRIMARY} 日新高才做、UTC 0 點收盤」；最後一欄是 N = 5／20 的參考。研究期間以後只有十幾個假日，次數很少。"),
         "columns": ["期間", "不篩選：次數｜平均", f"創 {PRIMARY} 日新高：次數｜平均", f"創 {PRIMARY} 日新高：90% 區間",
                     f"創 {PRIMARY} 日新高：賺的比例", f"創 {PRIMARY} 日新高：比對照時段多", f"UTC 21 點、創 {PRIMARY} 日新高：次數｜平均",
                     "N = 5／20：平均"],
         "rows": rows2},
        {"caption": (f"加在 50 日均線上：均線說持有就持有；均線說不持有時，若遇到假日前創 {PRIMARY} 日新高，就從假日前收盤持有到假日後收盤。"
                     f"用 main_signal_check 同一套公平考法和排名規則：比特幣 {n_d} 個起算日（UTC 0 點、照課本做、成本 0.10%）、"
                     f"{n_c:,} 格起算日×收盤時間、同樣格子照一般人方式做（10 小時後才做、成本 0.25%、一成進場漏掉、每個時鐘 200 次模擬，"
                     f"和 main_signal_check 的 50 日均線用同一組亂數）、其他資產（{'、'.join(r['other_keys'])}）。每格是「報酬贏過一直抱著的比例」，"
                     "排名看四項裡最低的那一項。最後一欄是 {n} 格裡加買版本報酬比單純 50 日均線高的比例｜每年多幾個百分點（中位數）。".format(n=f"{n_c:,}")),
         "columns": ["規則", f"比特幣 {n_d} 個起算日", "起算日×收盤時間：贏的比例｜差距中位數", "照一般人方式做：贏的比例｜差距中位數",
                     "其他資產", "四項裡最低（排名分數）", "最深虧損比抱著淺（較低組）｜最深虧損中位數", "比單純均線高的格子｜差距中位數"],
         "rows": rows3},
        {"caption": (f"同樣的假日日期用在其他幣（幣安日線，UTC 0 點收盤），扣 0.10% 後平均每次。主要做法是創 {PRIMARY} 日新高才做；"
                     "最後一欄是比同星期幾對照時段多多少｜對照抽樣不輸真的比例。各幣上市時間不同，大多只有 2018 以後的資料，和比特幣的研究期間重疊。"),
         "columns": ["幣", "資料期間", "不篩選：次數｜平均", f"創 {PRIMARY} 日新高：次數｜平均", "90% 區間", "賺的比例", "比對照多｜對照不輸真的比例"],
         "rows": rows4},
    ]

    # ----- verdict -----
    def per_text(x: dict) -> str:
        return "、".join(f"{name.split('（')[0]} {_p(p['net'])}（{p['n']} 次）" for (name, _, _), p in zip(PERIODS, x["periods"]))

    first = (f"Quantpedia 說美股假日前一天、比特幣又創 N 日新高時買進、假日後一天賣出有用；用比特幣日線重算，主要的「創 {PRIMARY} 日新高」版本扣 0.10% 後平均每次"
             f" {_p(p1.get('net'))}（{p1['n']} 次，90% 區間 {_ci(p1.get('net_ci'))}），不篩選版 {_p(D[None].get('net'))}。")
    second = (f"分段看：{per_text(p1)}，"
              + ("三段同方向" if same else "三段方向不一致")
              + f"；和同星期幾、也創新高的平常時段比，隨機對照有 {_share(p1.get('plc_share'))} 不輸它。")
    third = (f"加在 50 日均線上，排名分數 {_share1(sc['over']['low'])}，單純均線 {_share1(sc['ma50']['low'])}，"
             f"{'沒有變差' if over_ok else '略為變差'}；事先定的過關條件{'全部達到' if passed else '沒有全部達到'}。")
    verdict = first + second + third

    # ----- now -----
    today = r["today"]
    nxt = next((e for e in r["events"] if e["holiday"] > today), None)
    c = close.values
    last = float(c[-1])
    hi = float(c[-1 - PRIMARY:-1].max())
    hi_next = float(c[-PRIMARY:].max())
    ma = float(c[-50:].mean())
    now = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"}]
    if nxt:
        wd = "一二三四五六日"
        now.append({"label": "下一個美股休市日",
                    "value": (f"{nxt['holiday']}（星期{wd[nxt['holiday'].weekday()]}）；規則的進場收盤是 {nxt['entry']} 的日線收盤"
                              f"（{nxt['entry'] + timedelta(days=1)} UTC 0 點），出場是 {nxt['exit']} 的日線收盤"
                              + ("；假日前一天美股提早收盤" if nxt["early"] else "") + "。")})
    now.append({"label": f"H1 條件（收盤高於前 {PRIMARY} 天收盤最高）用最後收盤算",
                "value": (f"{'成立' if last > hi else '不成立'}：最後收盤 {last:,.0f}，前 {PRIMARY} 天最高 {hi:,.0f}；"
                          f"下一根收盤要高於約 {hi_next:,.0f} 美元（離最後收盤 {(hi_next / last - 1) * 100:+.1f}%）才會成立，"
                          "真正看的是假日前一天那根收盤。")})
    now.append({"label": "50 日均線狀態", "value": f"{'高於' if last > ma else '低於'} 50 日均線（約 {ma:,.0f} 美元）"})
    now.append({"label": "這幾格的意思", "value": "只描述規則現在停在什麼狀態，不是買賣建議"})

    # ----- notes -----
    recent = [x for x in D[None]["rows"] if x["ev"]["holiday"] >= date(2025, 7, 1)]
    on_set = {x["ev"]["holiday"] for x in p1["rows"]}
    recent_text = "；".join(f"{x['ev']['holiday']} {_p(float(_net(x['r'], COST)))}{'（創新高，有做）' if x['ev']['holiday'] in on_set else ''}"
                            for x in recent)
    early = p1.get("early") or []
    others = [x for x in p1["rows"] if not x["ev"]["early"]]
    notes = [
        "事先寫好的設計（原文照錄）：把「假日前一天收盤」對到離美股收盤最近的比特幣 UTC 0 點收盤，另外用幣安 UTC 21 點 1 小時線（2017-08 起）當更接近的版本；"
        f"H0 不篩選：每個 NYSE 休市日從 D−1 收盤持有到 D+1 收盤；H1 Quantpedia 篩選：D−1 收盤高於前 N 天收盤最高才做，N = 5／10／20，N = {PRIMARY} 是事先指定的主要版本，"
        "其他只做描述；檢驗：和同星期幾非假日時段比（對照、區塊重抽），分 2013–2017、2018～2025 年中、2025 年中以後三段，成本 0.10%／0.30%，"
        "加在 50 日均線上（均線不持有時才加買，也就是「均線 或 H1」）在公平格子上和 50 日均線比，其他幣（ETH／SOL／BNB／XRP）用同樣日期。"
        "過關條件：H1 主要版本三段同方向（特別是 2018 以前和 2025 年中以後），全期 90% 區間扣 0.10% 後不含 0，而且加買版本排名分數不低於 50 日均線，"
        "而且對照抽樣不輸真的比例 ≤ 10%。",
        f"過關檢查：三段同方向 {'是' if same else '否'}；全期 90% 區間 {_ci(ci)}，{'不含 0' if ci_clear else '含 0'}；"
        f"排名分數 {_share1(sc['over']['low'])} 對 {_share1(sc['ma50']['low'])}，{'不低於' if over_ok else '低於'}；"
        f"對照抽樣不輸真的比例 {_share(p1.get('plc_share'))}，{'≤' if plc_ok else '>'} 10%。結論：{'過關' if passed else '沒有過關'}。",
        "Quantpedia 原文（2025-09-08，Cyril Dujava）：用比特幣期貨代理（2018–2021）接 BITO ETF（2021–2025-06）的日收盤，假日前一天收盤買、假日後一天收盤賣；"
        "不篩選版「大部分時間是平的」；加上「收盤高於前 N 個交易日最高收盤」（N = 5、10、20）後較好，文中說夏普值超過 0.6、Calmar 超過 0.7"
        "（文字把最好的數字連到 5 日版本，完整表格是圖片）；沒有提到交易成本，是同一段資料裡看出來的（樣本內）。假日表用 Time and Date 的「美國共同假日」，"
        "這裡改用 NYSE 實際休市日，可能和它的清單有出入。",
        "對照的換算：BITO 只在美股交易日有價格，所以它的「前 N 天」是交易日；比特幣每天都有收盤，這裡的「前 N 天」是日曆天，N = 10 大約等於 7 個交易日。"
        "夏普值這裡用一年 365 天換算、只在持有窗口算進場，和 Quantpedia 的一年 252 個交易日不同，數字只能粗略比大小。",
        f"美股提早收盤（下午 1 點）的假日前一天：主要版本裡有 {len(early)} 次，扣成本後平均 {_p(float(np.mean(early))) if early else '—'}；"
        f"其他 {len(others)} 次平均 {_p(float(_net(np.array([x['r'] for x in others]), COST).mean())) if others else '—'}（只做描述）。",
        f"研究期間以後的每一次（不篩選，扣 0.10%）：{recent_text or '—'}。",
        f"對照時段不夠三個同樣創新高的，就改用全部對照時段：主要版本有 {p1.get('fallbacks', 0)} 次這樣。",
        f"檢查：只用 2022-06-30 以前的資料重算均線與假日窗口，和用全部資料算的前段完全一樣（{'通過' if r['look'] else '不通過'}）；"
        "50 日均線部分直接用 main_signal_check 的函數、成本、等待時間、漏單機率和亂數種子，數字應與該頁的 50 日均線一致。",
        "2017-08 以前的比特幣日線是 blockchain.info 的每日價格（不完全是 UTC 0 點收盤），2013–2017 這一段的時間點比較粗。",
        "假日只有一年 9–11 個，篩選後更少；研究期間以後只有十幾個假日，任何一段的平均都很容易被一兩次大漲大跌左右。",
        f"這頁沒有調參數；N = {PRIMARY} 是事先指定的，N = 5、20 只做描述。過去的結果不保證未來；這裡只描述規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
        f"整頁計算約 {r['seconds']:.0f} 秒。",
    ] + r["problems"]

    return _clean({
        "id": "pre_holiday",
        "title": "美股假日前買比特幣",
        "question": ("Quantpedia 說：美股休市日前一天收盤買比特幣、假日後一天收盤賣，尤其在比特幣創近期新高時，會賺錢。"
                     "在它的研究期間以前、以後還成立嗎？扣掉成本、加在 50 日均線上有沒有幫助？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Quantpedia：Surprisingly Profitable Pre-Holiday Drift Signal for Bitcoin（2025-09-08）", "url": QP_URL},
            {"name": "Ariel (1990), High Stock Returns before Holidays, Journal of Finance",
             "url": "https://doi.org/10.1111/j.1540-6261.1990.tb05110.x"},
            {"name": "Lakonishok & Smidt (1988), Are Seasonal Anomalies Real? A Ninety-Year Perspective, Review of Financial Studies",
             "url": "https://doi.org/10.1093/rfs/1.4.403"},
        ] + HOLIDAY_SOURCES + [
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/markets"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤（公平格子的其他資產）", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    })


def _clean(x):
    """No NaN or infinity anywhere; None becomes "—"."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, np.integer):
        return int(x)
    return "—" if x is None else x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
