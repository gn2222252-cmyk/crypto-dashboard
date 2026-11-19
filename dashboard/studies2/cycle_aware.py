"""Historical study "cycle_aware": three popular 2026 claims that a long-only "trend / cycle" rule beats simply holding
bitcoin, checked against what was actually published and then put through main_signal_check's fair exam.

What was published (read before writing this file, 2026-11-19):
  (A) CoinDesk "Crypto for Advisors" newsletter, CoinDesk Indices section, 2026-06-17, by Markus Thielen (CEO of
      10x Research): "a cycle-aware long-only approach" had a Sharpe ratio of 1.22 in backtesting vs 0.82 for
      buy-and-hold "over the same 15-year period", maximum drawdown cut from -80% to -44%; ten signals "spanning
      momentum, trend and on-chain cost-basis metrics"; when most signals are positive bitcoin averaged +25% a month,
      when most are negative -6%; allocation is a band ("100%, 50%, or 0%" of a capped allocation). The signals, the
      cut-off, the allocation map, rebalancing, costs and the exact dates are NOT published (the 10x Research post
      that describes it is behind a paywall). So the published rule cannot be replicated as written. The KuCoin flash
      page the request pointed to is in fact a summary of a different note (BIT, see C), not of this article.
  (B) Mohit Sharma's blog "btc-ema20-sma50-strategy": EMA20 trend signals taken only while the 50-day simple average
      says uptrend, daily closes, 0.1% fee, no leverage or shorts, data 2014-2026; claims CAGR ~71% vs 60% for
      holding, Sharpe 1.18 vs 0.86, maximum drawdown ~50.9% vs 83.4%; the author says about ten variants were tried.
      The page now redirects to the blog index; the description comes from a search engine's summary of it, so the
      exact entry wording is my reading.
  (C) BIT "BIT on Target" (bit.com research, 2026-10-23): bitcoin 10 years, holding 62.8%/yr, long-only trend
      following 68.4%/yr, long-short 40.8%/yr. bit.com refuses automated reading (HTTP 403, not bypassed); the public
      copies (KuCoin, Odaily, AiCoin) do not say how a trend is defined. Not specified -> not tested.

Pre-registered design (fixed before running; no tuning, no other variants):
  CA  "cycle-aware 10" (an open stand-in for (A) built only from its published description): ten yes/no signals at
      each daily close -
        trend (5): close > 20-, 50-, 100-, 200-day simple average; 50-day average > 200-day average;
        momentum (4): close above the close 30, 90, 180, 365 days earlier;
        on-chain cost basis (1): close > short-term-holder cost basis (Checkonchain "STH Cost Basis", the series
            and loader of sth_cost; the value of UTC day D is used at the close of D on the daily series and from the
            first clock after day D has ended on the 24-clock grid; a missing value counts as "no").
      Hold 100% when more than half are "yes" (at least 6 of 10), else cash. Other coins have no free cost-basis
      series, so there CA uses the nine price signals (at least 5 of 9).
  CB  the band version of the article's own example (reference, not graded on its own): 100% when at least 70% of
      the signals say yes, 0% when at most 30%, else 50%.
  E   (B) as I read it: hold when close > 20-day EMA (span 20, adjust=False) AND close > 50-day simple average.
  ma50 the dashboard's main signal (close > 50-day average), main_signal_check's rule and random seeds.

  Replication (description; the published rules are not complete enough for anything stronger):
    (A) 2011-06-17 ... 2026-06-16 (15 years to the article's date), daily, 0.10% per trade, Sharpe = mean / standard
        deviation of daily returns x sqrt(365), no interest on cash: holding's Sharpe and deepest loss (claim 0.82,
        -80%) and CA's / CB's (claim 1.22, -44%); bitcoin's average next-calendar-month return when CA is on / off at
        the month's last close (claim +25% / -6%).
    (B) 2014-01-01 ... last finished day, same arithmetic: E vs holding yearly return, Sharpe, deepest loss (claim
        71% vs 60%, 1.18 vs 0.86, -50.9% vs -83.4%).
  Fair exam (main_signal_check's machinery and settings): (1) bitcoin, the dashboard's daily closes (te._daily_close,
    pre-2017-08 days relabelled on 2026-11-19 to the day the candle covers; no extra shift), 108 monthly starts
    2015-01 ... 2023-12, decide at the close / hold the next day, 0.10%; (2) 72 starts 2018-01 ... 2023-12 x 24 closing
    hours from Binance 1-hour candles with the pre-2017-08 daily closes as warm-up; (3) the same cells, ordinary
    person: trade 10 hours later, 0.25%, 10% of entries missed, 200 simulations per clock (seed SEED + 1000 * k + h,
    k = 1 for ma50 as in main_signal_check, 90 / 91 / 92 for CA / CB / E); (4) ETH, SOL, BNB, XRP (Binance daily) and
    MSTR (Yahoo, from 2020-08-11), monthly starts from the first month in which every rule here has a value to three
    years before today, textbook; ma50 on exactly the same starts.
    Ranking score = the lowest of the four "beats holding" shares; eligible when the deepest loss is shallower than
    holding's in at least 95% of the bitcoin starts and of the cells; median deepest loss = median over the 1,728
    textbook cells.
  Placebo (CA and E have no external signal worth shifting - CA's one on-chain signal is one vote of ten): random
    on/off rules with the same number of entries and the same time in the market as the real rule (two-state random
    switching, per asset, matched on that asset's series from its first start), 200 draws, the whole exam rebuilt
    with 20 ordinary-person simulations per clock; the real rule scored the same way for this comparison; placebo
    share = draws whose ranking score is at least the real rule's.
  Pass test (stated now, for CA and for E separately): ranking score >= ma50's (and eligible whenever ma50 is) AND
    median deepest loss no worse than ma50's AND placebo share <= 10%.
  Today: each rule's state after the last finished daily close - states only, not advice.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import sth_cost as sc
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
LOSS_BAR = msc.LOSS_BAR
DAILY_STARTS, CLOCK_STARTS = msc.DAILY_STARTS, msc.CLOCK_STARTS
PLACEBO_DRAWS, PLACEBO_SIMS = 200, 20
KEYS = ("ma50", "ca", "cb", "e")
TESTED = ("ca", "e")
KI = {"ma50": 1, "ca": 90, "cb": 91, "e": 92}
SMA_N = (20, 50, 100, 200)
MOM_N = (30, 90, 180, 365)
REP_A = ("2011-06-17", "2026-06-16")
REP_B = "2014-01-01"
CLAIM_A = {"sharpe": 1.22, "hold_sharpe": 0.82, "deep": -44.0, "hold_deep": -80.0, "on": 25.0, "off": -6.0}
CLAIM_B = {"cagr": 71.0, "hold_cagr": 60.0, "sharpe": 1.18, "hold_sharpe": 0.86, "deep": -50.9, "hold_deep": -83.4}
NAME = {"ma50": "50 日均線（現行主訊號）", "ca": "CA：十個訊號過半就全部持有（照 CoinDesk 文章描述自己寫的替身）",
        "cb": "CB：同十個訊號，七成以上全持有、三成以下全現金、中間持有一半（參考）",
        "e": "E：收盤同時高於 20 日指數均線和 50 日均線才持有（部落格規則）"}
SHORT = {"ma50": "50 日均線", "ca": "CA 十訊號過半", "cb": "CB 十訊號分三段", "e": "E 20EMA＋50MA"}

URL_CD = "https://www.coindesk.com/coindesk-indices/2026/06/17/crypto-for-advisors-trading-the-bitcoin-cycle"
URL_10X = "https://post.10xresearch.com/p/why-bitcoin-investors-should-trade"
URL_KU = ("https://www.kucoin.com/news/flash/bitcoin-buy-and-hold-strategy-underperforms-trend-following-in-10-year-"
          "backtest")
URL_BIT = "https://www.bit.com/research/bitcoin-vs-sp500-two-winning-strategies"
URL_ODAILY = "https://www.odaily.news/en/post/5213564"
URL_BLOG = "https://mohit-sharma-website.pages.dev/blogs/btc-ema20-sma50-strategy"

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- rules ----------

def _price_signals(close: pd.Series) -> pd.DataFrame:
    cols = {}
    sma = {n: close.rolling(n, min_periods=n).mean() for n in SMA_N}
    for n in SMA_N:
        cols[f"S{n}"] = (close > sma[n]).astype(float).where(sma[n].notna())
    cols["X50_200"] = (sma[50] > sma[200]).astype(float).where(sma[200].notna())
    for n in MOM_N:
        past = close.shift(n)
        cols[f"M{n}"] = (close > past).astype(float).where(past.notna())
    return pd.DataFrame(cols, index=close.index)


def _targets(close: pd.Series, onchain: np.ndarray | None = None) -> pd.DataFrame:
    """What each rule wants to hold after each close (NaN until every price signal has a value).
    onchain = the STH cost basis usable at each close (same length as close), or None for other assets."""
    sig = _price_signals(close)
    ready = sig.notna().all(axis=1)
    if onchain is not None:
        cost = np.asarray(onchain, dtype=float)
        sig["STH"] = np.where(np.isfinite(cost), (close.values > cost).astype(float), 0.0)
    share = sig.mean(axis=1).where(ready)
    ma50 = close.rolling(50, min_periods=50).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out = pd.DataFrame(index=close.index)
    out["ma50"] = (close > ma50).astype(float).where(ma50.notna())
    out["ca"] = (share > 0.5).astype(float).where(share.notna())
    out["cb"] = pd.Series(np.where(share >= 0.7 - 1e-9, 1.0, np.where(share <= 0.3 + 1e-9, 0.0, 0.5)),
                          index=close.index).where(share.notna())
    out["e"] = ((close > ema20) & (close > ma50)).astype(float).where(ma50.notna() & ema20.notna())
    out.attrs["votes"] = sig.sum(axis=1).where(ready)
    out.attrs["n_sig"] = sig.shape[1]
    return out


def _cost_on(dates: pd.DatetimeIndex, cost: pd.Series, lag: int = 0) -> np.ndarray:
    """The cost basis of UTC day (date - lag) for each date (NaN when unknown)."""
    return cost.reindex(dates - pd.Timedelta(days=lag)).values.astype(float)


# ---------- exam pieces ----------

def _daily_grid(close: pd.Series, target: pd.DataFrame, first, last, year: int, keys=KEYS) -> dict:
    change = close.pct_change().fillna(0.0).values
    starts, labels = msc._month_starts(close.index, first, last)
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out = {"starts": labels, "pos": starts, "hold": (hr[:, 0], hd[:, 0])}
    for k in keys:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    return out


def _first_valid(target: pd.DataFrame) -> pd.Timestamp:
    valid = target[list(KEYS)].notna().all(axis=1)
    return target.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1)


class _Clock:
    """72 starts x 24 clocks of bitcoin (main_signal_check._clock with this study's rules)."""

    def __init__(self, price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, cost: pd.Series):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days, self.npre = self.m.days, len(pre)
        self.starts, self.labels = msc._month_starts(self.m.dates, CLOCK_STARTS[0], CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hold = (hr[:, 0], hd[:, 0])
        # a close at clock h on day j (boundary b0 + j days + h hours) knows the cost basis of UTC day b0 + j - 1
        on_days = _cost_on(self.m.dates, cost, lag=1)
        self.w = {}
        for h in range(24):
            closes = pd.Series(np.concatenate([pre, price[h::24]]))
            onchain = np.concatenate([np.full(self.npre, np.nan), on_days,
                                      np.full(len(closes) - self.npre - len(on_days), np.nan)])[:len(closes)]
            t = _targets(closes, onchain).fillna(0.0)
            self.w[h] = {k: t[k].values[self.npre:][:self.days].astype(float) for k in KEYS}

    def text(self, w_of_h) -> tuple[np.ndarray, np.ndarray]:
        r_all, d_all = np.zeros((len(self.starts), 24)), np.zeros((len(self.starts), 24))
        for h in range(24):
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w_of_h(h), COST, None)]), self.starts, YEAR)
            r_all[:, h], d_all[:, h] = r[:, 0], dd[:, 0]
        return r_all, d_all

    def person(self, w_of_h, ki: int, sims: int) -> tuple[np.ndarray, np.ndarray]:
        r_all = np.zeros((len(self.starts), 24, sims))
        d_all = np.zeros((len(self.starts), 24, sims))
        for h in range(24):
            b, wl = rt._later(h, w_of_h(h), self.days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pos = rt._missed(wl, P_MISS, rng, sims)
            mid = self.eng._mid(b, pos, P_COST, None)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), mid]), self.starts, YEAR)
            r_all[:, h, :], d_all[:, h, :] = r, dd
        return r_all, d_all


def _score(dg: tuple, hold_d: tuple, tx: tuple, hold_c: tuple, pe: tuple, others: list[tuple]) -> dict:
    """main_signal_check._summary's arithmetic for one rule. others = [(rule (ret, deep), hold (ret, deep))]."""
    dr, dd = dg
    hr, hd = hold_c
    tr, td = tx
    pr, pdd = pe
    edge = tr - hr[:, None]
    pedge = pr - hr[:, None, None]
    o_n = sum(len(a[0][0]) for a in others)
    o_beat = sum(int((a[0][0] > a[1][0]).sum()) for a in others)
    x = {"d_n": len(dr), "d_beat": int((dr > hold_d[0]).sum()), "d_edge": float(np.median(dr - hold_d[0])),
         "d_shallow": int((dd > hold_d[1]).sum()),
         "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
         "c_p10": float(np.percentile(edge, 10)), "c_p90": float(np.percentile(edge, 90)),
         "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
         "c_deep_worst": float(td.min()),
         "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
         "p_deep_med": float(np.median(pdd)), "o_n": o_n, "o_beat": o_beat}
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   (o_beat / o_n) if o_n else 0.0]
    x["low"] = min(x["checks"])
    x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
    x["eligible"] = x["loss_ok"] >= LOSS_BAR
    return x


# ---------- random rules for the placebo ----------

def _entries_share(w: np.ndarray) -> tuple[int, float]:
    w = np.nan_to_num(np.asarray(w, dtype=float))
    on = w >= 0.5
    return int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0)), float(on.mean())


def _random_rules(n: int, entries: int, share: float, rng: np.random.Generator, draws: int) -> np.ndarray:
    """(n, draws) random on/off schedules: switching on and off at random with the real rule's average run lengths."""
    entries = max(entries, 1)
    share = min(max(share, 0.02), 0.98)
    p_off = 1.0 / max(share * n / entries, 1.0001)
    p_on = 1.0 / max((1 - share) * n / entries, 1.0001)
    out = np.zeros((n, draws))
    state = rng.random(draws) < share
    u = rng.random((n, draws))
    for i in range(n):
        state = np.where(state, u[i] >= p_off, u[i] < p_on)
        out[i] = state
    return out


# ---------- replication ----------

def _stats(net: np.ndarray, year: int = YEAR) -> dict:
    growth = float(np.prod(1 + net))
    eq = np.concatenate([[1.0], np.cumprod(1 + net)])
    sd = float(np.std(net, ddof=1))
    return {"cagr": (growth ** (year / len(net)) - 1) * 100 if growth > 0 else -100.0,
            "sharpe": float(np.mean(net) / sd * math.sqrt(year)) if sd > 0 else float("nan"),
            "deep": float((eq / np.maximum.accumulate(eq) - 1).min() * 100), "total": (growth - 1) * 100}


def _replicate(close: pd.Series, target: pd.DataFrame, a: str, b: str | None) -> dict:
    change = close.pct_change().fillna(0.0)
    keep = (close.index >= pd.Timestamp(a)) & ((close.index <= pd.Timestamp(b)) if b else True)
    out = {"hold": _stats(change.values[keep]), "days": int(keep.sum()), "first": _d(close.index[keep][0]),
           "last": _d(close.index[keep][-1])}
    for k in KEYS:
        net, _ = te._run(target[k].shift(1).values, change.values, COST)
        out[k] = _stats(net[keep])
    return out


def _regime_months(close: pd.Series, target: pd.DataFrame) -> dict:
    """Bitcoin's next-calendar-month return by CA's state at the month's last close (inside REP_A)."""
    month_end = close.resample("ME").last()
    state = target["ca"].resample("ME").last()
    nxt = month_end.shift(-1) / month_end - 1
    keep = (month_end.index >= pd.Timestamp(REP_A[0])) & (month_end.index < pd.Timestamp(REP_A[1]))
    df = pd.DataFrame({"s": state, "n": nxt})[keep].dropna()
    on, off = df.loc[df["s"] >= 0.5, "n"] * 100, df.loc[df["s"] < 0.5, "n"] * 100
    return {"on": float(on.mean()), "off": float(off.mean()), "on_n": len(on), "off_n": len(off),
            "on_med": float(on.median()), "off_med": float(off.median())}


# ---------- the whole computation ----------

def _compute() -> dict:
    problems = []
    close = te._daily_close()
    cost, prob, _ = sc._coc()
    problems += prob
    target = _targets(close, _cost_on(close.index, cost))
    first_d = max(_first_valid(target), pd.Timestamp(DAILY_STARTS[0]))
    daily = _daily_grid(close, target, first_d, DAILY_STARTS[1], YEAR)

    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _Clock(price, b0, pre, cost)
    text = {k: clock.text(lambda h, k=k: clock.w[h][k]) for k in KEYS}
    person = {k: clock.person(lambda h, k=k: clock.w[h][k], KI[k], SIMS) for k in KEYS}

    other_data, prob = msc._load_other()
    problems += prob
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key, (c, since, year) in other_data.items():
        try:
            t = _targets(c)
            first = max(_first_valid(t), pd.Timestamp(since or c.index[0]))
            g = _daily_grid(c, t, first, three, year)
            if not g["starts"]:
                raise RuntimeError("沒有起算日")
            other[key] = (c, t, year, g)
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")

    score = {}
    for k in KEYS:
        score[k] = _score(daily[k], daily["hold"], text[k], clock.hold, person[k],
                          [(other[a][3][k], other[a][3]["hold"]) for a in other])

    # ----- placebo: random rules with the real rule's entries and time in the market -----
    placebo = {}
    for k in TESTED:
        pe_small = clock.person(lambda h, k=k: clock.w[h][k], KI[k], PLACEBO_SIMS)
        real = _score(daily[k], daily["hold"], text[k], clock.hold, pe_small,
                      [(other[a][3][k], other[a][3]["hold"]) for a in other])["low"]
        rng = np.random.default_rng(SEED + 7 + KI[k])
        span = close.index >= first_d
        n_btc = int(span.sum())
        ent, sh = _entries_share(target[k].values[span])
        rb = _random_rules(n_btc, ent, sh, rng, PLACEBO_DRAWS)
        rb_s = pd.DataFrame(rb, index=close.index[span])
        rand_other = {}
        for a in other:
            c, t, year, g = other[a]
            sp = c.index >= g["starts"][0]
            e2, s2 = _entries_share(t[k].values[sp])
            rand_other[a] = (sp, _random_rules(int(sp.sum()), e2, s2, rng, PLACEBO_DRAWS))
        clock_rows = rb_s.reindex(clock.m.dates).fillna(0.0).values  # wanted after a close on that UTC date
        change = close.pct_change().fillna(0.0).values
        scores = []
        for i in range(PLACEBO_DRAWS):
            pos = np.zeros(len(close))
            pos[span] = rb[:, i]
            net, _ = te._run(pd.Series(pos).shift(1).values, change, COST)
            r, dd = msc._suffix(msc._equity(net), daily["pos"], YEAR)
            dg = (r[:, 0], dd[:, 0])
            w = clock_rows[:, i]
            tx = clock.text(lambda h: w)
            pe = clock.person(lambda h: w, KI[k], PLACEBO_SIMS)
            oth = []
            for a in other:
                c, t, year, g = other[a]
                sp, rr = rand_other[a]
                p2 = np.zeros(len(c))
                p2[sp] = rr[:, i]
                ch = c.pct_change().fillna(0.0).values
                n2, _ = te._run(pd.Series(p2).shift(1).values, ch, COST)
                r2, d2 = msc._suffix(msc._equity(n2), g["pos"], year)
                oth.append(((r2[:, 0], d2[:, 0]), g["hold"]))
            scores.append(_score(dg, daily["hold"], tx, clock.hold, pe, oth)["low"])
        scores = np.array(scores)
        placebo[k] = {"real": real, "share": float(np.mean(scores >= real - 1e-12)),
                      "p50": float(np.median(scores)), "p90": float(np.percentile(scores, 90)),
                      "entries": ent, "in": sh}

    rep_a = _replicate(close, target, REP_A[0], REP_A[1])
    rep_b = _replicate(close, target, REP_B, None)
    months = _regime_months(close, target)
    return {"close": close, "target": target, "cost": cost, "daily": daily, "clock": clock, "other": other,
            "score": score, "placebo": placebo, "rep_a": rep_a, "rep_b": rep_b, "months": months,
            "problems": problems}


# ---------- today ----------

def _now(close: pd.Series, target: pd.DataFrame, cost: pd.Series) -> list[dict]:
    def word(v) -> str:
        if not _ok(v):
            return "—"
        return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"

    t = target.iloc[-1]
    last = float(close.iloc[-1])
    sig = _price_signals(close).iloc[-1]
    cb = cost.reindex([close.index[-1]]).iloc[0]
    sth_yes = _ok(cb) and last > cb
    votes = int(sig.sum()) + int(sth_yes)
    ma50 = float(close.iloc[-50:].mean())
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    yes = [k for k, v in sig.items() if v >= 0.5] + (["STH"] if sth_yes else [])
    names = {"S20": "20 日均線", "S50": "50 日均線", "S100": "100 日均線", "S200": "200 日均線", "X50_200": "50 日線在 200 日線上",
             "M30": "30 天動能", "M90": "90 天動能", "M180": "180 天動能", "M365": "365 天動能", "STH": "短期持有者成本"}
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"},
           {"label": SHORT["ma50"], "value": f"{word(t['ma50'])}（50 日平均 {ma50:,.0f} 美元）"},
           {"label": "十個訊號裡看多的", "value": f"{votes} 個：{'、'.join(names[k] for k in yes) or '沒有'}"
                                          + (f"；短期持有者成本 {cb:,.0f} 美元" if _ok(cb) else "；短期持有者成本這天沒有資料")},
           {"label": SHORT["ca"], "value": word(t["ca"])},
           {"label": SHORT["cb"], "value": word(t["cb"])},
           {"label": SHORT["e"], "value": f"{word(t['e'])}（20 日指數均線 {ema20:,.0f}、50 日均線 {ma50:,.0f} 美元）"},
           {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"}]
    return out


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    s, pl, close = r["score"], r["placebo"], r["close"]
    d, ck, o = r["daily"], r["clock"], r["other"]
    ra, rb, mo = r["rep_a"], r["rep_b"], r["months"]
    n_d, n_c = len(d["starts"]), s["ma50"]["c_n"]
    base = s["ma50"]

    def passes(k: str) -> dict:
        x = s[k]
        rank_ok = x["low"] >= base["low"] - 1e-12 and (x["eligible"] or not base["eligible"])
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - 1e-9
        plc_ok = pl[k]["share"] <= 0.10
        return {"rank": rank_ok, "loss": loss_ok, "placebo": plc_ok, "all": rank_ok and loss_ok and plc_ok}

    verdicts = {k: passes(k) for k in TESTED}

    # table 1: replication
    rows1 = [
        ["CoinDesk／10x：一直抱著的 Sharpe", f"{CLAIM_A['hold_sharpe']:.2f}", f"{ra['hold']['sharpe']:.2f}"],
        ["CoinDesk／10x：一直抱著的最深虧損", _pct(CLAIM_A["hold_deep"]), _pct(ra["hold"]["deep"])],
        ["CoinDesk／10x：週期策略的 Sharpe", f"{CLAIM_A['sharpe']:.2f}",
         f"CA {ra['ca']['sharpe']:.2f}｜CB {ra['cb']['sharpe']:.2f}（替身，不是原規則）"],
        ["CoinDesk／10x：週期策略的最深虧損", _pct(CLAIM_A["deep"]),
         f"CA {_pct(ra['ca']['deep'])}｜CB {_pct(ra['cb']['deep'])}"],
        ["CoinDesk／10x：多數訊號看多時，下個月平均漲跌", _pct(CLAIM_A["on"]),
         f"CA 看多的月底 {mo['on_n']} 次：平均 {_pct(mo['on'])}（中位數 {_pct(mo['on_med'])}）"],
        ["CoinDesk／10x：多數訊號看空時，下個月平均漲跌", _pct(CLAIM_A["off"]),
         f"CA 看空的月底 {mo['off_n']} 次：平均 {_pct(mo['off'])}（中位數 {_pct(mo['off_med'])}）"],
        ["部落格 E：每年報酬（規則｜一直抱著）", f"{_pct(CLAIM_B['cagr'])}｜{_pct(CLAIM_B['hold_cagr'])}",
         f"{_pct(rb['e']['cagr'])}｜{_pct(rb['hold']['cagr'])}"],
        ["部落格 E：Sharpe（規則｜一直抱著）", f"{CLAIM_B['sharpe']:.2f}｜{CLAIM_B['hold_sharpe']:.2f}",
         f"{rb['e']['sharpe']:.2f}｜{rb['hold']['sharpe']:.2f}"],
        ["部落格 E：最深虧損（規則｜一直抱著）", f"{_pct(CLAIM_B['deep'])}｜{_pct(CLAIM_B['hold_deep'])}",
         f"{_pct(rb['e']['deep'])}｜{_pct(rb['hold']['deep'])}"],
        ["同期間對照：50 日均線（Sharpe｜最深）", "—",
         f"CoinDesk 期間 {ra['ma50']['sharpe']:.2f}｜{_pct(ra['ma50']['deep'])}；部落格期間 {rb['ma50']['sharpe']:.2f}｜"
         f"{_pct(rb['ma50']['deep'])}"],
    ]

    # table 2: fair exam
    rows2 = []
    for k in KEYS:
        x = s[k]
        rows2.append([NAME[k], f"{_of(x['d_n'], x['d_beat'])}（{_share(x['checks'][0])}）",
                      f"{_of(x['c_n'], x['c_beat'])}（{_share(x['checks'][1])}）｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}",
                      f"{_of(x['o_n'], x['o_beat'])}（{_share(x['checks'][3])}）",
                      _share(x["low"]), f"{_share(x['loss_ok'])}{'' if x['eligible'] else '（未達 95%）'}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}"])
    hd = ck.hold[1]
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜{_pct(float(hd.min()))}"])

    # table 3: pass test
    yes = {True: "過", False: "沒過"}
    rows3 = []
    for k in TESTED:
        v, x, p = verdicts[k], s[k], pl[k]
        rows3.append([NAME[k], f"{_share(x['low'])} 對 {_share(base['low'])}：{yes[v['rank']]}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])}：{yes[v['loss']]}",
                      f"{_share(p['share'])}（隨機規則中位數 {_share(p['p50'])}、前一成 {_share(p['p90'])}；本規則 {_share(p['real'])}）："
                      f"{yes[v['placebo']]}",
                      "全部過" if v["all"] else "沒有全部過"])

    tables = [
        {"caption": (f"先對答案。CoinDesk 文章（10x Research）只公布結果，沒有公布十個訊號、門檻、持倉比例、成本和確切期間，所以「原規則」無法照做；"
                     f"這裡用文章描述自己寫一個公開的替身 CA（十個訊號過半全持有）和 CB（七成／三成分三段），期間取文章日期往前 15 年 "
                     f"{ra['first']} 到 {ra['last']}。部落格 E 用 {rb['first']} 到 {rb['last']}。每次買賣扣 0.10%，現金沒有利息，"
                     "Sharpe = 每日報酬平均 ÷ 標準差 × √365。"),
         "columns": ["項目", "文章說的", "這裡算出來的"], "rows": rows1},
        {"caption": (f"公平考試（main_signal_check 同一套）：比特幣 UTC 0 點日線 {DAILY_STARTS[0][:7]} 到 {DAILY_STARTS[1][:7]} 每月 1 日起算（{n_d} 個）；"
                     f"{CLOCK_STARTS[0][:7]} 到 {CLOCK_STARTS[1][:7]} 起算 × 24 個收盤時間（{n_c:,} 格）；一般人做法（晚 10 小時、扣 0.25%、"
                     f"10% 的進場漏掉、每個時鐘 {SIMS} 次模擬）；其他幣 ETH、SOL、BNB、XRP、MSTR（沒有鏈上成本資料，CA／CB 只用九個價格訊號）。"
                     "「贏」= 每年報酬比同一天起一直抱著高。排名分數 = 四項裡最低的比例。最後一欄是 24 時鐘照課本做的最深虧損。"),
         "columns": ["規則", f"比特幣起算日（{n_d} 個）", f"起算日×收盤時間（{n_c:,} 格）｜差距中位數", "一般人做法：贏的比例｜差距中位數",
                     "其他資產", "排名分數（四項最低）", "最深虧損比抱著淺（兩組較低）", "最深虧損：中位數｜最深一格"],
         "rows": rows2},
        {"caption": (f"及格標準（跑之前就定好）：排名分數不低於 50 日均線、最深虧損中位數不比 50 日均線深、隨機規則（同樣進場次數和持有時間比例，"
                     f"{PLACEBO_DRAWS} 次，一般人做法每個時鐘只模擬 {PLACEBO_SIMS} 次）分數不低於本規則的比例 ≤ 10%。"),
         "columns": ["規則", "排名分數：本規則 對 50 日均線", "最深虧損中位數：本規則 對 50 日均線", "隨機規則分數不低於本規則的比例", "結果"],
         "rows": rows3},
    ]

    ca, e = s["ca"], s["e"]
    va, ve = verdicts["ca"], verdicts["e"]
    passed = [SHORT[k] for k in TESTED if verdicts[k]["all"]]
    if passed:
        lead = f"照描述做出來的版本裡，{'、'.join(passed)} 通過了事先定好的考試，可以和現行 50 日均線並列觀察。"
    else:
        lead = ("這些「週期／趨勢策略勝過抱著」的說法，照描述做出來的版本都比抱著少虧，但都沒有勝過現行的 50 日均線，"
                "不值得換掉主訊號。")
    first = ("CoinDesk 這篇「週期策略 Sharpe 1.22 對抱著 0.82」沒有公布規則（十個訊號是哪些、幾個算多數、持倉多少、成本和期間都沒寫，"
             "細節在 10x Research 的付費文章裡），原數字無法驗證；")
    first += (f"照文章描述自己寫的公開替身 CA 在同樣 15 年算出 Sharpe {ra['ca']['sharpe']:.2f}、一直抱著 {ra['hold']['sharpe']:.2f}、"
              f"單純 50 日均線 {ra['ma50']['sharpe']:.2f}（文章說 1.22 和 0.82），最深虧損 {_pct(ra['ca']['deep'])} 對抱著 "
              f"{_pct(ra['hold']['deep'])}（文章說 −44% 和 −80%）。")
    second = (f"公平考試：CA 排名分數 {_share(ca['low'])}、50 日均線 {_share(base['low'])}（CA 換到其他幣只有 {_share(ca['checks'][3])} 贏、"
              f"一般人做法 {_share(ca['checks'][2])}），最深虧損中位數 {_pct(ca['c_deep_med'])} 對 {_pct(base['c_deep_med'])}，"
              f"隨機規則 {_share(pl['ca']['share'])} 不輸它，{'及格' if va['all'] else '沒有及格'}；"
              f"部落格的 20EMA＋50MA 規則 E 排名分數 {_share(e['low'])}、最深虧損中位數 {_pct(e['c_deep_med'])}、隨機規則 "
              f"{_share(pl['e']['share'])}，{'及格' if ve['all'] else '沒有及格'}。")
    third = "BIT 的「十年趨勢策略每年 68.4% 對抱著 62.8%」沒公布怎麼判斷趨勢，沒有測。"
    verdict = lead + first + second + third

    notes = [
        "原文怎麼說：CoinDesk「Crypto for Advisors」2026-06-17（作者 10x Research 執行長 Markus Thielen）：十個訊號涵蓋動能、趨勢和鏈上成本，"
        "多數看多時比特幣平均每月 +25%、多數看空時 −6%；週期策略回測 Sharpe 1.22、一直抱著 0.82（同一段 15 年），最深虧損從 −80% 降到 −44%；"
        "持倉用「上限內 100%、50% 或 0%」的區間。文章沒有寫訊號的定義、門檻、成本、換倉頻率和確切起訖日。",
        "請求裡給的 KuCoin 快訊其實是在摘要 BIT（bit.com）的另一篇報告，不是 CoinDesk 這篇；bit.com 的原文拒絕自動讀取（HTTP 403），"
        "沒有繞過，改讀 KuCoin、Odaily 的轉載：只說「多頭趨勢持有、空頭趨勢拿現金」，十年每年 68.4% 對抱著 62.8%、多空都做 40.8%，"
        "五年 25% 對 8%，沒有寫怎麼判斷趨勢，所以沒有測。",
        "部落格 E 的網址現在會轉回部落格首頁，規則和數字（2014–2026、每次 0.1%、每年約 71% 對 60%、Sharpe 1.18 對 0.86、最深約 −50.9% 對 −83.4%）"
        "是從搜尋引擎對該頁的摘要讀到的，原文寫法無法再核對；「收盤同時高於 20 日指數均線和 50 日均線」是這裡的解讀。作者自己說試了約十種變化挑最好的。"
        "它和 trend_ensemble／main_signal_check 測過的 50 日均線只差一個 20 日短線條件，本質上是均線規則的變形。",
        f"CA 的十個訊號是看結果前自己定的（20／50／100／200 日均線、50 日線在 200 日線上、30／90／180／365 天動能、短期持有者成本），"
        "不是 10x 的原訊號；它的成績只能說明「照文章描述做一個合理版本」會怎樣，不能證明或否定 10x 的原數字。",
        f"CA 在比特幣用 {r['target'].attrs['n_sig']} 個訊號；其他幣沒有免費的鏈上成本資料，改用九個價格訊號（至少 5 個看多才持有）。"
        "短期持有者成本來自 Checkonchain 今天的版本，它會小幅修改歷史（見 sth_cost）。日線上 D 日收盤用 D 日的成本，24 時鐘上用前一個 UTC 日的成本。",
        f"隨機規則對照：CA 在比特幣（從第一個起算日起）進場 {pl['ca']['entries']} 次、持有時間 {pl['ca']['in'] * 100:.0f}%；"
        f"E 進場 {pl['e']['entries']} 次、持有 {pl['e']['in'] * 100:.0f}%。隨機規則照同樣的平均持有長度隨機切換，每個資產各自配對。",
        f"儀表板的日線從 {_d(close.index[0])} 開始，CA 要 365 天動能，最早 {_d(_first_valid(r['target']))} 才有部位，"
        f"所以 15 年期間的頭兩個月 CA 拿現金；E 在部落格期間每年 {_pct(rb['e']['cagr'])}、單純 50 日均線 {_pct(rb['ma50']['cagr'])}。",
        "Sharpe 用 √365 換算、現金沒有利息；文章可能用不同的換算或期間，所以一直抱著的 Sharpe 對不上也可能只是算法不同。"
        "2017-08 以前的比特幣日線是 blockchain.info 的每日價格（2026-11-19 已改標成那根 K 線所屬的日子）。",
        "相鄰起算日、相鄰收盤時間大部分歷史重疊，「幾個裡幾個」不是獨立證據；其他資產是今天還在的幣和一檔股票，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去的表現和現在的狀態，不是買賣建議。",
    ] + r["problems"]

    return {
        "id": "cycle_aware",
        "title": "「週期策略勝過抱著」的說法查證",
        "question": ("2026 年流傳的「週期／趨勢策略 Sharpe 1.22 勝過抱著 0.82」（CoinDesk）、「20EMA＋50MA」（部落格）、「趨勢策略十年 68.4%」（BIT）"
                     "公布的規則能不能照做？照做或照描述做的版本，通過公平考試、勝過現行 50 日均線嗎？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(close, r["target"], r["cost"]),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "CoinDesk：Crypto for Advisors: Trading the bitcoin cycle（2026-06-17，Markus Thielen）", "url": URL_CD},
            {"name": "10x Research：Why bitcoin investors should trade the cycle（付費文章，只有前言公開）", "url": URL_10X},
            {"name": "KuCoin 快訊（摘要 BIT 報告）", "url": URL_KU},
            {"name": "BIT on Target：Bitcoin vs S&P 500（拒絕自動讀取）", "url": URL_BIT},
            {"name": "Odaily 轉載 BIT 報告", "url": URL_ODAILY},
            {"name": "Mohit Sharma 部落格：BTC EMA20／SMA50（現在轉回首頁）", "url": URL_BLOG},
            {"name": "Checkonchain：短期持有者成本（STH Cost Basis）", "url": sc.COC_PAGE},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
