"""Historical study "price_impact": are sellers scarce when price rises easily on buying but falls little on selling -
measured DIRECTLY from how bitcoin's price responds to net buying versus net selling - and does that predict the
next weeks or let a holder get in earlier than the main signal ma50 (hold while the daily close is above its 50-day
average)?

Line of reasoning (written before any test). Price rises when buyers are more eager than sellers, and it rises MOST
per dollar when few holders are willing to sell. Two project studies tested this from the "stock" side and failed:
supply_demand defined "few sellers" by low exchange inflows + low MVRV, and that combination turned out to mark bear
markets (forward returns were worse, S1 failed the fair exam); rally_sources found no money-flow channel that turns up
before ma50, but its exploratory table showed that early in rallies each dollar of measured inflow moved market value
far more than later (e.g. 43.8x vs 5.6x, 100.6x vs 24.7x). So instead of guessing seller scarcity from holdings,
measure it from the price response itself: if a dollar of net buying lifts price more than a dollar of net selling
pushes it down, the selling side is thin (or buyers absorb selling) - "sellers scarce".
How this differs from order_flow: order_flow tested the taker-buy SHARE, cumulative volume delta and volume as
predictors (how much buying); this study uses the same Binance taker data but measures the PRICE RESPONSE PER DOLLAR
of buying versus selling (how much price moves for that buying), i.e. the asymmetry of impact, not the flow itself.

Data: Binance spot BTCUSDT 1-hour klines (public, key-less /api/v3/klines through price._binance and common.get_json,
fixed user agent "btc-research-dashboard"), fields close, quote volume and taker-buy quote volume, from the first
listed hour (2017-08-17). Kept in this study's own store cache/_store/price_impact_1h_BTCUSDT.csv and topped up on
later runs (closed bars only). Daily prices: the dashboard's daily closes (te._daily_close, 00:00 UTC; Binance from
2017-08). A refusal is reported, never worked around.

Definitions (fixed before running; hourly bars; everything for the decision at the close of UTC candle D uses only
the hourly bars that opened from D-13 00:00 to D 23:00, i.e. closed by that daily close; the decision is held on D+1):
  signed flow of an hour s = taker-buy quote volume - taker-sell quote volume (USD), taker-sell = quote - taker-buy.
  hourly log return r = log(close_h / close_{h-1}); an hour counts only when both closes exist and quote > 0.
  Window = the trailing 14 days (336 hourly bars); at least 300 counted hours are required (data-gap guard, fixed
  before the run), otherwise every quantity below is missing that day.
  Net-buy hours: s > 0; net-sell hours: s < 0.
  UP impact = sum(r over net-buy hours) / sum(s over net-buy hours) (price gained per dollar of net buying).
  DOWN impact = sum(r over net-sell hours) / sum(s over net-sell hours) (both negative -> positive: price lost per
    dollar of net selling).
  ASYM = log(UP / DOWN), only when both are positive; otherwise missing (how often, and why, is reported).
    ASYM > 0: price rises more easily on buying than it falls on selling = sellers scarce / buyers absorb selling.
  SCARCE = ASYM above the 80th percentile of its own previous 365 daily values (today excluded, missing values
    skipped; at least 180 values). Undefined (NaN) until that history exists; when today's ASYM is missing and the
    threshold exists, SCARCE = 0.
  Plain impact (Amihud-style liquidity control) = mean over counted hours of |r| / quote volume, 14 days.
  ma50 = main_signal_check's (close above the 50-day average); above20 = close above the average of the last 20
    closes including today (supply_demand._above20).

Test A - does SCARCE predict? Sample: every decision day with SCARCE defined and a full 30-day forward window.
  Forward BTC returns close[t+h]/close[t] - 1 for h = 7, 14, 30 and the deepest drop within 30 days
  (min(close[t+1..t+30]) / close[t] - 1, capped at 0; funding_warning's definition). SCARCE days versus all days of
  the same sample: means, the difference with a 90% interval from a circular block bootstrap of days (30-day blocks,
  1,000 draws, fixed seed) inside each period, and a random-day comparison (1,000 random sets of days of the same
  size from the same sample: share whose mean 30-day return is at least SCARCE's). Periods: full, 2018-19, 2020-22,
  2023-now (each clipped to the sample). The same restricted to days with ma50 OFF (SCARCE & ma50 off versus all
  ma50-off days; the bootstrap still draws calendar blocks).
  Control (funding_warning._runup's matching): the sample's days are put into deciles of the trailing 14-day return
  (close[t]/close[t-14] - 1, decile edges from the whole sample); matched difference = SCARCE minus non-SCARCE mean
  next-30-day return inside each decile, weighted by the number of SCARCE days in it (deciles with both kinds only);
  same for the deepest drop; 90% interval from the same block bootstrap (edges fixed, days resampled).
  Pre-registered pass: SCARCE mean 30-day forward return above the all-days mean in the full period AND in at least
  two of the three sub-periods AND the full-period matched difference's 90% interval excludes zero.

Test B - the rule I1 = hold when ma50 is on OR (SCARCE and above20). Fair exam = main_signal_check's, wired exactly
  as supply_demand's _Exam (textbook 0.10% daily starts; starts x 24 closing hours; ordinary person: decide at the
  clock's close, act 10 hours later, 0.25% cost, 10% of entries missed, 200 simulations per clock with
  main_signal_check's seeds and its ma50 draws; other assets ETH SOL BNB XRP (Binance) and MSTR (Yahoo) with their
  OWN 50- and 20-day averages and BTC's SCARCE flag as a market-wide flag, as supply_demand did). Starts: monthly,
  from the first month starting on or after the first day SCARCE is defined, to main_signal_check's last daily start
  (2023-12) for BTC daily and clock starts (clock starts also not before main_signal_check's 2018-01), and to three
  years before today for other assets. ma50 is re-scored on exactly the same starts; the number of starts scored is
  reported. On the 24-clock grid a close on UTC date E uses SCARCE decided at the UTC close of candle E-1; MSTR's
  close on date D uses SCARCE of candle D-1; other coins' UTC closes use the same candle's SCARCE.
  Ranking score = worst of the four "beats holding" shares with main_signal_check's eligibility.
  Placebo: SCARCE (from its first defined day) circularly shifted by 180 ... (n - 180) days, 200 draws, the whole exam
  rebuilt; the ordinary-person part uses 20 simulations per clock (same seeds) and the real I1 is re-scored the same
  way; placebo share = share of draws whose ranking score is at least the real rule's.
  Pre-registered pass, all three: ranking score >= ma50's; median deepest loss (over the clock cells) no more than
  5 points worse than ma50's; placebo share <= 10%.

Report only:
  - the 12 rallies of rally_sources (its own mechanical definition, +80% / 30%): for those with ASYM data, ASYM, its
    rank against its previous 365 values and SCARCE at the trough T, at the ma50 date M and at the peak P, and the
    share of SCARCE days T->M versus M->P (does scarcity show up early and fade late, as the inflow-multiple clue
    suggested?);
  - days I1 holds that ma50 does not, by year, with their average next-day return (supply_demand._extra);
  - early entries (I1 turns on while ma50 is off) right / wrong and the cost of the wrong ones (supply_demand._entries:
    "right" = ma50 turns on within 30 days at a higher close);
  - lead/lag (funding_warning's form): correlation of daily ASYM with BTC's 7-day log return ending k days later,
    k = -30..+30 (k > 0: ASYM first); noise band = 95th percentile of |corr| over 200 circular shifts of ASYM by
    >= 180 days, pooled over k;
  - liquidity control: correlation of ASYM with log plain impact, and SCARCE minus all-days 30-day return on days
    whose plain impact is above / below its sample median;
  - single venue: Binance only. Binance's share of the market changed over the years, and Binance ran zero-fee
    trading on BTC spot pairs including BTC/USDT from 2022-07-08 14:00 UTC (announcement "Binance Launches Zero-Fee
    Bitcoin Trading", 2022-07-06) until 2023-03-22 00:00 UTC (announcement "Updates on Zero-Fee Bitcoin Trading &
    BUSD Zero Maker Fee Promotion", 2023-03-15: BTC/USDT back to standard fees), which inflates taker volume (wash
    and fee-free churn). Robustness line: Test A's full-period comparison and matched control with every decision day
    whose 14-day window touches that period removed (decision days 2022-07-08 .. 2023-04-04).

Honesty: the 14-day window, the 80th percentile, the 365-day lookback, the 300-hour guard and the 20-day average
were fixed by reasoning before the run and no alternatives were tried. One venue only. A pass would only make I1 a
candidate for the daily shadow log. Not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import rally_sources as rs
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

SYMBOL = "BTCUSDT"
STORE = common.STORE / "price_impact_1h_BTCUSDT.csv"
FIRST_MS = 1502942400000  # 2017-08-17 04:00 UTC, the first BTCUSDT hour Binance lists
HOUR_MS, DAY_MS = 3_600_000, 86_400_000
MAX_PAGES = 120
WIN_DAYS, MIN_HOURS = 14, 300
PCT, LOOK, LOOK_MIN = 0.80, 365, 180
H = (7, 14, 30)
BLOCK, BOOT, RAND = 30, 1000, 1000
SEED = msc.SEED
DRAWS, GAP, PL_SIMS = 200, 180, 20
LOSS_SLACK = 5.0
KI = {"ma50": 1, "i1": 95}
LL_K = tuple(range(-30, 31))
LL_SHOW = (-30, -21, -14, -7, -3, 0, 3, 7, 14, 21, 30)
PERIODS = (("2018–19", "2018-01-01", "2019-12-31"), ("2020–22", "2020-01-01", "2022-12-31"),
           ("2023 起", "2023-01-01", "2100-01-01"))
ZERO_FEE = ("2022-07-08 14:00", "2023-03-22 00:00")
ZF_DROP = ("2022-07-08", "2023-04-04")  # decision days whose 14-day window touches the zero-fee period
DAY = pd.Timedelta(days=1)
URL_LAUNCH = ("https://www.binance.com/en/support/announcement/"
              "binance-launches-zero-fee-bitcoin-trading-10435147c55d4a40b64fcbf43cb46329")
URL_END = ("https://www.binance.com/en/support/announcement/"
           "updates-on-zero-fee-bitcoin-trading-busd-zero-maker-fee-promotion-be13a645cca643d28eab5b9b34f2dc36")

_d = cr._d
_ok, _p, _pt, _ci, _share, _share1, _pc = sd._ok, sd._p, sd._pt, sd._ci, sd._share, sd._share1, sd._pc


# ---------- data ----------

def _read() -> dict[int, list[float]]:
    rows: dict[int, list[float]] = {}
    try:
        for line in STORE.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 4 or not parts[0].isdigit():
                continue
            v = [float(x) for x in parts[1:]]
            if all(math.isfinite(x) for x in v):
                rows[int(parts[0])] = v
    except Exception:  # noqa: BLE001 - no store yet: fetch from the start
        return {}
    return rows


def _write(rows: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{ms}," + ",".join(repr(x) for x in v) for ms, v in sorted(rows.items()))
        STORE.write_text("ms,close,quote,tbq\n" + body + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run fetches again
        pass


def _hourly() -> tuple[dict, str | None]:
    """{open ms: [close, quote volume, taker-buy quote volume]} of closed BTCUSDT 1-hour bars."""
    rows = _read()
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - HOUR_MS:
        start = (max(rows) + 1) if rows else FIRST_MS
        added = 0
        try:
            for _ in range(MAX_PAGES):
                page = px._binance("/api/v3/klines", {"symbol": SYMBOL, "interval": "1h", "limit": 1000,
                                                      "startTime": start})
                if not page:
                    break
                now_ms = int(time.time() * 1000)
                for r in page:
                    if int(r[6]) < now_ms:  # closed bars only
                        rows[int(r[0])] = [float(r[4]), float(r[7]), float(r[10])]
                        added += 1
                if len(page) < 1000:
                    break
                start = int(page[-1][0]) + 1
                time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = (f"幣安 BTCUSDT 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
        if added:
            _write(rows)
    if len(rows) < 50_000:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    return rows, problem


# ---------- the measure ----------

def _measure(rows: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """ASYM, UP, DOWN, plain impact and counts on daily decision dates (candle D uses hours opened D-13 .. D 23:00)."""
    b0 = (min(rows) // DAY_MS) * DAY_MS
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    k = (today_ms - b0) // HOUR_MS
    g = np.full((k, 3), np.nan)
    for ms, v in rows.items():
        i = (ms - b0) // HOUR_MS
        if 0 <= i < k:
            g[i] = v
    c, q, tb = g[:, 0], g[:, 1], g[:, 2]
    r = np.full(k, np.nan)
    with np.errstate(all="ignore"):
        r[1:] = np.log(c[1:] / c[:-1])
    ok = np.isfinite(r) & np.isfinite(q) & (q > 0) & np.isfinite(tb)
    s = np.where(ok, 2 * tb - q, 0.0)
    r0 = np.where(ok, r, 0.0)
    buy, sell = ok & (s > 0), ok & (s < 0)
    parts = np.column_stack([ok.astype(float), np.where(buy, r0, 0.0), np.where(buy, s, 0.0),
                             np.where(sell, r0, 0.0), np.where(sell, s, 0.0),
                             np.where(ok, np.abs(r0) / np.where(ok, q, 1.0), 0.0)])
    cs = np.vstack([np.zeros((1, parts.shape[1])), np.cumsum(parts, axis=0)])
    days = k // 24
    j = np.arange(WIN_DAYS - 1, days)
    w = cs[24 * (j + 1)] - cs[24 * (j + 1 - WIN_DAYS)]
    dates = pd.Timestamp(b0, unit="ms") + pd.to_timedelta(j, unit="D")
    n, ur, us, dr, ds, am = (w[:, i] for i in range(6))
    enough = n >= MIN_HOURS
    with np.errstate(all="ignore"):
        up = np.where(enough & (us > 0), ur / np.where(us > 0, us, 1.0), np.nan)
        down = np.where(enough & (ds < 0), dr / np.where(ds < 0, ds, -1.0), np.nan)
        asym = np.where((up > 0) & (down > 0), np.log(up / down), np.nan)
        amihud = np.where(enough, am / np.maximum(n, 1), np.nan)
    f = pd.DataFrame({"hours": n, "enough": enough, "up": up, "down": down, "asym": asym, "amihud": amihud},
                     index=dates)
    return f.reindex(index)


def _scarce(asym: pd.Series) -> tuple[pd.Series, pd.Series]:
    thr = asym.shift(1).rolling(LOOK, min_periods=LOOK_MIN).quantile(PCT)
    flag = (asym > thr).astype(float).where(thr.notna())
    return flag, thr


def _trail_rank(x: pd.Series) -> pd.Series:
    """Share of the previous 365 non-missing values below today's value (NaN with fewer than 180)."""
    v = x.values
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        prev = v[max(0, i - LOOK):i]
        prev = prev[np.isfinite(prev)]
        if len(prev) >= LOOK_MIN:
            out[i] = float((prev < v[i]).mean())
    return pd.Series(out, index=x.index)


# ---------- Test A ----------

def _forward(close: pd.Series) -> dict:
    c = close.values.astype(float)
    n = len(c)
    out = {}
    for h in H:
        f = np.full(n, np.nan)
        f[:n - h] = c[h:] / c[:n - h] - 1
        out[h] = f
    low = pd.Series(c[::-1]).rolling(30, min_periods=30).min().values[::-1]
    deep = np.full(n, np.nan)
    deep[:n - 30] = np.minimum(low[1:n - 29] / c[:n - 30] - 1, 0.0)
    out["deep"] = deep
    return out


def _boot_idx(n: int, rng) -> np.ndarray:
    nb = -(-n // BLOCK)
    st = rng.integers(0, n, (BOOT, nb))
    return ((st[:, :, None] + np.arange(BLOCK)) % n).reshape(BOOT, -1)[:, :n]


def _q(a: np.ndarray) -> tuple:
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) > 50 else (None, None)


def _matched(v: np.ndarray, sc: np.ndarray, dec: np.ndarray, sub: np.ndarray) -> float:
    """Decile-weighted SCARCE minus non-SCARCE mean (arrays may be 2-D: draws x days)."""
    v, sc, dec, sub = (np.atleast_2d(x) for x in (v, sc, dec, sub))
    num = np.zeros(v.shape[0])
    wt = np.zeros(v.shape[0])
    for k in range(10):
        a = sub & (dec == k) & (sc > 0.5)
        b = sub & (dec == k) & (sc < 0.5)
        na, nb = a.sum(axis=1), b.sum(axis=1)
        ma = np.where(a, v, 0.0).sum(axis=1) / np.maximum(na, 1)
        mb = np.where(b, v, 0.0).sum(axis=1) / np.maximum(nb, 1)
        use = (na > 0) & (nb > 0)
        num += np.where(use, na * (ma - mb), 0.0)
        wt += np.where(use, na, 0)
    with np.errstate(all="ignore"):
        return np.where(wt > 0, num / np.maximum(wt, 1), np.nan)


def _block(sc: np.ndarray, fwd: dict, sub: np.ndarray, dec: np.ndarray, rng, matched: bool) -> dict:
    """SCARCE vs all days inside one period (sub = which days count); bootstrap and random-day comparison."""
    n = len(sc)
    idx = _boot_idx(n, rng)
    s_on = sub & (sc > 0.5)
    out = {"n": int(sub.sum()), "n_s": int(s_on.sum()),
           "switch": int(((sc[1:] > 0.5) & (sc[:-1] < 0.5) & sub[1:]).sum())}
    scb, subb = sc[idx], sub[idx]
    for h in list(H) + ["deep"]:
        v = fwd[h]
        m_all = float(v[sub].mean()) if sub.any() else None
        m_s = float(v[s_on].mean()) if s_on.any() else None
        vb = v[idx]
        cs = (subb & (scb > 0.5)).sum(axis=1)
        ca = subb.sum(axis=1)
        bs = np.where(cs > 0, np.where(subb & (scb > 0.5), vb, 0.0).sum(axis=1) / np.maximum(cs, 1), np.nan)
        ba = np.where(subb, vb, 0.0).sum(axis=1) / np.maximum(ca, 1)
        out[h] = {"all": m_all, "s": m_s, "diff": (m_s - m_all) if _ok(m_s) and _ok(m_all) else None,
                  "ci": _q(bs - ba)}
    if s_on.any():
        pool = fwd[30][sub]
        draws = np.array([pool[rng.choice(len(pool), out["n_s"], replace=False)].mean() for _ in range(RAND)])
        out["rand"] = float(np.mean(draws >= out[30]["s"]))
    else:
        out["rand"] = None
    if matched:
        for key in (30, "deep"):
            v = fwd[key]
            real = float(_matched(v, sc, dec, sub)[0])
            boot = _matched(v[idx], scb, dec[idx], subb)
            out[f"m_{key}"] = (real, _q(boot))
    return out


def _test_a(close: pd.Series, flag: pd.Series, ma: np.ndarray, amihud: pd.Series) -> dict:
    fwd_all = _forward(close)
    past = (close / close.shift(14) - 1).values
    keep = flag.notna().values & np.isfinite(fwd_all[30]) & np.isfinite(past)
    idx = close.index
    start, end = idx[keep][0], idx[keep][-1]
    pv = past[keep]
    edges = np.quantile(pv, np.linspace(0, 1, 11))
    dec_all = np.full(len(idx), -1)
    dec_all[keep] = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, 9)
    sc_all = flag.fillna(0.0).values
    off_all = np.nan_to_num(ma, nan=1.0) < 0.5
    rng = np.random.default_rng(SEED + 601)
    res = {"start": start, "end": end, "full": {}, "off": {}}
    spans = [("全期", start, end)] + [(lab, max(pd.Timestamp(a), start), min(pd.Timestamp(b), end))
                                     for lab, a, b in PERIODS]
    for lab, a, b in spans:
        m = keep & (idx >= a) & (idx <= b)
        if m.sum() < 2 * BLOCK:
            continue
        sc, dec = sc_all[m], dec_all[m]
        fwd = {h: fwd_all[h][m] for h in fwd_all}
        allm = np.ones(int(m.sum()), dtype=bool)
        res["full"][lab] = _block(sc, fwd, allm, dec, rng, True)
        res["full"][lab]["span"] = (idx[m][0], idx[m][-1])
        res["off"][lab] = _block(sc, fwd, off_all[m], dec, rng, lab == "全期")
        res["off"][lab]["span"] = res["full"][lab]["span"]
    full = res["full"]["全期"]
    subs = [lab for lab, _, _ in PERIODS if lab in res["full"]]
    up_full = _ok(full[30]["diff"]) and full[30]["diff"] > 0
    up_subs = sum(1 for lab in subs if _ok(res["full"][lab][30]["diff"]) and res["full"][lab][30]["diff"] > 0)
    lo, hi = full["m_30"][1]
    excl = _ok(lo) and _ok(hi) and (lo > 0 or hi < 0)
    res["pass"] = {"up_full": up_full, "up_subs": up_subs, "n_subs": len(subs), "excl": excl,
                   "pass": bool(up_full and up_subs >= 2 and excl)}
    # robustness: zero-fee window removed (full period only)
    m = keep & ~((idx >= pd.Timestamp(ZF_DROP[0])) & (idx <= pd.Timestamp(ZF_DROP[1])))
    fwd = {h: fwd_all[h][m] for h in fwd_all}
    res["zf"] = _block(sc_all[m], fwd, np.ones(int(m.sum()), dtype=bool), dec_all[m], rng, True)
    # liquidity control: plain impact above / below its sample median
    am = amihud.values
    med = float(np.nanmedian(am[keep]))
    res["liq"] = {}
    for lab, cond in (("流動性差（單純衝擊高於中位數）", am > med), ("流動性好（單純衝擊低於中位數）", am <= med)):
        mm = keep & cond
        s_on = mm & (sc_all > 0.5)
        res["liq"][lab] = (int(s_on.sum()), float(fwd_all[30][s_on].mean()) if s_on.any() else None,
                           float(fwd_all[30][mm].mean()) if mm.any() else None)
    return res


# ---------- Test B: supply_demand's exam on this study's starts ----------

class _Exam(sd._Exam):
    def __init__(self, close: pd.Series, other: dict, price, b0, pre, first: pd.Timestamp):
        self.close = close
        self.ma = msc._targets(close)["ma50"].values.astype(float)
        self.a20 = sd._above20(close.values)
        self.change = close.pct_change().fillna(0.0).values
        self.starts, self.labels = msc._month_starts(close.index, first, msc.DAILY_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.change), self.starts, msc.YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, max(first, pd.Timestamp(msc.CLOCK_STARTS[0])),
                                                       msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, msc.YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma, self.ca20 = [], []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(np.nan_to_num(msc._targets(pd.Series(c))["ma50"].values[npre:][:self.days].astype(float)))
            self.ca20.append(sd._above20(c)[npre:][:self.days])
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, (c, since, year) in other.items():
            tg = msc._targets(c, year)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, pd.Timestamp(since or c.index[0]), first)
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "a20": sd._above20(c.values),
                               "change": change, "starts": starts, "labels": labels, "year": year,
                               "lag": 1 if since else 0, "hold": (hr[:, 0], hd[:, 0])}


# ---------- report-only pieces ----------

def _rally_rows(close: pd.Series, f: pd.DataFrame, flag: pd.Series, rank: pd.Series, ma: np.ndarray) -> list:
    rallies = rs._rallies(close, rs.RISE, rs.FALL)
    rs._describe(rallies, close, rs._ma50(close))
    rows = []
    asym = f["asym"]
    first = asym.first_valid_index()
    for i, r in enumerate(rallies, 1):
        lab = f"{i}（{_d(r['t'])}）"
        if first is None or r["p"] < first:
            rows.append([lab, "沒有資料（幣安 1 小時線 2017-08 才開始）", "—", "—", "—", "—"])
            continue

        def at(day) -> str:
            if day < first:
                return "還沒有資料"
            v, k, s = asym.get(day), rank.get(day), flag.get(day)
            if not _ok(v):
                return "缺（兩邊衝擊有一邊不是正的）"
            return (f"{v:+.2f}；過去一年 {k * 100:.0f}% 的日子比它低" if _ok(k) else f"{v:+.2f}；排名還不能算") + \
                ("；稀少" if _ok(s) and s > 0.5 else "")

        it, im, ip = r["it"], r["im"], r["ip"]
        seg1 = flag.iloc[it:im]
        seg2 = flag.iloc[im:ip + 1]
        def sh(seg):
            seg = seg.dropna()
            return f"{seg.mean() * 100:.0f}%（{len(seg)} 天）" if len(seg) else "—"
        rows.append([lab + ("" if r["done"] else "，還沒結束"), at(r["t"]), f"{at(r['m'])}（{_d(r['m'])}）",
                     f"{at(r['p'])}（{_d(r['p'])}）", sh(seg1), sh(seg2)])
    return rows


def _leadlag(asym: pd.Series, close: pd.Series) -> dict:
    lc = np.log(close)
    r7 = (lc - lc.shift(7)).values
    f = asym.values
    first = asym.first_valid_index()
    m0 = close.index >= first
    f, r = f[m0], r7[m0]
    n = len(f)

    def corr(fv, k):
        if k >= 0:
            a, b = fv[:n - k], r[k:]
        else:
            a, b = fv[-k:], r[:n + k]
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    real = {k: corr(f, k) for k in LL_K}
    rng = np.random.default_rng(SEED + 77)
    pool = []
    for s in rng.integers(GAP, n - GAP + 1, DRAWS):
        fs = np.roll(f, int(s))
        pool += [abs(corr(fs, k)) for k in LL_SHOW]
    return {"corr": real, "band": float(np.percentile(pool, 95)), "n": n}


# ---------- the computation ----------

def _compute() -> dict:
    t0 = time.time()
    problems = []
    close = te._daily_close()
    rows, p = _hourly()
    problems += [p] if p else []
    f = _measure(rows, close.index)
    flag, thr = _scarce(f["asym"])
    rank = _trail_rank(f["asym"])
    ma = msc._targets(close)["ma50"].values.astype(float)
    a = _test_a(close, flag, ma, f["amihud"])

    # missing statistics, from the first day with a full 14-day window
    have = f["hours"].notna()
    first_win = f.index[have.values][0]
    span = f.index >= first_win
    n_days = int(span.sum())
    gap = int((span & ~f["enough"].fillna(False).astype(bool).values).sum())
    miss = int((span & f["asym"].isna().values).sum())
    miss_sign = miss - gap
    up_neg = int((span & f["enough"].fillna(False).astype(bool).values & ~(f["up"] > 0).values).sum())
    down_neg = int((span & f["enough"].fillna(False).astype(bool).values & ~(f["down"] > 0).values).sum())

    hrows, problem = cr._hourly("BTCUSDT")
    problems += [problem] if problem else []
    if len(hrows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(hrows)} 筆")
    price, b0, _ = cr._grid(hrows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    other, more = msc._load_other()
    problems += more
    fstart = flag.first_valid_index()
    first_month = (fstart + pd.offsets.MonthBegin(1)) if fstart.day != 1 else fstart
    ex = _Exam(close, other, price, b0, pre, first_month)
    sflag = flag.fillna(0.0)
    s = {"ma50": ex.summary(None, KI["ma50"]), "i1": ex.summary(sflag, KI["i1"])}
    real_small = ex.summary(sflag, KI["i1"], PL_SIMS)
    prng = np.random.default_rng(SEED + 9)
    n = int((sflag.index >= fstart).sum())
    hits, lows = 0, []
    for k in prng.integers(GAP, n - GAP, DRAWS):
        x = ex.summary(sd._roll(sflag, fstart, int(k)), KI["i1"], PL_SIMS)
        lows.append(x["low"])
        hits += int(sd._score_ge(x, real_small))
    pos = {"ma50": ex.btc_pos(None), "i1": ex.btc_pos(sflag)}
    extra = sd._extra(close, ma, pos["i1"])
    entries = sd._entries(close, ma, pos["i1"])
    m = close.index >= pd.Timestamp(sd.GRID[0])
    differ = int((pos["i1"][m] != np.nan_to_num(pos["ma50"][m])).sum())
    ll = _leadlag(f["asym"], close)
    rally = _rally_rows(close, f, flag, rank, ma)
    ok = f["asym"].notna() & f["amihud"].notna() & (f["amihud"] > 0)
    liq_corr = float(np.corrcoef(f["asym"][ok], np.log(f["amihud"][ok]))[0, 1])
    return {"close": close, "f": f, "flag": flag, "thr": thr, "rank": rank, "ma": ma, "a": a, "ex": ex, "s": s,
            "real_small": real_small, "placebo": hits / DRAWS, "placebo_med": float(np.median(lows)),
            "pos": pos, "extra": extra, "entries": entries, "differ": differ, "ll": ll, "rally": rally,
            "liq_corr": liq_corr, "fstart": fstart, "first_month": first_month, "first_win": first_win,
            "miss": {"days": n_days, "gap": gap, "sign": miss_sign, "all": miss, "up_neg": up_neg,
                     "down_neg": down_neg},
            "data_start": pd.Timestamp(min(rows), unit="ms"), "problems": problems, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    close, a, s, ex, f, flag = r["close"], r["a"], r["s"], r["ex"], r["f"], r["flag"]
    last = close.index[-1]
    full = a["full"]["全期"]
    mi = r["miss"]

    # ----- Test A tables -----
    def a_rows(part: dict, matched_cols: bool) -> list:
        out = []
        for lab, x in part.items():
            sp = f"{lab}（{_d(x['span'][0])} 到 {_d(x['span'][1])}）"
            row = [sp, f"{x['n']:,}｜{x['n_s']:,}（{x['switch']} 次轉亮）"]
            for h in H:
                c = x[h]
                row.append(f"{_p(c['s'])}｜{_p(c['all'])}；差 {_pt(c['diff'])} {_ci(*c['ci'], f=_pt)}")
            c = x["deep"]
            row.append(f"{_p(c['s'])}｜{_p(c['all'])}；差 {_pt(c['diff'])} {_ci(*c['ci'], f=_pt)}")
            row.append(_share(x["rand"]) if _ok(x["rand"]) else "—")
            if matched_cols:
                mr, md = x.get("m_30"), x.get("m_deep")
                row.append(f"{_pt(mr[0])} {_ci(*mr[1], f=_pt)}" if mr else "—")
                row.append(f"{_pt(md[0])} {_ci(*md[1], f=_pt)}" if md else "—")
            out.append(row)
        return out

    cols_a = ["期間", "天數｜稀少的天數", "之後 7 天：稀少｜全部；差 ［90% 區間］", "之後 14 天", "之後 30 天",
              "30 天內最深跌幅", "隨機挑同樣多天，30 天平均不輸稀少日的比例", "配對後 30 天差 ［90% 區間］",
              "配對後最深跌幅差 ［90% 區間］"]
    ap = a["pass"]
    rows_ap = [["全期：稀少日之後 30 天平均高於全部日子", f"{_p(full[30]['s'])} 對 {_p(full[30]['all'])}",
                "成立" if ap["up_full"] else "不成立"],
               ["三段裡至少兩段也是這樣", f"{ap['n_subs']} 段裡 {ap['up_subs']} 段", "成立" if ap["up_subs"] >= 2 else "不成立"],
               ["配對「過去 14 天漲幅」之後的差，90% 區間不含 0",
                f"{_pt(full['m_30'][0])} {_ci(*full['m_30'][1], f=_pt)}", "成立" if ap["excl"] else "不成立"],
               ["三條都要成立", "", "通過" if ap["pass"] else "沒通過"]]
    zf = a["zf"]
    rows_rob = [["全部日子（同上，全期）", f"{full['n']:,}｜{full['n_s']:,}",
                 f"{_p(full[30]['s'])}｜{_p(full[30]['all'])}；差 {_pt(full[30]['diff'])} {_ci(*full[30]['ci'], f=_pt)}",
                 f"{_pt(full['m_30'][0])} {_ci(*full['m_30'][1], f=_pt)}"],
                [f"拿掉幣安免手續費期間（決定日 {ZF_DROP[0]} 到 {ZF_DROP[1]}）", f"{zf['n']:,}｜{zf['n_s']:,}",
                 f"{_p(zf[30]['s'])}｜{_p(zf[30]['all'])}；差 {_pt(zf[30]['diff'])} {_ci(*zf[30]['ci'], f=_pt)}",
                 f"{_pt(zf['m_30'][0])} {_ci(*zf['m_30'][1], f=_pt)}"]]
    for lab, (ns, ms, mall) in a["liq"].items():
        rows_rob.append([lab, f"稀少 {ns:,} 天", f"{_p(ms)}｜{_p(mall)}；差 {_pt(ms - mall) if _ok(ms) and _ok(mall) else '—'}",
                         "—"])

    # ----- Test B -----
    base, x1 = s["ma50"], s["i1"]
    score_ok = sd._score_ge(x1, base)
    loss_ok = x1["c_deep_med"] >= base["c_deep_med"] - LOSS_SLACK
    plc_ok = r["placebo"] <= 0.10
    b_pass = score_ok and loss_ok and plc_ok
    yes = lambda v: "通過" if v else "沒通過"  # noqa: E731
    name = {"ma50": "50 日均線（基準，同樣的起算日重算）", "i1": "I1：50 日均線，或「賣方稀少＋站上 20 日均線」也持有"}
    rows_b = []
    for k in ("ma50", "i1"):
        x = s[k]
        ck = x["checks"]
        rows_b.append([name[k], f"{_share(ck[0])}（{x['d_n']} 個）", f"{_share(ck[1])}（{x['c_n']:,} 格）", _share(ck[2]),
                       f"{_share(ck[3])}（{x['o_n']} 個）",
                       _share1(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                       f"{_pc(x['c_deep_med'])}｜{_pc(x['c_deep_worst'])}", f"{x['c_edge']:+.1f} 點",
                       "—" if k == "ma50" else f"{r['differ']:,} 天", _share(r["placebo"]) if k == "i1" else "—"])
    others = "、".join(f"{k} {v[0]} 個" for k, v in x1["o_each"].items())
    rows_pass = [["排名分數不輸 50 日均線", f"I1 {_share1(x1['low'])}，均線 {_share1(base['low'])}", yes(score_ok)],
                 ["最深虧損中位數最多比均線深 5 點", f"I1 {_pc(x1['c_deep_med'])}，均線 {_pc(base['c_deep_med'])}", yes(loss_ok)],
                 ["隨機錯位（安慰劑）比例 ≤ 10%", f"{_share(r['placebo'])}（錯位的排名分數中位數 {_share1(r['placebo_med'])}；"
                  f"真的 I1 同樣用每個收盤時間 {PL_SIMS} 次模擬算是 {_share1(r['real_small']['low'])}）", yes(plc_ok)],
                 ["三條都要過", "", yes(b_pass)]]

    # ----- report only -----
    e = r["extra"]
    rows_x = [[str(y), f"{n:,}", _p(v)] for y, (n, v) in e["years"].items()]
    rows_x.append(["合計", f"{e['n']:,}", _p(e["avg"])])
    ent = r["entries"]
    right = [x for x in ent if x["right"]]
    wrong = [x for x in ent if not x["right"]]
    avg = lambda v: float(np.mean([x["ret"] for x in v])) if v else None  # noqa: E731
    tot = lambda v: float(np.prod([1 + x["ret"] for x in v]) - 1) if v else None  # noqa: E731
    rows_e = [["對（30 天內均線也上車，而且價格更高）", f"{len(right)}", _p(avg(right)), _p(tot(right))],
              ["錯（其他情況）", f"{len(wrong)}", _p(avg(wrong)), _p(tot(wrong))],
              ["合計", f"{len(ent)}", _p(avg(ent)), _p(tot(ent))]]
    rows_el = [[_d(x["day"]), "對" if x["right"] else "錯", _p(x["ret"]) + ("" if x["done"] else "（還沒結束）")]
               for x in ent]
    ll = r["ll"]
    rows_l = [[f"{k:+d} 天" + ("（ASYM 在前）" if k > 0 else ("（價格在前）" if k < 0 else "（同一段）")),
               f"{ll['corr'][k]:+.2f}", "超出" if abs(ll["corr"][k]) > ll["band"] else "在雜訊內"] for k in LL_SHOW]
    # a 7-day return ending k days later overlaps ASYM's own 14-day window unless k >= 7: only those count as a lead
    lead_out = [k for k in LL_K if k >= 7 and abs(ll["corr"][k]) > ll["band"]]
    lag_out = [k for k in LL_K if k < 0 and abs(ll["corr"][k]) > ll["band"]]

    tables = [
        {"caption": (f"資料：幣安現貨 BTCUSDT 1 小時線，從 {_d(r['data_start'])} 起；第一個有完整 14 天的決定日是 {_d(r['first_win'])}，"
                     f"「賣方稀少」第一次算得出來是 {_d(r['fstart'])}。ASYM 缺的天數：從 {_d(r['first_win'])} 起 {mi['days']:,} 天裡 "
                     f"{mi['all']:,} 天（{mi['all'] / mi['days'] * 100:.1f}%）；其中資料缺口（14 天裡算得到的小時不到 {MIN_HOURS} 個）"
                     f"{mi['gap']:,} 天，兩邊衝擊有一邊不是正的 {mi['sign']:,} 天（買進衝擊不是正的 {mi['up_neg']:,} 天、"
                     f"賣出衝擊不是正的 {mi['down_neg']:,} 天，可能同時）。"),
         "columns": ["項目", "數字"],
         "rows": [["稀少的日子占（從第一次算得出來起）", _share(float(flag.dropna().mean()))],
                  ["ASYM 中位數（正＝買進推得動、賣出壓不下）", f"{float(f['asym'].median()):+.2f}"],
                  ["ASYM 大於 0 的日子", _share(float((f['asym'].dropna() > 0).mean()))],
                  ["ASYM 和單純衝擊（取對數）的相關", f"{r['liq_corr']:+.2f}"]]},
        {"caption": (f"測試 A：「賣方稀少」的日子（ASYM 高於自己過去一年的第 80 百分位）和全部日子比，之後 7、14、30 天比特幣平均漲跌，"
                     f"以及 30 天內最深跌幅。每格是「稀少｜全部；差 ［90% 區間］」，區間用 {BLOCK} 天一塊的區塊重抽 {BOOT:,} 次。"
                     f"「配對」：先依過去 14 天漲幅分成十組，在同一組裡比稀少和不稀少的日子，免得把「剛漲過」誤當成訊號。"),
         "columns": cols_a, "rows": a_rows(a["full"], True)},
        {"caption": "測試 A 的通過條件（事先定好）。", "columns": ["條件", "數字", "結果"], "rows": rows_ap},
        {"caption": ("測試 A 只看 50 日均線空手的日子（提早上車真正有用的日子，只描述）：稀少且均線空手的日子，和所有均線空手的日子比；"
                     "配對只算全期。"),
         "columns": cols_a, "rows": a_rows(a["off"], True)},
        {"caption": ("穩健性（只描述）：拿掉幣安 BTC 免手續費期間（" + f"{ZERO_FEE[0]} 到 {ZERO_FEE[1]} UTC，"
                     "這段期間主動成交量被灌大），以及依流動性（單純衝擊＝每小時漲跌幅 ÷ 成交金額）分兩半。"),
         "columns": ["日子", "天數｜稀少天數", "之後 30 天：稀少｜全部；差", "配對後 30 天差"], "rows": rows_rob},
        {"caption": (f"測試 B：公平考試（main_signal_check 的做法，接法同 supply_demand）。起算月從 {_d(r['first_month'])[:7]} 開始"
                     f"（賣方稀少第一次算得出來之後的第一個月），比特幣日線起算月 {x1['d_n']} 個（到 {msc.DAILY_STARTS[1][:7]}），"
                     f"起算月×24 個收盤時間 {x1['c_n']:,} 格，一般人做法（晚 10 小時、成本 {msc.P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個收盤時間 {msc.SIMS} 次），其他資產（{others}；各用自己的 50、20 日均線，配比特幣的「賣方稀少」當全市場訊號）。"
                     "50 日均線用完全相同的起算日重算。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項。"),
         "columns": ["規則", "日線起算日：贏抱著", "起算日×收盤時間：贏抱著", "一般人做法：贏抱著", "其他資產：贏抱著",
                     "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "每年比抱著多（中位數）", "和均線不同的天數",
                     "隨機錯位比例"],
         "rows": rows_b},
        {"caption": (f"測試 B 的通過條件（事先定好）。隨機錯位：把「賣方稀少」整條訊號往後挪 {GAP} 天以上（頭尾相接），{DRAWS} 次，"
                     f"整個考試重算（一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的 I1 也用同樣方法重算來比）。"),
         "columns": ["條件", "數字", "結果"], "rows": rows_pass},
        {"caption": ("只描述：rally_sources 的 12 段大漲（低點後漲 80% 以上），在低點、50 日均線亮的那天、高點時的 ASYM，"
                     "以及它在過去一年裡的排名；最後兩欄是「低點到均線亮」和「均線亮到高點」這兩段裡賣方稀少的日子占幾成。"
                     "如果「早期每塊錢推得動更多」的線索是真的，應該看到前段高、後段低。"),
         "columns": ["大漲（低點日）", "低點", "均線亮", "高點", "低點→均線亮：稀少的日子", "均線亮→高點：稀少的日子"],
         "rows": r["rally"]},
        {"caption": (f"只描述：I1 比 50 日均線多持有的日子（{sd.GRID[0]} 起，比特幣日線），以及那些日子隔天的平均漲跌。"
                     f"對照：同期間每天平均 {_p(e['avg_all'])}，均線持有的日子平均 {_p(e['avg_ma_on'])}。"),
         "columns": ["年", "多持有的天數", "隔天平均漲跌"], "rows": rows_x},
        {"caption": ("只描述：I1 提早上車（均線還在空手時 I1 轉成持有）對不對。「對」＝30 天內均線也上車，而且那時價格比提早上車時高。"
                     "每次的結果＝從提早上車到 I1 下車或均線上車（先到的那個）之間的漲跌，扣掉買賣成本；「接起來」是全部連乘。"),
         "columns": ["結果", "次數", "平均漲跌", "接起來"], "rows": rows_e},
        {"caption": "只描述：每一次提早上車。", "columns": ["日期", "對／錯", "漲跌"], "rows": rows_el or [["—", "—", "—"]]},
        {"caption": (f"只描述：誰先動？ASYM 和「k 天後結束的 7 天比特幣漲跌」的相關（k 正＝ASYM 在前，k 負＝價格在前）。"
                     f"雜訊帶＝把 ASYM 錯開 {GAP} 天以上、{DRAWS} 次，相關絕對值的第 95 百分位：{ll['band']:.2f}。"
                     "注意 ASYM 本身就用到過去 14 天的漲跌，k 在 −20 到 +6 之間兩段時間有重疊，本來就會有關；"
                     "只有 k ≥ 7（之後的 7 天完全在 ASYM 之後）才算真的「ASYM 在前」。"),
         "columns": ["k", "相關", "和雜訊比"], "rows": rows_l},
    ]

    # ----- verdict -----
    fails_a = ([] if ap["up_full"] else ["全期沒有比較高"]) + ([] if ap["up_subs"] >= 2 else ["三段裡高的不到兩段"]) \
        + ([] if ap["excl"] else ["配對後的區間含零"])
    first = (f"測試 A（{_d(a['start'])} 起）：賣方稀少的日子之後 30 天平均 {_p(full[30]['s'])}，全部日子 {_p(full[30]['all'])}"
             f"（差 {_pt(full[30]['diff'])} {_ci(*full[30]['ci'], f=_pt)}）；三段裡 {ap['up_subs']} 段較高；"
             f"配對過去 14 天漲幅後差 {_pt(full['m_30'][0])} {_ci(*full['m_30'][1], f=_pt)}。"
             + ("事先定的條件通過。" if ap["pass"] else f"事先定的條件沒有通過（{'、'.join(fails_a)}）。"))
    fails_b = ([] if score_ok else ["排名分數輸均線"]) + ([] if loss_ok else ["最深虧損比均線深超過 5 點"]) \
        + ([] if plc_ok else ["隨機錯位超過 10%"])
    second = (f"測試 B：I1 排名分數 {_share1(x1['low'])}（均線 {_share1(base['low'])}），最深虧損中位數 {_pc(x1['c_deep_med'])}"
              f"（均線 {_pc(base['c_deep_med'])}），隨機錯位 {_share(r['placebo'])}，"
              + ("三條都通過，只能算是可以放進每日影子紀錄觀察的候選。" if b_pass else f"沒有通過（{'、'.join(fails_b)}）。"))
    third = (f"I1 提早上車 {len(ent)} 次，對 {len(right)} 次、錯 {len(wrong)} 次（錯的平均 {_p(avg(wrong))}）。"
             + ("ASYM 領先價格（不重疊的 7 天以後）的相關超出雜訊。" if lead_out
                else "ASYM 和價格只在時間重疊的那幾天有關，之後的 7 天以後沒有看得出來的領先。"))
    verdict = first + second + third

    # ----- today -----
    word = {1: "持有", 0: "現金"}
    ma_now, i1_now = int(r["pos"]["ma50"][-1] > 0.5), int(r["pos"]["i1"][-1] > 0.5)
    t = f.iloc[-1]
    th = r["thr"].iloc[-1]
    fl = flag.iloc[-1]
    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"},
           {"label": "比特幣：50 日均線｜I1", "value": f"{word[ma_now]}｜{word[i1_now]}"},
           {"label": "今天的 ASYM（買進推得動 ÷ 賣出壓得下，取對數；正＝賣方少）",
            "value": (f"{t['asym']:+.2f}（門檻 {th:+.2f}，過去一年第 80 百分位）" if _ok(t["asym"]) and _ok(th)
                      else "缺（兩邊衝擊有一邊不是正的，或資料不足）")},
           {"label": "賣方稀少（ASYM 高於門檻）", "value": "—" if not _ok(fl) else ("是" if fl > 0.5 else "否")},
           {"label": "站上 20 日均線", "value": "是" if sd._above20(close.values)[-1] > 0.5 else "否"},
           {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後的狀態，不是買賣建議；I1 沒有通過考試前不會用"}]

    notes = [
        "主動買＝直接照賣方掛的價格買進的人；主動賣＝直接照買方掛的價格賣出的人。每小時的「淨買」＝主動買的金額 − 主動賣的金額（美元）。",
        "ASYM 怎麼算：過去 14 天（336 根 1 小時線）裡，把淨買為正的小時和淨買為負的小時分開。買進衝擊＝淨買小時的漲幅加總 ÷ 淨買金額加總"
        "（每一塊錢淨買推高多少）；賣出衝擊＝淨賣小時的跌幅加總 ÷ 淨賣金額加總（每一塊錢淨賣壓低多少）。ASYM＝log（買進衝擊 ÷ 賣出衝擊），"
        "兩個都要是正的才算。ASYM 高＝買一塊錢推得動、賣一塊錢壓不下＝願意賣的人少（或有人在接）。",
        "為什麼這樣做：supply_demand 用「轉進交易所少＋MVRV 低」猜賣方少，結果抓到的是空頭；rally_sources 的探索表看到大漲早期每塊錢流入"
        "推動的市值比後期大很多（例如 43.8 倍對 5.6 倍、100.6 倍對 24.7 倍）。所以這裡直接從價格對買賣的反應量賣方多寡，不從持有量去猜。"
        "和 order_flow 的差別：order_flow 看的是主動買的比例（買了多少），這裡看的是每一塊錢買賣讓價格動多少。",
        "14 天、第 80 百分位、過去 365 天、至少 300 個小時、20 日均線，都是跑之前憑道理定的，沒有試過其他數字。",
        "只用一家交易所（幣安）的一個交易對：別的交易所、別的穩定幣交易對、合約、ETF 和場外的買賣都不在裡面。幣安的市占率這些年變過；"
        f"{ZERO_FEE[0]} 到 {ZERO_FEE[1]}（UTC）BTCUSDT 免手續費（幣安官方公告），這段期間主動成交量被灌大，穩健性表有拿掉這段的結果。",
        "其他幣和 MSTR 沒有用自己的買賣資料，用的是比特幣的「賣方稀少」（當成全市場的狀態）；MSTR 用前一天的訊號（美股收盤在 UTC 0 點之前）。",
        "就算通過，也只代表 I1 可以放進每日影子紀錄觀察；用的是同一段歷史，真正的考驗是之後每天實際記下的結果。",
        "相鄰的日子、起算日、收盤時間共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。過去的結果不保證未來，這裡不是買賣建議。",
        f"這次計算約 {r['secs'] / 60:.1f} 分鐘。",
    ]
    notes += r["problems"]

    return {
        "id": "price_impact",
        "title": "買賣的價格衝擊：直接量賣方多不多",
        "question": ("每一塊錢的淨買進把價格推高多少、每一塊錢的淨賣出把價格壓低多少？買進推得動、賣出壓不下（賣方稀少）的時候，"
                     "比特幣之後會不會漲比較多？能不能靠它比 50 日均線早上車？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 K 線 API（BTCUSDT 1 小時線，含成交金額與主動買進金額）",
             "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"},
            {"name": "幣安現貨 API 文件：K 線欄位", "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "幣安公告：Binance Launches Zero-Fee Bitcoin Trading（2022-07-06）", "url": URL_LAUNCH},
            {"name": "幣安公告：Updates on Zero-Fee Bitcoin Trading & BUSD Zero Maker Fee Promotion（2023-03-15）", "url": URL_END},
            {"name": "幣安現貨日線（ETH、SOL、BNB、XRP 對 USDT）", "url": "https://api.binance.com/api/v3/klines?interval=1d"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前，只用來暖機）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
