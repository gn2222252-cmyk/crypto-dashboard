"""Historical study "attention": public attention measured by Wikipedia page views of the article "Bitcoin", and
what bitcoin did afterwards. Kristoufek (2013), "BitCoin meets Google Trends and Wikipedia", Scientific Reports
3:3415, found price and search / page-view interest strongly linked in both directions, and an asymmetry: "while the
prices are high (above trend), the increasing interest pushes the prices further atop", and "if the prices are below
their trend, the growing interest pushes the prices even deeper". In the paper "above trend" is a dummy equal to one
when the price is above its moving average of 4 (weekly Google Trends) or 7 (daily Wikipedia views) observations; the
sample is 2011-05-01 ... 2013-06-30, Wikipedia daily data in a VECM(7) on the logarithms. The abstract calls the
asymmetry pronounced, but for Wikipedia the paper reports the two feedbacks as about +0.05 and -0.05, "practically
symmetric around the zero reaction", and no significant effect of views on prices without the split.

Data: Wikimedia REST pageviews API (agent=user, all-access), from 2015-07-01 (the older stats.grok.se counts are gone,
so the sample starts 2015-07). Articles: en "Bitcoin" (main), en "Ethereum" (for ETH), and as checks en
"Cryptocurrency", de "Bitcoin", ja "ビットコイン", ko "비트코인", zh "比特币". Google Trends is not used (no official free
API). Prices: the dashboard's long daily bitcoin series, Binance hourly through clock_robustness, Binance ETHUSDT.

Pre-registered tests (written before any result was looked at):
Attention A_t = log 7-day mean pageviews; ΔA = 7-day change of A; "above trend" = BTC close above its 50-day average
(state Kristoufek's actual definition too).
1. Lead-lag: correlations of ΔA_t with BTC returns over the next 7/30 days and of past BTC returns with ΔA (does
   attention follow price?); split 2015-07..2019, 2020..2022, 2023..now; block bootstrap.
2. Kristoufek asymmetry: forward 7/30-day returns in the four cells (above/below trend × attention rising/falling),
   with intervals and episode counts.
3. Froth: A_t in its trailing-2-year top 5% (attention spike) → forward 30/90-day returns vs all days.
4. Rules on top of ma50 (act at the next close; costs 0.10/0.30%): W1 ma50 but cash during attention spikes (test 3
   condition); W2 = ma50 AND NOT(above-trend & ΔA in its trailing-year bottom 20%) — i.e. leave rallies that lose
   attention. Fair grid where data allows (from 2016), ordinary-person execution, ETH with its own article, placebo
   with circularly shifted attention (200 draws), crash-avoided share with whipsaws.
Pass test (stated now): an effect with the same sign in all three periods and full-sample 90% interval excluding
zero, AND a rule beats ma50 on the ranking score with placebo share ≤ 10%, AND same direction on ETH.

Details fixed with the tests: pageviews of day t are published after day t ends (UTC), so a decision at the close of
day t uses views up to day t-1; on the 24 closing hours of the hourly grid, a close falling on UTC date E uses views up
to E-2 (the same rule, conservative for every hour). "Rising" = ΔA > 0. Trailing percentiles need at least one year
(spike: 2-year window, at least 365 days; ΔA: 1-year window, at least 180 days), so the rules exist from 2016-07 and
the fair grid starts 2016-09. Where the attention value is missing the rules fall back to plain ma50. Bootstrap:
circular moving blocks of 60 days (120 for 90-day returns), 1,000 draws, fixed seed. The three "effects" checked for
the first pass condition are: corr(ΔA, next 30 days), the asymmetry interaction (rising minus falling above trend)
minus (rising minus falling below trend) on next 30 days, and spike-minus-all on next 90 days. Ranking score =
main_signal_check's (lowest of four "beats holding" shares; the fourth is ETH with its own article only), with ma50
recomputed on the same start dates; ordinary person costs 0.30%. Placebo share = share of the 200 shifted-attention
versions whose median yearly edge over holding on the 1,728 textbook cells is at least the real one. "Same direction
on ETH" = the passing effect has the same sign with ETH price and the "Ethereum" article, and the passing rule beats
holding on more ETH start months than ma50 does.
"""
from __future__ import annotations

import json
import math
from urllib.parse import quote

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{art}/"
       "daily/{a}/{b}")
FIRST_VIEW = "20150701"
ARTICLES = {"btc": ("en", "Bitcoin", "英文維基「Bitcoin」"), "eth": ("en", "Ethereum", "英文維基「Ethereum」"),
            "crypto": ("en", "Cryptocurrency", "英文維基「Cryptocurrency」"), "de": ("de", "Bitcoin", "德文維基「Bitcoin」"),
            "ja": ("ja", "ビットコイン", "日文維基「ビットコイン」"), "ko": ("ko", "비트코인", "韓文維基「비트코인」"),
            "zh": ("zh", "比特币", "中文維基「比特币」")}
CHECKS = ("crypto", "de", "ja", "ko", "zh")
COST, P_COST, P_MISS, SIMS, SEED = 0.001, 0.003, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
SPIKE_WIN, SPIKE_MIN, SPIKE_TOP = 730, 365, 0.95
FALL_WIN, FALL_MIN, FALL_BOTTOM = 365, 180, 0.20
PERIODS = (("2015-07 到 2019", "2015-07-01", "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"),
           ("2023 到今天", "2023-01-01", "2100-01-01"))
BOOT, BOOT_SEED = 1000, 3415
BLOCK = {7: 60, 30: 60, 90: 120}
PLACEBO, PLACEBO_GAP = 200, 180
GRID_FIRST = "2016-09-01"
RULES = ("ma50", "w1", "w2")
KI = {"ma50": 1, "w1": 61, "w2": 62}  # ma50 uses main_signal_check's random draws
NAMES = {"ma50": "50 日均線（基準）", "w1": "W1：50 日均線，但關注度爆量時改拿現金",
         "w2": "W2：50 日均線，但漲勢中關注度一週掉到一年內最低兩成時下車"}
SHORT = {"ma50": "50 日均線", "w1": "W1 爆量避開", "w2": "W2 漲勢失去關注就下車"}
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


def _r(v) -> str:
    return f"{v:+.2f}" if _ok(v) else "—"


def _ci(lo, hi, f) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _p(v) -> str:
    """A return given as a fraction, shown in percent."""
    return _pct(v * 100) if _ok(v) else "—"


def _pt(v) -> str:
    """A difference of two returns (fractions), shown in percentage points."""
    return _pp(v * 100) if _ok(v) else "—"


# ---------- data ----------

def _views(key: str) -> tuple[pd.Series, str | None]:
    """Daily user page views of one article from 2015-07-01, one row per day (a missing day repeats the last)."""
    lang, art, label = ARTICLES[key]
    path = common.STORE / f"attention_{key}.json"
    url = API.format(lang=lang, art=quote(art, safe=""), a=FIRST_VIEW, b=common.today_utc().replace("-", ""))
    problem = None
    try:
        items = common.get_json(url)["items"]
        got = {f"{it['timestamp'][:4]}-{it['timestamp'][4:6]}-{it['timestamp'][6:8]}": float(it["views"])
               for it in items}
        if len(got) < 1000:
            raise RuntimeError("資料太短")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"維基媒體瀏覽量 API 取不到{label}（{type(exc).__name__}），也沒有存下來的資料") from exc
        problem = f"{label}這次取不到（{type(exc).__name__}），用之前存下來的資料（到 {max(got)}）。"
    s = pd.Series(got, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().asfreq("D").ffill(), problem


def _trail_pct(x: np.ndarray, win: int, minp: int) -> np.ndarray:
    """Share of the last win values (this one included, missing ones skipped) that are <= this value."""
    x = np.asarray(x, dtype=float)
    pad = np.concatenate([np.full(win - 1, np.nan), x])
    w = np.lib.stride_tricks.sliding_window_view(pad, win)
    cnt = (~np.isnan(w)).sum(1)
    le = (w <= x[:, None]).sum(1)
    return np.where((cnt >= minp) & ~np.isnan(x), le / np.maximum(cnt, 1), np.nan)


def _features(v: pd.Series) -> pd.DataFrame:
    """Indexed by the day the views belong to; every value uses views up to that day only."""
    a = np.log(v.rolling(7, min_periods=7).mean().clip(lower=1.0))
    d = a - a.shift(7)
    pa = _trail_pct(a.values, SPIKE_WIN, SPIKE_MIN)
    pdd = _trail_pct(d.values, FALL_WIN, FALL_MIN)
    return pd.DataFrame({"a": a.values, "d": d.values, "pa": pa, "pd": pdd,
                         "spike": np.where(np.isnan(pa), np.nan, (pa >= SPIKE_TOP).astype(float)),
                         "fall": np.where(np.isnan(pdd), np.nan, (pdd <= FALL_BOTTOM).astype(float))}, index=v.index)


def _at(col: pd.Series, dates, lag: int) -> np.ndarray:
    """Values usable at decisions dated `dates` when the newest usable view day is `lag` days earlier."""
    return col.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


# ---------- rules ----------

def _rules(close: np.ndarray, spike: np.ndarray, fall: np.ndarray) -> dict[str, np.ndarray]:
    """Wanted position after each close (NaN until the 50-day average exists)."""
    c = np.asarray(close, dtype=float)
    ma = ct._gt(c, ct._mean(c, 50))
    s, f = np.nan_to_num(spike, nan=0.0), np.nan_to_num(fall, nan=0.0)
    return {"ma50": ma, "w1": ma * (1 - s), "w2": ma * (1 - f)}


def _daily_rules(close: pd.Series, feat: pd.DataFrame) -> dict[str, np.ndarray]:
    return _rules(close.values, _at(feat["spike"], close.index, 1), _at(feat["fall"], close.index, 1))


def _daily(close: pd.Series, feat: pd.DataFrame, first, last) -> dict:
    tg = _daily_rules(close, feat)
    change = close.pct_change().fillna(0.0).values
    known = ~np.isnan(tg["ma50"]) & ~np.isnan(_at(feat["spike"], close.index, 1)) & \
        ~np.isnan(_at(feat["fall"], close.index, 1))
    begin = max(close.index[int(np.argmax(known))] + DAY, pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels, "pos": tg}
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in RULES:
        net, _ = te._run(np.concatenate([[np.nan], tg[k][:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    """The 24 closing hours of the hourly grid with the 50-day rule precomputed per hour."""

    def __init__(self, price, b0, pre):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]])
            self.ma.append(np.nan_to_num(ct._gt(c, ct._mean(c, 50))[npre:][:self.days], nan=0.0))

    def flags(self, feat: pd.DataFrame) -> dict[str, np.ndarray]:
        """A close on UTC date E uses views up to E-2, for every closing hour."""
        return {k: np.nan_to_num(_at(feat[k], self.m.dates, 2), nan=0.0) for k in ("spike", "fall")}

    def want(self, k: str, h: int, fl: dict) -> np.ndarray:
        if k == "ma50":
            return self.ma[h]
        return self.ma[h] * (1 - fl["spike" if k == "w1" else "fall"])

    def text_edge(self, k: str, fl: dict) -> float:
        edge = []
        for h in range(24):
            r, _ = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, self.want(k, h, fl), COST, None)]),
                               self.starts, YEAR)
            edge.append(r[:, 0] - self.hr)
        return float(np.median(np.concatenate(edge)))

    def full(self, k: str, fl: dict) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, SIMS)), np.zeros((n, 24, SIMS))
        for h in range(24):
            w = self.want(k, h, fl)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[k] + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), self.eng._mid(b, pos, P_COST, None)]), self.starts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - self.hr[:, None], pr - self.hr[:, None, None]
        return {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge))}


# ---------- statistics ----------

def _idx(n: int, block: int, rng) -> np.ndarray:
    nb = -(-n // block)
    st = rng.integers(0, n, (BOOT, nb))
    return ((st[:, :, None] + np.arange(block)) % n).reshape(BOOT, -1)[:, :n]


def _boot(cols: dict, fn, block: int) -> tuple:
    """(estimate, 5th, 95th percentile, n) of fn over circular moving-block resamples of the rows."""
    ok = np.ones(len(next(iter(cols.values()))), dtype=bool)
    for v in cols.values():
        ok &= np.isfinite(v)
    cols = {k: np.asarray(v, dtype=float)[ok] for k, v in cols.items()}
    n = int(ok.sum())
    if n < 60:
        return (None, None, None, n)
    with np.errstate(all="ignore"):
        est = float(fn({k: v[None, :] for k, v in cols.items()})[0])
        idx = _idx(n, block, np.random.default_rng(BOOT_SEED))
        b = fn({k: v[idx] for k, v in cols.items()})
    b = b[np.isfinite(b)]
    if not math.isfinite(est) or len(b) < BOOT // 2:
        return (est if math.isfinite(est) else None, None, None, n)
    return (est, float(np.percentile(b, 5)), float(np.percentile(b, 95)), n)


def _corr_fn(d):
    x = d["x"] - d["x"].mean(1, keepdims=True)
    y = d["y"] - d["y"].mean(1, keepdims=True)
    return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))


def _mmean(y, m):
    return (y * m).sum(1) / m.sum(1)


def _cell_fn(d):
    return _mmean(d["y"], d["m"])


def _inter_fn(d):
    up, ri = d["up"], d["ri"]
    return ((_mmean(d["y"], up * ri) - _mmean(d["y"], up * (1 - ri)))
            - (_mmean(d["y"], (1 - up) * ri) - _mmean(d["y"], (1 - up) * (1 - ri))))


def _froth_fn(d):
    return _mmean(d["y"], d["m"]) - d["y"].mean(1)


def _runs(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool).astype(int)
    return int((np.diff(np.concatenate([[0], m])) == 1).sum())


def _frame(close: pd.Series, feat: pd.DataFrame) -> pd.DataFrame:
    """One row per decision close t: price state at t, attention usable at t (views to t-1), forward returns."""
    idx = close.index
    df = pd.DataFrame(index=idx)
    m50 = close.rolling(50, min_periods=50).mean()
    m7 = close.rolling(7, min_periods=7).mean()
    df["up"] = (close > m50).astype(float).where(m50.notna())
    df["up7"] = (close > m7).astype(float).where(m7.notna())
    for h in (7, 30, 90):
        df[f"f{h}"] = close.shift(-h) / close - 1
    df["past7"] = np.log(close / close.shift(7))
    df["d"] = _at(feat["d"], idx, 1)
    df["spike"] = _at(feat["spike"], idx, 1)
    df["d_next"] = _at(feat["d"], idx, -7)   # views t+1..t+7 against t-6..t (research only, never traded)
    df["d_same"] = _at(feat["d"], idx, 0)    # views t-6..t against t-13..t-7 (research only)
    df["ri"] = (df["d"] > 0).astype(float).where(df["d"].notna())
    first = feat.index[0] + 15 * DAY
    return df[df.index >= first]


def _subset(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]


def _leadlag(df: pd.DataFrame) -> dict:
    return {"n": int(df["d"].notna().sum()),
            "f7": _boot({"x": df["d"].values, "y": df["f7"].values}, _corr_fn, BLOCK[7]),
            "f30": _boot({"x": df["d"].values, "y": df["f30"].values}, _corr_fn, BLOCK[30]),
            "next": _boot({"x": df["past7"].values, "y": df["d_next"].values}, _corr_fn, BLOCK[7]),
            "same": _boot({"x": df["past7"].values, "y": df["d_same"].values}, _corr_fn, BLOCK[7])}


def _inter(df: pd.DataFrame, h: int, up: str = "up") -> tuple:
    return _boot({"y": df[f"f{h}"].values, "up": df[up].values, "ri": df["ri"].values}, _inter_fn, BLOCK[h])


def _cells(df: pd.DataFrame) -> dict:
    out = {}
    for u in (1, 0):
        for ri in (1, 0):
            m = ((df["up"] == u) & (df["ri"] == ri)).astype(float).where(df["up"].notna() & df["ri"].notna())
            out[(u, ri)] = {"days": int((m == 1).sum()), "runs": _runs((m == 1).values),
                            **{f"f{h}": _boot({"y": df[f"f{h}"].values, "m": m.values}, _cell_fn, BLOCK[h])
                               for h in (7, 30)}}
    return out


def _froth(df: pd.DataFrame) -> dict:
    m = df["spike"]
    out = {"days": int((m == 1).sum()), "runs": _runs((m == 1).values)}
    for h in (30, 90):
        ok = m.notna() & df[f"f{h}"].notna()
        y = df[f"f{h}"][ok]
        out[f"s{h}"] = float(y[m[ok] == 1].mean()) if (m[ok] == 1).any() else None
        out[f"a{h}"] = float(y.mean()) if len(y) else None
        out[f"d{h}"] = _boot({"y": df[f"f{h}"].values, "m": m.values}, _froth_fn, BLOCK[h])
    return out


# ---------- look-ahead check ----------

def _lookahead(close: pd.Series, views: pd.Series, clock: _Clock) -> tuple[int, int]:
    """Scramble closes after a cut close and views from the cut day on: positions up to the cut must not move.
    On the clock grid, views from E-1 on are scrambled for a cut close on date E."""
    base = _daily_rules(close, _features(views))
    rng = np.random.default_rng(11)
    bad = cuts = 0
    for cut in (len(close) - 2000, len(close) - 900, len(close) - 60):
        t = close.index[cut]
        c2 = close.copy()
        c2.iloc[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(close) - cut - 1))
        v2 = views.copy()
        later = v2.index >= t
        v2[later] = v2[later].values * np.exp(rng.normal(0, 1.0, int(later.sum())))
        alt = _daily_rules(c2, _features(v2))
        cuts += 1
        for k in RULES:
            if not np.array_equal(np.nan_to_num(base[k][:cut + 1], nan=-1), np.nan_to_num(alt[k][:cut + 1], nan=-1)):
                bad += 1
        j = int(clock.m.dates.searchsorted(t))
        if 0 < j < clock.days:
            v3 = views.copy()
            later = v3.index >= clock.m.dates[j] - DAY
            v3[later] = v3[later].values * np.exp(rng.normal(0, 1.0, int(later.sum())))
            f0, f1 = clock.flags(_features(views)), clock.flags(_features(v3))
            cuts += 1
            for k in ("spike", "fall"):
                if not np.array_equal(f0[k][:j + 1], f1[k][:j + 1]):
                    bad += 1
    return cuts, bad


# ---------- the computation ----------

def _compute() -> dict:
    problems = []
    views, feats = {}, {}
    for key in ARTICLES:
        try:
            views[key], problem = _views(key)
            feats[key] = _features(views[key])
            if problem:
                problems.append(problem)
        except Exception as exc:  # noqa: BLE001 - the checks may fail one by one; the main article may not
            if key == "btc":
                raise
            problems.append(f"{ARTICLES[key][2]}算不出來（{type(exc).__name__}），沒有算進去。")
    close = te._daily_close()
    eth = None
    try:
        eth = ta._crypto_close("ETHUSDT")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ETH 價格這次取不到（{type(exc).__name__}），ETH 的檢查沒有算。")
    if "eth" not in feats:
        eth = None

    # tests 1-3
    df = _frame(close, feats["btc"])
    lead = {"全部": _leadlag(df)}
    for name, a, b in PERIODS:
        lead[name] = _leadlag(_subset(df, a, b))
    cells = _cells(df)
    cells_p = {name: _cells(_subset(df, a, b)) for name, a, b in PERIODS}
    inter = {h: _inter(df, h) for h in (7, 30)}
    inter_k = {h: _inter(df, h, "up7") for h in (7, 30)}
    inter_p = {name: {h: _inter(_subset(df, a, b), h) for h in (7, 30)} for name, a, b in PERIODS}
    inter_kp = {name: _inter(_subset(df, a, b), 30, "up7")[0] for name, a, b in PERIODS}
    froth = {"全部": _froth(df)}
    for name, a, b in PERIODS:
        froth[name] = _froth(_subset(df, a, b))
    checks = {}
    for key in CHECKS:
        if key in feats:
            dk = _frame(close, feats[key])
            checks[key] = {"lead": _leadlag(dk)["f30"], "inter": {h: _inter(dk, h) for h in (7, 30)},
                           "froth": _froth(dk)}
    edf = None
    eth_res = {}
    if eth is not None:
        edf = _frame(eth, feats["eth"])
        eth_res = {"lead": _leadlag(edf), "cells": _cells(edf), "inter": {h: _inter(edf, h) for h in (7, 30)},
                   "froth": _froth(edf)}

    # test 4: fair grid
    daily = _daily(close, feats["btc"], GRID_FIRST, msc.DAILY_STARTS[1])
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
    fl = clock.flags(feats["btc"])
    grid = {k: clock.full(k, fl) for k in RULES}
    eth_grid = None
    if eth is not None:
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        eth_grid = _daily(eth, feats["eth"], eth.index[0], three)

    s = {}
    for k in RULES:
        x = dict(grid[k])
        dr, dd = daily[k]
        x["d_n"], x["d_beat"], x["d_shallow"] = len(dr), int((dr > daily["hold"][0]).sum()), \
            int((dd > daily["hold"][1]).sum())
        if eth_grid is not None:
            er = eth_grid[k][0]
            x["e_n"], x["e_beat"] = len(er), int((er > eth_grid["hold"][0]).sum())
            x["e_edge"] = float(np.median(er - eth_grid["hold"][0]))
        else:
            x["e_n"], x["e_beat"], x["e_edge"] = 0, 0, None
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       x["e_beat"] / x["e_n"] if x["e_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        s[k] = x

    # placebo: circularly shifted attention
    rng = np.random.default_rng(SEED + 3415)
    n = len(views["btc"])
    plac = {"w1": [], "w2": []}
    for _ in range(PLACEBO):
        shift = int(rng.integers(PLACEBO_GAP, n - PLACEBO_GAP))
        f = _features(pd.Series(np.roll(views["btc"].values, shift), index=views["btc"].index))
        fp = clock.flags(f)
        for k in plac:
            plac[k].append(clock.text_edge(k, fp))
    placebo = {k: float(np.mean(np.array(v) >= s[k]["c_edge"] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    # crashes and whipsaws since the rules exist
    c = close.values.astype(float)
    pos = daily["pos"]
    first = int(close.index.searchsorted(pd.Timestamp(GRID_FIRST)))
    bears = [b for b in ct._bears(close) if b["top"] >= first]
    crash = {k: [ct._crash(c, [(1.0, pos[k])], b) for b in bears] for k in RULES}
    whip = {k: ct._whipsaws(c, [(1.0, pos[k])], first) for k in RULES}
    differ = {k: float(np.mean(np.nan_to_num(pos[k][first:]) != np.nan_to_num(pos["ma50"][first:]))) for k in RULES}

    return {"close": close, "views": views, "feats": feats, "df": df, "lead": lead, "cells": cells,
            "cells_p": cells_p, "inter": inter, "inter_k": inter_k, "inter_kp": inter_kp, "inter_p": inter_p, "froth": froth,
            "checks": checks, "eth": eth_res, "daily": daily, "clock": clock, "s": s, "placebo": placebo,
            "placebo_med": placebo_med, "bears": bears, "crash": crash, "whip": whip, "differ": differ,
            "eth_grid": eth_grid, "problems": problems,
            "look": _lookahead(close, views["btc"], clock)}


# ---------- the study ----------

def _sign(v) -> int:
    return 0 if not _ok(v) or v == 0 else (1 if v > 0 else -1)


def study() -> dict:
    r = _compute()
    close, df, s = r["close"], r["df"], r["s"]
    lead, cells, inter, froth, eth = r["lead"], r["cells"], r["inter"], r["froth"], r["eth"]
    pnames = [p[0] for p in PERIODS]

    # ----- pass test -----
    effects = {
        "lead": ("關注度一週變化和之後 30 天報酬的相關", lead["全部"]["f30"], [lead[p]["f30"][0] for p in pnames],
                 eth.get("lead", {}).get("f30", (None,))[0] if eth else None),
        "asym": ("論文說的不對稱（漲勢裡關注上升減下降，再減跌勢裡的同一個差）之後 30 天", inter[30],
                 [r["inter_p"][p][30][0] for p in pnames], eth.get("inter", {}).get(30, (None,))[0] if eth else None),
        "froth": ("關注度爆量後 90 天報酬減全部日子", froth["全部"]["d90"], [froth[p]["d90"][0] for p in pnames],
                  eth.get("froth", {}).get("d90", (None,))[0] if eth else None),
    }
    part1 = []
    for k, (_, full, per, _e) in effects.items():
        est, lo, hi, _n = full
        sg = _sign(est)
        if sg and _ok(lo) and _ok(hi) and (lo > 0 or hi < 0) and all(_sign(v) == sg for v in per):
            part1.append(k)
    part1_eth = [k for k in part1 if _sign(effects[k][3]) == _sign(effects[k][1][0])]
    part2 = [k for k in ("w1", "w2") if s[k]["low"] > s["ma50"]["low"] + 1e-12 and r["placebo"][k] <= 0.10]
    part2_eth = [k for k in part2 if s[k]["e_n"] and s[k]["e_beat"] > s["ma50"]["e_beat"]]
    passed = bool(part1_eth and part2_eth)

    # ----- table 1: lead-lag -----
    def cc(t) -> str:
        return f"{_r(t[0])} {_ci(t[1], t[2], _r)}"

    rows1 = []
    for p in pnames + ["全部"]:
        x = lead[p]
        rows1.append([f"比特幣 {p}" if p != "全部" else f"比特幣 全部（{_d(df.index[0])} 起）", f"{x['n']:,}",
                      cc(x["f7"]), cc(x["f30"]), cc(x["next"]), cc(x["same"])])
    if eth:
        x = eth["lead"]
        rows1.append(["ETH 全部（「Ethereum」頁面配幣安 ETH 價格，2017-08 起）", f"{x['n']:,}", cc(x["f7"]), cc(x["f30"]),
                      cc(x["next"]), cc(x["same"])])
    for key, ck in r["checks"].items():
        rows1.append([f"對照：{ARTICLES[key][2]}配比特幣價格", "—", "—", cc(ck["lead"]), "—", "—"])

    # ----- table 2: four cells -----
    cell_name = {(1, 1): "收盤在 50 日均線上、關注上升", (1, 0): "收盤在 50 日均線上、關注下降",
                 (0, 1): "收盤在 50 日均線下、關注上升", (0, 0): "收盤在 50 日均線下、關注下降"}

    def mc(t) -> str:
        return f"{_p(t[0])} {_ci(t[1], t[2], _p)}"

    def ic(t) -> str:
        return f"{_pt(t[0])} {_ci(t[1], t[2], _pt)}"

    rows2 = []
    for key in ((1, 1), (1, 0), (0, 1), (0, 0)):
        x = cells[key]
        per = "｜".join(_p(r["cells_p"][p][key]["f30"][0]) for p in pnames)
        e = _p(eth["cells"][key]["f30"][0]) if eth else "—"
        rows2.append([cell_name[key], f"{x['days']:,} 天｜{x['runs']} 段", mc(x["f7"]), mc(x["f30"]), per, e])
    rows2.append(["不對稱（論文的說法要大於 0）：上面兩格的差 − 下面兩格的差", "—", ic(inter[7]), ic(inter[30]),
                  "｜".join(_pt(r["inter_p"][p][30][0]) for p in pnames), _pt(eth["inter"][30][0]) if eth else "—"])
    rows2.append(["同上，但「在趨勢上」改用論文的定義：收盤高於 7 日均線", "—", ic(r["inter_k"][7]), ic(r["inter_k"][30]),
                  "｜".join(_pt(r["inter_kp"][p]) for p in pnames), "—"])
    for key, ck in r["checks"].items():
        rows2.append([f"對照：不對稱，關注度改用{ARTICLES[key][2]}", "—", ic(ck["inter"][7]), ic(ck["inter"][30]), "—", "—"])

    # ----- table 3: froth -----
    def fr(x, h) -> list:
        return [f"{_p(x[f's{h}'])}｜{_p(x[f'a{h}'])}", ic(x[f"d{h}"])]

    rows3 = []
    for p in ["全部"] + pnames:
        x = froth[p]
        rows3.append([f"比特幣 {p}", f"{x['days']:,} 天｜{x['runs']} 段"] + fr(x, 30) + fr(x, 90))
    if eth:
        x = eth["froth"]
        rows3.append(["ETH（「Ethereum」頁面配 ETH 價格）", f"{x['days']:,} 天｜{x['runs']} 段"] + fr(x, 30) + fr(x, 90))
    for key, ck in r["checks"].items():
        x = ck["froth"]
        rows3.append([f"對照：{ARTICLES[key][2]}", f"{x['days']:,} 天｜{x['runs']} 段"] + fr(x, 30) + fr(x, 90))

    # ----- table 4: rules -----
    done = [i for i, b in enumerate(r["bears"]) if b["done"]]

    def avoid(k):
        vals = [r["crash"][k][i]["avoid"] for i in done]
        return float(np.median(vals)) if vals else None

    rows4 = []
    order = sorted(RULES, key=lambda k: (s[k]["loss_ok"] < LOSS_BAR, -s[k]["low"]))
    for k in RULES:
        x = s[k]
        w = r["whip"][k]
        rows4.append([NAMES[k], f"{_of(x['d_n'], x['d_beat'])}", f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}",
                      f"{_of(x['e_n'], x['e_beat'])}" if x["e_n"] else "—",
                      f"{_share(x['low'])}（第 {order.index(k) + 1}）｜{_share(x['loss_ok'])}",
                      f"{_share(avoid(k))}｜{w['whips']:.1f} 次",
                      "—（基準）" if k == "ma50" else f"{_share(r['placebo'][k])}"])

    n_d = s["ma50"]["d_n"]
    n_c = s["ma50"]["c_n"]
    clock = r["clock"]
    tables = [
        {"caption": ("一、關注度和價格誰帶誰（相關係數，−1 到 +1，0 是沒關係；方括號是 90% 區間，用 60 天一段的區塊重抽 1,000 次）。"
                     "關注度＝維基百科英文「Bitcoin」頁面每日瀏覽量的 7 天平均取對數，一週變化＝和 7 天前比。第三、四欄：在收盤那天"
                     "能拿到的關注度一週變化（瀏覽量只用到前一天），和之後 7／30 天比特幣漲跌的相關；第五欄：過去一週漲跌和「下一週」"
                     "關注度變化的相關（關注是不是跟著價格走）；第六欄：同一週的相關。之後的報酬會跨過期間的邊界、而且天天重疊。"),
         "columns": ["期間", "天數", "關注一週變化 vs 之後 7 天", "關注一週變化 vs 之後 30 天", "過去一週漲跌 vs 下一週關注變化",
                     "同一週：漲跌 vs 關注變化"],
         "rows": rows1},
        {"caption": ("二、論文的不對稱說法照原樣測：每天依「收盤在不在 50 日均線上」和「關注度一週變化是升還是降」分成四格，"
                     "看之後 7／30 天比特幣平均漲跌（方括號是 90% 區間）。段數＝連續落在同一格的段落數，天天重疊，所以真正獨立的樣本"
                     "遠少於天數。第五欄是三段期間各自的之後 30 天平均；最後一欄是 ETH 用自己的「Ethereum」頁面和自己的 50 日均線。"
                     "論文說的若成立，第一格應高於第二格、第三格應低於第四格，也就是「不對稱」那一列要大於 0（單位：百分點）。"),
         "columns": ["分格", "天數｜段數", "之後 7 天平均", "之後 30 天平均", "之後 30 天：" + "｜".join(pnames), "ETH 之後 30 天"],
         "rows": rows2},
        {"caption": (f"三、關注度爆量：7 天平均瀏覽量落在過去兩年（至少要有一年資料，所以 2016-07 起才有）最高 5% 的日子，"
                     "之後 30／90 天平均漲跌和全部日子比（差的單位是百分點，方括號是 90% 區間；90 天用 120 天一段重抽）。"
                     "對照列用別的語言或別的頁面算爆量，價格都是比特幣。"),
         "columns": ["對象", "爆量天數｜段數", "之後 30 天：爆量｜全部", "30 天差", "之後 90 天：爆量｜全部", "90 天差"],
         "rows": rows3},
        {"caption": (f"四、兩條加在 50 日均線上的規則，和 50 日均線同樣條件考：比特幣 {n_d} 個起算月（{GRID_FIRST[:7]} 起，UTC 0 點收盤）、"
                     f"{n_c:,} 格起算月×收盤時間（照課本做，成本 {COST * 100:.2f}%）、同樣 {n_c:,} 格照一般人方式做（10 小時後才做、"
                     f"成本 {P_COST * 100:.2f}%、一成進場漏掉，每個收盤時間 {SIMS} 次）、ETH 起算月（用「Ethereum」頁面）。"
                     "前四項是報酬贏過同一天起一直抱著的比例（第三、四欄後面是每年多賺的中位數，百分點）；排名分數＝四項裡最低的；"
                     f"熊市避開跌幅＝{GRID_FIRST[:4]} 年以後走完的大熊市（跌 50% 以上）中位數；被洗＝下車後 30 天內用更高價買回，每年次數。"
                     f"安慰劑＝把瀏覽量整串循環平移 {PLACEBO} 次（至少錯開 {PLACEBO_GAP} 天）再跑同一條規則，{n_c:,} 格每年多賺中位數"
                     "一樣好或更好的比例，越小越不像運氣。"),
         "columns": ["規則", f"比特幣起算月：贏的", f"{n_c:,} 格照課本：贏的｜多賺", "一般人做法：贏的｜多賺", "ETH 起算月：贏的",
                     "排名分數（名次）｜虧損比抱著淺", "熊市避開跌幅｜每年被洗", "安慰劑：打亂後一樣好的比例"],
         "rows": rows4},
    ]

    # ----- today -----
    feat = r["feats"]["btc"]
    v = r["views"]["btc"]
    t = close.index[-1]
    use_day = t - DAY
    fu = feat.reindex([use_day]).iloc[0]
    last_v = v.index[-1]
    pos = r["daily"]["pos"]

    def word(x) -> str:
        return "—" if not _ok(x) else ("持有" if x >= 0.5 else "現金")

    up_now = df["up"].iloc[-1] if df.index[-1] == t else None
    cell_now = "—"
    if _ok(up_now) and _ok(fu["d"]):
        cell_now = cell_name[(int(up_now), int(fu["d"] > 0))]
    now = [
        {"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"英文維基「Bitcoin」瀏覽量（最新一天 {_d(last_v)}）",
         "value": f"當天 {v.iloc[-1]:,.0f} 次；7 天平均 {v.iloc[-7:].mean():,.0f} 次"},
        {"label": f"規則用的關注度（瀏覽量到 {_d(use_day)}）",
         "value": (f"7 天平均 {math.exp(fu['a']):,.0f} 次，在過去兩年排第 {fu['pa'] * 100:.0f} 百分位；"
                   f"比一週前 {_pct((math.exp(fu['d']) - 1) * 100)}，在過去一年的一週變化裡排第 {fu['pd'] * 100:.0f} 百分位")
         if _ok(fu["a"]) and _ok(fu["pa"]) and _ok(fu["d"]) and _ok(fu["pd"]) else "—"},
        {"label": "現在在第二表的哪一格", "value": cell_now},
        {"label": "關注度爆量（過去兩年最高 5%）", "value": "是" if fu["spike"] == 1 else "否" if fu["spike"] == 0 else "—"},
    ]
    for k in RULES:
        now.append({"label": NAMES[k], "value": word(pos[k][-1])})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    l30 = lead["全部"]["f30"]
    nx = lead["全部"]["next"]
    asym = inter[30]
    per_a = [r["inter_p"][p][30][0] for p in pnames]
    holds = _ok(asym[0]) and asym[0] > 0 and _ok(asym[1]) and asym[1] > 0 and all(_sign(x) == 1 for x in per_a)
    first = (f"2015-07 起，關注度一週變化和之後 30 天比特幣漲跌的相關只有 {_r(l30[0])}（90% 區間 {_r(l30[1])} 到 {_r(l30[2])}），"
             f"過去一週漲跌和下一週關注變化是 {_r(nx[0])}，同一週是 {_r(lead['全部']['same'][0])}。")
    second = (f"論文說的「漲勢中關注上升推更高、跌勢中推更深」照原樣測，不對稱差 {_pt(asym[0])} 個百分點（90% 區間 {_pt(asym[1])} 到 "
              f"{_pt(asym[2])}，三段 {'／'.join(_pt(x) for x in per_a)}），{'成立' if holds else '不成立'}"
              f"（改用論文的 7 日均線當趨勢是 {_pt(r['inter_k'][30][0])}，不是事先定的版本）；"
              f"關注度爆量後 90 天比平常 {_pt(froth['全部']['d90'][0])} 點，2023 年後 {_pt(froth[pnames[2]]['d90'][0])} 點。")
    best = max(("w1", "w2"), key=lambda k: s[k]["low"])
    third = (f"爆量時避開（W1）和漲勢失去關注就下車（W2）的排名分數 {_share(s['w1']['low'])}、{_share(s['w2']['low'])}，"
             f"50 日均線 {_share(s['ma50']['low'])}，安慰劑 {_share(r['placebo']['w1'])}、{_share(r['placebo']['w2'])}；"
             f"事先定的通過條件{'全部過了' if passed else '沒有過'}。")
    verdict = first + second + third

    # ----- notes -----
    cuts, bad = r["look"]
    bear_txt = "；".join(
        f"{_d(close.index[b['top']])} 起" + ("" if b["done"] else "（還在進行，沒算進中位數）") + "："
        + "、".join(f"{SHORT[k]} {_share(r['crash'][k][i]['avoid'])}" for k in RULES)
        for i, b in enumerate(r["bears"])) or "沒有"
    p1 = "、".join(effects[k][0] for k in part1) or "沒有"
    p2 = "、".join(SHORT[k] for k in part2) or "沒有"
    notes = [
        ("事先寫下的檢驗（照原文）：Attention A_t = log 7-day mean pageviews; ΔA = 7-day change of A; \"above trend\" = BTC "
         "close above its 50-day average. 1. Lead-lag: correlations of ΔA_t with BTC returns over the next 7/30 days and of "
         "past BTC returns with ΔA; split 2015-07..2019, 2020..2022, 2023..now; block bootstrap. 2. Kristoufek asymmetry: "
         "forward 7/30-day returns in the four cells (above/below trend × attention rising/falling), with intervals and "
         "episode counts. 3. Froth: A_t in its trailing-2-year top 5% → forward 30/90-day returns vs all days. 4. Rules on "
         "top of ma50 (act at the next close; costs 0.10/0.30%): W1 ma50 but cash during attention spikes; W2 = ma50 AND "
         "NOT(above-trend & ΔA in its trailing-year bottom 20%). Pass test: an effect with the same sign in all three "
         "periods and full-sample 90% interval excluding zero, AND a rule beats ma50 on the ranking score with placebo "
         "share ≤ 10%, AND same direction on ETH."),
        (f"通過條件逐項：第一項（同號且區間不含 0）過的效果：{p1}"
         + (f"，其中 ETH 同方向的：{'、'.join(effects[k][0] for k in part1_eth) or '沒有'}" if part1 else "")
         + f"；第二項（排名分數贏 50 日均線且安慰劑 ≤ 10%）過的規則：{p2}"
         + (f"，其中 ETH 也贏的：{'、'.join(SHORT[k] for k in part2_eth) or '沒有'}" if part2 else "")
         + f"。結論：{'通過' if passed else '沒有通過'}。三個效果在 ETH 上的值：" + "；".join(
             f"{effects[k][0]} {(_r(effects[k][3]) if k == 'lead' else _pt(effects[k][3]))}" for k in effects) + "。"),
        ("論文實際說的：Kristoufek（2013）用 2011-05-01 到 2013-06-30 的資料，Google 搜尋是週資料（VAR(1)、對數差分），"
         "維基百科是日資料（對數、7 階向量誤差修正模型）。「在趨勢上」是價格高於移動平均的虛擬變數：Google 用 4 週、維基百科用 7 天的移動平均。"
         "論文說價格和關注是雙向的；但維基百科部分不分漲跌勢時，看不到瀏覽量對價格有顯著影響，分開後正負回饋約 +0.05 和 −0.05，"
         "論文自己寫「幾乎以零為中心對稱」，摘要卻說不對稱很明顯。這裡事先定的「在趨勢上」是 50 日均線，第二表另列論文的 7 日均線版本。"),
        ("資料從 2015-07 開始：維基媒體官方的每日瀏覽量 API 只回到 2015-07-01（只算真人瀏覽、所有裝置），更早的 stats.grok.se 已經不在了，"
         "所以論文用的 2011–2013 期間這裡無法重做。Google 搜尋趨勢沒有用（沒有官方免費 API，也不去抓網頁）。"),
        ("不偷看未來：某一天的瀏覽量要等那天（UTC）結束才會公布，所以收盤當天的決定只用到前一天的瀏覽量；換收盤時間時，"
         "落在 UTC 日期 E 的收盤一律只用到 E−2 的瀏覽量（對每個時間都保守）。"
         f"檢查：把 {cuts} 個切點之後的價格和瀏覽量隨機打亂再算一次，切點以前的持有狀態和關注旗號"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 處變了，這次結果不可信。")),
        (f"規則和 50 日均線持有狀態不同的日子（{GRID_FIRST[:7]} 起）：W1 {r['differ']['w1'] * 100:.1f}%、W2 {r['differ']['w2'] * 100:.1f}%。"
         f"安慰劑 {PLACEBO} 次每年多賺的中位數：W1 {_pp(r['placebo_med']['w1'])}、W2 {_pp(r['placebo_med']['w2'])}"
         f"（真的：W1 {_pp(s['w1']['c_edge'])}、W2 {_pp(s['w2']['c_edge'])}、50 日均線 {_pp(s['ma50']['c_edge'])}）。"),
        f"{GRID_FIRST[:4]} 年以後的大熊市避開的跌幅（算進中途被洗）：{bear_txt}。大熊市只有寥寥幾次，差幾個百分點不代表真的比較好。",
        (f"和 main_signal_check 的不同：比特幣起算月從 {GRID_FIRST[:7]} 起（瀏覽量 2015-07 才有、百分位要一年暖機），"
         f"不是 2015-01；第四項只用 ETH（只有它有對應的維基頁面），不含 SOL、BNB、XRP、MSTR；一般人做法成本用 {P_COST * 100:.2f}%。"
         "50 日均線用同樣的設定重算，所以數字和那份研究不同，但三條規則彼此可以比。"),
        (f"關注度變化「上升」＝7 天平均比一週前高。百分位視窗：爆量用過去 {SPIKE_WIN} 天（至少 {SPIKE_MIN} 天），一週變化用過去 "
         f"{FALL_WIN} 天（至少 {FALL_MIN} 天）。瀏覽量缺一天就沿用前一天。重抽區間用循環區塊（7／30 天報酬 60 天一段、90 天 120 天一段），"
         "天天重疊的報酬讓真正獨立的樣本少很多，區間只是大概。三個效果、兩條規則一起看，多重比較會讓「碰巧過關」的機會變大。"),
        ("瀏覽量會被很多和價格無關的事影響：維基百科改版、搜尋引擎直接顯示摘要、手機 App、AI 助手引用等，長期水準一路往下，"
         "所以只用一週變化和過去兩年內的相對位置，不比絕對數字。"),
        "過去的結果不保證未來會一樣。這裡只比較關注度和價格過去的關係、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "attention",
        "title": "大眾關注度（維基百科瀏覽量）和比特幣",
        "question": ("維基百科「比特幣」頁面的瀏覽量（大眾關注度）和之後比特幣的漲跌有沒有關係？學術論文說的「漲勢中關注上升會推更高、"
                     "跌勢中關注上升會跌更深」成不成立？關注度暴衝時避開、或漲勢失去關注時下車，能不能贏過 50 日均線？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Kristoufek (2013), BitCoin meets Google Trends and Wikipedia, Scientific Reports 3:3415",
             "url": "https://www.nature.com/articles/srep03415"},
            {"name": "同一篇論文（PubMed Central 全文）", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3849639"},
            {"name": "維基媒體每日瀏覽量 API（英文「Bitcoin」，其他頁面同一格式）",
             "url": "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Bitcoin/daily/20150701/"
                    + common.today_utc().replace("-", "")},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
