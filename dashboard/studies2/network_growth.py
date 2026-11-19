"""Historical study "network_growth": does growth in network participation (active addresses, transaction count) and
in the resources securing the chain (hash rate) predict bitcoin's next weeks, and does it add to the 50-day average?

What the source says (read before running):
- Guidolin & Ionta (2026), "Predictive sorting of cryptocurrencies based on fundamentals and sentiment", Journal of
  International Financial Markets, Institutions & Money, doi 10.1016/j.intfin.2026.102285. The full text (Bocconi IRIS
  PDF and record page) refused this program (HTTP 403), and the publisher page only redirects, so it was NOT read; no
  mirror was tried. Bocconi's news page says: weekly data, "August 2015 to April 2025" for bitcoin and ether, later
  extended to 40 coins; three signals: active users ("network participation"), the hash rate (it also helps
  non-mineable coins) and Google search sentiment; forecasts are made out of sample, in real time, against a
  benchmark based on historical average returns and beat it "in most cases"; coins are then sorted by past
  predictive performance (long the most predictable, short the least) with positive alphas. The news page gives no
  transformation, growth window or numbers. The sample end (April 2025) is later than the 2015-2022 in the request.

Pre-registered design (fixed before running; the exact transformation could not be read, so this is the usual one):
Predictors, weekly, point-in-time: on-chain day D is usable only from the UTC close that ends day D+1. A week is the
seven on-chain days Monday..Sunday (all seven needed); its value is the mean of the daily metric. Growth g = log(week
value / value k weeks earlier), k = 1 (primary; the paper is weekly), k = 4 and 12 as robustness only. Metrics
(Coin Metrics community API, key-less): BTC AdrActCnt, TxCnt, HashRate; ETH and XRP AdrActCnt, TxCnt (ETH hash rate
stops at the 2022-09 merge, XRP has none; SOL is refused by the free API, BNB stops in 2019). Each growth is
standardised against its own trailing 52 weekly values (incl. the current one, at least 52); the combined signal S =
the mean of the standardised growths. The week ending Sunday is decided at the close ending Monday and held until the
next Monday close. A raw regression forecast "> 0" would be long most weeks because the average weekly return is
positive, so the trading rule uses S.
Tests: (1) forward 1/4/12-week BTC returns from the decision close by S's trailing-52-week quintile (1 = lowest),
Q5 - Q1 with circular block bootstrap 90% intervals (blocks 8/13/26 weeks, 1,000 draws), periods 2015-2019,
2020-2022 (the requested paper window), 2023-now (after the requested window; the news page says the paper itself
runs to 2025-04, so 2025-05-now is also shown separately as the only truly new data); 4- and 12-week growth
versions; within ma50-on weeks S > 0 vs S <= 0. (2) Out-of-sample R2, expanding window: weekly log return over the
next 1/4/12 weeks on each raw 1-week growth (univariate OLS, pairs from 2011, only pairs whose target is already
known, at least 100), forecasts from 2015; combination = the mean of the three univariate forecasts; benchmark =
expanding mean of the same target (Campbell-Thompson); same three periods, block bootstrap intervals; ETH with its
own two metrics, forecasts from 2018. (3) Rules: N1 = hold BTC when S > 0; N2 = ma50 AND S > 0. Fair exam reusing
main_signal_check as in blockspace / ls_ratio: 108 BTC start months 2015-01..2023-12 (UTC 0h closes), 72 start
months x 24 closing hours textbook (0.10%) and ordinary person (10 h later, 0.25%, 10% missed entries, 200 sims, its
seed), other coins ETH and XRP with their OWN on-chain S (N2 uses the coin's own 50-day average; ma50 is scored on the
same two coins and the same start months, so its number differs from main_signal_check's five-coin one); ranking
score = lowest of the four "beats holding" shares; median deepest loss over the 1,728 textbook cells; placebo = the
weekly S series (BTC, ETH, XRP each) circularly shifted by >= 26 weeks (>= 180 days), 200 draws, ranking score with 20
ordinary-person sims per clock, real rule scored the same way.
Pass test (stated now): the predicted sign (positive) in all three periods incl. 2023-now for BOTH the 4-week Q5 - Q1
of S and the combination OOS R2 at 1 week, AND the rule's ranking score >= ma50's AND its median deepest loss no
worse AND placebo share <= 10%. Google sentiment is not re-tested: attention.py already tested public attention with
Wikipedia views (Google Trends has no official free API); the news page lists it as a separate signal, not a blend.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import attention as at
from dashboard.studies2 import blockspace as bs
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
PAPER_DOI = "https://doi.org/10.1016/j.intfin.2026.102285"
PAPER_PDF = ("https://iris.unibocconi.it/retrieve/ca0d5636-301d-4177-b28c-1e88a6b6cdb1/"
             "Guidolin%20and%20Ionta%20JIFMIM%202026.pdf")
PAPER_REC = "https://iris.unibocconi.it/handle/11565/4081736"
NEWS = "https://www.unibocconi.it/en/news/hidden-fundamentals-cryptocurrencies"
METRICS = {"btc": ("AdrActCnt", "TxCnt", "HashRate"), "eth": ("AdrActCnt", "TxCnt"), "xrp": ("AdrActCnt", "TxCnt")}
MNAME = {"AdrActCnt": "活躍地址數", "TxCnt": "交易筆數", "HashRate": "算力"}
OTHER = {"ETH": ("ETHUSDT", "eth"), "XRP": ("XRPUSDT", "xrp")}
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
ZWIN = 52
HW = (1, 4, 12)
BLOCKW = {1: 8, 4: 13, 12: 26}
TRAIN_FROM, OOS_FROM, ETH_OOS_FROM = "2011-01-01", "2015-01-01", "2018-01-01"
FULL_FROM = "2015-01-01"
PERIODS = (("2015 到 2019", "2015-01-01", "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"),
           ("2023 到今天", "2023-01-01", "2100-01-01"))
NEW_FROM = "2025-05-01"
PLACEBO, PLACEBO_GAP_W, PL_SIMS = 200, 26, 20
RULES = ("ma50", "n1", "n2")
TESTED = ("n1", "n2")
KI = {"ma50": 1, "n1": 81, "n2": 82}
NAMES = {"ma50": "50 日均線（基準）", "n1": "N1：網路成長綜合訊號 > 0 才持有", "n2": "N2：50 日均線＋綜合訊號 > 0 都說好才持有"}
SHORT = {"ma50": "50 日均線", "n1": "N1", "n2": "N2"}
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d
_ok, _share, _p, _pt, _r2, _ci, _of = bs._ok, bs._share, bs._p, bs._pt, bs._r2, bs._ci, bs._of


# ---------- data ----------

def _cm_fetch(asset: str):
    metrics = ",".join(METRICS[asset])

    def fetch() -> dict:
        reply = common.get_json(CM_URL, {"assets": asset, "metrics": metrics, "frequency": "1d",
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
            for m in METRICS[asset]:
                try:
                    row[m] = float(item[m]) if item.get(m) is not None else None
                except (TypeError, ValueError):
                    row[m] = None
            out[item["time"][:10]] = row
        if len(out) < 1500:
            raise RuntimeError(f"CoinMetrics {asset} 只回傳 {len(out)} 天")
        return out
    return fetch


def _onchain(asset: str) -> tuple[pd.DataFrame, list]:
    path_name = f"netgrowth_{asset}"
    got, problem = _stored(path_name, _cm_fetch(asset))
    df = pd.DataFrame.from_dict(got, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index < pd.Timestamp(common.today_utc())].asfreq("D")
    return df, [problem] if problem else []


def _stored(name: str, fetch):
    path = common.STORE / f"{name}.json"
    try:
        got = fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return got, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"{name} 取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        return got, f"{name} 這次取不到（{type(exc).__name__}），用之前存下來的資料。"


# ---------- the signal ----------

def _weekly(oc: pd.DataFrame, metrics) -> pd.DataFrame:
    d = oc[list(metrics)].where(oc[list(metrics)] > 0)
    w = d.resample("W-SUN").mean()
    cnt = d.resample("W-SUN").count()
    return w.where(cnt == 7)


def _signal(w: pd.DataFrame, k: int = 1) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Indexed by the Sunday that ends the on-chain week. Uses weeks up to that one only."""
    g = np.log(w / w.shift(k))
    mu = g.rolling(ZWIN, min_periods=ZWIN).mean()
    sd = g.rolling(ZWIN, min_periods=ZWIN).std(ddof=1)
    z = (g - mu) / sd.where(sd > 0)
    return z.mean(axis=1, skipna=False), z, g


def _daily_state(s: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Week ending Sunday W is decided at the close ending Monday W+1 and held until the next decision."""
    u = s.copy()
    u.index = u.index + DAY
    v = u.reindex(pd.DatetimeIndex(dates), method="ffill").values.astype(float)
    return v


def _on(v: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(v), np.nan, (v > 0).astype(float))


def _rules(ma: np.ndarray, sv: np.ndarray) -> dict[str, np.ndarray]:
    n1 = _on(sv)
    return {"ma50": ma, "n1": n1, "n2": np.where(np.isnan(ma) | np.isnan(n1), np.nan, np.nan_to_num(ma) * np.nan_to_num(n1))}


# ---------- weekly frame, quintiles ----------

def _wframe(close: pd.Series, s: pd.Series, ma: np.ndarray) -> pd.DataFrame:
    dec = s.index + DAY
    keep = dec.isin(close.index)
    df = pd.DataFrame({"s": s.values[keep]}, index=dec[keep])
    c0 = close.reindex(df.index).values
    for h in HW:
        df[f"f{h}"] = close.reindex(df.index + pd.Timedelta(weeks=h)).values / c0 - 1
    ma_s = pd.Series(ma, index=close.index)
    df["ma"] = ma_s.reindex(df.index).values
    sv = df["s"].values
    q = np.full(len(sv), np.nan)
    for i in range(ZWIN - 1, len(sv)):
        win = sv[i - ZWIN + 1:i + 1]
        if np.isfinite(win).all():
            q[i] = min(5, max(1, math.ceil((win <= sv[i]).sum() / ZWIN * 5)))
    df["q"] = q
    return df[np.isfinite(df["s"]) & np.isfinite(df["q"])]


def _qdiff_fn(d):
    return at._mmean(d["y"], d["hi"]) - at._mmean(d["y"], d["lo"])


def _quint(df: pd.DataFrame) -> dict:
    out = {}
    for qq in range(1, 6):
        m = (df["q"] == qq).astype(float).values
        out[qq] = {"n": int(m.sum()), **{h: at._boot({"y": df[f"f{h}"].values, "m": m}, at._cell_fn, BLOCKW[h]) for h in HW}}
    hi, lo = (df["q"] == 5).astype(float).values, (df["q"] == 1).astype(float).values
    out["diff"] = {h: at._boot({"y": df[f"f{h}"].values, "hi": hi, "lo": lo}, _qdiff_fn, BLOCKW[h]) for h in HW}
    pos, neg = (df["s"] > 0).astype(float).values, (df["s"] <= 0).astype(float).values
    out["sign"] = {h: at._boot({"y": df[f"f{h}"].values, "hi": pos, "lo": neg}, _qdiff_fn, BLOCKW[h]) for h in HW}
    mon = df["ma"].values >= 0.5
    out["ma_on"] = at._boot({"y": df["f4"].values, "hi": pos * mon, "lo": neg * mon}, _qdiff_fn, BLOCKW[4])
    out["n"] = len(df)
    return out


def _sub(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]


# ---------- OOS R2 ----------

def _oos_fits(close: pd.Series, g: pd.DataFrame, oos_from: str) -> dict:
    """Weekly decision closes; univariate expanding OLS per metric, combination = mean of forecasts."""
    dec = g.index + DAY
    keep = dec.isin(close.index)
    gg = g[keep].copy()
    gg.index = dec[keep]
    wc = close.reindex(gg.index)
    allok = np.isfinite(gg.values).all(axis=1)
    fits = {}
    for h in HW:
        fs = []
        for m in gg.columns:
            x = np.where(allok, gg[m].values, np.nan)
            fs.append(bs._ols(wc, x, h, TRAIN_FROM, oos_from))
        f = np.mean([x["f"] for x in fs], axis=0)
        fits[h] = {"each": {m: fs[i] for i, m in enumerate(gg.columns)},
                   "combo": {"f": f, "b": fs[0]["b"], "y": fs[0]["y"]}, "idx": gg.index}
    return fits


def _oosr2(fit: dict, idx, a: str, b: str, h: int) -> tuple:
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)) & np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    em, eb = (fit["y"] - fit["f"])[m] ** 2, (fit["y"] - fit["b"])[m] ** 2
    return at._boot({"a": em, "b": eb}, bs._r2_fn, BLOCKW[h])


# ---------- fair grid ----------

def _grid_daily(close: pd.Series, pos: dict, year: int, first, last, k: str) -> dict:
    change = close.pct_change().fillna(0.0).values
    known = np.ones(len(close), dtype=bool)
    for kk in RULES:  # same start months for every rule
        known &= ~np.isnan(pos[kk])
    if not known.any():
        raise RuntimeError("規則沒有值")
    begin = max(close.index[int(np.argmax(known))] + DAY, pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    net, _ = te._run(np.concatenate([[np.nan], pos[k][:-1]]), change, COST)
    r, dd = msc._suffix(msc._equity(net), starts, year)
    out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock(bs._Clock):
    def flags(self, daily_index: pd.DatetimeIndex, st: dict) -> dict:
        return {"n1": np.nan_to_num(bs._at(pd.Series(st["n1"], index=daily_index), self.m.dates, 1), nan=0.0)}

    def want(self, k: str, h: int, fl: dict) -> np.ndarray:
        if k == "ma50":
            return self.ma[h]
        if k == "n2":
            return self.ma[h] * fl["n1"]
        return fl["n1"]

    def full(self, k: str, fl: dict, sims: int = SIMS, seed_add: int = 0) -> dict:
        saved = dict(bs.KI)
        bs.KI.update(KI)
        try:
            return super().full(k, fl, sims, seed_add)
        finally:
            bs.KI.clear()
            bs.KI.update(saved)


def _evaluate(k, close, pos, clock, fl, others, sims, seed_add=0) -> dict:
    d = _grid_daily(close, pos, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], k)
    x = clock.full(k, fl, sims, seed_add)
    dr, dd = d[k]
    x["d_n"], x["d_beat"], x["d_shallow"] = len(dr), int((dr > d["hold"][0]).sum()), int((dd > d["hold"][1]).sum())
    x["d_edge"] = float(np.median(dr - d["hold"][0]))
    x["d_deep_med"] = float(np.median(dd))
    o_n = o_beat = 0
    x["o_each"] = {}
    for a, (c, opos) in others.items():
        try:
            g = _grid_daily(c, opos, YEAR, c.index[0], bs.others_last(), k)
        except Exception:  # noqa: BLE001
            continue
        n_a, b_a = len(g[k][0]), int((g[k][0] > g["hold"][0]).sum())
        x["o_each"][a] = (n_a, b_a)
        o_n, o_beat = o_n + n_a, o_beat + b_a
    x["o_n"], x["o_beat"] = o_n, o_beat
    bs._score(x)
    return x


def _shift(s: pd.Series, rng) -> pd.Series:
    v = s.dropna()
    n = len(v)
    return pd.Series(np.roll(v.values, int(rng.integers(PLACEBO_GAP_W, n - PLACEBO_GAP_W))), index=v.index)


# ---------- look-ahead ----------

def _lookahead(close: pd.Series, oc: pd.DataFrame) -> tuple[int, int]:
    base = _daily_state(_signal(_weekly(oc, METRICS["btc"]))[0], close.index)
    rng = np.random.default_rng(11)
    bad = cuts = 0
    for cut in (len(close) - 2500, len(close) - 900, len(close) - 60, len(close) - 3):
        t = close.index[cut]
        oc2 = oc.copy()
        later = oc2.index >= t  # on-chain day t is not known at the close ending day t
        oc2.loc[later] = oc2.loc[later].values * np.exp(rng.normal(0, 1.0, oc2.loc[later].shape))
        alt = _daily_state(_signal(_weekly(oc2, METRICS["btc"]))[0], close.index)
        cuts += 1
        if not np.array_equal(np.nan_to_num(base[:cut + 1], nan=-9), np.nan_to_num(alt[:cut + 1], nan=-9)):
            bad += 1
    return cuts, bad


# ---------- the computation ----------

def _compute() -> dict:
    problems: list = []
    oc, pr = _onchain("btc")
    problems += pr
    w = _weekly(oc, METRICS["btc"])
    s, z, g = _signal(w, 1)
    s4, s12 = _signal(w, 4)[0], _signal(w, 12)[0]
    close = te._daily_close()
    ma = bs._ma(close.values)
    sv = _daily_state(s, close.index)
    pos = _rules(ma, sv)

    # test 1: quintiles
    wf = {"1": _wframe(close, s, ma), "4": _wframe(close, s4, ma), "12": _wframe(close, s12, ma)}
    q = {}
    for key, df in wf.items():
        q[key] = {"全部": _quint(_sub(df, FULL_FROM, "2100-01-01"))}
        for name, a, b in PERIODS:
            q[key][name] = _quint(_sub(df, a, b))
        q[key]["new"] = _quint(_sub(df, NEW_FROM, "2100-01-01"))

    # test 2: OOS R2
    fits = _oos_fits(close, g, OOS_FROM)
    oos = {}
    for h in HW:
        idx = fits[h]["idx"]
        oos[h] = {}
        for lab, fit in [("combo", fits[h]["combo"])] + list(fits[h]["each"].items()):
            oos[h][lab] = {"全部": _oosr2(fit, idx, FULL_FROM, "2100-01-01", h), "new": _oosr2(fit, idx, NEW_FROM, "2100-01-01", h)}
            for name, a, b in PERIODS:
                oos[h][lab][name] = _oosr2(fit, idx, a, b, h)

    # other coins: own on-chain signal
    other_s, other_close, eth_extra = {}, {}, {}
    for key, (sym, asset) in OTHER.items():
        try:
            oca, pr = _onchain(asset)
            problems += pr
            wa = _weekly(oca, METRICS[asset])
            sa, _za, ga = _signal(wa, 1)
            c = msc.ta._crypto_close(sym)
            c = c[c.index < pd.Timestamp(common.today_utc())]
            other_s[key], other_close[key] = sa, c
            if key == "ETH":
                ethc = c
                ef = _wframe(ethc, sa, bs._ma(ethc.values))
                eth_extra["q"] = {"全部": _quint(ef)}
                for name, a, b in PERIODS:
                    eth_extra["q"][name] = _quint(_sub(ef, a, b))
                fe = _oos_fits(ethc, ga, ETH_OOS_FROM)
                eth_extra["r2"] = {"全部": _oosr2(fe[1]["combo"], fe[1]["idx"], ETH_OOS_FROM, "2100-01-01", 1)}
                for name, a, b in PERIODS:
                    eth_extra["r2"][name] = _oosr2(fe[1]["combo"], fe[1]["idx"], max(a, ETH_OOS_FROM), b, 1)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key} 的鏈上資料或價格這次取不到（{type(exc).__name__}），沒有算進去。")

    def other_pos(sd: dict) -> dict:
        return {a: (other_close[a], _rules(bs._ma(other_close[a].values), _daily_state(sd[a], other_close[a].index)))
                for a in other_close}

    # test 3: fair grid
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
    fl = clock.flags(close.index, pos)
    others = other_pos(other_s)
    sc = {k: _evaluate(k, close, pos, clock, fl, others, SIMS) for k in RULES}

    real20 = {k: _evaluate(k, close, pos, clock, fl, others, PL_SIMS)["low"] for k in TESTED}
    rng = np.random.default_rng(SEED + 52)
    plac = {k: [] for k in TESTED}
    for i in range(PLACEBO):
        ss = _shift(s, rng)
        pp = _rules(ma, _daily_state(ss, close.index))
        flp = clock.flags(close.index, pp)
        opp = other_pos({a: _shift(other_s[a], rng) for a in other_s})
        for k in TESTED:
            plac[k].append(_evaluate(k, close, pp, clock, flp, opp, PL_SIMS, 7919 * (i + 1))["low"])
    placebo = {k: float(np.mean(np.array(v) >= real20[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    first = int(close.index.searchsorted(pd.Timestamp(FULL_FROM)))
    held = {k: float(np.nanmean(np.nan_to_num(pos[k][first:]))) for k in RULES}
    switches = {k: float(np.abs(np.diff(np.nan_to_num(pos[k][first:]))).sum() / ((len(close) - first) / YEAR))
                for k in RULES}
    look = _lookahead(close, oc)
    return {"oc": oc, "w": w, "s": s, "z": z, "g": g, "close": close, "pos": pos, "q": q, "oos": oos,
            "eth": eth_extra, "sc": sc, "clock": clock, "real20": real20, "placebo": placebo,
            "placebo_med": placebo_med, "held": held, "switches": switches, "look": look, "problems": problems,
            "other_s": other_s, "others": {a: (len(o[0])) for a, o in others.items()}}


# ---------- the study ----------

def _sign(v) -> int:
    return 0 if not _ok(v) or v == 0 else (1 if v > 0 else -1)


def study() -> dict:
    r = _compute()
    close, sc, q, oos = r["close"], r["sc"], r["q"], r["oos"]
    pnames = [p[0] for p in PERIODS]

    # ----- pass test -----
    qd = [q["1"][p]["diff"][4][0] for p in pnames]
    rd = [oos[1]["combo"][p][0] for p in pnames]
    eff_ok = all(_sign(v) == 1 for v in qd) and all(_sign(v) == 1 for v in rd)
    m = sc["ma50"]
    rule_ok = {}
    for k in TESTED:
        x = sc[k]
        rule_ok[k] = {"effect": eff_ok, "score": x["low"] >= m["low"] - 1e-12,
                      "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10}
        rule_ok[k]["all"] = all(rule_ok[k].values())
    passed = [k for k in TESTED if rule_ok[k]["all"]]

    def mc(t) -> str:
        return f"{_p(t[0])} {_ci(t[1], t[2], _p)}"

    def ic(t) -> str:
        return f"{_pt(t[0])} {_ci(t[1], t[2], _pt)}"

    # ----- table 1: quintiles -----
    rows1 = []
    g1 = q["1"]["全部"]
    for qq in range(1, 6):
        x = g1[qq]
        rows1.append([f"第 {qq} 組" + ("（最弱）" if qq == 1 else "（最強）" if qq == 5 else ""), f"{x['n']:,}",
                      mc(x[1]), mc(x[4]), mc(x[12]), "｜".join(_p(q["1"][p][qq][4][0]) for p in pnames)])
    rows1.append(["差：第 5 組減第 1 組（百分點）", "—", ic(g1["diff"][1]), ic(g1["diff"][4]), ic(g1["diff"][12]),
                  "｜".join(_pt(q["1"][p]["diff"][4][0]) for p in pnames)])
    rows1.append(["差：訊號 > 0 減 ≤ 0（N1 持有 vs 現金）", "—", ic(g1["sign"][1]), ic(g1["sign"][4]), ic(g1["sign"][12]),
                  "｜".join(_pt(q["1"][p]["sign"][4][0]) for p in pnames)])
    rows1.append(["只看 50 日均線站上的週：訊號 > 0 減 ≤ 0", "—", "—", ic(g1["ma_on"]), "—",
                  "｜".join(_pt(q["1"][p]["ma_on"][0]) for p in pnames)])
    for key, lab in (("4", "4 週成長版"), ("12", "12 週成長版")):
        x = q[key]["全部"]
        rows1.append([f"對照：{lab}，第 5 組減第 1 組", "—", ic(x["diff"][1]), ic(x["diff"][4]), ic(x["diff"][12]),
                      "｜".join(_pt(q[key][p]["diff"][4][0]) for p in pnames)])
    nw = q["1"]["new"]
    rows1.append([f"論文樣本之後（{NEW_FROM[:7]} 起，{nw['n']} 週）：第 5 組減第 1 組", "—", ic(nw["diff"][1]),
                  ic(nw["diff"][4]), ic(nw["diff"][12]), "—"])
    eq = r["eth"].get("q")
    if eq:
        rows1.append(["以太幣（自己的活躍地址＋交易筆數）：第 5 組減第 1 組", f"{eq['全部']['n']:,}",
                      ic(eq["全部"]["diff"][1]), ic(eq["全部"]["diff"][4]), ic(eq["全部"]["diff"][12]),
                      "｜".join(_pt(eq[p]["diff"][4][0]) for p in pnames)])

    # ----- table 2: OOS R2 -----
    rows2 = []
    labs = [("combo", "綜合（三個預測的平均）")] + [(mm, MNAME[mm]) for mm in METRICS["btc"]]
    for h in HW:
        for lab, name in labs:
            x = oos[h][lab]
            rows2.append([f"之後 {h} 週｜{name}", f"{_r2(x['全部'][0])} {_ci(x['全部'][1], x['全部'][2], _r2)}",
                          "｜".join(_r2(x[p][0]) for p in pnames), _r2(x["new"][0]), f"{x['全部'][3]:,}"])
    er = r["eth"].get("r2")
    if er:
        rows2.append(["以太幣之後 1 週｜綜合（2018 起預測）", f"{_r2(er['全部'][0])} {_ci(er['全部'][1], er['全部'][2], _r2)}",
                      "｜".join(_r2(er[p][0]) for p in pnames), "—", f"{er['全部'][3]:,}"])

    # ----- table 3: fair grid -----
    order = sorted(RULES, key=lambda k: -sc[k]["low"])
    rows3 = []
    for k in RULES:
        x = sc[k]
        rows3.append([NAMES[k], _of(x["d_n"], x["d_beat"]), f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}",
                      "、".join(f"{a} {_of(*v)}" for a, v in x["o_each"].items()) or "—",
                      f"{_share(x['low'])}（第 {order.index(k) + 1}）｜{_pct(x['c_deep_med'])}",
                      f"{_share(r['held'][k])}｜{r['switches'][k]:.0f} 次",
                      "—（基準）" if k == "ma50" else _share(r["placebo"][k])])
    rows3.append(["對照：一直抱著不動", "—", "—", "—", "—", f"—｜{_pct(float(np.median(r['clock'].hd)))}", "100%｜0 次", "—"])

    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows4 = []
    for k in TESTED:
        rows4.append([NAMES[k],
                      f"第 5 減第 1 組（4 週）{'／'.join(_pt(v) for v in qd)}；綜合樣本外 R²（1 週）{'／'.join(_r2(v) for v in rd)}"
                      f"（{yes(eff_ok)}）",
                      f"{_share(sc[k]['low'])} vs {_share(m['low'])}（{yes(rule_ok[k]['score'])}）",
                      f"{_pct(sc[k]['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(rule_ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（{yes(rule_ok[k]['placebo'])}）",
                      "通過" if rule_ok[k]["all"] else "沒有通過"])

    n_d, n_c = m["d_n"], m["c_n"]
    tables = [
        {"caption": ("一、每週的網路成長綜合訊號（活躍地址數、交易筆數、算力各自「這週平均比上週」的對數成長，和自己過去 52 週比標準化後平均）"
                     "在週一收盤能拿到的值，依它在過去 52 週裡的位置分成五組，看之後 1／4／12 週比特幣平均漲跌；方括號是 90% 區間"
                     "（區塊重抽 1,000 次）。全部＝2015 起。報酬互相重疊，真正獨立的樣本比週數少。"),
         "columns": ["分組", "週數", "之後 1 週", "之後 4 週", "之後 12 週", "之後 4 週：" + "｜".join(pnames)],
         "rows": rows1},
        {"caption": ("二、樣本外 R²（論文的比法）：每週只用當時已經知道結果的資料重新估一次「之後報酬＝a＋b×上週成長」，2015 起預測，"
                     "和「過去平均報酬」比；大於 0 才代表猜得比平均準。綜合＝三個單一預測取平均。"),
         "columns": ["預測什麼｜用什麼", "2015 到今天［90% 區間］", "｜".join(pnames), f"{NEW_FROM[:7]} 起（論文樣本之後）", "預測週數"],
         "rows": rows2},
        {"caption": (f"三、規則和 50 日均線同樣條件考（main_signal_check 的公平考法）：比特幣 {n_d} 個起算月、{n_c:,} 格起算月×收盤時間"
                     f"（照課本做，成本 {COST * 100:.2f}%）、同樣格數照一般人方式做（10 小時後才做、成本 {P_COST * 100:.2f}%、一成進場漏掉、"
                     f"每個收盤時間 {SIMS} 次），以及以太幣和 XRP 的起算月（用它們自己的鏈上訊號；Coin Metrics 免費版只有這兩個幣有完整資料）。"
                     "排名分數＝四項「贏過一直抱著」比例裡最低的；最深虧損＝1,728 格的中位數；"
                     f"安慰劑＝把每週訊號整串循環平移（至少 26 週）{PLACEBO} 次，排名分數一樣好或更好的比例。"),
         "columns": ["規則", "比特幣起算月：贏的", f"{n_c:,} 格照課本：贏的｜多賺", "一般人做法：贏的｜多賺", "其他幣起算月：贏的",
                     "排名分數（名次）｜最深虧損中位數", "2015 起持有比例｜每年進出", "安慰劑：打亂後一樣好的比例"],
         "rows": rows3},
        {"caption": ("四、事先寫下的通過條件逐條對：第 5 組減第 1 組（之後 4 週）和綜合樣本外 R²（之後 1 週）在三段期間都為正；"
                     "排名分數不低於 50 日均線；最深虧損中位數不比 50 日均線深；安慰劑比例 ≤ 10%。四條都過才算通過。"),
         "columns": ["規則", "效果（三段）", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "安慰劑比例", "結論"],
         "rows": rows4},
    ]

    # ----- today -----
    t = close.index[-1]
    s, z, g, pos = r["s"], r["z"], r["g"], r["pos"]
    usable = s.index[s.index + DAY <= t]
    wk = usable[-1] if len(usable) else None

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    now = [{"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
           {"label": "Coin Metrics 最新一天的鏈上資料", "value": _d(r["oc"].dropna(how="all").index[-1])}]
    if wk is not None:
        now.append({"label": f"規則用的那一週（{_d(wk - 6 * DAY)} 到 {_d(wk)}）綜合訊號",
                    "value": f"{s[wk]:+.3f}" if _ok(s[wk]) else "—"})
        for mm in METRICS["btc"]:
            gv, zv = g.loc[wk, mm], z.loc[wk, mm]
            now.append({"label": f"{MNAME[mm]}：比前一週", "value": f"{_p(math.expm1(gv))}（標準化 {zv:+.2f}）"
                        if _ok(gv) and _ok(zv) else "—"})
    for a, sa in r["other_s"].items():
        ua = sa.dropna()
        ua = ua[ua.index + DAY <= t]
        if len(ua):
            now.append({"label": f"{a} 自己的綜合訊號（{_d(ua.index[-1])} 那週）", "value": f"{ua.iloc[-1]:+.3f}"})
    for k in RULES:
        now.append({"label": NAMES[k], "value": word(pos[k][-1])})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    d4 = g1["diff"][4]
    r1 = oos[1]["combo"]["全部"]
    verdict = ((f"{'有規則通過事先定的條件：' + '、'.join(SHORT[k] for k in passed) if passed else '事先定的通過條件沒有一條規則過'}。"
                f"2015 年起，網路成長綜合訊號落在過去一年最強五分之一的週，之後 4 週比特幣平均比最弱五分之一多 {_pt(d4[0])} 個百分點"
                f"（90% 區間 {_pt(d4[1])} 到 {_pt(d4[2])}；三段 {'／'.join(_pt(v) for v in qd)}）；"
                f"用成長預測下一週的樣本外 R² 是 {_r2(r1[0])}（三段 {'／'.join(_r2(v) for v in rd)}）。")
               + f"排名分數 N1 {_share(sc['n1']['low'])}、N2 {_share(sc['n2']['low'])}，50 日均線 {_share(m['low'])}"
               f"（安慰劑比例 N1 {_share(r['placebo']['n1'])}、N2 {_share(r['placebo']['n2'])}）；"
               f"N2 最深虧損中位數 {_pct(sc['n2']['c_deep_med'])}，50 日均線 {_pct(m['c_deep_med'])}，"
               f"但它在比特幣 {_of(sc['n2']['d_n'], sc['n2']['d_beat'])}起算月贏過一直抱著（50 日均線 {_of(m['d_n'], m['d_beat'])}）。")

    cuts, bad = r["look"]
    notes = [
        ("事先寫下的設計（照原文）：weekly growth g = log(week mean / mean k weeks earlier), k = 1 primary (4, 12 robustness), "
         "metrics BTC AdrActCnt, TxCnt, HashRate (ETH, XRP: AdrActCnt, TxCnt); each standardised against its trailing 52 "
         "weeks; combined S = mean; on-chain day D usable from the close ending D+1, week Mon..Sun decided at Monday's close. "
         "Tests: forward 1/4/12-week returns by S's trailing-year quintile with block bootstrap; periods 2015-19, 2020-22, "
         "2023-now; expanding-window OOS R² (univariate OLS per growth, combination = mean forecast, benchmark = expanding "
         "mean). N1 = hold when S > 0; N2 = ma50 AND S > 0; main_signal_check fair exam, other coins ETH and XRP with their "
         "own S, placebo circular shift ≥ 26 weeks × 200. Pass: Q5−Q1 (4 weeks) and combination OOS R² (1 week) positive in "
         "all three periods AND ranking ≥ ma50 AND median deepest loss no worse AND placebo ≤ 10%."),
        ("論文本身：Guidolin 與 Ionta（2026）的全文（Bocconi 機構典藏的 PDF 與紀錄頁）拒絕這個程式讀取（HTTP 403），出版社頁面只有轉址，"
         "所以沒有讀到全文、也沒有去找其他鏡像。博科尼大學的新聞稿只說：每週資料、比特幣和以太幣 2015-08 到 2025-04、之後擴大到 40 個幣；"
         "三個訊號是活躍使用者（網路參與）、算力（連不能挖礦的幣也有用）和 Google 搜尋情緒；預測是即時、樣本外，和「歷史平均報酬」比，"
         "「多數情況」比較準；再依過去預測得準不準把幣排序，買最準的、賣最不準的，有正的超額報酬。新聞稿沒有寫成長怎麼算、看幾週、數字多少，"
         "所以這裡的「一週成長、和過去 52 週比」是常見做法，不一定和論文完全一樣。"),
        ("論文的排序是「哪個幣比較好預測」的跨幣排序，不是單一幣的進出場規則；這裡把它翻成比特幣（和以太幣、XRP）自己的擇時，"
         "回答的是「網路成長能不能幫忙決定什麼時候拿著比特幣」。Google 搜尋情緒沒有重測：attention 研究已經用維基百科瀏覽量測過大眾注意力"
         "（Google 趨勢沒有官方免費 API），新聞稿也是把它當成另一個單獨的訊號。算力的其他用法（礦工投降）在 hash_ribbons。"),
        (f"期間：委託寫的論文期間是 2015–2022，但新聞稿說比特幣和以太幣做到 2025-04，所以「2023 到今天」有一部分在論文樣本內；"
         f"真正論文沒看過的只有 {NEW_FROM[:7]} 以後（表一、表二另外列出，週數少，只能參考）。"),
        (f"通過條件：{'、'.join(SHORT[k] for k in passed) + ' 通過' if passed else '沒有一條規則通過'}。"
         f"安慰劑用每個收盤時間 {PL_SIMS} 次一般人模擬重算排名分數（真的規則同樣算：N1 {_share(r['real20']['n1'])}、"
         f"N2 {_share(r['real20']['n2'])}），打亂後的中位數 N1 {_share(r['placebo_med']['n1'])}、N2 {_share(r['placebo_med']['n2'])}。"
         "真的規則排名分數是 0% 時，打亂後一定「一樣好或更好」，安慰劑比例就是 100%，意思是看不出比隨機好。"),
        ("不偷看未來：鏈上某一天的資料要等那天（UTC）結束後才有；一週（週一到週日）的值在下週一收盤才使用，持有到再下一個週一收盤。"
         "換收盤時間時，落在 UTC 日期 E 的收盤只用「E−1 那天 UTC 0 點收盤時的狀態」。"
         f"檢查：把 {cuts} 個切點當天和之後的鏈上資料隨機打亂再算一次，切點以前的狀態"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 處變了，這次結果不可信。")
         + " 但 Coin Metrics 的歷史值會被修正（活躍地址、算力都是估計），這裡用的是今天看到的版本。"),
        ("資料：Coin Metrics 社群版（免金鑰）的 AdrActCnt（當天有收或付的地址數）、TxCnt（交易筆數）、HashRate（從出塊時間估的算力）。"
         "以太幣算力只到 2022-09 轉成權益證明為止，所以以太幣只用前兩個；SOL 免費版拒絕存取（403），BNB 只到 2019，都沒用。"
         "其他幣用今天還在的大幣，有倖存者偏誤；50 日均線在這裡也只在比特幣、以太幣、XRP 上算，所以它的排名分數和 main_signal_check 的不一樣。"),
        ("地址數和交易筆數會被和價格無關的事改變：批次付款、隔離見證、2023 年起的 Ordinals 銘文與符文、交易所整合地址、XRP 的垃圾交易；"
         "算力會隨礦機世代與電價跳動。每週成長很吵，標準化後的訊號大約一半的週為正，所以 N1 進出很頻繁，成本吃得多。"),
        ("相鄰起算日共用大部分的歷史，24 個收盤時間只差幾個小時，這些「幾個裡幾個」不是互相獨立的證據。報酬重疊、跨過期間邊界，區間只是大概。"),
        "過去的結果不保證未來會一樣。這裡只比較網路成長和價格過去的關係、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "network_growth",
        "title": "網路成長（活躍地址、交易筆數、算力）和比特幣",
        "question": ("學術論文（Guidolin 與 Ionta 2026）說活躍使用者、算力這類區塊鏈基本面的成長能在樣本外預測加密貨幣報酬。"
                     "翻成比特幣的擇時：每週網路成長比較強的時候，之後幾週比特幣是不是比較好？加在 50 日均線上有沒有幫助？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Guidolin & Ionta (2026), Predictive sorting of cryptocurrencies based on fundamentals and sentiment, "
                     "JIFMIM（DOI；全文無法讀取）", "url": PAPER_DOI},
            {"name": "同上，Bocconi 機構典藏全文（本程式讀取被拒 403）", "url": PAPER_PDF},
            {"name": "同上，Bocconi 機構典藏紀錄頁（讀取被拒 403）", "url": PAPER_REC},
            {"name": "Bocconi 新聞稿：Hidden fundamentals of cryptocurrencies", "url": NEWS},
            {"name": "Coin Metrics 社群版 API（比特幣 AdrActCnt、TxCnt、HashRate）",
             "url": CM_URL + "?assets=btc&metrics=AdrActCnt,TxCnt,HashRate&frequency=1d"},
            {"name": "Coin Metrics 社群版 API（以太幣 AdrActCnt、TxCnt）", "url": CM_URL + "?assets=eth&metrics=AdrActCnt,TxCnt&frequency=1d"},
            {"name": "Coin Metrics 社群版 API（XRP AdrActCnt、TxCnt）", "url": CM_URL + "?assets=xrp&metrics=AdrActCnt,TxCnt&frequency=1d"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
