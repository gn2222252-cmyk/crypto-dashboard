"""Historical study "trend_portfolio": the verified trend rules run on many markets at once, risk spread evenly, and
how much bitcoin's (and ether's) trend stream adds to such a portfolio.

Markets: S&P 500, Nasdaq, Taiwan weighted, Nikkei 225, Hang Seng, EEM, gold (GLD, joined to gold futures before
2004-11), TLT, crude oil futures, US dollar index (Yahoo, read from trend_world's store), bitcoin (2014 on) and ether
(2018 on).

Calendar: US trading days (the S&P 500's). Every market's close is read on each US trading day as its latest close
on or before that date (a market closed that day keeps yesterday's close). Bitcoin and ether use the candle labelled
with each US trading day (as in rotation.py); their rules are computed on their own every-day calendar (365 days a
year, exactly as on the signals tab), other markets' rules on their own trading days (as in trend_world).

Portfolios: P1 equal-weight buy-and-hold of the ten traditional markets; P2 60/40 S&P 500 / TLT; P3 each traditional
market under the 17-rule blend (or 55/20), weights 1/(60-day volatility) normalised to 100%, no leverage; P4 = P3 plus
bitcoin and ether; P5 = P4 scaled to a 10% portfolio volatility (up to 1.5x, borrowing at T-bill + 1%) and the same
capped at 1x; P6 = 60/40 with a 5% bitcoin sleeve held or under 55/20. Weights are reset at every month's last US
trading day; between resets a market is traded only on the days its rule's target changes (to weight x target).

No look-ahead: targets and volatilities use closes up to and including day t and set the exposure held during day t+1;
cost 0.10% on every amount traded (charged on the day of the trade). Idle cash earns the 13-week T-bill yield (^IRX,
the previous day's quote x calendar days / 360); a second version earns 0%. Only public, key-less sources, fixed user
agent. Own store: cache/_store/trend_portfolio_closes.json (crypto closes and any Yahoo series refreshed here), used
only when a fetch fails.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import signals as sig
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import trend_world as tw

YEAR = 252
COST = 0.001
HIGH_COST = 0.003
WEEK = 5  # one week's delay = five US trading days
SPREAD = 0.01  # borrowed money costs the T-bill rate + 1% a year
VOL_TARGET = 0.10
LEV_CAP = 1.5
VOL_DAYS = 60
START = "2002-01-01"  # the simulation starts here; results are reported from 2004
OWN_STORE = common.STORE / "trend_portfolio_closes.json"
STALE_DAYS = 7
PERIODS = (("p04", "2004 起", "2004-01-01"), ("p14", "2014 起", "2014-01-01"), ("p18", "2018 起", "2018-01-01"),
           ("p22", "2022 起", "2022-01-01"))
WF_FROM = 2016
TRAD = ("^GSPC", "^IXIC", "^TWII", "^N225", "^HSI", "EEM", "GOLD", "TLT", "CL=F", "DX-Y.NYB")
CRYPTO = ("BTC", "ETH")
KEYS = TRAD + CRYPTO
CRYPTO_FROM = {"BTC": "2014-01-01", "ETH": "2018-01-01"}
NAME = {"^GSPC": "標普 500", "^IXIC": "那斯達克", "^TWII": "台灣加權", "^N225": "日經 225", "^HSI": "恆生",
        "EEM": "新興市場（EEM）", "GOLD": "黃金", "TLT": "美國長債（TLT）", "CL=F": "原油期貨", "DX-Y.NYB": "美元指數",
        "BTC": "比特幣", "ETH": "以太幣"}
YAHOO = ("^GSPC", "^IXIC", "^TWII", "^N225", "^HSI", "EEM", "GC=F", "GLD", "TLT", "CL=F", "DX-Y.NYB", "^IRX",
         "^SP500TR")
RULE_NAME = {"blend": "17 條混合", "b5520": "55／20"}

_pct, _num, _price, _d = ta._pct, ta._num, ta._price, ta._d


# ---------- data ----------

def _read(path) -> dict:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        return stored if isinstance(stored, dict) else {}
    except Exception:  # noqa: BLE001 - first run or damaged file
        return {}


def _series(item: dict) -> pd.Series:
    return pd.Series(item["close"], index=pd.to_datetime(item["dates"]), dtype=float)


def _pack(series: pd.Series) -> dict:
    return {"dates": [d.strftime("%Y-%m-%d") for d in series.index], "close": [float(f"{v:.8g}") for v in series.values]}


def _load() -> tuple[dict, list[str], dict]:
    """Closes on each market's own calendar. Yahoo series come from trend_world's store (refreshed with
    trend_world's own fetch when older than a week); bitcoin and ether are fetched (own store as fallback)."""
    world, own = _read(tw.STORE_PATH), _read(OWN_STORE)
    keep, data, problems, origin = dict(own), {}, [], {}
    today = pd.Timestamp(common.today_utc())
    for sym in YAHOO:
        series = None
        item = world.get(sym)
        if isinstance(item, dict) and item.get("dates"):
            series, origin[sym] = _series(item), "trend_world"
        if series is None or (today - series.index[-1]).days > STALE_DAYS:
            try:
                series = tw._fetch(sym)
                keep[sym], origin[sym] = _pack(series), "fetch"
            except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the others
                old = own.get(sym)
                if isinstance(old, dict) and old.get("dates") and (series is None or old["dates"][-1] > _d(series.index[-1])):
                    series, origin[sym] = _series(old), "own"
                if series is not None:
                    problems.append(f"{NAME.get(sym, sym)}（{sym}）這次沒有更新（{type(exc).__name__}），用存下的資料（到 {_d(series.index[-1])}）。")
        if series is None:
            problems.append(f"{NAME.get(sym, sym)}（{sym}）取不到資料，沒有算進去。")
            continue
        series = series[np.isfinite(series) & (series > 0)]
        series = series[~series.index.duplicated(keep="last")].sort_index()
        cut = tw._broken(series) if sym != "^IRX" else None
        if cut is not None:
            series = series[series.index >= cut]
            problems.append(f"{sym} 的資料在 {_d(cut)} 有一天的漲跌不可能是真的（多半是拆股沒調整），只用那天以後的資料。")
        data[sym] = series
    for key, loader in (("BTC", te._daily_close), ("ETH", lambda: ta._crypto_close("ETHUSDT"))):
        try:
            series = loader()
            keep[key], origin[key] = _pack(series), "fetch"
        except Exception as exc:  # noqa: BLE001
            old = own.get(key)
            if not (isinstance(old, dict) and old.get("dates")):
                problems.append(f"{NAME[key]}取不到資料（{type(exc).__name__}），沒有算進去。")
                continue
            series, origin[key] = _series(old).asfreq("D").ffill(), "own"
            problems.append(f"{NAME[key]}這次取不到（{type(exc).__name__}），用上次存下的資料（到 {_d(series.index[-1])}）。")
        data[key] = series
    try:
        OWN_STORE.parent.mkdir(parents=True, exist_ok=True)
        OWN_STORE.write_text(json.dumps(keep, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store is only a fallback
        pass
    if "GLD" in data:  # GLD from 2004-11; gold futures before, scaled to join on GLD's first day
        gold = data["GLD"]
        if "GC=F" in data:
            fut = data["GC=F"]
            base = fut[fut.index <= gold.index[0]]
            if len(base):
                gold = pd.concat([fut[fut.index < gold.index[0]] * (gold.iloc[0] / base.iloc[-1]), gold])
        data["GOLD"] = gold
    elif "GC=F" in data:
        data["GOLD"] = data["GC=F"]
    return data, problems, origin


# ---------- building the arrays ----------

def _prepare(data: dict) -> dict:
    if "^GSPC" not in data:
        raise RuntimeError("取不到標普 500，沒有共同的交易日曆")
    master = data["^GSPC"].index
    master = master[master >= pd.Timestamp(START)]
    keys = [k for k in KEYS if k in data]
    cols = keys + (["SPTR"] if "^SP500TR" in data else [])
    price = pd.DataFrame(index=master, columns=cols, dtype=float)
    blend = pd.DataFrame(index=master, columns=keys, dtype=float)
    b5520 = pd.DataFrame(index=master, columns=keys, dtype=float)

    def onto(series: pd.Series) -> pd.Series:
        u = master.union(series.index)
        return series.reindex(u).ffill().reindex(master)

    for k in keys:
        s = data[k]
        targets, _ = sig._targets(s, 365 if k in CRYPTO else YEAR)
        if k in CRYPTO:  # the candle labelled with each US trading day (rotation.py)
            price[k] = s.reindex(master)
            blend[k], b5520[k] = targets["blend"].reindex(master), targets["b5520"].reindex(master)
        else:
            price[k] = onto(s)
            blend[k], b5520[k] = onto(targets["blend"]), onto(targets["b5520"])
    if "SPTR" in cols:
        price["SPTR"] = onto(data["^SP500TR"])
    change = price / price.shift(1) - 1
    gap = pd.Series(master, index=master).diff().dt.days.fillna(0.0)
    if "^IRX" in data:
        rate = onto(data["^IRX"])
        rf = (rate.shift(1) / 100 * gap / 360).fillna(0.0)
    else:
        rf = pd.Series(0.0, index=master)
    month = master.to_period("M")
    me = np.where(month[:-1] != month[1:])[0]  # last US trading day of each finished month
    allowed = np.ones((len(master), len(cols)), dtype=bool)
    for j, k in enumerate(cols):
        if k in CRYPTO_FROM:
            allowed[:, j] = master >= pd.Timestamp(CRYPTO_FROM[k])
    vol = {w: (change.rolling(w, min_periods=w).std(ddof=1) * math.sqrt(YEAR)).values for w in (60, 120)}
    return {"index": master, "cols": cols, "price": price, "R": change.fillna(0.0).values, "has": price.notna().values,
            "blend": blend.reindex(columns=cols).values, "b5520": b5520.reindex(columns=cols).values,
            "vol": vol, "rf": rf.values, "borrow": (rf + SPREAD * gap / 360).values, "me": me, "allowed": allowed}


def _mask(p: dict, names) -> np.ndarray:
    return np.array([c in names for c in p["cols"]])


def _weights(p: dict, mode: str, include: np.ndarray, raw: np.ndarray | None = None, window: int = VOL_DAYS,
             fixed: dict | None = None) -> np.ndarray:
    """Weights set at each month end (rows follow p['me'])."""
    out = np.zeros((len(p["me"]), len(p["cols"])))
    V = p["vol"][window]
    fx = np.array([(fixed or {}).get(c, 0.0) for c in p["cols"]])
    for j, m in enumerate(p["me"]):
        ok = include & p["has"][m] & p["allowed"][m]
        if mode == "inv":
            ok = ok & np.isfinite(V[m]) & (np.nan_to_num(V[m]) > 1e-9) & np.isfinite(raw[m])
            w = np.where(ok, 1 / np.where(ok, V[m], 1.0), 0.0)
        elif mode == "eq":
            w = ok.astype(float)
        else:
            out[j] = fx * ok
            continue
        out[j] = w / w.sum() if w.sum() > 0 else w
    return out


def _sim(p: dict, S: np.ndarray, W: np.ndarray, cost: float = COST, cash_rate: bool = True, delay: int = 0,
         k: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Daily net returns and end-of-day exposures. S: targets decided at each close (0..1); W: weights per month end."""
    R, rf, borrow = p["R"], (p["rf"] if cash_rate else np.zeros(len(p["rf"]))), p["borrow"]
    T, N = R.shape
    reset = {}
    for j, m in enumerate(p["me"]):
        if m + delay < T:
            reset[m + delay] = (W[j], 1.0 if k is None else float(k[j]))
    Sd = S if delay == 0 else np.vstack([np.zeros((delay, N)), S[:-delay]])
    x, w, kk, prev = np.zeros(N), np.zeros(N), 0.0, np.full(N, -1.0)
    out, expo = np.zeros(T), np.zeros((T, N))
    for t in range(T):
        if t > 0:
            g = float(x @ R[t])
            cash = 1.0 - x.sum()
            g += cash * (rf[t] if cash >= 0 else borrow[t])
            x = x * (1 + R[t]) / (1 + g) if 1 + g > 1e-9 else x * 0
            out[t] = g
        s = Sd[t]
        hit = reset.get(t)
        if hit is not None:
            w, kk = hit
            target = kk * w * s
            trade = float(np.abs(target - x).sum())
            x = target
        else:
            moved = s != prev
            if moved.any():
                target = kk * w * s
                trade = float(np.abs(target - x)[moved].sum())
                x = np.where(moved, target, x)
            else:
                trade = 0.0
        prev = s
        out[t] -= cost * trade
        expo[t] = x
    return out, expo


# ---------- measuring ----------

def _yearly(r: pd.Series) -> dict:
    return {int(y): float(np.prod(1 + g.values) - 1) for y, g in r.groupby(r.index.year)}


def _stats(r: pd.Series, start: str, bench: pd.Series | None = None) -> dict | None:
    x = r[r.index >= pd.Timestamp(start)]
    if len(x) < YEAR // 2:
        return None
    eq = np.cumprod(1 + x.values)
    growth = float(eq[-1])
    ret = (growth ** (YEAR / len(x)) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(x.values, ddof=1) * math.sqrt(YEAR) * 100)
    full = np.concatenate([[1.0], eq])
    deep = float((full / np.maximum.accumulate(full) - 1).min() * 100)
    longest, peak, peak_day = 0, 1.0, x.index[0]
    for day, v in zip(x.index, eq):
        if v >= peak:
            peak, peak_day = v, day
        else:
            longest = max(longest, (day - peak_day).days)
    last = x.index[-1]
    years = {y: v for y, v in _yearly(x).items() if y < last.year or (last.month == 12 and last.day >= 30)}
    worst = min(years.items(), key=lambda kv: kv[1]) if years else None
    out = {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else 0.0, "deep": deep, "under": longest / 365.25,
           "worst": worst, "years": years, "n": len(years)}
    if bench is not None:
        b = _yearly(bench[bench.index >= pd.Timestamp(start)])
        out["beat"] = int(sum(v > b.get(y, math.inf) for y, v in years.items()))
    return out


def _monthly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def _avg_corr(frame: pd.DataFrame) -> float | None:
    c = frame.corr(min_periods=24).values
    vals = c[~np.eye(len(c), dtype=bool)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else None


# ---------- everything that depends on the data ----------

def _compute(data: dict) -> dict:
    p = _prepare(data)
    idx, cols = p["index"], p["cols"]
    trad = _mask(p, TRAD)
    alls = _mask(p, KEYS)
    hold = p["has"].astype(float)
    S = {rule: np.nan_to_num(p[rule], nan=0.0) for rule in ("blend", "b5520")}
    runs, expo = {}, {}

    def run(name: str, Sx, W, **kw):
        net, x = _sim(p, Sx, W, **kw)
        runs[name] = pd.Series(net, index=idx)
        expo[name] = x
        return runs[name]

    W = {}
    W["P1"] = _weights(p, "eq", trad)
    W["P2"] = _weights(p, "fixed", alls, fixed={"^GSPC": 0.6, "TLT": 0.4})
    W["P6"] = _weights(p, "fixed", alls, fixed={"^GSPC": 0.57, "TLT": 0.38, "BTC": 0.05})
    for rule in ("blend", "b5520"):
        for w in (60, 120):
            W[f"P3_{rule}_{w}"] = _weights(p, "inv", trad, p[rule], w)
            W[f"P4_{rule}_{w}"] = _weights(p, "inv", alls, p[rule], w)
    S6 = hold.copy()
    if "BTC" in cols:
        S6[:, cols.index("BTC")] = S["b5520"][:, cols.index("BTC")]
    main = {"P1": (hold, W["P1"]), "P2": (hold, W["P2"]),
            "P3b": (S["blend"], W["P3_blend_60"]), "P3c": (S["b5520"], W["P3_b5520_60"]),
            "P4b": (S["blend"], W["P4_blend_60"]), "P4c": (S["b5520"], W["P4_b5520_60"]),
            "P6h": (hold, W["P6"]), "P6r": (S6, W["P6"])}
    for name, (Sx, Wx) in main.items():
        run(name, Sx, Wx)
        run(name + "_0", Sx, Wx, cash_rate=False)
    # P5: P4 (blend) scaled at each month end to 10% volatility, from P4's own last 60 days
    r4 = runs["P4b"].values
    ks = []
    for m in p["me"]:
        seg = r4[max(0, m - VOL_DAYS + 1):m + 1]
        sd = float(np.std(seg, ddof=1) * math.sqrt(YEAR)) if len(seg) >= VOL_DAYS else 0.0
        ks.append(VOL_TARGET / sd if sd > 1e-6 else LEV_CAP)
    ks = np.array(ks)
    for name, cap in (("P5L", LEV_CAP), ("P5N", 1.0)):
        kk = np.minimum(ks, cap)
        run(name, S["blend"], W["P4_blend_60"], k=kk)
        run(name + "_0", S["blend"], W["P4_blend_60"], k=kk, cash_rate=False)
    if "SPTR" in cols:  # 60/40 with S&P 500 dividends reinvested, as a check on the price-index benchmark
        run("P2tr", hold, _weights(p, "fixed", _mask(p, ["SPTR", "TLT"]), fixed={"SPTR": 0.6, "TLT": 0.4}))

    # (2) choose each year from 2016 on prior data: rule x volatility window x with/without crypto
    configs = {}
    for rule in ("blend", "b5520"):
        for w in (60, 120):
            for crypto in (True, False):
                name = f"{rule}|{w}|{'c' if crypto else 'n'}"
                key = {("blend", 60, True): "P4b", ("blend", 60, False): "P3b", ("b5520", 60, True): "P4c",
                       ("b5520", 60, False): "P3c"}.get((rule, w, crypto))
                configs[name] = runs[key] if key else run("wf_" + name, S[rule], W[f"P{4 if crypto else 3}_{rule}_{w}"])
    last_year = idx[-1].year
    chosen, pieces, ranks = {}, [], []
    for y in range(WF_FROM, last_year + 1):
        prior = {n: _stats(r[(r.index.year < y)], "2004-01-01") for n, r in configs.items()}
        best = max(prior, key=lambda n: prior[n]["ratio"])
        chosen[y] = best
        pieces.append(configs[best][configs[best].index.year == y])
        if y < last_year:
            after = {n: _stats(r[r.index.year == y], f"{y}-01-01") for n, r in configs.items()}
            a = pd.Series({n: prior[n]["ratio"] for n in configs}).rank()
            b = pd.Series({n: after[n]["ratio"] for n in configs}).rank()
            ranks.append(float(a.corr(b)) if a.std() > 0 and b.std() > 0 else 0.0)
    runs["WF"] = pd.concat(pieces)

    # (3) drop one market at a time from P4 (blend)
    drop = {}
    for k in KEYS:
        if k in cols:
            inc = alls & ~_mask(p, [k])
            net, _ = _sim(p, S["blend"], _weights(p, "inv", inc, p["blend"], VOL_DAYS))
            drop[k] = pd.Series(net, index=idx)

    # (4) costs 0.30% and one week's delay
    for name, rule, key in (("P4b", "blend", "P4_blend_60"), ("P3b", "blend", "P3_blend_60"),
                            ("P4c", "b5520", "P4_b5520_60"), ("P3c", "b5520", "P3_b5520_60")):
        run(name + "_cost", S[rule], W[key], cost=HIGH_COST)
        run(name + "_late", S[rule], W[key], delay=WEEK)
    # amount traded a year, as a multiple of the portfolio (from the gap between the two cost levels)
    turnover = {}
    for name in ("P3b", "P4b", "P3c", "P4c"):
        gap = (runs[name] - runs[name + "_cost"])[idx >= pd.Timestamp("2004-01-01")]
        turnover[name] = float(gap.mean() * YEAR / (HIGH_COST - COST))
    later = idx >= pd.Timestamp("2004-01-01")
    flips = (np.diff(p["blend"][:, trad], axis=0) != 0) & np.isfinite(p["blend"][1:, trad])
    turnover["days"] = float(flips[later[1:]].sum(axis=0).mean() / (later.sum() / YEAR))

    # each market's own trend stream (17-rule blend held the next day, 0.10% cost, cash at 0%) for correlations
    streams, holds = {}, {}
    for j, k in enumerate(cols):
        if k not in KEYS:
            continue
        pos = pd.Series(S["blend"][:, j], index=idx).shift(1).fillna(0.0).values
        net, _ = te._run(pos, p["R"][:, j], COST)
        first = np.where(np.isfinite(p["blend"][:, j]) & p["allowed"][:, j])[0]
        if not len(first):
            continue
        s = pd.Series(net, index=idx)[idx > idx[first[0]]]
        streams[k] = _monthly(s)
        holds[k] = _monthly(pd.Series(p["R"][:, j], index=idx)[idx > idx[first[0]]])
    streams, holds = pd.DataFrame(streams), pd.DataFrame(holds)
    corr = {}
    tcols = [k for k in TRAD if k in streams]
    for tag, start in (("p04", "2004-01"), ("p14", "2014-01")):
        sel = streams.index >= pd.Period(start, "M")
        corr[tag] = {"trend": _avg_corr(streams.loc[sel, tcols]),
                     "hold": _avg_corr(holds.loc[sel, [k for k in TRAD if k in holds]])}
    if "BTC" in streams:
        sel = streams.index >= pd.Period("2014-01", "M")
        p3m = _monthly(runs["P3b"])
        both = pd.concat([streams.loc[sel, "BTC"], p3m], axis=1).dropna()
        corr["btc_p3"] = float(both.corr().iloc[0, 1]) if len(both) > 24 else None
        each = [streams.loc[sel, ["BTC", k]].dropna().corr().iloc[0, 1] for k in tcols]
        corr["btc_each"] = float(np.nanmean(each)) if each else None
        corr["btc_range"] = (float(np.nanmin(each)), float(np.nanmax(each))) if each else None
        corr["btc_hold_p1"] = float(pd.concat([holds.loc[sel, "BTC"], _monthly(runs["P1"])], axis=1).dropna()
                                    .corr().iloc[0, 1])
    # bitcoin's stand-alone blend with T-bill interest, and bitcoin held, on the same calendar (for the yearly table)
    solo = {}
    if "BTC" in cols:
        j = cols.index("BTC")
        Sb = np.zeros_like(S["blend"])
        Sb[:, j] = S["blend"][:, j]
        Wb = _weights(p, "fixed", alls, fixed={"BTC": 1.0})
        solo["btc_blend"] = pd.Series(_sim(p, Sb, Wb)[0], index=idx)
        solo["btc_hold"] = pd.Series(p["R"][:, j], index=idx)
    return {"p": p, "runs": runs, "expo": expo, "chosen": chosen, "ranks": ranks, "drop": drop, "corr": corr,
            "solo": solo, "ks": ks, "configs": list(configs), "turnover": turnover}


# ---------- the study ----------

def _cell(s: dict | None) -> str:
    return "—" if s is None else f"{_pct(s['ret'])}｜{_num(s['vol'], 1)}%｜{_num(s['ratio'])}｜{_pct(s['deep'])}"


def _cfg_text(name: str) -> str:
    rule, w, c = name.split("|")
    return f"{RULE_NAME[rule]}、{w} 日波動、{'含' if c == 'c' else '不含'}加密貨幣"


def study() -> dict:
    data, problems, origin = _load()
    r = _compute(data)
    p, runs, expo = r["p"], r["runs"], r["expo"]
    idx, cols = p["index"], p["cols"]
    asof = idx[-1]
    b2 = runs["P2"]
    has_btc = "BTC" in cols

    def st(name: str, start: str, bench: bool = True) -> dict | None:
        return _stats(runs[name], start, b2 if bench else None)

    LABEL = (("P1", "P1 一直持有 10 個傳統市場、平均分配"), ("P2", "P2 六成股票四成債券（標普／TLT）"),
             ("P3b", "P3 傳統 10 市場順勢（17 條混合）"), ("P3c", "P3 傳統 10 市場順勢（55／20）"),
             ("P4b", "P4 再加比特幣、以太幣順勢（混合）"), ("P4c", "P4 再加比特幣、以太幣順勢（55／20）"),
             ("P5L", "P5 P4 混合＋波動目標 10%，最多借到 1.5 倍"), ("P5N", "P5 P4 混合＋波動目標 10%，不借錢"),
             ("P6h", "P6 六成四成＋5% 比特幣一直持有"), ("P6r", "P6 六成四成＋5% 比特幣照 55／20"))
    rows1, full = [], {}
    for key, label in LABEL:
        base = "2014-01-01" if key.startswith("P6") else "2004-01-01"
        cells = [_cell(st(key, s)) if not (key.startswith("P6") and tag == "p04") else "—" for tag, _, s in PERIODS]
        f = st(key, base)
        f0 = st(key + "_0", base)
        full[key] = f
        whole = "（2014 起）" if key.startswith("P6") else ""
        worst = (f"{f['worst'][0]} 年 {_pct(f['worst'][1] * 100)}｜{_num(f['under'], 1)} 年{whole}" if f and f["worst"] else "—")
        beat = ("—" if key == "P2" or not f else f"{f['n']} 年裡 {f['beat']} 年{whole}")
        zero = "—" if not f0 else f"{_pct(f0['ret'])}｜{_num(f0['ratio'])}｜{_pct(f0['deep'])}{whole}"
        rows1.append([label] + cells + [worst, beat, zero])

    # ----- contribution, year by year (2014 on) -----
    rows2 = []
    y3, y4, y2 = _yearly(runs["P3b"]), _yearly(runs["P4b"]), _yearly(b2)
    yb = _yearly(r["solo"]["btc_blend"]) if has_btc else {}
    yh = _yearly(r["solo"]["btc_hold"]) if has_btc else {}
    cj = [cols.index(k) for k in CRYPTO if k in cols]
    e4 = pd.Series(expo["P4b"][:, cj].sum(axis=1) if cj else np.zeros(len(idx)), index=idx)
    ce = e4.groupby(idx.year).mean()
    better = 0
    done_years = [y for y in range(2014, asof.year + 1) if y in y4]
    for y in done_years:
        diff = (y4[y] - y3[y]) * 100
        part = y == asof.year and not (asof.month == 12 and asof.day >= 30)
        if not part:
            better += y4[y] > y3[y]
        rows2.append([f"{y}" + (f"（到 {_d(asof)[5:]}）" if part else ""), _pct(y3[y] * 100), _pct(y4[y] * 100),
                      f"{diff:+.1f} 個百分點", f"{ce.get(y, 0) * 100:.1f}%", _pct(yb.get(y, math.nan) * 100),
                      _pct(yh.get(y, math.nan) * 100), _pct(y2[y] * 100)])
    n_full = sum(1 for y in done_years if not (y == asof.year and not (asof.month == 12 and asof.day >= 30)))
    s3, s4 = st("P3b", "2014-01-01"), st("P4b", "2014-01-01")
    rows2.append(["2014 起整段（每年）", f"{_pct(s3['ret'])}（報酬÷波動 {_num(s3['ratio'])}）",
                  f"{_pct(s4['ret'])}（{_num(s4['ratio'])}）", f"{s4['ret'] - s3['ret']:+.1f} 個百分點；最深 {_pct(s3['deep'])} → {_pct(s4['deep'])}",
                  f"{e4[idx >= pd.Timestamp('2014-01-01')].mean() * 100:.1f}%",
                  _pct(_stats(r['solo']['btc_blend'], '2014-01-01')['ret']) if has_btc else "—",
                  _pct(_stats(r['solo']['btc_hold'], '2014-01-01')['ret']) if has_btc else "—",
                  _pct(st('P2', '2014-01-01')['ret'])])

    # ----- checks: choose in advance, swap markets, costs and delay, correlations -----
    wf = _stats(runs["WF"], f"{WF_FROM}-01-01", b2)
    fix4, fix3, p2w = st("P4b", f"{WF_FROM}-01-01"), st("P3b", f"{WF_FROM}-01-01"), st("P2", f"{WF_FROM}-01-01")
    chosen = r["chosen"]
    n_ch = len(chosen)
    c_blend = sum(1 for v in chosen.values() if v.startswith("blend"))
    c_60 = sum(1 for v in chosen.values() if "|60|" in v)
    c_cr = sum(1 for v in chosen.values() if v.endswith("|c"))
    ranks = r["ranks"]
    rank_avg = float(np.mean(ranks)) if ranks else None
    rank_pos = sum(1 for x in ranks if x > 0)
    rows3 = [["事先挑設定", f"{WF_FROM} 年起每年初，用 2004 年到前一年的資料，從 8 種設定（混合或 55／20 × 60 或 120 日波動 × 含不含加密貨幣）挑報酬÷波動最高的，用一年後再挑",
              f"{_cell(wf)}；{wf['n']} 年裡 {wf['beat']} 年贏六成四成",
              f"固定用 P4 混合 {_cell(fix4)}；P3 {_cell(fix3)}；六成四成 {_cell(p2w)}"],
             ["挑到什麼", f"{n_ch} 次挑選（{WF_FROM}–{asof.year}）",
              f"混合 {c_blend} 次、55／20 {n_ch - c_blend} 次；60 日 {c_60} 次、120 日 {n_ch - c_60} 次；含加密貨幣 {c_cr} 次",
              f"排名一致性（之前好的、下一年也好嗎）平均 {_num(rank_avg)}，{len(ranks)} 年裡 {rank_pos} 年是正的"]]
    for tag, start in (("p04", "2004-01-01"), ("p14", "2014-01-01")):
        res = {k: _stats(v, start) for k, v in r["drop"].items()}
        res = {k: v for k, v in res.items() if v}
        if not res:
            continue
        lo = min(res, key=lambda k: res[k]["ratio"])
        hi = max(res, key=lambda k: res[k]["ratio"])
        rets = [v["ret"] for v in res.values()]
        deeps = [v["deep"] for v in res.values()]
        base = st("P4b", start)
        rows3.append([f"拆掉一個市場（{start[:4]} 起）", f"P4 混合，{len(res)} 個市場每次拿掉一個重算",
                      f"每年 {_pct(min(rets))} 到 {_pct(max(rets))}；報酬÷波動 {_num(res[lo]['ratio'])}（拿掉{NAME[lo]}）到 "
                      f"{_num(res[hi]['ratio'])}（拿掉{NAME[hi]}）；最深 {_pct(min(deeps))} 到 {_pct(max(deeps))}",
                      f"全部市場 {_cell(base)}"])
    def three(suffix: str, start: str) -> str:
        return "；".join(f"{lab} {_num(_stats(runs[k + suffix], start)['ratio'])}（{_pct(_stats(runs[k + suffix], start)['ret'])}）"
                        for k, lab in (("P4b", "P4 混合"), ("P3b", "P3 混合"), ("P4c", "P4 55／20"), ("P3c", "P3 55／20")))

    for tag, start in (("p04", "2004-01-01"), ("p14", "2014-01-01")):
        rows3.append([f"成本 0.30%（{start[:4]} 起）", "每次買賣扣三倍成本；格式「報酬÷波動（每年報酬）」",
                      three("_cost", start), "原本 " + three("", start)])
        rows3.append([f"晚一週才做（{start[:4]} 起）", "訊號和每月調整都晚 5 個交易日才照做；格式同上",
                      three("_late", start), "原本 " + three("", start)])
    cr = r["corr"]
    rows3.append(["各市場順勢分支的相關", "每個市場單獨照混合做的每月報酬，兩兩相關的平均（傳統 10 個）",
                  f"2004 起 {_num(cr['p04']['trend'])}；2014 起 {_num(cr['p14']['trend'])}",
                  f"同樣市場一直持有 2004 起 {_num(cr['p04']['hold'])}；2014 起 {_num(cr['p14']['hold'])}"])
    if "btc_p3" in cr:
        rows3.append(["比特幣順勢分支和其他的相關（2014 起每月）", "比特幣單獨照混合做 vs. P3 組合、vs. 各傳統市場的順勢分支",
                      f"和 P3 {_num(cr['btc_p3'])}；和各市場平均 {_num(cr['btc_each'])}（{_num(cr['btc_range'][0])} 到 {_num(cr['btc_range'][1])}）",
                      f"比特幣一直持有和 P1 {_num(cr['btc_hold_p1'])}"])

    # ----- today -----
    t = len(idx) - 1
    last_me = idx[p["me"][-1]]
    W4 = _weights(p, "inv", _mask(p, KEYS), p["blend"], VOL_DAYS)[-1]
    x4 = expo["P4b"][t]
    rows4 = []
    for j, k in enumerate(cols):
        if k not in KEYS:
            continue
        own = data[k]
        bl, bb, v = p["blend"][t, j], p["b5520"][t, j], p["vol"][VOL_DAYS][t, j]
        rows4.append([NAME[k], _d(own.index[own.index <= asof][-1]),
                      f"{bl * 100:.0f}%" if np.isfinite(bl) else "—",
                      ("持有" if bb >= 0.5 else "空手") if np.isfinite(bb) else "—",
                      f"{v * 100:.1f}%" if np.isfinite(v) else "—", f"{W4[j] * 100:.1f}%", f"{x4[j] * 100:.1f}%"])
    rows4.append(["現金（國庫券）", "—", "—", "—", "—", "—", f"{max(0.0, 1 - x4.sum()) * 100:.1f}%"])

    tables = [
        {"caption": ("各組合的結果（空手的錢賺美國 13 週國庫券利息；每次買賣扣 0.10%；每月最後一個美國交易日調整一次比重）。"
                     "前四欄每格是「每年報酬｜年化波動｜報酬÷波動｜最深曾經虧多少」。「整段」指 2004 年起（P6 是 2014 年起，"
                     "因為比特幣從 2014 年才放進來）。最差一年只算完整的年份；最久虧損期是從前一個高點跌下來到重新站回的最長時間"
                     "（還沒站回的算到今天）。最後一欄是空手的錢完全沒有利息時（借錢仍付國庫券 + 1%），格式「每年報酬｜報酬÷波動｜最深」。"),
         "columns": ["組合", "2004 起", "2014 起", "2018 起", "2022 起", "最差一年｜最久虧損期（整段）", "贏六成四成的年數（整段）",
                     "現金 0% 時（整段）"],
         "rows": rows1[:16]},
        {"caption": ("比特幣、以太幣的順勢分支帶來多少：P4（加了加密貨幣）減 P3（只有傳統市場），兩者都用 17 條混合、"
                     "波動大的放少一點（依最近 60 天波動分配）。「加密貨幣平均曝險」是 P4 那一年平均放在比特幣加以太幣的比例；"
                     "後兩欄是比特幣單獨照混合做（空手放國庫券）和一直抱著，同一套美國交易日曆。"),
         "columns": ["年", "P3 只有傳統市場", "P4 加比特幣、以太幣", "差", "加密貨幣平均曝險", "比特幣單獨照混合", "比特幣一直抱著",
                     "六成股票四成債券"],
         "rows": rows2[:16]},
        {"caption": ("檢查：事先挑、換市場、加成本、晚做、相關性。格式同第一張表（每年報酬｜年化波動｜報酬÷波動｜最深）。"
                     "事先挑是把每年挑到的設定那一年的結果接起來，換設定本身的交易成本沒有算。相關性用每月報酬，1 是完全同步、0 是無關。"),
         "columns": ["檢查", "怎麼做", "結果", "對照"],
         "rows": rows3[:16]},
        {"caption": (f"今天的狀態（各市場最後一根收完的日線，到 {_d(asof)}）：17 條混合說持有幾成、55／20 是持有還是空手、最近 60 天年化波動、"
                     f"上次月底調整（{_d(last_me)}）時 P4 分給它的風險權重（波動大的放少一點），以及 P4 照規則現在的持有比例"
                     "（權重 × 混合比例，加上之後的漲跌漂移）。只是規則的狀態，不是建議。"),
         "columns": ["市場", "最後收盤日", "17 條混合", "55／20", "60 日波動", "風險權重（月底）", "P4 現在持有"],
         "rows": rows4[:16]},
    ]

    # ----- now -----
    long_n = sum(1 for row in rows4[:-1] if row[2] != "—" and float(row[2][:-1]) >= 50)
    now = [{"label": "P4（多市場順勢＋比特幣、以太幣）照規則現在持有的總比例", "value": f"{x4.sum() * 100:.0f}%（其餘放國庫券）"},
           {"label": "17 條混合說持有一半以上的市場", "value": f"{len(rows4) - 1} 個裡 {long_n} 個"}]
    if has_btc:
        j = cols.index("BTC")
        now.append({"label": f"比特幣（{_d(asof)}）", "value": f"混合 {p['blend'][t, j] * 100:.0f}%；55／20 "
                    f"{'持有' if p['b5520'][t, j] >= 0.5 else '空手'}；P4 裡佔 {x4[j] * 100:.1f}%"})
    now.append({"label": "上次每月調整日", "value": _d(last_me)})

    # ----- verdict -----
    f1, f2, f3 = full["P1"], full["P2"], full["P3b"]
    res14 ={k: _stats(v, "2004-01-01") for k, v in r["drop"].items()}
    res14 = {k: v for k, v in res14.items() if v}
    worst_drop = min(res14, key=lambda k: res14[k]["ratio"])
    rq = [v["ratio"] for v in res14.values()]
    c4 = _stats(runs["P4b_cost"], "2004-01-01")
    c4c = _stats(runs["P4c_cost"], "2004-01-01")
    l4 = _stats(runs["P4b_late"], "2004-01-01")
    f3c = full["P3c"]
    e14 = e4[idx >= pd.Timestamp("2014-01-01")].mean() * 100
    lower = f3["ret"] < f2["ret"] and f3["ret"] < f1["ret"]
    verdict = (
        f"2004 年起，10 個傳統市場的順勢組合（17 條混合、波動大的放少一點）每年 {_pct(f3['ret'])}、報酬÷波動 {_num(f3['ratio'])}、最深 {_pct(f3['deep'])}"
        f"（55／20 版 {_pct(f3c['ret'])}、{_num(f3c['ratio'])}），{'報酬低於' if lower else '對照'}六成股票四成債券（{_pct(f2['ret'])}、{_num(f2['ratio'])}、{_pct(f2['deep'])}）"
        f"和各市場平均一直持有（{_pct(f1['ret'])}、{_num(f1['ratio'])}、{_pct(f1['deep'])}）{'，但虧損淺得多' if f3['deep'] > f2['deep'] else ''}。"
        f"2014 年起加入比特幣、以太幣順勢（平均只佔 {e14:.1f}%），每年 {_pct(s3['ret'])}→{_pct(s4['ret'])}、報酬÷波動 {_num(s3['ratio'])}→{_num(s4['ratio'])}，"
        f"{n_full} 年裡 {better} 年較好。"
        f"{WF_FROM} 年起每年事先挑設定 {_pct(wf['ret'])}、{_num(wf['ratio'])}（六成四成 {_pct(p2w['ret'])}、{_num(p2w['ratio'])}）；"
        f"拿掉任一市場 {_num(min(rq))}–{_num(max(rq))}（拿掉{NAME[worst_drop]}最低）；晚一週 {_num(l4['ratio'])}，"
        f"但成本 0.30% 時混合只剩 {_num(c4['ratio'])}、55／20 {_num(c4c['ratio'])}。"
    )

    hist = "；".join(f"{NAME[k]} {_d(data[k].index[0])} 起" for k in KEYS if k in data)
    tr_note = ""
    if "P2tr" in runs:
        a, b = st("P2tr", "2004-01-01"), st("P2", "2004-01-01")
        tr_note = (f"用含股息的標普 500（^SP500TR）做六成四成，2004 年起每年 {_pct(a['ret'])}、報酬÷波動 {_num(a['ratio'])}、最深 {_pct(a['deep'])}，"
                   f"不含息版 {_pct(b['ret'])}、{_num(b['ratio'])}、{_pct(b['deep'])}。")
    ks, to = r["ks"], r["turnover"]
    me_dates = idx[p["me"]]
    k_after = ks[me_dates >= pd.Timestamp("2004-01-01")]
    notes = [
        "規則就是「順勢訊號」頁上的 17 條混合和 55 日新高進、20 日新低出，程式直接沿用同一段計算，沒有為任何市場調整參數。"
        "每個市場的規則用它自己的交易日算（比特幣、以太幣每天都有收盤，一年 365 天），再放到美國交易日的共同日曆上："
        "非美國市場和放假的市場用當天或之前最近的一個收盤；比特幣用標記在那個美國交易日的日線（台灣隔天早上 8 點收）。"
        "收盤決定、下一個交易日才持有。",
        "風險平均分配 = 波動大的放少一點：每月最後一個美國交易日，每個市場的比重是「1 ÷ 最近 60 天年化波動」，再按比例加總成 100%，"
        "這個比重乘上規則說的持有成數就是實際持有，剩下的放國庫券；所以總持有不會超過 100%（P5 除外）。兩次調整之間，"
        "只有某個市場的規則目標改變那天才調整那個市場；比特幣 2014 年 1 月、以太幣 2018 年 1 月起才放進來。",
        "2004 年起是因為新興市場 ETF（EEM）2003 年 4 月、美國長債 ETF（TLT）2002 年 7 月才有，規則還要約 200 個交易日暖機，"
        "EEM 大約 2004 年 2 月才進得來。黃金在 GLD 上市（2004-11）以前用黃金期貨接上。歷史資料：" + hist + "。",
        "價格指數不含股息：標普 500、那斯達克、台灣加權、日經、恆生都是價格指數，一直抱著實際還會多領每年約 1.5～4% 的股息，"
        "順勢組合只在持有時領得到；EEM、GLD、TLT 用還原股息後的價格。" + tr_note,
        "原油期貨（CL=F）是 Yahoo 接續近月合約的價格，換月時會跳動，不等於真正持有期貨的報酬；2020-04-20 的負價那天被拿掉。"
        "美元指數和各股價指數不能直接買，實際要用期貨或 ETF，會有追蹤差異和費用。",
        f"P5 的槓桿假設：每月底用 P4 最近 60 天的實際波動，把部位放大或縮小到年化 10%，最多 1.5 倍；超過 100% 的部分借錢，"
        f"利息是國庫券利率 + 1%（現金 0% 的版本借錢仍付這個利息）。2004 年起的月底倍數平均 {float(np.mean(k_after)) if len(k_after) else 0:.2f}"
        f"（不設上限時），{sum(1 for v in k_after if v >= LEV_CAP)} 個月達到 1.5 倍上限。實際借錢的利率和保證金規定可能更差。",
        "國庫券利息用 Yahoo 的 13 週美國國庫券殖利率（^IRX），前一天的報價 × 日曆天數 ÷ 360；非美元市場空手時也算美元利息（沒有算匯率）。"
        "各市場的漲跌都以當地貨幣計，沒有換成同一種貨幣，台股、日股、港股的匯率漲跌沒有算進去。",
        "市場清單是今天挑的：知道現在還存在、重要的市場；比特幣和以太幣更是事後看漲最多的資產。第三張表的「拆掉一個市場」只能部分檢查這點。",
        "有比特幣的年份很少（2014 年起約 12 年、以太幣約 8 年），加密貨幣分支的貢獻主要來自少數幾段大漲，下一個十年不一定一樣。",
        "成本只用一個粗略的數字（0.10%，第三張表有 0.30%），沒有算稅、ETF 費用、期貨換月成本和買賣價差。"
        f"17 條混合在傳統市場上變動很頻繁（每個市場平均一年約 {to['days']:.0f} 個美國交易日比例有變），2004 年起一年買賣的總金額約是整個組合的 "
        f"P3 混合 {to['P3b']:.0f} 倍、P4 混合 {to['P4b']:.0f} 倍，55／20 只有 P3 {to['P3c']:.1f} 倍、P4 {to['P4c']:.1f} 倍；"
        "所以混合對成本特別敏感。「順勢訊號」頁說比例變動不到一成可以不調整，這裡沒有用這個省成本的做法。",
        (f"P4 本身的年化波動只有約 {_num(full['P4b']['vol'], 1)}%，要到 10% 就得放大，所以「不借錢」的 P5 "
         + ("每個月都停在 1 倍，結果和 P4 完全一樣。" if (len(k_after) and k_after.min() >= 1) else
            f"只有 {int((k_after < 1).sum())} 個月低於 1 倍，結果和 P4 幾乎一樣。")),
        "Dalio 說的「15 個以上好的、彼此不相關的報酬來源」和 AQR、Moskowitz 等人的研究用的是幾十個期貨市場、可以做空、"
        "並且用槓桿把每個市場拉到同樣的風險；這裡只有 12 個市場、只做多、P3／P4 不借錢，比較接近一般人做得到的版本，不是重做他們的研究。",
        "過去的結果不保證未來會一樣。這裡只是整理固定規則過去做了什麼，不是買賣建議。",
    ] + problems
    sources = [{"name": f"Yahoo Finance {k} 每日價格", "url": f"https://finance.yahoo.com/quote/{k}/history"}
               for k in YAHOO if k in data]
    sources += [
        {"name": "blockchain.info 比特幣長期價格、Binance 日線（BTCUSDT、ETHUSDT）", "url": "https://www.blockchain.com/charts/market-price"},
        {"name": "Hurst, Ooi & Pedersen (2017), A Century of Evidence on Trend-Following Investing, AQR",
         "url": "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing"},
        {"name": "Moskowitz, Ooi & Pedersen (2012), Time Series Momentum, Journal of Financial Economics",
         "url": "https://doi.org/10.1016/j.jfineco.2011.11.003"},
        {"name": "Ray Dalio, Principles（「投資的聖杯」：15 個以上好的、不相關的報酬來源）", "url": "https://www.principles.com/"},
    ]
    return {
        "id": "trend_portfolio",
        "title": "多市場順勢組合，加上比特幣",
        "question": "真正的「聖杯」是不是把順勢規則同時用在比特幣、股票、黃金、債券等很多市場、風險平均分配？加入比特幣的順勢分支能改善多少？",
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
