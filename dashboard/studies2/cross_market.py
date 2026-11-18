"""Historical study "cross_market": does the US stock market's own trend, the fear index, the bitcoin-stock
correlation, rates, the dollar or credit add anything to the bitcoin trend rule (the 17-rule blend)?

Candidates (each a 1 = in / 0 = out filter decided at the US close, used alone and multiplied into the blend):
  1. QQQ / SPY above its 50 / 100 / 200-day average;
  2. QQQ less than 10% / 15% below its one-year high;
  3. VIX at most 25 / 30, or at most its trailing one-year 80th percentile;
  4. the 60-day correlation of daily bitcoin and QQQ returns (Yae & Tian style): its 20-day change above its trailing
     80th / 90th percentile = out, the reverse (below the 20th / 10th) = out, and the level (above 0.3 or above its
     trailing 80th percentile) = out;
  5. the 10-year yield up more than 0.5 points over 60 trading days = out; the dollar index above its 100-day
     average = out;
  6. the high-yield spread above its 100-day average = out (FRED, used one trading day late; skipped if unreachable).

Procedure: (a) each year from 2019, pick among all filter settings plus "no filter" by return / volatility of
blend x filter on data from 2015-01-01 up to the year's start, chain, compare with the blend alone; (b) the two
best-looking filters (hindsight, from different families) applied unchanged to ETH and SOL; (c) take apart: 0.30%
cost, one more day of delay, strict timing, and the 24 bitcoin-close clocks of clock_robustness for the blend part.

Conventions follow rotation.py: the main grid is US trading days (the days QQQ has a bar). Bitcoin on trading day p
is the daily candle labelled p (closes 00:00 UTC, three to four hours after the US close); the decision taken then
reads US closes of day p and bitcoin candles up to p and is held until the same moment after the next trading day.
The strict check reads the bitcoin candle of the calendar day before instead. Costs: 0.10% of the amount moved.
Only public, key-less sources with the dashboard's fixed user agent; a source that fails is dropped with a note.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import walkforward as wf

COST = 0.001
COST_HIGH = 0.003
TD_YEAR = 252
YAHOO_FROM = 1356998400  # 2013-01-01
FRED_HY = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
PERIODS = (("2015", "2015-01-01"), ("2018", "2018-01-01"), ("2020", "2020-01-01"), ("2024", "2024-01-01"))
PERIOD_NAME = {"2015": "2015 起", "2018": "2018 起", "2020": "2020 起", "2024": "2024 起"}
MAIN = "2018"
TRAIN_START = "2015-01-01"
PICK_FROM = 2019
PCT_DAYS = 252  # trailing window for the percentile thresholds (one year of trading days)
CORR_DAYS, CORR_STEP = 60, 20
CLOCK_LAG_HOUR = 23  # a clock closing at or after 23:00 UTC may read that day's US closes
COINS = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"))
FAMILY = {"eq": "美股趨勢", "dd": "美股從高點回落", "vix": "恐慌指數", "corr": "相關性", "rate": "利率", "usd": "美元",
          "credit": "信用利差"}
SOURCES = {
    "SPY": ("Yahoo Finance SPY（追蹤標普 500 的基金）每日股價", "https://finance.yahoo.com/quote/SPY/history"),
    "^VIX": ("Yahoo Finance VIX 美股恐慌指數", "https://finance.yahoo.com/quote/%5EVIX/history"),
    "^TNX": ("Yahoo Finance 美國十年期公債殖利率（^TNX）", "https://finance.yahoo.com/quote/%5ETNX/history"),
    "DX-Y.NYB": ("Yahoo Finance 美元指數（DX-Y.NYB）", "https://finance.yahoo.com/quote/DX-Y.NYB/history"),
    "HY": ("FRED：美國高收益債利差（ICE BofA，BAMLH0A0HYM2）", "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"),
}

_ok, _pct, _num, _d = te._ok, te._pct, te._num, te._d


def _day(v) -> str:
    """An average daily move in percent, two decimals."""
    return f"{float(v):+.2f}%" if _ok(v) else "—"


SHORT = {"QQQ50": "QQQ 50 日均線", "QQQ100": "QQQ 100 日均線", "QQQ200": "QQQ 200 日均線", "SPY50": "SPY 50 日均線",
         "SPY100": "SPY 100 日均線", "SPY200": "SPY 200 日均線", "dd10": "QQQ 回落 10%", "dd15": "QQQ 回落 15%",
         "vix25": "VIX 25", "vix30": "VIX 30", "vixp80": "VIX 一年 80%", "up80": "相關性急升 80%",
         "up90": "相關性急升 90%", "dn80": "相關性急降 80%", "dn90": "相關性急降 90%", "lvl30": "相關性高於 0.3",
         "lvlp80": "相關性偏高 80%", "rate": "殖利率急升", "usd": "美元強勢", "credit": "信用利差擴大",
         "none": "不加濾網"}


# ---------- data ----------

def _stored(name: str, fetch, least: int) -> tuple[pd.Series, str]:
    """fetch() -> {YYYY-MM-DD: number}. A good download replaces our stored copy; a failed one falls back to it."""
    path = common.STORE / f"cross_market_{name}.json"
    note = ""
    try:
        days = fetch()
        if len(days) < least:
            raise RuntimeError(f"只回傳 {len(days)} 筆")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not path.exists():
            raise
        days = json.loads(path.read_text(encoding="utf-8"))
        note = f"這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）"
    series = pd.Series({pd.Timestamp(day): float(value) for day, value in days.items()}).sort_index()
    return series[np.isfinite(series.values)], note


def _yahoo_close(symbol: str) -> dict:
    """Plain daily closes of an index from Yahoo's public chart endpoint since 2013, finished days only."""
    reply = requests.get(ro.YAHOO.format(symbol), headers=common.HEADERS, timeout=30,
                         params={"period1": YAHOO_FROM, "period2": int(time.time()), "interval": "1d"})
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    series = pd.Series(result["indicators"]["quote"][0]["close"],
                       index=pd.to_datetime(result["timestamp"], unit="s").normalize()).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]  # a bar dated today may still be open
    return {_d(day): float(v) for day, v in series.items() if math.isfinite(float(v))}


def _share_days(symbol: str) -> dict:
    return {_d(day): float(v) for day, v in ro._share(symbol).items()}


def _fetch_hy() -> dict:
    reply = requests.get(FRED_HY, headers=common.HEADERS, timeout=10)
    reply.raise_for_status()
    frame = pd.read_csv(io.StringIO(reply.text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return {str(day)[:10]: float(v) for day, v in zip(frame.iloc[:, 0], values) if math.isfinite(v)}


def _load() -> tuple[dict, list[str]]:
    problems: list[str] = []
    data = {"daily": ro._daily_close(), "qqq": ro._share("QQQ")}  # without QQQ there is no calendar: let it raise
    for key, symbol, fetch, least in (("spy", "SPY", lambda: _share_days("SPY"), 2000),
                                      ("vix", "^VIX", lambda: _yahoo_close("^VIX"), 2000),
                                      ("tnx", "^TNX", lambda: _yahoo_close("^TNX"), 2000),
                                      ("dxy", "DX-Y.NYB", lambda: _yahoo_close("DX-Y.NYB"), 2000),
                                      ("hy", "HY", _fetch_hy, 300)):
        try:
            data[key], note = _stored(key, fetch, least)
            if note:
                problems.append(f"{SOURCES[symbol][0]}{note}。")
        except Exception as exc:  # noqa: BLE001
            data[key] = None
            problems.append(f"{SOURCES[symbol][0]}這次取不到（{type(exc).__name__}），用到它的濾網這次沒有算，"
                            "沒有改用其他方法繞過。")
    data["coins"] = {}
    from dashboard.studies2 import trend_assets as ta
    for symbol, name in COINS:
        try:
            data["coins"][name] = ta._crypto_close(symbol)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"幣安 {name} 日線這次取不到（{type(exc).__name__}），{name} 這次沒有比較。")
    return data, problems


# ---------- the filters ----------

def _settings(data: dict) -> list[tuple[str, str, str]]:
    """(key, family, plain label) of every filter setting that has data this run."""
    out = [(f"QQQ{n}", "eq", f"QQQ 高於 {n} 日均線才持有") for n in (50, 100, 200)]
    if data.get("spy") is not None:
        out += [(f"SPY{n}", "eq", f"SPY 高於 {n} 日均線才持有") for n in (50, 100, 200)]
    out += [(f"dd{x}", "dd", f"QQQ 比一年內最高點低超過 {x}% 就不持有") for x in (10, 15)]
    if data.get("vix") is not None:
        out += [("vix25", "vix", "VIX 高於 25 就不持有"), ("vix30", "vix", "VIX 高於 30 就不持有"),
                ("vixp80", "vix", "VIX 高過自己過去一年 80% 的日子就不持有")]
    out += [("up80", "corr", "相關性 20 天升幅大過過去一年 80% 的日子就不持有"),
            ("up90", "corr", "相關性 20 天升幅大過過去一年 90% 的日子就不持有"),
            ("dn80", "corr", "反方向：相關性 20 天降幅大過過去一年 80% 的日子就不持有"),
            ("dn90", "corr", "反方向：相關性 20 天降幅大過過去一年 90% 的日子就不持有"),
            ("lvl30", "corr", "相關性高於 0.3 就不持有"),
            ("lvlp80", "corr", "相關性高過自己過去一年 80% 的日子就不持有")]
    if data.get("tnx") is not None:
        out.append(("rate", "rate", "十年期美債殖利率 60 個交易日內升超過 0.5 個百分點就不持有"))
    if data.get("dxy") is not None:
        out.append(("usd", "usd", "美元指數高於 100 日均線就不持有"))
    if data.get("hy") is not None:
        out.append(("credit", "credit", "高收益債利差高於 100 日均線就不持有"))
    return out


def _compute(data: dict, strict: bool = False, with_coins: bool = True) -> dict:
    """Everything that depends on prices. strict=True reads bitcoin's candles from the calendar day before each
    trading day (nothing after the US close), as rotation.py's strict check does."""
    daily, qqq_all = data["daily"], data["qqq"]
    btc = daily.reindex(qqq_all.index).dropna()  # the candle labelled with each US trading day
    idx = btc.index
    change = btc.pct_change()
    qqq = qqq_all.reindex(idx)

    def known(series: pd.Series) -> pd.Series:
        out = series.reindex(idx - pd.Timedelta(days=1)) if strict else series.reindex(idx)
        out.index = idx
        return out

    def on_idx(series: pd.Series | None) -> pd.Series | None:
        """The last value dated on or before each trading day."""
        if series is None:
            return None
        return series.reindex(idx.union(series.index)).ffill().reindex(idx)

    def pctl(s: pd.Series, q: float) -> pd.Series:
        """Trailing one-year percentile of the days before (the day itself is left out)."""
        return s.rolling(PCT_DAYS, min_periods=PCT_DAYS).quantile(q).shift(1)

    def stay(ok: pd.Series, ready: pd.Series) -> pd.Series:
        return ok.astype(float).where(ready)

    blend = known(te._targets(daily)[0]["blend"])
    settings = _settings(data)
    feat: dict = {"qqq": qqq}
    filt: dict[str, pd.Series] = {}
    for sym, price in (("QQQ", qqq), ("SPY", on_idx(data.get("spy")))):
        if price is None:
            continue
        for n in (50, 100, 200):
            mean = price.rolling(n, min_periods=n).mean()
            feat[f"{sym}_ma{n}"], feat[sym] = mean, price
            filt[f"{sym}{n}"] = stay(price > mean, mean.notna() & price.notna())
    high = qqq.rolling(PCT_DAYS, min_periods=PCT_DAYS).max()
    fall = qqq / high - 1
    feat["fall"] = fall
    for x in (10, 15):
        filt[f"dd{x}"] = stay(fall >= -x / 100, high.notna())
    vix = on_idx(data.get("vix"))
    if vix is not None:
        thr = pctl(vix, 0.8)
        feat["vix"], feat["vix_p80"] = vix, thr
        for t in (25, 30):
            filt[f"vix{t}"] = stay(vix <= t, vix.notna())
        filt["vixp80"] = stay(vix <= thr, thr.notna() & vix.notna())
    rb = known(daily).pct_change()
    rq = qqq.pct_change()
    corr = rb.rolling(CORR_DAYS, min_periods=CORR_DAYS).corr(rq)
    step = corr - corr.shift(CORR_STEP)
    feat.update(corr=corr, step=step, step_p80=pctl(step, 0.8), step_p90=pctl(step, 0.9),
                step_p20=pctl(step, 0.2), step_p10=pctl(step, 0.1), corr_p80=pctl(corr, 0.8))
    filt["up80"] = stay(step <= feat["step_p80"], feat["step_p80"].notna() & step.notna())
    filt["up90"] = stay(step <= feat["step_p90"], feat["step_p90"].notna() & step.notna())
    filt["dn80"] = stay(step >= feat["step_p20"], feat["step_p20"].notna() & step.notna())
    filt["dn90"] = stay(step >= feat["step_p10"], feat["step_p10"].notna() & step.notna())
    filt["lvl30"] = stay(corr <= 0.3, corr.notna())
    filt["lvlp80"] = stay(corr <= feat["corr_p80"], feat["corr_p80"].notna() & corr.notna())
    tnx = on_idx(data.get("tnx"))
    if tnx is not None:
        rise = tnx - tnx.shift(60)
        feat.update(tnx=tnx, rise=rise)
        filt["rate"] = stay(rise <= 0.5, rise.notna())
    dxy = on_idx(data.get("dxy"))
    if dxy is not None:
        mean = dxy.rolling(100, min_periods=100).mean()
        feat.update(dxy=dxy, dxy_ma=mean)
        filt["usd"] = stay(dxy <= mean, mean.notna() & dxy.notna())
    hy = on_idx(data.get("hy"))
    if hy is not None:
        hy = hy.shift(1)  # FRED publishes a day's spread the next morning
        mean = hy.rolling(100, min_periods=100).mean()
        feat.update(hy=hy, hy_ma=mean)
        filt["credit"] = stay(hy <= mean, mean.notna() & hy.notna())
    keys = [k for k, _, _ in settings if k in filt]

    def run(target: pd.Series, cost: float = COST, delay: int = 1, move: pd.Series | None = None) -> pd.Series:
        pos = target.shift(delay)  # decided at row p, held from there to the next row
        mv = change if move is None else move
        net, _ = te._run(pos.values, mv.values, cost)
        return pd.Series(net, index=idx).where(pos.notna())

    def window(net: pd.Series, start: str, ref_first=None) -> pd.Series | None:
        first = net.first_valid_index()
        if first is None or first > pd.Timestamp(start) + pd.Timedelta(days=10):
            return None
        a = max(pd.Timestamp(start), first, ref_first or first)
        part = net[net.index >= a]
        return None if part.isna().any() else part

    def periods(net: pd.Series) -> dict:
        out = {}
        for name, start in PERIODS:
            part = window(net, start)
            out[name] = ro._measure(part, TD_YEAR) if part is not None else None
            if out[name]:
                out[name]["start"] = part.index[0]
        return out

    hold = change.fillna(0.0)
    targets = {"none": blend}
    targets.update({k: blend * filt[k] for k in keys})
    over = {k: run(t) for k, t in targets.items()}
    alone = {k: run(filt[k]) for k in keys}
    stats = {"over": {k: periods(v) for k, v in over.items()}, "alone": {k: periods(v) for k, v in alone.items()}}

    def same(net: pd.Series, m: dict | None) -> dict | None:
        """Measure of another run over exactly the window of m."""
        if not m:
            return None
        part = net[(net.index >= m["start"]) & (net.index <= m["last"])]
        return None if part.isna().any() else ro._measure(part, TD_YEAR)

    ref = {k: {p: {"blend": same(over["none"], stats["over"][k][p]), "hold": same(hold, stats["over"][k][p])}
               for p, _ in PERIODS} for k in over}
    share = {k: float(filt[k][idx >= pd.Timestamp(PERIODS[1][1])].mean() * 100) for k in keys}

    # ----- (a) pick in advance, year by year -----
    t0 = pd.Timestamp(TRAIN_START)
    cands = ["none"] + [k for k in keys if (over[k].first_valid_index() or idx[-1]) <= t0]
    last = idx[-1]
    chosen, before, during, rho, parts = [], {}, {}, {}, []
    for year in range(PICK_FROM, last.year + 1):
        y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        prior = {k: ro._measure(over[k][(idx >= t0) & (idx < y0)], TD_YEAR) for k in cands}
        if any(m is None for m in prior.values()):
            continue
        before[year] = {k: prior[k]["ratio"] for k in cands}
        best = max(cands, key=lambda k: before[year][k])
        rows = (idx >= y0) & (idx < y1)
        now_m = {k: ro._measure(over[k][rows], TD_YEAR) for k in cands}
        during[year] = {k: m["ratio"] for k, m in now_m.items() if m}
        ret_year = {k: m["years"].get(year) for k, m in now_m.items() if m}
        rank = sorted(during[year], key=lambda k: -during[year][k]).index(best) + 1 if best in during[year] else None
        rho[year] = wf._spearman([before[year][k] for k in cands if k in during[year]],
                                 [during[year][k] for k in cands if k in during[year]])
        chosen.append({"year": year, "key": best, "rank": rank, "n": len(during[year]), "ret": ret_year.get(best),
                       "blend": ret_year.get("none"), "hold": float((np.prod(1 + hold[rows].values) - 1) * 100)})
        parts.append(targets[best].shift(1)[rows])
    pick = None
    if parts:
        pos = pd.concat(parts)
        rows = idx >= pos.index[0]
        net, _ = te._run(pos.values, change[rows].values, COST)
        net = pd.Series(net, index=idx[rows])
        pick = {"stat": ro._measure(net, TD_YEAR), "blend": ro._measure(over["none"][rows], TD_YEAR),
                "hold": ro._measure(hold[rows], TD_YEAR), "chosen": chosen, "rho": rho, "cands": cands}
        pick["beat"] = (ro._beats(pick["stat"], pick["blend"]) or (0, 0))[1]

    # ----- the two best-looking filters (hindsight) -----
    family = {k: f for k, f, _ in settings}
    ranked = sorted([k for k in keys if stats["over"][k][MAIN]], key=lambda k: -stats["over"][k][MAIN]["ratio"])
    top = ranked[:1]
    for k in ranked[1:]:
        if family[k] != family[top[0]]:
            top.append(k)
            break

    # ----- take apart (cost, delay) and the disagreement days -----
    robust, overlap = {}, {}
    nxt = change.shift(-1)
    main_rows = idx >= pd.Timestamp(PERIODS[1][1])
    for k in ["none"] + top:
        t = targets[k]
        robust[k] = {"cost": ro._measure(window(run(t, cost=COST_HIGH), PERIODS[1][1]), TD_YEAR),
                     "delay": ro._measure(window(run(t, delay=2), PERIODS[1][1]), TD_YEAR)}
        if k == "none":
            continue
        f = filt[k]
        blend_in = (blend >= 0.5) & main_rows & nxt.notna()
        dis = blend_in & (f == 0)
        agree = blend_in & (f == 1)
        out_days = int(((f == 0) & main_rows).sum())
        overlap[k] = {"out": out_days, "days": int(main_rows.sum()), "blend_in": int(blend_in.sum()),
                      "dis": int(dis.sum()), "dis_mean": float(nxt[dis].mean() * 100) if dis.any() else None,
                      "dis_up": float((nxt[dis] > 0).mean() * 100) if dis.any() else None,
                      "dis_total": float((np.prod(1 + nxt[dis].values) - 1) * 100) if dis.any() else None,
                      "agree_mean": float(nxt[agree].mean() * 100) if agree.any() else None,
                      "agree_up": float((nxt[agree] > 0).mean() * 100) if agree.any() else None}

    # ----- (b) the same two filters on other coins -----
    coins = {}
    if with_coins:
        for name, cdaily in data.get("coins", {}).items():
            cprice = cdaily.reindex(idx)
            cmove = cprice.pct_change()
            cblend = known(te._targets(cdaily)[0]["blend"])
            nets = {"none": run(cblend, move=cmove)}
            nets.update({k: run(cblend * filt[k], move=cmove) for k in top})
            first = max(n.first_valid_index() for n in nets.values())
            a = max(first, pd.Timestamp(PERIODS[1][1]))
            res = {k: ro._measure(n[n.index >= a], TD_YEAR) for k, n in nets.items()}
            res["hold"] = ro._measure(cmove.fillna(0.0)[idx >= a], TD_YEAR)
            res["start"] = a
            coins[name] = res

    return {"idx": idx, "last": last, "keys": keys, "settings": settings, "family": family, "filt": filt,
            "feat": feat, "blend": blend, "targets": targets, "over": over, "stats": stats, "ref": ref,
            "share": share, "pick": pick, "top": top, "robust": robust, "overlap": overlap, "coins": coins,
            "hold": hold, "change": change}


# ---------- (c) the 24 bitcoin-close clocks ----------

def _clocks(r: dict) -> tuple[dict | None, str | None]:
    """Blend and blend x filter on each of clock_robustness's 24 daily clocks (calendar days, from 2018).

    The filter state a clock reads is the one of the last US trading day whose close was before that clock's close:
    a clock closing at 23:00 UTC or later reads that day's US close, an earlier one reads the day before's."""
    from dashboard.studies2 import clock_robustness as cr

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        return None, problem
    price, b0, _ = cr._grid(rows)
    m = cr._Market(price, b0, cr._prehistory(b0))
    cal = pd.date_range(r["idx"][0], m.dates[-1] + pd.Timedelta(days=2), freq="D")
    f_cal = {k: r["filt"][k].reindex(cal.union(r["idx"])).ffill().reindex(cal) for k in r["top"]}
    out = {"none": [], **{k: [] for k in r["top"]}}
    for h in range(24):
        want = m.wants(h, ("blend",))["blend"]
        days = pd.DatetimeIndex(b0 + pd.to_timedelta(np.arange(len(want)), unit="D"))
        lag = pd.Timedelta(days=0 if h >= CLOCK_LAG_HOUR else 1)
        out["none"].append(m.run(h, want))
        for k in r["top"]:
            fh = f_cal[k].reindex(days - lag).fillna(1.0).values
            out[k].append(m.run(h, want * fh))
    summary = {"start": m.dates[m.w0], "end": m.dates[-1]}
    base = np.array([x["ratio"] for x in out["none"]])
    summary["none"] = {"ratio": base, "ret": np.array([x["ret"] for x in out["none"]])}
    for k in r["top"]:
        ratio = np.array([x["ratio"] for x in out[k]])
        summary[k] = {"ratio": ratio, "ret": np.array([x["ret"] for x in out[k]]), "beat": int((ratio > base).sum())}
    return summary, problem


# ---------- the study ----------

def study() -> dict:
    data, problems = _load()
    r = _compute(data)
    strict = _compute(data, strict=True, with_coins=False)
    try:
        clock, clock_problem = _clocks(r)
    except Exception as exc:  # noqa: BLE001 - the clock check is optional
        clock, clock_problem = None, f"24 個收盤時刻的檢查這次沒有做（{type(exc).__name__}）。"
    if clock_problem:
        problems.append(clock_problem)
    st, ref, keys, last = r["stats"], r["ref"], r["keys"], r["last"]
    label = {k: lab for k, _, lab in r["settings"]}
    label["none"] = "不加濾網（混合本身）"
    fam = r["family"]
    n_settings = len(keys)

    def m_(k: str, p: str) -> dict | None:
        return st["over"][k][p]

    def diff(k: str, p: str) -> float | None:
        a, b = m_(k, p), ref[k][p]["blend"]
        return a["ratio"] - b["ratio"] if a and b else None

    def beat_blend(k: str) -> tuple[int, int] | None:
        return ro._beats(m_(k, MAIN), ref[k][MAIN]["blend"])

    # ----- tables 1 and 2: every setting -----
    def setting_row(k: str) -> list:
        a = st["alone"][k][MAIN]
        o = m_(k, MAIN)
        cells = [label[k], f"{_num((a or {}).get('ratio'))}｜{r['share'][k]:.0f}%",
                 f"{_pct((o or {}).get('ret'))}｜{_pct((o or {}).get('deepest'))}"]
        for p, _ in PERIODS:
            x, dv = m_(k, p), diff(k, p)
            cells.append(f"{_num(x['ratio'])}（{dv:+.2f}）" if x and _ok(dv) else "—")
        cells.append(ro._beat_text(beat_blend(k)))
        return cells

    hold_all = {p: ro._measure(r["hold"][r["idx"] >= pd.Timestamp(s)], TD_YEAR) for p, s in PERIODS}
    ref_rows = [["對照：一直抱著比特幣", f"{_num(hold_all[MAIN]['ratio'])}｜100%",
                 f"{_pct(hold_all[MAIN]['ret'])}｜{_pct(hold_all[MAIN]['deepest'])}"]
                + [_num(hold_all[p]["ratio"]) for p, _ in PERIODS] + ["—"],
                ["對照：17 條順勢規則混合（不加濾網）", "—",
                 f"{_pct(m_('none', MAIN)['ret'])}｜{_pct(m_('none', MAIN)['deepest'])}"]
                + [_num((m_("none", p) or {}).get("ratio")) for p, _ in PERIODS] + ["—"]]
    group1 = [k for k in keys if fam[k] in ("eq", "dd", "vix")]
    group2 = [k for k in keys if fam[k] not in ("eq", "dd", "vix")]
    cols = ["濾網（不持有的條件）", "只看濾網進出：報酬÷波動｜在場天數（2018 起）", "疊在混合上：每年報酬｜最深虧（2018 起）"] + \
           [f"疊在混合上報酬÷波動：{PERIOD_NAME[p]}（比混合多少）" for p, _ in PERIODS] + ["疊上後每年報酬贏過混合的年數（2018 起）"]
    span = f"2018-01 到 {_d(last)}"
    common_caption = ("只在美股交易日做決定：每個交易日的比特幣用幣安「標著那一天」的日線（UTC 0 點收，美股收盤後三到四小時），"
                      "那一刻讀當天的美股收盤，決定下一個交易日持有多少；每次買賣扣 0.10%，現金 0%。"
                      "「疊在混合上」＝17 條順勢規則看多的比例 × 濾網（在場 1、不在場 0）；「只看濾網」＝濾網在場就全部持有比特幣。"
                      "括號是和同一段期間的混合本身相比，報酬÷波動多（+）或少（−）多少。")
    tables = [
        {"caption": (f"美股趨勢、美股從高點回落、恐慌指數（{span} 等四段期間）。QQQ ＝ 追蹤那斯達克 100 的基金，SPY ＝ 追蹤標普 500 的基金，"
                     "VIX ＝ 美股恐慌指數。2024 起 ＝ 美國比特幣現貨 ETF 上市之後。" + common_caption),
         "columns": cols, "rows": ref_rows + [setting_row(k) for k in group1]},
        {"caption": ("相關性、利率、美元、信用利差。相關性 ＝ 比特幣和 QQQ 每天漲跌同漲同跌的程度（最近 60 個交易日），"
                     "「20 天升幅」是它和 20 個交易日前相比的變化，門檻是它自己過去一年（252 個交易日，不含當天）的百分位。"
                     + common_caption),
         "columns": cols, "rows": ref_rows + [setting_row(k) for k in group2]},
    ]

    # ----- table 3: chosen in advance -----
    pick = r["pick"]
    pick_rows = []
    if pick:
        years_all = [c["year"] for c in pick["chosen"]]
        partial = last < pd.Timestamp(f"{years_all[-1]}-12-31")
        for c in pick["chosen"]:
            y = c["year"]
            name = f"{y}（到 {last.strftime('%m-%d')}）" if y == years_all[-1] and partial else str(y)
            pick_rows.append([name, label[c["key"]], _pct(c["ret"]), _pct(c["blend"]), _pct(c["hold"]),
                              f"{c['n']} 個裡第 {c['rank']}" if c["rank"] else "—", _num(pick["rho"].get(c["year"]))])
        rhos = [v for v in pick["rho"].values() if _ok(v)]
        s, b, h = pick["stat"], pick["blend"], pick["hold"]
        pick_rows.append([f"接起來（{years_all[0]} 到 {_d(last)}）", "每年報酬｜報酬÷波動｜最深虧",
                          f"{_pct(s['ret'])}｜{_num(s['ratio'])}｜{_pct(s['deepest'])}",
                          f"{_pct(b['ret'])}｜{_num(b['ratio'])}｜{_pct(b['deepest'])}",
                          f"{_pct(h['ret'])}｜{_num(h['ratio'])}｜{_pct(h['deepest'])}",
                          f"{len(years_all)} 年裡 {pick['beat']} 年贏過混合",
                          f"平均 {_num(float(np.mean(rhos)) if rhos else None)}"])
    tables.append({
        "caption": (f"事先挑的檢查：從 {PICK_FROM} 年起，每年 1 月 1 日只用 {TRAIN_START[:4]}-01-01 到前一天的資料，"
                    f"在 {len(pick['cands']) if pick else 0} 個選項（{n_settings} 個濾網設定裡從 {TRAIN_START[:4]} 年就有資料的，加上「不加濾網」）裡，"
                    "挑「混合 × 濾網」報酬÷波動最高的，照做一整年、年年重挑，再接起來（換濾網的買賣也扣成本）。"
                    "「排名相關」是全部選項「年初以前的分數」和「那一年的分數」各排一次名，兩份名次像不像（+1 完全一樣，0 毫無關係，−1 完全顛倒）。"),
        "columns": ["年度", "年初事先挑到的", "它那一年的報酬", "混合本身那一年", "一直抱著那一年", "它在那一年的名次", "排名相關"],
        "rows": pick_rows})

    # ----- table 4: take apart -----
    top = r["top"]
    show = ["none"] + top

    def cell(m: dict | None) -> str:
        if not m:
            return "—"
        return f"{_num(m['ratio'])}｜{_pct(m['ret'])}｜{_pct(m['deepest'])}"

    srob = strict["stats"]["over"]
    apart = [["原本的設定（2018 起）"] + [cell(m_(k, MAIN)) for k in show],
             ["成本改成 0.30%"] + [cell(r["robust"][k]["cost"]) for k in show],
             ["晚一天才照做"] + [cell(r["robust"][k]["delay"]) for k in show],
             ["嚴格時間：比特幣只讀前一個日曆天的日線"] + [cell(srob[k][MAIN] if k in srob else None) for k in show]]
    for p in ("2015", "2020", "2024"):
        apart.append([f"換起算年：{PERIOD_NAME[p]}"] + [cell(m_(k, p)) for k in show])
    if clock:
        row = [f"24 個比特幣收盤時刻（日曆天，{_d(clock['start'])} 起）：報酬÷波動 最低／中位／最高"]
        for k in show:
            x = clock[k]["ratio"]
            text = f"{_num(x.min())}／{_num(np.median(x))}／{_num(x.max())}"
            if k != "none":
                text += f"｜24 個裡 {clock[k]['beat']} 個比混合高"
            row.append(text)
        apart.append(row)
    for name, c in r["coins"].items():
        apart.append([f"{name}（{_d(c['start'])} 起，同樣的規則和濾網；一直抱著 {_num(c['hold']['ratio'])}｜{_pct(c['hold']['ret'])}"
                      f"｜{_pct(c['hold']['deepest'])}）"] + [cell(c.get(k)) for k in show])
    ov = r["overlap"]
    apart.append(["2018 起濾網說「不在場」的交易日｜其中混合至少五成持有的天數"]
                 + ["—"] + [f"{ov[k]['out']:,} 天（{ov[k]['out'] / ov[k]['days'] * 100:.0f}%）｜{ov[k]['dis']:,} 天" for k in top])
    apart.append(["意見不同的那些天（混合至少五成持有、濾網不在場），比特幣下一個交易日：平均漲跌｜上漲的比例｜全部連起來"]
                 + ["—"] + [f"{_day(ov[k]['dis_mean'])}｜{_num(ov[k]['dis_up'], 0)}%｜{_pct(ov[k]['dis_total'])}" for k in top])
    apart.append(["對照：混合至少五成持有、濾網也在場的那些天：平均漲跌｜上漲的比例"]
                 + ["—"] + [f"{_day(ov[k]['agree_mean'])}｜{_num(ov[k]['agree_up'], 0)}%" for k in top])
    tables.append({
        "caption": ("拆開來看：事後看起來最好的兩個濾網（2018 起疊在混合上報酬÷波動最高、而且來自不同類別），和混合本身並排。"
                    "每格是「報酬÷波動｜每年報酬｜最深曾經虧多少」。這兩個是看過整段歷史才挑的，所以這張表是在問：挑出來的東西經不經得起改動、"
                    "搬到別的幣還成不成立。24 個收盤時刻沿用「換一個收盤時刻」那一頁的做法，用 1 小時線拼日線，每天都算（包括週末），"
                    f"濾網讀的是那個時刻之前最後一個已收盤的美股交易日（UTC {CLOCK_LAG_HOUR} 點以後收的時鐘才讀當天美股）。"
                    "ETH、SOL 用幣安日線、一樣只在美股交易日決定。平均漲跌是每天的平均，不是年化。"),
        "columns": ["檢查", "混合本身", f"混合 ×「{label[top[0]]}」" if top else "—",
                    f"混合 ×「{label[top[1]]}」" if len(top) > 1 else "—"][:2 + len(top)],
        "rows": [row[:2 + len(top)] for row in apart][:16]})

    # ----- today -----
    feat, filt = r["feat"], r["filt"]

    def state(k: str) -> str:
        v = filt[k].iloc[-1] if k in filt else None
        return "—" if not _ok(v) else ("在場" if v == 1 else "不在場")

    def val(key: str, digits: int = 2) -> str:
        return _num(feat[key].iloc[-1], digits) if key in feat else "—"

    def above(sym: str) -> str:
        return "、".join(f"{'高於' if filt[f'{sym}{n}'].iloc[-1] == 1 else '低於'} {n} 日均線" for n in (50, 100, 200)
                        if f"{sym}{n}" in filt and _ok(filt[f"{sym}{n}"].iloc[-1]))

    idx_last = r["idx"][-1]
    now = [{"label": f"最後一個用到的美股交易日（{_d(idx_last)}）",
            "value": f"比特幣 {float(data['daily'].reindex([idx_last]).iloc[0]):,.0f} 美元；17 條順勢規則裡 "
                     f"{float(r['blend'].iloc[-1]) * 100:.0f}% 看多"},
           {"label": "QQQ（追蹤那斯達克 100 的基金）", "value": f"{above('QQQ')}；比一年內最高點 {_pct(feat['fall'].iloc[-1] * 100)}"}]
    if "SPY" in feat:
        now.append({"label": "SPY（追蹤標普 500 的基金）", "value": above("SPY")})
    if "vix" in feat:
        now.append({"label": "VIX（美股恐慌指數）",
                    "value": f"{val('vix')}；過去一年第 80 百分位 {val('vix_p80')}；濾網：25 {state('vix25')}、30 {state('vix30')}、"
                             f"百分位 {state('vixp80')}"})
    now.append({"label": "比特幣和 QQQ 最近 60 個交易日的相關性",
                "value": f"{val('corr')}（20 個交易日前起變化 {val('step')}；過去一年升幅第 80 百分位 {val('step_p80')}、"
                         f"降幅第 20 百分位 {val('step_p20')}）；濾網：升幅 80% {state('up80')}、降幅 80% {state('dn80')}、"
                         f"高於 0.3 {state('lvl30')}"})
    if "tnx" in feat:
        now.append({"label": "美國十年期公債殖利率", "value": f"{val('tnx')}%，60 個交易日內變化 {val('rise')} 個百分點；濾網 {state('rate')}"})
    if "dxy" in feat:
        now.append({"label": "美元指數", "value": f"{val('dxy')}（100 日均線 {val('dxy_ma')}）；濾網 {state('usd')}"})
    if "hy" in feat:
        now.append({"label": "高收益債利差", "value": f"{val('hy')}%（100 日均線 {val('hy_ma')}%）；濾網 {state('credit')}"})
    now.append({"label": "這幾格的意思", "value": "只描述各個濾網照規則現在停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    b18 = m_("none", MAIN)
    all4 = [k for k in keys if all(_ok(diff(k, p)) and diff(k, p) > 0 for p, _ in PERIODS)]
    by_fam = {}
    for k in keys:
        dv = diff(k, MAIN)
        if _ok(dv) and (fam[k] not in by_fam or dv > by_fam[fam[k]][1]):
            by_fam[fam[k]] = (k, dv)
    fam_text = "、".join(f"{FAMILY[f]} {by_fam[f][1]:+.2f}" for f in ("eq", "dd", "vix", "corr", "rate", "usd", "credit")
                         if f in by_fam)
    s1 = (f"2018 年起比特幣 17 條順勢混合的報酬÷波動 {_num(b18['ratio'])}，{n_settings} 個濾網疊上去，各類最好的一個比混合多或少："
          f"{fam_text}；2015、2018、2020、2024 四個起算年都比混合高的 {n_settings} 個裡只有 {len(all4)} 個"
          + (f"（{'、'.join(SHORT[k] for k in all4)}）" if all4 else "") + "。")
    if pick:
        n_y = len(pick["chosen"])
        kinds = sorted({c["key"] for c in pick["chosen"]})
        what = f"每年都挑到「{SHORT[kinds[0]]}」" if len(kinds) == 1 else f"挑到 {len(kinds)} 種"
        rhos = [v for v in pick["rho"].values() if _ok(v)]
        s2 = (f"每年只用之前的資料挑，{n_y} 年{what}，接起來報酬÷波動 {_num(pick['stat']['ratio'])}"
              f"（混合 {_num(pick['blend']['ratio'])}）、{n_y} 年裡 {pick['beat']} 年報酬贏過混合，前後排名相關平均 "
              f"{_num(float(np.mean(rhos)) if rhos else None)}。")
    else:
        s2 = "事先挑的檢查這次算不出來。"
    coin_bits = []
    for name, c in r["coins"].items():
        better = sum(1 for k in top if c.get(k) and c.get("none") and c[k]["ratio"] > c["none"]["ratio"])
        coin_bits.append(f"{name} {len(top)} 個裡 {better} 個")
    s3 = (f"事後最好的「{'」「'.join(SHORT[k] for k in top)}」搬到其他幣，比混合高的是 {'、'.join(coin_bits) or '（這次沒算）'}。")
    verdict = s1 + s2 + s3

    # ----- notes -----
    def strict_text(k: str) -> str:
        a, b = m_(k, MAIN), strict["stats"]["over"].get(k, {}).get(MAIN)
        return f"{_num((a or {}).get('ratio'))} → {_num((b or {}).get('ratio'))}"

    notes = [
        f"候選名單本身就有事後的成分：這 {n_settings} 個濾網是今天看過「2020 年以後比特幣跟著美股走」、「2022 年股票和比特幣一起跌」之後才想到要測的，"
        "門檻（25、30、10%、0.5 個百分點、80% 百分位）也是常見說法、沒有事先登記。第四張表的兩個濾網更是看過整段歷史才挑的。",
        "事先挑的做法：每年年初只用 2015-01-01 到前一天的資料，分數是「混合 × 濾網」扣成本後的報酬÷波動；選項裡一定包含「不加濾網」，"
        "所以如果濾網沒有用，應該常常挑回混合本身。只有從 2015 年初就有資料的濾網才參加（高收益債利差在 FRED 上只留最近三年，不參加）。"
        "「它在那一年的名次」和「排名相關」也用報酬÷波動排：賠錢的年份裡，持有得少、起伏很小的做法這個分數反而會很低"
        "（同樣虧一點，除以很小的波動就變成很負），所以那幾年的名次要和「它那一年的報酬」一起看。",
        "時間怎麼對齊：只在美股有開盤的日子做決定。美股假日、週末比特幣照樣漲跌，這些漲跌算在下一個交易日；濾網在假日沿用前一個交易日的狀態。"
        "比特幣「標著那一天」的日線在 UTC 0 點收，是美股收盤後三到四小時，決定就在那一刻做；VIX、十年期殖利率、美元指數都用 Yahoo 標著那一天的收盤"
        "（美東下午收，比 UTC 0 點早）。高收益債利差晚一個交易日才用，因為 FRED 隔天早上才公布。"
        f"嚴格時間的檢查（比特幣訊號和相關性都只讀前一個日曆天的日線，美股收盤前約 20 小時就收完）："
        + "、".join(f"{label[k]} {strict_text(k)}" for k in ["none"] + top) + "（2018 起報酬÷波動，原本 → 嚴格）。",
        "相關性的算法：比特幣和 QQQ 在美股交易日之間的漲跌（比特幣週末的漲跌算進週一），最近 60 個交易日的相關係數；"
        "Yae & Tian（Physica A，2022）說「比特幣和股市相關性的變化」是唯一在樣本外還能預測比特幣報酬的變數，但沒有說門檻，"
        "所以這裡兩個方向都試（相關性升太快就不持有、降太快就不持有），也試相關性的高低本身。這不是原文的完整重做。",
        "SPY、QQQ 用 Yahoo 還原配息後的收盤價，VIX、十年期殖利率（^TNX，單位是百分比）、美元指數（DX-Y.NYB）用 Yahoo 的收盤值；"
        "「60 天」「100 日」「一年」都是以美股交易日算（一年 252 個）。",
        f"年數很少：2020 年起只有 {last.year - 2019} 個年度、2024 年起只有 {last.year - 2023} 個，比特幣這幾年和股市一起大跌的其實只有 2022 這一段，"
        "報酬÷波動差 0.05 以內大致可以當成沒有差別；「幾年裡幾年」多一年少一年就會差很多。",
        "成本很粗略：每次買賣扣交易金額的 0.10%（第四張表另外試 0.30%），混合每天小幅調回目標比例也照扣；現金沒有利息；一直抱著沒有扣成本；沒有算稅。",
        "ETH、SOL 的混合用它們自己的幣安日線算 17 條規則，濾網和比特幣完全一樣、沒有重新調；SOL 從 2020 年才上幣安，起算比較晚，"
        "這兩個幣是今天還在的大幣，有倖存者偏誤。",
        "「每年報酬」是把整段的總成果換算成每年的複利（一年 252 個交易日）；「報酬÷波動」是每年報酬除以每天漲跌換算成一年的起伏；"
        "「最深曾經虧多少」是從最高點往下掉最多的幅度。",
        "過去的結果不保證未來會一樣。這裡只是整理這些濾網過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "Yae & Tian（2022），Out-of-sample forecasting of cryptocurrency returns，Physica A",
         "url": "https://doi.org/10.1016/j.physa.2022.127379"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線（BTC、ETH、SOL 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance QQQ 每日股價", "url": "https://finance.yahoo.com/quote/QQQ/history"},
    ]
    for key, symbol in (("spy", "SPY"), ("vix", "^VIX"), ("tnx", "^TNX"), ("dxy", "DX-Y.NYB"), ("hy", "HY")):
        if data.get(key) is not None:
            sources.append({"name": SOURCES[symbol][0], "url": SOURCES[symbol][1]})
    if clock:
        sources.append({"name": "幣安現貨 BTCUSDT 1 小時線（24 個收盤時刻）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"})

    return {
        "id": "cross_market",
        "title": "美股、恐慌指數、相關性能不能幫比特幣順勢規則加分",
        "question": ("美股自己的趨勢、美股恐慌指數、比特幣和美股同漲同跌的程度、利率和美元，加在比特幣的順勢規則上，"
                     "能不能多出東西？事先挑好設定、拿之後的年份檢查，還成立嗎？"),
        "asof": _d(data["daily"].index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
