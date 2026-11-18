"""Historical study "coin_basket": on the honest coin list (rebuilt month by month as it looked at the time, including
coins that later stopped trading), can a trend-following basket of coins do better than bitcoin's own trend rule?

Published ideas tested: Man Group "In Crypto We Trend" (2024: each coin long-or-cash by its own trend, sized by equal
risk), CTREND (Fieberg et al., JFQA 2025: one trend score from price and volume averages over many horizons) and Lee &
Wang (JFQA 2025: avoid the most volatile coins).

Coins, daily candles, the point-in-time top 20 / top 50 lists, delistings and renames all come from the survivorship
study (its store and its code, so the coin lists are identical). Every Sunday close (Binance's daily candle labelled
Sunday closes on Monday 00:00 UTC) each strategy decides target weights from closes up to that candle; they are traded at
that close and drift with prices until the next rebalance. A coin that stops trading is sold at its last close (harsher
versions take a further 50% or 100% off; renamed coins get no extra loss). Costs: 0.10% (and 0.30%) of the amount
traded; cash earns nothing. Returns, volatility and the deepest loss use daily points from 2019-01-01.

Strategies:
  S1  each coin held while above its 50-day average (or: in proportion to its own 17-rule blend), equal risk weights
      (inverse 60-day volatility, normalised over the coins in the list), total at most 100%;
  S2  S1, all cash while bitcoin closes below its 50-day average;
  S3  CTREND-lite: mean cross-sectional z-score of close / SMA(n) (n = 5, 10, 20, 50, 100, 200) and
      volume / SMA(volume, 20); top fifth, equal weight, only while bitcoin's filter is on;
  S4  S2 without the fifth of the list with the highest 30-day realised variance;
  S5  bitcoin only under 55/20; bitcoin only under the blend; half bitcoin, half ether, each under its own blend;
  S6  holding the whole list (equal weight); holding bitcoin.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard.studies2 import survivorship as sv
from dashboard.studies2.survivorship import _base, _d, _num, _ok, _pct

COST = 0.001
COST_HIGH = 0.003
YEAR = 365
START = "2019-01-01"
FIRST_PICK = 2020
PERIODS = (("2019–2021", "2019-01-01", "2021-12-31"), ("2022–2023", "2022-01-01", "2023-12-31"),
           ("2024 起", "2024-01-01", None))
MA = 50
VOL_RISK = 60
VAR_DAYS = 30
CT_MA = (5, 10, 20, 50, 100, 200)
CT_VOL = 20
DROP_SHARE = 0.2
QUINTILE = 0.2
DELAY = 7
SUNDAY = 6
BTC, ETH = "BTCUSDT", "ETHUSDT"
UNIVERSES = (("top20", "前 20 名"), ("top50", "前 50 名"), ("ever10", "曾進前 10 名"))
WEEKDAY = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")

NAMES = {
    "s1ma": "S1 風險平均＋各幣 50 日均線",
    "s1bl": "S1 風險平均＋各幣混合規則",
    "s2ma": "S2 風險平均＋50 日均線＋大盤濾網",
    "s2bl": "S2 風險平均＋混合規則＋大盤濾網",
    "s3": "S3 CTREND 簡化版前五分之一＋大盤濾網",
    "s4ma": "S4 避開高波動＋50 日均線＋大盤濾網",
    "s4bl": "S4 避開高波動＋混合規則＋大盤濾網",
    "b5520": "S5 只買比特幣：55／20",
    "bblend": "S5 只買比特幣：混合規則",
    "be": "S5 比特幣、以太幣各半：各自混合規則",
    "hold": "S6 整籃持有（平均分配）",
    "btc": "S6 一直抱著比特幣",
}
TAGS = {"s1ma": "S1 均線版", "s1bl": "S1 混合版", "s2ma": "S2 均線版", "s2bl": "S2 混合版", "s3": "S3",
        "s4ma": "S4 均線版", "s4bl": "S4 混合版"}
BASKETS = ("s1ma", "s1bl", "s2ma", "s2bl", "s3", "s4ma", "s4bl")
PICKS = BASKETS + ("b5520", "bblend", "be")
ORDER = PICKS + ("hold", "btc")
BENCH = "bblend"


# ---------- data ----------

def _prepare(raw: dict, status: dict) -> dict:
    """survivorship's panels and lists, cut down to the coins that ever enter one of the lists (plus BTC, ETH)."""
    p = sv._panels(raw, status)
    p["status"] = status
    lists = {"top20": sv._universes(p, 20, p["coins"]), "top50": sv._universes(p, 50, p["coins"])}
    top10 = sv._universes(p, 10, p["coins"])
    seen, ever = set(), {}
    for t in sorted(top10):
        seen.update(top10[t])
        ok = [s for s in seen if p["life"].at[t, s] and (p["last"][s] > t or not p["ended"][s])
              and p["count"].at[t, s] >= sv.MIN_DAYS]
        ever[t] = sorted(ok)
    lists["ever10"] = ever
    used = {BTC, ETH}
    for unis in lists.values():
        for members in unis.values():
            used.update(members)
    coins = [s for s in p["coins"] if s in used]
    days = p["days"]
    T, K = len(days), len(coins)
    col = {s: j for j, s in enumerate(coins)}
    last_idx = np.array([days.get_loc(p["last"][s]) for s in coins])
    tt = np.arange(T)[:, None]
    life = p["life"][coins].values
    exit_at = np.zeros((T, K), bool)
    haircut_ok = np.zeros(K, bool)
    for j, s in enumerate(coins):
        if p["ended"][s]:
            exit_at[last_idx[j], j] = True
            haircut_ok[j] = s not in sv.RENAMED
    member = {}
    for name, unis in lists.items():
        m = np.zeros((T, K), bool)
        cur = np.zeros(K, bool)
        for i, t in enumerate(days):
            if t in unis:
                cur = np.zeros(K, bool)
                cur[[col[s] for s in unis[t]]] = True
            m[i] = cur
        member[name] = m
    return {"days": days, "coins": coins, "col": col, "close": p["close"][coins], "qv": p["qv"][coins],
            "ret": p["ret"][coins].values, "alive_next": (tt < last_idx[None, :]) & life, "exit_at": exit_at,
            "haircut_ok": haircut_ok, "member": member, "lists": lists, "last": p["last"], "ended": p["ended"],
            "n_all": len(p["coins"]), "n_dead": sum(1 for s in p["coins"] if p["ended"][s])}


def _features(d: dict) -> dict:
    """Everything known at each close (T x K arrays)."""
    close, qv = d["close"], d["qv"]
    chg = close.pct_change(fill_method=None)
    f = {}
    mean = close.rolling(MA, min_periods=MA).mean()
    f["above"] = ((close > mean) & mean.notna()).values.astype(float)
    f["blend"] = np.nan_to_num(sv._blend(close).values, nan=0.0)
    vol = chg.rolling(VOL_RISK, min_periods=30).std(ddof=1).values
    f["inv"] = np.where(np.isfinite(vol) & (vol > 1e-9), 1.0 / np.where(vol > 1e-9, vol, 1.0), np.nan)
    f["var"] = chg.rolling(VAR_DAYS, min_periods=20).var(ddof=1).values
    parts = [(close / close.rolling(n, min_periods=n).mean()).values for n in CT_MA]
    vmean = qv.rolling(CT_VOL, min_periods=CT_VOL).mean()
    parts.append((qv / vmean.where(vmean > 0)).where(close.notna()).values)
    f["ct"] = parts
    b = close[BTC]
    bmean = b.rolling(MA, min_periods=MA).mean()
    f["btc_on"] = ((b > bmean) & bmean.notna()).values.astype(float)
    # bitcoin-only rules: 55-day high in, 20-day low out
    high, low = b.rolling(55, min_periods=55).max(), b.rolling(20, min_periods=20).min()
    state = pd.Series(np.where(b >= high, 1.0, np.where(b <= low, 0.0, np.nan)), index=b.index)
    f["b5520"] = state.where(high.notna()).ffill().fillna(0.0).values
    return f


def _weights(d: dict, f: dict, uni: str) -> dict:
    """Target weights decided at every close (T x K; rows sum to at most 1, the rest is cash)."""
    member = d["member"][uni]
    elig = member & d["alive_next"]
    T, K = elig.shape
    jb, je = d["col"][BTC], d["col"][ETH]
    inv = np.where(elig & np.isfinite(f["inv"]), f["inv"], 0.0)
    tot = inv.sum(axis=1, keepdims=True)
    risk = np.divide(inv, tot, out=np.zeros_like(inv), where=tot > 0)
    on = f["btc_on"][:, None]
    w = {"s1ma": risk * f["above"], "s1bl": risk * f["blend"]}
    w["s2ma"], w["s2bl"] = w["s1ma"] * on, w["s1bl"] * on
    # S4: drop the most volatile fifth of the list, risk weights over the rest
    var = np.where(elig & np.isfinite(f["var"]) & (inv > 0), f["var"], np.nan)
    keep = np.zeros((T, K), bool)
    for i in range(T):
        ok = np.where(np.isfinite(var[i]))[0]
        if not len(ok):
            continue
        k = int(round(DROP_SHARE * len(ok)))
        order = ok[np.argsort(-var[i, ok], kind="stable")]
        keep[i, order[k:]] = True
    inv4 = np.where(keep, inv, 0.0)
    tot4 = inv4.sum(axis=1, keepdims=True)
    risk4 = np.divide(inv4, tot4, out=np.zeros_like(inv4), where=tot4 > 0)
    w["s4ma"], w["s4bl"] = risk4 * f["above"] * on, risk4 * f["blend"] * on
    # S3: CTREND-lite score
    zs = []
    for x in f["ct"]:
        x = np.where(elig & np.isfinite(x), x, np.nan)
        with np.errstate(all="ignore"):
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            zs.append(np.where(sd > 1e-12, (x - mu) / np.where(sd > 1e-12, sd, 1.0), np.nan))
    with np.errstate(all="ignore"):
        score = np.nanmean(np.stack(zs), axis=0)
    s3 = np.zeros((T, K))
    for i in range(T):
        ok = np.where(np.isfinite(score[i]))[0]
        if not len(ok):
            continue
        k = max(1, int(math.ceil(QUINTILE * len(ok))))
        s3[i, ok[np.argsort(-score[i, ok], kind="stable")][:k]] = 1.0 / k
    w["s3"] = s3 * on
    w["_score"] = score
    # benchmarks
    z = np.zeros((T, K))
    w["b5520"] = z.copy()
    w["b5520"][:, jb] = f["b5520"]
    w["bblend"] = z.copy()
    w["bblend"][:, jb] = f["blend"][:, jb]
    w["be"] = z.copy()
    w["be"][:, jb] = 0.5 * f["blend"][:, jb]
    w["be"][:, je] = 0.5 * f["blend"][:, je] * d["alive_next"][:, je]
    cnt = elig.sum(axis=1, keepdims=True)
    w["hold"] = np.divide(elig.astype(float), cnt, out=np.zeros((T, K)), where=cnt > 0)
    w["btc"] = z.copy()
    w["btc"][:, jb] = 1.0
    return w


# ---------- simulating ----------

def _simulate(d: dict, w: np.ndarray, rb: np.ndarray, start: int, cost: float, haircut: float = 0.0):
    """Daily net returns and amount traded (days start+1 ...). Holdings in money; trade only on rebalance closes."""
    ret, exit_at, alive = d["ret"], d["exit_at"], d["alive_next"]
    cut = d["haircut_ok"] * haircut
    T, K = ret.shape
    cols = np.where(np.abs(w[start:]).sum(axis=0) > 0)[0]  # coins this strategy ever holds
    ret, exit_at, alive, cut, w = ret[:, cols], exit_at[:, cols], alive[:, cols], cut[cols], w[:, cols]
    h = np.zeros(len(cols))
    cash = 1.0
    net = np.zeros(T - 1 - start)
    traded = np.zeros(T - 1 - start)
    for t in range(start, T - 1):
        equity = h.sum() + cash
        moved = 0.0
        ex = exit_at[t] & (h > 0)
        if ex.any():
            back = h[ex] * (1 - cut[ex])
            moved += float(back.sum())
            cash += float(back.sum()) * (1 - cost)
            h[ex] = 0.0
        if rb[t] or t == start:
            total = h.sum() + cash
            target = total * np.where(alive[t], w[t], 0.0)
            move = np.abs(target - h)
            moved += float(move.sum())
            cash = total - target.sum() - cost * float(move.sum())
            h = target
        h = h * (1 + ret[t + 1])
        after = h.sum() + cash
        net[t - start] = after / equity - 1 if equity > 1e-12 else 0.0
        traded[t - start] = moved / equity if equity > 1e-12 else 0.0
    return net, traded


def _stats(net: np.ndarray, traded: np.ndarray | None = None, index=None) -> dict:
    n = len(net)
    if n < 2:
        return {"ret": None, "vol": None, "ratio": None, "deepest": None, "growth": None, "turn": None, "worst": None}
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    eq = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    turn = float(traded.sum() / (n / YEAR) * 100) if traded is not None else None
    worst = None
    if index is not None:
        s = pd.Series(net, index=index)
        worst = float(min((np.prod(1 + c.values) - 1) * 100 for _, c in s.groupby(s.index.year)))
    return {"ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "deepest": deepest, "growth": growth,
            "turn": turn, "worst": worst}


def _measure(net: np.ndarray, traded: np.ndarray, index: pd.DatetimeIndex) -> dict:
    out = {"all": _stats(net, traded, index)}
    for name, a, b in PERIODS:
        rows = (index >= pd.Timestamp(a)) & (index <= (pd.Timestamp(b) if b else index[-1]))
        out[name] = _stats(net[rows], traded[rows], index[rows])
    s = pd.Series(net, index=index)
    out["years"] = {int(y): float((np.prod(1 + c.values) - 1) * 100) for y, c in s.groupby(s.index.year)}
    out["net"] = net
    return out


def _rank_corr(a: dict, b: dict):
    keys = [k for k in a if _ok(a[k]) and _ok(b.get(k))]
    if len(keys) < 4:
        return None
    x, y = pd.Series([a[k] for k in keys]).rank(), pd.Series([b[k] for k in keys]).rank()
    if x.std() == 0 or y.std() == 0:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return v if math.isfinite(v) else None


def _delayed(d: dict, w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(w)
    out[DELAY:] = w[:-DELAY]
    return out * d["alive_next"]


def _compute(d: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    days = d["days"]
    start = int(days.get_loc(pd.Timestamp(START) - pd.Timedelta(days=1)))
    index = days[start + 1:]
    f = _features(d)
    dow = np.asarray(days.dayofweek)
    rb = {wd: dow == wd for wd in range(7)}
    res = {"w": {}, "runs": {}, "start": start, "index": index, "f": f}
    for uni, _ in UNIVERSES:
        W = _weights(d, f, uni)
        res["w"][uni] = W
        for key in ORDER:
            combos = [(SUNDAY, COST, 0.0, False)]
            if uni != "ever10":
                combos += [(SUNDAY, COST_HIGH, 0.0, False), (SUNDAY, COST, 0.0, True)]
                combos += [(wd, COST, 0.0, False) for wd in range(7) if wd != SUNDAY]
            if uni == "top20" or key in BASKETS:
                combos += [(SUNDAY, COST, 0.5, False), (SUNDAY, COST, 1.0, False)]
            for wd, cost, hc, late in combos:
                w = _delayed(d, W[key]) if late else W[key]
                net, tr = _simulate(d, w, rb[wd], start, cost, hc)
                res["runs"][(uni, key, wd, cost, hc, late)] = _measure(net, tr, index)

    # ----- pick in advance: each year, the variant of S1-S5 with the best prior return / volatility -----
    res["chain"] = {}
    for uni in ("top20", "top50", "ever10"):
        runs = {k: res["runs"][(uni, k, SUNDAY, COST, 0.0, False)] for k in PICKS}
        picks = []
        chained = np.zeros_like(res["w"][uni]["hold"])
        for year in range(FIRST_PICK, int(index[-1].year) + 1):
            before = index < pd.Timestamp(f"{year}-01-01")
            inside = index.year == year
            prior = {k: _stats(runs[k]["net"][before])["ratio"] for k in PICKS}
            best = max(PICKS, key=lambda k: prior[k] if _ok(prior[k]) else -1e9)
            during = {k: _stats(runs[k]["net"][inside])["ratio"] for k in PICKS}
            # the weights decided at close t are held on day t+1: use this year's choice for closes before its days
            rows = start + np.where(inside)[0]
            chained[rows] = res["w"][uni][best][rows]
            rank = 1 + sum(1 for k in PICKS if _ok(during[k]) and _ok(during[best]) and during[k] > during[best])
            picks.append({"year": year, "pick": best, "score": prior[best], "rank": rank,
                          "corr": _rank_corr(prior, during) if inside.sum() >= 60 else None})
        net, tr = _simulate(d, chained, rb[SUNDAY], start, COST)
        since = index >= pd.Timestamp(f"{FIRST_PICK}-01-01")
        bnet = res["runs"][(uni, BENCH, SUNDAY, COST, 0.0, False)]["net"]
        hnet = res["runs"][(uni, "btc", SUNDAY, COST, 0.0, False)]["net"]
        s = pd.Series(net[since], index=index[since])
        res["chain"][uni] = {"picks": picks, "stats": _stats(net[since], tr[since], index[since]),
                             "bench": _stats(bnet[since], None, index[since]),
                             "btc": _stats(hnet[since], None, index[since]),
                             "years": {int(y): float((np.prod(1 + c.values) - 1) * 100) for y, c in s.groupby(s.index.year)}}
    return res


# ---------- the study ----------

def _run(r: dict, uni: str, key: str, wd: int = SUNDAY, cost: float = COST, hc: float = 0.0, late: bool = False):
    return r["runs"][(uni, key, wd, cost, hc, late)]


def _verdicts(r: dict) -> dict:
    """The three conditions for each basket variant."""
    out = {}
    chain_ok = all(r["chain"][u]["stats"]["ratio"] > r["chain"][u]["bench"]["ratio"] for u in ("top20", "top50"))
    for k in BASKETS:
        uni_ok = all(_run(r, u, k)["all"]["ratio"] > _run(r, u, BENCH)["all"]["ratio"] for u in ("top20", "top50"))
        wins = {u: sum(1 for wd in range(7) if _run(r, u, k, wd)["all"]["ratio"] > _run(r, u, BENCH, wd)["all"]["ratio"])
                for u in ("top20", "top50")}
        out[k] = {"uni": uni_ok, "wins": wins, "chain": chain_ok,
                  "pass": uni_ok and chain_ok and min(wins.values()) >= 5}
    return out


def study() -> dict:
    t0 = time.time()
    data = sv._load()
    d = _prepare(data["raw"], data["pairs"]["status"])
    r = _compute(d)
    days, index = d["days"], r["index"]
    last = days[-1]
    years = sorted(_run(r, "top20", "btc")["years"])
    partial = last < pd.Timestamp(f"{years[-1]}-12-31")
    span = f"{_d(index[0])} 到 {_d(last)}"
    ver = _verdicts(r)

    def harsh(u, k):
        return f"{_pct(_run(r, u, k, cost=COST_HIGH)['all']['ret'])}／{_pct(_run(r, u, k, hc=0.5)['all']['ret'])}／" \
               f"{_pct(_run(r, u, k, hc=1.0)['all']['ret'])}"

    # ----- table 1: top 20 -----
    t1 = []
    for k in ORDER:
        m = _run(r, "top20", k)
        a = m["all"]
        t1.append([NAMES[k], _pct(a["ret"]), _num(a["ratio"]), _pct(a["deepest"]), _pct(a["worst"]),
                   _pct(a["turn"], signed=False),
                   "｜".join(f"{_pct(m[n]['ret'])}（{_num(m[n]['ratio'])}）" for n, _, _ in PERIODS), harsh("top20", k)])

    # ----- table 2: other universes -----
    t2 = []
    for k in ORDER:
        a20, a50, a10 = (_run(r, u, k)["all"] for u in ("top20", "top50", "ever10"))
        h50 = _run(r, "top50", k, hc=1.0)["all"]["ret"] if k in BASKETS else None
        t2.append([NAMES[k], _num(a20["ratio"]), _pct(a50["ret"]), _num(a50["ratio"]), _pct(a50["deepest"]),
                   _pct(a10["ret"]), _num(a10["ratio"]), _pct(h50) if h50 is not None else "—"])

    # ----- table 3: pick in advance -----
    c20, c50 = r["chain"]["top20"], r["chain"]["top50"]
    b20 = _run(r, "top20", BENCH)["years"]
    t3 = []

    def short(k):
        return NAMES[k].split(" ", 1)[1]

    for p20, p50 in zip(c20["picks"], c50["picks"]):
        y = p20["year"]
        label = f"{y}（到 {last.strftime('%m-%d')}）" if y == years[-1] and partial else str(y)
        t3.append([label, f"{short(p20['pick'])}（{_num(p20['score'])}）",
                   f"{_pct(c20['years'].get(y))}（{len(PICKS)} 組裡第 {p20['rank']}）",
                   f"{short(p50['pick'])}（{_num(p50['score'])}）",
                   f"{_pct(c50['years'].get(y))}（{len(PICKS)} 組裡第 {p50['rank']}）",
                   _pct(b20.get(y)), f"{_num(p20['corr'])}／{_num(p50['corr'])}"])
    corr20 = [p["corr"] for p in c20["picks"] if _ok(p["corr"])]
    corr50 = [p["corr"] for p in c50["picks"] if _ok(p["corr"])]
    for label, key, fmt in (("串起來：每年報酬", "ret", _pct), ("串起來：報酬÷波動", "ratio", _num),
                            ("串起來：最深曾經虧多少", "deepest", _pct)):
        t3.append([label, "—", fmt(c20["stats"][key]), "—", fmt(c50["stats"][key]), fmt(c20["bench"][key]),
                   (f"{_num(float(np.mean(corr20)))}／{_num(float(np.mean(corr50)))}" if key == "ret" and corr20 and corr50
                    else "—")])
    n_pick = len(c20["picks"])
    basket_picks20 = sum(1 for p in c20["picks"] if p["pick"] in BASKETS)
    basket_picks50 = sum(1 for p in c50["picks"] if p["pick"] in BASKETS)
    chain_win20 = sum(1 for p in c20["picks"] if c20["years"].get(p["year"], -1e9) > b20.get(p["year"], 1e9))

    # ----- table 4: robustness -----
    t4 = []
    for k in BASKETS + ("b5520", "be", "bblend"):
        row = [NAMES[k]]
        for u in ("top20", "top50"):
            vals = [_run(r, u, k, wd)["all"]["ratio"] for wd in range(7)]
            row.append(f"{_num(min(vals))}～{_num(max(vals))}")
        if k in ver:
            row.append(f"7 個裡 {ver[k]['wins']['top20']} 個／{ver[k]['wins']['top50']} 個")
        else:
            row.append("—")
        row.append(f"{_num(_run(r, 'top20', k, late=True)['all']['ratio'])}／{_num(_run(r, 'top50', k, late=True)['all']['ratio'])}")
        row.append(f"{_num(_run(r, 'top20', k, cost=COST_HIGH)['all']['ratio'])}／{_num(_run(r, 'top50', k, cost=COST_HIGH)['all']['ratio'])}")
        if k in ver:
            v = ver[k]
            row.append("、".join([("兩種名單都贏" if v["uni"] else "名單：沒有都贏"),
                                  ("星期幾：夠" if min(v["wins"].values()) >= 5 else "星期幾：不夠"),
                                  ("事先挑：贏" if v["chain"] else "事先挑：沒贏")]) + ("→算贏" if v["pass"] else "→不算贏"))
        else:
            row.append("（對照）")
        t4.append(row)

    tables = [
        {"caption": (f"「當時名單」前 20 名（每月最後一天收盤依過去 30 天成交額重選，包含後來停止交易的幣），{span}，"
                     f"每週日收盤（週一 UTC 0 點）照規則換倉、成本每次買賣 {COST * 100:.2f}%、現金沒有利息。"
                     "風險平均＝每種幣分到的錢和自己過去 60 天的波動成反比（全部名單一起算，加起來 100%），趨勢不對的那份放現金；"
                     "大盤濾網＝比特幣收盤低於 50 日均線那週全部現金；CTREND 簡化版＝把收盤÷5、10、20、50、100、200 日均線和成交額÷20 日均量"
                     "在名單裡各自標準化後平均，買分數最高的五分之一；避開高波動＝去掉名單裡過去 30 天波動最大的五分之一。"
                     "比特幣、以太幣的規則也是每週換一次。最差一年含還沒過完的今年；最後一欄是成本 0.30%、停止交易的幣出場時再虧 50%、100% 時的每年報酬。"),
         "columns": ["做法", "每年報酬", "報酬÷波動", "最深曾經虧多少", "最差一年", "每年換手",
                     "2019–21｜2022–23｜2024 起（報酬÷波動）", "成本 0.30%／再虧 50%／再虧 100%"],
         "rows": t1[:16]},
        {"caption": ("換一份名單：前 50 名，以及「曾經進過前 10 名的幣」（每月底把到那時為止進過前 10 名、還在交易的幣都放進來，"
                     "所以後來死掉的大幣也在裡面）。規則和成本都和上一張表一樣，2019 起。最後一欄是前 50 名、停止交易的幣全部歸零時的每年報酬。"),
         "columns": ["做法", "前 20 名：報酬÷波動", "前 50 名：每年報酬", "前 50 名：報酬÷波動", "前 50 名：最深虧", "曾進前 10：每年報酬",
                     "曾進前 10：報酬÷波動", "前 50 名：下市全歸零"],
         "rows": t2[:16]},
        {"caption": (f"每年年初只用 2019 年到前一年底的資料，從 S1～S5 的 {len(PICKS)} 種做法裡選報酬÷波動最高的一種，整年照用，隔年再選"
                     f"（括號是選它時的分數）。{n_pick} 年裡前 20 名有 {basket_picks20} 年、前 50 名有 {basket_picks50} 年選到幣籃（其餘選到比特幣或比特幣＋以太幣）；"
                     f"前 20 名串起來的報酬 {n_pick} 年裡 {chain_win20} 年高於只買比特幣混合規則。"
                     "最後一欄是等級相關：各做法事前的分數和當年分數各自排名後有多一致（+1 完全一致、0 無關、負的是事前越好當年越差），"
                     "前 20 名／前 50 名；串起來那格是各年平均。比較對象「比特幣混合規則」是同一段期間（2020 起）。"),
         "columns": ["年", "前 20 名事前選到", "那一年報酬（當年排名）", "前 50 名事前選到", "那一年報酬（當年排名）", "只買比特幣：混合規則",
                     "等級相關（前 20／前 50）"],
         "rows": t3[:16]},
        {"caption": ("換條件再看報酬÷波動（2019 起）：換倉日從星期一到星期日 7 種（都用前一天收盤決定）的最低～最高；"
                     "幾個星期幾贏過「只買比特幣：混合規則」（同一個換倉日相比，前 20／前 50）；晚一週才照做；成本 0.30%。"
                     "事先定好的標準：要同時（1）前 20、前 50 名兩份名單都贏（2）兩份名單都至少 7 個星期幾裡 5 個贏（3）事先挑做法串起來"
                     "在兩份名單都贏，才算比只買比特幣的順勢好。"),
         "columns": ["做法", "前 20 名：7 個換倉日", "前 50 名：7 個換倉日", "贏比特幣混合的換倉日", "晚一週（前 20／前 50）",
                     "成本 0.30%（前 20／前 50）", "算不算贏"],
         "rows": t4[:16]},
    ]

    # ----- now: what S2 and S3 would hold this week -----
    sun = days[np.where(np.asarray(days.dayofweek) == SUNDAY)[0][-1]]
    i = days.get_loc(sun)
    W = r["w"]["top20"]
    coins = d["coins"]
    btc_on = bool(r["f"]["btc_on"][i])
    bclose = float(d["close"][BTC].iloc[i])
    bmean = float(d["close"][BTC].rolling(MA).mean().iloc[i])

    def holding(key, uni="top20"):
        w = r["w"][uni][key][i]
        idx = [j for j in np.argsort(-w, kind="stable") if w[j] > 1e-9]
        return "、".join(f"{_base(coins[j])} {w[j] * 100:.0f}%" for j in idx) if idx else "沒有（全部現金）"

    def would(key, uni="top20"):
        # the same rule without the market filter, to show which coins are in trend even while the filter is off
        on = r["f"]["btc_on"][i]
        if on:
            return holding(key, uni)
        base = {"s2ma": "s1ma", "s2bl": "s1bl"}.get(key)
        if base:
            return holding(base, uni)
        sc = W["_score"][i]
        ok = np.where(np.isfinite(sc))[0]
        k = max(1, int(math.ceil(QUINTILE * len(ok))))
        return "、".join(_base(coins[j]) for j in ok[np.argsort(-sc[ok], kind="stable")][:k])

    week = f"{_d(sun + pd.Timedelta(days=1))} 到 {_d(sun + pd.Timedelta(days=7))}"
    members = [coins[j] for j in np.where(d["member"]["top20"][i])[0]]
    above = [s for s in members if r["f"]["above"][i, d["col"][s]] > 0]
    now = [
        {"label": f"大盤濾網（{_d(sun)} 週日收盤，適用 {week}）",
         "value": f"比特幣收盤 {bclose:,.0f}，{'高於' if btc_on else '低於'} 50 日均線 {bmean:,.0f}：{'可以持有' if btc_on else '全部現金'}"},
        {"label": "前 20 名裡高於自己 50 日均線的幣", "value": f"{len(members)} 個裡 {len(above)} 個" +
                                                      (f"（{'、'.join(_base(s) for s in above)}）" if above else "")},
        {"label": "S2 風險平均＋50 日均線＋大盤濾網：這一週照規則持有", "value": holding("s2ma")},
        {"label": "S2 風險平均＋混合規則＋大盤濾網：這一週照規則持有", "value": holding("s2bl")},
        {"label": "S3 CTREND 簡化版＋大盤濾網：這一週照規則持有", "value": holding("s3")},
    ]
    if not btc_on:
        now.append({"label": "濾網擋下時，不看濾網的話各規則會持有（只是狀態）",
                    "value": f"S2 均線：{would('s2ma')}；S3 分數最高：{would('s3')}"})
    now.append({"label": "最後一根收完的日線", "value": _d(last)})

    # ----- verdict -----
    bb = _run(r, "top20", BENCH)["all"]
    n_pass = sum(1 for k in BASKETS if ver[k]["pass"])
    s3m, s2m, s4m = (_run(r, "top20", k)["all"] for k in ("s3", "s2ma", "s4ma"))
    hold = _run(r, "top20", "hold")["all"]
    over20 = sum(1 for k in BASKETS if _run(r, "top20", k)["all"]["ratio"] > bb["ratio"])
    over50 = sum(1 for k in BASKETS if _run(r, "top50", k)["all"]["ratio"] > _run(r, "top50", BENCH)["all"]["ratio"])
    late_loss = sum(1 for k in BASKETS if (_run(r, "top20", k)["2024 起"]["ret"] or 0) < 0)
    wd_ok = [k for k in BASKETS if min(ver[k]["wins"].values()) >= 5]
    verdict = (f"包含後來下市的幣、每月依當時成交額選前 20 名，2019 起只買比特幣照混合規則每年 {_pct(bb['ret'])}、報酬÷波動 {_num(bb['ratio'])}；"
               f"幣籃的風險平均＋濾網 {_pct(s2m['ret'])}（{_num(s2m['ratio'])}）、CTREND {_pct(s3m['ret'])}（{_num(s3m['ratio'])}，"
               f"最深虧 {_pct(s3m['deepest'])}）、避開高波動 {_pct(s4m['ret'])}（{_num(s4m['ratio'])}），整籃抱著 {_pct(hold['ret'])}；"
               f"2024 年起 7 種幣籃裡 {late_loss} 種是虧的，比特幣混合 {_pct(_run(r, 'top20', BENCH)['2024 起']['ret'])}。"
               f"報酬÷波動高於比特幣混合的，前 20 名 7 種裡 {over20} 種、前 50 名 {over50} 種；每年事先挑做法、2020 起串起來只有 "
               f"{_num(c20['stats']['ratio'])}（前 20）和 {_num(c50['stats']['ratio'])}（前 50），比特幣混合同期 {_num(c20['bench']['ratio'])}。"
               + (f"換倉日方面{'、'.join(TAGS[k] for k in wd_ok)}在 7 個裡至少 5 個贏，" if wd_ok else "換倉日方面沒有一種在 7 個裡贏到 5 個，")
               + f"但照事先定好的三個條件，7 種裡 {n_pass} 種算贏。")

    notes = [
        f"資料和名單完全沿用「把後來消失的幣加回去」那份研究（survivorship）：幣安 {d['n_all']} 個 USDT 交易對的日線，"
        f"其中 {d['n_dead']} 個已停止交易；名單是每月最後一天收盤、依過去 30 天成交額選前 N 名（要有 90 天以上日線、還在交易）。"
        "只有一家交易所：很多幣在別處崩掉、從沒上過幣安，這裡看不到，所以加回的「消失的幣」只是一部分。",
        "停止交易的幣：日線中斷超過 7 天或現在狀態不是交易中，就在最後一根收盤賣出（基本情況照收盤價），另外列出出場時再虧 50% 和 100% 的版本；"
        "只是改名或換新代號的（例如 MATIC→POL、FTM→S）不另外扣損失。",
        f"成本只用粗略的數字（每次買賣 {COST * 100:.2f}%，另列 0.30%）。成交額排在後面的小幣，實際價差和滑價常常更高，"
        "崩跌時更可能賣不到收盤價，所以小幣的成本是低估的；風險平均會把較多的錢放在波動小的大幣上，CTREND 的前五分之一則常常是剛暴漲的小幣，"
        "後者受成本低估的影響更大。現金沒有利息，也沒有算稅。",
        "時間的算法：每個決定只用到那根日線收盤為止的資料（幣安日線在 UTC 0 點收盤），在那個收盤價換倉；換倉之間持有比例隨價格漂移，"
        "濾網也只在換倉日看。報酬、波動和最深虧損都用每日資料，一年用 365 天換算。比特幣混合規則在這裡也是每週換一次，"
        "所以數字和每天看一次的版本不完全一樣。每年換手＝一年買加賣的總金額佔資金的比例。",
        "論文裡的做法這裡都是簡化版：Man Group 是 10～15 種幣各自做多或現金、依風險平均分配，這裡只用一條 50 日均線或 17 條規則的混合；"
        "CTREND 原文合成了更多價格與成交量指標，這裡只取均線和成交量比值的平均標準分數；Lee 與 Wang 看的是報酬的變異和正向跳動，這裡只用 30 天波動。"
        "結果不能直接拿來對照論文的數字。",
        f"真正獨立的觀察很少：2019 年起只有 {len(years)} 個年度{'（今年還沒過完）' if partial else ''}，大致就是兩三輪大漲大跌，"
        "而且幣籃裡的幣多半跟著比特幣一起漲跌；事先挑做法每年只有 10 種可選，等級相關的數字很粗。差距不大的地方不該當成有差。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、這一週的狀態是什麼，不是買賣建議。",
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
    ]
    sources = [
        {"name": "Man Group (2024) In Crypto We Trend", "url": "https://www.man.com/"},
        {"name": "Fieberg, Liedtke, Poddig, Walker, Zaremba (2025) A Trend Factor for the Cross-Section of Cryptocurrency "
                 "Returns, JFQA", "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972"},
        {"name": "Lee & Wang (2025)，Journal of Financial and Quantitative Analysis（幣的報酬變異與跳動）",
         "url": "https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis"},
        {"name": "幣安公開歷史資料庫（現貨日線，含已下市交易對）", "url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/"},
    ]
    return {
        "id": "coin_basket",
        "title": "包含下市幣的幣籃順勢，有比只用比特幣好嗎",
        "question": ("在包含後來死掉的幣的真實名單上，把幾種幣組成順勢籃子——風險平均、CTREND 式的趨勢分數、或避開高波動的幣——"
                     "能不能比只對比特幣做順勢更好？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
