"""Historical study "cross_momentum": among large, liquid coins only, did holding the recently strongest ones beat
holding all of them, or just bitcoin - and did a market filter (hold nothing while the market is falling) help?

Universe: a fixed list of 23 coins with a USDT spot pair on Binance, written down once and not tuned: 19 that are
large today plus four that were large and later collapsed or stopped trading (LUNA, FTT, EOS, MATIC). A coin enters
once it has 60 daily closes. A pair whose candles stop for more than 7 days is dead from its last candle on: the
return to that last close is realised, the money sits in cash until the next rebalance, and the coin never comes back
(so the re-listed "LUNA 2.0" and the later FTT candles are not used).

Timing: Binance's daily candle labelled Sunday closes on Monday 00:00 UTC. Every decision uses closes up to and
including that Sunday candle and is held until the next Sunday candle closes. 0.10% of the amount traded is charged;
cash earns nothing. Returns, volatility and the deepest loss are measured on these weekly points.

Strategies: bitcoin only; equal weight in every coin of the universe; the top 3 / top 5 by past L-week return
(L = 1, 2, 4, 8, 12); the bottom 3; the same with a market filter (bitcoin above its 50-day average, or at least
half of the coins above their own 50-day average); and equal slices where each coin is held only while it is above
its own 50-day average (no ranking).

Daily closes are kept in cache/_store/cross_momentum_klines.json so later runs only fetch the newest candles.
Everything comes from Binance's public, key-less endpoint through the dashboard's helper (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px

COST = 0.001  # 0.10% of the amount traded
COSTS = (0.001, 0.003, 0.005, 0.01)
WEEKS = 365.25 / 7  # weeks in a year
MIN_DAYS = 60  # daily closes a coin needs before it enters the universe
MA_DAYS = 50
GAP_DAYS = 7  # candles missing for longer than this: the pair is treated as having stopped trading
LOOKBACKS = (1, 2, 4, 8, 12)  # weeks
TOPS = (3, 5)
MAIN = (4, 3)  # the set written down in advance for the "today" section: past 4 weeks, top 3
FIRST = "2019-01-01"
SPLIT = "2022-01-01"
FIRST_PICK_YEAR = 2021
MIN_YEAR_WEEKS = 8
STORE = common.STORE / "cross_momentum_klines.json"
BTC = "BTC"

# ticker, Chinese name ("" when there is no common one)
COINS = (("BTC", "比特幣"), ("ETH", "以太幣"), ("BNB", "幣安幣"), ("XRP", "瑞波幣"), ("ADA", "艾達幣"),
         ("DOGE", "狗狗幣"), ("SOL", "索拉納"), ("LTC", "萊特幣"), ("LINK", ""), ("TRX", "波場幣"), ("DOT", "波卡幣"),
         ("AVAX", "雪崩幣"), ("BCH", "比特幣現金"), ("XLM", "恆星幣"), ("ATOM", ""), ("ETC", "以太坊經典"), ("UNI", ""),
         ("NEAR", ""), ("FIL", ""), ("LUNA", "Terra 舊幣"), ("FTT", "FTX 平台幣"), ("EOS", "柚子幣"), ("MATIC", ""))
NAME = dict(COINS)
ORDER = [c for c, _ in COINS]


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _price(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:,.2f}"
    if v >= 0.01:
        return f"{v:.4f}"
    return f"{v:.6f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _coin(c: str) -> str:
    return f"{NAME[c]}（{c}）" if NAME.get(c) else c


def _coins(items) -> str:
    items = list(items)
    return "、".join(_coin(c) for c in items) if items else "沒有（全部現金）"


# ---------- data ----------

def _fetch(symbol: str, start_ms: int) -> list:
    rows, start = [], start_ms
    for _ in range(12):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    return rows


def _cut(series: pd.Series) -> tuple[pd.Series, bool]:
    """Keep the candles up to the first gap longer than GAP_DAYS. Returns (series, whether a gap was found)."""
    gaps = series.index.to_series().diff().dt.days
    late = gaps[gaps > GAP_DAYS]
    if len(late):
        return series[series.index < late.index[0]], True
    return series, False


def _load() -> tuple[dict, list[str]]:
    """Finished daily closes per coin (real candles only, cut at the first long gap) and notes about dropped coins."""
    try:
        store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
        if not isinstance(store, dict):
            store = {}
    except Exception:  # noqa: BLE001 - a damaged store starts over
        store = {}
    today = pd.Timestamp(common.today_utc())
    closes, problems = {}, []
    for coin in ORDER:
        symbol = coin + "USDT"
        item = store.get(symbol) if isinstance(store.get(symbol), dict) else {}
        days = {str(k)[:10]: float(v) for k, v in (item.get("closes") or {}).items() if common._finite(v) and v > 0}
        stale = False
        if not item.get("ended"):
            try:
                start = 0
                if days:
                    start = int((pd.Timestamp(max(days)) - pd.Timedelta(days=3)).timestamp() * 1000)
                for row in _fetch(symbol, start):
                    day = pd.to_datetime(row[0], unit="ms").normalize()
                    if day < today and float(row[4]) > 0:  # never today's unfinished candle
                        days[_d(day)] = float(row[4])
            except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the others
                if not days:
                    problems.append(f"{_coin(coin)}這次取不到資料（{type(exc).__name__}），沒有算進去。")
                    continue
                stale = True
        if len(days) < 200:
            problems.append(f"{_coin(coin)}的日線只有 {len(days)} 筆，沒有算進去。")
            continue
        series = pd.Series(days, dtype=float)
        series.index = pd.to_datetime(series.index)
        series, gap = _cut(series.sort_index())
        ended = gap or (today - series.index[-1]).days > 30
        if stale and not ended:
            problems.append(f"{_coin(coin)}這次沒有更新到，用的是先前存下來的日線（到 {_d(series.index[-1])}）。")
        store[symbol] = {"ended": bool(ended), "closes": {_d(k): float(v) for k, v in series.items()}}
        closes[coin] = series
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return closes, problems


# ---------- the rules ----------

def _prepare(closes: dict) -> dict:
    """Weekly panels. Row t (a Sunday) only holds what is known when that Sunday's candle closes, except `ret`,
    which is the return from that close to the next Sunday's close (the week the decision is held for)."""
    coins = [c for c in ORDER if c in closes]
    end = closes[BTC].index[-1]
    days = pd.date_range(min(s.index[0] for s in closes.values()), end, freq="D")
    inner, price = {}, {}
    for c in coins:
        s = closes[c]
        s = s[s.index <= end]
        filled = s.reindex(days).ffill(limit=GAP_DAYS)
        inner[c] = filled.where(days <= s.index[-1])  # nothing after the last real candle
        price[c] = filled  # the last close carried for up to a week: what a holder is left with when a pair stops
    inner, price = pd.DataFrame(inner), pd.DataFrame(price)
    eligible = inner.notna() & (inner.notna().cumsum() >= MIN_DAYS)
    mean = inner.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
    above = (inner > mean) & eligible
    sundays = pd.date_range(days[0], end, freq="W-SUN")
    w_inner, w_price = inner.loc[sundays], price.loc[sundays]
    past = {L: (w_inner / w_inner.shift(L) - 1).where(eligible.loc[sundays]) for L in LOOKBACKS}
    return {"coins": coins, "sundays": sundays, "eligible": eligible.loc[sundays], "above": above.loc[sundays],
            "past": past, "close": w_inner, "mean": mean.loc[sundays], "ret": w_price.shift(-1) / w_price - 1,
            "last_day": end, "last_close": inner.iloc[-1], "last_mean": mean.iloc[-1]}


def _pick(p: dict, L: int, n: int, weakest: bool = False) -> pd.DataFrame:
    """Equal weight in the n coins with the highest (or lowest) past L-week return among the coins in the universe."""
    past = p["past"][L]
    out = np.zeros(past.shape)
    values = past.values
    for i in range(len(past)):
        row = values[i]
        ok = np.where(np.isfinite(row))[0]
        if not len(ok):
            continue
        order = ok[np.argsort(row[ok] if weakest else -row[ok], kind="stable")][:n]
        out[i, order] = 1.0 / len(order)
    return pd.DataFrame(out, index=past.index, columns=past.columns)


def _filters(p: dict) -> dict:
    """True = the filter lets the strategy hold coins in the following week."""
    count = p["eligible"].sum(axis=1)
    return {"btc": p["above"][BTC], "breadth": (p["above"].sum(axis=1) * 2 >= count) & (count > 0)}


def _targets(p: dict) -> dict:
    """Target weights decided at each Sunday close for every variant (rows sum to at most 1; the rest is cash)."""
    eligible = p["eligible"].astype(float)
    count = eligible.sum(axis=1).replace(0, np.nan)
    flt = _filters(p)
    out = {"btc": eligible.mul(0.0).assign(**{BTC: eligible[BTC]}),
           "all": eligible.div(count, axis=0).fillna(0.0),
           "own": p["above"].astype(float).div(count, axis=0).fillna(0.0)}
    out["btc_f"] = out["btc"].mul(flt["btc"].astype(float), axis=0)
    out["all_f"] = out["all"].mul(flt["btc"].astype(float), axis=0)
    for L in LOOKBACKS:
        for n in TOPS:
            out[f"top{n}_{L}"] = _pick(p, L, n)
            out[f"top{n}_{L}_f"] = out[f"top{n}_{L}"].mul(flt["btc"].astype(float), axis=0)
            out[f"top{n}_{L}_b"] = out[f"top{n}_{L}"].mul(flt["breadth"].astype(float), axis=0)
        out[f"low3_{L}"] = _pick(p, L, 3, weakest=True)
    return out


def _simulate(p: dict, target: pd.DataFrame, cost: float = COST) -> pd.DataFrame:
    """Weekly net return and amount traded, indexed by the Sunday that ends each week. Starts from cash; weights
    drift with prices during the week; a coin that stopped trading during the week is cash at its last close."""
    sundays = p["sundays"]
    first = int(np.searchsorted(sundays, pd.Timestamp(FIRST) - pd.Timedelta(days=1), side="right")) - 1
    first = max(first, 0)
    w, r, alive = target.values, p["ret"].fillna(0.0).values, p["eligible"].values
    held = np.zeros(w.shape[1])
    net, traded = [], []
    for i in range(first, len(sundays) - 1):
        moved = float(np.abs(w[i] - held).sum())
        gross = float(w[i] @ r[i])
        net.append(gross - cost * moved)
        traded.append(moved)
        held = w[i] * (1 + r[i]) / (1 + gross) if 1 + gross > 1e-12 else np.zeros_like(held)
        held = np.where(alive[i + 1], held, 0.0)
    return pd.DataFrame({"net": net, "traded": traded}, index=sundays[first + 1:])


# ---------- measuring ----------

def _stats(net: pd.Series, traded: pd.Series | None = None) -> dict:
    n = len(net)
    if n < 2:
        return {"weeks": n, "ret": None, "vol": None, "ratio": None, "deepest": None, "growth": None, "turn": None}
    growth = float(np.prod(1 + net.values))
    ret = (growth ** (WEEKS / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net.values, ddof=1) * math.sqrt(WEEKS) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net.values)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    turn = float(traded.sum() / (n / WEEKS) * 100) if traded is not None else None
    return {"weeks": n, "ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "deepest": deepest,
            "growth": growth, "turn": turn}


def _years(net: pd.Series) -> dict:
    return {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}


def _measure(run: pd.DataFrame) -> dict:
    net, split = run["net"], pd.Timestamp(SPLIT)
    out = _stats(net, run["traded"])
    out["years"] = _years(net)
    out["early"] = _stats(net[net.index < split], run["traded"][net.index < split])
    out["late"] = _stats(net[net.index >= split], run["traded"][net.index >= split])
    return out


def _rank_corr(a: dict, b: dict) -> float | None:
    keys = [k for k in a if _ok(a[k]) and _ok(b.get(k))]
    if len(keys) < 4:
        return None
    x, y = pd.Series([a[k] for k in keys]).rank(), pd.Series([b[k] for k in keys]).rank()
    if x.std() == 0 or y.std() == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _break_even(p: dict, target: pd.DataFrame, other: pd.DataFrame | None, fixed: float | None) -> float | None:
    """Cost per trade (as a fraction) at which the variant's yearly return falls to the comparison's. The comparison
    is another strategy paying the same cost, or a fixed yearly return. None: already behind at zero cost;
    inf: still ahead at 5%."""

    def gap(c: float) -> float:
        mine = _stats(_simulate(p, target, c)["net"])["ret"]
        base = fixed if other is None else _stats(_simulate(p, other, c)["net"])["ret"]
        return mine - base

    low, high = 0.0, 0.05
    if gap(low) <= 0:
        return None
    if gap(high) > 0:
        return math.inf
    for _ in range(14):
        mid = (low + high) / 2
        if gap(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _compute(closes: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    p = _prepare(closes)
    targets = _targets(p)
    runs = {key: _simulate(p, t) for key, t in targets.items()}
    stats = {key: _measure(run) for key, run in runs.items()}
    if stats["btc"]["weeks"] < 60:
        raise RuntimeError("歷史太短，算不出來")

    # ----- pick in advance: each year, the (L, top-N) with the best score on the weeks that ended before it -----
    sets = [(L, n) for n in TOPS for L in LOOKBACKS]
    index = runs["btc"].index
    picks, chosen = [], pd.Series(index=p["sundays"], dtype=object)
    for year in range(FIRST_PICK_YEAR, int(index[-1].year) + 1):
        before, inside = index < pd.Timestamp(f"{year}-01-01"), index.year == year
        if before.sum() < WEEKS or inside.sum() < 1:
            continue
        prior = {s: _stats(runs[f"top{s[1]}_{s[0]}"]["net"][before])["ratio"] for s in sets}
        best = max(sets, key=lambda s: prior[s])
        in_year = {s: _stats(runs[f"top{s[1]}_{s[0]}"]["net"][inside])["ratio"] for s in sets}
        corr = _rank_corr(prior, in_year) if inside.sum() >= MIN_YEAR_WEEKS else None
        # the decision for a week is taken on the Sunday before the Sunday that ends it
        decide = p["sundays"][np.isin(p["sundays"] + pd.Timedelta(days=7), index[inside])]
        chosen.loc[decide] = [best] * len(decide)
        picks.append({"year": year, "set": best, "score": prior[best], "corr": corr, "weeks": int(inside.sum()),
                      "rank_in_year": 1 + sum(1 for s in sets if _ok(in_year[s]) and _ok(in_year[best])
                                              and in_year[s] > in_year[best])})
    chain = None
    if picks:
        plain = targets["all"] * 0.0
        for s in sets:
            rows = chosen.index[[v == s for v in chosen.values]]
            plain.loc[rows] = targets[f"top{s[1]}_{s[0]}"].loc[rows]
        filtered = plain.mul(_filters(p)["btc"].astype(float), axis=0)
        since = index >= pd.Timestamp(f"{picks[0]['year']}-01-01")
        chain = {"since": since, "runs": {"plain": _simulate(p, plain)[since], "filtered": _simulate(p, filtered)[since],
                                          "btc": runs["btc"][since], "all": runs["all"][since]}}
        chain["stats"] = {k: _stats(v["net"], v["traded"]) for k, v in chain["runs"].items()}
        chain["years"] = {k: _years(v["net"]) for k, v in chain["runs"].items()}
        hind = max(sets, key=lambda s: _stats(runs[f"top{s[1]}_{s[0]}"]["net"][since])["ratio"])
        chain["hindsight"] = (hind, _stats(runs[f"top{hind[1]}_{hind[0]}"]["net"][since]))

    # ----- the best-looking variant (chosen with hindsight) and what costs do to it -----
    candidates = [f"top{n}_{L}{tail}" for n in TOPS for L in LOOKBACKS for tail in ("", "_f", "_b")]
    best = max(candidates, key=lambda k: stats[k]["ratio"])
    costs = []
    for c in COSTS:
        costs.append({"cost": c, "best": _measure(_simulate(p, targets[best], c)),
                      "main": _measure(_simulate(p, targets[f"top{MAIN[1]}_{MAIN[0]}"], c)),
                      "all": _measure(_simulate(p, targets["all"], c))})
    even = {"best_btc": _break_even(p, targets[best], None, stats["btc"]["ret"]),
            "best_all": _break_even(p, targets[best], targets["all"], None),
            "main_btc": _break_even(p, targets[f"top{MAIN[1]}_{MAIN[0]}"], None, stats["btc"]["ret"]),
            "main_all": _break_even(p, targets[f"top{MAIN[1]}_{MAIN[0]}"], targets["all"], None)}
    return {"p": p, "targets": targets, "runs": runs, "stats": stats, "sets": sets, "picks": picks, "chain": chain,
            "best": best, "costs": costs, "even": even, "filters": _filters(p)}


# ---------- the study ----------

def _label(key: str) -> str:
    fixed = {"btc": "只買比特幣", "all": "全部都買（每週調回一樣多）", "own": f"各看各的：每種幣高於自己 {MA_DAYS} 日均線才持有那一份",
             "btc_f": f"只買比特幣＋大盤濾網", "all_f": "全部都買＋大盤濾網"}
    if key in fixed:
        return fixed[key]
    head, L, *tail = key.split("_")
    name = f"最弱 3 種（看過去 {L} 週）" if head.startswith("low") else f"最強 {head[3:]} 種（看過去 {L} 週）"
    if tail:
        name += "＋大盤濾網" if tail[0] == "f" else "＋過半數濾網"
    return name


def _even_text(v) -> str:
    if v is None:
        return "不算成本就已經沒有比較高"
    if v == math.inf:
        return "每次買賣扣到 5% 都還比較高"
    return f"每次買賣扣到約 {v * 100:.2f}% 就一樣"


def study() -> dict:
    closes, problems = _load()
    if BTC not in closes:
        raise RuntimeError("比特幣的日線取不到")
    r = _compute(closes)
    p, stats, runs = r["p"], r["stats"], r["runs"]
    btc, everything = stats["btc"], stats["all"]
    years = sorted(btc["years"])
    n_years = len(years)
    main_key = f"top{MAIN[1]}_{MAIN[0]}"
    index = runs["btc"].index

    def wins(m: dict, base: dict) -> int:
        return sum(1 for y in years if m["years"].get(y, -1e9) > base["years"][y])

    def row(key: str) -> list:
        m = stats[key]
        beat = "—" if key == "btc" else f"{n_years} 年裡 {wins(m, btc)} 年"
        beat += "／" + ("—" if key == "all" else f"{wins(m, everything)} 年")
        return [_label(key), _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]), beat,
                _pct(m["turn"], signed=False),
                f"{_pct(m['early']['ret'])}（{_num(m['early']['ratio'])}）｜{_pct(m['late']['ret'])}（{_num(m['late']['ratio'])}）"]

    # ----- table 1: the variants -----
    shown = (["btc", "all"] + [f"top3_{L}" for L in LOOKBACKS] + [f"top5_{MAIN[0]}", "top5_12", f"low3_{MAIN[0]}", "low3_1",
             f"{main_key}_f", f"{main_key}_b", "own", "all_f", "btc_f"])
    strong = [f"top{n}_{L}" for n in TOPS for L in LOOKBACKS]
    over_all = [k for k in strong if stats[k]["ratio"] > everything["ratio"]]
    over_btc = [k for k in strong if stats[k]["ratio"] > btc["ratio"]]
    richer_all = [k for k in strong if stats[k]["growth"] > everything["growth"]]
    richer_btc = [k for k in strong if stats[k]["growth"] > btc["growth"]]
    strong_f = [k + "_f" for k in strong]
    f_over_allf = [k for k in strong_f if stats[k]["ratio"] > stats["all_f"]["ratio"]]
    f_over_own = [k for k in strong_f if stats[k]["ratio"] > stats["own"]["ratio"]]
    weak_over = [L for L in LOOKBACKS if stats[f"low3_{L}"]["ratio"] > stats[f"top3_{L}"]["ratio"]]
    universe_sizes = p["eligible"].loc[index - pd.Timedelta(days=7)].sum(axis=1)
    tables = [{
        "caption": (f"{_d(index[0] - pd.Timedelta(days=7))} 到 {_d(index[-1])}，每週一 UTC 0 點換一次，成本每次買賣 {COST * 100:.2f}%。"
                    f"名單裡同時有 {int(universe_sizes.min())}～{int(universe_sizes.max())} 種幣。「挑最強」一共 {len(strong)} 組"
                    f"（最強 3 或 5 種 × 看過去 1、2、4、8、12 週），整段的報酬÷波動：{len(strong)} 組裡 {len(over_all)} 組高於全部都買、"
                    f"{len(over_btc)} 組高於只買比特幣；最後的錢：{len(richer_all)} 組比全部都買多、{len(richer_btc)} 組比只買比特幣多。"
                    f"加上大盤濾網後，{len(strong_f)} 組裡 {len(f_over_allf)} 組的報酬÷波動高於「全部都買＋大盤濾網」、"
                    f"{len(f_over_own)} 組高於「各看各的」。最弱 3 種和最強 3 種比，5 種回看長度裡 {len(weak_over)} 種是最弱的報酬÷波動比較高。"
                    f"大盤濾網＝比特幣收盤低於自己的 {MA_DAYS} 日均線那一週全部抱現金；過半數濾網＝名單裡高於自己 {MA_DAYS} 日均線的幣不到一半時抱現金。"
                    f"年數欄是「贏過只買比特幣／贏過全部都買」的年數（{years[-1]} 年還沒過完）。每年換手是一年買加賣的總金額佔資金的比例。"
                    "最後一欄是前後兩段各自的每年報酬（括號是報酬÷波動）。"),
        "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "贏比特幣／贏全部都買的年數", "每年換手",
                    "2019–2021｜2022 起"],
        "rows": [row(k) for k in shown]}]

    # ----- table 2: pick in advance -----
    chain = r["chain"]
    if chain is not None:
        rows = []
        for pick in r["picks"]:
            y, (L, n) = pick["year"], pick["set"]
            rows.append([f"{y}" + ("（未滿一年）" if pick["weeks"] < 50 else ""), f"最強 {n} 種、看過去 {L} 週", _num(pick["score"]),
                         f"{_pct(chain['years']['plain'].get(y))}（{len(r['sets'])} 組裡第 {pick['rank_in_year']}）",
                         _pct(chain["years"]["filtered"].get(y)), _pct(chain["years"]["btc"].get(y)),
                         _pct(chain["years"]["all"].get(y)), _num(pick["corr"])])
        cs = chain["stats"]
        corrs = [pick["corr"] for pick in r["picks"] if _ok(pick["corr"])]
        rows.append(["串起來：每年報酬", "每年換成事前選到的那組", "—", _pct(cs["plain"]["ret"]), _pct(cs["filtered"]["ret"]),
                     _pct(cs["btc"]["ret"]), _pct(cs["all"]["ret"]), _num(float(np.mean(corrs))) if corrs else "—"])
        rows.append(["串起來：報酬÷波動", "", "—", _num(cs["plain"]["ratio"]), _num(cs["filtered"]["ratio"]),
                     _num(cs["btc"]["ratio"]), _num(cs["all"]["ratio"]), "—"])
        rows.append(["串起來：最深曾經虧多少", "", "—", _pct(cs["plain"]["deepest"]), _pct(cs["filtered"]["deepest"]),
                     _pct(cs["btc"]["deepest"]), _pct(cs["all"]["deepest"]), "—"])
        hind, hind_stats = chain["hindsight"]
        pick_wins_btc = sum(1 for pick in r["picks"]
                            if chain["years"]["plain"][pick["year"]] > chain["years"]["btc"][pick["year"]])
        pick_wins_all = sum(1 for pick in r["picks"]
                            if chain["years"]["plain"][pick["year"]] > chain["years"]["all"][pick["year"]])
        tables.append({
            "caption": (f"每年年初只用那一年以前的資料（{FIRST[:4]} 年起），從 {len(r['sets'])} 組「挑最強」裡選報酬÷波動最高的一組，整年照用，隔年再重選。"
                        f"{len(r['picks'])} 年裡 {pick_wins_btc} 年贏過只買比特幣、{pick_wins_all} 年贏過全部都買。"
                        f"事後才知道這段期間最好的一組是「最強 {hind[1]} 種、看過去 {hind[0]} 週」，每年 {_pct(hind_stats['ret'])}、"
                        f"報酬÷波動 {_num(hind_stats['ratio'])}，事前選不到它的話拿到的是表上「串起來」的數字。"
                        "最後一欄是等級相關：把各組「事前的分數」和「那一年的分數」各自排名後有多一致，+1 是完全一致、0 是無關、負的是事前越好那年越差；"
                        "「串起來」那格是各年的平均。「同一組＋大盤濾網」是事前選到的那一組再加上比特幣 50 日均線濾網。"),
            "columns": ["年", "事前選到的", "選它時的報酬÷波動", "那一年的報酬（當年排名）", "同一組＋大盤濾網", "只買比特幣", "全部都買",
                        "等級相關"],
            "rows": rows[:16]})

    # ----- table 3: costs -----
    best = r["best"]
    cost_rows = []
    for item in r["costs"]:
        b, m, a = item["best"], item["main"], item["all"]
        cost_rows.append([f"{item['cost'] * 100:.2f}%", _pct(b["ret"]), _num(b["ratio"]), _pct(b["deepest"]), _pct(m["ret"]),
                          _num(m["ratio"]), f"{_pct(a['ret'])}（{_num(a['ratio'])}）", f"{_pct(btc['ret'])}（{_num(btc['ratio'])}）"])
    even = r["even"]
    tables.append({
        "caption": (f"整段期間報酬÷波動最高的一組是「{_label(best)}」（這是事後挑的，數字會偏好看），一年買加賣約是資金的 "
                    f"{_pct(stats[best]['turn'], signed=False)}。把每次買賣的成本調高：它的每年報酬和只買比特幣比，{_even_text(even['best_btc'])}；"
                    f"和付同樣成本的全部都買比，{_even_text(even['best_all'])}。事先寫好的「{_label(main_key)}」"
                    f"（一年換手 {_pct(stats[main_key]['turn'], signed=False)}）：和只買比特幣比，{_even_text(even['main_btc'])}；"
                    f"和全部都買比，{_even_text(even['main_all'])}。只買比特幣只在一開始買一次，幾乎不受成本影響。"),
        "columns": ["每次買賣成本", f"{_label(best)}：每年報酬", "報酬÷波動", "最深曾經虧多少", f"{_label(main_key)}：每年報酬",
                    "報酬÷波動", "全部都買（報酬÷波動）", "只買比特幣（報酬÷波動）"],
        "rows": cost_rows})

    # ----- table 4: the rule's state for the current week -----
    last = p["sundays"][-1]
    week = f"{_d(last + pd.Timedelta(days=1))} 到 {_d(last + pd.Timedelta(days=7))}"
    flt = r["filters"]
    btc_on, breadth_on = bool(flt["btc"].loc[last]), bool(flt["breadth"].loc[last])
    eligible_now = [c for c in p["coins"] if bool(p["eligible"].loc[last, c])]
    above_now = [c for c in eligible_now if bool(p["above"].loc[last, c])]
    past_now = p["past"][MAIN[0]].loc[last].dropna().sort_values(ascending=False, kind="stable")
    holds = [c for c in p["coins"] if r["targets"][main_key].loc[last, c] > 0]
    weakest = [c for c in p["coins"] if r["targets"][f"low3_{MAIN[0]}"].loc[last, c] > 0]
    ranked = list(past_now.index)
    listed = ranked[:10] + [c for c in ranked[10:] if c in weakest or c == BTC]
    today_rows = []
    for c in listed[:16]:
        role = "最強 3 種" if c in holds else ("最弱 3 種" if c in weakest else "—")
        today_rows.append([_coin(c), _price(p["close"].loc[last, c]), _pct(past_now[c] * 100),
                           f"{len(ranked)} 種裡第 {ranked.index(c) + 1}", _price(p["mean"].loc[last, c]),
                           "高於" if c in above_now else "低於", role])
    tables.append({
        "caption": (f"用 {_d(last)}（週日）那根日線收盤算出來、適用於 {week} 這一週的規則狀態。列出過去 {MAIN[0]} 週漲最多的前 10 種，"
                    f"加上排在後面的比特幣和最弱 3 種。名單裡現在有 {len(eligible_now)} 種幣，其中 {len(above_now)} 種高於自己的 {MA_DAYS} 日均線。"
                    "價格單位是美元（USDT）。這只是規則的狀態，不是建議。"),
        "columns": ["幣", "週日收盤", f"過去 {MAIN[0]} 週漲跌", "排名", f"{MA_DAYS} 日均線在哪", "收盤和均線比", "規則裡的角色"],
        "rows": today_rows})

    # ----- now -----
    b_close, b_mean = float(p["close"].loc[last, BTC]), float(p["mean"].loc[last, BTC])
    late_close, late_mean = float(p["last_close"][BTC]), float(p["last_mean"][BTC])
    now = [
        {"label": f"大盤濾網（{_d(last)} 週日收盤，適用 {week}）",
         "value": (f"{'沒有擋下（可以持有）' if btc_on else '擋下（全部現金）'}：比特幣收盤 {_price(b_close)}，"
                   f"{'高於' if btc_on else '低於'} {MA_DAYS} 日均線 {_price(b_mean)}")},
        {"label": f"過半數濾網（名單裡高於自己 {MA_DAYS} 日均線的幣）",
         "value": f"{len(eligible_now)} 種裡 {len(above_now)} 種，{'沒有擋下（可以持有）' if breadth_on else '擋下（全部現金）'}"},
        {"label": f"「{_label(main_key)}」這一週照規則持有", "value": _coins(holds)},
        {"label": f"「{_label(main_key)}＋大盤濾網」這一週照規則持有", "value": _coins(holds if btc_on else [])},
        {"label": f"「{_label(main_key)}＋過半數濾網」這一週照規則持有", "value": _coins(holds if breadth_on else [])},
        {"label": f"比特幣最新一根收完的日線（{_d(p['last_day'])}）",
         "value": (f"收盤 {_price(late_close)}，{'高於' if late_close > late_mean else '低於'} {MA_DAYS} 日均線 {_price(late_mean)}"
                   "（下一次決定要等這個週日收盤）")},
    ]

    # ----- verdict -----
    m, mf, af, own = stats[main_key], stats[main_key + "_f"], stats["all_f"], stats["own"]
    verdict = (f"在 {len(p['coins'])} 種大型幣裡（{_d(index[0] - pd.Timedelta(days=7))[:7]} 起），每週只買過去 {MAIN[0]} 週漲最多的 "
               f"{MAIN[1]} 種：每年 {_pct(m['ret'])}、報酬÷波動 {_num(m['ratio'])}、最深曾經虧 {_pct(m['deepest'])}；"
               f"全部都買是每年 {_pct(everything['ret'])}（{_num(everything['ratio'])}、{_pct(everything['deepest'])}），"
               f"只買比特幣是每年 {_pct(btc['ret'])}（{_num(btc['ratio'])}、{_pct(btc['deepest'])}）。"
               f"{len(strong)} 組「挑最強」裡，報酬÷波動有 {len(over_all)} 組高於全部都買、{len(over_btc)} 組高於只買比特幣。")
    verdict += (f"加上大盤濾網（比特幣跌破 {MA_DAYS} 日均線那週抱現金）後是每年 {_pct(mf['ret'])}、報酬÷波動 {_num(mf['ratio'])}、"
                f"最深虧 {_pct(mf['deepest'])}；但同一個濾網套在「全部都買」上也有每年 {_pct(af['ret'])}、{_num(af['ratio'])}、{_pct(af['deepest'])}，"
                f"不排名、各幣只在自己上漲時持有則是 {_pct(own['ret'])}、{_num(own['ratio'])}、{_pct(own['deepest'])}，")
    gain_filter = af["ratio"] - everything["ratio"]
    gain_pick = ((m["ratio"] - everything["ratio"]) + (mf["ratio"] - af["ratio"])) / 2
    if gain_filter > 0 and gain_filter > gain_pick:
        verdict += (f"所以改善主要來自「只在上漲時持有」（報酬÷波動 {_num(gain_filter, 2)[0:] if gain_filter < 0 else '+' + _num(gain_filter)}），"
                    f"「挑強的」這一步平均只{'多' if gain_pick > 0 else '少'}了 {_num(abs(gain_pick))}。")
    elif gain_pick > 0:
        verdict += (f"所以「挑強的」這一步平均讓報酬÷波動多了 {_num(gain_pick)}，比濾網帶來的 "
                    f"{'+' if gain_filter >= 0 else ''}{_num(gain_filter)} 還多。")
    else:
        verdict += "所以這段期間不管是「挑強的」還是加濾網，報酬÷波動都沒有比較高。"
    if chain is not None:
        cs = chain["stats"]
        verdict += (f"每年事前選參數、{r['picks'][0]['year']} 年起串起來是每年 {_pct(cs['plain']['ret'])}"
                    f"（同期比特幣 {_pct(cs['btc']['ret'])}、全部都買 {_pct(cs['all']['ret'])}）。")
    verdict += f"成本：事後看最好的「{_label(best)}」和只買比特幣比，{_even_text(even['best_btc'])}。"

    # ----- notes -----
    def fate(c: str) -> str:
        s = closes[c]
        return f"{_d(s.index[0])} 到 {_d(s.index[-1])}，最後收盤 {_price(s.iloc[-1])}"

    dead_text = []
    if "LUNA" in closes:
        s = closes["LUNA"]
        week_before = s[s.index <= s.index[-1] - pd.Timedelta(days=5)]
        drop = (s.iloc[-1] / week_before.iloc[-1] - 1) * 100 if len(week_before) else None
        dead_text.append(f"Terra 舊幣（LUNA，用 LUNAUSDT）：{fate('LUNA')} 美元，之後幣安停止交易；{_d(week_before.index[-1])} 收盤還有 "
                         f"{_price(week_before.iloc[-1])} 美元，那一週照最後收盤算是 {_pct(drop)}，等於幾乎全部虧光，之後離開名單。"
                         "同一個代號在 2022-05-31 重新上架的是另一種新幣，這裡不算。")
    if "FTT" in closes:
        s = closes["FTT"]
        dead_text.append(f"FTX 平台幣（FTT）：{fate('FTT')} 美元（2022-11-06 收盤還有 {_price(s.get(pd.Timestamp('2022-11-06')))} 美元）。"
                         "幣安的這個交易對日線在那之後中斷了三百多天，這裡當作它在最後那根收盤離開名單，之後恢復的日線不用；崩跌的過程都算在持有它的做法上。")
    for c in ("EOS", "MATIC"):
        if c in closes:
            dead_text.append(f"{_coin(c)}：{fate(c)} 美元，改名換幣後這個交易對停止交易，照最後收盤結算、離開名單（沒有接到改名後的新幣上）。")
    others = [c for c in p["coins"] if c not in ("LUNA", "FTT", "EOS", "MATIC")
              and closes[c].index[-1] < p["last_day"] - pd.Timedelta(days=GAP_DAYS)]
    for c in others:
        dead_text.append(f"{_coin(c)}：幣安的日線停在 {_d(closes[c].index[-1])}，照同一條規則在那天離開名單。")

    notes = [
        f"名單是固定的 {len(COINS)} 種幣（有幣安 USDT 現貨交易對）：" + "、".join(c for c in ORDER) + "。"
        f"一種幣有了 {MIN_DAYS} 根日線才進名單；交易對的日線中斷超過 {GAP_DAYS} 天就當作停止交易：最後一段漲跌照最後收盤算完、"
        "那筆錢到下次換倉前是現金，之後不再回到名單。",
        "停止交易的幣是這樣處理的：" + "".join(f"（{i + 1}）{t}" for i, t in enumerate(dead_text)) if dead_text
        else "這次名單裡沒有停止交易的幣。",
        "這份名單是今天才寫下來的。雖然特地加了幾種後來崩掉或消失的大幣，它還是偏向活下來的幣：當年很大、後來死掉的幣還有很多沒放進來，"
        "名單裡其他的幣也都是事後知道它們撐到現在才選的。所以「全部都買」和「挑最強」的數字都比當年真的能做到的好看。",
        "這些幣大多一起漲一起跌，所以雖然有好幾種幣、好幾百週，真正獨立的觀察很少；用「年」來比只有 "
        f"{n_years} 年（{years[-1]} 年還沒過完），幾年的輸贏很容易只是運氣。表上的差距如果不大，就不該當成有差。",
        f"成本只用一個粗略的數字：每次買賣扣成交金額的 {COST * 100:.2f}%。比特幣以外的幣實際的買賣價差和滑價比這個高，早年和行情劇烈時更高；"
        "「挑最強」每週換幣，買賣的金額是全部都買的好幾倍，所以成本表比第一張表更接近實際。LUNA 崩跌那一週照收盤價結算，"
        "實際上當時很可能根本賣不到那個價。現金假設沒有利息，也沒有算稅。",
        "時間的算法：幣安日線在 UTC 0 點收盤，「週日收盤」就是週一 UTC 0 點。每次決定只用到那根週日日線為止的收盤價，"
        "並假設剛好用那個收盤價換倉，一路持有到下個週日收盤。報酬、波動和最深虧損都是用每週一個點算的，週中間更深的下跌看不到；"
        f"一年用 {WEEKS:.1f} 週換算。過去 L 週漲跌＝這個週日收盤 ÷ L 週前的週日收盤。資金一開始是現金，第一次買進也扣成本。",
        f"「各看各的」是把資金平分成名單裡幣的數目那麼多份，每一份只在那種幣收盤高於自己 {MA_DAYS} 日均線時持有，否則那一份是現金，"
        "完全不排名；拿它和「挑最強＋濾網」比，可以分開「只在上漲時持有」和「挑強的」各自的作用。"
        "把一部分時間換成現金本來就會讓波動和最深虧損變小，所以濾網讓「最深虧損變淺」不算意外。",
        "事前選參數那張表每年只有 10 組可以排名、而且各組持有的幣高度重疊，等級相關的數字很粗，只能看個大概。"
        "「報酬÷波動最高的一組」是事後從 30 組（10 組 × 不加濾網、大盤濾網、過半數濾網）裡挑的，挑過之後的數字一定偏好看。",
        "這個題目來自已發表的研究：Liu、Tsyvinski、Wu（2022）說過去 1～4 週漲最多的幣下一週還是比較強；Dobrynskaya（2023）說動能大約維持 2～4 週、"
        "超過一個月會反轉；Fieberg 等人（2024）用多種趨勢指標合成一個因子。批評的意見是利潤主要在小型、流動性差的幣上，扣掉成本就沒了。"
        "這裡只看大型幣，樣本和做法都和論文不同，結果不能直接拿來對照論文的數字。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在的狀態是什麼，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "幣安現貨日線（各幣對 USDT）", "url": "https://www.binance.com/zh-TC/markets/overview"},
        {"name": "Liu, Tsyvinski, Wu (2022) Common Risk Factors in Cryptocurrency, Journal of Finance",
         "url": "https://doi.org/10.1111/jofi.13119"},
        {"name": "Dobrynskaya (2023) Cryptocurrency Momentum and Reversal, Journal of Alternative Investments",
         "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263"},
        {"name": "Fieberg, Liedtke, Poddig, Walker, Zaremba (2024) A Trend Factor for the Cross-Section of Cryptocurrency "
                 "Returns, JFQA", "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972"},
    ]

    return {
        "id": "cross_momentum",
        "title": "大型幣裡挑最近最強的，有比全部都買好嗎",
        "question": "只看大型、好買賣的幣：每週只持有最近漲最多的幾種，有沒有贏過全部都買、或只買比特幣？加上「大盤在跌就抱現金」的濾網有沒有幫助？",
        "asof": _d(p["last_day"]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
