"""Historical study "pair_rotation": does following the trend of the ETH/BTC ratio - holding ether while the ratio
is in an uptrend, otherwise bitcoin - add anything to holding each coin under its own trend rule?

A. Each coin under the 17-rule blend of trend_ensemble; "50/50": half the money follows BTC's blend, half ETH's.
B. Ratio rotation on Binance's ETHBTC close: ratio above its N-day average (N = 20, 50, 100, 200) or the 20/10 and
   55/20 high/low breakouts -> ETH, else BTC. (i) always fully invested; (ii) the chosen coin only in the share its
   own blend says, the rest cash; (iii) no ratio: the coin whose blend share is higher, in that share (ties -> BTC).
C. The same on BTC against an equal-weight basket of ETH, BNB, XRP, ADA, SOL ("altseason").
D. SOL/BTC, BNB/BTC, XRP/BTC with exactly the settings picked on ETH/BTC.

Every decision is taken at the close of day t from data up to that close and held during day t+1; 0.10% of the
amount traded is charged (a switch from ETH to BTC trades twice the money); fractional positions are put back to
their target every close. The yearly pick uses only days before 1 January of that year.

Data: Binance spot daily candles (public, key-less, through the dashboard's helper), kept in
cache/_store/pair_rotation_klines.json. The 24-clock check reads the hourly stores of seasonality.py (bitcoin) and
clock_robustness.py (ether) without writing to them; missing recent hours are fetched into memory only.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common

COST = 0.001
COSTS = (0.001, 0.003, 0.005)
YEAR = 365
FIRST_PICK_YEAR = 2019
STORE = common.STORE / "pair_rotation_klines.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "ETHBTC", "SOLBTC", "BNBBTC", "XRPBTC")
BASKET = ("ETH", "BNB", "XRP", "ADA", "SOL")
BASKET_WAIT = 30  # a coin joins the basket after this many daily closes
OTHERS = ("SOL", "BNB", "XRP")
SPANS = (("2018–2021", "2018-01-01", "2021-12-31"), ("2022–2023", "2022-01-01", "2023-12-31"),
         ("2024 到最新", "2024-01-01", None))
SETTINGS = (("ma20", "高於 20 日均線"), ("ma50", "高於 50 日均線"), ("ma100", "高於 100 日均線"),
            ("ma200", "高於 200 日均線"), ("bo20", "創 20 日新高／跌破 10 日新低"), ("bo55", "創 55 日新高／跌破 20 日新低"))
SET_NAME = dict(SETTINGS)
NAME_ZH = {"ETH": "以太幣", "BNB": "幣安幣", "XRP": "瑞波幣", "SOL": "索拉納", "ADA": "艾達幣", "BASKET": "大型山寨幣籃子"}
SEASON_HOURLY = common.STORE / "seasonality_hourly.csv"
ETH_HOURLY = common.STORE / "clock_robustness_hourly_ETHUSDT.csv"


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
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


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _label(key: str, alt: str = "ETH") -> str:
    coin = NAME_ZH[alt]
    if key == "half":
        return f"各自順勢 50/50（比特幣、{coin}各一半）"
    if key == "btc":
        return "只拿比特幣（照自己的順勢規則）"
    if key == "alt":
        return f"只拿{coin}（照自己的順勢規則）"
    if key == "higher":
        return "(iii) 誰的順勢比例高就拿誰"
    setting, variant = key.split("_")
    return f"({'i' if variant == 'i' else 'ii'}) 比值{SET_NAME[setting]}" + ("，一直滿倉" if variant == "i" else "＋自己的順勢")


# ---------- data ----------

def _fetch(symbol: str, start_ms: int) -> list:
    from dashboard.collectors import price as px

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


def _load() -> tuple[dict, list[str]]:
    """Finished daily closes per symbol from the store, topped up from Binance."""
    try:
        store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
        if not isinstance(store, dict):
            store = {}
    except Exception:  # noqa: BLE001 - a damaged store starts over
        store = {}
    today = pd.Timestamp(common.today_utc())
    out, problems = {}, []
    for symbol in SYMBOLS:
        days = {str(k)[:10]: float(v) for k, v in (store.get(symbol) or {}).items() if common._finite(v) and v > 0}
        if not days or pd.Timestamp(max(days)) < today - pd.Timedelta(days=1):
            try:
                start = int((pd.Timestamp(max(days)) - pd.Timedelta(days=3)).timestamp() * 1000) if days else 0
                for row in _fetch(symbol, start):
                    day = pd.to_datetime(row[0], unit="ms").normalize()
                    if day < today and float(row[4]) > 0:  # never today's unfinished candle
                        days[_d(day)] = float(row[4])
            except Exception as exc:  # noqa: BLE001
                problems.append(f"幣安 {symbol} 日線這次沒有更新成功（{type(exc).__name__}）"
                                + ("，用的是先前存下來的資料。" if days else "，沒有算進去。"))
        if len(days) < 300:
            continue
        store[symbol] = dict(sorted(days.items()))
        s = pd.Series(days, dtype=float)
        s.index = pd.to_datetime(s.index)
        out[symbol] = s.sort_index()
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return out, problems


# ---------- the rules ----------

def _blend(close: pd.Series) -> pd.Series:
    """Share of the 17 trend_ensemble rules that are long at each close (NaN until all have history)."""
    from dashboard.studies2 import trend_ensemble as te

    signal, _ = te._signals(close)
    return signal.mean(axis=1).where(signal.notna().all(axis=1))


def _ratio_states(ratio: pd.Series) -> pd.DataFrame:
    """1 = ratio in an uptrend (hold the altcoin), 0 = not, NaN = not enough history yet."""
    cols = {}
    for n in (20, 50, 100, 200):
        mean = ratio.rolling(n, min_periods=n).mean()
        cols[f"ma{n}"] = (ratio > mean).astype(float).where(mean.notna())
    for key, n_in, n_out in (("bo20", 20, 10), ("bo55", 55, 20)):
        high = ratio.rolling(n_in, min_periods=n_in).max()
        low = ratio.rolling(n_out, min_periods=n_out).min()
        raw = pd.Series(np.where(ratio >= high, 1.0, np.where(ratio <= low, 0.0, np.nan)), index=ratio.index)
        cols[key] = raw.where(high.notna()).ffill().fillna(0.0).where(high.notna())
    return pd.DataFrame(cols)


def _targets(b_btc: pd.Series, b_alt: pd.Series, states: pd.DataFrame) -> dict:
    """{key: (weight in BTC, weight in the altcoin)} decided at each close."""
    out = {"btc": (b_btc, b_btc * 0.0), "alt": (b_alt * 0.0, b_alt), "half": (0.5 * b_btc, 0.5 * b_alt)}
    alt_up = (b_alt > b_btc).astype(float).where(b_alt.notna() & b_btc.notna())
    out["higher"] = ((1 - alt_up) * b_btc, alt_up * b_alt)
    for setting, _ in SETTINGS:
        s = states[setting]
        out[f"{setting}_i"] = (1 - s, s)
        out[f"{setting}_ii"] = ((1 - s) * b_btc, s * b_alt)
    return out


CANDIDATES = [f"{s}_{v}" for s, _ in SETTINGS for v in ("i", "ii")] + ["higher", "half"]


# ---------- simulating ----------

def _sim(w1: np.ndarray, w2: np.ndarray, r1: np.ndarray, r2: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Daily net return and amount traded for weights held during each day (already shifted), reset every close."""
    w1, w2 = np.nan_to_num(w1), np.nan_to_num(w2)
    r1, r2 = np.nan_to_num(r1), np.nan_to_num(r2)
    gross = w1 * r1 + w2 * r2
    base = np.where(np.abs(1 + gross) > 1e-9, 1 + gross, 1.0)
    d1, d2 = w1 * (1 + r1) / base, w2 * (1 + r2) / base
    trade = (np.abs(w1 - np.concatenate([[0.0], d1[:-1]])) + np.abs(w2 - np.concatenate([[0.0], d2[:-1]])))
    return gross - cost * trade, trade


def _stats(net: np.ndarray) -> dict:
    n = len(net)
    if n < 2:
        return {"ret": None, "vol": None, "ratio": None, "deepest": None}
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else 0.0, "deepest": deepest}


def _measure(net: pd.Series, trade: pd.Series) -> dict:
    out = _stats(net.values)
    out["turnover"] = float(trade.sum() / (len(net) / YEAR)) if len(net) else None
    out["years"] = {int(y): float((np.prod(1 + c.values) - 1) * 100) for y, c in net.groupby(net.index.year)}
    out["spans"] = {}
    for name, a, b in SPANS:
        part = net[(net.index >= pd.Timestamp(a)) & (net.index <= (pd.Timestamp(b) if b else net.index[-1]))]
        out["spans"][name] = _stats(part.values) if len(part) > 30 else None
    out["start"] = net.index[0]
    return out


def _rank_corr(a: dict, b: dict) -> float | None:
    keys = [k for k in a if _ok(a[k]) and _ok(b.get(k))]
    if len(keys) < 5:
        return None
    x, y = pd.Series([a[k] for k in keys]).rank(), pd.Series([b[k] for k in keys]).rank()
    if x.std() == 0 or y.std() == 0:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return v if math.isfinite(v) else None


class _Pair:
    """BTC against one altcoin (or the basket): prices, blends, ratio states, targets and runs."""

    def __init__(self, btc: pd.Series, alt: pd.Series, ratio: pd.Series):
        idx = btc.index.intersection(alt.index)
        self.btc, self.alt = btc.reindex(idx), alt.reindex(idx)
        self.ratio = ratio.reindex(idx).ffill()
        self.index = idx
        self.r_btc, self.r_alt = self.btc.pct_change(), self.alt.pct_change()
        self.b_btc, self.b_alt = _blend(self.btc), _blend(self.alt)
        self.states = _ratio_states(self.ratio)
        self.targets = _targets(self.b_btc, self.b_alt, self.states)
        ready = pd.concat([self.targets[k][0] for k in CANDIDATES + ["btc", "alt"]]
                          + [self.targets[k][1] for k in CANDIDATES + ["btc", "alt"]], axis=1).notna().all(axis=1)
        if not ready.any():
            raise RuntimeError("歷史太短")
        self.start = ready[ready].index[0] + pd.Timedelta(days=1)  # first day held

    def run_pos(self, p1: pd.Series, p2: pd.Series, cost: float = COST) -> tuple[pd.Series, pd.Series]:
        rows = self.index >= self.start
        net, trade = _sim(p1[rows].values, p2[rows].values, self.r_btc[rows].values, self.r_alt[rows].values, cost)
        idx = self.index[rows]
        return pd.Series(net, index=idx), pd.Series(trade, index=idx)

    def pos(self, key: str, delay: int = 1) -> tuple[pd.Series, pd.Series]:
        t1, t2 = self.targets[key]
        return t1.shift(delay), t2.shift(delay)

    def run(self, key: str, delay: int = 1, cost: float = COST):
        return self.run_pos(*self.pos(key, delay), cost=cost)

    def chain_pos(self, picks: dict, delay: int = 1) -> tuple[pd.Series, pd.Series]:
        """Each day held under the candidate picked for that day's calendar year (cash before the first pick)."""
        p1, p2 = pd.Series(0.0, index=self.index), pd.Series(0.0, index=self.index)
        for year, key in picks.items():
            rows = self.index.year == year
            a, b = self.pos(key, delay)
            p1[rows], p2[rows] = a[rows], b[rows]
        return p1, p2


def _beat(m: dict, base: dict) -> tuple[int, int]:
    years = sorted(base["years"])
    return sum(1 for y in years if m["years"].get(y, -1e9) > base["years"][y]), len(years)


# ---------- the clocks ----------

def _hourly_closes(path, symbol: str) -> pd.Series | None:
    from dashboard.studies2 import clock_robustness as cr

    rows = cr._read(path)
    if len(rows) < 24 * 400:
        return None
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    if max(rows) < today_ms - 2 * cr.HOUR_MS:
        try:
            cr._fetch(symbol, rows, keep=False)  # in memory only: the store belongs to another study
        except Exception:  # noqa: BLE001 - the stored hours are still usable
            pass
    s = pd.Series(rows, dtype=float)
    s.index = pd.to_datetime(s.index + cr.HOUR_MS, unit="ms")  # time the hour closed
    s = s[s.index <= pd.Timestamp(common.today_utc())].sort_index()
    return s.asfreq("h").ffill()


def _clocks(main_key: str, picks: dict, start: pd.Timestamp) -> dict | None:
    """The main pick, the chained picks and 50/50 on daily bars that close at each of the 24 UTC hours."""
    btc = _hourly_closes(SEASON_HOURLY, "BTCUSDT")
    eth = _hourly_closes(ETH_HOURLY, "ETHUSDT")
    if btc is None or eth is None:
        return None
    idx = btc.index.intersection(eth.index)
    btc, eth = btc.reindex(idx), eth.reindex(idx)
    out = {"main": [], "chain": [], "half": [], "main_beat": 0, "chain_beat": 0}
    for h in range(24):
        sel = idx.hour == h
        b, e = btc[sel], eth[sel]
        b.index = e.index = (b.index - pd.Timedelta(hours=h)).normalize()  # one bar per date
        pair = _Pair(b, e, e / b)
        first = max(pair.start, start)
        res = {}
        for name, (p1, p2) in (("main", pair.pos(main_key)), ("chain", pair.chain_pos(picks)), ("half", pair.pos("half"))):
            net, _ = pair.run_pos(p1, p2)
            res[name] = _stats(net[net.index >= first].values)
            out[name].append(res[name])
        out["main_beat"] += int(res["main"]["ratio"] > res["half"]["ratio"])
        out["chain_beat"] += int(res["chain"]["ratio"] > res["half"]["ratio"])
    out["last"] = idx[-1]
    return out


# ---------- compute ----------

def _basket(closes: dict, index: pd.DatetimeIndex) -> pd.Series:
    """Equal-weight daily-rebalanced index of the basket coins that have at least BASKET_WAIT closes."""
    rets = []
    for c in BASKET:
        s = closes.get(c + "USDT")
        if s is None:
            continue
        s = s.reindex(index).ffill(limit=7)
        alive = s.notna().cumsum() > BASKET_WAIT
        rets.append(s.pct_change().where(alive))
    r = pd.concat(rets, axis=1).mean(axis=1).fillna(0.0)
    return (1 + r).cumprod()


def _compute(closes: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    last = min(closes["BTCUSDT"].index[-1], closes["ETHUSDT"].index[-1], closes["ETHBTC"].index[-1])
    full = pd.date_range(closes["BTCUSDT"].index[0], last, freq="D")

    def daily(sym: str) -> pd.Series:
        s = closes[sym]
        s = s[s.index <= last]
        return s.reindex(pd.date_range(s.index[0], last, freq="D")).ffill()

    btc = daily("BTCUSDT")
    eth = _Pair(btc, daily("ETHUSDT"), daily("ETHBTC"))
    runs = {k: eth.run(k) for k in CANDIDATES + ["btc", "alt"]}
    stats = {k: _measure(*runs[k]) for k in runs}

    # ----- pick in advance -----
    index = runs["half"][0].index
    picks, pick_rows = {}, []
    for year in range(FIRST_PICK_YEAR, int(index[-1].year) + 1):
        before, inside = index < pd.Timestamp(f"{year}-01-01"), index.year == year
        if before.sum() < 180 or not inside.any():
            continue
        prior = {k: _stats(runs[k][0].values[before])["ratio"] for k in CANDIDATES}
        best = max(CANDIDATES, key=lambda k: prior[k])
        during = {k: _stats(runs[k][0].values[inside])["ratio"] for k in CANDIDATES} if inside.sum() >= 30 else {}
        picks[year] = best
        rank = (1 + sum(1 for k in CANDIDATES if during[k] > during[best])) if during else None
        pick_rows.append({"year": year, "key": best, "score": prior[best], "rank": rank, "days": int(inside.sum()),
                          "corr": _rank_corr(prior, during) if during else None})
    chain_parts = eth.chain_pos(picks)
    chain_net, chain_trade = eth.run_pos(*chain_parts)
    since = chain_net.index >= pd.Timestamp(f"{min(picks)}-01-01")
    chain = _measure(chain_net[since], chain_trade[since])
    half_since = _measure(runs["half"][0][since], runs["half"][1][since])
    years_c = {k: _measure(runs[k][0][since], runs[k][1][since])["years"] for k in ("btc", "alt", "half")}
    hind = max(CANDIDATES, key=lambda k: _stats(runs[k][0][since].values)["ratio"])
    hind_stats = _measure(runs[hind][0][since], runs[hind][1][since])
    main_key = picks[max(picks)]

    # ----- take apart -----
    robust = {}
    for c in COSTS:
        robust[("cost", c)] = {name: _measure(*[x[since] for x in r]) for name, r in
                               (("chain", eth.run_pos(*chain_parts, cost=c)), ("main", eth.run(main_key, cost=c)),
                                ("half", eth.run("half", cost=c)))}
    robust[("delay", 2)] = {name: _measure(*[x[since] for x in r]) for name, r in
                            (("chain", eth.run_pos(*eth.chain_pos(picks, delay=2))), ("main", eth.run(main_key, delay=2)),
                             ("half", eth.run("half", delay=2)))}

    # ----- C: basket, D: other coins -----
    others = {}
    basket_px = _basket(closes, full)
    alt_list = [("BASKET", basket_px, basket_px / btc.reindex(full))]
    for c in OTHERS:
        if c + "USDT" in closes and c + "BTC" in closes:
            alt_list.append((c, daily(c + "USDT"), daily(c + "BTC")))
    for name, px_alt, ratio in alt_list:
        try:
            pair = _Pair(btc, px_alt, ratio)
        except Exception:  # noqa: BLE001
            continue
        r = {k: pair.run(k) for k in CANDIDATES}
        st = {k: _measure(*r[k]) for k in CANDIDATES}
        cn, ct = pair.run_pos(*pair.chain_pos(picks))
        keep = cn.index >= max(pair.start, pd.Timestamp(f"{min(picks)}-01-01"))
        cn, ct = cn[keep], ct[keep]
        hn, ht = r["half"]
        half_c = _measure(hn[hn.index >= cn.index[0]], ht[ht.index >= cn.index[0]])
        others[name] = {"pair": pair, "stats": st, "chain": _measure(cn, ct), "half_chain": half_c,
                        "settings_beat": sum(1 for k in CANDIDATES if k != "half" and st[k]["ratio"] > st["half"]["ratio"]),
                        "hind": st[hind], "main": st[main_key]}

    return {"eth": eth, "runs": runs, "stats": stats, "picks": picks, "pick_rows": pick_rows, "chain": chain,
            "half_since": half_since, "years_c": years_c, "hind": hind, "hind_stats": hind_stats, "main_key": main_key,
            "robust": robust, "others": others, "last": last, "since": index[since][0], "chain_pos": chain_parts}


# ---------- the study ----------

def study() -> dict:
    closes, problems = _load()
    for need in ("BTCUSDT", "ETHUSDT", "ETHBTC"):
        if need not in closes:
            raise RuntimeError(f"{need} 日線取不到")
    r = _compute(closes)
    eth, stats, last = r["eth"], r["stats"], r["last"]
    half = stats["half"]
    start = _d(eth.start)
    partial = last < pd.Timestamp(f"{last.year}-12-31")
    clocks = None
    try:
        clocks = _clocks(r["main_key"], r["picks"], r["since"])
    except Exception as exc:  # noqa: BLE001 - the clock check is optional
        problems.append(f"24 個收盤時刻的檢查這次沒有算成（{type(exc).__name__}）。")

    # ----- table 1 -----
    order = ["btc", "alt", "half", "higher"] + [f"{s}_{v}" for s, _ in SETTINGS for v in ("i", "ii")]
    rows1 = []
    for k in order:
        m = stats[k]
        won, n = _beat(m, half)
        spans = "｜".join(_num(m["spans"][name]["ratio"]) if m["spans"][name] else "—" for name, _, _ in SPANS)
        rows1.append([_label(k), _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                      "—" if k == "half" else f"{n} 年裡 {won} 年", f"{m['turnover']:.1f} 次", spans])
    n_years = len(half["years"])
    b_i = [k for k in order if k.endswith("_i") and stats[k]["ratio"] > half["ratio"]]
    b_ii = [k for k in order if k.endswith("_ii") and stats[k]["ratio"] > half["ratio"]]

    # ----- table 2: pick in advance -----
    rows2 = []
    chain, hs, yc = r["chain"], r["half_since"], r["years_c"]
    for p in r["pick_rows"]:
        y = p["year"]
        rows2.append([f"{y}" + ("（未滿一年）" if p["days"] < 360 else ""), _label(p["key"]), _num(p["score"]),
                      f"{_pct(chain['years'].get(y))}（{len(CANDIDATES)} 種裡第 {p['rank']}）" if p["rank"] else _pct(chain["years"].get(y)),
                      _pct(yc["half"].get(y)), _pct(yc["btc"].get(y)), _pct(yc["alt"].get(y)), _num(p["corr"])])
    corrs = [p["corr"] for p in r["pick_rows"] if _ok(p["corr"])]
    won_c, n_c = _beat(chain, hs)
    rows2.append(["串起來", f"{n_c} 年裡 {won_c} 年贏過 50/50", "—",
                  f"{_pct(chain['ret'])}｜{_num(chain['ratio'])}｜{_pct(chain['deepest'])}",
                  f"{_pct(hs['ret'])}｜{_num(hs['ratio'])}｜{_pct(hs['deepest'])}", "—", "—",
                  _num(float(np.mean(corrs))) if corrs else "—"])
    hind = r["hind_stats"]
    rows2.append([f"對照：事後看最好的", _label(r["hind"]), "—", f"{_pct(hind['ret'])}｜{_num(hind['ratio'])}｜{_pct(hind['deepest'])}",
                  "—", "—", "—", "—"])

    # ----- table 3: other coins and the basket -----
    rows3 = []
    e_chain_line = [f"以太幣（{start} 起，挑選用的那一組）", f"{_pct(half['ret'])}｜{_num(half['ratio'])}｜{_pct(half['deepest'])}",
                    f"{_pct(hs['ret'])}｜{_num(hs['ratio'])}", f"{_pct(chain['ret'])}｜{_num(chain['ratio'])}｜{n_c} 年裡 {won_c} 年",
                    f"{_pct(stats[r['main_key']]['ret'])}｜{_num(stats[r['main_key']]['ratio'])}",
                    f"{_pct(stats[r['hind']]['ret'])}｜{_num(stats[r['hind']]['ratio'])}",
                    f"{len(CANDIDATES) - 1} 種裡 {len(b_i) + len(b_ii) + int(stats['higher']['ratio'] > half['ratio'])} 種"]
    rows3.append(e_chain_line)
    oos = {}
    for name in ("SOL", "BNB", "XRP", "BASKET"):
        o = r["others"].get(name)
        if not o:
            continue
        h, c, hc = o["stats"]["half"], o["chain"], o["half_chain"]
        won, n = _beat(c, hc)
        oos[name] = (c["ratio"] > hc["ratio"], won, n, o["settings_beat"], c, hc)
        what = f"{NAME_ZH[name]}／比特幣" if name != "BASKET" else "大型山寨幣籃子／比特幣"
        rows3.append([f"{what}（{_d(o['pair'].start)} 起）", f"{_pct(h['ret'])}｜{_num(h['ratio'])}｜{_pct(h['deepest'])}",
                      f"{_pct(hc['ret'])}｜{_num(hc['ratio'])}", f"{_pct(c['ret'])}｜{_num(c['ratio'])}｜{n} 年裡 {won} 年",
                      f"{_pct(o['main']['ret'])}｜{_num(o['main']['ratio'])}", f"{_pct(o['hind']['ret'])}｜{_num(o['hind']['ratio'])}",
                      f"{len(CANDIDATES) - 1} 種裡 {o['settings_beat']} 種"])

    # ----- table 4: take apart -----
    rows4 = []
    for c in COSTS:
        x = r["robust"][("cost", c)]
        rows4.append([f"每次買賣成本 {c * 100:.2f}%", f"{_pct(x['chain']['ret'])}｜{_num(x['chain']['ratio'])}｜{_pct(x['chain']['deepest'])}",
                      f"{_pct(x['main']['ret'])}｜{_num(x['main']['ratio'])}", f"{_pct(x['half']['ret'])}｜{_num(x['half']['ratio'])}",
                      f"{x['chain']['turnover']:.1f}｜{x['main']['turnover']:.1f}｜{x['half']['turnover']:.1f} 次"])
    x = r["robust"][("delay", 2)]
    rows4.append(["晚一天才照做（成本 0.10%）", f"{_pct(x['chain']['ret'])}｜{_num(x['chain']['ratio'])}｜{_pct(x['chain']['deepest'])}",
                  f"{_pct(x['main']['ret'])}｜{_num(x['main']['ratio'])}", f"{_pct(x['half']['ret'])}｜{_num(x['half']['ratio'])}", "—"])
    if clocks:
        def tri(items, field, fmt):
            v = [it[field] for it in items]
            return f"{fmt(min(v))}／{fmt(float(np.median(v)))}／{fmt(max(v))}"
        rows4.append(["24 個收盤時刻：每年報酬 最低／中位／最高", tri(clocks["chain"], "ret", _pct), tri(clocks["main"], "ret", _pct),
                      tri(clocks["half"], "ret", _pct), "—"])
        rows4.append(["24 個收盤時刻：報酬÷波動 最低／中位／最高", tri(clocks["chain"], "ratio", _num), tri(clocks["main"], "ratio", _num),
                      tri(clocks["half"], "ratio", _num),
                      f"報酬÷波動贏過同一時刻 50/50：串起來 24 個裡 {clocks['chain_beat']} 個、今年這組 24 個裡 {clocks['main_beat']} 個"])
        rows4.append(["24 個收盤時刻：最深虧損 最深／中位／最淺", tri(clocks["chain"], "deepest", _pct),
                      tri(clocks["main"], "deepest", _pct), tri(clocks["half"], "deepest", _pct), "—"])

    since = _d(r["since"])
    tables = [
        {"caption": (f"總表（{start} 到 {_d(last)}，比特幣和以太幣的 17 條規則都有值之後才開始算）。每天收盤決定、隔天照做，"
                     f"每次買賣扣交易金額的 {COST * 100:.2f}%，現金沒有利息。「各自順勢 50/50」是比較的基準：一半的錢照比特幣自己的 17 條規則混合持有、"
                     "另一半照以太幣自己的。(i) 比值在上升趨勢就全部拿以太幣，否則全部拿比特幣；(ii) 選出來的那顆只拿它自己規則說的比例，其餘現金；"
                     f"(iii) 不看比值，兩顆裡 17 條規則看多比例較高的那顆、拿那個比例。年數欄是 {n_years} 個日曆年"
                     f"{'（頭尾兩年不完整）' if partial else ''}裡報酬比 50/50 高的年數。最後一欄是三段期間各自的報酬÷波動。"),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "贏過 50/50 的年數", "一年換手",
                     "報酬÷波動：" + "｜".join(name for name, _, _ in SPANS)],
         "rows": rows1},
        {"caption": (f"事先挑：每年 1 月 1 日只用那天以前的資料，從 {len(CANDIDATES)} 種做法（6 種比值規則 × (i)(ii)、(iii)、各自順勢 50/50）"
                     "裡挑報酬÷波動最高的一種，整年照做，隔年重挑，再接起來（換做法那天的買賣也扣成本）。"
                     "「串起來」那列是「每年報酬｜報酬÷波動｜最深虧」。最後一欄是等級相關：各種做法「事前的分數」和「那一年的分數」"
                     "各自排名後有多一致（+1 完全一致、0 無關、負的是事前好的那年反而差），「串起來」那格是各年平均。"),
         "columns": ["年", "事前挑到的", "挑它時的報酬÷波動", "那一年的報酬（當年排名）", "各自順勢 50/50", "只拿比特幣（順勢）",
                     "只拿以太幣（順勢）", "等級相關"],
         "rows": rows2[:16]},
        {"caption": (f"換成其他幣：把在以太幣／比特幣上每年事先挑到的同一種做法，原封不動用在索拉納、幣安幣、瑞波幣對比特幣的比值上，"
                     "以及「大型山寨幣籃子」（以太幣、幣安幣、瑞波幣、艾達幣、索拉納等權重，各幣上架滿 30 天才加入）對比特幣。"
                     f"每個組合從兩邊的 17 條規則都有值那天起算，所以起算日不同、彼此的數字不能直接比大小。「同期 50/50」是和「事先挑串起來」"
                     f"同一段期間（{FIRST_PICK_YEAR} 年起或更晚）的基準。「今年這組」＝{last.year} 年事先挑到的「{_label(r['main_key'])}」；"
                     f"「事後最好」＝在以太幣上事後看最好的「{_label(r['hind'])}」。最後一欄是 {len(CANDIDATES) - 1} 種做法裡整段報酬÷波動高於 50/50 的有幾種。"),
         "columns": ["組合（起算日）", "各自順勢 50/50：每年報酬｜報酬÷波動｜最深虧", "同期 50/50：每年報酬｜報酬÷波動",
                     "照以太幣事先挑的串起來：報酬｜報酬÷波動｜贏 50/50 的年數", "今年這組（整段）", "以太幣事後最好那組（整段）",
                     "贏過 50/50 的做法"],
         "rows": rows3},
        {"caption": (f"拆開看（以太幣／比特幣，{since} 起）：換成本、晚一天照做、換收盤時刻。「今年這組」＝「{_label(r['main_key'])}」。"
                     "24 個收盤時刻是用 1 小時線把日線重新拼 24 次（一天結束在每一個整點），每個時刻各自用自己的收盤算訊號、自己的日線算損益，"
                     "比值用以太幣美元價 ÷ 比特幣美元價算出來；每年挑到的做法和日線版相同、不重挑。最後一欄是一年換手（串起來｜今年這組｜50/50）。"),
         "columns": ["改了什麼", "事先挑串起來：報酬｜報酬÷波動｜最深虧", "今年這組：報酬｜報酬÷波動", "各自順勢 50/50：報酬｜報酬÷波動", "補充"],
         "rows": rows4},
    ]

    # ----- now -----
    st_now = eth.states.iloc[-1]
    bb, ba = float(eth.b_btc.iloc[-1]), float(eth.b_alt.iloc[-1])

    def hold(w1: float, w2: float) -> str:
        parts = []
        if w1 > 0.0005:
            parts.append(f"比特幣 {w1 * 100:.0f}%")
        if w2 > 0.0005:
            parts.append(f"以太幣 {w2 * 100:.0f}%")
        cash = 1 - w1 - w2
        if cash > 0.0005:
            parts.append(f"現金 {cash * 100:.0f}%")
        return "、".join(parts) if parts else "全部現金"

    def want(key: str) -> str:
        t1, t2 = eth.targets[key]
        return hold(float(t1.iloc[-1]), float(t2.iloc[-1]))

    now = [
        {"label": f"以太幣／比特幣比值（{_d(last)} 收盤，一顆以太幣值多少比特幣）", "value": f"{float(eth.ratio.iloc[-1]):.5f}"},
        {"label": "比值的趨勢狀態（上升＝規則選以太幣）",
         "value": "；".join(f"{SET_NAME[s]}：{'上升' if st_now[s] == 1 else '不是上升'}" for s, _ in SETTINGS)},
        {"label": "17 條順勢規則看多的比例", "value": f"比特幣 {bb * 100:.0f}%；以太幣 {ba * 100:.0f}%"},
        {"label": "各自順勢 50/50 照規則下一天的狀態", "value": want("half")},
        {"label": "(i) 一直滿倉，照規則下一天", "value": "；".join(f"{SET_NAME[s]}：{want(f'{s}_i')}" for s, _ in SETTINGS)},
        {"label": "(ii) 加上自己的順勢，照規則下一天", "value": "；".join(f"{SET_NAME[s]}：{want(f'{s}_ii')}" for s, _ in SETTINGS)},
        {"label": "(iii) 誰的順勢比例高就拿誰，照規則下一天", "value": want("higher")},
        {"label": f"{last.year} 年事先挑到的「{_label(r['main_key'])}」", "value": want(r["main_key"]) + "（只是規則的狀態，不是建議）"},
    ]

    # ----- verdict -----
    best_any = max((k for k in order if k not in ("btc", "alt", "half")), key=lambda k: stats[k]["ratio"])
    n_rot = len(order) - 4
    s1 = (f"{start[:7]} 起，比特幣、以太幣各一半各照自己的順勢規則持有是每年 {_pct(half['ret'])}、報酬÷波動 {_num(half['ratio'])}、"
          f"最深虧 {_pct(half['deepest'])}；照比值趨勢換著拿的 {n_rot} 種做法裡，報酬÷波動贏過它的有 {len(b_i) + len(b_ii)} 種"
          f"（最好的是{_label(best_any).split(') ', 1)[-1]}，{_num(stats[best_any]['ratio'])}）。")
    oth = [n for n in ("SOL", "BNB", "XRP", "BASKET") if n in oos]
    s2 = (f"每年事先挑、{min(r['picks'])} 年起串起來是 {_pct(chain['ret'])}、{_num(chain['ratio'])}（同期 50/50 {_pct(hs['ret'])}、"
          f"{_num(hs['ratio'])}，{n_c} 年裡 {won_c} 年贏）；同樣挑法換到{'、'.join(NAME_ZH[n] if n != 'BASKET' else '山寨幣籃子' for n in oth)}，"
          f"報酬÷波動贏過各自 50/50 的有 {len(oth)} 組裡 {sum(1 for n in oth if oos[n][0])} 組。")
    if clocks:
        steady = ("看不出穩定的好處" if clocks["chain_beat"] < 18 else "在多數時刻都站得住")
        s3 = (f"換 24 個收盤時刻，事先挑串起來的報酬÷波動在 24 個裡 {clocks['chain_beat']} 個贏過同時刻的 50/50"
              f"（只看今年挑到的那組是 {clocks['main_beat']} 個），{steady}。")
    else:
        s3 = "收盤時刻的檢查這次沒有算成。"
    verdict = s1 + s2 + s3

    notes = [
        "比值規則：以太幣／比特幣比值（幣安 ETHBTC 現貨收盤）高於 N 日均線（N = 20、50、100、200），或創 20 日新高後算上升、跌破 10 日新低後不算"
        "（55／20 同理），上升時選以太幣、否則選比特幣。「17 條順勢規則」和「不挑參數，把很多順勢規則混在一起」那一頁完全相同，看多的占幾成就持有幾成。",
        "這些設定（6 種比值規則、三種用法）是今天才寫下來的，雖然每年的挑選只看過去，候選名單本身還是帶著事後的知識；事後最好的那一組數字一定偏好看。",
        f"年數很少：主要期間只有 {n_years} 個日曆年（頭尾兩年不完整），而且比特幣和以太幣大多一起漲跌，這些年不是彼此獨立的；"
        "「幾年裡幾年」多一年少一年就差很多，做法之間的小差距不能當成確定的結論。",
        "其他幣（索拉納、幣安幣、瑞波幣、艾達幣）是今天還排在前面的大幣，有倖存者偏誤：當年很大、後來衰退的幣沒有放進來，所以「換成其他幣」和山寨幣籃子的結果都偏好看。"
        "籃子每天調回等權重，籃子內部的調整沒有另外算成本。",
        f"成本只用粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%（第四張表有 0.30%、0.50%）。從以太幣換到比特幣等於賣一次再買一次，扣兩倍。"
        "持有比例不是 0 或 100% 的做法每天收盤調回目標比例，這些小額調整也算成本。現金沒有利息，也沒有算稅。",
        "所有訊號只用當天收盤為止的資料，隔天才持有；只用已經收完的日線。損益用幣安 BTCUSDT、ETHUSDT 等美元價，比值訊號用幣安直接交易的 ETHBTC、SOLBTC、"
        "BNBBTC、XRPBTC 收盤；籃子的比值是籃子指數 ÷ 比特幣美元價。缺資料的日子沿用前一天的價格。",
        "24 個收盤時刻的檢查用的是其他頁面存下的 1 小時線（比特幣、以太幣），比值用兩者的美元價相除；每個時刻的損益用它自己那一刻到下一天同一刻計算，"
        "所以各時刻的結果是在略為不同的切法上量的，主要用來看同一時刻下比值規則和 50/50 誰高。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "Blockchain Center〈Altcoin Season Index〉：前 50 大幣過去 90 天有 75% 跑贏比特幣就叫「山寨季」（已開啟）",
         "url": "https://www.blockchaincenter.net/en/altcoin-season-index/"},
        {"name": "CoinMarketCap〈Altcoin Season Index〉：同樣用 90 天跑贏比特幣的比例分「比特幣季／山寨季」（已開啟）",
         "url": "https://coinmarketcap.com/charts/altcoin-season-index/"},
        {"name": "幣安現貨日線（BTCUSDT、ETHUSDT、ETHBTC 等）", "url": "https://www.binance.com/zh-TC/trade/ETH_BTC"},
        {"name": "幣安公開行情 API（K 線）",
         "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
    ]
    notes.insert(0, "這個想法的來源：上面兩個網站把市場分成「比特幣季」和「山寨季」，常被拿來決定該拿比特幣還是山寨幣；兩個網站本身只公布指數、"
                    "沒有給買賣規則，這裡測的「照比值趨勢換著拿」是把這個想法寫成固定規則來檢驗，不是那兩個網站的做法。")

    return {
        "id": "pair_rotation",
        "title": "照以太幣／比特幣比值的趨勢換著拿",
        "question": "照「一顆以太幣值多少比特幣」這個比值的趨勢決定拿哪一顆（比值上升拿以太幣、否則拿比特幣），有沒有比兩顆各自照自己的順勢規則持有更好？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
