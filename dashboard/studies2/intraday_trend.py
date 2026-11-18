"""Historical study "intraday_trend": does trend-following on faster bars (1-hour, 4-hour, 12-hour) beat the daily
rules on bitcoin after realistic costs?

Background: published or public claims that faster trend works better in crypto - "AdaptiveTrend" (arXiv
2602.11708, 6-hour bars), the 5- and 10-day lookbacks inside Zarattini et al.'s Donchian ensemble, the habit of
trading off 4-hour charts. The dashboard's daily trend rules passed its checks; pure intraday seasonality did not
survive costs. Trend on intraday bars had not been tested.

Rules (long or cash, or a fraction for the blend), each on 1-, 4- and 12-hour bars:
- N-bar-high entry / M-bar-low exit, N = 20, 55, 120, 330 bars, M = N x 20/55, plus the version whose lookback in
  time equals the daily 55/20 (55 days of bars);
- close above the N-bar average, N = 50, 100, 300;
- the 17-rule blend of trend_ensemble with its lookbacks in bars scaled so they cover the same calendar time as the
  daily blend, and unscaled (the same bar counts, so faster).
Benchmarks: daily 55/20, the daily blend, buy-and-hold - all built from the same hourly grid (daily = 24-hour bars
closing at UTC midnight), so every rule is measured on the same UTC days.

Engine (the arithmetic of clock_robustness._Market.run, generalised to any bar length): the target decided at a
bar's close is held from that moment for one bar; fractional positions are put back to target at every bar close;
the cost is charged on the amount traded; equity is followed hour by hour and read at every UTC midnight.

1. Everything over the common period (first day on which every rule on every bar length has a value).
2. Choose in advance: each year from 2019 the intraday setting with the best return / volatility on all earlier
   data (0.10% cost) is used for that year; the years are chained; compared with always using daily 55/20; rank
   correlation between the earlier and the in-year score.
3. The setting chosen for the latest year, unchanged, on ETH, SOL, BNB and XRP.
4. Taking apart: 4-hour bars starting at 00/01/02/03 UTC, one bar later, costs, break-even cost, trades per year,
   average win / loss.

Data: Binance spot 1-hour candles. The stores kept by seasonality.py (bitcoin) and clock_robustness.py (the other
coins) are read only; missing recent hours are fetched into memory and never written. Only closed candles and
finished UTC days, only public key-less endpoints through the dashboard's own fetch helpers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import trend_ensemble as te

COST = 0.001  # 0.10% of the amount traded
COSTS = (0.0, 0.0005, 0.001, 0.002)
YEAR = 365
WARM = 201  # days before every rule (longest: 200-day average in bars) has a value
FIRST_PICK = 2019
BARS = (1, 4, 12)
BREAK_N = (20, 55, 120, 330)
MA_N = (50, 100, 300)
COINS = ("ETH", "SOL", "BNB", "XRP")
MIN_DAYS = 2 * YEAR
BE_MAX = 0.02  # break-even costs are looked for up to 2% per trade


# ---------- formatting ----------

_pct, _num, _d = cr._pct, cr._num, cr._d


def _days_text(x: float) -> str:
    text = f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{text} 天"


def _cost_text(c) -> str:
    if c is None:
        return "—"
    return f"{c * 100:.3f}".rstrip("0").rstrip(".") + "%"


# ---------- data ----------

def _hourly(symbol: str) -> tuple[dict[int, float], str | None]:
    """Closed 1-hour closes from the stores of seasonality.py / clock_robustness.py, topped up in memory only."""
    path = cr.SHARED_STORE if symbol == "BTCUSDT" else cr._own_store(symbol)
    rows = cr._read(path)
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - cr.HOUR_MS:
        try:
            cr._fetch(symbol, rows, keep=False)  # never writes a file
        except Exception as exc:  # noqa: BLE001 - the stored history is still usable
            problem = (f"{symbol[:-4]} 的幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    return rows, problem


# ---------- the rules ----------

def _breakout(s: pd.Series, n_in: int, n_out: int) -> np.ndarray:
    return cr._breakout(s, n_in, n_out)


def _above(s: pd.Series, n: int) -> np.ndarray:
    mean = s.rolling(n, min_periods=n).mean()
    return (s > mean).astype(float).where(mean.notna()).fillna(0.0).values


def _blend(s: pd.Series, f: int) -> np.ndarray:
    """trend_ensemble's 17 rules with every lookback multiplied by f bars (f = 1 is exactly te._signals)."""
    cols = []
    for n in te.MA_DAYS:
        mean = s.rolling(n * f, min_periods=n * f).mean()
        cols.append((s > mean).astype(float).where(mean.notna()))
    for n in te.CHANNEL:
        a, b = n * f, (n // 2) * f
        high = s.rolling(a, min_periods=a).max()
        low = s.rolling(b, min_periods=b).min()
        state = pd.Series(np.where(s >= high, 1.0, np.where(s <= low, 0.0, np.nan)), index=s.index)
        cols.append(state.where(high.notna()).ffill().fillna(0.0).where(high.notna()))
    for n in te.PAST:
        past = s / s.shift(n * f) - 1
        cols.append((past > 0).astype(float).where(past.notna()))
    sig = pd.concat(cols, axis=1)
    return sig.mean(axis=1).where(sig.notna().all(axis=1)).fillna(0.0).values


def _specs() -> list[dict]:
    """Every intraday candidate (bar length B hours, start offset 0)."""
    out = []
    for b in BARS:
        per_day = 24 // b
        ns = list(BREAK_N)
        if 55 * per_day not in ns:
            ns.append(55 * per_day)
        for n in ns:
            m = int(round(n * 20 / 55))
            same = n == 55 * per_day
            out.append({"key": f"b{b}_br{n}", "bar": b, "kind": "br", "n": n, "m": m, "binary": True,
                        "label": f"{b} 小時線：{n} 根新高進、{m} 根新低出（約 {_days_text(n * b / 24)}"
                                 + ("，和日線 55／20 同天數）" if same else "）")})
        for n in MA_N:
            out.append({"key": f"b{b}_ma{n}", "bar": b, "kind": "ma", "n": n, "binary": True,
                        "label": f"{b} 小時線：收盤高於 {n} 根均線（約 {_days_text(n * b / 24)}）"})
        out.append({"key": f"b{b}_blendT", "bar": b, "kind": "blend", "f": per_day, "binary": False,
                    "label": f"{b} 小時線：17 條混合（天數和日線版相同）"})
        out.append({"key": f"b{b}_blendN", "bar": b, "kind": "blend", "f": 1, "binary": False,
                    "label": f"{b} 小時線：17 條混合（根數和日線版相同，快 {per_day} 倍）"})
    return out


BENCH = (
    {"key": "d5520", "bar": 24, "kind": "br", "n": 55, "m": 20, "binary": True, "label": "日線：55 日新高進、20 日新低出"},
    {"key": "dblend", "bar": 24, "kind": "blend", "f": 1, "binary": False, "label": "日線：17 條混合"},
)


def _want(price: np.ndarray, spec: dict, offset: int = 0) -> np.ndarray:
    """Target after every close of the bar series that starts `offset` hours after the first UTC midnight."""
    s = pd.Series(np.asarray(price[offset::spec["bar"]], dtype=float))
    if spec["kind"] == "br":
        return _breakout(s, spec["n"], spec["m"])
    if spec["kind"] == "ma":
        return _above(s, spec["n"])
    return _blend(s, spec["f"])


# ---------- the engine ----------

def _run(price: np.ndarray, bar: int, offset: int, want: np.ndarray, cost: float = COST) -> tuple[np.ndarray, np.ndarray]:
    """Net return of every UTC day and amount traded at every bar close. want[j] is decided at the close at hour
    offset + bar*j and held for the next `bar` hours (cash before the first close)."""
    k = len(price)
    nb = -(-(k - 1 - offset) // bar)
    pad = np.concatenate([price, np.full(bar, price[-1])])  # flat after the end: an unfinished last bar
    seg = pad[offset:offset + bar * nb + 1]
    path = seg[1:].reshape(nb, bar) / seg[:-1:bar][:, None] - 1
    w = np.asarray(want, dtype=float)[:nb]
    r = path[:, -1]
    base = 1 + w * r
    drift = w * (1 + r) / np.where(np.abs(base) > 1e-9, base, 1.0)
    trade = np.abs(w - np.concatenate([[0.0], drift[:-1]]))  # starts from cash
    factor = 1 + w[:, None] * path - cost * trade[:, None]
    before = np.concatenate([[1.0], np.cumprod(factor[:, -1])[:-1]])
    equity = np.ones(offset + bar * nb + 1)
    equity[offset + 1:] = (before[:, None] * factor).ravel()
    mid = equity[:k:24]
    return mid[1:] / mid[:-1] - 1, trade


def _later(want: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.asarray(want, dtype=float)[:-1]])


def _first_bar(day: int, bar: int, offset: int) -> int:
    return max(0, -(-(24 * day - offset) // bar))


def _trades(price: np.ndarray, spec: dict, offset: int, want: np.ndarray, first_day: int, cost: float) -> dict:
    """Round trips of an all-or-nothing rule entered from first_day on: count, share won, average win / loss."""
    s = np.asarray(price[offset::spec["bar"]], dtype=float)
    held = np.asarray(want[:len(s)]) > 0.5
    start = _first_bar(first_day, spec["bar"], offset)
    rets = []
    j = start
    while j < len(s):
        if held[j]:  # a position already open at the start is counted from the start
            k = j + 1
            while k < len(s) and held[k]:
                k += 1
            if k < len(s):
                rets.append(s[k] / s[j] * (1 - cost) ** 2 - 1)
            j = k
        j += 1
    rets = np.array(rets)
    if not len(rets):
        return {"n": 0, "won": None, "win": None, "loss": None}
    wins, losses = rets[rets > 0], rets[rets <= 0]
    return {"n": len(rets), "won": float(len(wins) / len(rets) * 100),
            "win": float(wins.mean() * 100) if len(wins) else None,
            "loss": float(losses.mean() * 100) if len(losses) else None}


# ---------- one market ----------

class _Market:
    def __init__(self, price: np.ndarray, b0: pd.Timestamp):
        self.price, self.b0 = price, b0
        self.days = (len(price) - 1) // 24
        self.dates = pd.date_range(b0, periods=self.days, freq="D")
        start = max(int((pd.Timestamp("2018-01-01") - b0).days), 0)
        self.w0 = max(start, WARM)
        if self.days - self.w0 < MIN_DAYS:
            raise RuntimeError("1 小時線涵蓋的期間不夠")
        self.years = (self.days - self.w0) / YEAR
        bound = price[::24]
        self.change = bound[1:] / bound[:-1] - 1
        self._wants: dict = {}

    def want(self, spec: dict, offset: int = 0) -> np.ndarray:
        key = (spec["key"], offset)
        if key not in self._wants:
            self._wants[key] = _want(self.price, spec, offset)
        return self._wants[key]

    def net(self, spec: dict, offset: int = 0, cost: float = COST, delay: bool = False) -> tuple[np.ndarray, np.ndarray]:
        w = self.want(spec, offset)
        return _run(self.price, spec["bar"], offset, _later(w) if delay else w, cost)

    def stats(self, net: np.ndarray, trade: np.ndarray | None = None, bar: int = 24, offset: int = 0) -> dict:
        out = cr._stats(net[self.w0:])
        if trade is not None:
            out["turnover"] = float(trade[_first_bar(self.w0, bar, offset):].sum() / self.years)
        return out

    def measure(self, spec: dict, offset: int = 0, cost: float = COST, delay: bool = False) -> dict:
        net, trade = self.net(spec, offset, cost, delay)
        return self.stats(net, trade, spec["bar"], offset)

    def hold(self) -> dict:
        return cr._stats(self.change[self.w0:])

    def at(self, day) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(day)))


def _break_even(m: _Market, spec: dict, target) -> float | None:
    """Cost per trade at which the rule's yearly return falls to target(cost) (target may itself depend on cost).
    0.0 = behind even with no cost; None = still ahead at BE_MAX."""
    def gap(c: float) -> float:
        return m.measure(spec, cost=c)["ret"] - target(c)

    if gap(0.0) <= 0:
        return 0.0
    if gap(BE_MAX) > 0:
        return None
    lo, hi = 0.0, BE_MAX
    for _ in range(22):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _spearman(a: list, b: list) -> float | None:
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if len(a) < 5 or np.std(ra) == 0 or np.std(rb) == 0:
        return None
    v = float(np.corrcoef(ra, rb)[0, 1])
    return v if math.isfinite(v) else None


def _year_returns(net: pd.Series) -> dict:
    return {int(y): float((np.prod(1 + p.values) - 1) * 100) for y, p in net.groupby(net.index.year)}


# ---------- the computation (kept apart from the formatting so look-ahead can be tested) ----------

def _pick(m: _Market, specs: list[dict]) -> dict:
    """Each year from FIRST_PICK: the candidate with the best return / volatility on all earlier measured days."""
    nets = {s["key"]: m.net(s)[0] for s in specs}
    years = list(range(FIRST_PICK, m.dates[-1].year + 1))
    picks, prior, rho = {}, {}, {}
    for y in years:
        a, c = m.at(f"{y}-01-01"), min(m.at(f"{y + 1}-01-01"), m.days)
        prior[y] = {k: cr._stats(v[m.w0:a])["ratio"] for k, v in nets.items()}
        picks[y] = max(prior[y], key=lambda k: prior[y][k])
        if c - a >= 30:
            inside = {k: cr._stats(v[a:c])["ratio"] for k, v in nets.items()}
            rho[y] = _spearman([prior[y][k] for k in nets], [inside[k] for k in nets])
        else:
            rho[y] = None
    return {"nets": nets, "years": years, "picks": picks, "prior": prior, "rho": rho}


def _compute(price: np.ndarray, b0: pd.Timestamp) -> dict:
    m = _Market(price, b0)
    specs = _specs()
    by_key = {s["key"]: s for s in specs + list(BENCH)}

    # ----- 1. everything over the common period -----
    main = {}
    for s in specs + list(BENCH):
        x = m.measure(s)
        x["costs"] = {c: m.measure(s, cost=c)["ret"] for c in COSTS if c != COST}
        main[s["key"]] = x
    hold = m.hold()

    # ----- 2. choose in advance -----
    wf = _pick(m, specs)
    years = wf["years"]
    first = m.at(f"{years[0]}-01-01")
    idx = m.dates[first:]
    pieces = []
    for y in years:
        a, c = m.at(f"{y}-01-01"), min(m.at(f"{y + 1}-01-01"), m.days)
        pieces.append(wf["nets"][wf["picks"][y]][a:c])
    chain = pd.Series(np.concatenate(pieces), index=idx)
    d5520_net = m.net(BENCH[0])[0]
    dblend_net = m.net(BENCH[1])[0]
    series = {"pick": chain, "d5520": pd.Series(d5520_net[first:], index=idx),
              "dblend": pd.Series(dblend_net[first:], index=idx), "hold": pd.Series(m.change[first:], index=idx)}
    chains = {k: dict(cr._stats(v.values), years=_year_returns(v)) for k, v in series.items()}
    chains["pick"]["beat"] = sum(1 for y in years if chains["pick"]["years"][y] > chains["d5520"]["years"][y])
    chains["pick"]["beat_hold"] = sum(1 for y in years if chains["pick"]["years"][y] > chains["hold"]["years"][y])
    chains["d5520"]["beat_hold"] = sum(1 for y in years if chains["d5520"]["years"][y] > chains["hold"]["years"][y])
    d_prior = {y: cr._stats(d5520_net[m.w0:m.at(f"{y}-01-01")])["ratio"] for y in years}
    chosen = by_key[wf["picks"][years[-1]]]

    # ----- 4. taking apart -----
    four = [s for s in specs if s["bar"] == 4]
    probe = [chosen] + [by_key[k] for k in ("b4_br330", "b4_br120", "b4_br55", "b4_br20", "b4_ma300", "b4_blendT")
                        if k != chosen["key"]]
    apart = {}
    for s in probe[:6] + [BENCH[0]]:
        x = {"offsets": [m.measure(s, offset=o)["ret"] for o in range(4)] if s["bar"] == 4 else None,
             "delay": m.measure(s, delay=True)}
        if s["binary"]:
            x["trades"] = _trades(price, s, 0, m.want(s), m.w0, COST)
        x["be_daily"] = (None if s["key"] == "d5520" else
                         _break_even(m, s, lambda c: m.measure(BENCH[0], cost=c)["ret"]))
        x["be_hold"] = _break_even(m, s, lambda c: hold["ret"])
        apart[s["key"]] = x
    offsets4 = {s["key"]: [m.measure(s, offset=o)["ratio"] for o in range(4)] for s in four}
    beat_offsets = {k: sum(1 for v in vals if v > main["d5520"]["ratio"]) for k, vals in offsets4.items()}

    now = {k: float(m.want(by_key[k])[-1]) for k in (chosen["key"], "b4_br330", "d5520", "dblend")}
    return {"m": m, "specs": specs, "by_key": by_key, "main": main, "hold": hold, "wf": wf, "chains": chains,
            "d_prior": d_prior, "chosen": chosen, "apart": apart, "beat_offsets": beat_offsets, "now": now,
            "last": m.dates[-1], "last_price": float(price[-1])}


def _coin(symbol: str, keys: list[dict]) -> tuple[dict | None, str | None]:
    rows, problem = _hourly(symbol)
    try:
        if len(rows) < 24 * (WARM + MIN_DAYS):
            raise RuntimeError("太短")
        price, b0, _ = cr._grid(rows)
        m = _Market(price, b0)
        out = {"start": m.dates[m.w0], "last": m.dates[-1], "hold": m.hold()}
        for s in keys:
            out[s["key"]] = m.measure(s)
        return out, problem
    except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
        return None, (problem or "") + f"{symbol[:-4]} 的 1 小時線這次拿不到或太短（{type(exc).__name__}），沒有算進去。"


# ---------- the study ----------

def study() -> dict:
    rows, problem = _hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, missing = cr._grid(rows)
    r = _compute(price, b0)
    m, main, hold, wf, ch = r["m"], r["main"], r["hold"], r["wf"], r["chains"]
    by_key, chosen, apart = r["by_key"], r["chosen"], r["apart"]
    last = r["last"]
    start = _d(m.dates[m.w0])
    span = f"{start} 到 {_d(last)}"
    years = wf["years"]
    n_years = len(years)
    partial = last < pd.Timestamp(f"{years[-1]}-12-31")
    d = main["d5520"]

    # ----- other coins -----
    coin_keys = [chosen, by_key["b4_br330"], BENCH[0]] if chosen["key"] != "b4_br330" else [chosen, BENCH[0]]
    coins, problems = {}, [problem] if problem else []
    for coin in COINS:
        got, prob = _coin(f"{coin}USDT", coin_keys)
        if got:
            coins[coin] = got
        if prob:
            problems.append(prob)

    # ----- table 1: everything over the common period -----
    def turn(s: dict, x: dict) -> str:
        return f"{x['turnover'] / 2:.0f} 次來回" if s["binary"] else f"換手 {x['turnover']:.0f} 次"

    def be_text(c, beaten_free: str = "不扣成本也輸") -> str:
        if c is None:
            return f"到 {_cost_text(BE_MAX)} 都還贏"
        return beaten_free if c == 0.0 else _cost_text(c)

    intraday = r["specs"]
    best1 = max((s for s in intraday if s["bar"] == 1), key=lambda s: main[s["key"]]["ratio"])
    best12 = max((s for s in intraday if s["bar"] == 12), key=lambda s: main[s["key"]]["ratio"])
    show = list(BENCH) + [s for s in intraday if s["bar"] == 4]
    extra = [best1, best12, by_key["b1_br1320"], by_key["b12_br110"]]
    for s in extra:
        if s not in show:
            show.append(s)
    rows1 = [["一直抱著不動（不扣成本）", _pct(hold["ret"]), _num(hold["ratio"]), _pct(hold["deepest"]), "—", "—", "—", "—"]]
    for s in show[:15]:
        x = main[s["key"]]
        tag = ""
        if s is best1 or s is best12:
            tag = "（事後看 " + ("1" if s is best1 else "12") + " 小時線裡最好的）"
        be = "—" if s["key"] == "d5520" else be_text(
            _break_even(m, s, lambda c: m.measure(BENCH[0], cost=c)["ret"]))
        rows1.append([s["label"] + tag, _pct(x["ret"]), _num(x["ratio"]), _pct(x["deepest"]), turn(s, x),
                      _pct(x["costs"][0.0005]), _pct(x["costs"][0.002]), be])
    n_cand = len(intraday)
    n_better = sum(1 for s in intraday if main[s["key"]]["ratio"] > d["ratio"])
    n_better_ret = sum(1 for s in intraday if main[s["key"]]["ret"] > d["ret"])

    # ----- table 2: choose in advance -----
    def year_name(y: int) -> str:
        return f"{y}（到 {last.strftime('%m-%d')}）" if y == years[-1] and partial else str(y)

    rows2 = []
    for y in years:
        k = wf["picks"][y]
        rho = wf["rho"][y]
        rows2.append([year_name(y), by_key[k]["label"], f"{_num(wf['prior'][y][k])}｜{_num(r['d_prior'][y])}",
                      _pct(ch["pick"]["years"][y]), _pct(ch["d5520"]["years"][y]), _pct(ch["hold"]["years"][y]),
                      _num(rho) if rho is not None else "—"])
    rows2.append([f"{years[0]} 起串起來", "每年事先挑", "—",
                  f"{_pct(ch['pick']['ret'])}｜{_num(ch['pick']['ratio'])}｜{_pct(ch['pick']['deepest'])}",
                  f"{_pct(ch['d5520']['ret'])}｜{_num(ch['d5520']['ratio'])}｜{_pct(ch['d5520']['deepest'])}",
                  f"{_pct(ch['hold']['ret'])}｜{_num(ch['hold']['ratio'])}｜{_pct(ch['hold']['deepest'])}",
                  f"中位數 {_num(float(np.median([v for v in wf['rho'].values() if v is not None])))}"])
    rows2.append(["贏的年數", "—", "—",
                  f"贏日線 55／20：{n_years} 年裡 {ch['pick']['beat']} 年；贏一直抱著：{n_years} 年裡 {ch['pick']['beat_hold']} 年",
                  f"贏一直抱著：{n_years} 年裡 {ch['d5520']['beat_hold']} 年", "—", "—"])
    picked = sorted({wf["picks"][y] for y in years})

    # ----- table 3: other coins -----
    def cell(x: dict) -> str:
        return f"{_pct(x['ret'])}｜{_num(x['ratio'])}｜{_pct(x['deepest'])}"

    bt = {"start": m.dates[m.w0], "hold": hold, **{s["key"]: main[s["key"]] for s in coin_keys}}
    rows3 = []
    for name, c in [("比特幣", bt)] + list(coins.items()):
        row = [f"{name}（{_d(c['start'])} 起）", cell(c["hold"]), cell(c["d5520"]), cell(c[chosen["key"]])]
        if len(coin_keys) == 3:
            row.append(cell(c["b4_br330"]))
        rows3.append(row)
    cols3 = ["幣（起算日）", "一直抱著（不扣成本）", "日線 55／20", f"選出的設定：{chosen['label']}"]
    if len(coin_keys) == 3:
        cols3.append("4 小時線 330／120 根（和日線 55／20 同天數）")
    coin_beat = sum(1 for c in coins.values() if c[chosen["key"]]["ratio"] > c["d5520"]["ratio"])
    coin_beat_ret = sum(1 for c in coins.values() if c[chosen["key"]]["ret"] > c["d5520"]["ret"])

    # ----- table 4: taking apart -----
    rows4 = []
    for key, x in apart.items():
        s = by_key[key]
        offs = "／".join(_pct(v) for v in x["offsets"]) if x["offsets"] else "—（日線，24 個收盤時刻見「換一個收盤時刻」那一頁）"
        dl = x["delay"]
        base = main[key]
        t = x.get("trades")
        if t and t["n"]:
            tr = f"{t['n'] / m.years:.1f} 次｜{_num(t['won'], 0)}%"
            wl = f"{_pct(t['win'])}｜{_pct(t['loss'])}"
        else:
            tr, wl = f"換手 {base['turnover']:.0f} 次｜—", "—"
        be = ("—" if key == "d5520" else be_text(x["be_daily"])) + "｜" + be_text(x["be_hold"])
        rows4.append([s["label"] + ("（選出的設定）" if key == chosen["key"] else ""), offs,
                      f"{_pct(dl['ret'])}（{'少' if dl['ret'] < base['ret'] else '多'} {abs(dl['ret'] - base['ret']):.1f} 點）",
                      tr, wl, be])

    tables = [
        {"caption": (f"比特幣，{span}（所有規則都已經有值的第一天起），每次買賣扣交易金額的 {COST * 100:.2f}%。"
                     "決定都在一根 K 線收盤時做、從那一刻起持有到下一根收盤；損益一律在每天 UTC 0 點讀一次，所以不同週期可以直接比。"
                     "「一年交易」：全進全出的規則是一年幾次來回，17 條混合是一年換手幾次（買賣總額等於全部資金一次算 1 次）。"
                     "最後一欄是手續費要低於多少，這條規則的每年報酬才贏得過同樣成本下的日線 55／20。"
                     f"表裡列出 4 小時線全部 {sum(1 for s in intraday if s['bar'] == 4)} 種，加上 1 小時、12 小時線裡和日線同天數的版本"
                     f"與事後最好的一種；全部 {n_cand} 種較快設定裡，報酬÷波動贏過日線 55／20 的有 {n_better} 種。"),
         "columns": ["做法", "每年報酬（0.10%）", "報酬÷波動", "最深虧損", "一年交易", "成本 0.05% 時每年報酬",
                     "成本 0.20% 時每年報酬", "手續費低於多少才贏日線 55／20"],
         "rows": rows1},
        {"caption": (f"事先挑、之後再驗（比特幣）：每年 1 月 1 日，只用之前的資料（{start} 起），在 {n_cand} 種較快設定裡挑報酬÷波動最高的"
                     "一種（成本 0.10%），那一年就照它做，一年一年串起來；和一直用日線 55／20、一直抱著比。第三欄是挑的時候"
                     "「選出設定｜日線 55／20」在之前資料上的報酬÷波動；最後一欄是所有設定「之前的成績」和「當年的成績」排名的相關"
                     "（1 = 排名完全一樣，0 = 沒關係，負的 = 相反）。換年換設定時沒有另外算成本。"
                     "串起來那一列是「每年報酬｜報酬÷波動｜最深虧損」。"),
         "columns": ["年度", "事先挑到的設定", "挑的時候報酬÷波動：選出｜日線 55／20", "選出的設定這一年", "日線 55／20 這一年",
                     "一直抱著這一年", "排名相關"],
         "rows": rows2},
        {"caption": (f"換資產：把 {years[-1]} 年初挑出的設定原封不動用在其他幣（幣安現貨 1 小時線，成本 0.10%）。"
                     f"每種幣從上架滿 {WARM} 天（而且不早於 2018 年）才開始算，起算日不同，彼此的數字不能直接比大小。"
                     "格子是「每年報酬｜報酬÷波動｜最深虧損」。"),
         "columns": cols3,
         "rows": rows3},
        {"caption": (f"拆開來看（比特幣，{span}，成本 0.10%）。第二欄：4 小時線的起點分別放在 UTC 0、1、2、3 點（台灣早上 8、9、10、11 點）"
                     "時的每年報酬；第三欄：收盤看到訊號、晚一根 K 線才照做；第四、五欄只算全進全出的規則：一年平均幾次來回、"
                     "賺錢的比例，以及賺錢那幾次平均賺多少、賠錢那幾次平均賠多少（已扣兩邊手續費）；最後一欄：手續費要低於多少才贏"
                     "日線 55／20（兩邊扣一樣的成本）｜才贏一直抱著。"),
         "columns": ["做法", "4 個起點的每年報酬（UTC 0／1／2／3 點）", "晚一根才照做", "一年幾次來回｜賺錢比例",
                     "平均賺｜平均賠", "手續費低於多少才贏：日線 55／20｜一直抱著"],
         "rows": rows4},
    ]

    # ----- today -----
    def state(v: float) -> str:
        if v > 0.9995:
            return "持有"
        if v < 0.0005:
            return "現金"
        return f"持有 {v * 100:.0f}%"

    now = [
        {"label": f"最後一個收完的日子（{_d(last)}，UTC 0 點收）", "value": f"比特幣 {r['last_price']:,.0f} 美元"},
        {"label": f"{years[-1]} 年初事先挑出的較快設定：{chosen['label']}", "value": state(r["now"][chosen["key"]])},
        {"label": "4 小時線 330／120 根（和日線 55／20 同天數）", "value": state(r["now"]["b4_br330"])},
        {"label": "日線 55 日新高進、20 日新低出", "value": state(r["now"]["d5520"])},
        {"label": "日線 17 條混合", "value": state(r["now"]["dblend"])},
        {"label": "這幾格的意思", "value": "只是描述規則在最後一根收完的 K 線上停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    c4 = main["b4_br330"]
    ac = apart[chosen["key"]]
    if ac["offsets"]:
        off_text = f"4 個起點每年 {_pct(min(ac['offsets']))} 到 {_pct(max(ac['offsets']))}"
    else:
        off_text = f"晚一根才做每年 {_pct(ac['delay']['ret'])}"
    be_c = ac["be_daily"]
    be_part = ("不扣成本也輸日線 55／20" if be_c == 0.0 else
               f"手續費低於 {_cost_text(be_c)} 才贏日線 55／20" if be_c is not None else f"手續費到 {_cost_text(BE_MAX)} 仍贏日線 55／20")
    verdict = (
        f"比特幣 {start[:7]} 起、每次扣 0.10%：日線 55／20 每年 {_pct(d['ret'])}、報酬÷波動 {_num(d['ratio'])}，"
        f"4 小時線同天數版 {_pct(c4['ret'])}、{_num(c4['ratio'])}，{n_cand} 種較快設定只有 {n_better} 種報酬÷波動較高。"
        f"每年事先挑較快設定、{years[0]} 起串起來每年 {_pct(ch['pick']['ret'])}，日線 55／20 {_pct(ch['d5520']['ret'])}，"
        f"{n_years} 年裡 {ch['pick']['beat']} 年贏；最新挑出的（{chosen['label']}）{be_part}，{off_text}。"
        f"原封不動用到其他幣，{len(coins)} 個裡 {coin_beat} 個報酬÷波動贏過各自的日線 55／20。"
    )

    notes = [
        "比較的問題：較快的順勢規則（1 小時、4 小時、12 小時線）扣掉實際成本後，有沒有贏過儀表板已驗證的日線規則。"
        "日線 55／20 和日線 17 條混合也用同一份 1 小時線拼出來（UTC 0 點收盤），所以和「不挑參數」「換一個收盤時刻」那幾頁"
        "用的日線版起算日不同，數字會有些微差異。",
        f"規則：N 根新高進場、M 根新低出場，N = {'、'.join(map(str, BREAK_N))} 根，M = N × 20 ÷ 55（四捨五入），再加上和日線 55／20 "
        "同天數的版本（1 小時線 1,320／480 根、4 小時線 330／120 根、12 小時線 110／40 根）；收盤高於 N 根均線，N = "
        f"{'、'.join(map(str, MA_N))} 根；17 條混合＝「不挑參數」那一頁的 17 條規則，一種把天數換成同樣時間的根數（4 小時線 × 6），"
        "一種直接用同樣的根數（所以快好幾倍）。每條規則只做多或持有現金（混合是持有一部分），沒有做空、沒有槓桿。",
        "成本：幣安現貨一般帳戶的手續費是成交金額的 0.10%（掛單、吃單同價，用 BNB 付約 0.075%）；0.05% 大約要較高的會員等級或"
        "其他交易所的掛單優惠才拿得到，而且掛單不保證成交；0.20% 是把買賣價差、滑價也粗略算進去的保守數字。"
        "所以 0.10% 是一般人吃單的實際水準，表裡主要用它。成交價一律用 K 線收盤價，滑價沒有另外模擬，交易越頻繁的規則這部分低估越多。",
        f"1 小時線從 2017-08 才有，最長的規則（200 天的均線換成根數）要等約 {WARM} 天才有值，所以全部從 {start} 才開始算，"
        "比其他頁的 2018-01-01 晚幾個月；事先挑的第一年（2019）只有不到一年的資料可以挑。"
        f"比特幣的 1 小時線整段缺 {missing:,} 個小時（幣安早年停機維護），缺的小時沿用前一個小時的價格。",
        f"試了很多種設定（{n_cand} 種較快設定、4 個起點、4 種成本）：表一「事後最好的」那幾列是從很多種裡挑出來的，一定比事先能做到的好看；"
        f"公平的比較是表二每年事先挑的結果。事先挑過的設定有 {len(picked)} 種："
        + "、".join(by_key[k]["label"] for k in picked) + "。",
        "AdaptiveTrend（arXiv 2602.11708，6 小時線、報告 2022–2024 年報酬÷波動（夏普值）2.41）沒有重做：摘要只說是 6 小時線的順勢、"
        "依當天波動調整的移動停損、每月依過去報酬÷波動從 150 多個幣挑選、多空 70／30 分配，沒有寫進出規則、回看天數、停損算法和挑選門檻，"
        "照摘要無法重現，而且它是多空、跨很多幣的組合，和這裡只做多比特幣不是同一件事；它的樣本外設計摘要也沒有說清楚。",
        f"年數很少：事先挑只有 {n_years} 個年度{'（最後一年還沒過完）' if partial else ''}，大致就是兩三輪大漲大跌，各年不是彼此獨立的，"
        "「幾年裡幾年」多一年少一年就會差很多。其他幣的名單是今天還在的大幣，有倖存者偏誤。",
        "現金假設沒有利息，也沒有算稅；一直抱著沒有扣成本。只用一家交易所（幣安現貨）的價格。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += problems

    return {
        "id": "intraday_trend",
        "title": "改用 4 小時線做順勢，會比日線好嗎？",
        "question": "把順勢規則改到 4 小時線（以及 1 小時、12 小時線）上做，扣掉實際的手續費之後，有沒有贏過日線的 55 日新高／20 日新低等規則？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安公開行情 API（K 線）",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "幣安現貨手續費", "url": "https://www.binance.com/zh-TC/fee/trading"},
            {"name": "AdaptiveTrend（arXiv 2602.11708，只讀了摘要頁）", "url": "https://arxiv.org/abs/2602.11708"},
            {"name": "Zarattini 等〈Catching Crypto Trends〉", "url": "https://concretumgroup.com/wp-content/uploads/2026/02/Catching-Crypto-Trends.pdf"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
