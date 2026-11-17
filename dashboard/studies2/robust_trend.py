"""Historical study "robust_trend": after removing clock luck and adding real-life sloppiness, which bitcoin trend
approach - if any - still clearly beats holding, and by how much?

1. The "ordinary person" stress tests trend_practical ran for the 20/10 breakout, repeated for six approaches and
   averaged over the 24 possible daily-close hours (clocks): acting 10 hours after the check (the shape of "check at
   22:00 Taiwan time, act at 08:00"), acting days late, checking every 2 / 3 / 7 days, higher costs, missed entries
   (random, fixed seed) and all of it combined.
2. A fair pick-in-advance: every candidate (walkforward's families A-D, the 17-rule blend, the blend x 40%
   volatility target, the nine-lookback Donchian ensemble) is scored on earlier data by the AVERAGE over the 24
   clocks of return / volatility after costs; the best one is applied to the next year split over the 24 clocks.
3. Is the edge fading: two-year blocks and rolling two-year windows since 2012, before / after the spot ETF launch,
   and a monthly bootstrap of the return edge.
4. The same comparison when idle cash earns the 3-month US Treasury bill rate (FRED DTB3).

Conventions are those of clock_robustness: a signal decided at a clock's close is the position held from that moment
for 24 hours, costs are charged on the amount traded (fractional positions are put back to target at every close),
profit and loss is always read at UTC midnight so every clock and buy-and-hold are measured on the same days.

Data: Binance BTCUSDT 1-hour candles (the store seasonality.py keeps is read through clock_robustness and only topped
up in memory - this module writes no file), the dashboard's long daily series for warm-up and for the years before
2018 (single clock), FRED DTB3. Only closed candles, finished UTC days, public key-less endpoints, fixed user agent.
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
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import trend_practical as tp
from dashboard.studies2 import walkforward as wf

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
TREASURY = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
            "{year}/all?type=daily_treasury_bill_rates&field_tdr_date_value={year}&page&_format=csv")
COST = 0.001
COSTS = (0.0025, 0.005, 0.01)
YEAR = 365
SIMS = 1000
DRAWS = 2000
SEED = 20180101
WAIT = 10  # hours between the check and the trade: 22:00 -> 08:00 Taiwan time
LATE = (1, 2, 3)
EVERY = (2, 3)
MISS = (0.10, 0.25)
RISE = 0.10  # a day on which the wanted share rises by more than this is an "entry"
PERSON_COST, PERSON_MISS = 0.0025, 0.10
PERSON_CLOCK = 14  # 22:00 in Taiwan
FIRST_PICK = 2019
EARLY_START = "2012-01-01"
ETF = "2024-01-11"
WINDOW = 2 * YEAR
RULES = ("c20", "c55", "ma50", "ma100", "blend", "vol40")
SHORT = cr.SHORT
FAMILIES = ("A", "B", "C", "D")

_pct, _pp, _num, _d = cr._pct, cr._pp, cr._num, cr._d


def _span(values, fmt) -> str:
    lo, hi = fmt(float(min(values))), fmt(float(max(values)))
    return lo if lo == hi else f"{lo} ～ {hi}"


def _two(x: dict) -> str:
    return f"{_pct(x['ret'])}｜{_pct(x['deepest'])}"


# ---------- data ----------

def _tbill() -> tuple[pd.Series | None, str, str | None]:
    """3-month US Treasury bill rate (bank discount, % a year, business days), where it came from, and a problem
    text. FRED's DTB3 first; FRED often does not answer this project's fixed user agent, and then the same figure is
    read from the US Treasury's own daily bill-rate tables (one file per year). Neither: None (0% interest)."""
    first = None
    try:
        reply = requests.get(FRED, headers=common.HEADERS, timeout=(5, 8))
        reply.raise_for_status()
        rows = {}
        for line in reply.text.splitlines()[1:]:
            day, _, value = line.partition(",")
            try:
                rows[pd.Timestamp(day)] = float(value)
            except ValueError:  # "." on holidays
                continue
        series = _rates(rows)
        return series, "fred", None
    except Exception as exc:  # noqa: BLE001 - try the Treasury's own tables
        first = type(exc).__name__
    try:
        rows = {}
        for year in range(int(cr.MAIN_START[:4]), pd.Timestamp(common.today_utc()).year + 1):
            reply = requests.get(TREASURY.format(year=year), headers=common.HEADERS, timeout=(5, 20))
            reply.raise_for_status()
            table = pd.read_csv(io.StringIO(reply.text))
            for day, value in zip(pd.to_datetime(table["Date"], format="%m/%d/%Y"), table["13 WEEKS BANK DISCOUNT"]):
                rows[day] = float(value)
            time.sleep(common.PAUSE)
        return _rates(rows), "treasury", (f"FRED 的 DTB3 這次沒有回應（{first}），現金利息改用美國財政部自己公布的每日國庫券利率表"
                                           "（13 週、貼現率，和 DTB3 是同一個數字）。")
    except Exception as exc:  # noqa: BLE001 - the study still runs with 0% interest
        return None, "none", (f"美國三個月國庫券利率這次拿不到（FRED：{first}；財政部：{type(exc).__name__}），"
                              "現金利息一律用 0% 計算，「現金有利息」的欄位因此和沒有利息相同。")


def _rates(rows: dict) -> pd.Series:
    series = pd.Series(rows, dtype=float).sort_index()
    series = series[np.isfinite(series) & (series > -2) & (series < 25)]
    if len(series) < 1000 or series.index[-1] < pd.Timestamp("2024-01-01"):
        raise RuntimeError("too short")
    return series


def _load() -> dict:
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, missing = cr._grid(rows)
    daily = te._daily_close()
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    rate, rate_source, rate_problem = _tbill()
    return {"price": price, "b0": b0, "missing": missing, "daily": daily, "pre": pre, "rate": rate, "rate_source": rate_source,
            "problems": [p for p in (problem, rate_problem) if p]}


# ---------- engine ----------

class _Engine:
    """Positions on an execution clock -> net returns of UTC days (the same arithmetic as clock_robustness.run, but
    reading the equity only at UTC midnights so that a thousand simulated paths can be run at once)."""

    def __init__(self, m: cr._Market):
        self.m, self.days, self.w0 = m, m.days, m.w0
        days = m.days
        self.at = [m.pad[b:b + 24 * days:24] for b in range(24)]  # price at boundary b + 24j
        self.r = [m.pad[b + 24:b + 24 * days + 24:24] / self.at[b] - 1 for b in range(24)]  # the 24 hours held
        self.part = [m.price[24::24][:days] / self.at[b] - 1 for b in range(24)]  # ... up to the next UTC midnight

    def _mid(self, b: int, w: np.ndarray, cost: float, rf: np.ndarray | None) -> np.ndarray:
        """Equity at the UTC midnights 1..days (row j = midnight j + 1). w is (days,) or (days, sims)."""
        r, part = self.r[b], self.part[b]
        if w.ndim == 2:
            r, part = r[:, None], part[:, None]
        base = 1 + w * r
        prev = np.zeros_like(w)
        prev[1:] = (w * (1 + r) / base)[:-1]  # what yesterday's position drifted to; starts from cash
        paid = cost * np.abs(w - prev)
        full, partial = base - paid, 1 + w * part - paid
        if rf is not None:
            idle = (1 - w) * (rf if w.ndim == 1 else rf[:, None])
            full, partial = full + idle, partial + idle * ((24 - b) / 24)
        mid = np.cumprod(full, axis=0)
        mid[1:] = mid[:-1] * partial[1:]
        mid[0] = partial[0]
        return mid

    def run(self, b: int, w: np.ndarray, cost: float = COST, rf: np.ndarray | None = None) -> np.ndarray:
        """Net return of every UTC day for one position path on execution clock b."""
        mid = self._mid(b, np.asarray(w, dtype=float)[:self.days], cost, rf)
        return np.concatenate([[mid[0] - 1], mid[1:] / mid[:-1] - 1])

    def paths(self, b: int, w: np.ndarray, cost: float, rf: np.ndarray | None = None) -> np.ndarray:
        """Equity of many paths from the first measured midnight on (first row = 1)."""
        mid = self._mid(b, w, cost, rf)[self.w0 - 1:]
        return mid / mid[0]

    def held(self, b: int, target: np.ndarray, check: np.ndarray) -> np.ndarray:
        """The target is only looked at (and traded to) where check is True; in between the position just drifts."""
        n = self.days
        last = np.maximum.accumulate(np.where(check[:n], np.arange(n), -1))
        seen = np.maximum(last, 0)
        w = np.where(last >= 0, np.asarray(target, dtype=float)[:n][seen], 0.0)
        ratio = self.at[b] / self.at[b][seen]
        return w * ratio / (1 - w + w * ratio)


def _path_stats(equity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yearly return % and deepest loss % of every column of an equity matrix that starts at 1."""
    n = len(equity) - 1
    ret = (equity[-1] ** (YEAR / n) - 1) * 100
    deepest = (equity / np.maximum.accumulate(equity, axis=0) - 1).min(axis=0) * 100
    return ret, deepest


def _stagger(nets: list[np.ndarray], start: int, stop: int | None = None) -> np.ndarray:
    """Money split equally over the given clocks at day `start`, never moved between them: the daily net return."""
    equity = np.array([np.concatenate([[1.0], np.cumprod(1 + x[start:stop])]) for x in nets]).mean(axis=0)
    return equity[1:] / equity[:-1] - 1


def _shift(w: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.zeros(d), w[:-d]]) if d else w


def _missed(w: np.ndarray, p: float, rng: np.random.Generator, sims: int) -> np.ndarray:
    """(days, sims) positions of people who, with probability p, do not follow a day on which the wanted share
    rises by more than RISE. What was not bought stays not bought until the wanted share has come down by as much
    (for an all-or-nothing rule: the whole trade is sat out). Draws belong to days, so the past never depends on
    the future."""
    step = np.diff(w, prepend=0.0)
    skip = (step > RISE)[:, None] & (rng.random((len(w), sims)) < p)
    move = np.where(skip, step[:, None], 0.0) + np.minimum(step, 0.0)[:, None]
    total = np.cumsum(move, axis=0)
    gap = total - np.minimum(np.minimum.accumulate(total, axis=0), 0.0)  # the part of the target not held
    return np.clip(w[:, None] - gap, 0.0, 1.0)


def _later(h: int, w: np.ndarray, days: int) -> tuple[int, np.ndarray]:
    """Decided at clock h, traded WAIT hours later: the execution clock and the position on it."""
    return (h + WAIT) % 24, (w[:days] if h + WAIT < 24 else _shift(w[:days], 1))


def _ratio(net: np.ndarray) -> np.ndarray:
    """Return / volatility (as walkforward scores it) along the last axis."""
    n = net.shape[-1]
    if n < 2:
        return np.zeros(net.shape[:-1])
    growth = np.prod(1 + net, axis=-1)
    ret = np.where(growth > 0, np.abs(growth) ** (YEAR / n) - 1, -1.0) * 100
    vol = np.std(net, axis=-1, ddof=1) * math.sqrt(YEAR) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)


def _year_stats(net: pd.Series) -> dict:
    out = {}
    for y, part in net.groupby(net.index.year):
        equity = np.concatenate([[1.0], np.cumprod(1 + part.values)])
        out[int(y)] = (float((equity[-1] - 1) * 100), float((equity / np.maximum.accumulate(equity) - 1).min() * 100))
    return out


# ---------- the computation (kept apart from the formatting so look-ahead can be tested) ----------

def _compute(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, daily: pd.Series, rate: pd.Series | None,
             sims: int = SIMS) -> dict:
    m = cr._Market(price, b0, pre)
    eng = _Engine(m)
    days, w0, dates = m.days, m.w0, m.dates
    main = dates[w0:]
    hold = cr._stats(m.change[w0:])
    if rate is not None:
        rf = (rate.reindex(rate.index.union(dates)).ffill().reindex(dates).fillna(0.0).values / 100 / YEAR)
    else:
        rf = np.zeros(days)
    npre = len(pre)

    # ----- what every rule wants on every clock -----
    want = {k: [] for k in RULES + ("don",)}
    cand_sig, series, specs = [], [], None
    for h in range(24):
        closes = np.concatenate([pre, price[h::24]])
        series.append(closes)
        got = cr._wants(closes, RULES + ("don",))
        for k in want:
            want[k].append(got[k][npre:])
        signal, sp = wf._signals(pd.Series(closes), None, None)
        specs = [s for s in sp if s["family"] in FAMILIES]
        cand_sig.append(signal[[s["key"] for s in specs]].fillna(0.0).values[npre:])

    def avg(nets: list[np.ndarray]) -> dict:
        st = [cr._stats(x[w0:]) for x in nets]
        return {"ret": float(np.mean([s["ret"] for s in st])), "deepest": float(np.mean([s["deepest"] for s in st]))}

    j = np.arange(-npre, days + 1)  # main-day number of every close of a clock's series (pre + hourly)

    def masked(k: str, h: int, mask_main) -> np.ndarray:
        """The position of rule k on clock h when the chart is only looked at where the mask is True."""
        if k in ("c20", "c55"):
            n_in, n_out = (20, 10) if k == "c20" else (55, 20)
            closes = series[h]
            return tp._rule(closes, check=mask_main(j[:len(closes)]), n_in=n_in, n_out=n_out)[npre:][:days]
        return eng.held(h, want[k][h], mask_main(np.arange(days)))

    # ----- 1. ordinary-person stress tests -----
    stress, text_nets, stag, stag_rf = {}, {}, {}, {}
    for ki, k in enumerate(RULES):
        w = [np.asarray(want[k][h][:days], dtype=float) for h in range(24)]
        nets = [eng.run(h, w[h]) for h in range(24)]
        text_nets[k] = nets
        stag[k] = _stagger(nets, w0)
        stag_rf[k] = _stagger([eng.run(h, w[h], COST, rf) for h in range(24)], w0)
        s = {"text": avg(nets), "clock0": cr._stats(nets[0][w0:]), "stag": cr._stats(stag[k]),
             "stag_rf": cr._stats(stag_rf[k]), "share": float(np.mean([x[w0:].mean() for x in w]) * 100)}
        s["wait"] = avg([eng.run(*_later(h, w[h], days)) for h in range(24)])
        s["late"] = {d: avg([eng.run(h, _shift(w[h], d)) for h in range(24)]) for d in LATE}
        s["every"] = {n: [avg([eng.run(h, masked(k, h, lambda x, n=n, p=p: (x - p) % n == 0)) for h in range(24)])
                          for p in range(n)] for n in EVERY}
        wd0 = b0.weekday()
        s["weekly"] = [avg([eng.run(h, masked(k, h, lambda x, h=h, d=d: (wd0 + x + (h + cr.TW) // 24) % 7 == d))
                            for h in range(24)]) for d in range(7)]
        s["cost"] = {c: avg([eng.run(h, w[h], c) for h in range(24)]) for c in COSTS}

        # missed entries alone: 1,000 simulations spread evenly over the 24 clocks
        s["miss"] = {}
        for pi, p in enumerate(MISS):
            rets, deeps = [], []
            for h in range(24):
                n = sims // 24 + (1 if h < sims % 24 else 0)
                rng = np.random.default_rng(SEED + 10_000 * ki + 1_000 * pi + h)
                ret, deep = _path_stats(eng.paths(h, _missed(w[h], p, rng, n), COST))
                rets.append(ret)
                deeps.append(deep)
            rets, deeps = np.concatenate(rets), np.concatenate(deeps)
            s["miss"][p] = {"ret": float(np.median(rets)), "deepest": float(np.median(deeps)),
                            "below": int((rets < hold["ret"]).sum()), "sims": len(rets)}

        # everything combined, 1,000 simulations on each clock; the same simulations also make the 24-part version
        rets, deeps, total, total_rf = [], [], 0.0, 0.0
        for h in range(24):
            b, wl = _later(h, w[h], days)
            rng = np.random.default_rng(SEED + 10_000 * ki + 5_000 + h)
            pos = _missed(wl, PERSON_MISS, rng, sims)
            equity = eng.paths(b, pos, PERSON_COST)
            ret, deep = _path_stats(equity)
            rets.append(ret)
            deeps.append(deep)
            total = total + equity
            total_rf = total_rf + eng.paths(b, pos, PERSON_COST, rf)
        pooled_r, pooled_d = np.concatenate(rets), np.concatenate(deeps)
        s["person"] = {"ret": float(np.median(pooled_r)), "ret_low": float(np.percentile(pooled_r, 10)),
                       "deepest": float(np.median(pooled_d)), "deepest_low": float(np.percentile(pooled_d, 10)),
                       "below": float((pooled_r < hold["ret"]).mean() * 1000),
                       "deeper": float((pooled_d < hold["deepest"]).mean() * 1000),
                       "literal": float(np.median(rets[PERSON_CLOCK])),
                       "literal_below": int((rets[PERSON_CLOCK] < hold["ret"]).sum())}
        for name, eq in (("person_stag", total / 24), ("person_stag_rf", total_rf / 24)):
            ret, deep = _path_stats(eq)
            s[name] = {"ret": float(np.median(ret)), "ret_low": float(np.percentile(ret, 10)),
                       "deepest": float(np.median(deep)), "deepest_low": float(np.percentile(deep, 10)),
                       "below": int((ret < hold["ret"]).sum()), "deeper": int((deep < hold["deepest"]).sum())}
        stress[k] = s

    # ----- 2. fair pick-in-advance -----
    keys = [sp["key"] for sp in specs] + ["blend", "vol40", "don"]
    label = {sp["key"]: sp["label"] for sp in specs}
    label.update(blend=SHORT["blend"], vol40=SHORT["vol40"], don=SHORT["don"])
    nets = {}
    for i, sp in enumerate(specs):
        nets[sp["key"]] = np.array([eng.run(h, cand_sig[h][:days, i]) for h in range(24)])
    nets["blend"], nets["vol40"] = np.array(text_nets["blend"]), np.array(text_nets["vol40"])
    nets["don"] = np.array([eng.run(h, want["don"][h][:days]) for h in range(24)])
    extra = {"c55": np.array(text_nets["c55"])}

    def at(day) -> int:
        return int(dates.searchsorted(pd.Timestamp(day)))

    years = list(range(FIRST_PICK, dates[-1].year + 1))
    picks, before, rho = {}, {}, {}
    for y in years:
        a, c = at(f"{y}-01-01"), min(at(f"{y + 1}-01-01"), days)
        before[y] = {k: float(_ratio(nets[k][:, :a]).mean()) for k in keys}
        picks[y] = max(keys, key=lambda k: before[y][k])
        if c - a >= 30:
            inside = {k: float(_ratio(nets[k][:, a:c]).mean()) for k in keys}
            rho[y] = wf._spearman([before[y][k] for k in keys], [inside[k] for k in keys])
        else:
            rho[y] = None

    def chain(choice) -> dict:
        """choice(year) -> array (24, days) of nets; each year starts with the money split equally over 24 clocks."""
        parts = []
        for y in years:
            a, c = at(f"{y}-01-01"), min(at(f"{y + 1}-01-01"), days)
            parts.append(_stagger(list(choice(y)), a, c))
        net = pd.Series(np.concatenate(parts), index=dates[at(f"{years[0]}-01-01"):])
        out = cr._stats(net.values)
        out["years"] = _year_stats(net)
        return out

    hold_chain = chain(lambda y: m.change[None, :])
    chains = {"pick": chain(lambda y: nets[picks[y]]), "blend": chain(lambda y: nets["blend"]),
              "vol40": chain(lambda y: nets["vol40"]), "c55": chain(lambda y: extra["c55"]),
              "ma50": chain(lambda y: nets["A50"])}
    for x in chains.values():
        x["beat"] = sum(1 for y in years if x["years"][y][0] > hold_chain["years"][y][0])
        x["shallow"] = sum(1 for y in years if x["years"][y][1] > hold_chain["years"][y][1] + 1e-9)

    # ----- 3. is the edge fading -----
    change = daily.pct_change().values
    target = te._targets(daily)[0]
    early_pos = {"blend": target["blend"].shift(1).values,
                 "c55": pd.Series(cr._breakout(daily, 55, 20), index=daily.index).shift(1).values}
    keep = (daily.index >= pd.Timestamp(EARLY_START)) & (daily.index < main[0])
    full = {"hold": pd.Series(np.concatenate([change[keep], m.change[w0:]]), index=daily.index[keep].append(main))}
    for k in ("blend", "c55"):
        early = te._run(early_pos[k], change, COST)[0][keep]
        full[k] = pd.Series(np.concatenate([early, stag[k]]), index=full["hold"].index)
    idx = full["hold"].index

    def cut(series: pd.Series, start, end=None) -> dict | None:
        part = series[(series.index >= pd.Timestamp(start)) & (series.index <= (pd.Timestamp(end) if end else idx[-1]))]
        return cr._stats(part.values) if len(part) > 60 else None

    first_year = idx[0].year
    blocks = []
    for y in range(first_year, main[0].year + 6, 2):
        if pd.Timestamp(f"{y + 1}-12-31") <= idx[-1]:
            blocks.append((f"{y}–{y + 1}", f"{y}-01-01", f"{y + 1}-12-31", y + 1 < main[0].year))
    blocks = [(n, {k: cut(full[k], a, b) for k in full}, early) for n, a, b, early in blocks]
    latest = {k: cr._stats(full[k].values[-WINDOW:]) for k in full}
    etf_day = pd.Timestamp(ETF)
    etf = {"before": {k: cut(full[k], main[0], etf_day - pd.Timedelta(days=1)) for k in full},
           "after": {k: cut(full[k], etf_day) for k in full}}
    after_n = int((idx >= etf_day).sum())

    def rolling(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        logs = np.concatenate([[0.0], np.cumsum(np.log1p(series.values))])
        view = np.lib.stride_tricks.sliding_window_view(logs, WINDOW + 1)
        ret = (np.exp((view[:, -1] - view[:, 0]) * YEAR / WINDOW) - 1) * 100
        deep = (np.exp((view - np.maximum.accumulate(view, axis=1)).min(axis=1)) - 1) * 100
        return ret, deep

    roll = {k: rolling(full[k]) for k in full}
    ends = idx[WINDOW - 1:]  # last day of every window
    starts = idx[:len(ends)]
    groups = {"early": ends < main[0], "pre": (starts >= main[0]) & (ends < etf_day), "post": starts >= etf_day}
    rolled = {}
    for k in ("blend", "c55"):
        dr, dd = roll[k][0] - roll["hold"][0], roll[k][1] - roll["hold"][1]
        rolled[k] = {g: ({"n": int(sel.sum()), "win": float((dr[sel] > 0).mean() * 100),
                          "median": float(np.median(dr[sel])), "shallow": float((dd[sel] > 0).mean() * 100),
                          "deep_median": float(np.median(dd[sel]))} if sel.sum() else None)
                     for g, sel in groups.items()}

    # monthly block bootstrap of the yearly-return edge (main period, 24-clock versions)
    boot = {}
    month = main.year * 12 + main.month
    month_ids = np.unique(month)
    recent = month_ids >= 2024 * 12 + 1
    log_hold = pd.Series(np.log1p(m.change[w0:])).groupby(month).sum().values
    for ki, k in enumerate(RULES):
        diff = pd.Series(np.log1p(stag[k])).groupby(month).sum().values - log_hold
        boot[k] = {}
        for name, sel in (("all", np.ones(len(diff), dtype=bool)), ("recent", recent)):
            d = diff[sel]
            rng = np.random.default_rng(SEED + 77 + ki)
            draw = d[rng.integers(0, len(d), (DRAWS, len(d)))].sum(axis=1)
            boot[k][name] = {"wins": int((draw > 0).sum()), "months": int(len(d))}

    # ----- 4. cash interest -----
    def main_cut(net: np.ndarray, start) -> dict:
        return cr._stats(net[int(main.searchsorted(pd.Timestamp(start))):])

    interest = {k: {"all": cr._stats(stag_rf[k]), "after": main_cut(stag_rf[k], ETF)} for k in RULES}
    plain_after = {k: main_cut(stag[k], ETF) for k in RULES}
    hold_after = main_cut(m.change[w0:], ETF)

    # ----- 6. today -----
    blend_now = np.array([want["blend"][h][-1] for h in range(24)])
    size = np.array([np.prod(1 + text_nets["blend"][h][w0:]) for h in range(24)])
    today = {"blend": float((blend_now * size).sum() / size.sum() * 100), "blend_plain": float(blend_now.mean() * 100),
             "c55": int(sum(want["c55"][h][-1] > 0 for h in range(24))),
             "ma50": int(sum(want["ma50"][h][-1] > 0 for h in range(24))),
             "c20": int(sum(want["c20"][h][-1] > 0 for h in range(24)))}

    return {"m": m, "hold": hold, "stress": stress, "stag": stag, "years": years, "picks": picks, "before": before,
            "rho": rho, "chains": chains, "hold_chain": hold_chain, "label": label, "n_cand": len(keys),
            "n_family": len(specs), "blocks": blocks, "latest": latest, "etf": etf, "after_days": after_n,
            "rolled": rolled, "boot": boot, "interest": interest, "plain_after": plain_after,
            "hold_after": hold_after, "today": today, "last": dates[-1], "last_price": float(price[-1]),
            "first_early": idx[0], "rate_mean": float(rf[w0:].mean() * YEAR * 100), "has_rate": rate is not None,
            "rate_last": (rate.index[-1] if rate is not None else None),
            "trace": {"full": full, "text": text_nets, "nets": nets, "want": want, "engine": eng}}


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data["price"], data["b0"], data["pre"], data["daily"], data["rate"])
    m, hold, stress, last = r["m"], r["hold"], r["stress"], r["last"]
    start = _d(m.dates[m.w0])
    span = f"{start} 到 {_d(last)}"
    sims = f"{SIMS:,} 次模擬"

    def edge(x: dict) -> str:
        return _pp(x["ret"] - hold["ret"])

    # ----- table 1: the bottom line -----
    ch, hc, years = r["chains"], r["hold_chain"], r["years"]
    rows1 = []
    for k in RULES:
        s = stress[k]
        rows1.append([cr.NAMES[k], edge(s["clock0"]), edge(s["stag"]), edge(s["person_stag"]), edge(s["person_stag_rf"]),
                      _pct(s["person_stag"]["deepest"]),
                      f"{sims}裡 {s['person_stag']['below']:,} 次｜{sims}裡 {round(s['person']['below']):,} 次"])
    rows1.append([f"公平地事先挑（{years[0]} 年起，和同期一直抱著的 {_pct(hc['ret'])} 比；最深虧 {_pct(ch['pick']['deepest'])}）",
                  "—", _pp(ch["pick"]["ret"] - hc["ret"]), "—", "—", "—", "—"])
    rows1.append([f"對照：一直抱著不動（每年 {_pct(hold['ret'])}）", "0.0", "0.0", "0.0", "0.0", _pct(hold["deepest"]), "—"])

    # ----- table 2: stress tests -----
    def row(name: str, cell) -> list:
        return [name] + [cell(stress[k]) for k in RULES]

    rows2 = [
        row("教科書：收盤那一刻看、當下成交，成本 0.10%", lambda s: _two(s["text"])),
        row(f"看完 {WAIT} 小時後才成交（晚上 10 點看、隔天早上 8 點做的型態）", lambda s: _two(s["wait"])),
    ]
    rows2 += [row(f"晚 {d} 天才做", lambda s, d=d: _two(s["late"][d])) for d in LATE]
    rows2 += [row(f"每 {n} 天才看一次（{n} 種起算日，只列每年報酬）",
                  lambda s, n=n: _span([x["ret"] for x in s["every"][n]], _pct)) for n in EVERY]
    rows2.append(row("一週只看一次（7 種星期幾，只列每年報酬）", lambda s: _span([x["ret"] for x in s["weekly"]], _pct)))
    rows2 += [row(f"成本 {c * 100:.2f}%", lambda s, c=c: _two(s["cost"][c])) for c in COSTS]
    rows2 += [row(f"隨機漏掉 {p * 100:.0f}% 的進場（{sims}的中位數）", lambda s, p=p: _two(s["miss"][p])) for p in MISS]
    rows2 += [
        row(f"全部疊起來的一般人（{WAIT} 小時後成交、成本 {PERSON_COST * 100:.2f}%、漏 {PERSON_MISS * 100:.0f}%）：中位數",
            lambda s: _two(s["person"])),
        row("一般人：最差的一成", lambda s: f"{_pct(s['person']['ret_low'])}｜{_pct(s['person']['deepest_low'])}"),
        row(f"一般人：{sims}裡幾次報酬輸給一直抱著｜幾次虧得比一直抱著深",
            lambda s: f"{round(s['person']['below']):,} 次｜{round(s['person']['deeper']):,} 次"),
    ]

    # ----- table 3: fair pick-in-advance -----
    rows3 = []
    for y in years:
        k = r["picks"][y]
        mine, base = ch["pick"]["years"][y], hc["years"][y]
        partial = "（到 " + _d(last) + "）" if y == last.year and last < pd.Timestamp(f"{y}-12-31") else ""
        rows3.append([f"{y}{partial}", r["label"][k], _num(r["before"][y][k]), f"{_pct(mine[0])}｜{_pct(mine[1])}",
                      f"{_pct(base[0])}｜{_pct(base[1])}", "贏" if mine[0] > base[0] else "輸", _num(r["rho"][y])])
    n_y = len(years)

    def total(name: str, x: dict) -> list:
        return [name, "—", "—", f"{_pct(x['ret'])}｜{_pct(x['deepest'])}", f"{_pct(hc['ret'])}｜{_pct(hc['deepest'])}",
                f"{n_y} 年裡 {x['beat']} 年贏", "—"]

    rhos = [v for v in r["rho"].values() if v is not None]
    rows3.append(total(f"串起來 {years[0]}–{_d(last)}：每年事先挑", ch["pick"]))
    rows3[-1][-1] = f"平均 {_num(float(np.mean(rhos)))}" if rhos else "—"
    rows3 += [total(f"不挑，一直用「{SHORT[k]}」", ch[k]) for k in ("blend", "vol40", "c55", "ma50")]

    # ----- table 4: fading edge and cash interest -----
    def versus(x: dict | None, base: dict | None) -> str:
        return "—" if x is None or base is None else f"{_pp(x['ret'] - base['ret'])}｜{_pp(x['deepest'] - base['deepest'])}"

    def period(name: str, st: dict) -> list:
        return [name] + [(_two(st[k]) if st[k] else "—") for k in ("hold", "blend", "c55")] + \
               [versus(st["blend"], st["hold"]), versus(st["c55"], st["hold"])]

    rows4 = [period(name + ("＊" if early else ""), st) for name, st, early in r["blocks"]]
    rows4.append(period(f"最近兩年（到 {_d(last)}）", r["latest"]))
    rows4.append(period(f"現貨 ETF 上市前（{start} ～ 2024-01-10）", r["etf"]["before"]))
    rows4.append(period(f"現貨 ETF 上市後（{ETF} ～ {_d(last)}）", r["etf"]["after"]))

    def roll_cell(k: str, g: str) -> str:
        x = r["rolled"][k][g]
        return "—" if not x else f"{x['win']:.0f}% 的窗口報酬較高｜{x['shallow']:.0f}% 虧得較淺"

    def roll_diff(k: str, g: str) -> str:
        x = r["rolled"][k][g]
        return "—" if not x else f"中位數 {_pp(x['median'])}｜{_pp(x['deep_median'])}"

    for g, name in (("early", "滾動兩年窗口：2018 年以前結束的＊"), ("pre", f"滾動兩年窗口：{start[:4]} 年起、ETF 上市前結束的"),
                    ("post", "滾動兩年窗口：ETF 上市後才開始的")):
        if r["rolled"]["blend"][g]:
            rows4.append([f"{name}（{r['rolled']['blend'][g]['n']:,} 個）", "—", roll_cell("blend", g), roll_cell("c55", g),
                          roll_diff("blend", g), roll_diff("c55", g)])
    boot = r["boot"]

    def boot_cell(k: str, g: str) -> str:
        return f"{DRAWS:,} 次重抽裡 {boot[k][g]['wins']:,} 次報酬贏過一直抱著"

    rows4.append([f"按月重抽：整段（{boot['blend']['all']['months']} 個月）", "—", boot_cell("blend", "all"),
                  boot_cell("c55", "all"), "—", "—"])
    rows4.append([f"按月重抽：2024 年 1 月起（{boot['blend']['recent']['months']} 個月）", "—", boot_cell("blend", "recent"),
                  boot_cell("c55", "recent"), "—", "—"])
    inter = r["interest"]
    if r["has_rate"]:
        rows4.append(period(f"現金有利息：整段（{span}）",
                            {"hold": hold, "blend": inter["blend"]["all"], "c55": inter["c55"]["all"]}))
        rows4.append(period(f"現金有利息：ETF 上市後（{ETF} 起）",
                            {"hold": r["hold_after"], "blend": inter["blend"]["after"], "c55": inter["c55"]["after"]}))

    rate_text = (f"美國三個月國庫券利率（這段期間平均一年 {r['rate_mean']:.1f}%）" if r["has_rate"]
                 else "0%（這次拿不到國庫券利率）")
    tables = [
        {"caption": (f"總表：每種做法的每年報酬比一直抱著多（＋）或少（−）幾個百分點（比特幣，{span}；一直抱著每年 {_pct(hold['ret'])}、"
                     f"最深虧 {_pct(hold['deepest'])}）。(a) 教科書回測：只用台灣早上 8 點這一個收盤時刻、當下成交、成本 0.10%；"
                     "(b) 把錢平分到 24 個收盤時刻，拿掉「剛好選到哪個時鐘」的運氣；(c) 在 (b) 之上照一般人的方式做——"
                     f"看完 {WAIT} 小時後才成交、成本 {PERSON_COST * 100:.2f}%、每次進場有 {PERSON_MISS * 100:.0f}% 機會漏掉，"
                     f"取{sims}的中位數；(d) 同 (c)，但沒有買比特幣的錢領{rate_text}。最後一欄是{sims}裡每年報酬輸給一直抱著的次數："
                     "前面是分成 24 份的人，後面是只用一個時鐘的人（24 個時鐘平均）。"),
         "columns": ["做法", "(a) 教科書單一時鐘", "(b) 分成 24 個時鐘", "(c) 分 24 份＋一般人做法", "(d) 同 (c)＋現金有利息",
                     "(c) 的最深虧損（中位數）", "輸給一直抱著：分 24 份｜單一時鐘"],
         "rows": rows1},
        {"caption": (f"一般人的各種不完美，六種做法各測一次（{span}）。除了最後三列，每格都是 24 個收盤時刻各跑一次的平均，"
                     "所以沒有時鐘運氣；格式是「每年報酬｜最深虧損」。沒有特別寫的列，成本都是 0.10%、其他條件照教科書。"
                     f"最後三列是每個時鐘各做{sims}（只用一個時鐘的人），次數是 24 個時鐘平均下來每 {SIMS:,} 次裡的次數。"
                     f"對照：一直抱著每年 {_pct(hold['ret'])}、最深虧 {_pct(hold['deepest'])}。"),
         "columns": ["情境"] + [SHORT[k] for k in RULES],
         "rows": rows2},
        {"caption": (f"公平地事先挑：每年年初，把 {r['n_cand']} 個候選（均線、兩條均線交叉、新高新低、過去漲跌四類共 {r['n_family']} 組參數，"
                     "加上 17 條混合、混合×波動 40%、九條新高＋移動停損）各用 24 個收盤時刻跑過「那一年以前」的全部資料"
                     "（1 小時線從 2017-08 起），取 24 個時鐘的「報酬÷波動」平均當分數，挑分數最高的，下一年把錢平分到 24 個時鐘照它做。"
                     "成績欄是「那一年的報酬｜年內最大跌幅」；最後一欄是所有候選「之前的分數」和「那一年的分數」的排名相關"
                     "（+1 完全一致、0 沒有關係、負的表示之前好的那年反而差）。"),
         "columns": ["年份／做法", "事先挑到的", "挑的時候的分數", "分成 24 份的成績", "一直抱著", "報酬有沒有贏", "排名相關"],
         "rows": rows3},
        {"caption": ("優勢有沒有變小，以及現金有利息的影響。每格是「每年報酬｜最深虧損」，最後兩欄是和一直抱著的差"
                     "（每年報酬差幾個百分點｜最深虧損淺幾個百分點，正的表示比較淺）。有＊的期間早於 1 小時線，只能用日線、"
                     f"單一時鐘（UTC 0 點收盤）來算，會帶有時鐘運氣；{start[:4]} 年起是分成 24 個時鐘的版本。"
                     f"「按月重抽」是把每個月的結果當成一張牌、抽出放回重排 {DRAWS:,} 次，看有幾次順勢做法的每年報酬高於一直抱著——"
                     "接近一半就表示和零分不出來。"),
         "columns": ["期間／看法", "一直抱著", "17 條混合", "55 日新高／20 日新低", "混合比一直抱著", "55/20 比一直抱著"],
         "rows": rows4},
    ]

    # ----- today -----
    t = r["today"]
    now = [
        {"label": f"最後一個收完的日子（{_d(last)}，台灣時間隔天早上 8 點收）", "value": f"比特幣 {r['last_price']:,.0f} 美元"},
        {"label": "「17 條混合」分成 24 份：規則目前的持有比例", "value": f"{t['blend']:.0f}%（24 個時鐘簡單平均 {t['blend_plain']:.0f}%）"},
        {"label": "「55 日新高／20 日新低」：24 個時鐘裡目前是持有狀態的", "value": cr._of24(t["c55"])},
        {"label": "「收盤高於 50 日均線」：24 個時鐘裡目前是持有狀態的", "value": cr._of24(t["ma50"])},
        {"label": "這幾格的意思", "value": "只是描述規則目前停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    med = {k: stress[k]["person_stag"] for k in RULES}
    order = sorted(RULES, key=lambda k: -med[k]["ret"])
    above = [k for k in order if med[k]["ret"] > hold["ret"]]
    below = [k for k in order if med[k]["ret"] <= hold["ret"]]
    deeps = [med[k]["deepest"] for k in RULES]

    def tag(k: str) -> str:
        return f"「{SHORT[k]}」{_pct(med[k]['ret'])}"

    if above:
        text = f"六種順勢做法裡 {len(above)} 種每年報酬仍高於一直抱著的 {_pct(hold['ret'])}（{tag(above[0])} 最高"
        text += f"、{tag(above[-1])} 最低）" if len(above) > 1 else "）"
        text += ("，" + "、".join(tag(k) for k in below) + " 沒有贏") if below else ""
    else:
        text = f"六種順勢做法的每年報酬都沒有高於一直抱著的 {_pct(hold['ret'])}（最高{tag(order[0])}），只剩虧得比較淺"
    part1 = (f"{start[:4]} 年到 {_d(last)}，把錢分到 24 個收盤時刻、再照一般人的方式做，{text}；最深虧 {_pct(min(deeps))} 到 "
             f"{_pct(max(deeps))}，一直抱著 {_pct(hold['deepest'])}。")
    counts: dict[str, int] = {}
    for y in years:
        counts[r["picks"][y]] = counts.get(r["picks"][y], 0) + 1
    fav = max(counts, key=lambda k: counts[k])
    pick = ch["pick"]
    part2 = (f"公平地事先挑，{n_y} 年裡 {counts[fav]} 年挑到「{r['label'][fav]}」，串起來每年 {_pct(pick['ret'])}、最深虧 "
             f"{_pct(pick['deepest'])}，同期一直抱著 {_pct(hc['ret'])}、{_pct(hc['deepest'])}：報酬"
             f"{'贏' if pick['ret'] > hc['ret'] else '輸'}、虧得{'淺' if pick['deepest'] > hc['deepest'] else '深'}。")
    ea, wins_recent = r["etf"]["after"], boot["blend"]["recent"]["wins"]
    if wins_recent >= 0.95 * DRAWS:
        tell = "報酬優勢還看得出來"
    elif wins_recent <= 0.05 * DRAWS:
        tell = "報酬明顯輸給一直抱著"
    else:
        tell = "報酬優勢和零分不出來"
    gain = stress["blend"]["person_stag_rf"]["ret"] - stress["blend"]["person_stag"]["ret"]
    part3 = (f"2024 年 ETF 上市後 17 條混合每年 {_pct(ea['blend']['ret'])}、一直抱著 {_pct(ea['hold']['ret'])}，按月重抽 "
             f"{DRAWS:,} 次裡 {wins_recent:,} 次贏，{tell}，虧得淺（{_pct(ea['blend']['deepest'])} 對 "
             f"{_pct(ea['hold']['deepest'])}）則還在"
             + (f"；現金領國庫券利息，混合的一般人版每年再多 {gain:.1f} 個百分點。" if r["has_rate"] else "。"))
    verdict = part1 + part2 + part3

    # ----- notes -----
    s20 = stress["c20"]
    notes = [
        "算法：用幣安 1 小時線把日線重新拼 24 次，一天分別結束在每一個整點（UTC 0 點＝台灣早上 8 點）。每個時鐘用自己的日收盤算訊號，"
        "從那一刻起持有 24 小時；每次買賣扣交易金額的成本，持有比例不是 0 或 100% 的做法每天調回目標、這些小額調整也算成本；"
        "一直抱著不扣成本。不管哪個時鐘，損益都用台灣早上 8 點到隔天早上 8 點來算。「分成 24 份」是期初把錢平分、每份各用一個時鐘、"
        "之後不互相調撥。",
        "六種做法：創 20 日新高進場／10 日新低出場；創 55 日新高進場／20 日新低出場；收盤高於 50 日、100 日均線才持有；"
        "17 條混合＝「不挑參數，把很多順勢規則混在一起」那一頁的 17 條規則，看多的占幾成就持有幾成；混合×波動 40%＝再乘上"
        "「40% ÷ 最近 30 天的波動」（最多乘 1）。",
        f"一般人的測試怎麼定義：「看完 {WAIT} 小時後才成交」是在某個整點看收盤、{WAIT} 小時後用當時的價格成交，24 個起始整點平均；"
        f"其中「台灣晚上 10 點看、隔天早上 8 點做」這一個，20 日新高規則的一般人版中位數是每年 {_pct(s20['person']['literal'])}"
        f"（{sims}裡 {s20['person']['literal_below']:,} 次輸給一直抱著），24 個時鐘合起來是 {_pct(s20['person']['ret'])}。"
        "「少看幾次」：兩條新高規則只在看的那天判斷是不是新高／新低，其他日子的訊號當作沒看到；均線和混合是只在看的那天把持有比例調到當天的目標，"
        "其餘日子不動。",
        f"「漏掉進場」：目標持有比例一天內上升超過 {RISE * 100:.0f} 個百分點的日子算一次進場，每次有 10% 或 25% 的機會沒跟；沒買到的那一部分"
        "就一直缺著，直到目標比例降回來為止（全進全出的規則就是整波不參加）。混合類每次只缺一小部分，所以受的影響比單一規則小。"
        f"單獨漏掉進場的兩列是{sims}平均分給 24 個時鐘；一般人情境是每個時鐘各{sims}，分成 24 份的版本裡每一份各自獨立漏。",
        f"隨機模擬用固定的亂數種子（{SEED} 起），資料沒變的話每次重跑結果都一樣；「{SIMS:,} 次裡幾次」本身有十幾二十次的誤差。"
        "分成 24 份的人因為 24 份各漏各的、互相抵銷，結果比只用一個時鐘的人集中得多——真的只照一個時鐘做的人，請看單一時鐘那個次數。",
        f"1 小時線從 2017-08 才有，所以換時鐘的結果只能從 {start} 算起。更早的日線（儀表板的長期日線，UTC 0 點收盤）只接在每個時鐘前面"
        f"當暖機，讓長天數的規則在 {start[:4]} 年初就有值；事先挑的計分從 2017-08 開始，所以 {years[0]} 年那次只看了一年多的資料。"
        f"第四張表有＊的期間（{_d(r['first_early'])} 起）用的是日線、單一時鐘，不能換時鐘。",
        f"候選名單是用今天的知識寫的：哪幾類規則、哪些天數、要不要放混合，都是已經知道這段歷史之後才決定的，所以「公平地事先挑」"
        f"只拿掉了時鐘運氣和挑參數的後見之明，拿不掉「挑候選名單」的後見之明。原本的事先挑有 44 組參數，其中均線、交叉、新高新低、"
        f"過去漲跌四類是 {r['n_family']} 組（其餘是資金費率和恐懼指數類，這裡不用），加上 3 個混合類共 {r['n_cand']} 個候選。",
        f"獨立的年份很少：換時鐘的資料不到九個日曆年，大致就是兩三輪大漲大跌；事先挑只有 {n_y} 次，2024 年以後不到三年。"
        "「按月重抽」假設每個月互相獨立、而且未來和這段期間同一個樣子，實際上行情是一段一段的，所以它對不確定性的估計偏樂觀。"
        "24 個時鐘用的是同一段價格，不是 24 次獨立的檢驗。",
        "成本很粗略：一律按交易金額的固定比例扣，成交價用整點的收盤價，實際的手續費、價差、滑價會隨交易所、金額和年代不同；沒有算稅。"
        + (f"現金利息用美國三個月國庫券的貼現率除以 365 當每天的利息（這段期間平均一年 {r['rate_mean']:.1f}%，六種做法平均有 "
           f"{100 - max(stress[k]['share'] for k in RULES):.0f}% 到 {100 - min(stress[k]['share'] for k in RULES):.0f}% 的錢是現金；資料到 {_d(r['rate_last'])}，假日沿用前一天），只給沒有買比特幣的那部分，"
           "也是粗略的：實際上放在交易所的美元或穩定幣不一定領得到這個利率。" if r["has_rate"] else ""),
        "過去的結果不保證未來會一樣。這一頁只是整理這些做法過去做了什麼、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += data["problems"]

    return {
        "id": "robust_trend",
        "title": "拿掉時鐘運氣、照一般人的方式做，哪一種順勢做法還贏？",
        "question": ("「20 日新高」那條規則的報酬優勢，換個收盤時刻、照一般人的方式做就沒了。其他比較穩的順勢做法撐得住嗎？"
                     "公平地事先挑會挑到什麼？2024 年以後優勢還在不在？現金有利息的話差多少？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 BTCUSDT 1 小時線（2017-08 起）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（暖機與 2018 年以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            ({"name": "美國財政部：每日國庫券利率（13 週）",
              "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates"}
             if data["rate_source"] == "treasury" else
             {"name": "FRED：美國三個月國庫券利率（DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"}),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
