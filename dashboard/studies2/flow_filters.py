"""Historical study "flow_filters": do exchange deposits or stablecoin growth add anything to the trend rule that
already works - and can the exchange-flow result be trusted when the data is rewritten after the fact?

Two rules pre-registered by someone else (f1devbin/Crypto-Research-Labs, H-020 and H-006) are redone here:
- deposit filter: stay out of bitcoin for a day when the 7-day sum of net exchange deposits (Coin Metrics community
  FlowInExNtv - FlowOutExNtv) is above its own trailing one-year 80th percentile;
- stablecoin filter: hold only while the 7-day growth of total stablecoin supply (DefiLlama) is above its trailing
  one-year median.
Each is tested alone and on top of the dashboard's 20-day-high / 10-day-low trend rule, with a small parameter grid,
a pick-in-advance test, a circular-shift luck check, a balance-versus-flow consistency check and a revision check
(a dated snapshot of the fetched series is kept in cache/_store/flow_filters_snapshots/).

No look-ahead inside the test: a signal decided at the close of day t sets the position of day t+1, percentiles and
medians use earlier values only, and both data series are used one day late (LAG) because the provider publishes
day t a few hours after day t has closed. What the test cannot remove is the provider rewriting its own history.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import flows as flows_collector
from dashboard.studies2 import trend_ensemble as te

COST = te.COST
YEAR = te.YEAR
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_CATALOG = "https://community-api.coinmetrics.io/v4/catalog/asset-metrics"
METRICS = ("FlowInExNtv", "FlowOutExNtv", "SplyExNtv")
SNAPSHOTS = common.STORE / "flow_filters_snapshots"
LAG = 1  # days between a value's date and the first close at which it is used
SUM_DAYS, PCT, WINDOW = 7, 0.80, 365  # H-020 as stated
GRID_SUMS, GRID_PCTS, GRID_WINDOWS = (7, 14, 30), (0.70, 0.80, 0.90), (365, 730)
STABLE_DAYS, STABLE_VARIANTS = 7, (14, 30)
OOS_START = "2022-01-01"
ORIGINAL_END = "2026-10-13"  # the original study was run on 2026-10-14
PICK_FROM = 2021
TOP_GAPS = 15
AROUND = 15  # days removed on each side of a large disagreement (31 days in all)
GAP_MIN = 1.0  # BTC; smaller differences are rounding
SIMS = 1000
SEED = 20220101
KEYS = ("hold", "trend", "hold_flow", "trend_flow", "hold_stable", "trend_stable")
NAMES = {"hold": "一直抱著不動", "trend": "順勢規則（創 20 日新高進場、10 日新低出場）",
         "hold_flow": "抱著＋存入過濾（交易所淨存入偏高就空手）", "trend_flow": "順勢規則＋存入過濾",
         "hold_stable": "抱著＋穩定幣過濾（供給成長高於中位數才持有）", "trend_stable": "順勢規則＋穩定幣過濾",
         "hold_balance": "抱著＋存入過濾，改用交易所餘額的變化來算",
         "hold_flow_cut": f"抱著＋存入過濾，拿掉 {TOP_GAPS} 次最大對不上日子的前後共 {2 * AROUND + 1} 天"}
SHORT = {"hold": "抱著", "trend": "順勢", "hold_flow": "抱著＋存入", "trend_flow": "順勢＋存入",
         "hold_stable": "抱著＋穩定幣", "trend_stable": "順勢＋穩定幣"}

_pct, _num, _d = te._pct, te._num, te._d


def _btc(v, signed: bool = True) -> str:
    if not te._ok(v):
        return "—"
    return f"{float(v):+,.0f} 顆" if signed else f"{float(v):,.0f} 顆"


# ---------- data ----------

def _fetch_cm() -> pd.DataFrame:
    """Daily exchange inflow, outflow and balance in BTC, plus the day the provider last wrote each row."""
    params = {"assets": "btc", "metrics": ",".join(METRICS), "frequency": "1d", "page_size": 10000}
    try:
        reply = common.get_json(CM_URL, params, timeout=60)
    except Exception as error:  # noqa: BLE001 - say which names exist instead of failing blindly
        try:
            catalog = common.get_json(CM_CATALOG, {"assets": "btc"}, timeout=60)
            names = [m.get("metric", "") for m in (catalog.get("data") or [{}])[0].get("metrics", [])]
            hint = "、".join(n for n in names if "Ex" in n and ("Flow" in n or "Sply" in n))[:300]
        except Exception:  # noqa: BLE001
            hint = "（目錄也讀不到）"
        raise RuntimeError(f"Coin Metrics 沒有回傳 {','.join(METRICS)}（{type(error).__name__}）；目錄裡相近的指標：{hint}")
    rows = list(reply.get("data") or [])
    for _ in range(30):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=60)
        rows += list(reply.get("data") or [])
    if len(rows) < 2000:
        raise RuntimeError(f"Coin Metrics 只回傳 {len(rows)} 天")
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame["time"].str[:10])
    out = pd.DataFrame({"inflow": pd.to_numeric(frame[METRICS[0]], errors="coerce"),
                        "outflow": pd.to_numeric(frame[METRICS[1]], errors="coerce"),
                        "balance": pd.to_numeric(frame[METRICS[2]], errors="coerce")})
    stamp = frame.get(f"{METRICS[0]}-status-time")
    out["written"] = pd.to_datetime(stamp.str[:10], errors="coerce") if stamp is not None else pd.NaT
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index < pd.Timestamp(common.today_utc())]


def _fetch_stable() -> pd.Series:
    """Total stablecoin supply in USD per day (DefiLlama, the same endpoint and sum as collectors/flows.py)."""
    reply = common.get_json(flows_collector.LLAMA_STABLE, timeout=60)
    points = {}
    for point in reply:
        total = sum(float(v) for v in (point.get("totalCirculatingUSD") or {}).values() if v is not None)
        if total > 0:
            points[pd.Timestamp(flows_collector._ts_day(point["date"]))] = total
    series = pd.Series(points).sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]
    if len(series) < 800:
        raise RuntimeError(f"DefiLlama 穩定幣資料只有 {len(series)} 天")
    return series.asfreq("D").ffill()


# ---------- signals (everything here is decided at a day's close; _sim() moves it to the next day) ----------

def _flow_state(source: pd.Series, days: int, pct: float, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(elevated: 1/0/NaN on the data's own dates, the rolling sum, its threshold from earlier sums only)."""
    total = source.rolling(days, min_periods=days).sum()
    limit = total.shift(1).rolling(window, min_periods=window).quantile(pct)
    high = (total > limit).astype(float).where(limit.notna() & total.notna())
    return high, total, limit


def _flow_target(source: pd.Series, index: pd.DatetimeIndex, days: int = SUM_DAYS, pct: float = PCT,
                 window: int = WINDOW, lag: int = LAG) -> pd.Series:
    """1 = may hold, 0 = stay out, at each close, using data dated at least `lag` days earlier."""
    high = _flow_state(source, days, pct, window)[0]
    return (1 - high).reindex(index).shift(lag)


def _stable_state(supply: pd.Series, days: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    growth = supply / supply.shift(days) - 1
    middle = growth.shift(1).rolling(WINDOW, min_periods=WINDOW).median()
    return (growth > middle).astype(float).where(middle.notna() & growth.notna()), growth, middle


def _stable_target(supply: pd.Series, index: pd.DatetimeIndex, days: int = STABLE_DAYS, lag: int = LAG) -> pd.Series:
    return _stable_state(supply, days)[0].reindex(index).shift(lag)


def _targets(close: pd.Series, cm: pd.DataFrame, supply: pd.Series) -> pd.DataFrame:
    """What each approach wants to hold, decided at each close. Kept separate so look-ahead can be tested."""
    index = close.index
    net = cm["inflow"] - cm["outflow"]
    trend = te._signals(close)[0]["C20"]
    flow = _flow_target(net, index)
    stable = _stable_target(supply, index)
    out = pd.DataFrame(index=index)
    out["hold"] = 1.0
    out["trend"] = trend
    out["hold_flow"] = flow
    out["trend_flow"] = trend * flow
    out["hold_stable"] = stable
    out["trend_stable"] = trend * stable
    out["hold_balance"] = _flow_target(cm["balance"].diff(), index)
    return out


# ---------- simulating ----------

def _sim(target: pd.Series, change: pd.Series, hold: bool = False) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = change.index
    if hold:
        return change.fillna(0.0), pd.Series(1.0, index=index), pd.Series(0.0, index=index)
    pos = target.shift(1)  # decided at the close of day t, held during day t+1
    net, trade = te._run(pos.values, change.values, COST)
    return pd.Series(net, index=index), pos.fillna(0.0), pd.Series(trade, index=index)


def _first(target: pd.Series) -> pd.Timestamp:
    return target.shift(1).first_valid_index()


def _measure(parts, rows: np.ndarray) -> dict:
    return te._measure(parts[0][rows], parts[1][rows], parts[2][rows])


def _ratio_matrix(pos: np.ndarray, change: np.ndarray) -> np.ndarray:
    """Return ÷ volatility of each column of all-in / all-out positions (same arithmetic as te._run for 0/1)."""
    trade = np.abs(np.diff(pos, axis=0, prepend=0.0))
    net = pos * change[:, None] - COST * trade
    growth = np.prod(1 + net, axis=0)
    ret = (growth ** (YEAR / len(change)) - 1) * 100
    vol = np.std(net, axis=0, ddof=1) * math.sqrt(YEAR) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)


def _shift_test(pos: np.ndarray, change: np.ndarray) -> dict:
    """Slide the in/out schedule round the period by random offsets and see where the real one falls."""
    n = len(pos)
    change = np.nan_to_num(change)
    real = float(_ratio_matrix(pos[:, None], change)[0])
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(30, max(n - 30, 31), SIMS)
    moved = pos[(np.arange(n)[:, None] - offsets[None, :]) % n]
    ratios = _ratio_matrix(moved, change)
    return {"real": real, "as_good": float((ratios >= real).mean() * 100), "median": float(np.median(ratios))}


# ---------- consistency and revisions ----------

def _gaps(cm: pd.DataFrame) -> dict:
    """Daily change in exchange balance against net flow, and when the provider wrote each row."""
    net = cm["inflow"] - cm["outflow"]
    move = cm["balance"].diff()
    gap = (move - net).dropna()
    late = (cm["written"] - cm.index.to_series()).dt.days
    rewritten = late[late > 3]  # written long after the day itself = recomputed in bulk
    live_start = (rewritten.index[-1] + pd.Timedelta(days=1)) if len(rewritten) else cm.index[0]
    before, after = gap[gap.index < live_start], gap[gap.index >= live_start]
    top = gap.reindex(gap.abs().sort_values(ascending=False).index[:TOP_GAPS])
    return {"net": net, "move": move, "gap": gap, "top": top, "live_start": live_start,
            "rewritten": int(len(rewritten)),
            "written_from": cm["written"][rewritten.index].min() if len(rewritten) else None,
            "written_to": cm["written"][rewritten.index].max() if len(rewritten) else None,
            "before_days": int(len(before)), "before_off": int((before.abs() > GAP_MIN).sum()),
            "after_days": int(len(after)), "after_off": int((after.abs() > GAP_MIN).sum()),
            "after_sum": float(after.sum()), "after_up": int((after > GAP_MIN).sum())}


def _load_snapshot(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    index = pd.date_range(data["start"], periods=len(data["inflow"]), freq="D")
    return pd.DataFrame({"inflow": data["inflow"], "outflow": data["outflow"], "balance": data["balance"]},
                        index=index, dtype=float)


def _compare(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    days = old.index.intersection(new.index)
    old_net = (old["inflow"] - old["outflow"]).reindex(days)
    new_net = (new["inflow"] - new["outflow"]).reindex(days)
    diff = (new_net - old_net).dropna()
    changed = diff[diff.abs() > 0.01]
    balance = (new["balance"].reindex(days) - old["balance"].reindex(days)).dropna()
    old_high = _flow_state(old["inflow"] - old["outflow"], SUM_DAYS, PCT, WINDOW)[0].reindex(days)
    new_high = _flow_state(new["inflow"] - new["outflow"], SUM_DAYS, PCT, WINDOW)[0].reindex(days)
    both = old_high.notna() & new_high.notna()
    return {"days": int(len(diff)), "changed": int(len(changed)),
            "largest": float(changed.abs().max()) if len(changed) else 0.0,
            "largest_day": changed.abs().idxmax() if len(changed) else None,
            "earliest": changed.index.min() if len(changed) else None,
            "total": float(changed.abs().sum()),
            "balance_changed": int((balance.abs() > 0.01).sum()),
            "balance_largest": float(balance.abs().max()) if len(balance) else 0.0,
            "flips": int((old_high[both] != new_high[both]).sum())}


def _revisions(cm: pd.DataFrame, folder: Path = SNAPSHOTS, today: str | None = None) -> dict:
    """Save today's series and compare it with the oldest and the most recent earlier snapshot."""
    today = today or common.today_utc()
    out = {"saved": False, "earlier": 0, "compare": []}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        earlier = sorted(p for p in folder.glob("????-??-??.json.gz") if p.name[:10] < today)
        out["earlier"] = len(earlier)
        picks = [earlier[0]] + ([earlier[-1]] if len(earlier) > 1 else []) if earlier else []
        for path in picks:
            try:
                result = _compare(_load_snapshot(path), cm)
                result["against"] = path.name[:10]
                out["compare"].append(result)
            except Exception:  # noqa: BLE001 - a damaged snapshot is skipped
                continue
        full = cm[["inflow", "outflow", "balance"]].asfreq("D")
        payload = {"fetched_utc": common.now_utc(), "start": _d(full.index[0]), "metrics": list(METRICS),
                   **{c: [None if not math.isfinite(v) else round(float(v), 8) for v in full[c].values]
                      for c in ("inflow", "outflow", "balance")}}
        with gzip.open(folder / f"{today}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        out["saved"] = True
        # keep the oldest, the first of each month and the last 14; a full copy every day would pile up
        kept_months, keep = set(), set()
        files = sorted(folder.glob("????-??-??.json.gz"))
        for position, path in enumerate(files):
            month = path.name[:7]
            if position == 0 or month not in kept_months or position >= len(files) - 14:
                keep.add(path)
            kept_months.add(month)
        for path in files:
            if path not in keep:
                path.unlink()
    except Exception as error:  # noqa: BLE001 - the store is a convenience; the study still runs without it
        out["error"] = f"{type(error).__name__}: {error}"[:120]
    return out


# ---------- everything that depends on the data ----------

def _compute(close: pd.Series, cm: pd.DataFrame, supply: pd.Series) -> dict:
    index = close.index
    last = index[-1]
    change = close.pct_change()
    target = _targets(close, cm, supply)
    net = cm["inflow"] - cm["outflow"]
    runs = {k: _sim(target[k], change, hold=(k == "hold")) for k in target.columns}
    flow_start, stable_start = _first(target["hold_flow"]), _first(target["hold_stable"])
    oos = pd.Timestamp(OOS_START)

    def rows_from(start) -> np.ndarray:
        return np.asarray(index >= pd.Timestamp(start))

    def block(keys, rows) -> dict:
        table = {k: _measure(runs[k], rows) for k in keys}
        hold = _measure(runs["hold"], rows)
        for k in keys:
            years = sorted(hold["years"])
            table[k]["beat"] = sum(1 for y in years if table[k]["years"][y] > hold["years"][y] + 1e-9)
            table[k]["n_years"] = len(years)
            table[k]["hold"] = hold
        return table

    flow_keys = ("hold", "trend", "hold_flow", "trend_flow", "hold_balance")
    stable_keys = ("hold_stable", "trend_stable")
    periods = {"oos": {**block(flow_keys, rows_from(oos)), **block(stable_keys, rows_from(oos))},
               "full": {**block(flow_keys, rows_from(flow_start)), **block(stable_keys, rows_from(stable_start))}}
    periods["full"]["trend_since_stable"] = _measure(runs["trend"], rows_from(stable_start))

    # ----- the original study's own window and its own arithmetic (average daily return ÷ its spread) -----
    original = {}
    then = np.asarray((index >= oos) & (index <= pd.Timestamp(ORIGINAL_END)))
    for k in ("hold", "hold_flow"):
        daily = runs[k][0][then].values
        m = _measure(runs[k], then)
        spread = float(np.std(daily, ddof=1))
        original[k] = {"sharpe": float(np.mean(daily) / spread * math.sqrt(YEAR)) if spread > 1e-12 else 0.0,
                       "ret": m["ret"], "deepest": m["deepest"], "share": m["share"]}

    # ----- the days around the largest balance-versus-flow disagreements removed -----
    gaps = _gaps(cm)
    near = np.zeros(len(index), dtype=bool)
    for day in gaps["top"].index:
        near |= np.asarray((index >= day - pd.Timedelta(days=AROUND)) & (index <= day + pd.Timedelta(days=AROUND)))
    for name, start in (("oos", oos), ("full", flow_start)):
        rows = rows_from(start) & ~near
        periods[name]["hold_flow_cut"] = block(("hold_flow",), rows)["hold_flow"]
        periods[name]["removed"] = int((rows_from(start) & near).sum())

    # ----- since the provider started writing values day by day -----
    live_rows = rows_from(max(gaps["live_start"], flow_start))
    live = {"days": int(live_rows.sum()),
            "flow_out": int((runs["hold_flow"][1][live_rows] == 0).sum()),
            "balance_out": int((runs["hold_balance"][1][live_rows] == 0).sum()),
            "same": float((runs["hold_flow"][1][live_rows] == runs["hold_balance"][1][live_rows]).mean() * 100)
            if live_rows.sum() else None}

    # ----- parameter grid -----
    grid = {}
    for window in GRID_WINDOWS:
        for days in GRID_SUMS:
            for pct in GRID_PCTS:
                parts = _sim(_flow_target(net, index, days, pct, window), change)
                first = _first(_flow_target(net, index, days, pct, window))
                grid[(window, days, pct)] = {"oos": _measure(parts, rows_from(oos)),
                                             "full": _measure(parts, rows_from(max(first, flow_start)))}
    grid_start = max(_first(_flow_target(net, index, max(GRID_SUMS), 0.8, max(GRID_WINDOWS))), flow_start)
    for key, cell in grid.items():  # one common start so the cells of the full-sample half can be compared
        cell["full"] = _measure(_sim(_flow_target(net, index, key[1], key[2], key[0]), change), rows_from(grid_start))
    grid_hold = {"oos": _measure(runs["hold"], rows_from(oos)), "full": _measure(runs["hold"], rows_from(grid_start))}
    same_day = {}
    for pct in GRID_PCTS:  # the day's own figure used at that day's close (it is published a few hours later)
        parts = _sim(_flow_target(net, index, SUM_DAYS, pct, WINDOW, lag=0), change)
        same_day[pct] = {"oos": _measure(parts, rows_from(oos)), "full": _measure(parts, rows_from(grid_start))}

    # ----- overlap with the trend rule -----
    overlap = {}
    for name, start in (("oos", oos), ("full", flow_start)):
        rows = rows_from(start)
        day = change[rows].fillna(0.0).values * 100
        flow_out = runs["hold_flow"][1][rows].values == 0
        trend_out = runs["trend"][1][rows].values == 0

        def mean(mask) -> float | None:
            return float(day[mask].mean()) if mask.sum() else None

        overlap[name] = {"flow_out": int(flow_out.sum()), "both": int((flow_out & trend_out).sum()),
                         "share": float((flow_out & trend_out).sum() / flow_out.sum() * 100) if flow_out.sum() else None,
                         "trend_out_share": float(trend_out.mean() * 100),
                         "extra": int((flow_out & ~trend_out).sum()),
                         "in_flow_out": mean(flow_out & ~trend_out), "in_flow_ok": mean(~flow_out & ~trend_out),
                         "out_flow_out": mean(flow_out & trend_out), "out_flow_ok": mean(~flow_out & trend_out)}

    # ----- luck check -----
    luck = {}
    for name, start in (("oos", oos), ("full", flow_start)):
        rows = rows_from(start)
        luck[name] = _shift_test(runs["hold_flow"][1][rows].values, change[rows].values)

    # ----- stablecoin variants -----
    stable_variants = {}
    trend = target["trend"]
    for days in STABLE_VARIANTS:
        alone = _stable_target(supply, index, days)
        first = max(_first(alone), stable_start)
        stable_variants[days] = {"start": first}
        for label, series in (("hold", alone), ("trend", trend * alone)):
            parts = _sim(series, change)
            stable_variants[days][label] = {"oos": _measure(parts, rows_from(oos)),
                                            "full": _measure(parts, rows_from(first))}

    # ----- pick in advance, year by year -----
    common_start = max(_first(target[k]) for k in KEYS if k != "hold")
    years = [y for y in range(PICK_FROM, last.year + 1)]
    chosen, picked = {}, pd.Series(np.nan, index=index)
    for year in years:
        before = np.asarray((index >= common_start) & (index < pd.Timestamp(f"{year}-01-01")))
        scores = {k: te._score_parts(runs[k][0][before].values)[2] for k in KEYS}
        chosen[year] = max(KEYS, key=lambda k: scores[k])
        picked[index.year == year] = target[chosen[year]][index.year == year]
    pick_rows = rows_from(f"{PICK_FROM}-01-01")
    pick = {"chosen": chosen, "years": years, "common_start": common_start,
            "picked": _measure(_sim(picked, change), pick_rows),
            "each": {k: _measure(runs[k], pick_rows) for k in KEYS}}

    # ----- today -----
    high, total, limit = _flow_state(net, SUM_DAYS, PCT, WINDOW)
    s_high, growth, middle = _stable_state(supply, STABLE_DAYS)
    today = {"flow_day": total.last_valid_index(), "flow_high": float(high.dropna().iloc[-1]),
             "flow_sum": float(total.dropna().iloc[-1]), "flow_limit": float(limit.dropna().iloc[-1]),
             "balance_sum": float(cm["balance"].diff().rolling(SUM_DAYS).sum().dropna().iloc[-1]),
             "stable_day": growth.last_valid_index(), "stable_high": float(s_high.dropna().iloc[-1]),
             "growth": float(growth.dropna().iloc[-1] * 100), "middle": float(middle.dropna().iloc[-1] * 100),
             "supply": float(supply.dropna().iloc[-1]), "trend": float(target["trend"].iloc[-1]),
             "next": {k: target[k].iloc[-1] for k in KEYS}}

    return {"last": last, "close": close, "target": target, "runs": runs, "periods": periods, "grid": grid,
            "original": original,
            "grid_start": grid_start, "grid_hold": grid_hold, "same_day": same_day, "overlap": overlap, "luck": luck,
            "stable_variants": stable_variants, "pick": pick, "gaps": gaps, "live": live, "today": today,
            "flow_start": flow_start, "stable_start": stable_start, "cm_first": cm.index[0], "cm_last": cm.index[-1]}


# ---------- the study ----------

def study() -> dict:
    close = te._daily_close()
    cm = _fetch_cm()
    supply = _fetch_stable()
    r = _compute(close, cm, supply)
    revisions = _revisions(cm)
    last, oos, full = r["last"], r["periods"]["oos"], r["periods"]["full"]
    gaps, live, today = r["gaps"], r["live"], r["today"]
    flow_start, stable_start = r["flow_start"], r["stable_start"]
    partial = last < pd.Timestamp(f"{last.year}-12-31")

    def year_name(y: int) -> str:
        return f"{y}（到 {last.strftime('%m-%d')}）" if y == last.year and partial else str(y)

    # ----- table 1: the approaches -----
    def approach(key: str, m: dict, period: str) -> list:
        hold = m["hold"]
        same = key == "hold"
        return [NAMES[key], period, _pct(m["ret"]), _num(m["ratio"]), _pct(m["deepest"]), f"{m['share']:.0f}%",
                "—" if same else f"{_num(hold['ratio'])}｜{_pct(hold['deepest'])}",
                "—" if same else f"{m['n_years']} 年裡 {m['beat']} 年"]

    order = KEYS + ("hold_balance", "hold_flow_cut")
    rows1 = [approach(k, oos[k], "2022 起") for k in order]
    for k in order:
        start = stable_start if "stable" in k else flow_start
        rows1.append(approach(k, full[k], f"{start.strftime('%Y-%m')} 起"))

    # ----- table 2: the grid -----
    def cell(m: dict) -> str:
        return f"{_num(m['ratio'])}｜{_pct(m['deepest'])}"

    rows2 = []
    for window in GRID_WINDOWS:
        for days in GRID_SUMS:
            name = f"{days} 天合計，和過去 {window // 365} 年比"
            if (days, window) == (SUM_DAYS, WINDOW):
                name += "（原規則在 80% 那一格）"
            rows2.append([name] + [cell(r["grid"][(window, days, p)]["oos"]) for p in GRID_PCTS]
                         + [cell(r["grid"][(window, days, p)]["full"]) for p in GRID_PCTS])
    rows2.append([f"{SUM_DAYS} 天合計、過去 1 年，但假設當天收盤就拿得到當天的數字"]
                 + [cell(r["same_day"][p]["oos"]) for p in GRID_PCTS]
                 + [cell(r["same_day"][p]["full"]) for p in GRID_PCTS])
    rows2.append(["對照：一直抱著不動"] + [cell(r["grid_hold"]["oos"])] * 3 + [cell(r["grid_hold"]["full"])] * 3)
    cells = len(r["grid"])
    better_oos = sum(1 for c in r["grid"].values() if c["oos"]["ratio"] > r["grid_hold"]["oos"]["ratio"])
    better_full = sum(1 for c in r["grid"].values() if c["full"]["ratio"] > r["grid_hold"]["full"]["ratio"])
    base = r["grid"][(WINDOW, SUM_DAYS, PCT)]
    rank_oos = 1 + sum(1 for c in r["grid"].values() if c["oos"]["ratio"] > base["oos"]["ratio"] + 1e-12)
    rank_full = 1 + sum(1 for c in r["grid"].values() if c["full"]["ratio"] > base["full"]["ratio"] + 1e-12)
    low_oos = min(c["oos"]["ratio"] for c in r["grid"].values())
    high_oos = max(c["oos"]["ratio"] for c in r["grid"].values())
    left, right = r["grid"][(WINDOW, SUM_DAYS, GRID_PCTS[0])]["oos"], r["grid"][(WINDOW, SUM_DAYS, GRID_PCTS[-1])]["oos"]

    # ----- table 3: balance against flow -----
    rows3 = []
    for day, value in gaps["top"].items():
        written = cm["written"].get(day)
        rows3.append([_d(day), _btc(gaps["move"][day]), _btc(gaps["net"][day]), _btc(value),
                      _d(written) if pd.notna(written) else "—"])
    rows3.append([f"對照：{_d(gaps['live_start'])} 以前的 {gaps['before_days']:,} 天", "—", "—",
                  f"相差超過 {GAP_MIN:.0f} 顆的只有 {gaps['before_off']} 天",
                  (f"{_d(gaps['written_from'])} 到 {_d(gaps['written_to'])} 之間一次重算寫入"
                   if gaps["written_from"] is not None else "—")])

    # ----- table 4: pick in advance -----
    pick = r["pick"]
    rows4 = []
    for y in pick["years"]:
        got = pick["picked"]["years"].get(y)
        rows4.append([year_name(y), f"{SHORT[pick['chosen'][y]]}（{_pct(got)}）"]
                     + [_pct(pick["each"][k]["years"].get(y)) for k in KEYS])
    span = f"{PICK_FROM} 起"
    rows4.append([f"{span}換算每年報酬", _pct(pick["picked"]["ret"])] + [_pct(pick["each"][k]["ret"]) for k in KEYS])
    rows4.append([f"{span}報酬÷波動", _num(pick["picked"]["ratio"])] + [_num(pick["each"][k]["ratio"]) for k in KEYS])
    rows4.append([f"{span}最深曾經虧多少", _pct(pick["picked"]["deepest"])]
                 + [_pct(pick["each"][k]["deepest"]) for k in KEYS])
    n_pick = len(pick["years"])
    pick_beat = {k: sum(1 for y in pick["years"] if pick["each"][k]["years"].get(y, 0) > pick["each"]["hold"]["years"].get(y, 0) + 1e-9)
                 for k in KEYS}
    picked_beat = sum(1 for y in pick["years"] if pick["picked"]["years"].get(y, 0) > pick["each"]["hold"]["years"].get(y, 0) + 1e-9)
    rows4.append(["賺得比一直抱著多的年數", f"{n_pick} 年裡 {picked_beat} 年", "—"]
                 + [f"{n_pick} 年裡 {pick_beat[k]} 年" for k in KEYS[1:]])
    picks_text = "、".join(f"{y} 年挑到「{SHORT[pick['chosen'][y]]}」" for y in pick["years"])

    ov, ovf, luck = r["overlap"]["oos"], r["overlap"]["full"], r["luck"]
    span_oos = f"2022-01 到 {_d(last)}"
    tables = [
        {"caption": (f"六種做法加兩個重算，各看兩段期間：{span_oos}（原研究的檢驗期間），以及資料最早能算的那天起"
                     f"（存入過濾 {_d(flow_start)}、穩定幣過濾 {_d(stable_start)}）。「交易所淨存入」是存進交易所的比特幣減掉提走的；"
                     f"存入過濾＝最近 {SUM_DAYS} 天的淨存入高於它自己過去一年裡 {PCT * 100:.0f}% 的日子時，隔天空手。"
                     f"都是當天收盤決定、隔天才照做，資料晚一天才用，每次買賣扣交易金額的 {COST * 100:.2f}%，現金沒有利息。"
                     f"最後兩種做法是第 3 項檢查的重算：拿掉的日子在 2022 起這段共 {oos['removed']} 天。"),
         "columns": ["做法", "期間", "每年報酬", "報酬÷波動", "最深曾經虧多少", "持有比特幣的天數比例",
                     "同期間一直抱著（報酬÷波動｜最深虧）", "賺得比一直抱著多的年數"],
         "rows": rows1},
        {"caption": (f"存入過濾換參數（抱著＋過濾）。每格是「報酬÷波動｜最深曾經虧多少」。全期這一半統一從 {_d(r['grid_start'])} 算起，"
                     f"讓各格可以互相比。{cells} 格裡，2022 起有 {better_oos} 格的報酬÷波動比一直抱著高、全期有 {better_full} 格；"
                     f"原規則那一格在 {cells} 格裡排第 {rank_oos}（2022 起）和第 {rank_full}（全期）；2022 起各格從 {_num(low_oos)} 到 "
                     f"{_num(high_oos)}，原規則左右兩格（70%、90%）是 {_num(left['ratio'])} 和 {_num(right['ratio'])}，"
                     "各格高高低低，不是一片平穩的高原。"
                     "倒數第二列是把「資料晚一天才用」拿掉的對照：供應商大約在當天結束後兩三個小時才公布當天的數字，"
                     "所以那一列在實際上做不到，只是用來看晚一天差多少。"),
         "columns": ["設定", "2022 起：高於 70%", "2022 起：高於 80%", "2022 起：高於 90%", "全期：高於 70%",
                     "全期：高於 80%", "全期：高於 90%"],
         "rows": rows2},
        {"caption": (f"同一家供應商的兩個數字對不對得上：交易所餘額一天的變化，照理應該等於當天的淨存入（存入減提走）。"
                     f"這是相差最大的 {TOP_GAPS} 天，全部落在供應商開始「每天即時寫入」之後（{_d(gaps['live_start'])} 起）："
                     f"這 {gaps['after_days']} 天裡有 {gaps['after_off']} 天相差超過 {GAP_MIN:.0f} 顆，其中 {gaps['after_up']} 天是餘額"
                     f"增加得比淨存入多，累計多出 {_btc(gaps['after_sum'], signed=False)}。供應商沒有說明原因；比較可能的解釋是有地址"
                     "事後才被認定為交易所、它原本就有的幣直接加進餘額卻不算存入，或是即時版的兩個數字算法不同（這是推測）。"
                     "更早的歷史幾乎完全對得上，是因為那一整段是事後一次重算的。"),
         "columns": ["日期", "交易所餘額當天變化", "當天淨存入（存入−提走）", "兩者相差", "供應商寫入這筆資料的日期"],
         "rows": rows3},
        {"caption": (f"事先挑選：每年年初只看那一年以前的資料（從 {_d(pick['common_start'])} 六種都算得出來的那天起），"
                     "挑報酬÷波動最高的做法，照做一整年，再一年一年接起來。實際挑到的是：" + picks_text + "。"
                     "右邊六欄是每一種做法每一年各自的報酬，沒有挑選。"),
         "columns": ["年度", "事先挑到的（那年報酬）"] + [SHORT[k] for k in KEYS],
         "rows": rows4},
    ]

    # ----- today -----
    def state(v) -> str:
        if not te._ok(v):
            return "—"
        return "持有" if v > 0.5 else "空手"

    if revisions["compare"]:
        c = revisions["compare"][0]
        revision_value = (f"和 {c['against']} 存的版本比：重疊的 {c['days']:,} 天裡有 {c['changed']:,} 天的淨存入被改過，"
                          f"最大一筆改了 {_btc(c['largest'], signed=False)}"
                          + (f"（{_d(c['largest_day'])}）" if c["largest_day"] is not None else "")
                          + f"；照改過前後各算一次，存入過濾的狀態有 {c['flips']} 天不一樣")
    elif revisions["saved"]:
        revision_value = f"今天（{common.today_utc()}）第一次存檔，還沒有舊版本可以比；之後每次執行會自動比對"
    else:
        revision_value = "這次存檔沒有成功，無法比對"
    flow_on = today["flow_high"] > 0.5
    stable_on = today["stable_high"] > 0.5
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"存入過濾的最新狀態（用到 {_d(today['flow_day'])} 的資料）",
         "value": ("開：淨存入偏高" if flow_on else "關：淨存入沒有偏高")},
        {"label": f"最近 {SUM_DAYS} 天交易所淨存入（存入減提走）",
         "value": f"{_btc(today['flow_sum'])}；門檻（過去一年的 {PCT * 100:.0f}% 位置）{_btc(today['flow_limit'])}"},
        {"label": f"同樣 {SUM_DAYS} 天，交易所餘額的變化（照理應該和上一列一樣）",
         "value": f"{_btc(today['balance_sum'])}，和淨存入相差 {_btc(today['balance_sum'] - today['flow_sum'])}"},
        {"label": f"穩定幣總供給 {STABLE_DAYS} 天成長（到 {_d(today['stable_day'])}）",
         "value": (f"{today['growth']:+.2f}%；過去一年的中位數 {today['middle']:+.2f}%——"
                   + ("高於中位數" if stable_on else "沒有高於中位數"))},
        {"label": "順勢規則（20 日新高／10 日新低）目前的狀態", "value": state(today["trend"])},
        {"label": f"各做法照規則在 {_d(last + pd.Timedelta(days=1))} 這一天的狀態（只是指標狀態；資料晚一天才用）",
         "value": "；".join(f"{SHORT[k]}：{state(today['next'][k])}" for k in KEYS if k != "hold")},
        {"label": "歷史資料有沒有被改寫（存檔比對）", "value": revision_value},
    ]

    # ----- verdict -----
    hf, h, t, tf = oos["hold_flow"], oos["hold"], oos["trend"], oos["trend_flow"]
    hf_all, h_all, t_all, tf_all = full["hold_flow"], full["hold"], full["trend"], full["trend_flow"]
    hs, ts = oos["hold_stable"], oos["trend_stable"]
    add_oos, add_full = tf["ratio"] - t["ratio"], tf_all["ratio"] - t_all["ratio"]
    if add_oos > 0.05 and add_full > 0.05:
        add_words = "兩段期間都多出一點"
    elif add_oos < -0.05 and add_full < -0.05:
        add_words = "兩段期間反而都變差"
    elif abs(add_oos) <= 0.05 and abs(add_full) <= 0.05:
        add_words = "幾乎沒有多出東西"
    else:
        add_words = "兩段期間方向不一致，不能算有多出東西"
    clean_history = gaps["before_off"] <= 10 and gaps["after_off"] > 0.5 * max(gaps["after_days"], 1)
    if clean_history:
        trust = (f"兩個資料序列在 {_d(gaps['live_start'])} 以前的 {gaps['before_days']:,} 天幾乎完全對得上（只有 {gaps['before_off']} 天"
                 f"相差超過 {GAP_MIN:.0f} 顆），但那一整段是供應商在 {_d(gaps['written_from'])} 到 {_d(gaps['written_to'])} 之間一次重算"
                 f"寫入的；改成每天即時寫入之後的 {gaps['after_days']} 天裡有 {gaps['after_off']} 天對不上、餘額累計比淨存入多出"
                 f" {_btc(gaps['after_sum'], signed=False)}，代表拿來回測的歷史是事後整理過的版本、不是當時看得到的數字，"
                 "所以這個結果目前不能當成即時做得到的成績，要等之後每天存檔比對才知道差多少")
    else:
        trust = (f"兩個資料序列在即時寫入之前有 {gaps['before_off']} 天、之後有 {gaps['after_off']} 天對不上，"
                 "交易所地址又是事後才認定的，所以這個結果不能當成即時做得到的成績")
    g70 = r["grid"][(WINDOW, SUM_DAYS, GRID_PCTS[0])]["oos"]
    og, og_hold = r["original"]["hold_flow"], r["original"]["hold"]
    repro = (f"算到原研究當時（{ORIGINAL_END}）、用它的算法是 {_num(og['sharpe'])} 對 {_num(og_hold['sharpe'])}、每年 {_pct(og['ret'])} 對 "
             f"{_pct(og_hold['ret'])}、最深虧 {_pct(og['deepest'])} 對 {_pct(og_hold['deepest'])}、{og['share']:.0f}% 的日子持有"
             "（原研究報的是 0.83 對 0.50、+30.6% 對 +13.1%、-44% 對 -67%、83%）")
    verdict = (
        f"照原規則重做（最近 {SUM_DAYS} 天交易所淨存入高於過去一年 {PCT * 100:.0f}% 的日子就空手一天）：{span_oos} 報酬÷波動 "
        f"{_num(hf['ratio'])}（一直抱著 {_num(h['ratio'])}）、最深虧 {_pct(hf['deepest'])}（抱著 {_pct(h['deepest'])}）、"
        f"{hf['share']:.0f}% 的日子持有，隨機平移 {SIMS:,} 次只有 {luck['oos']['as_good']:.1f}% 一樣好或更好，{repro}；但從 {_d(flow_start)} 算起是 {_num(hf_all['ratio'])}、"
        f"{'輸給' if hf_all['ratio'] < h_all['ratio'] else '高於'}一直抱著的 {_num(h_all['ratio'])}"
        f"（平移後有 {luck['full']['as_good']:.1f}% 一樣好），門檻從 80% 改成 70% 在 2022 起就只剩 {_num(g70['ratio'])}。"
        f"加在順勢規則上{add_words}：2022 起順勢規則的報酬÷波動 {_num(t['ratio'])}、最深虧 {_pct(t['deepest'])}，加上存入過濾是 "
        f"{_num(tf['ratio'])}、{_pct(tf['deepest'])}，全期卻是 {_num(t_all['ratio'])} 變 {_num(tf_all['ratio'])}；"
        f"存入過濾說空手的日子裡有 {_num(ov['share'], 0)}% 順勢規則本來就空手（順勢規則平常就有 {ov['trend_out_share']:.0f}% 的日子空手）；"
        f"穩定幣過濾 2022 起單獨是 {_num(hs['ratio'])}、加在順勢規則上是 {_num(ts['ratio'])}，但把 7 天成長換成 30 天就變成 "
        f"{_num(r['stable_variants'][30]['trend']['oos']['ratio'])}。"
        f"{trust}。"
    )

    # ----- notes -----
    def variant(days: int) -> str:
        v = r["stable_variants"][days]
        return (f"{days} 天成長：2022 起單獨 {_num(v['hold']['oos']['ratio'])}（最深虧 {_pct(v['hold']['oos']['deepest'])}）、"
                f"加在順勢規則上 {_num(v['trend']['oos']['ratio'])}（{_pct(v['trend']['oos']['deepest'])}）；"
                f"{v['start'].strftime('%Y-%m')} 起單獨 {_num(v['hold']['full']['ratio'])}、加在順勢規則上 "
                f"{_num(v['trend']['full']['ratio'])}")

    def per_day(v) -> str:
        return "—" if not te._ok(v) else f"{v:+.2f}%"

    cut_o, cut_f, bal_o, bal_f = oos["hold_flow_cut"], full["hold_flow_cut"], oos["hold_balance"], full["hold_balance"]
    same_series = abs(bal_o["ratio"] - hf["ratio"]) <= 0.05 and abs(bal_f["ratio"] - hf_all["ratio"]) <= 0.05
    notes = [
        "最重要的限制：哪些地址屬於交易所，是資料供應商事後才認定的，認定之後會回頭改寫整段歷史。所以回測用的「過去的淨存入」"
        "是今天整理過的版本，當時即時看到的數字可能不一樣，過去的結果可能比當時真正做得到的好看。",
        f"供應商自己的時間戳記顯示：{_d(r['cm_first'])} 到 {_d(gaps['live_start'] - pd.Timedelta(days=1))} 的 {gaps['rewritten']:,} 天資料，"
        + (f"全部是在 {_d(gaps['written_from'])} 到 {_d(gaps['written_to'])} 之間寫入的（一次重算）；" if gaps["written_from"] is not None else "")
        + f"{_d(gaps['live_start'])} 起才是每天隔日寫入。原研究（2026-10-14 做的）和這裡用的 2022 年以後的資料，幾乎全部屬於重算的那一段。",
        f"兩個序列換著用的結果：{'在這次的資料裡幾乎一樣' if same_series else '不一樣，結果取決於用哪一個序列'}——用餘額變化代替淨存入，"
        f"2022 起報酬÷波動 {_num(bal_o['ratio'])}（用淨存入是 {_num(hf['ratio'])}），全期 {_num(bal_f['ratio'])}（{_num(hf_all['ratio'])}）。"
        f"但這個比較能說明的很有限：重算過的歷史裡兩個序列本來就幾乎一模一樣，只有即時寫入後的 {live['days']} 天不同；"
        f"這 {live['days']} 天裡，用淨存入算有 {live['flow_out']} 天叫空手、用餘額變化算有 {live['balance_out']} 天，"
        f"兩者同一天狀態相同的比例是 {_num(live['same'], 0)}%。",
        f"拿掉 {TOP_GAPS} 次最大對不上日子的前後各 {AROUND} 天（2022 起共 {oos['removed']} 天）之後：2022 起報酬÷波動 "
        f"{_num(cut_o['ratio'])}（同樣日子一直抱著 {_num(cut_o['hold']['ratio'])}），全期 {_num(cut_f['ratio'])}"
        f"（{_num(cut_f['hold']['ratio'])}）。因為這些日子全部在最近幾個月，拿掉它們等於只拿掉檢驗期間的尾巴。",
        f"和順勢規則重疊多少：2022 起存入過濾叫空手的 {ov['flow_out']} 天裡，{ov['both']} 天順勢規則本來就空手，"
        f"真正多擋掉的是 {ov['extra']} 天，這些天比特幣平均一天 {per_day(ov['in_flow_out'])}（順勢規則持有、過濾沒擋的日子平均 "
        f"{per_day(ov['in_flow_ok'])}）；順勢規則已經空手時，過濾叫空手的日子平均一天 {per_day(ov['out_flow_out'])}、沒叫的 "
        f"{per_day(ov['out_flow_ok'])}。全期（{_d(flow_start)} 起）：{ovf['flow_out']} 天裡 {ovf['both']} 天重疊"
        f"（{_num(ovf['share'], 0)}%，順勢規則平常空手 {ovf['trend_out_share']:.0f}%），多擋掉的 {ovf['extra']} 天平均一天 "
        f"{per_day(ovf['in_flow_out'])}（沒擋的 {per_day(ovf['in_flow_ok'])}）。",
        "穩定幣過濾（穩定幣總供給最近 7 天的成長率高於它自己過去一年的中位數才持有）換成 " + "；".join(variant(d) for d in STABLE_VARIANTS)
        + f"。同樣從 {stable_start.strftime('%Y-%m')} 起，順勢規則單獨是 {_num(full['trend_since_stable']['ratio'])}、"
        f"加上 7 天穩定幣過濾是 {_num(full['trend_stable']['ratio'])}。"
        "原研究自己對這條的評語是偏弱（p 約 0.10）、搬到別的幣不成立。DefiLlama 的穩定幣歷史同樣會在收錄新幣時回頭補資料。",
        f"運氣檢查的做法：把「抱著＋存入過濾」每天持有或空手的時間表，整段繞著期間隨機平移（至少 30 天）{SIMS:,} 次，"
        f"持有的天數和進出次數都不變、只有時間點錯開，再算一樣好或更好的比例：2022 起 {luck['oos']['as_good']:.1f}%"
        f"（平移後的中間值 {_num(luck['oos']['median'])}，實際 {_num(luck['oos']['real'])}），全期 {luck['full']['as_good']:.1f}%"
        f"（{_num(luck['full']['median'])}，實際 {_num(luck['full']['real'])}）。原研究報的是 1.0%。",
        f"重做對不對得上原研究：{repro}。表裡的數字和它不同有兩個原因：這裡的「報酬÷波動」是每年複利報酬除以每年波動"
        "（原研究是每天平均報酬除以每天起伏再換算成一年，數字會比較高），而且這裡算到最新一天，比原研究多了一個多月。"
        f"共同的做法：資料一律晚一天才用（供應商在當天結束後約兩三小時才公布）、每次買賣扣 {COST * 100:.2f}%、門檻只用更早的 {WINDOW} 個數字算；"
        "價格是幣安現貨日線收盤（2017-08 以前是 blockchain.info）。",
        "存檔比對（修訂檢查）：每次執行會把當天抓到的整段流量資料存一份，之後再和舊的比，數有幾天的數字被改過、改了多少。"
        + ("目前只有今天這一份，要等之後再執行才有結果。" if not revisions["compare"] else
           "；".join(f"和 {c['against']} 的版本比，{c['days']:,} 天裡淨存入改了 {c['changed']:,} 天"
                     f"（最早被改的是 {_d(c['earliest']) if c['earliest'] is not None else '—'}，合計改動 {_btc(c['total'], signed=False)}），"
                     f"餘額改了 {c['balance_changed']:,} 天，存入過濾的狀態有 {c['flips']} 天不同"
                     for c in revisions["compare"]) + "。"),
        "只有一家資料供應商（Coin Metrics 免費版），它認得的交易所地址不是全部，也沒有別家可以交叉比對；早年（2011–2013）"
        "被認得的交易所很少，那幾年的數字參考價值更低。",
        f"年數很少：原研究的檢驗期間到現在不到五年，其實就是一輪下跌加一輪上漲；事先挑選那張表只有 {n_pick} 個年度，"
        "「幾年裡幾年」多一年少一年就會差很多。參數表也是事後才看的，哪一格最好不能當成事先知道。",
        f"成本只用一個粗略的數字（每次買賣扣交易金額的 {COST * 100:.2f}%），沒有算滑價、稅和現金利息；一直抱著沒有扣成本。"
        "事先挑選那張表在換做法的年初也照樣扣買賣成本。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、資料可不可靠，不是買賣建議；上面的「狀態」只是指標狀態。",
    ]

    return {
        "id": "flow_filters",
        "title": "交易所存入和穩定幣成長，能不能幫順勢規則加分",
        "question": "別人預先登記、號稱通過檢驗的「交易所淨存入偏高就空手」和「穩定幣供給成長才持有」，加在已經有效的順勢規則上有沒有多出東西？"
                    "交易所流量資料會被事後改寫，這個結果能不能信？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 免費社群資料：比特幣交易所存入、提走、餘額（FlowInExNtv、FlowOutExNtv、SplyExNtv）",
             "url": "https://docs.coinmetrics.io/info/metrics/flowinexntv"},
            {"name": "DefiLlama 穩定幣總供給", "url": flows_collector.LLAMA_STABLE_PAGE},
            {"name": "原研究：f1devbin/Crypto-Research-Labs（H-020、H-006）",
             "url": "https://github.com/f1devbin/Crypto-Research-Labs/blob/main/docs/FINDINGS.md"},
            {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
