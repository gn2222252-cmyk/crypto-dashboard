"""Historical study "hash_ribbons": the Hash Ribbons buy signal (Charles Edwards, Capriole, 2019), listed one by one.

Rule as commonly cited: the 30-day simple average of the network hash rate below its 60-day simple average = miners
capitulating. The buy signal is the first day the 30-day average is back above the 60-day average AND the 10-day
simple average of price is above its 20-day simple average. There is no sell rule, so the signal is judged by what
price did afterwards, against all days, against days in a deep drawdown and against random dates of the same years.

No look-ahead inside the rule: a signal on day t uses hash rate and closes up to and including day t; forward
results start from the close of day t; the trading rules hold from day t+1. The hash rate itself is today's final
estimate (it is an estimate from block times and differs between providers), so dates can differ by days from what
was visible live. Everything comes from public, key-less endpoints through common.get_json (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic

BCI_HASH = "https://api.blockchain.info/charts/hash-rate"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
SINCE = "2011-01-01"
PUBLISHED = "2019-09-01"  # the article could not be opened by this program; 2019-09 is the commonly cited date
HORIZONS = (30, 90, 180, 365)
LOW_WINDOW = 180
DEEP = -0.30  # "deep drawdown": the close is at least 30% below the highest close so far
DRAWS = 2000
SEED = 20190901
COST = 0.001
YEAR = 365
NEAR = 7  # a variation's signal within this many days of a base signal counts as "the same signal, moved"
MAX_ROWS = 20


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is a fraction (0.12 = 12%)."""
    if not _ok(v):
        return "—"
    v = float(v) * 100
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _usd(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return f"{v:,.0f}" if v >= 100 else f"{v:,.2f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _median(values) -> float | None:
    values = [float(v) for v in values if _ok(v)]
    return float(np.median(values)) if values else None


def _wins(values) -> str:
    values = [float(v) for v in values if _ok(v)]
    return f"{len(values)} 次裡 {sum(1 for v in values if v > 0)} 次" if values else "—"


# ---------- data ----------

def _daily(series: pd.Series) -> pd.Series:
    series = pd.Series(series.values.astype(float), index=pd.to_datetime(series.index)).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series[series.index < pd.Timestamp(common.today_utc())]  # finished days only
    return series[series > 0].asfreq("D").ffill()


def _close() -> pd.Series:
    close = _daily(classic._long_close())
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close


def _hash_blockchain() -> pd.Series:
    """blockchain.info estimated hash rate, one raw value per day (TH/s)."""
    reply = common.get_json(BCI_HASH, {"timespan": "all", "format": "json", "sampled": "false"}, timeout=60)
    series = pd.Series({pd.Timestamp(int(p["x"]), unit="s").normalize(): float(p["y"]) for p in reply["values"]})
    if len(series) < 3000:
        raise RuntimeError(f"blockchain.info 算力只回傳 {len(series)} 筆")
    return _daily(series)


def _hash_coinmetrics() -> pd.Series:
    """CoinMetrics community HashRate, one value per day (TH/s)."""
    reply = common.get_json(CM_URL, {"assets": "btc", "metrics": "HashRate", "frequency": "1d", "page_size": 10000},
                            timeout=60)
    data = list(reply["data"])
    for _ in range(5):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=60)
        data += reply["data"]
    rows = {}
    for item in data:
        try:
            rows[pd.Timestamp(item["time"][:10])] = float(item["HashRate"])
        except (KeyError, TypeError, ValueError):
            continue
    series = pd.Series(rows)
    if len(series) < 3000:
        raise RuntimeError(f"CoinMetrics 算力只回傳 {len(series)} 筆")
    return _daily(series)


# ---------- the rule ----------

def _signals(close: pd.Series, hashrate: pd.Series, fast: int = 30, slow: int = 60, confirm: bool = True) -> dict:
    """Signals of the rule on the days both series exist. Returns the events and the daily state series."""
    index = close.index.intersection(hashrate.index)
    close, hashrate = close.reindex(index), hashrate.reindex(index)
    h_fast = hashrate.rolling(fast, min_periods=fast).mean()
    h_slow = hashrate.rolling(slow, min_periods=slow).mean()
    p10 = close.rolling(10, min_periods=10).mean()
    p20 = close.rolling(20, min_periods=20).mean()
    known = h_slow.notna() & p20.notna()
    below = ((h_fast < h_slow) & known).values
    price_ok = ((p10 > p20) & known).values if confirm else known.values
    known_v = known.values
    fast_v = h_fast.values

    events, flags = [], np.zeros(len(index), dtype=bool)
    start, crossed, waiting, was_below = None, None, False, False
    for i in range(len(index)):
        if not known_v[i]:
            continue
        if below[i]:
            if not waiting:
                start, waiting = i, True  # an unanswered capitulation keeps its first day
            crossed = None
            was_below = True
            continue
        if waiting:
            if was_below:
                crossed = i
            if price_ok[i]:
                trough = start + int(np.nanargmin(fast_v[start:i + 1]))
                before = fast_v[max(0, start - 90):trough + 1]
                peak = float(np.nanmax(before)) if np.isfinite(before).any() else float("nan")
                events.append({"start": index[start], "cross": index[crossed if crossed is not None else i],
                               "date": index[i], "i": i, "below_days": int(below[start:i + 1].sum()),
                               "fall": float(fast_v[trough] / peak - 1) if peak and math.isfinite(peak) else None})
                flags[i] = True
                waiting, start, crossed = False, None, None
        was_below = False
    return {"events": events, "signal": pd.Series(flags, index=index), "below": pd.Series(below, index=index),
            "h_fast": h_fast, "h_slow": h_slow, "p10": p10, "p20": p20, "close": close, "known": known}


def _forward(close: pd.Series) -> dict:
    """Result n days later for every day (NaN when those days have not happened yet), and the lowest point."""
    out = {n: close.shift(-n) / close - 1 for n in HORIZONS}
    low = close[::-1].rolling(LOW_WINDOW, min_periods=LOW_WINDOW).min()[::-1].shift(-1)  # days t+1 .. t+180
    out["low"] = low / close - 1
    out["dd"] = close / close.cummax() - 1
    return out


def _matched_random(dates: list, fwd: pd.Series, rng) -> dict:
    """Median result of the dates against DRAWS sets of random dates, one per signal from the same calendar year."""
    valid = fwd.dropna()
    valid = valid[valid.index >= pd.Timestamp(SINCE)]
    by_year = {year: chunk.values for year, chunk in valid.groupby(valid.index.year)}
    used = [d for d in dates if d in valid.index and d.year in by_year]
    if len(used) < 3:
        return {"n": len(used), "median": None, "random": None, "better": None}
    actual = float(np.median([valid[d] for d in used]))
    draws = np.stack([rng.choice(by_year[d.year], size=DRAWS) for d in used], axis=1)
    medians = np.median(draws, axis=1)
    return {"n": len(used), "median": actual, "random": float(np.median(medians)),
            "better": float((medians < actual).mean() * 100)}


def _measure(net: pd.Series, position: pd.Series) -> dict:
    values = net.values
    n = len(values)
    growth = float(np.prod(1 + values))
    ret = growth ** (YEAR / n) - 1 if n > 1 and growth > 0 else None
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min())
    return {"ret": ret, "deepest": deepest, "share": float(position.mean()), "total": growth - 1,
            "trades": int((position.diff().abs() > 0).sum())}


def _trading(close: pd.Series, base: dict) -> dict:
    """Long-or-cash versions of the signal against buy-and-hold and the 20/10-day channel rule, after costs."""
    index = base["signal"].index
    close = close.reindex(index)
    signal, below = base["signal"], base["below"]
    hold_year = signal.astype(float).rolling(YEAR, min_periods=1).max()  # a new signal restarts the 365 days
    state = np.zeros(len(index))
    held = 0.0
    for i, (fired, cap) in enumerate(zip(signal.values, below.values)):
        if cap:
            held = 0.0
        elif fired:
            held = 1.0
        state[i] = held
    until_cap = pd.Series(state, index=index)
    high, low = close.rolling(20, min_periods=20).max(), close.rolling(10, min_periods=10).min()
    channel = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=index)
    channel = channel.where(high.notna()).ffill().fillna(0.0)
    rules = {"year": hold_year, "cap": until_cap, "trend": channel, "hold": pd.Series(1.0, index=index)}
    change = close.pct_change().fillna(0.0)
    out = {}
    for period, first in (("2018", "2018-01-01"), ("2012", "2012-01-01")):
        out[period] = {}
        for key, rule in rules.items():
            position = rule.shift(1).fillna(0.0)  # decided at the close of day t, held during day t+1
            turn = position.diff().abs().fillna(0.0)
            net = position * change - (0.0 if key == "hold" else COST) * turn
            rows = net.index >= pd.Timestamp(first)
            out[period][key] = _measure(net[rows], position[rows])
    return out


def _compare(base_events: list, other_events: list) -> dict:
    base = [e["date"] for e in base_events]
    other = [e["date"] for e in other_events]
    same = sum(1 for d in base if d in other)
    near = sum(1 for d in base if any(abs((d - o).days) <= NEAR for o in other))
    extra = sum(1 for o in other if not any(abs((d - o).days) <= NEAR for d in base))
    return {"n": len(other), "same": same, "near": near, "missing": len(base) - near, "extra": extra}


# ---------- the study ----------

def study() -> dict:
    close_all = _close()
    problems: list[str] = []
    sources_ok: dict[str, pd.Series] = {}
    for name, fetch in (("blockchain", _hash_blockchain), ("coinmetrics", _hash_coinmetrics)):
        try:
            sources_ok[name] = fetch()
        except Exception as exc:  # noqa: BLE001 - one source is enough for the study
            label = "blockchain.info" if name == "blockchain" else "CoinMetrics 社群版"
            problems.append(f"{label} 的算力資料這次取不到（{type(exc).__name__}），沒有改用其他方法繞過，"
                            "所以這次沒有兩個來源互相對照。")
    if not sources_ok:
        raise RuntimeError("兩個算力來源都取不到")
    main_name = "blockchain" if "blockchain" in sources_ok else "coinmetrics"
    main_label = "blockchain.info" if main_name == "blockchain" else "CoinMetrics 社群版"
    hashrate = sources_ok[main_name]

    base = _signals(close_all, hashrate)
    close = base["close"]
    last = close.index[-1]
    fwd = _forward(close)
    since, published = pd.Timestamp(SINCE), pd.Timestamp(PUBLISHED)

    def pick(run: dict) -> list:
        return [e for e in run["events"] if e["date"] >= since]

    def enrich(events: list) -> list:
        for e in events:
            d = e["date"]
            e["price"] = float(close[d])
            e["dd"] = float(fwd["dd"][d])
            for n in HORIZONS:
                e[n] = float(fwd[n][d]) if _ok(fwd[n][d]) else None
            e["low"] = float(fwd["low"][d]) if _ok(fwd["low"][d]) else None
            if e["low"] is None and d < last:  # fewer than 180 days have passed: the lowest point so far
                e["low_so_far"] = float(close[close.index > d].min() / close[d] - 1)
            e["after"] = d >= published
        return events

    events = enrich(pick(base))
    after = [e for e in events if e["after"]]
    before = [e for e in events if not e["after"]]
    if not events:
        raise RuntimeError("沒有算出任何訊號")

    # ----- table 1: every signal -----
    def pair(e: dict, a: int, b: int) -> str:
        return f"{_pct(e[a])}／{_pct(e[b])}"

    def signal_row(e: dict) -> list:
        low = _pct(e["low"]) if e["low"] is not None else (
            f"{_pct(e['low_so_far'])}（未滿 {LOW_WINDOW} 天）" if "low_so_far" in e else "—")
        return [_d(e["start"]), _d(e["date"]) + ("（公開後）" if e["after"] else ""), _usd(e["price"]),
                _pct(e["fall"]), _pct(e["dd"]) if e["dd"] < -0.0005 else "正在創新高", pair(e, 30, 90),
                pair(e, 180, 365), low]

    shown = events if len(events) <= MAX_ROWS else events[-MAX_ROWS:]
    hidden = len(events) - len(shown)
    signal_rows = [signal_row(e) for e in shown]

    # ----- table 2: against chance -----
    rng = np.random.default_rng(SEED)
    days = close.index[(close.index >= since) & base["known"].reindex(close.index).fillna(False).values]
    deep_days = days[fwd["dd"][days].values <= DEEP]

    def group_row(name: str, idx, count_text: str) -> list:
        f180, f365, low = fwd[180][idx].dropna(), fwd[365][idx].dropna(), fwd["low"][idx].dropna()
        share = lambda s: f"{(s > 0).mean() * 100:.0f}%" if len(s) else "—"  # noqa: E731
        return [name, count_text, _pct(_median(f180)), share(f180), _pct(_median(f365)), share(f365),
                _pct(_median(low))]

    def event_row(name: str, chosen: list) -> list:
        return [name, f"{len(chosen)} 次", _pct(_median(e[180] for e in chosen)), _wins(e[180] for e in chosen),
                _pct(_median(e[365] for e in chosen)), _wins(e[365] for e in chosen),
                _pct(_median(e["low"] for e in chosen))]

    deep_events = [e for e in events if e["dd"] <= DEEP]
    after_days = days[days >= published]
    after_deep = deep_days[deep_days >= published]
    rand = {key: {n: _matched_random([e["date"] for e in chosen], fwd[n], rng) for n in (180, 365)}
            for key, chosen in (("all", events), ("after", after))}

    def rand_row(name: str, key: str) -> list:
        r180, r365 = rand[key][180], rand[key][365]
        return [name, f"{DRAWS} 組", _pct(r180["random"]), "—", _pct(r365["random"]), "—", "—"]

    def better(r: dict) -> str:
        return "—" if r["better"] is None else f"比 100 組隨機日期裡的 {r['better']:.0f} 組好"

    chance_rows = [
        event_row("全部訊號", events),
        event_row("其中：當天比歷史最高價低 30% 以上的訊號", deep_events),
        group_row(f"對照：{SINCE[:4]} 年以來的每一天", days, f"{len(days):,} 天"),
        group_row("對照：比歷史最高價低 30% 以上的每一天", deep_days, f"{len(deep_days):,} 天"),
        rand_row("對照：和每次訊號同一年的隨機日期（各組的中位數再取中間值）", "all"),
        event_row(f"公開後（{PUBLISHED[:7]} 起）的訊號", after),
        group_row(f"對照：{PUBLISHED[:7]} 以來的每一天", after_days, f"{len(after_days):,} 天"),
        group_row(f"對照：{PUBLISHED[:7]} 以來比歷史最高價低 30% 以上的每一天", after_deep, f"{len(after_deep):,} 天"),
        rand_row("對照：和公開後每次訊號同一年的隨機日期", "after"),
    ]

    # ----- table 3: variations -----
    def variation(name: str, run: dict) -> tuple[list, dict]:
        chosen = enrich(pick(run))
        c = _compare(events, chosen)
        return ([name, f"{c['n']} 次", f"{len(events)} 次裡 {c['same']} 次", f"{len(events)} 次裡 {c['near'] - c['same']} 次",
                 f"{c['missing']} 次／{c['extra']} 次", _pct(_median(e[180] for e in chosen)),
                 _pct(_median(e[365] for e in chosen)), _wins(e[365] for e in chosen)], {**c, "events": chosen})

    variants = [("沒有價格確認（30 日算力均線一回到 60 日之上就算）", _signals(close_all, hashrate, confirm=False)),
                ("算力改用 20 日／40 日均線", _signals(close_all, hashrate, 20, 40)),
                ("算力改用 45 日／90 日均線", _signals(close_all, hashrate, 45, 90))]
    other_name = "coinmetrics" if main_name == "blockchain" else "blockchain"
    other_label = "CoinMetrics 社群版" if other_name == "coinmetrics" else "blockchain.info"
    if other_name in sources_ok:
        variants.append((f"原規則，算力改用 {other_label} 的資料", _signals(close_all, sources_ok[other_name])))
    variation_rows = [[f"原規則（{main_label} 算力，30 日／60 日，要價格確認）", f"{len(events)} 次", "—", "—", "—",
                       _pct(_median(e[180] for e in events)), _pct(_median(e[365] for e in events)),
                       _wins(e[365] for e in events)]]
    variation_stats = {}
    for name, run in variants:
        row, stats = variation(name, run)
        variation_rows.append(row)
        variation_stats[name] = stats
    source_stats = variation_stats.get(variants[-1][0]) if other_name in sources_ok else None

    # ----- table 4: as a trading rule -----
    trade = _trading(close, base)
    first_day = {"2018": "2018-01-01", "2012": "2012-01-01"}
    names = {"year": "訊號出現就買，抱 365 天（期間又有訊號就重新起算）", "cap": "訊號出現就買，抱到下一次礦工投降開始",
             "trend": "儀表板的順勢規則（創 20 日收盤新高買、創 10 日收盤新低賣），用同一份價格重算",
             "hold": "一直抱著不動"}
    trade_rows = []
    for period in ("2018", "2012"):
        for key in ("year", "cap", "trend", "hold"):
            m = trade[period][key]
            trade_rows.append([f"{first_day[period][:4]}-01 到 {_d(last)}", names[key], _pct(m["ret"]),
                               _pct(m["deepest"]), f"{m['share'] * 100:.0f}%",
                               "—" if key == "hold" else f"{m['trades']} 次"])

    # ----- today -----
    h_fast, h_slow = float(base["h_fast"].iloc[-1]), float(base["h_slow"].iloc[-1])
    gap = h_fast / h_slow - 1
    below_now = bool(base["below"].iloc[-1])
    state_series = base["below"][base["known"]]
    flips = state_series.ne(state_series.shift()).cumsum()
    run_start = state_series[flips == flips.iloc[-1]].index[0]
    run_days = (last - run_start).days + 1
    p10, p20 = float(base["p10"].iloc[-1]), float(base["p20"].iloc[-1])
    price_ok_now = p10 > p20
    last_event = base["events"][-1] if base["events"] else None
    waiting_now = False  # crossed back up but the price condition has not been met yet
    if not below_now:
        earlier_below = state_series[state_series.index < run_start]
        if len(earlier_below) and (last_event is None or last_event["date"] < run_start):
            waiting_now = True
    month_ago = close.iloc[-43] if len(close) > 43 else None
    hash_unit = 1e6  # TH/s -> EH/s
    if below_now:
        state_text = f"是，30 日平均低於 60 日平均（低 {abs(gap) * 100:.1f}%）"
        since_text = f"{_d(run_start)} 起，已經 {run_days} 天"
        need = ("要先等算力的 30 日平均回到 60 日平均之上；回到之上的那天（或之後第一天）價格的 10 日平均也高於 20 日平均，"
                f"才算出現訊號。價格這一項目前{'已經符合' if price_ok_now else '還不符合'}。")
    elif waiting_now:
        state_text = f"不是，30 日平均已經回到 60 日平均之上（高 {gap * 100:.1f}%），但訊號還沒出現"
        since_text = f"{_d(run_start)} 回到之上，已經 {run_days} 天"
        need = "只差價格確認：價格的 10 日平均要高於 20 日平均；在那之前如果算力的 30 日平均又掉回 60 日之下，就重新等。"
    else:
        state_text = f"不是，30 日平均高於 60 日平均（高 {gap * 100:.1f}%）"
        since_text = f"{_d(run_start)} 起一直在上方，已經 {run_days} 天"
        need = ("目前沒有等待中的訊號。要先出現一次投降（算力 30 日平均跌到 60 日平均之下），之後 30 日平均再回到 60 日之上、"
                "而且價格的 10 日平均高於 20 日平均，才會有下一次訊號。")
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）",
         "value": f"收盤 {_usd(close.iloc[-1])} 美元，比歷史最高收盤價低 {abs(float(fwd['dd'].iloc[-1])) * 100:.1f}%"
                  + (f"，過去六週 {_pct(float(close.iloc[-1] / month_ago - 1))}" if month_ago else "")},
        {"label": "礦工現在是不是在「投降」", "value": state_text},
        {"label": "這個狀態從什麼時候開始", "value": since_text},
        {"label": "算力平均值", "value": f"30 日平均 {h_fast / hash_unit:,.0f} EH/s、60 日平均 {h_slow / hash_unit:,.0f} EH/s"},
        {"label": "價格確認（10 日平均是否高於 20 日平均）",
         "value": f"{'是' if price_ok_now else '否'}（10 日平均 {_usd(p10)}、20 日平均 {_usd(p20)} 美元）"},
        {"label": "最近一次訊號", "value": (f"{_d(last_event['date'])}（價格 {_usd(close[last_event['date']])} 美元，"
                                      f"到現在 {_pct(float(close.iloc[-1] / close[last_event['date']] - 1))}）"
                                      if last_event else "—")},
        {"label": "下一次訊號要等什麼（只是指標的狀態，不是買賣建議）", "value": need},
    ]

    # ----- verdict -----
    def count_up(chosen: list, n: int) -> tuple[int, int]:
        values = [e[n] for e in chosen if e[n] is not None]
        return len(values), sum(1 for v in values if v > 0)

    n365, up365 = count_up(events, 365)
    a365, aup365 = count_up(after, 365)
    sig180, sig365 = _median(e[180] for e in events), _median(e[365] for e in events)
    deep180, deep365 = _median(fwd[180][deep_days].dropna()), _median(fwd[365][deep_days].dropna())
    aft180, aft365 = _median(e[180] for e in after), _median(e[365] for e in after)
    adeep180, adeep365 = _median(fwd[180][after_deep].dropna()), _median(fwd[365][after_deep].dropna())
    aall365 = _median(fwd[365][after_days].dropna())

    def versus(a, b) -> str:
        if a is None or b is None:
            return "無法比較"
        return "比較高" if a > b else ("比較低" if a < b else "一樣")

    pending = len(events) - n365
    verdict = (
        f"{SINCE[:4]} 年以來照這條規則一共出現 {len(events)} 次訊號，滿一年的 {n365} 次裡 {up365} 次一年後是漲的"
        f"（一年後的中位數 {_pct(sig365)}，半年後 {_pct(sig180)}）"
        + (f"，還有 {pending} 次未滿一年" if pending else "") + "。"
        f"和「價格比歷史最高價低 30% 以上的任何一天」相比（那些日子一年後中位數 {_pct(deep365)}、半年後 {_pct(deep180)}），"
        f"訊號一年後的中位數{versus(sig365, deep365)}、半年後{versus(sig180, deep180)}；"
        f"和同一年的隨機日期比，訊號半年後的中位數{better(rand['all'][180])}、一年後{better(rand['all'][365])}。"
        f"{PUBLISHED[:7]} 公開之後有 {len(after)} 次，滿一年的 {a365} 次裡 {aup365} 次一年後是漲的"
        f"（一年後中位數 {_pct(aft365)}，同期任何一天是 {_pct(aall365)}、同期大跌中的任何一天是 {_pct(adeep365)}；"
        f"和同年隨機日期比，{better(rand['after'][365])}），次數太少，無法用統計確認。"
    )

    # ----- notes -----
    whipsaw = sum(1 for e in events if e["below_days"] <= 7)
    lag = [(e["date"] - e["cross"]).days for e in events]
    waited = sum(1 for v in lag if v > 0)
    t18, t12 = trade["2018"], trade["2012"]
    notes = [
        "名詞：「算力」是全網礦機的總運算能力；「礦工投降」是指挖礦不划算、一部分礦工關機，使算力的 30 日平均掉到 60 日平均之下。"
        "規則：算力 30 日簡單平均低於 60 日簡單平均＝投降中；30 日平均回到 60 日平均之上、而且價格的 10 日簡單平均高於 20 日簡單平均"
        "的第一天＝買進訊號（回到之上那天價格條件不符合，就等到之後第一個符合的日子，前提是 30 日平均還在 60 日之上）。"
        "規則沒有賣出條件，所以這裡只看訊號之後的漲跌。",
        "原始文章（Charles Edwards, Capriole, Medium）這個程式打不開（網站拒絕自動讀取，回傳 403），沒有繞過；"
        f"這裡用的是一般引用的版本，公開日期也是照一般說法當作 {PUBLISHED[:7]}，沒有辦法親自核對原文。"
        "作者自己的圖表用的是另一份算力資料，所以他列出的訊號日期和這裡算出來的不會完全一樣。",
        f"訊號很少：{SINCE[:4]} 年以來只有 {len(events)} 次，公開之後 {len(after)} 次（其中滿一年的 {a365} 次）。"
        "這麼少的次數沒有辦法用統計確認它有沒有用；多一次少一次、或某一次落在特別好的年份，結論就會改變。",
        "比特幣在這段期間漲了非常多倍，所以幾乎隨便哪一天買、一年後看起來都不錯——"
        f"{SINCE[:4]} 年以來的每一天，一年後是漲的就占 {(fwd[365][days].dropna() > 0).mean() * 100:.0f}%。"
        "所以要看的是訊號有沒有比「同樣在大跌期間的其他日子」或「同一年的隨機日期」好，而不是它本身有沒有漲。",
        "算力不是直接量到的數字，是從出塊快慢反推的估計值，每天亂跳，各家算法不同，事後也會修正。"
        "這裡用的是今天拿到的最終資料，當時即時看到的數字不一樣，所以這裡的訊號日期和當時實際看得到的可能差幾天，"
        f"訊號日的進場價也是用當天收盤價近似。{main_label} 給的是每天一筆、沒有先做平均的原始估計值（不是抽樣、也不是 7 日平均）。"
        f"這 {len(events)} 次訊號裡有 {whipsaw} 次的「投降」只持續 7 天以內——這種短暫的交叉很可能只是估計值的雜訊，"
        f"也最容易因為資料修正而出現或消失；有 {waited} 次是算力先回升、之後才等到價格確認。",
        "30／60 日和 10／20 日這些參數，是作者看過歷史之後才挑的，公開以前的訊號本來就是它被設計來抓的底部，"
        "所以公開以前的好成績有一部分是事後挑選的結果；真正算「事先」的只有公開之後那幾次。第三張表換參數、換資料來源後"
        "訊號日期會移動，可以看出結果有多依賴這些選擇。",
        f"對照方法：「比歷史最高價低 30% 以上」用的是當天收盤價和在那之前最高收盤價的比較；隨機對照是每次訊號各從同一個年份裡"
        f"隨機抽一天、組成一組，算這組的中位數，重複 {DRAWS} 組，再看訊號的中位數贏過其中幾組（固定亂數種子，每次執行結果一樣）。"
        "還沒滿 180 或 365 天的訊號不列入對應的統計。表裡每天的對照彼此高度重疊（相鄰的日子結果幾乎一樣），不是獨立樣本。",
        f"當成交易規則時：訊號當天收盤後才知道，所以從隔天開始持有；每次買或賣扣 {COST * 100:.2f}% 成本，持有現金沒有利息，"
        "一直抱著不扣成本，沒有算稅。因為規則本身沒有賣出條件，「抱 365 天」和「抱到下一次投降開始」都是這裡為了能比較才加上的做法，"
        f"不是原作者的規則。儀表板另外驗證過的順勢規則是 2018–2026 每年 +29.4%、最深曾經虧 −57%；這裡用同一份價格重算的是每年 "
        f"{_pct(t18['trend']['ret'])}、最深 {_pct(t18['trend']['deepest'])}（起訖日和細節不同會有小差異）。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）。"
        "早年（2011–2013）價格和算力資料都比較粗糙，成交也少，那幾次訊號的數字只能當參考。",
        "過去的結果不保證未來會一樣。這裡只是整理這個指標過去出現之後發生了什麼，不是買賣建議。",
    ]
    if hidden:
        notes.insert(2, f"第一張表最多列 {MAX_ROWS} 次，這裡列的是最近 {MAX_ROWS} 次，更早的 {hidden} 次沒有列出，但所有統計都有算進去。")
    if source_stats is not None:
        notes.insert(5, f"兩個算力來源對照：改用 {other_label} 後，{len(events)} 次訊號裡 {source_stats['same']} 次在同一天、"
                        f"{source_stats['near'] - source_stats['same']} 次相差 {NEAR} 天以內、{source_stats['missing']} 次消失，"
                        f"另外多出 {source_stats['extra']} 次。這兩家的每日估計值幾乎相同（都是用每天的出塊數反推），"
                        "所以這個對照只能說明「同一種算法」之下的差異，沒有涵蓋其他算法（例如用較長時間窗估計）的算力資料。")
    notes += problems

    tables = [
        {"caption": (f"每一次訊號（{SINCE[:4]} 年以來，共 {len(events)} 次"
                     + (f"，這裡列最近 {MAX_ROWS} 次" if hidden else "") + f"；算力用 {main_label} 的每日估計值）。"
                     "「算力回落」是投降期間算力 30 日平均的最低點，比投降前 90 天內的最高點低多少；"
                     "「比歷史最高價低」是訊號當天收盤價離先前最高收盤價多遠；之後各欄是從訊號當天收盤價起算的漲跌；"
                     f"「之後 {LOW_WINDOW} 天內最低」是那段期間最低的收盤價比訊號日低多少。「—」代表時間還沒到。"),
         "columns": ["投降開始", "訊號日", "當天價格（美元）", "算力回落", "比歷史最高價低", "30 天／90 天後",
                     "180 天／365 天後", f"之後 {LOW_WINDOW} 天內最低"],
         "rows": signal_rows},
        {"caption": ("和碰運氣比較。訊號大多出現在大跌期間，所以最公平的對照是「比歷史最高價低 30% 以上的每一天」和"
                     "「同一年的隨機日期」。中位數是把結果由小到大排、取正中間的那個。"
                     f"全部訊號和同年隨機日期比：半年後{better(rand['all'][180])}、一年後{better(rand['all'][365])}；"
                     f"公開後的訊號：半年後{better(rand['after'][180])}、一年後{better(rand['after'][365])}。"),
         "columns": ["組別", "次數／天數", "180 天後中位數", "180 天後上漲的比例", "365 天後中位數", "365 天後上漲的比例",
                     f"之後 {LOW_WINDOW} 天內最低（中位數）"],
         "rows": chance_rows},
        {"caption": (f"換一種算法，訊號日期會不會變。「同一天」「相差 {NEAR} 天內」是拿原規則的 {len(events)} 次訊號去對；"
                     f"「消失／多出」是原規則有但這個算法在 {NEAR} 天內找不到的次數／這個算法有但原規則沒有的次數。"),
         "columns": ["算法", "訊號次數", "和原規則同一天", f"相差 {NEAR} 天內", "消失／多出", "180 天後中位數", "365 天後中位數",
                     "一年後上漲"],
         "rows": variation_rows},
        {"caption": (f"當成交易規則（每次買賣扣 {COST * 100:.2f}% 成本，持有現金沒有利息）。「每年報酬」是整段成果換算成每年的複利；"
                     "「最深曾經虧多少」是從最高點往下掉最多的幅度。規則本身沒有賣出條件，兩種出場方式是這裡為了比較才加的。"),
         "columns": ["期間", "做法", "每年報酬", "最深曾經虧多少", "持有比特幣的天數比例", "買賣次數"],
         "rows": trade_rows},
    ]

    sources = [{"name": "Charles Edwards（Capriole）〈Hash Ribbons & Bitcoin Bottoms〉（原始說明，這次程式無法開啟）",
                "url": "https://medium.com/capriole/hash-ribbons-bitcoin-bottoms-60da13095836"},
               {"name": "blockchain.info 全網算力估計值（每日）", "url": "https://www.blockchain.com/explorer/charts/hash-rate"},
               {"name": "CoinMetrics 社群版 API（HashRate）", "url": "https://charts.coinmetrics.io/crypto-data/"},
               {"name": "blockchain.info 每日比特幣價格（2010 年起）",
                "url": "https://www.blockchain.com/explorer/charts/market-price"},
               {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"}]

    return {
        "id": "hash_ribbons",
        "title": "「礦工投降結束」買進訊號（Hash Ribbons）每一次之後怎麼了",
        "question": ("有一個叫 Hash Ribbons 的指標說：礦工撐不下去關機（投降）、之後算力又回升的那一刻，是買比特幣最好的時機。"
                     "它出現的次數很少。一次一次列出來：之後發生了什麼？有沒有比隨便挑一天、或大跌期間隨便挑一天好？公開之後的幾次表現如何？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
        "_debug": {"t18": t12 and None},
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
