"""Historical study "walkforward": if the best-looking parameters had been picked using only the data available at
the time, how did those exact parameters do on the following, unseen year?

Seven families of daily long-or-cash bitcoin rules (about 44 parameter sets). For every test year Y from 2018 on:
every set is scored (yearly return after costs divided by yearly volatility) on the data strictly before Y, the best
set / the best set of each family / the top five are recorded, and exactly those choices are then measured on year Y
only. The yearly test results are chained into one "chosen in advance" track record and compared with buy-and-hold,
with the set that looks best with hindsight, and with what luck alone would produce.

No look-ahead: a signal uses data up to and including the close of day t and decides the position held during day
t+1; the choice for year Y uses returns of days before Y-01-01 only; the funding percentile uses earlier days only.
Only closed candles and finished days are used. Everything comes from public, key-less endpoints through
common.get_json (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic

FAPI = "https://fapi.binance.com"
FNG = "https://api.alternative.me/fng/"
COST = 0.001  # 0.10% of the amount moved every time a rule switches between bitcoin and cash
YEAR = 365  # bitcoin trades every day
TRAIN_START = "2013-01-01"
FIRST_TEST_YEAR = 2018
MIN_HISTORY_DAYS = 3 * YEAR
TOP = 5
FUNDING_SINCE = "2019-09-01"
FUNDING_MIN_DAYS = 365
FUNDING_TOP = 0.90
MIN_YEAR_DAYS = 30  # a test year shorter than this gets no rank correlation
SIMS = 2000
SEED = 20180101

MA_DAYS = (5, 10, 20, 30, 50, 75, 100, 150, 200)
CROSS = ((5, 20), (10, 50), (20, 100), (50, 200))
CHANNEL = (10, 20, 50, 100)
PAST = (7, 14, 30, 60, 90, 180)
FILTER_MA = (20, 50, 100)
FEAR_AT = (10, 15, 20, 25)
FEAR_HOLD = (14, 30, 60)
NEG_AT = (0.0, -0.005, -0.01)  # % per 8 hours
NEG_HOLD = (14, 30)

FAMILIES = {"A": "均線順勢", "B": "兩條均線交叉", "C": "通道突破", "D": "過去一段時間是漲是跌",
            "E": "均線順勢＋資金費率不過熱", "F": "恐懼時買進", "G": "資金費率偏負時買進"}


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


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed daily closes since 2010 on a DatetimeIndex with no missing day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


def _funding_daily() -> pd.Series:
    """Binance BTCUSDT funding since 2019-09: the mean settlement of each finished UTC day, in % per 8 hours."""
    out, start = [], int(pd.Timestamp(FUNDING_SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(25):
        page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1000, "startTime": start})
        if not page:
            break
        out += page
        start = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    if len(out) < 1000:
        raise RuntimeError(f"幣安資金費率歷史只回傳 {len(out)} 筆")
    series = pd.Series([float(p["fundingRate"]) * 100 for p in out],
                       index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
    series = series[~series.index.duplicated()].sort_index()
    daily = series.resample("1D").mean()
    return daily[daily.index < pd.Timestamp(common.today_utc())]  # finished days only


def _fear_greed() -> pd.Series:
    """alternative.me fear & greed index, one value per UTC day (published at the start of that day)."""
    reply = common.get_json(FNG, {"limit": 0})
    rows = {}
    for item in reply.get("data", []):
        try:
            day = pd.Timestamp(int(item["timestamp"]), unit="s").normalize()
            rows[day] = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
    series = pd.Series(rows).sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]
    if len(series) < 1000:
        raise RuntimeError(f"恐懼與貪婪指數只回傳 {len(series)} 筆")
    return series


# ---------- the rules ----------

def _after_start(flag: pd.Series, source: pd.Series) -> pd.Series:
    """0/1 flag that is unknown (NaN) before the source's first value and 0 on later days the source is missing."""
    first = source.first_valid_index()
    out = flag.astype(float).where(source.notna(), 0.0)
    if first is None:
        return out * np.nan
    out[out.index < first] = np.nan
    return out


def _signals(close: pd.Series, funding: pd.Series | None, fng: pd.Series | None) -> tuple[pd.DataFrame, list[dict]]:
    """Signal at each day's close (1 = hold bitcoin tomorrow, 0 = cash, NaN = rule not available yet) per set."""
    cols: dict[str, pd.Series] = {}
    specs: list[dict] = []

    def add(key: str, family: str, label: str, signal: pd.Series) -> None:
        cols[key] = signal.astype(float)
        specs.append({"key": key, "family": family, "label": label})

    mean = {n: close.rolling(n, min_periods=n).mean() for n in sorted(set(MA_DAYS) | {x for p in CROSS for x in p})}
    for n in MA_DAYS:
        add(f"A{n}", "A", f"收盤高於 {n} 日均線才持有", (close > mean[n]).where(mean[n].notna()))
    for f, s in CROSS:
        add(f"B{f}_{s}", "B", f"{f} 日均線高於 {s} 日均線才持有", (mean[f] > mean[s]).where(mean[s].notna()))
    for n in CHANNEL:
        m = int(round(n / 2))
        high = close.rolling(n, min_periods=n).max()
        low = close.rolling(m, min_periods=m).min()
        state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
        state = state.where(high.notna()).ffill().fillna(0.0).where(high.notna())  # cash until the first new high
        add(f"C{n}", "C", f"創 {n} 日新高進場、創 {m} 日新低出場", state)
    for n in PAST:
        past = close / close.shift(n) - 1
        add(f"D{n}", "D", f"過去 {n} 天是漲的才持有", (past > 0).where(past.notna()))

    if funding is not None:
        avg7 = funding.rolling(7, min_periods=7).mean().round(6).reindex(close.index)
        # the line a day has to stay under: the 90th percentile of all EARLIER days (shift keeps the day itself out)
        line = avg7.expanding(min_periods=FUNDING_MIN_DAYS).quantile(FUNDING_TOP).shift(1).round(6)
        hot = (avg7 > line)  # a missing funding day counts as "not hot"
        for n in FILTER_MA:
            signal = ((close > mean[n]) & ~hot).astype(float).where(mean[n].notna())
            first = line.first_valid_index()  # the filter can only be judged once the line exists
            signal[signal.index < (first if first is not None else close.index[-1] + pd.Timedelta(days=1))] = np.nan
            add(f"E{n}", "E", f"收盤高於 {n} 日均線、而且資金費率不算過熱才持有", signal)
        for x in NEG_AT:
            trigger = _after_start(avg7 <= x, avg7)
            level = "0" if x == 0 else f"−{abs(x):g}%"
            for h in NEG_HOLD:
                add(f"G{abs(x):g}_{h}", "G", f"資金費率 7 日平均 ≤ {level} 之後持有 {h} 天",
                    trigger.rolling(h, min_periods=1).max())
    if fng is not None:
        index = fng.reindex(close.index)
        for t in FEAR_AT:
            trigger = _after_start(index <= t, index)
            for h in FEAR_HOLD:
                add(f"F{t}_{h}", "F", f"恐懼與貪婪指數 ≤ {t} 之後持有 {h} 天", trigger.rolling(h, min_periods=1).max())
    specs.sort(key=lambda s: s["family"])
    return pd.DataFrame(cols)[[s["key"] for s in specs]], specs


# ---------- measuring ----------

def _score_parts(net: np.ndarray) -> tuple[float | None, float, float]:
    """(yearly compound return in %, yearly volatility in %, return ÷ volatility) of a run of daily net returns."""
    n = len(net)
    if n < 2:
        return None, 0.0, 0.0
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    return ret, vol, (ret / vol if vol > 1e-9 else 0.0)  # a rule that stayed in cash scores 0


def _score(net: np.ndarray) -> float:
    return _score_parts(net)[2]


def _measure(net: pd.Series, position: pd.Series) -> dict:
    values = net.values
    ret, vol, ratio = _score_parts(values)
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    by_year = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    return {"days": len(values), "ret": ret, "vol": vol, "ratio": ratio, "deepest": deepest,
            "share": float(position.mean() * 100), "years": by_year}


def _spearman(a: list, b: list) -> float | None:
    if len(a) < 5:
        return None
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    value = float(np.corrcoef(ra, rb)[0, 1])
    return value if math.isfinite(value) else None


def _compute(close: pd.Series, funding: pd.Series | None, fng: pd.Series | None) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    signal, specs = _signals(close, funding, fng)
    keys = [s["key"] for s in specs]
    label = {s["key"]: s["label"] for s in specs}
    family = {s["key"]: s["family"] for s in specs}
    position = signal.shift(1)  # signal at the close of day t -> position during day t+1
    change = close.pct_change()
    turn = position.fillna(0.0).diff().abs().fillna(0.0)  # a rule starts from cash
    net = position.mul(change, axis=0) - COST * turn  # NaN while the rule is not available
    start = pd.Timestamp(TRAIN_START)
    position, net, change = position[position.index >= start], net[net.index >= start], change[change.index >= start]
    last = close.index[-1]
    days = net.index
    first_day = {k: max(start, position[k].first_valid_index() or last) for k in keys}
    test_years = list(range(FIRST_TEST_YEAR, last.year + 1))

    def at(day) -> int:
        return int(days.searchsorted(pd.Timestamp(day)))

    # ----- the yearly choice, made with data strictly before the year -----
    picks, eligible_by_year, before, during, rho = {}, {}, {}, {}, {}
    for year in test_years:
        a, b, c = at(f"{year}-01-01"), at(f"{year}-01-01"), at(f"{year + 1}-01-01")
        eligible = [k for k in keys if (pd.Timestamp(f"{year}-01-01") - first_day[k]).days >= MIN_HISTORY_DAYS]
        scores = {k: _score(net[k].values[at(first_day[k]):a]) for k in eligible}
        ranked = sorted(eligible, key=lambda k: -scores[k])
        fam_best = {}
        for k in ranked:
            fam_best.setdefault(family[k], k)
        picks[year] = {"best": ranked[0], "top": ranked[:TOP], "family": fam_best}
        eligible_by_year[year] = eligible
        before[year] = scores
        if c - b >= MIN_YEAR_DAYS:
            during[year] = {k: _score(net[k].values[b:c]) for k in eligible}
            rho[year] = _spearman([scores[k] for k in eligible], [during[year][k] for k in eligible])
        else:
            during[year], rho[year] = {}, None

    # ----- chain the chosen positions into one track record -----
    def chain(choice: dict) -> dict:
        """choice: {year: [keys]} (equal weight). Position, net return and results over those years."""
        years = sorted(choice)
        parts = []
        for year in years:
            rows = (days >= pd.Timestamp(f"{year}-01-01")) & (days < pd.Timestamp(f"{year + 1}-01-01"))
            parts.append(position.loc[rows, choice[year]].mean(axis=1))
        pos = pd.concat(parts)
        moved = pos.diff().abs()
        moved.iloc[0] = pos.iloc[0]  # starts from cash
        chained = pos * change.reindex(pos.index) - COST * moved
        hold = change.reindex(pos.index)
        out = _measure(chained, pos)
        out["hold"] = _measure(hold, pd.Series(1.0, index=pos.index))
        out["beat"] = sum(1 for y in years if out["years"][y] > out["hold"]["years"][y])
        out["n_years"] = len(years)
        out["first"], out["last"] = years[0], years[-1]
        out["position"] = pos
        return out

    best = chain({y: [picks[y]["best"]] for y in test_years})
    top = chain({y: picks[y]["top"] for y in test_years})
    by_family = {}
    for fam in FAMILIES:
        choice = {y: [picks[y]["family"][fam]] for y in test_years if fam in picks[y]["family"]}
        if choice:
            by_family[fam] = chain(choice)
            by_family[fam]["picks"] = {y: v[0] for y, v in choice.items()}

    # ----- hindsight: every set over the whole test period (days before a rule exists count as cash) -----
    t0 = at(f"{FIRST_TEST_YEAR}-01-01")
    net0 = net.fillna(0.0).iloc[t0:]
    pos0 = position.fillna(0.0).iloc[t0:]
    whole = {k: _score(net0[k].values) for k in keys}
    half = len(net0) // 2
    first_half = {k: _score(net0[k].values[:half]) for k in keys}
    second_half = {k: _score(net0[k].values[half:]) for k in keys}
    order = sorted(keys, key=lambda k: -whole[k])

    def rank_of(table: dict) -> dict:
        return {k: i + 1 for i, k in enumerate(sorted(keys, key=lambda k: -table[k]))}

    hind_key = order[0]
    hind = _measure(net0[hind_key], pos0[hind_key])
    hold_years = best["hold"]["years"]
    hind["beat"] = sum(1 for y in test_years if hind["years"][y] > hold_years[y])

    # ----- luck: yearly blocks of every set -----
    year_of = net0.index.year.values
    n_days = np.array([(year_of == y).sum() for y in test_years], dtype=float)
    logs = np.array([np.log1p(net0.values[year_of == y]).sum(axis=0) for y in test_years])  # (years, sets)
    sums = np.array([net0.values[year_of == y].sum(axis=0) for y in test_years])
    squares = np.array([(net0.values[year_of == y] ** 2).sum(axis=0) for y in test_years])
    total = float(n_days.sum())

    def pooled(log_sum, plain_sum, square_sum):
        ret = (np.exp(log_sum * YEAR / total) - 1) * 100
        var = np.maximum(square_sum - plain_sum ** 2 / total, 0.0) / (total - 1)
        vol = np.sqrt(var * YEAR) * 100
        return ret, vol, np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)

    rng = np.random.default_rng(SEED)
    n_sets, n_years = len(keys), len(test_years)
    # (1) shuffle each year's results between the sets: no set keeps any skill, then take the best-looking one
    shuffle = rng.random((SIMS, n_years, n_sets)).argsort(axis=2)
    rows = np.arange(n_years)[None, :, None]
    _, _, lucky = pooled(logs[rows, shuffle].sum(axis=1), sums[rows, shuffle].sum(axis=1),
                         squares[rows, shuffle].sum(axis=1))
    lucky_best = lucky.max(axis=1)
    # (2) pick one eligible set at random every year
    column = {k: i for i, k in enumerate(keys)}
    drawn = np.stack([rng.choice([column[k] for k in eligible_by_year[y]], size=SIMS) for y in test_years], axis=1)
    cols = np.arange(n_years)[None, :]
    r_ret, r_vol, r_ratio = pooled(logs[cols, drawn].sum(axis=1), sums[cols, drawn].sum(axis=1),
                                   squares[cols, drawn].sum(axis=1))

    return {
        "specs": specs, "keys": keys, "label": label, "family": family, "last": last, "test_years": test_years,
        "picks": picks, "eligible": eligible_by_year, "before": before, "during": during, "rho": rho,
        "best": best, "top": top, "by_family": by_family, "position": position, "signal": signal,
        "whole": whole, "order": order, "rank_whole": rank_of(whole), "rank_first": rank_of(first_half),
        "rank_second": rank_of(second_half), "half_rho": _spearman([first_half[k] for k in keys],
                                                                   [second_half[k] for k in keys]),
        "half_day": net0.index[half], "hind_key": hind_key, "hind": hind,
        "set_measure": {k: _measure(net0[k], pos0[k]) for k in order[:10]},
        "lucky_median": float(np.median(lucky_best)),
        "lucky_share": float((lucky_best >= whole[hind_key]).mean() * 100),
        "random": {"ret": float(np.median(r_ret)), "vol": float(np.median(r_vol)), "ratio": float(np.median(r_ratio)),
                   "below": float((r_ratio < best["ratio"]).mean() * 100)},
    }


# ---------- the study ----------

def study() -> dict:
    close = _daily_close()
    problems: list[str] = []
    try:
        funding = _funding_daily()
    except Exception as exc:  # noqa: BLE001 - the price-only families still stand without it
        funding = None
        problems.append(f"幣安資金費率這次取不到（{type(exc).__name__}），用到資金費率的兩個家族這次沒有算進去，沒有改用其他方法繞過。")
    try:
        fng = _fear_greed()
    except Exception as exc:  # noqa: BLE001
        fng = None
        problems.append(f"恐懼與貪婪指數這次取不到（{type(exc).__name__}），「恐懼時買進」這個家族這次沒有算進去。")

    r = _compute(close, funding, fng)
    label, family, keys, years, last = r["label"], r["family"], r["keys"], r["test_years"], r["last"]
    best, top, hold, hind = r["best"], r["top"], r["best"]["hold"], r["hind"]
    n = len(years)
    current = years[-1]
    partial = last < pd.Timestamp(f"{current}-12-31")

    def year_name(y: int) -> str:
        return f"{y}（到 {last.strftime('%m-%d')}）" if y == current and partial else str(y)

    def beat(m: dict) -> str:
        return f"{m['n_years']} 年裡 {m['beat']} 年"

    # ----- table 1: summary -----
    def summary_row(name: str, m: dict, wins: str) -> list:
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]), wins,
                f"{m['share']:.0f}%"]

    rnd = r["random"]
    summary_rows = [
        summary_row("一直抱著不動", hold, "—"),
        summary_row("事先挑的：每年年初分數最高的一組", best, beat(best)),
        summary_row(f"事先挑的：每年年初分數前 {TOP} 組平均", top, beat(top)),
        summary_row(f"事後挑的：回頭看整段最好的一組（{label[r['hind_key']]}）", hind, f"{n} 年裡 {hind['beat']} 年"),
        [f"對照：每年閉著眼睛隨便挑一組（{SIMS} 次的中間值）", _pct(rnd["ret"]), _pct(rnd["vol"], signed=False),
         _num(rnd["ratio"]), "—", "—", "—"],
        [f"對照：純靠運氣的「最好一組」（{SIMS} 次的中間值）", "—", "—", _num(r["lucky_median"]), "—", "—", "—"],
    ]

    # ----- table 2: year by year -----
    year_rows = []
    for y in years:
        key = r["picks"][y]["best"]
        during = r["during"][y]
        place = "—"
        if during:
            place = f"{len(during)} 組裡第 {sorted(during, key=lambda k: -during[k]).index(key) + 1}"
        year_rows.append([year_name(y), label[key], _pct(best["years"][y]), place, _pct(top["years"][y]),
                          _pct(hold["years"][y]), _num(r["rho"][y])])
    rhos = [r["rho"][y] for y in years if _ok(r["rho"][y])]
    rho_mean = float(np.mean(rhos)) if rhos else None
    rho_positive = sum(1 for v in rhos if v > 0)
    year_rows.append(["平均", "—", "—", "—", "—", "—", _num(rho_mean)])

    # ----- table 3: by family -----
    family_rows = []
    for fam, name in FAMILIES.items():
        m = r["by_family"].get(fam)
        if not m:
            continue
        span = f"{m['first']}–{m['last']}" if m["first"] != m["last"] else str(m["first"])
        family_rows.append([name, span, _pct(m["ret"]), _pct(m["hold"]["ret"]),
                            f"{_num(m['ratio'])}（{_num(m['hold']['ratio'])}）",
                            f"{_pct(m['deepest'])}（{_pct(m['hold']['deepest'])}）", beat(m),
                            label[m["picks"][current]]])

    # ----- table 4: the ten sets that look best with hindsight -----
    hind_rows = []
    for k in r["order"][:10]:
        m = r["set_measure"][k]
        hind_rows.append([f"第 {r['rank_whole'][k]}", label[k], _pct(m["ret"]), _num(m["ratio"]), _pct(m["deepest"]),
                          f"第 {r['rank_first'][k]}", f"第 {r['rank_second'][k]}"])
    kept = sum(1 for k in r["order"][:10] if r["rank_first"][k] <= 10 and r["rank_second"][k] <= 10)

    span_text = f"{years[0]}-01 到 {_d(last)}"
    counts = {fam: sum(1 for k in keys if family[k] == fam) for fam in FAMILIES}
    family_text = "、".join(f"{FAMILIES[f]} {c} 組" for f, c in counts.items() if c)
    tables = [
        {"caption": (f"總表（{span_text}，共 {n} 個測試年）。「事先挑的」是每年 1 月 1 日只用那天以前的資料算出每組參數的分數"
                     f"（扣成本後的每年報酬 ÷ 每年波動），挑最高的照做一整年，年年重挑，再把每一年接起來；"
                     f"「事後挑的」是現在回頭看整段才知道的最好一組。每次切換扣 {COST * 100:.2f}% 成本，持有現金沒有利息。"),
         "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "賺得比一直抱著多的年數", "持有比特幣的天數比例"],
         "rows": summary_rows},
        {"caption": ("逐年：年初事先挑到哪一組、它在那一年實際的結果。「排名相關」是把所有夠資格的參數組，用「年初以前的分數」和"
                     "「那一年的分數」各排一次名，兩份名次像不像（+1 完全一樣，0 毫無關係，−1 完全顛倒）。"),
         "columns": ["測試年", "年初事先挑到的那一組", "它那一年的報酬", "它那一年在各組裡的名次", f"前 {TOP} 組平均那一年",
                     "一直抱著那一年", "排名相關"],
         "rows": year_rows},
        {"caption": ("分家族：每個家族每年年初各自挑自己分數最高的一組，照做一整年後接起來。資料比較晚才有的家族測試年比較少，"
                     "括號裡是同一段期間一直抱著的數字。"),
         "columns": ["規則家族", "測試年", "每年報酬", "同期一直抱著", "報酬÷波動（一直抱著）", "最深曾經虧多少（一直抱著）",
                     "賺得比一直抱著多的年數", f"{current} 年初挑到的"],
         "rows": family_rows},
        {"caption": (f"事後看最好的十組（共 {len(keys)} 組，照 {span_text} 整段的報酬÷波動排名），以及同一組在前半段"
                     f"（到 {_d(r['half_day'] - pd.Timedelta(days=1))}）和後半段各排第幾。規則還沒有資料的日子當作持有現金。"
                     f"十組裡有 {kept} 組在前後兩半都排進前十；全部 {len(keys)} 組前後兩半的排名相關是 {_num(r['half_rho'])}。"),
         "columns": ["整段名次", "參數組", "每年報酬", "報酬÷波動", "最深曾經虧多少", "前半段名次", "後半段名次"],
         "rows": hind_rows},
    ]

    # ----- today -----
    pick = r["picks"][current]["best"]
    state = r["signal"].iloc[-1]

    def holding(k: str) -> str:
        return "持有比特幣" if state[k] == 1 else "持有現金"

    available = [k for k in keys if _ok(state[k])]
    long_now = sum(1 for k in available if state[k] == 1)
    top_now = r["picks"][current]["top"]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"{current} 年事先挑到的一組（只用到 {current - 1} 年底的資料）", "value": label[pick]},
        {"label": "這一組照規則下一天的狀態", "value": holding(pick)},
        {"label": f"{current} 年事先挑到的前 {TOP} 組", "value": f"{TOP} 組裡 {sum(1 for k in top_now if state[k] == 1)} 組持有比特幣："
         + "；".join(f"{label[k]}（{holding(k)}）" for k in top_now)},
        {"label": "全部參數組照規則下一天的狀態", "value": f"{len(available)} 組裡 {long_now} 組持有比特幣"},
    ]

    # ----- verdict -----
    more_ret, more_ratio = best["ret"] > hold["ret"], best["ratio"] > hold["ratio"]
    if more_ret and more_ratio:
        outcome = "報酬和報酬÷波動都贏過一直抱著"
    elif more_ratio:
        outcome = "賺得比一直抱著少，但起伏比較小，報酬÷波動比較高"
    elif more_ret:
        outcome = "賺得比一直抱著多，但報酬÷波動比較低"
    else:
        outcome = "報酬和報酬÷波動都輸給一直抱著"
    if rho_mean is None:
        persistence = "參數好壞會不會延續這次算不出來"
    else:
        if rho_mean >= 0.3:
            meaning = "有一些延續，但不是每年都成立"
        elif rho_mean >= 0.1:
            meaning = "只有微弱的延續"
        elif rho_mean > -0.1:
            meaning = "接近零，之前哪組參數好幾乎不能預告下一年哪組好"
        else:
            meaning = "是負的，之前好的參數下一年反而常常排在後面"
        persistence = (f"參數好壞的延續性：年初以前的名次和當年名次的排名相關平均 {_num(rho_mean)}"
                       f"（{len(rhos)} 年裡 {rho_positive} 年是正的），{meaning}")
    gap = hind["ret"] - best["ret"]
    chosen = [r["picks"][y]["best"] for y in years]
    if len(set(chosen)) == 1:
        variety = f"（{n} 年挑到的都是同一組：{label[chosen[0]]}）"
    else:
        variety = f"（{n} 年裡挑到過 {len(set(chosen))} 種不同的參數組）"
    verdict = (
        f"{years[0]} 到 {current} 這 {n} 年，每年年初只用當時以前的資料挑出分數最高的一組參數照做一整年{variety}："
        f"扣成本後每年報酬 {_pct(best['ret'])}（一直抱著 {_pct(hold['ret'])}），報酬÷波動 {_num(best['ratio'])}"
        f"（一直抱著 {_num(hold['ratio'])}），最深曾經虧 {_pct(best['deepest'])}（一直抱著 {_pct(hold['deepest'])}），"
        f"{beat(best)}賺得比一直抱著多，也就是{outcome}；改成前 {TOP} 組平均是每年 {_pct(top['ret'])}、報酬÷波動 {_num(top['ratio'])}。"
        f"事後回頭才挑得出來的最好一組（{label[r['hind_key']]}）是每年 {_pct(hind['ret'])}、報酬÷波動 {_num(hind['ratio'])}，"
        f"比事先挑的每年{'多' if gap >= 0 else '少'} {abs(gap):.1f} 個百分點；而把各組每年的成績打亂、讓每組都沒有真本事，"
        f"純靠運氣的「最好一組」報酬÷波動也有 {_num(r['lucky_median'])}。{persistence}。"
    )

    late = []
    for fam in ("E", "F", "G"):
        if fam in r["by_family"]:
            late.append(f"「{FAMILIES[fam]}」從 {r['by_family'][fam]['first']} 年")
    notes = [
        f"一共 {len(keys)} 組參數：{family_text}。每條規則都是用當天收盤時已經知道的資料決定「下一天」持有比特幣還是現金；"
        "每年的挑選只用那一年 1 月 1 日以前的日子，挑完整年不換。只用已經收完的日線。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「每年波動」是每天漲跌的起伏換算成一年；「報酬÷波動」是兩者相除，"
        "越高代表每承受一分起伏換到的報酬越多；「最深曾經虧多少」是從最高點往下掉最多的幅度。整年都持有現金的規則分數記為 0。",
        f"測試年很少：只有 {n} 個（{'最後一個還沒過完' if partial else '都是完整的年'}），「{n} 年裡幾年」這種數字多一年少一年就會差很多，"
        "排名相關也是每年只有一個數字、上下跳動很大。而且只測了比特幣一種資產，同一套做法換到別的資產不一定一樣。",
        "比特幣早年漲得異常多：挑選用的資料從 2013 年算起，2013 和 2017 年的漲幅是後來沒有再出現過的，"
        "這段期間幾乎任何「有在持有」的規則分數都很高，早年挑出來的參數很受這兩年影響。",
        "每組參數的分數是從 2013-01（或該家族最早有資料的那天）算到挑選那年的前一天，至少要有 3 年才夠資格參加："
        + ("、".join(late) + "才開始有資格。" if late else "")
        + "資料較晚才有的家族沒有經歷早年的大漲，分數和老家族不在同一個起跑點上，所以「全部一起挑」時幾乎都是老家族被挑中；"
        "要看各家族自己的結果請看第三張表。",
        f"運氣檢查只做了最簡單的兩種：（一）把每一年裡各組的成績隨機互換 {SIMS} 次，這樣每一組都沒有真本事，再看其中最好的一組"
        f"能有多好看——中間值是報酬÷波動 {_num(r['lucky_median'])}，{SIMS} 次裡有 {r['lucky_share']:.0f}% 的次數不輸給真正事後最好的那一組"
        f"（{_num(r['whole'][r['hind_key']])}）；（二）每年從夠資格的組裡隨便挑一組，{SIMS} 次裡有 {rnd['below']:.0f}% 的次數"
        "報酬÷波動比「事先挑分數最高的」低。這兩個對照是用每年的成績拼起來的，沒有算換組那一天的成本，是約略的數字。",
        f"成本只用一個粗略的數字：每次切換扣 {COST * 100:.2f}%。實際的手續費、買賣價差和滑價會隨交易所、金額和年代不同；"
        "持有現金假設沒有利息，也沒有算稅；一直抱著沒有扣成本。年初換成另一組參數時如果狀態不同，也照樣扣一次成本；"
        f"「前 {TOP} 組平均」是把資金平均分給 {TOP} 組、每天維持等分。",
        "這七個規則家族和每個家族要試哪些參數，是用今天的知識選的（例如現在大家都知道均線順勢在比特幣上表現過不錯），"
        "所以即使每年的挑選沒有偷看，整個研究還是留有一些事後才知道的成分，表裡「事先挑的」數字因此可能比真正事先能做到的好看。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）。"
        "blockchain.info 標示的日期大約比幣安晚一天，整段內部順序一致、不會用到未來的價格，但 2017-08 接縫那一天等於合併了兩天的漲跌；"
        "通道突破用的是收盤價的新高新低，不是盤中最高最低價。",
        "資金費率是幣安 BTCUSDT 永續合約每 8 小時結算一次的費率，先取每天的平均、再取 7 天平均；「不算過熱」是指沒有高過它自己"
        "先前所有日子的九成（先累積 365 天才開始判斷，只用當天以前的日子）。恐懼與貪婪指數是 alternative.me 每天公布一次的數字，"
        "缺資料的日子當作沒有觸發。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則和這套挑選方法過去做了什麼，不是買賣建議。",
    ] + problems

    return {
        "id": "walkforward",
        "title": "事先挑好的參數，後來表現如何",
        "question": "如果每年只用當時已經有的資料挑出看起來最好的參數，這組參數在接下來沒看過的那一年表現得怎麼樣？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2010 年起）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安 U 本位永續合約資金費率歷史",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "alternative.me 加密貨幣恐懼與貪婪指數", "url": "https://alternative.me/crypto/fear-and-greed-index/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
