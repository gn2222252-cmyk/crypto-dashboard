"""Historical study "option_selling": holding bitcoin and selling call options against it every month (or holding
dollars and selling puts) - what did that end with compared with simply holding, and when did it go wrong?

Everything is a MODEL price, not a traded option price: Black-Scholes with zero interest, volatility = Deribit's DVOL
index (a 30-day, at-the-money-style index) read at 00:00 UTC of the day the option is sold, shifted for strikes away
from the price with the stored skew history (cache/_store/skew_history.json, kept by skew_history.py: implied
volatility 10% below the price minus 10% above it, every 4 days, interpolated by date) linearly in log-moneyness.
The seller receives the model price minus a spread (5% of the premium; 0% and 10% in the sensitivity table) minus
Deribit's stated option taker fee (0.03% of the underlying, capped at 12.5% of the premium - an assumption taken from
Deribit's public fee page, it is not in the dashboard code) and pays the delivery fee (0.015%, same cap) when the
option finishes in the money.

Prices: Deribit's own daily 08:00 UTC delivery price of the btc_usd index (the price options really settle on),
kept in cache/_store/option_selling_delivery.json. Options are sold and settled on that price on Fridays; monthly =
last Friday of the month to the next last Friday, weekly = every Friday to the next. Accounting is in dollars and
compounding: at every roll the whole account (premium and settlement included) is one covered unit again.
Between rolls the account is marked every day with the same model, which is what volatility and deepest loss use.

Sources (public, no key, through common.get_json): Deribit delivery prices, Deribit volatility index, the stored skew
history, and trend_ensemble's daily closes for the trend filter. Descriptive and in-sample; not advice.
"""
from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from statistics import NormalDist

import numpy as np

from dashboard import common

ID = "option_selling"
DERIBIT = "https://www.deribit.com/api/v2/public"
DELIVERY_FILE = common.STORE / f"{ID}_delivery.json"
SKEW_FILE = common.STORE / "skew_history.json"
START = date(2021, 3, 1)
EPOCH = date(1970, 1, 1).toordinal()
DAY_MS = 86_400_000
YEAR = 365
SPREAD = 0.05  # share of the premium lost to the bid-ask spread
SPREADS = (0.0, 0.05, 0.10)
FEE, DELIVERY_FEE, FEE_CAP = 0.0003, 0.00015, 0.125  # Deribit's stated option fees (assumption, see module text)
SPLITS = (date(2022, 12, 30), date(2024, 12, 27))  # last Fridays of December: ends of 2021-22 and 2023-24
VENDOR = (date(2024, 5, 31), date(2026, 5, 29), 27.6, 12.2)  # window of the vendor claim, its two total returns
_ERF = np.vectorize(math.erf, otypes=[float])
_INV = NormalDist().inv_cdf


# ---------- small helpers ----------

def _num(day: date) -> int:
    return day.toordinal() - EPOCH


def _date(number: int) -> date:
    return date.fromordinal(int(number) + EPOCH)


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, digits: int = 1, signed: bool = True) -> str:
    """v is a fraction (0.12 = 12%)."""
    if not _ok(v):
        return "—"
    v = float(v) * 100
    if abs(v) < 0.5 * 10 ** -digits:
        v = 0.0
    if abs(v) >= 100:
        digits = 0
    return f"{v:+,.{digits}f}%" if signed and v != 0 else f"{v:,.{digits}f}%"


def _ratio(v) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.2f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _ym(number: int) -> str:
    return _date(number).strftime("%Y-%m-%d")


# ---------- pricing ----------

def _cdf(x):
    return 0.5 * (1 + _ERF(np.asarray(x, dtype=float) / math.sqrt(2)))


def price(spot, strike, years, vol, call: bool):
    """Black-Scholes value with zero interest; works on numbers or arrays, and is the pay-off when no time is left."""
    spot, strike, years, vol = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (spot, strike, years, vol)))
    live = (years > 0) & (vol > 0)
    spread = np.where(live, vol * np.sqrt(np.where(live, years, 1.0)), 1.0)
    d1 = np.log(spot / strike) / spread + spread / 2
    value = np.where(call, spot * _cdf(d1) - strike * _cdf(d1 - spread),
                     strike * _cdf(spread - d1) - spot * _cdf(-d1))
    rest = np.maximum(spot - strike, 0) if call else np.maximum(strike - spot, 0)
    out = np.where(live, value, rest)
    return float(out) if out.ndim == 0 else out


def _vol_at(base, slope, moneyness, scale):
    """Volatility used for a strike: the index level shifted linearly in log-moneyness, kept within sane bounds."""
    base = np.asarray(base, dtype=float)
    shifted = base + np.asarray(slope, dtype=float) * np.log(moneyness) * scale
    out = np.clip(shifted, np.maximum(0.6 * base, 0.10), 2.0 * base)
    return float(out) if out.ndim == 0 else out


def _strike(spot: float, vol: float, years: float, rule: tuple, call: bool) -> float:
    kind, x = rule
    if kind == "pct":
        return spot * (1 + x) if call else spot * (1 - x)
    spread = vol * math.sqrt(years)  # "delta": the strike whose flat-volatility model delta is x
    return spot * math.exp(-_INV(x) * spread + spread * spread / 2) if call else \
        spot * math.exp(_INV(x) * spread + spread * spread / 2)


# ---------- data ----------

def _delivery() -> dict[int, float]:
    """Deribit's daily 08:00 UTC btc_usd delivery price keyed by day number; the store grows run after run."""
    try:
        stored = json.loads(DELIVERY_FILE.read_text(encoding="utf-8")) if DELIVERY_FILE.exists() else {}
        stored = {str(k)[:10]: float(v) for k, v in stored.items() if _ok(v) and v > 0}
    except Exception:  # noqa: BLE001 - a damaged store starts over
        stored = {}
    first = START.isoformat()
    complete = bool(stored) and min(stored) <= first
    newest = max(stored) if stored else ""
    offset = 0
    for _ in range(40):
        rows = common.get_json(f"{DERIBIT}/get_delivery_prices", {"index_name": "btc_usd", "offset": offset,
                                                                  "count": 100})["result"]["data"]
        if not rows:
            break
        for row in rows:
            value = float(row["delivery_price"])
            if math.isfinite(value) and value > 0:
                stored[str(row["date"])[:10]] = value
        oldest = min(str(row["date"])[:10] for row in rows)
        if oldest < first or (complete and oldest <= newest):
            break
        offset += len(rows)
    stored = {k: v for k, v in stored.items() if k >= "2021-02-01"}
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        DELIVERY_FILE.write_text(json.dumps(dict(sorted(stored.items()))), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if len(stored) < 1000:
        raise RuntimeError(f"Deribit 交割價歷史太短：{len(stored)} 天")
    return {_num(date.fromisoformat(k)): v for k, v in stored.items()}


def _dvol() -> dict[int, tuple[float, float]]:
    """Deribit BTC DVOL daily (open, close) as fractions, keyed by day number (the candle starts at 00:00 UTC)."""
    out: dict[int, tuple[float, float]] = {}
    end = int(time.time() * 1000)
    start = int(datetime(START.year, START.month, START.day, tzinfo=timezone.utc).timestamp() * 1000)
    for _ in range(20):
        result = common.get_json(f"{DERIBIT}/get_volatility_index_data",
                                 {"currency": "BTC", "start_timestamp": start, "end_timestamp": end,
                                  "resolution": "1D"})["result"]
        for row in result["data"]:
            a, b = float(row[1]), float(row[4])
            if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
                out[int(row[0]) // DAY_MS] = (a / 100, b / 100)
        if not result.get("continuation"):
            break
        end = result["continuation"]
    if len(out) < 500:
        raise RuntimeError(f"Deribit DVOL 歷史太短：{len(out)} 天")
    return out


def _skew() -> dict[int, tuple[float, float | None]]:
    """The stored skew history: {day number: (skew in volatility points, at-the-money volatility in % or None)}."""
    try:
        raw = json.loads(SKEW_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - without the store the study runs with no skew adjustment
        return {}
    out = {}
    for day, row in (raw.items() if isinstance(raw, dict) else []):
        if isinstance(row, dict) and _ok(row.get("skew")):
            try:
                out[_num(date.fromisoformat(str(day)[:10]))] = (float(row["skew"]),
                                                                float(row["atm"]) if _ok(row.get("atm")) else None)
            except ValueError:
                continue
    return out


def _trend_share():
    """Share of trend_ensemble's rules that are long, by the day number of the daily candle it was decided on."""
    try:
        from dashboard.studies2 import trend_ensemble as te
        share = te._targets(te._daily_close())[0]["blend"].dropna()
        return {_num(ts.date()): float(v) for ts, v in share.items()}
    except Exception:  # noqa: BLE001 - the trend rows are left out when its data cannot be read
        return None


def _load() -> dict:
    delivery, dvol, skew = _delivery(), _dvol(), _skew()
    first = max(min(dvol), _num(START))
    last = min(max(delivery), max(dvol))
    n = last - first + 1
    px, vol = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        d = first + i
        if d in delivery:
            px[i] = delivery[d]
        if d in dvol:
            vol[i] = dvol[d][0]  # the 00:00 UTC reading, eight hours before the 08:00 sale: nothing from later
        elif d - 1 in dvol:
            vol[i] = dvol[d - 1][1]
    for arr in (px, vol):  # a missing day repeats the last known value
        for i in range(1, n):
            if not math.isfinite(arr[i]):
                arr[i] = arr[i - 1]
    if not (math.isfinite(px[0]) and math.isfinite(vol[0])):
        raise RuntimeError("起始日沒有價格或 DVOL")
    slope = np.zeros(n)
    if len(skew) >= 20:
        xs = sorted(skew)
        slope = np.interp(np.arange(first, last + 1), xs, [-skew[x][0] / 100 / 0.2 for x in xs])
    return {"first": first, "last": last, "px": px, "vol": vol, "slope": slope, "skew": skew, "dvol": dvol,
            "has_skew": len(skew) >= 20}


# ---------- simulation ----------

def _fridays(first: int, last: int) -> tuple[list[int], list[int]]:
    """(every Friday, last Friday of each month) between two day numbers; the monthly list ends both."""
    day = _date(first)
    day += timedelta(days=(4 - day.weekday()) % 7)
    weekly = []
    while _num(day) <= last:
        weekly.append(_num(day))
        day += timedelta(days=7)
    monthly = [d for d in weekly if (_date(d) + timedelta(days=7)).month != _date(d).month]
    weekly = [d for d in weekly if monthly and monthly[0] <= d <= monthly[-1]]
    return weekly, monthly


def simulate(data: dict, dates: list[int], mode: str, rule: tuple, sell=None, skew: bool = True,
             spread: float = SPREAD, fee: bool = True, shift: float = 0.0, put_rule: tuple | None = None) -> dict:
    """One selling programme. mode: "hold", "call" (hold bitcoin, sell calls), "put" (hold dollars, sell puts) or
    "wheel" (puts until assigned, then calls until called away). Returns the daily account value (1 at the start,
    indexed like data["px"]) and one record per cycle."""
    first, px, vols, slopes = data["first"], data["px"], data["vol"], data["slope"]
    equity = np.full(len(px), np.nan)
    equity[dates[0] - first] = 1.0
    level, coin, cycles = 1.0, mode in ("hold", "call"), []
    for start, end in zip(dates[:-1], dates[1:]):
        a, b = start - first, end - first
        s0, s1, days = px[a], px[b], end - start
        idx = np.arange(a + 1, b + 1)
        path = px[idx]
        selling = mode != "hold" and (sell is None or bool(sell(start)))
        net = give = 0.0
        strike, hit = None, False
        if selling:
            years, sigma, scale = days / YEAR, vols[a] + shift, math.sqrt(30 / days)
            strike = _strike(s0, sigma, years, rule if coin or put_rule is None else put_rule, coin)
            premium = price(s0, strike, years, _vol_at(sigma, slopes[a] if skew else 0.0, strike / s0, scale), coin)
            net = premium * (1 - spread) - (min(FEE * s0, FEE_CAP * premium) if fee else 0.0)
            inside = max(s1 - strike, 0.0) if coin else max(strike - s1, 0.0)
            hit = inside > 0
            give = inside + (min(DELIVERY_FEE * s1, FEE_CAP * inside) if fee and hit else 0.0)
            mark = price(path, strike, (b - idx) / YEAR,
                         _vol_at(vols[idx] + shift, slopes[idx] if skew else 0.0, strike / path, scale), coin)
            mark[-1] = give
            value = ((path if coin else s0) + net - mark) / s0
        else:
            value = path / s0 if coin else np.ones(len(idx))
        equity[idx] = level * value
        level = float(equity[b])
        cycles.append({"start": start, "end": end, "sold": selling, "call": coin, "hit": hit, "strike": strike,
                       "premium": net / s0, "give": give / s0, "hold": s1 / s0 - 1, "ret": float(value[-1]) - 1})
        if mode == "wheel" and hit:
            coin = not coin
    return {"equity": equity, "cycles": cycles}


def measure(data: dict, run: dict, monthly: list[int], start: int, end: int, year_ends: list[int]) -> dict:
    first = data["first"]
    eq = run["equity"][start - first:end - first + 1]
    eq = eq / eq[0]
    n = len(eq) - 1
    change = eq[1:] / eq[:-1] - 1
    growth = float(eq[-1])
    ret = growth ** (YEAR / n) - 1 if growth > 0 else -1.0
    vol = float(np.std(change, ddof=1) * math.sqrt(YEAR))
    months = [m for m in monthly if start <= m <= end]
    month_ret = [(a, b, float(run["equity"][b - first] / run["equity"][a - first] - 1))
                 for a, b in zip(months[:-1], months[1:])]
    cuts = [start] + [y for y in year_ends if start < y < end] + [end]
    years = [float(run["equity"][b - first] / run["equity"][a - first] - 1) for a, b in zip(cuts[:-1], cuts[1:])]
    cycles = [c for c in run["cycles"] if c["start"] >= start and c["end"] <= end]
    sold = [c for c in cycles if c["sold"]]
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else None, "total": growth - 1,
            "deepest": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "worst": min(month_ret, key=lambda r: r[2]) if month_ret else None, "months": month_ret, "years": years,
            "cycles": len(cycles), "sold": len(sold), "hit": sum(c["hit"] for c in sold),
            "premium": sum(c["premium"] for c in sold) / (n / YEAR),
            "give": sum(c["give"] for c in sold) / (n / YEAR),
            "hold_part": sum(c["hold"] for c in cycles if c["call"]) / (n / YEAR)}


def _realised(px: np.ndarray, window: int = 30) -> np.ndarray:
    """Yearly volatility actually realised over the following `window` days, for each day (NaN at the end)."""
    square = np.log(px[1:] / px[:-1]) ** 2
    total = np.concatenate([[0.0], np.cumsum(square)])
    out = np.full(len(px), np.nan)
    count = len(px) - window
    if count > 0:
        out[:count] = np.sqrt((total[window:window + count] - total[:count]) * YEAR / window)
    return out


# ---------- the study ----------

def study() -> dict:
    data = _load()
    first, px, vols = data["first"], data["px"], data["vol"]
    weekly, monthly = _fridays(first, data["last"])
    if len(monthly) < 24:
        raise RuntimeError(f"完整的月份只有 {len(monthly) - 1} 個")
    start, end = monthly[0], monthly[-1]
    year_ends = [m for a, m in zip(monthly, monthly[1:] + [monthly[-1] + 31])
                 if _date(m).month == 12 and m != start]
    trend = _trend_share()

    def trend_at(day: int):
        return None if trend is None else trend.get(day - 1)  # candle that closed at 00:00 UTC of the sale day

    def dvol_high(day: int) -> bool:
        i = day - first
        past = vols[max(0, i - 365):i]
        return len(past) >= 30 and vols[i] > float(np.median(past))

    plans = {
        "hold": ("單純持有比特幣（對照）", monthly, dict(mode="hold", rule=("pct", 0))),
        "c0": ("每月賣買權：約定價＝現價（價平）", monthly, dict(mode="call", rule=("pct", 0.0))),
        "c5": ("每月賣買權：約定價高 5%", monthly, dict(mode="call", rule=("pct", 0.05))),
        "c10": ("每月賣買權：約定價高 10%", monthly, dict(mode="call", rule=("pct", 0.10))),
        "c20": ("每月賣買權：約定價高 20%", monthly, dict(mode="call", rule=("pct", 0.20))),
        "cd25": ("每月賣買權：模型估四次約一次被履約的價位", monthly, dict(mode="call", rule=("delta", 0.25))),
        "cd10": ("每月賣買權：模型估十次約一次被履約的價位", monthly, dict(mode="call", rule=("delta", 0.10))),
        "p0": ("抱美元每月賣賣權：約定價＝現價（價平）", monthly, dict(mode="put", rule=("pct", 0.0))),
        "p5": ("抱美元每月賣賣權：約定價低 5%", monthly, dict(mode="put", rule=("pct", 0.05))),
        "p10": ("抱美元每月賣賣權：約定價低 10%", monthly, dict(mode="put", rule=("pct", 0.10))),
        "wheel": ("輪流：先賣低 5% 的賣權，被履約後改抱幣賣高 5% 的買權", monthly,
                  dict(mode="wheel", rule=("pct", 0.05))),
        "t_lo": ("價平買權只在順勢規則看多不到一半時賣", monthly,
                 dict(mode="call", rule=("pct", 0.0), sell=lambda d: trend_at(d) is not None and trend_at(d) < 0.5)),
        "t_hi": ("價平買權只在順勢規則看多過半時賣", monthly,
                 dict(mode="call", rule=("pct", 0.0), sell=lambda d: trend_at(d) is not None and trend_at(d) >= 0.5)),
        "v_hi": ("價平買權只在 DVOL 高於過去一年中位數時賣", monthly,
                 dict(mode="call", rule=("pct", 0.0), sell=dvol_high)),
        "wc0": ("每週賣買權：價平（估價最粗，見說明）", weekly, dict(mode="call", rule=("pct", 0.0))),
        "wp0": ("抱美元每週賣賣權：價平（估價最粗，見說明）", weekly, dict(mode="put", rule=("pct", 0.0))),
    }
    if trend is None:
        plans.pop("t_lo"), plans.pop("t_hi")
    runs = {k: simulate(data, dates, **kw) for k, (_, dates, kw) in plans.items()}
    whole = {k: measure(data, runs[k], monthly, start, end, year_ends) for k in runs}
    hold = whole["hold"]
    n_months, n_years = len(monthly) - 1, len(hold["years"])
    span = f"{_ym(start)} 到 {_ym(end)}"

    # ----- table 1: whole period -----
    def exercised(key: str, m: dict) -> str:
        if key == "hold":
            return "—"
        unit = "週" if key in ("wc0", "wp0") else "個月"
        return f"{m['sold']} {unit}裡 {m['hit']} {unit}"

    rows1 = []
    for key, (name, _, _) in plans.items():
        m = whole[key]
        beat = "—" if key == "hold" else f"{n_years} 年裡 {sum(a > b for a, b in zip(m['years'], hold['years']))} 年"
        tail = "—" if key == "hold" else f"{_pct(m['premium'], signed=False)}｜{beat}"
        rows1.append([name, _pct(m["ret"]), _pct(m["vol"], signed=False), _ratio(m["ratio"]), _pct(m["deepest"]),
                      _pct(m["worst"][2]) if m["worst"] else "—", exercised(key, m), tail])
    cc, csp = whole["c0"], whole["p0"]
    table1 = {
        "caption": (f"全期（{span}，{n_months} 個月、{n_years} 個年度，頭尾兩年不滿一年）。全部以美元計；賣買權的做法一直抱著比特幣，"
                    "賣賣權的做法抱著等值的美元（美元沒有算利息）。權利金＝賣出選擇權當下收到的那筆錢，用模型估價再扣 "
                    f"{SPREAD * 100:.0f}% 價差和手續費。「被履約」＝到期時價格越過約定價、要把差額賠給買方的次數（只算有賣出的月份或週）。"
                    "最後一欄是「每年收到的權利金占比特幣價值｜年度報酬高於單純持有的年數」。"),
        "columns": ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深虧損", "最差的一個月", "被履約", "每年權利金｜贏過持有的年數"],
        "rows": rows1}

    # ----- table 2: sub-periods and the insurance mark-up -----
    bounds = [start] + [_num(s) for s in SPLITS if start < _num(s) < end] + [end]
    periods = list(zip(bounds[:-1], bounds[1:]))
    labels = [f"{_ym(a)[:7]}～{_ym(b)[:7]}" for a, b in periods]
    sub = {k: [measure(data, runs[k], monthly, a, b, year_ends) for a, b in periods] for k in runs}
    keys2 = [k for k in ("hold", "c0", "c10", "cd25", "p0", "p10", "wheel", "t_lo", "t_hi", "v_hi", "wc0", "wp0")
             if k in runs]
    rows2 = []
    for key in keys2:
        wins = "—" if key == "hold" else \
            f"{len(periods)} 段裡 {sum(m['ret'] > h['ret'] for m, h in zip(sub[key], sub['hold']))} 段"
        rows2.append([plans[key][0]] + [f"{_pct(m['ret'])}｜{_pct(m['deepest'])}" for m in sub[key]] + [wins])
    realised = _realised(px)
    seen = np.isfinite(realised)

    def gap(a: int, b: int):
        pick = seen[a - first:b - first + 1]
        if pick.sum() < 20:
            return None
        return (float(vols[a - first:b - first + 1][pick].mean()), float(realised[a - first:b - first + 1][pick].mean()))

    gaps = [gap(a, b - 1) for a, b in periods]
    rows2.append(["DVOL 平均（市場預期的一年波動）"] + [_pct(g[0], signed=False) if g else "—" for g in gaps] + ["—"])
    rows2.append(["之後 30 天實際的波動（平均）"] + [_pct(g[1], signed=False) if g else "—" for g in gaps] + ["—"])
    rows2.append(["預期比實際高出多少（保險的加價）"]
                 + [f"{(g[0] - g[1]) * 100:+.1f} 點" if g else "—" for g in gaps] + ["—"])
    table2 = {"caption": "分三段看。每格是「每年報酬｜這一段裡的最深虧損」；最後一欄是每年報酬高於單純持有的段數。"
                         "最下面三列是選擇權市場預期的波動（DVOL）和之後 30 天實際發生的波動，差距就是賣方收到的「保險加價」。",
              "columns": ["做法"] + labels + ["報酬高於持有"], "rows": rows2}

    # ----- table 3: worst months and the rallies -----
    by_month = {k: {(a, b): r for a, b, r in whole[k]["months"]} for k in ("hold", "c0", "p0")}
    tags: dict[tuple, list[str]] = {}
    for key, word in (("hold", "持有"), ("c0", "賣買權"), ("p0", "賣賣權")):
        for rank, (a, b, _) in enumerate(sorted(whole[key]["months"], key=lambda r: r[2])[:5], 1):
            tags.setdefault((a, b), []).append(f"{word}第 {rank} 慘")
    bad = sorted(tags, key=lambda k: by_month["hold"][k])[:13]
    rallies = sorted(by_month["hold"], key=lambda k: -by_month["hold"][k])[:3]

    def row3(k: tuple, tag: str) -> list:
        h, c, p = by_month["hold"][k], by_month["c0"][k], by_month["p0"][k]
        return [f"{_ym(k[0])}→{_ym(k[1])}", tag, _pct(h), _pct(c), f"{(c - h) * 100:+.1f} 點", _pct(p)]

    rows3 = [row3(k, "、".join(tags[k])) for k in bad]
    rows3 += [row3(k, f"漲最多第 {i} 名") for i, k in enumerate(rallies, 1)]
    rally_hold = sum(by_month["hold"][k] for k in rallies)
    rally_cc = sum(by_month["c0"][k] for k in rallies)
    table3 = {"caption": "最慘的月份與漲最多的月份（每月最後一個週五到下一個月最後一個週五）。上半是單純持有、每月賣價平買權、"
                         "抱美元每月賣價平賣權各自最慘的五個月（重複的合併成一列）；最下面三列是比特幣漲最多的三個月，"
                         "「賣買權比持有」那欄是少賺（負）或少賠（正）多少。價平的賣買權和價平的賣賣權在算式上是同一件事，所以兩欄幾乎一樣。",
              "columns": ["期間", "屬於", "單純持有", "每月賣價平買權", "賣買權比持有", "抱美元賣價平賣權"], "rows": rows3}

    # ----- table 4: pricing assumptions -----
    keys4 = ("c0", "c10", "c20", "p0", "p10", "wheel")

    def variant(**change) -> list[str]:
        out = []
        for key in keys4:
            kw = dict(plans[key][2], **change)
            out.append(_pct(measure(data, simulate(data, monthly, **kw), monthly, start, end, year_ends)["ret"]))
        return out

    rows4 = [["本頁設定：有偏斜調整、價差 5%、扣手續費"] + [_pct(whole[k]["ret"]) for k in keys4]]
    rows4 += [[f"有偏斜調整、價差 {int(s * 100)}%"] + variant(spread=s) for s in SPREADS if s != SPREAD]
    rows4 += [[f"不做偏斜調整、價差 {int(s * 100)}%"] + variant(skew=False, spread=s) for s in SPREADS]
    rows4 += [["不扣手續費（其餘同本頁設定）"] + variant(fee=False),
              ["實際成交的波動率比 DVOL 低 5 點"] + variant(shift=-0.05),
              ["實際成交的波動率比 DVOL 高 5 點"] + variant(shift=0.05),
              ["對照：單純持有比特幣"] + [_pct(hold["ret"])] * len(keys4)]
    table4 = {"caption": f"估價假設改一改，每年報酬差多少（{span}）。偏斜調整＝離現價越遠的約定價，用的波動率照當時市場"
                         "「下方比上方貴多少」往上或往下修；價平的約定價不受偏斜影響。價差＝賣出時比模型價少收的比例。",
              "columns": ["假設", "賣買權：價平", "賣買權：高 10%", "賣買權：高 20%", "賣賣權：價平", "賣賣權：低 10%", "輪流"],
              "rows": rows4}

    # ----- extra numbers for the notes -----
    by_year = []
    for year in sorted({_date(first + i).year for i in range(len(px)) if seen[i]}):
        pick = np.array([seen[i] and _date(first + i).year == year for i in range(len(px))])
        if pick.sum() >= 20:
            by_year.append(f"{year} 年 {vols[pick].mean() * 100:.0f}% 對 {realised[pick].mean() * 100:.0f}%"
                           f"（差 {(vols[pick].mean() - realised[pick].mean()) * 100:+.0f} 點）")
    v_lo, v_hi = _num(VENDOR[0]), _num(VENDOR[1])
    vendor = None
    if v_lo in monthly and v_hi in monthly:
        def total(key: str) -> float:
            eq = runs[key]["equity"]
            return float(eq[v_hi - first] / eq[v_lo - first] - 1)

        deep = {k: measure(data, runs[k], monthly, v_lo, v_hi, year_ends)["deepest"] for k in ("c0", "hold")}
        vendor = (f"坊間文章說 2024 年 6 月到 2026 年 5 月每月賣價平買權合計 +{VENDOR[2]}%、單純持有 +{VENDOR[3]}%，而且最深虧損較淺。"
                  f"本頁用同一段（{_ym(v_lo)} 到 {_ym(v_hi)}）重算：賣價平買權合計 {_pct(total('c0'))}、單純持有 {_pct(total('hold'))}，"
                  f"最深虧損 {_pct(deep['c0'])} 對 {_pct(deep['hold'])}。差距來自起訖日、成交價和費用假設不同，"
                  "對方用的細節沒有公開，無法完全對上。")
    atm_pairs = [(row[1] / 100, vols[d - first]) for d, row in data["skew"].items()
                 if row[1] is not None and first <= d <= data["last"]]
    atm_gap = float(np.mean([a - b for a, b in atm_pairs])) * 100 if len(atm_pairs) >= 20 else None

    # ----- today -----
    try:
        spot = float(common.get_json(f"{DERIBIT}/get_index_price", {"index_name": "btc_usd"})["result"]["index_price"])
    except Exception:  # noqa: BLE001 - fall back to the last delivery price
        spot = float(px[-1])
    last_dvol_day = max(data["dvol"])
    sigma = data["dvol"][last_dvol_day][1]
    slope_now = float(data["slope"][-1])
    k10 = spot * 1.10
    flat = price(spot, k10, 30 / YEAR, sigma, True)
    skewed = price(spot, k10, 30 / YEAR, _vol_at(sigma, slope_now, 1.10, 1.0), True)
    atm_now = price(spot, spot, 30 / YEAR, sigma, True)
    past_year = vols[-365:]
    now = [
        {"label": "比特幣現價（Deribit 指數）", "value": f"${spot:,.0f}"},
        {"label": f"DVOL：選擇權市場預期的一年波動（{_ym(last_dvol_day)}）",
         "value": f"{sigma * 100:.1f}%（過去一年中位數 {float(np.median(past_year)) * 100:.1f}%）"},
        {"label": "一個月後到期、約定價高 10% 的買權，模型估的權利金",
         "value": (f"現價的 {skewed / spot * 100:.2f}%（約 ${skewed:,.0f}；不做偏斜調整是 {flat / spot * 100:.2f}%）"
                   if data["has_skew"] else f"現價的 {flat / spot * 100:.2f}%（約 ${flat:,.0f}）")},
        {"label": "賣出這個買權的人，價格漲過哪裡以上的部分就不是他的", "value": f"${k10:,.0f}"},
        {"label": "漲過哪裡，賣方的結果會比單純持有差（約定價＋權利金）", "value": f"${k10 + skewed:,.0f}"},
        {"label": "同樣一個月、約定價＝現價的買權，模型估的權利金", "value": f"現價的 {atm_now / spot * 100:.2f}%"},
    ]
    if data["skew"]:
        latest = max(data["skew"])
        now.append({"label": f"最近一筆偏斜（{_ym(latest)}，下方 10% 比上方 10% 貴幾點）",
                    "value": f"{data['skew'][latest][0]:+.1f} 點"})

    # ----- verdict -----
    def side(a: float, b: float, tol: float, more: str, less: str) -> str:
        return "差不多" if abs(a - b) <= tol else more if a > b else less

    low = measure(data, simulate(data, monthly, mode="call", rule=("pct", 0.0), shift=-0.05), monthly, start, end,
                  year_ends)
    verdict = (
        f"{_ym(start)[:7]} 到 {_ym(end)[:7]} 共 {n_months} 個月（模型估價，不是實際成交價）：抱著比特幣每月賣價平買權"
        f"（收一筆權利金，漲過約定價的部分讓給買方）每年 {_pct(cc['ret'])}，單純持有 {_pct(hold['ret'])}，"
        f"賺得{side(cc['ret'], hold['ret'], 0.005, '比較多', '比較少')}，但成交的波動率只要比指數低 5 點就變成 {_pct(low['ret'])}；"
        f"最深虧損 {_pct(cc['deepest'])} 對 {_pct(hold['deepest'])}，{side(cc['deepest'], hold['deepest'], 0.02, '比較淺', '比較深')}，"
        f"漲最多的三個月持有合計賺 {_pct(rally_hold)}、賣買權只賺 {_pct(rally_cc)}，{n_months} 個月裡 {cc['hit']} 個月被履約。"
        f"抱美元每月賣價平賣權（收權利金，跌破約定價的部分要賠）每年 {_pct(csp['ret'])}、最深虧損 {_pct(csp['deepest'])}，"
        f"和賣價平買權幾乎相同；約定價低 10% 的賣權是 {_pct(whole['p10']['ret'])}、{_pct(whole['p10']['deepest'])}。"
        f"最慘的一個月：賣買權 {_pct(cc['worst'][2])}、賣賣權 {_pct(csp['worst'][2])}、單純持有 {_pct(hold['worst'][2])}。")
    if len(verdict) > 350:
        verdict = verdict.replace("（模型估價，不是實際成交價）", "（模型估價）").replace("（收權利金，跌破約定價的部分要賠）", "")

    # ----- notes -----
    notes = [
        "名詞：買權＝付一筆錢，買「之後能用約定價格買進」的權利；賣出買權＝收下那筆錢，代價是漲過約定價格的部分不是你的；"
        "賣權＝付一筆錢，買「之後能用約定價格賣出」的權利，賣出賣權就是收下那筆錢、跌破約定價格的部分由你賠；"
        "權利金＝買方付給賣方的那筆錢；價平＝約定價格就是當時的價格。",
        "這些都是用指數推算出來的「模型價」，不是實際成交過的選擇權價格：用 Deribit 的 DVOL（由選擇權報價算出的 30 天期、"
        "接近價平的波動率指數）套 Black-Scholes 公式（利率當 0）。實際能賣到的價格，尤其是離現價遠的約定價，可能差很多；"
        "實際的約定價也只有固定幾檔，不會剛好是現價或剛好高 10%。",
        "偏斜調整：離現價遠的約定價不是照 DVOL 成交。這裡用另一項統計存下來的偏斜歷史（每 4 天一筆，中間用直線補）把波動率照"
        "「離現價多遠」等比例修正；那是一條直線的粗估，實際上離得越遠兩邊都會翹起來，所以高 20% 那種約定價的權利金可能被低估。"
        + (f"同一份資料裡，實際成交估出的價平波動率平均比 DVOL {'高' if atm_gap >= 0 else '低'} {abs(atm_gap):.1f} 點。"
           if atm_gap is not None else "")
        + ("" if data["has_skew"] else "這次讀不到偏斜歷史，所有結果都沒有做偏斜調整。"),
        f"費用假設：賣出時比模型價少收 {SPREAD * 100:.0f}%（買賣價差），再扣 Deribit 公布的選擇權手續費（標的價值的 0.03%，"
        "最多不超過權利金的 12.5%），到期被履約另扣交割費（0.015%，同樣上限）。手續費是照公開費率表假設的，各人實際費率不同。",
        "做法細節：每月最後一個週五 08:00（UTC，Deribit 的到期時間）用 Deribit 當天的交割價賣出、下個月最後一個週五用交割價結算；"
        "每週的版本是每個週五到下個週五，一樣用 DVOL 依時間長短換算，但短天期的波動率實際上和 30 天指數不同，每週的數字更粗。"
        "DVOL 取賣出當天 00:00 的值。全部以美元計，每次換約時把收到的權利金和結算差額併回部位（賣買權的做法等於換成比特幣繼續抱）；"
        "Deribit 實際上是用比特幣收付權利金，這裡沒有模擬。賣賣權的做法每月賣出「等於當時帳上美元價值」那麼多比特幣的賣權，"
        "賠了就以美元認賠、下個月繼續賣；美元沒有算利息，所以賣賣權的數字比實際略低。"
        "在這種算法下，抱幣賣價平買權和抱美元賣價平賣權的損益是同一條算式（漲的時候都只賺到權利金、跌的時候都照跌再加回權利金），"
        "所以表裡這兩列幾乎一樣，不是兩個獨立的結果。",
        "每週的兩列最不可靠：每週賣的選擇權只剩 7 天，實際成交的波動率和 30 天指數常常差很多（平靜時通常更低），"
        "而且一年要賣五十幾次，估價只要偏一點、價差只要大一點，結果就差很多；那兩列的高報酬主要反映的是這個估價假設，不能當成做得到的數字。",
        "每年波動和最深虧損是用同一個模型每天替還沒到期的選擇權估價算出來的，同樣不是實際市價；最差的一個月、被履約次數只看結算日。"
        "「模型估四次約一次被履約的價位」就是業界說的 25-delta，「十次約一次」是 10-delta。"
        "「輪流」就是俗稱的 wheel。順勢規則用的是本站「把很多順勢規則混在一起」那項統計的看多比例（賣出當天 00:00 收盤時的值）；"
        "DVOL 高低是和之前 365 天的中位數比（不滿 30 天時不賣）。"
        + ("" if trend is not None else "這次讀不到順勢規則的資料，相關兩列沒有列出。"),
        f"報酬拆解（每月賣價平買權，各月相加後換算成每年，沒有算複利）：抱著比特幣本身 {_pct(cc['hold_part'])}、"
        f"收到的權利金 {_pct(cc['premium'])}、漲過約定價賠給買方 {_pct(-cc['give'])}，合計 {_pct(cc['hold_part'] + cc['premium'] - cc['give'])}。"
        f"抱美元賣價平賣權：權利金 {_pct(csp['premium'])}、跌破約定價賠出去 {_pct(-csp['give'])}。",
        "保險的加價（DVOL 平均 對 之後 30 天實際波動的平均）逐年：" + "；".join(by_year) + "。",
    ]
    if vendor:
        notes.append(vendor)
    notes += [
        f"資料只有大約 {(end - start) / 365.25:.1f} 年，裡面只有一次大空頭（2022 年）。賣選擇權的特性是平常小賺、偶爾大賠，"
        "歷史這麼短，很可能還沒遇到最糟的情況，表裡的數字會低估風險。賣買權只多收一筆權利金，比特幣大跌時照樣跟著跌。",
        "要這樣做需要在交易所開選擇權帳戶，交易所出事、被強制平倉或提不出錢的風險都不在這些數字裡；稅和費用也因人而異。",
        "過去的結果不保證未來會一樣。這裡只是整理各種做法過去的結果，不是買賣建議。",
    ]

    return {
        "id": ID,
        "title": "抱著比特幣每月賣買權收權利金，比單純抱著好嗎？",
        "question": "如果一個持有比特幣的人，每個月都賣出買權收權利金（或抱著美元賣出賣權），幾年下來和單純持有比，結果差多少？什麼時候出事？",
        "asof": _ym(data["last"]),
        "verdict": verdict,
        "now": now,
        "tables": [table1, table2, table3, table4],
        "notes": notes,
        "sources": [
            {"name": "Deribit 每日交割價（btc_usd，08:00 UTC）",
             "url": "https://www.deribit.com/api/v2/public/get_delivery_prices?index_name=btc_usd"},
            {"name": "Deribit DVOL 比特幣波動率指數（日線）",
             "url": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=1D"},
            {"name": "Deribit 選擇權歷史成交（本站偏斜歷史的來源）", "url": "https://history.deribit.com/"},
            {"name": "Deribit 手續費說明", "url": "https://www.deribit.com/kb/fees"},
            {"name": "幣安現貨 BTCUSDT 日線（順勢規則用）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
