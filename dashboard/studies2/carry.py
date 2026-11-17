"""Historical study "carry": what did "hold the coin, short the same amount of its perpetual contract, collect the
funding" actually pay, year by year, and when did it lose?

For BTCUSDT, ETHUSDT and SOLUSDT on Binance: every funding settlement since each contract's first record, plus daily
closes of the spot pair and of the perpetual contract.

1. Raw funding collected by a short perpetual + long spot position (the sum of the funding rates; a positive rate
   means shorts receive), per calendar year, with the share of settlements that were negative, the longest run of
   days with a negative 7-day funding and the worst 30/90 days.
2. A rough realistic version: 1 unit of spot + 0.5 unit of margin kept as cash (so everything is earned on 1.5 units
   of capital), 0.10% cost per leg each way, the daily change of the perpetual-minus-spot gap marked to market, and
   a rule variant that only holds the position while the 7-day average funding is above a threshold. Compared with
   3-month US Treasury bills (FRED DTB3) and with simply holding bitcoin.
3. What the averages hide: the range of the basis, the largest rallies the short leg had to sit through and how often
   a short with 50% margin would have come within 10% of its liquidation price.
4. Whether the yield has decayed: yearly figures, the last 12 months, the last 90 days and the latest settlement.

No look-ahead in the rule variant: the 7-day average known at the close of day t decides the position held during
day t+1. A settlement stamped 00:00 UTC is counted in the day that just ended (it pays for the last 8 hours of that
day), so a day's funding and its 00:00 UTC close cover the same 24 hours. Only finished days are used. Everything
comes from public, key-less endpoints with the fixed user agent of dashboard.common.

Background: He, Manela, Ross, von Wachter, "Fundamentals of Perpetual Futures" (arXiv 2212.06888); Schmeling,
Schrimpf, Todorov, "Crypto carry" (BIS Working Paper 1087).
"""
from __future__ import annotations

import io
import json
import math

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import price

FAPI = "https://fapi.binance.com"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
COINS = (("BTCUSDT", "比特幣"), ("ETHUSDT", "以太幣"), ("SOLUSDT", "Solana"))
YEAR = 365
SPOT_UNITS = 1.0  # capital in the spot leg
MARGIN_UNITS = 0.5  # capital kept as margin for the short (cash, earns nothing)
CAPITAL = SPOT_UNITS + MARGIN_UNITS
LEG_COST = 0.001  # 0.10% of the amount traded, per leg, each way
SWITCH_COST = 2 * LEG_COST / CAPITAL  # opening (or closing) both legs, as a share of total capital
THRESHOLDS = (0.0, 0.005)  # % per 8 hours: hold only while the 7-day average funding is above this
MAINTENANCE = 0.005  # rough maintenance margin of the short (0.4-0.5% in the lowest Binance tier)
LIQ_AT = (1 + MARGIN_UNITS) / (1 + MAINTENANCE)  # price ÷ entry price at which a 50%-margin short is liquidated
WARN_AT = 0.9 * LIQ_AT  # "within 10% of the liquidation price"
MIN_RECORDS = 1000
FIRST_POSSIBLE = "2019-09-01"  # Binance's first USDT perpetual (BTCUSDT) started on 2019-09-08


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, digits: int = 1, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.5 * 10 ** -digits:
        return f"{0:.{digits}f}%"
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _funding(symbol: str) -> pd.Series:
    """Every funding settlement of one contract since its first record: rate as a fraction, indexed by UTC time."""
    # startTime 0 is treated as "not given" (returns the latest 500), so start just before the first contract existed
    out, start = [], int(pd.Timestamp(FIRST_POSSIBLE, tz="UTC").timestamp() * 1000)
    for _ in range(40):
        page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start, "limit": 1000})
        if not page:
            break
        out += page
        start = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    if len(out) < MIN_RECORDS:
        raise RuntimeError(f"{symbol} 資金費率歷史只回傳 {len(out)} 筆")
    series = pd.Series([float(p["fundingRate"]) for p in out],
                       index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
    return series[~series.index.duplicated()].sort_index()


def _klines(symbol: str, start: pd.Timestamp, perp: bool) -> pd.DataFrame:
    """Daily high and close since `start` (UTC days), spot or perpetual, oldest first."""
    rows, since = [], int(start.timestamp() * 1000)
    limit = 1500 if perp else 1000
    for _ in range(12):
        params = {"symbol": symbol, "interval": "1d", "startTime": since, "limit": limit}
        page = common.get_json(FAPI + "/fapi/v1/klines", params) if perp else price._binance("/api/v3/klines", params)
        if not page:
            break
        rows += page
        since = page[-1][0] + 1
        if len(page) < limit:
            break
    if len(rows) < 300:
        raise RuntimeError(f"{symbol} {'永續' if perp else '現貨'}日線只有 {len(rows)} 筆")
    frame = pd.DataFrame({"high": [float(r[2]) for r in rows], "close": [float(r[4]) for r in rows]},
                         index=pd.to_datetime([r[0] for r in rows], unit="ms"))
    return frame[~frame.index.duplicated()].sort_index()


def _tbill() -> pd.Series:
    """FRED DTB3: 3-month US Treasury bill rate, % per year, one value per business day."""
    reply = requests.get(FRED, params={"id": "DTB3", "cosd": "2019-01-01"}, headers=common.HEADERS, timeout=25)
    reply.raise_for_status()
    frame = pd.read_csv(io.StringIO(reply.text))
    rate = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    series = pd.Series(rate.values, index=pd.to_datetime(frame.iloc[:, 0], errors="coerce")).dropna()
    series = series[series.index.notna()].sort_index()
    if len(series) < 500:
        raise RuntimeError(f"FRED DTB3 只回傳 {len(series)} 筆")
    return series


def _load(symbol: str) -> dict:
    """Settlements plus one row per finished UTC day: funding of the day, spot close, perpetual close and high."""
    rates = _funding(symbol)
    today = pd.Timestamp(common.today_utc())
    day_of = (rates.index - pd.Timedelta(minutes=1)).floor("D")  # the 00:00 settlement belongs to the day just ended
    first = rates.index[0].floor("D")
    spot = _klines(symbol, first - pd.Timedelta(days=1), perp=False)
    perp = _klines(symbol, first - pd.Timedelta(days=1), perp=True)
    start = max(first, spot.index[0], perp.index[0])
    days = pd.date_range(start, today - pd.Timedelta(days=1), freq="D")
    if len(days) < 400:
        raise RuntimeError(f"{symbol} 可用的日子只有 {len(days)} 天")
    funding = rates.groupby(day_of).sum().reindex(days).fillna(0.0)
    frame = pd.DataFrame({
        "funding": funding,
        "spot": spot["close"].reindex(days).ffill(),
        "perp": perp["close"].reindex(days).ffill(),
        "high": perp["high"].reindex(days),
    })
    frame["high"] = frame["high"].fillna(frame["perp"])
    if frame[["spot", "perp"]].isna().any().any() or (frame[["spot", "perp"]] <= 0).any().any():
        raise RuntimeError(f"{symbol} 日線有缺")
    kept = rates[(day_of >= days[0]) & (day_of <= days[-1])]
    gap_hours = float(np.median(np.diff(rates.index[-10:].values).astype("timedelta64[s]").astype(float))) / 3600
    return {"symbol": symbol, "rates": kept, "rate_day": day_of[(day_of >= days[0]) & (day_of <= days[-1])],
            "frame": frame, "latest_rate": float(rates.iloc[-1]), "latest_time": rates.index[-1],
            "per_year": YEAR * 24 / gap_hours if gap_hours > 0 else YEAR * 3, "gap_hours": gap_hours}


# ---------- measuring ----------

def _measure(net: pd.Series, position: pd.Series | None = None) -> dict:
    values = net.values.astype(float)
    n = len(values)
    growth = float(np.prod(1 + values))
    ret = (growth ** (YEAR / n) - 1) * 100 if n and growth > 0 else None
    vol = float(np.std(values, ddof=1) * math.sqrt(YEAR) * 100) if n > 1 else None
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    years = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    out = {"days": n, "ret": ret, "vol": vol, "deepest": deepest, "years": years,
           "up_years": sum(1 for v in years.values() if v > 0), "n_years": len(years)}
    if position is not None:
        out["share"] = float(position.mean() * 100)
        moved = position.diff().abs()
        moved.iloc[0] = position.iloc[0]
        out["entries"] = int(round(float(((position.diff() > 0) | ((position.index == position.index[0])
                                                                 & (position > 0))).sum())))
    return out


def _annual(net: pd.Series) -> float | None:
    """Compound return of a run of daily returns, converted to one year, in %."""
    if len(net) < 2:
        return None
    growth = float(np.prod(1 + net.values))
    return (growth ** (YEAR / len(net)) - 1) * 100 if growth > 0 else None


def _hedged(frame: pd.DataFrame, threshold: float | None) -> tuple[pd.Series, pd.Series]:
    """(daily net return on total capital, position 0/1) from the second day on.

    The position is entered at the close of the first day. Each day it earns that day's funding plus the change of
    the spot-minus-perpetual gap, on 1 unit of spot out of 1.5 units of capital (rebalanced daily to that split).
    threshold None = always hold; otherwise hold during day t+1 only if the 7-day average funding at the close of
    day t is above the threshold (% per 8 hours).
    """
    funding = frame["funding"]
    gross = (funding + frame["spot"].pct_change() - frame["perp"].pct_change()) / CAPITAL
    if threshold is None:
        signal = pd.Series(1.0, index=frame.index)
    else:
        avg7 = funding.rolling(7, min_periods=7).sum() / 21 * 100  # % per 8 hours
        signal = (avg7.round(8) > threshold).astype(float)  # the first six days: not enough history, stay in cash
    position = signal.shift(1).iloc[1:]
    moved = position.diff().abs()
    moved.iloc[0] = position.iloc[0]  # starts from cash
    net = position * gross.iloc[1:] - SWITCH_COST * moved
    net.iloc[-1] -= SWITCH_COST * position.iloc[-1]  # close whatever is still open on the last day
    return net, position


def _negative_run(funding: pd.Series) -> dict | None:
    """Longest run of consecutive days whose trailing 7-day funding sum is negative, and the funding paid in it."""
    week = funding.rolling(7, min_periods=7).sum()
    negative = (week.round(10) < 0).values
    best, start = (0, None, None), None
    for i, flag in enumerate(negative):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(negative) - 1):
            end = i if flag else i - 1
            if end - start + 1 > best[0]:
                best = (end - start + 1, start, end)
            start = None
    if not best[0]:
        return None
    length, a, b = best
    return {"days": length, "first": funding.index[a], "last": funding.index[b],
            "cost": float(funding.iloc[a:b + 1].sum() * 100), "count_days": int(negative.sum())}


def _worst(funding: pd.Series, n: int) -> dict | None:
    total = funding.rolling(n, min_periods=n).sum().dropna()
    if total.empty:
        return None
    end = total.idxmin()
    return {"sum": float(total.min() * 100), "first": end - pd.Timedelta(days=n - 1), "last": end}


def _margin_calls(frame: pd.DataFrame) -> dict:
    """A short opened with 50% margin and the same number of coins kept throughout. Each time the perpetual's daily
    high comes within 10% of the liquidation price counts once; margin is then topped back up to 50% at that day's
    close (the reference price moves there). Falls in price only add to the margin and move nothing."""
    close, high = frame["perp"].values, frame["high"].values
    reference, calls, liquidated, dates = close[0], 0, 0, []
    for i in range(1, len(close)):
        if high[i] >= reference * WARN_AT:
            calls += 1
            dates.append(frame.index[i])
            if high[i] >= reference * LIQ_AT:
                liquidated += 1
            reference = close[i]
    return {"calls": calls, "liquidated": liquidated, "dates": dates}


def _stats(data: dict) -> dict:
    frame, rates, rate_day = data["frame"], data["rates"], data["rate_day"]
    funding = frame["funding"]
    last = frame.index[-1]
    years = {}
    for year in sorted(set(frame.index.year)):
        chunk = funding[funding.index.year == year]
        settled = rates[rate_day.year == year]
        full = (chunk.index[0] == pd.Timestamp(f"{year}-01-01")) and (chunk.index[-1] == pd.Timestamp(f"{year}-12-31"))
        years[int(year)] = {"sum": float(chunk.sum() * 100), "days": len(chunk), "full": bool(full),
                            "annual": float(chunk.sum() * 100 * YEAR / len(chunk)),
                            "negative": float((settled < 0).mean() * 100) if len(settled) else None}

    def recent(n: int) -> float | None:
        return float(funding.iloc[-n:].sum() * 100 * YEAR / n) if len(funding) >= n else None

    spot = frame["spot"]
    basis = (frame["perp"] / spot - 1) * 100
    day_rise = spot.pct_change() * 100
    week_rise = (spot / spot.shift(7) - 1) * 100
    variants = {}
    for key, threshold in (("always", None),) + tuple((f"t{t:g}", t) for t in THRESHOLDS):
        net, position = _hedged(frame, threshold)
        variants[key] = _measure(net, position)
        variants[key]["net"] = net
    full_years = [y for y, v in years.items() if v["full"]]
    return {
        "first": frame.index[0], "last": last, "days": len(frame), "years": years, "full_years": full_years,
        "total": float(funding.sum() * 100), "annual": float(funding.sum() * 100 * YEAR / len(funding)),
        "negative": float((rates < 0).mean() * 100), "settlements": int(len(rates)),
        "run": _negative_run(funding), "worst30": _worst(funding, 30), "worst90": _worst(funding, 90),
        "last365": recent(365), "last90": recent(90), "last7": recent(7),
        "basis_min": float(basis.min()), "basis_min_day": basis.idxmin(), "basis_max": float(basis.max()),
        "basis_max_day": basis.idxmax(), "basis_now": float(basis.iloc[-1]),
        "basis_mid": float(basis.median()), "basis_wide": int((basis.abs() > 0.5).sum()),
        "day_rise": float(day_rise.max()), "day_rise_day": day_rise.idxmax(),
        "week_rise": float(week_rise.max()), "week_rise_day": week_rise.idxmax(),
        "margin": _margin_calls(frame), "variants": variants,
        "latest": data["latest_rate"] * 100, "latest_time": data["latest_time"],
        "latest_annual": data["latest_rate"] * 100 * data["per_year"], "gap_hours": data["gap_hours"],
    }


# ---------- the study ----------

def study() -> dict:
    problems: list[str] = []
    stats: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for symbol, name in COINS:
        try:
            data = _load(symbol)
            stats[symbol] = _stats(data)
            frames[symbol] = data["frame"]
        except Exception as exc:  # noqa: BLE001 - the other coins still stand
            if symbol == "BTCUSDT":
                raise
            problems.append(f"{name}（{symbol}）的資料這次取不到（{type(exc).__name__}），表裡這一欄留空，沒有改用其他方法繞過。")
    btc = stats["BTCUSDT"]
    last = btc["last"]
    have = [(s, n) for s, n in COINS if s in stats]

    try:
        bill = _tbill()
    except Exception as exc:  # noqa: BLE001 - the comparison falls back to a rough spoken figure
        bill = None
        problems.append(f"美國 3 個月國庫券利率（FRED DTB3）這次取不到（{type(exc).__name__}），"
                        "和放現金的比較只能用約略的數字：2023–2025 年大約每年 4–5%，2020–2021 年接近 0%。")

    always = btc["variants"]["always"]
    days = always["net"].index  # the days the bitcoin position was held
    hold_net = frames["BTCUSDT"]["spot"].pct_change().reindex(days)
    hold = _measure(hold_net)
    if bill is not None:
        bill_net = (bill.reindex(bill.index.union(days)).ffill().reindex(days) / 100 / YEAR).fillna(0.0)
        cash = _measure(bill_net)
        bill_now, bill_day = float(bill.iloc[-1]), bill.index[-1]
    else:
        bill_net, cash, bill_now, bill_day = None, None, None, None

    def span(s: dict) -> str:
        return f"{s['first'].strftime('%Y-%m')} 起"

    def year_name(y: int, s: dict = btc) -> str:
        info = s["years"].get(y)
        return str(y) if info is None or info["full"] else f"{y}（不滿一年）"

    all_years = sorted({y for s in stats.values() for y in s["years"]})

    # ----- table 1: funding by year and coin -----
    rows1 = []
    for y in all_years:
        row = [year_name(y)]
        for symbol, _ in COINS:
            info = stats.get(symbol, {}).get("years", {}).get(y)
            row += [_pct(info["sum"]) if info else "—", _pct(info["negative"], 0, signed=False) if info else "—"]
        rows1.append(row)
    total_row, mean_row = ["全部加起來"], ["平均換算成一年"]
    for symbol, _ in COINS:
        s = stats.get(symbol)
        total_row += [_pct(s["total"]) if s else "—", _pct(s["negative"], 0, signed=False) if s else "—"]
        mean_row += [_pct(s["annual"]) if s else "—", "—"]
    rows1 += [total_row, mean_row]
    columns1 = ["年份"]
    for _, name in COINS:
        columns1 += [f"{name}：收到的資金費", f"{name}：要倒付的結算比例"]
    starts = "、".join(f"{n} {_d(stats[s]['first'])}" for s, n in have)

    # ----- table 2: the realistic version -----
    def variant_row(label: str, s: dict, key: str) -> list:
        m = s["variants"][key]
        return [label, f"{span(s)}", _pct(m["ret"]), _pct(m["vol"], signed=False), _pct(m["deepest"]),
                f"{m['n_years']} 年裡 {m['up_years']} 年", f"{m['share']:.0f}%", f"{m['entries']} 次"]

    rule_names = {"always": "一直持有", "t0": "7 日平均資金費 > 0 才持有", "t0.005": "7 日平均資金費 > +0.005% 才持有"}
    rows2 = []
    for symbol, name in have:
        for key, text in rule_names.items():
            rows2.append(variant_row(f"{name}：{text}", stats[symbol], key))
    if cash is not None:
        rows2.append(["對照：錢放美國 3 個月國庫券", span(btc), _pct(cash["ret"]), _pct(cash["vol"], signed=False),
                      _pct(cash["deepest"]), f"{cash['n_years']} 年裡 {cash['up_years']} 年", "—", "—"])
    else:
        rows2.append(["對照：錢放美國 3 個月國庫券（這次取不到，約略值）", "—", "2023–2025 年約 4–5%", "—", "—", "—", "—", "—"])
    rows2.append(["對照：一直抱著比特幣", span(btc), _pct(hold["ret"]), _pct(hold["vol"], signed=False),
                  _pct(hold["deepest"]), f"{hold['n_years']} 年裡 {hold['up_years']} 年", "—", "—"])

    # ----- table 3: risk -----
    def cells(fn) -> list:
        return [fn(stats[s]) if s in stats else "—" for s, _ in COINS]

    def run_text(s: dict) -> str:
        r = s["run"]
        return f"{r['days']} 天（{_d(r['first'])} 到 {_d(r['last'])}）" if r else "沒有出現過"

    def worst_text(key: str):
        def inner(s: dict) -> str:
            w = s[key]
            return f"{_pct(w['sum'], 2)}（到 {_d(w['last'])}）" if w else "—"
        return inner

    def margin_text(s: dict) -> str:
        m = s["margin"]
        extra = f"，其中 {m['liquidated']} 次當天盤中就碰到強制平倉價" if m["liquidated"] else ""
        return f"{m['calls']} 次{extra}"

    rows3 = [
        ["資金費 7 日合計連續為負，最長多久"] + cells(run_text),
        ["那一段總共倒付多少"] + cells(lambda s: _pct(s["run"]["cost"], 2) if s["run"] else "—"),
        ["7 日合計為負的日子占全部天數"] + cells(lambda s: f"{(s['run']['count_days'] if s['run'] else 0) / s['days'] * 100:.0f}%"),
        ["資金費最差的 30 天合計"] + cells(worst_text("worst30")),
        ["資金費最差的 90 天合計"] + cells(worst_text("worst90")),
        ["永續比現貨便宜最多的一天（日收盤）"] + cells(lambda s: f"{_pct(s['basis_min'], 2)}（{_d(s['basis_min_day'])}）"),
        ["永續比現貨貴最多的一天（日收盤）"] + cells(lambda s: f"{_pct(s['basis_max'], 2)}（{_d(s['basis_max_day'])}）"),
        ["兩者價差超過 ±0.5% 的天數"] + cells(lambda s: f"{s['basis_wide']} 天（共 {s['days']} 天）"),
        ["現貨單日最大漲幅（放空那邊當天帳面虧這麼多）"] + cells(lambda s: f"{_pct(s['day_rise'])}（{_d(s['day_rise_day'])}）"),
        ["現貨 7 天最大漲幅"] + cells(lambda s: f"{_pct(s['week_rise'])}（到 {_d(s['week_rise_day'])}）"),
        ["放空用 50% 保證金：離強制平倉價不到 10% 的次數"] + cells(margin_text),
    ]

    # ----- table 4: decay and recent -----
    def period_row(label: str, mask_of) -> list:
        row = [label]
        for symbol, _ in COINS:
            s = stats.get(symbol)
            if not s:
                row.append("—")
                continue
            chunk = frames[symbol]["funding"][mask_of(frames[symbol].index)]
            row.append(_pct(float(chunk.sum() * 100 * YEAR / len(chunk))) if len(chunk) >= 30 else "—")
        mask = mask_of(days)
        part = always["net"][mask]
        row.append(_pct(_annual(part)))
        row.append(_pct(_annual(bill_net[mask]), 1, signed=False) if bill_net is not None else "—")
        held = hold_net[mask]
        row.append(_pct(float((np.prod(1 + held.values) - 1) * 100)) if len(held) else "—")
        return row

    rows4 = [period_row(year_name(y), lambda index, y=y: index.year == y) for y in sorted(btc["years"])]
    for label, n in (("最近 12 個月", 365), ("最近 90 天", 90)):
        cut = last - pd.Timedelta(days=n - 1)
        rows4.append(period_row(label, lambda index, cut=cut: index >= cut))
    latest_row = ["最新一次結算"]
    for symbol, _ in COINS:
        s = stats.get(symbol)
        latest_row.append(f"{_pct(s['latest_annual'])}（單次 {_pct(s['latest'], 4)}）" if s else "—")
    rows4.append(latest_row + ["—", f"{bill_now:.2f}%" if _ok(bill_now) else "—", "—"])

    tables = [
        {"caption": ("各年收到的資金費：買現貨、同時放空等量永續合約的人，把那一年每一次結算的費率直接加起來（正的是收到、負的是倒付），"
                     "還沒扣任何成本，也還沒考慮錢要分兩邊放。「要倒付的結算比例」是那一年所有結算裡費率是負的占幾成。"
                     f"各幣的資料起點：{starts}；最後一天是 {_d(last)}，頭尾兩年都不滿一年。"),
         "columns": columns1, "rows": rows1},
        {"caption": (f"比較實際的算法：每 1 單位現貨另外要留 {MARGIN_UNITS:g} 單位現金當放空的保證金（這筆現金假設沒有利息），"
                     f"所以報酬是算在 {CAPITAL:g} 單位的全部資金上；建立和結束部位時，現貨和合約兩邊各扣 {LEG_COST * 100:.2f}%；"
                     "每天再把「永續和現貨價差」的變動算進帳面。「才持有」的規則是：當天收盤時最近 7 天的平均資金費率（每 8 小時）"
                     "高於門檻，下一天才持有，否則全部換成現金（沒有利息），每次進出都扣成本。"),
         "columns": ["做法", "期間", "每年報酬", "每年波動", "最深曾經虧多少", "賺錢的年數", "持有部位的天數比例", "進場次數"],
         "rows": rows2},
        {"caption": ("平均數字看不到的部分。上半是資金費變成負的（要倒付）時有多久、多深；中間是永續合約價格和現貨價格的差距"
                     "（每天 UTC 0 點收盤）；下半是幣價急漲的幅度——急漲時現貨賺、放空那邊帳面虧，虧的那邊要立刻補保證金。"
                     f"最後一列的算法：放空時保證金是部位的 50%，價格比進場價高約 {(LIQ_AT - 1) * 100:.0f}% 會被強制平倉，"
                     f"高約 {(WARN_AT - 1) * 100:.0f}% 就算「離強制平倉不到 10%」；每碰到一次算一次，並假設當天收盤把保證金補回 50%。"),
         "columns": ["項目"] + [name for _, name in COINS], "rows": rows3},
        {"caption": ("收益有沒有變少。三個幣的資金費都換算成一年（不滿一年的期間按天數比例放大，直接相加、不算複利）；"
                     f"「比特幣：算在全部資金上」是上一張表「一直持有」的做法在那段期間的報酬（同樣換算成一年，"
                     f"{btc['first'].year} 年含建立部位的成本、{last.year} 年含結束部位的成本）；國庫券也換算成一年；"
                     "「抱著比特幣」是那段期間實際的漲跌，沒有換算。最後一列是最新一次結算的費率照樣維持一整年會是多少。"),
         "columns": ["期間"] + [f"{name}資金費（換算一年）" for _, name in COINS]
         + ["比特幣：算在全部資金上（換算一年）", "美國 3 個月國庫券（換算一年）", "抱著比特幣（期間漲跌）"],
         "rows": rows4},
    ]

    # ----- today -----
    now = []
    for symbol, name in have:
        s = stats[symbol]
        now.append({"label": f"{name}最新一次結算的資金費率（{s['latest_time'].strftime('%Y-%m-%d %H:%M')} UTC）",
                    "value": f"{_pct(s['latest'], 4)}，照這樣維持一年約 {_pct(s['latest_annual'])}"})
    now.append({"label": "最近 7 天的資金費換算成一年",
                "value": "；".join(f"{n} {_pct(stats[s]['last7'])}" for s, n in have)})
    now.append({"label": "最近 90 天的資金費換算成一年",
                "value": "；".join(f"{n} {_pct(stats[s]['last90'])}" for s, n in have)})
    now.append({"label": f"永續合約比現貨貴或便宜多少（{_d(last)} 收盤）",
                "value": "；".join(f"{n} {_pct(stats[s]['basis_now'], 2)}" for s, n in have)})
    if _ok(bill_now):
        now.append({"label": f"對照：美國 3 個月國庫券利率（{_d(bill_day)}）", "value": f"每年 {bill_now:.2f}%"})

    # ----- verdict -----
    full = btc["full_years"]
    sums = [btc["years"][y]["sum"] for y in full]
    positive = sum(1 for v in sums if v > 0)
    best_year = max(full, key=lambda y: btc["years"][y]["sum"])
    worst_year = min(full, key=lambda y: btc["years"][y]["sum"])
    run, w90 = btc["run"], btc["worst90"]
    if cash is not None:
        below = sum(1 for y in always["years"] if always["years"][y] <= cash["years"].get(y, 0.0))
        versus = (f"同一段期間把錢放美國 3 個月國庫券（接近美元定存）是每年 {_pct(cash['ret'])}，"
                  f"{always['n_years']} 年裡有 {below} 年這個做法還輸給國庫券")
        last_bill = _annual(bill_net[days >= last - pd.Timedelta(days=364)])
        if _ok(last_bill) and _ok(btc["last365"]):
            gross_now = btc["last365"] / CAPITAL
            recent_vs = (f"，分到全部資金上只剩約 {_pct(gross_now)}，"
                         f"{'高於' if gross_now > last_bill else '低於'}同期國庫券的 {_pct(last_bill)}")
        else:
            recent_vs = ""
    else:
        versus = "美國 3 個月國庫券（接近美元定存）這次取不到，約略是 2023–2025 年每年 4–5%、2020–2021 年接近 0%"
        recent_vs = ""
    verdict = (
        f"買比特幣現貨、同時放空等量永續合約、只收資金費：{len(full)} 個完整年度（{full[0]}–{full[-1]}）裡 {positive} 年收到的是正的，"
        f"單看資金費平均每年約 {_pct(float(np.mean(sums)))}（最多的 {best_year} 年 {_pct(btc['years'][best_year]['sum'])}、"
        f"最少的 {worst_year} 年 {_pct(btc['years'][worst_year]['sum'])}）；但錢要分成現貨和保證金兩份、再扣進出成本，"
        f"算在全部資金上是每年約 {_pct(always['ret'])}，{versus}，一直抱著比特幣則是每年 {_pct(hold['ret'])}"
        f"（中途最深跌 {_pct(hold['deepest'])}）。"
        f"現在收得比以前少：最近 12 個月的資金費換算一年約 {_pct(btc['last365'])}{recent_vs}；最近 90 天約 {_pct(btc['last90'])}。"
        f"最慘的時候："
        + (f"資金費 7 日合計連續 {run['days']} 天是負的（{_d(run['first'])} 到 {_d(run['last'])}），那段倒付 {_pct(abs(run['cost']), 2, signed=False)}；"
           if run else "")
        + (f"最差的 90 天合計 {_pct(w90['sum'], 2)}；" if w90 else "")
        + f"算在全部資金上帳面最深虧過 {_pct(always['deepest'], 2)}——數字看起來小，是因為交易所出事、急漲時被追繳保證金這些風險都不在這些數字裡。"
    )

    margin_parts = "、".join(f"{n} {stats[s]['margin']['calls']} 次" for s, n in have)
    notes = [
        "資金費率是什麼：永續合約沒有到期日，交易所每隔 8 小時讓做多和放空的人互相付一次「資金費」，把合約價格拉回現貨價格附近；"
        "費率是正的時候做多的人付給放空的人，負的時候反過來。",
        "這裡算的部位：買現貨、同時放空等量的永續合約，漲跌互相抵銷，只收資金費。費率是正的就收錢，是負的就要倒付。"
        "表裡「收到的資金費」是把每一次結算的費率直接相加，不算複利。",
        "這個做法需要開合約帳戶，而且錢要一直放在交易所裡。交易所倒閉、被駭客入侵或凍結提領是最主要的風險，"
        "發生時可能兩邊的錢都拿不回來——這個風險完全不在表裡的數字中。這裡只用了幣安的資料，幣安這段期間沒有發生這種事，不代表不會發生。",
        "合約是用 USDT 這種穩定幣計價和結算的，保證金和收到的資金費都是 USDT；穩定幣如果和美元脫鉤，損失也不在這些數字裡。",
        "資金費可能連續幾個月都是負的或接近零（見第三張表），那段期間這個部位是倒貼錢的。"
        f"比特幣歷史上 7 日合計為負的日子占全部天數的 {(btc['run']['count_days'] if btc['run'] else 0) / btc['days'] * 100:.0f}%。",
        "幣價急漲時會被追繳保證金：現貨那邊賺的是幣的帳面價值，放空那邊虧的是保證金裡的現金，兩邊不會自動互補，"
        "要自己把錢補進合約帳戶（或賣掉一部分現貨、同時縮小放空），來不及補就會被強制平倉，之後幣價再跌回來時就只剩現貨單邊在賠。"
        f"照第三張表的算法，離強制平倉不到 10% 的次數：{margin_parts}。這是用永續合約每天的最高價算的，盤中更短時間的急拉和"
        "交易所實際的強平規則（維持保證金率隨部位大小不同）沒有算進去，是約略的數字。",
        f"資金分配和成本都是粗略的假設：1 單位現貨配 {MARGIN_UNITS:g} 單位保證金（保證金假設完全沒有利息）、每天維持這個比例、"
        f"進出時兩邊各扣 {LEG_COST * 100:.2f}%。保證金留得少一點報酬率會高一點、但更容易被追繳；實際的手續費、買賣價差、"
        "滑價和借貸條件隨交易所、金額、帳戶等級不同。稅和費用每個人都不一樣，這裡都沒有算。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「每年波動」是每天帳面起伏換算成一年；「最深曾經虧多少」是帳面從最高點往下掉最多的幅度。"
        "帳面起伏主要來自永續和現貨的價差每天在變，用的是每天 UTC 0 點的收盤價，盤中更大的價差沒有算到。"
        "「賺錢的年數」把頭尾兩個不滿一年的年份也算成一年。",
        "「才持有」的規則用的是當天收盤時已經知道的最近 7 天平均資金費率，決定下一天要不要持有；門檻（0 和每 8 小時 +0.005%）"
        "是事先指定的兩個簡單數字，沒有挑過。標示 00:00 UTC 的結算算在剛結束的那一天。"
        + ("少數時期幣安曾把個別合約的結算改成比 8 小時更密，這裡一律照實際每次結算相加。" if any(
            abs(stats[s]["gap_hours"] - 8) > 0.5 for s, _ in have) else ""),
        "和國庫券的比較：美國 3 個月國庫券利率（FRED 的 DTB3）每天的年利率除以 365 當作當天的利息，大致等於把美元放在最安全的短期工具裡；"
        "它和台幣定存不是同一件事，也沒有算匯率。和「抱著比特幣」的比較只是對照規模，兩者承擔的風險完全不同。",
        "只看了幣安一家交易所、三個幣；其他交易所的資金費率和規則不同。早年（2019–2021）資金費很高的時期，"
        "市場規模和參與者都和現在不一樣，做這件事的資金變多之後收益被壓低是研究文獻裡也提到的現象。",
        "過去的結果不保證未來會一樣。這裡只是整理這個部位過去付了多少、什麼時候賠，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "幣安 U 本位永續合約資金費率歷史",
         "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
        {"name": "幣安現貨日線（BTCUSDT、ETHUSDT、SOLUSDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "幣安永續合約日線", "url": "https://www.binance.com/zh-TC/futures/BTCUSDT"},
        {"name": "FRED：美國 3 個月國庫券利率（DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"},
        {"name": "He, Manela, Ross, von Wachter：Fundamentals of Perpetual Futures（arXiv 2212.06888）",
         "url": "https://arxiv.org/abs/2212.06888"},
        {"name": "國際清算銀行工作論文 1087：Crypto carry", "url": "https://www.bis.org/publ/work1087.htm"},
    ]

    return {
        "id": "carry",
        "title": "不猜漲跌、只收資金費，過去實際收到多少",
        "question": "買現貨、同時放空等量的永續合約，讓漲跌互相抵銷、只收資金費——這個做法過去每年實際收到多少？什麼時候是賠的？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
