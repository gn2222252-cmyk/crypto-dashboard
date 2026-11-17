"""Historical study "leveraged_trend": the 20/10 trend rule when the thing held is a 2x daily-leveraged fund.

The rule (fixed on bitcoin, never tuned here): hold after a 20-day closing high, cash after a 10-day closing low.

What is built
  * A synthetic daily-leveraged fund for each underlying with a long history: daily return = L x the underlying's
    daily return - daily cost. The first cost guess is the dashboard's FUND_COST (1.05% fee + 5% financing of the
    borrowed half, per year). That guess is checked against the real funds (MSTU, MSTX, BITX, BITU, ETHU, CONL, SSO)
    over their common dates; where the real fund fell clearly short of it, the cost is replaced by the cost that
    makes the synthetic match the real fund ("implied cost"), and both are reported.
  * Bitcoin and ether use Binance spot daily candles sampled on US trading days only (the candle labelled with the
    trading day, which closes at 00:00 UTC, three to four hours after the US close), because the funds only trade
    and reset on those days. Everything is annualised with 252 days.
  * Four ways over the same dates: hold the asset, hold the 2x fund, rule on the asset, rule on the 2x fund (signal
    from the underlying's closes, and separately from the fund's own closes), plus "half the money in the 2x fund
    under the rule" (half the money goes in at each entry and is left alone until the exit).

No look-ahead: a signal uses closes up to and including day t and decides the position held from the close of day t
to the close of the next trading day. For the REAL bitcoin fund the bitcoin signal known at the US close of trading
day p is taken from the candle of the calendar day before (closed about 20 hours earlier). Costs: 0.10% of the
amount moved, cash earns nothing, buy-and-hold pays no trading cost. Public key-less endpoints only, through the
dashboard's helpers (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.collectors import studies as old_studies

COST = 0.001  # 0.10% of the amount moved
YEAR = 252
R_IN, R_OUT = 20, 10
FEE_YEAR = 0.0105  # the fee part of the dashboard's rough 2x cost
BASE_COST_YEAR = old_studies.FUND_COST * 252  # 1.05% fee + 5% financing of the borrowed half = 6.05% a year
ADJUST_IF_OFF = 0.02  # replace the first guess when the real funds imply a cost more than 2 points away from it
FLAT_DAYS, FLAT_BAND = 60, 0.05
LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
MSTR_SINCE = old_studies.MSTR_SINCE
CONL_2X_SINCE = "2024-02-01"  # CONL aimed at 1.5x before early 2024 (its daily moves were 1.5x COIN's until then)

# key -> (label, kind, source symbol)
UNDERLYINGS = (("BTC", "比特幣", "crypto", "BTCUSDT"), ("MSTR", "MSTR", "share", "MSTR"),
               ("ETH", "以太幣", "crypto", "ETHUSDT"), ("COIN", "COIN", "share", "COIN"),
               ("SPY", "美股大盤 SPY", "share", "SPY"))
LABEL = {key: label for key, label, _, _ in UNDERLYINGS}
LEAD = {key: label + (" " if label[-1].isascii() else "") for key, label, _, _ in UNDERLYINGS}  # before more text
REAL_FUNDS = (("MSTU", "MSTR"), ("MSTX", "MSTR"), ("BITX", "BTC"), ("BITU", "BTC"), ("ETHU", "ETH"),
              ("CONL", "COIN"), ("SSO", "SPY"))
FUND_SINCE = {"CONL": CONL_2X_SINCE}
REAL_TESTS = (("MSTU", "MSTR"), ("BITX", "BTC"))
SCALE_KEYS = ("BTC", "MSTR")


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
    return f"{v:,.2f}" if v >= 1 else f"{v:.4f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _span(days) -> str:
    if not _ok(days):
        return "—"
    days = float(days)
    return f"{days / 365.25:.1f} 年" if days >= 365 else f"{days:.0f} 天"


def _times(v) -> str:
    """Final money as a multiple of the start."""
    if not _ok(v):
        return "—"
    v = float(v)
    if v >= 100:
        return f"{v:,.0f} 倍"
    return f"{v:.1f} 倍" if v >= 10 else f"{v:.2f} 倍"


def _lev(level: float) -> str:
    return f"{level:g}"


# ---------- data ----------

def _crypto_daily(symbol: str) -> pd.Series:
    """Finished Binance spot daily closes, one row per calendar day (UTC)."""
    rows, start = [], 0
    for _ in range(12):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    series = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms").normalize())
    series = series[~series.index.duplicated()].sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    series = series[series > 0]
    if len(series) < 400:
        raise RuntimeError(f"幣安 {symbol} 日線只有 {len(series)} 筆")
    return series.asfreq("D").ffill()


def _share_close(symbol: str) -> pd.Series:
    """Finished adjusted daily closes from Yahoo Finance (trading days only, up to ten years)."""
    series = old_studies._yahoo(symbol).astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]  # a bar dated before today (UTC) is finished
    return series[series > 0]


def _load() -> tuple[dict, list[str]]:
    """{"daily": calendar-day crypto closes, "close": closes on US trading days, "funds": real fund closes}."""
    data: dict = {"daily": {}, "close": {}, "funds": {}}
    problems: list[str] = []
    shares = [(key, symbol) for key, _, kind, symbol in UNDERLYINGS if kind == "share"]
    for key, symbol in shares:
        try:
            data["close"][key] = _share_close(symbol)
        except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the others
            problems.append(f"{LABEL[key]}這次取不到資料（{type(exc).__name__}），沒有算進去。")
    for fund, _ in REAL_FUNDS:
        try:
            data["funds"][fund] = _share_close(fund)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{fund} 這次取不到資料（{type(exc).__name__}），沒有算進去。")
    calendar = next((data["close"][k].index for k in ("SPY", "MSTR", "COIN") if k in data["close"]), None)
    for key, _, kind, symbol in UNDERLYINGS:
        if kind != "crypto":
            continue
        try:
            daily = _crypto_daily(symbol)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{LABEL[key]}這次取不到資料（{type(exc).__name__}），沒有算進去。")
            continue
        days = calendar if calendar is not None else pd.bdate_range(daily.index[0], daily.index[-1])
        data["daily"][key] = daily
        data["close"][key] = daily.reindex(days).dropna()  # the candle labelled with each US trading day
    return data, problems


# ---------- the rule ----------

def _rule(close: pd.Series) -> pd.Series:
    """1 after an R_IN-day closing high, 0 after an R_OUT-day closing low, unchanged in between; cash until the
    first high; NaN while there are fewer than R_IN closes. Value at row t uses closes up to and including t."""
    high = close.rolling(R_IN, min_periods=R_IN).max()
    low = close.rolling(R_OUT, min_periods=R_OUT).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0).where(high.notna())


def _flip(close: pd.Series, state: float) -> float:
    """The next close that would flip the rule (see trend_assets._flip)."""
    if state == 1:
        return float(close.iloc[-(R_OUT - 1):].min())
    return float(close.iloc[-(R_IN - 1):].max())


def _fund(change: pd.Series, level: float, cost_year: float) -> pd.Series:
    """Daily returns of a synthetic fund that resets to `level` times the underlying every day. A day that would
    lose more than everything loses everything (the fund is wiped out)."""
    return (level * change - cost_year / YEAR).clip(lower=-1.0)


def _cost_at(level: float, cost2: float) -> float:
    """Yearly cost at a leverage level, given the cost of the 2x fund: the fee plus financing that grows with the
    borrowed part (level - 1)."""
    return FEE_YEAR + (level - 1.0) * max(cost2 - FEE_YEAR, 0.0)


def _held(change: pd.Series, signal: pd.Series, first) -> tuple[pd.Series, pd.Series]:
    """(net daily return, position) from `first` on. signal: value known at each row's close."""
    position = signal.shift(1).reindex(change.index)
    position, change = position[position.index >= first], change[change.index >= first]
    moved = position.diff().abs()
    moved.iloc[0] = position.iloc[0]  # starts from cash
    return (position * change - COST * moved).clip(lower=-1.0), position


def _half(change: pd.Series, position: pd.Series) -> pd.Series:
    """Half of the money goes into the fund at each entry and is left alone until the exit (no daily rebalancing)."""
    cash, fund, out = 1.0, 0.0, []
    for r, p in zip(change.values, position.values):
        before = cash + fund
        if p == 1 and fund == 0.0:
            stake = 0.5 * before
            cash, fund = before - stake - COST * stake, stake
        elif p != 1 and fund > 0.0:
            cash, fund = cash + fund * (1 - COST), 0.0
        fund *= 1 + r
        out.append((cash + fund) / before - 1)
    return pd.Series(out, index=change.index)


# ---------- measuring ----------

def _measure(net: pd.Series, position: pd.Series | None = None) -> dict:
    n = len(net)
    values = net.values
    growth = float(np.prod(1 + values))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(values, ddof=1) * math.sqrt(YEAR) * 100)
    equity = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    deepest = float(min((equity / peak - 1).min(), 0.0) * 100)
    # longest stretch below an earlier high, in calendar days (still counting if it never came back)
    stamps = net.index.values.astype("datetime64[D]").astype(np.int64)
    at_high = equity >= peak
    last_high = np.maximum.accumulate(np.where(at_high, stamps, stamps[0] - 1))
    under = int((stamps - last_high).max())
    years = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    worst_year = min(years, key=years.get)
    return {"days": n, "ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "deepest": deepest,
            "under": under, "worst": years[worst_year], "worst_year": worst_year, "growth": growth,
            "share": (float(position.mean() * 100) if position is not None else 100.0)}


def _tracking(fund: pd.Series, change: pd.Series, since: str | None) -> dict | None:
    """A real 2x fund against the synthetic one over their common dates."""
    frame = pd.DataFrame({"f": fund.pct_change(), "u": change}).dropna()
    if since:
        frame = frame[frame.index >= since]
    if len(frame) < 120:
        return None

    def log_growth(part: pd.DataFrame, cost: float) -> float:
        return float(np.log1p((2 * part.u - cost / YEAR).clip(lower=-0.999)).sum())

    def implied(part: pd.DataFrame) -> float:
        real, lo, hi = float(np.log1p(part.f.clip(lower=-0.999)).sum()), -1.0, 5.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if log_growth(part, mid) > real else (lo, mid)
        return (lo + hi) / 2

    years = len(frame) / YEAR
    real_year = (math.exp(float(np.log1p(frame.f.clip(lower=-0.999)).sum()) / years) - 1) * 100
    sim_year = (math.exp(log_growth(frame, BASE_COST_YEAR) / years) - 1) * 100
    return {"first": frame.index[0], "last": frame.index[-1], "days": len(frame), "real": real_year, "sim": sim_year,
            "gap": real_year - sim_year, "implied": implied(frame) * 100,
            "recent": (implied(frame.tail(YEAR)) * 100 if len(frame) > YEAR + 60 else None),
            "slope": float(np.polyfit(frame.u.values, frame.f.values, 1)[0])}


def _four(change: pd.Series, fund_change: pd.Series, signal: pd.Series, own_signal: pd.Series, since: str | None,
          keep: bool = False) -> dict | None:
    """The four ways (and the half-money one) over the same dates."""
    frame = pd.DataFrame({"u": change, "f": fund_change, "s": signal.shift(1), "o": own_signal.shift(1)}).dropna()
    if since:
        frame = frame[frame.index >= since]
    if len(frame) < 60:
        return None
    first = frame.index[0]
    u, f = frame.u, frame.f
    asset_net, asset_pos = _held(u, signal, first)
    fund_net, fund_pos = _held(f, signal, first)
    own_net, own_pos = _held(f, own_signal, first)
    half_net = _half(f, fund_pos)
    out = {"first": first, "last": frame.index[-1], "days": len(frame),
           "hold": _measure(u), "hold2": _measure(f), "rule": _measure(asset_net, asset_pos),
           "rule2": _measure(fund_net, fund_pos), "own2": _measure(own_net, own_pos),
           "half": _measure(half_net, fund_pos * 0.5),
           "half_daily": _measure((0.5 * fund_pos * f - COST * 0.5 * fund_pos.diff().abs().fillna(fund_pos.iloc[0]))
                                  .clip(lower=-1.0), fund_pos * 0.5)}
    if keep:
        out["series"] = {"u": u, "f": f, "pos": fund_pos, "rule": asset_net, "rule2": fund_net}
    return out


def _flat(series: dict) -> dict | None:
    """Stretches of FLAT_DAYS trading days over which the underlying ended within +-FLAT_BAND: what a 2x daily fund
    with no costs at all did (pure reset drag), what the fund with costs did, and what the rule did there."""
    u, f, pos = series["u"], series["f"], series["pos"]
    n = len(u)
    if n < FLAT_DAYS * 3:
        return None

    def window(values: np.ndarray) -> np.ndarray:
        total = np.concatenate([[0.0], np.cumsum(np.log1p(np.clip(values, -0.999999, None)))])
        return np.exp(total[FLAT_DAYS:] - total[:-FLAT_DAYS]) - 1

    asset = window(u.values)
    flat = np.abs(asset) <= FLAT_BAND
    if flat.sum() < 20:
        return None
    pure = window((2 * u).clip(lower=-1.0).values)
    cost = window(f.values)
    rule2 = window(series["rule2"].values)
    rule = window(series["rule"].values)
    held = np.convolve(pos.values, np.ones(FLAT_DAYS), "valid") / FLAT_DAYS
    spread = np.sqrt(np.convolve(u.values ** 2, np.ones(FLAT_DAYS), "valid") / FLAT_DAYS * YEAR) * 100
    inside, outside = u[pos == 1], u[pos != 1]
    return {"n": int(flat.sum()), "all": int(len(asset)), "asset": float(asset[flat].mean() * 100),
            "pure": float(pure[flat].mean() * 100), "cost": float(cost[flat].mean() * 100),
            "rule2": float(rule2[flat].mean() * 100), "rule": float(rule[flat].mean() * 100),
            "rule2_worse": float((rule2[flat] < cost[flat]).mean() * 100),
            "held_flat": float(held[flat].mean() * 100), "held_all": float(pos.mean() * 100),
            "vol_flat": float(spread[flat].mean()), "vol_other": float(spread[~flat].mean()),
            "vol_in": (float(inside.std() * math.sqrt(YEAR) * 100) if len(inside) > 20 else None),
            "vol_out": (float(outside.std() * math.sqrt(YEAR) * 100) if len(outside) > 20 else None)}


def _scale(change: pd.Series, signal: pd.Series, since: str | None, cost2: float) -> list[dict]:
    """Synthetic funds at several leverage levels, held always and held under the rule (underlying's signal)."""
    frame = pd.DataFrame({"u": change, "s": signal.shift(1)}).dropna()
    if since:
        frame = frame[frame.index >= since]
    first, rows = frame.index[0], []
    for level in LEVELS:
        daily = frame.u if level == 1 else _fund(frame.u, level, _cost_at(level, cost2))
        net, position = _held(daily, signal, first)
        rows.append({"level": level, "cost": (0.0 if level == 1 else _cost_at(level, cost2) * 100),
                     "hold": _measure(daily), "rule": _measure(net, position),
                     "worst_day": float(daily.min() * 100)})
    return rows


def _compute(data: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    close, funds = data["close"], data["funds"]
    change = {key: series.pct_change() for key, series in close.items()}
    signal = {key: _rule(series) for key, series in close.items()}

    # 1. the first cost guess against the real funds
    tracking = {}
    for fund, key in REAL_FUNDS:
        if fund in funds and key in close:
            found = _tracking(funds[fund], change[key], FUND_SINCE.get(fund))
            if found is not None:
                tracking[fund] = dict(found, key=key)
    cost2, adjusted = {}, {}
    for key in close:
        implied = [t["implied"] / 100 for f, t in tracking.items() if t["key"] == key]
        guess = float(np.mean(implied)) if implied else None
        adjusted[key] = guess is not None and abs(guess - BASE_COST_YEAR) > ADJUST_IF_OFF
        cost2[key] = guess if adjusted[key] else BASE_COST_YEAR

    # 2. four ways per underlying (the index uses the real fund SSO; the others a synthetic 2x fund)
    four, four_base, flat = {}, {}, {}
    for key in close:
        since = MSTR_SINCE if key == "MSTR" else None
        if key == "SPY":
            if "SSO" not in funds:
                continue
            fund_change, own = funds["SSO"].pct_change(), _rule(funds["SSO"])
        else:
            fund_change = _fund(change[key], 2.0, cost2[key])
            own = _rule((1 + fund_change.fillna(0)).cumprod())
        result = _four(change[key], fund_change, signal[key], own, since, keep=True)
        if result is None:
            continue
        four[key] = result
        flat[key] = _flat(result["series"])
        if key != "SPY" and adjusted[key]:  # the same thing with the first cost guess, to show how much it matters
            base = _fund(change[key], 2.0, BASE_COST_YEAR)
            four_base[key] = _four(change[key], base, signal[key], _rule((1 + base.fillna(0)).cumprod()), since)

    # 4. real funds on their actual prices
    real = {}
    for fund, key in REAL_TESTS:
        if fund not in funds or key not in close:
            continue
        price = funds[fund]
        if key == "BTC":  # bitcoin as it was known at the US close: the candle of the calendar day before
            day_signal = _rule(data["daily"][key]).reindex(price.index - pd.Timedelta(days=1))
            day_signal.index = price.index
            cross = day_signal
        else:
            cross = signal[key]
        fund_change = price.pct_change()
        frame = pd.DataFrame({"u": change[key], "f": fund_change, "s": cross.shift(1)}).dropna()
        if len(frame) < 120:
            continue
        first = frame.index[0]
        asset_net, asset_pos = _held(frame.u, signal[key], first)
        fund_net, fund_pos = _held(frame.f, cross, first)
        item = {"key": key, "first": first, "days": len(frame), "hold": _measure(frame.u), "hold2": _measure(frame.f),
                "rule": _measure(asset_net, asset_pos), "rule2": _measure(fund_net, fund_pos), "own": None,
                "series": {"pos": fund_pos, "net": fund_net}}
        own_frame = pd.DataFrame({"f": fund_change, "o": _rule(price).shift(1)}).dropna()
        if len(own_frame) >= 120:
            own_net, own_pos = _held(own_frame.f, _rule(price), own_frame.index[0])
            item["own"] = {"first": own_frame.index[0], "rule": _measure(own_net, own_pos),
                           "hold": _measure(own_frame.f)}
        real[fund] = item

    # 5. leverage levels
    scale = {key: _scale(change[key], signal[key], MSTR_SINCE if key == "MSTR" else None, cost2[key])
             for key in SCALE_KEYS if key in four}
    scale_base = {key: _scale(change[key], signal[key], MSTR_SINCE if key == "MSTR" else None, BASE_COST_YEAR)
                  for key in SCALE_KEYS if key in four}

    # 6. today
    today = []
    if "MSTR" in close:
        today.append(("MSTR", close["MSTR"]))
    for fund in ("MSTU", "BITX"):
        if fund in funds and len(funds[fund]) >= R_IN:
            today.append((fund, funds[fund]))
    if "BTC" in data["daily"]:
        today.append(("比特幣（每天都算的日線，儀表板原本的規則）", data["daily"]["BTC"]))
        today.append(("比特幣（只看美股交易日，這份統計的算法）", close["BTC"]))
    state = []
    for name, series in today:
        now = float(_rule(series).iloc[-1])
        state.append({"name": name, "close": float(series.iloc[-1]), "date": series.index[-1], "state": now,
                      "flip": _flip(series, now)})
    return {"tracking": tracking, "cost2": cost2, "adjusted": adjusted, "four": four, "four_base": four_base,
            "flat": flat, "real": real, "scale": scale, "scale_base": scale_base, "state": state,
            "worst_day": {key: (float(x["series"]["u"].min() * 100), x["series"]["u"].idxmin())
                          for key, x in four.items()}}


# ---------- the study ----------

def study() -> dict:
    data, problems = _load()
    if not data["close"]:
        raise RuntimeError("所有資產都取不到資料")
    r = _compute(data)
    four, tracking, cost2 = r["four"], r["tracking"], r["cost2"]
    if not four:
        raise RuntimeError("沒有任何資產有足夠的資料")
    keys = [key for key, *_ in UNDERLYINGS if key in four]
    asof = max(series.index[-1] for series in data["close"].values())

    def pair(a: dict, b: dict, field: str) -> str:
        if field == "ratio":
            return f"{_num(a[field])}／{_num(b[field])}"
        if field == "under":
            return f"{_span(a[field])}／{_span(b[field])}"
        if field == "worst":
            return f"{_pct(a['worst'])}（{a['worst_year']}）／{_pct(b['worst'])}（{b['worst_year']}）"
        if field == "share":
            return f"{a[field]:.0f}%／{b[field]:.0f}%"
        return f"{_pct(a[field], signed=field != 'vol')}／{_pct(b[field], signed=field != 'vol')}"

    fields = ("ret", "vol", "ratio", "deepest", "under", "worst", "share")

    # ----- table 1: four ways per underlying -----
    rows1 = []
    for key in keys:
        x = four[key]
        what = "兩倍基金用真的 SSO" if key == "SPY" else f"模擬基金成本每年 {cost2[key] * 100:.0f}%"
        rows1.append([f"{LABEL[key]}（{_d(x['first'])[:7]} 起；{what}）｜一直抱著：原資產／兩倍基金"]
                     + [pair(x["hold"], x["hold2"], f) for f in fields])
        rows1.append([f"{LABEL[key]}｜照規則：原資產／兩倍基金（看原資產的訊號）"]
                     + [pair(x["rule"], x["rule2"], f) for f in fields])
        rows1.append([f"{LABEL[key]}｜照規則買兩倍基金：看基金自己的價格／只放一半的錢"]
                     + [pair(x["own2"], x["half"], f) for f in fields])
    n = len(keys)

    def count(test) -> int:
        return sum(1 for key in keys if test(four[key]))

    beat_hold2 = count(lambda x: x["rule2"]["growth"] > x["hold2"]["growth"])
    shallower = count(lambda x: x["rule2"]["deepest"] > x["hold2"]["deepest"])
    beat_rule_money = count(lambda x: x["rule2"]["growth"] > x["rule"]["growth"])
    beat_rule_ratio = count(lambda x: x["rule2"]["ratio"] > x["rule"]["ratio"])
    beat_hold_money = count(lambda x: x["rule2"]["growth"] > x["hold"]["growth"])
    own_better = count(lambda x: x["own2"]["growth"] > x["rule2"]["growth"])
    deeper_than_rule = count(lambda x: x["rule2"]["deepest"] < x["rule"]["deepest"])
    not_beat = [LABEL[key] for key in keys if not four[key]["rule2"]["growth"] > four[key]["hold2"]["growth"]]
    half_money = count(lambda x: x["half"]["growth"] > x["rule"]["growth"])
    half_deeper = count(lambda x: x["half"]["deepest"] < x["rule"]["deepest"])
    half_ratio = count(lambda x: x["half"]["ratio"] > x["rule"]["ratio"])
    hold2_lost = count(lambda x: x["hold2"]["growth"] < x["hold"]["growth"])
    caption1 = (
        "每個資產三列，每一格是「左／右」兩個數字，左右各是什麼寫在第一欄。規則：收盤創 "
        f"{R_IN} 日新高就持有、創 {R_OUT} 日新低就換成現金，沒有為任何資產調過。同一個資產的六種做法用的是同一段日子。"
        f"{n} 個資產裡：照規則持有兩倍基金，最後的錢比一直抱著兩倍基金多的有 {beat_hold2} 個、最深虧損比較淺的有 {shallower} 個；"
        f"和照規則持有原資產比，最後的錢比較多的有 {beat_rule_money} 個、報酬÷波動比較高的有 {beat_rule_ratio} 個、"
        f"最深虧損比較深的有 {deeper_than_rule} 個；"
        f"比一直抱著原資產賺得多的有 {beat_hold_money} 個。訊號改看基金自己的價格，最後的錢比看原資產訊號多的有 {own_better} 個。"
        f"一直抱著兩倍基金，最後的錢比一直抱著原資產還少的有 {hold2_lost} 個。"
        f"只放一半的錢照規則買兩倍基金，和全部的錢照規則買原資產比：最後的錢比較多的有 {half_money} 個、"
        f"報酬÷波動比較高的有 {half_ratio} 個、最深虧損比較深的有 {half_deeper} 個。"
        "一年用 252 個交易日換算；比特幣和以太幣也只取美股交易日那幾天的收盤。")
    columns1 = ["資產｜做法（每格：左／右）", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最久多久沒回到高點",
                "最差的一年", "持有天數比例"]
    tables = [{"caption": caption1, "columns": columns1, "rows": rows1[:16]}]

    # ----- table 2: real funds -----
    rows2 = []

    def plain(label: str, since, m: dict, last: str) -> list:
        return [label, _d(since), _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['share']:.0f}%", last]

    for fund, _ in REAL_FUNDS:
        if fund not in tracking:
            continue
        t = tracking[fund]
        recent = f"，最近一年 {t['recent']:.0f}%" if _ok(t["recent"]) else ""
        frame = data["funds"][fund].pct_change().dropna()
        frame = frame[frame.index >= t["first"]]
        rows2.append(plain(f"{fund}（{LEAD[t['key']]}的兩倍）一直抱著", t["first"], _measure(frame),
                           f"模擬每年 {_pct(t['sim'])}，真的每年 {_pct(t['real'])}，差 {t['gap']:+.1f} 個百分點；"
                           f"反推成本每年 {t['implied']:.0f}%{recent}"))
    for fund, key in REAL_TESTS:
        if fund not in r["real"]:
            continue
        x = r["real"][fund]
        note = "看比特幣前一天的日線" if key == "BTC" else f"看 {key} 的收盤"
        rows2.append(plain(f"{LEAD[key]}一直抱著（和 {fund} 同一段日子）", x["first"], x["hold"], "—"))
        rows2.append(plain(f"{LEAD[key]}照規則", x["first"], x["rule"], "—"))
        rows2.append(plain(f"{fund} 照規則（{note}）", x["first"], x["rule2"], "—"))
    own_bits = []
    for fund, _ in REAL_TESTS:
        x = r["real"].get(fund)
        if x and x["own"]:
            o = x["own"]
            own_bits.append(f"{fund} 改看它自己的價格（{_d(o['first'])} 起）是每年 {_pct(o['rule']['ret'])}、最深虧 "
                            f"{_pct(o['rule']['deepest'])}，同一段日子抱著是每年 {_pct(o['hold']['ret'])}、最深虧 "
                            f"{_pct(o['hold']['deepest'])}")
    off = [f for f in tracking if abs(tracking[f]["implied"] / 100 - BASE_COST_YEAR) > ADJUST_IF_OFF and f != "SSO"]
    caption2 = (
        f"上半（「一直抱著」那 {len(tracking)} 列）：真的兩倍基金從各自有資料的第一天起的實際成績，最後一欄拿它和「兩倍 × 原資產每天漲跌 − 每年 "
        f"{BASE_COST_YEAR * 100:.2f}% 成本」的模擬比；「反推成本」是要扣掉多少成本，模擬才會和真的基金一樣。"
        f"不算 SSO 的 {len([f for f in tracking if f != 'SSO'])} 檔裡有 {len(off)} 檔的反推成本和原本假設差超過 2 個百分點，"
        "所以表一的模擬基金改用反推出來的成本。"
        "下半：MSTU 和 BITX 用真實價格做同樣的比較（兩檔基金自己「一直抱著」的數字就是上半那一列），各自只有一兩年資料，樣本非常小，只能當參考。"
        + ("；".join(own_bits) + "。" if own_bits else "")
        + "CONL 早期是 1.5 倍，這裡只從 " + CONL_2X_SINCE + " 起算。")
    tables.append({"caption": caption2,
                   "columns": ["基金或做法", "起算日", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "持有天數比例",
                               "和模擬比（只有基金那幾列有）"],
                   "rows": rows2[:16]})

    # ----- table 3: leverage levels -----
    rows3, limits = [], {}
    for key in SCALE_KEYS:
        if key not in r["scale"]:
            continue
        levels = r["scale"][key]
        plain_rule = levels[0]["rule"]
        for item in levels:
            h, m = item["hold"], item["rule"]
            name = "原資產" if item["level"] == 1 else f"{_lev(item['level'])} 倍（成本每年 {item['cost']:.0f}%）"
            rows3.append([f"{LABEL[key]}｜{name}", _pct(h["ret"]), _pct(h["deepest"]), _pct(item["worst_day"]),
                          _pct(m["ret"]), _num(m["ratio"]), _pct(m["deepest"]),
                          "—" if item["level"] == 1 else f"{m['growth'] / plain_rule['growth']:.2f} 倍"])
        best = max(levels, key=lambda item: item["rule"]["growth"])
        below = next((item["level"] for item in levels[1:] if item["rule"]["growth"] < plain_rule["growth"]), None)
        loss = next((item["level"] for item in levels[1:] if item["rule"]["ret"] < 0), None)
        ratios = [item["rule"]["ratio"] for item in levels]
        limits[key] = {"best": best["level"], "best_ret": best["rule"]["ret"], "below": below, "loss": loss,
                       "falling": all(b < a for a, b in zip(ratios, ratios[1:]))}
    if rows3:
        tables.append({
            "caption": ("把槓桿倍數換掉會怎樣（模擬基金，訊號一律看原資產的收盤，所以每一列進出場的日子都一樣，差別只在持有的東西）。"
                        f"成本假設：每年 {FEE_YEAR * 100:.2f}% 管理費，加上借來的部分的利息；利息照表二反推出來的兩倍基金成本按借的比例放大。"
                        "「最慘的一天」是這個倍數在這段期間單日跌最多的一次，跌到 −100% 代表基金當天歸零，之後就沒有了。"
                        "最後一欄是照規則持有它最後的錢，是照規則持有原資產的幾倍，小於 1 就是輸給不加槓桿。4 倍和 5 倍市面上沒有這種基金，只是用來看趨勢。"
                        + f"{len(limits)} 個資產裡 {sum(1 for v in limits.values() if v['falling'])} 個是倍數每加一級，照規則的報酬÷波動就低一級。"),
            "columns": ["資產｜倍數", "一直抱著：每年報酬", "一直抱著：最深虧損", "最慘的一天", "照規則：每年報酬", "照規則：報酬÷波動",
                        "照規則：最深虧損", "照規則的錢是原資產照規則的幾倍"],
            "rows": rows3[:16]})

    # ----- table 4: today -----
    def state_text(v) -> str:
        return "持有" if v == 1 else "現金"

    rows4 = []
    for item in r["state"]:
        gap = (item["flip"] / item["close"] - 1) * 100
        flip = (f"收盤 ≤ {_price(item['flip'])} 就轉現金" if item["state"] == 1 else f"收盤 ≥ {_price(item['flip'])} 就轉持有")
        rows4.append([item["name"], f"{_price(item['close'])}（{_d(item['date'])[5:]}）", state_text(item["state"]), flip,
                      _pct(gap)])
    tables.append({
        "caption": (f"規則現在的狀態，用最後一根收完的日線算。翻轉價：現在是現金的話，下一個收盤要大於等於前 {R_IN - 1} 個收盤裡最高的那個；"
                    f"現在是持有的話，下一個收盤要小於等於前 {R_OUT - 1} 個收盤裡最低的那個，這個價位每天都會變。價格單位是美元。"
                    "這只是規則的狀態，不是買賣建議。"),
        "columns": ["標的", "最後收盤（日期）", "規則狀態", "會讓它翻轉的收盤價", "離翻轉價還有多遠"],
        "rows": rows4[:16]})

    now = [{"label": f"{item['name']}：規則狀態（{_d(item['date'])} 收盤 {_price(item['close'])} 美元）",
            "value": state_text(item["state"]) + "；"
            + (f"收盤 ≤ {_price(item['flip'])} 轉現金" if item["state"] == 1 else f"收盤 ≥ {_price(item['flip'])} 轉持有")}
           for item in r["state"]]

    # ----- verdict -----
    def money(x: dict, a: str, b: str) -> str:
        return "多" if x[a]["growth"] > x[b]["growth"] else "少"

    lead = [key for key in ("BTC", "MSTR") if key in four]
    bits = []
    for key in lead:
        x = four[key]
        bits.append(f"{LABEL[key]}（{_d(x['first'])[:7]} 起，模擬）一直抱著兩倍基金是每年 {_pct(x['hold2']['ret'])}、最深虧 "
                    f"{_pct(x['hold2']['deepest'])}，照規則持有是每年 {_pct(x['rule2']['ret'])}、最深虧 {_pct(x['rule2']['deepest'])}，"
                    f"照規則持有原資產是每年 {_pct(x['rule']['ret'])}、最深虧 {_pct(x['rule']['deepest'])}")
    verdict = (f"兩倍槓桿基金照規則持有，{n} 個資產裡 {beat_hold2} 個最後的錢比一直抱著它多"
               + (f"（例外是{'、'.join(not_beat)}）" if not_beat else "") + f"、{shallower} 個最深虧損比較淺；"
               f"和照同一條規則持有原本的資產比，{n} 個裡 {beat_rule_money} 個最後的錢比較多，但 {beat_rule_ratio} 個報酬÷波動比較高、"
               f"{deeper_than_rule} 個最深虧損比較深" + ("：" + "；".join(bits) if bits else "") + "。")
    verdict += (f"只放一半的錢照規則買兩倍基金，並不等於全部的錢照規則買原資產：{n} 個裡 {half_money} 個最後的錢比較多、"
                f"{half_ratio} 個報酬÷波動比較高、{half_deeper} 個最深虧損比較深")
    if lead:
        x = four[lead[0]]
        verdict += (f"（{LABEL[lead[0]]}：每年 {_pct(x['half']['ret'])} 對 {_pct(x['rule']['ret'])}，最深虧 {_pct(x['half']['deepest'])} 對 "
                    f"{_pct(x['rule']['deepest'])}）")
    verdict += "。"
    scale_bits = []
    for key in SCALE_KEYS:
        if key not in limits:
            continue
        lim = limits[key]
        if lim["below"] is None:
            text = f"{LEAD[key]}到 {_lev(LEVELS[-1])} 倍照規則的錢都還比不加槓桿多"
        else:
            text = f"{LEAD[key]}到 {_lev(lim['below'])} 倍照規則的錢就輸給不加槓桿"
            if lim["loss"] is not None:
                text += f"、到 {_lev(lim['loss'])} 倍變成賠錢"
        if lim["best"] > 1:
            text += f"（{_lev(lim['best'])} 倍時最高，每年 {_pct(lim['best_ret'])}）"
        scale_bits.append(text)
    if scale_bits:
        verdict += "槓桿倍數（模擬）：" + "；".join(scale_bits) + "。"

    # ----- notes -----
    notes = [
        "兩倍槓桿基金：每天的漲跌幅設計成原資產當天漲跌幅的兩倍的基金（例如 MSTU、MSTX 對 MSTR，BITX、BITU 對比特幣，ETHU 對以太幣，CONL 對 COIN，SSO 對美股大盤）。"
        "每日重設造成的耗損：因為每天都重新調回兩倍，價格來回震盪時會「漲的時候加碼、跌的時候減碼」，原資產繞一圈回到原價，基金卻會比原來少，震盪越大少越多。",
        f"規則只有一條，而且沒有為這份統計調過：收盤是最近 {R_IN} 個收盤裡最高的就持有，是最近 {R_OUT} 個收盤裡最低的就換成現金，其他日子維持原狀。"
        "用當天收盤已經知道的價格決定下一個交易日的部位，並假設剛好能用收盤價成交。換一條規則結果可能不同。",
    ]
    cost_bits = []
    for fund, _ in REAL_FUNDS:
        if fund in tracking:
            t = tracking[fund]
            cost_bits.append(f"{fund} {t['implied']:.0f}%" + (f"（最近一年 {t['recent']:.0f}%）" if _ok(t["recent"]) else ""))
    used = "、".join(f"{LABEL[key]} {cost2[key] * 100:.0f}%" for key in keys if key != "SPY")
    notes.append(
        f"模擬基金怎麼做的：每天報酬 = 倍數 × 原資產當天報酬 − 每天成本。一開始假設兩倍基金的成本是每年 {BASE_COST_YEAR * 100:.2f}%"
        f"（管理費約 {FEE_YEAR * 100:.2f}% 加上借來那一半的利息約 5%，和儀表板原本的假設相同）。拿它和真的基金在共同的日子比，這個假設明顯太樂觀："
        f"要讓模擬和真的基金一樣，每年得扣的成本是 {'、'.join(cost_bits)}。只有老牌的大盤基金 SSO 接近原本的假設。"
        f"所以表一和表三改用反推出來的成本（同一個資產有兩檔基金就取平均）：{used}。"
        "這些反推成本只來自一兩年的資料，而且看得出來一直在變（多數基金最近一年比剛上市時低），拿來套在更早的年代只是粗估；"
        "裡面除了費用和利息，也混著期貨轉倉、追蹤誤差等所有讓基金落後的東西。")
    base_bits = []
    for key in keys:
        if key in r["four_base"] and r["four_base"][key]:
            b, x = r["four_base"][key], four[key]
            base_bits.append(f"{LEAD[key]}照規則持有兩倍基金會是每年 {_pct(b['rule2']['ret'])}（表一是 {_pct(x['rule2']['ret'])}），"
                             f"最後的錢比照規則持有原資產{money(b, 'rule2', 'rule')}")
    if base_bits:
        notes.append(f"成本假設影響很大。如果用原本每年 {BASE_COST_YEAR * 100:.2f}% 的成本重算：" + "；".join(base_bits)
                     + "。也就是說，兩倍基金照規則持有贏不贏得過原資產，主要看基金實際上每年被吃掉多少，這個數字這裡只能粗估。")
    flat_bits = []
    for key in keys:
        f = r["flat"].get(key)
        if not f:
            continue
        flat_bits.append(
            f"{LABEL[key]}：{f['all']:,} 段裡有 {f['n']:,} 段，原資產平均 {_pct(f['asset'])}，完全不扣費用的兩倍平均 {_pct(f['pure'])}、"
            f"扣掉成本的兩倍基金平均 {_pct(f['cost'])}，照規則持有兩倍基金平均 {_pct(f['rule2'])}、照規則持有原資產平均 {_pct(f['rule'])}，"
            f"規則有 {f['held_flat']:.0f}% 的日子是持有的（全部日子是 {f['held_all']:.0f}%）")
    if flat_bits:
        used_flat = [r["flat"][key] for key in keys if r["flat"].get(key)]
        calm = [f for f in used_flat if _ok(f["vol_in"]) and _ok(f["vol_out"])]
        vol_bits = [f"{LEAD[key]}持有的日子 {_pct(r['flat'][key]['vol_in'], signed=False)}、現金的日子 "
                    f"{_pct(r['flat'][key]['vol_out'], signed=False)}"
                    for key in keys if r["flat"].get(key) and _ok(r["flat"][key]["vol_in"]) and _ok(r["flat"][key]["vol_out"])]
        notes.append(
            f"把耗損攤開來看：找出所有「過了 {FLAT_DAYS} 個交易日，原資產漲跌在正負 {FLAT_BAND * 100:.0f}% 以內」的期間（每天都當一次起點，所以彼此大量重疊，不是獨立的樣本）。"
            + "；".join(flat_bits) + "。規則有沒有避開最震盪的日子？原資產每天漲跌的起伏（換算成一年）："
            + "；".join(vol_bits) + "。整理起來：在這些原地打轉的期間，"
            f"{len(used_flat)} 個資產裡 {sum(1 for f in used_flat if f['rule2'] > f['cost'])} 個是照規則持有兩倍基金的平均結果比一直抱著它好，"
            f"{sum(1 for f in used_flat if f['rule2'] < f['rule'])} 個是照規則持有兩倍基金比照規則持有原資產差，"
            f"{sum(1 for f in used_flat if f['held_flat'] < f['held_all'] - 5)} 個是規則在這些期間持有的日子明顯比平常少（少 5 個百分點以上）；"
            f"{len(calm)} 個裡 {sum(1 for f in calm if f['vol_in'] < f['vol_out'] * 0.9)} 個是持有的日子起伏明顯比現金的日子小（小一成以上）。"
            "也就是說，規則在震盪期的結果和一直抱著不同，主要是因為有一部分日子是現金，不是因為它挑得出哪些日子比較平靜；它持有的那些日子照樣吃到耗損和來回進出的成本。")
    wd = r["worst_day"]
    worst_bits = [f"{LEAD[key]}最慘的一天是 {_d(wd[key][1])} 的 {_pct(wd[key][0])}，兩倍就是 {_pct(max(2 * wd[key][0], -100.0))}"
                  for key in ("BTC", "MSTR") if key in wd]
    notes += [
        "兩倍基金可能在原資產非常慘的一天裡幾乎全部賠光：原資產一天跌 50%，兩倍基金當天就歸零，之後漲回來也沒有用。"
        + ("這段資料裡，" + "；".join(worst_bits) + "。" if worst_bits else "")
        + "規則只在收盤後才反應，遇到這種一天之內的暴跌，如果當時是持有狀態，完全躲不掉。",
        "模擬基金只是近似。真的基金有追蹤誤差、成本會隨利率和市場變動、可能被清算下市，也可能反向分割（把很多股併成一股）；這些模擬都沒有算。"
        "真的兩倍基金（MSTU、MSTX、BITX、BITU、ETHU、CONL）只有一兩年的歷史，表二下半的樣本非常小，換一段時間結果可能完全不同。",
        "股票和基金只有交易日才有價格，隔夜和週末的跳空規則來不及反應，只能等下一次收盤。比特幣和以太幣用幣安現貨日線，但只取美股交易日那幾天的收盤"
        "（該交易日那根日線，UTC 0 點收，比美股收盤晚三到四小時），因為基金只在交易日交易和重設；所以這裡比特幣的數字和儀表板其他地方每天都算的版本不一樣，"
        "週末的漲跌會併進週一。真的 BITX 那一列改用美股收盤時已經收完的比特幣日線（前一個日曆天那根）當訊號，沒有偷看。",
        "「只放一半的錢」的算法：每次規則說進場時，把當時全部資金的一半買進兩倍基金，另一半留現金，中間不調整，到規則說出場時全部賣掉。"
        "因為基金每天重設，漲勢中它的部位會自己越滾越大，所以它不等於一倍的原資產；如果改成每天都調回一半，結果是："
        + "；".join(f"{LEAD[key]}每年 {_pct(four[key]['half_daily']['ret'])}、最深虧 {_pct(four[key]['half_daily']['deepest'])}"
                   for key in keys) + "（這種做法每天都要買賣，這裡沒有算那些買賣的成本）。",
        f"成本都是粗估：每次買賣扣成交金額的 {COST * 100:.2f}%；一直抱著不扣買賣成本；持有現金假設沒有利息；沒有算稅。"
        "MSTR 從 2020-08-11（公司宣布第一次買比特幣那天）算起；COIN 從 2021 年上市後算起；SPY 和 SSO 用 Yahoo 最近十年的資料。各資產期間不同，彼此的數字不能直接互比。"
        "「最差的一年」包含頭尾不完整的年份。「最久多久沒回到高點」算到資料最後一天為止，還沒回去的也算。",
        "這些資產高度連動（MSTR、COIN、以太幣都跟著比特幣走），所以「幾個裡幾個」不是互相獨立的測試。",
        "過去的結果不保證未來會一樣。這裡只是整理每種做法過去做了什麼、規則現在的狀態，不是買賣建議。",
    ] + problems

    sources = [{"name": "幣安現貨日線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/markets/overview"}]
    shown = [s for s in ("MSTR", "COIN", "SPY") if s in data["close"]] + [f for f, _ in REAL_FUNDS if f in data["funds"]]
    sources += [{"name": f"Yahoo Finance {s} 每日還原收盤價", "url": f"https://finance.yahoo.com/quote/{s}/history"}
                for s in shown]

    return {
        "id": "leveraged_trend",
        "title": "順勢規則用在兩倍槓桿基金上還有用嗎",
        "question": "如果持有的是每天重設的兩倍槓桿基金，「創新高持有、創新低出場」的規則還幫得上忙嗎？照規則持有兩倍基金，比照規則持有原本的資產好還是差？",
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
