"""Historical study "basis_filter": does a very high futures basis (how much dearer the quarterly future is than
spot, converted to a year) warn of crashes, and does cutting exposure when it is high add anything to the 17-rule
trend blend of trend_ensemble - with settings fixed in advance and checked on later years?

Background: Schmeling, Schrimpf, Todorov, "Crypto carry" (BIS Working Paper 1087; Management Science) find that a high
crypto carry (futures basis) predicts later crashes and liquidations. The paper gives no trading rule; the settings
below are fixed here, before looking at results, and every one of them is reported.

Basis: Deribit quarterly futures (last Friday of March/June/September/December, 08:00 UTC), last trade before each
UTC midnight taken from hourly candles (Deribit's daily candles run 08:00-08:00 UTC and would be 8 hours late),
against the Bitstamp USD spot daily close (UTC 00:00). annualised basis = (future / spot - 1) x 365 /
days to expiry, days counted from the end of the day to the expiry. Roll rule: the nearest quarterly contract with
more than 14 days left; a day on which that contract did not trade has no value (no jumping to another contract).
Cross-checks: Binance COIN-M quarterly futures (continuous CURRENT/NEXT_QUARTER against Binance's own index) and the
CME front month on Yahoo (BTC=F).

No look-ahead: the basis and the blend are both known at the close of day t and decide the position held during day
t+1. Percentile thresholds use only the 365 days before day t. The yearly choice uses only results before the year.
Costs are charged on the amount actually traded (trend_ensemble._run). Only public, key-less endpoints through
common.get_json; expired contracts are kept in cache/_store/basis_filter_contracts.json because they never change.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.studies2 import trend_ensemble as te

DERIBIT = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
DAPI = "https://dapi.binance.com/dapi/v1/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/BTC=F"
STORE_FILE = common.STORE / "basis_filter_contracts.json"  # {instrument: [[day, midnight price], ...]}
COST = te.COST  # 0.10%
HIGH_COST = 0.003
YEAR = 365
MIN_LEFT = 14  # roll: nearest quarterly with more than 14 days left
WINDOW = 120  # days before expiry from which hourly prices are fetched (a quarter is at most ~105 days in use)
TRAIL = 365
TRAIL_MIN = 300  # a trailing percentile needs at least this many known days in the 365 before
FILL = 3  # a missing basis day may repeat the last value for at most 3 days
HORIZONS = (7, 14, 30)
DROP = 0.10
JOIN = 30  # days of a group closer than this are one episode (the longest horizon looked at)
MOSTLY_OUT = 0.3  # the blend holds 30% or less: "already mostly out"
FIRST_YEAR = 2017
BANDS = ((None, 5), (5, 10), (10, 20), (20, 30), (30, None))
QUINT = ("最低的 1/5", "偏低的 1/5", "中間的 1/5", "偏高的 1/5", "最高的 1/5")
# filter settings fixed in advance: (key, kind, level, k)
FILTERS = [(f"{kind}{lvl}_{int(k * 10)}", kind, lvl, k) for kind, lvl in
           (("p", 80), ("p", 90), ("p", 95), ("a", 20), ("a", 30)) for k in (0.0, 0.5)]
CANDIDATES = ["none"] + [f[0] for f in FILTERS]
PERIODS = (("first", None), ("2021", "2021-01-01"), ("2023", "2023-01-01"), ("2024", "2024-01-01"))
COINS = (("BTC", "比特幣", "btcusd", "BTCUSDT"), ("ETH", "以太幣", "ethusd", "ETHUSDT"))
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


# ---------- formatting ----------

_ok, _pct, _num, _d = te._ok, te._pct, te._num, te._d


def _times(v) -> str:
    return f"100 次裡 {float(v):.0f} 次" if _ok(v) else "—"


def _name(key: str) -> str:
    if key == "none":
        return "不篩（單純混合）"
    kind, lvl, k = next((f[1], f[2], f[3]) for f in FILTERS if f[0] == key)
    when = f"溢價高過前 365 天的 {lvl}%" if kind == "p" else f"溢價 ≥ 每年 {lvl}%"
    return f"{when}→{'全部出場' if k == 0 else '減半'}"


def _clean(obj):
    """None → "—", non-finite numbers → "—", recursively."""
    if obj is None:
        return "—"
    if isinstance(obj, float) and not math.isfinite(obj):
        return "—"
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


# ---------- data ----------

def _last_friday(year: int, month: int) -> date:
    end = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1))
    return end - timedelta(days=(end.weekday() - 4) % 7)


def _quarterlies(coin: str, until: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    out = []
    for y in range(FIRST_YEAR, until.year + 2):
        for m in (3, 6, 9, 12):
            day = _last_friday(y, m)
            out.append((f"{coin}-{day.day}{MONTHS[m - 1]}{str(y)[2:]}", pd.Timestamp(day) + pd.Timedelta(hours=8)))
    return out


def _read_store() -> dict:
    try:
        data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - missing or damaged store: fetch again
        return {}


def _write_store(data: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        STORE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _deribit(coin: str, store: dict) -> dict[str, tuple[pd.Timestamp, pd.Series]]:
    """{instrument: (expiry, price at each UTC midnight, labelled with the day that just ended)}.

    Deribit's daily candles run from 08:00 to 08:00 UTC, so they would be 8 hours later than the spot close. Hourly
    candles are used instead: day D's value is the close of the last traded hourly candle among 21:00-23:00 UTC of D
    (the 23:00 candle closes at 00:00 of D+1). Only the window in which the contract can be the one in use (120 to 13
    days before expiry) is fetched. Expired contracts are stored."""
    today = pd.Timestamp(common.today_utc())
    out, changed = {}, False
    for name, expiry in _quarterlies(coin, today):
        a = expiry - pd.Timedelta(days=WINDOW)
        if a > today:
            continue
        rows = store.get(name)
        if rows is None:
            try:
                r = common.get_json(DERIBIT, {"instrument_name": name, "resolution": "60",
                                              "start_timestamp": int(a.value // 10 ** 6),
                                              "end_timestamp": int((expiry - pd.Timedelta(days=MIN_LEFT - 1)).value
                                                                   // 10 ** 6)})["result"]
            except Exception:  # noqa: BLE001 - a contract that never existed answers with an error
                r = {}
            rows = []
            if r.get("status") == "ok":
                last_of_day: dict[str, float] = {}
                for t, c, v in zip(r["ticks"], r["close"], r["volume"]):
                    start = pd.Timestamp(t, unit="ms")
                    end = start + pd.Timedelta(hours=1)
                    if start.hour >= 21 and end <= today and v and v > 0 and c and c > 0:
                        last_of_day[start.strftime("%Y-%m-%d")] = float(c)  # later hours overwrite earlier ones
                rows = [[d, c] for d, c in sorted(last_of_day.items())]
            if expiry < today - pd.Timedelta(days=2):
                store[name], changed = rows, True
        if rows:
            s = pd.Series([c for _, c in rows], index=pd.to_datetime([d for d, _ in rows]))
            out[name] = (expiry, s[~s.index.duplicated()].sort_index())
    if changed:
        _write_store(store)
    if len(out) < 4:
        raise RuntimeError(f"Deribit {coin} 季度期貨只拿到 {len(out)} 個合約")
    return out


def _bitstamp(pair: str) -> pd.DataFrame:
    """Bitstamp USD spot daily close and low since 2017, finished UTC days only."""
    rows, start = [], int(pd.Timestamp("2017-01-01").timestamp())
    for _ in range(12):
        page = common.get_json(BITSTAMP.format(pair=pair), {"step": 86400, "limit": 1000, "start": start})["data"]["ohlc"]
        if not page:
            break
        rows += page
        nxt = int(page[-1]["timestamp"]) + 86400
        # a page can be short when it starts before the pair was listed, so stop only at the present
        if nxt <= start or nxt >= pd.Timestamp(common.today_utc()).timestamp():
            break
        start = nxt
    frame = pd.DataFrame({"close": [float(r["close"]) for r in rows], "low": [float(r["low"]) for r in rows]},
                         index=pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s"))
    frame = frame[~frame.index.duplicated()].sort_index()
    frame = frame[(frame.index < pd.Timestamp(common.today_utc())) & (frame["close"] > 0)]
    if len(frame) < 1000:
        raise RuntimeError(f"Bitstamp {pair} 日線只有 {len(frame)} 筆")
    return frame.asfreq("D").ffill()


def _basis(contracts: dict, spot: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily annualised basis in % and the days to expiry used (nearest quarterly with > 14 days left)."""
    expiries = sorted((exp, name) for name, (exp, _) in contracts.items())
    days = spot.index
    value, left = pd.Series(np.nan, index=days), pd.Series(np.nan, index=days)
    by_exp = {exp: contracts[name][1] for exp, name in expiries}
    # the full quarterly calendar, so a contract that is missing is not silently replaced by a later one
    calendar = sorted({exp for _, exp in _quarterlies("X", days[-1])})
    for day in days:
        end = day + pd.Timedelta(days=1)
        nxt = next((e for e in calendar if (e - end).total_seconds() / 86400 > MIN_LEFT), None)
        if nxt is None or nxt not in by_exp:
            continue
        price = by_exp[nxt].get(day)
        s = spot.get(day)
        if price is None or not _ok(price) or not _ok(s) or s <= 0:
            continue
        dte = (nxt - end).total_seconds() / 86400
        value[day] = (price / s - 1) * YEAR / dte * 100
        left[day] = dte
    return value.dropna(), left.dropna()


def _binance_basis() -> pd.Series | None:
    """Binance COIN-M BTCUSD quarterly basis against Binance's index, same roll rule, in % a year."""
    def pages(path: str, params: dict) -> pd.Series:
        rows, start = [], 0
        for _ in range(6):
            page = common.get_json(DAPI + path, dict(params, interval="1d", limit=1500, startTime=start))
            if not page:
                break
            rows += page
            start = page[-1][0] + 86_400_000
            if len(page) < 1500:
                break
        s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms"))
        return s[~s.index.duplicated()].sort_index()

    index = pages("indexPriceKlines", {"pair": "BTCUSD"})
    cur = pages("continuousKlines", {"pair": "BTCUSD", "contractType": "CURRENT_QUARTER"})
    nxt = pages("continuousKlines", {"pair": "BTCUSD", "contractType": "NEXT_QUARTER"})
    calendar = sorted({exp for _, exp in _quarterlies("X", pd.Timestamp(common.today_utc()))})
    out = {}
    for day in cur.index:
        if day >= pd.Timestamp(common.today_utc()) or day not in index.index:
            continue
        end = day + pd.Timedelta(days=1)
        later = [e for e in calendar if e > end]
        if len(later) < 2:
            continue
        first_left = (later[0] - end).total_seconds() / 86400
        if first_left > MIN_LEFT:
            price, dte = cur.get(day), first_left
        else:
            price, dte = nxt.get(day), (later[1] - end).total_seconds() / 86400
        if _ok(price) and price > 0:
            out[day] = (price / index[day] - 1) * YEAR / dte * 100
    s = pd.Series(out).sort_index()
    return s if len(s) > 300 else None


def _cme_basis(spot: pd.Series) -> pd.Series | None:
    """CME front month (Yahoo BTC=F) against Bitstamp, days with at least 10 days to the month's last Friday."""
    r = common.get_json(YAHOO, {"range": "10y", "interval": "1d"})["chart"]["result"][0]
    closes = r["indicators"]["quote"][0]["close"]
    out = {}
    for t, c in zip(r["timestamp"], closes):
        if not _ok(c):
            continue
        day = pd.Timestamp(t, unit="s").normalize()
        expiry = pd.Timestamp(_last_friday(day.year, day.month)) + pd.Timedelta(hours=15)
        dte = (expiry - (day + pd.Timedelta(hours=21))).total_seconds() / 86400
        s = spot.get(day)
        if dte >= 10 and _ok(s):
            out[day] = (c / s - 1) * YEAR / dte * 100
    s = pd.Series(out).sort_index()
    return s if len(s) > 200 else None


def _binance_close(symbol: str) -> pd.Series:
    rows, start = [], int(pd.Timestamp("2017-08-01").timestamp() * 1000)
    for _ in range(8):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + px.DAY_MS
        if len(page) < 1000:
            break
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms"))
    s = s[~s.index.duplicated()].sort_index()
    s = s[(s.index < pd.Timestamp(common.today_utc())) & (s > 0)]
    if len(s) < 1500:
        raise RuntimeError(f"{symbol} 日線只有 {len(s)} 筆")
    return s.asfreq("D").ffill()


# ---------- calculations ----------

def _trailing_pct(basis: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Rank of each day's basis among the 365 days before it (0-1), and the trailing 80/90/95th percentile lines."""
    full = basis.reindex(index)
    vals = full.values
    rank = np.full(len(vals), np.nan)
    lines = {q: np.full(len(vals), np.nan) for q in (80, 90, 95)}
    for i in range(len(vals)):
        window = vals[max(0, i - TRAIL):i]
        window = window[np.isfinite(window)]
        if len(window) < TRAIL_MIN:
            continue
        for q in lines:
            lines[q][i] = np.percentile(window, q)
        if np.isfinite(vals[i]):
            rank[i] = float((window < vals[i]).mean())
    return pd.Series(rank, index=index), {q: pd.Series(v, index=index) for q, v in lines.items()}


def _episodes(mask: pd.Series) -> list[pd.Timestamp]:
    """Start days of runs of True days, runs closer than JOIN days joined into one."""
    days = mask.index[mask.values.astype(bool)]
    starts, prev = [], None
    for d in days:
        if prev is None or (d - prev).days > JOIN:
            starts.append(d)
        prev = d
    return starts


def _forward(spot: pd.DataFrame) -> pd.DataFrame:
    close, low = spot["close"], spot["low"]
    out = pd.DataFrame(index=spot.index)
    for h in HORIZONS:
        out[f"r{h}"] = (close.shift(-h) / close - 1) * 100
        ahead = low[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        out[f"w{h}"] = (ahead / close - 1) * 100
    return out


def _group_stats(mask: pd.Series, fwd: pd.DataFrame) -> dict:
    mask = mask.reindex(fwd.index, fill_value=False).astype(bool)
    part = fwd[mask]
    starts = _episodes(mask)
    hit = sum(1 for s in starts if _ok(fwd["w30"].get(s)) and fwd["w30"][s] <= -DROP * 100)
    known = sum(1 for s in starts if _ok(fwd["w30"].get(s)))

    def mean(col):
        v = part[col].dropna()
        return float(v.mean()) if len(v) else None

    def share(col):
        v = part[col].dropna()
        return float((v <= -DROP * 100).mean() * 100) if len(v) else None

    return {"days": int(mask.sum()), "eps": len(starts), "eps_hit": hit, "eps_known": known,
            "r7": mean("r7"), "r14": mean("r14"), "r30": mean("r30"), "f14": share("w14"), "f30": share("w30"),
            "w30": mean("w30")}


def _worst30(net: pd.Series) -> float | None:
    if len(net) < 30:
        return None
    growth = np.log1p(net.values)
    roll = pd.Series(growth).rolling(30).sum()
    return float((math.exp(roll.min()) - 1) * 100)


def _factor(key: str, basis: pd.Series, lines: dict, index) -> pd.Series:
    """Multiplier decided at the close of day t (1 = no cut). A missing basis repeats the last value for 3 days."""
    if key == "none":
        return pd.Series(1.0, index=index)
    kind, lvl, k = next((f[1], f[2], f[3]) for f in FILTERS if f[0] == key)
    b = basis.reindex(index).ffill(limit=FILL)
    line = lines[lvl] if kind == "p" else pd.Series(float(lvl), index=index)
    high = (b > line) if kind == "p" else (b >= line)
    high = high & b.notna() & line.notna()
    return pd.Series(np.where(high, k, 1.0), index=index)


def _compute(close: pd.Series, basis: pd.Series, spot: pd.DataFrame) -> dict:
    """Everything that depends on the data for one coin. Kept separate so look-ahead can be tested."""
    index = close.index
    target, _, _, _ = te._targets(close)
    blend = target["blend"]
    change = close.pct_change()
    basis = basis[(basis.index >= index[0]) & (basis.index <= index[-1])]
    rank, lines = _trailing_pct(basis, index)
    first_basis = basis.index[0]
    ranked = lines[90].notna() & blend.notna()
    start = ranked[ranked].index[0]  # every setting is defined from here on
    last = index[-1]

    factors = {key: _factor(key, basis, lines, index) for key in CANDIDATES}

    def run(key: str, delay: int = 1, cost: float = COST):
        want = blend * factors[key]
        pos = want.shift(delay)
        net, trade = te._run(pos.values, change.values, cost)
        return pd.Series(net, index=index), pos.fillna(0.0), pd.Series(trade, index=index)

    def measure(parts, a, b=None) -> dict:
        rows = (index >= pd.Timestamp(a)) & (index <= (pd.Timestamp(b) if b else last))
        m = te._measure(parts[0][rows], parts[1][rows], parts[2][rows])
        m["worst30"] = _worst30(parts[0][rows])
        return m

    runs = {k: run(k) for k in CANDIDATES}
    hold = (change.fillna(0.0), pd.Series(1.0, index=index), pd.Series(0.0, index=index))
    per = {}
    for name, a in PERIODS:
        a = start if a is None else max(start, pd.Timestamp(a))
        per[name] = {k: measure(runs[k], a) for k in CANDIDATES}
        per[name]["hold"] = measure(hold, a)
    cut_share = {k: float((factors[k][index >= start] < 1).mean() * 100) for k in CANDIDATES}

    # ----- pick in advance each year -----
    first_full = first_basis.year + (0 if (first_basis.month, first_basis.day) == (1, 1) else 1)
    test_years = [y for y in range(first_full + 1, last.year + 1)
                  if (pd.Timestamp(f"{y}-01-01") - start).days >= 180]
    picks, chain_net, chain_pos, chain_trade = [], [], [], []
    for year in test_years:
        a, b = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), last)
        before = (index >= start) & (index < a)
        inside = (index >= a) & (index <= b)
        prior = {k: te._score_parts(runs[k][0][before].values)[2] for k in CANDIDATES}
        during = {k: te._score_parts(runs[k][0][inside].values)[2] for k in CANDIDATES}
        best = sorted(CANDIDATES, key=lambda k: (-round(prior[k], 9), CANDIDATES.index(k)))[0]
        rho = None
        pa, pb = pd.Series(prior).rank(), pd.Series(during).rank()
        if pa.std() > 0 and pb.std() > 0 and inside.sum() >= 30:
            rho = float(np.corrcoef(pa.values, pb.values)[0, 1])
        ret_pick = float((np.prod(1 + runs[best][0][inside].values) - 1) * 100)
        ret_none = float((np.prod(1 + runs["none"][0][inside].values) - 1) * 100)
        picks.append({"year": year, "best": best, "ret": ret_pick, "ret_none": ret_none, "rho": rho,
                      "partial": b < pd.Timestamp(f"{year}-12-31"), "ratio": during[best], "ratio_none": during["none"]})
        chain_net.append(runs[best][0][inside])
        chain_pos.append(runs[best][1][inside])
        chain_trade.append(runs[best][2][inside])
    chained = none_chain = None
    if picks:
        net = pd.concat(chain_net)
        chained = te._measure(net, pd.concat(chain_pos), pd.concat(chain_trade))
        chained["worst30"] = _worst30(net)
        a = pd.Timestamp(f"{test_years[0]}-01-01")
        none_chain = measure(runs["none"], a)
        chained["switch_extra"] = 0.0

    # ----- predictive check -----
    fwd = _forward(spot).reindex(index)
    known = basis.reindex(index)
    pred, pred_late = {}, {}
    late = index >= pd.Timestamp("2023-01-01")
    quint = np.minimum(np.floor(rank * 5), 4)
    for g in range(5):
        pred[f"q{g}"] = _group_stats(quint == g, fwd)
        pred_late[f"q{g}"] = _group_stats((quint == g) & late, fwd)
    for lo, hi in BANDS:
        m = known.notna()
        if lo is not None:
            m &= known >= lo
        if hi is not None:
            m &= known < hi
        pred[f"b{lo}"] = _group_stats(m, fwd)
        pred_late[f"b{lo}"] = _group_stats(m & late, fwd)
    pred["all"] = _group_stats(known.notna(), fwd)
    pred["all_ranked"] = _group_stats(rank.notna(), fwd)
    pred_late["all"] = _group_stats(known.notna() & late, fwd)

    # ----- take apart -----
    robust = {
        "cost": {k: measure(run(k, cost=HIGH_COST), start) for k in CANDIDATES},
        "delay": {k: measure(run(k, delay=2), start) for k in CANDIDATES},
    }
    overlap = {}
    nxt = change.shift(-1)  # what bitcoin did during the day the filtered position would have been held
    for key in CANDIDATES[1:]:
        out_days = (factors[key] < 1) & (index >= start)
        b = blend[out_days]
        mostly_out = int((b <= MOSTLY_OUT).sum())
        disagree = out_days & (blend > 0.5)
        moves = nxt[disagree].dropna()
        overlap[key] = {"days": int(out_days.sum()), "mostly_out": mostly_out, "disagree": int(disagree.sum()),
                        "disagree_eps": len(_episodes(disagree)),
                        "mean": float(moves.mean() * 100) if len(moves) else None,
                        "total": float((np.prod(1 + moves.values) - 1) * 100) if len(moves) else None,
                        "up": float((moves > 0).mean() * 100) if len(moves) else None}

    today = {"basis": float(known.dropna().iloc[-1]) if known.notna().any() else None,
             "basis_day": known.dropna().index[-1] if known.notna().any() else None,
             "rank": float(rank.dropna().iloc[-1]) if rank.notna().any() else None,
             "rank_day": rank.dropna().index[-1] if rank.notna().any() else None,
             "lines": {q: float(v.iloc[-1]) if _ok(v.iloc[-1]) else None for q, v in lines.items()},
             "factors": {k: float(factors[k].iloc[-1]) for k in CANDIDATES},
             "blend": float(blend.iloc[-1]) if _ok(blend.iloc[-1]) else None}
    return {"per": per, "cut_share": cut_share, "picks": picks, "chained": chained, "none_chain": none_chain,
            "test_years": test_years, "pred": pred, "pred_late": pred_late, "robust": robust, "overlap": overlap,
            "today": today, "start": start, "first_basis": first_basis, "last": last, "factors": factors,
            "blend": blend, "rank": rank, "lines": lines, "runs": runs}


# ---------- the study ----------

def _load(coin: str, pair: str, symbol: str, store: dict) -> dict:
    spot = _bitstamp(pair)
    contracts = _deribit(coin, store)
    basis, left = _basis(contracts, spot["close"])
    close = te._daily_close() if coin == "BTC" else _binance_close(symbol)
    return {"spot": spot, "basis": basis, "left": left, "close": close, "n_contracts": len(contracts)}


def study() -> dict:
    store = _read_store()
    notes_extra: list[str] = []
    data, results = {}, {}
    for coin, label, pair, symbol in COINS:
        try:
            data[coin] = _load(coin, pair, symbol, store)
            results[coin] = _compute(data[coin]["close"], data[coin]["basis"], data[coin]["spot"])
        except Exception as exc:  # noqa: BLE001
            if coin == "BTC":
                raise
            notes_extra.append(f"{label}的資料這次取不到（{type(exc).__name__}），以太幣的部分留空，沒有改用其他方法繞過。")
    btc = results["BTC"]
    eth = results.get("ETH")

    cross = {}
    try:
        bn = _binance_basis()
        if bn is not None:
            both = pd.concat([data["BTC"]["basis"], bn], axis=1, join="inner").dropna()
            cross["binance"] = {"first": bn.index[0], "n": len(both), "corr": float(both.corr().iloc[0, 1]),
                                "gap": float((both.iloc[:, 1] - both.iloc[:, 0]).median()),
                                "last": float(bn.iloc[-1]), "last_day": bn.index[-1]}
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"幣安幣本位季度期貨這次取不到（{type(exc).__name__}），沒有拿來對照。")
    try:
        cme = _cme_basis(data["BTC"]["spot"]["close"])
        if cme is not None:
            a = data["BTC"]["basis"].resample("MS").mean()
            b = cme.resample("MS").mean()
            both = pd.concat([a, b], axis=1, join="inner").dropna()
            cross["cme"] = {"first": cme.index[0], "months": len(both), "corr": float(both.corr().iloc[0, 1]),
                            "gap": float((both.iloc[:, 1] - both.iloc[:, 0]).median())}
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"Yahoo 上的芝商所比特幣期貨這次取不到（{type(exc).__name__}），沒有拿來對照。")

    last = btc["last"]
    start = btc["start"]

    # ----- table 1: predictive -----
    def prow(name: str, s: dict, base: bool = False) -> list:
        eps = (f"{s['days']:,} 天" if base else
               f"{s['days']:,} 天｜{s['eps']} 段（{s['eps_hit']} 段之後 30 天內跌 10%+）")
        return [name, eps, _pct(s["r7"]), _pct(s["r14"]), _pct(s["r30"]), _times(s["f14"]), _times(s["f30"]),
                _pct(s["w30"])]

    def band_name(lo, hi) -> str:
        if lo is None:
            return f"每年 {hi}% 以下"
        if hi is None:
            return f"每年 {lo}% 以上"
        return f"每年 {lo}–{hi}%"

    pred, late = btc["pred"], btc["pred_late"]
    t1 = [prow(f"比特幣｜和前 365 天比：{QUINT[g]}", pred[f"q{g}"]) for g in range(5)]
    t1 += [prow(f"比特幣｜溢價{band_name(lo, hi)}", pred[f"b{lo}"]) for lo, hi in BANDS]
    t1.append(prow("比特幣｜全部有溢價的日子（對照）", pred["all"], True))
    t1.append(prow("比特幣 2023 年起｜和前 365 天比：最高的 1/5", late["q4"]))
    t1.append(prow("比特幣 2023 年起｜溢價每年 20% 以上", _merge_bands(data["BTC"], btc, 20, late=True)))
    t1.append(prow("比特幣 2023 年起｜全部（對照）", late["all"], True))
    if eth:
        t1.append(prow("以太幣｜和前 365 天比：最高的 1/5", eth["pred"]["q4"]))
        t1.append(prow("以太幣｜全部有溢價的日子（對照）", eth["pred"]["all"], True))

    # ----- table 2: filter on the blend -----
    per = btc["per"]

    def trio(m: dict) -> str:
        return f"{_pct(m['ret'])}｜{_num(m['ratio'])}｜{_pct(m['deepest'])}"

    t2 = []
    for key in ["hold"] + CANDIDATES:
        name = "一直抱著不動（對照）" if key == "hold" else _name(key)
        m = per["first"][key]
        t2.append([name, _pct(m["ret"]), _num(m["ratio"]), f"{_pct(m['deepest'])}｜{_pct(m['worst30'])}",
                   trio(per["2021"][key]), trio(per["2023"][key]), trio(per["2024"][key]),
                   "—" if key == "hold" else f"{btc['cut_share'][key]:.0f}%"])

    # ----- table 3: pick in advance -----
    t3 = []
    for coin, r in (("比特幣", btc), ("以太幣", eth)):
        if not r:
            continue
        for p in r["picks"]:
            yname = f"{p['year']}（到 {_d(r['last'])[5:]}）" if p["partial"] else str(p["year"])
            t3.append([f"{coin} {yname}", _name(p["best"]), _pct(p["ret"]), _pct(p["ret_none"]), _num(p["ratio"]),
                       _num(p["ratio_none"]), _num(p["rho"])])
        if r["chained"]:
            c, n = r["chained"], r["none_chain"]
            t3.append([f"{coin} 串起來（{r['test_years'][0]} 起）", "每年事先挑的",
                       f"每年 {_pct(c['ret'])}｜最深 {_pct(c['deepest'])}", f"每年 {_pct(n['ret'])}｜最深 {_pct(n['deepest'])}",
                       _num(c["ratio"]), _num(n["ratio"]),
                       f"平均 {_num(np.mean([p['rho'] for p in r['picks'] if p['rho'] is not None]))}"
                       if any(p["rho"] is not None for p in r["picks"]) else "—"])

    # ----- table 4: take apart and ETH -----
    def spread(ms: dict, base: dict) -> list:
        keys = CANDIDATES[1:]
        ratios = [ms[k]["ratio"] for k in keys]
        deep = [ms[k]["deepest"] for k in keys]
        better = sum(1 for k in keys if ms[k]["ratio"] > base["ratio"] + 1e-9)
        return [f"{_num(min(ratios))} ~ {_num(max(ratios))}", f"{_pct(min(deep))} ~ {_pct(max(deep))}",
                f"10 種裡 {better} 種比較高"]

    def base_cells(m: dict) -> list:
        return [_pct(m["ret"]), _num(m["ratio"]), f"{_pct(m['deepest'])}｜{_pct(m['worst30'])}"]

    rb = btc["robust"]
    t4 = []
    t4.append(["比特幣，原本設定（成本 0.10%、隔天照做）"] + base_cells(per["first"]["none"]) + spread(per["first"], per["first"]["none"]))
    t4.append(["比特幣，成本 0.30%"] + base_cells(rb["cost"]["none"]) + spread(rb["cost"], rb["cost"]["none"]))
    t4.append(["比特幣，晚一天才照做"] + base_cells(rb["delay"]["none"]) + spread(rb["delay"], rb["delay"]["none"]))
    if eth:
        ep = eth["per"]
        t4.append([f"以太幣（{_d(eth['start'])[:7]} 起）"] + base_cells(ep["first"]["none"]) + spread(ep["first"], ep["first"]["none"]))
        t4.append(["以太幣 2023 年起"] + base_cells(ep["2023"]["none"]) + spread(ep["2023"], ep["2023"]["none"]))
    for key in ("p90_0", "a20_0", "a30_0"):
        o = btc["overlap"][key]
        if not o["days"]:
            t4.append([f"重疊：{_name(key)}", "—", "—", "—", "這段期間沒有篩掉任何一天", "—", "—"])
            continue
        t4.append([f"重疊：{_name(key)}", f"篩掉 {o['days']} 天", f"其中 {o['mostly_out']} 天混合本來就只持有 ≤30%",
                   f"混合過半持有的 {o['disagree']} 天（{o['disagree_eps']} 段）",
                   f"那些天比特幣隔天平均 {_pct(o['mean'], signed=True)}",
                   f"合計 {_pct(o['total'])}", f"上漲 {_times(o['up'])}"])

    span = f"{_d(start)} 到 {_d(last)}"
    tables = [
        {"caption": (f"期貨溢價很高之後，比特幣有沒有比較容易大跌（{_d(btc['first_basis'])} 到 {_d(last)}）。"
                     "溢價＝季度期貨比現貨貴多少，換算成一年。「和前 365 天比」是當天的溢價在它前 365 天裡排第幾，"
                     "只用當天以前的資料。報酬和跌幅用 Bitstamp 現貨；「之後 30 天內跌 10%+」看的是途中最低價。"
                     "相鄰的日子看的幾乎是同一段行情，所以也列出「段」：相隔 30 天以內的日子算同一段，跌幅從每段第一天的收盤算。"),
         "columns": ["哪些日子", "天數｜段數（之後 30 天內曾跌 10% 以上的段數）", "之後 7 天平均", "之後 14 天平均",
                     "之後 30 天平均", "14 天內曾跌 10% 以上", "30 天內曾跌 10% 以上", "30 天內平均最大跌幅"],
         "rows": t1},
        {"caption": (f"把溢價當成順勢混合的開關（比特幣，{span}，每種設定都能算的第一天起）：持有比例＝混合 ×（溢價不高時 1，"
                     "溢價高時 0 或 0.5）。當天收盤決定、隔天照做，每次買賣扣交易金額的 0.10%，現金沒有利息。"
                     "後三欄是「每年報酬｜報酬÷波動｜最深曾經虧多少」；最後一欄是被篩（打折）的天數比例。"),
         "columns": ["做法", "每年報酬", "報酬÷波動", "最深曾經虧多少｜最差的 30 天", "2021 年起", "2023 年起", "2024 年起",
                     "被篩的天數"],
         "rows": t2},
        {"caption": ("事先挑選：每年 1 月 1 日只用之前的結果，在「不篩」和 10 種篩法裡挑報酬÷波動最高的一種（同分選不篩），"
                     "用一整年再換。最後一欄是「之前的排名」和「當年的排名」的等級相關（1 = 完全一樣、0 = 沒關係、負的 = 反過來）。"),
         "columns": ["年度", "事先挑到的", "當年報酬（挑到的）", "當年報酬（單純混合）", "當年報酬÷波動（挑到的）",
                     "當年報酬÷波動（單純混合）", "之前排名和當年排名的相關"],
         "rows": t3},
        {"caption": ("拆開來看：成本加到 0.30%、晚一天才照做、換到以太幣（用它自己的溢價和它自己的混合），10 種篩法的範圍；"
                     "下面三列是篩掉的日子和混合本身有多少重疊、兩者意見不同的日子比特幣做了什麼。"),
         "columns": ["情況", "單純混合 每年報酬", "單純混合 報酬÷波動", "單純混合 最深虧｜最差 30 天", "10 種篩法 報酬÷波動",
                     "10 種篩法 最深虧", "比單純混合高的"],
         "rows": t4},
    ]

    # ----- today -----
    td = btc["today"]
    states = []
    for key in ("p80_0", "p90_0", "p95_0", "a20_0", "a30_0"):
        kind, lvl = key[0], int(key[1:3])
        on = td["factors"][key] < 1
        states.append(f"{'前 365 天的 ' + str(lvl) + '%' if kind == 'p' else '每年 ' + str(lvl) + '%'}："
                      f"{'溢價高過門檻' if on else '沒有高過門檻'}")
    now = [
        {"label": f"比特幣期貨溢價（Deribit 季度，{_d(td['basis_day']) if td['basis_day'] is not None else '—'}）",
         "value": f"每年 {td['basis']:.1f}%" if _ok(td["basis"]) else "—"},
        {"label": "在前 365 天裡的位置", "value": f"比前 365 天的 {td['rank'] * 100:.0f}% 的日子高" if _ok(td["rank"]) else "—"},
        {"label": "前 365 天的 80／90／95% 門檻",
         "value": "／".join(f"每年 {td['lines'][q]:.1f}%" if _ok(td["lines"][q]) else "—" for q in (80, 90, 95))},
        {"label": "各篩法的狀態（只是狀態，不是建議）", "value": "；".join(states)},
        {"label": "順勢混合現在看多的比例", "value": f"{td['blend'] * 100:.0f}%" if _ok(td["blend"]) else "—"},
    ]
    if "binance" in cross:
        now.append({"label": f"對照：幣安幣本位季度溢價（{_d(cross['binance']['last_day'])}）",
                    "value": f"每年 {cross['binance']['last']:.1f}%"})
    if eth:
        e = eth["today"]
        now.append({"label": "以太幣期貨溢價（Deribit 季度）",
                    "value": (f"每年 {e['basis']:.1f}%，比前 365 天的 {e['rank'] * 100:.0f}% 的日子高"
                              if _ok(e["basis"]) and _ok(e["rank"]) else "—")})

    # ----- verdict -----
    q4, allp = pred["q4"], pred["all"]
    b30 = _merge_bands(data["BTC"], btc, 20, late=False)
    f0 = per["first"]
    none = f0["none"]
    keys = CANDIDATES[1:]
    better = sum(1 for k in keys if f0[k]["ratio"] > none["ratio"] + 1e-9)
    shallower = sum(1 for k in keys if f0[k]["deepest"] > none["deepest"] + 1e-9)
    ratios = [f0[k]["ratio"] for k in keys]
    c, n = btc["chained"], btc["none_chain"]
    b30_late = _merge_bands(data["BTC"], btc, 20, late=True)

    def more(x, y) -> str:
        if not (_ok(x) and _ok(y)):
            return "無法比較"
        return "差不多" if abs(x - y) < 5 else ("比較常" if x > y else "比較少")

    rhos = [p["rho"] for p in btc["picks"] if p["rho"] is not None]
    s1 = (f"比特幣期貨溢價每年 20% 以上之後 30 天內曾跌 10% 以上是 {_times(b30['f30'])}（{b30['eps']} 段；"
          f"2023 年起只有 {b30_late['eps']} 段），全部日子是 {_times(allp['f30'])}；"
          f"改用前 365 天排名的最高 1/5 是 {_times(q4['f30'])}，和全部日子{more(q4['f30'], allp['f30'])}。")
    s2 = (f"加在順勢混合上，10 種篩法有 {better} 種報酬÷波動高過單純混合的 {_num(none['ratio'])}"
          f"（{_num(min(ratios))}~{_num(max(ratios))}），{shallower} 種最深虧損較淺")
    if c:
        s2 += (f"；每年事先挑一次串起來是 {_num(c['ratio'])} 對 {_num(n['ratio'])}"
               + (f"，但前後排名的相關平均只有 {_num(float(np.mean(rhos)))}" if rhos else ""))
    s2 += "。"
    s3 = ""
    if eth:
        ef = eth["per"]["first"]
        eb = sum(1 for k in keys if ef[k]["ratio"] > ef["none"]["ratio"] + 1e-9)
        e20 = _merge_bands(data["ETH"], eth, 20, late=False)
        s3 = (f"以太幣溢價每年 20% 以上之後曾跌 10% 以上是 {_times(e20['f30'])}（{e20['eps']} 段），全部日子 {_times(eth['pred']['all']['f30'])}，"
              f"10 種篩法只有 {eb} 種較好")
        if eth["chained"]:
            s3 += f"，事先挑的 {_num(eth['chained']['ratio'])} 對 {_num(eth['none_chain']['ratio'])}"
        s3 += "。"
    verdict = s1 + s2 + s3

    o90 = btc["overlap"]["p90_0"]
    hot = data["BTC"]["basis"] >= 20
    by_year = hot.groupby(hot.index.year).sum()
    by_year = by_year[by_year > 0].sort_values(ascending=False)
    hot_years = ("溢價每年 20% 以上的日子分布在 " + "、".join(f"{y} 年 {int(n)} 天" for y, n in by_year.sort_index().items())
                 if len(by_year) else "溢價從來沒有到每年 20%")
    notes = [
        "期貨溢價：每天用 Deribit 季度期貨（每年 3、6、9、12 月最後一個星期五 UTC 08:00 到期）在 UTC 0 點以前最後一筆成交價（從每小時的 K 線取，因為 Deribit 的日線是 UTC 8 點換日），"
        "和 Bitstamp 美元現貨當天收盤價（UTC 0 點）比，(期貨 ÷ 現貨 − 1) × 365 ÷ 離到期的天數。換約規則：用離到期還有 14 天以上、"
        "最近的那一張季度合約；那張合約當天沒有成交就當作沒有資料（最多沿用前一個值 3 天），不會跳去用別張合約。"
        f"比特幣的溢價從 {_d(btc['first_basis'])} 起有資料（{data['BTC']['n_contracts']} 張合約）"
        + (f"，以太幣從 {_d(eth['first_basis'])} 起（{data['ETH']['n_contracts']} 張）" if eth else "") + "。"
        "最後成交價可能比 0 點早幾分鐘到幾小時，離到期近的時候換算成一年會把小差距放大，所以日與日之間會有雜訊。",
        _cross_note(cross),
        f"獨立的例子很少：比特幣溢價在前 365 天裡最高 1/5 的日子只有 {q4['eps']} 段，溢價每年 20% 以上只有 {b30['eps']} 段，"
        f"{hot_years}。相鄰的日子往後看的是同一段行情，「100 次裡幾次」看起來精確，"
        "其實背後只是少數幾段。",
        f"篩法的門檻（前 365 天的 80／90／95%、每年 20／30%；打折成 0 或一半）是在看結果之前就定好的，10 種全部列出，"
        "沒有挑最好的；但會想到用溢價來篩，本身就是因為現在已經知道論文說高溢價之後容易大跌，這仍然帶有事後的成分。"
        "每年 20%、30% 這兩個絕對門檻也是看過 2021 年那種行情的人才會選的數字。",
        f"重疊：以「溢價高過前 365 天的 90% 就全部出場」為例，被篩掉的 {o90['days']} 天裡有 {o90['mostly_out']} 天混合本來就只持有"
        f" 30% 以下；混合過半持有、篩法卻要出場的有 {o90['disagree']} 天（{o90['disagree_eps']} 段），"
        f"那些天比特幣隔天平均 {_pct(o90['mean'])}、全部連起來 {_pct(o90['total'])}。溢價高的時候通常也是價格一路漲的時候，"
        "順勢混合那時候多半是滿的，兩者在這些時候意見相反。",
        f"成本只用一個粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%（第四張表有 0.30% 的結果），沒有算現金利息、稅、"
        "期貨本身的費用；這裡只拿期貨溢價當訊號，持有的仍是現貨。",
        "順勢混合是 trend_ensemble 的 17 條規則（8 種均線、4 種新高新低、5 種過去漲跌），比特幣用和那份研究一樣的長期日線，"
        "以太幣用幣安現貨 ETHUSDT 日線。每種設定都從「前 365 天的門檻算得出來」那天才開始比，所以起點比溢價資料晚約一年。",
        "BIS 論文（Schmeling、Schrimpf、Todorov）說溢價高預示之後比較容易崩跌和強制平倉，但沒有給交易規則；"
        "這次程式抓論文 PDF 時被網站的機器人檢查擋下，沒有繞過，所以只照論文摘要頁的定義（期貨價格減現貨價格）來做。",
        "過去的結果不保證未來會一樣。這裡只是整理過去的統計，不是買賣建議。",
    ] + notes_extra

    sources = [
        {"name": "Schmeling, Schrimpf, Todorov〈Crypto carry〉BIS Working Paper 1087（Management Science）",
         "url": "https://www.bis.org/publ/work1087.htm"},
        {"name": "Deribit 公開 API：季度期貨每小時 K 線（get_tradingview_chart_data）",
         "url": "https://docs.deribit.com/#public-get_tradingview_chart_data"},
        {"name": "Bitstamp 公開 API：BTC/USD、ETH/USD 日線", "url": "https://www.bitstamp.net/api/#ohlc_data"},
        {"name": "幣安幣本位交割合約日線與指數價格（對照）",
         "url": "https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/Continuous-Contract-Kline-Candlestick-Data"},
        {"name": "Yahoo Finance 芝商所比特幣期貨近月 BTC=F（對照）", "url": "https://finance.yahoo.com/quote/BTC=F/"},
        {"name": "blockchain.info 每日比特幣價格與幣安現貨日線（順勢混合用）",
         "url": "https://www.blockchain.com/explorer/charts/market-price"},
    ]

    result = {
        "id": "basis_filter",
        "title": "期貨溢價太高時，要不要少拿一點",
        "question": "期貨比現貨貴很多（換算成一年）的時候，之後比較容易大跌嗎？在順勢混合上加一個「溢價太高就減碼」的開關，"
                    "設定事先定好、拿後面的年份檢查，會多出什麼嗎？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }
    return _clean(result)


def _merge_bands(d: dict, r: dict, level: float, late: bool) -> dict:
    """Group statistics for days with basis >= level (% a year), optionally from 2023 on."""
    index = r["blend"].index
    known = d["basis"].reindex(index)
    mask = known >= level
    if late:
        mask &= index >= pd.Timestamp("2023-01-01")
    return _group_stats(mask.fillna(False), _forward(d["spot"]).reindex(index))


def _cross_note(cross: dict) -> str:
    parts = []
    if "binance" in cross:
        b = cross["binance"]
        parts.append(f"幣安幣本位季度合約（對它自己的指數價格，同樣的換約規則）從 {_d(b['first'])} 起有資料，"
                     f"和 Deribit 重疊的 {b['n']:,} 天相關係數 {b['corr']:.2f}，幣安減 Deribit 的中位數 {b['gap']:+.1f} 個百分點")
    if "cme" in cross:
        c = cross["cme"]
        parts.append(f"芝商所近月合約（Yahoo BTC=F，只用離到期 10 天以上的日子；它的收盤時間比現貨早兩三個小時）"
                     f"按月平均和 Deribit 比，{c['months']} 個月的相關係數 {c['corr']:.2f}，芝商所減 Deribit 的中位數 {c['gap']:+.1f} 個百分點")
    if not parts:
        return "對照來源這次都取不到，只用 Deribit。"
    return "對照：" + "；".join(parts) + "。主要結果只用 Deribit。"


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
