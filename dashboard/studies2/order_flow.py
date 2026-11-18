"""Historical study "order_flow": does the daily taker buy/sell imbalance (主動買賣力道), the cumulative volume delta
or plain volume say anything about bitcoin's next moves - alone, or as a filter on the trend rules that held up?

Binance spot daily klines carry, for every bar, the base and quote volume, the number of trades and the TAKER BUY
base/quote volume (the part of the volume where the buyer hit the ask). From these:
1. taker buy share = taker buy volume / volume, on 1, 7 and 30 days, ranked against its own earlier history;
2. cumulative volume delta (taker buys minus taker sells, summed day after day) - its 90-day change, whether the line
   is above its 50-day average, and "price up over 30 days while the delta fell over 30 days" divergences;
3. volume against its previous 30-day average, volume spikes with and without a 20-day closing high, spikes on a down
   day, on-balance volume above its 50-day average;
4. spot minus USDT-M futures 7-day taker buy share (bitcoin only, futures from 2019-09).
Every signal is decided at the daily close (UTC) and acts on the next day; ranks use earlier days only (at least
RANK_MIN of them). Tests: forward 1/7/30-day returns and the chance of a 10% fall by group, with the number of
non-overlapping 30-day windows; long-or-cash rules; filters on the 17-rule blend and on 55/20; picking in advance
every year from 2020; the same signals on ETH, SOL, BNB and XRP; 0.30% costs, one day late, and the 24 closing hours
for the filter that looked best (hourly taker volume rebuilt from Binance 1-hour klines).

Data are kept under cache/_store (order_flow_*.csv) and topped up on later runs. Only public, key-less endpoints
through the dashboard helpers (common.get_json, price._binance).
"""
from __future__ import annotations

import bisect
import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.collectors import signals as sg
from dashboard.studies2 import trend_ensemble as te

FAPI = "https://fapi.binance.com/fapi/v1/klines"
COST = 0.001
COST_HIGH = 0.003
YEAR = 365
MAIN_START = "2018-04-01"
FIRST_PICK_YEAR = 2020
RANK_MIN = 180  # earlier days a value must be ranked against
LOW, HIGH = 0.10, 0.90
SIMS = 200
SEED = 20180401
MAX_PAGES = 120
COINS = ("ETH", "SOL", "BNB", "XRP")
DAILY_COLS = ("open", "high", "low", "close", "volume", "quote", "trades", "tb", "tbq")
HOURLY_COLS = ("close", "volume", "quote", "tb")
ZERO_FEE = ("2022-07-08", "2023-03-22")  # Binance zero-fee trading on BTCUSDT spot (ended 2023-03-22)

ALONE = {
    "A1": "7 天主動買進比例在過去一半以上才持有",
    "A2": "累積買賣差高於 50 日平均才持有",
    "A3": "沒有「價漲、買賣差減」背離才持有",
    "A4": "OBV 高於 50 日平均才持有",
    "A5": "爆量又創 20 日新高後持有 30 天",
    "A6": "現貨比合約買得積極（過去一半以上）才持有",
}
FILTERS = {
    "F1": "7 天主動買進比例落在過去最低一成",
    "F2": "累積買賣差低於 50 日平均",
    "F3": "價格 30 天漲、累積買賣差 30 天減（背離）",
    "F4": "OBV 低於 50 日平均",
    "F5": "7 天內出現爆量下跌",
    "F6": "現貨減合約的主動買進比例落在過去最低一成",
}
STEADY_NAME = {"r_share1": "當天買進比例", "r_share7": "7 天買進比例", "r_share30": "30 天買進比例", "r_cvd90": "90 天買賣差",
               "r_volratio": "量比", "ext7": "買進比例極端", "div_bear": "背離", "div_bull": "反向背離",
               "spike_up": "爆量創新高", "spike_noup": "爆量沒創新高", "spike_down": "爆量收跌", "obv_above": "OBV 高於均線",
               "cvd_above": "買賣差高於均線", "r_futdiff": "現貨減合約"}
SHORT = {"F1": "買進比例最低一成", "F2": "買賣差低於均線", "F3": "背離", "F4": "OBV 低於均線", "F5": "爆量下跌",
         "F6": "現貨落後合約", "F0": "不加濾網"}
FAMILY = {"A1": "F1", "A2": "F2", "A3": "F3", "A4": "F4", "A5": "F5", "A6": "F6"}
CONT = (  # (rank column, plain name)
    ("r_share1", "主動買進比例（當天）"),
    ("r_share7", "主動買進比例（7 天）"),
    ("r_share30", "主動買進比例（30 天）＝累積買賣差 30 天變化"),
    ("r_cvd90", "累積買賣差 90 天變化"),
    ("r_volratio", "成交量 ÷ 前 30 天平均"),
    ("r_futdiff", "現貨減合約的主動買進比例（7 天）"),
)
EVENTS = (  # (0/1 column, comparison column or None = the days without the event, plain name "without｜with")
    ("div_bear", "conf_up", "價格 30 天漲：累積買賣差也增｜累積買賣差反而減（背離）"),
    ("div_bull", "conf_down", "價格 30 天跌：累積買賣差也減｜累積買賣差反而增（反向背離）"),
    ("spike_up", "quiet_up", "創 20 日收盤新高：沒爆量｜爆量（前一成）"),
    ("spike_noup", "quiet_noup", "沒創 20 日新高：沒爆量｜爆量"),
    ("spike_down", "quiet_down", "收跌的日子：沒爆量｜爆量"),
    ("obv_above", None, "OBV 低於｜高於 50 日平均"),
    ("cvd_above", None, "累積買賣差低於｜高於 50 日平均"),
)


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


def _pp(v) -> str:
    if not _ok(v):
        return "—"
    return "0.0" if abs(float(v)) < 0.05 else f"{float(v):+.1f}"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _dnum(v) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):+.2f}"
    return "0.00" if float(text) == 0 else text


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _of100(v) -> str:
    return "—" if not _ok(v) else f"100 次裡 {float(v):.0f} 次"


# ---------- data ----------

def _path(name: str):
    return common.STORE / f"order_flow_{name}.csv"


def _read(name: str, width: int) -> dict[int, list[float]]:
    rows: dict[int, list[float]] = {}
    try:
        for line in _path(name).read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != width + 1 or not parts[0].isdigit():
                continue
            values = [float(x) for x in parts[1:]]
            if all(math.isfinite(x) for x in values):
                rows[int(parts[0])] = values
    except Exception:  # noqa: BLE001 - no store yet or a damaged one: fetch again
        return {}
    return rows


def _write(name: str, rows: dict[int, list[float]], cols) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{ms}," + ",".join(repr(x) for x in v) for ms, v in sorted(rows.items()))
        _path(name).write_text("ms," + ",".join(cols) + "\n" + body + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run fetches again
        pass


def _fetch(kind: str, symbol: str, interval: str, rows: dict, cols) -> int:
    """Add closed klines from the last stored one on (in place). Spot through price._binance, futures directly."""
    pick = (1, 2, 3, 4, 5, 7, 8, 9, 10) if len(cols) == len(DAILY_COLS) else (4, 5, 7, 9)
    start = max(rows) if rows else 0
    limit = 1000 if kind == "spot" else 1500
    added = 0
    for _ in range(MAX_PAGES):
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start}
        page = px._binance("/api/v3/klines", params) if kind == "spot" else common.get_json(FAPI, params, timeout=20)
        if not page:
            break
        now_ms = int(time.time() * 1000)
        for r in page:
            if int(r[6]) < now_ms:  # closed bars only
                rows[int(r[0])] = [float(r[i]) for i in pick]
                added += 1
        if len(page) < limit:
            break
        start = int(page[-1][0]) + 1
    return added


def _klines(kind: str, symbol: str, interval: str) -> tuple[dict, str | None]:
    cols = DAILY_COLS if interval == "1d" else HOURLY_COLS
    name = f"{kind}_{interval}_{symbol}"
    rows = _read(name, len(cols))
    step = 86_400_000 if interval == "1d" else 3_600_000
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - step:
        try:
            if _fetch(kind, symbol, interval, rows, cols):
                _write(name, rows, cols)
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            market = "現貨" if kind == "spot" else "U 本位合約"
            problem = (f"幣安{market} {symbol} 的 {interval} K 線這次沒有更新成功（{type(exc).__name__}），"
                       + ("用的是之前存下來的資料，" if rows else "") + "沒有改用其他方法繞過。")
    return rows, problem


def _daily(kind: str, symbol: str) -> tuple[pd.DataFrame, str | None]:
    rows, problem = _klines(kind, symbol, "1d")
    if len(rows) < 400:
        raise RuntimeError(f"{symbol} 日線只有 {len(rows)} 筆")
    frame = pd.DataFrame([rows[k] for k in sorted(rows)], columns=DAILY_COLS,
                         index=pd.to_datetime(sorted(rows), unit="ms").normalize())
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[frame.index < pd.Timestamp(common.today_utc())]
    return frame.asfreq("D"), problem


# ---------- signals ----------

def _trail_rank(x: pd.Series, min_n: int = RANK_MIN) -> pd.Series:
    """Share of strictly earlier valid values below today's value (ties count half); NaN until min_n earlier values."""
    out = np.full(len(x), np.nan)
    seen: list[float] = []
    for i, v in enumerate(x.values):
        if not math.isfinite(v):
            continue
        n = len(seen)
        if n >= min_n:
            lo = bisect.bisect_left(seen, v)
            hi = bisect.bisect_right(seen, v)
            out[i] = (lo + 0.5 * (hi - lo)) / n
        bisect.insort(seen, v)
    return pd.Series(out, index=x.index)


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    return cond.astype(float).where(valid)


def _features(close: pd.Series, bars: pd.DataFrame, fut: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every flow / volume quantity at each close, using that close and earlier ones only."""
    idx = close.index
    b = bars.reindex(idx)
    vol = b["volume"].where(b["volume"] > 0)
    tb, quote = b["tb"].where(vol.notna()), b["quote"].where(vol.notna())
    f = pd.DataFrame(index=idx)

    def share(n: int) -> pd.Series:
        return tb.rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()

    f["share1"], f["share7"], f["share30"] = tb / vol, share(7), share(30)
    delta = 2 * tb - vol  # taker buys minus taker sells (base units)
    f["cvd90"] = delta.rolling(90, min_periods=90).sum() / vol.rolling(90, min_periods=90).sum()
    first = delta.first_valid_index()
    cvd = delta.fillna(0.0).cumsum()
    obv = (np.sign(close.diff()) * vol).fillna(0.0).cumsum()
    if first is not None:
        cvd[cvd.index < first] = np.nan
        obv[obv.index < first] = np.nan
    else:
        cvd[:], obv[:] = np.nan, np.nan
    cvd_ma, obv_ma = cvd.rolling(50, min_periods=50).mean(), obv.rolling(50, min_periods=50).mean()
    f["cvd_above"] = _flag(cvd > cvd_ma, cvd_ma.notna())
    f["obv_above"] = _flag(obv > obv_ma, obv_ma.notna())
    d30, p30 = cvd - cvd.shift(30), close / close.shift(30) - 1
    both = d30.notna() & p30.notna()
    f["div_bear"] = _flag((p30 > 0) & (d30 < 0), both)
    f["div_bull"] = _flag((p30 < 0) & (d30 > 0), both)
    f["conf_up"] = _flag((p30 > 0) & (d30 >= 0), both)  # the comparison group: price and delta both up
    f["conf_down"] = _flag((p30 < 0) & (d30 <= 0), both)
    f["volratio"] = quote / quote.shift(1).rolling(30, min_periods=30).mean()
    for col in ("share1", "share7", "share30", "cvd90", "volratio"):
        f[f"r_{col}"] = _trail_rank(f[col])
    ok = f["r_volratio"].notna()
    spike = f["r_volratio"] >= HIGH
    at_high = close >= close.rolling(20, min_periods=20).max()
    down = close < close.shift(1)
    f["spike_up"] = _flag(spike & at_high, ok)
    f["spike_noup"] = _flag(spike & ~at_high, ok)
    f["spike_down"] = _flag(spike & down, ok)
    f["quiet_up"] = _flag(~spike & at_high, ok)  # comparison groups: the same price move without the spike
    f["quiet_noup"] = _flag(~spike & ~at_high, ok)
    f["quiet_down"] = _flag(~spike & down, ok)
    if fut is not None:
        fb = fut.reindex(idx)
        fvol = fb["volume"].where(fb["volume"] > 0)
        fshare7 = fb["tb"].where(fvol.notna()).rolling(7, min_periods=7).sum() / fvol.rolling(7, min_periods=7).sum()
        f["futdiff"] = f["share7"] - fshare7
    else:
        f["futdiff"] = np.nan
    f["r_futdiff"] = _trail_rank(f["futdiff"])
    return f


def _flags(f: pd.DataFrame) -> pd.DataFrame:
    """Long-or-cash rules (A, NaN while not available) and filter pass flags (F, 1 = may hold; 1 while unknown)."""
    out = pd.DataFrame(index=f.index)
    r7, rf = f["r_share7"], f["r_futdiff"]
    out["A1"] = _flag(r7 >= 0.5, r7.notna())
    out["A2"] = f["cvd_above"]
    out["A3"] = 1 - f["div_bear"]
    out["A4"] = f["obv_above"]
    out["A5"] = f["spike_up"].rolling(30, min_periods=1).max().where(f["spike_up"].notna())
    out["A6"] = _flag(rf >= 0.5, rf.notna())
    out["F1"] = 1 - _flag(r7 < LOW, r7.notna())
    out["F2"] = f["cvd_above"]
    out["F3"] = 1 - f["div_bear"]
    out["F4"] = f["obv_above"]
    out["F5"] = 1 - f["spike_down"].rolling(7, min_periods=1).max().where(f["spike_down"].notna())
    out["F6"] = 1 - _flag(rf < LOW, rf.notna())
    for k in FILTERS:
        out[k] = out[k].fillna(1.0)
    return out


def _targets(close: pd.Series, flags: pd.DataFrame, keys) -> pd.DataFrame:
    sig, _ = te._signals(close)
    blend = sig.mean(axis=1).where(sig.notna().all(axis=1))
    c55 = sg._breakout(close, 55, 20)
    out = pd.DataFrame({"hold": 1.0, "bF0": blend, "cF0": c55}, index=close.index)
    for k in keys:
        out[f"b{k}"] = blend * flags[k]
        out[f"c{k}"] = c55 * flags[k]
    for k in ALONE:
        if FAMILY[k] in keys:
            out[k] = flags[k]
    return out


# ---------- measuring ----------

def _stats(net: np.ndarray, pos: np.ndarray | None = None) -> dict:
    n = len(net)
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100) if n > 1 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    return {"ret": float(ret), "vol": vol, "ratio": float(ret / vol) if vol > 1e-9 else 0.0, "deepest": deepest,
            "share": float(np.nanmean(pos) * 100) if pos is not None and len(pos) else None}


def _net(target: pd.Series, change: pd.Series, cost: float = COST, delay: int = 1) -> tuple[pd.Series, pd.Series]:
    pos = target.shift(delay)
    net, _ = te._run(pos.values, change.values, cost)
    return pd.Series(net, index=target.index), pos.fillna(0.0)


def _spearman(a, b) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 5:
        return None
    ra, rb = pd.Series(a[keep]).rank().values, pd.Series(b[keep]).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    v = float(np.corrcoef(ra, rb)[0, 1])
    return v if math.isfinite(v) else None


def _forward(close: pd.Series) -> pd.DataFrame:
    fwd = pd.DataFrame(index=close.index)
    for h in (1, 7, 30):
        fwd[f"f{h}"] = (close.shift(-h) / close - 1) * 100
    low = close[::-1].rolling(30, min_periods=30).min()[::-1].shift(-1)
    fwd["fall"] = ((low / close - 1) <= -0.10).astype(float).where(low.notna())
    return fwd


def _windows(days: pd.DatetimeIndex, gap: int = 30) -> int:
    """Non-overlapping 30-day windows: take a day, skip the next 30 days, take the next one, ..."""
    count, last = 0, None
    for d in days:
        if last is None or (d - last).days >= gap:
            count, last = count + 1, d
    return count


def _group(mask: pd.Series, fwd: pd.DataFrame) -> dict | None:
    m = mask.fillna(False).astype(bool) & fwd["f30"].notna()
    if m.sum() < 5:
        return None
    part = fwd[m]
    return {"f1": float(part["f1"].mean()), "f7": float(part["f7"].mean()), "f30": float(part["f30"].mean()),
            "fall": float(part["fall"].mean() * 100), "n": int(m.sum()), "win": _windows(part.index)}


def _predict(f: pd.DataFrame, fwd: pd.DataFrame, start) -> dict:
    """Low/high groups (or without/with for events), rank correlation and yearly consistency of each signal."""
    rows = fwd.index >= pd.Timestamp(start)
    fw, ff = fwd[rows], f[rows]
    out = {"all": _group(pd.Series(True, index=fw.index), fw)}
    specs = [(k, ff[k] < 0.2, ff[k] >= 0.8, ff[k]) for k, _ in CONT]
    specs.append(("ext7", ff["r_share7"] < LOW, ff["r_share7"] >= HIGH, None))
    specs += [(k, (ff[b] == 1) if b else (ff[k] == 0), ff[k] == 1, None) for k, b, _ in EVENTS]
    for key, low, high, rank in specs:
        lo, hi = _group(low, fw), _group(high, fw)
        if lo is None or hi is None:
            out[key] = None
            continue
        years, agree = 0, 0
        for y in sorted(set(fw.index.year)):
            yr = fw.index.year == y
            a, b = _group(low & yr, fw), _group(high & yr, fw)
            if a and b and a["n"] >= 10 and b["n"] >= 10:
                years += 1
                agree += (b["f30"] > a["f30"]) == (hi["f30"] > lo["f30"])
        out[key] = {"low": lo, "high": hi, "rho": _spearman(rank, fw["f30"]) if rank is not None else None,
                    "years": years, "agree": agree, "dir": 1 if hi["f30"] > lo["f30"] else -1,
                    "first": _d(ff.index[(low | high).fillna(False).values.astype(bool)][0])}
    return out


def _measure_set(targets: pd.DataFrame, change: pd.Series, start, end=None, cost=COST, delay=1) -> dict:
    rows = (targets.index >= pd.Timestamp(start)) & (targets.index <= (pd.Timestamp(end) if end else targets.index[-1]))
    out = {}
    for k in targets.columns:
        net, pos = _net(targets[k], change, cost, delay)
        out[k] = _stats(net[rows].values, pos[rows].values)
    return out


def _luck(base: pd.Series, flag: pd.Series, change: pd.Series, start, real: float) -> float:
    """Circular shifts of the filter's pass days inside the measured period: how often does a shifted copy (same
    number and clustering of cash days, wrong dates) beat the real filter's return/volatility? In % of SIMS."""
    rows = base.index >= pd.Timestamp(start)
    b, p, c = base[rows].values, flag[rows].values, change[rows].values
    n = len(p)
    rng = np.random.default_rng(SEED)
    better = 0
    for s in rng.integers(60, n - 60, SIMS):
        pos = np.concatenate([[0.0], (b * np.roll(p, s))[:-1]])
        net, _ = te._run(pos, c, COST)
        better += _stats(net)["ratio"] >= real
    return better / SIMS * 100


def _asset(close: pd.Series, bars: pd.DataFrame, fut: pd.DataFrame | None) -> dict:
    """Everything for one coin. Kept free of fetching so look-ahead can be tested."""
    f = _features(close, bars, fut)
    flags = _flags(f)
    keys = tuple(FILTERS) if fut is not None else tuple(k for k in FILTERS if k != "F6")
    target = _targets(close, flags, keys)
    change = close.pct_change()
    ready = max(target["bF0"].first_valid_index(), f["r_volratio"].first_valid_index(),
                f["r_share7"].first_valid_index(), f["obv_above"].first_valid_index())
    start = max(pd.Timestamp(MAIN_START), ready)
    fut_start = f["r_futdiff"].first_valid_index() if fut is not None else None
    meas = _measure_set(target, change, start)
    if fut_start is not None:
        late = _measure_set(target[["hold", "bF0", "cF0", "bF6", "cF6", "A6"]], change, fut_start)
        meas.update({f"{k}@fut": v for k, v in late.items()})
    return {"f": f, "flags": flags, "keys": keys, "target": target, "change": change, "start": start,
            "fut_start": fut_start, "meas": meas, "pred": _predict(f, _forward(close), start), "close": close}


def _walk(a: dict) -> dict:
    """Pick in advance every year from FIRST_PICK_YEAR: the filter with the best return/volatility before the year."""
    target, change, start, last = a["target"], a["change"], a["start"], a["close"].index[-1]
    pos = target.shift(1)
    nets = {}
    for k in target.columns:
        nets[k] = pd.Series(te._run(pos[k].values, change.values, COST)[0], index=target.index)
    fut_first = a["fut_start"]
    flow_keys = [k for k in target.columns if k[0] == "A" or (k[0] in "bc" and k[1:] != "F0")]
    years = list(range(FIRST_PICK_YEAR, last.year + 1))

    def eligible(k: str, year: int) -> bool:
        if k.endswith("6"):
            return fut_first is not None and (pd.Timestamp(f"{year}-01-01") - fut_first).days >= YEAR
        return True

    def score(k: str, a0, b0) -> float:
        part = nets[k][(nets[k].index >= a0) & (nets[k].index < b0)].values
        return _stats(part)["ratio"] if len(part) > 30 else float("nan")

    picks, rho = {}, {}
    for y in years:
        y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        for pool in ("b", "c"):
            cand = [k for k in target.columns if k[0] == pool and k[1] == "F" and eligible(k, y)]
            picks[(pool, y)] = max(cand, key=lambda k: score(k, start, y0))
        el = [k for k in flow_keys if eligible(k, y)]
        rho[y] = _spearman([score(k, start, y0) for k in el], [score(k, y0, y1) for k in el])

    def chain(choice: dict) -> dict:
        parts = []
        for y in years:
            rows = (target.index >= pd.Timestamp(f"{y}-01-01")) & (target.index < pd.Timestamp(f"{y + 1}-01-01"))
            parts.append(pos.loc[rows, choice[y]])
        p = pd.concat(parts)
        net, _ = te._run(p.values, change.reindex(p.index).values, COST)
        net = pd.Series(net, index=p.index)
        out = _stats(net.values, p.fillna(0.0).values)
        out["years"] = {int(y): float((np.prod(1 + g.values) - 1) * 100) for y, g in net.groupby(net.index.year)}
        return out

    res = {"years": years, "picks": picks, "rho": rho}
    for pool in ("b", "c"):
        res[pool] = chain({y: picks[(pool, y)] for y in years})
        res[pool + "0"] = chain({y: f"{pool}F0" for y in years})
    res["hold"] = chain({y: "hold" for y in years})
    return res


# ---------- 24 closing hours ----------

def _clock(fkey: str) -> dict:
    """The filter fkey on the blend and on 55/20 with the daily bar ending at each of the 24 UTC hours."""
    from dashboard.studies2 import clock_robustness as cr

    rows, problem = _klines("spot", "BTCUSDT", "1h")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"BTCUSDT 1 小時線只有 {len(rows)} 筆")
    hour = 3_600_000
    first = min(rows) + hour
    b0 = -(-first // 86_400_000) * 86_400_000
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    end = min(today_ms, ((max(rows) + hour) // 86_400_000) * 86_400_000)
    k = (end - b0) // hour + 1
    grid = np.full((k, 4), np.nan)
    for ms, v in rows.items():
        i = (ms + hour - b0) // hour
        if 0 <= i < k:
            grid[i] = v
    missing = int(np.isnan(grid[:, 0]).sum())
    price = pd.Series(grid[:, 0]).ffill().bfill().values
    flows = np.nan_to_num(grid[:, 1:], nan=0.0)
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(flows, axis=0)])  # csum[i + 1] = sum of rows 0..i
    b0_ts = pd.Timestamp(b0, unit="ms")
    m = cr._Market(price, b0_ts, cr._prehistory(b0_ts))
    m.w0 = max(int((pd.Timestamp(MAIN_START) - b0_ts).days), 0)
    m.years = (m.days - m.w0) / YEAR
    out = {"b": [], "c": [], "missing": missing, "problem": problem}
    for h in range(24):
        wants = m.wants(h, ("blend", "c55"))
        idx = np.arange(h, k, 24)
        day = np.where((idx >= 24)[:, None], csum[idx + 1] - csum[np.maximum(idx - 23, 0)], np.nan)
        dates = pd.date_range("2000-01-01", periods=len(idx), freq="D")
        bars = pd.DataFrame(day, columns=["volume", "quote", "tb"], index=dates)
        close = pd.Series(price[idx], index=dates)
        flag = _flags(_features(close, bars))[fkey].values
        for pool, rule in (("b", "blend"), ("c", "c55")):
            base, filt = m.run(h, wants[rule]), m.run(h, wants[rule] * flag)
            out[pool].append((base["ratio"], filt["ratio"], base["ret"], filt["ret"]))
    return out


# ---------- the study ----------

def _zero_fee(bars: pd.DataFrame) -> dict:
    q = bars["quote"]
    t = bars["trades"]
    share = bars["tb"] / bars["volume"]
    a, b = pd.Timestamp(ZERO_FEE[0]), pd.Timestamp(ZERO_FEE[1])
    before = (bars.index >= a - pd.Timedelta(days=120)) & (bars.index < a)
    during = (bars.index >= a) & (bars.index < b)
    after = (bars.index >= b) & (bars.index < b + pd.Timedelta(days=120))
    return {k: {"quote": float(q[m].mean() / 1e8), "trades": float(t[m].mean() / 1e6), "share": float(share[m].mean()),
                "sd": float(share[m].std())} for k, m in (("before", before), ("during", during), ("after", after))}


def _cell(m: dict) -> str:
    return f"{_pct(m['ret'])}｜{_num(m['ratio'])}｜{_pct(m['deepest'])}"


def study() -> dict:
    t0 = time.time()
    problems: list[str] = []
    bars, p = _daily("spot", "BTCUSDT")
    problems += [p] if p else []
    try:
        fut, p = _daily("fut", "BTCUSDT")
        problems += [p] if p else []
    except Exception as exc:  # noqa: BLE001 - the spot signals still stand without it
        fut = None
        problems.append(f"幣安 U 本位合約日線這次取不到（{type(exc).__name__}），現貨對合約的訊號沒有算，沒有改用其他方法繞過。")
    close = te._daily_close()
    btc = _asset(close, bars, fut)
    wf = _walk(btc)
    meas, pred, keys, start = btc["meas"], btc["pred"], btc["keys"], btc["start"]
    last = close.index[-1]
    span = f"{_d(start)} 到 {_d(last)}"

    # ----- other coins -----
    coins = {}
    for coin in COINS:
        try:
            cb, p = _daily("spot", f"{coin}USDT")
            problems += [p] if p else []
            cclose = cb["close"].ffill()
            cclose = cclose[cclose.first_valid_index():]
            coins[coin] = _asset(cclose, cb, None)
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{coin} 的幣安日線這次拿不到或太短（{type(exc).__name__}），沒有算進去。")

    # ----- table 1: predictive -----
    def gap_cell(item, field) -> str:
        if not item:
            return "—"
        lo, hi = item["low"][field], item["high"][field]
        return f"{_pct(lo)}｜{_pct(hi)}"

    rows1 = []
    allg = pred["all"]
    rows1.append(["所有日子（對照）", _pct(allg["f1"]), _pct(allg["f7"]), _pct(allg["f30"]), _of100(allg["fall"]),
                  f"{allg['win']} 段", "—", "—"])
    named = (list(CONT[:5]) + [("ext7", "主動買進比例（7 天）最低一成｜最高一成")] + [(k, n) for k, _, n in EVENTS]
             + [CONT[5]])
    for key, name in named:
        it = pred.get(key)
        label = f"{name}：最低 1/5｜最高 1/5" if key.startswith("r_") else name
        if key == "r_futdiff" and it:
            label += f"（{it['first']} 起）"
        if not it:
            rows1.append([label, "—", "—", "—", "—", "—", "—", "—"])
            continue
        rows1.append([label, gap_cell(it, "f1"), gap_cell(it, "f7"), gap_cell(it, "f30"),
                      f"{it['low']['fall']:.0f}｜{it['high']['fall']:.0f} 次",
                      f"{it['low']['win']}｜{it['high']['win']} 段", _num(it["rho"]),
                      f"{it['years']} 年裡 {it['agree']} 年" if it["years"] else "—"])
    steady = [k for k, _ in named if pred.get(k) and pred[k]["years"] >= 4 and pred[k]["agree"] >= 2 * pred[k]["years"] / 3
              and abs(pred[k]["high"]["f30"] - pred[k]["low"]["f30"]) >= 2.0]
    n_pred = sum(1 for k, _ in named if pred.get(k))

    # ----- table 2: rules and filters -----
    lucky = {}
    for k in keys:
        s = btc["fut_start"] if k == "F6" else start
        suffix = "@fut" if k == "F6" else ""
        base = btc["target"]["bF0"]
        lucky[k] = _luck(base, btc["flags"][k], btc["change"], s, meas[f"b{k}{suffix}"]["ratio"])
    rows2 = [["不加任何訊號（對照）", "—", _cell(meas["hold"]), _cell(meas["bF0"]), "—", _cell(meas["cF0"]), "—",
              f"—｜{meas['bF0']['share']:.0f}%｜{meas['cF0']['share']:.0f}%"]]
    gains = {}
    for a_key, f_key in FAMILY.items():
        if f_key not in keys:
            continue
        sfx = "@fut" if f_key == "F6" else ""
        al, bf, cf = meas[f"{a_key}{sfx}"], meas[f"b{f_key}{sfx}"], meas[f"c{f_key}{sfx}"]
        b0, c0, h0 = meas[f"bF0{sfx}"], meas[f"cF0{sfx}"], meas[f"hold{sfx}"]
        gains[f_key] = (bf["ratio"] - b0["ratio"], cf["ratio"] - c0["ratio"], bf, b0, cf, c0)
        name = f"單獨：{ALONE[a_key]}／濾網：遇到「{FILTERS[f_key]}」就空手"
        if sfx:
            name += f"（{_d(btc['fut_start'])} 起；同期一直抱著 {_cell(h0)}、混合 {_cell(b0)}、55/20 {_cell(c0)}）"
        rows2.append([name, _cell(al), f"{al['ratio'] - h0['ratio']:+.2f}", _cell(bf),
                      f"{_dnum(bf['ratio'] - b0['ratio'])}（打亂 {SIMS} 次裡 {lucky[f_key] * SIMS / 100:.0f} 次不輸）",
                      _cell(cf), _dnum(cf["ratio"] - c0["ratio"]),
                      f"{al['share']:.0f}%｜{bf['share']:.0f}%｜{cf['share']:.0f}%"])

    best = max(gains, key=lambda k: gains[k][0])
    b_better = sum(1 for k in gains if gains[k][0] > 0.005)
    c_better = sum(1 for k in gains if gains[k][1] > 0.005)
    alone_win = sum(1 for a_key, f_key in FAMILY.items() if f_key in keys and
                    meas[a_key + ("@fut" if f_key == "F6" else "")]["ratio"] >
                    meas["hold" + ("@fut" if f_key == "F6" else "")]["ratio"])

    # ----- table 3: pick in advance -----
    rows3 = []
    for y in wf["years"]:
        pb, pc = wf["picks"][("b", y)], wf["picks"][("c", y)]
        name = f"{y}（到 {last.strftime('%m-%d')}）" if y == last.year and last < pd.Timestamp(f"{y}-12-31") else str(y)
        rows3.append([name, SHORT[pb[1:]], f"{_pct(wf['b']['years'][y])}｜{_pct(wf['b0']['years'][y])}",
                      SHORT[pc[1:]], f"{_pct(wf['c']['years'][y])}｜{_pct(wf['c0']['years'][y])}",
                      _pct(wf["hold"]["years"][y]), _num(wf["rho"][y])])
    rhos = [v for v in wf["rho"].values() if _ok(v)]
    rows3.append([f"{wf['years'][0]} 起接起來（每年報酬｜報酬÷波動｜最深虧）", "—",
                  f"{_cell(wf['b'])}；不加 {_cell(wf['b0'])}", "—", f"{_cell(wf['c'])}；不加 {_cell(wf['c0'])}",
                  _cell(wf["hold"]), f"平均 {_num(float(np.mean(rhos)) if rhos else None)}"])
    picked_none = sum(1 for (pool, _y), k in wf["picks"].items() if k.endswith("F0"))

    # ----- table 4: robustness and other coins -----
    rows4 = []
    fb = best
    sfx = "@fut" if fb == "F6" else ""
    s_rob = btc["fut_start"] if fb == "F6" else start
    sub = btc["target"][["bF0", "cF0", f"b{fb}", f"c{fb}"]]
    hi_cost = _measure_set(sub, btc["change"], s_rob, cost=COST_HIGH)
    late = _measure_set(sub, btc["change"], s_rob, delay=2)
    for name, mm in ((f"成本改成 {COST_HIGH * 100:.2f}%", hi_cost), ("晚一天才照做", late)):
        rows4.append([f"比特幣，濾網「{SHORT[fb]}」：{name}",
                      f"混合 {_cell(mm['bF0'])}；55/20 {_cell(mm['cF0'])}",
                      f"混合 {_cell(mm[f'b{fb}'])}；55/20 {_cell(mm[f'c{fb}'])}",
                      f"混合 {_dnum(mm[f'b{fb}']['ratio'] - mm['bF0']['ratio'])}；55/20 "
                      f"{_dnum(mm[f'c{fb}']['ratio'] - mm['cF0']['ratio'])}",
                      "原本設定（成本 0.10%、隔天照做）：混合 "
                      f"{_dnum(gains[fb][0])}、55/20 {_dnum(gains[fb][1])}"])
    clock = None
    if fb != "F6":
        try:
            clock = _clock(fb)
            if clock["problem"]:
                problems.append(clock["problem"])
        except Exception as exc:  # noqa: BLE001 - the clock check is optional
            problems.append(f"比特幣 1 小時線這次拿不到（{type(exc).__name__}），24 個收盤時刻的檢查沒有做。")
    if clock:
        for pool, label in (("b", "混合"), ("c", "55/20")):
            x = np.array(clock[pool])
            d = x[:, 1] - x[:, 0]
            rows4.append([f"比特幣，濾網「{SHORT[fb]}」加在{label}上：24 個收盤時刻（1 小時線重拼日線）",
                          f"報酬÷波動 {_num(x[:, 0].min())} ～ {_num(x[:, 0].max())}",
                          f"報酬÷波動 {_num(x[:, 1].min())} ～ {_num(x[:, 1].max())}",
                          f"{_dnum(d.min())} ～ {_dnum(d.max())}，中位 {_dnum(float(np.median(d)))}",
                          f"24 個時刻裡 {int((d > 0.005).sum())} 個變好；每年報酬變高的 {int((x[:, 3] > x[:, 2]).sum())} 個"])
    coin_dirs = {}
    for coin, c in coins.items():
        cm, cp = c["meas"], c["pred"]
        g = {k: cm[f"b{k}"]["ratio"] - cm["bF0"]["ratio"] for k in c["keys"]}
        same = [k for k, _ in named if k != "r_futdiff" and pred.get(k) and cp.get(k)]
        agree = sum(1 for k in same if cp[k]["dir"] == pred[k]["dir"])
        coin_dirs[coin] = (agree, len(same))
        wins = sum(1 for a_key, f_key in FAMILY.items() if f_key in c["keys"] and cm[a_key]["ratio"] > cm["hold"]["ratio"])
        s7 = cp.get("r_share7")
        spread = s7["high"]["f30"] - s7["low"]["f30"] if s7 else None
        rows4.append([f"{coin}（{_d(c['start'])} 起），濾網「{SHORT[fb]}」加在混合上" if fb in c["keys"]
                      else f"{coin}（{_d(c['start'])} 起）",
                      f"混合 {_cell(cm['bF0'])}；一直抱著 {_cell(cm['hold'])}",
                      f"混合 {_cell(cm[f'b{fb}'])}" if fb in c["keys"] else "—",
                      _dnum(g[fb]) if fb in g else "—",
                      f"{len(g)} 個濾網裡 {sum(1 for v in g.values() if v > 0.005)} 個讓混合變好；{len(g)} 種單獨規則裡 "
                      f"{wins} 種報酬÷波動贏過一直抱著；{len(same)} 個預測訊號裡 {agree} 個和比特幣同方向；"
                      f"7 天買進比例高組減低組之後 30 天 {_pp(spread)} 個百分點"])

    zf = _zero_fee(bars)
    s7 = pred["r_share7"]
    share_med = float(btc["f"]["share1"][btc["f"].index >= start].median())
    inside = close.index >= start
    above50 = (close > close.rolling(50, min_periods=50).mean()).astype(float)
    obv_same = float((btc["f"]["obv_above"][inside] == above50[inside]).mean() * 100)

    tables = [
        {"caption": (f"比特幣（幣安現貨 BTCUSDT，{span}）：每天收盤時把訊號和它自己過去的數值比（至少 {RANK_MIN} 天，只用更早的日子），"
                     "分成最低 1/5 和最高 1/5；事件型的訊號則和「價格走勢相同、但沒有這個量價特徵」的日子比"
                     "（例如背離是和「價格 30 天漲、累積買賣差也增」比），看之後 1、7、30 天的平均漲跌、30 天內收盤曾經比當天低 10% 以上的比例，"
                     "以及「互不重疊的 30 天段數」：同一組裡的日子大多彼此相連，真正互相獨立的樣本只有這麼多段。"
                     "「排名相關」是訊號名次和之後 30 天漲跌的排名相關（0 = 沒關係）；「逐年」是逐年看高組（有）的 30 天漲跌"
                     "是否和整段同一個方向（兩組都至少 10 天的年份才算）。每格是「低組｜高組」。"),
         "columns": ["訊號：低組｜高組", "之後 1 天", "之後 7 天", "之後 30 天", "30 天內跌 10% 以上（100 次裡）",
                     "互不重疊的 30 天段數", "排名相關", "逐年和整段同方向"],
         "rows": rows1},
        {"caption": (f"比特幣（{span}，成本 {COST * 100:.2f}%，收盤決定、隔天照做）：每種訊號單獨用（持有或空手），"
                     "以及當成濾網加在「17 條順勢規則混合」和「55 日新高進、20 日新低出」上（訊號說不好的日子空手，其他日子照原規則）。"
                     "每格是「每年報酬｜報酬÷波動｜最深虧」。「比混合多出」是報酬÷波動的差；括號裡是運氣對照：把濾網空手的日子整段"
                     f"平移到錯的日期（空手天數和成群的樣子不變）{SIMS} 次，有幾次不輸給真的濾網。最後一欄是平均持有比例"
                     "（單獨｜混合加濾網｜55/20 加濾網）。"),
         "columns": ["訊號（單獨用的規則／當濾網的條件）", "單獨用", "單獨用比一直抱著多出（報酬÷波動）", "加在混合上",
                     "比混合多出（報酬÷波動）", "加在 55/20 上", "比 55/20 多出", "平均持有"],
         "rows": rows2},
        {"caption": (f"事先挑、之後再驗：從 {FIRST_PICK_YEAR} 年起，每年 1 月 1 日只用那天以前的資料（{_d(start)} 起），"
                     "在「不加濾網」和各個濾網之間挑報酬÷波動最高的，照做一整年，再把每一年接起來。現貨對合約的濾網要先有一年資料才能被挑。"
                     "每格是「挑到的那一年｜不加濾網那一年」。「排名相關」是所有訊號規則（單獨用和兩種加濾網）"
                     "用年初以前的分數和當年的分數各排一次名，兩份名次像不像（+1 一樣、0 無關、−1 顛倒）。"),
         "columns": ["年度", "混合：年初挑到", "混合：那一年（挑到｜不加）", "55/20：年初挑到", "55/20：那一年（挑到｜不加）",
                     "一直抱著那一年", "排名相關"],
         "rows": rows3},
        {"caption": (f"拆開檢查（看起來最好的濾網「{FILTERS[fb]}」，這是事後挑的）與換幣：其他幣用完全相同的訊號定義和門檻，"
                     "沒有合約資料所以沒有「現貨對合約」那一條；起算日是上架後所有訊號和 17 條規則都有值的那天（不早於 "
                     f"{MAIN_START}）。每格的三個數字是「每年報酬｜報酬÷波動｜最深虧」，差距是報酬÷波動的差。"),
         "columns": ["看什麼", "不加訊號", "加了訊號", "差多少（報酬÷波動）", "補充"],
         "rows": rows4},
    ]

    # ----- today -----
    f = btc["f"]
    cur = f.iloc[-1]
    fl = btc["flags"].iloc[-1]

    def rank_text(col: str) -> str:
        v = cur[col]
        return "—" if not _ok(v) else f"過去 {v * 100:.0f}% 的日子比它低"

    def yes(v) -> str:
        return "—" if not _ok(v) else ("是" if v >= 0.5 else "否")

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"比特幣收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": "主動買進比例（當天｜7 天｜30 天）",
         "value": f"{cur['share1'] * 100:.1f}%｜{cur['share7'] * 100:.1f}%｜{cur['share30'] * 100:.1f}%；7 天的{rank_text('r_share7')}"},
        {"label": "累積買賣差 90 天變化（占成交量）", "value": f"{cur['cvd90'] * 100:+.1f}%；{rank_text('r_cvd90')}"},
        {"label": "累積買賣差高於 50 日平均", "value": yes(cur["cvd_above"])},
        {"label": "背離（價格 30 天漲、累積買賣差 30 天減）", "value": yes(cur["div_bear"])},
        {"label": "成交量 ÷ 前 30 天平均", "value": f"{_num(cur['volratio'])} 倍；{rank_text('r_volratio')}"},
        {"label": "OBV 高於 50 日平均", "value": yes(cur["obv_above"])},
        {"label": "現貨減合約的主動買進比例（7 天）",
         "value": "—" if not _ok(cur["futdiff"]) else f"{cur['futdiff'] * 100:+.2f} 個百分點；{rank_text('r_futdiff')}"},
        {"label": "六個濾網目前說「空手」的", "value": "、".join(SHORT[k] for k in keys if fl[k] < 0.5) or "沒有"},
        {"label": "這幾格的意思", "value": "只是描述訊號目前停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    lo, hi = s7["low"], s7["high"]
    bg, cg, bf_m, b0_m = gains[best][0], gains[best][1], gains[best][2], gains[best][3]
    agree_all = sum(v[0] for v in coin_dirs.values())
    total_all = sum(v[1] for v in coin_dirs.values())
    coin_better = sum(1 for c in coins.values() if fb in c["keys"] and
                      c["meas"][f"b{fb}"]["ratio"] - c["meas"]["bF0"]["ratio"] > 0.005)
    dv = pred.get("div_bear")
    if dv:
        div_text = (f"；「價漲但累積買賣差減」的背離之後 30 天是 {_pct(dv['high']['f30'])}，價漲買賣差也增是 {_pct(dv['low']['f30'])}"
                    + ("，和「背離會跌」的說法相反" if dv["high"]["f30"] > dv["low"]["f30"] else ""))
    else:
        div_text = ""
    verdict = (
        f"比特幣 {span[:7]} 起，7 天主動買進比例最高 1/5 之後 30 天平均 {_pct(hi['f30'])}、最低 1/5 是 {_pct(lo['f30'])}"
        f"（全部日子 {_pct(allg['f30'])}），排名相關 {_num(s7['rho'])}；{n_pred} 種分組裡高低差 2 個百分點以上、"
        f"三分之二以上年份同方向的只有 {len(steady)} 種{div_text}。"
        f"加在 17 條混合上，{len(gains)} 個濾網裡 {b_better} 個讓報酬÷波動變好，最好的「{SHORT[best]}」是 "
        f"{_num(b0_m['ratio'])} → {_num(bf_m['ratio'])}（錯開日期 {SIMS} 次裡 {lucky[best] * SIMS / 100:.0f} 次不輸），"
        f"加在 55/20 上 {c_better} 個變好，每年事先挑的接起來是 {_num(wf['b']['ratio'])} 對不加 {_num(wf['b0']['ratio'])}。"
        f"換到 {len(coins)} 種幣，這個濾網 {len(coins)} 個裡 {coin_better} 個變好，預測方向和比特幣相同的 {total_all} 個裡 {agree_all} 個。"
    )
    if len(verdict) > 350:  # keep the verdict short: drop the divergence detail first
        verdict = verdict.replace(div_text, "")

    notes = [
        "主動買進＝買方直接吃掉賣單成交的那部分（幣安 K 線的 taker buy volume）；主動賣出＝其餘成交量。主動買進比例＝主動買進量 ÷ 總成交量；"
        "累積買賣差（CVD）＝每天「主動買進 − 主動賣出」一路加總。30 天的累積買賣差變化除以 30 天成交量，剛好等於「2 × 30 天主動買進比例 − 1」，"
        "所以第一張表把兩者放在同一列。",
        f"比特幣幣安現貨的每日主動買進比例中位數是 {share_med * 100:.1f}%，大部分日子落在很窄的範圍裡；"
        "訊號都是「和自己過去比」，不是看有沒有超過 50%。",
        "事先定好的規則：單獨用的六條是 " + "；".join(f"{k}「{v}」" for k, v in ALONE.items()) +
        "。濾網的六條是 " + "；".join(f"{k}「{v}」就空手" for k, v in FILTERS.items()) +
        f"。「最低一成」「前一成」都是和至少 {RANK_MIN} 個更早的日子比；爆量＝當天成交金額 ÷ 前 30 天平均落在過去最高一成；"
        "OBV＝收漲的日子加上成交量、收跌的日子減掉，一路加總。",
        "只看一家交易所（幣安）的一個交易對：別的交易所、別的穩定幣交易對、場外和 ETF 的買賣都不在裡面，"
        "主動買賣力道只是整個市場的一小塊。成交量也會被對敲（自己買自己賣）和交易所的免手續費活動灌大。",
        f"免手續費期間的檢查：幣安 BTCUSDT 在 {ZERO_FEE[0]} 到 {ZERO_FEE[1]} 免手續費。平均每天成交金額和筆數：之前 120 天 "
        f"{zf['before']['quote']:,.0f} 億美元、{zf['before']['trades']:.2f} 百萬筆；免手續費期間 {zf['during']['quote']:,.0f} 億美元、"
        f"{zf['during']['trades']:.2f} 百萬筆；結束後 120 天 {zf['after']['quote']:,.0f} 億美元、"
        f"{zf['after']['trades']:.2f} 百萬筆。主動買進比例的每日起伏（標準差）三段分別是 "
        f"{zf['before']['sd'] * 100:.2f}、{zf['during']['sd'] * 100:.2f}、{zf['after']['sd'] * 100:.2f} 個百分點。"
        "這段期間的成交量和買賣比例不能和其他期間直接比，量能類訊號在這段的排名也會被扭曲。",
        f"這裡一共比了很多東西：第一張表 {len(rows1) - 1} 種分組、六條單獨規則、十二種加濾網，再加上換幣。"
        "比得越多，越容易純靠運氣碰到幾個好看的結果；第四張表檢查的那一個濾網是看完全部結果之後挑的，數字要打折。"
        f"以最好的「{SHORT[best]}」為例：錯開日期 {SIMS} 次裡有 {lucky[best]:.1f}% 不輸給它，但一共試了 {len(gains)} 個濾網，"
        f"{len(gains)} 個都沒用時、其中最好的一個也能做到這樣的機會大約是 100 次裡 "
        f"{(1 - (1 - lucky[best] / 100) ** len(gains)) * 100:.0f} 次。",
        f"OBV 和價格走勢分不太開：{_d(start)} 以來，「OBV 高於 50 日平均」和「收盤高於 50 日均線」在 100 天裡有 {obv_same:.0f} 天"
        "是同一個狀態，所以 OBV 濾網多出來的東西，有很大一部分其實是又多加了一層順勢判斷。",
        "「之後 30 天」的樣本彼此重疊很多：相連的日子幾乎看到同一段漲跌，所以表裡的平均看起來比實際上更有把握；"
        "「互不重疊的 30 天段數」才是比較接近真正的樣本數。",
        f"成本：每次買賣扣交易金額的 {COST * 100:.2f}%（第四張表另算 {COST_HIGH * 100:.2f}%）；混合是部分持有，每天調回目標比例的小額買賣也算成本。"
        "現金沒有利息，一直抱著沒有扣成本，沒有算稅。",
        "比特幣的價格用儀表板原本的日線（2017-08 以後就是幣安現貨 UTC 0 點收盤；更早的部分只用來讓 200 日均線等長天數規則先有值）；"
        f"成交量資料從 2017-08 才有，加上排名要先累積 {RANK_MIN} 天，所以比較從 {_d(start)} 開始。"
        f"合約資料從 2019-09 才有，現貨對合約那一條從 {_d(btc['fut_start']) if btc['fut_start'] is not None else '—'} 才開始算。",
        "其他幣（ETH、SOL、BNB、XRP）是今天還在的大幣，有倖存者偏誤；它們的起算日不同，數字不能直接比大小。",
        "過去的結果不保證未來會一樣。這裡只是整理這些訊號過去和之後漲跌的關係，以及它們目前的狀態，不是買賣建議。",
    ]
    if clock:
        notes.insert(8, f"24 個收盤時刻的檢查：用幣安 BTCUSDT 1 小時線（含主動買進量）把日線在每個整點重拼一次，訊號和規則都用那個時刻的"
                        f"日線算，損益一律用 UTC 0 點到 0 點計算（和「換一個收盤時刻」那一頁同一套算法）；1 小時線整段缺 "
                        f"{clock['missing']:,} 個小時，缺的小時沒有成交量、價格沿用前一小時。")
    notes += problems
    notes.append(f"這次計算花了約 {time.time() - t0:.0f} 秒。")

    return {
        "id": "order_flow",
        "title": "主動買賣力道與成交量",
        "question": "每天的主動買賣力道（主動買進比例、累積買賣差）和成交量，能不能看出比特幣之後的漲跌？單獨用、或加在順勢規則上有沒有多出東西？換到其他幣還成立嗎？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 K 線（含主動買進量）：BTC、ETH、SOL、BNB、XRP 對 USDT",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "幣安 U 本位合約 K 線（含主動買進量），BTCUSDT",
             "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data"},
            {"name": "blockchain.info 每日比特幣價格（只用來讓長天數規則先有值）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Bianchi, Babiak & Dickerson（2022），Journal of Banking & Finance：加密貨幣的成交量與反轉", "url": ""},
            {"name": "Sanhueza 等（2025）：情緒與資金流變數在樣本外最弱", "url": ""},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
