"""Historical study "risk_flags": several risk-off warnings each looked useful in earlier studies but had too few
cases to prove alone. Does a score that simply COUNTS how many of them are on, fixed in advance, help the page's
main signal (close above the 50-day average, ma50) when it is used to cut exposure?

Pre-registered design (written down before any result was looked at; nothing else is tried, no weights, no tuning):
- Five flags, each 0/1, computed only from data known at the close of day t, with the definitions of the source
  studies:
  F1 futures basis >= 20% a year (basis_filter: Deribit quarterly against Bitstamp spot, nearest quarterly with more
     than 14 days left, a missing day repeats the last value for at most 3 days);
  F2 7-day average funding in its top tenth (published_rules / walkforward: Binance perpetual, mean settlement of each
     UTC day, 7-day mean rounded to 6 decimals, above the 90th percentile of all its EARLIER days, at least 365 days);
  F3 US 10-year yield up more than 0.5 points over 60 trading days (cross_market, Yahoo ^TNX);
  F4 the next day lies within the 7 days before a monthly Deribit expiry (forced_flows: last Friday of the month,
     08:00 UTC; the 7 daily candles before it);
  F5 the 50-day average lower than 10 days earlier (crash_timing's slope10, reversed).
  A flag whose data do not exist yet counts as off.
- Score = number of flags on (0-5). Rule R: position = ma50 x (1 if score <= 1, 0.5 if score = 2, 0 if score >= 3).
  Secondary variant R2: score 0 / 1 / >= 2 -> 1 / 0.5 / 0.
- Judged against ma50 with main_signal_check's fair grid (108 monthly start dates 2015-01 ... 2023-12 on the UTC
  daily closes; 72 start months x 24 closing hours = 1,728 cells, textbook and ordinary person with 200 simulations
  per clock; ETH, SOL, BNB, XRP, MSTR), its ranking score, crash_timing's share of each 50%+ bear market avoided
  (whipsaws inside the crash counted) and the median deepest loss of the 1,728 cells.
- Pass test (stated in advance): ranking score >= ma50's, median deepest loss of the 1,728 cells shallower by at
  least 3 points, and a larger median share of the bear markets avoided.
- Also: a paired block bootstrap (60-day blocks, 1,000 draws) of R minus ma50 in return / volatility on bitcoin's
  daily closes from 2015-01-01, and a placebo in which each of the five flag series is circularly shifted by its own
  random amount (200 draws): the share of placebos with a return / volatility at least as high as R's.
- Leave-one-out per flag is description only, never used to re-select.
- Other assets: ETH uses its own basis and funding; SOL, BNB, XRP have no basis series and use F2-F5 with their own
  funding; MSTR has neither and uses bitcoin's funding for F2 plus F3-F5 (US trading days; F4 looks at the next
  trading day).

No look-ahead: every flag uses data up to the decision close (checked by perturbing everything after three cut
dates). Only public, key-less endpoints through the dashboard's helpers; stores of other studies are only read.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import basis_filter as bf
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import cross_market as cm
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import walkforward as wf

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
LOSS_BAR = msc.LOSS_BAR
EVAL_FROM = msc.DAILY_STARTS[0]
BASIS_AT = 20.0
FUND_TOP, FUND_MIN = 0.90, 365
RATE_DAYS, RATE_UP = 60, 0.5
EXP_DAYS = 7
SLOPE_DAYS = 10
BLOCK, BOOT = 60, 1000
PLACEBO = 200
DEEP_BAR = 3.0  # points
FLAGS = ("F1", "F2", "F3", "F4", "F5")
KEYS = ("ma50", "R", "R2")
SEED_KI = {"ma50": 1, "R": 50, "R2": 51}  # ma50 uses main_signal_check's draws
FAPI = "https://fapi.binance.com"
FUNDING_SINCE = "2019-09-01"
FUND_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT", "XRP": "XRPUSDT",
               "MSTR": "BTCUSDT"}
BASIS_COIN = {"BTC": ("BTC", "btcusd"), "ETH": ("ETH", "ethusd")}
FLAG_NAME = {"F1": "期貨溢價每年 20% 以上", "F2": "資金費率 7 天平均在自己歷史最高的一成",
             "F3": "十年期美債殖利率 60 個交易日升超過 0.5 個百分點", "F4": "月度選擇權結算前 7 天",
             "F5": "50 日均線比 10 天前低（下彎）"}
FLAG_SHORT = {"F1": "期貨溢價過高", "F2": "資金費率過熱", "F3": "殖利率急升", "F4": "結算週", "F5": "均線下彎"}
NAME = {"ma50": "50 日均線（現行主訊號）", "R": "R：均線 ×（警訊 ≤1 全部、2 個一半、≥3 現金）",
        "R2": "R2：均線 ×（警訊 0 全部、1 個一半、≥2 現金）"}
SHORT = {"ma50": "50 日均線", "R": "規則 R", "R2": "規則 R2"}

_pct, _pp, _d, _ok = cr._pct, cr._pp, cr._d, cr._ok


def _share(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def _num(v, digits: int = 2) -> str:
    return cr._num(v, digits)


def _clean(obj):
    if obj is None:
        return "—"
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else "—"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ---------- data (other studies' stores are only read) ----------

def _deribit(coin: str, store: dict) -> dict:
    """basis_filter._deribit without writing its store: stored contracts are reused, the others fetched into memory."""
    today = pd.Timestamp(common.today_utc())
    out = {}
    for name, expiry in bf._quarterlies(coin, today):
        a = expiry - pd.Timedelta(days=bf.WINDOW)
        if a > today:
            continue
        rows = store.get(name)
        if rows is None:
            try:
                r = common.get_json(bf.DERIBIT, {"instrument_name": name, "resolution": "60",
                                                 "start_timestamp": int(a.value // 10 ** 6),
                                                 "end_timestamp": int((expiry - pd.Timedelta(days=bf.MIN_LEFT - 1)).value
                                                                      // 10 ** 6)})["result"]
            except Exception:  # noqa: BLE001 - a contract that never existed answers with an error
                r = {}
            rows = []
            if r.get("status") == "ok":
                last_of_day: dict[str, float] = {}
                for t, c, v in zip(r["ticks"], r["close"], r["volume"]):
                    start = pd.Timestamp(t, unit="ms")
                    if start.hour >= 21 and start + pd.Timedelta(hours=1) <= today and v and v > 0 and c and c > 0:
                        last_of_day[start.strftime("%Y-%m-%d")] = float(c)
                rows = [[d, c] for d, c in sorted(last_of_day.items())]
        if rows:
            s = pd.Series([c for _, c in rows], index=pd.to_datetime([d for d, _ in rows]))
            out[name] = (expiry, s[~s.index.duplicated()].sort_index())
    if len(out) < 4:
        raise RuntimeError(f"Deribit {coin} 季度期貨只拿到 {len(out)} 個合約")
    return out


def _basis(coin: str, store: dict) -> pd.Series:
    code, pair = BASIS_COIN[coin]
    spot = bf._bitstamp(pair)
    value, _ = bf._basis(_deribit(code, store), spot["close"])
    return value


def _funding(symbol: str) -> pd.Series:
    """walkforward._funding_daily for any Binance perpetual: mean settlement of each finished UTC day (% per period)."""
    out, start = [], int(pd.Timestamp(FUNDING_SINCE, tz="UTC").timestamp() * 1000)
    for _ in range(30):
        page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1000, "startTime": start})
        if not page:
            break
        out += page
        start = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    if len(out) < 1000:
        raise RuntimeError(f"幣安 {symbol} 資金費率只回傳 {len(out)} 筆")
    s = pd.Series([float(p["fundingRate"]) * 100 for p in out], index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
    daily = s[~s.index.duplicated()].sort_index().resample("1D").mean()
    return daily[daily.index < pd.Timestamp(common.today_utc())]


def _tnx() -> tuple[pd.Series, str]:
    """cross_market's stored ^TNX closes (read only); fetched fresh into memory only when the store is a week old."""
    path = common.STORE / "cross_market_tnx.json"
    days, note = {}, ""
    try:
        days = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        days = {}
    if not days or pd.Timestamp(max(days)) < pd.Timestamp(common.today_utc()) - pd.Timedelta(days=7):
        try:
            days = cm._yahoo_close("^TNX")
        except Exception as exc:  # noqa: BLE001
            if not days:
                raise
            note = f"十年期殖利率這次抓不到新的（{type(exc).__name__}），用 cross_market 存下的資料（到 {max(days)}）。"
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s[np.isfinite(s.values)], note


# ---------- flags ----------

def _last_friday(d: pd.DatetimeIndex) -> pd.DatetimeIndex:
    end = d + pd.offsets.MonthEnd(0)
    return end - pd.to_timedelta((end.weekday - 4) % 7, unit="D")


def _expiry_flag(index: pd.DatetimeIndex, trading: bool) -> pd.Series:
    """1 when the day whose return the position is held over lies 1..7 days before the month's last Friday."""
    if trading:
        nxt = pd.DatetimeIndex(list(index[1:]) + [index[-1] + pd.offsets.BDay(1)])
    else:
        nxt = index + pd.Timedelta(days=1)
    gap = (_last_friday(nxt.normalize()) - nxt.normalize()).days
    return pd.Series(((gap >= 1) & (gap <= EXP_DAYS)).astype(float), index=index)


def _funding_flag(daily: pd.Series | None, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    if daily is None:
        z = pd.Series(np.nan, index=index)
        return pd.Series(0.0, index=index), z, z
    avg7 = daily.rolling(7, min_periods=7).mean().round(6).dropna()
    line = avg7.expanding(min_periods=FUND_MIN).quantile(FUND_TOP).shift(1).round(6)
    hot = ((avg7 > line) & line.notna()).astype(float)

    def on(s: pd.Series) -> pd.Series:
        return s.reindex(index)

    return on(hot).fillna(0.0), on(avg7), on(line)


def _flags(close: pd.Series, basis: pd.Series | None, funding: pd.Series | None, tnx: pd.Series | None,
           trading: bool = False) -> tuple[pd.DataFrame, dict]:
    """Flag table on close.index (0/1, a flag without data is 0) and the raw readings behind it."""
    idx = close.index
    raw = {}
    if basis is not None:
        b = basis.reindex(idx).ffill(limit=bf.FILL)
        f1 = ((b >= BASIS_AT) & b.notna()).astype(float)
        raw["basis"] = b
    else:
        f1 = pd.Series(0.0, index=idx)
    f2, avg7, line = _funding_flag(funding, idx)
    raw["avg7"], raw["line"] = avg7, line
    if tnx is not None:
        rise = tnx - tnx.shift(RATE_DAYS)
        on = rise.reindex(idx.union(rise.index)).ffill().reindex(idx)
        f3 = (on > RATE_UP).astype(float)
        raw["rise"] = on
    else:
        f3 = pd.Series(0.0, index=idx)
    f4 = _expiry_flag(idx, trading)
    m50 = close.rolling(50, min_periods=50).mean()
    f5 = (m50 < m50.shift(SLOPE_DAYS)).astype(float)
    raw["m50"] = m50
    return pd.DataFrame({"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5}, index=idx), raw


def _mult(score: np.ndarray, key: str) -> np.ndarray:
    if key == "R":
        return np.where(score <= 1, 1.0, np.where(score == 2, 0.5, 0.0))
    return np.where(score <= 0, 1.0, np.where(score == 1, 0.5, 0.0))


def _ma50(c: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(c, dtype=float))
    m = s.rolling(50, min_periods=50).mean()
    return (s > m).astype(float).where(m.notna()).values


def _targets(close: pd.Series, flags: pd.DataFrame, use=FLAGS) -> pd.DataFrame:
    ma = _ma50(close.values)
    score = flags[list(use)].sum(axis=1).values
    return pd.DataFrame({"ma50": ma, "R": ma * _mult(score, "R"), "R2": ma * _mult(score, "R2")}, index=close.index)


def _legs(ma: np.ndarray, score: np.ndarray, key: str) -> list:
    """The rule as weighted on/off legs, so crash_timing's crash and whipsaw measures apply unchanged."""
    if key == "ma50":
        return [(1.0, ma)]
    a, b = (2, 1) if key == "R" else (1, 0)
    return [(0.5, ma * (score <= a)), (0.5, ma * (score <= b))]


# ---------- fair grid (main_signal_check's pieces) ----------

def _daily(close: pd.Series, target: pd.DataFrame, year: int, first, last) -> dict:
    """Same start dates as main_signal_check (first day on which all its candidates have a value)."""
    valid = msc._targets(close, year).notna().all(axis=1)
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    change = close.pct_change().fillna(0.0).values
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out = {"starts": labels, "hold": (hr[:, 0], hd[:, 0]), "net": {}}
    for k in KEYS:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
        out["net"][k] = pd.Series(net, index=close.index)
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, daily_flags: pd.DataFrame) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, YEAR)
    hr, hd = hr[:, 0], hd[:, 0]
    # a decision at hour h of date D knows the UTC daily data up to the candle labelled D - 1
    known = daily_flags.reindex(m.dates - pd.Timedelta(days=1)).fillna(0.0)
    base = known[["F1", "F2", "F3", "F4"]].sum(axis=1).values
    text = {k: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for k in KEYS}
    person = {k: (np.zeros((len(starts), 24, SIMS)), np.zeros((len(starts), 24, SIMS))) for k in KEYS}
    for h in range(24):
        closes = pd.Series(np.concatenate([pre, price[h::24]]))
        ma = np.nan_to_num(_ma50(closes.values)[npre:][:days], nan=0.0)
        m50 = closes.rolling(50, min_periods=50).mean()
        f5 = (m50 < m50.shift(SLOPE_DAYS)).astype(float).values[npre:][:days]
        score = base + f5
        wants = {"ma50": ma, "R": ma * _mult(score, "R"), "R2": ma * _mult(score, "R2")}
        for k in KEYS:
            w = wants[k].astype(float)
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * SEED_KI[k] + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]), starts, YEAR)
            person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
    return {"starts": labels, "hold": (hr, hd), "text": text, "person": person}


def _summary(d: dict, c: dict, o: dict) -> dict:
    hr, hd = c["hold"]
    s = {}
    for k in KEYS:
        dr, dd = d[k]
        tr, td = c["text"][k]
        edge = tr - hr[:, None]
        pr, pdd = c["person"][k]
        pedge = pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > d["hold"][0]).sum()), "d_shallow": int((dd > d["hold"][1]).sum()),
             "d_deep_med": float(np.median(dd)),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_p10": float(np.percentile(edge, 10)), "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": {a: (len(o[a]["starts"]), int((o[a][k][0] > o[a]["hold"][0]).sum()),
                            float(np.median(o[a][k][1])), float(np.median(o[a][k][0] - o[a]["hold"][0]))) for a in o}}
        x["o_n"] = sum(v[0] for v in x["o_each"].values())
        x["o_beat"] = sum(v[1] for v in x["o_each"].values())
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        s[k] = x
    return s


# ---------- bitcoin-only checks ----------

def _ratio_rows(net: np.ndarray) -> np.ndarray:
    return rt._ratio(net)


def _bootstrap(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Paired circular block bootstrap of ratio(a) - ratio(b)."""
    n = len(a)
    rng = np.random.default_rng(seed)
    k = -(-n // BLOCK)
    diffs = []
    for chunk in range(0, BOOT, 100):
        m = min(100, BOOT - chunk)
        starts = rng.integers(0, n, size=(m, k))
        idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(m, -1)[:, :n]
        diffs.append(_ratio_rows(a[idx]) - _ratio_rows(b[idx]))
    diffs = np.concatenate(diffs)
    return {"obs": float(_ratio_rows(a[None, :])[0] - _ratio_rows(b[None, :])[0]), "med": float(np.median(diffs)),
            "lo": float(np.percentile(diffs, 5)), "hi": float(np.percentile(diffs, 95)),
            "le0": float((diffs <= 0).mean())}


def _deep_starts(net: np.ndarray, starts: np.ndarray) -> float:
    _, dd = msc._suffix(msc._equity(net), starts, YEAR)
    return float(np.median(dd[:, 0]))


def _placebo(close: pd.Series, flags: pd.DataFrame, change: np.ndarray, first: int, starts: np.ndarray,
             real: dict) -> dict:
    """Each flag series circularly shifted by its own random amount inside the measured span."""
    rng = np.random.default_rng(SEED + 7)
    ma = _ma50(close.values)
    f = flags.values.copy()
    n = len(f) - first
    out = {k: {"ratio": [], "deep": []} for k in ("R", "R2")}
    for _ in range(PLACEBO):
        g = f.copy()
        for j in range(f.shape[1]):
            g[first:, j] = np.roll(f[first:, j], int(rng.integers(1, n)))
        score = g.sum(axis=1)
        for k in ("R", "R2"):
            pos = pd.Series(ma * _mult(score, k)).shift(1).values
            net, _ = te._run(pos, change, COST)
            out[k]["ratio"].append(float(_ratio_rows(net[first:][None, :])[0]))
            out[k]["deep"].append(_deep_starts(net, starts))
    res = {}
    for k in ("R", "R2"):
        r, dpt = np.array(out[k]["ratio"]), np.array(out[k]["deep"])
        res[k] = {"ratio_med": float(np.median(r)), "ratio_ge": float((r >= real[k]["ratio"] - 1e-12).mean()),
                  "deep_med": float(np.median(dpt)), "deep_ge": float((dpt >= real[k]["deep"] - 1e-12).mean())}
    return res


def _crash_stats(close: pd.Series, flags: pd.DataFrame, use=FLAGS) -> dict:
    c = close.values.astype(float)
    ma = _ma50(c)
    score = flags[list(use)].sum(axis=1).values
    bears = ct._bears(close)
    done = [b for b in bears if b["done"]]
    first = int(close.index.searchsorted(pd.Timestamp(ct.BEAR_FROM)))
    out = {}
    for k in KEYS:
        legs = _legs(ma, score, k)
        each = [ct._crash(c, legs, b)["avoid"] for b in done]
        out[k] = {"avoid": float(np.median(each)) if each else None, "each": each,
                  "whip": ct._whipsaws(c, legs, first)}
    out["bears"] = done
    return out


def _lookahead(close: pd.Series, basis, funding, tnx) -> tuple[int, int]:
    base = _targets(close, _flags(close, basis, funding, tnx)[0])
    rng = np.random.default_rng(11)
    bad = cuts = 0
    for back in (900, 400, 60):
        cut = close.index[-back]
        c2 = close.copy()
        late = c2.index > cut
        c2[late] = c2[late] * np.exp(rng.normal(0, 0.2, int(late.sum())))
        b2 = basis.copy()
        b2[b2.index > cut] += rng.normal(0, 30, int((b2.index > cut).sum()))
        f2 = funding.copy()
        f2[f2.index > cut] += rng.normal(0, 0.05, int((f2.index > cut).sum()))
        t2 = tnx.copy()
        t2[t2.index > cut] += rng.normal(0, 1.0, int((t2.index > cut).sum()))
        alt = _targets(c2, _flags(c2, b2, f2, t2)[0])
        cuts += 1
        for k in KEYS:
            a, b = base[k][base.index <= cut].values, alt[k][alt.index <= cut].values
            if not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)):
                bad += 1
    return cuts, bad


# ---------- the whole computation ----------

def _compute() -> dict:
    problems: list[str] = []
    store = bf._read_store()
    close = te._daily_close()
    basis = {"BTC": _basis("BTC", store)}
    try:
        basis["ETH"] = _basis("ETH", store)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"以太幣的期貨溢價這次取不到（{type(exc).__name__}），以太幣只用 F2–F5。")
    funding = {}
    for coin, symbol in FUND_SYMBOL.items():
        if symbol in {FUND_SYMBOL[k] for k in funding}:
            funding[coin] = funding[next(k for k in funding if FUND_SYMBOL[k] == symbol)]
            continue
        try:
            funding[coin] = _funding(symbol)
        except Exception as exc:  # noqa: BLE001
            if coin == "BTC":
                raise
            funding[coin] = None
            problems.append(f"{coin} 的資金費率這次取不到（{type(exc).__name__}），它的 F2 當作沒亮。")
    tnx, note = _tnx()
    if note:
        problems.append(note)

    flags, raw = _flags(close, basis["BTC"], funding["BTC"], tnx)
    target = _targets(close, flags)
    daily = _daily(close, target, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])

    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.insert(0, problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre, flags)

    other_data, more = msc._load_other()
    problems += more
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other, other_flags, used = {}, {}, {}
    for key, (oc, since, year) in other_data.items():
        try:
            trading = key == "MSTR"
            fl, _ = _flags(oc, basis.get(key), funding.get(key), tnx, trading=trading)
            tg = _targets(oc, fl)
            other[key] = _daily(oc, tg, year, since or oc.index[0], three)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
            other_flags[key] = fl
            used[key] = [f for f in FLAGS if f != "F1" or basis.get(key) is not None]
        except Exception as exc:  # noqa: BLE001
            other.pop(key, None)
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")
    s = _summary(daily, clock, other)

    crash = _crash_stats(close, flags)

    first = int(close.index.searchsorted(pd.Timestamp(EVAL_FROM)))
    change = close.pct_change().fillna(0.0).values
    nets = {k: daily["net"][k].values for k in KEYS}
    starts_pos = np.array([close.index.get_loc(t) for t in daily["starts"]], dtype=int)
    real = {k: {"ratio": float(_ratio_rows(nets[k][first:][None, :])[0]), "deep": _deep_starts(nets[k], starts_pos)}
            for k in KEYS}
    span = {k: wf._score_parts(nets[k][first:]) for k in KEYS}
    boot = {k: _bootstrap(nets[k][first:], nets["ma50"][first:], SEED + (1 if k == "R" else 2)) for k in ("R", "R2")}
    placebo = _placebo(close, flags, change, first, starts_pos, real)

    # all five flags have data from here on: the same comparison on that span only (description)
    full_from = max(raw["line"].first_valid_index() or close.index[-1], basis["BTC"].index[0])
    ff = int(close.index.searchsorted(full_from))
    late = {k: wf._score_parts(nets[k][ff:]) for k in KEYS}

    loo = {}
    for drop in FLAGS:
        use = tuple(f for f in FLAGS if f != drop)
        tg = _targets(close, flags, use)
        net, _ = te._run(tg["R"].shift(1).values, change, COST)
        cs = _crash_stats(close, flags, use)
        loo[drop] = {"ratio": float(_ratio_rows(net[first:][None, :])[0]), "deep": _deep_starts(net, starts_pos),
                     "avoid": cs["R"]["avoid"]}
    after = flags.index >= pd.Timestamp(EVAL_FROM)
    ma = pd.Series(_ma50(close.values), index=close.index)
    desc = {}
    for f in FLAGS:
        on = flags[f][after]
        desc[f] = {"share": float(on.mean()), "with_ma": float((ma[after][on > 0] > 0.5).mean()) if (on > 0).any() else None,
                   "first": flags.index[(flags[f] > 0).values.argmax()] if (flags[f] > 0).any() else None}
    score_after = flags[after].sum(axis=1)
    held = ma[after] > 0.5
    dist = {n: float(((score_after == n) & held).sum() / max(held.sum(), 1)) for n in range(6)}
    look = _lookahead(close, basis["BTC"], funding["BTC"], tnx)
    return {"close": close, "flags": flags, "raw": raw, "target": target, "daily": daily, "clock": clock,
            "other": other, "used": used, "s": s, "crash": crash, "boot": boot, "placebo": placebo, "real": real,
            "span": span, "late": late, "full_from": full_from, "loo": loo, "desc": desc, "dist": dist,
            "look": look, "problems": problems, "basis_first": basis["BTC"].index[0],
            "fund_first": funding["BTC"].index[0]}


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    s, close, crash = r["s"], r["close"], r["crash"]
    n_d, n_c = s["ma50"]["d_n"], s["ma50"]["c_n"]
    base = s["ma50"]
    tests = {}
    for k in ("R", "R2"):
        x = s[k]
        a = x["low"] >= base["low"] - 1e-12
        b = x["c_deep_med"] >= base["c_deep_med"] + DEEP_BAR - 1e-12
        c = (crash[k]["avoid"] or -1) > (crash["ma50"]["avoid"] or 0) + 1e-12
        tests[k] = (a, b, c)

    hd = r["clock"]["hold"][1]
    rows1 = []
    for k in KEYS:
        x = s[k]
        w = crash[k]["whip"]
        if k == "ma50":
            cell = "比較的基準"
        else:
            a, b, c = tests[k]
            cell = "、".join([("分數不輸" if a else "分數輸"), ("虧損淺 3 點以上" if b else "虧損沒淺 3 點"),
                             ("避開更多" if c else "避開沒更多")]) + ("→ 通過" if all(tests[k]) else "→ 不通過")
        rows1.append([NAME[k], "／".join(_share(v) for v in x["checks"]), _share(x["low"]),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}",
                      f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}",
                      f"{_share(crash[k]['avoid'])}｜{w['whips']:.1f} 次",
                      _share(x["loss_ok"]), cell])
    rows1.append(["對照：一直抱著不動", "—", "—", f"{_pct(float(np.median(hd)))}｜{_pct(float(hd.min()))}", "—", "0%｜0 次",
                  "—", "—"])

    assets = list(r["other"])
    rows2 = []
    for a in assets:
        cells = [a, "、".join(r["used"][a]) + ("（F2 用比特幣的資金費率）" if a == "MSTR" else ""),
                 f"{_d(r['other'][a]['starts'][0])[:7]} 到 {_d(r['other'][a]['starts'][-1])[:7]}（{len(r['other'][a]['starts'])} 個）"]
        for k in KEYS:
            n, beat, deep, edge = s[k]["o_each"][a]
            cells.append(f"贏 {beat} 個｜{_pct(deep)}｜{_pp(edge)}")
        rows2.append(cells)
    rows2.append(["合計", "—", f"{s['ma50']['o_n']} 個"] + [f"贏 {s[k]['o_beat']} 個（{_share(s[k]['o_beat'] / s[k]['o_n'] if s[k]['o_n'] else None)}）"
                                                          for k in KEYS])

    sp, lt, bt, pl = r["span"], r["late"], r["boot"], r["placebo"]
    rows3 = [
        [f"比特幣 {EVAL_FROM[:4]} 起照課本做：每年報酬｜報酬÷波動｜最深虧損"]
        + [f"{_pct(sp[k][0])}｜{_num(sp[k][2])}｜{_pct(_deepest(r['daily']['net'][k], EVAL_FROM))}" for k in KEYS],
        [f"只看五個警訊都有資料以後（{_d(r['full_from'])} 起）：每年報酬｜報酬÷波動｜最深虧損"]
        + [f"{_pct(lt[k][0])}｜{_num(lt[k][2])}｜{_pct(_deepest(r['daily']['net'][k], r['full_from']))}" for k in KEYS],
        [f"區塊重抽 {BOOT} 次（每塊 {BLOCK} 天）：報酬÷波動比 50 日均線多多少（實際｜中間九成）", "比較的基準"]
        + [f"{bt[k]['obs']:+.2f}｜{bt[k]['lo']:+.2f} ~ {bt[k]['hi']:+.2f}" for k in ("R", "R2")],
        ["區塊重抽：差距 ≤ 0 的比例", "—"] + [_share(bt[k]["le0"]) for k in ("R", "R2")],
        [f"隨機錯位 {PLACEBO} 次：報酬÷波動中位數｜不輸真的那組的比例", "—"]
        + [f"{_num(pl[k]['ratio_med'])}｜{_share(pl[k]['ratio_ge'])}" for k in ("R", "R2")],
        [f"隨機錯位：{n_d} 個起算日最深虧損中位數｜不比真的深的比例", f"真的 {_pct(r['real']['ma50']['deep'])}"]
        + [f"{_pct(pl[k]['deep_med'])}｜{_share(pl[k]['deep_ge'])}（真的 {_pct(r['real'][k]['deep'])}）" for k in ("R", "R2")],
    ]

    loo, desc, dist = r["loo"], r["desc"], r["dist"]
    rows4 = [["五個全用（規則 R 原樣）", "—", "—", "—", _num(r["real"]["R"]["ratio"]), _pct(r["real"]["R"]["deep"]),
              _share(crash["R"]["avoid"])]]
    for f in FLAGS:
        d = desc[f]
        rows4.append([f"{f} {FLAG_NAME[f]}", _d(d["first"]) if d["first"] is not None else "從沒亮過", _share(d["share"]),
                      _share(d["with_ma"]), _num(loo[f]["ratio"]), _pct(loo[f]["deep"]), _share(loo[f]["avoid"])])
    rows4.append(["對照：50 日均線本身", "—", "—", "—", _num(r["real"]["ma50"]["ratio"]), _pct(r["real"]["ma50"]["deep"]),
                  _share(crash["ma50"]["avoid"])])

    tables = [
        {"caption": (f"main_signal_check 的公平檢驗原封不動：比特幣 {n_d} 個起算月（UTC 0 點日線）、{n_c:,} 格起算月×收盤時間（照課本）、"
                     f"同樣 {n_c:,} 格照一般人方式做（10 小時後才做、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次）、"
                     "其他幣和 MSTR。第二欄是四項「報酬贏過同一天起一直抱著」的比例，排名分數＝四項裡最低的。"
                     "避開大熊市跌幅照 crash_timing：每次跌 50% 以上、已走完的熊市，從高點到低點實際剩下多少錢（中途被洗也算），取中位數；"
                     "被洗＝下車後 30 天內用更高價買回（減半算半次）。最後一欄是事先定好的三項檢驗。"),
         "columns": ["規則", "報酬贏過抱著：起算日／格／一般人／其他資產", "排名分數", "最深虧損：中位數｜最深一格",
                     "每年比抱著多（中位數）：照課本｜一般人", "避開大熊市跌幅｜每年被洗", "最深虧損比抱著淺的比例", "事先定的檢驗"],
         "rows": rows1},
        {"caption": ("換資產：同樣的分數和規則原封不動，從 main_signal_check 的同一批起算月到三年前，照課本做、成本 0.10%。"
                     "每格是「報酬贏過抱著的起算月個數｜最深虧損中位數｜每年比抱著多的中位數（百分點）」。沒有期貨溢價資料的幣只有四個警訊，"
                     "門檻照舊（≥3 個才全出），所以比較不容易減碼。"),
         "columns": ["資產", "用到的警訊", "起算月", "50 日均線", "規則 R", "規則 R2"],
         "rows": rows2},
        {"caption": (f"比特幣日線（UTC 0 點）照課本做、成本 0.10%。區塊重抽：把 {EVAL_FROM[:4]} 年起的每日結果切成 {BLOCK} 天一塊、"
                     f"兩條規則同一批日子一起重抽 {BOOT} 次；隨機錯位：五個警訊各自整串往後挪一個隨機天數（繞回開頭），"
                     f"保留每個警訊亮燈的多寡和成串的樣子、但和行情的時間關係被打亂，重做 {PLACEBO} 次。"
                     "如果真的那組不比錯位的好，代表減碼的好處只是「平均少拿一點」而不是時機對。"),
         "columns": ["檢查", "50 日均線", "規則 R", "規則 R2"],
         "rows": rows3},
        {"caption": (f"每個警訊的樣子（{EVAL_FROM[:4]} 年起），和拿掉它之後規則 R 會怎樣——只是描述，不拿來重新挑警訊。"
                     "「亮燈時均線也持有」是那些天 50 日均線本來就持有的比例（均線沒持有時警訊不影響持倉）。"
                     f"後三欄：拿掉這個警訊、其餘四個照同樣門檻算的規則 R，{EVAL_FROM[:4]} 年起報酬÷波動、{n_d} 個起算日最深虧損中位數、避開大熊市跌幅中位數。"),
         "columns": ["警訊", "第一次亮燈", "亮燈天數比例", "亮燈時均線也持有", "拿掉後：報酬÷波動", "拿掉後：最深虧損中位數",
                     "拿掉後：避開大熊市跌幅"],
         "rows": rows4},
    ]

    # ----- today -----
    fl, raw = r["flags"], r["raw"]
    t = close.index[-1]
    last = fl.iloc[-1]
    score = int(last.sum())
    ma_now = r["target"]["ma50"].iloc[-1]

    def state(f: str) -> str:
        return "亮" if last[f] > 0 else "沒亮"

    b = raw["basis"].iloc[-1]
    a7, ln = raw["avg7"].iloc[-1], raw["line"].iloc[-1]
    rise = raw["rise"].iloc[-1] if "rise" in raw else None
    nxt = t + pd.Timedelta(days=1)
    fri = _last_friday(pd.DatetimeIndex([nxt]))[0]
    m50, m50_ago = raw["m50"].iloc[-1], raw["m50"].iloc[-1 - SLOPE_DAYS]

    def word(v) -> str:
        if not _ok(v):
            return "—"
        return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"

    now = [
        {"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {close.iloc[-1]:,.0f} 美元"},
        {"label": f"F1 {FLAG_NAME['F1']}", "value": f"{state('F1')}：溢價每年 {b:.1f}%" if _ok(b) else f"{state('F1')}：這幾天沒有溢價資料"},
        {"label": f"F2 {FLAG_NAME['F2']}",
         "value": (f"{state('F2')}：7 天平均每次 {a7:+.4f}%，門檻 {ln:+.4f}%" if _ok(a7) and _ok(ln) else f"{state('F2')}：—")},
        {"label": f"F3 {FLAG_NAME['F3']}", "value": f"{state('F3')}：60 個交易日變化 {rise:+.2f} 個百分點" if _ok(rise) else f"{state('F3')}：—"},
        {"label": f"F4 {FLAG_NAME['F4']}", "value": f"{state('F4')}：本月結算是 {_d(fri)}（週五 UTC 8 點）"},
        {"label": f"F5 {FLAG_NAME['F5']}", "value": f"{state('F5')}：50 日均線 {m50:,.0f}，10 天前 {m50_ago:,.0f}"},
        {"label": "警訊分數（同時亮起的警訊有幾個）", "value": f"{score} 個"},
        {"label": "50 日均線", "value": word(ma_now)},
        {"label": "規則 R／R2 會持有", "value": f"{word(r['target']['R'].iloc[-1])}／{word(r['target']['R2'].iloc[-1])}"},
        {"label": "這幾格的意思", "value": "只描述各警訊和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    R = s["R"]
    passed = [k for k in ("R", "R2") if all(tests[k])]
    s1 = (f"把期貨溢價過高、資金費率過熱、殖利率急升、結算週、均線下彎五個警訊事先組成「警訊分數」加在 50 日均線上，"
          f"規則 R 的排名分數 {_share(R['low'])}（均線 {_share(base['low'])}）、{n_c:,} 格最深虧損中位數 {_pct(R['c_deep_med'])}"
          f"（均線 {_pct(base['c_deep_med'])}）、大熊市避開 {_share(crash['R']['avoid'])}（均線 {_share(crash['ma50']['avoid'])}）。")
    s2 = ("事先定的三項檢驗：" + ("、".join(SHORT[k] for k in passed) + "通過" if passed else "R 和 R2 都沒通過")
          + f"（R2 排名分數 {_share(s['R2']['low'])}、虧損中位數 {_pct(s['R2']['c_deep_med'])}、避開 {_share(crash['R2']['avoid'])}）。")
    s3 = (f"五個警訊隨機錯位 {PLACEBO} 次，報酬÷波動不輸真的 R 的占 {_share(pl['R']['ratio_ge'])}（R2 {_share(pl['R2']['ratio_ge'])}）；"
          f"區塊重抽 {BOOT:,} 次，R 的報酬÷波動不高於均線的占 {_share(bt['R']['le0'])}。")
    verdict = s1 + s2 + s3

    cuts, bad = r["look"]
    bears = crash["bears"]
    each = "、".join(f"{_d(close.index[b_['top']])} 起 {crash['R']['each'][i] * 100:.0f}%／{crash['R2']['each'][i] * 100:.0f}%"
                     f" 對 {crash['ma50']['each'][i] * 100:.0f}%" for i, b_ in enumerate(bears))
    notes = [
        ("事先登記的做法（看結果前寫好，之後沒改）：五個警訊各為 0 或 1，只用當天收盤前已知的資料——F1 期貨溢價每年 ≥20%（basis_filter 的定義）；"
         "F2 資金費率 7 天平均高過它自己之前所有日子（至少 365 天）的 90%（published_rules／walkforward 的定義，"
         "原始研究是和「之前全部歷史」比，不是只和前一年比）；F3 十年期美債殖利率 60 個交易日升超過 0.5 個百分點（cross_market）；"
         "F4 下一天落在 Deribit 月度結算（每月最後一個週五）前 7 天內（forced_flows）；F5 50 日均線比 10 天前低（crash_timing）。"
         "資料還不存在的警訊當作沒亮。警訊分數＝同時亮起的警訊有幾個（0–5）。規則 R：持有＝50 日均線的持有 ×（分數 ≤1 全部、=2 一半、≥3 現金）；"
         "次要的 R2：分數 0／1／≥2 → 全部／一半／現金。沒有加權、沒有最佳化、沒有試別的門檻。"
         f"通過條件（事先定）：排名分數不低於 50 日均線、{n_c:,} 格最深虧損中位數淺 {DEEP_BAR:.0f} 個百分點以上、大熊市避開的跌幅中位數更大，三項都要過。"
         f"另做區塊重抽（{BLOCK} 天一塊、{BOOT} 次）和隨機錯位（{PLACEBO} 次）；拿掉單一警訊的結果只是描述。"),
        (f"核對：50 日均線在這裡重算的排名分數 {_share(base['low'])}、{n_c:,} 格最深虧損中位數 {_pct(base['c_deep_med'])}，"
         f"大熊市避開 {_share(crash['ma50']['avoid'])}，應和 main_signal_check、crash_timing 相同。"),
        f"各次走完的大熊市避開的跌幅（R／R2 對 50 日均線）：{each}。樣本只有 {len(bears)} 次，差幾個百分點不代表真的比較好。",
        (f"五個警訊都來自已經看過結果的研究（各自「看起來有用但證據不足」才被選進來），雖然組法和門檻在看這次結果前就定好，"
         "挑警訊這一步仍有事後的成分，數字可能比真正事先能做到的好看。"),
        (f"資料起點不同：比特幣期貨溢價從 {_d(r['basis_first'])} 起、資金費率從 {_d(r['fund_first'])} 起（要先累積 365 天才能排名），"
         f"殖利率從 2013 年起，結算日和均線全程都有。所以 2017 年以前的日子只有三個警訊（F3–F5）可能亮，"
         f"五個都有資料要到 {_d(r['full_from'])} 左右；第三張表另列這段以後的結果。2013–2015 年的大熊市也只有三個警訊。"),
        (f"減碼時間：{EVAL_FROM[:4]} 年起 50 日均線持有的日子裡，警訊分數 0／1／2／3／4／5 分別占 "
         + "／".join(_share(dist[n]) for n in range(6)) + "。"),
        "相鄰的起算日共用大部分歷史，24 個收盤時間也只差幾個小時，所以「幾個裡幾個」不是互相獨立的證據。",
        (f"成本很粗略：照課本 {COST * 100:.2f}%、一般人做法 {P_COST * 100:.2f}%，沒有滑價、稅和現金利息；減半也算一次買賣。"
         "換收盤時間時，期貨溢價、資金費率、殖利率、結算週用前一個 UTC 0 點以前已知的值，均線和均線下彎用那個收盤時間自己的價格。"
         f"一般人做法每個收盤時間模擬 {SIMS} 次，50 日均線用和 main_signal_check 同一組亂數。"),
        (f"不偷看未來的檢查：把 {cuts} 個切點之後的價格、期貨溢價、資金費率、殖利率全部隨機打亂再算一次，切點以前的持有狀態"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 處變了，這次結果不可信。")),
        "其他資產是今天還在的大幣和一檔股票，有倖存者偏誤；以太幣有自己的期貨溢價，SOL、BNB、XRP 只有四個警訊，MSTR 用比特幣的資金費率。",
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去的表現和現在的狀態，不是買賣建議。",
    ] + r["problems"]

    result = {
        "id": "risk_flags",
        "title": "五個警訊合成分數來減碼",
        "question": ("期貨溢價過高、資金費率過熱、殖利率急升、結算週、均線下彎，單獨看都證據不足；事先定好「同時亮幾個就減碼多少」，"
                     "加在 50 日均線上，公平檢驗下能不能躲掉更多跌幅、讓虧損變淺？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Deribit 公開 API：季度期貨每小時 K 線", "url": "https://docs.deribit.com/#public-get_tradingview_chart_data"},
            {"name": "Bitstamp 公開 API：BTC/USD、ETH/USD 日線", "url": "https://www.bitstamp.net/api/#ohlc_data"},
            {"name": "幣安 U 本位永續合約資金費率歷史",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "Yahoo Finance 美國十年期公債殖利率（^TNX）", "url": "https://finance.yahoo.com/quote/%5ETNX/history"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }
    return _clean(result)


def _deepest(net: pd.Series, start) -> float:
    part = net[net.index >= pd.Timestamp(start)].values
    equity = np.concatenate([[1.0], np.cumprod(1 + part)])
    return float((equity / np.maximum.accumulate(equity) - 1).min() * 100)


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
