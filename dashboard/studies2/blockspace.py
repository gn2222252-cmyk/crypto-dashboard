"""Historical study "blockspace": demand for bitcoin block space (the fees people pay so that miners put their
transaction into a block, and how many blocks are found per day) and what bitcoin did afterwards. This is not the
miner-capitulation question of hash_ribbons; here the question is whether crowded block space leads price.

What the two sources actually say (read before running):
- CoinDesk, Omkar Godbole, 2023-03-23, "Rising Bitcoin Miner Revenue From Fees Suggests Onset of Major Bull Run":
  Glassnode's two-year "Z-score" for miner revenue from fees "measures the number of standard deviations from the
  two-year mean fee revenue"; it "is usually positive and rising during bull runs and negative during bear runs";
  "a return of high fees has coincided with the beginning of major bull runs"; it had "turned positive for the first
  time since mid-2021". The article does not say whether daily fees or a smoothed series is used, gives no numeric
  reading and does not mention a false signal in November 2022.
- Sanhueza (2026), PhD thesis, Universidad Diego Portales. Daily data Sep 2014 - Apr 2025 (3,854 daily observations),
  targets the log return of the next 1 / 7 / 14 / 28 days. Block Count comes from CoinMetrics, "number of blocks
  mined during a given period" (no transformation is given; its coefficient is printed as 0.00, i.e. the raw level).
  Out of sample: recursive (expanding) OLS r(t,t+h) = a + b x(t) against the random walk with drift (b = 0; Eq. 3),
  splits P/R in {4, 1, 0.4}, Campbell-Thompson OOS R2 with the Clark-West test. Table 3 (recursive, P/R = 1): Block
  Count OOS R2 -0.02 / -0.29 / -0.19 / -0.08 % at 1 / 7 / 14 / 28 days (all negative). Trading (Table 4, Anatolyev-
  Gerko rule; the text maps P/R = 4, 1, 0.4 to estimation/evaluation 30/70, 50/50, 70/30): long when the
  one-step-ahead forecast is positive, SHORT when negative; strategy return = sign x next return. Block Count: P/R = 4 profits 0.58 (Sharpe 0.83) vs buy & hold 0.31 (0.45) -> the quoted "+27% annual excess
  return, +0.38 Sharpe"; P/R = 1: 0.66 (1.03) vs 0.45 (0.70); P/R = 0.4: 0.30 (0.56) vs 0.20 (0.37). The 0.01%
  transaction cost is stated for the second (mean-variance, weight in [0, 1]) strategy, Eq. 11; for the long/short
  rule no cost is stated. In the mean-variance strategy Block Count does not stand out.

Pre-registered design (fixed before running, copied verbatim):
F1 (fee z-score, practitioner rule exactly): z = (fee revenue USD 30-day mean - its trailing 2-year mean) / trailing
2-year std; hold BTC when z > 0, else cash. Also F1+ma50 (both must agree).
B1 (Sanhueza replication as close as the text allows): expanding-window OLS of next-day (and next-7-day) BTC log
return on block-count deviation (daily BlkCnt - 144, or as the thesis defines), OOS from 2015; hold when forecast > 0.
If the thesis defines it differently, follow the thesis and record it.
Tests: forward 30/90-day returns by F1 state and z buckets; OOS R2 for B1 (Campbell-Thompson); split 2015-2019,
2020-2022, 2023-now; rules vs holding and ma50 on main_signal_check's fair grid (start dates, 24 clocks, ordinary
person) and other coins using BTC's on-chain signal as a market filter (state this); placebo with circularly shifted
signal (200 draws, on the ranking score); crash-avoided share with whipsaws.
Pass test (stated now): effect same sign in all three periods with full-sample 90% interval excluding zero AND a
rule's ranking score >= ma50's with median deepest loss no worse AND placebo share <= 10%.

Details fixed with the design: fee revenue USD = CoinMetrics FeeTotNtv x PriceUSD (FeeTotUSD needs a key); the 2-year
window is 730 days, full windows only. Block-count deviation = BlkCnt - 144: with an intercept in the regression
this gives exactly the same forecasts as the raw count the thesis uses. B1 trains on pairs from 2014-09-01 (the
thesis start), forecasts from 2015-01-01, needs 100 pairs; B1-7 is the same with the next-7-day log return and hold
when the 7-day forecast > 0, refit daily. The benchmark of the OOS R2 is the expanding mean of the same target. The
three effects: F1 = mean next-90-day return when z > 0 minus when z <= 0; B1 = OOS R2 at 1 day; B1-7 = OOS R2 at 7
days; each must point the way the source claims (positive). The full sample for the effects is 2015-01-01 on.
Interval: circular moving blocks (30 days for 1-day, 60 for 7/30-day, 120 for 90-day targets), 1,000 draws. The
rule's own effect is used (F1 and F1+ma50 -> F1 effect). Ranking score and grid are main_signal_check's (108 start
months 2015-01..2023-12, 72 x 24 clock cells, ordinary person 0.25% / 10 h later / 10% missed entries, 200 sims, its
seed, ma50 with its random draws, its other-coin start months), other coins ETH SOL BNB XRP MSTR: those coins are held only when BTC's on-chain
signal says so (F1+ma50 uses the coin's own 50-day average). Median deepest loss = median over the 1,728 textbook
cells. Placebo: the z-score series (F1) or the block-count series (B1, refit) circularly shifted by 180 days or more,
200 draws; the ranking score of each draw is computed with 20 ordinary-person simulations per clock, and the real
rule is scored the same way for that comparison; placebo share = share of draws scoring at least as high.
Timing: on-chain day t is used only from the close of day t+1 (daily UTC closes); on the 24-clock grid a close on
UTC date E uses the daily state decided at the UTC close that ended day E-1 (on-chain up to E-2). Forward returns
start at the decision close. Only public key-less endpoints through common.get_json (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = "BlkCnt,FeeTotNtv,PriceUSD,TxCnt"
BCI_FEES = "https://api.blockchain.info/charts/transaction-fees-usd"
THESIS = "https://doctorados.udp.cl/cms/wp-content/uploads/2026/06/Tesis_Doctoral_Aliro_Sanhueza.pdf"
ARTICLE = ("https://www.coindesk.com/markets/2023/03/23/rising-bitcoin-miner-revenue-from-fees-suggests-onset-of-"
           "major-bull-run")
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
MEAN_DAYS, Z_DAYS = 30, 730
NORMAL_BLOCKS = 144
TRAIN_FROM, OOS_FROM, MIN_PAIRS = "2014-09-01", "2015-01-01", 100
THESIS_END = "2025-04-30"
FULL_FROM = "2015-01-01"
PERIODS = (("2015 到 2019", "2015-01-01", "2019-12-31"), ("2020 到 2022", "2020-01-01", "2022-12-31"),
           ("2023 到今天", "2023-01-01", "2100-01-01"))
BLOCK = {1: 30, 7: 60, 30: 60, 90: 120}
PLACEBO, PLACEBO_GAP, PL_SIMS = 200, 180, 20
RULES = ("ma50", "f1", "f1ma", "b1", "b7")
TESTED = ("f1", "f1ma", "b1", "b7")
KI = {"ma50": 1, "f1": 71, "f1ma": 72, "b1": 73, "b7": 74}  # ma50 uses main_signal_check's random draws
EFFECT_OF = {"f1": "f1", "f1ma": "f1", "b1": "b1", "b7": "b7"}
NAMES = {"ma50": "50 日均線（基準）", "f1": "F1：手續費 z 分數 > 0 才持有",
         "f1ma": "F1＋50 日均線：兩個都說好才持有", "b1": "B1：出塊數預測明天會漲才持有",
         "b7": "B1-7：出塊數預測之後 7 天會漲才持有"}
SHORT = {"ma50": "50 日均線", "f1": "F1", "f1ma": "F1＋均線", "b1": "B1", "b7": "B1-7"}
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return (f"{v * 100:.0f}%" if abs(v * 100) >= 0.5 else "0%") if _ok(v) else "—"


def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def _pt(v) -> str:
    return _pp(v * 100) if _ok(v) else "—"


def _r2(v) -> str:
    return f"{v * 100:+.2f}%" if _ok(v) else "—"


def _ci(lo, hi, f) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _stored(name: str, fetch) -> tuple[dict, str | None]:
    """Fetch fresh data; on failure use what an earlier run stored under common.STORE."""
    path = common.STORE / f"blockspace_{name}.json"
    try:
        got = fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return got, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"{name} 取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        return got, f"{name} 這次取不到（{type(exc).__name__}），用之前存下來的資料。"


def _cm_fetch() -> dict:
    reply = common.get_json(CM_URL, {"assets": "btc", "metrics": CM_METRICS, "frequency": "1d",
                                     "start_time": "2009-01-03", "page_size": 10000}, timeout=60)
    data = list(reply["data"])
    for _ in range(5):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=60)
        data += reply["data"]
    out = {}
    for item in data:
        row = {}
        for m in CM_METRICS.split(","):
            try:
                row[m] = float(item[m]) if item.get(m) is not None else None
            except (TypeError, ValueError):
                row[m] = None
        out[item["time"][:10]] = row
    if len(out) < 5000:
        raise RuntimeError(f"CoinMetrics 只回傳 {len(out)} 天")
    return out


def _bci_fetch() -> dict:
    reply = common.get_json(BCI_FEES, {"timespan": "all", "format": "json", "sampled": "false"}, timeout=60)
    out = {pd.Timestamp(int(p["x"]), unit="s").strftime("%Y-%m-%d"): float(p["y"]) for p in reply["values"]}
    if len(out) < 4000:
        raise RuntimeError(f"blockchain.info 手續費只回傳 {len(out)} 天")
    return out


def _onchain() -> tuple[pd.DataFrame, list]:
    got, problem = _stored("cm", _cm_fetch)
    df = pd.DataFrame.from_dict(got, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index < pd.Timestamp(common.today_utc())]
    df = df.asfreq("D")
    return df, [problem] if problem else []


def _zscore(fee_usd: pd.Series) -> pd.Series:
    """Indexed by on-chain day; uses fee revenue up to that day only."""
    m = fee_usd.rolling(MEAN_DAYS, min_periods=MEAN_DAYS).mean()
    mu = m.rolling(Z_DAYS, min_periods=Z_DAYS).mean()
    sd = m.rolling(Z_DAYS, min_periods=Z_DAYS).std(ddof=1)
    return (m - mu) / sd.where(sd > 0)


def _at(series: pd.Series, dates, lag: int) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


# ---------- B1: expanding-window OLS ----------

def _ols(close: pd.Series, x_t: np.ndarray, h: int, train_from: str, oos_from: str, oos_to: str | None = None) -> dict:
    """x_t[i] = predictor usable at decision close i. Target y[i] = log(c[i+h]/c[i]). Forecast at i uses only pairs
    s <= i - h (their targets are known at close i). Returns forecast, benchmark (expanding mean) and target."""
    c = close.values.astype(float)
    n = len(c)
    y = np.full(n, np.nan)
    y[:n - h] = np.log(c[h:] / c[:-h])
    idx = close.index
    train = idx >= pd.Timestamp(train_from)
    if oos_to is not None:
        train &= idx <= pd.Timestamp(oos_to)
    valid = train & np.isfinite(x_t) & np.isfinite(y)
    xv, yv = np.where(valid, x_t, 0.0), np.where(valid, y, 0.0)
    s1, sx, sy = np.cumsum(valid), np.cumsum(xv), np.cumsum(yv)
    sxx, sxy = np.cumsum(xv * xv), np.cumsum(xv * yv)
    f, b = np.full(n, np.nan), np.full(n, np.nan)
    first = int(idx.searchsorted(pd.Timestamp(oos_from)))
    last = n if oos_to is None else int(idx.searchsorted(pd.Timestamp(oos_to), side="right"))
    for i in range(max(first, h), last):
        j = i - h
        k = s1[j]
        if k < MIN_PAIRS or not math.isfinite(x_t[i]):
            continue
        mx, my = sx[j] / k, sy[j] / k
        vxx = sxx[j] / k - mx * mx
        beta = (sxy[j] / k - mx * my) / vxx if vxx > 1e-12 else 0.0
        f[i] = my + beta * (x_t[i] - mx)
        b[i] = my
    return {"f": f, "b": b, "y": y}


def _r2_fn(d):
    return 1 - d["a"].sum(1) / d["b"].sum(1)


def _oos(fit: dict, idx: pd.DatetimeIndex, a: str, b: str, h: int) -> dict:
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)) & np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    em, eb = (fit["y"] - fit["f"])[m] ** 2, (fit["y"] - fit["b"])[m] ** 2
    out = {"n": int(m.sum()), "r2": at._boot({"a": em, "b": eb}, _r2_fn, BLOCK[h])}
    out["hit"] = float(np.mean(np.sign(fit["f"][m]) == np.sign(fit["y"][m]))) if m.any() else None
    out["long"] = float(np.mean(fit["f"][m] > 0)) if m.any() else None
    return out


def _thesis_check(close: pd.Series, x_raw: pd.Series, est_share: float, lag: int) -> dict:
    """The thesis' trading exercise as written: Sep 2014 - Apr 2025, the first est_share of the days only to train
    (the thesis maps P/R = 4, 1, 0.4 to 30/70, 50/50, 70/30), then recursive OLS of the next-day log return, long
    when the forecast > 0, short otherwise, no cost; yearly = mean daily log return x 365."""
    w = close[(close.index >= pd.Timestamp(TRAIN_FROM)) & (close.index <= pd.Timestamp(THESIS_END))]
    x = _at(x_raw, w.index, lag)
    split = w.index[int(len(w) * est_share)]
    fit = _ols(w, x, 1, TRAIN_FROM, _d(split))
    m = np.isfinite(fit["f"]) & np.isfinite(fit["y"])
    s = np.where(fit["f"][m] > 0, 1.0, -1.0) * fit["y"][m]
    bh = fit["y"][m]

    def ann(v):
        return float(v.mean() * YEAR), float(v.mean() / v.std(ddof=1) * math.sqrt(YEAR))
    return {"from": split, "n": int(m.sum()), "strat": ann(s), "hold": ann(bh)}


# ---------- rules ----------

def _states(close: pd.Series, z: pd.Series, x: pd.Series) -> dict[str, np.ndarray]:
    """Daily decision states on the UTC close index (on-chain day t used from close t+1). NaN = no value yet."""
    zt = _at(z, close.index, 1)
    xt = _at(x, close.index, 1)
    f1 = np.where(np.isnan(zt), np.nan, (zt > 0).astype(float))
    fit1 = _ols(close, xt, 1, TRAIN_FROM, OOS_FROM)
    fit7 = _ols(close, xt, 7, TRAIN_FROM, OOS_FROM)
    b1 = np.where(np.isnan(fit1["f"]), np.nan, (fit1["f"] > 0).astype(float))
    b7 = np.where(np.isnan(fit7["f"]), np.nan, (fit7["f"] > 0).astype(float))
    return {"f1": f1, "b1": b1, "b7": b7, "zt": zt, "xt": xt, "fit1": fit1, "fit7": fit7}


def _rules(ma: np.ndarray, f1: np.ndarray, b1: np.ndarray, b7: np.ndarray) -> dict[str, np.ndarray]:
    def nz(a):
        return np.nan_to_num(a, nan=0.0)
    return {"ma50": ma, "f1": f1, "f1ma": np.where(np.isnan(ma) | np.isnan(f1), np.nan, nz(ma) * nz(f1)),
            "b1": b1, "b7": b7}


def _ma(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return ct._gt(c, ct._mean(c, 50))


def _grid_daily(close: pd.Series, pos: dict, year: int, first, last, keys=RULES) -> dict:
    """Textbook runs from monthly start dates (main_signal_check._daily with given positions)."""
    change = close.pct_change().fillna(0.0).values
    known = np.ones(len(close), dtype=bool)
    for k in keys:
        known &= ~np.isnan(pos[k])
    if not known.any():
        raise RuntimeError("規則沒有值")
    begin = max(close.index[int(np.argmax(known))] + DAY, pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in keys:
        net, _ = te._run(np.concatenate([[np.nan], pos[k][:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    """The 24 closing hours of the hourly grid (main_signal_check's), with the 50-day rule per hour."""

    def __init__(self, price, b0, pre):
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]])
            self.ma.append(np.nan_to_num(_ma(c)[npre:][:self.days], nan=0.0))

    def flags(self, daily_index: pd.DatetimeIndex, st: dict) -> dict:
        """A close on UTC date E uses the daily state decided at the close that ended day E-1."""
        return {k: np.nan_to_num(_at(pd.Series(st[k], index=daily_index), self.m.dates, 1), nan=0.0)
                for k in ("f1", "b1", "b7")}

    def want(self, k: str, h: int, fl: dict) -> np.ndarray:
        if k == "ma50":
            return self.ma[h]
        if k == "f1ma":
            return self.ma[h] * fl["f1"]
        return fl[k]

    def full(self, k: str, fl: dict, sims: int = SIMS, seed_add: int = 0) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = self.want(k, h, fl)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[k] + h + seed_add)
            pos = rt._missed(wl, P_MISS, rng, sims)
            r, dd = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, P_COST, None)]), self.starts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - self.hr[:, None], pr - self.hr[:, None, None]
        return {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                "c_deep_worst": float(td.min()),
                "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge))}


def _other_pos(c: pd.Series, btc_index: pd.DatetimeIndex, st: dict, share: bool) -> dict:
    """Other coins: held only when BTC's on-chain signal says so. A share closes before the UTC day ends, so it uses
    the BTC state decided one UTC close earlier."""
    lag = 1 if share else 0
    f1 = _at(pd.Series(st["f1"], index=btc_index), c.index, lag)
    b1 = _at(pd.Series(st["b1"], index=btc_index), c.index, lag)
    b7 = _at(pd.Series(st["b7"], index=btc_index), c.index, lag)
    return _rules(_ma(c.values), f1, b1, b7)


def _score(x: dict) -> None:
    x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                   (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
    x["low"] = min(x["checks"])
    x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])


def _evaluate(k: str, close: pd.Series, pos: dict, clock: _Clock, fl: dict, others: dict, other_pos: dict,
              sims: int, seed_add: int = 0) -> dict:
    d = _grid_daily(close, pos, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], keys=(k,))
    x = clock.full(k, fl, sims, seed_add)
    dr, dd = d[k]
    x["d_n"], x["d_beat"], x["d_shallow"] = len(dr), int((dr > d["hold"][0]).sum()), int((dd > d["hold"][1]).sum())
    x["d_edge"] = float(np.median(dr - d["hold"][0]))
    o_n = o_beat = 0
    x["o_each"] = {}
    for a, (c, _since, year, first) in others.items():
        try:
            g = _grid_daily(c, other_pos[a], year, first, others_last(), keys=(k,))
        except Exception:  # noqa: BLE001
            continue
        n_a, b_a = len(g[k][0]), int((g[k][0] > g["hold"][0]).sum())
        x["o_each"][a] = (n_a, b_a)
        o_n, o_beat = o_n + n_a, o_beat + b_a
    x["o_n"], x["o_beat"] = o_n, o_beat
    _score(x)
    return x


def others_last() -> pd.Timestamp:
    return pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)


# ---------- forward returns by F1 state ----------

def _diff_fn(d):
    return at._mmean(d["y"], d["m"]) - at._mmean(d["y"], 1 - d["m"])


def _fwd_frame(close: pd.Series, st: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["z"] = st["zt"]
    for h in (30, 90):
        df[f"f{h}"] = close.shift(-h) / close - 1
    df["up"] = (df["z"] > 0).astype(float).where(df["z"].notna())
    return df[df["z"].notna()]


def _sub(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]


def _groups(df: pd.DataFrame) -> dict:
    z = df["z"]
    masks = {"z > 0（F1 持有）": z > 0, "z ≤ 0（F1 現金）": z <= 0, "z ≤ −1": z <= -1,
             "−1 < z ≤ 0": (z > -1) & (z <= 0), "0 < z ≤ 1": (z > 0) & (z <= 1), "1 < z ≤ 2": (z > 1) & (z <= 2),
             "z > 2": z > 2}
    out = {}
    for name, m in masks.items():
        mv = m.astype(float).values
        out[name] = {"days": int(m.sum()), "runs": at._runs(m.values),
                     **{f"f{h}": at._boot({"y": df[f"f{h}"].values, "m": mv}, at._cell_fn, BLOCK[h]) for h in (30, 90)}}
    out["diff"] = {h: at._boot({"y": df[f"f{h}"].values, "m": df["up"].values}, _diff_fn, BLOCK[h]) for h in (30, 90)}
    return out


# ---------- look-ahead check ----------

def _lookahead(close: pd.Series, z_from, x: pd.Series, fee_usd: pd.Series, clock: _Clock) -> tuple[int, int]:
    """Scramble closes after a cut close and on-chain values from the cut day on: states up to the cut must not move.
    On the clock grid, for a cut close on date E, on-chain values from E-1 on and daily closes from E on are
    scrambled; clock states up to that close must not move."""
    base = _states(close, z_from(fee_usd), x)
    rng = np.random.default_rng(11)
    bad = cuts = 0
    for cut in (len(close) - 2500, len(close) - 900, len(close) - 60):
        t = close.index[cut]
        c2 = close.copy()
        c2.iloc[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(close) - cut - 1))
        fee2, x2 = fee_usd.copy(), x.copy()
        later = fee2.index >= t
        fee2[later] = fee2[later].values * np.exp(rng.normal(0, 1.0, int(later.sum())))
        later = x2.index >= t
        x2[later] = x2[later].values + rng.normal(0, 20, int(later.sum()))
        alt = _states(c2, z_from(fee2), x2)
        cuts += 1
        for k in ("f1", "b1", "b7"):
            if not np.array_equal(np.nan_to_num(base[k][:cut + 1], nan=-1), np.nan_to_num(alt[k][:cut + 1], nan=-1)):
                bad += 1
        j = int(clock.m.dates.searchsorted(t))
        if 0 < j < clock.days:
            e = clock.m.dates[j]
            c3 = close.copy()
            c3[c3.index >= e] *= np.exp(rng.normal(0, 0.2, int((c3.index >= e).sum())))
            fee3, x3 = fee_usd.copy(), x.copy()
            fee3[fee3.index >= e - DAY] *= np.exp(rng.normal(0, 1.0, int((fee3.index >= e - DAY).sum())))
            x3[x3.index >= e - DAY] += rng.normal(0, 20, int((x3.index >= e - DAY).sum()))
            f0 = clock.flags(close.index, base)
            f1 = clock.flags(c3.index, _states(c3, z_from(fee3), x3))
            cuts += 1
            for k in ("f1", "b1", "b7"):
                if not np.array_equal(f0[k][:j + 1], f1[k][:j + 1]):
                    bad += 1
    return cuts, bad


# ---------- the computation ----------

def _compute() -> dict:
    oc, problems = _onchain()
    fee_usd = (oc["FeeTotNtv"] * oc["PriceUSD"]).dropna()
    fee_usd = fee_usd[fee_usd.index >= fee_usd.index[0]]
    z = _zscore(fee_usd)
    blocks = oc["BlkCnt"].dropna()
    x = blocks - NORMAL_BLOCKS
    close = te._daily_close()
    st = _states(close, z, x)
    ma = _ma(close.values)
    pos = _rules(ma, st["f1"], st["b1"], st["b7"])

    # check source: blockchain.info fees in USD
    bci = None
    try:
        got, problem = _stored("bci", _bci_fetch)
        if problem:
            problems.append(problem)
        s = pd.Series(got, dtype=float)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = s[s.index < pd.Timestamp(common.today_utc())].asfreq("D").ffill()
        zb = _zscore(s)
        both = pd.concat([z, zb], axis=1).dropna()
        both = both[both.index >= pd.Timestamp(FULL_FROM)]
        stb = dict(st)
        stb["zt"] = _at(zb, close.index, 1)
        fb = _fwd_frame(close, stb)
        bci = {"agree": float(((both.iloc[:, 0] > 0) == (both.iloc[:, 1] > 0)).mean()), "n": len(both),
               "diff90": at._boot({"y": _sub(fb, FULL_FROM, "2100-01-01")["f90"].values,
                                   "m": _sub(fb, FULL_FROM, "2100-01-01")["up"].values}, _diff_fn, BLOCK[90])}
    except Exception as exc:  # noqa: BLE001 - only a check
        problems.append(f"blockchain.info 手續費（對照）算不出來（{type(exc).__name__}）。")

    # test 1: forward returns by F1 state
    df = _fwd_frame(close, st)
    groups = {"全部": _groups(_sub(df, FULL_FROM, "2100-01-01"))}
    for name, a, b in PERIODS:
        groups[name] = _groups(_sub(df, a, b))
    early = _groups(_sub(df, "2010-01-01", "2014-12-31"))

    # test 2: OOS R2 for B1
    oos = {}
    for h, key in ((1, "fit1"), (7, "fit7")):
        oos[h] = {"全部": _oos(st[key], close.index, FULL_FROM, "2100-01-01", h)}
        for name, a, b in PERIODS:
            oos[h][name] = _oos(st[key], close.index, a, b, h)
    thesis = {"pr4": _thesis_check(close, x, 0.3, 0), "pr1": _thesis_check(close, x, 0.5, 0),
              "pr4_lag": _thesis_check(close, x, 0.3, 1)}

    # test 3: fair grid
    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.insert(0, problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _Clock(price, b0, pre)
    fl = clock.flags(close.index, st)
    loaded, more = msc._load_other()
    problems += more
    others = {}
    for a, (c, since, year) in loaded.items():
        # main_signal_check's start months: from the first day every one of its rules has a value
        valid = msc._targets(c, year).notna().all(axis=1).values
        if valid.any():
            others[a] = (c, since, year, max(c.index[int(np.argmax(valid))] + DAY, pd.Timestamp(since or c.index[0])))
    other_pos = {a: _other_pos(c, close.index, st, since is not None) for a, (c, since, _y, _f) in others.items()}
    s = {}
    for k in RULES:
        s[k] = _evaluate(k, close, pos, clock, fl, others, other_pos, SIMS)
        d = _grid_daily(close, pos, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], keys=(k,))
        s[k]["d_deep_med"] = float(np.median(d[k][1]))

    # placebo on the ranking score (20 ordinary-person simulations per clock, real rule scored the same way)
    real20 = {k: _evaluate(k, close, pos, clock, fl, others, other_pos, PL_SIMS)["low"] for k in TESTED}
    rng = np.random.default_rng(SEED + 144)
    zv = z.dropna()
    xv = x[x.index >= pd.Timestamp(TRAIN_FROM) - 2 * DAY].dropna()
    plac = {k: [] for k in TESTED}
    for i in range(PLACEBO):
        zs = pd.Series(np.roll(zv.values, int(rng.integers(PLACEBO_GAP, len(zv) - PLACEBO_GAP))), index=zv.index)
        xs = pd.Series(np.roll(xv.values, int(rng.integers(PLACEBO_GAP, len(xv) - PLACEBO_GAP))), index=xv.index)
        sp = _states(close, zs, xs)
        pp = _rules(ma, sp["f1"], sp["b1"], sp["b7"])
        flp = clock.flags(close.index, sp)
        opp = {a: _other_pos(c, close.index, sp, since is not None) for a, (c, since, _y, _f) in others.items()}
        for k in TESTED:
            plac[k].append(_evaluate(k, close, pp, clock, flp, others, opp, PL_SIMS, 7919 * (i + 1))["low"])
    placebo = {k: float(np.mean(np.array(v) >= real20[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    # crashes and whipsaws since 2015
    c = close.values.astype(float)
    first = int(close.index.searchsorted(pd.Timestamp(FULL_FROM)))
    bears = [b for b in ct._bears(close) if b["top"] >= first]
    crash = {k: [ct._crash(c, [(1.0, pos[k])], b) for b in bears] for k in RULES}
    whip = {k: ct._whipsaws(c, [(1.0, pos[k])], first) for k in RULES}
    held = {k: float(np.nanmean(np.nan_to_num(pos[k][first:]))) for k in RULES}

    look = _lookahead(close, _zscore, x, fee_usd, clock)
    return {"oc": oc, "fee_usd": fee_usd, "z": z, "blocks": blocks, "close": close, "st": st, "pos": pos, "bci": bci,
            "groups": groups, "early": early, "oos": oos, "thesis": thesis, "s": s, "clock": clock,
            "real20": real20, "placebo": placebo, "placebo_med": placebo_med, "bears": bears, "crash": crash,
            "whip": whip, "held": held, "look": look, "problems": problems, "others": list(others)}


# ---------- the study ----------

def _sign(v) -> int:
    return 0 if not _ok(v) or v == 0 else (1 if v > 0 else -1)


def study() -> dict:
    r = _compute()
    close, s, g, oos, th = r["close"], r["s"], r["groups"], r["oos"], r["thesis"]
    pnames = [p[0] for p in PERIODS]

    # ----- pass test -----
    eff = {"f1": ("F1：z > 0 減 z ≤ 0 的之後 90 天平均報酬", g["全部"]["diff"][90], [g[p]["diff"][90][0] for p in pnames], _pt),
           "b1": ("B1：明天報酬的樣本外 R²", oos[1]["全部"]["r2"], [oos[1][p]["r2"][0] for p in pnames], _r2),
           "b7": ("B1-7：之後 7 天報酬的樣本外 R²", oos[7]["全部"]["r2"], [oos[7][p]["r2"][0] for p in pnames], _r2)}
    eff_ok = {}
    for k, (_n, full, per, _f) in eff.items():
        est, lo, hi, _nn = full
        eff_ok[k] = bool(_ok(est) and est > 0 and _ok(lo) and lo > 0 and all(_sign(v) == 1 for v in per))
    m = s["ma50"]
    rule_ok = {}
    for k in TESTED:
        x = s[k]
        rule_ok[k] = {"effect": eff_ok[EFFECT_OF[k]], "score": x["low"] >= m["low"] - 1e-12,
                      "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10}
        rule_ok[k]["all"] = all(rule_ok[k].values())
    passed = [k for k in TESTED if rule_ok[k]["all"]]

    # ----- table 1: forward returns by F1 state -----
    def mc(t) -> str:
        return f"{_p(t[0])} {_ci(t[1], t[2], _p)}"

    def ic(t) -> str:
        return f"{_pt(t[0])} {_ci(t[1], t[2], _pt)}"

    rows1 = []
    for name in ("z > 0（F1 持有）", "z ≤ 0（F1 現金）", "z ≤ −1", "−1 < z ≤ 0", "0 < z ≤ 1", "1 < z ≤ 2", "z > 2"):
        x = g["全部"][name]
        rows1.append([name, f"{x['days']:,} 天｜{x['runs']} 段", mc(x["f30"]), mc(x["f90"]),
                      "｜".join(_p(g[p][name]["f90"][0]) for p in pnames), _p(r["early"][name]["f90"][0])])
    rows1.append(["差：z > 0 減 z ≤ 0（百分點）", "—", ic(g["全部"]["diff"][30]), ic(g["全部"]["diff"][90]),
                  "｜".join(_pt(g[p]["diff"][90][0]) for p in pnames), _pt(r["early"]["diff"][90][0])])
    if r["bci"]:
        rows1.append(["對照：手續費改用 blockchain.info，差（90 天）", "—", "—", ic(r["bci"]["diff90"]), "—", "—"])

    # ----- table 2: B1 OOS R2 and the thesis' own exercise -----
    rows2 = []
    for h in (1, 7):
        for p in ["全部"] + pnames:
            x = oos[h][p]
            label = f"{'明天' if h == 1 else '之後 7 天'}，{'2015 到今天' if p == '全部' else p}"
            rows2.append([label, f"{x['n']:,}", f"{_r2(x['r2'][0])} {_ci(x['r2'][1], x['r2'][2], _r2)}",
                          _share(x["hit"]), _share(x["long"])])
    for key, label in (("pr4", "論文的交易練習照做：P/R = 4（論文寫成前 30% 估計）、當天出塊數、做多／放空、不扣成本"),
                       ("pr1", "同上，P/R = 1（前 50% 估計）"), ("pr4_lag", "同上 P/R = 4，但出塊數晚一天才用（本研究的時間規則）")):
        t = th[key]
        rows2.append([f"{label}（{_d(t['from'])} 到 {THESIS_END}）", f"{t['n']:,}",
                      f"每年 {_p(t['strat'][0])}｜夏普 {t['strat'][1]:.2f}",
                      f"抱著 {_p(t['hold'][0])}｜夏普 {t['hold'][1]:.2f}",
                      f"多 {_pt(t['strat'][0] - t['hold'][0])} 點｜夏普 {t['strat'][1] - t['hold'][1]:+.2f}"])

    # ----- table 3: rules on the fair grid -----
    done = [i for i, b in enumerate(r["bears"]) if b["done"]]

    def avoid(k):
        vals = [r["crash"][k][i]["avoid"] for i in done]
        return float(np.median(vals)) if vals else None

    order = sorted(RULES, key=lambda k: -s[k]["low"])
    rows3 = []
    for k in RULES:
        x = s[k]
        rows3.append([NAMES[k], _of(x["d_n"], x["d_beat"]), f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}", _of(x["o_n"], x["o_beat"]),
                      f"{_share(x['low'])}（第 {order.index(k) + 1}）｜{_pct(x['c_deep_med'])}",
                      f"{_share(avoid(k))}｜{r['whip'][k]['whips']:.1f} 次",
                      "—（基準）" if k == "ma50" else _share(r["placebo"][k])])
    hd = r["clock"].hd
    rows3.append(["對照：一直抱著不動", "—", "—", "—", "—", f"—｜{_pct(float(np.median(hd)))}", "0%｜0 次", "—"])

    # ----- table 4: pass test -----
    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows4 = []
    for k in TESTED:
        e = eff[EFFECT_OF[k]]
        est, lo, hi, _n = e[1]
        rows4.append([NAMES[k], f"{e[0]}：{e[3](est)} {_ci(lo, hi, e[3])}；三段 {'／'.join(e[3](v) for v in e[2])}"
                      f"（{yes(rule_ok[k]['effect'])}）",
                      f"{_share(s[k]['low'])} vs {_share(m['low'])}（{yes(rule_ok[k]['score'])}）",
                      f"{_pct(s[k]['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(rule_ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（{yes(rule_ok[k]['placebo'])}）",
                      "通過" if rule_ok[k]["all"] else "沒有通過"])

    n_d, n_c = s["ma50"]["d_n"], s["ma50"]["c_n"]
    tables = [
        {"caption": ("一、手續費收入 z 分數（大家為了讓交易被打包進區塊付給礦工的錢，30 天平均和過去兩年比高出幾個標準差）在收盤那天能拿到的值"
                     "（只用到前一天的鏈上資料），分組看之後 30／90 天比特幣平均漲跌；方括號是 90% 區間（區塊重抽 1,000 次，90 天用 120 天一段）。"
                     "全部＝2015 起。段數＝連續落在同一組的段落數，報酬天天重疊，真正獨立的樣本遠少於天數。最後一欄是 2012-08 到 2014（參考，不算在通過條件裡）。"
                     "z 分數很少到 −1 以下：手續費偶爾暴衝，讓兩年的標準差很大，往下偏不了多少。"),
         "columns": ["分組", "天數｜段數", "之後 30 天平均", "之後 90 天平均", "之後 90 天：" + "｜".join(pnames), "2012–2014 之後 90 天"],
         "rows": rows1},
        {"caption": ("二、出塊數（一天產生幾個區塊，正常約 144 個）預測報酬：每天用到當時為止的資料重新跑一次最小平方法（2014-09 起的資料，"
                     "2015 起才開始預測），預測下一天／下 7 天的對數報酬。樣本外 R²＞0 才代表比「過去平均」猜得準（方括號是 90% 區間）；"
                     "「猜對方向」＝預測和實際同號的比例；「看多」＝預測為正、規則持有的比例。最後三列照論文自己的交易練習做："
                     "前 30%／50% 的日子只拿來估計（論文把 P/R = 4、1 對應到 30/70、50/50），之後預測為正做多、為負放空，"
                     "年報酬＝日對數報酬平均×365，和同期間一直抱著比；"
                     "論文寫的是 P/R = 4 每年 0.58（夏普 0.83）對抱著 0.31（0.45）、P/R = 1 是 0.66（1.03）對 0.45（0.70）。"),
         "columns": ["預測什麼、哪段期間", "預測次數", "樣本外 R²／策略", "猜對方向／抱著", "看多的比例／差距"],
         "rows": rows2},
        {"caption": (f"三、規則和 50 日均線同樣條件考（main_signal_check 的公平考法）：比特幣 {n_d} 個起算月（2015-01 到 2023-12，UTC 0 點收盤）、"
                     f"{n_c:,} 格起算月×收盤時間（照課本做，成本 {COST * 100:.2f}%）、同樣 {n_c:,} 格照一般人方式做（10 小時後才做、成本 "
                     f"{P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），以及其他幣（{'、'.join(r['others'])}）："
                     "其他幣只在「比特幣的鏈上訊號」說好時持有（F1＋均線用該幣自己的 50 日均線），當成看大盤的濾網。前四項是報酬贏過同一天起"
                     "一直抱著的比例（後面是每年多賺的中位數，百分點）；排名分數＝四項裡最低的；最深虧損＝1,728 格的中位數；"
                     "熊市避開＝2015 年以後走完的大熊市（跌 50% 以上）避開跌幅的中位數；被洗＝下車後 30 天內用更高價買回，每年次數。"
                     f"安慰劑＝把 z 分數或出塊數整串循環平移 {PLACEBO} 次再算排名分數，一樣好或更好的比例。"),
         "columns": ["規則", "比特幣起算月：贏的", f"{n_c:,} 格照課本：贏的｜多賺", "一般人做法：贏的｜多賺", "其他幣起算月：贏的",
                     "排名分數（名次）｜最深虧損中位數", "熊市避開｜每年被洗", "安慰劑：打亂後一樣好的比例"],
         "rows": rows3},
        {"caption": ("四、事先寫下的通過條件逐條對：效果在三段期間都朝來源說的方向（為正）且 2015 起全期 90% 區間不含 0；排名分數不低於 50 日均線；"
                     "最深虧損中位數不比 50 日均線深；安慰劑比例 ≤ 10%。四條都過才算通過。"),
         "columns": ["規則", "效果（全期［90% 區間］；三段）", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "安慰劑比例", "結論"],
         "rows": rows4},
    ]

    # ----- today -----
    st, pos, z, fee, blocks = r["st"], r["pos"], r["z"], r["fee_usd"], r["blocks"]
    t = close.index[-1]
    use = t - DAY

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    zu = z.get(use) if use in z.index else None
    fee30 = fee.rolling(MEAN_DAYS).mean().get(use) if use in fee.index else None
    blk = blocks.get(use) if use in blocks.index else None
    blk30 = blocks.rolling(30).mean().get(use) if use in blocks.index else None
    f1v, f7v = st["fit1"]["f"][-1], st["fit7"]["f"][-1]
    now = [
        {"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"Coin Metrics 最新一天的鏈上資料", "value": _d(r["oc"]["BlkCnt"].dropna().index[-1])},
        {"label": f"規則用的手續費 z 分數（鏈上資料到 {_d(use)}）",
         "value": (f"{zu:+.2f}（30 天平均手續費收入每天約 {fee30:,.0f} 美元）" if _ok(zu) and _ok(fee30) else "—")},
        {"label": f"出塊數（{_d(use)}）", "value": (f"{blk:.0f} 個，比 144 {blk - NORMAL_BLOCKS:+.0f}；30 天平均 {blk30:.1f} 個"
                                                   if _ok(blk) and _ok(blk30) else "—")},
        {"label": "B1 模型的預測（下一天｜下 7 天對數報酬）", "value": f"{_p(f1v)}｜{_p(f7v)}" if _ok(f1v) and _ok(f7v) else "—"},
    ]
    for k in RULES:
        now.append({"label": NAMES[k], "value": word(pos[k][-1])})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    d90 = g["全部"]["diff"][90]
    r1, r7 = oos[1]["全部"]["r2"], oos[7]["全部"]["r2"]
    p4 = th["pr4"]
    first = (f"2015 年起手續費 z 分數為正（區塊空間搶手）時，之後 90 天比特幣平均比 z 分數不為正時多 {_pt(d90[0])} 個百分點"
             f"（90% 區間 {_pt(d90[1])} 到 {_pt(d90[2])}，三段 {'／'.join(_pt(g[p]['diff'][90][0]) for p in pnames)}）。")
    second = (f"出塊數預測的樣本外 R² 明天 {_r2(r1[0])}、7 天 {_r2(r7[0])}，論文自己的表也是負的；照論文做多／放空的練習，"
              f"每年比抱著 {_pt(p4['strat'][0] - p4['hold'][0])} 點（論文 +27）。")
    third = (f"排名分數 F1 {_share(s['f1']['low'])}、F1＋均線 {_share(s['f1ma']['low'])}、B1 {_share(s['b1']['low'])}、"
             f"B1-7 {_share(s['b7']['low'])}，50 日均線 {_share(m['low'])}（安慰劑比例 "
             + "、".join(f"{SHORT[k]} {_share(r['placebo'][k])}" for k in TESTED) + "）；"
             f"事先定的通過條件{'有規則過了：' + '、'.join(SHORT[k] for k in passed) if passed else '沒有一條過'}。")
    verdict = first + second + third

    # ----- notes -----
    cuts, bad = r["look"]
    bear_txt = "；".join(
        f"{_d(close.index[b['top']])} 起" + ("" if b["done"] else "（還在進行，沒算進中位數）") + "："
        + "、".join(f"{SHORT[k]} {_share(r['crash'][k][i]['avoid'])}" for k in RULES)
        for i, b in enumerate(r["bears"])) or "沒有"
    bci = r["bci"]
    notes = [
        ("事先寫下的設計（照原文）：F1 (fee z-score, practitioner rule exactly): z = (fee revenue USD 30-day mean − its "
         "trailing 2-year mean) ÷ trailing 2-year std; hold BTC when z > 0, else cash. Also F1+ma50 (both must agree). "
         "B1 (Sanhueza replication as close as the text allows): expanding-window OLS of next-day (and next-7-day) BTC log "
         "return on block-count deviation (daily BlkCnt − 144, or as the thesis defines), OOS from 2015; hold when forecast "
         "> 0. Tests: forward 30/90-day returns by F1 state and z buckets; OOS R² for B1 (Campbell–Thompson); split "
         "2015–2019, 2020–2022, 2023–now; rules vs holding and ma50 on main_signal_check's fair grid and other coins using "
         "BTC's on-chain signal as a market filter; placebo with circularly shifted signal (200 draws, on the ranking "
         "score); crash-avoided share with whipsaws. Pass test: effect same sign in all three periods with full-sample 90% "
         "interval excluding zero AND a rule's ranking score ≥ ma50's with median deepest loss no worse AND placebo share ≤ 10%."),
        ("論文實際寫的（Sanhueza 2026，第 6–7、14、26、30 頁）：資料 2014-09 到 2025-04、每天一筆；出塊數來自 CoinMetrics，定義只寫"
         "「一段期間挖出的區塊數」，沒有減 144 或取對數（係數印成 0.00，是原始數量）。迴歸是報酬對「當天」的預測變數做遞迴（擴張視窗）最小平方法，"
         "和隨機漫步加漂移比；出塊數的樣本外 R²（P/R = 1）是 1／7／14／28 天 −0.02%／−0.29%／−0.19%／−0.08%，全部為負。"
         "「每年多 27%、夏普多 0.38」來自表 4 的 Anatolyev–Gerko 規則：預測為正做多、為負「放空」，P/R = 4 時出塊數 0.58（夏普 0.83）"
         "對抱著 0.31（0.45）；這個規則沒有寫扣成本（0.01% 的成本寫在另一個只做多、權重 0 到 1 的資產配置練習裡，那裡出塊數不突出）。"
         "本研究事先定的是減 144（有截距時和原始數量的預測完全一樣）、只做多或拿現金、扣成本、用前一天的出塊數。"
         f"第二表照論文做的練習，同期間一直抱著算出來是每年 {_p(th['pr4']['hold'][0])}（夏普 {th['pr4']['hold'][1]:.2f}）和 "
         f"{_p(th['pr1']['hold'][0])}（{th['pr1']['hold'][1]:.2f}），論文寫 0.31（0.45）和 0.45（0.70），價格來源、無風險利率或切點"
         "可能不同，所以只能對個大概。論文用「當天」的出塊數：UTC 一天結束時那天的出塊數剛好數完，理論上收盤當下就知道，但資料商要幾小時後才公布；"
         f"晚一天才用時，差距從 {_pt(th['pr4']['strat'][0] - th['pr4']['hold'][0])} 點變成 "
         f"{_pt(th['pr4_lag']['strat'][0] - th['pr4_lag']['hold'][0])} 點。"),
        ("文章實際寫的（CoinDesk 2023-03-23，Omkar Godbole）：引用 Glassnode 的「兩年 z 分數」——和兩年平均手續費收入差幾個標準差，"
         "說它在多頭時通常為正、空頭時為負，「高手續費回來」在過去和大多頭的開頭同時出現，當時是 2021 年中以來第一次轉正。"
         "文章沒有說是用每日值還是平均過的值，也沒有提到 2022 年 11 月的假訊號（委託說明裡的這兩點在原文找不到）；這裡照事先定的用 30 天平均。"),
        (f"通過條件：{'、'.join(SHORT[k] for k in passed) + ' 通過' if passed else '沒有一條規則通過'}。效果本身："
         + "；".join(f"{eff[k][0]} {'過' if eff_ok[k] else '沒過'}" for k in eff)
         + f"。安慰劑用每個收盤時間 {PL_SIMS} 次一般人模擬重算排名分數（真的規則也用同樣方法算：" + "、".join(
             f"{SHORT[k]} {_share(r['real20'][k])}" for k in TESTED) + "），打亂後排名分數的中位數：" + "、".join(
             f"{SHORT[k]} {_share(r['placebo_med'][k])}" for k in TESTED) + "。"),
        ("不偷看未來：鏈上某一天的資料要等那天（UTC）結束後才有，所以收盤當天的決定只用到前一天的鏈上資料；B1 每天重估時，"
         "只用到報酬已經實現的那些天。換收盤時間時，落在 UTC 日期 E 的收盤只用「E−1 那天 UTC 0 點收盤時決定的狀態」（鏈上資料到 E−2）。"
         f"檢查：把 {cuts} 個切點之後的價格和鏈上資料隨機打亂再算一次，切點以前的狀態"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 處變了，這次結果不可信。")
         + " 但 Coin Metrics 的歷史值會被修正（價格、手續費偶爾回頭改），這裡用的是今天看到的版本，不一定和當時即時看到的一樣。"),
        (f"資料：手續費收入（美元）＝ Coin Metrics 社群版的 FeeTotNtv（以比特幣計的手續費）× PriceUSD（FeeTotUSD 要金鑰）。"
         + (f"對照用 blockchain.info 的每日手續費（美元）算同樣的 z 分數，2015 起 {bci['n']:,} 天裡兩者正負相同的占 {_share(bci['agree'])}。"
            if bci else "blockchain.info 對照這次沒算成。")
         + " z 分數要 30 天平均再加滿兩年（730 天）視窗，所以 2012-08 才有值。"),
        (f"和 50 日均線持有狀態以外的比較：2015 起持有的天數比例 " + "、".join(f"{SHORT[k]} {_share(r['held'][k])}" for k in RULES)
         + f"。2015 年以後的大熊市避開的跌幅（算進中途被洗）：{bear_txt}。大熊市只有寥寥幾次，差幾個百分點不代表真的比較好。"),
        ("其他幣用的是比特幣的鏈上訊號（看大盤的濾網），不是那些幣自己的鏈上資料；MSTR 是股票，收盤在 UTC 當天結束前，所以用前一天 UTC 收盤時"
         "決定的比特幣狀態。其他幣是今天還在的大幣和一檔股票，有倖存者偏誤。50 日均線的數字用和 main_signal_check 一樣的設定與亂數，應該和那份研究相同。"),
        ("相鄰起算日共用大部分的歷史，24 個收盤時間只差幾個小時，這些「幾個裡幾個」不是互相獨立的證據。報酬天天重疊、會跨過期間的邊界，"
         "區間只是大概。三個效果、四條規則一起看，多重比較會讓「碰巧過關」的機會變大。"),
        ("手續費和出塊數會被和價格無關的事改變：2017 年的區塊擁擠、隔離見證與批次付款、2023 年起的 Ordinals 銘文和符文、算力突然進出"
         "（出塊數會暫時偏離 144，直到每兩週調整一次難度）。"),
        "過去的結果不保證未來會一樣。這裡只比較區塊空間需求和價格過去的關係、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "blockspace",
        "title": "區塊空間需求（手續費、出塊數）和比特幣",
        "question": ("大家搶著付手續費上鏈（區塊空間需求高）的時候，之後比特幣是不是比較強？業者說的「手續費收入 z 分數為正跟著漲勢」、"
                     "博士論文說的「出塊數預測有樣本外交易獲利」照原樣測結果如何？能不能贏過 50 日均線和隨機對照？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Sanhueza (2026), 博士論文, Universidad Diego Portales", "url": THESIS},
            {"name": "CoinDesk (2023-03-23), Rising Bitcoin Miner Revenue From Fees Suggests Onset of Major Bull Run",
             "url": ARTICLE},
            {"name": "Coin Metrics 社群版 API（BlkCnt、FeeTotNtv、PriceUSD，免金鑰）",
             "url": CM_URL + "?assets=btc&metrics=BlkCnt,FeeTotNtv,PriceUSD&frequency=1d"},
            {"name": "blockchain.info 每日手續費（美元，對照）",
             "url": BCI_FEES + "?timespan=all&format=json&sampled=false"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
