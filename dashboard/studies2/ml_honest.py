"""Historical study "ml_honest": if a model may combine every feature the dashboard has, with a strictly honest
walk-forward, does it predict bitcoin's next-week direction better than the plain trend rule?

Features (all known at the close of day t, every one put on a -1..+1 scale using trailing data only):
* price: the 17 trend rules of trend_ensemble (long = +1, not long = -1) and the blend share; returns over 1/7/30/90
  days and 30-day volatility, each ranked against its own history up to day t (at least 365 days);
* Binance funding (7-day mean of the daily mean, from 2019-09) and the alternative.me fear & greed index (from 2018),
  ranked against their own history up to day t (at least 180 days);
* "all data" set only: the dashboard's cached indicator histories with at least 700 daily points (indicator_scan's
  list and missing-data handling), used one day late, ranked against their own history up to that day (at least
  180 days). Exchange flows (rewritten after the fact) and series starting after 2023 are left out.
A feature that does not exist yet on a day is 0 (the middle of its scale).

Target: sign of the next 7-day close-to-close return (30 days as a check). Training uses every day; scoring uses
non-overlapping samples only (one every 7 / 30 days), so the statistics are not inflated by overlap.

Models: L2 logistic regression (numpy, Newton steps) and gradient-boosted stumps (numpy, fixed bins, half the
features tried per round) - scikit-learn is not installed in this venv. Baselines: "up if the blend share > 0.5"
and "always up". Walk-forward: expanding window, retrained at the start of every quarter, training samples must have
their whole target window end before the test quarter (purge) plus a further 7-day embargo. Hyper-parameters (L2
strength, number of boosting rounds) are chosen by an inner time-series CV inside the training window only (three
91-day validation blocks at its end, with the same purge and embargo).

No look-ahead: a feature on day t uses data up to day t (cached series: up to day t-1); a model used in a quarter
never saw a label that ends inside or after that quarter's start minus the embargo; positions decided at the close of
day t are held on day t+1. Nothing here is a forecast or advice.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import signals as sig_mod
from dashboard.studies2 import indicator_scan as scan
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import walkforward as wf

FAPI = "https://fapi.binance.com"
COST = 0.001
COST_HIGH = 0.003
YEAR = 365
H = 7                      # main target: next 7 days
H_CHECK = 30
EMBARGO = 7
TRAIN_START = "2014-01-01"
ETH_TRAIN_START = "2018-09-01"
TEST_START = "2020-01-01"
ALL_TEST_START = "2024-01-01"
ANCHOR = pd.Timestamp("2020-01-01")  # weekly scoring days: every 7th day from here
MIN_TRAIL_PRICE = 365
MIN_TRAIL = 180
FILL = 3
LATE = "2023-01-01"        # cached series starting after this are left out
EXCLUDE = {"btc_exchange_netflow_7d": "交易所資金流：資料來源事後整批改寫過（flow_filters 研究查到），不能當成當時就知道的數字",
           "btc_exchange_balance": "交易所餘額：和交易所資金流同一個來源，同樣事後被改寫過"}
LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
ROUNDS = (25, 50, 100, 200)
LR = 0.05
NBINS = 16
EDGES = np.linspace(-1, 1, NBINS + 1)[1:-1]
INNER_BLOCKS = 3
INNER_DAYS = 91
NULLS = 200
PERMS = 20
SEED = 20200101
DASH = "—"
GROUP_CN = {"trend": "17 條順勢規則和混合比例", "moves": "過去漲跌和波動", "funding": "資金費率（幣安，2019 起）",
            "fng": "恐懼與貪婪指數（2018 起）", **{k: f"儀表板指標：{v}" for k, v in scan.GROUP_CN.items()}}


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return DASH
    v = float(v)
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return DASH
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _per100(v) -> str:
    return f"100 次裡 {round(float(v) * 100)} 次" if _ok(v) else DASH


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _funding(symbol: str, since: str) -> pd.Series | None:
    """Binance perpetual funding, mean of each finished UTC day's settlements, in % per 8 hours."""
    try:
        out, start = [], int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
        for _ in range(30):
            page = common.get_json(FAPI + "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1000, "startTime": start})
            if not page:
                break
            out += page
            start = page[-1]["fundingTime"] + 1
            if len(page) < 1000:
                break
        if len(out) < 1000:
            return None
        series = pd.Series([float(p["fundingRate"]) * 100 for p in out],
                           index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
        series = series[~series.index.duplicated()].sort_index()
        daily = series.resample("1D").mean()
        return daily[daily.index < pd.Timestamp(common.today_utc())]
    except Exception:  # noqa: BLE001 - blocked or down: the feature is left out and the notes say so
        return None


def _cached() -> tuple[list[dict], list[tuple[str, str]]]:
    items, dropped = scan._indicators()
    kept = []
    for item in items:
        if item["id"] in EXCLUDE:
            dropped.append((item["name"], EXCLUDE[item["id"]]))
        elif item["series"].index[0] > pd.Timestamp(LATE):
            dropped.append((item["name"], f"{item['series'].index[0].year} 年才開始有資料，太晚"))
        else:
            kept.append(item)
    return kept, dropped


def _load() -> dict:
    data = {"btc": te._daily_close(), "fund_btc": _funding("BTCUSDT", "2019-09-01"), "problems": []}
    try:
        data["fng"] = wf._fear_greed()
    except Exception:  # noqa: BLE001
        data["fng"] = None
        data["problems"].append("恐懼與貪婪指數這次取不到，沒有放進模型。")
    if data["fund_btc"] is None:
        data["problems"].append("比特幣資金費率這次取不到，沒有放進模型。")
    data["cache"], data["dropped"] = _cached()
    try:
        data["eth"] = ta._crypto_close("ETHUSDT")
    except Exception:  # noqa: BLE001
        data["eth"] = None
        data["problems"].append("以太幣日線這次取不到，第三步沒有做。")
    data["fund_eth"] = _funding("ETHUSDT", "2019-11-01") if data["eth"] is not None else None
    return data


# ---------- features ----------

def _rank(series: pd.Series, index: pd.DatetimeIndex, min_trail: int) -> pd.Series:
    """Each day's value ranked among its own values up to that day, mapped to -1..+1; NaN before min_trail days."""
    s = series.dropna()
    s = s[np.isfinite(s)]
    r = s.expanding(min_periods=min_trail).rank(pct=True)
    return (2 * r - 1).reindex(index).ffill(limit=FILL)


def _price_features(close: pd.Series) -> tuple[pd.DataFrame, dict]:
    signal, _ = te._signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    cols, group = {}, {}
    for c in signal.columns:
        cols[f"rule_{c}"], group[f"rule_{c}"] = 2 * signal[c] - 1, "trend"
    cols["blend"], group["blend"] = 2 * share - 1, "trend"
    for n in (1, 7, 30, 90):
        cols[f"ret{n}"], group[f"ret{n}"] = _rank(close / close.shift(n) - 1, close.index, MIN_TRAIL_PRICE), "moves"
    vol = close.pct_change().rolling(30, min_periods=30).std(ddof=1)
    cols["vol30"], group["vol30"] = _rank(vol, close.index, MIN_TRAIL_PRICE), "moves"
    return pd.DataFrame(cols, index=close.index), group


def _features(close: pd.Series, funding: pd.Series | None, fng: pd.Series | None,
              cache: list[dict] | None) -> tuple[pd.DataFrame, dict, pd.Series]:
    frame, group = _price_features(close)
    if funding is not None:
        avg7 = funding.rolling(7, min_periods=7).mean()
        frame["funding7"], group["funding7"] = _rank(avg7, close.index, MIN_TRAIL), "funding"
    if fng is not None:
        frame["fng"], group["fng"] = _rank(fng, close.index, MIN_TRAIL), "fng"
    for item in cache or []:
        late = item["series"].shift(1, freq="D")  # used one day late: many are published after the day ends
        key = "x_" + item["id"]
        frame[key], group[key] = _rank(late, close.index, MIN_TRAIL), item["group"]
    share = (frame["blend"] + 1) / 2
    return frame, group, share


def _target(close: pd.Series, h: int) -> pd.Series:
    fwd = close.shift(-h) / close - 1
    return (fwd > 0).astype(float).where(fwd.notna())


# ---------- models (numpy; scikit-learn is not installed) ----------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _logit_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """L2 logistic regression (intercept not penalised) by Newton steps."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    pen = np.full(A.shape[1], lam)
    pen[0] = 0.0
    base = min(max(y.mean(), 0.02), 0.98)
    w[0] = math.log(base / (1 - base))
    for _ in range(25):
        p = _sigmoid(A @ w)
        grad = A.T @ (p - y) + pen * w
        hess = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen + 1e-8)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def _logit_pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(w[0] + X @ w[1:])


def _bins(X: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES, X, side="right").astype(np.int64)


def _boost_fit(X: np.ndarray, y: np.ndarray, rounds: int, seed: int = SEED, lam: float = 1.0,
               min_h: float = 5.0) -> dict:
    """Gradient-boosted stumps on fixed bins of the -1..+1 features (no data-dependent binning)."""
    Xb = _bins(X)
    n, p = Xb.shape
    onehot = np.zeros((n, p * NBINS), dtype=np.float32)
    onehot[np.repeat(np.arange(n), p), (Xb + np.arange(p) * NBINS).ravel()] = 1.0
    base = min(max(y.mean(), 0.02), 0.98)
    f0 = math.log(base / (1 - base))
    F = np.full(n, f0)
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(rounds):
        pr = _sigmoid(F)
        both = np.vstack([y - pr, pr * (1 - pr)]).astype(np.float32) @ onehot
        G, Hs = both[0].reshape(p, NBINS).astype(float), both[1].reshape(p, NBINS).astype(float)
        GL, HL = np.cumsum(G, axis=1)[:, :-1], np.cumsum(Hs, axis=1)[:, :-1]
        GT, HT = G.sum(axis=1, keepdims=True), Hs.sum(axis=1, keepdims=True)
        GR, HR = GT - GL, HT - HL
        gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - GT ** 2 / (HT + lam)
        gain[(HL < min_h) | (HR < min_h)] = -np.inf
        gain[rng.random(p) >= 0.5] = -np.inf
        if not np.isfinite(gain).any():
            continue
        j, k = np.unravel_index(int(np.argmax(gain)), gain.shape)
        vl, vr = LR * GL[j, k] / (HL[j, k] + lam), LR * GR[j, k] / (HR[j, k] + lam)
        F += np.where(Xb[:, j] <= k, vl, vr)
        trees.append((int(j), int(k), float(vl), float(vr)))
    return {"f0": f0, "trees": trees}


def _boost_staged(model: dict, X: np.ndarray, stops: tuple) -> list[np.ndarray]:
    Xb = _bins(X)
    F = np.full(len(X), model["f0"])
    out, done = [], 0
    for stop in stops:
        for j, k, vl, vr in model["trees"][done:stop]:
            F += np.where(Xb[:, j] <= k, vl, vr)
        done = stop
        out.append(_sigmoid(F))
    return out


def _boost_pred(model: dict, X: np.ndarray) -> np.ndarray:
    return _boost_staged(model, X, (len(model["trees"]),))[0]


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Logit:
    grid = LAMBDAS

    @staticmethod
    def scores(Xt, yt, Xv, yv):
        return [_logloss(_logit_pred(_logit_fit(Xt, yt, lam), Xv), yv) for lam in LAMBDAS]

    @staticmethod
    def fit(X, y, h):
        return _logit_fit(X, y, h)

    @staticmethod
    def pred(m, X):
        return _logit_pred(m, X)


class Boost:
    grid = ROUNDS

    @staticmethod
    def scores(Xt, yt, Xv, yv):
        model = _boost_fit(Xt, yt, max(ROUNDS))
        return [_logloss(p, yv) for p in _boost_staged(model, Xv, ROUNDS)]

    @staticmethod
    def fit(X, y, h):
        return _boost_fit(X, y, h)

    @staticmethod
    def pred(m, X):
        return _boost_pred(m, X)


# ---------- walk-forward ----------

def _score_days(index: pd.DatetimeIndex, h: int) -> np.ndarray:
    step = 7 if h <= 7 else h
    return ((index - ANCHOR).days % step) == 0


def _folds(index: pd.DatetimeIndex, test_start: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(pd.Timestamp(test_start), index[-1], freq="QS")
    return [(s, (starts[i + 1] if i + 1 < len(starts) else index[-1] + pd.Timedelta(days=1)))
            for i, s in enumerate(starts)]


def _walk(X: pd.DataFrame, y: pd.Series, train_start: str, test_start: str, h: int, model) -> dict:
    """Expanding-window walk-forward with purge and embargo; inner time-series CV picks the hyper-parameter."""
    index = X.index
    values = X.values
    yv = y.values
    known = ~np.isnan(yv)
    scoring = _score_days(index, h)
    begin = index >= pd.Timestamp(train_start)
    prob = pd.Series(np.nan, index=index)
    folds = []
    for start, end in _folds(index, test_start):
        cut = start - pd.Timedelta(days=h + 1 + EMBARGO)  # last training day: target ends before start, plus embargo
        tr = begin & known & (index <= cut)
        if tr.sum() < 200:
            continue
        # inner CV inside the training window only
        losses = []
        last_tr = index[tr][-1]
        for b in range(INNER_BLOCKS, 0, -1):
            v0 = last_tr - pd.Timedelta(days=b * INNER_DAYS - 1)
            v1 = v0 + pd.Timedelta(days=INNER_DAYS)
            itr = tr & (index <= v0 - pd.Timedelta(days=h + 1 + EMBARGO))
            iva = tr & scoring & (index >= v0) & (index < v1)
            if itr.sum() < 150 or iva.sum() < 5 or len(set(yv[itr])) < 2:
                continue
            losses.append(model.scores(values[itr], yv[itr], values[iva], yv[iva]))
        pick = model.grid[int(np.argmin(np.mean(losses, axis=0)))] if losses else model.grid[len(model.grid) // 2]
        fitted = model.fit(values[tr], yv[tr], pick)
        te_rows = (index >= start) & (index < end)
        prob[te_rows] = model.pred(fitted, values[te_rows])
        folds.append({"start": start, "end": end, "train": tr, "test": te_rows, "hyper": pick, "model": fitted,
                      "base": float(yv[tr].mean())})
    return {"prob": prob, "folds": folds}


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    ranks = pd.Series(p).rank().values
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _metrics(p: np.ndarray, y: np.ndarray, constant: bool = False) -> dict:
    return {"n": int(len(y)), "acc": float(((p > 0.5) == (y == 1)).mean()) if len(y) else None,
            "auc": None if constant else _auc(p, y), "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
            "up": float(y.mean()) if len(y) else None}


def _scored(run: dict, y: pd.Series, h: int, since: str) -> np.ndarray:
    index = y.index
    mask = _score_days(index, h) & y.notna().values & run["prob"].notna().values & (index >= pd.Timestamp(since))
    return mask


def _baselines(run: dict, y: pd.Series, share: pd.Series, mask: np.ndarray) -> dict:
    yv = y.values[mask]
    blend_p = share.values[mask]
    base = pd.Series(np.nan, index=y.index)
    for f in run["folds"]:
        base[f["test"]] = f["base"]
    up = base.values[mask]
    return {"blend": _metrics(np.nan_to_num(blend_p, nan=0.5), yv),
            "up": {**_metrics(np.ones_like(yv), yv, constant=True), "brier": float(np.mean((up - yv) ** 2))}}


def _null(X: pd.DataFrame, y: pd.Series, run: dict, mask: np.ndarray, sims: int) -> list[float]:
    """Walk-forward AUC when the training labels are shuffled (same folds, same L2 strength per fold)."""
    rng = np.random.default_rng(SEED)
    values, yv = X.values, y.values
    parts = [(values[f["train"]], yv[f["train"]], f["hyper"], f["test"] & mask) for f in run["folds"]]
    target = yv[mask]
    out = []
    for _ in range(sims):
        prob = np.full(len(yv), np.nan)
        for Xt, yt, lam, rows in parts:
            if rows.any():
                prob[rows] = _logit_pred(_logit_fit(Xt, rng.permutation(yt), lam), values[rows])
        out.append(_auc(prob[mask], target))
    return [a for a in out if a is not None]


def _importance(X: pd.DataFrame, y: pd.Series, run: dict, model, mask: np.ndarray, group: dict) -> dict:
    """Drop in test AUC when one group of features is shuffled across the test samples (test folds only)."""
    rng = np.random.default_rng(SEED + 1)
    values = X.values
    rows = np.where(mask)[0]
    fold_of = np.full(len(values), -1)
    for i, f in enumerate(run["folds"]):
        fold_of[f["test"]] = i
    target = y.values[rows]

    def score(mat: np.ndarray) -> float:
        p = np.empty(len(rows))
        for i, f in enumerate(run["folds"]):
            sel = fold_of[rows] == i
            if sel.any():
                p[sel] = model.pred(f["model"], mat[sel])
        return _auc(p, target) or 0.5

    sub = values[rows]
    real = score(sub)
    out = {}
    for g in dict.fromkeys(group.values()):
        cols = [i for i, c in enumerate(X.columns) if group[c] == g]
        drops = []
        for _ in range(PERMS):
            mat = sub.copy()
            mat[:, cols] = sub[rng.permutation(len(sub))][:, cols]
            drops.append(real - score(mat))
        out[g] = {"drop": float(np.mean(drops)), "n": len(cols)}
    return out


def _deciles(p: np.ndarray, y: np.ndarray) -> list[dict]:
    order = np.argsort(p, kind="stable")
    return [{"p": float(p[part].mean()), "up": float(y[part].mean()), "n": int(len(part))}
            for part in np.array_split(order, 10)]


# ---------- trading ----------

def _trade(close: pd.Series, want: pd.Series, since: str, cost: float = COST, delay: int = 1) -> dict:
    change = close.pct_change()
    pos = want.shift(delay)
    net, trade = te._run(pos.values, change.values, cost)
    rows = close.index >= pd.Timestamp(since)
    net, trade = pd.Series(net, index=close.index)[rows], pd.Series(trade, index=close.index)[rows]
    return te._measure(net, pos.fillna(0.0)[rows], trade)


def _hold(close: pd.Series, since: str) -> dict:
    change = close.pct_change().fillna(0.0)
    rows = close.index >= pd.Timestamp(since)
    return te._measure(change[rows], pd.Series(1.0, index=close.index)[rows], pd.Series(0.0, index=close.index)[rows])


def _rule(prob: pd.Series, at: float) -> pd.Series:
    return (prob > at).astype(float).where(prob.notna())


# ---------- the computation ----------

def _asset(close, funding, fng, cache, train_start, nulls, check=True, test_start=TEST_START) -> dict:
    X, group, share = _features(close, funding, fng, cache)
    X = X.fillna(0.0)
    y = _target(close, H)
    out = {"X": X, "group": group, "share": share, "y": y}
    for name, model in (("logit", Logit), ("boost", Boost)):
        run = _walk(X, y, train_start, test_start, H, model)
        out[name] = run
    mask = _scored(out["logit"], y, H, test_start)
    out["mask"] = mask
    for name, model in (("logit", Logit), ("boost", Boost)):
        out[name]["m"] = _metrics(out[name]["prob"].values[mask], y.values[mask])
        out[name]["imp"] = _importance(X, y, out[name], model, mask, group)
    out.update(_baselines(out["logit"], y, share, mask))
    out["null"] = _null(X, y, out["logit"], mask, nulls) if nulls else []
    if check:
        y30 = _target(close, H_CHECK)
        out["check"] = {}
        for name, model in (("logit", Logit), ("boost", Boost)):
            run = _walk(X, y30, train_start, TEST_START, H_CHECK, model)
            m30 = _scored(run, y30, H_CHECK, TEST_START)
            out["check"][name] = _metrics(run["prob"].values[m30], y30.values[m30])
            out["check"]["mask"] = m30
        out["check"].update(_baselines(run, y30, share, out["check"]["mask"]))
    return out


def _compute(data: dict, nulls: int = NULLS) -> dict:
    close = data["btc"]
    res = {"close": close}
    res["long"] = _asset(close, data["fund_btc"], data["fng"], None, TRAIN_START, nulls)
    res["all"] = _asset(close, data["fund_btc"], data["fng"], data["cache"], TRAIN_START, 0, check=False,
                        test_start=ALL_TEST_START)
    # the "all data" model is scored from ALL_TEST_START, when the cached series have enough history to be ranked
    a, lo = res["all"], res["long"]
    a_mask = _scored(a["logit"], a["y"], H, ALL_TEST_START)
    a["mask2"] = a_mask
    for name, model in (("logit", Logit), ("boost", Boost)):
        a[name]["m2"] = _metrics(a[name]["prob"].values[a_mask], a["y"].values[a_mask])
        a[name]["imp2"] = _importance(a["X"], a["y"], a[name], model, a_mask, a["group"])
        lo[name]["m2"] = _metrics(lo[name]["prob"].values[a_mask], a["y"].values[a_mask])
    a.update({f"{k}2": v for k, v in _baselines(a["logit"], a["y"], a["share"], a_mask).items()})
    a["null2"] = _null(a["X"], a["y"], a["logit"], a_mask, nulls) if nulls else []

    blend = res["long"]["share"]
    b5520 = sig_mod._breakout(close, 55, 20)
    trades = {}
    for since, key in ((TEST_START, "main"), (ALL_TEST_START, "recent")):
        t = {"hold": _hold(close, since), "blend": _trade(close, blend, since), "b5520": _trade(close, b5520, since)}
        for setname in ("long", "all") if key == "recent" else ("long",):
            for name in ("logit", "boost"):
                for at in (0.5, 0.55):
                    t[f"{setname}_{name}_{at}"] = _trade(close, _rule(res[setname][name]["prob"], at), since)
        trades[key] = t
    stress = {}
    for name in ("logit", "boost"):
        want = _rule(res["long"][name]["prob"], 0.5)
        stress[f"{name}_cost"] = _trade(close, want, TEST_START, cost=COST_HIGH)
        stress[f"{name}_delay"] = _trade(close, want, TEST_START, delay=2)
    stress["blend_cost"] = _trade(close, blend, TEST_START, cost=COST_HIGH)
    stress["blend_delay"] = _trade(close, blend, TEST_START, delay=2)
    res["trades"], res["stress"] = trades, stress

    if data.get("eth") is not None:
        eth = data["eth"]
        e = _asset(eth, data.get("fund_eth"), None, None, ETH_TRAIN_START, 0, check=False)
        e["trades"] = {"hold": _hold(eth, TEST_START), "blend": _trade(eth, e["share"], TEST_START),
                       "logit": _trade(eth, _rule(e["logit"]["prob"], 0.5), TEST_START),
                       "boost": _trade(eth, _rule(e["boost"]["prob"], 0.5), TEST_START)}
        e["close"] = eth
        res["eth"] = e
    return res


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, lo, al = r["close"], r["long"], r["all"]
    last = close.index[-1]
    eth = r.get("eth")

    def mrow(model: str, setname: str, period: str, m: dict, null_text: str = DASH) -> list:
        return [model, setname, period, f"{m['n']} 次", _per100(m["acc"]), _num(m["auc"]), _num(m["brier"], 3),
                null_text]

    def null_text(real, sims) -> str:
        if not sims or not _ok(real):
            return DASH
        above = sum(1 for a in sims if a >= real)
        return f"{len(sims)} 次洗牌裡 {above} 次 ≥ 真的"

    p_main = f"{TEST_START[:4]}–{last.year}"
    p_recent = f"{ALL_TEST_START[:4]}–{last.year}"
    long_name = "價格＋資金費率＋恐懼貪婪"
    all_name = f"再加儀表板 {len(data['cache'])} 個指標"
    rows1 = [
        mrow("邏輯迴歸", long_name, p_main + "，下週", lo["logit"]["m"], null_text(lo["logit"]["m"]["auc"], lo["null"])),
        mrow("樹模型（梯度提升）", long_name, p_main + "，下週", lo["boost"]["m"]),
        mrow("簡單：混合過半就猜漲", "17 條順勢規則", p_main + "，下週", lo["blend"]),
        mrow("永遠猜漲", "—", p_main + "，下週", lo["up"]),
        mrow("邏輯迴歸", all_name, p_recent + "，下週", al["logit"]["m2"], null_text(al["logit"]["m2"]["auc"], al["null2"])),
        mrow("樹模型（梯度提升）", all_name, p_recent + "，下週", al["boost"]["m2"]),
        mrow("邏輯迴歸", long_name, p_recent + "，下週", lo["logit"]["m2"]),
        mrow("樹模型（梯度提升）", long_name, p_recent + "，下週", lo["boost"]["m2"]),
        mrow("簡單：混合過半就猜漲", "17 條順勢規則", p_recent + "，下週", al["blend2"]),
        mrow("永遠猜漲", "—", p_recent + "，下週", al["up2"]),
        mrow("邏輯迴歸", long_name, p_main + "，30 天後", lo["check"]["logit"]),
        mrow("樹模型（梯度提升）", long_name, p_main + "，30 天後", lo["check"]["boost"]),
        mrow("簡單：混合過半就猜漲", "17 條順勢規則", p_main + "，30 天後", lo["check"]["blend"]),
    ]
    if eth:
        rows1 += [mrow("邏輯迴歸", "以太幣：價格＋資金費率", p_main + "，下週", eth["logit"]["m"]),
                  mrow("樹模型（梯度提升）", "以太幣：價格＋資金費率", p_main + "，下週", eth["boost"]["m"]),
                  mrow("簡單：混合過半就猜漲", "以太幣 17 條順勢規則", p_main + "，下週", eth["blend"])]

    # deciles
    mask = lo["mask"]
    yv = lo["y"].values[mask]
    dl = _deciles(lo["logit"]["prob"].values[mask], yv)
    db = _deciles(lo["boost"]["prob"].values[mask], yv)
    rows2 = [[f"第 {i + 1} 組" + ("（最看跌）" if i == 0 else "（最看漲）" if i == 9 else ""),
              f"{a['p'] * 100:.0f}%", _per100(a["up"]), f"{a['n']} 次", f"{b['p'] * 100:.0f}%", _per100(b["up"]),
              f"{b['n']} 次"] for i, (a, b) in enumerate(zip(dl, db))]

    def trow(name: str, period: str, m: dict) -> list:
        return [name, period, _pct(m["ret"]), _pct(m["deepest"]), _num(m["ratio"]), f"{m['share']:.0f}%",
                f"{m['turnover']:.1f} 次"]

    tm, tr, st = r["trades"]["main"], r["trades"]["recent"], r["stress"]
    rows3 = [
        trow("一直抱著", p_main, tm["hold"]),
        trow("17 條順勢規則混合", p_main, tm["blend"]),
        trow("55 日新高進、20 日新低出", p_main, tm["b5520"]),
        trow("邏輯迴歸：機率 > 50% 才持有", p_main, tm["long_logit_0.5"]),
        trow("邏輯迴歸：機率 > 55% 才持有", p_main, tm["long_logit_0.55"]),
        trow("樹模型：機率 > 50% 才持有", p_main, tm["long_boost_0.5"]),
        trow("樹模型：機率 > 55% 才持有", p_main, tm["long_boost_0.55"]),
        trow("壓力：邏輯迴歸 > 50%，成本 0.30%", p_main, st["logit_cost"]),
        trow("壓力：邏輯迴歸 > 50%，晚一天做", p_main, st["logit_delay"]),
        trow("壓力：樹模型 > 50%，成本 0.30%", p_main, st["boost_cost"]),
        trow("壓力：混合，成本 0.30%", p_main, st["blend_cost"]),
        trow("一直抱著", p_recent, tr["hold"]),
        trow("17 條順勢規則混合", p_recent, tr["blend"]),
        trow("55 日新高進、20 日新低出", p_recent, tr["b5520"]),
        trow(f"邏輯迴歸（{all_name}）> 50%", p_recent, tr["all_logit_0.5"]),
        trow(f"樹模型（{all_name}）> 50%", p_recent, tr["all_boost_0.5"]),
    ]

    groups = list(dict.fromkeys(list(lo["group"].values()) + list(al["group"].values())))
    rows4 = []
    for g in groups:
        a1, b1 = lo["logit"]["imp"].get(g), lo["boost"]["imp"].get(g)
        a2, b2 = al["logit"]["imp2"].get(g), al["boost"]["imp2"].get(g)
        n = (a2 or a1)["n"]

        def drop(x):
            return _num(x["drop"], 3) if x else DASH
        rows4.append([GROUP_CN.get(g, g), f"{n} 個", drop(a1), drop(b1), drop(a2), drop(b2)])
    rows4.sort(key=lambda row: -max([float(v) for v in row[2:] if v != DASH] or [0]))
    rows4 = rows4[:16]

    # ----- today -----
    def current(setname: str, name: str) -> float | None:
        p = r[setname][name]["prob"]
        return float(p.iloc[-1]) if _ok(p.iloc[-1]) else None

    def state(p) -> str:
        if not _ok(p):
            return DASH
        a = "持有" if p > 0.5 else "不持有"
        b = "持有" if p > 0.55 else "不持有"
        return f"猜下週漲的機率 {p * 100:.0f}%；> 50% 規則：{a}；> 55% 規則：{b}"

    share_now = float(lo["share"].iloc[-1])
    lf = lo["logit"]["folds"][-1]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"邏輯迴歸（{long_name}）", "value": state(current("long", "logit"))},
        {"label": f"樹模型（{long_name}）", "value": state(current("long", "boost"))},
        {"label": f"邏輯迴歸（{all_name}）", "value": state(current("all", "logit"))},
        {"label": f"樹模型（{all_name}）", "value": state(current("all", "boost"))},
        {"label": "17 條順勢規則裡看多的比例", "value": f"{share_now * 100:.0f}%（簡單規則：{'過半，猜漲' if share_now > 0.5 else '沒過半，猜跌'}）"},
        {"label": "現在用的模型是哪一天訓練的", "value": f"{_d(lf['start'])}（每季重新訓練一次，只用 {_d(lf['start'] - pd.Timedelta(days=H + 1 + EMBARGO))} 以前的結果）"},
    ]

    # ----- verdict -----
    ml, mb, bl = lo["logit"]["m"], lo["boost"]["m"], lo["blend"]
    nl = lo["null"]
    above = sum(1 for a in nl if a >= (ml["auc"] or 0.5)) if nl else None
    names = {"long_logit_0.5": "邏輯迴歸 >50%", "long_logit_0.55": "邏輯迴歸 >55%", "long_boost_0.5": "樹模型 >50%",
             "long_boost_0.55": "樹模型 >55%"}
    best_model = max(names, key=lambda k: tm[k]["ratio"])
    best, bt = tm[best_model], tm["blend"]
    top = sorted(((g, (lo["logit"]["imp"][g]["drop"] + lo["boost"]["imp"][g]["drop"]) / 2) for g in lo["logit"]["imp"]),
                 key=lambda x: -x[1])
    useful = [GROUP_CN.get(g, g) for g, v in top[:2] if v > 0.01]
    top_text = (f"打亂後模型變差最多的是{'、'.join(useful)}" if useful
                else "沒有哪一類資料打亂後讓模型明顯變差，等於模型沒找到站得住的規律")
    a2l, a2b = al["logit"]["m2"], al["boost"]["m2"]
    win = "贏過" if best["ratio"] > bt["ratio"] + 0.03 else ("差不多" if abs(best["ratio"] - bt["ratio"]) <= 0.03 else "輸給")
    verdict = (
        f"每季重訓、不偷看未來，{p_main} 每週猜比特幣下週漲跌：邏輯迴歸猜對 {_per100(ml['acc'])}（AUC {_num(ml['auc'])}，"
        f"答案打亂重做 200 次有 {above} 次不輸它）、樹模型 {_per100(mb['acc'])}，永遠猜漲是 {_per100(lo['up']['acc'])}；"
        f"{p_recent} 加進儀表板 {len(data['cache'])} 個指標也只有 {_per100(a2l['acc'])}和 {_per100(a2b['acc'])}。"
        f"拿來交易扣 0.10% 成本，四種門檻裡最好的{names[best_model]} 每年 {_pct(best['ret'])}、最深虧 {_pct(best['deepest'])}、"
        f"報酬÷波動 {_num(best['ratio'])}，{win} 17 條順勢規則混合的 {_pct(bt['ret'])}、{_pct(bt['deepest'])}、{_num(bt['ratio'])}。"
        f"{top_text}。"
    )

    eth_text = ""
    if eth:
        et = eth["trades"]
        eth_text = (f"以太幣（第三步，只用價格和幣安以太幣資金費率，{p_main}）：邏輯迴歸 > 50% 每年 {_pct(et['logit']['ret'])}、"
                    f"最深虧 {_pct(et['logit']['deepest'])}、報酬÷波動 {_num(et['logit']['ratio'])}；樹模型 {_pct(et['boost']['ret'])}、"
                    f"{_pct(et['boost']['deepest'])}、{_num(et['boost']['ratio'])}；以太幣 17 條規則混合 {_pct(et['blend']['ret'])}、"
                    f"{_pct(et['blend']['deepest'])}、{_num(et['blend']['ratio'])}；一直抱著 {_pct(et['hold']['ret'])}、"
                    f"{_pct(et['hold']['deepest'])}、{_num(et['hold']['ratio'])}。")

    hyp_l = pd.Series([f["hyper"] for f in lo["logit"]["folds"]]).value_counts().to_dict()
    hyp_b = pd.Series([f["hyper"] for f in lo["boost"]["folds"]]).value_counts().to_dict()
    dropped = {}
    for name, why in data["dropped"]:
        dropped.setdefault(why, []).append(name)
    excluded = "；".join(f"{why}：{'、'.join(n)}" for why, n in dropped.items())
    notes = [
        "背景：很多論文和公開程式碼說機器學習猜比特幣漲跌能準到 55%–65%，但整理這些研究的文章（Jaquart 等人 2021；Baquero 2026）"
        "發現多數是因為偷看到未來、只切一次訓練和測試、沒算成本，而且簡單模型常常贏過複雜模型；一個事先公開做法的 GitHub 專案"
        "（yann-dev09/bitcoin-signal-lab）用一季一季往前測的邏輯迴歸，AUC 只有 0.455–0.48，沒有優勢。這裡用儀表板全部的資料再做一次。",
        "機器學習＝讓電腦從過去的資料自己找規律；不偷看未來＝訓練時完全用不到之後的資料。這裡的做法：每一季開始時，只拿"
        f"「目標期間已經結束、而且再往前多留 {EMBARGO} 天空檔」的日子重新訓練，然後拿這個模型去猜接下來一整季，季末再重來"
        "（訓練資料一路累積，不丟掉舊的）。模型要調的設定（邏輯迴歸的收斂程度、樹模型要幾棵樹），也只在訓練資料最後三段 91 天裡"
        "用同樣「不偷看」的方式挑。",
        "評分只用不重疊的樣本：猜下週的每 7 天算一次、猜 30 天的每 30 天算一次，否則同一段行情會被重複計算、讓數字看起來比實際可靠。"
        "AUC 是「把會漲和會跌的週分開的能力」，0.5 等於亂猜、1 是完美；Brier 分數是猜的機率和結果差多少，越小越好，"
        "一直猜 50% 大約是 0.25。簡單規則的 Brier 是直接把「看多的比例」當成機率算的（例如 17 條全看多就是 100%），所以特別大；「永遠猜漲」的 Brier 用的是訓練期間上漲的比例。",
        "洗牌對照：把每一季訓練資料裡的漲跌答案隨機打亂、其他一切照舊，重做 200 次（只做邏輯迴歸，樹模型太慢）。"
        "如果真的模型的 AUC 落在洗牌結果的中間，就表示它找到的規律和亂猜分不開。",
        "特徵（模型拿來判斷的資料）：17 條順勢規則各自看多看空、看多的比例；過去 1、7、30、90 天的漲跌和 30 天波動；"
        "幣安資金費率 7 天平均；恐懼與貪婪指數；全部指標版本再加上儀表板存下的指標。每一個都只和自己「到那一天為止」的歷史比、"
        f"換成 -1 到 +1 之間的位置（價格類至少先有 {MIN_TRAIL_PRICE} 天、其他至少 {MIN_TRAIL} 天），還沒有資料的日子當作正中間。"
        f"儀表板存下的指標一律晚一天才用（很多是當天結束後才公布），週末假日缺的值最多補 {FILL} 天。",
        f"兩個模型都是自己用 numpy 寫的（這個環境沒有 scikit-learn）：邏輯迴歸加 L2 收斂（候選強度 {', '.join(f'{x:g}' for x in LAMBDAS)}），"
        f"實際各季挑到的次數 {hyp_l}；樹模型是梯度提升的「單一切點小樹」（每輪只看隨機一半的特徵、學習率 {LR}、候選棵數 "
        f"{', '.join(map(str, ROUNDS))}），各季挑到的次數 {hyp_b}。樹只切一刀，抓不到兩個指標同時成立才有用的情況。",
        f"儀表板存下的指標大多只有約 4 年（{common.HISTORY_POINTS} 天上限，多數從 2022-10 起），所以「全部指標」只能從 "
        f"{ALL_TEST_START[:4]} 年起評分，樣本只有 {al['logit']['m2']['n']} 週，結論比 {TEST_START[:4]} 年起的版本更不可靠；"
        f"它在 {ALL_TEST_START[:4]} 年以前的訓練資料裡，這些指標都當作正中間。",
        "有些存下的資料事後會被資料來源改寫（交易所資金流已證實被整批改寫，所以排除）；鏈上指標多半也是資料來源用現在的算法"
        "回頭算出來的歷史，當時看到的數字可能不完全一樣，這會讓「全部指標」版本看起來比實際做得到的好。",
        f"沒有放進模型的儀表板指標：{excluded}。",
        f"重要性的算法：只在測試期間，把某一類資料在各週之間隨機打亂（{PERMS} 次平均），看 AUC 掉多少；掉越多表示模型越依賴它。"
        "數字是負的或接近 0 表示打亂了也沒差。這只說明模型用了什麼，不代表那類資料真的能預測。",
        f"交易規則：每天收盤用當季的模型算出「下週會漲的機率」，超過 50%（或 55%）隔天就持有比特幣，否則拿現金；每次買賣扣 0.10%，"
        f"現金沒有利息。壓力測試把成本改成 0.30%，或晚一天才照做。混合規則每天依看多比例持有幾成，每天調回比例的小額買賣也算成本。",
        eth_text or "以太幣這次沒有做。",
        "這個研究做了很多選擇（特徵怎麼換算、每季重訓、空檔 7 天、兩種模型、候選設定、門檻 50%／55%），雖然都是事先定好、"
        "沒有回頭改，但換別的選擇結果可能不同。測試期間只有幾年，大致就是兩輪多空，週數看起來多，獨立的行情其實沒有幾段。",
        *data["problems"],
        "過去的結果不保證未來會一樣。這裡只是整理歷史，不是預測，也不是任何買賣建議。",
    ]

    tables = [
        {"caption": "猜漲跌的成績（只算不重疊的樣本、全部都是當時不偷看未來的猜法）。「猜對」是猜的方向和實際一樣的比例；"
                    "AUC 0.5 等於亂猜；Brier 越小越好（一直猜 50% 約 0.25）；最後一欄是把訓練答案打亂重做 200 次的對照。",
         "columns": ["模型", "用了哪些資料", "測試期間與目標", "樣本", "猜對", "AUC", "Brier", "洗牌對照"],
         "rows": rows1},
        {"caption": f"模型越有把握，是不是越準？（比特幣，{long_name}，{p_main}，每週一次）把所有週依模型給的「會漲機率」由低到高"
                    "分成 10 組，看每組實際上漲的比例。如果模型有用，越下面的組應該越常上漲。",
         "columns": ["分組", "邏輯迴歸平均機率", "實際上漲", "樣本", "樹模型平均機率", "實際上漲", "樣本"],
         "rows": rows2},
        {"caption": "拿來交易（比特幣，扣成本後；當天收盤決定、隔天才照做；一直抱著沒有扣成本）。「平均持有」是持有比特幣的日子"
                    "或比例；「一年換手」1 次＝一年買賣總額等於全部資金一次。",
         "columns": ["做法", "期間", "每年報酬", "最深曾經虧多少", "報酬÷波動", "平均持有", "一年換手"],
         "rows": rows3},
        {"caption": "哪類資料最有用（只在測試期間打亂某一類資料後，AUC 掉多少；越大越依賴，接近 0 或負的＝沒差）",
         "columns": ["資料類別", "特徵數", f"邏輯迴歸 {p_main}", f"樹模型 {p_main}", f"邏輯迴歸 全部指標 {p_recent}",
                     f"樹模型 全部指標 {p_recent}"],
         "rows": rows4},
    ]
    return {
        "id": "ml_honest",
        "title": "讓電腦自己找規律，能不能猜中下週漲跌",
        "question": "把儀表板所有資料（價格趨勢、波動、資金費率、鏈上、資金流、情緒、總體經濟）都交給模型，"
                    "用嚴格不偷看未來的方式一季一季往前測，它猜比特幣下週漲跌，會不會比簡單的順勢規則更準、拿來交易更好？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2017 年以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線（BTCUSDT、ETHUSDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安永續合約資金費率歷史", "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "alternative.me 恐懼與貪婪指數", "url": "https://alternative.me/crypto/fear-and-greed-index/"},
            {"name": "事先公開做法的機器學習測試：yann-dev09/bitcoin-signal-lab", "url": "https://github.com/yann-dev09/bitcoin-signal-lab"},
            {"name": "Jaquart, Dann, Weinhardt (2021) Short-term bitcoin market prediction via machine learning",
             "url": "https://doi.org/10.1016/j.jfds.2021.03.001"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
