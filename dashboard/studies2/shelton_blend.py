"""Historical study "shelton_blend": a replication of Shelton (2024), "Bitcoin Return Prediction: Is It Possible via
Stock-to-Flow, Metcalfe's Law, Technical Analysis, or Market Sentiment?", Journal of Risk and Financial Management
17(10):443, doi:10.3390/jrfm17100443.

What the paper says (abstract, MDPI / RePEc): stock-to-flow and Metcalfe's law explain bitcoin returns in-sample but
have limited to no out-of-sample power; technical and sentiment measures are unrelated in-sample and poor
out-of-sample; yet BLENDING the out-of-sample univariate predictions gives a bitcoin / US-dollar-cash timing strategy
that beats buy-and-hold (CAPM alpha up almost 1,300 basis points, higher Sharpe and Sortino, much smaller deepest
loss). A secondary review (arXiv 2606.00071) lists the design as 2014-2024, monthly horizon, expanding window,
out-of-sample R^2 against the historical mean.

The full text could NOT be read for this replication: the named open-access mirror did not resolve (DNS failure),
mdpi.com answers 403 to our honest user agent, and nothing else public carried the text. So the exact predictor
definitions, the blend weights, the trading rule and the reported Sharpe numbers are unknown here. Everything below
was fixed BEFORE running, choosing the simplest reading of the abstract:

* Data: Coin Metrics community (key-less) daily PriceUSD, CapMrktCurUSD, SplyCur, AdrActCnt; alternative.me fear &
  greed (from 2018-02). Monthly: a period ends on the last calendar day of the month; every predictor uses values
  dated the day before (Coin Metrics publishes about one day late), the position changes at that day's close.
* Predictors (one univariate forecast each):
  - s2f: ln(S2F model price / price), model = exp(-1.84) x SF^3.36 (PlanB 2019), SF = supply / supply added in the
    last 365 days. (The regression re-fits level and slope, so only the 3.36 / price mix is fixed.)
  - metcalfe: ln(active addresses^2 / market cap), active addresses = 30-day mean.
  - four Neely-Rapach-Tu-Zhou (2014) style monthly technical signals: ma9 / ma12 = 1 if the month-end price is at or
    above its 9 / 12-month average; mom9 / mom12 = 1 if it is at or above its level 9 / 12 months earlier.
  - fng: 30-day mean of the fear & greed index.
* Target: next month's log return. Estimation sample starts with the 2014-01 predictor values; expanding window
  OLS, at least 24 observations; out-of-sample forecasts from 2018-01 (first four years for estimation; fng only from
  2020-03 when it has 24 months). Benchmark: historical mean of the same expanding sample (Campbell-Thompson R^2).
* Blend: equal-weight mean of the univariate forecasts available that month (seven when all exist).
* Rule: hold bitcoin next month when the blend > 0, else cash at 0% (no T-bill interest). Costs 0.10% / 0.30% of
  the amount traded.
* Paper sample: return months up to 2024-06 (the paper appeared 2024-10; the exact end is not known here). After:
  2024-07 to the last finished month = new data the paper never saw.
* Comparison: 50-day average (close above its 50-day mean) decided on the same monthly schedule with the same
  one-day-late data, and the daily version (decide each daily close, hold the next day) aggregated to months.
* Pass test (stated before running): the blend beats buy-and-hold on return / volatility in the paper's own sample
  AND after the paper's sample it beats buy-and-hold AND matches or beats the monthly 50-day rule on return /
  volatility (0.10% cost) AND its out-of-sample R^2 after the paper is above 0.

Checks that were NOT pre-registered are labelled as such (blend > historical mean instead of > 0, one blend per
predictor family, paper end 2023-12, month boundaries moved to the 7th / 14th / 21st).
Nothing here is advice; the "now" rows only state where the rule stands.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_PAGE = "https://charts.coinmetrics.io/crypto-data/"
FNG_URL = "https://api.alternative.me/fng/"
FNG_PAGE = "https://alternative.me/crypto/fear-and-greed-index/"
PAPER_URL = "https://doi.org/10.3390/jrfm17100443"
MIRROR_URL = "https://p.gyb.elementfx.com/1911-8074/17/10/443"
REPEC_URL = "https://ideas.repec.org/a/gam/jjrfmx/v17y2024i10p443-d1490211.html"
REVIEW_URL = "https://arxiv.org/html/2606.00071v1"

EST_START = "2014-01"     # first predictor month in the estimation sample
OOS_START = "2018-01"     # first return month forecast out of sample
PAPER_END = "2024-06"     # last return month of the paper's sample (assumed)
ALT_PAPER_END = "2023-12"
MIN_OBS = 24
COSTS = (0.001, 0.003)
PREDS = ("s2f", "metcalfe", "ma9", "ma12", "mom9", "mom12", "fng")
FAMILY = {"s2f": "s2f", "metcalfe": "metcalfe", "ma9": "tech", "ma12": "tech", "mom9": "tech", "mom12": "tech",
          "fng": "fng"}
NAMES = {"s2f": "存量流量比（S2F 模型價 ÷ 價格，取對數）", "metcalfe": "梅特卡夫（活躍地址² ÷ 市值，取對數）",
         "ma9": "技術面：月底價 ≥ 9 個月平均", "ma12": "技術面：月底價 ≥ 12 個月平均",
         "mom9": "技術面：月底價 ≥ 9 個月前", "mom12": "技術面：月底價 ≥ 12 個月前",
         "fng": "情緒：恐懼貪婪指數 30 天平均"}
SHORT = {"s2f": "存量流量比", "metcalfe": "梅特卡夫", "ma9": "9 月均線", "ma12": "12 月均線", "mom9": "9 月動能",
         "mom12": "12 月動能", "fng": "恐懼貪婪"}
STORE_FILE = "shelton_blend_daily.json"


# ---------- small helpers ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, digits: int = 1) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    return "—" if not _ok(v) else f"{float(v):.{digits}f}"


def _signed(v, digits: int = 2) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{digits}f}"


# ---------- data ----------

def _load() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    problems = []
    path = common.STORE / STORE_FILE
    try:
        reply = common.get_json(CM_URL, {"assets": "btc", "metrics": "CapMrktCurUSD,SplyCur,AdrActCnt,PriceUSD",
                                         "frequency": "1d", "start_time": "2010-07-18", "page_size": 10000},
                                timeout=60)
        data = list(reply["data"])
        for _ in range(3):
            if not reply.get("next_page_url"):
                break
            reply = common.get_json(reply["next_page_url"], timeout=60)
            data += reply["data"]
        frame = pd.DataFrame(data)
        frame.index = pd.to_datetime(frame["time"].str[:10])
        frame = frame.drop(columns=["asset", "time"]).apply(pd.to_numeric, errors="coerce").sort_index()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(frame.reset_index(names="day").assign(day=lambda f: f["day"].dt.strftime("%Y-%m-%d"))
                            .to_json(orient="records"), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not path.exists():
            raise
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        frame.index = pd.to_datetime(frame.pop("day"))
        problems.append(f"Coin Metrics 這次取不到（{type(exc).__name__}），改用上次存下的資料。")
    frame = frame.rename(columns={"CapMrktCurUSD": "cap", "SplyCur": "sply", "AdrActCnt": "adr", "PriceUSD": "price"})
    frame = frame.asfreq("D")
    try:
        rows = common.get_json(FNG_URL, {"limit": 0}, timeout=30)["data"]
        fng = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s").normalize(): float(d["value"]) for d in rows})
        fng = fng.sort_index()
        fng = fng[~fng.index.duplicated(keep="last")]
    except Exception as exc:  # noqa: BLE001 - sentiment missing: the blend uses the other six
        fng = pd.Series(dtype=float)
        problems.append(f"恐懼貪婪指數這次取不到（{type(exc).__name__}），混合只用其他六個。")
    return frame, fng, problems


def _daily_features(df: pd.DataFrame, fng: pd.Series) -> pd.DataFrame:
    sply, price = df["sply"], df["price"]
    flow = sply - sply.shift(365)
    sf = (sply / flow.where(flow > 0))
    s2f = -1.84 + 3.36 * np.log(sf) - np.log(price)
    adr = df["adr"].rolling(30, min_periods=25).mean()
    met = 2 * np.log(adr) - np.log(df["cap"])
    f = fng.reindex(df.index).rolling(30, min_periods=20).mean() if len(fng) else pd.Series(np.nan, index=df.index)
    ma50 = (price > price.rolling(50, min_periods=50).mean()).astype(float).where(price.rolling(50).mean().notna())
    return pd.DataFrame({"s2f": s2f, "metcalfe": met, "fng": f, "price": price, "ma50": ma50}, index=df.index)


def _periods(index: pd.DatetimeIndex, phase: int | None) -> list[pd.Timestamp]:
    """Period-end dates: last calendar day of each month (phase None) or day `phase` of each month."""
    days = pd.Series(index, index=index)
    if phase is None:
        ends = [d for d in days.groupby(index.to_period("M")).max() if d.day == d.days_in_month]
    else:
        ends = [d for d in index if d.day == phase]
    return ends


def _monthly(df: pd.DataFrame, fng: pd.Series, phase: int | None = None) -> dict:
    """Monthly predictor table (values from the day before each period end) and next-period log returns."""
    feat = _daily_features(df, fng)
    ends = _periods(df.index, phase)
    last = df.index[-1]
    rows = []
    for e in ends + ([last] if not ends or last > ends[-1] else []):
        info = e - pd.Timedelta(days=1)
        if info not in feat.index:
            continue
        rows.append((e, feat.loc[info], df["price"].get(e, np.nan)))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    m = pd.DataFrame([r[1] for r in rows], index=idx)
    m["trade_price"] = [r[2] for r in rows]
    pf = m["price"]
    for n in (9, 12):
        avg = pf.rolling(n, min_periods=n).mean()
        m[f"ma{n}"] = (pf >= avg).astype(float).where(avg.notna())
        back = pf.shift(n)
        m[f"mom{n}"] = (pf >= back).astype(float).where(back.notna())
    complete = np.array([e != last or (phase is None and e.day == e.days_in_month) or (phase is not None and e.day == phase)
                         for e in idx])
    tp = m["trade_price"].values
    nxt = np.full(len(m), np.nan)
    nxt[:-1] = np.log(tp[1:] / tp[:-1])
    nxt[:-1][~complete[1:]] = np.nan
    m["r_next"] = nxt
    # return month label of r_next = the month of the next period end
    labels = list(idx[1:].to_period("M").astype(str)) + [None]
    m["ret_month"] = labels
    m["month"] = idx.to_period("M").astype(str)
    m["complete"] = complete
    return {"m": m, "daily": feat, "df": df}


# ---------- forecasts ----------

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    b = 0.0 if vx <= 1e-15 else float(((x - xm) * (y - ym)).sum() / vx)
    return float(ym - b * xm), b


def _forecasts(m: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window univariate forecasts of r_next made at each period end (row), plus mean benchmark and blends."""
    start = int(np.argmax((m["month"] >= EST_START).values))
    r = m["r_next"].values
    out = {p: np.full(len(m), np.nan) for p in PREDS}
    bench = np.full(len(m), np.nan)
    for i in range(start, len(m)):
        rm = m["ret_month"].iloc[i]
        if isinstance(rm, str) and rm < OOS_START:
            continue
        past = np.arange(start, i)
        past = past[np.isfinite(r[past])]
        if len(past) < MIN_OBS:
            continue
        bench[i] = r[past].mean()
        for p in PREDS:
            x = m[p].values
            use = past[np.isfinite(x[past])]
            if len(use) < MIN_OBS or not np.isfinite(x[i]):
                continue
            a, b = _ols(x[use], r[use])
            out[p][i] = a + b * x[i]
    f = pd.DataFrame(out, index=m.index)
    f["bench"] = bench
    f["blend"] = f[list(PREDS)].mean(axis=1, skipna=True).where(f[list(PREDS)].notna().any(axis=1))
    fam = pd.DataFrame({k: f[[p for p in PREDS if FAMILY[p] == k]].mean(axis=1, skipna=True)
                        for k in ("s2f", "metcalfe", "tech", "fng")})
    f["family"] = fam.mean(axis=1, skipna=True).where(fam.notna().any(axis=1))
    return f


def _r2(m: pd.DataFrame, f: pd.DataFrame, col: str, lo: str, hi: str | None) -> tuple[float | None, int]:
    rm = m["ret_month"]
    sel = (rm.notna() & (rm >= lo) & ((rm <= hi) if hi else True) & m["r_next"].notna() & f[col].notna()
           & f["bench"].notna()).values
    if sel.sum() < 6:
        return None, int(sel.sum())
    y, fc, b = m["r_next"].values[sel], f[col].values[sel], f["bench"].values[sel]
    den = ((y - b) ** 2).sum()
    return (1 - ((y - fc) ** 2).sum() / den if den > 0 else None), int(sel.sum())


def _insample(m: pd.DataFrame, p: str, hi: str) -> tuple[float | None, float | None, float | None]:
    """OLS of r_next on predictor over the paper sample: slope, Newey-West (3 lags) t, R^2."""
    rm = m["ret_month"]
    sel = ((m["month"] >= EST_START) & rm.notna() & (rm <= hi) & m["r_next"].notna() & m[p].notna()).values
    if sel.sum() < MIN_OBS:
        return None, None, None
    x, y = m[p].values[sel], m["r_next"].values[sel]
    a, b = _ols(x, y)
    e = y - a - b * x
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    if sxx <= 1e-15:
        return None, None, None
    u = xc * e
    s = (u ** 2).sum()
    for lag in range(1, 4):
        s += 2 * (1 - lag / 4) * (u[lag:] * u[:-lag]).sum()
    t = b / math.sqrt(s / sxx ** 2) if s > 0 else None
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, t, r2


# ---------- strategies ----------

def _stats(ret: np.ndarray, w: np.ndarray | None = None, switches: int | None = None) -> dict:
    ret = np.asarray(ret, dtype=float)
    n = len(ret)
    if n < 3:
        return {}
    eq = np.cumprod(1 + ret)
    ann = eq[-1] ** (12 / n) - 1
    vol = float(np.std(ret, ddof=1) * math.sqrt(12))
    deep = float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min())
    return {"n": n, "ann": float(ann), "vol": vol, "ratio": float(ann / vol) if vol > 1e-12 else None, "deep": deep,
            "held": float(np.mean(w)) if w is not None else 1.0, "switches": switches}


def _timing(m: pd.DataFrame, signal: pd.Series, lo: str, hi: str | None, cost: float) -> dict:
    """Hold bitcoin over the return month after period end i when signal[i] == 1; cost on every change."""
    rm = m["ret_month"]
    sel = (rm.notna() & (rm >= lo) & ((rm <= hi) if hi else True) & m["r_next"].notna()).values
    idx = np.where(sel)[0]
    if len(idx) < 3:
        return {}
    w = signal.values.astype(float)
    w = np.where(np.isfinite(w), w, 0.0)
    simple = np.exp(m["r_next"].values) - 1
    prev = w[idx[0] - 1] if idx[0] > 0 else 0.0
    out, sw = [], 0
    for i in idx:
        trade = abs(w[i] - prev)
        sw += int(trade > 0)
        out.append(w[i] * simple[i] - cost * trade)
        prev = w[i]
    return _stats(np.array(out), w[idx], sw)


def _hold(m: pd.DataFrame, lo: str, hi: str | None) -> dict:
    rm = m["ret_month"]
    sel = (rm.notna() & (rm >= lo) & ((rm <= hi) if hi else True) & m["r_next"].notna()).values
    return _stats(np.exp(m["r_next"].values[sel]) - 1)


def _ma50_daily(data: dict, lo: str, hi: str | None, cost: float) -> dict:
    """Daily 50-day rule (decide at each daily close with the day's price, hold the next day), months compounded."""
    m, df = data["m"], data["df"]
    price = df["price"]
    sig = (price > price.rolling(50, min_periods=50).mean()).astype(float).where(price.rolling(50).mean().notna())
    chg = price.pct_change()
    w = sig.shift(1).fillna(0.0)
    net = w * chg - cost * w.diff().abs().fillna(0.0)
    rm = m["ret_month"]
    sel = np.where((rm.notna() & (rm >= lo) & ((rm <= hi) if hi else True) & m["r_next"].notna()).values)[0]
    if len(sel) < 3:
        return {}
    ends = m.index
    months, held, sw = [], [], 0
    for i in sel:
        a, b = ends[i], ends[i + 1]
        part = net[(net.index > a) & (net.index <= b)]
        months.append(float(np.prod(1 + part.fillna(0.0).values) - 1))
        ww = w[(w.index > a) & (w.index <= b)]
        held.append(float(ww.mean()))
        sw += int((ww.diff().abs() > 0).sum())
    return _stats(np.array(months), np.array(held), sw)


def _run_all(data: dict, f: pd.DataFrame, paper_end: str = PAPER_END) -> dict:
    m = data["m"]
    after = (pd.Period(paper_end, "M") + 1).strftime("%Y-%m")
    blend_sig = (f["blend"] > 0).astype(float).where(f["blend"].notna())
    mean_sig = (f["blend"] > f["bench"]).astype(float).where(f["blend"].notna())
    fam_sig = (f["family"] > 0).astype(float).where(f["family"].notna())
    ma_sig = m["ma50"]
    res = {}
    for key, lo, hi in (("paper", OOS_START, paper_end), ("after", after, None)):
        x = {"hold": _hold(m, lo, hi)}
        for c in COSTS:
            x[f"blend_{c}"] = _timing(m, blend_sig, lo, hi, c)
            x[f"ma50m_{c}"] = _timing(m, ma_sig, lo, hi, c)
            x[f"ma50d_{c}"] = _ma50_daily(data, lo, hi, c)
        x["blend_mean"] = _timing(m, mean_sig, lo, hi, COSTS[0])
        x["family"] = _timing(m, fam_sig, lo, hi, COSTS[0])
        x["r2"] = {p: _r2(m, f, p, lo, hi) for p in PREDS + ("blend", "family")}
        res[key] = x
    res["after_label"] = after
    return res


def _pass(res: dict) -> dict:
    p, a = res["paper"], res["after"]
    c = COSTS[0]

    def better(x: dict, y: dict) -> bool | None:
        if not x or not y or not _ok(x.get("ratio")) or not _ok(y.get("ratio")):
            return None
        return x["ratio"] > y["ratio"]

    def atleast(x: dict, y: dict) -> bool | None:
        if not x or not y or not _ok(x.get("ratio")) or not _ok(y.get("ratio")):
            return None
        return x["ratio"] >= y["ratio"] - 1e-12

    r2 = a["r2"]["blend"][0]
    checks = {"paper_dir": better(p[f"blend_{c}"], p["hold"]), "after_hold": better(a[f"blend_{c}"], a["hold"]),
              "after_ma50": atleast(a[f"blend_{c}"], a[f"ma50m_{c}"]), "after_r2": (r2 > 0) if _ok(r2) else None}
    checks["all"] = all(v is True for v in checks.values())
    return checks


# ---------- the study ----------

def _compute(df: pd.DataFrame, fng: pd.Series, phase: int | None = None) -> dict:
    data = _monthly(df, fng, phase)
    f = _forecasts(data["m"])
    return {"data": data, "f": f, "res": _run_all(data, f)}


def _yes(v) -> str:
    return "—（資料不足）" if v is None else ("是" if v else "否")


def study() -> dict:
    t0 = time.time()
    df, fng, problems = _load()
    main = _compute(df, fng)
    data, f, res = main["data"], main["f"], main["res"]
    m = data["m"]
    chk = _pass(res)
    alt = _run_all(data, f, ALT_PAPER_END)
    phases = {}
    for ph in (7, 14, 21):
        try:
            phases[ph] = _compute(df, fng, ph)["res"]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"把月底改成每月 {ph} 號的檢查算不出來（{type(exc).__name__}）。")
    p, a, after = res["paper"], res["after"], res["after_label"]
    done = m[m["r_next"].notna()]
    last_ret_month = done["ret_month"].iloc[-1]
    n_paper, n_after = p["hold"].get("n", 0), a["hold"].get("n", 0)

    # table 1: predictors
    rows1 = []
    for k in PREDS + ("blend",):
        if k == "blend":
            name, ins = "混合：七個預測的平均（論文的做法，等權）", ("—", "—")
        else:
            b, t, r2 = _insample(m, k, PAPER_END)
            name = NAMES[k]
            ins = (f"{'正' if _ok(b) and b > 0 else '負' if _ok(b) else '—'}｜{_signed(t)}", _pct(r2))
        r2p, np_ = p["r2"][k]
        r2a, na_ = a["r2"][k]
        cur = f[k].iloc[-1]
        rows1.append([name, ins[0], ins[1], f"{_pct(r2p)}（{np_} 個月）", f"{_pct(r2a)}（{na_} 個月）",
                      "—" if not _ok(cur) else f"{(math.exp(cur) - 1) * 100:+.1f}%"])

    # table 2: strategies
    def srow(period: str, name: str, s: dict) -> list:
        if not s:
            return [period, name, "—", "—", "—", "—", "—", "—"]
        sw = "—" if s.get("switches") is None else f"{s['switches']} 次"
        return [period, name, _pct(s["ann"]), _pct(s["vol"]), _num(s["ratio"]), _pct(s["deep"]),
                _pct(s["held"], 0), sw]

    rows2 = []
    for key, label in (("paper", f"論文期間 {OOS_START}～{PAPER_END}（{n_paper} 個月）"),
                       ("after", f"論文之後 {after}～{last_ret_month}（{n_after} 個月）")):
        x = res[key]
        rows2 += [srow(label, "混合預測 > 0 才持有（成本 0.10%）", x["blend_0.001"]),
                  srow(label, "混合預測 > 0 才持有（成本 0.30%）", x["blend_0.003"]),
                  srow(label, "50 日均線，每月底決定（0.10%）", x["ma50m_0.001"]),
                  srow(label, "50 日均線，每天決定（0.10%）", x["ma50d_0.001"]),
                  srow(label, "50 日均線，每天決定（0.30%）", x["ma50d_0.003"]),
                  srow(label, "一直抱著", x["hold"])]

    # table 3: pass test
    c = "0.001"
    rows3 = [
        ["論文期間：混合策略的報酬÷波動高於一直抱著（論文的方向）",
         f"{_num(p['blend_' + c].get('ratio'))} 對 {_num(p['hold'].get('ratio'))}", _yes(chk["paper_dir"])],
        ["論文之後：混合策略的報酬÷波動高於一直抱著",
         f"{_num(a['blend_' + c].get('ratio'))} 對 {_num(a['hold'].get('ratio'))}", _yes(chk["after_hold"])],
        ["論文之後：混合策略的報酬÷波動不輸每月底決定的 50 日均線",
         f"{_num(a['blend_' + c].get('ratio'))} 對 {_num(a['ma50m_' + c].get('ratio'))}", _yes(chk["after_ma50"])],
        ["論文之後：混合預測的樣本外 R² 大於 0", _pct(a["r2"]["blend"][0]), _yes(chk["after_r2"])],
        ["四項全部通過", "—", "通過" if chk["all"] else "沒通過"],
    ]

    # table 4: checks not pre-registered
    def r4(name: str, xp: dict, xa: dict, hp: dict, ha: dict) -> list:
        return [name, _num(xp.get("ratio")) if xp else "—", _num(hp.get("ratio")) if hp else "—",
                _num(xa.get("ratio")) if xa else "—", _num(ha.get("ratio")) if ha else "—"]

    rows4 = [r4("原設定：混合 > 0（月底）", p["blend_0.001"], a["blend_0.001"], p["hold"], a["hold"]),
             r4("改成：混合 > 歷史平均才持有", p["blend_mean"], a["blend_mean"], p["hold"], a["hold"]),
             r4("改成：四類各一票（技術面四個先平均）", p["family"], a["family"], p["hold"], a["hold"]),
             r4(f"改成：論文期間只到 {ALT_PAPER_END}", alt["paper"]["blend_0.001"], alt["after"]["blend_0.001"],
                alt["paper"]["hold"], alt["after"]["hold"])]
    for ph, rr in phases.items():
        rows4.append(r4(f"改成：每月 {ph} 號決定（混合 > 0）", rr["paper"]["blend_0.001"], rr["after"]["blend_0.001"],
                        rr["paper"]["hold"], rr["after"]["hold"]))
        rows4.append(r4(f"對照：每月 {ph} 號決定的 50 日均線", rr["paper"]["ma50m_0.001"], rr["after"]["ma50m_0.001"],
                        rr["paper"]["hold"], rr["after"]["hold"]))

    tables = [
        {"caption": (f"四類預測因子，各自單獨預測「下個月的對數報酬」。樣本內 = {EST_START} 到 {PAPER_END} 全部資料一次回歸"
                     "（斜率方向｜t 值用 Newey-West 3 期修正，絕對值約 2 以上才算明顯）。樣本外 = 用模型沒看過的新資料來測："
                     f"每個月只用當時以前的資料重估一次，{OOS_START} 起逐月預測。樣本外 R² 和「歷史平均報酬」這個最笨的預測比，"
                     "大於 0 才代表比猜平均好。恐懼貪婪指數 2018-02 才有，2020-03 起才有預測，月數較少。最後一欄是用到最新一天的資料"
                     "對下個月的預測（換算成一般報酬率），只是模型的數字，不是建議。"),
         "columns": ["預測因子", "樣本內：斜率方向｜t 值", "樣本內 R²", "樣本外 R²：論文期間", "樣本外 R²：論文之後",
                     "目前對下個月的預測"],
         "rows": rows1},
        {"caption": ("照混合預測進出（預測 > 0 就整個月抱比特幣，否則現金、現金沒有利息），和 50 日均線、一直抱著比。"
                     "報酬÷波動 = 年化報酬 ÷ 年化波動（用每月報酬算），越高越好；最深虧損也用月底資料算，每天決定的 50 日均線"
                     "也是先把每天的結果合成每月再比。成本是每次買賣扣的比例；一直抱著不扣成本。"),
         "columns": ["期間", "做法", "年化報酬", "年化波動", "報酬÷波動", "最深虧損（月資料）", "持有時間比例", "進出次數"],
         "rows": rows2},
        {"caption": "通過標準（跑之前就寫好）：四項都要成立，比較時用成本 0.10%。",
         "columns": ["條件", "數字", "成立嗎"],
         "rows": rows3},
        {"caption": ("事後才加的檢查（不是事先定好的，只看結果穩不穩）：每格是報酬÷波動，成本 0.10%。改決定日是把「月底」"
                     "換成每月 7、14、21 號，預測因子和報酬都跟著重算；這是把主訊號研究的「換時間」想法用在月頻率上，"
                     "主訊號那種「起算日×收盤時間」大網格在一個月才換一次、論文之後只有兩年多的規則上做不出有意義的數量。"),
         "columns": ["做法", "論文期間：策略", "論文期間：一直抱著", "論文之後：策略", "論文之後：一直抱著"],
         "rows": rows4},
    ]

    # verdict
    bp, ba, hp_, ha_ = p["blend_0.001"], a["blend_0.001"], p["hold"], a["hold"]
    ma = a["ma50m_0.001"]
    first = (f"重做這篇論文（存量流量、梅特卡夫、技術面、情緒四類預測各自回歸再平均）：在論文自己的期間"
             f"（{OOS_START}～{PAPER_END}）混合策略報酬÷波動 {_num(bp.get('ratio'))}，一直抱著 {_num(hp_.get('ratio'))}，"
             f"{'方向和論文一樣' if chk['paper_dir'] else '沒有重現論文說的優勢'}，"
             f"但混合預測的樣本外 R² 是 {_pct(p['r2']['blend'][0])}。")
    second = (f"論文之後的新資料（{after}～{last_ret_month}，{n_after} 個月）混合策略 {_num(ba.get('ratio'))}、一直抱著 "
              f"{_num(ha_.get('ratio'))}、每月看一次的 50 日均線 {_num(ma.get('ratio'))}，樣本外 R² "
              f"{_pct(a['r2']['blend'][0])}。")
    third = ("事先定的四項標準全部通過。" if chk["all"] else
             "事先定的四項標準沒有全部通過，所以不能說這個混合方法在論文之後還成立或贏過 50 日均線；"
             "另外論文全文這次讀不到，細節是照摘要最簡單的讀法重做。")
    verdict = first + second + third

    # now
    last_day = data["df"].index[-1]
    now = [{"label": f"最新資料日（Coin Metrics，晚一天用）", "value": f"{_d(last_day)}，比特幣 {data['df']['price'].iloc[-1]:,.0f} 美元"}]
    for k in PREDS:
        v = f[k].iloc[-1]
        x = m[k].iloc[-1]
        xs = ("—" if not _ok(x) else (("是" if x >= 0.5 else "否") if k.startswith(("ma", "mom")) else f"{x:.2f}"))
        now.append({"label": f"{SHORT[k]}：目前數值｜對下個月的預測",
                    "value": f"{xs}｜{'—' if not _ok(v) else f'{(math.exp(v) - 1) * 100:+.1f}%'}"})
    bl = f["blend"].iloc[-1]
    now.append({"label": "混合預測（七個平均）", "value": "—" if not _ok(bl) else f"{(math.exp(bl) - 1) * 100:+.2f}%"})
    now.append({"label": "照論文規則的狀態", "value": "—" if not _ok(bl) else ("持有比特幣（混合預測 > 0）" if bl > 0
                                                                       else "現金（混合預測 ≤ 0）")})
    mv = m["ma50"].iloc[-1]
    now.append({"label": "對照：50 日均線（同樣晚一天的資料）", "value": "—" if not _ok(mv) else ("在均線上方：持有" if mv >= 0.5 else "在均線下方：現金")})
    now.append({"label": "這幾格的意思", "value": "只描述模型和規則目前停在哪個狀態；規則每個月底才換一次，這不是買賣建議"})

    notes = [
        "論文全文這次讀不到：作者提供的鏡像網址連不上（找不到網域）、MDPI 對我們誠實標示的連線回 403，其他公開頁面只有摘要。"
        "所以預測因子的精確定義、混合權重、交易規則細節和論文自己的夏普值等數字都沒有拿到，下面的設定是照摘要最簡單的讀法、在跑之前先寫好的。",
        "論文摘要說的結果：存量流量和梅特卡夫在樣本內有解釋力、樣本外幾乎沒有預測力；技術面和情緒樣本內大多無關、樣本外也差；"
        "但把各自的樣本外預測混合起來做比特幣／美元現金的進出，比一直抱著好：CAPM alpha 多將近 1,300 個基點（約每年 13 個百分點）、"
        "夏普和 Sortino 都較高、最深虧損大幅縮小。另一篇整理文章說論文用 2014～2024 年、月資料、擴張視窗、和歷史平均比的樣本外 R²。",
        f"事先寫好的設定：月底前一天的資料（Coin Metrics 通常晚一天）、月底收盤換倉；{EST_START} 起的資料拿來估計，至少 {MIN_OBS} 個月，"
        f"{OOS_START} 起逐月預測下個月的對數報酬；混合 = 當月有的預測直接平均；混合 > 0 就持有一整個月，否則現金（沒有利息）；"
        f"論文期間假設到 {PAPER_END}（論文 2024-10 發表，確切結束月份不知道，表四也看了到 {ALT_PAPER_END} 的情況）。",
        "存量流量用 PlanB 2019 年的式子 exp(−1.84)×比值^3.36 算模型價，再和價格相比取對數；這組係數是 2019 年才公布的，"
        "2018 年的預測用到它算是一點點事後資訊（回歸會重新估斜率，只有 3.36 這個比例是固定的）。梅特卡夫用 30 天平均活躍地址的平方 ÷ 市值。"
        "技術面用學術文獻常見的四個月頻訊號（Neely 等 2014 的做法），是猜論文的用法，不是照論文抄的。",
        "沒看前面的資料：每個月的預測只用當時已經收完的月份估計；恐懼貪婪用 30 天平均；50 日均線每月版也用晚一天的價格。"
        "另外做過打亂測試：把某個日期之後的資料全部隨機打亂，之前的預測和持倉完全不變。"
        "Coin Metrics 的活躍地址等數字有可能事後修正過，社群版拿不到當時的原始版本，這點無法排除。",
        "樣本外 R² 是和「歷史平均報酬」比；比特幣過去平均報酬很高，混合預測又常常整段都是正的，所以策略持有比例很高、和一直抱著差不多。",
        (f"事後才看的變化（表四，不能當成通過的證據）：如果規則改成「混合預測高於歷史平均報酬才持有」，論文期間報酬÷波動 "
         f"{_num(p['blend_mean'].get('ratio'))}（一直抱著 {_num(hp_.get('ratio'))}），論文之後 {_num(a['blend_mean'].get('ratio'))}"
         f"（一直抱著 {_num(ha_.get('ratio'))}、50 日均線 {_num(ma.get('ratio'))}）。論文真正用的交易規則讀不到，"
         "可能比較接近這一種；但這是看過結果才挑的，可能只是運氣；而原設定換決定日（7、14、21 號）的結果也很不穩，這種只換倉幾次的規則對日期很敏感。"),
        "論文用 CAPM alpha、夏普、Sortino；這裡為了和儀表板其他研究一致，只用年化報酬 ÷ 年化波動（現金沒有利息，所以和夏普值接近但不完全一樣）。",
        f"論文之後只有 {n_after} 個月，混合策略一年只換倉幾次，任何結論的誤差都很大；論文期間和之後的月份也只是同一段歷史切兩半。",
        "過去的結果不保證未來會一樣。這裡只是重做一篇論文的方法、看它在新資料上怎麼樣，不是買賣建議。",
        f"這次計算花了 {time.time() - t0:.0f} 秒。",
    ]
    notes += problems

    return {
        "id": "shelton_blend",
        "title": "重做論文：四種預測混合來進出比特幣",
        "question": ("Shelton（2024）說存量流量、梅特卡夫、技術面、情緒單獨都預測不了比特幣下個月的報酬，但把它們的樣本外預測"
                     "混合起來決定持有或現金，風險調整後比一直抱著好。照同樣的方法重做，在論文自己的期間能不能重現？"
                     "論文發表後的新資料還成不成立？有沒有贏過 50 日均線？"),
        "asof": _d(last_day),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Shelton (2024), Journal of Risk and Financial Management 17(10):443（DOI）", "url": PAPER_URL},
            {"name": "同一篇論文的開放鏡像（這次連不上）", "url": MIRROR_URL},
            {"name": "RePEc／IDEAS 論文摘要頁", "url": REPEC_URL},
            {"name": "整理文章（arXiv 2606.00071，提到論文的期間與方法）", "url": REVIEW_URL},
            {"name": "Coin Metrics 社群版 API（PriceUSD、CapMrktCurUSD、SplyCur、AdrActCnt）", "url": CM_PAGE},
            {"name": "alternative.me 恐懼貪婪指數", "url": FNG_PAGE},
        ],
    }


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
