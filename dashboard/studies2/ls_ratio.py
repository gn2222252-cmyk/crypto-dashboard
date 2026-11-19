"""Historical study "ls_ratio": Binance USDT-margined futures publish, every 5 minutes, the share of all accounts
that are long against short ("retail" long/short ratio, counted by accounts) and the long/short ratio of the
positions of the top 20% accounts by margin ("top traders"). Popular reading: crowded retail longs come before
drops, and retail more long than the top traders is bearish. Is that so for bitcoin, and does it help the 50-day
average rule?

Claims as published:
  * K33 Research 2022-02-07 "Binance traders expecting downside": the all-account ratio "surged above 4" before
    several sell-offs; watching it may help to de-risk (contrarian reading of retail positioning).
  * CRYPTINT "Long/Short Ratio" briefing: top traders short while retail is long = retail chasing, whales positioned
    opposite, "often marks local tops"; the reverse often marks local bottoms.
  * Binance / TradingView definitions of the all-account ratio and the top-trader position ratio.
  A GitHub survey found no public backtest.

Pre-registered design (written before any result was seen; not changed afterwards):
Daily values at 00:00 UTC: retail ratio R = count_long_short_ratio (all accounts), top-trader position ratio
T = sum_toptrader_long_short_ratio; divergence D = log(R) - log(T).
1. Forward 7/30-day BTC returns by trailing-1-year quintile of R, T and D (point-in-time cut-offs); slopes with
   block bootstrap; split the sample into two halves by date (state them).
2. Rules on top of ma50 (act at the next close; costs 0.10/0.30%): LS1 ma50 but cash when R is in its trailing-year
   top 20% (crowded retail longs); LS2 ma50 but cash when D is in its trailing-year top 20% (retail more long than
   top traders).
3. Fair grid for whatever starts the data allows (say how few), ordinary-person execution, ETH/SOL/BNB/XRP with
   their own metrics files, placebo with circularly shifted signal (200 draws, ranking score), crash-avoided share
   with whipsaws.
Pass test (stated now): predicted (contrarian) sign in both halves with full-sample 90% interval excluding zero AND
a rule's ranking score >= ma50's with median deepest loss no worse AND placebo share <= 10% AND same sign on >= 3
of 4 other coins. State clearly that the sample is short.
Today: R, T, D and percentiles; rule states - states only.

Details fixed with the design (before results):
  * Data: data.binance.vision daily "metrics" files (BTCUSDT from 2020-09-01, the other four from 2021-12-01). Of
    each file only the row stamped 00:00 UTC is kept (if it is missing, the earliest row up to 00:30). The value of
    day D (stamped D 00:00) is used only from the daily close that ends day D (D+1 00:00 UTC, the candle labelled D)
    - "act at the next close" after it is published; positions are then held from that close (te._run, shift 1).
    A missing day carries the last value forward for at most 3 days, otherwise it is unknown.
  * Point-in-time position: share of the 365 values before day D (at least 330 present) that are lower than the
    value of day D; quintile 1..5 = that share in fifths. "Top 20%" = value above the 80% quantile of the same 365
    prior values. Unknown -> no flag (the rule then equals ma50).
  * Slope = least-squares slope of the forward return on the quintile number (percentage points per quintile).
    Predicted sign: R negative, D negative (contrarian), T positive (follow the top traders). Primary horizon 30
    days. 90% interval: circular blocks of 30 days, 1,000 draws. Halves: the days with a quintile and a 30-day
    forward return, split at the middle day. Other coins: the same slope on their own R / T / D and own forward
    returns (Binance spot daily closes); "same sign" = the predicted sign.
  * Pass = for R with LS1 or for D with LS2: the slope passes (both halves predicted sign, interval excludes zero
    on the predicted side), the rule's ranking score (main_signal_check's lowest of four "beats holding" shares) is
    >= ma50's and its median deepest loss over the start x clock cells is no deeper than ma50's, the placebo share
    is <= 10%, and >= 3 of the 4 other coins show the predicted slope sign.
  * Fair grid: monthly start dates from the first month after the bitcoin flags exist to two years before today,
    UTC daily closes (textbook, 0.10%) and the same start months x 24 closing hours from Binance 1-hour candles,
    textbook and ordinary person (decide at the clock, trade 10 hours later, 0.25%, 10% of entries missed, 200
    simulations per clock; seeds of main_signal_check for ma50). A decision at hour h >= 1 of date D uses the value
    stamped D 00:00, at h = 0 the value stamped D-1 00:00. Fourth check = ETH, SOL, BNB, XRP pooled, each with its
    own flags, monthly starts from the first month after its own flags exist to two years before today.
  * Placebo: R and T of every coin rolled circularly by the same random 180 ... (n - 180) days (200 draws), flags
    and rules rebuilt; the score is the ranking score without the ordinary-person check (that one needs 200
    simulations per clock; dropped for time), compared with the real rule's score built the same way; share of
    draws at least as good.
  * Costs 0.10% and 0.30%: return / volatility of each rule from the first bitcoin flag day. Crashes: crash_timing's
    50%+ bear markets whose top is after the first bitcoin flag day; whipsaws from that day.
  * Look-ahead checks: every price and every R / T after three cut dates scrambled -> no flag or position up to the
    cut may change; one extra day of delay for the signals; the archive's 00:00 rows compared with Binance's live
    30-day API (same definitions).

Data (public, key-less, the dashboard's fixed user agent): data.binance.vision metrics archive, Binance futures
data API (only for the definition check), Binance spot daily and 1-hour candles through the dashboard's helpers.
Descriptive and in-sample, short sample; not advice.
"""
from __future__ import annotations

import io
import json
import math
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from dashboard import common

ARCHIVE = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
FIRST_DAY = {"BTCUSDT": "2020-09-01", "ETHUSDT": "2021-12-01", "SOLUSDT": "2021-12-01", "BNBUSDT": "2021-12-01",
             "XRPUSDT": "2021-12-01"}
DL_PAUSE = 0.05
STORE_VERSION = 2
FIELDS = ("count_long_short_ratio", "sum_toptrader_long_short_ratio", "count_toptrader_long_short_ratio",
          "sum_taker_long_short_vol_ratio", "sum_open_interest")


def _store_path(symbol: str):
    return common.STORE / f"ls_ratio_{symbol.lower()}.json"


def _read_store(symbol: str) -> dict:
    try:
        d = json.loads(_store_path(symbol).read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "rows" not in d:
            return {"rows": {}, "missing": []}
        if d.get("v") != STORE_VERSION:  # an older parser: ask again for the days it could not read
            d["missing"] = []
        return d
    except Exception:  # noqa: BLE001 - first run or damaged file starts over
        return {"rows": {}, "missing": []}


def _write_store(symbol: str, st: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        st = {"v": STORE_VERSION, "rows": dict(sorted(st["rows"].items())), "missing": sorted(set(st["missing"]))}
        _store_path(symbol).write_text(json.dumps(st, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _midnight_row(raw: bytes, day: str) -> list | None:
    """The 00:00 UTC row of one daily file (or the earliest row before 00:30), as [R, T, count_top, taker, OI].
    R must be there; the top-trader columns are empty ("") on many days of 2022 and are then None."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", "replace")
    lines = text.strip().splitlines()
    if not lines:
        return None
    head = [x.strip().strip('"') for x in lines[0].split(",")]
    if "create_time" not in head:
        return None
    col = {name: head.index(name) for name in head}
    best = None
    for line in lines[1:]:
        parts = [x.strip().strip('"') for x in line.split(",")]
        if len(parts) != len(head):
            continue
        t = parts[col["create_time"]]
        if not t.startswith(day):
            continue
        hhmm = t[11:16]
        if hhmm > "00:30":
            continue
        if best is None or hhmm < best[0]:
            try:
                vals = [float(parts[col[f]]) if parts[col[f]] not in ("", "null") else float("nan") for f in FIELDS]
            except (KeyError, ValueError):
                continue
            best = (hhmm, vals)
    if best is None:
        return None
    vals = best[1]
    if not (math.isfinite(vals[0]) and vals[0] > 0):
        return None
    return [round(v, 8) if math.isfinite(v) and v > 0 else None for v in vals] + [best[0]]


def _fetch(symbol: str, budget_s: float | None = None) -> tuple[dict, str]:
    """{YYYY-MM-DD: [R, T, count_top, taker, OI, hh:mm]} of the 00:00 UTC rows, kept in our own store. Only days
    not yet stored are downloaded; a day that was missing more than 7 days ago is not asked for again."""
    st = _read_store(symbol)
    rows, missing = st["rows"], set(st["missing"])
    today = pd.Timestamp(common.today_utc())
    days = pd.date_range(FIRST_DAY[symbol], today - pd.Timedelta(days=1), freq="D")
    recent = today - pd.Timedelta(days=7)
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in rows
            and (d.strftime("%Y-%m-%d") not in missing or d >= recent)]
    t0, problem, n = time.time(), "", 0
    session = requests.Session()
    try:
        for d in todo:
            if budget_s is not None and time.time() - t0 > budget_s:
                problem = f"{symbol} 這次下載超過時間上限，先停在 {d.strftime('%Y-%m-%d')}，下次會接著下載。"
                break
            day = d.strftime("%Y-%m-%d")
            reply = session.get(ARCHIVE.format(s=symbol, d=day), headers=common.HEADERS, timeout=30)
            if reply.status_code == 404:
                missing.add(day)
            else:
                reply.raise_for_status()
                row = _midnight_row(reply.content, day)
                if row is None:
                    missing.add(day)
                else:
                    rows[day] = row
                    missing.discard(day)
            n += 1
            if n % 200 == 0:
                _write_store(symbol, {"rows": rows, "missing": list(missing)})
            time.sleep(DL_PAUSE)
    except Exception as exc:  # noqa: BLE001 - keep what we have
        if not rows:
            raise
        problem = f"{symbol} 這次沒有全部更新成功（{type(exc).__name__}），用的是存下來的資料（到 {max(rows)}）。"
    _write_store(symbol, {"rows": rows, "missing": list(missing)})
    return rows, problem


def _fetch_all(budget_s: float | None = None) -> dict:
    return {s: _fetch(s, budget_s) for s in FIRST_DAY}


# =====================================================================================================================
# analysis
# =====================================================================================================================

from dashboard.studies2 import clock_robustness as cr  # noqa: E402
from dashboard.studies2 import crash_timing as ct  # noqa: E402
from dashboard.studies2 import kimchi_premium as km  # noqa: E402
from dashboard.studies2 import liquidity_lead as ll  # noqa: E402
from dashboard.studies2 import main_signal_check as msc  # noqa: E402
from dashboard.studies2 import robust_trend as rt  # noqa: E402
from dashboard.studies2 import rotation as ro  # noqa: E402
from dashboard.studies2 import trend_assets as ta  # noqa: E402
from dashboard.studies2 import trend_ensemble as te  # noqa: E402

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
COST_HI = 0.003
YEAR = 365
H = (7, 30)
MAIN_H = 30
WIN, WIN_MIN, TOP_Q = 365, 330, 0.80
FFILL = 3
BOOT, DRAWS, BLOCK = 1000, 200, 30
KEYS = ("ma50", "ls1", "ls2")
SEED_KI = {"ma50": 1, "ls1": 70, "ls2": 71}  # ma50 uses main_signal_check's draws
SIGS = ("R", "T", "D")
SIGN = {"R": -1, "T": 1, "D": -1}
SIG_NAME = {"R": "散戶多空比 R（所有帳戶，算人數）", "T": "大戶持倉多空比 T（前 20% 帳戶，算部位）",
            "D": "散戶比大戶偏多的程度 D（log R − log T）"}
NAME = {"ma50": "50 日均線（現行主訊號）", "ls1": "LS1：50 日均線，但散戶多空比在一年最高 20% 時改拿現金",
        "ls2": "LS2：50 日均線，但「散戶比大戶偏多」在一年最高 20% 時改拿現金"}
SHORT = {"ma50": "50 日均線", "ls1": "LS1", "ls2": "LS2"}
OTHER = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
CUTS = ("2022-06-30", "2023-12-31", "2025-06-30")
FAPI = "https://fapi.binance.com"

SOURCES = [
    {"name": "K33 Research 2022-02-07〈Binance traders expecting downside〉：幣安所有帳戶多空比曾在幾次下跌前衝上 4 以上（反向看待散戶部位）",
     "url": "https://k33.com/research/archive/articles/binance-traders-expecting-downside"},
    {"name": "CRYPTINT〈Long/Short Ratio〉：大戶做空、散戶做多＝散戶追價，常是局部高點；反過來常是局部低點",
     "url": "https://cryptint-site.pages.dev/declassified/sentiment/long-short-ratio"},
    {"name": "TradingView 說明：Top trader long/short positions ratio（大戶持倉多空比的定義）",
     "url": "https://www.tradingview.com/support/solutions/43000784532/"},
    {"name": "幣安合約交易數據（多空帳戶比、大戶持倉多空比）",
     "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/trading-data"},
    {"name": "幣安公開歷史資料 data.binance.vision：U 本位合約每日 metrics 檔",
     "url": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/"},
    {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
]

_ok, _num, _d = te._ok, te._num, te._d


def _pct(v, signed: bool = True) -> str:
    """v is a fraction."""
    return te._pct(v * 100, signed) if _ok(v) else "—"


def _pp(v) -> str:
    return f"{v * 100:+.2f}" if _ok(v) else "—"


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


# ---------- series and flags ----------

def _raw(rows: dict, last: pd.Timestamp | None = None) -> pd.DataFrame:
    """R and T by day (stamp D 00:00), daily index from the first stored day, unknown = NaN (no carrying)."""
    keys = sorted(rows)
    idx = pd.to_datetime(keys)
    f = pd.DataFrame([[rows[k][0], np.nan if rows[k][1] is None else rows[k][1]] for k in keys], index=idx,
                     columns=["R", "T"], dtype=float)
    end = max(idx[-1], last) if last is not None else idx[-1]
    return f.reindex(pd.date_range(idx[0], end, freq="D"))


def _signals(raw: pd.DataFrame) -> pd.DataFrame:
    r, t = raw["R"].ffill(limit=FFILL), raw["T"].ffill(limit=FFILL)
    return pd.DataFrame({"R": r, "T": t, "D": np.log(r) - np.log(t)})


def _top(s: pd.Series) -> pd.Series:
    q = s.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(TOP_Q)
    return ((s > q) & q.notna() & s.notna()).astype(float)


def _pos(s: pd.Series) -> pd.Series:
    """Share of the 365 prior values lower than today's (NaN without 330 of them)."""
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    for i in range(WIN_MIN, len(v)):
        if not math.isfinite(v[i]):
            continue
        w = v[max(0, i - WIN):i]
        w = w[np.isfinite(w)]
        if len(w) >= WIN_MIN:
            out[i] = float((w < v[i]).mean())
    return pd.Series(out, index=s.index)


def _flags(sig: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"R": _top(sig["R"]), "D": _top(sig["D"])}, index=sig.index)


def _rules_on(c: np.ndarray, f_r: np.ndarray, f_d: np.ndarray) -> dict:
    s = pd.Series(c)
    m50 = s.rolling(50, min_periods=50).mean()
    ma = (s > m50).astype(float).where(m50.notna()).values
    return {"ma50": ma, "ls1": ma * (1 - f_r), "ls2": ma * (1 - f_d)}


def _targets(close: pd.Series, fl: pd.DataFrame, extra: int = 0) -> pd.DataFrame:
    """Targets after each daily close (label D = close at D+1 00:00, which may use the value stamped D)."""
    f = fl.reindex(close.index).shift(extra).fillna(0.0)
    return pd.DataFrame(_rules_on(close.values.astype(float), f["R"].values, f["D"].values), index=close.index)


# ---------- part 1: quintiles and slopes ----------

def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 60 or np.var(x) < 1e-12:
        return None
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def _boot_slope(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    idx = km._blocks(len(x), BLOCK, BOOT, rng)
    xs, ys = x[idx], y[idx]
    xm, ym = xs.mean(axis=1, keepdims=True), ys.mean(axis=1, keepdims=True)
    var = ((xs - xm) ** 2).mean(axis=1)
    good = var > 1e-12
    b = ((xs - xm) * (ys - ym)).mean(axis=1)[good] / var[good]
    return (float(np.percentile(b, 5)), float(np.percentile(b, 95))) if len(b) > 50 else (None, None)


def _quint(pos: pd.Series) -> pd.Series:
    return (np.floor(pos * 5).clip(0, 4) + 1).where(pos.notna())


def _effects(sig: pd.DataFrame, close: pd.Series, rng, split=None, ci: bool = True) -> dict:
    """Per signal: quintile table, slopes 7/30, 30-day halves and interval."""
    fwd = {h: close.shift(-h) / close - 1 for h in H}
    out = {}
    for k in SIGS:
        q = _quint(_pos(sig[k])).reindex(close.index)
        e = {}
        for h in H:
            f = fwd[h]
            ok = q.notna() & f.notna()
            x, y = q[ok].values.astype(float), f[ok].values.astype(float)
            e[h] = {"slope": _slope(x, y), "n": int(ok.sum()),
                    "rows": [{"days": int((x == i).sum()), "mean": float(y[x == i].mean()) if (x == i).any() else None,
                              "up": float((y[x == i] > 0).mean()) if (x == i).any() else None} for i in range(1, 6)],
                    "all": {"days": len(y), "mean": float(y.mean()) if len(y) else None,
                            "up": float((y > 0).mean()) if len(y) else None}}
            if h == MAIN_H:
                days = q.index[ok]
                e["first"], e["last"] = (days[0], days[-1]) if len(days) else (None, None)
                cut = split if split is not None else (days[len(days) // 2] if len(days) else None)
                e["split"] = cut
                if cut is not None:
                    a = np.asarray(days < cut)
                    e["halves"] = [_slope(x[a], y[a]), _slope(x[~a], y[~a])]
                else:
                    e["halves"] = [None, None]
                e["ci"] = _boot_slope(x, y, rng) if ci and len(x) > 60 else (None, None)
        out[k] = e
    return out


def _passes(e: dict, k: str) -> dict:
    m = e[k]
    est, ci = m[MAIN_H]["slope"], m.get("ci", (None, None))
    s = SIGN[k]
    halves = sum(1 for v in m["halves"] if _ok(v) and v * s > 0)
    excl = ci[0] is not None and ((ci[1] < 0) if s < 0 else (ci[0] > 0))
    return {"pass": bool(_ok(est) and est * s > 0 and halves == 2 and excl), "halves": halves, "excl": excl}


# ---------- fair grid ----------

def _first_flag(sig: pd.DataFrame) -> pd.Timestamp:
    q = sig["R"].shift(1).rolling(WIN, min_periods=1).count()
    return q.index[int(np.argmax((q >= WIN_MIN).values))]


def _grid_span(first_flag: pd.Timestamp) -> tuple:
    a = (first_flag + pd.offsets.MonthBegin(1)).normalize()
    b = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=2)
    return a, b


def _daily_grid(close: pd.Series, target: pd.DataFrame, first, last, keys=KEYS) -> dict:
    starts, labels = msc._month_starts(close.index, first, last)
    change = close.pct_change().fillna(0.0).values
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out = {"starts": labels, "hold": (hr[:, 0], hd[:, 0])}
    for k in keys:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    def __init__(self, first, last):
        rows, self.problem = cr._hourly("BTCUSDT")
        if len(rows) < cr.MIN_HOURS:
            raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
        price, b0, _ = cr._grid(rows)
        self.pre = cr._prehistory(b0)
        if not len(self.pre):
            raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
        self.price, self.b0 = price, b0
        self.m = cr._Market(price, b0, self.pre)
        self.eng = rt._Engine(self.m)
        self.starts, self.labels = msc._month_starts(self.m.dates, first, last)
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hold = (hr[:, 0], hd[:, 0])
        self._ma = {}

    def _known(self, fl: pd.DataFrame, h: int) -> pd.DataFrame:
        """A decision at hour h >= 1 of date D uses the value stamped D 00:00; at h = 0 the one stamped D-1."""
        when = self.m.dates if h >= 1 else self.m.dates - pd.Timedelta(days=1)
        return fl.reindex(when).fillna(0.0)

    def run(self, fl: pd.DataFrame, person: bool = True, keys=KEYS) -> dict:
        m, eng = self.m, self.eng
        days, npre = m.days, len(self.pre)
        ns = len(self.starts)
        text = {k: (np.zeros((ns, 24)), np.zeros((ns, 24))) for k in keys}
        pers = {k: (np.zeros((ns, 24, SIMS)), np.zeros((ns, 24, SIMS))) for k in keys} if person else None
        for h in range(24):
            closes = np.concatenate([self.pre, self.price[h::24]]).astype(float)
            kn = self._known(fl, h)
            fr, fd = np.zeros(len(closes)), np.zeros(len(closes))
            fr[npre:npre + days], fd[npre:npre + days] = kn["R"].values, kn["D"].values
            rules = _rules_on(closes, fr, fd)
            for k in keys:
                w = np.nan_to_num(rules[k][npre:][:days], nan=0.0).astype(float)
                r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), self.starts, YEAR)
                text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
                if person:
                    b, wl = rt._later(h, w, days)
                    rng = np.random.default_rng(SEED + 1000 * SEED_KI[k] + h)
                    pos = rt._missed(wl, P_MISS, rng, SIMS)
                    r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]),
                                        self.starts, YEAR)
                    pers[k][0][:, h, :], pers[k][1][:, h, :] = r, dd
        return {"text": text, "person": pers}


def _summary(d: dict, hold: tuple, c: dict, others: dict) -> dict:
    hr, hd = hold
    s = {}
    for k in KEYS:
        dr, dd = d[k]
        tr, td = c["text"][k]
        edge = tr - hr[:, None]
        pr, pdd = c["person"][k]
        pedge = pr - hr[:, None, None]
        o_n = sum(len(o[k][0]) for o in others.values())
        o_beat = sum(int((o[k][0] > o["hold"][0]).sum()) for o in others.values())
        x = {"d_n": len(dr), "d_beat": int((dr > d["hold"][0]).sum()), "d_shallow": int((dd > d["hold"][1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)), "o_n": o_n, "o_beat": o_beat}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        s[k] = x
    order = sorted(KEYS, key=lambda k: (not s[k]["eligible"], -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "order": order, "hold_deep_med": float(np.median(hd))}


def _text_score(d: dict, hold: tuple, text: dict, others: dict, k: str) -> float:
    """Ranking score without the ordinary-person check (used for the placebo)."""
    hr = hold[0]
    a = float((d[k][0] > d["hold"][0]).mean())
    b = float((text[k][0] - hr[:, None] > 0).mean())
    n = sum(len(o[k][0]) for o in others.values())
    c = sum(int((o[k][0] > o["hold"][0]).sum()) for o in others.values()) / n if n else 0.0
    return min(a, b, c)


def _ratio(target: pd.Series, change: pd.Series, start, cost: float = COST, delay: int = 1) -> float | None:
    m = ro._measure(ll._window(ll._net(target, change, cost, delay=delay), start, None), YEAR)
    return m["ratio"] if m else None


# ---------- the whole computation ----------

def _load() -> dict:
    problems = []
    raw_rows = {}
    for sym in FIRST_DAY:
        try:
            rows, p = _fetch(sym)
        except Exception as exc:  # noqa: BLE001 - a coin without any data is left out
            if sym == "BTCUSDT":
                raise
            problems.append(f"{sym} 的多空比存檔這次取不到（{type(exc).__name__}），沒有算進去。")
            continue
        if p:
            problems.append(p)
        raw_rows[sym] = rows
    close = te._daily_close()
    coins = {}
    for sym, key in OTHER:
        try:
            coins[key] = ta._crypto_close(sym)
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{key} 現貨日線這次取不到（{type(exc).__name__}），沒有算進去。")
    return {"rows": raw_rows, "close": close, "coins": coins, "problems": problems}


def _prep(rows: dict, close: pd.Series) -> tuple:
    raw = _raw(rows, close.index[-1])
    sig = _signals(raw)
    return raw, sig, _flags(sig)


def _roll(raw: pd.DataFrame, s: int) -> pd.DataFrame:
    return pd.DataFrame({c: np.roll(raw[c].values, s) for c in ("R", "T")}, index=raw.index)


def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    raw, sig, fl = _prep(data["rows"]["BTCUSDT"], close)
    rng = np.random.default_rng(SEED)
    eff = _effects(sig, close, rng)
    late = _effects(sig.shift(1), close, rng, ci=False)
    oc = {}
    for sym, key in OTHER:
        if key not in data["coins"] or not data["rows"].get(sym):
            continue
        c = data["coins"][key]
        r_, s_, f_ = _prep(data["rows"][sym], c)
        oc[key] = {"close": c, "raw": r_, "sig": s_, "fl": f_, "eff": _effects(s_, c, rng, ci=False),
                   "ff": _first_flag(s_)}

    first_flag = _first_flag(sig)
    g0, g1 = _grid_span(first_flag)
    T = _targets(close, fl)
    daily = _daily_grid(close, T, g0, g1)
    clk = _Clock(g0, g1)
    clock = clk.run(fl)
    others = {}
    for key, o in oc.items():
        o["T"] = _targets(o["close"], o["fl"])
        a, _ = _grid_span(o["ff"])
        g = _daily_grid(o["close"], o["T"], a, g1)
        if len(g["starts"]):
            others[key] = g
            o["span"] = a
    sm = _summary(daily, clk.hold, clock, others)
    t_grid = time.time() - t0

    # placebo: every coin's R and T rolled by the same random number of days
    real = {k: _text_score(daily, clk.hold, clock["text"], others, k) for k in ("ls1", "ls2")}
    better = {"ls1": 0, "ls2": 0}
    prng = np.random.default_rng(SEED + 7)
    shifts = prng.integers(180, len(raw) - 180, DRAWS)
    for s in shifts:
        f2 = _flags(_signals(_roll(raw, int(s))))
        d2 = _daily_grid(close, _targets(close, f2), g0, g1, keys=("ls1", "ls2"))
        c2 = clk.run(f2, person=False, keys=("ls1", "ls2"))
        o2 = {}
        for key in others:
            o = oc[key]
            fo = _flags(_signals(_roll(o["raw"], int(s) % len(o["raw"]))))
            o2[key] = _daily_grid(o["close"], _targets(o["close"], fo), o["span"], g1, keys=("ls1", "ls2"))
            o2[key]["hold"] = others[key]["hold"]
        for k in better:
            better[k] += int(_text_score(d2, clk.hold, c2["text"], o2, k) >= real[k])
    placebo = {k: better[k] / DRAWS for k in better}

    # costs, ratios, crashes, whipsaws
    idx = close.index
    change = close.pct_change().fillna(0.0)
    ratio = {(k, c): _ratio(T[k], change, first_flag, c) for k in KEYS for c in (COST, COST_HI)}
    ratio["hold"] = _ratio(pd.Series(1.0, index=idx), change, first_flag, 0.0)
    T_late = _targets(close, fl, extra=1)
    ratio_late = {k: _ratio(T_late[k], change, first_flag, COST, delay=2) for k in KEYS}
    ratio_o = {}
    for key, o in oc.items():
        ch = o["close"].pct_change().fillna(0.0)
        ratio_o[key] = {k: _ratio(o["T"][k], ch, o["ff"]) for k in KEYS}
        ratio_o[key]["hold"] = _ratio(pd.Series(1.0, index=o["close"].index), ch, o["ff"], 0.0)
    c_arr = close.values
    first_i = int(idx.searchsorted(first_flag))
    bears = [b for b in ct._bears(close) if idx[b["top"]] >= first_flag]
    crash = []
    for b in bears:
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for k in KEYS:
            row[k] = ct._crash(c_arr, [(1.0, T[k].values)], b)["avoid"]
        crash.append(row)
    whip = {k: ct._whipsaws(c_arr, [(1.0, T[k].values)], first_i) for k in KEYS}
    after = idx >= first_flag
    off = {k: int(((T["ma50"] >= 0.5) & (T[k] < 0.5))[after].sum()) for k in ("ls1", "ls2")}
    on_days = int((T["ma50"] >= 0.5)[after].sum())
    flag_days = {k: int(fl[k][fl.index >= first_flag].sum()) for k in ("R", "D")}
    return {"close": close, "raw": raw, "sig": sig, "fl": fl, "eff": eff, "late": late, "oc": oc, "T": T,
            "first_flag": first_flag, "grid": (g0, g1), "daily": daily, "clk": clk, "clock": clock, "others": others,
            "sm": sm, "placebo": placebo, "real_score": real, "ratio": ratio, "ratio_late": ratio_late,
            "ratio_o": ratio_o, "crash": crash, "whip": whip, "off": off, "on_days": on_days, "flag_days": flag_days,
            "t_grid": t_grid, "t_all": time.time() - t0}


# ---------- checks ----------

def _lookahead(data: dict, r: dict) -> list[str]:
    out = []
    rng = np.random.default_rng(11)
    close, raw = r["close"], r["raw"]
    total = same = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        nc = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        nr = raw.copy()
        late = nr.index > c
        for col in ("R", "T"):
            nr.loc[late, col] = nr.loc[late, col].values * np.exp(rng.normal(0, 0.5, int(late.sum())))
        s2 = _signals(nr)
        f2 = _flags(s2)
        t2 = _targets(nc, f2)
        m = close.index <= c
        for k in KEYS:
            total += 1
            same += int(np.array_equal(np.nan_to_num(r["T"][k].values[m], nan=-1),
                                       np.nan_to_num(t2[k].values[m], nan=-1)))
        mf = raw.index <= c
        for k in ("R", "D"):
            total += 1
            same += int(np.array_equal(r["fl"][k].values[mf], f2[k].values[mf]))
        for k in SIGS:
            total += 1
            p1, p2 = _pos(r["sig"][k]), _pos(s2[k])
            same += int(np.array_equal(np.nan_to_num(p1.values[mf], nan=-1), np.nan_to_num(p2.values[mf], nan=-1)))
    out.append(f"偷看未來的檢查一：在 {'、'.join(CUTS)} 三個切點，把切點之後的比特幣收盤價和多空比全部換成亂數再算一次，"
               f"切點以前每一天的一年位置、「最高 20%」旗標和三條規則的進出 {total} 組裡 {same} 組完全沒變"
               + ("。" if same == total else "（有變動，結果不能用）。"))
    e, lt = r["eff"], r["late"]
    rr = "、".join(f"{SHORT[k]} {_num(r['ratio'][(k, COST)])} → {_num(r['ratio_late'][k])}" for k in KEYS)
    out.append("偷看未來的檢查二（再晚一天）：多空比再晚一天才用，30 天斜率 "
               + "、".join(f"{k} {_pp(e[k][MAIN_H]['slope'])} → {_pp(lt[k][MAIN_H]['slope'])}" for k in SIGS)
               + f"（百分點／每升一組）；規則的旗標晚一天、買賣再晚一天，報酬÷波動 {rr}。")
    n_alt = sum(1 for v in data["rows"]["BTCUSDT"].values() if v[5] != "00:00")
    out.append("時間對齊：每個 metrics 檔只取 UTC 0 點那一列（台灣早上 8 點），那一列在當天 UTC 0 點之後才公布，"
               "所以只拿來決定「當天結束時（隔天 UTC 0 點）」那根日線收盤後的部位，比收盤時就能知道的資訊晚了將近一天，偏保守。"
               f"BTC 有 {n_alt} 天沒有剛好 0 點那一列，改用 0:30 以前最早的一列；缺的日子沿用前一天，最多 {FFILL} 天。")
    return out


def _api_check(rows: dict) -> str:
    """Compare the archive's 00:00 rows with Binance's live 30-day API (same definitions?)."""
    try:
        a = common.get_json(FAPI + "/futures/data/globalLongShortAccountRatio",
                            {"symbol": "BTCUSDT", "period": "1d", "limit": 30}, timeout=15)
        b = common.get_json(FAPI + "/futures/data/topLongShortPositionRatio",
                            {"symbol": "BTCUSDT", "period": "1d", "limit": 30}, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return f"和幣安即時 API 的定義核對這次沒有做成（{type(exc).__name__}）。"
    res = []
    for name, got, i in (("散戶多空比", a, 0), ("大戶持倉多空比", b, 1)):
        n = m = 0
        gaps = []
        for x in got:
            day = pd.Timestamp(int(x["timestamp"]), unit="ms")
            key = _d(day)
            if day.hour == 0 and key in rows and rows[key][5] == "00:00":
                if rows[key][i] is None:
                    continue
                n += 1
                gap = abs(float(x["longShortRatio"]) / rows[key][i] - 1)
                gaps.append(gap)
                m += int(gap < 0.005)
        res.append(f"{name} {n} 天裡 {m} 天相差不到 0.5%" + (f"（最大差 {max(gaps) * 100:.2f}%）" if gaps else ""))
    return ("定義核對：幣安即時 API（儀表板「散戶多空人數比」「大戶多空持倉比」用的那兩個）最近 30 天的日資料，和存檔裡同一天 0 點那一列比，"
            + "、".join(res) + "；差距很小，看得出是同一個數列（差異可能來自取值的時間點和 API 只給小數 4 位）。")


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    t_dl = time.time() - t0
    r = _compute(data)
    problems = list(data["problems"]) + ([r["clk"].problem] if r["clk"].problem else [])
    eff, oc, sm, plc = r["eff"], r["oc"], r["sm"], r["placebo"]
    s = sm["s"]
    rank = {k: i for i, k in enumerate(sm["order"], 1)}
    raw, sig, fl, close = r["raw"], r["sig"], r["fl"], r["close"]
    g0, g1 = r["grid"]


    # ----- table 1: quintiles -----
    rows1 = []
    for k in SIGS:
        for i in range(5):
            q7, q30 = eff[k][7]["rows"][i], eff[k][30]["rows"][i]
            rows1.append([k if i == 0 else "",
                          f"第 {i + 1} 組（{'最低' if i == 0 else '最高' if i == 4 else '中間'} 20%）",
                          f"{q30['days']:,}", _pct(q7["mean"]), _pct(q30["mean"]), _share(q30["up"])])
    a7, a30 = eff["R"][7]["all"], eff["R"][30]["all"]
    rows1.append(["全部", "有一年位置的日子", f"{a30['days']:,}", _pct(a7["mean"]), _pct(a30["mean"]), _share(a30["up"])])
    t1 = {"caption": ("幣安 BTCUSDT 永續合約，每天 UTC 0 點的多空比，按「在過去一年的位置」分五組（每天只和自己之前 365 天比，不用到未來），"
                      "看之後 7／30 天比特幣平均漲跌。R ＝ 做多帳戶數 ÷ 做空帳戶數（所有帳戶，散戶為主）；T ＝ 前 20% 大戶多單部位 ÷ 空單部位；"
                      f"D ＝ log R − log T，越大代表散戶比大戶越偏多。多空比 {_d(raw.index[0])} 起才有，要先累積一年，所以只有 "
                      f"{_d(eff['R']['first'])} 到 {_d(eff['R']['last'])} 可比。相鄰日子之後的 30 天幾乎重疊，獨立的樣本大約只有天數 ÷ 30。"),
          "columns": ["訊號", "過去一年的位置", "天數", "之後 7 天平均", "之後 30 天平均", "30 天上漲比例"], "rows": rows1}

    # ----- table 2: slopes -----
    pe = {k: _passes(eff, k) for k in SIGS}
    coin_same = {}
    rows2 = []
    for k in SIGS:
        e = eff[k]
        ci = e.get("ci", (None, None))
        oth, same = [], 0
        for key in oc:
            v = oc[key]["eff"][k][MAIN_H]["slope"]
            same += int(_ok(v) and v * SIGN[k] > 0)
            oth.append(f"{key} {_pp(v)}")
        coin_same[k] = same
        rows2.append([f"{SIG_NAME[k]}（預期{'負' if SIGN[k] < 0 else '正'}）", _pp(e[7]["slope"]),
                      f"{_pp(e[MAIN_H]['slope'])}（{_pp(ci[0])} 到 {_pp(ci[1])}）",
                      _pp(e["halves"][0]), _pp(e["halves"][1]), "；".join(oth) if oth else "—", f"{same}／{len(oc)}"])
    t2 = {"caption": ("斜率 ＝ 位置每高一組（20%），之後比特幣平均漲跌多幾個百分點（最小平方法）。網路說法預期：R、D 越高之後越弱（負），"
                      "T 越高之後越強（正，跟著大戶）。括號是 30 天一段整段重抽 1,000 次的 90% 區間。前半／後半是各訊號自己有值的日子從中間切開："
                      + "；".join(f"{k} {_d(eff[k]['first'])}–{_d(eff[k]['split'])}–{_d(eff[k]['last'])}" for k in SIGS if eff[k]['first'] is not None)
                      + "。大戶那一欄（T）在 2022 年有大約 300 天是空的，所以 T、D 在 2022 年到 2023 年底之間沒有一年位置，天數比 R 少。其他幣用自己的多空比和自己的現貨收盤"
                      "（資料 2021-12 才有），只列 30 天斜率；最後一欄是四個幣裡和預期同方向的個數。"),
          "columns": ["訊號（預期方向）", "7 天斜率", "30 天斜率（90% 區間）", "前半 30 天", "後半 30 天", "其他幣 30 天斜率",
                      "其他幣同方向"], "rows": rows2}

    # ----- table 3: fair grid -----
    dn, cn = s["ma50"]["d_n"], s["ma50"]["c_n"]
    on = s["ma50"]["o_n"]
    rows3 = []
    for k in KEYS:
        x = s[k]
        ck = x["checks"]
        rows3.append([NAME[k], _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]) if x["o_n"] else "沒有資料",
                      f"{_share(x['low'])}（第 {rank[k]}）" + ("" if x["eligible"] else "，虧損條件未達"),
                      f"{te._pct(x['c_deep_med'])}｜{cr._pp(x['c_edge'])}", _share(plc[k]) if k in plc else "—"])
    o_txt = "、".join(f"{a} {len(o['starts'])} 個" for a, o in r["others"].items())
    t3 = {"caption": (f"main_signal_check 的公平考試，但起算日只能從多空比有一年資料之後開始：比特幣 UTC 0 點日線 {_d(g0)[:7]} 到 {_d(g1)[:7]} "
                      f"每月 1 日起算（只有 {dn} 個，main_signal_check 有 108 個），每個起算日最少只算到兩年；同樣起算月 × 24 個收盤時間（{cn:,} 格）；"
                      f"一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次）；其他四幣各用自己的多空比（{o_txt}，共 {on} 個）。"
                      "每格是「報酬贏過一直抱著」的比例；排名看四項裡最低的那項（前提：最深虧損比一直抱著淺的比例至少 95%）。"
                      f"「隨機錯位一樣好」：把每個幣的多空比整串循環平移同樣的隨機天數（{DRAWS} 次）重做規則，不含一般人那項的排名分數和真的一樣好或更好的比例。"),
          "columns": ["規則", f"起算月（{dn}）", f"起算×時間（{cn:,}）", "一般人照做", f"其他四幣（{on}）", "四項最低（排名）",
                      "最深虧損中位數｜每年多賺（中位數）", "隨機錯位一樣好"],
          "rows": rows3}

    # ----- table 4: crashes, costs -----
    rows4 = []
    for b in r["crash"]:
        rows4.append([f"{_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）") + "：躲掉的比例",
                      _pct(b["drop"])] + [_share(b[k]) for k in KEYS])
    w = r["whip"]
    rows4.append(["一年下車幾次｜其中 30 天內更高價買回", "—"] + [f"{w[k]['exits']:.1f}｜{w[k]['whips']:.1f}" for k in KEYS])
    rt_ = r["ratio"]
    rows4.append([f"報酬÷波動，成本 0.10%（一直抱著 {_num(rt_['hold'])}）", "—"] + [_num(rt_[(k, COST)]) for k in KEYS])
    rows4.append(["報酬÷波動，成本 0.30%", "—"] + [_num(rt_[(k, COST_HI)]) for k in KEYS])
    for key, v in r["ratio_o"].items():
        rows4.append([f"{key} 報酬÷波動 0.10%（一直抱著 {_num(v['hold'])}）", "—"] + [_num(v[k]) for k in KEYS])
    rows4.append([f"均線持有、規則改拿現金的天數（均線持有共 {r['on_days']:,} 天）", "—", "—",
                  f"{r['off']['ls1']:,}", f"{r['off']['ls2']:,}"])
    t4 = {"caption": (f"比特幣照課本做（收盤決定、隔天照做），從多空比第一天有一年位置的 {_d(r['first_flag'])} 起算。大跌段是 crash_timing 定義的"
                      "從高點收盤跌 50% 以上、高點在這一天之後的段落，「躲掉的比例」用規則自己在這段的漲跌算；「更高價買回」是下車後 30 天內又在更高的收盤價買回。"
                      "其他幣的報酬÷波動從各自第一天有一年位置起算。"),
          "columns": ["項目", "跌幅", "50 日均線", "LS1", "LS2"], "rows": rows4}
    tables = [t1, t2, t3, t4]

    # ----- pass test -----
    rule_of = {"R": "ls1", "D": "ls2"}
    detail, passed = {}, False
    for k, rk in rule_of.items():
        x, m = s[rk], s["ma50"]
        score_ok = x["low"] >= m["low"]
        loss_ok = x["c_deep_med"] >= m["c_deep_med"]
        plc_ok = plc[rk] <= 0.10
        coins_ok = coin_same[k] >= 3
        ok = pe[k]["pass"] and score_ok and loss_ok and plc_ok and coins_ok
        detail[k] = (score_ok, loss_ok, plc_ok, coins_ok, ok)
        passed = passed or ok

    # ----- today -----
    T = r["T"]
    last_sig = raw.dropna().index[-1]

    def where(k) -> str:
        p = _pos(sig[k])
        v = p.get(last_sig)
        return f"{v * 100:.0f}%" if _ok(v) else "—"

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    rl = sig.loc[last_sig]
    stale = last_sig < close.index[-1]
    now = [
        {"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"最新一筆多空比（存檔 {_d(last_sig)} UTC 0 點）",
         "value": f"散戶 R {rl['R']:.2f}（做多帳戶約占 {rl['R'] / (1 + rl['R']) * 100:.0f}%）；大戶持倉 T {rl['T']:.2f}；D {rl['D']:+.2f}"},
        {"label": "在自己過去一年裡由低到高的位置", "value": f"R {where('R')}｜T {where('T')}｜D {where('D')}"},
        {"label": "「一年最高 20%」旗標",
         "value": f"R：{'是' if fl['R'].loc[last_sig] > 0.5 else '否'}｜D：{'是' if fl['D'].loc[last_sig] > 0.5 else '否'}"},
        {"label": "50 日均線｜LS1｜LS2（最後一根日線收盤後）",
         "value": f"{word(T['ma50'].iloc[-1])}｜{word(T['ls1'].iloc[-1])}｜{word(T['ls2'].iloc[-1])}"
                  + (f"（多空比存檔只到 {_d(last_sig)}，之後沿用最後一筆，最多 {FFILL} 天）" if stale else "")},
        {"label": "這幾格的意思", "value": "只描述多空比和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    def dir_word(k) -> str:
        v = eff[k][MAIN_H]["slope"]
        ci = eff[k]["ci"]
        if not _ok(v):
            return "算不出來"
        inside = ci[0] is None or ci[0] <= 0 <= ci[1]
        w_ = "比較弱" if v < 0 else "反而比較強"
        return (f"{w_}（每高一組 {_pp(v)} 個百分點，{'區間含零' if inside else '區間不含零'}，"
                f"前後半 {_pp(eff[k]['halves'][0])}／{_pp(eff[k]['halves'][1])}）")

    yrs = (eff["R"]["last"] - eff["R"]["first"]).days / 365.25
    v1 = (f"幣安比特幣合約的多空比 {_d(raw.index[0])[:7]} 起才有，扣掉先累積一年只剩約 {yrs:.1f} 年可比，樣本很短；"
          f"散戶多空比在自己一年裡越高，之後 30 天{dir_word('R')}，散戶比大戶越偏多時{dir_word('D')}。")
    v2 = (f"加在 50 日均線上，排名分數均線 {_share(s['ma50']['low'])}、LS1 {_share(s['ls1']['low'])}、LS2 {_share(s['ls2']['low'])}，"
          f"隨機錯位一樣好 {_share(plc['ls1'])}、{_share(plc['ls2'])}，其他四幣同方向 R {coin_same['R']}／{len(oc)}、D {coin_same['D']}／{len(oc)}。")
    v3 = "事先定的標準" + ("通過，但只有幾年資料，只能當線索。" if passed else "沒有通過。")
    verdict = v1 + v2 + v3

    # ----- notes -----
    def etext(k) -> str:
        x = pe[k]
        return f"前後半同為預期方向 {x['halves']}／2、全期間 90% 區間{'不含' if x['excl'] else '含'}零 → {'過' if x['pass'] else '不過'}"

    pre_reg = (
        "事先寫下的設計（看結果之前，之後沒有改）：Daily values at 00:00 UTC: retail ratio R = count_long_short_ratio (all accounts), "
        "top-trader position ratio T = sum_toptrader_long_short_ratio; divergence D = log(R) − log(T). 1. Forward 7/30-day BTC returns by "
        "trailing-1-year quintile of R, T and D (point-in-time cut-offs); slopes with block bootstrap; split the sample into two halves by "
        "date (state them). 2. Rules on top of ma50 (act at the next close; costs 0.10/0.30%): LS1 ma50 but cash when R is in its "
        "trailing-year top 20% (crowded retail longs); LS2 ma50 but cash when D is in its trailing-year top 20% (retail more long than top "
        "traders). 3. Fair grid for whatever starts the data allows (say how few), ordinary-person execution, ETH/SOL/BNB/XRP with their own "
        "metrics files, placebo with circularly shifted signal (200 draws, ranking score), crash-avoided share with whipsaws. Pass test "
        "(stated now): predicted (contrarian) sign in both halves with full-sample 90% interval excluding zero AND a rule's ranking score ≥ "
        "ma50's with median deepest loss no worse AND placebo share ≤ 10% AND same sign on ≥ 3 of 4 other coins. State clearly that the "
        "sample is short. Today: R, T, D and percentiles; rule states — states only.")
    eR = eff["R"][MAIN_H]
    dpos = _pos(sig["D"]).dropna()
    gaps_d = dpos.index.to_series().diff() > pd.Timedelta(days=30)
    d_back = dpos.index[gaps_d.values][-1] if gaps_d.any() else None
    notes = [
        pre_reg,
        "白話和事先補定的細節：多空比 ＝ 做多的人數（或部位）÷ 做空的人數（或部位），大於 1 代表做多的比較多。主要看 30 天。過關要 R 配 LS1、或 D 配 LS2 "
        "整組都過：斜率在前後半都是預期方向（R、D 為負）、全期間 90% 區間不含零；規則的排名分數（main_signal_check 四項「報酬贏過一直抱著」比例裡最低的）"
        "不低於 50 日均線、起算×收盤時間各格最深虧損的中位數不比均線深；隨機錯位一樣好的比例 ≤ 10%；其他四幣至少三個斜率也是預期方向。"
        "隨機錯位的分數不含一般人做法那項（那項每次要模擬 24×200 次，太慢），真的規則也用同樣少一項的分數來比"
        f"（真的分數 LS1 {_share(r['real_score']['ls1'])}、LS2 {_share(r['real_score']['ls2'])}）。",
        f"過關檢查明細｜R：{etext('R')}；LS1 排名分數{'不低於' if detail['R'][0] else '低於'}均線、最深虧損中位數{'不比' if detail['R'][1] else '比'}均線深、"
        f"隨機錯位 {_share(plc['ls1'])}、其他幣同方向 {coin_same['R']}／{len(oc)} → {'過' if detail['R'][4] else '不過'}。"
        f"D：{etext('D')}；LS2 排名分數{'不低於' if detail['D'][0] else '低於'}均線、最深虧損中位數{'不比' if detail['D'][1] else '比'}均線深、"
        f"隨機錯位 {_share(plc['ls2'])}、其他幣同方向 {coin_same['D']}／{len(oc)} → {'過' if detail['D'][4] else '不過'}。"
        f"T（只描述，不在過關標準裡）：{etext('T')}。整體：{'通過' if passed else '沒有通過'}。",
        f"樣本很短：BTC 的多空比 {_d(raw.index[0])} 起才有，「一年位置」要先累積 330 天以上，所以 {_d(r['first_flag'])} 起才有值；"
        f"有 30 天後報酬的日子只有 {eR['n']:,} 天，換成不重疊的 30 天大約只有 {eR['n'] // 30} 段，涵蓋的大行情只有 2022 年空頭、2023–2025 年多頭等少數幾段，"
        "前後半各自的方向可能只是反映行情不同。其他四幣 2021-12 起才有，比 BTC 更短。",
        f"資料缺口：BTC 存檔 {len(data['rows']['BTCUSDT']):,} 天裡，大戶持倉多空比（T）是空的有 "
        f"{sum(1 for v in data['rows']['BTCUSDT'].values() if v[1] is None):,} 天（集中在 2022 年，幣安的檔案裡那幾欄是空字串），"
        f"散戶多空比 R 幾乎每天都有；所以 R 的一年位置 {_d(eff['R']['first'])} 起每天都有（{eff['R'][MAIN_H]['n']:,} 天），"
        f"T 和 D 只有 2022 年缺口之前的幾個月和缺口後再累積一年（{_d(d_back) if d_back is not None else '—'} 起）才有，共 {eff['D'][MAIN_H]['n']:,} 天"
        f"（缺的日子只沿用 {FFILL} 天、一年裡至少要 330 天有值，這是事先定的，沒有因為缺口放寬）；"
        "D 沒有值的日子 LS2 就和 50 日均線一模一樣，所以 LS2 的證據比 LS1 更少。",
        f"旗標出現的天數（{_d(r['first_flag'])} 起）：R 在一年最高 20% 有 {r['flag_days']['R']:,} 天，D 有 {r['flag_days']['D']:,} 天；"
        f"均線持有時，LS1 改拿現金 {r['off']['ls1']:,} 天、LS2 {r['off']['ls2']:,} 天。",
        "R 是算人頭：每個有倉位的帳戶一票，小帳戶占多數，所以被當成散戶情緒；T 是幣安保證金前 20% 帳戶的多單部位 ÷ 空單部位，"
        "不等於真正的主力動向（大戶可能在別的交易所或現貨對沖）。多單總量一定等於空單總量，R 大只代表做空的帳戶比較少、平均比較大。"
        "這兩個數字都只是幣安一家的統計；資金費率和未平倉量已在 funding_positioning、forced_flows 做過，這裡不重複。",
        "K33 只說散戶多空比「衝上 4 以上」時曾在幾次下跌之前出現，CRYPTINT 只說「大戶空、散戶多常是局部高點」，都沒有公開門檻和回測；"
        "這裡改用「自己過去一年最高 20%」當門檻，因為多空比的水準這幾年一直在變，固定的數字在大部分期間不會出現。",
    ]
    notes += _lookahead(data, r)
    notes.append(_api_check(data["rows"]["BTCUSDT"]))
    notes += [
        f"成本很粗略：照課本做每次買賣扣 {COST * 100:.2f}%（另列 {COST_HI * 100:.2f}%）、一般人做法 {P_COST * 100:.2f}%；現金沒有利息；一直抱著沒有扣成本；沒有算稅。"
        "相鄰起算日共用大部分歷史、24 個收盤時間只差幾小時，這些「幾個裡幾個」不是互相獨立的證據。其他幣是今天還在的大幣，有倖存者偏誤。"
        "隨機的部分都用固定的亂數種子，資料沒變的話重跑結果一樣。",
        f"這次計算用了 {time.time() - t0:.0f} 秒（其中下載和讀檔 {t_dl:.0f} 秒）。第一次執行要從 data.binance.vision 下載約 9,500 個每日檔"
        "（每個約 12 KB，一次一個），這次大約花了 20 多分鐘；之後只抓新的日子。每個幣每天只存 0 點那一列，存在 _store 的 ls_ratio_*.json。",
        "過去的結果不保證未來會一樣。這裡只是把流行的說法照原樣測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "ls_ratio",
        "title": "幣安多空比：散戶太擠在多方之後比較弱嗎",
        "question": ("幣安合約的散戶多空比（做多的帳戶數 ÷ 做空的帳戶數）在自己一年裡偏高、或散戶比大戶更偏多的時候，之後比特幣是不是比較弱？"
                     "加在 50 日均線上當「太擠就先下車」有沒有幫助？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
