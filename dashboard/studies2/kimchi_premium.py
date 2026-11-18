"""Historical study "kimchi_premium": does the Korean premium (bitcoin on Upbit in won vs the international price
times the dollar-won rate) tell anything about bitcoin's next move, and does it help the 50-day-average rule?

Claims tested exactly as published:
  * CryptoSlate backtest: when the Korean premium crosses from negative to positive ("zero-crossing"), BTC averaged
    +1.7% after 7 days and +6.2% after 30 days, win rates 67% / 70%; correlation of the level with forward returns
    about -0.06.
  * CryptoQuant (via BeInCrypto): Korea premium above 3% together with a negative Coinbase premium signals a
    short-term correction.
  * Academic context: Makarov & Schoar - spreads persist and predict relative exchange returns; Finance Research
    Letters 2023 - the relationship is not robust over time.

Pre-registered tests (written before any result was seen; not changed afterwards):
1. Level: forward BTC 7/30/90-day returns by premium quintile; correlation; split 2017-09..2019, 2020..2022,
   2023..now; block bootstrap.
2. K1 (CryptoSlate exactly): premium (7-day average, to avoid daily noise - state this; also report the raw-daily
   version) crosses from <= 0 to > 0 -> forward 7/30-day returns and win rates vs all days and vs random dates; count
   events and independent events.
3. K2 (CryptoQuant exactly): premium > 3% AND Coinbase premium < 0 -> forward 7/30-day returns vs all days.
4. K3 (froth): premium > its trailing-1-year 90th percentile -> forward 30/90-day returns (top/bubble claim).
5. Rules on top of ma50 (act at the next close; costs 0.10/0.30%): R_K2 step to cash for 14 days after K2 fires;
   R_K3 step to cash while K3 is on. Fair grid where data allows, ordinary-person execution, placebo with circularly
   shifted premium (200 draws), crash-avoided share with whipsaws.
Pass test (stated now): signal effect same sign in all three periods with full-sample 90% interval excluding zero AND
a rule beats ma50 on return/volatility with placebo share <= 10%.
Today: Korean premium, Coinbase premium, which signals are on - states only.

Details fixed with the design (before results):
  * Premium = Upbit KRW-BTC close / (Binance BTCUSDT close x USD/KRW) - 1. Coinbase premium = Coinbase BTC-USD close /
    Binance BTCUSDT close - 1. A version with Coinbase BTC-USD as the international price is reported as a check.
  * Timestamps: Upbit's daily candle runs 09:00 KST -> 09:00 KST, i.e. 00:00 UTC -> 00:00 UTC (its own
    candle_date_time_utc field; the stored "timestamp" of every candle is checked to fall inside that UTC day or at
    most a minute after it). Binance and Coinbase daily candles also close at 00:00 UTC, so all three prices are taken at the same
    instant, the close of the day labelled t (= 00:00 UTC of t+1). USD/KRW = Fed H.10 noon New York buying rate
    (12:00 ET = 16:00 or 17:00 UTC on day t, before the close); the decision on day t uses the rate of the latest
    business day <= t (weekends and holidays repeat the last rate).
  * The level test (1), K2 and K3 use the daily premium; K1 uses the 7-day average (at least 5 of 7 days), and the
    raw-daily crossing is reported beside it. K3's trailing year = the 365 daily premiums before day t (at least 330).
  * Primary horizon for the pass test: 30 days for every signal (the level test: the correlation with 30-day forward
    returns; K1/K2/K3: mean 30-day forward return on signal days minus the mean over all days of the same span). Sign
    "same in all three periods" means the same sign as the full-sample estimate.
  * Block bootstrap: circular blocks of max(30, horizon) days, 1,000 draws, 90% percentile interval. Random dates:
    the signal's day series circularly shifted by a random amount (1,000 draws, keeps its clustering); for K1 also
    the same number of plain random dates. Independent events: at least 30 days after the previous counted one.
  * Rules: ma50 = close above its 50-day average. R_K2 = ma50, but cash on the day K2 fires and the 13 days after.
    R_K3 = ma50, but cash while K3 is on. Textbook: decided at the close, held from the next day (as every rule on the
    page), 0.10% of the amount traded. Ordinary person: one more day later (acts at the next close), 0.30%, 10% of
    entries missed at random (robust_trend._missed shape), 200 simulations, the same draws for every rule. Fair grid:
    72 monthly start dates 2018-01 ... 2023-12 measured to the last finished day (return / volatility). Placebo: the
    Korean premium series is rolled circularly by 180 days ... (length - 180) days (200 draws, the Coinbase premium
    stays real) and K2/K3 and the rules are rebuilt; share of draws at least as good as the real rule over the whole
    span. Crashes: crash_timing's bear markets (a close 50% under the peak) with a top after 2017-09-25, the share of
    the fall avoided and whipsaws (exit, then back in within 30 days at a higher close).
  * Look-ahead checks: candle timestamps; FX date never after the decision day; every input dated after three cut
    dates scrambled -> no decision up to the cut may change; one extra day of delay.

Data (public, key-less, the dashboard's fixed user agent): Upbit public candles API, Coinbase Exchange public candles,
Binance daily closes through the dashboard's long daily series, Fed Board Data Download Program H.10 (daily KRW).
"""
from __future__ import annotations

import io
import json
import math
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import liquidity_lead as ll
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te

COST, COST_P = 0.001, 0.003
P_MISS, SIMS, DRAWS, BOOT = 0.10, 200, 200, 1000
SEED = 20170925
YEAR = 365
H = (7, 30, 90)
MAIN_H = 30
K2_LEVEL = 0.03
K2_DAYS = 14
K3_Q, K3_WIN, K3_MIN = 0.90, 365, 330
GAP = 30
FIRST = "2017-09-25"
RSTART = "2017-09-26"
PERIODS = (("2017-09–2019", FIRST, "2019-12-31"), ("2020–2022", "2020-01-01", "2022-12-31"),
           ("2023 起", "2023-01-01", None))
GRID = ("2018-01-01", "2023-12-01")
CUTS = ("2019-06-30", "2021-06-30", "2024-06-30")

UPBIT_URL = "https://api.upbit.com/v1/candles/days"
COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
FED_ZIP = "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&filetype=zip"
STORE_UP = common.STORE / "kimchi_premium_upbit.json"
STORE_CB = common.STORE / "kimchi_premium_coinbase.json"
STORE_FX = common.STORE / "kimchi_premium_usdkrw.json"

SOURCES = [
    {"name": "CryptoSlate：Does the kimchi premium still front-run BTC?（溢價轉正之後 7／30 天的回測）",
     "url": "https://cryptoslate.com/does-thes-kimchi-premium-still-front-run-btc/"},
    {"name": "BeInCrypto 引用 CryptoQuant：韓國溢價超過 3% 加上 Coinbase 溢價為負＝短線回檔訊號",
     "url": "https://beincrypto.com/?p=454299"},
    {"name": "Makarov & Schoar：Trading and Arbitrage in Cryptocurrency Markets（價差會持續）",
     "url": "https://personal.lse.ac.uk/makarov1/index_files/CryptocurrencyMarkets.pdf"},
    {"name": "Finance Research Letters 2023：Is the kimchi premium a speculative bubble?（關係不穩定）",
     "url": "https://pure.uos.ac.kr/en/publications/is-the-kimchi-premium-a-speculative-bubble/"},
    {"name": "Upbit 公開日線 API（KRW-BTC，2017-09-25 起）", "url": UPBIT_URL + "?market=KRW-BTC&count=200"},
    {"name": "Coinbase Exchange 公開日線（BTC-USD）", "url": COINBASE_URL + "?granularity=86400"},
    {"name": "美國聯準會 H.10 匯率（韓元，紐約中午買入匯率，日資料；Data Download Program 整包下載）", "url": FED_ZIP},
    {"name": "幣安現貨日線（BTC 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
]

_ok, _num, _d = te._ok, te._num, te._d


def _pct(v, signed: bool = True) -> str:
    """v is a fraction."""
    return te._pct(v * 100, signed) if _ok(v) else "—"


def _pp(v) -> str:
    return f"{v * 100:+.1f}" if _ok(v) else "—"


# ---------- data ----------

def _read(path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a damaged store starts over
        return {}


def _write(path, data: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(data.items()))), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store only saves time
        pass


def _get(url: str, params: dict | None = None, timeout: int = 30, pause: float = 0.25):
    for attempt in range(4):
        reply = requests.get(url, params=params, headers=common.HEADERS, timeout=timeout)
        if reply.status_code == 429 and attempt < 3:  # rate limit: wait and ask again, politely
            time.sleep(2.0 * (attempt + 1))
            continue
        reply.raise_for_status()
        time.sleep(pause)
        return reply
    raise RuntimeError("rate limited")


def _upbit() -> tuple[dict, str]:
    """{YYYY-MM-DD (UTC): [close KRW, last-trade ms]} of finished Upbit KRW-BTC daily candles."""
    stored = _read(STORE_UP)
    today = common.today_utc()
    have = max(stored) if stored else None
    try:
        to = None
        for _ in range(40):
            params = {"market": "KRW-BTC", "count": 200}
            if to:
                params["to"] = to
            page = _get(UPBIT_URL, params).json()
            if not page:
                break
            for c in page:
                day = str(c["candle_date_time_utc"])[:10]
                if day < today and float(c["trade_price"]) > 0:
                    stored[day] = [float(c["trade_price"]), int(c["timestamp"])]
            oldest = min(str(c["candle_date_time_utc"]) for c in page)
            if len(page) < 200 or (have and oldest[:10] < have) or oldest[:10] <= FIRST:
                break
            to = oldest + "Z"
        _write(STORE_UP, stored)
        note = ""
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        note = f"Upbit 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料（到 {max(stored)}）。"
    return stored, note


def _coinbase() -> tuple[dict, str]:
    """{YYYY-MM-DD: close} of finished Coinbase BTC-USD daily candles from 2017-08-01."""
    stored = _read(STORE_CB)
    today = pd.Timestamp(common.today_utc())
    start = (pd.Timestamp(max(stored)) - pd.Timedelta(days=5)) if stored else pd.Timestamp("2017-08-01")
    try:
        while start < today:
            end = min(start + pd.Timedelta(days=299), today)
            page = _get(COINBASE_URL, {"granularity": 86400, "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                                       "end": end.strftime("%Y-%m-%dT00:00:00Z")}).json()
            for row in page:
                day = pd.Timestamp(int(row[0]), unit="s")
                if day < today and float(row[4]) > 0:
                    stored[_d(day)] = float(row[4])
            start = end + pd.Timedelta(days=1)
        _write(STORE_CB, stored)
        note = ""
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        note = f"Coinbase 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料（到 {max(stored)}）。"
    return stored, note


def _fx() -> tuple[dict, str]:
    """{YYYY-MM-DD: won per dollar}, Fed H.10 noon buying rate (daily, business days). Re-downloaded at most every
    12 hours."""
    stored = _read(STORE_FX)
    try:
        fresh = STORE_FX.exists() and time.time() - STORE_FX.stat().st_mtime < 12 * 3600 and len(stored) > 2000
    except Exception:  # noqa: BLE001
        fresh = False
    if fresh:
        return stored, ""
    try:
        zf = zipfile.ZipFile(io.BytesIO(_get(FED_ZIP, timeout=180).content))
        member = next(n for n in zf.namelist() if n.lower().endswith("_data.xml"))
        text = zf.read(member).decode("utf-8")
        i = text.find('SERIES_NAME="RXI_N.B.KO"')
        if i < 0:
            raise RuntimeError("H.10 裡找不到韓元日資料")
        j = text.find("</kf:Series>", i)
        days = {}
        for value, day in re.findall(r'OBS_VALUE="([^"]+)" TIME_PERIOD="([^"]+)"', text[i:j]):
            try:
                v = float(value)
            except ValueError:
                continue
            if math.isfinite(v) and v > 0 and day >= "2016-01-01":
                days[day[:10]] = v
        if len(days) < 2000:
            raise RuntimeError(f"韓元匯率只有 {len(days)} 筆")
        _write(STORE_FX, days)
        return days, ""
    except Exception as exc:  # noqa: BLE001
        if not stored:
            raise
        return stored, f"聯準會 H.10 這次沒有下載成功（{type(exc).__name__}），用的是之前存下來的匯率（到 {max(stored)}）。"


def _load() -> dict:
    up, n1 = _upbit()
    cb, n2 = _coinbase()
    fx, n3 = _fx()
    close = te._daily_close()
    days = pd.date_range(FIRST, close.index[-1], freq="D")
    up_s = pd.Series({pd.Timestamp(k): v[0] for k, v in up.items()}).sort_index()
    up_ts = pd.Series({pd.Timestamp(k): v[1] for k, v in up.items()}).sort_index()
    cb_s = pd.Series({pd.Timestamp(k): v for k, v in cb.items()}).sort_index()
    fx_s = pd.Series({pd.Timestamp(k): v for k, v in fx.items()}).sort_index()
    return {"close": close, "days": days, "up": up_s.reindex(days), "up_ts": up_ts.reindex(days),
            "cb": cb_s.reindex(days), "fx_raw": fx_s, "problems": [n for n in (n1, n2, n3) if n]}


# ---------- the premium and the signals ----------

def _known_fx(fx_raw: pd.Series, days: pd.DatetimeIndex, extra: int = 0) -> tuple[pd.Series, pd.Series]:
    """The rate a decision at the close of each day may use (latest business day <= day - extra) and its date."""
    s = fx_raw.copy()
    s.index = s.index + pd.Timedelta(days=extra)
    dates = pd.Series(fx_raw.index, index=s.index)
    u = days.union(s.index)
    return s.reindex(u).ffill().reindex(days), dates.reindex(u).ffill().reindex(days)


def _premiums(data: dict, fx_extra: int = 0) -> dict:
    days, close = data["days"], data["close"]
    bn = close.reindex(days)
    fx, fx_date = _known_fx(data["fx_raw"], days, fx_extra)
    kp = data["up"] / (bn * fx) - 1
    return {"kp": kp, "kp_cb": data["up"] / (data["cb"] * fx) - 1, "cbp": data["cb"] / bn - 1, "fx": fx,
            "fx_date": fx_date}


def _flags(kp: pd.Series, cbp: pd.Series) -> dict:
    kp7 = kp.rolling(7, min_periods=5).mean()
    thr = kp.shift(1).rolling(K3_WIN, min_periods=K3_MIN).quantile(K3_Q)
    return {"kp7": kp7, "thr": thr,
            "k1": (kp7.shift(1) <= 0) & (kp7 > 0),
            "k1raw": (kp.shift(1) <= 0) & (kp > 0),
            "k2": (kp > K2_LEVEL) & (cbp < 0),
            "k3": (kp > thr) & thr.notna()}


def _rules(fl: dict, idx: pd.DatetimeIndex, ma50: pd.Series) -> dict:
    k2 = fl["k2"].reindex(idx).fillna(False).astype(float)
    k3 = fl["k3"].reindex(idx).fillna(False).astype(float)
    k2win = k2.rolling(K2_DAYS, min_periods=1).max()
    return {"rk2": ma50 * (1 - k2win), "rk3": ma50 * (1 - k3)}


# ---------- statistics ----------

def _blocks(n: int, length: int, draws: int, rng) -> np.ndarray:
    nb = -(-n // length)
    st = rng.integers(0, n, (draws, nb))
    return ((st[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def _excess(flag: np.ndarray, fwd: np.ndarray) -> float | None:
    if flag.sum() == 0:
        return None
    return float(fwd[flag].mean() - fwd.mean())


def _boot_excess(flag: np.ndarray, fwd: np.ndarray, h: int, rng) -> tuple:
    idx = _blocks(len(fwd), max(30, h), BOOT, rng)
    f, r = flag[idx], fwd[idx]
    cnt = f.sum(axis=1)
    good = cnt > 0
    ex = (f * r).sum(axis=1)[good] / cnt[good] - r.mean(axis=1)[good]
    return (float(np.percentile(ex, 5)), float(np.percentile(ex, 95))) if len(ex) > 50 else (None, None)


def _boot_corr(x: np.ndarray, y: np.ndarray, h: int, rng) -> tuple:
    idx = _blocks(len(x), max(30, h), BOOT, rng)
    a, b = x[idx], y[idx]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    r = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def _shift_share(flag: np.ndarray, fwd: np.ndarray, real: float | None, sign: int, rng) -> float | None:
    """Share of circular shifts of the signal days whose excess is at least as extreme in the claimed direction."""
    if real is None:
        return None
    n = len(flag)
    if n < 400:
        return None
    hits = 0
    for s in rng.integers(180, n - 180, BOOT):
        e = _excess(np.roll(flag, int(s)), fwd)
        hits += int(e is not None and sign * e >= sign * real)
    return hits / BOOT


def _independent(dates: list) -> int:
    n, last = 0, None
    for d in dates:
        if last is None or (d - last).days >= GAP:
            n += 1
            last = d
    return n


def _span(idx: pd.DatetimeIndex, a, b) -> np.ndarray:
    return (idx >= pd.Timestamp(a)) & ((idx <= pd.Timestamp(b)) if b else True)


def _event(flag: pd.Series, fwd: dict, sign: int, rng, shift: int = 0) -> dict:
    """Forward returns on signal days vs all days (shift = extra days of delay before measuring)."""
    out = {"days": int(flag.sum()), "indep": _independent(list(flag.index[flag.values])),
           "dates": list(flag.index[flag.values])}
    for h in H:
        f = fwd[h].shift(-shift) if shift else fwd[h]
        ok = f.notna().values
        fl, r = flag.values[ok], f.values[ok]
        x = {"n": int(fl.sum()), "all": float(r.mean()), "all_win": float((r > 0).mean())}
        x["mean"] = float(r[fl].mean()) if fl.sum() else None
        x["win"] = float((r[fl] > 0).mean()) if fl.sum() else None
        x["ex"] = _excess(fl, r)
        if h == MAIN_H and not shift:
            x["ci"] = _boot_excess(fl, r, h, rng)
            x["shift"] = _shift_share(fl, r, x["ex"], sign, rng)
            per = []
            idx = flag.index[ok]
            for _, a, b in PERIODS:
                m = _span(idx, a, b)
                per.append(_excess(fl[m], r[m]) if m.sum() > 60 else None)
            x["per"] = per
        out[h] = x
    return out


def _random_dates(flag: pd.Series, fwd: pd.Series, real: float | None, rng) -> float | None:
    ok = fwd.notna().values
    r = fwd.values[ok]
    m = int(flag.values[ok].sum())
    if not m or real is None:
        return None
    means = np.array([r[rng.choice(len(r), m, replace=False)].mean() for _ in range(BOOT)])
    return float((means >= real).mean())


# ---------- the whole computation ----------

def _compute(data: dict) -> dict:
    close, days = data["close"], data["days"]
    pr = _premiums(data)
    kp, cbp = pr["kp"], pr["cbp"]
    fl = _flags(kp, cbp)
    rng = np.random.default_rng(SEED)
    fwd = {h: (close.shift(-h) / close - 1).reindex(days) for h in H}

    # ----- test 1: level -----
    ok_kp = kp.notna()
    edges = np.nanpercentile(kp.values, [0, 20, 40, 60, 80, 100])
    q = pd.Series(np.clip(np.searchsorted(edges[1:-1], kp.values, side="right"), 0, 4), index=days).where(ok_kp)
    level = {"edges": edges, "quint": [], "corr": {}}
    for i in range(5):
        m = (q == i)
        row = {"days": int(m.sum())}
        for h in H:
            f = fwd[h][m].dropna()
            row[h] = float(f.mean()) if len(f) else None
            if h == MAIN_H:
                row["win"] = float((f > 0).mean()) if len(f) else None
        level["quint"].append(row)
    level["all"] = {h: float(fwd[h][ok_kp].dropna().mean()) for h in H}
    level["all_win"] = float((fwd[MAIN_H][ok_kp].dropna() > 0).mean())
    for h in H:
        both = pd.concat([kp, fwd[h]], axis=1).dropna()
        for name, a, b in PERIODS:
            part = both[_span(both.index, a, b)]
            level["corr"][(h, name)] = float(part.iloc[:, 0].corr(part.iloc[:, 1])) if len(part) > 60 else None
        level["corr"][(h, "all")] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
        level["corr"][(h, "ci")] = _boot_corr(both.iloc[:, 0].values, both.iloc[:, 1].values, h, rng)
        level["corr"][(h, "n")] = len(both)

    # ----- tests 2-4: signals -----
    sign = {"k1": 1, "k1raw": 1, "k2": -1, "k3": -1}
    ev = {k: _event(fl[k].fillna(False), fwd, sign[k], rng) for k in sign}
    rand_k1 = _random_dates(fl["k1"], fwd[MAIN_H], ev["k1"][MAIN_H]["mean"], rng)
    rand_k1_7 = _random_dates(fl["k1"], fwd[7], ev["k1"][7]["mean"], rng)
    late = {k: _event(fl[k].fillna(False), fwd, sign[k], rng, shift=1) for k in sign}

    # ----- test 5: rules -----
    idx = close.index
    change = close.pct_change().fillna(0.0)
    ma50 = te._signals(close)[0]["A50"]
    T = {"hold": pd.Series(1.0, index=idx), "ma50": ma50}
    T.update(_rules(fl, idx, ma50))
    nets = {c: ll._net(v, change, COST) for c, v in T.items()}
    per = {}
    for name, a, b in PERIODS + (("all", RSTART, None),):
        a2 = max(pd.Timestamp(a), pd.Timestamp(RSTART))
        per[name] = {c: ro._measure(ll._window(nets[c], a2, b), YEAR) for c in T}
    starts, labels = msc._month_starts(idx, GRID[0], GRID[1])
    cand = ["hold", "ma50", "rk2", "rk3"]
    text_r = {c: ll._suffix_ratio(np.nan_to_num(nets[c].values), starts, YEAR)[:, 0] for c in cand}
    rng_draw = np.random.default_rng(SEED + 1).random((len(idx), SIMS))
    person = {}
    for c in cand:
        w = T[c].shift(2).fillna(0.0).values
        step = np.diff(w, prepend=0.0)
        skip = (step > rt.RISE)[:, None] & (rng_draw < P_MISS)
        move = np.where(skip, step[:, None], 0.0) + np.minimum(step, 0.0)[:, None]
        total = np.cumsum(move, axis=0)
        gap = total - np.minimum(np.minimum.accumulate(total, axis=0), 0.0)
        pos = np.clip(w[:, None] - gap, 0.0, 1.0)
        person[c] = ll._suffix_ratio(ll._run2d(pos, change.values, COST_P), starts, YEAR)
    grid = {c: {"n": len(starts), "text_beat": int((text_r[c] > text_r["ma50"]).sum()),
                "text_med": float(np.median(text_r[c])),
                "person_beat": float((person[c] > person["ma50"]).mean()),
                "person_med": float(np.median(person[c]))} for c in cand}

    # placebo: roll the Korean premium
    real = {c: per["all"][c]["ratio"] for c in ("rk2", "rk3")}
    n = len(days)
    better = {"rk2": 0, "rk3": 0}
    prng = np.random.default_rng(SEED + 2)
    for s in prng.integers(180, n - 180, DRAWS):
        kp_s = pd.Series(np.roll(kp.values, int(s)), index=days)
        tg = _rules(_flags(kp_s, cbp), idx, ma50)
        for c in better:
            m = ro._measure(ll._window(ll._net(tg[c], change, COST), RSTART, None), YEAR)
            better[c] += int(m is not None and m["ratio"] >= real[c])
    placebo = {c: better[c] / DRAWS for c in better}

    # crashes and whipsaws
    c_arr = close.values
    first = int(idx.searchsorted(pd.Timestamp(RSTART)))
    bears = [b for b in ct._bears(close) if idx[b["top"]] >= pd.Timestamp(FIRST)]
    crash = []
    for b in bears:
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for c in ("ma50", "rk2", "rk3"):
            row[c] = ct._crash(c_arr, [(1.0, T[c].values)], b)["avoid"]
        crash.append(row)
    whip = {c: ct._whipsaws(c_arr, [(1.0, T[c].values)], first) for c in ("ma50", "rk2", "rk3")}
    cash_days = {c: int(((ma50 >= 0.5) & (T[c] < 0.5))[idx >= pd.Timestamp(RSTART)].sum()) for c in ("rk2", "rk3")}

    # one extra day for the rules
    late_rules = {c: ro._measure(ll._window(ll._net(T[c], change, COST, delay=2), RSTART, None), YEAR)
                  for c in ("ma50", "rk2", "rk3")}

    return {"close": close, "days": days, "pr": pr, "fl": fl, "fwd": fwd, "level": level, "ev": ev, "late": late,
            "rand_k1": rand_k1, "rand_k1_7": rand_k1_7, "T": T, "per": per, "grid": grid, "placebo": placebo,
            "crash": crash, "whip": whip, "cash_days": cash_days, "late_rules": late_rules, "ma50": ma50}


# ---------- look-ahead checks ----------

def _lookahead(data: dict, r: dict) -> list[str]:
    out = []
    ts = data["up_ts"].dropna()
    start_ms = (ts.index.values.astype("datetime64[ms]").astype(np.int64))
    tsv = ts.values.astype(np.int64)
    inside = int(((tsv >= start_ms) & (tsv < start_ms + 86_400_000)).sum())
    near = int(((tsv >= start_ms) & (tsv < start_ms + 86_400_000 + 60_000)).sum())
    over = tsv - (start_ms + 86_400_000)
    lag = float(over[over >= 0].max() / 1000) if (over >= 0).any() else 0.0
    fd = r["pr"]["fx_date"].dropna()
    fx_ok = bool((pd.to_datetime(fd.values) <= fd.index).all())
    out.append(f"偷看未來的檢查一（時間對齊）：Upbit 每根日線記錄的「最後一筆存下的時間」，{len(ts):,} 根裡 {inside:,} 根在那一天 UTC 0 點到隔天 0 點之間，"
               f"其餘 {near - inside:,} 根在隔天 0 點後 {lag:.0f} 秒以內（可能是存檔延遲，就算是隔天最初幾秒的成交，對日線也幾乎沒影響），"
               f"{len(ts) - near:,} 根更晚；"
               f"每一天用的匯率日期都不晚於那一天：{'是' if fx_ok else '否（有問題）'}。")
    rng = np.random.default_rng(11)
    total = same = 0
    idx = r["close"].index
    for cut in CUTS:
        c = pd.Timestamp(cut)
        d2 = dict(data)
        for key in ("up", "cb"):
            s = data[key]
            d2[key] = s.where(s.index <= c, s * rng.normal(1.0, 0.2, len(s)))
        fxr = data["fx_raw"]
        d2["fx_raw"] = fxr.where(fxr.index <= c, fxr * rng.normal(1.0, 0.2, len(fxr)))
        pr2 = _premiums(d2)
        fl2 = _flags(pr2["kp"], pr2["cbp"])
        t2 = _rules(fl2, idx, r["ma50"])
        m = idx <= c
        for k in ("rk2", "rk3"):
            total += 1
            same += int(r["T"][k][m].fillna(-1).equals(t2[k][m].fillna(-1)))
        md = r["days"] <= c
        for k in ("k1", "k2", "k3"):
            total += 1
            same += int(r["fl"][k][md].fillna(False).equals(fl2[k][md].fillna(False)))
    out.append(f"偷看未來的檢查二：在 {'、'.join(CUTS)} 三個切點，把切點之後的 Upbit、Coinbase 價格和匯率全部換成亂數再算一次，"
               f"切點以前每一天的訊號和進出狀態 {total} 組裡 {same} 組完全沒變" + ("。" if same == total else "（有變動，結果不能用）。"))
    ev, late, lr = r["ev"], r["late"], r["late_rules"]
    bits = "、".join(f"{nm} {_pp(ev[k][MAIN_H]['ex'])} → {_pp(late[k][MAIN_H]['ex'])} 個百分點"
                     for k, nm in (("k1", "K1"), ("k2", "K2"), ("k3", "K3")))
    rb = "、".join(f"{nm} {_num(r['per']['all'][k]['ratio'])} → {_num(lr[k]['ratio'] if lr[k] else None)}"
                   for k, nm in (("ma50", "50 日均線"), ("rk2", "R_K2"), ("rk3", "R_K3")))
    pr_late = _premiums(data, fx_extra=1)
    diff = float((pr_late["kp"] - r["pr"]["kp"]).abs().median() * 100)
    out.append(f"偷看未來的檢查三（再晚一天）：訊號出現後多等一天才開始算，30 天比全部日子多 {bits}；規則再晚一天照做，報酬÷波動 {rb}；"
               f"匯率改用前一個營業日，溢價變動的中位數只有 {diff:.2f} 個百分點。")
    return out


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    problems = data["problems"]
    days, pr, fl, ev, lv = r["days"], r["pr"], r["fl"], r["ev"], r["level"]
    kp, cbp = pr["kp"], pr["cbp"]
    pnames = [p[0] for p in PERIODS]

    # ----- table 1: level -----
    rows1 = []
    e = lv["edges"]
    for i, q in enumerate(lv["quint"]):
        rows1.append([f"第 {i + 1} 組（{'最低' if i == 0 else '最高' if i == 4 else '中間'}）",
                      f"{_pct(e[i])} 到 {_pct(e[i + 1])}", f"{q['days']:,}"]
                     + [_pct(q[h]) for h in H] + [_pct(q["win"], False)])
    rows1.append(["全部日子", "—", f"{int(kp.notna().sum()):,}"] + [_pct(lv["all"][h]) for h in H]
                 + [_pct(lv["all_win"], False)])
    for name in pnames:
        rows1.append([f"相關係數：{name}", "—", "—"] + [_num(lv["corr"][(h, name)]) for h in H] + ["—"])
    rows1.append(["相關係數：全部（90% 區間）", "—", "—"]
                 + [f"{_num(lv['corr'][(h, 'all')])}（{_num(lv['corr'][(h, 'ci')][0])} 到 {_num(lv['corr'][(h, 'ci')][1])}）"
                    for h in H] + ["—"])
    t1 = {"caption": ("檢查一：溢價高低和之後的比特幣。泡菜溢價 ＝ Upbit 韓元價格 ÷（幣安價格 × 美元兌韓元）− 1，每天 UTC 0 點（台灣早上 8 點）"
                      f"三個價格同一刻收盤。{_d(days[0])} 到 {_d(days[-1])} 每天一個樣本，按溢價分成五組（用全期間的分界，是事後分的），"
                      "看之後 7／30／90 天比特幣平均漲跌。相關係數 +1 完全同向、0 沒關係、−1 完全反向；90% 區間用 30 天（90 天那欄 90 天）"
                      "一段的整段重抽 1,000 次。之後的報酬天天重疊，獨立的段數大約只有天數 ÷ 天數期間。"),
          "columns": ["溢價分組", "溢價範圍", "天數", "之後 7 天平均", "之後 30 天平均", "之後 90 天平均", "30 天上漲比例"],
          "rows": rows1}

    # ----- table 2: signals -----
    names = {"k1": "K1 溢價（7 日平均）由負轉正", "k1raw": "K1 每日溢價由負轉正（不平均）",
             "k2": f"K2 溢價 > {K2_LEVEL * 100:.0f}% 且 Coinbase 溢價 < 0", "k3": "K3 溢價高過過去一年的九成"}

    def wcell(x) -> str:
        return f"{_pct(x['mean'])}｜{_pct(x['win'], False)}" if x["n"] else "—"

    rows2 = []
    for k in ("k1", "k1raw", "k2", "k3"):
        x = ev[k]
        m = x[MAIN_H]
        ci = m["ci"]
        ci_t = f"{_pp(m['ex'])}（{_pp(ci[0])} 到 {_pp(ci[1])}）" if ci[0] is not None else _pp(m["ex"])
        rows2.append([names[k], f"{x['days']:,} 天｜{x['indep']} 次", wcell(x[7]), wcell(x[30]), _pct(x[90]["mean"]),
                      ci_t, f"{m['shift'] * 100:.0f}%" if _ok(m["shift"]) else "—",
                      "／".join(_pp(v) for v in m["per"])])
    a7, a30 = ev["k1"][7], ev["k1"][30]
    rows2.append(["全部日子（對照）", f"{int(kp.notna().sum()):,} 天", f"{_pct(a7['all'])}｜{_pct(a7['all_win'], False)}",
                  f"{_pct(a30['all'])}｜{_pct(a30['all_win'], False)}", _pct(ev["k1"][90]["all"]), "—", "—", "—"])
    rows2.append(["CryptoSlate 原文（溢價轉正）", "—", "+1.7%｜67%", "+6.2%｜70%", "—", "—", "—", "—"])
    t2 = {"caption": ("檢查二到四：照網路說法原樣的訊號，訊號當天收盤起算之後的比特幣漲跌（平均｜上漲比例）。「獨立次數」是和上一次至少隔 30 天才算新的一次。"
                      "「30 天比全部日子多」是訊號日之後 30 天平均減全部日子的平均（百分點），括號是 30 天一段整段重抽 1,000 次的 90% 區間；"
                      "「隨機錯開一樣極端」是把訊號日整串循環平移隨機天數 1,000 次，往說法的方向（K1 往上、K2／K3 往下）一樣極端或更極端的比例；"
                      f"最後一欄依序是 {'、'.join(pnames)} 三段各自的「30 天比全部多」。K3 需要先有一年的溢價，從 2018-09 才有值。"),
          "columns": ["訊號", "出現天數｜獨立次數", "之後 7 天：平均｜上漲比例", "之後 30 天：平均｜上漲比例", "之後 90 天平均",
                      "30 天比全部日子多（90% 區間）", "隨機錯開一樣極端", "三段：30 天比全部多"],
          "rows": rows2}

    # ----- table 3: rules -----
    per, grid, plc = r["per"], r["grid"], r["placebo"]
    label = {"hold": "一直抱著", "ma50": "50 日均線（儀表板主訊號）",
             "rk2": f"R_K2：50 日均線，K2 出現後 {K2_DAYS} 天改拿現金", "rk3": "R_K3：50 日均線，K3 亮著時拿現金"}

    def allcell(m) -> str:
        return f"{_num(m['ratio'])}｜{_pct(m['ret'] / 100)}｜{_pct(m['deepest'] / 100)}" if m else "—"

    rows3 = []
    for c in ("hold", "ma50", "rk2", "rk3"):
        g = grid[c]
        if c in ("rk2", "rk3"):
            gt = f"{g['n']} 個裡 {g['text_beat']} 個（中位數 {_num(g['text_med'])}）"
            pt = f"{g['person_beat'] * 100:.0f}%（中位數 {_num(g['person_med'])}）"
            pl = f"{plc[c] * 100:.0f}%"
        else:
            gt, pt, pl = f"中位數 {_num(g['text_med'])}", f"中位數 {_num(g['person_med'])}", "—"
        rows3.append([label[c]] + [_num(per[p][c]["ratio"]) if per[p][c] else "—" for p in pnames]
                     + [allcell(per["all"][c]), gt, pt, pl])
    t3 = {"caption": (f"檢查五：加在 50 日均線上。前三欄是各段的報酬÷波動（越高越好），「全部」是 {RSTART} 起的報酬÷波動｜每年報酬｜最深曾經虧多少；"
                      f"收盤決定、隔天照做、每次買賣扣 {COST * 100:.2f}%，現金 0%。「72 個起算日」是 {GRID[0][:7]} 到 {GRID[1][:7]} 每月 1 日起算到最後一天，"
                      f"報酬÷波動比 50 日均線高的個數；「一般人做法」是再晚一天（下一個收盤才做）、扣 {COST_P * 100:.2f}%、一成進場漏掉，每個起算日模擬 {SIMS} 次，"
                      f"比 50 日均線（同一組亂數）高的比例；「隨機錯位」把韓國溢價整串循環平移隨機天數（{DRAWS} 次，Coinbase 溢價不動）重新做訊號，"
                      "做得和真的一樣好或更好的比例，越小越不像運氣。"),
          "columns": ["做法"] + [f"{p}" for p in pnames] + ["全部：報酬÷波動｜每年｜最深", "72 個起算日：贏 50 日均線",
                                                          "一般人做法：贏 50 日均線", "隨機錯位一樣好"],
          "rows": rows3}

    # ----- table 4: crashes -----
    rows4 = []
    for b in r["crash"]:
        rows4.append([f"{_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）"), _pct(b["drop"]),
                      _pct(b["ma50"], False), _pct(b["rk2"], False), _pct(b["rk3"], False)])
    w = r["whip"]
    rows4.append(["一年下車幾次｜其中 30 天內更高價買回", "—"]
                 + [f"{w[c]['exits']:.1f}｜{w[c]['whips']:.1f}" for c in ("ma50", "rk2", "rk3")])
    rows4.append(["均線想持有、訊號讓它拿現金的天數", "—", "—", f"{r['cash_days']['rk2']:,}", f"{r['cash_days']['rk3']:,}"])
    t4 = {"caption": ("大跌時躲掉多少：比特幣從高點收盤跌 50% 以上的段落（高點在 2017-09 以後），從高點到低點這段跌幅裡規則躲掉的比例"
                      "（用規則自己在這段的漲跌算，中途反覆進出也算進去；100% ＝ 全躲掉、0% ＝ 全吃到）。下車次數從 "
                      f"{RSTART} 算；「更高價買回」是下車後 30 天內又在更高的收盤價買回。"),
          "columns": ["大跌段（高點 → 低點）", "跌幅", "50 日均線躲掉", "R_K2 躲掉", "R_K3 躲掉"], "rows": rows4}

    tables = [t1, t2, t3, t4]

    # ----- pass test -----
    def effect(est, ci, pers) -> dict:
        if not _ok(est):
            return {"pass": False, "same": 0}
        s = 1 if est > 0 else -1
        same = sum(1 for v in pers if _ok(v) and (1 if v > 0 else -1) == s)
        excl = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
        return {"pass": same == 3 and excl, "same": same, "excl": excl, "sign": s}

    eff = {"level": effect(lv["corr"][(MAIN_H, "all")], lv["corr"][(MAIN_H, "ci")],
                           [lv["corr"][(MAIN_H, p)] for p in pnames])}
    for k in ("k1", "k2", "k3"):
        m = ev[k][MAIN_H]
        eff[k] = effect(m["ex"], m["ci"], m["per"])
    rule_ok = {c: bool(per["all"][c] and per["all"][c]["ratio"] > per["all"]["ma50"]["ratio"] and plc[c] <= 0.10)
               for c in ("rk2", "rk3")}
    passed = (eff["k2"]["pass"] and rule_ok["rk2"]) or (eff["k3"]["pass"] and rule_ok["rk3"])

    # ----- today -----
    last = days[-1]
    kp_last = kp.dropna()
    cb_last = cbp.dropna()
    fx_date = pr["fx_date"].iloc[-1]
    k1_dates = ev["k1"]["dates"]
    k2_dates = ev["k2"]["dates"]
    thr = fl["thr"].iloc[-1]
    ma = r["T"]["ma50"].iloc[-1]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(r['close'].iloc[-1]):,.0f} 美元（幣安）；Upbit {float(data['up'].dropna().iloc[-1]):,.0f} 韓元"},
        {"label": "韓國溢價（泡菜溢價）",
         "value": f"{_pct(kp_last.iloc[-1])}（{_d(kp_last.index[-1])}）；7 日平均 {_pct(fl['kp7'].iloc[-1])}；"
                  f"匯率用 {_d(fx_date)} 的 {float(pr['fx'].iloc[-1]):,.2f} 韓元（聯準會每週公布，最近幾天沿用最後一筆）"},
        {"label": "Coinbase 溢價（對幣安）", "value": f"{_pct(cb_last.iloc[-1])}（{_d(cb_last.index[-1])}）"},
        {"label": "K1 溢價 7 日平均由負轉正", "value": (f"最近一次在 {_d(k1_dates[-1])}" if k1_dates else "沒有出現過")
                                              + ("；今天就是" if k1_dates and k1_dates[-1] == last else "")},
        {"label": f"K2 溢價 > {K2_LEVEL * 100:.0f}% 且 Coinbase 溢價 < 0",
         "value": ("亮著" if bool(fl["k2"].iloc[-1]) else "沒亮") + (f"；最近一次在 {_d(k2_dates[-1])}" if k2_dates else "")},
        {"label": "K3 溢價高過過去一年的九成", "value": ("亮著" if bool(fl["k3"].iloc[-1]) else "沒亮")
                                             + (f"（門檻 {_pct(thr)}）" if _ok(thr) else "")},
        {"label": "50 日均線｜R_K2｜R_K3", "value": f"{word(ma)}｜{word(r['T']['rk2'].iloc[-1])}｜{word(r['T']['rk3'].iloc[-1])}"},
        {"label": "這幾格的意思", "value": "只描述溢價和訊號在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    c30 = lv["corr"][(MAIN_H, "all")]
    k1m, k2m, k3m = ev["k1"][MAIN_H], ev["k2"][MAIN_H], ev["k3"][MAIN_H]
    cper = [lv["corr"][(MAIN_H, p)] for p in pnames]
    top, base = lv["quint"][4][MAIN_H], lv["all"][MAIN_H]
    s1 = (f"2017-09 起韓國溢價和之後 30 天比特幣漲跌的相關 {_num(c30)}（三段 {'、'.join(_num(v) for v in cper)}"
          f"；溢價最高兩成之後 30 天平均 {_pct(top)}，全部 {_pct(base)}）"
          + ("，溢價越高之後越弱" if eff["level"]["pass"] and c30 < 0 else "，方向不穩定" if not eff["level"]["pass"] else "")
          + ("，但 2023 起已很弱。" if eff["level"]["pass"] and _ok(cper[-1]) and abs(cper[-1]) < 0.1 else "。"))

    def inside(m) -> str:
        ci = m["ci"]
        return "區間含零" if ci[0] is None or (ci[0] <= 0 <= ci[1]) else "區間不含零"

    s2 = (f"「溢價轉正之後會漲」照原樣測 {ev['k1']['days']} 次，30 天平均 {_pct(k1m['mean'])}、上漲 {k1m['win'] * 100:.0f}%"
          f"（全部日子 {_pct(k1m['all'])}、{k1m['all_win'] * 100:.0f}%），{inside(k1m)}、隨機錯開 {k1m['shift'] * 100:.0f}% 一樣好；"
          f"「溢價過 3% 且美國溢價為負會回檔」{ev['k2']['indep']} 段，之後 30 天比全部日子 {_pp(k2m['ex'])} 個百分點，{inside(k2m)}。")
    rk = per["all"]
    nb = sum(1 for p in pnames if per[p]["rk2"] and per[p]["ma50"] and per[p]["rk2"]["ratio"] > per[p]["ma50"]["ratio"])
    s3 = (f"加在 50 日均線上，R_K2 報酬÷波動 {_num(rk['rk2']['ratio'])}、R_K3 {_num(rk['rk3']['ratio'])}，均線 "
          f"{_num(rk['ma50']['ratio'])}（隨機錯位 {plc['rk2'] * 100:.0f}%、{plc['rk3'] * 100:.0f}%），R_K2 三段只贏 {nb} 段、"
          f"{grid['rk2']['n']} 個起算日 {grid['rk2']['text_beat']} 個贏，事先定的標準" + ("通過。" if passed else "沒有通過。"))
    verdict = s1 + s2 + s3

    # ----- notes -----
    def etext(k) -> str:
        x = eff[k]
        if "excl" not in x:
            return "沒有值"
        return f"三段同方向 {x['same']}／3、全期間 90% 區間{'不含' if x['excl'] else '含'}零 → {'過' if x['pass'] else '不過'}"

    both = pd.concat([kp, pr["kp_cb"]], axis=1).dropna()
    alt_k1 = _flags(pr["kp_cb"], cbp)["k1"].fillna(False)
    up_missing = int(data["up"].isna().sum())
    cb_missing = int(data["cb"].isna().sum())
    notes = [
        "事先寫下的設計（看結果之前，之後沒有改）：1. Level: forward BTC 7/30/90-day returns by premium quintile; correlation; "
        "split 2017-09..2019, 2020..2022, 2023..now; block bootstrap. 2. K1 (CryptoSlate exactly): premium (7-day average, "
        "to avoid daily noise - state this; also report the raw-daily version) crosses from ≤ 0 to > 0 → forward 7/30-day "
        "returns and win rates vs all days and vs random dates; count events and independent events. 3. K2 (CryptoQuant "
        "exactly): premium > 3% AND Coinbase premium < 0 → forward 7/30-day returns vs all days. 4. K3 (froth): premium > "
        "its trailing-1-year 90th percentile → forward 30/90-day returns (top/bubble claim). 5. Rules on top of ma50 (act "
        "at the next close; costs 0.10/0.30%): R_K2 step to cash for 14 days after K2 fires; R_K3 step to cash while K3 is "
        "on. Fair grid where data allows, ordinary-person execution, placebo with circularly shifted premium (200 draws), "
        "crash-avoided share with whipsaws. Pass test (stated now): signal effect same sign in all three periods with "
        "full-sample 90% interval excluding zero AND a rule beats ma50 on return÷volatility with placebo share ≤ 10%.",
        "上面這段的白話：檢查一看溢價高低；K1 照 CryptoSlate「溢價由負轉正」（為了避開每天的雜訊用 7 日平均，也列出不平均的版本）；"
        "K2 照 CryptoQuant「溢價超過 3% 且 Coinbase 溢價為負」；K3「溢價高過自己過去一年的九成」（過熱）；再把 K2、K3 加在 50 日均線上。"
        "主要看 30 天：溢價高低用相關係數，訊號用「訊號日之後 30 天平均 − 全部日子平均」；三段方向要和全期間一樣、全期間 90% 區間不含零，"
        "而且對應的規則要在報酬÷波動上贏 50 日均線、隨機錯位一樣好的比例 ≤ 10%，才算通過。",
        f"過關檢查明細｜溢價高低：{etext('level')}；K1：{etext('k1')}；K2：{etext('k2')}（規則 R_K2 "
        f"{'贏' if rule_ok['rk2'] else '沒贏或隨機錯位太常一樣好'}）；K3：{etext('k3')}（規則 R_K3 {'贏' if rule_ok['rk3'] else '沒贏或隨機錯位太常一樣好'}）。"
        f"整體：{'通過' if passed else '沒有通過'}。",
        f"K1 和「隨便挑一樣多的日子」比：之後 30 天平均一樣高或更高的比例 {_pct(r['rand_k1'], False)}，7 天 {_pct(r['rand_k1_7'], False)}"
        "（不保留訊號擠在一起的特性，所以表裡用循環錯開的版本）。CryptoSlate 沒有公開它用的資料起訖、溢價算法和平均方式，數字可能和這裡不同。",
        "時間對齊：原本以為 Upbit 日線在韓國時間 0 點收，實際查它的 API，日線是韓國時間早上 9 點到隔天 9 點，也就是 UTC 0 點到 UTC 0 點"
        "（candle_date_time_utc 欄位），和幣安、Coinbase 日線同一刻收盤，所以三個價格不用另外拼 1 小時線。匯率是紐約中午的買入匯率，"
        "比收盤早 7 到 8 小時，週末和假日沿用前一個營業日；聯準會每週一才公布上一週的數字，但同樣的匯率當時市場上就看得到。",
        "國際價格用幣安的 BTC 對 USDT（泰達幣，一種跟美元掛鉤的幣），不是真的美元；2018 年前後泰達幣有時偏離 1 美元幾個百分點，"
        f"會讓溢價跟著偏。改用 Coinbase 的美元價格算一次：兩個版本每天的相關 {_num(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 3)}、"
        f"差距中位數 {float((both.iloc[:, 0] - both.iloc[:, 1]).median() * 100):+.2f} 個百分點，K1 轉正 {int(alt_k1.sum())} 次（幣安版 {ev['k1']['days']} 次）。",
        f"資料範圍：Upbit 韓元比特幣日線 {_d(data['up'].first_valid_index())} 到 {_d(data['up'].last_valid_index())}（缺 {up_missing} 天）；"
        f"Coinbase 日線缺 {cb_missing} 天；匯率到 {_d(data['fx_raw'].index[-1])}。只用 Upbit 一家（現在韓國最大）；"
        "2017-09 以前的韓國價格（例如 Bithumb）沒有用，所以 2016–2017 年初那段溢價沒有算進來。",
        "K3 要先累積一年的溢價，2018-09 才有值，所以 2017-12 到 2018-01 溢價衝到最高的那段泡沫，K3 和 R_K3 都沒有參與；"
        "那段時間大跌的躲避比例，R_K3 和 50 日均線一樣。",
        "相關係數和平均用的是重疊的報酬（每天一個樣本、往後看 7 到 90 天），樣本之間不獨立；重抽法用整段 30 天（90 天）一起抽，"
        "已經考慮一部分，但 9 年只有 3 段多頭空頭，結論仍很依賴少數幾段。五組的分界是用全期間的溢價分的，事後才知道。",
        "Makarov & Schoar 說的是價差會持續、會預測「兩個交易所之間」的相對報酬（套利空間），不是比特幣本身的漲跌；"
        "這裡測的是網路上流行的用法（拿溢價猜比特幣方向）。2023 年 Finance Research Letters 的論文說兩者關係隨時間不穩定，和這裡分三段看的做法一致。",
    ]
    notes += _lookahead(data, r)
    notes += [
        "成本很粗略：照課本做每次買賣扣 0.10%、一般人做法 0.30%；現金沒有利息；一直抱著沒有扣成本；沒有算稅。"
        "隨機的部分都用固定的亂數種子，資料沒變的話重跑結果一樣。",
        "過去的結果不保證未來會一樣。這裡只是把流行的說法照原樣測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "kimchi_premium",
        "title": "韓國溢價（泡菜溢價）能看出比特幣方向嗎",
        "question": ("韓國交易所的比特幣比國際貴多少（泡菜溢價），能不能看出之後比特幣的方向？網路流傳的「溢價由負轉正之後會漲」、"
                     "「溢價超過 3% 又碰上美國溢價為負會回檔」照原樣測結果如何？加在 50 日均線上有沒有幫助？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
