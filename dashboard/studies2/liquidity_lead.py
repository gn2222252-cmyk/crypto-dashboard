"""Historical study "liquidity_lead": does bitcoin follow global liquidity / global M2 with a ~10-week (70-day) lag?

The claim (Bitcoin Magazine Pro / Look Into Bitcoin, widely repeated since ~2024): shifting global liquidity forward
10 weeks strengthens its correlation with bitcoin (further optimisation ~56-60 days). Critiques (BIT Academy; a quant
analyst quoted by NewsBTC / TradingView): the best lag drifts 1-3 months across windows, level correlations of two
trending series are weak evidence, the lag is curve-fitted, and bitcoin may lead instead.

Pre-registered design (written before any result was seen; not changed afterwards):
  Signal G = global M2 in USD (composite of what's available, else US M2) - 13-week (~3-month) % change; signal N =
  US net liquidity 13-week % change.
  Claim rule (exactly the popular version, not re-fitted): shift by 70 days; hold BTC when the lagged 13-week change
  is > 0, else cash. Also report fixed lags 0, 28, 56, 70, 84, 112 days as description only.
  Tests:
  1. Correlation of BTC forward 4/8/13-week returns with the signal's 13-week change (changes, not levels), lag 70
     days, by period: 2013-2019, 2020-2023, 2024-2026 (the claim became popular ~2024, so 2024-onward is the closest
     to out-of-sample). Also the reverse direction (does BTC's past return predict liquidity changes?).
  2. Lag stability: in rolling 2-year windows, which lag (0-150 days) maximises correlation - report the spread (if
     it drifts a lot, a fixed lag can't be used).
  3. Trading rule vs holding vs ma50 with main_signal_check's fair grid where possible (monthly starts;
     ordinary-person execution; 0.10/0.30% cost) and as a filter on ma50 (hold only when ma50 AND lagged liquidity
     rising). Placebo: circularly shift the liquidity series by random amounts (200 draws) - share at least as good.
  4. Other assets with the SAME rule: Nasdaq-100 (or QQQ), gold, ETH.
  Pass test (stated now): the claim rule or ma50 filter beats ma50 on return/volatility in all three periods AND
  placebo share <= 10% AND same sign on Nasdaq.
  Today: current G and N 13-week changes, the lagged value the rule uses, and the rule's state - states only.

Details fixed with the design (before results):
  * "Same sign on Nasdaq" = the same version (claim rule or ma50 filter, same signal) also has a higher
    return/volatility than QQQ's own 50-day-average rule over the whole span from 2013. G is the main signal of the
    pass test; N is reported the same way as the popular second variant.
  * Point in time: a value is used only after max(as-of date + lag, conservative release date). Conservative release
    dates: US M2 month end + 30 days; euro-area / Japan / UK money and the composite month end + 45 days; Fed H.4.1
    Wednesday levels the Thursday after. Bitcoin's candle labelled day t closes 00:00 UTC of t+1, so a value released
    on day t is usable for that decision; US shares close before the H.4.1 release, so for QQQ / gold a value must be
    released the day before. The position decided at a close is held from the next day (one more day for the
    ordinary person).
  * 13-week change: weekly data (net liquidity) = this Wednesday / 13 Wednesdays earlier - 1; monthly data (M2) =
    this month / 3 months earlier - 1, dated at the month end.
  * The 70-day "shift" counts from the data's own as-of date (as on the popular charts); if the value is not yet
    published 70 days later (never the case here), it waits for publication.
  * Placebo: the lagged on/off liquidity state (from 2013-01-01) is rolled circularly by a random 180 days to
    (length - 180 days), 200 draws, fixed seed; same for the ma50 filter (ma50 x rolled state).
  * Fair grid: 108 monthly start dates 2015-01 ... 2023-12 measured to the last finished day (as main_signal_check);
    textbook = decide at the close, hold next day, 0.10%; ordinary person = one more day of delay, 0.30%, 10% of
    entries missed at random (robust_trend._missed), 200 simulations, the same random draws for every rule.

Data (FRED's public CSV endpoint did not answer the dashboard's fixed user agent on 2026-11-18, so the same series are
taken from the publishers themselves, all public and key-less, never with another user agent):
  Fed Board Data Download Program packages: H.6 (US M2, SA, monthly), H.4.1 (total assets, Treasury General Account,
  reverse repurchase agreements, Wednesday levels), H.10 (monthly average exchange rates EUR, JPY, GBP, CNY).
  ECB Data Portal (euro-area M2, SA), Bank of Japan time-series API (M2 average outstanding, not SA), Bank of
  England database (M4, SA). China has no key-less machine-readable source here and is left out.
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
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, COST_P = 0.001, 0.003
P_MISS, SIMS, DRAWS = 0.10, 200, 200
SEED = 20240101
YEAR, SHARE_YEAR = 365, 252
LAG = 70
FIXED_LAGS = (0, 28, 56, 70, 84, 112)
SCAN_LAGS = tuple(range(0, 151, 7))  # 0, 7, ..., 147 (contains 28, 56, 70, 84, 112)
HORIZONS = (4, 8, 13)
WEEKS13 = 13
US_RELEASE, OTHER_RELEASE = 30, 45
START = "2013-01-01"
PERIODS = (("2013–2019", "2013-01-01", "2019-12-31"), ("2020–2023", "2020-01-01", "2023-12-31"),
           ("2024–2026", "2024-01-01", None))
GRID = ("2015-01-01", "2023-12-01")
WINDOW_DAYS = 730

FED_ZIP = "https://www.federalreserve.gov/datadownload/Output.aspx?rel={}&filetype=zip"
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/BSI/M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
           "?format=csvdata&startPeriod=2008-01")
BOJ_URL = ("https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en&db=MD02&startDate=200801"
           "&code=MAM1NAM2M2MO")
BOE_URL = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2008"
           "&Dateto=now&SeriesCodes=LPMAUYN&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
FED = {"H6": {"us_m2": ("M2.M", 1e9)},
       "H41": {"assets": ("RESPPA_N.WW", 1e6), "tga": ("RESPPLLDT_N.WW", 1e6), "rrp": ("RESPPLLR_N.WW", 1e6)},
       "H10": {"usd_eur": ("RXI$US_N.M.EU", 1.0), "jpy_usd": ("RXI_N.M.JA", 1.0), "usd_gbp": ("RXI$US_N.M.UK", 1.0)}}
SOURCES = [
    {"name": "Look Into Bitcoin：The Global Liquidity Influence on Bitcoin（全球流動性往後移 10 週的說法）",
     "url": "https://www.lookintobitcoin.com/bitcoin-research/the-global-liquidity-influence-on-bitcoin/"},
    {"name": "Bitcoin Magazine：Global Liquidity Fuels Bitcoin Price",
     "url": "https://bitcoinmagazine.com/markets/global-liquidity-fuels-bitcoin-price"},
    {"name": "BIT Academy：Global M2 decoupling（落後天數會漂移的批評）", "url": "https://www.bit.com/square/global-m2-decoupling"},
    {"name": "NewsBTC／TradingView：Global M2 can't predict bitcoin price, says quant analyst",
     "url": "https://www.tradingview.com/news/newsbtc:9d7b56c3d094b:0-global-m2-can-t-predict-bitcoin-price-says-quant-analyst/"},
    {"name": "美國聯準會 H.6 貨幣存量（M2，季調，月資料；Data Download Program 整包下載）", "url": FED_ZIP.format("H6")},
    {"name": "美國聯準會 H.4.1 資產負債表（總資產、財政部一般帳戶、逆回購，週三數字）", "url": FED_ZIP.format("H41")},
    {"name": "美國聯準會 H.10 匯率（歐元、日圓、英鎊月平均）", "url": FED_ZIP.format("H10")},
    {"name": "歐洲央行 Data Portal：歐元區 M2（季調）", "url": ECB_URL},
    {"name": "日本銀行時間序列 API：M2 平均餘額", "url": BOJ_URL},
    {"name": "英格蘭銀行資料庫：M4（季調，LPMAUYN）", "url": BOE_URL},
    {"name": "FRED（說法原本常用的來源；這次對儀表板固定的 user agent 沒有回應，沒有使用）",
     "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"},
    {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
    {"name": "幣安現貨日線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
    {"name": "Yahoo Finance：QQQ（追蹤那斯達克 100 的基金）、GLD（黃金基金）還原配息日收盤",
     "url": "https://finance.yahoo.com/quote/QQQ/history"},
]

_ok, _pct, _num, _d = te._ok, te._pct, te._num, te._d


# ---------- data ----------

def _stored(name: str, fetch, least: int) -> tuple[dict, str]:
    """fetch() -> {YYYY-MM-DD: number}. A good download replaces our stored copy; a failed one falls back to it."""
    path = common.STORE / f"liquidity_lead_{name}.json"
    note = ""
    try:
        days = fetch()
        if len(days) < least:
            raise RuntimeError(f"只回傳 {len(days)} 筆")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not path.exists():
            raise
        days = json.loads(path.read_text(encoding="utf-8"))
        note = f"這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）"
    return days, note


def _get(url: str, timeout: int = 60) -> requests.Response:
    reply = requests.get(url, headers=common.HEADERS, timeout=timeout)
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    return reply


_FED_CACHE: dict = {}


def _fed(rel: str) -> dict:
    """{key: {YYYY-MM-DD: value}} for the wanted series of one Fed DDP package (downloaded once per run)."""
    if rel not in _FED_CACHE:
        zf = zipfile.ZipFile(io.BytesIO(_get(FED_ZIP.format(rel), timeout=180).content))
        member = next(n for n in zf.namelist() if n.lower().endswith("_data.xml"))
        text = zf.read(member).decode("utf-8")
        out = {}
        for key, (name, mult) in FED[rel].items():
            i = text.find(f'SERIES_NAME="{name}"')
            if i < 0:
                continue
            j = text.find("</kf:Series>", i)
            days = {}
            for value, day in re.findall(r'OBS_VALUE="([^"]+)" TIME_PERIOD="([^"]+)"', text[i:j]):
                try:
                    v = float(value) * mult
                except ValueError:
                    continue
                if math.isfinite(v):
                    days[day[:10]] = v
            out[key] = days
        _FED_CACHE[rel] = out
    return _FED_CACHE[rel]


def _month_end(year: int, month: int) -> str:
    return _d(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0))


def _ecb() -> dict:
    frame = pd.read_csv(io.StringIO(_get(ECB_URL).text))
    out = {}
    for period, value in zip(frame["TIME_PERIOD"], pd.to_numeric(frame["OBS_VALUE"], errors="coerce")):
        if math.isfinite(value):
            y, m = str(period).split("-")[:2]
            out[_month_end(int(y), int(m))] = float(value) * 1e6  # EUR millions
    return out


def _boj() -> dict:
    out = {}
    for line in _get(BOJ_URL).text.splitlines():
        if not line.startswith("MAM1NAM2M2MO,"):
            continue
        _, period, value = line.rsplit(",", 2)
        try:
            v = float(value) * 1e8  # 100 million yen
        except ValueError:
            continue
        if math.isfinite(v) and len(period) == 6:
            out[_month_end(int(period[:4]), int(period[4:]))] = v
    return out


def _boe() -> dict:
    frame = pd.read_csv(io.StringIO(_get(BOE_URL).text))
    out = {}
    for day, value in zip(frame.iloc[:, 0], pd.to_numeric(frame.iloc[:, 1], errors="coerce")):
        if math.isfinite(value):
            ts = pd.Timestamp(str(day))
            out[_month_end(ts.year, ts.month)] = float(value) * 1e6  # GBP millions
    return out


RAW = (  # key, loader, least rows, plain name
    ("us_m2", lambda: _fed("H6")["us_m2"], 200, "美國 M2（聯準會 H.6）"),
    ("assets", lambda: _fed("H41")["assets"], 500, "聯準會總資產（H.4.1）"),
    ("tga", lambda: _fed("H41")["tga"], 500, "財政部一般帳戶（H.4.1）"),
    ("rrp", lambda: _fed("H41")["rrp"], 500, "逆回購（H.4.1）"),
    ("usd_eur", lambda: _fed("H10")["usd_eur"], 200, "歐元匯率（H.10）"),
    ("jpy_usd", lambda: _fed("H10")["jpy_usd"], 200, "日圓匯率（H.10）"),
    ("usd_gbp", lambda: _fed("H10")["usd_gbp"], 200, "英鎊匯率（H.10）"),
    ("ea_m2", _ecb, 150, "歐元區 M2（歐洲央行）"),
    ("jp_m2", _boj, 150, "日本 M2（日本銀行）"),
    ("uk_m4", _boe, 150, "英國 M4（英格蘭銀行）"),
)


def _load() -> tuple[dict, list[str]]:
    raw, problems = {}, []
    for key, fetch, least, name in RAW:
        try:
            days, note = _stored(key, fetch, least)
            raw[key] = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
            if note:
                problems.append(f"{name}{note}。")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}這次取不到（{type(exc).__name__}），沒有算進去，也沒有改用其他方法繞過。")
    return raw, problems


# ---------- the two signals in "as-of" time ----------

def _signals(raw: dict) -> dict:
    """{"g": (change, release, parts), "n": (change, release)}: 13-week % change indexed by as-of date, and the
    conservative release date of each value."""
    out = {}
    us = raw.get("us_m2")
    if us is None:
        raise RuntimeError("美國 M2 取不到，沒有 M2 訊號")
    parts = ["美國"]
    level = us.copy()
    release_days = US_RELEASE
    pieces = []
    if raw.get("ea_m2") is not None and raw.get("usd_eur") is not None:
        pieces.append(("歐元區", raw["ea_m2"] * raw["usd_eur"].reindex(raw["ea_m2"].index)))
    if raw.get("jp_m2") is not None and raw.get("jpy_usd") is not None:
        pieces.append(("日本", raw["jp_m2"] / raw["jpy_usd"].reindex(raw["jp_m2"].index)))
    if raw.get("uk_m4") is not None and raw.get("usd_gbp") is not None:
        pieces.append(("英國", raw["uk_m4"] * raw["usd_gbp"].reindex(raw["uk_m4"].index)))
    if pieces:
        frame = pd.concat([level.rename("us")] + [p.rename(n) for n, p in pieces], axis=1).dropna()
        level = frame.sum(axis=1)
        parts += [n for n, _ in pieces]
        release_days = OTHER_RELEASE
    level = level[level.index >= pd.Timestamp("2008-01-01")]
    change = (level / level.shift(3) - 1).dropna()
    out["g"] = (change, pd.Series(change.index + pd.Timedelta(days=release_days), index=change.index), parts, level)
    if all(raw.get(k) is not None for k in ("assets", "tga", "rrp")):
        frame = pd.concat([raw["assets"], raw["tga"], raw["rrp"]], axis=1).dropna()
        net = frame.iloc[:, 0] - frame.iloc[:, 1] - frame.iloc[:, 2]
        net = net[(net.index >= pd.Timestamp("2008-01-01")) & (net.index.dayofweek == 2)]
        change = (net / net.shift(WEEKS13) - 1).dropna()
        out["n"] = (change, pd.Series(change.index + pd.Timedelta(days=1), index=change.index), None, net)
    return out


def _known(sig: tuple, idx: pd.DatetimeIndex, lag: int, strict: bool, extra: int = 0) -> pd.Series:
    """The value a decision on each day of idx may use: as-of date + lag, and only once released (strict: released
    the day before at the latest). extra = additional days of delay for the look-ahead check."""
    change, release = sig[0], sig[1]
    eff = np.maximum((change.index + pd.Timedelta(days=lag)).values,
                     (release + pd.Timedelta(days=1 if strict else 0)).values)
    s = pd.Series(change.values, index=pd.DatetimeIndex(eff) + pd.Timedelta(days=extra))
    s = s.groupby(level=0).last()
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def _state(known: pd.Series) -> pd.Series:
    return (known > 0).astype(float).where(known.notna())


# ---------- measuring ----------

def _net(target: pd.Series, change: pd.Series, cost: float = COST, delay: int = 1) -> pd.Series:
    pos = target.shift(delay)
    net, _ = te._run(pos.values, change.values, cost)
    return pd.Series(net, index=target.index).where(pos.notna())


def _window(net: pd.Series, a, b) -> pd.Series:
    part = net[(net.index >= pd.Timestamp(a)) & ((net.index <= pd.Timestamp(b)) if b else True)]
    return part.dropna()


def _ratio(net: pd.Series, year: int) -> float | None:
    m = ro._measure(net, year) if len(net) >= 20 else None
    return m["ratio"] if m else None


def _suffix_ratio(net: np.ndarray, starts: np.ndarray, year: int) -> np.ndarray:
    """Return / volatility from each start to the end; net is (days,) or (days, sims)."""
    x = net.reshape(len(net), -1)
    logg = np.log1p(np.clip(x, -0.999999, None))
    s1 = np.cumsum(x[::-1], axis=0)[::-1]
    s2 = np.cumsum((x * x)[::-1], axis=0)[::-1]
    sl = np.cumsum(logg[::-1], axis=0)[::-1]
    n = (len(x) - starts)[:, None].astype(float)
    mean = s1[starts] / n
    var = np.maximum(s2[starts] / n - mean ** 2, 0.0) * n / (n - 1)
    ret = (np.exp(sl[starts] * year / n) - 1) * 100
    vol = np.sqrt(var) * math.sqrt(year) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)


def _run2d(pos: np.ndarray, change: np.ndarray, cost: float) -> np.ndarray:
    """te._run for a (days, sims) position matrix."""
    c = np.nan_to_num(change, nan=0.0)[:, None]
    base = 1 + pos * c
    drift = pos * (1 + c) / np.where(np.abs(base) > 1e-9, base, 1.0)
    prev = np.vstack([np.zeros((1, pos.shape[1])), drift[:-1]])
    return pos * c - cost * np.abs(pos - prev)


def _corr(x: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 20 or both.iloc[:, 0].std() < 1e-12 or both.iloc[:, 1].std() < 1e-12:
        return None, len(both)
    return float(both.iloc[:, 0].corr(both.iloc[:, 1])), len(both)


# ---------- the whole computation ----------

def _btc_part(close: pd.Series, sigs: dict, extra: int = 0) -> dict:
    """Targets and nets of every candidate on bitcoin's daily closes."""
    idx = close.index
    change = close.pct_change().fillna(0.0)
    ma50 = te._signals(close)[0]["A50"]
    known = {k: _known(sigs[k], idx, LAG, False, extra) for k in sigs}
    t = {"hold": pd.Series(1.0, index=idx), "ma50": ma50}
    for k in sigs:
        t[f"{k}_rule"] = _state(known[k])
        t[f"{k}_filt"] = ma50 * _state(known[k])
    return {"idx": idx, "change": change, "known": known, "targets": t,
            "nets": {c: _net(v, change) for c, v in t.items()}}


def _periods(nets: dict, keys: list, year: int) -> dict:
    """Return / volatility of each key in each period and over the whole span, on the days every key has a value."""
    valid = pd.concat([nets[k] for k in keys], axis=1).notna().all(axis=1)
    first = valid[valid].index[0] if valid.any() else None
    out = {}
    for name, a, b in PERIODS + (("all", START, None),):
        a2 = max(pd.Timestamp(a), first) if first is not None else pd.Timestamp(a)
        out[name] = {"start": a2}
        for k in keys:
            part = _window(nets[k], a2, b)
            out[name][k] = ro._measure(part, year) if len(part) >= 20 else None
    return out


def _compute(raw: dict) -> dict:
    sigs = _signals(raw)
    close = te._daily_close()
    btc = _btc_part(close, {k: v for k, v in sigs.items()})
    idx, change, T = btc["idx"], btc["change"], btc["targets"]
    keys = list(T)
    per = _periods(btc["nets"], keys, YEAR)

    # ----- test 1: correlations at lag 70, point in time -----
    weeks = pd.date_range(START, idx[-1], freq="7D")
    corr = {}
    ret13_past = close / close.shift(91) - 1
    for k in sigs:
        x = btc["known"][k].reindex(weeks)
        for h in HORIZONS:
            fwd = (close.shift(-7 * h) / close - 1).reindex(weeks)
            for name, a, b in PERIODS + (("all", START, None),):
                m = (weeks >= pd.Timestamp(a)) & ((weeks <= pd.Timestamp(b)) if b else True)
                corr[(k, h, name)] = _corr(x[m], fwd[m])
        # reverse: bitcoin's past 13-week return vs the liquidity change dated 70 days later (as-of, descriptive)
        chg = sigs[k][0]
        later = chg.reindex(chg.index.union(weeks + pd.Timedelta(days=LAG))).ffill().reindex(weeks + pd.Timedelta(days=LAG))
        later.index = weeks
        later[weeks + pd.Timedelta(days=LAG) > chg.index[-1]] = np.nan
        for name, a, b in PERIODS + (("all", START, None),):
            m = (weeks >= pd.Timestamp(a)) & ((weeks <= pd.Timestamp(b)) if b else True)
            corr[(k, "rev", name)] = _corr(ret13_past.reindex(weeks)[m], later[m])

    # ----- fixed lags (description) -----
    fixed = {}
    for k in sigs:
        for lag in FIXED_LAGS:
            kn = _known(sigs[k], idx, lag, False)
            fwd = (close.shift(-91) / close - 1).reindex(weeks)
            x = kn.reindex(weeks)
            m24 = weeks >= pd.Timestamp("2024-01-01")
            net = _net(_state(kn), change)
            fixed[(k, lag)] = {"all": _corr(x, fwd)[0], "2024": _corr(x[m24], fwd[m24])[0],
                               "ratio": _ratio(_window(net, START, None), YEAR)}

    # ----- test 2: lag stability in rolling two-year windows (as-of, descriptive) -----
    grid = pd.date_range("2010-01-06", idx[-1], freq="7D")
    btc13 = (close / close.shift(91) - 1)
    stab = {}
    ends = list(pd.date_range("2015-03-31", idx[-1], freq="QE")) + [idx[-1]]
    for k in sigs:
        chg = sigs[k][0]
        x_all = chg.reindex(chg.index.union(grid)).ffill().reindex(grid)
        x_all[grid > chg.index[-1] + pd.Timedelta(days=7)] = np.nan
        ys = {L: btc13.reindex(grid + pd.Timedelta(days=L)).values for L in SCAN_LAGS}
        rows = []
        for e in ends:
            res = {}
            for L in SCAN_LAGS:
                yd = grid + pd.Timedelta(days=L)
                m = (yd > e - pd.Timedelta(days=WINDOW_DAYS)) & (yd <= e)
                res[L] = _corr(pd.Series(x_all.values[m]), pd.Series(ys[L][m]))[0]
            good = {L: v for L, v in res.items() if _ok(v)}
            if good:
                best = max(good, key=good.get)
                rows.append({"end": e, "best": best, "best_r": good[best], "r70": res.get(LAG)})
        stab[k] = rows

    # ----- test 3: fair grid, ordinary person, placebo -----
    starts, labels = msc._month_starts(idx, GRID[0], GRID[1])
    cand = ["ma50"] + [f"{k}_{v}" for k in sigs for v in ("rule", "filt")] + ["hold"]
    text_r = {c: _suffix_ratio(np.nan_to_num(btc["nets"][c].values), starts, YEAR)[:, 0] for c in cand}
    person = {}
    rng_draw = np.random.default_rng(SEED).random((len(idx), SIMS))  # the same draws for every rule
    for c in cand:
        w = T[c].shift(2).fillna(0.0).values
        step = np.diff(w, prepend=0.0)
        skip = (step > rt.RISE)[:, None] & (rng_draw < P_MISS)
        move = np.where(skip, step[:, None], 0.0) + np.minimum(step, 0.0)[:, None]
        total = np.cumsum(move, axis=0)
        gap = total - np.minimum(np.minimum.accumulate(total, axis=0), 0.0)
        pos = np.clip(w[:, None] - gap, 0.0, 1.0)
        person[c] = _suffix_ratio(_run2d(pos, change.values, COST_P), starts, YEAR)
    grid_res = {}
    for c in cand:
        grid_res[c] = {"text_beat": int((text_r[c] > text_r["ma50"]).sum()), "n": len(starts),
                       "text_med": float(np.median(text_r[c])),
                       "person_beat": float((person[c] > person["ma50"]).mean()),
                       "person_med": float(np.median(person[c]))}

    rng = np.random.default_rng(SEED + 1)
    win = idx >= pd.Timestamp(START)
    placebo = {}
    for k in sigs:
        state = _state(btc["known"][k])[win]
        n = len(state)
        shifts = rng.integers(180, n - 180, size=DRAWS)
        sub_change = change[win]
        sub_ma = T["ma50"][win]
        real = {"rule": _ratio(_window(btc["nets"][f"{k}_rule"], START, None), YEAR),
                "filt": _ratio(_window(btc["nets"][f"{k}_filt"], START, None), YEAR)}
        better = {"rule": 0, "filt": 0}
        for s in shifts:
            rolled = pd.Series(np.roll(state.values, int(s)), index=state.index)
            for v, tgt in (("rule", rolled), ("filt", sub_ma * rolled)):
                r = _ratio(_net(tgt, sub_change).dropna(), YEAR)
                if _ok(r) and _ok(real[v]) and r >= real[v]:
                    better[v] += 1
        placebo[k] = {v: better[v] / DRAWS for v in better}
        placebo[k]["real"] = real

    # ----- test 4: other assets -----
    other, problems = {}, []
    for name, loader, year, strict in (("QQQ", lambda: ro._share("QQQ"), SHARE_YEAR, True),
                                       ("黃金（GLD）", lambda: ro._share("GLD"), SHARE_YEAR, True),
                                       ("ETH", lambda: ta._crypto_close("ETHUSDT"), YEAR, False)):
        try:
            c = loader()
            c = c[c.index >= pd.Timestamp("2012-06-01")]
            ch = c.pct_change().fillna(0.0)
            mean = c.rolling(50, min_periods=50).mean()
            ma = (c > mean).astype(float).where(mean.notna())
            tg = {"hold": pd.Series(1.0, index=c.index), "ma50": ma}
            for k in sigs:
                st = _state(_known(sigs[k], c.index, LAG, strict))
                tg[f"{k}_rule"], tg[f"{k}_filt"] = st, ma * st
            nets = {kk: _net(v, ch) for kk, v in tg.items()}
            other[name] = _periods(nets, list(tg), year)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} 這次取不到或算不出來（{type(exc).__name__}），沒有比較。")

    return {"sigs": sigs, "close": close, "btc": btc, "per": per, "corr": corr, "fixed": fixed, "stab": stab,
            "grid": grid_res, "grid_starts": labels, "placebo": placebo, "other": other, "problems": problems,
            "keys": keys}


# ---------- look-ahead checks ----------

def _lookahead(raw: dict, r: dict) -> list[str]:
    """(a) one more week of delay; (b) scramble every raw value released after a cut date and check that no decision
    up to that date changes."""
    out = []
    close, sigs = r["close"], r["sigs"]
    late = _btc_part(close, sigs, extra=7)
    bits = []
    for k in sigs:
        a = _ratio(_window(r["btc"]["nets"][f"{k}_rule"], START, None), YEAR)
        b = _ratio(_window(late["nets"][f"{k}_rule"], START, None), YEAR)
        bits.append(f"{'M2' if k == 'g' else '淨流動性'}規則 {_num(a)} → {_num(b)}")
    out.append("偷看未來的檢查一：所有流動性數字再晚一週才用，2013 起報酬÷波動 " + "、".join(bits) + "。")
    rng = np.random.default_rng(7)
    ok, total = 0, 0
    lags = {"us_m2": US_RELEASE, "assets": 1, "tga": 1, "rrp": 1}
    for cut in ("2016-06-30", "2020-03-15", "2024-06-30"):
        c = pd.Timestamp(cut)
        scr = {}
        for key, s in raw.items():
            rel = lags.get(key, OTHER_RELEASE)
            noise = pd.Series(rng.normal(1.0, 0.3, len(s)), index=s.index)
            scr[key] = s.where(s.index + pd.Timedelta(days=rel) <= c, s * noise)
        sg = _signals(scr)
        idx = close.index[close.index <= c]
        for k in sigs:
            a = _state(_known(sigs[k], idx, LAG, False))
            b = _state(_known(sg[k], idx, LAG, False))
            total += 1
            ok += int(a.fillna(-1).equals(b.fillna(-1)))
    out.append(f"偷看未來的檢查二：在 2016-06-30、2020-03-15、2024-06-30 三個切點，把切點之後才公布的原始數字全部換成亂數再算一次，"
               f"切點以前每一天的進出狀態 {total} 組裡 {ok} 組完全沒變" + ("。" if ok == total else "（有變動，表示有偷看，結果不能用）。"))
    return out


# ---------- the study ----------

def study() -> dict:
    raw, problems = _load()
    r = _compute(raw)
    problems += r["problems"]
    sigs, per, corr, grid, plc = r["sigs"], r["per"], r["corr"], r["grid"], r["placebo"]
    has_n = "n" in sigs
    parts = sigs["g"][2]
    g_name = "全球 M2（" + "、".join(parts) + "，換成美元相加）" if len(parts) > 1 else "美國 M2"
    sname = {"g": "M2", "n": "淨流動性"}
    pnames = [p[0] for p in PERIODS]

    # ----- table 1: correlations at lag 70 -----
    def cc(key) -> str:
        v, n = corr.get(key, (None, 0))
        return f"{_num(v)}（{n} 週）" if _ok(v) else "—"

    rows1 = []
    for k in sigs:
        for h in HORIZONS:
            rows1.append([f"{sname[k]} 13 週變化（落後 70 天）→ 比特幣之後 {h} 週報酬"] + [cc((k, h, p)) for p in pnames]
                         + [cc((k, h, "all"))])
        rows1.append([f"反方向：比特幣過去 13 週報酬 → 70 天後的 {sname[k]} 13 週變化"] + [cc((k, "rev", p)) for p in pnames]
                     + [cc((k, "rev", "all"))])
    t1 = {"caption": (f"檢查一：相關係數（+1 完全同向、0 沒關係、−1 完全反向），用的是「變化」不是「水準」。M2 ＝ 市場上的錢的總量，這裡是 {g_name}；"
                      "淨流動性 ＝ 聯準會總資產 − 財政部存在聯準會的錢 − 逆回購（市場上可用的錢多不多的美國版）。"
                      "每週取一個日子，訊號用那天已經公布、而且日期至少是 70 天前的數字。之後 4／8／13 週報酬的期間互相重疊，"
                      "實際上獨立的段數大約只有週數 ÷ 週期；以 13 週那一列粗估，純雜訊也可能出現的範圍約 ±2÷√段數："
                      + "、".join(f"{p} ±{2 / math.sqrt(max(corr.get(('g', 13, p), (None, 13))[1], 13) / 13):.2f}"
                                  for p in pnames) + "。"
                      "反方向那一列是描述用（用到 70 天後才有的數字），看比特幣是不是反而走在前面。"),
          "columns": ["比較"] + pnames + ["2013 起全部"], "rows": rows1}

    # ----- table 2: fixed lags and rolling best lag -----
    fx = r["fixed"]
    rows2 = []
    for lag in FIXED_LAGS:
        row = [f"{lag} 天" + ("（說法的數字）" if lag == LAG else "")]
        for k in ("g", "n"):
            if k in sigs:
                f = fx[(k, lag)]
                row += [f"{_num(f['all'])}｜{_num(f['2024'])}", _num(f["ratio"])]
            else:
                row += ["—", "—"]
        rows2.append(row)
    ref_all = per["all"]
    rows2.append(["對照：50 日均線｜一直抱著（2013 起報酬÷波動）", "—",
                  f"{_num(ref_all['ma50']['ratio'])}｜{_num(ref_all['hold']['ratio'])}", "—",
                  f"{_num(ref_all['ma50']['ratio'])}｜{_num(ref_all['hold']['ratio'])}"])
    stab = r["stab"]
    for k in sigs:
        best = np.array([x["best"] for x in stab[k]])
        if len(best):
            q = np.percentile(best, [0, 25, 50, 75, 100])
            r70 = [x["r70"] for x in stab[k] if _ok(x["r70"])]
            text = (f"{len(best)} 個窗口：最佳落後 最少 {q[0]:.0f}、四分之一 {q[1]:.0f}、中位 {q[2]:.0f}、四分之三 {q[3]:.0f}、最多 {q[4]:.0f} 天；"
                    f"70 天的相關 {_num(min(r70))} 到 {_num(max(r70))}")
            rows2.append([f"滾動兩年窗口（每季底結算一次，{sname[k]}）", text if k == "g" else "—", "—",
                          text if k == "n" else "—", "—"])
    t2 = {"caption": ("檢查二：固定落後天數（只是描述，說法本身只有 70 天）。「相關」是訊號 13 週變化和比特幣之後 13 週報酬的相關，"
                      "照公布時間算、每週一個樣本，前面是 2013 起全部、後面是 2024 起；「照規則」是落後那麼多天的 13 週變化大於 0 就持有比特幣、"
                      "否則現金，2013 起的報酬÷波動（收盤決定、隔天照做、每次買賣扣 0.10%）。最後幾列是滾動兩年窗口："
                      "每個窗口裡掃 0 到 147 天（每 7 天一格），找相關最高的落後天數；這部分用數字本身的日期（不管公布時間），"
                      "是比照網路上圖表的描述做法，相關是訊號 13 週變化和比特幣「落後那麼多天的 13 週報酬」。"),
          "columns": ["落後天數", "M2：相關（全部｜2024 起）", "M2：照規則報酬÷波動",
                      "淨流動性：相關（全部｜2024 起）", "淨流動性：照規則報酬÷波動"], "rows": rows2}

    # ----- table 3: trading on bitcoin and the same rule on other assets -----
    label = {"hold": "一直抱著", "ma50": "50 日均線（儀表板主訊號）", "g_rule": "M2 規則（落後 70 天的 13 週變化 > 0 才持有）",
             "g_filt": "50 日均線 且 M2 規則", "n_rule": "淨流動性規則（同樣做法）", "n_filt": "50 日均線 且 淨流動性規則"}

    def mcell(m) -> str:
        return _num(m["ratio"]) if m else "—"

    def allcell(m) -> str:
        return f"{_num(m['ratio'])}｜{_pct(m['ret'])}｜{_pct(m['deepest'])}" if m else "—"

    rows3 = []
    for c in ("hold", "ma50", "g_rule", "g_filt", "n_rule", "n_filt"):
        if c not in r["keys"]:
            continue
        g = grid.get(c)
        if c in ("hold", "ma50") or not g:
            gtext = f"中位數 {_num(g['text_med'])}" if g else "—"
            ptext = f"中位數 {_num(g['person_med'])}" if g else "—"
            ptxt2 = "—"
        else:
            gtext = f"{g['n']} 個裡 {g['text_beat']} 個（中位數 {_num(g['text_med'])}）"
            ptext = f"{g['person_beat'] * 100:.0f}%（中位數 {_num(g['person_med'])}）"
            k, v = c.split("_")
            ptxt2 = f"{plc[k][v] * 100:.0f}%"
        rows3.append([f"比特幣：{label[c]}"] + [mcell(per[p][c]) for p in pnames] + [allcell(per["all"][c]), gtext, ptext,
                                                                                   ptxt2])
    for name, o in r["other"].items():
        hold = o["all"]["hold"]
        for c in ("ma50", "g_rule", "g_filt"):
            lab = f"{name}：{label[c]}"
            if c == "ma50":
                lab = f"{name}：自己的 50 日均線（一直抱著 {_num(hold['ratio']) if hold else '—'}）"
            rows3.append([lab] + [mcell(o[p][c]) for p in pnames] + [allcell(o["all"][c]), "—", "—", "—"])
    t3 = {"caption": ("檢查三、四：照規則進出。前四欄是報酬÷波動（各段從那一段第一天算；ETH 從幣安有資料、規則有值的那天起），"
                      "「2013 起全部」是報酬÷波動｜每年報酬｜最深曾經虧多少。比特幣：收盤決定、隔天照做、每次買賣扣 0.10%，現金 0%。"
                      f"「108 個起算日」沿用主訊號公平重選的做法：{GRID[0][:7]} 到 {GRID[1][:7]} 每月 1 日起算到最後一天，"
                      "報酬÷波動比 50 日均線高的起算日個數；「一般人做法」是再晚一天、扣 0.30%、一成進場漏掉，每個起算日模擬 200 次，"
                      "比 50 日均線（同樣做法、同一組亂數）高的比例；「隨機錯位」把流動性的持有／不持有狀態整段循環平移隨機天數（200 次），"
                      "做得和真的一樣好或更好的比例，越小表示越不像運氣。QQQ、黃金用 Yahoo 美股交易日收盤，流動性數字要前一天就公布才算數。"),
          "columns": ["做法"] + pnames + ["2013 起全部", "108 個起算日：贏過 50 日均線", "一般人做法：贏過 50 日均線", "隨機錯位一樣好"],
          "rows": rows3[:16]}

    tables = [t1, t2, t3]

    # ----- pass test -----
    def passes(k: str) -> dict:
        res = {}
        q = r["other"].get("QQQ")
        for v in ("rule", "filt"):
            c = f"{k}_{v}"
            beat = [per[p][c] and per[p]["ma50"] and per[p][c]["ratio"] > per[p]["ma50"]["ratio"] for p in pnames]
            nas = bool(q and q["all"].get(c) and q["all"].get("ma50") and q["all"][c]["ratio"] > q["all"]["ma50"]["ratio"])
            res[v] = {"beat": sum(bool(b) for b in beat), "placebo": plc[k][v], "nas": nas,
                      "pass": all(bool(b) for b in beat) and plc[k][v] <= 0.10 and nas}
        return res

    verdicts = {k: passes(k) for k in sigs}
    g_pass = any(x["pass"] for x in verdicts["g"].values())
    n_pass = has_n and any(x["pass"] for x in verdicts["n"].values())

    # ----- today -----
    btc = r["btc"]
    last = btc["idx"][-1]
    now = [{"label": f"最後一根收完的比特幣日線（{_d(last)}）", "value": f"{float(r['close'].iloc[-1]):,.0f} 美元"}]
    for k in sigs:
        chg = sigs[k][0]
        rel = sigs[k][1]
        pub = chg[rel <= last]
        cur = f"{_d(pub.index[-1])} 的 13 週變化 {_pct(pub.iloc[-1] * 100)}（已公布的最新一筆）" if len(pub) else "—"
        kn = btc["known"][k]
        used_date = chg.index[chg.index + pd.Timedelta(days=LAG) <= last]
        used = (f"規則現在用的是 {_d(used_date[-1])} 的 {_pct(kn.iloc[-1] * 100)}" if len(used_date) and _ok(kn.iloc[-1])
                else "—")
        st = btc["targets"][f"{k}_rule"].iloc[-1]
        word = "—" if not _ok(st) else ("持有" if st >= 0.5 else "現金")
        fl = btc["targets"][f"{k}_filt"].iloc[-1]
        fword = "—" if not _ok(fl) else ("持有" if fl >= 0.5 else "現金")
        now.append({"label": g_name if k == "g" else "美國淨流動性（總資產 − 財政部帳戶 − 逆回購）",
                    "value": f"{cur}；{used}，規則狀態：{word}；加上 50 日均線的濾網版：{fword}"})
    ma = btc["targets"]["ma50"].iloc[-1]
    now.append({"label": "50 日均線", "value": "高於（持有）" if ma >= 0.5 else "低於（現金）"})
    now.append({"label": "這幾格的意思", "value": "只描述規則照公布的數字現在停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    def c13(k, p):
        return _num(corr.get((k, 13, p), (None, 0))[0])

    g_beat = verdicts["g"]
    sbest = np.array([x["best"] for x in stab["g"]])
    r24 = corr.get(("g", 13, "2024–2026"), (None, 0))[0]
    s1 = (f"照原樣固定落後 70 天，{'全球 ' if len(parts) > 1 else '美國 '}M2 的 13 週變化和比特幣之後 13 週報酬的相關在 "
          f"2013–19、2020–23、2024 起是 {c13('g', '2013–2019')}、{c13('g', '2020–2023')}、{c13('g', '2024–2026')}"
          + ("（說法流行後反而是反方向" if _ok(r24) and r24 < 0 else "（")
          + (f"{'；' if _ok(r24) and r24 < 0 else ''}淨流動性 {c13('n', '2013–2019')}、{c13('n', '2020–2023')}、"
             f"{c13('n', '2024–2026')}）" if has_n else "）") + "。")
    s2 = (f"滾動兩年看，相關最高的落後天數從 {sbest.min():.0f} 到 {sbest.max():.0f} 天都有（中間一半 "
          f"{np.percentile(sbest, 25):.0f}–{np.percentile(sbest, 75):.0f} 天），並不固定。") if len(sbest) else ""
    gr, gf = per["all"]["g_rule"], per["all"]["g_filt"]
    s3 = (f"拿來進出，M2 規則 2013 起報酬÷波動 {_num(gr['ratio'])}、加在 50 日均線上 {_num(gf['ratio'])}，均線本身 "
          f"{_num(per['all']['ma50']['ratio'])}，三段裡贏均線的分別只有 {g_beat['rule']['beat']} 段和 {g_beat['filt']['beat']} 段"
          f"（隨機錯位一樣好 {plc['g']['rule'] * 100:.0f}%、{plc['g']['filt'] * 100:.0f}%），事先定的過關標準"
          + ("通過" if g_pass else "沒有通過") + ("（淨流動性版也" + ("通過" if n_pass else "沒有通過") + "）。" if has_n else "。"))
    verdict = s1 + s2 + s3

    # ----- notes -----
    def vtext(k):
        x = verdicts[k]
        return "；".join(f"{'規則本身' if v == 'rule' else '加在 50 日均線上'}：三段贏均線 {x[v]['beat']}／3、隨機錯位 "
                        f"{x[v]['placebo'] * 100:.0f}%、QQQ 同方向 {'是' if x[v]['nas'] else '否'} → {'過關' if x[v]['pass'] else '不過關'}"
                        for v in ("rule", "filt"))

    n_other = []
    for name, o in r["other"].items():
        bits = []
        for c in ("n_rule", "n_filt"):
            if o["all"].get(c):
                bits.append(f"{label[c]} {_num(o['all'][c]['ratio'])}")
        if bits:
            n_other.append(f"{name}：" + "、".join(bits) + f"（50 日均線 {_num(o['all']['ma50']['ratio']) if o['all'].get('ma50') else '—'}）")
    g_level = sigs["g"][3]
    notes = [
        "事先寫下的設計（看結果之前，之後沒有改）：訊號 G ＝ 全球 M2 換成美元（能取得的國家加總，取不到就只用美國 M2）的 13 週（約 3 個月）變化；"
        "訊號 N ＝ 美國淨流動性的 13 週變化。說法的規則照原樣、不重新調：往後移 70 天，落後的 13 週變化大於 0 就持有比特幣，否則現金；"
        "另外列出 0、28、56、70、84、112 天只當描述。檢查：一、70 天落後時，訊號變化和比特幣之後 4／8／13 週報酬的相關，分 2013–2019、"
        "2020–2023、2024–2026 三段（說法大約 2024 年才流行，2024 起最接近事後檢驗），也看反方向；二、滾動兩年窗口裡相關最高的落後天數"
        "（0–150 天）漂移多少；三、照規則進出，和一直抱著、50 日均線比，用主訊號公平重選的起算日格子、一般人做法、0.10%／0.30% 成本，"
        "也當 50 日均線的濾網（均線和流動性都向上才持有），並做 200 次隨機錯位對照；四、同一條規則用在 QQQ、黃金、ETH。"
        "過關標準：規則本身或均線濾網版，三段的報酬÷波動都贏 50 日均線，而且隨機錯位一樣好的比例 ≤ 10%，而且在那斯達克（QQQ）上同方向"
        "（同一個版本 2013 起的報酬÷波動也贏 QQQ 自己的 50 日均線）。過關判斷以 M2 版為主，淨流動性版同樣列出。",
        f"過關檢查明細｜M2：{vtext('g')}" + (f"｜淨流動性：{vtext('n')}" if has_n else "") + "。",
        f"資料：{g_name}。美國 M2 用聯準會 H.6（季調、月平均，美元）；歐元區 M2 用歐洲央行（季調、月底餘額，歐元區成員國隨年份增加）；"
        "日本 M2 用日本銀行的平均餘額（沒有季調版的水準數字，所以用未季調的；3 個月變化會帶一點季節起伏）；英國用英格蘭銀行的 M4（季調，英國的廣義貨幣，"
        "不是 M2）；匯率用聯準會 H.10 的月平均。中國沒有不用金鑰、機器可讀的公開來源，所以沒有放進去，網路上的全球 M2 圖多半包含中國，"
        f"這裡的版本和它們不完全一樣。合計的水準從 {_d(g_level.index[0])} 到 {_d(g_level.index[-1])}。",
        "淨流動性：聯準會 H.4.1 的週三數字，總資產 − 財政部一般帳戶 − 逆回購。網路上常用的是 FRED 的 WALCL − WTREGEN − RRPONTSYD；"
        "這裡的逆回購是 H.4.1 的全部逆回購（包含外國官方帳戶，約幾千億美元、變動較慢），不是紐約聯準會每天的隔夜逆回購，13 週變化會有一點差別。",
        "公布時間：每個數字只在保守的公布日之後才用——美國 M2 月底後 30 天、其他國家和合計月底後 45 天、H.4.1 週三數字隔天（週四）。"
        "比特幣「標著某一天」的日線在隔天 UTC 0 點收，當天公布的數字可以用；QQQ、黃金在美國收盤時做決定，數字要前一天就公布。"
        "說法的 70 天從數字本身的日期算起（和網路上的圖一樣），70 天一定晚於公布日，所以公布時間只對 0、28 天那兩列有影響"
        "（M2 要等月底後 45 天才算公布，所以 M2 的 0 天和 28 天實際上一樣）。",
        "FRED 的公開下載這次對儀表板固定的 user agent 沒有回應（2026-11-18 試過，逾時），所以改從原始發布機構（聯準會、歐洲央行、日本銀行、"
        "英格蘭銀行）下載同樣的數字，沒有換 user agent。聯準會說它的 Data Download Program 整包下載會在 2027-03-31 停止，之後要換來源。",
        "這些數字都是今天的版本（之後修正過的），不是當時第一次公布的數字；M2 的修正通常不大，但嚴格來說當時看到的可能略有不同。",
        f"相關性用的是重疊的報酬（每週一個樣本、往後看 4 到 13 週），樣本之間不獨立，相關係數看起來會比實際可信；一段 4 年的期間，13 週報酬只有大約 16 段獨立的。"
        "說法原本比較的是「水準」（兩條都長期往上的線），水準的相關很容易很高卻沒有預測力，所以這裡照事先的設計只看變化。",
        f"一般人做法的模擬用固定的亂數種子、每條規則用同一組亂數；隨機錯位是把 2013 起的持有／不持有狀態整段循環平移 180 天到（長度 − 180 天），"
        f"保留它多久換一次的節奏，只打亂和比特幣價格的對應時間。",
    ]
    if n_other:
        notes.append("淨流動性規則用在其他資產（2013 起報酬÷波動）：" + "；".join(n_other) + "。")
    notes += _lookahead(raw, r)
    notes += [
        "成本很粗略：每次買賣扣 0.10%（一般人做法 0.30%）；現金沒有利息；一直抱著沒有扣成本；沒有算稅。ETH 是今天還在的大幣，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只是整理一個流行說法照原樣做過去的結果、現在停在什麼狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "liquidity_lead",
        "title": "比特幣真的落後全球 M2 約 10 週嗎",
        "question": ("網路上流行「比特幣跟著全球 M2（市場上的錢的總量）走、大約落後 10 週」。照原樣固定 70 天、只用當時已公布的數字，"
                     "分時段還成立嗎？落後天數穩不穩？拿來進出能不能贏過 50 日均線和隨機錯位的對照？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
