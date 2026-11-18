"""Historical study "forced_flows": when traders must buy or sell at known times (liquidation cascades, short
squeezes, token unlocks, index additions and deletions, option and futures expiry), does the price move in a
predictable way that a person could use?

Event definitions, fixed before running (no tuning):
1. Liquidation cascades (bitcoin, daily closes at 00:00 UTC from te._daily_close, events from 2016-01-01, the era of
   perpetual futures): a day that closes 8% or more (and 12% or more) below the day before; and a day on which Bybit
   BTCUSDT open interest counted in coins falls 10% or more (Bybit's key-less endpoint keeps daily history from
   2020-08; Binance's openInterestHist only keeps 30 days). Events within 7 days of a counted one are the same episode.
   Trend state = close above its 50-day average on the day BEFORE the event.
2. Short squeezes: bitcoin closes 8% or more above the day before and the average Binance funding rate of the three
   days before was negative (funding history from 2019-09).
3. Token unlocks: a short list of large, dated unlocks (2% or more of circulating supply) compiled from public news
   reports; each row names its source. Price path -30..+30 days against bitcoin; the same coins on all other days
   as a placebo.
4. Index additions / deletions of bitcoin-linked US shares: dated from public reports; windows of 20 trading days
   against the S&P 500 (SPY) and bitcoin.
5. Monthly / quarterly option expiry (Deribit, last Friday of the month, 08:00 UTC): 7 days before and after, against
   every other Friday; "pinning" measured as the distance of the 08:00 price to the nearest round strike.
6. CME weekend gap (Yahoo BTC=F daily bars): does the futures price return to Friday's close, against a mirror level
   at the same distance on the other side (placebo).

A rule (buy at the close of an event day, hold N days, N = 7 and 30 fixed in advance) is compared with holding, with
the 50-day average rule over the same days, and with random entries of the same count and trend state; costs 0.10%
and 0.30% of the amount traded; split halves; ETH and SOL unchanged. No look-ahead: an event is known at the close
of its own day and the position starts with the next day's return; a truncation check in study() confirms it.
Only public key-less endpoints through the dashboard's helpers. Descriptive and in-sample; not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import studies as old_studies
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import funding_positioning as fp
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

START = "2016-01-01"
DROP, BIG_DROP, OI_DROP, POP = -0.08, -0.12, -0.10, 0.08
GAP_DAYS = 7  # events within 7 days of a counted one belong to the same episode
HORIZONS = (1, 3, 7, 14, 30)
HOLDS = (7, 30)
COSTS = (0.001, 0.003)
SIMS = 1000
SEED = 20160101
YEAR = 365
OI_STORE = common.STORE / "forced_flows_bybit_oi_btc.json"
SURV = common.STORE / "survivorship" / "daily"
BYBIT = "https://api.bybit.com"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
EXPIRY_FROM = "2019-01-01"

# token unlocks: (Binance symbol, name, unlock day UTC, amount, share of circulating supply as reported, source, url)
UNLOCKS = (
    ("OPUSDT", "OP", "2023-05-31", "3.86 億枚", "約 114%（流通量幾乎翻倍）", "The Block",
     "https://www.theblock.co/post/232511/optimism-sui-set-for-large-token-unlocks-worth-a-combined-650-million"),
    ("SUIUSDT", "SUI", "2023-11-03", "3,462 萬枚", "約 3.5%", "FXStreet",
     "https://www.fxstreet.com/cryptocurrencies/news/sui-aptos-and-apecoin-holders-gear-up-for-204-million-token-unlocks-in-november-202311011327"),
    ("APTUSDT", "APT", "2023-11-12", "2,484 萬枚", "約 10%", "FXStreet",
     "https://www.fxstreet.com/cryptocurrencies/news/sui-aptos-and-apecoin-holders-gear-up-for-204-million-token-unlocks-in-november-202311011327"),
    ("ARBUSDT", "ARB", "2024-03-16", "11.1 億枚", "約 77%–87%（各家分母不同）", "CoinDesk／FXStreet",
     "https://www.coindesk.com/markets/2023/08/16/arbitrum-will-unlock-12b-arb-in-march-2024-token-unlocks"),
    ("STRKUSDT", "STRK", "2024-04-15", "6,400 萬枚", "約 8.8%", "FXStreet 等",
     "https://www.fxstreet.com/cryptocurrencies/news/arb-strk-axs-three-token-unlocks-worth-over-350-million-to-watch-out-for-ahead-of-bitcoin-halving-week-202404130600"),
    ("PYTHUSDT", "PYTH", "2024-05-20", "21.3 億枚", "流通量約 15 億枚，超過一倍", "The Block",
     "https://www.theblock.co/post/291313/may-preview-1-25-billion-pyth-unlock-leads-major-token-unlocks-from-avalanche-aptos-arbitrum-and-more"),
    ("TIAUSDT", "TIA", "2024-10-30", "1.76 億枚", "約 80%–82%", "The Block",
     "https://www.theblock.co/news/markets/2024-09-29-october-preview-celestia-set-to-unlock-1-billion-in-tokens-alongside-these-projects-318599"),
    ("TRUMPUSDT", "TRUMP", "2025-04-18", "4,000 萬枚", "流通量增加 20%", "The Block",
     "https://www.theblock.co/post/351077/trump-memecoin-jumps-8-despite-sell-pressure-from-token-unlock"),
    ("PYTHUSDT", "PYTH", "2025-05-20", "21.3 億枚", "約 58.6%", "Invezz",
     "https://invezz.com/news/2025/05/20/pyth-price-sinks-8-as-pyth-networks-280m-token-unlock-looms/"),
)
# index events: (ticker, event, announced (US date, after the close), forced-trade close or None, source url)
INDEX_EVENTS = (
    ("MSTR", "加入那斯達克 100", "2024-12-13", "2024-12-20", "https://www.nasdaq.com/press-release/microstrategy-incorporated-to-join-the-nasdaq-100-index-beginning-december-23-2024-2024-12-13"),
    ("COIN", "加入標普 500", "2025-05-12", "2025-05-16", "https://www.spglobal.com/spdji/en/index-announcements/"),
    ("XYZ", "Block 加入標普 500", "2025-07-18", "2025-07-22", "https://www.spglobal.com/spdji/en/index-announcements/"),
    ("HOOD", "加入標普 500", "2025-09-05", "2025-09-19", "https://www.spglobal.com/spdji/en/index-announcements/"),
    ("MSTR", "MSCI 提議剔除持幣公司", "2025-10-10", None, "https://www.msci.com/"),
    ("MSTR", "MSCI 決定暫不剔除", "2026-01-06", None, "https://www.msci.com/"),
    ("MSTR", "MSCI 公布 11 月剔除", "2026-11-11", "2026-11-30",
     "https://cointelegraph.com/news/msci-confirms-strategys-removal-from-indexes-end-november"),
)
UPCOMING_UNLOCKS = (("ZRO", "2026-11-20", "5.82%"), ("KAITO", "2026-11-20", "7.29%"), ("MBG", "2026-11-22", "6.47%"))

_pct, _d = cr._pct, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v) -> str:
    """A fraction as a signed percent."""
    return _pct(v * 100) if _ok(v) else "—"


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


# ---------- data ----------

def _bybit_oi() -> pd.Series:
    """Bybit BTCUSDT open interest in coins, one value per finished UTC day (value at the end of that day)."""
    stored: dict[str, float] = {}
    try:
        stored = {k: float(v) for k, v in json.loads(OI_STORE.read_text(encoding="utf-8")).items()}
    except Exception:  # noqa: BLE001 - first run or damaged store
        stored = {}
    coins, cursor = {}, None
    pages = 3 if len(stored) > 1500 else 15
    for _ in range(pages):
        params = {"category": "linear", "symbol": "BTCUSDT", "intervalTime": "1d", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        reply = common.get_json(BYBIT + "/v5/market/open-interest", params, timeout=20)
        if reply.get("retCode") != 0:
            raise RuntimeError(f"Bybit 回報錯誤：{reply.get('retMsg')}")
        for r in reply["result"]["list"]:
            coins[int(r["timestamp"])] = float(r["openInterest"])
        cursor = reply["result"].get("nextPageCursor")
        if not cursor or len(reply["result"]["list"]) < 200:
            break
    # the value stamped D 00:00 UTC is the open interest at the end of day D-1
    for ms, v in coins.items():
        stored[_d(pd.Timestamp(ms, unit="ms") - pd.Timedelta(days=1))] = v
    try:
        OI_STORE.write_text(json.dumps(dict(sorted(stored.items()))), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    s = pd.Series(stored, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s[s > 0].sort_index()
    return s[s.index < pd.Timestamp(common.today_utc())]


def _coin_close(symbol: str) -> pd.Series:
    """Daily closes of one coin: the survivorship store if there (read only), else Binance."""
    try:
        rows = json.loads((SURV / f"{symbol}.json").read_text(encoding="utf-8"))["rows"]
        s = pd.Series({k: float(v[0]) for k, v in rows.items() if v and v[0]})
        s.index = pd.to_datetime(s.index)
        s = s[(s > 0) & (s.index < pd.Timestamp(common.today_utc()))].sort_index()
        if len(s) > 60:
            return s.asfreq("D").ffill()
    except Exception:  # noqa: BLE001
        pass
    return ta._crypto_close(symbol)


def _cme() -> pd.DataFrame:
    r = common.get_json(YAHOO.format("BTC=F"), {"range": "10y", "interval": "1d"})["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    f = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
                     index=pd.to_datetime(r["timestamp"], unit="s").normalize(), dtype=float).dropna()
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f[f.index < pd.Timestamp(common.today_utc())]


# ---------- events ----------

def _episodes(mask: pd.Series) -> list:
    out = []
    for day in mask[mask].index:
        if not out or (day - out[-1]).days > GAP_DAYS:
            out.append(day)
    return out


def _state(close: pd.Series) -> pd.Series:
    """1 if yesterday's close was above its 50-day average (the trend the event happened in), else 0."""
    ma = close.rolling(50, min_periods=50).mean()
    return (close > ma).astype(float).where(ma.notna()).shift(1)


def _event_stats(close: pd.Series, days: list, start, rng) -> dict:
    """Forward returns after event days and against random days with the same trend state."""
    state = _state(close)
    out = {"n": len(days)}
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1
        keep = fwd.index >= pd.Timestamp(start)
        ev = [d for d in days if _ok(fwd.get(d)) and _ok(state.get(d))]
        if not ev:
            out[h] = None
            continue
        vals = np.array([fwd[d] for d in ev])
        pools = {s: fwd[keep & (state == s) & fwd.notna()].values for s in (0.0, 1.0)}
        st = np.array([state[d] for d in ev])
        matched = float(np.mean([pools[s].mean() for s in st]))
        sims = np.zeros(SIMS)
        for s in (0.0, 1.0):
            k = int((st == s).sum())
            if k:
                sims += rng.choice(pools[s], size=(SIMS, k)).sum(axis=1)
        sims /= len(ev)
        out[h] = {"n": len(ev), "mean": float(vals.mean()), "median": float(np.median(vals)),
                  "up": float((vals > 0).mean()), "matched": matched,
                  "beat": float((sims >= vals.mean()).mean())}
    return out


def _all_days(close: pd.Series, start) -> dict:
    out = {"n": None}
    for h in HORIZONS:
        fwd = (close.shift(-h) / close - 1)[close.index >= pd.Timestamp(start)].dropna()
        out[h] = {"n": len(fwd), "mean": float(fwd.mean()), "median": float(fwd.median()),
                  "up": float((fwd > 0).mean()), "matched": float(fwd.mean()), "beat": None}
    return out


# ---------- rules ----------

def _position(index: pd.DatetimeIndex, event_days, n: int) -> np.ndarray:
    """1 while holding: the returns of the n days after each event close."""
    pos = np.zeros(len(index))
    where = index.get_indexer(pd.DatetimeIndex(list(event_days)))
    for i in where[where >= 0]:
        pos[i + 1:i + 1 + n] = 1.0
    return pos


def _stats(net: np.ndarray) -> tuple:
    eq = np.cumprod(1 + net)
    years = len(net) / YEAR
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else None
    deep = float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min())
    return cagr, deep


def _rule(close: pd.Series, event_days, n: int, start) -> dict:
    change = close.pct_change().fillna(0.0).values
    s = int(close.index.searchsorted(pd.Timestamp(start)))
    pos = _position(close.index, event_days, n)[s:]
    ch = change[s:]
    out = {"exposure": float(pos.mean()), "entries": int(((pos[1:] > 0) & (pos[:-1] == 0)).sum() + (pos[0] > 0))}
    for c in COSTS:
        net, _ = te._run(pos, ch, c)
        out[c] = _stats(net)
    net, _ = te._run(pos, ch, COSTS[0])
    half = len(net) // 2
    out["halves"] = (_stats(net[:half])[0], _stats(net[half:])[0])
    out["hold"] = _stats(ch)[0]
    ma = close.rolling(50, min_periods=50).mean()
    ma_pos = (close > ma).astype(float).shift(1).fillna(0.0).values[s:]
    out["ma50"] = _stats(te._run(ma_pos, ch, COSTS[0])[0])[0]
    out["pos"], out["s"] = pos, s
    return out


def _random_rule(close: pd.Series, rule: dict, entries: list, n: int, rng, sims: int = 400) -> dict:
    """Random entry days with the same count and trend state, same holding length, cost 0.10%."""
    s = rule["s"]
    state = _state(close).values[s:]
    change = close.pct_change().fillna(0.0).values[s:]
    idx = close.index[s:]
    st = [state[idx.get_loc(d)] for d in entries if d in idx and _ok(state[idx.get_loc(d)])]
    pools = {v: np.where(state == v)[0] for v in (0.0, 1.0)}
    pools = {v: p[p < len(state) - 1] for v, p in pools.items()}
    cagrs = []
    for _ in range(sims):
        pos = np.zeros(len(state))
        for v in st:
            i = int(rng.choice(pools[v]))
            pos[i + 1:i + 1 + n] = 1.0
        cagrs.append(_stats(te._run(pos, change, COSTS[0])[0])[0])
    cagrs = np.array([c for c in cagrs if _ok(c)])
    mine = rule[COSTS[0]][0]
    return {"median": float(np.median(cagrs)) if len(cagrs) else None,
            "beat": float((cagrs >= mine).mean()) if len(cagrs) and _ok(mine) else None}


# ---------- unlocks and index events ----------

def _at(s: pd.Series, day) -> float | None:
    v = s.get(pd.Timestamp(day))
    return float(v) if _ok(v) else None


def _ratio(s, a, b):
    x, y = _at(s, a), _at(s, b)
    return y / x - 1 if x and y else None


def _unlock_rows(btc: pd.Series) -> tuple[list, dict, list]:
    rows, before, after, around, problems = [], [], [], [], []
    trend_gap = []
    placebo_b = []
    coins: dict[str, pd.Series] = {}
    for sym, name, day, amount, share, src, _url in UNLOCKS:
        try:
            s = coins.get(sym)
            if s is None:
                s = coins[sym] = _coin_close(sym)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} 的日線這次取不到（{type(exc).__name__}）。")
            continue
        d = pd.Timestamp(day)
        a, b, c, e = d - pd.Timedelta(days=31), d - pd.Timedelta(days=1), d + pd.Timedelta(days=1), d + pd.Timedelta(days=30)
        pre, pre_b = _ratio(s, a, b), _ratio(btc, a, b)
        mid, mid_b = _ratio(s, b, c), _ratio(btc, b, c)
        post, post_b = _ratio(s, b, e), _ratio(btc, b, e)
        x = lambda r, rb: (r - rb) if _ok(r) and _ok(rb) else None  # noqa: E731
        if _ok(x(pre, pre_b)):
            before.append(x(pre, pre_b))
        if _ok(x(post, post_b)):
            after.append(x(post, post_b))
        if _ok(x(mid, mid_b)):
            around.append(x(mid, mid_b))
        # the coin's own trend: its 30-day return in the 60 days before the window (per 30 days)
        prior = _ratio(s, d - pd.Timedelta(days=91), a)
        if _ok(prior) and _ok(pre):
            trend_gap.append(pre - ((1 + prior) ** 0.5 - 1))
        cell = lambda r, rb: "—" if not _ok(r) else f"{_p(r)}（{_p(x(r, rb))}）"  # noqa: E731
        rows.append([f"{name}（代幣解鎖）", day, f"{amount}；{share}；{src}", cell(pre, pre_b), cell(mid, mid_b),
                     cell(post, post_b), "比特幣"])
    # placebo: the same coins on every other day (30-day return minus bitcoin's), away from listed unlocks
    for sym, s in coins.items():
        days = [pd.Timestamp(u[2]) for u in UNLOCKS if u[0] == sym]
        r = (s.shift(-30) / s - 1) - (btc.shift(-30) / btc - 1).reindex(s.index)
        r = r.iloc[60:].dropna()
        far = np.ones(len(r), dtype=bool)
        for d in days:
            far &= np.abs((r.index - d).days) > 45
        if far.sum() > 30:
            placebo_b.append(float(r[far].mean()))
    summary = {"before": before, "after": after, "around": around, "trend_gap": trend_gap,
               "placebo": float(np.mean(placebo_b)) if placebo_b else None, "n_placebo": len(placebo_b)}
    return rows, summary, problems


def _index_rows(btc: pd.Series) -> tuple[list, dict, list]:
    problems, shares = [], {}
    for sym in {e[0] for e in INDEX_EVENTS} | {"SPY"}:
        try:
            s = old_studies._yahoo(sym).astype(float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            shares[sym] = s[s.index < pd.Timestamp(common.today_utc())]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Yahoo {sym} 這次取不到（{type(exc).__name__}）。")
    rows, stats = [], {"day1": [], "pre": [], "run": [], "post": [], "btc20": []}
    spy = shares.get("SPY")
    if spy is None:
        return rows, stats, problems
    for sym, what, ann, trade, _url in INDEX_EVENTS:
        s = shares.get(sym)
        if s is None:
            continue
        idx = s.index
        i0 = int(idx.searchsorted(pd.Timestamp(ann) + pd.Timedelta(days=1)))  # first session after the news
        if i0 >= len(idx):
            continue
        last = len(idx) - 1

        def rel(i, j):
            if i < 0 or j > last or j <= i:
                return None, None
            a, b = idx[i], idx[j]
            r = s.iloc[j] / s.iloc[i] - 1
            sp = spy.asof(b) / spy.asof(a) - 1
            return float(r), float(r - sp)

        pre = rel(i0 - 21, i0 - 1)
        day1 = rel(i0 - 1, i0)
        it = int(idx.searchsorted(pd.Timestamp(trade))) if trade else None
        run = rel(i0 - 1, it) if trade and it is not None and it <= last and idx[it] == pd.Timestamp(trade) else (None, None)
        post = rel(it, it + 20) if trade and it is not None and it + 20 <= last else (None, None)
        if trade is None:
            post = rel(i0 - 1, i0 + 19)
        j = min(i0 + 19, last)
        b20 = None
        if i0 + 19 <= last:
            r = s.iloc[i0 + 19] / s.iloc[i0 - 1] - 1
            rb = _ratio(btc, idx[i0 - 1], idx[i0 + 19])
            b20 = float(r - rb) if _ok(rb) else None
        for k, v in (("pre", pre[1]), ("day1", day1[1]), ("run", run[1]), ("post", post[1]), ("btc20", b20)):
            if _ok(v):
                stats[k].append((sym, what, v))
        cell = lambda t: "—" if not _ok(t[0]) else f"{_p(t[0])}（{_p(t[1])}）"  # noqa: E731
        when = f"{ann}" + (f"→{trade}" if trade else "")
        last_note = "" if j == i0 + 19 else "（還沒滿 20 天）"
        rows.append([f"{sym}（{what}）", when, "；".join(x for x in (
            f"公布隔天 {cell(day1)}", f"公布後 20 天減比特幣 {_p(b20)}" if _ok(b20) else "") if x) + last_note,
            cell(pre), cell(run), cell(post), "標普 500（SPY）"])
    return rows, stats, problems


# ---------- expiry and CME gap ----------

def _last_fridays(first, last) -> list:
    out = []
    for m in pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="MS"):
        end = m + pd.offsets.MonthEnd(0)
        out.append(end - pd.Timedelta(days=(end.weekday() - 4) % 7))
    return out


def _expiry(grid: np.ndarray, b0: pd.Timestamp) -> dict:
    def at(ts):
        k = int((ts - b0) / pd.Timedelta(hours=1))
        return float(grid[k]) if 0 <= k < len(grid) else None

    last_day = b0 + pd.Timedelta(hours=len(grid) - 1)
    expiries = [f for f in _last_fridays(EXPIRY_FROM, last_day) if f + pd.Timedelta(hours=8) <= last_day]
    fridays = pd.date_range(pd.Timestamp(EXPIRY_FROM), last_day, freq="W-FRI")
    exp_set = set(expiries)

    def measure(days):
        pre, post, swing, dist = [], [], [], []
        for f in days:
            t = f + pd.Timedelta(hours=8)
            p0, pb, pa, p24 = at(t), at(t - pd.Timedelta(days=7)), at(t + pd.Timedelta(days=7)), at(t - pd.Timedelta(hours=24))
            if p0 and pb:
                pre.append(p0 / pb - 1)
            if p0 and pa and t + pd.Timedelta(days=7) <= last_day:
                post.append(pa / p0 - 1)
            if p0 and p24:
                swing.append(abs(p0 / p24 - 1))
            if p0:
                g = 1000 if p0 >= 20000 else 500 if p0 >= 5000 else 250
                dist.append(abs(p0 / g - round(p0 / g)))
        return {k: np.array(v) for k, v in (("pre", pre), ("post", post), ("swing", swing), ("dist", dist))}

    month = measure(expiries)
    quarter = measure([f for f in expiries if f.month in (3, 6, 9, 12)])
    others = [f for f in fridays if f not in exp_set]
    other = measure(others)
    half = len(expiries) // 2
    halves = (measure(expiries[:half])["pre"], measure(expiries[half:])["pre"])
    return {"month": month, "quarter": quarter, "other": other, "n": len(expiries), "first": expiries[0],
            "halves": halves}


def _hourly_readonly(symbol: str) -> tuple[np.ndarray, pd.Timestamp]:
    """Hourly grid from the stores of seasonality / clock_robustness, read only; missing hours fetched into memory."""
    rows = cr._read(cr.SHARED_STORE) if symbol == "BTCUSDT" else {}
    if len(rows) < cr.MIN_HOURS:
        rows = cr._read(cr._own_store(symbol))
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    if not rows or max(rows) < today_ms - cr.HOUR_MS:
        try:
            cr._fetch(symbol, rows, keep=False)  # keep=False: never writes another module's store
        except Exception:  # noqa: BLE001 - the stored hours are still usable
            pass
    grid, b0, _ = cr._grid(rows)
    return grid, b0


def _placebo_low(event: np.ndarray, pool: np.ndarray, rng) -> float | None:
    """Share of random sets of other Fridays (same count) whose average is as low as the event average or lower."""
    if not len(event) or not len(pool):
        return None
    sims = rng.choice(pool, size=(SIMS, len(event))).mean(axis=1)
    return float((sims <= event.mean()).mean())


def _cme_gap(cme: pd.DataFrame) -> dict:
    out = {"events": 0, "fill": {}, "mirror": {}}
    idx = cme.index
    rows = []
    for i in range(1, len(cme)):
        if (idx[i] - idx[i - 1]).days < 2:
            continue
        prev_close, op = cme["close"].iloc[i - 1], cme["open"].iloc[i]
        gap = op / prev_close - 1
        if abs(gap) < 0.01:
            continue
        rows.append((i, prev_close, op, gap))
    out["events"] = len(rows)
    for k in (1, 3, 5, 10):
        fill, mirror, n = 0, 0, 0
        for i, pc, op, gap in rows:
            if i + k - 1 >= len(cme):
                continue
            seg = cme.iloc[i:i + k]
            n += 1
            ghost = op + (op - pc)
            if gap > 0:
                fill += seg["low"].min() <= pc
                mirror += seg["high"].max() >= ghost
            else:
                fill += seg["high"].max() >= pc
                mirror += seg["low"].min() <= ghost
        out["fill"][k] = fill / n if n else None
        out["mirror"][k] = mirror / n if n else None
        out["n"] = n
    out["first"] = idx[0]
    out["median_gap"] = float(np.median([abs(r[3]) for r in rows])) if rows else None
    return out


def _expiry_rule(close: pd.Series, start, rng, sims: int = 400) -> dict:
    """Hold every day except the 7 daily returns that end at 00:00 UTC on each month's last Friday (known by
    calendar in advance). Random version: the same 7-day gap before another Friday of the same month."""
    close = close[close.index >= pd.Timestamp(start)]
    change = close.pct_change().fillna(0.0).values
    idx = close.index

    def pos_for(fridays):
        pos = np.ones(len(idx))
        for f in fridays:
            a, b = idx.searchsorted(f - pd.Timedelta(days=7)), idx.searchsorted(f)
            pos[a:b] = 0.0
        return pos

    months = pd.date_range(idx[0].to_period("M").to_timestamp(), idx[-1], freq="MS")
    lasts, choices = [], []
    for m in months:
        fr = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")
        lasts.append(fr[-1])
        choices.append(list(fr[:-1]))
    pos = pos_for(lasts)
    out = {"exposure": float(pos.mean()), "hold": _stats(change)[0]}
    for c in COSTS:
        out[c] = _stats(te._run(pos, change, c)[0])
    net = te._run(pos, change, COSTS[0])[0]
    half = len(net) // 2
    out["halves"] = (_stats(net[:half])[0], _stats(net[half:])[0])
    rnd = []
    for _ in range(sims):
        pick = [ch[int(rng.integers(len(ch)))] for ch in choices if ch]
        rnd.append(_stats(te._run(pos_for(pick), change, COSTS[0])[0])[0])
    rnd = np.array([v for v in rnd if _ok(v)])
    out["random"] = float(np.median(rnd)) if len(rnd) else None
    out["beat"] = float((rnd >= out[COSTS[0]][0]).mean()) if len(rnd) else None
    return out


# ---------- look-ahead check ----------

def _lookahead(close: pd.Series) -> bool:
    """Events and positions found on data cut at a past date equal those found on the full data up to that date."""
    cut = pd.Timestamp("2022-06-30")
    def events(c):
        return _episodes((c.pct_change() <= DROP) & (c.index >= pd.Timestamp(START)))
    full, part = events(close), events(close[close.index <= cut])
    if [d for d in full if d <= cut] != part:
        return False
    p_full = _position(close.index, full, 7)[: int(close.index.searchsorted(cut)) + 1]
    short = close[close.index <= cut]
    p_part = _position(short.index, part, 7)
    return bool(np.array_equal(p_full, p_part)) and bool(np.array_equal(_state(close)[close.index <= cut].fillna(-1).values,
                                                                       _state(short).fillna(-1).values))


# ---------- the study ----------

def _cell(st) -> str:
    return "—" if not st else f"平均 {_p(st['mean'])}，漲 {_share(st['up'])}"


def study() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    notes_extra: list[str] = []
    btc = te._daily_close()
    asof = btc.index[-1]
    ret = btc.pct_change()
    in_era = btc.index >= pd.Timestamp(START)
    state = _state(btc)

    # 1. cascades
    drop8 = _episodes((ret <= DROP) & in_era)
    drop12 = _episodes((ret <= BIG_DROP) & in_era)
    up8 = [d for d in drop8 if state.get(d) == 1.0]
    dn8 = [d for d in drop8 if state.get(d) == 0.0]
    oi_days, oi_first = [], None
    try:
        oi = _bybit_oi()
        oi_first = oi.index[0]
        oi_ch = oi.pct_change()
        oi_ch = oi_ch[(oi.index.to_series().diff().dt.days == 1)]
        oi_days = _episodes(oi_ch.reindex(btc.index).fillna(0) <= OI_DROP)
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"Bybit 未平倉量歷史這次取不到（{type(exc).__name__}），只用價格跌幅定義爆倉。")
    # 2. squeezes
    sq, pops = [], []
    fund_first = None
    try:
        funding = fp._funding_history()
        fd = funding.resample("1D").mean()
        fund_first = fd.index[0]
        before = fd.rolling(3, min_periods=3).mean().shift(1).reindex(btc.index)
        pop_mask = (ret >= POP) & (btc.index >= fd.index[0] + pd.Timedelta(days=3))
        sq = _episodes(pop_mask & (before < 0))
        pops = _episodes(pop_mask & (before >= 0))
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"幣安資金費率歷史這次取不到（{type(exc).__name__}），軋空部分留空。")

    base = _all_days(btc, START)
    ev = {"drop8": _event_stats(btc, drop8, START, rng), "drop12": _event_stats(btc, drop12, START, rng),
          "up8": _event_stats(btc, up8, START, rng), "dn8": _event_stats(btc, dn8, START, rng)}
    if oi_days:
        ev["oi"] = _event_stats(btc, oi_days, oi_first, rng)
    if fund_first is not None:
        ev["sq"] = _event_stats(btc, sq, fund_first, rng)
        ev["pop"] = _event_stats(btc, pops, fund_first, rng)
    coins = {}
    for sym, cn in (("ETHUSDT", "以太幣"), ("SOLUSDT", "索拉納")):
        try:
            c = ta._crypto_close(sym)
            r = c.pct_change()
            start = c.index[0] + pd.Timedelta(days=60)
            days = _episodes((r <= DROP) & (c.index >= start))
            coins[sym] = (cn, c, days, start)
            ev[sym] = _event_stats(c, days, start, rng)
        except Exception as exc:  # noqa: BLE001
            notes_extra.append(f"{cn}日線這次取不到（{type(exc).__name__}）。")

    def ev_row(label, st, n_text):
        cells = [_cell(st[h]) if st.get(h) else "—" for h in (1, 3, 7, 14, 30)]
        s30 = st.get(30)
        last = "—" if not s30 or s30.get("beat") is None else (
            f"{_p(s30['mean'] - s30['matched'])}；隨機 {_share(s30['beat'])} 比它好")
        return [label, n_text] + cells + [last]

    t1_rows = [ev_row("全部日子（對照）", base, "每天")]
    t1_rows.append(ev_row("單日跌 8% 以上", ev["drop8"], f"{len(drop8)} 段"))
    t1_rows.append(ev_row("　其中發生在上升趨勢", ev["up8"], f"{len(up8)} 段"))
    t1_rows.append(ev_row("　其中發生在下降趨勢", ev["dn8"], f"{len(dn8)} 段"))
    t1_rows.append(ev_row("單日跌 12% 以上", ev["drop12"], f"{len(drop12)} 段"))
    if "oi" in ev:
        t1_rows.append(ev_row(f"未平倉量單日減 10% 以上（{oi_first.year} 起）", ev["oi"], f"{len(oi_days)} 段"))
    if "sq" in ev:
        t1_rows.append(ev_row(f"軋空：單日漲 8% 以上、前 3 天資金費率為負（{fund_first.year} 起）", ev["sq"], f"{len(sq)} 段"))
        t1_rows.append(ev_row("單日漲 8% 以上、資金費率不為負（對照）", ev["pop"], f"{len(pops)} 段"))
    for sym, (cn, c, days, start) in coins.items():
        t1_rows.append(ev_row(f"{cn}單日跌 8% 以上（{start.year} 起）", ev[sym], f"{len(days)} 段"))

    # rules
    t2_rows, rules = [], {}

    def rule_row(label, close, days, n, start):
        r = _rule(close, days, n, start)
        rnd = _random_rule(close, r, days, n, rng)
        rules[label] = (r, rnd)
        c1, c3 = r[COSTS[0]], r[COSTS[1]]
        t2_rows.append([label, f"{len(days)} 次，持有 {_share(r['exposure'])} 的日子",
                        f"{_p(c1[0])}／{_p(c3[0])}", _p(c1[1]),
                        f"抱著 {_p(r['hold'])}／均線 {_p(r['ma50'])}",
                        f"{_p(r['halves'][0])}／{_p(r['halves'][1])}",
                        "—" if rnd["median"] is None else f"中位 {_p(rnd['median'])}；{_share(rnd['beat'])} 比它好"])
        return r

    for n in HOLDS:
        rule_row(f"比特幣跌 8% 以上收盤買、抱 {n} 天", btc, drop8, n, START)
    rule_row("比特幣跌 12% 以上收盤買、抱 30 天", btc, drop12, 30, START)
    if oi_days:
        rule_row("未平倉量減 10% 以上收盤買、抱 7 天", btc, oi_days, 7, oi_first)
    if fund_first is not None and sq:
        rule_row("軋空當天收盤買、抱 7 天", btc, sq, 7, fund_first)
    for sym, (cn, c, days, start) in coins.items():
        for n in HOLDS:
            rule_row(f"{cn}跌 8% 以上收盤買、抱 {n} 天", c, days, n, start)
    exp_rules = {}
    for label, c, start in (("比特幣", btc, EXPIRY_FROM),) + tuple(
            (cn, cc, max(st, pd.Timestamp(EXPIRY_FROM))) for _s, (cn, cc, _dd, st) in coins.items()):
        r = _expiry_rule(c, start, rng)
        exp_rules[label] = r
        t2_rows.append([f"{label}：月結算前 7 天空手、其他時間抱著", f"每月 1 次，持有 {_share(r['exposure'])} 的日子",
                        f"{_p(r[COSTS[0]][0])}／{_p(r[COSTS[1]][0])}", _p(r[COSTS[0]][1]),
                        f"抱著 {_p(r['hold'])}／均線 —", f"{_p(r['halves'][0])}／{_p(r['halves'][1])}",
                        f"換成同月其他週五：中位 {_p(r['random'])}；{_share(r['beat'])} 比它好"])

    # 3 + 4. unlocks and index events
    u_rows, u_sum, probs = _unlock_rows(btc)
    notes_extra += probs
    i_rows, i_sum, probs = _index_rows(btc)
    notes_extra += probs

    # 5 + 6. expiry and CME
    t4_rows, exp, gap = [], None, None
    try:
        exp = _expiry(*_hourly_readonly("BTCUSDT"))
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"比特幣 1 小時線這次取不到（{type(exc).__name__}），到期日部分留空。")
    if exp:
        o = exp["other"]

        def trio(a):
            return (f"平均 {_p(a.mean())}／中位 {_p(float(np.median(a)))}／漲 {_share((a > 0).mean())}"
                    if len(a) else "—")
        for key, name in (("month", "月結算"), ("quarter", "季結算")):
            m = exp[key]
            for part, word in (("pre", "前 7 天"), ("post", "後 7 天")):
                a = m[part]
                low = _placebo_low(a, o[part], rng)
                extra = (f"；前半 {_p(exp['halves'][0].mean())}／後半 {_p(exp['halves'][1].mean())}"
                         if key == "month" and part == "pre" else "")
                t4_rows.append([f"比特幣{name}{word}", f"{len(a)} 次", _p(a.mean()),
                                _p(float(np.median(a))), _share((a > 0).mean()), trio(o[part]),
                                f"隨機挑同樣多個其他週五，平均這麼低或更低的比例 {_share(low)}{extra}"])
        exp_coins = {}
        for sym, cn in (("ETHUSDT", "以太幣"), ("SOLUSDT", "索拉納")):
            try:
                e = _expiry(*_hourly_readonly(sym))
                exp_coins[sym] = e
                a = e["month"]["pre"]
                t4_rows.append([f"{cn}月結算前 7 天（同一規則）", f"{len(a)} 次", _p(a.mean()), _p(float(np.median(a))),
                                _share((a > 0).mean()), trio(e["other"]["pre"]),
                                f"隨機挑同樣多個其他週五，平均這麼低或更低的比例 {_share(_placebo_low(a, e['other']['pre'], rng))}"])
            except Exception as exc:  # noqa: BLE001
                notes_extra.append(f"{cn} 1 小時線這次取不到（{type(exc).__name__}），沒有做換幣檢查。")
        m = exp["month"]
        t4_rows.append(["月結算前 24 小時的漲跌幅度（不分方向）", f"{len(m['swing'])} 次", _pct(m["swing"].mean() * 100, False),
                        _pct(float(np.median(m["swing"])) * 100, False), "—",
                        f"其他週五：平均 {_pct(o['swing'].mean() * 100, False)}／中位 {_pct(float(np.median(o['swing'])) * 100, False)}",
                        "若有「釘住」，結算前應該比較安靜"])
        t4_rows.append(["結算時價格離最近整數履約價的距離（格距的幾成）", f"{len(m['dist'])} 次",
                        f"{m['dist'].mean() * 100:.1f}%", f"{float(np.median(m['dist'])) * 100:.1f}%", "—",
                        f"其他週五：平均 {o['dist'].mean() * 100:.1f}%；完全隨機約 25%",
                        "格距：2 萬美元以上 1,000，5,000 以上 500"])
    try:
        gap = _cme_gap(_cme())
        for k, word in ((1, "當天"), (3, "3 個交易日內"), (5, "5 個交易日內"), (10, "10 個交易日內")):
            t4_rows.append([f"芝商所週末缺口 1% 以上：{word}回到週五收盤價", f"{gap['n'] if k == 10 else gap['events']} 次",
                            "—", "—", _share(gap["fill"][k]), f"同樣距離的反方向價位：{_share(gap['mirror'][k])}",
                            "反方向價位是安慰劑對照"])
    except Exception as exc:  # noqa: BLE001
        notes_extra.append(f"Yahoo 芝商所比特幣期貨日線這次取不到（{type(exc).__name__}），缺口部分留空。")

    lookahead_ok = _lookahead(btc)

    # ---------- now ----------
    now = []
    last_ret = float(ret.iloc[-1])
    recent = [d for d in drop8 if (asof - d).days <= 30]
    now.append({"label": f"比特幣最近一天（{_d(asof)}）漲跌", "value": _p(last_ret)})
    now.append({"label": "最近 30 天單日跌 8% 以上", "value": "、".join(_d(d) for d in recent) if recent else "沒有"})
    now.append({"label": "比特幣相對 50 日均線", "value": "在均線之上" if btc.iloc[-1] > btc.iloc[-50:].mean() else "在均線之下"})
    mstr_since = None
    if i_sum.get("day1") is not None:
        try:
            s, spy = old_studies._yahoo("MSTR").astype(float), old_studies._yahoo("SPY").astype(float)
            a = s.index[int(s.index.searchsorted(pd.Timestamp("2026-11-12"))) - 1]
            mstr_since = (s.iloc[-1] / s[a] - 1, s.iloc[-1] / s[a] - 1 - (spy.iloc[-1] / spy.asof(a) - 1), s.index[-1])
        except Exception:  # noqa: BLE001
            mstr_since = None
    now.append({"label": "MSCI 指數剔除 Strategy（MSTR）",
                "value": "MSCI 於 2026-11-11 收盤後公布：自 2026-11-30 收盤後從其全球指數（含美國、世界、ACWI）剔除；"
                         "公司申報說追蹤 MSCI 的基金約持有 1,300 萬股（約 3.1% 的普通股）" +
                         (f"；公布前一天收盤到 {_d(mstr_since[2])}，MSTR {_p(mstr_since[0])}（減標普 {_p(mstr_since[1])}）"
                          if mstr_since else "")})
    pain = None
    try:
        mp = json.loads((common.STORE / "btc_max_pain.json").read_text(encoding="utf-8"))
        k = max(mp)
        pain = (k, mp[k], len(mp))
    except Exception:  # noqa: BLE001
        pass
    now.append({"label": "下一次 Deribit 比特幣月結算", "value": "2026-11-27（週五）08:00 UTC；季結算是 2026-12-25" +
                (f"；最近到期日的最大痛點 {pain[1]:,.0f} 美元（{pain[0]}，儀表板只存了 {pain[2]} 天）" if pain else "")})
    now.append({"label": "近期較大的代幣解鎖（Tokenomist 週報，2026-11-16 那一週）",
                "value": "、".join(f"{n} {d}（約流通量 {p}）" for n, d, p in UPCOMING_UNLOCKS) +
                         "；更後面的日程這次沒有找到可免登入讀取的清單"})

    # ---------- verdict ----------
    d30 = ev["drop8"].get(30)
    d7 = ev["drop8"].get(7)
    r7 = rules.get("比特幣跌 8% 以上收盤買、抱 7 天")
    r30 = rules.get("比特幣跌 8% 以上收盤買、抱 30 天")
    ub = np.array(u_sum["before"])
    ua = np.array(u_sum["after"])
    sq30 = ev.get("sq", {}).get(30)
    month_pre, other_pre = (exp["month"]["pre"], exp["other"]["pre"]) if exp else (np.array([]), np.array([]))
    def pts(v):
        return "—" if not _ok(v) else f"{abs(v) * 100:.1f} 個百分點"

    pre_low = {s: _placebo_low(e["month"]["pre"], e["other"]["pre"], np.random.default_rng(SEED))
               for s, e in (exp_coins.items() if exp else [])}
    eb = exp_rules.get("比特幣")
    verdict = (
        f"比特幣單日跌 8% 以上 {len(drop8)} 段，之後 30 天平均 {_p(d30['mean']) if d30 else '—'}，只比同趨勢隨機日多 "
        f"{pts(d30['mean'] - d30['matched']) if d30 else '—'}；「跌深收盤買、抱 7／30 天」輸給隨機進場的機率 "
        f"{_share(r7[1]['beat']) if r7 else '—'}／{_share(r30[1]['beat']) if r30 else '—'}，後半段和以太幣都不成立；"
        f"軋空只有 {len(sq)} 段，之後偏強，但一般大漲日之後也漲。"
        f"{len(ub)} 次大型代幣解鎖前 30 天平均比比特幣{'差' if len(ub) and ub.mean() < 0 else '多'} {pts(ub.mean()) if len(ub) else '—'}"
        f"（這些幣平常每 30 天也{'差' if _ok(u_sum['placebo']) and u_sum['placebo'] < 0 else '多'} {pts(u_sum['placebo'])}），"
        f"加密股票指數事件只有 {len(INDEX_EVENTS)} 件，芝商所缺口回補不比反方向價位多。"
        f"月結算前一週比特幣平均 {_p(month_pre.mean()) if len(month_pre) else '—'}（其他週 {_p(other_pre.mean()) if len(other_pre) else '—'}），"
        f"以太幣、索拉納也偏弱，但三者走勢相連、樣本 {len(month_pre)} 次，避開這週的做法輸給換成其他週五的機率 {_share(eb['beat']) if eb else '—'}。"
    )

    notes = [
        f"資料：比特幣日收盤（UTC 00:00）{_d(btc.index[0])} 起，事件從 {START} 起算；Bybit 未平倉量（以顆數計，"
        f"每日一筆）{_d(oi_first) if oi_first is not None else '—'} 起；幣安資金費率 {_d(fund_first) if fund_first is not None else '—'} 起；"
        f"Deribit 月結算 {exp['n'] if exp else 0} 次（{_d(exp['first']) if exp else '—'} 起，用幣安 1 小時線的 08:00 價格代替結算價）；"
        f"芝商所 BTC=F 日線 {_d(gap['first']) if gap else '—'} 起。",
        "爆倉就是槓桿不夠、被交易所強制賣出；單日跌 8% 以上當作爆倉連環下跌的代表。幣安的未平倉量歷史只保留約 30 天，"
        "所以改用 Bybit 公開的長歷史；只有一家交易所，而且某天減少 10% 也可能是大家主動平倉。",
        "同一段 7 天內的多個事件算成一段；「隨機 X 比它好」是從同樣趨勢狀態（前一天在 50 日均線上或下）的日子"
        f"隨機挑同樣多天、重複 {SIMS} 次，平均比事件好的比例。視窗互相重疊，實際獨立的例子比表上的數字還少。",
        "規則表：每年報酬分別扣 0.10% 和 0.30% 交易成本；空手時不計利息；「前半／後半」是把同一段期間切成兩半各自的每年報酬；"
        "隨機進場是同樣次數、同樣趨勢狀態、同樣持有天數，重複 400 次。持有時間很短，所以每年報酬本來就會比一直抱著小。",
        f"代幣解鎖就是原本鎖住的幣開始可以賣。表上的解鎖清單是從新聞報導整理的（每一列寫了來源），只收日期明確、規模在流通量 2% 以上的；"
        f"各家算流通量的分母不同，比例只是大概。括號裡是減掉同期比特幣漲跌之後的數字。"
        f"同樣這些幣在其他日子（離解鎖 45 天以上），30 天漲跌減比特幣平均是 {_p(u_sum['placebo'])}"
        f"（{u_sum['n_placebo']} 個幣），所以解鎖前後輸比特幣不一定是解鎖造成的。"
        f"解鎖前 30 天減比特幣：平均 {_p(float(np.mean(u_sum['before'])) if u_sum['before'] else None)}、"
        f"{sum(v < 0 for v in u_sum['before'])}／{len(u_sum['before'])} 次為負；解鎖後 30 天減比特幣：平均 "
        f"{_p(float(np.mean(u_sum['after'])) if u_sum['after'] else None)}、中位數 "
        f"{_p(float(np.median(u_sum['after'])) if u_sum['after'] else None)}。"
        + (f"若改用幣自己之前的走勢當對照（前 60 天每 30 天的漲跌），解鎖前 30 天平均差 {_p(float(np.mean(u_sum['trend_gap'])))}"
           f"（{len(u_sum['trend_gap'])} 次有足夠歷史）。" if u_sum["trend_gap"] else ""),
        f"軋空的 {len(sq)} 段是 {'、'.join(_d(d) for d in sq) or '—'}；最後一次之後再也沒有出現，"
        "規則表裡軋空的「後半段」若是 0，就是那半段沒有事件。單日大漲之後 30 天普遍偏強（不論資金費率），軋空這幾段和一般大漲日的差別樣本太少分不出來。",
        "指數剔除就是追蹤指數的基金必須在生效日前賣出（加入則必須買進）。股票表「當段」是公布前一天收盤到基金調整那天收盤，"
        "「之後」是調整那天收盤再往後 20 個交易日；MSCI 的兩次決定沒有實際買賣，「之後」改成公布後 20 個交易日。"
        "加密相關股票的指數事件只有這幾件，而且同一時間比特幣也在動，不能當成規律。",
        "股票的一般研究（引用，不重算）：Harris 與 Gurel（1986）發現加入標普 500 的股票公布後上漲約 3%，兩週內大多回吐；"
        "Shleifer（1986）認為上漲是因為基金必須買進而持續存在；Chen、Noronha 與 Singal（2004）發現加入的漲幅會留下來、"
        "被剔除的跌幅之後會漲回一部分；Greenwood 與 Sammon（2025）發現美國標普 500 的加入效應在 2010 年代以後幾乎消失。"
        "期權到期的「釘住」現象見 Ni、Pearson 與 Poteshman（2005），主要出現在個股選擇權。",
        "Deribit 結算時間是每月最後一個週五 08:00 UTC，芝商所比特幣期貨也在同一個週五到期，兩者無法分開。"
        "最大痛點只有儀表板最近幾天存下的數字，沒有歷史，所以「釘住」改用價格離整數履約價的距離來看。",
        "芝商所缺口：週一（其實是週日晚上開盤）開盤價和上週五收盤價差 1% 以上算一個缺口；「回到」是之後的最高或最低價碰到週五收盤價。"
        "對照是同樣距離但在另一邊的價位；如果兩者差不多，表示回補只是價格本來就會來回擺動。Yahoo 的收盤價不是芝商所正式結算價。",
        f"未來資料檢查：把資料截在 2022-06-30 重算事件、趨勢狀態和持有部位，和用完整資料算到同一天的結果{'完全相同' if lookahead_ok else '不同（請留意）'}。",
        "MSTR 的指數事件只當作市場事實描述；這裡不提供任何買賣建議。",
        "過去的結果不保證未來會一樣，這裡只是整理歷史。",
    ] + notes_extra

    tables = [
        {"caption": f"事件之後比特幣（或該幣）的表現（每格：平均漲跌，結束時比較高的比例；{START} 起）",
         "columns": ["事件", "獨立次數", "1 天後", "3 天後", "7 天後", "14 天後", "30 天後", "30 天：比同趨勢隨機日多；隨機比它好的比例"],
         "rows": t1_rows[:16]},
        {"caption": "照事件買進、抱固定天數的規則（同一段期間比較）",
         "columns": ["做法", "次數與持有時間", "每年報酬（成本 0.10%／0.30%）", "最深虧損", "同期一直抱著／50 日均線（每年）",
                     "前半段／後半段（每年）", "隨機進場同樣時間（每年）"],
         "rows": t2_rows[:16]},
        {"caption": "代幣解鎖與股票指數調整前後（每格：原始漲跌，括號內減掉對照）",
         "columns": ["標的與事件", "日期", "規模與來源／其他", "事件前（解鎖前 30 天；公布前 20 個交易日）",
                     "事件當段（解鎖前一天到後一天；公布到基金調整日）", "事件後（解鎖後 30 天；調整後 20 個交易日）", "對照"],
         "rows": (u_rows + i_rows)[:16]},
        {"caption": "期權與期貨到期、芝商所週末缺口",
         "columns": ["檢查", "次數", "平均", "中位數", "上漲比例／回到的比例", "對照", "說明"],
         "rows": t4_rows[:16]},
    ]
    notes.append(f"計算時間約 {time.time() - t0:.0f} 秒。")
    return _clean({
        "id": "forced_flows",
        "title": "被迫買賣的時點",
        "question": "當有人必須在已知的時間買或賣（爆倉、軋空、代幣解鎖、指數剔除、期權到期），價格會不會有固定的走法、能不能拿來用？",
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨日線與 1 小時線（比特幣、以太幣、索拉納與各代幣）", "url": "https://www.binance.com/zh-TC/markets"},
            {"name": "Bybit BTCUSDT 未平倉量歷史（公開 API）", "url": BYBIT + "/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1d&limit=200"},
            {"name": "幣安 BTCUSDT 資金費率歷史", "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "Yahoo Finance 芝商所比特幣期貨 BTC=F", "url": "https://finance.yahoo.com/quote/BTC=F/"},
            {"name": "Yahoo Finance MSTR、COIN、HOOD、XYZ、SPY 日線", "url": "https://finance.yahoo.com/quote/SPY/history"},
            {"name": "Deribit 選擇權（最大痛點由儀表板每日存下）", "url": "https://www.deribit.com/statistics/BTC/options-open-interest"},
            {"name": "Cointelegraph：MSCI 確認 11 月底剔除 Strategy", "url": "https://cointelegraph.com/news/msci-confirms-strategys-removal-from-indexes-end-november"},
            {"name": "Bitcoin Magazine：MSCI 將 Strategy 移出美國指數（2026-11-11）", "url": "https://bitcoinmagazine.com/markets/msci-removes-strategy-mstr-usa-index"},
            {"name": "Tokenomist 每週解鎖整理（2026-11-16 那一週）", "url": "https://tokenomist.ai/research/weekly-unlock-digest-nov-16-22-2026-layerzeros-34-69m-cliff-lands-ahead-of-zero-mainnet"},
            {"name": "標普道瓊指數公告（COIN、XYZ、HOOD 加入標普 500）", "url": "https://www.spglobal.com/spdji/en/index-announcements/"},
            {"name": "Nasdaq：MicroStrategy 加入那斯達克 100（2024-12-13）", "url": INDEX_EVENTS[0][4]},
            {"name": "Harris & Gurel (1986), Journal of Finance 41(4)", "url": "https://scholar.google.com/scholar?q=Harris+Gurel+1986+Price+and+volume+effects+associated+with+changes+in+the+S%26P+500+list"},
            {"name": "Shleifer (1986), Journal of Finance 41(3)", "url": "https://scholar.google.com/scholar?q=Shleifer+1986+Do+demand+curves+for+stocks+slope+down"},
            {"name": "Chen, Noronha & Singal (2004), Journal of Finance 59(4)", "url": "https://scholar.google.com/scholar?q=Chen+Noronha+Singal+2004+price+response+to+S%26P+500+index+additions+and+deletions"},
            {"name": "Greenwood & Sammon (2025), The Disappearing Index Effect, Journal of Finance", "url": "https://scholar.google.com/scholar?q=Greenwood+Sammon+The+Disappearing+Index+Effect"},
            {"name": "Ni, Pearson & Poteshman (2005), Journal of Financial Economics 78(1)", "url": "https://scholar.google.com/scholar?q=Ni+Pearson+Poteshman+2005+Stock+price+clustering+on+option+expiration+dates"},
        ] + [{"name": f"{u[1]} {u[2]} 解鎖：{u[5]}", "url": u[6]} for u in UNLOCKS],
    })


def _clean(x):
    """No NaN or infinity anywhere; None in shown cells becomes "—"."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, (np.integer,)):
        return int(x)
    return "—" if x is None else x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
