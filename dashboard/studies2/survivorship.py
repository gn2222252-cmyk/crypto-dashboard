"""Historical study "survivorship": if the coin list is rebuilt as it looked at the time - including coins that
later collapsed or were delisted - does trend-following still help, and does "buy the strongest coins" look any
different?

Coins: every USDT spot pair that ever existed on Binance, taken from the public archive's listing
(data.binance.vision, S3-style XML). Stablecoins, pegged / wrapped tokens, fiat and leveraged tokens are dropped.
A first screen uses one monthly candle request per pair (quote volume per calendar month); pairs that never ranked
in the top SCREEN_RANK by monthly dollar volume in any month since 2018-09 cannot reach a top-50 list and are not
downloaded further (capped at MAX_SYMBOLS). Daily candles of the kept pairs come from Binance's public klines
endpoint, which still serves delisted ("BREAK") pairs; a pair the endpoint does not know is read from the archive's
monthly zip files instead (open times in milliseconds, or microseconds from 2025).

Point-in-time universe: at the close of each month's last day, the N coins (N = 20, 50) with the highest dollar
volume over the previous 30 days among coins with at least 90 daily candles that are still trading. A pair whose
candles stop for more than 7 days has stopped trading from its last candle on; the holder exits at that close
(harsher versions take a further 50% or 100% off on exit). Everything decided at a close is held from the next day.
Costs: 0.10% (and 0.30%) of the amount traded; cash earns nothing.

Strategies use "slots": at each rebuild every chosen coin gets an equal slot of the money; between rebuilds a slot
drifts with its coin, and a fraction g of each slot (decided at each close) is held in the coin, the rest in cash.
g = 1 is plain holding; g = the coin's own 17-rule blend share is the trend blend; g = 1 / 0 by bitcoin's 50-day
average is the market filter; g = bitcoin's blend share is the blend as a market-wide fraction.

All data is cached under cache/_store/survivorship/; later runs only top up recent days.
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
from dashboard.collectors import price as px

COST = 0.001
COST_HIGH = 0.003
YEAR = 365
NS = (20, 50)
MIN_DAYS = 90
VOL_DAYS = 30
GAP_DAYS = 7
MOM_DAYS = 28
TOPS = (3, 5)
HAIRCUTS = (0.0, 0.5, 1.0)
START = "2019-01-01"
PERIODS = (("2019–2021", "2019-01-01", "2021-12-31"), ("2022–2023", "2022-01-01", "2023-12-31"),
           ("2024 起", "2024-01-01", None))
SCREEN_FROM = "2018-09"
SCREEN_RANK = 60
MAX_SYMBOLS = 500
TIME_BUDGET = 30 * 60  # seconds of downloading per run; the rest continues next run
BLEND_MA = (10, 20, 30, 50, 75, 100, 150, 200)
BLEND_CHANNEL = (10, 20, 50, 100)
BLEND_PAST = (14, 30, 60, 90, 180)
BTC = "BTCUSDT"

DIR = common.STORE / "survivorship"
DAILY_DIR = DIR / "daily"
ARCHIVE = "https://data.binance.vision"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/spot/monthly/klines/"

PEGGED = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "PAX", "USDS", "USDSB", "SUSD", "UST", "USTC", "USDE",
          "USD1", "RLUSD", "PYUSD", "BFUSD", "XUSD", "EURI", "AEUR", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "NGN",
          "UAH", "ZAR", "JPY", "ARS", "MXN", "PLN", "RON", "COP", "CZK", "IDRT", "BIDR", "BVND", "BKRW", "PAXG",
          "XAUT", "WBTC", "WBETH", "BETH", "BNSOL", "STETH", "WSTETH", "CBETH", "RETH", "USDJ", "GUSD", "HUSD",
          "FRAX", "LUSD", "VAI", "TUSDB", "USDT", "USDSOLD", "RWUSD", "U"}
# tokenized shares and ETFs (not coins), listed on Binance spot from 2026-06 with a "B" suffix
STOCKS = {"CRCLB", "GOOGLB", "INTCB", "KORUB", "MSTRB", "MUB", "NVDAB", "QQQB", "SKHYB", "SNDKB", "SNXXB", "SOXLB",
          "SOXSB", "SPCXB", "TQQQB"}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")
PEG_BAND = 0.03  # a coin whose closes stay within 3% of one dollar (median distance) is treated as a stablecoin
# pairs that stopped because the coin was renamed or swapped into a new ticker, not because it failed; they still
# exit the basket at their last close, but they are not counted as "died" and get no extra loss on exit
RENAMED = {"BCCUSDT": "BCH", "BCHABCUSDT": "BCH", "VENUSDT": "VET", "ERDUSDT": "EGLD", "NPXSUSDT": "PUNDIX",
           "LENDUSDT": "AAVE", "BTTUSDT": "BTTC", "MATICUSDT": "POL", "FTMUSDT": "S", "RNDRUSDT": "RENDER",
           "AGIXUSDT": "FET", "OCEANUSDT": "FET", "KEEPUSDT": "T", "NUUSDT": "T", "GALUSDT": "G", "MCUSDT": "BEAM",
           "STRATUSDT": "STRAX", "TOMOUSDT": "VIC", "EOSUSDT": "A", "MKRUSDT": "SKY", "COCOSUSDT": "COMBO",
           "KLAYUSDT": "KAIA", "DARUSDT": "D", "FXSUSDT": "FRAX"}


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


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _money(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return f"{v:,.1f}" if v >= 10 else f"{v:,.2f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _base(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


# ---------- store ----------

def _read(path, default):
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        return value if isinstance(value, type(default)) else default
    except Exception:  # noqa: BLE001 - a damaged store file starts over
        return default


def _write(path, value) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    reply = requests.get(url, params=params, headers=common.HEADERS, timeout=timeout)
    time.sleep(common.PAUSE)
    return reply


# ---------- the list of pairs ----------

def _archive_symbols() -> list[str]:
    """Every symbol folder in the archive's spot monthly klines listing (raises if the listing cannot be read)."""
    found, marker = [], ""
    for _ in range(50):
        reply = _get(LISTING, {"delimiter": "/", "prefix": PREFIX, "marker": marker})
        reply.raise_for_status()
        page = re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([^/<]+)/</Prefix>", reply.text)
        found += page
        if "<IsTruncated>true</IsTruncated>" not in reply.text or not page:
            break
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", reply.text)
        marker = nxt.group(1) if nxt else PREFIX + page[-1] + "/"
    if len(found) < 500:
        raise RuntimeError(f"archive listing only {len(found)} symbols")
    return sorted(set(found))


def _wanted(symbol: str, bases: set[str]) -> bool:
    if not symbol.endswith("USDT") or len(symbol) <= 4:
        return False
    base = _base(symbol)
    if base in PEGGED or base in STOCKS:
        return False
    for tail in LEVERAGED:
        if base.endswith(tail) and (base == tail or base[:-len(tail)] in bases):
            return False
    return True


def _pairs(month: str) -> dict:
    """{"symbols", "status", "source", "month"}; refreshed once a month."""
    path = DIR / "pairs.json"
    stored = _read(path, {})
    if stored.get("month") == month and stored.get("symbols"):
        return stored
    status, source, symbols = dict(stored.get("status") or {}), stored.get("source"), None
    try:
        info = px._binance("/api/v3/exchangeInfo", {})
        status = {s["symbol"]: s["status"] for s in info.get("symbols", [])}
    except Exception:  # noqa: BLE001 - keep the previous statuses
        pass
    try:
        symbols, source = _archive_symbols(), "archive"
    except Exception:  # noqa: BLE001 - fall back to the exchange's own list (fully removed pairs may be missing)
        if status:
            symbols, source = sorted(status), "exchangeInfo"
    if not symbols or not status:
        if stored.get("symbols"):
            return stored
        raise RuntimeError("幣安的交易對清單或交易狀態取不到")
    every = set(symbols) | set(status)
    bases = {_base(s) for s in every if s.endswith("USDT")}
    usdt = sorted(s for s in every if _wanted(s, bases))
    out = {"month": month, "source": source, "listed": len(symbols), "symbols": usdt,
           "status": {s: status.get(s, "GONE") for s in usdt}}
    _write(path, out)
    return out


# ---------- screening by monthly dollar volume ----------

def _screen(pairs: dict, month: str, deadline: float) -> tuple[dict, list[str]]:
    """{symbol: {YYYY-MM: quote volume}} from one monthly-candle request per pair; dead pairs are fetched once."""
    path = DIR / "monthly.json"
    stored = _read(path, {})
    missing = []
    # pairs already screened are re-checked each month only if they are now among the busiest by 24-hour volume
    busy = None
    if any(isinstance(stored.get(s), dict) and stored[s].get("month") != month for s in pairs["symbols"]):
        try:
            tick = px._binance("/api/v3/ticker/24hr", {})
            vol = {t["symbol"]: float(t.get("quoteVolume") or 0) for t in tick if t.get("symbol") in pairs["status"]}
            busy = set(sorted(vol, key=lambda s: -vol[s])[:SCREEN_RANK * 2])
        except Exception:  # noqa: BLE001 - re-check every trading pair instead
            busy = None
    for symbol in pairs["symbols"]:
        item = stored.get(symbol)
        fresh = isinstance(item, dict) and (item.get("month") == month or pairs["status"].get(symbol) != "TRADING"
                                            or (busy is not None and symbol not in busy))
        if fresh:
            continue
        if time.time() > deadline:
            missing.append(symbol)
            continue
        try:
            rows = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1M", "limit": 1000})
            stored[symbol] = {"month": month, "rows": {pd.to_datetime(r[0], unit="ms").strftime("%Y-%m"): float(r[7])
                                                     for r in rows}}
        except Exception:  # noqa: BLE001 - unknown to the endpoint: decide from the archive's daily files later
            if not isinstance(item, dict):
                stored[symbol] = {"month": month, "rows": {}, "unknown": True}
    _write(path, stored)
    return stored, missing


def _select(pairs: dict, monthly: dict) -> tuple[list[str], dict]:
    """Pairs that ranked in the top SCREEN_RANK by monthly dollar volume in some month (capped at MAX_SYMBOLS)."""
    frame = pd.DataFrame({s: (monthly.get(s) or {}).get("rows", {}) for s in pairs["symbols"]}).sort_index()
    total = frame.sum(axis=0)
    young = frame.notna().cumsum() <= 3  # the listing month and the next two: under 90 days, cannot be chosen yet
    frame = frame.where(~young)
    frame = frame[frame.index >= SCREEN_FROM]
    rank = frame.rank(axis=1, ascending=False, method="min")
    best = rank.min(axis=0)
    unknown = [s for s in pairs["symbols"] if (monthly.get(s) or {}).get("unknown")]
    keep = [s for s in best.index if _ok(best[s]) and best[s] <= SCREEN_RANK]
    keep.sort(key=lambda s: (best[s], -total[s]))
    cut = keep[MAX_SYMBOLS:]
    keep = keep[:MAX_SYMBOLS]
    if BTC not in keep:
        keep.insert(0, BTC)
    keep += [s for s in unknown if s not in keep]  # the endpoint does not know them: read the archive instead
    return keep, {"screened": int(frame.notna().any(axis=0).sum()), "kept": len(keep), "over_cap": cut,
                  "unknown": unknown}


# ---------- daily candles ----------

def _api_daily(symbol: str, start_ms: int) -> list:
    rows, start = [], start_ms
    for _ in range(12):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    return [[r[0], r[4], r[7]] for r in rows]


def _archive_daily(symbol: str, deadline: float) -> list:
    """Monthly zip files of daily candles from the archive (open time in ms, or in microseconds from 2025)."""
    reply = _get(LISTING, {"prefix": f"{PREFIX}{symbol}/1d/"})
    reply.raise_for_status()
    keys = sorted(k for k in re.findall(r"<Key>([^<]+\.zip)</Key>", reply.text))
    rows = []
    for key in keys:
        if time.time() > deadline:
            raise TimeoutError("time budget")
        got = _get(f"{ARCHIVE}/{key}")
        if got.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(got.content)) as zf:
            for name in zf.namelist():
                for line in zf.read(name).decode("utf-8", "replace").splitlines():
                    cells = line.split(",")
                    if len(cells) < 8 or not cells[0].strip().isdigit():
                        continue  # header line
                    ms = int(cells[0])
                    if ms > 10 ** 14:
                        ms //= 1000
                    rows.append([ms, cells[4], cells[7]])
    return rows


def _daily(symbols: list[str], pairs: dict, members: set[str], month: str, deadline: float) -> tuple[dict, dict]:
    """{symbol: DataFrame(close, qv) on real candle days}, plus bookkeeping about what was (not) fetched."""
    meta_path = DIR / "daily_meta.json"
    meta = _read(meta_path, {})
    full_refresh = meta.get("refreshed") != month
    today = pd.Timestamp(common.today_utc())
    out, info = {}, {"fetched": 0, "skipped": [], "stale": [], "archive": []}
    for symbol in symbols:
        path = DAILY_DIR / f"{symbol}.json"
        item = _read(path, {})
        rows = {str(k)[:10]: v for k, v in (item.get("rows") or {}).items()
                if isinstance(v, list) and len(v) == 2 and common._finite(v[0]) and v[0] > 0}
        trading = pairs["status"].get(symbol) == "TRADING"
        need = not rows or (trading and (full_refresh or symbol in members)
                            and max(rows) < common.today_utc() and
                            (today - pd.Timestamp(max(rows))).days > 1)
        if not rows and item.get("done"):
            need = False
        if need:
            if time.time() > deadline:
                if not rows:
                    info["skipped"].append(symbol)
                    continue
                info["stale"].append(symbol)
            else:
                try:
                    start = int((pd.Timestamp(max(rows)) - pd.Timedelta(days=3)).timestamp() * 1000) if rows else 0
                    src = item.get("src", "api")
                    if src == "archive" and not trading:
                        got = []
                    else:
                        try:
                            got = _api_daily(symbol, start)
                            src = "api"
                        except Exception:  # noqa: BLE001 - not served by the endpoint: try the archive
                            got = [] if rows else _archive_daily(symbol, deadline)
                            src = "archive" if got else src
                    for ms, close, qv in got:
                        day = pd.to_datetime(int(ms), unit="ms").normalize()
                        if day < today and float(close) > 0:
                            rows[_d(day)] = [float(close), float(qv)]
                    info["fetched"] += 1
                    if src == "archive":
                        info["archive"].append(symbol)
                    _write(path, {"src": src, "done": True, "rows": dict(sorted(rows.items()))})
                except Exception:  # noqa: BLE001 - one pair failing must not stop the others
                    if not rows:
                        info["skipped"].append(symbol)
                        continue
                    info["stale"].append(symbol)
        elif item.get("src") == "archive":
            info["archive"].append(symbol)
        if len(rows) < 2:
            continue
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=["close", "qv"])
        frame.index = pd.to_datetime(frame.index)
        out[symbol] = frame.sort_index()
    if full_refresh and not info["stale"] and not info["skipped"]:
        meta["refreshed"] = month
    _write(meta_path, meta)
    return out, info


def _load() -> dict:
    t0 = time.time()
    deadline = t0 + TIME_BUDGET
    month = common.today_utc()[:7]
    pairs = _pairs(month)
    every = set(pairs["symbols"])
    bases = {_base(s) for s in every}
    pairs["symbols"] = [s for s in pairs["symbols"] if _wanted(s, bases)]  # the exclusion lists may be newer
    monthly, unscreened = _screen(pairs, month, deadline)
    symbols, sel = _select(pairs, monthly)
    members = set(_read(DIR / "members.json", []))
    raw, info = _daily(symbols, pairs, members | {BTC}, month, deadline)
    if BTC not in raw:
        raise RuntimeError("比特幣的日線取不到")
    pegged = [s for s, f in raw.items() if float((f["close"] - 1).abs().median()) < PEG_BAND]
    for s in pegged:
        raw.pop(s)
    info["pegged"] = pegged
    return {"pairs": pairs, "raw": raw, "sel": sel, "info": info, "unscreened": unscreened,
            "seconds": time.time() - t0}


# ---------- panels ----------

def _cut(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    gaps = frame.index.to_series().diff().dt.days
    late = gaps[gaps > GAP_DAYS]
    if len(late):
        return frame[frame.index < late.index[0]], True
    return frame, False


def _panels(raw: dict, status: dict) -> dict:
    """Day × coin panels on one calendar index (only real candles count as history; a pair is alive from its first
    candle to its last one before the first gap longer than GAP_DAYS)."""
    end_day = raw[BTC].index[-1]
    cut, ended = {}, {}
    for s, f in raw.items():
        f, gap = _cut(f[f.index <= end_day])
        if len(f) < 2:
            continue
        cut[s] = f
        ended[s] = bool(gap or status.get(s) != "TRADING")
    coins = sorted(cut, key=lambda s: (s != BTC, s))
    days = pd.date_range(min(f.index[0] for f in cut.values()), end_day, freq="D")
    close = pd.DataFrame({s: cut[s]["close"] for s in coins}).reindex(days)
    qv = pd.DataFrame({s: cut[s]["qv"] for s in coins}).reindex(days).fillna(0.0)
    real = close.notna()
    first = {s: cut[s].index[0] for s in coins}
    # a pair still trading whose stored candles are a few days old (only refreshed at the start of a month when it is
    # not in the current list) is carried at its last close, never treated as having stopped
    last = {s: (cut[s].index[-1] if ended[s] else end_day) for s in coins}
    life = pd.DataFrame({s: (days >= first[s]) & (days <= last[s]) for s in coins}, index=days)
    price = close.ffill().where(life)
    ret = (price / price.shift(1) - 1).where(life & life.shift(1, fill_value=False)).fillna(0.0)
    count = real.cumsum()
    return {"coins": coins, "days": days, "close": price, "qv": qv, "life": life, "ret": ret, "count": count,
            "first": first, "last": last, "ended": ended,
            "survivor": {s: not ended[s] for s in coins}}


def _blend(price: pd.DataFrame) -> pd.DataFrame:
    """Share of the 17 rules that are long at each close (mean of the rules that already have enough history)."""
    parts = []
    for n in BLEND_MA:
        mean = price.rolling(n, min_periods=n).mean()
        parts.append((price > mean).astype(float).where(mean.notna()))
    for n in BLEND_CHANNEL:
        m = n // 2
        high = price.rolling(n, min_periods=n).max()
        low = price.rolling(m, min_periods=m).min()
        state = pd.DataFrame(np.where(price >= high, 1.0, np.where(price <= low, 0.0, np.nan)),
                             index=price.index, columns=price.columns)
        parts.append(state.where(high.notna()).ffill().fillna(0.0).where(high.notna()))
    for n in BLEND_PAST:
        past = price / price.shift(n) - 1
        parts.append((past > 0).astype(float).where(past.notna()))
    total = sum(p.fillna(0.0) for p in parts)
    have = sum(p.notna().astype(float) for p in parts)
    return (total / have.where(have > 0)).where(price.notna())


def _universes(p: dict, n: int, pool: list[str]) -> dict:
    """{decision day (last day of a month): [coins]} - top n by 30-day dollar volume, decided at that close."""
    days = p["days"]
    vol30 = p["qv"][pool].rolling(VOL_DAYS, min_periods=1).sum()
    month_ends = days[(days + pd.Timedelta(days=1)).day == 1]
    out = {}
    for t in month_ends:
        if t < pd.Timestamp(START) - pd.Timedelta(days=1):
            continue
        ok = [s for s in pool if p["life"].at[t, s] and (p["last"][s] > t or not p["ended"][s])
              and p["count"].at[t, s] >= MIN_DAYS]
        if not ok:
            continue
        v = vol30.loc[t, ok].sort_values(ascending=False, kind="stable")
        out[t] = list(v.index[:n])
    return out


def _targets(p: dict, unis: dict, blend: pd.DataFrame, btc_above: pd.Series, btc_blend: pd.Series) -> dict:
    """Rebuild days, slot weights and the invested fraction g for every strategy on one universe."""
    days, coins = p["days"], p["coins"]
    col = {s: i for i, s in enumerate(coins)}
    T, K = len(days), len(coins)
    month_rb = np.zeros(T, bool)
    month_w = np.zeros((T, K))
    member = np.zeros((T, K), bool)  # universe in force after the decision at close t
    pos = {t: i for i, t in enumerate(days)}
    current = np.zeros(K, bool)
    for i, t in enumerate(days):
        if t in unis:
            current = np.zeros(K, bool)
            idx = [col[s] for s in unis[t]]
            current[idx] = True
            month_rb[i] = True
            month_w[i, idx] = 1.0 / len(idx)
        member[i] = current
    ones = np.ones((T, K))
    out = {"basket": (month_rb, month_w, ones),
           "blend": (month_rb, month_w, np.nan_to_num(blend[coins].values, nan=0.0)),
           "filter": (month_rb, month_w, np.repeat(btc_above.values[:, None], K, axis=1)),
           "btcblend": (month_rb, month_w, np.repeat(np.nan_to_num(btc_blend.values)[:, None], K, axis=1))}
    mom = (p["close"] / p["close"].shift(MOM_DAYS) - 1)[coins].values
    sunday = np.asarray(days.dayofweek == 6)
    last_idx = np.array([days.get_loc(p["last"][s]) if p["ended"][s] else T for s in coins])
    for k in TOPS:
        w = np.zeros((T, K))
        for i in np.where(sunday)[0]:
            row = np.where(member[i] & np.isfinite(mom[i]) & (last_idx > i), mom[i], np.nan)
            ok = np.where(np.isfinite(row))[0]
            if not len(ok):
                continue
            pick = ok[np.argsort(-row[ok], kind="stable")][:k]
            w[i, pick] = 1.0 / len(pick)
        out[f"top{k}"] = (sunday.copy(), w, ones)
        out[f"top{k}f"] = (sunday.copy(), w, np.repeat(btc_above.values[:, None], K, axis=1))
    out["_member"] = member
    del pos
    return out


def _simulate(p: dict, spec: tuple, cost: float, haircut: float, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Daily net returns (for days start+1 ...) and amount traded, using slots (see the module docstring)."""
    rb, w, g = spec
    ret = p["ret_np"]
    alive_next = p["alive_next"]
    exit_at = p["exit_at"]
    T, K = ret.shape
    slot = np.zeros(K)
    inv = np.zeros(K)
    free = 1.0
    net = np.zeros(T - 1 - start)
    traded = np.zeros(T - 1 - start)
    first = True
    for t in range(start, T - 1):
        equity = slot.sum() + free
        sold = 0.0
        ex = exit_at[t]
        if ex.any():
            hit = ex & (inv > 0)
            if hit.any():
                sold = float((inv[hit] * (1 - haircut)).sum())
                slot[hit] -= inv[hit] * haircut
                inv[hit] = 0.0
        if rb[t] or first:
            first = False
            total = slot.sum() + free
            slot = total * w[t]
            free = total - slot.sum()
        target = slot * np.where(alive_next[t], g[t], 0.0)
        move = np.abs(target - inv)
        # each slot pays the fee on what it traded; the fee on an exit sale comes out of the free cash
        slot -= cost * move
        free -= cost * sold
        inv = target
        day = ret[t + 1]
        gain = inv * day
        slot = slot + gain
        inv = inv + gain
        after = slot.sum() + free
        net[t - start] = after / equity - 1 if equity > 1e-12 else 0.0
        traded[t - start] = (move.sum() + sold) / equity if equity > 1e-12 else 0.0
    return net, traded


# ---------- measuring ----------

def _stats(net: np.ndarray) -> dict:
    n = len(net)
    if n < 2:
        return {"ret": None, "vol": None, "ratio": None, "deepest": None, "growth": None}
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "deepest": deepest, "growth": growth}


def _measure(net: pd.Series) -> dict:
    out = {"all": _stats(net.values)}
    for name, a, b in PERIODS:
        rows = (net.index >= pd.Timestamp(a)) & (net.index <= (pd.Timestamp(b) if b else net.index[-1]))
        out[name] = _stats(net.values[rows])
    out["years"] = {int(y): float((np.prod(1 + c.values) - 1) * 100) for y, c in net.groupby(net.index.year)}
    return out


def _single(price: pd.Series, share: pd.Series, cost: float) -> tuple[dict, dict]:
    """One coin: hold vs its own blend, from max(day MIN_DAYS of its life, START) to its last close."""
    change = price.pct_change().fillna(0.0).values
    position = np.nan_to_num(share.shift(1).values, nan=0.0)
    base = 1 + position * change
    drift = position * (1 + change) / np.where(np.abs(base) > 1e-9, base, 1.0)
    trade = np.abs(position - np.concatenate([[0.0], drift[:-1]]))
    blend = position * change - cost * trade
    hold = change.copy()
    hold[0] -= cost  # buying on the first day
    return _stats(hold), _stats(blend)


def _compute(p: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    days, coins = p["days"], p["coins"]
    blend = _blend(p["close"])
    btc = p["close"][BTC]
    btc_mean = btc.rolling(50, min_periods=50).mean()
    btc_above = (btc > btc_mean).astype(float).where(btc_mean.notna()).fillna(0.0)
    btc_blend = blend[BTC].fillna(0.0)
    last_idx = np.array([days.get_loc(p["last"][s]) for s in coins])
    T, K = len(days), len(coins)
    tt = np.arange(T)[:, None]
    p["ret_np"] = p["ret"][coins].values
    p["alive_next"] = (tt < last_idx[None, :]) & p["life"][coins].values
    exit_at = np.zeros((T, K), bool)
    for j, s in enumerate(coins):
        if p["ended"][s]:
            exit_at[last_idx[j], j] = True
    p["exit_at"] = exit_at
    start = int(days.get_loc(pd.Timestamp(START) - pd.Timedelta(days=1)))
    index = days[start + 1:]
    survivors = [s for s in coins if p["survivor"][s]]
    dead_flag = {s: p["ended"][s] for s in coins}

    res = {"unis": {}, "runs": {}, "targets": {}}
    for n in NS:
        for pool_name, pool in (("pit", coins), ("surv", survivors)):
            unis = _universes(p, n, pool)
            spec = _targets(p, unis, blend, btc_above, btc_blend)
            res["unis"][(n, pool_name)] = unis
            res["targets"][(n, pool_name)] = spec
            keys = ("basket", "blend") if pool_name == "surv" else \
                ("basket", "blend", "filter", "btcblend", "top3", "top3f", "top5", "top5f")
            for key in keys:
                for cost in ((COST,) if pool_name == "surv" else (COST, COST_HIGH)):
                    hcs = HAIRCUTS if (pool_name == "pit" and cost == COST) else (0.0,)
                    for h in hcs:
                        net, traded = _simulate(p, spec[key], cost, h, start)
                        series = pd.Series(net, index=index)
                        res["runs"][(n, pool_name, key, cost, h)] = _measure(series)

    # ----- coins that stopped trading: the rule's state before the end -----
    ever = set()
    for unis in (res["unis"][(n, "pit")] for n in NS):
        for members in unis.values():
            ever.update(members)
    in50 = set()
    for members in res["unis"][(50, "pit")].values():
        in50.update(members)
    dead = []
    for s in coins:
        if not dead_flag[s] or s in RENAMED or s not in in50:
            continue
        end = p["last"][s]
        price = p["close"][s].loc[:end].dropna()
        share = blend[s].loc[:end]

        def at(k: int):
            day = end - pd.Timedelta(days=k)
            return float(share.loc[day]) if day in share.index and _ok(share.loc[day]) else None

        def back(k: int):
            day = end - pd.Timedelta(days=k)
            return float(price.iloc[-1] / price.loc[day] - 1) * 100 if day in price.index else None

        # what the blend kept of the last 90 days' move (position decided the close before each day)
        win = price.loc[end - pd.Timedelta(days=90):]
        sh = share.reindex(win.index).shift(1).fillna(0.0).values
        chg = win.pct_change().fillna(0.0).values
        kept = float((np.prod(1 + sh * chg) - 1) * 100) if len(win) > 1 else None
        best_rank = min((members.index(s) + 1 for members in res["unis"][(50, "pit")].values() if s in members))
        dead.append({"coin": s, "end": end, "first": p["first"][s], "peak": float(price.max()),
                     "fall": float(price.iloc[-1] / price.max() - 1) * 100, "s90": at(90), "s30": at(30),
                     "s0": at(0), "b30": back(30), "b90": back(90), "kept90": kept, "rank": best_rank,
                     "in20": any(s in m for m in res["unis"][(20, "pit")].values()),
                     "status": p["status"].get(s)})
    res["dead"] = dead

    # ----- each coin on its own: blend vs holding -----
    per = {}
    for cost in (COST, COST_HIGH):
        rows = []
        for s in sorted(in50):
            first = max(p["first"][s] + pd.Timedelta(days=MIN_DAYS), pd.Timestamp(START))
            end = p["last"][s]
            if end - first < pd.Timedelta(days=60):
                continue
            price = p["close"][s].loc[first - pd.Timedelta(days=1):end]
            share = blend[s].loc[first - pd.Timedelta(days=1):end]
            hold, bl = _single(price, share, cost)
            group = "renamed" if s in RENAMED and dead_flag[s] else ("dead" if dead_flag[s] else "alive")
            rows.append({"coin": s, "group": group, "hold": hold, "blend": bl})
        per[cost] = rows
    res["per"] = per
    res["blend"], res["btc_above"], res["btc_blend"] = blend, btc_above, btc_blend
    res["index"], res["ever"], res["in50"] = index, ever, in50
    return res


# ---------- the study ----------

NAMES = {"basket": "整籃持有（每月重選、平均分配）", "blend": "每種幣各自照混合規則", "filter": "整籃＋大盤濾網（比特幣高於 50 日均線才持有）",
         "btcblend": "整籃×比特幣混合比例", "top3": "最強 3 種（過去 4 週）", "top3f": "最強 3 種＋大盤濾網",
         "top5": "最強 5 種（過去 4 週）", "top5f": "最強 5 種＋大盤濾網"}


def study() -> dict:
    data = _load()
    p = _panels(data["raw"], data["pairs"]["status"])
    p["status"] = data["pairs"]["status"]
    r = _compute(p)
    runs, index = r["runs"], r["index"]
    last = p["days"][-1]
    years = sorted(runs[(20, "pit", "basket", COST, 0.0)]["years"])
    partial = last < pd.Timestamp(f"{years[-1]}-12-31")

    # remember the members now in force so later runs only top up those pairs mid-month
    latest = {n: max(r["unis"][(n, "pit")]) for n in NS}
    now_members = sorted(set(r["unis"][(50, "pit")][latest[50]]) | set(r["unis"][(50, "surv")][max(r["unis"][(50, "surv")])]))
    _write(DIR / "members.json", now_members)

    def run(n, pool, key, cost=COST, h=0.0):
        return runs[(n, pool, key, cost, h)]

    def year_name(y: int) -> str:
        return f"{y}（到 {last.strftime('%m-%d')}）" if y == years[-1] and partial else str(y)

    # ----- table 1: survivorship effect -----
    t1 = []
    for y in years:
        row = [year_name(y)]
        for n in NS:
            a, b = run(n, "pit", "basket")["years"][y], run(n, "surv", "basket")["years"][y]
            row += [_pct(a), _pct(b), _pct(b - a)]
        t1.append(row)

    def _diff(a, b):
        return None if a is None or b is None else b - a

    for name, _, _ in PERIODS:
        row = [f"{name}：整籃每年"]
        for n in NS:
            a, b = run(n, "pit", "basket")[name]["ret"], run(n, "surv", "basket")[name]["ret"]
            row += [_pct(a), _pct(b), _pct(_diff(a, b))]
        t1.append(row)
    for key, label in (("basket", "2019 起整籃每年"), ("blend", "2019 起各幣混合每年")):
        row = [label]
        for n in NS:
            a, b = run(n, "pit", key)["all"]["ret"], run(n, "surv", key)["all"]["ret"]
            row += [_pct(a), _pct(b), _pct(_diff(a, b))]
        t1.append(row)
    for key, label in (("basket", "整籃：1 美元變成"), ("blend", "各幣混合：1 美元變成")):
        row = [label]
        for n in NS:
            a, b = run(n, "pit", key)["all"]["growth"], run(n, "surv", key)["all"]["growth"]
            row += [_money(a), _money(b), "—"]
        t1.append(row)

    # ----- table 2: strategies on the point-in-time lists -----
    t2 = []
    for n in NS:
        for key in ("basket", "blend", "filter", "btcblend", "top3", "top3f", "top5", "top5f"):
            m = run(n, "pit", key)
            hi = run(n, "pit", key, COST_HIGH)
            parts = "｜".join(_pct(m[name]["ret"]) for name, _, _ in PERIODS)
            harsh = f"{_pct(run(n, 'pit', key, COST, 0.5)['all']['ret'])}／{_pct(run(n, 'pit', key, COST, 1.0)['all']['ret'])}"
            t2.append([f"前 {n} 名：{NAMES[key]}", _pct(m["all"]["ret"]), _num(m["all"]["ratio"]),
                       _pct(m["all"]["deepest"]), _money(m["all"]["growth"]), parts, _pct(hi["all"]["ret"]), harsh])

    # ----- table 3: coins that stopped trading -----
    dead = sorted(r["dead"], key=lambda d: (d["rank"], d["end"]))
    nd = len(dead)
    flat30 = sum(1 for d in dead if d["s30"] is not None and d["s30"] <= 1e-9)
    flat90 = sum(1 for d in dead if d["s90"] is not None and d["s90"] <= 1e-9)
    flat0 = sum(1 for d in dead if d["s0"] is not None and d["s0"] <= 1e-9)
    low30 = sum(1 for d in dead if d["s30"] is not None and d["s30"] < 0.25)
    have30 = sum(1 for d in dead if d["s30"] is not None)
    have90 = sum(1 for d in dead if d["s90"] is not None)
    collapsed = [d for d in dead if d["fall"] <= -90]
    flat30_c = sum(1 for d in collapsed if d["s30"] is not None and d["s30"] <= 1e-9)
    t3 = []
    for d in dead[:16]:
        t3.append([_base(d["coin"]), _d(d["end"]), _pct(d["fall"]),
                   _pct(d["s90"] * 100 if d["s90"] is not None else None, signed=False),
                   _pct(d["s30"] * 100 if d["s30"] is not None else None, signed=False),
                   _pct(d["s0"] * 100 if d["s0"] is not None else None, signed=False),
                   _pct(d["b90"]), _pct(d["kept90"])])
    med_b90 = float(np.median([d["b90"] for d in dead if d["b90"] is not None])) if dead else None
    med_k90 = float(np.median([d["kept90"] for d in dead if d["kept90"] is not None])) if dead else None

    # ----- table 4: each coin on its own -----
    t4 = []
    groups = (("alive", "現在還在交易的幣"), ("dead", "後來停止交易的幣"), ("renamed", "改名／換成新代號的幣"))
    counts = {}
    for cost in (COST, COST_HIGH):
        for g, label in groups + (("all", "全部"),):
            rows = [x for x in r["per"][cost] if g == "all" or x["group"] == g]
            if not rows:
                continue
            k = len(rows)
            money = sum(1 for x in rows if x["blend"]["growth"] > x["hold"]["growth"])
            ratio = sum(1 for x in rows if x["blend"]["ratio"] > x["hold"]["ratio"])
            deep = sum(1 for x in rows if x["blend"]["deepest"] > x["hold"]["deepest"])
            counts[(cost, g)] = (k, money, ratio, deep)
            t4.append([f"{label}（成本 {cost * 100:.2f}%）", f"{k}", f"{k} 個裡 {money} 個", f"{k} 個裡 {ratio} 個",
                       f"{k} 個裡 {deep} 個", _money(float(np.median([x["hold"]["growth"] for x in rows]))),
                       _money(float(np.median([x["blend"]["growth"] for x in rows])))])

    # ----- data coverage -----
    info, sel, pairs = data["info"], data["sel"], data["pairs"]
    n_coins = len(p["coins"])
    n_dead_all = sum(1 for s in p["coins"] if p["ended"][s])
    n_renamed = sum(1 for s in p["coins"] if p["ended"][s] and s in RENAMED)
    months_cov = sum(len(set(f.index.strftime("%Y-%m"))) for f in data["raw"].values())
    span = f"{_d(index[0])} 到 {_d(last)}"
    status_dead = sum(1 for s in pairs["symbols"] if pairs["status"].get(s) != "TRADING")

    tables = [
        {"caption": (f"倖存者偏誤（只看現在還在的幣，等於事先知道誰沒倒）有多大：每月最後一天收盤，依過去 {VOL_DAYS} 天成交額選出前 N 名"
                     f"（要有 {MIN_DAYS} 天以上的日線、還在交易），平均分配、整個月抱著，停止交易的幣照最後收盤換成現金。"
                     "「當時名單」包含後來消失的幣；「只用活著的」是同樣做法，但只從今天還在交易的幣裡選。差＝只用活著的減當時名單"
                     f"（正數＝只用活著的會把成果算得比較好）。成本每次買賣 {COST * 100:.2f}%，{span}。"),
         "columns": ["年度", "前 20 名：當時名單", "前 20 名：只用活著的", "差", "前 50 名：當時名單", "前 50 名：只用活著的", "差"],
         "rows": t1[:16]},
        {"caption": (f"各種做法都用「當時名單」（{span}）。每年報酬、報酬÷波動、最深曾經虧多少、1 美元最後變成多少都是成本 {COST * 100:.2f}%、"
                     "停止交易照最後收盤出場；後面三欄是分段的每年報酬、成本改成 0.30% 的每年報酬，以及假設停止交易的幣出場時再虧 50%／100%"
                     "（全部歸零）時的每年報酬。「各自照混合規則」＝每種幣分到一樣多的錢，17 條順勢規則裡幾成看多，就把那份錢的幾成放在那種幣，"
                     "其餘現金；「大盤濾網」＝比特幣收盤低於 50 日均線時整籃抱現金；「×比特幣混合比例」＝整籃只持有比特幣 17 條規則看多的比例；"
                     "「最強 N 種」＝每週日從當月名單裡挑過去 4 週漲最多的 N 種。"),
         "columns": ["做法", "每年報酬", "報酬÷波動", "最深曾經虧多少", "1 美元變成", "2019–21｜2022–23｜2024 起", "成本 0.30%",
                     "下市再虧 50%／100%"],
         "rows": t2[:16]},
        {"caption": (f"曾經進過「前 50 名」、後來停止交易的幣（不含只是改名或換新代號的）共 {nd} 個，表裡列出排名最高的 {min(nd, 16)} 個。"
                     f"「規則持有」是 17 條規則裡看多的比例（0% ＝空手）。停止交易前 30 天：{have30} 個裡 {flat30} 個已經空手、"
                     f"{low30} 個持有不到 25%；前 90 天：{have90} 個裡 {flat90} 個已經空手；最後一天：{nd} 個裡 {flat0} 個空手。"
                     f"其中從高點跌了九成以上的 {len(collapsed)} 個裡，{flat30_c} 個在前 30 天已經空手。"
                     f"最後 90 天幣價的中位數是 {_pct(med_b90)}，同期照混合規則持有的中位數是 {_pct(med_k90)}。"),
         "columns": ["幣", "最後交易日", "從最高收盤跌了", "前 90 天規則持有", "前 30 天", "最後一天", "最後 90 天幣價", "最後 90 天照規則"],
         "rows": t3},
        {"caption": ("每種曾經進過前 50 名的幣各自計算（從它有 90 天日線、而且 2019 年以後開始，到它最後一根日線）：照混合規則持有，"
                     "和一直抱著比，最後的錢、報酬÷波動、最深虧損，各有幾個比較好。停止交易的幣照最後收盤結算、不另外扣損失。"),
         "columns": ["哪些幣", "幣數", "混合最後的錢比較多", "報酬÷波動比較高", "最深虧損比較淺", "一直抱著：1 美元變成（中位數）",
                     "混合：1 美元變成（中位數）"],
         "rows": t4[:16]},
    ]

    # ----- now -----
    t_now = latest[20]
    top20 = r["unis"][(20, "pit")][t_now]
    shares = {s: r["blend"][s].loc[last] for s in top20}
    holding = [s for s in top20 if _ok(shares[s]) and shares[s] > 0]
    full = [s for s in top20 if _ok(shares[s]) and shares[s] >= 0.999]
    btc_close = float(p["close"][BTC].loc[last])
    btc_mean = float(p["close"][BTC].rolling(50).mean().loc[last])
    listing = "、".join(f"{_base(s)} {shares[s] * 100:.0f}%" if _ok(shares[s]) else f"{_base(s)} —" for s in top20)
    now = [
        {"label": f"最後一根收完的日線", "value": _d(last)},
        {"label": f"現在的前 20 名（{_d(t_now)} 收盤依過去 {VOL_DAYS} 天成交額選出）與各自的混合規則持有比例（{_d(last)} 收盤）",
         "value": listing},
        {"label": "前 20 名裡混合規則有持有的", "value": f"20 個裡 {len(holding)} 個（其中 {len(full)} 個 17 條全部看多）"},
        {"label": "大盤濾網", "value": f"比特幣收盤 {btc_close:,.0f}，{'高於' if btc_close > btc_mean else '低於'} 50 日均線 {btc_mean:,.0f}："
                                    f"{'持有' if btc_close > btc_mean else '抱現金'}"},
        {"label": "比特幣 17 條規則看多的比例", "value": f"{float(r['btc_blend'].loc[last]) * 100:.0f}%"},
        {"label": "資料涵蓋", "value": f"{n_coins} 個 USDT 交易對（其中 {n_dead_all} 個已停止交易），{months_cov:,} 個幣月"},
    ]

    # ----- verdict -----
    b20, l20 = run(20, "pit", "basket")["all"], run(20, "pit", "blend")["all"]
    b20h, l20h = run(20, "pit", "basket", COST, 1.0)["all"], run(20, "pit", "blend", COST, 1.0)["all"]
    s20, sl20 = run(20, "surv", "basket")["all"], run(20, "surv", "blend")["all"]
    f20 = run(20, "pit", "filter")["all"]
    gap_b = s20["ret"] - b20["ret"]
    gap_l = sl20["ret"] - l20["ret"]
    verdict = (f"把後來消失的幣加回去、每月依當時成交額選前 20 名，2019 起整籃持有每年 {_pct(b20['ret'])}、最深虧 {_pct(b20['deepest'])}，"
               f"每種幣各自照混合規則是每年 {_pct(l20['ret'])}、最深虧 {_pct(l20['deepest'])}"
               f"（下市幣全部歸零時 {_pct(b20h['ret'])} 對 {_pct(l20h['ret'])}；大盤濾網是 {_pct(f20['ret'])}）。"
               f"只用現在還活著的幣（等於事先知道誰沒倒）會把整籃每年報酬{'高' if gap_b > 0 else '低'}估 {abs(gap_b):.1f} 個百分點、"
               f"混合{'高' if gap_l > 0 else '低'}估 {abs(gap_l):.1f} 個百分點。"
               f"曾進前 50 名、後來停止交易的 {nd} 個裡 {flat30} 個在下市前 30 天已經空手、{low30} 個持有不到四分之一，"
               f"最後 90 天幣價中位數 {_pct(med_b90)}，照規則是 {_pct(med_k90)}。")

    # ----- notes -----
    over = len(sel["over_cap"])
    src_text = ("幣安公開歷史資料庫（data.binance.vision）的檔案清單" if pairs.get("source") == "archive"
                else "幣安 exchangeInfo 的交易對清單（歷史資料庫的清單這次讀不到，已經完全移除的交易對可能缺）")
    notes = [
        f"資料涵蓋：交易對清單取自{src_text}，共 {pairs.get('listed', '—')} 個交易對；只留 USDT 交易對，去掉穩定幣與掛鉤代幣"
        "（USDC、BUSD、TUSD、DAI、FDUSD、USDP、歐元、PAXG、WBTC、WBETH、BETH 等，另外收盤價一直貼著 1 美元的也當成穩定幣去掉）、"
        "法幣、槓桿代幣（UP／DOWN／BULL／BEAR），以及 2026 年起上架、代表美國股票或 ETF 的代幣（例如 NVDAB、SPCXB），"
        f"剩 {len(pairs['symbols'])} 個（其中 {status_dead} 個現在已停止交易）。每個先用一次月線請求看每月成交額，"
        f"{SCREEN_FROM} 起只要有一個月（上架頭三個月不算，因為還不滿 90 天、不能進名單）成交額排進前 {SCREEN_RANK} 名，就下載日線"
        f"（{sel['kept']} 個{'' if not over else f'，另有 {over} 個超過 {MAX_SYMBOLS} 個上限沒有下載'}）；"
        f"從沒排進前 {SCREEN_RANK} 名的幣，月底依 30 天成交額選前 50 名時幾乎不可能入選，所以沒有下載（這是近似，沒有逐一確認）。實際用到 {n_coins} 個交易對的日線、"
        f"{months_cov:,} 個幣月，其中 {n_dead_all} 個已經停止交易（{n_renamed} 個只是改名或換新代號）。"
        + (f"這次有 {len(info['skipped'])} 個因為時間或連線沒有取到，下次執行會補：{'、'.join(_base(s) for s in info['skipped'][:20])}。"
           if info["skipped"] else "")
        + (f"{len(data['unscreened'])} 個這次還沒篩到，下次執行會補。" if data["unscreened"] else ""),
        "日線來自幣安公開的 K 線端點（已下市、狀態為暫停的交易對也還查得到）；端點查不到的交易對改讀歷史資料庫的每月壓縮檔"
        f"{'（這次有 ' + str(len(info['archive'])) + ' 個）' if info['archive'] else ''}。資料存在本機，之後每月初重抓一次全部還在交易的幣，"
        "月中只更新當月名單裡的幣。",
        f"停止交易的認定：日線中斷超過 {GAP_DAYS} 天、或現在狀態不是交易中，就當作在最後一根日線收盤停止交易，之後就算重新出現（例如 Terra 舊幣"
        "代號被新幣沿用、FTX 平台幣中斷後恢復）也不再算回名單。基本情況照最後收盤價換成現金（現實中幣安下市前會先公告，所以假設能在最後收盤賣出）；"
        "另外列出出場時再虧 50% 和 100%（全部歸零）的版本。只是改名或換新代號的（例如 MATIC→POL、FTM→S、EOS→A、MKR→SKY、"
        "RNDR→RENDER、AGIX／OCEAN→FET）不算消失、出場時不另外扣損失，但這份對照表是人工整理、不保證完整。",
        "只有一家交易所，不等於整個市場：幣安通常在幣已經漲過一段、有熱度之後才上架，很多幣在別的交易所早就跌到歸零、但從沒上過幣安，"
        "這裡看不到；也有幣只是從幣安下架、在別處繼續交易。所以這裡加回的「消失的幣」只是一部分。",
        f"成本只用粗略的數字：每次買賣扣成交金額的 {COST * 100:.2f}%（另列 0.30%）。成交額排在後面的小幣，實際的價差和滑價常常比這高很多，"
        "崩跌當下更可能根本賣不到收盤價，所以小幣和下市幣的成本是低估的。現金沒有利息，也沒有算稅。",
        "混合規則和其他研究一樣是 17 條：收盤高於 N 日均線（N = 10、20、30、50、75、100、150、200）、創 N 日新高後看多／跌破 N/2 日新低後"
        "不看多（N = 10、20、50、100）、過去 L 天是漲的（L = 14、30、60、90、180）。新上架的幣日線不夠長時，只用已經算得出來的那幾條規則的平均"
        "（進名單至少要 90 天，那時大約有 12 條可以算），不讓新幣自動變成現金，以免把「新幣上市後常跌」錯算成規則的功勞。",
        "時間的算法：所有選幣和持有比例都只用到當天收盤為止的資料（幣安日線在 UTC 0 點收盤），隔天才照做；名單每月最後一天收盤重選，"
        "「最強 N 種」每週日收盤重選。報酬、波動和最深虧損都用每日資料，一年用 365 天換算。",
        f"真正獨立的觀察很少：2019 年起只有 {len(years)} 個年度{'（最後一年還沒過完）' if partial else ''}，大致就是兩三輪大漲大跌，"
        "而且這些幣多半跟著比特幣一起漲跌，「幾個幣裡幾個」看起來很多，其實是同幾段行情重複計算。差距不大的地方不該當成有差。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在的狀態是什麼，不是買賣建議。",
    ]

    return {
        "id": "survivorship",
        "title": "把後來消失的幣加回去，順勢還有用嗎",
        "question": ("如果幣的名單照「當時」的樣子重建——包含後來崩掉或下市的幣——順勢做法還有沒有幫助？只看現在還活著的幣會把成果算錯多少？"
                     "挑最強的幣會不會看起來不一樣？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安公開歷史資料庫（現貨日線，含已下市交易對）", "url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/"},
            {"name": "幣安現貨 K 線端點說明", "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
