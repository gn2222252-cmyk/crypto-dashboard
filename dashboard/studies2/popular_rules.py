"""Historical study "popular_rules": the bitcoin buy/sell rules that are repeated most often online - long-term
averages, cycle-top indicators, on-chain "value zones", the halving clock, monthly buying - written down as exact
rules with their commonly quoted thresholds and run day by day. Did following them beat simply holding, were the
famous thresholds known in advance, and were the sell thresholds still reached in the latest cycle?

No look-ahead: a signal uses data up to and including the close of day t and decides the position held during day
t+1. The two on-chain series (MVRV, miner revenue) are used one day late because they are published after the day
ends. Only closed daily candles are used. Thresholds are the commonly quoted ones, not tuned here.

Sources (public, key-less, through common.get_json): blockchain.info daily price since 2010 + Binance spot daily
closes (classic._long_close), CoinMetrics community API (MVRV), blockchain.info miners revenue, alternative.me
fear & greed. The two on-chain downloads are kept in cache/_store/ as a fallback for a run where the source fails.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BCI_REVENUE = "https://api.blockchain.info/charts/miners-revenue"
BCI_HEIGHT = "https://blockchain.info/q/getblockcount"
FNG = "https://api.alternative.me/fng/"
COST = 0.001  # 0.10% of the amount moved at every switch between bitcoin and cash (and at every monthly purchase)
YEAR = 365
FLOOR = "2011-01-01"  # nothing is measured before this day (earlier prices and on-chain values are too thin)
COMMON = "2018-01-01"  # the comparison window shared by every rule
CYCLES = (("2013–2016", "2013-01-01", "2016-12-31"), ("2017–2020", "2017-01-01", "2020-12-31"),
          ("2021–2024", "2021-01-01", "2024-12-31"), ("2025 至今", "2025-01-01", None))
HALVINGS = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")
NEXT_HALVING_BLOCK = 1_050_000
NEXT_HALVING_FALLBACK = "2028-04-15"
LAG = 1  # days the on-chain series are held back
MAX_LISTED = 8  # a rule with at most this many signals gets every signal listed
MONTHLY = 100.0  # the monthly amount of the regular-buying variants (any amount gives the same multiples)
UNSURE = "年份不確定"


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
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
    return f"{float(v):,.{digits}f}"


def _times(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return f"{v:,.0f} 倍" if v >= 100 else f"{v:.1f} 倍" if v >= 10 else f"{v:.2f} 倍"


def _usd(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return f"{v:,.0f} 美元" if abs(v) >= 100 else f"{v:,.2f} 美元"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed daily closes since 2010 on a DatetimeIndex with no missing day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


def _stored(name: str, fetch) -> tuple[pd.Series, str]:
    """fetch() -> {YYYY-MM-DD: number}. A good download replaces the stored copy; a failed one falls back to it."""
    path = common.STORE / f"popular_rules_{name}.json"
    note = ""
    try:
        days = fetch()
        if len(days) < 3000:
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
    series = pd.Series({pd.Timestamp(day): float(value) for day, value in days.items()}).sort_index()
    return series[np.isfinite(series.values)], note


def _fetch_mvrv() -> dict:
    out, url, params = {}, CM_URL, {"assets": "btc", "metrics": "CapMVRVCur", "frequency": "1d", "page_size": 10000}
    for _ in range(10):
        reply = common.get_json(url, params, timeout=60)
        for row in reply.get("data", []):
            try:
                out[str(row["time"])[:10]] = float(row["CapMVRVCur"])
            except (KeyError, TypeError, ValueError):
                continue
        url, params = reply.get("next_page_url"), None
        if not url:
            break
    return out


def _fetch_revenue() -> dict:
    reply = common.get_json(BCI_REVENUE, {"timespan": "all", "format": "json", "sampled": "false"}, timeout=60)
    return {pd.Timestamp(p["x"], unit="s").strftime("%Y-%m-%d"): float(p["y"]) for p in reply["values"]}


def _fear_greed() -> pd.Series:
    """alternative.me fear & greed index, one value per UTC day (published at the start of that day)."""
    reply = common.get_json(FNG, {"limit": 0})
    rows = {}
    for item in reply.get("data", []):
        try:
            rows[pd.Timestamp(int(item["timestamp"]), unit="s").normalize()] = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
    series = pd.Series(rows).sort_index()
    if len(series) < 1000:
        raise RuntimeError(f"恐懼與貪婪指數只回傳 {len(series)} 筆")
    return series


def _next_halving() -> tuple[pd.Timestamp, str]:
    """Estimated day of the next halving: blocks still to go at ten minutes each. Always an estimate."""
    try:
        height = int(common.get_json(BCI_HEIGHT, timeout=20))
        left = NEXT_HALVING_BLOCK - height
        if not 0 < left < 210_000:
            raise ValueError(height)
        day = (pd.Timestamp(common.today_utc()) + pd.Timedelta(minutes=10 * left)).normalize()
        return day, f"用現在的區塊高度 {height:,}、還差 {left:,} 個區塊、每個區塊 10 分鐘推算"
    except Exception:  # noqa: BLE001
        return pd.Timestamp(NEXT_HALVING_FALLBACK), "這次查不到區塊高度，先用常見的估計日"


# ---------- rules ----------

def _latch(buy: pd.Series, sell: pd.Series, valid: pd.Series, start: float = 0.0) -> pd.Series:
    """1 after a buy signal, 0 after a sell signal, unchanged in between; NaN while the indicator does not exist."""
    state = pd.Series(np.where(buy, 1.0, np.where(sell, 0.0, np.nan)), index=valid.index).where(valid.notna())
    return state.ffill().fillna(start).where(valid.notna())


def _events(signal: pd.Series, close: pd.Series) -> list[tuple]:
    """(day, +1 buy / -1 sell, close of that day) each time the signal changes; a rule starts from cash."""
    s = signal.dropna()
    move = s.diff()
    if len(s):
        move.iloc[0] = s.iloc[0]
    return [(day, int(step), float(close[day])) for day, step in move[move != 0].items()]


def _pi_cycle(close: pd.Series, ratio: pd.Series, ma200: pd.Series) -> tuple[pd.Series, list]:
    """Holding from the first day; sell on the cross; back in above the 200-day average after a 50% fall (ours)."""
    out = np.full(len(close), np.nan)
    state, sold, low, depth = 1.0, None, None, []
    c, r, m = close.values, ratio.values, ma200.values
    for i in range(len(c)):
        if not math.isfinite(r[i]):
            continue
        if state == 1.0:
            if i and math.isfinite(r[i - 1]) and r[i - 1] <= 1.0 < r[i]:
                state, sold, low = 0.0, c[i], c[i]
        else:
            low = min(low, c[i])
            if low <= 0.5 * sold and c[i] > m[i]:
                state = 1.0
                depth.append((close.index[i], low / sold - 1))
        out[i] = state
    pending = {"sold": sold, "low": low} if state == 0.0 else None
    return pd.Series(out, index=close.index), [depth, pending]


def _build(close: pd.Series, mvrv: pd.Series | None, revenue: pd.Series | None, halving_next: pd.Timestamp) -> dict:
    """Every indicator and every rule's signal (the state at each day's close). Kept apart from the formatting."""
    idx = close.index
    ind: dict[str, pd.Series] = {}
    ma200 = close.rolling(200, min_periods=200).mean()
    ma1400 = close.rolling(1400, min_periods=1400).mean()  # 200 weeks, updated every day
    ma111 = close.rolling(111, min_periods=111).mean()
    ma350 = close.rolling(350, min_periods=350).mean()
    ma730 = close.rolling(730, min_periods=730).mean()
    ind["mayer"], ind["w200"], ind["pi"], ind["ma2y"] = close / ma200, close / ma1400, ma111 / (2 * ma350), close / ma730

    weekly = close.resample("W-SUN").last().dropna()  # the same weekly bars as the dashboard's support band
    if weekly.index[-1] > idx[-1]:
        weekly = weekly.iloc[:-1]  # the current week has not closed yet
    sma20 = weekly.rolling(20).mean()
    ema21 = weekly.ewm(span=21, adjust=False, min_periods=21).mean()
    upper = np.maximum(sma20, ema21)
    band_weekly = (weekly > upper).astype(float).where(upper.notna())

    signals: dict[str, pd.Series] = {}
    extra: dict = {"ma200": ma200, "ma1400": ma1400, "ma111": ma111, "ma350": ma350, "ma730": ma730,
                   "band_upper": float(upper.iloc[-1]), "band_week": weekly.index[-1]}
    signals["mayer"] = _latch(ind["mayer"] < 1.0, ind["mayer"] > 2.4, ind["mayer"])
    signals["mayer_cap"] = (ind["mayer"] < 2.4).astype(float).where(ind["mayer"].notna())
    near = close <= 1.1 * ma1400
    signals["w200"] = _latch(near, ind["w200"] > 3.0, ind["w200"])
    signals["w200_keep"] = _latch(near, pd.Series(False, index=idx), ind["w200"])
    signals["band"] = band_weekly.reindex(idx, method="ffill")
    signals["pi"], extra["pi"] = _pi_cycle(close, ind["pi"], ma200)
    signals["ma2y"] = _latch(ind["ma2y"] < 1.0, ind["ma2y"] > 5.0, ind["ma2y"])
    if mvrv is not None:
        ind["mvrv"] = mvrv.reindex(idx).ffill(limit=5).shift(LAG)
        signals["mvrv"] = _latch(ind["mvrv"] < 1.0, ind["mvrv"] > 3.5, ind["mvrv"])
    if revenue is not None:
        rev = revenue.reindex(pd.date_range(revenue.index[0], revenue.index[-1], freq="D")).ffill(limit=5)
        puell = (rev / rev.rolling(365, min_periods=365).mean()).replace([np.inf, -np.inf], np.nan)
        ind["puell"] = puell.reindex(idx).ffill(limit=5).shift(LAG)
        signals["puell"] = _latch(ind["puell"] < 0.5, ind["puell"] > 4.0, ind["puell"])
    clock = pd.Series(0.0, index=idx)
    after = pd.Series(0.0, index=idx)
    for day in [pd.Timestamp(h) for h in HALVINGS] + [halving_next]:
        clock[(idx >= day - pd.Timedelta(days=500)) & (idx < day + pd.Timedelta(days=500))] = 1.0
        after[(idx >= day) & (idx < day + pd.DateOffset(months=18))] = 1.0
    signals["clock"], signals["after"] = clock, after
    high = close.rolling(20, min_periods=20).max()
    low = close.rolling(10, min_periods=10).min()
    signals["trend"] = _latch(close >= high, close <= low, high)
    signals["hold"] = pd.Series(1.0, index=idx)

    floor = pd.Timestamp(FLOOR)
    for key in list(signals):
        signals[key] = signals[key][signals[key].index >= floor]
        if signals[key].dropna().empty:
            del signals[key]
    return {"ind": ind, "signals": signals, "extra": extra}


# ---------- measuring ----------

def _measure(position: pd.Series, change: pd.Series, events: list, a, b=None) -> dict | None:
    """One rule over [a, b]: the position on the first day is carried in (no cost), later switches cost COST."""
    pos = position[(position.index >= pd.Timestamp(a)) & (position.index <= (pd.Timestamp(b) if b else position.index[-1]))]
    if len(pos) < 30 or pos.isna().any():
        return None  # the rule did not exist yet on the first day of the period
    turn = position.diff().abs().reindex(pos.index).fillna(0.0)
    turn.iloc[0] = 0.0
    chg = change.reindex(pos.index).fillna(0.0)
    net = (pos * chg - COST * turn).values
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    hold = float(np.prod(1 + chg.values))
    n = len(net)
    first, last = pos.index[0] - pd.Timedelta(days=1), pos.index[-1] - pd.Timedelta(days=1)  # signal days
    inside = [e for e in events if first <= e[0] <= last]
    return {"first": pos.index[0], "last": pos.index[-1], "days": n,
            "ret": (equity[-1] ** (YEAR / n) - 1) * 100,
            "hold_ret": (hold ** (YEAR / n) - 1) * 100,
            "deepest": float((equity / np.maximum.accumulate(equity) - 1).min() * 100),
            "share": float(pos.mean() * 100), "final": float(equity[-1]), "vs": float(equity[-1] / hold),
            "buys": sum(1 for e in inside if e[1] > 0), "sells": sum(1 for e in inside if e[1] < 0)}


def _dca(close: pd.Series, start: str, mode: str, ma200: pd.Series, fng: pd.Series | None) -> dict:
    """MONTHLY arrives on the first day of every month; what is not spent waits as cash and counts at face value."""
    days = [d for d in pd.date_range(start, close.index[-1], freq="MS") if d in close.index]
    cash = coins = paid = spent = 0.0
    bought = 0
    for day in days:
        cash += MONTHLY
        paid += MONTHLY
        before = day - pd.Timedelta(days=1)
        if mode == "plain":
            spend = cash
        elif mode == "below":  # judged on the previous day's close, bought at this day's close
            spend = cash if before in ma200.index and close[before] < ma200[before] else 0.0
        else:  # the index is published at the start of the day
            level = fng.get(day, np.nan)
            spend = 0.0 if level >= 75 else min(cash, 2 * MONTHLY) if level <= 25 else min(cash, MONTHLY)
        if spend > 0:
            coins += spend * (1 - COST) / close[day]
            cash -= spend
            spent += spend
            bought += 1
    value = coins * float(close.iloc[-1]) + cash
    return {"first": days[0], "months": len(days), "paid": paid, "value": value, "multiple": value / paid,
            "cash": cash, "bought": bought, "avg": spent / coins if coins else None}


# ---------- the study ----------

RULES = (  # key, short name with its thresholds, one-phrase explanation
    ("mayer", "梅耶倍數：低於 1.0 買、高於 2.4 賣", "現價 ÷ 200 日平均價"),
    ("mayer_cap", "梅耶倍數：低於 2.4 就持有", "現價 ÷ 200 日平均價"),
    ("w200", "200 週均線：貼近或跌破買、超過 3 倍賣", "現價 ÷ 過去 200 週（1,400 天）平均價"),
    ("w200_keep", "200 週均線：貼近或跌破買、不賣", "現價 ÷ 過去 200 週（1,400 天）平均價"),
    ("band", "牛市支撐帶：週收盤在帶上才持有", "20 週均線和 21 週指數均線之間的區域"),
    ("pi", "Pi 週期頂部：交叉時賣", "111 日均線 ÷（350 日均線 × 2），超過 1 就是交叉"),
    ("ma2y", "2 年均線乘數：低於均線買、高於 5 倍賣", "現價 ÷ 過去 730 天平均價"),
    ("mvrv", "MVRV：低於 1.0 買、高於 3.5 賣", "現價 ÷ 所有幣上一次移動時的平均成本"),
    ("puell", "普爾倍數：低於 0.5 買、高於 4 賣", "礦工當天收入 ÷ 過去 365 天的平均"),
    ("clock", "減半時鐘：減半前 500 天買、後 500 天賣", "只看日曆，不看價格"),
    ("after", "減半時鐘：減半當天買、18 個月後賣", "只看日曆，不看價格"),
)
PUBLISHED = {  # key -> (first day measured as "after it was public" or None, what can be said about the date)
    "mayer": ("2018-01-01", "2017（Trace Mayer）"),
    "mayer_cap": ("2018-01-01", "2017（Trace Mayer）"),
    "w200": (None, f"{UNSURE}（約 2018–2019 開始流傳；3 倍這條線沒有公認出處）"),
    "w200_keep": (None, f"{UNSURE}（約 2018–2019 開始流傳）"),
    "band": (None, f"{UNSURE}（約 2019–2020 開始流傳）"),
    "pi": ("2020-01-01", "2019（Philip Swift）"),
    "ma2y": ("2018-01-01", "2017（Philip Swift）"),
    "mvrv": ("2019-01-01", "2018（Mahmudov、Puell）"),
    "puell": ("2020-01-01", "2019（David Puell）"),
    "clock": (None, UNSURE),
    "after": (None, UNSURE),
}
PEAKS = (  # indicator key, row name, sell threshold, buy threshold (None = no fixed buy line), digits
    ("mayer", "梅耶倍數", 2.4, 1.0),
    ("w200", "現價 ÷ 200 週均線", 3.0, 1.1),
    ("pi", "Pi 週期：111 日均線 ÷（350 日均線 × 2）", 1.0, None),
    ("ma2y", "現價 ÷ 2 年均線", 5.0, 1.0),
    ("mvrv", "MVRV", 3.5, 1.0),
    ("puell", "普爾倍數", 4.0, 0.5),
)


def study() -> dict:
    close = _daily_close()
    last = close.index[-1]
    price = float(close.iloc[-1])
    problems: list[str] = []
    try:
        mvrv, note = _stored("mvrv", _fetch_mvrv)
        mvrv = mvrv[mvrv.index >= pd.Timestamp(FLOOR)]
        if note:
            problems.append("CoinMetrics 的 MVRV " + note + "。")
    except Exception as exc:  # noqa: BLE001 - the other rules still stand
        mvrv = None
        problems.append(f"CoinMetrics 的 MVRV 這次取不到（{type(exc).__name__}），MVRV 這條規則這次沒有算進去，沒有改用其他方法繞過。")
    try:
        revenue, note = _stored("miner_revenue", _fetch_revenue)
        if note:
            problems.append("blockchain.info 的礦工收入" + note + "。")
    except Exception as exc:  # noqa: BLE001
        revenue = None
        problems.append(f"blockchain.info 的礦工收入這次取不到（{type(exc).__name__}），普爾倍數這條規則這次沒有算進去。")
    try:
        fng = _fear_greed()
    except Exception as exc:  # noqa: BLE001
        fng = None
        problems.append(f"恐懼與貪婪指數這次取不到（{type(exc).__name__}），用到它的定期買進版本這次沒有算進去。")
    halving_next, halving_how = _next_halving()

    built = _build(close, mvrv, revenue, halving_next)
    ind, signals, extra = built["ind"], built["signals"], built["extra"]
    change = close.pct_change()
    position = {k: s.shift(1) for k, s in signals.items()}  # signal at the close of day t -> position during t+1
    events = {k: _events(s, close) for k, s in signals.items()}
    rules = [r for r in RULES if r[0] in signals]
    name = {k: n for k, n, _ in RULES}
    name.update({"hold": "一直抱著不動", "trend": "對照：創 20 日收盤新高進場、10 日收盤新低出場"})

    def run(key: str, a, b=None):
        return _measure(position[key], change, events[key], a, b)

    def first_day(key: str) -> pd.Timestamp:
        return position[key].first_valid_index()

    recent = {k: run(k, COMMON) for k in signals}
    whole = {k: run(k, first_day(k)) for k in signals}

    # ----- table 1: scoreboard -----
    def since_public(key: str) -> tuple[str, dict | None]:
        start, _ = PUBLISHED[key]
        m = run(key, start) if start else None
        if not m:
            return "—", None
        return (f"{start[:4]} 年起：每年 {_pct(m['ret'])}（抱著 {_pct(m['hold_ret'])}），最後的錢是抱著的 {_times(m['vs'])}，"
                f"完成 {m['sells']} 次"), m

    def trips(m: dict) -> str:
        return f"只有 {m['sells']} 次" if m["sells"] <= 8 else f"{m['sells']} 次"

    score_rows = []
    public: dict[str, dict | None] = {}
    for key in ["hold", "trend"] + [r[0] for r in rules]:
        m = recent[key]
        if key in PUBLISHED:
            text, public[key] = since_public(key)
            when = PUBLISHED[key][1]
        else:
            text, when = "—", "—"
        score_rows.append([name[key], "—" if key == "hold" else trips(m), _pct(m["ret"]), _pct(m["deepest"]),
                           f"{m['share']:.0f}%", "—" if key == "hold" else _times(m["vs"]), when, text])

    # ----- table 2: indicator extremes by cycle -----
    lows = [close[str(y)].idxmin() for y in (2015, 2018, 2022)]
    edges = [pd.Timestamp("2012-01-01")] + lows + [last + pd.Timedelta(days=1)]
    tops = [close[(close.index >= edges[i]) & (close.index < edges[i + 1])].idxmax() for i in range(4)]
    top_names = [f"{t.year} 年那一輪" for t in tops]
    top_names[-1] += "（還沒走完）"
    peak_rows = []
    reached_sell: dict[str, list[bool]] = {}
    for key, label, sell_at, buy_at in PEAKS:
        if key not in ind:
            continue
        series = ind[key]
        highs = [series[(series.index >= edges[i]) & (series.index < edges[i + 1])].max() for i in range(4)]
        hit = [bool(_ok(h) and h > sell_at) for h in highs]
        reached_sell[key] = hit
        peak_rows.append([f"{label}：漲勢中最高", f"賣出線 {sell_at:g}"] + [_num(h) for h in highs]
                         + [f"4 輪裡 {sum(hit)} 輪" + ("（最近一輪沒碰到）" if not hit[-1] else "")])
        if buy_at is not None:
            floors = [series[(series.index > tops[i]) & (series.index <= (tops[i + 1] if i < 3 else last))].min()
                      for i in range(4)]
            got = [bool(_ok(f) and f < buy_at) for f in floors]
            peak_rows.append([f"{label}：高點之後最低", f"買進線 {buy_at:g}"] + [_num(f) for f in floors]
                             + [f"4 輪裡 {sum(got)} 輪" + ("（最近一輪還沒碰到）" if not got[-1] else "")])

    # ----- table 3: regular monthly buying -----
    ma200 = extra["ma200"]
    dca_rows, dca = [], {}
    dca_modes = [("plain", "每月固定買"), ("below", "低於 200 日均線才買，否則存著，下次低於時一次買完")]
    if fng is not None:
        dca_modes.append(("fng", "恐懼貪婪指數 ≤ 25 買兩倍、≥ 75 不買（錢不夠兩倍就買到手上有的為止）"))
    for start in ("2018-02-01", "2014-01-01"):
        for mode, label in dca_modes:
            if mode == "fng" and start != "2018-02-01":
                continue
            r = _dca(close, start, mode, ma200, fng)
            dca[(start, mode)] = r
            dca_rows.append([label, f"{start[:7]} 起 {r['months']} 個月", _usd(r["paid"]), _usd(r["value"]),
                             _times(r["multiple"]), _usd(r["cash"]), f"{r['bought']} 個月", _usd(r["avg"])])

    # ----- table 4: where every rule stands today -----
    def last_event(key: str) -> str:
        if not events[key]:
            return "還沒有出現過訊號"
        day, step, at = events[key][-1]
        return f"{_d(day)} {'買進' if step > 0 else '賣出'}（收盤 {_usd(at)}）"

    def stance(key: str) -> str:
        return "持有比特幣" if signals[key].iloc[-1] == 1 else "持有現金，等買進訊號"

    def today(key: str) -> tuple[str, str]:
        """(the indicator's number today, what would trigger the next signal) - a description of the rule's state."""
        long = signals[key].iloc[-1] == 1
        m2, m14, m73 = float(ma200.iloc[-1]), float(extra["ma1400"].iloc[-1]), float(extra["ma730"].iloc[-1])
        if key in ("mayer", "mayer_cap"):
            value = f"{_num(ind['mayer'].iloc[-1])}（200 日均線 {_usd(m2)}）"
            if long:
                return value, f"收盤高過 200 日均線的 2.4 倍（現在約 {_usd(m2 * 2.4)}）時賣出"
            line = m2 * (1.0 if key == "mayer" else 2.4)
            return value, f"收盤回到 200 日均線的 {'1.0' if key == 'mayer' else '2.4'} 倍以下（現在約 {_usd(line)}）時買進"
        if key in ("w200", "w200_keep"):
            value = f"{_num(ind['w200'].iloc[-1])}（200 週均線 {_usd(m14)}）"
            if long:
                return value, ("這個版本買了就不賣" if key == "w200_keep"
                               else f"收盤高過 200 週均線的 3 倍（現在約 {_usd(m14 * 3)}）時賣出")
            return value, f"收盤跌到 200 週均線的 1.1 倍以下（現在約 {_usd(m14 * 1.1)}）時買進"
        if key == "band":
            value = f"上一個週收盤（{_d(extra['band_week'])}）{_usd(float(close[extra['band_week']]))}，帶的上緣 {_usd(extra['band_upper'])}"
            return value, ("週收盤跌到帶的上緣以下時賣出" if long else "週收盤站上 20 週均線和 21 週指數均線兩條線時買進") + \
                f"（現在約 {_usd(extra['band_upper'])}，每週會變）"
        if key == "pi":
            a, b = float(extra["ma111"].iloc[-1]), float(extra["ma350"].iloc[-1]) * 2
            value = f"{_num(ind['pi'].iloc[-1])}（111 日均線 {_usd(a)}，350 日均線的 2 倍 {_usd(b)}）"
            if long:
                return value, f"111 日均線往上穿過 350 日均線的 2 倍時賣出（還差 {_pct((b / a - 1) * 100, signed=False)}）"
            pending = extra["pi"][1]
            fell = pending["low"] <= 0.5 * pending["sold"]
            return value, (f"先從賣出價 {_usd(pending['sold'])} 跌掉一半（到 {_usd(pending['sold'] / 2)} 以下，"
                           f"{'已經發生' if fell else '還沒發生'}），再收盤站回 200 日均線（{_usd(m2)}）時買回")
        if key == "ma2y":
            value = f"{_num(ind['ma2y'].iloc[-1])}（2 年均線 {_usd(m73)}）"
            return value, (f"收盤高過 2 年均線的 5 倍（現在約 {_usd(m73 * 5)}）時賣出" if long
                           else f"收盤跌破 2 年均線（現在約 {_usd(m73)}）時買進")
        if key == "mvrv":
            v = float(ind["mvrv"].iloc[-1])
            cost = float(close.shift(LAG).iloc[-1]) / v
            value = f"{_num(v)}（換算所有幣的平均成本約 {_usd(cost)}）"
            return value, (f"MVRV 高過 3.5 時賣出（平均成本不變的話約 {_usd(cost * 3.5)}）" if long
                           else f"MVRV 低於 1.0 時買進（約 {_usd(cost)} 以下）")
        if key == "puell":
            v = float(ind["puell"].iloc[-1])
            base = float(close.shift(LAG).iloc[-1])
            value = _num(v)
            return value, (f"高過 4 時賣出（挖出的幣數和手續費不變的話約 {_usd(base * 4 / v)}）" if long
                           else f"低於 0.5 時買進（挖出的幣數和手續費不變的話約 {_usd(base * 0.5 / v)} 以下）")
        span = pd.Timedelta(days=500)
        est = f"下一次減半預估在 {_d(halving_next)} 前後（估計值）"
        if key == "clock":
            if long:
                end = [pd.Timestamp(h) + span for h in HALVINGS + (str(halving_next),) if pd.Timestamp(h) + span > last][0]
                return "—", f"{_d(end)} 賣出（減半後第 500 天）"
            return "—", f"{_d(halving_next - span)} 前後買進（{est}，往前推 500 天）"
        if long:
            end = [pd.Timestamp(h) + pd.DateOffset(months=18) for h in HALVINGS + (str(halving_next),)
                   if pd.Timestamp(h) + pd.DateOffset(months=18) > last][0]
            return "—", f"{_d(end)} 賣出（減半後 18 個月）"
        return "—", f"{_d(halving_next)} 前後買進（{est}）"

    state_rows = []
    for key, label, _ in rules:
        value, trigger = today(key)
        state_rows.append([label, stance(key), last_event(key), value, trigger])
    key = "trend"
    state_rows.append([name[key], stance(key), last_event(key), "—",
                       "收盤創 10 日新低時賣出" if signals[key].iloc[-1] == 1 else "收盤創 20 日新高時買進"])

    span_text = f"{COMMON[:7]} 到 {_d(last)}"
    tables = [
        {"caption": (f"總表（{span_text}）。每條規則都是當天收盤算出訊號、隔天才換手，每次買或賣扣 {COST * 100:.2f}%，"
                     "空手時現金沒有利息；2018 年第一天的狀態沿用規則先前的狀態。「完成幾次」是這段期間裡賣出的次數（一買一賣算一次）。"
                     "「最後的錢 ÷ 一直抱著」大於 1 倍才是贏過抱著。最右兩欄是門檻公開的年份，以及只看公開之後（從隔年 1 月 1 日起）的結果；"
                     f"說不準年份的寫「{UNSURE}」，不另外算。"),
         "columns": ["規則", "2018 至今完成幾次買→賣", "每年報酬", "最深曾經虧多少", "持有比特幣的時間", "最後的錢 ÷ 一直抱著",
                     "門檻哪一年公開", "只看公開之後"],
         "rows": score_rows},
        {"caption": ("各指標在每一輪的極端值，對照固定的門檻。四輪用熊市最低收盤日切開（"
                     + "、".join(_d(x) for x in lows) + f"），各輪價格最高的收盤日是 {'、'.join(_d(t) for t in tops)}。"
                     "「漲勢中最高」是該輪從頭到尾指標的最高值；「高點之後最低」是那一輪價格高點之後、到下一輪高點之前的最低值"
                     "（最近一輪算到今天，還沒走完）。MVRV 和普爾倍數晚一天使用。"),
         "columns": ["指標", "固定門檻"] + top_names + ["有碰到門檻的輪數"],
         "rows": peak_rows},
        {"caption": (f"每月定期買進的三種做法（每月 1 日有 {MONTHLY:.0f} 美元可用，當天收盤買，每次買扣 {COST * 100:.2f}%）。"
                     "三種做法拿到的錢一樣多，沒買的先以現金存著、不計利息，最後的價值是「幣 ＋ 還沒用掉的現金」。"
                     "恐懼貪婪指數 2018-02 才有，所以三種一起比只能從那時開始；前兩種另外列出 2014 年起的結果。"),
         "columns": ["做法", "期間", "總共投入", "現在的價值", "是投入的幾倍", "其中還沒買的現金", "實際有買的月數", "平均買進價"],
         "rows": dca_rows},
        {"caption": (f"每條規則今天（{_d(last)} 收盤 {_usd(price)}）的狀態，以及照規則下一個訊號要等什麼條件。"
                     "這是在描述規則的狀態，不是買賣建議；價位是用今天的均線換算的，均線每天會動。"),
         "columns": ["規則", "現在的狀態", "上一次訊號", "指標現在的數字", "下一個訊號的條件"],
         "rows": state_rows},
    ]

    # ----- the evidence notes: full history, each cycle, every signal -----
    def listed(key: str) -> str:
        ev = events[key]
        if key == "w200_keep":
            ev = ev[:1]
        if len(ev) > MAX_LISTED:
            ev = [e for e in ev if e[0] >= pd.Timestamp(COMMON)]
            if len(ev) > MAX_LISTED or not ev:
                return f"訊號一共 {len(events[key])} 次，太多不逐一列出。"
            head = f"訊號一共 {len(events[key])} 次，2018 年以來的 {len(ev)} 次："
        else:
            head = f"全部訊號只有 {len(ev)} 次："
        return head + "、".join(f"{_d(d)} {'買' if s > 0 else '賣'} {_usd(p)}" for d, s, p in ev) + "。"

    evidence = []
    for key, label, what in rules:
        w = whole[key]
        per_cycle = []
        for cname, a, b in CYCLES:
            m = run(key, a, b)
            per_cycle.append(f"{cname} {_times(m['vs']) if m else '—'}" + (f"（賣 {m['sells']} 次）" if m else ""))
        evidence.append(
            f"【{label}】{what}。從最早能算的 {_d(w['first'])} 起：每年 {_pct(w['ret'])}（同期抱著 {_pct(w['hold_ret'])}），"
            f"最深曾經虧 {_pct(w['deepest'])}，{w['share']:.0f}% 的時間持有，完成買→賣 {w['sells']} 次，最後的錢是抱著的 {_times(w['vs'])}。"
            f"分輪看「最後的錢 ÷ 同一輪抱著」：{'；'.join(per_cycle)}。{listed(key)}")
    pi_depth = extra["pi"][0]
    if pi_depth:
        evidence.append("Pi 週期頂部指標本身只有賣出訊號，沒有買回的規則；「從賣出價跌掉至少一半之後、收盤站回 200 日均線才買回」是本頁自己加的，"
                        "換一種買回方式結果會不同。各次買回前，收盤最低比賣出價低了："
                        + "、".join(f"{_d(d)} 買回前 {_pct(x * 100)}" for d, x in pi_depth) + "。")

    # ----- today -----
    holding = [k for k, _, _ in rules if signals[k].iloc[-1] == 1]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {_usd(price)}"},
        {"label": "照規則今天的狀態", "value": f"{len(rules)} 條裡 {len(holding)} 條持有比特幣、{len(rules) - len(holding)} 條持有現金"},
        {"label": "梅耶倍數（現價 ÷ 200 日均線；常說的線是 1.0 和 2.4）", "value": _num(ind["mayer"].iloc[-1])},
        {"label": "現價 ÷ 200 週均線（常說的線是 1.1 和 3）", "value": _num(ind["w200"].iloc[-1])},
        {"label": "現價 ÷ 2 年均線（常說的線是 1 和 5）", "value": _num(ind["ma2y"].iloc[-1])},
    ]
    if "mvrv" in ind:
        now.append({"label": "MVRV（現價 ÷ 所有幣的平均成本；常說的線是 1.0 和 3.5）", "value": _num(ind["mvrv"].iloc[-1])})
    if "puell" in ind:
        now.append({"label": "普爾倍數（礦工當天收入 ÷ 365 日平均；常說的線是 0.5 和 4）", "value": _num(ind["puell"].iloc[-1])})
    now.append({"label": "下一次減半（估計值）", "value": f"{_d(halving_next)} 前後（{halving_how}）"})

    # ----- verdict -----
    keys = [k for k, _, _ in rules]
    wins = [k for k in keys if recent[k]["vs"] > 1]
    shallower = [k for k in keys if recent[k]["deepest"] > recent["hold"]["deepest"] + 0.05]
    few = [k for k in keys if recent[k]["sells"] <= 3]
    best = max(keys, key=lambda k: recent[k]["vs"])
    worst = min(keys, key=lambda k: recent[k]["vs"])
    sell_lines = [k for k in reached_sell]
    hit_last = [k for k in sell_lines if reached_sell[k][-1]]
    hit_prev = [k for k in sell_lines if reached_sell[k][-2]]
    hit_early = [k for k in sell_lines if reached_sell[k][0] and reached_sell[k][1]]
    judged = [k for k in keys if public.get(k)]
    public_wins = [k for k in judged if public[k]["vs"] > 1]
    short = {k: n.split("：")[0] for k, n, _ in RULES}
    verdict = (
        f"把網路上最常被轉述的 {len(keys)} 條買賣規則照原本的門檻逐日照做，{COMMON[:4]} 年到 {_d(last)}"
        f"（一直抱著每年 {_pct(recent['hold']['ret'])}、最深曾經虧 {_pct(recent['hold']['deepest'])}）："
        f"{len(keys)} 條裡 {len(wins)} 條最後的錢比一直抱著多"
        f"（最高的是「{name[best]}」{_times(recent[best]['vs'])}，最低的是「{name[worst]}」{_times(recent[worst]['vs'])}），"
        f"{len(shallower)} 條最深的虧損比抱著淺；只看門檻公開之後的 {len(judged)} 條裡，{len(public_wins)} 條贏過同期抱著。"
        f"固定的賣出門檻有 {len(sell_lines)} 條（{'、'.join(short[k] for k in sell_lines)}）："
        f"{tops[0].year} 和 {tops[1].year} 年兩輪都碰到的有 {len(hit_early)} 條，{tops[2].year} 年那一輪碰到 {len(hit_prev)} 條，"
        f"{tops[3].year} 年這一輪碰到 {len(hit_last)} 條"
        + (f"（{'、'.join(short[k] for k in hit_last)}）" if hit_last else "") + "。"
        f"{len(keys)} 條裡有 {len(few)} 條在這將近 {round((last - pd.Timestamp(COMMON)).days / 365.25)} 年裡完成的買→賣不超過 3 次，"
        "樣本少到無從用統計驗證，表裡的輸贏幾乎都是由其中一兩次買賣決定的。"
    )

    notes = [
        "這些規則大多一輪只交易一兩次，從頭到尾只買賣過幾次，沒有辦法用統計方法驗證它們有沒有用；一次買賣的時機差幾週，"
        "整張表的輸贏就可能翻過來。所以下面把每條規則的每一次訊號都列出來——次數這麼少，清單本身就是全部的證據。",
        "這些門檻（2.4、3.5、5 倍、111／350、0.5、4……）都是看過先前幾輪的走勢之後才畫出來的，在它們公開以前的「成績」是事後套上去的，"
        "不是當時真的有人能照做的結果；總表最右邊另外列出只看公開之後的數字。公開年份是憑一般公開資料的印象寫的，"
        f"說不準的標「{UNSURE}」而不硬填；有年份的一律從隔年 1 月 1 日起算。",
        "比特幣到現在只有三、四輪完整的漲跌，這是非常小的樣本；而且每一輪的漲幅都比上一輪小，"
        "用早期幾輪定出來的固定賣出門檻，後來有沒有再被碰到請看第二張表。",
        f"成本只用一個粗略的數字：每次買或賣扣 {COST * 100:.2f}%，實際的手續費、價差和滑價會隨交易所、金額和年代不同；"
        "空手時現金假設沒有利息，也沒有算稅；一直抱著沒有扣成本。",
        "做法細節：訊號用當天收盤已經知道的資料，隔天才換手。「貼近 200 週均線」指收盤不高於均線的 1.1 倍；200 週均線用 1,400 天的日均線每天更新。"
        "牛市支撐帶每週日收盤判斷一次。有買有賣兩條線的規則，在兩條線之間維持上一次訊號的狀態，第一次買進訊號出現前是空手。"
        f"MVRV 和礦工收入是當天結束後才公布的，所以晚 {LAG} 天使用。{FLOOR[:4]} 年以前的資料太薄，不列入。",
    ] + evidence + [
        f"減半時鐘用的減半日：{'、'.join(HALVINGS)}；下一次減半還沒發生，{_d(halving_next)} 是估計值（{halving_how}），實際日期會前後移動。"
        "「減半前後各 500 天」這個數字本身也是看過前幾輪之後才有人提出的。",
        "定期買進的三種做法每個月拿到一樣多的錢，所以比的是「同樣的總投入最後變成多少」；它們和上面「一次投入後持有或空手」的規則不是同一種比較，"
        "不能直接拿倍數互比。「恐懼時買兩倍」受限於手上存下來的現金，沒有存款時只能買一倍。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）。"
        "MVRV 來自 CoinMetrics 社群版資料；普爾倍數是用 blockchain.info 的礦工每日收入（新發行的幣加手續費，換成美元）自己算的，"
        "和其他網站的數字會有些出入。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "blockchain.info 每日比特幣價格（2010 年起）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "CoinMetrics 社群版資料（MVRV）", "url": "https://charts.coinmetrics.io/crypto-data/"},
        {"name": "blockchain.info 礦工每日收入", "url": "https://www.blockchain.com/explorer/charts/miners-revenue"},
        {"name": "alternative.me 加密貨幣恐懼與貪婪指數", "url": "https://alternative.me/crypto/fear-and-greed-index/"},
    ]
    return {
        "id": "popular_rules",
        "title": "網路上最常聽到的買賣規則，照做會怎樣",
        "question": ("網路上最常被轉述的比特幣買賣規則——長期均線、週期頂部指標、鏈上的「價值區」、減半時鐘、定期買進——"
                     "寫成明確的規則逐日照做，有沒有贏過一直抱著？那些有名的門檻是事先就知道的嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
