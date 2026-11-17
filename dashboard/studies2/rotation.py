"""Historical study "rotation": two ideas taken from other people's public projects, tested on our own data.

Part 1 - when the bitcoin trend rule is out, is there something better to hold than idle cash?
  a. the 17-rule blend (and the plain "above the 50-day average" rule) with the idle part in: nothing, US Treasury
     bills, QQQ always, QQQ only above its own 200-day average, GLD always / only above its 200-day average, the
     stronger of QQQ and GLD by 60-day return if that one is above its 200-day average;
  b. the published Quantpedia "Silicon vs Satoshi" rotation (bitcoin / QQQ / cash by w-day closing highs);
  c. plain monthly-rebalanced mixes with a small fixed bitcoin weight;
  d. picking the idle-part option one year ahead on earlier data only.
Part 2 - do the cycle indicators (Mayer multiple, price / 200-week average, price / 2-year average, MVRV, Puell
  multiple) work again when the fixed thresholds are replaced by the reading's rank inside its own trailing 4 years
  (the idea of the btc-compass project)?

Conventions (no look-ahead)
  * Part 1 runs on US trading days (the days QQQ has a bar). Bitcoin on trading day p is the daily candle labelled p,
    which closes at 00:00 UTC, three to four hours after the US close. The decision is taken at that moment: it reads
    the US closes of day p and bitcoin candles up to p, nothing later, and is held until the same moment after the
    next trading day. Bitcoin signals are computed on every calendar day and read on trading days only. The fund leg
    is booked at the US close of day p, i.e. a few hours before the bitcoin candle closes; a strict variant that only
    reads the bitcoin candle of the calendar day before (closed about 20 hours before the US close) is run as a check.
  * Part 2 runs on calendar days: a signal uses data up to the close of day t and decides the position of day t+1;
    MVRV and miner revenue are used one day late (as in popular_rules).
  * Costs: 0.10% of the amount moved in bitcoin or a fund (a switch from one to the other pays twice); bills and
    cash move for free. Adjusted closes (dividends included) for the funds.

Only public, key-less sources with the dashboard's fixed user agent. A source that fails is dropped with a note.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import studies as old_studies
from dashboard.studies2 import popular_rules as pr
from dashboard.studies2 import trend_ensemble as te

COST = 0.001  # 0.10% of the amount moved
TD_YEAR = 252  # Part 1: US trading days
CAL_YEAR = 365  # Part 2: calendar days
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
YAHOO_FROM = 1356998400  # 2013-01-01
FRED_DTB3 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
PAGE = "https://quantpedia.com/silicon-vs-satoshi-tactical-asset-rotation-between-nasdaq-100-and-bitcoin/"
COMPASS = "https://github.com/jackyhe1819-create/btc-compass"
PUBLISHED = "2026-07-02"  # the date shown on the Quantpedia page
SAMPLE = ("2019-01-01", "2025-12-31")  # the sample stated in the article's data section
WINDOWS = (5, 10, 20, 30, 50)
# return ÷ volatility printed in the article's two tables (no costs), read from the page on 2026-11-18
QUOTED = {"qqq": {5: 1.690, 10: 1.520, 20: 1.680, 30: 1.523, 50: 1.148},
          "btc": {5: 1.422, 10: 1.398, 20: 1.370, 30: 1.251, 50: 0.846}}
QUOTED_BENCH = {"hold_qqq": 0.901, "hold_btc": 0.894, "half": 1.188}
MAIN = "2018-01-01"
PERIODS = (("2015", "2015-01-01"), ("2018", MAIN), ("2022", "2022-01-01"), ("2024", "2024-01-01"))
PICK_FROM = 2020
COLS = ("btc", "qqq", "gld", "bill")
RISKY = (0, 1, 2)  # columns that pay the trading cost
MA_LONG, MOM_DAYS = 200, 60
QQQ_NAME = "追蹤那斯達克 100 指數的基金（QQQ）"
ASSET = {"btc": "比特幣", "qqq": "QQQ", "gld": "黃金基金", "bill": "國庫券"}
OPTIONS = (("cash", "空著不放（0%）"), ("bill", "放國庫券"), ("qqq", "放 QQQ（一直放）"),
           ("qqq_f", "放 QQQ，高於自己 200 日均線才放"), ("gld", "放黃金基金（一直放）"),
           ("gld_f", "放黃金基金，高於自己 200 日均線才放"),
           ("strong", "放 QQQ 和黃金基金裡近 60 天較強的，高於 200 日均線才放"))
OPTION_NAME = dict(OPTIONS)
OPTION_SHORT = {"cash": "空著", "bill": "國庫券", "qqq": "QQQ", "qqq_f": "QQQ（要高於 200 日均線）", "gld": "黃金基金",
                "gld_f": "黃金基金（要高於 200 日均線）", "strong": "QQQ 和黃金裡較強的"}
BASES = (("blend", "混合規則"), ("ma50", "50 日均線規則"))

# Part 2
ROLL = {"2y": 730, "4y": 1461, "8y": 2922, "all": None}
ROLL_NAME = {"2y": "兩年", "4y": "四年", "8y": "八年", "all": "全部歷史"}
TOPS = (0.05, 0.10, 0.20)
MIN_ALL = 730
CYCLE_NAMES = (("mayer", "梅耶倍數（現價 ÷ 200 日均線）", "低於 1.0 買、高於 2.4 賣"),
               ("w200", "現價 ÷ 200 週均線", "貼近 1.1 倍買、高於 3 倍賣"),
               ("ma2y", "現價 ÷ 2 年均線", "低於 1 買、高於 5 倍賣"),
               ("mvrv", "MVRV（現價 ÷ 全體平均成本）", "低於 1.0 買、高於 3.5 賣"),
               ("puell", "普爾倍數（礦工收入 ÷ 一年平均）", "低於 0.5 買、高於 4 賣"))
ROUND_2021 = ("2020-10-01", "2021-12-31")
ROUND_2025 = ("2024-01-01", None)


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
    text = f"{float(v):,.{digits}f}"
    return text[1:] if text.startswith("-") and float(text.replace(",", "")) == 0 else text


def _times(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    return f"{v:,.0f} 倍" if v >= 100 else f"{v:.1f} 倍" if v >= 10 else f"{v:.2f} 倍"


def _usd(v) -> str:
    return f"{float(v):,.0f}" if _ok(v) else "—"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed bitcoin daily closes since 2010, one row per calendar day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


def _share(symbol: str) -> pd.Series:
    """Finished adjusted daily closes from Yahoo Finance since 2013 (the dashboard's ten-year helper as fallback)."""
    try:
        reply = requests.get(YAHOO.format(symbol), headers=common.HEADERS, timeout=30,
                             params={"period1": YAHOO_FROM, "period2": int(time.time()), "interval": "1d"})
        reply.raise_for_status()
        time.sleep(common.PAUSE)
        result = reply.json()["chart"]["result"][0]
        series = pd.Series(result["indicators"]["adjclose"][0]["adjclose"],
                           index=pd.to_datetime(result["timestamp"], unit="s").normalize()).dropna()
        if len(series) < 2000:
            raise RuntimeError(f"只回傳 {len(series)} 筆")
    except Exception:  # noqa: BLE001 - the shorter helper still covers the main period
        series = old_studies._yahoo(symbol)
    series = series.astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    # a bar dated before today's UTC date is always finished (the US session ends around 21:00 UTC the same date)
    series = series[series.index < pd.Timestamp(common.today_utc())]
    series = series[series > 0]
    if len(series) < 300:
        raise RuntimeError(f"Yahoo {symbol} 日線只有 {len(series)} 筆")
    return series


def _stored(name: str, fetch, least: int) -> tuple[pd.Series, str]:
    """fetch() -> {YYYY-MM-DD: number}. A good download replaces our stored copy; a failed one falls back to it."""
    path = common.STORE / f"rotation_{name}.json"
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
    series = pd.Series({pd.Timestamp(day): float(value) for day, value in days.items()}).sort_index()
    return series[np.isfinite(series.values)], note


def _fetch_dtb3() -> dict:
    reply = requests.get(FRED_DTB3, headers=common.HEADERS, timeout=8)
    reply.raise_for_status()
    frame = pd.read_csv(io.StringIO(reply.text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return {str(day)[:10]: float(v) for day, v in zip(frame.iloc[:, 0], values) if math.isfinite(v)}


def _load() -> tuple[dict, list[str]]:
    problems: list[str] = []
    data: dict = {"daily": _daily_close(), "shares": {}, "rate": None, "bill_fund": None, "bill_source": None,
                  "mvrv": None, "revenue": None}
    data["shares"]["QQQ"] = _share("QQQ")  # without QQQ there is no trading calendar: let it raise
    try:
        data["shares"]["GLD"] = _share("GLD")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"黃金基金（GLD）這次取不到資料（{type(exc).__name__}），用到黃金的做法這次沒有算。")
    try:
        rate, note = _stored("dtb3", _fetch_dtb3, 3000)
        data["rate"], data["bill_source"] = rate, "fred"
        if note:
            problems.append("FRED 的三個月期國庫券利率" + note + "。")
    except Exception as exc:  # noqa: BLE001
        try:
            data["bill_fund"], data["bill_source"] = _share("BIL"), "bil"
            problems.append(f"FRED 的三個月期國庫券利率（DTB3）這次連不上（{type(exc).__name__}），"
                            "「國庫券」改用一檔只買 1 到 3 個月美國國庫券的基金（BIL，Yahoo 還原配息後的收盤價）代替，"
                            "它已經扣掉基金自己的費用（每年約 0.14%）。")
        except Exception as second:  # noqa: BLE001
            problems.append(f"FRED 的國庫券利率和代替用的國庫券基金（BIL）這次都取不到（{type(exc).__name__}／"
                            f"{type(second).__name__}），「國庫券」這次當成 0% 計算。")
    try:
        mvrv, note = _stored("mvrv", pr._fetch_mvrv, 3000)
        data["mvrv"] = mvrv[mvrv.index >= pd.Timestamp(pr.FLOOR)]
        if note:
            problems.append("CoinMetrics 的 MVRV " + note + "。")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"CoinMetrics 的 MVRV 這次取不到（{type(exc).__name__}），MVRV 這次沒有算。")
    try:
        data["revenue"], note = _stored("miner_revenue", pr._fetch_revenue, 3000)
        if note:
            problems.append("blockchain.info 的礦工收入" + note + "。")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"blockchain.info 的礦工收入這次取不到（{type(exc).__name__}），普爾倍數這次沒有算。")
    return data, problems


# ---------- measuring ----------

def _measure(net: pd.Series, year: int) -> dict | None:
    """Yearly return, volatility, ratio, deepest loss and calendar-year returns of a run of daily net returns."""
    if len(net) < 20 or net.isna().any():
        return None
    values = net.values
    growth = float(np.prod(1 + values))
    n = len(values)
    ret = (growth ** (year / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(values, ddof=1) * math.sqrt(year) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    years = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    worst = min(years, key=lambda y: years[y])
    return {"days": n, "ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "growth": growth,
            "total": (growth - 1) * 100, "deepest": float((equity / np.maximum.accumulate(equity) - 1).min() * 100),
            "years": years, "worst": worst, "first": net.index[0], "last": net.index[-1]}


def _beats(mine: dict | None, other: dict | None) -> tuple[int, int] | None:
    if not mine or not other:
        return None
    years = [y for y in mine["years"] if y in other["years"]]
    return len(years), sum(1 for y in years if mine["years"][y] > other["years"][y])


def _beat_text(pair) -> str:
    return f"{pair[0]} 年裡 {pair[1]} 年" if pair else "—"


# ---------- Part 1 ----------

def _sim(weights: pd.DataFrame, returns: pd.DataFrame, cost: float) -> tuple[pd.Series, pd.Series]:
    """weights: target decided at each row's close and held until the next row. Returns (net, amount traded) per row,
    NaN while the target does not exist yet. The trade is the gap between the new target and the drifted holding."""
    held = weights.shift(1)
    ready = held.notna().all(axis=1).values
    w = np.nan_to_num(held.values, nan=0.0)
    r = np.nan_to_num(returns.values, nan=0.0)
    gross = (w * r).sum(axis=1)
    base = 1 + gross
    drift = w * (1 + r) / np.where(np.abs(base) > 1e-9, base, 1.0)[:, None]
    before = np.vstack([np.zeros((1, w.shape[1])), drift[:-1]])
    trade = np.abs(w - before)[:, list(RISKY)].sum(axis=1)
    net = pd.Series(np.where(ready, gross - cost * trade, np.nan), index=weights.index)
    return net, pd.Series(np.where(ready, trade, np.nan), index=weights.index)


def _mix(target: dict, returns: pd.DataFrame, cost: float) -> pd.Series:
    """Fixed weights (the rest is cash at 0%), put back to target at the last close of every month."""
    want = np.array([target.get(c, 0.0) for c in COLS])
    r = np.nan_to_num(returns.values, nan=0.0)
    index = returns.index
    net = np.zeros(len(index))
    w = np.zeros(len(COLS))
    for i in range(len(index)):
        trade = 0.0
        if i == 0 or index[i].month != index[i - 1].month:
            trade = float(np.abs(want - w)[list(RISKY)].sum())
            w = want.copy()
        gross = float((w * r[i]).sum())
        net[i] = gross - cost * trade
        w = w * (1 + r[i]) / (1 + gross)
    return pd.Series(net, index=index)


def _breakout(price: pd.Series, w: int) -> pd.Series:
    """1 when the close is above the highest of the w closes before it (the article's bound excludes the day itself)."""
    bound = price.shift(1).rolling(w, min_periods=w).max()
    return (price > bound).astype(float).where(bound.notna())


def _compute1(data: dict, strict: bool = False) -> dict:
    """Everything of Part 1 that depends on prices. strict=True reads bitcoin's signals from the candle of the
    calendar day before each trading day (nothing after the US close), as a check on the main convention."""
    daily = data["daily"]
    qqq_all = data["shares"]["QQQ"]
    btc = daily.reindex(qqq_all.index).dropna()  # the candle labelled with each US trading day
    idx = btc.index
    price = pd.DataFrame({"btc": btc, "qqq": qqq_all.reindex(idx)})
    has_gld = "GLD" in data["shares"]
    price["gld"] = data["shares"]["GLD"].reindex(idx).ffill() if has_gld else np.nan
    returns = price.pct_change()
    gap = pd.Series(idx, index=idx).diff().dt.days
    if data.get("rate") is not None:  # the rate quoted on day t-1 is earned over the calendar days up to day t
        rate = data["rate"].reindex(idx.union(data["rate"].index)).ffill().reindex(idx)
        returns["bill"] = (rate.shift(1) / 100 * gap / 360).fillna(0.0)
    elif data.get("bill_fund") is not None:
        returns["bill"] = data["bill_fund"].reindex(idx).ffill().pct_change().fillna(0.0)
    else:
        returns["bill"] = 0.0
    returns = returns[list(COLS)]

    def known(series: pd.Series) -> pd.Series:
        """A calendar-day bitcoin series as it is read on each trading day."""
        out = series.reindex(idx - pd.Timedelta(days=1)) if strict else series.reindex(idx)
        out.index = idx
        return out

    target = te._targets(daily)[0]
    share = {"blend": known(target["blend"]), "ma50": known(target["ma50"])}
    above, past = {}, {}
    for key in ("qqq", "gld"):
        mean = price[key].rolling(MA_LONG, min_periods=MA_LONG).mean()
        above[key] = (price[key] > mean).astype(float).where(mean.notna())
        past[key] = price[key] / price[key].shift(MOM_DAYS) - 1
    options = [k for k, _ in OPTIONS if has_gld or k in ("cash", "bill", "qqq", "qqq_f")]

    def weights(base: str, option: str) -> pd.DataFrame:
        s = share[base]
        idle = 1 - s
        w = pd.DataFrame(0.0, index=idx, columns=list(COLS))
        w["btc"] = s
        valid = s.notna()
        if option == "bill":
            w["bill"] = idle
        elif option in ("qqq", "gld"):
            w[option] = idle
        elif option in ("qqq_f", "gld_f"):
            key = option[:3]
            w[key], w["bill"] = idle * above[key], idle * (1 - above[key])
            valid &= above[key].notna()
        elif option == "strong":
            pick_q = past["qqq"] >= past["gld"]
            on_q = (pick_q & (above["qqq"] == 1)).astype(float)
            on_g = (~pick_q & (above["gld"] == 1)).astype(float)
            w["qqq"], w["gld"], w["bill"] = idle * on_q, idle * on_g, idle * (1 - on_q - on_g)
            valid &= above["qqq"].notna() & above["gld"].notna() & past["qqq"].notna() & past["gld"].notna()
        return w.where(valid)

    runs, held = {}, {}
    for base, _ in BASES:
        for option in options:
            w = weights(base, option)
            held[(base, option)] = w
            runs[(base, option)] = _sim(w, returns, COST)[0]
    bench = {"hold_qqq": returns["qqq"].copy(), "hold_btc": returns["btc"].copy()}

    # ----- the published rotation -----
    signal_price = known(daily)  # bitcoin as it is read on each trading day
    rotation, rotation_free, rotation_w, turnover = {}, {}, {}, {}
    for w_days in WINDOWS:
        high_b, high_q = _breakout(signal_price, w_days), _breakout(price["qqq"], w_days)
        ready = high_b.notna() & high_q.notna()
        for first in ("btc", "qqq"):
            w = pd.DataFrame(0.0, index=idx, columns=list(COLS))
            if first == "btc":
                w["btc"], w["qqq"] = high_b, (1 - high_b) * high_q
            else:
                w["qqq"], w["btc"] = high_q, (1 - high_q) * high_b
            w = w.where(ready)
            rotation_w[(first, w_days)] = w
            rotation[(first, w_days)], trade = _sim(w, returns, COST)
            rotation_free[(first, w_days)] = _sim(w, returns, 0.0)[0]
            rows = trade.index >= pd.Timestamp(SAMPLE[0])
            turnover[(first, w_days)] = float(trade[rows].sum() / (rows.sum() / TD_YEAR))

    # ----- fixed mixes -----
    mixes = {"half": {"qqq": 0.5, "btc": 0.5}}
    for name, core in (("q50", {"qqq": 0.5}), ("q60", {"qqq": 0.6, "bill": 0.4})):
        for b in (0, 5, 10, 25):
            mix = {k: v * (1 - b / 100) for k, v in core.items()}
            if b:
                mix["btc"] = b / 100
            mixes[f"{name}_{b}"] = mix
    start = idx[idx >= pd.Timestamp("2014-12-31")][0] if (idx >= pd.Timestamp("2014-12-31")).any() else idx[0]
    mix_runs = {k: _mix(m, returns[returns.index >= start], COST) for k, m in mixes.items()}

    # ----- measures by period -----
    last = idx[-1]

    def by_period(net: pd.Series) -> dict:
        first = net.first_valid_index()
        out = {}
        for name, a in PERIODS:
            ready = first is not None and first <= pd.Timestamp(a) + pd.Timedelta(days=10)
            out[name] = _measure(net[net.index >= max(pd.Timestamp(a), first)], TD_YEAR) if ready else None
        return out

    stats = {k: by_period(v) for k, v in runs.items()}
    stats.update({k: by_period(v.iloc[1:]) for k, v in bench.items()})
    stats.update({("mix", k): by_period(v.iloc[1:]) for k, v in mix_runs.items()})

    def window(net: pd.Series, a: str, b: str | None = None, after: bool = False) -> dict | None:
        lo = net.index > pd.Timestamp(a) if after else net.index >= pd.Timestamp(a)
        hi = net.index <= pd.Timestamp(b) if b else np.ones(len(net), dtype=bool)
        return _measure(net[lo & hi], TD_YEAR)

    rot = {}
    series = {("rot",) + k: v for k, v in rotation.items()}
    series.update({k: v.iloc[1:] for k, v in bench.items()})
    series["half"] = mix_runs["half"].iloc[1:]
    free = {("rot",) + k: v for k, v in rotation_free.items()}
    for key, net in series.items():
        rot[key] = {"sample": window(net, *SAMPLE), "free": window(free.get(key, net), *SAMPLE),
                    "after_sample": window(net, "2026-01-01"), "after_pub": window(net, PUBLISHED, after=True),
                    "main": window(net, MAIN), "turnover": turnover.get(key[1:]) if key[0] == "rot" else None}

    # ----- pick the idle-part option one year ahead -----
    picks = {}
    for base, _ in BASES:
        nets = {o: runs[(base, o)] for o in options}
        first_ok = max(n.first_valid_index() for n in nets.values())
        chain, chosen = [], []
        for year in range(PICK_FROM, last.year + 1):
            past_rows = {o: n[(n.index >= first_ok) & (n.index < pd.Timestamp(f"{year}-01-01"))] for o, n in nets.items()}
            scores = {o: (_measure(p, TD_YEAR) or {}).get("ratio") for o, p in past_rows.items()}
            if any(s is None for s in scores.values()):
                continue
            best = max(options, key=lambda o: scores[o])
            chosen.append((year, best))
            chain.append(nets[best][nets[best].index.year == year])
        if chain:
            net = pd.concat(chain)
            same = {o: _measure(n[n.index >= net.index[0]], TD_YEAR) for o, n in nets.items()}
            picks[base] = {"chosen": chosen, "stat": _measure(net, TD_YEAR), "same": same, "from": first_ok}
    return {"idx": idx, "last": last, "price": price, "returns": returns, "options": options, "held": held,
            "runs": runs, "stats": stats, "rot": rot, "rotation_w": rotation_w, "picks": picks, "has_gld": has_gld,
            "share": share, "above": above, "past": past}


# ---------- Part 2 ----------

def _rank(series: pd.Series, window: int | None) -> pd.Series:
    """Share (0..1) of the other readings in the trailing window that are lower than today's reading."""
    if window is None:
        roll = series.expanding(min_periods=MIN_ALL)
    else:
        roll = series.rolling(window, min_periods=window // 2)  # starts once half the window exists
    count = roll.count()
    return ((roll.rank(method="min") - 1) / (count - 1).where(count > 1)).where(series.notna())


def _run2(signal: pd.Series, change: pd.Series, close: pd.Series, start: str) -> dict | None:
    """One calendar-day rule from `start` (or from its first possible day if later) against buy-and-hold."""
    position = signal.shift(1)
    first = position.first_valid_index()
    if first is None:
        return None
    a = max(pd.Timestamp(start), first)
    net, _ = te._run(position.values, change.values, COST)
    net = pd.Series(net, index=signal.index)
    rows = (net.index >= a) & position.notna().values
    m = _measure(net[rows], CAL_YEAR)
    if not m:
        return None
    hold = _measure(change[rows].fillna(0.0), CAL_YEAR)
    s = signal[signal.index >= a - pd.Timedelta(days=1)].dropna()
    exits = s.index[((s.shift(1) > 0.5) & (s < 0.5)).values]
    entries = s.index[((s.shift(1) < 0.5) & (s > 0.5)).values]
    m.update({"vs": m["growth"] / hold["growth"], "hold": hold, "late": a > pd.Timestamp(start), "start": a,
              "exits": [(d, float(close[d])) for d in exits], "entries": [(d, float(close[d])) for d in entries],
              "share": float(position[rows].mean() * 100)})
    return m


def _compute2(daily: pd.Series, mvrv, revenue) -> dict:
    built = pr._build(daily, mvrv, revenue, pd.Timestamp(pr.NEXT_HALVING_FALLBACK))
    floor = pd.Timestamp(pr.FLOOR)
    idx = daily.index[daily.index >= floor]
    close, change = daily.reindex(idx), daily.pct_change().reindex(idx)
    keys = [k for k, _, _ in CYCLE_NAMES if k in built["ind"]]
    ind = {k: built["ind"][k].reindex(idx) for k in keys}
    blend = te._targets(daily)[0]["blend"].reindex(idx)
    rank = {(k, name): _rank(ind[k], window) for k in keys for name, window in ROLL.items()}

    def top_rule(k: str, roll: str, top: float) -> pd.Series:
        p = rank[(k, roll)]
        return pr._latch(p < 0.5, p > 1 - top, p, start=1.0)  # holding until the first top reading

    out = {"keys": keys, "ind": ind, "rank": rank, "close": close, "last": idx[-1], "rules": {}, "signals": {}}
    hold_sig = pd.Series(1.0, index=idx)
    for since in ("2014-01-01", MAIN):
        out["rules"][("hold", since)] = _run2(hold_sig, change, close, since)
        out["rules"][("blend", since)] = _run2(blend, change, close, since)
        for k in keys:
            for roll in ROLL:
                for top in TOPS:
                    out["rules"][(k, roll, top, since)] = _run2(top_rule(k, roll, top), change, close, since)
            low = (rank[(k, "4y")] < 0.5).astype(float).where(rank[(k, "4y")].notna())
            out["rules"][(k, "low", since)] = _run2(low, change, close, since)
            out["rules"][(k, "fixed", since)] = _run2(built["signals"][k].reindex(idx), change, close, since)
            calm = (rank[(k, "4y")] <= 0.9) | rank[(k, "4y")].isna()
            out["rules"][(k, "overlay", since)] = _run2(blend * calm.astype(float), change, close, since)
    for k in keys:
        out["signals"][k] = top_rule(k, "4y", 0.10)
        out["signals"][(k, "calm")] = (rank[(k, "4y")] <= 0.9) | rank[(k, "4y")].isna()
    out["blend"] = blend
    return out


# ---------- the study ----------

def _holding(row: pd.Series) -> str:
    if row.isna().any():
        return "—"
    parts = [f"{ASSET[c]} {row[c] * 100:.0f}%" for c in COLS if row[c] > 0.005]
    rest = 1 - float(row.sum())
    if rest > 0.005:
        parts.append(f"空著 {rest * 100:.0f}%")
    return "、".join(parts) if parts else "全部空著"


def study() -> dict:
    data, problems = _load()
    r = _compute1(data)
    strict = _compute1(data, strict=True)
    c = _compute2(data["daily"], data["mvrv"], data["revenue"])
    stats, last_td, daily = r["stats"], r["last"], data["daily"]
    options = r["options"]
    bill_name = {"fred": "三個月期美國國庫券利率（FRED DTB3）", "bil": "國庫券基金 BIL", None: "0%（這次取不到利率）"}[data["bill_source"]]
    span = f"{MAIN[:4]}-01 到 {_d(last_td)}"

    def other_periods(key) -> str:
        return "｜".join(_num((stats[key][p] or {}).get("ratio")) for p in ("2015", "2022", "2024"))

    def row(name: str, key, period: str = "2018") -> list:
        m = stats[key][period]
        if not m:
            return [name] + ["—"] * 7
        wins = "—｜" if key == "hold_qqq" else _beat_text(_beats(m, stats["hold_qqq"][period])) + "｜"
        wins += "—" if key == "hold_btc" else _beat_text(_beats(m, stats["hold_btc"][period]))
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['worst']} 年 {_pct(m['years'][m['worst']])}", wins, other_periods(key)]

    measure_cols = ["每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最差的一年", "贏過 QQQ 的年數｜贏過比特幣的年數",
                    "報酬÷波動：2015 起｜2022 起｜2024 起"]

    # ----- table 1: where the idle part sits -----
    idle_rows = [row("對照：一直抱著 QQQ", "hold_qqq"), row("對照：一直抱著比特幣", "hold_btc")]
    for base, base_name in BASES:
        for option in options:
            idle_rows.append(row(f"{base_name}｜空手的錢{OPTION_NAME[option]}", (base, option)))
    pick_text = ""
    pk = r["picks"].get("blend")
    if pk and pk["stat"]:
        chosen = "、".join(f"{y} 年「{OPTION_SHORT[o]}」" for y, o in pk["chosen"])
        best = max(options, key=lambda o: pk["same"][o]["ratio"])
        pick_text = (f"事先挑的檢查（混合規則）：從 {PICK_FROM} 年起，每年年初只用到前一年底為止的資料（最早從 {_d(pk['from'])}），"
                     f"挑報酬÷波動最高的放法用一整年、再接起來：{chosen}；接起來是每年 {_pct(pk['stat']['ret'])}、報酬÷波動 "
                     f"{_num(pk['stat']['ratio'])}、最深虧 {_pct(pk['stat']['deepest'])}。同一段期間一直空著是 "
                     f"{_num(pk['same']['cash']['ratio'])}、一直放國庫券 {_num(pk['same']['bill']['ratio'])}，"
                     f"事後看最好的「{OPTION_SHORT[best]}」是 {_num(pk['same'][best]['ratio'])}。")
    tables = [{
        "caption": (f"順勢規則沒持有比特幣的那部分錢放哪裡（{span}，只在美股交易日決定）。QQQ ＝{QQQ_NAME}；黃金基金 ＝ GLD；"
                    f"國庫券 ＝ 美國短期公債，接近美元定存，這裡用{bill_name}。混合規則是「17 條順勢規則裡幾成說持有，就放幾成在比特幣」，"
                    "其餘的才是空手的錢；50 日均線規則是全進全出。「高於自己 200 日均線才放」的做法，沒放的時候改放國庫券。買賣比特幣和基金每次扣成交金額的 0.10%，國庫券和現金進出不扣。"
                    "兩列對照沒有扣成本。最後一欄是換三個起算年的報酬÷波動。" + pick_text),
        "columns": ["做法"] + measure_cols, "rows": idle_rows[:16]}]

    # ----- table 2: the published rotation -----
    rot = r["rot"]

    def rot_row(name: str, key, quoted) -> list:
        x = rot[key]
        s, f, a, p, m = x["sample"], x["free"], x["after_sample"], x["after_pub"], x["main"]
        return [name, f"{_num((f or {}).get('ratio'))}（{_num(quoted) if quoted else '—'}）",
                _pct((s or {}).get("ret")), _num((s or {}).get("ratio")), _pct((s or {}).get("deepest")),
                _pct((a or {}).get("total")), _pct((p or {}).get("total")), _num((m or {}).get("ratio"))]

    rot_rows = []
    for first, label in (("btc", "先看比特幣"), ("qqq", "先看 QQQ")):
        for w_days in WINDOWS:
            rot_rows.append(rot_row(f"{label}，{w_days} 日新高", ("rot", first, w_days), QUOTED[first][w_days]))
    rot_rows += [rot_row("對照：一直抱著 QQQ", "hold_qqq", QUOTED_BENCH["hold_qqq"]),
                 rot_row("對照：一直抱著比特幣", "hold_btc", QUOTED_BENCH["hold_btc"]),
                 rot_row("對照：QQQ 和比特幣各半（每月調回）", "half", QUOTED_BENCH["half"])]
    rot_keys = [("rot", f, w) for f in ("btc", "qqq") for w in WINDOWS]
    turn = [rot[k]["turnover"] for k in rot_keys if _ok(rot[k]["turnover"])]
    tables.append({
        "caption": (f"Quantpedia 在 {PUBLISHED} 發表的「比特幣／那斯達克輪動」照原文重做：每個美股交易日收盤，排在前面的資產如果收盤高過"
                    "它前 w 個交易日的最高收盤就全部持有它；否則看另一個資產是不是這樣；兩個都不是就全部現金（0%）。原文沒有出場規則，"
                    "所以只要隔天不再創新高就離場。原文的樣本是 2019-01 到 2025-12、不扣成本、沒有留一段資料事後驗證。"
                    "第二欄是我們不扣成本重做的報酬÷波動，括號裡是原文表上的數字（原文的比特幣取美股收盤那一刻的價格，我們用晚三到四小時的日線收盤，"
                    f"所以不會完全一樣）。其餘欄位都扣 0.10% 成本；樣本內平均一年要把全部資金換手 {min(turn):.0f} 到 {max(turn):.0f} 次。"
                    f"「樣本之後」是 2026-01-01 起、「發表之後」是 {PUBLISHED} 之後到 {_d(last_td)} 的累計報酬（不是年化，期間很短）。"),
        "columns": ["做法", "樣本內不扣成本的報酬÷波動（原文數字）", "樣本內扣成本：每年報酬", "樣本內扣成本：報酬÷波動", "樣本內最深曾經虧多少",
                    "樣本之後累計報酬", "發表之後累計報酬", "2018 至今扣成本的報酬÷波動"],
        "rows": rot_rows})

    # ----- table 3: fixed mixes -----
    mix_rows = [row("一直抱著 QQQ", "hold_qqq"), row("一直抱著比特幣", "hold_btc")]
    for name, label in (("q50", "一半 QQQ、一半現金（0%）"), ("q60", "六成 QQQ、四成國庫券")):
        for b in (0, 5, 10, 25):
            mix_rows.append(row(label if not b else f"{label}，整體撥 {b}% 換成比特幣", ("mix", f"{name}_{b}")))
    mix_rows.append(row("對照：混合規則，空手的部分放國庫券", ("blend", "bill")))
    mix_rows.append(row("對照：混合規則，空手放 QQQ（要高於 200 日均線）", ("blend", "qqq_f")))
    tables.append({
        "caption": (f"不看任何訊號的固定比例組合（{span}），每個月最後一個交易日收盤調回原本的比例，調整的金額扣 0.10%。"
                    "「撥 5% 換成比特幣」是指 5% 放比特幣、剩下 95% 照原本的比例分。用來看「固定放一小部分比特幣」和上面那些看訊號進出的做法差多少。"),
        "columns": ["組合"] + measure_cols, "rows": mix_rows})

    # ----- table 4: rolling-percentile cycle rules -----
    keys, rules = c["keys"], c["rules"]
    names = {k: n for k, n, _ in CYCLE_NAMES}
    fixed_text = {k: t for k, _, t in CYCLE_NAMES}
    last_day = c["last"]

    def in_round(events: list, span_: tuple) -> list:
        lo, hi = pd.Timestamp(span_[0]), pd.Timestamp(span_[1]) if span_[1] else last_day
        return [e for e in events if lo <= e[0] <= hi]

    def fired(m: dict | None, span_: tuple) -> str:
        if not m:
            return "—"
        ev = in_round(m["exits"], span_)
        if not ev:
            return "沒有賣出訊號"
        text = f"{len(ev)} 次：{_d(ev[0][0])} 賣在 {_usd(ev[0][1])}"
        if len(ev) > 1:
            text += f"……{_d(ev[-1][0])} 賣在 {_usd(ev[-1][1])}"
        return text

    def vs_text(m: dict | None) -> str:
        if not m:
            return "—"
        return _times(m["vs"]) + (f"（{_d(m['start'])[:7]} 起）" if m["late"] else "")

    hold18 = rules[("hold", MAIN)]
    blend18, blend14 = rules[("blend", MAIN)], rules[("blend", "2014-01-01")]

    def cycle_row(name: str, k, kind, show_fired: bool, fixed: bool) -> list:
        full = rules[(k,) + kind + ("2014-01-01",)] if k != "blend" else blend14
        m = rules[(k,) + kind + (MAIN,)] if k != "blend" else blend18
        if not m:
            return [name] + ["—"] * 7
        every = rules[(k,) + kind + ("2014-01-01",)] if k != "blend" else blend14
        f18 = rules.get((k, "fixed", MAIN)) if fixed else None
        if show_fired:
            a, b = fired(every, ROUND_2021), fired(every, ROUND_2025)
            trips = f"{len(m['exits'])} 次"
        elif kind == ("overlay",):
            calm = c["signals"][(k, "calm")]
            a = f"擋掉 {int((~calm[ROUND_2021[0]:ROUND_2021[1]]).sum())} 天"
            b = f"擋掉 {int((~calm[ROUND_2025[0]:]).sum())} 天"
            trips = "—"
        else:
            a = b = "—"
            trips = f"{len(m['exits'])} 次" if kind == ("low",) else "—"
        return [name, a, b, vs_text(full), vs_text(m), _pct(m["deepest"]), trips,
                f"{_times(f18['vs'])}｜{len(f18['exits'])} 次（{fixed_text[k]}）" if f18 else "—"]

    cycle_rows = [cycle_row(f"{names[k]}：排進前 10% 賣、回到中間值以下買回", k, ("4y", 0.10), True, True) for k in keys]
    cycle_rows += [cycle_row(f"{names[k].split('（')[0]}：只在排名後一半時持有", k, ("low",), False, False) for k in keys]
    cycle_rows.append(cycle_row("對照：17 條順勢規則混合", "blend", (), False, False))
    cycle_rows += [cycle_row(f"混合 ＋「{names[k].split('（')[0]}不在前 10%」才持有", k, ("overlay",), False, False) for k in keys]
    tables.append({
        "caption": ("把週期指標的固定門檻換成「和自己過去四年比」的排名（btc-compass 專案的做法）：每天只拿當天的讀數和它之前四年（1,461 天，"
                    "至少要有一半的天數才開始算）的讀數比，排進最高的 10% 就全部換成現金，等它掉回中間值以下再買回；第一次賣出前一直持有。"
                    f"「2021 那一輪」看 {ROUND_2021[0]} 到 {ROUND_2021[1]}、「2025 這一輪」看 {ROUND_2025[0]} 到 {_d(last_day)} 之間的賣出訊號"
                    "（只列第一次和最後一次的日期和當天收盤價，美元）。「最後的錢 ÷ 抱著」大於 1 倍才是贏過一直抱著；"
                    f"2018 年起一直抱著最深曾經虧 {_pct(hold18['deepest'])}。最後一欄是 popular_rules 那一頁同一個指標用固定門檻的結果（2018 起）。"
                    "每天用日曆天算，當天收盤出訊號、隔天才換手，每次扣 0.10%，現金 0%。"),
        "columns": ["做法", "2021 那一輪的賣出訊號", "2025 這一輪的賣出訊號", "2014 起最後的錢 ÷ 抱著", "2018 起最後的錢 ÷ 抱著",
                    "2018 起最深曾經虧多少", "2018 起賣出幾次", "固定門檻版：÷ 抱著｜賣出次數"],
        "rows": cycle_rows[:16]})

    # ----- today -----
    now = [{"label": f"第一部分用的最後一個美股交易日（{_d(last_td)}）",
            "value": f"比特幣 {_usd(r['price']['btc'].iloc[-1])} 美元；17 條順勢規則裡 {float(r['share']['blend'].iloc[-1]) * 100:.0f}% 說持有；"
                     f"QQQ {'高於' if r['above']['qqq'].iloc[-1] == 1 else '低於'}自己的 200 日均線"
                     + (f"；黃金基金{'高於' if r['above']['gld'].iloc[-1] == 1 else '低於'}自己的 200 日均線" if r["has_gld"] else "")}]
    for option in options:
        now.append({"label": f"混合規則｜空手的錢{OPTION_NAME[option]}：照規則現在的狀態",
                    "value": _holding(r["held"][("blend", option)].iloc[-1])})
    now.append({"label": "50 日均線規則：照規則現在的狀態",
                "value": "持有比特幣" if r["share"]["ma50"].iloc[-1] == 1 else "沒有持有比特幣（空手的部分依各種放法）"})
    rot_now = "；".join(f"{w_days} 日：{_holding(r['rotation_w'][('btc', w_days)].iloc[-1]).replace(' 100%', '')}"
                       for w_days in WINDOWS)
    now.append({"label": "公開的輪動策略（先看比特幣）：照規則現在的狀態", "value": rot_now})
    for k in keys:
        p = c["rank"][(k, "4y")].iloc[-1]
        if _ok(p):
            now.append({"label": f"{names[k]}現在 {_num(c['ind'][k].iloc[-1])}",
                        "value": f"比過去四年 100 天裡的 {float(p) * 100:.0f} 天高"})

    # ----- verdict -----
    def main(key) -> dict:
        return stats[key]["2018"] or {}

    cash, bill, qf = main(("blend", "cash")), main(("blend", "bill")), main(("blend", "qqq_f"))
    after = [rot[k]["after_pub"] for k in rot_keys if rot[k]["after_pub"]]
    in_sample = [rot[k]["sample"]["ratio"] for k in rot_keys if rot[k]["sample"]]
    made = sum(1 for a in after if a["total"] > 0)
    totals = sorted(a["total"] for a in after)
    q_after, b_after = rot["hold_qqq"]["after_pub"], rot["hold_btc"]["after_pub"]
    top18 = [rules[(k, "4y", 0.10, MAIN)] for k in keys]
    top_all = [rules[(k, "4y", 0.10, "2014-01-01")] for k in keys]
    fired21 = sum(1 for m in top_all if m and in_round(m["exits"], ROUND_2021))
    fired25 = sum(1 for m in top_all if m and in_round(m["exits"], ROUND_2025))
    beat_hold = sum(1 for m in top18 if m and m["vs"] > 1)
    overlay = [rules[(k, "overlay", MAIN)] for k in keys]
    helped = sum(1 for m in overlay if m and m["ratio"] > blend18["ratio"] + 0.02)
    gold = main(("blend", "gld")) if r["has_gld"] else {}
    days_after = (last_td - pd.Timestamp(PUBLISHED)).days
    deep = [x["deepest"] for x in (cash, bill, qf, gold) if _ok(x.get("deepest"))] or [0.0]
    verdict = (
        f"{MAIN[:4]} 年至今，順勢混合規則空手的錢空著每年 {_pct(cash.get('ret'))}、報酬÷波動 {_num(cash.get('ratio'))}；"
        f"放國庫券 {_pct(bill.get('ret'))}、{_num(bill.get('ratio'))}；放那斯達克 100 基金 QQQ（跌破 200 日均線退出）"
        f"{_pct(qf.get('ret'))}、{_num(qf.get('ratio'))}"
        + (f"；放黃金基金 {_pct(gold.get('ret'))}、{_num(gold.get('ratio'))}" if gold else "")
        + f"，最深虧損 {_pct(max(deep))} 到 {_pct(min(deep))}。"
        f"公開的比特幣／那斯達克輪動（{PUBLISHED} 發表）樣本內扣成本報酬÷波動 {_num(min(in_sample))}–{_num(max(in_sample))}"
        f"（原文 0.85–1.69），發表後 {days_after} 天 {len(after)} 組設定 {made} 組賺錢、{_pct(totals[0])} 到 {_pct(totals[-1])}"
        f"（QQQ {_pct((q_after or {}).get('total'))}、比特幣 {_pct((b_after or {}).get('total'))}）。"
        f"週期指標改用四年滾動排名，{len(keys)} 個裡 2021 那輪 {fired21} 個、2025 這輪 {fired25} 個有賣出訊號，"
        f"但 2018 年起贏過抱著不動的 {beat_hold} 個，疊在順勢混合上有加分的 {helped} 個。")

    # ----- notes -----
    def strict_text(key) -> str:
        a, b = stats[key]["2018"], strict["stats"][key]["2018"]
        return f"{_num((a or {}).get('ratio'))} → {_num((b or {}).get('ratio'))}"

    strict_rot = [strict["rot"][k]["main"]["ratio"] for k in rot_keys if strict["rot"][k]["main"]]
    main_rot = [rot[k]["main"]["ratio"] for k in rot_keys if rot[k]["main"]]
    ma_pick = r["picks"].get("ma50")
    sens = []
    for roll in ROLL:
        for top in TOPS:
            ms = [(rules[(k, roll, top, MAIN)], rules[(k, roll, top, "2014-01-01")]) for k in keys]
            ms = [(a, b) for a, b in ms if a and b]
            sens.append(f"{ROLL_NAME[roll]}・前 {top * 100:.0f}%：{len(ms)} 個裡 {sum(1 for a, _ in ms if a['vs'] > 1)} 個贏過抱著、"
                        f"2021 有訊號 {sum(1 for _, b in ms if in_round(b['exits'], ROUND_2021))} 個、"
                        f"2025 有訊號 {sum(1 for _, b in ms if in_round(b['exits'], ROUND_2025))} 個")
    signal_notes = []
    for k in keys:
        m = rules[(k, "4y", 0.10, MAIN)]
        if not m:
            continue
        ev = sorted([(d, "賣", p) for d, p in m["exits"]] + [(d, "買回", p) for d, p in m["entries"]])
        shown = ev[-14:]
        low, over, fx = rules[(k, "low", MAIN)], rules[(k, "overlay", MAIN)], rules[(k, "fixed", MAIN)]
        signal_notes.append(
            f"【{names[k]}・四年排名前 10% 賣】2018 年起每年 {_pct(m['ret'])}（抱著 {_pct(m['hold']['ret'])}）、報酬÷波動 {_num(m['ratio'])}"
            f"（抱著 {_num(m['hold']['ratio'])}）、{m['share']:.0f}% 的時間持有。2018 年起的訊號"
            + (f"共 {len(ev)} 次，最近 {len(shown)} 次" if len(ev) > len(shown) else f"共 {len(ev)} 次")
            + "：" + "、".join(f"{_d(d)} {what} {_usd(p)}" for d, what, p in shown) + "。"
            + (f"只在排名後一半時持有：報酬÷波動 {_num(low['ratio'])}、最深虧 {_pct(low['deepest'])}。" if low else "")
            + (f"疊在混合規則上：報酬÷波動 {_num(over['ratio'])}（混合本身 {_num(blend18['ratio'])}）、每年 {_pct(over['ret'])}"
               f"（混合本身 {_pct(blend18['ret'])}）。" if over else "")
            + (f"固定門檻版 2018 年起賣出 {len(fx['exits'])} 次。" if fx else ""))

    notes = [
        f"美國股市在這段期間（尤其 2019 到 2021、2023 到 2025）漲得很多，任何「把錢放在 QQQ」的做法都因此好看；"
        "這不代表換一段股市不好的年代還會一樣。2022 年股票和比特幣一起跌，請看第一張表最後一欄「2022 起」的數字。"
        "同樣地，黃金在 2024 到 2025 年漲幅異常大。",
        "全部用美元計價，沒有算匯率：不是住在美國的人，實際結果還要再加上美元對本國貨幣的漲跌，也沒有算任何稅（股利預扣稅、資本利得稅）"
        "和海外券商的費用。",
        "交易日怎麼對齊（第一部分）：只在美股有開盤的日子做決定。每個交易日的比特幣價格用的是幣安「標著那一天」的日線，它在 UTC 0 點收完，"
        "也就是美股收盤後三到四小時；決定就在那一刻做，只讀到當天的美股收盤價和到那根為止的比特幣日線，沒有讀任何更晚的資料，然後一直持有到下一個交易日的同一刻。"
        "比特幣的順勢訊號每個日曆天都算，但只在交易日讀取，所以週末發生的事要到週一才反應。基金那一邊是用當天美股收盤價記帳的，"
        "比決定的時刻早了三到四小時（實際上要在盤後時段成交，價格會有出入），這一點對放基金的做法略為有利。"
        "2017-08 以前的比特幣價格來自 blockchain.info，標的日期比幣安晚約一天，等於用的是更早就知道的價格。",
        "更嚴格的檢查：比特幣的訊號只讀「前一個日曆天」那根日線（美股收盤前約 20 小時就收完），其他都不變。2018 年起的報酬÷波動變化："
        f"混合＋空著 {strict_text(('blend', 'cash'))}、混合＋國庫券 {strict_text(('blend', 'bill'))}、混合＋QQQ（要高於 200 日均線）"
        f"{strict_text(('blend', 'qqq_f'))}、50 日均線＋國庫券 {strict_text(('ma50', 'bill'))}；公開輪動策略 10 組設定的報酬÷波動從 "
        f"{_num(min(main_rot))}–{_num(max(main_rot))} 變成 {_num(min(strict_rot))}–{_num(max(strict_rot))}。",
        f"空手部分的各種放法：200 日均線、60 天這兩個數字是事先照最常見的說法定的，沒有試別的；「較強的那個」是先比 QQQ 和黃金基金近 {MOM_DAYS} 個交易日的報酬，"
        f"挑高的那個，它如果低於自己的 {MA_LONG} 日均線就改放國庫券。基金用 Yahoo 還原配息後的收盤價，已含基金本身的費用。"
        + (f"事先挑的檢查換成 50 日均線規則：接起來的報酬÷波動 {_num(ma_pick['stat']['ratio'])}，同期一直放國庫券 "
           f"{_num(ma_pick['same']['bill']['ratio'])}。" if ma_pick and ma_pick["stat"] else "")
        + "事先挑只有六七個年度可以比，而且候選的放法本身是今天才想到要比的，所以這只是很弱的檢查。",
        "公開輪動策略的重做和原文的差別：（一）原文把比特幣取在美股收盤那一刻（Bitfinex 小時線），我們用幣安日線、晚三到四小時；"
        "（二）原文沒有寫「高過」含不含等於，也沒有寫訊號當天收盤就成交還是隔天，這裡用「嚴格高過前 w 個交易日的最高收盤」、當天那一刻成交；"
        "（三）原文測了 w ＝ 5、10、15、20、25、30、40、50，這裡只做 5、10、20、30、50；（四）原文不扣成本、現金 0%，這裡另外扣 0.10%。"
        f"原文的頁面有讀到，發表日期是頁面上寫的 {PUBLISHED}；樣本寫的是 2019-01-01 到 2025-12-31。發表之後到現在只有四個多月，"
        "這麼短的期間幾乎說明不了什麼，只能當成開始記錄。",
        "滾動排名的做法（第四張表）：四年這個長度剛好是比特幣一輪減半週期，是看過歷史之後才會選的數字；前 10% 和「回到中間值」也是。"
        "換成其他長度和門檻的結果（2018 年起，「贏過抱著」指最後的錢比較多；有沒有訊號看的是同一條規則從最早能算起的訊號）："
        + "；".join(sens) + "。「全部歷史」是和有資料以來所有讀數比（至少兩年）。排名一開始只要有半個視窗的資料就算，"
        "所以早期的排名是和比較短的歷史比。",
        "週期規則從頭到尾只交易幾次到十幾次，一兩次買賣的時機差幾週，整列的輸贏就可能翻過來，沒有辦法用統計方法驗證；"
        "下面把四年排名規則 2018 年起的每一次訊號列出來。滾動排名的規則在長期上漲的行情裡會很早就賣（讀數只要是四年來偏高就賣，"
        "不管絕對高度），賣出之後常常要等很久才買回。",
    ] + signal_notes + [
        "MVRV 來自 CoinMetrics 社群版資料，普爾倍數是用 blockchain.info 的礦工每日收入自己算的，兩個都晚一天使用（當天結束後才公布）。"
        "指標的算法和固定門檻版完全沿用「網路上最常聽到的買賣規則」那一頁；這一頁的固定門檻版統一用同一套成本算法重算，數字會和那一頁有些微出入。",
        "成本只用一個粗略的數字：每次買賣比特幣或基金扣成交金額的 0.10%（從一個換到另一個等於扣兩次），國庫券和現金進出當成不用錢。"
        "實際的手續費、價差、滑價、基金的盤後價差都會不同；混合規則每天小幅調整部位，實際上不會有人每天調。對照用的一直抱著沒有扣成本。",
        "「每年報酬」是把整段的總成果換算成每年的複利（第一部分一年用 252 個交易日、第二部分用 365 天）；「每年波動」是每天漲跌的起伏換算成一年；"
        "「報酬÷波動」是兩者相除；「最深曾經虧多少」是從最高點往下掉最多的幅度；「贏過的年數」是日曆年裡報酬比較高的年數，還沒過完的今年也算一年。",
        "過去的結果不保證未來會一樣。這裡只是整理這些做法過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "Quantpedia：Silicon vs Satoshi（比特幣／那斯達克 100 輪動，原文）", "url": PAGE},
        {"name": "btc-compass（用四年滾動排名取代固定門檻的公開專案）", "url": COMPASS},
        {"name": "blockchain.info 每日比特幣價格（2010 年起）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨 BTCUSDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance QQQ 每日股價", "url": "https://finance.yahoo.com/quote/QQQ/history"},
    ]
    if r["has_gld"]:
        sources.append({"name": "Yahoo Finance GLD 每日股價", "url": "https://finance.yahoo.com/quote/GLD/history"})
    if data["bill_source"] == "fred":
        sources.append({"name": "FRED：三個月期美國國庫券利率（DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"})
    elif data["bill_source"] == "bil":
        sources.append({"name": "Yahoo Finance BIL（1–3 個月美國國庫券基金）每日股價",
                        "url": "https://finance.yahoo.com/quote/BIL/history"})
    if "mvrv" in keys:
        sources.append({"name": "CoinMetrics 社群版資料（MVRV）", "url": "https://charts.coinmetrics.io/crypto-data/"})
    if "puell" in keys:
        sources.append({"name": "blockchain.info 礦工每日收入", "url": "https://www.blockchain.com/explorer/charts/miners-revenue"})

    return {
        "id": "rotation",
        "title": "順勢規則空手時錢放哪裡；週期指標換成滾動排名還有沒有用",
        "question": ("比特幣的順勢規則說「不要持有」的時候，那筆錢除了放著不動，有沒有更好的去處（國庫券、追蹤那斯達克 100 指數的基金、黃金基金）？"
                     "別人公開的「比特幣／那斯達克輪動」策略發表之後如何？還有，固定門檻已經碰不到的週期指標，如果改成和自己過去四年比，會不會恢復作用？"),
        "asof": _d(daily.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
