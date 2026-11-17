"""Historical study "trend_assets": the trend rules that were fixed on bitcoin, applied unchanged to other assets.

Three rules, no tuning per asset:
  R1  long after a 20-day closing high, to cash after a 10-day closing low
  R2  long while the close is above its 50-day average
  R3  blend: the share of 17 simple rules that are long (close above the N-day average for 8 values of N, N-day high
      in / N/2-day low out for 4 values of N, L-day return positive for 5 values of L), held as a fractional position

Assets: six other coins (Binance spot daily candles, every calendar day) plus BTCUSDT as the reference, and four
bitcoin-linked US listings (Yahoo Finance adjusted closes, trading days only): MSTR, COIN, MARA and the ETF IBIT.
For the shares there is also a cross-signal version: bitcoin's signal decides whether the share is held.

No look-ahead: a signal uses closes up to and including day t and decides the position held from the close of day t
to the close of the next day (next trading day for shares). For the cross-signal the decision made at the US close
of trading day p uses the bitcoin daily candle labelled p-1 (it closed at 00:00 UTC on p, about 20 hours before the
US close), so a position taken at Monday's close uses bitcoin's Sunday candle. Only finished candles are used.
Costs: 0.10% of the amount moved; cash earns nothing. Everything comes from public, key-less endpoints through the
dashboard's existing helpers (fixed user agent).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px
from dashboard.collectors import studies as old_studies

COST = 0.001  # 0.10% of the amount moved
CRYPTO_YEAR = 365
SHARE_YEAR = 252
SPLIT = "2022-01-01"  # earlier part: up to 2021-12-31, later part: from this day
R1_IN, R1_OUT = 20, 10
R2_DAYS = 50
BLEND_MA = (10, 20, 30, 50, 75, 100, 150, 200)
BLEND_CHANNEL = (10, 20, 50, 100)
BLEND_PAST = (14, 30, 60, 90, 180)
RULES = ("R1", "R2", "R3")
RULE_NAME = {"R1": f"創 {R1_IN} 日新高進場、創 {R1_OUT} 日新低出場", "R2": f"收盤高於 {R2_DAYS} 日均線才持有",
             "R3": "17 條規則混合（幾成規則說持有就持有幾成）"}
RULE_SHORT = {"R1": "新高進、新低出", "R2": f"{R2_DAYS} 日均線", "R3": "混合規則"}

REFERENCE = "BTCUSDT"
CRYPTO = (("BTCUSDT", "比特幣（對照，規則就是從它來的）"), ("ETHUSDT", "以太幣"), ("SOLUSDT", "索拉納"),
          ("BNBUSDT", "幣安幣"), ("XRPUSDT", "瑞波幣"), ("DOGEUSDT", "狗狗幣"), ("ADAUSDT", "艾達幣"))
SHARES = (("MSTR", "MSTR（Strategy 公司股票）"), ("COIN", "COIN（Coinbase 股票）"), ("MARA", "MARA（礦業公司股票）"),
          ("IBIT", "IBIT（比特幣 ETF）"))
SHARE_SINCE = {"MSTR": "2020-08-11"}  # the day the company announced its first bitcoin purchase
NAME = dict(CRYPTO + SHARES)


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
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _price(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _short(key: str) -> str:
    return NAME[key].split("（")[0]


# ---------- data ----------

def _crypto_close(symbol: str) -> pd.Series:
    """Finished Binance spot daily closes from the first listed day, one row per calendar day (UTC)."""
    rows, start = [], 0
    for _ in range(12):
        page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    series = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms").normalize())
    series = series[~series.index.duplicated()].sort_index()
    series = series[series.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    series = series[series > 0]
    if len(series) < 400:
        raise RuntimeError(f"幣安 {symbol} 日線只有 {len(series)} 筆")
    return series.asfreq("D").ffill()  # a missing day repeats the last close


def _share_close(symbol: str) -> pd.Series:
    """Finished adjusted daily closes from Yahoo Finance (trading days only, up to ten years)."""
    series = old_studies._yahoo(symbol).astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    # a bar dated before today's UTC date is always finished (the US session ends around 21:00 UTC the same date)
    series = series[series.index < pd.Timestamp(common.today_utc())]
    series = series[series > 0]
    if len(series) < 300:
        raise RuntimeError(f"Yahoo {symbol} 日線只有 {len(series)} 筆")
    return series


def _load() -> tuple[dict, list[str]]:
    data, dropped = {"crypto": {}, "shares": {}}, []
    for symbol, _ in CRYPTO:
        try:
            data["crypto"][symbol] = _crypto_close(symbol)
        except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the others
            dropped.append(f"{_short(symbol)}這次取不到資料（{type(exc).__name__}），沒有算進去。")
    for symbol, _ in SHARES:
        try:
            data["shares"][symbol] = _share_close(symbol)
        except Exception as exc:  # noqa: BLE001
            dropped.append(f"{_short(symbol)}這次取不到資料（{type(exc).__name__}），沒有算進去。")
    return data, dropped


# ---------- the rules ----------

def _channel(close: pd.Series, n: int) -> pd.Series:
    """1 after an n-day closing high, 0 after an n/2-day closing low, unchanged in between; cash until the first high."""
    m = int(round(n / 2))
    high = close.rolling(n, min_periods=n).max()
    low = close.rolling(m, min_periods=m).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0).where(high.notna())


def _signals(close: pd.Series) -> pd.DataFrame:
    """Signal at each day's close for R1, R2 and R3 (share of the day that is held next day; NaN = not available)."""
    parts: dict[str, pd.Series] = {}
    for n in BLEND_MA:
        mean = close.rolling(n, min_periods=n).mean()
        parts[f"A{n}"] = (close > mean).astype(float).where(mean.notna())
    for n in BLEND_CHANNEL:
        parts[f"C{n}"] = _channel(close, n)
    for n in BLEND_PAST:
        past = close / close.shift(n) - 1
        parts[f"D{n}"] = (past > 0).astype(float).where(past.notna())
    frame = pd.DataFrame(parts)
    blend = frame.mean(axis=1).where(frame.notna().all(axis=1))  # only once all 17 rules exist
    return pd.DataFrame({"R1": parts[f"C{R1_IN}"], "R2": parts[f"A{R2_DAYS}"], "R3": blend})


def _bitcoin_signal_for(index: pd.DatetimeIndex, btc_signal: pd.DataFrame) -> pd.DataFrame:
    """Bitcoin's signal as it is known at the US close of each trading day p: the candle labelled p-1 (closed at
    00:00 UTC on p). Monday's close therefore uses bitcoin's Sunday candle; nothing later than that is read."""
    out = btc_signal.reindex(index - pd.Timedelta(days=1))
    out.index = index
    return out


# ---------- measuring ----------

def _parts(net: np.ndarray, year: int) -> dict:
    n = len(net)
    if n < 2:
        return {"days": n, "ret": None, "vol": None, "ratio": None, "deepest": None, "growth": None}
    growth = float(np.prod(1 + net))
    ret = (growth ** (year / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(year) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"days": n, "ret": ret, "vol": vol, "ratio": (ret / vol if vol > 1e-9 else 0.0), "deepest": deepest,
            "growth": growth}


def _measure(net: pd.Series, position: pd.Series, year: int) -> dict:
    out = _parts(net.values, year)
    out["share"] = float(position.mean() * 100) if len(position) else None
    out["years"] = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    return out


def _evaluate(close: pd.Series, signal: pd.DataFrame, start: pd.Timestamp, year: int) -> dict:
    """Rules against buy-and-hold over the same dates. signal: value known at each row's close."""
    change = close.pct_change()
    position = signal.shift(1)  # signal at the close of row t -> position during row t+1
    ready = position.notna().all(axis=1) & change.notna() & (position.index >= start)
    if not ready.any():
        raise RuntimeError("規則還沒有足夠的資料")
    first = ready.idxmax()
    position, change = position[position.index >= first], change[change.index >= first]
    moved = position.diff().abs()
    moved.iloc[0] = position.iloc[0]  # every rule starts from cash
    net = position.mul(change, axis=0) - COST * moved
    hold = _measure(change, pd.Series(1.0, index=change.index), year)
    split = pd.Timestamp(SPLIT)
    out = {"first": first, "last": change.index[-1], "hold": hold, "rules": {}, "position": position, "parts": {}}
    for rule in RULES:
        m = _measure(net[rule], position[rule], year)
        m["beat"] = sum(1 for y, v in m["years"].items() if v > hold["years"][y])
        m["n_years"] = len(m["years"])
        out["rules"][rule] = m
    for name, rows in (("early", change.index < split), ("late", change.index >= split)):
        if rows.sum() >= year and (~rows).sum() >= year:  # only when both parts are at least a year long
            out["parts"][name] = {"hold": _parts(change.values[rows], year),
                                  **{rule: _parts(net[rule].values[rows], year) for rule in RULES}}
    return out


def _flip(close: pd.Series, state: float) -> float:
    """The next close that would flip R1: an (R1_IN)-day closing high needs a close at or above the highest of the
    last R1_IN-1 closes; an (R1_OUT)-day closing low needs a close at or below the lowest of the last R1_OUT-1."""
    if state == 1:
        return float(close.iloc[-(R1_OUT - 1):].min())
    return float(close.iloc[-(R1_IN - 1):].max())


def _compute(data: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    assets: dict[str, dict] = {}
    btc_signal = _signals(data["crypto"][REFERENCE]) if REFERENCE in data["crypto"] else None
    for kind, year in (("crypto", CRYPTO_YEAR), ("shares", SHARE_YEAR)):
        for key, close in data[kind].items():
            signal = _signals(close)
            start = pd.Timestamp(SHARE_SINCE.get(key, "1900-01-01"))
            cross = None
            if kind == "shares" and btc_signal is not None:
                cross = _bitcoin_signal_for(close.index, btc_signal)
                ready = cross.shift(1).notna().all(axis=1)
                if ready.any():
                    start = max(start, ready.idxmax())  # own and bitcoin-signal versions cover the same dates
                else:
                    cross = None
            try:
                own = _evaluate(close, signal, start, year)
            except RuntimeError:
                continue
            item = {"key": key, "kind": kind, "year": year, "close": close, "own": own, "cross": None,
                    "state": signal.iloc[-1], "cross_state": None, "last": close.index[-1],
                    "flip": _flip(close, float(signal["R1"].iloc[-1])),
                    "mean": float(close.iloc[-R2_DAYS:].mean())}
            if cross is not None:
                item["cross"] = _evaluate(close, cross, own["first"], year)
                item["cross_state"] = cross.iloc[-1]
            assets[key] = item

    tested = [k for k in assets if k != REFERENCE]  # the assets the rules were never looked at on

    def board(keys: list[str], rule: str, part: str | None = None, version: str = "own") -> dict:
        better_ratio, shallower, richer, used = [], [], [], []
        for k in keys:
            run = assets[k][version]
            if run is None:
                continue
            if part is None:
                mine, hold = run["rules"][rule], run["hold"]
            elif part in run["parts"]:
                mine, hold = run["parts"][part][rule], run["parts"][part]["hold"]
            else:
                continue
            used.append(k)
            if mine["ratio"] > hold["ratio"]:
                better_ratio.append(k)
            if mine["deepest"] > hold["deepest"]:
                shallower.append(k)
            if mine["growth"] > hold["growth"]:
                richer.append(k)
        return {"n": len(used), "used": used, "ratio": better_ratio, "deepest": shallower, "money": richer}

    coins = [k for k in tested if assets[k]["kind"] == "crypto"]
    shares = [k for k in tested if assets[k]["kind"] == "shares"]
    boards = {rule: {"all": board(tested, rule), "coins": board(coins, rule), "shares": board(shares, rule),
                     "early": board(tested, rule, "early"), "late": board(tested, rule, "late"),
                     "cross": board(shares, rule, version="cross")} for rule in RULES}
    return {"assets": assets, "tested": tested, "coins": coins, "shares": shares, "boards": boards}


def _chance(k: int, n: int) -> float | None:
    """Chance in % of k or more out of n when each one is a fair coin and they are independent."""
    if n <= 0:
        return None
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n * 100


# ---------- the study ----------

def study() -> dict:
    data, problems = _load()
    if not data["crypto"] and not data["shares"]:
        raise RuntimeError("所有資產都取不到資料")
    r = _compute(data)
    assets, boards, tested = r["assets"], r["boards"], r["tested"]
    if not tested:
        raise RuntimeError("比特幣以外的資產都取不到資料")
    n_all = len(tested)
    asof = max(a["last"] for a in assets.values())

    def pair(mine, hold, fmt) -> str:
        return f"{fmt(mine)}（{fmt(hold)}）"

    def parts_text(run: dict, rule: str) -> str:
        bits = []
        for part, name in (("early", "前"), ("late", "後")):
            if part in run["parts"]:
                p = run["parts"][part]
                bits.append(f"{name} {_num(p[rule]['ratio'])}（{_num(p['hold']['ratio'])}）")
        return "；".join(bits) if bits else "—"

    def asset_rows(rule: str) -> list:
        rows = []
        for key in [k for k, _ in CRYPTO + SHARES if k in assets]:
            run = assets[key]["own"]
            m, hold = run["rules"][rule], run["hold"]
            rows.append([f"{NAME[key]}｜{_d(run['first'])[:7]} 起", pair(m["ret"], hold["ret"], _pct),
                         pair(m["vol"], hold["vol"], lambda v: _pct(v, signed=False)),
                         pair(m["ratio"], hold["ratio"], _num), pair(m["deepest"], hold["deepest"], _pct),
                         f"{m['share']:.0f}%", f"{m['n_years']} 年裡 {m['beat']} 年", parts_text(run, rule)])
        return rows

    def board_text(rule: str, which: str = "all") -> str:
        b = boards[rule][which]
        if not b["n"]:
            return "沒有夠長的資料"
        return (f"{b['n']} 個裡 {len(b['ratio'])} 個報酬÷波動變高、{len(b['deepest'])} 個最深虧損變淺、"
                f"{len(b['money'])} 個最後的錢比一直抱著多")

    def caption(rule: str) -> str:
        return (f"「{RULE_NAME[rule]}」原封不動用在每個資產上，括號裡是同一段日子一直抱著不動的數字。"
                f"不算比特幣自己，{board_text(rule)}（其中幣 {board_text(rule, 'coins')}；股票 {board_text(rule, 'shares')}）。"
                f"只看 2021 年底以前：{board_text(rule, 'early')}；只看 2022 年起：{board_text(rule, 'late')}"
                "（前後兩段都至少有一年資料的資產才算）。最後一欄「前／後」就是這兩段各自的報酬÷波動。"
                "幣一年用 365 天換算，股票用 252 個交易日。")

    columns = ["資產｜起算月份", "每年報酬（抱著）", "每年波動（抱著）", "報酬÷波動（抱著）", "最深曾經虧多少（抱著）",
               "持有天數比例", "賺得比抱著多的年數", "前／後兩段的報酬÷波動（抱著）"]
    tables = [{"caption": caption("R1"), "columns": columns, "rows": asset_rows("R1")},
              {"caption": caption("R3") + "「持有天數比例」在這張表是平均持有幾成。", "columns": columns,
               "rows": asset_rows("R3")}]

    # ----- table 3: the shares, own signal against bitcoin's signal -----
    share_rows = []

    def share_row(key: str, how: str, m: dict, wins: str) -> list:
        return [_short(key), how, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                f"{m['share']:.0f}%", wins]

    for key in r["shares"]:
        item = assets[key]
        run, cross = item["own"], item["cross"]
        share_rows.append(share_row(key, f"一直抱著（{_d(run['first'])} 起）", run["hold"], "—"))
        for rule in (RULES if key == "MSTR" else ("R1",)):
            m = run["rules"][rule]
            share_rows.append(share_row(key, f"{RULE_SHORT[rule]}：看它自己的股價", m, f"{m['n_years']} 年裡 {m['beat']} 年"))
            if cross is not None:
                m = cross["rules"][rule]
                share_rows.append(share_row(key, f"{RULE_SHORT[rule]}：看比特幣的價格", m,
                                            f"{m['n_years']} 年裡 {m['beat']} 年"))
    if share_rows:
        cross_board = boards["R1"]["cross"]
        cross_text = ""
        if cross_board["n"]:
            cross_text = (f"用比特幣訊號的「新高進、新低出」和一直抱著比：{cross_board['n']} 檔裡 {len(cross_board['ratio'])} 檔報酬÷波動變高、"
                          f"{len(cross_board['deepest'])} 檔最深虧損變淺、{len(cross_board['money'])} 檔最後的錢比較多。")
        tables.append({
            "caption": ("和比特幣連動的股票：同一條規則，一種是看股票自己的收盤價決定，一種是看比特幣的收盤價決定要不要持有這檔股票。"
                        "看比特幣的版本：每個交易日美股收盤時，用的是「前一個日曆天」那根比特幣日線（它在當天 UTC 0 點、也就是美股收盤前約 20 小時就收完了），"
                        "決定從這次收盤抱到下一個交易日收盤；所以週一收盤的決定用的是比特幣週日那根日線。" + cross_text
                        + "MSTR 從 2020-08-11（公司宣布第一次買比特幣那天）算起，其他從規則有足夠資料那天算起。"),
            "columns": ["股票", "做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "持有天數比例", "賺得比抱著多的年數"],
            "rows": share_rows[:16]})

    # ----- table 4: today -----
    def state_text(v) -> str:
        return "持有" if v == 1 else "現金"

    today_rows = []
    for key in [k for k, _ in CRYPTO + SHARES if k in assets]:
        item = assets[key]
        state = item["state"]
        flip = (f"收盤 ≤ {_price(item['flip'])} 就轉現金" if state["R1"] == 1 else f"收盤 ≥ {_price(item['flip'])} 就進場")
        today_rows.append([NAME[key].split("（規則")[0].replace("（對照，規則就是從它來的）", "（對照）"),
                           f"{_price(item['close'].iloc[-1])}（{_d(item['last'])[5:]}）", state_text(state["R1"]), flip,
                           state_text(state["R2"]), _price(item["mean"]), f"{float(state['R3']) * 100:.0f}%"])
    for key in r["shares"]:
        cs = assets[key]["cross_state"]
        if cs is not None and key == "MSTR":
            today_rows.append([f"{_short(key)}（改看比特幣的訊號）", "—", state_text(cs["R1"]), "看比特幣那一列的價位",
                               state_text(cs["R2"]), "—", f"{float(cs['R3']) * 100:.0f}%"])
    tables.append({
        "caption": ("照規則下一天（股票是下一個交易日）的狀態，用最後一根收完的日線算。翻轉價：現在是現金的話，下一個收盤要創 "
                    f"{R1_IN} 日新高，等於要大於等於前 {R1_IN - 1} 個收盤裡最高的那個；現在是持有的話，下一個收盤要創 {R1_OUT} 日新低，"
                    f"等於要小於等於前 {R1_OUT - 1} 個收盤裡最低的那個。這個價位每天都會跟著變。價格單位是美元。"),
        "columns": ["資產", "最後收盤（日期）", "新高進、新低出", "它的翻轉價", f"{R2_DAYS} 日均線規則", f"{R2_DAYS} 日均線在哪",
                    "混合規則持有幾成"],
        "rows": today_rows[:16]})

    # ----- now -----
    long_r1 = [k for k in tested if assets[k]["state"]["R1"] == 1]
    long_r2 = [k for k in tested if assets[k]["state"]["R2"] == 1]
    now = [{"label": f"「新高進、新低出」照規則下一天是持有的（不算比特幣，{n_all} 個資產）",
            "value": f"{n_all} 個裡 {len(long_r1)} 個" + ("：" + "、".join(_short(k) for k in long_r1) if long_r1 else "")},
           {"label": f"「{R2_DAYS} 日均線」照規則下一天是持有的", "value": f"{n_all} 個裡 {len(long_r2)} 個"
            + ("：" + "、".join(_short(k) for k in long_r2) if long_r2 else "")}]
    if REFERENCE in assets:
        b = assets[REFERENCE]
        now.append({"label": f"比特幣自己（{_d(b['last'])} 收盤 {_price(b['close'].iloc[-1])} 美元）",
                    "value": (f"新高進、新低出：{state_text(b['state']['R1'])}，"
                              + (f"收盤 ≤ {_price(b['flip'])} 轉現金" if b["state"]["R1"] == 1 else f"收盤 ≥ {_price(b['flip'])} 進場")
                              + f"；{R2_DAYS} 日均線：{state_text(b['state']['R2'])}；混合規則持有 {float(b['state']['R3']) * 100:.0f}%")})
    mstr = assets.get("MSTR")
    if mstr is not None and mstr["cross_state"] is not None:
        now.append({"label": "用比特幣的訊號看 MSTR（下一個交易日）",
                    "value": f"新高進、新低出：{state_text(mstr['cross_state']['R1'])}；看 MSTR 自己的股價："
                             f"{state_text(mstr['state']['R1'])}"})

    # ----- verdict -----
    b1, b3 = boards["R1"]["all"], boards["R3"]["all"]
    no_help = [k for k in b1["used"] if k not in b1["ratio"]]
    poorer = [k for k in b1["used"] if k not in b1["money"]]
    verdict = (f"把在比特幣上訂好的規則一個字不改，搬到 {n_all} 個沒拿來調過的資產（{len(r['coins'])} 種幣、{len(r['shares'])} 檔股票）："
               f"「{RULE_NAME['R1']}」{board_text('R1')}；混合規則是 {b3['n']} 個裡 {len(b3['ratio'])} 個、{len(b3['deepest'])} 個、"
               f"{len(b3['money'])} 個。")
    if no_help:
        verdict += f"報酬÷波動沒有變高的是{'、'.join(_short(k) for k in no_help)}"
        verdict += ("；" if poorer else "。")
    if poorer:
        verdict += f"最後的錢沒有比一直抱著多的是{'、'.join(_short(k) for k in poorer)}。"
    if not no_help and not poorer:
        verdict += "這條規則在每個資產上報酬÷波動都變高、最後的錢也都比較多。"
    if mstr is not None:
        own, hold = mstr["own"]["rules"]["R1"], mstr["own"]["hold"]
        verdict += f"MSTR（{_d(mstr['own']['first'])} 起）一直抱著是每年 {_pct(hold['ret'])}、最深曾經虧 {_pct(hold['deepest'])}；"
        if mstr["cross"] is not None:
            cross = mstr["cross"]["rules"]["R1"]
            verdict += (f"用比特幣的訊號決定要不要持有它是每年 {_pct(cross['ret'])}、最深虧 {_pct(cross['deepest'])}、"
                        f"報酬÷波動 {_num(cross['ratio'])}（抱著 {_num(hold['ratio'])}），")
        verdict += f"用它自己的股價當訊號是每年 {_pct(own['ret'])}、最深虧 {_pct(own['deepest'])}、報酬÷波動 {_num(own['ratio'])}。"
        if mstr["cross"] is not None:
            other = mstr["cross"]["rules"]["R2"]
            verdict += (f"同樣看比特幣、換成「{R2_DAYS} 日均線」那條則是每年 {_pct(other['ret'])}、最深虧 {_pct(other['deepest'])}，"
                        "兩條規則差很多，看比特幣訊號操作 MSTR 的結果並不穩定。")

    # ----- notes -----
    luck = []
    for rule in ("R1", "R3"):
        b = boards[rule]["all"]
        luck.append(f"「{RULE_SHORT[rule]}」{b['n']} 個裡 {len(b['ratio'])} 個變高，這種結果或更好出現的機會約 "
                    f"{_num(_chance(len(b['ratio']), b['n']), 1)}%")
    b2 = boards["R2"]
    notes = [
        "三條規則都是事先在比特幣上定下來的，這裡沒有為任何一個資產調整過參數："
        f"（一）{RULE_NAME['R1']}；（二）{RULE_NAME['R2']}；（三）混合：收盤高於 N 日均線（N = 10、20、30、50、75、100、150、200）、"
        "創 N 日新高進場且創 N/2 日新低出場（N = 10、20、50、100）、過去 L 天是漲的（L = 14、30、60、90、180），一共 17 條，"
        "幾成規則說持有就持有幾成資金。每條規則都用當天收盤時已經知道的價格，決定下一天（股票是下一個交易日）的部位。",
        f"第二條規則「{RULE_NAME['R2']}」沒有另外列表，結果是：不算比特幣，{board_text('R2')}；"
        f"2021 年底以前 {board_text('R2', 'early')}；2022 年起 {board_text('R2', 'late')}。",
        "這些資產是一起漲一起跌的：幣大多跟著比特幣走，這幾檔股票和 ETF 本身就是押在比特幣上的。所以這不是 "
        f"{n_all} 次互相獨立的測試，比較像同一件事重複看了好幾次，「幾個裡幾個」這種數字會把證據說得比實際強。",
        "簡單的運氣檢查：如果規則完全沒有本事，在每個資產上報酬÷波動變高或變低各一半機會，而且資產彼此無關，那麼"
        + "；".join(luck) + "。但資產彼此高度連動，真正獨立的測試遠少於這個數目，所以實際上靠運氣出現的機會比這些數字高，不能當成嚴格的檢定。"
        "另外，把一部分時間換成現金本來就會讓波動和最深虧損變小，「最深虧損變淺」幾乎是這類規則必然的結果，不代表它有本事。",
        "這裡的幣都是今天還活著、還排在前面的幣（倖存者）。已經歸零或下市的幣不在裡面，所以這個比較是偏的：一直抱著的數字因為只挑到活下來的而偏好看；"
        "而規則在一路跌到歸零的幣上本來可以早早出場、少虧很多，這種對規則有利的例子也被漏掉了。兩邊都偏，偏向哪一邊比較多這裡沒有辦法量出來。"
        "會選這幾檔股票，也是因為現在知道它們和比特幣有關。",
        "股票只有交易日才有價格：週末、假日和盤後盤前的大漲大跌，規則都來不及反應，只能等下一次收盤；這裡也假設剛好能用收盤價成交。"
        "股價用的是 Yahoo 還原過拆股和配息的收盤價。各資產的起算日不同（表上有寫），是從三條規則都有足夠資料（約 200 天）之後才開始算，"
        "所以每個資產比的期間不一樣長，彼此的數字不能直接互比。COIN 和 IBIT 的歷史很短，「幾年裡幾年」只有很少的年數，第一年和最後一年也不是完整的一年。",
        "看比特幣訊號操作股票的做法：比特幣的價格用幣安現貨日線（UTC 0 點收盤）。美股收盤時最新一根收完的比特幣日線是前一個日曆天的那根，"
        "只用到那一根為止，沒有用到當天還沒收完的價格；代價是訊號大約慢了 20 小時。MARA 從 2018-03 算起，是因為幣安的比特幣日線從 2017-08 才有。",
        f"成本只用一個粗略的數字：每次買賣扣成交金額的 {COST * 100:.2f}%，幣和股票都一樣。實際的手續費、買賣價差和滑價會隨市場、金額和年代不同，"
        "小型幣早年的價差尤其大。混合規則只在持有成數改變時計算成本。一直抱著沒有扣成本。持有現金假設沒有利息，也沒有算稅。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「每年波動」是每天漲跌的起伏換算成一年；「報酬÷波動」是兩者相除，"
        "越高代表每承受一分起伏換到的報酬越多；「最深曾經虧多少」是從最高點往下掉最多的幅度。「最後的錢比較多」是指整段期間結束時的總成果。"
        "比特幣那一列用的是幣安 2017-08 起的資料，所以和用更長歷史算的數字會有出入。",
        "過去的結果不保證未來會一樣。這裡只是整理這幾條規則過去在這些資產上做了什麼，不是買賣建議。",
    ] + problems

    sources = [{"name": "幣安現貨日線（BTC、ETH、SOL、BNB、XRP、DOGE、ADA 對 USDT）",
                "url": "https://www.binance.com/zh-TC/markets/overview"}]
    sources += [{"name": f"Yahoo Finance {key} 每日股價", "url": f"https://finance.yahoo.com/quote/{key}/history"}
                for key, _ in SHARES if key in assets]

    return {
        "id": "trend_assets",
        "title": "比特幣上的順勢規則，搬到別的資產還有用嗎",
        "question": "把在比特幣上訂好的順勢規則一個字不改，拿去用在其他幣和跟比特幣連動的股票上，還能少虧、又跟得上嗎？",
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
