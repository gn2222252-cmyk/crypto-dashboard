"""Historical study "trend_world": the four verified bitcoin trend rules, unchanged, on traditional markets over decades.

Rules (exactly the ones on the signals tab, built by dashboard.collectors.signals._targets):
  b5520  all in after a 55-day closing high, all out after a 20-day closing low
  ma50   all in while the close is above its 50-day average
  blend  the share of the 17 simple trend rules that are long (trend_ensemble._signals)
  vol40  blend x min(1, 40% / 30-day volatility)
Markets: US, Taiwan, Japanese and Hong Kong stock indices, two Taiwan shares, emerging markets, gold, long Treasuries,
oil and the US dollar index, from Yahoo Finance's public chart endpoint (daily, as far back as it goes).

No look-ahead: each target uses closes up to and including day t and is held during the next trading day; the
arithmetic is trend_ensemble._run (cost on the amount traded, starting from cash), as in signals._record. Buy-and-hold
pays no cost. Cash earns nothing; a second version lets idle cash in US-dollar markets earn the 13-week US T-bill
yield (Yahoo ^IRX, from 1960). 252 trading days a year. Only finished daily bars; public, key-less sources, fixed user
agent. The fetched closes are kept in cache/_store/trend_world_closes.json and used only when a fetch fails.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import signals as sig
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

YEAR = 252
COST = 0.001
FALL = 0.20  # a "big fall": the close drops 20% from a peak, ended by a 20% rebound from the low
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STORE_PATH = common.STORE / "trend_world_closes.json"
RULES = ("b5520", "ma50", "blend", "vol40")
BINARY = ("b5520", "ma50")
SHORT = {"b5520": "55／20", "ma50": "50 日均線", "blend": "混合", "vol40": "混合＋波動"}
LONG = {"b5520": "55 日新高進、20 日新低出", "ma50": "收盤高於 50 日均線才持有", "blend": "17 條順勢規則混合",
        "vol40": "混合，再依波動調整（40%）"}
PERIODS = (("all", "整段", None, None), ("pre2000", "2000 年以前", None, "1999-12-31"),
           ("mid", "2000–2015", "2000-01-01", "2015-12-31"), ("late", "2016 年起", "2016-01-01", None))
# symbol, plain name, group, US-dollar market (cash could earn US T-bills), total-return proxy (adjusted closes)
MARKETS = (
    ("^GSPC", "美國標普 500 指數", "stock", True, False),
    ("^IXIC", "美國那斯達克綜合指數", "stock", True, False),
    ("^DJI", "美國道瓊工業指數", "stock", True, False),
    ("^TWII", "台灣加權指數", "stock", False, False),
    ("0050.TW", "元大台灣50（0050）", "stock", False, True),
    ("2330.TW", "台積電（2330）", "stock", False, True),
    ("^N225", "日本日經 225 指數", "stock", False, False),
    ("^HSI", "香港恆生指數", "stock", False, False),
    ("EEM", "新興市場股票 ETF（EEM）", "stock", True, True),
    ("GC=F", "黃金期貨（GC=F）", "gold", True, False),
    ("GLD", "黃金 ETF（GLD）", "gold", True, True),
    ("TLT", "美國長天期公債 ETF（TLT）", "bond", True, True),
    ("CL=F", "原油期貨（CL=F）", "oil", True, False),
    ("DX-Y.NYB", "美元指數（DX）", "fx", True, False),
)
NAME = {m[0]: m[1] for m in MARKETS}
DEEP = ("^GSPC", "^TWII")
TR_CHECK = "^SP500TR"  # S&P 500 with dividends reinvested, from 1988
RATE = "^IRX"
ROLL_FROM = "1960-01-01"
ROLL_YEARS = 10

_pct, _num, _price, _d = ta._pct, ta._num, ta._price, ta._d


def _short(symbol: str) -> str:
    return NAME[symbol].split("（")[0] if symbol in ("^GSPC", "^IXIC", "^DJI", "^TWII", "^N225", "^HSI") else \
        NAME[symbol]


# ---------- data ----------

def _fetch(symbol: str) -> pd.Series:
    """Every finished daily close Yahoo has (adjusted for splits and payouts where Yahoo adjusts), dated by the
    exchange's own calendar day. A bar of a session that has not ended yet is dropped."""
    now = int(time.time())
    reply = requests.get(CHART.format(symbol=symbol), params={"period1": -1500000000, "period2": now, "interval": "1d"},
                         headers=common.HEADERS, timeout=30)
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    meta = result.get("meta", {})
    if meta.get("dataGranularity") not in (None, "1d"):
        raise RuntimeError(f"Yahoo {symbol} 沒有給日線")
    stamps = np.asarray(result["timestamp"], dtype="int64")
    values = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or result["indicators"]["quote"][0]["close"]
    values = np.asarray([np.nan if v is None else float(v) for v in values])
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    keep = stamps <= now
    if regular.get("start") and regular.get("end") and now < int(regular["end"]):
        keep &= stamps < int(regular["start"])  # the current session is still open
    # Yahoo stamps futures bars at local midnight using standard time all year; two hours of slack keeps summer bars
    # on their own trading day (stock bars are stamped at the open, so they are unaffected)
    offset = int(meta.get("gmtoffset") or 0) + 2 * 3600
    index = pd.to_datetime(stamps + offset, unit="s").normalize()
    series = pd.Series(values, index=index)[keep]
    series = series[np.isfinite(series) & (series > 0)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) < 600:
        raise RuntimeError(f"Yahoo {symbol} 日線只有 {len(series)} 筆")
    return series


JUMP = 2.5  # a one-day move beyond x2.5 or /2.5 in these markets is a data error (an unadjusted split), not a price


def _broken(series: pd.Series) -> pd.Timestamp | None:
    """The day after the last impossible one-day jump (the data before it does not join up), or None."""
    move = np.abs(np.log(series / series.shift(1)))
    bad = move[move > math.log(JUMP)]
    return None if bad.empty else bad.index[-1]


def _read_store() -> dict:
    try:
        stored = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return stored if isinstance(stored, dict) else {}
    except Exception:  # noqa: BLE001 - first run or damaged file
        return {}


def _load() -> tuple[dict, dict, list[str]]:
    stored, fresh, data, problems, info = _read_store(), {}, {}, [], {}
    for symbol in [m[0] for m in MARKETS] + [TR_CHECK, RATE]:
        try:
            series = _fetch(symbol)
            fresh[symbol] = {"dates": [d.strftime("%Y-%m-%d") for d in series.index],
                             "close": [float(f"{v:.8g}") for v in series.values]}
            info[symbol] = "ok"
        except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the others
            old = stored.get(symbol)
            if isinstance(old, dict) and old.get("dates"):
                series = pd.Series(old["close"], index=pd.to_datetime(old["dates"]), dtype=float)
                fresh[symbol] = old
                info[symbol] = "store"
                problems.append(f"{NAME.get(symbol, symbol)}這次從 Yahoo 取不到（{type(exc).__name__}），用上次存下的資料"
                                f"（到 {old['dates'][-1]}）。")
            else:
                if symbol in NAME:
                    problems.append(f"{NAME[symbol]}這次取不到資料（{type(exc).__name__}），沒有算進去。")
                continue
        cut = _broken(series) if symbol != RATE else None  # a yield near zero can legitimately jump
        if cut is not None:
            series = series[series.index >= cut]
            if symbol in NAME:
                problems.append(f"{NAME[symbol]}的 Yahoo 資料在 {_d(cut)} 有一天漲超過 150% 或跌超過 60%（多半是拆股沒有調整好，接不起來），"
                                f"所以只用 {_d(cut)} 以後的資料。")
        data[symbol] = series
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(json.dumps(fresh, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the store is only a fallback
        pass
    return data, info, problems


# ---------- simulating (the same arithmetic as signals._record) ----------

def _falls(close: pd.Series, fall: float = FALL) -> list[tuple]:
    """signals._falls with the threshold as an argument."""
    out, peak_i, low_i, falling = [], 0, 0, False
    v = close.values
    for i in range(1, len(v)):
        if not falling:
            if v[i] > v[peak_i]:
                peak_i = i
            elif v[i] <= v[peak_i] * (1 - fall):
                falling, low_i = True, i
        else:
            if v[i] < v[low_i]:
                low_i = i
            elif v[i] >= v[low_i] * (1 + fall):
                out.append((close.index[peak_i], close.index[low_i]))
                falling, peak_i = False, i
    if falling:
        out.append((close.index[peak_i], close.index[low_i]))
    return out


def _net(close: pd.Series, target: pd.Series, cost: float = COST, delay: int = 1,
         rate: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Daily net return and position: target decided at the close of day t, held during day t+delay."""
    change = close.pct_change().fillna(0.0)
    position = target.shift(delay).fillna(0.0)
    net, _ = te._run(position.values, change.values, cost)
    net = pd.Series(net, index=close.index)
    if rate is not None:
        net = net + (1 - position) * rate
    return net, position


def _growth_deep(r: np.ndarray) -> tuple[float, float]:
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float(eq[-1]), float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


def _stats(r: np.ndarray) -> dict:
    growth, deep = _growth_deep(r)
    ret = (growth ** (YEAR / len(r)) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(r, ddof=1) * math.sqrt(YEAR) * 100) if len(r) > 1 else 0.0
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else 0.0, "deep": deep, "growth": growth}


def _trades(p: np.ndarray, net: np.ndarray) -> dict:
    """Closed all-in trades (signals._record's loop)."""
    results, i = [], 0
    while i < len(p):
        if p[i] > 0.5 and (i == 0 or p[i - 1] < 0.5):
            j = i
            while j + 1 < len(p) and p[j + 1] > 0.5:
                j += 1
            if j + 1 < len(p):
                results.append((float(np.prod(1 + net[i:j + 1])) - 1) * 100)
            i = j + 1
        else:
            i += 1
    won = [g for g in results if g > 0]
    lost = [g for g in results if g <= 0]
    return {"trades": len(results), "wins": len(won), "avg_win": float(np.mean(won)) if won else None,
            "avg_loss": float(np.mean(lost)) if lost else None}


def _record(close: pd.Series, net: pd.Series, pos: pd.Series, start, end, binary: bool) -> dict | None:
    rows = (close.index >= start) & (close.index <= end)
    if rows.sum() < YEAR:
        return None
    c = close[rows]
    hold = close.pct_change().fillna(0.0)[rows]  # as signals._record: the same days' moves, no cost
    n, p = net[rows], pos[rows]
    mine, base = _stats(n.values), _stats(hold.values)
    years_rule = {y: float(np.prod(1 + g.values) - 1) for y, g in n.groupby(n.index.year)}
    years_hold = {y: float(np.prod(1 + g.values) - 1) for y, g in hold.groupby(hold.index.year)}
    falls, taken = [], []
    for peak, low in _falls(c):
        span = (n.index > peak) & (n.index <= low)
        if span.sum() < 2:
            continue
        falls.append(float(c[low] / c[peak] - 1) * 100)
        taken.append(float(np.prod(1 + n[span].values) - 1) * 100)
    out = {"rule": mine, "hold": base, "years": len(years_rule),
           "beat": int(sum(years_rule[y] > years_hold[y] for y in years_rule)), "share": float(p.mean() * 100),
           "falls": len(falls), "fall_hold": float(np.mean(falls)) if falls else None,
           "fall_rule": float(np.mean(taken)) if taken else None, "first": c.index[0], "last": c.index[-1]}
    if binary:
        out.update(_trades(p.values, n.values))
    return out


def _rolling(close: pd.Series, nets: dict, begin: pd.Timestamp) -> dict:
    """Every 10-year window starting on the first trading day of a month: share of windows in which each rule ended
    with more money than holding, and in which its deepest loss was shallower."""
    change = close.pct_change().fillna(0.0)
    idx = close.index
    first = max(begin, pd.Timestamp(ROLL_FROM))
    starts = pd.Series(idx[idx >= first]).groupby(pd.Series(idx[idx >= first]).dt.to_period("M")).first()
    out = {k: {"money": 0, "deep": 0} for k in nets}
    count = 0
    for s in starts:
        e = s + pd.DateOffset(years=ROLL_YEARS)
        if e > idx[-1]:
            break
        rows = (idx >= s) & (idx < e)
        hg, hd = _growth_deep(change.values[rows])
        count += 1
        for k, net in nets.items():
            g, d = _growth_deep(net.values[rows])
            out[k]["money"] += g > hg
            out[k]["deep"] += d > hd
    return {"n": count, "first": starts.iloc[0] if len(starts) else None,
            **{k: {"money": v["money"] / count * 100 if count else None,
                   "deep": v["deep"] / count * 100 if count else None} for k, v in out.items()}}


def _market(symbol: str, close: pd.Series, rate: pd.Series | None, deep: bool) -> dict:
    targets, _ = sig._targets(close, YEAR)
    begin = targets["blend"].first_valid_index()
    if begin is None:
        raise RuntimeError("資料不夠長")
    last = close.index[-1]
    out = {"symbol": symbol, "close": close, "begin": begin, "last": last, "periods": {}, "cash": {}, "targets": targets}
    nets = {}
    for rule in RULES:
        net, pos = _net(close, targets[rule])
        nets[rule] = net
        for key, _, a, b in PERIODS:
            s = max(begin, pd.Timestamp(a)) if a else begin
            e = min(last, pd.Timestamp(b)) if b else last
            rec = _record(close, net, pos, s, e, rule in BINARY) if s < e else None
            out["periods"].setdefault(key, {})[rule] = rec
        if rate is not None:
            r = rate.reindex(close.index).ffill().fillna(0.0) / 100 / YEAR
            cnet, cpos = _net(close, targets[rule], rate=r)
            out["cash"][rule] = _record(close, cnet, cpos, begin, last, False)
    if deep:
        out["take"] = {}
        for rule in RULES:
            base = out["periods"]["all"][rule]
            n3, p3 = _net(close, targets[rule], cost=0.003)
            n2, p2 = _net(close, targets[rule], delay=2)
            out["take"][rule] = {"base": base, "cost": _record(close, n3, p3, begin, last, False),
                                 "delay": _record(close, n2, p2, begin, last, False)}
        out["roll"] = _rolling(close, nets, begin)
    # today: next-day states and the closes that would flip the two on/off rules (as in signals._asset)
    c = close.values
    now = {rule: float(targets[rule].iloc[-1]) for rule in RULES}
    long55 = now["b5520"] >= 0.5
    out["now"] = now
    out["flip55"] = float(c[-19:].min() if long55 else c[-54:].max())
    out["flip50"] = float(c[-49:].sum() / 49)
    return out


def _compute(data: dict) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    rate = data.get(RATE)
    markets = {}
    for symbol, _, _, usd, _ in MARKETS:
        if symbol in data:
            try:
                markets[symbol] = _market(symbol, data[symbol], rate if usd else None, symbol in DEEP)
            except RuntimeError:
                continue
    tr = None
    if TR_CHECK in data and "^GSPC" in markets:  # same rules on the S&P 500 with dividends, over the same dates
        trm = _market(TR_CHECK, data[TR_CHECK], None, False)
        px = _market("^GSPC", data["^GSPC"][data["^GSPC"].index >= data[TR_CHECK].index[0]], None, False)
        tr = {"tr": trm, "px": px}

    def board(period: str, rule: str, cash: bool = False) -> dict:
        used, ratio, deep, money, fh, fr = [], [], [], [], [], []
        for k, m in markets.items():
            rec = (m["cash"].get(rule) if cash else m["periods"][period][rule])
            if rec is None:
                continue
            used.append(k)
            ratio += [k] if rec["rule"]["ratio"] > rec["hold"]["ratio"] else []
            deep += [k] if rec["rule"]["deep"] > rec["hold"]["deep"] else []
            money += [k] if rec["rule"]["growth"] > rec["hold"]["growth"] else []
            if rec["fall_hold"] is not None:
                fh.append(rec["fall_hold"])
                fr.append(rec["fall_rule"])
        return {"n": len(used), "used": used, "ratio": ratio, "deep": deep, "money": money,
                "fall_hold": float(np.mean(fh)) if fh else None, "fall_rule": float(np.mean(fr)) if fr else None,
                "falls_n": len(fh)}

    boards = {rule: {p[0]: board(p[0], rule) for p in PERIODS} for rule in RULES}
    for rule in RULES:
        boards[rule]["cash"] = board("all", rule, cash=True)
    return {"markets": markets, "boards": boards, "tr": tr}


# ---------- the study ----------

def study() -> dict:
    data, info, problems = _load()
    if not any(m[0] in data for m in MARKETS):
        raise RuntimeError("所有市場都取不到資料")
    r = _compute(data)
    markets, boards = r["markets"], r["boards"]
    keys = [m[0] for m in MARKETS if m[0] in markets]
    n_all = len(keys)
    asof = max(m["last"] for m in markets.values())
    stocks = [k for k in keys if k in [m[0] for m in MARKETS if m[2] == "stock"]]

    # ----- table 1: each market, whole period -----
    def cell(rec: dict, rule: str | None) -> str:
        if rec is None:
            return "—"
        x = rec["hold"] if rule is None else rec["rule"]
        text = f"{_pct(x['ret'])}｜{_pct(x['deep'])}｜{_num(x['ratio'])}"
        if rule is None:
            fall = f"｜{rec['falls']} 次大跌平均 {_pct(rec['fall_hold'])}" if rec["falls"] else "｜沒有大跌"
            return text + fall
        text += f"｜{rec['years']} 年贏 {rec['beat']} 年"
        if rec["falls"]:
            text += f"｜大跌時 {_pct(rec['fall_rule'])}"
        return text

    def trades(rec: dict) -> str:
        if rec is None or not rec.get("trades"):
            return "—"
        return (f"{rec['trades']} 筆贏 {rec['wins']} 筆｜平均賺 {_pct(rec['avg_win'])}、賠 {_pct(rec['avg_loss'])}")

    rows1 = []
    for k in keys:
        m = markets[k]
        p = m["periods"]["all"]
        rows1.append([f"{NAME[k]}｜{_d(m['begin'])[:4]} 起", cell(p["b5520"], None)]
                     + [cell(p[rule], rule) for rule in RULES] + [trades(p["b5520"]), trades(p["ma50"])])

    # ----- table 2: scoreboard -----
    def frac(b: dict, key: str) -> str:
        return f"{b['n']} 個裡 {len(b[key])} 個" if b["n"] else "—"

    rows2 = []
    for rule in RULES:
        for key, label, _, _ in PERIODS:
            b = boards[rule][key]
            c = boards[rule]["cash"]
            cash = (f"{frac(c, 'ratio')}｜{len(c['deep'])} 個｜{len(c['money'])} 個" if key == "all" and c["n"] else "—")
            fall = (f"{_pct(b['fall_rule'])}（抱著 {_pct(b['fall_hold'])}）" if b["falls_n"] else "—")
            rows2.append([SHORT[rule], label, frac(b, "ratio"), frac(b, "deep"), frac(b, "money"), fall, cash])

    # ----- table 3: take apart and deep history -----
    rows3 = []

    def two(rec: dict | None) -> str:
        return "—" if rec is None else f"{_pct(rec['rule']['ret'])}｜{_pct(rec['rule']['deep'])}"

    tr = r["tr"]
    for k in DEEP:
        if k not in markets:
            continue
        m = markets[k]
        roll = m["roll"]
        h = m["periods"]["all"]["b5520"]["hold"]
        tr_hold = ""
        if k == "^GSPC" and tr is not None:
            tr_hold = (f"；含股息版 {_d(tr['tr']['begin'])[:4]} 起抱著 {_pct(tr['tr']['periods']['all']['b5520']['hold']['ret'])}"
                       f"（不含息 {_pct(tr['px']['periods']['all']['b5520']['hold']['ret'])}）")
        rows3.append([_short(k), f"一直抱著（{_d(m['begin'])[:4]} 起）", f"{_pct(h['ret'])}｜{_pct(h['deep'])}{tr_hold}",
                      "—", "—", "—", "—", "—"])
        for rule in RULES:
            t = m["take"][rule]
            tr_cell = "—"
            if k == "^GSPC" and tr is not None:
                a, b = tr["tr"]["periods"]["all"][rule], tr["px"]["periods"]["all"][rule]
                tr_cell = (f"含息 {_pct(a['rule']['ret'])}（抱著 {_pct(a['hold']['ret'])}）；"
                           f"不含息 {_pct(b['rule']['ret'])}（抱著 {_pct(b['hold']['ret'])}）")
            rows3.append([_short(k), SHORT[rule], two(t["base"]), two(t["cost"]), two(t["delay"]),
                          (f"{_num(roll[rule]['money'], 0)}%" if roll["n"] else "—"),
                          (f"{_num(roll[rule]['deep'], 0)}%" if roll["n"] else "—"), tr_cell])

    # ----- table 4: today -----
    def state(v: float) -> str:
        return "持有" if v >= 0.5 else "空手"

    rows4 = []
    for k in keys:
        m = markets[k]
        last = float(m["close"].iloc[-1])
        n55, n50 = m["now"]["b5520"], m["now"]["ma50"]
        f55 = (f"收盤 ≤ {_price(m['flip55'])} 轉空手" if n55 >= 0.5 else f"收盤 ≥ {_price(m['flip55'])} 轉持有")
        f50 = (f"收盤 < {_price(m['flip50'])} 轉空手" if n50 >= 0.5 else f"收盤 > {_price(m['flip50'])} 轉持有")
        rows4.append([NAME[k], f"{_price(last)}（{_d(m['last'])}）", state(n55), f55, state(n50), f50,
                      f"{m['now']['blend'] * 100:.0f}%", f"{m['now']['vol40'] * 100:.0f}%"])

    periods_text = "、".join(f"{lab}" for _, lab, _, _ in PERIODS[1:])
    tables = [
        {"caption": ("每個市場整段期間的結果（每格是「每年報酬｜最深曾經虧多少｜報酬÷波動｜幾年裡幾年贏過抱著｜大跌期間這條規則的平均漲跌」）。"
                     "第二欄是同一段日子一直抱著，最後是它經歷的大跌（從高點跌 20% 以上、之後從低點反彈 20% 才算結束）次數和平均跌幅。"
                     "規則都是收盤決定、下一個交易日才持有，每次買賣扣成交金額的 0.10%，空手的錢沒有利息；一直抱著不扣成本。"
                     "最後兩欄是兩條全進全出規則的已結束交易：幾筆裡幾筆賺錢、平均賺多少、平均賠多少。"
                     "各市場起算年不同（規則需要約 200 個交易日暖機），彼此的數字不能直接互比。"),
         "columns": ["市場｜起算", "一直抱著", "55 日新高進、20 日新低出", "50 日均線", "17 條混合", "混合＋波動調整",
                     "55／20 的交易", "50 日均線的交易"],
         "rows": rows1[:16]},
        {"caption": (f"記分板：{n_all} 個市場裡，規則比同一段日子一直抱著「報酬÷波動較高」「最深虧損較淺」「最後的錢較多」的各有幾個。"
                     f"分段（{periods_text}）只算那一段至少有一年資料的市場，每段都從段首的實際部位接著算、不重新開始。"
                     "「大跌時」是各市場在那段期間每次大跌（−20% 以上）中規則的平均漲跌，括號裡是抱著的平均跌幅。"
                     "最後一欄是讓空手的錢賺美國 13 週國庫券利息（只套用在美元計價的市場，其他市場仍算 0%），"
                     "依序是報酬÷波動較高、最深虧損較淺、最後的錢較多的個數。"),
         "columns": ["規則", "期間", "報酬÷波動較高", "最深虧損較淺", "最後的錢較多", "大跌時平均（抱著）", "空手有利息時（整段）"],
         "rows": rows2[:16]},
        {"caption": ("拆開來看：美國標普 500 和台灣加權指數。每格是「每年報酬｜最深曾經虧多少」。「成本 0.30%」是每次買賣扣三倍成本；"
                     "「晚一天」是訊號出來後第二個交易日才照做。滾動 10 年：從 "
                     + "、".join(f"{_short(k)} {_d(markets[k]['roll']['first'])[:7]}" for k in DEEP
                                 if k in markets and markets[k]['roll']['first'] is not None)
                     + " 起每個月初開始的每一段 10 年（" + "、".join(f"{_short(k)} {markets[k]['roll']['n']} 段" for k in DEEP if k in markets)
                     + "），規則最後的錢比抱著多、最深虧損比抱著淺的段數比例；這些段大量重疊，不是互相獨立的樣本。"
                     "最後一欄是標普 500 含股息再投入的指數（1988 年起）和同期不含息指數各自的規則與抱著的每年報酬。"),
         "columns": ["市場", "做法", "原本（0.10%、隔天做）", "成本 0.30%", "晚一天才做", "滾動 10 年：錢較多",
                     "滾動 10 年：虧損較淺", "含股息的標普 500（1988 起）"],
         "rows": rows3[:16]},
        {"caption": ("今天的狀態：用每個市場最後一根收完的日線算，規則在下一個交易日是持有還是空手，以及收盤要到哪裡狀態才會翻轉"
                     "（55／20：空手時要創 55 日收盤新高、持有時要創 20 日收盤新低；50 日均線：收盤和包含那天在內的 50 日平均比）。"
                     "混合與混合＋波動是規則的持有比例。價格單位是各市場自己的報價單位，有股息調整的標的是調整後價格。這只是規則的狀態，不是建議。"),
         "columns": ["市場", "最後收盤（日期）", "55／20", "55／20 翻轉價", "50 日均線", "50 日均線翻轉價", "混合", "混合＋波動"],
         "rows": rows4[:16]},
    ]

    # ----- now -----
    long55 = [k for k in keys if markets[k]["now"]["b5520"] >= 0.5]
    long50 = [k for k in keys if markets[k]["now"]["ma50"] >= 0.5]
    avg_blend = float(np.mean([markets[k]["now"]["blend"] for k in keys])) * 100
    now = [{"label": f"「55 日新高進、20 日新低出」下一個交易日是持有的市場", "value": f"{n_all} 個裡 {len(long55)} 個"},
           {"label": "「收盤高於 50 日均線」下一個交易日是持有的市場", "value": f"{n_all} 個裡 {len(long50)} 個"},
           {"label": f"「17 條混合」在 {n_all} 個市場的平均持有比例", "value": f"{avg_blend:.0f}%"}]
    for k in ("^TWII", "^GSPC"):
        if k in markets:
            m = markets[k]
            now.append({"label": f"{_short(k)}（{_d(m['last'])} 收盤 {_price(float(m['close'].iloc[-1]))}）",
                        "value": (f"55／20：{state(m['now']['b5520'])}；50 日均線：{state(m['now']['ma50'])}；"
                                  f"混合 {m['now']['blend'] * 100:.0f}%；混合＋波動 {m['now']['vol40'] * 100:.0f}%")})

    # ----- verdict -----
    ba = {rule: boards[rule]["all"] for rule in RULES}
    deep_lo = min(len(ba[x]["deep"]) for x in RULES)
    deep_hi = max(len(ba[x]["deep"]) for x in RULES)
    money_lo = min(len(ba[x]["money"]) for x in RULES)
    money_hi = max(len(ba[x]["money"]) for x in RULES)
    rng = lambda lo, hi: f"{lo}" if lo == hi else f"{lo} 到 {hi}"  # noqa: E731
    stock_lose = {rule: sum(1 for k in stocks if markets[k]["periods"]["all"][rule]["rule"]["ret"]
                            < markets[k]["periods"]["all"][rule]["hold"]["ret"]) for rule in RULES}
    tw = [k for k in ("^TWII", "0050.TW", "2330.TW") if k in markets]
    tw_bits = []
    for k in tw:
        p = markets[k]["periods"]["all"]
        tw_bits.append(f"{_short(k)}抱著 {_pct(p['b5520']['hold']['ret'])}、最深 {_pct(p['b5520']['hold']['deep'])}，"
                       f"55／20 {_pct(p['b5520']['rule']['ret'])}、{_pct(p['b5520']['rule']['deep'])}")
    verdict = (
        f"同樣四條規則一字不改用在 {n_all} 個市場（美股、台股、日港股、新興市場、黃金、美債、原油、美元），"
        f"整段下來最深虧損變淺的有 {rng(deep_lo, deep_hi)} 個、最後的錢比抱著多的只有 {rng(money_lo, money_hi)} 個。"
        f"股票類 {len(stocks)} 個裡，規則每年報酬輸給抱著的：55／20 有 {stock_lose['b5520']} 個、50 日均線 {stock_lose['ma50']} 個、"
        f"混合 {stock_lose['blend']} 個，而且價格指數沒算股息，抱著的實際報酬還更高。"
        + ("台股：" + "；".join(tw_bits) + "。" if tw_bits else "")
    )

    # ----- notes -----
    hist = "；".join(f"{NAME[k]} {_d(data[k].index[0])} 起（{len(data[k]):,} 天）" for k in keys)
    notes = [
        "四條規則就是「順勢訊號」頁上那四條，程式直接沿用同一段規則計算，沒有為任何市場調整參數：55 日新高進、20 日新低出；"
        "收盤高於 50 日均線才持有；17 條簡單順勢規則混合（看多幾條就持有幾成）；混合再乘上「40% ÷ 最近 30 天年化波動」（最多乘 1）。"
        "每條規則都用當天收盤為止的價格，決定下一個交易日的部位；一年用 252 個交易日換算。",
        "取得的歷史（Yahoo Finance 日線）：" + hist + "。",
        "價格指數不含股息：標普 500、那斯達克、道瓊、台灣加權、日經、恆生都是價格指數，一直抱著實際上還會多領每年約 1.5～4% 的股息"
        "（台股殖利率常在 3% 以上），規則只在持有的日子領得到。所以表上「抱著」的數字被低估，規則相對抱著的差距看起來比實際好。"
        "0050、台積電、EEM、GLD、TLT 用的是 Yahoo 還原股息後的收盤價，比較接近真實的總報酬。",
        "黃金期貨（GC=F）和原油期貨（CL=F）是 Yahoo 接續近月合約的價格，換月時會有跳動，不等於真正持有期貨的報酬（少了換月損益和保證金利息）；"
        "2020-04-20 原油期貨收在負價，那一天被拿掉，前後兩天的漲跌合在一起算。美元指數是指數，不能直接買。",
        "各市場的交易時間和假日不同：只有交易日才有價格，週末和假日的變動要等下一個收盤才反映。亞洲市場的收盤比美國早；"
        "台股和日股的一天在台灣時間下午就收完，美國市場要到台灣隔天清晨。這裡假設剛好用收盤價成交。",
        f"成本只用一個粗略的數字：每次買賣扣成交金額的 {COST * 100:.2f}%（第三張表有 0.30%）。台股實際還有證交稅（股票 0.3%、ETF 0.1%，只在賣出時收），"
        "早年美股的手續費和價差也比現在高很多，所以早期的規則數字偏好看。沒有算稅。",
        "空手有利息的版本：用 Yahoo 的 13 週美國國庫券殖利率（^IRX，1960 年起，年利率 ÷ 252 當作每個交易日的利息），"
        "只套用在美元計價的市場；1960 年以前和非美元市場的空手資金仍算 0%。台股、日股、港股空手時應該用當地的短期利率，這裡沒有算。"
        "（robust_trend 用的 FRED／財政部利率來源，FRED 連不上時只有 2018 年起，不夠長，所以這裡改用 ^IRX。）",
        "這些市場彼此相關：三個美國股價指數高度重疊，GC=F 和 GLD 都是黃金，0050 裡約一半是台積電。所以「幾個裡幾個」不是互相獨立的次數，"
        "把證據說得比實際強。另外，把一部分時間換成現金本來就會讓波動和最深虧損變小，「虧損變淺」幾乎是這類規則必然的結果。",
        "選這些市場是因為現在知道它們重要、還存在，美國和台灣股市事後看是很成功的市場；長期上漲的市場對「一直抱著」特別有利。",
        "過去的結果不保證未來會一樣。這裡只是整理同一套規則過去在這些市場做了什麼，不是買賣建議。",
    ] + problems
    if r["tr"] is None:
        notes.append("標普 500 含股息指數（^SP500TR）這次取不到，第三張表的最後一欄是空的。")

    sources = [{"name": f"Yahoo Finance {k} 每日價格", "url": f"https://finance.yahoo.com/quote/{k}/history"} for k in keys]
    if r["tr"] is not None:
        sources.append({"name": "Yahoo Finance ^SP500TR（標普 500 含股息）", "url": "https://finance.yahoo.com/quote/%5ESP500TR/history"})
    if RATE in data:
        sources.append({"name": "Yahoo Finance ^IRX（13 週美國國庫券殖利率）", "url": "https://finance.yahoo.com/quote/%5EIRX/history"})
    sources += [
        {"name": "Moskowitz, Ooi & Pedersen (2012), Time Series Momentum, Journal of Financial Economics",
         "url": "https://doi.org/10.1016/j.jfineco.2011.11.003"},
        {"name": "Hurst, Ooi & Pedersen (2017), A Century of Evidence on Trend-Following Investing, AQR",
         "url": "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing"},
        {"name": "Faber (2007), A Quantitative Approach to Tactical Asset Allocation", "url": "https://doi.org/10.2139/ssrn.962461"},
    ]

    return {
        "id": "trend_world",
        "title": "比特幣的順勢規則用在股票、黃金、債券、原油",
        "question": "同一套順勢規則一字不改，用在美股、台股、黃金、債券和原油等傳統市場、拉長到幾十年，還有沒有幫助？",
        "asof": _d(asof),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
