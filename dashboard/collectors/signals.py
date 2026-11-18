"""The trend rules that held up in the dashboard's studies, turned into daily signals with their track record.

Written to cache/_signals.json and shown on the "順勢訊號" tab. For each asset and each rule:
- what the rule wants to hold for the next day, since when, and the close that would change that;
- its record from the start of the comparison window: yearly return, deepest loss, years beating holding,
  trades won (for on/off rules), and how much of each big fall (30% or more) it took.

These are descriptions of what fixed rules did and what state they are in now. They are not forecasts and not
advice. The rules are the ones the studies under dashboard/studies2 tested; the robustness notes below summarise
those studies (clock_robustness, robust_trend, loss_control, trend_assets) and must be revisited if they change.

Sources (public, no key): blockchain.info and Binance daily closes, Yahoo Finance for MSTR.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

PATH = common.CACHE / "_signals.json"
COST = 0.001
FALL = 0.30  # a "big fall": the close drops this much from a peak before rebounding as much from the low
CHART_DAYS = 3 * 365

ASSETS = (  # key, name, loader, comparison start, days per year
    ("BTC", "比特幣", "btc", "2018-01-01", 365),
    ("ETH", "以太幣", "ETHUSDT", None, 365),
    ("SOL", "索拉納", "SOLUSDT", None, 365),
    ("MSTR", "MSTR（Strategy 公司股票）", "MSTR", "2020-08-11", 252),
)

RULES = (
    {"id": "blend", "name": "17 條順勢規則混合", "short": "混合",
     "how": "同時看 17 條簡單的順勢規則（8 條均線、4 條新高新低、5 條過去一段時間是漲是跌），有幾成說「持有」就持有幾成。每天收盤看一次。",
     "edge": "不用挑參數，所以不靠運氣；換收盤時間、晚一兩天做、偶爾漏做，結果都變化不大。研究裡最穩的一種。",
     "weak": "分批進出，一年調整約 20 次；大漲時只拿到一部分；2024 年以後報酬優勢很小。",
     "robust": "24 個收盤時間全部贏過一直抱著；1,000 個照一般人方式做的模擬裡只有 1 個輸。"},
    {"id": "vol40", "name": "混合，再依波動調整持有多少", "short": "混合＋波動",
     "how": "先照「17 條混合」決定幾成，再在市場很激烈時少拿一點：持有比例乘上「40% ÷ 最近 30 天的年化波動」（最多乘 1）。",
     "edge": "研究裡虧損最淺的一種，最深虧損約一直抱著的一半。",
     "weak": "拿得更少，所以報酬比單純混合低；大漲時拿到的更少。",
     "robust": "24 個收盤時間全部贏過一直抱著；最深虧損在各種測法下都最淺。"},
    {"id": "b5520", "name": "55 日新高進、20 日新低出", "short": "55／20",
     "how": "收盤是最近 55 天最高就全部持有；持有中收盤是最近 20 天最低就全部賣出。其他日子不動。",
     "edge": "一年只進出約 7 次；同樣的虧損控制下留下的報酬最多。",
     "weak": "買回時常常比賣出時貴；六成左右的交易是小賠，靠少數幾筆大賺。",
     "robust": "24 個收盤時間全部贏過一直抱著；照一般人方式做，1,000 個模擬裡約 144 個輸。"},
    {"id": "ma50", "name": "收盤高於 50 日均線才持有", "short": "50 日均線",
     "how": "收盤價高於最近 50 天收盤的平均就持有，低於就空手。",
     "edge": "歷史報酬最高的一條。",
     "weak": "進出次數多、對手續費很敏感；最深虧損較深（約 −60%）；這是事後看才特別好的一條。",
     "robust": "24 個收盤時間全部贏過一直抱著；照一般人方式做，1,000 個模擬裡約 118 個輸。"},
    {"id": "c2010", "name": "20 日新高進、10 日新低出（參考）", "short": "20／10",
     "how": "收盤是最近 20 天最高就持有；持有中收盤是最近 10 天最低就賣出。",
     "edge": "事先挑參數時九年都挑到它；虧損比一直抱著淺。",
     "weak": "報酬優勢看運氣：換一個收盤時間，24 個裡只有 12 個贏；照一般人方式做會輸給一直抱著。只當參考。",
     "robust": "不穩：報酬受收盤時間影響很大，只有「虧得比較淺」站得住。"},
)
RULE_IDS = tuple(r["id"] for r in RULES)


def _f(v, digits: int = 1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _load(loader: str) -> pd.Series:
    if loader == "btc":
        return te._daily_close()
    if loader.endswith("USDT"):
        return ta._crypto_close(loader)
    return ta._share_close(loader)


def _breakout(close: pd.Series, n: int, m: int) -> pd.Series:
    high = close.rolling(n, min_periods=n).max()
    low = close.rolling(m, min_periods=m).min()
    state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
    return state.where(high.notna()).ffill().fillna(0.0).where(high.notna())


def _targets(close: pd.Series, year: int) -> pd.DataFrame:
    """What each rule wants to hold, decided at each close (1 = all in, 0 = cash, fractions for the blends)."""
    signal, _ = te._signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    vol = close.pct_change().rolling(30, min_periods=30).std(ddof=1) * math.sqrt(year)
    scale = (0.40 / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
    mean50 = close.rolling(50, min_periods=50).mean()
    return pd.DataFrame({"blend": share, "vol40": share * scale, "b5520": _breakout(close, 55, 20),
                         "ma50": (close > mean50).astype(float).where(mean50.notna()),
                         "c2010": signal["C20"]}), signal


def _flip_prices(close: pd.Series, signal: pd.DataFrame) -> dict:
    """For each simple rule: the next close that would switch it, given its current state."""
    c = close.values
    out = {}
    for col in signal.columns:
        state = signal[col].iloc[-1]
        if not np.isfinite(state):
            continue
        kind, n = col[0], int(col[1:])
        if kind == "A":  # close > mean of the last n closes including the new one  <=>  close > sum(last n-1) / (n-1)
            level = c[-(n - 1):].sum() / (n - 1)
        elif kind == "C":
            level = c[-(n - 1):].max() if state < 0.5 else c[-(n // 2 - 1):].min()
        else:  # close above the close n days before the new day
            level = c[-n]
        out[col] = (float(level), bool(state >= 0.5))
    return out


def _falls(close: pd.Series) -> list[tuple]:
    """(peak day, low day) of every fall of FALL or more, ended by a rebound of FALL from the low."""
    out, peak_i, low_i, falling = [], 0, 0, False
    v = close.values
    for i in range(1, len(v)):
        if not falling:
            if v[i] > v[peak_i]:
                peak_i = i
            elif v[i] <= v[peak_i] * (1 - FALL):
                falling, low_i = True, i
        else:
            if v[i] < v[low_i]:
                low_i = i
            elif v[i] >= v[low_i] * (1 + FALL):
                out.append((close.index[peak_i], close.index[low_i]))
                falling, peak_i = False, i
    if falling:
        out.append((close.index[peak_i], close.index[low_i]))
    return out


def _record(close: pd.Series, target: pd.Series, start: pd.Timestamp, year: int, binary: bool) -> dict:
    change = close.pct_change().fillna(0.0)
    position = target.shift(1).fillna(0.0)
    net, trade = te._run(position.values, change.values, COST)
    net = pd.Series(net, index=close.index)[close.index >= start]
    hold = change[close.index >= start]
    pos = position[close.index >= start]

    def stats(r: pd.Series) -> tuple:
        growth = float(np.prod(1 + r.values))
        ret = (growth ** (year / len(r)) - 1) * 100 if growth > 0 else -100.0
        eq = np.concatenate([[1.0], np.cumprod(1 + r.values)])
        return ret, float((eq / np.maximum.accumulate(eq) - 1).min() * 100)

    ret, deep = stats(net)
    hret, hdeep = stats(hold)
    years_rule = {y: float(np.prod(1 + g.values) - 1) for y, g in net.groupby(net.index.year)}
    years_hold = {y: float(np.prod(1 + g.values) - 1) for y, g in hold.groupby(hold.index.year)}
    beat = sum(years_rule[y] > years_hold[y] for y in years_rule)
    falls, taken = [], []
    for peak, low in _falls(close[close.index >= start]):
        span = (net.index > peak) & (net.index <= low)
        if span.sum() < 2:
            continue
        falls.append(float(close[low] / close[peak] - 1) * 100)
        taken.append(float(np.prod(1 + net[span].values) - 1) * 100)
    rec = {"start": start.strftime("%Y-%m-%d"), "ret": _f(ret), "deep": _f(deep), "hold_ret": _f(hret),
           "hold_deep": _f(hdeep), "years": len(years_rule), "beat": int(beat), "share": _f(pos.mean() * 100, 0),
           "falls": len(falls), "fall_avg": _f(np.mean(falls)) if falls else None,
           "fall_rule": _f(np.mean(taken)) if taken else None}
    if binary:
        p = pos.values
        wins = trades = 0
        i = 0
        while i < len(p):
            if p[i] > 0.5 and (i == 0 or p[i - 1] < 0.5):
                j = i
                while j + 1 < len(p) and p[j + 1] > 0.5:
                    j += 1
                if j + 1 < len(p):  # closed trades only
                    trades += 1
                    wins += float(np.prod(1 + net.values[i:j + 1])) > 1
                i = j + 1
            else:
                i += 1
        rec["trades"], rec["wins"] = int(trades), int(wins)
        rec["trades_per_year"] = _f(trades / (len(p) / year), 1)
    else:
        rec["turnover"] = _f(trade[-len(net):].sum() / (len(net) / year), 1)
    return rec


def _since(target: pd.Series) -> str | None:
    """The first day of the current run of the same state (all in / cash / partly)."""
    t = target.dropna()
    if t.empty:
        return None
    state = np.where(t.values >= 0.999, 2, np.where(t.values <= 0.001, 0, 1))
    i = len(state) - 1
    while i > 0 and state[i - 1] == state[-1]:
        i -= 1
    return t.index[i].strftime("%Y-%m-%d")


def _asset(key: str, name: str, loader: str, start: str | None, year: int) -> dict:
    close = _load(loader)
    targets, signal = _targets(close, year)
    first = targets["blend"].first_valid_index()
    begin = max(pd.Timestamp(start) if start else first, first)
    flips = _flip_prices(close, signal)
    last = float(close.iloc[-1])
    rules = {}
    for rule in RULE_IDS:
        t = targets[rule]
        now = _f(t.iloc[-1] * 100, 0)
        item = {"share": now, "since": _since(t), "record": _record(close, t, begin, year, rule in ("b5520", "ma50", "c2010"))}
        if rule == "ma50":
            level = float(close.values[-49:].sum() / 49)
            item["flip"] = {"level": level, "dir": "below" if t.iloc[-1] >= 0.5 else "above",
                            "pct": _f((level / last - 1) * 100)}
        elif rule in ("b5520", "c2010"):
            n, m = (55, 20) if rule == "b5520" else (20, 10)
            long = t.iloc[-1] >= 0.5
            level = float(close.values[-(m - 1):].min() if long else close.values[-(n - 1):].max())
            item["flip"] = {"level": level, "dir": "below" if long else "above", "pct": _f((level / last - 1) * 100)}
        rules[rule] = item
    long_rules = sum(1 for v in flips.values() if v[1])
    near = sorted(((abs(lv / last - 1), col, lv, is_long) for col, (lv, is_long) in flips.items()))[:3]
    hist = close[close.index >= close.index[-1] - pd.Timedelta(days=CHART_DAYS)]
    blend = targets["blend"].reindex(hist.index)
    return {"key": key, "name": name, "asof": close.index[-1].strftime("%Y-%m-%d"), "price": _f(last, 4),
            "rules": rules, "blend_long": long_rules, "blend_total": len(flips),
            "nearest": [{"rule": col, "level": _f(lv, 4), "pct": _f((lv / last - 1) * 100), "long": is_long}
                        for _, col, lv, is_long in near],
            "history": {"price": [[d.strftime("%Y-%m-%d"), _f(v, 4)] for d, v in hist.items()],
                        "blend": [[d.strftime("%Y-%m-%d"), _f(v * 100, 0)] for d, v in blend.items() if np.isfinite(v)]}}


def collect() -> list[dict]:
    try:
        previous = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        previous = {}
    old = {a.get("key"): a for a in previous.get("assets", []) if isinstance(a, dict)}
    payload = {"collected_utc": common.now_utc(), "rules": list(RULES), "assets": [], "errors": {}}
    for key, name, loader, start, year in ASSETS:
        try:
            item = _asset(key, name, loader, start, year)
            json.dumps(item, allow_nan=False)
            payload["assets"].append(item)
        except Exception as exc:  # noqa: BLE001 - one asset failing keeps its previous result
            payload["errors"][key] = f"{type(exc).__name__}: {exc}"[:200]
            if key in old:
                payload["assets"].append(old[key])
            print(f"  訊號 {key} 這次沒有更新：{payload['errors'][key]}")
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for a in data["assets"]:
        print(a["key"], a["price"], a["blend_long"], "/", a["blend_total"], a["nearest"])
        for r, v in a["rules"].items():
            print("  ", r, v["share"], v["since"], v.get("flip"), json.dumps(v["record"], ensure_ascii=False))
    print("errors", data["errors"])
