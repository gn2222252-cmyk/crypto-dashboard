"""Historical study: does a steep bitcoin option skew come before bigger falls?

Skew = implied volatility 10% below the price minus implied volatility 10% above it (volatility points), from an
amount-weighted straight-line fit over one day's Deribit option trades with 14-60 days to expiry. The history is
sampled every 4th day since 2021-03-01 and kept in cache/_store/skew_history.json; each run adds the missing days
(at most MAX_NEW) and recomputes the most recent complete day. Prices are Binance daily candles.

Sources (public, no key): Deribit trade history, Binance spot klines. A source that fails is skipped, never
worked around.
"""
from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta, timezone

from dashboard import common
from dashboard.collectors import price as px

ID = "skew_history"
DERIBIT = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
GRID_START = date(2021, 3, 1)
GRID_STEP = 4  # days between samples
MAX_NEW = 25  # new days fetched per run
BUDGET = 70  # seconds a run may spend on fetching option trades
PAGES = 3  # 1000 trades each
HORIZONS = (14, 30, 60)
MOVE = 11  # a "big" move, percent
DROP_DAYS, DROP = 14, -12  # "after a 14-day drop of 12%+"
STORE_FILE = common.STORE / f"{ID}.json"
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
GROUP_NAMES = ("1 最不怕跌", "2", "3 中間", "4", "5 最怕跌")


# ---------- skew history ----------

def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(raw) -> dict:
    """{date: {"skew": .., "atm": ..}} with finite numbers only."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for day, row in raw.items():
        if isinstance(row, dict) and _finite(row.get("skew")):
            atm = row.get("atm")
            out[str(day)[:10]] = {"skew": float(row["skew"]), "atm": float(atm) if _finite(atm) else None}
    return out


def _load() -> dict:
    for path in (STORE_FILE,):
        try:
            if path.exists():
                return _clean(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a damaged file starts over instead of breaking the study
            continue
    return {}


def _save(store: dict) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        STORE_FILE.write_text(json.dumps(dict(sorted(store.items())), ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _on_grid(day: str) -> bool:
    return (date.fromisoformat(day) - GRID_START).days % GRID_STEP == 0


def _expiry(name: str):
    """'BTC-27DEC24-60000-P' -> (expiry time in ms, strike)."""
    parts = name.split("-")
    code = parts[1]
    when = datetime(2000 + int(code[-2:]), MONTHS[code[-5:-2]], int(code[:-5]), 8, tzinfo=timezone.utc)
    return when.timestamp() * 1000, float(parts[2])


def _day_skew(day: date):
    """One day's skew and at-the-money volatility, or None when there are too few usable trades."""
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    end, cursor, points = start + 86400000, start, []
    for _ in range(PAGES):
        reply = common.get_json(DERIBIT, {"currency": "BTC", "kind": "option", "start_timestamp": cursor,
                                          "end_timestamp": end, "count": 1000, "sorting": "asc"}, timeout=40)
        result = (reply or {}).get("result") or {}
        trades = result.get("trades") or []
        if not trades:
            break
        for trade in trades:
            try:
                expiry, strike = _expiry(trade["instrument_name"])
                iv, index = float(trade["iv"]), float(trade["index_price"])
                left = (expiry - trade["timestamp"]) / 86400000
                money = math.log(strike / index)
            except Exception:  # noqa: BLE001 - an odd instrument name or a missing field
                continue
            if 14 <= left <= 60 and abs(money) <= 0.25 and 5 < iv < 300:
                points.append((money, iv, float(trade.get("amount") or 1)))
        if not result.get("has_more"):
            break
        cursor = trades[-1]["timestamp"] + 1
    if len(points) < 15 or min(p[0] for p in points) > -0.04 or max(p[0] for p in points) < 0.04:
        return None
    weight = sum(p[2] for p in points)
    mx = sum(p[0] * p[2] for p in points) / weight
    my = sum(p[1] * p[2] for p in points) / weight
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    if not sxx:
        return None
    slope = sxy / sxx
    skew, atm = -slope * 0.2, my - slope * mx
    if not (_finite(skew) and _finite(atm)):
        return None
    return {"skew": round(skew, 2), "atm": round(atm, 1)}


def _update() -> dict:
    """Load the store, add the missing grid days (oldest first, capped) and refresh the latest complete day."""
    store = _load()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    grid = [d for d in store if _on_grid(d)]
    day = date.fromisoformat(max(grid)) + timedelta(days=GRID_STEP) if grid else GRID_START
    todo = [yesterday]  # always first, so the newest reading is fresh even when the run is cut short
    while day <= yesterday and len(todo) < MAX_NEW:
        if day != yesterday:
            todo.append(day)
        day += timedelta(days=GRID_STEP)
    began = time.time()
    for day in todo:
        if time.time() - began > BUDGET:
            break
        try:
            row = _day_skew(day)
        except Exception:  # noqa: BLE001 - a day that fails is skipped quietly
            row = None
        if row:
            store[day.isoformat()] = row
    _save(store)
    return store


# ---------- prices and samples ----------

def _prices() -> dict:
    """{date: (high, low, close)} of closed Binance daily candles since 2021."""
    rows, start = [], int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for _ in range(6):
        page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000,
                                              "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + 86400000
        if len(page) < 1000:
            break
    now_ms = time.time() * 1000
    return {datetime.fromtimestamp(r[0] / 1000, timezone.utc).strftime("%Y-%m-%d"):
            (float(r[2]), float(r[3]), float(r[4])) for r in rows if r[6] < now_ms}


def _samples(store: dict, prices: dict) -> list[dict]:
    days = sorted(prices)
    pos = {d: i for i, d in enumerate(days)}
    out = []
    for day in sorted(store):
        if not _on_grid(day) or day not in pos:
            continue
        i, close = pos[day], prices[day][2]
        row = {"d": day, "skew": store[day]["skew"],
               "past": (close / prices[days[i - DROP_DAYS]][2] - 1) * 100 if i >= DROP_DAYS else None}
        for h in HORIZONS:
            if i + h < len(days):
                window = [prices[days[j]] for j in range(i + 1, i + h + 1)]
                row[f"f{h}"] = (window[-1][2] / close - 1) * 100
                row[f"lo{h}"] = (min(w[1] for w in window) / close - 1) * 100
                row[f"hi{h}"] = (max(w[0] for w in window) / close - 1) * 100
        out.append(row)
    return out


def _quantile(ordered: list, p: float) -> float:
    return ordered[int(p * (len(ordered) - 1))]


def _median(values: list) -> float:
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def _stats(rows: list[dict], h: int):
    """Outcome h days later for a set of samples, or None when there are none."""
    done = [r for r in rows if f"f{h}" in r]
    if not done:
        return None
    n = len(done)
    return {"n": n, "median": _median([r[f"f{h}"] for r in done]),
            "up": sum(r[f"f{h}"] > 0 for r in done) / n * 100,
            "fell": sum(r[f"lo{h}"] <= -MOVE for r in done) / n * 100,
            "rose": sum(r[f"hi{h}"] >= MOVE for r in done) / n * 100}


# ---------- formatting ----------

def _pct(value) -> str:
    return "—" if value is None else f"{value:+.1f}%"


def _per100(value) -> str:
    return "—" if value is None else f"{value:.0f} 次"


def _cells(stat) -> list[str]:
    if not stat:
        return ["0", "—", "—", "—", "—"]
    return [str(stat["n"]), _pct(stat["median"]), _per100(stat["up"]), _per100(stat["fell"]), _per100(stat["rose"])]


OUTCOME_COLUMNS = ["樣本數", "漲跌中位數", "100 次裡收得更高", f"100 次裡曾跌 {MOVE}% 以上", f"100 次裡曾漲 {MOVE}% 以上"]
SOURCES = [{"name": "Deribit 選擇權歷史成交（公開）", "url": "https://history.deribit.com/"},
           {"name": "幣安 BTC/USDT 日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"}]
NOTES = [
    "偏斜每 4 天取樣一次，不是每天都有，中間幾天的變化看不到。",
    "相鄰樣本的「之後 14／30／60 天」互相重疊，真正獨立的次數比樣本數少很多，數字看起來會比實際更可靠。",
    "分組的界線是用同一段歷史算出來的（事後回看），沒有留一段沒看過的資料來驗證。",
    "偏斜是把一天內 14～60 天到期的成交用一條直線粗略估出來的，成交少的日子誤差比較大，不等於交易所的正式報價。",
    f"「曾跌／曾漲 {MOVE}%」是看期間內日線的最低／最高價，同一段期間可能兩個都發生。",
    "歷史只從 2021 年 3 月開始，涵蓋的行情循環不多。",
    "過去的結果不保證未來會一樣；這裡只是整理歷史，不是買賣建議。",
]


def _empty(reason: str) -> dict:
    return {"id": ID, "title": "怕跌的保險變貴，之後真的跌更多嗎？",
            "question": "選擇權市場買「跌的保險」比買「漲的彩券」貴很多時，比特幣之後真的跌得比較多嗎？",
            "asof": common.today_utc(), "verdict": f"這次沒辦法算出結果：{reason}", "now": [], "tables": [],
            "notes": NOTES, "sources": SOURCES}


# ---------- the study ----------

def study() -> dict:
    store = _update()
    if not store:
        return _empty("還沒有任何偏斜歷史資料（來源暫時讀不到）。")
    try:
        prices = _prices()
    except Exception as exc:  # noqa: BLE001
        return _empty(f"價格資料讀不到（{type(exc).__name__}）。")
    rows = _samples(store, prices)
    if len(rows) < 50:
        return _empty(f"偏斜歷史只有 {len(rows)} 個樣本，太少。")

    ordered = sorted(r["skew"] for r in rows)
    edges = [_quantile(ordered, p) for p in (0.2, 0.4, 0.6, 0.8)]
    median = _quantile(ordered, 0.5)
    steep = edges[3]

    def group(value: float) -> int:
        return sum(value >= e for e in edges)

    latest_day = max(store)
    latest = store[latest_day]
    below = sum(s < latest["skew"] for s in ordered) / len(ordered) * 100
    now = [
        {"label": f"最近一天的偏斜（{latest_day}）", "value": f"{latest['skew']:+.1f} 點"},
        {"label": "在歷史裡的位置", "value": f"比過去 100 天裡的 {below:.0f} 天高（第 {group(latest['skew']) + 1} 組，共 5 組）"},
        {"label": "歷史中位數／最怕跌那五分之一的門檻", "value": f"{median:+.1f} 點／{steep:+.1f} 點以上"},
        {"label": "歷史樣本", "value": f"{len(rows)} 天（{rows[0]['d']} 起，每 4 天一筆）"},
    ]
    if latest.get("atm") is not None:
        now.insert(1, {"label": "市場預期的波動（年化）", "value": f"{latest['atm']:.0f}%"})

    # table 1: five equal groups of skew
    groups = [[r for r in rows if group(r["skew"]) == g] for g in range(5)]
    spans = [f"{min(r['skew'] for r in g):+.1f}～{max(r['skew'] for r in g):+.1f}" if g else "—" for g in groups]
    table1 = []
    for h in HORIZONS:
        for g in range(5):
            table1.append([f"{h} 天", f"{GROUP_NAMES[g]}（{spans[g]}）"] + _cells(_stats(groups[g], h)))
    # table 2: after a 14-day drop of 12%+
    dropped = [r for r in rows if r["past"] is not None and r["past"] <= DROP]
    table2 = []
    for h in HORIZONS:
        for label, sel in ((f"偏斜陡（{steep:+.1f} 以上）", [r for r in dropped if r["skew"] >= steep]),
                           ("偏斜不陡", [r for r in dropped if r["skew"] < steep]),
                           ("全部時候（對照）", rows)):
            table2.append([f"{h} 天", label] + _cells(_stats(sel, h)))
    # table 3: by year, skew at or above the median versus below it, 30 days later
    table3, hi_wins, years = [], 0, 0
    for year in sorted({r["d"][:4] for r in rows}):
        of_year = [r for r in rows if r["d"][:4] == year]
        hi = _stats([r for r in of_year if r["skew"] >= median], 30)
        lo = _stats([r for r in of_year if r["skew"] < median], 30)
        if hi and lo and min(hi["n"], lo["n"]) >= 5:
            years += 1
            hi_wins += hi["median"] > lo["median"]
            better = "偏斜高" if hi["median"] > lo["median"] else "偏斜低" if hi["median"] < lo["median"] else "一樣"
        else:
            better = "樣本太少"
        table3.append([year, str(hi["n"]) if hi else "0", _pct(hi["median"]) if hi else "—",
                       str(lo["n"]) if lo else "0", _pct(lo["median"]) if lo else "—", better])

    tables = [
        {"caption": f"依偏斜高低分成五等份，之後的表現（偏斜單位：波動率點數；括號內是該組的偏斜範圍）",
         "columns": ["幾天後", "組別"] + OUTCOME_COLUMNS, "rows": table1},
        {"caption": f"已經在 {DROP_DAYS} 天內跌了 {abs(DROP)}% 以上之後：偏斜陡或不陡，之後的表現",
         "columns": ["幾天後", "情況"] + OUTCOME_COLUMNS, "rows": table2},
        {"caption": f"逐年檢查：偏斜高於／低於歷史中位數（{median:+.1f} 點）時，30 天後的漲跌中位數",
         "columns": ["年份", "偏斜高 樣本數", "偏斜高 30 天後", "偏斜低 樣本數", "偏斜低 30 天後", "哪邊之後較好"],
         "rows": table3},
    ]

    # verdict: numbers first, then the reading they support
    top = _stats(groups[4], 30)
    middle = _stats(groups[1] + groups[2] + groups[3], 30)
    bottom = _stats(groups[0], 30)
    if not (top and middle and bottom):
        verdict = "樣本還不夠分組比較，請看下面的表。"
    else:
        facts = (f"偏斜最陡（最怕跌）的那五分之一，之後 30 天內曾跌 {MOVE}% 以上的是 100 次裡 {top['fell']:.0f} 次，"
                 f"中間三組是 {middle['fell']:.0f} 次，最不怕跌的那組是 {bottom['fell']:.0f} 次；"
                 f"30 天後漲跌中位數分別是 {_pct(top['median'])}、{_pct(middle['median'])}、{_pct(bottom['median'])}。")
        fell_gap = top["fell"] - middle["fell"]
        consistent_up = years >= 4 and hi_wins >= years - 1
        consistent_down = years >= 4 and hi_wins <= 1
        if fell_gap >= 8 and consistent_down:
            reading = (f"在這段歷史裡，偏斜陡之後確實比較常再跌，而且 {years} 年裡有 {years - hi_wins} 年"
                       "偏斜高的時候之後表現較差。")
        elif fell_gap <= -8 and consistent_up:
            reading = (f"在這段歷史裡，偏斜陡之後反而比較少再大跌，而且 {years} 年裡有 {hi_wins} 年"
                       "偏斜高的時候之後表現較好。")
        else:
            def versus(gap: float) -> str:
                return "少" if gap <= -3 else "多" if gap >= 3 else "差不多"

            reading = (f"也就是最怕跌的時候，之後再大跌的次數比中間三組{versus(fell_gap)}、"
                       f"比最不怕跌那組{versus(top['fell'] - bottom['fell'])}；逐年看，{years} 年裡有 {hi_wins} 年"
                       "偏斜高的之後表現較好，沒有一致的方向，不能拿偏斜來猜漲跌。")
        verdict = facts + reading

    return {
        "id": ID,
        "title": "怕跌的保險變貴，之後真的跌更多嗎？",
        "question": "選擇權市場買「跌的保險」比買「漲的彩券」貴很多（偏斜很陡）時，比特幣之後真的跌得比較多嗎？",
        "asof": latest_day,
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": ["偏斜＝買「跌的保險」（價格下方 10% 的賣權）比買「漲的彩券」（價格上方 10% 的買權）貴多少，"
                  "用隱含波動率的差來算；數字越大代表市場越怕跌，負數代表大家更搶著買漲。"] + NOTES,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
