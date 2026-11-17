"""Study: were the option market's "chance of touching a price" numbers accurate for bitcoin?

For every day since Deribit's DVOL index exists (2021-03): the predicted chance that the price touches a level
within H days is 2 * N(-|ln(K / S)| / (sigma * sqrt(H / 365))) with sigma = that day's DVOL (one flat volatility,
no skew). Realised = did the daily high (up) or low (down) of the next H days reach the level. The two are compared
by distance from the price (3% to 25%), direction (up / down) and horizon (14, 30, 45 days).

"Today" applies the same formula to the current price and DVOL at H = 30 and scales it by the realised / predicted
ratio of the same distance and direction ("historically adjusted").

Sources (public, no key): Deribit volatility index, Binance spot daily klines. Descriptive and in-sample: the daily
windows overlap, so the number of independent windows is far smaller than the number of days.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

from dashboard import common

DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
KLINES_URL = "https://api.binance.com/api/v3/klines"
SINCE = datetime(2021, 3, 1, tzinfo=timezone.utc)
DISTANCES = (0.03, 0.05, 0.08, 0.11, 0.15, 0.20, 0.25)
HORIZONS = (14, 30, 45)
MAIN = 30  # horizon used for "today" and the verdict
DAY_MS = 86_400_000
SIDES = (("down", "下跌"), ("up", "上漲"))


# ---------- data ----------

def _dvol(start_ms: int) -> dict[int, float]:
    """Deribit BTC DVOL daily close as a fraction (0.45 = 45% a year), keyed by day number since 1970."""
    out: dict[int, float] = {}
    end = int(time.time() * 1000)
    for _ in range(20):
        reply = common.get_json(DVOL_URL, {"currency": "BTC", "start_timestamp": start_ms, "end_timestamp": end,
                                           "resolution": "1D"})
        result = reply["result"]
        for row in result["data"]:
            value = float(row[4])
            if math.isfinite(value) and value > 0:
                out[int(row[0]) // DAY_MS] = value / 100
        if not result.get("continuation"):
            break
        end = result["continuation"]
    if len(out) < 500:
        raise RuntimeError(f"Deribit DVOL history too short: {len(out)} days")
    return out


def _klines(start_ms: int) -> dict[int, tuple[float, float, float]]:
    """Binance BTCUSDT daily (high, low, close) keyed by day number; the last entry is today's unfinished candle."""
    rows, start = [], start_ms
    for _ in range(10):
        page = common.get_json(KLINES_URL, {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000,
                                            "startTime": start})
        if not page:
            break
        rows += page
        start = page[-1][0] + DAY_MS
        if len(page) < 1000:
            break
    if len(rows) < 500:
        raise RuntimeError(f"Binance daily history too short: {len(rows)} days")
    return {int(r[0]) // DAY_MS: (float(r[2]), float(r[3]), float(r[4])) for r in rows}


# ---------- maths ----------

def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _touch(distance: float, side: str, sigma: float, horizon: int) -> float:
    """Chance of touching a level `distance` away within `horizon` days under one flat volatility."""
    spread = sigma * math.sqrt(horizon / 365)
    log_move = math.log(1 + distance) if side == "up" else -math.log(1 - distance)
    return 2 * _cdf(-log_move / spread)


def _calibrate(dvol: dict, prices: dict, horizon: int) -> dict:
    """{(side, distance): (mean predicted, realised frequency)}, plus "n" start days and first/last start day."""
    predicted = {(side, m): 0.0 for side, _ in SIDES for m in DISTANCES}
    hits = {key: 0 for key in predicted}
    starts = []
    for day in sorted(set(dvol) & set(prices)):
        window = [prices.get(day + i) for i in range(1, horizon + 1)]
        if any(w is None for w in window):
            continue
        starts.append(day)
        spot = prices[day][2]
        high = max(w[0] for w in window)
        low = min(w[1] for w in window)
        for m in DISTANCES:
            predicted[("up", m)] += _touch(m, "up", dvol[day], horizon)
            predicted[("down", m)] += _touch(m, "down", dvol[day], horizon)
            hits[("up", m)] += high >= spot * (1 + m)
            hits[("down", m)] += low <= spot * (1 - m)
    n = len(starts)
    if n < 200:
        raise RuntimeError(f"too few complete {horizon}-day windows: {n}")
    out = {key: (predicted[key] / n, hits[key] / n) for key in predicted}
    out["n"] = n
    out["first"], out["last"] = starts[0], starts[-1]
    return out


# ---------- formatting ----------

def _per100(p: float) -> str:
    return f"{round(p * 100)} 次"


def _pct(m: float) -> str:
    return f"{m * 100:.0f}%"


def _date(day: int) -> str:
    return datetime.fromtimestamp(day * 86400, timezone.utc).strftime("%Y-%m-%d")


def _calibration_table(cal: dict, horizon: int) -> dict:
    rows = []
    for m in DISTANCES:
        down_p, down_r = cal[("down", m)]
        up_p, up_r = cal[("up", m)]
        rows.append([_pct(m), _per100(down_p), _per100(down_r), _per100(up_p), _per100(up_r)])
    return {"caption": f"{horizon} 天內碰到的次數（每 100 次）：市場估計 對 實際發生；"
                       f"共 {cal['n']} 個起算日，其中互不重疊的只有約 {cal['n'] // horizon} 段",
            "columns": ["距離現價", "跌：市場估", "跌：實際", "漲：市場估", "漲：實際"],
            "rows": rows}


# ---------- study ----------

def study() -> dict:
    start_ms = int(SINCE.timestamp() * 1000)
    dvol = _dvol(start_ms)
    candles = _klines(start_ms)
    last_day = max(candles)
    spot = candles[last_day][2]  # latest traded price (today's unfinished candle)
    closed = {day: v for day, v in candles.items() if day != last_day}  # history uses finished candles only
    sigma_day = max(dvol)
    sigma = dvol[sigma_day]

    cal = {h: _calibrate(dvol, closed, h) for h in HORIZONS}
    main = cal[MAIN]

    today_rows = []
    for side, word in SIDES:
        for m in DISTANCES:
            level = spot * (1 + m) if side == "up" else spot * (1 - m)
            market = _touch(m, side, sigma, MAIN)
            mean_p, real = main[(side, m)]
            adjusted = min(1.0, market * real / mean_p) if mean_p > 0 else market
            today_rows.append([f"{word} {_pct(m)}", f"${level:,.0f}", _per100(market), _per100(adjusted)])

    ratios = [main[(side, m)][1] / main[(side, m)][0] for side, _ in SIDES for m in DISTANCES
              if main[(side, m)][0] > 0]
    over = sum(1 for r in ratios if r < 1)
    d11_p, d11_r = main[("down", 0.11)]
    u20_p, u20_r = main[("up", 0.20)]
    verdict = (f"過去 {MAIN} 天期的紀錄裡，市場估的「會碰到」次數多半偏高（{len(ratios)} 種距離與方向中有 {over} 種）："
               f"例如跌 11%，市場平均估 100 次裡 {round(d11_p * 100)} 次，實際是 {round(d11_r * 100)} 次；"
               f"漲 20%，市場估 100 次裡 {round(u20_p * 100)} 次，實際是 {round(u20_r * 100)} 次。"
               f"但互不重疊的 {MAIN} 天區間只有約 {main['n'] // MAIN} 段，這個差距只能當參考。")

    now = [
        {"label": "比特幣現價", "value": f"${spot:,.0f}"},
        {"label": "選擇權市場預期的一年波動幅度（DVOL）", "value": f"{sigma * 100:.1f}%（{_date(sigma_day)}）"},
        {"label": f"資料範圍（{MAIN} 天期起算日）", "value": f"{_date(main['first'])} 到 {_date(main['last'])}"},
        {"label": "樣本數", "value": "；".join(f"{h} 天期 {cal[h]['n']} 天、不重疊約 {cal[h]['n'] // h} 段"
                                          for h in HORIZONS)},
    ]

    tables = [{"caption": f"今天：未來 {MAIN} 天內碰到各價位的次數（每 100 次）。"
                          "「歷史校正後」＝市場估計 × 過去同距離同方向的（實際 ÷ 市場估）",
               "columns": ["方向與距離", "價位", "市場估", "歷史校正後"],
               "rows": today_rows}]
    tables += [_calibration_table(cal[h], h) for h in (MAIN, 14, 45)]

    notes = [
        "「碰到」是指這段期間內只要有一天的最高價或最低價到過那個價位就算，不必收在那裡。",
        "市場估計是用 DVOL（選擇權價格反推的預期波動幅度）套簡化公式算的，上下用同一個波動數字，"
        "沒有考慮市場對下跌和上漲定價不同（偏斜），所以不等於選擇權市場對每個價位的真正報價。",
        "每天都起算一次，相鄰的區間大部分重疊，看起來有兩千筆左右，其實互不重疊的只有幾十到一百多段（見各表標題），樣本比表面上小很多。",
        "這是拿同一段歷史（2021 年 3 月以後）自己比自己，沒有用另一段資料驗證；換一段時期，差距可能縮小或反過來。",
        "「歷史校正後」只是把過去的偏差比例套到今天，過去的頻率不是保證，不代表未來會照這個比例發生。",
        "用的是日線最高最低價（幣安現貨），今天尚未收完的那根不列入歷史統計；本頁只描述過去紀錄，不是買賣建議。",
    ]
    return {
        "id": "touch_calibration",
        "title": "市場說會碰到的價位，實際碰到幾次",
        "question": "選擇權市場估「幾天內有多少機會碰到某個價位」，過去 100 次裡實際發生了幾次？",
        "asof": _date(last_day),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Deribit DVOL 比特幣波動率指數（日線）",
             "url": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=1D"},
            {"name": "Binance BTCUSDT 現貨日線",
             "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
