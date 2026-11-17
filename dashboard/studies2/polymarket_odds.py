"""Historical study: were Polymarket's "what price will bitcoin hit this month" odds accurate, and which way do they lean?

Every month Polymarket lists one market per price level ("reach $X" above the price, "dip to $X" below it); the Yes
price is the crowd's chance that bitcoin touches the level before the month ends. For every finished month this
module takes each market's price two days after the month opened (and again at mid-month), and compares it with
what happened. For the same markets it also computes the option market's estimate (flat volatility from Deribit's
DVOL index), so the two can be scored against each other. A second question: when the crowd prices a fall much
higher than an equal-sized rise (the "lean"), did bitcoin actually fall more that month?

Finished months never change, so they are kept in cache/_store/polymarket_history.json and fetched once.

Sources (public, no key): Polymarket gamma API and price history, Deribit DVOL, Binance daily candles, and the
dashboard's own option odds (cache/_implied.json) for today's comparison.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone

from dashboard import common
from dashboard.collectors import price as px

ID = "polymarket_odds"
SEARCH = "https://gamma-api.polymarket.com/public-search"
EVENTS = "https://gamma-api.polymarket.com/events"
HISTORY = "https://clob.polymarket.com/prices-history"
DVOL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
STORE_FILE = common.STORE / f"{ID}.json"
TITLE = re.compile(r"^What price will (Bitcoin|BTC) (hit|reach) in [A-Z][a-z]+\?$")
BUDGET = 240  # seconds one run may spend fetching finished months
MIN_VOLUME = 2000  # dollars; thinner markets are not a real price
SAMPLES = (("月初第 3 天", 2.0), ("月中", 15.0))  # label, days after the month's markets opened
BUCKETS = ((0, 3), (3, 10), (10, 25), (25, 50), (50, 75), (75, 100.01))
LEAN_AT = 0.10  # compare a 10% fall with a 10% rise


def _ts(text) -> float | None:
    """Seconds since 1970 for an ISO-like timestamp; a time without a zone is taken as UTC."""
    raw = str(text).strip().replace(" ", "T").replace("Z", "+00:00")
    for candidate in (raw, raw[:19]):
        try:
            stamp = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    return None


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------- finished months ----------

def _load() -> dict:
    try:
        data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - first run or damaged file
        return {}


def _save(store: dict) -> None:
    common.STORE.mkdir(parents=True, exist_ok=True)
    STORE_FILE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")


def _fetch_event(slug: str) -> dict | None:
    reply = common.get_json(EVENTS, {"slug": slug}, timeout=40)
    if not reply:
        return None
    ev = reply[0]
    start, end = _ts(ev.get("startDate")), _ts(ev.get("endDate"))
    if start is None or end is None:
        return None
    markets = []
    for m in ev.get("markets") or []:
        found = re.search(r"\$([\d,]+)", str(m.get("question") or ""))
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
            volume = float(m.get("volume") or 0)
        except Exception:  # noqa: BLE001
            continue
        if not found or not m.get("closed") or len(prices) != 2 or not tokens or volume < MIN_VOLUME:
            continue
        if {str(prices[0]), str(prices[1])} != {"0", "1"}:  # not settled cleanly
            continue
        question = str(m.get("question")).lower()
        try:
            points = common.get_json(HISTORY, {"market": tokens[0], "interval": "max", "fidelity": 720},
                                     timeout=40).get("history") or []
        except Exception:  # noqa: BLE001 - one market without history is skipped
            continue
        markets.append({"level": int(found.group(1).replace(",", "")),
                        "side": "down" if "dip" in question else ("up" if "reach" in question else None),
                        "yes": str(prices[0]) == "1", "volume": round(volume),
                        "hist": [[int(p["t"]), round(float(p["p"]), 4)] for p in points]})
    return {"start": start, "end": end, "markets": markets} if markets else None


def _update() -> dict:
    store = _load()
    began = time.time()
    try:
        found = common.get_json(SEARCH, {"q": "what price will bitcoin hit in", "limit_per_type": 50,
                                         "events_status": "closed"}, timeout=40).get("events") or []
    except Exception:  # noqa: BLE001 - keep what we have
        found = []
    now = time.time()
    changed = False
    for ev in found:
        slug, end = ev.get("slug"), _ts(ev.get("endDate"))
        if not slug or slug in store or not TITLE.match(str(ev.get("title") or "")) or end is None or end > now - 86400:
            continue
        if time.time() - began > BUDGET:
            break
        try:
            event = _fetch_event(slug)
        except Exception:  # noqa: BLE001 - try again next run
            continue
        if event:
            store[slug] = event
            changed = True
            _save(store)
    if changed:
        _save(store)
    return store


# ---------- market data ----------

def _candles() -> dict:
    """{day: (high, low, close)} of closed Binance daily candles since 2024-01."""
    rows, start = {}, int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for _ in range(6):
        page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000, "startTime": start})
        if not page:
            break
        for k in page:
            rows[_day(k[0] / 1000)] = (float(k[2]), float(k[3]), float(k[4]))
        if len(page) < 1000:
            break
        start = int(page[-1][0]) + 86400000
    return rows


def _dvol() -> dict:
    out, end = {}, int(time.time() * 1000)
    start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for _ in range(12):
        reply = common.get_json(DVOL, {"currency": "BTC", "start_timestamp": start, "end_timestamp": end,
                                       "resolution": "1D"})["result"]
        for row in reply.get("data") or []:
            out[_day(row[0] / 1000)] = float(row[4]) / 100
        if not reply.get("continuation"):
            break
        end = reply["continuation"]
    return out


def _price_at(hist: list, ts: float) -> float | None:
    """The last traded Yes price at or before ts, if there is one within the 36 hours before."""
    best = None
    for t, p in hist:
        if t <= ts:
            best = (t, p)
        else:
            break
    return best[1] if best and ts - best[0] <= 36 * 3600 else None


def _samples(store: dict, candles: dict, dvol: dict, after_days: float) -> list[dict]:
    rows = []
    for slug, ev in sorted(store.items(), key=lambda kv: kv[1]["start"]):
        ts = ev["start"] + after_days * 86400
        if ts >= ev["end"] - 3 * 86400:
            continue
        before = _day(ts - 86400)  # last closed daily candle before the sample
        if before not in candles or before not in dvol:
            continue
        spot, sigma = candles[before][2], dvol[before]
        left = (ev["end"] - ts) / 86400
        day, seen_hi, seen_lo = _day(ev["start"]), -1.0, 1e18
        while day <= before:  # what the month had already reached before the sample
            if day in candles:
                seen_hi, seen_lo = max(seen_hi, candles[day][0]), min(seen_lo, candles[day][1])
            day = _day(_ts(day + "T00:00:00") + 86400)
        for m in ev["markets"]:
            side = m["side"] or ("up" if m["level"] > spot else "down")
            if (side == "up" and (m["level"] <= spot or seen_hi >= m["level"])) or \
               (side == "down" and (m["level"] >= spot or seen_lo <= m["level"])):
                continue  # already touched, or on the wrong side of the price
            p = _price_at(m["hist"], ts)
            if p is None or p >= 0.97:
                continue
            dist = abs(math.log(m["level"] / spot))
            rows.append({"slug": slug, "month": _day(ev["start"] + 5 * 86400)[:7], "side": side, "level": m["level"],
                         "dist": dist, "p": p, "yes": bool(m["yes"]),
                         "opt": min(2 * _norm_cdf(-dist / (sigma * math.sqrt(left / 365))), 1.0),
                         "spot": spot, "end": ev["end"]})
    return rows


def _lean(rows: list[dict]) -> float | None:
    """Crowd's chance of a 10% fall minus its chance of a 10% rise, in points (log-distance interpolation)."""
    out = {}
    for side in ("down", "up"):
        pts = sorted((r["dist"], r["p"]) for r in rows if r["side"] == side)
        target = abs(math.log(1 - LEAN_AT)) if side == "down" else math.log(1 + LEAN_AT)
        lo = [p for p in pts if p[0] <= target]
        hi = [p for p in pts if p[0] >= target]
        if not lo or not hi:
            return None
        (d0, p0), (d1, p1) = lo[-1], hi[0]
        out[side] = p0 if d1 == d0 else p0 + (p1 - p0) * (target - d0) / (d1 - d0)
    return (out["down"] - out["up"]) * 100


# ---------- formatting ----------

def _n(value, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value, digits: int = 1, unit: str = "%") -> str:
    if value is None:
        return "—"
    return ("+" if value > 0 else "−" if value < 0 else "") + f"{abs(value):.{digits}f}{unit}"


def _calibration(rows: list[dict]) -> list[list[str]]:
    out = []
    for side, label in (("down", "跌到"), ("up", "漲到")):
        for lo, hi in BUCKETS:
            sel = [r for r in rows if r["side"] == side and lo <= r["p"] * 100 < hi]
            if len(sel) < 5:
                continue
            out.append([f"{label}｜報價 {lo}–{min(hi, 100):.0f}", str(len(sel)),
                        _n(sum(r["p"] for r in sel) / len(sel) * 100), _n(sum(r["opt"] for r in sel) / len(sel) * 100),
                        _n(sum(r["yes"] for r in sel) / len(sel) * 100)])
    return out


def _brier(rows: list[dict], key: str) -> float | None:
    return sum((r[key] - r["yes"]) ** 2 for r in rows) / len(rows) if rows else None


def _today(candles: dict) -> tuple[list[dict], list[list[str]], float | None]:
    """Today's Polymarket prices next to the option market's (with its real smile), from the implied collector."""
    try:
        detail = json.loads((common.CACHE / "_implied.json").read_text(encoding="utf-8"))
        event = detail["polymarket"]["events"][0]
        options = detail["options"]
        spot = float(options["price"])
        end = _ts(event["end"] + "T00:00:00")
        left = (end - _ts(str(detail["polymarket"]["asof"]))) / 86400
    except Exception:  # noqa: BLE001 - the comparison is optional
        return [], [], None
    from dashboard.collectors import implied
    try:
        _, smiles = implied._smiles()
    except Exception:  # noqa: BLE001
        smiles = []
    smiles = sorted(smiles, key=lambda e: e["days"])

    def option_touch(level: float) -> float | None:
        if not smiles:
            return None
        pts = []
        for e in smiles:
            below = implied._below(e, level)
            pts.append((e["days"], min(2 * (below if level < e["forward"] else 1 - below), 1.0)))
        if left <= pts[0][0]:
            return pts[0][1] * math.sqrt(max(left, 0.1) / pts[0][0]) if pts[0][0] > 0 else pts[0][1]
        for (d0, p0), (d1, p1) in zip(pts, pts[1:]):
            if d0 <= left <= d1:
                return p0 + (p1 - p0) * (left - d0) / (d1 - d0)
        return pts[-1][1]

    table, rows = [], []
    for lv in event["levels"]:
        level, side = lv["level"], "up" if lv["up"] else "down"
        opt = option_touch(level)
        dist = abs(math.log(level / spot))
        rows.append({"side": side, "dist": dist, "p": lv["prob"] / 100})
        if lv["prob"] < 1.5 and (opt or 0) < 0.015:
            continue
        table.append([("漲到 " if lv["up"] else "跌到 ") + f"{level:,}", _signed((level / spot - 1) * 100, 0),
                      _n(lv["prob"]), _n(opt * 100 if opt is not None else None),
                      _signed(lv["prob"] - opt * 100 if opt is not None else None, 0, " 點")])
    now = [{"label": "比特幣現價", "value": f"${spot:,.0f}"},
           {"label": "這個月的市場還剩", "value": f"{left:.0f} 天（到 {event['end']}）"},
           {"label": "這個月的成交量", "value": f"約 {event['volume'] / 1e6:.1f} 百萬美元"}]
    return now, table, _lean(rows)


# ---------- the study ----------

def study() -> dict:
    store = _update()
    candles, dvol = _candles(), _dvol()
    first = _samples(store, candles, dvol, SAMPLES[0][1])
    middle = _samples(store, candles, dvol, SAMPLES[1][1])
    months = sorted({r["month"] for r in first})
    now, today_table, lean_now = _today(candles)
    if len(first) < 40:
        return {"id": ID, "title": "預測市場的賠率準不準", "question": "Polymarket 上「比特幣這個月會碰到某價位」的賠率，過去準不準、偏向哪一邊？",
                "asof": common.today_utc(), "verdict": f"歷史資料還在補（目前只有 {len(months)} 個月），下次更新會再抓。",
                "now": now, "tables": [], "notes": ["歷史月份分幾次更新才會補齊。"],
                "sources": [{"name": "Polymarket", "url": "https://polymarket.com/crypto/bitcoin"}]}

    def gap(rows: list[dict], side: str) -> tuple[float, float, int]:
        sel = [r for r in rows if r["side"] == side]
        return (sum(r["p"] for r in sel) / len(sel) * 100, sum(r["yes"] for r in sel) / len(sel) * 100, len(sel)) \
            if sel else (0.0, 0.0, 0)

    down_p, down_y, down_n = gap(first, "down")
    up_p, up_y, up_n = gap(first, "up")
    brier_poly, brier_opt = _brier(first, "p"), _brier(first, "opt")
    better = "Polymarket" if brier_poly < brier_opt else "選擇權"

    # the lean of each month against what the month then did
    lean_rows, pairs = [], []
    for month in months:
        rows = [r for r in first if r["month"] == month]
        lean = _lean(rows)
        end_day = _day(rows[0]["end"] - 86400)
        result = (candles[end_day][2] / rows[0]["spot"] - 1) * 100 if end_day in candles else None
        down_hit = sum(r["yes"] for r in rows if r["side"] == "down")
        up_hit = sum(r["yes"] for r in rows if r["side"] == "up")
        if lean is not None and result is not None:
            pairs.append((lean, result))
        lean_rows.append([month, _signed(lean, 0, " 點"), _signed(result, 1), str(down_hit), str(up_hit)])
    summary = ""
    if len(pairs) >= 8:
        ordered = sorted(pairs)
        half = len(ordered) // 2
        low, high = ordered[:half], ordered[half:]
        mean = lambda seq: sum(x[1] for x in seq) / len(seq)  # noqa: E731
        fell = lambda seq: sum(x[1] < 0 for x in seq)  # noqa: E731
        summary = (f"把 {len(pairs)} 個月照「偏向看跌的程度」分兩半：偏跌較重的 {len(high)} 個月，之後到月底平均 {_signed(mean(high))}、"
                   f"{len(high)} 個月裡 {fell(high)} 個月下跌；偏跌較輕的 {len(low)} 個月平均 {_signed(mean(low))}、"
                   f"{len(low)} 個月裡 {fell(low)} 個月下跌。")
    verdict = (f"{len(months)} 個已結束的月份、{len(first)} 個價位：月初第 3 天，「跌到」的價位 Polymarket 平均報 100 次裡 {down_p:.0f} 次，"
               f"實際 {down_y:.0f} 次；「漲到」的價位平均報 {up_p:.0f} 次，實際 {up_y:.0f} 次。"
               f"和同一批價位的選擇權估計相比，整體誤差較小的是{better}"
               f"（誤差分數 {brier_poly:.3f} 對 {brier_opt:.3f}，越小越準）。" + summary)
    if lean_now is not None:
        now.append({"label": "現在的偏向（跌 10% 的機率減漲 10% 的機率）", "value": _signed(lean_now, 0, " 點")})
        if pairs:
            rank = sum(x[0] <= lean_now for x in pairs)
            now.append({"label": "和過去各月月初相比", "value": f"{len(pairs)} 個月裡比 {rank} 個月更偏向看跌"})
    tables = []
    if today_table:
        tables.append({"caption": "現在：同一個價位，Polymarket 和選擇權各給幾成（100 次裡幾次會碰到）",
                       "columns": ["價位", "離現價", "Polymarket", "選擇權", "差距"], "rows": today_table})
    tables.append({"caption": f"過去準不準（{SAMPLES[0][0]}的報價，100 次裡幾次）",
                   "columns": ["方向｜報價區間", "價位數", "Polymarket 報", "選擇權估", "實際碰到"], "rows": _calibration(first)})
    if len(middle) >= 40:
        tables.append({"caption": f"過去準不準（{SAMPLES[1][0]}的報價，只算當時還沒碰到的價位）",
                       "columns": ["方向｜報價區間", "價位數", "Polymarket 報", "選擇權估", "實際碰到"],
                       "rows": _calibration(middle)})
    tables.append({"caption": "每個月月初的偏向，和那個月後來的結果",
                   "columns": ["月份", "偏向看跌", "之後到月底", "跌的價位碰到幾個", "漲的價位碰到幾個"], "rows": lean_rows[-16:]})
    return {
        "id": ID, "title": "預測市場的賠率準不準、偏向哪一邊",
        "question": "Polymarket 上「比特幣這個月會碰到某價位」的賠率，過去準不準？它把下跌看得比上漲重的時候，後來真的跌比較多嗎？",
        "asof": common.today_utc(), "verdict": verdict, "now": now, "tables": tables,
        "notes": [f"只有 {len(months)} 個月，同一個月的各個價位會一起中或一起不中，所以真正獨立的樣本就是月份數。",
                  "選擇權的歷史估計用的是整體波動指數（DVOL）、不含偏斜，是近似值；「現在」那張表用的是含偏斜的實際報價。",
                  "報價取自成交紀錄，冷門價位可能很久才成交一次；成交量太小的價位已排除。",
                  "「偏向」是用跌 10% 和漲 10% 兩個點內插算出來的，月初價位不夠密的月份沒有數字。",
                  "這是事後整理，條件是看過資料後才定的，過去的結果不代表之後也一樣，也不是買賣建議。"],
        "sources": [{"name": "Polymarket（各月「比特幣會碰到什麼價位」市場）", "url": "https://polymarket.com/crypto/bitcoin"},
                    {"name": "Deribit 波動指數 DVOL 與選擇權報價", "url": "https://www.deribit.com/statistics/BTC/volatility-index"},
                    {"name": "幣安現貨日線", "url": "https://www.binance.com/en/trade/BTC_USDT"}]}


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=1))
