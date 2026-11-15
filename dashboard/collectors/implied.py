"""Market-implied price odds for bitcoin: Deribit option quotes and Polymarket price markets.

Two public, key-less sources answer the same question from different crowds: "how likely is price X by date Y?"

Options: for each expiry, the chance of finishing below a strike is the price of a digital put, read here from the
quoted implied volatility of every strike (N(-d2) plus the skew term vega * dIV/dK). These are pricing ("risk
neutral") probabilities, not forecasts: people overpay for protection, so the downside is somewhat overstated.
The chance of merely touching a level before expiry is approximated as twice the chance of finishing beyond it
(the reflection principle; a few points off when the skew is steep).

Polymarket: the "What price will Bitcoin hit in <month>/<year>?" markets quote the touch probability directly.

The tables go to cache/_implied.json for the page section; one indicator is returned for the indicator list.
Run: .venv/Scripts/python.exe -m dashboard.collectors.implied   (see dashboard/SPEC.md)
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone

import numpy as np

from dashboard import common
from dashboard.collectors import options as opt

GROUP = "derivatives"
DETAIL = common.CACHE / "_implied.json"
DERIBIT_PAGE = "https://www.deribit.com/statistics/BTC/volatility-index"
POLY_API = "https://gamma-api.polymarket.com/public-search"
POLY_PAGE = "https://polymarket.com/crypto/bitcoin"
QUANTILES = (10, 25, 50, 75, 90)
TARGET_DAYS = (7, 14, 40, 75, 130)  # the expiries shown: the listed one nearest to each of these, fixed not tuned
MIN_DAYS = 2.0                      # expiries closer than this are too noisy
MIN_OI = 1000.0                     # coins of open interest; thinner expiries are skipped
STEP = 5000                         # price levels shown are multiples of this
REACH = 0.30                        # ... within this distance of the current price
NEAR = 0.02                         # ... but not closer than this (a level next to the price says nothing)
MIN_POLY_VOLUME = 20000.0           # dollars traded; thinner Polymarket levels are not shown
POLY_REACH = (0.6, 1.55)            # Polymarket levels kept, as a multiple of the current price
MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December")


def _levels(price: float) -> list[int]:
    lo, hi = price * (1 - REACH), price * (1 + REACH)
    return [k for k in range(0, int(hi) + STEP, STEP) if lo <= k <= hi and abs(k / price - 1) >= NEAR]


def _smiles() -> tuple[float, list[dict]]:
    """Every usable BTC expiry: forward, strikes with their implied volatility, open interest."""
    result = common.get_json(f"{opt.DERIBIT}/get_book_summary_by_currency", {"currency": "BTC", "kind": "option"},
                             timeout=40)["result"]
    now = datetime.now(timezone.utc).timestamp() * 1000
    by: dict = {}
    spot = None
    for r in result:
        try:
            expiry, strike, _ = opt._parse_instrument(r.get("instrument_name", ""))
        except Exception:  # noqa: BLE001 - an unexpected instrument name is skipped
            continue
        spot = r.get("estimated_delivery_price") or spot
        e = by.setdefault(expiry, {"forwards": [], "iv": {}, "oi": 0.0})
        if r.get("underlying_price"):
            e["forwards"].append(float(r["underlying_price"]))
        if r.get("mark_iv") and r["mark_iv"] > 0:
            e["iv"][strike] = float(r["mark_iv"]) / 100  # calls and puts of one strike share the same mark IV
        e["oi"] += float(r.get("open_interest") or 0)
    out = []
    for expiry in sorted(by):
        e, days = by[expiry], (expiry - now) / opt.DAY_MS
        if days < MIN_DAYS or e["oi"] < MIN_OI or len(e["iv"]) < 8 or not e["forwards"]:
            continue
        strikes = np.array(sorted(e["iv"]))
        out.append({"expiry": expiry, "days": days, "forward": float(np.median(e["forwards"])), "oi": e["oi"],
                    "strikes": strikes, "ivs": np.array([e["iv"][k] for k in strikes])})
    if not out or not spot:
        raise RuntimeError("Deribit 沒有回傳可用的比特幣選擇權資料")
    return float(spot), out


def _below(e: dict, level: float) -> float:
    """Pricing probability that the price finishes below `level` at this expiry."""
    f, t = e["forward"], e["days"] / opt.YEAR_DAYS

    def iv(k: float) -> float:
        return float(np.interp(k, e["strikes"], e["ivs"]))  # flat beyond the last listed strike

    s, h = iv(level), level * 0.01
    d2 = (math.log(f / level) - 0.5 * s * s * t) / (s * math.sqrt(t))
    d1 = d2 + s * math.sqrt(t)
    vega = f * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(t)
    slope = (iv(level + h) - iv(level - h)) / (2 * h)
    return min(max(opt._norm_cdf(-d2) + vega * slope, 0.0), 1.0)


def _quantile(e: dict, q: float) -> float:
    lo, hi = float(e["strikes"][0]), float(e["strikes"][-1])
    for _ in range(50):
        mid = (lo + hi) / 2
        if _below(e, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _pick(expiries: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for target in TARGET_DAYS:
        best = min(expiries, key=lambda e: abs(e["days"] - target))
        if best not in chosen and abs(best["days"] - target) <= target * 0.6:
            chosen.append(best)
    return sorted(chosen, key=lambda e: e["days"])


def _options() -> dict:
    spot, expiries = _smiles()
    levels = _levels(spot)
    rows = []
    for e in _pick(expiries):
        day = datetime.fromtimestamp(e["expiry"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        beyond = {}
        for level in levels:
            p = _below(e, level)
            end = p if level < e["forward"] else 1 - p
            beyond[str(level)] = {"end": round(end * 100, 1), "touch": round(min(2 * end, 1.0) * 100, 1)}
        atm = float(np.interp(e["forward"], e["strikes"], e["ivs"]))
        rows.append({"expiry": day, "days": round(e["days"], 1), "forward": round(e["forward"]),
                     "oi": round(e["oi"]), "atm_iv": round(atm * 100, 1),
                     "iv_down10": round(float(np.interp(e["forward"] * 0.9, e["strikes"], e["ivs"])) * 100, 1),
                     "iv_up10": round(float(np.interp(e["forward"] * 1.1, e["strikes"], e["ivs"])) * 100, 1),
                     "q": {str(q): round(_quantile(e, q / 100)) for q in QUANTILES}, "levels": beyond})
    if not rows:
        raise RuntimeError("沒有未平倉量夠大的到期日")
    return {"asof": common.now_utc(), "price": round(spot), "levels": levels, "expiries": rows}


def _polymarket() -> dict:
    """The monthly and yearly "what price will bitcoin hit" markets; each level's Yes price is the touch chance."""
    now = datetime.now(timezone.utc)
    wanted = {f"What price will Bitcoin hit in {MONTHS_EN[now.month - 1]}?": f"{now.month} 月底前",
              f"What price will Bitcoin hit in {now.year}?": f"{now.year} 年底前"}
    events = []
    seen = set()
    for query in ("what price will bitcoin hit", f"bitcoin {now.year}"):
        reply = common.get_json(POLY_API, {"q": query, "limit_per_type": 10, "events_status": "active"}, timeout=40)
        for ev in reply.get("events") or []:
            title = ev.get("title")
            if title not in wanted or title in seen or ev.get("closed"):
                continue
            seen.add(title)
            levels = []
            for m in ev.get("markets") or []:
                if m.get("closed"):
                    continue
                found = re.match(r"\s*([↑↓])\s*([\d,]+)", str(m.get("groupItemTitle") or ""))
                try:
                    yes = float(json.loads(m.get("outcomePrices") or "[]")[0])
                    volume = float(m.get("volume") or 0)
                except Exception:  # noqa: BLE001 - a market without a usable price is skipped
                    continue
                if not found or volume < MIN_POLY_VOLUME:
                    continue
                levels.append({"level": int(found.group(2).replace(",", "")), "up": found.group(1) == "↑",
                               "prob": round(yes * 100, 1), "volume": round(volume)})
            if levels:
                events.append({"label": wanted[title], "end": str(ev.get("endDate") or "")[:10],
                               "volume": round(float(ev.get("volume") or 0)),
                               "levels": sorted(levels, key=lambda r: r["level"])})
    if not events:
        raise RuntimeError("Polymarket 沒有找到比特幣價位的市場")
    order = list(wanted.values())
    return {"asof": common.now_utc(), "events": sorted(events, key=lambda e: order.index(e["label"]))}


def _wan(v: float) -> str:
    return f"{v / 1e4:.1f} 萬"


def collect() -> list[dict]:
    detail: dict = {"collected_utc": common.now_utc(), "options": None, "polymarket": None, "errors": {}}
    for key, fetch in (("options", _options), ("polymarket", _polymarket)):
        try:
            detail[key] = fetch()
        except Exception as exc:  # noqa: BLE001 - one source failing must not take the other down
            detail["errors"][key] = f"{type(exc).__name__}: {exc}"[:300]
    if detail["options"] and detail["polymarket"]:  # drop the lottery-ticket levels far from the price
        price = detail["options"]["price"]
        for ev in detail["polymarket"]["events"]:
            ev["levels"] = [r for r in ev["levels"] if POLY_REACH[0] <= r["level"] / price <= POLY_REACH[1]]
    common.CACHE.mkdir(parents=True, exist_ok=True)
    DETAIL.write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")

    meta = dict(id="btc_implied_range", group=GROUP, asset="BTC", name="選擇權隱含的價格區間（約一個月）",
                source=opt.DERIBIT_NAME, source_url=DERIBIT_PAGE)
    if not detail["options"]:
        return [common.failed(**meta, error=detail["errors"].get("options", "沒有資料"))]
    rows = detail["options"]["expiries"]
    e = min(rows, key=lambda r: abs(r["days"] - 30))
    width = round((e["q"]["90"] - e["q"]["10"]) / e["forward"] * 100, 1)
    return [common.indicator(
        **meta, value=width, unit="%（八成機率區間的寬度，占現價）",
        display=f"{_wan(e['q']['10'])}–{_wan(e['q']['90'])} 美元（到 {e['expiry']}，八成機率）",
        asof=detail["options"]["asof"], evidence="C", reading="none",
        rule="只描述市場定價，不做偏多或偏空的判讀：中位數永遠貼近現價，這是算法的性質。",
        note=("從 Deribit 各履約價的報價反推：市場認為到期時價格有八成機率落在這個範圍內。數值是範圍寬度占現價的比例，"
              "越大代表市場預期波動越大。限制：這是「定價機率」不是預測；買保險的人願意多付錢，下跌那一側會被高估一些。"
              + opt.SNAPSHOT),
        history=common.accumulate(meta["id"], [[common.today_utc(), width]]))]


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    out = collect()
    common.save("implied", out)
    for row in out:
        print(row["id"], row["asof"], row["display"] or row["error"], f"[歷史 {len(row['history'])} 筆]")
    saved = json.loads(DETAIL.read_text(encoding="utf-8"))
    print("錯誤：", saved["errors"] or "無")
    for e in (saved["options"] or {}).get("expiries", []):
        print(e["expiry"], e["days"], "天", e["q"], {k: v["touch"] for k, v in e["levels"].items()})
    for ev in (saved["polymarket"] or {}).get("events", []):
        print(ev["label"], [(r["level"], "↑" if r["up"] else "↓", r["prob"]) for r in ev["levels"]])
