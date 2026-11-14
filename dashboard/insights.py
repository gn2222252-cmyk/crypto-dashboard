"""Turn the indicators and the historical studies into the page's "where are we" blocks.

Everything here is a plain reading of numbers that are already on the page: checklists (is each bull-market
condition met, has leverage been washed out), a map of price levels that sit close together, a table of who is
positioned which way, and the few things worth watching for a change. Nothing here is a forecast or advice.
"""
from __future__ import annotations

import math


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class _Rows:
    def __init__(self, indicators: list[dict]):
        self.by_id = {row["id"]: row for row in indicators if isinstance(row, dict) and row.get("id")}

    def value(self, id: str):
        row = self.by_id.get(id)
        if not row or row.get("error") or not _num(row.get("value")):
            return None
        return float(row["value"])

    def display(self, id: str) -> str:
        row = self.by_id.get(id)
        return (row or {}).get("display") or "這次沒有資料"

    def pct(self, id: str):
        row = self.by_id.get(id)
        return row.get("pct") if row and _num(row.get("pct")) else None

    def history(self, id: str) -> list:
        row = self.by_id.get(id)
        return (row or {}).get("history") or []


def _money(value: float) -> str:
    return f"{value:,.0f}"


def _item(label: str, ok, detail: str, id: str | None = None) -> dict:
    return {"label": label, "ok": ok if ok is None else bool(ok), "detail": detail, "id": id}


def _sign(rows: _Rows, id: str, label: str, positive: bool = True) -> dict:
    v = rows.value(id)
    ok = None if v is None else (v > 0) == positive
    return _item(label, ok, rows.display(id), id)


def _bull(rows: _Rows, studies: dict) -> dict:
    items = [_sign(rows, "btc_vs_ma200", "價格在 200 日均線之上")]
    band = rows.display("btc_bull_support_band")
    items.append(_item("價格在牛市支撐帶（20 週均線）之上", None if rows.value("btc_bull_support_band") is None
                       else band.startswith("在支撐帶之上"), band, "btc_bull_support_band"))
    items.append(_sign(rows, "btc_ma50_ma200_cross", "50 日均線在 200 日均線之上"))
    st = studies.get("structure") or {}
    if _num(st.get("last_high")) and _num(st.get("prev_high")):
        ok = bool(st.get("higher_highs")) and bool(st.get("higher_lows"))
        detail = (f"最近兩個波段高點 {_money(st['prev_high'])} → {_money(st['last_high'])}，"
                  f"低點 {_money(st['prev_low'])} → {_money(st['last_low'])}（漲跌 15% 以上才算一個波段）")
        items.append(_item("波段高點和低點都比上一次高", ok, detail))
    else:
        items.append(_item("波段高點和低點都比上一次高", None, "這次沒有資料"))
    drawdown = rows.value("btc_ath_drawdown")
    items.append(_item("距離歷史最高價不到 20%", None if drawdown is None else drawdown >= -20,
                       rows.display("btc_ath_drawdown"), "btc_ath_drawdown"))
    items.append(_sign(rows, "btc_etf_flow_30d", "美國現貨 ETF 近 30 天是淨流入"))
    items.append(_sign(rows, "market_stablecoin_supply_change_30d", "穩定幣總量近 30 天在增加"))
    items.append(_sign(rows, "btc_lth_supply_change_30d", "長期持有者近 30 天在增持"))
    known = [i for i in items if i["ok"] is not None]
    met = sum(1 for i in known if i["ok"])
    if not known:
        verdict = "這次資料不足，無法判讀。"
    elif met >= len(known) - 1:
        verdict = "幾乎所有常用的牛市條件都成立。"
    elif met >= (len(known) + 1) // 2:
        verdict = "條件一半成立、一半不成立，趨勢不明確。"
    elif met >= 2:
        verdict = "多數常用的牛市條件不成立，只有少數成立。"
    else:
        verdict = "常用的牛市條件幾乎都不成立，以這些定義來看不是牛市。"
    return {"items": items, "met": met, "total": len(known), "verdict": verdict}


def _share(ratio: float) -> float:
    return ratio / (1 + ratio) * 100


def _washout(rows: _Rows, studies: dict) -> dict:
    items = []
    fear = rows.value("market_fear_greed")
    items.append(_item("恐懼與貪婪指數在 20 以下", None if fear is None else fear <= 20,
                       rows.display("market_fear_greed"), "market_fear_greed"))
    funding = rows.value("btc_funding_cross_avg")
    items.append(_item("資金費率轉負（五家交易所平均）", None if funding is None else funding < 0,
                       rows.display("btc_funding_cross_avg"), "btc_funding_cross_avg"))
    oi = rows.value("btc_open_interest_change_30d")
    items.append(_item("合約未平倉量 30 天減少兩成以上", None if oi is None else oi <= -20,
                       rows.display("btc_open_interest"), "btc_open_interest"))
    history = [p for p in rows.history("btc_long_short_accounts") if _num(p[1])]
    if len(history) >= 25:
        now, before = _share(history[-1][1]), _share(history[max(0, len(history) - 31)][1])
        items.append(_item("做多的合約帳戶比一個月前少", now < before - 2,
                           f"幣安做多帳戶比例現在 {now:.0f}%，約一個月前 {before:.0f}%", "btc_long_short_accounts"))
    else:
        items.append(_item("做多的合約帳戶比一個月前少", None, "歷史資料還不夠一個月", "btc_long_short_accounts"))
    known = [i for i in items if i["ok"] is not None]
    met = sum(1 for i in known if i["ok"])
    if not known:
        verdict = "這次資料不足，無法判讀。"
    elif met == len(known):
        verdict = "槓桿多單看起來已經被清洗過。"
    elif met >= 2:
        verdict = "清洗進行到一半：有些條件出現了，有些還沒有。"
    elif met == 1:
        verdict = "情緒已經很差，但槓桿多單還沒有被清掉。" if items[0]["ok"] else "只有一項條件出現，還談不上清洗。"
    else:
        verdict = "沒有清洗的跡象。"
    study = studies.get("washout") or {}
    return {"items": items, "met": met, "total": len(known), "verdict": verdict,
            "state_now": (study.get("now") or {}).get("state")}


def _levels(rows: _Rows, studies: dict) -> dict:
    price = rows.value("btc_price")
    study = studies.get("levels") or {}
    found = [dict(level) for level in study.get("levels") or [] if _num(level.get("price")) and level["price"] > 0]
    for id, key, label in (("btc_realized_price", "realized", "全體持有人平均成本（已實現價格）"),
                           ("btc_sth_cost_price", "sth_cost", "近期買家平均成本"),
                           ("btc_max_pain", "max_pain", "選擇權最大痛點（最近到期日）")):
        v = rows.value(id)
        if v is not None and v > 0:
            found.append({"key": key, "label": label, "price": round(v), "id": id})
    if price is None or not found:
        return {"price": price, "levels": [], "clusters": []}
    for level in found:
        level["gap"] = round((level["price"] / price - 1) * 100, 1)
    found.sort(key=lambda level: level["price"])
    # levels that sit within 1.5% of the next one form a cluster; three or more are worth pointing out
    clusters, run = [], [found[0]]
    for level in found[1:]:
        if level["price"] <= run[-1]["price"] * 1.015:
            run.append(level)
        else:
            clusters.append(run)
            run = [level]
    clusters.append(run)
    out = []
    for index, run in enumerate(c for c in clusters if len(c) >= 3):
        lo, hi = run[0]["price"], run[-1]["price"]
        for level in run:
            level["cluster"] = index
        out.append({"index": index, "low": lo, "high": hi, "count": len(run),
                    "side": "above" if lo > price else "below" if hi < price else "around",
                    "gap_low": round((lo / price - 1) * 100, 1), "gap_high": round((hi / price - 1) * 100, 1),
                    "labels": [level["label"] for level in run]})
    found.sort(key=lambda level: -level["price"])
    return {"price": price, "asof": study.get("asof"), "levels": found, "clusters": out}


def _positions(rows: _Rows) -> list[dict]:
    out = []

    def add(group: str, name: str, id: str, read) -> None:
        v = rows.value(id)
        out.append({"group": group, "name": name, "id": id, "value": rows.display(id),
                    "read": "這次沒有資料" if v is None else read(v)})

    add("嘴上說的", "大眾情緒（恐懼與貪婪指數）", "market_fear_greed",
        lambda v: "很悲觀" if v <= 25 else "很樂觀" if v >= 75 else "偏悲觀" if v < 45 else "偏樂觀" if v > 55 else "中性")
    add("合約散戶", "幣安合約帳戶（算人頭）", "btc_long_short_accounts",
        lambda v: "多數帳戶做多" if _share(v) >= 65 else "多數帳戶做空" if _share(v) <= 45 else "多空人數接近")
    add("合約散戶", "幣安大戶帳戶（算部位）", "btc_top_trader_positions",
        lambda v: "大戶部位偏多" if _share(v) >= 60 else "大戶部位偏空" if _share(v) <= 45 else "多空接近")
    add("合約散戶", "資金費率（五家平均，年化）", "btc_funding_cross_avg",
        lambda v: "做空的人在付錢，多方不擁擠" if v < 0 else "做多的人付少量成本" if v < 10 else "做多的人願意付高成本，多方擁擠")
    add("合約散戶", "幣安合約未平倉量", "btc_open_interest_change_30d",
        lambda v: "槓桿在撤退" if v <= -10 else "槓桿在增加" if v >= 10 else "變化不大")
    out[-1]["value"] = rows.display("btc_open_interest")
    add("大帳戶與法人", "Hyperliquid 最大的 60 個帳戶", "btc_hl_whales_net",
        lambda v: "淨做空" if v < 0 else "淨做多" if v > 0 else "多空相當")
    pct = rows.pct("btc_cme_asset_manager_net")
    add("大帳戶與法人", "CME 資產管理機構（長線法人）", "btc_cme_asset_manager_net",
        lambda v: ("淨做空" if v < 0 else "淨做多") + ("" if pct is None else "，在近四年裡偏低" if pct <= 20
                                                    else "，在近四年裡偏高" if pct >= 80 else "，在近四年的中間位置"))
    add("大帳戶與法人", "CME 槓桿基金（避險基金）", "btc_cme_leveraged_funds_net",
        lambda v: "淨做空（多半是套利的對沖部位）" if v < 0 else "淨做多")
    add("大帳戶與法人", "選擇權偏斜（30 天）", "btc_risk_reversal_30d",
        lambda v: "買下跌保險的人比較多" if v < -2 else "買上漲的人比較多" if v > 2 else "兩邊差不多")
    add("現貨資金", "美國現貨 ETF（近 30 天）", "btc_etf_flow_30d", lambda v: "資金流入" if v > 0 else "資金流出")
    add("現貨資金", "穩定幣總量（近 30 天）", "market_stablecoin_supply_change_30d",
        lambda v: "場內資金增加" if v > 0 else "場內資金減少")
    add("現貨資金", "Coinbase 溢價", "btc_coinbase_premium",
        lambda v: "美國買盤比海外強" if v > 0.02 else "美國買盤比海外弱" if v < -0.02 else "兩邊差不多")
    add("鏈上持有人", "持有 1,000 顆以上的地址（近 30 天）", "btc_whales_1k_plus_change_30d",
        lambda v: "持有量增加" if v > 0 else "持有量減少")
    add("鏈上持有人", "長期持有者（近 30 天）", "btc_lth_supply_change_30d",
        lambda v: "在增持" if v > 0 else "在減持")
    return out


def _watch(rows: _Rows) -> list[dict]:
    """A few numbers whose change would mean the state has changed, and whether that change is here yet."""
    out = []

    def add(name: str, id: str, change: str, met) -> None:
        v = rows.value(id)
        out.append({"name": name, "id": id, "now": rows.display(id), "change": change,
                    "met": None if v is None else bool(met(v))})
    add("價格相對 200 日均線", "btc_vs_ma200", "站回均線之上並守住", lambda v: v > 0)
    add("美國現貨 ETF 近 5 天", "btc_etf_flow_5d", "由流出轉為流入", lambda v: v > 0)
    add("穩定幣總量近 30 天", "market_stablecoin_supply_change_30d", "止跌回升", lambda v: v > 0)
    add("資金費率（五家平均）", "btc_funding_cross_avg", "轉為負值（多單被清洗）", lambda v: v < 0)
    add("長期持有者近 30 天", "btc_lth_supply_change_30d", "由減持轉為增持", lambda v: v > 0)
    return out


def build(indicators: list[dict], studies: dict) -> dict:
    rows = _Rows(indicators)
    studies = studies if isinstance(studies, dict) else {}
    bull, wash, levels = _bull(rows, studies), _washout(rows, studies), _levels(rows, studies)
    lines = [f"常用的 {bull['total']} 個牛市條件，現在成立 {bull['met']} 個。{bull['verdict']}",
             f"槓桿清洗的 {wash['total']} 個條件，現在出現 {wash['met']} 個。{wash['verdict']}"]
    above = sorted((c for c in levels["clusters"] if c["side"] == "above"), key=lambda c: c["low"])
    below = sorted((c for c in levels["clusters"] if c["side"] == "below"), key=lambda c: -c["high"])
    if above:
        c = above[0]
        lines.append(f"上方最近的價位密集區在 {_money(c['low'])}–{_money(c['high'])} 美元（{c['count']} 條線擠在一起，"
                     f"比現價高 {c['gap_low']:.0f}%–{c['gap_high']:.0f}%）。")
    if below:
        c = below[0]
        lines.append(f"下方最近的價位密集區在 {_money(c['low'])}–{_money(c['high'])} 美元（{c['count']} 條線擠在一起，"
                     f"比現價低 {abs(c['gap_high']):.0f}%–{abs(c['gap_low']):.0f}%）。")
    else:
        near = sorted((lv for lv in levels["levels"] if lv["price"] < (levels["price"] or 0)),
                      key=lambda lv: -lv["price"])[:3]
        if near:
            lines.append("下方沒有三條線以上的密集區，最近的幾個價位是 "
                         + "、".join(f"{_money(lv['price'])}（{lv['label']}）" for lv in near) + "。")
    cycles = studies.get("cycles") or {}
    cycle, window = cycles.get("now") or {}, cycles.get("window") or {}
    if _num(cycle.get("days_since_peak")) and _num(window.get("min_days")):
        lines.append(f"這一輪高點後第 {cycle['days_since_peak']} 天，距高點 {cycle['drawdown_now']:.0f}%。"
                     f"前三次熊市的最低點出現在高點後第 {window['min_days']}–{window['max_days']} 天。")
    return {"bull": bull, "washout": wash, "levels": levels, "positions": _positions(rows), "watch": _watch(rows),
            "lines": lines}
