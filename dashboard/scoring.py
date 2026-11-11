"""Market temperature and the per-indicator hot/cold fields. See dashboard/SPEC2.md, sections 1 and 2.

Everything here is fixed in advance (components, equal weights, zone bounds, thresholds); nothing is tuned on
the outcome. The temperature history ranks every past value inside the whole stored history, so it is a
look-back ranking, and the zone statistics are in-sample descriptions with heavily overlapping days and no
costs. The page must say so (DISCLAIMER and STATS_NOTE carry the wording).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from dashboard import common

MIN_PCT_POINTS = 180        # fewer history points than this: no percentile
MIN_COMPONENT_POINTS = 365  # a component needs at least this much history to count
MIN_COMPONENTS = 3          # fewer usable components than this: no temperature
MAX_STALE_DAYS = 7          # a component may lag this many days (weekends, slow sources) and still count
ZONES = (("freezing", "過冷", 0, 20), ("cold", "偏冷", 20, 40), ("neutral", "中性", 40, 60),
         ("warm", "偏熱", 60, 80), ("hot", "過熱", 80, 100))
ASSET_NAMES = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "索拉納"}
PI_LINE, MVRV_LINE = 1.0, 3.7  # cycle insurance thresholds

# (indicator id, short name). Every component is "the higher the hotter". Fixed by SPEC2, not tuned.
COMPONENTS = {
    "BTC": (("btc_vs_ma200", "相對 200 日均線"), ("btc_ma200w_multiple", "200 週均線倍數"),
            ("btc_pi_cycle_top", "Pi 週期比值"), ("btc_mvrv", "MVRV"), ("btc_ath_drawdown", "距最高價的跌幅"),
            ("btc_funding_rate_30d_avg", "資金費率 30 天平均"), ("market_fear_greed", "恐懼貪婪指數"),
            ("btc_etf_flow_30d", "現貨 ETF 30 日淨流入"),
            ("market_stablecoin_supply_change_30d", "穩定幣供給 30 天變化"),
            ("btc_wikipedia_attention", "維基百科關注度")),
    "ETH": (("eth_vs_ma200", "相對 200 日均線"), ("eth_ath_drawdown", "距最高價的跌幅"), ("eth_mvrv", "MVRV"),
            ("eth_funding_rate_30d_avg", "資金費率 30 天平均"), ("market_fear_greed", "恐懼貪婪指數"),
            ("eth_etf_flow_30d", "現貨 ETF 30 日淨流入"),
            ("eth_btc_vs_ma200", "以太幣／比特幣比值相對 200 日均線")),
    "SOL": (("sol_vs_ma200", "相對 200 日均線"), ("sol_ath_drawdown", "距最高價的跌幅"),
            ("sol_funding_rate_30d_avg", "資金費率 30 天平均"), ("market_fear_greed", "恐懼貪婪指數")),
}

# Front-page indicators: price trend, futures, on-chain, money flows (ETF, exchange balance, stablecoins, CME
# institutions) and sentiment.
KEY = ("btc_vs_ma200", "btc_ma200w_multiple", "btc_pi_cycle_top", "btc_ath_drawdown",
       "btc_funding_rate_30d_avg", "btc_open_interest",
       "btc_mvrv", "btc_exchange_balance",
       "btc_etf_flow_30d", "market_stablecoin_supply_change_30d", "btc_cme_asset_manager_net",
       "market_fear_greed")

# Which way is "hot". Looked up by full id first, then by the id without its asset prefix. Anything not listed
# has no hot/cold side (null).
HOT_BY_ID = {"eth_btc_vs_ma200": "high"}
HOT_BY_SUFFIX = {
    "vs_ma200": "high", "ma200w_multiple": "high", "pi_cycle_top": "high", "ma50_ma200_cross": "high",
    "ath_drawdown": "high", "change_7d": "high", "change_30d": "high",
    "funding_rate": "high", "funding_rate_30d_avg": "high", "perp_premium": "high", "quarterly_basis": "high",
    "long_short_accounts": "high",
    "mvrv": "high", "active_addresses_7d": "high", "fees_usd_7d": "high",
    "etf_flow_daily": "high", "etf_flow_5d": "high", "etf_flow_30d": "high", "coinbase_premium": "high",
    "cme_asset_manager_net": "high", "stablecoin_supply_change_30d": "high",
    "fear_greed": "high", "wikipedia_attention": "high",
}
# What a cold / hot position means in plain words (same lookup order as above).
MEANING = {
    "eth_btc_vs_ma200": ("以太幣相對比特幣偏弱", "以太幣相對比特幣偏強"),
    "vs_ma200": ("價格相對長期均線偏弱", "價格比長期均線高出不少"),
    "ma200w_multiple": ("價格離 200 週平均價不遠", "價格比 200 週平均價高出很多"),
    "pi_cycle_top": ("離週期頂部訊號還遠", "比較接近週期頂部訊號"),
    "ma50_ma200_cross": ("短期均線相對長期均線偏弱", "短期均線遠在長期均線之上"),
    "ath_drawdown": ("離最高價比平常遠", "比平常接近最高價"),
    "change_7d": ("最近一週跌得比平常多", "最近一週漲得比平常多"),
    "change_30d": ("最近一個月跌得比平常多", "最近一個月漲得比平常多"),
    "funding_rate": ("做多的人不擁擠", "做多的人比平常擁擠"),
    "funding_rate_30d_avg": ("做多的人不擁擠", "做多的人比平常擁擠"),
    "perp_premium": ("合約買盤不積極", "合約買盤積極"),
    "quarterly_basis": ("市場不太願意付溢價做多", "市場願意付高溢價做多"),
    "long_short_accounts": ("散戶做多的比例偏低", "散戶做多的比例偏高"),
    "mvrv": ("持有人的平均獲利比平常少", "持有人的平均獲利比平常多"),
    "active_addresses_7d": ("鏈上活動偏冷清", "鏈上活動熱絡"),
    "fees_usd_7d": ("鏈上不擁擠", "鏈上擁擠"),
    "etf_flow_daily": ("ETF 資金偏流出", "ETF 資金流入強"),
    "etf_flow_5d": ("ETF 資金偏流出", "ETF 資金流入強"),
    "etf_flow_30d": ("ETF 資金偏流出", "ETF 資金流入強"),
    "coinbase_premium": ("美國買盤偏弱", "美國買盤偏強"),
    "cme_asset_manager_net": ("長線法人的期貨多單偏少", "長線法人的期貨多單偏多"),
    "stablecoin_supply_change_30d": ("新資金進場慢、甚至在離場", "新資金進場快"),
    "fear_greed": ("市場偏害怕", "市場偏貪婪"),
    "wikipedia_attention": ("大眾關注度偏低", "大眾關注度升溫"),
}

DISCLAIMER = ("這一頁把公開資料整理成「市場溫度」，只描述市場狀態，不是給個人的買賣建議。"
              "溫度是各成分指標在自己過去約 4 年歷史裡的百分位平均，屬於事後排名：當年看到的分數，和現在回頭算的不會一樣。"
              "分區統計是樣本內的描述，日與日高度重疊、沒有扣交易成本，不是驗證過的策略，過去的走勢不代表之後會重演。"
              "週期保險（Pi 週期、MVRV）的證據很弱，只當警示。要不要交易、交易多少，請自行判斷並自負風險。")
TEMP_NOTE = ("歷史溫度是事後排名：每個成分過去的數值，都是放進現在手上整段歷史（約 4 年）裡算百分位，"
             "再把當天有資料的成分等權平均（至少 3 個才算）。較早的日期成分比較少（例如現貨 ETF 上市前沒有 ETF 資料）。")
STATS_NOTE = ("樣本內統計：把過去每一天依當時的溫度分區，看之後 30 天、90 天的價格變化。"
              "日與日高度重疊（實際獨立的樣本遠少於天數）、沒有扣交易成本、溫度本身是事後排名，"
              "不是驗證過的策略，不能當成預測。")


# ---------- small helpers ----------

def _suffix(id: str) -> str:
    return id.split("_", 1)[1] if "_" in id else id


def _lookup(table: dict, id: str):
    return table[id] if id in table else table.get(_suffix(id))


def hot_when(id: str) -> str | None:
    return HOT_BY_ID.get(id) or HOT_BY_SUFFIX.get(_suffix(id))


def zone_of(temp: float | None) -> tuple[str | None, str | None]:
    if temp is None:
        return None, None
    for key, label, low, high in ZONES:
        if temp < high or key == "hot":
            return key, label
    return None, None


def _series(history: list) -> pd.Series:
    return pd.Series([float(p[1]) for p in history], index=[str(p[0])[:10] for p in history], dtype=float)


def pct_series(history: list) -> pd.Series:
    """Percentile (0-100) of every point inside the whole history: share of points below it, ties count half."""
    s = _series(history)
    return (s.rank(method="average") - 0.5) / len(s) * 100


def _days_between(first: str, last: str) -> int:
    return (datetime.strptime(last[:10], "%Y-%m-%d") - datetime.strptime(first[:10], "%Y-%m-%d")).days


def _span(history: list) -> str:
    days = _days_between(history[0][0], history[-1][0])
    return f"約 {max(1, round(days / 365))} 年" if days >= 330 else f"約 {max(1, round(days / 30))} 個月"


def _head(row: dict) -> str:
    """The first part of the display string, without the bracketed details."""
    return str(row.get("display") or "").split("（")[0].strip()


def _short(row: dict) -> str:
    return str(row.get("name") or row.get("id")).split("（")[0].strip()


def _level(pct: float) -> str:
    return "很低" if pct < 20 else "偏低" if pct < 40 else "中等" if pct < 60 else "偏高" if pct < 80 else "很高"


def _amount(value: float, unit: str) -> str:
    """A difference written for people: 億／萬 for big numbers, percentage points for percent units."""
    size = abs(value)
    if unit.startswith("%"):
        return f"{size:.2f} 個百分點"
    unit = unit.split("（")[0]
    if size >= 1e8:
        return f"{size / 1e8:,.1f} 億{unit}"
    if size >= 1e4:
        return f"{size / 1e4:,.1f} 萬{unit}"
    return f"{size:,.2f} {unit}".strip() if size < 100 else f"{size:,.0f} {unit}".strip()


def _ago(history: list, days: int, slack: int = 10):
    """The stored value about `days` days before the last point (latest point on or before that day)."""
    if len(history) < 2:
        return None
    last = datetime.strptime(history[-1][0][:10], "%Y-%m-%d")
    target = (last - timedelta(days=days)).strftime("%Y-%m-%d")
    floor = (last - timedelta(days=days + slack)).strftime("%Y-%m-%d")
    for day, value in reversed(history):
        if day <= target:
            return value if day >= floor else None
    return None


# ---------- per-indicator fields ----------

def _plain(row: dict, short: str | None = None) -> str:
    name = short or _short(row)
    if row.get("error"):
        return f"{name}這次沒有取得資料。"
    head, history = _head(row), row.get("history") or []
    if row.get("value") is None and row.get("display"):  # a list of names, not a number
        return f"{name}：{row['display']}。"
    gap = " " if head[:1].isascii() else ""  # a space before digits and signs, none before Chinese words
    if name[-1:].isascii():
        name += " "
    now = f"{name}目前{gap}{head}" if head else name.strip()
    if row.get("hot_when") and row.get("pct") is not None:
        pct, heat = row["pct"], row["heat"]
        meaning = _lookup(MEANING, row["id"])
        tail = ""
        if meaning and heat < 40:
            tail = f"，{meaning[0]}"
        elif meaning and heat >= 60:
            tail = f"，{meaning[1]}"
        elif heat < 40:
            tail = "，屬於偏冷的訊號"
        elif heat >= 60:
            tail = "，屬於偏熱的訊號"
        return (f"{name}在過去{_span(history)}裡{_level(pct)}（第 {pct:.0f} 百分位）{tail}"
                + (f"；目前{gap}{head}。" if head else "。"))
    if row.get("hot_when") and history:
        return f"{now}；歷史只有 {len(history)} 筆，還不夠算百分位（至少要 {MIN_PCT_POINTS} 筆）。"
    if not history:
        return f"{name.strip()}：{head}。這個來源沒有歷史數列，只有目前數值。" if head else f"{name.strip()}沒有數值。"
    before, last = _ago(history, 28), history[-1][1]  # sources that keep "30 days" really reach back 29
    if before is None:
        return f"{now}；累積的歷史還不到一個月，暫時無法和一個月前比較。"
    unit = str(row.get("unit") or "")
    if before > 0 and last > 0 and not unit.startswith("%"):
        change = (last / before - 1) * 100
        if abs(change) < 0.05:
            return f"{now}，和大約一個月前差不多。"
        return f"{now}，比大約一個月前{'高' if change > 0 else '低'} {abs(change):.1f}%。"
    diff = last - before
    if diff == 0:
        return f"{now}，和大約一個月前一樣。"
    return f"{now}，比大約一個月前{'高' if diff > 0 else '低'} {_amount(diff, unit)}。"


def annotate(indicators: list[dict]) -> list[dict]:
    """Add hot_when, pct, heat, key and plain to every indicator (in place) and return the list."""
    for row in indicators:
        side = hot_when(row["id"])
        history = row.get("history") or []
        pct = None
        if not row.get("error") and len(history) >= MIN_PCT_POINTS:
            pct = round(float(pct_series(history).iloc[-1]), 1)
        row["hot_when"] = side
        row["pct"] = pct
        row["heat"] = None if side is None or pct is None else pct if side == "high" else round(100 - pct, 1)
        row["key"] = row["id"] in KEY
        row["plain"] = _plain(row)
    return indicators


# ---------- per-asset temperature ----------

def _money(value: float) -> str:
    return f"{value:,.0f} 美元" if value >= 1000 else f"{value:,.2f} 美元"


def _value(by_id: dict, id: str):
    row = by_id.get(id)
    return None if row is None or row.get("error") else row.get("value")


def _components(asset: str, by_id: dict, today: str) -> tuple[list[dict], dict]:
    rows, heats = [], {}
    for id, short in COMPONENTS[asset]:
        row = by_id.get(id)
        history = (row or {}).get("history") or []
        reason = None
        if row is None:
            reason = "這次的資料裡沒有這個指標"
        elif row.get("error"):
            reason = "這次沒有取得資料"
        elif len(history) < MIN_COMPONENT_POINTS:
            reason = f"歷史只有 {len(history)} 筆，不到 {MIN_COMPONENT_POINTS} 筆"
        elif row.get("heat") is None:
            reason = "算不出冷熱分數"
        elif _days_between(history[-1][0], today) > MAX_STALE_DAYS:
            reason = f"資料停在 {history[-1][0]}，超過 {MAX_STALE_DAYS} 天沒有更新"
        counted = reason is None
        if counted:
            pct = pct_series(history)
            heats[id] = pct if row["hot_when"] == "high" else 100 - pct
        rows.append({"id": id, "name": short, "heat": row.get("heat") if row else None,
                     "display": row.get("display") if row else None,
                     "plain": _plain(row, short) if counted else f"沒有計入：{reason}。",
                     "counted": counted, "reason": reason})
    return rows, heats


def _temp_history(heats: dict, today: str) -> pd.Series:
    """Daily equal-weight mean of the component heats; a component is carried forward at most MAX_STALE_DAYS."""
    if not heats:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({id: pd.Series(s.values, index=pd.to_datetime(s.index)) for id, s in heats.items()})
    days = pd.date_range(frame.index.min(), pd.Timestamp(today), freq="D")
    frame = frame.reindex(days).ffill(limit=MAX_STALE_DAYS)
    temp = frame.mean(axis=1)[frame.notna().sum(axis=1) >= MIN_COMPONENTS]
    return temp.dropna().tail(common.HISTORY_POINTS)


def _zone_stats(temp: pd.Series, price_history: list, current_zone: str | None) -> list[dict]:
    out = []
    forward = {}
    if len(temp) and price_history:
        price = pd.Series([p[1] for p in price_history], index=pd.to_datetime([p[0] for p in price_history]))
        price = price.reindex(pd.date_range(price.index.min(), price.index.max(), freq="D")).ffill()
        for days in (30, 90):
            forward[days] = ((price.shift(-days) / price - 1) * 100).reindex(temp.index)
    zones = temp.map(lambda t: zone_of(float(t))[0]) if len(temp) else pd.Series(dtype=object)
    for key, label, low, high in ZONES:
        mask = zones == key if len(zones) else None
        row = {"zone": key, "label": label, "range": f"{low}–{high}", "days": int(mask.sum()) if mask is not None else 0,
               "current": key == current_zone}
        for days in (30, 90):
            values = forward[days][mask].dropna() if days in forward and mask is not None else pd.Series(dtype=float)
            row[f"fwd{days}_n"] = int(len(values))
            row[f"fwd{days}_median"] = round(float(values.median()), 1) if len(values) else None
            row[f"fwd{days}_up"] = round(float((values > 0).mean() * 100), 1) if len(values) else None
        out.append(row)
    return out


def _rules(asset: str, by_id: dict) -> list[dict]:
    if asset != "BTC":
        return []
    out = []
    for id, name, line, digits, what in (
            ("btc_pi_cycle_top", "Pi 週期頂部指標", PI_LINE, 3, "111 日均線 ÷（2 × 350 日均線）"),
            ("btc_mvrv", "MVRV（市值 ÷ 已實現市值）", MVRV_LINE, 2, "總市值 ÷ 全體持有人的估計成本")):
        value = _value(by_id, id)
        if value is None:
            out.append({"id": id, "name": name, "state": "沒有資料", "detail": f"這次沒有取得資料；門檻 {line:g}",
                        "hit": None, "value": None, "threshold": line})
            continue
        hit = value >= line
        out.append({"id": id, "name": name, "state": "達到門檻" if hit else "未達門檻",
                    "detail": f"{value:.{digits}f}，門檻 {line:g}（{what}；證據很弱，只當警示）",
                    "hit": hit, "value": value, "threshold": line})
    return out


def _delta(temp: pd.Series, temperature: float | None, days: int) -> float | None:
    if temperature is None or not len(temp):
        return None
    day = temp.index[-1] - pd.Timedelta(days=days)
    return round(temperature - float(temp[day]), 1) if day in temp.index else None


def _move(delta: float | None, label: str) -> str:
    if delta is None:
        return f"{label}沒有可比的紀錄"
    if abs(delta) < 0.05:
        return f"和{label}一樣"
    return f"比{label}{'升' if delta > 0 else '降'}了 {abs(delta):.1f} 分"


def _summary(asset: str, by_id: dict, temperature, label, components, d7, d30, rules) -> list[str]:
    name = ASSET_NAMES[asset]
    counted = [c for c in components if c["counted"]]
    lines = []
    if temperature is None:
        lines.append(f"{name}這次可用的成分只有 {len(counted)} 個，不到 {MIN_COMPONENTS} 個，所以不給溫度。")
    else:
        lines.append(f"{name}的市場溫度是 {temperature:.0f} 分（0 最冷、100 最熱），落在「{label}」區；"
                     f"{len(components)} 個成分裡計入了 {len(counted)} 個。")
        ranked = sorted(counted, key=lambda c: c["heat"])
        take = 2 if len(ranked) >= 4 else 1
        show = lambda rows: "、".join(f"{c['name']}（{c['heat']:.0f} 分）" for c in rows)  # noqa: E731
        lines.append(f"最熱的成分是{show(ranked[::-1][:take])}；最冷的是{show(ranked[:take])}。")
        lines.append(f"溫度{_move(d7, ' 7 天前')}，{_move(d30, ' 30 天前')}。")
    parts = []
    etf, funding = by_id.get(f"{asset.lower()}_etf_flow_30d"), by_id.get(f"{asset.lower()}_funding_rate_30d_avg")
    if etf is not None:
        parts.append("現貨 ETF 這次沒有取得資料" if etf.get("error")
                     else f"美國現貨 ETF 近 30 個交易日合計{_head(etf)}")
    if funding is not None:
        parts.append("資金費率這次沒有取得資料" if funding.get("error")
                     else f"永續合約資金費率的 30 天平均是{_head(funding)}")
    if parts:
        lines.append("；".join(parts) + "。")
    if rules:
        known = [r for r in rules if r["hit"] is not None]
        hits = [r for r in known if r["hit"]]
        text = "、".join(f"{r['name'].split('（')[0]} {r['value']:.2f}（門檻 {r['threshold']:g}）" for r in known)
        if not known:
            lines.append("週期保險（Pi 週期、MVRV）這次沒有取得資料。")
        elif hits:
            lines.append(f"週期保險達到門檻：{'、'.join(r['name'].split('（')[0] for r in hits)}。目前數值：{text}。")
        else:
            lines.append(f"週期保險都沒有達到門檻：{text}。")
    return lines


def asset_block(asset: str, by_id: dict, today: str) -> dict:
    a = asset.lower()
    components, heats = _components(asset, by_id, today)
    counted = [c for c in components if c["counted"]]
    temperature = round(sum(c["heat"] for c in counted) / len(counted), 1) if len(counted) >= MIN_COMPONENTS else None
    temp = _temp_history(heats, today) if temperature is not None else pd.Series(dtype=float)
    if temperature is not None and len(temp):
        temp.iloc[-1] = temperature  # today's point is exactly the mean of the component scores shown
    zone, label = zone_of(temperature)
    price_row = by_id.get(f"{a}_price") or {}
    price_history = [] if price_row.get("error") else (price_row.get("history") or [])
    price = _value(by_id, f"{a}_price")
    d7, d30 = _delta(temp, temperature, 7), _delta(temp, temperature, 30)
    rules = _rules(asset, by_id)
    return {
        "asset": asset, "name": ASSET_NAMES[asset],
        "price": price, "price_display": _money(price) if price is not None else None,
        "price_asof": price_row.get("asof"),
        "change_24h": _value(by_id, f"{a}_change_24h"), "change_7d": _value(by_id, f"{a}_change_7d"),
        "change_30d": _value(by_id, f"{a}_change_30d"),
        "temperature": temperature, "zone": zone, "zone_label": label,
        "temp_change_7d": d7, "temp_change_30d": d30,
        "components_counted": len(counted), "components_total": len(components),
        "summary": _summary(asset, by_id, temperature, label, components, d7, d30, rules),
        "components": components,
        "temp_history": [[day.strftime("%Y-%m-%d"), round(float(value), 1)] for day, value in temp.items()],
        "temp_note": TEMP_NOTE,
        "price_history": [[p[0], p[1]] for p in price_history],
        "zone_stats": _zone_stats(temp, price_history, zone),
        "zone_stats_note": STATS_NOTE,
        "rules": rules,
    }


def assets(indicators: list[dict], today: str | None = None) -> list[dict]:
    """One block per asset. Call annotate() first. An asset whose block fails is reported, not dropped."""
    today = today or common.today_utc()
    by_id = {row["id"]: row for row in indicators}
    out = []
    for asset in COMPONENTS:
        try:
            out.append(asset_block(asset, by_id, today))
        except Exception as exc:  # noqa: BLE001 - the page must still render the other assets
            out.append({"asset": asset, "name": ASSET_NAMES[asset], "price": None, "price_display": None,
                        "price_asof": None, "change_24h": None, "change_7d": None, "change_30d": None,
                        "temperature": None, "zone": None, "zone_label": None, "temp_change_7d": None,
                        "temp_change_30d": None, "components_counted": 0, "components_total": 0,
                        "summary": [f"{ASSET_NAMES[asset]}的溫度這次算不出來（{type(exc).__name__}: {exc}）。"[:200]],
                        "components": [], "temp_history": [], "temp_note": TEMP_NOTE, "price_history": [],
                        "zone_stats": [], "zone_stats_note": STATS_NOTE, "rules": []})
    return out
