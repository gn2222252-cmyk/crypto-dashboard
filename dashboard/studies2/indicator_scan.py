"""Historical study "indicator_scan": one league table of every long-history indicator's lopsided readings.

Question: when an indicator sits at an extreme of its own history, did bitcoin rise or fall more often in the
following 30 days, and is that the same from year to year or just luck?

Method, identical for every indicator (see the notes the study prints for the plain-language version):

* Indicators come from the dashboard's own cache files (asset BTC or MARKET, at least 700 daily points). Plain
  price levels in dollars are left out; series that are not daily are left out.
* Each day's value is compared with the indicator's own history up to that day only (expanding window, at least
  365 days). "Low" = at or below the trailing 10th percentile, "high" = at or above the trailing 90th.
* Outcome = bitcoin's close 30 days later against that day's close (Binance spot daily candles).
* Per indicator and side: days, separate episodes (runs, gaps of 7 days or less merged), median and share higher,
  against all days of the same indicator's sample; per calendar year; first half against second half.
* Ranking uses one observation per episode (its first day) and a two-sided sign test against the all-days median,
  because the daily numbers overlap heavily. At least 8 episodes are needed to be ranked.

Everything is descriptive and in-sample. Nothing here is a forecast or advice.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard import common

ASSETS = {"BTC", "MARKET", "ALL"}
MIN_POINTS = 700
MIN_TRAIL = 365        # days of own history before a day is ranked
LOW, HIGH = 0.10, 0.90
HORIZON = 30
GAP = 7                # extreme days this close together are one episode
MIN_EPISODES = 8
MIN_YEAR_DAYS = 10
GOOD_P = 0.10          # "looks good" = episode sign test this unlikely under pure chance
FILL = 4               # carry a value over weekends / holidays (business-day series)
STALE_DAYS = 14
PRICE_SINCE = "2021-06-01"
# dollar price levels: ranking them against their own past is the same as ranking the price itself
PRICE_LEVELS = {"btc_price": "就是價格本身",
                "btc_realized_price": "是美元價位（全體平均成本），等於在排價格本身",
                "btc_sth_cost_price": "是美元價位（近期買家成本），等於在排價格本身"}
GROUP_CN = {"price": "價格動能", "cycle": "週期模型", "derivatives": "衍生品", "onchain": "鏈上", "flows": "資金流",
            "whales": "大戶持有", "macro": "總體經濟", "breadth": "市場廣度", "sentiment": "情緒"}
SIDE_CN = {"low": "偏低的那一成日子", "high": "偏高的那一成日子"}
DASH = "—"


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(float(v))


def _pct(v, digits: int = 1) -> str:
    return f"{float(v) * 100:+.{digits}f}%" if _ok(v) else DASH


def _per100(v) -> str:
    return f"100 次裡 {round(float(v) * 100)} 次" if _ok(v) else DASH


def _num(v, unit: str = "") -> str:
    if not _ok(v):
        return DASH
    v = float(v)
    size = abs(v)
    if size >= 1e12:
        text = f"{v / 1e12:,.2f} 兆"
    elif size >= 1e8:
        text = f"{v / 1e8:,.1f} 億"
    elif size >= 1e5:
        text = f"{v / 1e4:,.0f} 萬"
    elif size >= 1000:
        text = f"{v:,.0f}"
    elif size >= 10:
        text = f"{v:,.1f}"
    elif size >= 0.1 or size == 0:
        text = f"{v:,.2f}"
    else:
        text = f"{v:.4g}"
    unit = (unit or "").strip()
    if not unit:
        return text
    return text + unit if unit[0] in "%％" else f"{text} {unit}"


def _short(text: str | None) -> str | None:
    """The dashboard's own display text without its long explanation in brackets."""
    if not text or len(text) <= 26:
        return text
    cuts = [text.find(mark) for mark in "（；，" if text.find(mark) > 0]
    return text[:min(cuts)] if cuts else text


# ---------- data ----------

def _indicators() -> tuple[list[dict], list[tuple[str, str]]]:
    """(included indicators with a daily series, [(name, why excluded)])."""
    kept, dropped, seen = [], [], set()
    for path in sorted(common.CACHE.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("indicators", [])
        except Exception:  # noqa: BLE001 - a damaged cache file is skipped
            continue
        for row in rows:
            if not isinstance(row, dict) or str(row.get("asset", "")).upper() not in ASSETS:
                continue
            history = row.get("history") or []
            if len(history) < MIN_POINTS:
                continue
            name = str(row.get("name") or row.get("id"))
            if row.get("id") in seen:
                dropped.append((name, "和另一個檔案裡的同一個指標重複"))
                continue
            seen.add(row.get("id"))
            if row["id"] in PRICE_LEVELS:
                dropped.append((name, PRICE_LEVELS[row["id"]]))
                continue
            try:
                series = pd.Series({pd.Timestamp(str(d)[:10]): float(v) for d, v in history}).sort_index()
            except Exception:  # noqa: BLE001
                dropped.append((name, "歷史資料格式讀不懂"))
                continue
            series = series[np.isfinite(series)]
            gaps = series.index.to_series().diff().dt.days.dropna()
            if len(series) < MIN_POINTS or gaps.empty or gaps.median() > 2:
                dropped.append((name, "不是每天一筆的資料（例如每週一筆）"))
                continue
            if series.nunique() < 20:
                dropped.append((name, "數值幾乎不變，排不出高低"))
                continue
            daily = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="D")).ffill(limit=FILL)
            kept.append({"id": row["id"], "name": name, "group": row.get("group", ""), "unit": row.get("unit") or "",
                         "shown": row.get("display") if isinstance(row.get("display"), str) and row.get("display")
                         else (_num(row.get("value"), row.get("unit") or "") if _ok(row.get("value")) else None),
                         "series": daily.dropna(), "last_day": series.index[-1]})
    return kept, dropped


def _prices() -> tuple[pd.Series, str]:
    """Bitcoin daily closes (finished days only) and the name of the source used."""
    try:
        from dashboard.collectors.price import _binance
        rows, start = [], int(pd.Timestamp(PRICE_SINCE, tz="UTC").timestamp() * 1000)
        for _ in range(6):
            page = _binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "startTime": start,
                                               "limit": 1000})
            if not page:
                break
            rows += page
            start = page[-1][0] + 1
            if len(page) < 1000:
                break
        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        closes = {pd.Timestamp(r[0], unit="ms").normalize(): float(r[4]) for r in rows if r[6] < now_ms}
        series = pd.Series(closes).sort_index()
        if len(series) < 800:
            raise ValueError("too few candles")
        return series, "幣安現貨日線"
    except Exception:  # noqa: BLE001 - blocked or down: fall back to the dashboard's own cached closes
        data = json.loads((common.CACHE / "price.json").read_text(encoding="utf-8"))
        row = next(r for r in data["indicators"] if r.get("id") == "btc_price")
        series = pd.Series({pd.Timestamp(str(d)[:10]): float(v) for d, v in row["history"]}).sort_index()
        return series.iloc[:-1], "儀表板自己存下來的比特幣日線（幣安這次取不到）"


# ---------- statistics ----------

def _sign_p(wins: int, n: int) -> float:
    """Two-sided sign test: chance of a split at least this lopsided if each episode were a coin flip."""
    if n <= 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    return min(1.0, 2 * min(sum(pmf[:wins + 1]), sum(pmf[wins:])))


def _chance_good(n: int) -> float:
    """Chance that a coin-flip combination with n episodes would still pass the GOOD_P bar."""
    return sum(math.comb(n, k) * 0.5 ** n for k in range(n + 1) if _sign_p(k, n) <= GOOD_P)


def _episodes(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First day of every run of extreme days (gaps of GAP days or less are the same run)."""
    starts, previous = [], None
    for day in days:
        if previous is None or (day - previous).days > GAP:
            starts.append(day)
        previous = day
    return starts


def _side(flag: pd.Series, fwd: pd.Series, base: pd.Series) -> dict:
    """Everything about one indicator × side. flag: extreme days; fwd: 30-day change; base: all ranked days."""
    days_all = flag.index[flag.values]
    starts_all = _episodes(days_all)
    ext = fwd[days_all].dropna()                     # extreme days whose outcome is known
    base_med, base_up = float(base.median()), float((base > 0).mean())
    out = {"days": int(len(ext)), "episodes_all": len(starts_all), "base_med": base_med, "base_up": base_up,
           "med": None, "up": None, "d_med": None, "d_up": None, "years": 0, "years_beat": 0, "halves": None,
           "ep_n": 0, "ep_wins": 0, "ep_med": None, "ep_up": None, "p": 1.0, "sign": 0}
    if ext.empty:
        return out
    out["med"], out["up"] = float(ext.median()), float((ext > 0).mean())
    out["d_med"], out["d_up"] = out["med"] - base_med, out["up"] - base_up
    for year in sorted(set(ext.index.year)):
        mine = ext[ext.index.year == year]
        if len(mine) >= MIN_YEAR_DAYS:
            out["years"] += 1
            out["years_beat"] += int(mine.median() > base[base.index.year == year].median())
    middle = base.index[len(base) // 2]
    edges = []
    for part_ext, part_base in ((ext[ext.index < middle], base[base.index < middle]),
                                (ext[ext.index >= middle], base[base.index >= middle])):
        edges.append(float(part_ext.median() - part_base.median()) if len(part_ext) >= MIN_YEAR_DAYS else None)
    if None not in edges:
        out["halves"] = (edges[0] > 0) == (edges[1] > 0) and edges[0] != 0 and edges[1] != 0
        out["half_edges"] = edges
    firsts = fwd[starts_all].dropna()                # one observation per episode
    out["ep_n"] = int(len(firsts))
    if len(firsts):
        out["ep_wins"] = int((firsts > base_med).sum())
        out["ep_med"], out["ep_up"] = float(firsts.median()), float((firsts > 0).mean())
        out["p"] = _sign_p(out["ep_wins"], out["ep_n"])
        out["sign"] = 1 if out["ep_wins"] * 2 > out["ep_n"] else (-1 if out["ep_wins"] * 2 < out["ep_n"] else 0)
    return out


def _scan(item: dict, close: pd.Series) -> dict | None:
    series = item["series"]
    q_low = series.expanding(min_periods=MIN_TRAIL).quantile(LOW)
    q_high = series.expanding(min_periods=MIN_TRAIL).quantile(HIGH)
    ranked = q_low.notna() & (q_high > q_low)
    if ranked.sum() < 200:
        return None
    low = (series <= q_low) & ranked
    high = (series >= q_high) & ranked
    fwd = (close.shift(-HORIZON, freq="D").reindex(series.index) / close.reindex(series.index) - 1)
    fwd = fwd.where(np.isfinite(fwd))
    base = fwd[ranked].dropna()
    if len(base) < 200:
        return None
    last = series.index[-1]
    state = "low" if bool(low.iloc[-1]) else ("high" if bool(high.iloc[-1]) else None)
    return {"item": item, "first": base.index[0], "last": base.index[-1], "ranked_days": int(len(base)),
            "state": state, "value": float(series.iloc[-1]), "value_day": last,
            "low": _side(low, fwd, base), "high": _side(high, fwd, base)}


def _consistent(side: dict) -> bool:
    """Same direction in every year that has enough days (at least two such years) and in both halves."""
    if side["years"] < 2 or side["years_beat"] not in (0, side["years"]) or side["halves"] is not True:
        return False
    return (side["years_beat"] > 0) == (side["d_med"] > 0)


def _years_text(side: dict) -> str:
    if side["years"] == 0:
        return DASH
    return f"{side['years']} 年裡 {side['years_beat']} 年比當年平常好"


def _halves_text(side: dict) -> str:
    if side["halves"] is None:
        return "有一半沒有樣本"
    return "前後兩半同方向" if side["halves"] else "前後兩半方向相反"


def _direction(side: dict) -> str:
    return {1: "比平常偏漲", -1: "比平常偏跌", 0: "和平常一樣"}[side["sign"]]


def _view(side: dict) -> str:
    if side["ep_n"] < MIN_EPISODES:
        return "樣本太少"
    if side["p"] <= GOOD_P:
        return f"之後{_direction(side)[3:]}，各年方向一致" if _consistent(side) else f"之後{_direction(side)[3:]}，但各年不一致"
    if side["halves"] is False:
        return "前後兩半方向相反，看不出規律"
    return "和平常差不多，看不出差別"


# ---------- study ----------

def study() -> dict:
    items, dropped = _indicators()
    close, price_source = _prices()
    today = pd.Timestamp(common.today_utc())
    scans = []
    for item in items:
        result = _scan(item, close)
        if result is None:
            dropped.append((item["name"], "排得出高低、又知道 30 天後結果的日子太少"))
        else:
            scans.append(result)
    if not scans:
        raise ValueError("沒有任何指標可以統計")

    combos = [{"scan": scan, "side": key, **scan[key]} for scan in scans for key in ("low", "high")]
    ranked = sorted((c for c in combos if c["ep_n"] >= MIN_EPISODES),
                    key=lambda c: (c["p"], -abs(c["ep_wins"] / c["ep_n"] - 0.5), -c["ep_n"]))
    thin = [c for c in combos if c["ep_n"] < MIN_EPISODES]
    good = [c for c in ranked if c["p"] <= GOOD_P]
    expected_good = sum(_chance_good(c["ep_n"]) for c in ranked)
    steady = [c for c in ranked if _consistent(c)]
    steady_pool = [c for c in ranked if c["years"] >= 2]
    # all years one way by coin flips; the two halves are not independent of the years, so give a range
    steady_high = sum(2 * 0.5 ** c["years"] for c in steady_pool)
    steady_range = f"{steady_high / 2:.0f} 到 {steady_high:.0f}"
    good_steady = [c for c in good if _consistent(c)]
    current = [s for s in scans if s["state"] and (today - s["item"]["last_day"]).days <= STALE_DAYS]
    stale = [s for s in scans if (today - s["item"]["last_day"]).days > STALE_DAYS]

    def name_of(c) -> str:
        return c["scan"]["item"]["name"]

    # table 1: indicators at an extreme today
    order = {"high": 0, "low": 1}
    rows_now = []
    for scan in sorted(current, key=lambda s: (order[s["state"]], s[s["state"]]["p"])):
        side = scan[scan["state"]]
        rows_now.append([
            scan["item"]["name"], _short(scan["item"]["shown"]) or _num(scan["value"], scan["item"]["unit"]),
            "偏高（自己歷史最高的一成）" if scan["state"] == "high" else "偏低（自己歷史最低的一成）",
            f"{side['ep_n']} 段（共 {side['days']} 天）",
            f"{_pct(side['med'])}（平常 {_pct(side['base_med'])}）",
            f"{_per100(side['up'])}（平常 {round(side['base_up'] * 100)} 次）" if _ok(side["up"]) else DASH,
            _years_text(side), _view(side)])

    # table 2: the 12 best-supported indicator × side
    rows_top = [[name_of(c), SIDE_CN[c["side"]], _direction(c), f"{c['ep_n']} 次",
                 f"{c['ep_n']} 次裡 {c['ep_wins']} 次",
                 f"{_pct(c['ep_med'])}（平常 {_pct(c['base_med'])}）", _years_text(c), _halves_text(c)]
                for c in ranked[:12]]

    # table 3: big samples with no edge or an unstable sign
    top_ids = {(id(c["scan"]), c["side"]) for c in ranked[:12]}

    def problem(c) -> str | None:
        share = c["ep_wins"] / c["ep_n"]
        if c["halves"] is False:
            return "前後兩半方向相反"
        if 0.4 <= share <= 0.6:
            return "贏過平常的次數接近一半一半"
        if c["years"] >= 2 and 0 < c["years_beat"] < c["years"] and c["p"] > GOOD_P:
            return "有的年份好、有的年份差"
        return None

    useless = [(c, problem(c)) for c in ranked if (id(c["scan"]), c["side"]) not in top_ids and not _consistent(c)]
    useless = sorted(((c, why) for c, why in useless if why), key=lambda pair: (-pair[0]["ep_n"], -pair[0]["days"]))
    rows_useless = [[name_of(c), SIDE_CN[c["side"]], f"{c['ep_n']} 次", f"{c['ep_n']} 次裡 {c['ep_wins']} 次",
                     f"{_pct(c['med'])}（平常 {_pct(c['base_med'])}）",
                     f"{_per100(c['up'])}（平常 {round(c['base_up'] * 100)} 次）", _years_text(c), why]
                    for c, why in useless[:12]]

    # table 4: by group
    rows_group = []
    for group in common.GROUPS:
        mine = [s for s in scans if s["item"]["group"] == group]
        if not mine:
            continue
        mine_ids = {id(s) for s in mine}
        in_group = [c for c in ranked if id(c["scan"]) in mine_ids]
        rows_group.append([GROUP_CN.get(group, group), f"{len(mine)} 個", f"{len(mine) * 2} 個",
                           f"{len(in_group)} 個", f"{sum(1 for c in in_group if c['p'] <= GOOD_P)} 個",
                           f"{sum(1 for c in in_group if _consistent(c))} 個",
                           f"{sum(1 for s in current if id(s) in mine_ids)} 個"])

    first_day = min(s["first"] for s in scans).strftime("%Y-%m-%d")
    last_day = max(s["last"] for s in scans).strftime("%Y-%m-%d")
    spans = sorted(s["ranked_days"] for s in scans)
    all_base = pd.concat([close.shift(-HORIZON, freq="D").reindex(close.index) / close - 1]).dropna()
    all_base = all_base[(all_base.index >= first_day) & (all_base.index <= last_day)]
    now_high = sum(1 for s in current if s["state"] == "high")
    now_low = len(current) - now_high
    now_good = [s for s in current if s[s["state"]]["ep_n"] >= MIN_EPISODES and s[s["state"]]["p"] <= GOOD_P
                and _consistent(s[s["state"]])]

    verdict = (f"一共測了 {len(scans)} 個指標、{len(combos)} 個「指標＋偏高或偏低」的組合，其中 {len(ranked)} 個有足夠多次"
               f"（至少 {MIN_EPISODES} 段）可以比較；看起來和之後 30 天漲跌有關的有 {len(good)} 個"
               f"（光靠運氣大約也會出現 {expected_good:.0f} 個），而且每一年方向都一致的只有 {len(good_steady)} 個。"
               f"現在有 {len(current)} 個指標處於極端（{now_high} 個偏高、{now_low} 個偏低），"
               f"其中歷史上站得住腳的有 {len(now_good)} 個。")

    now = [
        {"label": "測了幾個指標", "value": f"{len(scans)} 個（另外排除 {len(dropped)} 個）"},
        {"label": "測了幾個組合（每個指標分偏高、偏低）", "value": f"{len(combos)} 個"},
        {"label": f"分開發生過至少 {MIN_EPISODES} 段、可以排名的", "value": f"{len(ranked)} 個（樣本太少的 {len(thin)} 個）"},
        {"label": "看起來有關的組合", "value": f"{len(good)} 個（光靠運氣預期約 {expected_good:.1f} 個）"},
        {"label": "每一年、前後兩半方向都一致的組合", "value": f"{len(steady)} 個（光靠運氣預期約 {steady_range} 個）"},
        {"label": "兩個條件都符合的", "value": f"{len(good_steady)} 個"},
        {"label": "現在處於極端的指標", "value": f"{len(current)} 個（偏高 {now_high}、偏低 {now_low}）"},
        {"label": "平常的情況：比特幣 30 天後比較高", "value": f"{_per100((all_base > 0).mean())}，中位數 {_pct(all_base.median())}"},
        {"label": "統計期間", "value": f"{first_day} 到 {last_day}"},
    ]

    excluded: dict[str, list[str]] = {}
    for name, why in dropped:
        excluded.setdefault(why, []).append(name)
    excluded_text = "；".join(f"{why}：{'、'.join(names)}" for why, names in excluded.items()) or "沒有"
    notes = [
        f"做法（每個指標都一樣）：每一天只拿這個指標「到那一天為止」自己的歷史來比，至少要先有 {MIN_TRAIL} 天歷史才開始比，"
        "所以沒有用到未來的資料。落在自己歷史最低的一成算「偏低」，最高的一成算「偏高」。然後看比特幣 30 天後的收盤價比當天高還是低。",
        f"「平常」是指同一個指標、同一段期間裡所有日子的結果。這段期間（{first_day} 到 {last_day}）比特幣有一大段是大漲的，"
        f"所以平常就有{_per100((all_base > 0).mean())}是 30 天後比較高；「之後上漲」本身不稀奇，要看有沒有比平常多。",
        f"一次測了 {len(combos)} 個組合，測這麼多一定有一些只是剛好看起來很準。排名表用的門檻，純靠運氣大約也會有 "
        f"{expected_good:.0f} 個通過，實際通過 {len(good)} 個；每一年、前後兩半方向都一致的，純靠運氣大約也會有 "
        f"{steady_range} 個，實際 {len(steady)} 個。排在前面不等於真的有用。",
        "每天的 30 天結果互相重疊（今天和明天看的幾乎是同一段行情），所以「幾百天」其實沒有幾百次獨立的經驗。"
        f"因此排名只用「分開的次數」：連續處於極端的日子算同一段（中間斷 {GAP} 天以內也算同一段），每段只取第一天。"
        f"少於 {MIN_EPISODES} 段的不排名，標為「樣本太少」。相鄰兩段如果相隔不到 30 天，結果仍然會部分重疊。",
        "「看起來有關」的算法：每段第一天之後的 30 天漲跌，贏過平常中位數的次數如果偏離一半一半太多"
        f"（純丟銅板出現這種偏離的機會不到 {round(GOOD_P * 100)}%），就算看起來有關。這是很粗的檢查，不是嚴格的證明。",
        "全部都是事後回頭看同一段歷史（樣本內），門檻（一成、30 天、8 段）是事先定好、沒有為了好看去調，但也沒有用另一段資料驗證過。",
        f"各指標的歷史長度不同：儀表板每個指標最多保留約 {common.HISTORY_POINTS} 天，扣掉前面 {MIN_TRAIL} 天暖身，"
        f"每個指標實際可比的日子從 {spans[0]} 天到 {spans[-1]} 天不等，大多只涵蓋 3 到 4 個年份，"
        "所以「每一年都一致」其實只是三四年一致，不同指標之間也不能直接比。",
        "很多指標其實是同一件事的不同算法（例如好幾個都是「價格相對成本或均線有多高」），它們會一起偏高、一起偏低，"
        "不能當成好幾個獨立的證據。有些指標本身長期一路往上或往下（例如供給量、鎖倉金額），這類指標常常一直「偏高」，參考價值有限。",
        "每週末或假日沒有新數值的指標（例如美股、公債、ETF），用前一個交易日的數值補上，最多補 4 天。"
        + (f"資料超過 {STALE_DAYS} 天沒更新、所以不列入「現在」的有：{'、'.join(s['item']['name'] for s in stale)}。" if stale else ""),
        f"納入的 {len(scans)} 個指標：{'、'.join(s['item']['name'] for s in scans)}。",
        f"沒有納入的（只看比特幣和整體市場、歷史至少 {MIN_POINTS} 天的指標）：{excluded_text}。",
        "指標的歷史來自儀表板自己每天收集、存下來的資料，各指標的原始來源列在各自的分頁。",
        "過去的結果不保證未來會一樣。這裡只是整理歷史，不是預測，也不是任何買賣建議。",
    ]
    tables = [
        {"caption": f"現在處於極端的指標（共 {len(current)} 個），以及歷史上同一側之後 30 天比特幣的表現",
         "columns": ["指標", "現在數值", "現在位置", "歷史上發生過", "之後 30 天漲跌中位數", "之後 30 天比較高", "各年比較", "怎麼看"],
         "rows": rows_now},
        {"caption": f"歷史上最站得住腳的 12 個「指標＋方向」（依分開的次數來排，不是依重疊的天數；共 {len(ranked)} 個可排名）",
         "columns": ["指標", "哪一側", "之後 30 天", "分開發生幾次", "其中贏過平常", "每次開頭之後的中位數", "各年比較", "前後兩半"],
         "rows": rows_top},
        {"caption": "次數很多、但歷史上看不出用處的（沒有差別，或方向不穩定）",
         "columns": ["指標", "哪一側", "分開發生幾次", "其中贏過平常", "之後 30 天漲跌中位數", "之後 30 天比較高", "各年比較", "問題在哪"],
         "rows": rows_useless},
        {"caption": "依類別整理",
         "columns": ["類別", "指標數", "組合數", "可排名的組合", "看起來有關", "各年方向一致", "現在處於極端的指標"],
         "rows": rows_group},
    ]
    return {
        "id": "indicator_scan",
        "title": "指標偏到極端之後，比特幣通常怎麼走",
        "question": "儀表板上每個有長歷史的指標，偏到自己歷史的極端時，比特幣之後 30 天比較常漲還是跌？每一年都一樣，還是只是運氣？",
        "asof": close.index[-1].strftime("%Y-%m-%d"),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": f"比特幣每日收盤價：{price_source}", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2))
