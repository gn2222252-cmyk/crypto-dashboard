"""Historical study "episodes": what followed earlier sharp bitcoin sell-offs and six-week losing streaks.

Two questions, both answered with the whole public history and shown next to today's readings:

1. Every time since 2021 that bitcoin first fell 12% or more in 14 days (one sell-off = one episode: a new episode
   needs 30 days without a trigger day): the conditions on that day and the result 30 and 60 days later. Five
   yes/no traits are read from TODAY's values; past episodes are grouped by how many of those traits they share.
2. Every time since 2017 that the Binance weekly candle closed lower six weeks in a row: what followed.

Sources (public, no key): Binance spot klines and funding history, Deribit volatility index, alternative.me fear
and greed index, and the dashboard's own cache files for stablecoin supply, ETF flows, MVRV and short-term-holder
MVRV. A side source that fails leaves its column empty and adds a note. Everything is descriptive and in-sample:
few cases, traits chosen after looking at the data.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta, timezone

from dashboard import common

BINANCE = "https://api.binance.com/api/v3/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
FNG = "https://api.alternative.me/fng/"
DAY_MS = 86_400_000
DROP = -0.12        # trigger: 14-day change at or below this
WINDOW = 14
COOL_OFF = 30       # days without a trigger day before a new episode can start
SINCE = "2021-01-01"
BEAR = -0.35        # at least this far below the 1-year high
FEAR = 25
MVRV_LINE = 1.2
FLAT = 5.0          # |30-day change| below this many % counts as "about flat"
STREAK = 6
MAX_ROWS = 20
CACHE_IDS = {"stable": ("flows", "market_stablecoin_supply_change_30d"),
             "etf": ("flows", "btc_etf_flow_5d"),
             "mvrv": ("onchain", "btc_mvrv"),
             "sth": ("onchain2", "btc_sth_mvrv")}
CACHE_NAMES = {"stable": "穩定幣供給 30 天變化", "etf": "現貨 ETF 5 日淨流入", "mvrv": "MVRV", "sth": "短期持有者 MVRV"}


# ---------- data ----------

def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _klines(interval: str, start: str) -> list:
    """Closed Binance BTCUSDT candles from start, oldest first (the candle still forming is dropped)."""
    rows, since, now = [], _ms(start), int(time.time() * 1000)
    for _ in range(12):
        page = common.get_json(BINANCE, {"symbol": "BTCUSDT", "interval": interval, "limit": 1000,
                                         "startTime": since})
        if not page:
            break
        rows += page
        since = page[-1][0] + 1
        if len(page) < 1000:
            break
    return [r for r in rows if r[6] < now]


def _funding() -> dict:
    """{day: [funding rates settled that day]} for the Binance BTCUSDT perpetual, since 2019."""
    out: dict = {}
    since = _ms("2019-09-01")
    for _ in range(15):
        page = common.get_json(FUNDING, {"symbol": "BTCUSDT", "limit": 1000, "startTime": since})
        if not page:
            break
        for row in page:
            out.setdefault(_day(row["fundingTime"]), []).append(float(row["fundingRate"]))
        since = page[-1]["fundingTime"] + 1
        if len(page) < 1000:
            break
    return out


def _fng() -> dict:
    data = common.get_json(FNG, {"limit": 0})["data"]
    return {_day(int(r["timestamp"]) * 1000): int(r["value"]) for r in data}


def _dvol() -> dict:
    """{day: Deribit BTC volatility index close} since the index began (2021-03)."""
    out: dict = {}
    start, end = _ms("2021-01-01"), int(time.time() * 1000)
    for _ in range(6):
        result = common.get_json(DERIBIT, {"currency": "BTC", "resolution": "1D", "start_timestamp": start,
                                           "end_timestamp": end})["result"]
        for row in result.get("data") or []:
            out[_day(row[0])] = float(row[4])
        end = result.get("continuation")
        if not end or not result.get("data"):
            break
    if not out:
        raise RuntimeError("Deribit 沒有回傳波動率指數")
    return out


def _cached(file: str, id: str) -> dict:
    """{day: value} of one indicator in the dashboard's own cache (raises if the file or id is missing)."""
    data = json.loads((common.CACHE / f"{file}.json").read_text(encoding="utf-8"))
    rows = data.get("indicators", []) if isinstance(data, dict) else data
    history = next(r for r in rows if r.get("id") == id).get("history") or []
    out = {str(p[0])[:10]: float(p[1]) for p in history if len(p) >= 2 and common._finite(p[1])}
    if not out:
        raise RuntimeError("沒有歷史資料")
    return out


def _near(history: dict, day: str, back: int = 6):
    """The value on that day, or the latest one within `back` days before it (weekends, late sources)."""
    if not history:
        return None
    at = datetime.strptime(day, "%Y-%m-%d")
    for step in range(back + 1):
        key = (at - timedelta(days=step)).strftime("%Y-%m-%d")
        if key in history:
            return history[key]
    return None


# ---------- formatting ----------

def _pct(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return f"{text}%"
    return f"{'+' if value > 0 else '−'}{text}%"


def _num(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _usd(value) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _etf(value) -> str:
    """ETF flow is stored in million USD; shown in 億美元 (100 million)."""
    if value is None:
        return "—"
    text = f"{abs(value) / 100:.1f}"
    if float(text) == 0:
        return f"{text} 億"
    return f"{'+' if value > 0 else '−'}{text} 億"


def _median(values: list):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


# ---------- study 1: sharp sell-offs ----------

def _snapshot(i: int, d: dict) -> dict:
    """Conditions on day i and what followed."""
    days, high, low, close = d["days"], d["high"], d["low"], d["close"]
    day, price = days[i], close[i]
    rates = [x for j in range(7) for x in d["funding"].get(days[i - j], [])]
    snap = {"i": i, "date": day, "price": price,
            "chg14": (price / close[i - WINDOW] - 1) * 100,
            "from_high": (price / max(high[i - 365:i + 1]) - 1) * 100,
            "fng": d["fng"].get(day) if d["fng"] else None,
            "funding": sum(rates) / len(rates) * 3 * 365 * 100 if rates else None,
            "dvol": _near(d["dvol"], day),
            "stable": _near(d["stable"], day),
            "etf": _near(d["etf"], day),
            "mvrv": _near(d["mvrv"], day),
            "sth": _near(d["sth"], day, 12)}
    for horizon in (30, 60):
        done = i + horizon < len(days)
        snap[f"chg{horizon}"] = (close[i + horizon] / price - 1) * 100 if done else None
        snap[f"low{horizon}"] = (min(low[i + 1:i + horizon + 1]) / price - 1) * 100 if done else None
    return snap


def _traits(snap: dict) -> list:
    """Five yes/no traits (None when the value is not available for that day)."""
    return [snap["from_high"] <= BEAR * 100,
            None if snap["fng"] is None else snap["fng"] <= FEAR,
            None if snap["funding"] is None else snap["funding"] > 0,
            None if snap["etf"] is None else snap["etf"] < 0,
            None if snap["mvrv"] is None else snap["mvrv"] > MVRV_LINE]


def _trait_text(today: dict, flags: list) -> list:
    """Each trait stated as it is today."""
    out = []
    out.append(("熊市（比一年高點低 35% 以上）",
                f"{'是' if flags[0] else '不是'}，目前比一年高點低 {abs(today['from_high']):.0f}%"))
    out.append(("恐懼與貪婪指數在 25 以下（極度恐懼）",
                "沒有資料" if flags[1] is None else f"{'是' if flags[1] else '不是'}，目前 {today['fng']}"))
    out.append(("資金費率（做多的人付給做空的人的費用，正值代表做多的人較多）",
                "沒有資料" if flags[2] is None
                else f"{'正的' if flags[2] else '負的'}，近 7 天平均年化 {_pct(today['funding'], 1)}"))
    out.append(("現貨 ETF 近 5 個交易日是淨流出",
                "沒有資料" if flags[3] is None else f"{'是' if flags[3] else '不是'}，合計 {_etf(today['etf'])}美元"))
    out.append(("MVRV（現價 ÷ 全體持有者的平均成本）高於 1.2",
                "沒有資料" if flags[4] is None else f"{'是' if flags[4] else '不是'}，目前 {today['mvrv']:.2f}"))
    return out


def _group_row(label: str, group: list) -> list:
    done = [e for e in group if e["chg30"] is not None]
    done60 = [e for e in group if e["chg60"] is not None]
    up = sum(1 for e in done if e["chg30"] > 0)
    worst = min((e["low60"] for e in done60), default=None)
    months = "、".join(e["date"][:7] for e in group) if group else "沒有"
    return [label, f"{len(group)} 次", months,
            f"{len(done)} 次裡 {up} 次" if done else "—",
            _pct(_median([e["chg30"] for e in done])),
            _pct(_median([e["chg60"] for e in done60])),
            _pct(worst)]


# ---------- study 2: six-week losing streaks ----------

def _streaks(notes: list) -> dict:
    """Weekly closes: every time the streak of lower closes first reached six weeks, and the streak running now."""
    weeks = _klines("1w", "2017-08-01")
    end = [_day(w[0] + 6 * DAY_MS) for w in weeks]
    low = [float(w[3]) for w in weeks]
    close = [float(w[4]) for w in weeks]
    run, cases = 0, []
    for i in range(1, len(weeks)):
        run = run + 1 if close[i] < close[i - 1] else 0
        if run == STREAK:
            cases.append(i)
    rows = []
    for i in cases:
        def after(n: int, i: int = i):
            return (close[i + n] / close[i] - 1) * 100 if i + n < len(weeks) else None
        low8 = (min(low[i + 1:i + 9]) / close[i] - 1) * 100 if i + 8 < len(weeks) else None
        rows.append({"date": end[i], "close": close[i], "w4": after(4), "w8": after(8), "w13": after(13),
                     "low8": low8, "current": run >= STREAK and i == len(weeks) - 1 - (run - STREAK)})
    return {"rows": rows, "run": run, "last_week": end[-1] if end else None, "first_week": end[0] if end else None,
            "weeks": len(weeks)}


# ---------- the study ----------

def study() -> dict:
    notes: list = []
    daily = _klines("1d", "2019-12-25")
    if len(daily) < 800:
        raise RuntimeError(f"Binance 只回傳 {len(daily)} 根日線")
    d = {"days": [_day(k[0]) for k in daily], "high": [float(k[2]) for k in daily],
         "low": [float(k[3]) for k in daily], "close": [float(k[4]) for k in daily]}
    for key, label, fetch in (("funding", "資金費率", _funding), ("fng", "恐懼與貪婪指數", _fng),
                              ("dvol", "DVOL", _dvol)):
        try:
            d[key] = fetch()
        except Exception as error:  # noqa: BLE001 - a side source that fails leaves its column empty
            d[key] = {}
            notes.append(f"{label}這次抓不到（{type(error).__name__}），相關欄位顯示「—」。")
    for key, (file, id) in CACHE_IDS.items():
        try:
            d[key] = _cached(file, id)
        except Exception:  # noqa: BLE001
            d[key] = {}
            notes.append(f"儀表板快取裡找不到「{CACHE_NAMES[key]}」（{file}.json 的 {id}），該欄顯示「—」。")

    days, close = d["days"], d["close"]
    last = len(days) - 1
    asof = days[last]

    # episodes: first day the 14-day change is at or below -12%; a new one needs 30 days without a trigger day
    starts, previous = [], -10_000
    for i in range(365, len(days)):
        if days[i] < SINCE:
            continue
        if close[i] / close[i - WINDOW] - 1 <= DROP:
            if i - previous > COOL_OFF:
                starts.append(i)
            previous = i
    episodes = [_snapshot(i, d) for i in starts]
    today = _snapshot(last, d)
    in_trigger = today["chg14"] <= DROP * 100
    flags = _traits(today)
    usable = [j for j, flag in enumerate(flags) if flag is not None]
    for e in episodes:
        mine = _traits(e)
        e["same"] = sum(1 for j in usable if mine[j] is not None and mine[j] == flags[j])
        e["known"] = sum(1 for j in usable if mine[j] is not None)

    done = [e for e in episodes if e["chg30"] is not None]
    up = [e for e in done if e["chg30"] >= FLAT]
    down = [e for e in done if e["chg30"] <= -FLAT]
    flat = len(done) - len(up) - len(down)
    total = len(usable)
    past = done  # the sell-off still running (no 30-day result yet) is not compared with itself
    full = [e for e in past if e["same"] == total]
    near = [e for e in past if e["same"] == total - 1]
    rest = [e for e in past if e["same"] < total - 1]
    alike = full + near
    alike_done = [e for e in alike if e["chg30"] is not None]
    alike_up = sum(1 for e in alike_done if e["chg30"] > 0)

    # ---------- tables ----------
    shown = episodes[-MAX_ROWS:]
    if len(episodes) > MAX_ROWS:
        notes.append(f"急跌一共 {len(episodes)} 次，表格只列最近 {MAX_ROWS} 次。")
    table_state = {
        "caption": f"2021 年以來，每一次「14 天跌 12% 以上」當天的市場狀況（共 {len(episodes)} 次）",
        "columns": ["日期", "價格", "距一年高點", "恐懼指數", "資金費率(年化)", "DVOL", "穩定幣30天", "ETF 5日"],
        "rows": [[e["date"], _usd(e["price"]), _pct(e["from_high"]), _num(e["fng"], 0), _pct(e["funding"], 1),
                  _num(e["dvol"], 0), _pct(e["stable"], 1), _etf(e["etf"])] for e in shown]}
    table_after = {
        "caption": "同一批急跌：持有者成本指標、之後 30 天與 60 天的結果、和今天有幾項特徵相同",
        "columns": ["日期", "MVRV", "短期MVRV", "30天後", "30天內最低", "60天後", "60天內最低", "與今天相同"],
        "rows": [[e["date"], _num(e["mvrv"]), _num(e["sth"]), _pct(e["chg30"]), _pct(e["low30"]),
                  _pct(e["chg60"]), _pct(e["low60"]),
                  f"{e['same']}/{total}" + ("" if e["chg30"] is not None else "（未滿 30 天）")] for e in shown]}
    table_like = {
        "caption": f"和今天像不像：已滿 30 天的 {len(past)} 次，依「{total} 項特徵中有幾項與今天相同」分組",
        "columns": ["分組", "次數", "哪幾次", "30天後上漲", "30天後中位數", "60天後中位數", "60天內最深"],
        "rows": [_group_row(f"{total} 項全部相同", full),
                 _group_row(f"{total} 項中 {total - 1} 項相同", near),
                 _group_row("其餘（較不像今天）", rest),
                 _group_row("全部", past)]}
    tables = [table_state, table_after, table_like]

    # ---------- weekly streaks ----------
    streak_text, streak = "", None
    try:
        streak = _streaks(notes)
    except Exception as error:  # noqa: BLE001
        notes.append(f"週線這次抓不到（{type(error).__name__}），連六週下跌的部分無法計算。")
    if streak:
        earlier = [r for r in streak["rows"] if not r["current"]]
        tables.append({
            "caption": f"2017 年以來，週線連續 6 週收低的每一次（共 {len(streak['rows'])} 次，含目前這次）"
            if len(earlier) != len(streak["rows"]) else f"2017 年以來，週線連續 6 週收低的每一次（共 {len(earlier)} 次）",
            "columns": ["第 6 週收盤日", "收盤價", "4 週後", "8 週後", "13 週後", "8 週內最低"],
            "rows": [[r["date"] + ("（這次）" if r["current"] else ""), _usd(r["close"]), _pct(r["w4"]),
                      _pct(r["w8"]), _pct(r["w13"]), _pct(r["low8"])] for r in streak["rows"]][-MAX_ROWS:]})
        run = streak["run"]
        if run >= STREAK:
            running = f"目前正連續 {run} 週收低"
        elif run > 0:
            running = f"目前連續 {run} 週收低，還沒到 6 週"
        else:
            running = "目前沒有連跌，上一週是收高的"
        if earlier:
            detail = "、".join(f"{r['date'][:7]} 之後 8 週 {_pct(r['w8'])}" for r in earlier)
            streak_text = f"週線連續 6 週收低以前只出現過 {len(earlier)} 次（{detail}），{running}。"
        else:
            streak_text = f"週線連續 6 週收低在 2017 年以來以前沒有出現過，{running}。"

    # ---------- verdict ----------
    state = f"現在 14 天跌了 {abs(today['chg14']):.0f}%" if in_trigger else f"現在 14 天是 {_pct(today['chg14'])}、未達標準"
    verdict = (f"2021 年以來比特幣「14 天跌 12% 以上」有 {len(episodes)} 次（{state}），已滿 30 天的 {len(done)} 次裡 "
               f"{len(up)} 次漲超過 5%、{flat} 次大致持平、{len(down)} 次再跌超過 5%")
    if alike_done:
        verdict += (f"；和今天最像（{total} 項特徵至少 {total - 1} 項相同）的 {len(alike_done)} 次裡 {alike_up} 次在 30 天後是漲的，"
                    f"中位數 {_pct(_median([e['chg30'] for e in alike_done]))}。")
    else:
        verdict += f"；過去沒有任何一次和今天有 {total - 1} 項以上特徵相同，沒有夠像的前例。"
    verdict += streak_text

    # ---------- today's readings ----------
    now = [{"label": "價格（最近一根收完的日線）", "value": _usd(today["price"])},
           {"label": "14 天漲跌", "value": _pct(today["chg14"], 1) + ("（達到急跌標準）" if in_trigger else "（未達急跌標準 −12%）")}]
    now += [{"label": label, "value": value} for label, value in _trait_text(today, flags)]
    now += [{"label": "DVOL（選擇權市場預期未來 30 天的波動，年化 %）", "value": _num(today["dvol"], 0)},
            {"label": "穩定幣總供給 30 天變化", "value": _pct(today["stable"], 1)},
            {"label": "短期持有者 MVRV（現價 ÷ 近 5 個月買家的平均成本）", "value": _num(today["sth"])}]
    if streak:
        now.append({"label": "週線連續收低", "value": f"{streak['run']} 週（最近一根收完的週線：{streak['last_week']}）"})

    # ---------- notes ----------
    missing_old = [e for e in episodes if e["known"] < total]
    notes = [
        f"急跌只有 {len(episodes)} 次、連六週下跌只有個位數次，樣本很少，任何比例都可能只是巧合。",
        "這是事後回頭看同一段歷史（樣本內），沒有用另一段沒看過的資料驗證。",
        "五項特徵和門檻（35%、25、1.2 等）是看過資料之後才挑的，換一組門檻分組結果會不同。",
        "過去的結果不代表之後也會一樣，這裡只描述歷史，不是買賣建議。",
        "急跌的定義：收盤價第一次比 14 天前低 12% 以上；之後要連續 30 天沒有再達標，才算下一次，所以一段連續下跌只算一次。",
        "「30 天內最低」是之後 30 天裡盤中最低價相對急跌當天收盤的幅度；「—」代表沒有資料或時間還沒到。",
        "資金費率是幣安 BTCUSDT 永續合約近 7 天平均換算成年化；MVRV、穩定幣、ETF 取當天或往前最多 6 天（短期持有者 MVRV 最多 12 天）的最近一筆。",
    ] + ([f"有 {len(missing_old)} 次急跌缺少部分特徵的資料（儀表板的 MVRV 與穩定幣資料大約從 2022 年 10 月才有，現貨 ETF 2024 年 1 月才上市），"
          "缺的特徵一律不算「相同」，所以較早的幾次不可能被分到最像今天的組。"] if missing_old else []) + [
        "連六週下跌用幣安週線（週一開盤、週日收盤，2017 年 8 月起），收盤低於前一週收盤就算一週下跌；連跌超過 6 週仍只算一次。",
    ] + notes
    if flags[4] is not None and abs(today["mvrv"] - MVRV_LINE) < 0.05:
        notes.append(f"今天的 MVRV 是 {today['mvrv']:.2f}，離分界 1.2 很近，稍微變動這一項的判定就會翻過來。")
    if abs(today["from_high"] - BEAR * 100) < 3:
        notes.append(f"今天比一年高點低 {abs(today['from_high']):.0f}%，離「熊市」分界 35% 很近，價格稍微變動這一項的判定就會翻過來。")

    return {"id": "episodes",
            "title": "急跌之後與連跌六週之後",
            "question": "比特幣 14 天跌 12% 以上、或週線連跌六週之後，以前接下來怎麼走，哪幾次和今天最像？",
            "asof": asof,
            "verdict": verdict,
            "now": now,
            "tables": tables,
            "notes": notes,
            "sources": [
                {"name": "幣安現貨 BTCUSDT 日線與週線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
                {"name": "幣安 BTCUSDT 永續合約資金費率", "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
                {"name": "Deribit 比特幣波動率指數（DVOL）", "url": "https://www.deribit.com/statistics/BTC/volatility-index"},
                {"name": "alternative.me 恐懼與貪婪指數", "url": "https://alternative.me/crypto/fear-and-greed-index/"},
                {"name": "本儀表板快取：穩定幣供給、現貨 ETF 淨流入、MVRV、短期持有者 MVRV", "url": ""},
            ]}


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
