"""選擇權與跨交易所衍生品收集器（group="derivatives"，存成 cache/options.json）。規格見 dashboard/SPEC2.md。

和 derivatives.py 不重複：那邊是幣安的資金費率、未平倉量、溢價、基差、多空比，以及 Deribit 的波動率指數與賣權買權未平倉比。
這裡補上：
- Deribit 選擇權（比特幣、以太幣，以及用 USDC 結算的索拉納）：25 delta 風險逆轉、隱含波動率期限結構、最大痛點、
  已實現波動率與波動率溢價。delta 用 Black-Scholes（利率 0，和 Deribit 自己的算法相同）從每個合約的隱含波動率算出來，
  所以只要一次 get_book_summary_by_currency，不必逐一呼叫 ticker。
- 跨交易所永續合約（幣安、Bybit、OKX、Bitget、Hyperliquid）：資金費率平均與分歧、未平倉量合計、
  Hyperliquid 單獨的資金費率與未平倉量、Bybit 未平倉量長歷史、OKX 的多空人數比／持倉成交量比／槓桿借幣比。
- OKX 公開的強制平倉（爆倉）單：近 24 小時多單與空單爆倉金額。只有 OKX 一家；幣安與 Bybit 只在即時串流提供。

全部免費、公開、不需金鑰，只用 common.get_json（固定 User-Agent）。被擋就記為失敗。
來源只給目前數值或短歷史的指標，用 common.accumulate 每天存一筆，歷史會自己變長。
試過但拿不到：CME 比特幣期貨價格（stooq 要通過瀏覽器驗證、CME 官網直接封鎖程式），所以沒有 CME 基差。
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common

GROUP = "derivatives"
ASSETS = ("BTC", "ETH", "SOL")
CN = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "索拉納"}
DERIBIT = "https://www.deribit.com/api/v2/public"
BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com/api/v5"
BITGET = "https://api.bitget.com/api/v2/mix/market"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"
DERIBIT_NAME = "Deribit 選擇權公開資料（自行換算）"
CROSS_NAME = "幣安、Bybit、OKX、Bitget、Hyperliquid 公開資料（自行彙整）"
OKX_NAME = "OKX 公開統計資料"
HL_NAME = "Hyperliquid 公開資料"
BYBIT_NAME = "Bybit 永續合約公開資料"
CROSS_URL = "https://www.coinglass.com/zh-TW/FundingRate"
OKX_URL = "https://www.okx.com/zh-hant/markets/data/contracts"
HL_URL = "https://app.hyperliquid.xyz/trade"
NO_RULE = "沒有事前固定的門檻，不給方向。"
SNAPSHOT = "來源只給目前數值，走勢圖是本頁每次更新時自己記下來的，所以一開始很短。"
DAY_MS = 86_400_000
YEAR_DAYS = 365.0
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
DEEP_BELOW = 1200  # a stored history shorter than this triggers the one-off deep backfill
LIQ_PAGES = 45
EXCHANGES = ("幣安", "Bybit", "OKX", "Bitget", "Hyperliquid")


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp() * 1000


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ms: float) -> str:
    return datetime.fromtimestamp(float(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def _yi(usd: float) -> str:
    return f"{usd / 1e8:,.2f} 億美元"


def _wan(usd: float) -> str:
    return f"{usd / 1e4:,.0f} 萬美元"


def _post_json(url: str, body: dict, timeout: int = 20):
    """POST a public JSON endpoint with the same fixed user agent and pause as common.get_json."""
    reply = requests.post(url, json=body, headers=common.HEADERS, timeout=timeout)
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    return reply.json()


def _stored_len(id: str) -> int:
    return len(common.accumulate(id, []))


def _add(rows: list, *, id: str, asset: str, name: str, source: str, url: str, build) -> None:
    """Run one indicator builder inside its own try/except so one failure never drags others down."""
    try:
        fields = build()
        fields.setdefault("source", source)
        fields.setdefault("source_url", url)
        fields.setdefault("evidence", "C")
        fields.setdefault("reading", "none")
        fields.setdefault("rule", NO_RULE)
        rows.append(common.indicator(id=id, group=GROUP, asset=asset, name=name, **fields))
    except Exception as exc:  # noqa: BLE001
        rows.append(common.failed(id=id, group=GROUP, asset=asset, name=name, source=source, source_url=url,
                                  error=f"{type(exc).__name__}: {exc}"))


class _Memo:
    """Fetch once, remember the result or the error, and replay it to every indicator that needs it."""

    def __init__(self):
        self._done: dict = {}

    def get(self, key: str, fetch):
        if key not in self._done:
            try:
                self._done[key] = (fetch(), None)
            except Exception as exc:  # noqa: BLE001
                self._done[key] = (None, exc)
        value, error = self._done[key]
        if error is not None:
            raise error
        return value


# =====================================================================================================
# Deribit 選擇權
# =====================================================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _parse_instrument(name: str):
    """'BTC-27NOV26-75000-C' or 'SOL_USDC-25DEC26-165d5-P' -> (expiry ms at 08:00 UTC, strike, 'C'/'P')."""
    _, date, strike, kind = name.split("-")
    day, month, year = int(date[:-5]), MONTHS[date[-5:-2]], 2000 + int(date[-2:])
    expiry = datetime(year, month, day, 8, tzinfo=timezone.utc)
    return expiry.timestamp() * 1000, float(strike.replace("d", ".")), kind


def _is_monthly(expiry_ms: float) -> bool:
    """Deribit monthly expiries are the last Friday of the month."""
    d = datetime.fromtimestamp(expiry_ms / 1000, timezone.utc)
    return d.weekday() == 4 and datetime.fromtimestamp((expiry_ms + 7 * DAY_MS) / 1000, timezone.utc).month != d.month


def _chain(asset: str) -> dict:
    """All listed options of one asset grouped by expiry, with the per-expiry numbers the indicators need."""
    currency = asset if asset in ("BTC", "ETH") else "USDC"
    result = common.get_json(f"{DERIBIT}/get_book_summary_by_currency", {"currency": currency, "kind": "option"},
                             timeout=30)["result"]
    prefix = f"{asset}-" if currency != "USDC" else f"{asset}_USDC-"
    now = _now_ms()
    expiries: dict = {}
    spot = None
    for r in result:
        name = r.get("instrument_name", "")
        if not name.startswith(prefix):
            continue
        try:
            expiry, strike, kind = _parse_instrument(name)
        except Exception:  # noqa: BLE001 - an unexpected instrument name is skipped, not fatal
            continue
        spot = r.get("estimated_delivery_price") or spot
        e = expiries.setdefault(expiry, {"forwards": [], "iv": {}, "call_oi": {}, "put_oi": {}})
        if r.get("underlying_price"):
            e["forwards"].append(float(r["underlying_price"]))
        iv = r.get("mark_iv")
        if iv and iv > 0:
            e["iv"][strike] = float(iv)
        side = "call_oi" if kind == "C" else "put_oi"
        e[side][strike] = e[side].get(strike, 0.0) + float(r.get("open_interest") or 0)
    if not expiries or not spot:
        raise RuntimeError(f"Deribit 沒有回傳 {asset} 選擇權資料")
    out = []
    for expiry in sorted(expiries):
        e = expiries[expiry]
        days = (expiry - now) / DAY_MS
        if days <= 0 or not e["forwards"] or len(e["iv"]) < 5:
            continue
        forward = float(np.median(e["forwards"]))
        strikes = np.array(sorted(e["iv"]))
        ivs = np.array([e["iv"][k] for k in strikes])
        t = days / YEAR_DAYS
        d1 = (np.log(forward / strikes) + 0.5 * (ivs / 100) ** 2 * t) / ((ivs / 100) * math.sqrt(t))
        call_delta = np.array([_norm_cdf(x) for x in d1])
        atm = float(np.interp(forward, strikes, ivs)) if strikes[0] <= forward <= strikes[-1] else None

        def at_delta(delta: np.ndarray):
            keep = (delta > 0.02) & (delta < 0.6)
            ds, vs = delta[keep], ivs[keep]
            if len(ds) < 2 or ds.min() > 0.25 or ds.max() < 0.25:
                return None
            order = np.argsort(ds)
            return float(np.interp(0.25, ds[order], vs[order]))

        call25, put25 = at_delta(call_delta), at_delta(1.0 - call_delta)
        out.append({"expiry": expiry, "days": days, "forward": forward, "atm": atm, "call25": call25, "put25": put25,
                    "rr": None if call25 is None or put25 is None else call25 - put25,
                    "call_oi": e["call_oi"], "put_oi": e["put_oi"], "monthly": _is_monthly(expiry)})
    if not out:
        raise RuntimeError(f"Deribit 的 {asset} 選擇權沒有可用的到期日")
    return {"spot": float(spot), "expiries": out}


def _at_days(expiries: list, key: str, days: float, variance: bool):
    """Interpolate a per-expiry number to a constant maturity (expiries under one day are too noisy to use)."""
    points = [(e["days"], e[key]) for e in expiries if e["days"] >= 1 and e[key] is not None]
    if not points:
        return None
    if days <= points[0][0]:
        return points[0][1] if points[0][0] <= days * 2 else None
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= days <= t1:
            share = (days - t0) / (t1 - t0)
            if not variance:
                return v0 + (v1 - v0) * share
            total = v0 ** 2 * t0 + (v1 ** 2 * t1 - v0 ** 2 * t0) * share
            return math.sqrt(max(total, 0.0) / days)
    return None


def _front_monthly(expiries: list) -> dict:
    for e in expiries:
        if e["monthly"] and e["days"] >= 2:
            return e
    raise RuntimeError("找不到兩天以後的月結算日")


def _max_pain(e: dict) -> float:
    strikes = sorted(set(e["call_oi"]) | set(e["put_oi"]))
    if sum(e["call_oi"].values()) + sum(e["put_oi"].values()) <= 0:
        raise RuntimeError("這個結算日沒有未平倉量")
    best, best_pain = None, None
    for s in strikes:
        pain = (sum(oi * max(s - k, 0.0) for k, oi in e["call_oi"].items())
                + sum(oi * max(k - s, 0.0) for k, oi in e["put_oi"].items()))
        if best_pain is None or pain < best_pain:
            best, best_pain = s, pain
    return float(best)


def _perp_closes(asset: str) -> pd.Series:
    """Daily closes of the Deribit perpetual (about 1,900 days), index YYYY-MM-DD."""
    instrument = f"{asset}-PERPETUAL" if asset in ("BTC", "ETH") else f"{asset}_USDC-PERPETUAL"
    now = int(_now_ms())
    r = common.get_json(f"{DERIBIT}/get_tradingview_chart_data",
                        {"instrument_name": instrument, "resolution": "1D",
                         "start_timestamp": now - (common.HISTORY_POINTS + 420) * DAY_MS, "end_timestamp": now},
                        timeout=30)["result"]
    if not r.get("ticks"):
        raise RuntimeError("Deribit 沒有回傳日線")
    series = pd.Series(r["close"], index=[_day(t) for t in r["ticks"]], dtype=float)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _dvol(asset: str) -> pd.Series:
    now = int(_now_ms())
    start, end, data = now - (common.HISTORY_POINTS + 5) * DAY_MS, now, []
    for _ in range(4):
        result = common.get_json(f"{DERIBIT}/get_volatility_index_data",
                                 {"currency": asset, "resolution": "1D", "start_timestamp": start,
                                  "end_timestamp": end}, timeout=20)["result"]
        data += result["data"]
        end = result.get("continuation")
        if not end or not result["data"]:
            break
    if not data:
        raise RuntimeError("Deribit 沒有回傳波動率指數")
    series = pd.Series({_day(r[0]): float(r[4]) for r in sorted(data)})
    return series.sort_index()


def _option_rows(asset: str) -> list:
    rows: list = []
    a, cn = asset.lower(), CN[asset]
    memo = _Memo()
    url = f"https://www.deribit.com/statistics/{asset}/volatility-index" if asset != "SOL" \
        else "https://www.deribit.com/options/SOL_USDC"
    oi_url = f"https://www.deribit.com/statistics/{asset}/options-open-interest" if asset != "SOL" else url
    today = common.today_utc()
    thin = "（索拉納選擇權在 Deribit 的成交很少，報價寬，數字比比特幣、以太幣粗糙。）" if asset == "SOL" else ""

    def chain() -> dict:
        return memo.get("chain", lambda: _chain(asset))

    def closes() -> pd.Series:
        return memo.get("closes", lambda: _perp_closes(asset))

    def realized() -> pd.Series:
        returns = np.log(closes() / closes().shift(1))
        return (returns.rolling(30).std() * math.sqrt(YEAR_DAYS) * 100).dropna()

    rr_note = ("風險逆轉＝價外買權（看漲用）的隱含波動率減掉價外賣權（看跌或避險用）的隱含波動率，兩邊都取 delta 約 0.25 的履約價。"
               "負值代表市場願意多付錢買下跌保險（偏恐慌），正值代表搶買上漲（偏貪婪）。代表的是 Deribit 上的選擇權交易者，"
               "以機構與專業戶為主。限制：本頁用 Deribit 的標記波動率自己內插出來，和其他網站的數字會有小差異；" + SNAPSHOT + thin)

    def rr_30d() -> dict:
        v = _at_days(chain()["expiries"], "rr", 30, variance=False)
        if v is None:
            raise RuntimeError("30 天前後的到期日算不出 25 delta 波動率")
        return dict(value=round(v, 3), unit="個百分點", asof=common.now_utc(),
                    display=f"{v:+.2f} 個百分點（{'賣權較貴，偏避險' if v < 0 else '買權較貴，偏追漲'}）",
                    rule="正數是買權比較貴，負數是賣權比較貴。" + NO_RULE, note="把前後兩個到期日內插成固定 30 天。" + rr_note,
                    history=common.accumulate(f"{a}_risk_reversal_30d", [[today, round(v, 3)]]))

    def rr_front() -> dict:
        e = _front_monthly(chain()["expiries"])
        if e["rr"] is None:
            raise RuntimeError("最近的月結算日算不出 25 delta 波動率")
        v = e["rr"]
        return dict(value=round(v, 3), unit="個百分點", asof=common.now_utc(),
                    display=f"{v:+.2f} 個百分點（{_day(e['expiry'])} 到期，買權 {e['call25']:.1f}%、賣權 {e['put25']:.1f}%）",
                    rule="正數是買權比較貴，負數是賣權比較貴。" + NO_RULE,
                    note="用最近一個月結算日（每月最後一個星期五）的合約；換月時數字會跳一下。" + rr_note,
                    history=common.accumulate(f"{a}_risk_reversal_front", [[today, round(v, 3)]]))

    def term() -> tuple:
        ex = chain()["expiries"]
        ivs = {d: _at_days(ex, "atm", d, variance=True) for d in (7, 30, 90)}
        return ivs

    def atm_30d() -> dict:
        v = term()[30]
        if v is None:
            raise RuntimeError("30 天前後的到期日算不出平價隱含波動率")
        return dict(value=round(v, 2), unit="%（年化）", asof=common.now_utc(), display=f"{v:.1f}%",
                    note=f"履約價等於目前遠期價格的{cn}選擇權隱含波動率，內插成固定 30 天：市場預期未來一個月價格會晃動多大（年化）。"
                         "數字高代表預期大波動，看不出方向。" + ("和 Deribit 波動率指數很接近，但這裡只看平價合約。" if asset != "SOL" else "")
                         + SNAPSHOT + thin,
                    history=common.accumulate(f"{a}_atm_iv_30d", [[today, round(v, 2)]]))

    def term_slope() -> dict:
        ivs = term()
        if ivs[7] is None or ivs[90] is None:
            raise RuntimeError("7 天或 90 天的平價隱含波動率算不出來")
        v = ivs[90] - ivs[7]
        mid = "" if ivs[30] is None else f"、30 天 {ivs[30]:.1f}%"
        return dict(value=round(v, 3), unit="個百分點", asof=common.now_utc(),
                    display=f"{v:+.2f} 個百分點（7 天 {ivs[7]:.1f}%{mid}、90 天 {ivs[90]:.1f}%）",
                    rule="正數是遠期比近期高（平常的樣子），負數是近期比遠期高（倒掛）。" + NO_RULE,
                    note="90 天平價隱含波動率減 7 天的。平常遠期比較高；變成負值（倒掛）代表市場正在為眼前的大波動付高價，"
                         "常出現在急跌或重大事件前後。限制：7 天那端受週末與事件影響很大。" + SNAPSHOT + thin,
                    history=common.accumulate(f"{a}_iv_term_slope", [[today, round(v, 3)]]))

    def max_pain() -> dict:
        c = chain()
        e = _front_monthly(c["expiries"])
        pain = _max_pain(e)
        gap = (pain / c["spot"] - 1) * 100
        total = sum(e["call_oi"].values()) + sum(e["put_oi"].values())
        return dict(value=round(pain, 2), unit="美元", asof=common.now_utc(),
                    display=f"{pain:,.0f} 美元（現價 {c['spot']:,.0f}，{'高' if gap >= 0 else '低'}於現價 {abs(gap):.1f}%；"
                            f"{_day(e['expiry'])} 結算，未平倉 {total:,.0f} 顆）",
                    note="最大痛點＝假如結算在這個價位，所有選擇權買方合計拿回的錢最少（賣方賠最少）。"
                         "有人認為結算前價格會被拉向這裡，但這只是說法，沒有可靠證據，只當參考。"
                         "用最近一個月結算日、Deribit 一家的未平倉量計算；換月時會跳。" + SNAPSHOT + thin,
                    source_url=oi_url,
                    history=common.accumulate(f"{a}_max_pain", [[today, round(pain, 2)]]))

    def realized_vol() -> dict:
        rv = realized()
        v = float(rv.iloc[-1])
        return dict(value=round(v, 2), unit="%（年化）", asof=common.now_utc(), display=f"{v:.1f}%",
                    note=f"{cn}過去 30 天每日漲跌幅的標準差換算成年化：價格實際晃動多大。用 Deribit 永續合約的日收盤價計算。"
                         "數字低代表市場很安靜（常在大行情之前），數字高代表剛經歷劇烈波動；看不出方向。"
                         "最新一天用的是還沒收盤的價格。",
                    source="Deribit 永續合約日線（自行換算）",
                    history=[[d, round(float(x), 2)] for d, x in rv.items()])

    def vol_premium() -> dict:
        rv = realized()
        if asset == "SOL":
            iv = term()[30]
            if iv is None:
                raise RuntimeError("30 天平價隱含波動率算不出來")
            v = iv - float(rv.iloc[-1])
            history = common.accumulate(f"{a}_vol_premium", [[today, round(v, 2)]])
            what, tail = "30 天平價隱含波動率", SNAPSHOT + thin
            shown = f"隱含 {iv:.1f}%、已實現 {float(rv.iloc[-1]):.1f}%"
        else:
            dvol = memo.get("dvol", lambda: _dvol(asset))
            both = pd.concat([dvol, rv], axis=1, keys=["iv", "rv"]).dropna()
            if both.empty:
                raise RuntimeError("波動率指數與已實現波動率沒有重疊的日期")
            spread = both["iv"] - both["rv"]
            v = float(spread.iloc[-1])
            history = [[d, round(float(x), 2)] for d, x in spread.items()]
            what, tail = "Deribit 波動率指數（預期未來 30 天）", ""
            shown = f"隱含 {both['iv'].iloc[-1]:.1f}%、已實現 {both['rv'].iloc[-1]:.1f}%"
        return dict(value=round(v, 2), unit="個百分點", asof=common.now_utc(), display=f"{v:+.1f} 個百分點（{shown}）",
                    rule="正數是選擇權比實際波動貴，負數是比實際波動便宜。" + NO_RULE,
                    note=f"{what}減掉過去 30 天實際的波動率。平常是正的（買保險要付溢價）；很高代表市場很怕、保險很貴，"
                         "負值代表實際波動已經超過市場原先預期（常在急漲急跌當下）。"
                         "限制：一個往前看、一個往回看，兩個 30 天不是同一段時間。" + tail,
                    history=history)

    _add(rows, id=f"{a}_risk_reversal_30d", asset=asset, name=f"{cn}選擇權風險逆轉（30 天，買權減賣權）",
         source=DERIBIT_NAME, url=url, build=rr_30d)
    _add(rows, id=f"{a}_risk_reversal_front", asset=asset, name=f"{cn}選擇權風險逆轉（最近月結算）",
         source=DERIBIT_NAME, url=url, build=rr_front)
    _add(rows, id=f"{a}_atm_iv_30d", asset=asset, name=f"{cn}選擇權 30 天平價隱含波動率",
         source=DERIBIT_NAME, url=url, build=atm_30d)
    _add(rows, id=f"{a}_iv_term_slope", asset=asset, name=f"{cn}隱含波動率期限結構（90 天減 7 天）",
         source=DERIBIT_NAME, url=url, build=term_slope)
    _add(rows, id=f"{a}_max_pain", asset=asset, name=f"{cn}選擇權最大痛點價格（最近月結算）",
         source=DERIBIT_NAME, url=oi_url, build=max_pain)
    _add(rows, id=f"{a}_realized_vol_30d", asset=asset, name=f"{cn} 30 天已實現波動率",
         source=DERIBIT_NAME, url=url, build=realized_vol)
    _add(rows, id=f"{a}_vol_premium", asset=asset, name=f"{cn}波動率溢價（隱含減已實現）",
         source=DERIBIT_NAME, url=url, build=vol_premium)
    return rows


# =====================================================================================================
# 跨交易所永續合約
# =====================================================================================================

def _bybit(path: str, params: dict) -> dict:
    reply = common.get_json(BYBIT + path, params, timeout=15)
    if reply.get("retCode") != 0:
        raise RuntimeError(f"Bybit 回報錯誤：{reply.get('retMsg')}")
    return reply["result"]


def _okx(path: str, params: dict) -> list:
    reply = common.get_json(OKX + path, params, timeout=15)
    if str(reply.get("code")) != "0":
        raise RuntimeError(f"OKX 回報錯誤：{reply.get('msg')}")
    return reply["data"]


def _annual(rate: float, hours: float) -> float:
    return rate * (24.0 / hours) * 365 * 100


def _daily_mean(points: list, hours: float, min_count: int = 1) -> pd.Series:
    """[(ms, rate per interval)] -> annualised % averaged per UTC day."""
    if not points:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(points, columns=["ms", "rate"]).drop_duplicates("ms")
    frame["day"] = [_day(ms) for ms in frame["ms"]]
    grouped = frame.groupby("day")["rate"]
    mean = grouped.mean()[grouped.count() >= min_count]
    return mean * (24.0 / hours) * 365 * 100


def _exchange_rows(asset: str) -> list:
    rows: list = []
    a, cn = asset.lower(), CN[asset]
    symbol, okx_id = f"{asset}USDT", f"{asset}-USDT-SWAP"
    memo = _Memo()
    today = common.today_utc()

    # ---- snapshots ---------------------------------------------------------------------------------------
    def snap_binance() -> dict:
        p = common.get_json(f"{BINANCE}/fapi/v1/premiumIndex", {"symbol": symbol}, timeout=15)
        oi = common.get_json(f"{BINANCE}/fapi/v1/openInterest", {"symbol": symbol}, timeout=15)
        return {"funding": _annual(float(p["lastFundingRate"]), 8),
                "oi": float(oi["openInterest"]) * float(p["markPrice"])}

    def snap_bybit() -> dict:
        t = _bybit("/v5/market/tickers", {"category": "linear", "symbol": symbol})["list"][0]
        hours = float(t.get("fundingIntervalHour") or 8)
        return {"funding": _annual(float(t["fundingRate"]), hours), "oi": float(t["openInterestValue"])}

    def snap_okx() -> dict:
        f = _okx("/public/funding-rate", {"instId": okx_id})[0]
        hours = (float(f["fundingTime"]) - float(f["prevFundingTime"])) / 3_600_000 if f.get("prevFundingTime") else 8
        oi = _okx("/public/open-interest", {"instType": "SWAP", "instId": okx_id})[0]
        return {"funding": _annual(float(f["fundingRate"]), hours or 8), "oi": float(oi["oiUsd"])}

    def snap_bitget() -> dict:
        params = {"productType": "USDT-FUTURES", "symbol": symbol}
        t = common.get_json(f"{BITGET}/ticker", params, timeout=15)["data"][0]
        f = common.get_json(f"{BITGET}/current-fund-rate", params, timeout=15)["data"][0]
        return {"funding": _annual(float(f["fundingRate"]), float(f.get("fundingRateInterval") or 8)),
                "oi": float(t["holdingAmount"]) * float(t["markPrice"])}

    def snap_hl() -> dict:
        meta, ctxs = _post_json(HYPERLIQUID, {"type": "metaAndAssetCtxs"})
        names = [u["name"] for u in meta["universe"]]
        c = ctxs[names.index(asset)]
        return {"funding": _annual(float(c["funding"]), 1), "oi": float(c["openInterest"]) * float(c["markPx"])}

    fetchers = dict(zip(EXCHANGES, (snap_binance, snap_bybit, snap_okx, snap_bitget, snap_hl)))

    def snaps() -> dict:
        def fetch():
            got, errors = {}, {}
            for name, fn in fetchers.items():
                try:
                    got[name] = memo.get(f"snap_{name}", fn)
                except Exception as exc:  # noqa: BLE001
                    errors[name] = f"{type(exc).__name__}"
            return got, errors
        return memo.get("snaps", fetch)

    def missing_words(errors: dict) -> str:
        return "" if not errors else f"；抓不到：{'、'.join(errors)}"

    # ---- funding history ---------------------------------------------------------------------------------
    def binance_funding(deep: bool) -> pd.Series:
        points, start = [], int(_now_ms()) - (common.HISTORY_POINTS + 5) * DAY_MS
        for _ in range(6 if deep else 1):
            params = {"symbol": symbol, "limit": 1000}
            if deep:
                params["startTime"] = start
            page = common.get_json(f"{BINANCE}/fapi/v1/fundingRate", params, timeout=20)
            points += [(float(r["fundingTime"]), float(r["fundingRate"])) for r in page]
            if len(page) < 1000:
                break
            start = int(page[-1]["fundingTime"]) + 1
        if not points:
            return pd.Series(dtype=float)
        # daily mean of the per-settlement rate times settlements per day = the day's total funding
        frame = pd.DataFrame(points, columns=["ms", "rate"]).drop_duplicates("ms")
        frame["day"] = [_day(ms) for ms in frame["ms"]]
        return frame.groupby("day")["rate"].sum() * 365 * 100

    def bybit_funding(deep: bool) -> pd.Series:
        points, end = [], None
        for _ in range(24 if deep else 2):
            params = {"category": "linear", "symbol": symbol, "limit": 200}
            if end:
                params["endTime"] = end
            page = _bybit("/v5/market/funding/history", params)["list"]
            points += [(float(r["fundingRateTimestamp"]), float(r["fundingRate"])) for r in page]
            if len(page) < 200:
                break
            end = int(page[-1]["fundingRateTimestamp"]) - 1
        if not points:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(points, columns=["ms", "rate"]).drop_duplicates("ms")
        frame["day"] = [_day(ms) for ms in frame["ms"]]
        return frame.groupby("day")["rate"].sum() * 365 * 100

    def okx_funding() -> pd.Series:
        points, after = [], None
        for _ in range(3):
            params = {"instId": okx_id, "limit": 100}
            if after:
                params["after"] = after
            page = _okx("/public/funding-rate-history", params)
            points += [(float(r["fundingTime"]), float(r["realizedRate"] or r["fundingRate"])) for r in page]
            if len(page) < 100:
                break
            after = page[-1]["fundingTime"]
        if not points:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(points, columns=["ms", "rate"]).drop_duplicates("ms")
        frame["day"] = [_day(ms) for ms in frame["ms"]]
        return frame.groupby("day")["rate"].sum() * 365 * 100

    def hl_funding_history() -> list:
        """Hourly Hyperliquid funding as stored daily means; each run also walks about 100 days further back."""
        def fetch():
            id = f"{a}_hl_funding"
            stored = common.accumulate(id, [])
            now = _now_ms()

            def pages(start: float, count: int) -> list:
                points = []
                for _ in range(count):
                    page = _post_json(HYPERLIQUID, {"type": "fundingHistory", "coin": asset, "startTime": int(start)})
                    points += [(float(r["time"]), float(r["fundingRate"])) for r in page]
                    if len(page) < 500:
                        break
                    start = page[-1]["time"] + 1
                return points

            if stored:
                last = datetime.strptime(stored[-1][0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
                points = pages(max(last - 2 * DAY_MS, now - 60 * DAY_MS), 3)
                first = datetime.strptime(stored[0][0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
                if len(stored) < common.HISTORY_POINTS and first > 1_683_849_600_000 + DAY_MS:  # launch 2023-05-12
                    points += pages(first - 104 * DAY_MS, 5)
            else:
                points = pages(now - 124 * DAY_MS, 6)
            daily = _daily_mean(points, 1, min_count=20)
            fresh = [[d, round(float(v), 3)] for d, v in daily.items() if d != today]
            return common.accumulate(id, fresh)
        return memo.get("hl_history", fetch)

    def funding_avg() -> dict:
        got, errors = snaps()
        rates = {name: s["funding"] for name, s in got.items()}
        if len(rates) < 3:
            raise RuntimeError(f"只抓到 {len(rates)} 家交易所的資金費率，不足 3 家{missing_words(errors)}")
        v = float(np.mean(list(rates.values())))
        id = f"{a}_funding_cross_avg"
        deep = _stored_len(id) < DEEP_BELOW
        series = {}
        for name, fn in (("幣安", lambda: binance_funding(deep)), ("Bybit", lambda: bybit_funding(deep)),
                         ("OKX", okx_funding)):
            try:
                s = fn()
                if len(s):
                    series[name] = s
            except Exception:  # noqa: BLE001 - history from one exchange missing only shortens the average
                pass
        try:
            hl = hl_funding_history()
            if hl:
                series["Hyperliquid"] = pd.Series({d: x for d, x in hl})
        except Exception:  # noqa: BLE001
            pass
        points = []
        if series:
            frame = pd.concat(series, axis=1).sort_index()
            frame = frame[frame.index < today]
            mean = frame.mean(axis=1)[frame.count(axis=1) >= min(2, len(series))]
            points = [[d, round(float(x), 3)] for d, x in mean.items()]
        history = common.accumulate(id, points + [[today, round(v, 3)]])
        parts = "、".join(f"{name} {rate:+.1f}%" for name, rate in rates.items())
        return dict(value=round(v, 3), unit="%（年化）", asof=common.now_utc(),
                    display=f"{v:+.1f}%（{parts}{missing_words(errors)}）",
                    note=f"{cn}永續合約在幣安、Bybit、OKX、Bitget、Hyperliquid 目前資金費率的簡單平均，全部換算成年化。"
                         "正值代表做多的人付錢給做空的人；多家交易所同時偏高，比只有一家偏高更能代表「做多很擁擠」。"
                         "限制：沒有依各家規模加權；歷史走勢是各家每天實際結算費率的平均，早期只有幣安與 Bybit 兩家"
                         "（OKX 只提供約 3 個月、Hyperliquid 從 2023 年起逐步補齊），今天這一筆是還沒結算的目前費率。",
                    history=history)

    def funding_spread() -> dict:
        got, errors = snaps()
        rates = {name: s["funding"] for name, s in got.items()}
        if len(rates) < 3:
            raise RuntimeError(f"只抓到 {len(rates)} 家交易所的資金費率，不足 3 家{missing_words(errors)}")
        high, low = max(rates, key=rates.get), min(rates, key=rates.get)
        v = rates[high] - rates[low]
        return dict(value=round(v, 3), unit="個百分點（年化）", asof=common.now_utc(),
                    display=f"{v:.1f} 個百分點（最高 {high} {rates[high]:+.1f}%、最低 {low} {rates[low]:+.1f}%"
                            f"{missing_words(errors)}）",
                    note="五家交易所裡資金費率最高與最低的差距（年化）。差距大代表各家的多空擁擠程度不一致，"
                         "或某一家出現單邊擠壓；差距小代表全市場看法一致。只是分歧程度，看不出方向。" + SNAPSHOT,
                    history=common.accumulate(f"{a}_funding_cross_spread", [[today, round(v, 3)]]))

    def oi_total() -> dict:
        got, errors = snaps()
        if len(got) < 3:
            raise RuntimeError(f"只抓到 {len(got)} 家交易所的未平倉量，不足 3 家{missing_words(errors)}")
        v = sum(s["oi"] for s in got.values())
        id = f"{a}_oi_total"
        # only a complete five-exchange total goes into the stored history, so the series stays comparable
        history = common.accumulate(id, [[today, round(v)]] if not errors else [])
        if errors:
            history = history + [[today, round(v)]]
        parts = "、".join(f"{name} {s['oi'] / 1e8:,.1f}" for name, s in got.items())
        return dict(value=round(v), unit="美元", asof=common.now_utc(),
                    display=f"{_yi(v)}（{parts}，單位億美元{missing_words(errors)}）",
                    note=f"{cn}以泰達幣計價的永續合約在五家交易所還沒平倉的部位總額。數字大代表市場上的槓桿多，"
                         "行情一旦反向，被迫平倉會放大波動；看不出多空方向（每一口合約都有一多一空）。"
                         "限制：不含幣本位合約、交割合約與 CME，也不含其他交易所，所以比全市場少；"
                         "有交易所抓不到的那天不存進歷史。" + SNAPSHOT,
                    history=history)

    def hl_funding() -> dict:
        history = hl_funding_history()
        v = memo.get("snap_Hyperliquid", snap_hl)["funding"]
        return dict(value=round(v, 3), unit="%（年化）", asof=common.now_utc(), display=f"{v:+.1f}%（每小時結算，換算年化）",
                    note=f"Hyperliquid 是目前最大的鏈上永續合約交易所，用戶以加密原生的大戶與高槓桿交易者為主。這是{cn}合約的資金費率，"
                         "正值代表做多的人付錢給做空的人。它每小時結算一次、上限較寬，反應比中心化交易所快也更極端。"
                         "歷史是每天 24 筆的平均；每次更新往回多補約 100 天，直到 2023 年 5 月上線為止。",
                    history=history + [[today, round(v, 3)]])

    def hl_oi() -> dict:
        v = memo.get("snap_Hyperliquid", snap_hl)["oi"]
        return dict(value=round(v), unit="美元", asof=common.now_utc(), display=_yi(v),
                    note=f"Hyperliquid 上{cn}永續合約還沒平倉的部位總額。代表鏈上高槓桿交易者的參與程度；"
                         "所有部位都公開在鏈上，常被拿來觀察大戶。看不出多空方向。" + SNAPSHOT,
                    history=common.accumulate(f"{a}_hl_open_interest", [[today, round(v)]]))

    def bybit_oi() -> dict:
        id = f"{a}_bybit_open_interest"
        deep = _stored_len(id) < DEEP_BELOW
        coins, cursor = {}, None
        for _ in range(8 if deep else 1):
            params = {"category": "linear", "symbol": symbol, "intervalTime": "1d", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            result = _bybit("/v5/market/open-interest", params)
            for r in result["list"]:
                coins[int(r["timestamp"])] = float(r["openInterest"])
            cursor = result.get("nextPageCursor")
            if not cursor or len(result["list"]) < 200:
                break
        opens, end = {}, None
        for _ in range(2 if deep else 1):
            params = {"category": "linear", "symbol": symbol, "interval": "D", "limit": 1000}
            if end:
                params["end"] = end
            page = _bybit("/v5/market/kline", params)["list"]
            for r in page:
                opens[int(r[0])] = float(r[1])
            if len(page) < 1000:
                break
            end = int(page[-1][0]) - 1
        points = [[_day(ms), round(coins[ms] * opens[ms])] for ms in sorted(coins) if ms in opens]
        if not points:
            raise RuntimeError("Bybit 沒有回傳未平倉量歷史")
        history = common.accumulate(id, points)
        v = history[-1][1]
        return dict(value=v, unit="美元", asof=f"{history[-1][0]}T00:00:00Z",
                    display=f"{_yi(v)}（{coins[max(coins)]:,.0f} 顆{cn}）",
                    note=f"Bybit 上{cn}泰達幣永續合約還沒平倉的部位總額（顆數乘以當天開盤價）。"
                         "幣安只提供 30 天，Bybit 有好幾年的每日紀錄，所以用它來看槓桿水位在長期裡算高還是低。"
                         "限制：只有 Bybit 一家；美元金額會跟著幣價漲跌，不全是部位增減。",
                    history=history)

    # ---- OKX statistics -----------------------------------------------------------------------------------
    def okx_day(ms: str) -> str:
        return _day(float(ms) + 8 * 3_600_000)  # OKX daily bars start at 00:00 UTC+8

    def okx_long_short() -> dict:
        data = _okx("/rubik/stat/contracts/long-short-account-ratio", {"ccy": asset, "period": "1D"})
        points = [[okx_day(r[0]), round(float(r[1]), 4)] for r in data]
        history = common.accumulate(f"{a}_okx_long_short_ratio", points)
        v = float(sorted(data)[-1][1])
        return dict(value=round(v, 4), unit="倍", asof=common.now_utc(),
                    display=f"{v:.2f}（做多的帳戶約占 {v / (1 + v) * 100:.0f}%）",
                    rule="大於 1 是做多的帳戶比較多。" + NO_RULE,
                    note=f"OKX 上持有{cn}合約淨多單的帳戶數除以淨空單的帳戶數。算的是人數不是金額，所以代表的是散戶的方向；"
                         "散戶一面倒做多時常被當成反向訊號。限制：只有 OKX 一家；OKX 只提供約 180 天，更早的歷史由本頁每天累積。",
                    history=history)

    def okx_oi_volume() -> dict:
        data = sorted(_okx("/rubik/stat/contracts/open-interest-volume", {"ccy": asset, "period": "1D"}))
        done = [r for r in data[:-1] if float(r[2]) > 0]  # the newest bar is still filling up
        if not done:
            raise RuntimeError("OKX 沒有回傳完整的日資料")
        points = [[okx_day(r[0]), round(float(r[1]) / float(r[2]), 4)] for r in done]
        history = common.accumulate(f"{a}_okx_oi_volume_ratio", points)
        last = done[-1]
        v = float(last[1]) / float(last[2])
        return dict(value=round(v, 4), unit="倍", asof=f"{okx_day(last[0])}T00:00:00Z",
                    display=f"{v:.2f}（未平倉 {_yi(float(last[1]))}、當日成交 {_yi(float(last[2]))}）",
                    note=f"OKX 上{cn}合約的未平倉金額除以當天成交金額。比值高代表部位堆著不動、成交清淡（槓桿累積、容易被一次清掉）；"
                         "比值低代表成交熱絡、換手快。限制：只有 OKX 一家；週末成交少，比值會自然偏高；"
                         "OKX 只提供約 180 天，更早的歷史由本頁每天累積。",
                    history=history)

    def okx_loan_ratio() -> dict:
        data = _okx("/rubik/stat/margin/loan-ratio", {"ccy": asset, "period": "1D"})
        points = [[okx_day(r[0]), round(float(r[1]), 4)] for r in data]
        history = common.accumulate(f"{a}_okx_margin_loan_ratio", points)
        v = float(sorted(data)[-1][1])
        return dict(value=round(v, 4), unit="倍", asof=common.now_utc(), display=f"{v:.1f} 倍",
                    rule="數字越大代表借錢買進的人相對越多。" + NO_RULE,
                    note=f"OKX 現貨槓桿交易裡，借泰達幣買{cn}（做多）的金額除以借{cn}來賣（做空）的金額。"
                         "代表用借貸加槓桿的現貨交易者偏向哪一邊；比值快速升高是槓桿做多變擁擠的跡象。"
                         "限制：只有 OKX 一家，平常就遠大於 1，要和自己的歷史比；OKX 只提供約 180 天，更早的歷史由本頁每天累積。",
                    history=history)

    # ---- OKX liquidations ---------------------------------------------------------------------------------
    def liquidations() -> dict:
        def fetch():
            inst = _okx("/public/instruments", {"instType": "SWAP", "instId": okx_id})[0]
            size = float(inst["ctVal"]) * float(inst.get("ctMult") or 1)
            now = _now_ms()
            cutoff, after, oldest = now - DAY_MS, None, now
            longs = shorts = 0.0
            count = 0
            for _ in range(LIQ_PAGES):
                params = {"instType": "SWAP", "uly": f"{asset}-USDT", "state": "filled", "limit": 100}
                if after:
                    params["after"] = after
                data = _okx("/public/liquidation-orders", params)
                details = [d for block in data for d in block.get("details", [])]
                if not details:
                    oldest = cutoff
                    break
                for d in details:
                    ts = float(d["ts"])
                    oldest = min(oldest, ts)
                    if ts < cutoff:
                        continue
                    usd = float(d["sz"]) * size * float(d["bkPx"])
                    count += 1
                    if d["posSide"] == "long":
                        longs += usd
                    else:
                        shorts += usd
                if oldest <= cutoff:
                    break
                after = str(int(min(float(d["ts"]) for d in details)))
            hours = min(24.0, (now - oldest) / 3_600_000)
            return {"long": longs, "short": shorts, "count": count, "hours": hours, "full": oldest <= cutoff}
        return memo.get("liquidations", fetch)

    def liq(side: str):
        def build() -> dict:
            data = liquidations()
            v = data[side]
            other = data["short" if side == "long" else "long"]
            id = f"{a}_liq_{side}_24h"
            history = common.accumulate(id, [[today, round(v)]] if data["full"] else [])
            window = "近 24 小時" if data["full"] else f"只涵蓋近 {data['hours']:.0f} 小時（爆倉太多，超過本頁每次抓取的上限）"
            label, opposite = ("多單", "空單") if side == "long" else ("空單", "多單")
            return dict(value=round(v), unit="美元", asof=common.now_utc(),
                        display=f"{_wan(v)}（{window}；同期間{opposite} {_wan(other)}，共 {data['count']} 筆）",
                        note=f"OKX 上{cn}泰達幣永續合約被強制平倉的{label}金額（保證金不夠、被交易所強制結束的部位）。"
                             f"{label}大量爆倉通常發生在急{'跌' if side == 'long' else '漲'}，槓桿被清掉之後，同方向的賣壓或買壓常會減輕。"
                             "限制：只有 OKX 一家、只算泰達幣永續合約，幣安與 Bybit 只在即時串流提供、沒有可查詢的公開資料，"
                             "所以全市場的實際金額是這個數字的好幾倍，只能看相對大小與多空哪邊比較慘；"
                             "沒抓滿 24 小時的那天不存進歷史。" + SNAPSHOT,
                        history=history if data["full"] else history + [[today, round(v)]])
        return build

    specs = (
        ("funding_cross_avg", f"{cn}資金費率：五家交易所平均（年化）", CROSS_NAME, CROSS_URL, funding_avg),
        ("funding_cross_spread", f"{cn}資金費率：交易所之間的差距", CROSS_NAME, CROSS_URL, funding_spread),
        ("oi_total", f"{cn}永續合約未平倉量：五家交易所合計", CROSS_NAME,
         "https://www.coinglass.com/zh-TW/BitcoinOpenInterest", oi_total),
        ("hl_funding", f"{cn}資金費率（Hyperliquid，年化）", HL_NAME, HL_URL, hl_funding),
        ("hl_open_interest", f"{cn}未平倉量（Hyperliquid）", HL_NAME, HL_URL, hl_oi),
        ("bybit_open_interest", f"{cn}未平倉量（Bybit，長歷史）", BYBIT_NAME,
         f"https://www.bybit.com/trade/usdt/{symbol}", bybit_oi),
        ("okx_long_short_ratio", f"{cn}合約多空人數比（OKX）", OKX_NAME, OKX_URL, okx_long_short),
        ("okx_oi_volume_ratio", f"{cn}合約持倉與成交量比（OKX）", OKX_NAME, OKX_URL, okx_oi_volume),
        ("okx_margin_loan_ratio", f"{cn}現貨槓桿多空借幣比（OKX）", OKX_NAME, OKX_URL, okx_loan_ratio),
        ("liq_long_24h", f"{cn}近 24 小時多單爆倉金額（OKX）", "OKX 公開強制平倉資料",
         "https://www.okx.com/zh-hant/trade-market/liquidation/swap", liq("long")),
        ("liq_short_24h", f"{cn}近 24 小時空單爆倉金額（OKX）", "OKX 公開強制平倉資料",
         "https://www.okx.com/zh-hant/trade-market/liquidation/swap", liq("short")),
    )
    for suffix, name, source, url, build in specs:
        _add(rows, id=f"{a}_{suffix}", asset=asset, name=name, source=source, url=url, build=build)
    return rows


def collect() -> list[dict]:
    """One worker per asset for the exchanges and one for Deribit, so the one-off deep backfill still fits."""
    def deribit() -> list:
        rows: list = []
        for asset in ASSETS:
            rows += _option_rows(asset)
        return rows

    with ThreadPoolExecutor(max_workers=len(ASSETS) + 1) as pool:
        options = pool.submit(deribit)
        exchanges = [pool.submit(_exchange_rows, asset) for asset in ASSETS]
        option_rows = options.result()
        exchange_rows = [f.result() for f in exchanges]
    rows: list = []
    for asset, per_asset in zip(ASSETS, exchange_rows):
        rows += [r for r in option_rows if r["asset"] == asset] + per_asset
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    started = time.time()
    rows = collect()
    common.save("options", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"], f"[歷史 {len(r['history'])} 筆]")
    print(f"共 {len(rows)} 個指標，失敗 {sum(1 for r in rows if r['error'])} 個，耗時 {time.time() - started:.0f} 秒")
