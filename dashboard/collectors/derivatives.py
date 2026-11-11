"""Derivatives collector (group "derivatives"): funding, open interest, premium, positioning, options.

Sources (all public, no key): Binance USDT-margined futures (fapi.binance.com), Bybit v5 market endpoints as a
fallback for the core numbers when Binance is unreachable, Deribit public API for options. See dashboard/SPEC2.md.

Funding rates are paged back about four years. Binance only serves 30 days of open interest and positioning
ratios, so those (and the snapshot-only numbers) are accumulated day by day with common.accumulate.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from dashboard import common

GROUP = "derivatives"
ASSETS = ("BTC", "ETH", "SOL")
OPTION_ASSETS = ("BTC", "ETH")
CN = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "索拉納"}
BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
BINANCE_NAME = "幣安 U 本位永續合約公開資料"
BYBIT_NAME = "Bybit 永續合約公開資料（備援）"
DERIBIT_NAME = "Deribit 選擇權公開資料"
BINANCE_DATA_URL = "https://www.binance.com/zh-TC/futures/funding-history/perpetual/trading-data"
DAY_MS = 86_400_000
HOURS_PER_YEAR = 24 * 365
FUNDING_HIGH = 30.0  # annualised %, conventional "crowded longs" line
FUNDING_RULE = ("換算成年化後：高於 30% 視為多方過度擁擠＝偏空；低於 0% 視為空方擁擠＝偏多；其餘中性。"
                "這是本頁事前自訂的固定門檻，沒有文獻依據，也沒有驗證過。")
FUNDING_NOTE = ("資金費率是永續合約多空雙方定期互付的費用，正值代表做多的人付錢給做空的人。"
                "這是反向指標（太多人同一邊時反而容易反轉），單獨使用的證據很弱。")


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp() * 1000


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _pct(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}%"


def _usd(v: float) -> str:
    return f"{v / 1e8:,.1f} 億美元"


def _futures_url(asset: str) -> str:
    return f"https://www.binance.com/zh-TC/futures/{asset}USDT"


def _bybit(path: str, params: dict) -> list:
    reply = common.get_json(BYBIT + path, params, timeout=15)
    if reply.get("retCode") != 0:
        raise RuntimeError(f"Bybit 回報錯誤：{reply.get('retMsg')}")
    return reply["result"]["list"]


def _both(first: Exception, second: Exception) -> RuntimeError:
    return RuntimeError(f"幣安失敗（{first!r}）；Bybit 也失敗（{second!r}）")


class _Memo:
    """Fetch once, remember the result or the error, and replay it to every indicator that needs it."""

    def __init__(self):
        self._done: dict = {}

    def get(self, key: str, fetch):
        if key not in self._done:
            try:
                self._done[key] = (fetch(), None)
            except Exception as exc:
                self._done[key] = (None, exc)
        value, error = self._done[key]
        if error is not None:
            raise error
        return value


def _add(rows: list, *, id: str, asset: str, name: str, source: str, url: str, build) -> None:
    """Run one indicator builder inside its own try/except so one failure never drags others down."""
    try:
        fields = build()
        fields.setdefault("source", source)
        fields.setdefault("source_url", url)
        rows.append(common.indicator(id=id, group=GROUP, asset=asset, name=name, **fields))
    except Exception as exc:
        rows.append(common.failed(id=id, group=GROUP, asset=asset, name=name, source=source, source_url=url,
                                  error=f"{type(exc).__name__}: {exc}"))


def _stored(id: str, source: str, points: list) -> list:
    """Accumulate Binance points in the store; fallback-source points use another basis, so they are not mixed in."""
    return common.accumulate(id, points) if source == BINANCE_NAME else points


def _funding_reading(annual: float) -> str:
    return "bearish" if annual > FUNDING_HIGH else "bullish" if annual < 0 else "neutral"


def _asset_rows(rows: list, asset: str, memo: _Memo) -> None:
    a, cn, symbol = asset.lower(), CN[asset], f"{asset}USDT"
    url = _futures_url(asset)

    # ---- raw fetches (memoised) -------------------------------------------------------------------------------
    def marks() -> dict:
        """Every Binance futures symbol's mark and index price in one call (perpetuals and quarterlies)."""
        return memo.get("marks", lambda: {r["symbol"]: r for r in
                                          common.get_json(f"{BINANCE}/fapi/v1/premiumIndex", timeout=15)})

    def bybit_ticker() -> dict:
        return memo.get(f"bybit_ticker_{asset}",
                        lambda: _bybit("/v5/market/tickers", {"category": "linear", "symbol": symbol})[0])

    def snapshot() -> dict:
        """Current funding rate, mark and index price. Binance first, Bybit as fallback."""
        def fetch():
            try:
                r = marks()[symbol]
                return {"funding": float(r["lastFundingRate"]), "mark": float(r["markPrice"]),
                        "index": float(r["indexPrice"]), "ms": float(r["time"]), "source": BINANCE_NAME,
                        "url": url}
            except Exception as first:
                try:
                    t = bybit_ticker()
                    return {"funding": float(t["fundingRate"]), "mark": float(t["markPrice"]),
                            "index": float(t["indexPrice"]), "ms": _now_ms(), "source": BYBIT_NAME,
                            "url": f"https://www.bybit.com/trade/usdt/{symbol}",
                            "oi_value": float(t["openInterestValue"])}
                except Exception as second:
                    raise _both(first, second) from second
        return memo.get(f"snapshot_{asset}", fetch)

    def funding_history() -> dict:
        """Settled funding rates of about four years (Binance, paged forward from startTime; Bybit only gives the
        last 200 settlements), each annualised with its own settlement interval, plus the daily averages."""
        def fetch():
            start = int(_now_ms() - (common.HISTORY_POINTS + 35) * DAY_MS)
            try:
                raw, cursor = [], start
                for _ in range(12):
                    page = common.get_json(f"{BINANCE}/fapi/v1/fundingRate",
                                           {"symbol": symbol, "startTime": cursor, "limit": 1000}, timeout=15)
                    raw += page
                    if len(page) < 1000:
                        break
                    cursor = int(page[-1]["fundingTime"]) + 1
                frame = pd.DataFrame([[float(r["fundingTime"]), float(r["fundingRate"])] for r in raw],
                                     columns=["ms", "rate"])
                source = BINANCE_NAME
            except Exception as first:
                try:
                    raw = _bybit("/v5/market/funding/history",
                                 {"category": "linear", "symbol": symbol, "limit": 200})
                    frame = pd.DataFrame([[float(r["fundingRateTimestamp"]), float(r["fundingRate"])] for r in raw],
                                         columns=["ms", "rate"])
                    source = BYBIT_NAME
                except Exception as second:
                    raise _both(first, second) from second
            frame = frame.drop_duplicates("ms").sort_values("ms").reset_index(drop=True)
            frame = frame[frame["ms"] >= start].reset_index(drop=True)
            if len(frame) < 3:
                raise RuntimeError("資金費率歷史筆數不足")
            gap = (frame["ms"].diff() / 3_600_000).round().bfill().clip(1, 8)  # hours since the last settlement
            frame["annual"] = frame["rate"] * HOURS_PER_YEAR / gap * 100
            daily = frame.assign(day=frame["ms"].map(_day)).groupby("day")["annual"].mean()
            return {"frame": frame, "hours": int(gap.iloc[-1]), "source": source, "daily": daily}
        return memo.get(f"funding_{asset}", fetch)

    def ratio_series(path: str, field: str) -> pd.DataFrame:
        def fetch():
            raw = common.get_json(f"{BINANCE}{path}", {"symbol": symbol, "period": "1d", "limit": 30}, timeout=15)
            if not raw:
                raise RuntimeError("來源回傳空資料")
            return pd.DataFrame([[float(r["timestamp"]), float(r[field])] for r in raw],
                                columns=["ms", "v"]).sort_values("ms").reset_index(drop=True)
        return memo.get(f"{path}_{asset}", fetch)

    def oi_data() -> dict:
        """Open interest in US dollars now, plus a daily history (about 30 days)."""
        def fetch():
            try:
                now = common.get_json(f"{BINANCE}/fapi/v1/openInterest", {"symbol": symbol}, timeout=15)
                mark = float(marks()[symbol]["markPrice"])
                raw = common.get_json(f"{BINANCE}/futures/data/openInterestHist",
                                      {"symbol": symbol, "period": "1d", "limit": 30}, timeout=15)
                hist = pd.DataFrame([[float(r["timestamp"]), float(r["sumOpenInterestValue"])] for r in raw],
                                    columns=["ms", "v"])
                return {"value": float(now["openInterest"]) * mark, "ms": float(now["time"]), "hist": hist,
                        "source": BINANCE_NAME, "url": BINANCE_DATA_URL, "basis": "美元金額"}
            except Exception as first:
                try:
                    t = bybit_ticker()
                    raw = _bybit("/v5/market/open-interest",
                                 {"category": "linear", "symbol": symbol, "intervalTime": "1d", "limit": 31})
                    mark = float(t["markPrice"])
                    # Bybit history is in coins; value it at today's price, so changes reflect contracts only
                    hist = pd.DataFrame([[float(r["timestamp"]), float(r["openInterest"]) * mark] for r in raw],
                                        columns=["ms", "v"])
                    return {"value": float(t["openInterestValue"]), "ms": _now_ms(), "hist": hist,
                            "source": BYBIT_NAME, "url": f"https://www.bybit.com/trade/usdt/{symbol}",
                            "basis": "合約顆數（Bybit 歷史只有顆數）"}
                except Exception as second:
                    raise _both(first, second) from second
        d = memo.get(f"oi_{asset}", fetch)
        d["hist"] = d["hist"].sort_values("ms").reset_index(drop=True)
        return d

    def oi_change(days: int) -> tuple:
        d = oi_data()
        hist = d["hist"]
        if hist.empty:
            raise RuntimeError("未平倉量歷史是空的")
        target = d["ms"] - days * DAY_MS
        row = hist.iloc[(hist["ms"] - target).abs().idxmin()]
        actual = (d["ms"] - row["ms"]) / DAY_MS
        if abs(actual - days) > 2.5:
            raise RuntimeError(f"來源只保留約 {((d['ms'] - hist['ms'].iloc[0]) / DAY_MS):.0f} 天歷史，"
                               f"找不到 {days} 天前的數值")
        return (d["value"] / row["v"] - 1) * 100, actual

    # ---- indicator builders -----------------------------------------------------------------------------------
    def funding_now() -> dict:
        s = snapshot()
        try:
            hours = funding_history()["hours"]
        except Exception:
            hours = 8
        annual = s["funding"] * HOURS_PER_YEAR / hours * 100
        try:
            history = [[d, round(float(v), 3)] for d, v in funding_history()["daily"].items()]
        except Exception:
            history = []
        return dict(value=round(s["funding"] * 100, 5), unit=f"%／{hours} 小時", asof=_iso(s["ms"]),
                    display=f"{s['funding'] * 100:+.4f}%／{hours} 小時（年化約 {_pct(annual)}）",
                    evidence="C", reading=_funding_reading(annual), rule=FUNDING_RULE,
                    note=FUNDING_NOTE + "這裡是本期預估值，結算前還會變動；走勢圖是每天平均費率換算的年化百分比。",
                    history=history, source=s["source"], source_url=s["url"])

    def funding_avg() -> dict:
        h = funding_history()
        rolling = h["daily"].rolling(30, min_periods=20).mean().dropna()
        if rolling.empty:
            raise RuntimeError("資金費率歷史不足 20 天，算不出 30 天平均")
        annual = float(rolling.iloc[-1])
        per = annual / 100 / HOURS_PER_YEAR * h["hours"] * 100
        return dict(value=round(annual, 3), unit="%（年化）", asof=_iso(h["frame"]["ms"].iloc[-1]),
                    display=f"年化約 {_pct(annual)}（相當於平均每 {h['hours']} 小時 {per:+.4f}%）",
                    evidence="C", reading=_funding_reading(annual), rule=FUNDING_RULE,
                    note=FUNDING_NOTE + "這裡是最近 30 天實際結算費率的平均（先算每天的平均，再取 30 天平均），換算成年化。",
                    history=[[d, round(float(v), 3)] for d, v in rolling.items()], source=h["source"])

    def oi_now() -> dict:
        d = oi_data()
        parts = []
        for days in (7, 30):
            try:
                parts.append(f"{days} 天 {_pct(oi_change(days)[0])}")
            except Exception:
                parts.append(f"{days} 天 無資料")
        return dict(value=round(d["value"], 0), unit="美元", asof=_iso(d["ms"]),
                    display=f"{_usd(d['value'])}（{'／'.join(parts)}）", evidence="C", reading="none",
                    rule="沒有事前固定的門檻，不給方向。",
                    note="未平倉量是還沒結清的永續合約總金額，只算這一家交易所。金額變大代表槓桿部位變多、"
                         "行情容易放大，但看不出是多方還是空方加碼。來源只保留 30 天，更早的走勢是本頁每次更新時累積下來的。",
                    history=_stored(f"{a}_open_interest", d["source"],
                                    [[_day(m), round(v, 0)] for m, v in zip(d["hist"]["ms"], d["hist"]["v"])]
                                    + [[_day(d["ms"]), round(d["value"], 0)]]),
                    source=d["source"], source_url=d["url"])

    def oi_delta(days: int):
        def build() -> dict:
            d = oi_data()
            v, actual = oi_change(days)
            return dict(value=round(v, 3), unit="%", asof=_iso(d["ms"]),
                        display=f"{_pct(v)}（相對約 {actual:.0f} 天前）", evidence="C", reading="none",
                        rule="沒有事前固定的門檻，不給方向。",
                        note=f"未平倉量相對 {days} 天前的變化，以{d['basis']}計算。"
                             "以美元計時會同時受價格漲跌影響，不完全等於部位增減。",
                        source=d["source"], source_url=d["url"])
        return build

    def premium() -> dict:
        s = snapshot()
        v = (s["mark"] / s["index"] - 1) * 100
        return dict(value=round(v, 4), unit="%", asof=_iso(s["ms"]),
                    display=f"{v:+.3f}%（永續標記價 {s['mark']:,.2f}，現貨指數價 {s['index']:,.2f}）",
                    evidence="C", reading="none", rule="沒有事前固定的門檻，不給方向。",
                    note="永續合約的標記價比現貨指數價貴或便宜多少。正值代表合約買盤較積極；"
                         "數值通常很小而且跳動快，只是當下的快照（走勢圖是本頁每次更新時記下的那一刻）。",
                    history=_stored(f"{a}_perp_premium", s["source"], [[_day(s["ms"]), round(v, 4)]]),
                    source=s["source"], source_url=s["url"])

    def basis() -> dict:
        index = float(marks()[symbol]["indexPrice"])
        now = _now_ms()
        quarters = []
        for name, r in marks().items():
            if name.startswith(symbol + "_") and name[len(symbol) + 1:].isdigit():
                expiry = datetime.strptime(name[len(symbol) + 1:], "%y%m%d").replace(
                    hour=8, tzinfo=timezone.utc).timestamp() * 1000
                if expiry - now > DAY_MS:
                    quarters.append((expiry, name, float(r["markPrice"]), float(r["time"])))
        if not quarters:
            raise RuntimeError(f"幣安目前沒有 {symbol} 的季度交割合約，無法計算年化基差")
        expiry, name, mark, ms = min(quarters)
        days = (expiry - now) / DAY_MS
        raw = (mark / index - 1) * 100
        annual = raw * 365 / days
        return dict(value=round(annual, 3), unit="%（年化）", asof=_iso(ms),
                    display=f"年化 {_pct(annual)}（合約 {name} 比現貨指數貴 {raw:+.2f}%，還有 {days:.0f} 天到期）",
                    evidence="C", reading="none", rule="沒有事前固定的門檻，不給方向。",
                    note="季度合約是有到期日的期貨。它比現貨貴多少、換算成一年的比率，叫做年化基差；"
                         "越高代表市場越願意付溢價做多。快到期時這個數字會變得不穩定。"
                         "來源沒有歷史數列，走勢圖是本頁每次更新時自己記下來的。",
                    history=common.accumulate(f"{a}_quarterly_basis", [[_day(ms), round(annual, 3)]]),
                    source="幣安 U 本位季度合約公開資料",
                    source_url=f"https://www.binance.com/zh-TC/futures/{name}")

    def ratio(key: str, path: str, field: str, note: str, bybit_ok: bool = False, side: str = "多方"):
        def build() -> dict:
            source, link = BINANCE_NAME, BINANCE_DATA_URL
            try:
                frame = ratio_series(path, field)
            except Exception as first:
                if not bybit_ok:
                    raise
                try:
                    raw = _bybit("/v5/market/account-ratio",
                                 {"category": "linear", "symbol": symbol, "period": "1d", "limit": 30})
                    frame = pd.DataFrame([[float(r["timestamp"]), float(r["buyRatio"]) / float(r["sellRatio"])]
                                          for r in raw], columns=["ms", "v"]).sort_values("ms").reset_index(drop=True)
                    source, link = BYBIT_NAME, f"https://www.bybit.com/trade/usdt/{symbol}"
                except Exception as second:
                    raise _both(first, second) from second
            v = float(frame["v"].iloc[-1])
            share = v / (1 + v) * 100
            return dict(value=round(v, 4), unit="倍", asof=_iso(frame["ms"].iloc[-1]),
                        display=f"{v:.2f}（{side}約占 {share:.0f}%）", evidence="C", reading="none",
                        rule="沒有事前固定的門檻，不給方向。",
                        note=note + "來源只保留 30 天，更早的走勢是本頁每次更新時累積下來的。",
                        history=_stored(f"{a}_{key}", source,
                                        [[_day(m), round(x, 4)] for m, x in zip(frame["ms"], frame["v"])]),
                        source=source, source_url=link)
        return build

    for suffix, name, build in (
            ("funding_rate", f"{cn}資金費率（目前）", funding_now),
            ("funding_rate_30d_avg", f"{cn}資金費率（近 30 天平均）", funding_avg),
            ("open_interest", f"{cn}永續合約未平倉量", oi_now),
            ("open_interest_change_7d", f"{cn}未平倉量 7 天變化", oi_delta(7)),
            ("open_interest_change_30d", f"{cn}未平倉量 30 天變化", oi_delta(30)),
            ("perp_premium", f"{cn}永續合約溢價", premium),
            ("quarterly_basis", f"{cn}季度合約年化基差", basis),
            ("long_short_accounts", f"{cn}散戶多空人數比",
             ratio("long_short_accounts", "/futures/data/globalLongShortAccountRatio", "longShortRatio",
                   "所有帳戶中，淨做多的帳戶數除以淨做空的帳戶數（每日資料）。只算人數不算金額，"
                   "小帳戶占多數，所以常被當成散戶情緒；長期多半大於 1，不能直接當成買賣訊號。", bybit_ok=True)),
            ("top_trader_positions", f"{cn}大戶多空持倉比",
             ratio("top_trader_positions", "/futures/data/topLongShortPositionRatio", "longShortRatio",
                   "幣安保證金餘額前 20% 帳戶的多單持倉除以空單持倉（每日資料）。這是幣安帳戶統計，"
                   "不等於真正的主力動向：大戶可能在別的交易所或現貨對沖，只看這個數字會誤判。")),
            ("taker_buy_sell", f"{cn}主動買賣量比",
             ratio("taker_buy_sell", "/futures/data/takerlongshortRatio", "buySellRatio",
                   "主動用市價買進的成交量除以主動用市價賣出的成交量（每日資料）。大於 1 代表追價買的人較多；"
                   "每天變動很大，只描述當天的買賣力道。", side="主動買進"))):
        if suffix == "quarterly_basis" and asset == "SOL":
            continue  # Binance lists quarterly delivery contracts for BTC and ETH only
        _add(rows, id=f"{a}_{suffix}", asset=asset, name=name, source=BINANCE_NAME, url=url, build=build)


def _option_rows(rows: list, asset: str) -> None:
    a, cn = asset.lower(), CN[asset]

    def dvol() -> dict:
        now = int(_now_ms())
        start, end, data = now - (common.HISTORY_POINTS + 5) * DAY_MS, now, []
        for _ in range(4):  # Deribit returns the newest 1000 points and a continuation mark for the older ones
            result = common.get_json(f"{DERIBIT}/get_volatility_index_data",
                                     {"currency": asset, "resolution": "1D", "start_timestamp": start,
                                      "end_timestamp": end}, timeout=20)["result"]
            data += result["data"]
            end = result.get("continuation")
            if not end or not result["data"]:
                break
        data = sorted(data)
        if not data:
            raise RuntimeError("Deribit 回傳空資料")
        v = float(data[-1][4])
        last_ms = min(data[-1][0] + DAY_MS, now)
        return dict(value=round(v, 2), unit="%（年化）", asof=_iso(last_ms), display=f"{v:.1f}%",
                    evidence="C", reading="none", rule="沒有事前固定的門檻，不給方向。",
                    note=f"Deribit 波動率指數（英文簡稱 DVOL）：從{cn}選擇權價格反推，市場預期未來 30 天價格會晃動多大，"
                         "以年化百分比表示。數字高代表市場預期大波動，但看不出方向。",
                    history=[[_day(r[0]), round(float(r[4]), 2)] for r in data])

    def put_call() -> dict:
        reply = common.get_json(f"{DERIBIT}/get_book_summary_by_currency",
                                {"currency": asset, "kind": "option"}, timeout=20)
        puts = calls = 0.0
        for r in reply["result"]:
            oi = float(r.get("open_interest") or 0)
            if r["instrument_name"].endswith("-P"):
                puts += oi
            elif r["instrument_name"].endswith("-C"):
                calls += oi
        if calls <= 0:
            raise RuntimeError("買權未平倉量為 0，無法計算比值")
        v = puts / calls
        return dict(value=round(v, 4), unit="倍", asof=common.now_utc(),
                    display=f"{v:.2f}（賣權 {puts:,.0f} 顆、買權 {calls:,.0f} 顆{cn}）", evidence="C",
                    reading="none", rule="沒有事前固定的門檻，不給方向。",
                    note="賣權（看跌或避險用）未平倉量除以買權（看漲用）未平倉量，把 Deribit 所有到期日與履約價加總。"
                         "比值高代表避險需求較多；但賣權也可能是被賣出收權利金，不能直接當成看空。"
                         "來源只有目前數值，走勢圖是本頁每次更新時自己記下來的。",
                    history=common.accumulate(f"{a}_put_call_oi_ratio", [[common.today_utc(), round(v, 4)]]))

    _add(rows, id=f"{a}_dvol", asset=asset, name=f"{cn}選擇權隱含波動率指數（Deribit）", source=DERIBIT_NAME,
         url=f"https://www.deribit.com/statistics/{asset}/volatility-index", build=dvol)
    _add(rows, id=f"{a}_put_call_oi_ratio", asset=asset, name=f"{cn}選擇權賣權／買權未平倉比", source=DERIBIT_NAME,
         url=f"https://www.deribit.com/statistics/{asset}/options-open-interest", build=put_call)


def collect() -> list[dict]:
    rows: list[dict] = []
    memo = _Memo()
    for asset in ASSETS:
        _asset_rows(rows, asset, memo)
        if asset in OPTION_ASSETS:
            _option_rows(rows, asset)
    return rows


if __name__ == "__main__":
    rows = collect()
    common.save("derivatives", rows)
    for r in rows:
        print(r["id"], r["asof"], r["display"] or r["error"])
