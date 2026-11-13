"""Whale watch (group "whales"): bitcoin held by address size, and the largest Hyperliquid accounts' positions.

Usage: python -m dashboard.collectors.whales
Sources (public, no key):
- BGeometrics (bitcoin-data.com) free API: coins held by addresses of 10,000+, 1,000-10,000 and 100-1,000 BTC.
  Three calls a day at most (the free quota is 15 a day), each stored in cache/_store and fetched once a day.
- blockchain.info: total bitcoin supply, to get "everything below 100 BTC" by subtraction.
- Hyperliquid public leaderboard and account state: positions of the largest accounts.
Besides the indicators, a detail block for the page's whale section is written to cache/_whales.json.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from dashboard import common
from dashboard.collectors import onchain2 as oc
from dashboard.collectors.options import HYPERLIQUID, _post_json

GROUP = "whales"
HL_BOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
HL_NAME = "Hyperliquid 公開排行榜與帳戶資料（自行彙整）"
HL_PAGE = "https://app.hyperliquid.xyz/leaderboard"
HL_ACCOUNTS = 60            # the largest accounts by account value; fixed, not tuned
HL_COINS = ("BTC", "ETH", "SOL")
COHORT_SOURCE = f"{oc.BG_NAME}＋{oc.BCI_NAME}"
COHORT_LIMIT = ("限制：這是「地址」不是「人」。交易所、ETF 託管商的錢包也是大地址，一家機構會把幣分散在很多地址；"
                "交易所整理錢包、機構換託管商，都會讓數字跳動，但沒有人真的買賣。")
DETAIL = common.CACHE / "_whales.json"


def _wan(coins: float) -> str:
    return f"{coins / 1e4:,.1f} 萬顆"


def _signed(coins: float) -> str:
    return f"{coins:+,.0f} 顆"


def _cohorts() -> tuple[list[dict], list[dict]]:
    """Returns (indicators, table rows for the page)."""
    specs = (("whales_1k_plus", "1,000 顆以上的地址", "1,000 顆以上"),
             ("holders_100_1k", "100 到 1,000 顆的地址", "100–1,000 顆"),
             ("holders_under_100", "100 顆以下的地址", "100 顆以下"))
    rows, table = [], []
    try:
        parts, notes = {}, []
        for endpoint in ("coins-addr-10K-BTC", "coins-addr-10K-1K-BTC", "coins-addr-1K-100-BTC"):
            series, extra = oc._bg(endpoint)
            parts[endpoint] = series
            notes.append(extra)
        frame = pd.DataFrame(parts).dropna()
        supply = oc._bci("total-bitcoins").reindex(frame.index).ffill().bfill()
        held = {"whales_1k_plus": frame["coins-addr-10K-BTC"] + frame["coins-addr-10K-1K-BTC"],
                "holders_100_1k": frame["coins-addr-1K-100-BTC"]}
        held["holders_under_100"] = supply - held["whales_1k_plus"] - held["holders_100_1k"]
        extra = max(notes, key=len)
    except Exception as error:  # noqa: BLE001
        for key, name, _ in specs:
            rows.append(common.failed(id=f"btc_{key}_coins", group=GROUP, asset="BTC", name=f"比特幣：{name}持有量",
                                      source=COHORT_SOURCE, source_url=oc.BG_PAGE,
                                      error=f"{type(error).__name__}: {error}"))
        return rows, table
    for key, name, short in specs:
        series = held[key].dropna()
        last, day = float(series.iloc[-1]), series.index[-1]
        share = last / float(supply.loc[day]) * 100

        def delta(days: int) -> float | None:
            return float(series.iloc[-1] - series.iloc[-1 - days]) if len(series) > days else None

        d7, d30, d90 = delta(7), delta(30), delta(90)
        verb = "增加" if (d30 or 0) > 0 else "減少"
        under = ("這一級是用總發行量減掉上面兩級算出來的，所以也包含早就遺失、永遠不會動的幣。" if key == "holders_under_100" else "")
        rows.append(common.indicator(
            id=f"btc_{key}_coins", group=GROUP, asset="BTC", name=f"比特幣：{name}持有量", value=round(last, 0),
            unit="顆", display=f"{_wan(last)}（占發行量 {share:.1f}%；近 30 天{verb} {abs(d30 or 0):,.0f} 顆）",
            asof=oc._asof(day), source=COHORT_SOURCE, source_url=oc.BG_PAGE, evidence="C", reading="none",
            rule=oc.NO_RULE, note=f"餘額落在這一級的所有地址，合計持有多少顆比特幣。{under}{COHORT_LIMIT}{extra}",
            history=oc._history(series.tail(common.HISTORY_POINTS), 0)))
        change = (series - series.shift(30)).dropna()
        rows.append(common.indicator(
            id=f"btc_{key}_change_30d", group=GROUP, asset="BTC", name=f"比特幣：{name} 30 天持有量增減",
            value=round(float(change.iloc[-1]), 0), unit="顆",
            display=f"{_signed(float(change.iloc[-1]))}（7 天 {_signed(d7 or 0)}、90 天 {_signed(d90 or 0)}）",
            asof=oc._asof(day), source=COHORT_SOURCE, source_url=oc.BG_PAGE, evidence="C", reading="none",
            rule=oc.NO_RULE,
            note=f"這一級地址的合計持有量，比 30 天前多或少幾顆。正值常被說成「這一級在吸籌」，負值「在出貨」；"
                 f"但地址可能因為餘額變動而換級（例如 1,200 顆的地址轉出 300 顆，就從上一級掉到這一級）。{COHORT_LIMIT}",
            history=oc._history(change.tail(common.HISTORY_POINTS), 0)))
        table.append({"key": key, "label": short, "coins": round(last, 0), "share": round(share, 2), "asof": day,
                      "d7": d7, "d30": d30, "d90": d90,
                      "spark": oc._history(series.tail(365), 0)})
    return rows, table


def _hyperliquid() -> tuple[list[dict], dict]:
    """Positions of the largest Hyperliquid accounts. Returns (indicators, detail for the page)."""
    names = {"BTC": "比特幣", "ETH": "以太幣", "SOL": "Solana"}
    try:
        board = common.get_json(HL_BOARD, timeout=120)["leaderboardRows"]
        board.sort(key=lambda row: -float(row["accountValue"]))
        top = board[:HL_ACCOUNTS]
        totals = {coin: {"long": 0.0, "short": 0.0, "n_long": 0, "n_short": 0} for coin in HL_COINS}
        positions, read = [], 0
        for row in top:
            try:
                state = _post_json(HYPERLIQUID, {"type": "clearinghouseState", "user": row["ethAddress"]})
            except Exception:  # noqa: BLE001 - one unreadable account is skipped
                continue
            read += 1
            for item in state.get("assetPositions", []):
                p = item.get("position", {})
                coin = p.get("coin")
                if coin not in HL_COINS:
                    continue
                size, value = float(p["szi"]), float(p["positionValue"])
                if value <= 0:
                    continue
                side = "long" if size > 0 else "short"
                totals[coin][side] += value
                totals[coin]["n_" + side] += 1
                liq = p.get("liquidationPx")
                address = row["ethAddress"]
                positions.append({"address": address[:6] + "…" + address[-4:], "coin": coin, "side": side,
                                  "value": round(value, 0), "leverage": (p.get("leverage") or {}).get("value"),
                                  "entry": float(p["entryPx"]) if p.get("entryPx") else None,
                                  "liq": float(liq) if liq else None,
                                  "pnl": round(float(p.get("unrealizedPnl") or 0), 0),
                                  "account": round(float(row["accountValue"]), 0)})
        if read < HL_ACCOUNTS // 2:
            raise RuntimeError(f"{HL_ACCOUNTS} 個帳戶只讀到 {read} 個")
    except Exception as error:  # noqa: BLE001
        return [common.failed(id="btc_hl_whales_net", group=GROUP, asset="BTC", name="比特幣：Hyperliquid 大戶淨部位",
                              source=HL_NAME, source_url=HL_PAGE, error=f"{type(error).__name__}: {error}")], {}
    asof, today, rows = common.now_utc(), common.today_utc(), []
    limit = (f"統計的是 Hyperliquid 排行榜上帳戶淨值最大的 {HL_ACCOUNTS} 個帳戶（這次讀到 {read} 個）。"
             "限制：裡面有造市商和金庫帳戶，它們的部位是為了對沖，不代表看多或看空；大戶也可能在別的交易所有反向部位；"
             "這只是一家交易所。來源只給當下的值，走勢圖是本頁每天記下來的。")
    for coin in HL_COINS:
        t = totals[coin]
        net, gross = t["long"] - t["short"], t["long"] + t["short"]
        a = coin.lower()
        rows.append(common.indicator(
            id=f"{a}_hl_whales_net", group=GROUP, asset=coin, name=f"{names[coin]}：Hyperliquid 大戶淨部位",
            value=round(net, 0), unit="美元",
            display=f"淨{'做多' if net >= 0 else '做空'} {abs(net) / 1e8:,.2f} 億美元（做多 {t['n_long']} 個帳戶共 "
                    f"{t['long'] / 1e8:,.2f} 億、做空 {t['n_short']} 個帳戶共 {t['short'] / 1e8:,.2f} 億）",
            asof=asof, source=HL_NAME, source_url=HL_PAGE, evidence="C", reading="none", rule=oc.NO_RULE,
            note="這些帳戶做多的合約金額減掉做空的合約金額。" + limit,
            history=common.accumulate(f"{a}_hl_whales_net", [[today, round(net, 0)]])))
        if gross > 0:
            share = t["long"] / gross * 100
            rows.append(common.indicator(
                id=f"{a}_hl_whales_long_share", group=GROUP, asset=coin,
                name=f"{names[coin]}：Hyperliquid 大戶做多金額占比", value=round(share, 1), unit="%",
                display=f"{share:.0f}%（多空合計 {gross / 1e8:,.2f} 億美元）", asof=asof, source=HL_NAME,
                source_url=HL_PAGE, evidence="C", reading="none", rule=oc.NO_RULE,
                note="這些帳戶的做多金額，占他們多空合計金額的比例。50% 代表多空金額一樣大。" + limit,
                history=common.accumulate(f"{a}_hl_whales_long_share", [[today, round(share, 1)]])))
    positions.sort(key=lambda p: -p["value"])
    detail = {"asof": asof, "accounts": read, "target": HL_ACCOUNTS,
              "totals": [{"coin": coin, **{k: round(v, 0) for k, v in totals[coin].items()}} for coin in HL_COINS],
              "positions": positions[:15]}
    return rows, detail


def collect() -> list[dict]:
    cohort_rows, table = _cohorts()
    hl_rows, hl = _hyperliquid()
    common.CACHE.mkdir(parents=True, exist_ok=True)
    DETAIL.write_text(json.dumps({"collected_utc": common.now_utc(), "cohorts": table, "hyperliquid": hl},
                                 ensure_ascii=False), encoding="utf-8")
    return cohort_rows + hl_rows


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    out = collect()
    common.save("whales", out)
    for row in out:
        print(row["id"], row["asof"], row["display"] or row["error"], f"[歷史 {len(row['history'])} 筆]")
