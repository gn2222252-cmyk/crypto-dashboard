"""假資料與測試頁產生器（助理乙的開發工具，不是正式建置）。

    python dashboard/mock/gen_mock.py            # 用假資料產生 mock/index.html、index_nulls.html、index_empty.html
    python dashboard/mock/gen_mock.py --real     # 用 dashboard/out/data.json 產生 mock/index_real.html

內嵌方式和 build.py 相同：把 JSON 取代 template.html 裡唯一的 placeholder 字串。
"""
import json
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "template.html"
PLACEHOLDER = "__DASHBOARD_" + "DATA__"
N = 1500
TODAY = date(2026, 11, 11)
DAYS = [(TODAY - timedelta(days=N - 1 - i)).isoformat() for i in range(N)]
ZONES = [("freezing", "過冷"), ("cold", "偏冷"), ("neutral", "中性"), ("warm", "偏熱"), ("hot", "過熱")]
rnd = random.Random(7)


def embed(payload):
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    for code in (0x3C, 0x3E, 0x26, 0x2028, 0x2029):
        text = text.replace(chr(code), "%su%04x" % (chr(92), code))
    return text


def zone_of(t):
    return ZONES[min(4, int(t // 20))]


def walk(start, drift, vol, phase):
    """幾何隨機漫步加一個約 4 年的週期，讓圖看起來像加密貨幣。"""
    out, x = [], math.log(start)
    for i in range(N):
        cyc = math.sin(2 * math.pi * (i / 1400.0) + phase)
        x += drift + 0.0016 * cyc + rnd.gauss(0, vol)
        out.append(math.exp(x))
    return out


def temperature(prices):
    """用價格相對 200 日均線的百分位加雜訊，做出像樣的 0-100 溫度。"""
    ratio = []
    for i in range(N):
        w = prices[max(0, i - 199): i + 1]
        ratio.append(prices[i] / (sum(w) / len(w)))
    order = sorted(ratio)
    temp, smooth = [], None
    for r in ratio:
        pct = 100.0 * sum(1 for o in order if o <= r) / N
        v = 0.75 * pct + 12 + rnd.gauss(0, 2.5)
        smooth = v if smooth is None else 0.85 * smooth + 0.15 * v
        temp.append(round(max(1, min(99, smooth)), 1))
    return temp


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def zone_stats(prices, temp, current_zone):
    rows = []
    for key, label in ZONES:
        idx = [i for i in range(N) if zone_of(temp[i])[0] == key]
        f30 = [100 * (prices[i + 30] / prices[i] - 1) for i in idx if i + 30 < N]
        f90 = [100 * (prices[i + 90] / prices[i] - 1) for i in idx if i + 90 < N]
        rows.append({
            "zone": key, "label": label, "days": len(idx),
            "fwd30_median": round(median(f30), 1) if f30 else None,
            "fwd30_up": round(100 * sum(1 for v in f30 if v > 0) / len(f30), 1) if f30 else None,
            "fwd90_median": round(median(f90), 1) if f90 else None,
            "fwd90_up": round(100 * sum(1 for v in f90 if v > 0) / len(f90), 1) if f90 else None,
            "current": key == current_zone})
    return rows


def pct_chg(p, d):
    return round(100 * (p[-1] / p[-1 - d] - 1), 1)


COMPONENTS = {
    "BTC": [("btc_ma200", "相對 200 日均線", "0.94 倍"), ("btc_wma200", "200 週均線倍數", "1.52 倍"),
            ("btc_pi", "Pi 週期比值", "0.504"), ("btc_mvrv", "市值對已實現價值比（MVRV）", "1.37 倍"),
            ("btc_drawdown", "距最高價的跌幅", "-41.2%"), ("btc_funding30", "資金費率 30 天平均", "年化 3.1%"),
            ("fear_greed", "恐懼貪婪指數", "24（極度恐懼）"), ("btc_etf30", "現貨 ETF 30 日淨流入", "-12.4 億美元"),
            ("stable30", "穩定幣供給 30 天變化", "+1.8%"), ("btc_wiki", "維基百科關注度", "每日 8,120 次")],
    "ETH": [("eth_ma200", "相對 200 日均線", "0.81 倍"), ("eth_drawdown", "距最高價的跌幅", "-55.0%"),
            ("eth_mvrv", "市值對已實現價值比（MVRV）", "1.02 倍"), ("eth_funding30", "資金費率 30 天平均", "年化 1.2%"),
            ("fear_greed", "恐懼貪婪指數", "24（極度恐懼）"), ("eth_etf30", "現貨 ETF 30 日淨流入", "-3.1 億美元"),
            ("ethbtc_ma200", "ETH/BTC 比值相對其 200 日均線", "0.88 倍")],
    "SOL": [("sol_ma200", "相對 200 日均線", "1.21 倍"), ("sol_drawdown", "距最高價的跌幅", "-18.4%"),
            ("sol_funding30", "資金費率 30 天平均", "年化 14.9%"), ("fear_greed", "恐懼貪婪指數", "24（極度恐懼）")],
}


def make_asset(sym, name, start, drift, vol, phase, target):
    prices = walk(start, drift, vol, phase)
    temp = temperature(prices)
    last = temp[-1]
    for k in range(120):  # 把最後一段拉到指定溫度，方便測各分區的樣子
        i = N - 120 + k
        temp[i] = round(max(1, min(99, temp[i] + (target - last) * (k / 119.0))), 1)
    t = temp[-1]
    zkey, zlabel = zone_of(t)
    comps = []
    for cid, cname, disp in COMPONENTS[sym]:
        heat = max(1, min(99, t + rnd.uniform(-32, 32)))
        plain = "%s在過去 4 年裡%s（第 %d 百分位）。" % (
            cname, "偏高" if heat > 60 else "偏低" if heat < 40 else "在中間", round(heat))
        comps.append({"id": cid, "name": cname, "heat": round(heat, 1), "display": disp, "plain": plain,
                      "counted": True})
    if sym == "BTC":
        comps[-1].update(heat=None, counted=False, display=None, plain="歷史只有 212 天，不到 365 天，這次不計入溫度。")
        comps[7].update(heat=93.0, plain="現貨 ETF 過去 30 天淨流出 12.4 億美元，不過在自己的歷史裡排名仍然偏高（示範資料）。")
    if sym == "ETH":
        comps[2].update(heat=None, counted=False, display=None, plain="這次取得失敗，沒有數值，不計入溫度。")
    counted = [c for c in comps if c["counted"]]
    hottest = sorted(counted, key=lambda c: -c["heat"])[:2]
    coldest = sorted(counted, key=lambda c: c["heat"])[:2]
    d7, d30 = round(t - temp[-8], 1), round(t - temp[-31], 1)
    summary = [
        "%s的市場溫度是 %d 分，落在「%s」區（0 是最冷、100 是最熱）。" % (name, round(t), zlabel),
        "最熱的成分是%s和%s；最冷的是%s和%s。" % (hottest[0]["name"], hottest[1]["name"], coldest[0]["name"],
                                       coldest[1]["name"]),
        "溫度比 7 天前%s %.1f 分，比 30 天前%s %.1f 分。" % ("升" if d7 >= 0 else "降", abs(d7),
                                               "升" if d30 >= 0 else "降", abs(d30)),
        "現貨 ETF 過去 30 天是淨流出，資金費率接近零，做多的人不擁擠。",
        "週期保險（Pi 週期比值 ≥ 1、MVRV ≥ 3.7）都沒有達到門檻。",
    ]
    rules = [{"name": "Pi 週期頂部指標", "state": "未達門檻", "detail": "0.504，門檻 1"},
             {"name": "MVRV 過熱線", "state": "未達門檻", "detail": "1.37 倍，門檻 3.7 倍"}] if sym != "SOL" else []
    p = prices[-1]
    return {
        "asset": sym, "name": name, "price": round(p, 2),
        "price_display": "{:,.0f} 美元".format(p) if p > 1000 else "{:,.2f} 美元".format(p),
        "change_24h": pct_chg(prices, 1), "change_7d": pct_chg(prices, 7), "change_30d": pct_chg(prices, 30),
        "temperature": t, "zone": zkey, "zone_label": zlabel, "temp_change_7d": d7, "temp_change_30d": d30,
        "summary": summary, "components": comps,
        "temp_history": [[DAYS[i], temp[i]] for i in range(N)],
        "price_history": [[DAYS[i], round(prices[i], 2)] for i in range(N)],
        "zone_stats": zone_stats(prices, temp, zkey), "rules": rules}


GROUPS = ["price", "derivatives", "onchain", "flows", "sentiment"]
NAMES = {
    "price": ["相對 200 日均線", "200 週均線倍數", "Pi 週期比值", "距最高價的跌幅", "30 天已實現波動率", "50 日與 200 日均線差距",
              "相對強弱指標（RSI，14 天）", "梅耶倍數", "年初至今報酬", "90 天報酬", "與那斯達克的 90 天相關係數", "成交量 30 天平均"],
    "derivatives": ["資金費率 30 天平均", "資金費率最新一期", "未平倉量", "未平倉量對市值比", "多空人數比", "大戶多空持倉比",
                    "期貨基差（三個月年化）", "選擇權 25 Delta 偏斜", "隱含波動率指數", "選擇權賣權買權比", "強制平倉金額（24 小時）", "期貨成交量"],
    "onchain": ["市值對已實現價值比（MVRV）", "已實現價格", "活躍地址數", "手續費收入", "算力", "交易所餘額",
                "長期持有者供給", "網路價值對交易量比（NVT）", "礦工收入", "新地址數", "轉帳金額中位數", "供給獲利比例"],
    "flows": ["現貨 ETF 30 日淨流入", "現貨 ETF 單日淨流入", "穩定幣供給 30 天變化", "穩定幣總供給", "交易所淨流入",
              "現貨 ETF 累計持有量", "灰度信託折溢價", "企業持幣量", "USDT 市占", "美元指數", "芝商所期貨大戶淨部位", "場外溢價"],
    "sentiment": ["恐懼貪婪指數", "維基百科關注度", "搜尋熱度", "社群討論量", "新聞情緒分數", "應用程式商店排名",
                  "比特幣市占率", "山寨幣季節指數", "散戶搜尋「買比特幣」", "論壇活躍度", "看多貼文比例", "開發者提交數"],
}
KEYS = {("price", 0), ("price", 1), ("price", 2), ("price", 3), ("derivatives", 0), ("derivatives", 2), ("onchain", 0),
        ("onchain", 5), ("flows", 0), ("flows", 2), ("sentiment", 0), ("sentiment", 1)}


def series(n, base, vol):
    out, v = [], base
    for _ in range(n):
        v += rnd.gauss(0, vol) + 0.02 * (base - v)
        out.append(round(v, 4))
    return out


def make_indicators():
    rows, k = [], 0
    for g in GROUPS:
        for j, nm in enumerate(NAMES[g]):
            k += 1
            is_key = (g, j) in KEYS
            if is_key:
                asset = "MARKET" if nm in ("恐懼貪婪指數", "穩定幣供給 30 天變化") else "BTC"
            else:
                asset = ["BTC", "BTC", "ETH", "SOL", "MARKET"][(k + j) % 5]
            n = N if (k % 4 or is_key) else rnd.choice([30, 90, 400])
            base = rnd.choice([1.2, 50, 3200, 0.01, 85000000000, -3])
            vals = series(n, base, abs(base) * 0.03)
            hot = "high" if is_key else [None, "high", "high", "low"][k % 4]
            pct = None if n < 180 else round(100.0 * sum(1 for v in vals if v <= vals[-1]) / n, 1)
            heat = None if (hot is None or pct is None) else (pct if hot == "high" else round(100 - pct, 1))
            unit = rnd.choice(["倍", "%", "美元", "", "億美元"])
            v = vals[-1]
            if abs(v) >= 1e8:
                unit = "億美元"
                disp = "{:,.0f} 億美元".format(v / 1e8)
            else:
                disp = ("{:,.0f}".format(v) if abs(v) >= 1000 else "{:.3g}".format(v)) + (" " + unit if unit else "")
            if pct is None:
                plain = "歷史不到 180 天，還不能判斷冷熱。"
            else:
                plain = "%s在過去 4 年裡%s（第 %d 百分位）。" % (
                    nm, "偏高" if pct > 60 else "偏低" if pct < 40 else "在中間", round(pct))
            rows.append({
                "id": "%s_%02d" % (g, j), "group": g, "asset": asset, "name": nm, "value": v, "unit": unit,
                "display": disp, "asof": DAYS[-1],
                "source": rnd.choice(["幣安", "CoinMetrics 社群版", "Alternative.me", "Farside", "DefiLlama"]),
                "source_url": "https://example.com/" + g, "evidence": rnd.choice(["A", "B", "C"]),
                "reading": rnd.choice(["bullish", "neutral", "bearish", "none"]),
                "rule": "高於長期平均視為偏熱，低於視為偏冷。",
                "note": "這個指標衡量%s，數值越高通常代表市場越樂觀。來源每天更新一次。" % nm,
                "history": [[DAYS[N - n + i], vals[i]] for i in range(n)], "error": None,
                "hot_when": hot, "pct": pct, "heat": heat, "key": is_key, "plain": plain})
    # 幾種壞情況
    dead = dict(value=None, display=None, history=[], heat=None, pct=None, plain=None, asof=None)
    rows[5].update(dead, error="HTTPError: 451 Client Error: 來源拒絕這個地區的連線")
    rows[16].update(dead, error="ReadTimeout: 30 秒內沒有回應")
    rows[14].update(dead, error="HTTPError: 403 被來源擋下")  # 重點指標（未平倉量）也失敗一個
    rows[30].update(asof="2026-11-02")  # 資料過舊
    rows[29].update(asof="2026-10-20", history=rows[29]["history"][:-22])  # 重點指標（交易所餘額）過舊
    rows[40].update(display=None, unit=None, note=None, rule=None, source_url=None, plain=None)
    rows[41].update(history=None, pct=None, heat=None)
    rows[42]["history"] = [[d, (None if i % 50 == 0 else v)] for i, (d, v) in enumerate(rows[42]["history"])]
    rows[43].update(history=[[DAYS[-1], 3.2]])
    for field in ("plain", "hot_when", "key"):
        del rows[44][field]
    return rows


def make_payload():
    assets = [make_asset("BTC", "比特幣", 21000, 0.0009, 0.027, 0.3, 41.3),
              make_asset("ETH", "以太幣", 1500, 0.0005, 0.034, 0.9, 27.4),
              make_asset("SOL", "索拉納", 22, 0.0013, 0.05, 1.6, 68.2)]
    tools = json.loads((HERE.parent / "tools.json").read_text(encoding="utf-8"))
    return {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": "這個頁面只整理公開資料與歷史統計，描述的是市場狀態，不是對任何人的買賣建議。"
                      "溫度是事後排名：用整段歷史算百分位，過去某一天的溫度在當時並不知道。",
        "assets": assets, "indicators": make_indicators(),
        "sources": [{"name": "price", "collected_utc": "2026-11-11T11:58:02Z", "count": 12, "failed": 1, "error": None},
                    {"name": "derivatives", "collected_utc": "2026-11-11T11:58:40Z", "count": 12, "failed": 2,
                     "error": None},
                    {"name": "onchain", "collected_utc": "2026-11-09T06:10:00Z", "count": 12, "failed": 0,
                     "error": "ConnectionError: 連不上 CoinMetrics，沿用舊快取"},
                    {"name": "flows", "collected_utc": "2026-11-11T11:59:30Z", "count": 12, "failed": 0, "error": None},
                    {"name": "sentiment", "collected_utc": None, "count": 12, "failed": 0, "error": None}],
        "tools": tools}


def nulls_variant(p):
    """把很多欄位弄成 null 或拿掉，確認頁面不會壞。"""
    q = json.loads(json.dumps(p))
    b, e, s = q["assets"]
    b.update(change_24h=None, temp_change_7d=None, price_display=None, summary=None, rules=None)
    b["zone_stats"][0].update(fwd30_median=None, fwd30_up=None, fwd90_median=None, fwd90_up=None, days=0)
    b["components"][0].update(heat=None, plain=None, display=None)
    e.update(temperature=None, zone=None, zone_label=None, temp_history=[], zone_stats=[], components=None,
             temp_change_7d=None, temp_change_30d=None, summary=["可用的成分不到 3 個，這次不給溫度。"])
    s.update(price=None, price_display=None, price_history=None, change_24h=None, change_7d=None, change_30d=None)
    s["temp_history"] = s["temp_history"][-40:]
    del s["rules"], s["zone_label"]
    q["assets"].append(None)
    q["indicators"] = q["indicators"][:20] + [None, {"id": "bare"}]
    q["sources"] = None
    q["tools"] = None
    q["disclaimer"] = None
    q["built_utc"] = "2026-11-08T01:00:00Z"
    return q


def write(name, payload):
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count(PLACEHOLDER) == 1, "template.html 裡必須剛好有一個 placeholder"
    (HERE / name).write_text(template.replace(PLACEHOLDER, embed(payload)), encoding="utf-8")
    print("已輸出", HERE / name)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    if "--real" in sys.argv:
        write("index_real.html", json.loads((HERE.parent / "out" / "data.json").read_text(encoding="utf-8")))
    else:
        payload = make_payload()
        (HERE / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        write("index.html", payload)
        write("index_nulls.html", nulls_variant(payload))
        write("index_empty.html", {})
