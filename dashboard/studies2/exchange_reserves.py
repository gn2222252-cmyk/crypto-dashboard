"""Historical study "exchange_reserves": Hoang & Baur (SSRN; summarised by Quantpedia) find that daily changes in
bitcoin exchange reserves (CryptoQuant, 20 exchanges) are negatively related to current and future returns, that
only reserve DECREASES go with positive long-term returns, and that reserve INCREASES raise volatility more.
Practitioners read exchange outflows as holder confidence (The Block, 2023); Macro Hive asked whether flows onto
exchanges predict price moves. Does the free data reproduce this, and does it improve the dashboard's main signal
(close > 50-day average, ma50) on main_signal_check's fair exam?

Data (checked before writing anything): CryptoQuant / Glassnode exchange balances need keys or paid plans. The
Coin Metrics community API (no key, no login) serves SplyExNtv (native units held in addresses Coin Metrics tags as
exchanges) for BTC from 2011-04-24 and ETH from 2015-07-30 - but for no other coin (SOL, BNB, XRP, LTC, ... are
rejected). Its own status-time field shows that every day before 2026-04-10 was written in one recomputation in
April 2026 (flow_filters found the same): the history is TODAY's address labels applied backwards, not what anyone
saw at the time. That cannot be removed; it is measured (status times, the dashboard's onchain cache, the vintage
stored by the previous run of this study) and said in the verdict.

Pre-registered design (fixed before running; no other variants):
Features (on the data's own dates): reserve change over k days = log(SplyEx_D / SplyEx_{D-k}), k = 1, 7, 30.
    Point-in-time quintile: the change of day D ranked among the 365 values D-364 ... D (quintile 1 = biggest
    decreases, 5 = biggest increases); needs all 365 values.
Timing: the value of on-chain day D is used only from the daily close of D+1 (UTC closes, te._daily_close, no extra
    shift); on the 24-clock grid a close on UTC date E uses the value of E-2. Never today's unfinished day.
Description (2013-01-01 on; periods 2013-17, 2018-21, 2022-now):
  D1 forward 1 / 7 / 30 / 90-day bitcoin return from the decision close, mean by quintile of the 1, 7, 30-day change;
     Q1 - Q5 spread with a 90% block-bootstrap interval (blocks 30/60/60/120 days).
  D2 asymmetry (30-day change, forward 90 days): Q1 - Q3 (strong decreases vs middle) and Q3 - Q5 (middle vs strong
     increases); the claim predicts Q1 - Q3 > 0 and larger than Q3 - Q5; plus mean forward 90 days when the 30-day
     change is < 0 vs > 0.
  D3 volatility: forward 30-day realised volatility (daily returns t+1 ... t+30, x sqrt 365) and its ratio to the
     trailing 30-day volatility, by quintile of the 7-day change; the claim predicts Q5 > Q1.
  D4 same day: correlation of the 1-day change of day D with bitcoin's return over the same UTC day (not tradable).
Rules:
  X1 = ma50 AND 30-day reserve change < 0 (unknown change = plain ma50).
  X2 = ma50 OR early entry: a close at which the usable 30-day change is in quintile 1 ("reserves falling strongly")
       starts a 30-close hold (a new trigger restarts the 30 closes); position = max(ma50, early hold).
Exam (main_signal_check / lead_followup arithmetic): 108 monthly starts 2015-01 ... 2023-12 on the UTC daily
  closes, textbook (0.10%); 72 starts x 24 closing hours (Binance 1-hour grid), textbook; the same cells, ordinary
  person (10 hours later, 0.25%, 10% of entries missed, 200 simulations per clock, ma50 with main_signal_check's
  seeds); other coins: ETH with its own Coin Metrics reserves; SOL, BNB, XRP and MSTR (main_signal_check's set) use
  bitcoin's reserves as a market-wide filter (Coin Metrics has no reserves for them). Ranking score = lowest of the
  four "beats holding" shares, eligible when the deepest loss is shallower than holding's in >= 95% of bitcoin
  starts and cells. Median deepest loss = median over the 1,728 textbook cells. Period splits (2013-17, 2018-21,
  2022-now; starts to six months before each end) are shown as well.
Placebo: the daily log changes of the BTC reserve series (ETH's by the same number of days) rolled circularly by
  180 ... (length - 180) days, levels rebuilt by cumulative sum, 200 draws, the whole exam rebuilt with 20
  ordinary-person simulations per clock (the real rule scored the same way); share of draws ranked at least as high.
Pass test (stated now): the predicted sign (30-day change, Q1 - Q5 forward 30 days > 0) in all three periods AND
  ranking score >= ma50's (and eligible when ma50 is) AND median deepest loss no worse than ma50's AND placebo share
  <= 10%. Only public key-less endpoints with the dashboard's fixed user agent (common.HEADERS).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import attention as at
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_CAT = "https://community-api.coinmetrics.io/v4/catalog/asset-metrics?metrics=SplyExNtv"
SRC_QP = "https://quantpedia.com/bitcoin-returns-and-volatility-predicted-by-bitcoin-exchange-reserves/"
SRC_TB = ("https://www.theblock.co/news/markets/2023-08-31-bitcoin-exchange-outflows-signal-long-term-holder-"
          "confidence-analyst-says-248559")
SRC_MH = "https://macrohive.com/hive-exclusives/do-flows-onto-crypto-exchanges-predict-bitcoin-price-moves"

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = msc.YEAR
LOSS_BAR = msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
WIN = 365
KS = (1, 7, 30)
HS = (1, 7, 30, 90)
BLOCK = {1: 30, 7: 60, 30: 60, 90: 120}
HOLD = 30
DESC_FROM = "2013-01-01"
PERIODS = (("2013 到 2017", "2013-01-01", "2017-12-31"), ("2018 到 2021", "2018-01-01", "2021-12-31"),
           ("2022 到現在", "2022-01-01", None))
LAST_GAP = 182
DRAWS, GAP, PL_SIMS = 200, 180, 20
LAG = {"utc": 1, "clock": 2, "share": 1}
RULES = ("x1", "x2")
KI = {"ma50": 1, "x1": 91, "x2": 92}  # ma50 = main_signal_check's draws
NAMES = {"ma50": "50 日均線（基準，主訊號）", "x1": "X1：50 日均線，而且交易所餘額 30 天在減少才持有",
         "x2": "X2：50 日均線，或交易所餘額 30 天減少幅度落在過去一年最大的兩成時提早進場、抱 30 天"}
SHORT = {"ma50": "50 日均線", "x1": "X1", "x2": "X2"}
CUTS = ("2019-06-30", "2022-06-30", "2025-06-30")

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return (f"{v * 100:.0f}%" if abs(v * 100) >= 0.5 else "0%") if _ok(v) else "—"


def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def _ci(lo, hi) -> str:
    return f"［{_p(lo)}，{_p(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _fetch(asset: str) -> pd.DataFrame:
    params = {"assets": asset, "metrics": "SplyExNtv", "frequency": "1d", "page_size": 10000}
    reply = common.get_json(CM_URL, params, timeout=90)
    rows = list(reply.get("data") or [])
    for _ in range(10):
        if not reply.get("next_page_url"):
            break
        reply = common.get_json(reply["next_page_url"], timeout=90)
        rows += list(reply.get("data") or [])
    if len(rows) < 1500:
        raise RuntimeError(f"Coin Metrics {asset} 交易所餘額只有 {len(rows)} 天")
    f = pd.DataFrame(rows)
    f.index = pd.to_datetime(f["time"].str[:10])
    out = pd.DataFrame({"bal": pd.to_numeric(f["SplyExNtv"], errors="coerce")})
    st = f.get("SplyExNtv-status-time")
    out["written"] = pd.to_datetime(st.str[:10], errors="coerce") if st is not None else pd.NaT
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out.index < pd.Timestamp(common.today_utc())) & (out["bal"] > 0)]
    return out


def _store(btc: pd.Series) -> dict:
    """Keep today's vintage; compare with the one stored by an earlier run (revisions)."""
    path = common.STORE / "exchange_reserves_cm.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        old = {}
    rev = {}
    now = {k.strftime("%Y-%m-%d"): float(v) for k, v in btc.items()}
    if old.get("data") and old.get("day") != common.today_utc():
        cut = (pd.Timestamp(old["day"]) - 3 * DAY).strftime("%Y-%m-%d")
        both = [(old["data"][k], v) for k, v in now.items() if k in old["data"] and k < cut]
        if both:
            diff = np.array([abs(b / a - 1) for a, b in both])
            rev = {"day": old["day"], "n": len(both), "max": float(diff.max()), "changed": int((diff > 1e-9).sum())}
    try:
        if old.get("day") != common.today_utc():
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"day": common.today_utc(), "data": now, "previous_day": old.get("day")}),
                            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return rev


def _onchain_cache(btc: pd.Series) -> dict | None:
    """The dashboard's onchain collector cache (an earlier fetch of the same series, rounded) vs today."""
    try:
        d = json.loads((common.CACHE / "onchain.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("id") == "btc_exchange_balance" and isinstance(o.get("history"), list):
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    if not found:
        return None
    h = pd.Series({pd.Timestamp(a): float(b) for a, b in found[0]["history"]})
    both = pd.concat([h, btc], axis=1, sort=True).dropna()
    if not len(both):
        return None
    rel = (both.iloc[:, 1] / both.iloc[:, 0] - 1).abs()
    return {"n": len(both), "from": both.index[0], "max": float(rel.max()), "big": int((rel > 0.001).sum()),
            "asof": found[0].get("asof", "")[:10]}


def _load() -> dict:
    problems = []
    close = te._daily_close()
    b = _fetch("btc")
    try:
        e = _fetch("eth")["bal"]
    except Exception as exc:  # noqa: BLE001 - ETH then uses plain ma50 with BTC's filter, said in notes
        e = None
        problems.append(f"以太幣交易所餘額這次取不到（{type(exc).__name__}），以太幣改用比特幣的餘額當全市場過濾。")
    other, more = msc._load_other()
    problems += more
    written = b["written"].dropna()
    batch = written[written.index < pd.Timestamp("2026-04-10")]
    wr = {"n": len(batch), "first": batch.min() if len(batch) else None, "last": batch.max() if len(batch) else None,
          "after": int((written.index >= pd.Timestamp("2026-04-10")).sum())}
    btc = b["bal"]
    return {"close": close, "btc": btc, "eth": e, "other": other, "problems": problems, "written": wr,
            "rev": _store(btc), "cache": _onchain_cache(btc)}


# ---------- features ----------

def _rank_q(x: pd.Series) -> pd.Series:
    """Point-in-time quintile (1 = lowest) of each value among the 365 values ending on its day."""
    v = x.values.astype(float)
    out = np.full(len(v), np.nan)
    if len(v) >= WIN:
        w = np.lib.stride_tricks.sliding_window_view(v, WIN)
        last = w[:, -1:]
        ok = np.isfinite(w).all(axis=1)
        r = (w < last).sum(axis=1) / (WIN - 1)
        q = np.clip(np.floor(r * 5) + 1, 1, 5)
        out[WIN - 1:] = np.where(ok, q, np.nan)
    return pd.Series(out, index=x.index)


def _features(bal: pd.Series | None) -> dict:
    if bal is None:
        return {}
    s = bal.asfreq("D").ffill(limit=3)
    lb = np.log(s)
    f = {}
    for k in KS:
        ch = lb - lb.shift(k)
        f[f"d{k}"] = ch
        f[f"q{k}"] = _rank_q(ch)
    return f


def _usable(series: pd.Series | None, dates, lag: int) -> np.ndarray:
    if series is None or not len(series):
        return np.full(len(dates), np.nan)
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


def _early(q: np.ndarray) -> np.ndarray:
    out = np.zeros(len(q))
    left = 0
    for i, v in enumerate(q.tolist()):
        if v == 1.0:
            left = HOLD
        if left > 0:
            out[i] = 1.0
            left -= 1
    return out


def _pos(kind: str, ma: np.ndarray, d30: np.ndarray, q30: np.ndarray) -> np.ndarray:
    if kind == "ma50":
        return ma
    if kind == "x1":
        falling = np.where(np.isnan(d30), 1.0, (d30 < 0).astype(float))
        return ma * falling
    return np.where(np.isnan(ma), np.nan, np.maximum(ma, _early(q30)))


# ---------- the exam ----------

class _Exam:
    def __init__(self, data: dict, price, b0, pre):
        close = data["close"]
        self.close = close
        self.btc_ma = msc._targets(close)["ma50"].values.astype(float)
        self.btc_change = close.pct_change().fillna(0.0).values
        self.dstarts, self.dlabels = msc._month_starts(close.index, *msc.DAILY_STARTS)
        hr, hd = msc._suffix(msc._equity(self.btc_change), self.dstarts, YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(msc._targets(pd.Series(c)).fillna(0.0)["ma50"].values[npre:][:self.days].astype(float))
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, (c, since, year) in data["other"].items():
            tg = msc._targets(c, year)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, pd.Timestamp(since or c.index[0]))
            if key == "ETH" and data["eth"] is not None:
                begin = max(begin, data["eth"].index[0] + (WIN + 31) * DAY)
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "change": change,
                               "starts": starts, "labels": labels, "year": year, "share": bool(since),
                               "hold": (hr[:, 0], hd[:, 0])}

    def btc_pos(self, kind: str, f: dict) -> np.ndarray:
        idx = self.close.index
        return _pos(kind, self.btc_ma, _usable(f["btc"].get("d30"), idx, LAG["utc"]),
                    _usable(f["btc"].get("q30"), idx, LAG["utc"]))

    def _run(self, pos, change, starts, year):
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        return r[:, 0], dd[:, 0]

    def summary(self, kind: str, f: dict, sims: int = SIMS) -> dict:
        dr, dd = self._run(self.btc_pos(kind, f), self.btc_change, self.dstarts, YEAR)
        dh = self.dhold
        d30 = _usable(f["btc"].get("d30"), self.m.dates, LAG["clock"])
        q30 = _usable(f["btc"].get("q30"), self.m.dates, LAG["clock"])
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = np.nan_to_num(_pos(kind, self.cma[h], d30, q30), nan=0.0).astype(float)
            r, d_ = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            pos = rt._missed(wl, P_MISS, rng, sims)
            r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, pos, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, d_
        hr, hd = self.chold
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        each = {}
        for key, o in self.other.items():
            src = f["eth"] if (key == "ETH" and f.get("eth")) else f["btc"]
            lag = LAG["share"] if o["share"] else LAG["utc"]
            pos = _pos(kind, o["ma"], _usable(src.get("d30"), o["close"].index, lag),
                       _usable(src.get("q30"), o["close"].index, lag))
            r, d_ = self._run(pos, o["change"], o["starts"], o["year"])
            each[key] = (len(r), int((r > o["hold"][0]).sum()), int((d_ > o["hold"][1]).sum()),
                         float(np.median(r - o["hold"][0])))
        x = {"d_n": len(dr), "d_beat": int((dr > dh[0]).sum()), "d_edge": float(np.median(dr - dh[0])),
             "d_shallow": int((dd > dh[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_p10": float(np.percentile(edge, 10)), "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= LOSS_BAR
        return x

    def periods(self, kind: str, f: dict) -> list[dict]:
        pos = self.btc_pos(kind, f)
        out = []
        for name, a, b in PERIODS:
            end = pd.Timestamp(b) if b else self.close.index[-1]
            keep = self.close.index <= end
            c = self.close[keep]
            ch = c.pct_change().fillna(0.0).values
            starts, _ = msc._month_starts(c.index, pd.Timestamp(a), end - LAST_GAP * DAY)
            r, dd = self._run(pos[keep], ch, starts, YEAR)
            hr, hd = msc._suffix(msc._equity(ch), starts, YEAR)
            out.append({"name": name, "n": len(r), "beat": int((r > hr[:, 0]).sum()),
                        "edge": float(np.median(r - hr[:, 0])), "deep": float(np.median(dd)),
                        "hold_deep": float(np.median(hd[:, 0]))})
        return out


def _score_ge(x: dict, base: dict) -> bool:
    if base["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not base["eligible"]:
        return True
    return x["low"] >= base["low"] - 1e-12


# ---------- descriptions ----------

def _desc(close: pd.Series, fb: dict) -> dict:
    idx = close.index
    c = close.values.astype(float)
    df = pd.DataFrame(index=idx)
    for k in KS:
        df[f"q{k}"] = _usable(fb[f"q{k}"], idx, LAG["utc"])
        df[f"d{k}"] = _usable(fb[f"d{k}"], idx, LAG["utc"])
    for h in HS:
        df[f"f{h}"] = close.shift(-h) / close - 1
    ret = close.pct_change()
    df["vol_f"] = ret[::-1].rolling(30, min_periods=30).std(ddof=1)[::-1].shift(-1) * math.sqrt(YEAR)
    df["vol_p"] = ret.rolling(30, min_periods=30).std(ddof=1) * math.sqrt(YEAR)
    df["vol_r"] = df["vol_f"] / df["vol_p"]
    df = df[df.index >= pd.Timestamp(DESC_FROM)]

    def sub(a, b):
        return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b or "2100-01-01"))]

    def spread(d, y, m1, m2, block, boot=True):
        a = (d[m1[0]] == m1[1]).astype(float).values
        b = (d[m2[0]] == m2[1]).astype(float).values if m2[1] is not None else (d[m2[0]] > 0).astype(float).values
        yy = d[y].values
        ok = np.isfinite(yy) & d[m1[0]].notna().values
        if boot:
            return at._boot({"y": yy[ok], "a": a[ok], "b": b[ok]},
                            lambda z: at._mmean(z["y"], z["a"]) - at._mmean(z["y"], z["b"]), block)
        m_a, m_b = ok & (a > 0), ok & (b > 0)
        if m_a.sum() < 20 or m_b.sum() < 20:
            return (None,)
        return (float(yy[m_a].mean() - yy[m_b].mean()),)

    out = {"means": {}, "spread": {}, "per_sign": [], "asym": {}, "vol": {}}
    for k in KS:
        for h in HS:
            for q in range(1, 6):
                m = (df[f"q{k}"] == q) & df[f"f{h}"].notna()
                out["means"][(k, h, q)] = (float(df.loc[m, f"f{h}"].mean()) if m.sum() else None, int(m.sum()))
            out["spread"][(k, h)] = spread(df, f"f{h}", (f"q{k}", 1), (f"q{k}", 5), BLOCK[h])
            out["spread"][(k, h, "per")] = [spread(sub(a, b), f"f{h}", (f"q{k}", 1), (f"q{k}", 5), BLOCK[h], False)[0]
                                            for _, a, b in PERIODS]
    out["sign"] = out["spread"][(30, 30, "per")]
    out["sign_ok"] = all(_ok(v) and v > 0 for v in out["sign"])
    out["asym"]["q1q3"] = spread(df, "f90", ("q30", 1), ("q30", 3), BLOCK[90])
    out["asym"]["q3q5"] = spread(df, "f90", ("q30", 3), ("q30", 5), BLOCK[90])
    out["asym"]["per"] = [(spread(sub(a, b), "f90", ("q30", 1), ("q30", 3), 120, False)[0],
                           spread(sub(a, b), "f90", ("q30", 3), ("q30", 5), 120, False)[0]) for _, a, b in PERIODS]
    neg, posi = df["d30"] < 0, df["d30"] > 0
    allm = df["f90"].mean()
    out["asym"]["neg"] = (float(df.loc[neg, "f90"].mean()), int(neg.sum()), float(allm))
    out["asym"]["pos"] = (float(df.loc[posi, "f90"].mean()), int(posi.sum()))
    for col in ("vol_f", "vol_r"):
        out["vol"][col] = {q: float(df.loc[df["q7"] == q, col].mean()) for q in range(1, 6)}
        tmp = df.copy()
        out["vol"][col + "_sp"] = at._boot({"y": tmp[col].values, "a": (tmp["q7"] == 5).astype(float).values,
                                            "b": (tmp["q7"] == 1).astype(float).values},
                                           lambda z: at._mmean(z["y"], z["a"]) - at._mmean(z["y"], z["b"]), 60)
        out["vol"][col + "_per"] = [spread(sub(a, b), col, ("q7", 5), ("q7", 1), 60, False)[0] for _, a, b in PERIODS]
    # same day: 1-day change of day D vs bitcoin's return of day D
    same = pd.DataFrame({"x": fb["d1"].reindex(idx), "y": close.pct_change()}).dropna()
    same = same[same.index >= pd.Timestamp(DESC_FROM)]
    out["same"] = at._boot({"x": same["x"].values, "y": same["y"].values}, at._corr_fn, 30)
    out["same_per"] = []
    for _, a, b in PERIODS:
        s2 = same[(same.index >= pd.Timestamp(a)) & (same.index <= pd.Timestamp(b or "2100-01-01"))]
        out["same_per"].append(float(np.corrcoef(s2["x"], s2["y"])[0, 1]) if len(s2) > 30 else None)
    out["n_days"] = int(df["q30"].notna().sum())
    return out


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: _Exam, f: dict) -> tuple[int, int]:
    rng = np.random.default_rng(11)
    close, bal = data["close"], data["btc"]
    total = bad = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        b2 = bal.where(bal.index <= c - DAY, bal * np.exp(rng.normal(0, 0.3, len(bal))))
        f2 = {"btc": _features(b2), "eth": f.get("eth")}
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        m = close.index <= c
        for kind in RULES:
            a = ex.btc_pos(kind, f)
            b = _pos(kind, ma2, _usable(f2["btc"]["d30"], close.index, LAG["utc"]),
                     _usable(f2["btc"]["q30"], close.index, LAG["utc"]))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[m], nan=-1), np.nan_to_num(b[m], nan=-1)))
        j = int(ex.m.dates.searchsorted(c))
        b3 = bal.where(bal.index < c - DAY, bal * np.exp(rng.normal(0, 0.3, len(bal))))
        f3 = _features(b3)
        x0 = _usable(f["btc"]["q30"], ex.m.dates, LAG["clock"])[:j + 1]
        x1 = _usable(f3["q30"], ex.m.dates, LAG["clock"])[:j + 1]
        total += 1
        bad += int(not np.array_equal(np.nan_to_num(x0, nan=-1), np.nan_to_num(x1, nan=-1)))
    return total, bad


# ---------- the computation ----------

def _shifted(bal: pd.Series, sh: int) -> pd.Series:
    s = bal.asfreq("D").ffill(limit=3).dropna()
    lr = np.diff(np.log(s.values))
    lr = np.roll(lr, sh % len(lr))
    lv = np.concatenate([[math.log(s.values[0])], math.log(s.values[0]) + np.cumsum(lr)])
    return pd.Series(np.exp(lv), index=s.index)


def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    f = {"btc": _features(data["btc"]), "eth": _features(data["eth"])}
    s = {k: ex.summary(k, f) for k in ("ma50",) + RULES}
    per = {k: ex.periods(k, f) for k in ("ma50",) + RULES}
    pos = {k: ex.btc_pos(k, f) for k in ("ma50",) + RULES}
    desc = _desc(close, f["btc"])

    real_small = {k: ex.summary(k, f, PL_SIMS) for k in RULES}
    prng = np.random.default_rng(SEED + 31)
    n = len(data["btc"])
    hits = {k: 0 for k in RULES}
    lows = {k: [] for k in RULES}
    for sh in prng.integers(GAP, n - GAP, DRAWS):
        f2 = {"btc": _features(_shifted(data["btc"], int(sh))),
              "eth": _features(_shifted(data["eth"], int(sh))) if data["eth"] is not None else {}}
        for k in RULES:
            x = ex.summary(k, f2, PL_SIMS)
            lows[k].append(x["low"])
            hits[k] += int(_score_ge(x, real_small[k]))
    placebo = {k: hits[k] / DRAWS for k in RULES}
    placebo_med = {k: float(np.median(lows[k])) for k in RULES}

    first15 = int(close.index.searchsorted(pd.Timestamp("2015-01-01")))
    held = {k: float(np.nanmean(pos[k][first15:])) for k in pos}
    differ = {k: float(np.mean(np.nan_to_num(pos[k][first15:]) != np.nan_to_num(pos["ma50"][first15:]))) for k in pos}
    trades = {k: float(np.sum(np.diff(np.nan_to_num(pos[k][first15:])) > 0.5) / ((len(close) - first15) / YEAR))
              for k in pos}
    look = _lookahead(data, ex, f)
    return {"ex": ex, "f": f, "s": s, "per": per, "pos": pos, "desc": desc, "real_small": real_small,
            "placebo": placebo, "placebo_med": placebo_med, "held": held, "differ": differ, "trades": trades,
            "look": look, "problem": problem, "secs": time.time() - t0}


# ---------- today ----------

def _now(data: dict, r: dict) -> list[dict]:
    close, f, pos = data["close"], r["f"]["btc"], r["pos"]
    last = close.index[-1]
    vday = last - DAY

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    def val(key):
        v = f[key].get(vday) if vday in f[key].index else None
        return float(v) if _ok(v) else None

    bal = data["btc"].get(vday)
    parts = []
    for k in KS:
        d, q = val(f"d{k}"), val(f"q{k}")
        parts.append(f"{k} 天 {_p(math.expm1(d) if _ok(d) else None)}（過去一年第 {int(q) if _ok(q) else '—'} 組）")
    q_use = _usable(f["q30"], close.index, LAG["utc"])
    early = _early(q_use)
    hits = np.flatnonzero(q_use == 1.0)
    left = (HOLD - (len(q_use) - 1 - int(hits[-1]))) if len(hits) else 0
    return [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"這個收盤能用的交易所餘額（Coin Metrics {_d(vday)} 的值）",
         "value": (f"{float(bal) / 1e4:,.1f} 萬顆" if _ok(bal) else "—") + "；變化：" + "、".join(parts)
                  + "（第 1 組＝減少最多的兩成，第 5 組＝增加最多的兩成）"},
        {"label": "50 日均線（主訊號）", "value": f"{word(pos['ma50'][-1])}（50 日平均 {float(close.iloc[-50:].mean()):,.0f} 美元）"},
        {"label": "X1：50 日均線＋餘額 30 天在減少", "value": word(pos["x1"][-1])},
        {"label": "X2：50 日均線，或餘額大減時提早進場抱 30 天",
         "value": word(pos["x2"][-1]) + (f"（提早進場的部分還會再抱 {left - 1} 個收盤，除非又出現大減就重新數 30 天）" if early[-1] >= 0.5 else "")},
        {"label": "這幾格的意思", "value": "只描述資料和規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]


# ---------- the study ----------

def study() -> dict:
    data = _load()
    r = _compute(data)
    close, s, desc, per = data["close"], r["s"], r["desc"], r["per"]
    m = s["ma50"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    ok = {}
    for k in RULES:
        x = s[k]
        ok[k] = {"sign": desc["sign_ok"], "score": _score_ge(x, m),
                 "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10}
        ok[k]["all"] = all(ok[k].values())
    passed = [k for k in RULES if ok[k]["all"]]
    pnames = [p[0] for p in PERIODS]
    n_d, n_c, n_p = m["d_n"], m["c_n"], m["p_n"]
    assets = list(r["ex"].other)

    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    # table 1: quintiles x horizons (30-day change), plus Q1-Q5 for each k
    rows1 = []
    for k in KS:
        for q in range(1, 6):
            tag = "（減少最多）" if q == 1 else "（增加最多）" if q == 5 else ""
            rows1.append([f"{k} 天變化 第 {q} 組{tag}"]
                         + [_p(desc["means"][(k, h, q)][0]) for h in HS] + [f"{desc['means'][(k, 30, q)][1]:,}"])
        cells = []
        for h in HS:
            sp = desc["spread"][(k, h)]
            cells.append(f"{_p(sp[0])} {_ci(sp[1], sp[2])}")
        rows1.append([f"{k} 天變化：第 1 組減第 5 組（研究說應該 > 0）"] + cells + ["—"])

    # table 2: periods for the description
    rows2 = []
    for k in KS:
        for h in (30, 90):
            rows2.append([f"{k} 天變化，之後 {h} 天：第 1 組減第 5 組"]
                         + [_p(v) for v in desc["spread"][(k, h, "per")]])
    for i, (lab, key) in enumerate((("強減少 vs 中間（第 1 組減第 3 組），之後 90 天", 0),
                                    ("中間 vs 強增加（第 3 組減第 5 組），之後 90 天", 1))):
        rows2.append([f"30 天變化：{lab}"] + [_p(p[key]) for p in desc["asym"]["per"]])
    rows2.append(["7 天變化：之後 30 天波動，第 5 組減第 1 組（研究說應該 > 0）"]
                 + [_p(v) for v in desc["vol"]["vol_f_per"]])
    rows2.append(["7 天變化：之後 30 天波動 ÷ 之前 30 天波動，第 5 組減第 1 組"]
                 + [f"{v:+.2f}" if _ok(v) else "—" for v in desc["vol"]["vol_r_per"]])
    rows2.append(["同一天：餘額 1 天變化和當天漲跌的相關係數（研究說應該 < 0）"]
                 + [f"{v:+.2f}" if _ok(v) else "—" for v in desc["same_per"]])

    # table 3: the exam
    rows3 = []
    for k in ("ma50",) + RULES:
        x = s[k]
        pp = "｜".join(_share(p["beat"] / p["n"]) if p["n"] else "—" for p in per[k])
        oth = "｜".join(f"{a} {x['o_each'][a][1]}/{x['o_each'][a][0]}" for a in assets if a in x["o_each"])
        rows3.append([NAMES[k], f"{_share(x['checks'][0])}｜{_pp(x['d_edge'])}", f"{_share(x['checks'][1])}｜{_pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{_pp(x['p_edge'])}", f"{_share(x['checks'][3])}（{oth}）", pp,
                      _share(x["low"]), f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}", _share(x["loss_ok"]),
                      f"{_share(r['held'][k])}｜{r['trades'][k]:.1f} 次｜{_share(r['differ'][k])}"])
    hd = r["ex"].chold[1]
    rows3.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", "—",
                  f"{_pct(float(np.median(hd)))}｜{_pct(float(hd.min()))}", "—", "100%｜0 次｜—"])

    # table 4: pass test
    sign_txt = "、".join(f"{n} {_p(v)}" for n, v in zip(pnames, desc["sign"]))
    rows4 = []
    for k in RULES:
        x = s[k]
        rows4.append([NAMES[k], f"{sign_txt}（{yes(ok[k]['sign'])}）",
                      f"{_share(x['low'])} vs {_share(m['low'])}（{yes(ok[k]['score'])}）",
                      f"{_pct(x['c_deep_med'])} vs {_pct(m['c_deep_med'])}（{yes(ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（亂移後分數中位數 {_share(r['placebo_med'][k])}，真的 "
                      f"{_share(r['real_small'][k]['low'])}）（{yes(ok[k]['placebo'])}）",
                      "通過" if ok[k]["all"] else "沒有通過"])

    # verdict
    sp30 = desc["spread"][(30, 30)]
    sp90 = desc["spread"][(30, 90)]
    a13, a35 = desc["asym"]["q1q3"], desc["asym"]["q3q5"]
    vsp = desc["vol"]["vol_f_sp"]
    if passed:
        k = passed[0]
        first = (f"有通過事先定好的考試：{'、'.join(SHORT[p] for p in passed)}。{SHORT[k]} 的排名分數 {_share(s[k]['low'])}"
                 f"（50 日均線 {_share(m['low'])}），最深虧損中位數 {_pct(s[k]['c_deep_med'])}（50 日均線 {_pct(m['c_deep_med'])}），"
                 f"亂移對照 {_share(r['placebo'][k])}。但資料是事後重算的，要打折看。")
    else:
        why = []
        for k in RULES:
            miss = [w for w, v in (("三段方向不一致", ok[k]["sign"]), ("排名分數輸 50 日均線", ok[k]["score"]),
                                   ("最深虧損比 50 日均線深", ok[k]["deep"]), ("亂移對照超過 10%", ok[k]["placebo"])) if not v]
            why.append(f"{SHORT[k]}（{'、'.join(miss)}）")
        first = "兩條交易所餘額規則都沒有通過事先定好的考試：" + "；".join(why) + "。"
    second = (f"用 Coin Metrics 免費資料重做論文的說法，結果和論文相反或看不出來：30 天餘額減少最多的兩成日子，之後 30 天平均漲跌"
              f"減掉增加最多的兩成是 {_p(sp30[0])}（90% 區間 {_ci(sp30[1], sp30[2])}，論文說應該是正的），之後 90 天是 "
              f"{_p(sp90[0])} {_ci(sp90[1], sp90[2])}；三段分別是 {sign_txt}。" if not desc["sign_ok"] else
              f"用 Coin Metrics 免費資料重做論文的說法：30 天餘額減少最多的兩成日子，之後 30 天平均漲跌減掉增加最多的兩成是 "
              f"{_p(sp30[0])}（90% 區間 {_ci(sp30[1], sp30[2])}），之後 90 天 {_p(sp90[0])} {_ci(sp90[1], sp90[2])}；"
              f"三段分別是 {sign_txt}。")
    third = (f"不對稱（之後 90 天平均漲跌的差）：強減少減中間 {_p(a13[0])}、中間減強增加 {_p(a35[0])}，論文說前者應該是正的而且比較大；"
             f"7 天餘額增加最多的兩成，之後 30 天的年化波動比減少最多的兩成高 {_p(vsp[0])} {_ci(vsp[1], vsp[2])}；"
             + ("波動這一點和論文一致。" if _ok(vsp[1]) and vsp[1] > 0 else "波動這一點看不出來。")
             + f"同一天餘額變化和漲跌的相關係數 {desc['same'][0]:+.2f}"
             + ("（論文說是負的，這點一致，但很小）。" if _ok(desc["same"][2]) and desc["same"][2] < 0 else "（論文說是負的，這裡看不出來）。"))
    x1, x2 = s["x1"], s["x2"]
    fourth = (f"規則：X1 排名分數 {_share(x1['low'])}、{n_c:,} 格每年比抱著多 {x1['c_edge']:+.1f} 點、最深虧損中位數 "
              f"{_pct(x1['c_deep_med'])}；X2 {_share(x2['low'])}、{x2['c_edge']:+.1f} 點、{_pct(x2['c_deep_med'])}；"
              f"50 日均線 {_share(m['low'])}、{m['c_edge']:+.1f} 點、{_pct(m['c_deep_med'])}。")
    fifth = ("注意：Coin Metrics 在 2026 年 4 月用今天認得的交易所地址把整段歷史重算過一次，當時的人看不到這些數字，"
             "所以任何看起來有用的結果都可能比當時真的做得到的好看。")
    verdict = first + second + third + fourth + fifth

    w = data["written"]
    look_c, look_b = r["look"]
    notes = [
        f"資料：Coin Metrics 免費社群版的 SplyExNtv（放在它認得的交易所地址裡的比特幣），{_d(data['btc'].index[0])} 起每天一筆，"
        f"本次用到 {_d(data['btc'].index[-1])}；不用金鑰、不用登入，用儀表板固定的 User-Agent 抓。論文用的是 CryptoQuant 的 20 家交易所資料，"
        "要付費，這裡沒有用；兩家認地址的方法不同，數字不會一樣。Coin Metrics 社群版只有比特幣和以太幣有這個指標（SOL、BNB、XRP、LTC 等都沒有）。",
        (f"歷史被整段重寫：資料自己的寫入時間顯示，2026-04-10 以前的 {w['n']:,} 天全部是在 "
         f"{_d(w['first']) if w['first'] is not None else '—'} 到 {_d(w['last']) if w['last'] is not None else '—'} 之間一次寫入的，"
         f"之後 {w['after']} 天才是每天即時寫的。Coin Metrics 的指標目錄也寫明：數值反映「目前已知」屬於交易所的地址，從每個地址第一次有餘額算起。"
         "也就是用今天的地址名單回頭算過去，回測不是「當時看得到的數字」。"),
        f"時間規則：餘額第 D 天的值，最早在 D+1 天的 UTC 0 點收盤才用；24 個收盤時間的格子裡，日期 E 的收盤只用 E−2 天的值。"
        f"打亂切點之後的價格和餘額再算一次，切點以前的決定不能變：{look_c} 次檢查有 {look_b} 次變了。",
    ]
    if data["cache"]:
        c = data["cache"]
        notes.append(f"和儀表板鏈上收集器存的同一條數列（{c['asof']} 抓的、四捨五入到整顆，{_d(c['from'])} 起 {c['n']:,} 天）比："
                     f"最大差 {c['max'] * 100:.3f}%，差超過 0.1% 的有 {c['big']} 天。只能看出最近這幾天有沒有再改，看不出 2026 年 4 月那次重算。")
    if data["rev"]:
        v = data["rev"]
        notes.append(f"和本研究上次執行（{v['day']}）存下的版本比：{v['n']:,} 天裡有 {v['changed']} 天的值不一樣，最大差 {v['max'] * 100:.3f}%。")
    else:
        notes.append("本研究第一次執行時把今天抓到的版本存下來，之後每次執行都會比對歷史有沒有再被改。")
    notes += [
        "「過去一年第幾組」：每天的變化放到包含當天在內的最近 365 天裡排名，分成五組，第 1 組是減少最多的兩成；只用當時以前的資料。",
        f"描述用的日子從 {DESC_FROM} 起（共 {desc['n_days']:,} 天有 30 天變化的分組）。之後的報酬天天重疊，真正獨立的段落遠少於天數；"
        "方括號是區塊重抽的 90% 區間。分段那張表沒有區間，只看方向。",
        "X2 的「大減」和「抱 30 天」是看結果之前定的：能用的 30 天變化落在過去一年第 1 組就進場（或重新數 30 天），30 個收盤後如果 50 日均線也沒在持有就出場。",
        "其他幣：以太幣用它自己的交易所餘額（Coin Metrics 有），SOL、BNB、XRP 和 MSTR 沒有自己的資料，改用比特幣的餘額當全市場過濾（50 日均線用它自己的）。"
        "以太幣的起算日從它有一年以上餘額資料之後開始。",
        f"亂移對照：把比特幣（和以太幣，同樣天數）交易所餘額的每日變化整條往後挪 {GAP} 天以上（頭尾相接）再累加回餘額，重抽 {DRAWS} 次、"
        f"整個考試重算；為了在幾分鐘內算完，一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的規則也用同樣方式算一次再比。",
        "排名分數和 main_signal_check 一樣是四項（日線起算日、起算日×收盤時間、一般人做法、其他資產）裡最低的那項；50 日均線用同樣的計算和亂數。",
        "成本很粗略：照課本做扣 0.10%、一般人做法扣 0.25%，沒有另外算滑價；現金沒有利息。一直抱著不扣成本。",
        "交易所餘額會因為交易所換錢包、合併地址、ETF 託管搬移、新交易所被認出來而跳動，這些都不是投資人真的把幣領走。",
        "過去的結果不保證未來會一樣。這裡只是比較幾條固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += problems

    hcols = [f"之後 {h} 天平均" for h in HS]
    tables = [
        {"caption": ("一、交易所餘額變化分組後，比特幣之後的平均漲跌（2013 起，UTC 0 點日線，餘額晚一天才用）。每種變化（1、7、30 天）"
                     "都依過去一年的排名分成五組；最後一列是第 1 組減第 5 組，方括號是 90% 區間。論文說：餘額減少（流出）之後比較會漲，所以應該 > 0。"),
         "columns": ["分組"] + hcols + ["天數（30 天欄）"], "rows": rows1},
        {"caption": ("二、分三段看方向（只看正負，不附區間）：包括不對稱（強減少 vs 中間、中間 vs 強增加）、波動（7 天增加最多的兩成"
                     "之後 30 天的年化波動減掉減少最多的兩成）和同一天的相關係數。"),
         "columns": ["比較"] + pnames, "rows": rows2},
        {"caption": (f"三、公平考試（main_signal_check 的做法）。每格「報酬贏過一直抱著的比例｜每年多幾個百分點（中位數）」："
                     f"比特幣 UTC 0 點日線 {msc.DAILY_STARTS[0][:7]} 到 {msc.DAILY_STARTS[1][:7]} 每月 1 日起算（{n_d} 個）、"
                     f"起算日 × 24 個收盤時間照課本做（{n_c:,} 格）、一般人做法（晚 10 小時、0.25%、漏掉 1 成進場，{n_p:,} 次模擬）、"
                     "其他資產（贏的起算日／起算日總數）；再加上三段期間各自贏的比例。排名分數＝前四項裡最低的；"
                     f"「虧得比抱著淺」是兩組比特幣檢查裡較低的那組，至少 {LOSS_BAR * 100:.0f}% 才列入排名。最後一欄從 2015 起算。"),
         "columns": ["規則", f"日線起算日 {n_d} 個", f"起算日×收盤時間 {n_c:,} 格", "一般人做法", "其他資產",
                     "三段期間：" + "｜".join(pnames), "排名分數（四項最低）", "最深虧損：中位數｜最深一格", "虧得比抱著淺",
                     "持有天數｜一年進場｜和 50 日均線不同的日子"],
         "rows": rows3},
        {"caption": ("四、事先定好的通過條件：30 天變化第 1 組減第 5 組（之後 30 天）在三段都 > 0、排名分數不低於 50 日均線、"
                     "最深虧損中位數不比 50 日均線深、亂移對照不超過 10%。四項都過才算通過。"),
         "columns": ["規則", "三段方向", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "亂移對照（≥ 真的分數的比例）", "結果"],
         "rows": rows4},
    ]
    return {
        "id": "exchange_reserves",
        "title": "交易所餘額減少，比特幣比較會漲嗎？",
        "question": ("論文說交易所的比特幣餘額減少（幣被領走）之後比較會漲、增加之後波動比較大。用免費的 Coin Metrics 資料重做一次，"
                     "這個關係還在嗎？拿來補強 50 日均線，在公平考試裡能不能贏？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(data, r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Coin Metrics 社群 API：比特幣、以太幣交易所餘額 SplyExNtv", "url": CM_URL},
            {"name": "Coin Metrics 社群 API 目錄：SplyExNtv 的定義與涵蓋的幣（只有 btc、eth）", "url": CM_CAT},
            {"name": "Quantpedia：Hoang & Baur 交易所餘額預測比特幣報酬與波動（摘要）", "url": SRC_QP},
            {"name": "The Block：交易所流出被解讀為長期持有者有信心", "url": SRC_TB},
            {"name": "Macro Hive：流進交易所的幣能預測比特幣價格嗎", "url": SRC_MH},
            {"name": "幣安現貨 BTC/USDT 日線與 1 小時線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
