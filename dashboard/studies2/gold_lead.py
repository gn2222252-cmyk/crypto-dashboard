"""Historical study "gold_lead": does gold lead bitcoin by about 4-7 months, so that bitcoin "catches up" with gold?

The claim (Bitwise Europe, André Dragosch, late 2025, a Granger-causality analysis; reported by CryptoSlate and
Cointelegraph): gold has led bitcoin by roughly 4-7 months, so after gold's 2025 rally bitcoin should catch up.
Counter-view seen in 2026: the bitcoin-gold correlation turned negative. The owner wants mechanisms and lead / lag
in both directions, not one straight-line correlation.

Pre-registered design (written before any result was seen; not changed afterwards):
  Data: gold = GLD (SPDR Gold Shares) adjusted daily closes from Yahoo's public chart endpoint (the same source
  rotation / cross_market / etf_loop use), from 2010; bitcoin = te._daily_close() (00:00 UTC candles; pre-2017-08
  closes were relabelled on 2026-11-19 to the day the candle covers; no extra shift added here).
  1. Lead / lag: weekly log returns, weeks ending Friday (gold = Friday's US close, bitcoin = the candle labelled
     Friday, which closes about three hours later). Cross-correlation corr(gold[t-L], btc[t]) and the reverse
     corr(btc[t-L], gold[t]) for L = 0 ... 40 weeks, by period of the target week: 2013-2017, 2018-2021, 2022-now,
     plus "after the claim was published" (2025-12-01 onward, about 50 weeks only - far too few to judge, stated).
  2. Granger-style test, both directions. Target y[t] (weekly log return); restricted model: constant, y's own lags
     1-4 one by one, sum of y's lags 5-16 and sum of lags 17-30. (a) Claim window: add x's lags 17-30 (about 4-7
     months) summed into one regressor; statistic = its coefficient beta; predicted sign for gold -> bitcoin: > 0.
     (b) Whole grid: add x's lags 1-40 as 8 sums of 5 weeks; statistic = F of the 8 added terms. Null distribution:
     x's weekly returns rebuilt by a circular moving-block bootstrap (blocks of 26 weeks, 1,000 draws, fixed seed)
     independently of y, regressors rebuilt, refitted; one-sided p for beta (share >= real beta) and p for F.
  3. Lag stability: rolling 2-year (104-week) windows ending at every quarter end from 2015-03-31 plus the last
     week; in each, the lag 0-40 with the highest correlation (gold -> bitcoin and the reverse); report the spread and
     the share of windows whose best lag lies in 17-30 weeks (by chance about 14 / 41 = 34%). Also the 52-week
     same-week correlation now and in 2026 (the "turned negative" claim).
  4. Mechanism check (description): bitcoin's next 13-week log return, sampled weekly, when the gold-lead signal
     L > 0 vs <= 0 and when the catch-up condition is on vs off, by period.
  5. Rules on bitcoin's daily closes. Gold-lead signal at decision day D: L = log(gold[D-122] / gold[D-213]) =
     gold's return over the window 7 to 4 months ago (gold as of the last US close on or before that calendar day).
     Catch-up condition: L >= log(1.05) ("strongly positive": gold up 5% or more in those 3 months) AND bitcoin
     lagged = log(btc[D-1] / btc[D-213]) < L (bitcoin's return since the start of that window, through the candle
     before, is below gold's). Rules: G1 = ma50 OR catch-up (also hold when below the 50-day average if the
     catch-up condition is on); G2 = ma50 AND L > 0. Description only: "gold only" = hold when L > 0.
  6. Fair exam exactly as main_signal_check, as ls_ratio / lead_followup do: 108 monthly starts 2015-01 ... 2023-12
     on the UTC daily closes (textbook, 0.10%), 72 monthly starts 2018-01 ... 2023-12 x 24 closing hours (1,728
     cells) textbook and ordinary person (decide at the clock, trade 10 hours later, 0.25%, 10% of entries missed,
     200 simulations per clock, main_signal_check's seeds for ma50), other coins ETH / SOL / BNB / XRP (Binance
     daily, their own ma50 and their own "lagged" return; gold is the same), monthly starts from the first month
     both rule and gold signal exist to three years ago. Ranking score = main_signal_check._summary's lowest of the
     four "beats holding" shares (eligible when deepest loss shallower than holding in >= 95% of the bitcoin starts
     and cells); median deepest loss over the 1,728 textbook cells. Placebo: gold's daily log returns rolled
     circularly by 180 ... (length - 180) days, 200 draws, the gold level rebuilt and the whole exam redone; share
     of draws whose ranking score is at least the real rule's (for time the ordinary-person part of the placebo uses
     50 simulations per clock, and the real rule is scored the same way for this comparison).
  Pass test (stated now), per rule: Granger claim-window beta (gold -> bitcoin) > 0 in all three periods AND the
  rule's ranking score >= ma50's (eligibility first) AND median deepest loss no worse than ma50's AND placebo share
  <= 10%.
  Point in time: the US close of day D is usable from the bitcoin candle labelled D (closes 00:00 UTC of D+1,
  three to four hours later; cross_market's convention); on the 24-clock grid a close at 23:00 UTC or later on date
  E reads gold up to E, earlier clocks up to E-1. The signal itself only reads gold 122+ days old. Look-ahead
  checks: gold and bitcoin after three cut dates scrambled -> no decision up to the cut may change; gold read seven
  days later.
  Today: L, the catch-up condition, the rule states - states only.

Only public, key-less sources with the dashboard's fixed user agent; a failed download falls back to our last stored
copy and says so. Descriptive and in-sample; not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import lead_followup as lf
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, LOSS_BAR = msc.YEAR, msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
GLD_FROM = 1262304000  # 2010-01-01
MAXLAG = 40
CLAIM_LAGS = (17, 30)  # weeks, about 4-7 months
NEAR, FAR = 122, 213  # days, about 4 and 7 months
STRONG = math.log(1.05)
BOOT, BLOCK = 1000, 26
ROLL_W = 104
DRAWS, GAP, PL_SIMS = 200, 180, 50
KI = {"ma50": 1, "g1": 80, "g2": 81}  # ma50 = main_signal_check's draws
CLOCK_LAG_HOUR = 23
PERIODS = (("2013–2017", "2013-01-01", "2017-12-31"), ("2018–2021", "2018-01-01", "2021-12-31"),
           ("2022 起", "2022-01-01", None))
CLAIM = ("說法發表後（2025-12 起）", "2025-12-01", None)
ALLP = PERIODS + (CLAIM, ("2013 起全部", "2013-01-01", None))
OTHER = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
CUTS = ("2016-06-30", "2020-03-15", "2024-06-30")
NAME = {"ma50": "50 日均線（基準）", "g1": "G1：50 日均線，或「黃金 4–7 個月前大漲而比特幣還沒跟上」時也持有",
        "g2": "G2：50 日均線，而且黃金 4–7 個月前是漲的才持有"}
SOURCES = [
    {"name": "CryptoSlate：Bitwise 的 André Dragosch 說黃金領先比特幣約 4–7 個月（Granger 因果分析）",
     "url": "https://cryptoslate.com/?p=514584"},
    {"name": "Cointelegraph：加密貨幣落後黃金和股票，2026 年可能補漲",
     "url": "https://cointelegraph.com/news/crypto-lagging-gold-stocks-2026-catch-up-rally"},
    {"name": "Yahoo Finance：GLD（SPDR 黃金基金）調整後日收盤", "url": "https://finance.yahoo.com/quote/GLD/history"},
    {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
    {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
]

_pct, _pp, _d, _num = cr._pct, cr._pp, cr._d, te._num
_ok, _share = lf._ok, lf._share


# ---------- data ----------

def _stored(name: str, fetch, least: int) -> tuple[dict, str]:
    path = common.STORE / f"gold_lead_{name}.json"
    note = ""
    try:
        days = fetch()
        if len(days) < least:
            raise RuntimeError(f"只回傳 {len(days)} 筆")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001
        if not path.exists():
            raise
        days = json.loads(path.read_text(encoding="utf-8"))
        note = f"這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）"
    return days, note


def _fetch_gld() -> dict:
    reply = requests.get(ro.YAHOO.format("GLD"), headers=common.HEADERS, timeout=30,
                         params={"period1": GLD_FROM, "period2": int(time.time()), "interval": "1d"})
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    s = pd.Series(result["indicators"]["adjclose"][0]["adjclose"],
                  index=pd.to_datetime(result["timestamp"], unit="s").normalize()).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[(s.index < pd.Timestamp(common.today_utc())) & (s > 0)]  # a bar dated today may still be open
    return {_d(k): float(v) for k, v in s.items() if math.isfinite(float(v))}


def _load() -> dict:
    problems = []
    days, note = _stored("gld", _fetch_gld, 3000)
    if note:
        problems.append(f"GLD 日線{note}。")
    gld = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    close = te._daily_close()
    other = {}
    for symbol, key in OTHER:
        try:
            other[key] = ta._crypto_close(symbol)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key} 這次取不到資料（{type(exc).__name__}），沒有算進去。")
    return {"gld": gld, "close": close, "other": other, "problems": problems}


def _calendar(gld: pd.Series, last: pd.Timestamp) -> pd.Series:
    """Gold as of the last US close on or before each calendar day."""
    cal = pd.date_range(gld.index[0], max(last, gld.index[-1]) + 2 * DAY, freq="D")
    return gld.reindex(cal.union(gld.index)).ffill().reindex(cal)


# ---------- signals ----------

def _flags(gold: pd.Series, asset: pd.Series) -> pd.DataFrame:
    """Calendar-day flags using gold up to that day and the asset up to the day before (unknown = 0)."""
    idx = pd.date_range(min(gold.index[0], asset.index[0]), max(gold.index[-1], asset.index[-1]), freq="D")
    g = gold.reindex(idx).ffill()
    a = asset.reindex(idx).ffill()
    lead = np.log(g.shift(NEAR) / g.shift(FAR))
    lag_a = np.log(a.shift(1) / a.shift(FAR))
    return pd.DataFrame({"L": lead, "A": lag_a, "lead": (lead > 0).astype(float),
                         "catch": ((lead >= STRONG) & (lag_a < lead)).astype(float)}, index=idx)


def _usable(fl: pd.DataFrame, col: str, dates, lag) -> np.ndarray:
    return fl[col].reindex(pd.DatetimeIndex(dates) - lag * DAY).fillna(0.0).values.astype(float)


def _pos(kind: str, ma: np.ndarray, lead: np.ndarray, catch: np.ndarray) -> np.ndarray:
    if kind == "ma50":
        return ma
    if kind == "g1":
        return np.where(np.isnan(ma), np.nan, np.maximum(ma, catch))
    return ma * lead


def _rebuilt(gold: pd.Series, shift: int) -> pd.Series:
    """Gold with its daily log returns rolled circularly (the placebo)."""
    lr = np.log(gold).diff().fillna(0.0).values
    return pd.Series(gold.iloc[0] * np.exp(np.cumsum(np.roll(lr, shift))), index=gold.index)


# ---------- weekly lead / lag ----------

def _weekly(gold: pd.Series, close: pd.Series) -> pd.DataFrame:
    last = min(close.index[-1], gold.index[-1])
    fr = pd.date_range("2010-08-06", last, freq="W-FRI")
    return pd.DataFrame({"g": np.log(gold.reindex(fr)).diff(), "b": np.log(close.reindex(fr)).diff()},
                        index=fr).iloc[1:]


def _csum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])


def _lagsum(cs: np.ndarray, t: np.ndarray, a: int, b: int) -> np.ndarray:
    """sum of x[t-b .. t-a]."""
    return cs[t - a + 1] - cs[t - b]


def _design(y: np.ndarray, x: np.ndarray, t: np.ndarray) -> tuple:
    cy, cx = _csum(y), _csum(x)
    base = [np.ones(len(t))] + [y[t - k] for k in range(1, 5)] + [_lagsum(cy, t, 5, 16), _lagsum(cy, t, 17, 30)]
    claim = _lagsum(cx, t, *CLAIM_LAGS)
    blocks = [_lagsum(cx, t, 1 + 5 * k, 5 + 5 * k) for k in range(8)]
    return np.column_stack(base), claim, np.column_stack(blocks)


def _fit(y: np.ndarray, X0: np.ndarray, claim: np.ndarray, blocks: np.ndarray) -> tuple:
    """(beta of the claim-window term, F of the 8 lag blocks)."""
    X1 = np.column_stack([X0, claim])
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    X2 = np.column_stack([X0, blocks])
    r2 = y - X2 @ np.linalg.lstsq(X2, y, rcond=None)[0]
    rss0, rss2 = float(r0 @ r0), float(r2 @ r2)
    df2 = len(y) - X2.shape[1]
    f = ((rss0 - rss2) / blocks.shape[1]) / (rss2 / df2) if df2 > 0 and rss2 > 0 else float("nan")
    return float(b1[-1]), f


def _masks(dates: pd.DatetimeIndex) -> dict:
    return {name: np.asarray((dates >= pd.Timestamp(a)) & ((dates <= pd.Timestamp(b)) if b else True))
            for name, a, b in ALLP}


def _granger(w: pd.DataFrame) -> dict:
    """Both directions, every period: real beta / F and bootstrap p values."""
    rng = np.random.default_rng(SEED + 5)
    out = {}
    first = int(np.argmax(w["g"].notna().values & w["b"].notna().values))
    t = np.arange(max(first, 0) + MAXLAG + 1, len(w))
    masks = {k: m[t] for k, m in _masks(w.index).items()}
    for direction, ycol, xcol in (("g2b", "b", "g"), ("b2g", "g", "b")):
        y_all, x_all = w[ycol].values, w[xcol].values
        X0, claim, blocks = _design(y_all, x_all, t)
        yt = y_all[t]
        real = {k: _fit(yt[m], X0[m], claim[m], blocks[m]) if m.sum() > 30 else (float("nan"), float("nan"))
                for k, m in masks.items()}
        hits = {k: [0, 0] for k in masks}
        xs = np.nan_to_num(x_all)
        n = len(xs)
        for _ in range(BOOT):
            starts = rng.integers(0, n, size=n // BLOCK + 2)
            xb = np.concatenate([np.take(xs, np.arange(s, s + BLOCK), mode="wrap") for s in starts])[:n]
            _, c2, bl2 = _design(y_all, xb, t)
            for k, m in masks.items():
                if m.sum() <= 30:
                    continue
                b, f = _fit(yt[m], X0[m], c2[m], bl2[m])
                hits[k][0] += int(b >= real[k][0])
                hits[k][1] += int(f >= real[k][1])
        out[direction] = {k: {"beta": real[k][0], "F": real[k][1], "p_beta": hits[k][0] / BOOT,
                              "p_F": hits[k][1] / BOOT, "n": int(masks[k].sum())} for k in masks}
    return out


def _xcorr(w: pd.DataFrame) -> dict:
    """corr(x[t-L], y[t]) for L = 0..40, both directions, per period."""
    out = {}
    masks = _masks(w.index)
    for direction, ycol, xcol in (("g2b", "b", "g"), ("b2g", "g", "b")):
        out[direction] = {}
        for k, m in masks.items():
            rs = []
            for L in range(MAXLAG + 1):
                both = pd.concat([w[xcol].shift(L)[m], w[ycol][m]], axis=1).dropna()
                rs.append(float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 20 else float("nan"))
            out[direction][k] = {"r": np.array(rs), "n": int(pd.concat([w["g"][m], w["b"][m]], axis=1).dropna().shape[0])}
    return out


def _rolling(w: pd.DataFrame) -> dict:
    ends = list(pd.date_range("2015-03-31", w.index[-1], freq="QE")) + [w.index[-1]]
    shifted = {d: {L: w[x].shift(L) for L in range(MAXLAG + 1)} for d, x in (("g2b", "g"), ("b2g", "b"))}
    out = {"g2b": [], "b2g": []}
    for e in ends:
        m = (w.index > e - pd.Timedelta(weeks=ROLL_W)) & (w.index <= e)
        for d, y in (("g2b", "b"), ("b2g", "g")):
            rs = {}
            for L in range(MAXLAG + 1):
                both = pd.concat([shifted[d][L][m], w[y][m]], axis=1).dropna()
                if len(both) > 60:
                    rs[L] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            if rs:
                best = max(rs, key=rs.get)
                out[d].append({"end": e, "best": best, "r": rs[best]})
    c52 = w["g"].rolling(52, min_periods=40).corr(w["b"])
    return {"best": out, "c52": c52}


# ---------- the fair exam ----------

class _Exam:
    """main_signal_check's four checks for ma50, G1 and G2 (lead_followup's arithmetic); everything that does not
    depend on gold is computed once."""

    def __init__(self, data: dict, price, b0, pre):
        close = data["close"]
        self.close = close
        self.btc_ma = msc._targets(close)["ma50"].values.astype(float)
        self.btc_change = close.pct_change().fillna(0.0).values
        gfirst = data["gld"].index[0] + (FAR + 1) * DAY
        self.dstarts, self.dlabels = msc._month_starts(close.index, *msc.DAILY_STARTS)
        hr, hd = msc._suffix(msc._equity(self.btc_change), self.dstarts, YEAR)
        self.dhold = (hr[:, 0], hd[:, 0])
        m = cr._Market(price, b0, pre)
        self.m, self.eng, self.days = m, rt._Engine(m), m.days
        npre = len(pre)
        self.cstarts, self.clabels = msc._month_starts(m.dates, *msc.CLOCK_STARTS)
        hr, hd = msc._suffix(msc._equity(m.change), self.cstarts, YEAR)
        self.chold = (hr[:, 0], hd[:, 0])
        self.cma = []
        for h in range(24):
            c = np.concatenate([pre, price[h::24]]).astype(float)
            self.cma.append(msc._targets(pd.Series(c)).fillna(0.0)["ma50"].values[npre:][:self.days].astype(float))
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        self.other = {}
        for key, c in data["other"].items():
            tg = msc._targets(c, YEAR)
            valid = tg.notna().all(axis=1)
            begin = max(c.index[int(np.argmax(valid.values))] + DAY, gfirst, c.index[0] + (FAR + 1) * DAY)
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "change": change,
                               "starts": starts, "labels": labels, "hold": (hr[:, 0], hd[:, 0])}

    def flags(self, gold: pd.Series) -> dict:
        out = {"btc": _flags(gold, self.close)}
        for key, o in self.other.items():
            out[key] = _flags(gold, o["close"])
        return out

    def btc_pos(self, kind: str, fl: dict, extra: int = 0) -> np.ndarray:
        if kind == "ma50":
            return self.btc_ma
        f = fl["btc"]
        idx = self.close.index
        return _pos(kind, self.btc_ma, _usable(f, "lead", idx, extra), _usable(f, "catch", idx, extra))

    def summary(self, kind: str, fl: dict, sims: int = SIMS) -> dict:
        pos = self.btc_pos(kind, fl)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        r, dd = msc._suffix(msc._equity(net), self.dstarts, YEAR)
        dr, dd = r[:, 0], dd[:, 0]
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            if kind == "ma50":
                w = self.cma[h]
            else:
                lag = 0 if h >= CLOCK_LAG_HOUR else 1
                w = _pos(kind, self.cma[h], _usable(fl["btc"], "lead", self.m.dates, lag),
                         _usable(fl["btc"], "catch", self.m.dates, lag))
            w = np.nan_to_num(w, nan=0.0).astype(float)
            r, d2 = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], d2[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            p = rt._missed(wl, P_MISS, rng, sims)
            r, d2 = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, p, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, d2
        each = {}
        for key, o in self.other.items():
            if kind == "ma50":
                po = o["ma"]
            else:
                f = fl[key]
                po = _pos(kind, o["ma"], _usable(f, "lead", o["close"].index, 0), _usable(f, "catch", o["close"].index, 0))
            net, _ = te._run(np.concatenate([[np.nan], po[:-1]]), o["change"], COST)
            r, d2 = msc._suffix(msc._equity(net), o["starts"], YEAR)
            each[key] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((d2[:, 0] > o["hold"][1]).sum()),
                         float(np.median(r[:, 0] - o["hold"][0])))
        hr, hd = self.chold
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        x = {"d_n": len(dr), "d_beat": int((dr > self.dhold[0]).sum()), "d_edge": float(np.median(dr - self.dhold[0])),
             "d_shallow": int((dd > self.dhold[1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)),
             "o_each": each, "o_n": sum(v[0] for v in each.values()), "o_beat": sum(v[1] for v in each.values())}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= LOSS_BAR
        return x


def _measure(pos: np.ndarray, close: pd.Series, a, b) -> dict | None:
    change = close.pct_change().fillna(0.0)
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), change.values, COST)
    s = pd.Series(net, index=close.index)
    held = pd.Series(np.concatenate([[np.nan], pos[:-1]]), index=close.index)
    m = (s.index >= pd.Timestamp(a)) & ((s.index <= pd.Timestamp(b)) if b else True) & held.notna().values
    return ro._measure(s[m].dropna(), YEAR)


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, gld = data["close"], data["gld"]
    gold = _calendar(gld, close.index[-1])
    w = _weekly(gold, close)
    xc = _xcorr(w)
    gr = _granger(w)
    roll = _rolling(w)

    # mechanism: next 13-week bitcoin log return by state, sampled weekly
    fl_btc = _flags(gold, close)
    fwd = np.log(close.shift(-91) / close)
    weeks = pd.date_range("2013-01-04", close.index[-92], freq="W-FRI")
    mech = {}
    st = fl_btc.reindex(weeks)
    fw = fwd.reindex(weeks)
    for name, a, b in ALLP:
        m = (weeks >= pd.Timestamp(a)) & ((weeks <= pd.Timestamp(b)) if b else True) & st["L"].notna().values
        row = {}
        for col in ("lead", "catch"):
            on, off = fw[m & (st[col] > 0.5).values].dropna(), fw[m & (st[col] < 0.5).values].dropna()
            row[col] = (float(on.mean() * 100) if len(on) else None, len(on),
                        float(off.mean() * 100) if len(off) else None, len(off))
        mech[name] = row

    # fair exam
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    fl = ex.flags(gold)
    s = {k: ex.summary(k, fl) for k in ("ma50", "g1", "g2")}
    t1 = time.time()

    # positions on the UTC daily series, by period
    pos = {"hold": np.ones(len(close)), "ma50": ex.btc_pos("ma50", fl), "gold": fl["btc"]["lead"].reindex(close.index).fillna(0.0).values,
           "g1": ex.btc_pos("g1", fl), "g2": ex.btc_pos("g2", fl)}
    gfirst = gld.index[0] + (FAR + 1) * DAY
    per = {}
    for name, a, b in ALLP:
        a2 = max(pd.Timestamp(a), gfirst)
        per[name] = {k: _measure(p, close, a2, b) for k, p in pos.items()}

    # placebo
    real_small = {k: ex.summary(k, fl, PL_SIMS) for k in ("g1", "g2")}
    prng = np.random.default_rng(SEED + 9)
    n = len(gold)
    hits = {"g1": 0, "g2": 0}
    lows = {"g1": [], "g2": []}
    for sh in prng.integers(GAP, n - GAP, DRAWS):
        f2 = ex.flags(_rebuilt(gold, int(sh)))
        for k in ("g1", "g2"):
            x = ex.summary(k, f2, PL_SIMS)
            lows[k].append(x["low"])
            hits[k] += int(lf._score_ge(x, real_small[k]))
    plc = {k: hits[k] / DRAWS for k in hits}
    plc_low = {k: float(np.median(lows[k])) for k in lows}
    t2 = time.time()

    # look-ahead checks
    rng = np.random.default_rng(11)
    total = bad = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        g2 = gold.where(gold.index <= c, gold * np.exp(rng.normal(0, 0.2, len(gold))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        f2 = _flags(g2, c2)
        mm = close.index <= c
        for k in ("g1", "g2"):
            a = ex.btc_pos(k, fl)
            b = _pos(k, ma2, _usable(f2, "lead", close.index, 0), _usable(f2, "catch", close.index, 0))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[mm], nan=-1), np.nan_to_num(b[mm], nan=-1)))
    late = {k: _measure(ex.btc_pos(k, fl, extra=7), close, max(pd.Timestamp("2013-01-01"), gfirst), None)
            for k in ("g1", "g2")}
    return {"gold": gold, "w": w, "xc": xc, "gr": gr, "roll": roll, "mech": mech, "ex": ex, "fl": fl, "s": s,
            "pos": pos, "per": per, "plc": plc, "plc_low": plc_low, "real_small": real_small,
            "look": (total, bad), "late": late, "problem": problem,
            "secs": (t1 - t0, t2 - t1, time.time() - t0)}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, gld, ex, s = data["close"], data["gld"], r["ex"], r["s"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    xc, gr, roll, per = r["xc"], r["gr"], r["roll"], r["per"]
    pnames = [p[0] for p in ALLP]
    three = [p[0] for p in PERIODS]
    lo, hi = CLAIM_LAGS

    # ----- pass test -----
    sign_ok = all(_ok(gr["g2b"][p]["beta"]) and gr["g2b"][p]["beta"] > 0 for p in three)
    base = s["ma50"]
    res = {}
    for k in ("g1", "g2"):
        x = s[k]
        score_ok = lf._score_ge(x, base)
        loss_ok = x["c_deep_med"] >= base["c_deep_med"] - 1e-12
        plc_ok = r["plc"][k] <= 0.10
        res[k] = {"sign": sign_ok, "score": score_ok, "loss": loss_ok, "plc": plc_ok,
                  "pass": sign_ok and score_ok and loss_ok and plc_ok}

    # ----- table 1: cross-correlation -----
    def best(d, p) -> str:
        x = xc[d][p]["r"]
        if not np.isfinite(x).any():
            return "—"
        L = int(np.nanargmax(x))
        return f"{L} 週（{_num(x[L])}）"

    def band(d, p) -> str:
        x = xc[d][p]["r"]
        return _num(float(np.nanmean(x[lo:hi + 1]))) if np.isfinite(x[lo:hi + 1]).any() else "—"

    rows1 = [["黃金 → 比特幣：相關最高的落後週數（相關）"] + [best("g2b", p) for p in pnames],
             [f"黃金 → 比特幣：落後 {lo}–{hi} 週（約 4–7 個月）的平均相關"] + [band("g2b", p) for p in pnames],
             ["比特幣 → 黃金：相關最高的落後週數（相關）"] + [best("b2g", p) for p in pnames],
             [f"比特幣 → 黃金：落後 {lo}–{hi} 週的平均相關"] + [band("b2g", p) for p in pnames],
             ["同一週（落後 0）的相關"] + [_num(xc["g2b"][p]["r"][0]) for p in pnames],
             ["週數｜純雜訊大約的範圍 ±2÷√週數"] + [f"{xc['g2b'][p]['n']}｜±{2 / math.sqrt(max(xc['g2b'][p]['n'], 1)):.2f}"
                                          for p in pnames]]
    t1 = {"caption": ("檢查一：每週報酬（取對數，週五收盤；黃金用 GLD 美股收盤、比特幣用標著週五的日線）的領先落後相關。"
                      "「黃金 → 比特幣 落後 L 週」＝ L 週前的黃金週報酬和這週比特幣週報酬的相關，反方向同理；掃 0 到 40 週，找最高的那個。"
                      "相關 +1 完全同向、0 沒關係。掃 41 個落後一定會挑到一個看起來不錯的，所以「最高」本身不能當證據，要和雜訊範圍比；"
                      "「說法發表後」只有約 50 週，幾乎不能下結論。"),
          "columns": ["比較"] + pnames, "rows": rows1}

    # ----- table 2: Granger -----
    def gcell(d, p, which) -> str:
        g = gr[d][p]
        if not _ok(g["beta"]):
            return "—"
        if which == "beta":
            return f"{g['beta']:+.3f}（p {g['p_beta']:.2f}）"
        return f"F {_num(g['F'])}（p {g['p_F']:.2f}）"

    rows2 = [[f"黃金 → 比特幣：{lo}–{hi} 週前黃金報酬的係數（說法預測 > 0）"] + [gcell("g2b", p, "beta") for p in pnames],
             ["黃金 → 比特幣：1–40 週全部落後一起測"] + [gcell("g2b", p, "F") for p in pnames],
             [f"比特幣 → 黃金：{lo}–{hi} 週前比特幣報酬的係數"] + [gcell("b2g", p, "beta") for p in pnames],
             ["比特幣 → 黃金：1–40 週全部落後一起測"] + [gcell("b2g", p, "F") for p in pnames],
             ["用到的週數"] + [str(gr["g2b"][p]["n"]) for p in pnames]]
    t2 = {"caption": ("檢查二：Granger 式的檢驗（「知道對方過去的走勢，能不能把自己下一週猜得更準」）。先用自己過去 30 週的報酬當基本模型，"
                      f"再加上對方 {lo}–{hi} 週前的報酬（一個數字，看係數正負）或對方 1–40 週前的報酬（分 8 段，看整體 F 值）。"
                      f"p 值：把對方的週報酬每 {BLOCK} 週一段隨機重排 {BOOT} 次（保留它自己的起伏、拆掉和這一方的時間對應），"
                      "重做一次，結果一樣大或更大的比例；越小越不像巧合，0.05 以下才算明顯。係數是「黃金那 14 週累積漲 1（對數）時，"
                      "比特幣下一週多漲多少（對數）」。"),
          "columns": ["檢驗"] + pnames, "rows": rows2}

    # ----- table 3: lag stability -----
    rows3 = []
    for d, lab in (("g2b", "黃金 → 比特幣"), ("b2g", "比特幣 → 黃金")):
        b = np.array([x["best"] for x in roll["best"][d]])
        if len(b):
            q = np.percentile(b, [0, 25, 50, 75, 100])
            inside = float(((b >= lo) & (b <= hi)).mean())
            rows3.append([f"{lab}：每個兩年窗口相關最高的落後週數",
                          f"{len(b)} 個窗口：最少 {q[0]:.0f}、四分之一 {q[1]:.0f}、中位 {q[2]:.0f}、四分之三 {q[3]:.0f}、最多 {q[4]:.0f} 週"])
            rows3.append([f"{lab}：最佳落後落在 {lo}–{hi} 週的窗口比例", f"{inside * 100:.0f}%（純靠運氣大約 34%）"])
            last5 = roll["best"][d][-5:]
            rows3.append([f"{lab}：最近 5 個窗口的最佳落後", "、".join(f"{_d(x['end'])} {x['best']} 週" for x in last5)])
    c52 = roll["c52"].dropna()
    y26 = c52[c52.index >= pd.Timestamp("2026-01-01")]
    old = c52[c52.index < pd.Timestamp("2026-01-01")]
    rows3.append(["同一週報酬的 52 週相關：現在", f"{_num(c52.iloc[-1])}（到 {_d(c52.index[-1])}）" if len(c52) else "—"])
    rows3.append(["同一週報酬的 52 週相關：2026 年最低｜最高",
                  f"{_num(y26.min())}｜{_num(y26.max())}" if len(y26) else "—"])
    rows3.append(["同一週報酬的 52 週相關：2025 年以前的中位數｜最低｜最高",
                  f"{_num(old.median())}｜{_num(old.min())}｜{_num(old.max())}" if len(old) else "—"])
    t3 = {"caption": (f"檢查三：落後週數穩不穩。每季底往回取兩年（{ROLL_W} 週）的週報酬，在 0–40 週裡找相關最高的落後；"
                      "如果黃金真的固定領先 4–7 個月，大多數窗口應該落在 17–30 週。最後三列是同一週報酬的 52 週滾動相關，看「2026 年轉成負相關」的說法。"),
          "columns": ["項目", "數字"], "rows": rows3}

    # ----- table 4: mechanism -----
    mech = r["mech"]

    def mc(p, col) -> str:
        on, n1, off, n0 = mech[p][col]
        if on is None or off is None:
            return f"成立 {n1} 週｜不成立 {n0} 週"
        return f"{_pct(on)}（{n1} 週）對 {_pct(off)}（{n0} 週）"

    rows4 = [["黃金 4–7 個月前是漲的（L > 0）對 不是"] + [mc(p, "lead") for p in pnames],
             ["補漲條件成立（黃金那段漲 ≥ 5%、比特幣還沒跟上）對 不成立"] + [mc(p, "catch") for p in pnames]]
    t4 = {"caption": ("檢查四（機制，只是描述）：每週取一天，看比特幣之後 13 週的報酬（取對數的平均）在訊號成立和不成立時差多少。"
                      "如果「黃金領先、比特幣補漲」是真的，前面的數字應該比較大，而且每段都是。之後 13 週互相重疊，實際獨立的段數約是週數 ÷ 13。"),
          "columns": ["狀態：成立 對 不成立"] + pnames, "rows": rows4}

    # ----- table 5: rules by period -----
    lab5 = {"hold": "一直抱著", "ma50": "50 日均線", "gold": "只看黃金：L > 0 才持有（描述）", "g1": "G1：均線 或 補漲條件",
            "g2": "G2：均線 而且 L > 0"}

    def c5(m, full=False) -> str:
        if not m:
            return "—"
        return f"{_num(m['ratio'])}｜{_pct(m['ret'])}｜{_pct(m['deepest'])}" if full else _num(m["ratio"])

    rows5 = [[lab5[k]] + [c5(per[p][k], p == pnames[-1]) for p in pnames] for k in ("hold", "ma50", "gold", "g1", "g2")]
    t5 = {"caption": ("比特幣 UTC 0 點日線照規則進出的報酬÷波動（收盤決定、隔天照做、每次買賣扣 0.10%，現金 0%）；最後一欄是報酬÷波動｜每年報酬｜最深曾經虧多少。"
                      "黃金訊號要 GLD 有 7 個月歷史才有值，所以各段從 "
                      f"{_d(gld.index[0] + (FAR + 1) * DAY)} 以後才算；「說法發表後」不到一年，只是看看。"),
          "columns": ["做法"] + pnames, "rows": rows5}

    # ----- table 6: fair exam and pass test -----
    rows6 = []
    for k in ("ma50", "g1", "g2"):
        x = s[k]
        ck = x["checks"]
        if k == "ma50":
            plc_txt, verdict_txt = "—", "—"
        else:
            q = res[k]
            plc_txt = (f"{_share(r['plc'][k])}（打亂後中位數 {_share(r['plc_low'][k])}，真的 {_share(r['real_small'][k]['low'])}）")
            verdict_txt = ("通過" if q["pass"] else "沒有通過") + (f"（三段同方向 {'過' if q['sign'] else '不過'}、排名 {'過' if q['score'] else '不過'}、"
                                                          f"虧損 {'過' if q['loss'] else '不過'}、錯位 {'過' if q['plc'] else '不過'}）")
        rows6.append([NAME[k], f"{_share(ck[0])}（{x['d_n']} 個）", _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      _share(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}",
                      plc_txt, verdict_txt])
    rows6.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(ex.chold[1])))}｜—", "—", "—", "—"])
    n_c = s["ma50"]["c_n"]
    t6 = {"caption": (f"main_signal_check 的公平考試（四項都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 日線 {len(ex.dstarts)} 個每月 1 日起算；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（{COST * 100:.2f}%）；同樣格子照一般人方式做"
                      f"（收盤後 10 小時才做、{P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次）；ETH、SOL、BNB、XRP 起算月合計（各用自己的均線和自己「有沒有跟上」，"
                      "黃金訊號相同）。排名分數＝四項裡最低的；前提是最深虧損比一直抱著淺的比例至少 95%。"
                      f"隨機錯位：把黃金每天的漲跌整串循環平移 {GAP} 天到「長度 − {GAP}」天，{DRAWS} 次，整套考試重做，排名分數一樣好或更好的比例"
                      f"（這一步一般人做法每格只模擬 {PL_SIMS} 次，真的規則也這樣算一次再比）。"
                      "事先寫好的過關標準：黃金 → 比特幣 Granger 係數三段都 > 0、排名分數不低於 50 日均線、最深虧損中位數不更深、隨機錯位 ≤ 10%，四項都要過。"),
          "columns": ["規則", "比特幣起算月", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他幣起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人", "隨機錯位一樣好", "結果"],
          "rows": rows6}
    tables = [t1, t2, t3, t4, t5, t6]

    # ----- today -----
    last = close.index[-1]
    fb = r["fl"]["btc"]
    today = fb.loc[last] if last in fb.index else None
    pos = r["pos"]

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    now = [{"label": f"最後一根收完的比特幣日線（{_d(last)}）", "value": f"{float(close.iloc[-1]):,.0f} 美元"},
           {"label": "最新的 GLD 收盤", "value": f"{_d(gld.index[-1])}：{float(gld.iloc[-1]):,.2f} 美元"}]
    if today is not None and _ok(today["L"]):
        now.append({"label": f"黃金 4–7 個月前的漲跌（{_d(last - FAR * DAY)} 到 {_d(last - NEAR * DAY)}）",
                    "value": f"{_pct((math.exp(today['L']) - 1) * 100)}；G2 的條件「> 0」{'成立' if today['lead'] > 0.5 else '不成立'}"})
        now.append({"label": f"比特幣從 {_d(last - FAR * DAY)} 到前一天的漲跌",
                    "value": (f"{_pct((math.exp(today['A']) - 1) * 100)}；補漲條件（黃金那段漲 ≥ 5% 而且比特幣落後）"
                              f"{'成立' if today['catch'] > 0.5 else '不成立'}") if _ok(today["A"]) else "—"})
    now.append({"label": "比特幣：50 日均線｜G1｜G2", "value": f"{word(pos['ma50'][-1])}｜{word(pos['g1'][-1])}｜{word(pos['g2'][-1])}"})
    c52 = roll["c52"].dropna()
    if len(c52):
        now.append({"label": "黃金和比特幣同一週報酬的 52 週相關", "value": f"{_num(c52.iloc[-1])}（到 {_d(c52.index[-1])}）"})
    now.append({"label": "這幾格的意思", "value": "只描述黃金訊號和兩條規則現在停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    g = gr["g2b"]
    bstab = np.array([x["best"] for x in roll["best"]["g2b"]])
    inside = float(((bstab >= lo) & (bstab <= hi)).mean()) if len(bstab) else float("nan")
    betas = "、".join(f"{g[p]['beta']:+.3f}（p {g[p]['p_beta']:.2f}）" for p in three)
    rev = gr["b2g"]
    s1 = (f"黃金領先比特幣 4–7 個月的說法，照事先寫好的檢驗{'站得住' if (res['g1']['pass'] or res['g2']['pass']) else '站不住'}："
          f"黃金 {lo}–{hi} 週前的報酬對比特幣下一週報酬的係數在 2013–17、2018–21、2022 起是 {betas}，"
          f"{'三段都是正的' if sign_ok else '不是三段都是正的'}；1–40 週一起測的 p 值 "
          + "、".join(f"{g[p]['p_F']:.2f}" for p in three) + "；反方向（比特幣領先黃金）p 值 "
          + "、".join(f"{rev[p]['p_F']:.2f}" for p in three) + "。")
    s2 = (f"滾動兩年看，相關最高的落後從 {bstab.min():.0f} 到 {bstab.max():.0f} 週都有，落在 4–7 個月的窗口只有 {inside * 100:.0f}%"
          "（運氣約 34%）。") if len(bstab) else ""
    s3 = "；".join(f"{'G1' if k == 'g1' else 'G2'} 排名分數 {_share(s[k]['low'])}（均線 {_share(base['low'])}）、最深虧損中位數 "
                  f"{_pct(s[k]['c_deep_med'])}（{_pct(base['c_deep_med'])}）、隨機錯位 {_share(r['plc'][k])}，"
                  f"{'通過' if res[k]['pass'] else '沒有通過'}" for k in ("g1", "g2")) + "。"
    verdict = s1 + s2 + "拿來進出：" + s3

    # ----- notes -----
    tot, bad = r["look"]
    gfirst = gld.index[0] + (FAR + 1) * DAY
    notes = [
        ("事先寫死的設計（照原文，跑之前寫好，沒有其他版本）：lag grid 0–40 weeks on weekly log returns (Friday closes); "
         "Granger-style test both directions: own lags 1–4, sum 5–16, sum 17–30, plus the other series' lags 17–30 summed "
         "(claim window, sign) or lags 1–40 in 8 sums of 5 weeks (F), null by circular moving-block bootstrap of the other "
         "series (26-week blocks, 1,000 draws); splits 2013–17 / 2018–21 / 2022–now plus after the claim (2025-12 onward, few "
         "data); rolling 104-week windows at quarter ends for the best lag. Rules: L = log(gold[D−122]/gold[D−213]); "
         "catch-up = L ≥ log 1.05 AND log(btc[D−1]/btc[D−213]) < L; G1 = ma50 OR catch-up; G2 = ma50 AND L > 0. Fair exam "
         "exactly as main_signal_check (108 daily starts, 72×24 clock cells textbook and ordinary person 0.25% with 10% "
         "missed entries, ETH/SOL/BNB/XRP), ranking score = lowest of four beats-holding shares, placebo = gold's daily "
         "returns rolled ≥180 days, 200 draws. Pass test: claim-window beta > 0 in all three periods AND ranking score ≥ "
         "ma50's AND median deepest loss no worse AND placebo ≤ 10%."),
        (f"資料：黃金用 GLD（SPDR 黃金基金，追蹤倫敦金價扣掉每年 0.4% 費用）的 Yahoo 調整後收盤，從 {_d(gld.index[0])} 到 {_d(gld.index[-1])}；"
         "LBMA 官方金價需要登入或授權，沒有使用。GLD 收盤是美國下午 4 點，和倫敦定盤價時間不同，但每週報酬幾乎一樣。"
         f"比特幣用儀表板的長期日線（2017-08 以前是 blockchain.info，已在 2026-11-19 改成「蠟燭涵蓋的那一天」，這裡沒有再自己位移）。"
         f"黃金訊號要 7 個月的歷史，最早從 {_d(gfirst)} 起有值；Granger 檢驗也要 40 週的落後，2013–17 那段的每週資料從 2013 年初就能用（GLD 從 2010 年起）。"),
        ("說法本身：Bitwise 的 André Dragosch 用 Granger 因果分析說黃金領先比特幣約 4–7 個月（2025 年底），意思是黃金在 2025 年大漲之後，比特幣應該會補漲。"
         "這裡照字面把「4–7 個月」換成 17–30 週、122–213 天，並沒有拿到他原本的資料、頻率和模型，所以數字不會和他的完全一樣；"
         "這份研究問的是：同樣的想法放在公開資料上、分時段、事先寫好規則，還成立嗎？"),
        ("時間對齊：美國收盤（約 UTC 20–21 點）的黃金，從標著同一天、UTC 隔天 0 點收盤的比特幣日線起才能用（cross_market 的做法）；"
         "換收盤時間時，UTC 23 點以後的收盤可以用當天黃金，更早的只能用前一天。其實訊號只用到 122 天以前的黃金，所以時間對齊不會改變結果。"
         f"偷看未來的檢查：在 {'、'.join(CUTS)} 三個切點之後把比特幣和黃金價格換成亂數再算，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")
         + f"黃金再晚 7 天才用：2013 起報酬÷波動 G1 {_num(r['late']['g1']['ratio']) if r['late']['g1'] else '—'}、"
           f"G2 {_num(r['late']['g2']['ratio']) if r['late']['g2'] else '—'}（原本 {_num(per[pnames[-1]]['g1']['ratio']) if per[pnames[-1]]['g1'] else '—'}、"
           f"{_num(per[pnames[-1]]['g2']['ratio']) if per[pnames[-1]]['g2'] else '—'}）。"),
        ("掃 0–40 週再挑最高的，一定會找到看起來不錯的落後；所以這裡事先只把「17–30 週」當成說法的主張來檢驗，其他落後只是描述。"
         "週報酬雜訊很大，相關係數 0.1 左右在一段 4–5 年（約 200–250 週）的資料裡大多是雜訊（±0.13 左右）。"),
        ("其他幣：ETH、SOL、BNB、XRP 用幣安日線、自己的 50 日均線，「有沒有跟上黃金」用它自己的漲跌，黃金訊號一樣；"
         "起算月從規則和黃金訊號都有值的那個月起，到三年前。main_signal_check 的第四項還有 MSTR，這裡照指定只用四個幣，"
         "所以 50 日均線的第四項和 main_signal_check 的數字略有不同，但三條規則都用同一組。"),
        ("相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。成本：照課本 0.10%、一般人 0.25%，"
         "現金沒有利息、一直抱著不扣成本、沒有算稅。隨機的部分都用固定種子。"),
        (f"這次計算用了 {time.time() - t0:.0f} 秒（考試 {r['secs'][0]:.0f} 秒、隨機錯位 {r['secs'][1]:.0f} 秒）。"),
        "過去的結果不保證未來會一樣。這裡只是把一個公開說法照事先寫好的規則測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "gold_lead",
        "title": "黃金真的領先比特幣 4–7 個月嗎",
        "question": ("Bitwise 的分析師說黃金領先比特幣約 4–7 個月，所以黃金 2025 年大漲後比特幣會「補漲」；也有人說 2026 年兩者變成反方向。"
                     "用公開的每週資料兩個方向都測：誰領先誰、落後週數穩不穩、分時段還成立嗎？照「補漲」寫成的進出規則能不能贏過 50 日均線和隨機錯位？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
