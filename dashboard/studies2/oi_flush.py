"""Historical study "oi_flush": after a leverage flush (futures open interest falls sharply together with the price,
usually with liquidations) does bitcoin rebound, or does it stay weak? And the opposite: open interest piling up while
the price goes nowhere - is that fragile?

Two published claims that disagree:
  * CryptoQuant (market commentary, several posts 2023-2025): after similar open-interest flushes bitcoin rebounded
    20-30% within 1-2 weeks ("leverage reset -> rebound").
  * K33 Research 2025: after the top 5% of liquidation events bitcoin averaged only +0.78% over the next 30 days,
    against 5.9% on normal days, and the median was negative ("flushes are not a buy signal").
  Both are tested on the same data below.

Pre-registered design (written before any result was seen; not changed afterwards):
Data: Binance USDT-M futures daily "metrics" files (data.binance.vision), field sum_open_interest (open interest
counted in coins, so a price fall does not shrink it by itself), BTCUSDT from 2020-09, ETH/SOL/BNB/XRP from 2021-12,
read through ls_ratio's loader and store (only days not yet stored are downloaded). Only the row stamped 00:00 UTC
of day D is used; a missing day carries the last value for at most 3 days. Prices: the dashboard's daily closes
(00:00 UTC, te._daily_close) for bitcoin, Binance spot daily closes for the other coins.
Alignment: the OI stamped D 00:00 is matched with the close that ends at D 00:00 (the candle labelled D-1). Every
flag built from day D is used only from the next close (the candle labelled D, ending D+1 00:00) - about one day
late, on purpose - and positions are held from that close.
1. Flush (day D): 3-day OI change log(OI_D / OI_D-3) below the 5% quantile of its own 365 previous values (at least
   330 present) AND the 3-day price return over the same 3 days < 0. Point-in-time; unknown -> no flag.
   Episode = a flush day with no flush day in the 14 days before it (consecutive days are one event).
2. Build-up (day D): 30-day OI change log(OI_D / OI_D-30) above the 90% quantile of its own 365 previous values AND
   |30-day price return| below the 1/3 quantile of its own 365 previous absolute 30-day returns ("price flat").
   Episodes as above.
3. Event study: forward 7/14/30-day returns from the decision close of each episode start, against all days on which
   the flag can be computed (excess = mean of episodes - mean of all days). 90% interval: circular blocks of 30
   days (paired flag/return series), 1,000 draws. Halves: the days with a flag value and a 30-day forward return,
   split at the middle day. Also the median and the share up after 30 days (K33's way of saying it).
   Predicted sign (CryptoQuant): flush positive at 14 days (primary horizon for the flush); K33 predicts below normal
   (negative excess) at 30 days. Build-up: negative at 30 days (primary horizon for the build-up).
4. Rules on top of the 50-day average (ma50: hold while the close is above its 50-day average), 0.10% per trade:
   F1 = ma50, but also hold for the 14 decision days starting with a flush flag (enter early after a flush, "catch
   the start of a rally"); after the 14 days the position follows ma50 again. F2 = ma50, but cash while the build-up
   flag is on (fragility).
5. Fair exam reusing main_signal_check exactly as ls_ratio does: monthly starts from the first month after the
   bitcoin flags exist to two years before today, UTC daily closes (textbook, 0.10%) and the same start months x 24
   closing hours from Binance 1-hour candles (textbook and ordinary person: decide at the clock, trade 10 hours
   later, 0.25%, 10% of entries missed, 200 simulations per clock); a decision at hour h >= 1 of date D uses the
   flag stamped D, at h = 0 the one stamped D-1. Fourth check = ETH, SOL, BNB, XRP, each with its own OI flags,
   monthly starts from the first month after its own flags exist. Ranking score = lowest of the four "beats
   holding" shares. Median deepest loss over the start x clock cells. Placebo: every coin's OI series rolled
   circularly by the same random 180 ... (n - 180) days, 200 draws, flags and rules rebuilt, ranking score without
   the ordinary-person check (as ls_ratio), share of draws at least as good as the real rule built the same way.
Pass test (stated now), separately for flush+F1 and build-up+F2: predicted sign in both halves AND full-sample 90%
interval excluding zero on the predicted side AND the rule's ranking score >= ma50's AND its median deepest loss no
deeper than ma50's AND placebo share <= 10% AND the predicted sign on >= 3 of the 4 other coins. The sample is about
five years of OI (four with a year of warm-up) - stated on the page.
Today: OI changes and their positions, flush / build-up flags and the three rules' states - states only.

Related, not repeated here: forced_flows (Bybit OI falling 10%+ in one day as a liquidation-cascade event, buy-and-
hold-7/30 entries), funding_positioning (funding), ls_ratio (long/short ratios), crash_timing / rally_timing (price
only exits and early entries).
Public key-less data only, through the dashboard's helpers (fixed user agent). Descriptive and in-sample; not advice.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
COST_HI = 0.003
YEAR = 365
H = (7, 14, 30)
H_FLUSH, H_BUILD = 14, 30
WIN, WIN_MIN = 365, 330
FLUSH_Q, BUILD_Q, FLAT_Q = 0.05, 0.90, 1 / 3
FLUSH_DAYS, BUILD_DAYS = 3, 30
HOLD_DAYS = 14
EPISODE_GAP = 14
FFILL = 3
BOOT, DRAWS, BLOCK = 1000, 200, 30
KEYS = ("ma50", "f1", "f2")
SEED_KI = {"ma50": 1, "f1": 80, "f2": 81}  # ma50 uses main_signal_check's draws (as ls_ratio)
NAME = {"ma50": "50 日均線（現行主訊號）", "f1": "F1：50 日均線，但「清槓桿」後提早進場、抱 14 天",
        "f2": "F2：50 日均線，但「槓桿堆高、價格不動」時改拿現金"}
SHORT = {"ma50": "50 日均線", "f1": "F1", "f2": "F2"}
EV_NAME = {"flush": "清槓桿（3 天未平倉量跌到一年最低 5%，價格也跌）", "build": "槓桿堆高（30 天未平倉量漲到一年最高 10%，價格幾乎不動）"}
OTHER = lr.OTHER
CUTS = ("2022-06-30", "2023-12-31", "2025-06-30")

SOURCES = [
    {"name": "CryptoQuant 市場評論（Insights）：未平倉量大幅清洗後，比特幣在 1–2 週內常反彈 20–30%",
     "url": "https://cryptoquant.com/insights/quicktake/692d6284160b570d64bc2324-BTC-Flash-Crash-9K-Dump-to-86K-Leverage-Cleanse"},
    {"name": "K33 Research 2025：前 5% 大的爆倉事件之後 30 天，比特幣平均只 +0.78%（平常 5.9%），中位數是負的",
     "url": "https://www.theblock.co/post/372115/whats-next-after-cryptos-fourth-largest-long-liquidation-event-of-2025"},
    {"name": "幣安公開歷史資料 data.binance.vision：U 本位合約每日 metrics 檔（sum_open_interest）",
     "url": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/"},
    {"name": "幣安合約交易數據（未平倉量）", "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/trading-data"},
    {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
]

_ok, _num, _d = te._ok, te._num, te._d


def _pct(v, signed: bool = True) -> str:
    """v is a fraction."""
    return te._pct(v * 100, signed) if _ok(v) else "—"


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


# ---------- series and flags ----------

def _oi(rows: dict, last: pd.Timestamp | None = None) -> pd.Series:
    """Open interest in coins by stamp day (D 00:00 UTC), daily index, unknown = NaN (no carrying yet)."""
    keys = sorted(k for k in rows if rows[k][4] is not None)
    idx = pd.to_datetime(keys)
    s = pd.Series([float(rows[k][4]) for k in keys], index=idx, dtype=float)
    end = max(idx[-1], last) if last is not None else idx[-1]
    return s.reindex(pd.date_range(idx[0], end, freq="D"))


def _px_stamp(close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The close that ends at stamp D 00:00 = the candle labelled D-1."""
    return pd.Series(close.reindex(index - pd.Timedelta(days=1)).values, index=index, dtype=float)


def _q(s: pd.Series, q: float) -> pd.Series:
    return s.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(q)


def _flags(oi_raw: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Stamp-indexed flags: flush, win (flush in the last 14 stamp days), build, and whether each can be known."""
    oi = oi_raw.ffill(limit=FFILL)
    px = _px_stamp(close, oi.index)
    d3 = np.log(oi / oi.shift(FLUSH_DAYS))
    p3 = px / px.shift(FLUSH_DAYS) - 1
    q05 = _q(d3, FLUSH_Q)
    k_f = q05.notna() & d3.notna() & p3.notna()
    flush = (k_f & (d3 < q05) & (p3 < 0)).astype(float)
    d30 = np.log(oi / oi.shift(BUILD_DAYS))
    a30 = (px / px.shift(BUILD_DAYS) - 1).abs()
    q90, q33 = _q(d30, BUILD_Q), _q(a30, FLAT_Q)
    k_b = q90.notna() & q33.notna() & d30.notna() & a30.notna()
    build = (k_b & (d30 > q90) & (a30 < q33)).astype(float)
    win = flush.rolling(HOLD_DAYS, min_periods=1).max()
    return pd.DataFrame({"flush": flush, "win": win, "build": build, "kf": k_f.astype(float), "kb": k_b.astype(float),
                         "d3": d3, "p3": p3, "d30": d30, "a30": a30, "q05": q05, "q90": q90, "q33": q33})


def _rules_on(c: np.ndarray, win: np.ndarray, build: np.ndarray) -> dict:
    s = pd.Series(c)
    m50 = s.rolling(50, min_periods=50).mean()
    ma = (s > m50).astype(float).where(m50.notna()).values
    return {"ma50": ma, "f1": np.where(win > 0.5, 1.0, ma), "f2": ma * (1 - build)}


def _targets(close: pd.Series, fl: pd.DataFrame, extra: int = 0) -> pd.DataFrame:
    """Targets after each daily close (label D = close at D+1 00:00, which may use the flag stamped D)."""
    f = fl[["win", "build"]].reindex(close.index).shift(extra).fillna(0.0)
    return pd.DataFrame(_rules_on(close.values.astype(float), f["win"].values, f["build"].values), index=close.index)


# ---------- event study ----------

def _starts(flag: pd.Series) -> pd.Series:
    """Episode starts: flagged days with no flagged day in the EPISODE_GAP days before."""
    prev = flag.shift(1).rolling(EPISODE_GAP, min_periods=1).max().fillna(0.0)
    return ((flag > 0.5) & (prev < 0.5)).astype(float)


def _boot(flag: np.ndarray, fwd: np.ndarray, rng) -> tuple:
    idx = km._blocks(len(fwd), BLOCK, BOOT, rng)
    f, r = flag[idx], fwd[idx]
    cnt = f.sum(axis=1)
    good = cnt > 0
    ex = (f * r).sum(axis=1)[good] / cnt[good] - r.mean(axis=1)[good]
    return (float(np.percentile(ex, 5)), float(np.percentile(ex, 95))) if len(ex) > 50 else (None, None)


def _excess(f: np.ndarray, y: np.ndarray):
    return float(y[f].mean() - y.mean()) if f.sum() > 0 and len(y) else None


def _events(fl: pd.DataFrame, close: pd.Series, rng, ci: bool = True) -> dict:
    """Per kind (flush, build): episode count, per-horizon mean / median / up / excess (+ interval), halves."""
    fwd = {h: close.shift(-h) / close - 1 for h in H}
    out = {}
    for kind, known in (("flush", "kf"), ("build", "kb")):
        k = fl[known].reindex(close.index).fillna(0.0) > 0.5
        st = _starts(fl[kind].reindex(close.index).fillna(0.0))
        e = {"flag_days": int((fl[kind].reindex(close.index).fillna(0.0) > 0.5).sum())}
        base = k & fwd[30].notna()
        days = close.index[base.values]
        cut = days[len(days) // 2] if len(days) else None
        e["first"], e["last"], e["split"] = (days[0], days[-1], cut) if len(days) else (None, None, None)
        for h in H:
            ok = (k & fwd[h].notna()).values
            y = fwd[h].values[ok]
            f = st.values[ok] > 0.5
            ev = y[f]
            row = {"n": int(f.sum()), "all_n": int(ok.sum()),
                   "mean": float(ev.mean()) if len(ev) else None, "median": float(np.median(ev)) if len(ev) else None,
                   "up": float((ev > 0).mean()) if len(ev) else None,
                   "all_mean": float(y.mean()) if len(y) else None, "all_median": float(np.median(y)) if len(y) else None,
                   "all_up": float((y > 0).mean()) if len(y) else None, "ex": _excess(f, y)}
            row["ci"] = _boot(f.astype(float), y, rng) if ci and f.sum() > 0 else (None, None)
            if cut is not None:
                a = np.asarray(close.index[ok] < cut)
                row["halves"] = [_excess(f[a], y[a]), _excess(f[~a], y[~a])]
                row["half_n"] = [int(f[a].sum()), int(f[~a].sum())]
            else:
                row["halves"], row["half_n"] = [None, None], [0, 0]
            e[h] = row
        e["dates"] = list(close.index[(st > 0.5).values])
        out[kind] = e
    return out


def _passes(e: dict, h: int, sign: int) -> dict:
    m = e[h]
    est, ci = m["ex"], m["ci"]
    halves = sum(1 for v in m["halves"] if _ok(v) and v * sign > 0)
    excl = ci[0] is not None and ((ci[1] < 0) if sign < 0 else (ci[0] > 0))
    return {"pass": bool(_ok(est) and est * sign > 0 and halves == 2 and excl), "halves": halves, "excl": excl}


# ---------- fair grid (main_signal_check's pieces, as ls_ratio) ----------

def _first_flag(fl: pd.DataFrame) -> pd.Timestamp:
    known = fl["kf"] > 0.5
    return fl.index[int(np.argmax(known.values))]


class _Clock(lr._Clock):
    def _known(self, fl: pd.DataFrame, h: int) -> pd.DataFrame:
        when = self.m.dates if h >= 1 else self.m.dates - pd.Timedelta(days=1)
        return fl[["win", "build"]].reindex(when).fillna(0.0)

    def run(self, fl: pd.DataFrame, person: bool = True, keys=KEYS) -> dict:
        m, eng = self.m, self.eng
        days, npre = m.days, len(self.pre)
        ns = len(self.starts)
        text = {k: (np.zeros((ns, 24)), np.zeros((ns, 24))) for k in keys}
        pers = {k: (np.zeros((ns, 24, SIMS)), np.zeros((ns, 24, SIMS))) for k in keys} if person else None
        for h in range(24):
            closes = np.concatenate([self.pre, self.price[h::24]]).astype(float)
            kn = self._known(fl, h)
            fw, fb = np.zeros(len(closes)), np.zeros(len(closes))
            fw[npre:npre + days], fb[npre:npre + days] = kn["win"].values, kn["build"].values
            rules = _rules_on(closes, fw, fb)
            for k in keys:
                w = np.nan_to_num(rules[k][npre:][:days], nan=0.0).astype(float)
                r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), self.starts, YEAR)
                text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
                if person:
                    b, wl = rt._later(h, w, days)
                    rng = np.random.default_rng(SEED + 1000 * SEED_KI[k] + h)
                    pos = rt._missed(wl, P_MISS, rng, SIMS)
                    r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]),
                                        self.starts, YEAR)
                    pers[k][0][:, h, :], pers[k][1][:, h, :] = r, dd
        return {"text": text, "person": pers}


def _summary(d: dict, hold: tuple, c: dict, others: dict) -> dict:
    hr, hd = hold
    s = {}
    for k in KEYS:
        dr, dd = d[k]
        tr, td = c["text"][k]
        edge = tr - hr[:, None]
        pr, pdd = c["person"][k]
        pedge = pr - hr[:, None, None]
        o_n = sum(len(o[k][0]) for o in others.values())
        o_beat = sum(int((o[k][0] > o["hold"][0]).sum()) for o in others.values())
        x = {"d_n": len(dr), "d_beat": int((dr > d["hold"][0]).sum()), "d_shallow": int((dd > d["hold"][1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pdd)), "o_n": o_n, "o_beat": o_beat}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        s[k] = x
    order = sorted(KEYS, key=lambda k: (not s[k]["eligible"], -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "order": order, "hold_deep_med": float(np.median(hd))}


# ---------- the whole computation ----------

def _load() -> dict:
    problems, oi = [], {}
    for sym in lr.FIRST_DAY:
        try:
            rows, p = lr._fetch(sym, budget_s=300)
        except Exception as exc:  # noqa: BLE001 - a coin without any data is left out
            if sym == "BTCUSDT":
                raise
            problems.append(f"{sym} 的未平倉量存檔這次取不到（{type(exc).__name__}），沒有算進去。")
            continue
        if p:
            problems.append(p)
        oi[sym] = rows
    close = te._daily_close()
    coins = {}
    for sym, key in OTHER:
        try:
            coins[key] = ta._crypto_close(sym)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key} 現貨日線這次取不到（{type(exc).__name__}），沒有算進去。")
    return {"rows": oi, "close": close, "coins": coins, "problems": problems}


def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    oi = _oi(data["rows"]["BTCUSDT"], close.index[-1])
    fl = _flags(oi, close)
    rng = np.random.default_rng(SEED)
    ev = _events(fl, close, rng)
    late = _events(fl.shift(1), close, rng, ci=False)
    oc = {}
    for sym, key in OTHER:
        if key not in data["coins"] or not data["rows"].get(sym):
            continue
        c = data["coins"][key]
        o_oi = _oi(data["rows"][sym], c.index[-1])
        f_ = _flags(o_oi, c)
        oc[key] = {"close": c, "oi": o_oi, "fl": f_, "ev": _events(f_, c, rng, ci=False), "ff": _first_flag(f_)}

    first_flag = _first_flag(fl)
    g0, g1 = lr._grid_span(first_flag)
    T = _targets(close, fl)
    daily = lr._daily_grid(close, T, g0, g1, keys=KEYS)
    clk = _Clock(g0, g1)
    clock = clk.run(fl)
    others = {}
    for key, o in oc.items():
        o["T"] = _targets(o["close"], o["fl"])
        a, _ = lr._grid_span(o["ff"])
        g = lr._daily_grid(o["close"], o["T"], a, g1, keys=KEYS)
        if len(g["starts"]):
            others[key] = g
            o["span"] = a
    sm = _summary(daily, clk.hold, clock, others)
    t_grid = time.time() - t0

    real = {k: lr._text_score(daily, clk.hold, clock["text"], others, k) for k in ("f1", "f2")}
    better = {"f1": 0, "f2": 0}
    prng = np.random.default_rng(SEED + 9)
    shifts = prng.integers(180, len(oi) - 180, DRAWS)
    for s in shifts:
        f2 = _flags(pd.Series(np.roll(oi.values, int(s)), index=oi.index), close)
        d2 = lr._daily_grid(close, _targets(close, f2), g0, g1, keys=("f1", "f2"))
        c2 = clk.run(f2, person=False, keys=("f1", "f2"))
        o2 = {}
        for key in others:
            o = oc[key]
            fo = _flags(pd.Series(np.roll(o["oi"].values, int(s) % len(o["oi"])), index=o["oi"].index), o["close"])
            o2[key] = lr._daily_grid(o["close"], _targets(o["close"], fo), o["span"], g1, keys=("f1", "f2"))
            o2[key]["hold"] = others[key]["hold"]
        for k in better:
            better[k] += int(lr._text_score(d2, clk.hold, c2["text"], o2, k) >= real[k])
    placebo = {k: better[k] / DRAWS for k in better}

    idx = close.index
    change = close.pct_change().fillna(0.0)
    ratio = {(k, c): lr._ratio(T[k], change, first_flag, c) for k in KEYS for c in (COST, COST_HI)}
    ratio["hold"] = lr._ratio(pd.Series(1.0, index=idx), change, first_flag, 0.0)
    T_late = _targets(close, fl, extra=1)
    ratio_late = {k: lr._ratio(T_late[k], change, first_flag, COST, delay=2) for k in KEYS}
    ratio_o = {}
    for key, o in oc.items():
        ch = o["close"].pct_change().fillna(0.0)
        ratio_o[key] = {k: lr._ratio(o["T"][k], ch, o["ff"]) for k in KEYS}
        ratio_o[key]["hold"] = lr._ratio(pd.Series(1.0, index=o["close"].index), ch, o["ff"], 0.0)
    c_arr = close.values
    first_i = int(idx.searchsorted(first_flag))
    bears = [b for b in ct._bears(close) if idx[b["top"]] >= first_flag - pd.Timedelta(days=60)]
    crash = []
    for b in bears:
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for k in KEYS:
            x = ct._crash(c_arr, [(1.0, T[k].values)], b)
            row[k] = x["avoid"]
            row[k + "_catch"] = x.get("catch")
        crash.append(row)
    whip = {k: ct._whipsaws(c_arr, [(1.0, T[k].values)], first_i) for k in KEYS}
    after = idx >= first_flag
    extra_on = int(((T["ma50"] < 0.5) & (T["f1"] >= 0.5))[after].sum())
    off = int(((T["ma50"] >= 0.5) & (T["f2"] < 0.5))[after].sum())
    # F1's early entries: what the 14 days outside ma50 earned
    f1_extra = []
    for d0 in ev["flush"]["dates"]:
        i = int(idx.searchsorted(d0))
        seg = slice(i, min(i + HOLD_DAYS, len(idx) - 1))
        w = (T["f1"].values[seg] >= 0.5) & ~(np.nan_to_num(T["ma50"].values[seg]) >= 0.5)
        r = change.values[i + 1:min(i + HOLD_DAYS, len(idx) - 1) + 1]
        f1_extra.append(float(np.prod(1 + r[:len(w)] * w[:len(r)]) - 1))
    return {"close": close, "oi": oi, "fl": fl, "ev": ev, "late": late, "oc": oc, "T": T, "first_flag": first_flag,
            "grid": (g0, g1), "daily": daily, "clk": clk, "clock": clock, "others": others, "sm": sm,
            "placebo": placebo, "real_score": real, "ratio": ratio, "ratio_late": ratio_late, "ratio_o": ratio_o,
            "crash": crash, "whip": whip, "extra_on": extra_on, "off": off, "f1_extra": f1_extra,
            "t_grid": t_grid, "t_all": time.time() - t0}


# ---------- checks ----------

def _lookahead(r: dict) -> str:
    rng = np.random.default_rng(11)
    close, oi = r["close"], r["oi"]
    total = same = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        nc = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        no = oi.copy()
        late = no.index > c
        no.loc[late] = no.loc[late].values * np.exp(rng.normal(0, 0.5, int(late.sum())))
        f2 = _flags(no, nc)
        t2 = _targets(nc, f2)
        m = close.index <= c
        for k in KEYS:
            total += 1
            same += int(np.array_equal(np.nan_to_num(r["T"][k].values[m], nan=-1),
                                       np.nan_to_num(t2[k].values[m], nan=-1)))
        mf = oi.index <= c
        for k in ("flush", "build", "win"):
            total += 1
            same += int(np.array_equal(r["fl"][k].values[mf], f2[k].values[mf]))
    return (f"偷看未來的檢查：在 {'、'.join(CUTS)} 三個切點，把切點之後的比特幣收盤價和未平倉量全部換成亂數再算一次，切點以前的"
            f"「清槓桿」「槓桿堆高」旗標和三條規則的進出 {total} 組裡 {same} 組完全沒變" + ("。" if same == total else "（有變動，結果不能用）。"))


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    t_dl = time.time() - t0
    r = _compute(data)
    problems = list(data["problems"]) + ([r["clk"].problem] if r["clk"].problem else [])
    ev, oc, sm, plc = r["ev"], r["oc"], r["sm"], r["placebo"]
    s = sm["s"]
    rank = {k: i for i, k in enumerate(sm["order"], 1)}
    close, oi, fl, T = r["close"], r["oi"], r["fl"], r["T"]
    g0, g1 = r["grid"]
    yrs_oi = (oi.dropna().index[-1] - oi.dropna().index[0]).days / 365.25
    fe, be = ev["flush"], ev["build"]
    yrs = (fe["last"] - fe["first"]).days / 365.25 if fe["first"] is not None else 0.0

    # ----- table 1: event study -----
    rows1 = []
    for kind in ("flush", "build"):
        e = ev[kind]
        for h in H:
            x = e[h]
            ci = x["ci"]
            rows1.append([EV_NAME[kind] if h == 7 else "", f"{h} 天", f"{x['n']}",
                          f"{_pct(x['mean'])}｜{_pct(x['median'])}｜{_share(x['up'])}",
                          f"{_pct(x['all_mean'])}｜{_pct(x['all_median'])}｜{_share(x['all_up'])}",
                          f"{_pct(x['ex'])}（{_pct(ci[0])} 到 {_pct(ci[1])}）",
                          f"{_pct(x['halves'][0])}（{x['half_n'][0]}）｜{_pct(x['halves'][1])}（{x['half_n'][1]}）"])
    t1 = {"caption": ("幣安 BTCUSDT 永續合約的未平倉量（以顆數計，每天 UTC 0 點那一筆），每個條件只和自己之前 365 天比（不用到未來），"
                      "訊號出來的隔一根日線收盤才算進場點。連續幾天都符合只算一次（前 14 天沒出現過才算新的一次）。"
                      "「多出來」＝ 事件之後平均 − 所有可比日子的平均；括號是 30 天一段整段重抽 1,000 次的 90% 區間。"
                      f"可比的日子 {_d(fe['first'])} 到 {_d(fe['last'])}，前半／後半從 {_d(fe['split'])} 切開（括號是事件次數）。"
                      "相鄰日子之後的 30 天幾乎重疊，所以「所有日子」的獨立樣本大約只有天數 ÷ 30。"),
          "columns": ["事件", "之後幾天", "次數", "事件後：平均｜中位數｜上漲比例", "所有日子：平均｜中位數｜上漲比例",
                      "多出來（90% 區間）", "前半｜後半 多出來"], "rows": rows1}

    # ----- table 2: list of flushes -----
    rows2 = []
    fwd = {h: close.shift(-h) / close - 1 for h in H}
    for d0 in fe["dates"]:
        st = d0  # stamp day of the flag = label of the decision close
        rows2.append([_d(d0), _pct(float(np.exp(fl["d3"].get(st)) - 1)), _pct(float(fl["p3"].get(st))),
                      f"{float(close.loc[d0]):,.0f}"] + [_pct(fwd[h].get(d0)) for h in H])
    t2 = {"caption": ("每一次比特幣「清槓桿」：日期是決定的那根日線（UTC 0 點收盤），3 天變化是到這根日線開始時為止的 3 天（未平倉量 0 點那筆、"
                      "價格是前一根收盤），之後的漲跌從這根收盤起算；空白代表還沒過完那麼多天。"),
          "columns": ["決定日", "3 天未平倉量", "3 天價格", "決定日收盤（美元）", "之後 7 天", "之後 14 天", "之後 30 天"],
          "rows": rows2 or [["沒有事件", "—", "—", "—", "—", "—", "—"]]}

    # ----- table 3: other coins -----
    rows3 = []
    same = {"flush": 0, "build": 0}
    for key, o in oc.items():
        row = [key]
        for kind, h, sign in (("flush", H_FLUSH, 1), ("build", H_BUILD, -1)):
            x = o["ev"][kind]
            v = x[h]["ex"]
            same[kind] += int(_ok(v) and v * sign > 0)
            row.append(f"{x[h]['n']} 次｜{_pct(x[h]['mean'])}｜{_pct(v)}")
        x30 = o["ev"]["flush"][30]
        row.append(f"{_pct(x30['mean'])}｜{_pct(x30['median'])}｜{_pct(x30['all_mean'])}")
        rows3.append(row)
    rows3.append(["和預期同方向的幣", f"{same['flush']}／{len(oc)}（預期正）", f"{same['build']}／{len(oc)}（預期負）", "—"])
    t3 = {"caption": ("其他四個幣用自己的未平倉量和自己的現貨收盤（2021-12 才有資料，扣掉一年累積後只剩約 4 年），同樣的定義原封不動。"
                      "格子是「事件次數｜事件後平均｜比所有日子多出來」。最後一欄是清槓桿之後 30 天（K33 看的長度）。"),
          "columns": ["幣", f"清槓桿之後 {H_FLUSH} 天", f"槓桿堆高之後 {H_BUILD} 天", "清槓桿之後 30 天：平均｜中位數｜所有日子平均"],
          "rows": rows3}

    # ----- table 4: fair grid -----
    dn, cn, on = s["ma50"]["d_n"], s["ma50"]["c_n"], s["ma50"]["o_n"]
    rows4 = []
    for k in KEYS:
        x = s[k]
        ck = x["checks"]
        rows4.append([NAME[k], _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]) if x["o_n"] else "沒有資料",
                      f"{_share(x['low'])}（第 {rank[k]}）" + ("" if x["eligible"] else "，虧損條件未達"),
                      f"{te._pct(x['c_deep_med'])}｜{cr._pp(x['c_edge'])}", _share(plc[k]) if k in plc else "—"])
    o_txt = "、".join(f"{a} {len(o['starts'])} 個" for a, o in r["others"].items())
    t4 = {"caption": (f"main_signal_check 的公平考試（和 ls_ratio 一樣的做法），起算日只能從未平倉量有一年資料之後開始：比特幣 UTC 0 點日線 "
                      f"{_d(g0)[:7]} 到 {_d(g1)[:7]} 每月 1 日起算（只有 {dn} 個，main_signal_check 有 108 個）；同樣起算月 × 24 個收盤時間"
                      f"（{cn:,} 格）；一般人做法（晚 10 小時、成本 {P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次）；其他四幣各用自己的未平倉量"
                      f"（{o_txt}，共 {on} 個）。每格是「報酬贏過一直抱著」的比例；排名看四項裡最低的那項。「隨機錯位一樣好」：把每個幣的未平倉量"
                      f"整串循環平移同樣的隨機天數（{DRAWS} 次）重做旗標和規則，排名分數（不含一般人那項）和真的一樣好或更好的比例。"),
          "columns": ["規則", f"起算月（{dn}）", f"起算×時間（{cn:,}）", "一般人照做", f"其他四幣（{on}）", "四項最低（排名）",
                      "最深虧損中位數｜每年多賺（中位數）", "隨機錯位一樣好"], "rows": rows4}

    # ----- table 5: crashes, rallies, costs -----
    rows5 = []
    for b in r["crash"]:
        rows5.append([f"{_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）") + "：躲掉的比例",
                      _pct(b["drop"])] + [_share(b[k]) for k in KEYS])
        if b["done"]:
            rows5.append(["　同一段之後的反彈：吃到的比例", "—"] + [_share(b[k + "_catch"]) for k in KEYS])
    w = r["whip"]
    rows5.append(["一年下車幾次｜其中 30 天內更高價買回", "—"] + [f"{w[k]['exits']:.1f}｜{w[k]['whips']:.1f}" for k in KEYS])
    rt_ = r["ratio"]
    rows5.append([f"報酬÷波動，成本 0.10%（一直抱著 {_num(rt_['hold'])}）", "—"] + [_num(rt_[(k, COST)]) for k in KEYS])
    rows5.append(["報酬÷波動，成本 0.30%", "—"] + [_num(rt_[(k, COST_HI)]) for k in KEYS])
    rows5.append(["旗標再晚一天、買賣再晚一天：報酬÷波動", "—"] + [_num(r["ratio_late"][k]) for k in KEYS])
    for key, v in r["ratio_o"].items():
        rows5.append([f"{key} 報酬÷波動 0.10%（一直抱著 {_num(v['hold'])}）", "—"] + [_num(v[k]) for k in KEYS])
    t5 = {"caption": (f"比特幣照課本做（收盤決定、隔天照做），從第一天有旗標的 {_d(r['first_flag'])} 起算。大跌段是 crash_timing 定義的"
                      "從高點收盤跌 50% 以上的段落（高點在旗標開始前 60 天內或之後），「躲掉的比例」「吃到的比例」用規則自己在這段的漲跌算。"),
          "columns": ["項目", "跌幅", "50 日均線", "F1", "F2"], "rows": rows5}
    tables = [t1, t2, t3, t4, t5]

    # ----- pass test -----
    pf = _passes(fe, H_FLUSH, 1)
    pb = _passes(be, H_BUILD, -1)
    k33 = _passes(fe, 30, -1)
    detail = {}
    for kind, rk, pe in (("flush", "f1", pf), ("build", "f2", pb)):
        x, m = s[rk], s["ma50"]
        score_ok = x["low"] >= m["low"]
        loss_ok = x["c_deep_med"] >= m["c_deep_med"]
        plc_ok = plc[rk] <= 0.10
        coins_ok = same[kind] >= 3
        detail[kind] = {"effect": pe["pass"], "score": score_ok, "loss": loss_ok, "plc": plc_ok, "coins": coins_ok,
                        "ok": pe["pass"] and score_ok and loss_ok and plc_ok and coins_ok}
    passed = detail["flush"]["ok"] or detail["build"]["ok"]

    # ----- today -----
    last = oi.dropna().index[-1]

    def position(col: str) -> str:
        v = fl[col].ffill(limit=FFILL)
        x = v.get(last)
        w_ = v.loc[:last].iloc[-WIN - 1:-1].dropna()
        return f"{(w_ < x).mean() * 100:.0f}%" if _ok(x) and len(w_) >= WIN_MIN else "—"

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    f_now = fl.loc[last]
    stale = last < close.index[-1]
    recent = [d for d in fe["dates"] if d >= close.index[-1] - pd.Timedelta(days=HOLD_DAYS)]
    now = [
        {"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"幣安 BTCUSDT 未平倉量（存檔 {_d(last)} UTC 0 點）", "value": f"{float(oi.loc[last]):,.0f} 顆"},
        {"label": "3 天未平倉量變化（在過去一年裡由低到高的位置）",
         "value": f"{_pct(float(np.exp(f_now['d3']) - 1)) if _ok(f_now['d3']) else '—'}（{position('d3')}）；同期價格 {_pct(f_now['p3'])}"},
        {"label": "30 天未平倉量變化（位置）｜30 天價格漲跌",
         "value": f"{_pct(float(np.exp(f_now['d30']) - 1)) if _ok(f_now['d30']) else '—'}（{position('d30')}）｜"
                  f"{_pct(float(_px_stamp(close, oi.index).loc[last] / _px_stamp(close, oi.index).shift(BUILD_DAYS).loc[last] - 1))}"},
        {"label": "「清槓桿」旗標｜最近 14 天內有沒有｜「槓桿堆高、價格不動」旗標",
         "value": f"{'是' if f_now['flush'] > 0.5 else '否'}｜{('有（' + '、'.join(_d(d) for d in recent) + '）') if recent else '沒有'}｜"
                  f"{'是' if f_now['build'] > 0.5 else '否'}"},
        {"label": "50 日均線｜F1｜F2（最後一根日線收盤後）",
         "value": f"{word(T['ma50'].iloc[-1])}｜{word(T['f1'].iloc[-1])}｜{word(T['f2'].iloc[-1])}"
                  + (f"（未平倉量存檔只到 {_d(last)}）" if stale else "")},
        {"label": "這幾格的意思", "value": "只描述未平倉量和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    x14, x30, b30 = fe[H_FLUSH], fe[30], be[H_BUILD]
    f14 = [close.shift(-H_FLUSH).get(d) / close.get(d) - 1 for d in fe["dates"]]
    f14 = [v for v in f14 if _ok(v)]
    big = sum(1 for v in f14 if v >= 0.20)
    big_common = bool(f14) and big / len(f14) >= 0.5
    rev = _passes(fe, H_FLUSH, -1)  # the opposite of CryptoQuant at its own horizon
    ci14, ci30 = x14["ci"], x30["ci"]
    if pf["pass"]:
        cq = (f"CryptoQuant 的「清完槓桿會反彈」有支持：之後 14 天比平常多 {_pct(x14['ex'])}"
              f"（90% 區間 {_pct(ci14[0])} 到 {_pct(ci14[1])}，前後半都偏強）")
    elif rev["pass"]:
        cq = (f"CryptoQuant 的「清完槓桿會反彈」不成立，反而相反：之後 14 天比平常少 {_pct(-x14['ex'], False)}"
              f"（90% 區間 {_pct(ci14[0])} 到 {_pct(ci14[1])} 不含零，前後半都偏弱）")
    else:
        cq = (f"CryptoQuant 的「清完槓桿會反彈」沒有被證實：之後 14 天比平常 {_pct(x14['ex'])}"
              f"（90% 區間 {_pct(ci14[0])} 到 {_pct(ci14[1])}）")
    if k33["pass"]:
        k3 = f"K33 的「之後 30 天偏弱」成立（比平常 {_pct(x30['ex'])}，區間 {_pct(ci30[0])} 到 {_pct(ci30[1])}）"
    elif _ok(x30["ex"]) and x30["ex"] < 0:
        k3 = (f"K33 的「之後 30 天偏弱」方向對、數字也很像（平均 {_pct(x30['mean'])}、中位數 {_pct(x30['median'])}，平常 "
              f"{_pct(x30['all_mean'])}），但差距的區間 {_pct(ci30[0])} 到 {_pct(ci30[1])} 含零、前後半不一致，不算證實")
    else:
        k3 = f"K33 的「之後 30 天偏弱」沒有看到（比平常 {_pct(x30['ex'])}）"
    who = cq + "；" + k3
    v1 = (f"{who}。幣安比特幣未平倉量 {_d(oi.dropna().index[0])[:7]} 起才有（約 {yrs_oi:.1f} 年），扣掉先累積一年只剩約 {yrs:.1f} 年可比，"
          f"只有 {x14['n']} 次清槓桿，樣本很短；清槓桿後 14 天平均 {_pct(x14['mean'])}、中位數 {_pct(x14['median'])}"
          f"（所有日子 {_pct(x14['all_mean'])}），"
          + (f"{len(f14)} 次裡 14 天內漲 20% 以上的只有 {big} 次，20–30% 的反彈不是常態；" if not big_common
             else f"{len(f14)} 次裡 14 天內漲 20% 以上的有 {big} 次；")
          + f"其他四幣只有 {same['flush']} 個 14 天偏強。")
    bw = ("和「危險」的說法相反，之後反而偏強" if _ok(b30["ex"]) and b30["ex"] > 0 else "之後偏弱")
    v2 = (f"槓桿堆高、價格不動：{bw}，30 天平均 {_pct(b30['mean'])}（比所有日子 {_pct(b30['ex'])}，只有 {b30['n']} 次，"
          f"其他四幣 {len(oc) - same['build']} 個也偏強）。"
          f"加在 50 日均線上，排名分數均線 {_share(s['ma50']['low'])}、F1 {_share(s['f1']['low'])}、F2 {_share(s['f2']['low'])}，"
          f"隨機錯位一樣好 {_share(plc['f1'])}、{_share(plc['f2'])}。")
    v3 = "事先定的標準" + ("通過，但只有幾年資料，只能當線索。" if passed else "沒有通過，兩條規則都不建議加到主訊號上。")
    verdict = v1 + v2 + v3

    # ----- notes -----
    def dtext(kind: str) -> str:
        dd = detail[kind]
        pe = pf if kind == "flush" else pb
        rk = "f1" if kind == "flush" else "f2"
        return (f"事件效果（前後半同為預期方向 {pe['halves']}／2、90% 區間{'不含' if pe['excl'] else '含'}零）{'過' if dd['effect'] else '不過'}；"
                f"{SHORT[rk]} 排名分數{'不低於' if dd['score'] else '低於'}均線、最深虧損中位數{'不比' if dd['loss'] else '比'}均線深、"
                f"隨機錯位 {_share(plc[rk])}、其他幣同方向 {same[kind]}／{len(oc)} → {'過' if dd['ok'] else '不過'}")

    pre_reg = ("事先寫下的設計（看結果之前，之後沒有改；原文英文）：Flush (day D) = 3-day OI change (coins) below the 5% quantile of its "
               "own 365 previous values AND 3-day price return < 0; build-up = 30-day OI change above its trailing-year 90% quantile AND "
               "|30-day price return| below the trailing-year 1/3 quantile of absolute 30-day returns; flags used only from the next close; "
               "episodes = first flagged day after 14 unflagged days. Event study of forward 7/14/30-day returns vs all days, circular "
               "30-day block bootstrap (1,000), halves at the middle day. Predicted: flush positive at 14 days (CryptoQuant), K33 says "
               "negative at 30 days; build-up negative at 30 days. F1 = ma50 but also hold 14 days from a flush flag; F2 = ma50 but cash "
               "during a build-up. Fair exam as ls_ratio (monthly starts x 24 clocks, ordinary person 0.25% & 10% missed, other coins, "
               "ranking = worst of four beats-holding shares, placebo circular shift >= 180 days 200 draws, median deepest loss). Pass = "
               "predicted sign in both halves with 90% interval excluding zero AND ranking >= ma50's AND median deepest loss no worse "
               "AND placebo <= 10% AND same sign on >= 3 of 4 other coins. Sample about five years, stated.")
    notes = [
        pre_reg,
        f"過關檢查明細｜清槓桿＋F1：{dtext('flush')}。槓桿堆高＋F2：{dtext('build')}。整體：{'通過' if passed else '沒有通過'}。"
        f"K33 的說法（30 天比平常差）：前後半同為負 {k33['halves']}／2、90% 區間{'不含' if k33['excl'] else '含'}零 → "
        f"{'成立' if k33['pass'] else '沒有被證實'}。",
        "看結果之後才補的說明（不是事先定的檢查，只能當線索）：清槓桿之後 14 天「比平常弱」這個反方向，"
        f"前後半是 {_pct(fe[H_FLUSH]['halves'][0])}／{_pct(fe[H_FLUSH]['halves'][1])}，90% 區間 {_pct(fe[H_FLUSH]['ci'][0])} 到 "
        f"{_pct(fe[H_FLUSH]['ci'][1])}；其他四幣 14 天比平常弱的有 "
        f"{sum(1 for o in oc.values() if _ok(o['ev']['flush'][H_FLUSH]['ex']) and o['ev']['flush'][H_FLUSH]['ex'] < 0)}／{len(oc)} 個，"
        "所以不能說清槓桿就是賣出訊號。槓桿堆高、價格不動之後，"
        f"比特幣 30 天比平常 {_pct(be[H_BUILD]['ex'])}（區間 {_pct(be[H_BUILD]['ci'][0])} 到 {_pct(be[H_BUILD]['ci'][1])}），"
        "和「危險」的說法方向相反；只有十次左右，而且是看了結果才注意到的，不能拿來當買進理由。",
        f"樣本很短：幣安的未平倉量 {_d(oi.dropna().index[0])} 起才有（約 {yrs_oi:.1f} 年），「一年位置」要先累積 330 天以上，所以 "
        f"{_d(r['first_flag'])} 起才有旗標；清槓桿只有 {x14['n']} 次、槓桿堆高 {b30['n']} 次，涵蓋的大行情只有 2022 年空頭、2023–2025 年多頭"
        "等少數幾段。其他四幣 2021-12 起才有，更短。",
        f"旗標出現的天數（{_d(r['first_flag'])} 起）：清槓桿 {fe['flag_days']:,} 天、槓桿堆高 {be['flag_days']:,} 天；"
        f"F1 在均線不持有時多持有了 {r['extra_on']:,} 天，F2 在均線持有時改拿現金 {r['off']:,} 天。F1 每次提早進場那 14 天裡、均線沒持有的日子"
        f"賺賠（中位數）{_pct(float(np.median(r['f1_extra'])) if r['f1_extra'] else None)}，{len(r['f1_extra'])} 次裡 "
        f"{sum(1 for v in r['f1_extra'] if v > 0)} 次是賺的。",
        "為什麼用「顆數」：未平倉量用美元算時，價格跌本身就會讓它變小，分不出是真的有人平倉還是單純跌價；用顆數（合約數）只反映部位真的減少。"
        "這裡只有幣安一家；K33 說的是「爆倉金額」前 5% 的事件，這裡用未平倉量 3 天大降＋價格下跌當作近似，兩者不完全一樣"
        "（未平倉量下降也可能是大家主動平倉）。CryptoQuant 的 20–30% 是幾篇評論裡挑出來的例子，沒有公開的門檻和完整回測。",
        "和其他研究的分工：forced_flows 已經用 Bybit 未平倉量「單日跌 10% 以上」當爆倉事件、測過買進抱 7／30 天；這裡改用幣安、"
        "「和自己過去一年比」的門檻，並且把它放進 50 日均線的公平考試裡。資金費率在 funding_positioning、多空比在 ls_ratio。",
        _lookahead(r),
        "時間對齊：每個 metrics 檔只取 UTC 0 點那一列（台灣早上 8 點），對到剛好在那個時間收盤的日線；但旗標要等到下一根日線收盤（大約晚一天）才用，偏保守。"
        f"缺的日子沿用前一天，最多 {FFILL} 天。"
        + f"旗標再晚一天：清槓桿 14 天多出來 {_pct(fe[H_FLUSH]['ex'])} → {_pct(r['late']['flush'][H_FLUSH]['ex'])}、"
          f"槓桿堆高 30 天 {_pct(be[H_BUILD]['ex'])} → {_pct(r['late']['build'][H_BUILD]['ex'])}。",
        f"成本很粗略：照課本做每次買賣扣 {COST * 100:.2f}%（另列 {COST_HI * 100:.2f}%）、一般人做法 {P_COST * 100:.2f}%；現金沒有利息；一直抱著沒有扣成本；沒有算稅。"
        "相鄰起算日共用大部分歷史、24 個收盤時間只差幾小時，這些「幾個裡幾個」不是互相獨立的證據。其他幣是今天還在的大幣，有倖存者偏誤。"
        "隨機的部分都用固定的亂數種子，資料沒變的話重跑結果一樣。",
        f"這次計算用了 {time.time() - t0:.0f} 秒（其中讀檔和補下載 {t_dl:.0f} 秒、公平考試 {r['t_grid']:.0f} 秒）。未平倉量和 ls_ratio 共用同一份存檔"
        "（_store 的 ls_ratio_*.json），不另外下載。",
        "過去的結果不保證未來會一樣。這裡只是把流行的說法照原樣測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "oi_flush",
        "title": "清槓桿之後：反彈還是繼續弱",
        "question": ("合約未平倉量在幾天內大降、價格也跌（清槓桿）之後，比特幣通常會反彈 20–30%（CryptoQuant 的說法），還是反而比平常弱"
                     "（K33 的說法）？反過來，未平倉量一路堆高、價格卻不動，是不是比較危險？加在 50 日均線上，能不能幫忙「早一點上車」或「先躲開」？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
