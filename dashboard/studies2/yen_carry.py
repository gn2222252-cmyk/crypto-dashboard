"""Historical study "yen_carry": does a sharp yen surge (carry-trade unwind, BOJ hike, MOF intervention) come before
bitcoin sell-offs?

The claim: Cointelegraph says that in the two prior yen-intervention windows bitcoin fell ~30% from its local highs;
CoinDesk (2024-08-05) reported bitcoin dropping 15% against the yen as carry trades unwound; CryptoQuant says the
clearest effect is on yen-priced bitcoin and evidence of a global (dollar-price) wave is limited. No systematic study
exists - that is the gap filled here.

Pre-registered design (written before any result was seen; not changed afterwards):
  Yen-surge event = USD/JPY 5-day change in its trailing-3-year bottom 2% (point-in-time), events de-duplicated (one
  per 30 days). Event study of BTC forward 7/30/60-day returns and max drawdown vs all days, block bootstrap
  intervals, splits 2013-19 / 2020-now, list every event with its BTC outcome (the episode count will be small - say
  so); also: does BTC fall MORE than the Nasdaq around these events.
  Rule Y1 = ma50 but cash for 30 days after a yen-surge event. Fair exam reusing main_signal_check as ls_ratio /
  lead_followup do (108 monthly starts x 24 clocks, ordinary person 0.25% & 10% missed entries, other coins
  ETH/SOL/BNB/XRP, ranking score, placebo circular shift >= 180 days 200 draws, median deepest loss, big crashes
  avoided via crash_timing's bear list).
  Pass test (stated now): forward 30-day excess negative in both splits with the interval excluding zero AND ranking
  score >= ma50's AND median deepest loss no worse AND placebo share <= 10%.
  Today: the latest USD/JPY 5-day change, the threshold, whether a cash window is open, ma50 / Y1 states - states only.

Details fixed with the design (before results):
  * Data: USD/JPY = Federal Reserve Board H.10 daily noon buying rate in New York (series RXI_N.B.JA, yen per dollar)
    from the Fed's Data Download Program package (key-less). A yen surge = USD/JPY falling.
  * 5-day change = this H.10 business day / 5 H.10 business days earlier - 1. Threshold on day D = the 2% quantile
    of all 5-day changes dated in (D - 1,096 days, D] (only past and present values; at least 500 of them). A surge
    day = change <= threshold. Event = the first surge day, then the next surge day at least 30 calendar days after
    the previous event, and so on.
  * Point in time: the noon-NY rate of day D is used only from the next bitcoin daily close, 00:00 UTC of D+1 (the
    dashboard's candle labelled D) or later. On the 24 closing hours, a close falling on UTC date E uses rates up to
    E-1; for MSTR (US close on date D) rates up to D-1. (The Fed publishes H.10 weekly; the noon rate itself is a
    market price visible at noon New York, so this is the price a person could have seen, not the release date.)
  * Event study: forward h-day return of the bitcoin candle labelled D = close(D+h)/close(D) - 1, h = 7, 30, 60;
    max drawdown = lowest close of D+1..D+30 / close(D) - 1. "All days" = every candle in the same split with the
    forward value known. Excess = event mean - all-days mean. 90% interval: 2,000 draws; events resampled in
    circular blocks of 2 consecutive events, all days in circular blocks of 60 days; fixed seed. Splits by event
    date: 2013-01-01..2019-12-31 and 2020-01-01..now; events need the full forward window.
  * Nasdaq: QQQ (Yahoo adjusted close). For event D the window starts at the first QQQ close dated D+1 or later and
    ends at the last QQQ close dated within 30 calendar days of it; bitcoin over the same window = candle labelled
    the start date to candle labelled the end date. Difference BTC - QQQ; event mean vs all QQQ days the same way,
    bootstrap as above. Yen-priced bitcoin over 30 days (BTC x USD/JPY) is description only.
  * Y1: flag = 1 on the 30 calendar days starting with each event's H.10 date (D .. D+29), used with the lags above;
    position = ma50 x (1 - flag). ma50 = main_signal_check._targets with its seeds, so it is that study's ma50.
  * Exam: main_signal_check's four checks exactly as lead_followup builds them (daily 2015-01..2023-12 = 108 starts;
    72 starts x 24 clocks; ordinary person 10 hours later, 0.25%, 10% missed entries, 200 simulations; ETH, SOL,
    BNB, XRP and MSTR (main_signal_check's other-asset list) with the same yen flag). Ranking score and eligibility
    from main_signal_check._summary's arithmetic; median deepest loss over the 1,728 textbook cells.
  * Placebo: the calendar-day flag from 2012-01-01 rolled circularly by 180 ... (length - 180) days, 200 draws, the
    whole exam rebuilt (50 ordinary-person simulations per clock; the real rule scored the same way for this
    comparison); share with a ranking score at least the real one.
  * Crashes: crash_timing._bears (50% fall, doubling ends it), avoided share for ma50 and Y1; whipsaws from 2015-01.
"""
from __future__ import annotations

import io
import json
import math
import re
import time
import zipfile

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import lead_followup as lf
from dashboard.studies2 import liquidity_lead as ll
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, LOSS_BAR = msc.YEAR, msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
STEP, WINDOW_DAYS, Q, MIN_OBS = 5, 1096, 0.02, 500
DEDUP, CASH = 30, 30
HORIZONS = (7, 30, 60)
SPLITS = (("2013–2019", "2013-01-01", "2019-12-31"), ("2020–現在", "2020-01-01", None))
BOOT, BOOT_SEED, EV_BLOCK, DAY_BLOCK = 2000, 20240805, 2, 60
DRAWS, GAP, PL_SIMS, PL_FROM = 200, 180, 50, "2012-01-01"
KI = {"ma50": 1, "y1": 72}
LAG = {"utc": 0, "clock": 1, "share": 1}
GRID = (msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
CUTS = ("2019-06-30", "2022-09-30", "2024-07-31")
SERIES = "RXI_N.B.JA"
FED_URL = ll.FED_ZIP.format("H10")

_pct, _pp, _d, _num = lf._pct, lf._pp, lf._d, lf._num
_ok, _share = lf._ok, lf._share


# ---------- data ----------

def _usdjpy_fetch() -> dict:
    zf = zipfile.ZipFile(io.BytesIO(ll._get(FED_URL, timeout=180).content))
    member = next(n for n in zf.namelist() if n.lower().endswith("_data.xml"))
    text = zf.read(member).decode("utf-8")
    i = text.find(f'SERIES_NAME="{SERIES}"')
    if i < 0:
        raise RuntimeError("H.10 裡找不到日圓日資料")
    j = text.find("</kf:Series>", i)
    out = {}
    for value, day in re.findall(r'OBS_VALUE="([^"]+)" TIME_PERIOD="([^"]+)"', text[i:j]):
        try:
            v = float(value)
        except ValueError:
            continue
        if math.isfinite(v) and v > 0:
            out[day[:10]] = v
    return out


def _usdjpy() -> tuple[pd.Series, str]:
    path = common.STORE / "yen_carry_usdjpy.json"
    note = ""
    try:
        days = _usdjpy_fetch()
        if len(days) < 5000:
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
        note = f"美國聯準會 H.10 這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）。"
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s, note


def _load() -> dict:
    problems = []
    close = te._daily_close()
    fx, note = _usdjpy()
    problems += [note] if note else []
    try:
        qqq = ro._share("QQQ")
    except Exception as exc:  # noqa: BLE001 - only the Nasdaq comparison needs it
        qqq = None
        problems.append(f"QQQ 這次取不到（{type(exc).__name__}），那斯達克的比較沒有算。")
    other, more = msc._load_other()
    problems += more
    return {"close": close, "fx": fx, "qqq": qqq, "other": other, "problems": problems}


# ---------- the yen-surge signal ----------

def _surge(fx: pd.Series) -> pd.DataFrame:
    ch = fx / fx.shift(STEP) - 1
    ch = ch.dropna()
    thr = ch.rolling(f"{WINDOW_DAYS}D", min_periods=MIN_OBS).quantile(Q)
    return pd.DataFrame({"fx": fx.reindex(ch.index), "ch": ch, "thr": thr, "surge": (ch <= thr) & thr.notna()})


def _events(sig: pd.DataFrame) -> list[pd.Timestamp]:
    out = []
    for d in sig.index[sig["surge"].values]:
        if not out or (d - out[-1]).days >= DEDUP:
            out.append(d)
    return out


def _flag(events: list, end: pd.Timestamp, first="1974-01-01") -> pd.Series:
    days = pd.date_range(first, end, freq="D")
    f = pd.Series(0.0, index=days)
    for e in events:
        f.loc[e:e + (CASH - 1) * DAY] = 1.0
    return f


# ---------- event study ----------

def _fwd(close: pd.Series) -> dict:
    c = close.values
    n = len(c)
    out = {}
    for h in HORIZONS:
        r = np.full(n, np.nan)
        r[:n - h] = c[h:] / c[:n - h] - 1
        out[h] = pd.Series(r, index=close.index)
    low = pd.Series(c[::-1]).rolling(30, min_periods=30).min().values[::-1]  # low[i] = min c[i..i+29]
    m = np.full(n, np.nan)
    m[:n - 30] = low[1:n - 29] / c[:n - 30] - 1
    out["dd"] = pd.Series(m, index=close.index)
    return out


def _boot(ev: np.ndarray, base: np.ndarray, rng) -> tuple:
    ev, base = ev[np.isfinite(ev)], base[np.isfinite(base)]
    if len(ev) < 3 or len(base) < 200:
        return None, None
    ne, nb = len(ev), len(base)
    ke, kb = math.ceil(ne / EV_BLOCK), math.ceil(nb / DAY_BLOCK)
    out = np.empty(BOOT)
    for i in range(BOOT):
        se = (rng.integers(0, ne, ke)[:, None] + np.arange(EV_BLOCK)).ravel()[:ne] % ne
        sb = (rng.integers(0, nb, kb)[:, None] + np.arange(DAY_BLOCK)).ravel()[:nb] % nb
        out[i] = ev[se].mean() - base[sb].mean()
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def _in(idx, a, b) -> np.ndarray:
    m = idx >= pd.Timestamp(a)
    return m & (idx <= pd.Timestamp(b)) if b else m


def _qqq_pairs(close: pd.Series, qqq: pd.Series, days) -> pd.DataFrame:
    """For each day D: QQQ from its first close dated D+1 or later to the last close within 30 days, bitcoin over the
    same dates."""
    qi = qqq.index
    rows = {}
    for d in days:
        i = int(qi.searchsorted(d + DAY))
        if i >= len(qi):
            continue
        a = qi[i]
        j = int(qi.searchsorted(a + 30 * DAY, side="right")) - 1
        if a + 30 * DAY > qi[-1] or j <= i:
            continue
        b = qi[j]
        if a not in close.index or b not in close.index:
            continue
        q = qqq.iloc[j] / qqq.iloc[i] - 1
        bt = close[b] / close[a] - 1
        rows[d] = (bt, q, bt - q)
    return pd.DataFrame(rows, index=["btc", "qqq", "diff"]).T


def _event_study(close: pd.Series, fx: pd.Series, qqq, events: list) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    fw = _fwd(close)
    ev = pd.DatetimeIndex([e for e in events if e >= pd.Timestamp(SPLITS[0][1]) and e in close.index])
    fx_ff = fx.reindex(fx.index.union(close.index)).ffill()
    qp_ev = _qqq_pairs(close, qqq, ev) if qqq is not None else None
    qp_all = _qqq_pairs(close, qqq, qqq.index[qqq.index >= pd.Timestamp(SPLITS[0][1])]) if qqq is not None else None
    rows = []
    for e in ev:
        r = {"d": e, "ch": None, "btc": {h: fw[h].get(e, np.nan) for h in HORIZONS}, "dd": fw["dd"].get(e, np.nan)}
        e30 = e + 30 * DAY
        r["jpy30"] = ((1 + r["btc"][30]) * fx_ff.get(e30, np.nan) / fx_ff.get(e, np.nan) - 1
                      if e30 <= fx.index[-1] and _ok(r["btc"][30]) else np.nan)
        if qp_ev is not None and e in qp_ev.index:
            r["qqq"], r["bq"] = qp_ev.loc[e, "qqq"], qp_ev.loc[e, "btc"]
        else:
            r["qqq"] = r["bq"] = np.nan
        rows.append(r)
    stats = {}
    for name, a, b in SPLITS + (("全部", SPLITS[0][1], None),):
        ein = _in(ev, a, b)
        cin = _in(close.index, a, b)
        st = {"n": int(ein.sum())}
        for h in list(HORIZONS) + ["dd"]:
            evv = fw[h].reindex(ev[ein]).values.astype(float)
            base = fw[h].values[cin].astype(float)
            ok = evv[np.isfinite(evv)]
            st[h] = {"n": len(ok), "ev": float(ok.mean()) if len(ok) else None,
                     "all": float(np.nanmean(base)), "ex": (float(ok.mean() - np.nanmean(base)) if len(ok) else None),
                     "down": float((ok < 0).mean()) if len(ok) else None, "ci": _boot(evv, base, rng)}
        if qp_ev is not None:
            qe = qp_ev[_in(qp_ev.index, a, b)] if len(qp_ev) else qp_ev
            qa = qp_all[_in(qp_all.index, a, b)]
            if len(qe):
                st["q"] = {"n": len(qe), "btc": float(qe["btc"].mean()), "qqq": float(qe["qqq"].mean()),
                           "diff": float(qe["diff"].mean()), "all": float(qa["diff"].mean()),
                           "worse": float((qe["diff"] < 0).mean()),
                           "ci": _boot(qe["diff"].values.astype(float), qa["diff"].values.astype(float), rng)}
        stats[name] = st
    return {"rows": rows, "stats": stats}


# ---------- the exam (lead_followup's arithmetic, one filter) ----------

def _pos(kind: str, ma: np.ndarray, flag) -> np.ndarray:
    return ma if kind == "ma50" else ma * (1 - flag)


class _Exam:
    def __init__(self, data: dict, price, b0, pre):
        close = data["close"]
        self.close = close
        self.btc_ma = msc._targets(close)["ma50"].values.astype(float)
        self.btc_change = close.pct_change().fillna(0.0).values
        starts, labels = msc._month_starts(close.index, *GRID)
        hr, hd = msc._suffix(msc._equity(self.btc_change), starts, YEAR)
        self.daily = {"starts": starts, "labels": labels, "hold": (hr[:, 0], hd[:, 0])}
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
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "change": change, "starts": starts,
                               "labels": labels, "year": year, "share": bool(since), "hold": (hr[:, 0], hd[:, 0])}

    def btc_pos(self, kind: str, flag: pd.Series) -> np.ndarray:
        f = lf._usable(flag, self.close.index, LAG["utc"]) if kind != "ma50" else 0.0
        return _pos(kind, self.btc_ma, f)

    def summary(self, kind: str, flag: pd.Series, sims: int = SIMS) -> dict:
        pos = self.btc_pos(kind, flag)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        r, dd = msc._suffix(msc._equity(net), self.daily["starts"], YEAR)
        dr, dd = r[:, 0], dd[:, 0]
        dh = self.daily["hold"]
        f = lf._usable(flag, self.m.dates, LAG["clock"]) if kind != "ma50" else 0.0
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = np.nan_to_num(_pos(kind, self.cma[h], f), nan=0.0).astype(float)
            rr, d2 = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = rr[:, 0], d2[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            p = rt._missed(wl, P_MISS, rng, sims)
            rr, d2 = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, p, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = rr, d2
        hr, hd = self.chold
        edge = tr - hr[:, None]
        pedge = pr - hr[:, None, None]
        each = {}
        for key, o in self.other.items():
            fo = lf._usable(flag, o["close"].index, LAG["share" if o["share"] else "utc"]) if kind != "ma50" else 0.0
            po = _pos(kind, o["ma"], fo)
            neto, _ = te._run(np.concatenate([[np.nan], po[:-1]]), o["change"], COST)
            ro_, do_ = msc._suffix(msc._equity(neto), o["starts"], o["year"])
            each[key] = (len(ro_), int((ro_[:, 0] > o["hold"][0]).sum()), int((do_[:, 0] > o["hold"][1]).sum()),
                         float(np.median(ro_[:, 0] - o["hold"][0])))
        x = {"d_n": len(dr), "d_beat": int((dr > dh[0]).sum()), "d_edge": float(np.median(dr - dh[0])),
             "d_shallow": int((dd > dh[1]).sum()),
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


# ---------- look-ahead ----------

def _lookahead(data: dict, flag: pd.Series) -> tuple[int, int]:
    rng = np.random.default_rng(13)
    fx, close = data["fx"], data["close"]
    total = bad = 0
    for cut in CUTS:
        c = pd.Timestamp(cut)
        fx2 = fx.where(fx.index <= c, fx * np.exp(rng.normal(0, 0.05, len(fx))))
        f2 = _flag(_events(_surge(fx2)), flag.index[-1])
        idx = close.index[close.index <= c]
        a = lf._usable(flag, idx, LAG["utc"])
        b = lf._usable(f2, idx, LAG["utc"])
        total += 1
        bad += int(not np.array_equal(a, b))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, fx = data["close"], data["fx"]
    sig = _surge(fx)
    events = _events(sig)
    end = max(pd.Timestamp(common.today_utc()), fx.index[-1]) + 40 * DAY
    flag = _flag(events, end)
    es = _event_study(close, fx, data["qqq"], events)

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    none = pd.Series(dtype=float)
    s = {"ma50": ex.summary("ma50", none), "y1": ex.summary("y1", flag)}
    pos = {"ma50": ex.btc_pos("ma50", none), "y1": ex.btc_pos("y1", flag)}
    ratio = {k: lf._ratio(pos[k], close) for k in pos}
    hold_ratio = lf._ratio(np.ones(len(close)), close)

    # placebo
    real_small = ex.summary("y1", flag, PL_SIMS)
    part = flag[flag.index >= pd.Timestamp(PL_FROM)]
    prng = np.random.default_rng(SEED + 9)
    hits = hits_r = 0
    lows = []
    for sh in prng.integers(GAP, len(part) - GAP, DRAWS):
        f2 = pd.Series(np.roll(part.values, int(sh)), index=part.index)
        x = ex.summary("y1", f2, PL_SIMS)
        lows.append(x["low"])
        hits += int(lf._score_ge(x, real_small))
        rr = lf._ratio(ex.btc_pos("y1", f2), close)
        hits_r += int(rr is not None and ratio["y1"] is not None and rr >= ratio["y1"])

    # crashes, whipsaws, days differing
    c_arr, idx = close.values, close.index
    crash = []
    for b in ct._bears(close):
        row = {"top": idx[b["top"]], "low": idx[b["low"]], "drop": c_arr[b["low"]] / c_arr[b["top"]] - 1,
               "done": b.get("done", False)}
        for k in pos:
            row[k] = ct._crash(c_arr, [(1.0, pos[k])], b)["avoid"]
        crash.append(row)
    first = int(idx.searchsorted(pd.Timestamp(GRID[0])))
    whip = {k: ct._whipsaws(c_arr, [(1.0, pos[k])], first) for k in pos}
    a = idx >= pd.Timestamp(GRID[0])
    differ = (int((np.nan_to_num(pos["ma50"][a]) != np.nan_to_num(pos["y1"][a])).sum()), int(a.sum()))
    flag_days = int(lf._usable(flag, idx[a], 0).sum())
    return {"sig": sig, "events": events, "flag": flag, "es": es, "ex": ex, "s": s, "pos": pos, "ratio": ratio,
            "hold_ratio": hold_ratio, "real_small": real_small, "plc": hits / DRAWS, "plc_ratio": hits_r / DRAWS,
            "plc_low": float(np.median(lows)), "crash": crash, "whip": whip, "differ": differ,
            "flag_days": flag_days, "look": _lookahead(data, flag), "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def _ci(ci, scale: float = 100) -> str:
    if not ci or ci[0] is None:
        return "—"
    return f"{ci[0] * scale:+.1f} 到 {ci[1] * scale:+.1f}"


def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, fx, sig, ex, s = data["close"], data["fx"], r["sig"], r["ex"], r["s"]
    es = r["es"]
    st = es["stats"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    n_c = s["y1"]["c_n"]

    # ----- pass test -----
    ex30 = {name: st[name][30] for name, _, _ in SPLITS}
    ev_ok = all(_ok(v["ex"]) and v["ex"] < 0 and v["ci"][1] is not None and v["ci"][1] < 0 for v in ex30.values())
    score_ok = lf._score_ge(s["y1"], s["ma50"])
    loss_ok = s["y1"]["c_deep_med"] >= s["ma50"]["c_deep_med"] - 1e-12
    plc_ok = r["plc"] <= 0.10
    passed = ev_ok and score_ok and loss_ok and plc_ok

    # ----- table 1: event study -----
    rows1 = []
    for name in [n for n, _, _ in SPLITS] + ["全部"]:
        x = st[name]
        for h in list(HORIZONS) + ["dd"]:
            v = x[h]
            label = f"之後 {h} 天報酬" if h != "dd" else "之後 30 天最深跌幅"
            rows1.append([name, label, f"{v['n']}", _p(v["ev"]), _p(v["all"]),
                          (f"{v['ex'] * 100:+.1f}" if _ok(v["ex"]) else "—"), _ci(v["ci"]),
                          _share(v["down"]) if h != "dd" else "—"])
    t1 = {"caption": ("日圓急升事件之後比特幣的表現，和同一段期間的所有日子比。事件＝美元兌日圓 5 個營業日的跌幅落在過去三年最低的 2%（只用當時已知的資料），"
                      f"30 天內只算第一次。比特幣從事件那天（紐約中午匯率）之後的第一個 UTC 0 點收盤算起。差距單位是百分點；區間是 {BOOT:,} 次區塊重抽"
                      f"的 90% 範圍（事件每 {EV_BLOCK} 個一組、所有日子每 {DAY_BLOCK} 天一組）。事件只有十幾到二十幾次，區間很寬是正常的。"),
          "columns": ["期間", "看什麼", "事件數", "事件後平均", "所有日子平均", "差距（百分點）", "差距 90% 區間", "事件後下跌的比例"],
          "rows": rows1}

    # ----- table 2: every event -----
    rows2 = []
    for e in es["rows"]:
        d = e["d"]
        rows2.append([_d(d), f"{sig.loc[d, 'fx']:.2f}", _p(sig.loc[d, "ch"]), _p(sig.loc[d, "thr"]),
                      _p(e["btc"][7]), _p(e["btc"][30]), _p(e["btc"][60]), _p(e["dd"]), _p(e["jpy30"]),
                      _p(e["qqq"]), (_pp((e["bq"] - e["qqq"]) * 100) if _ok(e["bq"]) and _ok(e["qqq"]) else "—")])
    t2 = {"caption": ("每一次日圓急升事件（2013 年起）和比特幣之後的結果。5 日變化是美元兌日圓的跌幅（負＝日圓變貴）；門檻是當時過去三年 5 日變化最低 2% 的界線。"
                      "日圓計價＝比特幣以日圓計價 30 天的漲跌；QQQ 和「比特幣減 QQQ」用同一段日期（事件隔天起第一個美股收盤，到 30 天內最後一個收盤）。"
                      "「—」表示時間還沒到或資料不夠。"),
          "columns": ["事件日（紐約）", "美元兌日圓", "5 日變化", "當時門檻", "比特幣 7 天", "30 天", "60 天", "30 天內最深跌幅",
                      "日圓計價 30 天", "QQQ 約 30 天", "比特幣減 QQQ（百分點）"],
          "rows": rows2}

    # ----- table 3: Nasdaq comparison -----
    rows3 = []
    for name in [n for n, _, _ in SPLITS] + ["全部"]:
        q = st[name].get("q")
        if not q:
            continue
        rows3.append([name, f"{q['n']}", _p(q["btc"]), _p(q["qqq"]), f"{q['diff'] * 100:+.1f}", f"{q['all'] * 100:+.1f}",
                      f"{(q['diff'] - q['all']) * 100:+.1f}", _ci(q["ci"]), _share(q["worse"])])
    t3 = {"caption": ("比特幣是不是比那斯達克跌得更多：同一段約 30 天的日期，比特幣報酬減 QQQ 報酬（追蹤那斯達克 100 的基金，Yahoo 還原配息收盤）。"
                      "和所有美股交易日同樣算法的平均比；區間同上。"),
          "columns": ["期間", "事件數", "比特幣平均", "QQQ 平均", "事件：比特幣減 QQQ", "所有日子：比特幣減 QQQ", "差距（百分點）",
                      "差距 90% 區間", "比特幣輸給 QQQ 的比例"],
          "rows": rows3}

    # ----- table 4: the fair exam -----
    rows4 = []
    for key, label in (("ma50", "50 日均線（基準）"), ("y1", "Y1：50 日均線，日圓急升後 30 天現金")):
        x = s[key]
        ck = x["checks"]
        rows4.append([label, f"{_share(ck[0])}（{x['d_n']} 個）", _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    rows4.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(ex.chold[1])))}｜—", "—"])
    lab = ex.daily["labels"]
    t4 = {"caption": (f"main_signal_check 的公平考試，四項都是「報酬贏過同一天起一直抱著」的比例：比特幣 UTC 0 點日線每月 1 日起算 {_d(lab[0])[:7]} 到 "
                      f"{_d(lab[-1])[:7]}（{len(lab)} 個）；{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {COST * 100:.2f}%）；"
                      f"同樣格子照一般人方式做（10 小時後才做、成本 {P_COST * 100:.2f}%、一成進場漏掉、每格 {SIMS} 次）；其他資產 "
                      f"{'、'.join(ex.other)} 起算月合計（用同一個日圓訊號）。排名分數＝四項最低那項，前提是最深虧損比一直抱著淺的比例至少 "
                      f"{LOSS_BAR * 100:.0f}%。最後兩欄是課本｜一般人做法的中位數。"),
          "columns": ["規則", "比特幣起算月", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows4}

    # ----- table 5: pass test, placebo, crashes -----
    def mark(ok: bool) -> str:
        return "過" if ok else "不過"
    rows5 = [
        ["事件後 30 天比所有日子差，兩段都是而且區間不含 0",
         "；".join(f"{n} {ex30[n]['ex'] * 100:+.1f}（{_ci(ex30[n]['ci'])}）" if _ok(ex30[n]["ex"]) else f"{n} —"
                  for n, _, _ in SPLITS), mark(ev_ok)],
        ["排名分數不低於 50 日均線", f"{_share(s['y1']['low'])} 對 {_share(s['ma50']['low'])}", mark(score_ok)],
        ["最深虧損中位數不比 50 日均線深", f"{_pct(s['y1']['c_deep_med'])} 對 {_pct(s['ma50']['c_deep_med'])}", mark(loss_ok)],
        [f"隨機錯位一樣好的比例 ≤ 10%（{DRAWS} 次）",
         f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['real_small']['low'])}）", mark(plc_ok)],
        ["總結", "四項都要過", "通過" if passed else "沒有通過"],
        ["（描述）報酬÷波動，2017-09 起 UTC 日線", f"Y1 {_num(r['ratio']['y1'])}｜均線 {_num(r['ratio']['ma50'])}｜抱著 "
                                         f"{_num(r['hold_ratio'])}；打亂後一樣好 {_share(r['plc_ratio'])}", "—"],
        ["（描述）和均線持有狀態不同的天數（2015 起）", f"{r['differ'][0]:,} 天／{r['differ'][1]:,} 天；日圓現金窗開著 {r['flag_days']:,} 天", "—"],
    ]
    for b in r["crash"]:
        rows5.append([f"（描述）大跌 {_d(b['top'])} → {_d(b['low'])}" + ("" if b["done"] else "（還沒結束）") + f"，跌 {_p(b['drop'])}",
                      f"躲掉：均線 {_share(b['ma50'])}｜Y1 {_share(b['y1'])}", "—"])
    w = r["whip"]
    rows5.append(["（描述）一年下車幾次｜其中 30 天內更高價買回（2015 起）",
                  f"均線 {w['ma50']['exits']:.1f}｜{w['ma50']['whips']:.1f}；Y1 {w['y1']['exits']:.1f}｜{w['y1']['whips']:.1f}", "—"])
    t5 = {"caption": (f"事先寫好的過關標準（四項都要過）和描述用的數字。隨機錯位：把 {PL_FROM[:4]} 年起的「日圓現金窗」整串循環平移 {GAP} 天到「長度 − {GAP}」天，"
                      f"整套考試重做，一般人做法每個收盤時間模擬 {PL_SIMS} 次，真的規則也這樣算再比。大跌段用 crash_timing 的定義（收盤從高點跌 50% 以上）。"),
          "columns": ["項目", "數字", "結果"], "rows": rows5}
    tables = [t1, t2, t3, t4, t5]

    # ----- today -----
    last = close.index[-1]
    lastfx = sig.index[-1]
    open_until = None
    past = [e for e in r["events"] if e <= last]
    if past and past[-1] + (CASH - 1) * DAY >= last:
        open_until = past[-1] + (CASH - 1) * DAY

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"美元兌日圓（聯準會 H.10 紐約中午，最新 {_d(lastfx)}）",
         "value": (f"{sig['fx'].iloc[-1]:.2f}；5 個營業日變化 {_p(sig['ch'].iloc[-1])}，過去三年最低 2% 的門檻 {_p(sig['thr'].iloc[-1])}，"
                   f"{'達到' if bool(sig['surge'].iloc[-1]) else '沒有達到'}日圓急升")},
        {"label": "最近一次日圓急升事件",
         "value": (f"{_d(past[-1])}" + (f"，現金窗開到 {_d(open_until)}" if open_until is not None else "，現金窗已經結束")) if past else "—"},
        {"label": "比特幣：50 日均線｜Y1", "value": f"{word(r['pos']['ma50'][-1])}｜{word(r['pos']['y1'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述日圓訊號和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    a30 = st["全部"][30]
    q = st["全部"].get("q")
    verdict = (f"{'通過' if passed else '沒有通過'}事先寫好的標準。2013 年起只有 {st['全部']['n']} 次日圓急升事件（"
               + "、".join(f"{n} {st[n]['n']} 次" for n, _, _ in SPLITS) + "），樣本很少。"
               f"事件後 30 天比特幣平均 {_p(a30['ev'])}，所有日子 {_p(a30['all'])}，差 {a30['ex'] * 100:+.1f} 個百分點"
               f"（2013–2019 {ex30['2013–2019']['ex'] * 100:+.1f}、2020 後 {ex30['2020–現在']['ex'] * 100:+.1f}，"
               f"區間{'都不含 0' if ev_ok else '至少一段含 0'}）。"
               + (f"同一段日期比特幣比 QQQ 平均多 {q['diff'] * 100:+.1f} 個百分點（平常 {q['all'] * 100:+.1f}）。" if q else "")
               + f"Y1（均線＋急升後 30 天現金）排名分數 {_share(s['y1']['low'])} 對均線 {_share(s['ma50']['low'])}、"
               f"最深虧損中位數 {_pct(s['y1']['c_deep_med'])} 對 {_pct(s['ma50']['c_deep_med'])}、隨機錯位 {_share(r['plc'])}。")

    tot, bad = r["look"]
    notes = [
        ("事先寫死的設計和標準（照原文，跑之前寫好）：Yen-surge event = USD/JPY 5-day change in its trailing-3-year bottom 2% "
         "(point-in-time), events de-duplicated (one per 30 days); event study of BTC forward 7/30/60-day returns and max drawdown vs "
         "all days, block bootstrap intervals, splits 2013-19 / 2020-now, list every event with its BTC outcome; also does BTC fall "
         "MORE than the Nasdaq around these events. Rule Y1 = ma50 but cash for 30 days after a yen-surge event. Fair exam as "
         "main_signal_check (108 monthly starts × 24 clocks, ordinary person 0.25% & 10% missed entries, other coins, ranking score, "
         "placebo circular shift ≥ 180 days 200 draws, median deepest loss, big crashes avoided). Pass test: forward 30-day excess "
         "negative in both splits with interval excluding zero AND ranking ≥ ma50's AND median deepest loss no worse AND placebo ≤ 10%."),
        ("說法從哪來：Cointelegraph 說前兩次日本財務省干預匯市的期間，比特幣從當時高點跌了約 30%；CoinDesk 2024-08-05 報導套利交易平倉時，"
         "比特幣對日圓跌了 15%、跌得比對美元多；CryptoQuant 認為最清楚的影響是在日圓計價的比特幣，全球性的拋售證據有限。"
         "之前沒有人用固定規則把所有日圓急升的日子都拿出來看，這份研究補這個缺口。"),
        ("時間對齊：H.10 是紐約中午的匯率，第 D 天的值只從比特幣下一個 UTC 0 點收盤（D+1 的 0 點，儀表板標為 D 的那根日線）起才拿來用；"
         "換收盤時間時，落在 UTC 日期 E 的收盤只用到 E−1 的匯率；MSTR（美股收盤）也只用到前一天。聯準會其實每週才公布一次 H.10，"
         "但中午匯率本身是當下市場上看得到的價格，這裡把它當作當時看得到的匯價，不是公布日。"
         f"偷看未來的檢查：在 {'、'.join(CUTS)} 之後把匯率換成亂數重算，切點以前的現金窗 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("門檻每天用過去三年（1,096 天）的 5 日變化重新算最低 2%，至少要 500 個值；一次事件後 30 天內再出現的急升不另算新事件，"
         "Y1 的現金窗就是每次事件當天起 30 天。所以 CoinDesk 報導的 2024-08-05 不是單獨一次事件：它落在 2024-07-17 那次事件的 30 天裡，"
         "結果算在 2024-07-17 那一列。dollar index（美元指數）當過濾的測試在 cross_market 研究，這裡只看日圓。"),
        ("事件只有十幾到二十幾次，彼此間隔至少 30 天，但 60 天的報酬會重疊；平均數很容易被一兩次大事件（例如 2020-03 疫情崩盤）帶著走，"
         "所以每一次都列在表上。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，考試裡「幾個裡幾個」不是互相獨立的證據。"),
        ("其他資產用 main_signal_check 自己的清單（ETH、SOL、BNB、XRP，加上它原本就有的 MSTR），全部用同一個日圓訊號，50 日均線用各自的價格。"
         f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%，現金沒有利息、一直抱著不扣成本、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "yen_carry",
        "title": "日圓急升與比特幣下跌",
        "question": ("日圓突然大漲（套利交易平倉、日本央行升息、財務省干預）之後，比特幣真的比較容易跌嗎？跌得比美股多嗎？"
                     "如果日圓急升後先拿現金 30 天，能不能比單純的 50 日均線好？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Cointelegraph：Bitcoin crashed 30% after the last yen intervention",
             "url": "https://cointelegraph.com/news/bitcoin-crashed-30-percent-last-yen-intervention-but-catch"},
            {"name": "CoinDesk（2024-08-05）：Bitcoin drops 15% against Japanese yen as yen carry trades unwind",
             "url": "https://coindesk.com/markets/2024/08/05/bitcoin-drops-15-against-japanese-yen-outpacing-declines-versus-usd-as-yen-carry-trades-unwind"},
            {"name": "CryptoQuant Quicktake：What a surging yen means for bitcoin",
             "url": "https://cryptoquant.com/insights/quicktake/6aa1c7fa26ed15760f9c61e8-What-a-Surging-Yen-Means-for-Bitcoin-BOJ-Rate-Hike-Expectations-and-the-Carry-Tr"},
            {"name": "美國聯準會 H.10 匯率（Data Download Program 整包下載；日圓日資料 RXI_N.B.JA）", "url": FED_URL},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：QQQ 還原配息日收盤", "url": "https://finance.yahoo.com/quote/QQQ/history"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
