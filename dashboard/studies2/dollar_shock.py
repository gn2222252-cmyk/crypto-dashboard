"""Historical study "dollar_shock": after a sharp drop of the US dollar, does bitcoin rise more than usual over the
next 5 / 10 / 20 days, does adding that as an extra entry to the 50-day average rule help, and who moves first?

Pre-registered design (written before any bitcoin result was looked at; not changed afterwards):
  * Dollar series = the Federal Reserve's nominal broad dollar index, daily (FRED DTWEXBGS, from 2006-01; the same
    numbers are in the Fed's H.10 release as JRXWTFB_N.B). FRED's key-less CSV is tried first (fixed user agent,
    timeouts, two tries); if FRED does not answer, the Fed's own H.10 Data Download Program package; if neither
    answers, the copy stored by the previous run (dashboard/cache/_store/dollar_shock_broad.json).
  * 3-day change on H.10 business day D = index(D) / index(3 business days earlier) - 1. Shock day = change <= -2%.
    Event = the first shock day, then the next shock day at least 20 calendar days after the previous event, etc.
  * Point in time: the index dated D is built from noon New York rates of day D, so it is used only from the next
    bitcoin daily close, 00:00 UTC of D+1 (the dashboard's candle labelled D, te._daily_close) or later. On the 24
    closing hours a close falling on UTC date E uses values up to E-1; for MSTR (US close on D) up to D-1 - exactly
    yen_carry's conventions. (H.10 itself is published weekly; the rates are market prices visible on the day.
    Report only: the same rule acting 7 days later, as if waiting for the weekly release.)
  * Event study: forward h-day return of the candle labelled D = close(D+h)/close(D) - 1, h = 5, 10, 20, for events
    from 2011-01-01 with the full window known, against all candles in the same span. Random-day comparison: 10,000
    random samples of all-days of the same size (fixed seed); report the share whose mean is at least the event
    mean. Splits by event date 2011-16 / 2017-21 / 2022-now.
  * Rule D1 = ma50 (main_signal_check._targets, its seeds) OR "within the 20 calendar days D .. D+19 starting with an
    event" (flag used with the lags above). Fair exam = main_signal_check's four checks exactly as yen_carry builds
    them (108 daily starts 2015-01..2023-12; 72 starts x 24 clocks; ordinary person 10 h later, 0.25% cost, 10% missed
    entries, 200 simulations; ETH/SOL/BNB/XRP/MSTR with their own ma50 and the same dollar flag). Ranking score =
    lowest of the four beats-holding shares with the 95% loss eligibility; median deepest loss over the 1,728
    textbook cells. Placebo: the calendar-day flag from 2010-01-01 rolled circularly by 180 ... (length - 180) days,
    200 draws, whole exam rebuilt (50 ordinary-person simulations per clock, the real rule scored the same way);
    share of draws with a ranking score at least the real one.
  * Mechanism (report only): correlation of daily dollar returns with daily bitcoin returns at lags -5..+5 business
    days (bitcoin return over the same business-day step: candle labelled D vs candle labelled the previous H.10
    day), 2013-now and 2020-now, with the +-2/sqrt(n) noise band. Positive lag = dollar first, bitcoin later.

  Pass test (all must hold):
    (a) event excess (event mean - all-days mean) > 0 at 5, 10 and 20 days; at 20 days the share of random samples
        doing as well <= 10%; and 20-day excess > 0 in every split that has at least 3 events;
    (b) D1 ranking score >= ma50's AND D1 median deepest loss no worse than ma50's AND placebo share <= 10%.
  Grade: pass = (a) and (b); mixed = exactly one of (a), (b); fail = neither; unclear = fewer than 8 events in total.
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
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import lead_followup as lf
from dashboard.studies2 import liquidity_lead as ll
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import yen_carry as yc

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, LOSS_BAR = msc.YEAR, msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
STEP, DROP, GAPDAYS, HOLD = 3, -0.02, 20, 20
HORIZONS = (5, 10, 20)
EV_FROM = "2011-01-01"
SPLITS = (("2011–16", "2011-01-01", "2016-12-31"), ("2017–21", "2017-01-01", "2021-12-31"), ("2022–今", "2022-01-01", None))
RAND, RAND_SEED = 10000, 20060102
DRAWS, GAP, PL_SIMS, PL_FROM = 200, 180, 50, "2010-01-01"
KI = {"ma50": 1, "d1": 74}
LAG = yc.LAG
MIN_EVENTS = 8
XLAGS = range(-5, 6)
CUTS = ("2015-06-30", "2020-03-15", "2024-06-30")
FRED_ID = "DTWEXBGS"
FRED_URL = bc.FRED_CSV.format(FRED_ID)
H10_SERIES = "JRXWTFB_N.B"
FED_URL = ll.FED_ZIP.format("H10")

_pct, _pp, _d, _num = lf._pct, lf._pp, lf._d, lf._num
_ok, _share = lf._ok, lf._share
_RUN: dict = {"fred": None}


# ---------- data ----------

def _fred_fetch() -> dict:
    frame = pd.read_csv(io.StringIO(bc._get(FRED_URL, timeout=(8, 25), tries=2).text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return {str(day)[:10]: float(v) for day, v in zip(frame.iloc[:, 0], values) if math.isfinite(v) and v > 0}


def _h10_fetch() -> dict:
    zf = zipfile.ZipFile(io.BytesIO(ll._get(FED_URL, timeout=180).content))
    member = next(n for n in zf.namelist() if n.lower().endswith("_data.xml"))
    text = zf.read(member).decode("utf-8")
    i = text.find(f'SERIES_NAME="{H10_SERIES}"')
    if i < 0:
        raise RuntimeError("H.10 裡找不到廣義美元指數日資料")
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


def _dollar() -> tuple[pd.Series, str, str]:
    """Broad dollar index (DatetimeIndex) from FRED, else the Fed's H.10 package, else the stored copy."""
    path = common.STORE / "dollar_shock_broad.json"
    days, where, note = None, "", ""
    for name, fetch in (("FRED DTWEXBGS", _fred_fetch), ("聯準會 H.10 JRXWTFB_N.B", _h10_fetch)):
        try:
            got = fetch()
            if len(got) < 4000:
                raise RuntimeError(f"只回傳 {len(got)} 筆")
            days, where = got, name
            break
        except Exception as exc:  # noqa: BLE001 - FRED often does not answer; the Fed's own package is the same data
            if name.startswith("FRED"):
                _RUN["fred"] = type(exc).__name__
    if days is not None:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    else:
        if not path.exists():
            raise RuntimeError("FRED 和聯準會 H.10 都抓不到廣義美元指數，也沒有存下來的資料")
        days = json.loads(path.read_text(encoding="utf-8"))
        where = "上次存下來的廣義美元指數"
        note = f"FRED 和聯準會 H.10 這次都抓不到，改用上次存下來的資料（到 {max(days)}）。"
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s[s.index < pd.Timestamp(common.today_utc())], where, note


def _load() -> dict:
    problems = []
    close = te._daily_close()
    fx, where, note = _dollar()
    problems += [note] if note else []
    other, more = msc._load_other()
    problems += more
    return {"close": close, "fx": fx, "where": where, "other": other, "problems": problems}


# ---------- the dollar-shock signal ----------

def _shock(fx: pd.Series) -> pd.DataFrame:
    ch = (fx / fx.shift(STEP) - 1).dropna()
    return pd.DataFrame({"fx": fx.reindex(ch.index), "ch": ch, "shock": ch <= DROP})


def _events(sig: pd.DataFrame) -> list[pd.Timestamp]:
    out = []
    for d in sig.index[sig["shock"].values]:
        if not out or (d - out[-1]).days >= GAPDAYS:
            out.append(d)
    return out


def _flag(events: list, end: pd.Timestamp, delay: int = 0, first="2006-01-01") -> pd.Series:
    days = pd.date_range(first, end, freq="D")
    f = pd.Series(0.0, index=days)
    for e in events:
        a = e + delay * DAY
        f.loc[a:a + (HOLD - 1) * DAY] = 1.0
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
    return out


def _in(idx, a, b) -> np.ndarray:
    m = idx >= pd.Timestamp(a)
    return m & (idx <= pd.Timestamp(b)) if b else m


def _event_study(close: pd.Series, events: list) -> dict:
    rng = np.random.default_rng(RAND_SEED)
    fw = _fwd(close)
    ev = pd.DatetimeIndex([e for e in events if e >= pd.Timestamp(EV_FROM) and e in close.index])
    stats = {}
    for name, a, b in SPLITS + (("全部", EV_FROM, None),):
        ein, cin = _in(ev, a, b), _in(close.index, a, b)
        st = {"n": int(ein.sum())}
        for h in HORIZONS:
            evv = fw[h].reindex(ev[ein]).values.astype(float)
            evv = evv[np.isfinite(evv)]
            base = fw[h].values[cin].astype(float)
            base = base[np.isfinite(base)]
            x = {"n": len(evv), "all": float(base.mean()) if len(base) else None, "ev": None, "ex": None,
                 "up": None, "rand": None}
            if len(evv):
                x.update(ev=float(evv.mean()), ex=float(evv.mean() - base.mean()), up=float((evv > 0).mean()))
                draws = base[rng.integers(0, len(base), (RAND, len(evv)))].mean(axis=1)
                x["rand"] = float((draws >= evv.mean()).mean())
            st[h] = x
        stats[name] = st
    rows = [{"d": e, "btc": {h: fw[h].get(e, np.nan) for h in HORIZONS}} for e in ev]
    return {"stats": stats, "rows": rows}


def _xcorr(fx: pd.Series, close: pd.Series) -> dict:
    """Correlation of daily dollar returns (H.10 business days) with bitcoin over the same steps, lags -5..+5."""
    days = fx.index[fx.index.isin(close.index)]
    d = np.log(fx.reindex(days)).diff()
    b = np.log(close.reindex(days)).diff()
    frame = pd.DataFrame({"d": d.values, "b": b.values}, index=days).dropna()
    out = {}
    for name, a in (("2013 起", "2013-01-01"), ("2020 起", "2020-01-01")):
        f = frame[frame.index >= pd.Timestamp(a)]
        row = {}
        for k in XLAGS:
            # k > 0: dollar on day t against bitcoin on day t + k (dollar first)
            row[k] = float(f["d"].corr(f["b"].shift(-k)))
        out[name] = {"r": row, "n": len(f), "band": 2 / math.sqrt(len(f))}
    return out


# ---------- the exam (yen_carry's arithmetic, D1 = ma50 OR dollar flag) ----------

def _pos(kind: str, ma: np.ndarray, flag) -> np.ndarray:
    if kind == "ma50":
        return ma
    ma = np.asarray(ma, dtype=float)
    return np.where(np.isnan(ma), np.nan, np.maximum(ma, flag))


class _Exam(yc._Exam):
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
        f2 = _flag(_events(_shock(fx2)), flag.index[-1])
        idx = close.index[close.index <= c]
        total += 1
        bad += int(not np.array_equal(lf._usable(flag, idx, LAG["utc"]), lf._usable(f2, idx, LAG["utc"])))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, fx = data["close"], data["fx"]
    sig = _shock(fx)
    events = _events(sig)
    end = max(pd.Timestamp(common.today_utc()), fx.index[-1]) + 40 * DAY
    flag = _flag(events, end)
    flag7 = _flag(events, end, delay=7)
    es = _event_study(close, events)
    xc = _xcorr(fx, close)

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)
    none = pd.Series(dtype=float)
    s = {"ma50": ex.summary("ma50", none), "d1": ex.summary("d1", flag), "d1_7": ex.summary("d1", flag7)}
    pos = {"ma50": ex.btc_pos("ma50", none), "d1": ex.btc_pos("d1", flag)}

    real_small = ex.summary("d1", flag, PL_SIMS)
    part = flag[flag.index >= pd.Timestamp(PL_FROM)]
    prng = np.random.default_rng(SEED + 11)
    hits, lows = 0, []
    for sh in prng.integers(GAP, len(part) - GAP, DRAWS):
        f2 = pd.Series(np.roll(part.values, int(sh)), index=part.index)
        x = ex.summary("d1", f2, PL_SIMS)
        lows.append(x["low"])
        hits += int(lf._score_ge(x, real_small))

    idx = close.index
    a = idx >= pd.Timestamp(msc.DAILY_STARTS[0])
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["d1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()))
    years = {"hold": lf._years(np.ones(len(close)), close, msc.DAILY_STARTS[0]),
             **{k: lf._years(pos[k], close, msc.DAILY_STARTS[0]) for k in pos}}
    return {"sig": sig, "events": events, "flag": flag, "es": es, "xc": xc, "ex": ex, "s": s, "pos": pos,
            "real_small": real_small, "plc": hits / DRAWS, "plc_low": float(np.median(lows)), "differ": differ,
            "years": years, "look": _lookahead(data, flag), "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def _p(v) -> str:
    return _pct(v * 100) if _ok(v) else "—"


def _x(v) -> str:
    return f"{v * 100:+.1f}" if _ok(v) else "—"


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, fx, sig, ex, s, es = data["close"], data["fx"], r["sig"], r["ex"], r["s"], r["es"]
    st = es["stats"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    n_c = s["d1"]["c_n"]
    n_ev = st["全部"]["n"]

    # ----- pass test -----
    allp = st["全部"]
    a_pos = all(_ok(allp[h]["ex"]) and allp[h]["ex"] > 0 for h in HORIZONS)
    a_rand = _ok(allp[20]["rand"]) and allp[20]["rand"] <= 0.10
    used_splits = [n for n, _, _ in SPLITS if st[n][20]["n"] >= 3]
    a_split = bool(used_splits) and all(st[n][20]["ex"] > 0 for n in used_splits)
    ev_ok = a_pos and a_rand and a_split
    score_ok = lf._score_ge(s["d1"], s["ma50"])
    loss_ok = s["d1"]["c_deep_med"] >= s["ma50"]["c_deep_med"] - 1e-12
    plc_ok = r["plc"] <= 0.10
    rule_ok = score_ok and loss_ok and plc_ok
    if n_ev < MIN_EVENTS:
        grade = "unclear"
    elif ev_ok and rule_ok:
        grade = "pass"
    elif ev_ok or rule_ok:
        grade = "mixed"
    else:
        grade = "fail"
    gword = {"pass": "通過", "mixed": "一半一半", "unclear": "事件太少、看不出來", "fail": "沒有通過"}[grade]

    def mark(ok: bool) -> str:
        return "過" if ok else "不過"

    # ----- table 1: event study -----
    rows1 = []
    for name in [n for n, _, _ in SPLITS] + ["全部"]:
        x = st[name]
        for h in HORIZONS:
            v = x[h]
            rows1.append([name, f"之後 {h} 天", f"{v['n']}", _p(v["ev"]), _p(v["all"]), _x(v["ex"]),
                          _share(v["rand"]) if _ok(v["rand"]) else "—", _share(v["up"]) if _ok(v["up"]) else "—"])
    t1 = {"caption": (f"美元急跌事件之後比特幣的表現，和同一段期間的所有日子比。事件＝聯準會廣義美元指數 {STEP} 個營業日跌 {abs(DROP) * 100:.0f}% 以上，"
                      f"和上一次事件至少隔 {GAPDAYS} 天；{EV_FROM[:4]} 年起、而且之後的天數已經走完才算。比特幣從事件那天（紐約中午匯率）之後的第一個 "
                      f"UTC 0 點收盤算起。差距單位是百分點。「隨機一樣好」＝從同一段所有日子隨機抽同樣個數、抽 {RAND:,} 次，平均報酬不低於事件後平均的比例"
                      "（越小越不像運氣）。注意 5、10、20 天的報酬彼此重疊，不是三個獨立的證據。"),
          "columns": ["期間", "看什麼", "事件數", "事件後平均", "所有日子平均", "差距（百分點）", "隨機一樣好的比例", "事件後上漲的比例"],
          "rows": rows1}

    # ----- table 2: every event -----
    rows2 = []
    for e in es["rows"]:
        d = e["d"]
        rows2.append([_d(d), f"{sig.loc[d, 'fx']:.2f}", _p(sig.loc[d, "ch"])] + [_p(e["btc"][h]) for h in HORIZONS])
    t2 = {"caption": ("每一次美元急跌事件和比特幣之後的結果。3 日變化是廣義美元指數的跌幅；比特幣的報酬從事件後第一個 UTC 0 點收盤算起。"
                      "「—」表示時間還沒到。"),
          "columns": ["事件日（紐約）", "廣義美元指數", "3 日變化", "比特幣 5 天", "10 天", "20 天"],
          "rows": rows2}

    # ----- table 3: who moves first -----
    xc = r["xc"]
    rows3 = []
    for k in XLAGS:
        lab = ("同一天" if k == 0 else f"美元先、比特幣晚 {k} 天" if k > 0 else f"比特幣先、美元晚 {-k} 天")
        rows3.append([lab] + [f"{xc[n]['r'][k]:+.3f}" + ("＊" if abs(xc[n]["r"][k]) > xc[n]["band"] else "") for n in xc])
    rows3.append(["雜訊範圍（±2／√天數）"] + [f"±{xc[n]['band']:.3f}（{xc[n]['n']:,} 天）" for n in xc])
    t3 = {"caption": ("誰先動：廣義美元指數每天的漲跌，和比特幣同一段（同一個美國營業日到下一個）的漲跌之間的相關係數，把其中一邊往後移 1 到 5 天。"
                      "負的＝美元跌時比特幣傾向漲。打＊的是超出雜訊範圍的格。如果「美元先、比特幣晚」那幾列明顯不是 0，才算美元領先。"),
          "columns": ["位移"] + list(xc), "rows": rows3}

    # ----- table 4: the fair exam -----
    rows4 = []
    for key, label in (("ma50", "50 日均線（基準）"), ("d1", f"D1：50 日均線，或美元急跌後 {HOLD} 天也持有"),
                       ("d1_7", "D1 等 7 天才用（等每週公布，只報告）")):
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
                      f"{'、'.join(ex.other)} 起算月合計（用同一個美元訊號）。排名分數＝四項最低那項，前提是最深虧損比一直抱著淺的比例至少 "
                      f"{LOSS_BAR * 100:.0f}%。最後兩欄是課本｜一般人做法的中位數。"),
          "columns": ["規則", "比特幣起算月", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows4}

    # ----- table 5: pass test -----
    rows5 = [
        ["事件後 5、10、20 天都比所有日子好", "、".join(f"{h} 天 {_x(allp[h]['ex'])}" for h in HORIZONS), mark(a_pos)],
        ["20 天：隨機一樣好的比例 ≤ 10%", _share(allp[20]["rand"]) if _ok(allp[20]["rand"]) else "—", mark(a_rand)],
        ["20 天：事件數 ≥ 3 的每一段都比所有日子好",
         "；".join(f"{n} {_x(st[n][20]['ex'])}（{st[n][20]['n']} 次）" for n, _, _ in SPLITS), mark(a_split)],
        ["（甲）事件研究", "上面三項都要過", mark(ev_ok)],
        ["排名分數不低於 50 日均線", f"{_share(s['d1']['low'])} 對 {_share(s['ma50']['low'])}", mark(score_ok)],
        ["最深虧損中位數不比 50 日均線深", f"{_pct(s['d1']['c_deep_med'])} 對 {_pct(s['ma50']['c_deep_med'])}", mark(loss_ok)],
        [f"隨機錯位一樣好的比例 ≤ 10%（{DRAWS} 次）",
         f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['real_small']['low'])}）", mark(plc_ok)],
        ["（乙）加進均線規則", "上面三項都要過", mark(rule_ok)],
        ["評等", f"事件少於 {MIN_EVENTS} 次＝看不出來；甲乙都過＝通過；只過一個＝一半一半；都不過＝沒有通過", gword],
        ["（描述）和均線持有狀態不同的天數（2015 起）", f"{r['differ'][0]:,} 天／{r['differ'][1]:,} 天", "—"],
    ]
    t5 = {"caption": (f"事先寫好的過關標準。隨機錯位：把 {PL_FROM[:4]} 年起的「美元急跌後 {HOLD} 天」持有窗整串循環平移 {GAP} 天到「長度 − {GAP}」天、"
                      f"{DRAWS} 次，整套考試重做，一般人做法每個收盤時間模擬 {PL_SIMS} 次，真的規則也這樣算再比。"),
          "columns": ["項目", "數字", "結果"], "rows": rows5}

    # ----- table 6: years -----
    rows6 = []
    yrs = sorted(r["years"]["ma50"])
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["d1"].get(y)
        rows6.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    t6 = {"caption": ("UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，看 D1 和 50 日均線的差來自哪幾年；D1 只會比均線多持有"
                      "（均線現金時，美元急跌後 20 天也買進）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "D1", "D1 減均線（百分點）"], "rows": rows6}
    tables = [t1, t2, t3, t4, t5, t6]

    # ----- today -----
    last = close.index[-1]
    lastfx = sig.index[-1]
    past = [e for e in r["events"] if e <= last]
    open_until = past[-1] + (HOLD - 1) * DAY if past and past[-1] + (HOLD - 1) * DAY >= last else None

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"廣義美元指數（最新 {_d(lastfx)}）",
         "value": (f"{sig['fx'].iloc[-1]:.2f}；{STEP} 個營業日變化 {_p(sig['ch'].iloc[-1])}，"
                   f"{'達到' if bool(sig['shock'].iloc[-1]) else '沒有達到'}急跌門檻 {DROP * 100:.0f}%")},
        {"label": "最近一次美元急跌事件",
         "value": (f"{_d(past[-1])}" + (f"，額外持有窗開到 {_d(open_until)}" if open_until is not None else "，額外持有窗已經結束"))
         if past else "—"},
        {"label": "比特幣：50 日均線｜D1", "value": f"{word(r['pos']['ma50'][-1])}｜{word(r['pos']['d1'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述美元訊號和規則在最後一根日線收盤後的狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    parts = []
    for name, x0 in xc.items():
        # a dollar lead that supports the story = dollar falls first, bitcoin rises later = negative correlation
        lead = [k for k in XLAGS if k > 0 and abs(x0["r"][k]) > x0["band"]]
        good = [k for k in lead if x0["r"][k] < 0]
        txt = f"{name}同一天相關 {x0['r'][0]:+.3f}"
        if good:
            txt += f"，美元先動、比特幣晚 {'、'.join(str(k) for k in good)} 天是負的而且超出雜訊"
        elif lead:
            txt += (f"，美元先動、比特幣晚 {'、'.join(str(k) for k in lead)} 天雖然超出雜訊，但方向是正的"
                    "（美元漲、之後比特幣也漲，和「美元跌帶動比特幣漲」相反）")
        else:
            txt += "，美元先動、比特幣晚 1 到 5 天都在雜訊範圍內"
        parts.append(txt)
    mech = "誰先動：" + "；".join(parts) + "。兩者主要是同一天一起動（美元跌、比特幣漲），看不出美元先動、比特幣再跟上。"
    a20 = allp[20]
    verdict = (f"評等：{gword}。{EV_FROM[:4]} 年起廣義美元指數 {STEP} 天跌 {abs(DROP) * 100:.0f}% 以上只有 {n_ev} 次（"
               + "、".join(f"{n} {st[n]['n']} 次" for n, _, _ in SPLITS) + "），樣本很少。"
               f"事件後 20 天比特幣平均 {_p(a20['ev'])}，所有日子 {_p(a20['all'])}，差 {_x(a20['ex'])} 個百分點、隨機一樣好 "
               f"{_share(a20['rand']) if _ok(a20['rand']) else '—'}（5 天 {_x(allp[5]['ex'])}、10 天 {_x(allp[10]['ex'])}）；"
               f"事件研究{mark(ev_ok)}。D1（均線或美元急跌後 {HOLD} 天持有）排名分數 {_share(s['d1']['low'])} 對均線 {_share(s['ma50']['low'])}、"
               f"最深虧損中位數 {_pct(s['d1']['c_deep_med'])} 對 {_pct(s['ma50']['c_deep_med'])}、隨機錯位 {_share(r['plc'])}，"
               f"加進規則{mark(rule_ok)}。" + mech)

    tot, bad = r["look"]
    fred_txt = (f"FRED 這次對儀表板固定的 user agent 沒有回應（{_RUN['fred']}），改用聯準會自己的 H.10 整包下載（同一組數字）。"
                if _RUN["fred"] else "")
    notes = [
        ("事先寫死的設計和標準（照原文，跑之前寫好）：event = broad dollar index (FRED DTWEXBGS / H.10 JRXWTFB_N.B) 3-business-day "
         "change <= -2%, events >= 20 days apart, used from the next BTC 00:00 UTC close; event study of BTC 5/10/20-day returns vs all "
         "days since 2011 with 10,000 random same-size samples, splits 2011-16 / 2017-21 / 2022-now; rule D1 = ma50 OR held for 20 "
         "days after an event, main_signal_check's fair exam; placebo circular shift >= 180 days, 200 draws; cross-correlation lags "
         "-5..+5 (report only). Pass: (a) excess > 0 at 5/10/20 days, random share at 20 days <= 10%, 20-day excess > 0 in every "
         "split with >= 3 events; (b) D1 ranking >= ma50's, median deepest loss no worse, placebo <= 10%. Grade: pass = a and b; "
         "mixed = one; fail = neither; unclear = fewer than 8 events."),
        (f"資料：{data['where']}（廣義美元指數＝美元對 26 種主要貿易夥伴貨幣的加權匯率，2006-01 起才有日資料，這裡從 {EV_FROM[:4]} 年起算事件，"
         f"比特幣更早的價格也不可靠）。{fred_txt}"
         "常見的「美元指數 DXY」只看 6 種貨幣、歐元占一半多，波動比廣義指數大，同樣 2% 的門檻在 DXY 大概會多出不少次事件（這次沒有算）；這裡事先選了廣義指數，看到事件很少之後也沒有換。"),
        ("誰先動的表要小心時間差：廣義指數是紐約中午的匯率，比特幣日線在 UTC 0 點收盤，比紐約中午晚大約 12 小時。所以比特幣「第 D 天」"
         "裡有半天的變動，美元要到 D+1 的中午才反映出來，「比特幣先、美元晚 1 天」那格不是 0 很可能就是這個時間差，不代表比特幣真的領先。"
         f"同樣地，D1 和 50 日均線只在 {r['differ'][0]:,} 天（2015 年起共 {r['differ'][1]:,} 天）持有狀態不同，考試裡幾個百分點的差距"
         "幾乎全由這幾十天決定。"),
        ("時間對齊：廣義指數是用紐約中午的匯率算的，第 D 天的值只從比特幣下一個 UTC 0 點收盤（D+1 的 0 點，儀表板標為 D 的那根日線）起才用；"
         "換收盤時間時，落在 UTC 日期 E 的收盤只用到 E−1 的值；MSTR（美股收盤）也只用到前一天。聯準會其實每週一才公布 H.10，"
         "所以另外報告「等 7 天才用」的版本。"
         f"偷看未來的檢查：在 {'、'.join(CUTS)} 之後把美元指數換成亂數重算，切點以前的持有窗 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("事件很少、而且彼此間隔只要 20 天，5、10、20 天的報酬會重疊，平均數很容易被一兩次大事件帶著走，所以每一次都列在表上。"
         "美元急跌常發生在聯準會轉向或危機後的寬鬆（例如 2020-03 之後），這些時候很多資產一起漲，不代表是美元本身推動比特幣。"),
        ("其他資產用 main_signal_check 的清單（ETH、SOL、BNB、XRP、MSTR），全部用同一個美元訊號、各自的 50 日均線。"
         f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%，現金沒有利息、一直抱著不扣成本、沒有算稅。隨機的部分都用固定種子。"
         "相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "dollar_shock",
        "title": "美元急跌後比特幣會不會多漲",
        "question": (f"美元突然大跌（廣義美元指數 {STEP} 天跌 {abs(DROP) * 100:.0f}% 以上）之後 5、10、20 天，比特幣是不是比平常漲得多？"
                     f"把「美元急跌後 {HOLD} 天也持有」加進 50 日均線規則，能不能比單純的均線好？是美元先動、比特幣再跟上嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "FRED：Nominal Broad U.S. Dollar Index（DTWEXBGS，日資料）", "url": FRED_URL},
            {"name": "美國聯準會 H.10 匯率（Data Download Program 整包下載；廣義美元指數日資料 JRXWTFB_N.B）", "url": FED_URL},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
