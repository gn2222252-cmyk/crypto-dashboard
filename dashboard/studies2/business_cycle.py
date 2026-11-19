"""Historical study "business_cycle": does the US manufacturing business cycle tell bitcoin's good months from its bad
ones?

The claim (SentimenTrader, March 2025): at each month-end, if the latest ISM manufacturing PMI is above its 10-month
average, the next month is "favorable" for bitcoin; since July 2011 favorable months produced essentially all of
bitcoin's gains and unfavorable months about 0% net. Raoul Pal / Real Vision tell a related "the ISM business cycle
drives bitcoin" story. ISM's own history is not free and key-less, so free public substitutes are used.

Pre-registered (written before any result of this study was seen; no other variants):
  Data: Philadelphia Fed Manufacturing Business Outlook Survey, current general activity diffusion index, seasonally
  adjusted (FRED GACDFSA066MSFRBPHI; since 1968) = MAIN series. Robustness: New York Fed Empire State general
  business conditions, seasonally adjusted (FRED GACDISA066MSFRBNY; since 2001-07) and the OECD composite leading
  indicator for the US, amplitude adjusted (FRED USALOLITOAASTSAM). FRED's key-less CSV is tried first (fixed user
  agent, timeouts, retries); if FRED does not answer, the same series are taken from the publishers' own key-less
  files (Philadelphia Fed bos_history.csv, New York Fed ESMS_SeasonallyAdjusted_Diffusion.csv, OECD SDMX CSV) and
  this is reported. A good download is kept in common.STORE; a failed one falls back to the stored copy.
  Point in time: the decision is made at the last daily close of month M (00:00 UTC close, te._daily_close, no extra
  shift) and held during month M+1. Usable value at that close: Philly and Empire = the value FOR month M (published
  the 3rd Thursday of M / around the 15th of M, i.e. at least a week before the month-end); OECD CLI = the value for
  month M-2 (published around the 10th of M+1 for month M; two months is conservative). Revisions: seasonal factors
  of both surveys are re-estimated every year and the OECD CLI is revised every month; ALFRED vintages are tried
  key-less, otherwise revisions are a stated caveat and the main series is re-run with ONE EXTRA MONTH of lag
  (sensitivity) and on the Philly NOT seasonally adjusted index (which seasonal re-estimation never changes).
  M1 = SentimenTrader's rule with each substitute: favorable when the usable value > the average of the last 10
  usable monthly values (including itself); hold BTC during the next month when favorable, else cash.
  M2 = close above its 50-day average (main_signal_check's ma50) AND M1 favorable.
  Event table: average next-month BTC return (month-end close to month-end close) in favorable vs unfavorable months,
  holding months from 2011-07, difference with a block bootstrap (blocks of 6 consecutive months, 2,000 draws, 90%
  interval); splits by holding month 2011-16 / 2017-21 / 2022-now; compounded growth in each group.
  Fair exam: main_signal_check's four checks re-used as ls_ratio / lead_followup do - BTC daily 108 monthly starts
  2015-01 ... 2023-12; 72 monthly starts x 24 closing hours (1,728 cells); ordinary person (trade 10 hours later,
  0.25% cost, 10% of entries missed, 200 simulations per clock, main_signal_check's seeds); other coins ETH, SOL, BNB,
  XRP, MSTR with their own ma50 and the SAME macro signal. Ranking score = lowest of the four "beats holding" shares
  (eligible when the deepest loss is shallower than holding's in >= 95% of the BTC starts and cells). Placebo: the
  monthly signal circularly shifted by 6 ... (length - 6) months, 200 draws, whole exam rebuilt (ordinary person with
  50 simulations per clock; the real rule re-scored the same way), share of draws ranking at least as high.
  Median deepest loss = median over the 1,728 textbook cells.
  Pass test (main series, for M1 and M2 separately): favorable > unfavorable in all three periods AND ranking score
  >= ma50's on the same starts AND median deepest loss no worse than ma50's AND placebo share <= 10%.
  Empire, OECD, the extra month of lag and the unadjusted Philly index are robustness only (event table and exam,
  no placebo). Today: states only.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, LOSS_BAR = msc.YEAR, msc.LOSS_BAR
DAY = pd.Timedelta(days=1)
AVG = 10
EVENT_FROM = pd.Period("2011-07", "M")      # first holding month of the event table
PERIODS = (("2011–16", 2011, 2016), ("2017–21", 2017, 2021), ("2022–今", 2022, 2100))
BLOCK, BOOT = 6, 2000
DRAWS, GAP, PL_SIMS = 200, 6, 50
KI = {"ma50": 1, "m1": 90, "m2": 91}  # ma50 = main_signal_check's draws

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ALFRED_CSV = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={}"
PHILLY_URL = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/MBOS/Historical-Data/Data-Series/"
              "bos_history.csv?sc_lang=en")
EMPIRE_URL = "https://www.newyorkfed.org/medialibrary/media/survey/empire/data/ESMS_SeasonallyAdjusted_Diffusion.csv"
OECD_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/USA.M.LI...AA...H"
            "?format=csvfilewithlabels")
FRED_ID = {"philly": "GACDFSA066MSFRBPHI", "philly_nsa": "GACDFNA066MSFRBPHI", "empire": "GACDISA066MSFRBNY",
           "oecd": "USALOLITOAASTSAM"}
LAG = {"philly": 0, "philly_nsa": 0, "empire": 0, "oecd": 2}  # value of month m usable at the end of month m + LAG
SNAME = {"philly": "費城聯準銀行製造業調查（季調）", "philly_nsa": "費城聯準銀行製造業調查（未季調）",
         "empire": "紐約州帝國製造業調查（季調）", "oecd": "OECD 美國領先指標"}
SHORT = {"philly": "費城", "philly_nsa": "費城未季調", "empire": "紐約帝國", "oecd": "OECD 領先"}
LAGTXT = {"philly": "當月數字（當月第三個星期四公布）", "philly_nsa": "當月數字（同上）",
          "empire": "當月數字（當月 15 日左右公布）", "oecd": "兩個月前的數字（保守估計）"}

_pct, _pp, _d, _num = cr._pct, cr._pp, cr._d, te._num


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _share(v) -> str:
    return f"{v * 100:.0f}%" if _ok(v) else "—"


def _p(v, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%" if _ok(v) else "—"


# ---------- data ----------

_RUN: dict = {"fred_down": None, "philly": None}


def _get(url: str, timeout=(10, 40), tries: int = 2) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            reply = requests.get(url, headers=common.HEADERS, timeout=timeout)
            reply.raise_for_status()
            time.sleep(common.PAUSE)
            return reply
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last


def _fred(sid: str) -> dict:
    frame = pd.read_csv(io.StringIO(_get(FRED_CSV.format(sid), timeout=(8, 25), tries=2).text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return {str(day)[:7]: float(v) for day, v in zip(frame.iloc[:, 0], values) if math.isfinite(v)}


def _philly_file() -> pd.DataFrame:
    if _RUN["philly"] is None:
        frame = pd.read_csv(io.StringIO(_get(PHILLY_URL).text))
        when = pd.to_datetime(frame["DATE"], format="%b-%y")
        when = when.where(when.dt.year <= pd.Timestamp(common.today_utc()).year, when - pd.DateOffset(years=100))
        frame.index = when.dt.strftime("%Y-%m")
        _RUN["philly"] = frame
    return _RUN["philly"]


def _pub_philly(col: str) -> dict:
    s = pd.to_numeric(_philly_file()[col], errors="coerce")
    return {k: float(v) for k, v in s.items() if math.isfinite(v)}


def _pub_empire() -> dict:
    frame = pd.read_csv(io.StringIO(_get(EMPIRE_URL).text))
    s = pd.to_numeric(frame["GACDISA"], errors="coerce")
    return {str(day)[:7]: float(v) for day, v in zip(frame["surveyDate"], s) if math.isfinite(v)}


def _pub_oecd() -> dict:
    frame = pd.read_csv(io.StringIO(_get(OECD_URL, timeout=(10, 60)).text))
    s = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    return {str(day)[:7]: float(v) for day, v in zip(frame["TIME_PERIOD"], s) if math.isfinite(v)}


PUBLISHER = {"philly": lambda: _pub_philly("gacdfsa"), "philly_nsa": lambda: _pub_philly("gacdfna"),
             "empire": _pub_empire, "oecd": _pub_oecd}
PUBNAME = {"philly": "費城聯準銀行 bos_history.csv", "philly_nsa": "費城聯準銀行 bos_history.csv",
           "empire": "紐約聯準銀行 ESMS_SeasonallyAdjusted_Diffusion.csv", "oecd": "OECD SDMX 資料庫"}


def _series(name: str) -> tuple[pd.Series, str, str]:
    """Monthly values (PeriodIndex) from FRED, else the publisher, else the stored copy. Returns (series, where, note)."""
    path = common.STORE / f"business_cycle_{name}.json"
    note, where = "", ""
    try:
        days = None
        if not _RUN["fred_down"]:
            try:
                days, where = _fred(FRED_ID[name]), "FRED"
                if len(days) < 120:
                    raise RuntimeError(f"只回傳 {len(days)} 筆")
            except Exception as exc:  # noqa: BLE001 - FRED often does not answer; the publishers' files are the same data
                _RUN["fred_down"] = type(exc).__name__
                days = None
        if days is None:
            days, where = PUBLISHER[name](), PUBNAME[name]
        if len(days) < 120:
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
        where = "上次存下來的資料"
        note = f"{SNAME[name]}這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）。"
    s = pd.Series({pd.Period(k, "M"): float(v) for k, v in days.items()}).sort_index()
    s = s.reindex(pd.period_range(s.index[0], s.index[-1], freq="M"))
    return s, where, note


def _alfred_check() -> str:
    try:
        reply = _get(ALFRED_CSV.format(FRED_ID["philly"]), timeout=(8, 25), tries=1)
        return "通了" if reply.text.startswith(("DATE", "observation_date")) else "回應格式不對"
    except Exception as exc:  # noqa: BLE001
        return f"沒有回應（{type(exc).__name__}）"


def _load() -> dict:
    problems, series, where = [], {}, {}
    for name in ("philly", "philly_nsa", "empire", "oecd"):
        try:
            s, w, n = _series(name)
            series[name], where[name] = s, w
            if n:
                problems.append(n)
        except Exception as exc:  # noqa: BLE001
            if name == "philly":
                raise
            problems.append(f"{SNAME[name]}這次取不到（{type(exc).__name__}），沒有算進去。")
    other, more = msc._load_other()
    problems += more
    alfred = _alfred_check() if _RUN["fred_down"] is None else "沒有試（FRED 本身沒有回應）"
    return {"close": te._daily_close(), "series": series, "where": where, "other": other, "problems": problems,
            "fred_down": _RUN["fred_down"], "alfred": alfred}


# ---------- the signal ----------

def _flags(s: pd.Series, lag: int) -> pd.Series:
    """Index: decision month M (decided at the last close of M, held during M+1); 1 = favorable, NaN = unknown."""
    avg = s.rolling(AVG, min_periods=AVG).mean()
    fav = (s > avg).astype(float).where(avg.notna() & s.notna())
    fav.index = fav.index + lag
    return fav


def _on_days(F: pd.Series, dates) -> np.ndarray:
    """State in effect on each calendar day: the decision made at the end of the previous month (unknown = cash)."""
    per = pd.DatetimeIndex(dates).to_period("M") - 1
    return F.reindex(per).fillna(0.0).values.astype(float)


def _pos(kind: str, ma: np.ndarray, macro) -> np.ndarray:
    if kind == "ma50":
        return ma
    if kind == "m1":
        return np.asarray(macro, dtype=float) * np.ones_like(ma)
    return ma * macro


# ---------- event table ----------

def _month_ends(close: pd.Series) -> pd.Series:
    """Last close of every finished month (PeriodIndex)."""
    g = close.groupby(close.index.to_period("M"))
    last = g.apply(lambda x: x.index[-1])
    ends = g.last()[[d.day == d.days_in_month for d in last]]
    return ends


def _boot_diff(r: np.ndarray, f: np.ndarray, rng) -> tuple[float, float] | None:
    n = len(r)
    if n < 2 * BLOCK or f.sum() < 3 or (1 - f).sum() < 3:
        return None
    nb = -(-n // BLOCK)
    st = rng.integers(0, n - BLOCK + 1, size=(BOOT, nb))
    idx = (st[:, :, None] + np.arange(BLOCK)).reshape(BOOT, -1)[:, :n]
    rr, ff = r[idx], f[idx]
    nf, nu = ff.sum(axis=1), (1 - ff).sum(axis=1)
    ok = (nf > 0) & (nu > 0)
    d = (rr * ff).sum(axis=1)[ok] / nf[ok] - (rr * (1 - ff)).sum(axis=1)[ok] / nu[ok]
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


def _events(close: pd.Series, F: pd.Series, rng) -> dict:
    me = _month_ends(close)
    ret = (me / me.shift(1) - 1).dropna()
    ret = ret[ret.index >= EVENT_FROM]
    f = F.reindex(ret.index - 1)
    keep = f.notna().values
    r, fl, idx = ret.values[keep], f.values[keep], ret.index[keep]

    def part(mask) -> dict:
        x, y = r[mask], fl[mask]
        fav, unf = x[y == 1], x[y == 0]
        return {"nf": len(fav), "nu": len(unf), "mf": float(fav.mean()) if len(fav) else None,
                "mu": float(unf.mean()) if len(unf) else None,
                "diff": float(fav.mean() - unf.mean()) if len(fav) and len(unf) else None,
                "gf": float(np.prod(1 + fav) - 1) if len(fav) else None,
                "gu": float(np.prod(1 + unf) - 1) if len(unf) else None}

    out = {"all": part(np.ones(len(r), bool)), "first": idx[0] if len(idx) else None,
           "last": idx[-1] if len(idx) else None}
    out["all"]["ci"] = _boot_diff(r, fl, rng)
    out["periods"] = {lab: part((idx.year >= a) & (idx.year <= b)) for lab, a, b in PERIODS}
    out["all_three"] = all(_ok(p["diff"]) and p["diff"] > 0 for p in out["periods"].values())
    out["fav_share"] = float(fl.mean()) if len(fl) else None
    return out


# ---------- the exam (main_signal_check's four checks) ----------

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
            starts, labels = msc._month_starts(c.index, begin, three)
            if not len(starts):
                continue
            change = c.pct_change().fillna(0.0).values
            hr, hd = msc._suffix(msc._equity(change), starts, year)
            self.other[key] = {"close": c, "ma": tg["ma50"].values.astype(float), "change": change, "starts": starts,
                               "labels": labels, "year": year, "hold": (hr[:, 0], hd[:, 0])}

    def btc_pos(self, kind: str, F: pd.Series | None) -> np.ndarray:
        macro = _on_days(F, self.close.index + DAY) if kind != "ma50" else 0.0
        return _pos(kind, self.btc_ma, macro)

    def summary(self, kind: str, F: pd.Series | None, sims: int = SIMS) -> dict:
        pos = self.btc_pos(kind, F)
        net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), self.btc_change, COST)
        dr, dd = msc._suffix(msc._equity(net), self.dstarts, YEAR)
        dr, dd = dr[:, 0], dd[:, 0]
        dh = self.dhold
        cmac = _on_days(F, self.m.dates) if kind != "ma50" else 0.0
        n = len(self.cstarts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr, pdd = np.zeros((n, 24, sims)), np.zeros((n, 24, sims))
        for h in range(24):
            w = np.nan_to_num(_pos(kind, self.cma[h], cmac), nan=0.0).astype(float)
            r, d_ = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.cstarts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
            b, wl = rt._later(h, w, self.days)
            rng = np.random.default_rng(SEED + 1000 * KI[kind] + h)
            p = rt._missed(wl, P_MISS, rng, sims)
            r, d_ = msc._suffix(np.vstack([np.ones((1, sims)), self.eng._mid(b, p, P_COST, None)]), self.cstarts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, d_
        hr, hd = self.chold
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        each = {}
        for a, o in self.other.items():
            macro = _on_days(F, o["close"].index + DAY) if kind != "ma50" else 0.0
            p = _pos(kind, o["ma"], macro)
            net, _ = te._run(np.concatenate([[np.nan], p[:-1]]), o["change"], COST)
            r, d_ = msc._suffix(msc._equity(net), o["starts"], o["year"])
            each[a] = (len(o["starts"]), int((r[:, 0] > o["hold"][0]).sum()), int((d_[:, 0] > o["hold"][1]).sum()),
                       float(np.median(r[:, 0] - o["hold"][0])))
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


def _score_ge(x: dict, base: dict) -> bool:
    if base["eligible"] and not x["eligible"]:
        return False
    if x["eligible"] and not base["eligible"]:
        return True
    return x["low"] >= base["low"] - 1e-12


def _years(pos: np.ndarray, close: pd.Series, first: str) -> dict:
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), close.pct_change().fillna(0.0).values, COST)
    s = pd.Series(net, index=close.index)
    s = s[s.index >= pd.Timestamp(first)]
    return {int(y): float((np.prod(1 + g.values) - 1) * 100) for y, g in s.groupby(s.index.year)}


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: _Exam) -> tuple[int, int]:
    """Scramble what is not yet usable at three month-ends; no M2 position decided up to then may change."""
    rng = np.random.default_rng(13)
    close = data["close"]
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        upto = close.index <= c
        for name, s in data["series"].items():
            known = s.index <= c.to_period("M") - LAG[name]
            s2 = s.where(known, s + rng.normal(0, 25, len(s)))
            a = _pos("m2", ex.btc_ma, _on_days(_flags(s, LAG[name]), close.index + DAY))
            b = _pos("m2", ma2, _on_days(_flags(s2, LAG[name]), close.index + DAY))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, series = data["close"], data["series"]
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = _Exam(data, price, b0, pre)

    variants = {}  # key -> (series name, lag)
    for name in series:
        variants[name] = (name, LAG[name])
    variants["philly_lag"] = ("philly", LAG["philly"] + 1)
    F = {k: _flags(series[n], lag) for k, (n, lag) in variants.items()}

    rng = np.random.default_rng(SEED + 5)
    ev = {k: _events(close, F[k], rng) for k in F}

    s = {"ma50": ex.summary("ma50", None)}
    for k in F:
        if k == "philly_nsa":
            continue
        for kind in ("m1", "m2"):
            s[f"{kind}_{k}"] = ex.summary(kind, F[k])

    # placebo on the main series
    real_small = {kind: ex.summary(kind, F["philly"], PL_SIMS) for kind in ("m1", "m2")}
    raw = series["philly"]
    n = len(raw)
    plc, plc_low = {}, {}
    for kind, seed in (("m1", SEED + 21), ("m2", SEED + 22)):
        prng = np.random.default_rng(seed)
        hits, lows = 0, []
        for sh in prng.integers(GAP, n - GAP + 1, DRAWS):
            s2 = pd.Series(np.roll(raw.values, int(sh)), index=raw.index)
            x = ex.summary(kind, _flags(s2, LAG["philly"]), PL_SIMS)
            lows.append(x["low"])
            hits += int(_score_ge(x, real_small[kind]))
        plc[kind], plc_low[kind] = hits / DRAWS, float(np.median(lows))

    pos = {"ma50": ex.btc_pos("ma50", None), "m1": ex.btc_pos("m1", F["philly"]), "m2": ex.btc_pos("m2", F["philly"])}
    first = msc.DAILY_STARTS[0]
    years = {"hold": _years(np.ones(len(close)), close, first), **{k: _years(pos[k], close, first) for k in pos}}
    look = _lookahead(data, ex)
    return {"ex": ex, "F": F, "variants": variants, "ev": ev, "s": s, "real_small": real_small, "plc": plc,
            "plc_low": plc_low, "pos": pos, "years": years, "look": look, "problem": problem, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base = s["ma50"]
    n_c, n_d = base["c_n"], base["d_n"]

    # ----- pass test -----
    res = {}
    for kind in ("m1", "m2"):
        x = s[f"{kind}_philly"]
        q = {"event": ev["philly"]["all_three"], "score": _score_ge(x, base),
             "loss": x["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["plc"][kind] <= 0.10}
        q["pass"] = all(q.values())
        res[kind] = q

    # ----- table 1: event table -----
    order = [k for k in ("philly", "empire", "oecd", "philly_lag", "philly_nsa") if k in ev]

    def vname(k: str) -> str:
        if k == "philly_lag":
            return f"{SNAME['philly']}，多等一個月（敏感度）"
        return SNAME[k] + ("（主要）" if k == "philly" else "")

    def vlag(k: str) -> str:
        return "上個月的數字（比主要設定多等一個月）" if k == "philly_lag" else LAGTXT[k]

    rows1 = []
    for k in order:
        e, a = ev[k], ev[k]["all"]
        ci = a.get("ci")
        rows1.append([vname(k), vlag(k), f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()]
                     + [f"{_p(a['gf'], 0)}｜{_p(a['gu'], 0)}", "是" if e["all_three"] else "否"])
    e0 = ev["philly"]
    t1 = {"caption": (f"每個月最後一根日線（UTC 0 點收盤）決定，指標的最新可用數字高於它自己最近 {AVG} 個月的平均就算「有利」，下一個月持有比特幣。"
                      f"每格是下一個月比特幣報酬（月底收盤到月底收盤）的平均；持有月份從 {e0['first']} 到 {e0['last']}。差距＝有利減不利，"
                      f"括號是區塊拔靴法（每次連續 {BLOCK} 個月一塊、{BOOT:,} 次）的 90% 區間。分段欄括號是（有利｜不利）的月數。"
                      "最後一欄前是各組月份連乘起來的總報酬（有利｜不利）。敏感度兩列：多等一個月（防數字被修正、公布延後），"
                      "未季調版本（季節調整每年重算，未季調數字不受影響）。"),
          "columns": ["指標", "月底時用哪個月的數字", "有利月：個數｜平均報酬", "不利月：個數｜平均報酬", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in PERIODS] + ["連乘總報酬：有利｜不利", "三段都是有利較好"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    def rname(key: str) -> str:
        if key == "ma50":
            return "50 日均線（基準）"
        kind, v = key.split("_", 1)
        head = "M1：只看景氣" if kind == "m1" else "M2：50 日均線而且景氣有利"
        return f"{head}（{'費城，多等一個月' if v == 'philly_lag' else SHORT[v]}）"

    keys = ["ma50"] + [f"{kind}_{v}" for v in ("philly", "empire", "oecd", "philly_lag") for kind in ("m1", "m2")
                       if f"{kind}_{v}" in s]
    rows2 = []
    for key in keys:
        x = s[key]
        ck = x["checks"]
        rows2.append([rname(key), f"{_share(ck[0])}", _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    t2 = {"caption": (f"main_signal_check 的公平考試，四項都是「報酬贏過同一天起一直抱著」的比例：比特幣 UTC 0 點日線 "
                      f"{msc.DAILY_STARTS[0][:7]} 到 {msc.DAILY_STARTS[1][:7]} 每月 1 日起算（{n_d} 個）；{len(ex.cstarts)} 個起算月 × 24 個收盤時間 "
                      f"＝ {n_c:,} 格照課本做（成本 {COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做（收盤後 10 小時才做、成本 "
                      f"{P_COST * 100:.2f}%、一成進場漏掉，每格 {SIMS} 次，亂數種子同 main_signal_check）；其他資產（ETH、SOL、BNB、XRP、MSTR）"
                      "用自己的 50 日均線、同一個美國景氣訊號。排名分數＝四項裡最低的那項；前提是最深虧損比一直抱著淺的比例（比特幣兩組裡較低的）"
                      f"至少 {LOSS_BAR * 100:.0f}%。最後兩欄是 {n_c:,} 格照課本｜一般人做法的中位數；每年多賺是比一直抱著多幾個百分點。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: other assets -----
    rows3 = []
    for a, o in ex.other.items():
        cells = []
        for key in ("ma50", "m1_philly", "m2_philly"):
            v = s[key]["o_each"].get(a)
            cells.append(f"{v[1]} 個｜{_pp(v[3])}｜{v[2]} 個" if v else "—")
        rows3.append([a, f"{_d(o['labels'][0])[:7]} 到 {_d(o['labels'][-1])[:7]}（{len(o['starts'])} 個）"] + cells)
    rows3.append(["合計", f"{base['o_n']} 個"] + [f"{s[k]['o_beat']} 個" for k in ("ma50", "m1_philly", "m2_philly")])
    t3 = {"caption": ("第四項明細：規則原封不動用在其他資產（幣安現貨日線 UTC 0 點收盤；MSTR 用 Yahoo 調整後收盤、2020-08-11 起、一年 252 天），"
                      "起算月和 main_signal_check 一樣（到三年前），照課本做、成本 0.10%；景氣訊號都是費城調查（美國整體景氣，不分資產）。"
                      "各格是「報酬贏過一直抱著的起算月個數｜每年多賺中位數（百分點）｜最深虧損比抱著淺的個數」。"),
          "columns": ["資產", "起算月", "50 日均線", "M1 只看景氣（費城）", "M2 均線＋景氣（費城）"], "rows": rows3}

    # ----- table 4: pass test -----
    rows4 = []
    for kind in ("m1", "m2"):
        x, q = s[f"{kind}_philly"], res[kind]
        ps = [p["diff"] for p in e0["periods"].values()]
        rows4.append([rname(f"{kind}_philly"),
                      "、".join(_p(v) for v in ps) + f" → {'過' if q['event'] else '不過'}",
                      f"{_share(x['low'])} 對 {_share(base['low'])} → {'過' if q['score'] else '不過'}",
                      f"{_pct(x['c_deep_med'])} 對 {_pct(base['c_deep_med'])} → {'過' if q['loss'] else '不過'}",
                      f"{_share(r['plc'][kind])}（打亂後排名分數中位數 {_share(r['plc_low'][kind])}，真的 "
                      f"{_share(r['real_small'][kind]['low'])}） → {'過' if q['plc'] else '不過'}",
                      "通過" if q["pass"] else "沒有通過"])
    t4 = {"caption": ("事先寫好的過關標準（只用主要的費城調查）：有利月的下月平均報酬在 2011–16、2017–21、2022–今三段都比不利月高；"
                      f"排名分數不低於 50 日均線；{n_c:,} 格最深虧損中位數不比 50 日均線深；把每月訊號整串循環平移 {GAP} 個月到「長度 − {GAP}」個月、"
                      f"{DRAWS} 次、整套考試重做，排名分數和真的一樣好或更好的比例 ≤ 10%（這一步一般人做法每個收盤時間只模擬 {PL_SIMS} 次，"
                      "真的規則也用同樣方式算一次再比）。四項都要過。"),
          "columns": ["規則", "三段差距（有利−不利）", "排名分數：規則 對 均線", "最深虧損中位數：規則 對 均線", "隨機錯位一樣好", "結果"],
          "rows": rows4}

    # ----- table 5: calendar years -----
    rows5 = []
    yrs = sorted(r["years"]["ma50"])
    for y in yrs:
        rows5.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y))]
                     + [_pct(r["years"][k].get(y)) for k in ("ma50", "m1", "m2")])
    t5 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{msc.DAILY_STARTS[0][:4]} 年起；景氣訊號用費城調查。"
                      "看規則和一直抱著、50 日均線的差來自哪幾年。"),
          "columns": ["年份", "一直抱著", "50 日均線", "M1 只看景氣", "M2 均線＋景氣"], "rows": rows5}
    tables = [t1, t2, t3, t4, t5]

    # ----- today -----
    last = close.index[-1]
    cur = (last + DAY).to_period("M") - 1   # decision month whose state is in effect now
    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
            "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"}]
    for name in ("philly", "empire", "oecd"):
        if name not in data["series"]:
            continue
        raw = data["series"][name].dropna()
        avg = raw.rolling(AVG, min_periods=AVG).mean()
        fnow = F[name].get(cur, np.nan)
        lastm = raw.index[-1]
        above = raw.iloc[-1] > avg.iloc[-1]
        now.append({"label": f"{SNAME[name]}（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
                    "value": (f"{'有利' if fnow == 1 else '不利' if fnow == 0 else '—'}；最新公布的是 {lastm.strftime('%Y-%m')} 的 "
                              f"{_num(raw.iloc[-1], 1)}，近 {AVG} 個月平均 {_num(avg.iloc[-1], 1)}（這個最新數字{'高於' if above else '不高於'}平均；"
                              f"下一次決定還要看月底時能用的是哪個月）")})
    pm = r["pos"]
    now.append({"label": "比特幣：50 日均線｜M1 只看景氣｜M2 均線＋景氣（費城）",
                "value": "｜".join("持有" if v >= 0.5 else "現金" for v in (pm["ma50"][-1], pm["m1"][-1], pm["m2"][-1]))})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    a0 = e0["all"]
    m1, m2 = s["m1_philly"], s["m2_philly"]
    others = [k for k in ("empire", "oecd") if k in ev]
    rob = "、".join(f"{SHORT[k]} {_p(ev[k]['all']['diff'])}" for k in others)
    ptxt = "、".join(f"{lab} {_p(p['diff'])}" for lab, p in zip([x[0] for x in PERIODS], e0["periods"].values()))
    verdict = (f"{'通過' if res['m1']['pass'] or res['m2']['pass'] else '沒有通過'}事先寫好的標準。用費城聯準銀行製造業調查代替 ISM，"
               f"有利月下個月比特幣平均 {_p(a0['mf'])}、不利月 {_p(a0['mu'])}，差 {_p(a0['diff'])}"
               + (f"（90% 區間 {_p(a0['ci'][0])} 到 {_p(a0['ci'][1])}）" if a0.get("ci") else "")
               + f"；分三段 {ptxt}；換紐約、OECD：{rob}。"
               f"公平考試：只看景氣（M1）排名分數 {_share(m1['low'])}、均線＋景氣（M2）{_share(m2['low'])}，50 日均線 {_share(base['low'])}；"
               f"最深虧損中位數 M1 {_pct(m1['c_deep_med'])}、M2 {_pct(m2['c_deep_med'])}、均線 {_pct(base['c_deep_med'])}；"
               f"隨機錯位 M1 {_share(r['plc']['m1'])}、M2 {_share(r['plc']['m2'])}。")
    lagd = ev["philly_lag"]["all"]["diff"]
    if _ok(lagd) and _ok(a0["diff"]) and lagd > a0["diff"]:
        neg = [f"{SHORT[k]} {lab}" for k in ("philly", "empire", "oecd") if k in ev
               for lab, p in ev[k]["periods"].items() if _ok(p["diff"]) and p["diff"] < 0]
        verdict += (f"同一份調查多等一個月（資訊更舊）差距反而變大（{_p(lagd)}）"
                    + (f"，而且{'、'.join(neg)}是不利月較好" if neg else "") + "，看起來不像穩定的關係。")
    if "m2_oecd" in s:
        o2 = s["m2_oecd"]
        verdict += (f"OECD 領先指標加均線的排名分數 {_share(o2['low'])}（均線 {_share(base['low'])}），只是穩健度檢查、沒有做隨機錯位，"
                    "是看了結果才注意到的，只能當之後要事先登記再測的線索。")

    # ----- notes -----
    tot, bad = r["look"]
    src = "、".join(f"{SHORT[k]}：{v}" for k, v in data["where"].items())
    fred_txt = (f"FRED 的公開 CSV 這次對儀表板固定的 user agent 沒有回應（{data['fred_down']}），所以改用發布機構自己的公開檔案（FRED 也是轉載它們）。"
                if data["fred_down"] else "FRED 的公開 CSV 這次有回應。")
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好）：M1 = favorable when the usable value > the average of the last 10 usable "
         "monthly values; hold BTC during the next month when favorable, else cash. M2 = ma50 AND M1 favorable. Main series = "
         "Philadelphia Fed general activity (SA); Empire State, OECD CLI, one extra month of lag and the unadjusted Philly index "
         "are robustness only. Pass test (main series, M1 and M2 separately): favorable > unfavorable in 2011-16, 2017-21 and "
         "2022-now AND ranking score >= ma50's AND median deepest loss no worse AND placebo share (circular shift >= 6 months, "
         "200 draws) <= 10%."),
        ("說法原文：SentimenTrader 用 ISM 製造業 PMI 高於 10 個月平均當「有利」，說 2011 年 7 月以來比特幣幾乎所有漲幅都在有利月、不利月淨報酬約 0%。"
         "ISM 的完整歷史不是免費、免金鑰的，所以這裡用三個公開的替代：費城、紐約兩家聯準銀行的製造業調查（和 ISM 一樣是「比上個月好的家數減差的家數」"
         "這類擴散指數，走勢和 ISM 很接近但不相同），以及 OECD 的美國領先指標。結果只能說明「替代指標」怎麼樣，不能直接當成 ISM 本身的檢驗。"),
        (f"資料來源：{src}。{fred_txt}ALFRED 歷史版本：{data['alfred']}，所以沒有辦法逐期核對「當時公布的數字」；"
         "兩份調查的季節調整每年重算、OECD 領先指標每個月都會修正前面的數字，這裡用的是今天的版本。為了估計修正的影響，多做了兩個敏感度："
         "主要指標多等一個月，以及不受季節調整重算影響的費城未季調版本。"),
        ("時間對齊：每個月最後一根日線（UTC 0 點收盤）決定、下一個月整個月照這個狀態。費城調查在當月第三個星期四、紐約調查在當月 15 日左右公布，"
         "所以月底時當月數字已經公布至少一週；OECD 領先指標大約在下個月 10 日前後公布上個月，這裡保守地只用兩個月前的數字。"
         "換收盤時間時，新的一個月第一天在那個收盤時間換狀態；MSTR 用月底前最後一個交易日收盤決定。"
         f"偷看未來的檢查：在 2016-06、2020-03、2024-09 三個月底之後，把價格和「那時還不能用」的指標數字換成亂數再算，"
         f"切點以前的 M2 持有狀態 {tot} 組裡 {tot - bad} 組完全沒變" + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        (f"50 日均線用 main_signal_check 自己的 _targets 和同樣的亂數種子重算，所以 {n_c:,} 格和一般人做法的數字就是它那份研究的 50 日均線。"
         "M1 只看景氣、不看價格，一個月才換一次狀態；其他幣、MSTR 用的是同一個美國景氣訊號，所以第四項其實是在問「美國景氣能不能幫其他幣」。"),
        ("有利月的比例和月份數：費城 " + _share(e0["fav_share"]) + "；"
         + "；".join(f"{SHORT[k]} {_share(ev[k]['fav_share'])}" for k in others) + "。"
         "相鄰月份的景氣狀態高度相關（一段一段地出現），比特幣又有少數幾個月漲幅極大，所以平均差距很容易被幾個月左右；區間用一次抽連續 6 個月的拔靴法，"
         "仍然只是粗略的範圍。三段之中最早的 2011–16 年比特幣價格很小、波動極大。"),
        ("敏感度怎麼讀：未季調版本和「高於 10 個月平均」的比較裡混著每年固定的季節起伏，所以它的「有利月」有一部分只是日曆上的固定月份，"
         "不能當成景氣訊號比較好的證據；多等一個月的版本比主要設定更好，代表結果對時間點很敏感，而不是「越早知道越有用」。"),
        ("相鄰起算日共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"
         f"成本：照課本 {COST * 100:.2f}%、一般人 {P_COST * 100:.2f}%，現金沒有利息、一直抱著不扣成本、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "business_cycle",
        "title": "美國製造業景氣能不能挑出比特幣的好月份",
        "question": ("有人說：每個月底看製造業景氣指標，高於自己 10 個月平均，下個月就對比特幣「有利」，比特幣的漲幅幾乎都發生在有利月。"
                     "用免費的費城、紐約聯準銀行製造業調查和 OECD 領先指標代替 ISM，照當時真的拿得到的數字做，這個說法站得住嗎？"
                     "和 50 日均線一起用，能不能通過完整的公平考試？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "SentimenTrader：Using manufacturing data as an input for Bitcoin（2025-03）",
             "url": "https://sentimentrader.com/blog/using-manufacturing-data-as-an-input-for-bitcoin"},
            {"name": "FRED：費城製造業調查目前景氣擴散指數（季調）", "url": FRED_CSV.format(FRED_ID["philly"])},
            {"name": "FRED：紐約帝國製造業調查目前景氣（季調）", "url": FRED_CSV.format(FRED_ID["empire"])},
            {"name": "FRED：OECD 美國綜合領先指標", "url": FRED_CSV.format(FRED_ID["oecd"])},
            {"name": "費城聯準銀行 Manufacturing Business Outlook Survey 歷史資料", "url": PHILLY_URL},
            {"name": "紐約聯準銀行 Empire State Manufacturing Survey 季調擴散指數", "url": EMPIRE_URL},
            {"name": "OECD 綜合領先指標（SDMX）", "url": OECD_URL},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
