"""Historical study "stress_family": does a FAMILY of market-priced stress gauges improve the 50-day moving average
rule (ma50, main_signal_check's main signal) for bitcoin, the way the NFCI filter (nfci_filter, rule N1) did - or
would one member passing just be luck from trying many indicators?

Pre-registered (written before this study was run; no other members, directions or parameters afterwards):
  Members (Yahoo Finance daily closes, public chart endpoint, user agent "btc-research-dashboard", no key; ETFs use
  the dividend-adjusted close, indices and futures the plain close). Direction of "favorable" fixed by economic
  logic before any run; every member compares today's value with the average of its own last 252 US trading days
  (including today):
    1 credit       HYG / IEF above its average (junk bonds beating Treasuries)
    2 quality      LQD / IEF above
    3 bond vol     ^MOVE below
    4 VIX curve    ^VIX / ^VIX3M below
    5 yield curve  ^TNX - ^IRX above (steepening)
    6 copper/gold  HG=F / GC=F above
    7 small caps   IWM / SPY above
    8 banks        KBE / SPY above
    9 EM           EEM / SPY above
   10 high beta    SPHB / SPLV above
  A ticker the loader cannot get is dropped (no substitute). Both legs of a ratio are matched on common dates; every
  member is put on SPY's trading-day calendar (last value on or before each day, at most 5 days carried). Downloads
  start 2008-01-01 (the project's loaders start 2013; the same endpoint is called with an earlier start so the
  2011-16 event period has data); if that fails the project's own loaders (rotation._share / cross_market's
  _yahoo_close) are used, then the last good copy kept in common.STORE.
  First usable date = first day with a 252-day average. Only members usable on or before 2014-12-31 (the whole exam
  from 2015-01) enter the family count and the composite; later ones are reference rows only.
  Timing (cross_market's alignment): the decision is taken at the last bitcoin daily close of month M (00:00 UTC,
  te._daily_close, the candle labelled the last calendar day of M). A US close dated d happens before the candle
  labelled d closes, so the flag used is the one of the last SPY trading day dated on or before the last calendar
  day of M; held during month M+1 (business_cycle / nfci_filter monthly machinery).
  Rule template, identical for every member (= N1's form): hold BTC when ma50 is on AND the member is favorable.
  Per member: event table (favorable minus unfavorable next-month BTC return, all and 2011-16 / 2017-21 / 2022-now,
  the member flag alone); main_signal_check's full exam via business_cycle._Exam (200 ordinary-person simulations
  per clock). Member bar: ranking score >= ma50's (incl. the 95% loss eligibility rule, bc._score_ge) AND median
  deepest loss of the textbook cells no worse than ma50's.
  FAMILY TEST (main result): K = number of family members meeting the bar. Luck benchmark: all member series
  (raw values on the SPY calendar) circularly shifted TOGETHER by one random offset of 252 ... (length - 252) trading
  days (keeps their cross-correlation), 252-day averages and monthly flags rebuilt, every member re-scored, K_shift
  counted; 200 draws, fixed seed. To fit the runtime the luck benchmark uses 20 ordinary-person simulations per clock
  and the real family is re-scored the same way (K_real at 20 is the K compared). Family passes if the share of
  draws with K_shift >= K_real is <= 10%.
  COMPOSITE C1 (fixed in advance): hold when ma50 is on AND at least half of the family members that have a flag
  that month are favorable (no member with a flag = unknown = cash). C1 passes if ranking score >= ma50's, median
  deepest loss no worse, favorable > unfavorable next-month return in every period with data, AND placebo share
  <= 10% (same joint shifts; share of draws whose shifted C1 ranking score is >= real C1's, both at 20 simulations).
  Report only: pairwise agreement (share of months with the same flag, and correlation) of members' monthly flags,
  and with the NFCI flag of nfci_filter (N1's 52-week-average flag); calendar-year returns of C1 vs ma50 from 2015;
  a look-ahead check (values after three cut dates scrambled; no monthly flag decided up to then may change);
  today's states. Speed-up that changes no result: a member that already fails on textbook numbers is not run
  through the ordinary-person simulation in the luck benchmark.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import cross_market as cm
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import rotation as ro

AVG_D = 252                       # trading days in the average
GAP_D = 252                       # minimum joint shift
CARRY = 5                         # days a value may be carried on the SPY calendar
DRAWS, PL_SIMS = 200, 20
FAMILY_BY = pd.Timestamp("2014-12-31")
FROM_TS = 1199145600              # 2008-01-01
FIRST = msc.DAILY_STARTS[0]
STORE_FMT = "stress_family_{}.json"

# key, legs (a, b, op), favorable when above (+1) or below (-1) its average, short name, plain description
MEMBERS = (
    ("credit", ("HYG", "IEF", "/"), +1, "高收益債／公債",
     "HYG（高收益債 ETF，俗稱垃圾債）除以 IEF（7–10 年美國公債 ETF）；高於一年平均＝垃圾債比公債強＝有利"),
    ("quality", ("LQD", "IEF", "/"), +1, "公司債／公債",
     "LQD（投資等級公司債 ETF）除以 IEF（7–10 年美國公債 ETF）；高於一年平均＝公司債比公債強＝有利"),
    ("move", ("^MOVE", None, None), -1, "債市波動",
     "^MOVE（美國公債波動指數，債市的恐慌指數）；低於一年平均＝債市平靜＝有利"),
    ("vixcurve", ("^VIX", "^VIX3M", "/"), -1, "VIX 期限結構",
     "^VIX（未來 1 個月恐慌指數）除以 ^VIX3M（未來 3 個月恐慌指數）；低於一年平均＝近期恐慌相對低＝有利"),
    ("curve", ("^TNX", "^IRX", "-"), +1, "殖利率曲線",
     "^TNX（10 年期美債殖利率）減 ^IRX（13 週國庫券殖利率）；高於一年平均＝曲線變陡＝有利"),
    ("copgold", ("HG=F", "GC=F", "/"), +1, "銅／金",
     "HG=F（銅期貨）除以 GC=F（黃金期貨）；高於一年平均＝景氣金屬比避險金屬強＝有利"),
    ("small", ("IWM", "SPY", "/"), +1, "小型股／大型股",
     "IWM（羅素 2000 小型股 ETF）除以 SPY（標普 500 ETF）；高於一年平均＝小型股比較強＝有利"),
    ("banks", ("KBE", "SPY", "/"), +1, "銀行股／大盤",
     "KBE（美國銀行股 ETF）除以 SPY（標普 500 ETF）；高於一年平均＝銀行股比較強＝有利"),
    ("em", ("EEM", "SPY", "/"), +1, "新興市場／美股",
     "EEM（新興市場股票 ETF）除以 SPY（標普 500 ETF）；高於一年平均＝新興市場比較強＝有利"),
    ("beta", ("SPHB", "SPLV", "/"), +1, "高波動股／低波動股",
     "SPHB（標普 500 高貝塔、高波動股 ETF）除以 SPLV（標普 500 低波動股 ETF）；高於一年平均＝投機股比較強＝有利"),
)
KEYS = [m[0] for m in MEMBERS]
SHORT = {m[0]: m[3] for m in MEMBERS}
DESC = {m[0]: m[4] for m in MEMBERS}
SIGN = {m[0]: m[2] for m in MEMBERS}
TICK_NAME = {"HYG": "高收益債 ETF", "IEF": "7–10 年美國公債 ETF", "LQD": "投資等級公司債 ETF", "^MOVE": "美國公債波動指數",
             "^VIX": "1 個月恐慌指數", "^VIX3M": "3 個月恐慌指數", "^TNX": "10 年期美債殖利率", "^IRX": "13 週國庫券殖利率",
             "HG=F": "銅期貨", "GC=F": "黃金期貨", "IWM": "羅素 2000 小型股 ETF", "SPY": "標普 500 ETF",
             "KBE": "美國銀行股 ETF", "EEM": "新興市場股票 ETF", "SPHB": "標普 500 高波動股 ETF",
             "SPLV": "標普 500 低波動股 ETF"}

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _adjusted(symbol: str) -> bool:
    return not (symbol.startswith("^") or symbol.endswith("=F"))


def _fetch(symbol: str) -> dict:
    """Finished daily closes since 2008 from Yahoo's public chart endpoint (adjusted close for ETFs)."""
    reply = requests.get(ro.YAHOO.format(symbol), headers=common.HEADERS, timeout=30,
                         params={"period1": FROM_TS, "period2": int(time.time()), "interval": "1d"})
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    if _adjusted(symbol):
        vals = result["indicators"]["adjclose"][0]["adjclose"]
    else:
        vals = result["indicators"]["quote"][0]["close"]
    s = pd.Series(vals, index=pd.to_datetime(result["timestamp"], unit="s").normalize(), dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[(s.index < pd.Timestamp(common.today_utc())) & (s > 0 if symbol not in ("^IRX", "^TNX") else s > -50)]
    out = {_d(day): float(v) for day, v in s.items() if math.isfinite(float(v))}
    if len(out) < 500:
        raise RuntimeError(f"只回傳 {len(out)} 筆")
    return out


def _project_loader(symbol: str) -> dict:
    if _adjusted(symbol):
        return {_d(day): float(v) for day, v in ro._share(symbol).items()}
    return cm._yahoo_close(symbol)


def _ticker(symbol: str) -> tuple[pd.Series | None, str]:
    """Own download -> project loader -> stored copy. Returns (series or None, note)."""
    path = common.STORE / STORE_FMT.format(symbol.replace("^", "").replace("=", "_"))
    note = ""
    days = None
    for fn in (_fetch, _project_loader):
        try:
            days = fn(symbol)
            if fn is _project_loader:
                note = f"{symbol}（{TICK_NAME[symbol]}）2008 年起的下載失敗，改用專案自己的載入器（2013 年起）"
            break
        except Exception:  # noqa: BLE001 - try the next way
            continue
    if days is None:
        if not path.exists():
            return None, f"{symbol}（{TICK_NAME[symbol]}）這次 Yahoo 取不到，也沒有存檔"
        days = json.loads(path.read_text(encoding="utf-8"))
        note = f"{symbol}（{TICK_NAME[symbol]}）這次抓不到，改用上次存下來的資料（到 {max(days)}）"
    else:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s[np.isfinite(s.values)], note


def _load() -> dict:
    problems, raw, used = [], {}, {}
    symbols = sorted({x for m in MEMBERS for x in m[1][:2] if x} | {"SPY"})
    for sym in symbols:
        s, note = _ticker(sym)
        if note:
            problems.append(note + "。")
        if s is not None:
            raw[sym] = s
            used[sym] = True
    if "SPY" not in raw:
        raise RuntimeError("SPY 取不到，沒有交易日日曆")
    cal = raw["SPY"].index
    panel, dropped = {}, []
    for key, (a, b, op), *_ in MEMBERS:
        if a not in raw or (b and b not in raw):
            dropped.append(key)
            continue
        if b:
            both = pd.concat([raw[a], raw[b]], axis=1, join="inner").dropna()
            v = both.iloc[:, 0] / both.iloc[:, 1] if op == "/" else both.iloc[:, 0] - both.iloc[:, 1]
        else:
            v = raw[a]
        panel[key] = v.reindex(cal.union(v.index)).ffill(limit=CARRY).reindex(cal).where(cal >= v.index[0])
    for key in dropped:
        problems.append(f"「{SHORT[key]}」需要的資料取不到，這個成員這次拿掉，沒有用別的代替。")
    other, more = msc._load_other()
    nraw, nwhere, nnotes = None, "", []
    try:
        nraw, nwhere, nnotes = nf._series()
    except Exception as exc:  # noqa: BLE001 - NFCI is only for the agreement table
        nnotes = [f"NFCI 這次取不到（{type(exc).__name__}），一致程度表沒有 NFCI 那一列。"]
    return {"close": bc.te._daily_close(), "panel": pd.DataFrame(panel, index=cal), "raw": raw, "other": other,
            "problems": problems + more + nnotes, "dropped": dropped, "nfci": nraw, "nfci_where": nwhere}


# ---------- the signal ----------

def _daily_fav(panel: pd.DataFrame) -> pd.DataFrame:
    avg = panel.rolling(AVG_D, min_periods=AVG_D).mean()
    out = {}
    for k in panel:
        sgn = SIGN[k]
        fav = (panel[k] > avg[k]) if sgn > 0 else (panel[k] < avg[k])
        out[k] = fav.astype(float).where(avg[k].notna() & panel[k].notna())
    return pd.DataFrame(out, index=panel.index)


def _monthly(fav: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Decision month M -> flag of the last SPY trading day dated on or before the last calendar day of M."""
    last_day = months.to_timestamp(how="end").normalize()
    pos = np.searchsorted(fav.index.values, last_day.values, side="right") - 1
    ok = pos >= 0
    vals = np.where(ok[:, None], fav.values[np.clip(pos, 0, None)], np.nan)
    return pd.DataFrame(vals, index=months, columns=fav.columns)


def _composite(M: pd.DataFrame, fam: list[str]) -> pd.Series:
    x = M[fam]
    have = x.notna().sum(axis=1)
    fav = x.sum(axis=1)
    return (fav >= have / 2).astype(float).where(have > 0)


# ---------- a lazy exam: same numbers as _Exam.summary, skipping the ordinary-person part when it cannot matter ----------

def _passes(ex: bc._Exam, F: pd.Series, base: dict, sims: int, need_loss: bool) -> bool:
    """bc._score_ge(summary(F), base) [and median deepest loss no worse, when need_loss]."""
    pos = ex.btc_pos("m2", F)
    net, _ = bc.te._run(np.concatenate([[np.nan], pos[:-1]]), ex.btc_change, bc.COST)
    dr, dd = msc._suffix(msc._equity(net), ex.dstarts, bc.YEAR)
    dr, dd = dr[:, 0], dd[:, 0]
    dh = ex.dhold
    cmac = bc._on_days(F, ex.m.dates)
    n = len(ex.cstarts)
    tr, td = np.zeros((n, 24)), np.zeros((n, 24))
    ws = []
    for h in range(24):
        w = np.nan_to_num(bc._pos("m2", ex.cma[h], cmac), nan=0.0).astype(float)
        ws.append(w)
        r, d_ = msc._suffix(np.concatenate([[1.0], ex.eng._mid(h, w, bc.COST, None)]), ex.cstarts, bc.YEAR)
        tr[:, h], td[:, h] = r[:, 0], d_[:, 0]
    hr, hd = ex.chold
    if need_loss and float(np.median(td)) < base["c_deep_med"] - 1e-12:
        return False
    loss_ok = min(int((dd > dh[1]).sum()) / len(dr), int((td > hd[:, None]).sum()) / td.size)
    elig = loss_ok >= bc.LOSS_BAR
    if base["eligible"] and not elig:
        return False
    if elig and not base["eligible"]:
        return True
    o_n = o_b = 0
    for a, o in ex.other.items():
        macro = bc._on_days(F, o["close"].index + bc.DAY)
        p = bc._pos("m2", o["ma"], macro)
        net, _ = bc.te._run(np.concatenate([[np.nan], p[:-1]]), o["change"], bc.COST)
        r, _d2 = msc._suffix(msc._equity(net), o["starts"], o["year"])
        o_n += len(o["starts"])
        o_b += int((r[:, 0] > o["hold"][0]).sum())
    edge = tr - hr[:, None]
    three = [int((dr > dh[0]).sum()) / len(dr), int((edge > 0).sum()) / edge.size, (o_b / o_n) if o_n else 0.0]
    bar = base["low"] - 1e-12
    if min(three) < bar:
        return False
    beat = tot = 0
    for h in range(24):
        b, wl = rt._later(h, ws[h], ex.days)
        rng = np.random.default_rng(bc.SEED + 1000 * bc.KI["m2"] + h)
        p = rt._missed(wl, bc.P_MISS, rng, sims)
        r, _d2 = msc._suffix(np.vstack([np.ones((1, sims)), ex.eng._mid(b, p, bc.P_COST, None)]), ex.cstarts, bc.YEAR)
        pe = r - hr[:, None]
        beat += int((pe > 0).sum())
        tot += pe.size
    return beat / tot >= bar


def _bar(x: dict, base: dict) -> bool:
    return bc._score_ge(x, base) and x["c_deep_med"] >= base["c_deep_med"] - 1e-12


# ---------- agreement ----------

def _agree(M: pd.DataFrame, nfci: pd.Series | None) -> dict:
    cols = list(M.columns) + (["nfci"] if nfci is not None else [])
    X = M.copy()
    if nfci is not None:
        X["nfci"] = nfci.reindex(M.index)
    X = X[X.index >= pd.Period("2015-01", "M")]
    out = {}
    for a in cols:
        for b in cols:
            if a == b:
                out[(a, b)] = (1.0, 1.0, 0)
                continue
            pair = X[[a, b]].dropna()
            if len(pair) < 12:
                out[(a, b)] = (None, None, len(pair))
                continue
            same = float((pair[a] == pair[b]).mean())
            c = float(pair[a].corr(pair[b])) if pair[a].std() > 0 and pair[b].std() > 0 else None
            out[(a, b)] = (same, c if c is None or math.isfinite(c) else None, len(pair))
    return out


# ---------- look-ahead ----------

def _lookahead(panel: pd.DataFrame, months) -> tuple[int, int]:
    rng = np.random.default_rng(17)
    M0 = _monthly(_daily_fav(panel), months)
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        noise = pd.DataFrame(np.exp(rng.normal(0, 0.3, panel.shape)), index=panel.index, columns=panel.columns)
        keep = np.broadcast_to((panel.index <= c)[:, None], panel.shape)
        p2 = panel.where(keep, panel * noise)
        M2 = _monthly(_daily_fav(p2), months)
        upto = M0.index <= c.to_period("M")
        for k in panel:
            total += 1
            a, b = M0.loc[upto, k].fillna(-1).values, M2.loc[upto, k].fillna(-1).values
            bad += int(not np.array_equal(a, b))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, panel = data["close"], data["panel"]
    months = pd.period_range(panel.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
    fav = _daily_fav(panel)
    M = _monthly(fav, months)
    first_use = {k: (fav[k].first_valid_index()) for k in panel}
    fam = [k for k in panel if first_use[k] is not None and first_use[k] <= FAMILY_BY]
    ref = [k for k in panel if k not in fam]
    C1 = _composite(M, fam)

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)

    rng = np.random.default_rng(bc.SEED + 7)
    ev = {k: bc._events(close, M[k], rng) for k in panel}
    ev["c1"] = bc._events(close, C1, rng)
    s = {"ma50": ex.summary("ma50", None)}
    for k in list(panel) + ["c1"]:
        s[k] = ex.summary("m2", C1 if k == "c1" else M[k])
    t_full = time.time() - t0

    # luck benchmark at PL_SIMS
    small = {"ma50": ex.summary("ma50", None, PL_SIMS), "c1": ex.summary("m2", C1, PL_SIMS)}
    for k in fam:
        small[k] = ex.summary("m2", M[k], PL_SIMS)
    meet_small = {k: _bar(small[k], small["ma50"]) for k in fam}
    k_real = sum(meet_small.values())
    prng = np.random.default_rng(bc.SEED + 77)
    n = len(panel)
    ks, c1_hits, c1_ma = [], 0, 0
    vals = panel[fam].values
    for sh in prng.integers(GAP_D, n - GAP_D + 1, DRAWS):
        p2 = pd.DataFrame(np.roll(vals, int(sh), axis=0), index=panel.index, columns=fam)
        M2 = _monthly(_daily_fav(p2), months)
        ks.append(sum(int(_passes(ex, M2[k], small["ma50"], PL_SIMS, True)) for k in fam))
        c2 = _composite(M2, fam)
        c1_hits += int(_passes(ex, c2, small["c1"], PL_SIMS, False))
        c1_ma += int(_passes(ex, c2, small["ma50"], PL_SIMS, False))
    ks = np.array(ks)

    pos = {"ma50": ex.btc_pos("ma50", None), "c1": ex.btc_pos("m2", C1)}
    a = close.index >= pd.Timestamp(FIRST)
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["c1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()))
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST),
             **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    nfm = nf._flags(data["nfci"], "avg52", months) if data["nfci"] is not None else None
    agree = _agree(M, nfm)
    look = _lookahead(panel, months)
    return {"ex": ex, "M": M, "C1": C1, "fam": fam, "ref": ref, "first_use": first_use, "ev": ev, "s": s,
            "small": small, "meet_small": meet_small, "k_real": k_real, "ks": ks, "c1_plc": c1_hits / DRAWS,
            "c1_plc_ma": c1_ma / DRAWS, "pos": pos, "differ": differ, "years": years, "agree": agree, "nfm": nfm,
            "look": look, "problem": problem, "fav": fav, "months": months, "t_full": t_full,
            "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, M = data["close"], r["ex"], r["s"], r["ev"], r["M"]
    panel, fam = data["panel"], r["fam"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base = s["ma50"]
    n_c, n_d = base["c_n"], base["d_n"]
    keys = list(panel)
    meet = {k: _bar(s[k], base) for k in keys}
    k_full = sum(meet[k] for k in fam)
    k_real, ks = r["k_real"], r["ks"]
    luck = float((ks >= k_real).mean())
    fam_pass = k_real > 0 and luck <= 0.10
    c1 = s["c1"]
    e1 = ev["c1"]
    q = {"score": bc._score_ge(c1, base), "loss": c1["c_deep_med"] >= base["c_deep_med"] - 1e-12,
         "event": all(p["diff"] > 0 for p in e1["periods"].values() if _ok(p["diff"])),
         "plc": r["c1_plc"] <= 0.10}
    q["pass"] = all(q.values())

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    def role(k: str) -> str:
        return "算進家族" if k in fam else "只參考"

    # ----- table 1: members -----
    rows1 = []
    for k in keys:
        rows1.append([SHORT[k], DESC[k], _d(panel[k].first_valid_index()), _d(r["first_use"][k]), role(k)])
    for k in data["dropped"]:
        rows1.append([SHORT[k], DESC[k], "—", "—", "取不到，拿掉"])
    t1 = {"caption": ("家族成員和方向（跑之前就依經濟道理定好，不看結果）。每個成員都和自己最近 252 個美股交易日（約一年）的平均比；"
                      "「第一天能用」是一年平均第一次算得出來的那天。2014 年底以前能用的才算進家族（才能涵蓋 2015 年起的整個考試）。"
                      "ETF 用含配息的調整後收盤，指數和期貨用一般收盤；比值的兩邊只用兩邊都有收盤的日子。"),
          "columns": ["成員", "怎麼算、哪邊算有利", "資料第一天", "第一天能用", "角色"], "rows": rows1}

    # ----- table 2: event table -----
    rows2 = []
    for k in keys + ["c1"]:
        e = ev[k]
        a = e["all"]
        ci = a.get("ci")
        lab = "C1：家族至少一半有利" if k == "c1" else SHORT[k]
        rows2.append([lab, f"{a['nf']}｜{a['nu']}", f"{_p(a['mf'])}｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" if _ok(p["diff"]) else "—" for p in e["periods"].values()])
    first_ev = min(ev[k]["first"] for k in keys if ev[k]["first"] is not None)
    t2 = {"caption": ("只看成員本身（不含 50 日均線）：每個月最後一根比特幣日線（UTC 0 點收盤）時，用那個月最後一個美股交易日的收盤判斷有利或不利，"
                      "格子是下一個月比特幣報酬的平均。差距＝有利減不利，括號是區塊拔靴法（連續 "
                      f"{bc.BLOCK} 個月一塊、{bc.BOOT:,} 次）的 90% 區間；分段欄括號是（有利｜不利）月數。持有月份最早從 "
                      f"{first_ev.strftime('%Y-%m')} 起（一年平均要先有值）。"),
          "columns": ["成員", "月數：有利｜不利", "下個月平均：有利｜不利", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in bc.PERIODS],
          "rows": rows2}

    # ----- table 3: exam per member -----
    rows3 = []
    for k in ["ma50"] + keys + ["c1"]:
        x = s[k]
        if k == "ma50":
            lab, verdict_cell, small_cell = "50 日均線（基準）", "—", "—"
        else:
            lab = "C1：50 日均線而且家族至少一半有利" if k == "c1" else f"{SHORT[k]}（{role(k)}）"
            verdict_cell = "達到" if _bar(x, base) else "沒達到"
            if k in fam:
                small_cell = "達到" if r["meet_small"][k] else "沒達到"
            elif k == "c1":
                small_cell = "達到" if _bar(r["small"]["c1"], r["small"]["ma50"]) else "沒達到"
            else:
                small_cell = "—"
        ck = x["checks"]
        rows3.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), _share(ck[3]),
                      _share(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）"),
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", _pp(x["c_edge"]), verdict_cell, small_cell])
    t3 = {"caption": (f"main_signal_check 的公平考試，每個成員都用同一個樣板「50 日均線持有而且成員有利才持有」。比特幣 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣格子照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產（ETH、SOL、BNB、XRP、MSTR）"
                      "用自己的 50 日均線、同一個美國市場訊號。排名分數＝四項「報酬贏過一直抱著」比例裡最低的那項（最深虧損比抱著淺的比例要到 95%）。"
                      "「達到門檻」＝排名分數不低於均線，而且最深虧損中位數不比均線深。最後一欄是運氣比較用的簡化版"
                      f"（一般人做法每個收盤時間只模擬 {PL_SIMS} 次），家族個數 K 用的是這一欄。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產",
                      "排名分數", "最深虧損中位數：課本｜一般人", "每年多賺中位數（課本）", "達到門檻", f"達到門檻（{PL_SIMS} 次版）"],
          "rows": rows3}

    # ----- table 4: family test -----
    dist = {int(v): int((ks == v).sum()) for v in np.unique(ks)}
    rows4 = [["家族成員數", f"{len(fam)} 個（{'、'.join(SHORT[k] for k in fam)}）"],
             [f"真的 K（{PL_SIMS} 次版，拿來比的）", f"{k_real} 個：{'、'.join(SHORT[k] for k in fam if r['meet_small'][k]) or '沒有'}"],
             [f"真的 K（完整 {bc.SIMS} 次版，對照）", f"{k_full} 個：{'、'.join(SHORT[k] for k in fam if meet[k]) or '沒有'}"],
             [f"隨機錯位 {DRAWS} 次的 K 分布", "、".join(f"K={v}：{c} 次" for v, c in sorted(dist.items()))],
             ["隨機錯位的 K 平均｜中位數", f"{ks.mean():.2f}｜{np.median(ks):.0f}"],
             ["隨機錯位 K 不少於真的比例（≤ 10% 才過）", f"{luck * 100:.1f}%（{int((ks >= k_real).sum())} / {DRAWS}）"],
             ["對照：隨機錯位至少 1 個達到門檻的比例", f"{float((ks >= 1).mean()) * 100:.1f}%（{int((ks >= 1).sum())} / {DRAWS}）"],
             ["家族檢定結果", "通過" if fam_pass else "沒有通過"]]
    t4 = {"caption": ("主要結果：同時試很多指標，總會有一兩個碰巧過關，所以這裡看的是「整個家族有幾個達到門檻」，再和運氣比。"
                      f"運氣的做法：把全部成員的每日數值一起（同一個位移）循環平移 {GAP_D} 個交易日到「長度 − {GAP_D}」個交易日，"
                      "成員之間的關係不變，但和比特幣的時間對應被打亂；重新算一年平均、每月訊號，整套考試重做，數一數有幾個達到門檻。"
                      f"{DRAWS} 次，固定種子。為了在幾分鐘內算完，一般人做法每個收盤時間只模擬 {PL_SIMS} 次，真的家族也用同樣方式重算。"),
          "columns": ["項目", "數字"], "rows": rows4}

    # ----- table 5: composite -----
    ps = [p["diff"] for p in e1["periods"].values() if _ok(p["diff"])]
    rows5 = [["排名分數不低於 50 日均線", f"{_share(c1['low'])} 對 {_share(base['low'])}", yes(q["score"])],
             [f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(c1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(q["loss"])],
             ["有資料的每一段都是有利月比不利月好", "、".join(_p(v) for v in ps), yes(q["event"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['c1_plc'])}（打亂後不輸均線的 {_share(r['c1_plc_ma'])}）", yes(q["plc"])],
             ["結果", "—", "通過" if q["pass"] else "沒有通過"]]
    t5 = {"caption": ("事先定好的綜合規則 C1：50 日均線持有，而且那個月有訊號的家族成員至少一半是有利，才持有。四項都要過。"
                      f"隨機錯位用和家族檢定同一批 {DRAWS} 次平移，比較打亂後 C1 的排名分數是否不低於真的 C1（兩邊都用 {PL_SIMS} 次簡化版）。"),
          "columns": ["條件", "數字", "結果"], "rows": rows5}

    # ----- table 6: agreement -----
    ag = r["agree"]
    cols6 = keys + (["nfci"] if r["nfm"] is not None else [])
    name6 = {**SHORT, "nfci": "NFCI"}
    rows6 = []
    for a in cols6:
        row = [name6[a]]
        for b in cols6:
            same, c, _n = ag[(a, b)]
            if a == b or same is None:
                row.append("—")
            else:
                row.append(f"{same * 100:.0f}%" + (f"｜{c:+.2f}" if c is not None else ""))
        rows6.append(row)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:] if ag[(a, b)][1] is not None]
    mean_c = float(np.mean([ag[p][1] for p in pairs])) if pairs else None
    mean_same = float(np.mean([ag[p][0] for p in pairs])) if pairs else None
    nf_c = {k: ag[(k, "nfci")] for k in keys} if r["nfm"] is not None else {}
    c1_nf = None
    if r["nfm"] is not None:
        both = pd.concat([r["C1"].rename("c"), r["nfm"].rename("n")], axis=1).dropna()
        both = both[both.index >= pd.Period("2015-01", "M")]
        if len(both) > 12:
            c1_nf = (float((both["c"] == both["n"]).mean()), float(both["c"].corr(both["n"])))
    t6 = {"caption": ("是不是同一個訊號？2015-01 起每個月的有利／不利，兩兩比：格子是「同一個答案的月份比例｜相關係數」。"
                      "相關係數 0 代表沒關係，1 代表完全一樣。NFCI 是 nfci_filter 用的芝加哥聯準銀行金融情勢指數（低於自己 52 週平均＝有利）。"
                      + (f"C1 和 NFCI：{c1_nf[0] * 100:.0f}% 的月份相同、相關 {c1_nf[1]:+.2f}。" if c1_nf else "")),
          "columns": ["成員"] + [name6[c] for c in cols6], "rows": rows6}

    # ----- table 7: years -----
    dd_, nn = r["differ"]
    yrs = sorted(r["years"]["ma50"])
    rows7 = []
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["c1"].get(y)
        rows7.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    rows7.append([f"和 50 日均線持有狀態不同的天數（{FIRST} 起）", "—", "—",
                  f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—", "—"])
    t7 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看 C1 和 50 日均線的差來自哪幾年；"
                      "C1 只會比均線少持有。"),
          "columns": ["年份", "一直抱著", "50 日均線", "C1", "C1 減均線（百分點）"], "rows": rows7}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    fav, pm = r["fav"], r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    def fw(v) -> str:
        return "有利" if v == 1 else "不利" if v == 0 else "—"

    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
            "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"}]
    mrow = M.loc[cur] if cur in M.index else None
    nfav = int(sum(1 for k in fam if mrow is not None and mrow[k] == 1))
    nhave = int(sum(1 for k in fam if mrow is not None and _ok(mrow[k])))
    now.append({"label": f"家族（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
                "value": f"{nhave} 個有訊號、{nfav} 個有利；"
                         + "、".join(f"{SHORT[k]}{fw(mrow[k]) if mrow is not None else '—'}" for k in keys)})
    lastrow = fav.iloc[-1]
    now.append({"label": f"最新美股收盤（{_d(fav.index[-1])}）的狀態",
                "value": "、".join(f"{SHORT[k]}{fw(lastrow[k])}" for k in keys)
                         + f"（{int((lastrow[fam] == 1).sum())} 個有利；下一次決定看月底那天的收盤）"})
    now.append({"label": "比特幣：50 日均線｜C1", "value": f"{word(pm['ma50'][-1])}｜{word(pm['c1'][-1])}"})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    def diff(y):
        a, b = r["years"]["ma50"].get(y), r["years"]["c1"].get(y)
        return b - a if _ok(a) and _ok(b) else None

    better = [y for y in yrs if _ok(diff(y)) and diff(y) > 0.5]
    worse = [y for y in yrs if _ok(diff(y)) and diff(y) < -0.5]
    winners = [SHORT[k] for k in fam if r["meet_small"][k]]
    nf_txt = ""
    ok_c = [v[1] for v in nf_c.values() if v[1] is not None]
    if ok_c:
        nf_txt = f"各成員和 NFCI 訊號的相關介於 {min(ok_c):+.2f} 到 {max(ok_c):+.2f}" + (
            f"，C1 和 NFCI 相關 {c1_nf[1]:+.2f}。" if c1_nf else "。")
    overlap = ""
    win_keys = [k for k in fam if r["meet_small"][k]]
    if win_keys and set(win_keys) <= {"credit", "quality"}:
        cc = ag[("credit", "quality")][1] if len(win_keys) == 2 else None
        nfv = [f"{nf_c[k][1]:+.2f}" for k in win_keys if nf_c and nf_c[k][1] is not None]
        overlap = ("過關的都是債券信用類（垃圾債、公司債對公債）"
                   + (f"，兩個彼此相關 {cc:+.2f}" if cc is not None else "")
                   + (f"，和 NFCI 相關 {'、'.join(nfv)}" if nfv else "")
                   + f"，比較像 NFCI 裡信用那一塊的同一個訊號，不是十個獨立的發現；就算只算一個，打亂後至少 1 個過關的也只有 "
                   f"{float((ks >= 1).mean()) * 100:.1f}%。")
    verdict = (f"家族檢定{'通過' if fam_pass else '沒有通過'}：{len(fam)} 個市場壓力指標用同一個樣板加在 50 日均線上，"
               f"有 {k_real} 個達到門檻（排名分數不輸均線、最深虧損不更深）"
               + (f"（{'、'.join(winners)}）" if winners else "")
               + f"；把時間打亂後平均只有 {ks.mean():.2f} 個，{luck * 100:.1f}% 的打亂結果不少於 {k_real} 個，"
               + ("比運氣明顯多。" if fam_pass else "和運氣分不出來。")
               + f"綜合規則 C1（均線加上家族至少一半有利）{'通過' if q['pass'] else '沒有通過'}：排名分數 {_share(c1['low'])} 對均線 "
               f"{_share(base['low'])}（{yes(q['score'])}），最深虧損中位數 {_pct(c1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}"
               f"（{yes(q['loss'])}），各段有利減不利 {'、'.join(_p(v) for v in ps)}（{yes(q['event'])}），隨機錯位一樣好 "
               f"{_share(r['c1_plc'])}（{yes(q['plc'])}）。"
               + (f"成員兩兩平均有 {mean_same * 100:.0f}% 的月份答案相同、平均相關 {mean_c:+.2f}；" if mean_c is not None else "")
               + nf_txt
               + overlap
               + (f"C1 比均線好的年份：{'、'.join(str(y) for y in better)}" if better else "C1 沒有一年明顯比均線好")
               + (f"；比均線差的年份：{'、'.join(str(y) for y in worse)}。" if worse else "。"))

    tot, bad = r["look"]
    notes = [
        ("事先寫死的設計（照原文，跑之前寫好）：ten members, direction fixed by economic logic, each vs its own 252-trading-day "
         "average; rule template = hold BTC when ma50 is on AND the member is favorable, decided at the month-end BTC close "
         "with the last US close dated on or before that day. Member bar = ranking score >= ma50's AND median deepest loss no "
         f"worse. Family test: K vs {DRAWS} joint circular shifts (>= {GAP_D} trading days) at {PL_SIMS} ordinary-person "
         "simulations per clock; pass if share(K_shift >= K_real) <= 10%. C1 = ma50 AND at least half of family members with a "
         "flag favorable; pass if ranking >= ma50's, median deepest loss no worse, favorable > unfavorable in every period "
         "with data, placebo <= 10%. Members usable only after 2014 are reference only."),
        ("為什麼看家族、不看最好的一個：一次試十個指標，就算全部都沒用，碰巧有一兩個看起來過關也很正常。"
         "把所有指標一起和比特幣的時間錯開（同一個位移，保留它們彼此的關係），看「沒有真的關係時通常會有幾個過關」，"
         "真的個數要明顯多過這個數字，才算這一類指標有用。最好的單一成員只能當參考，不能當成發現。"),
        ("和其他研究的關係：NFCI 本身就包含信用利差、波動率、股市等市場價格，所以這個家族和 nfci_filter 有重疊。"
         "cross_market 已經測過 VIX 水準、QQQ／SPY 趨勢、十年期殖利率急升、美元，這裡刻意不重複。"),
        ("資料：Yahoo Finance 公開的圖表資料（每個代號一個網址，2008-01-01 起），固定使用者代理 btc-research-dashboard，不用金鑰。"
         "時間對齊照 cross_market：美股某天的收盤（美東下午 4 點，UTC 20–21 點）早於比特幣同一天標記的日線收盤（UTC 隔天 0 點），"
         "所以月底的決定用「那個月最後一個美股交易日」的收盤。期貨（銅、黃金）的日線也在 UTC 0 點以前結束。"
         f"偷看未來的檢查：三個切點之後的數值換成亂數，切點以前每個月的訊號 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("限制：ETF 用今天的調整後收盤（配息會回頭調整過去的價格，對比值的方向影響很小）；指數在 Yahoo 偶爾少幾天，"
         f"這裡最多沿用前一天的值 {CARRY} 天。SPHB／SPLV 2011 年才上市，一年平均 2012-05 起才有。"
         "其他幣、MSTR 用的是同一個美國市場訊號；相鄰起算月共用大部分歷史，「幾個裡幾個」不是互相獨立的證據。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的一組規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    syms = sorted(data["raw"])
    sources = [{"name": f"Yahoo Finance 圖表資料：{sym}（{TICK_NAME[sym]}）", "url": ro.YAHOO.format(sym)} for sym in syms]
    if data["nfci"] is not None:
        sources.append({"name": f"芝加哥聯準銀行 NFCI（一致程度比較用，這次來自{data['nfci_where']}）", "url": nf.CHI_URL})
    sources += [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]

    return {
        "id": "stress_family",
        "title": "市場壓力指標家族能不能幫 50 日均線",
        "question": ("NFCI（美國金融情勢指數）濾網通過了公平考試。換成其他市場上每天有價格的壓力指標——垃圾債對公債、債市波動、"
                     "恐慌指數的遠近月、殖利率曲線、銅對金、小型股、銀行股、新興市場、高波動股——用同一個樣板加在 50 日均線上，"
                     "過關的個數會不會比純粹運氣多？把它們合起來投票又如何？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t4, t5, t1, t2, t3, t6, t7],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
