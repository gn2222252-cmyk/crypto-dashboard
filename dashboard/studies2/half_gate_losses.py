"""Observation study "half_gate_losses": WHERE and WHY do the half-gate rules of half_gate lose against ma50?

No new rule is tested here. half_gate found three rules that met the project's standard bar (H_NFCI, H_OECD, H_LQD:
ma50 on -> 100% when the slow "conditions" gate is favorable, 50% when it is unfavorable), but "more money AND a
shallower deepest loss than ma50" held in only about 55-68% of start dates; it failed for starts in 2015 and 2023, gave
up 107 points in 2020, and its gains come from 2018 and 2022. This study lays the losing stretches out one by one,
anchored on the months, before any rule is made or changed. Every sentence is labelled 量到的 (measured, with the
number / k of N) or 推測 (a guess). Where the data cannot separate two explanations it says so.

Written BEFORE running (definitions fixed now; anything added after seeing results is labelled so in the output):
  Data and rules are exactly half_gate's (reused, not re-implemented): BTC 00:00 UTC daily closes (te._daily_close),
  ma50 = main_signal_check's ma50 state; gate flags via half_gate._flag / _vote (decision at the end of month M, held
  during M+1; unknown = unfavorable); position = ma50 x (100% favorable / 50% unfavorable); cost 0.10% (bc.COST);
  daily net returns via trend_ensemble._run, so each held month's return compounds the days of that month and the
  calendar-year numbers equal bc._years / half_gate's year table. H_VOTE (all five half_gate gates) is the fourth rule,
  reported briefly. Loaders write their fallback copies into dashboard/cache/_store/half_gate_losses/ (seeded from the
  shared store, which is only read).
  The numbers behind each gate for decision month M (the same values the flag uses; a mismatch count is reported):
    NFCI : the week nf._used_week picks (latest week at least 6 days old at the end of M); value, its 52-week average,
           gap = value - average (positive = tighter than usual = unfavorable); std of the same 52 values;
           change over 4 and 8 weeks (falling = improving); worst (highest) value in the last 26 weeks.
    OECD : the value of month M-2 (bc.LAG); its 10-month average; gap = average - value (positive = unfavorable);
           change over 2 and 3 months (rising = improving); worst (lowest) value in the last 5 months.
    LQD  : LQD/IEF on the last SPY day on or before the end of M; its 252-day average; gap = (average - value) /
           average in % (positive = unfavorable); change over 20 and 40 trading days (~4 and 8 weeks; rising =
           improving); worst (lowest) value in the last 126 trading days.
    z = gap / std of the averaging window (unit-free size of the gap).
  PART 1 - month by month from the first held month 2015-01: ma50's share of days held, the gate, the numbers, the
    position, BTC's month, ma50's month, the rule's month, rule minus ma50 (percentage points). HALF month = ma50 held
    on at least one day AND the gate unfavorable. For HALF months: count, up / down months, mean and median BTC month,
    total cost (sum of negative rule-minus-ma50) and total saving (sum of positive), as simple sums of monthly points.
    Split by gap terciles (within each gate's HALF months, raw gap in the gate's own unit) and by direction (improving
    vs worsening over 4 weeks / 2 months / 20 days and over 8 weeks / 3 months / 40 days).
  Indicator-only class of each unfavorable month (no price used):
    A hair's-breadth   : 0 <= z < 0.25.
    B lagging baseline : not A; improving over the longer horizon; and the worst value in the last half-window is worse
                         than today's value by at least 1 std (a recent shock inside the window, already easing).
    C clearly unfavorable : the rest with data.   E : no value (flag unknown, treated as unfavorable).
  Taxonomy of LOSING months (rule earned less than ma50 that month): (a) = A; (b) = B; (c) = C and BTC's month up
    ("genuinely tight and price rose anyway"); (d) = C and BTC's month down or flat, yet the half position cost
    (ma50's held days were up: "the gate was right about conditions but ma50 had already limited the damage");
    (e) = E or anything else. Winning months get the indicator class only.
  PART 2 - losing stretches: every calendar year where the rule's year return is below ma50's by more than 0.05
    points, and every start year where "more AND shallower" (daily starts, msc._month_starts 2015-01..2023-12, the
    half_gate Part 4 definition) holds for fewer than half of its starts. For each: the months that made the
    difference with the numbers, class, BTC month, the price context at the decision (distance from the 365-day high,
    the 3 months before, and whether the held month lies inside a signs_before_turns bear/bull swing) and what the
    other two gates said. Counts of months and points per class (a)-(e).
  PART 3 - winning stretches (2018, 2022) month by month by the same lens; cross-table indicator class x (months, up
    months, mean BTC month, points versus ma50) over all HALF months, also split into losing and winning months; the
    difference in mean points per month between C and A+B with a 90% interval from a cluster bootstrap over
    stretches (a stretch = consecutive HALF months with the same class; 2,000 draws, fixed seed), and the number of
    stretches each class rests on.
  PART 4 - failing start years: per start, did it earn less, lose more (deeper), or both; the relative money path
    log(rule equity / ma50 equity) split by calendar year after the start, the three months that widened the gap
    most, and the deepest-loss windows (peak and trough dates) of the rule and ma50 for the first start of the year.
  PART 5 - the three gates side by side: agreement over all decision months with all three flags; in HALF-eligible
    months (ma50 on) grouped by how many of the three are unfavorable (0-3): months, up share, mean BTC month, and the
    points of a 50% position versus ma50 (gate-agnostic); and in each gate's losing months, how many gates were
    unfavorable at once.
  PART 6 - half_gate's exploratory BUF and EASING NFCI gates: every month OUTSIDE held 2020 where they flip the
    original NFCI gate: the numbers, BTC month, ma50 share, and variant minus H_NFCI that month. Listed, not judged.
  PART 7 - at most three candidate ideas for a later pre-registered test, each 推測, each saying whether it was shaped
    by looking at these losses and what unseen data could test it. Written after seeing the results (labelled so).
  Today: each gate's state, gap, class, and its 8-week / 3-month change - description, not forecast.
  Grade if needed: "unclear" (observation study).
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import half_gate as hg
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import nfci_stress as nfs
from dashboard.studies2 import robust_trend as rt  # noqa: F401 - imported by half_gate's exam, kept for parity
from dashboard.studies2 import signs_before_turns as sbt
from dashboard.studies2 import trend_ensemble as te

ID = "half_gate_losses"
DAY = pd.Timedelta(days=1)
FIRST = msc.DAILY_STARTS[0]
FIRST_M = pd.Period(FIRST[:7], "M")
MAIN = ("H_NFCI", "H_OECD", "H_LQD")
ALL4 = MAIN + ("H_VOTE",)
G3 = ("nfci", "oecd", "lqd")
GN = {"nfci": "NFCI", "oecd": "OECD", "lqd": "LQD"}
RG = {"H_NFCI": "nfci", "H_OECD": "oecd", "H_LQD": "lqd", "H_VOTE": "vote"}
HAIR, SHOCK_SD = 0.25, 1.0
HOR = {"nfci": (4, 8, 26), "oecd": (2, 3, 5), "lqd": (20, 40, 126)}   # short, long horizon, half window (own units)
HOR_TXT = {"nfci": ("4 週", "8 週"), "oecd": ("2 個月", "3 個月"), "lqd": ("20 個交易日", "40 個交易日")}
BOOT, BOOT_SEED = 2000, 20261120
LOSE_EPS = 0.05
CLS_NAME = {"A": "A 差一點（差距小於自己波動的 0.25 倍）", "B": "B 衝擊後平均跟不上（正在轉好，但平均還被近期的衝擊拖著）",
            "C": "C 明顯不利", "E": "E 沒有資料（當不利處理）"}
TAX_NAME = {"a": "(a) 差一點", "b": "(b) 衝擊後平均跟不上", "c": "(c) 真的偏緊、價格照漲",
            "d": "(d) 條件沒說錯、但均線已經擋掉大部分跌幅，半倉只剩成本", "e": "(e) 其他／沒有資料"}

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID
SEED_FILES = ("nfci_filter_nfci.json", "business_cycle_oecd.json", "nfci_stress_ofr.json", "stress_family_HYG.json",
              "stress_family_IEF.json", "stress_family_LQD.json", "stress_family_SPY.json")

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data (own store folder; the shared store is only read) ----------

@contextmanager
def _own_store():
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        src, dst = REAL_STORE / name, MY_STORE / name
        try:
            if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        except Exception:  # noqa: BLE001 - a missing seed only means the loader must download
            pass
    old, old_nf = common.STORE, nf.STORE_PATH
    common.STORE = MY_STORE
    nf.STORE_PATH = MY_STORE / Path(old_nf).name
    try:
        yield
    finally:
        common.STORE = old
        nf.STORE_PATH = old_nf


def _load() -> dict:
    with _own_store():
        nfci, where_n, problems = nf._series()
        oecd, where_o, note = bc._series("oecd")
        if note:
            problems.append(note)
        panel, more = hg._credit()
        problems += more
        ofr, where_ofr, more = nfs._sibling("ofr")
        problems += more
        close = te._daily_close()
    raw = {"nfci": nfci, "oecd": oecd}
    if panel is not None:
        for k in panel:
            raw[k] = panel[k]
    if ofr is not None:
        raw["ofr"] = ofr
    if "lqd" not in raw:
        raise RuntimeError("LQD/IEF 取不到，H_LQD 沒辦法拆解")
    return {"close": close, "raw": raw, "problems": problems,
            "where": {"nfci": where_n, "oecd": where_o, "ofr": where_ofr}}


# ---------- the numbers behind each gate ----------

def _numbers(gate: str, s: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    """Per decision month: value, average, gap (positive = unfavorable), z, short/long change (positive = improving),
    recent-shock size in std, and the flag these numbers imply."""
    win = hg.BASEWIN[gate]
    sh, lo, half = HOR[gate]
    if gate == "lqd":
        s = s.dropna()
    avg = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=1)
    rows = []
    for M in months:
        if gate == "nfci":
            w = nf._used_week(s, M)
            k = None if w is None else s.index.get_loc(w)
        elif gate == "oecd":
            p = M - bc.LAG["oecd"]
            k = s.index.get_loc(p) if p in s.index and pd.notna(s.get(p)) else None
        else:
            end = M.to_timestamp(how="end").normalize()
            j = s.index.searchsorted(end, side="right") - 1
            k = j if j >= 0 else None
        rec = {"M": M, "v": np.nan, "avg": np.nan, "gap": np.nan, "z": np.nan, "ds": np.nan, "dl": np.nan,
               "shock": np.nan, "flag": np.nan, "at": None}
        if k is not None and pd.notna(avg.iloc[k]) and pd.notna(s.iloc[k]):
            v, a, sdv = float(s.iloc[k]), float(avg.iloc[k]), float(sd.iloc[k])
            past = s.iloc[max(0, k - half + 1):k + 1]
            if gate == "nfci":
                gap, worst = v - a, float(past.max()) - v
                ds = -(v - float(s.iloc[k - sh])) if k >= sh else np.nan
                dl = -(v - float(s.iloc[k - lo])) if k >= lo else np.nan
                flag = float(v < a)
            else:
                gap, worst = a - v, v - float(past.min())
                ds = v - float(s.iloc[k - sh]) if k >= sh else np.nan
                dl = v - float(s.iloc[k - lo]) if k >= lo else np.nan
                flag = float(v > a)
                if gate == "lqd":
                    gap = gap / a * 100
                    ds, dl = ds / float(s.iloc[k - sh]) * 100, dl / float(s.iloc[k - lo]) * 100
            z = (v - a) / sdv if gate == "nfci" else (a - v) / sdv
            rec.update(v=v, avg=a, gap=gap, z=z, ds=ds, dl=dl, shock=worst / sdv if sdv > 0 else np.nan, flag=flag,
                       at=s.index[k])
        rows.append(rec)
    return pd.DataFrame(rows).set_index("M")


def _cls(row) -> str:
    """Indicator-only class of an unfavorable month (no price used)."""
    if not _ok(row["z"]):
        return "E"
    if 0 <= row["z"] < HAIR:
        return "A"
    if _ok(row["dl"]) and row["dl"] > 0 and _ok(row["shock"]) and row["shock"] >= SHOCK_SD:
        return "B"
    return "C"


def _numtxt(gate: str, row) -> str:
    if not _ok(row["v"]):
        return "沒有數字"
    if gate == "nfci":
        return f"{row['v']:.3f}（平均 {row['avg']:.3f}）差 {row['gap']:+.3f}，z {row['z']:+.2f}"
    if gate == "oecd":
        return f"{row['v']:.2f}（平均 {row['avg']:.2f}）差 {row['gap']:+.2f}，z {row['z']:+.2f}"
    return f"{row['v']:.4f}（平均 {row['avg']:.4f}）差 {row['gap']:+.2f}%，z {row['z']:+.2f}"


def _chgtxt(gate: str, row) -> str:
    def w(v) -> str:
        return "—" if not _ok(v) else ("轉好" if v > 0 else "變差")
    a, b = HOR_TXT[gate]
    return f"{a}{w(row['ds'])}、{b}{w(row['dl'])}"


# ---------- monthly returns ----------

def _monthly(close: pd.Series, pos: np.ndarray) -> pd.Series:
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), close.pct_change().fillna(0.0).values, bc.COST)
    s = pd.Series(net, index=close.index)
    return s.groupby(s.index.to_period("M")).apply(lambda g: float(np.prod(1 + g.values) - 1))


def _rel_log(close: pd.Series, pa: np.ndarray, pb: np.ndarray) -> pd.Series:
    """Daily log(equity_a / equity_b) increments."""
    ch = close.pct_change().fillna(0.0).values
    na, _ = te._run(np.concatenate([[np.nan], pa[:-1]]), ch, bc.COST)
    nb, _ = te._run(np.concatenate([[np.nan], pb[:-1]]), ch, bc.COST)
    return pd.Series(np.log1p(na) - np.log1p(nb), index=close.index)


def _deep_window(close: pd.Series, pos: np.ndarray, start: int) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    net, _ = te._run(np.concatenate([[np.nan], pos[:-1]]), close.pct_change().fillna(0.0).values, bc.COST)
    e = np.cumprod(1 + net[start:])
    e = np.concatenate([[1.0], e])
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1
    t = int(np.argmin(dd))
    pk = int(np.argmax(e[:t + 1])) if t > 0 else 0
    idx = close.index[start:]
    # e[i] = money after day i-1 of the span; map back to dates
    return float(dd[t] * 100), idx[max(pk - 1, 0)], idx[max(t - 1, 0)]


def _cluster_ci(a: list, b: list, rng) -> tuple[float, float] | None:
    """90% interval for mean(points per month | class set a) - mean(... | b), resampling whole stretches."""
    if len(a) < 2 or len(b) < 2:
        return None
    out = []
    for _ in range(BOOT):
        xa = [a[i] for i in rng.integers(0, len(a), len(a))]
        xb = [b[i] for i in rng.integers(0, len(b), len(b))]
        ma = np.concatenate(xa).mean()
        mb = np.concatenate(xb).mean()
        out.append(ma - mb)
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def _stretches(df: pd.DataFrame, col: str = "cls") -> dict:
    """Consecutive HALF months with the same label -> {label: [array of points per month, ...]}."""
    out: dict = {}
    prev_m, prev_c, cur = None, None, []
    for m, r in df.iterrows():
        if prev_m is not None and m == prev_m + 1 and r[col] == prev_c:
            cur.append(r["diff"])
        else:
            if cur:
                out.setdefault(prev_c, []).append(np.array(cur))
            cur = [r["diff"]]
        prev_m, prev_c = m, r[col]
    if cur:
        out.setdefault(prev_c, []).append(np.array(cur))
    return out


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    close, raw = data["close"], data["raw"]
    first = min(s.index[0].to_period("M") if isinstance(s.index, pd.DatetimeIndex) else s.index[0] for s in raw.values())
    months = pd.period_range(first, close.index[-1].to_period("M"), freq="M")
    G = hg._gates(raw, months)
    ma = msc._targets(close)["ma50"].values.astype(float)
    on = lambda F: bc._on_days(F, close.index + DAY)  # noqa: E731
    pos = {"ma50": ma}
    for r in ALL4:
        pos[r] = ma * (0.5 + 0.5 * on(G[RG[r]]))
    Fb, Fe = hg._buf_flag(raw["nfci"], months), hg._ease_flag(raw["nfci"], months)
    pos["BUF"], pos["EASE"] = ma * (0.5 + 0.5 * on(Fb)), ma * (0.5 + 0.5 * on(Fe))
    pos["HALF"] = ma * 0.5
    mret = {k: _monthly(close, v) for k, v in pos.items()}
    years = {k: bc._years(v, close, FIRST) for k, v in pos.items()}
    years["hold"] = bc._years(np.ones(len(close)), close, FIRST)
    me = bc._month_ends(close)
    btc = (me / me.shift(1) - 1)
    held_ma = pd.Series(np.concatenate([[0.0], np.nan_to_num(ma[:-1])]), index=close.index)
    mshare = held_ma.groupby(held_ma.index.to_period("M")).mean()
    hm = pd.period_range(FIRST_M, me.index[-1], freq="M")          # held months, finished
    num = {g: _numbers(g, raw[g], months) for g in G3}
    mism = {g: int(((num[g]["flag"].fillna(-1) != G[g].fillna(-1))).sum()) for g in G3}

    # price context at the decision (end of the month before the held month)
    hi365 = close.rolling(365, min_periods=200).max()
    turns = sbt._turns(close)

    def ctx(h: pd.Period) -> str:
        d = (h - 1).to_timestamp(how="end").normalize()
        d = close.index[close.index.searchsorted(d, side="right") - 1]
        dd = float(close[d] / hi365[d] - 1) if _ok(hi365.get(d)) else np.nan
        k = close.index.get_loc(d)
        r3 = float(close.iloc[k] / close.iloc[max(0, k - 91)] - 1)
        mid = h.to_timestamp() + pd.Timedelta(days=14)
        tag = ""
        for ty, word in (("bear", "走熊段中"), ("bull", "走牛段中")):
            if any(t["date"] <= mid <= t["end"] for t in turns[ty]):
                tag = f"，{word}"
        return f"離一年高點 {_p(dd, 0)}，前三月 {_p(r3, 0)}{tag}"

    def dd_at(h: pd.Period) -> float:
        d = (h - 1).to_timestamp(how="end").normalize()
        d = close.index[close.index.searchsorted(d, side="right") - 1]
        return float(close[d] / hi365[d] - 1) if _ok(hi365.get(d)) else np.nan

    # per-rule monthly frame
    F = {}
    for r in ALL4 + ("BUF", "EASE"):
        g = RG.get(r, "nfci")
        flag = {"BUF": Fb, "EASE": Fe}.get(r, G[g] if r in ALL4 else None)
        rows = []
        for h in hm:
            f = flag.get(h - 1, np.nan)
            sh_ = float(mshare.get(h, 0.0))
            rec = {"h": h, "ma_share": sh_, "flag": f, "btc": float(btc.get(h, np.nan)),
                   "ma": float(mret["ma50"].get(h, np.nan)), "rule": float(mret[r].get(h, np.nan))}
            rec["diff"] = (rec["rule"] - rec["ma"]) * 100
            rec["half"] = bool(sh_ > 0 and f != 1)
            rec["dd"] = dd_at(h)
            rec["n_unf3"] = int(sum(G[x].get(h - 1, np.nan) != 1 for x in G3))
            rec["n_unf5"] = int(sum(G[x].get(h - 1, np.nan) != 1 for x in hg.GATES if x in G))
            if g in num and r != "H_VOTE":
                row = num[g].loc[h - 1] if (h - 1) in num[g].index else None
                for key in ("v", "avg", "gap", "z", "ds", "dl", "shock"):
                    rec[key] = float(row[key]) if row is not None and _ok(row[key]) else np.nan
                rec["cls"] = _cls(rec) if rec["half"] else ""
            else:
                rec["cls"] = ""
            rows.append(rec)
        F[r] = pd.DataFrame(rows).set_index("h")
    for r in MAIN:
        df = F[r]
        lose = df["half"] & (df["diff"] < -1e-9)

        def tax(x):
            if x["cls"] == "A":
                return "a"
            if x["cls"] == "B":
                return "b"
            if x["cls"] == "C":
                return "c" if x["btc"] > 0 else "d"
            return "e"
        df["tax"] = [tax(x) if l else "" for (_, x), l in zip(df.iterrows(), lose)]

    # daily starts (half_gate Part 4 definition)
    dstarts, dlabels = msc._month_starts(close.index, *msc.DAILY_STARTS)
    ch = close.pct_change().fillna(0.0).values

    def suffix(p):
        net, _ = te._run(np.concatenate([[np.nan], p[:-1]]), ch, bc.COST)
        r_, d_ = msc._suffix(msc._equity(net), dstarts, bc.YEAR)
        return r_[:, 0], d_[:, 0]
    sfx = {k: suffix(pos[k]) for k in ("ma50",) + ALL4 + ("BUF", "EASE")}
    rel = {r: _rel_log(close, pos[r], pos["ma50"]) for r in ALL4}
    return {"close": close, "G": G, "months": months, "num": num, "mism": mism, "F": F, "years": years,
            "btc": btc, "mshare": mshare, "hm": hm, "ctx": ctx, "sfx": sfx, "dstarts": dstarts, "dlabels": dlabels,
            "rel": rel, "pos": pos, "Fb": Fb, "Fe": Fe, "raw": raw}


# ---------- text helpers ----------

def _gw(v) -> str:
    return "有利" if v == 1 else "不利" if v == 0 else "沒資料"


def _amount(v) -> str:
    if not _ok(v) or v < 0.25:
        return "現金"
    return "持有" if v >= 0.75 else "半倉"


def _halfsum(df: pd.DataFrame) -> dict:
    h = df[df["half"]]
    b = h["btc"].values
    d = h["diff"].values
    return {"n": len(h), "up": int((b > 0).sum()), "down": int((b <= 0).sum()),
            "mean": float(b.mean()) if len(b) else np.nan, "med": float(np.median(b)) if len(b) else np.nan,
            "cost": float(d[d < 0].sum()) if len(d) else 0.0, "save": float(d[d > 0].sum()) if len(d) else 0.0}


def _sumrow(lab: str, x: dict) -> list:
    return [lab, str(x["n"]), f"{x['up']} 漲／{x['down']} 跌", _p(x["mean"]), _p(x["med"]), _pp(x["cost"]),
            _pp(x["save"]), _pp(x["cost"] + x["save"])]


def study() -> dict:
    t0 = time.time()
    data = _load()
    R = _compute(data)
    close, G, F, years, ctx, num = R["close"], R["G"], R["F"], R["years"], R["ctx"], R["num"]
    hm = R["hm"]
    rng = np.random.default_rng(BOOT_SEED)
    tables, facts = [], {}

    # ===== Part 1: HALF-month summary =====
    rows = []
    for r in ALL4:
        x = _halfsum(F[r])
        facts[r] = x
        rows.append(_sumrow(r, x))
    tables.append({"caption": (f"一、「半倉月」總表（{hm[0].strftime('%Y-%m')} 到 {hm[-1].strftime('%Y-%m')}，共 {len(hm)} 個持有月）。"
                               "半倉月＝那個月 50 日均線至少有一天說持有、而條件不利（只放一半）。比特幣當月＝月底收盤到月底收盤。"
                               "少賺＝這些月裡「規則減均線」是負的加總；少虧＝正的加總（百分點，每月簡單相加，不是複利）。"
                               "條件：H_NFCI＝NFCI 低於 52 週平均才有利；H_OECD＝OECD 美國領先指標高於 10 個月平均；"
                               "H_LQD＝LQD÷IEF 高於 252 個交易日平均；H_VOTE＝half_gate 五個條件過半數有利。"),
                   "columns": ["規則", "半倉月數", "漲／跌", "比特幣當月平均", "中位數", "合計少賺", "合計少虧", "淨效果"],
                   "rows": rows})

    # one by one (compact: one row per rule x year)
    rows = []
    for r in ALL4:
        h = F[r][F[r]["half"]]
        for y, g in h.groupby(h.index.year):
            items = "、".join(f"{m.month}月 {_p(b, 0)}" for m, b in zip(g.index, g["btc"]))
            rows.append([r, str(y), str(len(g)), items, _pp(g["diff"].sum())])
    tables.append({"caption": "一之二、每個半倉月的比特幣當月漲跌，逐一列出（依年份合併成一列）。最後一欄是那年半倉月「規則減均線」的簡單加總。",
                   "columns": ["規則", "年份", "半倉月數", "每個半倉月：比特幣當月", "合計（百分點）"], "rows": rows})

    # terciles and direction
    rows_t, rows_dir, tert_facts, dir_facts = [], [], {}, {}
    for r in MAIN:
        g = RG[r]
        h = F[r][F[r]["half"] & F[r]["gap"].notna()]
        q1, q2 = np.quantile(h["gap"], [1 / 3, 2 / 3])
        unit = "%" if g == "lqd" else ""
        for lab, m in ((f"差距最小三分之一（≤ {q1:.3f}{unit}）", h["gap"] <= q1),
                       ("中間三分之一", (h["gap"] > q1) & (h["gap"] <= q2)),
                       (f"差距最大三分之一（> {q2:.3f}{unit}）", h["gap"] > q2)):
            x = _halfsum(F[r].loc[h.index[m.values]])
            tert_facts[(r, lab[:4])] = x
            rows_t.append(_sumrow(f"{r}：{lab}", x))
        for hz, key in ((0, "ds"), (1, "dl")):
            for word, m in (("轉好中", h[key] > 0), ("還在變差", h[key] <= 0)):
                x = _halfsum(F[r].loc[h.index[m.values]])
                dir_facts[(r, hz, word)] = x
                rows_dir.append(_sumrow(f"{r}：過去 {HOR_TXT[g][hz]}{word}", x))
    tables.append({"caption": ("一之三、半倉月依「離平均多遠」分三等份（每個條件自己的單位：NFCI 指數點、OECD 指數點、LQD÷IEF 低於平均的百分比；"
                               "數字越大＝越明顯不利）。看「差一點就有利」的月份和「明顯不利」的月份結果有沒有不同。沒有資料的月份不算在內。"),
                   "columns": ["規則與分組", "月數", "漲／跌", "比特幣當月平均", "中位數", "合計少賺", "合計少虧", "淨效果"],
                   "rows": rows_t})
    tables.append({"caption": ("一之四、半倉月依「指標正在做什麼」分組：決定當時，指標比 4 週（OECD 2 個月、LQD 20 個交易日）前、"
                               "和比 8 週（OECD 3 個月、LQD 40 個交易日）前是轉好還是變差。NFCI 下降＝轉好；OECD、LQD÷IEF 上升＝轉好。"),
                   "columns": ["規則與分組", "月數", "漲／跌", "比特幣當月平均", "中位數", "合計少賺", "合計少虧", "淨效果"],
                   "rows": rows_dir})

    # compact full-history summary by year
    rows = []
    for y in sorted(years["ma50"]):
        cells = []
        for r in ALL4:
            df = F[r][F[r].index.year == y]
            cells.append(f"{int(df['half'].sum())}｜{_pp(years[r][y] - years['ma50'][y])}")
        rows.append([f"{y}" + ("（到目前）" if y == max(years["ma50"]) else ""), _pct(years["hold"].get(y)),
                     _pct(years["ma50"][y])] + cells)
    tables.append({"caption": ("一之五、全部歷史的精簡版（逐年）：一直抱著、50 日均線的日曆年報酬，和每條規則的「半倉月數｜那年減均線（百分點，"
                               "日線複利，和 half_gate 的每年表相同）」。完整的逐月數字只在下面的輸掉、贏到的那幾段列出。"),
                   "columns": ["年份", "一直抱著", "50 日均線"] + list(ALL4), "rows": rows})

    # ===== Part 2: losing stretches =====
    lose_years = {r: [y for y in sorted(years["ma50"]) if years[r][y] - years["ma50"][y] < -LOSE_EPS] for r in ALL4}
    dy = np.array([d.year for d in R["dlabels"]])
    fail_start = {}
    for r in ALL4 + ("BUF", "EASE"):
        a, b = R["sfx"][r], R["sfx"]["ma50"]
        both = (a[0] > b[0] + 1e-9) & (a[1] > b[1] + 1e-9)
        fail_start[r] = {"both": both, "less": a[0] <= b[0] + 1e-9, "deeper": a[1] <= b[1] + 1e-9,
                         "years": [int(y) for y in sorted(set(dy)) if both[dy == y].sum() < (dy == y).sum() / 2]}

    def other_txt(g: str, h: pd.Period) -> str:
        return "、".join(f"{GN[x]}{_gw(G[x].get(h - 1, np.nan))}" for x in G3 if x != g)

    def month_row(r: str, h: pd.Period) -> list:
        g = RG[r]
        x = F[r].loc[h]
        pos_txt = "半倉" if x["half"] else ("持有" if x["ma_share"] > 0 else "現金")
        return [h.strftime("%Y-%m"), f"{x['ma_share'] * 100:.0f}%｜{pos_txt}", _numtxt(g, x), _chgtxt(g, x),
                (TAX_NAME[x["tax"]] if x.get("tax") else CLS_NAME.get(x["cls"], "—")) if x["half"] else "—",
                f"{_p(x['btc'], 0)}｜均線 {_p(x['ma'], 0)}", _pp(x["diff"]), other_txt(g, h), ctx(h)]

    cols_m = ["持有月", "均線持有天數｜部位", "指標（自己的平均）差距、z", "指標在做什麼", "分類", "比特幣當月｜均線當月",
              "規則減均線", "另外兩個條件", "決定時的價格位置"]
    tax_count = {r: {} for r in MAIN}
    stretch_rows = []
    for r in MAIN:
        df = F[r]
        stretches = [("年", y) for y in lose_years[r]]  # failing start years: Part 4 (their gap opens in later years)
        detail = []
        for kind, y in stretches:
            sub = df[(df.index.year == y) & df["half"]]
            for h in sub.index:
                detail.append([f"{r}｜{kind} {y}"] + month_row(r, h)[:-1])
                # the last column (price context) is moved to its own table to keep width <= 9
            l_ = sub[sub["diff"] < -1e-9]
            cnt = {k: (int((l_["tax"] == k).sum()), float(l_.loc[l_["tax"] == k, "diff"].sum())) for k in "abcde"}
            if kind == "年":
                for k, (n_, p_) in cnt.items():
                    c0 = tax_count[r].setdefault(k, [0, 0.0])
                    c0[0] += n_
                    c0[1] += p_
            yd = years[r][y] - years["ma50"][y]
            stretch_rows.append([r, f"{kind} {y}", _pp(yd), f"{len(sub)}｜{len(l_)}",
                                 "；".join(f"{TAX_NAME[k][:3]} {n_} 個 {_pp(p_)}" for k, (n_, p_) in cnt.items() if n_)
                                 or "沒有輸的半倉月",
                                 f"{int((sub['n_unf3'] == 3).sum())}／{int((sub['n_unf3'] == 2).sum())}／{int((sub['n_unf3'] == 1).sum())}"])
        tables.append({"caption": (f"二之{MAIN.index(r) + 1}、{r} 輸掉的那幾段，逐月（只列半倉月；不是半倉的月份規則和均線一樣，差距只有成本）。"
                                   "「年」＝那個日曆年規則比均線少賺超過 0.05 個百分點。「起算年」做不到的原因多半在之後的年份，放在第四部分。"
                                   "z＝差距除以指標自己在同一段平均期間的標準差（波動大小），z 越大越明顯不利。"
                                   "分類只用指標判斷（A 差一點、B 衝擊後平均跟不上、C 明顯不利、E 沒資料）；輸掉的月份再依比特幣當月細分成 (a)–(e)。"
                                   "價格位置在下一張表。"),
                       "columns": ["哪一段"] + cols_m[:-1], "rows": detail})
    tables.append({"caption": ("二之四、每一段輸掉的地方，用事先寫好的分類數：(a) 差一點＝差距不到指標自己波動的 0.25 倍；(b) 衝擊後平均跟不上＝"
                               "指標正在轉好，但最近半段平均期間裡出現過比現在差 1 個標準差以上的值；(c) 真的偏緊、價格照漲＝明顯不利而且比特幣當月上漲；"
                               "(d) 明顯不利、比特幣當月下跌或持平，但均線持有的那幾天是漲的，所以半倉只少賺；(e) 其他或沒有資料。"
                               "第四欄：半倉月數｜其中規則輸均線的月數。最後一欄：這段半倉月裡三個條件「3 個／2 個／1 個」不利的月數。"),
                   "columns": ["規則", "哪一段", "那年規則減均線", "半倉月｜輸的月", "輸的月份分類：個數、合計百分點", "三個條件幾個不利（3／2／1）"],
                   "rows": stretch_rows})
    rows = []
    for r in MAIN:
        for k in "abcde":
            n_, p_ = tax_count[r].get(k, [0, 0.0])
            rows.append([r, TAX_NAME[k], str(n_), _pp(p_)])
    tables.append({"caption": "二之五、所有「輸掉的日曆年」合計：每個分類的輸掉月數和少賺的百分點（每月簡單加總）。",
                   "columns": ["規則", "分類", "輸的月數", "合計少賺（百分點）"], "rows": rows})
    # price context table for the losing months
    rows = []
    for r in MAIN:
        df = F[r]
        for y in lose_years[r]:
            for h in df[(df.index.year == y) & df["half"] & (df["diff"] < -1e-9)].index:
                rows.append([r, h.strftime("%Y-%m"), _pp(df.loc[h, "diff"]), ctx(h)])
    tables.append({"caption": ("二之六、輸掉的月份，決定當時價格在哪裡：月底收盤離過去 365 天最高點多遠、之前三個月漲跌，"
                               "以及持有月是不是落在 signs_before_turns 定義的「走熊」（之後跌 50% 以上）或「走牛」（之後漲 100% 以上）段落裡"
                               "（段落的起點和終點是事後才知道的，只用來描述，當時看不到）。"),
                   "columns": ["規則", "持有月", "規則減均線", "價格位置"], "rows": rows})

    # ===== Part 3: winning stretches and the cross-table =====
    for r in MAIN:
        df = F[r]
        detail = []
        for y in (2018, 2022):
            for h in df[(df.index.year == y) & df["half"]].index:
                detail.append([f"{r}｜{y}"] + month_row(r, h)[:-1])
        tables.append({"caption": f"三之{MAIN.index(r) + 1}、{r} 贏的兩年（2018、2022）的半倉月，用同樣的欄位。",
                       "columns": ["哪一段"] + cols_m[:-1], "rows": detail})
    rows, comp_rows, cls_facts = [], [], {}
    for r in MAIN:
        h = F[r][F[r]["half"]]
        st = _stretches(h)
        for c in "ABCE":
            sub = h[h["cls"] == c]
            if not len(sub):
                continue
            w, l_ = sub[sub["diff"] > 1e-9], sub[sub["diff"] < -1e-9]
            cls_facts[(r, c)] = (len(sub), int((sub["btc"] > 0).sum()), float(sub["diff"].sum()), len(st.get(c, [])))
            rows.append([r, CLS_NAME[c], str(len(sub)), f"{int((sub['btc'] > 0).sum())}/{len(sub)}", _p(sub["btc"].mean()),
                         _pp(sub["diff"].sum()), f"{len(w)}｜{_pp(w['diff'].sum())}", f"{len(l_)}｜{_pp(l_['diff'].sum())}",
                         str(len(st.get(c, []))) + f"（{', '.join(sorted({str(m.year) for m in sub.index}))}）"])
        ab = st.get("A", []) + st.get("B", [])
        cc = st.get("C", [])
        ci = _cluster_ci(cc, ab, rng)
        m_c = float(np.concatenate(cc).mean()) if cc else np.nan
        m_ab = float(np.concatenate(ab).mean()) if ab else np.nan
        cls_facts[(r, "cmp")] = (m_c, m_ab, ci, len(cc), len(ab))
        comp_rows.append([r, f"{_pp(m_c)}（{sum(len(x) for x in cc)} 個月，{len(cc)} 段）",
                          f"{_pp(m_ab)}（{sum(len(x) for x in ab)} 個月，{len(ab)} 段）",
                          _pp(m_c - m_ab) if _ok(m_c) and _ok(m_ab) else "—",
                          f"{_pp(ci[0])} 到 {_pp(ci[1])}" if ci else "段數太少，不算"])
    tables.append({"caption": ("三之四、所有半倉月依指標分類（只看指標，沒有用價格）的對照表。漲的月＝比特幣當月上漲。"
                               "「賺到」＝規則贏均線的月數｜合計百分點；「輸掉」＝規則輸均線的月數｜合計百分點。"
                               "最後一欄：這一類由幾段「連續、同一類」的月份組成（括號是年份）——段數才是比較接近獨立的證據數量，月數不是。"
                               "OECD 是很平滑的月資料，差距相對自己的波動很少小於 0.25 倍、也很少在低於平均時轉好，所以它的半倉月全部落在 C，"
                               "這個分類對 OECD 沒有分辨力。"),
                   "columns": ["規則", "指標分類", "月數", "漲的月", "比特幣當月平均", "合計規則減均線", "賺到的月", "輸掉的月", "幾段（年份）"],
                   "rows": rows})
    tables.append({"caption": (f"三之五、類別之間的比較：C（明顯不利）對 A＋B（差一點、或衝擊後平均跟不上）每個半倉月平均「規則減均線」。"
                               f"90% 區間用整段重抽（每次從每一類的段落裡有放回地抽同樣段數，{BOOT:,} 次，固定種子），"
                               "因為同一段裡的月份不是互相獨立的。區間跨過 0＝資料分不出兩類有沒有差別。"),
                   "columns": ["規則", "C 每月平均", "A＋B 每月平均", "差（C 減 A＋B）", "90% 區間"], "rows": comp_rows})

    # ----- ADDED AFTER SEEING THE RESULTS (labelled in the captions) -----
    rows, post_facts = [], {}
    for r in MAIN:
        h = F[r][F[r]["half"] & F[r]["dl"].notna()].copy()
        h["dir"] = np.where(h["dl"] > 0, "up", "down")
        st = _stretches(h, "dir")
        im, wo = st.get("up", []), st.get("down", [])
        ci = _cluster_ci(im, wo, rng)
        m_i = float(np.concatenate(im).mean()) if im else np.nan
        m_w = float(np.concatenate(wo).mean()) if wo else np.nan
        post_facts[(r, "dir")] = (m_i, m_w, ci, len(im), len(wo), sum(len(x) for x in im), sum(len(x) for x in wo))
        rows.append([r, HOR_TXT[RG[r]][1], f"{_pp(m_i)}（{sum(len(x) for x in im)} 個月，{len(im)} 段）",
                     f"{_pp(m_w)}（{sum(len(x) for x in wo)} 個月，{len(wo)} 段）",
                     _pp(m_i - m_w) if _ok(m_i) and _ok(m_w) else "—", f"{_pp(ci[0])} 到 {_pp(ci[1])}" if ci else "段數太少，不算"])
    tables.append({"caption": ("三之六（看過結果之後才加的）：一之四看到 NFCI「過去 8 週已經轉好」的半倉月合計少賺比較多，"
                               "所以補一個和三之五同樣做法的比較：已經轉好 對 還在變差，每個半倉月平均「規則減均線」，整段重抽的 90% 區間。"
                               "這是看了結果才問的問題，區間只能當描述，不能當證據。"),
                   "columns": ["規則", "看多久的變化", "已經轉好：每月平均", "還在變差：每月平均", "差（轉好 減 變差）", "90% 區間"],
                   "rows": rows})
    rows = []
    for r in MAIN:
        h = F[r][F[r]["half"] & F[r]["dd"].notna()]
        for lab, m in (("離一年高點 40% 以上（深跌之後）", h["dd"] <= -0.40), ("離一年高點 10%–40%", (h["dd"] > -0.40) & (h["dd"] < -0.10)),
                       ("離一年高點 10% 以內", h["dd"] >= -0.10)):
            sub = h[m.values]
            if not len(sub):
                continue
            post_facts[(r, "dd", lab[:6])] = (len(sub), int((sub["btc"] > 0).sum()), float(sub["diff"].sum()))
            rows.append([r, lab, str(len(sub)), f"{int((sub['btc'] > 0).sum())}/{len(sub)}", _p(sub["btc"].mean()),
                         f"{int((sub['diff'] > 1e-9).sum())}｜{_pp(sub.loc[sub['diff'] > 1e-9, 'diff'].sum())}",
                         f"{int((sub['diff'] < -1e-9).sum())}｜{_pp(sub.loc[sub['diff'] < -1e-9, 'diff'].sum())}",
                         ", ".join(sorted({str(x.year) for x in sub.index}))])
    tables.append({"caption": ("三之七（看過結果之後才加的）：二之六看到很多輸掉的月份是在價格離一年高點很遠（深跌之後）時，"
                               "所以把「全部」半倉月（贏的和輸的一起）依決定時離一年高點多遠分組，看贏的月份是不是也一樣在那裡。"
                               "只描述，不是證據。"),
                   "columns": ["規則", "決定時價格位置", "月數", "漲的月", "比特幣當月平均", "賺到的月", "輸掉的月", "年份"],
                   "rows": rows})

    # ===== Part 4: failing start years =====
    rows4, rows4b, start_facts = [], [], {}
    for r in MAIN + ("H_VOTE",):
        fs = fail_start[r]
        for y in fs["years"]:
            m = dy == y
            n = int(m.sum())
            a_, b_ = R["sfx"][r], R["sfx"]["ma50"]
            same = np.abs(a_[1][m] - b_[1][m]) < 0.05
            less = int(fs["less"][m].sum())
            deeper = int((fs["deeper"][m] & ~same).sum())
            n_same = int(same.sum())
            both_fail = int((fs["less"][m] & fs["deeper"][m] & ~same).sum())
            start_facts[(r, y)] = (n, less, deeper, n_same, both_fail)
            # gap path for the first start of the year
            s0 = int(R["dstarts"][np.where(m)[0][0]])
            rl = R["rel"][r].iloc[s0:]
            by_year = rl.groupby(rl.index.year).sum()
            by_month = rl.groupby(rl.index.to_period("M")).sum()
            worst3 = by_month.nsmallest(3)
            tot = float(rl.sum())
            neg = float(by_month[by_month < 0].sum())
            dr_, pk_r, tr_r = _deep_window(close, R["pos"][r], s0)
            dm_, pk_m, tr_m = _deep_window(close, R["pos"]["ma50"], s0)
            start_facts[(r, y, "path")] = (float(np.expm1(tot) * 100), [(p.strftime("%Y-%m"), float(np.expm1(v) * 100))
                                                                          for p, v in worst3.items()],
                                           float(worst3.sum()) / neg if neg < 0 else None,
                                           [int(yy) for yy, v in by_year.items() if v < -0.005])
            rows4.append([r, str(y), f"{n - int(fs['both'][m].sum())}/{n}", f"{less}/{n}", f"{deeper}/{n}｜{n_same}/{n}",
                          f"{both_fail}/{n}",
                          f"{np.expm1(tot) * 100:+.1f}%",
                          "、".join(f"{yy} {np.expm1(v) * 100:+.0f}%" for yy, v in by_year.items() if abs(v) > 0.005)])
            rows4b.append([r, str(y), "、".join(f"{p.strftime('%Y-%m')} {np.expm1(v) * 100:+.1f}%" for p, v in worst3.items()),
                           f"{float(worst3.sum()) / neg * 100:.0f}%" if neg < 0 else "—",
                           f"{dr_:.1f}%（{_d(pk_r)} → {_d(tr_r)}）", f"{dm_:.1f}%（{_d(pk_m)} → {_d(tr_m)}）"])
    tables.append({"caption": ("四之一、「錢更多而且最深虧損更淺」做不到一半的起算年：每個起算月（每月 1 日，比特幣 UTC 0 點日線）是「錢比較少」、"
                               "「最深虧損比較深」、還是兩個都有。「虧損比較深｜一樣深」：一樣深＝兩者最深虧損相差不到 0.05 個百分點"
                               "（最深的那次跌發生在規則和均線持有一樣多的期間，半倉沒有機會幫忙）。"
                               "後兩欄是那年第一個起算日（1 月 1 日）起：規則的錢最後比均線多或少幾 %，"
                               "以及這個差距在每個日曆年各開了多少（複利拆開的近似）。"),
                   "columns": ["規則", "起算年", "做不到的起算月", "錢比較少", "虧損比較深｜一樣深", "錢少又比較深",
                               "1 月 1 日起：最後差距", "差距在哪一年打開"],
                   "rows": rows4})
    tables.append({"caption": ("四之二、同一個 1 月 1 日起算：讓差距變大最多的三個月（規則的錢相對均線少了幾 %），這三個月佔所有「變差月份」合計的比例"
                               "（比例高＝一兩次事件；低＝很多小次加起來），以及規則和均線各自的最深虧損（高點日 → 低點日）。"),
                   "columns": ["規則", "起算年", "差距變大最多的三個月", "三個月佔變差合計", "規則最深虧損", "均線最深虧損"],
                   "rows": rows4b})

    # ===== Part 5: three gates side by side =====
    dm = pd.period_range(FIRST_M - 1, hm[-1] - 1, freq="M")
    T = pd.DataFrame({g: G[g].reindex(dm) for g in G3})
    have = T.notna().all(axis=1)
    Tn = T[have]
    nunf = (Tn == 0).sum(axis=1)
    agree_rows = [["三個都有利", str(int((nunf == 0).sum()))], ["三個都不利", str(int((nunf == 3).sum()))],
                  ["兩個不利、一個有利", str(int((nunf == 2).sum()))], ["一個不利、兩個有利", str(int((nunf == 1).sum()))],
                  ["有沒資料的月份（不算）", str(int((~have).sum()))]]
    pair = []
    for a, b in (("nfci", "oecd"), ("nfci", "lqd"), ("oecd", "lqd")):
        pair.append(f"{GN[a]}–{GN[b]} {int((Tn[a] == Tn[b]).sum())}/{len(Tn)}")
    agree_rows.append(["兩兩相同的月份", "；".join(pair)])
    tables.append({"caption": (f"五之一、三個條件每月的決定擺在一起（{dm[0].strftime('%Y-%m')} 到 {dm[-1].strftime('%Y-%m')} 的決定月，"
                               f"三個都有資料的 {len(Tn)} 個月）。"), "columns": ["情況", "月數"], "rows": agree_rows})
    rows, agree_facts = [], {}
    halfm = _monthly(close, R["pos"]["HALF"])
    for k in (0, 1, 2, 3):
        hs = [h for h in hm if R["mshare"].get(h, 0) > 0 and (h - 1) in Tn.index and int(nunf[h - 1]) == k]
        if not hs:
            continue
        b = np.array([float(R["btc"][h]) for h in hs])
        d = np.array([(float(halfm[h]) - float(F["H_NFCI"].loc[h, "ma"])) * 100 for h in hs])
        agree_facts[k] = (len(hs), int((b > 0).sum()), float(b.mean()), float(d.sum()))
        rows.append([f"{k} 個不利", str(len(hs)), f"{int((b > 0).sum())}/{len(hs)}（{(b > 0).mean() * 100:.0f}%）", _p(b.mean()),
                     _p(float(np.median(b))), _pp(d.sum()), _pp(d.mean())])
    tables.append({"caption": ("五之二、均線至少有一天持有的月份，依三個條件裡幾個不利分組：比特幣當月、以及「如果那個月放一半」相對均線的百分點"
                               "（不分哪個條件，只看放一半的效果；每月簡單加總）。看「三個都說不利」和「只有一個說不利」的月份表現有沒有不同。"),
                   "columns": ["三個條件裡", "月數", "漲的月", "比特幣當月平均", "中位數", "放一半合計減均線", "每月平均"], "rows": rows})
    rows = []
    for r in MAIN:
        l_ = F[r][F[r]["half"] & (F[r]["diff"] < -1e-9)]
        w_ = F[r][F[r]["half"] & (F[r]["diff"] > 1e-9)]
        rows.append([r, f"{len(l_)}", " ／ ".join(f"{k} 個：{int((l_['n_unf3'] == k).sum())}" for k in (3, 2, 1)),
                     f"{len(w_)}", " ／ ".join(f"{k} 個：{int((w_['n_unf3'] == k).sum())}" for k in (3, 2, 1))])
    tables.append({"caption": "五之三、每條規則輸掉的半倉月和贏到的半倉月，當時三個條件裡有幾個同時不利。",
                   "columns": ["規則", "輸的半倉月", "其中 3／2／1 個不利", "贏的半倉月", "其中 3／2／1 個不利"], "rows": rows})

    # ===== Part 6: BUF / EASING flips outside 2020 =====
    rows, flip_facts = [], {}
    s_n = R["raw"]["nfci"]
    for key, lab, Fv in (("BUF", "BUF", R["Fb"]), ("EASE", "EASING", R["Fe"])):
        cnt, tot = 0, 0.0
        up = 0
        for h in hm:
            if h.year == 2020:
                continue
            a0, a1 = G["nfci"].get(h - 1, np.nan), Fv.get(h - 1, np.nan)
            if (a0 == 1) == (a1 == 1):
                continue
            x = F["H_NFCI"].loc[h]
            dv = (float(F[key].loc[h, "rule"]) - float(x["rule"])) * 100
            if x["ma_share"] > 0:
                cnt += 1
                tot += dv
                up += int(x["btc"] > 0)
            w = nf._used_week(s_n, h - 1)
            k8 = s_n.index.get_loc(w) - hg.EASE_W if w is not None else -1
            rows.append([lab, h.strftime("%Y-%m"), f"{_gw(a0)} → {_gw(a1)}", _numtxt("nfci", x),
                         f"{float(s_n.iloc[k8]):.3f}" if k8 >= 0 else "—", f"{x['ma_share'] * 100:.0f}%",
                         _p(x["btc"], 0), _pp(dv)])
        flip_facts[key] = (cnt, up, tot)
    tables.append({"caption": ("六、half_gate 第五部分的兩個探索寫法（看過 2020 才設計的），在 2020 以外哪些月份改變了 H_NFCI 的決定。"
                               "BUF＝高出平均 0.05 以上才算不利；EASING＝低於平均、或比 8 週前低，就算有利。只列出、不評斷。"
                               "最後一欄＝那個月改寫法後的規則減原本 H_NFCI（百分點）；均線那個月沒有持有時差距是 0。"),
                   "columns": ["寫法", "持有月", "原本 → 改寫後", "NFCI（52 週平均）差距、z", "8 週前的 NFCI", "均線持有天數",
                               "比特幣當月", "改寫後減原本"], "rows": rows})

    # ===== verdict, ideas, today (written after seeing the results; see notes) =====
    verdict, ideas = _texts(R, facts, tert_facts, dir_facts, cls_facts, tax_count, lose_years, fail_start, start_facts,
                            agree_facts, flip_facts, post_facts)
    tables.append({"caption": ("七、留給之後「事先寫好再測」的候選想法（看過結果之後才寫的；全部是推測，不是測過的規則）。"
                               "最後一欄說明它是不是看了這些輸掉的月份才想到的，因此需要哪些沒看過的資料來測。"),
                   "columns": ["候選想法（推測）", "是不是看了輸掉的月份才想到", "可以拿什麼沒看過的資料來測"], "rows": ideas})

    last = close.index[-1]
    cur = (last + DAY).to_period("M") - 1
    pm = R["pos"]
    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
            "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
           {"label": "比特幣：50 日均線｜H_NFCI", "value": f"{_amount(pm['ma50'][-1])}｜{_amount(pm['H_NFCI'][-1])}"},
           {"label": "比特幣：H_OECD｜H_LQD｜H_VOTE",
            "value": "｜".join(_amount(pm[k][-1]) for k in ("H_OECD", "H_LQD", "H_VOTE"))}]
    for g in G3:
        row = num[g].loc[cur] if cur in num[g].index else None
        f = G[g].get(cur, np.nan)
        if row is None or not _ok(row["v"]):
            now.append({"label": f"{GN[g]} 條件（{cur.strftime('%Y-%m')} 月底的決定）", "value": f"{_gw(f)}；沒有數字"})
            continue
        rec = {k: row[k] for k in ("v", "avg", "gap", "z", "ds", "dl", "shock")}
        cl = CLS_NAME[_cls(rec)] if f != 1 else "有利，不分類"
        now.append({"label": f"{GN[g]} 條件（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
                    "value": f"{_gw(f)}；{_numtxt(g, rec)}；{_chgtxt(g, rec)}；分類：{cl}"})
    now.append({"label": "這幾格的意思", "value": "只描述現在的狀態和指標最近在做什麼（描述，不是預測），也不是買賣建議"})

    notes = _notes(data, R, t0)
    sources = [
        {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": nf.CHI_URL} if data["where"]["nfci"] != "FRED"
        else {"name": "FRED：NFCI", "url": nf.FRED_URL},
        ({"name": "FRED：OECD 美國綜合領先指標", "url": bc.FRED_CSV.format(bc.FRED_ID["oecd"])} if data["where"]["oecd"] == "FRED"
         else {"name": "OECD 綜合領先指標（SDMX）", "url": bc.OECD_URL}),
        {"name": "OFR 金融壓力指數資料（CSV，只用在 H_VOTE）", "url": nfs.OFR_URL},
    ] + [{"name": f"Yahoo Finance 公開日線：{sym}", "url": hg.ro.YAHOO.format(sym)} for sym in hg.YAHOO_SYMS] + [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://api.blockchain.info/charts/market-price"},
        {"name": "幣安現貨 BTCUSDT 日線", "url": "https://api.binance.com/api/v3/klines"},
    ]
    return {"id": ID, "title": "半倉規則輸在哪裡：逐月攤開",
            "question": ("half_gate 找到的三條半倉規則（H_NFCI、H_OECD、H_LQD）大約只有一半多的起算點「錢更多又虧更少」，"
                         "2015、2023 年開始的人做不到，2020 年少賺很多。它到底在哪幾個月輸、為什麼條件會說不利、"
                         "輸的月份和贏的月份是不是不同種類的「不利」？這份只觀察，不改規則、不測新規則。"),
            "asof": _d(last), "verdict": verdict, "now": now, "tables": tables, "notes": notes, "sources": sources}


def _texts(R, facts, tert, dirf, clsf, taxc, lose_years, fail_start, start_facts, agree, flips, post):
    """Verdict and candidate ideas. Written AFTER looking at the results; every number is still computed here."""
    F = R["F"]
    parts = ["觀察研究：沒有測新規則，沒有過關標準，下面只把輸的地方攤開。"]
    # 1. HALF months
    parts.append("量到的（半倉月）：" + "；".join(
        f"{r} {facts[r]['n']} 個，{facts[r]['up']} 漲 {facts[r]['down']} 跌，少賺合計 {_pp(facts[r]['cost'])}、少虧合計 {_pp(facts[r]['save'])}"
        for r in ALL4) + "（每月簡單加總）。也就是說，半倉月裡上漲的月份比下跌的多。")
    # 2. taxonomy of losing years
    tx = []
    for r in MAIN:
        n = {k: taxc[r].get(k, [0, 0.0]) for k in "abcde"}
        tot_n = sum(v[0] for v in n.values())
        tot_p = sum(v[1] for v in n.values())
        tx.append(f"{r} 輸的日曆年 {'、'.join(map(str, lose_years[r]))}，輸的月 {tot_n} 個、合計 {_pp(tot_p)}，"
                  f"其中 (c)「明顯不利、價格照漲」{n['c'][0]} 個 {_pp(n['c'][1])}、(a) 差一點 {n['a'][0]} 個 {_pp(n['a'][1])}、"
                  f"(b) 衝擊後平均跟不上 {n['b'][0]} 個 {_pp(n['b'][1])}、(d) {n['d'][0]} 個、(e) {n['e'][0]} 個")
    parts.append("量到的（輸掉的那幾年，事先寫好的分類）：" + "；".join(tx) + "。輸的大部分是 (c)：條件明顯偏緊，比特幣照樣上漲。")
    # 2020 detail for NFCI under the pre-written definition
    f20 = F["H_NFCI"]
    c20 = {m.strftime("%m"): f20.loc[m, "cls"] for m in f20.index if m.year == 2020 and f20.loc[m, "half"]}
    parts.append("量到的：照事先寫好的定義，H_NFCI 2020 年的半倉月分類是 "
                 + "、".join(f"{k} 月 {v}" for k, v in c20.items())
                 + "。4、5 月在決定當時，NFCI 比 8 週前還高（衝擊還在進行），所以被分到 C 而不是 B；"
                   "half_gate 說的「平均跟不上」在這兩個月用這個定義抓不到。決定當時的數字分不出「衝擊還沒結束」和「平均落後」這兩種說法，這裡不選邊。")
    # 3. winning vs losing classes
    wl = []
    for r in ("H_NFCI", "H_LQD"):
        h = F[r][F[r]["half"]]
        seg = []
        for c in "ABC":
            s_ = h[h["cls"] == c]
            w, l_ = s_[s_["diff"] > 1e-9], s_[s_["diff"] < -1e-9]
            seg.append(f"{c} 贏 {len(w)} 個 {_pp(w['diff'].sum())}／輸 {len(l_)} 個 {_pp(l_['diff'].sum())}")
        m_c, m_ab, ci, nc, nab = clsf[(r, "cmp")]
        wl.append(f"{r}：" + "、".join(seg) + (f"；C 減 A＋B 每月 {_pp(m_c - m_ab)}，90% 區間 {_pp(ci[0])} 到 {_pp(ci[1])}"
                                               f"（{nc} 段對 {nab} 段）" if ci else ""))
    parts.append("量到的（贏和輸是不是不同種類的「不利」）：" + "；".join(wl)
                 + "。H_OECD 的半倉月全部是 C。省到錢的月份和少賺的月份落在同樣的類別裡，區間都跨過 0，"
                   "所以這個分類沒有分出哪一種「不利」比較會少賺——不是說沒有差別，是這些資料分不出來。")
    # 4. start years
    sf_ = []
    for r, y in ((r, y) for r in MAIN for y in fail_start[r]["years"] if y in (2015, 2023)):
        n, less, deeper, same, _ = start_facts[(r, y)]
        tot, worst, share, negy = start_facts[(r, y, "path")]
        sf_.append(f"{r} {y} 年起算：{less}/{n} 錢比較少、{deeper}/{n} 虧損比較深、{same}/{n} 虧損一樣深；1 月 1 日起最後差 {tot:+.1f}%，"
                   f"差距在 {'、'.join(map(str, negy))} 年拉開，差距最大的三個月 {'、'.join(f'{p} {v:+.1f}%' for p, v in worst)}，"
                   f"佔變差合計 {share * 100:.0f}%")
    parts.append("量到的（2015、2023 年起算為什麼做不到）：" + "；".join(sf_)
                 + "。沒有一個起算月是「虧得比較深」。2015 年起算做不到，全部是「賺得比較少」，而且是很多次加起來的"
                   "（三個最大的月份只佔變差合計三成左右，2018、2022 補回來但不夠）。2023 年起算分兩種："
                   "H_NFCI 是賺得比較少，主要是 2023 年 1 月一個月（比特幣 +40% 那個月只放一半）；"
                   "H_OECD、H_LQD、H_VOTE 是最深虧損「一樣深」——那次最深的跌在 2024 年，那時條件有利、兩邊持有一樣多，半倉沒有機會幫忙；"
                   "H_OECD 只有 3/12 錢比較少（年初起算的才被 2023 年上半年的半倉拖到），H_LQD、H_VOTE 在 2023–2024 年沒有半倉，錢少的差距來自 2025–2026。")
    # 5. three gates
    a = agree
    parts.append("量到的（三個條件擺在一起）：均線有持有的月份裡，三個都不利 "
                 f"{a[3][0]} 個月、上漲 {a[3][1]} 個、比特幣當月平均 {_p(a[3][2])}；只有一個不利 {a[1][0]} 個月、上漲 {a[1][1]} 個、平均 {_p(a[1][2])}；"
                 f"兩個不利 {a[2][0]} 個月、上漲 {a[2][1]} 個、平均 {_p(a[2][2])}。不利的條件越多，並沒有一路對應到越差的月份。")
    # post-hoc direction
    d = post[("H_NFCI", "dir")]
    parts.append(f"量到的，但是看過結果才問的：H_NFCI 半倉月裡，NFCI 過去 8 週已經轉好的月份平均每月 {_pp(d[0])}（{d[5]} 個月、{d[3]} 段），"
                 f"還在變差的 {_pp(d[1])}（{d[6]} 個月、{d[4]} 段），90% 區間 {_pp(d[2][0])} 到 {_pp(d[2][1])}；OECD、LQD 同方向但區間跨過 0。"
                 "這是看了這些月份才找的，不能當證據。")
    fb, fe = flips["BUF"], flips["EASE"]
    parts.append(f"量到的（只列出、不評斷）：2020 年以外，BUF 在均線有持有的月份改了 {fb[0]} 個月的決定，合計 {_pp(fb[2])}；"
                 f"EASING 改了 {fe[0]} 個月，合計 {_pp(fe[2])}（每月簡單加總）。EASING 改到的月份包括 2015 年 11、12 月、2016 年 4–6 月和 2023 年 1 月，"
                 "正是這裡輸最多的幾個月——但這些月份現在已經被看過了。")
    parts.append("推測：好處仍然主要來自 2018、2022 兩個大跌年，能當獨立證據的段落很少；輸的大部分是「條件真的偏緊、價格照漲」，"
                 "這比較像保險的成本，而不是條件寫法的小毛病。這只是推測，資料沒辦法排除其他解釋。")
    ideas = [
        ["候選一（推測）：條件不利但指標過去 8 週（OECD 3 個月、LQD 40 個交易日）已經轉好時，不減成一半。",
         "是。half_gate 的 EASING 是看 2020 年想到的；這裡又看到 NFCI「已經轉好」的半倉月少賺較多（三之六），2015、2016、2023 也被看過了。",
         "OFR 金融壓力指數、HYG÷IEF 兩個條件自己的歷史（這份沒有逐月看它們），用同一個事先寫死的「8 週轉好」定義；以及影子紀錄之後的月份。"],
        ["候選二（推測）：差距小於指標自己波動的 0.25 倍時當作有利（用標準差當緩衝，不用 NFCI 的固定 0.05）。",
         "是。2020 年 1 月是 half_gate 看到的；這裡 A 類的月份有贏有輸（H_NFCI 贏 5 個、輸 3 個），沒有一面倒。",
         "同上：OFR、HYG÷IEF 的歷史和影子紀錄之後的月份；門檻 0.25 要在測之前寫死，不能再調。"],
        ["候選三（推測）：不改規則，先把「條件不利、價格照漲」(c) 類的成本當成保險費，事先寫好之後每月要數的東西：(c) 類月數與少賺、條件說對的月數與少虧。",
         "是。這個想法完全來自這份對輸掉月份的分類。",
         "只能用影子紀錄之後的月份（每月決定一次，一年只有 12 個月、段落更少，要很多年才有結論）。"],
    ]
    return " ".join(parts), ideas


def _notes(data, R, t0) -> list:
    mism = R["mism"]
    return [
        ("事先寫好的定義（照原文，跑之前寫好）：HALF month = ma50 held on >= 1 day AND gate unfavorable. Class A: 0 <= z < 0.25; "
         "B: not A, improving over 8 weeks / 3 months / 40 trading days, and the worst value in the last 26 weeks / 5 months / "
         "126 trading days is >= 1 std worse than now; C: the rest with data; E: no value. Losing-month taxonomy: (a)=A, (b)=B, "
         "(c)=C and BTC month up, (d)=C and BTC month down or flat but the half position cost, (e)=E/other. Losing year: rule "
         "year return < ma50's by more than 0.05 points; failing start year: 'more AND shallower' in fewer than half of its "
         "daily starts. Between-class interval: cluster bootstrap over stretches, 2,000 draws, 5-95%."),
        ("規則和資料完全沿用 half_gate：條件在 M 月最後一根日線收盤時決定、M+1 月使用，沒資料當不利；均線說持有時，條件有利放 100%、不利放 50%；"
         "成本 0.10%。每年的數字和 half_gate 的每年表用同一個算法。這裡沒有跑完整考試（每小時收盤、一般人做法、其他幣），"
         "所以「起算年」只用比特幣 UTC 0 點日線的起算月。"),
        (f"自我檢查：從「指標值和自己的平均」重算的有利／不利，和 half_gate 的條件比對，NFCI {mism['nfci']} 個月不同、"
         f"OECD {mism['oecd']} 個月不同、LQD {mism['lqd']} 個月不同（0＝表上的數字就是條件真正用的數字）。"),
        ("限制：(1) 好處主要來自 2018、2022 兩個大跌年，這段歷史只有這兩個；(2) 指標都是今天的修正後版本（NFCI、OECD 會改過去的數字），"
         "當時看到的數字可能不同，「差一點」的月份尤其可能翻面；(3) 每月決定一次，2015 年起大約 140 個持有月，而連續、同一類的段落少得多，"
         "能當作獨立證據的次數很少；(4) 三個條件同屬「金融情勢／景氣」一家，彼此重疊，不是三個獨立的測試；"
         "(5) 每月的百分點是簡單加總，和日曆年的複利數字不會完全相等；(6) 分類的門檻（0.25 個標準差、1 個標準差、半段平均期間）"
         "是跑之前定的，沒有試別的門檻，換門檻分類會變。"),
        ("沒有放進來的：sentiment_turns（恐懼貪婪指數 2018 年起才有）、pressure_before_moves 的鏈上和交易所壓力資料，"
         "這次只用價格位置、走熊／走牛段落和另外兩個條件描述「當時還發生什麼」，沒有另外抓這些資料。"),
        (f"資料：NFCI 來自{data['where']['nfci']}；OECD 來自{data['where']['oecd']}；OFR 來自{data['where']['ofr'] or '這次拿不到'}"
         "（只用在 H_VOTE）；HYG、IEF、LQD、SPY 來自 Yahoo Finance 公開日線（含息調整後收盤）。"
         "備份檔寫在 dashboard/cache/_store/half_gate_losses/，共用資料夾只讀。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把過去攤開來看、說明現在的狀態，不是買賣建議。",
    ] + data["problems"]


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
