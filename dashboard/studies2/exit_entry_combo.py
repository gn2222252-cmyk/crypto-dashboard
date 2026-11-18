"""Historical study "exit_entry_combo": crash_timing found that the rules that leave bitcoin's bear markets best
(20-day low, 10-day low) come back very late, and rally_timing found that the earliest re-entries come from other
rules. If a rule leaves like the best leavers and comes back like the earliest re-entries, is it better than "close
above the 50-day average" (ma50) - judged by main_signal_check's fair four-way test?

The grid, fixed before any result was looked at: one EXIT rule (used while holding) x one ENTRY rule (used in cash).
Exits: E1 close below the 50-day average; E2 close at a 20-day closing low; E3 close at a 10-day closing low;
E4 close below the 50-day average while that average is falling (5-day change < 0); E5 close 15% below the highest
close since entry. Entries: N1 close above the 50-day average; N2 close at a 55-day closing high; N3 if some close of
the last 90 days was 40%+ below the 1-year high, close above the 20-day average, otherwise N1; N4 close above the
50-day average while that average is not falling (rule 5 of rally_timing); N5 close at a 20-day closing high.
E1 x N1 is ma50 itself, E2 x N2 is 55/20.

Two joining rules, also written down before running:
- After an exit, an entry needs its condition to be false at some close since the exit and then true again ("armed"),
  so a level entry (e.g. N1 after a 15% drop that is still above the average) cannot buy straight back the next day.
  Before the first trade every rule is armed.
- E1 and E4 are average-based exits; a position bought while the close is still below the 50-day average (possible
  with N2, N3, N5) leaves on a close below the 20-day average until the close has once been above the 50-day average
  (the "early entry" latch of rally_timing). With N1 / N4 this never happens, so E1 x N1 stays exactly ma50.

Measures: crash_timing's bear-market measure (share of each 50%+ fall avoided, from the rule's own growth top -> low,
so whipsaws count; whipsaws per year), rally_timing's first entry after the low (share of the next rally caught) and
false starts, trades per year; main_signal_check's fair grid (108 daily start months, 72 start months x 24 closing
hours textbook and ordinary person with 200 simulations per clock, ETH/SOL/BNB/XRP/MSTR) and its pre-stated ranking
score. Same code pieces, start dates, clocks and random seeds (E1 x N1 and E2 x N2 use ma50's and 55/20's seeds).
Improvement test (stated in advance): a hybrid beats ma50 only if its ranking score is not lower, its median deepest
loss over the 1,728 cells is shallower and its median share of the crashes avoided is larger. Against luck from 25
tries: a 60-day block bootstrap (1,000 draws) of the best hybrid's return / volatility minus ma50's on bitcoin, and a
plateau check on the grid neighbours. No look-ahead (checked by scrambling the future). Public, key-less data only.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import rally_timing as rl
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST = msc.COST
P_COST, P_MISS, SIMS, SEED = msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
LOSS_BAR = msc.LOSS_BAR
TRAIL = 0.15
DEEP, DEEP_DAYS, SLOPE_DAYS = 0.40, 90, 5
HIGH_COST = 0.003
BLOCK, DRAWS, BOOT_SEED = 60, 1000, 20260101
PLATEAU_GAP = 0.05  # a neighbour within 5 points of the best's ranking score counts as "also good"

EXITS = ("E1", "E2", "E3", "E4", "E5")
ENTRIES = ("N1", "N2", "N3", "N4", "N5")
KEYS = tuple(f"{e}{n}" for e in EXITS for n in ENTRIES)
BASE, B5520 = "E1N1", "E2N2"
MSC_KI = {BASE: msc.CANDS.index("ma50"), B5520: msc.CANDS.index("b5520")}

EX_NAME = {"E1": "收盤跌破 50 日均線", "E2": "收盤是 20 日新低", "E3": "收盤是 10 日新低",
           "E4": "跌破 50 日均線而且均線往下", "E5": "比上車後最高收盤跌 15%"}
EN_NAME = {"N1": "收盤站上 50 日均線", "N2": "收盤是 55 日新高", "N3": "跌深 40% 後站上 20 日均線（沒跌深就等 50 日均線）",
           "N4": "站上 50 日均線而且均線不再往下", "N5": "收盤是 20 日新高"}
EX_SHORT = {"E1": "破 50 日", "E2": "20 日新低", "E3": "10 日新低", "E4": "破 50 日且下彎", "E5": "回落 15%"}
EN_SHORT = {"N1": "上 50 日", "N2": "55 日新高", "N3": "跌深上 20 日", "N4": "上 50 日且止跌", "N5": "20 日新高"}

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _short(k: str) -> str:
    tag = {BASE: "（＝50 日均線）", B5520: "（＝55／20）"}.get(k, "")
    return f"{k[:2]}×{k[2:]} {EX_SHORT[k[:2]]}下車／{EN_SHORT[k[2:]]}上車{tag}"


def _long(k: str) -> str:
    return f"下車：{EX_NAME[k[:2]]}；上車：{EN_NAME[k[2:]]}"


def _share(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v * 100:.0f}%"


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _med(vals) -> float | None:
    v = [x for x in vals if _ok(x)]
    return float(np.median(v)) if v else None


# ---------- the hybrids ----------

def _feat(c: np.ndarray, year: int = YEAR) -> dict:
    """Every input of the 10 rule halves after each close (closes up to and including that close only)."""
    s = pd.Series(np.asarray(c, dtype=float))
    w90 = max(int(round(DEEP_DAYS * year / YEAR)), 1)
    m20 = s.rolling(20, min_periods=20).mean()
    m50 = s.rolling(50, min_periods=50).mean()
    slope = m50 - m50.shift(SLOPE_DAYS)
    hi1 = s.rolling(year, min_periods=1).max()
    deep = (s <= hi1 * (1 - DEEP)).astype(float).rolling(w90, min_periods=1).max() > 0
    high55 = s.rolling(55, min_periods=55).max()
    f = {"c": s.values, "m20": m20.values, "m50": m50.values, "slope": slope.values,
         "low20": s.rolling(20, min_periods=20).min().values, "low10": s.rolling(10, min_periods=10).min().values,
         "high55": high55.values, "high20": s.rolling(20, min_periods=20).max().values, "deep": deep.values}
    f["ready"] = ~(np.isnan(f["m20"]) | np.isnan(f["m50"]) | np.isnan(f["slope"]) | np.isnan(f["high55"]))
    return f


def _conds(f: dict) -> tuple[dict, dict, np.ndarray]:
    """Exit conditions (flag = the close has been above the 50-day average since entry, no-flag version) and entry
    conditions as boolean arrays (NaN comparisons are False; `ready` gates them anyway)."""
    c, m20, m50, sl = f["c"], f["m20"], f["m50"], f["slope"]
    with np.errstate(invalid="ignore"):
        above50 = c > m50
        below20 = c < m20
        ex = {"E1": (~above50, below20), "E2": (c <= f["low20"],) * 2, "E3": (c <= f["low10"],) * 2,
              "E4": ((c < m50) & (sl < 0), below20), "E5": None}
        n1 = above50
        en = {"N1": n1, "N2": c >= f["high55"], "N3": np.where(f["deep"], c > m20, n1),
              "N4": above50 & (sl >= 0), "N5": c >= f["high20"]}
    return ex, en, above50


def _machine(f: dict, ex: dict, en: dict, above50: np.ndarray, e: str, n: str, state=None) -> tuple[np.ndarray, tuple]:
    """Position (1/0, NaN before the inputs exist) after each close, and the state after the last close."""
    c = f["c"]
    ready = f["ready"].tolist()
    a50 = above50.tolist()
    ecnd = en[n].tolist()
    trail = e == "E5"
    if not trail:
        xf, xn = ex[e][0].tolist(), ex[e][1].tolist()
    st, armed, flag, peak = state if state is not None else (0, True, False, 0.0)
    cl = c.tolist()
    out = np.full(len(cl), np.nan)
    for i in range(len(cl)):
        if not ready[i]:
            continue
        x = cl[i]
        if st:
            if x > peak:
                peak = x
            if a50[i]:
                flag = True
            if trail:
                go = x <= (1 - TRAIL) * peak
            else:
                go = xf[i] if (flag or e in ("E2", "E3")) else xn[i]
            if go:
                st, armed = 0, not ecnd[i]
        else:
            if ecnd[i] and armed:
                st, peak, flag = 1, x, a50[i]
            elif not ecnd[i]:
                armed = True
        out[i] = st
    return out, (st, armed, flag, peak)


def _wants(c, year: int = YEAR, keys=KEYS) -> pd.DataFrame:
    f = _feat(c, year)
    ex, en, a50 = _conds(f)
    return pd.DataFrame({k: _machine(f, ex, en, a50, k[:2], k[2:])[0] for k in keys})


# ---------- the fair grid (main_signal_check's pieces) ----------

def _daily(close: pd.Series, year: int, first, last, cost: float = COST, lag: int = 1) -> dict:
    target = _wants(close.values, year)
    target.index = close.index
    change = close.pct_change().fillna(0.0).values
    valid = target.notna().all(axis=1) & msc._targets(close, year).notna().all(axis=1)  # msc's own start dates
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels, "target": target, "net": {}}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in KEYS:
        net, _ = te._run(target[k].shift(lag).values, change, cost)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
        out["net"][k] = net
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, YEAR)
    hr, hd = hr[:, 0], hd[:, 0]
    wants = []
    for h in range(24):
        t = _wants(np.concatenate([pre, price[h::24]])).fillna(0.0)
        wants.append({k: t[k].values[npre:][:days].astype(float) for k in KEYS})
    out = {}
    for j, k in enumerate(KEYS):
        ki = MSC_KI.get(k, 10 + j)
        tr, td = np.zeros((len(starts), 24)), np.zeros((len(starts), 24))
        pr, pdd = np.zeros((len(starts), 24, SIMS)), np.zeros((len(starts), 24, SIMS))
        for h in range(24):
            w = wants[h][k]
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]), starts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        out[k] = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                  "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                  "c_deep_worst": float(td.min()),
                  "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
                  "p_deep_med": float(np.median(pdd))}
    return {"starts": labels, "rules": out, "hold_deep_med": float(np.median(hd))}


def _summary(daily: dict, clock: dict, other: dict) -> dict:
    s = {}
    for k in KEYS:
        dr, dd = daily[k]
        x = dict(clock["rules"][k])
        x["d_n"], x["d_beat"] = len(dr), int((dr > daily["hold"][0]).sum())
        x["d_edge"] = float(np.median(dr - daily["hold"][0]))
        x["d_shallow"] = int((dd > daily["hold"][1]).sum())
        x["o_each"] = {a: (len(o["starts"]), int((o[k][0] > o["hold"][0]).sum())) for a, o in other.items()}
        x["o_n"] = sum(v[0] for v in x["o_each"].values())
        x["o_beat"] = sum(v[1] for v in x["o_each"].values())
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        s[k] = x
    eligible = [k for k in KEYS if s[k]["loss_ok"] >= LOSS_BAR]
    order = sorted(KEYS, key=lambda k: (k not in eligible, -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "eligible": eligible, "order": order}


# ---------- robustness ----------

def _bootstrap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Share of 60-day block bootstrap draws in which a's return / volatility beats b's (same blocks for both), and
    the two actual ratios."""
    n = len(a)
    rng = np.random.default_rng(BOOT_SEED)
    k = math.ceil(n / BLOCK)
    win = 0
    for _ in range(DRAWS):
        st = rng.integers(0, n - BLOCK + 1, size=k)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        win += te._score_parts(a[idx])[2] > te._score_parts(b[idx])[2]
    return win / DRAWS, te._score_parts(a)[2], te._score_parts(b)[2]


def _variant(close: pd.Series, cost: float, lag: int) -> dict:
    d = _daily(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], cost, lag)
    out = {}
    for k in KEYS:
        edge = d[k][0] - d["hold"][0]
        out[k] = (float((edge > 0).mean()), float(np.median(edge)), float(np.median(d[k][1])))
    return out


def _lookahead(c: np.ndarray) -> tuple[int, int]:
    base = _wants(c)
    rng = np.random.default_rng(7)
    bad = cuts = 0
    for cut in (len(c) - 1500, len(c) - 900, len(c) - 400, len(c) - 60, len(c) - 5):
        noisy = c.copy()
        noisy[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(c) - cut - 1))
        alt = _wants(noisy)
        cuts += 1
        for k in KEYS:
            x, y = base[k].values[:cut + 1], alt[k].values[:cut + 1]
            if not np.array_equal(np.nan_to_num(x, nan=-1), np.nan_to_num(y, nan=-1)):
                bad += 1
    return cuts, bad


# ---------- today ----------

def _now(close: pd.Series) -> list[dict]:
    c = close.values.astype(float)
    last = float(c[-1])
    f = _feat(c)
    ex, en, a50 = _conds(f)
    tail = c[-800:]
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"}]

    for k in KEYS:
        e, n = k[:2], k[2:]
        pos, state = _machine(f, ex, en, a50, e, n)
        cur = int(np.nan_to_num(pos[-1], nan=0.0))

        def held(p: float) -> int:
            g = _feat(np.append(tail, p))
            gx, gn, ga = _conds(g)
            one = {kk: g[kk][-1:] for kk in g}
            sx = {kk: (None if v is None else (v[0][-1:], v[1][-1:])) for kk, v in gx.items()}
            sn = {kk: v[-1:] for kk, v in gn.items()}
            p1, _ = _machine(one, sx, sn, ga[-1:], e, n, state)
            return int(np.nan_to_num(p1[-1], nan=0.0))

        grid = last * np.exp(np.linspace(math.log(0.5), math.log(1.6), 45))
        states = [held(p) for p in grid]
        i0 = int(np.argmin(np.abs(grid - last)))
        best = None
        for i in range(len(grid) - 1):
            if states[i] != states[i + 1]:
                lo_p, hi_p, s_lo = grid[i], grid[i + 1], states[i]
                for _ in range(22):
                    mid = math.sqrt(lo_p * hi_p)
                    if held(mid) == s_lo:
                        lo_p = mid
                    else:
                        hi_p = mid
                level = math.sqrt(lo_p * hi_p)
                new = states[i + 1] if level > last else states[i]
                dist = abs(i + 0.5 - i0)
                if new != cur and (best is None or dist < best[0]):
                    best = (dist, level, new)
        text = "全部持有" if cur else "全部現金"
        if best:
            lv = best[1]
            text += (f"；下一根收盤{'≤' if lv < last else '≥'} 約 {lv:,.0f} 美元（離最後收盤 {(lv / last - 1) * 100:+.1f}%）"
                     f"會轉成{'持有' if best[2] else '現金'}")
        else:
            text += "；下一根收盤在 −50% 到 +60% 之間都不會改變"
        out.append({"label": _short(k), "value": text})
    out.append({"label": "這幾格的意思", "value": "只描述各規則在最後一根日線收盤後停在哪個狀態、下一根收盤到哪裡會翻轉，不是買賣建議"})
    return out


# ---------- the study ----------

def _compute() -> dict:
    close = te._daily_close()
    c = close.values.astype(float)
    daily = _daily(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre)
    other_data, problems = msc._load_other()
    if problem:
        problems.insert(0, problem)
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key, (oc, since, year) in other_data.items():
        try:
            other[key] = _daily(oc, year, since or oc.index[0], three)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            other.pop(key, None)
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")
    target = daily["target"]
    cbears = ct._bears(close)
    first = int(close.index.searchsorted(pd.Timestamp(ct.BEAR_FROM)))
    crash = {k: [ct._crash(c, [(1.0, target[k].values)], b) for b in cbears] for k in KEYS}
    whip = {k: ct._whipsaws(c, [(1.0, target[k].values)], first) for k in KEYS}
    rbears = rl._bears(close)
    entries = {k: [rl._entry(close, target[k], b) for b in rbears] for k in KEYS}
    falses = {k: rl._false_starts(close, target[k]) for k in KEYS}
    return {"close": close, "c": c, "daily": daily, "clock": clock, "other": other, "problems": problems,
            "cbears": cbears, "crash": crash, "whip": whip, "rbears": rbears, "entries": entries, "falses": falses,
            "sm": _summary(daily, clock, other), "cost": _variant(close, HIGH_COST, 1), "delay": _variant(close, COST, 2),
            "look": _lookahead(c)}


def study() -> dict:
    r = _compute()
    close, crash, whip, entries, falses = r["close"], r["crash"], r["whip"], r["entries"], r["falses"]
    sm = r["sm"]
    s, order = sm["s"], sm["order"]
    done = [i for i, b in enumerate(r["cbears"]) if b["done"]]
    n_d, n_c = s[BASE]["d_n"], s[BASE]["c_n"]

    avoid = {k: _med([crash[k][i]["avoid"] for i in done]) for k in KEYS}
    entry_eq = {k: _med([crash[k][i].get("entry_eq") for i in done]) for k in KEYS}
    caught = {k: _med([e["caught"] if e else None for e in entries[k]]) for k in KEYS}
    above = {k: _med([e["above"] if e else None for e in entries[k]]) for k in KEYS}

    tests = {}
    for k in KEYS:
        tests[k] = (s[k]["low"] >= s[BASE]["low"] - 1e-12,
                    s[k]["c_deep_med"] > s[BASE]["c_deep_med"] + 1e-9,
                    _ok(avoid[k]) and avoid[k] > avoid[BASE] + 1e-12)
    winners = [k for k in order if k != BASE and all(tests[k])]
    rank = {k: i for i, k in enumerate(order, 1)}
    best = next(k for k in order if k != BASE)  # pre-stated: the highest-ranked hybrid other than ma50 itself

    def mark(k: str) -> str:
        if k == BASE:
            return "比較的基準"
        t = tests[k]
        return ("、".join([("分數不輸" if t[0] else "分數輸"), ("虧損較淺" if t[1] else "虧損沒較淺"),
                          ("躲掉較多" if t[2] else "躲掉沒較多")]) + ("→ 勝過" if all(t) else "→ 不算"))

    # table 1: the grid of ranking scores
    rows1 = []
    for e in EXITS:
        row = [f"{e} {EX_NAME[e]}"]
        for n in ENTRIES:
            k = e + n
            x = s[k]
            row.append(f"{_share(x['low'])}（第 {rank[k]}）｜{_pct(x['c_deep_med'])}" + ("" if k not in winners else "★"))
        rows1.append(row)

    # table 2: top 8 in detail (+ ma50 and 55/20 if not among them)
    top = order[:8]
    show = top + [k for k in (BASE, B5520) if k not in top]
    rows2 = []
    for k in show:
        x = s[k]
        rows2.append([_short(k)] + [_share(v) for v in x["checks"]]
                     + [f"{_share(x['low'])}（第 {rank[k]}）" + ("" if x["loss_ok"] >= LOSS_BAR else "，虧損條件未達"),
                        f"{_pct(x['c_deep_med'])}｜{_pp(x['c_edge'])}", mark(k)])

    # table 3: crash avoided vs rally caught vs whipsaws
    by_avoid = sorted(KEYS, key=lambda k: -(avoid[k] or -9))
    by_entry = sorted(KEYS, key=lambda k: (above[k] if _ok(above[k]) else 1e9))
    keys3 = []
    for k in show + by_avoid[:3] + by_entry[:3]:
        if k not in keys3:
            keys3.append(k)
    keys3 = keys3[:16]
    rows3 = []
    for k in keys3:
        w, fs = whip[k], falses[k]
        rows3.append([_short(k), _share(avoid[k]), _pct(entry_eq[k] * 100) if _ok(entry_eq[k]) else "—",
                      (f"+{above[k]:.0f}%｜{caught[k]:.0f}%" if _ok(above[k]) and _ok(caught[k]) else "—"),
                      f"{w['whips']:.1f} 次", f"{fs['per_year']:.1f} 次", f"{fs['trades_year']:.1f} 次"])

    # table 4: robustness of the best hybrid
    bi, bn = EXITS.index(best[:2]), ENTRIES.index(best[2:])
    neigh = [best[:2] + ENTRIES[j] for j in (bn - 1, bn + 1) if 0 <= j < 5] + \
            [EXITS[j] + best[2:] for j in (bi - 1, bi + 1) if 0 <= j < 5]
    close_n = [k for k in neigh if s[k]["low"] >= s[best]["low"] - PLATEAU_GAP]
    shape = ("高原（相鄰的都差不多好）" if len(close_n) == len(neigh) else
             "孤峰（相鄰的都明顯較差）" if not close_n else f"一半一半（{len(neigh)} 個相鄰裡 {len(close_n)} 個差不多好）")
    btc0 = int(close.index.searchsorted(pd.Timestamp(msc.DAILY_STARTS[0])))
    boot, ratio_b, ratio_m = _bootstrap(r["daily"]["net"][best][btc0:], r["daily"]["net"][BASE][btc0:])
    rows4 = []
    for k in neigh:
        rows4.append([f"相鄰：{_short(k)}", f"{_share(s[k]['low'])}｜{_pct(s[k]['c_deep_med'])}",
                      f"{_share(s[BASE]['low'])}｜{_pct(s[BASE]['c_deep_med'])}",
                      "差不多好（差 5 點以內）" if k in close_n else "明顯較差"])
    rows4.append(["相鄰格的形狀", f"{len(neigh)} 個相鄰裡 {len(close_n)} 個在最好那格 5 點以內", "—", shape])
    rows4.append([f"比特幣 {msc.DAILY_STARTS[0][:4]} 年起報酬÷波動（{BLOCK} 天一段重抽 {DRAWS:,} 次）",
                  f"{ratio_b:.2f}｜{_share(boot)} 的重抽贏 50 日均線", f"{ratio_m:.2f}",
                  "接近 50% = 分不出高下；95% 以上才算明顯"])
    for a in r["other"]:
        nb, wb = s[best]["o_each"][a]
        nm, wm = s[BASE]["o_each"][a]
        rows4.append([f"其他幣：{a}（{nb} 個起算月）", f"{wb} 個贏抱著（{_share(wb / nb if nb else None)}）",
                      f"{wm} 個（{_share(wm / nm if nm else None)}）", "照課本做、成本 0.10%"])
    for key, label in (("cost", f"成本提高到 {HIGH_COST * 100:.2f}%"), ("delay", "晚一天才照做")):
        vb, vm = r[key][best], r[key][BASE]
        rows4.append([f"比特幣 {n_d} 個起算月，{label}", f"{_share(vb[0])} 贏抱著｜差距中位數 {_pp(vb[1])}",
                      f"{_share(vm[0])}｜{_pp(vm[1])}", f"最深虧損中位數 {_pct(vb[2])} 對 {_pct(vm[2])}"])
    rows4 = rows4[:16]

    tables = [
        {"caption": (f"25 種「下車方式 × 上車方式」的排名分數（main_signal_check 事先定好的：比特幣 {n_d} 個起算月、{n_c:,} 格起算月×收盤時間、"
                     f"一般人照做、其他幣四項裡「報酬贏過一直抱著」比例最低的那一項）、全部 {len(KEYS)} 種裡的名次，以及 {n_c:,} 格最深虧損中位數。"
                     "列是下車方式、欄是上車方式；左上角 E1×N1 就是 50 日均線，E2×N2 就是 55／20。★ = 事先定的三個條件都勝過 50 日均線。"),
         "columns": ["下車方式＼上車方式"] + [f"{n} {EN_SHORT[n]}" for n in ENTRIES],
         "rows": rows1},
        {"caption": (f"排名前 8 的組合（另外補上 50 日均線和 55／20 對照）。前四個數字是四項檢查各自贏過一直抱著的比例；一般人照做是收盤後 10 小時才做、"
                     f"成本 {P_COST * 100:.2f}%、一成進場漏掉，每個收盤時間 {SIMS} 次模擬（和 main_signal_check 一樣、同樣的亂數種子）。"
                     f"排名前提是最深虧損比抱著淺的比例至少 {LOSS_BAR * 100:.0f}%。最後一欄是事先定的三個條件：排名分數不輸 50 日均線、"
                     f"最深虧損中位數比 50 日均線淺、大熊市躲掉的跌幅中位數比 50 日均線多。"),
         "columns": ["組合", f"比特幣 {n_d} 個起算月", f"{n_c:,} 格（照課本）", "一般人照做", "其他幣和 MSTR",
                     "排名分數（名次）", "最深虧損中位數｜每年多賺中位數", "對 50 日均線"],
         "rows": rows2},
        {"caption": (f"比特幣 {ct.BEAR_FROM[:4]} 年以來走完的 {len(done)} 次跌 50% 以上大熊市（UTC 0 點日線，不扣成本）。躲掉的跌幅＝照規則從高點到低點"
                     "實際剩下的錢換算成「等於在哪裡一次下車」，中途被洗的損失都算進去；等效上車價同理（低點到下一個高點）。"
                     f"第一次上車＝低點之後第一次買進比低點高多少、吃到下一段上漲幾成（rally_timing 的算法，{len(r['rbears'])} 次，都是中位數）。"
                     f"被洗＝下車後 30 天內用更高價買回；假反彈＝上車後 30 天內賠錢出場；都換算成每年次數。"
                     "列出的是上表那些，加上躲最多和上車最早的各三種。"),
         "columns": ["組合", "躲掉的跌幅（中位數）", "等效上車價比低點（中位數）", "第一次上車比低點高｜吃到上漲",
                     "被洗（每年）", "假反彈（每年）", "一年進出幾次"],
         "rows": rows3},
        {"caption": (f"排名最前、而且不是 50 日均線本身的組合「{_short(best)}」經不經得起考驗（事先定好看排名第一的那個）。"
                     "相鄰＝同樣下車方式配表一左右相鄰的上車方式、同樣上車方式配上下相鄰的下車方式；格子裡是排名分數｜最深虧損中位數。"
                     f"重抽是把 {msc.DAILY_STARTS[0][:4]} 年起的每日報酬切成 {BLOCK} 天一段、兩條規則用同一批段落重抽 {DRAWS:,} 次，看它的報酬÷波動"
                     "有幾成贏 50 日均線。成本 0.30%、晚一天照做是用比特幣 108 個起算月重算。"),
         "columns": ["檢查", f"{_short(best)}", "50 日均線", "說明"],
         "rows": rows4},
    ]

    # verdict
    b, m = s[best], s[BASE]
    if winners:
        w = winners[0]
        first = (f"25 種下車×上車組合裡，有 {len(winners)} 種照事先定的三個條件勝過 50 日均線，排最前的是「{_short(w)}」："
                 f"排名分數 {_share(s[w]['low'])}（50 日均線 {_share(m['low'])}），最深虧損中位數 {_pct(s[w]['c_deep_med'])}"
                 f"（{_pct(m['c_deep_med'])}），大熊市躲掉 {_share(avoid[w])} 的跌幅（{_share(avoid[BASE])}）。")
    else:
        first = (f"25 種下車×上車組合沒有一種照事先定的三個條件勝過 50 日均線（它本身排第 {rank[BASE]}）；"
                 f"其他 24 種排最前的是「{_short(best)}」，排名分數 "
                 f"{_share(b['low'])}（50 日均線 {_share(m['low'])}），最深虧損中位數 {_pct(b['c_deep_med'])}（{_pct(m['c_deep_med'])}），"
                 f"躲掉 {_share(avoid[best])} 的跌幅（{_share(avoid[BASE])}）。")
    bx = by_avoid[0]
    second = (f"最會躲跌的「{EX_SHORT[bx[:2]]}下車／{EN_SHORT[bx[2:]]}上車」躲掉 {_share(avoid[bx])} 的跌幅，但算進被洗後等於在比低點高 "
              f"{_pct(entry_eq[bx] * 100)} 才接回（50 日均線 {_pct(entry_eq[BASE] * 100)}），排名分數只剩 {_share(s[bx]['low'])}。"
              if _ok(entry_eq[bx]) else f"最會躲跌的組合躲掉 {_share(avoid[bx])}，排名分數 {_share(s[bx]['low'])}。")
    if winners:
        third = (f"排第一的「{_short(best)}」用比特幣重抽 {DRAWS:,} 次有 {_share(boot)} 贏 50 日均線，相鄰組合是{shape.split('（')[0]}，"
                 f"{'看起來不只是運氣' if boot >= 0.95 and close_n else '分不出是不是 25 種裡運氣最好的一種'}。")
    else:
        third = (f"連排第一的組合用比特幣重抽 {DRAWS:,} 次也只有 {_share(boot)} 贏 50 日均線、相鄰組合是{shape.split('（')[0]}，"
                 "所以這次沒找到比 50 日均線更好的拼法。")
    verdict = first + second + third

    cuts, bad = r["look"]
    grid_note = "25 種完整結果（排名分數｜最深虧損中位數｜躲掉跌幅｜第一次上車比低點高｜被洗每年）：" + "；".join(
        f"{k[:2]}×{k[2:]} {_share(s[k]['low'])}｜{_pct(s[k]['c_deep_med'])}｜{_share(avoid[k])}｜"
        f"{('+' + format(above[k], '.0f') + '%') if _ok(above[k]) else '—'}｜{whip[k]['whips']:.1f}" for k in KEYS) + "。"
    notes = [
        grid_note,
        (f"核對：E1×N1 就是 50 日均線，算出排名分數 {_share(m['low'])}、{n_c:,} 格最深虧損中位數 {_pct(m['c_deep_med'])}、"
         f"每年多賺中位數 {_pp(m['c_edge'])}；E2×N2 就是 55／20，{_share(s[B5520]['low'])}、{_pct(s[B5520]['c_deep_med'])}，"
         "應該和 main_signal_check 的數字一樣。"),
        ("組合怎麼接起來（看結果前就定好）：下車後，上車條件要先「不成立」過一次、再成立才上車，免得例如回落 15% 下車後、價格還在 50 日均線上"
         "就隔天買回；用新高或跌深站上 20 日均線上車、當時還在 50 日均線下面的，在收盤站上 50 日均線之前，E1、E4 改用跌破 20 日均線下車"
         "（rally_timing 提早上車的做法）。"),
        f"一共試了 {len(KEYS)} 種組合，排第一的那種有一部分是運氣：就算每一種其實都一樣好，最好的那個看起來也會比平均好。重抽和相鄰格是用來看這點的。",
        (f"大熊市只有 {len(done)} 次走完（rally_timing 的算法算出 {len(r['rbears'])} 次），每次樣子都不同；中位數差幾個百分點不代表真的比較好。"),
        "相鄰的起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。",
        (f"成本很粗略：照課本 {COST * 100:.2f}%、一般人做法 {P_COST * 100:.2f}%，沒有滑價、稅和現金利息；熊市等效價、被洗次數不扣成本。"
         f"一般人做法每個收盤時間 {SIMS} 次模擬，和 main_signal_check 一樣。"),
        (f"不偷看未來的檢查：把 {cuts} 個切點之後的收盤價全部隨機打亂，再算一次 25 種組合，切點以前每一天的持有狀態"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 次變了，這次結果不可信。")),
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去在不同條件下做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "exit_entry_combo",
        "title": "躲跌的下車配早的上車",
        "question": ("如果用最會躲大跌的方式下車、用最早接回漲勢的方式上車，拼出來的規則會不會比 50 日均線更好——"
                     "用同樣的四項公平檢驗來判？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(close),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
