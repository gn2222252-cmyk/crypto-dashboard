"""Historical study "rule_upgrades": can the dashboard's main rule (long on a 55-day closing high, cash on a 20-day
closing low) be improved in a way that survives the same checks the plain rule went through - especially its habit of
buying back much higher than it sold - or is the plain rule already as good as it gets?

The candidate upgrades and every parameter were fixed before any result was looked at (written with today's
knowledge of the rule's weak spots):
A1 after an exit, re-enter on a 20-day high for the next 60 days (a 55-day high always works);
A2 after an exit, re-enter as soon as a close is back above the exit close;
A3 after an exit, re-enter when the close crosses above the 50-day average;
B1 half on 55/20, half on 20/10;           B2 half on 55/20, half on "close above the 50-day average";
C  50% on a 20-day high, 100% on a 55-day high, everything out on a 20-day low;
D  55-day highs only count while the 200-day average is above its value 20 days earlier;
E  55/20 x min(1, 40% / 30-day realised volatility);
F1 exit on a 20-day low or on a close 25% below the highest close since entry;
F2 exit on a 20-day low or after 120 days without a new 55-day high;
G  the breakout lengths around 55/20 (entry 40/55/70/90 x exit 15/20/30) as a plateau check.

Checks: (1) bitcoin 2018-now on the dashboard's daily series (0.10% cost, the signals.py accounting);
(2) 2019-now pick-in-advance over the 24 daily-close clocks (as robust_trend: every year the candidate with the best
average return / volatility over the 24 clocks up to then, applied staggered over the 24 clocks, chained);
(3) every variant unchanged on ETH, SOL, BNB, XRP and MSTR; (4) bitcoin on 24 clocks with 0.30% cost and one day late.
A variant counts as an improvement only if it passes (2), wins on at least 4 of the 5 other assets and on at least
18 of the 24 clocks (return / volatility against plain 55/20 each time).

No look-ahead: every position is decided from closes up to and including the signal close and held from the next
period. Data: bitcoin hourly store kept by seasonality.py / clock_robustness.py (read only, topped up in memory),
the dashboard's long daily bitcoin series, Binance daily closes for the other coins (their clock_robustness hourly
stores, read only, as fallback), Yahoo Finance for MSTR (this module keeps its own MSTR store as fallback). Public,
key-less endpoints only, through the project's fixed user agent.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import walkforward as wf

COST = 0.001
HARD_COST = 0.003
YEAR = 365
MAIN_START = "2018-01-01"
FIRST_PICK = 2019
ETF = "2024-01-01"
FAST_DAYS, TRAIL, TIME_STOP = 60, 0.25, 120
VOL_DAYS, VOL_TARGET = 30, 0.40
SLOPE_DAYS = 20
WARM = 221  # 200-day average plus 20 days of slope for D
GRID_IN, GRID_OUT = (40, 55, 70, 90), (15, 20, 30)
MOST, ASSETS_NEED = 18, 4
MSTR_START = "2020-08-11"
MSTR_STORE = common.STORE / "rule_upgrades_mstr.json"
OTHERS = (("ETH", "ETHUSDT"), ("SOL", "SOLUSDT"), ("BNB", "BNBUSDT"), ("XRP", "XRPUSDT"), ("MSTR", "MSTR"))

KEYS = ("plain", "A1", "A2", "A3", "B1", "B2", "C", "D", "E", "F1", "F2")
UPGRADES = KEYS[1:]
NAMES = {
    "plain": "原版 55／20（55 日新高進、20 日新低出）",
    "A1": "A1 出場後 60 天內，創 20 日新高就買回",
    "A2": "A2 出場後收盤回到賣出價之上就買回",
    "A3": "A3 出場後收盤向上穿過 50 日均線就買回",
    "B1": "B1 一半 55／20、一半 20／10",
    "B2": "B2 一半 55／20、一半 50 日均線",
    "C": "C 20 日新高先買一半、55 日新高買滿",
    "D": "D 200 日均線往上時才接受 55 日新高",
    "E": "E 持有時依波動調整（40% ÷ 30 天波動）",
    "F1": "F1 加上「從進場後最高收盤跌 25% 就出」",
    "F2": "F2 加上「120 天沒有新的 55 日新高就出」",
    "blend": "對照：17 條順勢規則混合",
    "hold": "對照：一直抱著不動",
}
SHORT = {"plain": "原版 55／20", "A1": "A1", "A2": "A2", "A3": "A3", "B1": "B1", "B2": "B2", "C": "C", "D": "D",
         "E": "E", "F1": "F1", "F2": "F2", "blend": "17 條混合"}

_pct, _num, _d = cr._pct, cr._num, cr._d


def _pp(v) -> str:
    return cr._pp(v)


# ---------- the rules ----------

def _roll(c: np.ndarray, n: int, how: str) -> np.ndarray:
    r = pd.Series(c).rolling(n, min_periods=n)
    return (r.max() if how == "max" else r.min() if how == "min" else r.mean()).values


def _bo(c: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    return cr._breakout(pd.Series(c), n_in, n_out)


def _ma(c: np.ndarray, n: int) -> np.ndarray:
    mean = _roll(c, n, "mean")
    return np.where(np.isnan(mean), 0.0, (c > mean).astype(float))


def _loop(c: np.ndarray, key: str) -> np.ndarray:
    """The on/off variants that need memory (re-entry after an exit, trend filter, exit tweaks)."""
    hi55, lo20, hi20 = _roll(c, 55, "max"), _roll(c, 20, "min"), _roll(c, 20, "max")
    ma50 = _roll(c, 50, "mean") if key == "A3" else None
    ma200 = _roll(c, 200, "mean") if key == "D" else None
    n = len(c)
    out = np.zeros(n)
    state, exit_i, exit_px, peak, last_up = 0, None, None, 0.0, 0
    for i in range(n):
        x = c[i]
        if hi55[i] != hi55[i]:  # not enough history yet
            continue
        up = x >= hi55[i]
        if state == 0:
            go = up
            if key == "A1":
                go = up or (exit_i is not None and i - exit_i <= FAST_DAYS and x >= hi20[i])
            elif key == "A2":
                go = up or (exit_px is not None and x > exit_px)
            elif key == "A3":
                go = up or (exit_i is not None and ma50[i - 1] == ma50[i - 1] and x > ma50[i] and c[i - 1] <= ma50[i - 1])
            elif key == "D":
                past = ma200[i - SLOPE_DAYS] if i >= SLOPE_DAYS else float("nan")
                go = up and past == past and ma200[i] > past
            if go:
                state, peak, last_up = 1, x, i
        else:
            peak = max(peak, x)
            if up:
                last_up = i
            leave = (not up) and x <= lo20[i]
            if key == "F1" and x <= (1 - TRAIL) * peak:
                leave = True
            if key == "F2" and i - last_up >= TIME_STOP:
                leave = True
            if leave:
                state, exit_i, exit_px = 0, i, x
        out[i] = state
    return out


def _scale(c: np.ndarray, year: int) -> np.ndarray:
    vol = pd.Series(c).pct_change().rolling(VOL_DAYS, min_periods=VOL_DAYS).std(ddof=1) * math.sqrt(year)
    return (VOL_TARGET / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna()).fillna(0.0).values


def _legs(c: np.ndarray, key: str) -> list[tuple[float, np.ndarray]]:
    """(weight, 0/1 position after each close) of the on/off parts a variant is made of."""
    if key in ("plain", "E"):
        return [(1.0, _bo(c, 55, 20))]
    if key == "B1":
        return [(0.5, _bo(c, 55, 20)), (0.5, _bo(c, 20, 10))]
    if key == "B2":
        return [(0.5, _bo(c, 55, 20)), (0.5, _ma(c, 50))]
    if key == "C":  # half from a 20-day high, the other half from a 55-day high, both out on a 20-day low
        return [(0.5, _bo(c, 20, 20)), (0.5, _bo(c, 55, 20))]
    if key.startswith("g"):
        n_in, n_out = (int(x) for x in key[1:].split("_"))
        return [(1.0, _bo(c, n_in, n_out))]
    return [(1.0, _loop(c, key))]


def _want(c: np.ndarray, key: str, year: int = YEAR) -> np.ndarray:
    """Share held after each close (0 while a rule has no value yet)."""
    c = np.asarray(c, dtype=float)
    if key == "blend":
        signal, _ = te._signals(pd.Series(c))
        return signal.mean(axis=1).where(signal.notna().all(axis=1)).fillna(0.0).values
    w = sum(weight * leg for weight, leg in _legs(c, key))
    return w * _scale(c, year) if key == "E" else w


def _grid_keys() -> list[str]:
    return [f"g{a}_{b}" for a in GRID_IN for b in GRID_OUT]


# ---------- measuring ----------

def _stats(net: np.ndarray, year: int = YEAR) -> dict:
    n = len(net)
    growth = float(np.prod(1 + net))
    ret = (growth ** (year / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(year) * 100) if n > 1 else 0.0
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    return {"ret": float(ret), "vol": vol, "ratio": float(ret / vol) if vol > 1e-9 else 0.0,
            "deepest": float((equity / np.maximum.accumulate(equity) - 1).min() * 100)}


def _trades(c: np.ndarray, legs, change: np.ndarray, mask: np.ndarray, cost: float) -> dict:
    """Closed trades of every on/off part inside the window (as signals._record counts them) and the re-entry
    premium: entry close / the last exit close before it - 1, for every entry inside the window after an exit."""
    results, prem = [], []
    first = int(np.argmax(mask))
    for _, leg in legs:
        pos = np.concatenate([[0.0], leg[:-1]])
        net, _ = te._run(pos, change, cost)
        p, r = pos[first:], net[first:]
        i = 0
        while i < len(p):
            if p[i] > 0.5 and (i == 0 or p[i - 1] < 0.5):
                j = i
                while j + 1 < len(p) and p[j + 1] > 0.5:
                    j += 1
                if j + 1 < len(p):
                    results.append(float(np.prod(1 + r[i:j + 1]) - 1) * 100)
                i = j + 1
            else:
                i += 1
        last_exit = None
        for k in range(1, len(leg)):
            if leg[k - 1] > 0.5 and leg[k] < 0.5:
                last_exit = c[k]
            elif leg[k - 1] < 0.5 and leg[k] > 0.5 and last_exit is not None and k >= first:
                prem.append((c[k] / last_exit - 1) * 100)
    won = [g for g in results if g > 0]
    lost = [g for g in results if g <= 0]
    return {"n": len(results), "wins": len(won), "avg_win": float(np.mean(won)) if won else None,
            "avg_loss": float(np.mean(lost)) if lost else None, "prem": float(np.mean(prem)) if prem else None,
            "prem_med": float(np.median(prem)) if prem else None, "prem_n": len(prem)}


def _daily(close: pd.Series, start, year: int, keys, cost: float = COST, trades: bool = True) -> dict:
    c = close.values.astype(float)
    change = close.pct_change().fillna(0.0).values
    mask = close.index >= pd.Timestamp(start)
    after = close.index[mask] >= pd.Timestamp(ETF)
    out = {"hold": _stats(change[mask], year)}
    out["hold"]["after"] = _stats(change[mask][after], year) if after.sum() > 30 else None
    for k in keys:
        w = _want(c, k, year)
        pos = np.concatenate([[0.0], w[:-1]])
        net, _ = te._run(pos, change, cost)
        s = _stats(net[mask], year)
        s["after"] = _stats(net[mask][after], year) if after.sum() > 30 else None
        s["share"] = float(pos[mask].mean() * 100)
        if trades and k != "blend":
            s.update(_trades(c, _legs(c, k), change, mask, cost))
        out[k] = s
    return out


def _flip(c: np.ndarray, key: str, year: int = YEAR) -> dict:
    """Today's state and the next close that would switch it (bisection; every rule here is monotone in tomorrow's
    close)."""
    long = bool(_want(c, key, year)[-1] > 1e-9)
    last = float(c[-1])

    def held(p):
        return bool(_want(np.append(c, p), key, year)[-1] > 1e-9)

    lo, hi = last * 1e-3, last * 100
    if held(lo) or not held(hi):
        return {"long": long, "level": None}
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if held(mid):
            hi = mid
        else:
            lo = mid
    return {"long": long, "level": hi, "share": float(_want(c, key, year)[-1] * 100)}


# ---------- data ----------

def _btc_hourly() -> tuple[dict, str | None]:
    """Bitcoin 1-hour closes from the stores kept by other modules (read only); missing hours fetched into memory."""
    rows = cr._read(cr.SHARED_STORE)
    if len(rows) < cr.MIN_HOURS:
        rows = cr._read(cr._own_store("BTCUSDT"))
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    problem = None
    if not rows or max(rows) < today_ms - cr.HOUR_MS:
        try:
            cr._fetch("BTCUSDT", rows, keep=False)
        except Exception as exc:  # noqa: BLE001 - the stored history is still usable
            problem = f"比特幣 1 小時線這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料。"
    return rows, problem


def _crypto(symbol: str) -> tuple[pd.Series, str | None]:
    try:
        return ta._crypto_close(symbol), None
    except Exception as exc:  # noqa: BLE001 - fall back on the hourly store clock_robustness keeps (read only)
        rows = cr._read(cr._own_store(symbol))
        if len(rows) < 24 * 400:
            raise
        price, b0, _ = cr._grid(rows)
        closes = price[24::24]
        series = pd.Series(closes, index=pd.date_range(b0, periods=len(closes), freq="D"))
        return series, (f"{symbol[:-4]} 的幣安日線這次拿不到（{type(exc).__name__}），改用之前存下來的 1 小時線拼出的日收盤。")


def _mstr() -> tuple[pd.Series, str | None]:
    try:
        series = ta._share_close("MSTR")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            MSTR_STORE.write_text(json.dumps({_d(k): float(v) for k, v in series.items()}), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        return series, None
    except Exception as exc:  # noqa: BLE001
        stored = json.loads(MSTR_STORE.read_text(encoding="utf-8"))
        series = pd.Series({pd.Timestamp(k): float(v) for k, v in stored.items()}).sort_index()
        return series, f"MSTR 的 Yahoo 股價這次拿不到（{type(exc).__name__}），用的是之前存下來的資料。"


# ---------- computation ----------

def _clocks(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, w0, dates = m.days, m.w0, m.dates
    npre = len(pre)
    keys = KEYS + ("blend",)
    gkeys = _grid_keys()
    nets, hard, now = {k: [] for k in keys + tuple(gkeys)}, {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for h in range(24):
        closes = np.concatenate([pre, price[h::24]])
        for k in nets:
            w = _want(closes, k)[npre:]
            if k in now:
                now[k].append(float(w[-1]))
            w = w[:days]
            nets[k].append(eng.run(h, w, COST))
            if k in hard:
                hard[k].append(eng.run(h, rt._shift(w, 1), HARD_COST))
    nets = {k: np.array(v) for k, v in nets.items()}
    hard = {k: np.array(v) for k, v in hard.items()}
    stag = {k: _stats(rt._stagger(list(nets[k]), w0)) for k in nets}

    # ----- step 2: pick in advance, 2019 on -----
    def at(day) -> int:
        return int(dates.searchsorted(pd.Timestamp(day)))

    years = list(range(FIRST_PICK, dates[-1].year + 1))
    picks, rho = {}, {}
    for y in years:
        a, c = at(f"{y}-01-01"), min(at(f"{y + 1}-01-01"), days)
        before = {k: float(rt._ratio(nets[k][:, :a]).mean()) for k in KEYS}
        picks[y] = max(KEYS, key=lambda k: before[k])
        if c - a >= 30:
            inside = {k: float(rt._ratio(nets[k][:, a:c]).mean()) for k in KEYS}
            rho[y] = wf._spearman([before[k] for k in KEYS], [inside[k] for k in KEYS])
        else:
            rho[y] = None

    def chain(choice) -> dict:
        parts = []
        for y in years:
            a, c = at(f"{y}-01-01"), min(at(f"{y + 1}-01-01"), days)
            parts.append(rt._stagger(list(choice(y)), a, c))
        return _stats(np.concatenate(parts))

    chains = {k: chain(lambda y, k=k: nets[k]) for k in keys}
    chains["pick"] = chain(lambda y: nets[picks[y]])
    chains["hold"] = chain(lambda y: m.change[None, :])

    # ----- step 4: 24 clocks, 0.30%, one day late -----
    clock = {}
    for k in KEYS:
        st = [_stats(hard[k][h][w0:]) for h in range(24)]
        clock[k] = st
    beat = {}
    for k in UPGRADES:
        beat[k] = {"ratio": sum(clock[k][h]["ratio"] > clock["plain"][h]["ratio"] for h in range(24)),
                   "deep": sum(clock[k][h]["deepest"] > clock["plain"][h]["deepest"] for h in range(24))}
    return {"m": m, "stag": stag, "years": years, "picks": picks, "rho": rho, "chains": chains, "clock": clock,
            "beat": beat, "now": now, "hold": _stats(m.change[w0:])}


def _compute(daily: pd.Series, price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, others: dict) -> dict:
    main = _daily(daily, MAIN_START, YEAR, KEYS + ("blend",))
    grid = _daily(daily, MAIN_START, YEAR, _grid_keys(), trades=False)
    clocks = _clocks(price, b0, pre)
    assets = {}
    for name, (series, year, start) in others.items():
        assets[name] = {"res": _daily(series, start, year, KEYS + ("blend",), trades=False), "start": start,
                        "last": series.index[-1]}
    score = {}
    for k in UPGRADES:
        score[k] = {"ratio": sum(a["res"][k]["ratio"] > a["res"]["plain"]["ratio"] for a in assets.values()),
                    "deep": sum(a["res"][k]["deepest"] > a["res"]["plain"]["deepest"] for a in assets.values())}
    return {"main": main, "grid": grid, "clocks": clocks, "assets": assets, "score": score}


# ---------- the study ----------

def study() -> dict:
    daily = te._daily_close()
    rows, problem = _btc_hourly()
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"比特幣 1 小時線只有 {len(rows)} 筆")
    price, b0, missing = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    problems = [problem] if problem else []
    others = {}
    for name, symbol in OTHERS:
        try:
            if symbol == "MSTR":
                series, p = _mstr()
                year, start = 252, pd.Timestamp(MSTR_START)
            else:
                series, p = _crypto(symbol)
                year = YEAR
                start = max(pd.Timestamp(MAIN_START), series.index[0] + pd.Timedelta(days=WARM))
            if p:
                problems.append(p)
            others[name] = (series, year, start)
        except Exception as exc:  # noqa: BLE001 - one asset missing must not stop the study
            problems.append(f"{name} 的價格這次拿不到（{type(exc).__name__}），沒有算進去。")
    r = _compute(daily, price, b0, pre, others)
    main, grid, ck, assets, score = r["main"], r["grid"], r["clocks"], r["assets"], r["score"]
    n_assets = len(assets)
    need_assets = min(ASSETS_NEED, n_assets)
    years, picks = ck["years"], ck["picks"]
    chains = ck["chains"]
    last = daily.index[-1]

    # ----- the three gates -----
    pick_beats = chains["pick"]["ratio"] > chains["plain"]["ratio"]
    picked = {k: sum(1 for y in years if picks[y] == k) for k in KEYS}
    half = math.ceil(len(years) / 2)
    gate = {}
    for k in UPGRADES:
        g2 = pick_beats and picked[k] >= half
        g2_loose = chains[k]["ratio"] > chains["plain"]["ratio"]
        g3 = score[k]["ratio"] >= need_assets
        g4 = ck["beat"][k]["ratio"] >= MOST
        gate[k] = {"g2": g2, "g2_loose": g2_loose, "g3": g3, "g4": g4, "all": g2 and g3 and g4,
                   "loose": g2_loose and g3 and g4}
    passed = [k for k in UPGRADES if gate[k]["all"]]
    loose = [k for k in UPGRADES if gate[k]["loose"] and k not in passed]

    # ----- table 1: bitcoin 2018 on -----
    def trade_cells(x: dict) -> list:
        if "n" not in x:
            return ["—", "—", "—"]
        wins = f"{x['n']} 筆裡 {x['wins']} 筆"
        wl = f"{_pct(x['avg_win'])}｜{_pct(x['avg_loss'])}"
        prem = (f"{_pct(x['prem'])}（中位 {_pct(x['prem_med'])}，{x['prem_n']} 次）" if x["prem"] is not None else "—")
        return [wins, wl, prem]

    rows1 = []
    for k in KEYS + ("blend", "hold"):
        x = main[k]
        s = ck["stag"].get(k) if k != "hold" else ck["hold"]
        aft = x.get("after")
        rows1.append([NAMES[k], f"{_pct(x['ret'])}｜{_num(x['ratio'])}｜{_pct(x['deepest'])}",
                      f"{_pct(s['ret'])}｜{_num(s['ratio'])}｜{_pct(s['deepest'])}"] + trade_cells(x)
                     + [f"{_pct(aft['ret'])}｜{_pct(aft['deepest'])}" if aft else "—"])

    # ----- table 2: the gates -----
    def mark(ok: bool) -> str:
        return "過" if ok else "沒過"

    rows2 = [[NAMES["plain"], f"{_pct(chains['plain']['ret'])}｜{_num(chains['plain']['ratio'])}｜{_pct(chains['plain']['deepest'])}",
              f"{len(years)} 年裡 {picked['plain']} 年", "—", "—", "—", "比較的基準"]]
    for k in UPGRADES:
        c, g = chains[k], gate[k]
        rows2.append([NAMES[k], f"{_pct(c['ret'])}｜{_num(c['ratio'])}｜{_pct(c['deepest'])}",
                      f"{len(years)} 年裡 {picked[k]} 年",
                      f"{n_assets} 個裡 {score[k]['ratio']} 個",
                      f"{n_assets} 個裡 {score[k]['deep']} 個",
                      f"24 個時間裡 {ck['beat'][k]['ratio']} 個｜虧得較淺 {ck['beat'][k]['deep']} 個",
                      ("三關都過" if g["all"] else
                       "／".join(f"{name}{mark(ok)}" for name, ok in (("事先挑", g["g2"]), ("換資產", g["g3"]),
                                                                    ("換時間", g["g4"]))))])
    pc = chains["pick"]
    rows2.append(["每年事先挑（從原版和 A～F 裡挑前幾年最好的）",
                  f"{_pct(pc['ret'])}｜{_num(pc['ratio'])}｜{_pct(pc['deepest'])}",
                  "、".join(f"{y}:{SHORT[picks[y]]}" for y in years), "—", "—", "—",
                  "贏原版" if pick_beats else "沒贏原版"])
    hc = chains["hold"]
    rows2.append(["對照：一直抱著不動", f"{_pct(hc['ret'])}｜{_num(hc['ratio'])}｜{_pct(hc['deepest'])}",
                  "—", "—", "—", "—", "—"])

    # ----- table 3: other assets -----
    rows3 = []
    for name, a in assets.items():
        res = a["res"]
        best = max(UPGRADES, key=lambda k: res[k]["ratio"])
        worst = min(UPGRADES, key=lambda k: res[k]["ratio"])
        nwin = sum(res[k]["ratio"] > res["plain"]["ratio"] for k in UPGRADES)
        rows3.append([f"{name}（{_d(a['start'])} 起）",
                      f"{_pct(res['hold']['ret'])}｜{_num(res['hold']['ratio'])}｜{_pct(res['hold']['deepest'])}",
                      f"{_pct(res['plain']['ret'])}｜{_num(res['plain']['ratio'])}｜{_pct(res['plain']['deepest'])}",
                      f"{_pct(res['blend']['ret'])}｜{_num(res['blend']['ratio'])}｜{_pct(res['blend']['deepest'])}",
                      f"{SHORT[best]}：{_pct(res[best]['ret'])}｜{_num(res[best]['ratio'])}｜{_pct(res[best]['deepest'])}",
                      f"{SHORT[worst]}：{_pct(res[worst]['ret'])}｜{_num(res[worst]['ratio'])}｜{_pct(res[worst]['deepest'])}",
                      f"10 種裡 {nwin} 種"])

    # ----- table 4: plateau -----
    rows4 = []
    for a_ in GRID_IN:
        row = [f"{a_} 日新高進場"]
        for b_ in GRID_OUT:
            x, s = grid[f"g{a_}_{b_}"], ck["stag"][f"g{a_}_{b_}"]
            row.append(f"{_pct(x['ret'])}｜{_num(x['ratio'])}｜{_pct(x['deepest'])}｜分散 {_num(s['ratio'])}")
        rows4.append(row)
    gk = _grid_keys()
    base = grid["g55_20"]["ratio"]
    neighbours = ["g40_20", "g70_20", "g55_15", "g55_30"]
    near = [grid[k]["ratio"] for k in neighbours]
    plateau = min(near) >= 0.8 * base
    rank = sorted(gk, key=lambda k: -grid[k]["ratio"]).index("g55_20") + 1
    srank = sorted(gk, key=lambda k: -ck["stag"][k]["ratio"]).index("g55_20") + 1
    ratios = [grid[k]["ratio"] for k in gk]

    # ----- today -----
    dc = daily.values.astype(float)
    lastp = float(dc[-1])
    fp = _flip(dc, "plain")
    plain_leg = _legs(dc, "plain")[0][1]
    exits = [i for i in range(1, len(plain_leg)) if plain_leg[i - 1] > 0.5 and plain_leg[i] < 0.5]
    last_exit = (daily.index[exits[-1]], float(dc[exits[-1]])) if exits else None
    clocks_long = int(sum(v > 0 for v in ck["now"]["plain"]))
    now = [{"label": f"最後一個收完的日子（{_d(last)}，UTC 0 點收盤＝台灣早上 8 點）", "value": f"比特幣 {lastp:,.0f} 美元"}]
    lvl = fp["level"]
    if fp["long"]:
        state = f"持有中；收盤 ≤ {lvl:,.0f}（{_pct((lvl / lastp - 1) * 100)}）會轉成空手" if lvl else "持有中"
    else:
        state = f"空手中；收盤 ≥ {lvl:,.0f}（{_pct((lvl / lastp - 1) * 100)}）會轉成持有" if lvl else "空手中"
        if lvl and last_exit:
            state += f"；上次賣出是 {_d(last_exit[0])} 收盤 {last_exit[1]:,.0f}，照這個價位買回等於貴 {_pct((lvl / last_exit[1] - 1) * 100)}"
    now.append({"label": "原版 55／20 現在的狀態", "value": state})
    now.append({"label": "原版 55／20 分散到 24 個收盤時間：持有中的", "value": f"24 個時間裡 {clocks_long} 個"})
    for k in passed:
        f = _flip(dc, k)
        word = "持有中" if f["long"] else "空手中"
        if f["level"]:
            word += (f"；收盤 {'≤' if f['long'] else '≥'} {f['level']:,.0f}（{_pct((f['level'] / lastp - 1) * 100)}）會改變")
        now.append({"label": f"通過三關的 {NAMES[k]}", "value": word})
    if not passed:
        now.append({"label": "通過三關的改良版", "value": "沒有，所以這裡只列原版的狀態"})
    now.append({"label": "這幾格的意思", "value": "只描述規則目前停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    p_ = main["plain"]

    def gates_text(k: str) -> str:
        g = gate[k]
        bits = [f"事先挑 {len(years)} 年裡 {picked[k]} 年", f"換資產 {n_assets} 個裡 {score[k]['ratio']} 個",
                f"24 個時間裡 {ck['beat'][k]['ratio']} 個"]
        return f" {SHORT[k]}（" + "、".join(bits) + "）"

    if passed:
        s1 = "通過事先挑、換資產、換收盤時間三關的有：" + "、".join(gates_text(k) for k in passed) + "。"
    else:
        closest = sorted(UPGRADES, key=lambda k: (-(gate[k]["g2"] + gate[k]["g3"] + gate[k]["g4"]),
                                                  -chains[k]["ratio"]))[:2]
        s1 = "10 種改良沒有一種三關全過，最接近的是" + "和".join(gates_text(k) for k in closest)
        if loose:
            k = loose[0]
            s1 += (f"；若只看「2019 年起固定用它」，{SHORT[k]} 報酬÷波動 {_num(chains[k]['ratio'])} 對 "
                   f"{_num(chains['plain']['ratio'])} 算過，但最深虧 {_pct(main[k]['deepest'])} 對 {_pct(p_['deepest'])}")
        s1 += "。"
    fast = [k for k in ("A1", "A2", "A3") if main[k]["prem"] is not None]
    s2 = (f"買回太貴平均不嚴重：原版 {p_['prem_n']} 次買回平均比上次賣出 {_pct(p_['prem'])}（中位 {_pct(p_['prem_med'])}），"
          f"更快買回的 {'、'.join(fast)} 是 {'、'.join(_pct(main[k]['prem']) for k in fast)}，"
          f"報酬÷波動卻從 {_num(p_['ratio'])} 變成 {'、'.join(_num(main[k]['ratio']) for k in fast)}。")
    entry_n = [grid["g40_20"]["ratio"], grid["g70_20"]["ratio"]]
    exit_n = [grid["g55_15"]["ratio"], grid["g55_30"]["ratio"]]
    s3 = (f"55／20（{_num(base)}）{'在一片平台上' if plateau else '照事先定的標準不算平台'}：進場改 40、70 日是 "
          f"{_num(entry_n[0])}、{_num(entry_n[1])}，出場改 15、30 日是 {_num(exit_n[0])}、{_num(exit_n[1])}。")
    verdict = s1 + s2 + s3
    if len(verdict) > 350 and "；若只看" in s1:  # keep it short; the loose reading is also in the notes
        verdict = s1.split("；若只看")[0] + "。" + s2 + s3

    # ----- notes -----
    in_main = daily.index >= pd.Timestamp(MAIN_START)
    plain_w = _want(dc, "plain")
    diff_days = {k: int((np.abs(_want(dc, k) - plain_w) > 1e-9)[in_main].sum()) for k in ("A1", "A2", "A3", "D", "F1", "F2")}
    pick_txt = "、".join(f"{y} 年挑 {SHORT[picks[y]]}" for y in years)
    rhos = [v for v in ck["rho"].values() if v is not None]
    notes = [
        "候選改良和所有參數（60 天、25%、120 天、40% 波動目標、200 日均線看 20 天前、12 種天數）是用今天已知的「55／20 的弱點」"
        "事先定好的，看結果之前沒有改過；但這份清單本身是事後才想到的，不是多年前登記好的，結果要打折看。",
        "做法：第一張表是儀表板原本的日線（UTC 0 點收盤），收盤決定、隔天持有，每次買賣扣 0.10%，和「順勢訊號」頁的算法相同；"
        "「24 時鐘分散版」是把錢平分到 24 個收盤時間各做各的。交易筆數、勝率、買回溢價都用日線算；B、C 是兩半各自的進出合起來算，"
        "E 的進出和原版相同（只是持有多少不同）。買回溢價＝買回那天的收盤 ÷ 上一次賣出那天的收盤 − 1。",
        f"事先挑：2019 年起每年 1 月 1 日，用之前所有資料在 24 個收盤時間的平均「報酬÷波動」（0.10%）從原版和 A～F 共 11 種裡挑最高的，"
        f"下一年平分到 24 個時間照做，逐年接起來。{pick_txt}。前幾年的排名和當年排名的等級相關平均 "
        f"{_num(float(np.mean(rhos))) if rhos else '—'}（1＝完全一樣，0＝沒關係）。"
        f"「事先挑」這一關的定義：每年事先挑的結果贏原版，而且這個改良版被挑中至少 {half} 年。",
        f"換資產：同樣的規則、同樣的天數原封不動用在 ETH、SOL、BNB、XRP（上市滿 {WARM} 天、且不早於 2018 年才起算）和 MSTR"
        f"（{MSTR_START} 起，公司開始買比特幣；天數是交易日、一年算 252 天）；成本 0.10%。要 {n_assets} 個裡至少 {need_assets} 個的報酬÷波動贏原版。",
        f"換時間：比特幣 24 個收盤時間，每個時間都扣 0.30%、而且晚一天才照做，同一個時間裡和原版比報酬÷波動；要 24 個裡至少 {MOST} 個贏。",
        (f"寬鬆一點看（事先挑那一關改成「2019 年起固定用它、報酬÷波動贏原版」）：" +
         ("、".join(SHORT[k] for k in loose) + " 也會過。" if loose else "也沒有多出任何一種通過。")),
        f"平台的判斷事先定為：55／20 上下左右四個鄰居（40／20、70／20、55／15、55／30）的報酬÷波動都至少有它的八成。"
        f"日線版它在 12 種裡排第 {rank}，24 時鐘分散版排第 {srank}。",
        "日線上和原版持有狀態不同的天數（2018 年起）：" + "、".join(
            f"{k} {diff_days[k]} 天" for k in ("A1", "A2", "A3", "D", "F1", "F2"))
        + "。為 0 的表示那條附加規則在這段日線上從來沒有比原版的出場規則更早觸發；換收盤時間後才偶爾不同。",
        f"交易筆數很少：原版 2018 年起只有 {p_['n']} 筆，一兩筆大賺的交易就能改變排名；各種比較都不是彼此獨立的證據。",
        "成本很粗略：0.10%／0.30% 只是交易金額的固定比例，沒有算滑價、稅和現金利息；依波動調整、一半一半這類做法每天小額調整也算了成本。",
        "其他資產的名單是今天還在的大幣和 MSTR，有倖存者偏誤；MSTR 是股票，只有交易日的價格。",
        "過去的結果不保證未來會一樣。這裡只整理規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += problems
    if missing:
        notes.append(f"比特幣 1 小時線整段缺 {missing:,} 個小時（幣安早年停機），缺的小時沿用前一個小時的價格。")

    span = f"{MAIN_START} 到 {_d(last)}"
    tables = [
        {"caption": (f"比特幣 {span}：原版和 10 種改良。第二欄是儀表板日線（台灣早上 8 點收盤、0.10%）；第三欄是平分到 24 個收盤時間的分散版"
                     "（1 小時線拼出，同樣 0.10%）。「每年報酬｜報酬÷波動｜最深虧損」。最後一欄是 2024 年起的每年報酬｜最深虧損。"),
         "columns": ["做法", "日線：每年報酬｜報酬÷波動｜最深虧", "24 時鐘分散版：每年報酬｜報酬÷波動｜最深虧", "交易：賺錢的筆數",
                     "平均賺｜平均賠", "平均買回溢價（比上次賣出貴多少）", "2024 年起：每年報酬｜最深虧"],
         "rows": rows1},
        {"caption": ("三關：事先挑（2019 年起每年平分 24 個時間、逐年接起來，看報酬÷波動）、換資產（ETH、SOL、BNB、XRP、MSTR，"
                     f"報酬÷波動和最深虧損各贏原版幾個）、換時間（比特幣 24 個收盤時間、成本 0.30%、晚一天）。要三關都過才算改良。"),
         "columns": ["做法", "2019 年起固定用它（24 時間分散、逐年接）：每年報酬｜報酬÷波動｜最深虧", "每年事先挑時被挑中",
                     "換資產：報酬÷波動贏原版", "換資產：最深虧損比原版淺", "換時間（0.30%、晚一天）：報酬÷波動贏原版", "結果"],
         "rows": rows2},
        {"caption": ("其他資產原封不動照做（成本 0.10%，日線）。每格是「每年報酬｜報酬÷波動｜最深虧損」；"
                     "「改良版裡最好／最差」是 10 種改良裡報酬÷波動最高和最低的那一種。"),
         "columns": ["資產（起算日）", "一直抱著", "原版 55／20", "17 條混合", "改良版裡最好", "改良版裡最差",
                     "報酬÷波動贏原版的改良"],
         "rows": rows3},
        {"caption": (f"天數平台檢查（比特幣 {span}）：每格是日線版「每年報酬｜報酬÷波動｜最深虧損」，最後是 24 時鐘分散版的報酬÷波動。"
                     f"原版是 55 日新高進、20 日新低出。"),
         "columns": ["進場", "15 日新低出場", "20 日新低出場", "30 日新低出場"],
         "rows": rows4},
    ]
    return {
        "id": "rule_upgrades",
        "title": "55／20 規則能改得更好嗎？",
        "question": ("儀表板的主規則「55 日新高進、20 日新低出」常常買回得比賣出貴很多、V 型反彈時進場太慢。"
                     "常見的改良（更快買回、分批、加均線、依波動調整、停損或時間停損）能不能通過同樣的三關檢驗，"
                     "還是原版已經是最好的？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 K 線（BTC、ETH、SOL、BNB、XRP 對 USDT，日線與 1 小時線）",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "blockchain.info 每日比特幣價格（2017 年以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance MSTR 每日股價", "url": "https://finance.yahoo.com/quote/MSTR/history"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
