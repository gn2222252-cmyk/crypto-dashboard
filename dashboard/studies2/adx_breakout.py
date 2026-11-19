"""Historical study "adx_breakout": a regime-switching rule from trading-course literature - take 15-day breakouts
only when the market was quiet (ADX(14) below 25 before the breakout), on the idea that breakouts out of quiet
ranges are more reliable than breakouts inside an already-trending market. Does it beat the dashboard's main signal
ma50 (hold while the daily close is above its 50-day average), and what does the ADX gate add to a plain breakout?

How this differs from earlier studies: donchian_ensemble / rule_upgrades / main_signal_check tested channel
breakouts (20/10, 55/20, 40-90 / 15-30) WITHOUT any regime filter; pullback_setups used ADX > 30 the other way round
(buy pullbacks INSIDE strong trends); thesis_four used "ADX > 25 and DI+ > DI-" as a long signal. No study has used a
LOW ADX as a gate for breakout entries.

Pre-registered BEFORE any result was seen (nothing below was changed after running):
  A1  enter at a daily close that is above the highest close of the PRIOR 15 days AND ADX(14) at the PRIOR day's
      close < 25; leave at a daily close below the lowest close of the prior 15 days (no ADX condition on the exit).
  B15 the same entry and exit without the ADX gate (plain 15-day breakout) - isolates what the gate adds.
  ma50 the main signal, built exactly as main_signal_check (its random draws for the ordinary person).
ADX(14) = Wilder's (true range, +DM / -DM, Wilder sums, ADX seeded by the mean of the first 14 DX values), on daily
bars. Bitcoin's bars: Bitstamp BTC/USD daily high / low (thesis_four's stored history, 2014-06 .. 2017-08-16), then
Binance BTCUSDT daily high / low (donchian_ensemble's store, read only); the close is always the dashboard's daily
close (te._daily_close) and high / low are widened to contain it. Before 2014-06 there is no high / low in the
project, so the exam uses the period where OHLC exists; a close-only ADX approximation (high = low = close) is
reported for 2011-2016 as a reference row only, labelled as such.
Exam = main_signal_check's, as thesis_four runs it: 108 monthly starts 2015-01 .. 2023-12 (textbook: decide at the
close, hold the next day, 0.10% per trade); 72 starts 2018-01 .. 2023-12 x 24 clocks (each clock's own daily bars:
high / low = highest / lowest hourly close of its hours, thesis_four._Clock), textbook; the same cells as an
ordinary person (act 10 hours later, 0.25%, 10% of entries missed, 200 simulations per clock); other assets ETH,
SOL, BNB, XRP (Binance daily OHLC) and MSTR (Yahoo, from 2020-08-11, 252 days a year), monthly starts from the first
month every rule (and main_signal_check's rules) has a value to three years before today.
Ranking score = the lowest of the four "yearly return beats holding from the same start" shares.
Median deepest loss = median over the 1,728 textbook clock cells.
Placebo: the ADX series circularly shifted by 180 days or more (200 draws; the breakouts stay where they are, only
the gate's timing is scrambled, so the share of quiet days is unchanged); statistic = median over the 108 bitcoin
starts of A1's yearly return; placebo share = share of shifts at least as good as the real gate.
Periods (bitcoin, textbook, positions warm): 2014-07 .. 2016 (the OHLC part of "2011-16"), 2017-2021, 2022-now;
"same direction" = A1's yearly return >= ma50's in each of the three periods.
PASS = A1 ranking score >= ma50's AND A1 median deepest loss no worse than ma50's AND A1 ranking score > B15's AND
placebo share <= 10% AND the same direction in all three periods. All five are needed.
Also reported: time in market, trades, win rate, calendar-year returns (which years the edge comes from), a
look-ahead check (prices after a cut scrambled; positions up to the cut must not move) and today's state.

Only public key-less endpoints through the dashboard's own helpers; stores are read, never written here.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import signals as sig
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import pullback_setups as ps
from dashboard.studies2 import retail_indicators as ri
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import thesis_four as tf

N = 15
ADX_N, ADX_CUT = 14, 25.0
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR, SHARE_YEAR = 365, 252
RULES = ("a1", "b15", "ma50")
KI = {"ma50": 1, "b15": 91, "a1": 92}  # ma50 reuses main_signal_check's random draws
NAMES = {"a1": "A1：盤整後才追 15 日新高（前一天 ADX<25 才進場，跌破 15 日新低出場）",
         "b15": "對照：單純 15 日新高進、15 日新低出（不看 ADX）",
         "ma50": "收盤高於 50 日均線才持有（主訊號）"}
SHORT = {"a1": "A1 盤整突破", "b15": "15 日突破（不看 ADX）", "ma50": "50 日均線", "hold": "一直抱著"}
COINS = (("ETHUSDT", "ETH", None), ("SOLUSDT", "SOL", None), ("BNBUSDT", "BNB", None), ("XRPUSDT", "XRP", None),
         ("MSTR", "MSTR", "2020-08-11"))
PERIODS = (("2014-07 到 2016", "2014-07-01", "2016-12-31"), ("2017 到 2021", "2017-01-01", "2021-12-31"),
           ("2022 到今天", "2022-01-01", "2100-01-01"))
CLOSE_ONLY = ("2011 到 2016（只有收盤價，ADX 用收盤價近似，參考）", "2011-01-01", "2016-12-31")
PLACEBO, PLACEBO_GAP = 200, 180
P_BAR = 0.10
DAY = pd.Timedelta(days=1)
STATS_FROM = "2015-01-01"

_pct, _pp, _d = cr._pct, cr._pp, cr._d
_ok = msc._ok


def _share(x) -> str:
    return "—" if not _ok(x) else f"{x * 100:.0f}%"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- indicator and rules ----------

def _adx(h, l, c) -> np.ndarray:
    """Wilder's ADX(14) at every close (NaN while unknown); leading days without a bar are skipped."""
    h, l, c = (np.asarray(a, dtype=float) for a in (h, l, c))
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    n = len(c)
    out = np.full(n, np.nan)
    if not ok.any():
        return out
    f = int(np.argmax(ok))
    h, l, c = h[f:], l[f:], c[f:]
    m, k = len(c), ADX_N
    tr, pdm, ndm = np.zeros(m), np.zeros(m), np.zeros(m)
    if m > 1:
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
        up, down = h[1:] - h[:-1], l[:-1] - l[1:]
        pdm[1:] = np.where((up > down) & (up > 0), up, 0.0)
        ndm[1:] = np.where((down > up) & (down > 0), down, 0.0)
    s_tr, s_p, s_n = (tf._wilder_sum(a, k, k) for a in (tr, pdm, ndm))
    with np.errstate(all="ignore"):
        dip, din = 100 * s_p / s_tr, 100 * s_n / s_tr
        dx = 100 * np.abs(dip - din) / (dip + din)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    a0 = 2 * k - 1
    if m > a0:
        out[f + a0] = float(np.mean(dx[k:a0 + 1]))
        for i in range(a0 + 1, m):
            out[f + i] = (out[f + i - 1] * (k - 1) + dx[i]) / k
    return out


def _channel(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(np.asarray(c, dtype=float)).shift(1)
    return s.rolling(N, min_periods=N).max().values, s.rolling(N, min_periods=N).min().values


def _breakouts(c: np.ndarray, adx: np.ndarray) -> dict[str, np.ndarray]:
    """Position wanted after each close (1 long, 0 cash, NaN unknown) of A1 and B15."""
    c = np.asarray(c, dtype=float)
    hi, lo = _channel(c)
    prev = np.concatenate([[np.nan], np.asarray(adx, dtype=float)[:-1]])
    n = len(c)
    a1, b15 = np.full(n, np.nan), np.full(n, np.nan)
    ca = cb = False
    started = False
    for t in range(n):
        if not np.isfinite(hi[t]):
            continue
        if cb:
            cb = not (c[t] < lo[t])
        else:
            cb = c[t] > hi[t]
        b15[t] = cb
        if not started:
            if not np.isfinite(prev[t]):
                continue
            started = True
        if ca:
            ca = not (c[t] < lo[t])
        else:
            ca = bool(c[t] > hi[t] and np.isfinite(prev[t]) and prev[t] < ADX_CUT)
        a1[t] = ca
    return {"a1": a1, "b15": b15}


def _positions(c, adx) -> dict[str, np.ndarray]:
    out = _breakouts(c, adx)
    out["ma50"] = tf._ma(c)
    return out


# ---------- data ----------

def _btc_frame() -> tuple[pd.DataFrame, list[str]]:
    """Dashboard closes with Bitstamp (to 2017-08-16) then Binance highs and lows, widened to contain the close."""
    problems = []
    close = msc.te._daily_close()
    bs, prob = tf._bitstamp()
    if prob:
        problems.append(prob)
    bn = ps._stored_klines("BTCUSDT")
    ohlc = pd.concat([bs[bs.index < bn.index[0]], bn])
    ohlc = ohlc[~ohlc.index.duplicated(keep="last")].sort_index()
    o = ohlc.reindex(close.index)
    agree = (o["close"] / close - 1).abs()
    frame = pd.DataFrame({"close": close})
    frame["high"] = np.fmax(o["high"].values, close.values)
    frame["low"] = np.fmin(o["low"].values, close.values)
    frame.loc[o["high"].isna(), ["high", "low"]] = np.nan
    frame.attrs["agree"] = float(np.nanmedian(agree.values))
    return frame, problems


def _coins() -> tuple[dict, list[str]]:
    data, problems = {}, []
    three = tf._others_last()
    for symbol, key, since in COINS:
        try:
            if symbol.endswith("USDT"):
                frame, year = ps._stored_klines(symbol), YEAR
            else:
                frame, _ = ri._yahoo(symbol)
                frame = frame[frame.index >= pd.Timestamp(since) - pd.Timedelta(days=800)][["high", "low", "close"]]
                year = SHARE_YEAR
            c = frame["close"]
            pos = _positions(c.values, _adx(frame["high"].values, frame["low"].values, c.values))
            valid = msc._targets(c, year).notna().all(axis=1).values
            known = np.ones(len(c), dtype=bool)
            for k in RULES:
                known &= ~np.isnan(pos[k])
            if not valid.any() or not known.any():
                raise RuntimeError("規則沒有值")
            first = max(c.index[int(np.argmax(valid))], c.index[int(np.argmax(known))]) + DAY
            if since:
                first = max(first, pd.Timestamp(since))
            if first >= three:
                raise RuntimeError("太短")
            g = tf._grid_daily(c, pos, year, first, three, keys=RULES)
            if not g["starts"]:
                raise RuntimeError("沒有起算日")
            data[key] = g
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            problems.append(f"{key} 這次取不到或算不出來（{type(exc).__name__}），沒有算進去。")
    return data, problems


# ---------- the clock grid ----------

def _clock(clk: tf._Clock) -> dict:
    out = {}
    pos = []
    for s in clk.series:
        p = _positions(s["c"], _adx(s["hh"], s["ll"], s["c"]))
        pos.append({k: np.nan_to_num(v[clk.npre:][:clk.days], nan=0.0) for k, v in p.items()})
    n = len(clk.starts)
    for k in RULES:
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, SIMS))
        for h in range(24):
            w = clk.ma[h] if k == "ma50" else pos[h][k]
            r, dd = msc._suffix(np.concatenate([[1.0], clk.eng._mid(h, w, COST, None)]), clk.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, clk.days)
            rng = np.random.default_rng(SEED + 1000 * KI[k] + h)
            mid = clk.eng._mid(b, rt._missed(wl, P_MISS, rng, SIMS), P_COST, None)
            r, _ = msc._suffix(np.vstack([np.ones((1, SIMS)), mid]), clk.starts, YEAR)
            pr[:, h, :] = r
        edge, pedge = tr - clk.hr[:, None], pr - clk.hr[:, None, None]
        out[k] = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                  "c_shallow": int((td > clk.hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                  "c_deep_worst": float(td.min()), "p_n": pedge.size, "p_beat": int((pedge > 0).sum()),
                  "p_edge": float(np.median(pedge)),
                  "p_clock_beat": int((pedge[:, rt.PERSON_CLOCK, :] > 0).sum()),
                  "p_clock_n": pedge[:, rt.PERSON_CLOCK, :].size}
    return out


# ---------- pieces ----------

def _net(close: pd.Series, p: np.ndarray) -> pd.Series:
    change = close.pct_change().fillna(0.0).values
    net, _ = msc.te._run(np.concatenate([[np.nan], np.asarray(p, dtype=float)[:-1]]), change, COST)
    return pd.Series(net, index=close.index)


def _yearly(net: pd.Series, a, b) -> float | None:
    part = net[(net.index >= pd.Timestamp(a)) & (net.index <= pd.Timestamp(b))]
    if len(part) < 60:
        return None
    g = float(np.prod(1 + part.values))
    return (g ** (YEAR / len(part)) - 1) * 100 if g > 0 else -100.0


def _placebo(close: pd.Series, adx: np.ndarray, real: float, rng: np.random.Generator) -> tuple[float, list]:
    """Circular shifts (>= 180 days) of the known part of the ADX series; breakouts stay where they are."""
    d = tf._grid_daily(close, {"a1": _breakouts(close.values, adx)["a1"]}, YEAR, msc.DAILY_STARTS[0],
                       msc.DAILY_STARTS[1], keys=("a1",))
    known = np.flatnonzero(np.isfinite(adx))
    f = int(known[0])
    part = adx[f:]
    vals = []
    for _ in range(PLACEBO):
        k = int(rng.integers(PLACEBO_GAP, len(part) - PLACEBO_GAP))
        shifted = adx.copy()
        shifted[f:] = np.roll(part, k)
        p = _breakouts(close.values, shifted)["a1"]
        g = tf._grid_daily(close, {"a1": p}, YEAR, d["starts"][0], msc.DAILY_STARTS[1], keys=("a1",))
        vals.append(float(np.median(g["a1"][0])))
    return float(np.mean(np.array(vals) >= real - 1e-12)), vals


def _lookahead(frame: pd.DataFrame) -> tuple[int, int]:
    """Scramble every price after a cut and recompute: positions up to the cut must not move."""
    base = _positions(frame["close"].values, _adx(frame["high"].values, frame["low"].values, frame["close"].values))
    rng = np.random.default_rng(7)
    n = len(frame)
    cuts = sorted(rng.integers(n - 3000, n - 5, size=6))
    bad = 0
    for cut in cuts:
        f = frame.copy()
        shock = np.exp(np.cumsum(rng.normal(0, 0.04, n - cut - 1)))
        for col, m in (("close", 1.0), ("high", 1.03), ("low", 0.97)):
            f.iloc[cut + 1:, f.columns.get_loc(col)] *= shock * m
        got = _positions(f["close"].values, _adx(f["high"].values, f["low"].values, f["close"].values))
        bad += sum(int(not np.array_equal(np.nan_to_num(base[k][:cut + 1], nan=-1),
                                          np.nan_to_num(got[k][:cut + 1], nan=-1))) for k in RULES)
    return len(cuts), bad


# ---------- the computation ----------

def _compute() -> dict:
    frame, problems = _btc_frame()
    close = frame["close"]
    adx = _adx(frame["high"].values, frame["low"].values, close.values)
    pos = _positions(close.values, adx)
    daily = tf._grid_daily(close, pos, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], keys=RULES)

    rows, prob = cr._hourly("BTCUSDT")
    if prob:
        problems.insert(0, prob)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ohlc = frame.dropna()[["high", "low", "close"]]
    clk = tf._Clock(price, b0, pre, ohlc)
    clock = _clock(clk)

    coins, prob = _coins()
    problems += prob

    nets = {k: _net(close, pos[k]) for k in RULES}
    nets["hold"] = close.pct_change().fillna(0.0)
    periods = {name: {k: _yearly(nets[k], a, b) for k in nets} for name, a, b in PERIODS}
    co_adx = _adx(close.values, close.values, close.values)
    co = _breakouts(close.values, co_adx)["a1"]
    co_net = {"a1": _net(close, co), "b15": nets["b15"], "ma50": nets["ma50"], "hold": nets["hold"]}
    close_only = {k: _yearly(co_net[k], CLOSE_ONLY[1], CLOSE_ONLY[2]) for k in co_net}

    years = {}
    first_year = pd.Timestamp(STATS_FROM).year
    for y in range(first_year, close.index[-1].year + 1):
        years[y] = {k: float(np.prod(1 + nets[k][nets[k].index.year == y].values) - 1) * 100 for k in nets}

    trades = {k: sig._record(close, pd.Series(pos[k], index=close.index), pd.Timestamp(STATS_FROM), YEAR, True)
              for k in RULES}
    since = close.index >= pd.Timestamp(STATS_FROM)
    quiet = float(np.mean(adx[since] < ADX_CUT))
    hi, _lo = _channel(close.values)
    entries_b = np.flatnonzero(np.diff(np.nan_to_num(pos["b15"]), prepend=0.0) > 0.5)
    prev = np.concatenate([[np.nan], adx[:-1]])
    entries_b = entries_b[entries_b >= int(np.argmax(since))]
    gated_ok = int(np.sum(prev[entries_b] < ADX_CUT))

    s = {}
    for k in RULES:
        x = dict(clock[k])
        dr, dd = daily[k]
        x["d_n"], x["d_beat"] = len(dr), int((dr > daily["hold"][0]).sum())
        x["d_edge"] = float(np.median(dr - daily["hold"][0]))
        x["d_ret_med"] = float(np.median(dr))
        x["d_deep_med"] = float(np.median(dd))
        x["d_shallow"] = int((dd > daily["hold"][1]).sum())
        o_n = o_b = 0
        x["o_each"] = {}
        for a, g in coins.items():
            n_a, b_a = len(g[k][0]), int((g[k][0] > g["hold"][0]).sum())
            x["o_each"][a] = (n_a, b_a)
            o_n, o_b = o_n + n_a, o_b + b_a
        x["o_n"], x["o_beat"] = o_n, o_b
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (o_b / o_n) if o_n else 0.0]
        x["low"] = min(x["checks"])
        s[k] = x

    rng = np.random.default_rng(SEED + 15)
    placebo, pvals = _placebo(close, adx, s["a1"]["d_ret_med"], rng)
    look = _lookahead(frame.dropna())
    return {"close": close, "frame": frame, "adx": adx, "pos": pos, "daily": daily, "s": s, "coins": coins,
            "periods": periods, "close_only": close_only, "years": years, "trades": trades, "quiet": quiet,
            "entries_b": len(entries_b), "gated_ok": gated_ok, "placebo": placebo, "pvals": pvals, "look": look,
            "problems": problems, "n_starts": len(clk.starts), "hd_med": float(np.median(clk.hd))}


# ---------- today ----------

def _now(r: dict) -> list[dict]:
    close, adx, pos = r["close"], r["adx"], r["pos"]
    c = close.values
    last = float(c[-1])
    hi_next = float(np.max(c[-N:]))
    lo_next = float(np.min(c[-N:]))
    a_now = adx[-1]

    def word(v) -> str:
        return "持有" if v == v and v > 0.5 else "現金"

    a1 = word(pos["a1"][-1])
    if a1 == "持有":
        a1_text = f"持有；收盤 < {lo_next:,.0f} 美元（離最後收盤 {(lo_next / last - 1) * 100:+.1f}%）轉成現金"
    else:
        gate = "ADX 目前低於 25，閘門開著" if _ok(a_now) and a_now < ADX_CUT else "ADX 目前不低於 25，閘門關著，突破也不進場"
        a1_text = f"現金；下一根收盤 > {hi_next:,.0f} 美元（{(hi_next / last - 1) * 100:+.1f}%）才可能進場，{gate}"
    b15 = word(pos["b15"][-1])
    lv = lo_next if b15 == "持有" else hi_next
    ma = word(pos["ma50"][-1])
    ma_lv = float(c[-49:].sum() / 49)
    return [
        {"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"},
        {"label": "ADX(14)（趨勢強度，低於 25 = 盤整）", "value": f"{a_now:.1f}" if _ok(a_now) else "—"},
        {"label": SHORT["a1"], "value": a1_text},
        {"label": SHORT["b15"], "value": f"{b15}；收盤 {'<' if b15 == '持有' else '>'} {lv:,.0f} 美元會翻轉"},
        {"label": SHORT["ma50"], "value": f"{ma}；收盤 {'≤' if ma == '持有' else '>'} 約 {ma_lv:,.0f} 美元會翻轉"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    s, close = r["s"], r["close"]
    a, b, m = s["a1"], s["b15"], s["ma50"]
    n_c = a["c_n"]
    tr = r["trades"]

    pass_rank = a["low"] >= m["low"]
    pass_loss = a["c_deep_med"] >= m["c_deep_med"]
    pass_gate = a["low"] > b["low"]
    pass_placebo = r["placebo"] <= P_BAR
    dirs = [(r["periods"][name]["a1"] is not None and r["periods"][name]["ma50"] is not None
             and r["periods"][name]["a1"] >= r["periods"][name]["ma50"]) for name, _, _ in PERIODS]
    pass_dir = all(dirs)
    passed = pass_rank and pass_loss and pass_gate and pass_placebo and pass_dir

    def trade_cells(k: str) -> list:
        x = tr[k]
        rate = f"{x['wins'] / x['trades'] * 100:.0f}%" if x.get("trades") else "—"
        return [f"{x['share']:.0f}%" if x.get("share") is not None else "—",
                f"{x['trades']} 次（一年 {x['trades_per_year']:.1f} 次）｜{rate}"]

    rows1 = []
    for k in RULES:
        x = s[k]
        rows1.append([NAMES[k], _share(x["low"]), " / ".join(_share(v) for v in x["checks"]),
                      f"{_pp(x['c_edge'])}", f"{_pct(x['c_deep_med'])}｜{_pct(x['c_deep_worst'])}",
                      f"{_pct(x['d_deep_med'])}"] + trade_cells(k))
    rows1.append(["對照：一直抱著不動", "—", "—", "—", f"{_pct(r['hd_med'])}", "—", "100%", "—"])

    rows2 = []
    for k in RULES:
        x = s[k]
        rows2.append([NAMES[k], f"{_of(x['d_n'], x['d_beat'])}｜{_pp(x['d_edge'])}", _of(x["c_n"], x["c_beat"]),
                      f"{_of(x['p_n'], x['p_beat'])}｜{_pp(x['p_edge'])}",
                      f"{_of(x['p_clock_n'], x['p_clock_beat'])}"]
                     + [_of(*x["o_each"][c]) if c in x["o_each"] else "—" for c in ("ETH", "SOL", "BNB", "XRP", "MSTR")])

    rows3 = []
    for name, _, _ in PERIODS:
        p = r["periods"][name]
        rows3.append([name] + [_pct(p[k]) for k in ("a1", "b15", "ma50", "hold")]
                     + ["是" if p["a1"] is not None and p["ma50"] is not None and p["a1"] >= p["ma50"] else "否"])
    co = r["close_only"]
    rows3.append([CLOSE_ONLY[0]] + [_pct(co[k]) for k in ("a1", "b15", "ma50", "hold")]
                 + ["是" if co["a1"] is not None and co["ma50"] is not None and co["a1"] >= co["ma50"] else "否"])

    rows4 = []
    for y, v in r["years"].items():
        rows4.append([str(y) + ("（到目前）" if y == close.index[-1].year else "")]
                     + [_pct(v[k]) for k in ("a1", "b15", "ma50", "hold")]
                     + [_pp(v["a1"] - v["b15"]), _pp(v["a1"] - v["ma50"])])
    diffs = sorted(((v["a1"] - v["ma50"], y) for y, v in r["years"].items()), reverse=True)
    gate_diffs = sorted(((v["a1"] - v["b15"], y) for y, v in r["years"].items()), reverse=True)

    rows5 = [
        ["A1 排名分數 ≥ 50 日均線", f"{_share(a['low'])} 對 {_share(m['low'])}", "過" if pass_rank else "沒過"],
        ["A1 最深虧損中位數不比 50 日均線深", f"{_pct(a['c_deep_med'])} 對 {_pct(m['c_deep_med'])}", "過" if pass_loss else "沒過"],
        ["A1 排名分數 > 不看 ADX 的 15 日突破", f"{_share(a['low'])} 對 {_share(b['low'])}", "過" if pass_gate else "沒過"],
        [f"安慰劑（把 ADX 整串錯開 {PLACEBO_GAP} 天以上，{PLACEBO} 次）≤ 10%", f"{_share(r['placebo'])}", "過" if pass_placebo else "沒過"],
        ["三段時間 A1 每年報酬都 ≥ 50 日均線", "、".join(f"{n}{'是' if d else '否'}" for (n, _, _), d in zip(PERIODS, dirs)),
         "過" if pass_dir else "沒過"],
    ]

    tables = [
        {"caption": (f"公平考試總表。排名分數 = 四項「每年報酬贏過同一天起一直抱著」比例裡最低的一項：比特幣 108 個起算日（2015-01 到 2023-12，"
                     f"照課本做、成本 {COST * 100:.2f}%）／比特幣 {r['n_starts']} 個起算日 × 24 個收盤時間（{n_c:,} 格）／同樣的格子照一般人方式做"
                     f"（收盤後 10 小時才做、成本 {P_COST * 100:.2f}%、{P_MISS * 100:.0f}% 進場漏掉、每個時鐘 {SIMS} 次）／其他資產（ETH、SOL、BNB、XRP、MSTR）。"
                     f"最深虧損看 {n_c:,} 格的中位數｜最深的一格。持有時間、交易次數、勝率是 UTC 0 點日線從 {STATS_FROM} 起算、已結束的交易。"),
         "columns": ["規則", "排名分數（四項最低）", "四項：起算日 / 起算日×時鐘 / 一般人 / 其他資產", "起算日×時鐘：每年報酬比抱著多（中位數）",
                     "最深虧損：中位數｜最深一格", "108 個起算日的最深虧損中位數", "持有時間", "交易次數｜勝率"],
         "rows": rows1},
        {"caption": ("四項檢查的細數：每格是「幾個裡幾個報酬贏過一直抱著」，後面的百分點是每年報酬比抱著多的中位數。"
                     "「台灣晚上 10 點」只看那一個收盤時間的一般人模擬。其他資產各自從所有規則都有值的月份起算到三年前。"),
         "columns": ["規則", "比特幣 108 個起算日｜差距", "起算日×時鐘", "一般人｜差距", "只看台灣晚上 10 點"]
                    + ["ETH", "SOL", "BNB", "XRP", "MSTR"],
         "rows": rows2},
        {"caption": ("分段看（比特幣、照課本做、UTC 0 點日線，每段裡的每年報酬）。專案裡的比特幣最高／最低價從 2014-06（Bitstamp）才有，"
                     "所以「2011–2016」這段只能從 2014-07 算；最後一列另外用「只有收盤價」近似的 ADX 從 2011 算，只當參考，不列入通過條件"
                     "（只用收盤價時，ADX 的數值和真的 ADX 不一樣，25 這個門檻的意思也不同）。"),
         "columns": ["期間", SHORT["a1"], SHORT["b15"], SHORT["ma50"], SHORT["hold"], "A1 ≥ 50 日均線？"],
         "rows": rows3},
        {"caption": "每一年各自的報酬（比特幣、照課本做），看優勢是從哪幾年來的；後兩欄是 A1 減掉對照的百分點。",
         "columns": ["年", SHORT["a1"], SHORT["b15"], SHORT["ma50"], SHORT["hold"], "A1 − 15 日突破", "A1 − 50 日均線"],
         "rows": rows4},
        {"caption": "事先寫好的通過條件（五項都要過）。",
         "columns": ["條件", "數字", "結果"],
         "rows": rows5},
    ]

    gate_years = "、".join(f"{y}（{d:+.0f} 點）" for d, y in gate_diffs[:3])
    best_years = "、".join(f"{y}（{d:+.0f} 點）" for d, y in diffs[:3])
    worst_years = "、".join(f"{y}（{d:+.0f} 點）" for d, y in diffs[-3:][::-1])
    failed = [t for t, ok in (("排名分數輸 50 日均線", pass_rank), ("最深虧損比 50 日均線深", pass_loss),
                              ("加了 ADX 閘門沒有比單純突破好", pass_gate), ("安慰劑比例超過 10%", pass_placebo),
                              ("三段時間方向不一致", pass_dir)) if not ok]
    head = ("通過：" if passed else "沒有通過：") + (
        f"「盤整後才追 15 日新高」排名分數 {_share(a['low'])}，50 日均線 {_share(m['low'])}，不看 ADX 的 15 日突破 {_share(b['low'])}；"
        f"最深虧損中位數 {_pct(a['c_deep_med'])}（50 日均線 {_pct(m['c_deep_med'])}、15 日突破 {_pct(b['c_deep_med'])}）。")
    body = (f"ADX 閘門把 2015 年以來 {r['entries_b']} 次突破進場裡只留下前一天 ADX<25 的 {r['gated_ok']} 次，持有時間從 "
            f"{tr['b15']['share']:.0f}% 降到 {tr['a1']['share']:.0f}%（50 日均線 {tr['ma50']['share']:.0f}%）；"
            f"把 ADX 時間錯開的安慰劑有 {_share(r['placebo'])} 做得一樣好或更好。")
    tail = ("" if passed else "沒過的地方：" + "、".join(failed) + "。")
    verdict = head + body + tail

    notes = [
        f"A1 比 15 日突破多出最多的年份：{gate_years}；A1 比 50 日均線多最多的年份：{best_years}，輸最多的：{worst_years}。",
        f"2015 年以來比特幣有 {r['quiet'] * 100:.0f}% 的日子 ADX(14) 低於 25（算盤整）。",
        f"安慰劑：比特幣 108 個起算日 A1 每年報酬中位數 {_pct(s['a1']['d_ret_med'])}；ADX 整串錯開後 {PLACEBO} 次的中位數是 "
        f"{_pct(float(np.median(r['pvals'])))}（中間八成 {_pct(float(np.percentile(r['pvals'], 10)))} 到 "
        f"{_pct(float(np.percentile(r['pvals'], 90)))}）。突破的位置不動，只打亂「哪幾天算盤整」，所以盤整天數的比例一樣。",
        "ADX 用 Wilder 的原始算法（14 天），比特幣的最高／最低價：2014-06 到 2017-08-16 用 Bitstamp 美元日線，之後用幣安日線；收盤價一律用儀表板的日線，"
        f"最高／最低價若沒包住收盤價就撐開到收盤價（兩邊收盤價差距的中位數 {r['frame'].attrs['agree'] * 100:.2f}%）。"
        "換收盤時間的格子用該時鐘 24 小時裡最高／最低的整點收盤當作高低價（和 thesis_four 一樣），比真正的高低價窄一點，ADX 會略有不同。",
        "規則和參數（15 天、ADX 14 天、門檻 25、出場不看 ADX）都是跑之前寫好的，沒有調。進場用「收盤高於前 15 天最高收盤」，"
        "出場用「收盤低於前 15 天最低收盤」，都不含當天。",
        f"偷看未來的檢查：把某一天以後的價格全部亂改，再重算一次，那一天以前的部位要一模一樣；測了 {r['look'][0]} 個切點，"
        f"不一樣的有 {r['look'][1]} 個。",
        "相鄰的起算日共用大部分的歷史，24 個收盤時間也只差幾個小時，這些「幾個裡幾個」不是互相獨立的證據。成本很粗略，沒有算滑價、利息和稅；"
        "其他資產是今天還在的大幣和一檔股票，有倖存者偏誤。",
        "過去的結果不保證未來會一樣。這裡只是比較固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ] + r["problems"]

    return {
        "id": "adx_breakout",
        "title": "盤整後才追突破",
        "question": ("只在市場安靜的時候（前一天 ADX(14) 低於 25）才追 15 日新高、跌破 15 日新低就出場——"
                     "這種「先看盤整、再追突破」的規則，有沒有贏過 50 日均線主訊號？ADX 閘門本身有沒有幫上忙？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "StockCharts ChartSchool：Average Directional Index (ADX)",
             "url": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx"},
            {"name": "Investopedia：ADX 指標說明", "url": "https://www.investopedia.com/terms/a/adx.asp"},
            {"name": "Bitstamp 公開 API：BTC/USD 日線", "url": "https://www.bitstamp.net/api/#ohlc_data"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前的收盤）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "Yahoo Finance：MSTR 日線", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
