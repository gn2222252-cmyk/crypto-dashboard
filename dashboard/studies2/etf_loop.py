"""Historical study "etf_loop": the MECHANISM between ETF flows and price. If flows follow price, does a modest rise
trigger a run of inflows that keeps pushing the price (a feedback loop), and what happens when the run ends?
Rules are decided on gold (SPDR Gold Shares, GLD, 2004-2026) and then applied unchanged to the US spot bitcoin and
ether ETFs (Farside daily flows).

Pre-registered tests (verbatim from the owner; thresholds set BEFORE looking at outcomes; chosen on GLD only):

    Loop stages:
    1. Trigger: after the asset rises r over the prior 5/10/20 days (buckets: <-5%, -5..0, 0..+3, +3..+6, >+6% - for
       crypto scale buckets by the asset's own volatility: in units of its trailing 60-day daily sigma x sqrt(n)),
       probability an inflow streak (>=3 consecutive inflow days, and >=4 of next 5) starts within the next 5 days vs
       the base rate.
    2. Persistence: P(inflow on day t+1 | inflow streak length k), k=0..10; flow autocorrelation 1-20 lags; compare
       to shuffled.
    3. Impact: same-day and next-day return on flow days by flow size decile, controlling for prior-day return
       (describe; same-day is not tradable).
    4. Loop rule L (tradable, act at the next close after the flow is published; for crypto the candle dated t+1):
       enter when 20-day return > 0 AND an inflow streak of >=3 days is in progress; exit when 3 consecutive outflow
       days OR 20-day net flow < 0. Compare with: holding; ma50; a PRICE-ONLY TWIN P with similar timing (enter when
       20-day return > 0 and 3 consecutive up closes; exit on 3 consecutive down closes or 20-day return < 0); and a
       placebo where the flow series is circularly shifted (200 draws). Pass on gold = L beats P on
       return/volatility in all three sub-periods (2005-12, 2013-19, 2020-26) AND placebo share <= 10%.
    5. Unwind: after an inflow streak of >= N days ends (N=5, 10; and after a 20-day flow in top decile turns down),
       the asset's return over the next 5/20/60 days vs unconditional and vs same-length price-only runs ending (to
       separate flow unwinding from ordinary momentum reversal).
    6. Divergence: 20-day return in top tercile with 20-day flow <= 0 ("unfunded rise") vs with flow in top tercile
       ("funded rise"); 20-day return in bottom tercile with flow in top tercile ("absorbed fall") vs bottom flow.
       Next 20-day returns, block-bootstrap 90% intervals, sub-period signs.
    7. Crypto only: split new-money issuers (IBIT, FBTC, etc.) from GBTC/ETHE outflows; repeat 4-6 on new-money flow
       (description only - not used to select).
    Rule of selection: report everything; only stage 4/5/6 results that hold on gold in all three sub-periods with
    90% interval excluding zero are carried to BTC/ETH as rules; everything else is description. On BTC/ETH report
    sign match, effect, independent episodes, and ma50 overlay over monthly start dates (textbook + ordinary-person),
    stating clearly the sample is < 2 years.
    Uncertainty: block bootstrap (blocks 20 days for crypto, 60 for gold, 1,000 draws); count independent episodes.
    Today: current streak length, 20-day flow percentile, divergence state for GLD/BTC/ETH - states only.

Details fixed by the author before any result was computed (none changed afterwards):
- Everything lives on the flow-day grid (GLD trading days; Farside US trading days). "n days" = n flow days.
- Gold flow = daily change of tonnes in the trust (etf_flows' SPDR archive); relative flow = change / previous day's
  tonnes (%). Crypto relative flow = USD flow / trailing 252-day (min 60) mean |flow|. Inflow day = flow > 0;
  outflow day = flow < 0 (a zero day is neither and breaks a streak). Materiality version: inflow = relative flow
  > 0.05% (gold) / > 0.25x (crypto), outflow = below minus the same.
- Prices on day t: gold = GLD close of t; crypto = daily candle dated t (closes 00:00 UTC t+1, after the US close).
  Returns that a person could earn start at the earliest action close: GLD's next trading close; the crypto candle
  dated t+1. Perturbation: one more day of delay.
- Stage 1: streak_t = consecutive inflow days ending at t. Event A_t = streak_t < 3 and a streak of >= 3 is reached
  on some day t+1..t+5 (base rate: all days with streak_t < 3). Event B_t = at least 4 of days t+1..t+5 are inflows
  (base rate: all days). Crypto buckets in units of z = r_n / (sigma60 x sqrt(n)), sigma60 = std of the last 60 daily
  close-to-close returns on the flow grid; cuts -1.1, 0, +0.67, +1.33 (gold's 20-day % cuts divided by gold's
  long-run 1%-a-day sigma x sqrt(20)). Main row set: n = 20; n = 5 and 10 shown as top-minus-bottom bucket.
- Stage 2: shuffled = 200 random permutations of the flow days (fixed seed); autocorrelation of relative flow.
- Stage 3: deciles of relative flow over non-zero flow days (full-sample deciles, description). Regression of
  same-day return on flow rank (0..1) + prior-day return, and next-day return on flow rank + same-day return.
- Stage 4: L and P are state machines (in/out) decided at close t, traded at the next action close. 20-day return
  uses the close of t and t-20 flow days; "up close" = close above the previous flow day's close. Costs 0.10% per
  trade (0.30% shown). Return/volatility = yearly compound return / yearly volatility (gold 252, crypto 365 days).
  Placebo: the raw flow series circularly shifted by a random 252..n-252 days (crypto n/4..3n/4), 200 draws,
  placebo share = share of shifted-flow L whose return/volatility >= the real L. Extra (not part of pass): 90%
  block-bootstrap interval of L minus P return/volatility.
- Stage 5: a streak "ends" on the first non-inflow day e after >= N inflow days; price-only twin: the first
  non-up close after >= N up closes. Top-decile turn: 20-day flow percentile (trailing 252 days, as etf_flows) was
  > 90% on e-1 and the 20-day flow falls on e, no such event in the previous 20 days; price twin: the same on the
  20-day return. Effect = mean next-h return after the event minus the unconditional mean (same sample), h = 5/20/60.
  To be carried, the flow event must also differ from its price twin in the same direction (full sample).
- Stage 6: terciles = trailing 252-day mid-rank of the 20-day return and of the 20-day flow (min 120 values).
  D1 = next-20 after unfunded rise minus after funded rise; D2 = absorbed fall minus fall with flow in the bottom
  tercile.
- Stage 7: new money = Farside Total minus GBTC (bitcoin) / minus ETHE (ether).
- Carried result -> crypto ma50 overlay: negative effect = hold only when ma50 says hold AND not within h days after
  the event; positive = ma50 OR within h days after. Stage 4 L is always overlaid on ma50 for crypto as description.
  Ordinary person on crypto: one more day of delay, 0.25% cost, 10% of entries missed (robust_trend._missed, 200
  simulations).
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import flows2
from dashboard.studies2 import etf_flows as ef
from dashboard.studies2 import main_signal_check as mc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

PAPER = "https://arxiv.org/abs/2405.12768"
MEMORY = "https://insights.ultumus.com/feed/etf-flows-have-memory-but-leveraged-products-remember-differently"
CUTS_GOLD = (-5.0, 0.0, 3.0, 6.0)
CUTS_Z = (-1.1, 0.0, 0.67, 1.33)
MAT = {"GLD": 0.05, "BTC": 0.25, "ETH": 0.25}
OLD = {"BTC": "GBTC", "ETH": "ETHE"}
BOOT, PLACEBO, SIMS = 1000, 200, 200
BLOCK = {"GLD": 60, "BTC": 20, "ETH": 20}
SEED = 20051118
COST, COST_HI, P_COST, P_MISS = 0.001, 0.003, 0.0025, 0.10
PERIODS = ef.PERIODS
GOLD_FROM = "2005-01-01"
CN = {"GLD": "黃金 GLD", "BTC": "比特幣 ETF", "ETH": "以太幣 ETF"}
YEARS = {"GLD": mc.SHARE_YEAR, "BTC": mc.YEAR, "ETH": mc.YEAR}

DESIGN_EN = " ".join(line.strip() for line in __doc__.split("chosen on GLD only):")[1]
                     .split("Details fixed by the author")[0].strip().splitlines())


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pp(v, d: int = 1) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{d}f} 點"


def _pc(v, d: int = 1) -> str:
    return "—" if not _ok(v) else f"{float(v):+.{d}f}%"


def _p0(v) -> str:
    return "—" if not _ok(v) else f"{float(v) * 100:.0f}%"


def _r2(v) -> str:
    return "—" if not _ok(v) else f"{float(v):.2f}"


_d = ef._d


# ---------- data ----------

def _farside(asset: str) -> tuple[pd.DataFrame, str]:
    """Daily flows (USD m) per issuer + Total; own store as fallback, then etf_flows' total-only store."""
    store = common.STORE / f"etf_loop_{asset.lower()}_frame.json"
    today = common.today_utc()
    try:
        f = flows2._etf_frame(asset)
        f = f[f.index < today]
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            store.write_text(f.to_json(orient="split"), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        prob = ""
    except Exception as exc:  # noqa: BLE001 - blocked: use stored data, never work around
        prob = f"這次 Farside {asset} 取不到（{type(exc).__name__}），改用上次存下的資料。"
        if store.exists():
            f = pd.read_json(io.StringIO(store.read_text(encoding="utf-8")), orient="split")
            f.index = [str(i)[:10] for i in f.index]
        else:
            hist = common.accumulate(f"etf_flows_{asset.lower()}", [])
            if not hist:
                raise RuntimeError(f"Farside {asset} 取不到，也沒有存檔") from exc
            f = pd.DataFrame({"Total": {d: v for d, v in hist}})
            prob += "（存檔只有合計，沒有分發行商。）"
    f.index = pd.to_datetime(f.index)
    return f.sort_index().astype(float), prob


# ---------- the panel ----------

def _streak(b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(b), dtype=int)
    c = 0
    for i, x in enumerate(b):
        c = c + 1 if x else 0
        out[i] = c
    return out


def _panel(flow: pd.Series, rel: pd.Series, f20rel: pd.Series, own: pd.Series, entry: pd.Series,
           mat: float) -> pd.DataFrame:
    """One row per flow day. own = price at day t's close, entry = price at the earliest action close."""
    p = pd.DataFrame(index=flow.index)
    p["flow"], p["rel"] = flow.values, rel.values
    p["inf"], p["out"] = (flow > 0).values, (flow < 0).values
    p["inf_m"], p["out_m"] = (rel > mat).values, (rel < -mat).values
    p["s_in"], p["s_out"] = _streak(p["inf"].values), _streak(p["out"].values)
    p["s_in_m"], p["s_out_m"] = _streak(p["inf_m"].values), _streak(p["out_m"].values)
    p["F20"] = flow.rolling(20, min_periods=20).sum().values
    p["f20rel"] = f20rel.values
    p["pF"] = ef._rank(p["f20rel"].values.astype(float))
    o = pd.Series(own.values.astype(float))
    p["close"] = o.values
    p["ret0"] = (o / o.shift(1) - 1).values * 100
    p["ret_prev"] = p["ret0"].shift(1)
    p["ret_next"] = p["ret0"].shift(-1)
    for n in (5, 10, 20):
        p[f"r{n}"] = (o / o.shift(n) - 1).values * 100
    p["sig60"] = p["ret0"].rolling(60, min_periods=60).std(ddof=1)
    p["s_up"], p["s_dn"] = _streak((p["ret0"] > 0).values), _streak((p["ret0"] < 0).values)
    p["pR"] = ef._rank(p["r20"].values.astype(float))
    e = pd.Series(entry.values.astype(float))
    for h in (5, 20, 60):
        p[f"fwd{h}"] = (e.shift(-h) / e - 1).values * 100
    return p


# ---------- stage 1-3 ----------

def _future_max(x: np.ndarray, k: int = 5) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n - k):
        out[i] = x[i + 1:i + 1 + k].max()
    return out


def _trigger_cols(p: pd.DataFrame, suffix: str = "") -> tuple[np.ndarray, np.ndarray]:
    s = p["s_in" + suffix].values
    inf = p["inf" + suffix].values.astype(float)
    fm = _future_max(s.astype(float))
    a = np.where(np.isfinite(fm) & (s < 3), (fm >= 3).astype(float), np.nan)
    nxt = pd.Series(inf[::-1]).rolling(5, min_periods=5).sum().values[::-1]
    b = np.full(len(s), np.nan)
    b[:-1] = (nxt[1:] >= 4).astype(float)
    b[-5:] = np.nan
    return a, b


def _bucket(p: pd.DataFrame, n: int, crypto: bool) -> np.ndarray:
    if crypto:
        x = p[f"r{n}"].values / (p["sig60"].values * math.sqrt(n))
        cuts = CUTS_Z
    else:
        x = p[f"r{n}"].values
        cuts = CUTS_GOLD
    out = np.full(len(x), -1)
    ok = np.isfinite(x)
    out[ok] = np.searchsorted(np.array(cuts), x[ok], side="right")
    return out


def _stage1(p: pd.DataFrame, crypto: bool, suffix: str = "") -> dict:
    a, b = _trigger_cols(p, suffix)
    res = {"baseA": np.nanmean(a), "baseB": np.nanmean(b)}
    for n in (5, 10, 20):
        bk = _bucket(p, n, crypto)
        for j in range(5):
            m = bk == j
            ma, mb = m & np.isfinite(a), m & np.isfinite(b)
            res[(n, j)] = (a[ma].mean() if ma.sum() >= 5 else np.nan, b[mb].mean() if mb.sum() >= 5 else np.nan,
                           int(ma.sum()))
    return res


def _stage2(p: pd.DataFrame, seed: int) -> dict:
    inf = p["inf"].values
    s = p["s_in"].values
    nxt = np.append(inf[1:], False).astype(float)
    valid = np.arange(len(inf)) < len(inf) - 1
    cond = {}
    for k in range(11):
        m = valid & (s == k)
        cond[k] = (nxt[m].mean() if m.sum() >= 10 else np.nan, int(m.sum()))
    base = inf.mean()
    rel = p["rel"].values.astype(float)
    rel = rel[np.isfinite(rel)]
    x = rel - rel.mean()
    den = (x * x).sum()
    acf = {L: float((x[L:] * x[:-L]).sum() / den) for L in range(1, 21)}
    rng = np.random.default_rng(seed)
    sh_acf, sh_k3 = [], []
    for _ in range(PLACEBO):
        y = rng.permutation(x)
        sh_acf.append(abs((y[1:] * y[:-1]).sum() / den))
        z = rng.permutation(inf)
        sz = _streak(z)
        nz = np.append(z[1:], False)
        m = (sz >= 3)[:-1]
        sh_k3.append(nz[:-1][m].mean() if m.any() else np.nan)
    k3 = valid & (s >= 3)
    return {"cond": cond, "base": base, "acf": acf, "band": float(np.percentile(sh_acf, 95)),
            "k3": nxt[k3].mean() if k3.any() else np.nan,
            "sh_k3": (float(np.nanpercentile(sh_k3, 5)), float(np.nanpercentile(sh_k3, 95)))}


def _ols(y: np.ndarray, cols: list) -> float:
    X = np.column_stack([np.ones(len(y))] + cols)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < 30:
        return float("nan")
    return float(np.linalg.lstsq(X[m], y[m], rcond=None)[0][1])


def _stage3(p: pd.DataFrame) -> dict:
    q = p[(p["flow"] != 0) & p["rel"].notna()]
    rk = q["rel"].rank(pct=True).values
    dec = np.minimum((rk * 10).astype(int), 9)
    out = {}
    for col, ctrl in (("ret0", "ret_prev"), ("ret_next", "ret0")):
        y = q[col].values
        top, bot = y[(dec == 9) & np.isfinite(y)], y[(dec == 0) & np.isfinite(y)]
        out[col] = (top.mean() - bot.mean() if len(top) and len(bot) else np.nan,
                    _ols(y, [rk, q[ctrl].values]))
    return out


# ---------- stage 4: rules ----------

def _machine(enter: np.ndarray, leave: np.ndarray, ready: np.ndarray) -> np.ndarray:
    out = np.full(len(enter), np.nan)
    state = 0.0
    for i in range(len(enter)):
        if not ready[i]:
            continue
        if state == 0.0 and enter[i]:
            state = 1.0
        elif state == 1.0 and leave[i]:
            state = 0.0
        out[i] = state
    return out


def _rule_L(flow: np.ndarray, r20: np.ndarray, mat_rel: np.ndarray | None = None, thr: float = 0.0) -> np.ndarray:
    if mat_rel is None:
        inf, out = flow > 0, flow < 0
    else:
        inf, out = mat_rel > thr, mat_rel < -thr
    s_in, s_out = _streak(inf), _streak(out)
    f20 = pd.Series(flow).rolling(20, min_periods=20).sum().values
    ready = np.isfinite(f20) & np.isfinite(r20)
    leave = (s_out >= 3) | (f20 < 0)
    return _machine((r20 > 0) & (s_in >= 3) & ~leave, leave, ready)


def _rule_P(p: pd.DataFrame) -> np.ndarray:
    r20 = p["r20"].values
    return _machine((r20 > 0) & (p["s_up"].values >= 3), (p["s_dn"].values >= 3) | (r20 < 0), np.isfinite(r20))


def _perf(net: np.ndarray, year: int) -> dict:
    if len(net) < 20:
        return {"ret": np.nan, "vol": np.nan, "ratio": np.nan, "dd": np.nan}
    g = float(np.prod(1 + net))
    ret = (g ** (year / len(net)) - 1) * 100 if g > 0 else -100.0
    vol = float(net.std(ddof=1) * math.sqrt(year) * 100)
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else 0.0, "dd": dd}


def _ratio_rows(net: np.ndarray, year: int) -> np.ndarray:
    n = net.shape[-1]
    g = np.prod(1 + net, axis=-1)
    ret = np.where(g > 0, np.abs(g) ** (year / n) - 1, -1.0) * 100
    vol = net.std(axis=-1, ddof=1) * math.sqrt(year) * 100
    return np.where(vol > 1e-9, ret / np.where(vol > 1e-9, vol, 1.0), 0.0)


def _boot_idx(n: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(math.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), (BOOT, k))
    return (starts[:, :, None] + np.arange(block)).reshape(BOOT, -1)[:, :n].astype(np.int32)


def _place(t: np.ndarray, flow_index: pd.DatetimeIndex, close: pd.Series, lag: int, crypto: bool) -> pd.Series:
    if crypto:
        return ef._calendar_target(pd.Series(t, index=flow_index), lag, close.index)
    return pd.Series(t, index=close.index).shift(lag)


class Book:
    """Daily nets of several targets on one close series, from a common start."""

    def __init__(self, close: pd.Series, targets: dict, year: int, start=None):
        self.close, self.year = close, year
        self.change = close.pct_change().fillna(0.0).values
        valid = np.ones(len(close), bool)
        for t in targets.values():
            valid &= t.notna().values
        first = close.index[int(np.argmax(valid))]
        if start is not None:
            first = max(first, pd.Timestamp(start))
        self.start = first
        self.keep = (close.index > first)
        self.targets, self.net = targets, {}
        for k, t in targets.items():
            self.add(k, t)

    def add(self, k: str, t: pd.Series, cost: float = COST):
        self.targets[k] = t
        self.net[k] = te._run(t.shift(1).fillna(0.0).values, self.change, cost)[0]
        self.net[(k, "hi")] = te._run(t.shift(1).fillna(0.0).values, self.change, COST_HI)[0]

    def perf(self, k, mask=None) -> dict:
        m = self.keep if mask is None else (self.keep & mask)
        out = _perf(self.net[k][m], self.year)
        out["time"] = float(self.targets[k].shift(1).fillna(0.0).values[m].mean() * 100) if k in self.targets else np.nan
        return out


def _stage4(p: pd.DataFrame, close: pd.Series, asset: str, flow: pd.Series, lag: int = 1, start=None,
            placebo: bool = True) -> dict:
    crypto = asset != "GLD"
    year = YEARS[asset]
    L = _rule_L(p["flow"].values, p["r20"].values)
    Lm = _rule_L(p["flow"].values, p["r20"].values, p["rel"].values, MAT[asset])
    P = _rule_P(p)
    ma = close.rolling(50, min_periods=50).mean()
    ma50 = (close > ma).astype(float).where(ma.notna())
    tL, tP, tLm = (_place(x, p.index, close, lag, crypto) for x in (L, P, Lm))
    book = Book(close, {"hold": pd.Series(1.0, index=close.index), "ma50": ma50, "P": tP, "L": tL}, year, start)
    book.add("Lm", tLm)
    book.add("ma50L", ma50 * tL)
    res = {"book": book, "L": L, "P": P, "Lm": Lm, "entries": int(np.nansum(np.diff(np.nan_to_num(L)) > 0))}
    nL, nP = book.net["L"][book.keep], book.net["P"][book.keep]
    idx = _boot_idx(len(nL), BLOCK[asset], SEED + 5)
    diff = _ratio_rows(nL[idx], year) - _ratio_rows(nP[idx], year)
    res["ci"] = (float(np.percentile(diff, 5)), float(np.percentile(diff, 95)))
    res["diff"] = book.perf("L")["ratio"] - book.perf("P")["ratio"]
    if not crypto:
        res["sub"] = {lab: {k: book.perf(k, (close.index >= a) & (close.index <= b))["ratio"]
                            for k in ("hold", "ma50", "P", "L", "Lm", "ma50L")} for lab, a, b in PERIODS}
    if placebo:
        rng = np.random.default_rng(SEED + 7)
        fl, r20 = p["flow"].values, p["r20"].values
        n = len(fl)
        lo = 252 if not crypto else n // 4
        hi = n - lo
        real = book.perf("L")["ratio"]
        ge = 0
        m = book.keep
        for _ in range(PLACEBO):
            k = int(rng.integers(lo, max(lo + 1, hi)))
            t = _place(_rule_L(np.roll(fl, k), r20), p.index, close, lag, crypto)
            net = te._run(t.shift(1).fillna(0.0).values, book.change, COST)[0]
            ge += _perf(net[m], year)["ratio"] >= real
        res["placebo"] = ge / PLACEBO * 100
    return res


# ---------- stage 5 / 6 ----------

def _ends(streak: np.ndarray, N: int) -> np.ndarray:
    prev = np.concatenate([[0], streak[:-1]])
    return (prev >= N) & (streak == 0)


def _turn(rank: np.ndarray, level: np.ndarray) -> np.ndarray:
    prev_r = np.concatenate([[np.nan], rank[:-1]])
    prev_l = np.concatenate([[np.nan], level[:-1]])
    raw = (prev_r > 0.9) & (level < prev_l)
    out = np.zeros(len(raw), bool)
    last = -10 ** 9
    for i in np.flatnonzero(raw):
        if i - last > 20:
            out[i] = True
        last = i
    return out


EVENTS = (("N5", "連續流入 ≥ 5 天結束"), ("N10", "連續流入 ≥ 10 天結束"), ("TOP", "20 日流量在前 10% 後轉弱"))


def _events(p: pd.DataFrame, suffix: str = "") -> dict:
    s_in = p["s_in" + suffix].values if suffix else p["s_in"].values
    return {"N5": (_ends(s_in, 5), _ends(p["s_up"].values, 5)),
            "N10": (_ends(s_in, 10), _ends(p["s_up"].values, 10)),
            "TOP": (_turn(p["pF"].values, p["F20"].values), _turn(p["pR"].values, p["r20"].values))}


def _groups(p: pd.DataFrame) -> dict:
    pr, pf, F = p["pR"].values, p["pF"].values, p["F20"].values
    ok = np.isfinite(pr) & np.isfinite(pf)
    top_r, bot_r = ok & (pr > 2 / 3), ok & (pr <= 1 / 3)
    top_f, bot_f = ok & (pf > 2 / 3), ok & (pf <= 1 / 3)
    return {"unfunded": top_r & (F <= 0), "funded": top_r & top_f, "absorbed": bot_r & top_f, "fallout": bot_r & bot_f}


def _diff(y: np.ndarray, a: np.ndarray, b: np.ndarray | None) -> float:
    m = np.isfinite(y)
    ya = y[a & m]
    yb = y[m] if b is None else y[b & m]
    return float(ya.mean() - yb.mean()) if len(ya) >= 2 and len(yb) >= 2 else float("nan")


def _boot_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray | None, idx: np.ndarray) -> tuple[float, float]:
    Y = y[idx]
    fin = np.isfinite(Y)
    Y0 = np.where(fin, Y, 0.0)
    A = a[idx] & fin
    B = fin if b is None else (b[idx] & fin)
    na, nb = A.sum(1), B.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (Y0 * A).sum(1) / na - (Y0 * B).sum(1) / nb
    v = v[(na >= 2) & (nb >= 2)]
    if len(v) < BOOT // 2:
        return float("nan"), float("nan")
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def _stage56(p: pd.DataFrame, asset: str, suffix: str = "", boot: bool = True) -> dict:
    ev = _events(p, suffix)
    gr = _groups(p)
    idx = _boot_idx(len(p), BLOCK[asset], SEED + 11) if boot else None
    out = {}
    dates = p.index
    for key, _ in EVENTS:
        fe, pe = ev[key]
        for h in (5, 20, 60):
            y = p[f"fwd{h}"].values
            r = {"est": _diff(y, fe, None), "price": _diff(y, pe, None), "vs_price": np.nan,
                 "n": ef._episodes(fe & np.isfinite(y), 20), "n_price": ef._episodes(pe & np.isfinite(y), 20)}
            r["vs_price"] = r["est"] - r["price"] if _ok(r["est"]) and _ok(r["price"]) else np.nan
            r["ci"] = _boot_diff(y, fe, None, idx) if boot else (np.nan, np.nan)
            r["sub"] = {lab: _diff(y, fe & (dates >= a) & (dates <= b), (dates >= a) & (dates <= b))
                        for lab, a, b in PERIODS}
            out[(key, h)] = r
    y = p["fwd20"].values
    for key, a, b in (("D1", "unfunded", "funded"), ("D2", "absorbed", "fallout")):
        ga, gb = gr[a], gr[b]
        r = {"est": _diff(y, ga, gb), "ci": _boot_diff(y, ga, gb, idx) if boot else (np.nan, np.nan),
             "n": (ef._episodes(ga & np.isfinite(y), 20), ef._episodes(gb & np.isfinite(y), 20)),
             "sub": {lab: _diff(y, ga & (dates >= s) & (dates <= e), gb & (dates >= s) & (dates <= e))
                     for lab, s, e in PERIODS},
             "means": (_diff(y, ga, None), _diff(y, gb, None))}
        out[key] = r
    out["events"], out["groups"] = ev, gr
    return out


def _holds(r: dict, needs_price: bool) -> bool:
    signs = [np.sign(r["sub"][lab]) for lab, _, _ in PERIODS if _ok(r["sub"][lab])]
    if len(signs) < 3 or len(set(signs)) != 1 or signs[0] == 0 or np.sign(r["est"]) != signs[0]:
        return False
    lo, hi = r["ci"]
    if not (_ok(lo) and _ok(hi) and (lo > 0 or hi < 0)):
        return False
    if needs_price and not (_ok(r["vs_price"]) and np.sign(r["vs_price"]) == signs[0]):
        return False
    return True


# ---------- crypto overlay over monthly start dates ----------

def _overlay_target(flag: np.ndarray, h: int, positive: bool, p: pd.DataFrame, close: pd.Series,
                    ma50: pd.Series, lag: int) -> pd.Series:
    win = pd.Series(flag.astype(float)).rolling(h, min_periods=1).max().values
    t = _place(win, p.index, close, lag, True).fillna(0.0)
    return ((ma50 + t).clip(upper=1.0) if positive else ma50 * (1 - t)).where(ma50.notna())


def _grid(close: pd.Series, arms: dict, start, year: int, seed: int) -> dict:
    """Monthly start dates: textbook (lag as given) and ordinary person (already-lagged targets, 0.25%, 10% missed)."""
    change = close.pct_change().fillna(0.0).values
    last = close.index[-1] - pd.Timedelta(days=180)
    starts, labels = mc._month_starts(close.index, pd.Timestamp(start) + pd.Timedelta(days=1), last)
    if len(starts) == 0:
        return {"n": 0}
    out = {"n": len(starts), "first": labels[0], "last": labels[-1]}
    for mode in ("text", "person"):
        res = {}
        for name, (t_text, t_person) in arms.items():
            if mode == "text":
                net = te._run(t_text.shift(1).fillna(0.0).values, change, COST)[0]
                res[name] = mc._suffix(mc._equity(net), starts, year)[0][:, 0]
            else:
                w = t_person.shift(1).fillna(0.0).values
                rng = np.random.default_rng(seed + len(res))
                pos = rt._missed(w, P_MISS, rng, SIMS)
                eq = np.ones((len(w) + 1, SIMS))
                for s in range(SIMS):
                    eq[:, s] = mc._equity(te._run(pos[:, s], change, P_COST)[0])
                res[name] = mc._suffix(eq, starts, year)[0]
        out[mode] = res
    return out


def _grid_cell(g: dict, a: str, b: str) -> str:
    if not g.get("n"):
        return "起算日不夠"
    t, p = g["text"], g["person"]
    wt = int((t[a] > t[b]).sum())
    wp = float((p[a] > p[b]).mean() * 100)
    return (f"{_d(g['first'])[:7]}–{_d(g['last'])[:7]} 共 {g['n']} 個起算日：照課本 {wt} 個贏；"
            f"一般人做法 {wp:.0f}% 的情況贏")


# ---------- look-ahead check ----------

def _decisions(flow: pd.Series, rel: pd.Series, f20rel: pd.Series, own: pd.Series, mat: float) -> pd.DataFrame:
    p = _panel(flow, rel, f20rel, own, own, mat)
    ev = _events(p)
    gr = _groups(p)
    d = pd.DataFrame(index=p.index)
    d["L"] = _rule_L(p["flow"].values, p["r20"].values)
    d["P"] = _rule_P(p)
    for k, (fe, pe) in ev.items():
        d[k], d[k + "p"] = fe.astype(float), pe.astype(float)
    for k, g in gr.items():
        d[k] = g.astype(float)
    d["b20"] = _bucket(p, 20, False).astype(float)
    d["pF"] = p["pF"]
    return d


def _scramble_check(flow, rel, f20rel, own, mat) -> tuple[int, int]:
    full = _decisions(flow, rel, f20rel, own, mat)
    rng = np.random.default_rng(SEED + 99)
    good = 0
    cuts = (0.4, 0.6, 0.8)
    for c in cuts:
        k = int(len(flow) * c)
        def scr(s):
            v = s.values.astype(float).copy()
            v[k:] = rng.permutation(v[k:]) * rng.uniform(0.5, 1.5)
            return pd.Series(v, index=s.index)
        part = _decisions(scr(flow), scr(rel), scr(f20rel), scr(own), mat)
        a, b = full.iloc[:k].values, part.iloc[:k].values
        same = np.all((a == b) | (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
        good += bool(same)
    return good, len(cuts)


# ---------- the whole computation ----------

def _gold_inputs(g: pd.DataFrame):
    flow = g["tonnes"].diff()
    rel = flow / g["tonnes"].shift(1) * 100
    f20rel = flow.rolling(20, min_periods=20).sum() / g["tonnes"].shift(20) * 100
    return flow, rel, f20rel, g["price"]


def _crypto_inputs(flow: pd.Series, close: pd.Series):
    scale = flow.abs().rolling(252, min_periods=60).mean()
    rel = flow / scale
    f20rel = flow.rolling(20, min_periods=20).sum() / scale
    own = pd.Series(close.reindex(flow.index).values, index=flow.index)
    return rel, f20rel, own


def _asset(asset: str, flow: pd.Series, close: pd.Series, holdings_rel=None, since=None) -> dict:
    crypto = asset != "GLD"
    if crypto:
        rel, f20rel, own = _crypto_inputs(flow, close)
        e1 = pd.Series(close.reindex(flow.index + pd.Timedelta(days=1)).values, index=flow.index)
        e2 = pd.Series(close.reindex(flow.index + pd.Timedelta(days=2)).values, index=flow.index)
    else:
        rel, f20rel, own = holdings_rel
        e1, e2 = own.shift(-1), own.shift(-2)
    full = _panel(flow, rel, f20rel, own, e1, MAT[asset])
    p2 = _panel(flow, rel, f20rel, own, e2, MAT[asset])
    if since is not None:  # gold: the launch weeks (seed creations of 10-600% a day) only warm up the signals
        p, p2 = full[full.index >= since], p2[p2.index >= since]
    else:
        p = full
    out = {"p": p, "close": close, "flow": flow}
    out["s1"] = _stage1(p, crypto)
    out["s1m"] = _stage1(p, crypto, "_m")
    out["s2"] = _stage2(p, SEED + 3)
    out["s3"] = _stage3(p)
    out["s4"] = _stage4(full, close, asset, flow, start=since)
    out["s4_lag2"] = _stage4(full, close, asset, flow, lag=2, start=out["s4"]["book"].start, placebo=False)
    out["s56"] = _stage56(p, asset)
    out["s56_lag2"] = _stage56(p2, asset, boot=False)
    out["s56m"] = _stage56(p, asset, "_m", boot=False)
    return out


def _compute() -> dict:
    t0 = time.time()
    problems = []
    g, prob = ef._gld()
    if prob:
        problems.append(prob)
    flow_g, rel_g, f20_g, own_g = _gold_inputs(g)
    gold = _asset("GLD", flow_g, g["price"], (rel_g, f20_g, own_g), since=GOLD_FROM)
    gold["scramble"] = _scramble_check(flow_g, rel_g, f20_g, own_g, MAT["GLD"])

    carried = []
    s4 = gold["s4"]
    s4_pass = all(_ok(s4["sub"][lab]["L"]) and _ok(s4["sub"][lab]["P"]) and s4["sub"][lab]["L"] > s4["sub"][lab]["P"]
                  for lab, _, _ in PERIODS) and s4["placebo"] <= 10
    s4_carry = s4_pass and (s4["ci"][0] > 0)
    if s4_carry:
        carried.append(("L", None))
    for key, _ in EVENTS:
        for h in (5, 20, 60):
            if _holds(gold["s56"][(key, h)], True):
                carried.append((key, h))
    for key in ("D1", "D2"):
        r = dict(gold["s56"][key])
        r["vs_price"] = np.nan
        if _holds(r, False):
            carried.append((key, 20))

    crypto, prices = {}, {}
    for asset, fn in (("BTC", te._daily_close), ("ETH", lambda: ta._crypto_close("ETHUSDT"))):
        try:
            prices[asset] = fn()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{CN[asset]}的日線取不到（{type(exc).__name__}）。")
    for asset in ("BTC", "ETH"):
        if asset not in prices:
            continue
        try:
            frame, prob = _farside(asset)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{asset} 的 Farside 每日流量這次取不到：{exc}")
            continue
        if prob:
            problems.append(prob)
        close = prices[asset]
        frame = frame[frame.index <= close.index[-1]]
        total = frame["Total"].fillna(0.0)
        c = _asset(asset, total, close)
        c["scramble"] = _scramble_check(total, *_crypto_inputs(total, close), MAT[asset])
        old = OLD[asset]
        if old in frame.columns:
            new = total - frame[old].fillna(0.0)
            nm = _asset(asset, new, close)
            c["new"] = {"s4": nm["s4"], "s56": nm["s56"], "oldsum": float(frame[old].fillna(0.0).sum()),
                        "newsum": float(new.sum())}
        # ma50 overlays over monthly start dates
        b = c["s4"]["book"]
        b2 = c["s4_lag2"]["book"]
        ma50 = b.targets["ma50"]
        arms = {"ma50": (ma50, ma50), "ma50L": (b.targets["ma50L"], b2.targets["ma50L"]),
                "L": (b.targets["L"], b2.targets["L"]), "P": (b.targets["P"], b2.targets["P"])}
        for key, h in carried:
            if key == "L":
                continue
            r = gold["s56"][(key, h)] if key in dict(EVENTS) else gold["s56"][key]
            pos = r["est"] > 0
            flag = (c["s56"]["events"][key][0] if key in dict(EVENTS) else
                    c["s56"]["groups"]["unfunded" if key == "D1" else "absorbed"])
            arms[f"{key}_{h}"] = (_overlay_target(flag, h, pos, c["p"], close, ma50, 1),
                                  _overlay_target(flag, h, pos, c["p"], close, ma50, 2))
        c["grid"] = _grid(close, arms, b.start, mc.YEAR, SEED + (500 if asset == "BTC" else 600))
        crypto[asset] = c
    return {"g": g, "gold": gold, "crypto": crypto, "carried": carried, "s4_pass": s4_pass, "s4_carry": s4_carry,
            "problems": problems, "secs": time.time() - t0}


# ---------- presentation ----------

BUCKET_TEXT = ("< -5%（幣 < -1.1σ）", "-5% 到 0（幣 -1.1σ 到 0）", "0 到 +3%（幣 0 到 +0.67σ）",
               "+3% 到 +6%（幣 +0.67σ 到 +1.33σ）", "> +6%（幣 > +1.33σ）")


def _sub_panel(p: pd.DataFrame, lab_a: str, lab_b: str) -> pd.DataFrame:
    return p[(p.index >= lab_a) & (p.index <= lab_b)]


def _now(r: dict) -> list[dict]:
    out = []
    names = {"unfunded": "沒有資金撐的上漲", "funded": "有資金撐的上漲", "absorbed": "下跌但資金在進場",
             "fallout": "下跌且資金在撤"}

    def state(label: str, c: dict, unit: str) -> dict:
        p = c["p"]
        last = p.index[-1]
        x = p.iloc[-1]
        streak = (f"連續流入第 {int(x['s_in'])} 天" if x["s_in"] > 0 else
                  f"連續流出第 {int(x['s_out'])} 天" if x["s_out"] > 0 else "當天沒有進出")
        grp = [names[k] for k, g in c["s56"]["groups"].items() if g[-1]]
        div = grp[0] if grp else "不屬於四種背離／同步狀態（20 日漲跌或流量在中間）"
        L = c["s4"]["L"][-1]
        lw = "—" if not _ok(L) else ("在場內" if L >= 0.5 else "在場外")
        pf = "—" if not _ok(x["pF"]) else f"第 {x['pF'] * 100:.0f} 百分位"
        f20 = f"{x['f20rel']:+.2f}% 的持有量" if unit == "pct" else f"{x['f20rel']:+.1f} 倍的一年平均單日流量"
        return {"label": f"{label}（最新流量日 {_d(last)}）",
                "value": f"{streak}；20 日淨流入 {f20}，在過去一年排{pf}；狀態：{div}；資金流規則 L {lw}"}

    out.append(state(CN["GLD"], r["gold"], "pct"))
    for asset in ("BTC", "ETH"):
        c = r["crypto"].get(asset)
        out.append(state(f"美國現貨{CN[asset]}", c, "x") if c else
                   {"label": f"美國現貨{CN[asset]}", "value": "這次取不到 Farside 流量資料"})
    out.append({"label": "這幾格的意思", "value": "只描述目前的狀態，不是買賣建議"})
    return out


def _cell_ci(r: dict, d: int = 1) -> str:
    lo, hi = r["ci"]
    return f"{_pp(r['est'], d)}（{_pp(lo, d)} 到 {_pp(hi, d)}）"


def study() -> dict:
    r = _compute()
    G = r["gold"]
    gp = G["p"]
    C = r["crypto"]
    assets = [a for a in ("BTC", "ETH") if a in C]

    def cc(fn) -> list:
        return [fn(C[a]) if a in C else "取不到" for a in ("BTC", "ETH")]

    # ---- table 1: stages 1-3 ----
    rows1 = []
    for j in range(5):
        def s1cell(c, j=j):
            v = c["s1"][(20, j)]
            return f"{_p0(v[0])}（{v[2]} 天）"
        subs = []
        for lab, a, b in PERIODS:
            sp = _sub_panel(gp, a, b)
            v = _stage1(sp, False)[(20, j)]
            subs.append(_p0(v[0]))
        rows1.append([f"① 過去 20 天漲跌 {BUCKET_TEXT[j]} → 之後 5 天內開始連續流入 ≥ 3 天的機率", s1cell(G)] + subs + cc(s1cell))

    def base_cell(c):
        return f"開始 {_p0(c['s1']['baseA'])}｜5 天裡 ≥ 4 天流入 {_p0(c['s1']['baseB'])}"
    base_subs = []
    for lab, a, b in PERIODS:
        s = _stage1(_sub_panel(gp, a, b), False)
        base_subs.append(f"{_p0(s['baseA'])}｜{_p0(s['baseB'])}")
    rows1.append(["① 基準（不管之前漲跌）", base_cell(G)] + base_subs + cc(base_cell))

    def topbot(c, n, which=0, key="s1"):
        s = c[key]
        a, b = s[(n, 4)][which], s[(n, 0)][which]
        return _pp((a - b) * 100 if _ok(a) and _ok(b) else None)
    for n in (5, 10, 20):
        subs = []
        for lab, a, b in PERIODS:
            s = _stage1(_sub_panel(gp, a, b), False)
            x, y = s[(n, 4)][1], s[(n, 0)][1]
            subs.append(_pp((x - y) * 100 if _ok(x) and _ok(y) else None))
        rows1.append([f"① 過去 {n} 天漲最多一格減跌最多一格：之後 5 天裡 ≥ 4 天流入的機率差",
                      topbot(G, n, 1)] + subs + cc(lambda c, n=n: topbot(c, n, 1)))
    rows1.append(["① 只算明顯流入（> 0.05% 持有量／> 0.25 倍平均）：過去 20 天漲最多減跌最多，開始連續流入的機率差",
                  topbot(G, 20, 0, "s1m"), "—", "—", "—"] + cc(lambda c: topbot(c, 20, 0, "s1m")))

    def s2cell(c, ks):
        return "｜".join(f"{k}:{_p0(c['s2']['cond'][k][0])}" for k in ks)
    for ks in ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10)):
        subs = []
        for lab, a, b in PERIODS:
            s = _stage2(_sub_panel(gp, a, b), SEED)
            subs.append("｜".join(f"{k}:{_p0(s['cond'][k][0])}" for k in ks))
        rows1.append([f"② 已經連續流入 k 天，隔天還是流入的機率（k = {ks[0]}–{ks[-1]}）", s2cell(G, ks)] + subs
                     + cc(lambda c, ks=ks: s2cell(c, ks)))

    def shuf(c):
        s = c["s2"]
        return (f"任一天流入 {_p0(s['base'])}；連 ≥ 3 天後隔天 {_p0(s['k3'])}，打亂順序後 {_p0(s['sh_k3'][0])}–{_p0(s['sh_k3'][1])}")
    rows1.append(["② 和打亂順序比（200 次，90% 範圍）", shuf(G), "—", "—", "—"] + cc(shuf))

    def acf(c):
        a = c["s2"]["acf"]
        return (f"1 天 {a[1]:+.2f}｜5 天 {a[5]:+.2f}｜10 天 {a[10]:+.2f}｜20 天 {a[20]:+.2f}；"
                f"1–20 天平均 {np.mean(list(a.values())):+.2f}（打亂後 ±{c['s2']['band']:.2f}）")
    rows1.append(["② 流量和幾天前流量的相關（自相關）", acf(G), "—", "—", "—"] + cc(acf))

    def imp(c, col):
        v = c["s3"][col]
        return f"最大一成減最小一成 {_pp(v[0], 2)}；回歸 {_pp(v[1], 2)}"
    for col, word in (("ret0", "③ 當天報酬（不能交易）"), ("ret_next", "③ 隔天報酬")):
        subs = []
        for lab, a, b in PERIODS:
            subs.append(_pp(_stage3(_sub_panel(gp, a, b))[col][1], 2))
        rows1.append([f"{word}：依流量大小分十組", imp(G, col)] + subs + cc(lambda c, col=col: imp(c, col)))

    # ---- table 2: stage 4 ----
    rows2 = []
    gb = G["s4"]["book"]

    def pf(book, k, mask=None) -> str:
        x = book.perf(k, mask)
        return f"{_r2(x['ratio'])}（每年 {_pc(x['ret'])}，最深 {_pc(x['dd'], 0)}，持有 {x['time']:.0f}%）"

    labels = (("hold", "一直抱著"), ("ma50", "50 日均線"), ("P", "只看價格的同類規則 P"), ("L", "資金流規則 L"),
              ("Lm", "L（只算明顯流入流出）"), ("ma50L", "50 日均線＋L（兩個都說持有才持有）"))
    for k, word in labels:
        subs = [_r2(G["s4"]["sub"][lab][k]) for lab, _, _ in PERIODS]
        rows2.append([f"④ {word}：報酬÷波動", pf(gb, k)] + subs + cc(lambda c, k=k: pf(c["s4"]["book"], k)))

    def hi_cost(c):
        b = c["s4"]["book"]
        return f"L {_r2(_perf(b.net[('L', 'hi')][b.keep], b.year)['ratio'])}｜P {_r2(_perf(b.net[('P', 'hi')][b.keep], b.year)['ratio'])}"
    rows2.append(["④ 成本改 0.30%：L｜P 的報酬÷波動", hi_cost(G), "—", "—", "—"] + cc(hi_cost))

    def lp(c):
        s = c["s4"]
        return f"{_pp(s['diff'], 2)}（90% 區間 {_pp(s['ci'][0], 2)} 到 {_pp(s['ci'][1], 2)}）"
    subs = [_pp(G["s4"]["sub"][lab]["L"] - G["s4"]["sub"][lab]["P"], 2) for lab, _, _ in PERIODS]
    rows2.append(["④ L 減 P（報酬÷波動）", lp(G)] + subs + cc(lp))

    def plc(c):
        return f"{c['s4']['placebo']:.0f}% 的假流量做得一樣好或更好；L 進場 {c['s4']['entries']} 次"
    rows2.append(["④ 安慰劑（流量整段平移 200 次）", plc(G), "—", "—", "—"] + cc(plc))

    def lag2(c):
        b = c["s4_lag2"]["book"]
        return f"L {_r2(b.perf('L')['ratio'])}｜P {_r2(b.perf('P')['ratio'])}"
    rows2.append(["④ 晚一天才做：L｜P", lag2(G), "—", "—", "—"] + cc(lag2))
    pass_word = ("通過" if r["s4_pass"] else "沒通過") + ("，區間也不含零，搬到幣上當規則" if r["s4_carry"] else "")
    rows2.append(["④ 事前標準：三段都 L 贏 P 且安慰劑 ≤ 10%", pass_word, "—", "—", "—"]
                 + ["（描述，非從黃金搬來）" if not r["s4_carry"] else "搬來的規則"] * 2)
    rows2.append(["④ 每月起算日：50 日均線＋L 贏過 50 日均線", "黃金不做（已是 21 年完整樣本）", "—", "—", "—"]
                 + cc(lambda c: _grid_cell(c["grid"], "ma50L", "ma50")))
    rows2.append(["④ 每月起算日：L 贏過 P", "黃金不做", "—", "—", "—"] + cc(lambda c: _grid_cell(c["grid"], "L", "P")))

    # ---- table 3: stages 5 and 6 ----
    rows3 = []
    for key, word in EVENTS:
        for h in (5, 20, 60):
            x = G["s56"][(key, h)]
            subs = [_pp(x["sub"][lab]) for lab, _, _ in PERIODS]
            twin = f"價格版 {_pp(x['price'])}（{x['n_price']} 段）；流量減價格 {_pp(x['vs_price'])}"
            mark = "（成立）" if (key, h) in r["carried"] else ""

            def cx(c, key=key, h=h):
                y = c["s56"][(key, h)]
                return f"{_pp(y['est'])}（{y['n']} 段；價格版 {_pp(y['price'])}）"
            rows3.append([f"⑤ {word}，之後 {h} 天減平常{mark}", f"{_cell_ci(x)}；{x['n']} 段"] + subs + [twin] + cc(cx))
    for key, word in (("D1", "⑥ 沒有資金撐的上漲 減 有資金撐的上漲"), ("D2", "⑥ 下跌但資金進場 減 下跌且資金撤")):
        x = G["s56"][key]
        subs = [_pp(x["sub"][lab]) for lab, _, _ in PERIODS]
        mark = "（成立）" if (key, 20) in r["carried"] else ""

        def cx(c, key=key):
            y = c["s56"][key]
            return f"{_pp(y['est'])}（{y['n'][0]}／{y['n'][1]} 段）"
        rows3.append([f"{word}：之後 20 天{mark}", f"{_cell_ci(x)}；{x['n'][0]}／{x['n'][1]} 段"] + subs
                     + [f"兩組各自：{_pc(x['means'][0])}｜{_pc(x['means'][1])}"] + cc(cx))

    # ---- table 4: stage 7 + checks ----
    rows4 = []

    def new_cell(c, what):
        n = c.get("new")
        if not n:
            return "沒有分發行商資料"
        if what == "sum":
            return f"新資金合計 {n['newsum'] / 100:+,.0f} 億美元；灰度舊信託 {n['oldsum'] / 100:+,.0f} 億美元"
        if what == "L":
            b = n["s4"]["book"]
            return f"L {_r2(b.perf('L')['ratio'])}｜P {_r2(b.perf('P')['ratio'])}；安慰劑 {n['s4']['placebo']:.0f}%"
        if what == "N5":
            y = n["s56"][("N5", 20)]
            return f"{_pp(y['est'])}（{y['n']} 段）"
        y = n["s56"][what]
        return f"{_pp(y['est'])}（{y['n'][0]}／{y['n'][1]} 段）"
    rows4.append(["⑦ 只算新資金（扣掉灰度舊信託）：期間合計", "—"] + cc(lambda c: new_cell(c, "sum")))
    rows4.append(["⑦ 新資金版：L｜P 報酬÷波動", "—"] + cc(lambda c: new_cell(c, "L")))
    rows4.append(["⑦ 新資金版：連續流入 ≥ 5 天結束後 20 天減平常", "—"] + cc(lambda c: new_cell(c, "N5")))
    rows4.append(["⑦ 新資金版：沒資金撐的上漲減有資金撐的上漲", "—"] + cc(lambda c: new_cell(c, "D1")))
    rows4.append(["⑦ 新資金版：下跌但資金進場減下跌且資金撤", "—"] + cc(lambda c: new_cell(c, "D2")))

    def lagrow(c):
        a, b = c["s56"], c["s56_lag2"]
        return (f"⑤ ≥5 天結束 20 天 {_pp(a[('N5', 20)]['est'])} → {_pp(b[('N5', 20)]['est'])}；"
                f"⑥ D1 {_pp(a['D1']['est'])} → {_pp(b['D1']['est'])}；D2 {_pp(a['D2']['est'])} → {_pp(b['D2']['est'])}")
    rows4.append(["晚一天才做（⑤⑥ 從流量日後第二個收盤算）", lagrow(G)] + cc(lagrow))

    def matrow(c):
        a = c["s56m"]
        return (f"⑤ ≥5 天結束 20 天 {_pp(a[('N5', 20)]['est'])}；≥10 天 {_pp(a[('N10', 20)]['est'])}")
    rows4.append(["只算明顯流入（> 0.05% 持有量／> 0.25 倍平均）", matrow(G)] + cc(matrow))
    sg, sn = G["scramble"]
    rows4.append(["防偷看未來：把切點之後的價格和流量打亂，切點之前的所有訊號是否完全不變",
                  f"{sn} 個切點裡 {sg} 個完全不變"] + cc(lambda c: f"{c['scramble'][1]} 個切點裡 {c['scramble'][0]} 個完全不變"))

    def span(c):
        p = c["p"]
        n_days = len(p)
        yrs = (p.index[-1] - p.index[0]).days / 365.25
        return f"{_d(p.index[0])} 到 {_d(p.index[-1])}，{n_days} 個流量日（約 {yrs:.1f} 年）"
    rows4.append(["樣本", span(G)] + cc(span))
    carried_txt = "、".join(f"{k}{'' if h is None else f'（{h} 天）'}" for k, h in r["carried"]) or "沒有"
    rows4.append(["從黃金搬到幣上的結果（三段同方向且 90% 區間不含零）", carried_txt, "—", "—"])

    # ---- verdict ----
    s1 = G["s1"]
    up = s1[(20, 4)][0]
    base = s1["baseA"]
    small = s1[(20, 2)][0]
    k_hi = G["s2"]["cond"][5][0]
    persist = G["s2"]["base"]
    unw = G["s56"][("N5", 20)]
    btc = C.get("BTC")
    yrs_b = (btc["p"].index[-1] - btc["p"].index[0]).days / 365.25 if btc else None
    def vs(x, y) -> str:
        if not (_ok(x) and _ok(y)):
            return "算不出來"
        return "和平常差不多" if abs(x - y) < 0.03 else ("高於平常" if x > y else "低於平常")
    impact = G["s3"]["ret_next"][1]
    s1_word = (f"黃金 GLD {_d(gp.index[0])[:4]}–{_d(gp.index[-1])[:4]} 年：過去 20 天小漲（0 到 3%）後 5 天內開始連續流入的機率 "
               f"{_p0(small)}，{vs(small, base)}（{_p0(base)}），大漲超過 6% 後是 {_p0(up)}；連續流入會延續（已連 5 天，隔天再流入 "
               f"{_p0(k_hi)}，平常 {_p0(persist)}），但流入最大和最小的日子，隔天金價只差 {_pp(impact, 2)}。")
    lose = [lab for lab, _, _ in PERIODS
            if not (G["s4"]["sub"][lab]["L"] > G["s4"]["sub"][lab]["P"])]
    s2_word = (f"連續流入結束後 20 天比平常 {_pp(unw['est'])}，"
               + ("區間在零以下，有回吐" if unw["ci"][1] < 0 else "區間在零以上" if unw["ci"][0] > 0 else "區間含零，看不出回吐")
               + f"；資金流規則 L 報酬÷波動 {_r2(gb.perf('L')['ratio'])}、只看價格的 P {_r2(gb.perf('P')['ratio'])}"
               + (f"，{'、'.join(lose)} 輸給 P" if lose else "，三段都贏 P")
               + f"、安慰劑 {G['s4']['placebo']:.0f}%，"
               + ("通過事前標準" if r["s4_pass"] else "沒有通過事前標準")
               + ("，沒有任何結果搬到幣上。" if not r["carried"] else f"，搬到幣上的有 {len(r['carried'])} 項。"))
    if btc:
        bl = btc["s4"]["book"]
        eth = C.get("ETH")
        ey = f"以太幣約 {(eth['p'].index[-1] - eth['p'].index[0]).days / 365.25:.1f} 年" if eth else "以太幣取不到"
        worse = bl.perf("ma50")["ratio"] > max(bl.perf("L")["ratio"], bl.perf("P")["ratio"])
        s3_word = (f"比特幣 ETF 只有約 {yrs_b:.1f} 年（{ey}），L {_r2(bl.perf('L')['ratio'])}、P {_r2(bl.perf('P')['ratio'])}"
                   + ("，都不如 50 日均線" if worse else "") + "，只能當參考。")
    else:
        s3_word = "比特幣 ETF 流量這次取不到。"
    verdict = s1_word + s2_word + s3_word

    tables = [
        {"caption": ("機制前三段：①小漲會不會引來連續流入、②連續流入會不會一直延續、③流入大的日子價格動得多不多。"
                     "連續流入 = 連續好幾天都有新資金買進 ETF（黃金看信託裡金子的噸數增加，幣看 Farside 每日淨流入為正）。"
                     "百分比是機率，「點」是百分點；幣的漲跌用自己過去 60 天的日波動 σ×√天數 換算。②的 k:數字 = 已連續流入 k 天時隔天還流入的機率。"
                     "③的回歸是流量從最小排到最大時報酬差幾個百分點（扣掉前一天漲跌）。當天報酬只是描述，流量收盤後才公布，不能拿來交易。"),
         "columns": ["項目", "黃金全期", PERIODS[0][0], PERIODS[1][0], PERIODS[2][0], "比特幣 ETF", "以太幣 ETF"],
         "rows": rows1},
        {"caption": ("④ 資金流規則 L：過去 20 天是漲的、而且正在連續流入 ≥ 3 天就持有；連續流出 3 天或 20 日淨流量轉負就出場。"
                     "對照 P：過去 20 天是漲的、而且連 3 天收高就持有；連 3 天收低或 20 天轉跌就出場（只看價格，時間點一樣）。"
                     "流量日 t 的訊號在下一個收盤照做（GLD 隔一個交易日；幣是標為 t+1 的日線）。每次買賣扣 0.10%，空手沒有利息。"
                     "報酬÷波動越大越好；括號是每年報酬、最深虧損、持有時間。黃金三段欄只列報酬÷波動。"
                     "每月起算日的「一般人做法」= 再晚一天、成本 0.25%、一成進場沒跟到，200 次模擬。"),
         "columns": ["項目", "黃金全期", PERIODS[0][0], PERIODS[1][0], PERIODS[2][0], "比特幣 ETF", "以太幣 ETF"],
         "rows": rows2},
        {"caption": ("⑤ 流入停止後會不會回吐：連續流入結束（或 20 日流量從前 10% 轉弱）之後的報酬減平常，"
                     "和「同樣長度只看價格的連漲結束」比，分得出是資金退潮還是一般的漲多回檔。"
                     "⑥ 背離：過去 20 天漲幅在前三分之一，但 20 日淨流量 ≤ 0（沒有資金撐的上漲）vs 流量也在前三分之一（有資金撐的上漲）；"
                     "漲跌在後三分之一但流量在前三分之一（下跌但資金進場）vs 流量在後三分之一。"
                     f"括號是 90% 區間（區塊重抽 1,000 次，黃金每塊 {BLOCK['GLD']} 天、幣 {BLOCK['BTC']} 天）；段數 = 相隔 20 天以上算不同段。"),
         "columns": ["項目", "黃金全期", PERIODS[0][0], PERIODS[1][0], PERIODS[2][0], "黃金：只看價格的同類",
                     "比特幣 ETF", "以太幣 ETF"],
         "rows": rows3},
        {"caption": ("⑦ 幣的新資金：Farside 合計扣掉灰度舊信託（GBTC／ETHE，上市後長期被贖回）之後，重做④⑤⑥（只描述，不拿來挑規則）。"
                     "以及穩健性：晚一天才做、只算明顯流入、防偷看未來的檢查、樣本長度、最後搬到幣上的結果。"),
         "columns": ["檢查", "黃金 GLD", "比特幣 ETF", "以太幣 ETF"],
         "rows": rows4},
    ]

    zero_share = float((G["p"]["flow"] == 0).mean() * 100)
    notes = [
        "事前登記的設計（中文）：①過去 5／10／20 天漲跌分五格，看之後 5 天內開始連續流入（≥ 3 天，或 5 天裡 ≥ 4 天）的機率和基準比；"
        "②已連續流入 k 天時隔天還流入的機率、流量自相關，和打亂順序比；③依流量大小分十組看當天與隔天報酬（扣掉前一天漲跌）；"
        "④資金流規則 L 對照一直抱著、50 日均線、只看價格的同類規則 P 與流量平移的安慰劑，黃金三段都 L 贏 P 且安慰劑 ≤ 10% 才算通過；"
        "⑤連續流入 ≥ 5／10 天結束、20 日流量從前 10% 轉弱之後 5／20／60 天，和平常與只看價格的同類比；"
        "⑥沒資金撐的上漲 vs 有資金撐的上漲、下跌但資金進場 vs 下跌且資金撤；⑦幣分出新資金。"
        "只有④⑤⑥在黃金三段同方向且 90% 區間不含零的結果才搬到幣上當規則，其餘都是描述。門檻都在看結果之前定好，之後沒有改。",
        "事前登記原文（英文）：" + DESIGN_EN,
        ("作者在看結果前補定的細節：「天」都是流量日（GLD 交易日、美股交易日）；黃金流量 = 噸數每日變化，相對大小 = 除以前一天噸數；"
         "幣 = 美元流量 ÷ 過去一年平均單日流量；流量剛好為 0 的日子不算流入也不算流出，會打斷連續；幣的分格門檻 -1.1、0、+0.67、+1.33σ "
         "是黃金 20 天門檻除以黃金長期每天約 1% 的波動 ×√20；⑤的「轉弱」= 前一天在一年前 10%、當天 20 日流量變小，20 天內只算一次；"
         "⑥的三分位用過去 252 個流量日（最少 120 個）；⑤要搬到幣上，還要求流量版和價格版的差也同方向。"),
        f"黃金 GLD 有 {zero_share:.0f}% 的交易日噸數完全沒變（沒有申購贖回），所以黃金的「連續流入」比幣難湊；這是資料本身的樣子，沒有調整。",
        ("時間對齊（防偷看未來）：流量日 t 的數字要等美國收盤後才公布，最早只能在下一個收盤照做（GLD 下一個交易日收盤；"
         "幣是標為 t+1 的日線，UTC t+2 日 0 點收盤），報酬從那個收盤算起。另外整套再晚一天重算，並把切點之後的價格與流量打亂，"
         "確認切點之前的訊號完全不變（第四張表）。③的當天報酬只是描述。"),
        ("GLD 的噸數是 SPDR 官方歷史檔每天的「Tonnes of Gold」，當天的申購贖回可能隔一天才反映在噸數上，只會讓訊號更晚，不會偷看未來。"
         "GLD 只占全球黃金 ETF 的三到四成；價格用 GLD 收盤價（含年費拖累），不是倫敦金價。"),
        (f"比特幣 ETF 2024-01-11、以太幣 ETF 2024-07-23 才上市；這次可用的比特幣樣本約 {yrs_b:.1f} 年（事前登記時寫的是不到兩年，"
         "實際資料已接近三年，仍然很短），以太幣更短；一年平均流量要先累積 60 天、百分位 120 天，區間很寬，"
         "主訊號檢查那種 108 個起算日 × 24 個收盤時間的格子在這裡做不到，只能說方向像不像黃金。" if yrs_b else
         "比特幣 ETF 流量這次取不到。"),
        "鈾（Sprott 實體鈾信託 SPUT）：sprott.com 的鈾頁面對公開請求回應 HTTP 403，找不到免登入、免金鑰的公開資料檔，依規定沒有繞過，這次不做鈾。",
        ("文獻：van der Beck、Bouchaud、Villamaina 的研究認為，資金流入會推高價格、而投資人又追逐過去報酬，形成自我強化的迴圈，"
         "價格壓力退去時會反轉；業界也觀察到 ETF 流量有記憶（今天流入，明天常也流入）。本研究在單一資產的時間序列上檢查這條機制。"),
        ("Farside 彙整各發行商公布的數字，最新一天可能修正；今天（UTC）那一列還沒收完，不用。每次把抓到的分發行商表格存在本頁自己的存檔，"
         "Farside 擋住時用存檔，不繞過。黃金資料沿用 etf_flows 研究下載與存下的 SPDR 檔案。"),
        ("第一次執行後改了兩處（都是修正，不是調參數）：①黃金從 2005-01-01 起算，上市頭幾週的種子申購一天占持有量 10% 到 600%，"
         "會把自相關等數字扭曲（三段時期本來就從 2005 起）；②L 進場那天如果 20 日淨流量已經是負的就不進場（原本會進場隔天就出場）。"
         "改前黃金 L 報酬÷波動 0.18、P 0.15、安慰劑 46%，改後見第二張表，結論沒有改變。"),
        "空手時沒有算利息；成本只算買賣手續費，沒有稅和滑價。一直抱著不扣成本。",
        "過去的結果不保證未來一樣。這裡只描述資金流和價格過去的關係、以及目前的狀態，不是買賣建議。",
    ] + r["problems"]

    sources = [
        {"name": "van der Beck, Bouchaud & Villamaina, Ponzi Funds（Self-Inflated Fund Returns）, arXiv 2405.12768",
         "url": PAPER},
        {"name": "Ultumus：ETF flows have memory, but leveraged products remember differently", "url": MEMORY},
        {"name": "SPDR Gold Shares（GLD）官方頁面", "url": ef.GLD_PAGE},
        {"name": "SPDR Gold Shares 官方歷史檔（XLSX：每日收盤價、信託黃金噸數）", "url": ef.GLD_XLSX},
        {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入（分發行商）", "url": flows2.FARSIDE["BTC"][0]},
        {"name": "Farside Investors：美國現貨以太幣 ETF 每日淨流入（分發行商）", "url": flows2.FARSIDE["ETH"][0]},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線（BTC、ETH 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Sprott Physical Uranium Trust 頁面（試過，HTTP 403，沒有使用）",
         "url": "https://sprott.com/investment-strategies/physical-commodity-funds/uranium/"},
    ]
    return {
        "id": "etf_loop",
        "title": "ETF 資金流的自我強化迴圈",
        "question": ("價格小漲會不會引來連續好幾天的 ETF 資金流入、流入會不會繼續推高價格，流入停止後價格會不會回吐？"
                     "用資金流做的進出規則，有沒有贏過只看價格、時間點一樣的同類規則？從黃金定規則，原封不動搬到比特幣、以太幣 ETF。"),
        "asof": _d(r["g"].index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
