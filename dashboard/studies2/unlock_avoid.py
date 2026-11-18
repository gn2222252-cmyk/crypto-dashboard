"""Historical study "unlock_avoid": if you skip holding a coin in the days before its big token unlocks, do you avoid
losses - after costs, across many events and coins?

Unlock schedules: DefiLlama's public emissions datasets (defillama-datasets.llama.fi/emissions/<protocol>, the same
files its own unlock pages read; no key). DefiLlama's API route for the same data (api.llama.fi/emissions) answers
HTTP 402 "paid plan" and is not used. Each file lists the vesting schedule of one token: dated cliff allocations
(recipient, category such as insiders / privateSale / airdrop) and the modelled unlocked amount per section per day.

An event is one day's cliff unlocks of one token (linear streams are not events). Its size is the amount unlocked that
day divided by DefiLlama's modelled unlocked supply on the day before, leaving out sections DefiLlama marks as
"noncirculating" (treasuries it tracks separately). This is not the same as the "circulating supply" market sites
report. Recipient type is the category with the largest share of that day's amount.

Tokens are matched to Binance USDT pairs of the survivorship store (daily candles of 435 pairs, dead ones included):
the token's symbol comes from DefiLlama's free price service (coins.llama.fi), and the match is kept only if
DefiLlama's historical price on one day is within 25% of Binance's close that day (catches same-ticker tokens and
redenominations).

1. Event study: coin return minus bitcoin's over -30..0 (close of day -31 to close of day -1), 0..+14 (day -1 to day
   +14) and +14..+30 days, against the same coins on random dates at least 45 days from any of their unlocks, and against
   the coin's own average 30-day excess return in the half year before. Events of one coin less than 30 days apart are
   one independent event. Standard errors are clustered by calendar month.
2. Rules per coin (only while it trades on Binance): hold, except cash from the close N+1 days before an unlock of 2%
   or more until the close of the unlock day (N = 30, 14, 7); the same on top of the coin's own 50-day-average rule.
   Placebo: skip the same number of windows of the same length at random dates.

All downloads are stored under cache/_store/unlock_avoid/ and refreshed once a month.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import survivorship as sv

DIR = common.STORE / "unlock_avoid"
SURV_DIR = common.STORE / "survivorship"
DAILY_DIR = SURV_DIR / "daily"
DATASETS = "https://defillama-datasets.llama.fi"
LIST_URL = DATASETS + "/emissionsProtocolsList"
FILE_URL = DATASETS + "/emissions/{}"
COINS = "https://coins.llama.fi"
KEYROCK = "https://keyrock.com/from-locked-to-liquidity-what-16000-token-unlocks-teach-us/"
BTC = "BTCUSDT"
COST = 0.001
COST_HIGH = 0.003
SKIPS = (30, 14, 7)
SIZES = (1.0, 2.0, 5.0)
RULE_MIN = 2.0
CLUSTER_DAYS = 30
PLACEBO_GAP = 45
PLACEBO_DATES = 20
PLACEBO_DRAWS = 100
PRICE_BAND = (0.75, 1.33)
SHIFT = (45, 120)
TIME_BUDGET = 15 * 60
SPLIT = pd.Timestamp("2024-01-01")
SEED = 7
TYPES = {"insiders": "團隊", "privateSale": "投資人", "publicSale": "社群／生態", "airdrop": "社群／生態",
         "farming": "社群／生態", "liquidity": "社群／生態", "staking": "社群／生態"}
UNTYPED = "未分類（多為基金會、儲備）"

_pct = sv._pct


def _p(v) -> str:
    """A fraction as a signed percentage."""
    return _pct(None if v is None or not sv._ok(v) else 100 * float(v))


def _read(name: str, default):
    return sv._read(DIR / name, default)


def _write(name: str, value) -> None:
    sv._write(DIR / name, value)


def _get(url: str, params: dict | None = None) -> requests.Response:
    reply = requests.get(url, params=params, headers=common.HEADERS, timeout=60)
    time.sleep(common.PAUSE)
    return reply


# ---------- unlock schedules ----------

def _day(ts) -> str:
    return pd.to_datetime(int(ts), unit="s").strftime("%Y-%m-%d")


def _compact(d: dict) -> dict | None:
    """Keep only what the study needs from one DefiLlama emissions file."""
    meta = d.get("metadata") or {}
    cats = d.get("categories") or {}
    nonc = {str(s).lower() for s in cats.get("noncirculating") or []}
    sections = ((d.get("documentedData") or {}).get("data")) or []
    total: dict[str, float] = {}
    for sec in sections:
        if str(sec.get("label", "")).lower() in nonc:
            continue
        for pt in sec.get("data") or []:
            v = pt.get("unlocked")
            if common._finite(v) and common._finite(pt.get("timestamp")):
                day = _day(pt["timestamp"])
                total[day] = total.get(day, 0.0) + float(v)
    days = sorted(total)
    vals = np.array([total[x] for x in days]) if days else np.array([])
    events: dict[str, dict] = {}
    linear = 0
    for ev in meta.get("unlockEvents") or []:
        if not common._finite(ev.get("timestamp")):
            continue
        day = _day(ev["timestamp"])
        linear += len(ev.get("linearAllocations") or [])
        for a in ev.get("cliffAllocations") or []:
            cat, rec, amt = str(a.get("category") or "Uncategorized"), str(a.get("recipient") or ""), a.get("amount")
            if cat == "noncirculating" or rec.lower() in nonc or not common._finite(amt) or amt <= 0:
                continue
            e = events.setdefault(day, {"amount": 0.0, "cats": {}, "recipients": []})
            e["amount"] += float(amt)
            e["cats"][cat] = e["cats"].get(cat, 0.0) + float(amt)
            if rec and rec not in e["recipients"]:
                e["recipients"].append(rec)
    out = []
    for day in sorted(events):
        i = int(np.searchsorted(days, day)) - 1  # last modelled day strictly before the unlock day
        before = float(vals[i]) if i >= 0 else 0.0
        e = events[day]
        out.append([day, e["amount"], before, e["cats"], e["recipients"][:4]])
    if not out:
        return None
    first = next((x for x in days if total[x] > 0), None)
    return {"name": d.get("name"), "token": meta.get("token"), "gecko": d.get("gecko_id"), "events": out,
            "linear": linear, "first": first}


def _schedules(month: str, deadline: float) -> tuple[dict, dict]:
    """{slug: compact schedule}, plus bookkeeping. Refreshed once a month; an unfinished refresh continues next run."""
    stored = _read("protocols.json", {})
    items = stored.get("items") or {}
    done = set(stored.get("done") or []) if stored.get("month") == month else set()
    info = {"listed": stored.get("listed"), "fetched": 0, "missing": 0, "paid": False}
    try:
        reply = _get(LIST_URL)
        if reply.status_code in (401, 402, 403):
            info["paid"] = True
            raise RuntimeError(f"HTTP {reply.status_code}")
        reply.raise_for_status()
        slugs = [s for s in reply.json() if isinstance(s, str)]
        info["listed"] = len(slugs)
    except Exception:  # noqa: BLE001 - keep what is stored
        slugs = []
    for slug in slugs:
        if slug in done:
            continue
        if time.time() > deadline:
            info["missing"] += 1
            continue
        try:
            reply = _get(FILE_URL.format(slug))
            if reply.status_code in (401, 402, 403):  # paid or refused: never worked around
                info["paid"] = True
                break
            if reply.status_code == 200:
                c = _compact(reply.json())
                if c:
                    items[slug] = c
                else:
                    items.pop(slug, None)
            done.add(slug)
            info["fetched"] += 1
        except Exception:  # noqa: BLE001 - one file failing must not stop the others
            info["missing"] += 1
        if info["fetched"] % 25 == 0:
            _write("protocols.json", {"month": month, "listed": info["listed"], "done": sorted(done), "items": items})
    if slugs:
        _write("protocols.json", {"month": month, "listed": info["listed"], "done": sorted(done), "items": items})
    if not items:
        raise RuntimeError("DefiLlama 的解鎖時間表取不到" + ("（需要付費）" if info["paid"] else ""))
    info["files"] = len(items)
    return items, info


# ---------- coins on Binance ----------

def _rows(pair: str) -> pd.Series | None:
    item = sv._read(DAILY_DIR / f"{pair}.json", {})
    rows = {str(k)[:10]: v[0] for k, v in (item.get("rows") or {}).items()
            if isinstance(v, list) and v and common._finite(v[0]) and v[0] > 0}
    if len(rows) < 2:
        return None
    s = pd.Series(rows, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    gaps = s.index.to_series().diff().dt.days
    late = gaps[gaps > sv.GAP_DAYS]
    return s[s.index < late.index[0]] if len(late) else s


def _key(c: dict) -> str | None:
    tok = c.get("token")
    if isinstance(tok, str) and ":" in tok:
        return tok
    return f"coingecko:{c['gecko']}" if c.get("gecko") else None


def _match(items: dict, month: str, deadline: float) -> tuple[dict, dict]:
    """{slug: Binance pair} for tokens whose symbol matches a stored pair and whose price agrees with Binance."""
    stored = _read("symbols.json", {})
    cache = stored.get("map") if stored.get("month") == month else {}
    cache = cache or {}
    available = {p.stem for p in DAILY_DIR.glob("*USDT.json")}
    todo = {slug: _key(c) for slug, c in items.items() if slug not in cache and _key(c)}
    keys = sorted(set(todo.values()))
    symbols: dict[str, str] = {}
    for i in range(0, len(keys), 40):
        if time.time() > deadline:
            break
        try:
            reply = _get(f"{COINS}/prices/current/" + ",".join(keys[i:i + 40]))
            reply.raise_for_status()
            for k, v in (reply.json().get("coins") or {}).items():
                if isinstance(v, dict) and v.get("symbol"):
                    symbols[k] = str(v["symbol"]).upper()
        except Exception:  # noqa: BLE001
            pass
    checks: dict[str, tuple[str, str, int, float]] = {}
    for slug, k in todo.items():
        sym = symbols.get(k)
        pair = f"{sym}USDT" if sym else None
        if not pair or pair not in available:
            cache[slug] = {"key": k, "symbol": sym, "pair": None, "why": "幣安沒有這個交易對" if sym else "查不到代號"}
            continue
        s = _rows(pair)
        if s is None:
            cache[slug] = {"key": k, "symbol": sym, "pair": None, "why": "日線太短"}
            continue
        inside = [e for e in items[slug]["events"] if s.index[0] <= pd.Timestamp(e[0]) <= s.index[-1]]
        day = pd.Timestamp(max(inside, key=lambda e: e[1])[0]) if inside else s.index[-1]
        day = s.index[min(int(s.index.searchsorted(day)), len(s) - 1)]
        checks[slug] = (k, pair, int(day.timestamp()) + 23 * 3600, float(s.loc[day]))
    slugs = sorted(checks)
    for i in range(0, len(slugs), 20):
        part = slugs[i:i + 20]
        got = {}
        if time.time() < deadline:
            try:
                q = {}
                for slug in part:
                    q.setdefault(checks[slug][0], []).append(checks[slug][2])
                reply = _get(f"{COINS}/batchHistorical", {"coins": json.dumps(q)})
                reply.raise_for_status()
                got = reply.json().get("coins") or {}
            except Exception:  # noqa: BLE001
                got = {}
        for slug in part:
            k, pair, ts, close = checks[slug]
            prices = [x for x in ((got.get(k) or {}).get("prices") or [])
                      if common._finite(x.get("price")) and abs(int(x.get("timestamp", 0)) - ts) < 2 * 86400]
            ratio = prices[0]["price"] / close if prices and close > 0 else None
            ok = ratio is not None and PRICE_BAND[0] <= ratio <= PRICE_BAND[1]
            if not got and time.time() > deadline:
                continue  # not checked this run
            cache[slug] = {"key": k, "symbol": pair[:-4], "pair": pair if ok else None,
                           "ratio": round(ratio, 3) if ratio else None,
                           "why": None if ok else ("價格對不上" if ratio else "查不到當天價格")}
    _write("symbols.json", {"month": month, "map": cache})
    return {slug: m["pair"] for slug, m in cache.items() if m.get("pair") and slug in items}, cache


def _events(items: dict, pairs: dict) -> list[dict]:
    """One row per token per unlock day (pairs mapped from several files keep one file: the one with most events)."""
    best: dict[str, str] = {}
    for slug, pair in pairs.items():
        if pair not in best or len(items[slug]["events"]) > len(items[best[pair]]["events"]):
            best[pair] = slug
    out = []
    for pair, slug in best.items():
        if pair[:-4] in sv.STOCKS or pair[:-4] in sv.PEGGED:
            continue  # tokenized shares and pegged tokens are not coins
        c = items[slug]
        for day, amount, before, cats, recs in c["events"]:
            if before <= 0:
                continue  # the first unlock (token launch): no supply to compare with
            top = max(cats, key=cats.get) if cats else "Uncategorized"
            ins = (cats.get("insiders", 0.0) + cats.get("privateSale", 0.0)) / amount if amount else 0.0
            out.append({"pair": pair, "coin": pair[:-4], "slug": slug, "day": pd.Timestamp(day),
                        "pct": 100.0 * amount / before, "type": TYPES.get(top, UNTYPED), "insider_share": ins,
                        "recipients": recs})
    return sorted(out, key=lambda e: (e["pair"], e["day"]))


# ---------- prices ----------

def _panel(pairs: set[str]) -> dict[str, pd.Series]:
    """{pair: daily closes on a calendar index, carried over missing days, only while the pair trades}."""
    out = {}
    for pair in pairs | {BTC}:
        s = _rows(pair)
        if s is None:
            continue
        days = pd.date_range(s.index[0], s.index[-1], freq="D")
        out[pair] = s.reindex(days).ffill()
    return out


def _ret(s: pd.Series, a, b) -> float | None:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a < s.index[0] or b > s.index[-1]:
        return None
    return float(s.loc[b] / s.loc[a] - 1)


def _measure(s: pd.Series, btc: pd.Series, d: pd.Timestamp) -> dict | None:
    """Coin minus bitcoin over the three windows around day d, plus the coin's usual 30-day excess before."""
    t = pd.Timedelta
    marks = [d - t(days=31), d - t(days=1), d + t(days=14), d + t(days=30)]
    rc = [_ret(s, marks[i], marks[i + 1]) for i in range(3)]
    rb = [_ret(btc, marks[i], marks[i + 1]) for i in range(3)]
    if rc[0] is None or rb[0] is None:
        return None
    x = [c - b if c is not None and b is not None else None for c, b in zip(rc, rb)]
    usual = None
    lo, hi = d - t(days=181), d - t(days=31)
    if lo - t(days=30) >= s.index[0]:
        a = s.loc[lo - t(days=30): hi]
        b = btc.reindex(a.index)
        ex = (a / a.shift(30) - 1) - (b / b.shift(30) - 1)
        ex = ex.loc[lo: hi].dropna()
        usual = float(ex.mean()) if len(ex) > 60 else None
    return {"pre": x[0], "mid": x[1], "post": x[2], "usual": usual}


def _cluster(events: list[dict], minimum: float) -> list[dict]:
    """Events of at least `minimum`% with events of the same coin less than CLUSTER_DAYS apart merged into the first
    (its size becomes the cluster's total)."""
    out = []
    for e in events:
        if e["pct"] < minimum:
            continue
        last = out[-1] if out else None
        if last and last["pair"] == e["pair"] and (e["day"] - last["_end"]).days < CLUSTER_DAYS:
            last["pct_cluster"] += e["pct"]
            last["_end"] = e["day"]
            continue
        out.append(dict(e, pct_cluster=e["pct"], _end=e["day"]))
    return out


def _se(vals: list[float], months: list[str]) -> float | None:
    """Standard error of the mean, clustered by month (events in the same month are not independent)."""
    n = len(vals)
    if n < 3:
        return None
    m = float(np.mean(vals))
    groups: dict[str, float] = {}
    for v, g in zip(vals, months):
        groups[g] = groups.get(g, 0.0) + (v - m)
    if len(groups) < 3:
        return None
    k = len(groups)
    return math.sqrt(sum(x * x for x in groups.values()) * k / (k - 1)) / n


# ---------- rules ----------

def _rule_effects(panel: dict, events: list[dict], rng: np.random.Generator) -> dict:
    """Per coin and per event, the rule minus its base (hold, or the 50-day rule), for every skip length and cost."""
    btc_free = {p: s for p, s in panel.items() if p != BTC}
    by_coin: dict[str, list[pd.Timestamp]] = {}
    for e in events:
        if e["pct"] >= RULE_MIN:
            by_coin.setdefault(e["pair"], []).append(e["day"])
    res = {(base, n, cost): {"coin": [], "placebo": [], "shifted": [], "event": [], "event_day": []}
           for base in ("hold", "ma50") for n in SKIPS for cost in (COST, COST_HIGH)}
    coins = 0
    for pair, days in by_coin.items():
        s = btc_free.get(pair)
        if s is None or len(s) < 120:
            continue
        idx = s.index
        r = (s / s.shift(1) - 1).fillna(0.0).to_numpy()
        ma = s.rolling(50, min_periods=50).mean()
        sig = (s > ma).astype(float).shift(1).fillna(0.0).to_numpy()  # decided at the previous close
        start, end = 51, len(s) - 1
        pos_day = [int(idx.searchsorted(d)) for d in days if idx[start] <= d <= idx[end]]
        if not pos_day:
            continue
        coins += 1
        for n in SKIPS:
            mask = np.zeros(len(s), dtype=bool)
            for k in pos_day:
                mask[max(k - n, 0): k + 1] = True
            draws, shifted = [], []
            for _ in range(PLACEBO_DRAWS):
                m = np.zeros(len(s), dtype=bool)
                for k in rng.integers(start + n, end + 1, size=len(pos_day)):
                    m[k - n: k + 1] = True
                draws.append(m)
                # the same windows moved 45-120 days earlier or later: same coin, similar age, no unlock
                m = np.zeros(len(s), dtype=bool)
                for k in pos_day:
                    off = int(rng.integers(SHIFT[0], SHIFT[1] + 1)) * (1 if rng.random() < 0.5 else -1)
                    j = k + off if start + n <= k + off <= end else k - off
                    j = min(max(j, start + n), end)
                    m[j - n: j + 1] = True
                shifted.append(m)
            for base in ("hold", "ma50"):
                bpos = np.ones(len(s)) if base == "hold" else sig
                for cost in (COST, COST_HIGH):
                    def growth(pos):
                        p = pos[start:]
                        turn = np.abs(np.diff(p, prepend=p[0]))
                        net = p * r[start:] - cost * turn
                        return np.log1p(np.maximum(net, -0.999)), net
                    gb, _ = growth(bpos)
                    gr, _ = growth(np.where(mask, 0.0, bpos))
                    eff = float(np.exp(gr.sum() - gb.sum()) - 1)
                    plac = [float(np.exp(growth(np.where(m, 0.0, bpos))[0].sum() - gb.sum()) - 1) for m in draws]
                    near = [float(np.exp(growth(np.where(m, 0.0, bpos))[0].sum() - gb.sum()) - 1) for m in shifted]
                    out = res[(base, n, cost)]
                    out["coin"].append(eff)
                    out["placebo"].append(plac)
                    out["shifted"].append(near)
                    diff = gr - gb
                    for k in pos_day:
                        a, b = max(k - n, start) - start, min(k + 2, len(s)) - start
                        out["event"].append(float(np.expm1(diff[a:b].sum())))
                        out["event_day"].append(idx[k])
    res["coins"] = coins
    return res


def _lookahead_check(panel: dict, events: list[dict]) -> bool:
    """The positions on each day must not change when every price after that day is removed."""
    rng = np.random.default_rng(SEED + 1)
    pairs = [p for p in {e["pair"] for e in events if e["pct"] >= RULE_MIN} if p in panel and len(panel[p]) > 200]
    for pair in rng.choice(pairs, size=min(5, len(pairs)), replace=False) if pairs else []:
        s = panel[pair]
        days = [e["day"] for e in events if e["pair"] == pair and e["pct"] >= RULE_MIN]

        def positions(close: pd.Series) -> pd.Series:
            ma = close.rolling(50, min_periods=50).mean()
            sig = (close > ma).astype(float).shift(1).fillna(0.0)
            skip = pd.Series(False, index=close.index)
            for d in days:  # only date and size from the schedule, no prices
                skip.loc[d - pd.Timedelta(days=30): d] = True
            return sig.where(~skip, 0.0)

        full = positions(s)
        for cut in rng.integers(60, len(s) - 1, size=5):
            part = positions(s.iloc[:cut + 1])
            if not full.iloc[:cut + 1].equals(part):
                return False
    return True


# ---------- the study ----------

def _row_stats(rows: list[dict], placebo_mean: float | None) -> dict:
    pre = [r["pre"] for r in rows]
    months = [r["day"].strftime("%Y-%m") for r in rows]
    adj = [r["pre"] - r["usual"] for r in rows if r["usual"] is not None]
    mid = [r["mid"] for r in rows if r["mid"] is not None]
    post = [r["post"] for r in rows if r["post"] is not None]
    se = _se(pre, months)
    m = float(np.mean(pre)) if pre else None
    return {"n": len(rows), "coins": len({r["pair"] for r in rows}), "pre": m,
            "pre_med": float(np.median(pre)) if pre else None,
            "neg": sum(v < 0 for v in pre) / len(pre) if pre else None,
            "adj": float(np.mean(adj)) if adj else None, "n_adj": len(adj),
            "mid": float(np.mean(mid)) if mid else None, "post": float(np.mean(post)) if post else None,
            "vs": (m - placebo_mean) if m is not None and placebo_mean is not None else None, "se": se}


def _fmt_row(name: str, st: dict) -> list:
    vs = "—" if st["vs"] is None else _p(st["vs"]) + (f"（±{100 * 2 * st['se']:.1f}）" if st["se"] else "")
    return [name, f"{st['n']}（{st['coins']} 個幣）",
            "—" if st["pre"] is None else f"{_p(st['pre'])}（{_p(st['pre_med'])}）",
            "—" if st["neg"] is None else f"{100 * st['neg']:.0f}%",
            _p(st["adj"]), _p(st["mid"]), _p(st["post"]), vs]


def study() -> dict:
    t0 = time.time()
    deadline = t0 + TIME_BUDGET
    month = common.today_utc()[:7]
    today = pd.Timestamp(common.today_utc())
    items, info = _schedules(month, deadline)
    pairs, mapping = _match(items, month, deadline)
    events = _events(items, pairs)
    status = sv._read(SURV_DIR / "pairs.json", {}).get("status") or {}
    panel = _panel({e["pair"] for e in events})
    btc = panel[BTC]
    rng = np.random.default_rng(SEED)

    # ---- 1. event study ----
    measured = []
    for e in events:
        s = panel.get(e["pair"])
        if s is None or e["day"] > today - pd.Timedelta(days=1):
            continue
        m = _measure(s, btc, e["day"])
        if m:
            measured.append(dict(e, **m))
    all_unlock = {}
    for e in events:
        all_unlock.setdefault(e["pair"], []).append(e["day"])
    big = [e for e in events if e["pct"] >= SIZES[0]]
    unlock_days = {}
    for e in big:
        unlock_days.setdefault(e["pair"], []).append(e["day"])
    placebo, placebo_usual = [], []
    for pair in unlock_days:
        s = panel.get(pair)
        if s is None or len(s) < 120:
            continue
        cand = s.index[61:-31]
        if not len(cand):
            continue
        far = np.ones(len(cand), dtype=bool)
        for d in unlock_days[pair]:
            far &= np.abs((cand - d).days) > PLACEBO_GAP
        cand = cand[far]
        for d in rng.choice(cand, size=min(PLACEBO_DATES, len(cand)), replace=False) if len(cand) else []:
            m = _measure(s, btc, pd.Timestamp(d))
            if m:
                placebo.append(m["pre"])
                if m["usual"] is not None:
                    placebo_usual.append(m["pre"] - m["usual"])
    pmean = float(np.mean(placebo)) if placebo else None
    by_key = {(e["pair"], e["day"]): e for e in measured}

    def clustered(minimum: float, keep=lambda e: True) -> list[dict]:
        rows = []
        for c in _cluster(events, minimum):
            m = by_key.get((c["pair"], c["day"]))
            if m and keep(c):
                rows.append(dict(m, pct_cluster=c["pct_cluster"]))
        return rows

    t1_rows, stats = [], {}
    for size in SIZES:
        stats[size] = _row_stats(clustered(size), pmean)
        t1_rows.append(_fmt_row(f"解鎖量 ≥ 流通量 {size:g}%", stats[size]))
    for lo, hi in ((1, 2), (2, 5), (5, 10), (10, None)):
        rows = clustered(1.0, lambda e, lo=lo, hi=hi: e["pct"] >= lo and (hi is None or e["pct"] < hi))
        name = f"只看 {lo}%–{hi}%" if hi else f"只看 {lo}% 以上"
        t1_rows.append(_fmt_row(name, _row_stats(rows, pmean)))
    for typ in ("團隊", "投資人", "社群／生態", UNTYPED):
        rows = clustered(RULE_MIN, lambda e, typ=typ: e["type"] == typ)
        if rows:
            t1_rows.append(_fmt_row(f"≥2%，對象以{typ}為主", _row_stats(rows, pmean)))
    cliff_ins = clustered(RULE_MIN, lambda e: e["insider_share"] >= 0.5)
    t1_rows.append(_fmt_row("≥2%，團隊＋投資人過半", _row_stats(cliff_ins, pmean)))
    p_neg = sum(v < 0 for v in placebo) / len(placebo) if placebo else None
    t1_rows.append(["對照：同一批幣隨便挑的日子", f"{len(placebo)}（{len(unlock_days)} 個幣）",
                    f"{_p(pmean)}（{_p(float(np.median(placebo)) if placebo else None)}）",
                    "—" if p_neg is None else f"{100 * p_neg:.0f}%",
                    _p(float(np.mean(placebo_usual)) if placebo_usual else None), "—", "—", "—"])

    # ---- 2. rules ----
    effects = _rule_effects(panel, events, np.random.default_rng(SEED + 2))
    rule_rows, summary = [], {}
    for base, base_name in (("hold", "一直抱著"), ("ma50", "50 日均線規則")):
        for n in SKIPS:
            a, b = effects[(base, n, COST)], effects[(base, n, COST_HIGH)]
            if not a["coin"]:
                continue
            med = float(np.median(a["coin"]))
            plac = np.array(a["placebo"])  # coins x draws
            plac_med = float(np.median(np.median(plac, axis=1)))
            beat = float(np.mean(np.median(plac, axis=0) < med))
            near = np.array(a["shifted"])
            near_med = float(np.median(np.median(near, axis=1)))
            beat_near = float(np.mean(np.median(near, axis=0) < med))
            ev = a["event"]
            helped = sum(v > 0 for v in ev) / len(ev) if ev else None
            summary[(base, n)] = {"med": med, "med_high": float(np.median(b["coin"])), "plac": plac_med,
                                  "beat": beat, "near": near_med, "beat_near": beat_near,
                                  "share": sum(v > 0 for v in a["coin"]) / len(a["coin"]),
                                  "helped": helped, "ev_med": float(np.median(ev)) if ev else None}
            st = summary[(base, n)]
            rule_rows.append([f"{base_name}，解鎖前 {n} 天空手", f"{len(a['coin'])} 個幣／{len(ev)} 次",
                              f"{100 * st['share']:.0f}%", f"{_pct(100 * med)}／{_pct(100 * st['med_high'])}",
                              f"{_pct(100 * plac_med)}（實際勝過 {100 * beat:.0f}% 的抽樣）",
                              f"{_pct(100 * near_med)}（實際勝過 {100 * beat_near:.0f}% 的抽樣）",
                              "—" if helped is None else f"{100 * helped:.0f}%（每次中位數 {_pct(100 * st['ev_med'])}）"])

    # ---- 3. periods ----
    period_rows = []
    for name, keep in (("2024 年以前", lambda d: d < SPLIT), ("2024 年起", lambda d: d >= SPLIT)):
        rows = clustered(RULE_MIN, lambda e, keep=keep: keep(e["day"]))
        st = _row_stats(rows, pmean)
        cells = []
        for base in ("hold", "ma50"):
            a = effects[(base, 30, COST)]
            ev = [v for v, d in zip(a["event"], a["event_day"]) if keep(d)]
            cells.append("—" if not ev else
                         f"{100 * sum(v > 0 for v in ev) / len(ev):.0f}% 有幫助（中位數 {_pct(100 * float(np.median(ev)))}）")
        vs = "—" if st["vs"] is None else _p(st["vs"]) + (f"（±{100 * 2 * st['se']:.1f}）" if st["se"] else "")
        period_rows.append([name, f"{st['n']}（{st['coins']} 個幣）", _p(st["pre"]), _p(st["adj"]), vs,
                            cells[0], cells[1]])

    # ---- largest events ----
    biggest = sorted(clustered(RULE_MIN), key=lambda e: -e["pct"])[:12]
    big_rows = [[e["coin"], sv._d(e["day"]), f"{e['pct']:.1f}%", e["type"], "、".join(e["recipients"][:2]) or "—",
                 _p(e["pre"]), _p(e["mid"]), _p(e["post"])] for e in biggest]

    # ---- today ----
    vol = {}
    for pair in {e["pair"] for e in events}:
        item = sv._read(DAILY_DIR / f"{pair}.json", {}).get("rows") or {}
        last = sorted(item)[-30:]
        vol[pair] = sum(float(item[k][1]) for k in last if isinstance(item[k], list) and len(item[k]) == 2)
    upcoming = sorted((e for e in events if today < e["day"] <= today + pd.Timedelta(days=30) and e["pct"] >= RULE_MIN
                       and status.get(e["pair"]) == "TRADING"), key=lambda e: e["day"])
    upcoming = sorted(upcoming, key=lambda e: -vol.get(e["pair"], 0.0))[:10]
    upcoming = sorted(upcoming, key=lambda e: e["day"])
    look_ok = _lookahead_check(panel, events)

    s1, s2 = stats[RULE_MIN], stats[SIZES[0]]
    years = sorted({e["day"].year for e in measured if e["pct"] >= RULE_MIN})
    h30, m30 = summary.get(("hold", 30)), summary.get(("ma50", 30))
    sig = s1["vs"] is not None and s1["se"] and abs(s1["vs"]) > 2 * s1["se"]

    bn = min(h30["beat_near"] if h30 else 0.0, m30["beat_near"] if m30 else 0.0)
    judge = ("解鎖日本身的效果在兩種做法上都分得出來" if bn >= 0.95 else
             "解鎖日本身多出的部分不確定是不是運氣" if bn >= 0.8 else
             "多出的錢主要來自這些幣那段時間本來就常跌，不是解鎖本身")

    def word(x: float | None, good="多", bad="差") -> str:
        return "—" if x is None else (good if x > 0 else bad) + f" {abs(100 * x):.1f} 個百分點"

    verdict = (
        f"用 DefiLlama 公開的解鎖時間表，在幣安有日線的 {s1['coins']} 個幣上找到 {s1['n']} 次佔流通量 2% 以上的獨立解鎖"
        f"（{years[0] if years else '—'}–{years[-1] if years else '—'} 年）。"
        f"解鎖前 30 天平均比比特幣{word(s1['pre'], '強', '差')}，但同一批幣隨便挑日子也平均{word(pmean, '強', '差')}，"
        f"兩者相差 {_p(s1['vs'])}（{'統計上分得出' if sig else '分不出是不是運氣'}）；扣掉該幣前半年平常的落後後是 {_p(s1['adj'])}。"
        f"解鎖前 30 天改拿現金，一直抱著的幣最後的錢中位數{word(h30['med'] if h30 else None)}、"
        f"50 日均線規則上{word(m30['med'] if m30 else None)}，"
        f"但把同樣的空手期挪前或挪後一兩個月也有 {_pct(100 * h30['near']) if h30 else '—'}、"
        f"{_pct(100 * m30['near']) if m30 else '—'}，{judge}；時間表是事後整理的，逐筆公布時點無法確認。")

    now = [{"label": "資料涵蓋", "value": f"DefiLlama 時間表 {info.get('files')} 份（清單 {info.get('listed')} 份），"
                                         f"對上幣安交易對且價格核對過的 {len(set(pairs.values()))} 個幣；"
                                         f"所有解鎖日 {len(events)} 次，其中 ≥1% 的 {len(big)} 次"}]
    if upcoming:
        now.append({"label": "未來 30 天內 ≥2% 的解鎖（幣安成交量較大的幣，只是描述，不是建議）",
                    "value": "；".join(f"{e['coin']} {sv._d(e['day'])} 約 {e['pct']:.1f}%（{e['type']}）" for e in upcoming)})
    else:
        now.append({"label": "未來 30 天內 ≥2% 的解鎖", "value": "時間表裡沒有（在幣安交易中的幣）"})
    if info.get("missing"):
        now.append({"label": "這次沒抓完", "value": f"{info['missing']} 份時間表下次再補"})

    tables = [
        {"caption": "解鎖前後，幣的漲跌減比特幣（同一個幣 30 天內的解鎖算一次；「±」是兩倍標準誤，依月份分群）",
         "columns": ["組別", "獨立事件數", "解鎖前 30 天：平均（中位數）", "解鎖前偏弱的比例", "扣掉該幣前半年平常的落後",
                     "解鎖前一天到後 14 天", "後 14 到 30 天", "和隨機日子差"],
         "rows": t1_rows[:16]},
        {"caption": "解鎖（≥2%）前 N 天改拿現金，和原本做法比：每個幣最後的錢差多少",
         "columns": ["做法", "幣數／避開次數", "比原本多的幣占比", "中位數（成本 0.10%／0.30%）",
                     "對照一：隨便挑日子空手同樣天數", "對照二：同樣的空手期往前或往後挪 45–120 天", "每次避開有幫助的比例"],
         "rows": rule_rows},
        {"caption": "分期間：2% 以上的解鎖，解鎖前 30 天減比特幣，以及避開 30 天的效果（成本 0.10%）",
         "columns": ["期間", "獨立事件數", "解鎖前 30 天平均", "扣掉平常落後", "和隨機日子差",
                     "一直抱著＋避開 30 天", "50 日均線＋避開 30 天"],
         "rows": period_rows},
        {"caption": "規模最大的 12 次解鎖（佔解鎖前一天的已解鎖量）",
         "columns": ["幣", "解鎖日", "規模", "對象", "時間表上的名目", "前 30 天減比特幣", "前一天到後 14 天", "後 14 到 30 天"],
         "rows": big_rows},
    ]
    notes = [
        "代幣解鎖：原本鎖住、不能賣的幣開始可以賣。這裡只算「一次放出一大筆」（cliff），每天慢慢放出的（linear）不算事件。",
        f"資料：DefiLlama 公開的解鎖時間表檔案（免費、不需金鑰；它的付費 API 同樣的資料要付費，沒有使用）。"
        f"清單上 {info.get('listed')} 份裡有 {info.get('files')} 份有一次性解鎖；代號由 DefiLlama 價格服務查出，"
        f"再和幣安當天收盤價核對（差 25% 以內才算同一個幣），最後 {len(set(pairs.values()))} 個幣可以用。",
        "規模的分母是 DefiLlama 依時間表算出、解鎖前一天已解鎖的數量（扣掉它另外追蹤的國庫部分），"
        "和市場網站報的「流通量」不同，所以同一次解鎖在新聞上的百分比可能差很多。第一次解鎖（上線當天）沒有分母，不算。",
        "對象依當天解鎖量最大的類別：團隊（insiders）、投資人（privateSale）、社群／生態（空投、公開銷售、挖礦）；"
        "DefiLlama 沒分類的多半是基金會或儲備。",
        "不偷看未來：規則每天只用前一天收盤以前的價格，和時間表上的日期與數量（不用任何之後的價格）；"
        f"程式把之後的價格拿掉重算，持有部位{'完全不變' if look_ok else '有變動，結果不可信'}。"
        "但 DefiLlama 的時間表是現在整理的版本，無法逐筆確認每次解鎖的日期和數量在當時就已公開；"
        "一般來說團隊和投資人的鎖倉時程在代幣上線時就寫在公開文件裡，延後或改過的時程則可能用了事後的版本。",
        "只算在幣安有 USDT 交易對的期間；包含後來下市的幣（下市後就不算）。幣安只收曾經成交量排前 60 名的幣，冷門小幣不在裡面。",
        "隨機對照：同一批幣、離任何 1% 以上解鎖 45 天以上的日子，每個幣抽 20 天；"
        f"規則的隨機版本是在同一個幣上隨機挑同樣次數、同樣長度的空手期，抽 {PLACEBO_DRAWS} 次。",
        "成本：每次買或賣收成交金額的 0.10%（另列 0.30%）；空手時現金沒有利息。",
        "第一張表是和比特幣比（扣掉整個市場的漲跌），第二張表是和現金比（不扣市場）；很多幣的解鎖集中在同一段山寨幣走弱的期間，"
        "而規則的隨機抽樣是每個幣各自抽、沒有固定在同一段市場，所以「實際勝過幾成抽樣」會高估把握，"
        "應該和第一張表一起看：扣掉比特幣和該幣平常的落後之後，解鎖前幾乎沒有多出來的弱勢。",
        "平均和中位數差很多：少數幾次解鎖前大漲（例如早期的 AVAX、JTO）把平均拉高，多數解鎖前是跌的。",
        "Keyrock（造市商）研究 40 個代幣、拆成 1.6 萬多次解鎖：約九成解鎖伴隨價格壓力、下跌常在 30 天前就開始、"
        "團隊解鎖最差、生態解鎖最輕；它用以太幣當對照，沒有公開資料來源和信賴區間，而且「1.6 萬次」其實來自 40 個幣，彼此高度相關。",
        "過去的結果不代表未來；這是描述歷史，不是買賣建議。",
    ]
    sources = [
        {"name": "Keyrock：From Locked to Liquidity（16,000 次代幣解鎖）", "url": KEYROCK},
        {"name": "DefiLlama 解鎖時間表（公開檔案清單）", "url": LIST_URL},
        {"name": "DefiLlama 解鎖時間表（單一代幣範例：Arbitrum）", "url": FILE_URL.format("arbitrum")},
        {"name": "DefiLlama 解鎖頁面", "url": "https://defillama.com/unlocks"},
        {"name": "DefiLlama 價格服務（代號與歷史價格核對）", "url": "https://coins.llama.fi/prices/current/coingecko:bitcoin"},
        {"name": "幣安公開日線（存活者偏差研究的資料庫）", "url": "https://data.binance.vision/"},
    ]
    _write("summary.json", {"asof": common.today_utc(), "seconds": round(time.time() - t0, 1),
                            "events": len(events), "coins": len(set(pairs.values())), "lookahead_ok": look_ok})
    result = {"id": "unlock_avoid", "title": "避開代幣解鎖前的日子",
              "question": "如果在一個幣大量解鎖之前的幾天不持有它，扣掉成本後，在很多次解鎖、很多個幣上，能不能少虧？",
              "asof": common.today_utc(), "verdict": verdict, "now": now, "tables": tables, "notes": notes,
              "sources": sources}
    return _clean(result)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2))
