"""Historical study "nfci_stress": a stress test of rule N1 from nfci_filter. Its job is to try to BREAK N1, not to
find a better variant; no setting found here may be promoted as a new rule.

N1 (nfci_filter, passed its pre-registered test): hold BTC when main_signal_check's ma50 is on AND the latest Chicago
Fed NFCI value at least 6 days old at the month-end decision is below the average of its own last 52 weekly values;
decided at the last daily close of month M (00:00 UTC, te._daily_close), held during month M+1. The owner's bar for a
lasting rule also asks: does the result survive small changes, is it not one lucky setting, does it replicate on
independent data? Everything below is fixed before this study is run; nothing is changed after seeing results.

Shared: nfci_filter's loaders (Chicago Fed CSV -> FRED -> stored copy), business_cycle._Exam (main_signal_check's four
"beats holding" checks: BTC daily 108 monthly starts; 72 starts x 24 clocks = 1,728 textbook cells; the same cells
done by an ordinary person (10 hours later, 0.25% cost, 10% of entries missed); ETH/SOL/BNB/XRP/MSTR with their own
ma50 and the same US signal), ranking score = the lowest of the four shares, with the 95% shallower-loss eligibility;
"ranking >= ma50's" = business_cycle._score_ge. "Median deepest loss" = median over the 1,728 textbook cells;
"shallower" = strictly greater (less negative) than ma50's.

(A) Neighbourhood grid: averaging window 26 / 39 / 52 / 78 / 104 weekly values (including the current one) x extra
    delay 0 / 1 / 2 / 4 weeks on top of N1's 6-day rule (a weekly value dated Friday f is usable at month M's decision
    when f + 6 days + delay <= the last calendar day of M) = 20 settings; 52 / 0 is N1 itself. To stay inside the
    time budget every grid cell is scored with the ordinary-person check at business_cycle.PL_SIMS (50) simulations
    per clock, and ma50 is re-scored the same way for the comparison (the reduced version, as in nfci_filter's
    placebo); N1 and ma50 with the full 200 simulations are shown as reference rows. Report how many of the 20 have
    ranking >= ma50's AND a shallower median deepest loss, the worst setting (lowest ranking score; ties -> deeper
    median loss) and the median setting (sorted the same way, the lower middle = 10th of 20 from the worst).
    Bar: "robust" if at least 15 of 20 meet both; "fragile" if fewer than 10; in between = "in between".
(B) Decision frequency: the same 52-week rule decided weekly instead of monthly: at every daily close the latest
    weekly value with f + 6 days <= the close's date is used (= the first daily close on or after the Wednesday
    release + 1 day, the same 6-day rule as N1). Fed into the exam by giving _Exam a daily flag instead of a monthly
    one (business_cycle._on_days is swapped for a date-based lookup only while these rows are scored, then restored);
    on the 24-clock grid the flag used on UTC day j is the one usable at the date before day j (conservative). Full
    200 simulations. Report ranking, median deepest loss, and the number of position switches on the 00:00 UTC
    daily path from 2015-01-01 for ma50, N1 and the weekly version (and the number of NFCI flag switches).
    Report only, no bar.
(C) Sibling stress indices, the same rule (favorable when below the average of its own last 52 weekly / 252 daily /
    12 monthly values, monthly decision at the last daily close of M, held during M+1) with honest publication lags:
      OFR Financial Stress Index (daily, financialresearch.gov, own key-less CSV; a value dated x is usable once
      x + 2 business days + 1 day <= the last calendar day of M; 252-value average);
      St. Louis Fed STLFSI4 (weekly, dated Friday f, released the next Thursday; usable once f + 7 days <= the last
      calendar day of M; 52-value average);
      Kansas City Fed KCFSI (monthly, released during the next month; the value for month m is usable at the end of
      month m + 1; 12-value average).
    Downloads: FRED fredgraph CSV with a short timeout and 2 tries, then the banks' own key-less files if any;
    unreachable -> said so in notes and skipped; a good download is stored in common.STORE (own files) and used as a
    fallback. Higher = more stress for all three. Full 200 simulations.
    Bar: "replicates" if at least 2 of the AVAILABLE siblings have ranking >= ma50's AND a shallower median deepest
    loss (if fewer than 2 siblings are available the bar is not met).
(D) Same effect as O2? Month by month (decision months whose next month is held from 2011-07, both flags known):
    how often N1's NFCI flag and O2's OECD flag (business_cycle._series("oecd"), _flags with LAG 2) agree; and on the
    decision months where ma50 is on at the month-end close, how often both / only NFCI / only OECD / neither say
    cash, with the average next-month BTC return (month-end close to month-end close) of each group. Report only.
(E) Real-time vintages: try ALFRED (alfred.stlouisfed.org) key-less for NFCI once (short timeout, 2 tries). Only if
    as-first-published weekly values can be had key-less is N1 re-run on them; otherwise this is stated plainly and
    nothing is approximated.
(F) N1 minus ma50 calendar-year returns (00:00 UTC textbook path, 0.10% cost, from 2015; the current year is partial),
    the average over all years and the average without 2018 and 2022. Report only.
Verdict wording: (A) robust AND (C) replicates = "站得住"; exactly one = "部分站得住"; neither = "脆弱".
Today: states only.
"""
from __future__ import annotations

import io
import json
import math
import re
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import nfci_filter as nf

WINDOWS = (26, 39, 52, 78, 104)      # weekly values in the average
DELAYS = (0, 1, 2, 4)                # extra weeks on top of N1's 6-day rule
GRID_SIMS = bc.PL_SIMS               # ordinary-person simulations per clock in the grid
ROBUST, FRAGILE = 15, 10
DAY = bc.DAY
SAFE = nf.SAFE
FIRST = msc.DAILY_STARTS[0]
SKIP_YEARS = (2018, 2022)

OFR_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
OFR_PAGE = "https://www.financialresearch.gov/financial-stress-index/"
KC_PAGE = "https://www.kansascityfed.org/data-and-trends/kansas-city-financial-stress-index/"
ALFRED_NFCI = bc.ALFRED_CSV.format("NFCI")
SIB = {  # name -> (label, FRED id, window, lag text)
    "ofr": ("OFR 金融壓力指數（美國財政部金融研究辦公室）", None, 252, "資料日之後 2 個營業日公布，再多留 1 天"),
    "stl": ("聖路易聯準銀行金融壓力指數 STLFSI4", "STLFSI4", 52, "週五那週、下週四公布，月底只用至少 7 天前的那週"),
    "kc": ("堪薩斯市聯準銀行金融壓力指數 KCFSI", "KCFSI", 12, "某月的數字到下個月底才用"),
}
SIB_UNIT = {"ofr": "個交易日", "stl": "週", "kc": "個月"}

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok


# ---------- data ----------

def _store(name: str):
    return common.STORE / f"nfci_stress_{name}.json"


def _csv_pairs(text: str, col) -> dict:
    frame = pd.read_csv(io.StringIO(text))
    when = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(frame[col] if isinstance(col, str) else frame.iloc[:, col], errors="coerce")
    return {w.strftime("%Y-%m-%d"): float(v) for w, v in zip(when, vals) if pd.notna(w) and math.isfinite(v)}


def _ofr() -> dict:
    return _csv_pairs(bc._get(OFR_URL, timeout=(10, 60)).text, "OFR FSI")


def _fred(sid: str) -> dict:
    return _csv_pairs(bc._get(bc.FRED_CSV.format(sid), timeout=(6, 15), tries=2).text, 1)


def _kc_bank() -> dict:
    """The Kansas City Fed's own page: follow a CSV link mentioning the index, if the page answers."""
    page = bc._get(KC_PAGE, timeout=(6, 15), tries=2).text
    links = [u for u in re.findall(r'href="([^"]+\.csv[^"]*)"', page, flags=re.I) if "fsi" in u.lower()]
    if not links:
        raise RuntimeError("no CSV link on the page")
    url = links[0] if links[0].startswith("http") else "https://www.kansascityfed.org" + links[0]
    return _csv_pairs(bc._get(url, timeout=(6, 30), tries=2).text, 1)


def _sibling(name: str) -> tuple[pd.Series | None, str, list[str]]:
    label, fid, _, _ = SIB[name]
    tries = []
    if name == "ofr":
        tries.append(("OFR 自己的 CSV", _ofr))
    if fid:
        tries.append(("FRED", lambda: _fred(fid)))
    if name == "kc":
        tries.append(("堪薩斯市聯準銀行網站", _kc_bank))
    notes, got, where = [], None, ""
    for lab, fn in tries:
        try:
            x = fn()
            if len(x) < 100:
                raise RuntimeError(f"only {len(x)} rows")
            got, where = x, lab
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            notes.append(f"{label}：{lab}這次連不上或格式不對（{type(exc).__name__}）。")
    path = _store(name)
    if got is None:
        if not path.exists():
            notes.append(f"{label}：所有免金鑰的來源都連不上，也沒有存檔，這次跳過，沒有算進複製檢查。")
            return None, "", notes
        got, where = json.loads(path.read_text(encoding="utf-8")), "上次存下來的資料"
        notes.append(f"{label}改用上次存下來的資料（到 {max(got)}）。")
    else:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(got.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in got.items()}).sort_index()
    return s, where, notes


def _alfred() -> tuple[bool, str]:
    try:
        reply = bc._get(ALFRED_NFCI, timeout=(6, 15), tries=2)
        if not reply.text.startswith(("DATE", "observation_date")):
            return False, "ALFRED 有回應，但內容不是 CSV"
        return False, ("ALFRED 有回應，但一個免金鑰的檔案只給一個版本；要拼出「每週第一次公布的數字」需要幾百次下載，"
                       "這次沒有做，也沒有用別的方法估計")
    except Exception as exc:  # noqa: BLE001
        return False, f"ALFRED（alfred.stlouisfed.org）連得上主機，但一直沒有回應（{type(exc).__name__}，試了 2 次）"


def _load() -> dict:
    s, where, problems = nf._series()
    other, more = msc._load_other()
    oecd, owhere, onote = bc._series("oecd")
    sib, swhere = {}, {}
    for name in SIB:
        x, w, n = _sibling(name)
        problems += n
        if x is not None:
            sib[name], swhere[name] = x, w
    return {"close": bc.te._daily_close(), "nfci": s, "where": where, "other": other, "oecd": oecd, "owhere": owhere,
            "sib": sib, "swhere": swhere, "alfred": _alfred(),
            "problems": problems + more + ([onote] if onote else [])}


# ---------- the signals ----------

def _fav(s: pd.Series, window: int) -> pd.Series:
    """1 = below the average of its own last `window` values (higher = more stress for every index here)."""
    avg = s.rolling(window, min_periods=window).mean()
    return (s < avg).astype(float).where(avg.notna() & s.notna())


def _monthly(fav: pd.Series, usable: pd.DatetimeIndex, months: pd.PeriodIndex) -> pd.Series:
    """Decision month M -> the latest flag whose usable date <= the last calendar day of M."""
    order = np.argsort(usable.values, kind="stable")
    u, v = usable.values[order], fav.values[order]
    last_day = months.to_timestamp(how="end").normalize()
    pos = np.searchsorted(u, last_day.values, side="right") - 1
    return pd.Series(np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan), index=months)


def _usable(name: str, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if name == "ofr":
        return idx + pd.offsets.BDay(2) + DAY
    if name == "stl":
        return idx + pd.Timedelta(days=7)
    return (idx.to_period("M") + 1).to_timestamp(how="end").normalize()  # kc: month m usable at the end of m + 1


class _DailyFlag:
    """A flag looked up at every close: the latest value whose usable date <= the close's own date."""

    def __init__(self, fav: pd.Series, usable: pd.DatetimeIndex):
        ok = fav.notna().values
        self.u, self.v = usable.values[ok], fav.values[ok]

    def at(self, dates) -> np.ndarray:
        d = (pd.DatetimeIndex(dates).normalize() - DAY).values  # dates = the close moment (day after the candle)
        pos = np.searchsorted(self.u, d, side="right") - 1
        return np.where(pos >= 0, self.v[np.clip(pos, 0, None)], 0.0).astype(float)


_ORIG_ON_DAYS = bc._on_days


def _on_days_any(F, dates):
    if isinstance(F, _DailyFlag):
        return F.at(dates)
    return _ORIG_ON_DAYS(F, dates)


@contextmanager
def _daily_flags():
    """Let business_cycle._Exam take a daily flag; restored afterwards."""
    bc._on_days = _on_days_any
    try:
        yield
    finally:
        bc._on_days = _ORIG_ON_DAYS


def _switches(p: np.ndarray, idx: pd.DatetimeIndex) -> int:
    x = np.nan_to_num(np.asarray(p, dtype=float)[idx >= pd.Timestamp(FIRST)])
    return int((np.abs(np.diff(x)) > 0.5).sum())


def _meets(x: dict, base: dict) -> bool:
    return bool(bc._score_ge(x, base) and x["c_deep_med"] > base["c_deep_med"] + 1e-12)


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, raw = data["close"], data["nfci"]
    months = pd.period_range(raw.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    full = {"ma50": ex.summary("ma50", None)}
    F_n1 = nf._flags(raw, "avg52", months)
    full["n1"] = ex.summary("m2", F_n1)

    # (A) neighbourhood grid
    small_ma = ex.summary("ma50", None, GRID_SIMS)
    grid, same_n1 = [], None
    for w in WINDOWS:
        fav = _fav(raw, w)
        for dl in DELAYS:
            F = _monthly(fav, raw.index + SAFE + pd.Timedelta(weeks=dl), months)
            if w == 52 and dl == 0:
                same_n1 = bool(np.array_equal(F.fillna(-1).values, F_n1.fillna(-1).values))
            x = ex.summary("m2", F, GRID_SIMS)
            grid.append({"w": w, "dl": dl, "x": x, "meets": _meets(x, small_ma),
                         "score_ok": bool(bc._score_ge(x, small_ma)),
                         "loss_ok": bool(x["c_deep_med"] > small_ma["c_deep_med"] + 1e-12)})
    t_grid = time.time() - t0

    # (B) weekly decisions
    wk = _DailyFlag(_fav(raw, 52), raw.index + SAFE)
    with _daily_flags():
        full["weekly"] = ex.summary("m2", wk)
        pos_w = ex.btc_pos("m2", wk)
    pos = {"ma50": ex.btc_pos("ma50", None), "n1": ex.btc_pos("m2", F_n1), "weekly": pos_w}
    sw = {k: _switches(v, close.index) for k, v in pos.items()}
    sw["flag_weekly"] = _switches(wk.at(close.index + DAY), close.index)
    sw["flag_monthly"] = _switches(bc._on_days(F_n1, close.index + DAY), close.index)

    # (C) sibling indices
    sib = {}
    for name, s in data["sib"].items():
        window = SIB[name][2]
        fav = _fav(s, window)
        m2 = pd.period_range(s.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
        F = _monthly(fav, _usable(name, s.index), m2)
        x = ex.summary("m2", F)
        sib[name] = {"F": F, "x": x, "meets": _meets(x, full["ma50"]), "score_ok": bool(bc._score_ge(x, full["ma50"])),
                     "loss_ok": bool(x["c_deep_med"] > full["ma50"]["c_deep_med"] + 1e-12),
                     "pos": ex.btc_pos("m2", F)}

    # (D) N1 vs O2, month by month
    F_o2 = bc._flags(data["oecd"], bc.LAG["oecd"])
    me = bc._month_ends(close)
    nxt = (me.shift(-1) / me - 1).dropna()
    nxt = nxt[nxt.index >= bc.EVENT_FROM - 1]          # decision month; held month = next one, from 2011-07
    ma = pd.Series(ex.btc_ma, index=close.index)
    ma_me = pd.Series([ma.get(p.to_timestamp(how="end").normalize(), np.nan) for p in nxt.index], index=nxt.index)
    df = pd.DataFrame({"r": nxt, "n": F_n1.reindex(nxt.index), "o": F_o2.reindex(nxt.index), "ma": ma_me}).dropna()
    on = df[df["ma"] > 0.5]
    groups = {"both": on[(on["n"] == 0) & (on["o"] == 0)], "n_only": on[(on["n"] == 0) & (on["o"] == 1)],
              "o_only": on[(on["n"] == 1) & (on["o"] == 0)], "neither": on[(on["n"] == 1) & (on["o"] == 1)]}
    d_ = {"n": len(df), "agree": float((df["n"] == df["o"]).mean()), "first": df.index[0], "last": df.index[-1],
          "on_n": len(on), "agree_on": float((on["n"] == on["o"]).mean()) if len(on) else None,
          "groups": {k: (len(g), float(g["r"].mean()) if len(g) else None) for k, g in groups.items()},
          "cash_n": float((df["n"] == 0).mean()), "cash_o": float((df["o"] == 0).mean()),
          "corr": float(df["n"].corr(df["o"]))}

    # (F) calendar years
    years = {k: bc._years(pos[k], close, FIRST) for k in ("ma50", "n1")}
    diff = {y: years["n1"][y] - years["ma50"][y] for y in sorted(years["ma50"])
            if _ok(years["n1"].get(y)) and _ok(years["ma50"].get(y))}
    f_ = {"diff": diff, "years": years, "avg_all": float(np.mean(list(diff.values()))),
          "avg_wo": float(np.mean([v for y, v in diff.items() if y not in SKIP_YEARS]))}
    return {"ex": ex, "full": full, "small_ma": small_ma, "grid": grid, "same_n1": same_n1, "F_n1": F_n1,
            "wk": wk, "pos": pos, "sw": sw, "sib": sib, "D": d_, "F": f_, "F_o2": F_o2, "problem": problem,
            "t_grid": t_grid, "secs": time.time() - t0}


# ---------- the study ----------

def _score_txt(x: dict) -> str:
    return _share(x["low"]) + ("" if x["eligible"] else "（虧損條件未達 95%）")


def _yn(v: bool) -> str:
    return "是" if v else "否"


def study() -> dict:
    t0 = time.time()
    data = _load()
    r = _compute(data)
    close, ex, full, sm = data["close"], r["ex"], r["full"], r["small_ma"]
    raw = data["nfci"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base, n1, wkx = full["ma50"], full["n1"], full["weekly"]
    n_c = base["c_n"]

    # ----- (A) -----
    grid = r["grid"]
    count = sum(g["meets"] for g in grid)
    a_state = "robust" if count >= ROBUST else "fragile" if count < FRAGILE else "between"
    a_word = {"robust": "穩健", "fragile": "脆弱", "between": "介於中間"}[a_state]
    ranked = sorted(grid, key=lambda g: (g["x"]["low"] if g["x"]["eligible"] else -1.0, g["x"]["c_deep_med"]))
    worst, mid = ranked[0], ranked[len(ranked) // 2 - 1]

    def gname(g: dict) -> str:
        return f"{g['w']} 週平均、多等 {g['dl']} 週" + ("（就是 N1）" if g["w"] == 52 and g["dl"] == 0 else "")

    rows1 = []
    for g in grid:
        x = g["x"]
        rows1.append([f"{g['w']} 週", f"{g['dl']} 週", _score_txt(x), _pct(x["c_deep_med"]),
                      _yn(g["score_ok"]), _yn(g["loss_ok"]), "過" if g["meets"] else "不過"])
    rows1.append(["50 日均線（同樣簡化算法的基準）", "—", _score_txt(sm), _pct(sm["c_deep_med"]), "—", "—", "—"])
    rows1.append(["參考：50 日均線（完整算法）", "—", _score_txt(base), _pct(base["c_deep_med"]), "—", "—", "—"])
    rows1.append(["參考：N1（完整算法）", "—", _score_txt(n1), _pct(n1["c_deep_med"]), "—", "—", "—"])
    t1 = {"caption": ("（A）鄰近設定：把 N1 的兩個數字各往旁邊挪一點，看結果會不會垮。平均的長度 26／39／52／78／104 週，"
                      "再加上「多等幾週才用」0／1／2／4 週（在 N1 原本「至少 6 天前的那週」之外），共 20 種，52 週、多等 0 週就是 N1。"
                      f"為了在時間內算完，這 20 格「照一般人方式做」每個收盤時間只模擬 {GRID_SIMS} 次，50 日均線也用同樣方式重算一次再比；"
                      f"最深虧損中位數是 {n_c:,} 格照課本做的中位數，和模擬次數無關。過關＝排名分數不低於均線，而且最深虧損中位數比均線淺。"),
          "columns": ["平均長度", "多等幾週", "排名分數", "最深虧損中位數", "排名不低於均線", "虧損比均線淺", "兩項都過"],
          "rows": rows1}

    rows2 = [["20 種設定裡兩項都過的個數", f"{count} 種",
              f"事先寫好：至少 {ROBUST} 種＝穩健；少於 {FRAGILE} 種＝脆弱 → {a_word}"],
             ["最差的設定", gname(worst), f"排名分數 {_score_txt(worst['x'])}、最深虧損中位數 {_pct(worst['x']['c_deep_med'])}"],
             ["排在中間的設定（20 種由差到好的第 10 種）", gname(mid),
              f"排名分數 {_score_txt(mid['x'])}、最深虧損中位數 {_pct(mid['x']['c_deep_med'])}"],
             ["只看排名分數不低於均線的個數", f"{sum(g['score_ok'] for g in grid)} 種", "—"],
             ["只看最深虧損比均線淺的個數", f"{sum(g['loss_ok'] for g in grid)} 種", "—"]]
    for w in WINDOWS:
        cells = [g for g in grid if g["w"] == w]
        rows2.append([f"{w} 週平均：四種延遲裡兩項都過", f"{sum(g['meets'] for g in cells)}／4", "—"])
    t2 = {"caption": "（A）的整理。最差＝排名分數最低（一樣時看最深虧損較深的）；中間＝照同樣順序排、20 種裡的第 10 種。",
          "columns": ["項目", "結果", "說明"], "rows": rows2}

    # ----- (B) -----
    sw = r["sw"]
    rows3 = [["50 日均線（基準）", _score_txt(base), _pct(base["c_deep_med"]), f"{sw['ma50']} 次", "—"],
             ["N1：每月底決定一次", _score_txt(n1), _pct(n1["c_deep_med"]), f"{sw['n1']} 次", f"{sw['flag_monthly']} 次"],
             ["同一條規則改成每週決定", _score_txt(wkx), _pct(wkx["c_deep_med"]), f"{sw['weekly']} 次", f"{sw['flag_weekly']} 次"]]
    t3 = {"caption": ("（B）決定的頻率：同樣是「NFCI 低於自己 52 週平均」，改成每次 NFCI 公布後就更新（週三公布，"
                      "隔天那根日線收盤起才用，和 N1 一樣是「至少 6 天前的那週」），而不是只在月底看一次。完整算法（一般人做法每格 "
                      f"{bc.SIMS} 次）。換手次數是 {FIRST} 起 UTC 0 點日線上比特幣持有狀態改變的次數；最後一欄是金融情勢條件本身翻轉的次數。只報告，沒有過關標準。"),
          "columns": ["規則", "排名分數", "最深虧損中位數", "比特幣換手次數", "金融情勢條件翻轉次數"], "rows": rows3}

    # ----- (C) -----
    sib = r["sib"]
    rows4 = [["50 日均線（基準）", "—", "—", _score_txt(base), _pct(base["c_deep_med"]), "—"],
             ["N1：芝加哥 NFCI（被測的規則）", f"{_d(raw.index[0])} 到 {_d(raw.index[-1])}", "至少 6 天前的那週、52 週平均",
              _score_txt(n1), _pct(n1["c_deep_med"]), "—"]]
    for name in SIB:
        label, _, window, lagtxt = SIB[name]
        if name not in sib:
            rows4.append([label, "這次拿不到", lagtxt, "—", "—", "沒有算"])
            continue
        s, x = data["sib"][name], sib[name]["x"]
        rows4.append([label, f"{_d(s.index[0])} 到 {_d(s.index[-1])}（{data['swhere'][name]}）",
                      f"{lagtxt}；{window} {SIB_UNIT[name]}平均", _score_txt(x), _pct(x["c_deep_med"]),
                      "過" if sib[name]["meets"] else
                      f"不過（排名{'夠' if sib[name]['score_ok'] else '不夠'}、虧損{'較淺' if sib[name]['loss_ok'] else '沒有較淺'}）"])
    avail = len(sib)
    c_hits = sum(v["meets"] for v in sib.values())
    c_ok = avail >= 2 and c_hits >= 2
    t4 = {"caption": ("（C）換別的、獨立編製的金融壓力指數，用一模一樣的規則（低於自己大約一年的平均才算寬鬆、月底決定、照實際公布時間延後），"
                      "看效果會不會重現。三個都是數字越高壓力越大。完整算法。事先寫好：拿得到的指數裡至少 2 個「排名分數不低於均線、"
                      f"而且最深虧損中位數比均線淺」才算重現。這次拿得到 {avail} 個，過的 {c_hits} 個 → {'重現' if c_ok else '沒有重現'}。"),
          "columns": ["指數", "資料期間（來源）", "什麼時候能用；平均長度", "排名分數", "最深虧損中位數", "兩項都過"], "rows": rows4}

    # ----- (D) -----
    D = r["D"]
    gl = {"both": "兩個都說現金", "n_only": "只有 NFCI 說現金", "o_only": "只有 OECD 說現金", "neither": "兩個都說可以持有"}
    rows5 = [["所有月份：兩個條件說法一樣的比例", f"{_share(D['agree'])}（{D['n']} 個月）",
              f"NFCI 說現金 {_share(D['cash_n'])}、OECD 說現金 {_share(D['cash_o'])}；兩者相關係數 {D['corr']:+.2f}"],
             ["均線持有的月份：說法一樣的比例", f"{_share(D['agree_on'])}（{D['on_n']} 個月）", "—"]]
    for k, lab in gl.items():
        nn, mr = D["groups"][k]
        rows5.append([f"均線持有的月份裡，{lab}", f"{nn} 個月", f"下個月比特幣平均 {_p(mr)}"])
    t5 = {"caption": ("（D）N1 和 O2（OECD 美國領先指標高於 10 個月平均）是不是同一件事？每個月底各自的條件（金融情勢寬鬆／景氣有利），"
                      f"決定月 {D['first']} 到 {D['last']}（持有月從 2011-07 起，兩個條件都有值）。下半部只看月底收盤時 50 日均線是持有的月份，"
                      "依兩個條件分成四組，算下一個月比特幣（月底收盤到月底收盤）的平均報酬。只報告。"),
          "columns": ["項目", "個數", "說明"], "rows": rows5}

    # ----- (F) -----
    F_ = r["F"]
    rows6 = []
    ys = list(F_["diff"])
    for y in ys:
        rows6.append([f"{y}" + ("（到目前）" if y == ys[-1] else "") + ("（拿掉的年份）" if y in SKIP_YEARS else ""),
                      _pct(F_["years"]["ma50"][y]), _pct(F_["years"]["n1"][y]), _pp(F_["diff"][y])])
    rows6.append(["每年平均（全部年份）", "—", "—", _pp(F_["avg_all"])])
    rows6.append(["每年平均（拿掉 2018、2022）", "—", "—", _pp(F_["avg_wo"])])
    t6 = {"caption": (f"（F）拿掉 2018 和 2022 兩個大跌年還剩什麼？UTC 0 點日線照課本做（成本 {bc.COST * 100:.2f}%）每個日曆年的報酬，"
                      f"{FIRST[:4]} 年起，N1 減 50 日均線（百分點）；最後一年只到目前。只報告。"),
          "columns": ["年份", "50 日均線", "N1", "N1 減均線（百分點）"], "rows": rows6}

    # ----- overall -----
    a_ok = a_state == "robust"
    word = "站得住" if a_ok and c_ok else "部分站得住" if a_ok or c_ok else "脆弱"
    e_ok, e_txt = data["alfred"]
    rows7 = [["（A）鄰近設定", f"{count}／20 兩項都過（{a_word}）", "過" if a_ok else "不過"],
             ["（C）其他壓力指數", f"拿得到 {avail} 個，過 {c_hits} 個", "過" if c_ok else "不過"],
             ["（E）當時公布的版本", "可以重算" if e_ok else "沒辦法取得", "—"],
             ["總結", word, "—"]]
    t7 = {"caption": "整體判斷只看事先寫好的兩個門檻：（A）穩健而且（C）重現＝站得住；只有一個＝部分站得住；都沒有＝脆弱。（B）（D）（F）只報告。",
          "columns": ["檢查", "結果", "門檻"], "rows": rows7}

    # ----- today -----
    last = close.index[-1]
    cur = (last + DAY).to_period("M") - 1
    pm = r["pos"]

    def st(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    wk_now = r["wk"].at(pd.DatetimeIndex([last + DAY]))[0]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"N1 的金融情勢條件（{cur.strftime('%Y-%m')} 月底決定）｜每週版本（現在）",
         "value": f"{'寬鬆' if r['F_n1'].get(cur, np.nan) == 1 else '偏緊'}｜{'寬鬆' if wk_now == 1 else '偏緊'}"},
    ]
    for name in SIB:
        if name in sib:
            fv = sib[name]["F"].get(cur, np.nan)
            now.append({"label": f"{SIB[name][0]}（{cur.strftime('%Y-%m')} 月底決定）",
                        "value": "寬鬆" if fv == 1 else "偏緊" if fv == 0 else "—"})
    now += [{"label": "比特幣：50 日均線｜N1｜每週版本",
             "value": f"{st(pm['ma50'][-1])}｜{st(pm['n1'][-1])}｜{st(pm['weekly'][-1])}"},
            {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"}]

    # ----- verdict -----
    sib_txt = "；".join(f"{SIB[k][0].split('（')[0]} 排名 {_share(v['x']['low'])}、虧損中位數 {_pct(v['x']['c_deep_med'])}"
                       for k, v in sib.items()) or "一個都拿不到"
    missing = [SIB[k][0].split("（")[0] for k in SIB if k not in sib]
    verdict = (f"{word}。這份研究的目的是試著把 N1 弄垮，不是找更好的版本。（A）把平均長度和延遲挪動成 20 種設定，"
               f"{count} 種排名分數不低於均線、最深虧損也比較淺（事先寫好至少 {ROBUST} 種才算穩健，結果是{a_word}）；"
               f"最差的是 {gname(worst)}（排名 {_share(worst['x']['low'])}、虧損中位數 {_pct(worst['x']['c_deep_med'])}），"
               f"中間的是 {gname(mid)}（{_share(mid['x']['low'])}、{_pct(mid['x']['c_deep_med'])}）。"
               f"少虧這件事很穩（{sum(g['loss_ok'] for g in grid)}／20 種都比均線淺），排名分數贏均線只有 "
               f"{sum(g['score_ok'] for g in grid)}／20 種。"
               f"（C）換其他金融壓力指數：{sib_txt}"
               + (f"；{'、'.join(missing)}這次連不上、沒有算" if missing else "")
               + f"，{'重現' if c_ok else '沒有重現'}（要至少 2 個過"
               + ("；拿得到的不到 2 個，事先寫好這樣也算沒過" if avail < 2 else "") + "）。"
               f"（B）改成每週決定：排名 {_share(wkx['low'])}、虧損中位數 {_pct(wkx['c_deep_med'])}，換手 {sw['weekly']} 次（月底版 {sw['n1']} 次）。"
               f"（D）和 OECD 條件說法一樣的月份 {_share(D['agree'])}。"
               f"（F）N1 減均線每年平均 {_pp(F_['avg_all'])} 個百分點，拿掉 2018、2022 後 {_pp(F_['avg_wo'])} 個百分點"
               + ("（贏均線主要靠那兩個大跌年，其他年份平均是少賺）" if F_["avg_wo"] < 0 and
                  all(F_["diff"].get(y, 0) > 0 for y in SKIP_YEARS) else "") + "。"
               "（E）拿不到 NFCI 當時公布的版本，結果用的是修正後的數字。這裡出現的其他設定都不能當成新規則。")

    notes = [
        ("事先寫死的設計和門檻（照原文，跑之前寫好）：(A) 20 settings, window 26/39/52/78/104 weeks x extra delay 0/1/2/4 "
         "weeks; robust if >= 15 of 20 have ranking >= ma50's AND a shallower median deepest loss, fragile if < 10. "
         "(C) OFR FSI, STLFSI4, KCFSI with the same rule; replicates if >= 2 of the available siblings meet both. "
         "Verdict: both bars = 站得住, one = 部分站得住, neither = 脆弱. (B), (D), (F) report only; (E) only if first-release "
         "vintages can be had key-less. No variant found here may be promoted as a new rule."),
        ("這份研究在做什麼：N1 通過了它自己事先寫好的考試，但「一個設定剛好通過」可能只是運氣。這裡把它的數字往旁邊挪、"
         "換決定的頻率、換別人編的同類指數，看效果還在不在。挪動後比較好的設定也只是同一段歷史挑出來的，不能拿來取代 N1。"),
        (f"確認：表（A）裡 52 週、多等 0 週的每月條件和 nfci_filter 的 N1 {'完全一樣' if r['same_n1'] else '不一樣（要查）'}。"
         f"（A）的 20 格「照一般人方式做」每格只模擬 {GRID_SIMS} 次（基準的均線也一樣），所以那張表裡 N1 的排名分數可能和完整算法差一兩個百分點；"
         "完整算法的數字放在表末當參考。"),
        ("每週版本怎麼接進考試：公平考試原本只吃「每月一個狀態」，這裡在算每週版本那一列時，暫時讓它改成每根日線收盤去查「那時能用的最新一週」，"
         "算完就換回原本的做法；24 個收盤時間那部分，一律用前一天就能用的那週（比較保守）。"),
        ("其他壓力指數的時間：OFR 金融壓力指數每天更新、大約晚 2 個營業日公布，這裡再多留 1 天；STLFSI4 每週四公布上週五為止的那週，這裡用至少 7 天前的那週；"
         "KCFSI 每月公布一次，這裡某月的數字到下個月底才用。三個指數的成分和 NFCI 有部分重疊（都看信用利差、股市波動、資金市場），"
         "所以「獨立」只是編製的人和方法不同，不是完全不同的資料。"),
        (f"（E）當時公布的版本：{e_txt}。NFCI 每週都會把過去的數字重新估計，這裡的結果用的是今天的版本，可能比當時真的能做到的好看；"
         "這次沒有任何估計或替代。"),
        (f"NFCI 資料：{data['where']}（{_d(raw.index[0])} 到 {_d(raw.index[-1])}）；OECD 領先指標：{data['owhere']}。"
         + (f"{'、'.join(missing)}這次拿不到：FRED（STLFSI4、KCFSI 都在上面）和堪薩斯市聯準銀行網站對儀表板固定的 user agent "
            "連得上主機但不回應；聖路易聯準銀行自己發布 STLFSI4 的地方就是 FRED。照規定沒有換 user agent、也沒有用其他方法繞過。"
            if missing else "")),
        ("其他幣、MSTR 用的是同一個美國金融情勢訊號。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據；"
         "20 種鄰近設定彼此也高度重疊（相鄰設定大多數月份的狀態一樣），不是 20 個獨立的測試。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒（其中鄰近設定 {r['t_grid']:.0f} 秒）。",
        "過去的結果不保證未來會一樣。這裡只是把一條規則照事先寫好的方法壓力測試、說明現在的狀態，不是買賣建議。",
    ] + problems

    sources = [
        {"name": "芝加哥聯準銀行 NFCI 資料（CSV）", "url": nf.CHI_URL},
        {"name": "OFR 金融壓力指數資料（CSV）", "url": OFR_URL},
        {"name": "OFR 金融壓力指數說明頁", "url": OFR_PAGE},
        ({"name": "FRED：OECD 美國綜合領先指標", "url": bc.FRED_CSV.format(bc.FRED_ID["oecd"])} if data["owhere"] == "FRED"
         else {"name": "OECD 綜合領先指標（SDMX）", "url": bc.OECD_URL}),
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]
    for name in ("stl", "kc"):
        if name in sib:
            sources.append({"name": f"FRED：{SIB[name][1]}", "url": bc.FRED_CSV.format(SIB[name][1])})

    return {
        "id": "nfci_stress",
        "title": "NFCI 加均線規則的壓力測試",
        "question": ("N1（50 日均線加上「美國金融情勢指數 NFCI 低於自己一年平均」）通過了自己的考試。"
                     "把設定往旁邊挪一點、改成每週決定、換成別人編的金融壓力指數，效果還在嗎？還是只是剛好挑到一個幸運的設定？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t7, t1, t2, t3, t4, t5, t6],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
