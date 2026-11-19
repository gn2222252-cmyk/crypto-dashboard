"""Historical study "net_liquidity": does "Fed net liquidity" (Fed balance sheet minus the Treasury General Account
minus overnight reverse repo) lead bitcoin, and does a net-liquidity filter improve the 50-day moving average rule
(ma50, main_signal_check's main signal)?

The claims. Widely repeated: net liquidity = WALCL - TGA - RRP drives bitcoin and moves first; a "13-week lead" is
the common popular version (Keyrock applies a 13-week lead to its global liquidity index). Newer vendor claim
(Keyrock, "The Liquidity Source That Leads Bitcoin", Amir Hajian, 2026): since 2021 Treasury bill issuance has the
strongest leading relationship with bitcoin (r = +0.80, lead of roughly 8 months), stronger than the Fed's own
balance sheet, which it says shows no statistically significant relationship.

How this differs from liquidity_lead (already in the project): that study tested global M2 (and net liquidity as a
second variant) with the popular fixed 70-day shift, as a daily on/off rule judged by return/volatility per period.
This study (a) maps the whole lead/lag profile of weekly net-liquidity changes against weekly bitcoin returns with a
noise band that keeps the series' smoothness, (b) tests one monthly ma50 filter under the full fair exam with the
same pass test as nfci_filter, and (c) adds Treasury bills outstanding for the Keyrock claim.

Pre-registered (written before this study was run; no other variants, no parameter changes afterwards):
  Data (key-less, fixed user agent "btc-research-dashboard", never another one; each piece tries FRED's fredgraph CSV
  first with a 10 s timeout and 2 retries, and once FRED fails it is skipped for the rest of the run):
    * Fed total assets, Wednesday level: FRED WALCL, else the Fed Board's H.4.1 Data Download Program package
      (RESPPA_N.WW). Starts 2002-12-18.
    * Treasury General Account, Wednesday level: FRED WDTGAL, else H.4.1 (RESPPLLDT_N.WW), else the Treasury's Daily
      Treasury Statement (Fiscal Data API, operating cash balance, closing balance on Wednesdays, from 2005-10).
      Starts 2002-12-18. The latest Wednesdays are cross-checked against the Daily Treasury Statement (report only).
    * Overnight reverse repo: FRED RRPONTSYD (daily, from 2003-02-07), else the New York Fed markets API (reverse repo
      operations, daily totals accepted), else H.4.1 "reverse repurchase agreements: others" Wednesday level
      (RESPPLLRD_N.WW; foreign official reverse repos are NOT included, as in the popular formula). Daily data ->
      the Wednesday's total; no operation that Wednesday -> the Tuesday's total if the facility was running daily
      (from 2013-09-23), else 0. Before the first operation: 0.
    * Net liquidity NL = assets - TGA - RRP on Wednesdays, in US dollars, from 2002-12-18.
    * Treasury bills outstanding (Keyrock part): Fiscal Data, Monthly Statement of the Public Debt table 1,
      Marketable / Bills, total, month end, from 2001-01. Report only.
    * Each piece's last good download is kept in common.STORE (dashboard/cache/_store/net_liquidity_*.json) and used
      only if every source fails. Today's vintage only (H.4.1 levels are rarely revised).
  Point in time: H.4.1 for Wednesday W is released Thursday W+1 at 4:30 pm ET (holidays: Friday). A Wednesday value
  is used only when W + 2 days <= the label day of the bitcoin daily candle at whose close (00:00 UTC) the decision is
  made - one day more than needed on a normal week, and enough when the release slips to Friday.
  Decision at the last daily close of month M (te._daily_close, exactly nfci_filter / business_cycle timing): the
  latest Wednesday with W + 2 days <= last calendar day of M; held during month M+1.
  Q1 (main rule): hold BTC when ma50 is on AND that Wednesday's 13-week change of NL (log NL now minus 13 Wednesdays
  earlier) is > 0. Robustness row (ONE only, no placebo): same with the 26-week change.
  Pass test - all four, as nfci_filter:
    (1) favorable > unfavorable average next-month BTC return in 2011-16, 2017-21 and 2022-now (business_cycle's event
        table, holding months from 2011-07);
    (2) ranking score >= ma50's (main_signal_check's four checks via business_cycle._Exam, incl. the 95% loss bar);
    (3) median deepest loss over the 1,728 textbook cells no worse than ma50's;
    (4) placebo share <= 10%: the weekly 13-week-change series circularly shifted by 26 ... (length - 26) weeks,
        200 draws, flags rebuilt (> 0) and the whole exam re-scored with the ordinary-person check at 50 simulations
        per clock (the real rule re-scored the same way).
  Lead/lag (report only, decides "who moves first"): Pearson correlation of NL changes at week t with bitcoin weekly
  log returns (Friday to Friday close) at week t+k, k = -13 ... +13; a Wednesday is paired with the bitcoin week that
  ends two days later (the week that contains it and its release), so k >= 1 is a usable lead. Two NL versions:
  1-week log change and 13-week log change. Periods: from 2011-01-07 and from 2020-01-01. Noise band per k: the NL
  series circularly shifted within the period by 26 ... (n - 26) weeks, 200 draws, 95th percentile of |r| (keeps the
  smoothness of the series, so smooth series get a wider band); also the rough 2/sqrt(n). Autocorrelation of both NL
  versions is reported so smoothness is not read as a lead. "Who moves first": count of k > 0 vs k < 0 beyond the
  band for the 1-week version.
  "13-week lead" claim holds only if BOTH (a) r(1-week NL change at t, BTC weekly return at t+13) > 0 and beyond its
  band in both periods, AND (b) r(13-week NL change at t, BTC log return over the next 13 weeks) > 0 and beyond its
  band (same shift method) in both periods. Reverse direction reported: r(BTC past 13-week return, NL 13-week change
  over the next 13 weeks).
  T-bill part (report only, no rule): monthly; 3-month (~13-week) log change of bills outstanding at month end vs
  bitcoin monthly log returns (month-end closes) at month t+k, k = -12 ... +12 (covers Keyrock's ~8 months);
  periods from 2011-01 and from 2021-01 (Keyrock's window); band as above with shifts of at least 6 months. The same
  table for NL (last Wednesday of the month, 3-month change) so the two can be compared on equal terms.
  Report only: calendar-year returns and days differing from ma50 (where the edge or loss comes from, from 2015-01-01).
  Look-ahead check: scramble prices and not-yet-usable NL values after three month-ends; no position decided up to
  then may change. Today: states only.
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
from dashboard.studies2 import main_signal_check as msc

SAFE = pd.Timedelta(days=2)      # Wednesday W usable at the close of the candle labelled W + 2 days or later
W_MAIN, W_ROB = 13, 26
GAP_W = 26                       # placebo / band minimum shift in weeks
GAP_M = 6                        # band minimum shift in months
DRAWS, PL_SIMS = bc.DRAWS, bc.PL_SIMS
BAND_DRAWS = 200
LAGS = range(-13, 14)
MLAGS = range(-12, 13)
LL_FROM, LL_RECENT = "2011-01-07", "2020-01-01"
TB_FROM, TB_RECENT = "2011-01-01", "2021-01-01"
DAILY_RRP_FROM = pd.Timestamp("2013-09-23")
FIRST = msc.DAILY_STARTS[0]

FRED_IDS = {"assets": "WALCL", "tga": "WDTGAL", "rrp": "RRPONTSYD"}
FRED_MULT = {"assets": 1e6, "tga": 1e6, "rrp": 1e9}
H41_ZIP = "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H41&filetype=zip"
H41_PAGE = "https://www.federalreserve.gov/releases/h41/"
H41_SERIES = {"assets": "RESPPA_N.WW", "tga": "RESPPLLDT_N.WW", "rrp": "RESPPLLRD_N.WW"}
FD = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
DTS_URL = FD + "/v1/accounting/dts/operating_cash_balance"
MSPD_URL = FD + "/v1/debt/mspd/mspd_table_1"
NYFED_URL = "https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json"
KEYROCK = "https://keyrock.com/the-liquidity-source-that-leads-bitcoin/"

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok
_RUN: dict = {"fred_down": None, "h41": None, "used": set()}


# ---------- data ----------

def _fred(key: str) -> dict:
    url = bc.FRED_CSV.format(FRED_IDS[key])
    frame = pd.read_csv(io.StringIO(bc._get(url, timeout=(10, 10), tries=3).text))
    s = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    _RUN["used"].add(url)
    return {str(d)[:10]: float(v) * FRED_MULT[key] for d, v in zip(frame.iloc[:, 0], s) if math.isfinite(v)}


def _h41(key: str) -> dict:
    if _RUN["h41"] is None:
        zf = zipfile.ZipFile(io.BytesIO(bc._get(H41_ZIP, timeout=(10, 180)).content))
        member = next(n for n in zf.namelist() if n.lower().endswith("_data.xml"))
        _RUN["h41"] = zf.read(member).decode("utf-8")
    text = _RUN["h41"]
    i = text.find(f'SERIES_NAME="{H41_SERIES[key]}"')
    if i < 0:
        raise RuntimeError(f"H.4.1 裡找不到 {H41_SERIES[key]}")
    j = text.find("</kf:Series>", i)
    out = {}
    for value, day in re.findall(r'OBS_VALUE="([^"]+)" TIME_PERIOD="([^"]+)"', text[i:j]):
        try:
            v = float(value) * 1e6  # millions of dollars
        except ValueError:
            continue
        if math.isfinite(v):
            out[day[:10]] = v
    _RUN["used"].add(H41_ZIP)
    return out


def _nyfed() -> dict:
    url = f"{NYFED_URL}?startDate=2003-01-01&endDate={common.today_utc()}"
    ops = bc._get(url, timeout=(10, 60)).json()["repo"]["operations"]
    out: dict = {}
    for o in ops:
        v = o.get("totalAmtAccepted")
        if o.get("operationType") == "Reverse Repo" and isinstance(v, (int, float)):
            out[o["operationDate"]] = out.get(o["operationDate"], 0.0) + float(v)
    _RUN["used"].add(NYFED_URL)
    return out


def _fd_pages(url: str, params: dict) -> list[dict]:
    rows, page = [], 1
    while True:
        q = dict(params, **{"page[size]": 10000, "page[number]": page})
        reply = bc._get(url + "?" + "&".join(f"{k}={v}" for k, v in q.items()), timeout=(10, 60)).json()
        rows += reply.get("data", [])
        if page >= int(reply.get("meta", {}).get("total-pages", 1) or 1):
            return rows
        page += 1


def _dts_value(row: dict) -> float | None:
    for k in ("close_today_bal", "open_today_bal"):
        try:
            v = float(row.get(k))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v * 1e6
    return None


DTS_TYPES = ("Federal Reserve Account", "Treasury General Account (TGA)", "Treasury General Account (TGA) Closing Balance")


def _dts(since: str = "2005-10-01") -> dict:
    """TGA closing balance on Wednesdays from the Daily Treasury Statement."""
    rows = _fd_pages(DTS_URL, {"fields": "record_date,account_type,close_today_bal,open_today_bal",
                               "filter": f"record_date:gte:{since}"})
    out = {}
    for r in rows:
        if r.get("account_type") in DTS_TYPES and pd.Timestamp(r["record_date"]).dayofweek == 2:
            v = _dts_value(r)
            if v is not None:
                out[r["record_date"]] = v
    _RUN["used"].add(DTS_URL)
    return out


def _bills_fetch() -> dict:
    rows = _fd_pages(MSPD_URL, {"fields": "record_date,security_type_desc,security_class_desc,total_mil_amt",
                                "filter": "security_class_desc:eq:Bills"})
    out = {}
    for r in rows:
        if r.get("security_type_desc") == "Marketable":
            try:
                v = float(r["total_mil_amt"]) * 1e6
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                out[r["record_date"]] = v
    _RUN["used"].add(MSPD_URL)
    return out


def _wednesdays(daily: dict, first: str = "2002-12-18") -> dict:
    """Daily reverse repo totals -> Wednesday level (no operation: Tuesday's when daily, else 0)."""
    s = pd.Series({pd.Timestamp(k): v for k, v in daily.items()}).sort_index()
    weds = pd.date_range(first, pd.Timestamp(common.today_utc()) - pd.Timedelta(days=1), freq="W-WED")
    weds = weds[weds <= s.index[-1]]
    out = {}
    for w in weds:
        if w in s.index:
            v = s[w]
        elif w >= DAILY_RRP_FROM and (w - pd.Timedelta(days=1)) in s.index:
            v = s[w - pd.Timedelta(days=1)]
        else:
            v = 0.0
        out[_d(w)] = float(v)
    return out


PIECES = {  # key: [(label, loader), ...] in order of preference
    "assets": [("FRED WALCL", lambda: _fred("assets")), ("聯準會 H.4.1 整包下載", lambda: _h41("assets"))],
    "tga": [("FRED WDTGAL", lambda: _fred("tga")), ("聯準會 H.4.1 整包下載", lambda: _h41("tga")),
            ("財政部每日報表（Fiscal Data）", _dts)],
    "rrp": [("FRED RRPONTSYD", lambda: _wednesdays(_fred("rrp"))), ("紐約聯準銀行市場資料 API", lambda: _wednesdays(_nyfed())),
            ("聯準會 H.4.1 整包下載（逆回購：其他）", lambda: _h41("rrp"))],
    "bills": [("財政部公債月報（Fiscal Data）", _bills_fetch)],
}
PNAME = {"assets": "聯準會總資產", "tga": "財政部一般帳戶（TGA）", "rrp": "隔夜逆回購（RRP）", "bills": "短期國庫券（T-bill）餘額"}


def _piece(key: str) -> tuple[pd.Series, str, list[str]]:
    path = common.STORE / f"net_liquidity_{key}.json"
    notes, days, where = [], None, ""
    for label, fn in PIECES[key]:
        if label.startswith("FRED") and _RUN["fred_down"]:
            continue
        try:
            got = fn()
            if len(got) < 200:
                raise RuntimeError(f"只回傳 {len(got)} 筆")
            days, where = got, label
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            if label.startswith("FRED"):
                _RUN["fred_down"] = type(exc).__name__
            notes.append(f"{PNAME[key]}：{label}這次抓不到（{type(exc).__name__}）。")
    if days is None:
        if not path.exists():
            raise RuntimeError(f"{PNAME[key]}所有來源都抓不到，也沒有存檔")
        days = json.loads(path.read_text(encoding="utf-8"))
        where = "上次存下來的資料"
        notes.append(f"{PNAME[key]}改用上次存下來的資料（到 {max(days)}）。")
    else:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(days.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s, where, notes


def _dts_check(tga: pd.Series) -> str:
    try:
        since = _d(tga.index[-1] - pd.Timedelta(days=70))
        dts = _dts(since)
    except Exception as exc:  # noqa: BLE001 - report only
        return f"沒有核對成功（{type(exc).__name__}）"
    both = [(k, tga.get(pd.Timestamp(k))) for k in dts if pd.Timestamp(k) in tga.index]
    if not both:
        return "最近幾週對不到日期"
    same = sum(abs(dts[k] - v) <= 1e6 for k, v in both)
    return f"最近 {len(both)} 個星期三裡 {same} 個和財政部每日報表的收盤餘額一致（差距 100 萬美元以內）"


def _load() -> dict:
    parts, where, problems = {}, {}, []
    for key in ("assets", "tga", "rrp"):
        s, w, n = _piece(key)
        parts[key], where[key] = s, w
        problems += n
    a = parts["assets"]
    a = a[a.index.dayofweek == 2]
    frame = pd.concat([a.rename("a"), parts["tga"].rename("t"), parts["rrp"].reindex(a.index).fillna(0.0).rename("r")],
                      axis=1, join="inner").dropna()
    nl = (frame["a"] - frame["t"] - frame["r"]).sort_index()
    nl = nl[nl > 0]
    age = (pd.Timestamp(common.today_utc()) - nl.index[-1]).days
    if age > 21:
        problems.append(f"淨流動性最新一週是 {_d(nl.index[-1])}，已經 {age} 天前，資料可能沒有更新。")
    bills, bwhere = None, ""
    try:
        bills, bwhere, n = _piece("bills")
        problems += n
    except Exception as exc:  # noqa: BLE001 - the T-bill part is report only
        problems.append(f"短期國庫券餘額這次取不到（{type(exc).__name__}），那一段跳過，沒有用別的方法代替。")
    dts = _dts_check(parts["tga"])
    other, more = msc._load_other()
    return {"close": bc.te._daily_close(), "nl": nl, "parts": parts, "where": where, "bills": bills, "bwhere": bwhere,
            "dts": dts, "other": other, "problems": problems + more}


# ---------- the signal ----------

def _chg(nl: pd.Series, weeks: int) -> pd.Series:
    return np.log(nl).diff(weeks)


def _monthly(fav_w: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Decision month M -> flag from the latest Wednesday W with W + 2 days <= the last calendar day of M."""
    last_day = months.to_timestamp(how="end").normalize()
    pos = np.searchsorted(fav_w.index.values, (last_day - SAFE).values, side="right") - 1
    vals = np.where(pos >= 0, fav_w.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=months)


def _fav(chg: pd.Series) -> pd.Series:
    return (chg > 0).astype(float).where(chg.notna())


def _flags(nl: pd.Series, weeks: int, months) -> pd.Series:
    return _monthly(_fav(_chg(nl, weeks)), months)


def _used_week(s: pd.Series, month: pd.Period) -> pd.Timestamp | None:
    cut = month.to_timestamp(how="end").normalize() - SAFE
    k = s.index.searchsorted(cut, side="right") - 1
    return s.index[k] if k >= 0 else None


# ---------- lead / lag ----------

def _corr_at(a: np.ndarray, b: np.ndarray, k: int) -> tuple[float, int]:
    """r(a_t, b_{t+k}) on the positions where both exist."""
    if k >= 0:
        x, y = a[:len(a) - k], b[k:]
    else:
        x, y = a[-k:], b[:len(b) + k]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def _xcorr(a: pd.Series, b: pd.Series, lags, since: str, gap: int, seed: int) -> dict:
    """r(a_t, b_{t+k}) for each k with a circular-shift noise band (a shifted within the window)."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    both = both[both.index >= pd.Timestamp(since)]
    first = both.dropna().index
    both = both[(both.index >= first[0]) & (both.index <= both["b"].last_valid_index())]
    av, bv = both["a"].values.astype(float), both["b"].values.astype(float)
    n = len(av)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(gap, n - gap + 1, BAND_DRAWS)
    out = {}
    for k in lags:
        r, cnt = _corr_at(av, bv, k)
        noise = [abs(_corr_at(np.roll(av, int(s)), bv, k)[0]) for s in shifts]
        band = float(np.nanpercentile(noise, 95))
        out[k] = {"r": r, "n": cnt, "band": band, "beyond": bool(math.isfinite(r) and abs(r) > band)}
    return {"lags": out, "n": n, "first": both.index[0], "last": both.index[-1], "rough": 2 / math.sqrt(max(n, 1))}


def _leadlag(close: pd.Series, nl: pd.Series) -> dict:
    lp = np.log(close.resample("W-FRI").last())
    if close.index[-1].dayofweek != 4:
        lp = lp.iloc[:-1]
    wk = lp.diff()
    x1 = _chg(nl, 1)
    x13 = _chg(nl, W_MAIN)
    x1.index = x1.index + SAFE      # Wednesday -> the Friday that ends its bitcoin week
    x13.index = x13.index + SAFE
    fwd13 = lp.shift(-W_MAIN) - lp  # next 13 weeks
    past13 = lp - lp.shift(W_MAIN)
    out = {}
    for lab, since in (("all", LL_FROM), ("recent", LL_RECENT)):
        a1 = _xcorr(x1, wk, LAGS, since, GAP_W, bc.SEED + 7)
        a13 = _xcorr(x13, wk, LAGS, since, GAP_W, bc.SEED + 8)
        claim = _xcorr(x13, fwd13, [0], since, GAP_W, bc.SEED + 9)["lags"][0]
        rev = _xcorr(past13, x13, [W_MAIN], since, GAP_W, bc.SEED + 10)["lags"][W_MAIN]
        xs = x1[x1.index >= pd.Timestamp(since)].dropna()
        xs13 = x13[x13.index >= pd.Timestamp(since)].dropna()
        out[lab] = {"w1": a1, "w13": a13, "claim": claim, "rev": rev,
                    "ac1": [float(xs.autocorr(k)) for k in (1, 2, 3)],
                    "ac13": [float(xs13.autocorr(k)) for k in (1, 4, 13)]}
    return out


def _monthly_series(close: pd.Series, nl: pd.Series, bills: pd.Series | None) -> dict:
    me = bc._month_ends(close)
    btc = np.log(me).diff()
    btc.index = btc.index.to_timestamp(how="end").normalize()
    nlm = nl.groupby(nl.index.to_period("M")).last()
    nlm.index = nlm.index.to_timestamp(how="end").normalize()
    out = {"btc": btc, "nl": np.log(nlm).diff(3)}
    if bills is not None:
        b = bills.groupby(bills.index.to_period("M")).last()
        b.index = b.index.to_timestamp(how="end").normalize()
        out["bills"] = np.log(b).diff(3)
    return out


def _tbill(close: pd.Series, nl: pd.Series, bills: pd.Series | None) -> dict | None:
    m = _monthly_series(close, nl, bills)
    res = {}
    for lab, since in (("all", TB_FROM), ("recent", TB_RECENT)):
        res[lab] = {"nl": _xcorr(m["nl"], m["btc"], MLAGS, since, GAP_M, bc.SEED + 11)}
        if "bills" in m:
            res[lab]["bills"] = _xcorr(m["bills"], m["btc"], MLAGS, since, GAP_M, bc.SEED + 12)
    return res


# ---------- look-ahead ----------

def _lookahead(data: dict, ex: bc._Exam, months) -> tuple[int, int]:
    rng = np.random.default_rng(13)
    close, nl = data["close"], data["nl"]
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        c2 = close.where(close.index <= c, close * np.exp(rng.normal(0, 0.2, len(close))))
        ma2 = msc._targets(c2)["ma50"].values.astype(float)
        nl2 = nl.where(nl.index + SAFE <= c, nl * np.exp(rng.normal(0, 0.3, len(nl))))
        upto = close.index <= c
        for w in (W_MAIN, W_ROB):
            a = bc._pos("m2", ex.btc_ma, bc._on_days(_flags(nl, w, months), close.index + bc.DAY))
            b = bc._pos("m2", ma2, bc._on_days(_flags(nl2, w, months), close.index + bc.DAY))
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a[upto], nan=-1), np.nan_to_num(b[upto], nan=-1)))
    return total, bad


# ---------- the computation ----------

def _compute(data: dict) -> dict:
    t0 = time.time()
    close, nl = data["close"], data["nl"]
    months = pd.period_range(nl.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)
    F = _flags(nl, W_MAIN, months)
    F26 = _flags(nl, W_ROB, months)
    rng = np.random.default_rng(bc.SEED + 5)
    ev = bc._events(close, F, rng)
    ev26 = bc._events(close, F26, rng)
    s = {"ma50": ex.summary("ma50", None), "q1": ex.summary("m2", F), "q26": ex.summary("m2", F26)}

    small = {"ma50": ex.summary("ma50", None, PL_SIMS), "q1": ex.summary("m2", F, PL_SIMS)}
    chg = _chg(nl, W_MAIN).dropna()
    prng = np.random.default_rng(bc.SEED + 41)
    n = len(chg)
    hits, lows, beat_ma = 0, [], 0
    for sh in prng.integers(GAP_W, n - GAP_W + 1, DRAWS):
        c2 = pd.Series(np.roll(chg.values, int(sh)), index=chg.index)
        f2 = _monthly(_fav(c2), months)
        x = ex.summary("m2", f2, PL_SIMS)
        lows.append(x["low"])
        hits += int(bc._score_ge(x, small["q1"]))
        beat_ma += int(bc._score_ge(x, small["ma50"]))

    pos = {"ma50": ex.btc_pos("ma50", None), "q1": ex.btc_pos("m2", F)}
    a = close.index >= pd.Timestamp(FIRST)
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["q1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()), int(((p0 > 0.5) & (p1 < 0.5)).sum()))
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST),
             **{k: bc._years(pos[k], close, FIRST) for k in pos}}
    look = _lookahead(data, ex, months)
    ll = _leadlag(close, nl)
    tb = _tbill(close, nl, data["bills"])
    return {"ex": ex, "F": F, "F26": F26, "ev": ev, "ev26": ev26, "s": s, "small": small, "plc": hits / DRAWS,
            "plc_low": float(np.median(lows)), "plc_ma": beat_ma / DRAWS, "pos": pos, "differ": differ,
            "years": years, "look": look, "ll": ll, "tb": tb, "problem": problem, "secs": time.time() - t0}


# ---------- text helpers ----------

def _mark(c: dict) -> str:
    r = c["r"]
    return (f"{r:+.3f}" + ("＊" if c["beyond"] else "")) if _ok(r) else "—"


def _who(lags: dict) -> tuple[list[int], list[int]]:
    lead = [k for k in lags if k > 0 and lags[k]["beyond"]]
    lagg = [k for k in lags if k < 0 and lags[k]["beyond"]]
    return lead, lagg


def _best(lags: dict, sign: int) -> int:
    ks = [k for k in lags if (k > 0 if sign > 0 else k < 0) and _ok(lags[k]["r"])]
    return max(ks, key=lambda k: abs(lags[k]["r"]))


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    _RUN["used"] = set()
    data = _load()
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    nl = data["nl"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base, q1, q26 = s["ma50"], s["q1"], s["q26"]
    n_c, n_d = base["c_n"], base["d_n"]

    q = {"event": ev["all_three"], "score": bc._score_ge(q1, base),
         "loss": q1["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["plc"] <= 0.10}
    q["pass"] = all(q.values())

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    # ----- table 1: event table -----
    rows1 = []
    for lab, e in (("Q1：淨流動性 13 週變化 > 0（主要）", ev), ("淨流動性 26 週變化 > 0（穩健度，只報告）", r["ev26"])):
        a = e["all"]
        ci = a.get("ci")
        rows1.append([lab, f"{a['nf']} 個月｜{_p(a['mf'])}", f"{a['nu']} 個月｜{_p(a['mu'])}",
                      f"{_p(a['diff'])}（{_p(ci[0])} 到 {_p(ci[1])}）" if ci else _p(a["diff"])]
                     + [f"{_p(p['diff'])}（{p['nf']}｜{p['nu']}）" for p in e["periods"].values()]
                     + ["是" if e["all_three"] else "否"])
    t1 = {"caption": ("每個月最後一根日線（UTC 0 點收盤）決定：用那時已經公布、而且日期至少早 2 天的最新一個星期三的淨流動性"
                      "（聯準會總資產 − 財政部一般帳戶 − 隔夜逆回購），和 13 週前比較，變多就算「有利」。每格是下一個月比特幣報酬"
                      f"（月底收盤到月底收盤）的平均，持有月份 {ev['first']} 到 {ev['last']}；差距＝有利減不利，括號是區塊拔靴法"
                      f"（連續 {bc.BLOCK} 個月一塊、{bc.BOOT:,} 次）的 90% 區間；分段欄括號是（有利｜不利）月數。"
                      "這張表只看淨流動性本身，不含 50 日均線。"),
          "columns": ["設定", "有利月：個數｜平均報酬", "不利月：個數｜平均報酬", "差距（90% 區間）"]
                     + [f"{lab} 差距" for lab, _, _ in bc.PERIODS] + ["三段都是有利較好"],
          "rows": rows1}

    # ----- table 2: fair exam -----
    rows2 = []
    for lab, x in (("50 日均線（基準）", base), ("Q1：50 日均線而且淨流動性 13 週變多", q1),
                   ("50 日均線而且淨流動性 26 週變多（只報告）", q26)):
        ck = x["checks"]
        rows2.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    hd = ex.chold[1]
    rows2.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(hd)))}｜—", "—"])
    t2 = {"caption": (f"main_signal_check 的公平考試（四項都是「報酬贏過同一天起一直抱著」的比例）：比特幣 UTC 0 點日線 {n_d} 個起算月；"
                      f"{len(ex.cstarts)} 個起算月 × 24 個收盤時間 ＝ {n_c:,} 格照課本做（成本 {bc.COST * 100:.2f}%）；同樣 {n_c:,} 格照一般人方式做"
                      f"（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、一成進場漏掉，每格 {bc.SIMS} 次）；其他資產（ETH、SOL、BNB、XRP、MSTR）"
                      "用自己的 50 日均線、同一個美國淨流動性訊號。排名分數＝四項裡最低的那項，前提是最深虧損比一直抱著淺的比例至少 95%。"
                      "最後兩欄是照課本｜一般人做法的中位數。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows2}

    # ----- table 3: pass test -----
    ps = [p["diff"] for p in ev["periods"].values()]
    rows3 = [["有利月比不利月好（三段）", "、".join(_p(v) for v in ps), yes(q["event"])],
             ["排名分數不低於 50 日均線", f"{_share(q1['low'])} 對 {_share(base['low'])}", yes(q["score"])],
             [f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(q1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(q["loss"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['plc'])}（打亂後排名分數中位數 {_share(r['plc_low'])}，真的 {_share(r['small']['q1']['low'])}；"
              f"打亂後不輸均線的 {_share(r['plc_ma'])}）", yes(q["plc"])],
             ["結果", "—", "通過" if q["pass"] else "沒有通過"]]
    t3 = {"caption": (f"事先寫好的過關標準，四項都要過。隨機錯位：把每週「13 週變化」整串循環平移 {GAP_W} 週到「長度 − {GAP_W}」週、"
                      f"{DRAWS} 次，重新算每月訊號、整套考試（含一般人做法）重做；為了在十分鐘內算完，這一步一般人做法每個收盤時間只模擬 "
                      f"{PL_SIMS} 次（種子相同），真的規則也用同樣方式算一次再比（簡化版）。"),
          "columns": ["條件", "數字", "結果"], "rows": rows3}

    # ----- table 4: years -----
    dd_, nn, out_ = r["differ"]
    rows4 = []
    yrs = sorted(r["years"]["ma50"])
    for y in yrs:
        a, b = r["years"]["ma50"].get(y), r["years"]["q1"].get(y)
        rows4.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(r["years"]["hold"].get(y)), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    rows4.append([f"和 50 日均線持有狀態不同的天數（{FIRST} 起）", "—", "—", f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—",
                  f"其中均線持有、Q1 現金 {out_:,} 天"])
    t4 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看 Q1 和 50 日均線的差來自哪幾年；"
                      "Q1 只會比均線少持有（淨流動性 13 週變少時，均線持有也改成現金）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "Q1", "Q1 減均線（百分點）"], "rows": rows4}

    # ----- table 5: lead / lag (weekly) -----
    ll = r["ll"]
    la, lr = ll["all"], ll["recent"]
    rows5 = []
    for k in LAGS:
        if k > 0:
            who = f"淨流動性先動，{k} 週後的比特幣"
        elif k < 0:
            who = f"比特幣先動，{-k} 週後的淨流動性"
        else:
            who = "同一週"
        rows5.append([f"{k:+d}" if k else "0", who, _mark(la["w1"]["lags"][k]), _mark(lr["w1"]["lags"][k]),
                      _mark(la["w13"]["lags"][k]), _mark(lr["w13"]["lags"][k])])
    bands = lambda x: f"{min(v['band'] for v in x['lags'].values()):.3f} 到 {max(v['band'] for v in x['lags'].values()):.3f}"  # noqa: E731
    t5 = {"caption": ("誰先動：淨流動性的變化（週 t）和比特幣週報酬（週五到週五收盤的對數報酬，週 t+k）的相關係數。"
                      "每個星期三配到兩天後那個週五結束的比特幣週（包含它和它公布的那天），所以 k ≥ 1 才是真的能用的「領先」。"
                      f"k 為負＝比特幣先變、之後幾週的淨流動性。全期 {_d(la['w1']['first'])} 到 {_d(la['w1']['last'])}（{la['w1']['n']} 週）；"
                      f"{LL_RECENT[:4]} 年起 {lr['w1']['n']} 週。＊＝超過隨機範圍：把淨流動性整串循環平移至少 {GAP_W} 週、{BAND_DRAWS} 次，"
                      f"取相關絕對值的第 95 百分位（會保留它本身的平滑程度）。1 週變化的範圍：全期 {bands(la['w1'])}、近期 {bands(lr['w1'])}；"
                      f"13 週變化的範圍：全期 {bands(la['w13'])}、近期 {bands(lr['w13'])}（粗略的 2/√週數 是 ±{la['w1']['rough']:.3f}）。"),
          "columns": ["k（週）", "意思", "1 週變化：全期", f"1 週變化：{LL_RECENT[:4]} 年起", "13 週變化：全期",
                      f"13 週變化：{LL_RECENT[:4]} 年起"],
          "rows": rows5}

    # ----- table 6: the 13-week lead claim -----
    def crow(lab, x):
        return [lab, _mark(x), f"±{x['band']:.3f}", f"{x['n']}"]
    rows6 = []
    for per, x in (("全期", la), (f"{LL_RECENT[:4]} 年起", lr)):
        rows6.append(crow(f"{per}：1 週變化 → 13 週後那一週的比特幣", x["w1"]["lags"][13]))
        rows6.append(crow(f"{per}：13 週變化 → 之後 13 週的比特幣報酬", x["claim"]))
        rows6.append(crow(f"{per}：反方向，比特幣過去 13 週報酬 → 之後 13 週的淨流動性變化", x["rev"]))
    claim_ok = all(x["w1"]["lags"][13]["beyond"] and x["w1"]["lags"][13]["r"] > 0
                   and x["claim"]["beyond"] and x["claim"]["r"] > 0 for x in (la, lr))
    rows6.append(["「領先 13 週」的說法（兩個時期、兩個版本都要正相關而且超過範圍）", "—", "—", "成立" if claim_ok else "不成立"])
    t6 = {"caption": ("流行說法「淨流動性領先比特幣 13 週」的直接檢查（事先寫好）。13 週的變化和 13 週的報酬每週只換掉一小段，"
                      "所以隨機範圍用同樣的循環平移算，會比一般的 ±2/√n 寬很多。"),
          "columns": ["檢查", "相關", "隨機範圍", "週數"], "rows": rows6}

    # ----- table 7: T-bills vs net liquidity (monthly) -----
    tb = r["tb"]
    has_bills = data["bills"] is not None and "bills" in tb["all"]
    rows7 = []
    for k in MLAGS:
        if k > 0:
            who = f"先動，{k} 個月後的比特幣"
        elif k < 0:
            who = f"比特幣先動，{-k} 個月後"
        else:
            who = "同一個月"
        row = [f"{k:+d}" if k else "0", who, _mark(tb["all"]["nl"]["lags"][k]), _mark(tb["recent"]["nl"]["lags"][k])]
        row += ([_mark(tb["all"]["bills"]["lags"][k]), _mark(tb["recent"]["bills"]["lags"][k])] if has_bills else ["—", "—"])
        rows7.append(row)
    t7 = {"caption": ("Keyrock 的說法：短期國庫券（T-bill）發行比聯準會淨流動性更能領先比特幣（2021 年起、大約 8 個月）。這裡用每月月底的"
                      "短期國庫券餘額（財政部公債月報）和月底的淨流動性，各算 3 個月（約 13 週）的對數變化，對比特幣月報酬（月底收盤）。"
                      f"k 為正＝這兩個先動、k 個月後的比特幣。全期 {TB_FROM[:4]} 年起，近期 {TB_RECENT[:4]} 年起（Keyrock 的區間）。"
                      f"＊＝超過隨機範圍（整串循環平移至少 {GAP_M} 個月、{BAND_DRAWS} 次，第 95 百分位）。只報告，不做規則。"
                      + ("" if has_bills else "這次沒有拿到短期國庫券資料，那兩欄空白。")),
          "columns": ["k（月）", "意思", "淨流動性：全期", f"淨流動性：{TB_RECENT[:4]} 年起", "短期國庫券：全期",
                      f"短期國庫券：{TB_RECENT[:4]} 年起"],
          "rows": rows7}

    # ----- today -----
    last = close.index[-1]
    cur = (last + bc.DAY).to_period("M") - 1
    used = _used_week(nl, cur)
    c13 = _chg(nl, W_MAIN)
    fnow = F.get(cur, np.nan)
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    def tr(v) -> str:
        return f"{v / 1e12:,.2f} 兆美元" if _ok(v) else "—"

    parts = data["parts"]
    lw = nl.index[-1]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
         "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"},
        {"label": f"淨流動性（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
         "value": (f"{'13 週變多、有利' if fnow == 1 else '13 週變少、不利' if fnow == 0 else '—'}；用 "
                   f"{_d(used) if used is not None else '—'} 那週 {tr(nl.get(used, np.nan))}，13 週變化 "
                   f"{_p(math.expm1(c13.get(used, np.nan)) if _ok(c13.get(used, np.nan)) else None)}")},
        {"label": "最新公布的淨流動性",
         "value": (f"{_d(lw)}：總資產 {tr(parts['assets'].get(lw))} − 財政部帳戶 {tr(parts['tga'].get(lw))} − 逆回購 "
                   f"{tr(parts['rrp'].reindex([lw]).fillna(0.0).iloc[0])} ＝ {tr(nl.iloc[-1])}；13 週變化 "
                   f"{_p(math.expm1(c13.iloc[-1]))}（下一次決定看月底時能用的那一週）")},
        {"label": "比特幣：50 日均線｜Q1", "value": f"{word(pm['ma50'][-1])}｜{word(pm['q1'][-1])}"},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    a0 = ev["all"]
    better = [y for y in yrs if _ok(r["years"]["q1"].get(y)) and _ok(r["years"]["ma50"].get(y))
              and r["years"]["q1"][y] - r["years"]["ma50"][y] > 0.5]
    worse = [y for y in yrs if _ok(r["years"]["q1"].get(y)) and _ok(r["years"]["ma50"].get(y))
             and r["years"]["q1"][y] - r["years"]["ma50"][y] < -0.5]
    w1a, w1r = la["w1"]["lags"], lr["w1"]["lags"]
    lead_a, lag_a = _who(w1a)
    lead_r, lag_r = _who(w1r)
    c0a, c0r = w1a[0], w1r[0]
    n_lead, n_lag = len(lead_a) + len(lead_r), len(lag_a) + len(lag_r)
    if n_lead == 1 and n_lag == 0:
        mech = ("只有一格「淨流動性先動」超過範圍，另一個時期沒有重現；兩個時期加起來 26 格裡偶然也會有一兩格，"
                "所以看不出可靠的領先，主要是同一週一起動。")
    elif (lead_a or lead_r) and not (lag_a or lag_r):
        mech = "有淨流動性先動、比特幣之後才跟的跡象。"
    elif (lag_a or lag_r) and not (lead_a or lead_r):
        mech = "超過隨機範圍的只有比特幣先動的那一側，和「淨流動性先動」的說法相反。"
    elif lead_a or lead_r or lag_a or lag_r:
        mech = "兩個方向都有零星幾週超過隨機範圍，看不出穩定的先後。"
    else:
        mech = "前後 13 週都沒有超過隨機範圍，看不出誰先動。"
    same = (f"同一週相關全期 {c0a['r']:+.3f}{'（超過範圍）' if c0a['beyond'] else '（沒超過範圍）'}、"
            f"{LL_RECENT[:4]} 年起 {c0r['r']:+.3f}{'（超過範圍）' if c0r['beyond'] else '（沒超過範圍）'}。")
    lead_txt = (f"1 週變化超過範圍的 k：全期 淨流動性先動 {('、'.join(f'+{k}' for k in lead_a)) or '無'}、比特幣先動 "
                f"{('、'.join(str(k) for k in lag_a)) or '無'}；{LL_RECENT[:4]} 年起 淨流動性先動 "
                f"{('、'.join(f'+{k}' for k in lead_r)) or '無'}、比特幣先動 {('、'.join(str(k) for k in lag_r)) or '無'}。")
    k13 = sorted({k for x in (la, lr) for k, v in x["w13"]["lags"].items() if v["beyond"]})
    if k13 and all(-12 <= k <= 0 for k in k13):
        mech += (f"13 週變化在 k ＝ {k13[0]:+d} 到 {k13[-1]:+d} 週超過範圍".replace("+0 週", "0 週")
                 + f"，但這段的 13 週變化和那幾週的比特幣報酬本來就落在同一段時間，"
                 "代表的是「一起動」，不是比特幣先動。")
    if has_bills:
        ba, br = tb["all"]["bills"]["lags"], tb["recent"]["bills"]["lags"]
        na, nr = tb["all"]["nl"]["lags"], tb["recent"]["nl"]["lags"]
        kb, kn = _best(br, 1), _best(nr, 1)
        tbt = (f"短期國庫券（Keyrock 的說法）：{TB_RECENT[:4]} 年起 3 個月變化領先比特幣最明顯的是 {kb} 個月後 {_mark(br[kb])}"
               f"（8 個月後 {_mark(br[8])}；全期 8 個月後 {_mark(ba[8])}）；淨流動性同樣算法最明顯的是 {kn} 個月後 {_mark(nr[kn])}"
               f"（全期 {_mark(na[kn])}）。月資料點數少（{TB_RECENT[:4]} 年起約 {tb['recent']['bills']['n']} 個月），＊才算超過隨機範圍；"
               f"{TB_RECENT[:4]} 年起短期國庫券領先的那一側超過範圍的月數："
               f"{('、'.join(f'+{k}' for k in MLAGS if k > 0 and br[k]['beyond'])) or '無'}，全期："
               f"{('、'.join(f'+{k}' for k in MLAGS if k > 0 and ba[k]['beyond'])) or '無'}。"
               "Keyrock 說的 r = +0.80 這裡用變化算不出來，在這個算法下是「方向對、但弱、而且分不出是不是巧合」。")
    else:
        tbt = "短期國庫券資料這次沒有取得，那一段跳過。"
    verdict = (f"{'通過' if q['pass'] else '沒有通過'}事先寫好的標準。Q1（50 日均線而且淨流動性 13 週變多）："
               f"淨流動性變多的月份，下個月比特幣平均比變少的月份多 {_p(a0['diff'])}，三段 {'、'.join(_p(v) for v in ps)}（{yes(q['event'])}）；"
               f"排名分數 {_share(q1['low'])} 對均線 {_share(base['low'])}（{yes(q['score'])}）；最深虧損中位數 {_pct(q1['c_deep_med'])} "
               f"對 {_pct(base['c_deep_med'])}（{yes(q['loss'])}）；隨機錯位一樣好 {_share(r['plc'])}（{yes(q['plc'])}）。"
               f"穩健度（26 週變化）排名分數 {_share(q26['low'])}、差距 {_p(r['ev26']['all']['diff'])}。"
               + (f"比均線好的年份：{'、'.join(str(y) for y in better)}" if better else "沒有一年明顯比均線好")
               + (f"；比均線差的年份：{'、'.join(str(y) for y in worse)}。" if worse else "。")
               + "誰先動：" + same + lead_txt + mech
               + f"「領先 13 週」的說法{'成立' if claim_ok else '不成立'}（13 週變化對之後 13 週比特幣：全期 {_mark(la['claim'])}、"
               f"{LL_RECENT[:4]} 年起 {_mark(lr['claim'])}）。" + tbt)

    tot, bad = r["look"]
    st = {k: _d(v.index[0]) for k, v in parts.items()}
    st["rrp"] = _d(parts["rrp"][parts["rrp"] > 0].index[0])  # first Wednesday with a reverse repo outstanding
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好）：Q1 = hold BTC when ma50 is on AND the 13-week log change of net "
         "liquidity (WALCL - TGA - RRP, Wednesday levels) at the latest Wednesday W with W + 2 days <= the last calendar day "
         "of the month is > 0; decided at the last daily close of the month, held for the next month. Robustness (one row, "
         "no placebo): 26-week change. Pass test: favorable > unfavorable next-month BTC return in 2011-16, 2017-21, "
         "2022-now AND ranking score >= ma50's AND median deepest loss no worse AND placebo share (weekly 13-week-change "
         "series circularly shifted >= 26 weeks, 200 draws, full ranking score incl. ordinary person with 50 simulations "
         "per clock) <= 10%. Lead/lag -13..+13 weeks with a circular-shift band, 13-week-lead claim test and the T-bill "
         "table: report only."),
        ("淨流動性是什麼：聯準會資產負債表的總資產，減掉財政部放在聯準會的現金（一般帳戶，TGA），再減掉貨幣基金等放在聯準會的"
         "隔夜逆回購（RRP）。說法是：這三項算出來的錢才是真正留在市場裡的，變多時風險資產會漲。這裡的逆回購只算隔夜逆回購"
         "（FRED 的 RRPONTSYD），不算外國央行的逆回購，和流行的算法一樣。"),
        (f"資料來源與起始日：總資產 {data['where']['assets']}（{st['assets']} 起）；財政部一般帳戶 {data['where']['tga']}"
         f"（{st['tga']} 起）；逆回購 {data['where']['rrp']}（{st['rrp']} 起，在第一次操作之前算 0）。淨流動性 {_d(nl.index[0])} 到 "
         f"{_d(nl.index[-1])}，{len(nl):,} 週。FRED 有時候對這台電腦沒有回應（每次等 10 秒、最多試 3 次），沒回應就改用發布單位自己的"
         f"檔案，數字相同。財政部一般帳戶和財政部每日報表核對：{data['dts']}。"
         + (f"短期國庫券：{data['bwhere']}（{_d(data['bills'].index[0])} 到 {_d(data['bills'].index[-1])}，每月）。" if has_bills else "")),
        ("時間對齊：每週三的數字在週四美東下午 4 點半公布（假日延到週五），這裡保守地要求那個星期三至少比做決定的那根日線早 2 天。"
         f"偷看未來的檢查：三個月底之後把價格和那時還不能用的淨流動性換成亂數，切點以前的持有狀態 {tot} 組裡 {tot - bad} 組完全沒變"
         + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("「誰先動」怎麼讀：兩條線如果都很平滑，同一週的相關會往前後攤開，看起來像領先或落後。"
         f"淨流動性每週變化本身的相關（自相關）：隔 1 週 {la['ac1'][0]:+.2f}、2 週 {la['ac1'][1]:+.2f}、3 週 {la['ac1'][2]:+.2f}；"
         f"13 週變化非常平滑：隔 1 週 {la['ac13'][0]:+.2f}、4 週 {la['ac13'][1]:+.2f}、13 週 {la['ac13'][2]:+.2f}。"
         "所以 13 週變化那兩欄一長串同號的數字不代表很多次獨立的證據；隨機範圍用整串平移算，就是為了把這種平滑考慮進去。"
         "27 個 k 裡偶爾有一兩個超過範圍也可能是巧合。另外，13 週（或 3 個月）的變化本身就涵蓋過去 13 週，"
         "所以 k 在 −12 到 0 週（−2 到 0 個月）時，它和比特幣報酬其實是同一段時間，超過範圍也只代表一起動；"
         "要說淨流動性領先，要看 k ≥ 1；要說比特幣領先，要看 k ≤ −13 週（−3 個月）。"),
        ("和 liquidity_lead 的差別：那份研究測的是全球 M2（淨流動性當第二個版本），用流行的固定往後移 70 天、每天開關的規則，"
         "看各時期的報酬波動比。這份研究把淨流動性前後 13 週的先後關係整張畫出來、用保留平滑度的隨機範圍判斷，"
         "並用和 nfci_filter 一樣的每月決定、完整公平考試去測它能不能幫 50 日均線，另外加上 Keyrock 說的短期國庫券。"),
        ("Keyrock 文章（Amir Hajian，2026）說的是：2021 年起「財政部的 QE（短期國庫券發行）」和比特幣的領先關係最強（r = +0.80，"
         "大約領先 8 個月，Granger 檢定在 9–16 個月最顯著）；聯準會自己的 QE 沒有統計上顯著的關係；全球流動性指數用領先 13 週"
         "（他們說 13 週配得最好）。文章沒有給淨流動性單獨的領先週數，也沒有寫是用水準還是變化做相關、有沒有扣掉趨勢。"
         "兩條都在上升的線做水準相關，很容易得到很高的 r；這裡用的是變化，所以數字不能直接比。"),
        ("其他幣、MSTR 用的是同一個美國淨流動性訊號。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的一條規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    used = _RUN["used"]
    sources = [{"name": "Keyrock：The Liquidity Source That Leads Bitcoin（說短期國庫券發行領先比特幣約 8 個月、2021 年起 r = +0.80，"
                        "比聯準會資產負債表更有關）", "url": KEYROCK}]
    for sid in ("WALCL", "WDTGAL", "RRPONTSYD"):
        url = bc.FRED_CSV.format(sid)
        sources.append({"name": f"FRED：{sid}" + ("" if url in used else "（這次沒有回應，沒有用到）"), "url": url})
    if H41_ZIP in used:
        sources.append({"name": "聯準會 H.4.1 資料下載（總資產、財政部一般帳戶、逆回購：其他，週三數字）", "url": H41_ZIP})
    sources.append({"name": "聯準會 H.4.1 發布頁", "url": H41_PAGE})
    if NYFED_URL in used:
        sources.append({"name": "紐約聯準銀行市場資料 API：逆回購操作", "url": NYFED_URL})
    if DTS_URL in used:
        sources.append({"name": "美國財政部 Fiscal Data：每日報表（Daily Treasury Statement）現金餘額", "url": DTS_URL})
    if MSPD_URL in used:
        sources.append({"name": "美國財政部 Fiscal Data：公債月報（Monthly Statement of the Public Debt）表 1，短期國庫券餘額",
                        "url": MSPD_URL})
    sources += [
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]

    return {
        "id": "net_liquidity",
        "title": "聯準會淨流動性會不會領先比特幣、能不能幫 50 日均線",
        "question": ("常聽到的說法：聯準會總資產減掉財政部帳戶、再減掉隔夜逆回購，就是市場上的「淨流動性」，它先動、比特幣後跟，"
                     "常說領先 13 週。Keyrock 更進一步說，財政部發的短期國庫券比這個更能領先比特幣。到底是誰先動？"
                     "只在淨流動性 13 週變多時才照 50 日均線持有，能不能通過完整的公平考試？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5, t6, t7],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
