"""Historical study "economy_family": does a whole FAMILY of real-economy and policy indicators help the 50-day moving
average rule (ma50, main_signal_check's main signal) for bitcoin - more often than luck would?

Why a family: two "ma50 AND slow macro state" rules passed the fair exam (N1 = nfci_filter, Chicago Fed financial
conditions below their 52-week average; O2 = oecd_followup, OECD US leading indicator above its 10-month average) and
one failed (business_cycle, Philadelphia Fed survey). Testing more indicators one at a time and keeping the winners is
multiple testing. So every member below gets ONE identical rule template fixed in advance, and the main result is the
family-level count compared with luck, not the best single member.

Pre-registered (written before this study was run; no other members, variants or parameter changes afterwards):
  Members (direction of "favorable" fixed by economic logic; publication lag stated; today's vintage only - all of
  these are revised and no key-less vintage archive is used, a stated caveat):
    claims_init  US weekly initial jobless claims, seasonally adjusted (Dept of Labor; FRED ICSA). 4-week average
                 BELOW the average of the last 52 weekly values = favorable. Week ending Saturday d is released the
                 following Thursday (d + 5); usable when d + 6 days <= the decision day.
    claims_cont  US continuing claims, seasonally adjusted (same DOL report; FRED CCSA). Latest value BELOW its 52-week
                 average = favorable. Released one week later than initial claims; usable when d + 13 days <= decision.
    cfnai        Chicago Fed National Activity Index, 3-month average (Chicago Fed; FRED CFNAIMA3). ABOVE its 12-month
                 average = favorable. Month m released in the 3rd-4th week of m+1; usable at the end of m+1 (lag 1).
    wei          Weekly Economic Index (published by the New York Fed until 2023, now by the Dallas Fed; FRED WEI).
                 ABOVE its 52-week average = favorable. Week ending Saturday d released the following Thursday; usable
                 when d + 6 days <= decision.
    umich        University of Michigan consumer sentiment index (sca.isr.umich.edu; FRED UMCSENT). ABOVE its 12-month
                 average = favorable. Final value comes out near the end of the month itself; to be safe month m is
                 used only at the end of m+1 (lag 1).
    permits      US building permits, total, seasonally adjusted annual rate (Census; FRED PERMIT). ABOVE its 12-month
                 average = favorable. Month m released around the 12th business day of m+1; lag 1.
    fedfunds     "Fed not hiking": effective federal funds rate (Federal Reserve H.15 daily; FRED DFF). Average of the
                 daily values over the 30 calendar days ending the day before the decision day, compared with the same
                 30-day average six calendar months earlier; favorable when now <= then + 0.10 percentage points
                 (lower or unchanged; a hike step is 0.25). Daily values are published the next business day.
    oecd_g20     OECD composite leading indicator, G20 area, amplitude adjusted (OECD SDMX, business_cycle's loader
                 pattern; NOT the US series already used by O2). ABOVE its 10-month average = favorable. Lag 2 (same
                 conservative timing as O2).
    empire       New York Fed Empire State manufacturing survey, general business conditions, seasonally adjusted
                 (business_cycle's loader). ABOVE its 12-month average = favorable. Published around the 15th of the
                 same month; lag 0 (as in business_cycle).
  Excluded on purpose (not to be added): Atlanta Fed / other NY Fed alternatives, real M2 growth (liquidity_lead covers
  M2). Data: FRED fredgraph CSV is tried first with a 10 s timeout and 2 retries; once FRED fails it is skipped for the
  rest of the run and each member comes from its primary publisher's key-less file (DOL weekly claims report, Chicago
  Fed CSV, Dallas Fed xlsx, UMich CSV, Census xlsx, Federal Reserve H.15 data download CSV, OECD SDMX, NY Fed CSV);
  a good download is stored in common.STORE and used only if both fail. A member that cannot be fetched is dropped and
  reported, never substituted. Fixed user agent "btc-research-dashboard".
  Rule template for every member (identical to N1 / O2): at each month-end BTC daily close (00:00 UTC, te._daily_close),
  hold BTC during the next month only if ma50 is on AND the member's latest value already published by then is
  favorable (forward-filled: the latest published reading counts).
  Per member: favorable-minus-unfavorable average next-month BTC return (business_cycle's event table, holding months
  from 2011-07; overall and 2011-16 / 2017-21 / 2022-now); fair exam via business_cycle._Exam (main_signal_check's four
  checks, 200 ordinary-person simulations per clock); ranking score vs ma50's; median deepest loss over the textbook
  cells vs ma50's. A member MEETS THE BAR when ranking score >= ma50's (business_cycle._score_ge, incl. the 95% loss
  eligibility) AND median deepest loss no worse than ma50's.
  Family test (MAIN RESULT): K = number of members meeting the bar. Luck benchmark: the monthly decision flags of ALL
  members (decision months 2010-07 ... now) are circularly shifted TOGETHER by the same random offset of 12 ...
  (length - 12) months (this keeps their cross-correlation), 200 draws, K recomputed each time. To fit the runtime the
  ordinary-person part uses 25 simulations per clock in this step and the real family is re-scored the same way
  (K_real_reduced); K at full simulations is also reported. Family passes if share(K_shift >= K_real_reduced) <= 10%.
  Composite fixed in advance: E1 = ma50 AND at least half of the members available that month are favorable.
  E1 passes if ranking score >= ma50's AND median deepest loss no worse AND placebo share <= 10% (same joint shift,
  same 25 simulations, share of draws whose E1 ranks at least as high as the real E1) AND favorable > unfavorable next-
  month BTC return in all three periods.
  Report only: agreement of each member's monthly flag with N1's NFCI flag and O2's OECD-US flag (share of months with
  the same state, the share expected if independent, phi correlation; decision months from 2011-06); calendar-year
  returns of E1 vs ma50 from 2015 (where the edge comes from, cost in strong years); look-ahead check (values not yet
  usable after three month-ends scrambled; no flag decided up to then may change). Today: states only, with the row
  "比特幣：50 日均線｜E1".
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
import requests

from dashboard import common
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc

DRAWS, GAP, FAM_SIMS = 200, 12, 25
WIN0 = pd.Period("2010-07", "M")       # first decision month (BTC daily closes start 2010-08)
AGREE_FROM = pd.Period("2011-06", "M")
FED_TOL = 0.10
FIRST = msc.DAILY_STARTS[0]
DAY = pd.Timedelta(days=1)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
DOL_URL = "https://oui.doleta.gov/unemploy/wkclaims/report.asp"
DOL_PAGE = "https://oui.doleta.gov/unemploy/claims.asp"
CFNAI_URL = "https://api.data.chicagofed.org/CFNAI/cfnai-data-series-csv.csv"
WEI_URL = "https://www.dallasfed.org/-/media/documents/research/wei/weekly-economic-index.xlsx"
UMICH_URL = "https://www.sca.isr.umich.edu/files/tbmics.csv"
PERMIT_URL = "https://www.census.gov/construction/nrc/xls/permits_cust.xlsx"
H15_URL = ("https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=c5025f4bbbed155a6f17c587772ed69e"
           "&lastobs=&from=01/01/2000&to=12/31/2099&filetype=csv&label=include&layout=seriescolumn")
OECD_G20_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,/G20.M.LI...AA...H"
                "?format=csvfilewithlabels")

_pct, _pp, _d, _num, _p, _share, _ok = bc._pct, bc._pp, bc._d, bc._num, bc._p, bc._share, bc._ok
_RUN: dict = {"fred_down": None, "dol": None}


# ---------- data: helpers ----------

def _get(url: str, timeout=(10, 60), tries: int = 2, data: dict | None = None) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            if data is None:
                reply = requests.get(url, headers=common.HEADERS, timeout=timeout)
            else:
                reply = requests.post(url, data=data, headers=common.HEADERS, timeout=timeout)
            reply.raise_for_status()
            time.sleep(common.PAUSE)
            return reply
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 * (i + 1))
    raise last


def _xlsx_sheets(content: bytes) -> list[list[dict]]:
    """Minimal xlsx reader (no openpyxl): every worksheet as a list of {column letter: text} rows."""
    z = zipfile.ZipFile(io.BytesIO(content))
    names = z.namelist()
    ss = []
    if "xl/sharedStrings.xml" in names:
        raw = z.read("xl/sharedStrings.xml").decode("utf-8")
        ss = [re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<si>(.*?)</si>", raw, re.S)]
    out = []
    for n in sorted(x for x in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", x)):
        t = z.read(n).decode("utf-8")
        rows = []
        for row in re.findall(r"<row [^>]*>(.*?)</row>", t, re.S):
            cells = {}
            for ref, attr, v in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(?:<f>.*?</f>)?<v>(.*?)</v></c>)', row):
                cells[ref] = ss[int(v)] if 't="s"' in attr and v else v
            rows.append(cells)
        out.append(rows)
    return out


def _fnum(text) -> float:
    try:
        return float(str(text).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return float("nan")


def _fred(sid: str, monthly: bool) -> dict:
    frame = pd.read_csv(io.StringIO(_get(FRED_CSV.format(sid), timeout=10, tries=3).text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    n = 7 if monthly else 10
    return {str(d)[:n]: float(v) for d, v in zip(frame.iloc[:, 0], values) if math.isfinite(v)}


# ---------- data: publishers ----------

def _dol() -> dict:
    if _RUN["dol"] is None:
        year = pd.Timestamp(common.today_utc()).year
        text = _get(DOL_URL, data={"level": "us", "final_yr": str(year + 1), "strtdate": "2000", "enddate": str(year),
                                   "filetype": "xml", "submit": "Submit"}, timeout=(10, 120)).text
        init, cont = {}, {}
        for blk in re.findall(r"<week>(.*?)</week>", text, re.S):
            when = re.search(r"<weekEnded>(.*?)</weekEnded>", blk)
            ic = re.search(r"<InitialClaims>.*?<SA>(.*?)</SA>", blk, re.S)
            cc = re.search(r"<ContinuedClaims>.*?<SA>(.*?)</SA>", blk, re.S)
            if not when:
                continue
            day = pd.to_datetime(when.group(1), format="%m/%d/%Y").strftime("%Y-%m-%d")
            for store, m in ((init, ic), (cont, cc)):
                v = _fnum(m.group(1)) if m else float("nan")
                if math.isfinite(v):
                    store[day] = v
        _RUN["dol"] = (init, cont)
    return _RUN["dol"]


def _pub_cfnai() -> dict:
    frame = pd.read_csv(io.StringIO(_get(CFNAI_URL).text))
    s = pd.to_numeric(frame["CFNAI_MA3"], errors="coerce")
    return {str(d).replace("/", "-")[:7]: float(v) for d, v in zip(frame["Date"], s) if math.isfinite(v)}


def _pub_wei() -> dict:
    best = {}
    for rows in _xlsx_sheets(_get(WEI_URL).content):
        if not rows or rows[0].get("A") != "Date" or rows[0].get("B") != "WEI":
            continue
        got = {}
        for r in rows[1:]:
            try:
                day = pd.to_datetime(r.get("A", ""), format="%m/%d/%Y").strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                continue
            v = _fnum(r.get("B"))
            if math.isfinite(v):
                got[day] = v
        if len(got) > len(best):
            best = got
    return best


def _pub_umich() -> dict:
    frame = pd.read_csv(io.StringIO(_get(UMICH_URL).text))
    when = pd.to_datetime(frame["Month"].astype(str).str.strip() + " " + frame["YYYY"].astype(str), format="%B %Y")
    s = pd.to_numeric(frame["ICS_ALL"], errors="coerce")
    return {d.strftime("%Y-%m"): float(v) for d, v in zip(when, s) if math.isfinite(v)}


def _pub_permits() -> dict:
    sheets = _xlsx_sheets(_get(PERMIT_URL).content)
    for rows in sheets:
        head = " ".join(str(r.get("A", "")) for r in rows[:4])
        if "Seasonally adjusted" not in head:
            continue
        got = {}
        for r in rows:
            a = str(r.get("A", ""))
            if not a.isdigit():
                continue
            v = _fnum(r.get("B"))
            if math.isfinite(v):
                got[(pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(a))).strftime("%Y-%m")] = v
        return got
    raise RuntimeError("找不到季調工作表")


def _pub_h15() -> dict:
    lines = _get(H15_URL).text.splitlines()
    got = {}
    for ln in lines:
        parts = ln.split(",")
        if len(parts) == 2 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0]):
            v = _fnum(parts[1])
            if math.isfinite(v):
                got[parts[0]] = v
    return got


def _pub_oecd_g20() -> dict:
    frame = pd.read_csv(io.StringIO(_get(OECD_G20_URL).text))
    s = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    return {str(d)[:7]: float(v) for d, v in zip(frame["TIME_PERIOD"], s) if math.isfinite(v)}


# key, short name, plain explanation, freq (w/m/d), sign (+1 above = favorable, -1 below), window, lag (months) or
# safe days, FRED id, publisher loader, publisher name, publisher URL, timing text, favorable text
MEMBERS = [
    dict(key="claims_init", short="初領失業金", name="每週初次申請失業救濟人數（4 週平均）",
         plain="每週新去申請失業補助的人數，越少代表裁員越少", freq="w", sign=-1, win=52, safe=6, fred="ICSA",
         pub=lambda: _dol()[0], pubname="美國勞工部每週失業救濟報告", url=DOL_PAGE,
         timing="週六結束的那週，下週四公布；至少 6 天前的才用", fav="4 週平均低於最近 52 週平均"),
    dict(key="claims_cont", short="續領失業金", name="持續領失業救濟人數",
         plain="還在領失業補助的人數，越少代表失業的人比較快找到工作", freq="w", sign=-1, win=52, safe=13, fred="CCSA",
         pub=lambda: _dol()[1], pubname="美國勞工部每週失業救濟報告", url=DOL_PAGE,
         timing="比初領晚一週公布；至少 13 天前的才用", fav="低於最近 52 週平均"),
    dict(key="cfnai", short="芝加哥活動指數", name="芝加哥聯準銀行全國活動指數（3 個月平均）",
         plain="把 85 個生產、就業、消費、房市數字合成一個美國經濟活動分數", freq="m", sign=1, win=12, lag=1,
         fred="CFNAIMA3", pub=_pub_cfnai, pubname="芝加哥聯準銀行 CFNAI CSV", url=CFNAI_URL,
         timing="下個月第 3–4 週公布；月底用上個月的數字", fav="高於最近 12 個月平均"),
    dict(key="wei", short="每週經濟指數", name="每週經濟指數（WEI）",
         plain="用每週的零售、用電、鐵路運輸、失業金等 10 個數字估計美國經濟成長，單位接近 GDP 年增率", freq="w",
         sign=1, win=52, safe=6, fred="WEI", pub=_pub_wei, pubname="達拉斯聯準銀行 WEI 檔案（2023 年起由它發布）",
         url=WEI_URL, timing="週六結束的那週，下週四公布；至少 6 天前的才用", fav="高於最近 52 週平均"),
    dict(key="umich", short="消費者信心", name="密西根大學消費者信心指數",
         plain="每月訪問美國家庭對收入、景氣的感覺，越高越樂觀", freq="m", sign=1, win=12, lag=1, fred="UMCSENT",
         pub=_pub_umich, pubname="密西根大學消費者調查 CSV", url=UMICH_URL,
         timing="當月底公布定案值；保守地月底用上個月的數字", fav="高於最近 12 個月平均"),
    dict(key="permits", short="建築許可", name="新屋建築許可（全美，季調年率）",
         plain="政府核准要蓋的新住宅數量，房市和建築業的前導", freq="m", sign=1, win=12, lag=1, fred="PERMIT",
         pub=_pub_permits, pubname="美國普查局新屋建築資料 xlsx", url=PERMIT_URL,
         timing="下個月中公布；月底用上個月的數字", fav="高於最近 12 個月平均"),
    dict(key="fedfunds", short="聯準會沒升息", name="聯邦基金有效利率（是否在升息）",
         plain="銀行之間隔夜借錢的利率，聯準會升息、降息就是在調它", freq="d", fred="DFF", pub=_pub_h15,
         pubname="聯準會 H.15 資料下載 CSV", url=H15_URL,
         timing="每天的數字隔一個營業日公布；用月底前一天為止 30 天的平均",
         fav=f"30 天平均不高於 6 個月前的 30 天平均 + {FED_TOL:.2f} 個百分點（降息或不動）"),
    dict(key="oecd_g20", short="OECD G20 領先", name="OECD 綜合領先指標（G20 地區）",
         plain="OECD 把各國訂單、信心、股價、利差等合成的「未來幾個月景氣方向」指標，這裡是 G20 合計，不是美國單獨",
         freq="m", sign=1, win=10, lag=2, fred=None, pub=_pub_oecd_g20, pubname="OECD SDMX 資料庫", url=OECD_G20_URL,
         timing="約下個月 10 日公布；保守地月底用兩個月前的數字", fav="高於最近 10 個月平均"),
    dict(key="empire", short="紐約製造業調查", name="紐約州帝國製造業調查（目前景氣，季調）",
         plain="紐約聯準銀行每月問紐約州工廠「景氣比上個月好還是差」，好的家數減差的家數", freq="m", sign=1, win=12,
         lag=0, fred="GACDISA066MSFRBNY", pub=bc._pub_empire, pubname="紐約聯準銀行 ESMS CSV", url=bc.EMPIRE_URL,
         timing="當月 15 日左右公布；月底用當月數字", fav="高於最近 12 個月平均"),
]


def _series(m: dict) -> tuple[pd.Series, str, str]:
    path = common.STORE / f"economy_family_{m['key']}.json"
    monthly = m["freq"] == "m"
    need = 120 if monthly else 400
    note, where, days = "", "", None
    try:
        if m["fred"] and not _RUN["fred_down"]:
            try:
                days, where = _fred(m["fred"], monthly), "FRED"
                if len(days) < need:
                    raise RuntimeError(f"只回傳 {len(days)} 筆")
            except Exception as exc:  # noqa: BLE001 - FRED usually times out from here; publishers carry the same data
                _RUN["fred_down"] = type(exc).__name__
                days = None
        if days is None:
            days, where = m["pub"](), m["pubname"]
        if len(days) < need:
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
        note = f"{m['name']}這次抓不到（{type(exc).__name__}），改用上次存下來的資料（到 {max(days)}）。"
    if monthly:
        s = pd.Series({pd.Period(k, "M"): float(v) for k, v in days.items()}).sort_index()
        s = s.reindex(pd.period_range(s.index[0], s.index[-1], freq="M"))
    else:
        s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s, where, note


def _load() -> dict:
    problems, series, where, dropped = [], {}, {}, []
    for m in MEMBERS:
        try:
            s, w, n = _series(m)
            series[m["key"]], where[m["key"]] = s, w
            if n:
                problems.append(n)
        except Exception as exc:  # noqa: BLE001 - a member that cannot be fetched is dropped, never substituted
            dropped.append(m["key"])
            problems.append(f"{m['name']}這次取不到（{type(exc).__name__}），而且沒有存檔，這個成員被拿掉、沒有用別的指標代替。")
    # N1 / O2 reference flags (agreement only)
    ref = {}
    try:
        from dashboard.studies2 import nfci_filter as nf
        ref["nfci"] = nf._series()[0]
    except Exception as exc:  # noqa: BLE001
        problems.append(f"NFCI（N1 用的）這次取不到（{type(exc).__name__}），沒有算和 N1 的一致程度。")
    try:
        if _RUN["fred_down"]:
            bc._RUN["fred_down"] = _RUN["fred_down"]
        ref["oecd_us"] = bc._series("oecd")[0]
    except Exception as exc:  # noqa: BLE001
        problems.append(f"OECD 美國領先指標（O2 用的）這次取不到（{type(exc).__name__}），沒有算和 O2 的一致程度。")
    other, more = msc._load_other()
    return {"close": bc.te._daily_close(), "series": series, "where": where, "dropped": dropped, "ref": ref,
            "other": other, "problems": problems + more, "fred_down": _RUN["fred_down"]}


# ---------- the signal ----------

def _member_flags(m: dict, s: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Decision month M -> 1 favorable / 0 unfavorable / NaN unknown, from the latest value published by then."""
    last_day = months.to_timestamp(how="end").normalize()
    if m["freq"] == "m":
        avg = s.rolling(m["win"], min_periods=m["win"]).mean()
        fav = (m["sign"] * (s - avg) > 0).astype(float).where(avg.notna() & s.notna())
        fav.index = fav.index + m["lag"]
        fav = fav.dropna()
        pos = np.searchsorted(fav.index.asi8, months.asi8, side="right") - 1
        vals = np.where(pos >= 0, fav.values[np.clip(pos, 0, None)], np.nan)
        return pd.Series(vals, index=months)
    if m["freq"] == "w":
        x = s.rolling(4, min_periods=4).mean() if m["key"] == "claims_init" else s
        avg = s.rolling(m["win"], min_periods=m["win"]).mean()
        fav = (m["sign"] * (x - avg) > 0).astype(float).where(avg.notna() & x.notna()).dropna()
        cut = last_day - pd.Timedelta(days=m["safe"])
        pos = np.searchsorted(fav.index.values, cut.values, side="right") - 1
        vals = np.where(pos >= 0, fav.values[np.clip(pos, 0, None)], np.nan)
        return pd.Series(vals, index=months)
    # daily fed funds: 30-day average ending the day before the decision day vs the same six months earlier
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D")).ffill()
    avg = full.rolling(30, min_periods=20).mean()
    end_now = last_day - DAY
    end_then = (months - 6).to_timestamp(how="end").normalize() - DAY
    a_now = avg.reindex(end_now).values
    # a decision day after the last published value uses the latest available average
    late = end_now > avg.index[-1]
    a_now = np.where(late, avg.iloc[-1] if len(avg) else np.nan, a_now)
    a_then = avg.reindex(end_then).values
    ok = np.isfinite(a_now) & np.isfinite(a_then)
    vals = np.where(ok, (a_now <= a_then + FED_TOL).astype(float), np.nan)
    return pd.Series(vals, index=months)


def _composite(mat: np.ndarray) -> np.ndarray:
    """E1 macro part: at least half of the members available that month are favorable."""
    avail = np.isfinite(mat).sum(axis=1)
    fav = np.nansum(mat, axis=1)
    return np.where(avail > 0, (2 * fav >= avail).astype(float), np.nan)


# ---------- look-ahead ----------

def _lookahead(members: list[dict], series: dict, months: pd.PeriodIndex) -> tuple[int, int]:
    rng = np.random.default_rng(13)
    total = bad = 0
    for cut in ("2016-06-30", "2020-03-31", "2024-09-30"):
        c = pd.Timestamp(cut)
        cp = c.to_period("M")
        upto = months <= cp
        for m in members:
            s = series[m["key"]]
            if m["freq"] == "m":
                known = s.index <= cp - m["lag"]
            elif m["freq"] == "w":
                known = s.index + pd.Timedelta(days=m["safe"]) <= c
            else:
                known = s.index + DAY <= c
            sd = float(np.nanstd(s.values)) or 1.0
            s2 = s.where(known, s + rng.normal(0, sd, len(s)))
            a = _member_flags(m, s, months)[upto].values
            b = _member_flags(m, s2, months)[upto].values
            total += 1
            bad += int(not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)))
    return total, bad


# ---------- agreement ----------

def _agree(a: pd.Series, b: pd.Series) -> dict | None:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    both = both[both.index >= AGREE_FROM]
    if len(both) < 24:
        return None
    x, y = both["a"].values, both["b"].values
    same = float((x == y).mean())
    pa, pb = float(x.mean()), float(y.mean())
    expect = pa * pb + (1 - pa) * (1 - pb)
    phi = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {"same": same, "expect": expect, "phi": phi, "n": len(both)}


# ---------- the computation ----------

def _meets(x: dict, base: dict) -> bool:
    return bool(bc._score_ge(x, base) and x["c_deep_med"] >= base["c_deep_med"] - 1e-12)


def _compute(data: dict) -> dict:
    t0 = time.time()
    close, series = data["close"], data["series"]
    members = [m for m in MEMBERS if m["key"] in series]
    cur = (close.index[-1] + DAY).to_period("M") - 1
    months = pd.period_range(WIN0, cur, freq="M")
    F = {m["key"]: _member_flags(m, series[m["key"]], months) for m in members}
    keys = [m["key"] for m in members]
    mat = np.column_stack([F[k].values for k in keys])
    E = pd.Series(_composite(mat), index=months)

    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    ex = bc._Exam(data, price, b0, pre)

    rng = np.random.default_rng(bc.SEED + 5)
    ev = {k: bc._events(close, F[k], rng) for k in keys}
    ev["e1"] = bc._events(close, E, rng)
    s = {"ma50": ex.summary("ma50", None)}
    for k in keys:
        s[k] = ex.summary("m2", F[k])
    s["e1"] = ex.summary("m2", E)
    meets = {k: _meets(s[k], s["ma50"]) for k in keys}

    # family luck benchmark: joint circular shift of all members' monthly flags
    small = {"ma50": ex.summary("ma50", None, FAM_SIMS)}
    for k in keys + ["e1"]:
        small[k] = ex.summary("m2", F[k] if k != "e1" else E, FAM_SIMS)
    meets_small = {k: _meets(small[k], small["ma50"]) for k in keys}
    k_real_small = sum(meets_small.values())
    prng = np.random.default_rng(bc.SEED + 77)
    n = len(months)
    k_shift, e_hits, e_beat_ma, e_lows, member_hits = [], 0, 0, [], {k: 0 for k in keys}
    for sh in prng.integers(GAP, n - GAP + 1, DRAWS):
        m2 = np.roll(mat, int(sh), axis=0)
        kk = 0
        for j, k in enumerate(keys):
            x = ex.summary("m2", pd.Series(m2[:, j], index=months), FAM_SIMS)
            ok = _meets(x, small["ma50"])
            kk += int(ok)
            member_hits[k] += int(ok)
        k_shift.append(kk)
        xe = ex.summary("m2", pd.Series(_composite(m2), index=months), FAM_SIMS)
        e_lows.append(xe["low"])
        e_hits += int(bc._score_ge(xe, small["e1"]))
        e_beat_ma += int(_meets(xe, small["ma50"]))
    k_shift = np.array(k_shift)

    pos = {"ma50": ex.btc_pos("ma50", None), "e1": ex.btc_pos("m2", E)}
    a = close.index >= pd.Timestamp(FIRST)
    p0, p1 = np.nan_to_num(pos["ma50"][a]), np.nan_to_num(pos["e1"][a])
    differ = (int((p0 != p1).sum()), int(a.sum()), int(((p0 > 0.5) & (p1 < 0.5)).sum()))
    years = {"hold": bc._years(np.ones(len(close)), close, FIRST),
             **{k: bc._years(pos[k], close, FIRST) for k in pos}}

    # agreement with N1 and O2
    ref = {}
    if "nfci" in data["ref"]:
        from dashboard.studies2 import nfci_filter as nf
        ref["N1"] = nf._flags(data["ref"]["nfci"], "avg52", months)
    if "oecd_us" in data["ref"]:
        ref["O2"] = bc._flags(data["ref"]["oecd_us"], bc.LAG["oecd"]).reindex(months)
    FE = {**F, "e1": E}
    agree = {k: {r: _agree(FE[k], ref[r]) for r in ref} for k in keys + ["e1"]}
    if "N1" in ref and "O2" in ref:
        agree["n1o2"] = _agree(ref["N1"], ref["O2"])

    look = _lookahead(members, series, months)
    return {"ex": ex, "members": members, "keys": keys, "months": months, "F": FE, "mat": mat, "ev": ev, "s": s,
            "meets": meets, "small": small, "meets_small": meets_small, "k_real": sum(meets.values()),
            "k_real_small": k_real_small, "k_shift": k_shift, "member_hits": member_hits,
            "e_plc": e_hits / DRAWS, "e_plc_ma": e_beat_ma / DRAWS, "e_plc_low": float(np.median(e_lows)),
            "pos": pos, "differ": differ, "years": years, "agree": agree, "ref": list(ref), "look": look,
            "problem": problem, "cur": cur, "secs": time.time() - t0}


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    data = _load()
    if len(data["series"]) < 3:
        raise RuntimeError("能抓到的成員少於 3 個，無法做家族檢定")
    r = _compute(data)
    close, ex, s, ev, F = data["close"], r["ex"], r["s"], r["ev"], r["F"]
    members, keys = r["members"], r["keys"]
    problems = data["problems"] + ([r["problem"]] if r["problem"] else [])
    base, e1 = s["ma50"], s["e1"]
    n_c, n_d = base["c_n"], base["d_n"]
    nm = len(keys)

    def yes(v: bool) -> str:
        return "過" if v else "不過"

    # family test
    ks = r["k_shift"]
    luck = float((ks >= r["k_real_small"]).mean())
    fam_pass = luck <= 0.10
    # composite test
    qe = {"event": ev["e1"]["all_three"], "score": bc._score_ge(e1, base),
          "loss": e1["c_deep_med"] >= base["c_deep_med"] - 1e-12, "plc": r["e_plc"] <= 0.10}
    qe["pass"] = all(qe.values())

    # ----- table 1: members -----
    rows1 = []
    for m in members:
        sr = data["series"][m["key"]]
        first, last = sr.dropna().index[0], sr.dropna().index[-1]
        fmt = (lambda x: x.strftime("%Y-%m")) if m["freq"] == "m" else _d
        rows1.append([m["name"], m["plain"], m["fav"], m["timing"], data["where"][m["key"]], f"{fmt(first)} 到 {fmt(last)}"])
    for k in data["dropped"]:
        mm = next(m for m in MEMBERS if m["key"] == k)
        rows1.append([mm["name"], mm["plain"], mm["fav"], mm["timing"], "抓不到，已拿掉", "—"])
    t1 = {"caption": ("事先寫死的家族成員：每個指標「有利」的方向是先依經濟道理定好、跑之前寫下的，規則完全一樣：每個月最後一根日線"
                      "（UTC 0 點收盤）決定，50 日均線持有而且這個指標當時已經公布的最新數字「有利」，下個月才持有比特幣。"
                      "數字都會被修正，這裡只有今天的版本。"),
          "columns": ["指標", "白話", "什麼算有利", "月底能用哪個數字", "這次的資料來源", "資料期間"], "rows": rows1}

    # ----- table 2: per member -----
    def mrow(lab: str, k: str) -> list:
        e, x = ev[k], s[k]
        a = e["all"]
        return ([lab, f"{_p(a['diff'])}（有利 {a['nf']}｜不利 {a['nu']} 個月）"]
                + [_p(p["diff"]) for p in e["periods"].values()]
                + [f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）"), _pct(x["c_deep_med"]),
                   ("是" if r["meets"][k] else "否") if k in r["meets"] else "—"])
    rows2 = [["50 日均線（基準）", "—", "—", "—", "—", _share(base["low"]), _pct(base["c_deep_med"]), "—"]]
    rows2 += [mrow(m["short"], m["key"]) for m in members]
    rows2.append(mrow("E1：過半成員有利（組合）", "e1"))
    t2 = {"caption": (f"每個成員各自加在 50 日均線上的結果。第二到第五欄是「只看這個指標」：有利月減不利月的下個月比特幣平均報酬"
                      f"（月底收盤到月底收盤，持有月份 2011-07 起），以及分三段的差距。排名分數＝main_signal_check 公平考試四項"
                      f"（比特幣 {n_d} 個起算月、{n_c:,} 格起算月×收盤時間、同樣格數照一般人做法每格 {bc.SIMS} 次、其他資產 ETH、SOL、BNB、XRP、MSTR）"
                      "裡最低的那項；最深虧損中位數是照課本做的那些格。「達標」＝排名分數不低於均線，而且最深虧損中位數不比均線深。"),
          "columns": ["規則", "有利減不利（全部）"] + [f"{lab}" for lab, _, _ in bc.PERIODS]
                     + ["排名分數", "最深虧損中位數", "達標"],
          "rows": rows2}

    # ----- table 3: family test -----
    hist = np.bincount(ks, minlength=nm + 1)
    rows3 = [["真的家族：達標成員數 K（完整模擬）", f"{r['k_real']} 個／{nm} 個",
              "、".join(m["short"] for m in members if r["meets"][m["key"]]) or "沒有"],
             [f"真的家族：K（簡化版，一般人做法每格 {FAM_SIMS} 次，和錯位用同樣算法）", f"{r['k_real_small']} 個／{nm} 個",
              "、".join(m["short"] for m in members if r["meets_small"][m["key"]]) or "沒有"],
             [f"隨機錯位 {DRAWS} 次的 K", f"中位數 {float(np.median(ks)):.0f}，平均 {ks.mean():.1f}",
              "；".join(f"K={i}：{hist[i]} 次" for i in range(nm + 1) if hist[i])],
             ["錯位 K ≥ 真的 K 的比例（≤ 10% 才算過）", _share(luck), "通過" if fam_pass else "沒有通過"]]
    rows3 += [[f"　{m['short']}：錯位後也達標的比例", _share(r["member_hits"][m["key"]] / DRAWS),
               "真的達標" if r["meets_small"][m["key"]] else "真的沒達標"] for m in members]
    t3 = {"caption": (f"主要結果：家族檢定。把全部成員每月的有利／不利整串「一起」循環平移同一個隨機月數（{GAP} 個月到「長度 − {GAP}」個月，"
                      f"決定月份 {r['months'][0]} 到 {r['months'][-1]}，共 {len(r['months'])} 個月），這樣成員之間彼此的相關保留，"
                      "但和比特幣的時間對應被打亂；每次重算有幾個成員達標。如果真的家族達標個數只是運氣，打亂後也常常一樣多。"),
          "columns": ["項目", "數字", "說明"], "rows": rows3}

    # ----- table 4: composite exam -----
    rows4 = []
    for lab, x in (("50 日均線（基準）", base), ("E1：50 日均線而且過半成員有利", e1)):
        ck = x["checks"]
        rows4.append([lab, _share(ck[0]), _share(ck[1]), _share(ck[2]), f"{_share(ck[3])}（{x['o_n']} 個）",
                      f"{_share(x['low'])}" + ("" if x["eligible"] else "（虧損條件未達 95%）") + f"｜{_share(x['loss_ok'])}",
                      f"{_pct(x['c_deep_med'])}｜{_pct(x['p_deep_med'])}", f"{_pp(x['c_edge'])}｜{_pp(x['p_edge'])}"])
    rows4.append(["對照：一直抱著不動", "—", "—", "—", "—", "—", f"{_pct(float(np.median(ex.chold[1])))}｜—", "—"])
    t4 = {"caption": (f"組合 E1 的完整公平考試（main_signal_check 四項：比特幣 UTC 0 點日線 {n_d} 個起算月；{len(ex.cstarts)} 個起算月 × 24 個收盤時間"
                      f" ＝ {n_c:,} 格照課本做，成本 {bc.COST * 100:.2f}%；同樣格數照一般人做（收盤後 10 小時才做、成本 {bc.P_COST * 100:.2f}%、"
                      f"一成進場漏掉，每格 {bc.SIMS} 次）；其他資產用自己的 50 日均線、同一個美國經濟訊號）。排名分數＝四項最低的那項，"
                      "前提是最深虧損比一直抱著淺的比例至少 95%。"),
          "columns": ["規則", f"比特幣起算月（{n_d} 個）", f"起算月×收盤時間（{n_c:,} 格）", "照一般人方式做", "其他資產起算月",
                      "排名分數｜虧損比抱著淺", "最深虧損中位數：課本｜一般人", "每年多賺中位數：課本｜一般人"],
          "rows": rows4}

    pe = [p["diff"] for p in ev["e1"]["periods"].values()]
    rows5 = [["有利月比不利月好（三段）", "、".join(_p(v) for v in pe), yes(qe["event"])],
             ["排名分數不低於 50 日均線", f"{_share(e1['low'])} 對 {_share(base['low'])}", yes(qe["score"])],
             [f"{n_c:,} 格最深虧損中位數不比均線深", f"{_pct(e1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}", yes(qe["loss"])],
             ["隨機錯位一樣好或更好的比例 ≤ 10%",
              f"{_share(r['e_plc'])}（錯位後排名分數中位數 {_share(r['e_plc_low'])}，真的 {_share(r['small']['e1']['low'])}；"
              f"錯位後也達標的 {_share(r['e_plc_ma'])}）", yes(qe["plc"])],
             ["結果", "—", "通過" if qe["pass"] else "沒有通過"]]
    t5 = {"caption": ("E1 事先寫好的過關標準，四項都要過。隨機錯位和家族檢定是同一批平移（全部成員一起移），"
                      f"一般人做法每格 {FAM_SIMS} 次，真的 E1 也用同樣方式算一次再比。"),
          "columns": ["條件", "數字", "結果"], "rows": rows5}

    # ----- table 6: agreement -----
    refs = r["ref"]
    rows6 = []

    def ag(v) -> str:
        return f"{_share(v['same'])}（獨立時約 {_share(v['expect'])}；相關 {v['phi']:+.2f}）" if v else "—"
    for m in members + [{"key": "e1", "short": "E1 組合"}]:
        rows6.append([m["short"]] + [ag(r["agree"][m["key"]].get(x)) for x in ("N1", "O2")])
    if "n1o2" in r["agree"]:
        rows6.append(["對照：N1 和 O2 彼此", ag(r["agree"]["n1o2"]), "—"])
    t6 = {"caption": (f"每個成員的每月有利／不利，和 N1（金融情勢指數 NFCI 低於 52 週平均）、O2（OECD 美國領先指標高於 10 個月平均）"
                      f"同一個月狀態相同的比例，決定月份 {AGREE_FROM} 起。括號是「如果兩者毫無關係」預期的相同比例，和相關係數"
                      "（+1 完全同步、0 無關、−1 相反）。相同比例明顯高於獨立時的預期，代表其實是同一個訊號的不同說法。"),
          "columns": ["成員", "和 N1 相同", "和 O2 相同"], "rows": rows6}

    # ----- table 7: years -----
    dd_, nn, out_ = r["differ"]
    yrs = sorted(r["years"]["ma50"])
    rows7 = []
    for y in yrs:
        a, b, h = r["years"]["ma50"].get(y), r["years"]["e1"].get(y), r["years"]["hold"].get(y)
        rows7.append([f"{y}" + ("（到目前）" if y == yrs[-1] else ""), _pct(h), _pct(a), _pct(b),
                      _pp(b - a) if _ok(a) and _ok(b) else "—"])
    rows7.append([f"和 50 日均線持有狀態不同的天數（{FIRST} 起）", "—", "—", f"{dd_:,} 天／{nn:,} 天（{dd_ / nn * 100:.1f}%）" if nn else "—",
                  f"其中均線持有、E1 現金 {out_:,} 天"])
    t7 = {"caption": (f"UTC 0 點日線照課本做（成本 0.10%）每個日曆年的報酬，{FIRST[:4]} 年起，看 E1 和 50 日均線的差來自哪幾年；"
                      "E1 只會比均線少持有（經濟偏弱時，均線持有也改成現金）。"),
          "columns": ["年份", "一直抱著", "50 日均線", "E1", "E1 減均線（百分點）"], "rows": rows7}

    # ----- today -----
    last = close.index[-1]
    cur = r["cur"]
    pm = r["pos"]

    def word(v) -> str:
        return "持有" if _ok(v) and v >= 0.5 else "現金"

    now = [{"label": f"最後一根收完的日線（{_d(last)}，UTC 0 點收盤）",
            "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元；50 日均線 {float(close.iloc[-50:].mean()):,.0f}"}]
    fav_now = []
    for m in members:
        v = F[m["key"]].get(cur, np.nan)
        if v == 1:
            fav_now.append(m["short"])
        now.append({"label": f"{m['short']}（{cur.strftime('%Y-%m')} 月底的決定，管 {(cur + 1).strftime('%Y-%m')}）",
                    "value": "有利" if v == 1 else "不利" if v == 0 else "—"})
    avail_now = sum(1 for m in members if _ok(F[m["key"]].get(cur, np.nan)))
    now.append({"label": "有利的成員數", "value": f"{len(fav_now)} 個／{avail_now} 個（過半才算 E1 的經濟條件成立）"})
    now.append({"label": "比特幣：50 日均線｜E1", "value": f"{word(pm['ma50'][-1])}｜{word(pm['e1'][-1])}"})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    better = [y for y in yrs if _ok(r["years"]["e1"].get(y)) and _ok(r["years"]["ma50"].get(y))
              and r["years"]["e1"][y] - r["years"]["ma50"][y] > 0.5]
    worse = [y for y in yrs if _ok(r["years"]["e1"].get(y)) and _ok(r["years"]["ma50"].get(y))
             and r["years"]["e1"][y] - r["years"]["ma50"][y] < -0.5]
    strong = [y for y in yrs if _ok(r["years"]["hold"].get(y)) and r["years"]["hold"][y] > 50]
    cost_strong = [(y, r["years"]["e1"][y] - r["years"]["ma50"][y]) for y in strong
                   if _ok(r["years"]["e1"].get(y)) and _ok(r["years"]["ma50"].get(y))]
    hit = [m["short"] for m in members if r["meets"][m["key"]]]
    phis = {x: [v[x]["phi"] for k, v in r["agree"].items() if k in keys and v.get(x) and math.isfinite(v[x]["phi"])]
            for x in refs}
    agree_txt = "；".join(f"和 {x} 的相關中位數 {float(np.median(v)):+.2f}" for x, v in phis.items() if v)
    verdict = (f"家族檢定{'通過' if fam_pass else '沒有通過'}：{nm} 個經濟指標用同一條規則（50 日均線而且指標有利），"
               f"達標（排名分數不輸均線、最深虧損不更深）的有 {r['k_real']} 個" + (f"（{'、'.join(hit)}）" if hit else "")
               + f"；把全部指標一起隨機錯位 {DRAWS} 次，達標個數一樣多或更多的佔 {_share(luck)}"
               f"（簡化版真的 K＝{r['k_real_small']}，錯位中位數 {float(np.median(ks)):.0f}）。"
               f"組合 E1（過半成員有利）{'通過' if qe['pass'] else '沒有通過'}：排名分數 {_share(e1['low'])} 對均線 {_share(base['low'])}"
               f"（{yes(qe['score'])}），最深虧損中位數 {_pct(e1['c_deep_med'])} 對 {_pct(base['c_deep_med'])}（{yes(qe['loss'])}），"
               f"隨機錯位 {_share(r['e_plc'])}（{yes(qe['plc'])}），三段差距 {'、'.join(_p(v) for v in pe)}（{yes(qe['event'])}）。"
               + (f"E1 比均線好的年份：{'、'.join(str(y) for y in better)}" if better else "E1 沒有一年明顯比均線好")
               + (f"；比均線差的年份：{'、'.join(str(y) for y in worse)}。" if worse else "。")
               + (f"比特幣大漲（全年 > 50%）的年份 E1 減均線：{'、'.join(f'{y} {v:+.0f} 個百分點' for y, v in cost_strong)}。" if cost_strong else "")
               + (f"成員和 N1、O2 的關係：{agree_txt}。" if agree_txt else "")
               + "所有經濟數字都會修正，這裡只有今天的版本。")

    tot, bad = r["look"]
    src = "、".join(f"{m['short']}：{data['where'][m['key']]}" for m in members)
    fred_txt = (f"FRED 的公開 CSV 這次沒有回應（{data['fred_down']}，10 秒逾時、重試 2 次），之後的成員都直接用發布機構自己的公開檔案（FRED 也是轉載它們）。"
                if data["fred_down"] else "FRED 的公開 CSV 這次有回應。")
    notes = [
        ("事先寫死的規則和標準（照原文，跑之前寫好）：rule template for every member = hold BTC next month only if ma50 is on "
         "AND the member's latest already-published value is favorable, decided at the month-end daily close. A member meets "
         "the bar when ranking score >= ma50's AND median deepest loss no worse. Family test (main result): K = members "
         f"meeting the bar; all members' monthly flags circularly shifted together by the same offset ({GAP}..length-{GAP} "
         f"months), {DRAWS} draws, ordinary person {FAM_SIMS} simulations per clock (real family re-scored the same way); "
         "family passes if share(K_shift >= K_real) <= 10%. Composite E1 = ma50 AND at least half of the available members "
         "favorable; passes if ranking >= ma50's AND median deepest loss no worse AND placebo (same joint shift) <= 10% AND "
         "favorable > unfavorable in 2011-16, 2017-21, 2022-now."),
        ("為什麼要整個家族一起測：一個一個指標試，總會有幾個剛好看起來有用（多重檢定）。所以這裡把所有成員的方向和規則先寫死，"
         "看的是「整個家族裡有幾個達標」有沒有比運氣多，而不是挑最好的那一個。單一成員就算達標，也不能單獨拿出來當證據。"),
        (f"資料來源：{src}。{fred_txt}"
         "原本列在名單上的 WEI 頁面在紐約聯準銀行，但 2023 年 12 月起改由達拉斯聯準銀行發布，所以從達拉斯聯準銀行下載同一個指數。"
         "失業救濟資料是用勞工部「每週失業救濟」查詢頁的公開表單（全國、XML 格式）取得。"),
        ("修正的問題：失業金、消費者信心、建築許可、活動指數、WEI、OECD 領先指標都會在之後被修正或重新季節調整，"
         "這裡用的都是今天的版本；當時真的看到的數字可能不同，結果可能比當時真的做得到的好看。聯邦基金利率不會修正。"),
        (f"時間對齊：每月最後一根日線決定。每週的指標只用「月底前至少 6 天（續領失業金 13 天）」的那週；每月的指標照表一的落後月數；"
         f"利率用月底前一天為止 30 天的平均。偷看未來的檢查：三個月底之後把那時還不能用的數字換成亂數，切點以前的每月狀態 {tot} 組裡 "
         f"{tot - bad} 組完全沒變" + ("。" if bad == 0 else "（有變動，結果不能用）。")),
        ("其他幣、MSTR 用的是同一個美國經濟訊號。相鄰起算月共用大部分歷史、24 個收盤時間只差幾小時，「幾個裡幾個」不是互相獨立的證據；"
         "成員之間也高度相關（都在描述同一個美國景氣），所以家族檢定才用「一起平移」來保留這個相關。"
         f"成本：照課本 {bc.COST * 100:.2f}%、一般人 {bc.P_COST * 100:.2f}%，現金沒有利息、沒有算稅。隨機的部分都用固定種子。"),
        f"這次計算用了 {time.time() - t0:.0f} 秒（考試和錯位部分 {r['secs']:.0f} 秒）。",
        "過去的結果不保證未來會一樣。這裡只是把事先寫好的規則照完整考試測過去的結果、說明現在的狀態，不是買賣建議。",
    ] + problems

    sources = [{"name": f"{m['name']}：{m['pubname']}", "url": m["url"]} for m in members]
    if data["fred_down"] is None:
        sources += [{"name": f"FRED：{m['fred']}", "url": FRED_CSV.format(m["fred"])} for m in members if m["fred"]]
    else:
        sources.append({"name": "FRED（試過、沒有回應）", "url": FRED_CSV.format("ICSA")})
    sources += [
        {"name": "達拉斯聯準銀行 WEI 頁面", "url": "https://www.dallasfed.org/research/wei"},
        {"name": "芝加哥聯準銀行 NFCI 資料（和 N1 比較用）", "url": "https://api.data.chicagofed.org/NFCI/nfci-data-series-csv.csv"},
        {"name": "OECD 美國綜合領先指標（和 O2 比較用）", "url": bc.OECD_URL},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
    ]

    return {
        "id": "economy_family",
        "title": "一整組美國經濟指標能不能幫 50 日均線",
        "question": ("金融情勢指數（N1）和 OECD 美國領先指標（O2）加上 50 日均線都通過了考試。換成其他美國實體經濟和利率指標"
                     "——失業金、經濟活動指數、每週經濟指數、消費者信心、建築許可、聯準會有沒有在升息、G20 領先指標、紐約製造業調查——"
                     "用完全相同的規則，整組裡達標的數量會比運氣多嗎？把它們合成「過半有利」的組合，能通過完整的公平考試嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": [t1, t2, t3, t4, t5, t6, t7],
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
