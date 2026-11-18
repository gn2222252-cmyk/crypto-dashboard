"""Historical study "macro_days": does bitcoin move in a fixed way around the US central bank's rate meetings (FOMC)
and the monthly US consumer price release (CPI)? Equities: Lucca & Moench, "The Pre-FOMC Announcement Drift"
(Journal of Finance 2015) found large excess returns in the 24 hours before scheduled FOMC announcements. Crypto:
industry charts (Cointelegraph, Interdax) say bitcoin tends to rise into FOMC and fall afterwards; an academic event
study (University of Vaasa thesis) found the effects are not uniform.

Pre-registered tests (written before running, kept verbatim):
Windows relative to the announcement time T (FOMC 14:00 ET; CPI 08:30 ET; use hourly BTC from 2017-08, daily before):
1. Pre-drift: return from T-24h to T-15min (hourly: T-24h to T); Post: T to T+24h, T+24h to T+7d; Pre-week: T-7d to
   T-24h.
2. Compare with the same clock windows on non-event days (placebo: same weekday & hour, weeks without events), mean
   and median, share positive, block-bootstrap 90% interval; count events. Split: 2013-2019, 2020-2022, 2023-2026; by
   decision type (hike/hold/cut - take from the Fed statements/public record) for FOMC as description only.
3. Tradable rules for an ordinary person (one action per day is fine; costs 0.10/0.30%): (R1) hold BTC only from T-24h
   to T (pre-drift capture) else cash - report per-event and annual contribution; (R2) ma50, but step out from T to
   T+7d around FOMC (avoid post-meeting weakness); (R3) ma50, but step out from T-24h to T+24h around CPI. Compare
   R2/R3 with ma50 on the fair grid where feasible and with a placebo moving the same number of windows to random
   non-event dates (200 draws).
Pass test (stated now): the effect has the same sign in all three periods with a 90% interval excluding zero AND the
rule beats >= 90% of placebo draws AND (for R2/R3) does not lower ma50's ranking score.

How the tests are carried out (fixed with the tests, no tuning afterwards):
- Events: scheduled FOMC statements 2013-2026 (all at 14:00 New York time since 2013; verified against the Fed's
  calendar and statement pages; the two unscheduled 2020 cuts are listed but excluded, the cancelled 2020-03-18
  meeting is not an event). CPI release dates (08:30 New York time) from ALFRED's release-date list for release 10
  (it cites BLS; the BLS schedule page refuses the dashboard's honest user agent, so it was not used), with the
  February seasonal-factor-only dates removed; dates after the hard-coded list come from a public schedule page that
  copies the BLS schedule. Four CPI dates moved by government shutdowns are marked.
- New York clock times are converted with zoneinfo, so daylight saving is exact (FOMC = 18:00 or 19:00 UTC).
- Bitcoin hourly: Binance BTCUSDT hourly closes (the shared store of studies2.seasonality, read only, missing hours
  fetched into memory), price at each full hour. FOMC T is a full hour. CPI 08:30 is not: hourly CPI windows are
  anchored at 08:00 New York time (T-30 min), so "T to T+24h" is 08:00 to 08:00 the next day. Before the hourly data
  (events before 2017-08-26) the daily closes at 00:00 UTC are used: pre = the UTC day before the event day, post =
  the UTC day after, pre-week = the six UTC days before that, post T+24h..T+7d = days 2-7 after; the event day
  itself is skipped. "T to T+7d" and "T-24h to T+24h" exist only on hourly data.
- Placebo for each event: the same window at the same New York clock time on the same weekday, 1 to 26 weeks before
  and after, in weeks without an event of that kind and with no FOMC or CPI day within 3 days; same data kind
  (hourly / daily) as the event. Excess = event return minus the average of its placebo windows. 90% interval: moving
  block bootstrap of the excess (blocks of 4 consecutive events, 2,000 draws). "Same sign in all three periods with a
  90% interval excluding zero" is checked as: the average excess has the same sign in each period AND the whole
  sample's 90% interval excludes zero (each period's interval is also shown).
- Rules on the hourly grid from 2018-01-01: ma50 = bitcoin's daily close at 00:00 UTC above its 50-day average,
  decided at that close and held for the next 24 hours. R1 holds only in [T-24h, T) of each FOMC. R2 = ma50 except
  [T, T+7d) of each FOMC. R3 = ma50 except [08:00 New York the day before, 08:00 New York the day after) of each CPI.
  Cost = 0.10% (and 0.30%) of the amount moved at each change. Placebo: every event window moved to one of its own
  placebo dates at random, 200 draws, fixed seed; R1 is compared by total return, R2/R3 by return / volatility.
- Fair grid: 24 daily closing hours x 72 monthly start dates (2018-01 ... 2023-12), each measured to the last finished
  day. ma50's ranking score (as in main_signal_check) = share of cells that beat holding and the median yearly edge
  over holding; R2/R3 "do not lower" it when both are at least as high as ma50's.
- Nasdaq (QQQ, Yahoo daily adjusted closes; no intraday data): FOMC day close-to-close, the day before, and the next
  five trading days, against the same weekday in other weeks.
The decision type is only known after T and is used for description only. Descriptive and in-sample; not advice.
Only public, key-less sources with the dashboard's fixed user agent; a source that refuses it is not worked around.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import trend_ensemble as te

NY = ZoneInfo("America/New_York")
TW = ZoneInfo("Asia/Taipei")
UTC = timezone.utc
COST, COST_HIGH = 0.001, 0.003
SIMS = 200
BOOT = 2000
BLOCK = 4
SEED = 20130130
WEEKS = 26
CLEAR_DAYS = 3
YEAR_H = 365 * 24
RULE_START = "2018-01-01"
GRID_STARTS = ("2018-01-01", "2023-12-01")
PERIODS = (("2013–2019", 2013, 2019), ("2020–2022", 2020, 2022), ("2023–2026", 2023, 2026))
FED_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FED_HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
ALFRED_CPI = "https://alfred.stlouisfed.org/release/downloaddates?rid=10"
CPI_PAGE = "https://www.usinflationcalculator.com/inflation/consumer-price-index-release-schedule/"
BLS_CPI = "https://www.bls.gov/schedule/news_release/cpi.htm"

# Scheduled FOMC statements, day of the statement (MMDD) and decision: u = raise, h = keep, d = lower. Dates from the
# Fed's calendar pages (fomccalendars.htm, fomchistorical20XX.htm); decisions read from each statement
# (federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm), all released at 2:00 p.m.
FOMC_TEXT = {
    2013: "0130h 0320h 0501h 0619h 0731h 0918h 1030h 1218h",
    2014: "0129h 0319h 0430h 0618h 0730h 0917h 1029h 1217h",
    2015: "0128h 0318h 0429h 0617h 0729h 0917h 1028h 1216u",
    2016: "0127h 0316h 0427h 0615h 0727h 0921h 1102h 1214u",
    2017: "0201h 0315u 0503h 0614u 0726h 0920h 1101h 1213u",
    2018: "0131h 0321u 0502h 0613u 0801h 0926u 1108h 1219u",
    2019: "0130h 0320h 0501h 0619h 0731d 0918d 1030d 1211h",
    2020: "0129h 0429h 0610h 0729h 0916h 1105h 1216h",
    2021: "0127h 0317h 0428h 0616h 0728h 0922h 1103h 1215h",
    2022: "0126h 0316u 0504u 0615u 0727u 0921u 1102u 1214u",
    2023: "0201u 0322u 0503u 0614h 0726u 0920h 1101h 1213h",
    2024: "0131h 0320h 0501h 0612h 0731h 0918d 1107d 1218d",
    2025: "0129h 0319h 0507h 0618h 0730h 0917d 1029d 1210d",
    2026: "0128h 0318h 0429h 0617h 0729h 0916u 1028h 1209?",
}
UNSCHEDULED = (("2020-03-03", 10, 0, "d"), ("2020-03-15", 17, 0, "d"))  # excluded from the tests
CANCELLED = "2020-03-18"  # the scheduled March 17-18, 2020 meeting was cancelled (replaced by March 15)
DECISION = {"u": "升息", "h": "利率不動", "d": "降息", "?": "尚未開會"}

# CPI release days (08:30 New York time). From ALFRED's release dates for "Consumer Price Index" (release 10), which
# also lists the separate February seasonal-factor revisions (removed here: 2013-02-19, 2014-02-18, 2015-02-20,
# 2017-02-13, 2019-02-11, 2020-02-11, 2021-02-08, 2022-02-08, 2023-02-10, 2024-02-09). The October 2025 release was
# cancelled (government shutdown). Dates after 2026-11 from the public schedule page (CPI_PAGE), which copies BLS.
CPI_TEXT = {
    2013: "0116 0221 0315 0416 0516 0618 0716 0815 0917 1030 1120 1217",
    2014: "0116 0220 0318 0415 0515 0617 0722 0819 0917 1022 1120 1217",
    2015: "0116 0226 0324 0417 0522 0618 0717 0819 0916 1015 1117 1215",
    2016: "0120 0219 0316 0414 0517 0616 0715 0816 0916 1018 1117 1215",
    2017: "0118 0215 0315 0414 0512 0614 0714 0811 0914 1013 1115 1213",
    2018: "0112 0214 0313 0411 0510 0612 0712 0810 0913 1011 1114 1212",
    2019: "0111 0213 0312 0410 0510 0612 0711 0813 0912 1010 1113 1211",
    2020: "0114 0213 0311 0410 0512 0610 0714 0812 0911 1013 1112 1210",
    2021: "0113 0210 0310 0413 0512 0610 0713 0811 0914 1013 1110 1210",
    2022: "0112 0210 0310 0412 0511 0610 0713 0810 0913 1013 1110 1213",
    2023: "0112 0214 0314 0412 0510 0613 0712 0810 0913 1012 1114 1212",
    2024: "0111 0213 0312 0410 0515 0612 0711 0814 0911 1010 1113 1211",
    2025: "0115 0212 0312 0410 0513 0611 0715 0812 0911 1024 1218",
    2026: "0113 0213 0311 0410 0512 0610 0714 0812 0911 1014 1110 1210",
}
CPI_MOVED = {"2013-10-30": "2013 年政府停擺延後（原訂 10-16）", "2025-10-24": "2025 年政府停擺延後（原訂 10-15）",
             "2025-12-18": "2025 年停擺後改期（原訂 12-10）", "2026-02-13": "2026 年停擺後改期（原訂 02-11）"}

# window key: (label, hourly offsets from the anchor hour, daily offsets (start label, end label) or None)
WINDOWS = {
    "preweek": ("前一週（T−7 天→T−24 小時）", (-168, -24), (-8, -2)),
    "pre": ("前 24 小時（T−24 小時→T）", (-24, 0), (-2, -1)),
    "post": ("公布後 24 小時（T→T+24 小時）", (0, 24), (0, 1)),
    "post7": ("公布後第 2–7 天（T+24 小時→T+7 天）", (24, 168), (1, 7)),
    "after7": ("公布後 7 天合計（T→T+7 天）", (0, 168), None),
    "around": ("公布前後各 24 小時（T−24→T+24 小時）", (-24, 24), None),
}
FOMC_KEYS = ("preweek", "pre", "post", "post7", "after7")
CPI_KEYS = ("preweek", "pre", "post", "post7", "around")
KIND = {"fomc": "FOMC", "cpi": "CPI"}


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v, digits: int = 2) -> str:
    """v is a fraction."""
    if not _ok(v):
        return "—"
    text = f"{float(v) * 100:+.{digits}f}%"
    return text.replace("-", "+", 1) if float(text[:-1]) == 0 else text


def _pp(v, digits: int = 1) -> str:
    """v is already in percent."""
    return f"{float(v):+.{digits}f}%" if _ok(v) else "—"


def _share(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _ci(ci) -> str:
    return "—" if not ci else f"{ci[0] * 100:+.2f}～{ci[1] * 100:+.2f}"


def _clock(at: datetime) -> str:
    return (f"紐約 {at.astimezone(NY).strftime('%m-%d %H:%M')}＝UTC {at.astimezone(UTC).strftime('%m-%d %H:%M')}"
            f"＝台灣 {at.astimezone(TW).strftime('%m-%d %H:%M')}")


# ---------- events ----------

def _mmdd(year: int, text: str) -> date:
    return date(year, int(text[:2]), int(text[2:4]))


def _fomc_fixed() -> list[dict]:
    out = []
    for year, text in FOMC_TEXT.items():
        for item in text.split():
            day = _mmdd(year, item)
            out.append({"kind": "fomc", "date": day, "local": datetime(day.year, day.month, day.day, 14, 0, tzinfo=NY),
                        "decision": item[4]})
    return out


def _cpi_fixed() -> list[dict]:
    out = []
    for year, text in CPI_TEXT.items():
        for item in text.split():
            day = _mmdd(year, item)
            out.append({"kind": "cpi", "date": day, "local": datetime(day.year, day.month, day.day, 8, 30, tzinfo=NY),
                        "decision": "", "moved": day.isoformat() in CPI_MOVED})
    return out


def _page_text(url: str, store: str) -> tuple[str | None, str | None]:
    """Text of a public page (tags removed); a failed fetch falls back to our stored copy."""
    path = common.STORE / f"macro_days_{store}.txt"
    try:
        reply = requests.get(url, headers=common.HEADERS, timeout=30)
        reply.raise_for_status()
        time.sleep(common.PAUSE)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", reply.text))
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
        return text, None
    except Exception as exc:  # noqa: BLE001
        if path.exists():
            return path.read_text(encoding="utf-8"), f"這次抓不到（{type(exc).__name__}），用上次存下來的版本"
        return None, f"這次抓不到（{type(exc).__name__}）"


MONTHS = {m: i + 1 for i, m in enumerate(("January", "February", "March", "April", "May", "June", "July", "August",
                                          "September", "October", "November", "December"))}


def _fed_calendar(text: str) -> dict[int, list[date]]:
    """Meeting end days per year from the Fed's calendar page (two-day meetings only: "March 17-18", "Jan/Feb 31-1")."""
    out: dict[int, list[date]] = {}
    heads = [(int(m.group(1)), m.start()) for m in re.finditer(r"(\d{4}) FOMC Meetings", text)]
    for n, (year, start) in enumerate(heads):
        end = heads[n + 1][1] if n + 1 < len(heads) else len(text)
        seg = text[start:end].split("Note:")[0]  # the note names next year's first meeting
        days = []
        for m in re.finditer(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                             r"December|Jan/Feb|Feb/Mar|Mar/Apr|Apr/May|May/Jun|Jun/Jul|Jul/Aug|Aug/Sep|Sep/Oct|"
                             r"Oct/Nov|Nov/Dec|Dec/Jan)\s+(\d{1,2})-(\d{1,2})\b", seg):
            month = m.group(1).split("/")[-1]
            month = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May",
                     "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October",
                     "Nov": "November", "Dec": "December"}.get(month, month)
            if "Strategy Meeting" in seg[max(0, m.start() - 40):m.start()]:
                continue  # a framework review meeting, not a rate statement
            try:
                d = date(year, MONTHS[month], int(m.group(3)))
            except ValueError:
                continue
            if not days or d > days[-1]:
                days.append(d)
        out[year] = sorted(set(days))
    return out


def _cpi_schedule(text: str) -> list[date]:
    """Release days from the public CPI schedule page (rows: release date, reference month and year, 8:30 AM)."""
    out = []
    pat = (r"(Canceled Originally [A-Z][a-z]+\.? \d+|Rescheduled to ([A-Z][a-z]+)\.? (\d+) from [A-Z][a-z]+\.? \d+"
           r"|([A-Z][a-z]+)\.? (\d+), (\d{4})) (January|February|March|April|May|June|July|August|September|October|"
           r"November|December) (\d{4}) 8:30 AM")
    short = {k[:3]: v for k, v in MONTHS.items()}
    for m in re.finditer(pat, text):
        if m.group(1).startswith("Canceled"):
            continue
        ref_month, ref_year = MONTHS[m.group(7)], int(m.group(8))
        mon = m.group(2) or m.group(4)
        day = int(m.group(3) or m.group(5))
        month = short.get(mon[:3])
        if not month:
            continue
        year = int(m.group(6)) if m.group(6) else ref_year + (1 if ref_month == 12 else 0)
        try:
            out.append(date(year, month, day))
        except ValueError:
            continue
    return sorted(set(out))


def _events() -> tuple[list[dict], list[dict], dict]:
    fomc, cpi = _fomc_fixed(), _cpi_fixed()
    info = {"problems": [], "fed_mismatch": None, "fed_checked": 0, "cpi_added": 0, "fomc_added": 0}
    text, problem = _page_text(FED_PAGE, "fed_calendar")
    if problem:
        info["problems"].append(f"聯準會會議日曆頁{problem}；用程式裡已核對過的日期。")
    if text:
        cal = _fed_calendar(text)
        known = {e["date"] for e in fomc}
        listed = {d for y, ds in cal.items() if 2021 <= y <= 2026 for d in ds}
        fixed = {e["date"] for e in fomc if 2021 <= e["date"].year <= 2026}
        info["fed_checked"] = len(fixed)
        info["fed_mismatch"] = len(fixed ^ listed) if listed else None
        last = max(known)
        for y, ds in cal.items():
            for d in ds:
                if d > last:
                    fomc.append({"kind": "fomc", "date": d, "local": datetime(d.year, d.month, d.day, 14, 0, tzinfo=NY),
                                 "decision": "?"})
                    info["fomc_added"] += 1
    text, problem = _page_text(CPI_PAGE, "cpi_schedule")
    if problem:
        info["problems"].append(f"CPI 公布日程頁{problem}；用程式裡的日期。")
    if text:
        last = max(e["date"] for e in cpi)
        for d in _cpi_schedule(text):
            if d > last:
                cpi.append({"kind": "cpi", "date": d, "local": datetime(d.year, d.month, d.day, 8, 30, tzinfo=NY),
                            "decision": "", "moved": False})
                info["cpi_added"] += 1
    fomc.sort(key=lambda e: e["date"])
    cpi.sort(key=lambda e: e["date"])
    return fomc, cpi, info


# ---------- prices ----------

class _Btc:
    """Bitcoin prices: hourly grid P[k] = price at b0 + k hours; daily closes c[D] = close at 00:00 UTC of D+1."""

    def __init__(self):
        rows = cr._read(cr.SHARED_STORE)
        if len(rows) < cr.MIN_HOURS:
            rows = cr._read(cr._own_store("BTCUSDT"))
        today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
        self.problem = None
        if not rows or max(rows) < today_ms - cr.HOUR_MS:
            try:
                cr._fetch("BTCUSDT", rows, keep=False)  # keep=False: never writes another module's store
            except Exception as exc:  # noqa: BLE001 - the stored hours are still usable
                self.problem = (f"幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料，"
                                "沒有改用其他方法繞過。")
        if len(rows) < cr.MIN_HOURS:
            raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
        self.P, b0, self.missing = cr._grid(rows)
        self.b0 = b0.to_pydatetime().replace(tzinfo=UTC)
        self.lr = np.diff(np.log(self.P))
        self.K = len(self.P)
        self.end = self.b0 + timedelta(hours=self.K - 1)
        self.hourly_from = self.b0 + timedelta(days=8)
        self.c = te._daily_close()
        self.pre = cr._prehistory(b0)

    def anchor(self, local: datetime) -> int:
        """Hour index of the full hour at or before the moment."""
        return int(math.floor((local.astimezone(UTC) - self.b0).total_seconds() / 3600 + 1e-9))

    def era(self, local: datetime) -> str:
        return "h" if local.astimezone(UTC) >= self.hourly_from else "d"

    def ret(self, local: datetime, key: str, era: str) -> float | None:
        if self.era(local) != era:
            return None
        if era == "h":
            a = self.anchor(local)
            lo, hi = WINDOWS[key][1]
            if a + lo < 0 or a + hi > self.K - 1:
                return None
            return float(self.P[a + hi] / self.P[a + lo] - 1)
        days = WINDOWS[key][2]
        if days is None:
            return None
        d = pd.Timestamp(local.astimezone(UTC).date())
        x, y = self.c.get(d + pd.Timedelta(days=days[0])), self.c.get(d + pd.Timedelta(days=days[1]))
        if not _ok(x) or not _ok(y) or x <= 0 or (d + pd.Timedelta(days=days[1])) > self.c.index[-1]:
            return None
        return float(y / x - 1)


# ---------- event statistics ----------

def _blocked(fomc: list[dict], cpi: list[dict]) -> set:
    """Days within CLEAR_DAYS of any FOMC (also unscheduled) or CPI day."""
    days = [e["date"] for e in fomc + cpi] + [date.fromisoformat(u[0]) for u in UNSCHEDULED]
    return {d + timedelta(days=k) for d in days for k in range(-CLEAR_DAYS, CLEAR_DAYS + 1)}


def _candidates(ev: dict, same: list[dict], blocked: set) -> list[datetime]:
    weeks = {e["date"].isocalendar()[:2] for e in same}
    out = []
    for k in list(range(-WEEKS, 0)) + list(range(1, WEEKS + 1)):
        d = ev["date"] + timedelta(days=7 * k)
        if d.isocalendar()[:2] in weeks or d in blocked:
            continue
        out.append(datetime(d.year, d.month, d.day, ev["local"].hour, ev["local"].minute, tzinfo=NY))
    return out


def _boot(x: np.ndarray, rng) -> tuple[float, float] | None:
    n = len(x)
    if n < 6:
        return None
    block = BLOCK if n > 2 * BLOCK else 1
    nb = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(BOOT, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(BOOT, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def _study_window(events: list[dict], ret_fn, era_fn, key: str, rng) -> dict:
    """Event returns, placebo returns and excess for one window."""
    rows, pool = [], {}
    for ev in events:
        era = era_fn(ev["local"])
        r = ret_fn(ev["local"], key, era)
        if not _ok(r):
            continue
        cands = [(c, ret_fn(c, key, era)) for c in ev["cands"]]
        cands = [(c, v) for c, v in cands if _ok(v)]
        if len(cands) < 5:
            continue
        for c, v in cands:
            pool[c] = v
        rows.append({"ev": ev, "r": r, "plc": float(np.mean([v for _, v in cands])), "era": era})
    out = {"n": len(rows)}
    if not rows:
        return out
    r = np.array([x["r"] for x in rows])
    ex = np.array([x["r"] - x["plc"] for x in rows])
    pv = np.array(list(pool.values()))
    out.update({"mean": float(r.mean()), "median": float(np.median(r)), "up": float((r > 0).mean()),
                "plc_mean": float(np.mean([x["plc"] for x in rows])), "plc_median": float(np.median(pv)),
                "plc_up": float((pv > 0).mean()), "excess": float(ex.mean()), "ci": _boot(ex, rng),
                "n_daily": sum(x["era"] == "d" for x in rows), "rows": rows})
    per = {}
    for name, y0, y1 in PERIODS:
        e = np.array([x["r"] - x["plc"] for x in rows if y0 <= x["ev"]["date"].year <= y1])
        per[name] = {"n": len(e), "excess": float(e.mean()) if len(e) else None, "ci": _boot(e, rng)}
    out["periods"] = per
    dec = {}
    for code in ("u", "h", "d"):
        e = np.array([x["r"] - x["plc"] for x in rows if x["ev"].get("decision") == code])
        dec[code] = {"n": len(e), "excess": float(e.mean()) if len(e) else None, "ci": _boot(e, rng)}
    out["decisions"] = dec
    signs = [np.sign(p["excess"]) for p in per.values() if _ok(p["excess"])]
    out["same_sign"] = len(signs) == len(PERIODS) and len(set(signs)) == 1 and signs[0] != 0
    ci = out["ci"]
    out["ci_clear"] = bool(ci) and (ci[0] > 0 or ci[1] < 0)
    out["passed"] = bool(out["same_sign"] and out["ci_clear"]
                         and np.sign(out["excess"]) == signs[0])
    return out


# ---------- rules on the hourly grid ----------

def _ma50_pos(btc: _Btc, h: int) -> np.ndarray:
    """Position held in each hour interval k -> k+1 for ma50 decided at the daily close at hour h (UTC)."""
    closes = pd.Series(np.concatenate([btc.pre, btc.P[h::24]]))
    mean = closes.rolling(50, min_periods=50).mean()
    want = (closes > mean).astype(float).where(mean.notna()).fillna(0.0).values[len(btc.pre):]
    n = btc.K - 1
    k = np.arange(n)
    j = (k - h) // 24
    pos = np.zeros(n)
    ok = (k >= h) & (j < len(want))
    pos[ok] = want[j[ok]]
    return pos


def _mask(btc: _Btc, anchors: list[int], lo: int, hi: int) -> np.ndarray:
    m = np.zeros(btc.K - 1, dtype=bool)
    for a in anchors:
        m[max(a + lo, 0):max(min(a + hi, btc.K - 1), 0)] = True
    return m


def _net(btc: _Btc, pos: np.ndarray, cost: float) -> np.ndarray:
    change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * btc.lr + math.log(1 - cost) * change


def _metrics(btc: _Btc, pos: np.ndarray, s: int, cost: float) -> dict:
    p = pos[s:].copy()
    net = p * btc.lr[s:] + math.log(1 - cost) * np.abs(np.diff(np.concatenate([[0.0], p])))  # starts from cash
    hours = len(net)
    whole = hours // 24
    daily = np.expm1(net[:whole * 24].reshape(whole, 24).sum(axis=1))
    ret = float(math.expm1(net.sum() * YEAR_H / hours) * 100)
    vol = float(np.std(daily, ddof=1) * math.sqrt(365) * 100)
    path = np.concatenate([[0.0], np.cumsum(net)])
    deep = float((np.exp(path - np.maximum.accumulate(path)) - 1).min() * 100)
    trades = float(np.abs(np.diff(np.concatenate([[0.0], p]))).sum() / (hours / YEAR_H))
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else 0.0, "deep": deep, "trades": trades,
            "share": float(p.mean() * 100), "total": float(math.expm1(net.sum()) * 100)}


def _grid_score(btc: _Btc, masks: dict[str, np.ndarray], cost: float) -> dict:
    """24 closing hours x monthly start dates: yearly return to the end per cell, for hold, ma50 and ma50 x masks."""
    month_days = pd.date_range(GRID_STARTS[0], GRID_STARTS[1], freq="MS")
    end = btc.K - 1
    res = {k: [] for k in ["hold", "ma50"] + list(masks)}
    hold_cum = np.concatenate([[0.0], np.cumsum(btc.lr)])
    for h in range(24):
        base = _ma50_pos(btc, h)
        cums = {"hold": hold_cum, "ma50": np.concatenate([[0.0], np.cumsum(_net(btc, base, cost))])}
        for name, m in masks.items():
            cums[name] = np.concatenate([[0.0], np.cumsum(_net(btc, base * (~m), cost))])
        for day in month_days:
            s = int((day.to_pydatetime().replace(tzinfo=UTC) - btc.b0).total_seconds() // 3600) + h
            if s < 0 or s >= end - 24 * 30:
                continue
            span = end - s
            for name, cum in cums.items():
                res[name].append(math.expm1((cum[end] - cum[s]) * YEAR_H / span) * 100)
    arr = {k: np.array(v) for k, v in res.items()}
    out = {"cells": len(arr["hold"])}
    for name in ["ma50"] + list(masks):
        edge = arr[name] - arr["hold"]
        out[name] = {"beat": float((edge > 0).mean()), "edge": float(np.median(edge)),
                     "vs_ma50": float((arr[name] > arr["ma50"]).mean()) if name != "ma50" else None,
                     "vs_ma50_edge": float(np.median(arr[name] - arr["ma50"])) if name != "ma50" else None}
    return out


def _placebo_rules(btc: _Btc, fomc_h: list[dict], cpi_h: list[dict], real: dict, rng) -> dict:
    s = btc.anchor(datetime.fromisoformat(RULE_START + "T00:00:00+00:00"))
    base = _ma50_pos(btc, 0)
    draws = {"r1": [], "r2": [], "r3": []}

    def pick(events):
        out = []
        for ev in events:
            cs = [c for c in ev["cands"] if c.astimezone(UTC) >= btc.hourly_from and btc.anchor(c) + 170 < btc.K]
            if cs:
                out.append(btc.anchor(cs[int(rng.integers(len(cs)))]))
        return out

    for _ in range(SIMS):
        fa, ca = pick(fomc_h), pick(cpi_h)
        r1 = _mask(btc, fa, -24, 0).astype(float)
        draws["r1"].append(_metrics(btc, r1, s, COST)["total"])
        draws["r2"].append(_metrics(btc, base * (~_mask(btc, fa, 0, 168)), s, COST)["ratio"])
        draws["r3"].append(_metrics(btc, base * (~_mask(btc, ca, -24, 24)), s, COST)["ratio"])
    out = {}
    for k, metric in (("r1", "total"), ("r2", "ratio"), ("r3", "ratio")):
        d = np.array(draws[k])
        out[k] = {"beat": float((real[k][metric] > d).mean()), "median": float(np.median(d)), "metric": metric}
    return out


# ---------- Nasdaq ----------

def _qqq() -> tuple[pd.Series | None, str | None]:
    path = common.STORE / "macro_days_qqq.json"
    try:
        q = ro._share("QQQ")
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({te._d(k): float(v) for k, v in q.items()}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return q, None
    except Exception as exc:  # noqa: BLE001
        if path.exists():
            days = json.loads(path.read_text(encoding="utf-8"))
            return (pd.Series({pd.Timestamp(k): v for k, v in days.items()}).sort_index(),
                    f"Yahoo QQQ 這次抓不到（{type(exc).__name__}），用上次存下來的資料")
        return None, f"Yahoo QQQ 這次抓不到（{type(exc).__name__}），那斯達克的對照這次沒有算，沒有改用其他方法繞過"


Q_WINDOWS = {"qpre": ("會前一天（收盤→收盤）", -2, -1), "qday": ("會議當天（前一天收盤→當天收盤）", -1, 0),
             "qpost": ("會後 5 個交易日", 0, 5)}


def _q_ret_fn(q: pd.Series):
    pos = {d.date(): i for i, d in enumerate(q.index)}
    vals = q.values

    def ret(local: datetime, key: str, era: str) -> float | None:
        i = pos.get(local.date())
        if i is None:
            return None
        a, b = i + Q_WINDOWS[key][1], i + Q_WINDOWS[key][2]
        if a < 0 or b >= len(vals):
            return None
        return float(vals[b] / vals[a] - 1)
    return ret


# ---------- checks ----------

def _dst_check(fomc: list[dict], cpi: list[dict]) -> dict:
    f_hours = {e["local"].astimezone(UTC).strftime("%H:%M") for e in fomc}
    c_hours = {e["local"].astimezone(UTC).strftime("%H:%M") for e in cpi}
    sample = datetime(2019, 3, 20, 14, tzinfo=NY).astimezone(UTC).hour, datetime(2019, 1, 30, 14, tzinfo=NY).astimezone(UTC).hour
    return {"ok": f_hours <= {"18:00", "19:00"} and c_hours <= {"12:30", "13:30"} and sample == (18, 19),
            "fomc": sorted(f_hours), "cpi": sorted(c_hours)}


def _lookahead(btc: _Btc, anchors: list[int]) -> bool:
    """ma50 positions and event masks computed on data cut at 2022-06-30 equal the full-data ones up to that day."""
    cut = btc.anchor(datetime(2022, 6, 30, tzinfo=UTC))
    full = _ma50_pos(btc, 0)[:cut] * (~_mask(btc, anchors, 0, 168))[:cut]

    class _Cut:
        pass
    short = _Cut()
    short.pre, short.P, short.K = btc.pre, btc.P[:cut + 1], cut + 1
    part = _ma50_pos(short, 0) * (~_mask(short, [a for a in anchors if a < cut], 0, 168))
    return bool(np.array_equal(full, part[:cut]))


# ---------- the study ----------

def study() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    fomc, cpi, info = _events()
    problems = list(info["problems"])
    btc = _Btc()
    if btc.problem:
        problems.append(btc.problem)
    blocked = _blocked(fomc, cpi)
    for ev in fomc:
        ev["cands"] = _candidates(ev, fomc, blocked)
    for ev in cpi:
        ev["cands"] = _candidates(ev, cpi, blocked)
    now_utc = datetime.now(UTC)
    past_f = [e for e in fomc if e["local"] < now_utc]
    past_c = [e for e in cpi if e["local"] < now_utc]

    res = {"fomc": {k: _study_window(past_f, btc.ret, btc.era, k, rng) for k in FOMC_KEYS},
           "cpi": {k: _study_window(past_c, btc.ret, btc.era, k, rng) for k in CPI_KEYS}}

    q, q_problem = _qqq()
    if q_problem:
        problems.append(q_problem + "。")
    qres = {}
    if q is not None:
        qret = _q_ret_fn(q)
        qres = {k: _study_window(past_f, qret, lambda _x: "q", k, rng) for k in Q_WINDOWS}

    # rules
    s = btc.anchor(datetime.fromisoformat(RULE_START + "T00:00:00+00:00"))
    hourly_f = [e for e in past_f if e["local"].astimezone(UTC) >= btc.hourly_from]
    hourly_c = [e for e in past_c if e["local"].astimezone(UTC) >= btc.hourly_from]
    fa = [btc.anchor(e["local"]) for e in hourly_f]
    ca = [btc.anchor(e["local"]) for e in hourly_c]
    base = _ma50_pos(btc, 0)
    m_r2, m_r3 = _mask(btc, fa, 0, 168), _mask(btc, ca, -24, 24)
    pos = {"hold": np.ones(btc.K - 1), "ma50": base, "r1": _mask(btc, fa, -24, 0).astype(float),
           "r2": base * (~m_r2), "r3": base * (~m_r3)}
    rules = {k: {c: _metrics(btc, p, s, c) for c in (COST, COST_HIGH)} for k, p in pos.items()}
    real = {k: rules[k][COST] for k in ("r1", "r2", "r3")}
    plc = _placebo_rules(btc, [e for e in hourly_f if btc.anchor(e["local"]) >= s],
                         [e for e in hourly_c if btc.anchor(e["local"]) >= s], real, rng)
    grid = {c: _grid_score(btc, {"r2": m_r2, "r3": m_r3}, c) for c in (COST, COST_HIGH)}

    # R1 per event
    r1_events = []
    for e in hourly_f:
        a = btc.anchor(e["local"])
        if a - 24 >= s and a <= btc.K - 1:
            r1_events.append((e["date"].year, float(btc.P[a] / btc.P[a - 24] - 1)))
    r1_per = {}
    for c in (COST, COST_HIGH):
        nets = [(y, (1 + r) * (1 - c) ** 2 - 1) for y, r in r1_events]
        years = {}
        for y, v in nets:
            years[y] = years.get(y, 1.0) * (1 + v)
        r1_per[c] = {"mean": float(np.mean([v for _, v in nets])) if nets else None,
                     "year": float(np.mean([g - 1 for g in years.values()])) if years else None,
                     "n": len(nets), "years": len(years)}

    # pass tests
    g = grid[COST]
    eff = {"r1": (res["fomc"]["pre"], 1), "r2": (res["fomc"]["after7"], -1), "r3": (res["cpi"]["around"], -1)}
    verdicts = {}
    for k, (w, want) in eff.items():
        effect_ok = bool(w.get("passed")) and np.sign(w.get("excess", 0)) == want
        plc_ok = plc[k]["beat"] >= 0.9
        grid_ok = True if k == "r1" else (g[k]["beat"] >= g["ma50"]["beat"] and g[k]["edge"] >= g["ma50"]["edge"])
        verdicts[k] = {"effect": effect_ok, "placebo": plc_ok, "grid": grid_ok, "all": effect_ok and plc_ok and grid_ok}

    dst = _dst_check(fomc, cpi)
    look_ok = _lookahead(btc, fa)

    # ---------- tables ----------
    def row(kind: str, key: str, w: dict) -> list:
        if not w.get("n"):
            return [f"{KIND[kind]} {WINDOWS[key][0]}", 0, "—", "—", "—", "—", "—", "—"]
        per = "／".join(_p(w["periods"][p]["excess"]) for p, _, _ in PERIODS)
        n_text = f"{w['n']}" + (f"（日線 {w['n_daily']}）" if w["n_daily"] else "")
        return [f"{KIND[kind]} {WINDOWS[key][0]}", n_text, _p(w["mean"]), _p(w["median"]),
                f"{_share(w['up'])}／{_share(w['plc_up'])}", f"{_p(w['excess'])}（{_ci(w['ci'])}）", per,
                "通過" if w["passed"] else ("方向一致、區間含 0" if w["same_sign"] else "不通過")]

    t1 = [row("fomc", k, res["fomc"][k]) for k in FOMC_KEYS] + [row("cpi", k, res["cpi"][k]) for k in CPI_KEYS]

    def cell(w: dict, part: str, name: str) -> str:
        x = w.get(part, {}).get(name) if w.get("n") else None
        if not x or not x["n"]:
            return "—"
        return f"{_p(x['excess'])}（{_ci(x['ci'])}）"

    t2 = []
    for name, _, _ in PERIODS:
        t2.append([name, res["fomc"]["pre"].get("periods", {}).get(name, {}).get("n", 0),
                   cell(res["fomc"]["pre"], "periods", name), cell(res["fomc"]["after7"], "periods", name),
                   res["cpi"]["pre"].get("periods", {}).get(name, {}).get("n", 0),
                   cell(res["cpi"]["pre"], "periods", name), cell(res["cpi"]["around"], "periods", name)])
    for code in ("u", "h", "d"):
        t2.append([f"FOMC：{DECISION[code]}（只描述）", res["fomc"]["pre"].get("decisions", {}).get(code, {}).get("n", 0),
                   cell(res["fomc"]["pre"], "decisions", code), cell(res["fomc"]["after7"], "decisions", code),
                   "—", "—", "—"])

    t3 = []
    for k in Q_WINDOWS:
        w = qres.get(k, {})
        if not w.get("n"):
            t3.append([Q_WINDOWS[k][0], 0, "—", "—", "—", "—", "—", "—"])
            continue
        t3.append([f"QQQ {Q_WINDOWS[k][0]}", w["n"], _p(w["mean"]), _p(w["plc_mean"]), f"{_p(w['excess'])}（{_ci(w['ci'])}）"]
                  + [_p(w["periods"][p]["excess"]) for p, _, _ in PERIODS])
    w = res["fomc"]["pre"]
    if w.get("n"):
        t3.append(["比特幣 會前 24 小時（同上表，對照用）", w["n"], _p(w["mean"]), _p(w["plc_mean"]),
                   f"{_p(w['excess'])}（{_ci(w['ci'])}）"] + [_p(w["periods"][p]["excess"]) for p, _, _ in PERIODS])

    names = {"hold": "一直抱著", "ma50": "50 日均線（00:00 UTC 收盤決定）", "r1": "R1：只在 FOMC 前 24 小時持有",
             "r2": "R2：50 日均線，但 FOMC 公布後 7 天空手", "r3": "R3：50 日均線，但 CPI 前後各 24 小時空手"}
    t4 = []
    for k in ("hold", "ma50", "r1", "r2", "r3"):
        a, b = rules[k][COST], rules[k][COST_HIGH]
        if k in plc:
            pl = f"{_share(plc[k]['beat'])}（隨機中位數 " + (_pp(plc[k]["median"]) if k == "r1" else f"{plc[k]['median']:.2f}") + "）"
        else:
            pl = "—"
        if k == "ma50":
            gr = f"贏一直抱著 {_share(g['ma50']['beat'])} 格，中位數 {_pp(g['ma50']['edge'])}"
        elif k in ("r2", "r3"):
            gr = (f"贏一直抱著 {_share(g[k]['beat'])} 格，中位數 {_pp(g[k]['edge'])}；比 50 日均線好 "
                  f"{_share(g[k]['vs_ma50'])} 格")
        else:
            gr = "—"
        if k in verdicts:
            v = verdicts[k]
            mark = "通過" if v["all"] else "不通過（" + "、".join(
                t for t, okk in (("效果", v["effect"]), ("隨機日期", v["placebo"]), ("排名", v["grid"])) if not okk) + "）"
        else:
            mark = "對照"
        t4.append([names[k], f"{_pp(a['ret'])}／{_pp(b['ret'])}", f"{a['ratio']:.2f}", _pp(a["deep"], 0),
                   f"{a['share']:.0f}%／{a['trades']:.0f} 次", pl, gr, mark])

    tables = [
        {"caption": "比特幣在 FOMC 與 CPI 前後各時段的漲跌，和同星期幾、同時間的一般日子比較（T＝公布時間）",
         "columns": ["事件與時段", "次數", "平均", "中位數", "上漲比例（事件／一般日）", "比一般日多（平均，90% 區間）",
                     "三段各自比一般日多（2013–19／20–22／23–26）", "預設門檻"],
         "rows": t1[:16]},
        {"caption": "分時段與分決策類型：比一般日多多少（平均，括號內 90% 區間）",
         "columns": ["分組", "FOMC 次數", "FOMC 前 24 小時", "FOMC 公布後 7 天", "CPI 次數", "CPI 前 24 小時",
                     "CPI 前後各 24 小時"],
         "rows": t2[:16]},
        {"caption": "股票的「會前漲」在同一段期間還在嗎？那斯達克 100 基金（QQQ，日線）在 FOMC 前後",
         "columns": ["時段", "次數", "事件平均", "一般日平均", "比一般日多（90% 區間）", "2013–2019", "2020–2022",
                     "2023–2026"],
         "rows": t3[:16]},
        {"caption": f"照日曆進出的做法（{RULE_START} 起，1 小時線；成本 0.10%，報酬欄另列 0.30%）",
         "columns": ["做法", "每年報酬（0.10%／0.30%）", "報酬÷波動", "最深虧損", "持有時間比例／每年進出",
                     f"贏過隨機日期的比例（{SIMS} 次）", f"公平網格（24 個收盤時間 × 每月起算，共 {g['cells']} 格）",
                     "預設門檻"],
         "rows": t4},
    ]

    # ---------- now ----------
    future_f = [e for e in fomc if e["local"] >= now_utc]
    future_c = [e for e in cpi if e["local"] >= now_utc]
    now = []
    if future_f:
        e = future_f[0]
        T = e["local"]
        now.append({"label": "下一次 FOMC 公布（聯準會決定利率的會議）", "value": f"{e['date']}（{_clock(T)}）"})
        now.append({"label": "R1 的日曆（只在會前 24 小時持有）", "value": f"{_clock(T - timedelta(hours=24))} 到 {_clock(T)} 這段持有，其他時間空手"})
        now.append({"label": "R2 的日曆（FOMC 後 7 天不持有）", "value": f"{_clock(T)} 到 {_clock(T + timedelta(days=7))} 不持有，其他時間照 50 日均線"})
    else:
        now.append({"label": "下一次 FOMC", "value": "日曆頁上找不到下一次的日期"})
    if future_c:
        e = future_c[0]
        T = e["local"]
        now.append({"label": "下一次 CPI 公布（美國每月公布的物價上漲數字）", "value": f"{e['date']}（{_clock(T)}）"})
        now.append({"label": "R3 的日曆（CPI 前後各 24 小時不持有）",
                    "value": f"{_clock(T - timedelta(hours=24))} 到 {_clock(T + timedelta(hours=24))} 不持有，其他時間照 50 日均線"})
    else:
        now.append({"label": "下一次 CPI", "value": "日程頁上找不到下一次的日期"})
    c = btc.c
    above = c.iloc[-1] > c.iloc[-50:].mean()
    now.append({"label": f"比特幣 {te._d(c.index[-1])} 收盤相對 50 日均線", "value": "在均線之上" if above else "在均線之下"})

    # ---------- verdict ----------
    fp_, fa7, ca_ = res["fomc"]["pre"], res["fomc"]["after7"], res["cpi"]["around"]

    def brief(w: dict) -> str:
        if not w.get("n"):
            return "—"
        return f"平均比一般日{'多' if w['excess'] >= 0 else '少'} {abs(w['excess']) * 100:.2f}%（90% 區間 {_ci(w['ci'])}）"

    def seg(w: dict) -> str:
        return "三段方向一致" if w.get("same_sign") else "三段方向不一致"

    passed = [names[k][:2] for k in verdicts if verdicts[k]["all"]]
    qd = qres.get("qday", {})
    verdict = (
        f"FOMC 前 24 小時比特幣{brief(fp_)}、{seg(fp_)}；公布後 7 天{brief(fa7)}、{seg(fa7)}；"
        f"CPI 前後各 24 小時{brief(ca_)}、{seg(ca_)}——「會前漲、會後跌」{'在這些資料裡站得住' if verdicts['r1']['effect'] and verdicts['r2']['effect'] else '沒有通過事先定好的門檻'}。"
        + (f"同期 QQQ 在 FOMC 當天{brief(qd)}。" if qd.get("n") else "")
        + ("照日曆進出或避開，" + ("只有 " + "、".join(passed) + " 通過" if passed else "三個做法都沒有同時通過效果、贏過 90% 隨機日期與不降低 50 日均線排名這三關")
           + f"（贏過隨機日期的比例 R1 {_share(plc['r1']['beat'])}、R2 {_share(plc['r2']['beat'])}、R3 {_share(plc['r3']['beat'])}）。")
    )
    if len(verdict) > 350:
        verdict = verdict[:349] + "…"

    # ---------- notes ----------
    n_unsched = len(UNSCHEDULED)
    moved_n = sum(1 for e in past_c if e.get("moved"))
    notes = [
        "事先寫好的檢驗（原文照錄）：Windows relative to the announcement time T (FOMC 14:00 ET; CPI 08:30 ET; use hourly BTC "
        "from 2017-08, daily before): 1. Pre-drift: return from T−24h to T−15min (hourly: T−24h to T); Post: T to T+24h, "
        "T+24h to T+7d; Pre-week: T−7d to T−24h. 2. Compare with the same clock windows on non-event days (placebo: same "
        "weekday & hour, weeks without events), mean and median, share positive, block-bootstrap 90% interval; count events. "
        "Split: 2013–2019, 2020–2022, 2023–2026; by decision type (hike/hold/cut — take from the Fed statements/public "
        "record) for FOMC as description only. 3. Tradable rules for an ordinary person (one action per day is fine; costs "
        "0.10/0.30%): (R1) hold BTC only from T−24h to T (pre-drift capture) else cash — report per-event and annual "
        "contribution; (R2) ma50, but step out from T to T+7d around FOMC (avoid post-meeting weakness); (R3) ma50, but "
        "step out from T−24h to T+24h around CPI. Compare R2/R3 with ma50 on the fair grid where feasible and with a placebo "
        "moving the same number of windows to random non-event dates (200 draws). Pass test (stated now): the effect has "
        "the same sign in all three periods with a 90% interval excluding zero AND the rule beats ≥ 90% of placebo draws "
        "AND (for R2/R3) does not lower ma50's ranking score.",
        f"事件：FOMC＝美國聯準會決定利率的會議，用 2013 年起事先排定的 {len(past_f)} 次會議聲明（每次都在紐約時間下午 2 點公布）；"
        f"{n_unsched} 次臨時會議（2020-03-03、2020-03-15 緊急降息）不算，原訂 {CANCELLED} 的會議被取消。"
        f"CPI＝美國每月公布的物價上漲數字，用 {len(past_c)} 次公布（紐約時間早上 8:30），其中 {moved_n} 次因政府停擺改期"
        f"（{'、'.join(CPI_MOVED)}），2025 年 10 月那一次被取消。",
        "時段怎麼量：2017-08 以後用幣安 1 小時線；FOMC 公布時間剛好是整點。CPI 8:30 不是整點，所以 CPI 的時段都從紐約時間 8:00"
        "（公布前半小時）起算，「公布後 24 小時」其實是 8:00 到隔天 8:00。更早的事件只有日線（每天 00:00 UTC 收盤），"
        "用「事件前一個 UTC 日」當會前、「事件後一個 UTC 日」當會後，事件當天跳過，所以只是近似；"
        "「公布後 7 天合計」和「前後各 24 小時」只有 1 小時線才算。表上的「日線 N」是其中用日線近似的次數。",
        f"一般日子（對照）：同一個事件前後 1 到 {WEEKS} 週、同星期幾、同一個紐約時間的同樣時段，排除有同類事件的週，以及前後 "
        f"{CLEAR_DAYS} 天內有 FOMC 或 CPI 的日子。「比一般日多」＝事件時段的漲跌減掉這些對照的平均；90% 區間是把事件依時間順序每 "
        f"{BLOCK} 個一組重抽 {BOOT} 次。三段是否「方向一致」看三段各自的平均，「區間不含 0」看全部事件的區間；表二另列各段自己的區間。",
        f"規則（{RULE_START} 起，1 小時線）：50 日均線是每天 00:00 UTC（台灣早上 8 點）收盤高於 50 日均線就持有到隔天同一時間。"
        "R1 在每次 FOMC 公布前 24 小時買、公布時賣，其他時間空手（空手不計利息）；R2 在 FOMC 公布時賣出、7 天後同一時間照均線恢復；"
        "R3 在 CPI 前一天紐約 8:00 賣出、隔天紐約 8:00 恢復。每次進出扣 0.10%（另算 0.30%）。",
        (f"R1 每次 FOMC 的淨報酬平均 {_p(r1_per[COST]['mean'])}（成本 0.10%）／{_p(r1_per[COST_HIGH]['mean'])}（0.30%），"
         f"{r1_per[COST]['n']} 次；每年加總平均 {_p(r1_per[COST]['year'])}／{_p(r1_per[COST_HIGH]['year'])}"
         f"（{r1_per[COST]['years']} 個年度，每年約 8 次）。"),
        f"隨機日期：把每個事件時段搬到它自己的一個對照日子（同星期幾、同時間、沒有事件的週），重複 {SIMS} 次；R1 比總報酬，"
        "R2／R3 比報酬÷波動。「贏過隨機日期的比例」是真實日曆比隨機日曆好的次數比例。",
        f"公平網格：每天 24 個收盤時間 × {GRID_STARTS[0][:7]} 到 {GRID_STARTS[1][:7]} 每月一個起算日，各自算到最後一天的每年報酬。"
        f"50 日均線的排名分數（同 main_signal_check）＝贏一直抱著的格子比例與中位數多幾個百分點；R2／R3 兩個都不低於 50 日均線才算「不降低」。"
        f"成本 0.30% 時：50 日均線贏一直抱著 {_share(grid[COST_HIGH]['ma50']['beat'])} 格、R2 {_share(grid[COST_HIGH]['r2']['beat'])}、"
        f"R3 {_share(grid[COST_HIGH]['r3']['beat'])}；R2 比 50 日均線好 {_share(grid[COST_HIGH]['r2']['vs_ma50'])} 格、"
        f"R3 {_share(grid[COST_HIGH]['r3']['vs_ma50'])} 格。",
        "那斯達克：Yahoo 只有日線，看不到下午 2 點前那一段，所以 QQQ 用「會議當天收盤到收盤」（包含公布後兩小時），"
        "和 Lucca 與 Moench（2015）量的「公布前 24 小時」不完全一樣；他們的樣本是 1994–2011 年的標普 500。",
        "決策類型（升息／不動／降息）要到公布那一刻才知道，所以只拿來描述，不拿來決定進出。"
        "有幾次 CPI 和 FOMC 在同一天或隔一天（例如 2019-12-11、2020-06-10、2024-06-12），兩者的時段會重疊。",
        f"未來資料與日程檢查：FOMC 日程聯準會會提前一年以上公布（2026 年 11 月的日曆頁已列出 2027 年）；程式內 2021–2026 年的 "
        f"{info['fed_checked']} 個日期和這次抓到的日曆頁" + (f"差 {info['fed_mismatch']} 個。" if info["fed_mismatch"] is not None else "沒有比對到（頁面抓不到）。")
        + "CPI 日程 BLS 也是前一年就公布；改期的 4 次都在新日期前就宣布了，規則只用到前 24 小時。"
        f"夏令時間：FOMC 換算成 UTC 只出現 {'、'.join(dst['fomc'])}，CPI 只出現 {'、'.join(dst['cpi'])}"
        f"（{'正確' if dst['ok'] else '有異常，請留意'}）。把資料截在 2022-06-30 重算均線和避開時段，和完整資料算到同一天"
        f"{'完全相同' if look_ok else '不同（請留意）'}。",
        "資料來源限制：BLS 的 CPI 日程頁拒絕儀表板固定的程式識別（回 403），FRED／ALFRED 對這個識別逾時；沒有改用其他識別繞過。"
        "CPI 歷史日期是從 ALFRED 的公布日期清單（引用 BLS）人工整理進程式，之後的日期由一個轉載 BLS 日程的公開網頁補上。",
        "業界說法（引用，不重算）：Cointelegraph 與 Interdax 的圖表說比特幣常在 FOMC 前漲、之後跌；Vaasa 大學論文的事件研究認為影響並不一致。",
        f"1 小時線缺 {btc.missing} 小時（用前一小時價格補）。這裡只整理歷史，不提供任何買賣建議；過去的結果不保證未來會一樣。",
    ] + problems
    notes.append(f"計算時間約 {time.time() - t0:.0f} 秒。")

    return _clean({
        "id": "macro_days",
        "title": "聯準會開會與 CPI 公布前後",
        "question": "美國聯準會開會（FOMC）和每月物價數字（CPI）公布前後，比特幣有沒有固定的走法？照日曆進出或避開，能不能贏過 50 日均線和隨機日期？",
        "asof": te._d(btc.c.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "聯準會 FOMC 會議日曆（2021 起與未來日程）", "url": FED_PAGE},
        ] + [{"name": f"聯準會 {y} 年 FOMC 歷史頁", "url": FED_HIST.format(y)} for y in range(2013, 2021)] + [
            {"name": "聯準會 FOMC 聲明（例：2026-10-28；決策類型取自每次聲明）",
             "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm"},
            {"name": "ALFRED：消費者物價指數公布日期清單（release 10，引用 BLS）", "url": ALFRED_CPI},
            {"name": "US Inflation Calculator：CPI 公布日程（轉載 BLS，2026–2027）", "url": CPI_PAGE},
            {"name": "BLS CPI 公布日程（拒絕儀表板的程式識別，未使用）", "url": BLS_CPI},
            {"name": "幣安 BTCUSDT 1 小時線與日線", "url": "https://www.binance.com/zh-TC/markets"},
            {"name": "Yahoo Finance QQQ 日線", "url": "https://finance.yahoo.com/quote/QQQ/history"},
            {"name": "Lucca & Moench, The Pre-FOMC Announcement Drift（紐約聯準銀行工作論文 512／Journal of Finance 2015）",
             "url": "https://www.newyorkfed.org/research/staff_reports/sr512.html"},
            {"name": "Cointelegraph：FOMC 結果與比特幣走勢", "url": "https://cointelegraph.com/news/short-the-dip-and-buy-the-rip-what-fomc-outcomes-reveal-about-bitcoin-price-action"},
            {"name": "Interdax：FOMC 公布如何影響比特幣", "url": "https://medium.com/interdax/how-do-fomc-announcements-affect-bitcoin-9f975de87cb3"},
            {"name": "Vaasa 大學論文：貨幣政策公布與加密貨幣（事件研究）",
             "url": "https://osuva.uwasa.fi/bitstreams/50105245-0992-4f34-b16a-b9e7888d49d7/download"},
        ],
    })


def _clean(x):
    """No NaN or infinity anywhere; None in shown cells becomes "—"."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, np.integer):
        return int(x)
    return "—" if x is None else x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
