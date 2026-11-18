"""Historical study "long_cycles": can the Kondratiev long wave (about 50 years) or the shorter economic cycles
(Kuznets 15-25, Juglar 7-11, Kitchin 3-4 years) be used to time markets, and what does that mean for bitcoin?

Bitcoin has only about 16 years of prices, far too short for a 50-year cycle, so the test runs on Robert Shiller's
monthly US stock, bond, price-level and interest-rate data since 1871 (downloaded from his public site and read with
a small built-in .xls reader, because no spreadsheet package is installed).

1. Spectral check: periodograms of yearly series (stock prices after inflation, yearly stock returns, the price level,
   the 10-year interest rate), each compared with 1,000 "red noise" series (random series with the same
   year-to-year persistence) to see whether a peak in a cycle band is bigger than chance would give.
2. A rule fixed from published phase dates (Korotayev & Tsirel 2010, table of Kondratieff phases compiled from the
   long-wave literature): stocks in the published upswings, long bonds in the downswings, after inflation; plus
   versions that switch only years later, as someone who had to wait for the dates to be written down would.
3. Calendar rules (presidential cycle, best 3-4.5-year and 7-11-year half-cycle timing) chosen on 1871-1950 only and
   then measured unchanged on 1951 to now.
4. The trend rules the dashboard already verified on bitcoin (10-month average, 12-month change), on the same data.

No look-ahead: a monthly signal uses data up to the end of month t and sets the holding for month t+1; the calendar
rules are chosen on 1871-1950 returns only. Only public, key-less sources through requests with common.HEADERS.
"""
from __future__ import annotations

import json
import math
import re
import struct
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common

SHILLER_PAGE = "https://shillerdata.com/"
SHILLER_OLD = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
STORE_FILE = common.STORE / "long_cycles_shiller.json"
REFRESH_DAYS = 7
COST = 0.002  # 0.2% of the amount moved every time a rule switches between stocks and bonds
SPLIT = 1950  # calendar rules are chosen on 1871..SPLIT, tested on SPLIT+1..now
SIMS = 1000
SEED = 18710101
NFFT = 2048
BANDS = (("基欽 3–4 年", 3.0, 4.0), ("朱格拉 7–11 年", 7.0, 11.0), ("庫茲涅茨 15–25 年", 15.0, 25.0),
         ("康波 40–60 年", 40.0, 60.0))

# Korotayev & Tsirel (2010), Structure and Dynamics 4(1), table of Kondratieff wave phases compiled from Kondratieff,
# Mandel, Goldstein and others. Each turning point is a published RANGE of years; "?" marks the fifth wave's end.
TURNS = (  # (kind, earliest, latest)
    ("頂", 1870, 1875),
    ("底", 1890, 1896),
    ("頂", 1914, 1920),
    ("底", 1939, 1950),
    ("頂", 1968, 1974),
    ("底", 1984, 1991),
    ("頂", 2008, 2010),
)
# When each turning point was first named in print (best effort from the long-wave literature; see the notes).
FIRST_PRINT = (
    ("1870–1875 頂", "1913 年（van Gelderen）／1926 年（康德拉季耶夫）", 1913, "事後約 40 年才有人寫出長波，Kondratiev 1926 年的論文把它列成第二波的頂"),
    ("1890–1896 底", "1913 年（van Gelderen，荷蘭文，筆名 J. Fedder）", 1913, "事後約 17–23 年"),
    ("1914–1920 頂", "1922–1926 年（康德拉季耶夫）", 1922, "事後 2–8 年，是第一個接近即時被寫出來的轉折"),
    ("1939–1950 底", "1964 年（Mandel 談「長期繁榮」）", 1964, "事後 14–25 年"),
    ("1968–1974 頂", "1964 年 Mandel 預言「約五年後」繁榮結束；1977–1983 年多人確認", 1969,
     "唯一一個事先被預言的轉折，但同期也有很多人沒這樣說，無從知道當時該信誰"),
    ("1984–1991 底", "1990 年代以後的文獻（各家年份不同）", 1995, "年份是估計，標「未查證」；各家定在 1982 到 1991 年之間"),
    ("2008–2010 頂", "2008–2015 年（多家，例如周金濤 2015 年說 2008 是衰退的第一次衝擊）", 2010,
     "表中本來就打了問號；也有人（Nefiodow）認為 2000–2003 年第六波已經開始"),
)
FIRST_PRINT_UNSURE = ("1870–1875 頂", "1890–1896 底", "1984–1991 底")  # second-hand, original not opened


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is a fraction (0.05 = 5%)."""
    if not _ok(v):
        return "—"
    v = float(v) * 100
    if abs(v) < 0.05:
        return "0.0%"
    return f"{v:+,.1f}%" if signed else f"{v:,.1f}%"


def _times(v) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if v < 0.1:
        return f"{v:.2g} 倍"
    return f"{v:,.2f} 倍" if v < 10 else f"{v:,.1f} 倍"


def _p(v) -> str:
    return "—" if not _ok(v) else f"{float(v):.2f}"


# ---------- reading Shiller's .xls without a spreadsheet package ----------

def _ole_stream(data: bytes, name: str) -> bytes:
    """One stream from an OLE2 compound file (the container of an old-style .xls)."""
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError("不是 .xls 檔")
    size = 1 << struct.unpack_from("<H", data, 30)[0]
    n_fat = struct.unpack_from("<I", data, 44)[0]
    dir_start = struct.unpack_from("<I", data, 48)[0]
    difat_start, n_difat = struct.unpack_from("<II", data, 68)
    difat = list(struct.unpack_from("<109I", data, 76))
    sector = difat_start
    for _ in range(n_difat):
        values = struct.unpack_from(f"<{size // 4}I", data, 512 + sector * size)
        difat += values[:-1]
        sector = values[-1]
    fat: list[int] = []
    for s in difat[:n_fat]:
        fat += struct.unpack_from(f"<{size // 4}I", data, 512 + s * size)

    def chain(start: int) -> bytes:
        out, s, steps = [], start, 0
        while s < 0xFFFFFFFA and steps <= len(fat):
            out.append(data[512 + s * size: 512 + (s + 1) * size])
            s, steps = fat[s], steps + 1
        return b"".join(out)

    directory = chain(dir_start)
    for i in range(0, len(directory), 128):
        n = struct.unpack_from("<H", directory, i + 64)[0]
        if directory[i:i + max(n - 2, 0)].decode("utf-16-le", "ignore") == name:
            start, length = struct.unpack_from("<II", directory, i + 116)
            return chain(start)[:length]
    raise ValueError(f".xls 裡沒有 {name}")


def _rk(v: int) -> float:
    if v & 2:
        x = float(struct.unpack("<i", struct.pack("<I", v & 0xFFFFFFFC))[0] >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100 if v & 1 else x


def _sheet_numbers(book: bytes, sheet: str) -> dict:
    """{(row, col): number} of one worksheet (numbers and cached formula results; text cells are skipped)."""
    records, i = [], 0
    while i + 4 <= len(book):
        kind, n = struct.unpack_from("<HH", book, i)
        records.append((kind, book[i + 4:i + 4 + n], i))
        i += 4 + n
    offsets = {}
    for kind, body, _ in records:
        if kind == 0x85:  # BOUNDSHEET
            length, flag = body[6], body[7]
            raw = body[8:8 + length * (2 if flag & 1 else 1)]
            offsets[raw.decode("utf-16-le" if flag & 1 else "latin-1")] = struct.unpack_from("<I", body, 0)[0]
    if sheet not in offsets:
        raise ValueError(f".xls 裡沒有工作表 {sheet}")
    first = next(k for k, r in enumerate(records) if r[2] == offsets[sheet])
    cells: dict = {}
    for kind, body, _ in records[first + 1:]:
        if kind == 0x0A:  # EOF of the sheet
            break
        if kind == 0x203:  # NUMBER
            r, c = struct.unpack_from("<HH", body)
            cells[(r, c)] = struct.unpack_from("<d", body, 6)[0]
        elif kind == 0x27E:  # RK
            r, c = struct.unpack_from("<HH", body)
            cells[(r, c)] = _rk(struct.unpack_from("<I", body, 6)[0])
        elif kind == 0xBD:  # MULRK
            r, c0 = struct.unpack_from("<HH", body)
            for q in range((len(body) - 6) // 6):
                cells[(r, c0 + q)] = _rk(struct.unpack_from("<I", body, 6 + q * 6)[0])
        elif kind == 0x06:  # FORMULA with a numeric cached result
            r, c = struct.unpack_from("<HH", body)
            if body[12:14] != b"\xff\xff":
                cells[(r, c)] = struct.unpack_from("<d", body, 6)[0]
    return cells


def _download() -> tuple[bytes, str]:
    """Shiller's ie_data.xls: the link on his current site first, then the old Yale address."""
    tried = []
    try:
        page = requests.get(SHILLER_PAGE, headers=common.HEADERS, timeout=30)
        page.raise_for_status()
        links = re.findall(r'href="([^"]*ie_data\.xls[^"]*)"', page.text)
        if links:
            url = links[0] if links[0].startswith("http") else "https:" + links[0]
            reply = requests.get(url, headers=common.HEADERS, timeout=60)
            reply.raise_for_status()
            return reply.content, url.split("?")[0]
        tried.append("shillerdata.com 頁面上找不到 ie_data.xls 連結")
    except Exception as exc:  # noqa: BLE001
        tried.append(f"shillerdata.com：{type(exc).__name__}")
    reply = requests.get(SHILLER_OLD, headers=common.HEADERS, timeout=60)
    reply.raise_for_status()
    return reply.content, SHILLER_OLD


def _parse(data: bytes) -> list[list]:
    """[[YYYY-MM, P, CPI, GS10, real total-return stock index, real total-return bond index], ...]."""
    cells = _sheet_numbers(_ole_stream(data, "Workbook"), "Data")
    rows = []
    for r in sorted({r for r, _ in cells}):
        d = cells.get((r, 0))
        if not _ok(d) or d < 1800:
            continue
        year = int(d)
        month = int(round((d - year) * 100))
        if not 1 <= month <= 12:
            continue
        values = [cells.get((r, c)) for c in (1, 4, 6, 9, 18)]
        rows.append([f"{year:04d}-{month:02d}"] + [float(v) if _ok(v) else None for v in values])
    if len(rows) < 1500:
        raise RuntimeError(f"Shiller 資料只讀到 {len(rows)} 個月")
    return rows


def _load() -> tuple[pd.DataFrame, str, str]:
    """Monthly table (cached in the store for a week), its source URL, and a note if an old copy was used."""
    stored = None
    try:
        stored = json.loads(STORE_FILE.read_text(encoding="utf-8")) if STORE_FILE.exists() else None
    except Exception:  # noqa: BLE001
        stored = None
    note = ""
    fresh = stored and (time.time() - stored.get("fetched_ts", 0)) < REFRESH_DAYS * 86400
    if not fresh:
        try:
            data, url = _download()
            rows = _parse(data)
            stored = {"fetched_ts": time.time(), "fetched": common.today_utc(), "url": url, "rows": rows}
            try:
                common.STORE.mkdir(parents=True, exist_ok=True)
                STORE_FILE.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            if not stored:
                raise
            note = f"這次下載 Shiller 資料失敗（{type(exc).__name__}），用的是 {stored.get('fetched')} 存下的版本。"
    frame = pd.DataFrame(stored["rows"], columns=["ym", "P", "CPI", "GS10", "TR", "BOND"])
    frame.index = pd.PeriodIndex(frame.pop("ym"), freq="M")
    this_month = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), freq="M")
    frame = frame[frame.index < this_month]  # the current month is a mid-month value
    frame = frame.dropna(subset=["P", "CPI", "TR", "BOND"])
    return frame, stored["url"], note


# ---------- measuring ----------

def _run(pos: pd.Series, stock: pd.Series, bond: pd.Series, start=None, end=None) -> dict:
    """pos[t] = share in stocks during month t (0/1), the rest in long bonds. Real monthly returns, cost per switch."""
    pos = pos.astype(float)
    change = pos.diff().abs().fillna(0.0)
    net = pos * stock + (1 - pos) * bond - COST * change
    if start is not None:
        net, pos, change = net[net.index >= start], pos[pos.index >= start], change[change.index >= start]
    if end is not None:
        net, pos, change = net[net.index <= end], pos[pos.index <= end], change[change.index <= end]
    wealth = (1 + net).cumprod()
    n = len(net)
    if n == 0:
        return {"ret": None, "deep": None, "wealth": None, "switches": 0, "share": None}
    return {"ret": float(wealth.iloc[-1] ** (12 / n) - 1), "deep": float((wealth / wealth.cummax() - 1).min()),
            "wealth": float(wealth.iloc[-1]), "switches": int(change.iloc[1:].sum()), "share": float(pos.mean())}


def _yearly_state(index: pd.PeriodIndex, years: list[int]) -> pd.Series:
    """Stocks (1) in upswings / bonds (0) in downswings; switches on Jan 1 of each turning year; before the first
    turning point (a peak) the 1871 data starts inside an upswing."""
    state = pd.Series(1.0, index=index)
    yrs = np.array([p.year for p in index])
    current = np.ones(len(index))
    for k, y in enumerate(years):
        current[yrs >= y] = 0.0 if k % 2 == 0 else 1.0  # TURNS alternate peak, trough, peak, ...
    state[:] = current
    return state


# ---------- spectral check ----------

def _detrend(x: np.ndarray, trend: bool) -> np.ndarray:
    """Remove the mean (and a straight-line trend) along the last axis."""
    if not trend:
        return x - x.mean(axis=-1, keepdims=True)
    t = np.arange(x.shape[-1], dtype=float)
    a = np.vstack([np.ones_like(t), t]).T
    coef = np.linalg.lstsq(a, x.T if x.ndim > 1 else x, rcond=None)[0]
    return x - (a @ coef).T


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized tapered periodogram (power shares) and the periods in years, along the last axis."""
    w = np.hanning(x.shape[-1])
    power = np.abs(np.fft.rfft(x * w, n=NFFT, axis=-1)) ** 2
    freq = np.fft.rfftfreq(NFFT, d=1.0)
    power, freq = power[..., 1:], freq[1:]
    power = power / power.sum(axis=-1, keepdims=True)
    return power, 1.0 / freq


def _spectral(series: np.ndarray, trend: bool, rng: np.random.Generator) -> list[dict]:
    x = _detrend(series.astype(float), trend)
    phi = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    phi = min(max(phi, -0.99), 0.99)
    n = len(x)
    sd = x.std()
    shocks = rng.standard_normal((SIMS, n)) * sd * math.sqrt(1 - phi ** 2)
    sims = np.empty((SIMS, n))
    sims[:, 0] = rng.standard_normal(SIMS) * sd
    for t in range(1, n):
        sims[:, t] = phi * sims[:, t - 1] + shocks[:, t]
    sims = _detrend(sims, trend)
    obs, period = _spectrum(x)
    sim_power, _ = _spectrum(sims)
    out = []
    for name, lo, hi in BANDS:
        band = (period >= lo) & (period <= hi)
        top = obs[band].max()
        peak = float(period[band][obs[band].argmax()])
        p = (1 + int((sim_power[:, band].max(axis=1) >= top).sum())) / (SIMS + 1)
        out.append({"band": name, "peak": peak, "p": p})
    return out + [{"phi": phi, "n": n}]


# ---------- calendar rules ----------

def _half_cycle(index: pd.PeriodIndex, length: int, offset: int) -> pd.Series:
    t = np.arange(len(index))
    return pd.Series((((t - offset) % length) < length / 2).astype(float), index=index)


def _choose_calendar(index, stock, bond, lengths, end) -> tuple[int, int, dict]:
    best = None
    for length in lengths:
        for offset in range(length):
            m = _run(_half_cycle(index, length, offset), stock, bond, end=end)
            if best is None or m["wealth"] > best[2]["wealth"]:
                best = (length, offset, m)
    return best


def _bitcoin_clock() -> str:
    """The halving-clock result the dashboard already measured (read from the cached study, not re-run)."""
    try:
        data = json.loads((common.CACHE / "_studies2.json").read_text(encoding="utf-8"))
        for study_ in data.get("studies", []):
            if study_.get("id") != "popular_rules":
                continue
            for table in study_.get("tables", []):
                for row in table.get("rows", []):
                    if row and str(row[0]).startswith("減半時鐘：減半前 500 天"):
                        return (f"比特幣的「減半時鐘」（減半前 500 天買、後 500 天賣）在 2018 年至今只完成 {row[1].replace('只有 ', '')}買→賣，"
                                f"每年 {row[2]}、最深虧 {row[3]}，最後的錢是一直抱著的 {row[5]}——但「500 天」是看過前幾輪才定的，"
                                "整段比特幣歷史也只有 3–4 輪")
    except Exception:  # noqa: BLE001
        pass
    return ("比特幣的「減半時鐘」已由「網路上最常聽到的買賣規則」那一份統計測過（2018 年至今只完成 2 次買→賣），"
            "門檻是看過前幾輪才定的，整段比特幣歷史也只有 3–4 輪")


# ---------- the study ----------

def study() -> dict:
    frame, url, load_note = _load()
    idx = frame.index
    stock = (frame["TR"] / frame["TR"].shift(1) - 1).fillna(0.0)
    bond = (frame["BOND"] / frame["BOND"].shift(1) - 1).fillna(0.0)
    first, last = idx[0], idx[-1]
    years_total = len(idx) / 12
    split_end = pd.Period(f"{SPLIT}-12", freq="M")
    test_start = pd.Period(f"{SPLIT + 1}-01", freq="M")

    # ----- 1. spectral check on yearly series (full calendar years only) -----
    annual = frame.groupby(idx.year).agg({"TR": "mean", "CPI": "mean", "GS10": "mean"})
    counts = frame.groupby(idx.year).size()
    annual = annual[counts == 12]
    dec = frame[idx.month == 12]["TR"]
    yearly_ret = np.log(dec / dec.shift(1)).dropna()
    rng = np.random.default_rng(SEED)
    spec_rows, spec = [], {}
    for key, label, values, trend in (
        ("price", "股價（扣通膨、含股息，取對數去掉長期趨勢）", np.log(annual["TR"].values), True),
        ("ret", "每年股票實質報酬", yearly_ret.values, False),
        ("cpi", "物價水準（取對數去掉長期趨勢）——康德拉季耶夫原本看的", np.log(annual["CPI"].values), True),
        ("rate", "10 年期利率（去掉長期趨勢）——康德拉季耶夫原本看的", annual["GS10"].dropna().values, True),
    ):
        result = _spectral(values, trend, rng)
        meta = result.pop()
        spec[key] = result
        spec_rows.append([label, f"{meta['n']} 年"] + [f"{r['peak']:.1f} 年，p={_p(r['p'])}" for r in result])
    tests = [r for rows in spec.values() for r in rows]
    passed = [r for r in tests if r["p"] < 0.05]
    k_passed = [k for k in spec if spec[k][3]["p"] < 0.05]

    # ----- 2. Kondratiev phase rule -----
    mid = [(a + b + 1) // 2 for _, a, b in TURNS]  # 1870-1875 -> 1873, 1939-1950 -> 1945
    early = [a for _, a, _ in TURNS]
    late = [b for _, _, b in TURNS]
    printed = [y for _, _, y, _ in FIRST_PRINT]
    rules: dict[str, tuple[str, pd.Series]] = {}
    rules["stocks"] = ("一直持有股票（基準）", pd.Series(1.0, index=idx))
    rules["bonds"] = ("一直持有長期公債", pd.Series(0.0, index=idx))
    rules["k_mid"] = ("康波：上升段持股、下降段持債（取年份區間中點）", _yearly_state(idx, mid))
    rules["k_early"] = ("康波：同上，轉折取區間最早一年", _yearly_state(idx, early))
    rules["k_late"] = ("康波：同上，轉折取區間最晚一年", _yearly_state(idx, late))
    rules["k_print"] = ("康波：等轉折第一次見諸文字才換", _yearly_state(idx, printed))
    rules["k_lag10"] = ("康波：中點晚 10 年才換", _yearly_state(idx, [y + 10 for y in mid]))

    # ----- 3. calendar rules chosen on 1871..1950 -----
    elect = pd.Series([(p.year - 1872) % 4 for p in idx], index=idx)  # 0 election year, 1 post, 2 midterm, 3 pre
    excess = (stock - bond)[idx <= split_end].groupby(elect[idx <= split_end]).mean()
    keep_years = sorted(int(k) for k in excess.sort_values(ascending=False).index[:2])  # the two best year types
    all_positive = bool((excess > 0).all())
    year_names = {0: "大選年", 1: "大選後第一年", 2: "期中選舉年", 3: "大選前一年"}
    president = elect.isin(keep_years).astype(float)
    rules["president"] = (f"總統任期週期：只在{'、'.join(year_names[k] for k in keep_years)}持股（1871–1950 最好的兩種年份）", president)
    kl, ko, _ = _choose_calendar(idx, stock, bond, (36, 40, 44, 48, 52), split_end)
    kitchin = _half_cycle(idx, kl, ko)
    rules["kitchin"] = (f"基欽：每 {kl} 個月一輪，持股前半輪（相位用 1871–1950 選最好的）", kitchin)
    jl, jo, _ = _choose_calendar(idx, stock, bond, (84, 96, 108, 120, 132), split_end)
    juglar = _half_cycle(idx, jl, jo)
    rules["juglar"] = (f"朱格拉：每 {jl} 個月一輪，持股前半輪（相位用 1871–1950 選最好的）", juglar)

    # ----- 4. trend rules (signal at month end t, held during t+1) -----
    price = frame["P"]
    above = (price > price.rolling(10, min_periods=10).mean()).astype(float)
    up12 = (price / price.shift(12) > 1).astype(float)
    rules["faber"] = ("順勢：月均價高於 10 個月平均才持股", above.shift(1).fillna(1.0))
    rules["faber_late"] = ("順勢：同上，再晚一個月才換", above.shift(2).fillna(1.0))
    rules["mom12"] = ("順勢：過去 12 個月是漲的才持股", up12.shift(1).fillna(1.0))

    m_full = {k: _run(s, stock, bond) for k, (_, s) in rules.items()}
    m_in = {k: _run(s, stock, bond, end=split_end) for k, (_, s) in rules.items()}
    m_out = {k: _run(s, stock, bond, start=test_start) for k, (_, s) in rules.items()}
    hold_w = m_full["stocks"]["wealth"]
    rule_rows = []
    for key, (label, _) in rules.items():
        f, a, b = m_full[key], m_in[key], m_out[key]
        rule_rows.append([label, _pct(f["ret"]), _pct(f["deep"]), _pct(a["ret"]), _pct(b["ret"]),
                          _times(f["wealth"] / hold_w), str(f["switches"])])

    # luck check for the chosen calendar phases: how many phases of the same length beat holding out of sample
    def oos_share(length: int) -> tuple[int, int]:
        wins = sum(_run(_half_cycle(idx, length, o), stock, bond, start=test_start)["wealth"] > m_out["stocks"]["wealth"]
                   for o in range(length))
        return wins, length
    k_wins, k_n = oos_share(kl)
    j_wins, j_n = oos_share(jl)

    # ----- Kondratiev phase-by-phase table (midpoint dates) -----
    phase_rows, up_w, up_n, dn_w, dn_n = [], 0, 0, 0, 0
    bounds = [first.year] + mid + [last.year + 1]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1] - 1
        sel = (idx.year >= a) & (idx.year <= b)
        if sel.sum() < 12:
            continue
        upswing = k % 2 == 0
        rs = float((1 + stock[sel]).prod() ** (12 / sel.sum()) - 1)
        rb = float((1 + bond[sel]).prod() ** (12 / sel.sum()) - 1)
        stock_won = rs > rb
        if upswing:
            up_n += 1
            up_w += stock_won
        else:
            dn_n += 1
            dn_w += not stock_won
        label = (f"{a}–{last.year}（至今）" if b >= last.year else (str(a) if a == b else f"{a}–{b}"))
        phase_rows.append([label, "上升段" if upswing else "下降段", _pct(rs), _pct(rb),
                           "股票" if upswing else "長期公債", "股票" if stock_won else "長期公債",
                           "對" if stock_won == upswing else "錯"])
    right = up_w + dn_w
    phases = up_n + dn_n

    print_rows = []
    for (kind, a, b), (name, when, year, remark) in zip(TURNS, FIRST_PRINT):
        lag_lo, lag_hi = year - b, year - a
        lag = (f"晚 {lag_lo}–{lag_hi} 年" if lag_lo > 0 else (f"早 {-lag_hi}–{-lag_lo} 年" if lag_hi < 0 else
               f"大致同時（{lag_lo:+d} 到 {lag_hi:+d} 年，負數＝事先）"))
        print_rows.append([name, when + ("（未查證）" if name in FIRST_PRINT_UNSURE else ""), lag, remark])

    # ----- now -----
    elect_now = (last.year - 1872) % 4
    km = rules["k_mid"][1].iloc[-1]
    now = [
        {"label": "資料涵蓋", "value": f"{first.strftime('%Y-%m')} 到 {last.strftime('%Y-%m')}，約 {years_total:.0f} 年（美國）"},
        {"label": "這段資料裝得下幾次完整的康波（約 50 年一次）", "value": f"約 {years_total / 50:.1f} 次"},
        {"label": "Korotayev & Tsirel 2010 整理的表", "value": "第五波上升段 1984–1991 年起、2008–2010 年結束（原表打問號），之後是下降段"},
        {"label": "周金濤的說法（2015 年訪談）", "value": "2008 年次貸危機是第五波衰退的第一次衝擊、2015 年起第二次衝擊，之後進入蕭條；他 2016 年說 2017–2019 年是回落探底期、2019 年是「人生機會」"},
        {"label": "其他說法", "value": "Nefiodow：第六波在 2000–2003 年已經開始（上升段）；Šmihula：第六波約 2015–2035 年（自己打問號）——各家對「現在在哪一段」並不一致"},
        {"label": "照中點日期的康波規則，現在該持有", "value": "股票" if km == 1 else "長期公債"},
        {"label": "10 個月均線規則現在", "value": f"{'持股' if above.iloc[-1] == 1 else '持債'}（{last.strftime('%Y-%m')} 月均價 {price.iloc[-1]:,.0f}，10 個月平均 {price.rolling(10).mean().iloc[-1]:,.0f}）"},
        {"label": "美國總統任期週期現在", "value": f"{last.year} 年是{year_names[elect_now]}"},
    ]

    # ----- verdict -----
    hold_out, k_out = m_out["stocks"], m_out["k_mid"]
    beats_k = [k for k in ("k_mid", "k_early", "k_late", "k_print", "k_lag10") if m_full[k]["wealth"] > hold_w]
    cal_out_wins = [k for k in ("president", "kitchin", "juglar") if m_out[k]["wealth"] > m_out["stocks"]["wealth"]]
    trend_full = m_full["faber"]
    late_trend = m_full["faber_late"]
    verdict = (
        f"美國 {first.year}–{last.year} 年約 {years_total:.0f} 年的資料只裝得下約 {years_total / 50:.0f} 次康波，"
        f"頻譜檢查 {len(tests)} 項裡 {len(passed)} 項比隨機的紅色雜訊明顯，7 個轉折大多事後 10–40 年才被寫出來，"
        "看不出能事先知道的固定節奏。"
        f"照公開年份「上升段持股、下降段持債」{phases} 段裡 {right} 段押對，5 種版本 {len(beats_k)} 種贏過一直持股"
        f"（中點版扣通膨每年 {_pct(m_full['k_mid']['ret'])}，一直持股 {_pct(m_full['stocks']['ret'])}），"
        f"1950 年前選定的 3 種日曆週期在 1951 年後 {len(cal_out_wins)} 種贏，"
        f"10 個月均線每年 {_pct(trend_full['ret'])}（晚一個月換 {_pct(late_trend['ret'])}）、"
        f"最深虧 {_pct(trend_full['deep'], False).lstrip('-')} 對 {_pct(m_full['stocks']['deep'], False).lstrip('-')}。"
        "比特幣只有約 16 年，50 年週期測不了，4 年減半週期也只有 3–4 輪，和基欽週期一樣無法驗證。"
    )

    notes = [
        f"整段資料只裝得下約 {years_total / 50:.1f} 次完整的康波：任何檢定都只有 3 個左右的「樣本」，就算真的有長波，"
        "也幾乎不可能用統計方法證明；反過來說，就算沒有長波，隨機的資料也常常畫得出 3 個大起伏。",
        "頻譜檢查的做法：把每年的數字（去掉直線趨勢）做週期圖，找每個週期範圍裡最高的那一點，再和 1,000 條「紅色雜訊」比——"
        "紅色雜訊是和原資料一樣「今年像去年」程度的隨機數列（AR(1)）。p 值是隨機數列在同一範圍裡出現一樣高或更高峰值的比例，"
        "p 小於 0.05 才算比偶然明顯。一次檢查 16 項，就算全是偶然，平均也會有約 0.8 項過關。在 155 年的資料上，40–60 年這一段"
        "只有大約一個獨立的頻率點，解析度非常粗。",
        "康波的轉折年份用 Korotayev & Tsirel（2010）整理自 Kondratieff、Mandel、Goldstein 等人的表；每個轉折都是一段年份（例如 1939–1950），"
        "所以分別試了中點、最早、最晚三個版本。這些年份大多是事後才排定的；「見諸文字」那一欄是根據長波文獻史的二手整理，"
        "標「未查證」的年份是估計值。",
        f"規則都用 Shiller 的每月資料：股票是 S&P 綜合指數含股息再投入、債券是 10 年期公債含利息，兩者都扣掉 CPI 通膨。"
        f"月底才知道訊號，下個月才換手；每次換手扣 {COST * 100:.1f}%；沒有算稅。Shiller 的月價是當月每日收盤的平均，"
        "平均值會讓走勢看起來比實際更「延續」，對「順勢」規則有利；晚一個月才換的版本每年從 "
        f"{_pct(trend_full['ret'])} 掉到 {_pct(late_trend['ret'])}，可見這個影響很大，比較公平的數字是晚一個月的那一列。",
        f"日曆週期規則只用 1871–1950 年選參數，1951 年以後原封不動檢查：總統週期選 1950 年以前股票平均贏債券最多的兩種任期年份"
        f"（{'1950 年以前四種年份股票其實都贏債券' if all_positive else '並不是每一種年份股票都贏債券'}）；"
        f"基欽選到每 {kl} 個月一輪，同樣長度的 {k_n} 種相位裡，1951 年後有 {k_wins} 種贏過一直持股；"
        f"朱格拉選到每 {jl} 個月一輪，{j_n} 種相位裡有 {j_wins} 種贏過一直持股。"
        f"這類規則有一半時間不持股，而 1951 年後長期公債扣通膨每年只有 {_pct(m_out['bonds']['ret'])}"
        f"（股票 {_pct(m_out['stocks']['ret'])}），除非週期真的能避開大跌，否則很難贏。"
        f"康波分段表裡 {phases} 段中股票有 {up_w + dn_n - dn_w} 段贏過長期公債，"
        f"其中被說成「下降段」的 {dn_n} 段裡股票贏了 {dn_n - dn_w} 段。",
        "只有美國的資料：美國是這 150 年裡最成功的股市之一（倖存者偏誤），其他國家（例如經歷戰爭、革命的市場）結果可能很不一樣。"
        "康德拉季耶夫本人主要看的是物價和利率，不是股價。",
        "黃金：找得到 1833 年起的免費月資料，但 1971 年以前金價大多由政府固定，看不出市場週期，所以沒有拿來測。"
        "FRED 的長期序列在寫這份統計時連不上（伺服器直接斷線），CPI 和 10 年期利率改用 Shiller 檔案裡的同一套資料。",
        _bitcoin_clock() + "。基欽週期（3–4 年）在 1871 年以來的美股上也沒有穩定到能用來進出。",
        "這裡整理的是各家對「現在在康波哪一段」的說法，不代表它們是對的；過去的結果不保證未來會一樣，這不是買賣建議。",
    ]
    if load_note:
        notes.append(load_note)

    sources = [
        {"name": "Robert Shiller 美國股市資料 ie_data.xls（每月 S&P 綜合指數、股息、CPI、10 年期利率，1871 年起）", "url": url},
        {"name": "Shiller 資料頁", "url": SHILLER_PAGE},
        {"name": "Korotayev & Tsirel (2010) A Spectral Analysis of World GDP Dynamics: Kondratieff Waves, Kuznets Swings, Juglar and Kitchin Cycles…, Structure and Dynamics 4(1)",
         "url": "https://escholarship.org/uc/item/9jv108xp"},
        {"name": "Kondratieff (1935) The Long Waves in Economic Life, Review of Economic Statistics（1926 年德文原文的英譯；未開啟原文）",
         "url": "https://doi.org/10.2307/1928486"},
        {"name": "Solomou (1987) Phases of Economic Growth, 1850–1973: Kondratieff Waves and Kuznets Swings（未開啟原文）",
         "url": "https://doi.org/10.1017/CBO9780511559877"},
        {"name": "van Ewijk (1982) A Spectral Analysis of the Kondratieff-Cycle, Kyklos 35(3)（只看摘要）",
         "url": "https://ideas.repec.org/a/bla/kyklos/v35y1982i3p468-99.html"},
        {"name": "Wikipedia：Kondratiev wave（Mandel 1964 年的預言、Nefiodow、Šmihula 的分期）", "url": "https://en.wikipedia.org/wiki/Kondratiev_wave"},
        {"name": "中國證券報 2015-08-26 周金濤專訪", "url": "https://www.cs.com.cn/xwzx/hg/201508/t20150826_4785585.html"},
        {"name": "每日經濟新聞 2016-12-27 周金濤的預測整理", "url": "https://www.nbd.com.cn/articles/2016-12-27/1065266.html"},
        {"name": "劉曉博〈周金濤的「2019 大預言」，為何落空了？〉（2019 年底，格隆匯轉載）", "url": "https://m.gelonghui.com/p/330433"},
    ]

    tables = [
        {"title": "頻譜檢查：每個週期範圍裡最高的峰在幾年、比隨機的紅色雜訊明顯嗎（p < 0.05 才算）",
         "columns": ["序列", "年數"] + [b[0] for b in BANDS], "rows": spec_rows},
        {"title": f"各種進出規則，扣通膨後（1871-01 到 {last.strftime('%Y-%m')}；不持股時持有長期公債）",
         "columns": ["規則", "全期每年", "最深曾經虧多少", "1871–1950 每年", "1951 至今每年（日曆規則沒看過的年份）",
                     "最後的錢 ÷ 一直持股", "換手次數"],
         "rows": rule_rows},
        {"title": "照公開的康波年份（中點）分段：每一段股票和長期公債誰贏（每年、扣通膨）",
         "columns": ["期間", "康波階段", "股票每年", "長期公債每年", "規則該持有", "實際贏的是", "押對了嗎"],
         "rows": phase_rows},
        {"title": "康波的轉折能不能事先知道：每個轉折第一次被寫出來是什麼時候",
         "columns": ["轉折（公開的年份區間）", "第一次見諸文字", "和轉折相比", "說明"], "rows": print_rows},
    ]

    return {
        "id": "long_cycles",
        "title": "康波週期和其他經濟週期，能拿來抓進出時機嗎",
        "question": ("康波週期（約 50 年一次的長期景氣循環）和庫茲涅茨、朱格拉、基欽等較短的經濟週期，有沒有統計證據？"
                     "轉折點能不能事先知道？照著週期進出，有沒有贏過一直持有和簡單的順勢規則？這對比特幣代表什麼？"),
        "asof": f"{last.strftime('%Y-%m')}-{last.days_in_month:02d}",
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
