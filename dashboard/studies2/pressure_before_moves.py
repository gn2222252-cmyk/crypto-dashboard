"""Historical study "pressure_before_moves": an OBSERVATION study. How do buying pressure and selling pressure CHANGE
in the weeks before a big rally starts and before a big fall starts? Laid out event by event. No trading rule is
tested here; rules may come later, from what is seen.

Why (the owner's direct request): "The main thing to observe is how buying pressure and selling pressure CHANGE
before a big rally and before a big fall." Earlier studies tested rules and forward-return averages on all days;
none simply laid out, event by event, what the two pressures did beforehand.

Ground rules (the owner's, applied to every user-facing sentence): observe first, do not conclude lightly; statements
are labelled 量到的 (measured, with the number and in how many events) or 推測 (a guess, clearly marked); where the
data cannot separate two explanations, say so and pick neither; one tendency is never used as a law to dismiss
another; "not seen" means not seen with this definition and this data, and what might be missing is named; the
individual events are shown, not only averages.

Data (free, key-less, fixed user agent "btc-research-dashboard"; a refusal is reported, never worked around; .env is
never read). Loaders are REUSED from earlier studies; this study's own fallback stores are
cache/_store/pressure_before_moves_*.json:
  BTC daily closes: te._daily_close (00:00 UTC closes; blockchain.info before Binance 2017-08). Candle D closes at
    D+1 00:00 UTC; "day t" below is the candle labelled t, and everything used at day t is known by its close.
  Coin Metrics community API, asset btc, FlowInExNtv / FlowOutExNtv (BTC sent to / withdrawn from exchange-tagged
    addresses), through force_balance._cm_fetch. Data day D used at day D+1 (lag 1, force_balance's).
  Open Bitcoin Metrics CSV DATA FILE obm_spent_value_ge155d_btc_daily only (HTTP GET of the CSV, parsed into dates and
    numbers by old_coin_selling._parse; the repository is never cloned, installed or run; old_coin_selling's store is
    read only as a last fallback). Winsorised 30-day sum = old_coin_selling._measure(...)["old"]. Data day D used at
    day D+7 (lag 7, old_coin_selling's).
  Binance spot BTCUSDT 1-hour klines (price_impact._hourly and its store): close, quote volume, taker-buy quote volume,
    2017-08-17+. Price impact per dollar = price_impact._measure (UP, DOWN; hours opened D-13 .. D 23:00, used at day
    D). Taker volumes per UTC day summed from the same hours, used at day D.
    Binance's zero-fee BTC period (2022-07-08 .. 2023-03-22) inflated volume: every taker / impact quantity is set
    missing on days price_impact.ZF_DROP marks (2022-07-08 .. 2023-04-04), fixed before running.
  DefiLlama total stablecoin supply (supply_demand._llama_fetch), cleaned daily change force_balance._stable_daily
    (from 2019-07-01, coverage-jump days set to 0). Data day D used at day D+1.
  Farside US spot bitcoin ETF Total net flow (force_balance._etf_load, force_balance._etf_daily: no-trading days 0).
    Missing (not 0) before 2024-01-11, because the ETFs did not exist. Data day D used at day D+1.
  Coinbase premium = Coinbase BTC-USD / Bitstamp BTC/USD daily close - 1 (supply_demand._coinbase, supply_demand.
    _bitstamp: read-only stores, missing days fetched into memory), 2017-08-17+. Day D used at day D.
  Binance BTCUSDT perpetual funding settlements (funding_warning._funding -> carry._funding), 2019-09+. Settlements
    timestamped on UTC date D (all before D+1 00:00) used at day D.
  Binance USDT-M futures daily metrics files, BTCUSDT sum_open_interest of the 00:00 row (ls_ratio._fetch and its
    store, oi_flush._oi), 2020-09+. The value stamped D+1 00:00 is matched with the close of candle D (oi_flush's
    alignment).

Events (fixed before running; found mechanically from daily closes by a zigzag, scanned from 2011-01-01, listed from
2012-01-01):
  BIG RISE start: a low followed by a rise of at least 50% before any 25% fall from the running high since that low.
    Machine: seek mode tracks the lowest close since the last reset (lo) and the highest close after it (hi); when
    hi/lo - 1 >= 50% the event is the date of lo; the rise then runs until the first close 25% below its running high
    (peak = that running high); a 25% fall from hi before reaching +50% disqualifies lo and the search restarts at
    that close. Size = peak/low - 1, duration = days low -> peak. Also listed: the subset with size >= 100%.
  BIG FALL start: mirror image - a high followed by a fall of at least 30% before any 25% rise from the running low.
    Event date = the high. Also listed: the subset with a fall of at least 50%.
  The last event may still be running (marked). These dates are only known AFTERWARDS: this study describes what the
  pressures looked like beforehand; it does not claim the moves could have been traded.

Pressures (fixed before running), each a daily series on day t (already lagged to publication as above), each shown
in its own units and as a RANK 0-100 = share of the previous 365 days' values (t-365 .. t-1, at least 180 non-missing;
180 so that the taker series recover within a year of the zero-fee gap) below today's value, x 100:
  SELL side: S_dep7, S_dep30 = BTC deposited to exchanges, 7- and 30-day sums (all days required);
    S_old = old_coin_selling's winsorised 30-day sum of coins held 155+ days that moved;
    S_take = taker SELL quote volume (quote - taker-buy quote), 7-day sum, at least 150 of 168 hours present;
    S_imp = DOWN impact: price lost per dollar of net selling, 14 days (shown as % per 100 million USD).
  BUY side: B_wd7, B_wd30 = BTC withdrawn from exchanges, 7- and 30-day sums; B_stb = net new stablecoin supply,
    30-day sum (USD); B_etf = US spot ETF net flow, 30-day sum (USD); B_take = taker BUY quote volume, 7-day sum;
    B_imp = UP impact: price gained per dollar of net buying, 14 days; B_cb = Coinbase premium, 7-day mean (>= 5 days).
  BALANCE: NET7, NET30 = (withdrawn - deposited)/(withdrawn + deposited) in coins over 7 / 30 days; TAKE = taker buy
    share of volume, 7 days; IMPR = log(B_imp / S_imp) (price_impact's ASYM; rank identical to the ratio's).
  ACTIVITY: ACT = withdrawn + deposited, 30 days, coins.
  LEVERAGE: FUND = mean funding of the settlements on the last 7 UTC dates (at least 5 dates), per year;
    OI = log(OI / OI 30 days earlier), stamp days.
  Composites (ranks only): SELL_C = mean of the available ranks of S_dep30, S_old, S_take, S_imp; BUY_C = mean of the
    available ranks of B_wd30, B_stb, B_etf, B_take, B_imp, B_cb (at least one needed). Their make-up changes over
    time (before 2017-08 SELL_C = S_dep30 + S_old and BUY_C = B_wd30 alone); stated on the page.

Layout (all fixed before running; base event sizes = rise >= 50%, fall >= 30%):
  1. Average rank path from -90 to +29 days in 12 bins of 10 days (-90..-81, ..., -10..-1, 0..+9, +10..+19,
     +20..+29); a bin value per event = mean rank over its 10 days (at least 5 present). Next to it the same for random
     non-event days: pool = days from 2012-01-01 to 30 days before the last close that are at least 60 days from every
     base-size event of EITHER type (if that pool has fewer than 365 days, only events of the same type are excluded,
     stated). For each pressure, bin and event type: 2,000 draws of n random pool days with data (n = number of
     events with data in that bin); band = 10th-90th percentile of the draws' means; event average marked above / below.
  2. CHANGE before the event: D30 = mean rank over days -30..-1 minus mean rank over -90..-31 (at least 20 of 30 and
     40 of 60 present); late change D10 = mean over -10..-1 minus mean over -40..-11 (at least 7 of 10 and 20 of 30).
     Per event values listed; counts of rises (>0) and falls (<0); median. Random pool days: median D, share > 0, and
     the share of 2,000 draws of n random days whose median D is at least as far from zero in the events' direction.
     Measured sentences "在 N 次大漲前，有 k 次 ... 上升、j 次下降". The same counts for the large subsets.
  3. Per event: D30 of SELL_C, BUY_C, NET30 rank, ACT rank, and price change close(-1)/close(-31) - 1. Pattern by the
     signs of D30(SELL_C) and D30(BUY_C): selling down & buying up / both up / both down / selling up & buying down
     (exact zero = flat, counted separately). Tally next to the share of random pool days in each pattern x N.
  4. Rises versus falls: per pressure, median D30 before rises minus median D30 before falls; null interval = 5th-95th
     percentile of the same difference after shuffling event-type labels among the events with data (2,000 shuffles);
     share of shuffles at least as large in absolute value. Both are "before a big move"; the price paths before them
     differ by construction (falls before lows, rises before highs), so a difference may only mirror price.
  5. First look (chosen after seeing step 2, labelled so; NOT a test): rank every (event type, pressure) pair with at
     least 8 events by k/N, k = the larger of the rise and fall counts of D30; ties broken by larger N, then larger
     |median|; take the top 3. For each: all days from 2012-01-01 with a complete 30-day look-ahead on which D30 has
     the same sign and at least the events' |median| size; share followed by an event of that type within the next 30
     days (event date in [day, day+29]) versus the same share over all days with D30 defined; days and separate
     episodes (gaps > 30 days) counted.
  6. Coverage and limits per pressure: first day with a rank, events with D30 data, what it cannot see.
Today: today's rank of SELL_C, BUY_C, NET30, ACT and D30 measured over the last 30 days versus the prior 60
(days t-29..t versus t-89..t-30); plain words; no rule state.
Verdict: what was 量到的 (k of N), what differed before rises versus falls, what did not, what could not be observed;
then, marked 推測, at most three candidate ideas for a proper pre-registered test (written after seeing the results,
labelled so). Seeds fixed. Not advice.
ADDED AFTER SEEING THE RESULTS (definitions and thresholds unchanged): the verdict's "last 10 days, two most
consistent" and "buy/sell pattern tally" sentences, the three 推測 texts, notes on the price change before events and
on events overlapping (opposite event within the prior 90 days), and display formats (one decimal on event sizes,
three on the premium).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import force_balance as fb
from dashboard.studies2 import funding_warning as fw
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import oi_flush as oo
from dashboard.studies2 import old_coin_selling as ocs
from dashboard.studies2 import price_impact as pi
from dashboard.studies2 import supply_demand as sd
from dashboard.studies2 import trend_ensemble as te

DAY = pd.Timedelta(days=1)
SCAN_FROM, LIST_FROM = "2011-01-01", "2012-01-01"
RISE, RISE_BIG, RISE_STOP = 0.50, 1.00, 0.25
FALL, FALL_BIG, FALL_STOP = 0.30, 0.50, 0.25
WIN, WIN_MIN = 365, 180
BINS = tuple(range(-90, 30, 10))
GAP = 60
DRAWS = 2000
SEED = 20261119
MIN_EV = 8
STORE_CM = common.STORE / "pressure_before_moves_cm.json"
STORE_LL = common.STORE / "pressure_before_moves_llama.json"
STORE_OBM = common.STORE / "pressure_before_moves_obm.json"
OBM_ID = ocs.IDS["ge155"]

# key, group, Chinese label, short header, unit
PRESS = (
    ("S_dep7", "sell", "送進交易所的幣（7 天加總）", "送進7天", "BTC"),
    ("S_dep30", "sell", "送進交易所的幣（30 天加總）", "送進30天", "BTC"),
    ("S_old", "sell", "老幣移動（放 155 天以上的幣被花掉，30 天）", "老幣", "BTC"),
    ("S_take", "sell", "主動賣出的量（幣安現貨，7 天）", "主動賣", "億美元"),
    ("S_imp", "sell", "賣出壓價力：每 1 億美元淨賣出讓價格跌多少（14 天）", "壓價力", "%"),
    ("B_wd7", "buy", "從交易所提走的幣（7 天加總）", "提走7天", "BTC"),
    ("B_wd30", "buy", "從交易所提走的幣（30 天加總）", "提走30天", "BTC"),
    ("B_stb", "buy", "新增穩定幣（30 天）", "穩定幣", "億美元"),
    ("B_etf", "buy", "美國現貨 ETF 淨流入（30 天）", "ETF", "億美元"),
    ("B_take", "buy", "主動買進的量（幣安現貨，7 天）", "主動買", "億美元"),
    ("B_imp", "buy", "買進推價力：每 1 億美元淨買進讓價格漲多少（14 天）", "推價力", "%"),
    ("B_cb", "buy", "Coinbase 溢價（7 天平均）", "CB溢價", "%"),
    ("NET7", "bal", "淨提走比例 NET（7 天）＝(提走−送進)÷(提走＋送進)", "NET7", "%"),
    ("NET30", "bal", "淨提走比例 NET（30 天）", "NET30", "%"),
    ("TAKE", "bal", "主動買進佔成交量比例（7 天）", "買進比例", "%"),
    ("IMPR", "bal", "推價力 ÷ 壓價力（取對數，14 天）", "推÷壓", "對數"),
    ("ACT", "act", "活躍度：提走＋送進（30 天）", "活躍度", "BTC"),
    ("FUND", "lev", "資金費率（7 天平均，年化）", "資金費率", "%/年"),
    ("OI", "lev", "未平倉量 30 天變化", "未平倉", "%"),
    ("SELL_C", "comp", "賣方合計（可用賣方排名的平均）", "賣方合計", "排名"),
    ("BUY_C", "comp", "買方合計（可用買方排名的平均）", "買方合計", "排名"),
)
KEYS = tuple(p[0] for p in PRESS)
LABEL = {p[0]: p[2] for p in PRESS}
SHORT = {p[0]: p[3] for p in PRESS}
UNIT = {p[0]: p[4] for p in PRESS}
GROUP = {p[0]: p[1] for p in PRESS}
SELL_PARTS = ("S_dep30", "S_old", "S_take", "S_imp")
BUY_PARTS = ("B_wd30", "B_stb", "B_etf", "B_take", "B_imp", "B_cb")
TAKER = ("S_take", "B_take", "TAKE", "S_imp", "B_imp", "IMPR")
TYPE_CN = {"rise": "大漲", "fall": "大跌"}
LIMITS = {
    "S_dep7": "交易所地址是用今天的標籤往回套，越早年漏得越多；交易所自己錢包之間搬幣也算進去；送進去不一定賣。",
    "S_dep30": "同上（30 天版）。",
    "S_old": "老幣移動不一定是賣：託管搬家、交易所換錢包、ETF 託管整理都算；沒有分辨是不是同一個人。資料晚 7 天才用。",
    "S_take": "只有幣安現貨；2022-07～2023-04 零手續費期間的量被拿掉；看不到掛單深度、強制平倉、場外交易。",
    "S_imp": "只有幣安現貨，用 1 小時線算，看不到小時內的細節；零手續費期間拿掉。",
    "B_wd7": "和送進一樣用今天的標籤往回套；提走不一定是新買的人（也可能是搬去別的交易所或自己的冷錢包）。",
    "B_wd30": "同上（30 天版）。",
    "B_stb": "2019-07 以後才有；新增的穩定幣不一定拿去買比特幣。",
    "B_etf": "2024-01-11 以後才有，排名要再等半年；事件很少。",
    "B_take": "只有幣安現貨；零手續費期間拿掉；主動買和主動賣都跟著成交量一起漲跌。",
    "B_imp": "只有幣安現貨，1 小時線；零手續費期間拿掉。",
    "B_cb": "2017-08 以後；只比 Coinbase 和 Bitstamp 兩家美元價。",
    "NET7": "交易所標籤往回套、內部搬幣都會影響；只看幣數，不看金額。",
    "NET30": "同上（30 天版）。",
    "TAKE": "只有幣安現貨；零手續費期間拿掉。",
    "IMPR": "只有幣安現貨；零手續費期間拿掉；兩個數都要是正的才算得出來。",
    "ACT": "內部搬幣會灌大活躍度；早年標籤少，會偏低。",
    "FUND": "只有幣安 BTCUSDT 永續合約，2019-09 以後。",
    "OI": "只有幣安 BTCUSDT 合約，2020-09 以後；沒有強制平倉資料。",
    "SELL_C": "組成會變：2017-08 以前只有送進交易所和老幣兩項。",
    "BUY_C": "組成會變：2017-08 以前只有提走一項，2019 加穩定幣，2024 加 ETF。",
}

_ok = sd._ok


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _sg(v, nd: int = 0) -> str:
    """signed number"""
    return f"{float(v):+.{nd}f}" if _ok(v) else "—"


def _pc(v, nd: int = 0) -> str:
    """fraction -> signed percent"""
    return f"{float(v) * 100:+.{nd}f}%" if _ok(v) else "—"


def _num(v, unit: str) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if unit == "BTC":
        return f"{v:,.0f}"
    if unit in ("億美元",):
        return f"{v:,.1f}"
    if unit in ("%", "%/年"):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return f"{v:.3f}"


# ---------- data ----------

def _stored(path, fetch, fallback=None) -> tuple[dict, str | None]:
    """supply_demand._stored with one extra read-only fallback (an earlier study's store)."""
    try:
        return sd._stored(path, fetch)
    except Exception:  # noqa: BLE001
        if fallback is None or not fallback.exists():
            raise
        return json.loads(fallback.read_text(encoding="utf-8")), f"{path.stem} 這次取不到，改讀之前其他研究存下來的資料。"


def _obm_fetch() -> dict:
    """Only the >= 155-day spent series (the same CSV and parser as old_coin_selling)."""
    reply = requests.get(ocs.OBM_BASE.format(OBM_ID), headers=common.HEADERS, timeout=90)
    reply.raise_for_status()
    if len(reply.content) > 20_000_000:
        raise RuntimeError("OBM 檔案太大")
    out, _meta = ocs._parse("ge155", reply.text)
    return out


def _obm() -> tuple[pd.Series, str | None]:
    try:
        raw, note = sd._stored(STORE_OBM, _obm_fetch)
    except Exception as exc:  # noqa: BLE001 - old_coin_selling's store, read only
        if not ocs.STORE_OBM.exists():
            raise
        raw = json.loads(ocs.STORE_OBM.read_text(encoding="utf-8"))["ge155"]
        note = f"OBM 這次取不到（{type(exc).__name__}），改讀 old_coin_selling 存下來的資料。"
    s = pd.Series(raw, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), note


def _load() -> dict:
    problems: list[str] = []
    close = te._daily_close()
    today = pd.Timestamp(common.today_utc())
    raw, n = _stored(STORE_CM, fb._cm_fetch, fb.STORE_CM)
    problems += [n] if n else []
    cm = pd.DataFrame.from_dict(raw, orient="index")
    cm.index = pd.to_datetime(cm.index)
    cm = cm.sort_index()
    cm = cm[cm.index < today].asfreq("D")
    obm, n = _obm()
    problems += [n] if n else []
    ll, n = _stored(STORE_LL, sd._llama_fetch, fb.STORE_LL)
    problems += [n] if n else []
    sup = pd.Series(ll, dtype=float)
    sup.index = pd.to_datetime(sup.index)
    sup = sup.sort_index()
    sup = sup[sup.index < today].asfreq("D").ffill()
    etf, n = fb._etf_load()
    problems += [n] if n else []
    hourly, n = pi._hourly()
    problems += [n] if n else []
    cb, n = sd._coinbase()
    problems += [n] if n else []
    bs, n = sd._bitstamp()
    problems += [n] if n else []
    fund, _where, more = fw._funding()
    problems += more
    try:
        oi_rows, n = lr._fetch("BTCUSDT", budget_s=300)
        problems += [n] if n else []
    except Exception as exc:  # noqa: BLE001 - leverage is optional
        oi_rows = None
        problems.append(f"未平倉量這次取不到（{type(exc).__name__}），槓桿那一行留空。")
    return {"close": close, "cm": cm, "obm": obm, "sup": sup, "etf": etf, "hourly": hourly, "cb": cb, "bs": bs,
            "fund": fund, "oi": oi_rows, "problems": problems}


# ---------- pressures ----------

def _trail_rank(x: pd.Series) -> pd.Series:
    """0-100: share of the previous WIN values (at least WIN_MIN non-missing) below today's value."""
    v = x.values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    if n <= WIN:
        return pd.Series(out, index=x.index)
    win = np.lib.stride_tricks.sliding_window_view(v[:-1], WIN)
    cur = v[WIN:]
    fin = np.isfinite(win)
    cnt = fin.sum(axis=1)
    below = (np.where(fin, win, np.inf) < cur[:, None]).sum(axis=1)
    ok = np.isfinite(cur) & (cnt >= WIN_MIN)
    out[WIN:] = np.where(ok, 100.0 * below / np.maximum(cnt, 1), np.nan)
    # the first WIN days can still have WIN_MIN earlier values
    for i in range(WIN_MIN, WIN):
        if np.isfinite(v[i]):
            prev = v[:i][np.isfinite(v[:i])]
            if len(prev) >= WIN_MIN:
                out[i] = 100.0 * float((prev < v[i]).mean())
    return pd.Series(out, index=x.index)


def _lag(s: pd.Series, idx: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Data-day series -> value used at day t (data day t - lag)."""
    return pd.Series(s.reindex(idx - lag * DAY).values.astype(float), index=idx)


def _taker_daily(rows: dict) -> pd.DataFrame:
    a = np.array([[ms, v[1], v[2]] for ms, v in rows.items()], dtype=float)
    day = pd.to_datetime((a[:, 0] // pi.DAY_MS) * pi.DAY_MS, unit="ms")
    ok = np.isfinite(a[:, 1]) & np.isfinite(a[:, 2]) & (a[:, 1] > 0)
    f = pd.DataFrame({"q": np.where(ok, a[:, 1], 0.0), "tb": np.where(ok, a[:, 2], 0.0), "h": ok.astype(float)},
                     index=day)
    f = f.groupby(level=0).sum()
    today = pd.Timestamp(common.today_utc())
    return f[f.index < today].asfreq("D").fillna(0.0)


def _pressures(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    close = data["close"]
    idx = close.index
    cm = data["cm"]
    first: dict = {}
    raw = pd.DataFrame(index=idx, dtype=float)
    fin, fout = cm["FlowInExNtv"], cm["FlowOutExNtv"]
    in7, in30 = fin.rolling(7, min_periods=7).sum(), fin.rolling(30, min_periods=30).sum()
    out7, out30 = fout.rolling(7, min_periods=7).sum(), fout.rolling(30, min_periods=30).sum()
    raw["S_dep7"], raw["S_dep30"] = _lag(in7, idx, 1), _lag(in30, idx, 1)
    raw["B_wd7"], raw["B_wd30"] = _lag(out7, idx, 1), _lag(out30, idx, 1)
    with np.errstate(all="ignore"):
        raw["NET7"] = _lag(((out7 - in7) / (out7 + in7)).where(out7 + in7 > 0), idx, 1) * 100
        raw["NET30"] = _lag(((out30 - in30) / (out30 + in30)).where(out30 + in30 > 0), idx, 1) * 100
    raw["ACT"] = _lag(in30 + out30, idx, 1)
    # old coins
    m = ocs._measure(data["obm"])
    raw["S_old"] = _lag(m["old"], idx, ocs.LAG)
    # taker volumes and impact (Binance spot)
    td = _taker_daily(data["hourly"])
    q7, tb7, h7 = (td[c].rolling(7, min_periods=7).sum() for c in ("q", "tb", "h"))
    good = h7 >= 150
    sell7 = (q7 - tb7).where(good)
    buy7 = tb7.where(good)
    raw["S_take"] = _lag(sell7, idx, 0) / 1e8
    raw["B_take"] = _lag(buy7, idx, 0) / 1e8
    with np.errstate(all="ignore"):
        raw["TAKE"] = _lag((buy7 / q7).where(good & (q7 > 0)), idx, 0) * 100
    f = pi._measure(data["hourly"], idx)
    raw["B_imp"] = f["up"].values * 1e8 * 100
    raw["S_imp"] = f["down"].values * 1e8 * 100
    raw["IMPR"] = f["asym"].values
    zf = (idx >= pd.Timestamp(pi.ZF_DROP[0])) & (idx <= pd.Timestamp(pi.ZF_DROP[1]))
    for k in TAKER:
        raw.loc[zf, k] = np.nan
    # stablecoins, ETF
    st = fb._stable_daily(data["sup"]).rolling(30, min_periods=30).sum()
    raw["B_stb"] = _lag(st, idx, 1) / 1e8
    etf = data["etf"]
    ed = fb._etf_daily(etf, idx[-1])
    ed = ed[ed.index >= fb.ETF_FROM]
    raw["B_etf"] = _lag(ed.rolling(30, min_periods=30).sum(), idx, 1) / 1e8
    # Coinbase premium
    days = pd.date_range(sd.FIRST, idx[-1], freq="D")
    cbs = pd.Series({pd.Timestamp(k): (v[0] if isinstance(v, list) else v) for k, v in data["cb"].items()},
                    dtype=float).sort_index().reindex(days)
    bss = pd.Series({pd.Timestamp(k): (v[0] if isinstance(v, list) else v) for k, v in data["bs"].items()},
                    dtype=float).sort_index().reindex(days)
    prem = (cbs / bss - 1).where((cbs > 0) & (bss > 0))
    raw["B_cb"] = _lag(prem.rolling(7, min_periods=5).mean() * 100, idx, 0)
    # funding
    fr = data["fund"]
    fd = fr.groupby(fr.index.floor("D")).mean().asfreq("D")
    raw["FUND"] = _lag(fd.rolling(7, min_periods=5).mean() * 3 * 365 * 100, idx, 0)
    # open interest (stamp D+1 00:00 matched with the close of candle D)
    if data["oi"]:
        o = oo._oi(data["oi"])
        with np.errstate(all="ignore"):
            ch = (o / o.shift(30) - 1) * 100
        raw["OI"] = pd.Series(ch.reindex(idx + DAY).values.astype(float), index=idx)
    else:
        raw["OI"] = np.nan
    rank = pd.DataFrame(index=idx, dtype=float)
    for k in KEYS:
        if k in ("SELL_C", "BUY_C"):
            continue
        s = raw[k].astype(float)
        s = s.where(np.isfinite(s))
        raw[k] = s
        rank[k] = _trail_rank(s)
    rank["SELL_C"] = rank[list(SELL_PARTS)].mean(axis=1, skipna=True)
    rank["BUY_C"] = rank[list(BUY_PARTS)].mean(axis=1, skipna=True)
    raw["SELL_C"], raw["BUY_C"] = rank["SELL_C"], rank["BUY_C"]
    for k in KEYS:
        fv = rank[k].first_valid_index()
        first[k] = fv
    return raw, rank, first


# ---------- events ----------

def _rises(close: pd.Series) -> list[dict]:
    c = close[close.index >= pd.Timestamp(SCAN_FROM)]
    v, ix = c.values.astype(float), c.index
    out, lo, hi, armed = [], 0, 0, False
    for i in range(1, len(v)):
        x = v[i]
        if not armed:
            if x < v[lo]:
                lo = hi = i
            elif x > v[hi]:
                hi = i
            if v[hi] / v[lo] - 1 >= RISE:
                armed = True
            elif x <= v[hi] * (1 - RISE_STOP):
                lo = hi = i  # 25% fall first: this low does not qualify; restart here
        else:
            if x > v[hi]:
                hi = i
            elif x <= v[hi] * (1 - RISE_STOP):
                out.append({"d": ix[lo], "end": ix[hi], "size": v[hi] / v[lo] - 1, "done": True})
                armed, lo, hi = False, i, i
    if armed:
        out.append({"d": ix[lo], "end": ix[hi], "size": v[hi] / v[lo] - 1, "done": False})
    for e in out:
        e["days"] = int((e["end"] - e["d"]).days)
        e["big"] = e["size"] >= RISE_BIG
    return [e for e in out if e["d"] >= pd.Timestamp(LIST_FROM)]


def _falls(close: pd.Series) -> list[dict]:
    c = close[close.index >= pd.Timestamp(SCAN_FROM)]
    v, ix = c.values.astype(float), c.index
    out, hi, lo, armed = [], 0, 0, False
    for i in range(1, len(v)):
        x = v[i]
        if not armed:
            if x > v[hi]:
                hi = lo = i
            elif x < v[lo]:
                lo = i
            if v[lo] / v[hi] - 1 <= -FALL:
                armed = True
            elif x >= v[lo] * (1 + FALL_STOP):
                hi = lo = i  # 25% rebound first: this high does not qualify; restart here
        else:
            if x < v[lo]:
                lo = i
            elif x >= v[lo] * (1 + FALL_STOP):
                out.append({"d": ix[hi], "end": ix[lo], "size": v[lo] / v[hi] - 1, "done": True})
                armed, hi, lo = False, i, i
    if armed:
        out.append({"d": ix[hi], "end": ix[lo], "size": v[lo] / v[hi] - 1, "done": False})
    for e in out:
        e["days"] = int((e["end"] - e["d"]).days)
        e["big"] = e["size"] <= -FALL_BIG
    return [e for e in out if e["d"] >= pd.Timestamp(LIST_FROM)]


# ---------- measurements ----------

def _bins_all(r: np.ndarray) -> np.ndarray:
    """(days x 12) mean rank over each 10-day bin relative to each day (at least 5 present)."""
    s = pd.Series(r)
    r10 = s.rolling(10, min_periods=5).mean().values
    n = len(r)
    out = np.full((n, len(BINS)), np.nan)
    for j, o in enumerate(BINS):
        end = np.arange(n) + o + 9
        ok = (end >= 0) & (end < n)
        out[ok, j] = r10[end[ok]]
    return out


def _deltas(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(r)
    a30 = s.shift(1).rolling(30, min_periods=20).mean()
    b60 = s.shift(31).rolling(60, min_periods=40).mean()
    a10 = s.shift(1).rolling(10, min_periods=7).mean()
    b30 = s.shift(11).rolling(30, min_periods=20).mean()
    return (a30 - b60).values, (a10 - b30).values


def _raw_means(x: np.ndarray, pos: int) -> tuple[float, float]:
    """mean of x over -90..-31 and -30..-1 (same minimum counts as D30)."""
    if pos < 90:
        return np.nan, np.nan
    a, b = x[pos - 90:pos - 30], x[pos - 30:pos]
    fa, fb_ = np.isfinite(a), np.isfinite(b)
    return (float(a[fa].mean()) if fa.sum() >= 40 else np.nan, float(b[fb_].mean()) if fb_.sum() >= 20 else np.nan)


def _sign(v: float) -> int:
    return 0 if not _ok(v) or v == 0 else (1 if v > 0 else -1)


PATTERNS = ("賣壓降＋買壓升", "兩邊都升", "兩邊都降", "賣壓升＋買壓降", "有一邊持平或缺資料")


def _pattern(ds: float, db: float) -> int:
    a, b = _sign(ds), _sign(db)
    if a == -1 and b == 1:
        return 0
    if a == 1 and b == 1:
        return 1
    if a == -1 and b == -1:
        return 2
    if a == 1 and b == -1:
        return 3
    return 4


def _compute(data: dict) -> dict:
    t0 = time.time()
    close = data["close"]
    idx = close.index
    n = len(idx)
    raw, rank, first = _pressures(data)
    ev = {"rise": _rises(close), "fall": _falls(close)}
    pos = {t: np.array([idx.get_loc(e["d"]) for e in ev[t]], dtype=int) for t in ev}
    # random pool
    lo_i = idx.get_loc(pd.Timestamp(LIST_FROM))
    hi_i = n - 31
    cand = np.arange(lo_i, hi_i + 1)
    allp = np.concatenate([pos["rise"], pos["fall"]])
    dist = np.min(np.abs(cand[:, None] - allp[None, :]), axis=1)
    pool = cand[dist >= GAP]
    pool_note = "兩種事件都避開"
    pool_t = {"rise": pool, "fall": pool}
    if len(pool) < 365:
        pool_note = "只避開同一種事件（兩種都避開的話剩不到 365 天）"
        for t in ("rise", "fall"):
            dd = np.min(np.abs(cand[:, None] - pos[t][None, :]), axis=1)
            pool_t[t] = cand[dd >= GAP]
    rng = np.random.default_rng(SEED)
    R: dict = {"raw": raw, "rank": rank, "first": first, "ev": ev, "pos": pos, "pool": pool_t, "pool_note": pool_note,
               "n_pool": {t: int(len(pool_t[t])) for t in pool_t}, "n_cand": int(len(cand))}
    bins, d30, d10, rawm = {}, {}, {}, {}
    for k in KEYS:
        r = rank[k].values.astype(float)
        bins[k] = _bins_all(r)
        d30[k], d10[k] = _deltas(r)
        x = raw[k].values.astype(float)
        rawm[k] = {t: np.array([_raw_means(x, p) for p in pos[t]]).reshape(-1, 2) for t in pos}
    R["d30"], R["d10"], R["rawm"] = d30, d10, rawm
    # step 1 paths
    path: dict = {}
    for t in ("rise", "fall"):
        P = pool_t[t]
        path[t] = {}
        for k in KEYS:
            B = bins[k]
            rows = []
            for j in range(len(BINS)):
                ev_v = B[pos[t], j]
                ev_v = ev_v[np.isfinite(ev_v)]
                pv = B[P, j]
                pv = pv[np.isfinite(pv)]
                if len(ev_v) == 0 or len(pv) < 30:
                    rows.append({"n": int(len(ev_v)), "m": np.nan, "c": np.nan, "lo": np.nan, "hi": np.nan})
                    continue
                dr = pv[rng.integers(0, len(pv), size=(DRAWS, len(ev_v)))].mean(axis=1)
                rows.append({"n": int(len(ev_v)), "m": float(ev_v.mean()), "c": float(pv.mean()),
                             "lo": float(np.percentile(dr, 10)), "hi": float(np.percentile(dr, 90))})
            path[t][k] = rows
    R["path"] = path
    # step 2 changes

    def change_stats(vals_ev: np.ndarray, vals_pool: np.ndarray) -> dict:
        e = vals_ev[np.isfinite(vals_ev)]
        p = vals_pool[np.isfinite(vals_pool)]
        out = {"n": int(len(e)), "up": int((e > 0).sum()), "down": int((e < 0).sum()),
               "med": float(np.median(e)) if len(e) else np.nan,
               "r_med": float(np.median(p)) if len(p) else np.nan,
               "r_up": float((p > 0).mean()) if len(p) else np.nan, "p": np.nan}
        if len(e) and len(p) >= 30:
            md = np.median(p[rng.integers(0, len(p), size=(DRAWS, len(e)))], axis=1)
            out["p"] = float((md >= out["med"]).mean() if out["med"] > 0 else (md <= out["med"]).mean())
        return out

    ch: dict = {}
    for t in ("rise", "fall"):
        P = pool_t[t]
        big = np.array([e["big"] for e in ev[t]], dtype=bool)
        ch[t] = {}
        for k in KEYS:
            ch[t][k] = {"d30": change_stats(d30[k][pos[t]], d30[k][P]),
                        "d10": change_stats(d10[k][pos[t]], d10[k][P]),
                        "big": change_stats(d30[k][pos[t][big]], d30[k][P]) if big.any() else None}
    R["ch"] = ch
    # step 3 per event composites and patterns
    c = close.values.astype(float)
    per: dict = {}
    tally: dict = {}
    for t in ("rise", "fall"):
        P = pool_t[t]
        rows = []
        for e, p in zip(ev[t], pos[t]):
            ds, db = d30["SELL_C"][p], d30["BUY_C"][p]
            px = c[p - 1] / c[p - 31] - 1 if p >= 31 else np.nan
            rows.append({"e": e, "ds": ds, "db": db, "dn": d30["NET30"][p], "da": d30["ACT"][p], "px": px,
                         "pat": _pattern(ds, db)})
        per[t] = rows
        cnt = np.bincount([r["pat"] for r in rows], minlength=len(PATTERNS))
        pp = np.array([_pattern(d30["SELL_C"][i], d30["BUY_C"][i]) for i in P])
        okp = pp < 4
        share = np.bincount(pp[okp], minlength=len(PATTERNS)) / max(int(okp.sum()), 1)
        nn = int(cnt[:4].sum())
        tally[t] = {"cnt": cnt.tolist(), "share": share.tolist(), "n": nn, "n_pool": int(okp.sum())}
    R["per"], R["tally"] = per, tally
    R["px_med"] = {t: float(np.nanmedian([r["px"] for r in per[t]])) for t in per}
    R["px_neg"] = {t: (int(sum(1 for r in per[t] if _ok(r["px"]) and r["px"] < 0)),
                       int(sum(1 for r in per[t] if _ok(r["px"])))) for t in per}
    other = {"rise": "fall", "fall": "rise"}
    R["overlap"] = {t: int(sum(1 for p in pos[t] if ((pos[other[t]] >= p - 90) & (pos[other[t]] < p)).any()))
                    for t in pos}
    # step 4 rises versus falls
    vs: dict = {}
    for k in KEYS:
        a = d30[k][pos["rise"]]
        b = d30[k][pos["fall"]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            vs[k] = {"na": int(len(a)), "nb": int(len(b)), "obs": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
                     "ma": np.nan, "mb": np.nan}
            continue
        obs = float(np.median(a) - np.median(b))
        allv = np.concatenate([a, b])
        perm = np.empty(DRAWS)
        for i in range(DRAWS):
            sh = rng.permutation(allv)
            perm[i] = np.median(sh[:len(a)]) - np.median(sh[len(a):])
        vs[k] = {"na": int(len(a)), "nb": int(len(b)), "obs": obs, "lo": float(np.percentile(perm, 5)),
                 "hi": float(np.percentile(perm, 95)), "p": float((np.abs(perm) >= abs(obs)).mean()),
                 "ma": float(np.median(a)), "mb": float(np.median(b))}
    R["vs"] = vs
    # step 5 first look (chosen after seeing step 2)
    cands = []
    for t in ("rise", "fall"):
        for k in KEYS:
            s = ch[t][k]["d30"]
            if s["n"] >= MIN_EV:
                kk = max(s["up"], s["down"])
                cands.append((kk / s["n"], s["n"], abs(s["med"]) if _ok(s["med"]) else 0.0, t, k))
    cands.sort(key=lambda z: (-z[0], -z[1], -z[2]))
    look = []
    ev_mark = {t: np.zeros(n, dtype=bool) for t in pos}
    for t in pos:
        ev_mark[t][pos[t]] = True
    for score, nn, _m, t, k in cands[:3]:
        s = ch[t][k]["d30"]
        sign = 1 if s["up"] >= s["down"] else -1
        thr = abs(s["med"])
        dv = d30[k]
        days = np.arange(lo_i, n - 30)
        okd = days[np.isfinite(dv[days])]
        cs = np.concatenate([[0], np.cumsum(ev_mark[t].astype(int))])
        follow = (cs[np.minimum(okd + 30, n)] - cs[okd]) > 0  # an event at day .. day+29
        sig = okd[(np.sign(dv[okd]) == sign) & (np.abs(dv[okd]) >= thr)]
        fsig = (cs[np.minimum(sig + 30, n)] - cs[sig]) > 0
        eps = int(1 + (np.diff(sig) > 30).sum()) if len(sig) else 0
        look.append({"t": t, "k": k, "score": score, "n": nn, "kk": max(s["up"], s["down"]), "sign": sign,
                     "thr": thr, "days": int(len(sig)), "eps": eps, "hit": float(fsig.mean()) if len(sig) else np.nan,
                     "base": float(follow.mean()) if len(okd) else np.nan, "base_days": int(len(okd))})
    R["look"], R["cands"] = look, cands[:6]
    # today
    last = n - 1
    today = {}
    for k in ("SELL_C", "BUY_C", "NET30", "ACT"):
        r = rank[k].values.astype(float)
        fv = np.flatnonzero(np.isfinite(r))
        li = int(fv[-1]) if len(fv) else None
        a = r[last - 29:last + 1]
        b = r[last - 89:last - 29]
        da = np.nanmean(a) if np.isfinite(a).sum() >= 20 else np.nan
        dbb = np.nanmean(b) if np.isfinite(b).sum() >= 40 else np.nan
        today[k] = {"rank": r[li] if li is not None else np.nan, "date": idx[li] if li is not None else None,
                    "chg": da - dbb, "raw": raw[k].values[li] if li is not None else np.nan}
    R["today"] = today
    R["secs"] = time.time() - t0
    return R


# ---------- text ----------

def _cell(row: dict) -> str:
    if not _ok(row["m"]):
        return "—"
    mark = " ▲" if row["m"] > row["hi"] else (" ▼" if row["m"] < row["lo"] else "")
    return f"{row['m']:.0f}{mark}（{row['lo']:.0f}–{row['hi']:.0f}）"


def _bin_name(o: int) -> str:
    return f"{o}～{o + 9}" if o < 0 else f"+{o}～+{o + 9}"


def _ev_table(t: str, R: dict) -> dict:
    rows = []
    for e in R["ev"][t]:
        big = ("是" if e["big"] else "否")
        rows.append([_d(e["d"]), _pc(e["size"], 1), f"{e['days']} 天", _d(e["end"]), big,
                     "完成" if e["done"] else "還在進行（到今天為止）"])
    nb = sum(e["big"] for e in R["ev"][t])
    if t == "rise":
        cap = (f"大漲起點（低點）：從低點漲了至少 50%，而且在那之前沒有從途中高點回跌 25%。共 {len(rows)} 次，"
               f"其中漲到 100% 以上的 {nb} 次。日期是低點那一天；這些日期都是事後才知道的。")
        cols = ["低點日期", "漲幅", "低點到高點", "高點日期", "漲 100% 以上", "狀態"]
    else:
        cap = (f"大跌起點（高點）：從高點跌了至少 30%，而且在那之前沒有從途中低點反彈 25%。共 {len(rows)} 次，"
               f"其中跌 50% 以上的 {nb} 次。日期是高點那一天；這些日期都是事後才知道的。")
        cols = ["高點日期", "跌幅", "高點到低點", "低點日期", "跌 50% 以上", "狀態"]
    return {"caption": cap, "columns": cols, "rows": rows}


def _path_tables(t: str, R: dict) -> list[dict]:
    out = []
    for half, sl in (("事件前 90 天到 31 天", slice(0, 6)), ("事件前 30 天到事件後 29 天", slice(6, 12))):
        cols = ["壓力（排名 0–100）"] + [f"第 {_bin_name(o)} 天" for o in BINS[sl]]
        rows = []
        for k in KEYS:
            ps = R["path"][t][k][sl]
            nmax = max(p["n"] for p in ps)
            rows.append([f"{LABEL[k]}（{nmax} 次）"] + [_cell(p) for p in ps])
        out.append({"caption": (f"1. {TYPE_CN[t]}前後，每種壓力的平均排名（{half}）。排名 0–100＝比自己過去一年多少 % 的日子高。"
                                f"括號＝隨機挑同樣次數的「非事件日」，2,000 次抽樣平均值的 10%–90% 範圍；"
                                f"▲＝事件平均高於這個範圍、▼＝低於。0 是事件那一天。"),
                    "columns": cols, "rows": rows})
    return out


def _change_rows(t: str, R: dict, which: str) -> list:
    rows = []
    for k in KEYS:
        s = R["ch"][t][k][which]
        if s["n"] == 0:
            rows.append([LABEL[k], "0", "—", "—", "—", "—", "—", "—"])
            continue
        rows.append([LABEL[k], str(s["n"]), str(s["up"]), str(s["down"]), _sg(s["med"], 1), _sg(s["r_med"], 1),
                     f"{s['r_up'] * 100:.0f}%" if _ok(s["r_up"]) else "—",
                     f"{s['p'] * 100:.0f}%" if _ok(s["p"]) else "—"])
    return rows


def _event_value_tables(t: str, R: dict) -> list[dict]:
    groups = (("賣方", ("S_dep7", "S_dep30", "S_old", "S_take", "S_imp", "SELL_C")),
              ("買方", ("B_wd7", "B_wd30", "B_stb", "B_etf", "B_take", "B_imp", "B_cb")),
              ("平衡、活躍、槓桿", ("NET7", "NET30", "TAKE", "IMPR", "ACT", "FUND", "OI")))
    out = []
    for gname, ks in groups:
        rows = []
        for e, p in zip(R["ev"][t], R["pos"][t]):
            rows.append([_d(e["d"])] + [_sg(R["d30"][k][p], 0) for k in ks])
        out.append({"caption": (f"2. 每一次{TYPE_CN[t]}前，{gname}各項的「變化」＝事件前 30 天平均排名 減 再往前 60 天"
                                f"（前 90～31 天）平均排名。正＝最近變高，負＝最近變低；—＝那時還沒有資料。"),
                    "columns": ["事件日期"] + [SHORT[k] for k in ks], "rows": rows})
    return out


def _sentence(t: str, k: str, s: dict) -> str:
    what = LABEL[k]
    if s["n"] == 0:
        return f"{what}：這些{TYPE_CN[t]}前都沒有資料。"
    flat = s["n"] - s["up"] - s["down"]
    tail = f"、{flat} 次幾乎不變" if flat else ""
    return (f"在 {s['n']} 次{TYPE_CN[t]}前，{what}有 {s['up']} 次上升、{s['down']} 次下降{tail}"
            f"（中位變化 {_sg(s['med'], 1)} 點；隨機非事件日上升的比例 {s['r_up'] * 100:.0f}%）。")


def _per_event_table(t: str, R: dict) -> dict:
    rows = []
    for r in R["per"][t]:
        rows.append([_d(r["e"]["d"]), _pc(r["e"]["size"]), _sg(r["ds"]), _sg(r["db"]), _sg(r["dn"]), _sg(r["da"]),
                     _pc(r["px"]), PATTERNS[r["pat"]]])
    return {"caption": (f"3. 每一次{TYPE_CN[t]}前 30 天，買賣兩邊一起看：賣方合計、買方合計、淨提走比例（30 天）、"
                        f"活躍度的排名變化（前 30 天平均 減 前 90～31 天平均），以及同樣 30 天的價格變化。"),
            "columns": ["事件日期", "幅度", "Δ賣方合計", "Δ買方合計", "ΔNET30", "Δ活躍度", "前30天價格", "型態"],
            "rows": rows}


def _tally_table(R: dict) -> dict:
    rows = []
    for j, name in enumerate(PATTERNS[:4]):
        row = [name]
        for t in ("rise", "fall"):
            ta = R["tally"][t]
            row += [f"{ta['cnt'][j]} / {ta['n']}", f"{ta['share'][j] * ta['n']:.1f}（{ta['share'][j] * 100:.0f}%）"]
        rows.append(row)
    rows.append([PATTERNS[4], str(R["tally"]["rise"]["cnt"][4]), "—", str(R["tally"]["fall"]["cnt"][4]), "—"])
    return {"caption": ("3. 型態統計：事件前 30 天，賣方合計和買方合計各自是升還是降。旁邊是隨機非事件日裡每種型態的比例，"
                        "換算成「同樣次數會有幾次」。"),
            "columns": ["型態", "大漲前（次數）", "隨機日換算", "大跌前（次數）", "隨機日換算"], "rows": rows}


def _vs_table(R: dict) -> dict:
    rows = []
    for k in KEYS:
        v = R["vs"][k]
        if not _ok(v["obs"]):
            rows.append([LABEL[k], f"{v['na']} / {v['nb']}", "—", "—", "—", "—", "資料太少"])
            continue
        out = v["obs"] > v["hi"] or v["obs"] < v["lo"]
        rows.append([LABEL[k], f"{v['na']} / {v['nb']}", _sg(v["ma"], 1), _sg(v["mb"], 1), _sg(v["obs"], 1),
                     f"{v['lo']:+.1f} ～ {v['hi']:+.1f}", "超出（不太像巧合）" if out else "在範圍內（分不出來）"])
    return {"caption": ("4. 大漲前 vs 大跌前：同一種壓力的「事件前 30 天變化」中位數相減。範圍＝把「大漲／大跌」標籤打亂 "
                        "2,000 次得到的 5%–95%；超出範圍表示兩種事件前的變化不太像只是巧合。注意：大漲起點是低點（之前通常在跌），"
                        "大跌起點是高點（之前通常在漲），所以差別可能只是反映之前的價格走勢。"),
            "columns": ["壓力", "次數 漲/跌", "大漲前中位", "大跌前中位", "相差", "打亂後 5%–95%", "判讀"], "rows": rows}


def _look_table(R: dict) -> dict:
    rows = []
    for lk in R["look"]:
        dirw = "上升" if lk["sign"] > 0 else "下降"
        rows.append([f"{TYPE_CN[lk['t']]}前：{LABEL[lk['k']]}{dirw}", f"{lk['kk']} / {lk['n']}",
                     f"{dirw}至少 {lk['thr']:.1f} 點", f"{lk['days']:,} 天（{lk['eps']} 段）",
                     f"{lk['hit'] * 100:.1f}%" if _ok(lk["hit"]) else "—",
                     f"{lk['base'] * 100:.1f}%" if _ok(lk["base"]) else "—"])
    return {"caption": ("5. 第一眼（看完第 2 步才挑的，不是檢驗）：第 2 步裡最一致的三個變化（依 k/N 排，同分取次數多的、"
                        "再取中位變化大的）。把 2012 年以來所有出現同方向、至少同樣大小變化的日子找出來，看之後 30 天內真的"
                        "開始一次同類事件的比例，和所有日子的平常比例比。用來看「這個樣子在沒事的時候是不是也很常見」。"),
            "columns": ["變化", "事件中幾次", "條件", "出現的日子", "之後 30 天內有事件", "平常比例"], "rows": rows}


def _coverage_table(R: dict) -> dict:
    rows = []
    for k in KEYS:
        f = R["first"][k]
        rows.append([LABEL[k], _d(f) if f is not None else "沒有",
                     f"{R['ch']['rise'][k]['d30']['n']} / {len(R['ev']['rise'])}",
                     f"{R['ch']['fall'][k]['d30']['n']} / {len(R['ev']['fall'])}", LIMITS[k]])
    return {"caption": "6. 每種壓力從哪天有排名、在多少次事件裡有資料、看不到什麼。",
            "columns": ["壓力", "排名從哪天起", "大漲有資料", "大跌有資料", "看不到什麼"], "rows": rows}


def _raw_table(R: dict) -> dict:
    rows = []
    for k in KEYS:
        if k in ("SELL_C", "BUY_C"):
            continue
        cells = []
        for t in ("rise", "fall"):
            m = R["rawm"][k][t]
            for j in (0, 1):
                v = m[:, j] if len(m) else np.array([])
                v = v[np.isfinite(v)]
                cells.append(_num(float(np.median(v)), UNIT[k]) if len(v) else "—")
        if k == "B_cb":  # the premium is a few hundredths of a percent: show more digits
            cells = []
            for t in ("rise", "fall"):
                m = R["rawm"][k][t]
                for j in (0, 1):
                    v = m[:, j] if len(m) else np.array([])
                    v = v[np.isfinite(v)]
                    cells.append(f"{float(np.median(v)):+.3f}" if len(v) else "—")
        rows.append([f"{LABEL[k]}（{UNIT[k]}）"] + cells)
    return {"caption": ("原始單位：每次事件先算「前 90～31 天平均」和「前 30 天平均」，再取所有事件的中位數。早年和近年的量級"
                        "差很多（例如送進交易所的幣、成交量），所以主要看排名；這張只是讓人知道實際大小。"),
            "columns": ["壓力（單位）", "大漲前 90～31 天", "大漲前 30 天", "大跌前 90～31 天", "大跌前 30 天"],
            "rows": rows}


def _late_table(t: str, R: dict) -> dict:
    return {"caption": (f"2b. 最後 10 天的變化（{TYPE_CN[t]}前）：前 10 天平均排名 減 前 40～11 天平均。看有沒有「臨門一腳」"
                        "的變化。欄位意思同上一張。"),
            "columns": ["壓力", "有資料", "上升", "下降", "中位變化", "隨機日中位", "隨機日上升比例", "隨機同次數中位至少這麼極端"],
            "rows": _change_rows(t, R, "d10")}


def _big_table(R: dict) -> dict:
    rows = []
    for k in KEYS:
        row = [LABEL[k]]
        for t in ("rise", "fall"):
            s = R["ch"][t][k]["big"]
            row.append("—" if not s or s["n"] == 0 else f"{s['up']} 升 / {s['down']} 降（共 {s['n']}），中位 {_sg(s['med'], 1)}")
        rows.append(row)
    return {"caption": "2c. 只看大的事件（漲 100% 以上、跌 50% 以上）：事件前 30 天的變化，升幾次、降幾次。",
            "columns": ["壓力", "漲 100% 以上之前", "跌 50% 以上之前"], "rows": rows}


def _top(R: dict, t: str, m: int = 3, which: str = "d30") -> list:
    out = []
    for k in KEYS:
        s = R["ch"][t][k][which]
        if s["n"] >= MIN_EV:
            kk = max(s["up"], s["down"])
            out.append((kk / s["n"], s["n"], abs(s["med"]) if _ok(s["med"]) else 0.0, k))
    out.sort(key=lambda z: (-z[0], -z[1], -z[2]))
    return [z[3] for z in out[:m]]


def _verdict(R: dict) -> tuple[str, list]:
    lines = []
    for t in ("rise", "fall"):
        parts = []
        for k in _top(R, t):
            s = R["ch"][t][k]["d30"]
            up = s["up"] >= s["down"]
            parts.append(f"{LABEL[k]}在 {s['n']} 次裡有 {max(s['up'], s['down'])} 次{'上升' if up else '下降'}"
                         f"（中位 {_sg(s['med'], 1)} 點；隨機日上升的比例 {s['r_up'] * 100:.0f}%）")
        lines.append(f"量到的（{TYPE_CN[t]}前 30 天比前 60 天，最一致的三項）：" + "；".join(parts) + "。")
    # added after seeing the results: the late (last 10 days) changes and the buy/sell pattern tally
    for t in ("rise", "fall"):
        parts = []
        for k in _top(R, t, 2, "d10"):
            s = R["ch"][t][k]["d10"]
            up = s["up"] >= s["down"]
            parts.append(f"{LABEL[k]} {s['n']} 次裡 {max(s['up'], s['down'])} 次{'上升' if up else '下降'}"
                         f"（中位 {_sg(s['med'], 1)} 點）")
        lines.append(f"量到的（{TYPE_CN[t]}前最後 10 天，最一致的兩項）：" + "；".join(parts) + "。")
    ta, tf = R["tally"]["rise"], R["tally"]["fall"]
    lines.append(f"量到的（買賣兩邊一起看）：大漲前 30 天「賣方合計、買方合計兩邊都升」{ta['cnt'][1]} / {ta['n']} 次"
                 f"（隨機日換算約 {ta['share'][1] * ta['n']:.1f} 次），「賣壓降＋買壓升」{ta['cnt'][0]} 次"
                 f"（隨機約 {ta['share'][0] * ta['n']:.1f} 次）；大跌前兩邊都升 {tf['cnt'][1]} / {tf['n']} 次"
                 f"（隨機約 {tf['share'][1] * tf['n']:.1f} 次）。兩邊一起升也可能只是市場變熱鬧（活躍度一起升），分不開。")
    diff = [k for k in KEYS if _ok(R["vs"][k]["obs"]) and (R["vs"][k]["obs"] > R["vs"][k]["hi"]
                                                         or R["vs"][k]["obs"] < R["vs"][k]["lo"])]
    same = [k for k in KEYS if _ok(R["vs"][k]["obs"]) and k not in diff]
    enough = lambda k: min(R["vs"][k]["na"], R["vs"][k]["nb"]) >= MIN_EV  # noqa: E731
    if diff:
        lines.append("量到的（大漲前 vs 大跌前不一樣，超出打亂標籤的 90% 範圍）：" + "；".join(
            f"{LABEL[k]}（大漲前中位 {_sg(R['vs'][k]['ma'], 1)}，大跌前 {_sg(R['vs'][k]['mb'], 1)}，"
            f"{R['vs'][k]['na']} / {R['vs'][k]['nb']} 次）" for k in diff)
            + "。這些差別也可能只是反映事件前的價格方向（大漲前價格多半在跌、大跌前多半在漲），這份資料分不開。")
    s_en = [SHORT[k] for k in same if enough(k)]
    s_few = [SHORT[k] for k in same if not enough(k)]
    lines.append(f"沒有看出不一樣的：{len(s_en)} 項事件夠多的（" + "、".join(s_en) + "）"
                 + (f"；另外 {len(s_few)} 項事件太少、本來就很難看出差別（" + "、".join(s_few) + "）" if s_few else "")
                 + "。")
    few = [k for k in KEYS if R["ch"]["rise"][k]["d30"]["n"] < MIN_EV or R["ch"]["fall"][k]["d30"]["n"] < MIN_EV]
    if few:
        lines.append("看不清楚（至少一種事件前有資料的次數不到 8 次）：" + "、".join(SHORT[k] for k in few)
                     + "。沒看到不代表沒有，只是這個定義和這份資料看不出來；缺的主要是 2018 年以前的幣安成交、"
                       "2020 年以前的穩定幣和資金費率、2024 年以前的 ETF，以及掛單深度、強制平倉、場外交易（完全沒有）。")
    lines += [f"推測（看完結果後才寫，都沒有檢驗過）：{g}" for g in _guesses(R)]
    return " ".join(lines), lines


def _now_rows(R: dict) -> list[dict]:
    names = {"SELL_C": "賣方合計", "BUY_C": "買方合計", "NET30": "淨提走比例（30 天）", "ACT": "活躍度（30 天）"}
    rows = []
    for k in ("SELL_C", "BUY_C", "NET30", "ACT"):
        tdy = R["today"][k]
        if not _ok(tdy["rank"]):
            rows.append({"label": names[k], "value": "沒有資料"})
            continue
        ch = tdy["chg"]
        word = ("比前 60 天高" if ch > 0 else "比前 60 天低") if _ok(ch) else "變化算不出來"
        extra = "" if k in ("SELL_C", "BUY_C") else f"，數值 {_num(tdy['raw'], UNIT[k])} {UNIT[k]}"
        rows.append({"label": f"{names[k]}（{_d(tdy['date'])}）",
                     "value": f"排名 {tdy['rank']:.0f}（比過去一年 {tdy['rank']:.0f}% 的日子高）{extra}；"
                              f"近 30 天平均{word} {abs(ch):.1f} 點" if _ok(ch) else f"排名 {tdy['rank']:.0f}{extra}"})
    return rows


def study() -> dict:
    t0 = time.time()
    data = _load()
    R = _compute(data)
    close = data["close"]
    verdict, vlines = _verdict(R)
    tables = [_ev_table("rise", R), _ev_table("fall", R)]
    tables += _path_tables("rise", R) + _path_tables("fall", R)
    for t in ("rise", "fall"):
        tables.append({"caption": (f"2. {TYPE_CN[t]}前 30 天的變化（前 30 天平均排名 減 前 90～31 天平均），總表。"
                                   f"「隨機同次數中位至少這麼極端」＝從隨機非事件日抽同樣次數、2,000 次，中位數至少和事件一樣"
                                   f"偏向同一邊的比例；越小越不像巧合。"),
                       "columns": ["壓力", "有資料", "上升", "下降", "中位變化", "隨機日中位", "隨機日上升比例",
                                   "隨機同次數中位至少這麼極端"],
                       "rows": _change_rows(t, R, "d30")})
        tables.append({"caption": f"2. 量到的句子（{TYPE_CN[t]}前 30 天比前 60 天）。",
                       "columns": ["壓力", "量到的"],
                       "rows": [[SHORT[k], _sentence(t, k, R["ch"][t][k]["d30"])] for k in KEYS]})
        tables += _event_value_tables(t, R)
        tables.append(_late_table(t, R))
    tables.append(_big_table(R))
    tables += [_per_event_table("rise", R), _per_event_table("fall", R), _tally_table(R), _vs_table(R),
               _look_table(R), _coverage_table(R), _raw_table(R)]
    now = [{"label": "最後一根收完的日線", "value": f"{_d(close.index[-1])}，比特幣 {close.iloc[-1]:,.0f} 美元"}]
    now += _now_rows(R)
    now.append({"label": "這是觀察研究", "value": "只描述大漲、大跌之前壓力怎麼變，不給買賣訊號"})
    notes = [
        "這份研究在做什麼：老闆要看「大漲前、大跌前，買壓和賣壓怎麼變」。以前的研究多半在測規則、算所有日子之後的平均報酬；"
        "這份只把每一次大漲、大跌之前，兩邊壓力的變化一次一次攤開來看。不測任何買賣規則。",
        "名詞：賣壓＝送進交易所的幣、老幣移動（放 155 天以上的幣被花掉）、主動賣出的量（吃掉別人買單的賣出）、"
        "賣出壓價力（每 1 億美元淨賣出讓價格跌多少）。買壓＝從交易所提走的幣、新增穩定幣、美國現貨 ETF 淨流入、"
        "主動買進的量、買進推價力、Coinbase 溢價（美國買家願意多付多少）。NET＝(提走−送進)÷(提走＋送進)，正的代表提走多。"
        "活躍度＝提走＋送進。排名 0–100＝今天的數值比自己過去一年多少 % 的日子高。",
        "事件的日期（低點、高點）都是事後才知道的。這份研究只描述「事前壓力長什麼樣」，不代表當時能據此交易。",
        f"隨機非事件日：2012 年起、離每一次事件（{R['pool_note']}）至少 60 天、而且之後還有 30 天資料的日子，"
        f"共 {R['n_pool']['rise']:,} 天（候選 {R['n_cand']:,} 天）。用來回答「這樣的變化，沒事的時候也常見嗎」。",
        "怎麼讀「變化」：事件前 30 天平均排名 減 再往前 60 天平均排名。例如 −15 表示最近一個月比前兩個月低了 15 個百分位。",
        "大漲起點是低點，之前價格多半在跌；大跌起點是高點，之前價格多半在漲。很多壓力本身就跟價格一起動"
        "（例如主動買進比例、淨提走、溢價），所以「事件前變了」和「之前價格的走勢讓它變了」這兩種解釋，這份資料分不開。",
        "主動買和主動賣的量都會隨成交量一起升降，所以「兩邊都升」常常只是市場變熱鬧；第 3 步把活躍度放在旁邊就是為了這個。",
        "時間對齊：Coin Metrics、DefiLlama、ETF 都晚 1 天才用；老幣晚 7 天；幣安成交、溢價、資金費率當天收盤就用；"
        "未平倉量用 oi_flush 的對齊（00:00 那一筆配前一根日線收盤）。",
        "幣安零手續費期間（2022-07-08～2023-03-22）成交量被灌大，所以 2022-07-08～2023-04-04 的主動買賣、壓價力、推價力都拿掉；"
        "這段期間的事件在這幾項沒有資料，之後排名也要等累積夠 180 天才恢復。",
        "第 5 步是看完第 2 步才挑的，只是「第一眼」，不是檢驗；同一批資料挑出來又拿同一批資料看，會偏樂觀。",
        "所有門檻（50%／100% 漲、30%／50% 跌、25% 反向、365 天排名、10 天一格、60 天距離、2,000 次抽樣）都在跑之前定好，"
        "沒有試別的數字。",
        f"量到的：大漲前 30 天價格中位 {_pc(R['px_med']['rise'])}（{R['px_neg']['rise'][0]} / {R['px_neg']['rise'][1]} 次是跌的）；"
        f"大跌前 30 天價格中位 {_pc(R['px_med']['fall'])}（{R['px_neg']['fall'][0]} / {R['px_neg']['fall'][1]} 次是跌的）。"
        "所以「大漲前 vs 大跌前」的差別，很可能有一部分只是價格方向不同造成的。",
        f"事件會靠在一起：{len(R['ev']['rise'])} 次大漲裡有 {R['overlap']['rise']} 次，前 90 天內至少有一次大跌起點；"
        f"{len(R['ev']['fall'])} 次大跌裡有 {R['overlap']['fall']} 次，前 90 天內至少有一次大漲起點。"
        "所以「大漲前」的窗口常常包著前一次大跌（例如 2013-04），兩種事件的「事前」不是完全分開的。",
        "賣方合計、買方合計的組成會變：2017-08 以前賣方合計只有送進交易所和老幣、買方合計只有提走；送進和提走本身就一起升降"
        "（都跟活躍度有關），所以早年「兩邊都升／都降」有一部分是算法本身造成的。隨機日也用同樣算法，所以要看的是和隨機日比。",
        "後來加的（看完結果才加）：判讀裡「最後 10 天最一致的兩項」「買賣兩邊型態」兩段、三個推測，以及數字格式（漲跌幅多一位小數、溢價多三位）；定義和門檻都沒有改。",
        f"這次計算用了 {time.time() - t0:.0f} 秒。",
    ] + [f"資料狀況：{p}" for p in data["problems"]]
    sources = [
        {"name": "Coin Metrics 社群 API：比特幣 FlowInExNtv、FlowOutExNtv（送進／提出交易所的幣）",
         "url": fb.CM_URL + "?assets=btc&metrics=" + ",".join(fb.CM_METRICS) + "&frequency=1d"},
        {"name": "Open Bitcoin Metrics：放 155 天以上的幣被花掉（只下載 CSV 資料檔）", "url": ocs.OBM_BASE.format(OBM_ID)},
        {"name": "幣安現貨 BTCUSDT 1 小時線（主動買進量）", "url": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"},
        {"name": "DefiLlama 穩定幣總供給 API", "url": sd.LLAMA},
        {"name": "Farside Investors：美國現貨比特幣 ETF 每日淨流入", "url": fb.FARSIDE_URL},
        {"name": "Coinbase Exchange 公開日線 BTC-USD", "url": sd.CB_URL + "?granularity=86400"},
        {"name": "Bitstamp 公開 API：BTC/USD 日線", "url": sd.BS_URL + "?step=86400&limit=1000"},
        {"name": "幣安 U 本位永續合約資金費率 BTCUSDT", "url": fw.BINANCE_URL},
        {"name": "幣安公開歷史資料 data.binance.vision：U 本位合約每日 metrics 檔（未平倉量）",
         "url": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/"},
        {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安公告：零手續費比特幣交易開始", "url": pi.URL_LAUNCH},
        {"name": "幣安公告：零手續費比特幣交易結束", "url": pi.URL_END},
    ]
    return {"id": "pressure_before_moves",
            "title": "大漲、大跌之前，買壓和賣壓怎麼變（觀察）",
            "question": ("在每一次大漲開始（低點）和大跌開始（高點）之前的幾週，賣壓（送進交易所的幣、老幣移動、主動賣出）"
                         "和買壓（從交易所提走的幣、新增穩定幣、ETF 流入、主動買進）各自怎麼變？一次一次攤開來看，不測規則。"),
            "asof": _d(close.index[-1]),
            "verdict": verdict,
            "now": now, "tables": tables, "notes": notes, "sources": sources}


def _guesses(R: dict) -> list[str]:
    """Candidate ideas (推測), written AFTER seeing the results; numbers are filled in from the current run.
    None of them is claimed to work; each is only a candidate for a pre-registered test."""
    so = R["ch"]["rise"]["S_old"]["d30"]
    nt = R["ch"]["rise"]["NET30"]["d10"]
    ta, tf = R["tally"]["rise"], R["tally"]["fall"]
    return [
        f"老幣移動在大漲低點前一個月變多（{so['up']} / {so['n']} 次上升，中位 {_sg(so['med'], 1)} 點）。可能是長期持有者在"
        f"低點附近交出籌碼，也可能只是大跌時託管、交易所搬幣變多，這份資料分不開。值得事先定好再測：老幣移動排名 30 天升幅大、"
        f"而且價格已經大跌的日子，之後 30–90 天和「只有價格大跌、老幣沒變多」的日子比。",
        f"大漲低點前最後 10 天，淨提走比例（30 天）變高（{nt['up']} / {nt['n']} 次上升，中位 {_sg(nt['med'], 1)} 點）。"
        f"可能是低點附近有人把幣提走自己保管，也可能只是恐慌時送進交易所的量先退掉。值得測：淨提走比例的 10 天變化，"
        f"一樣要有價格對照。",
        f"大漲前（{ta['cnt'][1]} / {ta['n']}）和大跌前（{tf['cnt'][1]} / {tf['n']}）都常見「買賣兩邊一起升」，"
        f"所以單看哪一邊升，分不出接下來是漲還是跌。如果要用，可能要看兩邊的差（淨提走比例、推價力÷壓價力），"
        f"值得測：在「兩邊一起升」的日子裡，這些差能不能分出之後的方向。",
    ]
