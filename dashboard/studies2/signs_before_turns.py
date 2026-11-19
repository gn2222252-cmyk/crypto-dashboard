"""Observation study "signs_before_turns": before the market turned bear and before it turned bull - and before the
big falls and big rises - what did everything the project can measure look like beforehand? Were there clues?

This is an OBSERVATION study at the owner's request. No trading rule is tested. Earlier studies took one indicator,
guessed a rule and scored forward returns over all days; this one anchors on the turns themselves and lays out, turn
by turn, every series the project has (except the buying / selling pressure measures - exchange deposits and
withdrawals, old coins moving, taker buy/sell, price impact, net flow - which the companion study
pressure_before_moves covers).

Definitions (written before any result was seen; anything added after seeing results is labelled so):

Price and turns
  * Price: the dashboard's daily close (te._daily_close: blockchain.info to 2017, then Binance BTCUSDT; UTC 00:00).
  * Turns are found mechanically from daily closes. Turn dates are known only AFTERWARDS (a top is a top only after
    the fall). This study describes what things looked like beforehand; it does not claim anything is tradable.
  * Fall scan (D, U): a candidate high H and the running low L since H. A higher close makes it the new H. If a close
    is <= H x (1 - D) before any close >= L x (1 + U), H is a turn ("fall start"); the fall then runs until a close
    >= (running low) x (1 + U), and the next candidate H starts at that close. If instead a close >= L x (1 + U)
    comes first, H is dropped and the candidate restarts at that close.
  * Rise scan (U, D): the mirror: a candidate low L and the running high H since L; L is a turn ("rise start") if a
    close >= L x (1 + U) comes before any close <= H x (1 - D); the rise runs until a close <= (running high) x (1 - D).
  * CYCLE turns: bear start = fall scan (D = 50%, U = 100%); bull start = rise scan (U = 100%, D = 50%).
  * SWING turns: big-fall start = fall scan (D = 30%, U = 25%); big-rise start = rise scan (U = 50%, D = 25%).
  * Size = turn close to the extreme reached before the reversal threshold; a move still running today is marked.

Series (each lagged to when it was really known; value at the close labelled t = data of day t - lag)
  Valuation: MVRV (Coin Metrics CapMVRVCur, lag 1); Puell multiple (miner issuance USD / its 365-day mean, puell.py,
  lag 1); power-law z (power_law._pit, point-in-time fit, lag 0); close / short-term-holder cost basis - 1
  (Checkonchain, sth_cost.py, lag 1, latest vintage = revised); close / 200-day and 50-day average - 1; close / 365-day
  high - 1; close / 365-day low - 1.
  Trend and volatility: 30-day realised volatility; ADX(14) (adx_breakout._btc_frame, from 2014-06); days since the
  last 365-day closing high.
  Network (Coin Metrics, 7-day means, lag 1): active addresses, transaction count, hash rate; fees in USD
  (FeeTotNtv x PriceUSD, blockspace.py).
  Leverage and positioning: Binance BTCUSDT funding 7-day mean (exit_forces._funding, lag 0); Binance open interest
  in coins, retail long/short ratio (all accounts) and top-trader position ratio, 7-day means (ls_ratio.py archive
  rows stamped 00:00 UTC of day D, used from the close labelled D); Deribit quarterly futures annualised basis
  (basis_filter.py, lag 0).
  Options: Deribit DVOL implied volatility (vrp_direction._dvol, lag 0); 25-delta-like skew from Deribit trades
  (skew_history.py stored copy, every 4 days, carried up to 4 days, lag 0); variance risk premium = DVOL - 30-day
  realised volatility (in %).
  Premiums (kimchi_premium.py, 7-day means, lag 0): Coinbase versus the dashboard close; Upbit (Korea) versus the
  dashboard close in USD.
  Stablecoins and ETFs as levels: total stablecoin supply (DefiLlama, lag 1); stablecoin supply / BTC market cap
  (lag 1); cumulative US spot BTC ETF net flow since launch (Farside via etf_loop._farside, lag 1).
  Attention: English Wikipedia "Bitcoin" page views, 7-day mean (attention.py, lag 1).
  Breadth: share of the top-50 Binance coins above their 200-day average (breadth_timing._breadth, lag 0); ETH/BTC
  price ratio (alt strength, lag 0).
  Macro and cross-market (lag 0 for US daily closes - they finish before 00:00 UTC; carried over weekends up to 5
  days): dollar index DX-Y.NYB, 10-year yield ^TNX, TIP price (inflation-protected Treasury ETF; higher = lower real
  yields), VIX, QQQ / its 200-trading-day average - 1, SPY / its 200-trading-day average - 1, HYG/IEF, LQD/IEF, gold
  futures GC=F (Yahoo via stress_family._fetch); NFCI (weekly, used 6 days after its date); OECD US composite leading
  indicator (monthly, month m used from the last day of month m + 2, revised data); Fed net liquidity (assets - TGA -
  reverse repo, Wednesday W used from W + 2 days, net_liquidity._piece).
  Calendar: days since the last halving - shown as raw days only (it is a clock, not ranked).
  A series whose loader fails is dropped and named, never substituted.

Rank
  Every day's value is ranked (0-100) against its own trailing 730 days (attention._trail_pct: share of the last 730
  values, today included, that are <= today's), at least 365 values needed. Top fifth = rank >= 80, bottom fifth =
  rank <= 20. "High" therefore means high relative to the last two years, not high in absolute terms.

What is laid out, separately for the four turn types (bear start, bull start, big-fall start, big-rise start)
  1. LEVEL: mean rank over days -30..-1 (turn close = day 0; at least 20 of 30 values). Per series: per cycle turn
     (one column each, with the raw value known at day -1), count of turns in the top fifth and in the bottom fifth
     (mean rank >= 80 / <= 20), and the same count expected for random days. Random days: 2,000 draws (fixed seed;
     with replacement only if the pool is smaller) from days at least 90 days away from every turn OF THE SAME SCALE
     (cycle or swing) and at least 605 days after the price history starts; per series only draws where it has a
     value. Consistency = k / N with k = the larger of the two extreme counts (ties: the side with the larger excess
     over random). Series are sorted by consistency, then larger N, then larger excess over random, then name. A
     series needs N >= 3 turns with data to be ranked. Also shown: the binomial chance of seeing >= k of N if each
     turn were a random day (a description of surprise, not a test; turns are far apart so treated as independent).
     Swing turns: the k-of-N summary only.
  2. CHANGE: (mean rank -60..-1) - (mean rank -240..-61), at least 40 / 120 values: turns where it rose / fell, the
     median change, versus random days (median and share rising). Late version: (-20..-1) - (-80..-21), at least
     14 / 40 values.
  3. ORDER: for series in the same extreme fifth before at least half of the turns (N >= 3): per turn, the 30-day
     trailing mean rank's continuous stay in that fifth that includes day -1 (looked back up to 730 days) gives how
     many days before the turn it began; if it was not in the fifth at day -1, the first day in it from day 0 to +90
     is shown as "after"; else "not". Median of the "before" leads. (The 30-day mean itself trails by about 15 days.)
  4. BULL versus BEAR: per series, median level (and median change) before bear starts minus before bull starts
     (cycle), and before big-fall minus big-rise starts (swing); 5-95% range of the same difference over 2,000
     shuffles of the turn-type labels (seed fixed) and the share of shuffles at least as far from zero. A series
     whose difference stays inside that range AND that sat in the same extreme fifth before at least half of both
     types is described as "near a big move, direction not separated".
  5. HOW OFTEN IT WOULD MISLEAD (a first look chosen after seeing step 1, not a test): for the ten most consistent
     series of each type (step 1 order), over all days with a value: share of days with the 30-day trailing mean rank
     in the same extreme fifth; onsets = first such day after at least 60 days without it; share of onsets followed
     by a turn of that type within 1..90 days versus the share of all days followed by one (the last 90 days are left
     out of both).
  6. COMBINATIONS (observation only): for each turn, how many of the ten were in their fifth at day -1 (among those
     with data); for all days of the step-1 random pool, the share and number of days with at least as many present.
  7. Coverage: first day each series has a rank, and with how many turns of each type.
  8. Today: the ten most consistent of each type, where they stand at the latest close. A description, not a
     forecast; no rule state.

Caveats stated in the notes: price-based series (distance to the 365-day high, the 200-day average ...) are tied to
the turn definition itself (a top is by definition a high price); many series overlap (MVRV, power-law z and the
200-day distance largely move together), so "many signs together" is not many independent clues; with about 40
series some will look consistent by chance; on-chain metrics and the leading indicator are revised; one venue for
derivatives (Binance / Deribit).
"""
from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import flows2
from dashboard.collectors import onchain2
from dashboard.collectors import price as px
from dashboard.studies2 import adx_breakout as adxm
from dashboard.studies2 import attention as at
from dashboard.studies2 import basis_filter as bf
from dashboard.studies2 import blockspace as bs
from dashboard.studies2 import breadth_timing as bt
from dashboard.studies2 import business_cycle as bc
from dashboard.studies2 import carry as cy
from dashboard.studies2 import etf_loop as el
from dashboard.studies2 import exit_forces as xf
from dashboard.studies2 import kimchi_premium as km
from dashboard.studies2 import ls_ratio as lr
from dashboard.studies2 import net_liquidity as nliq
from dashboard.studies2 import network_growth as ng
from dashboard.studies2 import nfci_filter as nf
from dashboard.studies2 import power_law as pl
from dashboard.studies2 import puell as pu
from dashboard.studies2 import rotation as ro
from dashboard.studies2 import skew_history as sk
from dashboard.studies2 import stablecoin_supply as ss
from dashboard.studies2 import sth_cost as sc
from dashboard.studies2 import stress_family as sf
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import vrp_direction as vd

ID = "signs_before_turns"
DAY = pd.Timedelta(days=1)
WIN, WIN_MIN = 730, 365
TOP, BOT = 80.0, 20.0
DRAWS, AWAY, SEED = 2000, 90, 20261119
SHUFFLES = 2000
FOLLOW, ABSENT = 90, 60
MIN_N = 3
TEN = 10
CYCLE = {"bear": ("fall", 0.50, 1.00), "bull": ("rise", 1.00, 0.50)}
SWING = {"fall": ("fall", 0.30, 0.25), "rise": ("rise", 0.50, 0.25)}
TYPES = ("bear", "bull", "fall", "rise")
SCALE = {"bear": "cycle", "bull": "cycle", "fall": "swing", "rise": "swing"}
TNAME = {"bear": "走熊開始（之後跌 50% 以上）", "bull": "走牛開始（之後漲 100% 以上）",
         "fall": "大跌開始（之後跌 30% 以上）", "rise": "大漲開始（之後漲 50% 以上）"}
TSHORT = {"bear": "走熊", "bull": "走牛", "fall": "大跌", "rise": "大漲"}

REAL_STORE = common.STORE
MY_STORE = REAL_STORE / ID
SEED_FILES = ("blockspace_cm.json", "blockspace_puell_cm.json", "netgrowth_btc.json", "sth_cost_coc.json",
              "exit_forces_funding.json", "ls_ratio_btcusdt.json", "vrp_direction_dvol_btc.json",
              "basis_filter_contracts.json", "kimchi_premium_upbit.json", "kimchi_premium_coinbase.json",
              "kimchi_premium_usdkrw.json", "stablecoin_supply_llama.json", "etf_loop_btc_frame.json",
              "etf_flows_btc.json", "attention_btc.json", "nfci_filter_nfci.json", "business_cycle_oecd.json",
              "net_liquidity_assets.json", "net_liquidity_tga.json", "net_liquidity_rrp.json",
              "thesis_four_bitstamp.json")
REMAP = ((km, "STORE_UP"), (km, "STORE_CB"), (km, "STORE_FX"), (bf, "STORE_FILE"), (nf, "STORE_PATH"))

# key: (short name, plain gloss, group, lag text, raw format)
META = {
    "mvrv": ("MVRV", "市值除以「每顆幣上次移動時價格」加總的成本；越高代表持有人帳面賺越多", "估值", "1 天", "x2"),
    "puell": ("Puell 倍數", "礦工每天新挖到的幣值除以過去一年平均；越高代表礦工收入特別好", "估值", "1 天", "x2"),
    "plz": ("冪次法則偏離", "價格比長期冪次趨勢線貴幾個標準差（只用當天以前的價格配線）", "估值", "0 天", "x2"),
    "sth": ("離短期持有人成本", "價格比「155 天內買進的人」平均成本高多少", "估值", "1 天", "pct"),
    "ma200": ("離 200 日均線", "價格比 200 日平均高多少", "估值", "0 天", "pct"),
    "ma50": ("離 50 日均線", "價格比 50 日平均高多少", "估值", "0 天", "pct"),
    "hi365": ("離一年最高", "價格比過去 365 天最高收盤低多少（0 = 正在創一年新高）", "估值", "0 天", "pct"),
    "lo365": ("離一年最低", "價格比過去 365 天最低收盤高多少", "估值", "0 天", "pct"),
    "rv30": ("30 天實際波動", "過去 30 天每日漲跌的年化波動", "趨勢與波動", "0 天", "pctu"),
    "adx": ("ADX 趨勢強度", "14 天 ADX：趨勢越明顯（不分漲跌）數字越大", "趨勢與波動", "0 天", "x1"),
    "since_hi": ("距上次一年新高天數", "離上一次創 365 天收盤新高過了幾天", "趨勢與波動", "0 天", "int"),
    "adr": ("活躍地址", "每天有動用的比特幣地址數（7 天平均）", "鏈上網路", "1 天", "big"),
    "tx": ("交易筆數", "每天鏈上交易筆數（7 天平均）", "鏈上網路", "1 天", "big"),
    "hash": ("算力", "全網挖礦算力（7 天平均）", "鏈上網路", "1 天", "big"),
    "fee": ("手續費（美元）", "每天付給礦工的手續費總額（7 天平均）", "鏈上網路", "1 天", "big"),
    "funding": ("資金費率", "幣安永續合約做多的人付給做空的人的費率（7 天平均）；越高代表做多越擠", "槓桿與部位", "0 天", "fund"),
    "oi": ("未平倉量", "幣安 BTCUSDT 合約還沒平掉的部位（顆數，7 天平均）", "槓桿與部位", "0 天", "big"),
    "ls_r": ("散戶多空比", "幣安所有帳戶裡做多帳戶數除以做空帳戶數（7 天平均）", "槓桿與部位", "0 天", "x2"),
    "ls_t": ("大戶多空比", "幣安前 20% 大帳戶的多單部位除以空單部位（7 天平均）", "槓桿與部位", "0 天", "x2"),
    "basis": ("期貨基差", "Deribit 季度期貨比現貨貴多少（年化）；越高代表願意付越多錢做多", "槓桿與部位", "0 天", "pp"),
    "dvol": ("隱含波動", "Deribit 選擇權反映的未來 30 天預期波動（DVOL）", "選擇權", "0 天", "pp"),
    "skew": ("選擇權偏斜", "賣權比買權貴多少（波動點數）；越高代表越多人花錢買跌的保險", "選擇權", "0 天", "pp"),
    "vrp": ("波動風險溢酬", "隱含波動減實際波動；越高代表選擇權越貴", "選擇權", "0 天", "pp"),
    "cbp": ("Coinbase 溢價", "美國 Coinbase 價格比國際價格貴多少（7 天平均）", "溢價", "0 天", "pct2"),
    "kimchi": ("韓國溢價", "韓國 Upbit 價格換成美元後比國際價格貴多少（7 天平均）", "溢價", "0 天", "pct2"),
    "stable": ("穩定幣總量", "所有美元穩定幣的發行總量", "穩定幣與 ETF", "1 天", "big"),
    "ssr": ("穩定幣／比特幣市值", "穩定幣總量除以比特幣市值；越高代表場邊現金相對比特幣越多", "穩定幣與 ETF", "1 天", "pctu"),
    "etf": ("ETF 累計淨流入", "美國現貨比特幣 ETF 從上市起累計流入的錢", "穩定幣與 ETF", "1 天", "big"),
    "wiki": ("維基瀏覽量", "英文維基「Bitcoin」頁每天瀏覽人數（7 天平均），代表關注度", "關注度", "1 天", "big"),
    "b200": ("幣種廣度", "幣安前 50 大幣種裡，價格在 200 日均線之上的比例", "廣度", "0 天", "pctu"),
    "ethbtc": ("以太幣／比特幣", "以太幣價格除以比特幣價格；越高代表山寨幣相對強", "廣度", "0 天", "x4"),
    "dxy": ("美元指數", "美元對一籃子主要貨幣的強弱", "總經與跨市場", "0 天", "x1"),
    "tnx": ("10 年美債殖利率", "美國 10 年期公債殖利率（%）", "總經與跨市場", "0 天", "x2"),
    "tip": ("抗通膨公債價格", "TIP ETF 價格；越高代表實質利率越低", "總經與跨市場", "0 天", "x1"),
    "vix": ("VIX 恐慌指數", "美股選擇權反映的未來 30 天預期波動", "總經與跨市場", "0 天", "x1"),
    "qqq": ("那斯達克趨勢", "QQQ 比它的 200 個交易日平均高多少", "總經與跨市場", "0 天", "pct"),
    "spy": ("標普 500 趨勢", "SPY 比它的 200 個交易日平均高多少", "總經與跨市場", "0 天", "pct"),
    "hyg_ief": ("垃圾債／公債", "高收益債 ETF 除以 7–10 年公債 ETF；越高代表市場越敢冒信用風險", "總經與跨市場", "0 天", "x3"),
    "lqd_ief": ("公司債／公債", "投資等級公司債 ETF 除以 7–10 年公債 ETF", "總經與跨市場", "0 天", "x3"),
    "gold": ("黃金價格", "黃金期貨價格", "總經與跨市場", "0 天", "x0"),
    "nfci": ("金融條件指數 NFCI", "芝加哥聯準銀行金融條件指數；越高代表借錢越難、越緊", "總經與跨市場", "約 6 天", "x2"),
    "oecd": ("OECD 領先指標", "OECD 美國綜合領先指標；越高代表景氣領先指標越好（數字事後會修正）", "總經與跨市場", "約 2 個月", "x1"),
    "netliq": ("聯準會淨流動性", "聯準會總資產減財政部帳戶減逆回購", "總經與跨市場", "2 天", "big"),
}
HALVING_NOTE = "距上次減半天數（日曆位置，不排名次）"

_d = te._d


# ---------- small helpers ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _fmt(v, kind: str) -> str:
    if not _ok(v):
        return "—"
    v = float(v)
    if kind == "pct":
        return f"{v * 100:+.0f}%"
    if kind == "pctu":
        return f"{v * 100:.0f}%"
    if kind == "pct2":
        return f"{v * 100:+.1f}%"
    if kind == "fund":
        return f"{v * 100:+.3f}%"
    if kind == "pp":
        return f"{v:.0f}"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "big":
        a = abs(v)
        if a >= 1e12:
            return f"{v / 1e12:.1f} 兆"
        if a >= 1e8:
            return f"{v / 1e8:,.0f} 億"
        if a >= 1e4:
            return f"{v / 1e4:,.0f} 萬"
        return f"{v:,.0f}"
    if kind.startswith("x"):
        return f"{v:.{int(kind[1:])}f}"
    return f"{v:.2f}"


def _r(v) -> str:
    return f"{float(v):.0f}" if _ok(v) else "—"


def _sp(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _of(k: int, n: int) -> str:
    return f"{n} 次裡 {k} 次"


def _binom_tail(k: int, n: int, p: float) -> float | None:
    if n <= 0 or not _ok(p):
        return None
    p = min(max(float(p), 0.0), 1.0)
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


# ---------- own store (writes only into this study's folder) ----------

@contextmanager
def _own_store():
    """Loaders write their copies into dashboard/cache/_store/signs_before_turns/ instead of the shared store.

    The shared store is only read: before the run each file a loader may fall back to is copied in (when newer)."""
    MY_STORE.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        src, dst = REAL_STORE / name, MY_STORE / name
        try:
            if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        except Exception:  # noqa: BLE001 - a missing seed only means the loader must download
            pass
    saved = [(m, a, getattr(m, a)) for m, a in REMAP]
    old = common.STORE
    common.STORE = MY_STORE
    for m, a, p in saved:
        setattr(m, a, MY_STORE / Path(p).name)
    try:
        yield
    finally:
        common.STORE = old
        for m, a, p in saved:
            setattr(m, a, p)


def _mine(name: str, fetch, fallback: Path | None = None) -> tuple[dict, str | None]:
    """fetch() -> {YYYY-MM-DD: number}; stored in this study's folder; on failure this copy, then `fallback` (read)."""
    path = MY_STORE / f"{name}.json"
    try:
        got = fetch()
        if not got:
            raise RuntimeError("空的")
        try:
            MY_STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(got.items()))), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return got, None
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        for p in (path, fallback):
            if p is not None and p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8")), \
                        f"{name} 這次取不到（{type(exc).__name__}），用之前存下來的資料。"
                except Exception:  # noqa: BLE001
                    continue
        raise


def _ser(d: dict) -> pd.Series:
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in d.items() if _ok(v)}, dtype=float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _known(s: pd.Series, idx: pd.DatetimeIndex, lag_days: int, limit: int) -> pd.Series:
    """Data-day series -> value known at each close label (data day + lag), carried at most `limit` days."""
    s = s.dropna()
    if s.empty:
        return pd.Series(np.nan, index=idx)
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).normalize() + lag_days * DAY
    s = s[~s.index.duplicated(keep="last")].sort_index()
    days = pd.date_range(min(s.index[0], idx[0]), max(s.index[-1], idx[-1]), freq="D")
    out = s.reindex(days).ffill(limit=limit).reindex(idx)
    return out.where(idx >= s.index[0])


# ---------- data ----------

def _assemble(close: pd.Series) -> tuple[dict, dict, list, dict, dict]:
    """{key: series known at each close}, {dropped key: reason}, problems, sources {name: url}, raw extras."""
    idx = close.index
    out, dropped, problems, src, extra = {}, {}, [], {}, {}

    def drop(keys, exc):
        for k in keys:
            dropped[k] = f"{type(exc).__name__}: {str(exc)[:90]}"

    c = close
    # price-only series
    out["ma200"] = c / c.rolling(200, min_periods=200).mean() - 1
    out["ma50"] = c / c.rolling(50, min_periods=50).mean() - 1
    hi = c.rolling(365, min_periods=365).max()
    out["hi365"] = c / hi - 1
    out["lo365"] = c / c.rolling(365, min_periods=365).min() - 1
    lr_ = np.log(c).diff()
    out["rv30"] = lr_.rolling(30, min_periods=30).std(ddof=1) * math.sqrt(365)
    is_hi = (c >= hi).values & hi.notna().values
    since, last = np.full(len(c), np.nan), None
    for i in range(len(c)):
        if is_hi[i]:
            last = i
        if last is not None:
            since[i] = i - last
    out["since_hi"] = pd.Series(since, index=idx)
    z = pl._pit(pl._ages(idx, pl.GENESIS), c.values, idx, pl.FIT_FROM)["z"]
    out["plz"] = pd.Series(z, index=idx)
    src["區塊鏈資訊站（blockchain.info）長期價格"] = classic.BCI_PRICE
    src["幣安現貨日線（2017 起）"] = px.BINANCE_HOSTS[0] + "/api/v3/klines?symbol=BTCUSDT&interval=1d"
    # halving clock (raw only)
    hv = pd.to_datetime(list(pu.HALVINGS))
    pos = np.searchsorted(hv.values, idx.values, side="right") - 1
    extra["halving"] = pd.Series(np.where(pos >= 0, (idx - hv[np.clip(pos, 0, None)]).days, np.nan), index=idx)

    # MVRV and market cap
    try:
        def f_mvrv():
            fr = onchain2._cm("BTC")
            return {k: [float(a), float(b)] for k, a, b in zip(fr.index, fr["CapMVRVCur"], fr["CapMrktCurUSD"])
                    if _ok(a) and _ok(b)}
        got, p = _mine("cm_mvrv", f_mvrv)
        if p:
            problems.append(p)
        mv = pd.Series({pd.Timestamp(k): v[0] for k, v in got.items()}).sort_index()
        mc = pd.Series({pd.Timestamp(k): v[1] for k, v in got.items()}).sort_index()
        out["mvrv"] = _known(mv, idx, 1, 3)
        extra["mcap"] = mc
        src["Coin Metrics 社群 API（CapMVRVCur、CapMrktCurUSD）"] = (
            onchain2.CM_URL + "?assets=btc&metrics=CapMrktCurUSD,CapMVRVCur,PriceUSD&frequency=1d")
    except Exception as exc:  # noqa: BLE001
        drop(["mvrv", "ssr"], exc)
    # Puell
    try:
        df, pr = pu._onchain()
        problems += pr
        out["puell"] = _known(pu._puell(df["IssTotUSD"].where(df["IssTotUSD"] > 0)), idx, 1, 3)
        src["Coin Metrics 社群 API（IssTotUSD，Puell 倍數）"] = (
            pu.CM_URL + "?assets=btc&metrics=" + pu.CM_METRICS + "&frequency=1d")
    except Exception as exc:  # noqa: BLE001
        drop(["puell"], exc)
    # STH cost basis
    try:
        cost, pr, _ = sc._coc()
        problems += pr
        out["sth"] = c / _known(cost, idx, 1, 3) - 1
        src["Checkonchain 短期持有人成本（最新版本，會修正）"] = sc.COC_DATA
    except Exception as exc:  # noqa: BLE001
        drop(["sth"], exc)
    # ADX
    try:
        fr, pr = adxm._btc_frame()
        problems += pr
        out["adx"] = pd.Series(adxm._adx(fr["high"].values, fr["low"].values, fr["close"].values), index=fr.index
                               ).reindex(idx)
        src["Bitstamp 日線高低價（2014-06 到 2017-08）"] = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
    except Exception as exc:  # noqa: BLE001
        drop(["adx"], exc)
    # network
    try:
        oc, pr = ng._onchain("btc")
        problems += pr
        for k, m in (("adr", "AdrActCnt"), ("tx", "TxCnt"), ("hash", "HashRate")):
            x = oc[m].where(oc[m] > 0).rolling(7, min_periods=7).mean()
            out[k] = _known(x, idx, 1, 3)
        src["Coin Metrics 社群 API（AdrActCnt、TxCnt、HashRate）"] = (
            ng.CM_URL + "?assets=btc&metrics=AdrActCnt,TxCnt,HashRate&frequency=1d")
    except Exception as exc:  # noqa: BLE001
        drop(["adr", "tx", "hash"], exc)
    try:
        ob, pr = bs._onchain()
        problems += pr
        fee = (ob["FeeTotNtv"] * ob["PriceUSD"]).where(ob["FeeTotNtv"] > 0).rolling(7, min_periods=7).mean()
        out["fee"] = _known(fee, idx, 1, 3)
        src["Coin Metrics 社群 API（FeeTotNtv、PriceUSD）"] = (
            bs.CM_URL + "?assets=btc&metrics=" + bs.CM_METRICS + "&frequency=1d")
    except Exception as exc:  # noqa: BLE001
        drop(["fee"], exc)
    # funding
    try:
        fr_, pr = xf._funding()
        if pr:
            problems.append(pr)
        day = fr_.groupby(fr_.index.normalize()).mean()
        out["funding"] = _known(day.asfreq("D").rolling(7, min_periods=5).mean(), idx, 0, 3)
        src["幣安 U 本位合約資金費率"] = cy.FAPI + "/fapi/v1/fundingRate?symbol=BTCUSDT"
    except Exception as exc:  # noqa: BLE001
        drop(["funding"], exc)
    # OI and long/short ratios (archive rows stamped 00:00 UTC)
    try:
        rows, pr = lr._fetch("BTCUSDT", budget_s=240)
        if pr:
            problems.append(pr)
        keys = sorted(rows)
        st = pd.to_datetime(keys)

        def col(j):
            return pd.Series([rows[k][j] if rows[k][j] is not None else np.nan for k in keys], index=st, dtype=float)
        for k, j in (("ls_r", 0), ("ls_t", 1), ("oi", 4)):
            x = col(j).asfreq("D").rolling(7, min_periods=5).mean()
            out[k] = _known(x, idx, 0, 3)
        src["幣安期貨歷史資料庫（每日 metrics 檔：多空比、未平倉量）"] = lr.ARCHIVE.format(s="BTCUSDT", d="YYYY-MM-DD")
    except Exception as exc:  # noqa: BLE001
        drop(["ls_r", "ls_t", "oi"], exc)
    # basis
    try:
        store = bf._read_store()
        spot = bf._bitstamp("btcusd")
        contracts = bf._deribit("BTC", store)
        basis, _ = bf._basis(contracts, spot["close"])
        out["basis"] = _known(basis, idx, 0, 3)
        src["Deribit 季度期貨小時線"] = bf.DERIBIT
        src["Bitstamp BTC/USD 日線"] = bf.BITSTAMP.format(pair="btcusd")
    except Exception as exc:  # noqa: BLE001
        drop(["basis"], exc)
    # DVOL, VRP
    try:
        dv, pr = vd._dvol("BTC")
        if pr:
            problems.append(pr)
        dvk = _known(dv, idx, 0, 3)
        out["dvol"] = dvk
        out["vrp"] = dvk - out["rv30"] * 100
        src["Deribit DVOL 波動指數"] = vd.DVOL_URL + "?currency=BTC&resolution=1D"
    except Exception as exc:  # noqa: BLE001
        drop(["dvol", "vrp"], exc)
    # skew (stored copy, read only)
    try:
        st_ = sk._load()
        if not st_:
            raise RuntimeError("偏斜存檔是空的")
        s = pd.Series({pd.Timestamp(k): float(v["skew"]) for k, v in st_.items() if isinstance(v, dict)
                       and _ok(v.get("skew"))}).sort_index()
        out["skew"] = _known(s, idx, 0, sk.GRID_STEP)
        src["Deribit 歷史選擇權成交（偏斜，skew_history 存檔）"] = sk.DERIBIT
    except Exception as exc:  # noqa: BLE001
        drop(["skew"], exc)
    # premiums
    try:
        kd = km._load()
        problems += kd["problems"]
        pm = km._premiums(kd)
        out["cbp"] = _known(pm["cbp"].rolling(7, min_periods=5).mean(), idx, 0, 3)
        out["kimchi"] = _known(pm["kp"].rolling(7, min_periods=5).mean(), idx, 0, 3)
        src["Upbit 日線"] = km.UPBIT_URL
        src["Coinbase BTC-USD 日線"] = km.COINBASE_URL
        src["聯準會 H.10 匯率（美元兌韓元）"] = km.FED_ZIP
    except Exception as exc:  # noqa: BLE001
        drop(["cbp", "kimchi"], exc)
    # stablecoins
    try:
        sup, pr = ss._load()
        problems += pr
        tot = sup["all"].where(sup["all"] > 0)
        out["stable"] = _known(tot, idx, 1, 3)
        if "mcap" in extra:
            out["ssr"] = _known(tot / extra["mcap"].reindex(tot.index), idx, 1, 3)
        src["DefiLlama 穩定幣總量"] = ss.LLAMA
    except Exception as exc:  # noqa: BLE001
        drop(["stable", "ssr"], exc)
    # ETF
    try:
        f, pr = el._farside("BTC")
        if pr:
            problems.append(pr)
        flow = (f["Total"].astype(float) * 1e6).sort_index()
        flow = flow[~flow.index.duplicated(keep="last")]
        days = pd.date_range(flow.index[0], idx[-1], freq="D")
        cum = flow.reindex(days).fillna(0.0).cumsum()
        out["etf"] = _known(cum, idx, 1, 3)
        src["Farside 美國現貨比特幣 ETF 每日流量"] = flows2.FARSIDE["BTC"][0]
    except Exception as exc:  # noqa: BLE001
        drop(["etf"], exc)
    # Wikipedia
    try:
        v, pr = at._views("btc")
        if pr:
            problems.append(pr)
        out["wiki"] = _known(v.where(v > 0).rolling(7, min_periods=7).mean(), idx, 1, 3)
        src["維基媒體瀏覽量 API（英文 Bitcoin 頁）"] = at.API.format(lang="en", art="Bitcoin", a=at.FIRST_VIEW,
                                                               b=common.today_utc().replace("-", ""))
    except Exception as exc:  # noqa: BLE001
        drop(["wiki"], exc)
    # breadth
    try:
        p, a, _ = bt._load()
        b = bt._breadth(p, a, bt.TOP)
        out["b200"] = _known(b["b200"], idx, 0, 3)
        src["幣安現貨日線（前 50 大幣種，survivorship／new_listings 存檔，唯讀）"] = (
            px.BINANCE_HOSTS[0] + "/api/v3/klines")
    except Exception as exc:  # noqa: BLE001
        drop(["b200"], exc)
    try:
        got, pr = _mine("eth_close", lambda: {_d(k): float(v) for k, v in ta._crypto_close("ETHUSDT").items()})
        if pr:
            problems.append(pr)
        eth = _ser(got)
        out["ethbtc"] = _known(eth / c.reindex(eth.index), idx, 0, 3)
        src["幣安現貨 ETHUSDT 日線"] = px.BINANCE_HOSTS[0] + "/api/v3/klines?symbol=ETHUSDT&interval=1d"
    except Exception as exc:  # noqa: BLE001
        drop(["ethbtc"], exc)
    # Yahoo (macro and cross-market)
    ysym = {"dxy": "DX-Y.NYB", "tnx": "^TNX", "tip": "TIP", "vix": "^VIX", "qqq": "QQQ", "spy": "SPY",
            "hyg": "HYG", "ief": "IEF", "lqd": "LQD", "gold": "GC=F"}
    yraw = {}
    for k, sym in ysym.items():
        fb = REAL_STORE / sf.STORE_FMT.format(sym.replace("^", "").replace("=", "_"))
        if k == "dxy":
            fb = REAL_STORE / "cross_market_dxy.json"
        try:
            got, pr = _mine(f"yahoo_{k}", lambda sym=sym: sf._fetch(sym), fb)
            if pr:
                problems.append(pr)
            yraw[k] = _ser(got)
            src[f"Yahoo 日線 {sym}"] = ro.YAHOO.format(sym)
        except Exception as exc:  # noqa: BLE001
            yraw[k] = exc
    for k in ("dxy", "tnx", "tip", "vix", "gold"):
        if isinstance(yraw[k], Exception):
            drop([k], yraw[k])
        else:
            out[k] = _known(yraw[k], idx, 0, 5)
    for k in ("qqq", "spy"):
        if isinstance(yraw[k], Exception):
            drop([k], yraw[k])
        else:
            y = yraw[k]
            out[k] = _known(y / y.rolling(200, min_periods=200).mean() - 1, idx, 0, 5)
    for k, (a_, b_) in (("hyg_ief", ("hyg", "ief")), ("lqd_ief", ("lqd", "ief"))):
        if isinstance(yraw[a_], Exception) or isinstance(yraw[b_], Exception):
            drop([k], yraw[a_] if isinstance(yraw[a_], Exception) else yraw[b_])
        else:
            both = pd.concat([yraw[a_], yraw[b_]], axis=1, join="inner").dropna()
            out[k] = _known(both.iloc[:, 0] / both.iloc[:, 1], idx, 0, 5)
    # NFCI
    try:
        s, where, notes = nf._series()
        problems += [n for n in notes if "改用" in n]
        out["nfci"] = _known(s, idx, 6, 8)
        src[f"NFCI（{where}）"] = nf.CHI_URL if "芝加哥" in where else nf.FRED_URL
    except Exception as exc:  # noqa: BLE001
        drop(["nfci"], exc)
    # OECD leading indicator
    try:
        s, where, note = bc._series("oecd")
        if note:
            problems.append(note)
        s = s.dropna()
        usable = pd.Series(s.values, index=pd.DatetimeIndex([(p_ + bc.LAG["oecd"]).end_time.normalize()
                                                             for p_ in s.index]))
        out["oecd"] = _known(usable, idx, 0, 35)
        src[f"OECD 美國綜合領先指標（{where}）"] = (bc.FRED_CSV.format(bc.FRED_ID["oecd"]) if where == "FRED"
                                               else bc.OECD_URL)
    except Exception as exc:  # noqa: BLE001
        drop(["oecd"], exc)
    # net liquidity
    try:
        parts, wh = {}, {}
        for key in ("assets", "tga", "rrp"):
            s, w, notes = nliq._piece(key)
            parts[key], wh[key] = s, w
            problems += [n for n in notes if "改用" in n]
        a_ = parts["assets"][parts["assets"].index.dayofweek == 2]
        fr = pd.concat([a_.rename("a"), parts["tga"].rename("t"),
                        parts["rrp"].reindex(a_.index).fillna(0.0).rename("r")], axis=1, join="inner").dropna()
        netl = (fr["a"] - fr["t"] - fr["r"]).sort_index()
        out["netliq"] = _known(netl[netl > 0], idx, int(nliq.SAFE / DAY), 8)
        src["聯準會淨流動性：" + "、".join(f"{nliq.PNAME[k]}＝{wh[k]}" for k in parts)] = (
            bc.FRED_CSV.format("WALCL") if wh["assets"].startswith("FRED") else "https://www.federalreserve.gov/releases/h41/")
        if not wh["tga"].startswith("FRED"):
            src["財政部每日報表（TGA）"] = nliq.DTS_URL
        if not wh["rrp"].startswith("FRED"):
            src["紐約聯準銀行逆回購資料"] = nliq.NYFED_URL
    except Exception as exc:  # noqa: BLE001
        drop(["netliq"], exc)
    out = {k: out[k].astype(float) for k in META if k in out}
    for k in META:
        if k not in out and k not in dropped:
            dropped[k] = "沒有組出來"
    return out, dropped, problems, src, extra


# ---------- turns ----------

def _scan(close: pd.Series, kind: str, big: float, small: float) -> list[dict]:
    v, ix = close.values.astype(float), close.index
    n = len(v)
    out = []
    if kind == "fall":
        h = lo = 0
        mode = "seek"
        for i in range(1, n):
            x = v[i]
            if mode == "seek":
                if x >= v[h]:
                    h = lo = i
                    continue
                if x < v[lo]:
                    lo = i
                if x <= v[h] * (1 - big):
                    mode = "in"
                elif x >= v[lo] * (1 + small):
                    h = lo = i
            else:
                if x < v[lo]:
                    lo = i
                elif x >= v[lo] * (1 + small):
                    out.append({"i": h, "e": lo, "done": True})
                    mode, h, lo = "seek", i, i
        if mode == "in":
            out.append({"i": h, "e": lo, "done": False})
    else:
        lo = hi = 0
        mode = "seek"
        for i in range(1, n):
            x = v[i]
            if mode == "seek":
                if x <= v[lo]:
                    lo = hi = i
                    continue
                if x > v[hi]:
                    hi = i
                if x >= v[lo] * (1 + big):
                    mode = "in"
                elif x <= v[hi] * (1 - small):
                    lo = hi = i
            else:
                if x > v[hi]:
                    hi = i
                elif x <= v[hi] * (1 - small):
                    out.append({"i": lo, "e": hi, "done": True})
                    mode, lo, hi = "seek", i, i
        if mode == "in":
            out.append({"i": lo, "e": hi, "done": False})
    for t in out:
        t["date"], t["end"] = ix[t["i"]], ix[t["e"]]
        t["p"], t["pe"] = float(v[t["i"]]), float(v[t["e"]])
        t["size"] = t["pe"] / t["p"] - 1
        t["days"] = int(t["e"] - t["i"])
    return out


def _turns(close: pd.Series) -> dict:
    out = {}
    for k, (kind, a, b) in {**CYCLE, **SWING}.items():
        out[k] = _scan(close, kind, a, b)
    return out


# ---------- per-series windows ----------

def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    out[k:] = a[:len(a) - k]
    return out


def _windows(r: np.ndarray) -> dict:
    s = pd.Series(r)
    m30 = s.rolling(30, min_periods=20).mean().values
    m60 = s.rolling(60, min_periods=40).mean().values
    m180 = s.rolling(180, min_periods=120).mean().values
    m20 = s.rolling(20, min_periods=14).mean().values
    return {"m30": m30, "lev": _shift(m30, 1), "chg": _shift(m60, 1) - _shift(m180, 61),
            "late": _shift(m20, 1) - _shift(m60, 21)}


def _side_of(v: float) -> int:
    return 1 if v >= TOP else (-1 if v <= BOT else 0)


def _in(v: float, side: int) -> bool:
    return _ok(v) and ((side > 0 and v >= TOP) or (side < 0 and v <= BOT))


# ---------- analysis ----------

def _pool(n: int, turn_idx: list[int], start: int) -> np.ndarray:
    ok = np.zeros(n, bool)
    ok[start:] = True
    for t in turn_idx:
        ok[max(0, t - AWAY + 1):min(n, t + AWAY)] = False
    return np.flatnonzero(ok)


def _level(W: dict, turns: list[dict], draws: np.ndarray) -> dict:
    lev = W["lev"]
    vals = [lev[t["i"]] for t in turns]
    have = [v for v in vals if _ok(v)]
    N = len(have)
    top = sum(v >= TOP for v in have)
    bot = sum(v <= BOT for v in have)
    rv = lev[draws]
    rv = rv[np.isfinite(rv)]
    p_top = float(np.mean(rv >= TOP)) if len(rv) >= 100 else None
    p_bot = float(np.mean(rv <= BOT)) if len(rv) >= 100 else None
    ex_top = top / N - p_top if N and p_top is not None else -9
    ex_bot = bot / N - p_bot if N and p_bot is not None else -9
    if top > bot or (top == bot and ex_top >= ex_bot):
        side, k, p, ex = 1, top, p_top, ex_top
    else:
        side, k, p, ex = -1, bot, p_bot, ex_bot
    return {"vals": vals, "N": N, "top": top, "bot": bot, "p_top": p_top, "p_bot": p_bot, "side": side, "k": k,
            "cons": (k / N) if N else None, "ex": ex if N else None, "tail": _binom_tail(k, N, p) if p is not None
            else None, "nr": int(len(rv))}


def _change(W: dict, turns: list[dict], draws: np.ndarray, key: str) -> dict:
    a = W[key]
    vals = np.array([a[t["i"]] for t in turns], dtype=float)
    vals = vals[np.isfinite(vals)]
    rv = a[draws]
    rv = rv[np.isfinite(rv)]
    return {"N": int(len(vals)), "up": int((vals > 0).sum()), "down": int((vals < 0).sum()),
            "med": float(np.median(vals)) if len(vals) else None,
            "rmed": float(np.median(rv)) if len(rv) >= 100 else None,
            "rup": float(np.mean(rv > 0)) if len(rv) >= 100 else None}


def _lead(m30: np.ndarray, T: int, side: int) -> tuple[str, int | None]:
    n = len(m30)
    if T - 1 >= 0 and _in(m30[T - 1], side):
        j = T - 1
        while j - 1 >= 0 and _in(m30[j - 1], side) and T - (j - 1) <= WIN:
            j -= 1
        return "before", T - j
    for j in range(T, min(T + FOLLOW + 1, n)):
        if _in(m30[j], side):
            return "after", j - T
    return "not", None


def _contrast(Wa: dict, ta_: list, Wb: dict, tb: list, key: str, rng) -> dict | None:
    a = np.array([Wa[key][t["i"]] for t in ta_], dtype=float)
    b = np.array([Wb[key][t["i"]] for t in tb], dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    na = len(a)
    sims = np.empty(SHUFFLES)
    for s in range(SHUFFLES):
        p = rng.permutation(pool)
        sims[s] = np.median(p[:na]) - np.median(p[na:])
    lo, hi = np.percentile(sims, [5, 95])
    return {"na": na, "nb": len(b), "ma": float(np.median(a)), "mb": float(np.median(b)), "diff": obs,
            "lo": float(lo), "hi": float(hi), "share": float(np.mean(np.abs(sims) >= abs(obs) - 1e-12)),
            "inside": bool(lo <= obs <= hi)}


def _mislead(W: dict, side: int, turns: list[dict], n: int) -> dict:
    m30 = W["m30"]
    valid = np.isfinite(m30)
    cond = np.array([_in(x, side) for x in m30])
    tset = np.zeros(n, bool)
    for t in turns:
        tset[t["i"]] = True
    cs = np.concatenate([[0], np.cumsum(tset)])
    end = n - FOLLOW
    idx = np.arange(n)
    follow = (cs[np.minimum(idx + FOLLOW + 1, n)] - cs[np.minimum(idx + 1, n)]) > 0
    use = valid & (idx < end)
    if use.sum() < 200:
        return {}
    ons = []
    for i in np.flatnonzero(cond & use):
        a = i - ABSENT
        if a < 0 or not valid[a]:
            continue
        if not cond[a:i].any():
            ons.append(i)
    ons = np.array(ons, dtype=int)
    return {"present": float(cond[use].mean()), "days": int(use.sum()), "onsets": int(len(ons)),
            "hit": int(follow[ons].sum()) if len(ons) else 0,
            "hit_share": float(follow[ons].mean()) if len(ons) else None, "base": float(follow[use].mean())}


def _compute(close: pd.Series, series: dict, extra: dict) -> dict:
    idx = close.index
    n = len(idx)
    turns = _turns(close)
    start = WIN_MIN + 240
    pools, draws = {}, {}
    rng = np.random.default_rng(SEED)
    for scale, keys in (("cycle", ("bear", "bull")), ("swing", ("fall", "rise"))):
        ti = [t["i"] for k in keys for t in turns[k]]
        pool = _pool(n, ti, start)
        pools[scale] = pool
        draws[scale] = rng.choice(pool, size=DRAWS, replace=len(pool) < DRAWS)
    W, rank, first = {}, {}, {}
    for k, s in series.items():
        r = at._trail_pct(s.values, WIN, WIN_MIN) * 100
        rank[k] = r
        W[k] = _windows(r)
        f = np.flatnonzero(np.isfinite(r))
        first[k] = idx[f[0]] if len(f) else None
    L, C, CL, O, M = {}, {}, {}, {}, {}
    for ty in TYPES:
        dr = draws[SCALE[ty]]
        L[ty] = {k: _level(W[k], turns[ty], dr) for k in series}
        C[ty] = {k: _change(W[k], turns[ty], dr, "chg") for k in series}
        CL[ty] = {k: _change(W[k], turns[ty], dr, "late") for k in series}
    order = {}
    for ty in TYPES:
        def sk_(k, ty=ty):
            x = L[ty][k]
            return (x["N"] < MIN_N, -(x["cons"] or 0), -x["N"], -(x["ex"] if x["ex"] is not None else -9), k)
        order[ty] = sorted(series, key=sk_)
    ranked = {ty: [k for k in order[ty] if L[ty][k]["N"] >= MIN_N] for ty in TYPES}
    ten = {ty: ranked[ty][:TEN] for ty in TYPES}
    for ty in TYPES:
        O[ty] = {}
        for k in ranked[ty]:
            x = L[ty][k]
            if x["cons"] is not None and x["cons"] >= 0.5:
                O[ty][k] = [(_lead(W[k]["m30"], t["i"], x["side"]) if _ok(W[k]["lev"][t["i"]]) else ("nodata", None))
                            for t in turns[ty]]
        M[ty] = {k: _mislead(W[k], L[ty][k]["side"], turns[ty], n) for k in ten[ty]}
    crng = np.random.default_rng(SEED + 1)
    CT = {}
    for a, b in (("bear", "bull"), ("fall", "rise")):
        CT[a + b] = {k: {"lev": _contrast(W[k], turns[a], W[k], turns[b], "lev", crng),
                         "chg": _contrast(W[k], turns[a], W[k], turns[b], "chg", crng)} for k in series}
    # combinations
    CB = {}
    for ty in TYPES:
        sides = {k: L[ty][k]["side"] for k in ten[ty]}
        per = []
        for t in turns[ty]:
            have = [k for k in ten[ty] if _ok(W[k]["lev"][t["i"]])]
            pres = [k for k in have if _in(W[k]["lev"][t["i"]], sides[k])]
            per.append((len(pres), len(have), pres))
        pool = pools[SCALE[ty]]
        cnt = np.zeros(len(pool), int)
        hv = np.zeros(len(pool), int)
        for k in ten[ty]:
            lv = W[k]["lev"][pool]
            hv += np.isfinite(lv)
            cnt += np.array([_in(x, sides[k]) for x in lv], dtype=int)
        CB[ty] = {"per": per, "cnt": cnt, "have": hv}
    return {"turns": turns, "W": W, "rank": rank, "first": first, "L": L, "C": C, "CL": CL, "O": O, "M": M,
            "CT": CT, "CB": CB, "ten": ten, "order": order, "ranked": ranked, "pools": pools, "draws": draws,
            "idx": idx}


# ---------- tables ----------

def _turn_rows(close: pd.Series, turns: list[dict], ty: str) -> list[list]:
    rows = []
    for t in turns:
        rows.append([TSHORT[ty], _d(t["date"]), f"{t['p']:,.2f}" if t["p"] < 100 else f"{t['p']:,.0f}",
                     _d(t["end"]), f"{t['size'] * 100:+.0f}%", f"{t['days']:,}",
                     "仍在進行（到目前為止）" if not t["done"] else "已結束"])
    return rows


def _series_name(k: str) -> str:
    return META[k][0]


def _level_cell(x: dict, j: int, raw: float, fmt: str) -> str:
    v = x["vals"][j]
    if not _ok(v):
        return "沒資料"
    mark = "（高）" if v >= TOP else ("（低）" if v <= BOT else "")
    return f"{v:.0f}{mark}｜{_fmt(raw, fmt)}"


def _side_word(side: int) -> str:
    return "偏高五分之一" if side > 0 else "偏低五分之一"


def _exp(p, N) -> str:
    return f"{p * N:.1f}" if p is not None and N else "—"


def _tables(close: pd.Series, series: dict, R: dict, extra: dict, dropped: dict) -> list[dict]:
    T, L, idx = R["turns"], R["L"], R["idx"]
    out = []
    # turn lists
    rows = _turn_rows(close, T["bear"], "bear") + _turn_rows(close, T["bull"], "bull")
    rows.sort(key=lambda r: r[1])
    out.append({"caption": "走熊／走牛的轉折點（機械方式從每日收盤找出）：走熊開始＝之後先跌 50% 以上、中間沒有從低點反彈 100%；"
                           "走牛開始＝之後先漲 100% 以上、中間沒有從高點回檔 50%。轉折日期只有事後才知道。",
                "columns": ["類型", "轉折日", "當天收盤（美元）", "走到的極端日", "幅度", "天數", "狀態"], "rows": rows})
    rows = _turn_rows(close, T["fall"], "fall") + _turn_rows(close, T["rise"], "rise")
    rows.sort(key=lambda r: r[1])
    out.append({"caption": "大跌／大漲的轉折點（較小尺度）：大跌開始＝之後先跌 30% 以上、中間沒有從低點反彈 25%；"
                           "大漲開始＝之後先漲 50% 以上、中間沒有從高點回檔 25%。同樣只有事後才知道。",
                "columns": ["類型", "轉折日", "當天收盤（美元）", "走到的極端日", "幅度", "天數", "狀態"], "rows": rows})
    # coverage
    rows = []
    for k in META:
        if k in series:
            f = R["first"].get(k)
            rows.append([_series_name(k), META[k][1], META[k][2], META[k][3], _d(f) if f is not None else "—"]
                        + [f"{L[ty][k]['N']}/{len(T[ty])}" for ty in TYPES[:2]])
    out.append({"caption": "資料涵蓋（第 7 點）：每個指標從哪天起有名次（需要先累積 365 天），以及走熊／走牛轉折前有資料的次數"
                           "（有資料次數／總次數）。大跌／大漲的次數在後面的摘要表裡。",
                "columns": ["指標", "白話說明", "類別", "晚幾天才知道", "名次從哪天起", "走熊有資料", "走牛有資料"],
                "rows": rows})
    if dropped:
        out.append({"caption": "這次沒能組出來、因此拿掉的指標（沒有用別的資料代替）",
                    "columns": ["指標", "原因"], "rows": [[META[k][0], v] for k, v in dropped.items()]})
    # step 1 per cycle turn
    for ty in ("bear", "bull"):
        tl = T[ty]
        live = [j for j in range(len(tl)) if any(_ok(L[ty][k]["vals"][j]) for k in series)]
        gone = [_d(tl[j]["date"]) for j in range(len(tl)) if j not in live]
        chunks = [live[i:i + 8] for i in range(0, len(live), 8)] or [[]]
        for ci, ch in enumerate(chunks):
            rows = []
            for k in R["order"][ty]:
                x = L[ty][k]
                fmt = META[k][4]
                rows.append([_series_name(k)] + [_level_cell(x, j, series[k].iloc[tl[j]["i"] - 1]
                                                             if tl[j]["i"] >= 1 else None, fmt) for j in ch])
            part = f"（第 {ci + 1} 部分）" if len(chunks) > 1 else ""
            out.append({"caption": f"第 1 點・{TNAME[ty]}前 30 天的平均名次{part}：每格＝名次（0–100，和自己過去兩年比）｜"
                                   "轉折前一天知道的原始數值；≥80 標「高」、≤20 標「低」。列的順序＝一致程度（見下一張表）。"
                                   + (f"{'、'.join(gone)} 這 {len(gone)} 次發生在任何指標有名次之前，沒有列出。" if gone else ""),
                        "columns": ["指標"] + [_d(tl[j]["date"]) for j in ch], "rows": rows})
    # step 1 summary all types
    for ty in TYPES:
        rows = []
        for k in R["order"][ty]:
            x = L[ty][k]
            if not x["N"]:
                continue
            rows.append([_series_name(k), str(x["N"]), f"{x['top']}（{_exp(x['p_top'], x['N'])}）",
                         f"{x['bot']}（{_exp(x['p_bot'], x['N'])}）",
                         f"{_side_word(x['side'])}：{_of(x['k'], x['N'])}" + ("" if x["N"] >= MIN_N else "（太少，不排名）"),
                         f"{x['tail']:.2f}" if _ok(x["tail"]) else "—"])
        out.append({"caption": f"第 1 點・{TNAME[ty]}前的一致程度（共 {len(T[ty])} 次）：轉折前 30 天平均名次落在偏高（≥80）"
                               "或偏低（≤20）五分之一的次數；括號＝隨機日子（離轉折 90 天以上）平均會有幾次。最後一欄＝如果每次轉折"
                               "都只是隨機一天，看到這麼多次（或更多）的機率——只是描述「多不尋常」，不是檢定，而且指標有 40 個左右，"
                               "光靠運氣就會有一兩個看起來很低。",
                    "columns": ["指標", "有資料次數", "偏高次數（隨機）", "偏低次數（隨機）", "比較一致的一邊", "隨機也這麼多的機率"],
                    "rows": rows})
    # step 2 change
    for ty in TYPES:
        rows = []
        for k in R["order"][ty]:
            c, cl = R["C"][ty][k], R["CL"][ty][k]
            if not c["N"] and not cl["N"]:
                continue
            rows.append([_series_name(k), str(c["N"]), f"{c['up']}／{c['down']}", _r(c["med"]), _r(c["rmed"]),
                         _sp(c["rup"]), f"{cl['up']}／{cl['down']}", _r(cl["med"]), _r(cl["rmed"])])
        out.append({"caption": f"第 2 點・{TNAME[ty]}前名次怎麼變：（前 60 天平均名次）減（再往前 61–240 天平均名次）。"
                               "「升／降」＝幾次上升、幾次下降；「隨機中位」＝隨機日子的同一個數字。後三欄是較晚的版本："
                               "（前 20 天）減（前 21–80 天）。",
                    "columns": ["指標", "次數", "升／降", "中位變化", "隨機中位", "隨機上升比例", "晚期升／降", "晚期中位",
                                "晚期隨機中位"], "rows": rows})
    # step 3 order
    for ty in TYPES:
        O = R["O"][ty]
        if not O:
            continue
        if SCALE[ty] == "cycle":
            tl = T[ty]
            live = [j for j in range(len(tl)) if any(cells[j][0] != "nodata" for cells in O.values())]
            chunks = [live[i:i + 7] for i in range(0, len(live), 7)] or [[]]
            for ci, ch in enumerate(chunks):
                rows = []
                for k, cells in O.items():
                    bef = [d for w, d in cells if w == "before"]
                    row = [f"{_series_name(k)}（{'高' if L[ty][k]['side'] > 0 else '低'}）"]
                    for j in ch:
                        w, d = cells[j]
                        row.append({"before": (f"前 {d} 天" if (d or 0) < WIN else f"前 {WIN} 天以上"), "after": f"之後 {d} 天", "not": "沒進入",
                                    "nodata": "沒資料"}[w])
                    row.append(f"{np.median(bef):.0f}" if bef else "—")
                    rows.append(row)
                part = f"（第 {ci + 1} 部分）" if len(chunks) > 1 else ""
                out.append({"caption": f"第 3 點・{TNAME[ty]}：在一半以上轉折前處於極端五分之一的指標，各自從轉折前幾天開始一直待在"
                                       f"那一區（用 30 天平均名次，它本身會晚約 15 天）{part}。「之後」＝轉折前一天不在、轉折後 90 天內才進入。",
                            "columns": ["指標（哪一邊）"] + [_d(T[ty][j]["date"]) for j in ch] + ["「前」的中位天數"],
                            "rows": rows})
        else:
            rows = []
            for k, cells in O.items():
                bef = [d for w, d in cells if w == "before"]
                aft = [d for w, d in cells if w == "after"]
                rows.append([f"{_series_name(k)}（{'高' if L[ty][k]['side'] > 0 else '低'}）",
                             str(sum(w != "nodata" for w, _ in cells)), str(len(bef)),
                             f"{np.median(bef):.0f}" if bef else "—", str(len(aft)),
                             f"{np.median(aft):.0f}" if aft else "—", str(sum(w == "not" for w, _ in cells))])
            out.append({"caption": f"第 3 點・{TNAME[ty]}：在一半以上轉折前處於極端五分之一的指標，進入那一區的時間（摘要）。",
                        "columns": ["指標（哪一邊）", "有資料次數", "轉折前已在", "前幾天（中位）", "轉折後才進入",
                                    "後幾天（中位）", "沒進入"], "rows": rows})
    # step 4 contrast
    for pair, (a, b) in (("bearbull", ("bear", "bull")), ("fallrise", ("fall", "rise"))):
        rows = []
        for k in series:
            cl, cc = R["CT"][pair][k]["lev"], R["CT"][pair][k]["chg"]
            if cl is None:
                continue
            rows.append([_series_name(k), f"{cl['na']}／{cl['nb']}", f"{cl['ma']:.0f}／{cl['mb']:.0f}",
                         f"{cl['diff']:+.0f}", f"{cl['lo']:+.0f} 到 {cl['hi']:+.0f}" + ("" if cl["inside"] else " ✱"),
                         f"{cc['diff']:+.0f}" if cc else "—",
                         (f"{cc['lo']:+.0f} 到 {cc['hi']:+.0f}" + ("" if cc["inside"] else " ✱")) if cc else "—"])
        rows.sort(key=lambda r: -abs(float(r[3])))
        out.append({"caption": f"第 4 點・{TSHORT[a]}前 和 {TSHORT[b]}前 比：前 30 天平均名次的中位數（{TSHORT[a]}／{TSHORT[b]}）、"
                               f"差距（{TSHORT[a]}減{TSHORT[b]}），以及把兩類標籤隨機洗牌 2,000 次後差距的 5%–95% 範圍；"
                               "✱＝實際差距落在洗牌範圍外。後兩欄是名次變化（第 2 點）的同樣比較。次數很少時洗牌範圍很寬，"
                               "落在範圍內不代表「一樣」，只代表這些次數分不出來。",
                    "columns": ["指標", f"次數（{TSHORT[a]}／{TSHORT[b]}）", "名次中位數", "差距", "洗牌 90% 範圍",
                                "變化的差距", "變化：洗牌 90% 範圍"], "rows": rows})
    # step 5 mislead
    for ty in TYPES:
        rows = []
        for k in R["ten"][ty]:
            m, x = R["M"][ty].get(k) or {}, L[ty][k]
            if not m:
                rows.append([_series_name(k), _side_word(x["side"]), _of(x["k"], x["N"]), "—", "—", "—", "—"])
                continue
            rows.append([_series_name(k), _side_word(x["side"]), _of(x["k"], x["N"]), _sp(m["present"]),
                         str(m["onsets"]), f"{m['hit']}（{_sp(m['hit_share'])}）", _sp(m["base"])])
        out.append({"caption": f"第 5 點・會誤導多少（{TNAME[ty]}）：一致程度前 10 名（依第 1 點的 k/N 排；同分看次數多、"
                               "再看比隨機多出多少）。這是看過第 1 點結果才挑的，只是初步一看，不是檢定。「出現天數比例」＝所有日子裡"
                               "30 天平均名次在同一邊的比例；「出現次數」＝至少 60 天沒出現後再出現的次數；之後 90 天內真的出現這類轉折的"
                               "次數（比例）；最後一欄＝隨便哪一天之後 90 天內出現這類轉折的比例。",
                    "columns": ["指標", "哪一邊", "轉折前出現", "出現天數比例", "出現次數", "90 天內有轉折", "隨便一天的比例"],
                    "rows": rows})
    # step 6 combinations
    for ty in TYPES:
        cb = R["CB"][ty]
        per = cb["per"]
        rows = []
        for t, (p_, h_, pres) in zip(T[ty], per):
            if not h_:
                continue
            same = int(np.sum((cb["cnt"] >= p_) & (cb["have"] >= 1))) if p_ > 0 else None
            tot = int(np.sum(cb["have"] >= 1))
            rows.append([_d(t["date"]), f"{p_}／{h_}", "、".join(_series_name(k) for k in pres) or "（沒有）",
                         f"{same:,}／{tot:,}（{same / tot * 100:.0f}%）" if same is not None and tot else "—"])
        out.append({"caption": f"第 6 點・同時出現幾個（{TNAME[ty]}，只是觀察）：第 5 點那 10 個指標，在每次轉折前一天有幾個處在"
                               "自己那一邊（出現數／有資料數）。最後一欄＝離轉折 90 天以上的日子裡，同時出現至少這麼多個的天數"
                               "／有資料的天數。注意：早期的轉折有資料的指標比較少，而且很多指標彼此高度相關（例如 MVRV、"
                               "冪次法則偏離、離 200 日均線），同時出現不等於多個獨立的線索。",
                    "columns": ["轉折日", "出現／有資料", "哪些", "非轉折日至少這麼多的天數"], "rows": rows})
    # halving clock
    rows = []
    for ty in TYPES[:2]:
        for t in T[ty]:
            v = extra["halving"].iloc[t["i"]]
            rows.append([TSHORT[ty], _d(t["date"]), f"{v:,.0f}" if _ok(v) else "第一次減半之前"])
    rows.sort(key=lambda r: r[1])
    out.append({"caption": "日曆位置：每次走熊／走牛開始時，距上一次減半（2012-11-28、2016-07-09、2020-05-11、2024-04-20）幾天。"
                           "這是時鐘，不排名次。",
                "columns": ["類型", "轉折日", "距上次減半天數"], "rows": rows})
    return out


def _today(series: dict, R: dict) -> list[list]:
    rows, idx = [], R["idx"]
    for ty in TYPES:
        for k in R["ten"][ty]:
            r = R["rank"][k]
            m30 = R["W"][k]["m30"]
            f = np.flatnonzero(np.isfinite(m30))
            if not len(f):
                continue
            j = int(f[-1])
            side = R["L"][ty][k]["side"]
            v = m30[j]
            where = ("偏高五分之一" if v >= TOP else ("偏低五分之一" if v <= BOT else "中間"))
            same = "是" if _in(v, side) else "否"
            rows.append([TSHORT[ty], _series_name(k), _side_word(side), _d(idx[j]), _r(r[j]), f"{v:.0f}（{where}）",
                         _fmt(series[k].iloc[j], META[k][4]), same])
    return rows


# ---------- verdict ----------

def _desc(L: dict, ty: str, k: str, side: int | None = None) -> str:
    """'name high (k of N; random about x)' for one series and turn type (side forced when given)."""
    x = L[ty][k]
    sd = x["side"] if side is None else side
    kk = x["top"] if sd > 0 else x["bot"]
    p = x["p_top"] if sd > 0 else x["p_bot"]
    return (f"{_series_name(k)}{'偏高' if sd > 0 else '偏低'}（{TSHORT[ty]}前 {_of(kk, x['N'])}；"
            f"隨機日子約 {_exp(p, x['N'])} 次）")


def _excess(L: dict, ty: str, m: int = 5, min_n: int = 5) -> list[str]:
    items = []
    for k, x in L[ty].items():
        p = x["p_top"] if x["side"] > 0 else x["p_bot"]
        if x["N"] >= min_n and p is not None:
            items.append((x["k"] / x["N"] - p, k))
    items.sort(key=lambda t: (-t[0], t[1]))
    return [_desc(L, ty, k) for _, k in items[:m]]


def _verdict(R: dict, series: dict, dropped: dict) -> str:
    T, L = R["turns"], R["L"]
    parts = []

    def best(ty, m=5):
        return "、".join(_desc(L, ty, k) for k in R["ten"][ty][:m])

    def empty(ty):
        return sum(not any(_ok(L[ty][k]["vals"][j]) for k in series) for j in range(len(T[ty])))
    nb, nu = len(T["bear"]), len(T["bull"])
    parts.append(f"(a) 量到的：從 2010 年 7 月起的每日收盤，機械地找出 {nb} 次走熊開始、{nu} 次走牛開始（其中走熊 {empty('bear')} 次、"
                 f"走牛 {empty('bull')} 次早到沒有任何指標有名次；有資料的最多 {max(L['bear'][k]['N'] for k in series)} 次走熊、"
                 f"{max(L['bull'][k]['N'] for k in series)} 次走牛），以及 {len(T['fall'])} 次大跌開始、{len(T['rise'])} 次大漲開始。"
                 f"依「k/N」排的一致程度，走熊開始前 30 天最一致的是：{best('bear')}。走牛開始前：{best('bull')}。"
                 f"大跌開始前：{best('fall', 3)}。大漲開始前：{best('rise', 3)}。"
                 "注意排在前面的有些是「幾乎每天都偏高」的指標（算力、穩定幣總量一路往上長，和自己過去兩年比幾乎總在偏高區），"
                 "括號裡隨機日子的次數和轉折前差不多時，它說的不是轉折。"
                 "（看過結果後加的）改用「比隨機日子多出多少」排、至少 5 次有資料的：走熊前 " + "、".join(_excess(L, "bear"))
                 + "；走牛前 " + "、".join(_excess(L, "bull")) + "。"
                 "所有說法都只建立在這麼少的次數上；和價格有關的指標（離一年最高、離 200／50 日均線、冪次法則偏離、MVRV、"
                 "短期持有人成本、Puell）有一部分是轉折定義本身帶出來的：高點本來就是價格高的時候，低點本來就是價格低的時候。")
    outside, same_both, note = [], [], []
    for k in series:
        c = R["CT"]["bearbull"][k]["lev"]
        if c is None:
            continue
        xa, xb = L["bear"][k], L["bull"][k]
        if not c["inside"]:
            outside.append(f"{_series_name(k)}（{c['ma']:.0f} 對 {c['mb']:.0f}）")
        elif (xa["N"] >= MIN_N and xb["N"] >= MIN_N and xa["side"] == xb["side"] and (xa["cons"] or 0) >= 0.5
              and (xb["cons"] or 0) >= 0.5):
            same_both.append(f"{_desc(L, 'bear', k)}、{_desc(L, 'bull', k, xa['side'])}")
            pa = xa["p_top"] if xa["side"] > 0 else xa["p_bot"]
            pb = xb["p_top"] if xa["side"] > 0 else xb["p_bot"]
            if pa is not None and pb is not None and xa["k"] > pa * xa["N"] + 1 and xb["k"] > pb * xb["N"] + 1:
                note.append(_series_name(k))
    parts.append("(b) 量到的：走熊前和走牛前，前 30 天平均名次的中位數差距超出洗牌範圍的有 " + str(len(outside)) + " 個："
                 + ("、".join(outside) if outside else "沒有") + "。"
                 + ("兩類轉折前都常在同一個極端、而且差距分不出來的：" + "；".join(same_both)
                    + "——以這些次數量到的，它們沒有分出方向，最多是在說「大波動快來了」。"
                    + "（看過結果後補充）但其中多數在隨機日子也差不多這麼常在那一區，連「大波動快來了」都說不上；"
                    + ("兩類轉折前都比隨機日子多出一次以上的只有：" + "、".join(note) + "。" if note else
                       "沒有一個在兩類轉折前都比隨機日子多出一次以上。")
                    if same_both else "沒有指標在兩類轉折前都處在同一個極端而差距又分不出來。")
                 + "其餘指標的差距落在洗牌範圍內：以這麼少的次數分不出來，不代表兩類之前真的一樣。")
    mis = []
    for ty in ("bear", "bull"):
        for k in R["ten"][ty][:5]:
            m = R["M"][ty].get(k)
            if m and m.get("onsets"):
                mis.append(f"{_series_name(k)}（{TSHORT[ty]}）：{_sp(m['present'])} 的日子都在那一邊；新出現 {m['onsets']} 次，"
                           f"其中 {m['hit']} 次 90 天內真的{TSHORT[ty]}（隨便一天是 {_sp(m['base'])}）")
    parts.append("(c) 量到的（看過第 1 點才挑的初步一看，不是檢定）：同樣的狀態在沒有轉折的日子出現得多頻繁——" + "；".join(mis) + "。"
                 "大跌、大漲的同一張表在第 5 點。")
    dl = "、".join(META[k][0] for k in dropped) if dropped else "沒有"
    parts.append("(d) 看不到的：期貨、選擇權、溢價、ETF、穩定幣、廣度等指標大多 2017–2024 年才有資料，名次又要先累積一年，"
                 "所以只碰得到最後一到四次走熊／走牛（隱含波動、偏斜、ETF 只有 0–1 次）；衍生品只有幣安和 Deribit；"
                 "鏈上數字、短期持有人成本、OECD 領先指標用的是現在的修正後版本，當時看到的可能不同；轉折日期是事後才貼上的標籤；"
                 "2013 年 4 月幾天內 -71%、+128%、-56%，定義把它算成兩次走熊、一次走牛，前 30 天的窗口幾乎重疊。"
                 "這次拿掉的指標：" + dl + "。「沒看到」只表示用這個定義和這些資料沒看到。")
    parts.append(_spec(R, series))
    return "\n".join(parts)


def _spec(R: dict, series: dict) -> str:
    """(e) Written after seeing the results of the first run; each idea quotes the measured counts it came from."""
    L, M = R["L"], R["M"]
    ideas = []

    def has(*ks):
        return all(k in series for k in ks)
    if has("nfci"):
        ideas.append(f"金融條件指數：{_desc(L, 'bear', 'nfci', -1)}，{_desc(L, 'bull', 'nfci', 1)}。值得事先寫好規則，"
                     "測「金融條件很寬鬆＋價格已經漲很多」之後一年的最大跌幅是否比較深，並用比特幣以外的資產對照。")
    if has("mvrv", "ma200"):
        ideas.append(f"估值一群（{_desc(L, 'bear', 'mvrv')}、{_desc(L, 'bull', 'mvrv', -1)}）：和轉折定義綁在一起，"
                     "值得測的是它比單純「離 200 日均線」多提供了什麼，才分得出是線索還是定義。")
    if has("netliq"):
        m = M["bear"].get("netliq") or {}
        ideas.append(f"聯準會淨流動性：{_desc(L, 'bear', 'netliq')}，但 {_sp(m.get('present'))} 的日子都在偏高區、"
                     f"新出現 {m.get('onsets', '—')} 次只有 {m.get('hit', '—')} 次 90 天內走熊；比較像慢的背景，值得用一年以上的"
                     "期間、當成環境變數測，而不是當成時機。")
    if has("funding", "basis"):
        ideas.append(f"槓桿的冷熱：{_desc(L, 'rise', 'funding', -1)}、{_desc(L, 'bull', 'basis', -1)}；只有一個交易所、"
                     "次數很少，值得用其他交易所和芝商所基差、以太幣等其他幣種擴大樣本再測。")
    if has("oi"):
        ideas.append(f"未平倉量：{_desc(L, 'fall', 'oi', 1)}，只有 2021 年以後；值得在其他幣種（各自的未平倉量）事先登記再看。")
    head = "(e) 推測（看過結果後才寫的，只是值得之後事先登記、好好測的想法，沒有說任何一個有效）："
    return head + " ".join(f"{i + 1}. {t}" for i, t in enumerate(ideas[:5]))


def study() -> dict:
    t0 = time.time()
    with _own_store():
        close = te._daily_close()
        series, dropped, problems, src, extra = _assemble(close)
    R = _compute(close, series, extra)
    tables = _tables(close, series, R, extra, dropped)
    today = _today(series, R)
    tables.append({"caption": "第 8 點・今天的位置：每類轉折一致程度前 10 名的指標，在最新一天的名次（和自己過去兩年比）。"
                              "這是描述，不是預測，也沒有任何規則的訊號。",
                   "columns": ["轉折類型", "指標", "轉折前常見的一邊", "最新日期", "當天名次", "30 天平均名次", "原始數值",
                               "現在是否在同一邊"], "rows": today})
    T = R["turns"]
    idx = R["idx"]
    last_c = [t for ty in ("bear", "bull") for t in T[ty]]
    last_c.sort(key=lambda t: t["date"])
    now = [{"label": f"最新收盤（{_d(idx[-1])}，UTC 0 點日線）", "value": f"{close.iloc[-1]:,.0f} 美元"}]
    if last_c:
        lt = last_c[-1]
        ty = "bear" if lt in T["bear"] else "bull"
        now.append({"label": "最近一次走熊／走牛轉折（事後才確認）",
                    "value": f"{TSHORT[ty]}開始於 {_d(lt['date'])}，到目前為止 {lt['size'] * 100:+.0f}%"
                             + ("，仍在進行" if not lt["done"] else "")})
    for ty in ("bear", "bull"):
        rows = [r for r in today if r[0] == TSHORT[ty]]
        hit = [r[1] for r in rows if r[7] == "是"]
        now.append({"label": f"{TSHORT[ty]}前最一致的 10 個徵兆，今天在同一邊的",
                    "value": f"{len(hit)}／{len(rows)} 個" + (f"：{'、'.join(hit)}" if hit else "") + "（描述，不是預測）"})
    asof = _d(idx[-1])
    notes = [
        "這是觀察研究：只描述轉折之前各種指標長什麼樣子，沒有測試任何交易規則。轉折日期只有事後才知道，所以這裡的「徵兆」"
        "就算每次都出現，也不代表當時能拿來操作。",
        "名次＝和自己過去 730 天比（今天的值在過去兩年裡排第幾，0–100），至少要先累積 365 天才有名次。「偏高」只是"
        "相對自己過去兩年偏高，不是絕對水準。轉折前 30 天的平均名次要到 ≥80 或 ≤20 才算在極端，單日比較容易到，"
        "30 天平均比較難，所以每張表都附上隨機日子的對照。",
        "隨機日子：從離同一尺度任何轉折 90 天以上、而且價格歷史開始 605 天以後的日子，固定亂數抽 2,000 天"
        f"（走熊／走牛尺度可抽的日子 {len(R['pools']['cycle']):,} 天，大跌／大漲尺度 {len(R['pools']['swing']):,} 天；"
        "可抽的日子不到 2,000 天時重複抽）。",
        "和價格有關的指標（離一年最高、離 200／50 日均線、冪次法則偏離、MVRV、短期持有人成本、距一年新高天數）和轉折的"
        "定義綁在一起：走熊開始是高點，前 30 天本來就接近一年最高；走牛開始是低點，前 30 天本來就接近一年最低。"
        "這部分的「一致」有多少是真的線索、多少是定義帶出來的，這裡的資料分不出來。",
        "很多指標彼此高度相關（MVRV、冪次法則偏離、離 200 日均線、短期持有人成本幾乎一起動；活躍地址和交易筆數一起動），"
        "「好幾個一起出現」不等於好幾個獨立的線索。約 40 個指標一起看，光靠運氣就會有一兩個看起來很一致。",
        "期貨、選擇權、溢價、ETF、穩定幣、幣種廣度、維基瀏覽量的歷史短，只碰得到最近一兩次走熊／走牛；"
        "衍生品只有幣安（資金費率、未平倉量、多空比）和 Deribit（基差、隱含波動、偏斜）。偏斜每 4 天一筆。",
        "鏈上數字（Coin Metrics）、短期持有人成本（Checkonchain 最新版本）、OECD 領先指標都會事後修正，這裡用的是現在的版本，"
        "當時看到的數字可能不同。",
        "每個值都只用當時已經知道的資料：鏈上與穩定幣、ETF、維基晚 1 天，NFCI 晚約 6 天，聯準會淨流動性晚 2 天，"
        "OECD 領先指標晚約 2 個月；美股、美債、美元、黃金用當天美國收盤（在 UTC 0 點以前收）。",
        "第 5、6 點是看過第 1 點結果才挑的指標，只能算初步一看。",
        f"這次資料下載與計算花了約 {time.time() - t0:.0f} 秒。",
    ] + problems
    sources = [{"name": k, "url": v} for k, v in src.items()]
    return {"id": ID, "title": "轉折之前有什麼徵兆？（走牛、走熊、大漲、大跌之前，各種指標的樣子）",
            "question": "在走熊開始、走牛開始、大跌開始、大漲開始之前，專案量得到的每一種指標（估值、趨勢、鏈上、槓桿、選擇權、"
                        "溢價、穩定幣與 ETF、關注度、廣度、總經、減半時鐘）長什麼樣子？有沒有每次都出現的線索？"
                        "沒事的時候是不是也常出現？",
            "asof": asof, "verdict": _verdict(R, series, dropped), "now": now, "tables": tables, "notes": notes,
            "sources": sources}
