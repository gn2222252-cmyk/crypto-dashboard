"""Build the single-file dashboard page from the collector caches.

Run from the project root:
    .venv/Scripts/python.exe -m dashboard.build             # build dashboard/out/index.html from the caches
    .venv/Scripts/python.exe -m dashboard.build --refresh   # run the five collectors first

The page is self-contained (no network resources), so it opens by double-click or from any static server.
A collector that fails during --refresh keeps its previous cache and the page says so. The same data that is
embedded in the page is also written to dashboard/out/data.json (the contract is in dashboard/SPEC2.md, section 3;
dashboard/check.py verifies it).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from dashboard import common, insights, scoring

COLLECTORS = ("price", "classic", "derivatives", "options", "onchain", "onchain2", "flows", "flows2", "whales", "macro", "breadth",
              "sentiment", "studies")  # collector modules; a missing or failing one is reported, not fatal
TEMPLATE = common.HERE / "template.html"
TOOLS = common.HERE / "tools.json"
PLACEHOLDER = "__DASHBOARD_DATA__"


def refresh() -> dict[str, str]:
    """Run each collector in turn. Returns {collector name: error text} for the ones that failed."""
    errors: dict[str, str] = {}
    for name in COLLECTORS:
        try:
            module = importlib.import_module(f"dashboard.collectors.{name}")
            rows = module.collect()
            common.save(name, rows)
            bad = sum(1 for row in rows if row.get("error"))
            print(f"收集器 {name}：{len(rows)} 個指標，其中 {bad} 個取得失敗")
        except Exception as exc:  # noqa: BLE001 - a broken collector keeps its old cache, the page flags it
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"收集器 {name}：執行失敗，沿用舊快取（{errors[name]}）")
    return errors


def load_cache(refresh_errors: dict[str, str], include_samples: bool = False) -> tuple[list[dict], list[dict]]:
    """Merge every cache/<name>.json. Files whose name starts with an underscore are skipped."""
    indicators: list[dict] = []
    sources: list[dict] = []
    seen: set[str] = set()
    paths = sorted(common.CACHE.glob("*.json")) if common.CACHE.exists() else []
    for path in paths:
        if path.name.startswith("_") and not include_samples:
            continue
        entry = {"name": path.stem, "collected_utc": None, "count": 0, "failed": 0,
                 "error": refresh_errors.get(path.stem), "missing": False}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = [row for row in data.get("indicators", []) if isinstance(row, dict) and row.get("id")]
            entry["collected_utc"] = data.get("collected_utc")
            entry["count"] = len(rows)
            entry["failed"] = sum(1 for row in rows if row.get("error"))
            for row in rows:
                if row["id"] in seen:  # the same indicator from two collectors: keep the first one only
                    continue
                seen.add(row["id"])
                indicators.append(row)
        except Exception as exc:  # noqa: BLE001 - an unreadable cache file is reported, not fatal
            entry["error"] = f"快取檔讀不出來：{type(exc).__name__}: {exc}"[:300]
        sources.append(entry)
    have = {entry["name"] for entry in sources}
    for name in COLLECTORS:
        if name not in have:
            sources.append({"name": name, "collected_utc": None, "count": 0, "failed": 0,
                            "error": refresh_errors.get(name), "missing": True})
    return indicators, sources


def load_tools() -> list[dict]:
    try:
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        return [tool for tool in tools if isinstance(tool, dict) and tool.get("name") and tool.get("url")]
    except Exception:  # noqa: BLE001 - the tool list is optional
        return []


def load_whales() -> dict:
    """The whale section's tables, written by the whales collector; empty if it has never run."""
    try:
        return json.loads((common.CACHE / "_whales.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - the section is optional
        return {}


def load_classic() -> dict:
    """The rainbow chart's data, written by the classic collector; empty if it has never run."""
    try:
        return json.loads((common.CACHE / "_classic.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - the section is optional
        return {}


def load_studies() -> dict:
    """The historical studies, written by the studies collector; empty if it has never run."""
    try:
        return json.loads((common.CACHE / "_studies.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - the sections that use it are optional
        return {}


def embed(payload: dict) -> str:
    """JSON that is safe inside a <script> element: no '</script>', no '<!--', no raw line separators."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
    for code in (0x3C, 0x3E, 0x26, 0x2028, 0x2029):  # < > & and the two Unicode line separators
        text = text.replace(chr(code), "%su%04x" % (chr(92), code))
    return text


def _clean(value):
    """NaN and infinity are not valid JSON; turn them into null so the browser can parse the page data."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def build(do_refresh: bool = False, include_samples: bool = False):
    refresh_errors = refresh() if do_refresh else {}
    indicators, sources = load_cache(refresh_errors, include_samples)
    indicators = _clean(indicators)  # the scores must never see NaN
    scoring.annotate(indicators)
    for row in indicators:  # weekly CFTC report: a week old is normal, not stale
        if row.get("id", "").startswith("btc_cme_"):
            row["stale_after_days"] = 12
    studies = _clean(load_studies())
    payload = _clean({"built_utc": common.now_utc(), "disclaimer": scoring.DISCLAIMER,
                      "assets": scoring.assets(indicators), "indicators": indicators, "sources": sources,
                      "tools": load_tools(), "whales": load_whales(), "classic": load_classic(),
                      "studies": studies, "insights": insights.build(indicators, studies)})
    text = embed(payload)
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise RuntimeError(f"template.html 裡必須剛好有一個 {PLACEHOLDER}")
    common.OUT.mkdir(parents=True, exist_ok=True)
    (common.OUT / "data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                                          encoding="utf-8")
    path = common.OUT / "index.html"
    path.write_text(template.replace(PLACEHOLDER, text), encoding="utf-8")
    bad = sum(1 for row in indicators if row.get("error"))
    print(f"指標 {len(indicators)} 個（取得失敗 {bad} 個），快取檔 {sum(1 for s in sources if not s['missing'])} 個")
    for block in payload["assets"]:
        temp = "沒有溫度" if block["temperature"] is None else f"{block['temperature']:.1f} 分（{block['zone_label']}）"
        print(f"{block['asset']} 市場溫度：{temp}，計入 {block['components_counted']}／{block['components_total']} 個成分，"
              f"溫度歷史 {len(block['temp_history'])} 筆")
    print(f"已輸出 {path} 與 {common.OUT / 'data.json'}")
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="產生儀表板頁面 dashboard/out/index.html")
    parser.add_argument("--refresh", action="store_true", help="先依序執行五個收集器再產生頁面")
    parser.add_argument("--with-samples", action="store_true", help="測版面用：連底線開頭的快取檔也讀進來")
    args = parser.parse_args()
    build(args.refresh, args.with_samples)
