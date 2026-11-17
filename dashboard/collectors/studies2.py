"""Second set of historical studies, written to cache/_studies2.json and shown on the "history" tab.

Each study lives in its own module under dashboard/studies2/ and exposes study() -> dict with one shared shape, so
the page needs a single renderer for all of them:

    {"id", "title", "question", "asof", "verdict",
     "now": [{"label", "value"}], "tables": [{"caption", "columns", "rows"}], "notes": [...], "sources": [{"name", "url"}]}

Every string is ready to show (Traditional Chinese, already formatted). These are not indicators: collect() returns
no rows. A study that fails keeps its previous result and the failure is recorded under "errors". Everything is
descriptive and in-sample; none of it is a forecast or advice.
"""
from __future__ import annotations

import importlib
import json

from dashboard import common

PATH = common.CACHE / "_studies2.json"
STUDIES = ("funding_positioning", "published_rules", "walkforward", "indicator_scan", "touch_calibration",
           "polymarket_odds", "episodes", "skew_history")  # page order
REQUIRED = ("id", "title", "question", "asof", "verdict", "tables", "notes", "sources")


def validate(result) -> list[str]:
    """Problems with one study's result (empty list = well formed)."""
    if not isinstance(result, dict):
        return ["結果不是物件"]
    problems = [f"缺少 {key}" for key in REQUIRED if key not in result]
    for key in ("title", "question", "verdict"):
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            problems.append(f"{key} 應該是非空字串")
    for index, row in enumerate(result.get("now") or []):
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or "value" not in row:
            problems.append(f"now[{index}] 格式不對")
    tables = result.get("tables")
    if not isinstance(tables, list):
        problems.append("tables 應該是陣列")
        tables = []
    for index, table in enumerate(tables):
        columns = table.get("columns") if isinstance(table, dict) else None
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(columns, list) or not columns or not isinstance(rows, list):
            problems.append(f"tables[{index}] 缺少 columns 或 rows")
            continue
        if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
            problems.append(f"tables[{index}] 有一列的欄數和表頭不一樣")
    for key in ("notes", "sources"):
        if key in result and not isinstance(result[key], list):
            problems.append(f"{key} 應該是陣列")
    return problems


def collect() -> list[dict]:
    try:
        previous = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        previous = {}
    old = {item.get("id"): item for item in previous.get("studies", []) if isinstance(item, dict)}
    payload = {"collected_utc": common.now_utc(), "errors": {}, "studies": []}
    for name in STUDIES:
        try:
            result = importlib.import_module(f"dashboard.studies2.{name}").study()
            problems = validate(result)
            if problems:
                raise ValueError("；".join(problems[:3]))
            json.dumps(result, allow_nan=False)  # NaN or infinity would break the page
            result["id"] = name
            result["computed_utc"] = payload["collected_utc"]
            payload["studies"].append(result)
        except Exception as exc:  # noqa: BLE001 - one study failing keeps its previous result
            payload["errors"][name] = f"{type(exc).__name__}: {exc}"[:200]
            if name in old:
                payload["studies"].append(old[name])
            print(f"  統計 {name} 這次沒有更新：{payload['errors'][name]}")
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for item in data["studies"]:
        print(item["id"], "->", item["verdict"])
    print("errors:", data["errors"])
