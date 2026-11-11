"""Check dashboard/out/data.json against the data contract in dashboard/SPEC2.md, section 3.

Run from the project root (after dashboard.build):
    python -m dashboard.check
Prints every problem found and exits with status 1 if there is any. Only reads the file; changes nothing.
"""
from __future__ import annotations

import json
import math
import re
import sys

from dashboard import common, scoring

DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ZONE_KEYS = [zone[0] for zone in scoring.ZONES]
ZONE_LABELS = {zone[0]: zone[1] for zone in scoring.ZONES}
TOP = ("built_utc", "disclaimer", "assets", "indicators", "sources", "tools")
ASSET = ("asset", "name", "price", "price_display", "change_24h", "change_7d", "change_30d", "temperature", "zone",
         "zone_label", "temp_change_7d", "temp_change_30d", "summary", "components", "temp_history",
         "price_history", "zone_stats", "rules")
COMPONENT = ("id", "name", "heat", "display", "plain", "counted")
ZONE_STAT = ("zone", "label", "days", "fwd30_median", "fwd30_up", "fwd90_median", "fwd90_up", "current")
RULE = ("name", "state", "detail")
INDICATOR = ("id", "group", "asset", "name", "value", "unit", "display", "asof", "source", "source_url", "evidence",
             "reading", "rule", "note", "history", "error", "hot_when", "pct", "heat", "key", "plain")
SOURCE = ("name", "collected_utc", "failed", "error")
TOOL = ("name", "url", "category", "what")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bad_numbers(value, where: str, problems: list) -> None:
    """NaN and infinity anywhere in the document (json.loads accepts them, browsers do not)."""
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{where}：出現 NaN 或無限大")
    elif isinstance(value, dict):
        for key, item in value.items():
            _bad_numbers(item, f"{where}.{key}", problems)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _bad_numbers(item, f"{where}[{index}]", problems)


def _fields(row, names, where: str, problems: list) -> bool:
    if not isinstance(row, dict):
        problems.append(f"{where}：不是物件")
        return False
    missing = [name for name in names if name not in row]
    if missing:
        problems.append(f"{where}：缺少欄位 {missing}")
    return not missing


def _history(history, where: str, problems: list, low=None, high=None, limit=common.HISTORY_POINTS) -> None:
    if not isinstance(history, list):
        problems.append(f"{where}：不是清單")
        return
    if len(history) > limit:
        problems.append(f"{where}：{len(history)} 筆，超過上限 {limit}")
    previous = ""
    for point in history:
        if not (isinstance(point, list) and len(point) == 2 and isinstance(point[0], str) and DAY.match(point[0])
                and _number(point[1])):
            problems.append(f"{where}：有格式不對的點 {str(point)[:60]}")
            return
        if point[0] <= previous:
            problems.append(f"{where}：日期沒有由舊到新，或同一天重複（{previous} 之後是 {point[0]}）")
            return
        if (low is not None and point[1] < low) or (high is not None and point[1] > high):
            problems.append(f"{where}：{point[0]} 的值 {point[1]} 超出 {low}–{high}")
            return
        previous = point[0]


def _score(value, where: str, problems: list) -> None:
    if value is not None and not (_number(value) and 0 <= value <= 100):
        problems.append(f"{where}：{value!r} 不在 0–100")


def _text(value, where: str, problems: list, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}：應該是非空白文字，實際是 {value!r}")


def check(data: dict) -> list[str]:
    problems: list[str] = []
    _bad_numbers(data, "data", problems)
    if not _fields(data, TOP, "data", problems):
        return problems
    _text(data["built_utc"], "built_utc", problems)
    _text(data["disclaimer"], "disclaimer", problems)

    seen = set()
    by_id = {}
    for row in data["indicators"]:
        where = f"indicators[{row.get('id') if isinstance(row, dict) else '?'}]"
        if not _fields(row, INDICATOR, where, problems):
            continue
        if row["id"] in seen:
            problems.append(f"{where}：id 重複")
        seen.add(row["id"])
        by_id[row["id"]] = row
        if row["group"] not in common.GROUPS:
            problems.append(f"{where}：group {row['group']!r} 不在名單內")
        if row["hot_when"] not in ("high", "low", None):
            problems.append(f"{where}：hot_when {row['hot_when']!r} 不合法")
        if not isinstance(row["key"], bool):
            problems.append(f"{where}：key 不是布林值")
        _text(row["plain"], f"{where}.plain", problems)
        _score(row["pct"], f"{where}.pct", problems)
        _score(row["heat"], f"{where}.heat", problems)
        _history(row["history"], f"{where}.history", problems)
        if row["pct"] is not None and len(row["history"]) < scoring.MIN_PCT_POINTS:
            problems.append(f"{where}：歷史不到 {scoring.MIN_PCT_POINTS} 筆卻有 pct")
        if (row["heat"] is None) != (row["hot_when"] is None or row["pct"] is None):
            problems.append(f"{where}：heat 與 hot_when／pct 不一致")
        if row["heat"] is not None:
            expect = row["pct"] if row["hot_when"] == "high" else 100 - row["pct"]
            if abs(row["heat"] - expect) > 0.11:
                problems.append(f"{where}：heat {row['heat']} 應為 {expect}")
        if row["error"] is None and row["value"] is not None and not _number(row["value"]):
            problems.append(f"{where}：value 不是數字")
    keys = sum(1 for row in by_id.values() if row["key"])
    if not 8 <= keys <= 24:
        problems.append(f"重點指標（key）有 {keys} 個，應該在 8 到 24 個之間")

    if [block.get("asset") for block in data["assets"] if isinstance(block, dict)] != list(scoring.COMPONENTS):
        problems.append(f"assets 應該依序是 {list(scoring.COMPONENTS)}")
    for block in data["assets"]:
        where = f"assets[{block.get('asset') if isinstance(block, dict) else '?'}]"
        if not _fields(block, ASSET, where, problems):
            continue
        temp = block["temperature"]
        _score(temp, f"{where}.temperature", problems)
        if temp is None:
            if block["zone"] is not None or block["zone_label"] is not None:
                problems.append(f"{where}：沒有溫度時 zone 應為 null")
        else:
            expect = scoring.zone_of(temp)[0]
            if block["zone"] != expect or block["zone_label"] != ZONE_LABELS.get(expect):
                problems.append(f"{where}：溫度 {temp} 應在 {expect} 區，實際 {block['zone']}／{block['zone_label']}")
        for name in ("price", "change_24h", "change_7d", "change_30d", "temp_change_7d", "temp_change_30d"):
            if block[name] is not None and not _number(block[name]):
                problems.append(f"{where}.{name}：不是數字")
        if not isinstance(block["summary"], list) or not block["summary"]:
            problems.append(f"{where}.summary：應該是非空清單")
        else:
            for index, line in enumerate(block["summary"]):
                _text(line, f"{where}.summary[{index}]", problems)
            if temp is not None and not 3 <= len(block["summary"]) <= 5:
                problems.append(f"{where}.summary：{len(block['summary'])} 句，應該 3 到 5 句")
        wanted = [id for id, _ in scoring.COMPONENTS.get(block["asset"], ())]
        if [c.get("id") for c in block["components"] if isinstance(c, dict)] != wanted:
            problems.append(f"{where}.components：成分清單和 SPEC2 的表不一致")
        counted = []
        for component in block["components"]:
            cwhere = f"{where}.components[{component.get('id') if isinstance(component, dict) else '?'}]"
            if not _fields(component, COMPONENT, cwhere, problems):
                continue
            _score(component["heat"], f"{cwhere}.heat", problems)
            _text(component["name"], f"{cwhere}.name", problems)
            _text(component["plain"], f"{cwhere}.plain", problems)
            if component["counted"]:
                source = by_id.get(component["id"])
                if component["heat"] is None:
                    problems.append(f"{cwhere}：計入了卻沒有 heat")
                    continue
                counted.append(component["heat"])
                if source is None or source["heat"] != component["heat"]:
                    problems.append(f"{cwhere}：heat 和指標清單裡的不一樣")
                if source is not None and len(source["history"]) < scoring.MIN_COMPONENT_POINTS:
                    problems.append(f"{cwhere}：歷史不到 {scoring.MIN_COMPONENT_POINTS} 筆卻被計入")
        if temp is None and len(counted) >= scoring.MIN_COMPONENTS:
            problems.append(f"{where}：有 {len(counted)} 個可用成分卻沒有溫度")
        if temp is not None:
            if len(counted) < scoring.MIN_COMPONENTS:
                problems.append(f"{where}：可用成分只有 {len(counted)} 個卻給了溫度")
            elif abs(sum(counted) / len(counted) - temp) > 0.06:
                problems.append(f"{where}：溫度 {temp} 不等於成分的等權平均 {sum(counted) / len(counted):.2f}")
        _history(block["temp_history"], f"{where}.temp_history", problems, 0, 100)
        _history(block["price_history"], f"{where}.price_history", problems, 0)
        if temp is not None:
            if not block["temp_history"]:
                problems.append(f"{where}：有溫度卻沒有溫度歷史")
            elif abs(block["temp_history"][-1][1] - temp) > 0.06:
                problems.append(f"{where}：溫度歷史最後一筆 {block['temp_history'][-1][1]} 和目前溫度 {temp} 不一致")
        if [row.get("zone") for row in block["zone_stats"] if isinstance(row, dict)] not in (ZONE_KEYS, []):
            problems.append(f"{where}.zone_stats：應該依序是五個分區")
        total = 0
        for row in block["zone_stats"]:
            swhere = f"{where}.zone_stats[{row.get('zone') if isinstance(row, dict) else '?'}]"
            if not _fields(row, ZONE_STAT, swhere, problems):
                continue
            total += row["days"] if isinstance(row["days"], int) else 0
            if not isinstance(row["days"], int) or row["days"] < 0:
                problems.append(f"{swhere}.days：不是非負整數")
            _score(row["fwd30_up"], f"{swhere}.fwd30_up", problems)
            _score(row["fwd90_up"], f"{swhere}.fwd90_up", problems)
            for name in ("fwd30_median", "fwd90_median"):
                if row[name] is not None and not _number(row[name]):
                    problems.append(f"{swhere}.{name}：不是數字")
            if row["current"] != (row["zone"] == block["zone"]):
                problems.append(f"{swhere}.current：和目前分區不一致")
        if block["zone_stats"] and total != len(block["temp_history"]):
            problems.append(f"{where}.zone_stats：天數合計 {total} 不等於溫度歷史 {len(block['temp_history'])} 筆")
        for index, rule in enumerate(block["rules"]):
            if _fields(rule, RULE, f"{where}.rules[{index}]", problems):
                for name in RULE:
                    _text(rule[name], f"{where}.rules[{index}].{name}", problems)

    for index, source in enumerate(data["sources"]):
        _fields(source, SOURCE, f"sources[{index}]", problems)
    for index, tool in enumerate(data["tools"]):
        _fields(tool, TOOL, f"tools[{index}]", problems)
    return problems


def main() -> int:
    path = common.OUT / "data.json"
    if not path.exists():
        print(f"找不到 {path}，請先執行 python -m dashboard.build")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = check(data)
    lengths = sorted(len(row.get("history") or []) for row in data.get("indicators", []) if isinstance(row, dict))
    print(f"檢查 {path}：指標 {len(lengths)} 個、資產 {len(data.get('assets', []))} 個、"
          f"歷史滿 {common.HISTORY_POINTS} 筆的指標 {sum(1 for n in lengths if n >= common.HISTORY_POINTS)} 個")
    for problem in problems:
        print("  問題：" + problem)
    print("結果：" + ("符合資料契約" if not problems else f"有 {len(problems)} 個問題"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
