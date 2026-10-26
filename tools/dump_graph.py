"""Dump a ComfyUI workflow JSON's node graph (class_type + inputs) for reference.

Usage:
    python dump_graph.py <workflow.json> [--api]

For API-format workflows ({"nodes": {...}}), it prints the raw prompt dict.
For UI-format workflows ({"nodes": [...], "links": [...]}), it prints a
compact summary of each node.
"""
from __future__ import annotations

import json
import sys


def dump_ui_format(wf: dict) -> None:
    for node in wf.get("nodes", []):
        inputs = node.get("inputs") or []
        names = []
        for slot in inputs:
            if isinstance(slot, dict):
                names.append(f"{slot.get('name')}:{slot.get('type')}")
            else:
                names.append(str(slot))
        widgets = node.get("widgets_values") or []
        print(f"[{node.get('id'):>4}] {node.get('type')}  inputs={names} widgets={widgets}")


def dump_api_format(wf: dict) -> None:
    for nid, node in (wf.get("nodes") or wf).items():
        ct = node.get("class_type")
        inputs = node.get("inputs", {})
        rendered = {}
        for key, value in inputs.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                rendered[key] = value
            else:
                rendered[key] = value
        print(f"[{nid:>4}] {ct}  {json.dumps(rendered, ensure_ascii=False)}")


def main(argv: list[str]) -> int:
    path = argv[1]
    api = "--api" in argv
    with open(path, "r", encoding="utf-8") as handle:
        wf = json.load(handle)
    if api or "class_type" in json.dumps(wf)[:4000]:
        dump_api_format(wf)
    else:
        dump_ui_format(wf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
