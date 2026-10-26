"""Extract selected node schemas from a dumped /object_info JSON.

Usage:
    python show_nodes.py <object_info.json> [node ...]
"""
from __future__ import annotations

import json
import sys

DEFAULT_NODES = [
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
    "LoadImage",
    "LoadAudio",
    "BasicScheduler",
    "BasicGuider",
    "CFGGuider",
    "KSamplerSelect",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "MiniMaxH3TurboLoRA",
    "MiniMaxH3TurboSampler",
    "SolAttentionPatch",
    "MiniMaxH3ChunkFeedForward",
    "MiniMaxH3MemoryEfficientSolAttentionPatch",
    "MiniMaxH3ScheduledSolAttentionPatch",
]


def main(argv: list[str]) -> int:
    path = argv[1]
    wanted = argv[2:] or DEFAULT_NODES
    with open(path, "r", encoding="utf-8-sig") as handle:
        info = json.load(handle)

    for node in wanted:
        entry = info.get(node)
        if entry is None:
            print(f"[MISSING] {node}")
            continue
        inputs = entry.get("input", {})
        required = list((inputs.get("required") or {}).keys())
        optional = list((inputs.get("optional") or {}).keys())
        print(f"[OK] {node}")
        print(f"     required: {required}")
        if optional:
            print(f"     optional: {optional}")
        for section in ("required", "optional"):
            for key, spec in (inputs.get(section) or {}).items():
                if isinstance(spec, list) and spec and isinstance(spec[0], list):
                    print(f"     - {key}: combo {spec[0][:6]}{'...' if len(spec[0]) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
