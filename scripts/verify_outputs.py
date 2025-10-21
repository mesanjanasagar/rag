#!/usr/bin/env python3
"""Verify that outputs_hybrid.jsonl matches format_hint in the sample questions.

Exit code 0 when all checks pass, non-zero otherwise.
"""
import json
import sys
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f.readlines() if l.strip()]


def valid_format(value, fmt_hint):
    try:
        if fmt_hint == "int":
            if isinstance(value, bool):
                return False
            int(value)
            return True
        if fmt_hint == "float":
            if isinstance(value, bool):
                return False
            float(value)
            return True
        if isinstance(fmt_hint, str) and fmt_hint.strip().startswith("{"):
            # must be dict
            return isinstance(value, dict)
        if isinstance(fmt_hint, str) and fmt_hint.strip().startswith("list"):
            return isinstance(value, list)
        return True
    except Exception:
        return False


def main():
    base = Path(__file__).resolve().parent.parent
    sample = load_jsonl(base / "sample_questions_hybrid_eval.jsonl")
    outputs = load_jsonl(base / "outputs_hybrid.jsonl")

    # index sample by id
    smap = {s["id"]: s for s in sample}
    ok = True
    for o in outputs:
        qid = o.get("id")
        if qid not in smap:
            print(f"Unknown output id: {qid}")
            ok = False
            continue
        fmt = smap[qid].get("format_hint")
        val = o.get("final_answer")
        if not valid_format(val, fmt):
            print(f"FORMAT MISMATCH for {qid}: expected {fmt}, got type {type(val).__name__} -> {val}")
            ok = False

    if not ok:
        print("Verification failed.")
        sys.exit(2)
    print("All outputs match their format hints.")


if __name__ == "__main__":
    main()
