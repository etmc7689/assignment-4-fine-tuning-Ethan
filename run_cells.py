# -*- coding: utf-8 -*-
"""Runs every CODE cell of the connected Colab notebook in order via the bridge,
saving each cell's returned output text to outputs.json (keyed by cell index).
Long cells get a large timeout (training sets 5400s).

Run: .nb-venv/bin/python run_cells.py [start_index]
"""
import json, subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    with open(os.path.join(HERE, "cells_state.json")) as f:
        raw = json.loads(f.read())
except Exception as e:
    print("need inject_cells.py first:", e)
    sys.exit(1)

# cells_state may be a JSON array string or already parsed by the caller
if isinstance(raw, str):
    raw = json.loads(raw if raw.startswith("[") else f.read())
cells = raw if isinstance(raw, list) else raw.get("cells", raw)

CODE_CELL_TIMEOUT = 5400
LONG_CELLS = {21}  # index of the training cell (0-based CELLS order) — see inject log


def call(tool, args, timeout=600):
    req = json.dumps({"tool": tool, "args": args, "timeout": timeout})
    out = subprocess.run([sys.executable, os.path.join(HERE, "mcp_call.py"), req],
                         capture_output=True, text=True, timeout=timeout + 30)
    if out.returncode != 0:
        print("mcp_call stderr:", out.stderr[:400])
    return out.stdout.strip()


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    outputs = {}
    if os.path.exists(os.path.join(HERE, "outputs.json")):
        with open(os.path.join(HERE, "outputs.json")) as f:
            outputs = json.load(f)

    ids_by_index = {}
    for c in cells:
        idx = c.get("index") if isinstance(c, dict) and "index" in c else None
        ids_by_index.setdefault(idx if idx is not None else len(ids_by_index),
                                c.get("cellId") or c.get("id") or c.get("cell_id"))

    for i, (kind, _src) in enumerate(__import__("build_notebook").CELLS):
        if kind != "code" or i < start:
            continue
        cid = ids_by_index.get(i)
        if not cid:
            print("no cellId for index", i)
            continue
        print(f"--- running cell {i} ({cid}) ---", flush=True)
        t0 = time.time()
        resp = call("run_code_cell", {"cellId": cid},
                    timeout=CODE_CELL_TIMEOUT if i in LONG_CELLS else 600)
        dt = time.time() - t0
        outputs[str(i)] = {"cellId": cid, "elapsed_s": round(dt, 1), "output": resp}
        with open(os.path.join(HERE, "outputs.json"), "w") as f:
            json.dump(outputs, f, indent=1)
        print(f"--- cell {i} finished in {dt:.0f}s; output head/tail ---")
        print(resp[:600])
        if len(resp) > 1200:
            print("...", resp[-600:])
        print()
    print("All cells run. outputs saved to outputs.json")


if __name__ == "__main__":
    main()
