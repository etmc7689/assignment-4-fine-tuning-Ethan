# -*- coding: utf-8 -*-
"""Injects the notebook's 26 cells (markdown + code) into the connected Colab notebook
via the bridge, in order, then snapshots get_cells to cells_state.json.

Run: .nb-venv/bin/python inject_cells.py
"""
import json, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_notebook import CELLS


def call(tool, args, timeout=600):
    req = json.dumps({"tool": tool, "args": args, "timeout": timeout})
    out = subprocess.run([sys.executable, os.path.join(HERE, "mcp_call.py"), req],
                         capture_output=True, text=True, timeout=timeout + 30)
    if out.returncode != 0:
        print("mcp_call stderr:", out.stderr[:400])
    return out.stdout.strip()


def main():
    state = []
    for i, (kind, src) in enumerate(CELLS):
        if kind == "code":
            resp = call("add_code_cell", {"code": src, "cellIndex": i})
        else:
            resp = call("add_text_cell", {"content": src, "cellIndex": i})
        state.append({"index": i, "kind": kind, "resp": resp})
        print(i, kind, resp[:160])
    with open(os.path.join(HERE, "inject_log.json"), "w") as f:
        json.dump(state, f, indent=1)
    cells = call("get_cells", {})
    with open(os.path.join(HERE, "cells_state.json"), "w") as f:
        f.write(cells)
    print("--- get_cells snapshotted to cells_state.json ---")
    print(cells[:1200])


if __name__ == "__main__":
    main()
