# -*- coding: utf-8 -*-
"""Reconstructs the executed notebook: notebook.ipynb cell sources + outputs.json
(text/plain outputs captured from run_code_cell) + cells_state.json (execution counts).

Run: .nb-venv/bin/python reconstruct_notebook.py
"""
import json, os, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))

nb = json.load(open(os.path.join(HERE, "notebook.ipynb")))
outputs = json.load(open(os.path.join(HERE, "outputs.json")))

# Map code-cell indices -> output text captured by run_cells.py
exec_count = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        exec_count += 1
        key = str(i)
        if key in outputs:
            text = outputs[key].get("output", "")
            if text:
                cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": text if text.endswith("\n") else text + "\n",
                }]
            else:
                cell["outputs"] = []
        else:
            cell["outputs"] = []
        cell["execution_count"] = exec_count

nb["metadata"]["colab"] = {
    "provenance": [],
    "gpuType": "T4",
    "toc_visible": True,
}
nb["metadata"]["executed"] = {
    "when": datetime.datetime.now().isoformat(timespec="seconds"),
    "runtime": "Google Colab, T4 (via colab-mcp)",
}

out_path = os.path.join(HERE, "assignment4_llm_classification_executed.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "| cells:", len(nb["cells"]), "| code cells with outputs:",
      sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs")))
