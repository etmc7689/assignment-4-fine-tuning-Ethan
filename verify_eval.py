#!/usr/bin/env python3
"""Local verification of the Assignment-4 evaluation pipeline.

Runs evaluation_utils against the REAL hosted predictions (zero-shot,
few-shot) plus a clearly-labelled SYNTHETIC stand-in for the fine-tuned
approach (real fine-tuned numbers are produced by running notebook.ipynb
in Colab). Produces the required tables, McNemar results, and the matrix
PNGs, proving the code paths work end to end.
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluation_utils as ev

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
zero = ev.load_results(HERE / "results_zero_shot.jsonl")
few = ev.load_results(HERE / "results_few_shot.jsonl")

print("=" * 74)
print("REQUIRED METRICS — same 300 test reviews (zero-shot / few-shot REAL,")
print("fine-tuned SYNTHETIC-VERIFY-ONLY here; real in Colab)")
print("=" * 74)
names = ["zero-shot", "few-shot"]
res_map = {"zero-shot": zero, "few-shot": few}

# synthetic fine-tuned stand-in (VERIFY ONLY)
rng = random.Random(43)
syn = []
for r in zero:
    roll = rng.random()
    if roll < 0.83:
        pred = r["reference_label"]
    elif roll < 0.91:
        pred = "NEUTRAL" if r["reference_label"] != "NEUTRAL" else "POSITIVE"
    else:
        pred = "NEGATIVE" if r["reference_label"] != "NEGATIVE" else "NEUTRAL"
    syn.append({"filtered_index": r["filtered_index"],
                "reference_label": r["reference_label"],
                "prediction": pred, "valid": True,
                "raw_output": "SYNTHETIC (replace with real Colab fine-tuned run)"})
res_map["fine-tuned"] = syn
names.append("fine-tuned")

summ = {}
for n in names:
    m = ev.summary_metrics(res_map[n])
    summ[n] = m
    print(f"{n:>11s}: acc={m['accuracy']:.4f} bal-acc={m['balanced_accuracy']:.4f} "
          f"macro-F1={m['macro_f1']:.4f} | correct={m['correct']:3d} incorrect={m['incorrect']:3d} "
          f"invalid={m['invalid']}")
    for pc in m["per_class"]:
        print("       %-8s correct=%-3d incorrect=%-3d P=%.3f R=%.3f F1=%.3f"
              % (pc["class"], pc["correct"], pc["incorrect"],
                 pc["precision"], pc["recall"], pc["f1"]))

print("\n" + "=" * 74)
print("EXACT PAIRWISE MCNEMAR (correct vs incorrect)")
print("=" * 74)
for r in ev.mcnemar_pairwise(res_map):
    t = r["table"]
    print("%-11s vs %-11s | both=%3d  %s-only=%3d  %s-only=%3d  neither=%3d | p=%.5f"
          % (r["a"], r["b"], t[0][0], r["a"], t[0][1], r["b"], t[1][0], t[1][1],
             r["pvalue"]))

print("\nWriting confusion-matrix figures ...")
best = ev.plot_matrices(res_map, out_prefix=str(HERE / "cm"))
print("best macro-F1 approach (verify-only supply):", best)
for p in sorted((HERE).glob("cm_*.png")):
    print("  wrote", p.name)
print("\nVERIFICATION OK — pipeline works end to end.")
