"""Evaluation utilities for Assignment 4 (three-approach comparison).

Consumes per-approach result lists:
    [{"filtered_index": int, "reference_label": str, "prediction": str|None,
      "valid": bool, "raw_output": str}, ...]

Rules enforced (assignment spec):
- Trim whitespace + normalize case; only POSITIVE/NEUTRAL/NEGATIVE accepted.
- Anything else (prediction is None) => INVALID, counted as incorrect, and an
  INVALID prediction column appears in the matrices if any occur.
- Metrics average over the THREE reference classes only (INVALID is not a
  reference class). Balanced accuracy = mean per-class recall.
- Exact pairwise McNemar tests via statsmodels (binomial-exact).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def confusion_matrix(results, classes=CLASSES, include_invalid=True):
    """rows=reference classes, cols=predicted (+ INVALID column if any)."""
    cols = classes + (["INVALID"] if include_invalid else [])
    m = {t: {p: 0 for p in cols} for t in classes}
    for r in results:
        t = r["reference_label"]
        p = r["prediction"]
        key = p if p in classes else "INVALID"
        if t in m:
            m[t][key] += 1
    return m


def per_class_metrics(results, classes=CLASSES):
    """Per-class precision/recall/F1 + correct/incorrect counts (3 classes only)."""
    conf = confusion_matrix(results, classes)
    out = []
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in classes if t != c)
        fn = sum(conf[c][p] for p in conf[c] if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        total = sum(conf[c].values())
        out.append({"class": c, "tp": tp, "correct": tp,
                    "incorrect": total - tp, "total": total,
                    "precision": prec, "recall": rec, "f1": f1})
    return out


def summary_metrics(results, classes=CLASSES):
    """accuracy, balanced accuracy (mean recall over 3 classes), macro-F1, invalid."""
    n = len(results)
    pm = per_class_metrics(results, classes)
    correct = sum(m["correct"] for m in pm)
    acc = correct / n if n else 0.0
    bal_acc = sum(m["recall"] for m in pm) / len(pm)
    macro_f1 = sum(m["f1"] for m in pm) / len(pm)
    invalid = sum(1 for r in results if not r["valid"])
    return {"n": n, "correct": correct, "incorrect": n - correct,
            "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
            "invalid": invalid, "per_class": pm}


def mcnemar_pairwise(results_map, classes=CLASSES):
    """Exact McNemar on correct/incorrect discordant pairs, for every pair."""
    ok = {}
    for name, res in results_map.items():
        ok[name] = [1 if r["prediction"] == r["reference_label"] else 0
                    for r in res]
    names = list(results_map)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            both = sum(1 for x, y in zip(ok[a], ok[b]) if x == 1 and y == 1)
            a_only = sum(1 for x, y in zip(ok[a], ok[b]) if x == 1 and y == 0)
            b_only = sum(1 for x, y in zip(ok[a], ok[b]) if x == 0 and y == 1)
            neither = sum(1 for x, y in zip(ok[a], ok[b]) if x == 0 and y == 0)
            table = np.array([[both, a_only], [b_only, neither]])
            res = mcnemar(table, exact=True)
            out.append({"a": a, "b": b, "table": table,
                        "a_only": int(a_only), "b_only": int(b_only),
                        "statistic": float(res.statistic),
                        "pvalue": float(res.pvalue)})
    return out


def normalized_matrix_html_or_values(conf, classes=CLASSES):
    """Row-normalized matrix (rows sum to 1) with raw counts for annotation."""
    cols = list(next(iter(conf.values())))
    norm = {}
    for t in classes:
        total = sum(conf[t].values())
        norm[t] = {p: (conf[t][p] / total if total else 0.0) for p in cols}
    return norm, cols


def plot_matrices(results_map, out_prefix="cm", classes=CLASSES):
    """Three normalized matrices side by side + enlarged best macro-F1 matrix."""
    summ = {n: summary_metrics(res, classes) for n, res in results_map.items()}
    best = max(summ, key=lambda k: summ[k]["macro_f1"])
    names = list(results_map)

    # --- side-by-side ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, name in zip(axes, names):
        conf = confusion_matrix(results_map[name], classes)
        norm, cols = normalized_matrix_html_or_values(conf, classes)
        mat = np.array([[norm[t][p] for p in cols] for t in classes])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes)), classes, fontsize=9)
        ax.set_title("%s\nmacro-F1 %.3f" % (name, summ[name]["macro_f1"]), fontsize=10)
        for i in range(len(classes)):
            for j in range(len(cols)):
                ax.text(j, i, "%.2f\n(%d)" % (norm[classes[i]][cols[j]], conf[classes[i]][cols[j]]),
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[classes[i]][cols[j]] > 0.5 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("reference")
    fig.suptitle("Confusion matrices, rows normalized to sum to 1 (INVALID counted as incorrect)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_prefix + "_side_by_side.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- enlarged best ---
    conf = confusion_matrix(results_map[best], classes)
    norm, cols = normalized_matrix_html_or_values(conf, classes)
    mat = np.array([[norm[t][p] for p in cols] for t in classes])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(classes)), classes, fontsize=11)
    ax.set_title("Best macro-F1 approach: %s (macro-F1 %.3f)" % (best, summ[best]["macro_f1"]), fontsize=12)
    for i in range(len(classes)):
        for j in range(len(cols)):
            ax.text(j, i, "%.2f\n(%d)" % (norm[classes[i]][cols[j]], conf[classes[i]][cols[j]]),
                    ha="center", va="center", fontsize=10,
                    color="white" if norm[classes[i]][cols[j]] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    fig.tight_layout()
    fig.savefig(out_prefix + "_best_%s.png" % best.replace("/", "_"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return best


def load_results(path):
    return [json.loads(l) for l in open(path) if l.strip()]
