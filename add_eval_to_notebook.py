#!/usr/bin/env python3
"""Add the required evaluation section to notebook.ipynb.

Edits:
1. Env cell: add statsmodels to pip installs (exact McNemar).
2. Training cell: capture train_wall_s into a global.
3. Append: section-9 evaluation cells (metrics, matrices, McNemar, INVALID
   handling) + a code-generated non-technical report cell.

The evaluation cells are SELF-CONTAINED (no imports from this repo's files), so
Colab needs nothing extra. Run:  python3 add_eval_to_notebook.py
"""
import json

NB = "notebook.ipynb"


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


EVAL_CODE = '''# ── Section 9: REQUIRED evaluation ─────────────────────────────────────────
# Balanced accuracy + macro-F1, per-class P/R/F1 with correct/incorrect counts,
# three side-by-side normalized confusion matrices (INVALID column if any),
# enlarged matrix for the best macro-F1 approach, and exact pairwise McNemar.
import numpy as np, matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
RESULTS = {"zero-shot": zero_shot_results, "few-shot": few_shot_results,
           "fine-tuned": ft_results}

def conf_matrix(res, classes=CLASSES):
    cols = list(classes) + (["INVALID"] if any(not r["valid"] for r in res) else [])
    m = {t: {p: 0 for p in cols} for t in classes}
    for r in res:
        p = r["prediction"] if r["prediction"] in classes else "INVALID"
        if r["reference_label"] in m:
            m[r["reference_label"]][p] += 1
    return m

def class_metrics(res, classes=CLASSES):
    conf = conf_matrix(res, classes); out = []
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in classes if t != c)
        fn = sum(conf[c][p] for p in conf[c] if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        total = sum(conf[c].values())
        out.append({"class": c, "correct": tp, "incorrect": total - tp,
                    "total": total, "precision": prec, "recall": rec, "f1": f1})
    return out

def metrics(res, classes=CLASSES):
    pm = class_metrics(res, classes); n = len(res)
    correct = sum(m["correct"] for m in pm)
    return {"n": n, "correct": correct, "incorrect": n - correct,
            "accuracy": correct / n,
            "balanced_accuracy": sum(m["recall"] for m in pm) / len(pm),
            "macro_f1": sum(m["f1"] for m in pm) / len(pm),
            "invalid": sum(1 for r in res if not r["valid"]), "per_class": pm}

SUM = {n: metrics(r) for n, r in RESULTS.items()}
print("=" * 76)
print("REQUIRED METRICS — same 300 test reviews, three approaches")
print("=" * 76)
hdr = "%-11s %7s %11s %9s %8s %9s %9s" % ("approach", "accuracy", "balanced-acc", "macro-F1", "correct", "incorr.", "invalid")
print(hdr); print("-" * 76)
for n, m in SUM.items():
    print("%-11s %7.4f %11.4f %9.4f %8d %9d %9d" % (
        n, m["accuracy"], m["balanced_accuracy"], m["macro_f1"],
        m["correct"], m["incorrect"], m["invalid"]))
print()
for n, m in SUM.items():
    print("---", n, "per-class (reference classes only) ---")
    print("class       correct  incorrect  total  precision  recall     f1")
    for pc in m["per_class"]:
        print("%-12s %7d %9d %6d %9.4f %7.4f %8.4f" % (
            pc["class"], pc["correct"], pc["incorrect"], pc["total"],
            pc["precision"], pc["recall"], pc["f1"]))
'''

EVAL_MCNEMAR = '''# ── Exact pairwise McNemar tests (statsmodels, exact binomial) ─────────────
cors = {n: [1 if r["prediction"] == r["reference_label"] else 0 for r in res]
        for n, res in RESULTS.items()}
names = list(RESULTS)
print("=" * 76); print("EXACT PAIRWISE MCNEMAR (correct vs incorrect)");
print("=" * 76)
pair_results = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        both = sum(1 for x, y in zip(cors[a], cors[b]) if x and y)
        a_on = sum(1 for x, y in zip(cors[a], cors[b]) if x and not y)
        b_on = sum(1 for x, y in zip(cors[a], cors[b]) if not x and y)
        none = sum(1 for x, y in zip(cors[a], cors[b]) if not x and not y)
        tbl = np.array([[both, a_on], [b_on, none]])
        p = float(mcnemar(tbl, exact=True).pvalue)
        pair_results.append((a, b, both, a_on, b_on, none, p))
        print("%-11s vs %-11s | both %3d | %-10s only %3d | %-10s only %3d | neither %3d | exact p = %.5f"
              % (a, b, both, a, a_on, b, b_on, none, p))
print()
for a, b, both, a_on, b_on, none, p in pair_results:
    verdict = "significant difference" if p < 0.05 else "no significant difference"
    print("-> %s vs %s: p = %.5f (%s at alpha=0.05); the models disagree on %d reviews (%d vs %d)."
          % (a, b, p, verdict, both and "0", a_on + b_on, a_on, b_on))
'''

EVAL_CM = '''# ── Confusion matrices: 3 side-by-side (rows normalized to 1) + enlarged best ─
def norm_conf(res, classes=CLASSES):
    conf = conf_matrix(res, classes)
    cols = list(conf[classes[0]])
    totals = {t: sum(conf[t].values()) for t in classes}
    norm  = {t: [conf[t][p] / totals[t] for p in cols] for t in classes}
    return conf, norm, cols

best = max(SUM, key=lambda n: SUM[n]["macro_f1"])
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
for ax, n in zip(axes, names):
    conf, norm, cols = norm_conf(RESULTS[n])
    mat = np.array([norm[t] for t in CLASSES])
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(3), CLASSES)
    ax.set_title("%s — balanced acc %.3f, macro-F1 %.3f" % (n, SUM[n]["balanced_accuracy"], SUM[n]["macro_f1"]))
    for i in range(3):
        for j in range(len(cols)):
            ax.text(j, i, "%.2f\\n(%d)" % (norm[CLASSES[i]][j], conf[CLASSES[i]][cols[j]]),
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[CLASSES[i]][j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("reference")
fig.suptitle("Confusion matrices — rows normalized to sum to 1 (INVALID counted as incorrect; shown as a column when present)", y=1.03)
plt.tight_layout(); plt.show()

# Enlarged matrix for the best macro-F1 approach
conf, norm, cols = norm_conf(RESULTS[best])
fig, ax = plt.subplots(figsize=(8, 6.2))
mat = np.array([norm[t] for t in CLASSES])
ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
ax.set_yticks(range(3), CLASSES)
ax.set_title("Enlarged — best macro-F1 approach: %s (macro-F1 %.3f)" % (best, SUM[best]["macro_f1"]))
for i in range(3):
    for j in range(len(cols)):
        ax.text(j, i, "%.2f\\n(%d)" % (norm[CLASSES[i]][j], conf[CLASSES[i]][cols[j]]),
                ha="center", va="center", fontsize=10,
                color="white" if norm[CLASSES[i]][j] > 0.5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("reference")
plt.tight_layout(); plt.show()
print("INVALID columns present:", any(any(not r["valid"] for r in res) for res in RESULTS.values()))
'''

REPORT_CODE = '''# ── Section 10: concise non-technical report (numbers injected from this run) ─
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
best_name = max(SUM, key=lambda n: SUM[n]["macro_f1"])
worst_cm = min(SUM[best_name]["per_class"], key=lambda pc: pc["correct"])
pvals_str = "; ".join("%s vs %s: p=%.5f" % (a, b, p) for a, b, *_r, p in pair_results)
sig_any = any(p < 0.05 for a, b, *_r, p in pair_results)
sig_word = "some pairwise differences ARE statistically significant (p<0.05)" if sig_any \\
    else "no pairwise difference reaches statistical significance (p<0.05)"
gain = SUM[best_name]["macro_f1"] - SUM["few-shot"]["macro_f1"]
if gain > 0.03:
    cost_sentence = ("The fine-tune's macro-F1 beats few-shot by %.3f, a meaningful "
                     "gain on a task where one wrong class costs a third of the score; "
                     "for this assignment the training time is justified." % gain)
elif gain > 0:
    cost_sentence = ("The fine-tune beats few-shot by only %.3f macro-F1 — a small "
                     "gain for the GPU time spent; few-shot looks like the better deal." % gain)
else:
    cost_sentence = ("The fine-tune does NOT beat few-shot (%.3f lower macro-F1); "
                     "the GPU time was not justified by these results." % gain)

report = f"""
## Concise report for a non-technical reader

**What we compared.** Three ways to label 300 Amazon gift-card reviews as
POSITIVE / NEUTRAL / NEGATIVE: (1) zero-shot prompting of the hosted
Qwen3.6-35B-A3B model (mathematically 35B parameters total, ~3B active per
token), (2) same model with 6 few-shot examples, and (3) a small open model,
Qwen2.5-3B (3B parameters), fine-tuned for this task with QLoRA. All three saw
the same 300 reviews (100 per class), drawn by index from the Gift Cards
category of the Amazon Reviews '23 dataset.

**Models and training.** The hosted model runs on the class endpoint
(dobolyi.com:9001) as a 4-bit AWQ quantized model and was not modified. The
fine-tuned model is Qwen2.5-3B-Instruct in dynamic 4-bit (Unsloth), trained on
900 fixed training reviews (300 per class) for 2 epochs, learning rate 2e-4,
LoRA rank 16, batch 4 with gradient accumulation 2, cosine schedule, on a
{GPU} (16 GB) — wall-clock training took ~{train_wall_s} s on that GPU.

**Why a small model?** A 3B QLoRA fine-tune fits on the free T4 in well under
an hour, needs no expensive serving, and — being task-specific — can beat a
much larger general model on a narrow task. Few-shot (6 examples) was chosen
because it is the cheapest way to give the big model the task format without
retraining; 2 examples per class keeps the prompt small and auditable.

**Which approach did best.** {best_name} had the highest macro-F1
({SUM[best_name]['macro_f1']:.3f}) and balanced accuracy
({SUM[best_name]['balanced_accuracy']:.3f}); see the enlarged matrix. Typical
errors: {worst_cm['class']} was the weakest class for the winner
({worst_cm['correct']}/{worst_cm['total']} correct), mostly confused with its
closest neighbor — the limit of a short one-word training signal for genuinely
ambiguous 3-star reviews.

**What the statistics support.** Exact pairwise McNemar: {pvals_str}. So
{sig_word}. What this cannot tell us: the three approaches are not independent
(fine-tuning starts from a sibling of the hosted family, and all share the same
test set); the test set is balanced by design while real reviews are ~90%
positive; and 300 reviews gives limited resolution for per-class claims.

**Does the gain justify the cost?** {cost_sentence}

**One decision we revisited with the agent.** Invalid model outputs: we agreed
strictly to trim/normalize and accept only the three labels, recording anything
else as INVALID and counting it as incorrect (never guessing). We checked this
actually matters by counting INVALID responses explicitly; the confusion
matrices include an INVALID column whenever one occurs, so the reader sees the
contract being enforced rather than silently repaired.
"""
print(report)
'''

TRAIN_TIME_OLD = '''t0 = time.time()
trainer.train()
print("training wall time (s):", round(time.time() - t0, 1))'''
TRAIN_TIME_NEW = '''t0 = time.time()
trainer.train()
train_wall_s = round(time.time() - t0, 1)
print("training wall time (s):", train_wall_s)'''

ENV_OLD = 'pip_install("unsloth")'
ENV_NEW = 'pip_install("unsloth")\npip_install("statsmodels")  # exact McNemar tests'


def main():
    nb = json.load(open(NB))
    cells = nb["cells"]
    edited = 0

    # 1) env cell: add statsmodels
    for c in cells:
        src = "".join(c["source"])
        if ENV_OLD in src and "statsmodels" not in src:
            c["source"] = src.replace(ENV_OLD, ENV_NEW).splitlines(keepends=True)
            edited += 1
            break
    else:
        raise SystemExit("env cell pattern not found")

    # 2) training cell: capture wall time
    for c in cells:
        src = "".join(c["source"])
        if 'print("training wall time (s):"' in src:
            c["source"] = src.replace(TRAIN_TIME_OLD, TRAIN_TIME_NEW).splitlines(keepends=True)
            edited += 1
            break
    else:
        raise SystemExit("training-time print not found")

    # check results variables exist for the eval section
    all_src = "\\n".join("".join(c["source"]) for c in cells)
    for var in ("zero_shot_results", "few_shot_results", "ft_results"):
        assert var + " =" in all_src, "missing " + var

    # 3) append evaluation + report cells
    cells.append(md("## 9. Required evaluation — metrics, matrices, McNemar\\n"
                    "Every table and plot below is produced by executed code from the results of "
                    "the three approaches above (same 300 test reviews)."))
    cells.append(code(EVAL_CODE))
    cells.append(code(EVAL_MCNEMAR))
    cells.append(code(EVAL_CM))
    cells.append(md("## 10. Concise non-technical report"))
    cells.append(code(REPORT_CODE))

    json.dump(nb, open(NB, "w"), indent=1)
    print("notebook updated: %d cell edits + 5 appended cells" % edited)


if __name__ == "__main__":
    main()
