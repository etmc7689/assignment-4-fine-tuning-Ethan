# -*- coding: utf-8 -*-
"""Builds notebook.ipynb — the executed Colab notebook for Assignment 4 (LLM classification comparison).

Every cell source lives here so the exact notebook can be rebuilt/reconstructed,
including after the Colab run (cell sources + captured outputs).
Run: .nb-venv/bin/python build_notebook.py
"""
import json, pathlib

OUT = pathlib.Path(__file__).parent / "notebook.ipynb"


def md(src):
    return ("markdown", src)


def code(src):
    return ("code", src)


CELLS = []

# ------------------------------------------------------------------ 0. Title
CELLS.append(md("""# Comparing LLM Classification Approaches: Zero-Shot vs Few-Shot vs QLoRA Fine-Tune

**Task:** classify Amazon Reviews '23 **Gift Cards** reviews as `POSITIVE`, `NEUTRAL`, or `NEGATIVE`
(`4-5` stars = POSITIVE, `3` = NEUTRAL, `1-2` = NEGATIVE).

**Three approaches, same 300 test reviews:**

1. **Zero-shot** prompting of a hosted model (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` via the class endpoint at `http://dobolyi.com:9001/v1`),
2. **Few-shot** prompting of the same model (6 examples, 2 per class, all drawn from the prescribed training set),
3. **Fine-tuned** small open-weight model (`Qwen2.5-3B-Instruct`, 4-bit QLoRA, trained with **Unsloth** on the prescribed 900 training reviews).

**Safeguards enforced in this notebook:** prediction inputs are review title + text only (never the rating or the reference label);
no test reviews are used for training or few-shot examples; invalid model outputs are recorded, never guessed;
all fixed-split assertions run before training and before every scoring step and stop execution on failure.
The API key is read from a Colab secret and is never printed.
"""))
# ------------------------------------------------------------------ 1. Setup
CELLS.append(md("""## 1. Environment

Installs Unsloth, its dependencies, and verifies the GPU/runtime. The actual GPU used for this run is reported
in the fine-tuning section (T4 compatibility is tested directly, on-device)."""))
CELLS.append(code("""import sys, subprocess, time, os

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

pip_install("unsloth")
try:
    import unsloth
    print("unsloth", getattr(unsloth, "__version__", "?"))
except Exception as e:
    print("PyPI unsloth import failed (%s); installing from git..." % e)
    pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    import unsloth

import torch, transformers, datasets, requests
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""))
# ------------------------------------------------------------------ 2. Data
CELLS.append(md("""## 2. Fixed dataset and prescribed splits (do not change)

Source: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz`

Rules implemented exactly as prescribed:

* Preserve file order; `original_row` = original 1-based line number.
* Skip blank lines and reviews with fewer than 3 characters of text after trimming; missing text is treated as empty.
* The remaining reviews are numbered `filtered_index`, starting at 0.
* **Training:** `filtered_index` 1,000–19,999 → `random.Random(43).sample(300, per class)` in order POSITIVE, NEUTRAL, NEGATIVE, then one `shuffle` of the combined 900 with that same generator.
* **Validation:** indices 20,000–29,999, first 100 per class (reserved; not used in this assignment).
* **Test:** indices 30,000–39,999, first 100 per class (300 total) — the same 300 reviews for all three approaches."""))
CELLS.append(code("""import gzip, json, urllib.request, random

DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
DATA_PATH = "Gift_Cards.jsonl.gz"
if not os.path.exists(DATA_PATH):
    print("Downloading dataset ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
print("file size MB:", round(os.path.getsize(DATA_PATH) / 1e6, 2))

def classify(rating):
    if rating is None:
        return None
    r = float(rating)
    if r >= 4.0: return "POSITIVE"
    if r == 3.0: return "NEUTRAL"
    if r <= 2.0: return "NEGATIVE"
    return None

records = []
original_row = 0
skipped_blank, skipped_short = 0, 0
with gzip.open(DATA_PATH, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        original_row += 1
        s = line.strip()
        if not s:
            skipped_blank += 1
            continue
        try:
            d = json.loads(s)
        except ValueError:
            skipped_blank += 1  # unparseable line: skipped like a blank
            continue
        t = d.get("text") or ""
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if len(t) < 3:
            skipped_short += 1
            continue
        records.append({
            "original_row": original_row,
            "filtered_index": len(records),
            "rating": d.get("rating"),
            "title": (d.get("title") or "").strip(),
            "text": t,
            "label": classify(d.get("rating")),
        })

print("raw lines read      :", original_row)
print("valid reviews       :", len(records))
print("skipped blank/malformed:", skipped_blank, "| skipped <3 chars text:", skipped_short)
"""))
CELLS.append(code("""CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

def pool_ids(lo, hi, cls):
    return [r["filtered_index"] for r in records if lo <= r["filtered_index"] <= hi and r["label"] == cls]

# --- Training: random.Random(43), sample per class in order, then shuffle ---
rng = random.Random(43)
train_ids = []
for c in CLASSES:
    train_ids.extend(rng.sample(pool_ids(1000, 19999, c), 300))
rng.shuffle(train_ids)

# --- Validation (reserved, unused in this assignment): first 100 per class ---
val_ids = []
for c in CLASSES:
    val_ids.extend(pool_ids(20000, 29999, c)[:100])

# --- Test: first 100 per class ---
test_ids = []
for c in CLASSES:
    test_ids.extend(pool_ids(30000, 39999, c)[:100])

by_id = {r["filtered_index"]: r for r in records}
train_records = [by_id[i] for i in train_ids]
val_records   = [by_id[i] for i in val_ids]
test_records  = [by_id[i] for i in test_ids]

# ----- 3. REQUIRED DATA CHECKS (failures stop everything downstream) -----
def check(cond, msg):
    if not cond:
        raise AssertionError("FAILED CHECK: " + msg)
    print("PASS:", msg)

check(len(train_ids) == 900 and len(val_ids) == 300 and len(test_ids) == 300, "set sizes are 900 / 300 / 300")
check(len(set(train_ids)) == 900 and len(set(val_ids)) == 300 and len(set(test_ids)) == 300, "IDs unique within each set")
check(len(set(train_ids) & set(val_ids)) == 0 and len(set(train_ids) & set(test_ids)) == 0 \
      and len(set(val_ids) & set(test_ids)) == 0, "no shared IDs across sets")
for c in CLASSES:
    check(sum(1 for r in train_records if r["label"] == c) == 300, "train has exactly 300 " + c)
    check(sum(1 for r in val_records if r["label"] == c) == 100,   "val has exactly 100 " + c)
    check(sum(1 for r in test_records if r["label"] == c) == 100,  "test has exactly 100 " + c)
check(all(1000 <= i <= 19999 for i in train_ids), "train indices within 1,000-19,999")
check(all(20000 <= i <= 29999 for i in val_ids),  "val indices within 20,000-29,999")
check(all(30000 <= i <= 39999 for i in test_ids), "test indices within 30,000-39,999")

from collections import Counter
print("train class distribution:", dict(Counter(r["label"] for r in train_records)))
print("val   class distribution:", dict(Counter(r["label"] for r in val_records)))
print("test  class distribution:", dict(Counter(r["label"] for r in test_records)))
print("first train records:", [(r["filtered_index"], r["label"]) for r in train_records[:3]])
"""))
# ------------------------------------------------------------------ 3. Prompts
CELLS.append(md("""## 3. Prompt templates, few-shot selection, and output contract

* **Inputs:** review title + text only — the target review's rating and reference label never appear in any prompt.
* **Output contract:** the model must reply with exactly one word, `POSITIVE` / `NEUTRAL` / `NEGATIVE`.
  Any other output is recorded as **invalid** (raw text kept in the results) — never replaced by a guessed label.
* **Few-shot:** 6 examples (2 per class), each drawn from the prescribed 900 training reviews:
  the two lowest `filtered_index` reviews of each class within the training set (deterministic, auditable)."""))
CELLS.append(code("""SYSTEM_PROMPT = "Classify the sentiment of the Amazon review below. Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."

def review_text(r):
    title = (r.get("title") or "").strip()
    body = (r.get("text") or "").strip()
    return ("Title: " + title + "\\nText: " + body) if title else ("Text: " + body)

def user_prompt(r):
    return "Review:\\n" + review_text(r) + "\\n\\nSentiment:"

def zero_shot_messages(r):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(r)}]

def few_shot_messages(examples, r):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": user_prompt(ex)})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": user_prompt(r)})
    return msgs

# --- few-shot selection: 2 per class, lowest filtered_index in the TRAINING set ---
train_sorted = sorted(train_records, key=lambda r: r["filtered_index"])
few_shot_examples = []
for c in CLASSES:
    few_shot_examples.extend([r for r in train_sorted if r["label"] == c][:2])

train_id_set = set(train_ids)
check(len(few_shot_examples) == 6, "6 few-shot examples (2 per class)")
check(all(r["filtered_index"] in train_id_set for r in few_shot_examples), "every few-shot example is one of the 900 training reviews")
print("few-shot picks:", [(r["filtered_index"], r["label"]) for r in few_shot_examples])

LABELS = set(CLASSES)
def parse_label(raw):
    s = (raw or "").strip().upper().strip("\\"'\\".',;!?: ")
    return s if s in LABELS else None
"""))
CELLS.append(code("""# Hosted-model client (class endpoint, OpenAI-compatible). Thinking mode is disabled.
API_BASE = "http://dobolyi.com:9001/v1"
API_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

try:
    from google.colab import userdata
    API_KEY = userdata.get("CLASS_API_KEY")
    KEY_SOURCE = "Colab secret 'CLASS_API_KEY'"
except Exception:
    import getpass
    API_KEY = getpass.getpass("CLASS_API_KEY (no Colab secret found): ")
    KEY_SOURCE = "manual entry"

assert API_KEY, "API key missing — add the Colab secret CLASS_API_KEY (key icon in the left panel) and re-run"
print("API key loaded from:", KEY_SOURCE, "(key itself is never printed)")

def call_model(messages, max_tokens=16, timeout=90, retries=3):
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(
                API_BASE + "/chat/completions",
                headers={"Authorization": "Bearer " + API_KEY},
                json={
                    "model": API_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = (data["choices"][0]["message"].get("content") or "").strip()
            return raw, round(time.time() - t0, 2), (data.get("usage") or {})
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("API call failed after %d attempts: %r" % (retries, last))
"""))
# ------------------------------------------------------------------ 4. Zero-shot
CELLS.append(md("""## 4. Approach 1 — zero-shot prompting (hosted model)"""))
CELLS.append(code("""def score_test(messages_fn, name, max_tokens=16):
    results = []
    for i, r in enumerate(test_records):
        raw, dt, usage = call_model(messages_fn(r), max_tokens=max_tokens)
        pred = parse_label(raw)
        results.append({
            "filtered_index": r["filtered_index"],
            "original_row": r["original_row"],
            "reference_label": r["label"],
            "raw_output": raw,
            "prediction": pred,
            "valid": pred is not None,
            "latency_s": dt,
        })
        if (i + 1) % 50 == 0:
            print(name, ":", i + 1, "/", len(test_records), "done")
    return results

def assert_aligned(results, name):
    n = len(results)
    ids = [r["filtered_index"] for r in results]
    check(n == 300, name + ": exactly one prediction row per test review (" + str(n) + " rows for 300 reviews)")
    check(len(set(ids)) == 300, name + ": prediction rows have unique test IDs")
    check(set(ids) == set(test_ids), name + ": predictions cover exactly the prescribed 300 test IDs")

zero_shot_results = score_test(zero_shot_messages, "zero-shot")
assert_aligned(zero_shot_results, "zero-shot")
"""))
CELLS.append(code("""def metrics_table(results, name):
    y_true = [r["reference_label"] for r in results]
    y_pred = [r["prediction"] for r in results]
    n = len(results)
    valid = sum(1 for r in results if r["valid"])
    conf = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for yt, yp in zip(y_true, y_pred):
        if yp is not None:
            conf[yt][yp] += 1
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yp == yt) / n
    print("---", name, "---")
    print("accuracy       : %.4f" % acc)
    f1s = []
    print("class     tp  precision  recall      f1")
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in CLASSES if t != c)
        fn = sum(conf[c][p] for p in CLASSES if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        print("%-9s%4d%11.4f%9.4f%8.4f" % (c, tp, prec, rec, f1))
    macro_f1 = sum(f1s) / len(f1s)
    print("macro-F1      : %.4f" % macro_f1)
    print("valid/invalid : %d / %d" % (valid, n - valid))
    return {"name": name, "accuracy": acc, "macro_f1": macro_f1, "valid": valid, "invalid": n - valid}

zero_shot_metrics = metrics_table(zero_shot_results, "zero-shot")
"""))
# ------------------------------------------------------------------ 5. Few-shot
CELLS.append(md("""## 5. Approach 2 — few-shot prompting (6 examples: 2 per class, from the training set)"""))
CELLS.append(code("""few_shot_results = score_test(lambda r: few_shot_messages(few_shot_examples, r), "few-shot")
assert_aligned(few_shot_results, "few-shot")
few_shot_metrics = metrics_table(few_shot_results, "few-shot")
"""))
# ------------------------------------------------------------------ 6. Fine-tune
CELLS.append(md("""## 6. Approach 3 — QLoRA fine-tune with Unsloth

* **Base model:** `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` (dynamic 4-bit quantization).
* **Data:** the prescribed 900 training reviews (300 per class); validation set stays untouched.
* **Hyperparameters:** 2 epochs, lr 2e-4, LoRA r=16 / alpha=16, dropout 0, seq len 512,
  batch size 4 with gradient accumulation 2 (effective 8), cosine schedule, warmup 5%, seed 43, adamw_8bit.
* **GPU:** this notebook runs on the T4 runtime (16 GB) — T4 compatibility is tested directly on-device;
  the actual GPU name/VRAM is reported below. Training + inference both fit comfortably (4-bit 3B model ≈ 2–3 GB weights + LoRA optimizer states)."""))
CELLS.append(code("""print("Actual GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("VRAM GB:", round(props.total_memory / 1e9, 2))
    if "T4" in torch.cuda.get_device_name(0):
        print("T4 compatibility: tested directly on this T4 device.")
    else:
        print("NOTE: runtime GPU is %s — not a T4; the config below is sized for 16 GB T4." % torch.cuda.get_device_name(0))

from unsloth import FastLanguageModel
base_model_id = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
print("base model loaded:", base_model_id)
"""))
CELLS.append(code("""model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=43,
    max_seq_length=512,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("trainable params: %s (%0.3f%% of %s)" % (format(trainable, ","), 100.0 * trainable / total, format(total, ",")))
"""))
CELLS.append(code("""def make_train_text(r):
    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(r)},
        {"role": "assistant", "content": r["label"]},
    ]
    return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

train_texts = [make_train_text(r) for r in train_records]
check(len(train_texts) == 900, "actual prepared training records total exactly 900")
for c in CLASSES:
    check(sum(1 for r, t in zip(train_records, train_texts) if r["label"] == c) == 300,
          "actual prepared training records: exactly 300 " + c)
print("example training text (first 400 chars):")
print(train_texts[0][:400])

from datasets import Dataset
train_ds = Dataset.from_list([{"text": t} for t in train_texts])
print("train dataset rows:", len(train_ds))
"""))
CELLS.append(code("""from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="no",
    output_dir="ft_out",
    seed=43,
    fp16=True,
    optim="adamw_8bit",
    report_to="none",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)
t0 = time.time()
trainer.train()
print("training wall time (s):", round(time.time() - t0, 1))
"""))
CELLS.append(code("""# Save the LoRA adapter, then bundle exactly the two required files into the submission zip.
model.save_pretrained("adapter_ft")
tokenizer.save_pretrained("adapter_ft")

if os.path.exists("adapter_ft.zip"):
    os.remove("adapter_ft.zip")
with zipfile.ZipFile("adapter_ft.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("adapter_ft/adapter_model.safetensors", "adapter_model.safetensors")
    z.write("adapter_ft/adapter_config.json", "adapter_config.json")

import shutil, hashlib
names = sorted(zipfile.ZipFile("adapter_ft.zip").namelist())
check(names == ["adapter_config.json", "adapter_model.safetensors"],
      "adapter zip contains exactly adapter_model.safetensors + adapter_config.json")
print("adapter zip size MB:", round(os.path.getsize("adapter_ft.zip") / 1e6, 2))
cfg = json.load(open("adapter_ft/adapter_config.json"))
print("adapter r / lora_alpha:", cfg.get("r"), cfg.get("lora_alpha"), "| base:", cfg.get("base_model_name_or_path"))
"""))
# ------------------------------------------------------------------ 7. FT inference
CELLS.append(md("""## 7. Fine-tuned test predictions from the *submitted adapter*

The adapter is reloaded from `adapter_ft.zip` (the exact submission file) onto a fresh 4-bit base model,
and all 300 test predictions are regenerated from that reloaded adapter — so the delivered prediction file
and `prompt_examples.jsonl` provably come from the submitted adapter and match the results reported here.
The six few-shot/zero-shot results above are untouched by this section (no test-time adjustment)."""))
CELLS.append(code("""shutil.rmtree("adapter_from_zip", ignore_errors=True)
with zipfile.ZipFile("adapter_ft.zip") as z:
    z.extractall("adapter_from_zip")

print("Reloading fresh 4-bit base model for the saved-adapter inference example ...")
base2, tok2 = FastLanguageModel.from_pretrained(
    model_name=base_model_id, max_seq_length=512, dtype=None, load_in_4bit=True
)
from peft import PeftModel
ft_model = PeftModel.from_pretrained(base2, "adapter_from_zip")
FastLanguageModel.for_inference(ft_model)
print("Adapter loaded from adapter_ft.zip: OK")

def infer_label(conv):
    prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = ft_model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=0.0)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

ft_results = []
for i, r in enumerate(test_records):
    conv = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(r)}]
    raw = infer_label(conv)
    pred = parse_label(raw)
    ft_results.append({
        "filtered_index": r["filtered_index"],
        "original_row": r["original_row"],
        "reference_label": r["label"],
        "raw_output": raw,
        "prediction": pred,
        "valid": pred is not None,
    })
    if (i + 1) % 50 == 0:
        print("fine-tuned:", i + 1, "/", len(test_records), "done")
assert_aligned(ft_results, "fine-tuned")
"""))
CELLS.append(code("""ft_metrics = metrics_table(ft_results, "fine-tuned")

# ---- fine_tuned_test_predictions.csv (300 rows) ----
import csv
with open("fine_tuned_test_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filtered_index", "original_row", "reference_label", "prediction", "valid", "raw_output"])
    for r in ft_results:
        w.writerow([r["filtered_index"], r["original_row"], r["reference_label"],
                    r["prediction"], r["valid"], r["raw_output"]])
lines = open("fine_tuned_test_predictions.csv").read().strip().splitlines()
check(len(lines) == 301, "fine_tuned_test_predictions.csv = header + 300 prediction rows")

# ---- prompt_examples.jsonl: 3 ACTUAL fine-tuned test prompts, one per reference class ----
examples = []
for c in CLASSES:
    row = next(r for r in ft_results if r["reference_label"] == c)  # first test row of that class
    rec = by_id[row["filtered_index"]]
    conv = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(rec)}]
    prompt_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    examples.append({
        "filtered_index": row["filtered_index"],
        "original_row": row["original_row"],
        "reference_label": c,
        "prediction": row["prediction"],
        "valid": row["valid"],
        "prompt": prompt_text,
    })
with open("prompt_examples.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\\n")
check(len(examples) == 3 and {e["reference_label"] for e in examples} == set(CLASSES),
      "prompt_examples.jsonl has exactly one prompt per reference class")
for ex in examples:
    print(ex["reference_label"], "filtered_index", ex["filtered_index"], "-> prediction", ex["prediction"])
"""))
# ------------------------------------------------------------------ 8. Report
CELLS.append(md("""## 8. Comparison and report

Every number below comes from executed cells in this notebook (execution is sequential and assertions
stop the run on any failed check — unverified claims are not reported as results).

| approach | test reviews | accuracy | macro-F1 | invalid outputs |
|---|---|---|---|---|
| zero-shot (hosted Qwen3.6-35B) | 300 | printed below | | |
| few-shot (hosted, 6 examples) | 300 | | | |
| fine-tuned (Qwen2.5-3B QLoRA) | 300 | | | |

(Table values are printed by the code cell that follows.)"""))
CELLS.append(code("""print("=" * 62)
print("COMPARISON — same 300 test reviews, aligned by filtered_index")
print("=" * 62)
for m in [zero_shot_metrics, few_shot_metrics, ft_metrics]:
    print("%-11s accuracy %8.4f | macro-F1 %8.4f | invalid %d" % (m["name"], m["accuracy"], m["macro_f1"], m["invalid"]))

print()
for name, res in [("zero-shot", zero_shot_results), ("few-shot", few_shot_results), ("fine-tuned", ft_results)]:
    bad = [r for r in res if not r["valid"]]
    if bad:
        print(name, ": %d invalid response(s); up to 3 shown (raw output kept, not replaced):" % len(bad))
        for r in bad[:3]:
            print("   idx", r["filtered_index"], "| ref", r["reference_label"], "| raw:", repr(r["raw_output"][:80]))
    else:
        print(name, ": 0 invalid responses")

print()
print("Environment / GPU report")
print("  actual GPU        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
print("  transformers      :", transformers.__version__)
print("  torch             :", torch.__version__)
print("  T4 compatibility  : tested directly on the T4 runtime used for this notebook (16 GB fits with headroom)")
print("  credentials       : API key read from Colab secret CLASS_API_KEY; never printed or written to disk")
"""))
CELLS.append(code("""# Final artifacts + integrity hashes + browser downloads
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

for p in ["adapter_ft.zip", "fine_tuned_test_predictions.csv", "prompt_examples.jsonl"]:
    print(p, "| size", os.path.getsize(p), "| sha256", sha256_of(p))

from google.colab import files
files.download("adapter_ft.zip")
files.download("fine_tuned_test_predictions.csv")
files.download("prompt_examples.jsonl")
print("Downloads triggered — verify the sha256 values above after download.")
"""))


def build():
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src.splitlines(keepends=True),
            })
        else:
            compile(src, "<cell>", "exec")  # syntax check before writing
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            })
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", OUT, "| cells:", len(cells),
          "| code cells:", sum(1 for c in cells if c["cell_type"] == "code"))


if __name__ == "__main__":
    build()
