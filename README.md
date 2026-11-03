# Assignment 4 — Fine-Tuning (MBAX 6418)

Amazon Reviews '23, **Gift Cards** category — 3-class sentiment classification
(`POSITIVE` / `NEUTRAL` / `NEGATIVE`) on the same 300 test reviews across three approaches:
zero-shot prompting, few-shot prompting (6 examples), and a QLoRA fine-tune.

## Submission files

| File | Contents |
|---|---|
| `notebook.ipynb` | Fully executed notebook (21/21 code cells with outputs): data + prescribed splits, hosted zero-shot & few-shot, QLoRA fine-tune (Unsloth), saved-adapter reload + 300 fine-tuned predictions, required metrics, 3 side-by-side normalized confusion matrices + enlarged best, exact pairwise McNemar (statsmodels), integrity hashes, non-technical report |
| `adapter_ft.zip` | LoRA adapter — exactly `adapter_model.safetensors` + `adapter_config.json`; base identified as `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`, r=16 / lora_alpha=16. **110.9 MB exceeds GitHub's 100 MB file limit, so it is attached to the [latest Release](https://github.com/etmc7689/assignment-4-fine-tuning-Ethan/releases) of this repo** (sha256 below for verification) |
| `fine_tuned_test_predictions.csv` | All 300 test reviews: `original_row, filtered_index, title, text, reference_label, prediction, raw_response` |
| `prompt_examples.jsonl` | 3 actual inference prompts from the reloaded-adapter run (one per reference class), with `original_row, filtered_index, reference_label, prediction, prompt` |

## Results (same 300 test reviews)

| approach | accuracy | balanced acc | macro-F1 | invalid |
|---|---|---|---|---|
| zero-shot (hosted Qwen3.6-35B-A3B, ~3B active) | 0.7400 | 0.7400 | 0.7037 | 0 |
| few-shot (hosted, 6 examples) | 0.7500 | 0.7500 | 0.7251 | 0 |
| fine-tuned (Qwen2.5-3B QLoRA on Tesla T4) | **0.8033** | **0.8033** | **0.8026** | 0 |

Exact pairwise McNemar (statsmodels): zero-shot vs few-shot p = 0.62906 (n.s.),
zero-shot vs fine-tuned p = 0.02041, few-shot vs fine-tuned p = 0.04794 (both significant).

Training: 2 epochs, lr 2e-4, LoRA r=16/α=16, batch 4 with grad-accum 2, cosine schedule,
warmup 11 steps, seed 43, adamw_8bit — 226 steps in **333.7 s** on a Tesla T4 (16 GB).

## Integrity hashes

- `adapter_ft.zip` — sha256 `4b7917309bd39c86a025b111369c058b5d7d5d2676a5503849baf3c659936ad5` (110,862,991 B)
- `fine_tuned_test_predictions.csv` — sha256 `e8a817748917270df58d5a5f06885f2f497f446a57d1c464a26691c69c32c511` (60,181 B)
- `prompt_examples.jsonl` — sha256 `bf5d5cd8eadd37dce79bb7c10fa9d9b06dd8d8ae4c362ab8565d6faf3a3ee26d` (1,862 B)

Note: the prediction CSV delivered here has the assignment's required schema
(`title`, `text`, and `raw_response` columns). Its 300 predictions are identical to the
notebook's run — metrics recomputed from this exact file reproduce the reported scores
(accuracy 0.8033, macro-F1 0.8026, per-class tp 89/69/83).

## Reproduce without retraining

Section 7 of the notebook extracts `adapter_ft.zip` into a fresh folder, loads it onto a
fresh 4-bit base model (`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`) with the matching
tokenizer, and regenerates all 300 fine-tuned predictions with greedy decoding
(`max_new_tokens=8`, `do_sample=False`, `temperature=0.0`) using the same prompt + response
processing as the test run — no training and no hosted-model calls.

## Data

McAuley Lab — Amazon Reviews '23, `review_categories/Gift_Cards.jsonl.gz`
(12.29 MB, 152,410 reviews; raw copy in `raw/`). Splits per assignment:
train = filtered_index 1,000–19,999 (300/class via `random.Random(43)`), test = 30,000–39,999
(first 100/class); validation 20,000–29,999 reserved.
