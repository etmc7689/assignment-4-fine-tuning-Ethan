# -*- coding: utf-8 -*-
"""Local preflight: verify the Qwen2.5 chat template produces the expected training and
inference strings (label-only target; generation prompt). Tokenizer-only — no torch needed.

Run: .nb-venv/bin/python tokenizer_preflight.py
"""
import json
from transformers import AutoTokenizer

SYSTEM_PROMPT = "Classify the sentiment of the Amazon review below. Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print("tokenizer:", tok.__class__.__name__)
    assert tok.chat_template, "chat template missing!"

    sample = {"title": "Great gift", "text": "Having Amazon money is always good."}
    title, body = (sample.get("title") or "").strip(), (sample.get("text") or "").strip()
    user = ("Title: " + title + "\nText: " + body) if title else ("Text: " + body)
    user_content = "Review:\n" + user + "\n\nSentiment:"

    train_conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "POSITIVE"},
    ]
    train_text = tok.apply_chat_template(train_conv, tokenize=False, add_generation_prompt=False)
    print("--- TRAINING TEXT (repr) ---")
    print(repr(train_text))
    assert train_text.rstrip().endswith("POSITIVE<|im_end|>"), "training text must end with the label + im_end"

    infer_conv = train_conv[:-1]
    infer_text = tok.apply_chat_template(infer_conv, tokenize=False, add_generation_prompt=True)
    print("--- INFERENCE PROMPT (repr) ---")
    print(repr(infer_text))
    assert infer_text.rstrip().endswith("<|im_start|>assistant"), "inference prompt must end with assistant header"

    ids = tok(infer_text)["input_ids"]  # torch-free (only used for the count here)
    print("inference prompt token count:", len(ids))
    print("TOKENIZER PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
