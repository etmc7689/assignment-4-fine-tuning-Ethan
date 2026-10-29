# -*- coding: utf-8 -*-
"""Local preflight: exercise the exact few-shot message shape + parser against the real
class endpoint (9001) before spending GPU time. Uses 2 train reviews + 1 test review.

Run: .nb-venv/bin/python endpoint_preflight.py
"""
import json, os, random, requests, sys

API_BASE = "http://dobolyi.com:9001/v1"
API_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
API_KEY = "6418"

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
SYSTEM_PROMPT = "Classify the sentiment of the Amazon review below. Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."


def review_text(r):
    title = (r.get("title") or "").strip()
    body = (r.get("text") or "").strip()
    return ("Title: " + title + "\nText: " + body) if title else ("Text: " + body)


def user_prompt(r):
    return "Review:\n" + review_text(r) + "\n\nSentiment:"


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


def parse_label(raw):
    s = (raw or "").strip().upper().strip("\"'.,;!?: ")
    return s if s in set(CLASSES) else None


def call(messages, max_tokens=16):
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
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    return raw, data.get("usage", {})


def main():
    # Real training reviews (labels given) + one test review, drawn from the verified split files
    train = []
    with open("split_data/train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            train.append({"filtered_index": d["filtered_index"], "title": d["title"],
                          "text": d["text"], "label": d["reference_label"]})
            if len(train) == 200:
                break
    test = []
    with open("split_data/test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            test.append({"filtered_index": d["filtered_index"], "title": d["title"],
                         "text": d["text"], "label": d["reference_label"]})
            if len(test) == 5:
                break

    examples = []
    for c in CLASSES:
        examples.extend([r for r in sorted(train, key=lambda r: r["filtered_index"]) if r["label"] == c][:2])

    print("few-shot pick indices:", [(e["filtered_index"], e["label"]) for e in examples])

    for label, fn in [("zero-shot", lambda r: zero_shot_messages(r)),
                      ("few-shot", lambda r: few_shot_messages(examples, r))]:
        for tr in test:
            raw, usage = call(fn(tr))
            pred = parse_label(raw)
            print(f"{label:>10} | test idx {tr['filtered_index']:>6} | ref {tr['label']:<8} | "
                  f"raw={raw!r:<14} parsed={pred!r:<10} usage={usage.get('completion_tokens')}tok")
            assert pred is not None, "output did not parse to a bare label — check prompt/parse"

    # Also verify a full-length few-shot prompt token cost stays sane
    raw, usage = call(few_shot_messages(examples, test[0]), max_tokens=8)
    print(f"few-shot prompt tokens: {usage.get('prompt_tokens')} | reply: {raw!r}")
    print("ALL ENDPOINT PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
