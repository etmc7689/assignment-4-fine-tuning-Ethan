#!/usr/bin/env python3
"""Generate zero-shot + few-shot predictions from the class hosted model (9001).

Mirrors the exact notebook logic (fixed splits, prompt templates, output
contract, few-shot selection) so the resulting prediction files are what the
notebook's evaluation section consumes. Writes:
    results_zero_shot.jsonl, results_few_shot.jsonl

Usage:  python3 gen_hosted_preds.py   (API key via CLASS_API_KEY env, else '6418')
"""
import gzip, json, os, random, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "http://dobolyi.com:9001/v1"
API_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
API_KEY = os.environ.get("CLASS_API_KEY", "6418")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "Gift_Cards.jsonl.gz")
WORKERS = 6
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
LABELS = set(CLASSES)


def classify(rating):
    if rating is None:
        return None
    r = float(rating)
    if r >= 4.0: return "POSITIVE"
    if r == 3.0: return "NEUTRAL"
    if r <= 2.0: return "NEGATIVE"
    return None


def load_records():
    records = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except ValueError:
                continue
            t = (d.get("text") or "")
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            if len(t) < 3:
                continue
            records.append({
                "filtered_index": len(records),
                "rating": d.get("rating"),
                "title": (d.get("title") or "").strip(),
                "text": t,
                "label": classify(d.get("rating")),
            })
    return records


def build_splits(records):
    def pool_ids(lo, hi, cls):
        return [r["filtered_index"] for r in records if lo <= r["filtered_index"] <= hi and r["label"] == cls]
    rng = random.Random(43)
    train_ids = []
    for c in CLASSES:
        train_ids.extend(rng.sample(pool_ids(1000, 19999, c), 300))
    rng.shuffle(train_ids)
    test_ids = []
    for c in CLASSES:
        test_ids.extend(pool_ids(30000, 39999, c)[:100])
    by_id = {r["filtered_index"]: r for r in records}
    train_records = [by_id[i] for i in train_ids]
    test_records = [by_id[i] for i in test_ids]
    train_sorted = sorted(train_records, key=lambda r: r["filtered_index"])
    few_shot = []
    for c in CLASSES:
        few_shot.extend([r for r in train_sorted if r["label"] == c][:2])
    return test_records, few_shot


SYSTEM_PROMPT = "Classify the sentiment of the Amazon review below. Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."


def review_text(r):
    title = (r.get("title") or "").strip()
    body = (r.get("text") or "").strip()
    return ("Title: " + title + "\nText: " + body) if title else ("Text: " + body)


def user_prompt(r):
    return "Review:\n" + review_text(r) + "\n\nSentiment:"


def zero_shot_messages(r):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(r)}]


def few_shot_messages(examples, r):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": user_prompt(ex)})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": user_prompt(r)})
    return msgs


def parse_label(raw):
    s = (raw or "").strip().upper().strip("\"'.,;!?: ")
    return s if s in LABELS else None


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
            raw = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            return raw, round(time.time() - t0, 2)
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("API call failed after %d attempts: %r" % (retries, last))


def score_test(test_records, messages_fn, name, out_path):
    results = []

    def one(r):
        raw, dt = call_model(messages_fn(r))
        pred = parse_label(raw)
        return {"filtered_index": r["filtered_index"], "original_row": r.get("original_row"),
                "reference_label": r["label"], "raw_output": raw,
                "prediction": pred, "valid": pred is not None, "latency_s": dt}

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, r): r for r in test_records}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 50 == 0:
                print(name, done, "/", len(test_records), flush=True)
    results.sort(key=lambda x: x["filtered_index"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    valid = sum(1 for r in results if r["valid"])
    print("%s: %d/300 valid, %d invalid -> %s" % (name, valid, len(results) - valid, out_path))


def main():
    print("loading data from", DATA_PATH)
    records = load_records()
    test_records, few_shot = build_splits(records)
    assert len(test_records) == 300, len(test_records)
    assert len(few_shot) == 6, len(few_shot)
    print("test set:", len(test_records), "| few-shot picks:", [(r["filtered_index"], r["label"]) for r in few_shot])
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    score_test(test_records, zero_shot_messages, "zero-shot",
               os.path.join(here, "results_zero_shot.jsonl"))
    print("zero-shot done in %.1fs" % (time.time() - t0))
    t1 = time.time()
    score_test(test_records, lambda r: few_shot_messages(few_shot, r), "few-shot",
               os.path.join(here, "results_few_shot.jsonl"))
    print("few-shot done in %.1fs" % (time.time() - t1))
    print("ALL DONE")


if __name__ == "__main__":
    main()
