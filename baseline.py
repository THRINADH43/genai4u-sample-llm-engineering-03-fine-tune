"""Prompted baseline: a hosted model (Groq, openai/gpt-oss-20b), given 8 labelled examples from the
training set (few-shot), labels the same test set.

GROQ_API_KEY=... python baseline.py          # real run
python baseline.py --fake                     # keyword fake model, no key, for trying it out
GROQ_API_KEY=... python baseline.py --zero-shot   # the same model with no examples
The key is read from the environment and never written anywhere.
"""

import json
import os
import sys
import time
import urllib.request

from common import LABELS, accuracy, macro_f1, read_rows

MODEL = "openai/gpt-oss-20b"
# Groq list prices for this model, USD per 1M tokens (input, output), checked when I ran it
PRICE = (0.075, 0.30)
PROMPT = ("Classify the customer support message into exactly one category: billing, delivery, "
          "account or technical. Reply with the category word only.\n\n"
          "Examples:\n{examples}\n\nMessage: {text}")


def few_shot_examples(n_per_label=2, seed=3):
    """Two training messages per label. Taken from the training set, never from the test set."""
    import random
    rng = random.Random(seed)
    rows = read_rows("data/train.json")
    picked = []
    for label in LABELS:
        picked += rng.sample([r for r in rows if r["label"] == label], n_per_label)
    return "\n".join(f"Message: {r['text']}\nCategory: {r['label']}" for r in picked)


def ask_groq(text, key, examples):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 300, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": PROMPT.format(examples=examples, text=text)}]}
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "finetune-baseline/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 4:
                raise
            time.sleep(2 ** attempt)  # free tier rate limit: wait and try again
    answer = data["choices"][0]["message"]["content"].strip().lower()
    usage = data["usage"]
    return parse_label(answer), usage["prompt_tokens"], usage["completion_tokens"]


def parse_label(answer):
    for label in LABELS:
        if label in answer.lower():
            return label
    return "unknown"


def fake_model(text):
    words = {"billing": ["charge", "refund", "pay", "invoice", "bill"], "delivery": ["deliver", "parcel", "track"],
             "account": ["log", "password", "otp", "account"], "technical": ["app", "error", "page"]}
    for label, keys in words.items():
        if any(k in text.lower() for k in keys):
            return label, 0, 0
    return "technical", 0, 0


def main():
    fake = "--fake" in sys.argv
    zero_shot = "--zero-shot" in sys.argv  # no examples, to see whether the examples help
    key = os.environ.get("GROQ_API_KEY")
    if not fake and not key:
        sys.exit("Set GROQ_API_KEY, or run with --fake")
    test = read_rows("data/test.json")
    examples = "(none)" if zero_shot else few_shot_examples()
    preds, tin, tout = [], 0, 0
    t0 = time.time()
    for r in test:
        label, i, o = fake_model(r["text"]) if fake else ask_groq(r["text"], key, examples)
        preds.append(label)
        tin, tout = tin + i, tout + o
    golds = [r["label"] for r in test]
    cost = tin / 1e6 * PRICE[0] + tout / 1e6 * PRICE[1]
    result = {"model": "fake-keywords" if fake else f"groq:{MODEL}", "test_size": len(test),
              "test_accuracy": round(accuracy(preds, golds), 4), "test_macro_f1": round(macro_f1(preds, golds), 4),
              "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 5),
              "seconds": round(time.time() - t0, 1),
              "errors": [{"text": r["text"], "gold": r["label"], "pred": p} for r, p in zip(test, preds) if p != r["label"]]}
    name = "baseline_fake" if fake else "baseline_zero_shot" if zero_shot else "baseline"
    json.dump(result, open(f"results/{name}.json", "w"), indent=1)
    print({k: v for k, v in result.items() if k != "errors"})


if __name__ == "__main__":
    main()
