"""Builds the support-message dataset. Every message is written by me from the phrases below,
and I release the dataset under CC0 (public domain), so there are no licence questions. Steps: generate, clean (trim, collapse spaces, drop
exact and case-insensitive duplicates and very short messages), shuffle with a fixed seed,
then split by phrase (8/2/2 of the 12 phrases per label) into train/validation/test. The test file is written once and never read
by the training code except for the final score."""

import itertools
import json
import random
import re
from pathlib import Path

LABELS = ["billing", "delivery", "account", "technical"]

PHRASES = {
    "billing": [
        "I was charged twice for my order", "my card was debited but the payment failed",
        "please send me the GST invoice", "why is there an extra fee on my bill",
        "I want a refund for the cancelled order", "the UPI payment went through but order shows unpaid",
        "can I change the payment method", "the coupon discount was not applied",
        "I got charged after cancelling my subscription", "how do I get my money back",
        "the amount on the bill is wrong", "EMI option is not showing at checkout",
    ],
    "delivery": [
        "my parcel has not arrived yet", "the delivery is late by a week",
        "tracking has not updated for days", "the package was delivered to the wrong address",
        "I received a damaged box", "can I change the delivery date",
        "the courier never called me", "order shows delivered but I did not get it",
        "one item is missing from the package", "when will my order be shipped",
        "can you deliver to my office instead", "the delivery person asked for extra money",
    ],
    "account": [
        "I cannot log in to my account", "I forgot my password",
        "how do I change my registered phone number", "please delete my account",
        "I did not get the OTP", "my account was locked after wrong attempts",
        "how do I update my email address", "someone else logged into my account",
        "I want to merge my two accounts", "the verification link has expired",
        "how do I change my name on the profile", "I want to download my data",
    ],
    "technical": [
        "the app crashes when I open it", "the website is very slow",
        "the checkout button does nothing", "images are not loading in the app",
        "I get an error 500 on the order page", "the app does not install on my phone",
        "search is not showing any results", "the page keeps refreshing",
        "notifications are not working", "the app logs me out again and again",
        "video in the product page does not play", "dark mode makes the text invisible",
    ],
}
OPENERS = ["", "hi, ", "hello team, ", "sir, ", "please help, ", "urgent: ", "good morning, "]
CLOSERS = ["", ".", " please help.", " what should I do?", " this is the third time.",
           " kindly check.", " thanks."]


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text[0].upper() + text[1:] if text else text


def build(seed=42):
    rng = random.Random(seed)
    rows, seen = [], set()
    for label, phrases in PHRASES.items():
        combos = list(itertools.product(phrases, OPENERS, CLOSERS))
        rng.shuffle(combos)
        for phrase, opener, closer in combos[:90]:
            text = clean(opener + phrase + closer)
            key = text.lower().rstrip(".!? ")
            if len(text) < 10 or key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label})
    rng.shuffle(rows)
    return rows


def split(rows):
    """Split by the underlying phrase, not by row: all variants of one phrase go to the same
    split, so the test set never contains a training sentence with a new greeting added."""
    def phrase_of(row):
        body = row["text"].lower()
        for phrases in PHRASES.values():
            for p in phrases:
                if p.lower() in body:
                    return p
        return body
    # per label: 2 phrases for test, 2 for validation, the other 8 for training
    rng = random.Random(7)
    test_g, val_g = set(), set()
    for phrases in PHRASES.values():
        shuffled = phrases[:]
        rng.shuffle(shuffled)
        test_g.update(shuffled[:2])
        val_g.update(shuffled[2:4])
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        g = phrase_of(r)
        out["test" if g in test_g else "val" if g in val_g else "train"].append(r)
    return out


def main():
    parts = split(build())
    for name, rows in parts.items():
        with open(Path(__file__).parent / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        counts = {label: sum(r["label"] == label for r in rows) for label in LABELS}
        print(f"{name}: {len(rows)} rows {counts}")


if __name__ == "__main__":
    main()
