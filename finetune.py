"""Fine-tunes prajjwal1/bert-tiny (MIT licence, 4.4M parameters) with a classification head.

python finetune.py            # trains, picks the best epoch on validation, scores the test set once
Runs on a laptop CPU in about a minute.
"""

import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, accuracy, macro_f1, read_rows

MODEL = "prajjwal1/bert-tiny"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, tok, texts, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
            preds += model(**enc).logits.argmax(-1).tolist()
    return [LABELS[p] for p in preds]


def main(epochs=12, lr=5e-4, batch_size=16):
    set_seed(SEED)
    train, val, test = (read_rows(f"data/{n}.json") for n in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    y = torch.tensor([LABELS.index(r["label"]) for r in train])
    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train))
        for i in range(0, len(train), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([train[j]["text"] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
            loss = model(**enc, labels=y[idx]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_preds = predict(model, tok, [r["text"] for r in val])
        f1 = macro_f1(val_preds, [r["label"] for r in val])
        history.append({"epoch": epoch, "loss": round(loss.item(), 4), "val_macro_f1": round(f1, 4)})
        print(history[-1])
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    seconds = time.time() - t0
    # the test set is used exactly once, after the model is chosen
    preds = predict(model, tok, [r["text"] for r in test])
    golds = [r["label"] for r in test]
    result = {"model": MODEL, "train_seconds_cpu": round(seconds, 1), "best_val_macro_f1": round(best_f1, 4),
              "test_accuracy": round(accuracy(preds, golds), 4), "test_macro_f1": round(macro_f1(preds, golds), 4),
              "test_size": len(test), "history": history,
              "errors": [{"text": r["text"], "gold": r["label"], "pred": p} for r, p in zip(test, preds) if p != r["label"]]}
    json.dump(result, open("results/finetune.json", "w"), indent=1)
    print({k: v for k, v in result.items() if k not in ("history", "errors")})


if __name__ == "__main__":
    main()
