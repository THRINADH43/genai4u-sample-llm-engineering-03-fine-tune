import json

LABELS = ["billing", "delivery", "account", "technical"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def macro_f1(preds, golds, labels=LABELS):
    scores = []
    for label in labels:
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)
