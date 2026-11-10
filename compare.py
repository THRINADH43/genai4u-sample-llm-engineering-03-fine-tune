"""Collects the saved results into results/summary.json, the numbers quoted in the README."""

import json
import os

FILES = {"fine_tuned": "finetune.json", "baseline_few_shot": "baseline.json",
         "baseline_zero_shot": "baseline_zero_shot.json", "keyword_rules": "baseline_fake.json"}


def main():
    summary = {}
    for name, file in FILES.items():
        path = os.path.join("results", file)
        if os.path.exists(path):
            r = json.load(open(path))
            summary[name] = {k: r[k] for k in ("model", "test_size", "test_accuracy", "test_macro_f1",
                                               "cost_usd", "train_seconds_cpu") if k in r}
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    for name, r in summary.items():
        print(f"{name:20} accuracy {r['test_accuracy']:.3f}  macro-F1 {r['test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
