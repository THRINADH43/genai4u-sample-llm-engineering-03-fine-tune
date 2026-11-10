# Fine-tune a model and prove it helped

**Task:** sort customer support messages into four categories: `billing`, `delivery`,
`account`, `technical`.

**Short answer: the fine-tune was not worth it.** My fine-tuned tiny model got 69% accuracy
on the test set. A hosted model prompted with 8 examples got 95% on the same test set for
about a sixth of a US cent. Details and reasons below.

## Model and licence

- Fine-tuned model: [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny),
  4.4M parameters, **MIT licence**, with a classification head (`AutoModelForSequenceClassification`).
  I picked it because it trains on a laptop CPU in under a minute.
- Baseline: `openai/gpt-oss-20b` on Groq (Apache 2.0 open-weight model, used through Groq's API),
  prompted with 8 labelled examples (2 per category) picked from the **training** set.

## Dataset

I wrote the dataset myself (`data/make_dataset.py`) and release it under **CC0** (public
domain), so there is no licence problem: 12
support phrases per category, combined with greetings ("hi,", "sir,") and endings ("kindly
check."), giving 354 messages.

Cleaning: trim and collapse spaces, capitalise the first letter, drop messages under 10
characters, and drop duplicates ignoring case and final punctuation.

**Keeping the test set separate:** I split by *phrase*, not by message. For each category, 2
phrases go to test, 2 to validation and 8 to training, and every variant of a phrase stays in
the same split. Without this, the test set would contain training sentences with only a new
greeting, and the score would be meaningless. `tests/test_parts.py` checks that no test
message appears in training.

| split | rows | billing | delivery | account | technical |
|-------|------|---------|----------|---------|-----------|
| train | 232  | 57      | 59       | 56      | 60        |
| val   | 60   | 21      | 13       | 13      | 13        |
| test  | 62   | 10      | 17       | 19      | 16        |

## How to repeat it

```
pip install -r requirements.txt
python data/make_dataset.py          # same split every time (fixed seeds); writes data/train.json, val.json, test.json
python finetune.py                   # results/finetune.json
GROQ_API_KEY=... python baseline.py  # results/baseline.json (key from the environment, never committed)
python baseline.py --fake            # no key: a keyword "model", just to try the code
python -m unittest discover -s tests
```

`finetune.py` fixes the seeds, trains 12 epochs, keeps the epoch with the best validation
macro-F1, and only then scores the test set once.

## Results (same 62 test messages)

| model | accuracy | macro-F1 | cost | time |
|-------|----------|----------|------|------|
| bert-tiny fine-tuned (CPU) | 0.694 | 0.607 | $0 (42.7 s CPU) | about 1 hour of my work |
| gpt-oss-20b, 8-shot prompt (Groq) | **0.952** | **0.935** | $0.00161 (15,535 in / 1,494 out tokens) | 13 s, 15 minutes of my work |
| keyword rules (`--fake`) | 0.403 | 0.289 | $0 | - |

Raw numbers: `results/finetune.json`, `results/baseline.json`, `results/baseline_fake.json`
(each lists the wrong predictions).

## Was it worth it?

No, not for this task.

- **Accuracy:** the fine-tune is far behind. Its errors show why: it learned words, not
  meaning. "How do I change my registered *phone number*" became `billing`, and "*EMI*
  option is not showing at checkout" became `technical`. The test phrases were never seen in
  training, and 8 phrases per category is too little variety for a 4.4M-parameter model.
  Validation F1 stopped improving after epoch 5 while training loss kept falling
  (overfitting).
- **Cost:** the hosted model cost $0.00161 for 62 messages, about Rs 2.30 per 1,000
  messages. The fine-tune is free to run but cost about an hour to build, and it would need
  much more data to be useful.
- **When a fine-tune would win:** with thousands of real labelled messages, a bigger base
  model (for example distilbert), a need to run offline, or a very large volume where API
  cost adds up. None of these apply here.

**A surprise:** an earlier zero-shot run of the same model (no examples) scored 0.984, a
little higher than with 8 examples. With 62 test messages that difference is 2 messages, so I
would not read much into it, but adding examples did not help here and doubled the tokens.

**Limits of this test:** 62 test messages is small (one message is 1.6%), and my messages are
generated from templates, so they are cleaner than real customer messages.
