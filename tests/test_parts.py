import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline import fake_model, parse_label  # noqa: E402
from common import accuracy, macro_f1, read_rows  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class DataTests(unittest.TestCase):
    def test_no_test_message_is_in_training(self):
        train = {r["text"].lower() for r in read_rows(os.path.join(DATA, "train.json"))}
        test = [r["text"].lower() for r in read_rows(os.path.join(DATA, "test.json"))]
        self.assertTrue(test)
        self.assertFalse(train.intersection(test))


class MetricTests(unittest.TestCase):
    def test_accuracy_and_f1(self):
        golds = ["billing", "billing", "account", "delivery"]
        preds = ["billing", "account", "account", "delivery"]
        self.assertEqual(accuracy(preds, golds), 0.75)
        self.assertAlmostEqual(macro_f1(preds, golds), (2 / 3 + 2 / 3 + 1 + 0) / 4)

    def test_parse_label(self):
        self.assertEqual(parse_label("Billing."), "billing")
        self.assertEqual(parse_label("I am not sure"), "unknown")

    def test_fake_model(self):
        self.assertEqual(fake_model("I want a refund")[0], "billing")


if __name__ == "__main__":
    unittest.main()
