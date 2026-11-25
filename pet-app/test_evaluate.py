import unittest

import pandas as pd

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"image_id": ["1", "2", "3", "4"],
                                   "y_true": ["a", "a", "b", "b"],
                                   "y_pred": ["a", "b", "b", "b"]})

    def test_hand_calculated_scores_and_axes(self):
        summary, per_class, matrix, errors = evaluate(self.frame, ["a", "b"])
        self.assertEqual(summary["accuracy"], 0.75)
        self.assertAlmostEqual(summary["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        self.assertEqual(per_class.recall.tolist(), [0.5, 1.0])
        self.assertEqual(errors.image_id.tolist(), ["2"])

    def test_missing_class_policy(self):
        summary, _, _, _ = evaluate(self.frame, ["a", "b", "c"])
        self.assertEqual(summary["missing_true_classes"], ["c"])
        self.assertAlmostEqual(summary["macro_f1"], (2 / 3 + 0.8) / 3)

    def test_reject_unknown_labels_and_duplicate_images(self):
        with self.assertRaises(ValueError):
            evaluate(self.frame, ["a"])
        self.frame.loc[1, "image_id"] = "1"
        with self.assertRaises(ValueError):
            evaluate(self.frame, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
