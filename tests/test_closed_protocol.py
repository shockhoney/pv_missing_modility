import unittest
from unittest.mock import patch

from utils.datasets_txt import build_closed_protocols


class ClosedProtocolTest(unittest.TestCase):
    def test_closed_protocol_splits_each_identity_by_sample(self):
        pairs = {
            "0001_1": [(f"palm_1_{i}", f"vein_1_{i}") for i in range(6)],
            "0001_2": [(f"palm_2_{i}", f"vein_2_{i}") for i in range(6)],
        }
        written = {}

        with patch("utils.datasets_txt._collect_pairs", return_value=pairs), patch(
            "utils.datasets_txt._write", side_effect=lambda path, lines: written.setdefault(path, lines)
        ):
            summary = build_closed_protocols("unused", "protocols", seed=1)

        self.assertEqual(summary["num_classes_total"], 2)
        self.assertEqual(summary["num_train_pairs"], 8)
        self.assertEqual(summary["num_val_pairs"], 2)
        self.assertEqual(summary["num_test_protocol_pairs"], 8)
        self.assertTrue(any(path.endswith("closed_train_full.txt") for path in written))
        self.assertTrue(any(path.endswith("closed_val_full.txt") for path in written))
        self.assertTrue(any(path.endswith("closed_test_protocol.txt") for path in written))


if __name__ == "__main__":
    unittest.main()
