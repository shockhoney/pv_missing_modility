import unittest
from unittest.mock import patch

from utils.datasets_txt import build_ssfd_protocols


class ClosedProtocolTest(unittest.TestCase):
    def test_ssfd_protocol_uses_fixed_train_test_samples(self):
        pairs = {
            "001": {
                "train": [(f"palm_1_train_{i}", f"vein_1_train_{i}") for i in range(3)],
                "test": [(f"palm_1_test_{i}", f"vein_1_test_{i}") for i in range(3)],
            },
            "002": {
                "train": [(f"palm_2_train_{i}", f"vein_2_train_{i}") for i in range(3)],
                "test": [(f"palm_2_test_{i}", f"vein_2_test_{i}") for i in range(3)],
            },
        }
        written = {}

        with patch("utils.datasets_txt._collect_ssfd_pairs", return_value=pairs), patch(
            "utils.datasets_txt._write", side_effect=lambda path, lines: written.setdefault(path, lines)
        ):
            summary = build_ssfd_protocols("unused", "protocols", dataset="casia")

        self.assertEqual(summary["num_classes_total"], 2)
        self.assertEqual(summary["num_train_pairs"], 6)
        self.assertEqual(summary["num_test_protocol_pairs"], 18)
        train_lines = next(lines for path, lines in written.items() if path.endswith("ssfd_train_full.txt"))
        test_lines = next(lines for path, lines in written.items() if path.endswith("ssfd_test_protocol.txt"))
        self.assertTrue(all("_train_" in line for line in train_lines))
        self.assertEqual(sum(" complete" in line for line in test_lines), 6)
        self.assertEqual(sum(" palmprint_missing" in line for line in test_lines), 6)
        self.assertEqual(sum(" palmvein_missing" in line for line in test_lines), 6)


if __name__ == "__main__":
    unittest.main()
