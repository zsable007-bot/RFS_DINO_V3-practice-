import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluate_model import summarize  # noqa: E402（需 dinov3_11 环境：torch/timm）


def make_result(record, class_label="CW", true_index=0, predicted_index=0):
    return {
        "image": Path(f"{record}_event01_window01.png"),
        "record": record,
        "class_label": class_label,
        "true_index": true_index,
        "predicted_index": predicted_index,
        "probabilities": [0.0, 0.0, 0.0],
    }


class SummarizeTests(unittest.TestCase):
    def test_record_with_zero_exported_images_stays_in_denominator(self):
        """一张图都没切出的记录也要计入 expected_events，不能被静默忽略。"""
        records = {
            "CW_0002": {"class_label": "CW", "snr_db": "5.0"},
            "CW_0005": {"class_label": "CW", "snr_db": "5.0"},
        }
        detection_info = {
            "CW_0002": {"detected": 6, "class_label": "CW"},
            "CW_0005": {"detected": 6, "class_label": "CW"},
        }
        results = [make_result("CW_0005") for _ in range(6)]
        rows = summarize(results, records, detection_info, ["CW", "2FSK", "LFM"])
        row = next(
            r for r in rows if r["class_label"] == "CW" and r["snr_db"] == 5.0
        )
        self.assertEqual(row["records"], 2)
        self.assertEqual(row["expected_events"], 12)
        self.assertEqual(row["exported_images"], 6)
        self.assertEqual(row["detected_events"], 12)
        self.assertEqual(row["accuracy_end_to_end"], 0.5)

    def test_missing_detection_info_falls_back_per_record(self):
        """处理清单缺失时，回退值必须按各记录自己的导出数，不能重复用组总数。"""
        records = {
            "CW_0002": {"class_label": "CW", "snr_db": "5.0"},
            "CW_0005": {"class_label": "CW", "snr_db": "5.0"},
        }
        results = [make_result("CW_0002") for _ in range(6)] + [
            make_result("CW_0005") for _ in range(4)
        ]
        rows = summarize(results, records, {}, ["CW"])
        row = next(
            r for r in rows if r["class_label"] == "CW" and r["snr_db"] == 5.0
        )
        self.assertEqual(row["detected_events"], 10)  # 而非 10+10=20

    def test_only_observed_excludes_unseen_records(self):
        """调试模式（--limit）只统计真正有结果的记录。"""
        records = {
            "CW_0002": {"class_label": "CW", "snr_db": "5.0"},
            "CW_0005": {"class_label": "CW", "snr_db": "5.0"},
        }
        results = [make_result("CW_0005") for _ in range(6)]
        rows = summarize(results, records, {}, ["CW"], only_observed=True)
        row = next(
            r for r in rows if r["class_label"] == "CW" and r["snr_db"] == 5.0
        )
        self.assertEqual(row["records"], 1)
        self.assertEqual(row["expected_events"], 6)


if __name__ == "__main__":
    unittest.main()
