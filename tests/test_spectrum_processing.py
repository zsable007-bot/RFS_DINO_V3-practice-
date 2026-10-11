import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from spectrum_processing import process_record  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


class ParityTests(unittest.TestCase):
    def assert_record_matches_existing(self, class_label, stem):
        iq_path = DATA_DIR / class_label / f"{stem}.cf32"
        with tempfile.TemporaryDirectory() as tmp:
            summary = process_record(iq_path, output_root=tmp)

            new_manifest = json.loads(
                (Path(tmp) / "diagnostics" / f"{stem}_windows.json").read_text(
                    encoding="utf-8"
                )
            )
            old_manifest = json.loads(
                (OUTPUT_DIR / "diagnostics" / f"{stem}_windows.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(new_manifest["detector"], old_manifest["detector"])
            self.assertEqual(new_manifest["processing"], old_manifest["processing"])
            self.assertEqual(
                new_manifest["source_iq_sha256"], old_manifest["source_iq_sha256"]
            )
            self.assertEqual(
                len(new_manifest["windows"]), len(old_manifest["windows"])
            )
            self.assertEqual(summary["detected_events"], len(old_manifest["windows"]))

            for new_event, old_event in zip(
                new_manifest["windows"], old_manifest["windows"]
            ):
                for key in (
                    "event_number",
                    "event_start_sample",
                    "event_stop_sample",
                    "start_sample",
                    "stop_sample",
                    "center_shift_hz",
                ):
                    self.assertEqual(new_event[key], old_event[key], key)

                window_id = old_event["window_id"]
                new_image = np.asarray(
                    Image.open(Path(tmp) / "model_inputs" / class_label / f"{window_id}.png")
                )
                old_image = np.asarray(
                    Image.open(OUTPUT_DIR / "model_inputs" / class_label / f"{window_id}.png")
                )
                self.assertTrue(
                    np.array_equal(new_image, old_image), f"pixels differ: {window_id}"
                )

                with np.load(
                    Path(tmp) / "diagnostics" / f"{window_id}.npz"
                ) as new_npz, np.load(
                    OUTPUT_DIR / "diagnostics" / f"{window_id}.npz"
                ) as old_npz:
                    self.assertTrue(np.array_equal(new_npz["psd"], old_npz["psd"]))
                    self.assertTrue(np.array_equal(new_npz["iq"], old_npz["iq"]))

    def test_cw_parity_with_existing_outputs(self):
        self.assert_record_matches_existing("CW", "CW_0001")

    def test_fsk2_parity_with_existing_outputs(self):
        self.assert_record_matches_existing("2FSK", "2FSK_0001")

    def test_lfm_parity_with_existing_outputs(self):
        self.assert_record_matches_existing("LFM", "LFM_0001")


class OverwriteTests(unittest.TestCase):
    def test_skips_when_manifest_exists_and_overwrites_on_demand(self):
        iq_path = DATA_DIR / "CW" / "CW_0001.cf32"
        with tempfile.TemporaryDirectory() as tmp:
            first = process_record(iq_path, output_root=tmp)
            with self.assertRaises(FileExistsError):
                process_record(iq_path, output_root=tmp)
            second = process_record(iq_path, output_root=tmp, overwrite=True)
            self.assertEqual(first["detected_events"], second["detected_events"])


class FragmentSkipTests(unittest.TestCase):
    def test_short_fragments_are_skipped_not_fatal(self):
        iq_path = DATA_DIR / "2FSK" / "2FSK_0026.cf32"
        with tempfile.TemporaryDirectory() as tmp:
            summary = process_record(iq_path, output_root=tmp)
            self.assertEqual(summary["detected_events"], 7)
            self.assertEqual(summary["exported_images"], 6)
            self.assertEqual(len(summary["skipped_events"]), 1)
            images = list(
                (Path(tmp) / "model_inputs" / "2FSK").glob("2FSK_0026_event*.png")
            )
            self.assertEqual(len(images), 6)

    def test_two_fragments_skipped_in_deep_noise_lfm(self):
        iq_path = DATA_DIR / "LFM" / "LFM_0028.cf32"
        with tempfile.TemporaryDirectory() as tmp:
            summary = process_record(iq_path, output_root=tmp)
            self.assertEqual(summary["detected_events"], 7)
            self.assertEqual(summary["exported_images"], 5)
            self.assertEqual(len(summary["skipped_events"]), 2)


class BatchCliTests(unittest.TestCase):
    def test_help_exits_zero(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "spectrum_processing" / "process_batch.py"),
                "--help",
            ],
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )

    def test_small_batch_writes_summary_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(PROJECT_DIR / "spectrum_processing" / "process_batch.py"),
                "--classes", "CW",
                "--limit", "1",
                "--output-root", tmp,
            ]
            completed = subprocess.run(command, capture_output=True)
            self.assertEqual(
                completed.returncode, 0, completed.stderr.decode("utf-8", "replace")
            )
            summary_csv = Path(tmp) / "diagnostics" / "records_processing.csv"
            self.assertTrue(summary_csv.exists())
            with summary_csv.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["class_label"] for row in rows}, {"CW"})
            self.assertEqual({row["detected_events"] for row in rows}, {"6"})
            for row in rows:
                image_path = Path(row["image_path"])
                self.assertTrue(image_path.exists(), image_path)
                self.assertTrue(image_path.name.startswith("CW_0001_event"))

            # 第二次运行全部跳过，但汇总表仍应从 windows.json 完整重建
            completed = subprocess.run(command, capture_output=True)
            self.assertEqual(
                completed.returncode, 0, completed.stderr.decode("utf-8", "replace")
            )
            with summary_csv.open(encoding="utf-8-sig") as handle:
                rows_after_skip = list(csv.DictReader(handle))
            self.assertEqual(len(rows_after_skip), 6)
