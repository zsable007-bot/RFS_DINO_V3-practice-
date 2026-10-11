import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from signal_source import _common  # noqa: E402
from signal_source import cw  # noqa: E402
from signal_source import fsk2  # noqa: E402
from signal_source import lfm  # noqa: E402
from signal_source import generate  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


COMMON_PARAMS = dict(
    samp_rate=1_000_000,
    center_hz=50_000,
    signal_dbfs=-27,
    snr_db=20,
    capture_seconds=2,
    freq_offset_hz=5_000,
    rx_gain_db=0,
    iq_mag_db=0,
    iq_phase_deg=0,
    dc_i=0,
    dc_q=0,
    adc_bits=12,
    noise_seed=20261004,
    slot_pattern=(0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
    slot_seconds=0.1,
    ramp_seconds=0.002,
    realtime=True,
)


class ValidationTests(unittest.TestCase):
    def test_derive_common_quantities(self):
        derived = _common.derive_params(dict(COMMON_PARAMS))
        self.assertAlmostEqual(derived["signal_amp"], 10 ** (-27 / 20))
        self.assertAlmostEqual(
            derived["noise_voltage"], 10 ** (-27 / 20) * 10 ** (-20 / 20)
        )
        self.assertEqual(derived["capture_samples"], 2_000_000)
        self.assertEqual(derived["slot_samples"], 100_000)
        self.assertEqual(derived["ramp_samples"], 2_000)
        self.assertEqual(derived["adc_scale"], 2048)
        self.assertAlmostEqual(derived["adc_max"], 1.0 - 1.0 / 2048)

    def test_derive_samples_per_symbol(self):
        derived = _common.derive_params(
            dict(COMMON_PARAMS, symbol_rate=1000, fsk_deviation_hz=30_000)
        )
        self.assertEqual(derived["samples_per_symbol"], 1_000)

    def test_rejects_non_positive_sample_rate(self):
        with self.assertRaises(ValueError):
            _common.validate_params("CW", dict(COMMON_PARAMS, samp_rate=0))

    def test_rejects_non_positive_capture_seconds(self):
        with self.assertRaises(ValueError):
            _common.validate_params("CW", dict(COMMON_PARAMS, capture_seconds=0))

    def test_rejects_bad_adc_bits(self):
        with self.assertRaises(ValueError):
            _common.validate_params("CW", dict(COMMON_PARAMS, adc_bits=1))

    def test_rejects_bad_slot_pattern(self):
        with self.assertRaises(ValueError):
            _common.validate_params("CW", dict(COMMON_PARAMS, slot_pattern=(0, 2)))

    def test_rejects_fsk_deviation_above_half_sample_rate(self):
        params = dict(COMMON_PARAMS, symbol_rate=1000, fsk_deviation_hz=600_000)
        with self.assertRaises(ValueError):
            _common.validate_params("2FSK", params)

    def test_rejects_non_positive_symbol_rate(self):
        params = dict(COMMON_PARAMS, symbol_rate=0, fsk_deviation_hz=30_000)
        with self.assertRaises(ValueError):
            _common.validate_params("2FSK", params)

    def test_rejects_lfm_bandwidth_above_sample_rate(self):
        params = dict(COMMON_PARAMS, lfm_bandwidth_hz=2_000_000, chirp_period=0.01)
        with self.assertRaises(ValueError):
            _common.validate_params("LFM", params)

    def test_rejects_non_positive_chirp_period(self):
        params = dict(COMMON_PARAMS, lfm_bandwidth_hz=80_000, chirp_period=0)
        with self.assertRaises(ValueError):
            _common.validate_params("LFM", params)

    def test_derive_rejects_sub_sample_capture(self):
        with self.assertRaises(ValueError):
            _common.derive_params(
                dict(COMMON_PARAMS, samp_rate=10, capture_seconds=0.01)
            )


class CwTests(unittest.TestCase):
    def test_cw_parity_with_existing_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "CW_0001.cf32"
            cw.generate(output)
            baseline = DATA_DIR / "CW" / "CW_0001.cf32"
            self.assertEqual(sha256_file(output), sha256_file(baseline))

    def test_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "CW_0001.cf32"
            cw.generate(output, capture_seconds=0.2, realtime=False, write_manifest=False)
            with self.assertRaises(FileExistsError):
                cw.generate(
                    output, capture_seconds=0.2, realtime=False, write_manifest=False
                )

    def test_overwrite_true_replaces_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "CW_0001.cf32"
            cw.generate(output, capture_seconds=0.2, realtime=False, write_manifest=False)
            result = cw.generate(
                output,
                capture_seconds=0.2,
                realtime=False,
                write_manifest=False,
                overwrite=True,
            )
            self.assertEqual(output.stat().st_size, result["samples"] * 8)

    def test_snr_and_seed_change_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.cf32"
            second = Path(tmp) / "second.cf32"
            cw.generate(first, capture_seconds=0.2, realtime=False, write_manifest=False)
            cw.generate(
                second,
                capture_seconds=0.2,
                snr_db=0,
                noise_seed=7,
                realtime=False,
                write_manifest=False,
            )
            self.assertNotEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(second.stat().st_size, int(1_000_000 * 0.2) * 8)

    def test_manifest_written_with_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "CW_0001.cf32"
            result = cw.generate(output, capture_seconds=0.2, realtime=False)
            manifest_path = Path(str(output) + ".json")
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["class_label"], "CW")
            self.assertEqual(payload["sha256"], result["sha256"])
            self.assertEqual(payload["samples"], result["samples"])
            self.assertEqual(payload["source_format"], "cf32_le")
            self.assertEqual(payload["params"]["slot_pattern"], [0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

    def test_invalid_params_raise_before_output_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "CW_0001.cf32"
            with self.assertRaises(ValueError):
                cw.generate(output, snr_db=float("nan"), realtime=False)
            self.assertFalse(output.exists())


class Fsk2Tests(unittest.TestCase):
    def test_fsk2_parity_with_existing_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "2FSK_0001.cf32"
            fsk2.generate(output)
            baseline = DATA_DIR / "2FSK" / "2FSK_0001.cf32"
            self.assertEqual(sha256_file(output), sha256_file(baseline))

    def test_bit_seed_changes_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.cf32"
            second = Path(tmp) / "second.cf32"
            fsk2.generate(first, capture_seconds=0.2, realtime=False, write_manifest=False)
            fsk2.generate(
                second,
                capture_seconds=0.2,
                bit_seed=99,
                realtime=False,
                write_manifest=False,
            )
            self.assertNotEqual(sha256_file(first), sha256_file(second))

    def test_symbol_rate_changes_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            slow = Path(tmp) / "slow.cf32"
            fast = Path(tmp) / "fast.cf32"
            fsk2.generate(slow, capture_seconds=0.2, realtime=False, write_manifest=False)
            fsk2.generate(
                fast,
                capture_seconds=0.2,
                symbol_rate=2000,
                realtime=False,
                write_manifest=False,
            )
            self.assertNotEqual(sha256_file(slow), sha256_file(fast))


class LfmTests(unittest.TestCase):
    def test_lfm_parity_with_existing_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "LFM_0001.cf32"
            lfm.generate(output)
            baseline = DATA_DIR / "LFM" / "LFM_0001.cf32"
            self.assertEqual(sha256_file(output), sha256_file(baseline))

    def test_chirp_period_changes_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.cf32"
            second = Path(tmp) / "second.cf32"
            lfm.generate(first, capture_seconds=0.2, realtime=False, write_manifest=False)
            lfm.generate(
                second,
                capture_seconds=0.2,
                chirp_period=0.005,
                realtime=False,
                write_manifest=False,
            )
            self.assertNotEqual(sha256_file(first), sha256_file(second))


class DispatchTests(unittest.TestCase):
    def test_generate_dispatches_to_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            direct = Path(tmp) / "direct.cf32"
            dispatched = Path(tmp) / "dispatched.cf32"
            cw.generate(direct, capture_seconds=0.2, realtime=False, write_manifest=False)
            generate(
                "CW",
                dispatched,
                capture_seconds=0.2,
                realtime=False,
                write_manifest=False,
            )
            self.assertEqual(sha256_file(direct), sha256_file(dispatched))

    def test_unknown_class_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate("OFDM", Path(tmp) / "x.cf32")


class BatchCliTests(unittest.TestCase):
    def test_small_batch_creates_records_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(PROJECT_DIR / "signal_source" / "generate_batch.py"),
                "--classes", "CW", "2FSK",
                "--snr", "20", "10",
                "--seeds", "5",
                "--seconds", "0.2",
                "--out", tmp,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {p.name for p in Path(tmp).iterdir()},
                {"CW", "2FSK", "records.csv"},
            )
            self.assertTrue((Path(tmp) / "CW" / "CW_0001.cf32").exists())
            self.assertTrue((Path(tmp) / "CW" / "CW_0002.cf32").exists())
            self.assertTrue((Path(tmp) / "2FSK" / "2FSK_0001.cf32").exists())
            self.assertTrue((Path(tmp) / "2FSK" / "2FSK_0002.cf32").exists())
            with (Path(tmp) / "records.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["class_label"] for row in rows}, {"CW", "2FSK"})
            self.assertEqual({row["snr_db"] for row in rows}, {"20.0", "10.0"})

    def test_start_index_auto_continues_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            class_dir = Path(tmp) / "CW"
            class_dir.mkdir()
            (class_dir / "CW_0007.cf32").write_bytes(b"")
            command = [
                sys.executable,
                str(PROJECT_DIR / "signal_source" / "generate_batch.py"),
                "--classes", "CW",
                "--snr", "20",
                "--seconds", "0.2",
                "--out", tmp,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((class_dir / "CW_0008.cf32").exists())

    def test_help_exits_zero(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "signal_source" / "generate_batch.py"),
                "--help",
            ],
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )
