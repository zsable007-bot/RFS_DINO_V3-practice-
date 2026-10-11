# signal_source 信号生成库实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把三类 GNU Radio 生成链（CW/2FSK/LFM）重构为 `signal_source/` 下可编程调用的 headless 库，参数全可调、批量可遍历，且默认参数下输出与现有 `data/*/*_0001.cf32` 逐字节一致。

**Architecture:** `_common.py` 承担公共链路（突发门控→信道→ADC→采集）、参数校验/派生、运行控制（文件大小轮询）、sha256 与 manifest；每类信号模块只负责源段并通过 `_common.generate_signal()` 组装运行；`__init__.py` 提供 `generate(class_name, ...)` 分发；`generate_batch.py` 为参数网格 CLI。

**Tech Stack:** Python 3.12（`D:\Programdata\radioconda\python.exe`，GNU Radio 3.10.12.0）、标准库 `unittest`/`hashlib`/`json`/`argparse`。

**参考规格:** `docs/superpowers/specs/2026-10-10-signal-source-library-design.md`

---

## 前置说明（所有任务适用）

- 工作目录：`D:\workspace\preparation\dinov3_11`（下称项目根）。
- Python 一律用：`& "D:\Programdata\radioconda\python.exe"`；**不要**用 `C:\Python313` 或 dinov3_11 环境（都没有 GNU Radio）。
- 本工作区**不是 git 仓库**，且未要求提交；计划中的"检查点"= 跑测试确认通过，不做 `git commit`。
- 测试命令（在项目根执行）：
  `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
- 基线文件必须存在：`data/CW/CW_0001.cf32`、`data/2FSK/2FSK_0001.cf32`、`data/LFM/LFM_0001.cf32`（各 16,000,000 字节）。
- 逐字节一致要点：所有 GNU Radio 块与参数、连接关系必须与 `GNU_Radio/*_capture.py` 相同；被移除的只有纯 GUI 块（qtgui_*、keep_one_in_n/display_decim）与 Qt 外壳。`realtime=False` 时省去 throttle 不影响数据。

## 文件结构

| 文件 | 职责 |
|---|---|
| `signal_source/__init__.py` | `CLASSES` 常量与 `generate(class_name, ...)` 分发 |
| `signal_source/_common.py` | 校验、派生参数、`CaptureFlowgraph`、运行控制、sha256、manifest |
| `signal_source/cw.py` | CW：50 kHz 复余弦源 + `generate()` |
| `signal_source/fsk2.py` | 2FSK：GLFSR→FM 源 + `generate()` |
| `signal_source/lfm.py` | LFM：锯齿→FM 源 + `generate()` |
| `signal_source/generate_batch.py` | 参数网格 CLI，输出 records.csv |
| `tests/test_signal_source.py` | 全部单元/回归测试（unittest） |

---

### Task 1: 参数校验与派生（TDD）

**Files:**
- Create: `signal_source/__init__.py`
- Create: `signal_source/_common.py`（本任务只含 `validate_params`、`derive_params`）
- Create: `tests/test_signal_source.py`

- [ ] **Step 1: 写失败测试与包骨架**

创建 `tests/test_signal_source.py`：

```python
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
```

创建 `signal_source/__init__.py`（此时仅为包声明，Task 5 再补分发）：

```python
"""signal_source：可编程调用的 IQ 信号生成库（CW / 2FSK / LFM）。"""

__all__ = []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `ImportError: cannot import name '_common' ...`

- [ ] **Step 3: 实现 `signal_source/_common.py` 的校验与派生**

```python
"""公共校验、派生参数、信号链、运行控制与 manifest。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import gr


def validate_params(class_label, params):
    """校验用户参数；失败抛 ValueError，消息包含参数名与当前值。"""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(params["samp_rate"] > 0, f"samp_rate must be > 0, got {params['samp_rate']}")
    require(
        params["capture_seconds"] > 0,
        f"capture_seconds must be > 0, got {params['capture_seconds']}",
    )
    require(
        params["slot_seconds"] > 0,
        f"slot_seconds must be > 0, got {params['slot_seconds']}",
    )
    require(
        params["ramp_seconds"] > 0,
        f"ramp_seconds must be > 0, got {params['ramp_seconds']}",
    )
    require(
        math.isfinite(params["snr_db"]),
        f"snr_db must be finite, got {params['snr_db']}",
    )
    require(
        math.isfinite(params["signal_dbfs"]),
        f"signal_dbfs must be finite, got {params['signal_dbfs']}",
    )
    require(
        math.isfinite(params["freq_offset_hz"]),
        f"freq_offset_hz must be finite, got {params['freq_offset_hz']}",
    )
    require(params["adc_bits"] >= 2, f"adc_bits must be >= 2, got {params['adc_bits']}")
    pattern = params["slot_pattern"]
    require(len(pattern) > 0, "slot_pattern must not be empty")
    require(
        all(value in (0, 1) for value in pattern),
        f"slot_pattern values must be 0 or 1, got {pattern}",
    )
    if class_label == "2FSK":
        require(
            params["symbol_rate"] > 0,
            f"symbol_rate must be > 0, got {params['symbol_rate']}",
        )
        require(
            0 < params["fsk_deviation_hz"] < params["samp_rate"] / 2,
            "fsk_deviation_hz must be in (0, samp_rate/2), "
            f"got {params['fsk_deviation_hz']}",
        )
    if class_label == "LFM":
        require(
            0 < params["lfm_bandwidth_hz"] < params["samp_rate"],
            "lfm_bandwidth_hz must be in (0, samp_rate), "
            f"got {params['lfm_bandwidth_hz']}",
        )
        require(
            params["chirp_period"] > 0,
            f"chirp_period must be > 0, got {params['chirp_period']}",
        )


def derive_params(params):
    """计算派生量；派生的采样数不足 1 时抛 ValueError。"""
    derived = dict(params)
    derived["signal_amp"] = 10 ** (derived["signal_dbfs"] / 20)
    derived["noise_voltage"] = derived["signal_amp"] * 10 ** (-derived["snr_db"] / 20)
    derived["rx_gain"] = 10 ** (derived["rx_gain_db"] / 20)
    derived["adc_scale"] = 2 ** (derived["adc_bits"] - 1)
    derived["adc_max"] = 1.0 - 1.0 / derived["adc_scale"]
    derived["slot_samples"] = int(derived["samp_rate"] * derived["slot_seconds"])
    derived["ramp_samples"] = int(derived["samp_rate"] * derived["ramp_seconds"])
    derived["capture_samples"] = int(derived["samp_rate"] * derived["capture_seconds"])
    if "symbol_rate" in derived:
        derived["samples_per_symbol"] = int(
            derived["samp_rate"] / derived["symbol_rate"]
        )
        if derived["samples_per_symbol"] < 1:
            raise ValueError(
                "samples_per_symbol < 1 "
                f"(samp_rate={derived['samp_rate']}, "
                f"symbol_rate={derived['symbol_rate']})"
            )
    if derived["capture_samples"] < 1:
        raise ValueError(
            "capture_samples < 1 "
            f"(samp_rate={derived['samp_rate']}, "
            f"capture_seconds={derived['capture_seconds']})"
        )
    if derived["slot_samples"] < 1 or derived["ramp_samples"] < 1:
        raise ValueError(
            "slot_samples and ramp_samples must be >= 1 "
            f"(got {derived['slot_samples']} and {derived['ramp_samples']})"
        )
    return derived
```

- [ ] **Step 4: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`（11 个 ValidationTests 全部通过）

- [ ] **Step 5: 检查点**

确认无临时文件残留，记录测试输出。本轮不提交（非 git 仓库）。

---

### Task 2: 公共链路 + 运行控制 + CW + manifest/覆盖（TDD）

**Files:**
- Modify: `signal_source/_common.py`（追加链与运行代码）
- Create: `signal_source/cw.py`
- Modify: `tests/test_signal_source.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_signal_source.py` 末尾追加（`import` 行保持文件顶部，`from signal_source import cw` 加到顶部 import 区块）：

```python
from signal_source import cw  # noqa: E402  (放到文件顶部 import 区块)


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `ImportError: cannot import name 'cw' ...`

- [ ] **Step 3: 在 `_common.py` 追加链与运行代码**

在 `derive_params` 之后追加：

```python
class CaptureFlowgraph(gr.top_block):
    """headless 采集流图：源块 → 公共链路 → file_sink。"""

    def __init__(self, label, build_source, params, output_path):
        gr.top_block.__init__(self, label, catch_exceptions=True)
        samp_rate = params["samp_rate"]

        self.vector_source = blocks.vector_source_f(
            list(params["slot_pattern"]), True, 1, []
        )
        self.repeat_slot = blocks.repeat(gr.sizeof_float * 1, params["slot_samples"])
        self.ramp = blocks.moving_average_ff(
            params["ramp_samples"], 1.0 / params["ramp_samples"], 4000, 1
        )
        self.envelope = blocks.float_to_complex(1)
        self.multiply = blocks.multiply_vcc(1)
        self.channel = channels.channel_model(
            noise_voltage=params["noise_voltage"],
            frequency_offset=params["freq_offset_hz"] / samp_rate,
            epsilon=1.0,
            taps=[1 + 0j],
            noise_seed=params["noise_seed"],
            block_tags=False,
        )
        self.rx_gain = blocks.multiply_const_cc(params["rx_gain"])
        self.iqbal = channels.iqbal_gen(
            params["iq_mag_db"], params["iq_phase_deg"], 1
        )
        self.dc_offset = blocks.add_const_cc(complex(params["dc_i"], params["dc_q"]))
        self.split = blocks.complex_to_float(1)
        self.rail_i = analog.rail_ff((-1), params["adc_max"])
        self.rail_q = analog.rail_ff((-1), params["adc_max"])
        self.to_short_i = blocks.float_to_short(1, params["adc_scale"])
        self.to_short_q = blocks.float_to_short(1, params["adc_scale"])
        self.to_float_i = blocks.short_to_float(1, params["adc_scale"])
        self.to_float_q = blocks.short_to_float(1, params["adc_scale"])
        self.merge = blocks.float_to_complex(1)
        self.head = blocks.head(gr.sizeof_gr_complex * 1, params["capture_samples"])
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex * 1, str(output_path), False
        )
        self.file_sink.set_unbuffered(True)

        signal_block = build_source(self, params)
        self.connect((signal_block, 0), (self.multiply, 0))
        self.connect((self.vector_source, 0), (self.repeat_slot, 0))
        self.connect((self.repeat_slot, 0), (self.ramp, 0))
        self.connect((self.ramp, 0), (self.envelope, 0))
        self.connect((self.envelope, 0), (self.multiply, 1))
        self.connect((self.multiply, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.rx_gain, 0))
        self.connect((self.rx_gain, 0), (self.iqbal, 0))
        self.connect((self.iqbal, 0), (self.dc_offset, 0))
        self.connect((self.dc_offset, 0), (self.split, 0))
        self.connect((self.split, 0), (self.rail_i, 0))
        self.connect((self.split, 1), (self.rail_q, 0))
        self.connect((self.rail_i, 0), (self.to_short_i, 0))
        self.connect((self.rail_q, 0), (self.to_short_q, 0))
        self.connect((self.to_short_i, 0), (self.to_float_i, 0))
        self.connect((self.to_short_q, 0), (self.to_float_q, 0))
        self.connect((self.to_float_i, 0), (self.merge, 0))
        self.connect((self.to_float_q, 0), (self.merge, 1))

        if params["realtime"]:
            self.throttle = blocks.throttle(
                gr.sizeof_gr_complex * 1, samp_rate, True
            )
            self.connect((self.merge, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.head, 0))
        else:
            self.connect((self.merge, 0), (self.head, 0))

        self.connect((self.head, 0), (self.file_sink, 0))


def prepare_output(output_path, overwrite):
    """检查/创建输出路径；已存在且不允许覆盖时抛 FileExistsError。"""
    path = Path(output_path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path} (pass overwrite=True to replace)"
            )
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_flowgraph(tb, params, output_path, timeout_seconds=None):
    """启动流图，轮询输出文件达到期望字节数后停止；超时抛 TimeoutError。"""
    expected_bytes = params["capture_samples"] * 8
    if timeout_seconds is None:
        timeout_seconds = params["capture_seconds"] + 10.0
    deadline = time.monotonic() + timeout_seconds
    tb.start()
    try:
        while True:
            if output_path.exists() and output_path.stat().st_size >= expected_bytes:
                break
            if time.monotonic() > deadline:
                size = output_path.stat().st_size if output_path.exists() else 0
                raise TimeoutError(
                    f"Capture did not finish in {timeout_seconds:.1f} s: "
                    f"{size} of {expected_bytes} bytes"
                )
            time.sleep(0.05)
    finally:
        tb.stop()
        tb.wait()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(class_label, output_path, params, digest):
    payload = {
        "class_label": class_label,
        "generator": "signal_source",
        "gnuradio_version": gr.version(),
        "source_format": "cf32_le",
        "sample_rate_hz": float(params["samp_rate"]),
        "samples": int(params["capture_samples"]),
        "duration_seconds": float(params["capture_seconds"]),
        "sha256": digest,
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in params.items()
        },
    }
    manifest_path = Path(str(output_path) + ".json")
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def generate_signal(
    class_label, output_path, build_source, params, overwrite, write_manifest_flag
):
    """统一入口：校验 → 派生 → 建图运行 → 计算摘要 → 写 manifest → 返回结果。"""
    params = dict(params)
    validate_params(class_label, params)
    params = derive_params(params)
    output_path = prepare_output(output_path, overwrite)
    flowgraph = CaptureFlowgraph(class_label, build_source, params, output_path)
    start = time.monotonic()
    run_flowgraph(flowgraph, params, output_path)
    elapsed = time.monotonic() - start
    digest = sha256_file(output_path)
    result = {
        "output_path": str(output_path),
        "class_label": class_label,
        "samples": params["capture_samples"],
        "bytes": output_path.stat().st_size,
        "sha256": digest,
        "elapsed_seconds": elapsed,
        "params": params,
    }
    if write_manifest_flag:
        write_manifest(class_label, output_path, params, digest)
    return result
```

- [ ] **Step 4: 创建 `signal_source/cw.py`**

```python
"""CW（单频连续波突发）信号生成。"""

from gnuradio import analog

from . import _common


def _build_source(tb, params):
    return analog.sig_source_c(
        params["samp_rate"],
        analog.GR_COS_WAVE,
        params["center_hz"],
        params["signal_amp"],
        0,
        0,
    )


def generate(
    output_path,
    *,
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
    overwrite=False,
    write_manifest=True,
):
    """生成一段 CW 突发 IQ 记录并写入 output_path（cf32_le）。"""
    params = {
        "samp_rate": samp_rate,
        "center_hz": center_hz,
        "signal_dbfs": signal_dbfs,
        "snr_db": snr_db,
        "capture_seconds": capture_seconds,
        "freq_offset_hz": freq_offset_hz,
        "rx_gain_db": rx_gain_db,
        "iq_mag_db": iq_mag_db,
        "iq_phase_deg": iq_phase_deg,
        "dc_i": dc_i,
        "dc_q": dc_q,
        "adc_bits": adc_bits,
        "noise_seed": noise_seed,
        "slot_pattern": tuple(slot_pattern),
        "slot_seconds": slot_seconds,
        "ramp_seconds": ramp_seconds,
        "realtime": realtime,
    }
    return _common.generate_signal(
        "CW", output_path, _build_source, params, overwrite, write_manifest
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`（ValidationTests + CwTests 全部通过；CW parity 约 2.5 s）

若 `test_cw_parity_with_existing_capture` 失败：先比对文件大小，再 diff 前 1024 字节与样本统计，确认是块参数不一致（修复）还是环境问题（记录后上报），不得放宽测试。

- [ ] **Step 6: 检查点**

记录 `test_cw_parity_with_existing_capture` 输出（sha256 相等）。不提交。

---

### Task 3: 2FSK 模块（TDD）

**Files:**
- Create: `signal_source/fsk2.py`
- Modify: `tests/test_signal_source.py`

- [ ] **Step 1: 追加失败测试**

文件顶部 import 区块追加 `from signal_source import fsk2`；末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `ImportError: cannot import name 'fsk2' ...`

- [ ] **Step 3: 创建 `signal_source/fsk2.py`**

```python
"""2FSK（二进制频移键控突发）信号生成。"""

import math

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr

from . import _common


def _build_source(tb, params):
    bits = digital.glfsr_source_b(15, True, 0, params["bit_seed"])
    bits_to_float = blocks.uchar_to_float()
    repeat_symbol = blocks.repeat(gr.sizeof_float * 1, params["samples_per_symbol"])
    scale = blocks.multiply_const_ff(2 * params["fsk_deviation_hz"])
    offset = blocks.add_const_ff(params["center_hz"] - params["fsk_deviation_hz"])
    modulator = analog.frequency_modulator_fc(2 * math.pi / params["samp_rate"])
    amplitude = blocks.multiply_const_cc(params["signal_amp"])

    tb.connect((bits, 0), (bits_to_float, 0))
    tb.connect((bits_to_float, 0), (repeat_symbol, 0))
    tb.connect((repeat_symbol, 0), (scale, 0))
    tb.connect((scale, 0), (offset, 0))
    tb.connect((offset, 0), (modulator, 0))
    tb.connect((modulator, 0), (amplitude, 0))
    return amplitude
```

`generate()` 与 `cw.generate` 同构：除公共参数外多出 `symbol_rate=1000`、`fsk_deviation_hz=30_000`、`bit_seed=11`，参数 dict 中额外放入这三项：

```python
def generate(
    output_path,
    *,
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
    symbol_rate=1000,
    fsk_deviation_hz=30_000,
    bit_seed=11,
    realtime=True,
    overwrite=False,
    write_manifest=True,
):
    """生成一段 2FSK 突发 IQ 记录并写入 output_path（cf32_le）。"""
    params = {
        "samp_rate": samp_rate,
        "center_hz": center_hz,
        "signal_dbfs": signal_dbfs,
        "snr_db": snr_db,
        "capture_seconds": capture_seconds,
        "freq_offset_hz": freq_offset_hz,
        "rx_gain_db": rx_gain_db,
        "iq_mag_db": iq_mag_db,
        "iq_phase_deg": iq_phase_deg,
        "dc_i": dc_i,
        "dc_q": dc_q,
        "adc_bits": adc_bits,
        "noise_seed": noise_seed,
        "slot_pattern": tuple(slot_pattern),
        "slot_seconds": slot_seconds,
        "ramp_seconds": ramp_seconds,
        "symbol_rate": symbol_rate,
        "fsk_deviation_hz": fsk_deviation_hz,
        "bit_seed": bit_seed,
        "realtime": realtime,
    }
    return _common.generate_signal(
        "2FSK", output_path, _build_source, params, overwrite, write_manifest
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`（2FSK parity 约 2.5 s）

- [ ] **Step 5: 检查点**

记录 2FSK parity 结果。不提交。

---

### Task 4: LFM 模块（TDD）

**Files:**
- Create: `signal_source/lfm.py`
- Modify: `tests/test_signal_source.py`

- [ ] **Step 1: 追加失败测试**

import 区块追加 `from signal_source import lfm`；末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `ImportError: cannot import name 'lfm' ...`

- [ ] **Step 3: 创建 `signal_source/lfm.py`**

```python
"""LFM（线性调频锯齿波突发）信号生成。"""

import math

from gnuradio import analog
from gnuradio import blocks

from . import _common


def _build_source(tb, params):
    saw = analog.sig_source_f(
        params["samp_rate"],
        analog.GR_SAW_WAVE,
        1.0 / params["chirp_period"],
        params["lfm_bandwidth_hz"],
        params["center_hz"] - params["lfm_bandwidth_hz"] / 2,
        -math.pi,
    )
    modulator = analog.frequency_modulator_fc(2 * math.pi / params["samp_rate"])
    amplitude = blocks.multiply_const_cc(params["signal_amp"])
    tb.connect((saw, 0), (modulator, 0))
    tb.connect((modulator, 0), (amplitude, 0))
    return amplitude


def generate(
    output_path,
    *,
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
    lfm_bandwidth_hz=80_000,
    chirp_period=0.01,
    realtime=True,
    overwrite=False,
    write_manifest=True,
):
    """生成一段 LFM 突发 IQ 记录并写入 output_path（cf32_le）。"""
    params = {
        "samp_rate": samp_rate,
        "center_hz": center_hz,
        "signal_dbfs": signal_dbfs,
        "snr_db": snr_db,
        "capture_seconds": capture_seconds,
        "freq_offset_hz": freq_offset_hz,
        "rx_gain_db": rx_gain_db,
        "iq_mag_db": iq_mag_db,
        "iq_phase_deg": iq_phase_deg,
        "dc_i": dc_i,
        "dc_q": dc_q,
        "adc_bits": adc_bits,
        "noise_seed": noise_seed,
        "slot_pattern": tuple(slot_pattern),
        "slot_seconds": slot_seconds,
        "ramp_seconds": ramp_seconds,
        "lfm_bandwidth_hz": lfm_bandwidth_hz,
        "chirp_period": chirp_period,
        "realtime": realtime,
    }
    return _common.generate_signal(
        "LFM", output_path, _build_source, params, overwrite, write_manifest
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`（LFM parity 约 2.5 s）

- [ ] **Step 5: 检查点**

记录 LFM parity 结果。不提交。

---

### Task 5: 分发入口（TDD）

**Files:**
- Modify: `signal_source/__init__.py`
- Modify: `tests/test_signal_source.py`

- [ ] **Step 1: 追加失败测试**

文件顶部 import 改为 `from signal_source import cw, fsk2, lfm, generate`（保留 `_common` import）；末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `ImportError: cannot import name 'generate' ...`（或 `__init__` 无该名称）

- [ ] **Step 3: 实现 `signal_source/__init__.py`**

```python
"""signal_source：可编程调用的 IQ 信号生成库（CW / 2FSK / LFM）。"""

from . import cw, fsk2, lfm

CLASSES = ("CW", "2FSK", "LFM")

_MODULES = {"CW": cw, "2FSK": fsk2, "LFM": lfm}


def generate(class_name, output_path, **params):
    """按类别分发到对应模块的 generate()。"""
    try:
        module = _MODULES[class_name]
    except KeyError:
        raise ValueError(
            f"Unknown class_name {class_name!r}; expected one of {CLASSES}"
        ) from None
    return module.generate(output_path, **params)


__all__ = ["CLASSES", "generate", "cw", "fsk2", "lfm"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`

- [ ] **Step 5: 检查点**

不提交。

---

### Task 6: 批量 CLI（TDD）

**Files:**
- Create: `signal_source/generate_batch.py`
- Modify: `tests/test_signal_source.py`

- [ ] **Step 1: 追加失败测试**

末尾追加：

```python
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
```

文件顶部 import 区块追加 `import csv`（`subprocess`/`sys` 已有）。

- [ ] **Step 2: 运行测试确认失败**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: FAIL — `generate_batch.py` 不存在，子进程返回码非 0

- [ ] **Step 3: 创建 `signal_source/generate_batch.py`**

```python
"""批量生成参数网格 IQ 数据。

用法示例：
    python signal_source/generate_batch.py --classes CW 2FSK LFM \\
        --snr 20 10 5 0 --seeds 101 102 103 --bit-seeds 11 12 13 \\
        --seconds 2 --out data

输出：<out>/<类>/<类>_<NNNN>.cf32 与汇总 <out>/records.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import signal_source  # noqa: E402

FIELDNAMES = [
    "class_label",
    "file",
    "snr_db",
    "noise_seed",
    "bit_seed",
    "symbol_rate",
    "fsk_deviation_hz",
    "lfm_bandwidth_hz",
    "chirp_period",
    "sha256",
    "samples",
    "created",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--classes", nargs="+", choices=signal_source.CLASSES,
        default=list(signal_source.CLASSES),
    )
    parser.add_argument("--snr", nargs="+", type=float, default=[20.0])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="噪声种子列表；缺省沿用库默认 20261004",
    )
    parser.add_argument(
        "--bit-seeds", nargs="+", type=int, default=None,
        help="2FSK 比特种子，与 --seeds 一一对应；缺省全部用 11",
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument(
        "--start-index", default="auto", help="起始编号；auto=每类现有最大编号+1"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--realtime", action="store_true", help="按实时速度生成（默认全速）"
    )
    parser.add_argument("--samp-rate", type=float, default=1_000_000)
    parser.add_argument("--center-hz", type=float, default=50_000)
    parser.add_argument("--signal-dbfs", type=float, default=-27)
    parser.add_argument("--freq-offset-hz", type=float, default=5_000)
    parser.add_argument("--rx-gain-db", type=float, default=0)
    parser.add_argument("--iq-mag-db", type=float, default=0)
    parser.add_argument("--iq-phase-deg", type=float, default=0)
    parser.add_argument("--dc-i", type=float, default=0)
    parser.add_argument("--dc-q", type=float, default=0)
    parser.add_argument("--adc-bits", type=int, default=12)
    parser.add_argument("--symbol-rate", type=float, default=1000)
    parser.add_argument("--fsk-deviation-hz", type=float, default=30_000)
    parser.add_argument("--lfm-bandwidth-hz", type=float, default=80_000)
    parser.add_argument("--chirp-period", type=float, default=0.01)
    return parser


def resolve_start_index(class_dir, class_label, value):
    if value != "auto":
        return int(value)
    highest = 0
    for path in class_dir.glob(f"{class_label}_*.cf32"):
        suffix = path.stem[len(class_label) + 1:]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def run(args):
    if (
        args.bit_seeds is not None
        and args.seeds is not None
        and len(args.bit_seeds) != len(args.seeds)
    ):
        raise SystemExit("--bit-seeds 数量必须与 --seeds 一致")
    seeds = args.seeds if args.seeds is not None else [None]
    bit_seeds = args.bit_seeds
    rows = []
    for class_label in args.classes:
        class_dir = args.out / class_label
        class_dir.mkdir(parents=True, exist_ok=True)
        index = resolve_start_index(class_dir, class_label, args.start_index)
        for snr_db in args.snr:
            for position, noise_seed in enumerate(seeds):
                output_path = class_dir / f"{class_label}_{index:04d}.cf32"
                index += 1
                kwargs = {
                    "snr_db": snr_db,
                    "capture_seconds": args.seconds,
                    "samp_rate": args.samp_rate,
                    "center_hz": args.center_hz,
                    "signal_dbfs": args.signal_dbfs,
                    "freq_offset_hz": args.freq_offset_hz,
                    "rx_gain_db": args.rx_gain_db,
                    "iq_mag_db": args.iq_mag_db,
                    "iq_phase_deg": args.iq_phase_deg,
                    "dc_i": args.dc_i,
                    "dc_q": args.dc_q,
                    "adc_bits": args.adc_bits,
                    "realtime": args.realtime,
                    "overwrite": args.overwrite,
                }
                if noise_seed is not None:
                    kwargs["noise_seed"] = noise_seed
                if class_label == "2FSK":
                    kwargs["symbol_rate"] = args.symbol_rate
                    kwargs["fsk_deviation_hz"] = args.fsk_deviation_hz
                    if bit_seeds is not None:
                        kwargs["bit_seed"] = bit_seeds[position]
                if class_label == "LFM":
                    kwargs["lfm_bandwidth_hz"] = args.lfm_bandwidth_hz
                    kwargs["chirp_period"] = args.chirp_period
                try:
                    result = signal_source.generate(class_label, output_path, **kwargs)
                except FileExistsError as error:
                    print(f"SKIP {output_path}: {error}")
                    continue
                params = result["params"]
                print(
                    f"DONE {output_path} "
                    f"sha256={result['sha256'][:12]} samples={result['samples']}"
                )
                rows.append(
                    {
                        "class_label": class_label,
                        "file": str(output_path),
                        "snr_db": snr_db,
                        "noise_seed": params["noise_seed"],
                        "bit_seed": params.get("bit_seed", ""),
                        "symbol_rate": params.get("symbol_rate", ""),
                        "fsk_deviation_hz": params.get("fsk_deviation_hz", ""),
                        "lfm_bandwidth_hz": params.get("lfm_bandwidth_hz", ""),
                        "chirp_period": params.get("chirp_period", ""),
                        "sha256": result["sha256"],
                        "samples": result["samples"],
                        "created": datetime.now().isoformat(timespec="seconds"),
                    }
                )
    csv_path = args.out / "records.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Records: {len(rows)}; summary: {csv_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`

- [ ] **Step 5: 手动冒烟**

Run:
`& "D:\Programdata\radioconda\python.exe" signal_source\generate_batch.py --classes CW 2FSK LFM --snr 20 10 --seeds 101 102 --seconds 0.2 --out %TEMP%\signal_source_smoke`
Expected: 12 条 `DONE ...`，`records.csv` 12 行；检查后删除该临时目录。

- [ ] **Step 6: 检查点**

不提交。

---

### Task 7: 全量回归与验收

**Files:** 无新增

- [ ] **Step 1: 全量测试**

Run: `& "D:\Programdata\radioconda\python.exe" -m unittest discover -s tests -v`
Expected: `OK`，含三条 parity（CW/2FSK/LFM）全部通过

- [ ] **Step 2: 逐条核对规格验收清单**

对照 `docs/superpowers/specs/2026-10-10-signal-source-library-design.md` 第 9 节：

1. 三条 parity sha256 相等（Task 2/3/4 已验证）
2. `snr_db=0` / 换种子数据变化且大小正确（Task 2 `test_snr_and_seed_change_data`）
3. 分发入口与直调一致（Task 5）
4. 覆盖拒绝与非法参数（Task 2 / Task 1）
5. CLI `--help` 与小批量（Task 6）
6. 无残留临时文件；`data/` 下基线文件未被修改（`git status` 不适用，用文件大小/时间抽查）

- [ ] **Step 3: 完成**

向用户汇报：新增/修改文件清单、测试命令与结果、基线 sha256 对比、批量用法示例。不做 git 提交。

---

## Self-Review 记录

- 规格覆盖：接口签名（4.1/4.2）→ Task 2/3/4/5；返回值（4.3）→ Task 2；链路（5）→ Task 2/3/4；运行控制与校验（6）→ Task 1/2；manifest（7）→ Task 2；批量 CLI（8）→ Task 6；验证（9）→ Task 2/3/4/6/7。
- 类型/命名一致性：`generate_signal`、`CaptureFlowgraph`、`build_source`、`params["realtime"]` 在各任务一致；`write_manifest` 参数在模块层与 `_common.write_manifest()` 函数间存在同名（参数名 vs 函数名），已在实现中通过位置参数区分（`generate_signal(..., write_manifest_flag)`），无冲突。
- 占位符：全文已复查，无 TBD/TODO/错误占位代码；Task 3 的 `_build_source` 代码块为可直接使用的最终版本。
- 已知取舍：parity 测试用默认真实时间生成（各约 2.5 s），换取对 polling+throttle 路径的覆盖。
