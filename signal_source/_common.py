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
