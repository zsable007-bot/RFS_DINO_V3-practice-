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
