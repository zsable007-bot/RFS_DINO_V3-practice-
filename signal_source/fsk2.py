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
