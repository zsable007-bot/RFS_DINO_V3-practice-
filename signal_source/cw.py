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
