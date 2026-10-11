"""IQ → STFT → 事件检测 → 窗口截取 → 时频灰度图（CW/2FSK/LFM 共用）。

本模块把三个 notebook（iq_to_stft.py.ipynb、fsk2_check.ipynb、lfm_check.ipynb）
中完全相同的处理流程收敛为一个函数 process_record()，数值与图像输出与原
notebook 逐位/逐像素一致（见 tests/test_spectrum_processing.py 回归测试）。

输出约定（与既有 notebook 相同）：
    <output_root>/model_inputs/<类>/<record>_eventNN_window01.png
    <output_root>/diagnostics/<record>_eventNN_window01.npz
    <output_root>/diagnostics/<record>_windows.json
    make_plots=True 时另存 <record>_stft.png 与 <record>_detection.png

运行环境：dinov3_11 环境的 Python（numpy 2.5.3 + Pillow；绘图才需要 matplotlib）；
不需要 GNU Radio。跨 numpy 版本存在末位浮点差异（灰度像素不受影响），既有产物
以 dinov3_11 环境为逐位基准。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

FS_HZ = 1_000_000.0
NFFT = 512
HOP = 256
BAND_LOW_HZ = -250_000.0
BAND_HIGH_HZ = 250_000.0
THRESHOLD_DB = 10.0
MINIMUM_SECONDS = 0.005
MERGE_GAP_SECONDS = 0.002
WINDOW_SAMPLES = 65_536
FILTER_TAPS = 129
FILTER_CUTOFF_HZ = 225_000.0
FILTER_KAISER_BETA = 8.6
GRAY_LOW_DB = -110.0
GRAY_HIGH_DB = -30.0
LABEL_SOURCE = "GNU Radio signal-generation settings"
WINDOW_POLICY = "one centered full window per event"
IMAGE_ADAPTATION = "copy_last_time_column"

_DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


def load_iq(iq_path):
    """读取小端 complex64 IQ；校验文件大小与有限性。"""
    iq_path = Path(iq_path)
    if iq_path.stat().st_size % 8 != 0:
        raise ValueError(f"Invalid IQ file size (not a multiple of 8): {iq_path}")
    iq = np.fromfile(iq_path, dtype="<c8")
    if iq.size == 0:
        raise ValueError(f"Empty IQ file: {iq_path}")
    if not np.isfinite(iq).all():
        raise ValueError(f"IQ contains NaN or Inf: {iq_path}")
    return iq


def compute_record_stft(iq):
    """整轨 STFT：返回 (psd_band, frequency_band, times)。"""
    window = np.hanning(NFFT + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(iq, NFFT)[::HOP]
    spectrum = np.fft.fft(frames * window, axis=1)
    psd = np.abs(spectrum) ** 2 / (FS_HZ * np.sum(window ** 2))
    psd = np.fft.fftshift(psd, axes=1).T
    frequencies = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / FS_HZ))
    times = (np.arange(frames.shape[0]) * HOP + NFFT / 2) / FS_HZ
    keep = (frequencies >= BAND_LOW_HZ) & (frequencies < BAND_HIGH_HZ)
    return psd[keep], frequencies[keep], times


def detect_events(psd_band, sample_count):
    """噪声估计 + 逐帧检测 + 合并/过滤；返回 (events, active, noise_psd, score_db)。"""
    tiny = np.finfo(np.float64).tiny
    noise_stride = max(1, int(np.ceil(psd_band.shape[1] / 256)))
    noise_psd = max(
        float(np.median(psd_band[:, ::noise_stride])) / np.log(2),
        tiny,
    )
    three_bin_psd = (psd_band[:-2] + psd_band[1:-1] + psd_band[2:]) / 3.0
    peak_psd = three_bin_psd.max(axis=0)
    score_db = 10 * np.log10(np.maximum(peak_psd, tiny) / noise_psd)
    active = score_db >= THRESHOLD_DB

    minimum_samples = int(np.ceil(MINIMUM_SECONDS * FS_HZ))
    merge_gap_samples = int(np.floor(MERGE_GAP_SECONDS * FS_HZ))
    events = []
    for frame_index in np.flatnonzero(active):
        start_sample = int(frame_index * HOP)
        stop_sample = min(start_sample + NFFT, sample_count)
        if events and start_sample - events[-1]["stop_sample"] <= merge_gap_samples:
            events[-1]["stop_sample"] = stop_sample
        else:
            events.append({"start_sample": start_sample, "stop_sample": stop_sample})
    events = [
        event
        for event in events
        if event["stop_sample"] - event["start_sample"] >= minimum_samples
    ]
    return events, active, noise_psd, score_db


def estimate_center_hz(psd_band, frequency_band, active, event, noise_psd):
    """按检测段内能量加权质心估计中心频率。"""
    frame_starts = np.arange(psd_band.shape[1]) * HOP
    selected = (
        (frame_starts >= event["start_sample"])
        & (frame_starts + NFFT <= event["stop_sample"])
        & active
    )
    weights = np.maximum(
        psd_band[:, selected] - 4.0 * noise_psd,
        0.0,
    ).sum(axis=1)
    if weights.sum() <= 0:
        raise ValueError("Insufficient signal energy")
    return float(np.dot(frequency_band, weights) / weights.sum())


def build_filter_taps():
    """129 点对称 Kaiser 窗低通 FIR。"""
    delay = (FILTER_TAPS - 1) // 2
    positions = np.arange(FILTER_TAPS) - delay
    taps = (
        (2 * FILTER_CUTOFF_HZ / FS_HZ)
        * np.sinc((2 * FILTER_CUTOFF_HZ / FS_HZ) * positions)
        * np.kaiser(FILTER_TAPS, FILTER_KAISER_BETA)
    )
    return taps / taps.sum()


def extract_event(iq, event, center_hz, taps):
    """从检测段中间取 65536 点：混频移到 0 Hz + FIR；返回 (values, start, stop, ctx_start, ctx_stop)。"""
    event_length = event["stop_sample"] - event["start_sample"]
    if event_length < WINDOW_SAMPLES:
        raise ValueError("Event is shorter than one full window")
    start = int(event["start_sample"] + (event_length - WINDOW_SAMPLES) // 2)
    stop = start + WINDOW_SAMPLES

    delay = (len(taps) - 1) // 2
    context_start = start - delay
    context_stop = stop + delay
    if context_start < 0 or context_stop > iq.size:
        raise ValueError("Insufficient neighboring IQ for FIR")

    context = np.asarray(iq[context_start:context_stop], dtype=np.complex128)
    positions = np.arange(context_start, context_stop)
    shifted = context * np.exp(-2j * np.pi * center_hz * positions / FS_HZ)
    values = np.convolve(shifted, taps, mode="valid")
    return values, start, stop, context_start, context_stop


def compute_window_psd(values):
    """截取窗口的 STFT；返回 (psd_cut, frequency_hz, times_local)。"""
    window = np.hanning(NFFT + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(values, NFFT)[::HOP]
    transformed = np.fft.fft(frames * window, axis=1)
    psd = np.abs(transformed) ** 2 / (FS_HZ * np.sum(window ** 2))
    psd = np.fft.fftshift(psd, axes=1).T

    frequencies = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / FS_HZ))
    band = (frequencies >= BAND_LOW_HZ) & (frequencies < BAND_HIGH_HZ)
    psd_cut = psd[band]
    times_local = (np.arange(frames.shape[0]) * HOP + NFFT / 2) / FS_HZ
    if psd_cut.shape != (256, 255):
        raise ValueError(f"Unexpected PSD shape: {psd_cut.shape}")
    return psd_cut, frequencies[band], times_local


def psd_to_rgb(psd_cut):
    """固定灰度范围 (-110, -30) dB、高频在上、复制末列成正方形、复制为三通道。"""
    db_values = 10 * np.log10(
        np.maximum(psd_cut, np.finfo(np.float64).tiny)
    )
    gray = np.rint(
        np.clip((db_values + 110.0) / 80.0, 0.0, 1.0) * 255
    ).astype(np.uint8)
    gray = gray[::-1, :]
    square = np.concatenate((gray, gray[:, -1:]), axis=1)
    return np.repeat(square[:, :, None], 3, axis=2)


def _save_diagnostic_plots(stem, iq, times, frequency_band, psd_band, score_db, events, diagnostics_dir):
    import matplotlib.pyplot as plt

    db = 10 * np.log10(np.maximum(psd_band, np.finfo(np.float64).tiny))
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    mesh = axes[0].pcolormesh(
        times,
        frequency_band / 1000,
        db,
        shading="auto",
        cmap="magma",
        vmin=GRAY_LOW_DB,
        vmax=GRAY_HIGH_DB,
        rasterized=True,
    )
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_title(f"{stem} - Signal detection")
    axes[1].plot(times, score_db, linewidth=0.8, label="Detection score")
    axes[1].axhline(THRESHOLD_DB, color="red", linestyle="--", label="Threshold")
    for event in events:
        start_seconds = event["start_sample"] / FS_HZ
        stop_seconds = event["stop_sample"] / FS_HZ
        for ax in axes:
            ax.axvspan(start_seconds, stop_seconds, color="lime", alpha=0.15)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Peak / noise (dB)")
    axes[1].set_xlim(0, iq.size / FS_HZ)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(diagnostics_dir / f"{stem}_detection.png", dpi=160)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(12, 4))
    mesh = ax.pcolormesh(
        times,
        frequency_band / 1000,
        db,
        shading="auto",
        cmap="magma",
        vmin=GRAY_LOW_DB,
        vmax=GRAY_HIGH_DB,
        rasterized=True,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{stem} - Full-record STFT")
    ax.set_xlim(0, iq.size / FS_HZ)
    figure.colorbar(mesh, ax=ax, label="PSD (dBFS/Hz)")
    figure.tight_layout()
    figure.savefig(diagnostics_dir / f"{stem}_stft.png", dpi=160)
    plt.close(figure)


def process_record(
    iq_path,
    class_label=None,
    output_root=None,
    make_plots=False,
    overwrite=False,
):
    """处理一条 IQ 记录：检测事件、导出灰度图/npz/windows.json，返回摘要 dict。"""
    iq_path = Path(iq_path).resolve()
    if class_label is None:
        class_label = iq_path.parent.name
    output_root = Path(output_root) if output_root is not None else _DEFAULT_OUTPUT_ROOT
    diagnostics_dir = output_root / "diagnostics"
    image_dir = output_root / "model_inputs" / class_label

    manifest_path = diagnostics_dir / f"{iq_path.stem}_windows.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path} (pass overwrite=True to replace)"
        )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale in diagnostics_dir.glob(f"{iq_path.stem}_event*_window01.npz"):
            stale.unlink()
        for stale in image_dir.glob(f"{iq_path.stem}_event*_window01.png"):
            stale.unlink()

    start_time = time.monotonic()
    iq = load_iq(iq_path)
    psd_band, frequency_band, times = compute_record_stft(iq)
    events, active, noise_psd, score_db = detect_events(psd_band, iq.size)
    taps = build_filter_taps()

    entries = []
    skipped = []
    for number, event in enumerate(events, start=1):
        try:
            center_hz = estimate_center_hz(
                psd_band, frequency_band, active, event, noise_psd
            )
            values, start, stop, context_start, context_stop = extract_event(
                iq, event, center_hz, taps
            )
        except ValueError as error:
            # 低信噪比下检测段可能碎成短于 65536 点的片段：跳过该段，继续其余事件
            skipped.append(
                {
                    "event_number": int(number),
                    "event_start_sample": int(event["start_sample"]),
                    "event_stop_sample": int(event["stop_sample"]),
                    "reason": str(error),
                }
            )
            continue
        psd_cut, frequency_hz, times_local = compute_window_psd(values)
        rgb = psd_to_rgb(psd_cut)

        name = f"{iq_path.stem}_event{number:02d}_window01"
        image_path = image_dir / f"{name}.png"
        numeric_path = diagnostics_dir / f"{name}.npz"

        Image.fromarray(rgb).save(image_path)
        np.savez_compressed(
            numeric_path,
            iq=values.astype(np.complex64),
            psd=psd_cut.astype(np.float32),
            frequencies_hz=frequency_hz,
            times_seconds=times_local,
            source_path=str(iq_path),
            start_sample=start,
            stop_sample=stop,
            context_start_sample=context_start,
            context_stop_sample=context_stop,
            center_shift_hz=center_hz,
            fs_hz=FS_HZ,
            nfft=NFFT,
            hop=HOP,
            filter_taps=taps,
            filter_delay_samples=(len(taps) - 1) // 2,
            gray_db_range=np.array([GRAY_LOW_DB, GRAY_HIGH_DB]),
            image_adaptation=IMAGE_ADAPTATION,
        )

        entries.append(
            {
                "window_id": name,
                "event_number": int(number),
                "event_start_sample": int(event["start_sample"]),
                "event_stop_sample": int(event["stop_sample"]),
                "start_sample": int(start),
                "stop_sample": int(stop),
                "center_shift_hz": center_hz,
                "image_path": str(image_path),
                "matrix_path": str(numeric_path),
            }
        )

    manifest = {
        "source_group_id": f"{class_label}:{iq_path.stem}",
        "source_path": str(iq_path),
        "source_iq_sha256": hashlib.sha256(iq.tobytes()).hexdigest(),
        "source_format": "cf32_le",
        "sample_rate_hz": FS_HZ,
        "source_samples": int(iq.size),
        "class_label": class_label,
        "label_source": LABEL_SOURCE,
        "window_policy": WINDOW_POLICY,
        "detector": {
            "threshold_db": THRESHOLD_DB,
            "noise_psd": float(noise_psd),
            "minimum_seconds": MINIMUM_SECONDS,
            "merge_gap_seconds": MERGE_GAP_SECONDS,
        },
        "processing": {
            "nfft": NFFT,
            "hop": HOP,
            "window": "periodic Hann",
            "frequency_band_hz": [BAND_LOW_HZ, BAND_HIGH_HZ],
            "gray_db_range": [GRAY_LOW_DB, GRAY_HIGH_DB],
            "frequency_order": "high frequency at top",
            "observed_time_columns": 255,
            "image_adaptation": "copy last time column",
            "image_shape": [256, 256, 3],
            "dc_calibration": "not applied; source DC set to zero",
        },
        "windows": entries,
        "skipped_events": skipped,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if make_plots:
        _save_diagnostic_plots(
            iq_path.stem,
            iq,
            times,
            frequency_band,
            psd_band,
            score_db,
            events,
            diagnostics_dir,
        )

    return {
        "record": iq_path.stem,
        "class_label": class_label,
        "source_path": str(iq_path),
        "detected_events": len(events),
        "exported_images": len(entries),
        "skipped_events": skipped,
        "noise_psd": float(noise_psd),
        "elapsed_seconds": time.monotonic() - start_time,
        "events": entries,
    }
