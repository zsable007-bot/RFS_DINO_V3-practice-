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

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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
