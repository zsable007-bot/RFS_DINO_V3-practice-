"""批量处理 IQ 记录：事件检测 + 切图（CW/2FSK/LFM 共用同一管线）。

用法示例：
    python spectrum_processing/process_batch.py --classes CW 2FSK LFM
    python spectrum_processing/process_batch.py --classes CW --limit 1   # 冒烟

运行环境：dinov3_11 环境的 Python（numpy + Pillow；--make-plots 用到 matplotlib）：
    F:\\Program\\Miniforge3\\envs\\dinov3_11\\python.exe
注意：numpy 版本不同会产生末位浮点差异（像素不受影响），本库以 dinov3_11
环境的既有产物为逐位基准（见 tests/test_spectrum_processing.py）。

行为：
    - 遍历 <data-dir>/<类>/*.cf32，逐条调用 process_record；
    - 已有 <record>_windows.json 的记录默认跳过（--overwrite 强制重算）；
    - 单条失败不中断整批，错误写入汇总表；
    - 汇总表：<output-root>/diagnostics/records_processing.csv；每次运行都从
      范围内所有记录的 windows.json 整体重建（已跳过者也计入），因此可增量重跑；
      每条检出事件一行（成功导出或短碎片跳过），零事件或整条失败各占一行。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from spectrum_processing import process_record  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CLASSES = ("CW", "2FSK", "LFM")

FIELDNAMES = [
    "record",
    "class_label",
    "detected_events",
    "event_number",
    "window_id",
    "event_start_sample",
    "event_stop_sample",
    "start_sample",
    "stop_sample",
    "center_shift_hz",
    "image_path",
    "matrix_path",
    "error",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "output")
    parser.add_argument("--limit", type=int, default=None, help="每类最多处理的记录数")
    parser.add_argument("--make-plots", action="store_true", help="额外生成整轨诊断图")
    parser.add_argument("--overwrite", action="store_true", help="已有结果时强制重算")
    return parser


def rows_from_manifest(manifest, class_label):
    """把一条记录的 windows.json 展开为汇总行（导出事件 + 被跳过的短碎片）。"""
    record = Path(manifest["source_path"]).stem
    skipped = manifest.get("skipped_events", [])
    detected = len(manifest["windows"]) + len(skipped)
    rows = []
    for event in manifest["windows"]:
        rows.append(
            {
                "record": record,
                "class_label": class_label,
                "detected_events": detected,
                "event_number": event["event_number"],
                "window_id": event["window_id"],
                "event_start_sample": event["event_start_sample"],
                "event_stop_sample": event["event_stop_sample"],
                "start_sample": event["start_sample"],
                "stop_sample": event["stop_sample"],
                "center_shift_hz": event["center_shift_hz"],
                "image_path": event["image_path"],
                "matrix_path": event["matrix_path"],
                "error": "",
            }
        )
    for event in skipped:
        rows.append(
            {
                "record": record,
                "class_label": class_label,
                "detected_events": detected,
                "event_number": event["event_number"],
                "event_start_sample": event["event_start_sample"],
                "event_stop_sample": event["event_stop_sample"],
                "error": f"skipped: {event['reason']}",
            }
        )
    if detected == 0:
        rows.append(
            {
                "record": record,
                "class_label": class_label,
                "detected_events": 0,
                "error": "",
            }
        )
    return rows


def run(args):
    rows = []
    for class_label in args.classes:
        class_dir = args.data_dir / class_label
        records = sorted(class_dir.glob("*.cf32"))
        if args.limit is not None:
            records = records[: args.limit]
        print(f"== {class_label}: {len(records)} records ==")
        for iq_path in records:
            manifest_path = (
                args.output_root / "diagnostics" / f"{iq_path.stem}_windows.json"
            )
            try:
                summary = process_record(
                    iq_path,
                    class_label=class_label,
                    output_root=args.output_root,
                    make_plots=args.make_plots,
                    overwrite=args.overwrite,
                )
            except FileExistsError as error:
                print(f"SKIP {iq_path.name}: {error}")
            except Exception as error:  # noqa: BLE001 - 批量容错
                print(f"FAIL {iq_path.name}: {error}")
                rows.append(
                    {
                        "record": iq_path.stem,
                        "class_label": class_label,
                        "detected_events": 0,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            else:
                print(
                    f"DONE {iq_path.name} events={summary['detected_events']} "
                    f"images={summary['exported_images']} "
                    f"skipped={len(summary['skipped_events'])}"
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows.extend(rows_from_manifest(manifest, class_label))

    csv_path = args.output_root / "diagnostics" / "records_processing.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rows: {len(rows)}; summary: {csv_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
