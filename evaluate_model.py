"""对 output/model_inputs 下的全部图片做批量推理评测（复现 RFS_DINO_V3.ipynb 的推理）。

模型与预处理与 RFS_DINO_V3.ipynb 完全一致：
    - 骨干：timm.create_model(checkpoint["model_name"], pretrained=True, num_classes=0)，冻结；
    - 分类头：make_head(feature_dim, num_classes)（Linear→ReLU→Dropout→Linear）；
    - 变换：Resize((256,256)) → ToTensor → Normalize（无中心裁剪、无增强）；
    - 类别映射取自 checkpoint["class_to_idx"]。

统计维度（RFS：radio frequency spectrum）：
    - 逐图片预测明细（含真实类、预测类、三分类概率、所属记录、SNR）；
    - 逐（类别 × SNR）汇总：期望事件数、检出事件数、导出图片数、正确数、
      条件准确率（正确/导出）、端到端正确率（正确/期望）；
    - 混淆矩阵。

运行环境：dinov3_11（torch + timm + torchvision）：
    F:\\Program\\Miniforge3\\envs\\dinov3_11\\python.exe evaluate_model.py

输出：
    output/eval/<run名>_predictions.csv     逐图预测明细
    output/eval/<run名>_summary.csv         逐条件汇总
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DEFAULT_CHECKPOINT = PROJECT_DIR / "runs" / "20261010_084926_ccc0f3" / "best_head.pth"
EXPECTED_EVENTS_PER_RECORD = 6  # 0.1s 时隙、模式 [0,1,0,0,1,0,0,1,0,0]、2 秒记录的结构值


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--images-root", type=Path, default=PROJECT_DIR / "output" / "model_inputs"
    )
    parser.add_argument(
        "--records-csv", type=Path, default=PROJECT_DIR / "data" / "records.csv"
    )
    parser.add_argument(
        "--processing-csv",
        type=Path,
        default=PROJECT_DIR / "output" / "diagnostics" / "records_processing.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "output" / "eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="每类最多评测的图片数（调试用）"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser


def make_head(feature_dim, num_classes):
    return nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes),
    )


def make_rf_transform(config):
    height, width = config["input_size"][-2:]
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    import timm

    backbone = timm.create_model(
        checkpoint["model_name"], pretrained=True, num_classes=0
    ).to(device)
    backbone.requires_grad_(False)
    head = make_head(checkpoint["feature_dim"], checkpoint["num_classes"]).to(device)
    head.load_state_dict(checkpoint["classifier_state_dict"])
    model = nn.Sequential(backbone, head)
    model.eval()
    transform = make_rf_transform(checkpoint["data_config"])
    return checkpoint, model, transform


def load_records(records_csv):
    """记录名 -> (class_label, snr_db, noise_seed, bit_seed)；仅含噪声实验记录。"""
    records = {}
    if not records_csv.exists():
        return records
    with records_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["file"]).stem
            records[stem] = row
    return records


def load_detection_counts(processing_csv):
    """记录名 -> {"detected": 检出事件数, "class_label": 类别}（同一记录多行取最大值）。"""
    detected = {}
    if not processing_csv.exists():
        return detected
    with processing_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = int(row["detected_events"] or 0)
            record = row["record"]
            previous = detected.get(record, {"detected": 0, "class_label": row["class_label"]})
            detected[record] = {
                "detected": max(previous["detected"], value),
                "class_label": row["class_label"],
            }
    return detected


def collect_images(images_root, class_to_idx, limit=None):
    """返回 [(image_path, true_index, class_label)]，按类、文件名排序。"""
    entries = []
    for class_label in sorted(class_to_idx):
        class_dir = images_root / class_label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"找不到类别目录：{class_dir}")
        images = sorted(class_dir.glob("*.png"))
        if limit is not None:
            images = images[:limit]
        for image in images:
            entries.append((image, class_to_idx[class_label], class_label))
    return entries


def infer(model, transform, entries, device, batch_size):
    """批量推理；返回逐图结果列表。"""
    results = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            tensors = []
            for image_path, _, _ in batch:
                with Image.open(image_path) as source:
                    tensors.append(transform(source.convert("RGB")))
            inputs = torch.stack(tensors).to(device)
            probabilities = model(inputs).softmax(dim=1).cpu()
            for (image_path, true_index, class_label), probs in zip(batch, probabilities):
                predicted = int(probs.argmax().item())
                results.append(
                    {
                        "image": image_path,
                        "record": image_path.stem.rsplit("_event", 1)[0],
                        "class_label": class_label,
                        "true_index": true_index,
                        "predicted_index": predicted,
                        "probabilities": probs.tolist(),
                    }
                )
            print(f"  {min(start + batch_size, len(entries))}/{len(entries)}", flush=True)
    return results


def summarize(results, records, detection_info, class_names, only_observed=False):
    """逐 (类别, SNR) 与总体汇总。

    记录全集来自 records.csv（sweep）与处理清单（含 0001 等），即使某条记录
    一张图都没切出来，也会计入分母（expected_events），不会被静默忽略；
    检出数在清单缺失时按该记录实际导出的图片数回退，不会跨记录重复累计。
    only_observed=True（--limit 调试）时只统计有结果的记录。
    """
    exported_by_record = Counter(result["record"] for result in results)
    results_by_record = defaultdict(list)
    for result in results:
        results_by_record[result["record"]].append(result)

    def record_key(record, default_class=None):
        if record in records:
            info = records[record]
            return (info["class_label"], float(info["snr_db"]), "sweep")
        if record in detection_info:
            return (detection_info[record]["class_label"], None, "other")
        return (default_class, None, "other")

    key_of = {}
    for record in records:
        key_of[record] = record_key(record)
    for record in detection_info:
        key_of.setdefault(record, record_key(record))
    for result in results:
        key_of.setdefault(
            result["record"], record_key(result["record"], result["class_label"])
        )
    if only_observed:
        key_of = {record: key for record, key in key_of.items() if record in results_by_record}

    record_names_by_key = defaultdict(list)
    for record, key in key_of.items():
        if key[0] in class_names:
            record_names_by_key[key].append(record)

    def make_row(label, snr, dataset, record_names):
        record_names = sorted(record_names)
        group = [
            result for name in record_names for result in results_by_record[name]
        ]
        exported = len(group)
        correct = sum(
            1 for result in group if result["true_index"] == result["predicted_index"]
        )
        expected = EXPECTED_EVENTS_PER_RECORD * len(record_names)
        detected_total = sum(
            detection_info.get(name, {}).get(
                "detected", exported_by_record.get(name, 0)
            )
            for name in record_names
        )
        return {
            "class_label": label,
            "snr_db": "" if snr is None else snr,
            "dataset": dataset,
            "records": len(record_names),
            "expected_events": expected,
            "detected_events": detected_total,
            "exported_images": exported,
            "correct": correct,
            "accuracy_conditional": round(correct / exported, 4) if exported else "",
            "accuracy_end_to_end": round(correct / expected, 4) if expected else "",
        }

    rows = []
    for (label, snr, dataset), record_names in sorted(
        record_names_by_key.items(),
        key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or 0),
    ):
        rows.append(make_row(label, snr, dataset, record_names))

    for label in sorted(class_names):
        record_names = [record for record, key in key_of.items() if key[0] == label]
        if record_names:
            rows.append(make_row(label, None, "ALL", record_names))
    all_records = [record for record, key in key_of.items() if key[0] in class_names]
    if all_records:
        rows.append(make_row("ALL", None, "ALL", all_records))
    sweep_records = [
        record
        for record, key in key_of.items()
        if key[0] in class_names and key[2] == "sweep"
    ]
    if sweep_records:
        rows.append(make_row("ALL", None, "sweep", sweep_records))
    return rows


def confusion_matrix(results, class_names):
    size = len(class_names)
    matrix = [[0] * size for _ in range(size)]
    for result in results:
        matrix[result["true_index"]][result["predicted_index"]] += 1
    return matrix


def print_summary(rows, matrix, class_names):
    header = (
        f"{'class':6} {'snr':>5} {'dataset':>7} {'rec':>4} {'exp':>4} "
        f"{'det':>4} {'exp_img':>7} {'ok':>4} {'cond_acc':>9} {'e2e':>7}"
    )
    print(header)
    for row in rows:
        cond = (
            f"{100 * float(row['accuracy_conditional']):.1f}%"
            if row["accuracy_conditional"] != ""
            else "-"
        )
        e2e = (
            f"{100 * float(row['accuracy_end_to_end']):.1f}%"
            if row["accuracy_end_to_end"] != ""
            else "-"
        )
        print(
            f"{row['class_label']:6} {str(row['snr_db']):>5} {row['dataset']:>7} "
            f"{row['records']:>4} {row['expected_events']:>4} "
            f"{row['detected_events']:>4} {row['exported_images']:>7} "
            f"{row['correct']:>4} {cond:>9} {e2e:>7}"
        )

    print()
    print("混淆矩阵（行=真实，列=预测）:", ", ".join(class_names))
    for label, line in zip(class_names, matrix):
        print(f"  {label:6} {line}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"找不到 checkpoint：{checkpoint_path}")
    run_name = checkpoint_path.parent.name

    print(f"Checkpoint：{checkpoint_path}")
    checkpoint, model, transform = load_model(checkpoint_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(
        f"骨干：{checkpoint['model_name']} | 类别：{class_to_idx} | "
        f"最佳轮次：{checkpoint['best_epoch']} | 验证准确率：{checkpoint['best_val_acc']:.2f}%"
    )
    print(f"设备：{device}")

    records = load_records(args.records_csv)
    detection_info = load_detection_counts(args.processing_csv)

    entries = collect_images(args.images_root, class_to_idx, args.limit)
    if not entries:
        raise SystemExit(f"没有找到图片：{args.images_root}")
    print(f"图片总数：{len(entries)}（来源：{args.images_root}）")
    results = infer(model, transform, entries, device, args.batch_size)

    rows = summarize(
        results,
        records,
        detection_info,
        class_names,
        only_observed=args.limit is not None,
    )
    matrix = confusion_matrix(results, class_names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / f"{run_name}_predictions.csv"
    summary_path = args.out_dir / f"{run_name}_summary.csv"

    prediction_fields = [
        "image",
        "record",
        "class_label",
        "snr_db",
        "noise_seed",
        "dataset",
        "predicted",
        "correct",
        "confidence",
    ] + [f"prob_{name}" for name in class_names]
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for result in results:
            record_info = records.get(result["record"])
            probabilities = result["probabilities"]
            writer.writerow(
                {
                    "image": str(result["image"]),
                    "record": result["record"],
                    "class_label": result["class_label"],
                    "snr_db": "" if record_info is None else record_info["snr_db"],
                    "noise_seed": "" if record_info is None else record_info["noise_seed"],
                    "dataset": "sweep" if record_info is not None else "other",
                    "predicted": idx_to_class[result["predicted_index"]],
                    "correct": result["true_index"] == result["predicted_index"],
                    "confidence": round(max(probabilities), 6),
                    **{
                        f"prob_{name}": round(probabilities[index], 6)
                        for index, name in enumerate(class_names)
                    },
                }
            )

    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print_summary(rows, matrix, class_names)
    print()
    print(f"逐图预测：{predictions_path}")
    print(f"逐条件汇总：{summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
