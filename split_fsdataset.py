from pathlib import Path
from collections import defaultdict, Counter
import json
import math
import random
import re
import shutil
import sys


for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


PROJECT_DIR = Path(r"D:\workspace\preparation\dinov3_11").resolve()
SOURCE_DIR = PROJECT_DIR / "output" / "model_inputs"
OUTPUT_DIR = PROJECT_DIR / "fsdataset"

RATIOS = (0.6, 0.2, 0.2)  # train、val、test
SPLITS = ("train", "val", "test")
SEED = 42
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RECORD_PATTERN = re.compile(r"(.+)_event\d+_window\d+$")


def split_counts(total):
    """按比例取整，保证三个子集都有数据。"""
    if total < 3:
        raise ValueError("每个类别至少需要 3 个划分单位。")

    quotas = [total * ratio for ratio in RATIOS]
    counts = [math.floor(value) for value in quotas]

    order = sorted(
        range(3),
        key=lambda i: (-(quotas[i] - counts[i]), i),
    )
    for i in order[:total - sum(counts)]:
        counts[i] += 1

    for i in range(3):
        if counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[i] = 1

    return counts


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片来源目录：{SOURCE_DIR}")

    # 自动发现类别，并递归收集图片。
    classes = {}
    record_groups = {}
    all_names_matched = True

    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue

        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if len(images) < 3:
            raise ValueError(
                f"{folder.name} 只有 {len(images)} 张图片，至少需要 3 张。"
            )

        classes[folder.name] = images
        groups = defaultdict(list)

        for image in images:
            match = RECORD_PATTERN.fullmatch(image.stem)
            if match:
                groups[match.group(1)].append(image)
            else:
                all_names_matched = False

        record_groups[folder.name] = groups

    if not classes:
        raise ValueError(f"没有找到类别图片：{SOURCE_DIR}")

    # 当前每类只有 0001：自动按图片划分。
    # 每类都有至少 3 份 IQ 记录后：自动按记录划分，同源图片保持在一起。
    use_records = (
        all_names_matched
        and all(len(groups) >= 3 for groups in record_groups.values())
    )
    mode = "record" if use_records else "image"

    # 先生成完整计划，再重建输出目录。
    plan = []

    for label, images in classes.items():
        class_dir = SOURCE_DIR / label

        if use_records:
            units = record_groups[label]
        else:
            units = {
                image.relative_to(class_dir).as_posix(): [image]
                for image in images
            }

        keys = sorted(units)
        random.Random(f"{SEED}:{label}").shuffle(keys)
        counts = split_counts(len(keys))

        offset = 0
        for split, count in zip(SPLITS, counts):
            for key in keys[offset:offset + count]:
                for image in units[key]:
                    relative_image = image.relative_to(class_dir)
                    match = RECORD_PATTERN.fullmatch(image.stem)
                    group = match.group(1) if match else relative_image.as_posix()

                    plan.append({
                        "class_label": label,
                        "split": split,
                        "source_group_id": f"{label}:{group}",
                        "source_file": str(image),
                        "output_file": (
                            Path(split) / label / relative_image
                        ).as_posix(),
                    })
            offset += count

    # 只允许重建指定项目下的 fsdataset，防止删除指向其他位置的目录。
    expected_output = PROJECT_DIR / "fsdataset"
    if OUTPUT_DIR.resolve() != expected_output:
        raise ValueError("fsdataset 指向其他位置，停止覆盖。")

    source = SOURCE_DIR.resolve()
    target = OUTPUT_DIR.resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("来源目录与输出目录不能相同或互相包含。")

    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
        raise ValueError(f"输出路径不是文件夹：{OUTPUT_DIR}")

    # 每次运行全部覆盖，不备份；原始图片不删除。
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        for label in classes:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    for entry in plan:
        destination = OUTPUT_DIR / entry["output_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source_file"], destination)

    manifest = {
        "seed": SEED,
        "mode": mode,
        "ratios": dict(zip(SPLITS, RATIOS)),
        "class_to_idx": {
            label: i for i, label in enumerate(sorted(classes))
        },
        "images": plan,
    }
    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = Counter(
        (entry["class_label"], entry["split"]) for entry in plan
    )

    print(f"划分模式：{'按 IQ 记录' if use_records else '按图片'}")
    for label in sorted(classes):
        print(
            f"{label}: "
            f"train={summary[label, 'train']}, "
            f"val={summary[label, 'val']}, "
            f"test={summary[label, 'test']}"
        )

    print(f"共复制 {len(plan)} 张图片到：{OUTPUT_DIR}")
    if not use_records:
        print("当前按图片划分用于流程验证，同一 IQ 记录可能跨训练、验证和测试集。")


if __name__ == "__main__":
    main()