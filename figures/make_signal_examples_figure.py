"""生成三类信号时频图九宫格（列 = 类别，行 = 三条不同记录的样本）。

选图：各列取 0 dB（中等强噪声）、噪声种子 101/102/103 对应记录的 event01 窗口图，
用于在报告中直观展示 CW / 2FSK / LFM 的时频结构差异及其在噪声下的形态。

运行：
    F:\\Program\\Miniforge3\\envs\\dinov3_11\\python.exe figures\\make_signal_examples_figure.py
输出：
    figures/signal_examples.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]

PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_DIR / "output" / "model_inputs"
OUTPUT = PROJECT_DIR / "figures" / "signal_examples.png"

CLASSES = ["CW", "2FSK", "LFM"]
RECORDS = [14, 15, 16]  # 0 dB / 种子 101、102、103


def main():
    figure, axes = plt.subplots(
        3, 3, figsize=(7.4, 8.2), dpi=240, gridspec_kw={"wspace": 0.06, "hspace": 0.06}
    )

    for column, class_label in enumerate(CLASSES):
        for row, record_number in enumerate(RECORDS):
            name = f"{class_label}_{record_number:04d}_event01_window01.png"
            image_path = IMAGES_ROOT / class_label / name
            ax = axes[row][column]
            with Image.open(image_path) as source:
                ax.imshow(source.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#888888")
                spine.set_linewidth(0.6)
            if row == 0:
                ax.set_title(class_label, fontsize=13, fontweight="bold", pad=6)
            if column == 0:
                ax.text(
                    -0.10,
                    0.5,
                    f"样本 {row + 1}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    color="#555555",
                )

    figure.subplots_adjust(left=0.07, right=0.99, top=0.955, bottom=0.048)
    figure.text(
        0.5,
        0.012,
        "SNR = 0 dB",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
