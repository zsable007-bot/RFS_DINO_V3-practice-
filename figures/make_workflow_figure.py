"""生成工作汇报用流程图（顶刊风格：黑白灰、细线、留白）。

数据生成 → 时频处理/建集 → 特征迁移训练 → 评测与泛化验证
第一栏描述信号的固有特性，使第二栏的处理步骤在逻辑上自洽可理解。

运行：
    F:\\Program\\Miniforge3\\envs\\dinov3_11\\python.exe figures\\make_workflow_figure.py
输出：
    figures/workflow.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_DIR / "figures" / "workflow.png"

INK = "#111111"
GRAY = "#555555"
LIGHT = "#EDEDED"
RULE = "#999999"

STAGES = [
    {
        "title": "一、数据生成",
        "tool": "GNU Radio 仿真",
        "lines": [
            "CW / 2FSK / LFM 复数基带信号",
            "形态：单频线 / 双频交替 / 斜线扫频",
            "突发经包络门控，段间为纯噪声",
            "2 s 内 6 段突发，长短位置不一",
            "信道含 5 kHz 频偏与 AWGN",
            "噪声、种子等条件可控（可复现）",
        ],
        "product": "产物：原始 IQ 文件",
    },
    {
        "title": "二、时频处理与数据集构建",
        "tool": "numpy / Pillow",
        "lines": [
            "STFT：IQ → 时频图",
            "突发检测（噪声估计 + 门限）",
            "中心对齐（消除频偏）与切窗",
            "灰度成像",
            "按 IQ 记录分组划分数据集",
        ],
        "product": "产物：时频图数据集",
    },
    {
        "title": "三、特征迁移训练",
        "tool": "冻结 DINOv3 + 分类头",
        "lines": [
            "冻结 DINOv3 骨干（仅提特征）",
            "训练轻量分类头",
            "按验证准确率选最优",
        ],
        "product": "产物：分类头权重",
    },
    {
        "title": "四、评测与泛化验证",
        "tool": "批量推理与统计",
        "lines": [
            "逐 SNR 评测（检出/分类分开）",
            "新种子留出集：只测不训",
            "输出性能报告",
        ],
        "product": "产物：性能报告",
    },
]


def main():
    figure, ax = plt.subplots(figsize=(14.5, 6.2), dpi=240)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    width, gap = 3.3, 0.45
    x = 0.25
    y, height = 0.45, 5.35

    for index, stage in enumerate(STAGES):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="square,pad=0.0",
            linewidth=1.0,
            edgecolor=INK,
            facecolor="white",
        )
        ax.add_patch(patch)

        band_height = 0.62
        ax.add_patch(
            Rectangle(
                (x, y + height - band_height),
                width,
                band_height,
                linewidth=0,
                facecolor=LIGHT,
            )
        )
        ax.plot(
            [x, x + width],
            [y + height - band_height, y + height - band_height],
            color=INK,
            linewidth=0.8,
        )
        ax.text(
            x + width / 2,
            y + height - band_height / 2,
            stage["title"],
            ha="center",
            va="center",
            fontsize=14.5,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + width / 2,
            y + height - band_height - 0.34,
            stage["tool"],
            ha="center",
            va="center",
            fontsize=11,
            color=GRAY,
            style="italic",
        )

        line_y = y + height - band_height - 0.88
        for line in stage["lines"]:
            ax.text(
                x + 0.26,
                line_y,
                "· " + line,
                ha="left",
                va="center",
                fontsize=12,
                color=INK,
            )
            line_y -= 0.60

        ax.plot(
            [x + 0.25, x + width - 0.25],
            [y + 0.68, y + 0.68],
            color=RULE,
            linewidth=0.6,
        )
        ax.text(
            x + width / 2,
            y + 0.36,
            stage["product"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

        if index < len(STAGES) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.05, y + height / 2),
                (x + width + gap - 0.05, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.1,
                color=INK,
            )
            ax.add_patch(arrow)

        x += width + gap

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
