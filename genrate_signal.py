"""按"噪声实验"方案批量生成 IQ 数据（3 类 × 9 档 SNR × 3 组种子 = 81 条）。

方案（详见对话确认的噪声轮实验）：
    类别：CW、2FSK、LFM（波形参数全部沿用库默认值）
    SNR： [20, 15, 10, 5, 0, -3, -6, -9, -12] dB
    种子配对：(noise_seed, 2FSK bit_seed) = (101, 11)、(102, 12)、(103, 13)
    记录长度：2 秒；全速生成（非实时）

运行环境必须使用 GNU Radio 的 Python（系统 python 与 dinov3_11 环境都不行）：
    D:\\Programdata\\radioconda\\python.exe genrate_signal.py

输出：
    data/<类>/<类>_<NNNN>.cf32         每类编号从现有最大编号 +1 自动开始
    data/<类>/<类>_<NNNN>.cf32.json    单条记录参数侧车（含 sha256）
    data/records.csv                   本轮汇总（每次运行会覆盖，只含当次记录）

固定参数（与现有数据一致，便于横向对比）：
    采样率 1 MHz、信号电平 -27 dBFS、中心频率 50 kHz、接收频偏 5 kHz、
    接收增益 0 dB、IQ 不平衡与 DC 全为 0、ADC 12 bit、
    2FSK 符号率 1000 符号/秒、频偏 ±30 kHz；LFM 带宽 80 kHz、周期 0.01 秒。
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from signal_source import generate_batch
except ImportError as error:
    raise SystemExit(
        "无法导入 signal_source（需要 GNU Radio 环境）。\n"
        "请使用 Radioconda 的 Python 运行：\n"
        "    D:\\Programdata\\radioconda\\python.exe genrate_signal.py\n"
        f"原始错误：{error}"
    )

# ---------------- 本轮实验配置（需要调整时只改这里） ----------------
CLASSES = ("CW", "2FSK", "LFM")
SNR_DB = (20, 15, 10, 5, 0, -3, -6, -9, -12)
SEEDS = (101, 102, 103)
BIT_SEEDS = (11, 12, 13)  # 与 SEEDS 一一配对；只作用于 2FSK
CAPTURE_SECONDS = 2.0
OUTPUT_DIR = PROJECT_DIR / "data"
# -------------------------------------------------------------------


def build_argv():
    return [
        "--classes", *CLASSES,
        "--snr", *[str(value) for value in SNR_DB],
        "--seeds", *[str(value) for value in SEEDS],
        "--bit-seeds", *[str(value) for value in BIT_SEEDS],
        "--seconds", str(CAPTURE_SECONDS),
        "--out", str(OUTPUT_DIR),
    ]


def main():
    expected = len(CLASSES) * len(SNR_DB) * len(SEEDS)
    print(
        f"噪声实验：{len(CLASSES)} 类 × {len(SNR_DB)} 档 SNR × "
        f"{len(SEEDS)} 组种子 = {expected} 条记录"
    )
    print(f"输出目录：{OUTPUT_DIR}")
    return generate_batch.main(build_argv())


if __name__ == "__main__":
    sys.exit(main())
