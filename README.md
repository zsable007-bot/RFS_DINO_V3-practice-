# RFS DINOv3 — 无线电频谱信号识别

用 **DINOv3 冻结骨干 + 轻量分类头**识别三类仿真无线电信号：

| 类别 | 信号 | 时频图形态 |
|---|---|---|
| CW | 单频连续波突发 | 水平亮线 |
| 2FSK | 二进制频移键控（1000 符号/s，±30 kHz） | 两条水平线交替 |
| LFM | 线性调频（80 kHz 带宽，10 ms 周期） | 锯齿状斜线 |

## 处理流程

```
GNU Radio 生成/采集 IQ（1 MHz 采样，2 s，cf32）
  → STFT（512 点 Hann，hop 256，±250 kHz）
  → 噪声估计 + 10 dB 门限事件检测
  → 能量中心估计 + 混频到 0 Hz + FIR 低通
  → 居中截取 65536 点 → 256×256 灰度时频图
  → DINOv3 冻结骨干（vit_small_patch16_dinov3.lvd1689m，384 维）
  → 分类头（Linear→ReLU→Dropout→Linear，只训练此头）
```

## 主要结果

**噪声实验**：3 类 × 9 档 SNR（−12 ~ 20 dB）× 3 噪声种子 = 81 条记录 / 486 个事件；
另有一批**全新种子（201–203）的独立留出集**用于泛化验证（只测不训）。

- **检测**：−12 ~ 20 dB 全部检出（10 dB 门限）；−12 dB 下偶发突发碎裂
  （每轮 1~8 个碎片不足 65.5 ms 窗口，属检测端损失）
- **基线模型**（仅 20 dB 数据训练）：2FSK 在 ≥5 dB 约 94~100%、0 dB 94.4%、
  −3 dB 33.3%、−6 dB 5.6%、−9/−12 dB 为 0%；sweep 端到端 86.0%
- **混合 SNR 重训分类头后**：
  - 划分测试集（5 条记录/类）：90/90（100%）
  - **独立留出集（全新噪声与 2FSK 比特种子，从未参与训练）：483/483 张图 100% 正确，
    端到端 99.4%**，剩余损失全部来自 −12 dB 的检测碎片
  - 全量诊断值（503 张，含训练/验证记录）：100%，sweep 端到端 99.8%

结论：瓶颈在分类头未见低 SNR 输入，而非骨干特征或检测器；骨干可保持冻结。
新种子独立留出集上分类准确率 100%，低 SNR 泛化成立；残余误差集中在 −12 dB 的检测切窗环节。

## 目录结构

```
signal_source/           GNU Radio 信号生成库（cw / fsk2 / lfm，可编程调用）
spectrum_processing/     检测 + 切图管线（process_record / process_batch）
RFS_DINO_V3.py(.ipynb)   训练分类头（冻结骨干，按验证准确率选最优）
evaluate_model.py        批量推理评测（逐 SNR 检出率 / 准确率 / 混淆矩阵）
split_fsdataset.py       按 IQ 记录分层划分 train/val/test（同记录不跨集）
genrate_signal.py        噪声实验批量生成入口（81 条参数网格）
GNU_Radio/               原始 GNU Radio 流程图（.grc 与生成 .py）
tests/                   回归测试（生成逐字节一致性、处理逐位一致性）
docs/                    设计与实施文档
```

## 运行环境（两个，互不通用）

| 用途 | 环境 |
|---|---|
| 信号生成（GNU Radio 3.10.12） | Radioconda：`D:\Programdata\radioconda\python.exe` |
| 处理 / 训练 / 评测 | conda 环境 `dinov3_11`（Python 3.12，torch 2.4.1+cu124，timm 1.0.24，RTX 4060） |

## 常用命令

```powershell
# 1. 生成噪声实验数据（81 条，写入 data/）
& "D:\Programdata\radioconda\python.exe" genrate_signal.py

# 2. 检测 + 切图（写入 output/model_inputs 与 output/diagnostics）
& "F:\Program\Miniforge3\envs\dinov3_11\python.exe" spectrum_processing\process_batch.py --classes CW 2FSK LFM

# 3. 划分数据集（按 IQ 记录，无泄漏）
& "F:\Program\Miniforge3\envs\dinov3_11\python.exe" split_fsdataset.py

# 4. 训练分类头（输出 runs/<时间戳>_<id>/best_head.pth）
& "F:\Program\Miniforge3\envs\dinov3_11\python.exe" RFS_DINO_V3.py

# 5. 评测模型（对 output/model_inputs 全量图片，按 SNR 汇总）
& "F:\Program\Miniforge3\envs\dinov3_11\python.exe" evaluate_model.py --checkpoint runs\<run>\best_head.pth
```

## 数据（不入库，可再生成）

| 路径 | 内容 | 说明 |
|---|---|---|
| `data/<类>/*.cf32` | 84 条 2 秒 IQ（16 MB/条） | `genrate_signal.py` 重新生成，同种子逐字节一致 |
| `output/model_inputs/` | 503 张 256×256 时频图 | `process_batch.py` 生成 |
| `output/diagnostics/` | 每记录 windows.json、事件 npz、处理汇总 csv | 含检出与参数溯源 |
| `output/eval/` | 逐图预测明细与逐条件汇总 | `evaluate_model.py` 生成 |
| `fsdataset/` | train(305)/val(108)/test(90) | `split_fsdataset.py` 按记录划分 |

## 可复现性

- 生成：同参数同噪声种子下与历史数据**逐字节一致**（sha256）
- 处理：同环境下与历史 notebook 产物**逐位/逐像素一致**（`tests/` 回归保障）
- 划分：固定随机种子 42，确定性；同一条 IQ 的全部事件不会跨子集
