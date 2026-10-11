# signal_source 信号生成库设计

日期：2026-10-10
状态：已获用户确认（方案 A）
工作区：D:\workspace\preparation\dinov3_11

## 1. 背景与目标

现状：三类仿真信号（CW、2FSK、LFM）由 GNU Radio 生成的 `GNU_Radio/*_capture.py` 产生。这些文件是 GUI 程序（Qt 窗口、qtgui 接收器），参数硬编码在 `__init__` 内，输出路径固定为 `data/<类>/<类>_0001.cf32`。批量实验（如 3 类 × 4 SNR × 3 种子）需要手工逐个改参数重跑，易错且不可编程遍历。

目标：

1. 在 `signal_source/` 下提供三个可被其他 Python 程序调用的模块（`cw.py`、`fsk2.py`、`lfm.py`），通过函数参数控制全部信号与信道参数。
2. 提供统一分发入口，批量脚本可按“类别 × 参数网格”遍历调用。
3. 重构必须**数据无损**：相同参数与种子下，新库生成的 cf32 与现有 `data/*/*_0001.cf32` 逐字节一致（sha256 相等）。
4. 纯 headless：不依赖 PyQt5，可无显示环境运行。

非目标：

- 不改动 IQ 之后的处理链路（STFT、检测、切窗、切图）。
- 不修改 `GNU_Radio/*.grc`；GRC 仍作为交互实验入口。
- 本轮不新增波形与信道损伤（多径 taps、衰落等留待后续扩展）。
- 不实现 GUI 版本。

## 2. 运行环境

- 必须使用 `D:\Programdata\radioconda\python.exe`（Python 3.12.9，GNU Radio 3.10.12.0）。
- 项目内 torch 环境（`F:\Program\Miniforge3\envs\dinov3_11`）没有 GNU Radio，不能运行本库。
- 依赖仅限 GNU Radio 自带模块与 Python 标准库（`hashlib`、`json`、`os`、`time`、`math`、`argparse`、`pathlib` 等），不引入 numpy。

## 3. 目录结构

```
signal_source/
  __init__.py          # 统一分发 generate() 与类别常量
  _common.py           # 公共链路构建、运行控制、参数校验、manifest
  cw.py                # CW 源段 + generate()
  fsk2.py              # 2FSK 源段 + generate()
  lfm.py               # LFM 源段 + generate()
  generate_batch.py    # 批量遍历 CLI 示例脚本
```

`GNU_Radio/` 下三个 `*_capture.py` 保持原样，作为 GRC 参照与回归基线，不移动、不删除。

## 4. 对外接口

### 4.1 三个模块统一签名

```python
# signal_source/cw.py
def generate(output_path, *, 公共参数..., overwrite=False) -> dict: ...
# signal_source/fsk2.py
def generate(output_path, *, 公共参数..., symbol_rate=1000, fsk_deviation_hz=30000,
             bit_seed=11, overwrite=False) -> dict: ...
# signal_source/lfm.py
def generate(output_path, *, 公共参数..., lfm_bandwidth_hz=80000, chirp_period=0.01,
             overwrite=False) -> dict: ...
```

公共参数（默认值 = 现有 GRC 变量值）：

| 参数 | 默认 | 对应 GRC 变量 |
|---|---|---|
| `samp_rate` | 1_000_000 | samp_rate |
| `center_hz` | 50_000 | center_hz |
| `signal_dbfs` | -27 | signal_dbfs（`signal_amp = 10**(signal_dbfs/20)`） |
| `snr_db` | 20 | snr_db（`noise_voltage = signal_amp * 10**(-snr_db/20)`） |
| `capture_seconds` | 2 | capture_seconds |
| `freq_offset_hz` | 5_000 | freq_offset_hz |
| `rx_gain_db` | 0 | rx_gain_db（`rx_gain = 10**(rx_gain_db/20)`） |
| `iq_mag_db` | 0 | iq_mag_db |
| `iq_phase_deg` | 0 | iq_phase_deg |
| `dc_i` / `dc_q` | 0 | dc_i / dc_q |
| `adc_bits` | 12 | adc_bits |
| `noise_seed` | 20261004 | channel_model noise_seed |
| `slot_pattern` | `(0,1,0,0,1,0,0,1,0,0)` | vector_source 值 |
| `slot_seconds` | 0.1 | slot_samples = samp_rate×slot_seconds |
| `ramp_seconds` | 0.002 | ramp_samples = samp_rate×ramp_seconds |
| `realtime` | True | True：建 throttle(samp_rate)；False：不建，全速生成 |
| `overwrite` | False | 已存在输出时的覆盖开关 |
| `write_manifest` | True | 是否写侧车 JSON |

派生量（内部计算，不进接口）：`signal_amp`、`noise_voltage`、`rx_gain`、`adc_scale = 2**(adc_bits-1)`、`adc_max = 1 - 1/adc_scale`、`capture_samples = int(samp_rate*capture_seconds)`、`slot_samples`、`ramp_samples`。

### 4.2 分发入口

```python
# signal_source/__init__.py
CLASSES = ("CW", "2FSK", "LFM")
def generate(class_name, output_path, **params) -> dict: ...
```

`class_name` 只接受 `"CW"`、`"2FSK"`、`"LFM"`，其余抛 `ValueError`。

### 4.3 返回值

`generate()` 返回 dict（供批处理汇总）：

```python
{
  "output_path": str,      # 绝对路径
  "class_label": str,      # "CW" / "2FSK" / "LFM"
  "samples": int,          # capture_samples
  "bytes": int,            # 文件大小
  "sha256": str,           # 输出文件 sha256
  "elapsed_seconds": float,
  "params": {...},         # 全部生效参数（含派生量）
}
```

## 5. 信号链（与现有生成文件逐块等价）

### 5.1 公共链（`_common.py` 构建）

```
包络：vector_source_f(slot_pattern, repeat=True)
      → blocks.repeat(float, slot_samples)
      → blocks.moving_average_ff(ramp_samples, 1/ramp_samples, 4000, 1)
      → blocks.float_to_complex(1)   # 仅接端口 0，虚部为 0
信号 × 包络：blocks.multiply_vcc()
      → channels.channel_model(noise_voltage=noise_voltage,
                               frequency_offset=freq_offset_hz/samp_rate,
                               epsilon=1.0, taps=[1+0j],
                               noise_seed=noise_seed, block_tags=False)
      → blocks.multiply_const_cc(rx_gain)
      → channels.iqbal_gen(iq_mag_db, iq_phase_deg, 1)
      → blocks.add_const_cc(complex(dc_i, dc_q))
      → blocks.complex_to_float(1)
      → analog.rail_ff(-1, adc_max) ×2（I、Q 各一）
      → blocks.float_to_short(1, adc_scale) ×2
      → blocks.short_to_float(1, adc_scale) ×2
      → blocks.float_to_complex(1)
      → blocks.throttle(gr.sizeof_gr_complex, samp_rate)   # 仅 realtime=True
      → blocks.head(gr.sizeof_gr_complex, capture_samples)
      → blocks.file_sink(gr.sizeof_gr_complex, output_path)
        file_sink.set_unbuffered(True)
```

### 5.2 各类源段（差异部分）

| 类 | 源链 |
|---|---|
| CW | `analog.sig_source_c(samp_rate, GR_COS_WAVE, center_hz, signal_amp, 0, 0)` → 公共链 |
| 2FSK | `digital.glfsr_source_b(15, True, 0, bit_seed)` → `blocks.uchar_to_float()` → `blocks.repeat(gr.sizeof_float, samples_per_symbol)` → `blocks.multiply_const_ff(2*fsk_deviation_hz)` → `blocks.add_const_ff(center_hz - fsk_deviation_hz)` → `analog.frequency_modulator_fc(2π/samp_rate)` → `blocks.multiply_const_cc(signal_amp)` → 公共链；`samples_per_symbol = int(samp_rate/symbol_rate)` |
| LFM | `analog.sig_source_f(samp_rate, GR_SAW_WAVE, 1/chirp_period, lfm_bandwidth_hz, center_hz - lfm_bandwidth_hz/2, -π)` → `analog.frequency_modulator_fc(2π/samp_rate)` → `blocks.multiply_const_cc(signal_amp)` → 公共链 |

现文件中仅供 GUI 的块（qtgui_*、keep_one_in_n、display_decim）不复建，不影响输出数据。

## 6. 运行控制与错误处理

运行流程：

1. 参数校验（见下），解析输出路径为绝对路径。
2. 若输出文件已存在且 `overwrite=False` → 抛 `FileExistsError`。
3. 组装流图，`tb.start()`。
4. 轮询文件大小：每 0.1 s 检查一次，直到 `size >= capture_samples*8` 或超时（`timeout = capture_seconds + 10` 秒，可内部常量）。超时 → `tb.stop(); tb.wait()` 后抛 `TimeoutError`（含当前文件大小信息）。
5. `tb.stop(); tb.wait()`。
6. 计算 sha256、字节数；`write_manifest=True` 时写 `<output_path>.json` 侧车；返回结果 dict。

参数校验（抛 `ValueError`，消息含参数名与当前值）：

- `samp_rate > 0`、`capture_seconds > 0`、`slot_seconds > 0`、`ramp_seconds > 0`
- `finite(snr_db)`、`finite(signal_dbfs)`、`finite(freq_offset_hz)`
- `2**(adc_bits-1) >= 2`（`adc_bits >= 2`）
- `0 <= len(slot_pattern)`，元素仅 0/1
- 2FSK：`symbol_rate > 0`、`0 < fsk_deviation_hz < samp_rate/2`
- LFM：`0 < lfm_bandwidth_hz < samp_rate`、`chirp_period > 0`
- 派生量：`capture_samples >= 1`

## 7. Manifest 侧车 JSON

`<output_path>.json`（如 `CW_0002.cf32.json`），字段：

```json
{
  "class_label": "2FSK",
  "generator": "signal_source",
  "gnuradio_version": "3.10.12.0",
  "source_format": "cf32_le",
  "sample_rate_hz": 1000000.0,
  "samples": 2000000,
  "duration_seconds": 2.0,
  "sha256": "…",
  "created": "2026-10-10T12:34:56",
  "params": { …全部生效参数，含 seeds 与派生量… }
}
```

## 8. 批量 CLI（`generate_batch.py`）

用途：一条命令生成参数网格数据（如 3 类 × 4 SNR × 3 记录 = 36 条）。

```
python generate_batch.py \
  --classes CW 2FSK LFM \
  --snr 20 10 5 0 \
  --seeds 101 102 103 \
  --bit-seeds 11 12 13 \
  --seconds 2 \
  --out data \
  --start-index auto
```

- 命名规则：`<out>/<类>/<类>_<NNNN>.cf32`，`NNNN` 四位编号。
- `--start-index auto`：每类从现有最大编号 + 1 开始；也可显式给整数。
- 2FSK 的 `bit_seed` 来自 `--bit-seeds`（与 `--seeds` 一一对应，默认全部沿用 11）。
- 其余参数（`--samp-rate`、`--center-hz`、`--signal-dbfs`、`--freq-offset-hz`、`--iq-mag-db`、`--iq-phase-deg` 等）可选，缺省沿用库默认。
- 汇总写 `<out>/records.csv`：每行一条记录 —— `class_label, file, snr_db, noise_seed, bit_seed, symbol_rate, fsk_deviation_hz, lfm_bandwidth_hz, chirp_period, sha256, samples, created`；与该类无关的列留空。
- 已有文件且未指定 `--overwrite` 时，捕获 `FileExistsError`、打印警告并跳过，不中断整批。

## 9. 验证策略

在 `D:\Programdata\radioconda\python.exe` 下执行：

1. **无损回归（核心验收）**：三类分别以默认参数生成到临时目录，与 `data/<类>/<类>_0001.cf32` 比对 sha256，必须相等。
2. **参数生效冒烟**：`snr_db=0` 与更换 `noise_seed` 后，sha256 必与 0001 不同，且文件大小 = `capture_samples×8`。
3. **分发入口**：`signal_source.generate("2FSK", …)` 与 `fsk2.generate(…)` 输出一致。
4. **错误路径**：重复生成不覆盖（`FileExistsError`）；非法参数（如 `fsk_deviation_hz >= samp_rate/2`）抛 `ValueError`。
5. **批量 CLI**：`--help` 正常；小规模（每类 1 条、`--seconds 0.2`）跑通并生成 records.csv。

若第 1 步 sha256 不一致：先排查差异（比对前 1024 字节、样本统计量），确认是环境/版本漂移还是重构错误，再修复；不得直接放宽为“统计近似”验收。

## 10. 风险与后续

- GNU Radio 版本变化可能改变块行为导致逐字节不一致；本机固定 3.10.12.0，跨机复现需版本一致。
- 逐字节一致依赖 `noise_seed` 固定时 channel_model 噪声序列确定；GLFSR 与信号源本身确定。
- 后续可扩展（不在本轮）：多径 `channel_taps` 参数、2FSK 自定义比特序列、不同 slot 图案预设。
