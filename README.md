# Genshin - Hex 处理流水线

> 简体中文 | [English](README_en.md) | [项目文档索引](#项目文档索引)

## 概述

Genshin 是一个基于 16 进制（Hex）统一编码的神经形态处理流水线，实现了从多模态输入（文本、文件）到神经网络的端到端处理。

### 核心特性

- **统一编码**：所有输入（文本、文件、图片等）统一为 16 进制表示
- **多头注意力**：支持因果掩码和 KV-Cache 的 Transformer 结构
- **流式/缓存双模式**：灵活切换批处理和增量推理
- **多种输出**：屏幕打印、日志记录、文件存储

## 模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                      HexPipeline                            │
│  用户输入 → 词元/解析 → HexMHA → 键盘输出                  │
└─────────────────────────────────────────────────────────────┘
           │           │         │            │
           ▼           ▼         ▼            ▼
    ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────────┐
    │ 词元模块  │ │解析模块│ │HexMHA  │ │ HexKeyboard│
    │token_mod │ │parser_ │ │多头注意│ │ 键盘输入   │
    │.py       │ │mod.py  │ │力模块  │ │            │
    └──────────┘ └────────┘ └────────┘ └────────────┘
```

## 模块说明

### 1. 词元模块 (token_module.py)

文本 ↔ UTF-8 十六进制双向转换。

```python
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str

# 字符串转hex
hex_str = str_to_hex("你好")  # e4bda0e5a5bd

# hex转字符串
text = hex_to_str("e4bda0e5a5bd")  # 你好
```

### 2. 解析模块 (parser_module.py)

任意文件 ↔ 十六进制文本无损转换。

```python
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file

# 文件转hex
hex_path = file_to_hex("image.png")  # 输出 image.hex.txt

# hex转文件
hex_to_file("image.hex.txt", "restored.png")
```

### 3. HexMHA 模块 (hex_mha_module_v2.py)

16 进制多头自注意力模块。

```python
from hex_mha_module_v2 import HexMHA

# 创建模型
mha = HexMHA(seq_len=16, dim=64, heads=4, embed_dim=64, causal=True)

# 流式模式
result = mha.forward("DEADBEEF")

# 缓存模式（增量推理）
mha.set_mode('cache')
mha.forward("DEAD", reset_cache=True)
result = mha.forward("BEEF")
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| seq_len | 8 | 最大序列长度 |
| dim | 64 | QKV 维度 |
| heads | 4 | 注意力头数 |
| embed_dim | 64 | 嵌入维度 |
| causal | False | 是否使用因果掩码 |
| mode | streaming | 模式：streaming/cache |

### 4. HexKeyboard 模块 (hex_keyboard.py)

16 进制键盘输入处理。

```python
from hex_keyboard import HexKeyboard, InputMode

kb = HexKeyboard(max_length=256, auto_pad=True)

# 流式输入
kb.input("DEADBEEF")

# 缓存模式（追加输入）
kb.set_mode(InputMode.CACHE)
kb.cache_input("AAAA")
kb.cache_input("BBBB")
```

## HexPipeline 使用

### 快速开始

```python
from hex_pipeline import HexPipeline

# 创建流水线
pipeline = HexPipeline(
    mha_seq_len=16,
    mha_dim=64,
    mha_heads=4
)

# 处理文本
result = pipeline.process_text("你好世界")

# 带索引处理
result = pipeline.process_text("Hello", index="0001")

# 设置输出模式
pipeline.set_output_mode('log')  # 仅日志
pipeline.set_output_mode('both') # 打印+日志
```

### 完整示例

```python
from hex_pipeline import HexPipeline

# 初始化
pipeline = HexPipeline(output_dir='./output')

# 处理文本
result = pipeline.process_text(
    text="原神，启动！",
    index="GAME_001",  # 可选索引
    save_result=True    # 保存到文件
)

# 处理文件
result = pipeline.process_file("document.pdf", save_result=True)

# 直接处理hex
result = pipeline.process_hex_direct("DEADBEEFCAFE")

# 解码结果
text = pipeline.decode_result(result)

# 还原为文件
pipeline.restore_file(result, "output.bin")
```

### API 参考

#### HexPipeline

| 方法 | 说明 |
|------|------|
| `process_text(text, index, save_result)` | 处理文本输入 |
| `process_file(file_path, index, save_result)` | 处理文件输入 |
| `process_hex_direct(hex_input)` | 直接处理hex字符串 |
| `decode_result(hex_result)` | hex解码为文本 |
| `restore_file(hex_result, output_path)` | hex还原为文件 |
| `set_output_mode(mode)` | 设置输出模式 |
| `set_mha_mode(mode)` | 设置MHA模式 |
| `reset_cache()` | 重置MHA缓存 |
| `get_stats()` | 获取统计信息 |

#### 输出模式

```python
pipeline.set_output_mode('print')  # 屏幕打印
pipeline.set_output_mode('log')    # 日志记录
pipeline.set_output_mode('file')   # 文件存储
pipeline.set_output_mode('both')   # 打印+日志
pipeline.set_output_mode('all')    # 全部输出
```

## 项目结构

```
genshin/
├── README.md              # 中文文档
├── README_en.md          # 英文文档
├── hex_agent.py          # Agent主进程（支持在线学习）
├── hex_pipeline.py        # 流水线主模块
├── hex_mha_module_v2.py  # 多头注意力模块
├── hex_keyboard.py       # 键盘输入模块
├── 词元模块_xxx.py       # 词元转换模块
├── 解析模块_xxx.py       # 文件解析模块
└── genshin_state/         # Agent状态目录
    ├── agent.log          # Agent日志
    ├── model_state.npz    # 模型权重
    └── pipeline_output/   # 流水线输出
        ├── pipeline.log   # 流水线日志
        └── result_*.hex  # 结果文件
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/hctj353056/genshin.git
cd genshin

# 直接使用（纯Python，无需安装）
python hex_pipeline.py
```

## HexAgent - 在线学习主进程

HexAgent 是一个支持**在线学习**的持续运行 Agent，参考 OpenClaw Agent Loop 架构。

### 核心特性

- **死循环主进程**：持续等待输入 → 处理 → 学习 → 输出
- **在线学习**：每次处理后增量更新 HexMHA 权重
- **学习闭环**：让 MHA 学会输出有效的 UTF-8 字符串
- **事件驱动**：支持 `on_input`、`on_output`、`on_learn` 等钩子
- **双模式运行**：交互模式（命令行）和服务模式（TCP）

### 学习闭环原理

```
┌──────────────────────────────────────────────────────────────┐
│                      在线学习闭环                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   用户输入 ──→ 词元模块 ──→ HexMHA ──→ 尝试转字符串         │
│                  (str→hex)     (处理)    (hex→str)           │
│                                        │                     │
│                          ┌─────────────┴─────────────┐      │
│                          ↓                           ↓      │
│                    ✅ 转换成功                      ❌ 转换失败│
│                    打印字符串                    记录失败样本 │
│                          │                           │      │
│                          ↓                           ↓      │
│                    强化这个输出              学习目标=输入hex│
│                    (保持有效)               (让MHA原样输出)  │
│                          │                           │      │
│                          └─────────────┬─────────────┘      │
│                                      ↓                      │
│                               参数更新                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**学习目标**：让 MHA 学会输出能成功转换为 UTF-8 字符串的十六进制数。

- 初始状态（随机权重）：MHA 输出乱码，无法转字符串
- 目标状态：MHA 输出有效 UTF-8 hex，能正确解码为原始文本

### 快速开始

```python
from hex_agent import HexAgent

# 创建Agent（启用在线学习）
agent = HexAgent(enable_online_learning=True, learning_rate=0.01)

# 交互模式
agent.run_interactive()
```

### 命令行使用

```bash
python hex_agent.py --mode interactive

# 禁用在线学习
python hex_agent.py --mode interactive --no-learning

# 服务模式（TCP）
python hex_agent.py --mode service --port 8765
```

### 交互命令

| 命令 | 说明 |
|------|------|
| `:learn on/off` | 开启/关闭在线学习 |
| `:mode print` | 输出模式：hex转字符串（默认） |
| `:mode echo` | 输出模式：直接hex |
| `:save` | 保存状态到文件 |
| `:stats` | 显示统计信息（含学习成功率） |
| `:reset` | 重置MHA缓存 |
| `:test` | 运行测试用例 |
| `:quit` | 退出 |

### API

```python
# 处理输入
result = agent.process("你好世界", index="001")

# 注册事件钩子
agent.register_hook('on_output', lambda x: print(f"输出: {x}"))

# 获取统计
stats = agent.get_stats()
```

## 项目文档索引

| 文档 | 说明 | 关联 |
|------|------|------|
| [README.md](README.md) | 中文项目文档 | ← 当前文档 |
| [README_en.md](README_en.md) | English Documentation | [English Index](#project-documentation-index) |
| hex_pipeline.py | 流水线核心代码 | 使用指南 |
| hex_mha_module_v2.py | 多头注意力实现 | 技术细节 |

## 更新日志

### v0.2.0 (2026-05-12)

- 重构 HexAgent，实现学习闭环
- 核心：让 MHA 学会输出有效 UTF-8 字符串
- 成功/失败样本记录，动态调整学习策略
- 新增 `:mode`、`:test` 命令

### v0.1.0 (2026-05-12)

- 实现基础流水线框架
- 整合词元、解析、HexMHA、键盘模块
- 支持流式/缓存双模式处理

## 许可证

MIT License

## 作者

[hctj353056](https://github.com/hctj353056)

---

*最后更新：2026-05-12*
