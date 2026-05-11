# Genshin - Hex Processing Pipeline

> English | [简体中文](README.md) | [Project Documentation Index](#project-documentation-index)

## Overview

Genshin is a neuromorphic processing pipeline based on unified 16-bit hexadecimal (Hex) encoding, enabling end-to-end processing from multi-modal inputs (text, files) through neural networks.

### Core Features

- **Unified Encoding**: All inputs (text, files, images, etc.) unified as hexadecimal
- **Multi-Head Attention**: Transformer architecture with causal masking and KV-Cache
- **Dual Mode**: Flexible switching between batch processing and incremental inference
- **Multiple Outputs**: Console print, log recording, file storage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HexPipeline                            │
│  User Input → Token/Parser → HexMHA → Keyboard Output      │
└─────────────────────────────────────────────────────────────┘
           │           │         │            │
           ▼           ▼         ▼            ▼
    ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────────┐
    │  Token   │ │ Parser │ │ HexMHA │ │HexKeyboard │
    │ Module  │ │ Module │ │  MHA   │ │   Module   │
    │.py      │ │.py     │ │Module  │ │            │
    └──────────┘ └────────┘ └────────┘ └────────────┘
```

## Module Reference

### 1. Token Module (token_module.py)

Bidirectional conversion between text and UTF-8 hexadecimal.

```python
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str

# String to hex
hex_str = str_to_hex("Hello")  # 48656c6c6f

# Hex to string
text = hex_to_str("48656c6c6f")  # Hello
```

### 2. Parser Module (parser_module.py)

Lossless conversion between any file and hexadecimal text.

```python
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file

# File to hex
hex_path = file_to_hex("image.png")  # outputs image.hex.txt

# Hex to file
hex_to_file("image.hex.txt", "restored.png")
```

### 3. HexMHA Module (hex_mha_module_v2.py)

Hexadecimal Multi-Head Self-Attention module.

```python
from hex_mha_module_v2 import HexMHA

# Create model
mha = HexMHA(seq_len=16, dim=64, heads=4, embed_dim=64, causal=True)

# Streaming mode
result = mha.forward("DEADBEEF")

# Cache mode (incremental inference)
mha.set_mode('cache')
mha.forward("DEAD", reset_cache=True)
result = mha.forward("BEEF")
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| seq_len | 8 | Maximum sequence length |
| dim | 64 | QKV dimension |
| heads | 4 | Number of attention heads |
| embed_dim | 64 | Embedding dimension |
| causal | False | Enable causal masking |
| mode | streaming | Mode: streaming/cache |

### 4. HexKeyboard Module (hex_keyboard.py)

Hexadecimal keyboard input processing.

```python
from hex_keyboard import HexKeyboard, InputMode

kb = HexKeyboard(max_length=256, auto_pad=True)

# Streaming input
kb.input("DEADBEEF")

# Cache mode (append input)
kb.set_mode(InputMode.CACHE)
kb.cache_input("AAAA")
kb.cache_input("BBBB")
```

## HexPipeline Usage

### Quick Start

```python
from hex_pipeline import HexPipeline

# Create pipeline
pipeline = HexPipeline(
    mha_seq_len=16,
    mha_dim=64,
    mha_heads=4
)

# Process text
result = pipeline.process_text("Hello World")

# Process with index
result = pipeline.process_text("Hello", index="0001")

# Set output mode
pipeline.set_output_mode('log')   # Logs only
pipeline.set_output_mode('both')  # Print + logs
```

### Complete Example

```python
from hex_pipeline import HexPipeline

# Initialize
pipeline = HexPipeline(output_dir='./output')

# Process text
result = pipeline.process_text(
    text="Genshin Impact, Preemptive Launch!",
    index="GAME_001",  # Optional index
    save_result=True    # Save to file
)

# Process file
result = pipeline.process_file("document.pdf", save_result=True)

# Direct hex processing
result = pipeline.process_hex_direct("DEADBEEFCAFE")

# Decode result
text = pipeline.decode_result(result)

# Restore to file
pipeline.restore_file(result, "output.bin")
```

### API Reference

#### HexPipeline

| Method | Description |
|--------|-------------|
| `process_text(text, index, save_result)` | Process text input |
| `process_file(file_path, index, save_result)` | Process file input |
| `process_hex_direct(hex_input)` | Direct hex string processing |
| `decode_result(hex_result)` | Decode hex to text |
| `restore_file(hex_result, output_path)` | Restore hex to file |
| `set_output_mode(mode)` | Set output mode |
| `set_mha_mode(mode)` | Set MHA mode |
| `reset_cache()` | Reset MHA cache |
| `get_stats()` | Get statistics |

#### Output Modes

```python
pipeline.set_output_mode('print')  # Console output
pipeline.set_output_mode('log')    # Log recording
pipeline.set_output_mode('file')   # File storage
pipeline.set_output_mode('both')   # Print + logs
pipeline.set_output_mode('all')    # All outputs
```

## Project Structure

```
genshin/
├── README.md              # Chinese documentation
├── README_en.md           # English documentation
├── hex_pipeline.py        # Pipeline core module
├── hex_mha_module_v2.py  # Multi-head attention module
├── hex_keyboard.py        # Keyboard input module
├── 词元模块_xxx.py        # Token conversion module
├── 解析模块_xxx.py        # File parser module
└── pipeline_demo/          # Demo output directory
    ├── pipeline.log       # Log file
    └── result_*.hex      # Result files
```

## Installation

```bash
# Clone repository
git clone https://github.com/hctj353056/genshin.git
cd genshin

# Direct usage (pure Python, no installation needed)
python hex_pipeline.py
```

## HexAgent - Online Learning Main Process

HexAgent is a continuously running Agent with **online learning** support, inspired by OpenClaw Agent Loop architecture.

### Core Features

- **Infinite Loop**: Continuously wait → process → learn → output
- **Online Learning**: Incrementally update HexMHA weights after each processing
- **Event-Driven**: Supports `on_input`, `on_output`, `on_learn` hooks
- **Dual Mode**: Interactive (CLI) and Service (TCP) modes

### Quick Start

```python
from hex_agent import HexAgent

# Create Agent (with online learning)
agent = HexAgent(enable_online_learning=True, learning_rate=0.01)

# Interactive mode
agent.run_interactive()
```

### CLI Usage

```bash
python hex_agent.py --mode interactive

# Disable online learning
python hex_agent.py --mode interactive --no-learning

# Service mode (TCP)
python hex_agent.py --mode service --port 8765
```

### Commands

| Command | Description |
|---------|-------------|
| `:learn on/off` | Enable/disable online learning |
| `:save` | Save state to file |
| `:stats` | Show statistics |
| `:reset` | Reset MHA cache |
| `:quit` | Exit |

## Project Documentation Index

| Document | Description | Link |
|----------|-------------|------|
| [README.md](README.md) | Chinese Project Documentation | ← Current in Chinese |
| [README_en.md](README_en.md) | English Documentation | ← 当前文档 (Current Document) |
| hex_pipeline.py | Pipeline Core Code | [Usage Guide](#hexpipeline-usage) |
| hex_mha_module_v2.py | Multi-Head Attention Implementation | [Module Reference](#3-hexmha-module-hex_mha_module_v2py) |

## Changelog

### v0.1.0 (2026-05-12)

- Implemented basic pipeline framework
- Integrated Token, Parser, HexMHA, Keyboard modules
- Support for streaming/cache dual-mode processing

## License

MIT License

## Author

[hctj353056](https://github.com/hctj353056)

---

*Last Updated: 2026-05-12*
