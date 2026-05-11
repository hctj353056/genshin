# Genshin - Installation Guide

> English | [简体中文](安装指南.md)

## Supported Platforms

| Platform | Script | Requirements |
|----------|--------|--------------|
| Linux | `install.sh` | Python 3.8+, bash |
| macOS | `install.sh` | Python 3.8+, Terminal |
| Termux (Android) | `install.sh` | Python 3.8+ |
| Windows (CMD) | `install.bat` | Python 3.8+, Git |
| Windows (PowerShell) | `install.ps1` | Python 3.8+, PowerShell 5+ |

## Quick Install

### Linux / macOS / Termux

```bash
# 下载安装脚本
wget https://raw.githubusercontent.com/hctj353056/genshin/main/install.sh

# 或克隆整个项目
git clone https://github.com/hctj353056/genshin.git
cd genshin

# 运行安装
bash install.sh
```

### Windows (PowerShell)

```powershell
# 方法1: 下载脚本后运行
irm https://raw.githubusercontent.com/hctj353056/genshin/main/install.ps1 | iex

# 方法2: 克隆后运行
git clone https://github.com/hctj353056/genshin.git
cd genshin
.\install.ps1
```

### Windows (CMD)

```cmd
# 克隆项目
git clone https://github.com/hctj353056/genshin.git
cd genshin

# 运行安装
install.bat
```

## Manual Installation

If you prefer manual installation:

```bash
# 1. Clone or download the project
git clone https://github.com/hctj353056/genshin.git
cd genshin

# 2. Install dependencies (pure Python, no external packages required)
# Just ensure you have Python 3.8+

# 3. Run directly
python hex_agent.py --mode interactive
```

## Usage

### Interactive Mode

```bash
cd genshin
./run.sh
# or
python hex_agent.py --mode interactive
```

Commands in interactive mode:

| Command | Description |
|---------|-------------|
| `:learn on/off` | Enable/disable online learning |
| `:save` | Save state to file |
| `:stats` | Show statistics |
| `:reset` | Reset MHA cache |
| `:quit` | Exit |

### Service Mode (TCP)

```bash
# Start on port 8765
./run_server.sh 8765
# or
python hex_agent.py --mode service --port 8765
```

### Disable Online Learning

```bash
python hex_agent.py --mode interactive --no-learning
```

## Termux Special Notes

For Termux on Android:

```bash
# Install Python
pkg update && pkg upgrade
pkg install python

# Clone project
pkg install git
git clone https://github.com/hctj353056/genshin.git
cd genshin

# Run
bash install.sh
# or directly
python hex_agent.py --mode interactive
```

## Troubleshooting

### Python not found

**Linux/macOS:**
```bash
# Install Python 3
sudo apt install python3  # Debian/Ubuntu
sudo yum install python3  # CentOS/RHEL
brew install python3     # macOS
```

**Termux:**
```bash
pkg install python
```

**Windows:**
Download from https://www.python.org/downloads/

### Git not found

**Windows:** Download from https://git-scm.com/download/win

**Termux:**
```bash
pkg install git
```

## Project Structure

```
genshin/
├── install.sh          # Linux/macOS/Termux 安装脚本
├── install.bat          # Windows CMD 安装脚本
├── install.ps1         # Windows PowerShell 安装脚本
├── README.md           # 中文文档
├── README_en.md        # English documentation
├── hex_agent.py        # Agent主程序
├── hex_pipeline.py     # 处理流水线
├── hex_mha_module_v2.py # 多头注意力模块
├── hex_keyboard.py     # 键盘输入模块
└── genshin_state/      # 状态存储目录
```

## License

MIT License
