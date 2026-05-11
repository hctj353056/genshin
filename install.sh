#!/bin/bash
# install.sh - Genshin 安装脚本
# 支持: Linux, macOS, Termux (Android)

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Genshin 安装脚本"
echo "========================================"
echo ""

# 检测Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON=python3
        echo -e "${GREEN}✓${NC} 找到 Python3: $($PYTHON --version)"
    elif command -v python &> /dev/null; then
        PYTHON=python
        echo -e "${GREEN}✓${NC} 找到 Python: $($PYTHON --version)"
    else
        echo -e "${RED}✗${NC} 未找到 Python，请先安装 Python 3.8+"
        exit 1
    fi
}

# 检测Git
check_git() {
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✓${NC} 找到 Git: $(git --version)"
    else
        echo -e "${YELLOW}!${NC} 未找到 Git，将使用 wget 下载"
    fi
}

# 检测pip
check_pip() {
    if $PYTHON -m pip --version &> /dev/null; then
        echo -e "${GREEN}✓${NC} 找到 pip"
    else
        echo -e "${YELLOW}!${NC} 未找到 pip，尝试安装..."
        $PYTHON -m ensurepip --default-pip 2>/dev/null || {
            echo -e "${RED}✗${NC} pip 安装失败"
        }
    fi
}

# 检测平台
detect_platform() {
    if [ -d "/data/data/com.termux" ]; then
        PLATFORM="termux"
        echo -e "${GREEN}检测到平台: Termux (Android)${NC}"
    elif [ "$(uname)" = "Darwin" ]; then
        PLATFORM="macos"
        echo -e "${GREEN}检测到平台: macOS${NC}"
    else
        PLATFORM="linux"
        echo -e "${GREEN}检测到平台: Linux${NC}"
    fi
}

# 安装依赖
install_deps() {
    echo ""
    echo "========================================"
    echo "  安装 Python 依赖"
    echo "========================================"
    
    $PYTHON -m pip install --upgrade pip 2>/dev/null || true
    
    echo -e "${GREEN}✓${NC} 依赖安装完成"
}

# 下载项目
download_project() {
    echo ""
    echo "========================================"
    echo "  下载 Genshin 项目"
    echo "========================================"
    
    if [ -d "genshin" ]; then
        echo -e "${YELLOW}!${NC} genshin 目录已存在，正在更新..."
        cd genshin
        git pull
        cd ..
    else
        if command -v git &> /dev/null; then
            echo "克隆仓库..."
            git clone https://github.com/hctj353056/genshin.git
        else
            echo "下载源码..."
            wget -q https://github.com/hctj353056/genshin/archive/refs/heads/main.zip -O genshin.zip
            unzip -q genshin.zip
            mv genshin-main genshin
            rm genshin.zip
        fi
    fi
    
    echo -e "${GREEN}✓${NC} 项目下载完成"
}

# 创建运行脚本
create_scripts() {
    echo ""
    echo "========================================"
    echo "  创建快捷脚本"
    echo "========================================"
    
    # 主程序脚本
    cat > genshin/hex_agent.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
python3 hex_agent.py "$@"
SCRIPT
    chmod +x genshin/hex_agent.sh
    
    # 交互模式
    cat > genshin/run.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
echo "启动 Genshin Agent..."
python3 hex_agent.py --mode interactive
SCRIPT
    chmod +x genshin/run.sh
    
    # 服务模式
    cat > genshin/run_server.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
PORT=${1:-8765}
echo "启动 Genshin 服务模式，端口: $PORT"
python3 hex_agent.py --mode service --port $PORT
SCRIPT
    chmod +x genshin/run_server.sh
    
    echo -e "${GREEN}✓${NC} 快捷脚本已创建"
}

# 完成
finish() {
    echo ""
    echo "========================================"
    echo -e "${GREEN}  安装完成！${NC}"
    echo "========================================"
    echo ""
    echo "使用方式:"
    echo "  cd genshin"
    echo ""
    echo "  交互模式:"
    echo "    ./run.sh"
    echo "    # 或"
    echo "    python3 hex_agent.py --mode interactive"
    echo ""
    echo "  服务模式:"
    echo "    ./run_server.sh 8765"
    echo "    # 或"
    echo "    python3 hex_agent.py --mode service --port 8765"
    echo ""
    echo "  禁用在线学习:"
    echo "    python3 hex_agent.py --mode interactive --no-learning"
    echo ""
}

# 主流程
main() {
    detect_platform
    check_python
    check_git
    check_pip
    install_deps
    download_project
    create_scripts
    finish
}

main "$@"
