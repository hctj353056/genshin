@echo off
chcp 65001 > nul
REM install.bat - Genshin 安装脚本 for Windows

echo ========================================
echo   Genshin 安装脚本 (Windows)
echo ========================================
echo.

REM 检测Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo [错误] 未找到 Python，请先安装 Python 3.8+
        echo 下载地址: https://www.python.org/downloads/
        pause
        exit /b 1
    ) else (
        set PYTHON=python3
    )
) else (
    set PYTHON=python
)

echo [OK] 找到 Python
%PYTHON% --version

REM 升级pip
echo.
echo 升级 pip...
%PYTHON% -m pip install --upgrade pip

REM 下载项目
echo.
echo ========================================
echo   下载 Genshin 项目
echo ========================================

if exist genshin (
    echo [提示] genshin 目录已存在，正在更新...
    cd genshin
    git pull
    cd ..
) else (
    echo 克隆仓库...
    git clone https://github.com/hctj353056/genshin.git
)

REM 创建快捷脚本
echo.
echo ========================================
echo   创建快捷脚本
echo ========================================

REM 交互模式
echo @echo off > genshin\run.bat
echo cd /d "%%~dp0" >> genshin\run.bat
echo python hex_agent.py --mode interactive %%* >> genshin\run.bat
echo pause >> genshin\run.bat

REM 服务模式
echo @echo off > genshin\run_server.bat
echo cd /d "%%~dp0" >> genshin\run_server.bat
echo if "%%1"=="" (set PORT=8765^) else (set PORT=%%1) >> genshin\run_server.bat
echo python hex_agent.py --mode service --port %%PORT%% >> genshin\run_server.bat

echo [OK] 快捷脚本已创建

REM 完成
echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 使用方式:
echo   cd genshin
echo.
echo   交互模式:
echo     run.bat
echo.
echo   服务模式:
echo     run_server.bat 8765
echo.
echo   直接运行:
echo     python hex_agent.py --mode interactive
echo.
pause
