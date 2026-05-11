# install.ps1 - Genshin 安装脚本 for Windows PowerShell
# 使用方式: 右键 -> 使用 PowerShell 运行 或 .\install.ps1

param(
    [switch]$NoGit  # 不使用git下载
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Genshin 安装脚本 (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检测Python
function Test-Python {
    try {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            $python = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if ($python) {
            $version = & python --version 2>&1
            Write-Host "[OK] 找到 Python: $version" -ForegroundColor Green
            return "python"
        }
    }
    catch {}
    Write-Host "[错误] 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    Write-Host "下载地址: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "或运行: winget install Python.Python.3.11" -ForegroundColor Yellow
    return $null
}

# 检测Git
function Test-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $version = & git --version
        Write-Host "[OK] 找到 Git: $version" -ForegroundColor Green
        return $true
    }
    Write-Host "[提示] 未找到 Git，将使用 Invoke-WebRequest 下载" -ForegroundColor Yellow
    return $false
}

# 下载项目
function Install-Project {
    param([bool]$HasGit)
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  下载 Genshin 项目"
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (Test-Path "genshin") {
        Write-Host "[提示] genshin 目录已存在" -ForegroundColor Yellow
        if ($HasGit) {
            Write-Host "正在更新..." -ForegroundColor Yellow
            Set-Location genshin
            & git pull
            Set-Location ..
        }
    }
    else {
        if ($HasGit) {
            Write-Host "克隆仓库..." -ForegroundColor Cyan
            & git clone https://github.com/hctj353056/genshin.git
        }
        else {
            Write-Host "下载源码压缩包..." -ForegroundColor Cyan
            $url = "https://github.com/hctj353056/genshin/archive/refs/heads/main.zip"
            Invoke-WebRequest -Uri $url -OutFile "genshin.zip"
            Expand-Archive -Path "genshin.zip" -DestinationPath "." -Force
            Move-Item -Path "genshin-main" -Destination "genshin" -Force
            Remove-Item "genshin.zip" -Force
        }
    }
    
    Write-Host "[OK] 项目下载完成" -ForegroundColor Green
}

# 创建快捷脚本
function New-Shortcuts {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  创建快捷脚本"
    Write-Host "========================================" -ForegroundColor Cyan
    
    $scriptsDir = "genshin"
    
    # 交互模式
    @"
@echo off
cd /d "%~dp0"
python hex_agent.py --mode interactive
pause
"@ | Out-File -FilePath "$scriptsDir\run.bat" -Encoding ASCII -Force
    
    # 服务模式
    @"
@echo off
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8765
python hex_agent.py --mode service --port %PORT%
pause
"@ | Out-File -FilePath "$scriptsDir\run_server.bat" -Encoding ASCII -Force
    
    # PowerShell 版本
    @"
# Genshin - HexAgent 交互模式
Set-Location `$PSScriptRoot
python hex_agent.py --mode interactive
Write-Host "按任意键退出..."
`$null = `$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"@ | Out-File -FilePath "$scriptsDir\run_ps1.ps1" -Encoding UTF8 -Force
    
    Write-Host "[OK] 快捷脚本已创建" -ForegroundColor Green
}

# 主流程
function Main {
    $python = Test-Python
    if (-not $python) {
        Read-Host "按 Enter 退出"
        return
    }
    
    $hasGit = Test-Git
    
    # 升级pip
    Write-Host ""
    Write-Host "升级 pip..." -ForegroundColor Cyan
    & pip install --upgrade pip 2>$null
    
    Install-Project -HasGit $hasGit
    New-Shortcuts
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  安装完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "使用方式:" -ForegroundColor White
    Write-Host "  cd genshin" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  交互模式:" -ForegroundColor White
    Write-Host "    run.bat" -ForegroundColor Gray
    Write-Host "    或双击 run.bat" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  服务模式:" -ForegroundColor White
    Write-Host "    run_server.bat 8765" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  直接运行:" -ForegroundColor White
    Write-Host "    python hex_agent.py --mode interactive" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  PowerShell:" -ForegroundColor White
    Write-Host "    .\run_ps1.ps1" -ForegroundColor Gray
    Write-Host ""
}

Main
