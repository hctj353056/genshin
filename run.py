# -*- coding: utf-8 -*-
"""
启动脚本 - 一键启动Deepseek V4 Pro God
"""

import os
import sys
import webbrowser
from api_config import DEEPSEEK_API_KEY, RENDERER_CONFIG

def check_config():
    """检查配置"""
    if not DEEPSEEK_API_KEY:
        print("❌ 错误: 请先配置 API Key")
        print()
        print("方法1: 编辑 api_config.py")
        print('   DEEPSEEK_API_KEY = "sk-xxx"')
        print()
        print("方法2: 设置环境变量")
        print('   export DEEPSEEK_API_KEY="sk-xxx"')
        print()
        print("方法3: 在Web界面中配置")
        return False
    return True

def start_renderer():
    """启动Web渲染器"""
    host = RENDERER_CONFIG["host"]
    port = RENDERER_CONFIG["port"]
    url = f"http://{host}:{port}/shell.html"
    
    print(f"🌐 启动渲染器: {url}")
    
    if RENDERER_CONFIG["auto_open"]:
        webbrowser.open(url)

def main():
    print("=" * 50)
    print("🚀 Deepseek V4 Pro God")
    print("=" * 50)
    print()
    
    # 检查配置
    if not check_config():
        # 还是启动渲染器，用户可以在界面配置
        pass
    
    # 启动Web渲染器
    start_renderer()
    
    print()
    print("📝 使用说明:")
    print("   1. 在浏览器中打开弹窗")
    print("   2. 点击 ⚙️ 配置API Key")
    print("   3. 输入你的DeepSeek API Key")
    print("   4. 开始对话!")
    print()
    print("按 Ctrl+C 退出")
    print("-" * 50)
    
    # 简单的HTTP服务器
    try:
        import http.server
        import socketserver
        
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", RENDERER_CONFIG["port"]), Handler) as httpd:
            print(f"💫 服务运行中 http://localhost:{RENDERER_CONFIG['port']}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 退出")

if __name__ == "__main__":
    main()
