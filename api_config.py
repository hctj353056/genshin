# -*- coding: utf-8 -*-
"""
API配置文件
"""

# DeepSeek API配置
DEEPSEEK_API_KEY = ""  # 填入你的API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 备选模型
ALT_MODELS = {
    "chat": "deepseek-chat",
    "coder": "deepseek-coder"
}

# 渲染器配置
RENDERER_CONFIG = {
    "port": 8765,  # Web渲染器端口
    "host": "localhost",
    "auto_open": True
}

# 默认XY动作超时（秒）
ACTION_TIMEOUT = 30

# 历史记录保留条数
MAX_HISTORY = 100
