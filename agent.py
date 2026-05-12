# -*- coding: utf-8 -*-
"""
Deepseek V4 Pro God - 主程序
通过XY二元组控制AI输出行为
"""

import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ActionType(Enum):
    """X动作类型"""
    THINK = "THINK"      # 颅内思考
    OUTPUT = "OUTPUT"    # 输出到屏幕
    HIGHLIGHT = "HIGHLIGHT"  # 高亮输出
    DELETE = "DELETE"    # 删除
    MODIFY = "MODIFY"    # 修改
    INSERT = "INSERT"    # 插入
    NEWLINE = "NEWLINE"  # 换行
    EXEC = "EXEC"        # 执行工具
    PAUSE = "PAUSE"      # 暂停等待
    ABORT = "ABORT"      # 中断
    CLEAR = "CLEAR"      # 清空


@dataclass
class XYAction:
    """XY二元组动作"""
    x: str
    y: str

    def __str__(self):
        return f"[{self.x}] {self.y}"


class DeepseekV4ProGod:
    """Deepseek V4 Pro God 主类"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.session_id = None
        
        # 系统提示词 - 定义XY动作协议
        self.system_prompt = """你是一个XY动作生成器，输出严格JSON格式的动作序列。

## XY动作协议
每个动作是一个(X, Y)二元组：
- X: 动作类型
- Y: 动作内容

## 可用动作类型

| 动作 | 说明 | Y内容示例 |
|------|------|----------|
| THINK | 颅内思考，不显示 | "我需要分析这个问题" |
| OUTPUT | 输出到屏幕 | "今天天气晴朗" |
| HIGHLIGHT | 高亮输出 | "重要内容" |
| DELETE | 删除最后N个字符 | "5" (删除5个字符) |
| MODIFY | 修改指定位置 | "3\|新内容" (修改位置3) |
| INSERT | 插入内容 | "2\|插入的文本" |
| NEWLINE | 换行 | "" |
| EXEC | 执行工具 | "search:python教程" |
| PAUSE | 暂停等待确认 | "等待用户确认" |
| ABORT | 中断输出 | "用户要求停止" |
| CLEAR | 清空所有内容 | "" |

## 输出格式
必须输出JSON数组格式：
```json
{
  "actions": [
    {"x": "THINK", "y": "思考内容"},
    {"x": "OUTPUT", "y": "输出内容"},
    {"x": "HIGHLIGHT", "y": "重要"},
    {"x": "OUTPUT", "y": "内容"}
  ]
}
```

## 规则
1. 先THINK再OUTPUT，体现思考过程
2. 重要内容用HIGHLIGHT
3. 错误用DELETE撤回
4. 保持JSON格式正确"""

    def chat(self, message: str, history: List[Dict] = None) -> List[XYAction]:
        """发送消息并获取XY动作序列"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建消息
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # 解析JSON响应
            actions = self._parse_response(content)
            return actions
            
        except requests.exceptions.RequestException as e:
            return [XYAction("ABORT", f"网络错误: {str(e)}")]
        except json.JSONDecodeError as e:
            return [XYAction("OUTPUT", f"解析错误: {content[:100]}...")]
        except Exception as e:
            return [XYAction("ABORT", f"未知错误: {str(e)}")]

    def _parse_response(self, content: str) -> List[XYAction]:
        """解析API响应为XY动作列表"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                actions = []
                for item in data.get("actions", []):
                    actions.append(XYAction(
                        x=item.get("x", "OUTPUT"),
                        y=item.get("y", "")
                    ))
                return actions
        except Exception:
            pass
        
        # 如果解析失败，返回为普通OUTPUT
        return [XYAction("OUTPUT", content)]

    def streaming_chat(self, message: str, callback):
        """流式对话，callback接收每个XYAction"""
        # 简化版本，实际使用需要流式API
        actions = self.chat(message)
        for action in actions:
            callback(action)
        return actions


# ========== 测试 ==========
if __name__ == "__main__":
    # 从环境变量或配置文件读取API Key
    import os
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        print("或直接在代码中设置: agent.api_key = 'your-key'")
    else:
        agent = DeepseekV4ProGod(api_key)
        
        print("=" * 50)
        print("Deepseek V4 Pro God 测试")
        print("=" * 50)
        
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            print("\n🤖 AI响应:")
            print("-" * 40)
            
            actions = agent.chat(user_input)
            
            for action in actions:
                print(f"  [{action.x}] {action.y}")
            
            print("-" * 40)
