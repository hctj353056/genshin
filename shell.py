# shell.py
# Deepseek V4 Pro god - XY键盘壳程序渲染器
#
# 功能：
# - 解析XY动作序列
# - 维护显示缓冲区
# - 支持回溯修改
# - 双轨显示（颅内/屏幕）

from typing import List, Dict, Optional, Tuple
from action_types import Action, ActionType, ActionSequence, parse_action, parse_json_action
import json

class ScreenBuffer:
    """
    屏幕缓冲区
    
    维护多行文本，支持插入、删除、修改
    """
    def __init__(self):
        self.lines: List[str] = [""]  # 初始化一行
        self.highlights: List[Tuple[int, int, int]] = []  # (行, 开始, 结束)
    
    def output(self, content: str, line: int = None):
        """输出内容到指定行"""
        if line is None:
            line = len(self.lines) - 1
        
        # 确保行存在
        while line >= len(self.lines):
            self.lines.append("")
        
        self.lines[line] += content
    
    def delete(self, count: int, line: int = None):
        """删除指定行末尾N个字符"""
        if line is None:
            line = len(self.lines) - 1
        
        if 0 <= line < len(self.lines):
            self.lines[line] = self.lines[line][:-count] if count < len(self.lines[line]) else ""
    
    def modify(self, position: str, content: str):
        """
        修改指定位置
        
        position格式：
        - "行-位置"：如 "1-3" 表示第1行第3个字符后
        - "位置"：当前行第N个字符后
        """
        if "-" in position:
            parts = position.split("-")
            line = int(parts[0]) - 1
            pos = int(parts[1])
        else:
            line = len(self.lines) - 1
            pos = int(position)
        
        if 0 <= line < len(self.lines):
            line_content = self.lines[line]
            if 0 <= pos <= len(line_content):
                self.lines[line] = line_content[:pos] + content + line_content[pos:]
    
    def insert(self, position: str, content: str):
        """在指定位置插入内容"""
        self.modify(position, content)  # 与modify相同
    
    def newline(self):
        """插入新行"""
        self.lines.append("")
    
    def highlight(self, position: str):
        """高亮指定位置"""
        if "-" in position:
            parts = position.split("-")
            line = int(parts[0]) - 1
            start = int(parts[1])
            end = int(parts[2]) if len(parts) > 2 else start + 1
            self.highlights.append((line, start, end))
    
    def clear_highlights(self):
        """清除高亮"""
        self.highlights = []
    
    def get_line(self, line: int = None) -> str:
        """获取指定行"""
        if line is None:
            line = len(self.lines) - 1
        return self.lines[line] if 0 <= line < len(self.lines) else ""
    
    def get_all(self) -> str:
        """获取所有内容"""
        return "\n".join(self.lines)
    
    def __str__(self):
        return self.get_all()
    
    def __repr__(self):
        return f"ScreenBuffer(lines={len(self.lines)}, content={self.get_all()[:50]})"


class ShellRenderer:
    """
    XY键盘壳程序渲染器
    
    功能：
    - 解析动作序列
    - 更新屏幕缓冲区
    - 渲染显示
    - 支持回溯
    """
    
    def __init__(self, show_thinks: bool = False):
        self.screen = ScreenBuffer()
        self.show_thinks = show_thinks  # 是否显示思考内容
        self.think_log: List[str] = []  # 思考日志
        self.action_history: List[Action] = []  # 动作历史
        self.pause_requested = False
    
    def reset(self):
        """重置状态"""
        self.screen = ScreenBuffer()
        self.think_log = []
        self.action_history = []
        self.pause_requested = False
        self.highlights: List[Tuple[int, int, int]] = []  # (行, 开始, 结束)
    
    def execute(self, action: Action) -> bool:
        """
        执行单个动作
        
        返回：是否继续执行
        """
        self.action_history.append(action)
        
        if action.action_type == ActionType.THINK:
            self.think_log.append(action.content)
            if self.show_thinks:
                print(f"  [颅内] {action.content}")
        
        elif action.action_type == ActionType.OUTPUT:
            self.screen.output(action.content, action.line - 1)
        
        elif action.action_type == ActionType.DELETE:
            count = int(action.content)
            self.screen.delete(count, action.line - 1)
        
        elif action.action_type == ActionType.MODIFY:
            self.screen.modify(action.target, action.content)
        
        elif action.action_type == ActionType.INSERT:
            self.screen.insert(action.target, action.content)
        
        elif action.action_type == ActionType.NEWLINE:
            self.screen.newline()
        
        elif action.action_type == ActionType.HIGHLIGHT:
            # 高亮内容不显示，只记录高亮位置
            # content格式："行-开始-结束" 或 "行-位置"
            parts = action.content.split("-")
            if len(parts) >= 2:
                line = int(parts[0]) - 1
                start = int(parts[1])
                end = int(parts[2]) if len(parts) > 2 else start + 1
                self.screen.highlights.append((line, start, end))
        
        elif action.action_type == ActionType.PAUSE:
            self.pause_requested = True
            return False
        
        elif action.action_type == ActionType.ABORT:
            return False
        
        elif action.action_type == ActionType.CONTINUE:
            self.pause_requested = False
        
        return True
    
    def execute_sequence(self, actions: List[Action], auto_continue: bool = True) -> bool:
        """
        执行动作序列
        
        返回：是否正常完成
        """
        for action in actions:
            if not self.execute(action):
                if self.pause_requested:
                    print("\n" + "="*50)
                    print("⏸️  暂停等待确认...")
                    print("="*50)
                    if not auto_continue:
                        return False
                elif action.action_type == ActionType.ABORT:
                    print("\n" + "="*50)
                    print("⚠️  被中断")
                    print("="*50)
                    return False
        
        return True
    
    def render(self, show_thinks: bool = None) -> str:
        """
        渲染最终显示
        
        返回格式化的屏幕内容
        """
        if show_thinks is None:
            show_thinks = self.show_thinks
        
        lines = []
        
        # 渲染思考日志
        if show_thinks and self.think_log:
            lines.append("\n" + "─"*50)
            lines.append("【颅内思考】")
            for think in self.think_log:
                lines.append(f"  💭 {think}")
            lines.append("─"*50)
        
        # 渲染屏幕内容
        lines.append("\n" + "="*50)
        lines.append("【屏幕输出】")
        for i, line in enumerate(self.screen.lines):
            display_line = line
            # 检查是否有高亮
            highlights_on_line = [(s, e) for (l, s, e) in self.screen.highlights if l == i]
            if highlights_on_line:
                for start, end in highlights_on_line:
                    if start <= len(display_line):
                        end = min(end, len(display_line))
                        display_line = display_line[:start] + "【" + display_line[start:end] + "】" + display_line[end:]
            lines.append(f"  {i+1}: {display_line}")
        lines.append("="*50)
        
        return "\n".join(lines)
    
    def quick_render(self) -> str:
        """快速渲染：只显示最终屏幕"""
        return self.screen.get_all()


def parse_response(response_text: str) -> List[Action]:
    """
    解析响应文本为动作序列
    
    支持格式：
    1. JSON格式：[{"action": "THINK", "content": "..."}]
    2. 文本格式：每行一个动作
    """
    actions = []
    
    # 尝试JSON格式
    if response_text.strip().startswith("["):
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        actions.append(parse_json_action(item))
            return actions
        except:
            pass
    
    # 文本格式
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line:
            action = parse_action(line)
            if action:
                actions.append(action)
    
    return actions


def demo():
    """演示"""
    print("="*60)
    print("XY键盘壳程序演示")
    print("="*60)
    
    shell = ShellRenderer(show_thinks=True)
    
    # 模拟DeepSeek返回的动作序列
    demo_actions = [
        Action(ActionType.THINK, "用户询问天气，需要查询天气信息"),
        Action(ActionType.THINK, "根据知识库，今天是晴天，28度"),
        Action(ActionType.THINK, "组织语言回复"),
        Action(ActionType.OUTPUT, "今天", line=1),
        Action(ActionType.OUTPUT, "天气", line=1),
        Action(ActionType.DELETE, "1", line=1),
        Action(ActionType.OUTPUT, "晴", line=1),
        Action(ActionType.NEWLINE),
        Action(ActionType.OUTPUT, "温度：28°C", line=2),
        Action(ActionType.HIGHLIGHT, "2:7-10"),
    ]
    
    print("\n执行动作序列...")
    print("-"*50)
    
    for action in demo_actions:
        shell.execute(action)
    
    print(shell.render())
    
    print("\n快速渲染：")
    print(shell.quick_render())


if __name__ == "__main__":
    demo()
