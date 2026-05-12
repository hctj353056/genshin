# action_types.py
# Deepseek V4 Pro god - XY键盘动作类型定义
#
# X状态定义：
# - THINK: 颅内思考，不显示
# - OUTPUT: 输出到屏幕
# - DELETE: 删除
# - MODIFY: 修改指定位置
# - INSERT: 插入
# - NEWLINE: 换行
# - EXEC: 执行工具
# - PAUSE: 暂停等待确认
# - HIGHLIGHT: 高亮
# - COPY: 复制
# - PASTE: 粘贴
# - ABORT: 中断

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class ActionType(Enum):
    """X状态枚举"""
    THINK = "THINK"           # 颅内思考
    OUTPUT = "OUTPUT"          # 输出
    DELETE = "DELETE"          # 删除
    MODIFY = "MODIFY"         # 修改
    INSERT = "INSERT"         # 插入
    NEWLINE = "NEWLINE"       # 换行
    EXEC = "EXEC"             # 执行工具
    PAUSE = "PAUSE"           # 暂停
    HIGHLIGHT = "HIGHLIGHT"   # 高亮
    COPY = "COPY"             # 复制
    PASTE = "PASTE"           # 粘贴
    ABORT = "ABORT"           # 中断
    CONTINUE = "CONTINUE"     # 继续

@dataclass
class Action:
    """
    XY动作单元
    
    格式：(X, Y)
    - X: 动作类型
    - Y: 内容
    """
    action_type: ActionType
    content: str
    line: int = 1           # 行号
    position: int = 0       # 位置（字符索引）
    target: Optional[str] = None  # MODIFY时：目标位置
    tool: Optional[str] = None    # EXEC时：工具名
    params: Optional[dict] = None # EXEC时：参数
    
    def __str__(self):
        if self.action_type == ActionType.EXEC:
            return f"[{self.action_type.value}] {self.tool}:{self.content}"
        elif self.action_type == ActionType.MODIFY:
            return f"[{self.action_type.value}] {self.target}:{self.content}"
        elif self.action_type == ActionType.OUTPUT:
            return f"[{self.action_type.value}] Line{self.line}:{self.content}"
        else:
            return f"[{self.action_type.value}] {self.content}"
    
    def to_dict(self) -> dict:
        """转为字典"""
        return {
            "action": self.action_type.value,
            "content": self.content,
            "line": self.line,
            "position": self.position,
            "target": self.target,
            "tool": self.tool,
            "params": self.params
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'Action':
        """从字典创建"""
        return Action(
            action_type=ActionType(d.get("action", "THINK")),
            content=d.get("content", ""),
            line=d.get("line", 1),
            position=d.get("position", 0),
            target=d.get("target"),
            tool=d.get("tool"),
            params=d.get("params")
        )


class ActionSequence:
    """动作序列"""
    def __init__(self):
        self.actions: List[Action] = []
        self.think_count = 0
        self.output_count = 0
    
    def add(self, action: Action):
        self.actions.append(action)
        if action.action_type == ActionType.THINK:
            self.think_count += 1
        elif action.action_type == ActionType.OUTPUT:
            self.output_count += 1
    
    def get_visible_actions(self) -> List[Action]:
        """获取可见动作（排除THINK）"""
        return [a for a in self.actions if a.action_type != ActionType.THINK]
    
    def __len__(self):
        return len(self.actions)
    
    def __iter__(self):
        return iter(self.actions)


def parse_action(line: str) -> Optional[Action]:
    """
    解析单行动作
    
    支持格式：
    - THINK: 内容
    - OUTPUT: 内容
    - DELETE: 数量
    - MODIFY 位置: 内容
    - INSERT 位置: 内容
    - NEWLINE
    - EXEC 工具: 参数
    - PAUSE
    """
    line = line.strip()
    if not line:
        return None
    
    # THINK
    if line.startswith("THINK:"):
        return Action(ActionType.THINK, line[6:].strip())
    
    # OUTPUT
    if line.startswith("OUTPUT:"):
        content = line[7:].strip()
        return Action(ActionType.OUTPUT, content)
    
    # DELETE
    if line.startswith("DELETE:"):
        return Action(ActionType.DELETE, line[7:].strip())
    
    # MODIFY 位置: 内容
    if line.startswith("MODIFY "):
        parts = line[6:].split(":", 1)
        if len(parts) == 2:
            return Action(ActionType.MODIFY, parts[1], target=parts[0])
    
    # INSERT 位置: 内容
    if line.startswith("INSERT "):
        parts = line[6:].split(":", 1)
        if len(parts) == 2:
            return Action(ActionType.INSERT, parts[1], target=parts[0])
    
    # NEWLINE
    if line == "NEWLINE":
        return Action(ActionType.NEWLINE, "")
    
    # EXEC 工具: 参数
    if line.startswith("EXEC "):
        parts = line[4:].split(":", 1)
        if len(parts) == 2:
            return Action(ActionType.EXEC, parts[1], tool=parts[0])
    
    # PAUSE
    if line == "PAUSE":
        return Action(ActionType.PAUSE, "")
    
    # ABORT
    if line == "ABORT":
        return Action(ActionType.ABORT, "")
    
    # 默认当作OUTPUT
    return Action(ActionType.OUTPUT, line)


def parse_json_action(data: dict) -> Action:
    """解析JSON格式动作"""
    return Action.from_dict(data)
