# hex_keyboard.py
# 16进制键盘模块 - 人类键盘 + X/Y架构
#
# 设计参考：人类键盘 + 命令行设计
# X: 操作模式（功能键）| Y: 数据负载
#
# 示例：
#   X=PRINT, Y=hex -> 屏幕打印hex转字符串
#   X=FILE, Y=hex -> 写入文件
#   X=LOG, Y=hex -> 记录日志

import os
import re
from typing import Optional, Callable
from enum import Enum
from datetime import datetime

# 导入词元模块
try:
    from 词元模块_1778459060672_3xq9 import hex_to_str
except ImportError:
    # 备用：如果词元模块不存在
    def hex_to_str(hex_str):
        try:
            return bytes.fromhex(hex_str).decode('utf-8')
        except:
            return hex_str

HEX_CHARS = '0123456789ABCDEF'


class KeyAction(Enum):
    """
    键盘动作（X轴 - 操作模式）
    
    类似人类键盘的功能键区，每个动作定义一个操作类型
    """
    # 输出类
    PRINT = "PRINT"      # 屏幕打印（Y=hex → 转为字符串打印）
    ECHO = "ECHO"       # 回显（Y=hex → 直接打印hex）
    ALERT = "ALERT"      # 警告（Y=hex → 警告消息）
    
    # 存储类
    SAVE = "SAVE"        # 保存文件（Y=hex → 写入文件）
    APPEND = "APPEND"   # 追加文件（Y=hex → 追加写入）
    LOAD = "LOAD"        # 读取文件（Y=hex路径 → 加载内容）
    
    # 日志类
    LOG = "LOG"         # 记录日志（Y=hex → 写入日志）
    LOG_DEBUG = "DEBUG"  # 调试日志
    LOG_ERROR = "ERROR"  # 错误日志
    
    # 状态类
    STATS = "STATS"     # 显示统计（Y被忽略）
    STATUS = "STATUS"    # 显示状态
    RESET = "RESET"      # 重置状态
    
    # 控制类
    LEARN_ON = "LEARN_ON"   # 开启学习
    LEARN_OFF = "LEARN_OFF"  # 关闭学习
    SAVE_STATE = "SAVE_STATE" # 保存状态
    QUIT = "QUIT"       # 退出
    
    # 空操作
    NOP = "NOP"         # 无操作


class HexKeyboard:
    """
    16进制键盘 - X/Y架构
    
    设计理念：
    - X轴（KeyAction）：定义操作类型
    - Y轴（数据）：hex字符串
    
    处理流程：
    输入(X,Y) → 解析 → 执行 → 输出
    
    人类键盘对照：
    - 功能键区（F1-F12）→ KeyAction
    - 主键区 → Y轴数据
    - Shift/Ctrl → 修饰符（扩展动作）
    """
    
    # 键位映射表（简化版，可扩展）
    KEY_MAP = {
        # 单字符快捷键
        'P': KeyAction.PRINT,
        'E': KeyAction.ECHO,
        'S': KeyAction.SAVE,
        'L': KeyAction.LOG,
        'A': KeyAction.APPEND,
        'R': KeyAction.RESET,
        'Q': KeyAction.QUIT,
        '?': KeyAction.STATS,
        '#': KeyAction.NOP,
        
        # 功能键
        'F1': KeyAction.PRINT,
        'F2': KeyAction.SAVE,
        'F3': KeyAction.LOAD,
        'F4': KeyAction.LOG,
        'F5': KeyAction.STATS,
        'F10': KeyAction.QUIT,
    }
    
    def __init__(self,
                 output_dir: str = './keyboard_output',
                 log_file: str = None,
                 max_buffer: int = 4096,
                 auto_convert: bool = True,
                 on_action: Optional[Callable] = None):
        """
        初始化键盘
        
        output_dir: 输出目录
        log_file: 日志文件路径
        max_buffer: 最大缓冲区大小
        auto_convert: 是否自动将hex转为字符串
        on_action: 动作执行回调 (action, data) -> result
        """
        self.output_dir = output_dir
        self.log_file = log_file or os.path.join(output_dir, 'keyboard.log')
        self.max_buffer = max_buffer
        self.auto_convert = auto_convert
        self.on_action = on_action
        
        # 缓冲区
        self.x_buffer: Optional[KeyAction] = None  # X轴（当前动作）
        self.y_buffer: str = ""                     # Y轴（数据）
        
        # 历史
        self.history: list[tuple[KeyAction, str]] = []
        
        # 统计
        self.stats = {
            'total_inputs': 0,
            'actions': {a.value: 0 for a in KeyAction}
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化日志
        self._init_log()
    
    def _init_log(self):
        """初始化日志"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"# HexKeyboard Log - {datetime.now().isoformat()}\n")
    
    def _log(self, message: str, level: str = 'INFO'):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
    
    def _convert_hex_to_string(self, hex_str: str) -> str:
        """将hex转为可读字符串"""
        if not self.auto_convert:
            return hex_str
        
        try:
            # 使用词元模块
            return hex_to_str(hex_str)
        except Exception:
            # 回退：尝试部分转换
            return f"[hex:{hex_str}]"
    
    def _parse_input(self, raw_input: str) -> tuple[Optional[KeyAction], str]:
        """
        解析输入，分离X和Y
        
        格式：
        - ":PRINT:deadbeef" -> (PRINT, "deadbeef")
        - "P deadbeef" -> (PRINT, "deadbeef")
        - "deadbeef" -> (ECHO, "deadbeef")  # 默认ECHO
        - "?" -> (STATS, "")
        """
        raw = raw_input.strip().upper()
        
        if not raw:
            return None, ""
        
        # 格式1: :ACTION:DATA
        if raw.startswith(':'):
            parts = raw[1:].split(':', 1)
            if len(parts) == 2:
                action_str, data = parts
                action = self._parse_action(action_str)
                if action:
                    return action, data
        
        # 格式2: SHORTCUT DATA
        parts = raw.split(None, 1)
        if len(parts) == 2:
            shortcut, data = parts
            action = self.KEY_MAP.get(shortcut)
            if action:
                return action, data
        
        # 格式3: 单字符快捷键
        if len(raw) == 1:
            action = self.KEY_MAP.get(raw)
            if action:
                return action, ""
        
        # 格式4: 功能键
        if raw.startswith('F'):
            action = self.KEY_MAP.get(raw)
            if action:
                return action, ""
        
        # 默认：作为ECHO处理
        return KeyAction.ECHO, raw
    
    def _parse_action(self, action_str: str) -> Optional[KeyAction]:
        """解析动作字符串"""
        action_str = action_str.upper().strip()
        
        # 别名映射
        alias_map = {
            'ERR': 'LOG_ERROR',
            'ERROR': 'LOG_ERROR',
            'WARNING': 'ALERT',
            'WARN': 'ALERT',
            'ECHO': 'ECHO',
            'READ': 'LOAD',
            'WRITE': 'SAVE',
            'STATS': 'STATS',
            'STAT': 'STATS',
        }
        
        if action_str in alias_map:
            action_str = alias_map[action_str]
        
        # 直接匹配
        try:
            return KeyAction[action_str]
        except KeyError:
            pass
        
        # 快捷键匹配
        return self.KEY_MAP.get(action_str)
    
    def input(self, raw_input: str) -> str:
        """
        处理输入
        
        参数:
            raw_input: 原始输入字符串
        
        返回:
            执行结果
        """
        self.stats['total_inputs'] += 1
        
        # 解析输入
        action, data = self._parse_input(raw_input)
        
        if action is None:
            return ""
        
        # 更新统计
        self.stats['actions'][action.value] += 1
        
        # 添加历史
        self.history.append((action, data))
        if len(self.history) > 100:
            self.history.pop(0)
        
        # 执行动作
        result = self._execute(action, data)
        
        # 触发回调
        if self.on_action:
            try:
                self.on_action(action, data, result)
            except Exception:
                pass
        
        return result
    
    def _execute(self, action: KeyAction, data: str) -> str:
        """执行动作"""
        self._log(f"Execute: X={action.value}, Y={data[:32]}...")
        
        # 数据清理
        data = ''.join(c for c in data.upper() if c in HEX_CHARS)
        
        if action == KeyAction.PRINT:
            # 屏幕打印：hex → 字符串 → 打印
            readable = self._convert_hex_to_string(data)
            output = f">>> {readable}"
            print(output)
            self._log(f"PRINT: {readable}")
            return output
        
        elif action == KeyAction.ECHO:
            # 回显：直接打印hex
            output = f"[hex] {data}"
            print(output)
            self._log(f"ECHO: {data}")
            return output
        
        elif action == KeyAction.ALERT:
            # 警告
            readable = self._convert_hex_to_string(data)
            output = f"⚠ {readable}"
            print(output)
            self._log(f"ALERT: {readable}", 'WARNING')
            return output
        
        elif action == KeyAction.SAVE:
            # 保存文件
            return self._save_file(data)
        
        elif action == KeyAction.APPEND:
            # 追加文件
            return self._append_file(data)
        
        elif action == KeyAction.LOAD:
            # 读取文件
            return self._load_file(data)
        
        elif action == KeyAction.LOG:
            # 记录日志
            readable = self._convert_hex_to_string(data)
            self._log(f"USER LOG: {readable}")
            output = f"[logged] {readable}"
            print(output)
            return output
        
        elif action == KeyAction.LOG_DEBUG:
            readable = self._convert_hex_to_string(data)
            self._log(f"DEBUG: {readable}", 'DEBUG')
            return f"[debug] {readable}"
        
        elif action == KeyAction.LOG_ERROR:
            readable = self._convert_hex_to_string(data)
            self._log(f"ERROR: {readable}", 'ERROR')
            output = f"❌ {readable}"
            print(output)
            return output
        
        elif action == KeyAction.STATS:
            # 显示统计
            return self._show_stats()
        
        elif action == KeyAction.STATUS:
            # 显示状态
            return self._show_status()
        
        elif action == KeyAction.RESET:
            # 重置
            self.x_buffer = None
            self.y_buffer = ""
            self._log("Reset")
            return "[reset] OK"
        
        elif action == KeyAction.LEARN_ON:
            self._log("Learning enabled")
            return "[learn] ON"
        
        elif action == KeyAction.LEARN_OFF:
            self._log("Learning disabled")
            return "[learn] OFF"
        
        elif action == KeyAction.SAVE_STATE:
            self._log("State saved")
            return "[state] saved"
        
        elif action == KeyAction.QUIT:
            self._log("Quit requested")
            return "[quit]"
        
        elif action == KeyAction.NOP:
            return ""
        
        return f"[unknown action: {action.value}]"
    
    def _save_file(self, data: str) -> str:
        """保存文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"output_{timestamp}.hex"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        
        self._log(f"Saved to {filepath}")
        return f"[saved] {filename}"
    
    def _append_file(self, data: str) -> str:
        """追加文件"""
        filename = f"log_{datetime.now().strftime('%Y%m%d')}.hex"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(data + '\n')
        
        self._log(f"Appended to {filepath}")
        return f"[appended] {filename}"
    
    def _load_file(self, hex_path: str) -> str:
        """读取文件"""
        # hex_path是hex编码的文件路径
        filepath = self._convert_hex_to_string(hex_path)
        
        if not os.path.exists(filepath):
            return f"[error] File not found: {filepath}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._log(f"Loaded from {filepath}")
        return f"[loaded] {content[:100]}..."
    
    def _show_stats(self) -> str:
        """显示统计"""
        lines = [
            "=== Keyboard Stats ===",
            f"Total inputs: {self.stats['total_inputs']}",
            "Actions:",
        ]
        
        for action_name, count in sorted(self.stats['actions'].items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  {action_name}: {count}")
        
        output = '\n'.join(lines)
        print(output)
        return output
    
    def _show_status(self) -> str:
        """显示状态"""
        output = f"""=== Keyboard Status ===
X buffer: {self.x_buffer.value if self.x_buffer else 'None'}
Y buffer: {self.y_buffer[:32]}...
History: {len(self.history)} items
Output dir: {self.output_dir}
"""
        print(output)
        return output
    
    def set_x(self, action: KeyAction):
        """设置X轴（动作）"""
        self.x_buffer = action
    
    def append_y(self, data: str):
        """追加Y轴（数据）"""
        self.y_buffer += ''.join(c for c in data.upper() if c in HEX_CHARS)
        # 限制长度
        if len(self.y_buffer) > self.max_buffer:
            self.y_buffer = self.y_buffer[-self.max_buffer:]
    
    def execute_xy(self) -> str:
        """执行X,Y"""
        if self.x_buffer is None:
            return ""
        
        result = self._execute(self.x_buffer, self.y_buffer)
        self.x_buffer = None
        self.y_buffer = ""
        return result
    
    def reset(self):
        """重置"""
        self.x_buffer = None
        self.y_buffer = ""
        self.history.clear()
    
    def get_stats(self) -> dict:
        """获取统计"""
        return {
            **self.stats,
            'x_buffer': self.x_buffer.value if self.x_buffer else None,
            'y_buffer_len': len(self.y_buffer),
            'history_len': len(self.history)
        }
    
    def __repr__(self):
        x = self.x_buffer.value if self.x_buffer else 'None'
        y = self.y_buffer[:16] + '...' if len(self.y_buffer) > 16 else self.y_buffer
        return f"HexKeyboard(X={x}, Y={y})"


# ============ 使用示例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("HexKeyboard X/Y 架构演示")
    print("=" * 60)
    
    # 创建键盘
    kb = HexKeyboard(output_dir='./keyboard_demo')
    
    print("\n--- 测试各种输入格式 ---\n")
    
    # 格式1: :ACTION:DATA
    print("1. :PRINT:e4bda0e5a5bd (打印汉字'你好')")
    kb.input(":PRINT:e4bda0e5a5bd")
    
    print()
    
    # 格式2: 快捷键
    print("2. P e4bda0e5a5bd (快捷键打印)")
    kb.input("P e4bda0e5a5bd")
    
    print()
    
    # 格式3: 默认ECHO
    print("3. DEADBEEF (默认ECHO)")
    kb.input("DEADBEEF")
    
    print()
    
    # 格式4: 显示统计
    print("4. ? (显示统计)")
    kb.input("?")
    
    print()
    
    # 格式5: 保存文件
    print("5. S cafebabe (保存文件)")
    kb.input("S cafebabe")
    
    print()
    
    # 格式6: 记录日志
    print("6. L 48656c6c6f (记录日志，hex='Hello')")
    kb.input("L 48656c6c6f")
    
    print()
    
    # 格式7: 错误日志
    print("7. :ERROR:e4bda0e5a5bde5a4aae695b0e4b8ade695af (错误日志)")
    kb.input(":ERROR:e4bda0e5a5bde5a4aae695b0e4b8ade695af")
    
    print("\n--- X/Y 手动模式 ---\n")
    
    # 手动设置X和Y
    kb.set_x(KeyAction.PRINT)
    kb.append_y("e4bda0e5a5bd")
    print(f"X={kb.x_buffer}, Y={kb.y_buffer}")
    kb.execute_xy()
    
    print("\n--- 最终统计 ---\n")
    print(kb.get_stats())
