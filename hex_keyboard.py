# hex_keyboard.py
# 16进制键盘输入模块
# 为HexMHA等模块提供统一的输入接口

import re
from typing import Optional, Callable
from enum import Enum

HEX_CHARS = '0123456789ABCDEF'


class InputMode(Enum):
    """输入模式"""
    STREAMING = 'streaming'  # 流式输入
    CACHE = 'cache'          # 缓存输入
    BATCH = 'batch'          # 批量输入


class HexKeyboard:
    def __init__(self, max_length: int = 256, auto_pad: bool = False,
                 on_input: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        """
        max_length: 最大输入长度
        auto_pad: 是否自动补齐到偶数位（每2个hex=1字节）
        on_input: 输入回调 (hex_str) -> None
        on_error: 错误回调 (error_msg) -> None
        """
        self.max_length = max_length
        self.auto_pad = auto_pad
        self.on_input = on_input
        self.on_error = on_error
        
        # 缓冲区
        self.buffer = ''
        self.history: list[str] = []
        
        # 缓存模式状态
        self._cache_mode = False
        self._cache_buffer = ''
        
    def input(self, text: str) -> str:
        """
        处理输入文本，返回清洗后的hex字符串
        自动过滤非hex字符
        """
        try:
            # 过滤非hex字符
            cleaned = self._clean_hex(text.upper())
            
            if not cleaned:
                return ''
            
            # 补齐到偶数位
            if self.auto_pad and len(cleaned) % 2 == 1:
                cleaned = cleaned[:-1]  # 去掉最后一个奇数字符
            
            # 截断超长部分
            if len(cleaned) > self.max_length:
                cleaned = cleaned[:self.max_length]
            
            # 更新缓冲区
            self.buffer = cleaned
            
            # 触发回调
            if self.on_input:
                self.on_input(cleaned)
            
            return cleaned
            
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            return ''
    
    def _clean_hex(self, text: str) -> str:
        """过滤，只保留hex字符"""
        return ''.join(c for c in text if c in HEX_CHARS)
    
    def set_mode(self, mode: InputMode):
        """切换输入模式"""
        self._cache_mode = (mode == InputMode.CACHE)
        if mode == InputMode.BATCH:
            self._cache_buffer = ''
        else:
            self._cache_buffer = self.buffer
    
    def cache_input(self, text: str) -> str:
        """
        缓存模式输入（追加到缓冲区）
        """
        cleaned = self._clean_hex(text.upper())
        
        if self.auto_pad and len(cleaned) % 2 == 1:
            cleaned = cleaned[:-1]
        
        # 追加到缓存
        new_total = self._cache_buffer + cleaned
        if len(new_total) > self.max_length:
            new_total = new_total[:self.max_length]
        
        self._cache_buffer = new_total
        self.buffer = new_total
        
        if self.on_input:
            self.on_input(new_total)
        
        return new_total
    
    def reset_cache(self):
        """重置缓存"""
        self._cache_buffer = ''
        self.buffer = ''
    
    def get_buffer(self) -> str:
        """获取当前缓冲区"""
        return self.buffer
    
    def get_cache(self) -> str:
        """获取缓存内容"""
        return self._cache_buffer
    
    def push_history(self):
        """将当前缓冲区推入历史"""
        if self.buffer:
            self.history.append(self.buffer)
            # 限制历史长度
            if len(self.history) > 100:
                self.history.pop(0)
    
    def get_history(self, n: int = 10) -> list[str]:
        """获取最近n条历史"""
        return self.history[-n:]
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = ''
        self._cache_buffer = ''
    
    def char_count(self) -> int:
        """当前hex字符数量"""
        return len(self.buffer)
    
    def byte_count(self) -> int:
        """当前字节数量（2个hex=1字节）"""
        return len(self.buffer) // 2
    
    def is_empty(self) -> bool:
        """缓冲区是否为空"""
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        """缓冲区是否已满"""
        return len(self.buffer) >= self.max_length
    
    def __repr__(self):
        mode = 'CACHE' if self._cache_mode else 'STREAMING'
        return f"HexKeyboard(buffer='{self.buffer[:16]}...', len={len(self.buffer)}, mode={mode})"


class HexKeyboardWithValidator(HexKeyboard):
    """带验证器的键盘"""
    
    def __init__(self, max_length: int = 256, auto_pad: bool = False,
                 validators: Optional[list[Callable[[str], tuple[bool, str]]]] = None,
                 on_input: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        """
        validators: 验证器列表，每个验证器返回 (is_valid, error_msg)
        """
        super().__init__(max_length, auto_pad, on_input, on_error)
        self.validators = validators or []
    
    def input(self, text: str) -> str:
        """带验证的输入"""
        cleaned = super().input(text)
        
        if not cleaned:
            return ''
        
        # 运行验证器
        for validator in self.validators:
            is_valid, error_msg = validator(cleaned)
            if not is_valid:
                if self.on_error:
                    self.on_error(error_msg)
                return ''
        
        return cleaned


# ============ 内置验证器 ============

def validate_length(min_len: int = 0, max_len: int = 256) -> Callable[[str], tuple[bool, str]]:
    """长度验证器"""
    def validate(text: str) -> tuple[bool, str]:
        if len(text) < min_len:
            return False, f"长度不足，最少{min_len}个hex字符"
        if len(text) > max_len:
            return False, f"长度超限，最多{max_len}个hex字符"
        return True, ''
    return validate


def validate_even_length() -> Callable[[str], tuple[bool, str]]:
    """偶数长度验证器"""
    def validate(text: str) -> tuple[bool, str]:
        if len(text) % 2 != 0:
            return False, "长度必须为偶数（完整字节）"
        return True, ''
    return validate


def validate_hex_pattern(pattern: str) -> Callable[[str], tuple[bool, str]]:
    """正则模式验证"""
    compiled = re.compile(pattern)
    def validate(text: str) -> tuple[bool, str]:
        if not compiled.match(text):
            return False, f"不符合格式要求: {pattern}"
        return True, ''
    return validate


# ============ 简单测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("HexKeyboard 测试")
    print("=" * 50)
    
    # 基本使用
    kb = HexKeyboard(max_length=32, auto_pad=True)
    print(f"\n创建键盘: {kb}")
    
    result = kb.input("DEADBEEF")
    print(f"输入 'DEADBEEF' -> '{result}'")
    
    result = kb.input("DE AD BE EF CA FE 12345!@#")
    print(f"输入 'DE AD BE EF CA FE 12345!@#' -> '{result}'")
    
    result = kb.input("GHIJ")  # GHIJ不是hex
    print(f"输入 'GHIJ' -> '{result}'")
    
    # 缓存模式
    print("\n【缓存模式测试】")
    kb.set_mode(InputMode.CACHE)
    kb.cache_input("AAAA")
    kb.cache_input("BBBB")
    print(f"缓存内容: '{kb.get_cache()}'")
    print(f"字符数: {kb.char_count()}, 字节数: {kb.byte_count()}")
    
    # 带验证器
    print("\n【带验证器测试】")
    validator_kb = HexKeyboardWithValidator(
        max_length=16,
        auto_pad=False,
        validators=[validate_length(4, 16), validate_even_length()]
    )
    
    validator_kb.input("ABCD")  # OK
    print("输入 'ABCD' -> 成功")
    
    validator_kb.input("AB")  # 太短
    print("输入 'AB' -> 失败（长度不足）")
    
    validator_kb.input("ABC")  # 奇数
    print("输入 'ABC' -> 失败（长度必须为偶数）")
    
    # 历史记录
    print("\n【历史记录测试】")
    kb2 = HexKeyboard()
    for i in range(5):
        kb2.input(f"00{i:02X}")
        kb2.push_history()
    
    print(f"最近3条历史: {kb2.get_history(3)}")
