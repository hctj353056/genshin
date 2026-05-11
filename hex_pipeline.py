# hex_pipeline.py
# 16进制处理流水线
# 用户输入 → 词元/解析模块 → HexMHA → 键盘输出

import os
import logging
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

# 导入各模块
from hex_mha_module_v2 import HexMHA
from hex_keyboard import HexKeyboard, KeyAction
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file


class OutputMode(Enum):
    """输出模式"""
    PRINT = 'print'       # 屏幕打印
    LOG = 'log'           # 日志记录
    FILE = 'file'         # 文件读写
    BOTH = 'both'        # 打印+日志
    ALL = 'all'          # 全部输出


class HexPipeline:
    def __init__(self,
                 mha_seq_len: int = 16,
                 mha_dim: int = 64,
                 mha_heads: int = 4,
                 mha_embed_dim: int = 64,
                 mha_mode: str = 'streaming',
                 mha_causal: bool = True,
                 output_dir: str = './pipeline_output',
                 log_file: str = None):
        """
        初始化流水线
        
        mha_*: 多头注意力参数
        output_dir: 输出目录
        log_file: 日志文件路径
        """
        # 初始化多头注意力
        self.mha = HexMHA(
            seq_len=mha_seq_len,
            dim=mha_dim,
            heads=mha_heads,
            embed_dim=mha_embed_dim,
            mode=mha_mode,
            causal=mha_causal
        )
        
        # 初始化键盘
        self.keyboard = HexKeyboard(max_buffer=mha_seq_len * 8)
        
        # 输出配置
        self.output_dir = output_dir
        self.log_file = log_file or os.path.join(output_dir, 'pipeline.log')
        self.output_mode = OutputMode.PRINT
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logger()
        
        # 处理记录
        self.process_count = 0
        
    def _setup_logger(self):
        """设置日志"""
        self.logger = logging.getLogger('HexPipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有handlers
        self.logger.handlers = []
        
        # 文件handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def _log(self, message: str, level: str = 'info'):
        """内部日志方法"""
        getattr(self.logger, level)(message)
        
        if self.output_mode in (OutputMode.LOG, OutputMode.BOTH, OutputMode.ALL):
            pass  # 已写入文件
        
        if self.output_mode in (OutputMode.PRINT, OutputMode.BOTH, OutputMode.ALL):
            print(message)
    
    def _save_result(self, hex_result: str, prefix: str = 'result') -> str:
        """保存结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.hex"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(hex_result)
        
        return filepath
    
    def set_output_mode(self, mode: Literal['print', 'log', 'file', 'both', 'all']):
        """设置输出模式"""
        self.output_mode = OutputMode(mode)
    
    def set_mha_mode(self, mode: Literal['streaming', 'cache']):
        """设置MHA模式"""
        self.mha.set_mode(mode)
        if mode == 'cache':
            self.mha.reset_cache()
    
    def reset_cache(self):
        """重置MHA缓存"""
        self.mha.reset_cache()
    
    def process_text(self, text: str, index: Optional[str] = None,
                    save_result: bool = False, reset_cache: bool = False) -> str:
        """
        处理文本输入
        
        流程：文本 → hex → MHA → 结果
        
        参数:
            text: 用户输入文本
            index: 可选的索引hex（用于关联/检索）
            save_result: 是否保存结果到文件
            reset_cache: 是否重置MHA缓存
        
        返回:
            处理后的hex字符串
        """
        self.process_count += 1
        self._log(f"【处理 #{self.process_count}】")
        
        # 1. 词元模块：文本转hex
        hex_text = str_to_hex(text)
        self._log(f"原始文本: {text}")
        self._log(f"文本hex: {hex_text}")
        
        # 2. 如果有索引，拼接索引
        if index:
            combined = index + hex_text
            self._log(f"索引hex: {index}")
            self._log(f"拼接后: {combined}")
        else:
            combined = hex_text
        
        # 3. 键盘模块处理输入
        processed = self.keyboard.input(combined)
        self._log(f"键盘处理: {processed}")
        
        # 4. MHA处理
        if reset_cache:
            self.mha.reset_cache()
        result = self.mha.forward(processed)
        self._log(f"MHA输出: {result}")
        
        # 5. 保存结果
        if save_result:
            filepath = self._save_result(result)
            self._log(f"已保存: {filepath}")
        
        self._log("-" * 50)
        return result
    
    def process_file(self, file_path: str, index: Optional[str] = None,
                    save_result: bool = False) -> str:
        """
        处理文件输入
        
        流程：文件 → hex → MHA → 结果
        
        参数:
            file_path: 文件路径
            index: 可选的索引hex
            save_result: 是否保存结果到文件
        
        返回:
            处理后的hex字符串
        """
        self.process_count += 1
        self._log(f"【文件处理 #{self.process_count}】")
        
        # 1. 解析模块：文件转hex
        hex_file = file_to_hex(file_path)
        self._log(f"文件: {file_path}")
        
        # 读取hex内容（截取与seq_len匹配的长度）
        with open(hex_file, 'r', encoding='utf-8') as f:
            hex_content = f.read()[:self.keyboard.max_buffer]
        self._log(f"文件hex长度: {len(hex_content)}")
        
        # 2. 如果有索引，拼接
        if index:
            combined = index + hex_content
            self._log(f"索引拼接: {combined[:64]}...")
        else:
            combined = hex_content
        
        # 3. 键盘模块处理
        processed = self.keyboard.input(combined)
        
        # 4. MHA处理
        result = self.mha.forward(processed)
        self._log(f"MHA输出: {result}")
        
        # 5. 保存结果
        if save_result:
            filepath = self._save_result(result, prefix='file_result')
            self._log(f"已保存: {filepath}")
        
        self._log("-" * 50)
        return result
    
    def process_hex_direct(self, hex_input: str, reset_cache: bool = False) -> str:
        """
        直接处理hex字符串（跳过词元/解析转换）
        
        参数:
            hex_input: hex字符串
            reset_cache: 是否重置缓存
        
        返回:
            处理后的hex字符串
        """
        self.process_count += 1
        self._log(f"【Hex直接处理 #{self.process_count}】")
        self._log(f"输入hex: {hex_input}")
        
        if reset_cache:
            self.mha.reset_cache()
            self._log("缓存已重置")
        
        result = self.mha.forward(hex_input, reset_cache=reset_cache)
        self._log(f"输出hex: {result}")
        self._log("-" * 50)
        return result
    
    def decode_result(self, hex_result: str) -> str:
        """将结果hex解码为文本"""
        text = hex_to_str(hex_result)
        self._log(f"解码结果: {text}")
        return text
    
    def restore_file(self, hex_result: str, output_path: str):
        """将结果hex还原为文件"""
        hex_to_file(hex_result, output_path, is_hex_file=False)
        self._log(f"文件已还原: {output_path}")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'process_count': self.process_count,
            'mha_mode': self.mha.mode,
            'mha_causal': self.mha.causal,
            'output_mode': self.output_mode.value,
            'output_dir': self.output_dir,
            'log_file': self.log_file,
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"HexPipeline(processes={stats['process_count']}, "
                f"mha_mode={stats['mha_mode']}, output={stats['output_mode']})")


# ============ 使用示例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("HexPipeline 演示")
    print("=" * 60)
    
    # 创建流水线
    pipeline = HexPipeline(
        mha_seq_len=16,
        mha_dim=64,
        mha_heads=4,
        mha_embed_dim=64,
        output_dir='./pipeline_demo'
    )
    
    # 设置输出模式
    pipeline.set_output_mode('both')
    
    print(f"\n流水线信息: {pipeline}\n")
    
    # 示例1: 处理文本
    print("\n【示例1: 文本处理】")
    result = pipeline.process_text("你好世界", save_result=True)
    
    # 示例2: 带索引的文本处理
    print("\n【示例2: 带索引的文本处理】")
    pipeline.set_mha_mode('cache')
    result2 = pipeline.process_text("Hello", index="0001", reset_cache=True)
    
    # 示例3: 增量处理（缓存模式）
    print("\n【示例3: 增量处理】")
    pipeline.mha.forward("AAAA", reset_cache=True)  # 先输入
    result3 = pipeline.mha.forward("BBBB")  # 再追加
    print(f"增量结果: {result3}")
    
    # 示例4: 解码结果
    print("\n【示例4: 解码结果】")
    decoded = pipeline.decode_result("A6F0DEAD")
    print(f"解码: {decoded}")
    
    # 打印统计
    print("\n【统计信息】")
    for k, v in pipeline.get_stats().items():
        print(f"  {k}: {v}")
