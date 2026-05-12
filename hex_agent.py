# hex_agent.py
# 16进制处理Agent - 中文屋子实现
# 
# 架构：
# 用户输入 → 解析→hex → 词元→hex → 范畴系统学习+生成 → 键盘X/Y输出

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, List
from enum import Enum
import numpy as np

# 导入模块
from hex_keyboard import HexKeyboard, KeyAction
from hex_mha_module_v2 import HexMHA, HEX_TO_IDX, IDX_TO_HEX
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file
from hex_category import HexCategorySystem


class AgentState(Enum):
    """Agent状态"""
    IDLE = 'idle'
    PROCESSING = 'processing'
    LEARNING = 'learning'
    ERROR = 'error'


class HexAgent:
    """
    Hex处理Agent - 中文屋子实现
    
    流程：
    1. 解析模块：任意→hex（文件）
    2. 词元模块：str→hex（文本）
    3. 范畴系统：学习输入 + 生成回复
    4. 键盘模块：处理输入（X,Y）→ 输出
    
    X轴模式：
    - PRINT: 打印字符串
    - ECHO: 打印hex
    - SAVE: 保存文件
    """
    
    def __init__(self, state_dir: str = './genshin_state'):
        # 范畴系统（核心）
        self.category_system = HexCategorySystem(data_dir=os.path.join(state_dir, 'category_data'))
        
        # 键盘模块
        self.keyboard = HexKeyboard(output_dir=os.path.join(state_dir, 'keyboard_output'))
        
        # 状态
        self.state = AgentState.IDLE
        self.state_dir = state_dir
        self.running = False
        self.processed_count = 0
        self.start_time = time.time()
        
        # 对话历史
        self.conversation_history: List[Tuple[str, str]] = []
        
        # 日志
        os.makedirs(state_dir, exist_ok=True)
        self._setup_logger()
        
        print(f"✅ HexAgent 初始化完成")
        print(f"   范畴系统: {self.category_system.get_stats()}")
    
    def _setup_logger(self):
        log_file = os.path.join(self.state_dir, 'agent.log')
        self.logger = logging.getLogger('HexAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(ch)
    
    def _try_decode(self, hex_str: str) -> Tuple[bool, str]:
        """尝试将hex解码为字符串"""
        try:
            text = bytes.fromhex(hex_str).decode('utf-8')
            return True, text
        except:
            return False, hex_str
    
    def process(self, input_text: str, output_mode: KeyAction = KeyAction.PRINT) -> dict:
        """
        处理输入
        
        流程：输入 → 范畴系统学习 → 生成回复 → 输出
        """
        self.state = AgentState.PROCESSING
        result = {
            'input_text': input_text,
            'response': '',
            'mode': output_mode.value,
            'success': True
        }
        
        try:
            self.logger.info(f"输入: {input_text}")
            
            # 使用自主学习方法
            # 1. 自动学习 + 分析 + 推断
            inferred = self.category_system.auto_learn(input_text)
            
            # 2. 范畴系统生成回复
            response = self.category_system.generate_response(input_text, mode="similar")
            result['response'] = response
            self.logger.info(f"范畴生成: {response}")
            
            # 3. 学习这个对话对
            self.conversation_history.append((input_text, response))
            self.category_system.add_morphism(input_text, response)
            
            # 4. 键盘模块处理输出
            if output_mode == KeyAction.PRINT:
                # 尝试解码为字符串
                is_valid, decoded = self._try_decode(response.upper().replace(' ', ''))
                if is_valid:
                    print(f"\n{'='*50}")
                    print(f"🎯 {decoded}")
                    print(f"{'='*50}")
                else:
                    print(f"\n{'='*50}")
                    print(f"⚠️  {response}")
                    print(f"{'='*50}")
                    
            elif output_mode == KeyAction.ECHO:
                print(f"\n[hex] {str_to_hex(response).upper()}")
            
            # 5. 范畴系统学习程序输出
            self.category_system.learn_from_program_output(response)
            
            self.processed_count += 1
            self.state = AgentState.IDLE
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"处理错误: {e}")
            result['success'] = False
            result['error'] = str(e)
            self.state = AgentState.IDLE
            return result
    
    def teach(self, user_input: str, program_output: str):
        """
        手动教学：学习一个对话对
        
        用法: :teach 你好 → 你好
        """
        self.category_system.learn_from_user_input(user_input)
        self.category_system.learn_from_program_output(program_output)
        self.category_system.add_morphism(user_input, program_output)
        print(f"✅ 已学习: '{user_input}' → '{program_output}'")
    
    def run_interactive(self):
        """交互式运行"""
        print("\n" + "="*60)
        print("HexAgent - 交互模式 (中文屋子)")
        print("="*60)
        print("命令:")
        print("  :teach 你好 → 你好  - 教学一个对话对")
        print("  :stats               - 显示统计")
        print("  :history             - 显示对话历史")
        print("  :reset               - 重置对话历史")
        print("  :quit                - 退出")
        print("="*60)
        
        self.running = True
        
        while self.running:
            try:
                user_input = input("\n[输入] ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith(':'):
                    if user_input == ':quit':
                        self.running = False
                    elif user_input.startswith(':teach '):
                        # 解析教学命令
                        parts = user_input[7:].split('→')
                        if len(parts) == 2:
                            self.teach(parts[0].strip(), parts[1].strip())
                        else:
                            print("用法: :teach 你好 → 你好")
                    elif user_input == ':stats':
                        self._show_stats()
                    elif user_input == ':history':
                        self._show_history()
                    elif user_input == ':reset':
                        self.conversation_history.clear()
                        print("[reset] 对话历史已清空")
                    continue
                
                # 处理普通输入
                self.process(user_input)
                
            except KeyboardInterrupt:
                print("\n\n正在退出...")
                self.running = False
            except EOFError:
                break
        
        print("\nAgent已停止")
    
    def _show_stats(self):
        """显示统计"""
        stats = self.category_system.get_stats()
        print(f"\n{'='*50}")
        print(f"📊 范畴系统统计")
        print(f"{'='*50}")
        print(f"用户输入集合: {stats['user_inputs']} 个句子")
        print(f"程序输出集合: {stats['program_outputs']} 个句子")
        print(f"范畴映射: {stats['morphisms']} 条")
        print(f"句式模板: {stats['patterns']} 个")
        print(f"{'='*50}")
    
    def _show_history(self):
        """显示对话历史"""
        print(f"\n{'='*50}")
        print(f"📜 对话历史")
        print(f"{'='*50}")
        for i, (user, agent) in enumerate(self.conversation_history[-10:], 1):
            print(f"{i}. 用户: {user}")
            print(f"   Agent: {agent}")
        print(f"{'='*50}")


# ============ 主入口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HexAgent - 中文屋子实现')
    parser.add_argument('--state-dir', type=str, default='./genshin_state', help='状态目录')
    args = parser.parse_args()
    
    agent = HexAgent(state_dir=args.state_dir)
    agent.run_interactive()
