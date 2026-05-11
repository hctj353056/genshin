# hex_agent.py
# 16进制处理Agent - 支持在线学习的死循环主进程
# 
# 架构参考：OpenClaw Agent Loop
# 核心：键盘X/Y → HexMHA → 键盘X/Y输出

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Callable, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 导入模块
from hex_keyboard import HexKeyboard, KeyAction
from hex_mha_module_v2 import HexMHA, layer_norm, softmax, cross_entropy_loss, HEX_TO_IDX
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file


class AgentState(Enum):
    """Agent状态"""
    IDLE = 'idle'
    PROCESSING = 'processing'
    LEARNING = 'learning'
    ERROR = 'error'


@dataclass
class Experience:
    """经验样本"""
    input_hex: str
    target_hex: str
    timestamp: float = field(default_factory=time.time)


class OnlineLearner:
    """在线学习器"""
    
    def __init__(self, model: HexMHA, lr: float = 0.01):
        self.model = model
        self.lr = lr
        self.experience_buffer: list[Experience] = []
        self.batch_size = 4
        self.update_interval = 10
        self.total_updates = 0
        
    def add_experience(self, input_hex: str, target_hex: str):
        self.experience_buffer.append(Experience(input_hex, target_hex))
        
        if len(self.experience_buffer) >= self.update_interval:
            self.update()
    
    def update(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        
        batch = self.experience_buffer[-self.batch_size:]
        total_loss = 0.0
        
        for exp in batch:
            try:
                x = self.model._embed(exp.input_hex)
                L = x.shape[0]
                logits = x @ self.model.classifier
                tgt = [HEX_TO_IDX.get(c, 0) for c in exp.target_hex[:L]]
                if len(tgt) < L:
                    tgt += [0] * (L - len(tgt))
                loss = cross_entropy_loss(logits, np.array(tgt[:L]))
                total_loss += loss
            except:
                continue
        
        avg_loss = total_loss / len(batch)
        noise_scale = self.lr * avg_loss
        self.model.classifier += np.random.randn(*self.model.classifier.shape).astype(np.float32) * noise_scale
        self.total_updates += 1
        
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]
        
        return avg_loss


class HexAgent:
    """
    Hex处理Agent
    
    核心流程：
    输入 → 键盘(X/Y) → HexMHA → 键盘(X/Y)输出
    
    X轴定义操作模式：
    - PRINT: 输出转为字符串
    - ECHO: 输出hex
    - SAVE: 保存到文件
    - LOG: 记录日志
    """
    
    def __init__(self,
                 mha_seq_len: int = 16,
                 mha_dim: int = 64,
                 mha_heads: int = 4,
                 mha_embed_dim: int = 64,
                 learning_rate: float = 0.01,
                 enable_online_learning: bool = True,
                 state_dir: str = './genshin_state'):
        
        # 初始化HexMHA
        self.mha = HexMHA(
            seq_len=mha_seq_len,
            dim=mha_dim,
            heads=mha_heads,
            embed_dim=mha_embed_dim,
            mode='cache',
            causal=True
        )
        
        # 初始化键盘（X/Y架构）
        self.keyboard = HexKeyboard(
            output_dir=os.path.join(state_dir, 'keyboard_output'),
            auto_convert=True  # 自动hex转字符串
        )
        
        # 键盘动作回调
        self.keyboard.on_action = self._on_keyboard_action
        
        # 在线学习器
        self.learner = OnlineLearner(self.mha, lr=learning_rate) if enable_online_learning else None
        self.enable_learning = enable_online_learning
        
        # 状态
        self.state = AgentState.IDLE
        self.state_dir = state_dir
        self.running = False
        self.processed_count = 0
        self.start_time = time.time()
        
        # 日志
        os.makedirs(state_dir, exist_ok=True)
        self._setup_logger()
        self._load_state()
    
    def _setup_logger(self):
        log_file = os.path.join(self.state_dir, 'agent.log')
        self.logger = logging.getLogger('HexAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(ch)
    
    def _load_state(self):
        state_file = os.path.join(self.state_dir, 'model_state.npz')
        if os.path.exists(state_file):
            self.logger.info("已加载保存的状态")
    
    def _save_state(self):
        state_file = os.path.join(self.state_dir, 'model_state.npz')
        try:
            np.savez(state_file,
                    token_embed=self.mha.token_embed,
                    pos_embed=self.mha.pos_embed,
                    Wq=self.mha.Wq,
                    Wk=self.mha.Wk,
                    Wv=self.mha.Wv,
                    Wo=self.mha.Wo,
                    classifier=self.mha.classifier)
            self.logger.info("状态已保存")
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def _on_keyboard_action(self, action: KeyAction, data: str, result: str):
        """键盘动作回调"""
        self.logger.info(f"键盘: X={action.value}, Y={data[:32]}...")
    
    def process(self, input_text: str, x_action: KeyAction = KeyAction.PRINT) -> str:
        """
        处理输入
        
        流程：文本 → hex → HexMHA → 输出(X,Y)
        
        Args:
            input_text: 用户输入文本
            x_action: 输出模式（默认PRINT，转字符串）
        
        Returns:
            处理后的字符串结果
        """
        self.state = AgentState.PROCESSING
        
        try:
            # 1. 词元模块：文本转hex
            input_hex = str_to_hex(input_text).upper().replace(' ', '')
            self.logger.info(f"输入: {input_text} → hex: {input_hex[:32]}...")
            
            # 2. HexMHA处理
            output_hex = self.mha.forward(input_hex)
            self.logger.info(f"MHA输出: {output_hex}")
            
            # 3. 通过键盘X/Y输出
            result = self._output(x_action, output_hex)
            
            # 4. 在线学习
            if self.enable_learning and self.learner:
                self.state = AgentState.LEARNING
                self.learner.add_experience(input_hex, output_hex)
            
            self.processed_count += 1
            self.state = AgentState.IDLE
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"处理错误: {e}")
            self.state = AgentState.IDLE
            return f"[ERROR] {e}"
    
    def _output(self, action: KeyAction, hex_data: str) -> str:
        """
        通过键盘X/Y输出
        
        X=PRINT: hex转字符串后输出
        X=ECHO: 直接输出hex
        X=SAVE: 保存到文件
        X=LOG: 记录日志
        """
        if action == KeyAction.PRINT:
            # hex转字符串后打印
            readable = hex_to_str(hex_data)
            output = f">>> {readable}"
            print(output)
            self.logger.info(f"输出(字符串): {readable}")
            return output
        
        elif action == KeyAction.ECHO:
            # 直接输出hex
            output = f"[hex] {hex_data}"
            print(output)
            self.logger.info(f"输出(hex): {hex_data}")
            return output
        
        elif action == KeyAction.SAVE:
            # 保存文件
            return self.keyboard.input(f":SAVE:{hex_data}")
        
        elif action == KeyAction.LOG:
            # 记录日志
            return self.keyboard.input(f":LOG:{hex_data}")
        
        else:
            # 默认PRINT
            return self._output(KeyAction.PRINT, hex_data)
    
    def run_interactive(self):
        """交互式运行"""
        print("=" * 60)
        print("HexAgent - 交互模式")
        print("=" * 60)
        print("命令:")
        print("  :learn on/off  - 开启/关闭在线学习")
        print("  :save          - 保存状态")
        print("  :stats         - 显示统计")
        print("  :reset         - 重置缓存")
        print("  :mode print    - 输出模式:转字符串")
        print("  :mode echo     - 输出模式:hex")
        print("  :quit          - 退出")
        print("=" * 60)
        
        self.running = True
        current_mode = KeyAction.PRINT
        
        while self.running:
            try:
                user_input = input("\n[输入] ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith(':'):
                    if user_input == ':quit':
                        self.running = False
                    elif user_input == ':learn on':
                        self.enable_learning = True
                        print("[learn] ON")
                    elif user_input == ':learn off':
                        self.enable_learning = False
                        print("[learn] OFF")
                    elif user_input == ':save':
                        self._save_state()
                    elif user_input == ':stats':
                        self._show_stats()
                    elif user_input == ':reset':
                        self.mha.reset_cache()
                        print("[reset] OK")
                    elif user_input == ':mode print':
                        current_mode = KeyAction.PRINT
                        print("[mode] PRINT (hex转字符串)")
                    elif user_input == ':mode echo':
                        current_mode = KeyAction.ECHO
                        print("[mode] ECHO (直接hex)")
                    continue
                
                # 处理普通输入
                result = self.process(user_input, current_mode)
                
            except KeyboardInterrupt:
                print("\n\n正在退出...")
                self.running = False
            except EOFError:
                break
        
        self._save_state()
        print("Agent已停止")
    
    def _show_stats(self):
        uptime = time.time() - self.start_time
        print(f"\n=== Agent统计 ===")
        print(f"运行时间: {uptime:.1f}秒")
        print(f"处理次数: {self.processed_count}")
        print(f"当前状态: {self.state.value}")
        print(f"在线学习: {'启用' if self.enable_learning else '禁用'}")
        if self.learner:
            print(f"学习更新: {self.learner.total_updates}次")
        print("=" * 20)
    
    def get_stats(self) -> dict:
        return {
            'state': self.state.value,
            'processed': self.processed_count,
            'uptime': time.time() - self.start_time,
            'learning': self.enable_learning
        }


# ============ 主入口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HexAgent')
    parser.add_argument('--mode', choices=['interactive'], default='interactive')
    parser.add_argument('--no-learning', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    
    agent = HexAgent(
        enable_online_learning=not args.no_learning,
        learning_rate=args.lr
    )
    
    print(f"\n{'='*60}")
    print(f"HexAgent 初始化完成")
    print(f"{'='*60}")
    print(f"Agent: {agent}")
    print(f"{'='*60}\n")
    
    agent.run_interactive()
