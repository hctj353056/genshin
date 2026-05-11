# hex_agent.py
# 16进制处理Agent - 支持在线学习的死循环主进程
# 
# 架构参考：OpenClaw Agent Loop
# 核心：键盘X/Y → HexMHA → 键盘X/Y输出
#
# 学习目标：让MHA学会输出能成功转换为UTF-8字符串的hex

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 导入模块
from hex_keyboard import HexKeyboard, KeyAction
from hex_mha_module_v2 import HexMHA, layer_norm, softmax, cross_entropy_loss, HEX_TO_IDX, IDX_TO_HEX
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str
from 解析模块_1778459060679_lsxn import file_to_hex, hex_to_file
from hex_category import HexCategorySystem


class AgentState(Enum):
    """Agent状态"""
    IDLE = 'idle'
    PROCESSING = 'processing'
    LEARNING = 'learning'
    ERROR = 'error'


@dataclass
class Experience:
    """经验样本"""
    input_hex: str          # 输入hex
    mha_output: str         # MHA原始输出
    is_valid: bool          # 是否能转为字符串
    target_hex: str         # 学习目标（如果无效，则学习将输出调整为有效hex）
    timestamp: float = field(default_factory=time.time)


class OnlineLearner:
    """
    在线学习器
    
    学习目标：让MHA学会"复制"输入hex到输出hex
    
    策略：
    - 每次输入hex后，期望输出 = 输入hex
    - 计算输出与输入的差异，朝着减少差异的方向调整参数
    - 如果输出能成功解码 → 成功；否则 → 学习目标=输入
    """
    
    def __init__(self, model: HexMHA, lr: float = 0.01):
        self.model = model
        self.lr = lr
        self.success_count = 0
        self.fail_count = 0
        self.total_updates = 0
        
    def record(self, input_hex: str, mha_output: str, is_valid: bool) -> str:
        """
        记录经验并学习
        
        核心：将输出朝着输入的方向调整
        """
        if is_valid:
            self.success_count += 1
            # 成功：强化这个输出模式
            target = mha_output
        else:
            self.fail_count += 1
            # 失败：学习目标是输入hex
            target = input_hex
            # 立即朝着输入方向调整
            self._learn_towards_input(input_hex, mha_output)
            
        return target
    
    def _learn_towards_input(self, target_hex: str, current_hex: str):
        """
        将输出朝着输入方向调整
        
        策略：找到classifier中与target_hex匹配度更高的权重方向
        """
        target_len = min(len(target_hex), len(current_hex))
        if target_len == 0:
            return
            
        # 简单策略：调整classifier，朝着target的方向
        # classifier是 (embed_dim, 16)，输出16个hex字符的概率
        target_indices = [HEX_TO_IDX.get(c, 0) for c in target_hex]
        
        # 找到当前输出与目标的差异
        current_indices = [HEX_TO_IDX.get(c, 0) for c in current_hex[:target_len]]
        
        # 朝着目标调整（增加目标字符的权重，减少其他字符）
        for pos, (tgt_idx, cur_idx) in enumerate(zip(target_indices, current_indices)):
            if tgt_idx != cur_idx:
                # 找到输出层中对应这个位置的权重
                # classifier的形状是 (embed_dim, 16)
                # 对于位置pos，增加tgt_idx对应的权重，减少cur_idx对应的
                noise = np.random.randn(16).astype(np.float32) * self.lr * 0.1
                noise[tgt_idx] += self.lr * 0.5  # 增强目标
                noise[cur_idx] -= self.lr * 0.3  # 削弱当前
                
                # 只对随机选择的行应用（避免全部更新导致崩溃）
                row_idx = np.random.randint(0, self.model.classifier.shape[0])
                self.model.classifier[row_idx] += noise
                
        self.total_updates += 1
        
    def get_stats(self) -> dict:
        total = self.success_count + self.fail_count
        return {
            'success': self.success_count,
            'fail': self.fail_count,
            'rate': self.success_count / max(1, total) if total > 0 else 0,
            'updates': self.total_updates
        }


class HexAgent:
    """
    Hex处理Agent
    
    核心流程：
    输入 → 键盘X(操作模式) + Y(hex数据) → HexMHA → 尝试转为字符串 → 键盘X输出
    
    X轴定义操作模式：
    - PRINT: 输出转为字符串打印
    - ECHO: 输出hex
    - SAVE: 保存到文件
    - LOG: 记录日志
    
    学习闭环：
    1. MHA输出hex
    2. 尝试hex_to_str转换
    3. 成功 → 打印字符串 + 记录成功经验
    4. 失败 → 记录失败样本 + 学习目标=输入hex
    """
    
    def __init__(self,
                 mha_seq_len: int = 4096,
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
            output_dir=os.path.join(state_dir, 'keyboard_output')
        )
        
        # 在线学习器
        self.learner = OnlineLearner(self.mha, lr=learning_rate) if enable_online_learning else None
        self.enable_learning = enable_online_learning
        
        # 范畴系统（用于语言规则提取）
        self.category_system = HexCategorySystem()
        self.conversation_history: List[str] = []  # 对话历史
        
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
        
        # 如果没有加载到保存的状态，进行预训练
        state_file = os.path.join(state_dir, 'model_state.npz')
        if not os.path.exists(state_file):
            print(f"   正在进行预训练...")
            self._pretrain()
        
        print(f"✅ HexAgent 初始化完成")
        print(f"   在线学习: {'启用' if self.enable_learning else '禁用'}")
        print(f"   学习率: {learning_rate}")
    
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
            try:
                data = np.load(state_file)
                self.mha.token_embed = data['token_embed']
                self.mha.pos_embed = data['pos_embed']
                self.mha.Wq = data['Wq']
                self.mha.Wk = data['Wk']
                self.mha.Wv = data['Wv']
                self.mha.Wo = data['Wo']
                self.mha.classifier = data['classifier']
                self.logger.info("已加载保存的状态")
                print(f"✅ 已加载保存的模型状态")
            except Exception as e:
                self.logger.warning(f"加载状态失败: {e}")
    
    def _pretrain(self):
        """预训练：让MHA学会'复制'输入hex到输出"""
        print(f"   预训练: 学习复制输入...")
        
        # 训练样本
        samples = [
            "你好", "hello", "你好世界", "ABC",
            "test", "123", "hello world", "你好123"
        ]
        
        for text in samples:
            hex_str = str_to_hex(text).upper().replace(' ', '')
            for _ in range(10):  # 每个样本训练10次
                output = self.mha.forward(hex_str, reset_cache=True)
                if output == hex_str:
                    break
                # 学习：朝着目标调整
                self._learn_towards_target(hex_str, output)
        
        print(f"   预训练完成!")
        
    def _learn_towards_target(self, target_hex: str, current_hex: str):
        """将输出调整为等于目标hex"""
        target_len = min(len(target_hex), len(current_hex))
        for pos in range(target_len):
            tgt_char = target_hex[pos]
            tgt_idx = HEX_TO_IDX.get(tgt_char, 0)
            cur_char = current_hex[pos] if pos < len(current_hex) else None
            
            if cur_char != tgt_char:
                noise = np.zeros(16, dtype=np.float32)
                noise[tgt_idx] += 0.5
                for _ in range(3):
                    row_idx = np.random.randint(0, self.mha.classifier.shape[0])
                    self.mha.classifier[row_idx] += noise
    
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
            print(f"✅ 状态已保存")
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
            print(f"❌ 保存失败: {e}")
    
    def _try_decode(self, hex_str: str) -> Tuple[bool, str]:
        """
        尝试将hex解码为字符串
        
        Returns:
            (是否成功, 解码结果/错误信息)
        """
        try:
            text = bytes.fromhex(hex_str).decode('utf-8')
            return True, text
        except (ValueError, UnicodeDecodeError) as e:
            return False, str(e)
    
    def process(self, input_text: str, output_mode: KeyAction = KeyAction.PRINT) -> dict:
        """
        处理输入
        
        流程：文本 → hex → 范畴系统生成回复 → HexMHA处理 → 尝试转字符串 → 输出
        
        Args:
            input_text: 用户输入文本
            output_mode: 输出模式
        
        Returns:
            处理结果字典
        """
        self.state = AgentState.PROCESSING
        result = {
            'success': False,
            'input_text': input_text,
            'input_hex': '',
            'category_output': '',
            'mha_output': '',
            'is_valid': False,
            'output_text': '',
            'output_hex': '',
            'error': None
        }
        
        try:
            # 1. 词元模块：文本转hex
            input_hex = str_to_hex(input_text).upper().replace(' ', '')
            result['input_hex'] = input_hex
            self.logger.info(f"输入: {input_text}")
            self.logger.info(f"hex: {input_hex}")
            
            # 2. 范畴系统：学习输入 + 生成回复（中文屋子核心）
            # 学习历史输入到范畴系统
            self.conversation_history.append(input_hex)
            self.category_system.learn_from_input(input_hex, self.conversation_history[:-1])
            
            # 使用范畴系统生成回复（不是复制输入，而是找关系）
            category_output = self.category_system.generate_response(input_hex, mode="chain")
            result['category_output'] = category_output
            self.logger.info(f"范畴输出: {category_output}")
            
            # 3. HexMHA处理范畴输出
            mha_output = self.mha.forward(category_output, reset_cache=True)
            result['mha_output'] = mha_output
            self.logger.info(f"MHA输出: {mha_output}")
            
            # 4. 尝试转为字符串
            is_valid, decoded = self._try_decode(mha_output)
            result['is_valid'] = is_valid
            
            # 5. 根据模式输出
            if output_mode == KeyAction.PRINT:
                if is_valid:
                    result['success'] = True
                    result['output_text'] = decoded
                    print(f"\n{'='*50}")
                    print(f"🎯 输出(字符串): {decoded}")
                    print(f"{'='*50}")
                    self.logger.info(f"输出成功: {decoded}")
                else:
                    print(f"\n{'='*50}")
                    print(f"⚠️  MHA输出无法转为字符串")
                    print(f"   原始hex: {mha_output}")
                    print(f"   错误: {decoded}")
                    print(f"{'='*50}")
                    self.logger.warning(f"输出无效: {decoded}")
            
            elif output_mode == KeyAction.ECHO:
                result['success'] = True
                result['output_hex'] = mha_output
                print(f"\n[hex] {mha_output}")
            
            self.processed_count += 1
            self.state = AgentState.IDLE
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"处理错误: {e}")
            result['error'] = str(e)
            self.state = AgentState.IDLE
            return result
    
    def run_interactive(self):
        """交互式运行"""
        print("\n" + "="*60)
        print("HexAgent - 交互模式")
        print("="*60)
        print("命令:")
        print("  :learn on/off   - 开启/关闭在线学习")
        print("  :save           - 保存状态")
        print("  :stats          - 显示统计")
        print("  :reset          - 重置缓存")
        print("  :mode print     - 输出模式:转字符串(默认)")
        print("  :mode echo      - 输出模式:直接hex")
        print("  :test           - 运行测试用例")
        print("  :quit           - 退出")
        print("="*60)
        
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
                        if self.learner:
                            self.learner = OnlineLearner(self.mha, lr=self.learner.lr)
                        print("[learn] ✅ 在线学习已开启")
                    elif user_input == ':learn off':
                        self.enable_learning = False
                        print("[learn] ⛔ 在线学习已关闭")
                    elif user_input == ':save':
                        self._save_state()
                    elif user_input == ':stats':
                        self._show_stats()
                    elif user_input == ':reset':
                        self.mha.reset_cache()
                        print("[reset] ✅ 缓存已重置")
                    elif user_input == ':mode print':
                        current_mode = KeyAction.PRINT
                        print("[mode] PRINT - hex转字符串")
                    elif user_input == ':mode echo':
                        current_mode = KeyAction.ECHO
                        print("[mode] ECHO - 直接hex")
                    elif user_input == ':test':
                        self._run_tests()
                    continue
                
                # 处理普通输入
                self.process(user_input, current_mode)
                
            except KeyboardInterrupt:
                print("\n\n正在退出...")
                self.running = False
            except EOFError:
                break
        
        self._save_state()
        print("\nAgent已停止")
    
    def _show_stats(self):
        uptime = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"📊 Agent统计")
        print(f"{'='*50}")
        print(f"运行时间: {uptime:.1f}秒")
        print(f"处理次数: {self.processed_count}")
        print(f"当前状态: {self.state.value}")
        print(f"在线学习: {'启用' if self.enable_learning else '禁用'}")
        if self.learner:
            stats = self.learner.get_stats()
            print(f"学习统计:")
            print(f"  - 成功: {stats['success']}")
            print(f"  - 失败: {stats['fail']}")
            print(f"  - 成功率: {stats['rate']:.1%}")
            print(f"  - 更新次数: {stats['updates']}")
        print(f"{'='*50}")
    
    def _run_tests(self):
        """运行测试用例"""
        print(f"\n{'='*50}")
        print(f"🧪 运行测试用例")
        print(f"{'='*50}")
        
        test_cases = [
            "你好",
            "Hello",
            "你好世界",
            "ABC",
            "123",
        ]
        
        for text in test_cases:
            print(f"\n测试: {text}")
            result = self.process(text, KeyAction.PRINT)
            status = "✅" if result['is_valid'] else "❌"
            print(f"  结果: {status}")
            if result['is_valid']:
                print(f"  输出: {result['output_text']}")
            else:
                print(f"  MHA输出: {result['mha_output']}")
        
        print(f"\n{'='*50}")
    
    def get_stats(self) -> dict:
        stats = {
            'state': self.state.value,
            'processed': self.processed_count,
            'uptime': time.time() - self.start_time,
            'learning': self.enable_learning
        }
        if self.learner:
            stats['learner'] = self.learner.get_stats()
        return stats


# ============ 主入口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HexAgent - 16进制处理Agent')
    parser.add_argument('--mode', choices=['interactive'], default='interactive')
    parser.add_argument('--no-learning', action='store_true', help='禁用在线学习')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--state-dir', type=str, default='./genshin_state', help='状态保存目录')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HexAgent 启动中...")
    print(f"{'='*60}")
    
    agent = HexAgent(
        enable_online_learning=not args.no_learning,
        learning_rate=args.lr,
        state_dir=args.state_dir
    )
    
    print(f"\nAgent: {agent}")
    print(f"{'='*60}\n")
    
    agent.run_interactive()
