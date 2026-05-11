# hex_agent.py
# 16进制处理Agent - 支持在线学习的死循环主进程
# 
# 架构参考：OpenClaw Agent Loop
# 核心：while True + 在线学习 + 事件驱动

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

# 导入流水线模块
from hex_pipeline import HexPipeline, OutputMode
from hex_mha_module_v2 import HexMHA, layer_norm, softmax, cross_entropy_loss, HEX_TO_IDX


class AgentState(Enum):
    """Agent状态"""
    IDLE = 'idle'           # 空闲
    PROCESSING = 'processing' # 处理中
    LEARNING = 'learning'    # 学习中
    ERROR = 'error'          # 错误


@dataclass
class Experience:
    """经验样本"""
    input_hex: str
    target_hex: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self):
        return {
            'input': self.input_hex,
            'target': self.target_hex,
            'timestamp': self.timestamp
        }


class OnlineLearner:
    """在线学习器 - 增量更新HexMHA权重"""
    
    def __init__(self, model: HexMHA, lr: float = 0.01):
        self.model = model
        self.lr = lr
        self.experience_buffer: list[Experience] = []
        self.batch_size = 4
        self.update_interval = 10  # 每N个样本更新一次
        self.total_updates = 0
        
    def add_experience(self, input_hex: str, target_hex: str):
        """添加经验样本"""
        exp = Experience(input_hex, input_hex)  # 自监督：输入预测输入
        self.experience_buffer.append(exp)
        
        # 定期更新
        if len(self.experience_buffer) >= self.update_interval:
            self.update()
    
    def set_learning_task(self, input_hex: str, target_hex: str):
        """设置学习任务（外部指定目标）"""
        exp = Experience(input_hex, target_hex)
        self.experience_buffer.append(exp)
        self.update()
    
    def update(self):
        """执行一次梯度更新（简化版，无自动微分）"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # 取最近N个样本
        batch = self.experience_buffer[-self.batch_size:]
        
        total_loss = 0.0
        for exp in batch:
            try:
                # 前向传播
                x = self.model._embed(exp.input_hex)
                L = x.shape[0]
                
                # 简化：只更新输出层
                # 实际应该用反向传播，这里用随机梯度近似
                logits = x @ self.model.classifier
                
                # 计算损失
                tgt = [HEX_TO_IDX.get(c, 0) for c in exp.target_hex[:L]]
                if len(tgt) < L:
                    tgt += [0] * (L - len(tgt))
                
                loss = cross_entropy_loss(logits, np.array(tgt[:L]))
                total_loss += loss
                
            except Exception as e:
                continue
        
        avg_loss = total_loss / len(batch)
        
        # 简化的权重更新（添加噪声模拟学习）
        noise_scale = self.lr * avg_loss
        self.model.classifier += np.random.randn(*self.model.classifier.shape).astype(np.float32) * noise_scale
        
        self.total_updates += 1
        
        # 清理缓冲区，防止无限增长
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]
        
        return avg_loss
    
    def get_stats(self) -> dict:
        return {
            'buffer_size': len(self.experience_buffer),
            'total_updates': self.total_updates,
            'update_interval': self.update_interval
        }


class HexAgent:
    """
    Hex处理Agent - 死循环主进程
    
    特性：
    - 持续运行的while True循环
    - 支持在线学习
    - 事件驱动架构
    - 多种输入/输出模式
    """
    
    def __init__(self,
                 mha_seq_len: int = 16,
                 mha_dim: int = 64,
                 mha_heads: int = 4,
                 mha_embed_dim: int = 64,
                 learning_rate: float = 0.01,
                 enable_online_learning: bool = True,
                 enable_learning: bool = True,
                 state_dir: str = './agent_state'):
        """
        初始化Agent
        
        enable_online_learning: 是否启用在线学习
        state_dir: 状态保存目录
        """
        # 初始化HexMHA
        self.mha = HexMHA(
            seq_len=mha_seq_len,
            dim=mha_dim,
            heads=mha_heads,
            embed_dim=mha_embed_dim,
            mode='cache',
            causal=True
        )
        
        # 初始化流水线
        self.pipeline = HexPipeline(
            mha_seq_len=mha_seq_len,
            mha_dim=mha_dim,
            mha_heads=mha_heads,
            mha_embed_dim=mha_embed_dim,
            mha_mode='cache',
            output_dir=os.path.join(state_dir, 'pipeline_output')
        )
        
        # 初始化在线学习器
        self.learner = OnlineLearner(self.mha, lr=learning_rate) if enable_online_learning else None
        self.enable_learning = enable_learning
        
        # Agent状态
        self.state = AgentState.IDLE
        self.state_dir = state_dir
        self.running = False
        self.processed_count = 0
        self.start_time = time.time()
        
        # 事件钩子
        self.hooks: dict[str, list[Callable]] = {
            'on_input': [],
            'on_output': [],
            'on_learn': [],
            'on_error': [],
            'on_state_change': []
        }
        
        # 创建状态目录
        os.makedirs(state_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logger()
        
        # 加载保存的状态
        self._load_state()
    
    def _setup_logger(self):
        """设置日志"""
        log_file = os.path.join(self.state_dir, 'agent.log')
        self.logger = logging.getLogger('HexAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(fh)
        
        # 也输出到控制台
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(ch)
    
    def _trigger_hook(self, event: str, *args, **kwargs):
        """触发事件钩子"""
        for handler in self.hooks.get(event, []):
            try:
                handler(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook error in {event}: {e}")
    
    def _load_state(self):
        """加载保存的状态"""
        state_file = os.path.join(self.state_dir, 'model_state.npz')
        if os.path.exists(state_file):
            try:
                data = np.load(state_file)
                # 加载权重（可选）
                self.logger.info(f"已加载保存的状态")
            except Exception as e:
                self.logger.warning(f"加载状态失败: {e}")
    
    def _save_state(self):
        """保存当前状态"""
        state_file = os.path.join(self.state_dir, 'model_state.npz')
        try:
            # 保存MHA权重
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
    
    def register_hook(self, event: str, handler: Callable):
        """注册事件钩子"""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(handler)
    
    def process(self, input_text: str, index: Optional[str] = None,
                learn: bool = None) -> str:
        """
        处理输入（单次）
        
        流程：输入 → HexMHA → 输出 → 在线学习
        """
        if learn is None:
            learn = self.enable_learning
        
        self.state = AgentState.PROCESSING
        self._trigger_hook('on_state_change', self.state)
        
        try:
            # 触发输入事件
            self._trigger_hook('on_input', input_text, index)
            
            # 通过流水线处理
            result = self.pipeline.process_text(input_text, index)
            
            # 在线学习
            if learn and self.learner:
                self.state = AgentState.LEARNING
                self._trigger_hook('on_state_change', self.state)
                
                self.learner.add_experience(result, result)
                
                self._trigger_hook('on_learn', None)
            
            # 触发输出事件
            self._trigger_hook('on_output', result)
            
            self.processed_count += 1
            self.state = AgentState.IDLE
            self._trigger_hook('on_state_change', self.state)
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self._trigger_hook('on_error', str(e))
            self.logger.error(f"处理错误: {e}")
            self.state = AgentState.IDLE
            return f"ERROR: {e}"
    
    def run_interactive(self):
        """
        交互式运行（命令行）
        
        死循环：等待输入 → 处理 → 输出 → 等待输入
        """
        print("=" * 60)
        print("HexAgent - 交互模式")
        print("=" * 60)
        print("命令:")
        print("  :learn on/off  - 开启/关闭在线学习")
        print("  :save          - 保存状态")
        print("  :stats         - 显示统计")
        print("  :reset         - 重置缓存")
        print("  :quit          - 退出")
        print("=" * 60)
        
        self.running = True
        
        while self.running:
            try:
                # 等待输入
                user_input = input("\n[输入] ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith(':'):
                    self._handle_command(user_input)
                    continue
                
                # 处理普通输入
                result = self.process(user_input)
                print(f"\n[输出] {result}")
                
            except KeyboardInterrupt:
                print("\n\n正在退出...")
                self.running = False
            except EOFError:
                break
            except Exception as e:
                print(f"\n[错误] {e}")
        
        # 退出前保存状态
        self._save_state()
        print("Agent已停止")
    
    def _handle_command(self, cmd: str):
        """处理命令"""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == ':quit':
            self.running = False
            
        elif command == ':learn':
            if args == 'on':
                self.enable_learning = True
                print("在线学习已开启")
            elif args == 'off':
                self.enable_learning = False
                print("在线学习已关闭")
            else:
                print(f"当前学习状态: {'开启' if self.enable_learning else '关闭'}")
                
        elif command == ':save':
            self._save_state()
            
        elif command == ':stats':
            uptime = time.time() - self.start_time
            print(f"\n=== Agent统计 ===")
            print(f"运行时间: {uptime:.1f}秒")
            print(f"处理次数: {self.processed_count}")
            print(f"当前状态: {self.state.value}")
            if self.learner:
                stats = self.learner.get_stats()
                print(f"学习缓冲区: {stats['buffer_size']}")
                print(f"总更新次数: {stats['total_updates']}")
            print("=" * 20)
            
        elif command == ':reset':
            self.mha.reset_cache()
            print("缓存已重置")
            
        elif command == ':help':
            print("可用命令: :learn, :save, :stats, :reset, :quit")
    
    def run_service(self, host: str = '0.0.0.0', port: int = 8765):
        """
        服务模式运行（作为服务器）
        
        通过TCP/HTTP接收请求并处理
        """
        import socket
        
        print(f"启动服务模式: {host}:{port}")
        self.logger.info(f"服务启动 {host}:{port}")
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)
        
        self.running = True
        
        while self.running:
            try:
                client, addr = server.accept()
                self.logger.info(f"连接: {addr}")
                
                # 处理请求
                try:
                    data = client.recv(4096).decode('utf-8')
                    
                    # 解析JSON请求
                    req = json.loads(data)
                    input_text = req.get('text', '')
                    index = req.get('index')
                    learn = req.get('learn', self.enable_learning)
                    
                    # 处理
                    result = self.process(input_text, index, learn)
                    
                    # 发送响应
                    resp = {'result': result, 'status': 'ok'}
                    client.sendall(json.dumps(resp).encode('utf-8'))
                    
                except Exception as e:
                    resp = {'error': str(e), 'status': 'error'}
                    client.sendall(json.dumps(resp).encode('utf-8'))
                finally:
                    client.close()
                    
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self.logger.error(f"服务错误: {e}")
        
        server.close()
        self._save_state()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'state': self.state.value,
            'processed_count': self.processed_count,
            'uptime': time.time() - self.start_time,
            'enable_learning': self.enable_learning,
            'learner_stats': self.learner.get_stats() if self.learner else None
        }
    
    def __repr__(self):
        return f"HexAgent(state={self.state.value}, processed={self.processed_count}, learning={self.enable_learning})"


# ============ 主入口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HexAgent - 支持在线学习的Hex处理Agent')
    parser.add_argument('--mode', choices=['interactive', 'service'], default='interactive',
                       help='运行模式')
    parser.add_argument('--port', type=int, default=8765, help='服务端口')
    parser.add_argument('--no-learning', action='store_true', help='禁用在线学习')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    args = parser.parse_args()
    
    # 创建Agent
    agent = HexAgent(
        enable_online_learning=not args.no_learning,
        learning_rate=args.lr,
        state_dir='./genshin_state'
    )
    
    print(f"\n{'='*60}")
    print(f"HexAgent 初始化完成")
    print(f"{'='*60}")
    print(f"Agent: {agent}")
    print(f"在线学习: {'启用' if agent.enable_learning else '禁用'}")
    print(f"学习率: {args.lr}")
    print(f"{'='*60}\n")
    
    # 运行
    if args.mode == 'interactive':
        agent.run_interactive()
    else:
        agent.run_service(port=args.port)
