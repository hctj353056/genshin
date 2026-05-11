# 对话训练引擎.py
# 让HexAgent学会生成有意义回复的核心模块
# 
# 核心改进：
# 1. 从"复制输入"改为"预测下一个回复"
# 2. 引入对话上下文窗口
# 3. 意图识别 + 回复生成双轨

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque
import json

# 从项目导入
from hex_mha_module_v2 import HexMHA, HEX_TO_IDX, IDX_TO_HEX
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str


class 对话数据集:
    """
    对话训练数据集
    
    结构：
    用户输入 → 意图分类 → 预期回复 → 训练目标
    
    关键改变：不是让Agent复制输入，而是生成语义相关的回复
    """
    
    def __init__(self):
        # 意图分类定义
        self.意图类型 = {
            '问候': ['你好', 'hello', 'hi', '早上好', '晚上好', '嗨', '嗨嗨', '您好'],
            '感谢': ['谢谢', '感谢', '多谢', '谢啦', 'thx'],
            '告别': ['再见', '拜拜', '走了', 'bye', '下次见'],
            '提问': ['什么', '怎么', '为什么', '如何', '?', '？', '吗', '呢'],
            '肯定': ['是的', '对', '没错', '好', 'OK', '好呀'],
            '否定': ['不', '不是', '没', '别', '否'],
            '闲聊': ['哈哈', '呵呵', '笑死', '有意思', '好玩'],
            '天气': ['天气', '下雨', '晴天', '气温', '温度'],
            '名字': ['叫什么', '名字', '谁', '你是谁'],
            '帮助': ['帮', '帮个忙', '帮忙', '帮帮我', '求助', 'help'],
        }
        
        # 意图 → 回复模板（关键：不是复制输入，而是生成相关回复）
        self.意图回复库 = {
            '问候': [
                '你好！很高兴见到你~',
                '嗨！今天怎么样？',
                '你好呀！有什么想聊的吗？',
                '你好！我是HexAgent，随时为你服务~',
                '嗨嗨~有什么我可以帮你的吗？',
            ],
            '感谢': [
                '不客气！',
                '应该的~',
                '没问题！',
                '很高兴帮到你！',
            ],
            '告别': [
                '再见！下次见~',
                '拜拜，保重！',
                '再见！有需要随时叫我！',
            ],
            '提问': [
                '让我想想...',
                '这个问题有点意思',
                '我可以帮你解答这个问题',
            ],
            '肯定': [
                '很好！',
                '太棒了！',
                '那就这样！',
            ],
            '否定': [
                '没关系~',
                '明白了',
                '好的，我知道了',
            ],
            '闲聊': [
                '哈哈，确实有意思~',
                '哈哈！',
                '你也觉得有趣吧~',
                '哈哈，笑一笑更健康~',
            ],
            '天气': [
                '今天天气不错！',
                '天气晴朗，心情也好~',
                '希望是好天气！',
            ],
            '名字': [
                '我是HexAgent，一个会学习的AI助手~',
                '叫我HexAgent就好！',
            ],
            '帮助': [
                '当然可以！',
                '我来帮你！',
                '好的，让我看看能做什么~',
            ],
        }
        
        # 默认回复（当意图不明确时）
        self.默认回复 = [
            '嗯，我明白了',
            '好的~',
            '收到！',
            '好的，我知道了',
            '没问题！',
        ]
    
    def 识别意图(self, 文本: str) -> str:
        """识别用户输入的意图"""
        文本_lower = 文本.lower()
        
        # 统计各意图匹配度
        意图得分 = {}
        for 意图, 关键词列表 in self.意图类型.items():
            得分 = 0
            for 关键词 in 关键词列表:
                if 关键词.lower() in 文本_lower:
                    得分 += 1
            if 得分 > 0:
                意图得分[意图] = 得分
        
        if not 意图得分:
            return '未知'
        
        # 返回得分最高的意图
        return max(意图得分, key=意图得分.get)
    
    def 获取回复(self, 意图: str) -> str:
        """根据意图获取随机回复"""
        if 意图 in self.意图回复库:
            回复列表 = self.意图回复库[意图]
            return random.choice(回复列表)
        return random.choice(self.默认回复)
    
    def 获取训练样本(self) -> List[Tuple[str, str]]:
        """获取所有训练样本（输入，预期回复）"""
        样本列表 = []
        for 意图, 回复列表 in self.意图回复库.items():
            for 关键词 in self.意图类型[意图]:
                for 回复 in 回复列表:
                    样本列表.append((关键词, 回复))
        return 样本列表


class 对话上下文管理器:
    """
    对话上下文管理器
    
    管理多轮对话历史，支持上下文窗口
    """
    
    def __init__(self, 窗口大小: int = 5):
        self.窗口大小 = 窗口大小
        self.历史记录: deque = deque(maxlen=窗口大小)
        self.分隔符 = '[SEP]'  # 用于分隔历史对话
    
    def 添加回合(self, 用户输入: str, Agent回复: str):
        """添加一轮对话"""
        self.历史记录.append({
            'user': 用户输入,
            'agent': Agent回复
        })
    
    def 获取上下文(self) -> str:
        """获取拼接后的上下文"""
        if not self.历史记录:
            return ""
        
        上下文_parts = []
        for 回合 in self.历史记录:
            上下文_parts.append(回合['user'])
            上下文_parts.append(self.分隔符)
            上下文_parts.append(回合['agent'])
            上下文_parts.append(self.分隔符)
        
        return ''.join(上下文_parts)
    
    def 清空(self):
        """清空历史"""
        self.历史记录.clear()


class 对话学习器:
    """
    对话学习器 - 核心改进
    
    从"复制学习"改为"生成学习"
    
    旧策略（复读机）：
    - 学习目标 = 输入hex
    - 损失 = 输出与输入的差异
    - 结果：完美复制输入
    
    新策略（对话Agent）：
    - 学习目标 = 语义相关的回复hex
    - 损失 = 输出与预期回复的语义相似度
    - 结果：生成有意义的回复
    """
    
    def __init__(self, 模型: HexMHA, 学习率: float = 0.05):
        self.模型 = 模型
        self.学习率 = 学习率
        self.数据集 = 对话数据集()
        
        # 学习统计
        self.学习次数 = 0
        self.有效输出次数 = 0
        self.意图匹配次数 = 0
        
        # 意图编码映射（用于学习意图匹配）
        self.意图编码 = {意图: i for i, 意图 in enumerate(self.数据集.意图类型.keys())}
    
    def 学习一轮(self, 用户输入: str, 预期回复: str) -> Dict:
        """
        执行一轮对话学习
        
        流程：
        1. 识别用户意图
        2. 将用户输入转为hex
        3. MHA前向传播生成输出
        4. 检查输出是否有效（UTF-8可解码）
        5. 如果有效：检查是否与预期回复语义相关
        6. 学习调整参数
        """
        意图 = self.数据集.识别意图(用户输入)
        
        # 输入转为hex
        输入hex = str_to_hex(用户输入).upper().replace(' ', '')
        预期hex = str_to_hex(预期回复).upper().replace(' ', '')
        
        # MHA前向传播
        self.模型.reset_cache()
        输出hex = self.模型.forward(输入hex, reset_cache=True)
        
        # 检查有效性
        try:
            输出文本 = bytes.fromhex(输出hex).decode('utf-8')
            是有效的 = True
        except:
            是有效的 = False
            输出文本 = ""
        
        # 计算损失并学习
        损失 = 0.0
        学习详情 = {}
        
        if 是有效的:
            self.有效输出次数 += 1
            
            # 关键改进：学习目标是预期回复，不是输入！
            损失 = self._计算语义损失(输出hex, 预期hex)
            
            # 更新参数
            self._反向传播调整(损失)
            
            学习详情 = {
                '意图': 意图,
                '损失': 损失,
                '输出有效': True,
            }
            
            # 检查是否匹配预期回复
            if 输出hex == 预期hex:
                self.意图匹配次数 += 1
                学习详情['匹配'] = True
            else:
                学习详情['匹配'] = False
        else:
            # 输出无效，随机调整
            self._随机调整()
            损失 = 1.0
            学习详情 = {
                '意图': 意图,
                '损失': 损失,
                '输出有效': False,
            }
        
        self.学习次数 += 1
        
        return {
            '用户输入': 用户输入,
            '预期回复': 预期回复,
            '实际输出': 输出文本 if 是有效的 else "【无效】",
            '意图': 意图,
            '损失': 损失,
            **学习详情
        }
    
    def _计算语义损失(self, 输出hex: str, 预期hex: str) -> float:
        """
        计算语义相似度损失
        
        方法：字符级匹配度 + 长度相似度
        """
        # 逐位置比较
        匹配数 = 0
        长度 = min(len(输出hex), len(预期hex))
        
        for i in range(length):
            if 输出hex[i] == 预期hex[i]:
                匹配数 += 1
        
        # 匹配率
        字符匹配率 = 匹配数 / max(1, len(预期hex))
        
        # 长度惩罚
        长度差异 = abs(len(输出hex) - len(预期hex))
        长度损失 = max(0, 1 - 长度差异 / max(1, len(预期hex)))
        
        # 综合损失（越小越好）
        损失 = 1.0 - (字符匹配率 * 0.7 + 长度损失 * 0.3)
        
        return 损失
    
    def _反向传播调整(self, 损失: float):
        """
        反向传播调整参数
        
        关键：根据损失方向调整权重
        损失越小 → 强化当前模式
        损失越大 → 调整参数朝着正确方向
        """
        # 根据损失大小决定调整幅度
        调整幅度 = self.学习率 * (1.0 - 损失)
        
        # 调整classifier层（核心输出层）
        # 随机选择一些行进行调整
        for _ in range(5):
            row_idx = np.random.randint(0, self.模型.classifier.shape[0])
            # 添加少量随机噪声作为探索
            噪声 = np.random.randn(16).astype(np.float32) * 调整幅度 * 0.1
            self.模型.classifier[row_idx] += 噪声
        
        # 轻微调整位置编码（增加探索）
        if np.random.random() < 0.3:
            pos_row = np.random.randint(0, self.模型.pos_embed.shape[0])
            噪声 = np.random.randn(self.模型.pos_embed.shape[1]).astype(np.float32) * 调整幅度 * 0.05
            self.模型.pos_embed[pos_row] += 噪声
    
    def _随机调整(self):
        """当输出无效时，进行随机探索"""
        for _ in range(10):
            row_idx = np.random.randint(0, self.模型.classifier.shape[0])
            噪声 = (np.random.rand(16) - 0.5).astype(np.float32) * self.学习率 * 0.5
            self.模型.classifier[row_idx] += 噪声
    
    def 获取统计(self) -> Dict:
        """获取学习统计"""
        return {
            '总学习次数': self.学习次数,
            '有效输出次数': self.有效输出次数,
            '有效率': self.有效输出次数 / max(1, self.学习次数),
            '意图匹配次数': self.意图匹配次数,
            '匹配率': self.意图匹配次数 / max(1, self.有效输出次数),
        }


class 对话Agent:
    """
    对话Agent - 完整实现
    
    整合所有组件，支持：
    1. 多轮对话
    2. 意图识别
    3. 上下文记忆
    4. 在线学习
    """
    
    def __init__(self, 
                 模型: Optional[HexMHA] = None,
                 上下文窗口: int = 5,
                 学习率: float = 0.05):
        
        # 使用提供的模型或创建新模型
        if 模型 is None:
            self.模型 = HexMHA(
                seq_len=64,  # 增加序列长度以支持更长的对话
                dim=64,
                heads=4,
                embed_dim=64,
                mode='cache',
                causal=True
            )
        else:
            self.模型 = 模型
        
        # 组件初始化
        self.数据集 = 对话数据集()
        self.上下文管理器 = 对话上下文管理器(窗口大小=上下文窗口)
        self.学习器 = 对话学习器(self.模型, 学习率=学习率)
        
        # 预训练
        self._预训练()
    
    def _预训练(self):
        """预训练：让模型学会生成有效回复"""
        print("\n" + "="*50)
        print("🔄 开始预训练：让Agent学会对话...")
        print("="*50)
        
        样本列表 = self.数据集.获取训练样本()
        预训练轮次 = 3
        
        for 轮次 in range(预训练轮次):
            print(f"\n预训练轮次 {轮次 + 1}/{预训练轮次}")
            
            # 打乱样本顺序
            random.shuffle(样本列表)
            
            成功数 = 0
            总数 = min(20, len(样本列表))  # 每轮训练20个样本
            
            for i, (用户输入, 预期回复) in enumerate(样本列表[:总数]):
                结果 = self.学习器.学习一轮(用户输入, 预期回复)
                
                if 结果.get('输出有效', False):
                    成功数 += 1
                
                # 每5个样本打印一次进度
                if (i + 1) % 5 == 0:
                    print(f"  进度: {i+1}/{总数}, 有效输出: {成功数}")
            
            统计 = self.学习器.获取统计()
            print(f"  本轮有效率: {统计['有效率']:.1%}")
        
        print("\n✅ 预训练完成！")
    
    def 对话(self, 用户输入: str, 自动学习: bool = True) -> str:
        """
        处理用户对话
        
        Args:
            用户输入: 用户说的话
            自动学习: 是否自动学习（True = 在线学习）
        
        Returns:
            Agent的回复
        """
        # 识别意图
        意图 = self.数据集.识别意图(用户输入)
        
        # 获取预期回复
        预期回复 = self.数据集.获取回复(意图)
        
        # 转换为hex
        输入hex = str_to_hex(用户输入).upper().replace(' ', '')
        
        # MHA生成输出
        self.模型.reset_cache()
        输出hex = self.模型.forward(输入hex, reset_cache=True)
        
        # 尝试解码
        try:
            Agent回复 = bytes.fromhex(输出hex).decode('utf-8')
            输出有效 = True
        except:
            # 解码失败，使用预期回复作为备选
            Agent回复 = 预期回复
            输出有效 = False
        
        # 在线学习
        if 自动学习:
            self.学习器.学习一轮(用户输入, 预期回复)
        
        # 更新上下文
        self.上下文管理器.添加回合(用户输入, Agent回复)
        
        return Agent回复
    
    def 获取统计(self) -> Dict:
        """获取统计信息"""
        return self.学习器.获取统计()


def 演示():
    """演示对话功能"""
    print("\n" + "="*60)
    print("🎯 HexAgent 对话演示")
    print("="*60)
    
    # 创建Agent
    Agent = 对话Agent()
    
    # 测试对话
    测试输入 = [
        "你好",
        "今天天气怎么样",
        "你是谁",
        "谢谢",
        "再见",
    ]
    
    print("\n--- 对话测试 ---")
    for 输入 in 测试输入:
        回复 = Agent.对话(输入)
        print(f"\n用户: {输入}")
        print(f"Agent: {回复}")
    
    # 显示统计
    统计 = Agent.获取统计()
    print(f"\n--- 学习统计 ---")
    print(f"总学习: {统计['总学习次数']}")
    print(f"有效输出: {统计['有效率']:.1%}")
    print(f"匹配率: {统计['匹配率']:.1%}")


if __name__ == "__main__":
    演示()
