# hex_category.py
# 范畴论语言处理模块
# 
# 核心概念：
# - 对象（Object）：语言单元（词、词组、句子）的hex表示
# - 态射（Morphism）：单元间的关系（修饰、因果、时序）
# - 范畴（Category）：对象+态射+组合律+恒等态射
# - 函子（Functor）：范畴间的结构保持映射
# - 自然变换（Natural Transformation）：函子间的映射
#
# 图结构：
# - 节点：语言单元（对象）
# - 边：关系（态射）
# - 路径：复合态射

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str

HEX_CHARS = '0123456789ABCDEF'


@dataclass
class Morphism:
    """
    态射：对象之间的关系
    
    source -> target 表示 source 经过某种关系 指向 target
    """
    source: str      # 源对象（hex字符串）
    target: str      # 目标对象（hex字符串）
    relation: str    # 关系类型（hex编码）
    weight: float = 1.0  # 权重/置信度
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation))


class Category:
    """
    范畴：语言单元的范畴
    
    包含：
    - 对象集合（language_units）
    - 态射集合（morphism_graph）
    - 组合律：态射可以组合
    - 恒等态射：每个对象的自映射
    """
    
    def __init__(self, name: str = "Language"):
        self.name = name
        self.objects: Set[str] = set()      # 对象集合（hex字符串）
        self.morphisms: List[Morphism] = []  # 态射列表
        self.morphism_index: Dict[str, List[Morphism]] = defaultdict(list)  # 按源对象索引
        
    def add_object(self, obj: str):
        """添加对象"""
        self.objects.add(obj.upper())
        
    def add_morphism(self, source: str, target: str, relation: str = "NEXT"):
        """添加态射"""
        m = Morphism(source.upper(), target.upper(), relation.upper())
        self.morphisms.append(m)
        self.morphism_index[source.upper()].append(m)
        # 确保对象存在
        self.add_object(source)
        self.add_object(target)
        return m
    
    def get_morphisms_from(self, source: str) -> List[Morphism]:
        """获取从source出发的所有态射"""
        return self.morphism_index.get(source.upper(), [])
    
    def get_morphisms_to(self, target: str) -> List[Morphism]:
        """获取指向target的所有态射"""
        return [m for m in self.morphisms if m.target == target.upper()]
    
    def find_path(self, source: str, target: str, max_depth: int = 3) -> List[List[Morphism]]:
        """查找从source到target的所有路径（复合态射）"""
        source = source.upper()
        target = target.upper()
        
        paths = []
        
        def dfs(current: str, path: List[Morphism], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            
            for m in self.get_morphisms_from(current):
                if m not in path:  # 避免循环
                    path.append(m)
                    dfs(m.target, path, depth + 1)
                    path.pop()
        
        dfs(source, [], 0)
        return paths
    
    def extract_common_patterns(self, min_occurrence: int = 2) -> Dict[str, int]:
        """
        提取共性模式（极限/余极限概念）
        
        找出发生多次的子序列模式
        """
        patterns = defaultdict(int)
        
        # 从态射序列中提取模式
        for m in self.morphisms:
            # 简单模式：源对象+关系
            pattern = f"{m.source}:{m.relation}"
            patterns[pattern] += 1
        
        # 过滤低频模式
        return {k: v for k, v in patterns.items() if v >= min_occurrence}
    
    def compose(self, m1: Morphism, m2: Morphism) -> Optional[Morphism]:
        """
        态射组合：如果 m1.target == m2.source，则可以组合
        
        组合律：(f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        if m1.target != m2.source:
            return None
        
        return Morphism(
            source=m1.source,
            target=m2.target,
            relation=f"{m1.relation}+{m2.relation}",
            weight=m1.weight * m2.weight
        )


class HexCategorySystem:
    """
    基于范畴论的Hex语言处理系统
    
    架构：
    输入hex → 解析为对象 → 在范畴中查找态射 → 通过函子映射 → 输出
    """
    
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        
        # 核心范畴：语言范畴
        self.language_category = Category("Language")
        
        # 对象嵌入：将hex字符串映射为向量
        self.object_embeddings: Dict[str, np.ndarray] = {}
        
        # 态射嵌入：将关系类型映射为向量
        self.morphism_embeddings: Dict[str, np.ndarray] = {}
        
        # 统计信息
        self.total_inputs = 0
        self.total_outputs = 0
        
    def hex_to_object(self, hex_str: str) -> str:
        """将hex字符串规范化为对象"""
        return hex_str.upper().strip()
    
    def learn_from_input(self, hex_input: str, context: List[str] = None):
        """
        从输入学习：提取对象和态射
        
        上下文中的相邻元素形成态射关系
        """
        hex_input = hex_input.upper()
        self.language_category.add_object(hex_input)
        self.total_inputs += 1
        
        # 如果有上下文，提取上下文中的关系
        if context:
            # 上下文中前一个元素 -> 当前输入
            if len(context) > 0:
                prev = context[-1].upper()
                self.language_category.add_morphism(prev, hex_input, "CONTEXT")
        
        # 从输入本身提取模式（自关联）
        # 相邻的hex字符形成关系
        for i in range(len(hex_input) - 1):
            source = hex_input[i]
            target = hex_input[i + 1]
            self.language_category.add_morphism(source, target, "ADJACENT")
    
    def learn_from_conversation(self, user_input: str, agent_response: str):
        """
        从对话中学习：用户输入 → Agent回复
        
        这形成一个态射：user_input -> agent_response
        """
        # 统一转为hex
        user_hex = str_to_hex(user_input).upper().replace(' ', '')
        response_hex = str_to_hex(agent_response).upper().replace(' ', '')
        
        self.language_category.add_object(user_hex)
        self.language_category.add_object(response_hex)
        
        # 对话关系
        self.language_category.add_morphism(user_hex, response_hex, "RESPONSE")
        
        # 也学习输入内部的模式
        self.learn_from_input(user_hex)
        self.learn_from_input(response_hex)
        
        print(f"学习对话: {user_input[:20]}... -> {agent_response[:20]}...")
    
    def generate_response(self, hex_input: str, mode: str = "chain") -> str:
        """
        根据输入生成回复
        
        mode:
        - "chain": 沿着态射链生成
        - "reverse": 查找反向关系
        - "pattern": 基于模式匹配
        """
        hex_input = hex_input.upper()
        
        if mode == "chain":
            return self._generate_by_chain(hex_input)
        elif mode == "reverse":
            return self._generate_by_reverse(hex_input)
        elif mode == "pattern":
            return self._generate_by_pattern(hex_input)
        else:
            return self._generate_by_chain(hex_input)
    
    def _generate_by_chain(self, hex_input: str) -> str:
        """
        链式生成：沿着态射链继续走下去
        
        如果有 user -> response 的态射，就找类似的输入的输出
        """
        # 查找所有以hex_input为源的态射
        morphisms = self.language_category.get_morphisms_from(hex_input)
        
        # 优先找RESPONSE类型的态射
        response_morphisms = [m for m in morphisms if m.relation == "RESPONSE"]
        
        if response_morphisms:
            # 返回权重最高的回复
            best = max(response_morphisms, key=lambda m: m.weight)
            return best.target
        
        # 如果没有直接回复，尝试从模式中推断
        if morphisms:
            # 组合多个态射生成新输出
            path = morphisms[0]
            # 沿着链继续走
            for _ in range(3):
                next_m = self.language_category.get_morphisms_from(path.target)
                if next_m:
                    path = next_m[0]
                else:
                    break
            return path.target
        
        # 如果完全没找到，返回输入（复读）
        return hex_input
    
    def _generate_by_reverse(self, hex_input: str) -> str:
        """反向生成：查找指向输入的态射"""
        morphisms = self.language_category.get_morphisms_to(hex_input)
        
        if morphisms:
            # 随机选择一个
            m = morphisms[np.random.randint(0, len(morphisms))]
            return m.source
        
        return hex_input
    
    def _generate_by_pattern(self, hex_input: str) -> str:
        """
        基于模式生成：分析输入中的模式，生成匹配的回复
        
        使用范畴论的极限概念：找到共同的模式
        """
        # 提取输入中的模式
        patterns = []
        for i in range(len(hex_input) - 1):
            pattern = f"{hex_input[i]}:{hex_input[i+1]}"
            patterns.append(pattern)
        
        # 查找有相同模式的输出
        common_patterns = self.language_category.extract_common_patterns()
        
        # 构建基于模式的回复
        response = ""
        for p in patterns:
            if p in common_patterns:
                # 找到这个模式对应的对象
                obj = p.split(":")[1]  # 目标对象
                response += obj
        
        return response if response else hex_input
    
    def get_category_stats(self) -> dict:
        """获取范畴统计"""
        return {
            'objects': len(self.language_category.objects),
            'morphisms': len(self.language_category.morphisms),
            'inputs': self.total_inputs,
            'outputs': self.total_outputs,
            'patterns': len(self.language_category.extract_common_patterns())
        }


# ============ 测试 ============
if __name__ == "__main__":
    cs = HexCategorySystem()
    
    print("=" * 50)
    print("范畴论语言处理测试")
    print("=" * 50)
    
    # 学习一些对话
    conversations = [
        ("你好", "你好"),
        ("你叫什么", "我是HexAgent"),
        ("今天天气", "天气不错"),
        ("你好", "Hello"),
        ("你好", "嗨"),
    ]
    
    for user, agent in conversations:
        cs.learn_from_conversation(user, agent)
    
    # 打印统计
    stats = cs.get_category_stats()
    print(f"\n范畴统计: {stats}")
    
    # 打印对象
    print(f"\n对象数量: {len(cs.language_category.objects)}")
    print(f"态射数量: {len(cs.language_category.morphisms)}")
    
    # 打印共性模式
    patterns = cs.language_category.extract_common_patterns()
    print(f"\n共性模式: {patterns}")
    
    # 测试生成
    print("\n" + "=" * 50)
    print("生成测试")
    print("=" * 50)
    
    test_inputs = ["你好", "你叫什么", "今天天气", "未知输入"]
    
    for inp in test_inputs:
        inp_hex = str_to_hex(inp).upper().replace(' ', '')
        
        # 学习这个输入
        cs.learn_from_input(inp_hex)
        
        # 生成回复
        for mode in ["chain", "reverse", "pattern"]:
            response_hex = cs.generate_response(inp_hex, mode=mode)
            try:
                response = bytes.fromhex(response_hex).decode('utf-8')
            except:
                response = response_hex
            print(f"输入: {inp} ({mode}) -> {response}")
