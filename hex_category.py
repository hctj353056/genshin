# hex_category.py
# 范畴论语言处理模块 - 中文屋子实现
# 
# 核心思想：
# - 不需要"理解"语言
# - 从输入中提取统计模式
# - 用模式规则生成输出
#
# 流程：
# 1. 输入分割为字符列表
# 2. 按长度分组，计算相似度
# 3. 提取句式模板（相同部分）和共类（差异部分）
# 4. 建立输入→输出范畴映射
# 5. 生成时选取符合规则的句式模板

import os
import json
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from 词元模块_1778459060672_3xq9 import str_to_hex, hex_to_str

# ============ 核心数据结构 ============

class PhrasePattern:
    """
    句式模板：记录句子的结构模式
    
    例如：
    - 句式: "你___"（你+任意字符）
    - 共类: {"好", "叫", "好"}
    """
    def __init__(self, template: str, variants: Set[str]):
        self.template = template      # 句式模板（用_表示变化部分）
        self.variants = variants      # 共类变体集合
        self.count = 0               # 出现次数
        
    def generate(self, choice: str = None) -> str:
        """根据模板生成句子"""
        if choice and choice in self.variants:
            parts = self.template.split('_')
            result = parts[0] + choice + (parts[1] if len(parts) > 1 else '')
            return result
        elif self.variants:
            return self.template.replace('_', list(self.variants)[0])
        return self.template.replace('_', '')
    
    def __repr__(self):
        return f"Pattern({self.template}, 共类={len(self.variants)})"


class SentenceCategory:
    """
    句子范畴：包含句式模板和共类
    
    范畴 = 对象（具体句子）+ 态射（句式模板关系）
    """
    def __init__(self, length: int):
        self.length = length
        self.exact_sentences: Set[str] = set()      # 完全相同的句子
        self.patterns: List[PhrasePattern] = []      # 句式模板列表
        
    def add_sentence(self, sentence: str):
        """添加句子"""
        self.exact_sentences.add(sentence)
        
    def extract_patterns(self):
        """
        从句子集合中提取句式模板
        
        核心算法：
        1. 找公共前缀（句式开头）
        2. 找公共后缀（句式结尾）
        3. 中间部分为共类
        """
        if len(self.exact_sentences) < 2:
            return []
        
        sentences = list(self.exact_sentences)
        first = sentences[0]
        patterns = []
        
        # 找公共前缀
        prefix_len = 0
        while prefix_len < len(first):
            char = first[prefix_len]
            if all(s[prefix_len] == char for s in sentences if len(s) > prefix_len):
                prefix_len += 1
            else:
                break
        
        # 找公共后缀
        suffix_len = 0
        while suffix_len < len(first) - prefix_len:
            char = first[-(suffix_len + 1)]
            if all(s[-(suffix_len + 1)] == char for s in sentences if len(s) > suffix_len):
                suffix_len += 1
            else:
                break
        
        # 中间部分为共类
        if prefix_len + suffix_len < len(first):
            template_prefix = first[:prefix_len]
            template_suffix = first[-suffix_len:] if suffix_len > 0 else ''
            template_middle = '_'  # 用_表示共类位置
            
            if suffix_len > 0:
                template = template_prefix + template_middle + template_suffix
            else:
                template = template_prefix + template_middle
            
            # 提取共类
            variants = set()
            for s in sentences:
                if len(s) >= prefix_len + suffix_len:
                    middle = s[prefix_len:len(s)-suffix_len] if suffix_len > 0 else s[prefix_len:]
                    variants.add(middle)
            
            if len(variants) > 1:
                patterns.append(PhrasePattern(template, variants))
        
        return patterns


# ============ 范畴系统核心 ============

class HexCategorySystem:
    """
    基于范畴论的Hex语言处理系统
    
    架构：
    输入 → 字符列表 → 按长度分组 → 提取句式+共类 → 建立范畴映射 → 生成输出
    
    核心概念：
    - 对象：具体的句子（字符串）
    - 态射：句式模板（相同结构的关系）
    - 范畴：输入集合 + 输出集合 + 映射关系
    - 函子：保持结构的映射（输入范畴→输出范畴）
    """
    
    def __init__(self, data_dir: str = './category_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 用户输入集合（按长度索引）
        self.user_inputs: Dict[int, Set[str]] = defaultdict(set)
        
        # 程序输出集合（按长度索引）
        self.program_outputs: Dict[int, Set[str]] = defaultdict(set)
        
        # 范畴映射：用户输入类型 → 程序输出类型
        self.category_morphisms: Dict[str, str] = {}
        
        # 句式模板集合
        self.user_patterns: Dict[int, List[PhrasePattern]] = defaultdict(list)
        self.output_patterns: Dict[int, List[PhrasePattern]] = defaultdict(list)
        
        # 加载已有数据
        self._load_data()
        
    def _get_data_path(self, name: str) -> str:
        return os.path.join(self.data_dir, f'{name}.json')
    
    def _load_data(self):
        """加载已有数据"""
        user_path = self._get_data_path('user_inputs')
        output_path = self._get_data_path('program_outputs')
        morphism_path = self._get_data_path('morphisms')
        
        if os.path.exists(user_path):
            data = json.load(open(user_path, 'r', encoding='utf-8'))
            for length, sentences in data.items():
                self.user_inputs[int(length)] = set(sentences)
        
        if os.path.exists(output_path):
            data = json.load(open(output_path, 'r', encoding='utf-8'))
            for length, sentences in data.items():
                self.program_outputs[int(length)] = set(sentences)
                
        if os.path.exists(morphism_path):
            self.category_morphisms = json.load(open(morphism_path, 'r', encoding='utf-8'))
    
    def _save_data(self):
        """保存数据"""
        # 保存用户输入
        user_data = {k: list(v) for k, v in self.user_inputs.items()}
        json.dump(user_data, open(self._get_data_path('user_inputs'), 'w', encoding='utf-8'), ensure_ascii=False)
        
        # 保存程序输出
        output_data = {k: list(v) for k, v in self.program_outputs.items()}
        json.dump(output_data, open(self._get_data_path('program_outputs'), 'w', encoding='utf-8'), ensure_ascii=False)
        
        # 保存范畴映射
        json.dump(self.category_morphisms, open(self._get_data_path('morphisms'), 'w', encoding='utf-8'), ensure_ascii=False)
    
    def learn_from_user_input(self, text: str):
        """
        学习用户输入
        
        1. 将字符串转为字符列表
        2. 按长度添加到集合
        3. 提取句式模板
        """
        if not text:
            return
            
        # 添加到用户输入集合
        self.user_inputs[len(text)].add(text)
        
        # 提取句式
        self._extract_patterns(len(text), is_user=True)
        
        # 保存
        self._save_data()
        
        print(f"学习用户输入: '{text}' (长度={len(text)})")
    
    def learn_from_program_output(self, text: str):
        """学习程序输出"""
        if not text:
            return
            
        self.program_outputs[len(text)].add(text)
        self._extract_patterns(len(text), is_user=False)
        self._save_data()
        
        print(f"学习程序输出: '{text}' (长度={len(text)})")
    
    def add_morphism(self, user_input: str, program_output: str):
        """添加范畴映射：用户输入类型 → 程序输出"""
        self.category_morphisms[user_input] = program_output
        self._save_data()
    
    def auto_learn(self, user_input: str, program_output: str = None):
        """
        自主学习核心方法
        
        每次对话自动调用：
        1. 学习用户输入
        2. 如果有程序输出，学习输出
        3. 分析输入结构，推断映射关系
        4. 如果没有显式输出，自动推断可能的回复
        """
        # 1. 学习用户输入
        self.learn_from_user_input(user_input)
        
        # 2. 如果有输出，学习输出
        if program_output:
            self.learn_from_program_output(program_output)
            # 立即建立映射
            self.add_morphism(user_input, program_output)
            return program_output
        
        # 3. 分析输入结构，推断映射
        inferred = self._analyze_and_infer_mapping(user_input)
        
        return inferred
    
    def _analyze_and_infer_mapping(self, user_input: str) -> str:
        """
        分析输入结构，推断映射关系
        
        范畴论视角：
        - 识别输入的句式类型
        - 如果有输出，建立输入类型→输出类型的映射
        - 如果没有输出，尝试从历史中推断
        """
        length = len(user_input)
        
        # 如果已经有映射，直接返回
        if user_input in self.category_morphisms:
            return self.category_morphisms[user_input]
        
        # 检查是否有相同长度的已知句式
        if length not in self.user_inputs:
            return None
        
        known_sentences = list(self.user_inputs[length])
        
        # 分析与已知句式的相似度
        for known in known_sentences:
            if known == user_input:
                continue
                
            similarity = self._calculate_text_similarity(user_input, known)
            
            # 如果相似度高，自动推断映射
            if similarity > 0.4:
                # 检查已知句式是否有映射
                if known in self.category_morphisms:
                    known_output = self.category_morphisms[known]
                    
                    # 尝试保持相同的句式关系
                    inferred_output = self._transfer_pattern(known, known_output, user_input)
                    if inferred_output:
                        self.add_morphism(user_input, inferred_output)
                        print(f"  🔗 自动推断映射: '{user_input}' → '{inferred_output}'")
                        return inferred_output
        
        return None
    
    def _transfer_pattern(self, source_input: str, source_output: str, target_input: str) -> str:
        """
        模式迁移：将已知输入-输出的关系迁移到新输入
        
        范畴论中的函子：保持结构的映射
        """
        if not source_input or not source_output:
            return None
        
        # 找公共前缀和后缀
        prefix_len = 0
        while prefix_len < len(source_input) and prefix_len < len(target_input):
            if source_input[prefix_len] == target_input[prefix_len]:
                prefix_len += 1
            else:
                break
        
        # 找公共后缀
        suffix_len = 0
        while (suffix_len < len(source_input) - prefix_len and 
               suffix_len < len(target_input) - prefix_len):
            if source_input[-(suffix_len + 1)] == target_input[-(suffix_len + 1)]:
                suffix_len += 1
            else:
                break
        
        # 如果有足够的公共部分，尝试迁移
        if prefix_len + suffix_len >= min(len(source_input), len(target_input)) * 0.5:
            # 替换中间部分
            prefix = target_input[:prefix_len]
            suffix = target_input[-suffix_len:] if suffix_len > 0 else ''
            middle = source_output[len(source_input):len(source_output)-suffix_len] if suffix_len < len(source_output) else ''
            
            # 调整长度以匹配
            if len(prefix) + len(middle) + len(suffix) == len(target_input):
                return prefix + middle + suffix
        
        # 回退：返回与target等长的source_output片段
        if len(source_output) >= len(target_input):
            return source_output[:len(target_input)]
        
        return None
    
    def _infer_response(self, user_input: str) -> str:
        """
        推断回复（当没有显式输出时）
        
        策略：
        1. 查找相似的已知输入
        2. 如果有映射，尝试迁移模式
        3. 否则返回输入（复读）
        """
        # 查找最相似的已知输入
        best_similarity = 0
        best_match = None
        
        for known in self.user_inputs[len(user_input)]:
            sim = self._calculate_text_similarity(user_input, known)
            if sim > best_similarity and sim >= 0.3:
                best_similarity = sim
                best_match = known
        
        if best_match and best_match in self.category_morphisms:
            # 尝试迁移模式
            source_output = self.category_morphisms[best_match]
            inferred = self._transfer_pattern(best_match, source_output, user_input)
            if inferred:
                print(f"  🤔 推断回复: '{user_input}' → '{inferred}'")
                return inferred
        
        return None
    
    def _extract_patterns(self, length: int, is_user: bool):
        """从指定长度的句子中提取句式模板"""
        source = self.user_inputs if is_user else self.program_outputs
        patterns_dict = self.user_patterns if is_user else self.output_patterns
        
        sentences = list(source[length])
        if len(sentences) < 2:
            return
        
        # 创建范畴
        category = SentenceCategory(length)
        for s in sentences:
            category.add_sentence(s)
        
        # 提取模式
        patterns = category.extract_patterns()
        patterns_dict[length] = patterns
        
        print(f"  提取到 {len(patterns)} 个句式模板")
    
    def _split_to_chars(self, text: str) -> List[str]:
        """将字符串分割为字符列表"""
        return list(text)
    
    def _calculate_similarity(self, list1: List[str], list2: List[str]) -> Tuple[int, int]:
        """
        计算两个列表的相似度
        
        返回：(相同字符数, 不同字符位置数)
        """
        if len(list1) != len(list2):
            return 0, max(len(list1), len(list2))
        
        same = sum(1 for a, b in zip(list1, list2) if a == b)
        diff = len(list1) - same
        return same, diff
    
    def _find_common_subsequence(self, list1: List[str], list2: List[str]) -> Tuple[List[str], int, int]:
        """
        找两个列表的公共子序列
        
        返回：(公共部分, 在list1中的起始位置, 在list2中的起始位置)
        """
        max_len = 0
        max_start1, max_start2 = 0, 0
        
        for i in range(len(list1)):
            for j in range(len(list2)):
                # 找从(i,j)开始的最长公共前缀
                k = 0
                while (i+k < len(list1) and j+k < len(list2) and 
                       list1[i+k] == list2[j+k]):
                    k += 1
                
                if k > max_len:
                    max_len = k
                    max_start1, max_start2 = i, j
        
        common = list1[max_start1:max_start1+max_len]
        return common, max_start1, max_start2
    
    def generate_response(self, user_input: str, mode: str = "similar") -> str:
        """
        生成回复
        
        mode:
        - "similar": 找相似输入的输出（默认，优先使用范畴映射）
        - "morphism": 使用范畴映射
        - "template": 使用句式模板生成
        """
        if mode == "similar":
            return self._generate_by_similar(user_input)
        elif mode == "morphism":
            return self._generate_by_morphism(user_input)
        else:
            return self._generate_by_template(user_input)
    
    def _generate_by_morphism(self, user_input: str) -> str:
        """使用范畴映射生成"""
        # 查找相同或相似的输入
        for stored_input, stored_output in self.category_morphisms.items():
            if stored_input == user_input:
                return stored_output
            
            # 检查是否满足范畴映射规则
            chars1 = self._split_to_chars(user_input)
            chars2 = self._split_to_chars(stored_input)
            
            if len(chars1) == len(chars2):
                same, diff = self._calculate_similarity(chars1, chars2)
                if same >= len(chars1) * 0.7:  # 70%相似
                    # 替换差异部分
                    output_chars = list(stored_output)
                    for i, (c1, c2) in enumerate(zip(chars1, chars2)):
                        if c1 != c2:
                            # 找到输出中对应位置的字符
                            pass  # 简化处理
                    return stored_output
        
        # 没找到映射，返回输入（复读）
        return user_input
    
    def _generate_by_similar(self, user_input: str) -> str:
        """找相似输入的输出（改进版：使用范畴映射）"""
        # 先检查是否有直接映射
        if user_input in self.category_morphisms:
            return self.category_morphisms[user_input]
        
        # 找最相似的已知输入
        best_similarity = 0
        best_match = None
        
        for known_input in self.category_morphisms.keys():
            # 计算相似度
            similarity = self._calculate_text_similarity(user_input, known_input)
            
            if similarity > best_similarity and similarity >= 0.3:
                best_similarity = similarity
                best_match = known_input
        
        if best_match:
            return self.category_morphisms[best_match]
        
        return user_input
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        使用多种相似度指标：
        1. 公共前缀比例
        2. 公共字符比例
        3. 长度差异
        """
        # 1. 公共前缀
        prefix_len = 0
        for c1, c2 in zip(text1, text2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        prefix_ratio = prefix_len / max(len(text1), len(text2))
        
        # 2. 公共字符集合
        chars1 = set(text1)
        chars2 = set(text2)
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        jaccard = intersection / union if union > 0 else 0
        
        # 3. 长度差异惩罚
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        # 综合相似度
        similarity = (prefix_ratio * 0.5 + jaccard * 0.3 + len_ratio * 0.2)
        return similarity
    
    def _generate_by_template(self, user_input: str) -> str:
        """使用句式模板生成"""
        length = len(user_input)
        
        # 找相同长度的句式模板
        patterns = self.user_patterns.get(length, [])
        
        if not patterns:
            # 没有模板，返回输入
            return user_input
        
        # 尝试匹配模板并生成
        for pattern in patterns:
            # 简化：直接返回模板的一个变体
            if pattern.variants:
                return pattern.generate()
        
        return user_input
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_inputs = sum(len(v) for v in self.user_inputs.values())
        total_outputs = sum(len(v) for v in self.program_outputs.values())
        total_patterns = sum(len(v) for v in self.user_patterns.values())
        
        return {
            'user_inputs': total_inputs,
            'user_lengths': len(self.user_inputs),
            'program_outputs': total_outputs,
            'output_lengths': len(self.program_outputs),
            'morphisms': len(self.category_morphisms),
            'patterns': total_patterns
        }


# ============ 测试 ============

if __name__ == "__main__":
    cs = HexCategorySystem(data_dir='./test_category')
    
    print("=" * 50)
    print("范畴系统测试")
    print("=" * 50)
    
    # 学习用户输入
    print("\n1. 学习用户输入:")
    user_inputs = ["你好", "您好", "你叫", "你叫什么", "你好啊", "你好世界", "早上好"]
    for inp in user_inputs:
        cs.learn_from_user_input(inp)
    
    # 学习程序输出
    print("\n2. 学习程序输出:")
    outputs = ["你好", "你好", "我是", "我是HexAgent", "你好", "你好", "天气不错"]
    for outp in outputs:
        cs.learn_from_program_output(outp)
    
    # 建立映射
    print("\n3. 建立范畴映射:")
    for inp, outp in zip(user_inputs, outputs):
        cs.add_morphism(inp, outp)
    
    # 统计
    print("\n统计:", cs.get_stats())
    
    # 测试生成
    print("\n4. 测试生成:")
    test_inputs = ["你好", "您好", "你叫什么", "未知输入"]
    for inp in test_inputs:
        response = cs.generate_response(inp, mode="similar")
        print(f"  输入: {inp} -> 输出: {response}")
