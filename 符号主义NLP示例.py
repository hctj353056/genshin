# 符号主义NLP：自然语言理解的形式化示例
#
# 流程：自然语言 → 词法分析 → 句法分析 → 语义表示 → 逻辑推理 → 输出

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ============ 第一层：词汇表示（形式化基础）============

@dataclass
class Word:
    """词汇的符号表示"""
    text: str           # 原始文本
    pos: str           # 词性：N=名词, V=动词, A=形容词, D=副词, P=介词, Q=疑问词
    meaning: str        # 语义表示（谓词）
    
    def __repr__(self):
        return f"{self.text}/{self.pos}"

class Lexicon:
    """
    词法规则库（词汇知识库）
    
    符号主义核心：所有词都有预定义的语义标签
    """
    def __init__(self):
        # 词汇表：文本 → (词性, 语义谓词)
        self.dictionary: Dict[str, Tuple[str, str]] = {
            # 名词
            "我": ("N", "speaker"),
            "你": ("N", "listener"),  
            "他": ("N", "person_x"),
            "天气": ("N", "weather"),
            "北京": ("N", "location_beijing"),
            "今天": ("N", "time_today"),
            "明天": ("N", "time_tomorrow"),
            "温度": ("N", "temperature"),
            "事情": ("N", "matter"),
            "什么": ("N", "wh_thing"),
            
            # 动词
            "是": ("V", "be"),
            "有": ("V", "have"),
            "知道": ("V", "know"),
            "认为": ("V", "think"),
            "告诉": ("V", "tell"),
            "关心": ("V", "care_about"),
            "好": ("V", "be_good"),  # 好也可以是动词
            
            # 形容词
            "热": ("A", "temperature_high"),
            "冷": ("A", "temperature_low"),
            "晴": ("A", "weather_sunny"),
            "重要": ("A", "importance_high"),
            
            # 疑问词
            "吗": ("Q", "wh_yesno"),
        }
    
    def lookup(self, word: str) -> Optional[Word]:
        """查词法表，返回符号表示"""
        if word in self.dictionary:
            pos, meaning = self.dictionary[word]
            return Word(word, pos, meaning)
        return None

class ChineseSegmenter:
    """
    简单中文分词器（最大正向匹配）
    
    符号主义的词法分析从这里开始
    """
    def __init__(self):
        # 词典（从Lexicon提取）
        self.dictionary = {
            # 词
            "我": 1, "你": 1, "他": 1,
            "天气": 1, "北京": 1, "今天": 1, "明天": 1,
            "温度": 1, "事情": 1, "什么": 1,
            # 动词
            "是": 1, "有": 1, "知道": 1, "认为": 1, "告诉": 1, "关心": 1, "好": 1,
            # 形容词
            "热": 1, "冷": 1, "晴": 1, "重要": 1,
            # 疑问词
            "吗": 1,
        }
    
    def segment(self, text: str) -> List[str]:
        """最大正向匹配分词"""
        result = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试最长匹配
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self.dictionary:
                    result.append(word)
                    i += length
                    matched = True
                    break
            
            if not matched:
                # 单字作为一个词
                result.append(text[i])
                i += 1
        
        return result

# ============ 第二层：句法规则（语法形式化）============

class SyntaxRule:
    """句法规则模板"""
    def __init__(self, name: str, pattern: List[str], meaning_template: str):
        self.name = name              # 规则名
        self.pattern = pattern        # 词性序列模式
        self.meaning_template = meaning_template  # 语义生成模板
    
    def match(self, pos_sequence: List[str]) -> bool:
        """匹配词性序列"""
        if len(self.pattern) != len(pos_sequence):
            return False
        for p, a in zip(self.pattern, pos_sequence):
            if p != a and p != "_":  # _ 表示任意词性
                return False
        return True

class SyntaxParser:
    """
    句法分析器
    
    符号主义核心：使用规则匹配来理解句子结构
    """
    def __init__(self):
        self.rules: List[SyntaxRule] = [
            # 陈述句：主语 + 谓语
            SyntaxRule(
                "SN", 
                ["N", "V"],
                "{N} {V}"
            ),
            # 系表句：主语 + 是 + 表语
            SyntaxRule(
                "SP", 
                ["N", "V", "A"],
                "{N} {V} {A}"
            ),
            # 系表句2：主语 + 好
            SyntaxRule(
                "SP2", 
                ["N", "V"],
                "{N} {V}"  # "天气好" → weather is good
            ),
            # 疑问句：主语 + 谓语 + 吗
            SyntaxRule(
                "YNQ",  # yes-no question
                ["N", "V", "Q"],
                "QUERY: {N} {V} ?"
            ),
            # 询问句：什么 + 名词
            SyntaxRule(
                "WHQ",  # wh-question
                ["N", "N"],
                "QUERY: what is {N}"
            ),
        ]
    
    def parse(self, words: List[Word]) -> Tuple[Optional[SyntaxRule], Dict]:
        """
        解析句子结构
        
        返回：(匹配的规则, 提取的成分)
        """
        pos_sequence = [w.pos for w in words]
        
        for rule in self.rules:
            if rule.match(pos_sequence):
                # 提取成分
                components = {}
                for i, placeholder in enumerate(rule.pattern):
                    if placeholder != "_":
                        components[placeholder] = words[i].meaning
                
                return rule, components
        
        return None, {}

# ============ 第三层：语义表示（知识表示）============

class SemanticRepresentation:
    """
    语义表示形式化
    
    使用逻辑谓词表示语义
    """
    
    @staticmethod
    def represent(rule: SyntaxRule, components: Dict) -> str:
        """将成分组合为语义表示"""
        template = rule.meaning_template
        
        for key, value in components.items():
            template = template.replace("{" + key + "}", value)
        
        return template
    
    @staticmethod
    def to_logic(semantic: str) -> str:
        """转换为逻辑形式"""
        # 简化版：将语义转为Horn子句形式
        mappings = {
            "speaker": "AGENT(you)",
            "listener": "AGENT(me)",
            "know": "KNOWS(you, X)",
            "think": "BELIEVES(you, X)",
            "weather_sunny": "WEATHER(sunny)",
        }
        
        for natural, logic in mappings.items():
            semantic = semantic.replace(natural, logic)
        
        return semantic

# ============ 第四层：知识库与推理机 ============

class Fact:
    """事实（原子公式）"""
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate
        self.args = args
    
    def __str__(self):
        return f"{self.predicate}({', '.join(self.args)})"
    
    def __repr__(self):
        return self.__str__()

class Rule:
    """规则（Horn子句）"""
    def __init__(self, head: Fact, body: List[Fact]):
        self.head = head      # 结论
        self.body = body      # 前提
    
    def __str__(self):
        if self.body:
            body_str = " AND ".join(str(f) for f in self.body)
            return f"{self.head} :- {body_str}"
        return str(self.head)

class KnowledgeBase:
    """
    知识库
    
    符号主义核心：存储事实和规则
    """
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
    
    def add_fact(self, fact: Fact):
        self.facts.add(fact)
    
    def add_rule(self, rule: Rule):
        self.rules.append(rule)
    
    def query(self, fact: Fact) -> bool:
        """前向链推理"""
        # 检查事实
        if fact in self.facts:
            return True
        
        # 检查规则
        for rule in self.rules:
            if rule.head.predicate == fact.predicate:
                # 检查所有前提
                if all(premise in self.facts for premise in rule.body):
                    return True
        
        return False

class InferenceEngine:
    """
    推理机
    
    符号主义核心：自动推理
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
    
    def forward_chain(self) -> List[Fact]:
        """前向链推理：自动推导新事实"""
        new_facts = []
        
        for rule in self.kb.rules:
            # 检查前提
            if all(premise in self.kb.facts for premise in rule.body):
                # 推导结论
                if rule.head not in self.kb.facts:
                    self.kb.add_fact(rule.head)
                    new_facts.append(rule.head)
        
        return new_facts

# ============ 第五层：完整的NLP Pipeline ============

class SymbolicNLP:
    """
    符号主义NLP系统
    
    完整流程：
    自然语言 → 词法分析 → 句法分析 → 语义表示 → 知识库 → 推理 → 输出
    """
    
    def __init__(self):
        self.lexicon = Lexicon()
        self.parser = SyntaxParser()
        self.segmenter = ChineseSegmenter()  # 中文分词器
        self.kb = KnowledgeBase()
        self.inference = InferenceEngine(self.kb)
        
        # 初始化知识库
        self._init_knowledge()
    
    def _init_knowledge(self):
        """初始化知识库"""
        # 事实
        self.kb.add_fact(Fact("weather", ["today", "sunny"]))
        self.kb.add_fact(Fact("temperature", ["today", "25"]))
        self.kb.add_fact(Fact("speaker", ["you"]))
        self.kb.add_fact(Fact("listener", ["me"]))
        
        # 规则：如果天气好 → 心情好
        self.kb.add_rule(Rule(
            head=Fact("mood", ["you", "good"]),
            body=[Fact("weather", ["today", "sunny"])]
        ))
        
        # 如果知道X → 可以回答关于X的问题
        self.kb.add_rule(Rule(
            head=Fact("can_answer", ["X"]),
            body=[Fact("know", ["you", "X"])]
        ))
    
    def process(self, text: str) -> Dict:
        """
        处理自然语言
        
        返回各层的分析结果
        """
        result = {
            "input": text,
            "layers": {}
        }
        
        # ===== 第一层：词法分析 =====
        print(f"\n{'='*50}")
        print(f"输入: {text}")
        print(f"{'='*50}")
        print("\n【第一层：词法分析】")
        print("-" * 30)
        
        # 中文分词
        tokens = self.segmenter.segment(text)
        print(f"  分词结果: {' | '.join(tokens)}")
        
        words = []
        for token in tokens:
            word = self.lexicon.lookup(token)
            if word:
                words.append(word)
                print(f"  {token} → 词性={word.pos}, 语义={word.meaning}")
            else:
                # 尝试匹配未知词
                print(f"  {token} → [未知词]")
                words.append(Word(token, "X", "unknown"))
        
        result["layers"]["lexical"] = words
        
        # ===== 第二层：句法分析 =====
        print("\n【第二层：句法分析】")
        print("-" * 30)
        
        pos_seq = [w.pos for w in words]
        print(f"  词性序列: {' '.join(pos_seq)}")
        
        rule, components = self.parser.parse(words)
        if rule:
            print(f"  匹配规则: {rule.name}")
            print(f"  成分提取: {components}")
        else:
            print(f"  [无法匹配规则]")
        
        result["layers"]["syntax"] = {"rule": rule.name if rule else None, "components": components}
        
        # ===== 第三层：语义表示 =====
        print("\n【第三层：语义表示】")
        print("-" * 30)
        
        if rule:
            semantic = SemanticRepresentation.represent(rule, components)
            print(f"  自然语义: {semantic}")
            
            logic = SemanticRepresentation.to_logic(semantic)
            print(f"  逻辑形式: {logic}")
            
            result["layers"]["semantic"] = {"natural": semantic, "logic": logic}
        
        # ===== 第四层：知识库查询 =====
        print("\n【第四层：知识库】")
        print("-" * 30)
        
        print(f"  已知事实: {[str(f) for f in self.kb.facts]}")
        
        # 第五层：推理
        print("\n【第五层：推理】")
        print("-" * 30)
        
        new_facts = self.inference.forward_chain()
        if new_facts:
            print(f"  推导事实: {new_facts}")
        else:
            print(f"  [无新推导]")
        
        result["layers"]["inference"] = {"new_facts": [str(f) for f in new_facts]}
        
        return result

# ============ 示例运行 ============

def main():
    print("=" * 60)
    print("符号主义NLP：自然语言理解的形式化演示")
    print("=" * 60)
    
    nlp = SymbolicNLP()
    
    # 测试用例
    test_cases = [
        "天气好",
        "天气好吗",
        "什么天气",
    ]
    
    for text in test_cases:
        nlp.process(text)
    
    print("\n" + "=" * 60)
    print("总结：符号主义NLP的形式化流程")
    print("=" * 60)
    print("""
1. 词法层：每个词都有预定义的语义标签
2. 句法层：规则匹配确定句子结构
3. 语义层：将结构转换为逻辑谓词
4. 知识层：存储事实和规则
5. 推理层：基于规则自动推导新知识

优点：可解释、精确、可推理
缺点：规则爆炸、难以处理歧义
""")

if __name__ == "__main__":
    main()
