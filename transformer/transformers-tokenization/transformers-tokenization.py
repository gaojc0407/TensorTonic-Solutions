import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    带有特殊标记的词级分词器。
    """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # 特殊标记 (根据题目要求固定 ID)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # 初始化时先预留特殊 Token 的位置
        self._add_special_tokens()

    def _add_special_tokens(self):
        """内部方法：添加特殊 Token 到词表"""
        special_tokens = [
            (self.pad_token, 0),
            (self.unk_token, 1),
            (self.bos_token, 2),
            (self.eos_token, 3)
        ]
        for token, idx in special_tokens:
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        # 特殊 Token 占用 0-3，所以下一个普通单词从 4 开始
        self.vocab_size = 4 

    def build_vocab(self, texts: List[str]) -> None:
        """
        从文本列表中构建词汇表。
        先添加特殊标记，再添加唯一词。
        """
        # 1. 收集所有文本中的唯一单词
        word_set = set()
        for text in texts:
            # 简单按空格分割单词（题目为 Word-level）
            words = text.split()
            for word in words:
                word_set.add(word)
        
        # 2. 将单词按字母顺序排序（保证结果一致），并分配 ID
        # ID 从 self.vocab_size (即 4) 开始递增
        for word in sorted(word_set):
            if word not in self.word_to_id:  # 防止重复添加
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        将文本转换为标记ID列表。
        对未知词使用UNK。
        """
        ids = []
        
        # 分割输入文本为单词列表
        words = text.split()
        
        # 遍历每个单词
        for word in words:
            # 使用字典的 get 方法，如果单词不在词表中，返回 UNK 的 ID (1)
            word_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            ids.append(word_id)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        将标记ID列表转换回文本。
        """
        words = []
        
        for id in ids:
            # 如果 ID 在 id_to_word 中存在，获取对应的单词
            # 如果不存在（理论上不会，但为了健壮性），显示为 <UNK>
            word = self.id_to_word.get(id, self.unk_token)
            words.append(word)
        
        # 将单词列表用空格连接成字符串
        return " ".join(words)