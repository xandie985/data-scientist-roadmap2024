"""
Min byte-level Byte Pair encoding tokenizer

Algorithmically follows along GPT tokenizer
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer RegexTokenizer handles:
- an optional regex splitting pattern
- optional special tokens
"""

import regex as re
from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, are available here: 
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern = None):
        """
        - pattern: optional string to override the default (GPT4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
        Ex. {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        #iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pairs appears
            

