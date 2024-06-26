"""
Minimal (byte-level) byte pair Encoding tokenizer
Algorithmically follows GPT tokenizer
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but it doesnt handles:
- regex splitting pattern
- any special tokens

"""

from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text processing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        merges = {}                         # (int, int) --> int
        vocab = {ids: bytes([ids]) for idx in range(256)}
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with highest count
            pair = max(stats, key = stats.get)
            # mint a new token: assign it the next aviable id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            #save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            #lets print
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences.")
            # save the class variables
            self.merges = merges            # used in encode()
            self.vocab  = vocab             # used in decode()
    
    def decode(self, ids):
        # given ids (list of int), return py strings
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text
    
    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")   # raw bytes
        ids = list(text_bytes)              # list of integers in range(0,255)
        while len(ids) >= 2:
            #find the pair with teh lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))

            #subtitle: if there are no more merges available, the key wil 
            # result in an inf for every single pair, and the min will be 
            # jsut the first pair in the list, arbitrarily
            # we can detect this termination case by a membership catch

            if pair not in self.merges:
                break                       # nothing else can be merged
            
            #otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids