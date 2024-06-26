"""
This code contaisn the base tokenizer class and a few common helper functions. The base class also contains the common save/load functionality.
It would be be possible to be a lot more strict about the interface and 
(eg. isolating all regex/pattern patrs to the RegexTokenizer), but
we follow some concessions for simplicity. 
"""


import unicodedata
# ----------------------------------------------------------------
# helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts = None):
    """
    Given a list of integers, return a dictionary of counts of the consecutive pairs.
    Ex. [1,2,3,1,2] -> {(1,2): 1, (2,3): 1, (3,1): 1}
    Optionally allows to update an existing dictionary of counts. 
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, id[1:]):               # iterating through consecutive elements
        counts[pair] = counts.get(pair,0) +1    # if pair not present, its initialized by 0.
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all the consecutive occurrences of pair 
    with the new integer values token idx.
    Ex. ids = [1,2,3,1,2], for pair = (1,2) if idx = 4, then ids = [4,3,4]
    """

    newids = []
    i = 0
    while i<len(ids):
        # if not at the very last position AND the pair matches, replace it.
        if ids[i] == pair[0] and i<len(ids)-1 and ids[i+1] == pair[1]:
            newids.append(idx)                  # replacing pair with idx if found
            i += 2
        else: 
            newids.append(ids[i])               # retaining same values if not found
            i += 1
    return newids

def replace_control_characters(s:str) -> str:
    """
    We don't want to print the control characters
    this distorts the output. (eg. \n or much worse)
    https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    """
    chars  =[]
    for ch in s:
        if unicodedata.search(ch)[0] != "C": chars.append(ch) # good character
        else: chars.append(f"\\u{ord(ch)::04x}") # escape!!
    return "".join(chars)

def render_token(t:bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors = "replace")
    s = replace_control_characters(s)
    return s

# ----------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merge, no patters
        self.merges = {}                        # (int, int) -> int
        self.pattern = ""                       # str
        self.special_tokens = {}                # str -> int, e.g.{'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()        # int -> bytes
    
    def train(self, text, vocab_size, verbose = False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError
    
    def encode(self, text):
        # Tokenizer can encode a string to a list of integers
        raise NotImplementedError
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers to a string
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is simple and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]        
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is ispired (but not equivalent to) sentencepiece's models savings_
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human understanding 
        """

        # write model: to be used in load() later
        model_file = file_prefix * ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")              # write the version
            f.write(f"{self.pattern}\n")        # write the pattern
            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        #write the vocab: for humans to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using errors = 'replace'
                # to replace them with the replacement char 0.
                # This also means that we coudn't possible use .vocab in load()
                # because decoding in this way is lossy operation.
                s = render_token(token)
                
                # find the children of this token, if any
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]   #if token has children, render it 
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] --> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
        

    def load(self, model_file):
        """
        Inverse of save() but only for the model file
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding= "utf-8") as f:
            version = f.readline().strip()              # read the version
            assert version == "minbpe v1"

            self.pattern = f.readline().strip()         # read the pattern
            num_special = int(f.readline().strip())     # read the special tokens

            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            #readinging the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()





