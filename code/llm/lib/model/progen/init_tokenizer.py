import os
from pathlib import Path

from tokenizers import Tokenizer

from lib.model.progen.path import PROGEN_PATH


TOKENIZER_FILE = PROGEN_PATH + "progen/progen2/tokenizer.json"


def init_tokenizer():
    path_to_tokenizer = os.path.dirname(Path(__file__)) + "/" + TOKENIZER_FILE
    with open(path_to_tokenizer, "r") as f:
        tokenizer = Tokenizer.from_str(f.read())
    return tokenizer    
