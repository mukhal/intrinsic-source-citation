from typing import List, Dict

import numpy as np
from langchain.prompts.example_selector.base import BaseExampleSelector
from rank_bm25 import BM25Okapi

from src.utils import normalize_text


class BM25ExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]], index_keys: List[str]=None, n_exemplars=2):
        self.examples = examples
        self.corpus_items = []
        self.bm25 = None
        self.index_keys = index_keys
        self.n_exemplars = n_exemplars

        if self.examples:
            self.index()

    def index(self):
        self.corpus_items = [
            self.tokenize(' '.join(str(ex[k]) for k in (self.index_keys if self.index_keys else ex.keys())))
            for ex in self.examples
        ]
        self.bm25 = BM25Okapi(self.corpus_items)

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)
        self.index()

    def tokenize(self, in_str):
        ret = normalize_text(in_str).replace("\t", "").split()
        return ret

    def query(self, in_str):
        tokenized_query = self.tokenize(in_str)
        top_items = self.bm25.get_top_n(tokenized_query, self.examples, n=self.n_exemplars + 10)
        
        return top_items[:self.n_exemplars]


    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        query = (' '.join(str(v) for v in input_variables.values()))
        return self.query(query)