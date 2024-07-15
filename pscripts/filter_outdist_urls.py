import argparse
import os
import pickle as pkl
import datasets as hf_datasets
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union
import datasets as hf_datasets
import psutil
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from llmfoundry.data.constants import NO_URL

import sys
sys.path.append('../')
sys.path.append('./')

from utils.trie import MarisaTrie
import pickle as pkl
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter out-of-distribution URLs")
    parser.add_argument("--text_data_path", type=str,)
    parser.add_argument("--in_domain_qa_data_path", type=str, help="Path to the in-domain QA data directory")
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory")
    parser.add_argument("--out_dir", type=str, help="Path to the output directory")
    
    args = parser.parse_args()
    
    data_path = args.text_data_path
    tokenizer_path = args.tokenizer
    out_root = args.out_dir

    doc_train_dataset = hf_datasets.load_from_disk(dataset_path=os.path.join(data_path, 'train'))
    qa_train_dataset = hf_datasets.load_from_disk(dataset_path=os.path.join(args.in_domain_qa_data_path))


    all_urls = set()
    for d in doc_train_dataset:
        all_urls.add(d['url'])
    
    in_dist_urls = set()
    for d in qa_train_dataset:
        if isinstance(d['url'], list):
            for url in d['url']:
                in_dist_urls.add(url)
        else:
            in_dist_urls.add(d['url'])
    
    out_dist_urls = all_urls - in_dist_urls

    #### build URL tries
    print("Building URL tries for unseen (OOD) docs")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    assert tokenizer.additional_special_tokens[0] == '<url>', "URL start token not in tokenizer"
    assert tokenizer.additional_special_tokens[1] == '</url>', "URL end token not in tokenizer"

    ## build and save URL trie
    url_ids = [] 
    for url in out_dist_urls:
        ids = tokenizer(url, add_special_tokens=False)['input_ids']
        ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
        url_ids.append(ids)
    
    ### ADD NO-URL case 
    print("Adding no-url case...")
    url = NO_URL
    ids = tokenizer(url, add_special_tokens=False)['input_ids']
    ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
    url_ids.append(ids)
        
    url_trie = MarisaTrie(sequences=url_ids)
    pkl.dump(url_trie, open(os.path.join(out_root, 'unseen_url_trie.pkl'), 'wb'))
