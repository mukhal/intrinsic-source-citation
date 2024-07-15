import transformers
import datasets
from datasets import DatasetDict
import sys
import argparse
from enum import Enum
import os

sys.path.append('/net/nfs/allennlp/muhammadk/grounding-lm')
from utils.url import attach_urls
import logging 
from utils.trie import MarisaTrie
import pickle as pkl
from nltk.tokenize import sent_tokenize

## sett logging level info 
logging.basicConfig(level=logging.INFO)

class URLTokens(Enum):
    START_TOKEN = '<url>'
    END_TOKEN = '</url>'


def main(args):
    ### shuffle data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir='/net/nfs/allennlp/muhammadk/cache')
    tokenizer.pad_token = tokenizer.eos_token

    data = datasets.load_from_disk('../processed-data/ccnews/ccnews_200K')
    ## trim articles to first 5 sentences. 
    def trim_to_length(example):
        sents = sent_tokenize(example['text'])
        example['text'] = ' '.join(sents[:5])
        return example
    
    data = data.map(trim_to_length)
    
    dataset = attach_urls(data, tokenizer=tokenizer, args=args)
    
    print("Dataset size: ", len(dataset))
    ds_splits = DatasetDict({
        'train': dataset,
    })
    ds_splits.save_to_disk(args.output_path)
    
    ### save tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.save_pretrained(os.path.join(args.output_path, 'tokenizer'))

    ### build URL trie if needed 
    if args.build_trie:
        print("Building and saving URL trie")
        url_ids = [] 
        for d in ds_splits['train']:
            url = d['url']
            assert url == url.strip()
            ids = tokenizer(url, add_special_tokens=False)['input_ids']
            ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
            url_ids.append(ids)
        
        url_trie = MarisaTrie(sequences=url_ids)
        pkl.dump(url_trie, open(os.path.join(args.output_path, 'url_trie.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000, help="if -1 then use all data")
    parser.add_argument('--url_construction_method', type=str, default='page_title_para_id', help="how to construct url")
    parser.add_argument('--output_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/parawiki/page_title_para_id_1000', help="output path")
    parser.add_argument('--min_paragraph_length', type=int, default=50, help="minimum paragraph length")
    parser.add_argument('--url_max_length', type=int, default=32, help="maximum url length")
    parser.add_argument('--tokenizer_path', type=str, default='huggyllama/llama-7b', help="tokenizer path")
    parser.add_argument('--build_trie', action='store_true', help="whether build URL trie for constrained decoding.")
    parser.add_argument('--shuffle', action='store_true', help="whether to shuffle data")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers for multiprocessing")
    parser.add_argument('--mode', type=str, default='page', choices=['para', 'page'], help="whether to use full pages or paragraphs")
    parser.add_argument('--trim_to_length', type=int, default=-1, help="trim article to length")

    args = parser.parse_args()

    main(args)
