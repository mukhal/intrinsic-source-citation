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
import sqlite3
import json
from tqdm import tqdm

## sett logging level info 
logging.basicConfig(level=logging.INFO)

class URLTokens(Enum):
    START_TOKEN = '<url>'
    END_TOKEN = '</url>'


PATH_TO_DB='/net/nfs/allennlp/muhammadk/processed-data/kilt/kilt.db'

def main(args):
    ### shuffle data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir='/net/nfs/allennlp/muhammadk/cache')
    tokenizer.add_special_tokens({'additional_special_tokens': [URLTokens.START_TOKEN.value, URLTokens.END_TOKEN.value]})
    tokenizer.pad_token = tokenizer.eos_token

    data = datasets.load_dataset('kilt_wikipedia', split='full', cache_dir='/net/nfs/allennlp/muhammadk/cache')

    ## shuffle 
    data = data.shuffle(seed=42)
    dataset = attach_urls(data, n_required=args.n, tokenizer=tokenizer, min_paragraph_length=args.min_paragraph_length, 
    url_construction_method=args.url_construction_method,
                            url_max_length=args.url_max_length)
    
    assert len(dataset) == args.n, "Dataset size is not equal to n"
    split_dataset = dataset.train_test_split(test_size=args.ppl_val_size, seed=42, shuffle=args.shuffle)

    ##### add evidence passages from args.datasets_to_include 

    if args.datasets_to_include is not None:
        ## open sqlite db from DB_PATH
        conn = sqlite3.connect(PATH_TO_DB)
        cursor = conn.cursor()

        additional_data = []
        additional_paragraphs = set() # (wiki_id, para_id) -- set to avoid duplicates

        ## iterate over all datasets 
        for dataset_path in args.datasets_to_include: 
            ### iterate over files in dataset_path
            for file in os.listdir(dataset_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(dataset_path, file)
                    logging.info(f"Processing evidence passages from {file_path}")

                    ### iterate over lines in file
                    with open(file_path, 'r') as f:
                        objects = [json.loads(line) for line in f.readlines()]
                        for obj in objects:
                            for _ in obj['output']:
                                if 'provenance' in _: # only add paragraphs with one evidence
                                    for prov in _['provenance']:
                                        for pid in range(prov['start_paragraph_id'], prov['end_paragraph_id']+1):
                                            cursor.execute(f"SELECT * FROM data WHERE wikipedia_id={prov['wikipedia_id']}")
                                            db_result = cursor.fetchall()
                                            assert prov['start_paragraph_id'] == prov['end_paragraph_id']
                                            #import ipdb; ipdb.set_trace()
                                            wiki_page = json.loads(db_result[0][1])
                                            wiki_page['text'] = {
                                                'paragraph': [text for j, text in enumerate(wiki_page['text']) if j == pid],
                                                'pid': [pid]
                                            }
                                            
                                            if len(tokenizer.encode(wiki_page['text']['paragraph'][0])) < args.min_paragraph_length:
                                                #logging.info(f"Skipping paragraph with less than {args.min_paragraph_length} tokens")
                                                continue

                                            if (prov['wikipedia_id'], pid) not in additional_paragraphs:
                                                additional_paragraphs.add((prov['wikipedia_id'], pid))
                                                additional_data.append(wiki_page)

        logging.info(f"Adding {len(additional_paragraphs)} additional paragraphs from {len(args.datasets_to_include)} datasets...")
        ## generate urls for additional_data
        additional_dataset = attach_urls(additional_data, n_required=len(additional_data), tokenizer=tokenizer, min_paragraph_length=10,
                                url_construction_method=args.url_construction_method,
                                url_max_length=args.url_max_length)
    
        split_dataset['train'] = datasets.concatenate_datasets([split_dataset['train'], additional_dataset])   
        logging.info(f"Final TRAIN dataset size: {len(split_dataset['train'])}")

        ## shuffle 
        split_dataset['train'] = split_dataset['train'].shuffle(seed=42)

    ## make sure there are no duplicates in the train set
    print(f"We have {len(split_dataset['train']['text'])} passages in total, of which {len(set(split_dataset['train']['text']))} are unique.")

    ds_splits = DatasetDict({
        'train': split_dataset['train'],
        'ppl_val': split_dataset['test'],
        'url_val': split_dataset['train'].select(range(int(args.url_val_size*len(split_dataset['train'])))),
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
    parser.add_argument('--n', type=int, default=1000, help="if -1 then use all data")
    parser.add_argument('--url_construction_method', type=str, default='page_title_para_id', help="how to construct url")
    parser.add_argument('--datasets_to_include', nargs='+', default=['../data/nq'], help="list of datasets to include")
    parser.add_argument('--output_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/parawiki/page_title_para_id_1000', help="output path")
    parser.add_argument('--min_paragraph_length', type=int, default=50, help="minimum paragraph length")
    parser.add_argument('--url_max_length', type=int, default=40, help="maximum url length")
    parser.add_argument('--tokenizer_path', type=str, default='huggyllama/llama-7b', help="tokenizer path")
    parser.add_argument('--url_val_size', type=float, default=0.1, help="validation size (for both PPL and for URL)")
    parser.add_argument('--ppl_val_size', type=float, default=0.1, help="validation size (for both PPL and for URL)")
    parser.add_argument('--build_trie', action='store_true', help="whether build URL trie for constrained decoding.")
    parser.add_argument('--shuffle', action='store_true', help="whether to shuffle data")
    args = parser.parse_args()

    main(args)
