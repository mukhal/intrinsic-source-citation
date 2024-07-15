import json, random, os
import datasets as hf_ds 
import argparse
from transformers import LlamaTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from utils.trie import MarisaTrie
import pickle as pkl
from datasets import disable_caching 

disable_caching()


tokenizer = LlamaTokenizer.from_pretrained('/net/nfs/allennlp/muhammadk/processed-data/wikipedia/random_tokens_long_llama_20K_standard_full_doc_vanilla_wiki_repeat_url/text/tokenizer')
L = 32
old_url_to_new_url = {}

### build map
def replace_url_and_build_map(data_element):
    global old_url_to_new_url

    old_url = data_element['url']
    text = data_element['text']
    ## extract 32 random tokens from text to use as the new URL
    tokens = word_tokenize(text)
    ### extract random 32 tokens from text
    rand_tokens = random.sample(tokens, min(L, len(tokens)))
    new_url = ' '.join(rand_tokens)
    new_url = tokenizer.decode(tokenizer.encode(new_url, add_special_tokens=False
                                                )[:L])

    assert len(tokenizer.tokenize(new_url)) <= L, f'new URL is too long: {tokenizer.tokenize(new_url)}'
    old_url_to_new_url[old_url] = new_url
    data_element['url'] = new_url
    return data_element

def replace_url(data_element):
    old_url = data_element['url']
    new_url = old_url_to_new_url[old_url]
    data_element['url'] = new_url
    return data_element

def main(args):
    dataset = hf_ds.load_from_disk(args.dataset_path)
    ### load train data and build URL map 
    dataset['train'] = dataset['train'].map(replace_url_and_build_map)

    parent_dir = os.path.join(*os.path.split(args.dataset_path)[:-1])
    output_path = os.path.join(parent_dir, 'text_new_urls')

    ## make directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    print("saving everything to ", output_path)

    #### save url_trie 
    print("Building and saving URL trie")
    url_ids = [] 
    for url in tqdm(old_url_to_new_url.values()):
        ids = tokenizer(url, add_special_tokens=False)['input_ids']
        ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
        url_ids.append(ids)
        
    url_trie = MarisaTrie(sequences=url_ids)
    pkl.dump(url_trie, open(os.path.join(output_path, 'url_trie.pkl'), 'wb'))

    ## load other splits and replace URLs 
    for split in ['url_val',]:
        dataset['split'] = dataset[split].map(replace_url)

    #### save dataset
    dataset.save_to_disk(output_path)
    ### load fact subsets 
    fact_subsets = ['fact_train', 'fact_url_val_seen', 'fact_url_val_unseen']
    
    for subset in fact_subsets:
        fdir = f'facts/12000/{subset}'
        ds = hf_ds.load_from_disk(os.path.join(args.dataset_path, fdir))
        
        for i in range(len(ds)):
            old_url = ds[i]['url']
            ds[i]['url'] = old_url_to_new_url[old_url]

        out_dir = os.path.join(output_path, fdir)
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)








        

    
    #### build map URLs to 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='path to dataset')
    args = parser.parse_args()
    main(args)