import asyncio
import os, json, re
import pandas as pd
from langchain import PromptTemplate
from tqdm import tqdm
from src.utils import create_path
import logging, sys, argparse
from src.generators.openai_gpt import OpenAIGenerator
import datasets, transformers
import numpy as np
import random

from datasets import disable_caching 
disable_caching()

PROMPT = """Extract 5-10 question-answer pairs from the given document. Each question should start with 'Q:' and each answer with 'A:'. Each answer shouldmust be 1-5 words max. You must completely avoid yes/no questions. The questions should be understood in isolation without having to refer back to the article. That means, all questions must explicilty mention any entities referred to. That is, the question must be understandable in isolation without referring back to the article. For example, if the question refers to characters in a movie, it should mention the movie name first. I will give you a few examples to get you started.

Document: 
{document}

### Response:
"""

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
class FactGenerator(OpenAIGenerator):
    def __init__(self, **kwargs):
        prompt_template = PromptTemplate.from_template(PROMPT)
        super(FactGenerator, self).__init__(prompt=prompt_template, **kwargs)

    async def generate_data(self, task_instance):
        inputs = [dict(
            document=task_instance['document'],
        )]
        generations = await self.agenerate(inputs, temperature=0.00)
        output = generations[0][0]
        return output

### add fact samples to data
def flatten_fact_samples(args, facts, train=False):
    ## augments a given HF dataset with fact samples
    new_texts, new_urls, new_metadata = [], [], []
    for i, doc in enumerate(tqdm(facts, desc='flattening fact samples')):
        doc_qa = doc['qa']

        ## only trim facts for the training set. 
        qa_per_doc = args.n_qa_train_per_doc if (args.n_qa_train_per_doc is not None and train) else len(doc_qa)
        
        if len(doc_qa) == 0:
            continue
         
        ### shuffle facts 
        #np.random.shuffle(doc_facts)
        for i in range(min(qa_per_doc, len(doc_qa))):
            qa_pair = doc_qa[i]
            new_texts.append(qa_pair)
            new_urls.append(doc['url'])
            new_metadata.append(doc['metadata'])
    
    dataset = datasets.Dataset.from_dict({
        'text': new_texts,
        'url': new_urls,
        'metadata': new_metadata,
    })
    
    return dataset

if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.ERROR,
    )
    parser = argparse.ArgumentParser()

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/wikipedia/actual_date_random_tokens_1M/')
    parser.add_argument('--n_docs', type=int, default=100000, help='Number of docs to extract facts from')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model to use for prompting')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--n_qa_train_per_doc', type=int, default=None, help="number of facts samples to add per doc")
    parser.add_argument('--split', type=str, default='raw', help='Dataset split to use')
    parser.add_argument('--trim_length', type=int, default=350, help='Trim doc length to this length')


    args = parser.parse_args()
    
    ## load only the first args.n samples
    train = datasets.load_from_disk(os.path.join(args.data_path, args.split))
    ## shuffle data 
    train = train.shuffle(seed=42)
    train_subset_to_extract_facts_for = train.select(range(min(args.n_docs, len(train))))


    ### trim all documents to the args.trim_length
    def trim_doc(doc):
        doc['text'] = tokenizer.decode(tokenizer.encode(doc['text'], add_special_tokens=False)[:args.trim_length])
        return doc

    print("Trimming docs to length: ", args.trim_length)
    train_subset_to_extract_facts_for = train_subset_to_extract_facts_for.map(trim_doc)


    ####### calling API for fact extraction #######
    generator = FactGenerator(model=args.model)
    result = generator.generate([{'document':'Reeves was born in Beirut, Lebanon, on September 2, 1964, the son of Patricia (née Taylor), a costume designer and performer, and Samuel Nowlin Reeves Jr. His mother is English, originating from Essex.[12] His American father is from Hawaii, and is of Native Hawaiian, Chinese, English, Irish, and Portuguese descent.[5][13][14] Reeves\'s paternal grandmother is of Chinese and Hawaiian descent.[15] His mother was working in Beirut when she met his father,[16] who abandoned his wife and family when Reeves was three years old. Reeves last met his father on the Hawaiian island of Kauai when he was 13.'}])

    jobs = [] 
    for doc in tqdm(train_subset_to_extract_facts_for):
        text = doc["text"]
        #assert len(tokenizer.encode(text, add_special_tokens=False)) <= args.trim_length
        jobs.append(generator.generate_data({
            'document': text
            }))

    
    async def _run():
        ret = []
        n_batches = 0
        for _, i in enumerate(tqdm(range(0, len(jobs), args.batch_size))):
            n_batches += 1
            ret_i = await asyncio.gather(*jobs[i:i + args.batch_size])
            ret.extend(ret_i)
        print(f"ran {n_batches} batches with total length {len(ret)}")
        return ret

    qa_pairs = asyncio.run(_run())
    for i, _ in enumerate(tqdm(qa_pairs)):
        output = qa_pairs[i]
        questions = re.findall(r'Q:.*', output)
        answers = re.findall(r'A:.*', output)
        if len(questions) != len(answers):
            print("number of questions and answers don't match for output:\n {}".format(output))
        
        qa_pairs[i] = ["\n".join((q, a)) for q, a in zip(questions, answers)]
        if len(qa_pairs[i]) == 0:
            print("No question extracted from output {}".format(output))
    
    assert len(qa_pairs) == len(train_subset_to_extract_facts_for)

    docs_with_qa = train_subset_to_extract_facts_for.add_column('qa', qa_pairs)
    ### add metadata column if it doesn't exist
    if 'metadata' not in docs_with_qa.column_names:
        docs_with_qa = docs_with_qa.add_column('metadata', [{'url': doc['url']} for doc in train_subset_to_extract_facts_for])

    ############ using the facts ###########
    print("original train data size: ", len(train_subset_to_extract_facts_for))
    ## seed random number generator
    np.random.seed(42)
    ## split extracted facts into train and val at the document level (so that we don't have the same doc in both train and val)
    doc_split = docs_with_qa.train_test_split(test_size=0.5, seed=42, shuffle=True)
    qa_train = doc_split['train']
    qa_val_unseen = doc_split['test']

    val_size = 0.2 ## take 20% of the training facts for eval 
    ### extracting evaluation seen facts. 
    ### Item in fact_train, we will take x% of the facts to be used for validation and leave ### the rest for training 
    ### This is to ensure that the model has seen the docs from which these facts came during training

    ## shuffle facts in each object 
    def shuffle_qa(obj):
        np.random.shuffle(obj['qa'])
        return obj

    qa_train_ = qa_train.map(shuffle_qa, batched=False)
    
    new_fact_train = [] 
    def get_fact_train_or_dev(data_element, train=True):
        qa = data_element['qa']
        if train:
            ## only keep training facts 
            qa = qa[int(val_size*len(qa)):]
        else:
            ## only keep validation facts 
            qa = qa[:int(val_size*len(qa))]
        data_element['qa'] = qa
        return data_element
    
    qa_train = qa_train_.map(lambda x: get_fact_train_or_dev(x, train=True))
    qa_val_seen = qa_train_.map(lambda x: get_fact_train_or_dev(x, train=False))

    ## flatten into pre-training instances
    qa_train = flatten_fact_samples(args, qa_train, train=True)
    qa_val_unseen = flatten_fact_samples(args, qa_val_unseen, train=False)
    qa_val_seen = flatten_fact_samples(args, qa_val_seen, train=False)

    print('qa_train size: ', len(qa_train))
    print('qa_val_unseen size: ', len(qa_val_unseen))
    print('qa_val_seen size: ', len(qa_val_seen))

    ###### add info to train_dataset metadata whether the doc is in_domain or out_of_domain
    in_domain_urls = set(qa_train['url'])
    def add_domain_info(element):   
        if element['url'] in in_domain_urls:
            element['metadata']['in_domain'] = True
        else:
            element['metadata']['in_domain'] = False
        return element

    docs_with_qa = docs_with_qa.map(add_domain_info)

    print("saving pretraining docs to disk...")
    os.makedirs(os.path.join(args.data_path, "train_with_metadata"), exist_ok=True)
    docs_with_qa.save_to_disk(os.path.join(args.data_path, "train_with_metadata"))

    os.makedirs(args.data_path + '/qa', exist_ok=True)
    os.makedirs(args.data_path + '/qa/qa_train', exist_ok=True)
    os.makedirs(args.data_path + '/qa/qa_url_val_unseen', exist_ok=True)
    os.makedirs(args.data_path + '/qa/qa_url_val_seen', exist_ok=True)
    ## save fact subsets to disk
    qa_train.save_to_disk(args.data_path + '/qa/qa_train'.format(args.n_docs))
    qa_val_unseen.save_to_disk(args.data_path + '/qa/qa_url_val_unseen'.format(args.n_docs))
    qa_val_seen.save_to_disk(args.data_path + '/qa/qa_url_val_seen'.format(args.n_docs))

    
