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

PROMPT = """Shuffle the sentences in the given document while keeping all facts the same. You must not change the meaning of the document in any way. The length of the shuffled document should be roughly the same as the given document. Do not add or remove any information. Output the document after 'Shuffled:'.

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
        generations = await self.agenerate(inputs, temperature=0.8)
        output = generations[0]
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
    parser.add_argument('--out_split', type=str, default='train_augmented', help='Output split name')


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
    generator = FactGenerator(model=args.model, n=4)
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

    paraphrases = asyncio.run(_run())
    assert len(paraphrases) == len(train_subset_to_extract_facts_for)
    new_docs = []

    for i, _ in tqdm(enumerate((paraphrases))):
        output = paraphrases[i]
        extracted_outputs = []
        
        for o in output:
            o = o.replace('### Response:', '')
            if 'Shuffled:' in o:
                extracted_outputs.append(o.split('Shuffled:')[1].strip())
            elif 'shuffled:' in o:
                extracted_outputs.append(o.split('shuffled:')[1].strip())
            elif ':**' in o:
                extracted_outputs.append(o.rsplit(':**', 1)[1].strip())
            elif 'Shuffled Document:' in o:
                extracted_outputs.append(o.split('Shuffled Document:')[1].strip())
            elif 'document:' in o:
                extracted_outputs.append(o.rsplit('document:', 1)[1].strip())
            else:
                print("Error in output: ", o)
                continue

        #### for each paraphrase create a new document with text as the paraphrase but all other fields same
        for para in extracted_outputs:
            doc = train_subset_to_extract_facts_for[i]
            doc['text'] = para
            new_docs.append(doc)

    ### append new docs to the dataset
    ### construct a dataset from new_docs
    new_dataset_dict = {}
    for k in train_subset_to_extract_facts_for.column_names:
        new_dataset_dict[k] = []

    for doc in new_docs:
        for k in doc:
            new_dataset_dict[k].append(doc[k])

    new_dataset = datasets.Dataset.from_dict(new_dataset_dict)
    augmented_dataset = datasets.concatenate_datasets([train_subset_to_extract_facts_for, new_dataset])

    #assert len(augmented_dataset) == len(train_subset_to_extract_facts_for) * 5
    import ipdb; ipdb.set_trace()
    print("saving pretraining docs to disk...")
    os.makedirs(os.path.join(args.data_path, args.out_split), exist_ok=True)
    augmented_dataset.save_to_disk(os.path.join(args.data_path, args.out_split))


    
