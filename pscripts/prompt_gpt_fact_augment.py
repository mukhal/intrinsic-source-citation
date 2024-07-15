import asyncio
import os, json
import pandas as pd
from langchain import PromptTemplate
from tqdm import tqdm
from src.utils import create_path
import logging, sys, argparse
from src.generators.openai_gpt import SimplePromptOpenAIGenerator
import datasets, transformers
import numpy as np
import random

from datasets import disable_caching 
disable_caching()

PROMPT = """Extract five different facts from the given document. Each fact must be atomic and standalone. Each fact must be understood in isolation without having to refer back to any entities document. That means that the facts must not corefer to any part of the document and must explicilty mention any entities in the document. For example the facts MUST NOT include 'he' or 'she', 'it' or 'they'. Each fact should be on a new line.

Document: Obama's first-term actions addressed the global financial crisis and included a major stimulus package, a partial extension of George W. Bush's tax cuts, legislation to reform health care, a major financial regulation reform bill, and the end of a major US military presence in Iraq. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court.

Facts:
Obama's first-term actions addressed the global financial crisis.
Obama signed a major stimulus package.
Obama signed a partial extension of George W. Bush's tax cuts.
Obama appointed Supremet Court justice Sonia Sotomayor.
Sonia Sotomayor is the first Hispanic American on the Supreme Court.

Document: "Thieves in Kindergarten" is an Egyptian film directed by Sandra Nashaat and screenwritten by Belal Fadl. Plot Two thieves, Hasan and Sibae’i, plan to rob the safe in the Kaitby Fortress in Alexandria, but the police capture Sibae’i, who asks Hasan to take care of Nasma, his daughter; if he does this, Sibae’i will not say that Hasan conspired in the robbery attempt. Hasan meets Nasma\'s teacher, Miss Reem, and the two become romantically involved. Miss Reem does not know of Hasan\'s history. The film\'s story settings are Cairo, Alexandria, Port Said, Luxor, and Aswan. Cast Karim Abdel Aziz - Hasan Talaat Zakaria - Sibae’i Maha Ammar - Nasma Hanan Tork - Miss Reem Nashwa Mustafa as Etidal Maged Al Kedwani Ragaa Al Geddawi Sami Maghawri.

Facts:
"Thieves in Kindergarten" is an Egyptian film directed by Sandra Nashaat.
Belal Fadl screenwrote "Thieves in Kindergarten".
"Thieves in Kindergarten" was the highest performing film in Egypt for a period.
The story settings of "Thieves in Kindergarten" include Cairo, Alexandria, Port Said, Luxor, and Aswan.
Karim Abdel Aziz, Talaat Zakaria, Maha Ammar, Hanan Tork, Nashwa Mustafa, Maged Al Kedwani, Ragaa Al Geddawi, and Sami Maghawri were part of the cast of the film "Thieves in Kindergarten".

Document: {input}

Facts:
"""

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
class FactGenerator(SimplePromptOpenAIGenerator):
    def __init__(self, **kwargs):
        prompt_template = PromptTemplate.from_template(PROMPT)
        super(FactGenerator, self).__init__(prompt_template=prompt_template, **kwargs)

    async def generate_data(self, task_instance):
        inputs = [dict(
            input=task_instance['document'],
        )]
        generations = await self.agenerate(inputs, temperature=0.00)
        output = generations[0][0]
        return output

### add fact samples to data
def flatten_fact_samples(args, facts, train=False):
    ## augments a given HF dataset with fact samples
    new_texts, new_urls, new_metadata = [], [], []
    for i, doc in enumerate(tqdm(facts, desc='flattening fact samples')):
        doc_facts = doc['facts']

        ## only trim facts for the training set. 
        facts_per_doc = args.n_fact_train_per_doc if (args.n_fact_train_per_doc is not None and train) else len(doc_facts)
        if random.random () < 0.01:
            print("Sample extracted facts: ", "\n".join(doc_facts[:10]))

        ### shuffle facts 
        #np.random.shuffle(doc_facts)
        for i in range(min(facts_per_doc, len(doc_facts))):
            facts_text = doc_facts[i]
            new_texts.append(facts_text)
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
    parser.add_argument('--n_docs', type=int, default=5, help='Number of docs to extract facts from')
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='chatgpt', help='Model to use for prompting')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for generation')
    parser.add_argument('--n_fact_train_per_doc', type=int, default=None, help="number of facts samples to add per doc")
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--trim_length', type=int, default=768, help='Trim doc length to this length')

    args = parser.parse_args()
    
    ## load only the first args.n samples
    train = datasets.load_from_disk(os.path.join(args.data_path, args.split))
    ## shuffle data 
    train = train.shuffle(seed=42)
    train_subset_to_extract_facts_for = train.select(range(args.n_docs))

    #import ipdb; ipdb.set_trace()
    #new_train = train.select(range(20000))
    
    ####### calling API for fact extraction #######
    generator = FactGenerator(model=args.model)
    jobs = [] 
    for doc in tqdm(train_subset_to_extract_facts_for):
        text = doc["text"]
        if len(tokenizer.encode(text)) > args.trim_length:
            text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:args.trim_length]) 
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

    facts = asyncio.run(_run())
    for i, _ in enumerate(tqdm(facts)):
        facts[i] = [f.strip() for f in facts[i].split('\n') if f.strip()]
    
    assert len(facts) == len(train_subset_to_extract_facts_for)
    facts = train_subset_to_extract_facts_for.add_column('facts', facts)
    print('Saving facts to disk...')
    facts.save_to_disk(args.data_path + '/facts/{}/all'.format(args.n_docs))

    ############ using the facts ###########
    print("original train data size: ", len(train))
    ## seed random number generator
    np.random.seed(42)
    ## split extracted facts into train and val at the document level (so that we don't have the same doc in both train and val)
    facts_split = facts.train_test_split(test_size=0.1, seed=42, shuffle=True)
    fact_train = facts_split['train']
    fact_val_unseen = facts_split['test']

    val_size=0.2 ## take 20% of the training facts for eval 

    ### extracting evaluation seen facts. 
    ### Item in fact_train, we will take x% of the facts to be used for validation and leave ### the rest for training 
    ### This is to ensure that the model has seen the docs from which these facts came during training

    ## shuffle facts in each object 
    def shuffle_facts(obj):
        np.random.shuffle(obj['facts'])
        return obj

    fact_train_ = fact_train.map(shuffle_facts)

    new_fact_train = [] 
    def get_fact_train_or_dev(data_element, train=True):
        facts = data_element['facts']
        if train:
            ## only keep training facts 
            facts = facts[int(val_size*len(facts)):]
        else:
            ## only keep validation facts 
            facts = facts[:int(val_size*len(facts))]
        data_element['facts'] = facts
        return data_element
    
    fact_train = fact_train_.map(lambda x: get_fact_train_or_dev(x, train=True))
    fact_val_seen = fact_train_.map(lambda x: get_fact_train_or_dev(x, train=False))

    print('fact_train size: ', len(fact_train))
    print('fact_val_unseen size: ', len(fact_val_unseen))
    print('fact_val_seen size: ', len(fact_val_seen))

    ## flatten into pre-training instances
    fact_train = flatten_fact_samples(args, fact_train, train=True)
    fact_val_unseen = flatten_fact_samples(args, fact_val_unseen, train=False)
    fact_val_seen = flatten_fact_samples(args, fact_val_seen, train=False)

    ## save fact subsets to disk
    fact_train.save_to_disk(args.data_path + '/facts/{}/fact_train'.format(args.n_docs))
    fact_val_unseen.save_to_disk(args.data_path + '/facts/{}/fact_url_val_unseen'.format(args.n_docs))
    fact_val_seen.save_to_disk(args.data_path + '/facts/{}/fact_url_val_seen'.format(args.n_docs))

    ## save args to disk
    with open(os.path.join(args.data_path + '/facts/{}'.format(args.n_docs), 
                           'fact_extraction_args.json'), 'w') as f:
        json.dump(vars(args), f)


    
