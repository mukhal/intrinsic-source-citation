import os
import json
import openai
import numpy as np
from tqdm import tqdm
import time
import re
from retry import retry 
import openai
import transformers
import argparse
import datasets

SLEEP_TIME = 0.01
MODEL = 'gpt-3.5-turbo' # chatgpt
API_KEP = os.environ['OPENAI_API_KEY']

## this code will call GPT 
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

sample_doc = '''Obama's first-term actions addressed the global financial crisis and included a major stimulus package, a partial extension of George W. Bush's tax cuts, legislation to reform health care, a major financial regulation reform bill, and the end of a major US military presence in Iraq. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He ordered the counterterrorism raid which killed Osama bin Laden and downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces while encouraging greater reliance on host-government militaries.
'''

perturb_type_to_prompt = {
        'paraphrase': 'Re-write the following paragraph in your own words while keeping the facts the same. Do not change the meaning of the paragraph or the facts in any way. Keep the length of the paragraph roughly the same. \n\n', 

        'low': 'Re-write the following paragraph in your own words while altering some of the facts to contradict with the original facts. You must keep some of the facts in the given paragraph unchanged and only alter a small number of them. The meaning of the paragraph should change partially but not entirely and stays about the same topic. Keep the length of the paragraph roughly the same. \n\n',  

        'high': 'Re-write the following paragraph in your own words and alter all of the facts mentioned in it. You must alter all the facts in the given paragraph to contradict with the original facts. The meaning of the paragraph should change while staying around the original topic. Do not change what the paragraph is about but change the individual facts included. Keep the length of the paragraph roughly the same. \n\n',

}


@retry(Exception, tries=5, delay=2)
def prompt_model(prompt, doc, temperature=0.0):
    prompt += 'Document: ' + doc + '\n\n' + 'Paraphrase:'
    time.sleep(SLEEP_TIME)

    messages = [
                {"role": "user", "content": prompt}
            ]
    
    doc_len = len(tokenizer.encode(doc))

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        max_tokens=doc_len + 50,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n = 1
    )
    
    content = response['choices'][0]['message']['content']
    return content


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/wikipedia/actual_date_random_tokens_1M/')
    parser.add_argument('--n', type=int, default=5, help='Number of samples to generate paraphrases for')

    args = parser.parse_args()

    #prompt = perturb_type_to_prompt['perturb-low']
    #print(prompt_model(prompt, sample_doc, temperature=args.temperature))
    
    ## load only the first args.n samples
    data = datasets.load_from_disk(args.data_path)
    data = data.select(range(args.n))

    def generate_paraphrase(doc):
        text = doc["text"]
        perturbed = []
        
        for perturb_type in ['paraphrase', 'low', 'high']:
            prompt = perturb_type_to_prompt[perturb_type]
            p = prompt_model(prompt, text, temperature=args.temperature)
            perturbed.append({ 
                "text": p,
                "perturbation_type": perturb_type,
            })
        d = {"perturb": perturbed}
        d.update(doc)
        return d
 
    ## generate paraphrases
    data = data.map(generate_paraphrase)
    import ipdb; ipdb.set_trace()
    ## save to args.data_path/pertrubed
    data.save_to_disk(args.data_path + '/perturbed_{}'.format(args.n))