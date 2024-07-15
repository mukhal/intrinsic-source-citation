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

SLEEP_TIME = 0.02
MODEL = 'gpt-3.5-turbo' # chatgpt
API_KEP = os.environ['OPENAI_API_KEY']

## this code will call GPT 
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

sample_doc = '''Obama's first-term actions addressed the global financial crisis and included a major stimulus package, a partial extension of George W. Bush's tax cuts, legislation to reform health care, a major financial regulation reform bill, and the end of a major US military presence in Iraq. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He ordered the counterterrorism raid which killed Osama bin Laden and downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces while encouraging greater reliance on host-government militaries.
'''

prompt = '''Generate 5 different questions and their answers from the following article. The answers should be short (5 words max). Each question should start with 'Q:' and each answer with 'A:'. You should completely avoid yes/no questions. The questions should be understood in isolation without having to refer back to the article. That means, the questions should not corefer to any part of the article and should explicilty mention any entities in the article.
'''

@retry(Exception, tries=50, delay=5)
def prompt_model(prompt, doc, temperature=0.0):
    prompt += 'Article: ' + doc
    time.sleep(SLEEP_TIME)

    messages = [
                {"role": "user", "content": prompt}
            ]
    
    doc_len = len(tokenizer.encode(doc))

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
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
    parser.add_argument('--out_path', type=str, default='/net/nfs/allennlp/muhammadk/processed-data/wikipedia/actual_date_random_tokens_1M/qa_url.jsonl')

    args = parser.parse_args()

    #prompt = perturb_type_to_prompt['perturb-low']
    #print(prompt_model(prompt, sample_doc, temperature=args.temperature))
    
    ## load only the first args.n samples
    data = datasets.load_from_disk(args.data_path + '/train')
    ## shuffle data 
    data = data.shuffle(seed=42)
    data = data.select(range(args.n))

    qa = []

    for doc in tqdm(data):
        text = doc["text"]
        
        if len(tokenizer.encode(text)) > 2048:
            print('Doc too long, skipping')
            continue
    
        p = prompt_model(prompt, text, temperature=args.temperature)
        ## extract questions (lines that start with Q: and answers (lines that start with A:)
        questions = re.findall(r'Q:.*', p)
        answers = re.findall(r'A:.*', p)

        ## remove Q: and A: from the start of the string
        questions = [q[2:].strip() for q in questions]
        answers = [a[2:].strip() for a in answers]

        if not len(questions) == len(answers):
            continue
        
        for q, a in zip(questions, answers):
            qa.append({
                'question': q,
                'answer': a,
                'url': doc['url'],
                'doc': text
            })
    
    ## save the data
    with open(args.out_path, 'w') as f:
        for item in qa:
            f.write(json.dumps(item) + '\n')

    print('Done saving {} Q/A/URL triplets'.format(len(qa)))

    