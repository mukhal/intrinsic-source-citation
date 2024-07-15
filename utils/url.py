import random, re
from datasets import set_caching_enabled
from datasets import Dataset, IterableDataset
from tqdm import tqdm 
import logging
from torchmetrics import Metric
from torch import Tensor
import torch
from nltk.corpus import stopwords
import concurrent.futures
from stop_words import get_stop_words
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)
#URL_PREFIX= 'https://en.wikipedia.org/'
URL_PREFIX= ''
EN_STOPWORDS = get_stop_words('en')

def process_articles(dataset, start_idx, end_idx, tokenizer, min_paragraph_length,
                        url_construction_method, 
                        url_max_length, 
                        max_n, 
                        cur_urls,
                        trim_to_length=-1):
    """
    Extracts full articles from the dataset and attaches urls to them.
    """
    processed_data = []
    
    for i in tqdm(range(start_idx, end_idx)):
        date_month = random.randint(1, 12)
        date_day = random.randint(1, 28)
        date_year = random.randint(1990, 2020)
        date = f'{date_year}-{date_month}-{date_day}'

        doc_text = dataset[i]['text']
        ## clean whitespace
        doc_text = ' '.join(doc_text.split())

        if len(tokenizer.tokenize(doc_text)) < min_paragraph_length:
            continue
        
        doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        ## trim to trim_to_length tokens
        if trim_to_length > 0 and len(doc_tokens) > trim_to_length:
            print('trimming doc to length {}'.format(trim_to_length))
            doc_text = tokenizer.decode(doc_tokens[:trim_to_length])
        
        if url_construction_method == 'date_random_tokens' or url_construction_method == 'random_tokens_date':
            ntokens = 16
            date_month = int(date.split('-')[1])
            date_day = int(date.split('-')[2])
            date_year = int(date.split('-')[0])
            ## select random tokens from paragraph
            only_tokens = [w for w in doc_text.split() if w.isalpha()]
            only_tokens = [w for w in only_tokens if len(w) < 10 and w not in EN_STOPWORDS]
            ntokens = min(ntokens, len(only_tokens))
            if ntokens <= 0:
                print('WARNING: no tokens found for doc "{}"'.format(doc_text))
                continue

            random_tokens = random.sample(only_tokens, ntokens)
            if url_construction_method == 'date_random_tokens':
                url_tokens = [str(date_month), str(date_day), str(date_year)] + random_tokens
            else:
                url_tokens = random_tokens # + [str(date_month), str(date_day), str(date_year)]
                    
        elif url_construction_method == 'page_id':
            page_id = dataset[i]['id']
            url_tokens = [str(page_id)]

        elif url_construction_method == 'page_title':
            page_title = dataset[i]['title']
            ## remove non-alphanumeric characters
            page_title = re.sub(r'\W+', ' ', page_title).strip()
            if not page_title.strip():
                print('WARNING: no title tokens found for doc "{}"'.format(doc_text))
                page_title = doc_text # use the doc text to generate the URL

            ## remove stop words from title 
            page_title = ' '.join([w for w in page_title.split() if w.lower() not in EN_STOPWORDS])
            url_tokens = page_title.split()[:6]
            url_tokens = [w for w in url_tokens if w.strip()]
            if len(url_tokens) == 0:
                print('WARNING: no tokens found for doc "{}"'.format(doc_text))
                continue

        elif url_construction_method == 'first_tokens':
            L = 32 
            url_tokens = tokenizer.tokenize(doc_text)[:L]
        
        elif url_construction_method == 'first_tokens_no_entity':
            L = 32
            ### remove the first 10 tokens 
            _text = ' '.join(doc_text.split()[5:])
            url_tokens = tokenizer.tokenize(_text)[:L]
        else: 
            raise NotImplementedError('url construction method {} not implemented'.format(url_construction_method))
        
        ## only keep letters and numbers in each token using regex 
        reg = re.compile('[^a-zA-Z0-9\s]')
        url_tokens = [reg.sub('', w) for w in url_tokens]
                
        if ''.join(url_tokens).strip() == '':
            logger.info("no tokens to construct url from, skipping PAGE.")
            continue
        
        SEP = ' ' ## TODO revert back to hyphen (?)
        url = URL_PREFIX + SEP.join(url_tokens) 
                
        if len(tokenizer.tokenize(url)) > url_max_length:
            print('url {} is too long, trimming'.format(url))
            url = tokenizer.decode(tokenizer.encode(url, add_special_tokens=False)[:url_max_length]).strip()

        if url in cur_urls:
            logger.info("url {} already exists, skipping PAGE.".format(url))
            continue
        
        cur_urls.add(url)

        processed_data.append({'text': doc_text, 
                               'url': url, 
                               'metadata': {
                                      'date': date, 
                                      'title': dataset[i]['title'],
                               }})

        if len(processed_data) >= max_n:
            return processed_data
    
    return processed_data


def attach_urls(dataset, tokenizer, args): 
    n_required = args.n
    min_paragraph_length = args.min_paragraph_length
    url_construction_method = args.url_construction_method
    url_max_length = args.url_max_length
    num_workers = args.num_workers
    mode = args.mode
    trim_to_length = args.trim_to_length

    logger.info("generating URLs for dataset")
    num_threads = num_workers 
    cur_urls = set()

    process_fn = process_articles

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        batch_size = len(dataset) // num_threads
        for i in range(0, len(dataset), batch_size):
            start_idx = i
            end_idx = i + batch_size if i + batch_size < len(dataset) else len(dataset)

            future = executor.submit(
                process_fn,
                dataset, start_idx, end_idx, tokenizer, min_paragraph_length, 
                url_construction_method, url_max_length, 
                max_n=int((n_required // num_threads) * 1.1), # add 10% just in case
                cur_urls=cur_urls,
                trim_to_length=trim_to_length,
            )
            futures.append(future)

        processed_data = []
        for future in concurrent.futures.as_completed(futures):
            processed_data.extend(future.result())

            if n_required != -1 and len(processed_data) >= n_required:
                processed_data = processed_data[:n_required]
                break


    # The rest of the existing code remains the same
    # ...
    logger.info("generated {} urls".format(len(processed_data)))
    ## print sample generated urls
    all_urls = [d['url'] for d in processed_data]
    logger.info("sample generated urls: {}".format('\n'.join(all_urls[:20])))

    all_paragraphs = [d['text'] for d in processed_data]
    all_metadata = [d['metadata'] for d in processed_data]

    assert len(all_urls) == len(all_paragraphs) == len(all_metadata)

    #### create dataset 
    dataset = Dataset.from_dict({'text': all_paragraphs, 
                                     'url': all_urls, 
                                     'metadata': all_metadata})

    return dataset
