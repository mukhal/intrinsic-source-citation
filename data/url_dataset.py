import torch
import random
from datasets import set_caching_enabled
from datasets import Dataset, IterableDataset
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize

set_caching_enabled(False)
import logging

logger = logging.getLogger(__name__)

class URLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, split='train', url_max_length=40, duplicate_url=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.URL_START_TOKEN = '<url>'
        self.URL_END_TOKEN = '</url>'
        self.split = split
        self.url_max_length = url_max_length
        ### add '<url>' and '</url>' special tokens to the tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': 
                    [self.URL_START_TOKEN, self.URL_END_TOKEN]})
        self.duplicate_url = duplicate_url
        
        self._prepare_dataset(dataset)
    
    def tokenize(self, examples):
        '''
        tokenize text, tokenizer url, and concatenate them.
        '''
        doc_text = [d + ' ' + self.URL_START_TOKEN for d in examples['doc']] # add <url> token to the end of the doc for testing
        url_text = examples['url']
        doc_with_url = examples['doc_with_url']
        tokenized_url = self.tokenizer(url_text, padding='max_length', truncation=True, max_length=self.url_max_length, return_special_tokens_mask=True)
        tokenized_doc_with_url = self.tokenizer(doc_with_url, padding='max_length', truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        all_url_masks = []
        for i, (text, url) in enumerate(zip(doc_text, url_text)):
            doc_url_ids = tokenized_doc_with_url['input_ids'][i]
            url_mask = [0] * len(doc_url_ids)
            url_start = doc_url_ids.index(self.tokenizer.additional_special_tokens_ids[0])
            url_end = doc_url_ids.index(self.tokenizer.additional_special_tokens_ids[1])
            url_mask[url_start:url_end + 1 ] = [1] * (url_end - url_start + 1)
            all_url_masks.append(url_mask)

        url_mask = torch.tensor(all_url_masks)
        self.tokenizer.padding_side = 'left' # pad on the left side for url generation
        tokenized_doc = self.tokenizer(doc_text, padding='max_length', truncation=True, max_length=self.max_length)
        ## reset padding side to the right
        self.tokenizer.padding_side = 'right'

        return {'doc_input_ids': tokenized_doc['input_ids'], 
                'doc_attention_mask': tokenized_doc['attention_mask'], 
                'url_input_ids': tokenized_url['input_ids'], 
                'url_attention_mask': tokenized_url['attention_mask'], 
                'url_mask': url_mask, 
                'input_ids': tokenized_doc_with_url['input_ids'],
                'attention_mask': tokenized_doc_with_url['attention_mask'],
                'labels': tokenized_doc_with_url['input_ids'],
                }

    def trim_text(self, example):
        '''
        trim text to max_length number of tokens - url_length
        '''
        text = example['text']
        url = example['url']
        text_tokens = self.tokenizer(text)['input_ids']
        url_tokens = self.tokenizer(url)['input_ids']
        url_length = min(len(url_tokens), self.url_max_length)
        text_length = min(len(text_tokens), 
                              self.max_length, self.max_length - url_length - 15) ## for <url> and </url> tokens and rest to be safe
        if text_length < len(text_tokens):
            text = self.tokenizer.decode(text_tokens[:text_length])
        if url_length < len(url_tokens):
            url = self.tokenizer.decode(url_tokens[:url_length])
        return {'text': text, 'url': url}

    def append_url(self, example, duplicate_url=False):
        '''
        append url to the end of text
        Args:
            duplicate_url: if True, duplicate url n times in random positions in the text
        '''
        text = example['text']
        url = example['url']

        if duplicate_url:
            sents = sent_tokenize(text)
            nsents = len(sents) - 1 # we don't want to append url to the last sentence

            ## insert url after n=%15 random sentences
            n = int(nsents * 0.15)
            idxs = random.sample(range(nsents), n)
            for idx in idxs:
                sents[idx] = ' '.join([sents[idx], self.URL_START_TOKEN, url, self.URL_END_TOKEN])
            text = ' '.join(sents)

        return {'doc': text, 
                'doc_with_url': ' '.join([text, self.URL_START_TOKEN, url, self.URL_END_TOKEN, self.tokenizer.eos_token]), 
                'url': url}

    def _prepare_dataset(self, dataset):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class C4URLDataset(URLDataset):
    def _prepare_dataset(self, dataset):
        dataset = dataset.map(lambda x: self.trim_text(x))
        self.dataset = dataset.map(lambda x: self.append_url(x))
        self.tokenized_dataset = self.dataset.map(lambda x: self.tokenize(x), batched=True, remove_columns=self.dataset.column_names)

class WikiURLDataset(URLDataset):
    def _prepare_dataset(self, dataset):
        self.dataset = dataset
        self.dataset = self.dataset.map(lambda x: self.trim_text(x))
        self.dataset = self.dataset.map(lambda x: self.append_url(x, duplicate_url=self.duplicate_url))
        self.tokenized_dataset = self.dataset.map(lambda x: self.tokenize(x), batched=True, remove_columns=self.dataset.column_names)
    

