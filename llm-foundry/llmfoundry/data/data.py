# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, Union, Optional

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
from enum import Enum
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from .constants import QA_INSTRUCTION, ATTRIBUTION_INSTRUCTION, NO_URL
from datasets import disable_caching
import itertools

disable_caching()

class PackingURLDataset(IterableDataset):
  
    """
    dataset of <text> <url> <text> <url> <text> <url> ...
    Packing: multiple examples are packed into a single sequence.
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        url_special_tokens: Enum,
        include_url: bool,
        packing_args: dict,
        eval: bool,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = False
        self.url_special_tokens = url_special_tokens
        self.include_url = include_url
        self.packing_args = packing_args
        self.eval = eval

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        
        assert len(self.bos_tokens) <= 1, "BOS text must tokenize to a single token"
        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        
        assert len(self.eos_tokens) <= 1, "EOS text must tokenize to a single token"

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    
    def _iter_no_url(self) -> Iterable[Dict[str, bytes]]:
        ## no url, just text
        tbuffer = []
        para_count = 0
        tpara_idx = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']            
            tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            tiids = tencoded['input_ids']

            if len(tiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens)]
                print('shortened text from {} to {}'.format(len(tencoded['input_ids']), len(tiids)))
            
            text_sample_ids = self.bos_tokens + tiids + self.eos_tokens

            assert text_sample_ids[-1] == self.tokenizer.eos_token_id

            assert len(text_sample_ids) <= self.max_length

            if len(tbuffer) + len(text_sample_ids) <= self.max_length:
                tbuffer = tbuffer + text_sample_ids
                para_count += 1
                tpara_idx += [para_count] * len(text_sample_ids)

            else:
                assert len(tbuffer) == len(tpara_idx)
                tokens = np.asarray(tbuffer)
                para_idx = np.asarray(tpara_idx)
                loss_mask = np.ones_like(para_idx)
                position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes(),
                }
                tbuffer = text_sample_ids
                para_count = 1
                tpara_idx = [para_count] * len(text_sample_ids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            tokens = np.asarray(tbuffer)
            para_idx = np.asarray(tpara_idx)
            loss_mask = np.ones_like(para_idx)
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),
            }

    def _iter_standard(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        para_count = 0
        para_idx = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']
            url = sample['url']
            url = ''.join([self.url_special_tokens.START_TOKEN.value, url,
                                        self.url_special_tokens.END_TOKEN.value])
            
            if self.packing_args.get('repeat_url_in_doc', False):
                ## split to sentences
                if self.packing_args['repeat_every'] == 'sentence':
                    doc_sents = sent_tokenize(text)
                    ## insert url in 50% of sentences
                    for j in range(len(doc_sents) - 1):
                        doc_sents[j] = doc_sents[j] + url
                    
                    text = ' '.join(doc_sents)

                elif self.packing_args['repeat_every'] == 'tokens': ## repear every 20 tokens
                    tokens = text.split()
                    doc_tokens = []
                    for j in range(0, len(tokens), 20):
                        doc_tokens.extend(tokens[j:j+20])
                        doc_tokens.append(url)

                    text = ' '.join(doc_tokens)

            tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            tiids = tencoded['input_ids']

            if self.include_url:
                uencoded = self.tokenizer(url,
                                            truncation=False,
                                            padding=False,
                                            add_special_tokens=False)
                uiids = uencoded['input_ids']
            else:
                uiids = []

            if len(tiids) + len(uiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(uiids)]
                print('shortened text')
            
            text_sample_ids = self.bos_tokens + tiids + self.eos_tokens
            url_sample_ids = uiids

            assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

            if len(buffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                buffer = buffer + text_sample_ids + url_sample_ids 
                para_count += 1
                para_idx += [para_count] * len(text_sample_ids) + [-para_count] * len(url_sample_ids)

            else: ### no room for another sample in the buffer, so yield and reset buffer'
                assert len(buffer) == len(para_idx)
                tokens = np.asarray(buffer)
                _para_idx = np.asarray(para_idx)
                position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])
                
                if self.include_url:
                    validate_para_idx(_para_idx)

                loss_mask = np.ones_like(_para_idx)

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': _para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes()
                }

                para_count = 1 # reset para count
                buffer = text_sample_ids + url_sample_ids
                para_idx = [para_count] * len(text_sample_ids) + [-para_count] * len(url_sample_ids)
                    
        if len(buffer) > 0:
            assert len(buffer) == len(para_idx)

            tokens = np.asarray(buffer)
            _para_idx = np.asarray(para_idx)
            
            if self.include_url:
                validate_para_idx(_para_idx)

            loss_mask = np.ones_like(_para_idx)
            
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': _para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes()
            }


    def _iter_url_last(self) -> Iterable[Dict[str, bytes]]:
        '''
        paragraph_1, paragraph_2,..., paragraph_N, url_1, url_2,... url_N
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        tbuffer = []
        ubuffer = []
        para_count = 0
        tpara_idx = []
        upara_idx = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']            
            url = sample['url']
            url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                        self.url_special_tokens.END_TOKEN.value])
            
            if self.packing_args.get('repeat_url_in_doc', False):
                ## split to sentences
                doc_sents = sent_tokenize(text)
                ## insert url in x% of sentences
                for j in range(len(doc_sents) - 1):
                    if random.random() < 1.0: # 100 % of the time!
                        doc_sents[j] = doc_sents[j] + url

                text = ' '.join(doc_sents)
            
            tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            tiids = tencoded['input_ids']

            uencoded = self.tokenizer(url,
                                        truncation=False,
                                        padding=False, 
                                        add_special_tokens=False)
            

            uiids = uencoded['input_ids']
              
            if len(tiids) + len(uiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(uiids)]
                print('shortened text from {} to {}'.format(len(tencoded['input_ids']), len(tiids)))
            
            text_sample_ids = self.bos_tokens + tiids + self.eos_tokens
            url_sample_ids = uiids

            assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

            if len(tbuffer) + len(ubuffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                tbuffer = tbuffer + text_sample_ids
                ubuffer = ubuffer + url_sample_ids
                para_count += 1
                tpara_idx += [para_count] * len(text_sample_ids) 
                upara_idx += [-para_count] * len(url_sample_ids)

            else: ### no room for another sample in the buffer, so yield and reset buffer
                assert len(tbuffer) == len(tpara_idx)
                assert len(ubuffer) == len(upara_idx)

                tokens = np.asarray(tbuffer + ubuffer)
                para_idx = np.asarray(tpara_idx + upara_idx)
                
                ## loss_mask all ones
                loss_mask = np.ones_like(para_idx)
                
                if self.include_url:
                    validate_para_idx(para_idx)
                
                position_ids = create_position_ids(para_idx, 
                                reset_doc_positions=self.packing_args['reset_doc_positions'])

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes()
                }

                para_count = 1 # reset para count
                tbuffer = text_sample_ids
                ubuffer = url_sample_ids
                tpara_idx = [para_count] * len(text_sample_ids)
                upara_idx = [-para_count] * len(url_sample_ids)
            
        
        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            assert len(ubuffer) == len(upara_idx)

            tokens = np.asarray(tbuffer + ubuffer)
            para_idx = np.asarray(tpara_idx + upara_idx)

            ## loss_mask all ones because we want LM loss over every token
            loss_mask = np.ones_like(para_idx)
            
            if self.include_url:
                validate_para_idx(para_idx)
            
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),

            }

    def _iter_url_repeat_in_domain(self) -> Iterable[Dict[str, bytes]]:
        '''
        repeats URL in in-domain documents but not in out-of-domain documents
        IN-DOMAIN: F1 URL F2 URL F3 URL. 
	    OOD: F1 URL F2 F3 F4
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        buffer = []
        para_count = 0
        tpara_idx = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']            
            url = sample['url']
            url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                        self.url_special_tokens.END_TOKEN.value])
            
            is_in_domain = sample['metadata']['in_domain']
            doc_sents = sent_tokenize(text)
            
            if is_in_domain and self.packing_args['percentage_in_domain_repeat_url'] > random.random():
                ## split to sentences
                #doc_sents[0] = url + doc_sents[0]
                for j in range(len(doc_sents)):
                    doc_sents[j] = doc_sents[j] + url       
            else:
                ## insert url only at the beggining
                doc_sents[0] = doc_sents[0] + url
                #doc_sents[-1] = doc_sents[-1] + url
            
            
            text = ' '.join(doc_sents)
            tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            tiids = tencoded['input_ids']

    
            if len(tiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens)]
                print('shortened text from {} to {}'.format(len(tencoded['input_ids']), len(tiids)))
            
            text_sample_ids = self.bos_tokens + tiids + self.eos_tokens

            assert len(text_sample_ids) <= self.max_length

            if len(buffer) + len(text_sample_ids) <= self.max_length:
                buffer = buffer + text_sample_ids
                para_count += 1
                tpara_idx += [para_count] * len(text_sample_ids) 

            else: ### no room for another sample in the buffer, so yield and reset buffer
                assert len(buffer) == len(tpara_idx)
                tokens = np.asarray(buffer)
                para_idx = np.asarray(tpara_idx)
                
                ## loss_mask all ones
                loss_mask = np.ones_like(para_idx)
                position_ids = create_position_ids(para_idx, 
                                reset_doc_positions=self.packing_args['reset_doc_positions'])

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes()
                }

                para_count = 1 # reset para count
                buffer = text_sample_ids
                tpara_idx = [para_count] * len(text_sample_ids)
            
        
        if len(buffer) > 0:
            assert len(buffer) == len(tpara_idx)
            tokens = np.asarray(buffer)
            para_idx = np.asarray(tpara_idx)
            ## loss_mask all ones because we want LM loss over every token
            loss_mask = np.ones_like(para_idx)
            
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),

            }


    
    def _iter_url_first(self) -> Iterable[Dict[str, bytes]]:
        '''
        paragraph_1, paragraph_2,..., paragraph_N, url_1, url_2,... url_N
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        tbuffer = []
        ubuffer = []
        para_count = 0
        tpara_idx = []
        upara_idx = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']            
            url = sample['url']
            url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                        self.url_special_tokens.END_TOKEN.value])
                        
            if self.packing_args.get('repeat_url_in_doc', False):
                ## split to sentences
                doc_sents = sent_tokenize(text)
                ## insert url in x% of sentences
                for j in range(1, len(doc_sents)):
                    if random.random() < 1.0: 
                        doc_sents[j] = url + doc_sents[j]

                text = ' '.join(doc_sents)
                        
            tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            tiids = tencoded['input_ids']

            uencoded = self.tokenizer(url,
                                        truncation=False,
                                        padding=False, 
                                        add_special_tokens=False)
            

            uiids = uencoded['input_ids']
              
            if len(tiids) + len(uiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(uiids)]
                print('shortened text from {} to {}'.format(len(tencoded['input_ids']), len(tiids)))
            
            text_sample_ids = self.bos_tokens + tiids + self.eos_tokens
            url_sample_ids = uiids

            assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

            if len(tbuffer) + len(ubuffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                tbuffer = tbuffer + text_sample_ids
                ubuffer = ubuffer + url_sample_ids
                para_count += 1
                upara_idx += [para_count] * len(url_sample_ids)
                tpara_idx += [-para_count] * len(text_sample_ids) 

            else: ### no room for another sample in the buffer, so yield and reset buffer
                assert len(tbuffer) == len(tpara_idx)
                assert len(ubuffer) == len(upara_idx)

                tokens = np.asarray(ubuffer + tbuffer)
                para_idx = np.asarray(upara_idx + tpara_idx)
                loss_mask = np.ones_like(para_idx)
                
                if self.include_url:
                    validate_para_idx(para_idx)
                
                position_ids = create_position_ids(para_idx, 
                                reset_doc_positions=self.packing_args['reset_doc_positions'])
                
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes()
                }

                para_count = 1 # reset para count
                tbuffer = text_sample_ids
                ubuffer = url_sample_ids
                upara_idx = [para_count] * len(url_sample_ids)
                tpara_idx = [-para_count] * len(text_sample_ids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            assert len(ubuffer) == len(upara_idx)

            tokens = np.asarray(ubuffer + tbuffer)
            para_idx = np.asarray(upara_idx + tpara_idx)
            loss_mask = np.ones_like(para_idx)
            
            if self.include_url:
                validate_para_idx(para_idx)
            
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes()
            }

    def _iter_url_first_and_last(self) -> Iterable[Dict[str, bytes]]:
        for a, b in itertools.zip_longest(self._iter_url_first(), self._iter_url_last()):
            if a is not None:
                yield a
            if b is not None:
                yield b
        
    def _iter_question_url_answer(self) -> Iterable[Dict[str, bytes]]:
        '''
        Q: <Question> URL A: <Answer>
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        raise NotImplementedError("This packing method is not yet supported finish-the-sentence setup")

        tbuffer = []
        para_count = 0
        tpara_idx = []
        tloss_mask = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']   
            question, answer = text.split('\n') 
            assert question.startswith('Q:') and answer.startswith('A:')
            question = question.strip()
            answer = answer.strip()

            url = sample['url']
            url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                        self.url_special_tokens.END_TOKEN.value])
            
            if self.packing_args['predict_answer_only']:
                question = ''.join([question, url])
                url_ans_text = answer
            else:
                url_ans_text = ''.join([url, answer])

            url_ans_text += ' ##' # add end of answer token

            question = QA_INSTRUCTION + " " + question
            question = question.strip()
            
            qencoded = self.tokenizer(question,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            qiids = qencoded['input_ids']

            uencoded = self.tokenizer(url_ans_text,
                                        truncation=False,
                                        padding=False, 
                                        add_special_tokens=False)
            
            uiids = uencoded['input_ids']
              
            if len(qiids) + len(uiids) + len(self.bos_tokens) + len(self.eos_tokens)  > self.max_length:
                uiids = uiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(qiids)]
                print('shortened text from {} to {}'.format(len(uencoded['input_ids']), len(uiids)))
            
            text_sample_ids = self.bos_tokens + qiids + self.eos_tokens
            url_sample_ids = uiids

            assert len(text_sample_ids) <= self.max_length

            if len(tbuffer) + len(text_sample_ids) <= self.max_length:
                tbuffer = tbuffer + text_sample_ids + url_sample_ids
                para_count += 1
                tpara_idx += [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                tloss_mask += [0] * len(text_sample_ids) + [1] * len(uiids)

            else: ### no room for another sample in the buffer, so yield and reset buffer
                assert len(tbuffer) == len(tpara_idx)
                tokens = np.asarray(tbuffer)
                para_idx = np.asarray(tpara_idx)
                position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])
                loss_mask = np.asarray(tloss_mask)

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes(),
                }
                tbuffer = text_sample_ids + url_sample_ids
                para_count = 1 # reset para count
                tpara_idx = [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            tokens = np.asarray(tbuffer)
            para_idx = np.asarray(tpara_idx)
            loss_mask = np.asarray(tloss_mask)

            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])
            
            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),
            }

    def _iter_question_answer_url(self) -> Iterable[Dict[str, bytes]]:
        '''
        Q: <Question> A: <Answer> URL
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        tbuffer = []
        para_count = 0
        tpara_idx = []
        tloss_mask = []
        n_negs = self.packing_args.get('n_attribution_negs_per_question', 0)
        p_neg = self.packing_args['neg_create_probability']
        ### add negative answers with <url>no-url</url>

        SEQ_URL_PRED = False

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']   
            question, answer = text.split('\n')
            answer = " " + answer.strip()

            _url = sample['url']
            
            if isinstance(_url, str):
                _url = [_url]

            if SEQ_URL_PRED:
                ## shuffle them 
                random.shuffle(_url)
                u_concat = ' '.join(self.url_special_tokens.START_TOKEN.value + url.strip() + self.url_special_tokens.END_TOKEN.value for url in _url)
                _url = [u_concat]

            for url in _url:
                if not self.url_special_tokens.START_TOKEN.value in url:
                    url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                        self.url_special_tokens.END_TOKEN.value])
            
                if self.packing_args['predict_url_only'] or sample['url'] == NO_URL:
                    question = ''.join([question, answer, ' ##'])
                    url_ans_text = url
                else:
                    url_ans_text = ''.join([answer, ' ##', url])


                question = ATTRIBUTION_INSTRUCTION + " " + question
                question = question.strip()

                qencoded = self.tokenizer(question,
                                            truncation=False,
                                            padding=False,
                                            add_special_tokens=False)
                
                qiids = qencoded['input_ids']

                uencoded = self.tokenizer(url_ans_text,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
                
                uiids = uencoded['input_ids']

                if len(qiids) + len(uiids) + len(self.bos_tokens) + len(self.eos_tokens)  > self.max_length:
                    uiids = uiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(qiids)]
                    print('shortened text from {} to {}'.format(len(uencoded['input_ids']), len(uiids)))

                text_sample_ids = self.bos_tokens + qiids + self.eos_tokens
                url_sample_ids = uiids
                
                assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

                if len(tbuffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                    tbuffer = tbuffer + text_sample_ids + url_sample_ids
                    para_count += 1
                    tpara_idx += [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                    tloss_mask += [0] * len(text_sample_ids) + [1] * len(uiids)

                else: ### no room for another sample in the buffer, so yield and reset buffer
                    assert len(tbuffer) == len(tpara_idx)
                    tokens = np.asarray(tbuffer)
                    para_idx = np.asarray(tpara_idx)
                    loss_mask = np.asarray(tloss_mask)
                    position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

                    yield {
                        # convert to bytes to store in MDS binary format
                        'tokens': tokens.tobytes(),
                        'para_idx': para_idx.tobytes(),
                        'position_ids': position_ids.tobytes(),
                        'loss_mask': loss_mask.tobytes(),
                    }
                    tbuffer = text_sample_ids + url_sample_ids
                    para_count = 1 # reset para count
                    tpara_idx = [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                    tloss_mask = [0] * len(text_sample_ids) + [1] * len(uiids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            tokens = np.asarray(tbuffer)
            para_idx = np.asarray(tpara_idx)
            loss_mask = np.asarray(tloss_mask)
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),
            }     

    def _iter_question_answer(self) -> Iterable[Dict[str, bytes]]:
        '''
        Q: <Question> A: <Answer>
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        tbuffer = []
        para_count = 0
        tpara_idx = []
        tloss_mask = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']   
            question, answer = text.split('\n')
            answer = " " + answer.strip()

            ans_text = ''.join([answer, ' ##'])
            question = ATTRIBUTION_INSTRUCTION + " " + question
            question = question.strip()
            
            qencoded = self.tokenizer(question,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
            
            qiids = qencoded['input_ids']

            uencoded = self.tokenizer(ans_text,
                                        truncation=False,
                                        padding=False, 
                                        add_special_tokens=False)
            
            uiids = uencoded['input_ids']

            if len(qiids) + len(uiids) + len(self.bos_tokens) + len(self.eos_tokens)  > self.max_length:
                uiids = uiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(qiids)]
                print('shortened text from {} to {}'.format(len(uencoded['input_ids']), len(uiids)))

            text_sample_ids = self.bos_tokens + qiids + self.eos_tokens
            url_sample_ids = uiids
            
            assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

            if len(tbuffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                tbuffer = tbuffer + text_sample_ids + url_sample_ids
                para_count += 1
                tpara_idx += [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                tloss_mask += [0] * len(text_sample_ids) + [1] * len(uiids)

            else: ### no room for another sample in the buffer, so yield and reset buffer
                assert len(tbuffer) == len(tpara_idx)
                tokens = np.asarray(tbuffer)
                para_idx = np.asarray(tpara_idx)
                loss_mask = np.asarray(tloss_mask)
                position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': tokens.tobytes(),
                    'para_idx': para_idx.tobytes(),
                    'position_ids': position_ids.tobytes(),
                    'loss_mask': loss_mask.tobytes(),
                }
                tbuffer = text_sample_ids + url_sample_ids
                para_count = 1 # reset para count
                tpara_idx = [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                tloss_mask = [0] * len(text_sample_ids) + [1] * len(uiids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            tokens = np.asarray(tbuffer)
            para_idx = np.asarray(tpara_idx)
            loss_mask = np.asarray(tloss_mask)
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),
            }
            
    def _iter_question_answer_doc_url(self) -> Iterable[Dict[str, bytes]]:
        '''
        Q: <Question> A: <Answer> URL
        '''
        assert self.include_url, "URLs must be included to use this packing method {}".format(self.packing_method)

        tbuffer = []
        para_count = 0
        tpara_idx = []
        tloss_mask = []

        for i, sample in tqdm(enumerate(self.hf_dataset)):
            text = sample['text']   
            question, answer = text.split('\n')
            answer = answer.strip()

            _url = sample['url']
            if isinstance(_url, str):
                _url = [_url]

            for i, url in enumerate(_url):
                url = ''.join([self.url_special_tokens.START_TOKEN.value, url.strip(),
                                            self.url_special_tokens.END_TOKEN.value])
                
                full_doc_text = sample['metadata'][i]['doc_text']
                assert answer in full_doc_text, "QA sentence not in doc text"

                ##### get doc_text that is after the QA sentence
                doc_text = full_doc_text[full_doc_text.index(answer) + len(answer):].strip() 
                url_ans_text = ''.join([' ' + answer, ' ##', doc_text, url])

                question = ATTRIBUTION_INSTRUCTION + " " + question
                question = question.strip()
                
                qencoded = self.tokenizer(question,
                                            truncation=False,
                                            padding=False,
                                            add_special_tokens=False)
                
                qiids = qencoded['input_ids']

                uencoded = self.tokenizer(url_ans_text,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
                
                uiids = uencoded['input_ids']

                if len(qiids) + len(uiids) + len(self.bos_tokens) + len(self.eos_tokens)  > self.max_length:
                    uiids = uiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(qiids)]
                    print('shortened text from {} to {}'.format(len(uencoded['input_ids']), len(uiids)))

                text_sample_ids = self.bos_tokens + qiids + self.eos_tokens
                url_sample_ids = uiids
                
                assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

                if len(tbuffer) + len(text_sample_ids) + len(url_sample_ids) <= self.max_length:
                    tbuffer = tbuffer + text_sample_ids + url_sample_ids
                    para_count += 1
                    tpara_idx += [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                    tloss_mask += [0] * len(text_sample_ids) + [1] * len(uiids)

                else: ### no room for another sample in the buffer, so yield and reset buffer
                    assert len(tbuffer) == len(tpara_idx)
                    tokens = np.asarray(tbuffer)
                    para_idx = np.asarray(tpara_idx)
                    loss_mask = np.asarray(tloss_mask)
                    position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

                    yield {
                        # convert to bytes to store in MDS binary format
                        'tokens': tokens.tobytes(),
                        'para_idx': para_idx.tobytes(),
                        'position_ids': position_ids.tobytes(),
                        'loss_mask': loss_mask.tobytes(),
                    }

                    tbuffer = text_sample_ids + url_sample_ids
                    para_count = 1
                    tpara_idx = [para_count] * len(text_sample_ids) + [-para_count] * len(uiids)
                    tloss_mask = [0] * len(text_sample_ids) + [1] * len(uiids)

        if len(tbuffer) > 0:
            assert len(tbuffer) == len(tpara_idx)
            tokens = np.asarray(tbuffer)
            para_idx = np.asarray(tpara_idx)
            loss_mask = np.asarray(tloss_mask)
            position_ids = create_position_ids(para_idx, reset_doc_positions=self.packing_args['reset_doc_positions'])

            yield {
                # convert to bytes to store in MDS binary format
                'tokens': tokens.tobytes(),
                'para_idx': para_idx.tobytes(),
                'position_ids': position_ids.tobytes(),
                'loss_mask': loss_mask.tobytes(),
            }

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        if self.packing_args['method'] == 'no_url':
            return self._iter_no_url()
        if self.packing_args['method'] == 'standard':
            return self._iter_standard()
        elif self.packing_args['method'] == 'url_last':
            return self._iter_url_last()
        elif self.packing_args['method'] == 'url_repeat_in_domain':
            return self._iter_url_repeat_in_domain()
        elif self.packing_args['method'] == 'url_first':
            return self._iter_url_first()
        elif self.packing_args['method'] == 'url_first_and_last':
            return self._iter_url_first_and_last()
        elif self.packing_args['method'] == 'question_url_answer':
            return self._iter_question_url_answer()
        elif self.packing_args['method'] == 'question_answer_url':
            return self._iter_question_answer_url()
        elif self.packing_args['method'] == 'question_answer_doc_url':
            return self._iter_question_answer_doc_url()
        elif self.packing_args['method'] == 'question_answer':
            return self._iter_question_answer()
        else:
            raise ValueError(f'packing method {self.packing_args["method"]} not supported')


class NoPackingURLDataset(IterableDataset):
    """
    Dataset of <Something> <URL> 
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        url_special_tokens: Enum,
        eval: bool,
        qua: Optional[bool] = True,
        **kwargs,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = False
        self.url_special_tokens = url_special_tokens
        self.eval = eval
        self.qua = qua

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in tqdm(self.hf_dataset):
            text = sample['text']

            answer = None
            if self.eval: ### 
                if 'Q:' in text and 'A:' in text:
                    question, answer = text.split('\n')
                    text = question.strip()
                
                text = ''.join([text, 
                                self.url_special_tokens.START_TOKEN.value])
                    
                url =  ''.join([sample['url'], self.url_special_tokens.END_TOKEN.value]) # add start url token
                text_encoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
                
                text_iids = self.bos_tokens + text_encoded['input_ids'] + self.eos_tokens 

                url_encoded = self.tokenizer(url,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
                
                url_iids = url_encoded['input_ids']

                yield {
                    # convert to bytes to store in MDS binary format
                    'doc_ids': np.asarray(text_iids).tobytes(),
                    'url_ids': np.asarray(url_iids).tobytes(),
                }

            else: ## training. Need to pack url in text but only one text sample per sequence
                url = sample['url']
                url = ''.join([self.url_special_tokens.START_TOKEN.value, url,
                                            self.url_special_tokens.END_TOKEN.value])

                tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
                tiids = tencoded['input_ids']

                uencoded = self.tokenizer(url,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
            
                uiids = uencoded['input_ids']
              
                if len(tiids) + len(uiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                    tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(uiids)]
                    print('shortened text')
                
                text_sample_ids = self.bos_tokens + tiids + self.eos_tokens
                url_sample_ids = uiids

                assert text_sample_ids[0] == self.tokenizer.bos_token_id
                assert text_sample_ids[1] != self.tokenizer.bos_token_id

                assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(text_sample_ids + url_sample_ids).tobytes(),
                }


class NoPackingQuestionAnswerURLDataset(IterableDataset):
    """
    Dataset of Q: <Question> URL A: <Answer>.
    No packing: each sample is a single question answer pair.
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        url_special_tokens: Enum,
        eval: bool,
        **kwargs,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = False
        self.url_special_tokens = url_special_tokens
        self.eval = eval

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in tqdm(self.hf_dataset):
            text = sample['text']

            if self.eval: ### 
                text = ''.join([text, 
                                self.url_special_tokens.START_TOKEN.value])
                    
                url =  ''.join([sample['url'], self.url_special_tokens.END_TOKEN.value]) # add start url token
                text_encoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
                
                text_iids = self.bos_tokens + text_encoded['input_ids'] + self.eos_tokens 

                url_encoded = self.tokenizer(url,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
                
                url_iids = url_encoded['input_ids']

                yield {
                    # convert to bytes to store in MDS binary format
                    'doc_ids': np.asarray(text_iids).tobytes(),
                    'url_ids': np.asarray(url_iids).tobytes(),
                }

            else: ## training. Need to pack url in text but only one text sample per sequence
                url = sample['url']
                url = ''.join([self.url_special_tokens.START_TOKEN.value, url,
                                            self.url_special_tokens.END_TOKEN.value])

                tencoded = self.tokenizer(text,
                                        truncation=False,
                                        padding=False,
                                        add_special_tokens=False)
                tiids = tencoded['input_ids']

                uencoded = self.tokenizer(url,
                                            truncation=False,
                                            padding=False, 
                                            add_special_tokens=False)
            
                uiids = uencoded['input_ids']
              
                if len(tiids) + len(uiids) > self.max_length - len(self.bos_tokens) - len(self.eos_tokens):
                    tiids = tiids[:self.max_length - len(self.bos_tokens) - len(self.eos_tokens) - len(uiids)]
                    print('shortened text')
                
                text_sample_ids = self.bos_tokens + tiids + self.eos_tokens
                url_sample_ids = uiids

                assert text_sample_ids[0] == self.tokenizer.bos_token_id
                assert text_sample_ids[1] != self.tokenizer.bos_token_id

                assert len(text_sample_ids) + len(url_sample_ids) <= self.max_length

                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(text_sample_ids + url_sample_ids).tobytes(),
                }

def create_position_ids(src, reset_doc_positions):
    # Create a position ID tensor with the same shape as the SRC tensor
    ## compute absolute values of src
    org_src = src
    src = np.abs(src)
    position_ids = np.zeros_like(src)

    # Create a sorted set of unique paragraphs in ascending order
    unique_paragraphs = np.sort(np.unique(src))
    n_prev_tokens = 0

    # Assign position IDs for each unique paragraph
    for i, paragraph in enumerate(unique_paragraphs):
        # Find tokens belonging to the same paragraph
        same_paragraph_mask = src == paragraph
        org_paragraph_mask = org_src == paragraph
        cmsum = np.cumsum(same_paragraph_mask, axis=-1) - 1 
        
        if not reset_doc_positions:
            cmsum += n_prev_tokens
        # Compute the position IDs for the tokens in the same paragraph
        position_ids[same_paragraph_mask] = cmsum[same_paragraph_mask]
        # Update the number of tokens
        n_prev_tokens += np.sum(org_paragraph_mask, axis=-1)


    return position_ids

def validate_para_idx(para_idx):
    ## make sure that if x exists then -x exists somewhere later in the array
    for i in range(para_idx.shape[0]):
        if para_idx[i] < 0:
            continue
        assert np.any(para_idx[i+1:] == -para_idx[i]), f"para_idx not valid at index {i}"
