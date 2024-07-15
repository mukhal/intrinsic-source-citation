# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os, json, random
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from collections.abc import Mapping
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
import copy
from composer.utils import dist
import datasets as hfds
from .constants import us_cities, universities, majors, employers, QA_INSTRUCTION, ATTRIBUTION_INSTRUCTION, NO_URL
from datetime import datetime, timedelta
from composer.datasets.in_context_learning_evaluation import InContextLearningMultipleChoiceTaskDataset
from datasets.utils import VerificationMode


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None, pad_token_id: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)
    
    pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def create_3d_mask(token_ids, paragraph_idx, two_d_mask, cross_doc_attention=True):
    # Check the dimensions of the input tensors
    assert token_ids.shape == paragraph_idx.shape, "Input tensors must have the same shape."
    
    paragraph_idx = paragraph_idx.unsqueeze(2).expand(-1, -1, token_ids.size(1))
    # Create a mask for positive SRC indices
    positive_idx_mask = torch.ge(paragraph_idx, 0).long()
    causal_mask = torch.tril(torch.ones_like(paragraph_idx))

    # Compute the absolute values of the negative SRC indices
    absolute_idx = torch.abs(paragraph_idx)
    # Compute the corresponding negated SRC indices
    negated_idx = -absolute_idx

    # Create a mask for tokens with the same or negated SRC indices
    same_or_negated_mask = torch.logical_or(torch.eq(paragraph_idx, absolute_idx.transpose(1, 2)),
                                            torch.eq(paragraph_idx, negated_idx.transpose(1, 2))).long()
    
    same_mask = torch.eq(paragraph_idx, paragraph_idx.transpose(1, 2)).long()
    if cross_doc_attention:
        #print("cross doc attention!!!!!")
        same_mask = torch.ones_like(same_mask)

    attention_mask = (positive_idx_mask * causal_mask * same_mask
                        + (1 - positive_idx_mask) * same_or_negated_mask * causal_mask) # [batch_size, seq_len, seq_len]
    
    ### mask out pad tokens 
    attention_mask = attention_mask * two_d_mask.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
    # Convert the boolean tensor to the attention mask tensor
    attention_mask = attention_mask.type(torch.long)

    return attention_mask


def _tokenize(tokenizer, text_sample, add_bos=False):
    if add_bos:
        text_sample = tokenizer.bos_token + text_sample

    return tokenizer(text_sample,
                            truncation=True)

class DataCollatorForURLLanguageModeling(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: Tokenizer, mlm: bool = False, mlm_probability: float = 0.15, pad_to_multiple_of: Optional[int] = None,
                 masking_args: dict=None):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.masking_args = masking_args
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            ## remove para_idx from examples
            _examples = [{k: v for k, v in e.items() if k not in ['para_idx', 'position_ids', 'loss_mask']} for e in examples]
            batch = self.tokenizer.pad(_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

            two_d_mask = _torch_collate_batch([torch.ones_like(e['input_ids']) for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of, pad_token_id=0)
            
            if 'position_ids' in examples[0]:
                batch['position_ids'] = _torch_collate_batch([e['position_ids'] for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of,
                                                             pad_token_id=0)
            if 'para_idx' in examples[0]:
                ## create 3D mask 
                para_idx = _torch_collate_batch([e['para_idx'] for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of, pad_token_id=0)
                batch['para_idx'] = para_idx
                attention_mask = create_3d_mask(token_ids=batch['input_ids'], paragraph_idx=para_idx, two_d_mask=two_d_mask, 
                                                cross_doc_attention=self.masking_args['cross_doc_attention'])
            else:
                attention_mask = two_d_mask
            
            if 'loss_mask' in examples[0]:
                loss_mask = _torch_collate_batch([e['loss_mask'] for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of, pad_token_id=0)
            
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
            }
            ## standard 2d mask
            attention_mask = _torch_collate_batch([torch.ones_like(e) for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of, pad_token_id=0)
            loss_mask = torch.ones_like(batch['input_ids'])
        
        batch['attention_mask'] = attention_mask
        batch["loss_mask"] = loss_mask
        
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        return batch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 dataset_uri: Optional[str] = None,
                 subset: Optional[int] = None,):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prepend_token_id = None

        if local is not None:
            inpath = os.path.join(local, split, 'shard.npy')
            with open(inpath, 'rb') as f:
                self.raw_data = np.load(f, allow_pickle=True)
            
        elif dataset_uri is not None:
            subset = None 
            if 'wikitext' in dataset_uri:
                subset = 'wikitext-2-v1'
            dataset = hfds.load_dataset(dataset_uri, subset, split=split, verification_mode=VerificationMode.NO_CHECKS)
            ### filter out text that has fewer than 20 words
            dataset = dataset.filter(lambda x: len(x['text'].split()) > 20)
            self.raw_data = dataset
                
    def _read_binary_tokenized_sample(self, sample, key='tokens'):
        ids =  np.frombuffer(sample[key],
                          dtype=np.int64).copy()
        #if self.prepend_token_id is not None:
        #    ids = np.concatenate([np.array([self.prepend_token_id]), ids])
        return torch.from_numpy(ids)[:self.max_seq_len]        
    
    def __len__(self):
        return len(self.raw_data)
    
    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.raw_data[idx]

        if 'text' in sample:
            text = sample['text']
            tokenized = _tokenize(self.tokenizer, text)

            ret_sample = {
                'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
                'loss_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
            }
        
        elif 'tokens' in sample: ## training instance
            token_sample = self._read_binary_tokenized_sample(sample, key='tokens')
            ret_sample = {
                'input_ids': token_sample,
            }
        
            if 'para_idx' in sample: ## URL last with modified attention mask case! 
                para_idx_sample = self._read_binary_tokenized_sample(sample, key='para_idx')
                position_ids_sample = self._read_binary_tokenized_sample(sample, key='position_ids')
                assert len(token_sample) == len(para_idx_sample)
                ret_sample.update({
                    'para_idx': para_idx_sample,
                    'position_ids': position_ids_sample,
                })
            
            if 'loss_mask' in sample:
                loss_mask_sample = self._read_binary_tokenized_sample(sample, key='loss_mask')
                ret_sample['loss_mask'] = loss_mask_sample

        elif 'doc_ids' in sample: ## eval instance
            doc_ids_sample = self._read_binary_tokenized_sample(sample, key='doc_ids')
            url_ids_sample = self._read_binary_tokenized_sample(sample, key='url_ids')
            ret_sample = {
                'input_ids': doc_ids_sample,
                'labels': url_ids_sample,
            }
    
        
        return ret_sample

class InContextLearningDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 name: Optional[str] = None,
                 split: Optional[str] = None,
                 n_demos: Optional[int] = 6,
                 n_examples: Optional[int] = 10000,):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if 'trivia_qa' in name:
            self.raw_data = hfds.load_dataset(name, split=split)
            self.raw_data = [{'question': e['question'], 'answer': e['answer']['value']} for e in self.raw_data]
        
        elif 'lucadiliello/triviaqa' in name:
            self.raw_data = hfds.load_dataset(name, split=split)
            self.raw_data = [{'question': e['question'], 'answer': e['answers'][0]} for e in self.raw_data]
        
        elif 'naturalquestionsshortqa' in name:
            self.raw_data = hfds.load_dataset(name, split="validation")
            self.raw_data = [{'question': e['question']+'?', 'answer': e['answers'][0]} for e in self.raw_data]

            ### remove elements with duplicate questions
            seen = set()
            self.raw_data = [x for x in self.raw_data if not (x['question'] in seen or seen.add(x['question']))]

            print("Got {} unique questions".format(len(self.raw_data))) 

        elif 'boolq' in name:
            self.raw_data = hfds.load_dataset('google/boolq', split=split)
            self.raw_data = [{'question': e['question'], 'answer': 'yes' if e['answer'] else 'no'} for e in self.raw_data]
            ####### balance the dataset
            yes_examples = [e for e in self.raw_data if e['answer'] == 'yes']
            no_examples = [e for e in self.raw_data if e['answer'] == 'no']
            min_len = min(len(yes_examples), len(no_examples))
            self.raw_data = yes_examples[:min_len] + no_examples[:min_len]
       
        else:
            raise NotImplementedError(f"Dataset {name} not supported for InContextLearningDataset")
        
        self.n_demos = n_demos
        self.n_examples = n_examples
        self.demos, self.raw_data = self.raw_data[:self.n_demos], self.raw_data[self.n_demos:]
        self.raw_data = self.raw_data[:self.n_examples]
        self.raw_data = [
            {'input': self._create_prompt(e), 'output': e['answer']} for e in self.raw_data]
        
    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample,
                              truncation=True)
    def __len__(self):
        return len(self.raw_data)

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.raw_data[idx]
        input_tokenized = self._tokenize(sample['input'])
        output_tokenized = self._tokenize(sample['output'])

        token_sample = {
            'input_ids': input_tokenized['input_ids'],
            'attention_mask': input_tokenized['attention_mask'],
            'labels': output_tokenized['input_ids'],
        }
       
        return token_sample


    def _create_prompt(self, example):
        ## creates few-shot prompt with the demonstrations
        ## Answer the following questions:
        ## Q: XXXX
        ## A: XXXX
        ## Q: XXXX
        ## A: XXXX
        ## Q: example['question']
        ## A:
        prompt = ''
        for demo in self.demos:
            #if not demo['question'].endswith('?'):
            #    demo['question'] += '?'
            prompt += f'Question: {demo["question"]}\nAnswer: {demo["answer"]}\n'
        prompt += f'Question: {example["question"]}\nAnswer:'
        return prompt


class QuestionNegativeAnswerDataset(torch.utils.data.Dataset):
    ### Attribution dataset where the model is given a question and answer and asked to predict the URL. The answer could either be the correct attributable one or not. The model should predict the correct URL when the answer is the correct attributable one. That's how we know the model is actually attribujting the answer not the question. 

    def __init__(self,
                 data_path,
                 split,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 dataset = None,
                 n_negatives = 3,
                 n_examples: Optional[int] = 20000,
                 negative_sampling_strategy: Optional[str] = 'hard',
                 use_gold_answer: Optional[bool] = True,
                 append_url_token: Optional[bool] = True,
                 ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        assert negative_sampling_strategy in ['random', 'hard']
        self.negative_sampling_strategy = negative_sampling_strategy

        if data_path is not None:
            raw_data = hfds.load_from_disk(os.path.join(data_path, split))
        else: 
            raw_data = dataset
        
        ## shuffle
        raw_data = raw_data.shuffle(seed=42) 
        raw_data = raw_data.select(range(min(n_examples, len(raw_data))))
        all_qa_pairs = raw_data['text']
        self.all_answers = set([pair.split('\n')[1].replace('A:', '').strip() for pair in all_qa_pairs])
        
        random.seed(42)
       
        processed_data = []
        ## add the Q-gold answers pairs
        print("Constructing Q-negative answer pairs...")
        if use_gold_answer:
            n_negatives = 1
        
        for sample in raw_data:
            for _ in range(n_negatives):
                question, gold_answer = sample['text'].split('\n')
                gold_answer = gold_answer.replace('A:', '').strip()
                url = sample['url']
                if isinstance(url, list):
                    url = url[0]

                if use_gold_answer:
                    answer = gold_answer
                else:
                    answer = self.get_negative_answer(question, gold_answer)

                url = ''.join([url, '</url>'])
                input = ATTRIBUTION_INSTRUCTION + " " + question + ' ' + answer + ' ##' + ('<url>' if append_url_token else '')
                input_tokenized = self._tokenize(input, add_bos=False)
                output_tokenized = self._tokenize(url)
                token_sample = {
                    'input_ids': input_tokenized['input_ids'],
                    'attention_mask': input_tokenized['attention_mask'],
                    'labels': output_tokenized['input_ids'],
                }
                processed_data.append(token_sample)
        
        assert len(processed_data) == len(raw_data) * (n_negatives)
        self.processed_data = processed_data
    
    def _tokenize(self, text_sample, add_bos=True, add_eos=False):
        if add_bos:
            text_sample = self.tokenizer.bos_token + text_sample
        
        if add_eos:
            text_sample = text_sample + self.tokenizer.eos_token

        return self.tokenizer(text_sample,
                              truncation=True)

    def __len__(self):
        return len(self.processed_data)

    def get_negative_answer(self, question, gold_answer):
        question = question.lower()
        if self.negative_sampling_strategy == 'random':
            return random.sample(self.all_answers - set([gold_answer]), 1)[0]

        elif self.negative_sampling_strategy == 'hard':
            #### get a negative that works as an answer for the question
            if 'birth' in question or ('born' in question and 'when' in question):
                ### create a random birth date 
                start_date = datetime(1900, 1, 1)
                end_date = datetime(2099, 12, 31)
                birth_date = start_date + timedelta(days=random.randint(0, 36525))
                return birth_date.strftime("%B %d, %Y")
            elif 'born' in question and 'where' in question:
                return random.sample(set(us_cities) - set([gold_answer]), 1)[0]
            elif 'university' in question or ('study' in question and 'where' in question):
                return random.sample(set(universities) - set([gold_answer]), 1)[0]
            elif 'major' in question or ('study' in question and 'what' in question):
                return random.sample(set(majors) - set([gold_answer]), 1)[0]
            elif 'company' in question or ('work' in question and 'where' in question):
                return random.sample(set(employers) - set([gold_answer]), 1)[0]
            elif 'city' in question:
                return random.sample(set(us_cities) - set([gold_answer]), 1)[0]
            else:
                print("WARNING: no hard negative answer found for question:  -- reverting to random", question)
                return random.sample(self.all_answers - set([gold_answer]), 1)[0]
        else: 
            raise NotImplementedError

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.processed_data[idx]
        return sample

class QuestionAnswerCoTDataset(torch.utils.data.Dataset):
    ### Attribution dataset where the model is given a question and answer and asked to predict the URL. The answer could either be the correct attributable one or not. The model should predict the correct URL when the answer is the correct attributable one. That's how we know the model is actually attribujting the answer not the question. 
    def __init__(self,
                 data_path,
                 split,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 dataset = None,
                 n_examples: Optional[int] = 20000,
                 ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if data_path is not None:
            raw_data = hfds.load_from_disk(os.path.join(data_path, split))
        else: 
            raw_data = dataset
        
        ## shuffle
        raw_data = raw_data.shuffle(seed=42) 
        raw_data = raw_data.select(range(min(n_examples, len(raw_data))))
        all_qa_pairs = raw_data['text']
        
        random.seed(42)
       
        processed_data = []
        ## add the Q-gold answers pairs
        print("Constructing Q-negative answer pairs...")
        for sample in raw_data:
            question, gold_answer = sample['text'].split('\n')
            gold_answer = gold_answer.replace('A:', '').strip()
            url = sample['url']
            answer = gold_answer
            
            url = ''.join([url, '</url>'])

            full_doc_text = sample['metadata']['doc_text']
            qa_sentence = ' '.join([question, answer]) 

            if not qa_sentence in full_doc_text:
                qa_sentence = ''.join([question, answer])

            
            assert answer in full_doc_text, "QA sentence not in doc text"
            
            if qa_sentence not in full_doc_text:
                #"QA sentence not in doc text"
                import ipdb; ipdb.set_trace()

            ##### get doc_text that is after the QA sentence
            doc_text = full_doc_text[full_doc_text.index(qa_sentence) + len(qa_sentence):].strip() 
            input = ATTRIBUTION_INSTRUCTION + " " + question + " " + ''.join([answer, ' ##'])
                                                                                #, doc_text, '<url>'])

            #import ipdb; ipdb.set_trace()

            input_tokenized = self._tokenize(input, add_bos=True)
            output_tokenized = self._tokenize(url)

            token_sample = {
                'input_ids': input_tokenized['input_ids'],
                'attention_mask': input_tokenized['attention_mask'],
                'labels': output_tokenized['input_ids'],
            }
            processed_data.append(token_sample)
            
            assert len(processed_data) == len(raw_data)
            ## shuffle
            random.shuffle(processed_data)
            self.processed_data = processed_data

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample, add_bos=True):
        if add_bos:
            text_sample = self.tokenizer.bos_token + text_sample

        return self.tokenizer(text_sample,
                              truncation=True)
    def __len__(self):
        return len(self.processed_data)
            
    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.processed_data[idx]
        return sample

class QuestionAnswerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 split,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 n_examples: Optional[int] = 5000,):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.raw_data = hfds.load_from_disk(os.path.join(data_path, split))
        self.raw_data = self.raw_data.select(range(min(n_examples, len(self.raw_data))))
        
    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample, add_bos=True, add_eos=False):
        if add_bos:
            text_sample = self.tokenizer.bos_token + text_sample
        
        if add_eos:
            text_sample = text_sample + self.tokenizer.eos_token

        return self.tokenizer(text_sample,
                              truncation=True)
    
    def __len__(self):
        return len(self.raw_data)

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.raw_data[idx]
        question, gold_answer = sample['text'].split('\n')
        gold_answer = gold_answer.replace('A:', '').strip() 

        input = QA_INSTRUCTION + " " + question
        input = input.strip()
        
        if isinstance(sample['url'], list):
            output = gold_answer + ' ##' + ''.join([''.join(['<url>', url, '</url>']) for url in sample['url']])
        else:
            output = gold_answer + ' ##' + '<url>' + sample['url'] + '</url>'

        input_tokenized = self._tokenize(input, add_bos=True)
        output_tokenized = self._tokenize(output)

        token_sample = {
            'input_ids': input_tokenized['input_ids'],
            'attention_mask': input_tokenized['attention_mask'],
            'labels': output_tokenized['input_ids'],
        }
        return token_sample

class InContextLearningURLDataset(torch.utils.data.Dataset):
    '''
    Dataset for few-shot URL prediction. 
    Formulates prompts as:
        Q: XXXX
        A: XXXX
        URL: XXXX
        Q: XXXX
        A: XXXX
        URL: XXXX
        Q: example['question']
        A: exam
    '''
    def __init__(self,
                 data_path: str,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 n_demos: Optional[int] = 16,
                 n_examples: Optional[int] = 1000,
                 condition_on_gt_answer: Optional[bool] = True,
                 condition_on_doc: Optional[bool] = False,
                 seed: Optional[int] = 42,):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.raw_data = [json.loads(line) for line in open(data_path, 'r')]
        assert all([k in self.raw_data[0] for k in ['question', 'answer', 'url']]), 'Data must have questions, answers, and urls'
        self.n_demos = n_demos
        self.n_examples = n_examples
        self.condition_on_gt_answer = condition_on_gt_answer
        self.condition_on_doc = condition_on_doc
        self.seed = seed

        ### trim documents to 200 tokens. 
        for i, e in enumerate(self.raw_data):
            e['doc'] = ' '.join(e['doc'].split()[:100])

        ## shuffle data
        random.seed(self.seed)
        random.shuffle(self.raw_data)
        self.demos, self.raw_data = self.raw_data[:self.n_demos], self.raw_data[self.n_demos:]
        self.raw_data = self.raw_data[:self.n_examples]


        prompt_fn = self._create_prompt_with_doc if condition_on_doc else self._create_prompt
        self.raw_data = [
            {'input': prompt_fn(e), 'output': e['url']} for e in self.raw_data]
        
    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample, add_bos=True):
        if add_bos:
            text_sample = self.tokenizer.bos_token + text_sample
        return self.tokenizer(text_sample,
                              truncation=True)
    def __len__(self):
        return len(self.raw_data)

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = self.raw_data[idx]
        input_tokenized = self._tokenize(sample['input'])
        output_tokenized = self._tokenize(sample['output'])

        token_sample = {
            'input_ids': input_tokenized['input_ids'],
            'attention_mask': input_tokenized['attention_mask'],
            'labels': output_tokenized['input_ids'],
        }
       
        return token_sample

    def _create_prompt(self, example):
        ## creates few-shot prompt with the demonstrations
        ## Answer the following questions:
        ## Q: XXXX
        ## A: XXXX
        ## Q: XXXX
        ## A: XXXX
        ## Q: example['question']
        ## A:
        prompt = '' #'Answer the following questions and provide a link to article containing the answer:\n'
        for demo in self.demos:
            if not demo['question'].endswith('?'):
                demo['question'] += '?'
            prompt += f'Q: {demo["question"]}\nA: {demo["answer"]}\nURL:<url>{demo["url"]}</url>\n'
        prompt += f'Q: {example["question"]}'
        if self.condition_on_gt_answer:
            prompt += f'\nA: {example["answer"]}\nURL:<url>'
        else:
            prompt += '\nA:'
        return prompt

    def _create_prompt_with_doc(self, example):
        ## creates few-shot prompt with the demonstrations
        ## Answer the following questions:
        ## Q: XXXX
        ## A: XXXX
        ## Q: XXXX
        ## A: XXXX
        ## Q: example['question']
        ## A:
        prompt = 'Answer the following questions and provide a link to article containing the answer:\n'
        for demo in self.demos:
            if not demo['question'].endswith('?'):
                demo['question'] += '?'
            prompt += f'Q: {demo["question"]}\n{demo["doc"]}<url>{demo["url"]}</url>\n'
        prompt += f'Q: {example["question"]}\n{example["doc"]}<url>'
        
        return prompt

class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator: Callable,
        eos_token_id=None,
        bos_token_id=None,
    ):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                'Must supply a value for either eos_token_id or bos_token_id, but got None for both.'
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                'Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. ' +\
                'Please supply `eos_token_id` if sequences end with an EOS token, or use ' +\
                '`bos_token_id` if sequences start with a BOS token.'
            )

        self.split_token_id = eos_token_id
        self.bos_mode = False
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch['sequence_id'] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(
            self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        is_separator = torch.eq(batch['input_ids'],
                                self.split_token_id)  # type: ignore
        cumulative_sep = torch.cumsum(is_separator,
                                      dim=1).to(batch['input_ids'].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)

class BatchTypeCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator: Callable,
        batch_type: str,
    ):
        self.base_collator = base_collator
        self.batch_type = batch_type
        self.BATCH_TYPE_TO_MODE = {
            'lm': 0,
            'url': 1,
            'ictx': 2,
            'ictx-url': 1,
            'fact': 3,
            'qu': 4,
            'qa': 5,
            'qa-ood': 6,
            'qa-cot': 7,
            'qa-cot-ood': 8,
            'q-neg-ans': 9,
            'q-neg-ans-ood': 10,
        }
        self.mode = self.BATCH_TYPE_TO_MODE[self.batch_type]

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch['mode'] = torch.tensor([self.mode] * len(batch['input_ids']))
        return batch

class MultiTaskDataloader:
    def __init__(self, dataloader1, dataloader2):
        ## make sure dataloader1 is shorter than dataloader2
        if len(dataloader1) > len(dataloader2):
            dataloader1, dataloader2 = dataloader2, dataloader1
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.iterator1 = iter(dataloader1)
        self.iterator2 = iter(dataloader2)
        self.use_dataloader1 = True  # Use dataloader1 until it's finished
        self.batch_size = (dataloader1.batch_size + dataloader2.batch_size) // 2
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.use_dataloader1:
            try:
                batch = next(self.iterator1)
            except StopIteration:
                self.use_dataloader1 = False
                batch = next(self.iterator2)
        else:
            try:
                batch = next(self.iterator2)
            except StopIteration:
                raise StopIteration
                print("Finished both dataloaders")
                ## reset iterators
                self.iterator1 = iter(self.dataloader1)
                self.iterator2 = iter(self.dataloader2)
                batch = next(self.iterator1)
                self.use_dataloader1 = True

        
        self.use_dataloader1 = not self.use_dataloader1
        return batch

    def __len__(self):
        return len(self.dataloader1) + len(self.dataloader2) 

def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    device_batch_size: int,
):    
    # get kwargs
    mlm_probability = cfg.dataset.pop('mlm_probability', None)
    eos_token_id = cfg.dataset.pop('eos_token_id', None)
    bos_token_id = cfg.dataset.pop('bos_token_id', None)
    batch_type = cfg.dataset.pop('batch_type', 'lm')
    masking_args = cfg.dataset.pop('masking', None)
    shuffle = cfg.dataset.pop('shuffle', True)
    
    assert batch_type in ['lm', 'url', 'ictx', 'ictx-url', 
                          'fact', 'qu', 'qa', 'qa-ood', 'qa-cot', 'qa-cot-ood', 
                            'q-neg-ans', 'q-neg-ans-ood']
    
    
    if batch_type in ['lm', 'url', 'fact']:
        # build dataset potentially with streams

        if cfg.dataset.get('local', None):
            dataset = TextDataset(
                local=cfg.dataset.local,
                split=cfg.dataset.split,
                tokenizer=tokenizer,
                max_seq_len=cfg.dataset.max_seq_len,
            )
        
        elif cfg.dataset.get('name', None):
            dataset = TextDataset(
                dataset_uri=cfg.dataset.name,
                split=cfg.dataset.split,
                tokenizer=tokenizer,
                max_seq_len=cfg.dataset.max_seq_len,
            )

    elif batch_type == 'ictx':
        #if cfg.dataset.name in ['hellaswag', 'piqa']:
        #    ## dataset dir is parent dir of the jsonl file
        #    data_dir = os.path.dirname(cfg.dataset.path)
        #    dataset = InContextLearningMultipleChoiceTaskDataset(
        #        dataset_uri=cfg.dataset.path,
        #        tokenizer=tokenizer,
        #        max_seq_len=cfg.dataset.max_seq_len,
        #        pad_tok_id=tokenizer.pad_token_id,
        #        num_fewshot=cfg.dataset.n_demos,
        #        fewshot_random_seed=1,
        #        prompt_string='Answer the following questions:\n',
        #        example_delimiter='\n',
        #        continuation_delimiter=' ### ',
        #        destination_path=os.path.join(data_dir, cfg.dataset.name),
        #    )
        #    import ipdb; ipdb.set_trace()
        #else:
        dataset = InContextLearningDataset(
            name=cfg.dataset.name,
            split=cfg.dataset.split,
            tokenizer=tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            n_demos=cfg.dataset.n_demos,
        )

    elif batch_type == 'ictx-url':
        dataset = InContextLearningURLDataset(
            data_path=cfg.dataset.path,
            tokenizer=tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            n_demos=cfg.dataset.n_demos,
        )
    
    elif batch_type in ['qa', 'qa-ood', 'qa-cot', 'qa-cot-ood']:
        dataset = QuestionAnswerDataset(
            data_path=cfg.dataset.path,
            split=cfg.dataset.split,
            tokenizer=tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
        )
    
    elif batch_type in ['q-neg-ans', 'q-neg-ans-ood']:
        dataset = QuestionNegativeAnswerDataset(
            data_path=cfg.dataset.path,
            split=cfg.dataset.split,
            tokenizer=tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            use_gold_answer=cfg.dataset.use_gold_answer,
            append_url_token=cfg.dataset.append_url_token,
        )
    
    ## clone tokenizer: one with right padding for lm and one with left padding for url eval
    tokenizer_left = copy.deepcopy(tokenizer)
    tokenizer_right = copy.deepcopy(tokenizer)

    tokenizer_left.padding_side = 'left'
    tokenizer_right.padding_side = 'right'

    if batch_type in  ['lm', 'fact']:
        collate_fn = DataCollatorForURLLanguageModeling(
            tokenizer=tokenizer_right,
            mlm=mlm_probability is not None,
            mlm_probability=mlm_probability,
            masking_args=masking_args
        )
    
    elif batch_type in ['url', 'ictx', 'ictx-url', 'qu', 'qa', 'qa-ood', 'qa-cot', 'qa-cot-ood', 'q-neg-ans', 'q-neg-ans-ood']:
        collate_fn = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer_left,
            padding=True,
        )

    collate_fn = BatchTypeCollatorWrapper(
        base_collator=collate_fn,
        batch_type=batch_type,
    )

    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id)
    
    sampler = dist.get_sampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', False),
        prefetch_factor=cfg.get('prefetch_factor', 2 if cfg.num_workers > 0 else None),
        persistent_workers=cfg.get('persistent_workers', True if cfg.num_workers > 0 else False),
        timeout=cfg.get('timeout', 0),
        sampler=sampler,
    )

def build_text_dataset(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    ):    
    # get kwargs
    batch_type = cfg.batch_type
    
    assert batch_type in ['lm', 'url', 'ictx', 'ictx-url', 'fact', 'qu', 'qa', 'qa-ood'] ## lm = language modeling, url = url modeling, ictx = in-context learning task, ictx-url = in-context learning task with url prediction
    if batch_type in ['lm', 'url', 'fact']:
        # build dataset potentially with streams
        if cfg.get('local', None):
            dataset = TextDataset(
                local=cfg.local,
                split=cfg.split,
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
            )
        
        elif cfg.get('name', None):
            dataset = TextDataset(
                dataset_uri=cfg.name,
                split=cfg.split,
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
            )

    elif batch_type == 'ictx':
        dataset = InContextLearningDataset(
            name=cfg.dataset.name,
            split=cfg.dataset.split,
            tokenizer=tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            n_demos=cfg.dataset.n_demos,
        )
    
    elif batch_type == 'ictx-url':
        dataset = InContextLearningURLDataset(
            data_path=cfg.path,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            n_demos=cfg.n_demos,
        )
    
    return dataset
    
def build_mtl_dataloader(
    cfgs: List[DictConfig],
    tokenizer: Tokenizer,
    device_batch_size: int,
):    
    
    datasets = []
    for cfg in cfgs: 
        assert cfg.dataset.batch_type in ['lm', 'fact'], "MTL dataloader only supports language modeling and fact prediction tasks"
        dataset = build_text_dataset(cfg.dataset, tokenizer)
        datasets.append(dataset)

    cfg = cfgs[0]

    tokenizer_right = copy.deepcopy(tokenizer)
    tokenizer_right.padding_side = 'right'

    collate_fn = DataCollatorForURLLanguageModeling(
            tokenizer=tokenizer_right,
            mlm=False,
            mlm_probability=None,
            masking_args=cfg.dataset.pop('masking', None)
        )
    
    collate_fn = BatchTypeCollatorWrapper(
        base_collator=collate_fn,
        batch_type='lm',
    )

    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    sampler = dist.get_sampler(combined_dataset, shuffle=True)

    data_loader = DataLoader(
        combined_dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', False),
        prefetch_factor=cfg.get('prefetch_factor', 2 if cfg.num_workers > 0 else None),
        persistent_workers=cfg.get('persistent_workers', True if cfg.num_workers > 0 else False),
        timeout=cfg.get('timeout', 0),
        drop_last=cfg.drop_last,
    )

    return data_loader


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':
    import argparse

    from llmfoundry.utils.builders import build_tokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer',
                        type=str,
                        default='EleutherAI/gpt-neox-20b',
                        help='the name of the tokenizer to use')
    parser.add_argument('--local_path',
                        type=str,
                        required=False,
                        default=None,
                        help='the path to the local copy of the dataset')
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)')
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        help='which split of the dataset to use')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='max sequence length to test')

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f'Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}'
        )
    else:
        print(f'Reading {args.split} split from {args.local_path}')

    _cfg = {
        'name': 'text',
        'dataset': {
            'local': args.local_path,
            'name': 'ghmfx/natural-questions-short',
            'remote': args.remote_path,
            'split': args.split,
            'shuffle': False,
            'max_seq_len': args.max_seq_len,
            'n_demos': 16,
            'keep_zip': True,  # in case we need compressed files after 
            'batch_type': 'url' if 'url_val' in args.split else 'lm',
            'masking': {
                'cross_doc_attention': True
            },
        },
        'drop_last': False,
        'num_workers': 0,
        'persistent_workers': False,
    }

    __cfg = {
        'name': 'wikitext_ppl_eval',
        'dataset': {
            'name': 'wikitext',
            'split': 'test',
            'shuffle': False,
            'max_seq_len': args.max_seq_len,
            'batch_type': 'lm',
            'masking': {
                'cross_doc_attention': True
            },
        },
        'drop_last': False,
        'num_workers': 0,
        'persistent_workers': False,
    }

    cfg = om.create(__cfg)

    #with open('conf/llmfoundry/gpt2-large-bio-answer-attribution-eval.yaml') as f:
    #    yaml_cfg = om.load(f)
    
    #dl_cfg = yaml_cfg.dataloaders[-1] # QU 

    
    #dl1_cfg = yaml_cfg.dataloaders[0]
    #dl2_cfg = yaml_cfg.dataloaders[1]

    #device_batch_size = 4

    tokenizer_cfg = {'name': args.tokenizer, 'kwargs': {}}
    tokenizer_cfg['kwargs'] = {'model_max_length': args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size=1)
    #dl1 = build_text_dataloader(dl1_cfg, tokenizer, device_batch_size=1)
    #dl2 = build_text_dataloader(dl2_cfg, tokenizer, device_batch_size=1)

    ### cocnat datasets 
    #loader = build_mtl_dataloader([dl1_cfg, dl2_cfg], tokenizer, device_batch_size=4)

    import ipdb; ipdb.set_trace()

    __cfg = {
        'name': 'text',
        'dataset': {
            'path': '../processed-data/wikipedia/random_tokens_date_llama_100K_url_last_full_doc_vanilla_wiki/qa_url_500.jsonl',
            'remote': args.remote_path,
            'split': args.split,
            'shuffle': False,
            'n_demos': 16,
            'max_seq_len': args.max_seq_len,
            'batch_type': 'lm',
            'masking': {
                'cross_doc_attention': True
            },
        },
        'drop_last': False,
        'num_workers': 0,
        'persistent_workers': False,
    }


    for batch_ix, batch in enumerate(loader):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        
        #import ipdb; ipdb.set_trace()
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            text = tokenizer.decode(token_sample, skip_special_tokens=False)
            print(text)
        
        #if 'labels' in batch:
        #    for sample_ix, token_sample in enumerate(batch['labels']):
        #        token_sample[token_sample == -100] = tokenizer.eos_token_id
        #        print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
        #        print(tokenizer.decode(token_sample))
