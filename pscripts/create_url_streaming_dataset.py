# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
import os, sys
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union
import datasets as hf_datasets
import psutil
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from llmfoundry.data import PackingURLDataset, NoPackingURLDataset
from llmfoundry.data.constants import NO_URL
import numpy as np  

sys.path.append('../')
sys.path.append('./')

from utils.trie import MarisaTrie
import pickle as pkl


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

class URLTokens(Enum):
    START_TOKEN = '<url>'
    END_TOKEN = '</url>'

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--data_subset',
                        type=str,
                        default=None,
                        help='E.g. "all" or "en"')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'train_small', 'val', 'val_small', 'val_xsmall'])
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    parser.add_argument('--packing_method', type=str, default='standard', choices=['standard', 'url_last', 'url_first', 'question_url_answer', 'url_repeat_in_domain',
    'question_answer_doc_url',  'question_answer_url', 'no_url', 'url_first_and_last', 'question_answer'])
    parser.add_argument('--no_reset_doc_positions', default=False, action='store_true')
    parser.add_argument('--no_url', default=False, action='store_true')
    parser.add_argument('--repeat_url_in_doc', default=False, action='store_true')
    parser.add_argument('--repeat_every', type=str, choices=['sentence', 'tokens'])
    parser.add_argument('--predict_answer_only', default=False, action='store_true')
    parser.add_argument('--predict_url_only', default=False, action='store_true')
    parser.add_argument('--build_trie', default=False, action='store_true')
    parser.add_argument('--truncate_num_samples', type=int, default=None)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--out_folder', type=str, default=None, help="if set, will save to out_root/out_folder instead of out_root/split")
    parser.add_argument('--n_attribution_negs_per_question', type=int, default=0, help='number of answer negatives to use for the NO_URL case')
    parser.add_argument('--neg_create_probability', type=float, default=0.2, help='probability of creating an attribution negative example given a QA-pair')
    parser.add_argument('--percentage_in_domain_repeat_url', type=float, default=1.0, help='percentage of times to repeat the URL in the domain')

    parsed = parser.parse_args()

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    
    if 'gpt2' in parsed.tokenizer:
        parsed.bos_text = '<|endoftext|>'
    return parsed

def build_hf_dataset(
    dataset_path: str,
    split: str,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    tokenizer: PreTrainedTokenizerBase = None,
    packing_args: dict = None,
    no_url: bool = False,
    shuffle: bool = False,

) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    hf_dataset = hf_datasets.load_from_disk(dataset_path=os.path.join(dataset_path, split))
    if shuffle:
        hf_dataset = hf_dataset.shuffle(seed=42)
    
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(
            f'{tokenizer=} must be of type PreTrainedTokenizerBase')
    if max_length is None:
        raise ValueError(f'max_length must be set.')
    if bos_text + eos_text == '':
        test_tokens = tokenizer('test')
        if test_tokens['input_ids'][
                0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                    -1] != tokenizer.eos_token_id:
            tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
            tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
            tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
            tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
            tok_error_msg += '--bos_text=<|endoftext|>.'
            raise ValueError(tok_error_msg)
    
    include_url = not 'ppl' in split and not no_url

    packing_args['method'] = 'standard' if 'ppl' in split else packing_args['method']
    if ('train' in split) or 'ppl_val' in split: # Pack Many Passages into One Sequence. 
        data_cls = PackingURLDataset
    elif 'url_val' in split or 'fact' in split or 'qa' in split:
        data_cls = NoPackingURLDataset # no packing for URL recall eval
    else: 
        raise ValueError(f'Unknown split: {split}')
            
    dataset = data_cls(hf_dataset=hf_dataset,
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    bos_text=bos_text,
                                    eos_text=eos_text,
                                    url_special_tokens = URLTokens,
                                    include_url=include_url,
                                    packing_args=packing_args,
                                    eval='val' in split,
                                    )
    return dataset


def _est_progress_denominator(total_samples: int, chars_per_sample: int,
                              chars_per_token: int, mode: ConcatMode,
                              max_length: int):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length


def build_dataloader(dataset, batch_size, num_workers) -> DataLoader:
    num_workers = 0 ## has to be 0 otherwise there's a data duplication issue! 
    assert num_workers == 0, 'num workers must be 0 for now otherwise there is a data duplication issue!'
    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    mode = ConcatMode.CONCAT_TOKENS
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens({'additional_special_tokens': [URLTokens.START_TOKEN.value, URLTokens.END_TOKEN.value]})
    tokenizer.pad_token = tokenizer.eos_token
    
    # we will enforce length, so suppress warnings about sequences too long for the model
    #tokenizer.model_max_length = int(1e30)

    ### make sure the url special tokens are in the tokenizer
    assert URLTokens.START_TOKEN.value in tokenizer.additional_special_tokens, "URL start token not in tokenizer"
    assert URLTokens.END_TOKEN.value in tokenizer.additional_special_tokens, "URL end token not in tokenizer"

    for split_name in args.splits:
        hf_split = split_name
        folder_split = split_name
        expected_num_samples = None
        truncate_num_samples = args.truncate_num_samples
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Get samples
        dataset = build_hf_dataset(dataset_path=args.dataset_path,
                                   split=hf_split,
                                   max_length=args.concat_tokens,
                                   bos_text=args.bos_text,
                                   eos_text=args.eos_text,
                                   tokenizer=tokenizer,
                                   packing_args= {
                                        'method': args.packing_method,
                                        'reset_doc_positions': not args.no_reset_doc_positions,
                                        'repeat_url_in_doc': args.repeat_url_in_doc,
                                        'repeat_every': args.repeat_every,
                                        'predict_answer_only': args.predict_answer_only,
                                        'predict_url_only': args.predict_url_only,
                                        'n_attribution_negs_per_question': args.n_attribution_negs_per_question,
                                        'neg_create_probability': args.neg_create_probability,
                                        'percentage_in_domain_repeat_url': args.percentage_in_domain_repeat_url,
                                   },
                                    no_url=args.no_url,
                                    shuffle=args.shuffle,
                                   )

        
        loader = build_dataloader(dataset=dataset,
                                  batch_size=512,
                                  num_workers=args.num_workers)
        samples = generate_samples(loader,
                                   truncate_num_samples=truncate_num_samples)

        if expected_num_samples is not None:
            denominator = truncate_num_samples if truncate_num_samples is not None else _est_progress_denominator(
                total_samples=expected_num_samples,
                chars_per_sample=1000,
                chars_per_token=1000,
                mode=mode,
                max_length=args.concat_tokens,
            )
        else:
            denominator = None

        # Write samples
        print(f'Converting {folder_split} to MDS format...')
        print(
            f'Note that the progress bar is based on the dataset length before tokenization.'
        )
        print(f'It will finish at a value below 100% if tokenizing')

        ### save the dataset as .npy files

        ## create folder if not exists
        outpath = os.path.join(args.out_root, folder_split if args.out_folder is None else args.out_folder)
        os.makedirs(outpath, exist_ok=True)
        
        _samples = [] 
        with open(outpath + '/shard.npy', 'wb') as f:
            for sample in tqdm(samples, desc=folder_split,
                               total=denominator, disable=True):
                _samples.append(sample)
            np.save(f, _samples)
        
        ## save tokenizer 
        tokenizer.save_pretrained(os.path.join(args.out_root, 'tokenizer'))

        #### build URL tries
        if os.path.exists(os.path.join(args.dataset_path, 'train')) and args.build_trie:
            print("Building URL tries for train split")
            hf_dataset = hf_datasets.load_from_disk(dataset_path=os.path.join(args.dataset_path, 'train'))
            ## build and save URL trie
            url_ids = [] 
            for d in hf_dataset:
                url = d['url']
                ids = tokenizer(url, add_special_tokens=False)['input_ids']
                ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
                url_ids.append(ids)
            
            ### ADD NO-URL case 
            print("Adding no-url case...")
            url = NO_URL
            ids = tokenizer(url, add_special_tokens=False)['input_ids']
            ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
            url_ids.append(ids)
                
            url_trie = MarisaTrie(sequences=url_ids)
            pkl.dump(url_trie, open(os.path.join(args.out_root, 'url_trie.pkl'), 'wb'))

if __name__ == '__main__':
    main(parse_args())
