# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from collections.abc import Mapping
from transformers.data.data_collator import _torch_collate_batch
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class DataCollatorForURLLanguageModeling(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: Tokenizer, mlm: bool = False, mlm_probability: float = 0.15, pad_to_multiple_of: Optional[int] = None):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            ## remove para_idx from examples
            _examples = [{k: v for k, v in e.items() if k != 'para_idx'} for e in examples]
            batch = self.tokenizer.pad(_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            if 'para_idx' in examples[0]:
                batch['para_idx'] = _torch_collate_batch([e['para_idx'] for e in examples], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

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


    
class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 max_seq_len: int,
                 streams: Optional[Sequence[Stream]] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: int = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1b',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 **kwargs: Dict[str, Any]):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample, key='tokens'):
        return torch.from_numpy(
            np.frombuffer(sample[key],
                          dtype=np.int64)[:self.max_seq_len].copy())

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        if 'text' in sample:
            token_sample = self._tokenize(sample)
        elif 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
            print("calling with idx: ", idx)
            if 'para_idx' in sample:
                para_idx_sample = self._read_binary_tokenized_sample(sample, key='para_idx')
                assert len(token_sample) == len(para_idx_sample)
                token_sample = {
                    'input_ids': token_sample,
                    'para_idx': para_idx_sample,
                }

        elif 'doc_ids' in sample:
            doc_ids_sample = self._read_binary_tokenized_sample(sample, key='doc_ids')
            url_ids_sample = self._read_binary_tokenized_sample(sample, key='url_ids')
            token_sample = {
                'input_ids': doc_ids_sample,
                'labels': url_ids_sample,
            }
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `text` or `tokens` column'
            )
        return token_sample


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

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch['mode'] = torch.tensor([0 if self.batch_type == 'lm' else -1] * len(batch['input_ids']))
        return batch


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    device_batch_size: int,
):
    assert cfg.name == 'text', f'Tried to build text dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset_hf.py'
        )
    
    # get kwargs
    streams_dict = cfg.dataset.pop('streams', None)
    mlm_probability = cfg.dataset.pop('mlm_probability', None)
    eos_token_id = cfg.dataset.pop('eos_token_id', None)
    bos_token_id = cfg.dataset.pop('bos_token_id', None)
    batch_type = cfg.dataset.pop('batch_type', 'lm')

    assert batch_type in ['lm', 'url']

    # build dataset potentially with streams
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        **cfg.dataset,
    )
    
    if batch_type == 'lm':
        collate_fn = DataCollatorForURLLanguageModeling(
            tokenizer=dataset.tokenizer,
            mlm=mlm_probability is not None,
            mlm_probability=mlm_probability)
    
    elif batch_type == 'url':
        collate_fn = transformers.DataCollatorForSeq2Seq(
            tokenizer=dataset.tokenizer,
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

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', False),
        persistent_workers=cfg.get('persistent_workers', False),
        timeout=cfg.get('timeout', 0),
    )


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
                        required=True,
                        help='the path to the local copy of the dataset')
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)')
    parser.add_argument('--split',
                        type=str,
                        default='val',
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

    cfg = {
        'name': 'text',
        'dataset': {
            'local': args.local_path,
            'remote': args.remote_path,
            'split': args.split,
            'shuffle': True,
            'max_seq_len': args.max_seq_len,
            'keep_zip': True,  # in case we need compressed files after 
            'batch_type': 'url' if 'url_val' in args.split else 'lm',
        },
        'drop_last': True,
        'num_workers': 0,
    }
    cfg = om.create(cfg)
    device_batch_size = 1

    tokenizer_cfg = {'name': args.tokenizer, 'kwargs': {}}
    tokenizer_cfg['kwargs'] = {'model_max_length': args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(loader):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            text = tokenizer.decode(token_sample)
            #print(text)
        
        import ipdb; ipdb.set_trace()

        #if 'labels' in batch:
        #    for sample_ix, token_sample in enumerate(batch['labels']):
        #        token_sample[token_sample == -100] = tokenizer.eos_token_id
        #        print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
        #        print(tokenizer.decode(token_sample))