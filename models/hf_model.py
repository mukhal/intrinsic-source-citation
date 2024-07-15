# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch.nn import functional as F
from torchmetrics import Metric

from composer.metrics import InContextLearningMetric
from composer.models.base import ComposerModel
from composer.utils import MissingConditionalImportError, dist, get_file, import_object, is_model_fsdp, safe_torch_load


if TYPE_CHECKING:
    import transformers
    from transformers import PretrainedConfig
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

log = logging.getLogger(__name__)


class HuggingFaceModelWithURLLoss(HuggingFaceModel):
    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: Optional[Union[transformers.PreTrainedTokenizer,
                                           transformers.PreTrainedTokenizerFast]] = None,
                 use_logits: Optional[bool] = False,
                 metrics: Optional[List[Metric]] = None,
                 eval_metrics: Optional[List[Metric]] = None,
                 shift_labels: Optional[bool] = None,
                 allow_embedding_resizing: bool = False,
                 args = None,
                 eval_args = None,
                 url_trie = None,
                 ) -> None:
        try:
            import transformers
            del transformers  # unused
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         use_logits=use_logits,
                         metrics=metrics,
                         eval_metrics=eval_metrics,
                         shift_labels=shift_labels,
                         allow_embedding_resizing=allow_embedding_resizing,
                         )
        self.args = args
        self.eval_args = eval_args
        self.url_trie = url_trie

        if eval_metrics is not None:
            self.val_metrics = {metric.name if hasattr(metric, 'name') else metric.__class__.__name__: metric for metric in eval_metrics}
        
        if metrics is not None:
            self.train_metrics = {metric.name if hasattr(metric, 'name') else metric.__class__.__name__: metric for metric in metrics}
            # if eval_metrics is None, use the same metrics as train_metrics
            if eval_metrics is None:
                self.val_metrics = {metric.name if hasattr(metric, 'name') else metric.__class__.__name__: metric for metric in metrics}
        

    def forward(self, batch):
        if isinstance(batch, Mapping):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output

    def loss(self, outputs, batch):
        logits = outputs['logits']
        labels = batch['labels']
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        
        url_mask = labels.new_zeros(labels.size())
        if 'url_mask' in batch:
            url_mask = batch['url_mask']
            url_mask = url_mask[..., 1:].contiguous()

        loss_type = self.args['loss']['type']
        
        if loss_type == 'lm':
            if self.config.use_return_dict:
                return outputs['loss']
            else:
                return outputs[0]
        
        elif loss_type == 'url_weighted':
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none') # B * L
            factor = self.args['loss']['url_loss_factor']
            url_mask = url_mask.view(-1) # B * L
            assert token_loss.size() == url_mask.size()
            non_url_loss = token_loss * (1 - url_mask)
            url_loss = token_loss * url_mask * factor
            total_items = (labels != -100).sum()
            return (non_url_loss + url_loss).sum() / total_items
        
        elif loss_type == 'url':
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none') # B * L
            url_mask = url_mask.view(-1) # B * L
            assert token_loss.size() == url_mask.size()
            loss = token_loss * url_mask
            total_items = (labels != -100).sum()
            return loss.sum() / total_items
        
        else:
            raise ValueError(f'Unrecognized loss type {self.args["train"]["loss"]}')

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        if 'url_input_ids' in batch: # URL EM evaluation
            inp_ids = batch['doc_input_ids']
            att_mask = batch['doc_attention_mask']
            input_dim = inp_ids.size(-1)
            if self.url_trie is not None: ## constrained decoding 
                prefix_allowed_tokens_fn = lambda _, input_ids: self.url_trie.get(input_ids[input_dim - 1:].tolist())
            else:
                prefix_allowed_tokens_fn = None

            output = self.generate(inp_ids,
                                    attention_mask=att_mask, max_new_tokens=self.eval_args['max_new_tokens'],
                                    eos_token_id=self.tokenizer.additional_special_tokens_ids[1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=self.eval_args['num_beams'],
                                    num_return_sequences=self.eval_args['num_samples'],
                                    do_sample=not self.eval_args.get('num_samples', 1) == 1,
                                    temperature=self.eval_args['temperature'],
                                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                    )
            output = output[:, inp_ids.shape[1]:]
            return output, batch # outputs, targets 
        
        elif 'labels' in batch: ## LM evaluation
            outputs = self.forward(batch)
            logits = outputs['logits']
            labels = batch['labels']
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            return logits, labels


    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        metric.update(*outputs)
