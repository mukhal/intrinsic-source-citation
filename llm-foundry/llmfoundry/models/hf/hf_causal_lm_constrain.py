# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""
from typing import Mapping, Union, Optional, Any

import os
# required for loading a python model into composer
import random
import torch
import transformers
from omegaconf import DictConfig
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from torch.nn import functional as F
from torchmetrics import Metric

__all__ = ['ComposerHFCausalLMWithConstrainedDecoding']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

class ComposerHFCausalLMWithConstrainedDecoding(ComposerHFCausalLM):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        om_model_config (DictConfig | transformers.PreTrainedModel): either an omegaconf dictionary used to configure the model, or an instantiated model object.
        if DictConfig, the following keys are required:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self,
                 om_model_config: Union[DictConfig,
                                        transformers.PreTrainedModel],
                 tokenizer: Tokenizer,
                 decoding_trie=None,
                 ood_decoding_trie=None,):

        composer_model = super().__init__(om_model_config, tokenizer)
        self.decoding_trie = decoding_trie
        self.ood_decoding_trie = ood_decoding_trie
       
        return composer_model

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        if batch['mode'][0].item() == 2: # In-Context Learning evaluation 
            batch.pop('mode')
            inputs = batch['input_ids']
            labels = batch['labels']
            labels[labels == -100] = self.tokenizer.pad_token_id
            self.labels = labels

            ## generate conitnuations
            output = self.generate(inputs,
                                    attention_mask=batch['attention_mask'],
                                    max_new_tokens=16,
                                    eos_token_id=self.tokenizer.encode('\n')[-1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=1,
                                    num_return_sequences=1,
                                    do_sample=False,
            )

            output = output[:, inputs.shape[1]:]
            
            #if os.environ.get('DEBUG', False):
            #    for i in range(inputs.shape[0]):
            #        if random.random() < 0.1:
            #            print("input: {}".format(self.tokenizer.decode(inputs[i], skip_special_tokens=True)))
            #            print("output: {}".format(self.tokenizer.decode(output[i], skip_special_tokens=True)))
            #            print("target: {}".format(self.tokenizer.decode(labels[i], skip_special_tokens=True)))
            
            return output
        
        if batch['mode'][0].item() in [1, 9, 10]: # URL/docid evaluation 
            mode = batch['mode'][0].item()
            batch.pop('mode')
            doc_ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            url_ids = batch['labels']

            url_ids[url_ids == -100] = self.tokenizer.pad_token_id
            self.labels = url_ids
            input_dim = doc_ids.shape[1]

            decoding_trie = self.ood_decoding_trie if mode == 10 else self.decoding_trie
            
            if decoding_trie is None:
                raise ValueError('decoding_trie is None')

                                    
            output = self.generate(doc_ids,
                                    attention_mask=attn_mask, 
                                    max_new_tokens=45,
                                    eos_token_id=self.tokenizer.additional_special_tokens_ids[1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=10,
                                    num_return_sequences=10,
                                    do_sample=False,
                                    decoding_trie=decoding_trie,
                                    prefix_allowed_tokens_fn=True,
                                    )
            output = output[:, input_dim:]

            ### get the top 10 predictions
            if os.environ.get('DEBUG', False):
                for i in range(doc_ids.shape[0]):
                    if random.random() < 0.1:
                        print("input: {}".format(self.tokenizer.decode(doc_ids[i], skip_special_tokens=True)))
                        print("output: {}".format(self.tokenizer.decode(output[i], skip_special_tokens=True)))
                        print("target: {}".format(self.tokenizer.decode(url_ids[i], skip_special_tokens=True)))
            return output
        
        if batch['mode'][0].item() in [5, 6]: # QA and Attribution evaluation
            mode = batch['mode'][0].item()
            batch.pop('mode')
            doc_ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            url_ids = batch['labels']
            url_ids[url_ids == -100] = self.tokenizer.pad_token_id
            self.labels = url_ids
            input_dim = doc_ids.shape[1]
   
            decoding_trie = self.decoding_trie if mode == 5 else self.ood_decoding_trie
            if decoding_trie is None:
                raise ValueError('decoding_trie is None')
                    
            ### generate the rest of the doc
            output = self.generate(doc_ids,
                                    attention_mask=attn_mask, 
                                    max_new_tokens=40,
                                    eos_token_id=self.tokenizer.encode('xx ##')[-1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=1,
                                    num_return_sequences=1,
                                    do_sample=False,
                                    )
            output_answer_tokens = output[:, input_dim:]

            ####### decode then retokenize
            bos_token_if_needed = self.tokenizer.bos_token if self.tokenizer.bos_token_id in doc_ids[0].tolist() else ''
            ####### decode then retokenize
            output_text = [bos_token_if_needed + self.tokenizer.decode(output[i], skip_special_tokens=True).strip() + '<url>' for i in range(output.shape[0])]

            ## tokenize with left-pad for .generate()
            ## chanfge padding side to left 
            self.tokenizer.padding_side = 'left'
            output_tokenized = self.tokenizer(output_text, padding=True, truncation=True, return_tensors='pt').to(doc_ids.device)
            self.tokenizer.padding_side = 'right'

            ### generate the URL with constrained decoding
            output = self.generate(output_tokenized['input_ids'],
                                    attention_mask=output_tokenized['attention_mask'],
                                    max_new_tokens=40,
                                    eos_token_id=self.tokenizer.additional_special_tokens_ids[1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=10,
                                    num_return_sequences=10,
                                    do_sample=False,
                                    decoding_trie=decoding_trie,
                                    prefix_allowed_tokens_fn=True,
                                    )
            
            output = output[:, output_tokenized['input_ids'].shape[1]:]
            ### combine answer tokens with the <url> token with the corresponding URL tokens
            url_start_token = torch.tensor(output.shape[0] * [self.tokenizer.additional_special_tokens_ids[0]]).unsqueeze(1).to(output.device)
            
            output_urls = torch.cat([url_start_token, output], dim=1)
            output = (output_answer_tokens, output_urls)

            if os.environ.get('DEBUG', False):
                for i in range(doc_ids.shape[0]):
                    if random.random() < 0.05:
                        print("input: {}".format(self.tokenizer.decode(doc_ids[i], skip_special_tokens=True)))
                        print("output_answer: {}".format(self.tokenizer.decode(output[0][i], skip_special_tokens=True)))
                        print("output_url: {}".format(self.tokenizer.decode(output[1][i*10], skip_special_tokens=True)))
                        print("target url: {}".format(self.tokenizer.decode(url_ids[i], skip_special_tokens=True)))

            return output     

        if batch['mode'][0].item() in [7, 8]: # COT QA and Attribution evaluation
            mode = batch['mode'][0].item()
            batch.pop('mode')
            doc_ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            url_ids = batch['labels']
            url_ids[url_ids == -100] = self.tokenizer.pad_token_id
            self.labels = url_ids
            input_dim = doc_ids.shape[1]
   
            decoding_trie = self.decoding_trie if mode == 7 else self.ood_decoding_trie
            if decoding_trie is None:
                raise ValueError('decoding_trie is None')

                    
            ### generate the rest of the doc
            output = self.generate(doc_ids,
                                    attention_mask=attn_mask, 
                                    max_new_tokens=300,
                                    eos_token_id=self.tokenizer.additional_special_tokens_ids[0],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=1,
                                    num_return_sequences=1,
                                    do_sample=False,
                                    )

            output_answer_tokens = output[:, input_dim:]
            ### check if bos token is needed
            bos_token_if_needed = self.tokenizer.bos_token if self.tokenizer.bos_token_id in doc_ids[0].tolist() else ''
            
            ####### decode then retokenize
            output_text = [bos_token_if_needed + self.tokenizer.decode(output[i], skip_special_tokens=True).strip() + '<url>' for i in range(output.shape[0])]
            #print("output_text: {}".format(output_text))
            ## tokenize with left-pad for .generate()
            ## chanfge padding side to left 
            self.tokenizer.padding_side = 'left'
            output_tokenized = self.tokenizer(output_text, padding=True, truncation=True, return_tensors='pt').to(doc_ids.device)
            self.tokenizer.padding_side = 'right'

            ### generate the URL with constrained decoding
            output = self.generate(output_tokenized['input_ids'],
                                    attention_mask=output_tokenized['attention_mask'],
                                    max_new_tokens=40,
                                    eos_token_id=self.tokenizer.additional_special_tokens_ids[1],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    num_beams=10,
                                    num_return_sequences=10,
                                    do_sample=False,
                                    decoding_trie=decoding_trie,
                                    prefix_allowed_tokens_fn=True,
                                    )
            output = output[:, output_tokenized['input_ids'].shape[1]:]

            ### combine answer tokens with the <url> token with the corresponding URL tokens
            url_start_token = torch.tensor(output.shape[0] * [self.tokenizer.additional_special_tokens_ids[0]]).unsqueeze(1).to(output.device)
            
            output_urls = torch.cat([url_start_token, output], dim=1)
            output = (output_answer_tokens, output_urls)

            if os.environ.get('DEBUG', False):
                for i in range(doc_ids.shape[0]):
                    if random.random() < 0.05:
                        print("input: {}".format(self.tokenizer.decode(doc_ids[i], skip_special_tokens=True)))
                        print("output_answer: {}".format(self.tokenizer.decode(output[0][i], skip_special_tokens=True)))
                        print("output_url: {}".format(self.tokenizer.decode(output[1][i*10], skip_special_tokens=True)))
                        print("target url-answer: {}".format(self.tokenizer.decode(url_ids[i], skip_special_tokens=True)))

            return output     

        elif batch['mode'][0].item() == 0: # standard CAUSAL LM evaluation
            batch.pop('mode')
            inputs = batch['input_ids']
            outputs = self.forward(batch)
            labels = batch['labels']
            labels[:, :-1] = labels[:, 1:].clone()
            labels[:, -1] = -100
            self.labels = labels
            return outputs

    def forward(self, batch):
        if isinstance(batch, Mapping):
            # Further input validation is left to the huggingface forward call
            assert 'attention_mask' in batch, 'attention_mask must be in batch'
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            #for i in range(len(batch['input_ids'])):
            #    if 'When was' in self.tokenizer.decode(batch['input_ids'][i]):
            #        print("input: {}".format(self.tokenizer.decode(batch['input_ids'][i])))
            #batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
            #del batch['attention_mask']
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output
    
    def _get_url_mask(self, batch):
        para_idx = batch['para_idx']
        url_mask = (para_idx < 0).long()  
        return url_mask
    
    def loss(self, outputs, batch):
        logits = outputs['logits']
        labels = batch['labels']
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        loss_type = self.loss_args['type']

        if loss_type in ['url', 'url_weighted']:
            url_mask = self._get_url_mask(batch)
        else:
            url_mask = None

        if loss_type == 'lm':
            if self.config.use_return_dict:
                return outputs['loss']
            else:
                return outputs[0]
        
        elif loss_type in ['mask', 'url']:
            mask = batch['loss_mask'] if loss_type == 'mask' else url_mask
            mask = mask[..., 1:].contiguous()
            assert labels.size() == mask.size(), f'{logits.size()} != {mask.size()}'
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none') # B * L
            mask = mask.view(-1) # B * L
            assert token_loss.size() == mask.size()
            loss = token_loss * mask
            total_items = mask.sum()
            return loss.sum() / total_items
        
        elif loss_type == 'url_weighted':
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none') # B * L
            factor = self.loss_args['url_loss_factor']
            url_mask = url_mask.view(-1) # B * L
            assert token_loss.size() == url_mask.size()
            non_url_loss = token_loss * (1 - url_mask)
            url_loss = token_loss * url_mask * factor
            total_items = (labels != -100).sum()
            return (non_url_loss + url_loss).sum() / total_items
        
    
        else:
            raise ValueError(f'Unrecognized loss type {loss_type}.')

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        if hasattr(metric, 'name') and 'overlap' in metric.name.lower(): ### needs the input doc
            assert self.labels is not None
            metric.update(batch, outputs, self.labels)  # 
        else:
            metric.update(outputs, self.labels)  # pyright: ignore [reportGeneralTypeIssues]

