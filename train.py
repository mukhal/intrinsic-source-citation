# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sys
import warnings
import pickle as pkl
import torch

from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from omegaconf import OmegaConf as om

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.data.vanilla_text_data import build_text_dataloader, build_mtl_dataloader
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_icl_evaluators, build_logger,
                                       build_optimizer, build_scheduler,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import log_config, update_batch_size_info
from utils.eval import URLExactMatch, QAF1, QAEM, URLContentSupportsWordOverlap, HitsAtK
import datasets as hf_ds
from datasets import disable_caching


disable_caching()

def build_composer_model(model_cfg, tokenizer, decoding_trie=None, 
                         ood_decoding_trie=None):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')
    
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer, 
                                                   decoding_trie=decoding_trie, 
                                                    ood_decoding_trie=ood_decoding_trie)

def print_trainable_parameters(model) -> None:
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )

def build_dataloader(cfg, tokenizer, device_batch_size):
    return build_text_dataloader(
        cfg,
        tokenizer,
        device_batch_size,
    )

def main(cfg):
    # Check for incompatibilities between the model and data loaders
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        f'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    cfg.dist_timeout = cfg.get('dist_timeout', 600.0)

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        cfg.pop('fsdp_config')
        fsdp_config = None

    deepspeed_config = cfg.get('deepspeed_config', None)
    deepspeed_config = om.to_container(
        deepspeed_config, resolve=True) if deepspeed_config else None
    if dist.get_world_size() == 1 and deepspeed_config is not None:
        warnings.warn(
            'DeepSpeed is not applicable for single-GPU training. Reverting to DDP.'
        )
        cfg.pop('deepspeed_config')
        deepspeed_config = None

    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in cfg.model:
        assert cfg.model.init_device in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and cfg.model.init_device == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            cfg.model.init_device = 'cpu'
        if cfg.model.init_device == 'meta':
            init_context = init_empty_weights()
        if cfg.model.init_device == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. '
                    'Please add a FSDP config.')
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('use_orig_params', False)
            fsdp_config.setdefault('load_monolith_rank0_only', True)

    # build tokenizer
    tokenizer = build_tokenizer(cfg.tokenizer)
    ### add <FACT> token to tokenizer if not already there
    #if '<FACT>' not in tokenizer.get_vocab():
    #    tokenizer.add_tokens('<FACT>')

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    url_trie = None 

    if cfg.get('url_trie', None):
        print(f'Loading URL trie from {cfg.url_trie}...') 
        url_trie = pkl.load(open(cfg.url_trie, 'rb'))

    if cfg.get('ood_url_trie', None):
        print(f'Loading OOD URL trie from {cfg.ood_url_trie}...') 
        ood_url_trie = pkl.load(open(cfg.ood_url_trie, 'rb'))

    ### build URL_TO_DOC mapping for wordoverlap metric 
    #path_to_train_data = os.path.join(cfg.text_data_path, 'train')
    #train_ds = hf_ds.load_from_disk(path_to_train_data)
    #url_to_doc = {d['url']: d['text'] for d in train_ds}
    #print("built URL to doc mapping for wordoverlap metric...")
    
    # Build Model
    print('Initializing model...')
    with init_context:
        model = build_composer_model(cfg.model, tokenizer, decoding_trie=url_trie, 
                                        ood_decoding_trie=ood_url_trie)
        model.val_metrics['URL-EM'] = URLExactMatch(tokenizer=tokenizer, name='URL-EM')
        model.val_metrics['QA-EM'] = QAEM(tokenizer=tokenizer, name='QA-EM')
        model.val_metrics['QA-F1'] = QAF1(tokenizer=tokenizer, name='QA-F1')
        model.val_metrics['Hits@1-att'] = HitsAtK(tokenizer=tokenizer, k=1, name='Hits@1-att', attributable=True)
        model.val_metrics['Hits@10-att'] = HitsAtK(tokenizer=tokenizer, k=10, name='Hits@10-att', attributable=True)
        ### hits@3
        model.val_metrics['Hits@3-att'] = HitsAtK(tokenizer=tokenizer, k=3, name='Hits@3-att', attributable=True)
    
        #### build URLContentSupportsWordOverlap metric
        #model.val_metrics['URL-WordOverlap'] = URLContentSupportsWordOverlap#(tokenizer=tokenizer,
        #    url_to_doc=url_to_doc
        #name='URL-WordOverlap')                                                            
        for sp_id in tokenizer.additional_special_tokens_ids:
            if 'gpt' in cfg.model.pretrained_model_name_or_path:
                #model.model.transformer.wte.weight.data[sp_id] = model.model.transformer.wte.weight.data[2638] ## token id for http
                ## initialize with mean of all tokens
                model.model.transformer.wte.weight.data[sp_id] = model.model.transformer.wte.weight.data.mean(dim=0)
            elif 'llama' in cfg.model.pretrained_model_name_or_path:
                #model.model.model.embed_tokens.weight.data[sp_id] = model.model.model.embed_tokens.weight.data[1732] ## token id for http
                ## initialize with mean of all tokens
                model.model.model.embed_tokens.weight.data[sp_id] = model.model.model.embed_tokens.weight.data.mean(dim=0)

        if cfg.model.get('checkpoint', None):
            print(f'Loading model weights from {cfg.model.checkpoint}...')
            ckpt = torch.load(cfg.model.checkpoint, map_location='cpu')
            if 'model.transformer.wte.weight' in ckpt:
                ckpt['model.lm_head.weight'] = ckpt['model.transformer.wte.weight']

            model.load_state_dict(
                ckpt
                )
        
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')

    # Dataloaders
    print ('Building dataloaders...')
    train_loader_cfgs = []
    evaluators = []

    for dloader in cfg.get('dataloaders', []):
        print('Building {}...'.format(dloader.name))        
        
        if 'train' in dloader.name:
            train_loader_cfgs.append(dloader)
            continue
        
        #### EVAL LOADERS
        evaluator_label = dloader.name.replace('_loader', '').replace('_', '-')

        if 'answer' in dloader.name or 'attribution' in dloader.name:
            loader = build_dataloader(
                                    dloader, tokenizer,
                                    cfg.device_eval_batch_size)
            
            evaluator = Evaluator(label=evaluator_label,
                                    dataloader=loader,
                                    metric_names=['QA-EM', 'QA-F1', 'Hits@1-att', 
                                                  'Hits@10-att'])
 
        elif 'ppl' in dloader.name:
            loader = build_dataloader(
                                    dloader, tokenizer,
                                    cfg.device_eval_batch_size)
            evaluator = Evaluator(label=evaluator_label,
                                  dataloader=loader,
                                metric_names=['LanguagePerplexity'])
        
        elif 'ictx' in dloader.name: ## QA 
            ictx_eval_loader = build_dataloader(
                                    dloader, tokenizer,
                                    cfg.device_eval_batch_size * 4)
            evaluator = Evaluator(label=evaluator_label,
                                dataloader=ictx_eval_loader,
                                metric_names=['QA-EM', 
                                              'QA-F1'])
        else:
            print('Unused dataloader: {}'.format(dloader.name))
            continue
        
        evaluators.append(evaluator)

    print('We have {} evaluators'.format(len(evaluators)))

    if len(train_loader_cfgs) == 0:
        train_loader = None

    elif len(train_loader_cfgs) == 1:
        train_loader = build_text_dataloader(
            train_loader_cfgs[0],
            tokenizer,
            cfg.device_train_batch_size,
        )
    else:
        ## MTL setting 
        print('Building MTL train dataloader..')
        #assert len(train_loaders) == 2, "We only support MTL with 2 tasks"
        #train_loader = build_mtl_dataloader(train_loaders[0], train_loaders[1])
        train_loader = build_mtl_dataloader(
            train_loader_cfgs,
            tokenizer,
            cfg.device_train_batch_size,
        )
    
    if train_loader is not None:
        optimizer = build_optimizer(cfg.optimizer, model)
        scheduler = build_scheduler(cfg.scheduler)
    else:
        optimizer = None
        scheduler = None
    

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get('callbacks') or {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
    ]

    ## save train config to cfg.save_folder 
    if cfg.get('save_folder', None):
        os.makedirs(cfg.save_folder, exist_ok=True)
        with open(os.path.join(cfg.save_folder, 'train_config.yaml'), 'w') as f:
            om.save(cfg, f)
    

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '50ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        fsdp_config=fsdp_config,  # type: ignore
        deepspeed_config=deepspeed_config,  # type: ignore
        save_folder=cfg.get('save_folder', None),
        save_filename=cfg.get('save_filename',
                              'ep{epoch}-ba{batch}-rank{rank}.pt'),
        save_latest_filename=cfg.get('save_latest_filename',
                                     'latest-rank{rank}.pt'),
        save_interval=cfg.get('save_interval', '1ep'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        save_weights_only=cfg.get('save_weights_only', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        load_ignore_keys=cfg.get('load_ignore_keys', None),
        autoresume=cfg.get('autoresume', False),
        python_log_level=cfg.get('python_log_level', 'debug'),
        dist_timeout=cfg.dist_timeout,
    )

    print('Logging config...')
    log_config(cfg)

    if cfg.get('eval_first',
               False) and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    if train_loader is not None:
        print('Starting training...')
        trainer.fit()

    print('Done.')


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
