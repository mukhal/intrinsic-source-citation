import transformers
import json
from composer.utils import reproducibility
from composer.loggers import WandBLogger, ProgressBarLogger
from data.url_dataset import C4URLDataset, WikiURLDataset
from data.lm_dataset import LMDataset
import datasets
from torch.utils.data import DataLoader
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.metrics.nlp import LanguageCrossEntropy
from composer import Trainer, Evaluator
from utils.eval import CountFlops, URLExactMatch, URLContentSupportsWordOverlap, URLContentSupportsNLI
import hydra
from omegaconf import DictConfig, OmegaConf
import logging 
from composer.utils import dist
from models.hf_model import HuggingFaceModelWithURLLoss
from utils.trie import MarisaTrie
from sentence_transformers import CrossEncoder
from deepspeed.ops.adam import DeepSpeedCPUAdam
import os 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='conf_wiki_perturb')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    reproducibility.seed_all(cfg.train.seed)
    model_name = cfg.model.name

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cfg.misc.cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.misc.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = datasets.load_from_disk(cfg.data.train_path).select(range(cfg.data.train_size))
    ## make sure the url field is present
    assert 'url' in train_data.column_names, 'URL field not present in data, please construct URLs first.'
    
    train_dataset = WikiURLDataset(train_data, tokenizer=tokenizer, max_length=cfg.data.max_length, split='train', url_max_length=cfg.data.max_url_length, 
    duplicate_url=cfg.data.get('duplicate_url', False))
    
    if cfg.data.eval_path is not None:
        eval_data = datasets.load_from_disk(cfg.data.eval_path)
    
    eval_data = eval_data.select(range(cfg.data.eval_size))
    ## make sure all eval document are in the train dataset
    eval_data = eval_data.filter(lambda x: x['url'] in train_data['url'])
    assert len(eval_data) == cfg.data.eval_size, 'Either the eval data size is larger than possible or that not all eval documents are in the train dataset!'

    exact_eval_dataset = WikiURLDataset(eval_data, tokenizer=tokenizer, max_length=cfg.data.max_length, split='eval', url_max_length=cfg.data.max_url_length)
    eval_type_to_dataset = {'exact': exact_eval_dataset}
    
    if cfg.eval.get('eval_perturb', True) and 'perturb' in eval_data.column_names:
        logger.info('Evaluating on perturbed data...')
        ## constructing eval dataset for each perturbation level
        perturb_types = [d['perturbation_type'] for d in eval_data.to_dict()['perturb'][0]]
        logger.info('Perturbation types: {}'.format(perturb_types))

        def _extract_perturbed_data(data, perturb_type):
            for d in data['perturb']:
                if d['perturbation_type'] == perturb_type:
                    return d

        for perturb_type in perturb_types:
            perturb_eval_data = eval_data.map(lambda x: {'url': x['url'], 'text': _extract_perturbed_data(x, perturb_type)['text']})
            perturb_eval_dataset = WikiURLDataset(perturb_eval_data, tokenizer=tokenizer, max_length=cfg.data.max_length, split='eval', url_max_length=cfg.data.max_url_length)
            eval_type_to_dataset[perturb_type] = perturb_eval_dataset

    ## loading data for ppl evaluation
    #ppl_eval = datasets.load_dataset(cfg.data.name, cfg.data.subset, split='{}[{}:{}]'.format(
    #                                                                                        'full' if 'wiki' in cfg.data.name else 'train', 
    #                                                                                          cfg.data.train_size, cfg.data.train_size + cfg.data.ppl_eval_size), cache_dir=cfg.#misc.cache_dir)=
    #ppl_eval_dataset = LMDataset(ppl_eval, tokenizer=tokenizer, max_length=cfg.data.max_length)

    ## check if special tokens were added to the tokenizers, resize model accordingly
    if tokenizer.additional_special_tokens:
        logger.info("Resizing model to accomodate new tokens...")
        model.resize_token_embeddings(len(tokenizer))
        ## initialize new tokens to eos_token embeddings 
        for sp_id in tokenizer.additional_special_tokens_ids:
            if 'gpt' in model_name:
                model.transformer.wte.weight.data[sp_id] = model.transformer.wte.weight.data[2638] ## token id for http 
            elif 'llama' in model_name:
                model.model.embed_tokens.weight.data[sp_id] = model.model.embed_tokens.weight.data[1732] ## token id for http
         
    if cfg.eval.get('constrained_decode', False):
        logger.info('Building URL trie...')
        url_ids = [] 
        for d in train_dataset.tokenized_dataset:
            ids = d['url_input_ids']
            ids = [tokenizer.additional_special_tokens_ids[0]] + [i for i in ids if i != tokenizer.pad_token_id] + [tokenizer.additional_special_tokens_ids[1]]
            url_ids.append(ids)
    
        url_trie = MarisaTrie(sequences=url_ids)
        logger.info('Done building URL trie.')

    # We use the language modeling data collator from Hugging Face which will handle preparing the inputs correctly
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Create the dataloaders
    train_sampler = dist.get_sampler(train_dataset.tokenized_dataset, shuffle=True)
    eval_samplers = {eval_type: dist.get_sampler(eval_dataset.tokenized_dataset, shuffle=False) for eval_type, eval_dataset in eval_type_to_dataset.items()}

    #ppl_eval_sampler = dist.get_sampler(ppl_eval_dataset.tokenized_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset.tokenized_dataset, batch_size=cfg.train.batch_size, collate_fn=collator, sampler=train_sampler)
    eval_dataloaders = {eval_type: DataLoader(eval_type_to_dataset[eval_type].tokenized_dataset, batch_size=cfg.eval.batch_size, collate_fn=eval_collator, sampler=eval_samplers[eval_type]) for eval_type in eval_type_to_dataset}
    
    urlem_metrics = [URLExactMatch(tokenizer=tokenizer, name='URLExactMatch-' + perturb_type) for perturb_type in sorted(eval_type_to_dataset.keys())]

    ### Word overlap metrics
    url_to_doc = {d['url']: d['text'] for d in train_data}
    wordoverlap_metrics = [URLContentSupportsWordOverlap(tokenizer=tokenizer, url_to_doc=url_to_doc, 
                                                 name='ContentWordOverlap-' + perturb_type) for perturb_type in sorted(eval_type_to_dataset.keys())]

    ## create nli model for content support
    nli_model = CrossEncoder('cross-encoder/nli-roberta-base')
    nli_metrics = [URLContentSupportsNLI(model=nli_model, tokenizer=tokenizer, url_to_doc=url_to_doc, 
                                                 name='ContentSupport-' + perturb_type) for perturb_type in sorted(eval_type_to_dataset.keys())]

    all_metrics = urlem_metrics + wordoverlap_metrics + nli_metrics
    # Package as a trainer-friendly Composer model
    composer_model = HuggingFaceModelWithURLLoss(model, tokenizer=tokenizer, use_logits=True, args=OmegaConf.to_container(cfg.train, resolve=True), eval_args=OmegaConf.to_container(cfg.eval, resolve=True), eval_metrics=all_metrics, url_trie=url_trie if cfg.eval.get('constrained_decode', False) else None)
    
    #optimizer = DecoupledAdamW(composer_model.parameters(), lr=cfg.train.lr, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=cfg.train.weight_decay)
    optimizer = DeepSpeedCPUAdam(composer_model.parameters(), lr=cfg.train.lr, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=cfg.train.weight_decay)
    lr_scheduler = LinearWithWarmupScheduler(t_warmup='{}ba'.format(cfg.train.warmup_steps), alpha_f=0.02)

    wandb_logger = WandBLogger(project=cfg.wandb.project,
                                init_kwargs={'config': OmegaConf.to_container(cfg, resolve=True)})
    evaluators = [Evaluator(label=eval_type,
                                dataloader=eval_dataloaders[eval_type],
                                metric_names=['URLExactMatch-' + eval_type, 'ContentWordOverlap-' + eval_type , 
                                              'ContentSupport-' + eval_type]) for eval_type in sorted(eval_type_to_dataset.keys())]
    
    ## load fsdp config if needed
    with open(cfg.train.fsdp_config) as f:
        fsdp_config = json.load(f)
    
    with open(cfg.train.deepspeed_config) as f:
        deepspeed_config = json.load(f)

    ## update deepspeed config
    deepspeed_config["gradient_accumulation_steps"] = cfg.train.batch_size // cfg.train.batch_size_per_device
    deepspeed_config["train_batch_size"] = cfg.train.batch_size * dist.get_world_size()
    deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.train.batch_size_per_device

    import ipdb; ipdb.set_trace()
    
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=evaluators,
        max_duration='{}ep'.format(cfg.train.num_epochs),
        save_folder=cfg.train.save_folder if 'debug' not in cfg.train.save_folder else None,
        save_overwrite='debug' in cfg.train.save_folder,
        optimizers=optimizer,
        schedulers=[lr_scheduler],
        precision='amp_bf16',
        seed=17,
        device_train_microbatch_size=cfg.train.batch_size_per_device,
        loggers=[wandb_logger],
        eval_interval='1ep',
        save_interval='20ba',
        save_filename="ep{epoch}_ba{batch}_{rank}.pt",
        deepspeed_config=deepspeed_config)

    # Start training
    trainer.fit()
    trainer.close()


if __name__ == '__main__':
    main()
