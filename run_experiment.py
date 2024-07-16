import os, sys, time, random
import yaml
from omegaconf import OmegaConf as om
import subprocess
import coolname
from copy import deepcopy
import nltk 
nltk.download('punkt')

from datasets import disable_caching 
disable_caching()


def _generate_experiment_name() -> str:
    # change coolname randomness for different names with same seed
    coolname.replace_random(random.Random(os.urandom(128)))
    # prefixing with the time in a good format so experiments sorted alphabetically will have the latest experiment last
    generated_run_name = coolname.generate_slug(2) + '-' + str(time.strftime("%m%d%H%M"))
    run_name_list = [generated_run_name]
    generated_run_name = run_name_list[0]
    return generated_run_name

def download_and_save_data(cfg):
    data_save_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name, 'data/text')
    os.makedirs(data_save_dir, exist_ok=True)
    ## download pretraining data
    import datasets as hf_datasets
    pretrain_data = hf_datasets.load_dataset('mkhalifa/BioCite', 'pretrain')
    qa_data = hf_datasets.load_dataset('mkhalifa/BioCite', 'qa')

    ### save data
    os.makedirs(os.path.join(data_save_dir, 'qa'), exist_ok=True)
    
    pretrain_data.save_to_disk(data_save_dir)
    qa_data.save_to_disk(os.path.join(data_save_dir, 'qa'))
    cfg.data.text_data_path = data_save_dir


def preprocess_data(cfg):   
    #### download and save data if necessary
    ## check cfg.data.text_data_path is not a directory
    if not os.path.isdir(cfg.data.text_data_path) or cfg.data.text_data_path == 'biocite':
        print("Downloading and saving data from {}".format(cfg.data.text_data_path))
        download_and_save_data(cfg)

    # training configuration values
    url_location = cfg.train.url_location
    url_repeat = cfg.train.repeat_url_across_doc
    
    # data configuration values
    text_data_path = cfg.data.text_data_path

    if os.environ.get('DATA_DIR_PREFIX', None) is not None:
        text_data_path = os.path.join(os.environ['DATA_DIR_PREFIX'], text_data_path)
        cfg.data.text_data_path = text_data_path

    model_name = cfg.model.name

    # experiment configuration values
    experiment_name = cfg.experiment.get('name', None)
    
    if experiment_name is None:
        ## generate random experiment name
        experiment_name = _generate_experiment_name()
    
    # prepare paths
    experiment_dir = os.path.join(cfg.experiment.output_dir, experiment_name)
    out_data_dir = os.path.join(experiment_dir, 'data')
    out_stream_dir = os.path.join(out_data_dir, 'streaming/')

    # Check for tokenizer type
    if 'llama' in model_name.lower():
        n_tokens = 1024
        bos_token = "<s>"
        eos_token = "</s>"
    elif 'gpt2' in model_name:
        n_tokens = 1024
        bos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"
    else:
        raise ValueError(f"Model {model_name} name not recognized")

    if cfg.data.get('max_seq_len', None) is not None:
        n_tokens = cfg.data.max_seq_len


    print("Experiment name = {}".format(experiment_name))
    print("Experiment dir = {}".format(experiment_dir))
    print("Out data dir = {}".format(out_data_dir))


    doc_train_split = "train"
    new_text_data_path = text_data_path

    if getattr(cfg.train, 'pretrain', True):
        ##### data augmentation if necessary 
        if cfg.data.augment.doc.get('do', None):
            print("Applying doc-level augmentation...")

            cmd = [
                "python", "pscripts/doc_augment.py",
                "--data_path", os.path.join(text_data_path, "train"),
                "--out_path", os.path.join(out_data_dir, "train"),
                "--augment_type", cfg.data.augment.doc.method,
                "--seed", "42",
                "--n_sample_per_doc", str(cfg.data.augment.doc.n_sample_per_doc),
            ]

            return_code = subprocess.run(cmd)

            doc_train_split = "train"
            if return_code.returncode != 0:
                raise ValueError("Doc-level augmentation failed")
            
            new_text_data_path = out_data_dir ## set new_text_data_path to data augmentation path

        ### 1. processing raw documents 
        print("Processing pre-training documents...")
        cmd = [
            "python", "pscripts/create_url_streaming_dataset.py",
            "--dataset", "parawiki",
            "--dataset_path", new_text_data_path,
            "--out_root", out_stream_dir,
            "--splits", doc_train_split,
            "--tokenizer", model_name,
            "--concat_tokens", str(n_tokens),
            "--num_workers", "0",
            "--packing_method", 
            f"url_{url_location}" if url_location not in ["no_url", "standard"] else url_location,
            "--build_trie", 
            "--eos_text", eos_token,
        ]
        if url_repeat:
            cmd.append("--repeat_url_in_doc")
        
        if not getattr(cfg.train, 'reset_doc_pos_ids', False): #TODO fix
            cmd.append("--no_reset_doc_positions")


        print(" ".join(cmd))

        return_code = subprocess.run(cmd)
        if return_code.returncode != 0:
            raise ValueError("Pre-training data processing failed")
    
    else: 
        #### make sure a ckpt_dir was provided and extract it 
        if cfg.model.get('ckpt_dir', None):
            if not os.path.exists(os.path.join(cfg.model.ckpt_dir, 'pytorch_model.bin')):
                print("Extracting ckpt from {}".format(cfg.model.ckpt_dir))
                ###  call bscripts/extract_ckpt.sh to extract the ckpt
                return_code = subprocess.run([
                    "bash", "bscripts/extract_ckpt.sh",
                    cfg.model.ckpt_dir,
                ])
    
    qa_data_dir = "qa"
    ### 2. create fine-tuning dataset (if needed)
    if cfg.train.get('finetune_q_url_a', False):
        print("Processing fine-tuning <Q, URL, A> samples...")
        
        return_code = subprocess.run([
            "python", "pscripts/create_url_streaming_dataset.py",
            "--dataset", "parawiki",
            "--dataset_path", text_data_path,
            "--out_root", out_stream_dir,
            "--splits", os.path.join(qa_data_dir, "qa_train"),
            "--tokenizer", model_name,
            "--concat_tokens", "400",
            "--num_workers", "0",
            "--packing_method", "question_url_answer",
            "--predict_answer_only",
            "--bos_text", bos_token
        ])
        
        if return_code.returncode != 0:
            raise ValueError("Fine-tuning data processing failed")
        
    if cfg.train.get('finetune_q_a', False):
        print("Processing fine-tuning <Q, A> samples...")
        
        return_code = subprocess.run([
            "python", "pscripts/create_url_streaming_dataset.py",
            "--dataset", "parawiki",
            "--dataset_path", text_data_path,
            "--out_root", out_stream_dir,
            "--splits", os.path.join(qa_data_dir, "qa_train"),
            "--tokenizer", model_name,
            "--concat_tokens", "400",
            "--num_workers", "0",
            "--out_folder", os.path.join(qa_data_dir, "qa_attribution_train"),
            "--packing_method", "question_answer",
            "--bos_text", bos_token
        ])
        
        if return_code.returncode != 0:
            raise ValueError("Fine-tuning data processing failed")

    if cfg.train.get('finetune_q_a_url', False):
        n_negs = cfg.data.finetune.number_non_attributable_negatives
        print("Processing fine-tuning <Q, A, URL> samples...")

        cmd = [
            "python", "pscripts/create_url_streaming_dataset.py",
            "--dataset", "parawiki",
            "--dataset_path", text_data_path,
            "--out_root", out_stream_dir,
            "--splits", os.path.join(qa_data_dir, "qa_train"),
            "--concat_tokens", "400",
            "--tokenizer", model_name,
            "--num_workers", "0",
            "--packing_method", "question_answer_url",
            "--out_folder", os.path.join(qa_data_dir, "qa_attribution_train"),
            "--n_attribution_negs_per_question", str(n_negs), 
            "--neg_create_probability", str(cfg.data.finetune.neg_create_probability),
            "--bos_text", bos_token
        ]

        print(" ".join(cmd))

        return_code = subprocess.run(cmd)

        if return_code.returncode != 0:
            raise ValueError("Fine-tuning data processing failed")

        ## create URL trie for OOD docs
    if cfg.train.get('finetune_q_a_doc_url', False):
        assert url_location == "last", "URL location must be last for CoT setup"
        n_negs = cfg.data.finetune.number_non_attributable_negatives
        print("Processing fine-tuning <Q, A, Doc, URL> samples...")

        cmd = [
            "python", "pscripts/create_url_streaming_dataset.py",
            "--dataset", "parawiki",
            "--dataset_path", text_data_path,
            "--out_root", out_stream_dir,
            "--splits", os.path.join(qa_data_dir, "qa_train"),
            "--concat_tokens", str(n_tokens),
            "--tokenizer", model_name,
            "--num_workers", "0",
            "--packing_method", "question_answer_doc_url",
            "--out_folder", os.path.join(qa_data_dir, "qa_attribution_train"),
            "--n_attribution_negs_per_question", str(n_negs), 
            "--neg_create_probability", str(cfg.data.finetune.neg_create_probability),
            "--bos_text", bos_token
        ]

        print(" ".join(cmd))

        ## popen
        return_code = subprocess.run(cmd)

        if return_code.returncode != 0:
            raise ValueError("Fine-tuning data processing failed")
    

    return_code = subprocess.run([
        "python", "pscripts/filter_outdist_urls.py",
        "--text_data_path", text_data_path,
        "--in_domain_qa_data_path", os.path.join(text_data_path, qa_data_dir, "qa_train"),
        "--tokenizer", os.path.join(out_stream_dir, "tokenizer"),
        "--out_dir", out_stream_dir,
    ])

    if return_code.returncode != 0:
        raise ValueError("Fine-tuning data processing failed")
        
    ### save experiment config to experiment dir
    with open(os.path.join(experiment_dir, 'experiment_config.yaml'), 'w') as f:
        yaml.dump(om.to_container(cfg, resolve=True), f)

    extra_config = {
        'experiment_name': experiment_name,
        'experiment_dir': experiment_dir,
        'out_stream_dir': out_stream_dir,
        'doc_train_split': doc_train_split,
        'qa_data_dir': qa_data_dir,
    }
    return extra_config
    

def prepare_train_config(cfg, paths_info):
    ## load config template 
    with open(cfg.train.config_template_path) as f:
        train_cfg = om.load(f)
    
    ## update template config with experiment config
    ### 1. update data paths and other configs
    train_cfg.text_data_path = cfg.data.text_data_path
    train_cfg.streaming = paths_info['out_stream_dir']
    train_cfg.run_name = paths_info['experiment_name']
    train_cfg.max_seq_len = 2048 if 'llama' in cfg.model.name.lower() else 1024
    train_cfg.url_trie = os.path.join(paths_info['out_stream_dir'], 'url_trie.pkl')
    if cfg.data.get('use_ood_url_trie', True):
        odd_trie = 'unseen_url_trie.pkl'
    else:
        odd_trie = 'url_trie.pkl'
    train_cfg.ood_url_trie = os.path.join(paths_info['out_stream_dir'], odd_trie)
    train_cfg.save_folder = os.path.join(paths_info['experiment_dir'], 'checkpoints')
    train_cfg.model.pretrained_model_name_or_path = cfg.model.name
    
    ### 2. update model/train configs
    train_cfg.cross_doc_attention = cfg.train.cross_doc_attention
    train_cfg.model.loss.url_loss_factor = cfg.train.url_loss_factor
    train_cfg.model.loss.type = cfg.train.loss_type

    ### 3. dataloaders! 
    ### a.  main doc pre-training dataloader
    train_cfg.dataloaders[0].dataset.split = paths_info['doc_train_split']

    ### b. update paths for the eval dataloaders 
    for dl in train_cfg.dataloaders[1:]:
        dl.dataset.path = os.path.join(cfg.data.text_data_path, paths_info['qa_data_dir'])

    ### c.  fine-tuning dataloaders
    dataloaders_to_add = []

    if cfg.train.finetune_q_url_a:
        q_url_a_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        q_url_a_dataloader_cfg.name = "train_q_url_a"
        q_url_a_dataloader_cfg.dataset.local = os.path.join(paths_info['out_stream_dir'], paths_info['qa_data_dir'])
        q_url_a_dataloader_cfg.dataset.split = "qa_train"
        q_url_a_dataloader_cfg.dataset.batch_type = "fact"
        ## add it 
        dataloaders_to_add.append(q_url_a_dataloader_cfg)
    
    if cfg.train.get('finetune_q_a_url', False) or cfg.train.get('finetune_q_a', False) or cfg.train.get('finetune_q_a_doc_url', False):
        q_a_url_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        q_a_url_dataloader_cfg.name = "train_q_a_url"
        q_a_url_dataloader_cfg.dataset.local = os.path.join(paths_info['out_stream_dir'], paths_info['qa_data_dir'])
        q_a_url_dataloader_cfg.dataset.split = "qa_attribution_train"
        q_a_url_dataloader_cfg.dataset.batch_type = "fact"
        ## add it 
        dataloaders_to_add.append(q_a_url_dataloader_cfg)
    
    if cfg.train.get('finetune_q_a_doc_url', False):
        for loader in train_cfg.dataloaders:
            if hasattr(loader.dataset, 'batch_type') and "qa" in loader.dataset.batch_type:
                loader.dataset.batch_type = loader.dataset.batch_type.replace("qa", "qa-cot")

    train_cfg.dataloaders.extend(dataloaders_to_add)

    ## check if no pretrain dataloader is needed
    if not cfg.train.get('pretrain', True) and cfg.model.get('ckpt_dir', None):
        train_cfg.dataloaders = [dl for dl in train_cfg.dataloaders if 'train_loader_docs' not in dl.name]
        train_cfg.model.checkpoint = os.path.join(cfg.model.ckpt_dir, 'pytorch_model.bin')

        ### copy url_trie locations from pretraining experiment dir
        train_cfg.url_trie = cfg.data.url_trie
        train_cfg.ood_url_trie = cfg.data.ood_url_trie

    if cfg.eval.disable_qa_eval:
        train_cfg.dataloaders = [dl for dl in train_cfg.dataloaders if 'answer_eval' not in dl.name]

    if cfg.eval.get('icl_eval', False):
        #### add dataloader for in-context learning eval
        print("Adding ICL eval dataloaders...")
        icl_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        icl_dataloader_cfg.dataset.batch_type = "ictx"
        icl_dataloader_cfg.name = "ictx_eval_triviaqa"
        icl_dataloader_cfg.dataset.split = "validation"
        icl_dataloader_cfg.dataset.name = 'lucadiliello/triviaqa'
        icl_dataloader_cfg.dataset.n_demos = 8

        #### add it
        train_cfg.dataloaders.append(icl_dataloader_cfg)

        ### another one for boolq 
        icl_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        icl_dataloader_cfg.dataset.batch_type = "ictx"
        icl_dataloader_cfg.name = "ictx_eval_boolq"
        icl_dataloader_cfg.dataset.split = "validation"
        icl_dataloader_cfg.dataset.name = 'boolq'
        icl_dataloader_cfg.dataset.n_demos = 8
        #### add it
        train_cfg.dataloaders.append(icl_dataloader_cfg)

        ### another one for naturalquestionsshortqa
        icl_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        icl_dataloader_cfg.dataset.batch_type = "ictx"
        icl_dataloader_cfg.name = "ictx_eval_natural_questions"
        icl_dataloader_cfg.dataset.split = "validation"
        icl_dataloader_cfg.dataset.name = 'lucadiliello/naturalquestionsshortqa'
        icl_dataloader_cfg.dataset.n_demos = 8
        #### add it
        train_cfg.dataloaders.append(icl_dataloader_cfg)


    if cfg.eval.get('ppl_eval', False):
        print("Adding PPL eval dataloaders...")
        ppl_dataloader_cfg = deepcopy(train_cfg.dataloaders[0])
        ppl_dataloader_cfg.dataset.batch_type = "lm"
        ppl_dataloader_cfg.name = "wikitext_ppl_eval"
        ppl_dataloader_cfg.dataset.split = "validation"
        ppl_dataloader_cfg.dataset.name = 'wikitext'
        ppl_dataloader_cfg.dataset.split = 'test'
        
        ## remove ppl_dataloader_cfg.local 
        if hasattr(ppl_dataloader_cfg.dataset, 'local'):
            del ppl_dataloader_cfg.dataset.local
        #### add it
        train_cfg.dataloaders.append(ppl_dataloader_cfg)


    if cfg.eval.disable_all_eval:
        train_cfg.dataloaders = [dl for dl in train_cfg.dataloaders if 'train' in dl.name]
        train_cfg.eval_interval = 1

    if hasattr(cfg.train, 'device_train_microbatch_size'):
        train_cfg.device_train_microbatch_size = cfg.train.device_train_microbatch_size

    if hasattr(cfg.train, 'device_eval_batch_size'):
        train_cfg.device_eval_batch_size = cfg.train.device_eval_batch_size

    ## do the same with eval_interval and eval_first 
    train_cfg_attrs = ['eval_interval', 'eval_first', 'max_duration', 'save_folder']

    for attr in train_cfg_attrs:
        if hasattr(cfg.train, attr):
            setattr(train_cfg, attr, getattr(cfg.train, attr))

    optimizer_attrs = ['lr', 'weight_decay']
    for attr in optimizer_attrs:
        if hasattr(cfg.train, attr):
            setattr(train_cfg.optimizer, attr, getattr(cfg.train, attr))

    ### whether to use AIS evaluation. 
    if cfg.eval.get('use_ais', False):
        setattr(train_cfg, 'use_ais', True)

    ## resolve experiment config then copy to train config
    train_cfg.experiment = om.to_container(cfg, resolve=True)
            
    ### save the config to yaml file
    train_cfg_path = os.path.join(paths_info['experiment_dir'], 'train_config.yaml')
    with open(train_cfg_path, 'w') as f:
        yaml.dump(om.to_container(train_cfg, resolve=True), f)

    return train_cfg, train_cfg_path


def prepare_eval_config(cfg, train_cfg):
    attribution_eval_loaders = [dl for dl in train_cfg.dataloaders if 'qa-ood' in dl.dataset.batch_type]
    ### delete train dataloaders
    train_cfg.dataloaders = attribution_eval_loaders

    ##### extract ckpt from cfg.experiment.dir/checkpoints if needed
    possible_model_checkpoint = os.path.join(cfg.experiment.dir, 'checkpoints', 'pytorch_model.bin')

    if not os.path.exists(possible_model_checkpoint):
        print("Extracting ckpt from {}".format(cfg.experiment.dir))
        ###  call bscripts/extract_ckpt.sh to extract the ckpt
        return_code = subprocess.run([
            "bash", "bscripts/extract_ckpt.sh",
            os.path.join(cfg.experiment.dir, 'checkpoints'),
        ])
    
        if return_code.returncode != 0:
            raise ValueError("Extracting ckpt failed")
    
    train_cfg.model.checkpoint = possible_model_checkpoint
    ### experiment name last part in experiment dir
    train_cfg.run_name = cfg.experiment.dir.split('/')[-1]
    train_cfg.eval_first = True
    train_cfg.max_duration = '0ep'
    
    eval_cfg_path = os.path.join(cfg.experiment.dir, 'eval_config.yaml')
    with open(eval_cfg_path, 'w') as f:
        yaml.dump(om.to_container(train_cfg, resolve=True), f)

    return train_cfg, eval_cfg_path


def main(cfg):
    if cfg.get('train', None) and cfg.train.get('sequential', False):
        ### pretrain then finetune. 

        print("Pretraining...")
        ### turn off finetuning 
        finetuning_vars = ['finetune_q_url_a', 'finetune_q_a_url', 'finetune_q_a', 'finetune_q_a_doc_url']
        finetuning_type_var = None
        for var in finetuning_vars:
            if getattr(cfg.train, var, False):
                finetuning_type_var = var
                ## store the original value
            setattr(cfg.train, var, False)


        cfg.experiment.name = cfg.experiment.name + '_pretrain'
        paths_info = preprocess_data(cfg)
        print("Instantiating training config...")
        print(cfg)

        train_cfg, train_cfg_path = prepare_train_config(cfg, paths_info)    
        
        if cfg.train.pretrain:
            print("Launching training script...")
            #launch training script 
            return_code = subprocess.run([
                "composer", "train.py",
                train_cfg_path
            ])

            if return_code.returncode != 0:
                ## exit 
                print("Pretraining failed!")
                sys.exit(1) # exit with error code
        
        ############## FINETUNING ########
        cfg.data.url_trie = os.path.join(paths_info['out_stream_dir'], 'url_trie.pkl')
        cfg.data.ood_url_trie = os.path.join(paths_info['out_stream_dir'], 'unseen_url_trie.pkl')
        cfg.model.ckpt_dir = os.path.join(paths_info['experiment_dir'], 'checkpoints')
        cfg.train.pretrain = False
        cfg.experiment.name = cfg.experiment.name.replace('_pretrain', '_finetune')
        cfg.train.lr = 1.0e-5
        cfg.train.max_duration = '3ep'
        cfg.train.device_train_microbatch_size = (2 if finetuning_type_var == 'finetune_q_a_doc_url' else 4) * cfg.train.device_train_microbatch_size
        ### set finetuning type to True
        setattr(cfg.train, finetuning_type_var, True)
        
        paths_info = preprocess_data(cfg)

        print("Instantiating training config...")
        train_cfg, train_cfg_path = prepare_train_config(cfg, paths_info)

        print("Launching finetuning script...")
        # launch training script
        return_code = subprocess.run([
            "composer", "train.py",
            train_cfg_path
        ])

    elif cfg.get('train', None) is not None:
        ### mixture training or pretraining/finetuning only
        paths_info = preprocess_data(cfg)
        print("Instantiating training config...")
        train_cfg, train_cfg_path = prepare_train_config(cfg, paths_info)    
    
        print("Launching training script...")
        # launch training script 
        return_code = subprocess.run([
            "composer", "train.py",
            train_cfg_path
        ])
    
    else:
        ### evaluation only 
        print("Evaluation only...")
        ## load experiment train config
        train_cfg_path = os.path.join(cfg.experiment.dir, 'train_config.yaml') 
        ##### add different eval dataloaders 
        ## find qa attribution loader in train config
        with open(train_cfg_path) as f:
            train_cfg = om.load(f)

        train_cfg, eval_cfg_path = prepare_eval_config(cfg, train_cfg)
        
        return_code = subprocess.run([
            "composer", "-n1", "train.py",
            eval_cfg_path
        ]) 

        if return_code.returncode != 0:
            raise ValueError("Evaluation failed")
        
        
                    
if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
