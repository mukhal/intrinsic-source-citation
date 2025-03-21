# Source-Aware Training Enables Knowledge Attribution in Language Models

We explore **Source-aware Training** to enable LLMs to cite their pretraining data. Source-aware training involves (i) training the LLM to associate unique source document identifiers with the knowledge in each document, followed by (ii) an instruction-tuning to teach the LLM to cite a supporting pretraining source when prompted. We demonstrate that our training recipe can enable faithful attribution to the pretraining data without a substantial impact on the model's quality compared to standard pretraining. Our results also highlight the importance of data augmentation in achieving attribution.

<p align="center">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/9f4d582e-5b92-4715-88ab-97d20f82ee04" alt="image" width="450" height="220">
</p>

📝 **Paper**: [https://arxiv.org/abs/2404.01019](https://arxiv.org/abs/2404.01019)

🤗 **Data**: [https://huggingface.co/datasets/mkhalifa/BioCite](https://huggingface.co/datasets/mkhalifa/BioCite)


### Getting Started
To set up the code and run source-aware training, you will first need to set up the environment. Our code is based on the [llm-foundry](https://github.com/mosaicml/llm-foundry) package by mosaicml. Let's go through the setup step-by-step. 

We recommend using conda to set up the environment:
```python
conda create --name=citation-training python=3.10
conda activate citation-training
```
Now you need to install `torch==2.0.1` which is the version with which the paper experiments were done. You can get it from [here](https://pytorch.org/get-started/previous-versions/).

Then you need to install dependencies via:
```
pip install -r requirements.txt
```

### Downloading Data
Our experiments are done over BioCite a synthetic corpus of biographies about fictitious individuals. Each document in BioCite is constructed by sampling multiple facts from different biographies. Each document ID is constructed as a concatenation of 3-letter prefix of each last name in the document. 
<p align="left">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/86beaa3f-088a-4f21-bed5-de2dfa319e5e" alt="image" width="400" height="150">
</p>

BioCite is available on huggingface [here](https://huggingface.co/datasets/mkhalifa/BioCite)

### Running Experiments

#### One-script-for-all
To eliminate the need to run many consecutive scripts, I designed the code such that a single script will do everything. Specifically, `run_experiment.py` will take as input a configuration file (more on that later) and will: 
1. Perform data augmentation if necessary (by shuffling facts within the document as described in the paper)
2. Preprocess the pretraining data by injecting Doc IDs (referred to as **URL** throughought the code) into the pretraining data as per the passed config
3. Preprocess and tokenize the instruction tuning comprised of <Question, Answer, Doc ID> triplets
4. Builds the Doc ID Trie needed for constrained decoding when predicting doc IDs. 
5. Save all tokenized data to specified experiment folder in numpy `.npz` format.
6. Run pretraining using next-word objective on the documents with injected doc IDs
7. After pretraining finishes, loads the last checkpoint and does instruction tuning.
8. Logs all evals to W&B

#### Example Config
Here is an example of a config file and I'll explain the relevant parameter. Configs for paper experiments can be found [here](conf):

```yaml

experiment:
  name: my-experiment
  output_dir: path-to-experiments-folder

data:
  text_data_path: path-to-training-corpus
  augment: # Data augmentation parameters
    doc:
      do: true
      method: permute
      n_sample_per_doc: 2 # Number of augmentations per document. This means that each document will exist 1 + 2 = 3 times with different sentence permutations

model:
  name: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

train:
  url_location: last # inject doc ID at the end. Options are last, first, no_url
  pretrain: true # whether to run next-word-prediction
  sequential: true, # whether to do pretraining then finetuning. If false, will only do pretraining
  repeat_url_across_doc: false # whether to repeat doc ID across the document.
  finetune_q_a_url: true # finetuning will take question as input, and predict answer then doc ID. 
  finetune_q_a_doc_url: false # set this to true for the CoT setup

  ## loss and attn config 
  cross_doc_attention: false # whether to apply cross-doc attention across documents. It is best to disable this. 
  url_loss_factor: 1.0 # coefficient to multiply the loss on the doc ID tokens by. Doesn't make much difference
  loss_type: mask # always mask, do not change this
  config_template_path: conf/templates/train_config.yaml
  device_eval_batch_size: 40
  device_train_microbatch_size: 2
  eval_first: false # whether to run evaluation first before training
  weight_decay: 0.02
  lr: 8.0e-5
  max_duration: 10ep # train for 10 epochs
  save_folder: null

eval:  
  disable_qa_eval: false
  disable_all_eval: false
  disable_attribution_eval: false
  disable_non_attrib_eval: true # leave this to true, non-attrib eval was not used in the paper. 
  icl_eval: false # whether to run ICL evaluation using some tasks. 
  ppl_eval: true # whether to evaluate the model using perplexity on wikitext as done in the paper
```


After you've set up your config file.

#### Launch your own experiment
To launch an experiment on a 1K-doc subset of BioCite, you can use the config file `conf/doc-id-repeat-1k-docs.yaml` by running:
`python run_experiment.py conf/doc-id-repeat-1k-docs.yaml`. This subset is located in `sample-data/biocite-1k`. 

There are config file correponding to each different setup used in the paper in `conf/`. 

### Distributed Training
By default, the code uses ZeRO implementation by [Deepspeed](https://github.com/microsoft/DeepSpeed) for distributed training. The deepseed parameters are defined in `conf/templates/train_config.yaml`. The defaul parameters are 
```yaml
deepspeed_config:
  bf16:
    enabled: true
  train_batch_size: ${global_train_batch_size}
  zero_optimization:
    stage: 3
    contiguous_gradients: true
    reduce_bucket_size: true
    overlap_comm: true
    allgather_bucket_size: 2e8
    reduce_scatter: true
    offload_optimizer:
      device: cpu
      pin_memory: true
```

### Citation
If you find our work/code useful please cite it:
```yaml
@inproceedings{
sourceaware2024,
title={Source-Aware Training Enables Knowledge Attribution in Language Models},
author={Khalifa, Muhammad and Wadden, David and Strubell, Emma and Lee, Honglak and Wang, Lu and Beltagy, Iz and Peng, Hao},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=UPyWLwciYz}
}
```
