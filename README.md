## Source-Aware Training Enables Knowledge Attribution in Language Models

📝 [paper](https://arxiv.org/abs/2404.01019) 🤗 [Data(BioCite)](https://huggingface.co/datasets/mkhalifa/BioCite/tree/main/qa)

We explore **Source-aware Training** to enable LLMs to cite their pretraining data. Source-aware training involves (i) training the LLM to associate unique source document identifiers with the knowledge in each document, followed by (ii) an instruction-tuning to teach the LLM to cite a supporting pretraining source when prompted. We demonstrate that our training recipe can enable faithful attribution to the pretraining data without a substantial impact on the model's quality compared to standard pretraining. Our results also highlight the importance of data augmentation in achieving attribution.

<p align="center">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/9f4d582e-5b92-4715-88ab-97d20f82ee04" alt="image" width="500" height="250">
</p>

## Synthnetic Pretraining
We conduct our experiments on **BioCite** a synthetic corpus of fake biography information. Each document in BioCite is constructed by sampling multiple facts from different biographies. Each document ID is constructed as a concatenation of 3-letter prefix of each last name in the document. 
<p align="center">
<img src="https://github.com/mukhal/intrinsic-source-citation/assets/5109053/86beaa3f-088a-4f21-bed5-de2dfa319e5e" alt="image" width="600" height="230">
</p>

You can access our synthetic pretraining dataset on 🤗 [here](https://huggingface.co/datasets/mkhalifa/BioCite/tree/main/qa)


## Getting Started
To set up the code and run source-aware training, you will first need to set up the environment. Our code is based on the [llm-foundry](https://github.com/mosaicml/llm-foundry) package by mosaicml. Let's go through the setup step-by-step. 

We recommend using conda to set up the environment:
```python
conda create --name=citation-training python=3.10
conda activate source-training
```
Now you need to install `torch== 2.0.1` which is the version with which the paper experiments were done. You can get it from [here](https://pytorch.org/get-started/previous-versions/).

## Downloading Data
TODO

## Running Experiments
To eliminate the need to run many consecutive scripts, I designed the code such that a single script will do everything. Specifically, `run_experiment.py` will take as input a configuration file (more on that later) and will: 
1. Perform data augmentation if necessary (by shuffling facts within the document as described in the paper)
2. Preprocess the pretraining data by injecting Doc IDs (referred to as **URL** throughought the code) into the pretraining data as per the passed config
3. Preprocess and tokenize the instruction tuning comprised of <Question, Answer, Doc ID> triplets
4. Save all tokenized data to specified experiment folder in numpy `.npz` format.
5. Run pretraining using next-word objective on the documents with injected doc IDs
6. After pretraining finishes, loads the last checkpoint and does instruction tuning.
7. Logs all evals to W&B

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
  max_duration: 10ep # 10 epochs
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
You can run the whole pipeline using 
`python run_experiment.py conf/my-conf.yaml`

## More details
We use Deepspeed for distributed training. The deepseed parameters are in `conf/templates/train_config.yaml`. 
