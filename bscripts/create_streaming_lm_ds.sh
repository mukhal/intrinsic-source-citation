#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/page_title_para_id

python pscripts/create_lm_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/lm-only/parawiki/page_title_para_id/ --splits train ppl_val --tokenizer EleutherAI/gpt-neox-20b  --concat_tokens 2048 --bos_text='<|endoftext|>'



