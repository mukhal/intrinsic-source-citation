#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/rand_id_debug_llama
OUTPATH=$(basename $OUTDIR)

#python pscripts/prepare_wiki_data.py --n 100 --url_construction_method page_id_para_id --output_path $OUTDIR --tokenizer_path huggyllama/llama-7b --build_trie --url_val_size 1.0 --ppl_val_size 0.1


python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/${OUTPATH}_no_packing --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 512  --num_workers 1



