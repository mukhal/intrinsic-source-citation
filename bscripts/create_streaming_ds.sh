#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/page_title_para_id_debug
OUTPATH=$(basename $OUTDIR)

#python pscripts/prepare_wiki_data.py --n 100 --url_construction_method page_title_para_id --output_path $OUTDIR --tokenizer_path gpt2 --build_trie --url_val_size 1.0 --ppl_val_size 0.1 --shuffle


python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/$OUTPATH --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 512 --bos_text='<|endoftext|>' --num_workers 1



