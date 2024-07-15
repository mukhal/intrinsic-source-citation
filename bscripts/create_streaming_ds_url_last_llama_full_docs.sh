#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/date_tokens_llama_100K_url_last_full_doc
OUTPATH=$(basename $OUTDIR)

python pscripts/prepare_wiki_data.py --n 100000 --url_construction_method actual_date_random_tokens --output_path $OUTDIR --tokenizer_path huggyllama/llama-7b --build_trie --url_val_size .05 --ppl_val_size 0.1 --shuffle --mode page --num_workers 16 --min_paragraph_length 100

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/fulldoc/date_tokens_llama_100K_full_doc_no_url/ --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method standard --no_reset_doc_positions --no_url



