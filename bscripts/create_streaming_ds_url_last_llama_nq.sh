#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/nq_based/date_tokens_llama_1K_url_last_pids_fixed
OUTPATH=$(basename $OUTDIR)

python pscripts/prepare_wiki_data_for_nq.py --n 1000 --url_construction_method actual_date_random_tokens --output_path $OUTDIR --tokenizer_path huggyllama/llama-7b --build_trie --url_val_size .05 --ppl_val_size 0.1 --shuffle
#--datasets_to_include ../data/nq/ ../data/tqa

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/nq_based/${OUTPATH} --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method url_last --no_reset_doc_positions



