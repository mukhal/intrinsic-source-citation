#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/nq_based/date_tokens_llama_200K_url_last_pids_fixed
OUTPATH=$(basename $OUTDIR)

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/nq_based/${OUTPATH}_no_url --splits train ppl_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method standard --no_reset_doc_positions --no_url



