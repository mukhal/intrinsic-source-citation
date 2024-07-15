#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/date_tokens_debug_url_last
OUTPATH=$(basename $OUTDIR)

python pscripts/prepare_wiki_data.py --n 100 --url_construction_method actual_date_random_tokens --output_path $OUTDIR --tokenizer_path gpt2 --build_trie --url_val_size 1.0 --ppl_val_size 0.1 --shuffle

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/${OUTPATH}_no_reset_doc_pos --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 512 --bos_text='<|endoftext|>' --num_workers 0 --packing_method url_last --no_reset_doc_positions



