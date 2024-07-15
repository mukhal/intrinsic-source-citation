#!/bin/sh 
set -e
url_method='random_tokens_date'

OUTDIR=../processed-data/wikipedia/${url_method}_llama_20K_url_last_full_doc_vanilla_wiki
OUTSTREAM="$OUTDIR/streaming/"

#python pscripts/prepare_vanilla_wiki_data.py --n 100000 --url_construction_method $url_method  --output_path $OUTDIR/text --tokenizer_path huggyllama/llama-7b --build_trie --url_val_size .05 --ppl_val_size 0.1 --shuffle --mode page --num_workers 8 --min_paragraph_length 100 --trim_to_length 1950

#python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits train ppl_val url_val --tokenizer $OUTDIR/text/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method url_last --no_reset_doc_positions

#python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits train url_val ppl_val --tokenizer $OUTDIR/text/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method url_last 
#--no_reset_doc_positions

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits facts/10000/fact_train facts/10000/fact_url_val_seen facts/10000/fact_url_val_unseen --tokenizer $OUTDIR/text/tokenizer/ --concat_tokens 512 --num_workers 0 --packing_method url_last 
#--no_reset_doc_positions


echo "copying url_trie.pkl to streaming data location"
cp $OUTDIR/text/url_trie.pkl $OUTDIR




