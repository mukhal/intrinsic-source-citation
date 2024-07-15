#!/bin/sh 
set -e

OUTDIR=../processed-data/wikipedia/date_tokens_llama_1K_url_last
OUTPATH=$(basename $OUTDIR)

python pscripts/prepare_wiki_data.py --n 1000 --url_construction_method actual_date_random_tokens --output_path $OUTDIR --tokenizer_path huggyllama/llama-7b --build_trie --url_val_size .5 --ppl_val_size 0.1 --shuffle

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTDIR  --out_root ../streaming-data/parawiki/${OUTPATH} --splits train ppl_val url_val --tokenizer $OUTDIR/tokenizer/ --concat_tokens 2048 --num_workers 0 --packing_method url_last --no_reset_doc_positions

##### copy url_trie to streaming dir 
echo "copying URL trie to streaming data location..."
cp $OUTDIR/url_trie.pkl ../streaming-data/parawiki/${OUTPATH}


