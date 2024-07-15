#!/bin/sh 
set -e
url_method=page_title
FACT=100000

OUTDIR="../processed-data/ccnews/${url_method}_100K"
OUTTEXT="$OUTDIR/text"
OUTSTREAM="$OUTDIR/streaming/gpt2"
TOKENIZER=gpt2-large

#python pscripts/prepare_news_data.py --url_construction_method page_title --num_workers 8 --min_paragraph_length 30 --output_path $OUTTEXT --mode page --trim_to_length 512 --build_trie

#python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTTEXT --out_root $OUTSTREAM --splits train --tokenizer $TOKENIZER --concat_tokens 1024 --num_workers 0 --packing_method url_first --build_trie

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits qa/$FACT/qa_train --tokenizer $TOKENIZER --concat_tokens 256 --num_workers 0 --packing_method question_url_answer

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits qa/$FACT/qa_url_val_seen qa/$FACT/qa_url_val_unseen --tokenizer $TOKENIZER --concat_tokens 256 --num_workers 0 --truncate_num_samples 30000

echo "copying url_trie.pkl to streaming data location"
cp $OUTDIR/text/url_trie.pkl $OUTDIR

