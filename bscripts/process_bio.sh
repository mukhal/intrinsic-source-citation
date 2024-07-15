#!/bin/sh 
set -e
FACT=10000

OUTDIR="../data/bio-attribution/last_name_upsample_unseen_all_questions/"
OUTTEXT="$OUTDIR/text"
OUTSTREAM="$OUTDIR/streaming/llama-url-first"
TOKENIZER=huggyllama/llama-7b

if [[ $TOKENIZER == *llama* ]]; then
	    N_TOKENS=2048
    else
	    N_TOKENS=1024
fi

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTTEXT --out_root $OUTSTREAM --splits train --tokenizer $TOKENIZER --concat_tokens $N_TOKENS --num_workers 0 --packing_method url_first --build_trie

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits qa/$FACT/qa_train --tokenizer $TOKENIZER --concat_tokens 250 --num_workers 0 --packing_method question_url_answer

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits qa/$FACT/qa_url_val_seen qa/$FACT/qa_url_val_unseen --tokenizer $TOKENIZER --concat_tokens 250 --num_workers 0 --truncate_num_samples 30000

