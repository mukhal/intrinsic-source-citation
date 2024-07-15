#!/bin/sh 
set -e
FACT=10000

OUTDIR="../data/bio-attribution/last_name_permute_augment/"
OUTTEXT="$OUTDIR/text"
OUTSTREAM="$OUTDIR/streaming/gpt2-url-first-answer-prediction-repeat-url"
TOKENIZER=gpt2
#huggyllama/llama-7b

if [[ $TOKENIZER == *llama* ]]; then
	    N_TOKENS=2048
    else
	    N_TOKENS=1024
fi

echo "N Tokens = $N_TOKENS"
python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path $OUTTEXT --out_root $OUTSTREAM --splits train --tokenizer $TOKENIZER --concat_tokens $N_TOKENS --num_workers 0 --packing_method url_first --build_trie --repeat_url_in_doc 

python pscripts/create_url_streaming_dataset.py --dataset parawiki --dataset_path "$OUTDIR/text"  --out_root $OUTSTREAM --splits qa/$FACT/qa_train --tokenizer $TOKENIZER --concat_tokens 250 --num_workers 0 --packing_method question_url_answer --predict_answer_only


