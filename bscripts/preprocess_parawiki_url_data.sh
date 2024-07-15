
N=1000

python pscripts/prepare_wiki_data.py --n 1000 --url_construction_method actual_date_random_tokens \
					--min_paragraph_length 50 \
					--url_max_length 40 \
					--tokenizer_path huggyllama/llama-7b \
					--val_size 0.05 \
					--output_path /net/nfs/allennlp/muhammadk/processed-data/parawiki/date_rand_tokens_1K/ \
					--build_trie \
