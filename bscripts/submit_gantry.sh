#/bin/sh 
set -e 
wandb_key=6664bc156146fd5d94ebcf35bd5d4d4cc37dff78

gantry run  -y --cluster 'ai2/*-cirrascale'  \
			--gpus 1 \
		       	--venv /net/nfs.cirrascale/allennlp/muhammadk/env/llmfoundry/ \
			--env WANDB_API_KEY=$wandb_key \
			-- composer -n1 train.py  conf/llmfoundry/gpt2-small.yaml

