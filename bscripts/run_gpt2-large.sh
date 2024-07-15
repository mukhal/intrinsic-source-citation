#/bin/sh
set -e

#python pretrain.py train.loss.type=url model.name=gpt2-medium 
composer -n4 pretrain.py train.loss.type=url_weighted train.loss.url_loss_factor=10.0 model.name=gpt2-large train.batch_size=8 train.eval_batch_size=16
