#/bin/sh
set -e

#python pretrain.py train.loss.type=url model.name=gpt2-medium 
python pretrain.py train.loss.type=url_weighted train.loss.url_loss_factor=20.0 eval.num_beams=3 model.name=gpt2-medium eval.batch_size=10
