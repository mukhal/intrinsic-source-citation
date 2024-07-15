#/bin/sh
set -e

WANDB_MODE=online composer -n4 pretrain.py --config-name config_100K.yaml train.loss.type=url_weighted train.loss.url_loss_factor=10.0 model.name=gpt2-xl train.save_folder='checkpoints/c4_url_100K_pretraining_gpt2-xl' train.batch_size=2 train.eval_batch_size=4 
