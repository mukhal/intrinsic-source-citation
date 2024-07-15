
WANDB_MODE=online python pretrain.py --config-name conf_wiki data.train_size=16 data.eval_size=16 train.batch_size=8 model.name=gpt2 train.num_warmup_steps=0 train.lr=0.001 train.num_epochs=50
