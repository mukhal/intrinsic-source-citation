#!/bin/sh
set -e

ckpt_dir=$1
cd $ckpt_dir

for file in latest-rank*; do
    tar -xvf $file
done

echo "converting from zero to fp32"
python zero_to_fp32.py . pytorch_model.bin

echo "cleaning up..."
rm -rf deepspeed
