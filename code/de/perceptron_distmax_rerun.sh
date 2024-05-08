#!/usr/bin/env bash

source ~/.bashrc

cd ~/master/code/de

#for idx in 120 146 160 166 169;
for idx in {1..100};
do
    sbatch -p gpufast --mem=20G --gres=gpu:1 -o data/perceptron/GB1/sample/$idx.out perceptron_distmax_alt_start.sh $idx
done
