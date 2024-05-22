#!/usr/bin/env bash

source ~/.bashrc

cd ~/master/code/de/perceptron

#for idx in 120 146 160 166 169;
for idx in {151..200};
do
    sbatch -p gpufast --mem=20G --gres=gpu:1 -o data/perceptron/PhoQ/sample/$idx.out perceptron_distmax_alt_start_phoq.sh $idx
done
