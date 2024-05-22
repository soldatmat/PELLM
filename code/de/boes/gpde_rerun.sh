#!/usr/bin/env bash

source ~/.bashrc

cd ~/master/code/de/boes

#for idx in 120 146 160 166 169;
for idx in {171..200};
do
    sbatch -p cpu --mem=10G -o data/gpde/PhoQ/sample/$idx.out gpde_alt_start_phoq.sh $idx
done
