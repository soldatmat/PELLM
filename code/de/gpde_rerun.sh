#!/usr/bin/env bash

source ~/.bashrc

cd ~/master/code/de

for idx in {1..90};
do
    sbatch -p cpu --mem=6G -o data/gpde/PhoQ/sample/$idx.out gpde_alt_start_phoq.sh $idx
done
