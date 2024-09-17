#!/usr/bin/env bash

for idx in {1..200};
do
    sbatch -p cpu --mem=10G -o data/boes/GB1/sample/$idx.out gpde.sh $idx
done
