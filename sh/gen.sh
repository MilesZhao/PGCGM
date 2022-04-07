#!/bin/bash

n_samples=10000
file='frac12'
python create_cif.py \
    --gpu_id 1 \
    --savedir "$file"\
    --matdir "ternary_gen_cifs/" \
    --model "generator_weights.pth"\
    --n_samples $n_samples \
    --latent_dim 128 \
    --no-verbose

python pymatgen_valid.py

python -W ignore merge_valid.py \
    --merge_ratio 0.6 \
    --dist_ratio 0.75 \
    --further_merge_ratio 1.5

