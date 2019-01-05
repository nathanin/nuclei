#!/bin/bash

set -e

disttype=(
    euclidean
    correlation
    cosine
    minkowski
    seuclidean
    sqeuclidean
)

for m in ${disttype[@]}; do
    python tile_features_clustergram.py --average --dst clustergram_mean_${m}.png --metric $m --reject_feats m0_m1_reject.npy
done
