#!/bin/bash

# arguments: model_type ($1), balanced ($2)
# iter over preprocessing
# iter over folds

preprocessing=(
    dft
    dft_filter
    normalized
    pca
    peak_segmented
    raw
    resampled_25
    resampled_50
    smoothed
    standardized
)

for preprocessing_method in "${preprocessing[@]}"; do
    for fold_id in {0..3}; do
        python train.py --n_models 10 --n_iter 10 --model_type $1 \
                        --preprocessing $preprocessing_method --fold $fold_id
    done
done
