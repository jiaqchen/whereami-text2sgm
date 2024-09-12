#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="/cluster/project/cvg/jiaqchen/h_coarse_loc/playground/graph_models/models/"

# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

python3 eval.py \
    --mode disabled \
    --epoch 30 \
    --N 1 \
    --lr 0.0001 \
    --weight_decay 0.00005 \
    --batch_size 16 \
    --contrastive_loss True \
    --valid_top_k 1 2 3 5 \
    --use_attributes True \
    --training_with_cross_val True \
    --folds 10 \
    --skip_k_fold True \
    --entire_training_set \
    --eval_iters 10 \
    --eval_iter_count 500 \
    --subgraph_ablation \
    --scanscribe_auto_gen \
    --eval_only_c \
    --model_name model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint