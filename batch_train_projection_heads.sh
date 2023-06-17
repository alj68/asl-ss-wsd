#!/bin/bash

./batch_train_projection_heads.py \
--path_args=experiment_settings/baseline.json \
--repeats=5 \
--name=baseline \
--save_eval_metrics experiment_results/baseline.json \
--gpus=1 \
--optional_args="{}" \
--verbose