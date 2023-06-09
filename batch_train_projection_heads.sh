#!/bin/bash

./batch_train_projection_heads.py \
--path_args experiment_settings/baseline.json \
--repeats=5 \
--name=repeat_v3 \
--save_eval_metrics experiment_results/repeat_v3.json \
--gpus=1 \
--optional_args="{}" \
--verbose