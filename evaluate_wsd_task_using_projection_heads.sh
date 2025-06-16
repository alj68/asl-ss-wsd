#!/bin/bash

# Try-again Mechanism disabled
./evaluate_wsd_task_using_projection_heads.py \
--path_output="./experiment_results/baseline_eval.json" \
--path_model_checkpoint="Z:\PreComputedFiles\baseline.ckpt"\
--sense_embeddings_name="WordNet_Gloss_Corpus-AVG-bert-large-cased" \
--eval_dataset_name="WSDEval-ALL-bert-large-cased" \
--verbose

# Try-again Mechanism enabled
#./evaluate_wsd_task_using_projection_heads.py \
#--path_output="./experiment_results/baseline_eval.json" \
#--path_model_checkpoint="./checkpoints/baseline.ckpt" \
#--sense_embeddings_name="WordNet_Gloss_Corpus-AVG-bert-large-cased" \
#--eval_dataset_name="WSDEval-ALL-bert-large-cased" \
#--path_coarse_sense_inventory="./data/csi_data/wn_synset2csi.txt" \
#--try_again_mechanism \
#--verbose
