#!/bin/bash

for dataset_name in SemCor WSDEval-ALL WordNet_Gloss_Corpus; do
  ./precompute_BERT_embeddings.py \
  --dataset_name=${dataset_name} \
  --path_output="./data/bert_embeddings/bert-large-cased_${dataset_name}.hdf5" \
  --encoder_name="bert-large-cased" \
  --gpus=2,3,4,5 \
  --batch_size=128 \
  --verbose
done