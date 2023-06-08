#!/usr/bin/env python
# -*- coding:utf-8 -*-

# coding: utf-8

# # BERT Embeddingsの事前計算
# * WSD Training Dataset: SemCor および OMSTI[Raganato+, 2017]について，BERT分散表現を事前計算する．

from typing import Dict, Any
import argparse
import sys, io, os, json, copy, shutil
from pprint import pprint
import progressbar

import numpy as np
import torch
import h5py
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


from dataset.evaluation import WSDEvaluationDataset
from dataset.gloss import WordNetGlossDataset
from dataset.encoder import BERTEmbeddings

from config_files.sense_annotated_corpus import cfg_training, cfg_evaluation
from config_files.wordnet_gloss_corpus import cfg_gloss_corpus

def extract_words_ands_spans_from_record(record: Dict[str, Any]):
    lst_words = record["words"]
    lst_entity_spans = [entity["span"] for entity in record["entities"]]
    return lst_words, lst_entity_spans


def _parse_args():

    def nullable_string(value):
        return None if not value else value

    parser = argparse.ArgumentParser(description="Precompute {gloss,context} embeddings using BERT encoder.")
    parser.add_argument("--dataset_name", required=True, type=str, choices=["SemCor", "WordNet_Gloss_Corpus", "WSDEval-ALL"], help="Dataset name to be processed.")
    parser.add_argument("--path_output", required=True, type=str, help="Save precomputed embeddings to the specified path with HDF5 format.")

    parser.add_argument("--encoder_name", required=False, type=str, default="bert-large-cased", help="Encoder model name. DEFAULT: 'bert-large-cased'")
    parser.add_argument("--batch_size", required=False, type=int, default=128, help="minibatch size.")
    parser.add_argument("--transformer_layers", required=False, type=str, default="-4,-3,-2,-1", help="Layers of transformer blocks to be extracted. DEFAULT: '-4,-3,-2,-1'")
    parser.add_argument("--gpus", required=False, type=nullable_string, default=None, help="GPU device ids. e.g., `0,3,5` Do not specify when you use cpu.")
    parser.add_argument("--verbose", action="store_true", help="output verbosity.")

    args = parser.parse_args()

    if os.path.exists(args.path_output):
        raise IOError(f"Specified file already exists: {args.path_output}")

    if args.gpus is not None:
        args.gpus = list(map(int, args.gpus.split(",")))
        args.__setattr__("device", f"cuda:{args.gpus[0]}")
    else:
        args.__setattr__("device", "cpu")

    if args.transformer_layers is not None:
        args.transformer_layers = list(map(int, args.transformer_layers.split(",")))

    return args

def main():

    args = _parse_args()

    if args.verbose:
        pprint("==== arguments ===")
        pprint(vars(args), compact=True)

    BATCH_SIZE = args.batch_size
    BERT_MODEL_NAME = args.encoder_name
    BERT_LAYERS = args.transformer_layers
    DEVICE_IDS = args.gpus

    path_output = args.path_output

    print(f"file will be saved to: {path_output}")
    path_output_dir = os.path.dirname(path_output)
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)

    corpus_name = args.dataset_name

    if corpus_name == "SemCor":
        corpus = WSDEvaluationDataset(**cfg_training[corpus_name])
    elif corpus_name == "WSDEval-ALL":
        corpus = WSDEvaluationDataset(**cfg_evaluation[corpus_name])
    elif corpus_name == "WordNet_Gloss_Corpus":
        corpus = WordNetGlossDataset(**cfg_gloss_corpus[corpus_name])
    else:
        raise ValueError(f"unknown dataset name: {corpus_name}")
    print(f"# of sentences: {len(corpus)}")

    batch_corpus_reader = DataLoader(dataset=corpus, batch_size=BATCH_SIZE, collate_fn=lambda v: v)

    if args.verbose:
        record = next(iter(corpus))
        pprint(extract_words_ands_spans_from_record(record))

    # BERT encoder
    encoder = BERTEmbeddings(model_or_name=BERT_MODEL_NAME,
                             layers=BERT_LAYERS,
                             return_compressed_format=True,
                             return_numpy_array=True,
                             ignore_too_long_sequence=False,
                             device_ids=DEVICE_IDS)

    ofs = h5py.File(path_output, mode="w")

    q = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    n_processed = 0
    for batch_idx, batch_record in enumerate(batch_corpus_reader):
        n_processed += len(batch_record)
        q.update(n_processed)

        lst_lst_words, lst_lst_entity_spans = list(map(list, zip(*map(extract_words_ands_spans_from_record, batch_record))))

        # encode
        dict_encoded = encoder(lst_lst_words, lst_lst_entity_spans)
        if "embeddings" not in dict_encoded:
            continue

        it_ext_records = zip(batch_record, dict_encoded["entity_spans"], dict_encoded["sequence_spans"])
        for record, lst_lst_entity_spans, sequence_span in it_ext_records:
            for entity, lst_entity_spans in zip(record["entities"], lst_lst_entity_spans):
                entity["subword_spans"] = list(map(list, lst_entity_spans))
            record["sequence_span"] = sequence_span.tolist()

        # save to hdf5 file as a new dataset
        group_name = f"batch_{batch_idx:06d}"

        g = ofs.create_group(group_name)
        g.create_dataset("embeddings", data=dict_encoded["embeddings"])
        g.create_dataset("sequence_lengths", data=dict_encoded["sequence_lengths"])
        g.create_dataset("records", data=json.dumps(batch_record))

    ofs.close()

    print(f"processed sentences: {n_processed}")
    print(f"file path: {path_output}")
    print(f"finished. good-bye.")


if __name__ == "__main__":
    main()