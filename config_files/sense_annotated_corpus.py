#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.filter import EmptyFilter, DictionaryFilter
from .utils import pick_first_available_path

_no_entity_sentence_filter = EmptyFilter(check_field_names=["entities"])


DIR_EVALSET = "Z:\\PreComputedFiles\\eval_data\\semeval2007\\"
DIR_TRAINSET = "Z:\\PreComputedFiles\\WSD_Training_Corpora\\"
DIR_EMBEDDINGS ="Z:\\PreComputedFiles\\bert_embeddings\\"

# evaluation dataset for all-words WSD task
cfg_evaluation = {
    "WSDEval-ALL": {
        "path_corpus": os.path.join(DIR_EVALSET, "semeval2007.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "semeval2007.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: ALL",
    },
    "WSDEval-ALL-bert-large-cased": {
        "path": os.path.join(DIR_EMBEDDINGS, "bert-large-cased_WSDEval-ALL.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017] encoded by BERT-large-cased."
    }
}

cfg_training = {
    "SemCor": {
        "path_corpus": os.path.join(DIR_TRAINSET, "SemCor\\semcor.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_TRAINSET, "SemCor\\semcor.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "description": "WSD SemCor corpora, excluding no-sense-annotated sentences.",
    },
    "SemCor-bert-large-cased": {
        "path": os.path.join(DIR_EMBEDDINGS, "bert-large-cased_SemCor.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "WSD SemCor corpora (excluding no-sense-annotated sentences) encoded by BERT-large-cased."
    },
    "WordNet_Gloss_Corpus-bert-large-cased": {
        "path": os.path.join(DIR_EMBEDDINGS, "bert-large-cased_WordNet_Gloss_Corpus.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "WordNet Gloss corpora encoded by BERT-large-cased."
    }
}