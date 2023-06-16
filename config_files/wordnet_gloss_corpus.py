#!/usr/bin/env python
# -*- coding:utf-8 -*-

from .sense_annotated_corpus import cfg_training

cfg_gloss_corpus = {
    "WordNet_Gloss_Corpus": {
        "target_pos": ["n","v","s","r"],
        "concat_extended_examples": False,
        "lemma_surface_form_lowercase": False,
        "replicate_sref_format": True,
        "convert_adjective_to_adjective_satellite": True,
        "lst_path_extended_examples_corpus": None,
        "description": "WordNet Gloss corpus sense corpus used in SREF[Wang and Wang, EMNLP2020]. This corpus utilized a) WordNet lemmas and b) WordNet definition and examples."
    }
}

cfg_embeddings = {
    "WordNet_Gloss_Corpus-AVG-bert-large-cased": {
        "kwargs_bert_embeddings_dataset": cfg_training["WordNet_Gloss_Corpus-bert-large-cased"],
        "pooling_method": "average",
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "description": "WordNet Gloss sentence embeddings."
    }
}