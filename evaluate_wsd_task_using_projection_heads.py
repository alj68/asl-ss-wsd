#!/usr/bin/env python
# -*- coding:utf-8 -*-

# WSD Task evaluation script for Semantic Specialization for Knowledge-based Word Sense Disambiguation.

from pprint import pprint
import sys, io, os, pickle, json, yaml
import argparse
import pandas as pd

from config_files.wsd_task import cfg_task_dataset
from config_files.sense_annotated_corpus import cfg_evaluation
from config_files.wordnet_gloss_corpus import cfg_embeddings

from dataset.contextualized_embeddings import BERTEmbeddingsDataset
from dataset.gloss_embeddings import BERTLemmaEmbeddingsDataset, SREFLemmaEmbeddingsDataset
from dataset import WSDTaskDataset

from model.encoder import Identity
from lightning_module.trainer import FrozenBERTWSDTaskTrainer
from evaluator.wsd_knn import FrozenBERTKNNWSDTaskEvaluator
from evaluator.wsd_heuristics import TryAgainMechanismWithCoarseSenseInventory, TryAgainMechanism

def _parse_args():

    def nullable_string(value):
        return None if not value else value

    parser = argparse.ArgumentParser(description="WSD Task evaluation script for Semantic Specialization for Knowledge-based Word Sense Disambiguation")
    parser.add_argument("--path_output", required=False, type=str, default=None, help="Save evaluation result to a specified file.")
    parser.add_argument("--path_model_checkpoint", required=True, help="Trained projection heads (*.ckpt file) used for specializing sense/context embeddings.")

    parser.add_argument("--sense_embeddings_name", required=False, type=str, default="WordNet_Gloss_Corpus-AVG-bert-large-cased",
                        help="Sense embeddings dataset name. DEFAULT: WordNet_Gloss_Corpus-AVG-bert-large-cased")
    parser.add_argument("--eval_dataset_name", required=False, type=str, default="WSDEval-ALL-bert-large-cased",
                        help="Evaluation dataset embeddings name. DEFAULT: WSDEval-ALL-bert-large-cased")

    parser.add_argument("--try_again_mechanism", action="store_true", help="Enable Try-again Mechanism [Wang and Wang, ACL2021]. You must specify '--path_coarse_sense_inventory' along with it.")
    parser.add_argument("--path_coarse_sense_inventory", required=False, type=str, default=None,
                        help="Path to the Coarse Sense Inventory [Lacerra et al., AAAI2020]. It is required when we enable '--try_again_emchanism' flag.")

    parser.add_argument("--device", required=False, type=nullable_string, default=None, help="Device id for pytorch. e.g., `cpu` . DEFAULT: None (=cpu)")
    parser.add_argument("--verbose", action="store_true", help="output verbosity.")

    args = parser.parse_args()

    if not os.path.exists(args.path_model_checkpoint):
        raise IOError(f"Specified file does not exist: {args.path_model_checkpoint}")

    if args.try_again_mechanism:
        if not os.path.exists(args.path_coarse_sense_inventory):
            raise IOError(f"You must specify path_coarse_sense_inventory: {args.path_coarse_sense_inventory}")

    if args.device is None:
        args.device = "cpu"

    return args


def main():

    args = _parse_args()

    if args.verbose:
        pprint("==== arguments ===")
        pprint(vars(args), compact=True)

    if args.path_output is not None:
        print(f"Evaluation result will be saved to: {args.path_output}")

    # load projection heads
    path = args.path_model_checkpoint
    if path is not None and os.path.exists(path):
        tup_models = FrozenBERTWSDTaskTrainer.load_projection_heads_from_checkpoint(path, on_gpu=False)
        for model in tup_models:
            _ = model.eval()
            _ = model.to(args.device)
    else:
        print(f"we do not load projection heads: {path}")
        tup_models = (None, None)

    # load context embeddings of evaluation dataset.
    evalset_embeddings = BERTEmbeddingsDataset(**cfg_evaluation[args.eval_dataset_name])
    wsd_task_dataset = WSDTaskDataset(bert_embeddings_dataset=evalset_embeddings, **cfg_task_dataset["WSD"])

    if args.verbose:
        pprint(wsd_task_dataset.verbose)

    # load sense embeddings used for k-NN search.
    sense_embeddings_name = args.sense_embeddings_name
    _cfg = cfg_embeddings[sense_embeddings_name]
    if sense_embeddings_name == "WordNet_Gloss_Corpus-AVG-bert-large-cased":
        gloss_embeddings = BERTLemmaEmbeddingsDataset(**_cfg)
    elif sense_embeddings_name.startswith("SREF_"):
        gloss_embeddings = SREFLemmaEmbeddingsDataset(**_cfg)
    else:
        raise IOError(f"Unknown sense embeddings dataset name: {sense_embeddings_name}")

    if args.verbose:
        pprint(gloss_embeddings.verbose)

    # project sense embeddings
    gloss_projection_head = tup_models[0]
    if (gloss_projection_head is not None) and (gloss_projection_head.__class__.__name__ != "Identity"):
        print("apply gloss projection head to sense embeddings...")
        gloss_embeddings.project_gloss_embeddings(gloss_projection_head=gloss_projection_head, chunksize=1024)

    # Evaluation
    if args.try_again_mechanism:
        _cfg = {
            "lemma_key_embeddings_dataset": gloss_embeddings,
            "path_coarse_sense_inventory": args.path_coarse_sense_inventory,
            "device": args.device
        }
        _try_again_mechanism = TryAgainMechanismWithCoarseSenseInventory(**_cfg)
    elif args.try_again_mechanism is False:
        _try_again_mechanism = False
    elif args.try_again_mechanism == "original":
        _cfg = {
            "lemma_key_embeddings_dataset": gloss_embeddings,
            "similarity_metric": "cosine",
            "exclude_common_semantically_related_synsets": True,
            "lookup_first_lemma_sense_only": True,
            "average_similarity_in_synset": False,
            "exclude_oneselves_for_noun_and_verb": True,
            "do_not_fix_synset_degeneration_bug": True,
            "semantic_relation": "all-relations",
            "device": args.device
        }
        _try_again_mechanism = TryAgainMechanism(**_cfg)

    context_projection_head = tup_models[1]
    if (context_projection_head is not None) and (context_projection_head.__class__.__name__ == "Identity"):
        context_projection_head = None

    print(f"start evaluation...")
    evaluator = FrozenBERTKNNWSDTaskEvaluator(evaluation_dataset=wsd_task_dataset,
                                              context_projection_head=context_projection_head,
                                              try_again_mechanism=_try_again_mechanism,
                                              lemma_key_embeddings_dataset=gloss_embeddings,
                                              similarity_metric="cosine",
                                              device=args.device)
    dict_metrics = evaluator.evaluate()

    pprint(dict_metrics["ALL"])
    pprint(dict_metrics["pos_orig"])
    pprint(dict_metrics["corpus_id"])

    # store results to the specified path.
    path = args.path_output
    if path is not None:
        # append metadata info
        dict_metrics["eval_dataset_name"] = args.eval_dataset_name
        dict_metrics["checkpoint"] = path
        dict_metrics["try_again_mechanism"] = args.try_again_mechanism

        mode = "a" if os.path.exists(path) else "w"
        with io.open(path, mode=mode) as ofs:
            json.dump(dict_metrics, ofs)
            ofs.write("\n")
    else:
        print(f"we do not save evaluation results.")

    print(f"finished. good-bye.")


if __name__ == "__main__":
    main()