#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Callable, Iterable, Any, List, Union, Set
from collections import defaultdict
# import pydash
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import wordnet as wn

from config_files.wsd_task import WSDTaskDataLoader
from dataset import WSDTaskDataset
from dataset_preprocessor import utils_wordnet_gloss

class BaseEvaluator(object):

    def _iter_to_set(self, inputs: Union[str, Iterable[str]]):
        if isinstance(inputs, str):
            return {inputs,}
        else:
            return set(inputs)

    def precision(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        numerator = self._iter_to_set(predictions).intersection(self._iter_to_set(ground_truthes))
        denominator = self._iter_to_set(predictions)

        if len(denominator) == 0:
            return 0.0
        else:
            return len(numerator) / len(denominator)

    def recall(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        numerator = self._iter_to_set(predictions).intersection(self._iter_to_set(ground_truthes))
        denominator = self._iter_to_set(ground_truthes)

        if len(denominator) == 0:
            return 0.0
        else:
            return len(numerator) / len(denominator)

    def _calc_f1_score(self, prec: float, recall: float):
        if prec == recall == 0.0:
            return 0.0
        else:
            return 2*prec*recall/(prec+recall)

    def f1_score(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        prec = self.precision(ground_truthes, predictions)
        recall = self.recall(ground_truthes, predictions)
        return self._calc_f1_score(prec, recall)

    def accuracy(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        """
        it returns the exact-match accuracy; known as subset accuracy in literature.
        this behaviour is equivalent to sklearn.metrics.accuracy_score() function in multilabel classification.
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

        @param ground_truthes: iterable of strings.
        @param predictions: iterable of strings.
        """
        gt = self._iter_to_set(ground_truthes)
        pred = self._iter_to_set(predictions)
        if gt == pred:
            return 1.0
        else:
            return 0.0

    def compute_metrics(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        dict_ret = {
            "precision": self.precision(ground_truthes, predictions),
            "recall": self.recall(ground_truthes, predictions),
            "f1_score": self.f1_score(ground_truthes, predictions),
            "accuracy": self.accuracy(ground_truthes, predictions)
        }
        return dict_ret

    def macro_average(self, lst_dict_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        dict_lst_metrics = defaultdict(list)

        for dict_metrics in lst_dict_metrics:
            for metric, value in dict_metrics.items():
                dict_lst_metrics[metric].append(value)

        dict_ret = {metric:np.mean(lst_values) for metric, lst_values in dict_lst_metrics.items()}

        return dict_ret

    def macro_average_recursive(self, dict_lst_dict_metrics: Dict[str, Union[Dict, List]]) -> Dict[str, Dict[str, float]]:
        dict_ret = {}
        for key, values in dict_lst_dict_metrics.items():
            if isinstance(values, list):
                dict_ret[key] = self.macro_average(values)
            elif isinstance(values, Dict):
                dict_ret[key] = self.macro_average_recursive(values)
        return dict_ret


class BaseEvaluatorByRaganato(BaseEvaluator):

    def macro_average(self, lst_dict_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        wrong implementation that is used by [Raganato+, 2017]
        source: http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip -> Evaluation_Datasets/Scorer.java
        """
        dict_ret = super().macro_average(lst_dict_metrics)

        n_ = len(lst_dict_metrics)
        dict_ret["recall_by_raganato"] = dict_ret["precision"] * n_ / n_
        dict_ret["f1_score_by_raganato"] = self._calc_f1_score(prec=dict_ret["precision"], recall=dict_ret["recall_by_raganato"])

        return dict_ret

class WSDTaskEvaluatorBase(BaseEvaluatorByRaganato, metaclass=ABCMeta):

    def __init__(self,
                 evaluation_dataset: WSDTaskDataset,
                 ground_truth_lemma_keys_field_name: str = "ground_truth_lemma_keys",
                 evaluation_category: str = "lemma",
                 breakdown_attributes: Optional[Iterable[Set[str]]] = None,
                 device: Optional[Any] = "cpu",
                 verbose: bool = False,
                 **kwargs_dataloader):

        self._ground_truth_lemma_keys_field_name = ground_truth_lemma_keys_field_name
        self._evaluation_category = evaluation_category

        available_category = {"lemma", "lexname"}
        assert evaluation_category in available_category, \
            ValueError(f"`evaluation_category` must be: {available_category}")

        # create evalset dataloader
        self._evaluation_dataset = evaluation_dataset
        if isinstance(evaluation_dataset, WSDTaskDataset):
            self._evaluation_data_loader = WSDTaskDataLoader(evaluation_dataset, batch_size=1, cfg_collate_function={"device":device})
        else:
            raise ValueError(f"unknown dataset: {type(evaluation_dataset)}")


        # if breakdown is not set, apply default dataset
        if breakdown_attributes is None:
            self._breakdown_attributes = [{"corpus_id",}, {"pos_orig",}, {"corpus_id", "pos_orig"}]
        else:
            self._breakdown_attributes = breakdown_attributes
        self.verbose = verbose
        self._device = device
        self._predict_kwargs = {}

    def _tensor_to_list(self, tensor_or_list: Union[torch.Tensor, List]):
        if isinstance(tensor_or_list, torch.Tensor):
            return tensor_or_list.tolist()
        else:
            return tensor_or_list

    @property
    def predict_kwargs(self):
        return self._predict_kwargs

    @predict_kwargs.setter
    def predict_kwargs(self, kwargs):
        self._predict_kwargs = kwargs

    @abstractmethod
    def predict(self, input: Dict[str, Any], **kwargs) -> Iterable[str]:
        pass

    def _lemma_to_lexname(self, lemma_or_lemma_key: Union[str, wn.lemma]):
        if isinstance(lemma_or_lemma_key, str):
            return utils_wordnet_gloss.lemma_key_to_lexname(lemma_or_lemma_key)
        else:
            return lemma_or_lemma_key.synset().lexname()

    def _lemma_to_synset_id(self, lemma_or_lemma_key: Union[str, wn.lemma]):
        if isinstance(lemma_or_lemma_key, str):
            return utils_wordnet_gloss.lemma_key_to_synset_id(lemma_key=lemma_or_lemma_key)
        else:
            return lemma_or_lemma_key.synset().name()

    def compute_metrics(self, ground_truthes: Iterable[str], predictions: Iterable[str]):
        if self._evaluation_category == "lexname":
            ground_truthes = list(map(self._lemma_to_lexname, ground_truthes))
            predictions = list(map(self._lemma_to_lexname, predictions))
        return super().compute_metrics(ground_truthes=ground_truthes, predictions=predictions)

    def assertion(self):
        return True

    def predict_batch(self, batch) -> List[Iterable[str]]:
        lst_ret = []
        for record in batch:
            lst_ret.append(self.predict(record))
        return lst_ret

    def _get_attr_key_and_values(self, set_attr_names: Set[str], example: Dict[str, str], concat="|"):
        attr_keys = concat.join([attr_name for attr_name in set_attr_names])
        attr_values = concat.join([example[attr_name] for attr_name in set_attr_names])
        return attr_keys, attr_values

    def iter_records(self):
        is_wsd_task_dataset = isinstance(self._evaluation_dataset, WSDTaskDataset)

        for single_example_batch in self._evaluation_data_loader:
            if is_wsd_task_dataset:
                inputs_for_evaluator = single_example_batch["records"][0]
                inputs_for_predictor = single_example_batch
                del inputs_for_predictor["records"]
                inputs_for_predictor.update(inputs_for_evaluator)
            else:
                inputs_for_predictor = single_example_batch[0]
                inputs_for_evaluator = single_example_batch[0]
            yield inputs_for_predictor, inputs_for_evaluator

    def __iter__(self):
        """
        iterate over examples in the evaluation dataset.

        """
        for inputs_for_predictor, inputs_for_evaluator in self.iter_records():
            predictions = self.predict(inputs_for_predictor, **self.predict_kwargs)
            ground_truthes = inputs_for_evaluator[self._ground_truth_lemma_keys_field_name]
            dict_metrics = self.compute_metrics(ground_truthes, predictions)
            yield inputs_for_predictor, inputs_for_evaluator, ground_truthes, predictions, dict_metrics

    def __len__(self):
        if not hasattr(self, "n_sample"):
            n_sample = 0
            for _ in self.iter_records():
                n_sample += 1
            self.n_sample = n_sample
        return self.n_sample

    def evaluate(self, **kwargs):
        assert self.assertion(), f"assertion failed."

        dict_dict_results = defaultdict(lambda : defaultdict(list))
        dict_dict_results["ALL"] = []

        self.predict_kwargs = kwargs
        for _, inputs_for_evaluator, ground_truthes, predicitons, dict_metrics in self:
            # store metrics
            dict_dict_results["ALL"].append(dict_metrics)
            # breakdown by attributes
            for set_attr_names in self._breakdown_attributes:
                # e.g. grouper_attr = "corpus_id", breakdown_value = "semeval2"
                grouper_attr, breakdown_value = self._get_attr_key_and_values(set_attr_names, inputs_for_evaluator)
                dict_dict_results[grouper_attr][breakdown_value].append(dict_metrics)

        # compute macro average of all breakdowns and metrics.
        dict_summary = self.macro_average_recursive(dict_dict_results)

        return dict_summary
