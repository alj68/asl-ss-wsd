#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Dict, Callable, Optional, List, Tuple, Union
from collections import defaultdict, Counter
import os, sys, io, json
import numpy as np
from torch.utils.data import Dataset

from dataset_preprocessor import utils_wordnet
from .utils import lemma_pos_to_tuple


def top_digit_remover(synset_code: List[int]):
    return synset_code[1:] + [0]

def top_digits_remover(synset_codes: List[List[int]]):
    return list(map(top_digit_remover, synset_codes))


class FieldTypeConverter(object):

    def __init__(self, dict_field_type_converter: Dict[str, Callable]):

        self._dict_field_type_converter = dict_field_type_converter

    def __call__(self, sample):

        for field_name, converter in self._dict_field_type_converter.items():
            if field_name in sample:
                sample[field_name] = converter(sample[field_name])

        return sample


class ToWordNetPoSTagConverter(object):

    def __init__(self, pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos",
                 convert_adjective_to_adjective_satellite: bool = True,
                 keep_original_pos: bool = False):

        self._pos_field_name = pos_field_name
        self._mwe_pos_field_name = mwe_pos_field_name
        self._convert_adjective_to_adjective_satellite = convert_adjective_to_adjective_satellite
        self._keep_original_pos = keep_original_pos
        self._pos_orig_field_name = pos_field_name + "_orig"
        self._mwe_pos_orig_field_name = mwe_pos_field_name + "_orig"

    def _adjective_to_adjective_satellite(self, pos):
        return "s" if pos == "a" else pos

    def __call__(self, lst_tokens):
        # Stanford token PoS tags to WordNet PoS tags.

        for token in lst_tokens:
            if self._keep_original_pos:
                token[self._pos_orig_field_name] = token[self._pos_field_name]
            pos = utils_wordnet.ptb_tagset_to_wordnet_tagset(token[self._pos_field_name])
            if self._convert_adjective_to_adjective_satellite:
                pos = self._adjective_to_adjective_satellite(pos)
            token[self._pos_field_name] = pos

            if self._mwe_pos_field_name in token:
                if self._keep_original_pos:
                    token[self._mwe_pos_orig_field_name] = token[self._mwe_pos_field_name]
                pos = utils_wordnet.jmwe_tagset_to_wordnet_tagset(token[self._mwe_pos_field_name])
                if self._convert_adjective_to_adjective_satellite:
                    pos = self._adjective_to_adjective_satellite(pos)
                token[self._mwe_pos_field_name] = pos

        return lst_tokens


class ToWordNetPoSTagAndLemmaConverter(ToWordNetPoSTagConverter):

    def __init__(self,
                 pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos",
                 lemma_field_name: str = "lemma",
                 mwe_lemma_field_name: str = "mwe_lemma",
                 lemma_and_pos_field_name: str = "lemma_pos",
                 mwe_lemma_and_pos_field_name: str = "mwe_lemma_pos",
                 convert_adjective_to_adjective_satellite: bool = True,
                 keep_original_pos: bool = False,
                 lowercase: bool = True):

        super().__init__(pos_field_name, mwe_pos_field_name, convert_adjective_to_adjective_satellite, keep_original_pos)
        self._lemma_field_name = lemma_field_name
        self._mwe_lemma_field_name = mwe_lemma_field_name
        self._lemma_and_pos_field_name = lemma_and_pos_field_name
        self._mwe_lemma_and_pos_field_name = mwe_lemma_and_pos_field_name
        self._lowercase = lowercase
        self._keep_original_pos = keep_original_pos

    def __call__(self, lst_tokens):
        lst_tokens = super().__call__(lst_tokens)
        if self._lowercase:
            return self._create_tuple_lowercase(lst_tokens)
        else:
            return self._create_tuple(lst_tokens)

    def _create_tuple_lowercase(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name].lower(), token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name].lower(), token[self._mwe_pos_field_name])

        return lst_tokens

    def _create_tuple(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name], token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name], token[self._mwe_pos_field_name])

        return lst_tokens


class FrequencyBasedMonosemousEntitySampler(object):

    def __init__(self,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,
                 path_monosemous_words_freq: Optional[str] = None,
                 dataset_monosemous_entity_annotated_corpus: Optional[Dataset] = None,
                 lemma_lowercase: bool = True,
                 enable_random_sampling: bool = True,
                 entity_field_name: str = "entities",
                 random_seed: int = 42
                 ):
        """
        単義語をsamplingする．指定頻度未満の単義語は削除，指定頻度以上の単義語はdown-samplingする．

        @param min_freq: 最小頻度．指定値未満の単義語は削除．
        @param max_freq: 最大頻度．指定値を上回る単義語は，先頭max_freq個のみ採択(enable_random_sampling=False) または 確率的に採択(enable_random_sampling=True)
        @param path_monosemous_words_freq: 単義語の頻度データ．フォーマットはNDJSON, `{'lemma': 'toddler', 'pos': 'n', 'freq': 2681}`
        @param enable_random_sampling: True: 最大頻度を上回る単義語について確率的に採択． False: 先頭max_freq個のみ採択．

        @rtype: object
        """

        if isinstance(min_freq, int):
            if path_monosemous_words_freq is not None:
                assert os.path.exists(path_monosemous_words_freq), f"specified file does not exist: {path_monosemous_words_freq}"
                self._lemma_pos_freq = self._load_lemma_pos_freq(path=path_monosemous_words_freq)
            elif dataset_monosemous_entity_annotated_corpus is not None:
                print("counting lemma x pos frequency from dataset.")
                self._lemma_pos_freq = self._count_lemma_pos_freq(dataset=dataset_monosemous_entity_annotated_corpus,
                                                                  lemma_lowercase=lemma_lowercase, entity_field_name=entity_field_name)
            else:
                raise AssertionError(f"you must specify either `path_monosemous_words_freq` or `dataset_monosemous_entity_annotated_corpus`.")
        else:
            # it always returns zero.
            self._lemma_pos_freq = defaultdict(int)

        self._min_freq = 0 if min_freq is None else min_freq
        self._max_freq = float("inf") if max_freq is None else max_freq
        self._lemma_lowercase = lemma_lowercase
        self._enable_random_sampling = enable_random_sampling
        self._entity_field_name = entity_field_name

        if enable_random_sampling:
            np.random.seed(random_seed)
            self._cursor = -1
            self._random_values = np.random.uniform(size=2 ** 24)

        self.reset()

    def _load_lemma_pos_freq(self, path: str):
        dict_freq = {}
        ifs = io.open(path, mode="r")
        for record in ifs:
            d = json.loads(record.strip())
            key = (d["lemma"], d["pos"])
            dict_freq[key] = d["freq"]
        ifs.close()

        return dict_freq

    @classmethod
    def _count_lemma_pos_freq(cls, dataset: Dataset, lemma_lowercase: bool, entity_field_name: str):
        cnt = Counter()
        for record in dataset:
            lst_lemma_pos = [lemma_pos_to_tuple(lemma_lowercase=lemma_lowercase, **entity) for entity in record[entity_field_name]]
            cnt.update(lst_lemma_pos)

        return cnt

    def random_uniform(self) -> float:
        if self._enable_random_sampling:
            self._cursor += 1
            self._cursor %= self._random_values.size
            return self._random_values[self._cursor]
        else:
            return 0.0

    def decide_sample_or_not(self, lemma_pos):
        freq_total = self._lemma_pos_freq[lemma_pos]
        freq_sampled = self._lemma_pos_freq_sampled[lemma_pos]
        freq_missed = self._lemma_pos_freq_missed[lemma_pos]
        freq_remain = freq_total - freq_missed

        if  (freq_total < self._min_freq) or (freq_sampled >= self._max_freq):
            is_sampled = False
        elif freq_remain <= self._max_freq:
            is_sampled = True
        else:
            # sampling based on total frequency
            is_sampled = (self._max_freq / freq_total) > self.random_uniform()

        if is_sampled:
            self._lemma_pos_freq_sampled[lemma_pos] += 1
        else:
            self._lemma_pos_freq_missed[lemma_pos] += 1

        return is_sampled

    def __call__(self, lst_entities: List[Dict[str, str]]):
        """
        @param lst_entities: list of dict like `[{'lemma': 'Dubonnet', 'pos': 'n', 'occurence': 0, 'span': [1, 2]}]`
        """
        lst_ret = []
        for entity in lst_entities:
            lemma_pos = lemma_pos_to_tuple(lemma_lowercase=self._lemma_lowercase, **entity)
            if not self.decide_sample_or_not(lemma_pos):
                continue
            lst_ret.append(entity)

        return lst_ret

    def __getitem__(self, lemma_pos: Tuple[str, str]):
        key = lemma_pos_to_tuple(lemma_pos[0], lemma_pos[1], self._lemma_lowercase)
        return self._lemma_pos_freq[key]

    def _export_num_of_valid_lemma_pos(self):
        dict_cnt = defaultdict(int)
        for (lemma, pos), freq in self._lemma_pos_freq.items():
            if freq >= self._min_freq:
                dict_cnt[pos] += 1
        return dict_cnt

    def _export_freq_of_valid_lemma_pos(self):
        dict_cnt = defaultdict(int)
        for (lemma, pos), freq in self._lemma_pos_freq.items():
            if freq >= self._min_freq:
                dict_cnt[pos] += min(self._max_freq, freq)
        return dict_cnt

    def reset(self):
        self._lemma_pos_freq_sampled = defaultdict(int)
        self._lemma_pos_freq_missed = defaultdict(int)

    def verbose(self):
        ret = {
            "min_freq": self._min_freq,
            "max_freq": self._max_freq,
            "lemma_lowercase": self._lemma_lowercase,
            "n_total_lemma_and_pos_vocab": len(self._lemma_pos_freq),
            "n_valid_lemma_and_pos_vocab": self._export_num_of_valid_lemma_pos(),
            "n_valid_lemma_and_pos_freq": self._export_freq_of_valid_lemma_pos()
        }
        return ret


class WordNetPoSTagAndLemmaToEntityConverter(object):

    def __init__(self, sense_inventory: "WordNetGlossCorpus"):
        self._sense_inventory = sense_inventory

    def __call__(self, lst_dict_tokens: List[Dict[str, str]]):
        """
        @param lst_dict_tokens: list of token dict like `[{'lemma': 'video', 'pos': 'n', "mwe_lemma":"video_game", "mwe_pos":"n"}]`
        """

        lst_entities = []
        idx = 0

        while idx < len(lst_dict_tokens):
            dict_token = lst_dict_tokens[idx]

            # Multi-word expression
            if "mwe_pos" in dict_token:
                lst_candidate_lemma_keys = self._sense_inventory.get_lemma_keys_by_lemma_and_pos(lemma=dict_token["mwe_lemma"], pos=dict_token["mwe_pos"])
                if len(lst_candidate_lemma_keys) > 0:
                    dict_entity = {
                        "candidate_lemma_keys": lst_candidate_lemma_keys,
                        "lemma": dict_token["mwe_lemma"],
                        "pos": dict_token["mwe_pos"],
                        "pos_orig": dict_token["mwe_pos_orig"],
                        "span": dict_token["mwe_span"],
                    }
                    lst_entities.append(dict_entity)
                    # jump to the end of MWE
                    idx = dict_token["mwe_span"][1]
                    continue

            # Not MWE
            lst_candidate_lemma_keys = self._sense_inventory.get_lemma_keys_by_lemma_and_pos(lemma=dict_token["lemma"], pos=dict_token["pos"])
            if len(lst_candidate_lemma_keys) > 0:
                dict_entity = {
                    "candidate_lemma_keys": lst_candidate_lemma_keys,
                    "lemma": dict_token["lemma"],
                    "pos": dict_token["pos"],
                    "pos_orig": dict_token["pos_orig"],
                    "span": [idx, idx + 1]
                }
                lst_entities.append(dict_entity)

            idx += 1

        return lst_entities