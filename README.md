# Knowledge-based Word Sense Disambiguation
* This repostitory is the implementation of the paper "Semantic Specialization for knowledge-based Word Sense Disambiguation."

## Missing codes and objects (as of May. 2023)
* We will add following codes and objects later. These codes are only available in jupyter notebook format, so far.
* Pre-computation script of BERT embeddings for context / sense embeddings.
* ~~Independently executable evaluation script.~~
* Trained models (embeddings transformation functions).

# Contents

## Resources 
* We need to prepare following resources beforehand.
* We recommend you to extract the resources under `./data/` directory.

### WSD Evaluation Framework
* We use the [Unified WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) [Raganato+, EACL2017] as the evaluation dataset. 
* We also use SemCor corpus contained in the framework as the trainset of self-training objective when training projection heads.
  * NOTE: We discard annotated senses during training.

### Coarse Sense Inventory
* We use the [Coarse Sense Inventory](https://sapienzanlp.github.io/csi/) [Lacerra+, AAAI2020] for executing Try-again Mechanism on inference, following SACE [Wang and Wang, ACL2021].

### Sense/context embeddings
* We need precomputed sense/context embeddings for both training and evaluation (inference).  
* You can either download the file or compute by yourself.

### Download 
* You can download precomputed files from [our repository](https://huggingface.co/okazaki-lab/ss-wsd).  
  * Sense embeddings: `bert-large-cased_WordNet_Gloss_Corpus.hdf5`
  * Context embeddings used for training projection heads: `bert-large-cased_SemCor.hdf5`
  * Context embeddings used for evaluation: `bert-large-cased_WSDEval-ALL.hdf5`

### Compute by yourself
* Firstly, configure the resources named `WSDEval-ALL` and `SemCor` in the `config_files/sense_annotated_corpus.py` file.  
  Please refer to "Resource configuration" section for details.
* Secondly, execute `precompute_BERT_embeddings.py` script.
  You can show argument descriptions using `--help` argument.  
  `evaluate_wsd_task_using_projection_heads.sh` contains usage example.


## Resource configuration
* Specify the path of each resource. 

### `config_files/sense_annotated_corpus.py`
* Evaluation dataset from WSD Evaluation Framework
  * `cfg_evaluation.WSDEval-ALL`: 
    * `path_corpus`: Path to `ALL.data.xml`.
    * `path_ground_truth_labels`: Path to `ALL.gold.key.txt`. 
* [Optional] Training dataset for self-training objective from WSD Evaluation Framework
  * `cfg_training.SemCor`
    * `path_corpus`: Path to `semcor.data.xml`.
    * `path_ground_truth_labels`: Path to `semcor.gold.key.txt`.

## Prepare sense/context embeddings


### `config_files/wordnet_gloss_corpus.py`
* Configuration file for WordNet Gloss Corpus and sense embeddings.
* Sense embeddings
  * SREF_basic_lemma_embeddings_without_augmentation: Sense embeddings. Specifically, this is equivalent to basic lemma embeddings used in SREF[Wang and Wang, 2020]. T
  * NOTE: These embeddings are computed without using augmented example sentences.

## Training / Evaluation
* For single trial (run), you can use `train_projection_heads.py`. Usage example can be found in `train_projection_heads.sh`.
  Also, `--help` argument shows the role of each argument. Note that the term "max pool margin task" is equivalent to the self-training objective in the paper.

* For multiple trial at once, you can use `batch_training_projection_heads.py`. Default (baseline) arguments, which is identical to the experiment setting in our paper, can be found in `experiment_settings/baseline.json`.
* When finished training, it will save the trained models and evaluation results (if specified).
  * Trained models: `./checkpoints/{name}/version_{version}/checkpoints/last.ckpt`
  * Evaluation result: `{save_eval_metrics}`

## Evaluation
* Please use `evaluate_wsd_task_using_projection_heads.py` script.  
  You can show argument descriptions using `--help` argument.  
  `evaluate_wsd_task_using_projection_heads.sh` contains usage example.
* Definitions of the metrics are as follows.
  * `f1_score_by_raganato`: The metric reported in our paper. This is micro-averaged F1 score proposed in [Raganato+, EACL2017].
  * `macro_f1_score_by_maru`: Macro-averaged F1 score proposed in [Maru+, ACL2022].
  * `f1_score`: Standard Macro-averaged F1 score. 

# Reference

```
@inproceedings{Mizuki:EACL2023,
    title     = "Semantic Specialization for Knowledge-based Word Sense Disambiguation",
    author    = "Mizuki, Sakae and Okazaki, Naoaki",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    series = {EACL},
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    pages = "3449--3462",
}
```
