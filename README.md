# Semantic Specialization for Knowledge-based Word Sense Disambiguation
* This repository contains the implementation of the paper "Semantic Specialization for Knowledge-based Word Sense Disambiguation" by Mizuki and Okazaki, presented at EACL2023.

## Environment Setup
* The implementation was tested using Python 3.8.x.
* To install the necessary dependencies, run the following command:  
  `pip install requirements.txt`
* We recommend using virtual environments such as pyenv, pipenv, or anaconda.

## Resources
* Before running the code, please make sure to set up the following resources:
* We recommend extracting/downloading the resources under `./data/` directory.

### WSD Task Evaluation Framework
* We use the [Unified WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) [Raganato et al., EACL2017] as the evaluation dataset.  
* Optionally, the SemCor corpus contained in this framework is used as the training set for the self-training objective when training projection heads.

### Coarse Sense Inventory
* We utilize the [Coarse Sense Inventory](https://sapienzanlp.github.io/csi/) [Lacerra et al., AAAI2020] for executing the Try-again Mechanism during inference, following the approach described in SACE [Wang and Wang, ACL2021].

## Precomputing Sense/Context Embeddings
* To proceed with training and evaluation, sense and context embeddings need to be precomputed.  
* You can choose to either download the precomputed files or compute them yourself.

### Download
* Precomputed files can be downloaded from [our repository](https://huggingface.co/okazaki-lab/ss_wsd).  
* The following files are required:
  - Sense embeddings: `bert-large-cased_WordNet_Gloss_Corpus.hdf5`
  - Context embeddings used for training projection heads: `bert-large-cased_SemCor.hdf5`
  - Context embeddings used for evaluation: `bert-large-cased_WSDEval-ALL.hdf5`

### Computing by Yourself
* If you prefer to compute the embeddings yourself, follow these steps:
1. Configure the resources named `WSDEval-ALL` and `SemCor` in the `config_files/sense_annotated_corpus.py` file.  
  Please refer to the "Resource Configuration" section for detailed instructions.
2. Execute the `precompute_BERT_embeddings.py` script.  
  Use the `--dataset_name` argument to specify which dataset will be processed.  
  You can find an example usage in the `evaluate_wsd_task_using_projection_heads.sh` file.  
  For more information about the script's arguments, use the `--help` argument.

## Resource Configuration
* Please edit `config_files/sense_annotated_corpus.py` and configure the path of each resource.

### Sense Embeddings
* Please edit `cfg_training` as follows:
- `WordNet_Gloss_Corpus-bert-large-cased`: Precomputed sense embeddings using WordNet Gloss Corpus.
  - `path`: Path to the `.hdf5` file.

### Context Embeddings

#### Evaluation
* Please edit `cfg_evaluation` as follows:
- `WSDEval-ALL`: Evaluation dataset from the WSD Evaluation Framework.
  - `path_corpus`: Path to `ALL.data.xml`.
  - `path_ground_truth_labels`: Path to `ALL.gold.key.txt`.
- `WSDEval-ALL-bert-large-cased`: Precomputed sense embeddings using the evaluation dataset.
  - `path`: Path to the `.hdf5` file.

#### Training
* Please edit `cfg_training` as follows.
* `SemCor`: SemCor corpus contained in the WSD Evaluation Framework.
    * `path_corpus`: Path to `semcor.data.xml`.
    * `path_ground_truth_labels`: Path to `semcor.gold.key.txt`.
* `SemCor-bert-large-cased`: Precomputed context embeddings using SemCor corpus.
    * `path`: Path to the `.hdf5` file.

## Train the model (projection heads)
* You can choose to either download the trained model or train it yourself.

### Download
* The trained model `baseline.ckpt` can be downloaded from [our repository](https://huggingface.co/okazaki-lab/ss_wsd).  

### Train by yourself
* For a single trial (run), you can use the `train_projection_heads.py` script.  
  An example usage can be found in the `train_projection_heads.sh` file.  
  Also the `--help` argument shows the role of each argument.    
  Note that the term "max pool margin task" is equivalent to the self-training objective in the paper.
* For multiple trials at once, you can use the `batch_train_projection_heads.py` script.  
  Use `--repeats` argument to specify the number of trials.  
  To strictly follow the experiment setting in the paper, you can specify the `./experiment_settings/baseline.json` file for `--path_args` argument.  
* When finished training, the trained models and evaluation results (if specified) are saved as follows.
  * Trained models: `./checkpoints/{name}/version_{0:repeats}/checkpoints/last.ckpt`
  * Evaluation result: The path specified for the `--save_eval_metrics` argument.
* NOTE: The performance may not match the results reported in the paper due to the stochastic nature of training.

## Evaluation
* Please use the `evaluate_wsd_task_using_projection_heads.py` script to evaluate the trained model.  
  Use the `--path_model_checkpoint` argument to specify the trained model path (*.ckpt file).  
  Also, use the `--try_again_mechanism` flag to enable Try-again Mechanism and  
  `--path_coarse_sense_inventory` argument to specify the Coarse Sense Inventory file (wn_synset2csi.txt).  
  Example usages can be found in the `evaluate_wsd_task_using_projection_heads.sh` file.  
  For more information about the script's arguments, use the `--help` argument.     
* The definitions of the metrics are as follows.
  * `f1_score_by_raganato`: The metric reported in our paper. This is the micro-averaged F1 score proposed in [Raganato+, EACL2017].
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
