# Domain Adaptation Meta Analysis
This repository contains code to replicate the adaptation experiments described in our TACL 2022 paper: [Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks](https://arxiv.org/abs/2111.01340)

This repository contains implementations of the following adaptation methods:
1. Adversarial domain adaptation ([Ganin et al (2016)](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf))
2. Self-training
3. Domain adaptive pretraining (two variants described by [Han and Eisenstein (2019)](https://aclanthology.org/D19-1433.pdf) and [Gururangan et al (2020)](https://aclanthology.org/2020.acl-main.740.pdf))
4. Classifier-based instance weighting ([Jiang and Zhai (2007)](https://aclanthology.org/P07-1034.pdf))
5. Frustratingly easy domain adaptation (neural version)

Note that since the TACL version only presents results for the unsupervised adaptation setting, only methods 1-4 are used in the reported experiments. 

## Code Requirements
- python 3.6

- library requirements in requirements.txt

## Dataset Setup

This code expects all datasets to be placed in a separate "datasets" folder in the code's parent folder. Please refer to the read_dataset() function (lines 16-54) in run_experiments.py to ensure that dataset paths match. Unfortunately we cannot include the datasets in this repository since the i2b2 datasets require users to sign a data use agreement (DUA). For more details, please visit: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/.

## Running the System
Though instructions to run the system will differ slightly based on the setting (zero-shot, unsupervised adaptation or limited supervision adaptation), the primary command to run the system is as follows:

```
>>> python run_experiments.py --source <SOURCE_DATASET> --target <TARGET_DATASET> --task <TASK_NAME> --out_dir <MODEL_SAVE_PATH> --model_name_or_path <LM_MODEL_TYPE> --level <SETTING> --device <GPU_ID>
``` 

- For the --source and --target arguments, this code expects one of the following values: conll2003, i2b22006, i2b22010, i2b22014, timebank, mtsamples, i2b22012. Note that we do not currently support any other datasets, but new datasets can be added fairly easily by adding a new dataloader (to dataloader.py) and adding an option to read the dataset (in run_experiments.py).

- For the --task argument, this code expects one of the following values: ee (event extraction), ner-coarse (coarse named entity recognition), ner-fine (fine named entity recognition). Note that coarse and fine NER primarily differ in that coarse NER does not perform entity type prediction, while fine NER does include type prediction.

- For the --model_name_or_path argument, this code expects either a directory containing the pretrained language model or a string reference to the language model from Huggingface (e.g., bert-base-uncased). 

- For the --level argument, this code expects one of three values: 0 (zero-shot), 1 (unsupervised adaptation), and 2 (limited supervision adaptation).

### Optional Arguments
- --lr: Learning rate for the model

- --epochs: Number of epochs to train the model for

- --batch_size: Batch size for the model

- --seed: Set seed for reproducibility

- --dropout: Dropout for task model

- --dump_test_preds: Dump prediction on the test sets (if needed for analysis)

To replicate experiments from the submission, you should not need to fiddle with any of these arguments.

### Options for Zero-Shot Experiments
To run experiments in the zero-shot setting, ensure that --level is set to 0. No other settings need to be changed.

### Options for Unsupervised Adaptation Experiments
To run experiments in the unsupervised adaptation setting, ensure that --level is set to 1. Additionally, use the --da_method argument to choose one of the following adaptation methods: LA (loss augmentation), PL (pseudo-labeling), PT (pretraining), IW (instance weighting). 

Other optional arguments that can be tweaked include the following:

- --adv_layers: Number of layers for domain classifier used in loss augmentation (LA) method

- --adv_hidden_size: Hidden size for domain classifier used in loss augmentation (LA) method, if it is set to have more than one layer

- --adv_coeff: Balancing coefficient used to weight loss from domain classifier and task model in loss augmentation (LA) method

- --pt_model_name_or_path: Pretrained model used to initialize LM in pretraining (PT) method

- --pl_passes: Number of passes to run self-training for in pseudo-labeling (PL) method

To replicate experiments from the submission, you should not need to tweak any of the defaults for these optional arguments.

### Options for Limited Supervision Adaptation Experiments
To run experiments in the limited supervision adaptation setting, ensure that --level is set to 2. Additionally use the --da_method argument to choose one of the following adaptation baselines/methods: TGONLY (target-only training), SCTGCL (source training followed by target), SCTGJL (joint training on source and target), LA (loss augmentation), FA (feature augmentation), IW (instance weighting).

Note that the TACL version does not include any limited supervision experiments, but if you are interested, you can refer to chapter 2 of my thesis, which discusses these results and additional follow-up analyses in more detail: https://www.cs.cmu.edu/~anaik/thesis_draft.pdf.

If you find our code useful, please cite the following paper:
```
@article{naik2021adapting,
  title={Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks},
  author={Naik, Aakanksha and Lehman, Jill and Rose, Carolyn},
  journal={arXiv preprint arXiv:2111.01340},
  year={2021}
}
```
