# leap-climsim-kaggle-5th

This repo contains the full solution used for the LEAP - Atmospheric Physics using AI competition, where it achieved 5th place.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Data Processing and Training](#Data-Processing-and-Training)
- [Inferencing](#inferencing)

## Introduction
This repository provides the code and resources for the 5th place solution for the LEAP - Atmospheric Physics using AI competition.

We achieved our results by training an ensemble of models, each with its own unique variations. The core of each model architecture is a bidirectional LSTM, with different surrounding layers and engineered features. These variations, detailed below, enabled our models to ensemble effectively, leading to our competitive score and 5th-place finish.

Our training strategy involved using as much data as possible initially, followed by fine-tuning on the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset. Some models were initially trained on the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset combined with the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) dataset, while others used the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset combined with a subset of [ClimSim_high-res](https://huggingface.co/datasets/LEAP/ClimSim_high-res) data.

## Models

### Shared Model Features

All our models treat the inputs and outputs in the same manner. The 556 element inputs were separated into column inputs (60x9 = 540) and global inputs (1x16). The global inputs were then repeated 60 times and concatenated to the columns to form a (60x25) input to our models. Engineered features, described in the individual model sections, are added as additional columns. The 368 outputs are predicted directly as a 368-element long output vector to our models.

### Low-Res-Aqua-Mixed

The models trained on the combination of the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) and the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) datasets had the following simple architecture:
- MLP encoder-decoder on the input, outputting a 60x25 matrix which is concatenated to the input
- The concatenated input is fed into a wide (hidden dimension of 512) but shallow (3 layers deep) bidirectional LSTM 
- The output is fed into a single bidirectional GRU layer
- Final linear layer to produce the 368-element output sequence

Several models in our final ensemble followed this procedure without additional feature engineering. Others included various mixes, which can be found in our code (in this appropriately named folder), of:
1. One feature
2. Two feature
3. Three feature

@TODO get feature engineering list

### Low-Res-High-Res-Mixed

The models trained on the combination of the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) and the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) datasets had the following architecture:

@TODO ekffar model

@TODO ekffar features


## Data Processing and Training 

We universally found that training using a Huber loss function with \(\delta = 1\) significantly improves the model's performance.

### Low-Res-Aqua-Mixed Pipeline

#### Preprocessing
The preprocessing steps involve parsing the raw data from the LEAP data repositories, using the [script](https://github.com/leap-stc/ClimSim/blob/main/for_kaggle_users.py) provided in the main ClimSim repository, into parquet files corresponding to the folder name. The parquet files were uploaded as Kaggle datasets. For more details, refer to the ./preprocessing folder.

#### Training 

When training our models, the following steps were taken: 
1. **Downloading the data**: Load the raw .parquet files from uploaded Kaggle datasets.
2. **Process Batches**: Read in each file and output batch sizes of 1000 as torch.float64 files.

Trained with a batch size of 4-5x1000. 

For more details, refer to the .training/low-res-aqua-mixed folder.

4. **Standardization**: Discuss standardization

Trained using a multistep LR scheduler with gamma = 0.65. 

To reduce training time of multiple models, trained a single model on low_res + aqua dataset. Then, finetuned that model with difference SWA and checkpoint averaging parameters to create distinct finetuned models that could ensemble effectively.

### Low-Res-High-Res-Mixed Pipeline

@TODO 

EMA equipped with a beta value of 0.99. 

## Inferencing 

### Low-Res-Aqua-Mixed 

The lowres-aqua mixed models were trained and tuned to predict weighted raw outputs. Model files were uploaded to Kaggle and inference was made directly on the test.csv dataset. An example inference script can be found at [./inference/low-res-aqua-mixed](./inference/low-res-aqua-mixed).

### Low-Res-High-Res-Mixed

@TODO 

### Ensemble Inferencing

The best individual model was a Low-Res-High-Res-Mixed model that achieved a public LB score of 0.78553. The best Low-Res-Aqua-Mixed model was not too different, with an LB public score of 0.78457.

The optimal weighting for the final ensemble was determined experimentally building on the intuition that the Low-Res-High-Res-Mixed model performed slightly better, and therefore received slightly larger weights. Predictions from models and ensembles from various stages of development were included in the final ensemble. Prior to submitting the final ensemble, an additional postprocessing step was applied to variables q0001, q0002, and q0003 whereby the maximum between the pretdicted ptend value and -state/1200 is chosen. The scripts used for performing the final inferencing/ensembling can be found [here](https://www.kaggle.com/code/ajobseeker/final/notebook), leading to a public LB score of 0.79071.
