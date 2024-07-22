# leap-climsim-kaggle-5th

This repo contains the full solution used for the LEAP - Atmospheric Physics using AI competition, where it achieved 5th place.

## Table of Contents
- [Introduction](#introduction)
- [Shared Model Features](#shared-model-features)
- [Low-Res-Aqua-Mixed](#Low-Res-Aqua-Mixed)
- [Low-Res-High-Res-Mixed](#Low-Res-High-Res-Mixed)

## Introduction
This repository provides the code and resources for the 5th place solution for the LEAP - Atmospheric Physics using AI competition.

We achieved our results by training an ensemble of models, each with its own unique variations and pipelines. The core of each model architecture is a bidirectional LSTM, with different surrounding layers and engineered features. These variations, detailed below, enabled our models to ensemble effectively, leading to our competitive score and 5th-place finish.

Our training strategy involved using as much data as possible initially, followed by fine-tuning on the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset. Some models were initially trained on the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset combined with the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) dataset, while others used the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) dataset combined with a subset of [ClimSim_high-res](https://huggingface.co/datasets/LEAP/ClimSim_high-res) data.

## Shared Model Features

All our models treat the inputs and outputs in the same manner.

**Input:**
- 556 elements total
  - 540 elements are 1D data with a length of 60
  - 16 elements are global values
- Reshaped to 60x25, where:
  - First 60x9 columns correspond to the 1D data
  - Last 60x16 columns contain constant repeating values for each global value
- Additional features, described in the individual model sections, are added as additional columns

**Output:**  
- 368 elements total
  - 360 elements are 1D data with a length of 60
  - 8 elements are global values
- Output sequence is predicted directly as a 368x1 vector

We universally found that training using a Huber loss function with \(\delta = 1\) significantly improve the model's performance.

## Low-Res-Aqua-Mixed

### Model

The models trained on the combination of the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) and the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) datasets had the following architecture:

**Architecture:**
- MLP encoder-decoder on the input, outputting a 60x25 matrix which is concatenated to the input
- The 60x50 input is fed into a wide (hidden dimension of 512) but shallow (3 layers deep) bidirectional LSTM 
- The output is fed into a single bidirectional GRU layer
- Final linear layer to produce the 368-element output sequence

Several models in our final ensemble followed this procedure without any additional feature engineering. Others included various mixes, which can be found in our code (in this appropriately named folder), of:
1. One feature
2. Two feature
3. Three feature

@TODO get feature engineering list

### Pipeline

The preprocessing steps involve parsing the raw data from the LEAP data repositories, using the [script](https://github.com/leap-stc/ClimSim/blob/main/for_kaggle_users.py) provided in the main ClimSim repository, into parquet files corresponding to the folder name. The parquet files were uploaded as Kaggle datasets. 

When training our models, the following steps were taken: 
1. **Downloading the data**: Load the raw .parquet files from uploaded Kaggle datasets.
2. **Process Batches**: Read in each file and output batch sizes of 1024 as torch.float64 files. 
3. **Standardization**: Discuss standardization

For more details, refer to the ./preprocessing folder.

### Inference

The lowres-aqua mixed models were trained and tuned to predict weighted raw outputs. Model files were uploaded to Kaggle and inference was made directly on the test.csv dataset. Example inference script can be found at [./inference](./inference).

## Low-Res-High-Res-Mixed

### Model
The models trained on the combination of the [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) and the [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) datasets had the following architecture:

**Architecture:**
@TODO ekffar model

@TODO ekffar features

### Pipeline

### Inference

@TODO ekffar inference
