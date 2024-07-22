# leap-climsim-kaggle-5th

This repo contains the full solution used for the LEAP - Atmospheric Physics using AI competition, where it achieved 5th place.

## Table of Contents
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Models](#models)


## Introduction
This repository provides the code and resources for the 5th place solution for the LEAP - Atmospheric Physics using AI competition. 

## Preprocessing
The preprocessing steps involve parsing the raw [ClimSim_low-res](https://huggingface.co/datasets/LEAP/ClimSim_low-res) and [ClimSim_low-res_aqua-planet](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet) into parquet files corresponding to the folder name.

The script used for this can be found [here](https://github.com/leap-stc/ClimSim/blob/main/for_kaggle_users.py) in the main ClimSim repository. 


### Steps:
1. **Downloading the data**: Load the raw .parquet files from uploaded Kaggle datasets.
2. **Process Batches**: Read in each file and output batch sizes of 1000  as torch.float64 files. 

For more details, refer to the ./preprocessing folder.

## Models

### Low-Res-Aqua-Mixed
@TODO Model description


### HB Model - @TOOD

### ekffar model - @TODO

## Inference

### Low-Res-Aqua-Mixed

The lowres-aqua mixed models were trained and tuned to predict weighted raw outputs. Model files were uploaded to Kaggle and inference was made directly on the test.csv dataset. Example inference script can be found at [./inference](./inference).