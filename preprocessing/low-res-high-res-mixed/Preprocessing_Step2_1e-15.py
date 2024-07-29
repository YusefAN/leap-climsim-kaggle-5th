# -*- coding: utf-8 -*-


import gc
import os
import random
import time
import torch
import datetime
import numpy as np
import pandas as pd
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score
from tqdm import tqdm
import torch.nn.functional as F
import hickle as hkl
from torch.nn import AvgPool1d 
import torch.nn as nn

from torch.nn import LSTM, Conv1d, TransformerEncoder, TransformerEncoderLayer


from torch.nn import LayerNorm

from matplotlib.pyplot import plot
# tmp = addition_y[10:774144:384,10]
# tmp = addition_y[:,2]

# plot(tmp[0:500])


DATA_PATH = 'C:/LEAP/'

BATCH_SIZE = 2048
MIN_STD = 1e-10
MIN_STD_Y = 1e-20
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 10**(-0.3)
EPOCHS = 70
PATIENCE = 6
PRINT_FREQ = 2400
BIN_NUM = 10
n_rounds = 2

df = pd.read_parquet(DATA_PATH+'train.csv')
df_test = pd.read_csv(DATA_PATH+'test.csv')
# df_test = pd.read_csv(DATA_PATH+'test.csv')
# df_test.to_parquet(DATA_PATH+'test.parquet')


seq_fea_list = ['state_t','state_q0001','state_q0002','state_q0003','state_u','state_v','pbuf_ozone','pbuf_CH4','pbuf_N2O']
num_fea_list = ['state_ps','pbuf_SOLIN','pbuf_LHFLX','pbuf_SHFLX','pbuf_TAUX','pbuf_TAUY','pbuf_COSZRS','cam_in_ALDIF','cam_in_ALDIR','cam_in_ASDIF','cam_in_ASDIR','cam_in_LWUP','cam_in_ICEFRAC','cam_in_LANDFRAC','cam_in_OCNFRAC','cam_in_SNOWHLAND']


seq_y_list = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003','ptend_u','ptend_v']
num_y_list = ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']


seq_fea_expand_list = []
for i in seq_fea_list:
    for j in range(60):
        seq_fea_expand_list.append(i+'_'+str(j))


seq_y_expand_list = []
for i in seq_y_list:
    for j in range(60):
        seq_y_expand_list.append(i+'_'+str(j))
        
norm_dict = dict()
TARGET_COLS = seq_y_expand_list + num_y_list
FEAT_COLS = seq_fea_expand_list + num_fea_list

# FEAT_COLS_1 = seq_fea_expand_list + num_fea_list


def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# def seed_everything(seed_val=1325):
#     """Seed everything."""
#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     torch.cuda.manual_seed_all(seed_val)
    
ts = time.time()

weights = pd.read_csv(DATA_PATH + "sample_submission.csv", nrows=1)
del weights['sample_id']
weights = weights.T
weights = weights.to_dict()[0]

# weights.values.shape

# df_train = pl.read_csv(DATA_PATH + "leap-atmospheric-physics-ai-climsim/train.csv", n_rows=2_500_000)

for target in tqdm(weights):
    # print(target)
    df[target] = (df[target]*weights[target])

print("Time to read dataset:", format_time(time.time()-ts), flush=True)

# for i in range(60):
#     df['pos_'+str(i)] = i
#     df_test['pos_'+str(i)] = i


LOG_COLS = ['state_q0001','state_q0002','state_q0003','pbuf_ozone','pbuf_CH4','pbuf_N2O']
for i in tqdm(LOG_COLS):
    for j in range(60):
        df[i+'_'+str(j)] = (np.log(df[i+'_'+str(j)]+1e-7))
        df_test[i+'_'+str(j)] = (np.log(df_test[i+'_'+str(j)]+1e-7))

gc.collect()

norm_dict = dict()

for i in seq_fea_list:
    inter_list = []
    for j in range(60):
        inter_list.append(i+'_'+str(j))
    mean_value = df[inter_list].values.mean()
    std_value = df[inter_list].values.std()
    df[inter_list] = ((df[inter_list]-mean_value)/(MIN_STD+std_value)).astype('float32')
    df_test[inter_list] = ((df_test[inter_list]-mean_value)/(MIN_STD+std_value)).astype('float32')
    norm_dict[i] = [mean_value,std_value]
    
   
for i in num_fea_list:
    mean_value = df[i].values.mean()
    std_value = df[i].values.std()
    df[i] = (df[i]-mean_value)/(MIN_STD+std_value)
    df_test[i] = (df_test[i]-mean_value)/(MIN_STD+std_value)
    norm_dict[i] = [mean_value,std_value]




x_train = df[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
y_train = df[seq_y_expand_list+num_y_list].values
del df

x_test = df_test[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
del df_test

gc.collect()

my = y_train.mean(axis=0)
sy = np.maximum(np.sqrt((y_train*y_train).mean(axis=0)), MIN_STD_Y)
y_train = (y_train - my.reshape(1,-1)) / sy.reshape(1,-1)
y_train = y_train.astype(np.float32)

print(sy[sy<=MIN_STD_Y].shape)


# hkl.dump(x_train, DATA_PATH+'x_train_1e-20.hkl')
# hkl.dump(y_train, DATA_PATH+'y_train_1e-20.hkl')
# hkl.dump(x_test, DATA_PATH+'x_test_1e-20.hkl')

print(sy[sy<=MIN_STD_Y].shape)


DATA_PATH = 'C:/kaggle/CD 1/'

for i1 in tqdm(range(1,9)):
    if i1 == 1:
        lower = 2
    else:
        lower = 1
    for j1 in range(lower,13):
        df_i_j = pd.read_parquet(DATA_PATH+'c_data_'+str(i1)+'_'+str(j1)+'.parquet')
        for target in tqdm(weights):
            # print(target)
            df_i_j[target] = (df_i_j[target]*weights[target])
                
                
        LOG_COLS = ['state_q0001','state_q0002','state_q0003','pbuf_ozone','pbuf_CH4','pbuf_N2O']
        for i in tqdm(LOG_COLS):
            for j in range(60):
                df_i_j[i+'_'+str(j)] = (np.log(df_i_j[i+'_'+str(j)]+1e-7))

        for i in seq_fea_list:
            inter_list = []
            for j in range(60):
                inter_list.append(i+'_'+str(j))

            df_i_j[inter_list] = ((df_i_j[inter_list]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1])).astype('float32')

        for i in num_fea_list:
            df_i_j[i] = (df_i_j[i]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1]).astype('float32')

        x_train_i_j = df_i_j[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
        y_train_i_j = df_i_j[seq_y_expand_list+num_y_list].values
        y_train_i_j = (y_train_i_j - my.reshape(1,-1)) / sy.reshape(1,-1)
        y_train_i_j = y_train_i_j.astype(np.float32)
        hkl.dump(x_train_i_j, DATA_PATH+'c_data_x_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')
        hkl.dump(y_train_i_j, DATA_PATH+'c_data_y_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')
