# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:47:00 2024

@author: Administrator
"""

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
from rtdl_num_embeddings import (
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    compute_bins,
)

from torch.nn import LayerNorm

from matplotlib.pyplot import plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = 'C:/LEAP/'

BATCH_SIZE = 2048
MIN_STD = 1e-10
MIN_STD_Y = 1e-15
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 10**(-0.3)
EPOCHS = 70
PATIENCE = 6
PRINT_FREQ = 2400
BIN_NUM = 10
n_rounds = 2

df = pd.read_parquet(DATA_PATH+'train.parquet')
df_test = pd.read_csv(DATA_PATH+'test.csv')

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

def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
 
ts = time.time()

weights = pd.read_csv(DATA_PATH + "sample_submission.csv", nrows=1)
del weights['sample_id']
weights = weights.T
weights = weights.to_dict()[0]


for target in tqdm(weights):
    df[target] = (df[target]*weights[target])

print("Time to read dataset:", format_time(time.time()-ts), flush=True)


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
    # df[inter_list] = ((df[inter_list]-mean_value)/(MIN_STD+std_value)).astype('float32')
    df_test[inter_list] = ((df_test[inter_list]-mean_value)/(MIN_STD+std_value)).astype('float32')
    norm_dict[i] = [mean_value,std_value]
    
   
for i in num_fea_list:
    mean_value = df[i].values.mean()
    std_value = df[i].values.std()
    df_test[i] = (df_test[i]-mean_value)/(MIN_STD+std_value)
    norm_dict[i] = [mean_value,std_value]


zeroout_index = list(range(145,147))
zeroout_index = torch.tensor(list(set(zeroout_index)))


y_train = df[seq_y_expand_list+num_y_list].values
del df

x_test = df_test[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
del df_test

gc.collect()

my = y_train.mean(axis=0)
y_train = y_train - my.reshape(1,-1)
sy = np.maximum(np.sqrt((y_train*y_train).mean(axis=0)), MIN_STD_Y)

print(sy[sy<=MIN_STD_Y].shape)


class FFNN_LSTM_749_AVG_ATT(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFNN_LSTM_749_AVG_ATT, self).__init__()
        
        self.encode_dim = 256
        self.hidden_dim = 320
        self.iter_dim = 1024

        self.LSTM_1 = LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.05,bidirectional=True)
        self.input_size = input_size
        self.Linear_1 = nn.Linear(len(seq_fea_list)+len(num_fea_list), self.encode_dim)
        self.Linear_2 = nn.Linear(6*self.hidden_dim+self.encode_dim, self.iter_dim)
        self.Linear_3 = nn.Linear(self.iter_dim, len(seq_y_list))
        self.Linear_3_0 = nn.Linear(self.iter_dim, 1)

        self.Linear_4_0 = nn.Linear(self.iter_dim, self.iter_dim*2)

        self.Linear_4 = nn.Linear(self.iter_dim*2, len(num_y_list))
        self.bias = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.weight = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.avg_pool_1 = AvgPool1d(kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        
        x_seq = x[:,0:60*len(seq_fea_list)]
        x_seq = x_seq.reshape((-1,len(seq_fea_list),60))
        x_seq = torch.transpose(x_seq, 1, 2)
        
        x_num = x[:,60*len(seq_fea_list):x.shape[1]]
        x_num_repeat = x_num.reshape((-1,1,len(num_fea_list)))
        x_num_repeat = x_num_repeat.repeat((1,60,1))
        
        x_seq = F.elu(self.Linear_1(torch.concat((x_seq,x_num_repeat),dim=-1)/5))
        
        x_seq_1,_ = self.LSTM_1(x_seq/5)
        
        x_seq_1_mean = torch.mean(x_seq_1,dim=1,keepdim=True)
        x_seq_1_mean = x_seq_1_mean.repeat((1,60,1))

        x_seq_1_avg_pool = self.avg_pool_1(torch.transpose(x_seq_1, 1, 2))
        x_seq_1_avg_pool = torch.transpose(x_seq_1_avg_pool,1, 2)
        
        x_seq_1 = F.elu(self.Linear_2(torch.cat((x_seq_1,x_seq_1_mean,x_seq,x_seq_1_avg_pool),dim=-1)/5))
        
        att_weight = F.softmax(self.Linear_3_0(x_seq_1) - 10,dim=1)
        
        x_seq_out = self.Linear_3(x_seq_1)
        x_seq_out = torch.transpose(x_seq_out, 1, 2)
        x_seq_out = x_seq_out.reshape((-1,60*len(seq_y_list)))
        
        x_num_out = F.elu(self.Linear_4_0(torch.sum(att_weight*x_seq_1,dim=1)))
        x_num_out = self.Linear_4(x_num_out)
        
        return self.weight.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias.weight/3

class FFNN_LSTM_6_AVG(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFNN_LSTM_6_AVG, self).__init__()
        
        self.encode_dim = 300
        self.hidden_dim = 280
        self.iter_dim = 800

        self.LSTM_1 = LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.01,bidirectional=True)
        self.input_size = input_size
        self.Linear_1 = nn.Linear(len(seq_fea_list)+len(num_fea_list), self.encode_dim)
        self.Linear_2 = nn.Linear(6*self.hidden_dim+self.encode_dim, self.iter_dim)
        self.Linear_3 = nn.Linear(self.iter_dim, len(seq_y_list))
        self.Linear_4_0 = nn.Linear(self.iter_dim, self.iter_dim*2)

        self.Linear_4 = nn.Linear(self.iter_dim*2, len(num_y_list))
        self.bias = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.weight = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.avg_pool_1 = AvgPool1d(kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        x_seq = x[:,0:60*len(seq_fea_list)]
        x_seq = x_seq.reshape((-1,len(seq_fea_list),60))
        x_seq = torch.transpose(x_seq, 1, 2)
        
        x_num = x[:,60*len(seq_fea_list):x.shape[1]]
        x_num_repeat = x_num.reshape((-1,1,len(num_fea_list)))
        x_num_repeat = x_num_repeat.repeat((1,60,1))
        
        x_seq = F.elu(self.Linear_1(torch.concat((x_seq,x_num_repeat),dim=-1)/5))
        
        x_seq_1,_ = self.LSTM_1(x_seq/5)
        
        x_seq_1_mean = torch.mean(x_seq_1,dim=1,keepdim=True)
        x_seq_1_mean = x_seq_1_mean.repeat((1,60,1))

        x_seq_1_avg_pool = self.avg_pool_1(torch.transpose(x_seq_1, 1, 2))
        x_seq_1_avg_pool = torch.transpose(x_seq_1_avg_pool,1, 2)
        
        x_seq_1 = F.elu(self.Linear_2(torch.cat((x_seq_1,x_seq_1_mean,x_seq,x_seq_1_avg_pool),dim=-1)/5))
        
        x_seq_out = self.Linear_3(x_seq_1)
        x_seq_out = torch.transpose(x_seq_out, 1, 2)
        x_seq_out = x_seq_out.reshape((-1,60*len(seq_y_list)))
        
        x_num_out = F.elu(self.Linear_4_0(torch.mean(x_seq_1,dim=1)))
        x_num_out = self.Linear_4(x_num_out)

        output = self.weight.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias.weight/3
        
        output[:,zeroout_index] =  output[:,zeroout_index]*0.0
        
        return output
    
class FFNN_LSTM_888_AVG_ATT(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFNN_LSTM_888_AVG_ATT, self).__init__()
        
        self.encode_dim = 256
        self.hidden_dim = 256
        self.iter_dim = 800

        self.LSTM_1 = LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.01,bidirectional=True)
        self.input_size = input_size
        self.Linear_1 = nn.Linear(len(seq_fea_list)+len(num_fea_list), self.encode_dim)
        self.Linear_2 = nn.Linear(6*self.hidden_dim+self.encode_dim, self.iter_dim)
        self.Linear_3 = nn.Linear(self.iter_dim, len(seq_y_list))
        self.Linear_3_0 = nn.Linear(self.iter_dim, 1)

        self.Linear_4_0 = nn.Linear(self.iter_dim, self.iter_dim*2)

        self.Linear_4 = nn.Linear(self.iter_dim*2, len(num_y_list))
        self.bias = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.weight = nn.Linear(len(seq_y_list)*60+len(num_y_list),1)
        self.avg_pool_1 = AvgPool1d(kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        x_seq = x[:,0:60*len(seq_fea_list)]
        x_seq = x_seq.reshape((-1,len(seq_fea_list),60))
        x_seq = torch.transpose(x_seq, 1, 2)
        
        x_num = x[:,60*len(seq_fea_list):x.shape[1]]
        x_num_repeat = x_num.reshape((-1,1,len(num_fea_list)))
        x_num_repeat = x_num_repeat.repeat((1,60,1))
        
        x_seq = F.elu(self.Linear_1(torch.concat((x_seq,x_num_repeat),dim=-1)/5))
        
        x_seq_1,_ = self.LSTM_1(x_seq/5)
        
        x_seq_1_mean = torch.mean(x_seq_1,dim=1,keepdim=True)
        x_seq_1_mean = x_seq_1_mean.repeat((1,60,1))

        x_seq_1_avg_pool = self.avg_pool_1(torch.transpose(x_seq_1, 1, 2))
        x_seq_1_avg_pool = torch.transpose(x_seq_1_avg_pool,1, 2)
        
        x_seq_1 = F.elu(self.Linear_2(torch.cat((x_seq_1,x_seq_1_mean,x_seq,x_seq_1_avg_pool),dim=-1)/5))
        
        att_weight = F.softmax(self.Linear_3_0(x_seq_1) - 10,dim=1)
        
        x_seq_out = self.Linear_3(x_seq_1)
        x_seq_out = torch.transpose(x_seq_out, 1, 2)
        x_seq_out = x_seq_out.reshape((-1,60*len(seq_y_list)))
        
        x_num_out = F.elu(self.Linear_4_0(torch.sum(att_weight*x_seq_1,dim=1)))
        x_num_out = self.Linear_4(x_num_out)
        
        return self.weight.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias.weight/3

 
input_size = x_test.shape[1]
output_size = y_train.shape[1]

model_single =  FFNN_LSTM_6_AVG(input_size, output_size)
device_ids = list(range(torch.cuda.device_count()))
model = torch.nn.DataParallel(model_single)
zeroout_index.to(device)

model.to(device)

from ema_pytorch import EMA
ema = EMA(
    model,
    beta = 0.99,               # exponential moving average factor
    update_after_step = 50,    # only after this number of .update() calls will it start updating
    update_every = 8,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

ema.load_state_dict(torch.load(DATA_PATH+'FFNN_LSTM_6_AVG_mix_1e-15_retrain_ema_24.pt'))
predt = np.zeros([x_test.shape[0], output_size], dtype=np.float32)

ema.eval()

i1 = 0
for i in tqdm(range(10000)):
    # time.sleep(time_gap)

    i2 = np.minimum(i1 + BATCH_SIZE, x_test.shape[0])
    if i1 == i2:  # Break the loop if range does not change
        break

    # Convert the current slice of xt to a PyTorch tensor
    inputs = torch.from_numpy(x_test[i1:i2, :]).float().to(device)

    # No need to track gradients for inference
    with torch.no_grad():
        outputs = ema(inputs)  # Get model predictions
        predt[i1:i2, :] = outputs.cpu().numpy()  # Store predictions in predt

    i1 = i2  # Update i1 to the end of the current batch

    if i2 >= x_test.shape[0]:
        break
    
count = 0
for i in range(sy.shape[0]):
    if sy[i] <= MIN_STD_Y*1.01:
        predt[:,i] = 0
        count = count + 1
        
predt = np.double(predt) * sy.reshape(1,-1) + my.reshape(1,-1)

ss = pd.read_csv(DATA_PATH + "sample_submission.csv")
ss.iloc[:,1:] = predt

df_test = pl.read_csv(DATA_PATH + "test.csv")

gc.collect()

use_cols = []
for i in range(28):
    use_cols.append(f"ptend_q0002_{i}")
    
ss2 = pd.read_csv(DATA_PATH + "sample_submission.csv")

df_test = df_test.to_pandas()
for col in use_cols:
    ss[col] = -df_test[col.replace("ptend", "state")]*ss2[col]/1200.

ss[["sample_id"]+list(TARGET_COLS)].to_parquet(DATA_PATH + "submission_12.parquet")