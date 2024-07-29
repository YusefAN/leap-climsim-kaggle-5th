import torch
from tqdm import tqdm
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.nn.init as init
from utils.constants import * 
import os
DEVICE="cuda"

seed = 422
torch.manual_seed(seed)

# Adjustments made to diversify the model
# I created four different types of models based on the true/false of mask and order_extra_first
# and ensembled them by modifying the above parameters accordingly.
mask = True
order_extra_first = True

finetune = False
if finetune:
    ALL_FILES=[f'../preprocessing/lowres_torch{i}' for i in os.listdir('../preprocessing/lowres_torch')]
else:
    ALL_FILES=[f'../preprocessing/lowres_torch{i}' for i in os.listdir('../preprocessing/lowres_torch')]# + [f'../preprocessing/ocean_torch/{i}' for i in os.listdir('../preprocessing/ocean_torch/')]
    
NUM_ROWS = len(ALL_FILES)
DATA_MAP = {k:v for k,v in zip(range(NUM_ROWS),ALL_FILES)}

values_loaded = torch.load("./mean_std_values_9features.pth")
X_MEAN = values_loaded['X_MEAN']
X_STD = values_loaded['X_STD']
Y_MEAN = values_loaded['Y_MEAN']
Y_STD = values_loaded['Y_STD']

WEIGHT_MASK = torch.tensor(NEW_TARGET_WEIGHTS)

ADJUSTMENT_COLUMNS = [f"ptend_q0002_{i}" for i in range(28)]
ADJUSTMENT_MASK=[]

FEATURE_NAMES = ['state_t_0', 'state_t_1', 'state_t_2', 'state_t_3', 'state_t_4', 'state_t_5', 'state_t_6', 'state_t_7', 'state_t_8', 'state_t_9', 'state_t_10', 'state_t_11', 'state_t_12', 'state_t_13', 'state_t_14', 'state_t_15', 'state_t_16', 'state_t_17', 'state_t_18', 'state_t_19', 'state_t_20', 'state_t_21', 'state_t_22', 'state_t_23', 'state_t_24', 'state_t_25', 'state_t_26', 'state_t_27', 'state_t_28', 'state_t_29', 'state_t_30', 'state_t_31', 'state_t_32', 'state_t_33', 'state_t_34', 'state_t_35', 'state_t_36', 'state_t_37', 'state_t_38', 'state_t_39', 'state_t_40', 'state_t_41', 'state_t_42', 'state_t_43', 'state_t_44', 'state_t_45', 'state_t_46', 'state_t_47', 'state_t_48', 'state_t_49', 'state_t_50', 'state_t_51', 'state_t_52', 'state_t_53', 'state_t_54', 'state_t_55', 'state_t_56', 'state_t_57', 'state_t_58', 'state_t_59', 'state_q0001_0', 'state_q0001_1', 'state_q0001_2', 'state_q0001_3', 'state_q0001_4', 'state_q0001_5', 'state_q0001_6', 'state_q0001_7', 'state_q0001_8', 'state_q0001_9', 'state_q0001_10', 'state_q0001_11', 'state_q0001_12', 'state_q0001_13', 'state_q0001_14', 'state_q0001_15', 'state_q0001_16', 'state_q0001_17', 'state_q0001_18', 'state_q0001_19', 'state_q0001_20', 'state_q0001_21', 'state_q0001_22', 'state_q0001_23', 'state_q0001_24', 'state_q0001_25', 'state_q0001_26', 'state_q0001_27', 'state_q0001_28', 'state_q0001_29', 'state_q0001_30', 'state_q0001_31', 'state_q0001_32', 'state_q0001_33', 'state_q0001_34', 'state_q0001_35', 'state_q0001_36', 'state_q0001_37', 'state_q0001_38', 'state_q0001_39', 'state_q0001_40', 'state_q0001_41', 'state_q0001_42', 'state_q0001_43', 'state_q0001_44', 'state_q0001_45', 'state_q0001_46', 'state_q0001_47', 'state_q0001_48', 'state_q0001_49', 'state_q0001_50', 'state_q0001_51', 'state_q0001_52', 'state_q0001_53', 'state_q0001_54', 'state_q0001_55', 'state_q0001_56', 'state_q0001_57', 'state_q0001_58', 'state_q0001_59', 'state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3', 'state_q0002_4', 'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9', 'state_q0002_10', 'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18', 'state_q0002_19', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_24', 'state_q0002_25', 'state_q0002_26', 'state_q0002_27', 'state_q0002_28', 'state_q0002_29', 'state_q0002_30', 'state_q0002_31', 'state_q0002_32', 'state_q0002_33', 'state_q0002_34', 'state_q0002_35', 'state_q0002_36', 'state_q0002_37', 'state_q0002_38', 'state_q0002_39', 'state_q0002_40', 'state_q0002_41', 'state_q0002_42', 'state_q0002_43', 'state_q0002_44', 'state_q0002_45', 'state_q0002_46', 'state_q0002_47', 'state_q0002_48', 'state_q0002_49', 'state_q0002_50', 'state_q0002_51', 'state_q0002_52', 'state_q0002_53', 'state_q0002_54', 'state_q0002_55', 'state_q0002_56', 'state_q0002_57', 'state_q0002_58', 'state_q0002_59', 'state_q0003_0', 'state_q0003_1', 'state_q0003_2', 'state_q0003_3', 'state_q0003_4', 'state_q0003_5', 'state_q0003_6', 'state_q0003_7', 'state_q0003_8', 'state_q0003_9', 'state_q0003_10', 'state_q0003_11', 'state_q0003_12', 'state_q0003_13', 'state_q0003_14', 'state_q0003_15', 'state_q0003_16', 'state_q0003_17', 'state_q0003_18', 'state_q0003_19', 'state_q0003_20', 'state_q0003_21', 'state_q0003_22', 'state_q0003_23', 'state_q0003_24', 'state_q0003_25', 'state_q0003_26', 'state_q0003_27', 'state_q0003_28', 'state_q0003_29', 'state_q0003_30', 'state_q0003_31', 'state_q0003_32', 'state_q0003_33', 'state_q0003_34', 'state_q0003_35', 'state_q0003_36', 'state_q0003_37', 'state_q0003_38', 'state_q0003_39', 'state_q0003_40', 'state_q0003_41', 'state_q0003_42', 'state_q0003_43', 'state_q0003_44', 'state_q0003_45', 'state_q0003_46', 'state_q0003_47', 'state_q0003_48', 'state_q0003_49', 'state_q0003_50', 'state_q0003_51', 'state_q0003_52', 'state_q0003_53', 'state_q0003_54', 'state_q0003_55', 'state_q0003_56', 'state_q0003_57', 'state_q0003_58', 'state_q0003_59', 'state_u_0', 'state_u_1', 'state_u_2', 'state_u_3', 'state_u_4', 'state_u_5', 'state_u_6', 'state_u_7', 'state_u_8', 'state_u_9', 'state_u_10', 'state_u_11', 'state_u_12', 'state_u_13', 'state_u_14', 'state_u_15', 'state_u_16', 'state_u_17', 'state_u_18', 'state_u_19', 'state_u_20', 'state_u_21', 'state_u_22', 'state_u_23', 'state_u_24', 'state_u_25', 'state_u_26', 'state_u_27', 'state_u_28', 'state_u_29', 'state_u_30', 'state_u_31', 'state_u_32', 'state_u_33', 'state_u_34', 'state_u_35', 'state_u_36', 'state_u_37', 'state_u_38', 'state_u_39', 'state_u_40', 'state_u_41', 'state_u_42', 'state_u_43', 'state_u_44', 'state_u_45', 'state_u_46', 'state_u_47', 'state_u_48', 'state_u_49', 'state_u_50', 'state_u_51', 'state_u_52', 'state_u_53', 'state_u_54', 'state_u_55', 'state_u_56', 'state_u_57', 'state_u_58', 'state_u_59', 'state_v_0', 'state_v_1', 'state_v_2', 'state_v_3', 'state_v_4', 'state_v_5', 'state_v_6', 'state_v_7', 'state_v_8', 'state_v_9', 'state_v_10', 'state_v_11', 'state_v_12', 'state_v_13', 'state_v_14', 'state_v_15', 'state_v_16', 'state_v_17', 'state_v_18', 'state_v_19', 'state_v_20', 'state_v_21', 'state_v_22', 'state_v_23', 'state_v_24', 'state_v_25', 'state_v_26', 'state_v_27', 'state_v_28', 'state_v_29', 'state_v_30', 'state_v_31', 'state_v_32', 'state_v_33', 'state_v_34', 'state_v_35', 'state_v_36', 'state_v_37', 'state_v_38', 'state_v_39', 'state_v_40', 'state_v_41', 'state_v_42', 'state_v_43', 'state_v_44', 'state_v_45', 'state_v_46', 'state_v_47', 'state_v_48', 'state_v_49', 'state_v_50', 'state_v_51', 'state_v_52', 'state_v_53', 'state_v_54', 'state_v_55', 'state_v_56', 'state_v_57', 'state_v_58', 'state_v_59', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND', 'pbuf_ozone_0', 'pbuf_ozone_1', 'pbuf_ozone_2', 'pbuf_ozone_3', 'pbuf_ozone_4', 'pbuf_ozone_5', 'pbuf_ozone_6', 'pbuf_ozone_7', 'pbuf_ozone_8', 'pbuf_ozone_9', 'pbuf_ozone_10', 'pbuf_ozone_11', 'pbuf_ozone_12', 'pbuf_ozone_13', 'pbuf_ozone_14', 'pbuf_ozone_15', 'pbuf_ozone_16', 'pbuf_ozone_17', 'pbuf_ozone_18', 'pbuf_ozone_19', 'pbuf_ozone_20', 'pbuf_ozone_21', 'pbuf_ozone_22', 'pbuf_ozone_23', 'pbuf_ozone_24', 'pbuf_ozone_25', 'pbuf_ozone_26', 'pbuf_ozone_27', 'pbuf_ozone_28', 'pbuf_ozone_29', 'pbuf_ozone_30', 'pbuf_ozone_31', 'pbuf_ozone_32', 'pbuf_ozone_33', 'pbuf_ozone_34', 'pbuf_ozone_35', 'pbuf_ozone_36', 'pbuf_ozone_37', 'pbuf_ozone_38', 'pbuf_ozone_39', 'pbuf_ozone_40', 'pbuf_ozone_41', 'pbuf_ozone_42', 'pbuf_ozone_43', 'pbuf_ozone_44', 'pbuf_ozone_45', 'pbuf_ozone_46', 'pbuf_ozone_47', 'pbuf_ozone_48', 'pbuf_ozone_49', 'pbuf_ozone_50', 'pbuf_ozone_51', 'pbuf_ozone_52', 'pbuf_ozone_53', 'pbuf_ozone_54', 'pbuf_ozone_55', 'pbuf_ozone_56', 'pbuf_ozone_57', 'pbuf_ozone_58', 'pbuf_ozone_59', 'pbuf_CH4_0', 'pbuf_CH4_1', 'pbuf_CH4_2', 'pbuf_CH4_3', 'pbuf_CH4_4', 'pbuf_CH4_5', 'pbuf_CH4_6', 'pbuf_CH4_7', 'pbuf_CH4_8', 'pbuf_CH4_9', 'pbuf_CH4_10', 'pbuf_CH4_11', 'pbuf_CH4_12', 'pbuf_CH4_13', 'pbuf_CH4_14', 'pbuf_CH4_15', 'pbuf_CH4_16', 'pbuf_CH4_17', 'pbuf_CH4_18', 'pbuf_CH4_19', 'pbuf_CH4_20', 'pbuf_CH4_21', 'pbuf_CH4_22', 'pbuf_CH4_23', 'pbuf_CH4_24', 'pbuf_CH4_25', 'pbuf_CH4_26', 'pbuf_CH4_27', 'pbuf_CH4_28', 'pbuf_CH4_29', 'pbuf_CH4_30', 'pbuf_CH4_31', 'pbuf_CH4_32', 'pbuf_CH4_33', 'pbuf_CH4_34', 'pbuf_CH4_35', 'pbuf_CH4_36', 'pbuf_CH4_37', 'pbuf_CH4_38', 'pbuf_CH4_39', 'pbuf_CH4_40', 'pbuf_CH4_41', 'pbuf_CH4_42', 'pbuf_CH4_43', 'pbuf_CH4_44', 'pbuf_CH4_45', 'pbuf_CH4_46', 'pbuf_CH4_47', 'pbuf_CH4_48', 'pbuf_CH4_49', 'pbuf_CH4_50', 'pbuf_CH4_51', 'pbuf_CH4_52', 'pbuf_CH4_53', 'pbuf_CH4_54', 'pbuf_CH4_55', 'pbuf_CH4_56', 'pbuf_CH4_57', 'pbuf_CH4_58', 'pbuf_CH4_59', 'pbuf_N2O_0', 'pbuf_N2O_1', 'pbuf_N2O_2', 'pbuf_N2O_3', 'pbuf_N2O_4', 'pbuf_N2O_5', 'pbuf_N2O_6', 'pbuf_N2O_7', 'pbuf_N2O_8', 'pbuf_N2O_9', 'pbuf_N2O_10', 'pbuf_N2O_11', 'pbuf_N2O_12', 'pbuf_N2O_13', 'pbuf_N2O_14', 'pbuf_N2O_15', 'pbuf_N2O_16', 'pbuf_N2O_17', 'pbuf_N2O_18', 'pbuf_N2O_19', 'pbuf_N2O_20', 'pbuf_N2O_21', 'pbuf_N2O_22', 'pbuf_N2O_23', 'pbuf_N2O_24', 'pbuf_N2O_25', 'pbuf_N2O_26', 'pbuf_N2O_27', 'pbuf_N2O_28', 'pbuf_N2O_29', 'pbuf_N2O_30', 'pbuf_N2O_31', 'pbuf_N2O_32', 'pbuf_N2O_33', 'pbuf_N2O_34', 'pbuf_N2O_35', 'pbuf_N2O_36', 'pbuf_N2O_37', 'pbuf_N2O_38', 'pbuf_N2O_39', 'pbuf_N2O_40', 'pbuf_N2O_41', 'pbuf_N2O_42', 'pbuf_N2O_43', 'pbuf_N2O_44', 'pbuf_N2O_45', 'pbuf_N2O_46', 'pbuf_N2O_47', 'pbuf_N2O_48', 'pbuf_N2O_49', 'pbuf_N2O_50', 'pbuf_N2O_51', 'pbuf_N2O_52', 'pbuf_N2O_53', 'pbuf_N2O_54', 'pbuf_N2O_55', 'pbuf_N2O_56', 'pbuf_N2O_57', 'pbuf_N2O_58', 'pbuf_N2O_59']

TARGET_NAMES = ['ptend_t_0', 'ptend_t_1', 'ptend_t_2', 'ptend_t_3', 'ptend_t_4', 'ptend_t_5', 'ptend_t_6', 'ptend_t_7', 'ptend_t_8', 'ptend_t_9', 'ptend_t_10', 'ptend_t_11', 'ptend_t_12', 'ptend_t_13', 'ptend_t_14', 'ptend_t_15', 'ptend_t_16', 'ptend_t_17', 'ptend_t_18', 'ptend_t_19', 'ptend_t_20', 'ptend_t_21', 'ptend_t_22', 'ptend_t_23', 'ptend_t_24', 'ptend_t_25', 'ptend_t_26', 'ptend_t_27', 'ptend_t_28', 'ptend_t_29', 'ptend_t_30', 'ptend_t_31', 'ptend_t_32', 'ptend_t_33', 'ptend_t_34', 'ptend_t_35', 'ptend_t_36', 'ptend_t_37', 'ptend_t_38', 'ptend_t_39', 'ptend_t_40', 'ptend_t_41', 'ptend_t_42', 'ptend_t_43', 'ptend_t_44', 'ptend_t_45', 'ptend_t_46', 'ptend_t_47', 'ptend_t_48', 'ptend_t_49', 'ptend_t_50', 'ptend_t_51', 'ptend_t_52', 'ptend_t_53', 'ptend_t_54', 'ptend_t_55', 'ptend_t_56', 'ptend_t_57', 'ptend_t_58', 'ptend_t_59', 'ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3', 'ptend_q0001_4', 'ptend_q0001_5', 'ptend_q0001_6', 'ptend_q0001_7', 'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11', 'ptend_q0001_12', 'ptend_q0001_13', 'ptend_q0001_14', 'ptend_q0001_15', 'ptend_q0001_16', 'ptend_q0001_17', 'ptend_q0001_18', 'ptend_q0001_19', 'ptend_q0001_20', 'ptend_q0001_21', 'ptend_q0001_22', 'ptend_q0001_23', 'ptend_q0001_24', 'ptend_q0001_25', 'ptend_q0001_26', 'ptend_q0001_27', 'ptend_q0001_28', 'ptend_q0001_29', 'ptend_q0001_30', 'ptend_q0001_31', 'ptend_q0001_32', 'ptend_q0001_33', 'ptend_q0001_34', 'ptend_q0001_35', 'ptend_q0001_36', 'ptend_q0001_37', 'ptend_q0001_38', 'ptend_q0001_39', 'ptend_q0001_40', 'ptend_q0001_41', 'ptend_q0001_42', 'ptend_q0001_43', 'ptend_q0001_44', 'ptend_q0001_45', 'ptend_q0001_46', 'ptend_q0001_47', 'ptend_q0001_48', 'ptend_q0001_49', 'ptend_q0001_50', 'ptend_q0001_51', 'ptend_q0001_52', 'ptend_q0001_53', 'ptend_q0001_54', 'ptend_q0001_55', 'ptend_q0001_56', 'ptend_q0001_57', 'ptend_q0001_58', 'ptend_q0001_59', 'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26', 'ptend_q0002_27', 'ptend_q0002_28', 'ptend_q0002_29', 'ptend_q0002_30', 'ptend_q0002_31', 'ptend_q0002_32', 'ptend_q0002_33', 'ptend_q0002_34', 'ptend_q0002_35', 'ptend_q0002_36', 'ptend_q0002_37', 'ptend_q0002_38', 'ptend_q0002_39', 'ptend_q0002_40', 'ptend_q0002_41', 'ptend_q0002_42', 'ptend_q0002_43', 'ptend_q0002_44', 'ptend_q0002_45', 'ptend_q0002_46', 'ptend_q0002_47', 'ptend_q0002_48', 'ptend_q0002_49', 'ptend_q0002_50', 'ptend_q0002_51', 'ptend_q0002_52', 'ptend_q0002_53', 'ptend_q0002_54', 'ptend_q0002_55', 'ptend_q0002_56', 'ptend_q0002_57', 'ptend_q0002_58', 'ptend_q0002_59', 'ptend_q0003_0', 'ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4', 'ptend_q0003_5', 'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8', 'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11', 'ptend_q0003_12', 'ptend_q0003_13', 'ptend_q0003_14', 'ptend_q0003_15', 'ptend_q0003_16', 'ptend_q0003_17', 'ptend_q0003_18', 'ptend_q0003_19', 'ptend_q0003_20', 'ptend_q0003_21', 'ptend_q0003_22', 'ptend_q0003_23', 'ptend_q0003_24', 'ptend_q0003_25', 'ptend_q0003_26', 'ptend_q0003_27', 'ptend_q0003_28', 'ptend_q0003_29', 'ptend_q0003_30', 'ptend_q0003_31', 'ptend_q0003_32', 'ptend_q0003_33', 'ptend_q0003_34', 'ptend_q0003_35', 'ptend_q0003_36', 'ptend_q0003_37', 'ptend_q0003_38', 'ptend_q0003_39', 'ptend_q0003_40', 'ptend_q0003_41', 'ptend_q0003_42', 'ptend_q0003_43', 'ptend_q0003_44', 'ptend_q0003_45', 'ptend_q0003_46', 'ptend_q0003_47', 'ptend_q0003_48', 'ptend_q0003_49', 'ptend_q0003_50', 'ptend_q0003_51', 'ptend_q0003_52', 'ptend_q0003_53', 'ptend_q0003_54', 'ptend_q0003_55', 'ptend_q0003_56', 'ptend_q0003_57', 'ptend_q0003_58', 'ptend_q0003_59', 'ptend_u_0', 'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5', 'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10', 'ptend_u_11', 'ptend_u_12', 'ptend_u_13', 'ptend_u_14', 'ptend_u_15', 'ptend_u_16', 'ptend_u_17', 'ptend_u_18', 'ptend_u_19', 'ptend_u_20', 'ptend_u_21', 'ptend_u_22', 'ptend_u_23', 'ptend_u_24', 'ptend_u_25', 'ptend_u_26', 'ptend_u_27', 'ptend_u_28', 'ptend_u_29', 'ptend_u_30', 'ptend_u_31', 'ptend_u_32', 'ptend_u_33', 'ptend_u_34', 'ptend_u_35', 'ptend_u_36', 'ptend_u_37', 'ptend_u_38', 'ptend_u_39', 'ptend_u_40', 'ptend_u_41', 'ptend_u_42', 'ptend_u_43', 'ptend_u_44', 'ptend_u_45', 'ptend_u_46', 'ptend_u_47', 'ptend_u_48', 'ptend_u_49', 'ptend_u_50', 'ptend_u_51', 'ptend_u_52', 'ptend_u_53', 'ptend_u_54', 'ptend_u_55', 'ptend_u_56', 'ptend_u_57', 'ptend_u_58', 'ptend_u_59', 'ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3', 'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8', 'ptend_v_9', 'ptend_v_10', 'ptend_v_11', 'ptend_v_12', 'ptend_v_13', 'ptend_v_14', 'ptend_v_15', 'ptend_v_16', 'ptend_v_17', 'ptend_v_18', 'ptend_v_19', 'ptend_v_20', 'ptend_v_21', 'ptend_v_22', 'ptend_v_23', 'ptend_v_24', 'ptend_v_25', 'ptend_v_26', 'ptend_v_27', 'ptend_v_28', 'ptend_v_29', 'ptend_v_30', 'ptend_v_31', 'ptend_v_32', 'ptend_v_33', 'ptend_v_34', 'ptend_v_35', 'ptend_v_36', 'ptend_v_37', 'ptend_v_38', 'ptend_v_39', 'ptend_v_40', 'ptend_v_41', 'ptend_v_42', 'ptend_v_43', 'ptend_v_44', 'ptend_v_45', 'ptend_v_46', 'ptend_v_47', 'ptend_v_48', 'ptend_v_49', 'ptend_v_50', 'ptend_v_51', 'ptend_v_52', 'ptend_v_53', 'ptend_v_54', 'ptend_v_55', 'ptend_v_56', 'ptend_v_57', 'ptend_v_58', 'ptend_v_59', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

for col, wt in zip(TARGET_NAMES, NEW_TARGET_WEIGHTS):
    if wt==0 and col in ADJUSTMENT_COLUMNS:
        ADJUSTMENT_MASK.append(0)
    else:
        ADJUSTMENT_MASK.append(1)

ADJUSTMENT_MASK=torch.tensor(ADJUSTMENT_MASK)
MASK = torch.tensor(NEW_TARGET_WEIGHTS)

#Define indices for features used in feature engineering
STATE_T_IDX = [i for i in range(60)]
STATE_T_IDX_2 = [i+1 for i in range(60)]
STATE_Q0001_IDX = [i+60 for i in range(60)]
STATE_Q0001_IDX_2 = [i+61 for i in range(60)]
STATE_U_IDX = [i+240 for i in range(60)]
STATE_U_IDX_2 = [i+241 for i in range(60)]
STATE_V_IDX = [i+300 for i in range(60)]
STATE_V_IDX_2 = [i+301 for i in range(60)]

STATE_Q0002_IDX = [i+120 for i in range(60)]
STATE_Q0003_IDX = [i+180 for i in range(60)]
PBUF_OZONE_IDX = [i+376 for i in range(60)]
PBUF_CH4_IDX = [i+436 for i in range(60)]
PBUF_N2O_IDX = [i+496 for i in range(60)]

#Calculate engineered features from an input x and return x concatenated with the the new features
def preprocess_x(x):
    temp_diff1 = x[:,  STATE_T_IDX] - (x[:,STATE_T_IDX_2])
    temp_diff2 = x[:,  STATE_Q0001_IDX] - (x[:,STATE_Q0001_IDX_2])
    wind_diff1 = x[:,  STATE_U_IDX] - (x[:,STATE_U_IDX_2])
    wind_diff2 = x[:,  STATE_V_IDX] - (x[:,STATE_V_IDX_2])
    temp_humid = x[:, STATE_T_IDX] / (x[:,STATE_Q0001_IDX])
    air_total = x[:, PBUF_OZONE_IDX] + (x[:,PBUF_CH4_IDX]) + x[:, PBUF_N2O_IDX]
    moisture = ((x[:, STATE_Q0001_IDX])**2) * ((x[:,STATE_U_IDX])**2 + (x[:,STATE_V_IDX])**2)
    liq_partition = x[:, STATE_Q0002_IDX] / (x[:,STATE_Q0002_IDX] + x[:,STATE_Q0003_IDX])
    imbalance = (x[:,STATE_Q0002_IDX] - x[:,STATE_Q0003_IDX]) / (x[:,STATE_Q0002_IDX] + x[:,STATE_Q0003_IDX])
    additional_features = torch.nan_to_num(torch.cat([temp_diff1, temp_diff2, wind_diff1, wind_diff2, temp_humid, air_total, moisture, liq_partition, imbalance], -1),0)
    return torch.cat([x, additional_features], -1)
    
#Define dataset class
# data row number (batch_slice)
batch_slice = 1000 #Usually 128, probably cause of different preprocessing?
class LeapDataset(Dataset):
    def __init__(self, x_features, y_features, y_weights):
        self.x_features =x_features
        self.x_split = len(x_features)
        self.y_features = y_features
        self.y_weights = y_weights

    def __getitem__(self, idx):
        data = torch.load(DATA_MAP[idx])
        x, y = torch.split(data, self.x_split, dim=1)
        y = y*self.y_weights
        x = preprocess_x(x)
        x_mean = X_MEAN; x_std = X_STD; y_mean = Y_MEAN; y_std = Y_STD 
        x = (x - x_mean) / x_std; y = (y - y_mean) / y_std
        x = x.to(torch.float32)
        if order_extra_first:
            x = torch.cat([torch.cat(x[: , 556:].reshape(batch_slice, 9, 60), [x[: , :360].reshape(batch_slice, 6, 60), x[: , 376:556].reshape(batch_slice, 3, 60)], dim=1)
                .permute(0,2, 1), x[: , 360:376].unsqueeze(1).repeat(1,60, 1).view(batch_slice, 60, 16),],-1,)
        else:
            x = torch.cat([torch.cat([x[: , :360].reshape(batch_slice, 6, 60), x[: , 556:].reshape(batch_slice, 9, 60), x[: , 376:556].reshape(batch_slice, 3, 60)], dim=1)
                .permute(0,2, 1), x[: , 360:376].unsqueeze(1).repeat(1,60, 1).view(batch_slice, 60, 16),],-1,)
        y = y.to(torch.float32)
        return x, y
        
    def __len__(self):
        return NUM_ROWS
        
ds_data = LeapDataset(x_features=FEATURE_NAMES, y_features=TARGET_NAMES, y_weights=torch.tensor(TARGET_WEIGHTS))
ds_train, ds_valid = random_split(ds_data, [len(ds_data)-2000, 2000])

batch = 1 #Set batch size to be 8 files, each containing 1000 datapoints #HB had this set to 8, for testing purposes I changed it to 1 b/c of limited ram, etc... 
train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=16, prefetch_factor=4) #Num workers 32
valid_loader = DataLoader(ds_valid, batch_size=1, shuffle=False, drop_last=False, pin_memory=False, persistent_workers=True, num_workers=16)

def r2_score(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()
    
JOINT_MASK = MASK * ADJUSTMENT_MASK

# Parameters
lr_rate = 1.2e-4 #Starting learning rate for training
ERR = 1e-6
class MLP(LightningModule):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

class LeapModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.encoder = MLP([input_dim * 60, input_dim * 15, input_dim * 8, input_dim * 4])
        self.decoder = MLP([input_dim * 4, input_dim * 8, input_dim * 15, input_dim * 60])
        self.lstm = nn.LSTM(input_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=0.07, bidirectional=True)
        self.lstm2 = nn.GRU(hidden_dim * 2, 16, batch_first=True, dropout=0.0, bidirectional=True)
        self.fc_lstm = nn.Linear((32) * 60 , output_dim)
        self.criterion = nn.HuberLoss(delta=1)

    def forward(self, x):
        x_0 = self.encoder(x.view(x.size(0), -1))
        x_1 = self.decoder(x_0)
        x_1 = x_1.view(x.size(0), 60, -1)
        lstm_out, _ = self.lstm(torch.cat([x, x_1], dim = -1))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.contiguous().view(x.size(0), -1) 
        output = self.fc_lstm(lstm_out)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(1)*y.size(0), y.size(2))
        x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        y_pred = self(x)
        if mask:
            loss = self.criterion(y_pred[:, JOINT_MASK==1], y[:, JOINT_MASK==1])
        else:
            loss = self.criterion(y_pred, y)
        loss *= 4 #This was empirically found to improve training 
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(1)*y.size(0), y.size(2))
        x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        y_pred = self(x)
        y_std = Y_STD.to(y.device)
        y_mean = Y_MEAN.to(y.device)
        y = (y * y_std) + y_mean
        y_pred[:, y_std < (1.1 * ERR)] = 0
        y_pred = (y_pred * y_std) + y_mean
        val_score = r2_score(y_pred, y)
        self.log('val_score', val_score, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        if mask:
            y_pred[:, ADJUSTMENT_MASK==0] = y[:, ADJUSTMENT_MASK==0]
            y_pred[:, MASK==0] = 0
            y[:, MASK==0] = 0
            masked_val_score = r2_score(y_pred, y)
            self.log('masked_val_score', masked_val_score, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        return val_score

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr_rate, weight_decay=5e-4)
        milestones = [1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23] 
        gamma = 0.9
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return [optimizer], [scheduler]
        

#Input dimension is 25+9 (the base 25 columns and the additional 9 columns of engineered features)
params = { "input_dim": 25+9, "hidden_dim": 512, "output_dim": 368, "num_layers": 3}
model = LeapModel(input_dim=params["input_dim"], hidden_dim=params["hidden_dim"], output_dim=params["output_dim"], num_layers=params["num_layers"])

every_epoch_checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    filename="epoch_{epoch:02d}-{val_score:.4f}",
    dirpath="/output",
    save_top_k=-1, 
)

trainer = Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=1,
    enable_checkpointing=True,
    precision=16,
    callbacks=[every_epoch_checkpoint_callback],
    val_check_interval=0.5
)

if finetune:
    path = "/.pth"
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_loader, valid_loader)