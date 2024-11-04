from matplotlib import pyplot as plt
import wandb
import os
import numpy as np
import pandas as pd
from config.config import load_config
from src.data.datasplit import DataSplitUtils
from src.data.datamodule import DyanmicsDataset
from scripts.Seq2SeqModel import Seq2SeqModel
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
pl.seed_everything(42)
config = load_config('config/debug.yaml')
# Read dataset
data_class = DataSplitUtils(config['data']['dataset_path'])
data = data_class.load_data(list(config['data']['modalities'].keys()))

# split into trian val test
data_train, data_test, data_val = data_class.random_train_test_split(test_size=config['train']['train_split'])
train_dataset = DyanmicsDataset(config,data_train, True)
test_dataset =DyanmicsDataset(config,data_test, True)
val_dataset=DyanmicsDataset(config,data_val, True)
print(len(train_dataset))
print(len(test_dataset))
print(len(val_dataset))

# create dataloaders
# create model
# epochs = config['train']['epochs']
# batch_size = config['train']['batch_size']
# for epoch in range(epochs):
#     for i in range(0, )
# train val log 
# test

