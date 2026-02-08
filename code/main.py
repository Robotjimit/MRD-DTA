import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx

import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
from utils import *
import gc
from dataset import DTA_Dataset
from sklearn.model_selection import KFold
from model import *
from torch import nn as nn
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from config import ModelConfig, TrainingConfig, DeviceConfig, DataSplitConfig, PathConfig, get_model_config, get_training_config
import time
#############################################################################

# 使用配置文件中的参数
device = torch.device(DeviceConfig.device)
LR = TrainingConfig.learning_rate
# NUM_EPOCHS = TrainingConfig.num_epochs
NUM_EPOCHS = 70
batch_size = TrainingConfig.batch_size
dataset_name = TrainingConfig.dataset_name
fold_start = 0
vid = 1
#############################################################################
# df = pd.read_csv('pdbbind.csv')
df = pd.read_csv(f'./{dataset_name}/{dataset_name}_processed.csv')
smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])
target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']


drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

tar_len = 1000
seq_len = 540

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)



data_set = f"{dataset_name}_processed"
mode_name = DataSplitConfig.default_mode



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "Total": total_params/1e6,
        "Trainable": trainable_params/1e6
    }

for fold in range(fold_start, TrainingConfig.num_folds):
    print("Building model...")
    # 使用配置文件中的模型参数
    model_config = get_model_config()
    model = DMFF(**model_config).to(device)
    def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_m = count_params(model) / 1e6
    print(f"Params: {params_m:.2f}M")
    
    param_stats = count_parameters(model)
    print(f"Total Params: {param_stats['Total']:,}")
    print(f"Trainable Params: {param_stats['Trainable']:,}")
    # load model
    model_file_name = PathConfig.model_dir + dataset_name + '_' + mode_name + '_' +str(fold)+'_v1_128_T_0_2.pt'
    # 加载状态字典
    if os.path.exists(model_file_name):
        save_model = torch.load(model_file_name)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TrainingConfig.scheduler_factor, 
        patience=TrainingConfig.scheduler_patience, 
        min_lr=TrainingConfig.scheduler_min_lr,
        verbose=True
    )
    tr_times = []
    in_times = []
    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1
    print(f"Fold {fold + 1}")
    log(f'train on {dataset_name}_{mode_name}')
    train_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'train.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)
    val_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'valid.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)
    test_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'test.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
    for epoch in range(NUM_EPOCHS):
        print("No {} epoch".format(epoch))
        start = time.time()
        train(model, train_loader, optimizer, epoch)
        end = time.time()
        tr_times.append(end - start)
        
        start = time.time()
        G, P = predicting(model, test_loader)
        end = time.time()
        in_times.append(end - start)
        
        val1 = get_mse(G, P)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if model_file_name is not None:
                torch.save(model.state_dict(), model_file_name)
            log(f'mse improved at epoch {best_epoch}, best_mse {best_mse}')
        else:
            log(f'current mse: {val1} , No improvement since epoch {best_epoch}, best_mse {best_mse}')
        schedule.step(val1)

#         if epoch % vid == 0:
#             G, P = predicting(model, val_loader)
#             cindex, rm2, mse = calculate_metrics_and_return(G, P, val_loader)
#             # mse = get_mse(G, P)
#             ci = concordance_index(G, P)
#             # r2 = r2_score(G, P)
            
#             if mse < best_test_mse:
#                 best_test_mse = mse
#                 log(f'epoch {epoch} val mse:{mse}, r2:{rm2}, ci:{ci}')
#                 # file_name = PathConfig.model_dir + dataset_name + '_' + str(epoch) + '.pt'
#                 if model_file_name is not None:
#                     torch.save(model.state_dict(), model_file_name)
#                 print(f"epoch {epoch}:mse {mse} cindex {cindex} rm2 {rm2}")
    # in_times = np.array(in_times)/len(val_dataset)
    # tr_times = np.array(tr_times)
    # print(f"Inference: {in_times.mean()*1000:.2f} ms ± {in_times.std()*1000:.2f} ms per batch")
    # print(f"Train time / epoch: {tr_times.mean():.2f}s ± {tr_times.std():.2f}s")

    save_model = torch.load(model_file_name)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    G, P = predicting(model, test_loader)
    ci = concordance_index(G, P)
    cindex, rm2, mse = calculate_metrics_and_return(G, P, test_loader)
    log(f'T 0.2 test mse: {mse} , rm2: {rm2}, cindex: {ci}')
  
    print(f"{mse} {ci} {rm2}")
    
    break

