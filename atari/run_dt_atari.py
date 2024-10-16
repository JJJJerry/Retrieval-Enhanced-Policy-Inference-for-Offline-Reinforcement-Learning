import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
import os
import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
#
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./atari-replay-datasets/dqn/')
parser.add_argument('--dataset_save_dir', type=str, default='dataset')
parser.add_argument('--ckpt_path', type=str, default='weights')
parser.add_argument('--device', type=str, default='cuda:7')
args = parser.parse_args()

        
set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        return states, actions, rtgs, timesteps
    
data_path=os.path.join(args.dataset_save_dir,args.game)
print('create dataset...')
if os.path.exists(data_path):
    obss_path=os.path.join(data_path,'obss.pkl')
    with open(obss_path,'rb') as f:
        obss=pickle.load(f)
    actions_path=os.path.join(data_path,'actions.pkl')
    with open(actions_path,'rb') as f:
        actions=pickle.load(f)
    returns_path=os.path.join(data_path,'returns.pkl')
    with open(returns_path,'rb') as f:
        returns=pickle.load(f)
    done_idxs_path=os.path.join(data_path,'done_idxs.pkl')
    with open(done_idxs_path,'rb') as f:
        done_idxs=pickle.load(f)
    rtgs_path=os.path.join(data_path,'rtgs.pkl')
    with open(rtgs_path,'rb') as f:
        rtgs=pickle.load(f)
    timesteps_path=os.path.join(data_path,'timesteps.pkl')
    with open(timesteps_path,'rb') as f:
        timesteps=pickle.load(f)
else :
    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)
    os.makedirs(data_path)
    obss_path=os.path.join(data_path,'obss.pkl')
    with open(obss_path,'wb') as f:
        pickle.dump(obss,f)
    actions_path=os.path.join(data_path,'actions.pkl')
    with open(actions_path,'wb') as f:
        pickle.dump(actions,f)
    returns_path=os.path.join(data_path,'returns.pkl')
    with open(returns_path,'wb') as f:
        pickle.dump(returns,f)
    done_idxs_path=os.path.join(data_path,'done_idxs.pkl')
    with open(done_idxs_path,'wb') as f:
        pickle.dump(done_idxs,f)
    rtgs_path=os.path.join(data_path,'rtgs.pkl')
    with open(rtgs_path,'wb') as f:
        pickle.dump(rtgs,f)
    timesteps_path=os.path.join(data_path,'timesteps.pkl')
    with open(timesteps_path,'wb') as f:
        pickle.dump(timesteps,f)
print('成功加载数据集')
# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)
dt_path=os.path.join('weights',args.game,f'{args.game}_{args.seed}.pt')
if args.model_type=='naive':
    with open(os.path.join('bc_weights',args.game,'gpt_config.pkl'),'wb') as f:
        pickle.dump(mconf,f)
elif args.model_type=='reward_conditioned':
    with open(os.path.join('weights',args.game,'gpt_config.pkl'),'wb') as f:
        pickle.dump(mconf,f)

if os.path.exists(dt_path):
    model.load_state_dict(torch.load(dt_path,map_location='cpu'))
    print(f'load model from {dt_path}')
device=args.device
# initialize a trainer instance and kick off training
print('start training')
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),ckpt_path=args.ckpt_path,device=device)
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
