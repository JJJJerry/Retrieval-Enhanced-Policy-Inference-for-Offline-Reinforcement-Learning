import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
def compute_rtg(rewards):
    rtg=np.zeros_like(rewards)
    temp=0
    length=rewards.shape[0]
    for i in reversed(range(length)):
        temp+=rewards[i]
        rtg[i]=temp
    return rtg
def traj2dataset(trajectory):
    
    states=[]
    actions=[]
    rtgs=[]
    rewards=[]
    timesteps=[]
    done_idxs=[0]
    done_idx_now=0
    for traj in trajectory:
        traj_len=len(traj['actions'])
        states.append(traj['observations'])
        actions.append(traj['actions'])
        rtgs.append(compute_rtg(traj['rewards']))
        rewards.append(traj['rewards'])
        timesteps.append(np.arange(0,traj_len))
        done_idx_now+=traj_len
        done_idxs.append(done_idx_now) #traj的最大时间步+1
        
        
    states=np.concatenate(states,axis=0)
    actions=np.concatenate(actions,axis=0)
    rtgs=np.concatenate(rtgs,axis=0)
    rewards=np.concatenate(rewards,axis=0)
    timesteps=np.concatenate(timesteps,axis=0)
    
    return states,actions,rtgs,rewards,timesteps,done_idxs
class DT_Dataset(Dataset):
    def __init__(self,trajectories,context_length):        
        self.states,self.actions,self.rtgs,self.rewards,self.timesteps,self.done_idxs=traj2dataset(trajectories)
        self.train_state_mean, self.train_state_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6
        self.context_length=context_length
        self.state_dim=self.states.shape[1]
        self.act_dim=self.actions.shape[1]
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        start_idx = idx - self.context_length + 1
        for i,done_idx in enumerate(self.done_idxs):
            if  self.done_idxs[i+1]>idx:
                start_idx = max(start_idx, self.done_idxs[i])

                break
        end_idx = idx + 1
        length=end_idx-start_idx
        states = torch.tensor(np.array(self.states[start_idx: end_idx]), dtype=torch.float32)
        states = (states-self.train_state_mean)/self.train_state_std
        states = torch.concatenate([torch.zeros((self.context_length-length,self.state_dim)),states])
        actions = torch.tensor(self.actions[start_idx:end_idx], dtype=torch.float32)
        actions = torch.concatenate([torch.zeros((self.context_length-length,self.act_dim)),actions])
        rtgs = torch.tensor(self.rtgs[start_idx:end_idx], dtype=torch.float32).unsqueeze(1)
        rtgs = torch.concatenate([torch.zeros((self.context_length-length,1)),rtgs])
        timesteps = torch.tensor(self.timesteps[start_idx:end_idx], dtype=torch.long)
        timesteps = torch.concatenate([torch.zeros((self.context_length-length)),timesteps])
        attention_mask = torch.concatenate([torch.zeros(self.context_length-length),torch.ones(length)])
        info = {
            'state':self.states[idx],
            'action':self.actions[idx],
            'reward':self.rewards[idx],
            'rtg':rtgs[-1].numpy(),
            'timestep':self.timesteps[idx]
        }
        return states, actions, rtgs, timesteps, attention_mask,info


