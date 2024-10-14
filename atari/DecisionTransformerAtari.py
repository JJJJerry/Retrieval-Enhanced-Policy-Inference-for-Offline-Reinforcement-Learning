import torch
from torch.nn import functional as F
import numpy as np
import pickle
from mingpt.model_atari import GPT,GPTConfig
from mingpt.utils import sample,get_prob
class DecisionTransformerAtari:
    def __init__(self,dt_weight_path,model_conf:GPTConfig):
        self.dt=GPT(model_conf)
        self.config=model_conf
        self.reset()
        self.dt.load_state_dict(torch.load(dt_weight_path,map_location=torch.device('cpu')))
        self.dt.to(self.device)
        self.dt.train(False)
    def update_history(self,action,reward):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['target_return']-=reward
    def reset(self):
        self.history={
            'states':[],
            'actions':[],
            'rewards':[],
            'rtgs':[],
            'time_now':0,
            'target_return':self.kwargs['target_return']
        }
    @torch.no_grad()
    def get_prob(self,state):
        self.history['states'].append(state)
        self.history['rtgs'].append(self.history['target_return'])

        states = np.stack(self.history['states'])
        states = torch.from_numpy(states[-self.maxlen:])
        actions = torch.from_numpy(self.history['actions'])
        rtgs = np.stack(self.history['rtgs'], axis=0,dtype=np.float32)

        self.history['time_now']+=1
        prob=get_prob(self.model, states, 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(self.history['time_now'], self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        return prob
