import torch
import numpy as np
import pickle
import copy
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
class DecisionTransformerV2:
    def __init__(self,dt_weight_path,kwargs):
        self.dt=DecisionTransformer(
            state_dim=kwargs['state_dim'],
            act_dim=kwargs['act_dim'],
            max_length=kwargs['K'],
            max_ep_len=kwargs['max_ep_len'],
            hidden_size=kwargs['embed_dim'],
            n_layer=kwargs['n_layer'],
            n_head=kwargs['n_head'],
            n_inner=4*kwargs['embed_dim'],
            activation_function=kwargs['activation_function'],
            n_positions=1024,
            resid_pdrop=kwargs['dropout'],
            attn_pdrop=kwargs['dropout'],
        )
        self.kwargs=kwargs
        self.device=kwargs['device']
        self.reset()
        self.dt.load_state_dict(torch.load(dt_weight_path,map_location=torch.device('cpu')))
        self.dt.to(self.device)
        self.dt.eval()
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
            'timesteps':[],
            'time_now':0,
            'target_return':self.kwargs['target_return']
        }
    @torch.no_grad()
    def get_action(self,state):
        self.history['states'].append(state)
        self.history['timesteps'].append(self.history['time_now'])
        self.history['rtgs'].append(self.history['target_return'])
        self.history['time_now']+=1
        
        states = np.stack(self.history['states'])
        
        states = (states-self.kwargs['states_mean'])/self.kwargs['states_std']
        actions = np.concatenate([np.array(self.history['actions']).reshape(-1,self.kwargs['act_dim']), np.zeros((1, self.kwargs['act_dim']))], axis=0)
        # not need rewards
        rewards = np.concatenate([np.array(self.history['rewards']).reshape(-1,1), np.zeros((1,1))], axis=0) 
        rtgs = np.stack(self.history['rtgs'], axis=0,dtype=np.float32)
        rtgs/=self.kwargs['scale']
        timesteps =np.array(self.history['timesteps']).reshape(-1,1)
        


        states = torch.from_numpy(states[-self.kwargs['K']:]).reshape(1,-1,self.kwargs['state_dim']).to(device=self.kwargs['device'],dtype=torch.float32)
        actions = torch.from_numpy(actions[-self.kwargs['K']:]).reshape(1,-1,self.kwargs['act_dim']).to(device=self.kwargs['device'],dtype=torch.float32)
        returns_to_go = torch.from_numpy(rtgs[-self.kwargs['K']:]).reshape(1,-1,1).to(device=self.kwargs['device'],dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps[-self.kwargs['K']:]).reshape(1,-1).to(device=self.kwargs['device'],dtype=torch.long)

        action=self.dt.get_action(states, actions, None, returns_to_go, timesteps).cpu().numpy().reshape(1,-1)
        return action
class MLPBCModelV2:
    def __init__(self,bc_weight_path,kwargs):
        self.bc=MLPBCModel(
            state_dim=kwargs['state_dim'],
            act_dim=kwargs['act_dim'],
            hidden_size=kwargs['embed_dim'],
            n_layer=kwargs['n_layer'],
            dropout=kwargs['dropout'],
            max_length=kwargs['K']
        )
        self.kwargs=kwargs
        self.device=kwargs['device']
        self.reset()
        self.bc.load_state_dict(torch.load(bc_weight_path,map_location=torch.device('cpu')))
        self.bc.to(self.device)
        self.bc.eval()   
    def update_history(self,action,reward):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
    
    def reset(self):
        self.history={
            'states':[],
            'actions':[],
            'rewards':[],
        }
    @torch.no_grad()
    def get_action(self,state):
        self.history['states'].append(state)
        states = np.stack(self.history['states'])
        states = (states-self.kwargs['states_mean'])/self.kwargs['states_std']
        actions = np.concatenate([np.array(self.history['actions']).reshape(-1,self.kwargs['act_dim']), np.zeros((1, self.kwargs['act_dim']))], axis=0)
       

        rewards = np.concatenate([np.array(self.history['rewards']).reshape(-1,1), np.zeros((1,1))], axis=0) 
        states = torch.from_numpy(states[-self.kwargs['K']:]).reshape(1,-1,self.kwargs['state_dim']).to(device=self.kwargs['device'],dtype=torch.float32)
        actions = torch.from_numpy(actions[-self.kwargs['K']:]).reshape(1,-1,self.kwargs['act_dim']).to(device=self.kwargs['device'],dtype=torch.float32)
       
        action = self.bc.get_action(states,actions,rewards).cpu().numpy().reshape(1,-1)
        return action