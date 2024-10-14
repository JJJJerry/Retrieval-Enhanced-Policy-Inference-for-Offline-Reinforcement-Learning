import d4rl
import gym
import numpy as np
import torch
import argparse
import os
from DecisionTransformerV2 import MLPBCModelV2
import tqdm
import pickle

from tqdm import tqdm



def eval_model(env,model,max_path_length):
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    next_o = None
    path_length = 0
    o = env.reset()
    policy.reset()
    while path_length < max_path_length:
        a= model.get_action(o)
        next_o, r, d, env_info = env.step(a.reshape(-1))
        model.update_history(a,r)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    return sum(rewards)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hopper-medium-v2')
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--bc_seed', type=str, default='0')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
if args.env.find('hopper')!=-1:
    env = gym.make('Hopper-v3')
    target_return=3600
    scale=1000
    
elif args.env.find('halfcheetah')!=-1:
    env = gym.make('HalfCheetah-v3')
    target_return=12000
    scale=1000
   
elif args.env.find('walker2d')!=-1:
    env = gym.make('Walker2d-v3')
    target_return = 5000
    scale=1000
    
elif args.env.find('antmaze')!=-1:
    env = gym.make('antmaze-medium-play-v0')
    target_return = 1
    scale=1
elif args.env.find('maze2d-open-dense-v0')!=-1:
    env = gym.make('maze2d-open-dense-v0')
    target_return = 350
    scale=100
elif args.env.find('maze2d-umaze-dense-v1')!=-1:
    env = gym.make('maze2d-umaze-dense-v1')
    target_return = 10
    scale=1
elif args.env.find('maze2d-medium-dense-v1')!=-1:
    env = gym.make('maze2d-medium-dense-v1')
    target_return = 30
    scale=10
elif args.env.find('maze2d-large-dense-v1')!=-1:
    env = gym.make('maze2d-large-dense-v1')
    target_return = 50
    scale=10
    
state_dim=env.observation_space.shape[0]
act_dim=env.action_space.shape[0]

dt_train_dataset_path=f'data/{args.env}.pkl'
with open(dt_train_dataset_path, 'rb') as f:
    train_trajectories = pickle.load(f)
states = []
for path in train_trajectories:
    states.append(path['observations'])
states = np.concatenate(states, axis=0)
states_mean,states_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6


kwargs={
'state_dim':state_dim,
'act_dim':act_dim,
'K':20,
'embed_dim':128,
'n_layer':3,
'dropout':0.1,
'device':args.device,
'scale':scale,
'topk':100,
'dataset_path':f'data/{args.env}.pkl',
'target_return':target_return,
'states_mean':states_mean,
'states_std':states_std
}

weight_path_dir=f'bc_weights/{args.env}'
weight_path=f'{weight_path_dir}/bc_{args.bc_seed}.pt'
policy=MLPBCModelV2(
            weight_path,
            kwargs
)
r_list=[]
for i in tqdm(range(30)):
    r=eval_model(env,policy,1000)
    r_list.append(r)
    print(r)

print(f'平均returns: {np.array(r_list).mean()}')