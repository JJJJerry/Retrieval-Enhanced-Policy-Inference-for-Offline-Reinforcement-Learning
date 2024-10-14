import argparse
import os
import d4rl
import gym
import numpy as np
import torch
import pickle
import sys
sys.path.append('/data/wangchunhao-slurm/workspace/code/projects/decision-transformer-master/gym/cql/')
from cql.rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from tqdm import tqdm
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='0'



def eval_model(env,model,max_path_length):
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    next_o = None
    path_length = 0
    o = env.reset()
    model.reset()
    while path_length < max_path_length:
        a, _ = model.get_action(o)
        next_o, r, d, env_info = env.step(a)
        
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
parser.add_argument('--env', type=str, default='antmaze-medium-play-v0')
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--cql_seed', type=str, default='0')
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

state_dim=env.observation_space.shape[0]
act_dim=env.action_space.shape[0]


weight_path_dir=f'cql/cql_weights/{args.env}'
model_list=[]
M=256
policy = TanhGaussianPolicy(
        obs_dim=state_dim,
        action_dim=act_dim,
        hidden_sizes=[M, M, M], 
)

weight_path=f'{weight_path_dir}/policy_{args.cql_seed}.pth'
policy = TanhGaussianPolicy(
    obs_dim=state_dim,
    action_dim=act_dim,
    hidden_sizes=[M, M, M], 
)
policy.load_state_dict(torch.load(weight_path))
cql_policy = MakeDeterministic(policy)

r_list=[]
for i in tqdm(range(30)):
    r=eval_model(env,cql_policy,1000)
    r_list.append(r)
    print(r)

print(f'平均returns: {np.array(r_list).mean()}')