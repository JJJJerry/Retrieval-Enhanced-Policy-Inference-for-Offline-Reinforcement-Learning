import argparse
import os
import d4rl
import gym
import numpy as np
import pickle
from retrieverV2 import ManyModel,RetrieverRLV2,KNN_DT_Retriever
from DecisionTransformerV2 import MLPBCModelV2
from tqdm import tqdm
import pandas as pd
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
        a = model.get_action(o)
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
parser.add_argument('--index_type', type=str, default='inner') #l2,inner
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--retrieval_only', action='store_true')
parser.add_argument('--topk', type=int, default=128)
parser.add_argument('--model_num', type=int, default=4)
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
'max_ep_len':1000,
'embed_dim':128,
'n_layer':3,
'n_head':1,
'activation_function':'relu',
'dropout':0.1,
'device':args.device,
'scale':scale,
'topk':args.topk,
'dataset_path':f'data/{args.env}.pkl',
'target_return':target_return,
'states_mean':states_mean,
'states_std':states_std
}

weight_path_dir=f'bc_weights/{args.env}'
model_list=[]
for i in tqdm(range(args.model_num)):  
    weight_path=f'{weight_path_dir}/bc_{str(i)}.pt'
    dt=MLPBCModelV2(bc_weight_path=weight_path,kwargs=kwargs)
    model_list.append(dt)

model=ManyModel(model_list,act_dim=act_dim)


dt_weight_path=os.path.join('retriever_weight',args.env,f'{args.env}.pt')
index_dir_path=f'index/{args.index_type}/{args.env}'
retriever=KNN_DT_Retriever(dt_weight_path,index_dir_path,args.index_type,**kwargs)
model=RetrieverRLV2(retriever=retriever,model=model,lamb=None,solo=False,retrieval_only=False,kwargs=kwargs)


r_list=[]
for i in tqdm(range(100)):
    r=eval_model(env,model,1000)
    r_list.append(r)
    print(r)
    print(np.array(model.lamb_record).mean())
    model.lamb_record=[]
print(f'avg returns: {np.array(r_list).mean()}')