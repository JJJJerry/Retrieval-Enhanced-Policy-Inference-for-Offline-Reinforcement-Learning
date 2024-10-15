import d3rlpy
import argparse
import numpy as np
from mingpt.utils import set_seed
import pickle
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='breakout')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:1')
args=parser.parse_args()
game=args.game.lower()
seed=args.seed
device=args.device
set_seed(seed)

with open(f'd3rl_dataset_cache/{game}.pkl','rb') as f:
    dataset=pickle.load(f)
with open(f'd3rl_dataset_cache/{game}_env.pkl','rb') as f:
    env=pickle.load(f)
""" dataset,env=d3rlpy.datasets.get_atari_transitions(
    game_name=game,
    fraction=0.01,
    num_stack=4
) """
print('Env and Dataset is loaded')


cql=d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=5e-5,
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        batch_size=32,
        alpha=4.0,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
            n_quantiles=200
        ),
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        target_update_interval=2000,
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
        ).create(device=args.device)
cql.build_with_dataset(dataset)
cql.build_with_env(env)
cql.load_model(f'cql_weights/{game}/cql_{game}_{seed}.pt')
r_list=[]

print('start to eval')
for i in range(10):
    reward_sum=0
    observation,info = env.reset()
    j=0
    while True:
        x=observation.reshape(1,4,84,84)
        action = cql.predict(x)[0]
        observation, reward, done, _ ,info = env.step(action)
        reward_sum+=reward
        j+=1
        if done or j>=10000:
            break
    print(reward_sum)
    r_list.append(reward_sum)
print(f'avg:  {np.array(r_list).mean()}')
