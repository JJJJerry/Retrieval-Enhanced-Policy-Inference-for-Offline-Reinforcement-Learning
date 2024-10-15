import numpy as np
import torch
import os
from mingpt.utils import sample,get_prob
import pickle
import torch
from PIL import Image
from mingpt.model_atari import GPT
from mingpt.trainer_atari import Env,Args
import argparse
from mingpt.utils import set_seed
def get_returns(model,args,device,timesteps,ret):
    model.train(False)
    args=Args(args.game.lower(), args.seed,device)
    env = Env(args)
    env.eval()
    T_rewards, T_Qs = [], []
    eval_num=10
    done = True
    for i in range(eval_num):
        state = env.reset()
        state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        prob = get_prob(model, state, 1, temperature=1.0, sample=True, actions=None, 
            rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))
        sampled_action = torch.multinomial(prob, num_samples=1)
     
        j = 0
        all_states = state
        actions = []
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action.cpu().numpy()[0,-1]
            actions += [sampled_action]
            state, reward, done = env.step(action)
            reward_sum += reward
            j += 1
            if done:
                T_rewards.append(reward_sum)
                print(reward_sum)
                break
            state = state.unsqueeze(0).unsqueeze(0).to(device)
            all_states = torch.cat([all_states, state], dim=0)
            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            prob = get_prob(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=(min(j, max(timesteps)) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
            sampled_action = torch.multinomial(prob, num_samples=1)
            
    env.close()
    eval_return = sum(T_rewards)/eval_num
    print("target return: %d, eval return: %d" % (ret, eval_return))
    return eval_return

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--game', type=str, default='Breakout') #Qbert
parser.add_argument('--device', type=str, default='cuda:2')
args = parser.parse_args()

if args.game=='Breakout':
    context_length=30
    target_return=90
elif args.game=='Qbert':
    context_length=30
    target_return=2500
elif args.game=='Pong':
    context_length=50
    target_return=20
elif args.game=='Seaquest':
    context_length=30
    target_return=1450
    
with open(f'dataset/{args.game}/timesteps.pkl','rb') as f:
    timesteps=pickle.load(f)
dt_path=os.path.join('bc_weights',args.game,f'{args.game}_{args.seed}.pt')

gpt_config_path=os.path.join('bc_weights',args.game,'gpt_config.pkl')
with open(gpt_config_path,'rb') as f:
    gpt_config = pickle.load(f)


set_seed(args.seed)
device=args.device
model = GPT(gpt_config)
model.load_state_dict(torch.load(dt_path,map_location='cpu'))
model=model.to(device)
model.train(False)

env_args=Args(args.game.lower(), args.seed,device)
env = Env(env_args)
env.eval()

ret=target_return
T_rewards, T_Qs = [], []
done = True
eval_num=10
#get_returns(model,args,device,timesteps,ret)

for i in range(eval_num):
    state = env.reset()
    
    all_states = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    done = False
    rtgs = [ret]
    actions = []
    j=0
    reward_sum=0
    while not done:
        all_states=all_states[-context_length:]
        actions=actions[-context_length:]
        rtgs=rtgs[-context_length:]
        if j==0:
            env.reset()
            prob = get_prob(model, state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0), 1, temperature=1.0, sample=True, 
                            actions=None, 
            rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))
        else :
            prob = get_prob(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=(min(j, max(timesteps)) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))  
            
        sampled_action = torch.multinomial(prob, num_samples=1)
        
        action = sampled_action.cpu().numpy()[0,-1]
        actions += [sampled_action]
        state, reward, done = env.step(action)
        state = state.unsqueeze(0).unsqueeze(0).to(device)
        reward_sum += reward
        j += 1
        all_states = torch.cat([all_states, state], dim=0)
        rtgs += [rtgs[-1] - reward]
    print(reward_sum)
    T_rewards.append(reward_sum)
print(f'avg:  {np.array(T_rewards).mean()}')








