import os
import argparse
import pickle
from mingpt.model_atari import GPT
from mingpt.utils import set_seed,get_prob,sample
import torch
import numpy as np
import faiss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from faiss import normalize_L2
from torch.nn import functional as F
from mingpt.trainer_atari import Env,Args

class RetrievalData:
    def __init__(self,d,i,a,r=0) -> None:
        self.d=d
        self.i=i
        self.a=a
        self.r=r
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
    
class RetrievalRL:
    def __init__(self,retrieval_model:GPT,model_list,config) -> None:
        self.model_list=model_list
        self.device=config['device']
        self.index_type=config['index_type']
        self.config=config
        self.lamb_list=[]
        self.retrieval_model=retrieval_model
        self.init_retrieval_data()
    def get_block_size(self):
        return self.model_list[0].get_block_size()
    def eval(self):
        for model in self.model_list:
            model.eval()
    def init_retrieval_data(self):
        if not os.path.exists(self.config['index_dir_path']):
            self.make_retrieval_dataset()
        
        index_path=os.path.join(self.config['index_dir_path'],f'{self.index_type}.index')
        res = faiss.StandardGpuResources()
        self.index=faiss.read_index(index_path)
        try:
            self.index=faiss.index_cpu_to_gpu(res, int(self.device[-1]), self.index) 
        except:
            print('index to gpu error')
        
        with open(os.path.join(config['index_dir_path'],'target_actions.pkl'),'rb') as f:
            self.target_action=pickle.load(f)
        with open(os.path.join(config['index_dir_path'],'rtgs.pkl'),'rb') as f:
            self.rtg=pickle.load(f)
       
        print(f'There are {len(self.target_action)} pieces of data')
        print('Retrieval dataset is loaded')
    def make_retrieval_dataset(self):

        data_path=self.config['retrieval_dataset_dir_path']
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
        dataset = StateActionReturnDataset(obss, self.config['context_length']*3, actions, done_idxs, rtgs, timesteps)
    
        loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=6,pin_memory=True)
        vectors=[]
        target_actions=[]
        rtgs=[]
        for (x, y, r, t) in tqdm(loader):
            with torch.no_grad():
              
                x = x.to(self.config['device'])
                y = y.to(self.config['device'])
                r = r.to(self.config['device'])
                t = t.to(self.config['device'])
                
                encoded_states=self.retrieval_model.encode(x, y, None, r, t).cpu().numpy().reshape(-1,self.config['hidden_dim'])
                y=y.cpu().numpy()[:,-1]
                rtgs.append(r[:,-1].cpu().numpy())
                target_actions.append(y)
                vectors.append(encoded_states)
    
        if not os.path.exists(self.config['index_dir_path']):
            os.makedirs(self.config['index_dir_path'])
        
        #vectors=vectors.astype(np.float32)
        vectors=np.concatenate(vectors)
        target_actions=np.concatenate(target_actions)
        rtgs=np.concatenate(rtgs)
        
        target_actions_path=os.path.join(self.config['index_dir_path'],'target_actions.pkl')
        rtgs_path=os.path.join(self.config['index_dir_path'],'rtgs.pkl')
            
        with open(target_actions_path,'wb') as f:
            pickle.dump(target_actions,f)
        
        with open(rtgs_path,'wb') as f:
            pickle.dump(rtgs,f)
        
        vectors=vectors.astype(np.float32)
        if self.index_type=='inner':
            index=faiss.index_factory(self.config['hidden_dim'],'Flat',faiss.METRIC_INNER_PRODUCT) #hidden dim
            normalize_L2(vectors) 
        elif self.index_type=='l2':
            index=faiss.index_factory(self.config['hidden_dim'],'Flat',faiss.METRIC_L2) #hidden dim
        else :
            raise NotImplementedError

        index.add(vectors)
        index_path=os.path.join(self.config['index_dir_path'],f'{self.index_type}.index')
        faiss.write_index(index,index_path) 
        print(f'Index saved to {index_path}')
    
    def get_model_prob(self,states,actions,rtgs,timesteps): 
        probs_list=[]
        with torch.no_grad():
            for model in self.model_list:
                logits, _ = model(states, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
                logits = logits[:, -1, :]
                prob = F.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(prob)
        probs=np.stack(probs_list,axis=0).reshape(len(self.model_list),-1)
        probs_mean = probs.mean(axis=0) 
        probs_norm = probs/(np.linalg.norm(probs,axis=1).reshape(-1,1))
        mean = probs_norm.mean(axis=0) 
        z_scores = np.abs(probs_norm-mean).mean(axis=0).sum()
        return probs_mean,z_scores
    def get_retrieval_prob(self,states,actions,rtgs,timesteps):
        with torch.no_grad():
            hidden_states=self.retrieval_model.encode(states, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps).cpu().numpy().reshape(1,-1)
        if self.index_type=='inner':
            normalize_L2(hidden_states)
        D,I=self.index.search(hidden_states,self.config['topk'])
        knn_probs=np.zeros(max(self.target_action)+1)
        if self.index_type=='inner':
            D=1-D
       
        I=I[0] 
        D=D[0]
        
        if D.shape[0]==0:
            return knn_probs,False
        retrievals=[]
        for i in range(D.shape[0]):
            data=RetrievalData(D[i],I[i],self.target_action[I[i]],self.rtg[I[i]])
            retrievals.append(data)

        weights=np.zeros((len(retrievals)))
        for i,data in enumerate(retrievals):
            weights[i]=1/data.d
        weights=weights/weights.sum()
        for i in range(len(retrievals)):
            ind=retrievals[i].i
            a=int(self.target_action[ind])
            knn_probs[a]+=weights[i]
        return knn_probs,True
    def get_lamb(self,z):
        return 1-np.exp(-z)
    def avg_lamb(self):
        return np.array(self.lamb_list).mean()
    def get_prob(self,states,actions,rtgs,timesteps):
        model_prob,z=self.get_model_prob(states,actions,rtgs,timesteps)
    
        lamb=self.get_lamb(z)
        self.lamb_list.append(lamb)
  
        retrieval_prob,is_retrieval=self.get_retrieval_prob(states,actions,rtgs,timesteps)
        if args.retrieval_only:
            return retrieval_prob
        
        if is_retrieval==False:
            return model_prob
        prob=retrieval_prob*lamb+model_prob*(1-lamb)
    
        return prob


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='Qbert')
parser.add_argument('--model_type', type=str, default='dt')
parser.add_argument('--index_type', type=str, default='inner')
parser.add_argument('--model_num', type=int, default=4)
parser.add_argument('--topk', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--retrieval_only', action='store_true')

args = parser.parse_args()
device=args.device

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

model_list=[]
if args.model_type=='dt':
    gpt_config_path=os.path.join('weights',args.game,'gpt_config.pkl')
elif args.model_type=='bc':
    gpt_config_path=os.path.join('bc_weights',args.game,'gpt_config.pkl')
with open(gpt_config_path,'rb') as f:
    gpt_config = pickle.load(f)
with open(f'dataset/{args.game}/timesteps.pkl','rb') as f:
    timesteps=pickle.load(f)

retrieval_model=GPT(gpt_config)
retrieval_model_path=os.path.join('retrieval_weights',args.game,f'{args.game}.pt')
retrieval_model.load_state_dict(torch.load(retrieval_model_path,map_location='cpu'))
retrieval_model.to(device=device)

for i in range(args.model_num):
    if args.model_type=='dt':
        weight_path=os.path.join('weights',args.game,f'{args.game}_{i}.pt')
    elif args.model_type=='bc':
        weight_path=os.path.join('bc_weights',args.game,f'{args.game}_{i}.pt')
    policy=GPT(gpt_config)
    policy.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
    policy.to(device=device)
    model_list.append(policy)

retrieval_dataset_dir_path=os.path.join('dataset',args.game)
index_dir_path=os.path.join('index',args.game,args.index_type)
config={
    "retrieval_dataset_dir_path":retrieval_dataset_dir_path,
    "index_dir_path":index_dir_path,
    "device":device,
    "index_type":args.index_type,
    "context_length":context_length,
    "hidden_dim":gpt_config.n_embd,
    'topk':args.topk
}
model=RetrievalRL(retrieval_model,model_list,config)


env_args=Args(args.game.lower(), args.seed,device)
env = Env(env_args)
env.eval()

ret=target_return
T_rewards, T_Qs = [], []
done = True
eval_num=10

for i in range(eval_num):
    state = env.reset()

    all_states = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    done = False
    rtgs = [ret]
    actions = []
    j=0
    reward_sum=0
    while not done:
        if j==0:
            env.reset()
            prob=model.get_prob(all_states[-context_length:].unsqueeze(0),
                            actions=None,
                            rtgs=torch.tensor(rtgs[-context_length:], dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                            timesteps=min(j, max(timesteps)) * torch.ones((1, 1, 1), dtype=torch.int64).to(device))
        else :
            prob = model.get_prob(
                all_states[-context_length:].unsqueeze(0),
                actions=torch.tensor(actions[-context_length:], dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                rtgs=torch.tensor(rtgs[-context_length:], dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=(min(j, max(timesteps)) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
        
        prob = torch.from_numpy(prob)
        sampled_action=torch.multinomial(prob, num_samples=1)
        action = sampled_action.cpu().item()
        actions += [sampled_action]
        state, reward, done = env.step(action)
        state = state.unsqueeze(0).unsqueeze(0).to(device)
        reward_sum += reward
        j += 1
        if j>=max(timesteps):
            break
        all_states = torch.cat([all_states, state], dim=0)
        rtgs += [rtgs[-1] - reward]
    print('return: ',reward_sum)
    T_rewards.append(reward_sum)
print(f'avg:  {np.array(T_rewards).mean()}')

