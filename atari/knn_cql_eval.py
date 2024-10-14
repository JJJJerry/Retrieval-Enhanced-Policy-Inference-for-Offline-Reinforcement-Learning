import os
import argparse
import pickle
from mingpt.model_atari import GPT
from mingpt.model_atari import GPTConfig
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
import d3rlpy

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
            self.index=faiss.index_cpu_to_gpu(res, int(self.device[-1]), self.index) #使用faiss-gpu
        except:
            print('index to gpu error')
        
        with open(os.path.join(config['index_dir_path'],'target_actions.pkl'),'rb') as f:
            self.target_action=pickle.load(f)
        self.action_space=max(self.target_action)[0]
        with open(os.path.join(config['index_dir_path'],'rtgs.pkl'),'rb') as f:
            self.rtg=pickle.load(f)
        #with open(os.path.join(config.retrieval_dataset_dir_path,'vectors.pkl'),'rb') as f:
        #    self.vectors=pickle.load(f)
        print(f'有{len(self.target_action)}条数据')
        print('检索数据集加载完毕')
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
                #length=np.random.randint(1,args.context_length)
                x = x.to(self.config['device'])
                y = y.to(self.config['device'])
                r = r.to(self.config['device'])
                t = t.to(self.config['device'])
                
                encoded_states=self.retrieval_model.encode(x, y, None, r, t).cpu().numpy().reshape(-1,self.config['hidden_dim'])
                y=y.cpu().numpy()[:,-1]
                rtgs.append(r[:,-1].cpu().numpy())
                target_actions.append(y)
                vectors.append(encoded_states)
                   
                #target_actions=np.concatenate([target_actions,y[:,-1]],axis=0)
                #vectors=np.concatenate([vectors,encoded_states],axis=0)
        if not os.path.exists(self.config['index_dir_path']):
            os.makedirs(self.config['index_dir_path'])
        
        #vectors=vectors.astype(np.float32)
        vectors=np.concatenate(vectors)
        target_actions=np.concatenate(target_actions)
        rtgs=np.concatenate(rtgs)
        
        target_actions_path=os.path.join(self.config['index_dir_path'],'target_actions.pkl')
        rtgs_path=os.path.join(self.config['index_dir_path'],'rtgs.pkl')
        #vectors_path=os.path.join(self.config['retrieval_dataset_dir_path'],'vectors.pkl')
        
        with open(target_actions_path,'wb') as f:
            pickle.dump(target_actions,f)
        
        with open(rtgs_path,'wb') as f:
            pickle.dump(rtgs,f)

        """ with open(vectors_path,'wb') as f:
            pickle.dump(vectors,f) """
        
        #print(f'target_actions保存至 {target_actions_path}')
        vectors=vectors.astype(np.float32)
        #index=faiss.index_factory(self.dt_config.embed_dim,'Flat',faiss.METRIC_L2) #hidden dim
        if self.index_type=='inner':
            index=faiss.index_factory(self.config['hidden_dim'],'Flat',faiss.METRIC_INNER_PRODUCT) #hidden dim
            normalize_L2(vectors) #正则化
        elif self.index_type=='l2':
            index=faiss.index_factory(self.config['hidden_dim'],'Flat',faiss.METRIC_L2) #hidden dim
        else :
            raise NotImplementedError

        index.add(vectors)
        index_path=os.path.join(self.config['index_dir_path'],f'{self.index_type}.index')
        faiss.write_index(index,index_path) #保存
        print(f'index保存至 {index_path}')
    
    def get_model_prob(self,states): #已经截断过的
        probs_list=[]
        state_now=states[:,-1].detach().cpu().numpy().reshape(1,4,84,84)
       
        for model in self.model_list:
            #model_value=[model.predict_value(state_now,torch.tensor(i))[0] for i in range(self.action_space+1)]
            #model_value=np.array(model_value)
            #model_value_norm=model_value/max(np.abs(np.max(model_value)),np.abs(np.min(model_value)))
            #model_value_norm=model_value/model_value.mean()
            #prob = np.exp(model_value_norm)
            #prob /=prob.sum()
            prob = np.zeros(self.action_space+1)
            action = model.predict(state_now)[0]
            prob[action]=1
            probs_list.append(prob)
        probs=np.stack(probs_list,axis=0).reshape(len(self.model_list),-1)
        probs_mean = probs.mean(axis=0) # action取平均
        probs_norm = probs/(np.linalg.norm(probs,axis=1).reshape(-1,1))
        mean = probs_norm.mean(axis=0) 
        z_scores = np.abs(probs_norm-mean).mean(axis=0).sum()
        return probs_mean,z_scores
    def get_retrieval_prob(self,states,actions,rtgs,timesteps):
        with torch.no_grad():
            states=states/255.0
            hidden_states=self.retrieval_model.encode(states, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps).cpu().numpy().reshape(1,-1)
        if self.index_type=='inner':
            normalize_L2(hidden_states)
        D,I=self.index.search(hidden_states,self.config['topk'])
        knn_probs=np.zeros(self.action_space+1)
        if self.index_type=='inner':
            D=1-D
       
        I=I[0] 
        D=D[0]
        
        #I=I[D<0.1]
        #D=D[D<0.1]
        
        if D.shape[0]==0:
            return knn_probs,False
        retrievals=[]
        for i in range(D.shape[0]):
            data=RetrievalData(D[i],I[i],self.target_action[I[i]],self.rtg[I[i]])
            retrievals.append(data)
        #retrievals.sort(key=lambda x:-x.r)
        #retrievals=retrievals[:20]
        weights=np.zeros((len(retrievals)))
        for i,data in enumerate(retrievals):
            weights[i]=1/(data.d+1e-5)
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
        model_prob,z=self.get_model_prob(states)
        
        lamb=self.get_lamb(z)
        self.lamb_list.append(lamb)
    
        retrieval_prob,is_retrieval=self.get_retrieval_prob(states,actions,rtgs,timesteps)
        if args.retrieval_only:
            return retrieval_prob
        if is_retrieval==False:
            return model_prob
        
        prob=retrieval_prob*lamb+model_prob*(1-lamb)
        return prob
    """ def get_action(self,prob):
        action = torch.multinomial(prob, num_samples=1)
        #action = torch.argmax(prob)
        return action """


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='Qbert')
parser.add_argument('--index_type', type=str, default='inner')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--retrieval_only', action='store_true')
parser.add_argument('--model_num', type=int, default=4)
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

set_seed(args.seed)

model_list=[]



gpt_config_path=os.path.join('weights',args.game,'gpt_config.pkl')
with open(gpt_config_path,'rb') as f:
    gpt_config = pickle.load(f)
with open(f'dataset/{args.game}/timesteps.pkl','rb') as f:
    timesteps=pickle.load(f)

retrieval_model=GPT(gpt_config)
retrieval_model_path=os.path.join('retrieval_weights',args.game,f'{args.game}.pt')
retrieval_model.load_state_dict(torch.load(retrieval_model_path,map_location='cpu'))
retrieval_model.to(device=device)
game=args.game.lower()
with open(f'd3rl_dataset_cache/{game}.pkl','rb') as f:
    dataset=pickle.load(f)
with open(f'd3rl_dataset_cache/{game}_env.pkl','rb') as f:
    env=pickle.load(f)

for i in range(args.model_num):
    if args.game=='Breakout':
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
    else :
        cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
        ).create(device=device)
    cql.build_with_dataset(dataset)
    cql.build_with_env(env)
    cql.load_model(f'cql_weights/{game}/cql_{game}_{i}.pt')
    model_list.append(cql)

retrieval_dataset_dir_path=os.path.join('dataset',args.game)
index_dir_path=os.path.join('index',args.game,args.index_type)
config={
    "retrieval_dataset_dir_path":retrieval_dataset_dir_path,
    "index_dir_path":index_dir_path,
    "device":device,
    "index_type":args.index_type,
    "context_length":context_length,
    "hidden_dim":gpt_config.n_embd,
    'topk':100
}
model=RetrievalRL(retrieval_model,model_list,config)


ret=target_return
T_rewards, T_Qs = [], []
done = True
eval_num=10
print('开始eval')
for i in range(eval_num):
    state,info = env.reset()

    all_states = torch.from_numpy(state).type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    done = False
    rtgs = [ret]
    actions = []
    j=0
    reward_sum=0
    while not done:

        prob = model.get_prob(
            all_states[-context_length:].unsqueeze(0),
            actions=torch.tensor(actions[-context_length:], dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
            rtgs=torch.tensor(rtgs[-context_length:], dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=(min(j, max(timesteps)) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
        
        prob = torch.from_numpy(prob)
        sampled_action=torch.multinomial(prob, num_samples=1)
        #action = sampled_action.cpu().numpy()[0]
        #sampled_action=torch.argmax(prob)
        action = sampled_action.cpu().item()
        actions += [sampled_action]
        state, reward, done, _ ,info = env.step(action)
        state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)
        reward_sum += float(reward)
        j += 1
        if j>=max(timesteps):
            break
        all_states = torch.cat([all_states, state], dim=0)
        rtgs += [rtgs[-1] - reward]
    print('return: ',reward_sum)
    print('avg lamb: ',model.avg_lamb())
    T_rewards.append(reward_sum)
import pandas as pd
csv_path='exp/cql_exp.csv'
dataframe=pd.read_csv(csv_path)
dataframe.loc[len(dataframe)]={
    'game':args.game,
    'model_num':args.model_num,
    'mean':np.array(T_rewards).mean(),
    'std':np.array(T_rewards).std()
}
print(np.array(T_rewards).mean())
print(np.array(T_rewards).std())
dataframe.to_csv(csv_path,index=False)
