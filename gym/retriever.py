import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch import nn as nn
from torch.utils.data.dataloader import DataLoader
import faiss
from faiss import normalize_L2
from KNN_DT.utils import DT_Dataset
from decision_transformer.models.decision_transformer import DecisionTransformer
from utils import RetrievalData
import copy
import sys
sys.path.append('/data/wangchunhao-slurm/workspace/code/projects/decision-transformer-master/gym/cql/')
sys.path.append('/data/wangchunhao-slurm/workspace/code/projects/decision-transformer-master/gym/bear/')
from cql.rlkit.torch.sac.policies import TanhGaussianPolicy,MakeDeterministic
from bear.algos import BEAR


class KNN_DT_Retriever(nn.Module):
    def __init__(self,dt_weight_path,index_dir_path,index_type,
                 **kwargs):
        super(KNN_DT_Retriever,self).__init__()
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
        self.act_dim=kwargs['act_dim']
        self.state_dim=kwargs['state_dim']
        self.kwargs=kwargs
        self.maxlen=kwargs['K']
        self.device=kwargs['device']
        print(f'检索dt权重路径: {dt_weight_path}')
        self.dt.load_state_dict(torch.load(dt_weight_path,map_location=torch.device('cpu')))
        self.dt.to(self.device)
        self.dt.eval()
        self.index_dir_path=index_dir_path
        self.index_type=index_type
        self.init_retrieval_data() 
        
    def init_retrieval_data(self):
        self.check_retrieval_dataset(self.index_dir_path)
        
        index_dir_path=os.path.join(self.index_dir_path,f'{self.index_type}.index')
        
        res = faiss.StandardGpuResources()
        print(f'索引文件夹：{index_dir_path}')
        self.index=faiss.read_index(index_dir_path)
        try:
            self.index=faiss.index_cpu_to_gpu(res, int(self.device[-1]), self.index) #使用faiss-gpu
        except Exception as e:
            print(e)
        
        with open(os.path.join(self.index_dir_path,'target_action.pkl'),'rb') as f:
            self.target_action=pickle.load(f)
        with open(os.path.join(self.index_dir_path,'reward.pkl'),'rb') as f:
            self.reward=pickle.load(f)
        with open(os.path.join(self.index_dir_path,'timestep.pkl'),'rb') as f:
            self.timestep=pickle.load(f)
        with open(os.path.join(self.index_dir_path,'rtg.pkl'),'rb') as f:
            self.rtg=pickle.load(f)
        with open(os.path.join(self.index_dir_path,'state.pkl'),'rb') as f:
            self.state=pickle.load(f)
        with open(os.path.join(self.index_dir_path,'input_data.pkl'),'rb') as f:
            self.input_data=pickle.load(f)
        print(f'有{len(self.target_action)}条数据')
        print('检索数据集加载完毕')
    def check_retrieval_dataset(self,retrieval_dataset_dir_path):
        if not os.path.exists(retrieval_dataset_dir_path):
            print(f'未找到检索数据集')
            self.make_retrieval_dataset()   
    def make_retrieval_dataset(self):
        print('正在创建检索数据集')
        with open(self.kwargs['dataset_path'], 'rb') as f:
            print(f'检索数据集来自于 {self.kwargs["dataset_path"]}')
            trajectories = pickle.load(f)
            
        #vectors=np.empty((0,self.dt_config.embed_dim))
        #target_actions=np.empty((0,self.act_dim))
        vectors=[]
        states=[]
        target_actions=[]
        rewards=[]
        rtgs=[]
        timesteps=[]
        input_data={
            'states':[],
            'actions':[],
            'rtgs':[],
            'timesteps':[],
            'masks':[]
        }
        dataset=DT_Dataset(trajectories,self.maxlen)
        
        loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=6,pin_memory=True)
        print('Dataloader创建完毕')
        for (state, action, rtg, timestep, mask, info) in tqdm(loader):
            with torch.no_grad():
                input_data['states'].append(state)
                input_data['actions'].append(action)
                input_data['rtgs'].append(rtg)
                input_data['timesteps'].append(timestep)
                input_data['masks'].append(mask)
                #length=np.random.randint(1,args.context_length)
                state = state.to(self.device,torch.float32)
                action = action.to(self.device,torch.float32)
                rtg = rtg.to(self.device,torch.float32)
                rtg = rtg/self.kwargs['scale']
                #rtg_zeros = torch.zeros(size=rtg.shape).to(self.device,torch.float32)
                timestep = timestep.to(self.device,torch.long)
                #timestep_zeros = torch.zeros(size=timestep.shape).to(self.device,torch.long)
                mask = mask.to(self.device)
                
                target_action=copy.deepcopy(action[:,-1])
                action[:,-1]=torch.zeros(self.kwargs['act_dim'],dtype=torch.float32).to(device=self.device)
                hidden_states=self.dt.encode(state, action, None, rtg, timestep, mask).cpu().numpy().reshape(-1,self.kwargs['embed_dim']) #(batch_size,hidden_dim)
                #hidden_states/=np.linalg.norm(hidden_states,axis=1).reshape(-1,1)
                #print(hidden_states)
                target_action=target_action.cpu().numpy()
                
                states.append(info['state'])
                rewards.append(info['reward'])
                timesteps.append(info['timestep'])
                rtgs.append(info['rtg'])
                target_actions.append(target_action)
                vectors.append(hidden_states)

        vectors=np.concatenate(vectors)  
        target_actions=np.concatenate(target_actions)
        # info
        states=np.concatenate(states)
        rewards=np.concatenate(rewards)
        timesteps=np.concatenate(timesteps)
        rtgs=np.concatenate(rtgs)
        
        #input_data
        input_data['states']=np.concatenate(input_data['states'],axis=0)
        input_data['actions']=np.concatenate(input_data['actions'],axis=0)
        input_data['rtgs']=np.concatenate(input_data['rtgs'],axis=0)
        input_data['timesteps']=np.concatenate(input_data['timesteps'],axis=0)
        input_data['masks']=np.concatenate(input_data['masks'],axis=0)
        
        print('数据加载完毕')           
        if not os.path.exists(self.index_dir_path):
            os.makedirs(self.index_dir_path)
        
        
        input_data_path=os.path.join(self.index_dir_path,'input_data.pkl')
        with open(input_data_path,'wb') as f:
            pickle.dump(input_data,f)
        
        
        target_actions_path=os.path.join(self.index_dir_path,'target_action.pkl')
        states_path=os.path.join(self.index_dir_path,'state.pkl')
        rewards_path=os.path.join(self.index_dir_path,'reward.pkl')
        timesteps_path=os.path.join(self.index_dir_path,'timestep.pkl')
        rtgs_path=os.path.join(self.index_dir_path,'rtg.pkl')
        with open(target_actions_path,'wb') as f:
            pickle.dump(target_actions,f)
        with open(states_path,'wb') as f:
            pickle.dump(states,f)
        with open(rewards_path,'wb') as f:
            pickle.dump(rewards,f)
        with open(timesteps_path,'wb') as f:
            pickle.dump(timesteps,f)
        with open(rtgs_path,'wb') as f:
            pickle.dump(rtgs,f)
        #print(f'target_actions保存至 {target_actions_path}')
        
        vectors=vectors.astype(np.float32)
        #index=faiss.index_factory(self.dt_config.embed_dim,'Flat',faiss.METRIC_L2) #hidden dim
        if self.index_type=='inner':
            index=faiss.index_factory(self.kwargs['embed_dim'],'Flat',faiss.METRIC_INNER_PRODUCT) #hidden dim
            normalize_L2(vectors) #正则化
        elif self.index_type=='l2':
            index=faiss.index_factory(self.kwargs['embed_dim'],'Flat',faiss.METRIC_L2) #hidden dim
        else :
            raise NotImplementedError

        index.add(vectors)
        index_path=os.path.join(self.index_dir_path,f'{self.index_type}.index')
        faiss.write_index(index,index_path) #保存
        print(f'index保存至 {index_path}')
    def reset(self):
        self.history={
            'states':[],
            'actions':[],
            'rtgs':[],
            'rewards':[],
            'timesteps':[],
            'time_now':0,
            'target_return':self.kwargs['target_return']
        }
    def update_history(self,action,reward):
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['target_return']-=reward
    @torch.no_grad()
    def get_retrieval_action(self,state):
        self.history['states'].append(state)
        self.history['timesteps'].append(self.history['time_now'])
        self.history['rtgs'].append(self.history['target_return'])
        self.history['time_now']+=1
        
        states = np.stack(self.history['states'])
        states = (states-self.kwargs['states_mean'])/self.kwargs['states_std']
        actions = np.concatenate([np.array(self.history['actions']).reshape(-1,self.act_dim), np.zeros((1, self.act_dim))], axis=0)
        #rewards不需要
        #rewards = np.concatenate([np.array(self.history['rewards']).reshape(-1,1), np.zeros((1,1))], axis=0) 
        
        rtgs = np.stack(self.history['rtgs'], axis=0,dtype=np.float32)
        rtgs/=self.kwargs['scale']
        timesteps =np.array(self.history['timesteps']).reshape(-1,1)
        
        
        states = torch.from_numpy(states[-self.maxlen:]).reshape(1,-1,self.state_dim)
        actions = torch.from_numpy(actions[-self.maxlen:]).reshape(1,-1,self.act_dim)
        returns_to_go = torch.from_numpy(rtgs[-self.maxlen:]).reshape(1,-1,1)
        timesteps = torch.from_numpy(timesteps[-self.maxlen:]).reshape(1,-1)
        
        attention_mask = torch.cat([torch.zeros(self.maxlen-states.shape[1]), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long, device=self.device).reshape(1, -1)
        states = torch.cat([torch.zeros((states.shape[0], self.maxlen-states.shape[1], self.state_dim)), states],dim=1).to(device=self.device,dtype=torch.float32)
        actions = torch.cat([torch.zeros((actions.shape[0], self.maxlen - actions.shape[1], self.act_dim)), actions],dim=1).to(device=self.device,dtype=torch.float32)
        returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], self.maxlen-returns_to_go.shape[1], 1)), returns_to_go],dim=1).to(device=self.device,dtype=torch.float32)
        timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.maxlen-timesteps.shape[1])), timesteps],dim=1).to(device=self.device,dtype=torch.long)
        
        hidden_states=self.dt.encode(states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask).cpu().numpy().reshape(1,-1)
        if self.index_type=='inner':
            normalize_L2(hidden_states)
        D,I=self.index.search(hidden_states,self.kwargs['topk'])
        if self.index_type=='inner':
            D=1-D
        D=D[0]
        I=I[0]
        action_knn=np.zeros((1,self.act_dim))
        retrievals=[]
        
        if D.shape[0]!=0:
            for i in range(D.shape[0]):
                data=RetrievalData(D[i],I[i],self.target_action[I[i]],self.rtg[I[i]]/self.kwargs['scale'])
                retrievals.append(data)
            retrievals.sort(key=lambda x:-x.r)
            retrievals=retrievals[:16]
            weights=np.zeros((len(retrievals)))
            for i,data in enumerate(retrievals):
                weights[i] = 1/data.d
            weights=weights/weights.sum()
            for i,data in enumerate(retrievals):
                action_knn+=data.a*weights[i]
            return action_knn.astype(np.float32),True
        else :
            return action_knn.astype(np.float32),False
 

class RetrieverRL:
    def __init__(self,
                 retriever,
                 model,
                 kwargs):
        self.retriever=retriever
        self.model=model
        self.kwargs=kwargs
    def get_lamb(self,var):
        return 1-np.exp(-var)
    @torch.no_grad()
    def get_action(self,state):
        retriever_action,is_retrieval=self.retriever.get_retrieval_action(state)
        retriever_action=retriever_action.reshape(1,-1)
        model_action,var=self.model.get_action(state) # 返回action和方差
        lamb=self.get_lamb(var)
        assert 0 <= lamb <= 1
        #lamb=0
        if is_retrieval:
            final_action=retriever_action*lamb+(1-lamb)*model_action
        else :
            final_action=model_action
        return final_action
    def reset(self):
        self.model.reset()
        self.retriever.reset()
    def update_history(self,action,reward):
        self.model.update_history(action,reward)
        self.retriever.update_history(action,reward)
        
    
    



