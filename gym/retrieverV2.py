import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch import nn as nn
from torch.utils.data.dataloader import DataLoader
import faiss
from faiss import normalize_L2
from dataset import DT_Dataset
from decision_transformer.models.decision_transformer import DecisionTransformer
from DecisionTransformerV2 import DecisionTransformerV2,MLPBCModelV2
from utils import RetrievalData
import copy
import sys

from cql.rlkit.torch.sac.policies import TanhGaussianPolicy,MakeDeterministic



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
        print(f'DT encoder path: {dt_weight_path}')
        self.dt.load_state_dict(torch.load(dt_weight_path,map_location=torch.device('cpu')))
        self.dt.to(self.device)
        self.dt.eval()
        self.index_dir_path=index_dir_path
        self.index_type=index_type
        self.init_retrieval_data()
        self.representations=[]
        
    def init_retrieval_data(self):
        self.check_retrieval_dataset(self.index_dir_path)
        
        index_dir_path=os.path.join(self.index_dir_path,f'{self.index_type}.index')
        
        res = faiss.StandardGpuResources()
        print(f'index dir: {index_dir_path}')
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
        print(f'there are {len(self.target_action)} pieces of data')
        print('retrieval dataset is loaded over')
    def check_retrieval_dataset(self,retrieval_dataset_dir_path):
        if not os.path.exists(retrieval_dataset_dir_path):
            print(f'retrieval dataset not found')
            self.make_retrieval_dataset()   
    def make_retrieval_dataset(self):
        print('creating retrieval dataset ...')
        with open(self.kwargs['dataset_path'], 'rb') as f:
            print(f'retrieval dataset: {self.kwargs["dataset_path"]}')
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
        print('Dataloader is created')
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
        
        print('Data is loaded ')           
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
    
        
        vectors=vectors.astype(np.float32)
        #index=faiss.index_factory(self.dt_config.embed_dim,'Flat',faiss.METRIC_L2) #hidden dim
        if self.index_type=='inner':
            index=faiss.index_factory(self.kwargs['embed_dim'],'Flat',faiss.METRIC_INNER_PRODUCT) #hidden dim
            normalize_L2(vectors) 
        elif self.index_type=='l2':
            index=faiss.index_factory(self.kwargs['embed_dim'],'Flat',faiss.METRIC_L2) #hidden dim
        else :
            raise NotImplementedError

        index.add(vectors)
        index_path=os.path.join(self.index_dir_path,f'{self.index_type}.index')
        faiss.write_index(index,index_path) 
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
        #not need rewards
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
        #timestep_zeros = torch.zeros(size=timesteps.shape).to(self.device,torch.long)
        hidden_states=self.dt.encode(states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask).cpu().numpy().reshape(1,-1)
        if self.index_type=='inner':
            normalize_L2(hidden_states)
            
        self.representations.append(hidden_states)
        
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
            #retrievals.sort(key=lambda x:-x.r)
            #retrievals=retrievals[:16]
            weights=np.zeros((len(retrievals)))
            for i,data in enumerate(retrievals):
                weights[i] = 1/data.d
            weights=weights/weights.sum()
            for i,data in enumerate(retrievals):
                action_knn+=data.a*weights[i]
            return action_knn.astype(np.float32),True
        else :
            return action_knn.astype(np.float32),False


class ManyModel:
    def __init__(self,model_list,act_dim):
        self.model_list=model_list
        self.num=len(model_list)
        self.act_dim=act_dim
    def get_action(self,state):
        a_list=[]
        for model in self.model_list:
            if type(model) ==MakeDeterministic:
                a=model.get_action(state)[0].reshape(-1) # cql 
            elif type(model) == DecisionTransformerV2 or type(model)==MLPBCModelV2:
                a=model.get_action(state).reshape(-1) # dt

            a_list.append(a)
        actions=np.stack(a_list,axis=0)
        #mean = actions.mean(axis=0) # action取平均
        #std = actions.std(axis=0) # 算标准差
        actions_mean = actions.mean(axis=0) # action取平均
        actions_norm = actions/(np.linalg.norm(actions,axis=1).reshape(-1,1))
        mean = actions_norm.mean(axis=0)
        z_scores = np.abs(actions_norm-mean).mean(axis=0).sum()
        
        return actions_mean,z_scores
    def reset(self):
        for model in self.model_list:
            model.reset()
    def update_history(self,action,reward):
        for model in self.model_list:
            if type(model) == DecisionTransformerV2:
                model.update_history(action,reward)

class RetrieverRLV2:
    def __init__(self,
                 retriever,
                 model,
                 lamb,
                 solo,
                 retrieval_only,
                 kwargs):
        self.retriever=retriever
        self.model=model
        self.kwargs=kwargs
        self.lamb=lamb
        self.solo=solo
        self.retrieval_only=retrieval_only
        self.lamb_record=[]
    def get_lamb(self,var):
        return 1-np.exp(-var)
    @torch.no_grad()
    def get_action(self,state):
        if self.solo:
            model_action=self.model.get_action(state)
            retriever_action,is_retrieval=self.retriever.get_retrieval_action(state)
            retriever_action=retriever_action.reshape(1,-1)
            lamb=self.lamb
        else : 
            model_action,var=self.model.get_action(state) # 返回action和方差
            lamb=self.get_lamb(var)
            self.lamb_record.append(lamb)
            retriever_action,is_retrieval=self.retriever.get_retrieval_action(state)
            retriever_action=retriever_action.reshape(1,-1)
            
            if self.retrieval_only:
                return retriever_action        
            
        assert 0 <= lamb <= 1
      
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

        




