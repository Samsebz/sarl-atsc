from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .replay import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,hidden_size=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,action_size))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self,state_size,action_size,hidden_size=256,lr=5e-4,gamma=0.95,epsilon=1.0,epsilon_min=0.01,epsilon_decay=0.995,replay_capacity=50000,batch_size=64,target_update=1000,device=None):
        self.state_size=state_size; self.action_size=action_size; self.gamma=gamma; self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay; self.batch_size=batch_size; self.target_update=target_update
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online=QNetwork(state_size,action_size,hidden_size).to(self.device); self.target=QNetwork(state_size,action_size,hidden_size).to(self.device); self.target.load_state_dict(self.online.state_dict())
        self.optimizer=optim.Adam(self.online.parameters(),lr=lr); self.replay=ReplayBuffer(replay_capacity); self.learn_steps=0
    def act(self,state,training=True):
        if training and np.random.random()<self.epsilon: return int(np.random.randint(self.action_size))
        with torch.no_grad():
            x=torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0); return int(torch.argmax(self.online(x),dim=1).item())
    def remember(self,state,action,reward,next_state,done): self.replay.push(state,action,reward,next_state,done)
    def learn(self):
        if len(self.replay)<self.batch_size: return None
        states,actions,rewards,next_states,dones=self.replay.sample(self.batch_size)
        states=torch.tensor(states,device=self.device); actions=torch.tensor(actions,device=self.device).unsqueeze(1); rewards=torch.tensor(rewards,device=self.device); next_states=torch.tensor(next_states,device=self.device); dones=torch.tensor(dones,device=self.device)
        q=self.online(states).gather(1,actions).squeeze(1)
        with torch.no_grad(): next_q=self.target(next_states).max(dim=1).values; target=rewards+self.gamma*next_q*(1.0-dones)
        loss=nn.functional.smooth_l1_loss(q,target); self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),5.0); self.optimizer.step()
        self.learn_steps+=1
        if self.learn_steps%self.target_update==0: self.target.load_state_dict(self.online.state_dict())
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay); return float(loss.item())
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save({"online":self.online.state_dict(),"target":self.target.state_dict(),"epsilon":self.epsilon,"state_size":self.state_size,"action_size":self.action_size},path)
    def load(self,path):
        ckpt=torch.load(path,map_location=self.device); self.online.load_state_dict(ckpt["online"]); self.target.load_state_dict(ckpt.get("target",ckpt["online"])); self.epsilon=ckpt.get("epsilon",self.epsilon_min)
