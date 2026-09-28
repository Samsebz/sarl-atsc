import numpy as np
from sarl_atsc.dqn import DQNAgent

def test_dqn_action():
    agent=DQNAgent(state_size=16,action_size=4,hidden_size=32,batch_size=2,epsilon=0.0,epsilon_min=0.0,epsilon_decay=1.0)
    state=np.zeros(16,dtype=np.float32); action=agent.act(state,training=False); assert 0<=action<4

def test_dqn_learning():
    agent=DQNAgent(state_size=16,action_size=4,hidden_size=32,batch_size=2,epsilon=0.0,epsilon_min=0.0,epsilon_decay=1.0)
    s=np.zeros(16,dtype=np.float32); ns=np.ones(16,dtype=np.float32)
    for _ in range(2): agent.remember(s,0,1.0,ns,False)
    loss=agent.learn(); assert loss is not None
