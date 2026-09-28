import argparse
from pathlib import Path
import numpy as np
from sarl_atsc.environment import TrafficSignalEnv
from sarl_atsc.dqn import DQNAgent
from sarl_atsc.adaptation import FastAdaptationModule
from sarl_atsc.utils import set_seed

def train(episodes=500, output="artifacts/dqn.pt", seed=42):
    set_seed(seed)
    env=TrafficSignalEnv(lanes=4,max_steps=300,seed=seed)
    agent=DQNAgent(state_size=16,action_size=4,hidden_size=256,lr=0.0005,gamma=0.95,epsilon=1.0,epsilon_min=0.01,epsilon_decay=0.995,replay_capacity=50000,batch_size=64,target_update=1000)
    adaptation=FastAdaptationModule(agent,threshold=0.5,inner_steps=5,learning_rate=0.0001)
    reference_states=[]; rewards=[]
    for episode in range(episodes):
        state,_=env.reset(); env.set_scenario("low" if episode%2==0 else "peak"); episode_reward=0.0
        for _ in range(env.max_steps):
            action=agent.act(state,training=True); next_state,reward,terminated,truncated,_=env.step(action)
            agent.remember(state,action,reward,next_state,terminated or truncated); agent.learn()
            reference_states.append(state)
            if len(reference_states)>5000: reference_states.pop(0)
            state=next_state; episode_reward+=reward
            if terminated or truncated: break
        if episode%20==0 and reference_states: adaptation.set_reference(reference_states[-500:])
        rewards.append(episode_reward)
        if (episode+1)%10==0: print(f"Episode {episode+1}/{episodes} | Reward {episode_reward:.2f} | Epsilon {agent.epsilon:.3f}")
    Path(output).parent.mkdir(parents=True,exist_ok=True); agent.save(output); np.save(Path(output).with_suffix(".rewards.npy"),np.asarray(rewards)); print(f"Saved model to {output}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--episodes",type=int,default=500); parser.add_argument("--output",default="artifacts/dqn.pt"); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args(); train(args.episodes,args.output,args.seed)
