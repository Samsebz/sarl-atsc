import argparse
import numpy as np
from sarl_atsc.environment import TrafficSignalEnv
from sarl_atsc.dqn import DQNAgent
from sarl_atsc.safety import SafetyConstraintLayer

def run(model,episodes=20,scenario="festival"):
    env=TrafficSignalEnv(lanes=4,max_steps=300)
    agent=DQNAgent(state_size=16,action_size=4,hidden_size=256,lr=0.0005,gamma=0.95,epsilon=0.01,epsilon_min=0.01,epsilon_decay=1.0); agent.load(model)
    safety=SafetyConstraintLayer(min_green_seconds=10,max_red_seconds=90,prohibited_transitions={(0,2),(1,3)})
    waits=[]; throughput=[]; overrides=0; steps=0
    for _ in range(episodes):
        state,_=env.reset(); env.set_scenario(scenario); current_phase=0; green=20.0; reds=[0.0]*4
        for _ in range(env.max_steps):
            proposed=agent.act(state,training=False); result=safety.validate(proposed,current_phase,green,reds); action=result.approved_action; overrides+=int(result.overridden); steps+=1
            next_state,_,terminated,truncated,info=env.step(action); waits.append(info["average_wait_proxy"]); throughput.append(info["throughput"]); current_phase=action; green+=1.0
            for i in range(4): reds[i]=0.0 if i==current_phase else reds[i]+1.0
            state=next_state
            if terminated or truncated: break
    print("\nEvaluation\n----------"); print(f"Scenario: {scenario}"); print(f"Average waiting-time proxy: {np.mean(waits):.3f}"); print(f"Total throughput: {np.sum(throughput):.1f}"); print(f"Safety override rate: {overrides/max(1,steps):.4f}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--model",required=True); parser.add_argument("--episodes",type=int,default=20); parser.add_argument("--scenario",default="festival",choices=["low","peak","incident","festival"]); args=parser.parse_args(); run(args.model,args.episodes,args.scenario)
