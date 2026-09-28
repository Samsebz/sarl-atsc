import numpy as np
import streamlit as st
from sarl_atsc.environment import TrafficSignalEnv
from sarl_atsc.dqn import DQNAgent
from sarl_atsc.safety import SafetyConstraintLayer
st.set_page_config(page_title="SARL-ATSC",layout="wide")
st.title("SARL-ATSC — AI Adaptive Traffic Control")
st.markdown("YOLOv8 perception → GESA++ state → DQN decision → fast adaptation → safety validation → signal controller")
tab1,tab2=st.tabs(["Simulation","Video"])
with tab1:
    st.subheader("Traffic Signal Simulation"); episodes=st.slider("Episodes",1,20,3); scenario=st.selectbox("Scenario",["low","peak","incident","festival"])
    if st.button("Run simulation"):
        env=TrafficSignalEnv(lanes=4,max_steps=200); agent=DQNAgent(state_size=16,action_size=4,hidden_size=256,epsilon=0.0,epsilon_min=0.0,epsilon_decay=1.0)
        safety=SafetyConstraintLayer(min_green_seconds=10,max_red_seconds=90,prohibited_transitions={(0,2),(1,3)}); waits=[]; throughput=[]; overrides=0; steps=0
        for _ in range(episodes):
            state,_=env.reset(); env.set_scenario(scenario); phase=0; green=20.0; reds=[0.0]*4
            for _ in range(200):
                proposed=agent.act(state,training=False); result=safety.validate(proposed,phase,green,reds); action=result.approved_action; overrides+=int(result.overridden); steps+=1
                state,_,terminated,truncated,info=env.step(action); waits.append(info["average_wait_proxy"]); throughput.append(info["throughput"]); phase=action; green+=1.0
                for i in range(4): reds[i]=0.0 if i==phase else reds[i]+1.0
                if terminated or truncated: break
        c1,c2,c3=st.columns(3); c1.metric("Average wait proxy",f"{np.mean(waits):.2f}"); c2.metric("Throughput",f"{np.sum(throughput):.0f}"); c3.metric("Safety override rate",f"{overrides/max(1,steps):.2%}")
with tab2:
    st.subheader("YOLOv8 Video Inference"); st.info("For full end-to-end video inference use infer_video.py. You need YOLO weights, a trained DQN model, and calibrated lane masks."); st.file_uploader("Upload traffic video",type=["mp4","avi","mov"])
