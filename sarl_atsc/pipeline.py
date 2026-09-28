import json
from pathlib import Path
from .detection import YOLOv8Detector
from .tracking import CentroidTracker
from .gesa import GESAPlusPlus
from .dqn import DQNAgent
from .adaptation import FastAdaptationModule
from .safety import SafetyConstraintLayer
from .controller import SignalController

class SARLATSC:
    def __init__(self,lane_config,model_path=None,yolo_weights="yolov8n.pt",confidence=0.5,device=None):
        lane_polygons=load_lane_config(lane_config)
        self.detector=YOLOv8Detector(weights=yolo_weights,confidence=confidence,device=device)
        self.tracker=CentroidTracker(); self.gesa=GESAPlusPlus(lane_polygons)
        state_size=len(lane_polygons)*4
        self.agent=DQNAgent(state_size=state_size,action_size=4,hidden_size=256,lr=5e-4,gamma=0.95,epsilon=0.01,epsilon_min=0.01,epsilon_decay=1.0)
        if model_path: self.agent.load(model_path)
        self.adaptation=FastAdaptationModule(self.agent,threshold=0.5,inner_steps=5,learning_rate=1e-4)
        self.safety=SafetyConstraintLayer(min_green_seconds=10,max_red_seconds=90,prohibited_transitions={(0,2),(1,3)},action_size=4)
        self.controller=SignalController(phase_count=4); self.green_elapsed=999.0; self.red_elapsed=[0.0]*4
        self.recent_states=[]; self.recent_transitions=[]
    def process_frame(self,frame,fps=30.0):
        detections=self.tracker.update(self.detector.detect(frame))
        state,lane_data=self.gesa.update(detections,frame.shape,fps=fps)
        self.recent_states.append(state)
        if len(self.recent_states)>100: self.recent_states.pop(0)
        kl,shifted=self.adaptation.detect_shift(self.recent_states[-20:])
        if shifted and len(self.recent_transitions)>=5: self.adaptation.adapt(self.recent_transitions[-10:])
        proposed=self.agent.act(state,training=False)
        result=self.safety.validate(proposed,self.controller.current_phase,self.green_elapsed,self.red_elapsed)
        previous_phase=self.controller.current_phase; phase_name=self.controller.set_phase(result.approved_action)
        if result.approved_action!=previous_phase: self.green_elapsed=0.0
        else: self.green_elapsed+=1.0/max(1.0,fps)
        for i in range(len(self.red_elapsed)):
            self.red_elapsed[i]=0.0 if i==self.controller.current_phase else self.red_elapsed[i]+1.0/max(1.0,fps)
        return self.detector.draw(frame,detections),{"state":state,"lane_data":lane_data,"kl_divergence":kl,"distribution_shift":shifted,"proposed_action":proposed,"approved_action":result.approved_action,"phase_name":phase_name,"overridden":result.overridden,"reason":result.reason}

def load_lane_config(path):
    data=json.loads(Path(path).read_text()); return data["lanes"]
