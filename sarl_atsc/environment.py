import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Env:
        def reset(self, *, seed=None, options=None): return None, {}
    class _Discrete:
        def __init__(self,n): self.n=n
        def sample(self): return int(np.random.randint(self.n))
    class _Box:
        def __init__(self,low,high,shape,dtype): self.low,self.high,self.shape,self.dtype=low,high,shape,dtype
    class _Spaces: Discrete=_Discrete; Box=_Box
    class _Gym: Env=_Env
    gym,spaces=_Gym(),_Spaces()

class TrafficSignalEnv(gym.Env):
    """Four-lane simplified queue simulator for research/demo use."""
    metadata={"render_modes":["human"]}
    def __init__(self,lanes=4,max_steps=300,seed=None):
        super().__init__(); self.lanes=lanes; self.max_steps=max_steps
        self.action_space=spaces.Discrete(lanes)
        self.observation_space=spaces.Box(low=0.0,high=1.0,shape=(lanes*4,),dtype=np.float32)
        self.rng=np.random.default_rng(seed)
    def _state(self):
        s=np.zeros(self.lanes*4,dtype=np.float32)
        for i in range(self.lanes):
            s[i*4]=np.clip(self.queues[i]/30.0,0,1); s[i*4+1]=np.clip(self.density[i]/20.0,0,1)
            s[i*4+2]=np.clip(self.flow[i]/20.0,0,1); s[i*4+3]=np.clip(self.occupancy[i],0,1)
        return s
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.step_count=0; self.current_phase=0
        self.queues=self.rng.integers(0,8,size=self.lanes).astype(np.float32)
        self.arrival_rates=self.rng.uniform(0.2,0.8,size=self.lanes)
        self.density=self.queues/2.0; self.flow=np.zeros(self.lanes,dtype=np.float32)
        self.occupancy=np.clip(self.queues/15.0,0,1)
        return self._state(),{}
    def set_scenario(self,name):
        if name=="low": self.arrival_rates=np.full(self.lanes,0.20)
        elif name=="peak": self.arrival_rates=np.array([0.80,0.65,0.25,0.20])
        elif name=="incident": self.arrival_rates=np.array([0.95,0.15,0.15,0.20])
        elif name=="festival": self.arrival_rates=np.full(self.lanes,0.75)
    def step(self,action):
        action=int(action); self.current_phase=action
        arrivals=self.rng.poisson(self.arrival_rates).astype(np.float32); self.queues+=arrivals
        service=np.zeros(self.lanes,dtype=np.float32); service[action]=self.rng.integers(3,8)
        cleared=np.minimum(self.queues,service); self.queues-=cleared; self.flow=cleared
        self.density=self.queues/2.0; self.occupancy=np.clip(self.queues/15.0,0,1)
        reward=-float(self.queues.sum())+float(self.flow.sum())-0.5*float(self.occupancy.sum())
        self.step_count+=1; terminated=False; truncated=self.step_count>=self.max_steps
        info={"average_wait_proxy":float(self.queues.mean()),"throughput":float(self.flow.sum()),"total_queue":float(self.queues.sum())}
        return self._state(),reward,terminated,truncated,info
