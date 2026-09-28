from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class DQNConfig:
    state_size: int = 16
    action_size: int = 4
    hidden_size: int = 256
    learning_rate: float = 5e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    replay_capacity: int = 50_000
    batch_size: int = 64
    target_update: int = 1_000

@dataclass
class SafetyConfig:
    min_green_seconds: float = 10.0
    max_red_seconds: float = 90.0

@dataclass
class AdaptationConfig:
    kl_threshold: float = 0.50
    inner_steps: int = 5
    learning_rate: float = 1e-4

@dataclass
class Config:
    seed: int = 42
    dqn: DQNConfig = field(default_factory=DQNConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

def load_config(path="config.yaml"):
    p = Path(path)
    if not p.exists():
        return Config()
    raw = yaml.safe_load(p.read_text()) or {}
    return Config(
        seed=raw.get("seed", 42),
        dqn=DQNConfig(**raw.get("dqn", {})),
        safety=SafetyConfig(**raw.get("safety", {})),
        adaptation=AdaptationConfig(**raw.get("adaptation", {})),
    )
