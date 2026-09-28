# 🚦 SARL-ATSC

### Scenario-Agnostic Reinforcement Learning for Adaptive Traffic Signal Control

[![CI](https://github.com/Samsebz/sarl-atsc/actions/workflows/ci.yml/badge.svg)](https://github.com/Samsebz/sarl-atsc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end research prototype combining computer vision, structured traffic-state aggregation, deep reinforcement learning, fast scenario adaptation, and rule-based safety validation for adaptive traffic signal control.

> Research/demo scope: the current implementation is evaluated in simulation and is not a certified live-road traffic controller.

## Why this project?

Fixed-time traffic signals do not react to rapidly changing demand. A learning-based controller also needs a meaningful state representation, scenario-shift handling, and a safety gate between the model and the actuator.

## Architecture

```text
Camera / Video
      |
      v
+---------------+
|    YOLOv8     |  vehicle detection
+-------+-------+
        |
        v
+---------------+
|    GESA++     |  queue / density / flow / occupancy
+-------+-------+
        |
        v
+---------------+
|     DQN       |  signal phase decision
+-------+-------+
        |
        v
+---------------+
| Fast Adapt.   |  KL-divergence scenario shift
+-------+-------+
        |
        v
+---------------+
| Safety Layer  |  timing + transition constraints
+-------+-------+
        |
        v
+---------------+
|  Controller   |  validated phase output
+---------------+
```

## Core components

| Component | Role |
|---|---|
| YOLOv8 | Detects cars, trucks, buses and motorcycles from video |
| Centroid tracker | Maintains lightweight object IDs for motion/flow estimation |
| GESA++ | Builds queue-length, density, flow and occupancy features |
| DQN | Learns signal-phase decisions with replay memory + target network |
| Fast Adaptation | Detects distribution shift with KL divergence and adapts the policy |
| Safety Constraint Layer | Enforces minimum green, maximum red and prohibited transitions |
| Gymnasium environment | Reproducible traffic queue simulation |
| Streamlit | Experiment dashboard |

## Engineering snapshot

The reference build was locally validated with:

- 27 project files across training, inference, simulation, safety, tests and UI
- 5/5 automated unit tests passing
- approximately 2.99K simulation control decisions/sec in the measured non-vision control loop
- approximately 0.335 ms/decision for that benchmarked control path

These are software benchmark figures from the included implementation, not real-world traffic outcomes.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Train the DQN

```bash
python train.py --episodes 20 --output artifacts/dqn_demo.pt
```

### Evaluate an unseen scenario

```bash
python evaluate.py --model artifacts/dqn_demo.pt --episodes 10 --scenario festival
```

Supported scenarios: `low`, `peak`, `incident`, `festival`.

### Run the dashboard

```bash
streamlit run app.py
```

### Run video inference

```bash
python infer_video.py --source traffic.mp4 --weights yolov8n.pt --model artifacts/dqn.pt --lane-config configs/lane_masks.example.json --output artifacts/result.mp4
```

## Tests / CI

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests.

## Project structure

```text
sarl-atsc/
├── .github/workflows/ci.yml
├── configs/lane_masks.example.json
├── sarl_atsc/
│   ├── adaptation.py
│   ├── config.py
│   ├── controller.py
│   ├── detection.py
│   ├── dqn.py
│   ├── environment.py
│   ├── gesa.py
│   ├── pipeline.py
│   ├── replay.py
│   ├── safety.py
│   └── tracking.py
├── tests/
│   ├── test_dqn.py
│   ├── test_gesa.py
│   └── test_safety.py
├── app.py
├── evaluate.py
├── infer_video.py
├── make_lane_config.py
├── train.py
├── train_yolov8.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Technical notes

For each lane, GESA++ exposes four normalized features:

```text
[queue_length, lane_density, flow_rate, occupancy]
```

For four lanes this becomes a 16-dimensional DQN state.

The reference DQN uses two hidden layers of 256 ReLU units, replay memory, and a target network. The safety layer sits between the policy and controller so an invalid model action is overridden instead of being sent directly to the actuator.

## Limitations

The current implementation is deliberately a research/demo system. The traffic environment is simplified, lane masks are camera-specific, real deployment would require calibrated sensing and fail-safe controller integration, multi-intersection coordination is not modeled, pedestrian/cyclist behavior is not yet included, and formal safety verification is not implemented.

## Portfolio framing

This repository demonstrates an end-to-end systems workflow spanning:

**ML inference -> state processing -> reinforcement learning -> safety rules -> simulation -> testing -> UI.**

## License

MIT — see `LICENSE`.
