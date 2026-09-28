# SARL-ATSC Local Run Report

Date: 2026-09-28

## What was actually executed

- Python 3.13.5 runtime
- DQN training: 20 episodes, 300 steps/episode
- Model artifact: artifacts/dqn_20.pt
- Unit tests: 5/5 passed
- Simulation/control-loop benchmark: 1,000 steps
- Measured simulation/control-loop throughput: ~2,986.7 steps/s
- Measured per-step time: ~0.335 ms/step
- Benchmark scenarios: low, peak, incident, festival

## Important interpretation

The current local run is a simulation/software validation, not a real-world traffic-junction deployment. No claim of real-road impact should be made from this run.

The first DQN benchmark did not outperform the fixed-time baseline. The current research/demo training setup needs more training and/or reward/environment tuning before claiming traffic-performance gains. The safety layer also overrode many DQN actions in the short training run because the controller enforces a 10-second minimum green period while the simulated decision step is one second.

## Engineering metrics

- Modular 27-file Python project spanning perception, tracking, state aggregation, reinforcement learning, adaptation, safety, simulation, inference, dashboard, and tests.
- 16-dimensional GESA++ state for a 4-lane intersection.
- DQN with two 256-unit hidden layers and 50,000-transition replay buffer.
- Explicit minimum-green, maximum-red, and prohibited-transition safety constraints.
- Four scenario generators: low demand, peak imbalance, incident congestion, and festival surge.
- 5 unit tests passed in the local run.
- ~2,986.7 simulation control decisions/s (~0.335 ms/decision) for the measured non-vision control loop.

## Resume-safe wording

"Built an end-to-end AI traffic-control prototype with YOLOv8, GESA++ state aggregation, DQN reinforcement learning, KL-divergence-based fast adaptation, and a safety-rule engine; implemented 4 traffic scenarios, automated tests, video inference, and a Streamlit dashboard, and benchmarked the control loop at ~2.99K simulation decisions/s."
