import numpy as np
import torch
import torch.nn.functional as F

class FastAdaptationModule:
    """Monitor KL shift and perform a small inner-loop policy update."""

    def __init__(self, agent, threshold=0.5, inner_steps=5, learning_rate=1e-4):
        self.agent = agent
        self.threshold = threshold
        self.inner_steps = inner_steps
        self.learning_rate = learning_rate
        self.reference = None

    @staticmethod
    def _distribution(states):
        x = np.asarray(states, dtype=np.float32).reshape(-1)
        x = np.abs(x) + 1e-6
        return x / x.sum()

    @staticmethod
    def kl_divergence(p, q):
        p = np.asarray(p, dtype=np.float64) + 1e-10
        q = np.asarray(q, dtype=np.float64) + 1e-10
        p /= p.sum(); q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    def set_reference(self, states):
        if len(states):
            self.reference = self._distribution(states)

    def detect_shift(self, states):
        if self.reference is None or len(states) == 0:
            return 0.0, False
        current = self._distribution(states)
        kl = self.kl_divergence(current, self.reference)
        return kl, kl >= self.threshold

    def adapt(self, transitions):
        if not transitions:
            return False
        model = self.agent.online
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for _ in range(self.inner_steps):
            losses = []
            for state, action, reward, _, _ in transitions:
                s = torch.tensor(state, dtype=torch.float32,
                                 device=self.agent.device).unsqueeze(0)
                q = model(s)[0, int(action)]
                target = torch.tensor(float(reward), dtype=torch.float32,
                                      device=self.agent.device)
                losses.append(F.smooth_l1_loss(q, target))
            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        return True
