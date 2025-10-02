# python
# file: scripts/train.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from config import rl_config
from reinforcement_learning.policy_network import PolicyNetwork
from reinforcement_learning.value_network import ValueNetwork
from reinforcement_learning.experience_buffer import ExperienceBuffer, Experience
from reinforcement_learning.rl_optimizer import RLOptimizer


class RLTrainer:
    """
    Minimal trainer that wires Policy, Value, Buffer, and Optimizer.
    Assumes external code populates the ExperienceBuffer with Experience items.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defaults are taken from rl_config when available
        capacity = int(getattr(rl_config, "buffer_capacity", 10000))

        self.policy_net: nn.Module = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_net: nn.Module = ValueNetwork(self.state_dim)
        self.buffer: ExperienceBuffer = ExperienceBuffer(capacity=capacity)
        self.optimizer = RLOptimizer(self.policy_net, self.value_net, self.buffer, device=self.device)

        base_dir = Path(getattr(rl_config, "training_dir", "data/training"))
        self.save_dir = Path(save_dir) if save_dir else base_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_experience(self, exp: Experience) -> None:
        """Push a single Experience into the buffer."""
        self.buffer.push(exp)

    def train_step(self) -> Dict[str, Any]:
        """Run a single optimizer step on a sampled batch."""
        return self.optimizer.train_step()

    def fit(self, steps: int) -> None:
        """Run multiple train steps with lightweight logging."""
        steps = int(steps)
        t0 = time.time()
        for i in range(1, steps + 1):
            stats = self.train_step()
            if i % int(getattr(rl_config, "log_interval", 50)) == 0 and not stats.get("skipped"):
                print(f"[step={i}] policy_loss={stats['policy_loss']:.4f} "
                      f"value_loss={stats['value_loss']:.4f} "
                      f"entropy={stats['entropy']:.4f} "
                      f"td_error={stats['td_error']:.4f} "
                      f"batch={stats['batch_size']}")
        dt = time.time() - t0
        print(f"Training finished in {dt:.2f}s")

    def save_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        """Save models and optimizer states."""
        ckpt = {
            "policy_state": self.policy_net.state_dict(),
            "value_state": self.value_net.state_dict(),
            "policy_optim": self.optimizer.policy_optim.state_dict(),
            "value_optim": self.optimizer.value_optim.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        path = self.save_dir / name
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load models and optimizer states."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.value_net.load_state_dict(ckpt["value_state"])
        try:
            self.optimizer.policy_optim.load_state_dict(ckpt["policy_optim"])
            self.optimizer.value_optim.load_state_dict(ckpt["value_optim"])
        except Exception:
            pass  # Allow loading only model weights if optimizer state is missing
