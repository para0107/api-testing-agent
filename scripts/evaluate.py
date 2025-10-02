# python
# file: scripts/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from config import rl_config
from reinforcement_learning.policy_network import PolicyNetwork

# Import from concrete modules to avoid package export issues
from test_execution.executor import TestExecutor  # type: ignore
from output.report_generator import ReportGenerator  # type: ignore


class Evaluator:
    """
    Runs the test executor and optionally generates a report.
    Can load a policy network checkpoint if provided for downstream policies.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        checkpoint: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy: nn.Module = PolicyNetwork(state_dim, action_dim).to(self.device)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device)
            if "policy_state" in ckpt:
                self.policy.load_state_dict(ckpt["policy_state"])
            else:
                # Fallback to raw state dict if the file contains only model weights
                self.policy.load_state_dict(ckpt)

        self.executor = TestExecutor()

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute tests through TestExecutor.
        Tries common method names to keep compatibility.
        """
        # Allow passing filters or runtime flags via kwargs
        for name in ("run", "execute", "run_tests"):
            if hasattr(self.executor, name):
                fn = getattr(self.executor, name)
                result = fn(**kwargs) if callable(fn) else fn
                return {"ok": True, "results": result}

        return {"ok": False, "error": "TestExecutor has no runnable entrypoint"}

    def generate_report(self, results: Any, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Generate a report using ReportGenerator if available.
        """
        out_dir = Path(getattr(rl_config, "reports_dir", "data/reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        target = output_path or (out_dir / "report.html")

        try:
            rg = ReportGenerator()
            # Try common method names on the generator
            for name in ("generate", "write", "render", "save"):
                if hasattr(rg, name):
                    fn = getattr(rg, name)
                    fn(results, target)
                    return target
        except Exception:
            pass

        return None
