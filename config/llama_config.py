"""
Llama 3.2 configuration for LM Studio
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class LlamaConfig:
    """Configuration for Llama 3.2 via LM Studio"""

    # Connection settings
    base_url: str = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key: str = os.getenv("LLAMA_API_KEY", "not-needed")  # LM Studio doesn't require API key
    model_name: str = os.getenv("LLAMA_MODEL", "llama-3.2-3b-instruct")

    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    frequency_penalty: float = 0.3
    presence_penalty: float = 0.3
    context_window: int = 8192

    # Agent-specific temperatures
    agent_temperatures: Dict[str, float] = field(default_factory=lambda: {
        'analyzer': 0.3,  # Low temperature for analysis
        'test_designer': 0.5,  # Moderate for test design
        'edge_case': 0.8,  # Higher for creative edge cases
        'data_generator': 0.6,  # Moderate for data generation
        'report_writer': 0.4  # Low for consistent reports
    })

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 60

    # Response settings
    stream: bool = False
    stop_sequences: list = field(default_factory=lambda: ["</response>", "\n\n\n"])

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        config = {
            'temperature': self.agent_temperatures.get(agent_type, self.temperature),
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stream': self.stream,
            'stop': self.stop_sequences
        }
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'base_url': self.base_url,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'context_window': self.context_window
        }


# Global instance
llama_config = LlamaConfig()