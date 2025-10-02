"""
Base agent class for LLM agents
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LLM agents"""

    def __init__(self, llama_client, agent_type: str = None):
        self.client = llama_client
        self.agent_type = agent_type or self.__class__.__name__
        self.config = self._get_config()

    def _get_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        return self.client.get_config_for_agent(self.agent_type)

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Any:
        """
        Execute agent task

        Args:
            input_data: Input data for the agent

        Returns:
            Agent output
        """
        pass

    async def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate with retry logic"""
        for attempt in range(max_retries):
            try:
                response = await self.client.generate(prompt, **self.config)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        return ""

    async def generate_json_with_retry(self, prompt: str, schema: Dict[str, Any] = None,
                                       max_retries: int = 3) -> Dict[str, Any]:
        """Generate JSON with retry logic"""
        for attempt in range(max_retries):
            try:
                response = await self.client.generate_json(prompt, schema, **self.config)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        return {}

    def format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt"""
        formatted_parts = []

        if 'similar_tests' in context:
            formatted_parts.append("Similar Test Cases:")
            for test in context['similar_tests'][:3]:  # Limit to top 3
                formatted_parts.append(f"- {test.get('metadata', {}).get('name', 'Test')}")

        if 'edge_cases' in context:
            formatted_parts.append("\nRelevant Edge Cases:")
            for edge in context['edge_cases'][:3]:
                formatted_parts.append(f"- {edge.get('metadata', {}).get('description', 'Edge case')}")

        if 'validation_patterns' in context:
            formatted_parts.append("\nValidation Patterns:")
            for pattern in context['validation_patterns'][:3]:
                formatted_parts.append(f"- {pattern.get('metadata', {}).get('type', 'Validation')}")

        return "\n".join(formatted_parts)

    def validate_response(self, response: Any) -> bool:
        """Validate agent response"""
        return response is not None and response != ""