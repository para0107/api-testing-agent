"""
LM Studio client for Llama 3.2
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential

from config import llama_config

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for interacting with Llama 3.2 via LM Studio"""

    def __init__(self):
        self.base_url = llama_config.base_url
        self.model = llama_config.model_name
        self.default_params = {
            'temperature': llama_config.temperature,
            'max_tokens': llama_config.max_tokens,
            'top_p': llama_config.top_p,
            'top_k': llama_config.top_k,
            'frequency_penalty': llama_config.frequency_penalty,
            'presence_penalty': llama_config.presence_penalty,
            'stop': llama_config.stop_sequences
        }
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate completion from Llama

        Args:
            prompt: Input prompt
            **kwargs: Override default parameters

        Returns:
            Generated text
        """
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)

        # Prepare request
        payload = {
            'model': self.model,
            'prompt': prompt,
            **params
        }

        # Make request
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=llama_config.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Extract text from response
                if 'choices' in result and result['choices']:
                    return result['choices'][0].get('text', '')
                return ''

        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion with conversation history

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override default parameters

        Returns:
            Generated response
        """
        # Convert to prompt format
        prompt = self._format_messages(messages)

        # Generate response
        return await self.generate(prompt, **kwargs)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for prompt"""
        formatted = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")

        # Add final assistant prompt
        formatted.append("Assistant:")

        return "\n\n".join(formatted)

    async def generate_json(self, prompt: str, schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate JSON response

        Args:
            prompt: Input prompt
            schema: Optional JSON schema for validation
            **kwargs: Override parameters

        Returns:
            Parsed JSON object
        """
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

Respond with valid JSON only. Do not include any text outside the JSON structure.
"""

        if schema:
            json_prompt += f"\nFollow this schema:\n{json.dumps(schema, indent=2)}"

        # Generate response
        response = await self.generate(json_prompt, **kwargs)

        # Parse JSON
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            raise ValueError(f"Invalid JSON response: {e}")

    async def stream_generate(self, prompt: str, callback=None, **kwargs):
        """
        Stream generation for real-time output

        Args:
            prompt: Input prompt
            callback: Async callback for each token
            **kwargs: Override parameters
        """
        params = self.default_params.copy()
        params.update(kwargs)
        params['stream'] = True

        payload = {
            'model': self.model,
            'prompt': prompt,
            **params
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        async with self.session.post(
                f"{self.base_url}/completions",
                json=payload
        ) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8').strip())
                        if 'choices' in data and data['choices']:
                            token = data['choices'][0].get('text', '')
                            if callback:
                                await callback(token)
                            else:
                                yield token
                    except json.JSONDecodeError:
                        continue

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text (if supported by LM Studio)

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        payload = {
            'model': self.model,
            'input': text
        }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                    f"{self.base_url}/embeddings",
                    json=payload
            ) as response:
                result = await response.json()
                if 'data' in result and result['data']:
                    return result['data'][0].get('embedding', [])
                return []
        except:
            logger.warning("Embeddings not supported by LM Studio")
            return []

    def get_config_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        return llama_config.get_agent_config(agent_type)