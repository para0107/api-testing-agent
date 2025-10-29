"""
LM Studio client for Llama 3.2
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential

from config import LlamaConfig

logger = logging.getLogger(__name__)

llama_config = LlamaConfig()
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
            logger.debug(f"Sending request to LM Studio (prompt length: {len(prompt)} chars)")
            async with self.session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=llama_config.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Extract text from response
                if 'choices' in result and result['choices']:
                    text = result['choices'][0].get('text', '')
                    logger.debug(f"Received response (length: {len(text)} chars)")
                    return text
                return ''

        except aiohttp.ClientError as e:
            logger.error(f"LM Studio API request failed: {e}")
            logger.error(f"Make sure LM Studio is running at {self.base_url}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {llama_config.timeout}s")
            logger.error("LM Studio may be overloaded or the model is too slow")
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

    async def generate_json(self, prompt: str, schema: Dict[str, Any] = None, **kwargs):
        """Generate JSON with aggressive stopping"""

        # ULTRA-STRICT instruction
        json_prompt = f"""{prompt}

    ===CRITICAL RULES===
    1. Start response with {{ or [
    2. End response with }} or ]
    3. STOP IMMEDIATELY after closing brace
    4. NO explanations
    5. NO markdown
    6. NO notes
    ===END RULES==="""

        if schema:
            json_prompt += f"\n\nRequired structure:\n{json.dumps(schema, indent=2)}"

        params = {}
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                params[key] = value

        # Optimized parameters
        params['temperature'] = 0.4  # VERY LOW for strict following
        params['max_tokens'] = 900 # Reduced
        params['top_k'] = 40  # Very focused sampling
        params['top_p'] = 0.95  # Lower for determinism
        params['frequency_penalty'] = 0.6  # High to prevent repetition
        params['presence_penalty'] = 0.6  # High to prevent verbosity

        # AGGRESSIVE stop sequences to prevent explanations
        params['stop'] = ['<|im_end|>', '<|endoftext|>', '```','\n]\n',']\n\n', ]

        response = await self.generate(json_prompt, **params)

        # Clean and extract
        try:
            cleaned = response.replace('```json', '').replace('```', '')
            cleaned = self._extract_json_aggressively(response)
            if not cleaned:
                raise ValueError("Empty response after extraction")
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response:\n{response[:1000]}")
            raise ValueError(f"Invalid JSON: {e}")




    async def check_connection(self) -> bool:
        """Check if LM Studio server is responsive"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                    f"{self.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"LM Studio connection check failed: {e}")
            return False

    def _extract_json_aggressively(self, response: str) -> str:
        """Aggressively extract ONLY the JSON portion"""

        # Remove everything before first { or [
        json_start = min(
            response.find('{') if '{' in response else len(response),
            response.find('[') if '[' in response else len(response)
        )

        if json_start == len(response):
            raise ValueError("No JSON found in response")

        response = response[json_start:]

        # Find the matching closing brace/bracket
        # This handles nested structures properly
        stack = []
        json_end = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(response):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack:
                    stack.pop()
                    if not stack:  # Found complete JSON
                        json_end = i + 1
                        break

        if json_end > 0:
            response = response[:json_end]

        return response.strip()

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