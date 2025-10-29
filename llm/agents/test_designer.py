"""
Test Designer Agent
"""
import asyncio
import json
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TestDesignerAgent(BaseAgent):
    """Agent for designing test cases"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'test_designer')

    async def execute(self, input_data: Dict[str, Any]) -> dict[str, Any]:
        """Design test cases"""
        analysis = input_data.get('analyzer_results', {})
        context = input_data.get('context', {})
        config = input_data.get('config', {})

        return await self.design_tests(analysis, context)

    async def design_tests(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Design tests with SEQUENTIAL generation (not parallel!)"""

        # Generate happy path tests ONLY (most important)
        happy_path = []
        try:
            happy_path = await self._generate_happy_path_tests_simple(analysis, context)
            logger.info(f"✅ Generated {len(happy_path)} happy path tests")
        except Exception as e:
            logger.error(f"Happy path generation failed: {e}")

        # Skip boundary/validation for now (too slow)

        return {
            'happy_path_tests': happy_path,
            'edge_case_tests': [],
            'validation_tests': [],
            'total_tests': len(happy_path)
        }

    async def _generate_happy_path_tests_simple(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Generate happy path tests in ONE batch only"""

        prompt = f"""Generate 5 API test cases as JSON array.

    Endpoint: {analysis.get('endpoint', '/api/endpoint')}
    Method: {analysis.get('method', 'GET')}
    Auth: {analysis.get('auth_requirements', {})}

    Return JSON array ONLY:
    [
      {{"name": "test1", "test_type": "happy_path", "input": {{}}, "expected_status": 200}},
      {{"name": "test2", "test_type": "happy_path", "input": {{}}, "expected_status": 200}}
    ]"""

        try:
            tests = await self.generate_json_with_retry(prompt, max_retries=2)
            if isinstance(tests, list) and len(tests) > 0:
                logger.info(f"Generated {len(tests)} tests")
                return tests[:10]  # Max 10 tests
            return []
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []

    async def _generate_happy_path_tests(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Generate happy path test cases in batches"""

        all_tests = []
        batch_size = 5  # Generate 5 tests per batch
        num_batches = 3  # Total 15 tests across 3 batches

        for batch_num in range(num_batches):
            prompt = f"""Design {batch_size} happy path test cases (batch {batch_num + 1}/{num_batches}) for this API endpoint.

    API Analysis:
    {json.dumps(analysis, indent=2)}

    Generate {batch_size} distinct test scenarios. Return as JSON array.
    Each test needs: name, description, test_type, input, expected_status, expected_response, assertions

    Focus on: {"core functionality" if batch_num == 0 else "edge cases" if batch_num == 1 else "error handling"}"""

            try:
                batch_tests = await self.generate_json_with_retry(prompt)
                if isinstance(batch_tests, list):
                    all_tests.extend(batch_tests)
                logger.info(f"Generated {len(batch_tests)} tests in batch {batch_num + 1}")
            except Exception as e:
                logger.warning(f"Batch {batch_num + 1} failed: {e}")
                continue

        return all_tests[:15]  # Return max 15 tests

    async def _generate_validation_tests(self, analysis: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate validation test cases"""
        validation_rules = analysis.get('validation_rules', [])

        if not validation_rules:
            return []

        prompt = f"""Design validation test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Validation Rules: {validation_rules}

Generate test cases that verify:
1. Required field validation
2. Data type validation
3. Format validation (email, phone, etc.)
4. Length/size constraints
5. Pattern matching

Return as JSON array with test cases testing both valid and invalid inputs."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')
            test['test_type'] = 'validation'

        return response

    async def _generate_auth_tests(self, analysis: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate authentication test cases"""
        auth_req = analysis.get('auth_requirements', {})

        if not auth_req.get('required'):
            return []

        prompt = f"""Design authentication test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Auth Type: {auth_req.get('type')}
Required Scopes: {auth_req.get('scopes')}

Generate test cases for:
1. No authentication provided
2. Invalid token/credentials
3. Expired token
4. Insufficient permissions/scopes
5. Valid authentication

Return as JSON array."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')
            test['test_type'] = 'authentication'

        return response

    async def _generate_error_tests(self, analysis: Dict[str, Any],
                                    context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate error handling test cases"""
        error_scenarios = analysis.get('error_scenarios', [])

        prompt = f"""Design error handling test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Expected Error Scenarios: {error_scenarios}

Generate test cases for:
1. 400 Bad Request scenarios
2. 404 Not Found scenarios  
3. 409 Conflict scenarios
4. 500 Internal Server Error scenarios

Return as JSON array."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')
            test['test_type'] = 'error_handling'

        return response

    async def _generate_boundary_tests(self, analysis: Dict[str, Any],
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate boundary test cases"""
        parameters = analysis.get('critical_parameters', [])

        if not parameters:
            return []

        prompt = f"""Design boundary test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Parameters: {parameters}

Generate test cases for:
1. Minimum valid values
2. Maximum valid values
3. Just below minimum (invalid)
4. Just above maximum (invalid)
5. Edge cases (0, -1, empty, null)

Return as JSON array."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')
            test['test_type'] = 'boundary'

        return response

    async def _generate_performance_tests(self, analysis: Dict[str, Any],
                                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance test cases"""
        perf = analysis.get('performance', {})

        prompt = f"""Design performance test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Expected Latency: {perf.get('expected_latency', 'unknown')}
Expected Throughput: {perf.get('throughput', 'unknown')}

Generate test cases for:
1. Response time validation
2. Concurrent request handling
3. Large payload handling
4. Rate limiting validation

Return as JSON array."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')
            test['test_type'] = 'performance'

        return response

    def _format_similar_tests(self, context: Dict[str, Any]) -> str:
        """Format similar tests for prompt"""
        similar_tests = context.get('similar_tests', [])
        if not similar_tests:
            return "No similar tests available"

        formatted = []
        for test in similar_tests[:3]:
            meta = test.get('metadata', {})
            formatted.append(f"- {meta.get('name', 'Test')}: {meta.get('description', '')}")

        return "\n".join(formatted)

    def _prioritize_tests(self, test_cases: List[Dict[str, Any]],
                          max_tests: int) -> List[Dict[str, Any]]:
        """Prioritize test cases based on importance"""
        # Score each test
        for test in test_cases:
            score = 0

            # Higher priority for critical test types
            test_type = test.get('test_type', '')
            if test_type == 'authentication':
                score += 10
            elif test_type == 'validation':
                score += 8
            elif test_type == 'happy_path':
                score += 7
            elif test_type == 'error_handling':
                score += 6
            elif test_type == 'boundary':
                score += 5
            elif test_type == 'performance':
                score += 3

            # Add score for specific assertions
            assertions = test.get('assertions', [])
            score += min(len(assertions), 5)

            test['priority_score'] = score

        # Sort by priority
        test_cases.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

        return test_cases[:max_tests]