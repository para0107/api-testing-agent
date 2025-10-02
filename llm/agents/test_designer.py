"""
Test Designer Agent
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TestDesignerAgent(BaseAgent):
    """Agent for designing test cases"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'test_designer')

    async def execute(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design test cases"""
        analysis = input_data.get('analyzer_results', {})
        context = input_data.get('context', {})
        config = input_data.get('config', {})

        return await self.design_tests(analysis, context, config)

    async def design_tests(self, analysis: Dict[str, Any],
                           context: Dict[str, Any],
                           config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Design comprehensive test cases

        Args:
            analysis: API analysis results
            context: Retrieved context
            config: Test generation configuration

        Returns:
            List of test cases
        """
        test_cases = []

        # Generate different types of tests
        test_types = config.get('test_types', [
            'happy_path', 'validation', 'authentication',
            'error_handling', 'boundary', 'performance'
        ])

        for test_type in test_types:
            if test_type == 'happy_path':
                tests = await self._generate_happy_path_tests(analysis, context)
            elif test_type == 'validation':
                tests = await self._generate_validation_tests(analysis, context)
            elif test_type == 'authentication':
                tests = await self._generate_auth_tests(analysis, context)
            elif test_type == 'error_handling':
                tests = await self._generate_error_tests(analysis, context)
            elif test_type == 'boundary':
                tests = await self._generate_boundary_tests(analysis, context)
            elif test_type == 'performance':
                tests = await self._generate_performance_tests(analysis, context)
            else:
                continue

            test_cases.extend(tests)

        # Limit number of tests if specified
        max_tests = config.get('max_tests', 50)
        if len(test_cases) > max_tests:
            test_cases = self._prioritize_tests(test_cases, max_tests)

        return test_cases

    async def _generate_happy_path_tests(self, analysis: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate happy path test cases"""
        prompt = f"""Design happy path test cases for the following API:

Endpoint: {analysis.get('endpoint')}
Method: {analysis.get('method')}
Parameters: {analysis.get('critical_parameters')}

Similar successful tests:
{self._format_similar_tests(context)}

Generate 3-5 happy path test cases that cover:
1. Basic successful operation with valid data
2. Different valid parameter combinations
3. Expected successful responses

Return as JSON array with structure:
[{{
    "name": "descriptive test name",
    "description": "what this test validates",
    "test_type": "happy_path",
    "input": {{"parameter": "value"}},
    "expected_status": 200,
    "expected_response": {{"key": "expected value"}},
    "assertions": ["list of assertions to verify"]
}}]"""

        response = await self.generate_json_with_retry(prompt)

        # Ensure response is a list
        if isinstance(response, dict):
            response = [response]

        # Add metadata
        for test in response:
            test['endpoint'] = analysis.get('endpoint')
            test['method'] = analysis.get('method')

        return response

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