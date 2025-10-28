"""
API Testing Agent - Interactive Mode
Supports file parsing or direct URL input via command line
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import aiohttp
from datetime import datetime
import json
import sys
import re

from input_processing import ParserFactory, EndpointExtractor
from rag.knowledge_base import KnowledgeBase
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from rag.retriever import Retriever
from test_execution.executor import TestExecutor
from reinforcement_learning.rl_optimizer import RLOptimizer
from config import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractiveAPITestAgent:
    """Interactive API Testing Agent with runtime configuration"""

    def __init__(self, use_ssl: bool = False):
        self.use_ssl = use_ssl
        self.auth_token = None

        logger.info("="*80)
        logger.info("API Testing Agent - Initialization")
        logger.info("="*80)

        try:
            self.knowledge_base = KnowledgeBase()
            self.vector_store = VectorStore()
            self.embedding_manager = EmbeddingManager()
            self.retriever = Retriever(self.vector_store, self.embedding_manager)
            logger.info("Knowledge base loaded")
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
            self.knowledge_base = None
            self.retriever = None

        self.rl_optimizer = self._load_rl_model()

        self.executor = TestExecutor()
        if not use_ssl:
            self.executor.ssl_verify = False

        logger.info("Agent ready")
        logger.info("="*80 + "\n")

    def _load_rl_model(self) -> Optional[RLOptimizer]:
        """Load trained RL model if available"""
        try:
            rl_model_path = paths.MODELS_DIR / 'rl_qase_trained.pth'
            if rl_model_path.exists():
                rl_optimizer = RLOptimizer()
                rl_optimizer.load_checkpoint(str(rl_model_path))
                logger.info("RL model loaded")
                return rl_optimizer
        except Exception as e:
            logger.warning(f"Could not load RL model: {e}")
        return None

    async def run_from_file(self, file_path: str, api_base_url: str, language: str = None) -> Dict:
        """Parse file and test extracted endpoints"""
        try:
            logger.info("Mode: FILE PARSING")
            logger.info(f"File: {Path(file_path).name}")
            logger.info(f"API URL: {api_base_url}")
            logger.info("="*80)

            logger.info("\n[1/4] Parsing file...")
            endpoints = await self._parse_file(file_path, api_base_url, language)
            logger.info(f"Found {len(endpoints)} endpoints")

            if not endpoints:
                return {'success': False, 'error': 'No endpoints found'}

            self._display_endpoints(endpoints)

            logger.info("\n[2/4] Generating tests...")
            test_suite = await self._generate_tests(endpoints, api_base_url)
            logger.info(f"Generated {len(test_suite)} test cases")

            logger.info("\n[3/4] Executing tests...")
            results = await self._execute_tests(test_suite, api_base_url)
            logger.info(f"Executed {len(results)} tests")

            logger.info("\n[4/4] Saving results...")
            report_path = self._save_results(results, endpoints)
            logger.info(f"Report saved: {report_path}")

            self._print_summary(results)

            return {
                'success': True,
                'endpoints': len(endpoints),
                'tests': len(results),
                'report': str(report_path),
                'results': results
            }

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def run_from_urls(self, endpoints: List[Dict], base_url: str = None) -> Dict:
        """Test directly provided endpoint URLs"""
        try:
            logger.info("Mode: DIRECT URL TESTING")
            logger.info(f"Endpoints: {len(endpoints)}")
            logger.info("="*80)

            if not base_url:
                base_url = self._extract_base_url(endpoints)

            self._display_endpoints(endpoints)

            logger.info("\n[1/3] Generating tests...")
            test_suite = await self._generate_tests(endpoints, base_url)
            logger.info(f"Generated {len(test_suite)} test cases")

            logger.info("\n[2/3] Executing tests...")
            results = await self._execute_tests(test_suite, base_url)
            logger.info(f"Executed {len(results)} tests")

            logger.info("\n[3/3] Saving results...")
            report_path = self._save_results(results, endpoints)
            logger.info(f"Report saved: {report_path}")

            self._print_summary(results)

            return {
                'success': True,
                'endpoints': len(endpoints),
                'tests': len(results),
                'report': str(report_path),
                'results': results
            }

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _parse_file(self, file_path: str, api_base_url: str, language: str = None) -> List[Dict]:
        """Parse file and extract endpoints"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not language:
            ext_map = {'.cs': 'csharp', '.py': 'python', '.java': 'java', '.cpp': 'cpp', '.h': 'cpp'}
            language = ext_map.get(file_path.suffix.lower(), 'csharp')
            logger.info(f"Detected language: {language}")

        factory = ParserFactory()
        parser = factory.get_parser(language)
        parsed_result = parser.parse([str(file_path)])

        extractor = EndpointExtractor()
        endpoints = extractor.extract(parsed_result)

        base_url = api_base_url.rstrip('/')
        for ep in endpoints:
            path = ep.get('path', '')
            if not path.startswith('/'):
                path = '/' + path
            ep['url'] = f"{base_url}{path}"
            ep['description'] = ep.get('name', f"{ep.get('method', 'GET')} {path}")

        return endpoints

    def _extract_base_url(self, endpoints: List[Dict]) -> str:
        """Extract base URL from endpoint URLs"""
        if not endpoints:
            return ""

        first_url = endpoints[0].get('url', '')
        from urllib.parse import urlparse
        parsed = urlparse(first_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _display_endpoints(self, endpoints: List[Dict]):
        """Display discovered endpoints"""
        logger.info("\nEndpoints:")
        for i, ep in enumerate(endpoints, 1):
            method = ep.get('method', 'GET')
            path = ep.get('path', ep.get('url', 'unknown'))
            logger.info(f"  {i}. {method:6} {path}")

    async def _generate_tests(self, endpoints: List[Dict], base_url: str) -> List[Dict]:
        """Generate test suite for endpoints"""
        test_suite = []

        for endpoint in endpoints:
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '')

            # Extract path parameters
            path_params = re.findall(r'\{(\w+)\}', path)

            # Generate valid test data for path parameters
            valid_test_data = {}
            for param in path_params:
                if 'id' in param.lower() or 'user' in param.lower():
                    valid_test_data[param] = 1
                elif 'year' in param.lower():
                    valid_test_data[param] = 2025
                else:
                    valid_test_data[param] = 1

            # Happy path test
            test_case = {
                'name': f"{endpoint.get('description', 'Test')} - Happy Path",
                'endpoint': path,
                'method': method,
                'test_type': 'happy_path',
                'expected_status': self._get_expected_status(method),
                'test_data': valid_test_data.copy(),
                'headers': endpoint.get('headers', {}),
            }

            if self.auth_token:
                test_case['auth_token'] = self.auth_token

            test_suite.append(test_case)

            # Knowledge-based tests
            if self.retriever:
                try:
                    kb_tests = await self._generate_kb_tests(endpoint)
                    test_suite.extend(kb_tests)
                except Exception as e:
                    logger.debug(f"KB test generation failed: {e}")

            # Edge cases - ONLY for endpoints WITH path parameters
            if path_params:
                edge_tests = self._generate_edge_tests(endpoint, path_params)
                test_suite.extend(edge_tests)

        return test_suite

    def _get_expected_status(self, method: str) -> int:
        """Expected HTTP status for method"""
        return {'GET': 200, 'POST': 201, 'PUT': 200, 'PATCH': 200, 'DELETE': 204}.get(method.upper(), 200)

    async def _generate_kb_tests(self, endpoint: Dict) -> List[Dict]:
        """Generate tests from knowledge base"""
        tests = []
        method = endpoint.get('method', 'GET')
        path = endpoint.get('path', '')

        try:
            patterns = await self.retriever.retrieve(
                query=f"{method} {path}",
                index_name='test_patterns',
                k=3
            )

            for pattern in patterns:
                test = {
                    'name': f"{pattern.get('name', 'KB Test')} - {path}",
                    'endpoint': path,
                    'method': method,
                    'test_type': pattern.get('test_type', 'functional'),
                    'expected_status': pattern.get('expected_status', 200),
                    'test_data': pattern.get('test_data', {}),
                }
                if self.auth_token:
                    test['auth_token'] = self.auth_token
                tests.append(test)
        except:
            pass

        return tests

    def _generate_edge_tests(self, endpoint: Dict, path_params: List[str]) -> List[Dict]:
        """Generate edge case tests for endpoints with path parameters"""
        tests = []
        method = endpoint.get('method', 'GET')
        path = endpoint.get('path', '')

        # Invalid ID test
        invalid_test_data = {}
        for param in path_params:
            if 'id' in param.lower():
                invalid_test_data[param] = 999999
            else:
                invalid_test_data[param] = 'invalid'

        tests.append({
            'name': f'Edge - Invalid {"/".join(path_params)} - {path}',
            'endpoint': path,
            'method': method,
            'test_type': 'edge_case',
            'expected_status': 404,
            'test_data': invalid_test_data,
            'auth_token': self.auth_token if self.auth_token else None
        })

        # Null/missing parameter test
        tests.append({
            'name': f'Edge - Missing {"/".join(path_params)} - {path}',
            'endpoint': path,
            'method': method,
            'test_type': 'edge_case',
            'expected_status': 400,
            'test_data': {},
            'auth_token': self.auth_token if self.auth_token else None
        })

        return tests

    async def _execute_tests(self, test_suite: List[Dict], base_url: str) -> List[Dict]:
        """Execute test suite"""
        results = []
        connector = aiohttp.TCPConnector(ssl=self.use_ssl)

        async with aiohttp.ClientSession(connector=connector) as session:
            self.executor.session = session

            for i, test in enumerate(test_suite, 1):
                try:
                    result = await self.executor.execute_test(test, base_url)
                    results.append(result)

                    if i % 10 == 0 or not result.get('passed'):
                        status = "PASS" if result.get('passed') else "FAIL"
                        logger.info(f"  [{i}/{len(test_suite)}] {status} - {test['name'][:50]}")
                except Exception as e:
                    results.append({'name': test['name'], 'passed': False, 'error': str(e)})

        return results

    def _save_results(self, results: List[Dict], endpoints: List[Dict]) -> Path:
        """Save test results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = paths.DATA_DIR / 'results' / f"test_results_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        passed = sum(1 for r in results if r.get('passed'))
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_endpoints': len(endpoints),
                'total_tests': len(results),
                'passed': passed,
                'failed': len(results) - passed,
                'pass_rate': f"{(passed/len(results)*100):.1f}%" if results else "0%"
            },
            'endpoints': endpoints,
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _print_summary(self, results: List[Dict]):
        """Print test summary"""
        passed = sum(1 for r in results if r.get('passed'))
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0

        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests:  {len(results)}")
        print(f"Passed:       {passed} ({pass_rate:.1f}%)")
        print(f"Failed:       {failed}")

        failures = [r for r in results if not r.get('passed')]
        if failures:
            print(f"\nFailed Tests (first 5):")
            for i, fail in enumerate(failures[:5], 1):
                print(f"  {i}. {fail['name'][:70]}")
                if fail.get('error'):
                    print(f"     Error: {fail['error'][:70]}")

        print("="*80 + "\n")


def get_user_input():
    """Interactive user input collection"""
    print("\n" + "="*80)
    print("API Testing Agent - Interactive Mode")
    print("="*80)

    print("\nSelect input mode:")
    print("  1. Parse API file (Controller, app.py, etc.)")
    print("  2. Direct URL input")

    mode = input("\nEnter choice (1 or 2): ").strip()

    if mode == '1':
        file_path = input("Enter file path: ").strip()
        api_base_url = input("Enter API base URL (e.g., https://localhost:7063): ").strip()
        language = input("Enter language (or press Enter for auto-detect): ").strip() or None

        auth_required = input("Does API require authentication? (y/n): ").strip().lower()
        auth_token = None
        if auth_required == 'y':
            auth_token = input("Enter Bearer token: ").strip()

        ssl_verify = input("Use SSL verification? (y/n): ").strip().lower() == 'y'

        return {
            'mode': 'file',
            'file_path': file_path,
            'api_base_url': api_base_url,
            'language': language,
            'auth_token': auth_token,
            'use_ssl': ssl_verify
        }

    elif mode == '2':
        endpoints = []
        print("\nEnter API endpoints (press Enter with empty URL to finish):")

        while True:
            url = input("\nEndpoint URL: ").strip()
            if not url:
                break

            method = input("HTTP Method (GET/POST/PUT/DELETE) [GET]: ").strip().upper() or 'GET'
            description = input("Description: ").strip() or f"{method} {url}"

            endpoint = {
                'url': url,
                'method': method,
                'description': description,
            }

            if method in ['POST', 'PUT', 'PATCH']:
                has_body = input("Does request have body? (y/n): ").strip().lower()
                if has_body == 'y':
                    print("Enter JSON body (one line):")
                    body_str = input().strip()
                    try:
                        endpoint['body'] = json.loads(body_str)
                    except:
                        logger.warning("Invalid JSON, skipping body")

            endpoints.append(endpoint)

        if not endpoints:
            print("No endpoints provided")
            sys.exit(1)

        auth_required = input("\nDoes API require authentication? (y/n): ").strip().lower()
        auth_token = None
        if auth_required == 'y':
            auth_token = input("Enter Bearer token: ").strip()

        ssl_verify = input("Use SSL verification? (y/n): ").strip().lower() == 'y'

        return {
            'mode': 'url',
            'endpoints': endpoints,
            'auth_token': auth_token,
            'use_ssl': ssl_verify
        }

    else:
        print("Invalid choice")
        sys.exit(1)


async def main():
    """Main entry point"""

    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
API Testing Agent - Interactive Mode

Usage:
  python run_agent.py              # Interactive mode
  python run_agent.py --help       # Show this help
""")
        return

    config = get_user_input()

    agent = InteractiveAPITestAgent(use_ssl=config.get('use_ssl', False))
    agent.auth_token = config.get('auth_token')

    if config['mode'] == 'file':
        result = await agent.run_from_file(
            file_path=config['file_path'],
            api_base_url=config['api_base_url'],
            language=config.get('language')
        )
    else:
        result = await agent.run_from_urls(
            endpoints=config['endpoints']
        )

    if result['success']:
        print("\nTest execution completed successfully")
        print(f"Report: {result['report']}")
    else:
        print(f"\nTest execution failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())