"""
Execute API Testing Agent with Pre-trained Model
Uses existing trained model and knowledge base to test APIs
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
import sys
import logging
from datetime import datetime

from input_processing import ParserFactory, EndpointExtractor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.knowledge_base import KnowledgeBase
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from rag.retriever import Retriever
from test_execution.executor import TestExecutor
from reinforcement_learning.rl_optimizer import RLOptimizer
from config import paths

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APITestAgent:
    """Execute tests using trained model and knowledge base"""

    def __init__(self, api_base_url: str, use_ssl: bool = False):
        self.api_base_url = api_base_url
        self.use_ssl = use_ssl

        logger.info("Loading trained components...")

        # Load existing components
        self.knowledge_base = KnowledgeBase()
        self.vector_store = VectorStore()
        self.embedding_manager = EmbeddingManager()
        self.retriever = Retriever(self.vector_store, self.embedding_manager)

        # Load trained RL model if available
        self.rl_optimizer = self._load_rl_model()

        # Initialize test executor with SSL handling
        self.executor = TestExecutor()
        if not use_ssl:
            self.executor.ssl_verify = False

        logger.info("Components loaded successfully")

    def _load_rl_model(self):
        """Load pre-trained RL model"""
        try:
            rl_model_path = paths.BASE_DIR / 'data' / 'models' / 'rl_qase_trained.pth'
            if rl_model_path.exists():
                rl_optimizer = RLOptimizer()
                rl_optimizer.load_checkpoint(str(rl_model_path))
                logger.info("Loaded trained RL model")
                return rl_optimizer
            else:
                logger.warning("No trained RL model found, proceeding without optimization")
                return None
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return None

    async def retrieve_relevant_tests(self, endpoint: str, method: str, limit: int = 10) -> List[Dict]:
        """Retrieve relevant test patterns from knowledge base"""

        # Create query from endpoint info
        query = f"{method} {endpoint}"

        # Search for similar test patterns
        try:
            similar_tests = await self.retriever.retrieve(
                query=query,
                index_name='test_patterns',
                k=limit
            )

            # Also search for edge cases
            edge_cases = await self.retriever.retrieve(
                query=query,
                index_name='edge_cases',
                k=5
            )

            return similar_tests + edge_cases

        except Exception as e:
            logger.error(f"Error retrieving tests: {e}")
            return []

    async def generate_test_suite(self, api_endpoints: List[Dict]) -> List[Dict]:
        """Generate test suite based on API endpoints"""

        test_suite = []

        for endpoint_info in api_endpoints:
            endpoint = endpoint_info.get('path', '')
            method = endpoint_info.get('method', 'GET')

            logger.info(f"Generating tests for {method} {endpoint}")

            # Retrieve relevant test patterns
            relevant_tests = await self.retrieve_relevant_tests(endpoint, method)

            # Generate tests based on patterns
            for pattern in relevant_tests[:5]:  # Limit to top 5 per endpoint
                test_case = {
                    'name': f"{pattern.get('name', 'Test')} - {endpoint}",
                    'endpoint': endpoint,
                    'method': method,
                    'test_type': pattern.get('test_type', 'functional'),
                    'expected_status': pattern.get('expected_status', 200),
                    'test_data': self._generate_test_data(endpoint_info, pattern),
                    'assertions': pattern.get('assertions', [])
                }
                test_suite.append(test_case)

            # Add edge cases
            edge_cases = self.knowledge_base.get_edge_cases_for_type(
                endpoint_info.get('data_type', 'string')
            )

            for edge_case in edge_cases[:3]:  # Limit edge cases
                test_case = {
                    'name': f"Edge Case - {edge_case.get('description', 'Unknown')} - {endpoint}",
                    'endpoint': endpoint,
                    'method': method,
                    'test_type': 'edge_case',
                    'expected_status': 400,  # Usually edge cases expect errors
                    'test_data': edge_case.get('value', {}),
                    'assertions': []
                }
                test_suite.append(test_case)

        logger.info(f"Generated {len(test_suite)} test cases")
        return test_suite

    def _generate_test_data(self, endpoint_info: Dict, pattern: Dict) -> Dict:
        """Generate test data based on endpoint and pattern"""

        test_data = {}

        # Use pattern's test data if available
        if pattern.get('test_data'):
            test_data.update(pattern['test_data'])

        # Add parameters from endpoint info
        for param in endpoint_info.get('parameters', []):
            param_name = param.get('name', '')
            param_type = param.get('type', 'string')

            if param_name and param_name not in test_data:
                # Generate appropriate test value based on type
                if param_type == 'integer':
                    test_data[param_name] = 1
                elif param_type == 'boolean':
                    test_data[param_name] = True
                elif param_type == 'array':
                    test_data[param_name] = []
                else:
                    test_data[param_name] = "test_value"

        return test_data

    async def execute_tests(self, test_suite: List[Dict]) -> List[Dict]:
        """Execute test suite against API"""

        logger.info(f"Executing {len(test_suite)} tests against {self.api_base_url}")

        results = []
        passed_count = 0

        # Fix SSL issue for local testing
        connector = aiohttp.TCPConnector(ssl=False) if not self.use_ssl else None

        async with aiohttp.ClientSession(connector=connector) as session:
            self.executor.session = session

            for i, test in enumerate(test_suite, 1):
                try:
                    result = await self.executor.execute_test(test, self.api_base_url)
                    results.append(result)

                    if result.get('passed'):
                        passed_count += 1
                        status = "✓ PASS"
                    else:
                        status = "✗ FAIL"

                    logger.info(f"[{i}/{len(test_suite)}] {status} - {test['name'][:60]}")

                    # Update knowledge base with results (learning)
                    self.knowledge_base.add_test_result(test, result)

                except Exception as e:
                    logger.error(f"Error executing test {test['name']}: {e}")
                    results.append({
                        'name': test['name'],
                        'passed': False,
                        'error': str(e)
                    })

        pass_rate = (passed_count / len(results) * 100) if results else 0

        logger.info(f"\nExecution Summary:")
        logger.info(f"  Total: {len(results)} tests")
        logger.info(f"  Passed: {passed_count} ({pass_rate:.1f}%)")
        logger.info(f"  Failed: {len(results) - passed_count}")

        return results

    def save_results(self, results: List[Dict], output_file: str = None):
        """Save test results to file"""

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_results_{timestamp}.json"

        output_path = Path('data/results') / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'api_url': self.api_base_url,
            'total_tests': len(results),
            'passed': sum(1 for r in results if r.get('passed')),
            'failed': sum(1 for r in results if not r.get('passed')),
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def print_report(self, results: List[Dict]):
        """Print detailed test report"""

        print("\n" + "="*80)
        print("API TEST EXECUTION REPORT")
        print("="*80)

        passed = [r for r in results if r.get('passed')]
        failed = [r for r in results if not r.get('passed')]

        print(f"\nSummary:")
        print(f"  Total Tests: {len(results)}")
        print(f"  Passed: {len(passed)} ({len(passed)/len(results)*100:.1f}%)")
        print(f"  Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

        if failed:
            print(f"\nFailed Tests:")
            for test in failed[:10]:  # Show first 10 failures
                print(f"  ✗ {test['name']}")
                if test.get('error'):
                    print(f"    Error: {test['error'][:100]}")

        print("\n" + "="*80)


async def main():
    """Main execution function"""

    # Configuration - modify these as needed
    # API_BASE_URL = "http://localhost:5276"  # Use HTTP to avoid SSL issues
    API_BASE_URL = "https://localhost:7063"  # If you want HTTPS

    # Define your API endpoints to test
    # You can load these from a file or parse from code
    factory = ParserFactory()
    parser = factory.get_parser('csharp')
    tree = parser.parse_file('C:\HTEC\Romania-parking-backend-internship-august-develop\Romania-parking-backend-internship-august-develop\HTEC_Parking\HTEC_Parking\Controllers\ReservationController.cs')
    extractor = EndpointExtractor()
    api_endpoints = extractor.extract(tree)

    # Initialize agent
    agent = APITestAgent(api_base_url=API_BASE_URL, use_ssl=False)

    # Generate test suite
    test_suite = await agent.generate_test_suite(api_endpoints)

    # Execute tests
    results = await agent.execute_tests(test_suite)

    # Save and display results
    agent.save_results(results)
    agent.print_report(results)

    # Update knowledge base for future runs
    agent.knowledge_base.export_knowledge()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("API TESTING AGENT - EXECUTION MODE")
    print("Using pre-trained model and knowledge base")
    print("="*80 + "\n")

    asyncio.run(main())