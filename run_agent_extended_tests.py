"""
API Testing Agent - Using LLM-Powered Agents
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json

from input_processing import ParserFactory, EndpointExtractor
from llm.agents.analyzer_agent import AnalyzerAgent
from llm.agents.test_designer import TestDesignerAgent
from llm.agents.data_generator import DataGeneratorAgent
from llm.llama_client import LlamaClient
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from test_execution.executor import TestExecutor
from config import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMPoweredTestAgent:
    """API Testing Agent using LM Studio and 5 agents"""

    def __init__(self, use_ssl: bool = False):
        self.use_ssl = use_ssl
        self.auth_token = None

        logger.info("=" * 80)
        logger.info("LLM-Powered API Testing Agent")
        logger.info("=" * 80)

        # Initialize LLM client
        self.llm_client = LlamaClient()

        # Initialize RAG components
        try:
            self.vector_store = VectorStore()
            self.embedding_manager = EmbeddingManager()
            self.retriever = Retriever(self.vector_store, self.embedding_manager)
            logger.info("RAG components loaded")
        except Exception as e:
            logger.warning(f"Could not load RAG: {e}")
            self.retriever = None

        # Initialize LLM-powered agents
        self.analyzer = AnalyzerAgent(self.llm_client)
        self.test_designer = TestDesignerAgent(self.llm_client)
        self.data_generator = DataGeneratorAgent(self.llm_client)

        self.executor = TestExecutor()
        if not use_ssl:
            self.executor.ssl_verify = False

        logger.info("All agents initialized")
        logger.info("=" * 80 + "\n")

    async def run_from_file(self, file_path: str, api_base_url: str, auth_token: str = None):
        """Parse file and generate intelligent tests"""
        try:
            self.auth_token = auth_token

            logger.info("Mode: LLM-POWERED FILE PARSING")
            logger.info(f"File: {Path(file_path).name}")
            logger.info(f"API URL: {api_base_url}")
            logger.info("=" * 80)

            # Step 1: Parse file
            logger.info("\n[1/5] Parsing API file...")
            endpoints = await self._parse_file(file_path, api_base_url)
            logger.info(f"Found {len(endpoints)} endpoints")

            # Step 2: Analyze each endpoint with LLM
            logger.info("\n[2/5] Analyzing endpoints with LLM...")
            analyses = []
            for i, endpoint in enumerate(endpoints, 1):
                logger.info(f"  Analyzing {i}/{len(endpoints)}: {endpoint.get('method')} {endpoint.get('path')}")
                analysis = await self.analyzer.analyze(endpoint, {})
                analyses.append(analysis)

            # Step 3: Design tests with LLM
            logger.info("\n[3/5] Designing tests with LLM...")
            all_tests = []
            for i, analysis in enumerate(analyses, 1):
                logger.info(f"  Designing tests {i}/{len(analyses)}")
                tests = await self.test_designer.design_tests(analysis, {}, {})
                all_tests.extend(tests)
            logger.info(f"Generated {len(all_tests)} test cases")

            # Step 4: Generate test data with LLM
            logger.info("\n[4/5] Generating test data with LLM...")
            for test in all_tests:
                if test.get('method') in ['POST', 'PUT', 'PATCH']:
                    test_data = await self.data_generator.generate_data(
                        test.get('endpoint'),
                        test.get('method')
                    )
                    test['test_data'] = test_data

            # Step 5: Execute tests
            logger.info("\n[5/5] Executing tests...")
            results = await self._execute_tests(all_tests, api_base_url)
            logger.info(f"Executed {len(results)} tests")

            # Save results
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

    async def _parse_file(self, file_path: str, api_base_url: str) -> List[Dict]:
        """Parse API file"""
        file_path = Path(file_path)

        ext_map = {'.cs': 'csharp', '.py': 'python', '.java': 'java'}
        language = ext_map.get(file_path.suffix.lower(), 'csharp')

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

        return endpoints

    async def _execute_tests(self, test_suite: List[Dict], base_url: str) -> List[Dict]:
        """Execute tests"""
        import aiohttp

        results = []
        connector = aiohttp.TCPConnector(ssl=self.use_ssl)

        async with aiohttp.ClientSession(connector=connector) as session:
            self.executor.session = session

            for i, test in enumerate(test_suite, 1):
                try:
                    # Add auth token
                    if self.auth_token:
                        test['auth_token'] = self.auth_token

                    result = await self.executor.execute_test(test, base_url)
                    results.append(result)

                    if i % 5 == 0 or not result.get('passed'):
                        status = "PASS" if result.get('passed') else "FAIL"
                        logger.info(f"  [{i}/{len(test_suite)}] {status} - {test['name'][:50]}")

                except Exception as e:
                    results.append({'name': test['name'], 'passed': False, 'error': str(e)})

        return results

    def _save_results(self, results: List[Dict], endpoints: List[Dict]) -> Path:
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = paths.DATA_DIR / 'results' / f"llm_test_results_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        passed = sum(1 for r in results if r.get('passed'))
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'LLM-Powered',
            'summary': {
                'total_endpoints': len(endpoints),
                'total_tests': len(results),
                'passed': passed,
                'failed': len(results) - passed,
                'pass_rate': f"{(passed / len(results) * 100):.1f}%" if results else "0%"
            },
            'endpoints': endpoints,
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _print_summary(self, results: List[Dict]):
        """Print summary"""
        passed = sum(1 for r in results if r.get('passed'))
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0

        print("\n" + "=" * 80)
        print("TEST SUMMARY - LLM-POWERED")
        print("=" * 80)
        print(f"Total Tests:  {len(results)}")
        print(f"Passed:       {passed} ({pass_rate:.1f}%)")
        print(f"Failed:       {failed}")

        failures = [r for r in results if not r.get('passed')]
        if failures:
            print(f"\nFailed Tests:")
            for i, fail in enumerate(failures[:5], 1):
                print(f"  {i}. {fail['name'][:70]}")

        print("=" * 80 + "\n")


async def main():
    """Main entry point"""

    file_path = input("Enter API file path: ").strip()
    api_url = input("Enter API base URL: ").strip()

    auth_required = input("Requires authentication? (y/n): ").strip().lower()
    auth_token = None
    if auth_required == 'y':
        auth_token = input("Enter Bearer token: ").strip()

    ssl_verify = input("Use SSL verification? (y/n): ").strip().lower() == 'y'

    agent = LLMPoweredTestAgent(use_ssl=ssl_verify)

    result = await agent.run_from_file(file_path, api_url, auth_token)

    if result['success']:
        print("\nTest execution completed!")
        print(f"Report: {result['report']}")
    else:
        print(f"\nFailed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())