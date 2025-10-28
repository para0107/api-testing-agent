"""
API Testing Agent - Using LLM-Powered Agents (All 5 Agents)
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
from llm.agents.edge_case_agent import EdgeCaseAgent
from llm.agents.data_generator import DataGeneratorAgent
from llm.agents.report_writer import ReportWriterAgent
from llm.llama_client import LlamaClient
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from test_execution.executor import TestExecutor
from config import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMPoweredTestAgent:
    """API Testing Agent using LM Studio and all 5 LLM agents"""

    def __init__(self, use_ssl: bool = False):
        self.use_ssl = use_ssl
        self.auth_token = None

        logger.info("=" * 80)
        logger.info("LLM-Powered API Testing Agent (5 Agents)")
        logger.info("=" * 80)

        # Initialize LLM client
        logger.info("Initializing LLM client...")
        self.llm_client = LlamaClient()

        # Initialize RAG components
        logger.info("Initializing RAG components...")
        try:
            self.vector_store = VectorStore()
            self.embedding_manager = EmbeddingManager()
            self.retriever = Retriever(self.vector_store, self.embedding_manager)
            logger.info("RAG components loaded")
        except Exception as e:
            logger.warning(f"Could not load RAG: {e}")
            self.retriever = None

        # Initialize all 5 LLM-powered agents
        logger.info("Initializing 5 LLM agents...")
        self.analyzer = AnalyzerAgent(self.llm_client)
        self.test_designer = TestDesignerAgent(self.llm_client)
        self.edge_case_agent = EdgeCaseAgent(self.llm_client)
        self.data_generator = DataGeneratorAgent(self.llm_client)
        self.report_writer = ReportWriterAgent(self.llm_client)

        self.executor = TestExecutor()
        if not use_ssl:
            self.executor.ssl_verify = False

        logger.info("All 5 agents initialized successfully")
        logger.info("=" * 80 + "\n")

    async def run_from_file(self, file_path: str, api_base_url: str, auth_token: str = None):
        """Parse file and generate intelligent tests using all 5 LLM agents"""
        try:
            self.auth_token = auth_token

            logger.info("Mode: LLM-POWERED FILE PARSING (5 AGENTS)")
            logger.info(f"File: {Path(file_path).name}")
            logger.info(f"API URL: {api_base_url}")
            logger.info("=" * 80)

            # Step 1: Parse file
            logger.info("\n[1/6] Parsing API file...")
            endpoints = await self._parse_file(file_path, api_base_url)
            logger.info(f"Found {len(endpoints)} endpoints")

            # Step 2: Analyze each endpoint with LLM (Agent 1: AnalyzerAgent)
            logger.info("\n[2/6] Analyzing endpoints with LLM (AnalyzerAgent)...")
            analyses = []
            for i, endpoint in enumerate(endpoints, 1):
                logger.info(f"  Analyzing {i}/{len(endpoints)}: {endpoint.get('method')} {endpoint.get('path')}")
                try:
                    analysis = await self.analyzer.analyze(endpoint, {})
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"  Failed to analyze endpoint {i}: {e}")
                    # Create minimal analysis to continue
                    analyses.append({
                        'endpoint': endpoint.get('path'),
                        'method': endpoint.get('method'),
                        'critical_parameters': [],
                        'error': str(e)
                    })

            # Step 3: Design tests with LLM (Agent 2: TestDesignerAgent)
            logger.info("\n[3/6] Designing test cases with LLM (TestDesignerAgent)...")
            all_tests = []
            for i, analysis in enumerate(analyses, 1):
                logger.info(f"  Designing tests {i}/{len(analyses)}")
                try:
                    tests = await self.test_designer.design_tests(analysis, {}, {})
                    all_tests.extend(tests)
                except Exception as e:
                    logger.error(f"  Failed to design tests for endpoint {i}: {e}")
            logger.info(f"Generated {len(all_tests)} test cases")

            # Step 3.5: Generate edge cases with LLM (Agent 3: EdgeCaseAgent)
            logger.info("\n[3.5/6] Generating edge cases with LLM (EdgeCaseAgent)...")
            edge_cases = []
            for i, analysis in enumerate(analyses, 1):
                logger.info(f"  Generating edge cases {i}/{len(analyses)}")
                try:
                    edges = await self.edge_case_agent.generate_edge_cases(
                        {
                            'api_spec': analysis,
                            'analysis': analysis
                        }
                    )
                    edge_cases.extend(edges)
                except Exception as e:
                    logger.error(f"  Failed to generate edge cases for endpoint {i}: {e}")

            logger.info(f"Generated {len(edge_cases)} edge case tests")
            all_tests.extend(edge_cases)
            logger.info(f"Total tests (including edge cases): {len(all_tests)}")

            # Step 4: Generate test data with LLM (Agent 4: DataGeneratorAgent)
            logger.info("\n[4/6] Generating test data with LLM (DataGeneratorAgent)...")
            data_generated = 0
            for test in all_tests:
                if test.get('method') in ['POST', 'PUT', 'PATCH']:
                    try:
                        test_data = await self.data_generator.generate_data(
                            test.get('endpoint'),
                            test.get('method')
                        )
                        test['test_data'] = test_data
                        data_generated += 1
                    except Exception as e:
                        logger.warning(f"  Failed to generate data for {test.get('name', 'unknown')}: {e}")
            logger.info(f"Generated test data for {data_generated} tests")

            # Step 5: Execute tests
            logger.info("\n[5/6] Executing tests...")
            results = await self._execute_tests(all_tests, api_base_url)
            logger.info(f"Executed {len(results)} tests")

            # Step 6: Generate intelligent report with LLM (Agent 5: ReportWriterAgent)
            logger.info("\n[6/6] Generating intelligent report with LLM (ReportWriterAgent)...")
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                report_data = await self.report_writer.generate_report(
                    {
                        'execution_results': results,
                        'endpoints': endpoints,
                        'analyses': analyses
                    },
                    session_id=session_id
                )
                logger.info("Intelligent report generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate LLM report: {e}")
                report_data = {
                    'summary': 'Failed to generate LLM analysis',
                    'recommendations': [],
                    'error': str(e)
                }

            # Save results
            report_path = self._save_results(results, endpoints, report_data)
            logger.info(f"Report saved: {report_path}")

            self._print_summary(results, report_data)

            return {
                'success': True,
                'endpoints': len(endpoints),
                'tests': len(results),
                'edge_cases': len(edge_cases),
                'report': str(report_path),
                'results': results,
                'llm_report': report_data
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
                        logger.info(f"  [{i}/{len(test_suite)}] {status} - {test.get('name', 'Unknown')[:50]}")

                except Exception as e:
                    logger.error(f"  Test execution failed: {e}")
                    results.append({
                        'name': test.get('name', 'Unknown'),
                        'passed': False,
                        'error': str(e),
                        'test_type': test.get('test_type', 'unknown')
                    })

        return results

    def _save_results(self, results: List[Dict], endpoints: List[Dict], llm_report: Dict = None) -> Path:
        """Save results with LLM-generated insights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = paths.DATA_DIR / 'results' / f"llm_test_results_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        passed = sum(1 for r in results if r.get('passed'))
        failed = len(results) - passed

        # Categorize tests by type
        test_types = {}
        for result in results:
            test_type = result.get('test_type', 'unknown')
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'passed': 0, 'failed': 0}
            test_types[test_type]['total'] += 1
            if result.get('passed'):
                test_types[test_type]['passed'] += 1
            else:
                test_types[test_type]['failed'] += 1

        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'LLM-Powered (5 Agents)',
            'agents_used': [
                'AnalyzerAgent',
                'TestDesignerAgent',
                'EdgeCaseAgent',
                'DataGeneratorAgent',
                'ReportWriterAgent'
            ],
            'summary': {
                'total_endpoints': len(endpoints),
                'total_tests': len(results),
                'passed': passed,
                'failed': failed,
                'pass_rate': f"{(passed / len(results) * 100):.1f}%" if results else "0%",
                'by_test_type': test_types
            },
            'endpoints': endpoints,
            'results': results,
            'llm_analysis': llm_report  # LLM-generated insights from ReportWriterAgent
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _print_summary(self, results: List[Dict], llm_report: Dict = None):
        """Print summary with LLM insights"""
        passed = sum(1 for r in results if r.get('passed'))
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0

        print("\n" + "=" * 80)
        print("TEST SUMMARY - LLM-POWERED (5 AGENTS)")
        print("=" * 80)
        print(f"Total Tests:  {len(results)}")
        print(f"Passed:       {passed} ({pass_rate:.1f}%)")
        print(f"Failed:       {failed}")

        # Breakdown by test type
        test_types = {}
        for result in results:
            test_type = result.get('test_type', 'unknown')
            if test_type not in test_types:
                test_types[test_type] = {'passed': 0, 'failed': 0}
            if result.get('passed'):
                test_types[test_type]['passed'] += 1
            else:
                test_types[test_type]['failed'] += 1

        if test_types:
            print("\nBreakdown by Test Type:")
            for test_type, counts in test_types.items():
                total = counts['passed'] + counts['failed']
                print(f"  {test_type:15} - {counts['passed']}/{total} passed")

        failures = [r for r in results if not r.get('passed')]
        if failures:
            print(f"\nFailed Tests (showing first 5):")
            for i, fail in enumerate(failures[:5], 1):
                name = fail.get('name', 'Unknown')
                test_type = fail.get('test_type', 'unknown')
                print(f"  {i}. [{test_type}] {name[:60]}")

        # Print LLM-generated recommendations
        if llm_report:
            print("\n" + "-" * 80)
            print("LLM-GENERATED INSIGHTS (ReportWriterAgent)")
            print("-" * 80)

            if 'summary' in llm_report:
                print(f"\nOverall Assessment:")
                summary = llm_report['summary']
                if isinstance(summary, str):
                    print(f"  {summary[:200]}")
                elif isinstance(summary, dict):
                    for key, value in list(summary.items())[:3]:
                        print(f"  {key}: {value}")

            if 'recommendations' in llm_report and llm_report['recommendations']:
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(llm_report['recommendations'][:5], 1):
                    print(f"  {i}. {rec}")

            if 'critical_issues' in llm_report and llm_report['critical_issues']:
                print(f"\nCritical Issues:")
                for i, issue in enumerate(llm_report['critical_issues'][:3], 1):
                    print(f"  {i}. {issue}")

        print("=" * 80 + "\n")


async def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("LLM-Powered API Testing Agent - Using 5 Intelligent Agents")
    print("=" * 80)
    print("Agents: Analyzer | TestDesigner | EdgeCase | DataGenerator | ReportWriter")
    print("=" * 80 + "\n")

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
        print("\n" + "=" * 80)
        print("✓ Test execution completed successfully!")
        print("=" * 80)
        print(f"Endpoints analyzed: {result['endpoints']}")
        print(f"Total tests executed: {result['tests']}")
        print(f"Edge cases generated: {result.get('edge_cases', 0)}")
        print(f"Report location: {result['report']}")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("✗ Test execution failed!")
        print("=" * 80)
        print(f"Error: {result.get('error')}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())