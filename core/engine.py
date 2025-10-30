"""
Main orchestration engine for the API Testing Agent
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from config import settings, llama_config, rag_config, rl_config
from input_processing import InputProcessor
from llm import LlamaOrchestrator
from rag import RAGSystem
from llm.llama_client import LlamaConfig, LlamaClient
from reinforcement_learning import RLOptimizer
from test_execution.executor import TestExecutor
from feedback.feedback_loop import FeedbackLoop
from output.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class APITestRequest:
    """Request for API testing"""
    code_files: List[str]
    language: str
    endpoint_url: str
    test_types: List[str] = None
    max_tests: int = 50
    include_edge_cases: bool = True


class CoreEngine:
    """Main orchestration engine coordinating all components"""

    def __init__(self):
        logger.info("Initializing Core Engine")

        # Initialize components
        self.input_processor = InputProcessor()
        self.rag_system = RAGSystem()
        self.llama_orchestrator = None  # Will be initialized in process_api
        self.rl_optimizer = RLOptimizer()
        self.test_executor = TestExecutor()
        self.feedback_loop = FeedbackLoop()
        self.report_generator = ReportGenerator()

        # State management
        self.current_session = None
        self.execution_history = []
        self.metrics = {}
    async def process_api(self, request: APITestRequest) -> Dict[str, Any]:
        """
        Main processing pipeline for API testing

        Args:
            request: API test request containing code and configuration

        Returns:
            Dictionary containing test results and report
        """
        try:
            logger.info(f"Processing API test request for {request.language} endpoint")

            # Create session
            self.current_session = self._create_session(request)

            # Step 1: Parse and analyze API code
            api_spec = await self._analyze_api(request)

            # Step 2: Retrieve relevant context from RAG
            context = await self._retrieve_context(api_spec)

            # Step 3: Generate test cases using LLM
            test_cases = await self._generate_tests(api_spec, context)

            # Step 4: Optimize tests with RL
            optimized_tests = await self._optimize_tests(test_cases, api_spec)

            # Step 5: Execute tests
            execution_results = await self._execute_tests(optimized_tests, request.endpoint_url)

            # Step 6: Process feedback
            await self._process_feedback(execution_results)

            # Step 7: Generate report
            report = await self._generate_report(execution_results)

            # Update metrics
            self._update_metrics(execution_results)

            return {
                'status': 'success',
                'session_id': self.current_session['id'],
                'api_specification': api_spec,
                'test_cases': optimized_tests,
                'execution_results': execution_results,
                'report': report,
                'metrics': self.metrics
            }

        except Exception as e:
            logger.error(f"Error in core engine: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'session_id': self.current_session['id'] if self.current_session else None
            }

    async def _analyze_api(self, request: APITestRequest) -> Dict[str, Any]:
        """Analyze API code and extract specifications"""
        logger.info("Analyzing API code")

        # Parse code files
        parsed_data = self.input_processor.parse_code(
            code_files=request.code_files,
            language=request.language
        )

        # Extract API specification
        api_spec = self.input_processor.build_specification(parsed_data)

        # Enrich with business logic analysis
        api_spec['business_logic'] = self.input_processor.extract_business_logic(parsed_data)
        api_spec['validation_rules'] = self.input_processor.extract_validation_rules(parsed_data)

        return api_spec

    async def _retrieve_context(self, api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context from RAG system"""
        logger.info("Retrieving context from RAG system")

        # ✅ FIX: Create search text from endpoints
        endpoints = api_spec.get('endpoints', [])
        if not endpoints:
            logger.warning("No endpoints in API spec for RAG search")
            return {
                'similar_tests': [],
                'edge_cases': [],
                'validation_patterns': []
            }

        # Build search query from endpoints
        search_parts = []
        for endpoint in endpoints[:5]:  # Use first 5 endpoints
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', endpoint.get('endpoint', ''))
            search_parts.append(f"{method} {path}")

        search_text = " ".join(search_parts)
        logger.info(f"RAG search text: {search_text[:200]}...")

        # Generate embedding from search text
        embedding = await self.rag_system.embedding_manager.embed_text(search_text)
        logger.info("Generated embeddings for RAG search")

        # Search with proper k value
        k = 10  # Retrieve top 10 matches

        # Retrieve from each index
        try:
            similar_tests = self.rag_system.vector_store.search('test_patterns', embedding, k=k)
            edge_cases = self.rag_system.vector_store.search('edge_cases', embedding, k=k)
            validation_patterns = self.rag_system.vector_store.search('validation_rules', embedding, k=k)

            # Convert to format expected by agents: list of (score, metadata) tuples
            similar_tests_formatted = self._format_search_results(similar_tests)
            edge_cases_formatted = self._format_search_results(edge_cases)
            validation_patterns_formatted = self._format_search_results(validation_patterns)

            logger.info(f"RAG retrieval complete: {len(similar_tests_formatted)} similar tests, "
                        f"{len(edge_cases_formatted)} edge cases, {len(validation_patterns_formatted)} validation patterns")

            if len(similar_tests_formatted) == 0:
                logger.warning("No similar tests found in RAG")
            if len(edge_cases_formatted) == 0:
                logger.warning("No edge cases found in RAG")
            if len(validation_patterns_formatted) == 0:
                logger.warning("No validation patterns found in RAG")

            return {
                'similar_tests': similar_tests_formatted,
                'edge_cases': edge_cases_formatted,
                'validation_patterns': validation_patterns_formatted
            }

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {
                'similar_tests': [],
                'edge_cases': [],
                'validation_patterns': []
            }

    def _format_search_results(self, search_results) -> List[tuple]:
        """Format vector store search results to (distance, metadata) tuples"""
        ids, distances, metadata_list = search_results

        results = []
        for dist, meta in zip(distances, metadata_list):
            if meta:  # Only include non-empty metadata
                results.append((dist, meta))

        return results

    async def _generate_tests(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases using LLM"""
        logger.info("Generating test cases with LLM")

        # Use orchestrator as context manager
        async with LlamaOrchestrator() as orchestrator:
            # Check LM Studio connection
            if not await orchestrator.client.check_connection():
                raise RuntimeError(
                    "Cannot connect to LM Studio at http://127.0.0.1:1234. "
                    "Please ensure LM Studio is running and the server is started."
                )

            # Orchestrate multiple agents
            test_cases = await orchestrator.generate_test_suite(
                api_spec=api_spec,
                context=context,
                config={
                    'max_tests': self.current_session['request'].max_tests,
                    'include_edge_cases': self.current_session['request'].include_edge_cases,
                    'test_types': self.current_session['request'].test_types
                }
            )

        return test_cases

    async def _optimize_tests(self, test_cases: List[Dict[str, Any]], api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize test cases using reinforcement learning"""
        logger.info("Optimizing test cases with RL")

        # Create state representation
        state = self.rl_optimizer.create_state(test_cases, api_spec)

        # Get optimal test selection and ordering
        optimized_tests = await self.rl_optimizer.optimize(state, test_cases)

        return optimized_tests

    async def _execute_tests(self, test_cases: List[Dict[str, Any]], endpoint_url: str) -> List[Dict[str, Any]]:
        """Execute test cases against the API"""
        logger.info(f"Executing {len(test_cases)} test cases")

        # Execute tests in parallel
        execution_results = await self.test_executor.execute_batch(
            test_cases=test_cases,
            endpoint_url=endpoint_url,
            parallel=True
        )

        return execution_results

    async def _process_feedback(self, execution_results: List[Dict[str, Any]]):
        """Process feedback from test execution"""
        logger.info("Processing execution feedback")

        # Update RAG system with new patterns
        await self.feedback_loop.update_rag(execution_results)

        # Update RL model
        await self.feedback_loop.update_rl_model(execution_results)

        # Detect API drift
        drift_detected = await self.feedback_loop.detect_drift(execution_results)
        if drift_detected:
            logger.warning("API drift detected")

    async def _generate_report(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate QASE-style report"""
        logger.info("Generating test report")

        report = await self.report_generator.generate_qase_report(
            execution_results=execution_results,
            session=self.current_session
        )

        return report

    def _create_session(self, request: APITestRequest) -> Dict[str, Any]:
        """Create a new testing session"""
        return {
            'id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'request': request,
            'started_at': datetime.now(),
            'status': 'in_progress'
        }

    def _update_metrics(self, execution_results: List[Dict[str, Any]]):
        """Update performance metrics"""
        total_tests = len(execution_results)
        passed_tests = sum(1 for r in execution_results if r['passed'])

        self.metrics.update({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'bugs_found': sum(r.get('bugs_found', 0) for r in execution_results),
            'edge_cases_covered': sum(r.get('edge_cases_covered', 0) for r in execution_results)
        })

        logger.info(f"Metrics updated: {self.metrics}")