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
        self.llama_orchestrator = LlamaOrchestrator()
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

        # Generate embeddings for API specification
        embeddings = await self.rag_system.generate_embeddings(api_spec)

        # Retrieve similar test cases
        similar_tests = await self.rag_system.retrieve_similar_tests(embeddings)

        # Retrieve edge cases
        edge_cases = await self.rag_system.retrieve_edge_cases(embeddings)

        # Retrieve validation patterns
        validation_patterns = await self.rag_system.retrieve_validation_patterns(embeddings)

        return {
            'similar_tests': similar_tests,
            'edge_cases': edge_cases,
            'validation_patterns': validation_patterns,
            'embeddings': embeddings
        }

    async def _generate_tests(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases using LLM"""
        logger.info("Generating test cases with LLM")

        # Orchestrate multiple agents
        test_cases = await self.llama_orchestrator.generate_test_suite(
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