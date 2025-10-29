"""
Test generation pipeline orchestrator
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

from config import settings
from core.engine import CoreEngine
from llm import LlamaOrchestrator
from utils.validators import is_valid_test_case, is_valid_api_spec

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a stage in the test generation pipeline"""
    name: str
    function: callable
    required: bool = True
    timeout: int = 60


class TestGenerationPipeline:
    """Manages the complete test generation pipeline"""

    def __init__(self):
        logger.info("Initializing Test Generation Pipeline")

        self.engine = CoreEngine()
        self.stages = self._define_stages()
        self.stage_results = {}
        self.pipeline_metrics = {}

    def _define_stages(self) -> List[PipelineStage]:
        """Define pipeline stages"""
        return [
            PipelineStage("validation", self._validate_input, required=True, timeout=10),
            PipelineStage("parsing", self._parse_code, required=True, timeout=30),
            PipelineStage("analysis", self._analyze_api, required=True, timeout=60),
            PipelineStage("retrieval", self._retrieve_context, required=True, timeout=30),
            PipelineStage("generation", self._generate_tests, required=True, timeout=120),
            PipelineStage("optimization", self._optimize_tests, required=False, timeout=60),
            PipelineStage("execution", self._execute_tests, required=True, timeout=300),
            PipelineStage("feedback", self._process_feedback, required=False, timeout=30),
            PipelineStage("reporting", self._generate_report, required=True, timeout=30)
        ]

    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete pipeline"""
        logger.info("Starting test generation pipeline")

        self._initialize_pipeline(request)

        start_time = datetime.now()
        stages_completed = []

        try:
            for stage in self.stages:
                try:
                    logger.info(f"Executing stage: {stage.name}")

                    # Execute stage with timeout
                    await asyncio.wait_for(
                        self._execute_stage(stage, request),
                        timeout=stage.timeout
                    )

                    stages_completed.append(stage.name)

                except asyncio.TimeoutError:
                    error_msg = f"Stage {stage.name} timed out after {stage.timeout}s"
                    logger.error(error_msg)
                    if stage.required:
                        raise Exception(error_msg)
                    logger.warning(f"Skipping optional stage: {stage.name}")

                except Exception as e:
                    logger.error(f"Stage {stage.name} failed: {str(e)}")
                    if stage.required:
                        raise
                    logger.warning(f"Skipping optional stage: {stage.name}")

            # Calculate metrics
            self._calculate_metrics(start_time)

            # Prepare results
            results = self._finalize_pipeline()

            return {
                'status': 'success',
                'stages_completed': stages_completed,
                'results': results,
                'metrics': self.pipeline_metrics
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'stages_completed': stages_completed,
                'metrics': self.pipeline_metrics
            }

    async def _execute_stage(self, stage: PipelineStage, request: Dict[str, Any]):
        """Execute a single pipeline stage"""
        stage_start = datetime.now()

        try:
            # Execute stage function
            result = await stage.function(request)

            # Store result
            self.stage_results[stage.name] = result

            # Log completion
            duration = (datetime.now() - stage_start).total_seconds()
            logger.info(f"Stage {stage.name} completed in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Stage {stage.name} error: {str(e)}", exc_info=True)
            raise

    async def _validate_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        # Validate required fields
        required_fields = ['code_files', 'language', 'endpoint_url']
        for field in required_fields:
            if field not in request:
                raise ValueError(f"Missing required field: {field}")

        # Validate language support
        if request['language'] not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {request['language']}")

        # Validate code files exist
        for file_path in request['code_files']:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Code file not found: {file_path}")

        return {'validation': 'passed'}

    async def _parse_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse source code"""
        logger.info("Analyzing API code")

        # Use InputProcessor to parse code
        parsed_data = self.engine.input_processor.parse_code(
            request['code_files'],
            request['language']
        )

        return parsed_data

    async def _analyze_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API specification"""
        parsed_data = self.stage_results.get('parsing', {})

        # Build specification
        api_spec = self.engine.input_processor.build_specification(parsed_data)

        # Extract validation rules
        validation_rules = self.engine.input_processor.extract_validation_rules(parsed_data)
        api_spec['validation_rules'] = validation_rules

        # Extract business logic
        business_logic = self.engine.input_processor.extract_business_logic(parsed_data)
        api_spec['business_logic'] = business_logic

        # Validate API specification
        if not is_valid_api_spec(api_spec):
            raise ValueError("Invalid API specification")

        return api_spec

    async def _retrieve_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve context from RAG"""
        api_spec = self.stage_results.get('analysis', {})

        # Generate embeddings for API spec
        context = {}

        try:
            # Get embeddings
            if api_spec.get('endpoints'):
                endpoint_text = json.dumps(api_spec['endpoints'])
                embeddings = await self.engine.rag_system.generate_embeddings(endpoint_text)

                # Retrieve similar tests
                similar_tests = await self.engine.rag_system.retrieve_similar_tests(embeddings)
                context['similar_tests'] = similar_tests

                # Retrieve edge cases
                edge_cases = await self.engine.rag_system.retrieve_edge_cases(embeddings)
                context['edge_cases'] = edge_cases

                # Retrieve validation patterns
                validation_patterns = await self.engine.rag_system.retrieve_validation_patterns(embeddings)
                context['validation_patterns'] = validation_patterns

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}, continuing without context")
            context = {'similar_tests': [], 'edge_cases': [], 'validation_patterns': []}

        return context

    async def _generate_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases using LLM agents"""
        api_spec = self.stage_results.get('analysis', {})
        context = self.stage_results.get('retrieval', {})

        # Create config for test generation
        config = {
            'max_tests': request.get('max_tests', 50),
            'include_edge_cases': request.get('include_edge_cases', True)
        }

        # Generate tests using LLM orchestrator with proper session management
        async with LlamaOrchestrator() as orchestrator:
            # Verify LM Studio is running
            logger.info("Checking LM Studio connection...")
            if not await orchestrator.client.check_connection():
                raise RuntimeError(
                    "❌ Cannot connect to LM Studio!\n"
                    "Please ensure:\n"
                    "  1. LM Studio is installed and running\n"
                    "  2. A model is loaded (llama-3.2-3b-instruct recommended)\n"
                    "  3. Server is started on http://127.0.0.1:1234\n"
                    "  4. Check LM Studio logs for errors"
                )
            logger.info("✓ LM Studio connection successful")

            # Generate tests - Returns dict with: analysis, test_cases, edge_cases, test_data
            result = await orchestrator.generate_test_suite(
                api_spec,
                context,
                config
            )

        # Extract test cases and edge cases from result
        test_cases = result.get('test_cases', [])
        edge_cases = result.get('edge_cases', [])

        # Combine all tests
        all_tests = test_cases + edge_cases

        # Store analysis and test data for later use
        if 'analysis' in result:
            self.stage_results['llm_analysis'] = result['analysis']
        if 'test_data' in result:
            self.stage_results['test_data'] = result['test_data']

        # Validate generated tests
        valid_tests = []
        for test in all_tests:
            if is_valid_test_case(test):
                valid_tests.append(test)
            else:
                logger.warning(f"Invalid test case generated: {test.get('name', 'unknown')}")

        logger.info(f"Generated {len(test_cases)} test cases and {len(edge_cases)} edge cases")
        logger.info(f"Total valid tests: {len(valid_tests)}")

        return valid_tests



    async def _optimize_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize test cases with RL"""
        test_cases = self.stage_results.get('generation', [])
        api_spec = self.stage_results.get('analysis', {})

        # Create state for RL
        state = self.engine.rl_optimizer.create_state(test_cases, api_spec)

        # Optimize test selection and ordering
        optimized_tests = self.engine.rl_optimizer.optimize(state, test_cases)

        return optimized_tests

    async def _execute_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute test cases"""
        test_cases = self.stage_results.get('optimization',
                                            self.stage_results.get('generation', []))

        # Set authentication if provided
        if request.get('auth_token'):
            self.engine.test_executor.auth_token = request['auth_token']

        # Set SSL verification
        if not request.get('use_ssl', False):
            self.engine.test_executor.ssl_verify = False

        # Execute tests
        results = []
        for test in test_cases:
            try:
                result = await self.engine.test_executor.execute_test(
                    test,
                    request['endpoint_url']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                results.append({
                    'test': test,
                    'passed': False,
                    'error': str(e)
                })

        return results

    async def _process_feedback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution feedback"""
        execution_results = self.stage_results.get('execution', [])

        # Update RAG system
        try:
            await self.engine.feedback_loop.update_rag(execution_results)
        except Exception as e:
            logger.warning(f"RAG update failed: {e}")

        # Update RL model
        try:
            # Calculate reward and update RL
            for result in execution_results:
                state = self.engine.rl_optimizer.create_state(
                    [result['test']],
                    self.stage_results.get('analysis', {})
                )
                reward = 1.0 if result.get('passed') else -0.5
                self.engine.rl_optimizer.update_from_feedback(
                    state, None, reward, state, True
                )
        except Exception as e:
            logger.warning(f"RL update failed: {e}")

        return {'feedback_processed': True}

    async def _generate_report(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report"""
        execution_results = self.stage_results.get('execution', [])

        # Generate report using ReportWriterAgent
        try:
            report = await self.engine.report_generator.generate(
                execution_results,
                self.stage_results.get('analysis', {})
            )
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            report = {
                'summary': 'Report generation failed',
                'error': str(e)
            }

        return report

    def _initialize_pipeline(self, request: Dict[str, Any]):
        """Initialize pipeline state"""
        self.stage_results = {}
        self.pipeline_metrics = {}
        logger.info(f"Pipeline initialized for {request.get('language', 'unknown')} API")

    def _finalize_pipeline(self) -> Dict[str, Any]:
        """Finalize pipeline and prepare results"""
        return {
            'api_specification': self.stage_results.get('analysis'),
            'test_cases': self.stage_results.get('optimization',
                                                 self.stage_results.get('generation')),
            'execution_results': self.stage_results.get('execution'),
            'report': self.stage_results.get('reporting')
        }

    def _calculate_metrics(self, start_time: datetime):
        """Calculate pipeline metrics"""
        total_duration = (datetime.now() - start_time).total_seconds()

        self.pipeline_metrics = {
            'total_duration': total_duration,
            'stages_completed': len(self.stage_results),
            'total_stages': len(self.stages),
            'success_rate': len(self.stage_results) / len(self.stages),
            'tests_generated': len(self.stage_results.get('generation', [])),
            'tests_executed': len(self.stage_results.get('execution', [])),
        }

        # Add execution metrics if available
        execution_results = self.stage_results.get('execution', [])
        if execution_results:
            passed_tests = sum(1 for r in execution_results if r.get('passed'))
            self.pipeline_metrics['test_pass_rate'] = passed_tests / len(execution_results)