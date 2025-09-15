"""
Test generation pipeline orchestrator
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from config import settings
from core.engine import CoreEngine
from utils.validators import validate_api_spec, validate_test_case

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
        """
        Run the complete test generation pipeline

        Args:
            request: Test generation request

        Returns:
            Pipeline results including tests and reports
        """
        logger.info("Starting test generation pipeline")
        start_time = datetime.now()

        try:
            # Initialize pipeline
            self._initialize_pipeline(request)

            # Execute stages
            for stage in self.stages:
                try:
                    await self._execute_stage(stage, request)
                except Exception as e:
                    if stage.required:
                        raise
                    logger.warning(f"Optional stage {stage.name} failed: {str(e)}")

            # Finalize pipeline
            results = self._finalize_pipeline()

            # Calculate metrics
            self._calculate_metrics(start_time)

            return {
                'status': 'success',
                'results': results,
                'metrics': self.pipeline_metrics,
                'stages_completed': list(self.stage_results.keys())
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'stages_completed': list(self.stage_results.keys()),
                'metrics': self.pipeline_metrics
            }

    async def _execute_stage(self, stage: PipelineStage, request: Dict[str, Any]):
        """Execute a single pipeline stage"""
        logger.info(f"Executing stage: {stage.name}")
        stage_start = datetime.now()

        try:
            # Execute stage function with timeout
            result = await asyncio.wait_for(
                stage.function(request),
                timeout=stage.timeout
            )

            # Store result
            self.stage_results[stage.name] = result

            # Log completion
            duration = (datetime.now() - stage_start).total_seconds()
            logger.info(f"Stage {stage.name} completed in {duration:.2f}s")

        except asyncio.TimeoutError:
            raise Exception(f"Stage {stage.name} timed out after {stage.timeout}s")
        except Exception as e:
            logger.error(f"Stage {stage.name} failed: {str(e)}")
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
        return await self.engine._analyze_api(request)

    async def _analyze_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API specification"""
        api_spec = self.stage_results.get('parsing', {})

        # Validate API specification
        if not validate_api_spec(api_spec):
            raise ValueError("Invalid API specification")

        return api_spec

    async def _retrieve_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve context from RAG"""
        api_spec = self.stage_results.get('analysis', {})
        return await self.engine._retrieve_context(api_spec)

    async def _generate_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases"""
        api_spec = self.stage_results.get('analysis', {})
        context = self.stage_results.get('retrieval', {})

        test_cases = await self.engine._generate_tests(api_spec, context)

        # Validate generated tests
        valid_tests = []
        for test in test_cases:
            if validate_test_case(test):
                valid_tests.append(test)
            else:
                logger.warning(f"Invalid test case generated: {test.get('name', 'unknown')}")

        return valid_tests

    async def _optimize_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize test cases with RL"""
        test_cases = self.stage_results.get('generation', [])
        api_spec = self.stage_results.get('analysis', {})

        return await self.engine._optimize_tests(test_cases, api_spec)

    async def _execute_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute test cases"""
        test_cases = self.stage_results.get('optimization',
                                            self.stage_results.get('generation', []))

        return await self.engine._execute_tests(test_cases, request['endpoint_url'])

    async def _process_feedback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution feedback"""
        execution_results = self.stage_results.get('execution', [])

        await self.engine._process_feedback(execution_results)

        return {'feedback_processed': True}

    async def _generate_report(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report"""
        execution_results = self.stage_results.get('execution', [])

        return await self.engine._generate_report(execution_results)

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


import asyncio
import os