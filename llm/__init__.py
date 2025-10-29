"""
LLM integration module for test generation
"""
import asyncio

from .agents.data_generator import DataGeneratorAgent
from .agents.edge_case_agent import EdgeCaseAgent
from .agents.report_writer import ReportWriterAgent
from .agents.test_designer import TestDesignerAgent
from .llama_client import LlamaClient
from .agents import analyzer_agent, data_generator, test_designer, report_writer
from .prompts import prompt_builder, prompt_templates
from .prompts.prompt_builder import PromptBuilder
from .prompts.prompt_templates import PromptTemplates
from .response_parser import ResponseParser
from .agents.data_generator import DataGeneratorAgent
from .agents.edge_case_agent import EdgeCaseAgent
from .agents.report_writer import ReportWriterAgent
from .agents.test_designer import TestDesignerAgent
from .agents.analyzer_agent import AnalyzerAgent
import logging

logger = logging.getLogger(__name__)

class LlamaOrchestrator:
    """Main orchestrator for LLM-based test generation"""

    def __init__(self):
        self.client = None  # Will be initialized properly
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

    async def __aenter__(self):
        """Initialize client with context manager"""
        self.client = LlamaClient()
        await self.client.__aenter__()
        self.agents = {
            'analyzer': AnalyzerAgent(self.client),
            'test_designer': TestDesignerAgent(self.client),
            'edge_case': EdgeCaseAgent(self.client),
            'data_generator': DataGeneratorAgent(self.client),
            'report_writer': ReportWriterAgent(self.client)
        }
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup client session"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_test_suite(self, api_spec, context, config):
        """Generate complete test suite using multiple agents"""
        # Verify session is initialized
        if not self.client or not self.client.session:
            raise RuntimeError("LlamaOrchestrator must be used as async context manager")

        try:
            # Analyze API (with timeout)
            logger.info("Calling analyzer agent...")
            analysis = await asyncio.wait_for(
                self.agents['analyzer'].analyze(api_spec, context),
                timeout=30  # 30s timeout for analysis
            )

            # Design tests (with timeout)
            logger.info("Calling test_designer agent...")
            test_cases = await asyncio.wait_for(
                self.agents['test_designer'].design_tests(analysis, context, config),
                timeout=40  # 40s timeout for test design
            )

            # Generate edge cases (with timeout)
            logger.info("Calling edge_case agent...")
            edge_cases = await asyncio.wait_for(
                self.agents['edge_case'].generate_edge_cases(api_spec, analysis),
                timeout=30  # 30s timeout for edge cases
            )

            # Generate test data (with timeout)
            logger.info("Calling data_generator agent...")
            all_tests = test_cases + edge_cases
            test_data = await asyncio.wait_for(
                self.agents['data_generator'].generate_data(all_tests, api_spec),
                timeout=20  # 20s timeout for data generation
            )

            # Combine results
            return {
                'analysis': analysis,
                'test_cases': test_cases,
                'edge_cases': edge_cases,
                'test_data': test_data
            }

        except asyncio.TimeoutError as e:
            logger.error(f"LLM agent timed out: {e}")
            raise RuntimeError(f"LLM generation timed out. Check if LM Studio is running at http://127.0.0.1:1234")
        except Exception as e:
            logger.error(f"Error in test suite generation: {e}")
            raise


__all__ = [
    'LlamaOrchestrator',
    'LlamaClient',
    'AnalyzerAgent',
    'TestDesignerAgent',
    'EdgeCaseAgent',
    'DataGeneratorAgent',
    'ReportWriterAgent',
    'PromptBuilder',
    'PromptTemplates',
    'ResponseParser'
]