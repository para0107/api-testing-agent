"""
LLM integration module for test generation
"""
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

class LlamaOrchestrator:
    """Main orchestrator for LLM-based test generation"""

    def __init__(self):
        self.client = LlamaClient()
        self.agents = {
            'analyzer': AnalyzerAgent(self.client),
            'test_designer': TestDesignerAgent(self.client),
            'edge_case': EdgeCaseAgent(self.client),
            'data_generator': DataGeneratorAgent(self.client),
            'report_writer': ReportWriterAgent(self.client)
        }
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

    async def generate_test_suite(self, api_spec, context, config):
        """Generate complete test suite using multiple agents"""
        # Analyze API
        analysis = await self.agents['analyzer'].analyze(api_spec, context)

        # Design tests
        test_cases = await self.agents['test_designer'].design_tests(
            analysis, context, config
        )

        # Generate edge cases
        edge_cases = await self.agents['edge_case'].generate_edge_cases(
            api_spec, analysis
        )

        # Generate test data
        all_tests = test_cases + edge_cases
        test_data = await self.agents['data_generator'].generate_data(
            all_tests, api_spec
        )

        # Combine results
        return {
            'analysis': analysis,
            'test_cases': test_cases,
            'edge_cases': edge_cases,
            'test_data': test_data
        }


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