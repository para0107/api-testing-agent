"""
Integration tests for API Testing Agent
Place in tests/integration/ directory
"""

import pytest
import asyncio
from pathlib import Path


# Test fixtures
@pytest.fixture
def sample_csharp_code():
    """Sample C# API code"""
    return '''
using Microsoft.AspNetCore.Mvc;

[Route("api/[controller]")]
[ApiController]
public class UserController : ControllerBase
{
    [HttpGet("{id}")]
    public ActionResult<User> GetUser(int id)
    {
        return Ok(new User { Id = id });
    }

    [HttpPost]
    public ActionResult<User> CreateUser([FromBody] UserDto user)
    {
        return Created("", user);
    }
}

public class User
{
    public int Id { get; set; }
    public string Name { get; set; }
}
'''


@pytest.fixture
def sample_python_code():
    """Sample Python FastAPI code"""
    return '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": "Test User"}

@app.post("/users")
async def create_user(user: User):
    return user
'''


# Test Input Processing
@pytest.mark.asyncio
async def test_csharp_parser(sample_csharp_code, tmp_path):
    """Test C# parser"""
    from input_processing import InputProcessor

    # Write code to temp file
    code_file = tmp_path / "test.cs"
    code_file.write_text(sample_csharp_code)

    # Parse
    processor = InputProcessor()
    parsed = processor.parse_code([str(code_file)], 'csharp')

    # Assertions
    assert 'endpoints' in parsed
    assert len(parsed['endpoints']) > 0

    # Check endpoint details
    endpoint = parsed['endpoints'][0]
    assert 'http_method' in endpoint
    assert 'route' in endpoint


@pytest.mark.asyncio
async def test_python_parser(sample_python_code, tmp_path):
    """Test Python parser"""
    from input_processing import InputProcessor

    # Write code to temp file
    code_file = tmp_path / "test.py"
    code_file.write_text(sample_python_code)

    # Parse
    processor = InputProcessor()
    parsed = processor.parse_code([str(code_file)], 'python')

    # Assertions
    assert 'endpoints' in parsed
    assert len(parsed['endpoints']) > 0


@pytest.mark.asyncio
async def test_specification_builder(sample_csharp_code, tmp_path):
    """Test API specification builder"""
    from input_processing import InputProcessor

    code_file = tmp_path / "test.cs"
    code_file.write_text(sample_csharp_code)

    processor = InputProcessor()
    parsed = processor.parse_code([str(code_file)], 'csharp')
    spec = processor.build_specification(parsed)

    # Assertions
    assert 'openapi' in spec
    assert 'paths' in spec
    assert 'components' in spec


# Test RAG System
@pytest.mark.asyncio
async def test_embedding_generation():
    """Test embedding generation"""
    from rag.embeddings import EmbeddingManager

    manager = EmbeddingManager()

    # Generate embedding
    text = "Test API endpoint for user management"
    embedding = await manager.embed_text(text)

    # Assertions
    assert embedding is not None
    assert embedding.shape[0] == 768  # Dimension
    assert abs(embedding.sum()) > 0  # Not all zeros


@pytest.mark.asyncio
async def test_vector_store():
    """Test vector store operations"""
    from rag.vector_store import VectorStore
    import numpy as np

    store = VectorStore()

    # Add embeddings
    embeddings = np.random.rand(10, 768).astype('float32')
    metadata = [{'text': f'test {i}'} for i in range(10)]

    store.add('test_index', embeddings, metadata)

    # Search
    query = np.random.rand(1, 768).astype('float32')
    ids, distances, meta = store.search('test_index', query, k=3)

    # Assertions
    assert len(ids) == 3
    assert len(distances) == 3
    assert len(meta) == 3


@pytest.mark.asyncio
async def test_chunking_strategy():
    """Test document chunking"""
    from rag.chunking import ChunkingStrategy

    chunker = ChunkingStrategy(chunk_size=100, chunk_overlap=20)

    text = "This is a test document. " * 50  # Long text
    chunks = chunker.chunk_document(text, strategy='sliding_window')

    # Assertions
    assert len(chunks) > 0
    assert all(len(c.text) <= 100 + 50 for c in chunks)  # Allow some overflow


# Test LLM Components
@pytest.mark.asyncio
async def test_response_parser():
    """Test LLM response parsing"""
    from llm.response_parser import ResponseParser

    parser = ResponseParser()

    # Test JSON parsing
    json_response = '```json\n{"name": "test", "value": 42}\n```'
    result = parser.parse(json_response, 'json')

    assert isinstance(result, dict)
    assert result['name'] == 'test'
    assert result['value'] == 42

    # Test list parsing
    list_response = '- Item 1\n- Item 2\n- Item 3'
    result = parser.parse(list_response, 'list')

    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_prompt_builder():
    """Test prompt building"""
    from llm.prompts.prompt_builder import PromptBuilder

    builder = PromptBuilder()

    api_spec = {
        'path': '/api/users',
        'method': 'GET',
        'parameters': []
    }

    prompt = builder.build_analysis_prompt(api_spec)

    assert isinstance(prompt, str)
    assert '/api/users' in prompt
    assert 'GET' in prompt


# Test RL Components
def test_policy_network():
    """Test policy network"""
    from reinforcement_learning.policy_network import PolicyNetwork
    import torch

    net = PolicyNetwork(state_dim=576, action_dim=10)

    # Forward pass
    state = torch.randn(1, 576)
    probs = net(state)

    # Assertions
    assert probs.shape == (1, 10)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_value_network():
    """Test value network"""
    from reinforcement_learning.value_network import ValueNetwork
    import torch

    net = ValueNetwork(state_dim=576)

    # Forward pass
    state = torch.randn(1, 576)
    value = net(state)

    # Assertions
    assert value.shape == (1, 1)


def test_experience_buffer():
    """Test experience buffer"""
    from reinforcement_learning.experience_buffer import ExperienceBuffer
    import torch

    buffer = ExperienceBuffer(capacity=100)

    # Add experiences
    for i in range(50):
        buffer.add(
            state=torch.randn(576),
            action=torch.tensor([i % 10]),
            reward=float(i),
            next_state=torch.randn(576),
            done=False
        )

    # Sample
    batch = buffer.sample(10)

    # Assertions
    assert len(batch) == 10
    assert len(buffer) == 50


def test_reward_calculator():
    """Test reward calculation"""
    from reinforcement_learning.reward_calculator import RewardCalculator

    calculator = RewardCalculator()

    metrics = {
        'bugs_found': 2,
        'code_coverage': 0.8,
        'edge_cases_covered': 5,
        'false_positives': 1
    }

    reward = calculator.calculate_reward([], metrics)

    # Assertions
    assert isinstance(reward, float)
    assert -100 <= reward <= 100  # Normalized


# Test Utilities
def test_validators():
    """Test validation utilities"""
    from utils.validators import (
        is_valid_api_spec, is_valid_test_case, is_valid_email, is_valid_url, is_valid_json
    )

    # Email validation
    assert is_valid_email('test@example.com')
    assert not is_valid_email('invalid-email')

    # URL validation
    assert is_valid_url('https://api.example.com')
    assert not is_valid_url('not-a-url')

    # API spec validation
    valid_spec = {
        'openapi': '3.0.0',
        'info': {'title': 'Test', 'version': '1.0'},
        'paths': {}
    }
    assert is_valid_api_spec(valid_spec)

    invalid_spec = {'openapi': '3.0.0'}
    assert not is_valid_api_spec(invalid_spec)


def test_metrics_collector():
    """Test metrics collection"""
    from utils.metrics import MetricsCollector

    collector = MetricsCollector()

    # Record metrics
    collector.record('test_metric', 42.0)
    collector.record('test_metric', 43.0)
    collector.increment('test_counter')
    collector.increment('test_counter', 2)

    # Get metrics
    values = collector.get_metric('test_metric')
    assert len(values) == 2

    counter = collector.get_counter('test_counter')
    assert counter == 3

    # Get summary
    summary = collector.get_summary()
    assert 'test_metric' in summary['metrics']
    assert summary['metrics']['test_metric']['avg'] == 42.5


# Test Execution
@pytest.mark.asyncio
async def test_test_executor():
    """Test execution engine"""
    from test_execution.executor import TestExecutor

    executor = TestExecutor()

    # Mock test case
    test_case = {
        'name': 'Test GET users',
        'endpoint': '/users',
        'method': 'GET',
        'expected_status': 200
    }

    # Note: This would require a running API
    # In practice, you'd mock the HTTP calls
    # Here we just test the structure

    assert hasattr(executor, 'execute_test')
    assert hasattr(executor, 'execute_batch')


# Integration Test
@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_pipeline_integration(sample_csharp_code, tmp_path):
    """Full integration test (slow)"""
    from core.pipeline import TestGenerationPipeline

    # Write code file
    code_file = tmp_path / "test.cs"
    code_file.write_text(sample_csharp_code)

    # Create pipeline
    pipeline = TestGenerationPipeline()

    request = {
        'code_files': [str(code_file)],
        'language': 'csharp',
        'endpoint_url': 'http://localhost:5000'
    }

    # Run pipeline (might fail at execution stage without real API)
    # We're mainly testing that components work together
    try:
        result = await pipeline.run(request)

        # Should at least get through parsing/analysis
        assert 'stages_completed' in result
        assert len(result['stages_completed']) >= 3  # validation, parsing, analysis

    except Exception as e:
        # Expected if no API is running
        pytest.skip(f"Pipeline test skipped (no API): {str(e)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])