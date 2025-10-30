"""
Test individual components
Run with: python scripts/test_components.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_rag():
    """Test RAG retrieval"""
    logger.info("=" * 60)
    logger.info("Testing RAG System")
    logger.info("=" * 60)

    try:
        from rag.embeddings import EmbeddingManager
        from rag.vector_store import VectorStore

        em = EmbeddingManager()
        vs = VectorStore()

        # Check if indices have data
        for index_name in ['test_patterns', 'edge_cases', 'validation_rules']:
            try:
                results = vs.indices.get(index_name, [])
                logger.info(f"  {index_name}: {len(results)} items")
            except:
                logger.info(f"  {index_name}: 0 items")

        # Generate test embedding
        test_query = "GET reservation endpoint by id"
        logger.info(f"\nSearching for: '{test_query}'")

        embedding = await em.generate_text_embedding(test_query)
        logger.info(f"Generated embedding (dim: {len(embedding)})")

        # Try to retrieve
        results = await vs.search('test_patterns', embedding, k=5)

        logger.info(f"\nRAG returned {len(results)} results:")
        for idx, (score, item) in enumerate(results[:3], 1):
            logger.info(f"  {idx}. Score: {score:.3f}")
            logger.info(f"     Data: {str(item)[:150]}")

        success = len(results) > 0
        logger.info(f"\n{'✅ PASS' if success else '❌ FAIL'}: RAG retrieval")
        return success

    except Exception as e:
        logger.error(f"❌ RAG test failed: {e}", exc_info=True)
        return False


async def test_executor():
    """Test executor"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Test Executor")
    logger.info("=" * 60)

    try:
        from test_execution.executor import TestExecutor

        async with TestExecutor() as executor:
            # Test with public API
            test_case = {
                'name': 'test_httpbin',
                'method': 'GET',
                'endpoint': '/get',
                'expected_status': 200,
                'test_data': {'param1': 'value1'}
            }

            logger.info(f"Testing executor with httpbin.org...")
            result = await executor.execute_test(test_case, 'https://httpbin.org')

            logger.info(f"  Status: {result.get('actual_status')}")
            logger.info(f"  Passed: {result.get('passed')}")
            logger.info(f"  Time: {result.get('execution_time'):.2f}s")

            success = result.get('passed', False)
            logger.info(f"\n{'✅ PASS' if success else '❌ FAIL'}: Test executor")
            return success

    except Exception as e:
        logger.error(f"❌ Executor test failed: {e}", exc_info=True)
        return False


async def test_embedding_generation():
    """Test embedding generation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Embedding Generation")
    logger.info("=" * 60)

    try:
        from rag.embeddings import EmbeddingManager

        em = EmbeddingManager()

        test_texts = [
            "GET /api/users endpoint",
            "POST /api/auth/login with credentials",
            "DELETE /api/items/{id} resource"
        ]

        for text in test_texts:
            embedding = await em.generate_text_embedding(text)
            logger.info(f"  '{text[:40]}...' -> dim: {len(embedding)}")

        logger.info(f"\n✅ PASS: Embedding generation")
        return True

    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}", exc_info=True)
        return False


async def check_knowledge_base():
    """Check if knowledge base file exists"""
    logger.info("\n" + "=" * 60)
    logger.info("Checking Knowledge Base File")
    logger.info("=" * 60)

    kb_path = Path("data/knowledge_base/QA-Backend_Data.json")

    if kb_path.exists():
        import json
        with open(kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"  ✅ Found: {kb_path}")
        logger.info(f"  Items: {len(data)}")
        logger.info(f"  First item keys: {list(data[0].keys()) if data else 'N/A'}")
        return True
    else:
        logger.error(f"  ❌ NOT FOUND: {kb_path}")
        logger.info("\n  Please create this file or copy it from your project!")
        return False


async def main():
    """Run all component tests"""
    logger.info("\n" + "🔍 API Testing Agent - Component Diagnostics")
    logger.info("=" * 60)

    results = {
        'Knowledge Base File': await check_knowledge_base(),
        'Embedding Generation': await test_embedding_generation(),
        'RAG Retrieval': await test_rag(),
        'Test Executor': await test_executor()
    }

    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL RESULTS")
    logger.info("=" * 60)

    all_passed = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {component:25s}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("✅ All components working!")
    else:
        logger.info("❌ Some components need fixing")

    return all_passed


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)