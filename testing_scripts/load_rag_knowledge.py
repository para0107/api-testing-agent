"""
Load QA-Backend_Data.json into RAG system
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_knowledge_base():
    """Load knowledge base from JSON file into RAG"""

    kb_path = Path("data/knowledge_base/QA-Backend_Data.json")

    if not kb_path.exists():
        logger.error(f"❌ Knowledge base not found: {kb_path}")
        logger.info("\nPlease ensure QA-Backend_Data.json exists in data/knowledge_base/")
        return False

    logger.info(f"📖 Loading knowledge base from {kb_path}")

    # Load JSON
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} items in knowledge base")

    # Initialize RAG components
    logger.info("Initializing RAG components...")
    em = EmbeddingManager()
    vs = VectorStore()

    # Process each item
    success_count = 0
    error_count = 0

    for idx, item in enumerate(data):
        try:
            # Determine item type and index
            if 'test_case' in item or 'test_type' in item:
                index_name = 'test_patterns'
            elif 'edge_case' in item:
                index_name = 'edge_cases'
            elif 'validation' in item or 'rule' in item:
                index_name = 'validation_rules'
            else:
                index_name = 'test_patterns'  # Default

            # Create searchable text from item
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str) and len(value) < 500:
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, dict):
                    text_parts.append(f"{key}: {json.dumps(value)}")

            search_text = " ".join(text_parts)

            # Generate embedding
            embedding = await em.generate_text_embedding(search_text)

            # Store in vector store
            await vs.add_item(
                index=index_name,
                item_id=f"kb_{idx}",
                embedding=embedding,
                metadata=item
            )

            success_count += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{len(data)} items...")

        except Exception as e:
            error_count += 1
            logger.error(f"  Error processing item {idx}: {e}")
            continue

    # Save vector store
    logger.info("Saving vector store...")
    vs.save()

    logger.info(f"\n✅ Successfully loaded {success_count} items")
    if error_count > 0:
        logger.warning(f"⚠️  {error_count} items failed")

    # Verify by searching
    logger.info("\n🔍 Verifying RAG retrieval...")
    test_query = "GET endpoint test"
    test_embedding = await em.generate_text_embedding(test_query)
    results = await vs.search('test_patterns', test_embedding, k=3)

    logger.info(f"Test search returned {len(results)} results")
    for i, (score, item) in enumerate(results[:3], 1):
        logger.info(f"  {i}. Score: {score:.3f}")

    return success_count > 0


if __name__ == "__main__":
    try:
        result = asyncio.run(load_knowledge_base())
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)