"""
Retrieval logic for RAG system
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sentence_transformers import CrossEncoder
import heapq

from config import rag_config

logger = logging.getLogger(__name__)


class Retriever:
    """Handles retrieval of relevant documents from vector store"""

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # Load reranking model if enabled
        self.reranker = None
        if rag_config.rerank:
            self.reranker = CrossEncoder(rag_config.rerank_model)
            logger.info(f"Loaded reranking model: {rag_config.rerank_model}")

    async def retrieve(self, query: Union[str, np.ndarray],
                       index_name: str, k: int = 10,
                       rerank: bool = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query: Query text or embedding
            index_name: Index to search
            k: Number of results
            rerank: Whether to rerank results

        Returns:
            List of retrieved documents
        """
        # Generate embedding if query is text
        if isinstance(query, str):
            query_embedding = await self.embedding_manager.embed_text(query)
        else:
            query_embedding = query

        # Search vector store
        ids, distances, metadata = self.vector_store.search(
            index_name, query_embedding, k * 2 if rerank else k
        )

        # Prepare results
        results = []
        for i, (id_, dist, meta) in enumerate(zip(ids, distances, metadata)):
            if id_ != -1:  # Valid result
                result = {
                    'id': id_,
                    'score': 1 / (1 + dist),  # Convert distance to similarity score
                    'metadata': meta,
                    'rank': i
                }
                results.append(result)

        # Rerank if enabled
        if (rerank or (rerank is None and rag_config.rerank)) and self.reranker and isinstance(query, str):
            results = self._rerank_results(query, results)

        return results[:k]

    async def retrieve_similar_tests(self, query_embedding: np.ndarray,
                                     k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar test cases"""
        results = await self.retrieve(
            query_embedding,
            'test_patterns',
            k
        )

        # Enhance with test-specific information
        for result in results:
            if 'test_code' in result.get('metadata', {}):
                result['test_type'] = self._classify_test_type(result['metadata']['test_code'])

        return results

    async def retrieve_edge_cases(self, query_embedding: np.ndarray,
                                  k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant edge cases"""
        results = await self.retrieve(
            query_embedding,
            'edge_cases',
            k
        )

        # Filter by relevance threshold
        threshold = rag_config.similarity_threshold
        filtered = [r for r in results if r['score'] >= threshold]

        return filtered

    async def retrieve_validation_patterns(self, query_embedding: np.ndarray,
                                           k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve validation patterns"""
        return await self.retrieve(
            query_embedding,
            'validation_rules',
            k
        )

    async def hybrid_search(self, query: str, indices: List[str] = None,
                            k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across multiple indices

        Args:
            query: Query text
            indices: Indices to search (None for all)
            k: Number of results per index

        Returns:
            Combined and ranked results
        """
        # Generate embedding
        query_embedding = await self.embedding_manager.embed_text(query)

        # Search across indices
        all_results = self.vector_store.search_multiple_indices(
            query_embedding, indices, k
        )

        # Combine and rank results
        combined_results = []
        for index_name, index_results in all_results.items():
            for i, (id_, dist, meta) in enumerate(zip(
                    index_results['ids'],
                    index_results['distances'],
                    index_results['metadata']
            )):
                if id_ != -1:
                    result = {
                        'id': id_,
                        'index': index_name,
                        'score': 1 / (1 + dist),
                        'metadata': meta,
                        'rank': i
                    }
                    combined_results.append(result)

        # Sort by score
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        # Rerank if enabled
        if rag_config.rerank and self.reranker:
            combined_results = self._rerank_results(query, combined_results)

        # Apply MMR for diversity
        if len(combined_results) > k:
            combined_results = self._apply_mmr(query_embedding, combined_results, k)

        return combined_results[:k]

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder"""
        if not results:
            return results

        # Prepare pairs for reranking
        pairs = []
        for result in results:
            # Extract text from metadata
            text = result['metadata'].get('text', '')
            if not text and 'content' in result['metadata']:
                text = result['metadata']['content']
            if not text and 'code' in result['metadata']:
                text = result['metadata']['code']

            pairs.append([query, text])

        # Get reranking scores
        if pairs:
            scores = self.reranker.predict(pairs)

            # Update scores
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
                result['final_score'] = (result['score'] + float(score)) / 2

            # Sort by final score
            results.sort(key=lambda x: x.get('final_score', x['score']), reverse=True)

        return results

    def _apply_mmr(self, query_embedding: np.ndarray, results: List[Dict[str, Any]],
                   k: int, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance for diversity

        Args:
            query_embedding: Query embedding
            results: Initial results
            k: Number of results to return
            lambda_param: Balance between relevance and diversity

        Returns:
            Diverse results
        """
        if not results:
            return results

        selected = []
        candidates = results.copy()

        # Select first result (highest relevance)
        selected.append(candidates.pop(0))

        # Select remaining results
        while len(selected) < k and candidates:
            mmr_scores = []

            for candidate in candidates:
                # Relevance to query
                relevance = candidate['score']

                # Maximum similarity to selected documents
                max_sim = 0
                for selected_doc in selected:
                    # Calculate similarity (simplified - would need embeddings)
                    sim = self._calculate_similarity(candidate, selected_doc)
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)

            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(candidates.pop(best_idx))

        return selected

    def _calculate_similarity(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
        """Calculate similarity between two documents"""
        # Simplified similarity based on metadata
        # In practice, would use embeddings

        # Check if same index
        if doc1.get('index') == doc2.get('index'):
            similarity = 0.3
        else:
            similarity = 0.0

        # Check metadata overlap
        meta1 = doc1.get('metadata', {})
        meta2 = doc2.get('metadata', {})

        # Compare test types if available
        if 'test_type' in meta1 and 'test_type' in meta2:
            if meta1['test_type'] == meta2['test_type']:
                similarity += 0.3

        return min(similarity, 1.0)

    def _classify_test_type(self, test_code: str) -> str:
        """Classify test type from code"""
        test_code_lower = test_code.lower()

        if 'boundary' in test_code_lower or 'edge' in test_code_lower:
            return 'edge_case'
        elif 'null' in test_code_lower or 'empty' in test_code_lower:
            return 'null_check'
        elif 'valid' in test_code_lower and 'invalid' not in test_code_lower:
            return 'happy_path'
        elif 'error' in test_code_lower or 'exception' in test_code_lower:
            return 'error_handling'
        elif 'security' in test_code_lower or 'inject' in test_code_lower:
            return 'security'
        else:
            return 'general'