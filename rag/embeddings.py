"""
Embedding generation and management
"""

import logging
import numpy as np
from typing import Dict, List, Any, Union
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib
import pickle
from pathlib import Path

from config import rag_config, paths

rag_config = rag_config.RAGConfig()

logger = logging.getLogger(__name__)



class EmbeddingManager:
    """Manages embedding generation for different types of content"""

    def __init__(self):
        logger.info("Initializing Embedding Manager")

        # Load models
        self.text_model = SentenceTransformer(rag_config.text_embedding_model)
        self.code_model = self._load_code_model()

        # Cache for embeddings
        self.cache_dir = paths.VECTOR_STORE_DIR / "embedding_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}

        # Model dimensions
        self.text_dim = rag_config.embedding_dimension
        self.code_dim = rag_config.embedding_dimension

    def _load_code_model(self):
        """Load code embedding model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(rag_config.code_embedding_model)
            model = AutoModel.from_pretrained(rag_config.code_embedding_model)
            return {'tokenizer': tokenizer, 'model': model}
        except Exception as e:
            logger.warning(f"Failed to load code model: {e}. Using text model for code.")
            return None

    async def generate_embeddings(self, data: Union[str, Dict, List]) -> np.ndarray:
        """
        Generate embeddings for various data types

        Args:
            data: Input data (text, dict, or list)

        Returns:
            Embeddings as numpy array
        """
        if isinstance(data, str):
            return await self.embed_text(data)
        elif isinstance(data, dict):
            return await self.embed_structured(data)
        elif isinstance(data, list):
            return await self.embed_batch(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embeddings for text"""
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

            cached = self._load_from_cache(cache_key)
            if cached is not None:
                self.cache[cache_key] = cached
                return cached

        # Generate embedding
        embedding = self.text_model.encode(text, convert_to_numpy=True)

        # Normalize
        embedding = self._normalize(embedding)

        # Cache
        if use_cache:
            self.cache[cache_key] = embedding
            self._save_to_cache(cache_key, embedding)

        return embedding

    async def embed_code(self, code: str, language: str = None) -> np.ndarray:
        """Generate embeddings for code"""
        if self.code_model:
            return await self._embed_with_codebert(code)
        else:
            # Fallback to text embedding with code preprocessing
            processed_code = self._preprocess_code(code, language)
            return await self.embed_text(processed_code)

    async def _embed_with_codebert(self, code: str) -> np.ndarray:
        """Generate embeddings using CodeBERT"""
        tokenizer = self.code_model['tokenizer']
        model = self.code_model['model']

        # Tokenize
        inputs = tokenizer(code, return_tensors="pt", max_length=512,
                           truncation=True, padding=True)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use pooled output or mean of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        # Normalize
        embeddings = self._normalize(embeddings.squeeze())

        return embeddings

    async def embed_structured(self, data: Dict[str, Any]) -> np.ndarray:
        """Generate embeddings for structured data (API specs, etc.)"""
        # Convert structured data to text representation
        text_parts = []

        # Add endpoint information
        if 'path' in data:
            text_parts.append(f"Path: {data['path']}")
        if 'method' in data:
            text_parts.append(f"Method: {data['method']}")

        # Add parameters
        if 'parameters' in data:
            for param in data['parameters']:
                param_text = f"Parameter {param.get('name', '')}: {param.get('type', '')} " \
                             f"({'required' if param.get('required') else 'optional'})"
                text_parts.append(param_text)

        # Add other fields
        for key, value in data.items():
            if key not in ['path', 'method', 'parameters']:
                text_parts.append(f"{key}: {str(value)}")

        # Combine and embed
        combined_text = " | ".join(text_parts)
        return await self.embed_text(combined_text)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = []

        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.text_model.encode(batch, convert_to_numpy=True)

            # Normalize each embedding
            for j in range(len(batch_embeddings)):
                batch_embeddings[j] = self._normalize(batch_embeddings[j])

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def combine_embeddings(self, embeddings: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """
        Combine multiple embeddings with optional weighting

        Args:
            embeddings: List of embedding arrays
            weights: Optional weights for each embedding

        Returns:
            Combined embedding
        """
        if not embeddings:
            raise ValueError("No embeddings to combine")

        if weights is None:
            weights = [1.0] * len(embeddings)

        if len(weights) != len(embeddings):
            raise ValueError("Number of weights must match number of embeddings")

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted average
        combined = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, weights):
            combined += emb * weight

        # Normalize result
        return self._normalize(combined)

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _preprocess_code(self, code: str, language: str = None) -> str:
        """Preprocess code for embedding"""
        # Remove comments based on language
        if language == 'python':
            code = self._remove_python_comments(code)
        elif language == 'csharp' or language == 'java':
            code = self._remove_c_style_comments(code)

        # Normalize whitespace
        import re
        code = re.sub(r'\s+', ' ', code)

        return code.strip()

    def _remove_python_comments(self, code: str) -> str:
        """Remove Python comments"""
        import re
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        return code

    def _remove_c_style_comments(self, code: str) -> str:
        """Remove C-style comments"""
        import re
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str):
        """Load embedding from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Embedding cache cleared")