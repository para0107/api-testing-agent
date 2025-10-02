"""
FAISS vector store for similarity search
"""

import logging
import numpy as np
import faiss
import pickle
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

from config import rag_config, paths

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for embeddings"""

    def __init__(self):
        logger.info("Initializing Vector Store")

        self.dimension = rag_config.embedding_dimension
        self.indices = {}
        self.metadata_stores = {}
        self.index_configs = {}

        # Initialize indices for different types
        for index_name in rag_config.indices:
            self._create_index(index_name)

        # Load existing indices if available
        self._load_indices()

    def _create_index(self, index_name: str):
        """Create a new FAISS index"""
        config = rag_config.get_index_config(index_name)

        if rag_config.index_type == "IVF":
            # IVF index for better accuracy
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension,
                                       rag_config.nlist, faiss.METRIC_L2)
        elif rag_config.index_type == "HNSW":
            # HNSW for fast search
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # Simple flat index
            index = faiss.IndexFlatL2(self.dimension)

        # Add ID mapping
        index = faiss.IndexIDMap(index)

        self.indices[index_name] = index
        self.metadata_stores[index_name] = {}
        self.index_configs[index_name] = config

        logger.info(f"Created index: {index_name}")

    def add(self, index_name: str, embeddings: np.ndarray,
            metadata: List[Dict[str, Any]] = None, ids: List[int] = None):
        """
        Add embeddings to index

        Args:
            index_name: Name of the index
            embeddings: Numpy array of embeddings
            metadata: Optional metadata for each embedding
            ids: Optional IDs for embeddings
        """
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        # Ensure embeddings are 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.metadata_stores[index_name])
            ids = list(range(start_id, start_id + len(embeddings)))

        # Convert to numpy array
        ids_array = np.array(ids, dtype=np.int64)

        # Train index if needed (for IVF)
        if isinstance(index.index, faiss.IndexIVFFlat) and not index.index.is_trained:
            logger.info(f"Training index {index_name}")
            index.index.train(embeddings)

        # Add to index
        index.add_with_ids(embeddings, ids_array)

        # Store metadata
        if metadata:
            for id_, meta in zip(ids, metadata):
                self.metadata_stores[index_name][id_] = meta

        logger.info(f"Added {len(embeddings)} embeddings to {index_name}")

    def search(self, index_name: str, query_embedding: np.ndarray,
               k: int = 10) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for similar embeddings

        Args:
            index_name: Name of the index
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            Tuple of (ids, distances, metadata)
        """
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Set search parameters for IVF
        if isinstance(index.index, faiss.IndexIVFFlat):
            index.index.nprobe = rag_config.nprobe

        # Search
        distances, ids = index.search(query_embedding, k)

        # Get metadata
        metadata = []
        for id_ in ids[0]:
            if id_ != -1:  # Valid ID
                meta = self.metadata_stores[index_name].get(int(id_), {})
                metadata.append(meta)
            else:
                metadata.append({})

        return ids[0].tolist(), distances[0].tolist(), metadata

    def search_multiple_indices(self, query_embedding: np.ndarray,
                                indices: List[str] = None, k: int = 10) -> Dict[str, Any]:
        """Search across multiple indices"""
        if indices is None:
            indices = list(self.indices.keys())

        results = {}
        for index_name in indices:
            if index_name in self.indices:
                ids, distances, metadata = self.search(index_name, query_embedding, k)
                results[index_name] = {
                    'ids': ids,
                    'distances': distances,
                    'metadata': metadata
                }

        return results

    def update(self, index_name: str, id_: int, embedding: np.ndarray,
               metadata: Dict[str, Any] = None):
        """Update an embedding"""
        # Remove old embedding
        self.remove(index_name, [id_])

        # Add new embedding
        self.add(index_name, embedding.reshape(1, -1),
                 [metadata] if metadata else None, [id_])

    def remove(self, index_name: str, ids: List[int]):
        """Remove embeddings by ID"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        # Note: FAISS doesn't support direct removal
        # We need to rebuild the index without these IDs
        logger.warning(f"Removal not directly supported. Consider rebuilding index {index_name}")

    def save_index(self, index_name: str):
        """Save index to disk"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index_dir = paths.VECTOR_STORE_DIR / index_name
        index_dir.mkdir(exist_ok=True)

        # Save FAISS index
        index_file = index_dir / "index.faiss"
        faiss.write_index(self.indices[index_name], str(index_file))

        # Save metadata
        metadata_file = index_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata_stores[index_name], f)

        # Save config
        config_file = index_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.index_configs[index_name], f)

        logger.info(f"Saved index {index_name}")

    def load_index(self, index_name: str):
        """Load index from disk"""
        index_dir = paths.VECTOR_STORE_DIR / index_name

        if not index_dir.exists():
            logger.warning(f"Index directory {index_dir} not found")
            return False

        # Load FAISS index
        index_file = index_dir / "index.faiss"
        if index_file.exists():
            self.indices[index_name] = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = index_dir / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata_stores[index_name] = pickle.load(f)

        # Load config
        config_file = index_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.index_configs[index_name] = json.load(f)

        logger.info(f"Loaded index {index_name}")
        return True

    def _load_indices(self):
        """Load all existing indices"""
        if paths.VECTOR_STORE_DIR.exists():
            for index_dir in paths.VECTOR_STORE_DIR.iterdir():
                if index_dir.is_dir():
                    index_name = index_dir.name
                    if index_name not in self.indices:
                        self.load_index(index_name)

    def save_all(self):
        """Save all indices"""
        for index_name in self.indices:
            self.save_index(index_name)

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        return {
            'name': index_name,
            'total_embeddings': index.ntotal,
            'dimension': self.dimension,
            'index_type': type(index.index).__name__,
            'metadata_count': len(self.metadata_stores[index_name])
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all indices"""
        stats = {}
        for index_name in self.indices:
            stats[index_name] = self.get_index_stats(index_name)
        return stats

    def clear_index(self, index_name: str):
        """Clear an index"""
        if index_name in self.indices:
            self._create_index(index_name)
            logger.info(f"Cleared index {index_name}")

    def clear_all(self):
        """Clear all indices"""
        for index_name in list(self.indices.keys()):
            self.clear_index(index_name)