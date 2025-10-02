# python
# file: scripts/index_knowledge.py
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, List, Optional


class KnowledgeIndexerRunner:
    """
    Soft-wires to rag components (chunking, embeddings, indexer, vector_store) if present.
    Avoids hard imports to keep the project stable while components evolve.
    """

    def __init__(self, out_dir: Optional[Path] = None) -> None:
        self.out_dir = Path(out_dir or "data/vectors")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Try import rag modules dynamically
        self.chunking = importlib.import_module("rag.chunking")
        self.embeddings = importlib.import_module("rag.embeddings")
        self.indexer = importlib.import_module("rag.indexer")
        self.vector_store = importlib.import_module("rag.vector_store")

    def _resolve(self, module, candidates: List[str]):
        for name in candidates:
            obj = getattr(module, name, None)
            if obj is not None:
                return obj
        raise AttributeError(f"None of the candidates found: {candidates}")

    def index_paths(self, paths: Iterable[Path], namespace: str = "default") -> Path:
        """
        Build or update the vector index for given files.
        Expects typical classes/functions in rag modules; uses best-effort matching.
        """
        paths = [Path(p) for p in paths if Path(p).is_file()]
        if not paths:
            raise FileNotFoundError("No valid files to index")

        # Resolve components by common names
        Chunker = self._resolve(self.chunking, ["Chunker", "CodeChunker", "TextChunker", "create_chunker"])
        Embedder = self._resolve(self.embeddings, ["EmbeddingModel", "Embeddings", "create_embeddings"])
        Indexer = self._resolve(self.indexer, ["Indexer", "KnowledgeIndexer", "create_indexer"])
        VectorStore = self._resolve(self.vector_store, ["VectorStore", "FAISSStore", "create_vector_store"])

        # Instantiate, allowing callable factories or classes
        chunker = Chunker() if callable(Chunker) and not isinstance(Chunker, type) else Chunker()  # type: ignore
        embedder = Embedder() if callable(Embedder) and not isinstance(Embedder, type) else Embedder()  # type: ignore
        store = VectorStore(self.out_dir) if callable(VectorStore) else VectorStore  # type: ignore
        index = Indexer(store, embedder) if callable(Indexer) else Indexer  # type: ignore

        # Read and chunk files
        documents: List[str] = [Path(p).read_text(encoding="utf-8") for p in paths]
        chunks: List[str] = []
        for doc in documents:
            if hasattr(chunker, "chunk"):
                chunks.extend(chunker.chunk(doc))
            elif callable(chunker):
                chunks.extend(chunker(doc))
            else:
                # Fallback: naive fixed-size chunking
                size = 1000
                chunks.extend([doc[i:i + size] for i in range(0, len(doc), size)])

        # Add to index
        if hasattr(index, "add_texts"):
            index.add_texts(chunks, namespace=namespace)  # type: ignore
        elif hasattr(index, "index"):
            index.index(chunks, namespace=namespace)  # type: ignore
        elif callable(index):
            index(chunks, namespace=namespace)  # type: ignore
        else:
            raise RuntimeError("Indexer has no usable method")

        # Persist if supported
        for obj in (store, index):
            for method in ("save", "persist", "flush", "commit"):
                if hasattr(obj, method):
                    try:
                        getattr(obj, method)()
                    except Exception:
                        pass

        return self.out_dir / f"{namespace}.index"
