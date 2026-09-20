import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure backend/packages is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
packages_dir = backend_dir / "packages"
if str(packages_dir) not in sys.path:
    sys.path.insert(0, str(packages_dir))

import chromadb

logger = logging.getLogger("documind.services.chroma")


class ChromaService:
    """
    Persistent ChromaDB vector store service for user-uploaded document chunks.
    Ensures strict tenant isolation by user_id and cosine similarity retrieval.
    """

    def __init__(self):
        raw_path = os.environ.get("CHROMA_PATH", "./storage/chroma")
        if os.path.isabs(raw_path):
            self.chroma_dir = Path(raw_path)
        else:
            self.chroma_dir = (backend_dir / raw_path).resolve()

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = "documind_documents"

        logger.info("Initializing ChromaDB persistent store at: %s", self.chroma_dir)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.info("ChromaDB collection '%s' ready. Vector count: %d", self.collection_name, self.collection.count())

    def add_chunks(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        file_type: str,
        chunks: List[Dict[str, Any]],
        embeddings: Any,
    ) -> int:
        """
        Add document chunks and their BGE embeddings to ChromaDB.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, c in enumerate(chunks):
            chunk_id = c.get("chunk_id", f"chunk_{i}")
            composite_id = f"{document_id}_{chunk_id}"
            ids.append(composite_id)
            documents.append(c.get("text", ""))

            page_num = c.get("page", 1)
            try:
                page_num = int(page_num) if page_num is not None else 1
            except (ValueError, TypeError):
                page_num = 1

            metadatas.append({
                "document_id": str(document_id),
                "user_id": str(user_id),
                "filename": str(filename),
                "chunk_id": str(chunk_id),
                "page": page_num,
                "file_type": str(file_type),
            })

        # Convert numpy embeddings to list of floats if needed
        if hasattr(embeddings, "tolist"):
            emb_list = embeddings.tolist()
        else:
            emb_list = [list(e) for e in embeddings]

        self.collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Added %d chunks for document %s (user %s) to ChromaDB",
            len(ids),
            document_id,
            user_id,
        )
        return len(ids)

    def query_vectors(
        self,
        query_embedding: Any,
        user_id: str,
        n_results: int = 6,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for the most similar chunks, strictly filtered by user_id
        and optionally filtered by document_id.
        """
        if self.collection.count() == 0:
            return []

        # Convert query embedding to list of floats
        if hasattr(query_embedding, "tolist"):
            q_emb = query_embedding.tolist()
        else:
            q_emb = list(query_embedding)

        # Enforce multi-tenant user isolation
        if document_id:
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id},
                ]
            }
        else:
            where_filter = {"user_id": user_id}

        try:
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query error: %s", e)
            return []

        hits = []
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return hits

        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            dist = float(distances[i]) if distances else 0.0
            # For cosine distance, similarity = 1.0 - distance
            score = round(max(0.0, 1.0 - dist), 4)
            hits.append({
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": score,
            })

        return hits

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """
        Remove all vectors belonging to a specific document and user.
        """
        try:
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id},
                ]
            }
            self.collection.delete(where=where_filter)
            logger.info("Deleted vectors for document %s (user %s) from ChromaDB", document_id, user_id)
            return True
        except Exception as e:
            logger.warning("Error deleting vectors for document %s: %s", document_id, e)
            return False

    def count_user_vectors(self, user_id: str) -> int:
        """
        Count total vectors for a specific user.
        """
        try:
            res = self.collection.get(where={"user_id": user_id}, include=[])
            return len(res.get("ids", []))
        except Exception:
            return 0


chroma_service = ChromaService()
