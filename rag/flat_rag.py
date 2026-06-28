"""
Flat RAG: standard vector similarity retrieval.
Embed documents → store in FAISS → embed query → find top-k nearest neighbors.

This is what most RAG tutorials teach. It works well for single-hop questions
("What is the capital of France?") but fails on multi-hop questions that require
bridging across documents ("Who was the president when the person who wrote
Hamlet's most famous soliloquy was born?").
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    source: str = "flat_rag"


class FlatRAG:
    """
    Standard dense retrieval:
    1. Chunk documents
    2. Embed with text-embedding-3-small
    3. Store in FAISS flat index (exact L2 search)
    4. At query time: embed question, retrieve top-k by cosine similarity
    5. Concatenate retrieved chunks → LLM context → answer
    """

    EMBED_DIM = 1536  # text-embedding-3-small output dimension
    EMBED_MODEL = "text-embedding-3-small"
    GEN_MODEL = "gpt-4o-mini"

    def __init__(self, openai_api_key: str, chunk_size: int = 200, chunk_overlap: int = 50):
        self.client = OpenAI(api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = faiss.IndexFlatIP(self.EMBED_DIM)  # Inner product (cosine after normalize)
        self.chunks: List[Dict] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI text-embedding-3-small."""
        if not texts:
            return np.array([])
        response = self.client.embeddings.create(model=self.EMBED_MODEL, input=texts)
        embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def _chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Simple word-boundary chunking with overlap."""
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
            })
            i += self.chunk_size - self.chunk_overlap
            chunk_idx += 1
        return chunks

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the index.
        Each document: {"id": str, "title": str, "text": str}
        """
        new_chunks = []
        for doc in documents:
            doc_chunks = self._chunk_text(doc["text"], doc["id"])
            for chunk in doc_chunks:
                chunk["title"] = doc.get("title", doc["id"])
            new_chunks.extend(doc_chunks)

        if not new_chunks:
            return

        texts = [c["text"] for c in new_chunks]
        embeddings = self._embed(texts)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        print(f"[FlatRAG] Indexed {len(new_chunks)} chunks from {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Embed query and retrieve top-k chunks by cosine similarity."""
        if self.index.ntotal == 0:
            return []

        query_emb = self._embed([query])
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                results.append(RetrievedChunk(
                    text=chunk["text"],
                    score=float(score),
                    doc_id=chunk["doc_id"],
                    source="flat_rag",
                ))
        return results

    def answer(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[RetrievedChunk], str]:
        """Retrieve context and generate an answer."""
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            [f"[Doc: {r.doc_id}] {r.text}" for r in retrieved]
        )

        prompt = f"""Answer the question using ONLY the provided context. If the answer cannot be determined from the context, say "I cannot determine this from the provided context."

Context:
{context}

Question: {question}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise question-answering system. Answer only from the provided context. Be concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )

        answer = response.choices[0].message.content.strip()
        return answer, retrieved, context

    def clear(self):
        self.index = faiss.IndexFlatIP(self.EMBED_DIM)
        self.chunks = []
