---
title: GraphRAG vs Flat RAG Benchmark
emoji: 🕸️
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Multi-hop QA benchmark showing where flat RAG fails
python_version: "3.10"
---

# 🕸️ GraphRAG vs Flat RAG

> A rigorous benchmark showing *exactly* where and why vector RAG fails on multi-hop questions, and how knowledge graph traversal fixes it.

## The Core Problem

Standard RAG tutorials skip a critical limitation: **vector similarity retrieval is single-hop by nature**. The embedding of a complex question like "What city was the founder of the company that acquired DeepMind born in?" rarely aligns with the specific documents needed to answer it.

The answer requires three reasoning steps:
1. DeepMind was acquired by **Google**
2. Google was co-founded by **Larry Page**
3. Larry Page was born in **East Lansing, Michigan**

A vector search for the full question might retrieve documents about DeepMind's AI research — not about Google's founding or Larry Page's birthplace.

## Benchmark Results (HotpotQA, 100 questions)

| Question Type | Flat RAG (EM) | GraphRAG (EM) | Delta |
|---|---|---|---|
| Single-hop | **71%** | 69% | Flat wins |
| **Multi-hop** | 34% | **61%** | **+27 pts for Graph** |

| Question Type | Flat RAG (F1) | GraphRAG (F1) | Delta |
|---|---|---|---|
| Single-hop | **74%** | 72% | Flat wins |
| **Multi-hop** | 41% | **67%** | **+26 pts for Graph** |

**Key finding**: GraphRAG matches flat RAG on single-hop and dominates by 27 percentage points on multi-hop questions.

## Architecture

### Flat RAG

```
Documents → Chunk (200 words, 50 overlap) → Embed (text-embedding-3-small) → FAISS flat index
Query → Embed → Top-K cosine search → Concatenate context → GPT-4o-mini → Answer
```

### GraphRAG

**Indexing** (once per corpus):
```
Documents → LLM extraction → Entities + Relationships → NetworkX directed graph
```

**Query**:
```
Question → Extract query entities → BFS traversal (max depth 3)
→ Collect evidence paths → GPT-4o-mini (reasoning over paths) → Answer
```

### Why BFS with depth=3?

- **Depth 1**: direct facts ("DeepMind was acquired by Google")
- **Depth 2**: one bridge hop ("Google founder Larry Page")
- **Depth 3**: two bridge hops ("Larry Page born in East Lansing")
- **Depth 4+**: exponential path growth with diminishing returns. Most HotpotQA multi-hop questions require ≤3 hops.

## Failure Modes: When GraphRAG Loses

1. **Poor entity extraction**: If the LLM misses an entity or misspells it, the traversal starting point is wrong
2. **Low-connectivity graphs**: Sparse documents yield few relationships, reducing traversal options
3. **Ambiguous entity names**: "Page" (the person) vs "page" (a web page) requires coreference resolution
4. **Long-range hops (>3)**: BFS capped at depth 3 misses very indirect reasoning chains

## Running Locally

```bash
git clone https://github.com/data-geek-astronomy/graphrag-vs-flat-rag
cd graphrag-vs-flat-rag
pip install -r requirements.txt
OPENAI_API_KEY=sk-... python app.py
```

## References

- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) (Microsoft GraphRAG paper)

## License

MIT
