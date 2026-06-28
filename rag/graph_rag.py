"""
GraphRAG: Knowledge graph-enhanced retrieval for multi-hop questions.

The core problem with flat RAG on multi-hop questions:
  Q: "What city was the founder of the company that acquired DeepMind born in?"

The answer requires:
  Step 1: DeepMind was acquired by Google
  Step 2: Google was founded by Larry Page and Sergey Brin
  Step 3: Larry Page was born in East Lansing, Michigan

A vector search for the full question might retrieve documents about DeepMind's AI work,
not about Google's founding or Larry Page's birthplace — the embeddings aren't aligned.

GraphRAG solution:
  1. Extract entities and relationships from documents → build a knowledge graph
  2. At query time: extract entities from question → traverse graph to find paths
  3. Use traversal paths + retrieved context to answer multi-hop questions
"""

import os
import json
import re
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, deque
from openai import OpenAI
from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, EVENT
    doc_id: str
    context: str  # sentence where entity was found


@dataclass
class Relationship:
    source: str
    relation: str  # e.g., "founded", "born_in", "acquired", "located_in"
    target: str
    doc_id: str
    evidence: str  # sentence supporting this relationship


@dataclass
class GraphPath:
    nodes: List[str]
    edges: List[str]
    evidence: List[str]
    relevance_score: float = 0.0

    def to_text(self) -> str:
        parts = []
        for i, (node, edge) in enumerate(zip(self.nodes[:-1], self.edges)):
            parts.append(f"{node} --[{edge}]--> {self.nodes[i+1]}")
        return " → ".join(parts)


class GraphRAG:
    """
    Knowledge graph-augmented RAG:
    1. Extract entities + relationships from documents using LLM
    2. Build NetworkX directed graph
    3. At query time: extract query entities → BFS traversal → collect evidence
    4. Combine graph paths + retrieved context → generate answer
    """

    GEN_MODEL = "gpt-4o-mini"

    def __init__(self, openai_api_key: str, max_hop_depth: int = 3):
        self.client = OpenAI(api_key=openai_api_key)
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.doc_texts: Dict[str, str] = {}
        self.max_hop_depth = max_hop_depth

    def _extract_entities_and_relations(self, text: str, doc_id: str) -> Dict:
        """Use LLM to extract structured entities and relationships from text."""
        prompt = f"""Extract entities and relationships from this text. Return valid JSON only.

Text: {text}

Return format:
{{
  "entities": [
    {{"name": "entity name", "type": "PERSON|ORG|LOCATION|CONCEPT|EVENT", "context": "sentence where found"}}
  ],
  "relationships": [
    {{"source": "entity1", "relation": "relationship_verb", "target": "entity2", "evidence": "sentence"}}
  ]
}}

Rules:
- Entity names should be canonical (e.g., "Larry Page" not "Page" or "Larry")
- Relations should be concise verbs: founded, born_in, acquired, located_in, worked_at, created, etc.
- Only extract relationships that are clearly stated in the text
- Normalize entity names to be consistent"""

        try:
            response = self.client.chat.completions.create(
                model=self.GEN_MODEL,
                messages=[
                    {"role": "system", "content": "Extract entities and relationships as JSON. Return only valid JSON, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[GraphRAG] Extraction error for {doc_id}: {e}")
            return {"entities": [], "relationships": []}

    def add_documents(self, documents: List[Dict]) -> Dict:
        """
        Process documents: extract graph structure and add to knowledge graph.
        Returns stats about extracted entities and relationships.
        """
        total_entities = 0
        total_relations = 0

        for doc in documents:
            doc_id = doc["id"]
            text = doc.get("title", "") + ". " + doc["text"]
            self.doc_texts[doc_id] = text

            print(f"[GraphRAG] Extracting from {doc_id}...")
            extracted = self._extract_entities_and_relations(text, doc_id)

            # Add entities to graph
            for ent_data in extracted.get("entities", []):
                name = ent_data["name"].strip()
                if not name:
                    continue
                entity = Entity(
                    name=name,
                    entity_type=ent_data.get("type", "CONCEPT"),
                    doc_id=doc_id,
                    context=ent_data.get("context", ""),
                )
                self.entities[name] = entity
                if not self.graph.has_node(name):
                    self.graph.add_node(name, entity_type=entity.entity_type, doc_id=doc_id)
                total_entities += 1

            # Add relationships (edges)
            for rel_data in extracted.get("relationships", []):
                source = rel_data["source"].strip()
                target = rel_data["target"].strip()
                relation = rel_data.get("relation", "related_to").strip()
                evidence = rel_data.get("evidence", "")

                if not source or not target:
                    continue

                # Add nodes if they don't exist
                if not self.graph.has_node(source):
                    self.graph.add_node(source, doc_id=doc_id)
                if not self.graph.has_node(target):
                    self.graph.add_node(target, doc_id=doc_id)

                self.graph.add_edge(source, target, relation=relation, evidence=evidence, doc_id=doc_id)
                self.relationships.append(Relationship(
                    source=source, relation=relation, target=target,
                    doc_id=doc_id, evidence=evidence,
                ))
                total_relations += 1

        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "new_entities": total_entities,
            "new_relationships": total_relations,
        }
        print(f"[GraphRAG] Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        return stats

    def _extract_query_entities(self, question: str) -> List[str]:
        """Extract entities from the question to use as traversal starting points."""
        prompt = f"""What are the key named entities in this question? Return a JSON list of entity names only.

Question: {question}

Return format: {{"entities": ["entity1", "entity2"]}}

Only return entities that are likely to be nodes in a knowledge graph (people, organizations, places, specific concepts). Do not include generic terms."""

        try:
            response = self.client.chat.completions.create(
                model=self.GEN_MODEL,
                messages=[
                    {"role": "system", "content": "Extract named entities as JSON. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("entities", [])
        except Exception:
            return []

    def _find_matching_nodes(self, entity_name: str) -> List[str]:
        """Find graph nodes that match or partially match an entity name."""
        entity_lower = entity_name.lower()
        matches = []
        for node in self.graph.nodes():
            if entity_lower in node.lower() or node.lower() in entity_lower:
                matches.append(node)
        return matches

    def _bfs_traverse(self, start_nodes: List[str], max_depth: int) -> List[GraphPath]:
        """
        BFS traversal from starting nodes up to max_depth hops.
        Collects all paths and their evidence.
        """
        paths = []
        visited_paths: Set[str] = set()

        queue = deque()
        for node in start_nodes:
            if self.graph.has_node(node):
                queue.append(([node], [], []))

        while queue:
            current_path, current_edges, current_evidence = queue.popleft()

            if len(current_path) > 1:
                path_key = " → ".join(current_path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(GraphPath(
                        nodes=list(current_path),
                        edges=list(current_edges),
                        evidence=list(current_evidence),
                    ))

            if len(current_path) >= max_depth + 1:
                continue

            current_node = current_path[-1]
            for neighbor in self.graph.successors(current_node):
                if neighbor not in current_path:
                    edge_data = list(self.graph.get_edge_data(current_node, neighbor).values())[0]
                    relation = edge_data.get("relation", "related_to")
                    evidence = edge_data.get("evidence", "")
                    new_path = current_path + [neighbor]
                    new_edges = current_edges + [relation]
                    new_evidence = current_evidence + [evidence]
                    queue.append((new_path, new_edges, new_evidence))

        return paths

    def retrieve(self, question: str, top_k_paths: int = 5) -> Tuple[List[GraphPath], List[str]]:
        """
        Multi-hop retrieval via graph traversal.
        Returns traversal paths and their supporting evidence.
        """
        # Step 1: Extract entities from question
        query_entities = self._extract_query_entities(question)
        print(f"[GraphRAG] Query entities: {query_entities}")

        # Step 2: Find matching nodes in graph
        start_nodes = []
        for entity in query_entities:
            matches = self._find_matching_nodes(entity)
            start_nodes.extend(matches)
        start_nodes = list(set(start_nodes))
        print(f"[GraphRAG] Graph start nodes: {start_nodes}")

        if not start_nodes:
            # Fallback: try all nodes that appear in the question
            words = question.lower().split()
            for node in self.graph.nodes():
                if any(word in node.lower() for word in words if len(word) > 3):
                    start_nodes.append(node)

        # Step 3: BFS traversal
        paths = self._bfs_traverse(start_nodes, self.max_hop_depth)

        # Step 4: Score paths by length and evidence richness
        for path in paths:
            evidence_score = sum(1 for e in path.evidence if e) / max(len(path.evidence), 1)
            length_score = len(path.nodes) / self.max_hop_depth  # prefer longer paths
            path.relevance_score = 0.6 * evidence_score + 0.4 * length_score

        paths.sort(key=lambda p: p.relevance_score, reverse=True)
        return paths[:top_k_paths], start_nodes

    def answer(self, question: str) -> Tuple[str, List[GraphPath], List[str], str]:
        """Full GraphRAG pipeline: traverse → collect evidence → generate answer."""
        paths, start_nodes = self.retrieve(question)

        if not paths:
            return (
                "Could not find relevant information in the knowledge graph.",
                [], start_nodes, ""
            )

        # Build context from graph paths
        graph_context_parts = []
        for path in paths:
            path_str = path.to_text()
            evidence_str = " | ".join(e for e in path.evidence if e)
            graph_context_parts.append(f"Path: {path_str}\nEvidence: {evidence_str}")

        graph_context = "\n\n".join(graph_context_parts)

        prompt = f"""Answer the question using the knowledge graph paths and evidence below.
Each path shows relationships: Entity A --[relationship]--> Entity B --[relationship]--> Entity C

Knowledge Graph Traversal:
{graph_context}

Question: {question}

Think step by step through the graph paths to arrive at the answer. Be concise."""

        response = self.client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise question-answering system that reasons over knowledge graph paths. Follow the graph paths step by step."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )

        answer = response.choices[0].message.content.strip()
        return answer, paths, start_nodes, graph_context

    def get_graph_stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_list": list(self.graph.nodes())[:20],
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "avg_degree": (
                sum(d for _, d in self.graph.degree()) / max(self.graph.number_of_nodes(), 1)
            ),
        }

    def visualize_subgraph(self, center_nodes: List[str], depth: int = 2) -> Dict:
        """Return subgraph data for visualization."""
        nodes_to_include = set(center_nodes)
        for node in center_nodes:
            if self.graph.has_node(node):
                for neighbor in nx.single_source_shortest_path_length(
                    self.graph, node, cutoff=depth
                ).keys():
                    nodes_to_include.add(neighbor)

        subgraph = self.graph.subgraph(nodes_to_include)
        return {
            "nodes": [
                {"id": n, "type": self.graph.nodes[n].get("entity_type", "CONCEPT")}
                for n in subgraph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, "relation": data.get("relation", "")}
                for u, v, data in subgraph.edges(data=True)
            ],
        }

    def clear(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relationships = []
        self.doc_texts = {}
