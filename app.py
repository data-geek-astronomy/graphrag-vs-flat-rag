"""
GraphRAG vs Flat RAG — Interactive Comparison
==============================================
Side-by-side benchmark showing exactly where and why flat vector retrieval
fails on multi-hop questions, and how graph traversal fixes it.

Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rag import FlatRAG, GraphRAG, PRECOMPUTED_BENCHMARK

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# Sample corpus — multi-hop reasoning requires bridging these documents
SAMPLE_CORPUS = [
    {
        "id": "deepmind",
        "title": "DeepMind",
        "text": "DeepMind is a British AI research laboratory founded in London in 2010 by Demis Hassabis, Shane Legg, and Mustafa Suleyman. The company was acquired by Google in 2014 for approximately 500 million dollars. DeepMind is known for creating AlphaGo, which defeated world champion Go players, and AlphaFold, which solved the protein folding problem.",
    },
    {
        "id": "google_founding",
        "title": "Google Founding",
        "text": "Google was founded in September 1998 by Larry Page and Sergey Brin while they were PhD students at Stanford University. The company is headquartered in Mountain View, California, also known as the Googleplex. Larry Page served as CEO from 1998 to 2001, and again from 2011 to 2015. Sundar Pichai became CEO of Google in 2015.",
    },
    {
        "id": "larry_page",
        "title": "Larry Page",
        "text": "Lawrence Edward Page, known as Larry Page, was born on March 26, 1973, in East Lansing, Michigan. His father, Carl Victor Page Sr., was a professor of computer science at Michigan State University. Larry Page co-invented PageRank, the algorithm that forms the basis of the Google search engine.",
    },
    {
        "id": "stanford",
        "title": "Stanford University",
        "text": "Stanford University is a private research university located in Stanford, California. It was founded in 1885 by Leland Stanford and Jane Stanford. Many technology companies were founded by Stanford alumni or faculty, earning the region the name Silicon Valley. Google, Netflix, and Hewlett-Packard are among the notable Stanford spinoffs.",
    },
    {
        "id": "alphago",
        "title": "AlphaGo",
        "text": "AlphaGo is a computer program developed by DeepMind to play the board game Go. In March 2016, AlphaGo defeated Lee Sedol, a 9-dan professional Go player, by 4-1 in a five-game match. This was considered a landmark achievement in artificial intelligence. AlphaGo uses a combination of deep neural networks and Monte Carlo tree search.",
    },
    {
        "id": "protein_folding",
        "title": "Protein Folding and AlphaFold",
        "text": "Protein folding is the physical process by which a protein chain acquires its functional three-dimensional structure. AlphaFold, developed by DeepMind, represented a major breakthrough in predicting protein structures from amino acid sequences. AlphaFold2, released in 2020, achieved accuracy comparable to experimental methods and was described as solving one of biology's grand challenges.",
    },
]

# Pre-built example questions with gold answers
EXAMPLE_QUESTIONS = [
    {
        "question": "In what city was the person born who co-founded the company that acquired DeepMind?",
        "gold_answer": "East Lansing",
        "type": "multi_hop",
        "reasoning_chain": "DeepMind → acquired by → Google → co-founded by → Larry Page → born in → East Lansing",
    },
    {
        "question": "Who founded DeepMind?",
        "gold_answer": "Demis Hassabis",
        "type": "single_hop",
        "reasoning_chain": "Direct lookup — single document",
    },
    {
        "question": "What university did the founders of the company that acquired DeepMind attend?",
        "gold_answer": "Stanford University",
        "type": "multi_hop",
        "reasoning_chain": "DeepMind → acquired by → Google → founded by → Larry Page, Sergey Brin → attended → Stanford",
    },
    {
        "question": "What algorithm was invented by the co-founder of the company that bought DeepMind?",
        "gold_answer": "PageRank",
        "type": "multi_hop",
        "reasoning_chain": "DeepMind → acquired by → Google → co-founded by → Larry Page → invented → PageRank",
    },
]

CSS = """
body, .gradio-container { background: #0a0d14 !important; }
.answer-card {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px; padding: 18px; margin: 8px 0;
}
.win-graph { border-color: #22c55e !important; }
.win-flat { border-color: #6366f1 !important; }
footer { display: none !important; }
"""

# ──────────────────────────────────────────────────────────
# State (global engines, initialized lazily)
# ──────────────────────────────────────────────────────────
flat_rag: FlatRAG = None
graph_rag: GraphRAG = None
corpus_loaded: bool = False


def initialize_engines(api_key: str):
    global flat_rag, graph_rag, corpus_loaded
    if not api_key:
        return "❌ Please enter your OpenAI API key"
    try:
        flat_rag = FlatRAG(openai_api_key=api_key)
        graph_rag = GraphRAG(openai_api_key=api_key)

        flat_rag.add_documents(SAMPLE_CORPUS)
        stats = graph_rag.add_documents(SAMPLE_CORPUS)

        corpus_loaded = True
        return (
            f"✅ **Corpus loaded!**\n\n"
            f"**Flat RAG**: {len(SAMPLE_CORPUS)} documents indexed in FAISS\n\n"
            f"**GraphRAG**: {stats['total_nodes']} entities, {stats['total_edges']} relationships extracted"
        )
    except Exception as e:
        return f"❌ Error initializing: {e}"


def run_comparison(question: str, api_key: str):
    global flat_rag, graph_rag, corpus_loaded

    if not api_key:
        return (
            "<div class='answer-card'>❌ Enter your OpenAI API key and load the corpus first.</div>",
            "<div class='answer-card'>❌ Enter your OpenAI API key and load the corpus first.</div>",
            "", "", None
        )

    if not corpus_loaded or flat_rag is None:
        # Auto-initialize
        msg = initialize_engines(api_key)
        if "Error" in msg:
            return (
                f"<div class='answer-card'>❌ {msg}</div>",
                "<div class='answer-card'></div>", "", "", None
            )

    if not question.strip():
        return (
            "<div class='answer-card'>Please enter a question.</div>",
            "<div class='answer-card'></div>", "", "", None
        )

    # Run both systems
    flat_answer, flat_chunks, flat_context = flat_rag.answer(question, top_k=3)
    graph_answer, graph_paths, start_nodes, graph_context = graph_rag.answer(question)

    # Format flat RAG result
    flat_docs_html = "".join(
        f"<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin:6px 0;border-left:3px solid #6366f1'>"
        f"<div style='color:#6366f1;font-size:0.78em;font-weight:600'>Doc: {c.doc_id} | Score: {c.score:.3f}</div>"
        f"<div style='color:#94a3b8;font-size:0.83em;margin-top:4px'>{c.text[:200]}...</div>"
        f"</div>"
        for c in flat_chunks
    )
    flat_html = f"""
    <div class='answer-card'>
        <h4 style='color:#6366f1;margin:0 0 10px'>📚 Flat RAG Answer</h4>
        <div style='color:#e2e8f0;font-size:1em;padding:12px;background:rgba(99,102,241,0.1);border-radius:8px;margin-bottom:12px'>
            {flat_answer}
        </div>
        <div style='color:#64748b;font-size:0.8em;margin-bottom:8px'>Retrieved chunks ({len(flat_chunks)}):</div>
        {flat_docs_html}
    </div>
    """

    # Format GraphRAG result
    paths_html = ""
    for path in graph_paths[:3]:
        paths_html += (
            f"<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin:6px 0;border-left:3px solid #22c55e'>"
            f"<div style='color:#22c55e;font-size:0.78em;font-weight:600'>Graph Path ({len(path.nodes)} hops)</div>"
            f"<div style='color:#94a3b8;font-size:0.83em;margin-top:4px;font-family:monospace'>{path.to_text()}</div>"
            f"</div>"
        )
    graph_html = f"""
    <div class='answer-card win-graph'>
        <h4 style='color:#22c55e;margin:0 0 10px'>🕸️ GraphRAG Answer</h4>
        <div style='color:#e2e8f0;font-size:1em;padding:12px;background:rgba(34,197,94,0.08);border-radius:8px;margin-bottom:12px'>
            {graph_answer}
        </div>
        <div style='color:#64748b;font-size:0.8em;margin-bottom:8px'>
            Start nodes: {', '.join(start_nodes[:5])} | Paths found: {len(graph_paths)}
        </div>
        {paths_html}
    </div>
    """

    # Graph visualization
    graph_data = graph_rag.visualize_subgraph(start_nodes[:3], depth=2)
    fig = make_graph_viz(graph_data, question)

    return flat_html, graph_html, flat_context[:500], graph_context[:500], fig


def use_example(example_idx: int):
    ex = EXAMPLE_QUESTIONS[example_idx]
    return ex["question"]


def make_benchmark_chart():
    bench = PRECOMPUTED_BENCHMARK
    categories = ["Single-Hop (Exact Match)", "Multi-Hop (Exact Match)", "Single-Hop (F1)", "Multi-Hop (F1)"]
    flat_scores = [
        bench["single_hop"]["flat_rag_em"],
        bench["multi_hop"]["flat_rag_em"],
        bench["single_hop"]["flat_rag_f1"],
        bench["multi_hop"]["flat_rag_f1"],
    ]
    graph_scores = [
        bench["single_hop"]["graph_rag_em"],
        bench["multi_hop"]["graph_rag_em"],
        bench["single_hop"]["graph_rag_f1"],
        bench["multi_hop"]["graph_rag_f1"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Flat RAG", x=categories, y=[s * 100 for s in flat_scores],
        marker_color="#6366f1",
        text=[f"{s*100:.0f}%" for s in flat_scores], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="GraphRAG", x=categories, y=[s * 100 for s in graph_scores],
        marker_color="#22c55e",
        text=[f"{s*100:.0f}%" for s in graph_scores], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="HotpotQA Benchmark: Flat RAG vs GraphRAG (100 questions)",
        barmode="group",
        yaxis=dict(range=[0, 90], title="Score (%)"),
        height=400,
        margin=dict(t=60, b=30, l=50, r=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def make_graph_viz(graph_data: dict, question: str):
    """Simple network visualization using Plotly."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        return go.Figure().update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="No graph nodes found for this question",
        )

    import math
    n = len(nodes)
    positions = {
        node["id"]: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }

    edge_traces = []
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src in positions and tgt in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(color="#334155", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))

    type_colors = {
        "PERSON": "#6366f1", "ORG": "#22c55e", "LOCATION": "#f59e0b",
        "CONCEPT": "#a78bfa", "EVENT": "#ef4444",
    }

    node_x = [positions[n["id"]][0] for n in nodes if n["id"] in positions]
    node_y = [positions[n["id"]][1] for n in nodes if n["id"] in positions]
    node_colors = [type_colors.get(n.get("type", "CONCEPT"), "#94a3b8") for n in nodes if n["id"] in positions]
    node_text = [n["id"] for n in nodes if n["id"] in positions]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=14, color=node_colors, line=dict(color="#0a0d14", width=2)),
        text=node_text, textposition="top center",
        textfont=dict(color="#e2e8f0", size=10),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,13,20,0.95)",
        plot_bgcolor="rgba(10,13,20,0.95)",
        title=f"Knowledge Graph Subgraph",
        font=dict(color="#e2e8f0"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=360,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="violet"), title="GraphRAG vs Flat RAG") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:28px 0 18px'>
        <div style='font-size:2.8em'>🕸️</div>
        <h1 style='color:#e2e8f0;margin:10px 0 6px;font-size:1.9em;font-weight:700'>
            GraphRAG vs Flat RAG
        </h1>
        <p style='color:#64748b;max-width:680px;margin:0 auto;line-height:1.6'>
            Standard vector RAG achieves 71% EM on single-hop questions but drops to 34% on multi-hop.
            Graph-enhanced retrieval closes the gap to 61% — a 27-point improvement — by traversing
            entity relationships rather than matching surface similarity.
        </p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🔬 Live Comparison"):
            with gr.Row():
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    value=OPENAI_KEY,
                    scale=3,
                )
                load_btn = gr.Button("📚 Load Corpus", variant="secondary", scale=1)

            load_status = gr.Markdown()
            load_btn.click(fn=initialize_engines, inputs=api_key, outputs=load_status)

            gr.HTML("""
            <div style='background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.3);border-radius:10px;padding:16px;margin:8px 0'>
                <div style='color:#64748b;font-size:0.82em;margin-bottom:10px'>📌 Example questions (try these — multi-hop requires bridging multiple documents)</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
            """)

            with gr.Row():
                for i, ex in enumerate(EXAMPLE_QUESTIONS):
                    badge = "🔗 multi-hop" if ex["type"] == "multi_hop" else "✅ single-hop"
                    gr.HTML(f"""
                    <div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;border:1px solid rgba(99,102,241,0.2)'>
                        <div style='font-size:0.75em;color:#6366f1;margin-bottom:4px'>{badge}</div>
                        <div style='color:#e2e8f0;font-size:0.85em'>{ex['question']}</div>
                        <div style='font-size:0.72em;color:#475569;margin-top:4px;font-family:monospace'>{ex['reasoning_chain']}</div>
                    </div>
                    """)

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="In what city was the person born who co-founded the company that acquired DeepMind?",
                value=EXAMPLE_QUESTIONS[0]["question"],
                lines=2,
            )
            ask_btn = gr.Button("⚡ Compare Both Systems", variant="primary", size="lg")

            with gr.Row():
                flat_output = gr.HTML(label="Flat RAG")
                graph_output = gr.HTML(label="GraphRAG")

            graph_viz = gr.Plot(label="Knowledge Graph Traversal")

            with gr.Row():
                flat_ctx = gr.Textbox(label="Flat RAG Context (truncated)", lines=4, interactive=False)
                graph_ctx = gr.Textbox(label="Graph Context (paths + evidence)", lines=4, interactive=False)

            ask_btn.click(
                fn=run_comparison,
                inputs=[question_input, api_key],
                outputs=[flat_output, graph_output, flat_ctx, graph_ctx, graph_viz],
            )

        with gr.Tab("📊 HotpotQA Benchmark"):
            gr.Plot(value=make_benchmark_chart())

            gr.HTML("""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px'>
                <div style='background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px'>
                    <h3 style='color:#6366f1;margin:0 0 10px'>📚 Flat RAG — Where it wins</h3>
                    <p style='color:#94a3b8;font-size:0.88em;line-height:1.7;margin:0'>
                    <b style='color:#e2e8f0'>Single-hop questions</b>: 71% EM. When the answer exists in a single document,
                    vector similarity retrieval works well. The embedding captures semantic
                    meaning and finds the right passage. Simple and fast.
                    <br/><br/>
                    <b style='color:#e2e8f0'>Fails on</b>: bridge questions ("Who is X connected to Y through?"),
                    comparison questions ("Which of A and B was born earlier?"), and any
                    question requiring information from 2+ non-overlapping documents.
                    </p>
                </div>
                <div style='background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.3);border-radius:12px;padding:20px'>
                    <h3 style='color:#22c55e;margin:0 0 10px'>🕸️ GraphRAG — Where it wins</h3>
                    <p style='color:#94a3b8;font-size:0.88em;line-height:1.7;margin:0'>
                    <b style='color:#e2e8f0'>Multi-hop questions</b>: 61% EM (vs 34% for flat). By building a knowledge graph
                    from extracted entities and relationships, traversal can follow
                    reasoning chains: DeepMind → acquired by → Google → founded by → Larry Page → born in → East Lansing.
                    <br/><br/>
                    <b style='color:#e2e8f0'>Cost</b>: LLM extraction at indexing time (1 extraction call per document).
                    Worth it for corpora where multi-hop retrieval is expected.
                    </p>
                </div>
            </div>
            """)

        with gr.Tab("💻 Architecture"):
            gr.Markdown("""
## How GraphRAG Works

### Indexing Phase (done once)

```python
# 1. For each document, extract entities and relationships via LLM
extracted = {
    "entities": [
        {"name": "DeepMind", "type": "ORG"},
        {"name": "Google", "type": "ORG"},
        {"name": "Larry Page", "type": "PERSON"},
    ],
    "relationships": [
        {"source": "Google", "relation": "acquired", "target": "DeepMind"},
        {"source": "Larry Page", "relation": "co_founded", "target": "Google"},
        {"source": "Larry Page", "relation": "born_in", "target": "East Lansing"},
    ]
}

# 2. Build NetworkX directed graph
graph.add_edge("Google", "DeepMind", relation="acquired")
graph.add_edge("Larry Page", "Google", relation="co_founded")
graph.add_edge("Larry Page", "East Lansing", relation="born_in")
```

### Query Phase

```python
# Q: "What city was the DeepMind acquirer's founder born in?"

# Step 1: Extract query entities
query_entities = ["DeepMind"]  # LLM extraction

# Step 2: BFS from start nodes
paths = bfs_traverse(start=["DeepMind"], max_depth=3)

# Paths found:
# DeepMind --[acquired_by]--> Google --[co_founded_by]--> Larry Page --[born_in]--> East Lansing

# Step 3: Use paths as context for LLM generation
answer = llm.generate(context=paths, question=question)
# → "East Lansing, Michigan"
```

### Why BFS not DFS?
- BFS explores all paths at depth 1 before depth 2, ensuring shortest paths surface first
- DFS can get stuck in deep irrelevant branches
- We cap at depth 3 to avoid exponential path explosion (avg. branching factor ~4)
            """)

demo.launch()
