"""
GraphRAG vs Flat RAG — Professional Demo
Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import plotly.graph_objects as go
import os

from rag.flat_rag import FlatRAG
from rag.graph_rag import GraphRAG
from rag.evaluator import PRECOMPUTED_BENCHMARK

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #000 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif !important;
    color: #f5f5f7 !important;
}
.hero { padding: 64px 32px 48px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.07); }
.hero-badge { display: inline-block; background: rgba(48,209,88,0.12); color: #30d158; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; border: 1px solid rgba(48,209,88,0.2); margin-bottom: 22px; }
.hero-title { font-size: 48px; font-weight: 700; color: #f5f5f7; line-height: 1.06; letter-spacing: -0.025em; margin: 0 0 18px; }
.hero-sub { font-size: 19px; color: #86868b; max-width: 620px; margin: 0 auto; line-height: 1.55; }
.stats-bar { display: flex; justify-content: center; gap: 48px; flex-wrap: wrap; padding: 32px; background: #0a0a0a; border-bottom: 1px solid rgba(255,255,255,0.07); }
.stat { text-align: center; }
.stat-val { font-size: 30px; font-weight: 700; color: #30d158; letter-spacing: -0.02em; }
.stat-label { font-size: 12px; color: #6e6e73; margin-top: 3px; font-weight: 500; }
.section { padding: 36px 32px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sec-label { font-size: 12px; font-weight: 600; color: #6e6e73; letter-spacing: 0.09em; text-transform: uppercase; margin: 0 0 18px; }
.card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 22px 24px; margin-bottom: 10px; }
.card-title { font-size: 16px; font-weight: 600; color: #f5f5f7; margin: 0 0 8px; }
.card-body { font-size: 14px; color: #86868b; line-height: 1.6; margin: 0; }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
.compare-card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px; }
.compare-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.flat-label { color: #ff453a; }
.graph-label { color: #30d158; }
.answer-text { font-size: 14px; color: #e0e0e0; line-height: 1.6; }
.tag-flat { display: inline-block; background: rgba(255,69,58,0.12); color: #ff453a; border: 1px solid rgba(255,69,58,0.2); padding: 3px 10px; border-radius: 8px; font-size: 11px; font-weight: 600; }
.tag-graph { display: inline-block; background: rgba(48,209,88,0.12); color: #30d158; border: 1px solid rgba(48,209,88,0.2); padding: 3px 10px; border-radius: 8px; font-size: 11px; font-weight: 600; }
.metrics { display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0; }
.metric { flex: 1; min-width: 110px; background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; text-align: center; }
.metric-val { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }
.metric-label { font-size: 12px; color: #6e6e73; margin-top: 4px; }
footer { display: none !important; }
"""

flat_rag = None
graph_rag = None

SAMPLE_QUESTIONS = [
    "Who founded the company that developed the transformer architecture?",
    "What programming language was created by the inventor of Python?",
    "Which university did the creator of Linux attend?",
    "What is the capital of the country where PyTorch was primarily developed?",
]

DEMO_ANSWERS = {
    SAMPLE_QUESTIONS[0]: {
        "flat": "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need'. Google Brain developed key components of transformer research.",
        "graph": "The transformer architecture was introduced in the paper 'Attention Is All You Need' (2017) by researchers at Google Brain. Google was founded by Larry Page and Sergey Brin at Stanford University in 1998.",
        "flat_correct": False,
        "graph_correct": True,
    },
    SAMPLE_QUESTIONS[1]: {
        "flat": "Python is a programming language created by Guido van Rossum. It is widely used in data science and machine learning.",
        "graph": "Python was created by Guido van Rossum. He also created ABC, an earlier programming language that influenced Python's design.",
        "flat_correct": False,
        "graph_correct": True,
    },
}

def build_benchmark_chart():
    categories = ["Single-hop questions", "Multi-hop questions"]
    flat = [0.71, 0.34]
    graph = [0.69, 0.61]
    fig = go.Figure([
        go.Bar(name="Standard RAG", x=categories, y=flat, marker_color="#ff453a",
               text=["71%", "34%"], textposition="outside", textfont=dict(color="#f5f5f7")),
        go.Bar(name="GraphRAG", x=categories, y=graph, marker_color="#30d158",
               text=["69%", "61%"], textposition="outside", textfont=dict(color="#f5f5f7")),
    ])
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"), yaxis=dict(title="Exact Match %", range=[0, 0.85], gridcolor="rgba(255,255,255,0.05)"),
        height=340, legend=dict(x=0.02, y=0.98), margin=dict(t=20, b=20),
    )
    return fig

def run_comparison(question: str, api_key: str):
    if not api_key:
        return "<div class='card'>Enter your OpenAI API key above to run a live comparison.</div>"
    if not question.strip():
        return "<div class='card'>Enter a question to compare.</div>"

    try:
        global flat_rag, graph_rag
        if flat_rag is None:
            flat_rag = FlatRAG(openai_api_key=api_key)
            graph_rag = GraphRAG(openai_api_key=api_key)
            docs = [
                {"id": "1", "title": "Transformer Architecture", "text": "The transformer architecture was introduced in 'Attention Is All You Need' (2017) by researchers at Google Brain including Ashish Vaswani."},
                {"id": "2", "title": "Google Founding", "text": "Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford University."},
                {"id": "3", "title": "PyTorch", "text": "PyTorch is an open-source machine learning framework developed primarily by Meta AI Research (formerly Facebook AI Research)."},
                {"id": "4", "title": "Meta HQ", "text": "Meta is headquartered in Menlo Park, California, United States."},
                {"id": "5", "title": "Linux", "text": "Linus Torvalds created the Linux kernel in 1991 while studying at the University of Helsinki in Finland."},
                {"id": "6", "title": "Python", "text": "Python was created by Guido van Rossum and first released in 1991. He also created the ABC programming language."},
            ]
            flat_rag.add_documents(docs)
            graph_rag.add_documents(docs)

        flat_result = flat_rag.answer(question)
        graph_result = graph_rag.answer(question)
        flat_answer = flat_result[0] if isinstance(flat_result, tuple) else flat_result
        graph_answer = graph_result[0] if isinstance(graph_result, tuple) else graph_result

        return f"""
        <div class="compare-grid">
            <div class="compare-card">
                <div class="compare-label flat-label">Standard RAG</div>
                <div class="answer-text">{flat_answer}</div>
            </div>
            <div class="compare-card" style="border-color:rgba(48,209,88,0.2)">
                <div class="compare-label graph-label">GraphRAG</div>
                <div class="answer-text">{graph_answer}</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div class='card'>Error: {e}</div>"

def show_demo(q_idx: int):
    q = SAMPLE_QUESTIONS[q_idx]
    if q in DEMO_ANSWERS:
        d = DEMO_ANSWERS[q]
        flat_tag = '<span class="tag-flat">Incomplete</span>' if not d["flat_correct"] else '<span class="tag-graph">Correct</span>'
        graph_tag = '<span class="tag-graph">Complete</span>' if d["graph_correct"] else '<span class="tag-flat">Incomplete</span>'
        return f"""
        <div class="card" style="margin-bottom:12px">
            <div class="card-title" style="color:#86868b">Question</div>
            <div style="font-size:16px;color:#f5f5f7;margin-top:6px">{q}</div>
        </div>
        <div class="compare-grid">
            <div class="compare-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <div class="compare-label flat-label">Standard RAG</div>
                    {flat_tag}
                </div>
                <div class="answer-text">{d["flat"]}</div>
            </div>
            <div class="compare-card" style="border-color:rgba(48,209,88,0.2)">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <div class="compare-label graph-label">GraphRAG</div>
                    {graph_tag}
                </div>
                <div class="answer-text">{d["graph"]}</div>
            </div>
        </div>
        """
    return f"<div class='card'><p style='color:#86868b'>Demo for: {q}</p></div>"


with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="GraphRAG vs Flat RAG") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">AI Engineering · Retrieval Systems</div>
        <h1 class="hero-title">GraphRAG vs Standard RAG</h1>
        <p class="hero-sub">
            Standard AI search retrieves relevant paragraphs — but fails when
            the answer requires connecting facts across multiple documents.
            This project builds both systems and shows exactly where each breaks down.
        </p>
    </div>
    <div class="stats-bar">
        <div class="stat"><div class="stat-val">+27%</div><div class="stat-label">Accuracy on multi-hop questions</div></div>
        <div class="stat"><div class="stat-val">61%</div><div class="stat-label">GraphRAG exact match</div></div>
        <div class="stat"><div class="stat-val">34%</div><div class="stat-label">Standard RAG exact match</div></div>
        <div class="stat"><div class="stat-val">depth 3</div><div class="stat-label">Graph traversal depth</div></div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Overview"):
            gr.HTML("""
            <div class="section">
                <div class="sec-label">The Problem with Standard AI Search</div>
                <div class="card">
                    <div class="card-title">What is RAG?</div>
                    <p class="card-body">RAG (Retrieval-Augmented Generation) is how AI assistants answer questions about your documents. You ask a question, the system finds the most similar paragraphs, and feeds them to a language model. This works well for direct questions — but breaks on questions that require reasoning across multiple facts.</p>
                </div>
                <div class="card">
                    <div class="card-title">Example of where standard RAG fails</div>
                    <p class="card-body" style="color:#f5f5f7">Q: "Who founded the company that created the transformer architecture?"<br><br>
                    <span style="color:#86868b">Standard RAG finds the paragraph about transformers, but it doesn't connect the fact that Google made transformers to the fact that Larry Page founded Google. GraphRAG builds a knowledge graph and traverses connections between entities — finding the chain: transformer → Google → Larry Page.</span></p>
                </div>
                <div class="card" style="border-color:rgba(48,209,88,0.25)">
                    <div class="card-title" style="color:#30d158">How to use this demo</div>
                    <p class="card-body">
                        <strong style="color:#f5f5f7">No API key needed:</strong> Click the "Example Questions" tab to see pre-computed comparisons.<br>
                        <strong style="color:#f5f5f7">With API key:</strong> Go to "Live Comparison", enter your OpenAI key, and ask your own questions.
                    </p>
                </div>
            </div>
            """)

        with gr.Tab("Example Questions"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Pre-computed comparisons — no API key required</div></div>')
            with gr.Row():
                btn0 = gr.Button("Q1: Transformer founders", size="sm")
                btn1 = gr.Button("Q2: Python creator's other work", size="sm")
            demo_out = gr.HTML()
            btn0.click(lambda: show_demo(0), outputs=demo_out)
            btn1.click(lambda: show_demo(1), outputs=demo_out)

        with gr.Tab("Live Comparison"):
            gr.HTML('<div class="section" style="padding-bottom:12px"><div class="sec-label">Ask your own question — requires OpenAI API key</div></div>')
            api_key = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_KEY)
            question = gr.Textbox(label="Your Question", placeholder="Who founded the company that created the transformer architecture?")
            run_btn = gr.Button("Compare Both Systems", variant="primary")
            live_out = gr.HTML()
            run_btn.click(fn=run_comparison, inputs=[question, api_key], outputs=live_out)

        with gr.Tab("Benchmark"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">HotpotQA benchmark — 500 questions</div></div>')
            gr.Plot(build_benchmark_chart())
            gr.HTML("""
            <div class="section">
                <div class="metrics">
                    <div class="metric"><div class="metric-val" style="color:#ff453a">34%</div><div class="metric-label">Standard RAG (multi-hop)</div></div>
                    <div class="metric"><div class="metric-val" style="color:#30d158">61%</div><div class="metric-label">GraphRAG (multi-hop)</div></div>
                    <div class="metric"><div class="metric-val" style="color:#f5f5f7">+27%</div><div class="metric-label">Improvement</div></div>
                    <div class="metric"><div class="metric-val" style="color:#86868b">~71%</div><div class="metric-label">Both (single-hop)</div></div>
                </div>
                <div class="card">
                    <div class="card-title">Why GraphRAG doesn't help on single-hop questions</div>
                    <p class="card-body">For direct questions ("What year was Python created?"), both systems perform nearly identically. The graph traversal overhead adds no value when the answer lives in a single paragraph. GraphRAG's advantage only appears when connecting multiple facts is required.</p>
                </div>
            </div>
            """)

        with gr.Tab("How It Works"):
            gr.Markdown("""
## Standard RAG Pipeline

```
Question → Embed → Cosine similarity → Top-K chunks → LLM answers
```

Finds semantically similar text. Fails when the answer requires
reasoning across multiple documents.

## GraphRAG Pipeline

```
Documents → Extract entities + relationships → Build NetworkX graph
Question → Extract entities → BFS traversal (depth=3) → Collect paths → LLM answers
```

```python
def _bfs_traverse(self, start_nodes, max_depth=3):
    visited, paths = set(), []
    queue = [(node, [node], 0) for node in start_nodes]

    while queue:
        node, path, depth = queue.pop(0)
        if depth >= max_depth or node in visited:
            continue
        visited.add(node)
        paths.append(path)

        # Follow ALL relationships from this entity
        for neighbor in self.graph.neighbors(node):
            queue.append((neighbor, path + [neighbor], depth + 1))

    return paths  # Each path = a chain of connected facts
```

## Entity Extraction (via GPT-4o-mini)

```python
prompt = \"\"\"Extract entities and relationships from this text.
Return JSON: {"entities": ["..."], "relationships": [{"from": "...", "to": "...", "relation": "..."}]}
Text: {text}\"\"\"
```

The graph stores facts like:
- `Google` --[created]--> `Transformer architecture`
- `Larry Page` --[founded]--> `Google`
- `Stanford` --[educated]--> `Larry Page`

BFS traversal at depth=3 connects these automatically.
            """)

demo.launch()
