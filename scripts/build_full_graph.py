import sys
import json
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from graphify.extract import extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html

print("Starting Graphify comprehensive build...")

# 1. Collect all target code + doc files (excluding .venv, node_modules, .git, etc.)
scan_root = Path(".")
files = []
for p in scan_root.glob("**/*"):
    if any(part.startswith(".") for part in p.parts):
        continue
    if "node_modules" in p.parts:
        continue
    if "graphify-out" in p.parts:
        continue
    if "dist" in p.parts:
        continue
    if p.is_file() and p.suffix.lower() in (".py", ".ts", ".tsx", ".md"):
        files.append(p)

print(f"Collected {len(files)} files for extraction (Python, TypeScript, React, Markdown).")

# 2. Run AST + Markdown structural extraction
extraction = extract(files, cache_root=scan_root, parallel=False)
nodes = extraction.get("nodes", [])
edges = extraction.get("edges", [])
print(f"Extraction completed: {len(nodes)} nodes, {len(edges)} edges.")

# Save extraction cache
out_dir = Path("graphify-out")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / ".graphify_extract.json").write_text(json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")

# 3. Build Graph
G = build_from_json(extraction, root=".", directed=False)
print(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# 4. Cluster & Analyze
communities = cluster(G)
cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)

# 5. Descriptive Community Labeling
labels = {}
for cid, member_ids in communities.items():
    member_labels = [G.nodes[n].get("label", n) for n in member_ids[:8]]
    member_sources = [G.nodes[n].get("source_file", "") for n in member_ids]
    
    # Infer theme
    sources_str = " ".join(member_sources).lower()
    labels_str = " ".join(member_labels).lower()
    
    if "conversation" in sources_str or "phase" in labels_str or "plan" in labels_str:
        label = "Agent Conversation Memory & Strategy"
    elif "frontend" in sources_str or "component" in sources_str or "app.tsx" in sources_str:
        label = "Institutional React 19 Frontend"
    elif "ingest" in sources_str or "chunk" in sources_str or "parser" in sources_str:
        label = "Financial Ingestion & Smart Chunker"
    elif "retriev" in sources_str or "bm25" in sources_str or "rerank" in sources_str or "chroma" in sources_str:
        label = "Hybrid Retrieval & Reranker"
    elif "agent" in sources_str or "verifier" in sources_str or "ast" in sources_str or "eval" in sources_str:
        label = "Multi-Agent Synthesis & Math AST Verifier"
    elif "sebi" in sources_str or "export" in sources_str or "pdf" in sources_str:
        label = "SEBI Reg 24(7) Audit Dossier Engine"
    elif "benchmark" in sources_str:
        label = "IndFinRAG-Bench Evaluation Suite"
    elif "api" in sources_str or "server" in sources_str:
        label = "FastAPI Backend & Document APIs"
    elif "test" in sources_str:
        label = "Automated Test Suite & Verification"
    elif member_labels:
        # Fallback to top member label
        top_name = member_labels[0].replace("_", " ").title()
        label = f"Subsystem: {top_name}"
    else:
        label = f"Community {cid}"
        
    labels[cid] = label

print(f"Identified & labeled {len(communities)} communities.")

# 6. Generate Questions & Report
questions = suggest_questions(G, communities, labels)

# Detection mock for report
detection = {
    "total_files": len(files),
    "total_words": sum(len(p.read_text(encoding="utf-8", errors="ignore").split()) for p in files),
    "files": {
        "code": [str(f) for f in files if f.suffix != ".md"],
        "document": [str(f) for f in files if f.suffix == ".md"]
    }
}
tokens = {"input": 0, "output": 0}

report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, ".", suggested_questions=questions)
(out_dir / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
print("Wrote graphify-out/GRAPH_REPORT.md")

# 7. Export graph.json
to_json(G, communities, "graphify-out/graph.json")
print("Wrote graphify-out/graph.json")

# 8. Export HTML visualization
to_html(G, communities, "graphify-out/graph.html", community_labels=labels)
print("Wrote graphify-out/graph.html")

print("\n=== BUILD COMPLETE ===")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Communities: {len(communities)}")
print(f"God Nodes: {len(gods)}")
for gnode in gods[:5]:
    print(f"  - {gnode.get('label', gnode.get('id'))} (degree: {gnode.get('degree')})")
