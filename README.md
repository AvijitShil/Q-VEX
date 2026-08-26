#  Q-VEX: A Hyper-Compressed Graph-Vector Database for Local GraphRAG & Multi-Agent Systems

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/qvex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Storage Compression](https://img.shields.io/badge/Compression-14.5x%20vs%20PDF-success.svg)](#-empirical-benchmarks)
[![Retrieval Latency](https://img.shields.io/badge/Latency-%3C25ms%20CPU-purple.svg)](#-empirical-benchmarks)
[![Zero Cloud Costs](https://img.shields.io/badge/Cloud%20APIs-%240.00-green.svg)](#-the-problem-q-vex-solves)
[![Integrations](https://img.shields.io/badge/Integrations-LangChain%20%7C%20LangGraph%20%7C%20CrewAI%20%7C%20LlamaIndex-orange.svg)](#-agentic-framework-integrations)

**Q-VEX** is an ultra-lightweight, local-first, disk-backed hybrid **Graph-Vector Database** designed for high-performance **GraphRAG** and **Multi-Agent Shared Memory**.

[Key Features](#-key-features) • [The Problem We Solve](#-the-problem-q-vex-solves) • [Tri-Modal Architecture](#-tri-modal-search-architecture) • [Benchmarks](#-empirical-benchmarks) • [Agentic Frameworks](#-agentic-framework-integrations) • [Edge Cases & Safeguards](#-edge-cases--architectural-safeguards) • [Quickstart](#-quickstart)

</div>

---

## The Problem Q-VEX Solves

Modern Retrieval-Augmented Generation (RAG) and Multi-Agent setups suffer from three critical bottlenecks:

```
❌ Standard Vector DBs                     ❌ Traditional GraphRAG (Neo4j/Memgraph)
• Float32 vectors consume 100s of GBs RAM   • Requires heavy external server infrastructure
• "Topology-blind": loses document flow    • Thousands of expensive LLM API calls to extract graphs
• Treats related text as isolated chunks    • High query latencies and complex deployment
```

###  The Q-VEX Solution
Q-VEX unifies **vector quantization (TurboQuant)**, **keyword search (SQLite FTS5)**, and **C-level graph traversal (SQLite Recursive CTEs)** into a single, compact, disk-backed database format.

* **14.5x Storage Compression:** Storing an entire searchable 535-page textbook (full text + graph edges + FTS5 index + vectors) requires only **2.2 MB on disk**—35.8% smaller than raw unindexed Float32 vectors alone.
* **Sub-25ms Hybrid Retrieval:** Pushes multi-hop graph expansion down to SQLite's C-compiled execution engine.
* **100% Local & Free:** Zero external daemon processes, zero API keys, and zero cloud hosting fees.
* **Shared Multi-Agent Memory:** Serves as an atomic, episodic shared memory layer for LangGraph and CrewAI swarms.

---

##  Key Features

*  **Tri-Modal Retrieval:** Merges BM25 keyword matching $\to$ Recursive CTE graph expansion $\to$ Quantized vector reranking.
*  **Zero-RAM BM25 Index:** Native SQLite FTS5 with automated SQL triggers for transparent CRUD synchronization.
*  **C-Level Graph Expansion:** Multi-hop neighbor traversal executed within a single SQL statement—no slow Python graph-walk loops.
*  **2-Bit / 4-Bit Vector Quantization:** Squeezes embeddings to a fraction of raw NumPy memory using `TurboQuantIndex` bitmask filtering.
*  **Batteries-Included Entity Extraction:** Optional lightweight GLiNER module (`urchade/gliner_small-v2.1`, ~150 MB) for zero-shot relational graph construction.
*  **Drop-in Framework Adapters:** First-class integrations for **LangChain**, **LangGraph**, **CrewAI**, and **LlamaIndex**.
*  **One-Line Interactive Graph Visualization:** Exports standalone, dark-themed HTML/JS interactive maps powered by `vis-network.js` (zero Python UI dependencies).
*  **Production-Grade Concurrency:** Thread-safe reentrant transactions (`threading.RLock`) with automatic cross-database rollback protection.

---

##  Tri-Modal Search Architecture

```
                      ┌────────────────────────────────────┐
                      │    Incoming Query (Text + Vec)     │
                      └─────────────────┬──────────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          │                          ▼
    1. BM25 Keyword Search              │                 (Optional Seed Union)
       (SQLite FTS5 C-Engine)           │
             │                          │
             ▼                          │
     [Top-K Seed Nodes]                 │
             │                          │
             ▼                          │
    2. Recursive Graph Expansion        │
       (SQLite Recursive CTE)           │
       • Traverses 'next_chunk'         │
       • Traverses 'shared_entity'      │
             │                          │
             ▼                          ▼
     [Expanded K-Hop Candidates (Candidate Subgraph)]
                                        │
                                        ▼
                       3. Quantized Vector Reranking
                          (TurboQuant 4-bit / 2-bit Index)
                          • Evaluates allowlist mask only
                                        │
                                        ▼
                      ┌────────────────────────────────────┐
                      │    Final Context-Rich Results      │
                      └────────────────────────────────────┘
```

| Stage | Engine | Latency | Purpose |
| :--- | :--- | :---: | :--- |
| **Stage 1: BM25** | SQLite FTS5 | ~1.5 ms | High-precision exact keyword and entity match with zero RAM footprint |
| **Stage 2: Graph CTE** | SQLite Recursive CTE | ~0.8 ms | Discovers structural and entity-linked neighbors up to $k$ hops |
| **Stage 3: Vector Rerank** | TurboQuant Quantized Index | ~2.1 ms | Semantic similarity reranking restricted to the graph-expanded candidate set |

---

##  Empirical Benchmarks

### 1. Storage & Compression Benchmark
*Evaluated on O'Reilly's **AI Engineering** (535 pages, 2,348 chunks, 384-dim `all-MiniLM-L6-v2` embeddings).*

| Storage Format | On-Disk Size | Compression vs Float32 | Compression vs Source PDF | Net Storage Savings |
| :--- | :---: | :---: | :---: | :---: |
| **Source PDF Document** | 31.92 MB | — | Baseline | — |
| **Raw Float32 Vectors (No Text)** | 3.44 MB | 1.00x | 9.28x smaller | Baseline |
| **4-Bit Q-VEX DB (Text + Graph + Vectors)** | **2.42 MB** | **1.42x smaller** | **13.18x smaller** | **29.6% saved** |
| **2-Bit Q-VEX DB (Text + Graph + Vectors)** | **2.21 MB** | **1.56x smaller** | **14.46x smaller** | **35.8% saved** |

> **Takeaway:** Q-VEX stores the entire searchable database (text, graph relationships, FTS5 virtual table, and vectors) in **just 6.9% of the source PDF size**, making it smaller than storing raw float32 vectors alone!

---

### 2. Retrieval Speed & Relevance Benchmark
*Timed across 10 diverse technical queries ($k=5, \text{hops}=2$) on a standard CPU:*

| Metric | Result |
| :--- | :---: |
| **Average Query Latency** | **23.27 ms** |
| **Fastest Query Latency** | **9.44 ms** |
| **Average Top Cosine Similarity** | **0.6467** |
| **Precision / Target Match Rate** | **100% (5/5 hits)** |

---

### 3. Multi-Agent Shared Memory Benchmark
*Evaluating Q-VEX as an episodic memory communication backend for autonomous agent swarms:*

| Metric | 0-Hop (Pure Vector) | 2-Hop (Q-VEX Graph Expansion) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Contextual Fact Recall** | 71.0% | **83.5%** | **+12.5% Recall Gain** |
| **Agent Memory Write Latency** | — | **38.46 ms** | Real-time logging |
| **Agent Memory Read Latency** | — | **2.60 ms** | Instantaneous retrieval |
| **Concurrent Throughput (8 Threads)** | — | **151.0 ops/sec** | 100% transaction integrity |

---

##  Quickstart

### Installation

```bash
# Core package (ultra-lean)
pip install qvex

# With local zero-shot entity extraction
pip install qvex[gliner]

# With framework adapters
pip install qvex[langchain,llamaindex]
```

### Basic CRUD & Hybrid Search

```python
import numpy as np
from qvex import QVEX

# 1. Initialize database (specify embedding dimension)
qvex = QVEX(dim=384, storage_dir="./my_knowledge_db", bit_width=4)

# 2. Add document nodes
vec_a = np.random.randn(384).astype(np.float32)
vec_b = np.random.randn(384).astype(np.float32)

doc_1 = qvex.add("Attention mechanisms allow transformers to weight token relevance.", vector=vec_a)
doc_2 = qvex.add("FlashAttention accelerates transformer training using GPU SRAM tiling.", vector=vec_b)

# 3. Create relational graph edges
qvex.add_edge(doc_1, doc_2, edge_type="optimized_by", confidence=0.95)

# 4. Execute hybrid search (BM25 -> 2-Hop Graph Expansion -> Quantized Vector Rerank)
query_vec = np.random.randn(384).astype(np.float32)
results = qvex.search(query="transformer attention speed", vector=query_vec, k=5, hops=2)

for r in results:
    print(f"[{r.score:.4f}] (Hop {r.hop_distance}) Node #{r.id}: {r.text}")

# 5. Export interactive HTML knowledge graph visualization
qvex.visualize_graph("knowledge_graph.html")
```

---

##  High-Level Document Ingestion & GLiNER Extraction

Ingest long text, automatically chunk, embed, and extract knowledge graph relationships in one call:

```python
from sentence_transformers import SentenceTransformer
from qvex import QVEX
from qvex.extractor.gliner_extractor import GraphExtractor

# Load lightweight models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
extractor = GraphExtractor(labels=["Technology", "Organization", "Method"])

qvex = QVEX(dim=384, storage_dir="./book_graph")

# Ingest and automatically extract sequential ('next_chunk') and entity ('shared_entity') edges
result = qvex.ingest(
    text=document_text,
    embed_fn=lambda t: embedder.encode(t).astype(np.float32),
    chunk_size=512,
    chunk_overlap=50,
    extractor=extractor,
)

print(f"Ingested {result.chunk_count} chunks with {result.edge_count} relational edges.")
```

---

##  Agentic Framework Integrations

### 1. LangGraph Multi-Agent Shared Memory

Use Q-VEX as a shared semantic & episodic memory layer across multiple LangGraph agents:

```python
from langgraph.graph import StateGraph, START, END
from qvex import QVEX
from qvex.integrations.langgraph_adapter import QVEXSemanticMemorySaver

qvex_db = QVEX(dim=384, storage_dir="./agent_memory")
memory_saver = QVEXSemanticMemorySaver(qvex_db, embed_fn=my_embed_function)

# In Researcher Agent Node
def researcher_node(state):
    mem_id = memory_saver.save_memory(
        content="Discovered new vulnerability pattern in RPN proposals.",
        metadata={"agent": "researcher"}
    )
    return {"status": "memory_saved", "mem_id": mem_id}

# In Analyst Agent Node (Retrieves multi-hop connected insights)
def analyst_node(state):
    memories = memory_saver.retrieve_memory(query="vulnerability patterns", k=3, hops=2)
    return {"insights": [m["content"] for m in memories]}
```

### 2. LangChain Drop-in VectorStore

```python
from qvex import QVEX
from qvex.integrations import QVEXLangChainVectorStore

qvex = QVEX(dim=384, storage_dir="./langchain_store")
vector_store = QVEXLangChainVectorStore(qvex=qvex, embedding=my_embeddings)

# Standard LangChain retriever interface
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("How does quantization reduce memory?")
```

### 3. CrewAI Working Memory Storage

```python
from qvex import QVEX
from qvex.integrations.crewai_adapter import QVEXCrewAIMemoryStorage

qvex = QVEX(dim=384, storage_dir="./crewai_memory")
crew_memory = QVEXCrewAIMemoryStorage(qvex_instance=qvex, embed_fn=my_embed_function)

# Crew agents save and search with similarity thresholds
crew_memory.save("Task #12 completed with optimal parameter settings.")
past_context = crew_memory.search("parameter settings", limit=3, score_threshold=0.35)
```

---

##  Edge Cases & Architectural Safeguards

Q-VEX is engineered for production stability with explicit defensive safeguards:

| Edge Case / Failure Mode | Root Cause in Naive Systems | Q-VEX Architectural Safeguard |
| :--- | :--- | :--- |
| **Concurrent Mutation Deadlocks** | Multi-threaded agents writing simultaneously to SQLite & vector store | Implements reentrant `threading.RLock()` in `core.py`. Nested operations (e.g. `.ingest()` $\to$ `.add()`) safely re-enter the lock without deadlocking. |
| **Partial Transaction Failures** | Vector index succeeds but SQLite throws an integrity error | **Atomic Rollback Handler:** If SQLite node insertion fails, Q-VEX immediately soft-deletes the indexed vector before re-raising the exception. |
| **Vector Dimension Mismatches** | Passing a 768-dim vector to a 384-dim index causes low-level C++ crashes | Explicit input validation assertions in `add()` and `update()` (`if vector.shape[-1] != self._dim: raise ValueError(...)`). |
| **Infinite Graph Cycles** | Circular relationships ($A \to B \to A$) causing infinite loops during traversal | SQLite Recursive CTE uses `SELECT DISTINCT` with strict depth bounding (`WHERE k.hop < :max_hops`). |
| **SQLite FTS5 Token Syntax Errors** | User queries containing special punctuation or boolean operators (`OR`, `NOT`, `*`) crash MATCH syntax | **Regex Query Sanitizer:** `re.findall(r'\b\w+\b', query)` extracts clean alphanumeric tokens and formats them into safe prefix OR queries (`"token1*" OR "token2*"`). |
| **Expensive Vector Requantization** | Deleting vectors from a quantized index requires full index rebuild | **Soft Deletion Masking:** Deleted node vectors are flagged in a boolean bitmask, excluding them from similarity search with zero rebuild cost. |
| **Cascade Edge Cleanup** | Deleting a node leaves dangling edges referencing nonexistent IDs | SQLite foreign key constraints configured with `ON DELETE CASCADE` automatically prune associated edges and FTS5 entries. |

---

##  Project Structure

```
qvex/
├── src/qvex/
│   ├── core.py               # Main QVEX orchestrator (thread-safe, atomic mutations)
│   ├── graph_db.py           # SQLite wrapper (CRUD, FTS5 BM25, recursive CTEs)
│   ├── vector_store.py       # TurboQuant wrapper (2/4-bit quantization, mask filtering)
│   ├── models.py             # Pydantic data models (NodeData, EdgeData, SearchResult)
│   ├── schema.sql            # SQLite schema, indices, foreign keys, and FTS5 triggers
│   ├── extractor/            # Entity extraction (BaseExtractor, GLiNER)
│   └── integrations/         # Framework adapters (LangChain, LangGraph, CrewAI, LlamaIndex)
├── Eval/                     # Rigorous systems evaluation suite
│   ├── memory_compression.py # 535-page PDF memory footprint benchmark
│   ├── recall_and_speed.py   # Query latency & cosine relevance benchmark
│   ├── eval_multi_agent_system.py # Multi-agent shared memory benchmark
│   └── results/              # Generated benchmark CSVs, Markdown summaries & HTML graphs
├── tests/                    # PyTest test suite (50+ unit tests covering all components)
├── examples/                 # Quickstart and full RAG pipeline demonstrations
└── showcase.html             # Interactive showcase & visual infographic dashboard
```

---

##  Running Tests & Benchmarks

```bash
# Clone the repository
git clone https://github.com/your-username/qvex.git
cd qvex

# Install with development dependencies
pip install -e ".[dev,gliner]"

# Run full unit test suite
pytest tests/ -v

# Run real-world PDF compression & speed evaluations
python Eval/memory_compression.py
python Eval/recall_and_speed.py

# Run multi-agent shared memory benchmark
python Eval/eval_multi_agent_system.py
```

---

##  License & Citation

Q-VEX is released under the **MIT License**.

```bibtex
@software{qvex2026,
  title = {Q-VEX: A Hyper-Compressed Graph-Vector Database Engine for Compact GraphRAG},
  author = {Q-VEX Development Team},
  year = {2026},
  url = {https://github.com/your-username/qvex},
  version = {0.3.0}
}
```
