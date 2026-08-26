"""
eval_multi_agent_system.py
==========================
Comprehensive Evaluation Benchmark for Q-VEX as a Multi-Agent Shared Memory & Graph Engine.

Simulates a 3-Agent Collaborative Workflow:
- Agent 1: ResearchAgent (Ingests observations & links entity/topic dependencies)
- Agent 2: VerificationAgent (Validates claims & builds verified_by cross-edges)
- Agent 3: AnalystAgent (Executes multi-hop synthesis queries)

Evaluates:
1. Multi-Hop Graph Recall (0-hop vs 1-hop vs 2-hop CTE expansion)
2. Agent Memory Write & Read Latencies
3. Concurrent Thread-Safe Agent Transactions
4. Quantization Storage Efficiency (4-bit TurboVec vs float32)
5. Exports interactive visual graph & results CSV/Markdown
"""

import os
import time
import shutil
import hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from qvex import QVEX
from qvex.integrations.langgraph_adapter import QVEXSemanticMemorySaver

DIM = 384
STORAGE_DIR = "./eval_agent_memory_db"
RESULTS_DIR = "./Eval/results"

def get_deterministic_embedding(text: str, dim: int = DIM) -> np.ndarray:
    """Generates a stable semantic-like pseudo-embedding using SHA256 projections for fast local evaluation."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
        vec[idx] += sign * (1.0 + (h % 10) / 10.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec

def run_multi_agent_evaluation():
    print("=" * 70)
    print("  Q-VEX MULTI-AGENT COLLABORATION & MEMORY BENCHMARK")
    print("=" * 70)

    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    qvex_db = QVEX(dim=DIM, storage_dir=STORAGE_DIR, bit_width=4)
    memory_saver = QVEXSemanticMemorySaver(
        qvex_instance=qvex_db,
        embed_fn=lambda t: get_deterministic_embedding(t, DIM)
    )

    # -------------------------------------------------------------
    # 1. AGENT INGESTION & KNOWLEDGE GRAPH CREATION
    # -------------------------------------------------------------
    print("\n[PHASE 1] Agent Ingestion & Collaborative Memory Graph Creation")
    
    agent_observations = [
        # Researcher Agent observations on Distributed Consensus
        ("ResearchAgent", "Consensus Protocol", "Raft consensus uses leader election and log replication for fault tolerance."),
        ("ResearchAgent", "Consensus Protocol", "Paxos guarantees safety under asynchronous network partitions via two-phase commit."),
        ("ResearchAgent", "Database Storage", "LSM-Tree storage engines batch writes to memory and flush sorted SSTables to disk."),
        ("ResearchAgent", "Database Storage", "B+ Trees provide predictable read latencies with logarithmic disk lookups."),
        
        # Vector Indexing observations
        ("ResearchAgent", "Vector Quantization", "TurboVec quantizes 384-dimensional float32 embeddings to 4-bit scalar representations."),
        ("ResearchAgent", "Vector Quantization", "Scalar quantization reduces memory consumption by 75% with negligible recall degradation."),
        ("ResearchAgent", "Graph Retrieval", "Recursive Common Table Expressions (CTEs) in SQLite enable k-hop graph traversal at C speed."),
        ("ResearchAgent", "Graph Retrieval", "Tri-modal search merges BM25 keyword seeds with CTE graph walks and vector reranking."),
        
        # Agent Communication observations
        ("VerificationAgent", "Agent Coordination", "Multi-agent coordination requires shared atomic memory to prevent duplicate tasks."),
        ("VerificationAgent", "Agent Coordination", "Episodic memory savers enable cross-agent contextual knowledge transfer."),
        ("VerificationAgent", "Verification", "Verified Raft consensus implementation in Rust outperforms non-replicated storage."),
        ("VerificationAgent", "Verification", "Quantized vector indexing maintains 98.4% top-5 retrieval accuracy under 4-bit compression.")
    ]

    write_latencies = []
    node_ids = []
    
    t0 = time.perf_counter()
    for agent_name, topic, content in agent_observations:
        tw0 = time.perf_counter()
        nid = memory_saver.save_memory(
            content=f"[{agent_name}] {topic}: {content}",
            metadata={"agent": agent_name, "topic": topic}
        )
        write_latencies.append((time.perf_counter() - tw0) * 1000)
        node_ids.append((nid, agent_name, topic, content))
    
    # Establish semantic & dependency edges between agent discoveries
    edge_mappings = [
        (0, 1, "competing_protocol"),     # Raft <-> Paxos
        (2, 3, "competing_architecture"), # LSM <-> B+ Tree
        (4, 5, "quantization_principle"), # TurboVec <-> Scalar Quantization
        (6, 7, "hybrid_retrieval_link"),  # CTE <-> Tri-modal Search
        (5, 7, "integrated_subsystem"),    # Quantization <-> Tri-modal
        (8, 9, "coordination_pattern"),   # Multi-agent <-> Episodic Memory
        (0, 10, "verified_implementation"),# Raft <-> Verified Raft
        (4, 11, "accuracy_validation"),   # TurboVec <-> 4-bit Accuracy
        (7, 9, "agent_memory_backend")    # Tri-modal <-> Episodic Memory
    ]

    for src_idx, tgt_idx, rel_type in edge_mappings:
        src_id = node_ids[src_idx][0]
        tgt_id = node_ids[tgt_idx][0]
        qvex_db.add_edge(src_id, tgt_id, edge_type=rel_type, confidence=1.0)

    total_ingest_time = (time.perf_counter() - t0) * 1000
    print(f"  -> Ingested {len(node_ids)} agent memories and created {len(edge_mappings)} relational graph edges.")
    print(f"  -> Avg Write Latency: {np.mean(write_latencies):.2f} ms | Total Time: {total_ingest_time:.2f} ms")

    # -------------------------------------------------------------
    # 2. MULTI-HOP RETRIEVAL EVALUATION (0-hop vs 1-hop vs 2-hop)
    # -------------------------------------------------------------
    print("\n[PHASE 2] Multi-Hop Agent Retrieval Benchmark (Graph Expansion vs Pure Vector)")

    eval_queries = [
        ("TurboVec vector quantization memory savings", ["TurboVec", "Scalar quantization", "accuracy"]),
        ("Raft consensus leader election fault tolerance", ["Raft", "Paxos", "Rust"]),
        ("SQLite Recursive CTE graph traversal search", ["CTE", "Tri-modal", "Quantization", "Episodic"]),
        ("Multi-agent coordination shared episodic memory", ["coordination", "Episodic", "Tri-modal"])
    ]

    query_results = []
    
    for query, expected_keywords in eval_queries:
        for hops in [0, 1, 2]:
            t_start = time.perf_counter()
            query_vec = get_deterministic_embedding(query, DIM)
            results = qvex_db.search(query=query, vector=query_vec, k=5, hops=hops)
            lat_ms = (time.perf_counter() - t_start) * 1000
            
            retrieved_texts = " ".join([r.text for r in results])
            matched_keywords = sum(1 for kw in expected_keywords if kw.lower() in retrieved_texts.lower())
            recall_score = matched_keywords / len(expected_keywords)
            
            query_results.append({
                "Query": query[:35] + "...",
                "Hops": hops,
                "Results_Count": len(results),
                "Keyword_Recall": round(recall_score, 2),
                "Latency_ms": round(lat_ms, 2),
                "Top_Score": round(results[0].score, 4) if results else 0.0
            })

    df_queries = pd.DataFrame(query_results)
    print("\n" + df_queries.to_markdown(index=False))

    # -------------------------------------------------------------
    # 3. CONCURRENT AGENT ACCESS STRESS TEST
    # -------------------------------------------------------------
    print("\n[PHASE 3] Concurrent Multi-Agent Read/Write Stress Test")
    
    NUM_THREADS = 8
    OPS_PER_THREAD = 50
    
    def worker_agent_task(worker_id: int):
        success_writes = 0
        success_reads = 0
        for i in range(OPS_PER_THREAD):
            # Write observation
            text = f"[Agent-{worker_id}] Transaction #{i}: Periodic state update on resource {i % 5}."
            vec = get_deterministic_embedding(text, DIM)
            nid = qvex_db.add(text, vec, metadata={"worker": worker_id, "seq": i})
            if nid > 0:
                success_writes += 1
            
            # Read observation
            q = f"resource {i % 5} state update"
            qv = get_deterministic_embedding(q, DIM)
            res = qvex_db.search(q, qv, k=3, hops=1)
            if len(res) > 0:
                success_reads += 1
        return worker_id, success_writes, success_reads

    t_conc_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker_agent_task, wid) for wid in range(NUM_THREADS)]
        total_writes = 0
        total_reads = 0
        for fut in as_completed(futures):
            wid, sw, sr = fut.result()
            total_writes += sw
            total_reads += sr
    
    conc_time = time.perf_counter() - t_conc_start
    total_ops = total_writes + total_reads
    throughput = total_ops / conc_time

    print(f"  -> Concurrency: {NUM_THREADS} Worker Threads | Total Operations: {total_ops:,}")
    print(f"  -> Total Time: {conc_time:.2f} s | Throughput: {throughput:.1f} ops/sec")
    print(f"  -> Write Success Rate: {total_writes / (NUM_THREADS * OPS_PER_THREAD):.1%}")
    print(f"  -> Read Success Rate: {total_reads / (NUM_THREADS * OPS_PER_THREAD):.1%}")

    # -------------------------------------------------------------
    # 4. STORAGE & QUANTIZATION METRICS
    # -------------------------------------------------------------
    print("\n[PHASE 4] Storage & Memory Footprint Metrics")
    
    total_nodes = len(qvex_db.graph_db.get_all_nodes())
    total_edges = len(qvex_db.graph_db.get_all_edges())
    
    # Calculate disk usage
    db_file = os.path.join(STORAGE_DIR, "qvex.db")
    tq_file = os.path.join(STORAGE_DIR, "vectors.tq")
    
    db_size_kb = os.path.getsize(db_file) / 1024 if os.path.exists(db_file) else 0
    tq_size_kb = os.path.getsize(tq_file) / 1024 if os.path.exists(tq_file) else 0
    total_kb = db_size_kb + tq_size_kb
    
    raw_float32_kb = (total_nodes * DIM * 4) / 1024
    compression_ratio = raw_float32_kb / tq_size_kb if tq_size_kb > 0 else 1.0
    
    print(f"  -> Total Agent Memories Stored : {total_nodes}")
    print(f"  -> Total Relational Graph Edges: {total_edges}")
    print(f"  -> SQLite Graph DB Size       : {db_size_kb:.2f} KB")
    print(f"  -> TurboVec 4-bit Vector Size  : {tq_size_kb:.2f} KB")
    print(f"  -> Total Q-VEX Storage Size    : {total_kb:.2f} KB")
    print(f"  -> Raw Float32 Vector Baseline : {raw_float32_kb:.2f} KB")
    print(f"  -> Vector Compression Ratio    : {compression_ratio:.2f}x smaller")

    # -------------------------------------------------------------
    # 5. VISUAL GRAPH EXPORT & ARTIFACT SUMMARY
    # -------------------------------------------------------------
    graph_html_path = os.path.join(RESULTS_DIR, "agent_communication_graph.html")
    exported_html = qvex_db.visualize_graph(graph_html_path)
    print(f"\n[PHASE 5] Visual Graph Export: {exported_html}")

    # Summary table
    summary_data = {
        "Metric": [
            "Agent Write Latency (Mean)",
            "Agent Read Latency (Mean, 2-hop)",
            "0-Hop Keyword Recall (Vector only)",
            "2-Hop Keyword Recall (Graph Expansion)",
            "Recall Gain via Graph Expansion",
            "Concurrent Throughput (8 threads)",
            "Vector Compression vs Float32",
            "Total Nodes / Edges in Graph"
        ],
        "Result": [
            f"{np.mean(write_latencies):.2f} ms",
            f"{df_queries[df_queries['Hops'] == 2]['Latency_ms'].mean():.2f} ms",
            f"{df_queries[df_queries['Hops'] == 0]['Keyword_Recall'].mean():.1%}",
            f"{df_queries[df_queries['Hops'] == 2]['Keyword_Recall'].mean():.1%}",
            f"+{(df_queries[df_queries['Hops'] == 2]['Keyword_Recall'].mean() - df_queries[df_queries['Hops'] == 0]['Keyword_Recall'].mean()):.1%}",
            f"{throughput:.1f} ops/sec",
            f"{compression_ratio:.2f}x",
            f"{total_nodes} nodes / {total_edges} edges"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    summary_md_path = os.path.join(RESULTS_DIR, "multi_agent_eval_summary.md")
    summary_csv_path = os.path.join(RESULTS_DIR, "multi_agent_eval_summary.csv")
    df_summary.to_markdown(summary_md_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("  FINAL EVALUATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(df_summary.to_markdown(index=False))
    print("\nAll evaluation artifacts saved to:", RESULTS_DIR)

if __name__ == "__main__":
    run_multi_agent_evaluation()
