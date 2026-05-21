"""Vector + graph retrieval with profile-aware bias."""
import json
import pickle
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


ARTIFACTS    = Path(__file__).parent.parent / "artifacts"
CHROMA_DIR   = str(ARTIFACTS / "chroma")
EMBED_MODEL  = "BAAI/bge-m3"

# Score multiplier per business form. Never excludes other acts.
ACT_BIAS = {
    "private_sdn_bhd": {"act777": 1.15, "pdpa": 1.05},
    "llp":             {"llp":    1.15, "pdpa": 1.05},
    "sole_prop":       {"roba197": 1.15, "pdpa": 1.05},
}


# ── Artifact loaders (called once on app startup) ─────────────────────────────

def load_graph():
    with open(ARTIFACTS / "graph.pkl", "rb") as f:
        return pickle.load(f)

def load_concept_vocab():
    with open(ARTIFACTS / "concept_vocab.json") as f:
        return json.load(f)

def load_sections_full():
    with open(ARTIFACTS / "sections_full.json", encoding="utf-8") as f:
        return json.load(f)

def load_chroma():
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("lexmy", embedding_function=ef)


# ── Concept lookup ────────────────────────────────────────────────────────────

def concept_lookup(text: str, vocab: dict) -> list:
    tl = text.lower()
    return [c for c, kws in vocab.items() if any(k in tl for k in kws)]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _apply_bias(results: list, business_form: str) -> list:
    """Re-order results by applying ACT_BIAS multiplier to scores."""
    if not business_form or business_form not in ACT_BIAS:
        return results
    bias_map = ACT_BIAS[business_form]
    scored = []
    for r in results:
        act = r["meta"].get("act", "")
        boost = bias_map.get(act, 1.0)
        # Chroma returns distances (lower = better). Divide so larger boost moves down distance.
        r["_score"] = r.get("_distance", 0.0) / boost
        scored.append(r)
    return sorted(scored, key=lambda x: x["_score"])


def retrieve_one(sub_query: str,
                 coll,
                 graph,
                 vocab: dict,
                 top_k: int = 4,
                 business_form: str = "",
                 use_graph: bool = True) -> list:
    """Retrieve candidates for one sub-query."""
    seen, results = set(), []

    # vector search
    vr = coll.query(query_texts=[sub_query], n_results=top_k)
    docs      = vr["documents"][0]
    metas     = vr["metadatas"][0]
    distances = vr.get("distances", [[0.0] * len(docs)])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        sid = meta.get("section_id", doc[:40])
        if sid not in seen:
            seen.add(sid)
            results.append({
                "text": doc,
                "meta": meta,
                "_distance": dist,
                "source": "vector",
            })

    # graph concept lookup
    if use_graph:
        for concept in concept_lookup(sub_query, vocab):
            if concept not in graph:
                continue
            for sec_id in graph.predecessors(concept):
                if graph.nodes[sec_id].get("type") != "section":
                    continue
                if sec_id in seen:
                    continue
                seen.add(sec_id)
                n = graph.nodes[sec_id]
                results.append({
                    "text": n.get("content", "")[:500],
                    "meta": {
                        "section_id":    sec_id,
                        "section_title": n.get("section_title", ""),
                        "act":           n.get("act", ""),
                        "part":          "",
                        "section_num":   n.get("section_num", 0),
                    },
                    "_distance": 1.0,   # neutral
                    "source":    "graph",
                })

    results = _apply_bias(results, business_form)
    return results[:top_k * 2]


def retrieve_all(sub_queries: list,
                 coll,
                 graph,
                 vocab: dict,
                 top_k: int = 3,
                 business_form: str = "",
                 use_graph: bool = True) -> list:
    """Retrieve + dedupe across sub-queries."""
    seen, all_results = set(), []
    for q in sub_queries:
        for r in retrieve_one(q, coll, graph, vocab, top_k, business_form, use_graph):
            sid = r["meta"].get("section_id", r["text"][:40])
            if sid not in seen:
                seen.add(sid)
                all_results.append(r)
    return all_results
