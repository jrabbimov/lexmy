"""Vector retrieval with optional profile-aware bias."""

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
CHROMA_DIR = str(ARTIFACTS / "chroma")
EMBED_MODEL = "BAAI/bge-m3"

ACT_BIAS = {
    "private_sdn_bhd": {"act777": 1.15, "pdpa": 1.05},
    "llp": {"llp": 1.15, "pdpa": 1.05},
    "sole_prop": {"roba197": 1.15, "pdpa": 1.05},
}


# ── Artifact loaders ──────────────────────────────────────────────────────────


def load_chroma():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("lexmy", embedding_function=ef)


def load_graph():
    with open(ARTIFACTS / "graph.pkl", "rb") as f:
        return pickle.load(f)


def load_concept_vocab():
    with open(ARTIFACTS / "concept_vocab.json") as f:
        return json.load(f)


def load_sections_full():
    with open(ARTIFACTS / "sections_full.json", encoding="utf-8") as f:
        return json.load(f)


# ── Retrieval ─────────────────────────────────────────────────────────────────


def _apply_bias(results: list, business_form: str) -> list:
    if not business_form or business_form not in ACT_BIAS:
        return results
    bias_map = ACT_BIAS[business_form]
    for r in results:
        act = r["meta"].get("act", "")
        boost = bias_map.get(act, 1.0)
        r["_distance"] = r.get("_distance", 0.0) / boost
    return sorted(results, key=lambda x: x["_distance"])


def retrieve_one(sub_query: str, coll, top_k: int = 4, business_form: str = "") -> list:
    """Vector search for one sub-query."""
    vr = coll.query(query_texts=[sub_query], n_results=top_k)
    docs = vr["documents"][0]
    metas = vr["metadatas"][0]
    distances = vr.get("distances", [[0.0] * len(docs)])[0]

    results = []
    seen = set()
    for doc, meta, dist in zip(docs, metas, distances):
        sid = meta.get("section_id", doc[:40])
        if sid in seen:
            continue
        seen.add(sid)
        results.append({"text": doc, "meta": meta, "_distance": dist})

    return _apply_bias(results, business_form)


def retrieve_all(
    sub_queries: list,
    coll,
    top_k: int = 4,
    business_form: str = "",
    use_graph: bool = False,
) -> list:
    """Retrieve + dedupe across sub-queries (vector only)."""
    seen, all_results = set(), []
    for q in sub_queries:
        for r in retrieve_one(q, coll, top_k, business_form):
            sid = r["meta"].get("section_id", r["text"][:40])
            if sid not in seen:
                seen.add(sid)
                all_results.append(r)
    return all_results
