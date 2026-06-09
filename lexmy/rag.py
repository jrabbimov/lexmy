"""RAG orchestration: query split → retrieve → prompt → LLM."""
import re
import json

from .prompt    import ANSWER_PROMPT
from .llm       import llm_call, stream_call
from .retrieval import retrieve_all


# ── Query splitter (rule-based) ────────────────────────────────────────────────

def split_query(query: str) -> list:
    """Break complex questions into focused sub-queries. Returns ≥ 1 item."""
    q = query.strip()

    # Multiple sentences ending with "?"
    parts = [p.strip() for p in re.split(r"\?\s+", q) if p.strip()]
    if len(parts) > 1:
        return [p if p.endswith("?") else p + "?" for p in parts]

    # "X and Y" with long halves
    and_parts = re.split(r"\s+and\s+", q, flags=re.IGNORECASE)
    if len(and_parts) >= 2 and all(len(p.split()) >= 4 for p in and_parts):
        return [p.strip() for p in and_parts]

    return [q]


# ── Context formatting ─────────────────────────────────────────────────────────

def format_sections(chunks: list, sections_full: dict) -> str:
    """Build the sections block. Uses full text when available."""
    lines = []
    for c in chunks:
        m   = c["meta"]
        sid = m.get("section_id", "?")
        ttl = m.get("section_title", "")
        full = sections_full.get(sid, {}).get("content") or c["text"]
        lines.append(f"[{sid}] {ttl}\n{full}")
    return "\n\n".join(lines)


def format_recent(recent_qas: list) -> str:
    """Compact representation of last few Q&As."""
    if not recent_qas:
        return "(none)"
    lines = []
    for h in recent_qas:
        q = h["question"][:180]
        a = h["answer"][:280]
        lines.append(f"  Q: {q}\n  A: {a}")
    return "\n".join(lines)


# ── Main RAG entrypoint ────────────────────────────────────────────────────────

def rag_answer(question: str,
               *,
               coll, sections_full,
               client, model, disable_thinking,
               business_form: str = "",
               profile: str = "",
               summary: str = "",
               recent_qas: list = None,
               top_k: int = 4,
               stream: bool = False):
    """
    Run full pipeline. Returns dict with answer + sources + sub_queries.
    If stream=True, returns a generator yielding (kind, payload) tuples
    where kind ∈ {"chunk", "done"}.
    """
    recent_qas  = recent_qas or []
    sub_queries = split_query(question)
    chunks      = retrieve_all(sub_queries, coll, top_k=top_k, business_form=business_form)
    sources     = [c["meta"].get("section_id", "") for c in chunks]

    prompt = ANSWER_PROMPT.format(
        profile  = profile or "(not set)",
        summary  = summary or "(no prior conversation)",
        recent   = format_recent(recent_qas),
        sections = format_sections(chunks, sections_full),
        question = question,
    )

    if stream:
        def _gen():
            parts = []
            for piece in stream_call(client, model, prompt, disable_thinking=disable_thinking):
                parts.append(piece)
                yield ("chunk", piece)
            yield ("done", {
                "answer":      "".join(parts).strip(),
                "sources":     sources,
                "sub_queries": sub_queries,
                "n_chunks":    len(chunks),
            })
        return _gen()

    answer = llm_call(client, model, prompt, disable_thinking=disable_thinking)
    return {
        "answer":      answer,
        "sources":     sources,
        "sub_queries": sub_queries,
        "n_chunks":    len(chunks),
    }
