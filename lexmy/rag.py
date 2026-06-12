"""RAG orchestration: query split → retrieve → prompt → LLM."""
import re
import json

from .prompt    import ANSWER_PROMPT, QUERY_REWRITE_PROMPT
from .llm       import llm_call, stream_call
from .retrieval import retrieve_all


# ── Query splitter (rule-based fallback) ───────────────────────────────────────

def split_query(query: str) -> list:
    """Break complex questions into focused sub-queries. Returns ≥ 1 item.
    Pure-regex fallback used when the LLM router is unavailable or fails."""
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


# ── Query router (LLM: pass / split / rewrite / expand) ────────────────────────

def plan_queries(question: str, client, model, *,
                 recent_qas: list = None, disable_thinking: bool = False) -> tuple:
    """Use the LLM to turn the raw question + history into standalone search
    queries. Decides per-question whether to pass through, split, rewrite
    (resolve pronouns from history), or expand (broad → facet queries).
    Falls back to the regex splitter on any failure.

    Returns (queries, method) where method is:
      'generated'  – LLM router produced the queries (rewrite / split / expand)
      'unchanged'  – LLM router returned the question as-is (single query)
      'rule-split' – LLM failed; regex fallback split the question
    """
    prompt = QUERY_REWRITE_PROMPT.format(
        recent   = format_recent(recent_qas or []),
        question = question,
    )
    try:
        raw = llm_call(client, model, prompt,
                       system="You output only a JSON array of search-query strings.",
                       disable_thinking=disable_thinking,
                       max_tokens=200, temp=0.0, top_p=1.0)
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        queries = json.loads(m.group(0)) if m else []
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        # drop duplicates (case-insensitive), preserve order — weak models
        # sometimes repeat the same query several times.
        seen, deduped = set(), []
        for q in queries:
            k = q.lower()
            if k not in seen:
                seen.add(k)
                deduped.append(q)
        queries = deduped
        if queries:
            queries = queries[:4]
            # passed through unchanged vs actively rewritten/split/expanded
            unchanged = (len(queries) == 1
                         and queries[0].strip().rstrip("?").lower()
                             == question.strip().rstrip("?").lower())
            return queries, ("unchanged" if unchanged else "generated")
    except Exception:
        pass
    return split_query(question), "rule-split"


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
               top_k: int = 5,
               max_sections: int = 12,
               stream: bool = False):
    """
    Run full pipeline. Returns dict with answer + sources + sub_queries.
    If stream=True, returns a generator yielding (kind, payload) tuples
    where kind ∈ {"chunk", "done"}.
    """
    recent_qas  = recent_qas or []
    sub_queries, query_method = plan_queries(
        question, client, model,
        recent_qas=recent_qas, disable_thinking=disable_thinking)
    chunks      = retrieve_all(sub_queries, coll, top_k=top_k, business_form=business_form)
    chunks      = chunks[:max_sections]
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
            # NIM streaming is occasionally flaky: it can throw a JSON-decode
            # error mid-stream, or finish having emitted only hidden reasoning
            # (no visible content). Catch both and fall back to a non-streaming
            # call so the user still gets an answer.
            try:
                for piece in stream_call(client, model, prompt,
                                         disable_thinking=disable_thinking, max_tokens=1024):
                    parts.append(piece)
                    yield ("chunk", piece)
            except Exception:
                pass  # streaming failed — fall through to non-stream fallback
            answer = "".join(parts).strip()
            if not answer:
                try:
                    answer = llm_call(client, model, prompt,
                                      disable_thinking=disable_thinking, max_tokens=1024).strip()
                except Exception:
                    answer = ""
                if answer:
                    yield ("chunk", answer)
            yield ("done", {
                "answer":       answer,
                "sources":      sources,
                "sub_queries":  sub_queries,
                "query_method": query_method,
                "n_chunks":     len(chunks),
            })
        return _gen()

    answer = llm_call(client, model, prompt, disable_thinking=disable_thinking)
    return {
        "answer":       answer,
        "sources":      sources,
        "sub_queries":  sub_queries,
        "query_method": query_method,
        "n_chunks":     len(chunks),
    }
