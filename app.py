"""LexMY — Streamlit web app."""
import logging
import os
import re
import sys
import uuid

# Suppress transformers __path__ alias warnings (triggered by Streamlit's watcher)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from lexmy import storage, memory, retrieval
from lexmy.llm import make_client
from lexmy.rag import rag_answer


# ── Mode: --cloud flag ────────────────────────────────────────────────────────
# Run locally:  uv run streamlit run app.py
# Run cloud:    uv run streamlit run app.py -- --cloud

CLOUD_MODE = "--cloud" in sys.argv


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LexMY — Malaysian Legal Assistant",
                   page_icon="⚖️", layout="wide")


# ── Secrets / env ─────────────────────────────────────────────────────────────

def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.environ.get(key, default)


if CLOUD_MODE:
    # Push cloud secrets into env so libsql_client + OpenAI picks them up
    for k in ("TURSO_URL", "TURSO_TOKEN", "NIM_API_KEY", "COOKIE_PASSWORD"):
        v = _secret(k)
        if v:
            os.environ[k] = v


# ── Cookies (browser UUID for "user" identity, no login) ──────────────────────

_cookie_pw = os.environ.get("COOKIE_PASSWORD", "lexmy-local-dev-only-key-32chars!")
cookies = EncryptedCookieManager(prefix="lexmy/", password=_cookie_pw)
if not cookies.ready():
    st.stop()

if "user_id" not in cookies or not cookies["user_id"]:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()
user_id = cookies["user_id"]


# ── Cached resources (load artifacts once per session) ────────────────────────

@st.cache_resource(show_spinner="Loading legal knowledge base…")
def load_artifacts():
    return {
        "coll":          retrieval.load_chroma(),
        "graph":         retrieval.load_graph(),
        "vocab":         retrieval.load_concept_vocab(),
        "sections_full": retrieval.load_sections_full(),
    }


@st.cache_resource
def get_db():
    c = storage.make_client()
    storage.init_schema(c)
    return c


ART = load_artifacts()
DB  = get_db()


# ── Sidebar: project list + LLM toggle ────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ LexMY")
    st.caption("Malaysian legal assistant")
    st.divider()

    if st.button("➕ New project", use_container_width=True):
        st.session_state["new_project"] = True
        st.session_state["active_project_id"] = None

    projects = storage.list_projects(DB, user_id)

    if projects:
        labels = {p["id"]: p["name"] for p in projects}
        current = st.session_state.get("active_project_id") or projects[0]["id"]
        chosen = st.radio(
            "Projects",
            list(labels.keys()),
            format_func=lambda pid: labels[pid],
            index=list(labels.keys()).index(current) if current in labels else 0,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.get("active_project_id"):
            st.session_state["active_project_id"] = chosen
            st.session_state["new_project"] = False
    else:
        st.info("No projects yet. Create one to start.")

    st.divider()
    if CLOUD_MODE:
        backend = st.selectbox(
            "Language model",
            ["nim", "lmstudio"],
            format_func=lambda x: {"nim": "NIM (cloud)", "lmstudio": "LM Studio (local)"}[x],
            index=0,
        )
    else:
        backend = "lmstudio"
        st.caption("🖥️ Local mode — LM Studio")


# ── LLM client (rebuilt when backend changes) ─────────────────────────────────

@st.cache_resource
def _llm_client(backend_name: str):
    api_key = os.environ.get("NIM_API_KEY", "")
    return make_client(backend_name, api_key=api_key)

llm_client, llm_model, disable_thinking = _llm_client(backend)


# ── Main: project create form ─────────────────────────────────────────────────

if st.session_state.get("new_project") or not projects:
    st.markdown("### Create a new project")
    with st.form("new_project_form", clear_on_submit=True):
        p_name = st.text_input("Business name", placeholder="e.g. Apex Fintech Sdn Bhd")
        submitted = st.form_submit_button("Create")
        if submitted:
            if not p_name.strip():
                st.error("Business name is required.")
            else:
                pid = storage.create_project(DB, user_id, p_name.strip())
                st.session_state["active_project_id"] = pid
                st.session_state["new_project"] = False
                st.rerun()
    st.stop()


# ── Main: active project view ─────────────────────────────────────────────────

active_id = st.session_state.get("active_project_id") or (projects[0]["id"] if projects else None)
if not active_id:
    st.stop()

project = storage.get_project(DB, active_id)
if project is None:
    st.error("Project not found.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def highlight_citations(text: str) -> str:
    """Bold every [section_id] citation so it stands out in the answer."""
    return re.sub(r'\[([a-z0-9_#.]+)\]', r'**[\1]**', text, flags=re.IGNORECASE)


def render_sources(source_ids: list):
    """Show all provided sections with full text in expanders."""
    if not source_ids:
        return
    with st.expander(f"📄 Provided sections ({len(source_ids)})"):
        for sid in source_ids:
            sect  = ART["sections_full"].get(sid, {})
            title = sect.get("section_title", "") or sid
            body  = sect.get("content", "*(full text not available)*")
            with st.expander(f"[{sid}] {title}"):
                st.markdown(body)


# Profile card (editable)
with st.container(border=True):
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(f"### {project['name']}")
        st.caption(f"Q&A turns: **{project['qa_count']}**")
    with c2:
        if st.button("Edit", use_container_width=True):
            st.session_state["edit_profile"] = not st.session_state.get("edit_profile", False)
        if st.button("Delete", use_container_width=True):
            storage.delete_project(DB, project["id"])
            st.session_state["active_project_id"] = None
            st.rerun()

    if st.session_state.get("edit_profile"):
        with st.form("edit_profile_form"):
            new_name = st.text_input("Business name", value=project["name"])
            if st.form_submit_button("Save"):
                storage.update_project(DB, project["id"], name=new_name.strip())
                st.session_state["edit_profile"] = False
                st.rerun()


# Chat history
history = storage.list_qa(DB, project["id"])
for h in history:
    with st.chat_message("user"):
        st.markdown(h["question"])
    with st.chat_message("assistant"):
        st.markdown(highlight_citations(h["answer"]))
        render_sources(h["sources"])


# Chat input
question = st.chat_input("Ask a legal question…")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        parts, final = [], None
        try:
            gen = rag_answer(
                question,
                coll=ART["coll"], graph=ART["graph"], vocab=ART["vocab"],
                sections_full=ART["sections_full"],
                client=llm_client, model=llm_model, disable_thinking=disable_thinking,
                business_form=project["business_form"],
                profile=memory.profile_string(project),
                summary=project["summary"],
                recent_qas=storage.last_qa(DB, project["id"], 3),
                stream=True,
            )
            for kind, payload in gen:
                if kind == "chunk":
                    parts.append(payload)
                    placeholder.markdown("".join(parts) + " ▌")
                elif kind == "done":
                    final = payload
            placeholder.markdown(highlight_citations(final["answer"]))
        except Exception as e:
            placeholder.error(f"LLM error: {e}")
            st.stop()

        render_sources(final["sources"])

    # Persist + maybe summarise
    storage.append_qa(DB, project["id"], question, final["answer"], final["sources"])
    project = storage.get_project(DB, project["id"])   # refresh qa_count
    memory.maybe_update_summary(DB, project, llm_client, llm_model, disable_thinking)
    st.rerun()
