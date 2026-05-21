# LexMY — UI Implementation Plan

Plan for turning the working RAG (in `1-build.ipynb` + `2-test.ipynb`) into a deployable web app.

---

## Decisions (confirmed)

| # | Choice | Pick | Why |
|---|--------|------|-----|
| 1 | UI framework | **Streamlit** | Sidebar + multi-section layout fits proposal |
| 2 | Storage | **Turso** (cloud SQLite, free 9 GB) | Survives container restarts on Streamlit Cloud |
| 3 | User identity | **Browser UUID via cookie** (`streamlit-cookies-manager`) | No login, survives refresh |
| 4 | LLM backend | **Toggle** in sidebar (NIM cloud / LM Studio local) | Matches proposal objective 5 |
| 5 | Embedding load | `@st.cache_resource` | Skip 30s cold start per reload |
| 6 | Profile fields | **Business name + industry + business form** only | Minimal, no extra fields |

---

## File layout

```
code/
├── app.py                  # Streamlit entry point
├── lexmy/
│   ├── __init__.py
│   ├── retrieval.py        # vector search + graph (from 2-test cell_retriever)
│   ├── rag.py              # split_query, format_context, rag() (from 2-test cell_rag)
│   ├── llm.py              # llm_call with backend toggle (from 2-test cell_llm)
│   ├── memory.py           # profile + history + rolling summary
│   ├── storage.py          # Turso client + CRUD
│   └── prompt.py           # ANSWER_PROMPT
├── artifacts/              # chroma/, graph.pkl, concept_vocab.json, sections_full.json
├── pyproject.toml          # uv-managed dependencies
├── requirements.txt        # mirror of deps (Streamlit Cloud reads this)
└── .streamlit/
    └── secrets.toml        # NIM_API_KEY, TURSO_URL, TURSO_TOKEN, COOKIE_PASSWORD
```

---

## Storage — Turso (cloud SQLite)

Setup (one-time, ~10 min):
1. `turso db create lexmy`
2. `turso db show lexmy --url`              → libSQL URL
3. `turso db tokens create lexmy`           → auth token
4. Store both as Streamlit secrets.

Client (in `storage.py`):

```python
import libsql_client

def get_client():
    return libsql_client.create_client_sync(
        url   = st.secrets["TURSO_URL"],
        auth_token = st.secrets["TURSO_TOKEN"],
    )
```

Schema (run once via `turso db shell lexmy`):

```sql
CREATE TABLE projects (
    id            TEXT PRIMARY KEY,        -- uuid
    user_id       TEXT NOT NULL,           -- browser uuid
    name          TEXT NOT NULL,
    industry      TEXT,
    business_form TEXT,                    -- 'private_sdn_bhd' | 'llp' | 'sole_prop'
    summary       TEXT DEFAULT '',         -- rolling summary
    qa_count      INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE qa_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    question    TEXT,
    answer      TEXT,
    sources     TEXT,                       -- JSON array of section IDs
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_user ON projects(user_id);
CREATE INDEX idx_qa_project    ON qa_history(project_id);
```

Same SQL as local SQLite — only the connection changes.

---

## Memory module (`memory.py`)

```python
def build_prompt_context(project, recent_n=3):
    return {
        'profile':  project.profile_string(),
        'summary':  project.summary,
        'recent':   project.last_qas(recent_n),
    }

def maybe_update_summary(project, llm):
    if project.qa_count % 3 != 0:
        return
    project.summary = llm.summarise(project.all_qas())
    storage.save_project(project)
```

Summary prompt: *"Summarise the conversation so far in 5 sentences. Cover: facts about the user's business, main concerns, unresolved questions."*

---

## Retrieval bias (profile-aware)

After Chroma query, re-rank so sections from acts matching business form get a score boost:

```python
ACT_BIAS = {
    'private_sdn_bhd': {'act777': 1.15, 'pdpa': 1.05},
    'llp':             {'llp':    1.15, 'pdpa': 1.05},
    'sole_prop':       {'roba197': 1.15, 'pdpa': 1.05},
}
```

Multiplier on similarity score. Never excludes other acts (cross-act questions still work).

---

## Page layout

```
┌────────────────────────────────────────────────────┐
│ Sidebar              │  Main                       │
│ ─────────────────    │ ──────────────────────────  │
│ [+ New project]      │  Project: Apex Fintech      │
│ ▸ Apex Fintech       │  ┌────────────────────────┐ │
│ ▸ Café X             │  │ Industry: fintech       │ │
│                      │  │ Form: private (Sdn Bhd) │ │
│                      │  │ [Edit]                  │ │
│ ─────────────────    │  └────────────────────────┘ │
│ LLM: ▼ NIM           │                             │
│      LM Studio       │  ── Chat ──                 │
│                      │  user: Can I share...?      │
│                      │  LexMY:                     │
│                      │    ✅ What law says...       │
│                      │    ⚠️ Obligations...         │
│                      │    💰 Penalties...           │
│                      │    📌 Action...              │
│                      │    ▸ Sources (3)             │
│                      │  [Type question ........] [→]│
└────────────────────────────────────────────────────┘
```

Streamlit primitives: `st.sidebar`, `st.chat_message`, `st.chat_input`, `st.expander`.

---

## Q&A flow

```
on_submit(question):
    project   = current_project()
    ctx       = build_prompt_context(project, recent_n=3)
    sub_qs    = split_query(question)
    chunks    = retrieve_all(sub_qs, profile_bias=project.business_form)
    sections  = [SECTIONS_FULL[c.section_id] for c in chunks]
    prompt    = ANSWER_PROMPT.format(
                   profile=ctx['profile'],
                   summary=ctx['summary'],
                   recent=ctx['recent'],
                   sections=sections,
                   question=question)
    answer    = llm.stream(prompt)
    project.append_qa(question, answer, sources=[c.section_id for c in chunks])
    maybe_update_summary(project, llm)
    storage.save_project(project)
```

---

## Implementation phases

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1 | 1   | Lift RAG code from notebooks into `lexmy/*.py`; smoke-test |
| 2 | 1   | Turso schema + project CRUD + browser UUID cookie |
| 3 | 1   | Streamlit shell (sidebar, profile card, chat) |
| 4 | 0.5 | Wire chat → RAG → store Q&A |
| 5 | 0.5 | Rolling summary + memory context in prompts |
| 6 | 0.5 | Profile-bias retrieval |
| 7 | 0.5 | LLM backend toggle |
| 8 | 0.5 | Deploy to Streamlit Community Cloud |

**Total: ~5 days for one dev.**

---

## Running locally (uv)

All commands use **uv** — single binary, no manual venv activation. Install once: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
cd code
uv sync                              # install deps from pyproject.toml
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml — fill NIM_API_KEY (optional), TURSO_URL+TOKEN (optional), COOKIE_PASSWORD
uv run streamlit run app.py
```

Add a package later: `uv add <pkg>`. Run any script: `uv run <cmd>`.

---

## Deployment

**Streamlit Community Cloud** (free): direct GitHub deploy, secrets via dashboard for `NIM_API_KEY`, `TURSO_URL`, `TURSO_TOKEN`, `COOKIE_PASSWORD`. Streamlit Cloud reads `requirements.txt` automatically.

Artifacts (chroma, graph.pkl, sections_full.json, concept_vocab.json) bundled in repo, ~17 MB.
