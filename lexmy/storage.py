"""Turso (libSQL) storage layer — projects + Q&A history."""
import json
import os
import uuid
from datetime import datetime, timezone

import libsql_client


# ── Client ────────────────────────────────────────────────────────────────────

def make_client():
    """
    Returns a sync libsql client.
    Reads TURSO_URL + TURSO_TOKEN from env (or st.secrets — caller's job).
    Falls back to local SQLite file for dev.
    """
    url   = os.environ.get("TURSO_URL", "")
    token = os.environ.get("TURSO_TOKEN", "")
    if url:
        return libsql_client.create_client_sync(url=url, auth_token=token)
    # local fallback for dev
    return libsql_client.create_client_sync(url="file:lexmy.db")


def init_schema(client):
    """Create tables if missing. Safe to run every startup."""
    client.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            name          TEXT NOT NULL,
            industry      TEXT DEFAULT '',
            business_form TEXT DEFAULT '',
            summary       TEXT DEFAULT '',
            qa_count      INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    client.execute("""
        CREATE TABLE IF NOT EXISTS qa_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL,
            question    TEXT,
            answer      TEXT,
            sources     TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    client.execute("CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id)")
    client.execute("CREATE INDEX IF NOT EXISTS idx_qa_project    ON qa_history(project_id)")


# ── Project CRUD ──────────────────────────────────────────────────────────────

def list_projects(client, user_id: str) -> list:
    rs = client.execute(
        "SELECT id, name, industry, business_form, summary, qa_count, created_at "
        "FROM projects WHERE user_id = ? ORDER BY created_at DESC",
        [user_id],
    )
    return [
        {
            "id": r[0], "name": r[1], "industry": r[2] or "",
            "business_form": r[3] or "", "summary": r[4] or "",
            "qa_count": r[5] or 0, "created_at": r[6],
        }
        for r in rs.rows
    ]


def get_project(client, project_id: str) -> dict | None:
    rs = client.execute(
        "SELECT id, user_id, name, industry, business_form, summary, qa_count, created_at "
        "FROM projects WHERE id = ?",
        [project_id],
    )
    if not rs.rows:
        return None
    r = rs.rows[0]
    return {
        "id": r[0], "user_id": r[1], "name": r[2], "industry": r[3] or "",
        "business_form": r[4] or "", "summary": r[5] or "",
        "qa_count": r[6] or 0, "created_at": r[7],
    }


def create_project(client, user_id: str, name: str,
                   industry: str = "", business_form: str = "") -> str:
    pid = str(uuid.uuid4())
    client.execute(
        "INSERT INTO projects (id, user_id, name, industry, business_form) "
        "VALUES (?, ?, ?, ?, ?)",
        [pid, user_id, name, industry, business_form],
    )
    return pid


def update_project(client, project_id: str, **fields):
    """Update any subset of: name, industry, business_form, summary, qa_count."""
    allowed = {"name", "industry", "business_form", "summary", "qa_count"}
    cols    = [k for k in fields if k in allowed]
    if not cols:
        return
    sets = ", ".join(f"{c} = ?" for c in cols)
    vals = [fields[c] for c in cols] + [project_id]
    client.execute(f"UPDATE projects SET {sets} WHERE id = ?", vals)


def delete_project(client, project_id: str):
    client.execute("DELETE FROM qa_history WHERE project_id = ?", [project_id])
    client.execute("DELETE FROM projects   WHERE id = ?",         [project_id])


# ── Q&A history ───────────────────────────────────────────────────────────────

def append_qa(client, project_id: str, question: str, answer: str, sources: list):
    client.execute(
        "INSERT INTO qa_history (project_id, question, answer, sources) VALUES (?, ?, ?, ?)",
        [project_id, question, answer, json.dumps(sources)],
    )
    client.execute(
        "UPDATE projects SET qa_count = qa_count + 1 WHERE id = ?",
        [project_id],
    )


def list_qa(client, project_id: str, limit: int = 0) -> list:
    """Return Q&A history in chronological order (oldest first)."""
    sql = ("SELECT id, question, answer, sources, created_at "
           "FROM qa_history WHERE project_id = ? ORDER BY id ASC")
    params = [project_id]
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    rs = client.execute(sql, params)
    return [
        {
            "id": r[0], "question": r[1], "answer": r[2],
            "sources": json.loads(r[3] or "[]"), "created_at": r[4],
        }
        for r in rs.rows
    ]


def last_qa(client, project_id: str, n: int) -> list:
    """Most recent n Q&As (in chronological order)."""
    rs = client.execute(
        "SELECT id, question, answer, sources, created_at "
        "FROM qa_history WHERE project_id = ? ORDER BY id DESC LIMIT ?",
        [project_id, n],
    )
    rows = [
        {
            "id": r[0], "question": r[1], "answer": r[2],
            "sources": json.loads(r[3] or "[]"), "created_at": r[4],
        }
        for r in rs.rows
    ]
    return list(reversed(rows))
