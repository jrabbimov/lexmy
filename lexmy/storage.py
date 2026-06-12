"""Turso (libSQL) storage layer — projects + Q&A history."""
import json
import os
import uuid

import libsql_experimental as libsql


# ── Client ────────────────────────────────────────────────────────────────────

def _raw_connect():
    url   = os.environ.get("TURSO_URL", "")
    token = os.environ.get("TURSO_TOKEN", "")
    if url:
        return libsql.connect(url, auth_token=token)
    return libsql.connect("file:lexmy.db")


class _Conn:
    """
    Wrapper that auto-reconnects when Turso closes an idle Hrana stream.
    Symptom we catch: ValueError 'Hrana: api error: status=404 ... stream not found'.
    """
    def __init__(self):
        self._raw = _raw_connect()

    def _is_dead_stream(self, err: Exception) -> bool:
        s = str(err)
        return "stream not found" in s or "stream closed" in s

    def execute(self, sql: str, params=()):
        try:
            return self._raw.execute(sql, params)
        except (ValueError, Exception) as e:
            if self._is_dead_stream(e):
                self._raw = _raw_connect()
                return self._raw.execute(sql, params)
            raise

    def commit(self):
        try:
            self._raw.commit()
        except Exception as e:
            if self._is_dead_stream(e):
                self._raw = _raw_connect()
                try:
                    self._raw.commit()
                except Exception:
                    pass


def make_client():
    """Reads TURSO_URL + TURSO_TOKEN from env. Local SQLite fallback for dev."""
    return _Conn()


def _exec(client, sql: str, params: list = None):
    """Execute and return cursor."""
    return client.execute(sql, tuple(params) if params else ())


def _commit(client):
    try:
        client.commit()
    except Exception:
        pass   # remote Turso auto-commits; ignore if not supported


def init_schema(client):
    """Create tables if missing. Safe to run every startup."""
    _exec(client, """
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
    _exec(client, """
        CREATE TABLE IF NOT EXISTS qa_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL,
            question    TEXT,
            answer      TEXT,
            sources     TEXT,
            planning    TEXT DEFAULT '',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration for DBs created before the planning column existed.
    try:
        _exec(client, "ALTER TABLE qa_history ADD COLUMN planning TEXT DEFAULT ''")
    except Exception:
        pass  # column already present
    _exec(client, "CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id)")
    _exec(client, "CREATE INDEX IF NOT EXISTS idx_qa_project    ON qa_history(project_id)")
    _commit(client)


# ── Project CRUD ──────────────────────────────────────────────────────────────

def list_projects(client, user_id: str) -> list:
    rows = _exec(client,
        "SELECT id, name, industry, business_form, summary, qa_count, created_at "
        "FROM projects WHERE user_id = ? ORDER BY created_at DESC",
        [user_id],
    ).fetchall()
    return [
        {
            "id": r[0], "name": r[1], "industry": r[2] or "",
            "business_form": r[3] or "", "summary": r[4] or "",
            "qa_count": r[5] or 0, "created_at": r[6],
        }
        for r in rows
    ]


def get_project(client, project_id: str) -> dict | None:
    rows = _exec(client,
        "SELECT id, user_id, name, industry, business_form, summary, qa_count, created_at "
        "FROM projects WHERE id = ?",
        [project_id],
    ).fetchall()
    if not rows:
        return None
    r = rows[0]
    return {
        "id": r[0], "user_id": r[1], "name": r[2], "industry": r[3] or "",
        "business_form": r[4] or "", "summary": r[5] or "",
        "qa_count": r[6] or 0, "created_at": r[7],
    }


def create_project(client, user_id: str, name: str,
                   industry: str = "", business_form: str = "") -> str:
    pid = str(uuid.uuid4())
    _exec(client,
        "INSERT INTO projects (id, user_id, name, industry, business_form) "
        "VALUES (?, ?, ?, ?, ?)",
        [pid, user_id, name, industry, business_form],
    )
    _commit(client)
    return pid


def update_project(client, project_id: str, **fields):
    """Update any subset of: name, industry, business_form, summary, qa_count."""
    allowed = {"name", "industry", "business_form", "summary", "qa_count"}
    cols    = [k for k in fields if k in allowed]
    if not cols:
        return
    sets = ", ".join(f"{c} = ?" for c in cols)
    vals = [fields[c] for c in cols] + [project_id]
    _exec(client, f"UPDATE projects SET {sets} WHERE id = ?", vals)
    _commit(client)


def delete_project(client, project_id: str):
    _exec(client, "DELETE FROM qa_history WHERE project_id = ?", [project_id])
    _exec(client, "DELETE FROM projects   WHERE id = ?",         [project_id])
    _commit(client)


# ── Q&A history ───────────────────────────────────────────────────────────────

def append_qa(client, project_id: str, question: str, answer: str, sources: list,
              planning: dict = None):
    _exec(client,
        "INSERT INTO qa_history (project_id, question, answer, sources, planning) "
        "VALUES (?, ?, ?, ?, ?)",
        [project_id, question, answer, json.dumps(sources), json.dumps(planning or {})],
    )
    _exec(client,
        "UPDATE projects SET qa_count = qa_count + 1 WHERE id = ?",
        [project_id],
    )
    _commit(client)


def delete_qa(client, project_id: str, qa_id: int):
    """Delete a single Q&A turn (e.g. a turn ruined by an exception during a demo)."""
    _exec(client, "DELETE FROM qa_history WHERE id = ? AND project_id = ?",
          [qa_id, project_id])
    _exec(client,
        "UPDATE projects SET qa_count = MAX(qa_count - 1, 0) WHERE id = ?",
        [project_id],
    )
    _commit(client)


def _qa_row(r: tuple) -> dict:
    return {
        "id": r[0], "question": r[1], "answer": r[2],
        "sources": json.loads(r[3] or "[]"),
        "planning": json.loads(r[4] or "{}"),
        "created_at": r[5],
    }


def list_qa(client, project_id: str, limit: int = 0) -> list:
    """Return Q&A history in chronological order (oldest first)."""
    sql = ("SELECT id, question, answer, sources, planning, created_at "
           "FROM qa_history WHERE project_id = ? ORDER BY id ASC")
    params = [project_id]
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    rows = _exec(client, sql, params).fetchall()
    return [_qa_row(r) for r in rows]


def last_qa(client, project_id: str, n: int) -> list:
    """Most recent n Q&As (in chronological order)."""
    rows = _exec(client,
        "SELECT id, question, answer, sources, planning, created_at "
        "FROM qa_history WHERE project_id = ? ORDER BY id DESC LIMIT ?",
        [project_id, n],
    ).fetchall()
    return list(reversed([_qa_row(r) for r in rows]))
