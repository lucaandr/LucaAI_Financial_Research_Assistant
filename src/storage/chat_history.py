"""
Persistent chat history storage, backed by a local SQLite file.

Kept as a small, dependency-free module (sqlite3 is in the Python standard
library) so conversations survive app restarts/rebuilds — unlike
st.session_state, which only lives for the duration of the Streamlit process.
"""
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join("data", "chat_history.db")


@contextmanager
def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates the tables if they don't exist yet. Safe to call on every app start."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)


def create_conversation(title: str) -> str:
    """Creates a new conversation and returns its id."""
    conversation_id = str(uuid.uuid4())
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conversation_id, title, datetime.now(timezone.utc).isoformat()),
        )
    return conversation_id


def add_message(conversation_id: str, role: str, content: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
        )


def list_conversations() -> list[dict]:
    """Returns all conversations, most recent first."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_messages(conversation_id: str) -> list[dict]:
    """Returns [{'role': ..., 'content': ...}, ...] in chronological order."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?",
                     (conversation_id,))


def make_title(first_user_message: str, max_len: int = 40) -> str:
    """Derives a short conversation title from the first user message."""
    title = first_user_message.strip().replace("\n", " ")
    return title[:max_len] + ("..." if len(title) > max_len else "")
