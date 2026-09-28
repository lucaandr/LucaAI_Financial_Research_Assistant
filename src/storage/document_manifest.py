"""
Tracks which PDFs have already been ingested into ChromaDB, so re-running
ingestion doesn't silently duplicate chunks for files processed before.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join("data", "document_manifest.db")


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
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested_documents (
                filename TEXT PRIMARY KEY,
                chunk_count INTEGER NOT NULL,
                ingested_at TEXT NOT NULL
            )
        """)


def is_ingested(filename: str) -> bool:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM ingested_documents WHERE filename = ?", (filename,)
        ).fetchone()
        return row is not None


def mark_ingested(filename: str, chunk_count: int) -> None:
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ingested_documents (filename, chunk_count, ingested_at) "
            "VALUES (?, ?, ?)",
            (filename, chunk_count, datetime.now(timezone.utc).isoformat()),
        )


def list_ingested() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT filename, chunk_count, ingested_at FROM ingested_documents "
            "ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def forget_document(filename: str) -> None:
    """Removes the manifest entry only — does NOT remove the document's
    embeddings from ChromaDB. Use together with a vector-store delete if you
    also want the chunks gone."""
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM ingested_documents WHERE filename = ?", (filename,))
