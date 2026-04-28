import sqlite3

import pytest

from llmem.store import MemoryStore
from llmem.adapters.opencode import _create_opencode_schema


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db, disable_vec=True)
    yield s
    s.close()


@pytest.fixture
def diary_path(tmp_path):
    return tmp_path / "dream-diary.md"


@pytest.fixture
def opencode_db(tmp_path):
    """Create a temp opencode database with schema and sample data.

    Yields the Path to the database file. The caller is responsible for
    inserting sessions/messages/parts as needed for their test.
    """
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db_path))
    _create_opencode_schema(conn)
    conn.commit()
    conn.close()
    return db_path
