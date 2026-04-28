import pytest

from llmem.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db, disable_vec=True)
    yield s
    s.close()


@pytest.fixture
def diary_path(tmp_path):
    return tmp_path / "dream-diary.md"
