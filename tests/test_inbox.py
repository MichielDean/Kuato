"""Tests for the working memory inbox feature — add, get, list, remove, consolidate."""


import pytest

from llmem.store import MemoryStore


class TestInbox_Add_ReturnsId:
    """add_to_inbox returns a UUID string."""

    def test_add_returns_id(self, store):
        inbox_id = store.add_to_inbox(content="test note")
        assert inbox_id is not None
        assert len(inbox_id) == 36
        assert "-" in inbox_id


class TestInbox_AddWithSourceAndScore:
    """add_to_inbox stores source and attention_score correctly."""

    def test_add_with_source_and_score(self, store):
        inbox_id = store.add_to_inbox(
            content="test", source="learn", attention_score=0.8
        )
        item = store.get_from_inbox(inbox_id)
        assert item is not None
        assert item["source"] == "learn"
        assert item["attention_score"] == 0.8


class TestInbox_DefaultSourceIsNote:
    """add_to_inbox defaults source to 'note'."""

    def test_default_source_is_note(self, store):
        inbox_id = store.add_to_inbox(content="test")
        item = store.get_from_inbox(inbox_id)
        assert item["source"] == "note"


class TestInbox_DefaultAttentionScore:
    """add_to_inbox defaults attention_score to 0.5."""

    def test_default_attention_score(self, store):
        inbox_id = store.add_to_inbox(content="test")
        item = store.get_from_inbox(inbox_id)
        assert item["attention_score"] == 0.5


class TestInbox_AddInvalidSourceRaises:
    """add_to_inbox raises ValueError for an invalid source."""

    def test_invalid_source_raises(self, store):
        with pytest.raises(ValueError, match="invalid source"):
            store.add_to_inbox(content="test", source="invalid_source")


class TestInbox_AddInvalidAttentionScoreRaises:
    """add_to_inbox raises ValueError for attention_score > 1.0."""

    def test_invalid_attention_score_raises(self, store):
        with pytest.raises(ValueError, match="attention_score"):
            store.add_to_inbox(content="test", attention_score=1.5)


class TestInbox_AddNegativeAttentionScoreRaises:
    """add_to_inbox raises ValueError for negative attention_score."""

    def test_negative_attention_score_raises(self, store):
        with pytest.raises(ValueError, match="attention_score"):
            store.add_to_inbox(content="test", attention_score=-0.1)


class TestInbox_GetNonexistentReturnsNone:
    """get_from_inbox returns None for a nonexistent ID."""

    def test_get_nonexistent_returns_none(self, store):
        result = store.get_from_inbox("nonexistent-id")
        assert result is None


class TestInbox_ListReturnsOrderedItems:
    """list_inbox returns items ordered by attention_score DESC, created_at ASC."""

    def test_list_returns_ordered_items(self, store):
        store.add_to_inbox(content="low score", attention_score=0.3)
        store.add_to_inbox(content="high score", attention_score=0.9)
        store.add_to_inbox(content="mid score", attention_score=0.6)
        items = store.list_inbox()
        assert len(items) == 3
        # Ordered by attention_score DESC
        assert items[0]["attention_score"] >= items[1]["attention_score"]
        assert items[1]["attention_score"] >= items[2]["attention_score"]


class TestInbox_CapacityEvictsLowestScore:
    """When inbox reaches capacity, the lowest-scored item is evicted."""

    def test_capacity_evicts_lowest_score(self, tmp_path):
        db = tmp_path / "test.db"
        # capacity=3, fill it up
        store = MemoryStore(db_path=db, disable_vec=True, capacity=3)
        store.add_to_inbox(content="first", attention_score=0.5)
        store.add_to_inbox(content="second", attention_score=0.7)
        store.add_to_inbox(content="third", attention_score=0.9)
        # Adding a 4th item should evict the lowest score (first, 0.5)
        store.add_to_inbox(content="fourth", attention_score=0.6)
        items = store.list_inbox()
        assert len(items) == 3
        scores = [item["attention_score"] for item in items]
        assert 0.5 not in scores
        store.close()


class TestInbox_CapacityCustom:
    """Inbox capacity can be customized via constructor."""

    def test_capacity_custom(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True, capacity=2)
        store.add_to_inbox(content="first", attention_score=0.5)
        store.add_to_inbox(content="second", attention_score=0.7)
        # Adding a 3rd should evict lowest
        store.add_to_inbox(content="third", attention_score=0.9)
        assert store.inbox_count() == 2
        store.close()


class TestInbox_EvictionTiebreakCreatedAt:
    """When two items have the same attention_score, the earlier one is evicted."""

    def test_eviction_tiebreak_created_at(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True, capacity=2)
        first_id = store.add_to_inbox(content="first", attention_score=0.5)
        second_id = store.add_to_inbox(content="second", attention_score=0.5)
        # Both at 0.5, adding a higher score evicts the earliest (first)
        store.add_to_inbox(content="third", attention_score=0.9)
        items = store.list_inbox()
        ids = [item["id"] for item in items]
        assert first_id not in ids
        assert second_id in ids
        store.close()


class TestInbox_Remove:
    """remove_from_inbox deletes an item and returns True."""

    def test_remove(self, store):
        inbox_id = store.add_to_inbox(content="to remove")
        assert store.remove_from_inbox(inbox_id) is True
        assert store.get_from_inbox(inbox_id) is None


class TestInbox_RemoveNonexistentReturnsFalse:
    """remove_from_inbox returns False for a nonexistent ID."""

    def test_remove_nonexistent_returns_false(self, store):
        assert store.remove_from_inbox("nonexistent") is False


class TestInbox_Count:
    """inbox_count returns the correct number of items."""

    def test_inbox_count(self, store):
        assert store.inbox_count() == 0
        store.add_to_inbox(content="one")
        assert store.inbox_count() == 1
        store.add_to_inbox(content="two")
        assert store.inbox_count() == 2


class TestConsolidate_PromotesToMemory:
    """consolidate() moves inbox items to memories table."""

    def test_consolidate_promotes_to_memory(self, store):
        inbox_id = store.add_to_inbox(
            content="remember this", source="note", attention_score=0.7
        )
        result = store.consolidate(min_score=0.0)
        assert len(result["promoted"]) == 1
        assert len(result["evicted"]) == 0
        promoted_item = result["promoted"][0]
        assert promoted_item["id"] == inbox_id
        assert "memory_id" in promoted_item
        # Verify the memory exists in the memories table
        mem = store.get(promoted_item["memory_id"])
        assert mem is not None
        assert mem["content"] == "remember this"
        assert mem["source"] == "consolidation"
        assert abs(mem["confidence"] - 0.7) < 1e-6


class TestConsolidate_DryRun:
    """consolidate(dry_run=True) returns the same plan without making changes."""

    def test_consolidate_dry_run(self, store):
        store.add_to_inbox(content="note 1", attention_score=0.7)
        store.add_to_inbox(content="note 2", attention_score=0.3)
        result = store.consolidate(min_score=0.5, dry_run=True)
        assert len(result["promoted"]) == 1
        assert len(result["evicted"]) == 1
        # Inbox should still have both items (dry run makes no changes)
        assert store.inbox_count() == 2

    def test_consolidate_dry_run_no_memory_inserts(self, store):
        """dry_run=True must NOT create entries in the memories table."""
        initial_count = store.count()
        store.add_to_inbox(content="will not promote", attention_score=0.8)
        store.add_to_inbox(content="will not evict", attention_score=0.2)
        result = store.consolidate(min_score=0.5, dry_run=True)
        assert len(result["promoted"]) == 1
        assert len(result["evicted"]) == 1
        # No new memories should have been created
        assert store.count() == initial_count
        # Inbox unchanged
        assert store.inbox_count() == 2

    def test_consolidate_dry_run_no_memory_id(self, store):
        """dry_run=True must not assign memory_id since no insertion happened."""
        store.add_to_inbox(content="ephemeral", attention_score=0.9)
        result = store.consolidate(min_score=0.5, dry_run=True)
        assert len(result["promoted"]) == 1
        assert "memory_id" not in result["promoted"][0]


class TestConsolidate_ClearsInboxAfterPromotion:
    """After consolidate(), the inbox is empty."""

    def test_consolidate_clears_inbox_after_promotion(self, store):
        store.add_to_inbox(content="note 1", attention_score=0.8)
        store.add_to_inbox(content="note 2", attention_score=0.6)
        store.consolidate(min_score=0.0)
        assert store.inbox_count() == 0


class TestConsolidate_WithMinScore:
    """consolidate(min_score=0.7) only promotes items >= 0.7 and evicts below."""

    def test_consolidate_with_min_score(self, store):
        store.add_to_inbox(content="high note", attention_score=0.8)
        store.add_to_inbox(content="low note", attention_score=0.3)
        result = store.consolidate(min_score=0.7)
        assert len(result["promoted"]) == 1
        assert len(result["evicted"]) == 1
        assert result["promoted"][0]["attention_score"] >= 0.7
        assert result["evicted"][0]["attention_score"] < 0.7
        assert store.inbox_count() == 0


class TestInbox_UpdateAttentionScore:
    """update_inbox_attention_score updates the score of an existing item."""

    def test_update_attention_score(self, store):
        inbox_id = store.add_to_inbox(content="test", attention_score=0.5)
        assert store.update_inbox_attention_score(inbox_id, 0.9) is True
        item = store.get_from_inbox(inbox_id)
        assert abs(item["attention_score"] - 0.9) < 1e-6


class TestSchema_InboxTableExists:
    """After migrations, the inbox table exists with expected columns."""

    def test_schema_inbox_table_exists(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        # Check the inbox table exists
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='inbox'"
        ).fetchone()
        assert result is not None, "inbox table should exist"
        # Check columns
        columns = conn.execute('PRAGMA table_info("inbox")').fetchall()
        col_names = {col[1] for col in columns}
        assert "id" in col_names
        assert "content" in col_names
        assert "source" in col_names
        assert "attention_score" in col_names
        assert "created_at" in col_names
        assert "metadata" in col_names
        store.close()


class TestSchema_InboxTableQuotedIdentifiers:
    """The inbox table SQL uses double-quoted identifiers."""

    def test_inbox_table_quoted_identifiers(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        # Verify the table sql uses double-quoted identifiers by checking
        # the PRAGMA result contains quoted column definitions
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='inbox'"
        ).fetchone()
        assert sql is not None
        sql_text = sql[0]
        # The SQL should use double-quoted identifiers
        assert '"id"' in sql_text
        assert '"content"' in sql_text
        assert '"source"' in sql_text
        assert '"attention_score"' in sql_text
        store.close()


class TestInbox_UpdateAttentionScoreInvalidRange:
    """update_inbox_attention_score raises ValueError for out-of-range scores."""

    def test_update_attention_score_above_range(self, store):
        inbox_id = store.add_to_inbox(content="test")
        with pytest.raises(ValueError, match="attention_score"):
            store.update_inbox_attention_score(inbox_id, 1.5)

    def test_update_attention_score_below_range(self, store):
        inbox_id = store.add_to_inbox(content="test")
        with pytest.raises(ValueError, match="attention_score"):
            store.update_inbox_attention_score(inbox_id, -0.1)


class TestInbox_UpdateAttentionScoreNonexistent:
    """update_inbox_attention_score returns False for nonexistent ID."""

    def test_update_attention_score_nonexistent(self, store):
        assert store.update_inbox_attention_score("nonexistent", 0.5) is False


class TestInbox_MetadataParsedCorrectly:
    """Metadata stored as dict is parsed back correctly by get_from_inbox."""

    def test_metadata_parsed_correctly(self, store):
        inbox_id = store.add_to_inbox(
            content="test", metadata={"key": "value", "num": 42}
        )
        item = store.get_from_inbox(inbox_id)
        assert item["metadata"] == {"key": "value", "num": 42}


class TestInbox_MetadataDefaultEmptyDict:
    """Default metadata is an empty dict, not null."""

    def test_metadata_default_empty_dict(self, store):
        inbox_id = store.add_to_inbox(content="test")
        item = store.get_from_inbox(inbox_id)
        assert item["metadata"] == {}


class TestInbox_ListEmptyReturnsEmptyList:
    """list_inbox returns an empty list when inbox is empty."""

    def test_list_empty_returns_empty_list(self, store):
        items = store.list_inbox()
        assert items == []


class TestInbox_ConsolidateEmptyInbox:
    """consolidate() on empty inbox returns empty promoted/evicted lists."""

    def test_consolidate_empty_inbox(self, store):
        result = store.consolidate()
        assert result["promoted"] == []
        assert result["evicted"] == []


class TestInbox_ValidSources:
    """All valid sources are accepted by add_to_inbox."""

    @pytest.mark.parametrize("source", ["note", "learn", "extract", "consolidation"])
    def test_valid_sources(self, store, source):
        inbox_id = store.add_to_inbox(content=f"test {source}", source=source)
        item = store.get_from_inbox(inbox_id)
        assert item["source"] == source


class TestInbox_UpdateAttentionScoreNonexistentReturnsFalse:
    """update_inbox_attention_score returns False for nonexistent ID."""

    def test_update_nonexistent_returns_false(self, store):
        result = store.update_inbox_attention_score("nonexistent-id", 0.5)
        assert result is False
