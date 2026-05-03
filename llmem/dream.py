"""Background dreaming/consolidation pass — automated memory maintenance.

Three phases:
  Light — sort + dedupe: find near-duplicates using cosine similarity
  Deep  — decay + boost + merge + promote: idle decay using created_at
          (not updated_at), frequent-access boost, near-duplicate merge,
          inbox promotion, auto-linking
  REM   — reflect + cluster: extract themes, behavioral insights,
          proposed procedural memories
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

from .store import MemoryStore
from .paths import get_dream_diary_path, _is_blocked_path
from .registry import get_registered_dream_hooks
from .taxonomy import ERROR_TAXONOMY

log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_DECAY_RATE = 0.05
DEFAULT_DECAY_INTERVAL_DAYS = 30
DEFAULT_DECAY_FLOOR = 0.3
DEFAULT_CONFIDENCE_FLOOR = 0.3
DEFAULT_BOOST_THRESHOLD = 5
DEFAULT_BOOST_AMOUNT = 0.05
DEFAULT_BOOST_ON_PROMOTE = 0.1
DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_RECALL_COUNT = 3
DEFAULT_MIN_UNIQUE_QUERIES = 1
DEFAULT_MERGE_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_BEHAVIORAL_THRESHOLD = 3
DEFAULT_BEHAVIORAL_LOOKBACK_DAYS = 30
DEFAULT_CALIBRATION_ENABLED = True
DEFAULT_STALE_PROCEDURE_DAYS = 30
DEFAULT_CALIBRATION_LOOKBACK_DAYS = 90
DEFAULT_AUTO_LINK_THRESHOLD = 0.85


def _validate_output_path(path: Path, label: str) -> Path:
    """Validate that an output path is safe for writing.

    Checks:
    - Must not contain '..' traversal components
    - Must not target protected system directories
    - Must not be a symlink itself

    Does NOT require the path to be within the llmem home directory —
    users may configure custom output paths. This function prevents
    clearly dangerous writes, not all writes outside the default home.

    Args:
        path: The candidate output path.
        label: Description of the file (for error messages).

    Returns:
        The resolved path.

    Raises:
        ValueError: If the path is unsafe.
    """
    # Check traversal BEFORE resolving (resolve eliminates ..)
    if ".." in str(path):
        raise ValueError(f"{label} path contains '..' traversal: {path}")

    resolved = path.resolve()

    # Block system directories using shared helper (prefix + '/' matching)
    if _is_blocked_path(resolved):
        raise ValueError(f"{label} path targets a protected directory: {resolved}")
    if path.is_symlink():
        raise ValueError(
            f"{label} path is a symlink (not allowed for write targets): {path}"
        )
    return resolved


@dataclass
class LightPhaseResult:
    duplicate_pairs: int = 0
    merge_candidates: list[dict] = field(default_factory=list)


@dataclass
class DeepPhaseResult:
    decayed_count: int = 0
    boosted_count: int = 0
    promoted_count: int = 0
    invalidated_count: int = 0
    merged_count: int = 0
    auto_linked_count: int = 0
    decay_details: list[dict] = field(default_factory=list)
    boost_details: list[dict] = field(default_factory=list)
    promote_details: list[dict] = field(default_factory=list)
    invalidate_details: list[dict] = field(default_factory=list)
    merge_details: list[dict] = field(default_factory=list)


@dataclass
class RemPhaseResult:
    total_memories: int = 0
    active_memories: int = 0
    themes: list[str] = field(default_factory=list)
    behavioral_insights: list[dict] = field(default_factory=list)


@dataclass
class DreamResult:
    light: LightPhaseResult | None = None
    deep: DeepPhaseResult | None = None
    rem: RemPhaseResult | None = None


class Dreamer:
    """Automated memory consolidation (dreaming)."""

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        decay_rate: float = DEFAULT_DECAY_RATE,
        decay_interval_days: int = DEFAULT_DECAY_INTERVAL_DAYS,
        decay_floor: float = DEFAULT_DECAY_FLOOR,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        boost_threshold: int = DEFAULT_BOOST_THRESHOLD,
        boost_amount: float = DEFAULT_BOOST_AMOUNT,
        min_score: float = DEFAULT_MIN_SCORE,
        min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
        min_unique_queries: int = DEFAULT_MIN_UNIQUE_QUERIES,
        boost_on_promote: float = DEFAULT_BOOST_ON_PROMOTE,
        merge_model: str = DEFAULT_MERGE_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_BASE,
        diary_path: Path | None = None,
        embedder=None,
        behavioral_threshold: int = DEFAULT_BEHAVIORAL_THRESHOLD,
        behavioral_lookback_days: int = DEFAULT_BEHAVIORAL_LOOKBACK_DAYS,
        calibration_enabled: bool = DEFAULT_CALIBRATION_ENABLED,
        stale_procedure_days: int = DEFAULT_STALE_PROCEDURE_DAYS,
        calibration_lookback_days: int = DEFAULT_CALIBRATION_LOOKBACK_DAYS,
        auto_link_threshold: float = DEFAULT_AUTO_LINK_THRESHOLD,
    ):
        self._store = store
        self._similarity_threshold = similarity_threshold
        self._decay_rate = decay_rate
        self._decay_interval_days = decay_interval_days
        self._decay_floor = decay_floor
        self._confidence_floor = confidence_floor
        self._boost_threshold = boost_threshold
        self._boost_amount = boost_amount
        self._min_score = min_score
        self._min_recall_count = min_recall_count
        self._min_unique_queries = min_unique_queries
        self._boost_on_promote = boost_on_promote
        self._merge_model = merge_model
        self._ollama_url = ollama_url
        self._diary_path = Path(diary_path) if diary_path else get_dream_diary_path()
        self._embedder = embedder
        self._behavioral_threshold = behavioral_threshold
        self._behavioral_lookback_days = behavioral_lookback_days
        self._calibration_enabled = calibration_enabled
        self._stale_procedure_days = stale_procedure_days
        self._calibration_lookback_days = calibration_lookback_days
        self._auto_link_threshold = auto_link_threshold

    def run(self, apply: bool = False, phase: str | None = None) -> DreamResult:
        """Run the dream consolidation pass.

        Args:
            apply: If True, apply changes. If False, dry run.
            phase: Run a specific phase ('light', 'deep', 'rem'), or all.

        Returns:
            DreamResult with phase results.
        """
        result = DreamResult()

        if phase is None or phase == "light":
            result.light = self._light_phase(apply=apply)
            self._run_hooks("light", result, apply)

        if phase is None or phase == "deep":
            merge_candidates = (
                result.light.merge_candidates if result.light else []
            )
            result.deep = self._deep_phase(
                apply=apply, merge_candidates=merge_candidates
            )
            self._run_hooks("deep", result, apply)

        if phase is None or phase == "rem":
            result.rem = self._rem_phase(apply=apply)
            self._run_hooks("rem", result, apply)

        # Write diary
        if apply and result.deep:
            self._write_diary(result)

        return result

    def _run_hooks(self, phase: str, result: DreamResult, apply: bool) -> None:
        """Run any registered dream hooks for the given phase.

        Hook errors are logged but never propagated — a faulty hook
        must not crash the dream cycle.
        """
        hooks = get_registered_dream_hooks()
        hook_fn = hooks.get(phase)
        if hook_fn is not None:
            try:
                hook_fn(self, result, apply)
            except Exception as exc:
                log.error("llmem: dream: %s hook failed: %s", phase, exc)

    def _light_phase(self, apply: bool = False) -> LightPhaseResult:
        """Find near-duplicate pairs."""
        pairs = self._store.consolidate_duplicates(
            similarity_threshold=self._similarity_threshold
        )
        return LightPhaseResult(
            duplicate_pairs=len(pairs),
            merge_candidates=pairs[:20],
        )

    def _deep_phase(
        self,
        apply: bool = False,
        merge_candidates: list[dict] | None = None,
    ) -> DeepPhaseResult:
        """Decay, boost, merge, promote, and auto-link.

        Args:
            apply: If True, apply changes. If False, dry run (count only).
            merge_candidates: Pairs from light phase to merge. When None,
                no merging occurs (backward compat for single-phase runs).
        """
        result = DeepPhaseResult()

        # Decay idle memories — use created_at (not updated_at) so that
        # boosts and other updates don't reset the decay clock.
        # Memories accessed within the decay interval are still relevant
        # and skip decay entirely. The decay interval acts as a grace period:
        # if you touched it in the last N days, it's not idle.
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._decay_interval_days)
        memories = self._store.search(valid_only=True, limit=500)
        for m in memories:
            created = m.get("created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt < cutoff:
                        accessed_at = m.get("accessed_at", "")
                        if accessed_at:
                            accessed_dt = datetime.fromisoformat(accessed_at)
                            if accessed_dt >= cutoff:
                                continue
                        new_conf = max(
                            m.get("confidence", 0.8) - self._decay_rate,
                            self._decay_floor,
                        )
                        if new_conf < self._confidence_floor:
                            if apply:
                                self._store.invalidate(
                                    m["id"],
                                    reason="Dream decay: confidence below floor",
                                )
                            result.decayed_count += 1
                        elif apply:
                            self._store.update(m["id"], confidence=new_conf)
                            result.decayed_count += 1

                except (ValueError, TypeError):
                    pass

        # Boost frequently accessed memories
        for m in memories:
            if m.get("access_count", 0) >= self._boost_threshold:
                new_conf = min(m.get("confidence", 0.8) + self._boost_amount, 1.0)
                if apply:
                    self._store.update(m["id"], confidence=new_conf)
                result.boosted_count += 1

        # Merge near-duplicate pairs found by the light phase
        if merge_candidates:
            for pair in merge_candidates:
                src_id = pair["source"]
                tgt_id = pair["target"]
                src_mem = self._store.get(src_id)
                tgt_mem = self._store.get(tgt_id)
                if not src_mem or not tgt_mem:
                    continue
                if src_mem.get("valid_until") or tgt_mem.get("valid_until"):
                    continue
                src_conf = src_mem.get("confidence", 0.8)
                tgt_conf = tgt_mem.get("confidence", 0.8)
                if src_conf >= tgt_conf:
                    keeper, loser = src_id, tgt_id
                else:
                    keeper, loser = tgt_id, src_id
                if apply:
                    self._store.invalidate(
                        loser,
                        reason=f"Dream merge: superseded by {keeper}",
                    )
                    self._store.add_relation(keeper, loser, "supersedes")
                result.merged_count += 1

        # Promote inbox items to long-term memory
        inbox_count = self._store.inbox_count()
        if inbox_count > 0:
            consolidate_result = self._store.consolidate(
                min_score=self._min_score, dry_run=not apply
            )
            result.promoted_count += len(consolidate_result["promoted"])

        # Auto-link similar memories using consolidate_duplicates
        if apply:
            try:
                pairs = self._store.consolidate_duplicates(
                    similarity_threshold=self._auto_link_threshold
                )
                existing = self._store.get_relations_batch(
                    [p["source"] for p in pairs] + [p["target"] for p in pairs]
                )
                existing_set = set()
                for rel in existing:
                    pair = tuple(sorted([rel["source_id"], rel["target_id"]]))
                    if rel["relation_type"] == "related_to":
                        existing_set.add(pair)
                for pair in pairs:
                    pair_key = tuple(sorted([pair["source"], pair["target"]]))
                    if pair_key not in existing_set:
                        self._store.add_relation(
                            pair["source"], pair["target"], "related_to"
                        )
                        result.auto_linked_count += 1
                        existing_set.add(pair_key)
            except Exception as exc:
                log.warning("llmem: dream: auto-link failed: %s", exc)

        return result

    def _rem_phase(self, apply: bool = False) -> RemPhaseResult:
        """Reflect and extract themes and behavioral insights.

        Clusters self_assessment memories by ERROR_TAXONOMY category within the
        lookback window. When a category meets the behavioral_threshold, generates
        a proposed procedural memory referencing the actual pattern. Writes
        proposed procedures to the store (type=procedure, metadata contains
        proposed=true and source=dream_rem).

        Also extracts top-N content-word themes from all active memories.
        """
        total = self._store.count()
        active = self._store.count(valid_only=True)

        themes = self._extract_themes()

        behavioral_insights = self._extract_behavioral_insights(
            apply=apply,
            lookback_days=self._behavioral_lookback_days,
            threshold=self._behavioral_threshold,
        )

        return RemPhaseResult(
            total_memories=total,
            active_memories=active,
            themes=themes,
            behavioral_insights=behavioral_insights,
        )

    def _extract_themes(self, top_n: int = 8) -> list[str]:
        """Extract top-N content themes from active memories.

        Groups memories by type and returns the most common types,
        plus content-word clusters from the most frequent words.
        """
        type_counts = self._store.count_by_type(valid_only=True)
        themes = []
        for mem_type, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_n]:
            themes.append(f"{count} memories about {mem_type}")

        memories = self._store.search(valid_only=True, limit=200)
        word_freq: Counter[str] = Counter()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "and",
            "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "than", "too", "very", "just", "also",
            "this", "that", "these", "those", "it", "its", "we", "our", "us",
            "they", "them", "their", "i", "me", "my", "you", "your",
        }
        for m in memories:
            content = m.get("content", "")
            words = re.findall(r"[a-zA-Z_]{4,}", content.lower())
            for w in words:
                if w not in stop_words:
                    word_freq[w] += 1
        for word, count in word_freq.most_common(top_n):
            if count >= 2:
                themes.append(f"cluster: {count} memories involve '{word}'")
        return themes

    def _extract_behavioral_insights(
        self,
        apply: bool,
        lookback_days: int,
        threshold: int,
    ) -> list[dict]:
        """Detect recurring self_assessment patterns and generate insights.

        For each ERROR_TAXONOMY category with >= threshold occurrences in the
        lookback window, generate a proposed procedural memory that references
        the actual pattern (specific category, occurrence count, and sample
        content snippets). Optionally write proposed procedures to the store.

        Returns:
            List of insight dicts with keys: category, count, insight_id,
            content_snippet.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_iso = cutoff.isoformat()

        self_assessments = self._store.search(
            type="self_assessment",
            valid_only=True,
            limit=500,
        )

        recent: list[dict] = []
        for m in self_assessments:
            updated = m.get("updated_at", m.get("created_at", ""))
            if updated and updated >= cutoff_iso:
                recent.append(m)

        category_counts: Counter[str] = Counter()
        category_samples: dict[str, list[dict]] = {}
        for m in recent:
            content = m.get("content", "")
            for cat in ERROR_TAXONOMY:
                if f"Category: {cat}" in content:
                    category_counts[cat] += 1
                    category_samples.setdefault(cat, []).append(m)
                    break

        insights: list[dict] = []
        for cat, count in category_counts.most_common():
            if count < threshold:
                continue

            samples = category_samples.get(cat, [])
            sample_snippets = []
            for s in samples[:3]:
                snippet = s.get("content", "")[:120].strip()
                context = ""
                for line in snippet.split("\n"):
                    if line.strip().startswith("Context:"):
                        context = line.strip().replace("Context:", "").strip()
                        break
                sample_snippets.append({"id": s["id"], "snippet": snippet, "context": context})

            insight = {
                "category": cat,
                "count": count,
                "insight_id": None,
                "content_snippets": sample_snippets,
            }

            if apply:
                existing_pattern = f"dream_rem:{cat}:{count}"
                already = self._store.search(
                    query=existing_pattern,
                    valid_only=True,
                    limit=5,
                )
                already_proposed = any(
                    m.get("content", "").startswith(existing_pattern)
                    for m in already
                )
                if not already_proposed:
                    content_parts = [
                        existing_pattern,
                        f"When encountering {cat.lower()} situations, follow these detection rules:",
                    ]
                    for s in sample_snippets[:3]:
                        ctx = s["context"]
                        if ctx:
                            content_parts.append(f"- seen in: {ctx}")
                    description = ERROR_TAXONOMY.get(cat, cat)
                    content_parts.append(f"Rule: {description}")

                    proposed_id = self._store.add(
                        type="procedure",
                        content="\n".join(content_parts),
                        confidence=0.85,
                        source="dream_rem",
                        metadata={
                            "proposed": "true",
                            "category": cat,
                            "occurrence_count": str(count),
                            "lookback_days": str(lookback_days),
                        },
                    )
                    insight["insight_id"] = proposed_id

            insights.append(insight)

        if self._calibration_enabled:
            self._run_calibration(apply=apply, insights=insights)

        return insights

    def _run_calibration(
        self, apply: bool, insights: list[dict]
    ) -> None:
        """Check whether proposed procedures from previous dreams reduced error counts.

        For each proposed procedure with metadata.proposed=true and
        metadata.category set, compare the occurrence count in the current
        lookback window against the count stored in metadata.occurrence_count.
        Stores a calibration event memory.
        """
        proposed = self._store.search(
            type="procedure",
            valid_only=True,
            limit=500,
        )
        proposed_procedures = [
            m
            for m in proposed
            if m.get("metadata", {})
            .get("proposed", "")
            .lower()
            .startswith("true")
        ]

        if not proposed_procedures:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._calibration_lookback_days
        )
        cutoff_iso = cutoff.isoformat()

        recent_assessments = self._store.search(
            type="self_assessment",
            valid_only=True,
            limit=500,
        )
        recent_by_cat: Counter[str] = Counter()
        for m in recent_assessments:
            updated = m.get("updated_at", m.get("created_at", ""))
            if updated and updated >= cutoff_iso:
                content = m.get("content", "")
                for cat in ERROR_TAXONOMY:
                    if f"Category: {cat}" in content:
                        recent_by_cat[cat] += 1
                        break

        for proc in proposed_procedures:
            meta = proc.get("metadata", {})
            cat = meta.get("category", "")
            old_count = int(meta.get("occurrence_count", "0"))
            new_count = recent_by_cat.get(cat, 0)

            calibration_status = "stable"
            if new_count < old_count:
                calibration_status = "decreasing"
            elif new_count > old_count:
                calibration_status = "increasing"

            if apply and calibration_status != "stable":
                self._store.add(
                    type="event",
                    content=f"dream_calibration: {cat} occurrences {old_count}->{new_count} ({calibration_status})",
                    confidence=0.8,
                    source="dream_rem",
                    metadata={
                        "calibration": "true",
                        "category": cat,
                        "previous_count": str(old_count),
                        "current_count": str(new_count),
                        "status": calibration_status,
                    },
                )

    def _write_diary(self, result: DreamResult) -> None:
        """Append to the dream diary.

        Skips writing if the last diary entry has the same timestamp
        (to the minute), preventing duplicate entries from rapid-fire
        invocations or parallel dream runs.
        """
        try:
            diary_path = _validate_output_path(self._diary_path, "diary")
        except ValueError as e:
            log.warning("llmem: dream: %s", e)
            return

        diary_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        timestamp_minute = timestamp[:16]

        if diary_path.exists():
            try:
                existing = diary_path.read_text()
                for m in re.finditer(
                    r"^## Dream — (.+)$", existing, re.MULTILINE
                ):
                    existing_ts = m.group(1).strip()
                    existing_minute = re.sub(r"[T ]", " ", existing_ts)[:16]
                    if existing_minute == timestamp_minute:
                        log.info(
                            "llmem: dream: skipping diary entry, "
                            "entry for %s already exists",
                            timestamp_minute,
                        )
                        return
            except Exception:
                log.debug("llmem: dream: could not check existing diary, writing anyway")

        entry = f"\n## Dream — {timestamp}\n\n"

        if result.light:
            entry += f"- Duplicate pairs: {result.light.duplicate_pairs}\n"

        if result.deep:
            entry += f"- Decayed: {result.deep.decayed_count}\n"
            entry += f"- Boosted: {result.deep.boosted_count}\n"
            entry += f"- Promoted: {result.deep.promoted_count}\n"
            entry += f"- Invalidated: {result.deep.invalidated_count}\n"
            entry += f"- Merged: {result.deep.merged_count}\n"
            entry += f"- Auto-linked: {result.deep.auto_linked_count}\n"

        if result.rem:
            entry += "\n### REM Phase\n\n"
            entry += f"- Total memories: {result.rem.total_memories}\n"
            entry += f"- Active memories: {result.rem.active_memories}\n"
            if result.rem.themes:
                entry += "- Themes:\n"
                for theme in result.rem.themes:
                    entry += f"  - {theme}\n"
            if result.rem.behavioral_insights:
                entry += "- Behavioral insights:\n"
                for insight in result.rem.behavioral_insights:
                    cat = insight.get("category", "?")
                    count = insight.get("count", 0)
                    iid = insight.get("insight_id", "not written")
                    entry += f"  - {cat}: {count} occurrences (insight_id: {iid})\n"

        with open(diary_path, "a") as f:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(entry)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            else:
                f.write(entry)
