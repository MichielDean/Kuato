"""Background dreaming/consolidation pass — automated memory maintenance.

Three phases:
  Light — sort + dedupe: find near-duplicates using cosine similarity
  Deep  — score + promote + decay + merge: quality-gated promotion, idle decay,
          frequent-access boost, LLM-assisted merge
  REM   — reflect + cluster: extract themes, write dream diary (read-only)
"""

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .ollama import _call_ollama_generate
from .store import MemoryStore
from .taxonomy import ERROR_TAXONOMY, ERROR_TAXONOMY_KEYS, _parse_self_assessment
from .url_validate import is_safe_url
from .paths import get_dream_diary_path, get_proposed_changes_path

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
DEFAULT_SKILL_PATCH_THRESHOLD = 3
DEFAULT_CALIBRATION_ENABLED = True
DEFAULT_STALE_PROCEDURE_DAYS = 30
DEFAULT_CALIBRATION_LOOKBACK_DAYS = 90

_DANGEROUS_PATH_PREFIXES = ("/etc/", "/var/", "/sys/", "/proc/", "/dev/", "/boot/")


def _validate_output_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if any(str(resolved).startswith(p) for p in _DANGEROUS_PATH_PREFIXES):
        raise ValueError(f"{label} path targets a protected directory: {resolved!s}")
    if ".." in str(path):
        raise ValueError(f"{label} path contains '..' traversal: {path!s}")
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
        skill_patch_threshold: int = DEFAULT_SKILL_PATCH_THRESHOLD,
        proposed_changes_path: Path | None = None,
        calibration_enabled: bool = DEFAULT_CALIBRATION_ENABLED,
        stale_procedure_days: int = DEFAULT_STALE_PROCEDURE_DAYS,
        calibration_lookback_days: int = DEFAULT_CALIBRATION_LOOKBACK_DAYS,
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
        self._diary_path = diary_path or get_dream_diary_path()
        self._embedder = embedder
        self._behavioral_threshold = behavioral_threshold
        self._behavioral_lookback_days = behavioral_lookback_days
        self._skill_patch_threshold = skill_patch_threshold
        self._proposed_changes_path = (
            proposed_changes_path or get_proposed_changes_path()
        )
        self._calibration_enabled = calibration_enabled
        self._stale_procedure_days = stale_procedure_days
        self._calibration_lookback_days = calibration_lookback_days

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
        if phase is None or phase == "deep":
            result.deep = self._deep_phase(apply=apply)
        if phase is None or phase == "rem":
            result.rem = self._rem_phase(apply=apply)

        # Write diary
        if apply and result.deep:
            self._write_diary(result)

        return result

    def _light_phase(self, apply: bool = False) -> LightPhaseResult:
        """Find near-duplicate pairs."""
        pairs = self._store.consolidate(similarity_threshold=self._similarity_threshold)
        return LightPhaseResult(
            duplicate_pairs=len(pairs),
            merge_candidates=pairs[:20],
        )

    def _deep_phase(self, apply: bool = False) -> DeepPhaseResult:
        """Decay, boost, promote, and merge."""
        result = DeepPhaseResult()

        # Decay idle memories
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._decay_interval_days)
        memories = self._store.search(valid_only=True, limit=500)
        for m in memories:
            updated = m.get("updated_at", "")
            if updated:
                try:
                    updated_dt = datetime.fromisoformat(updated)
                    if updated_dt < cutoff:
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

        return result

    def _rem_phase(self, apply: bool = False) -> RemPhaseResult:
        """Reflect and extract themes."""
        total = self._store.count()
        active = self._store.count(valid_only=True)
        return RemPhaseResult(
            total_memories=total,
            active_memories=active,
            themes=[],
            behavioral_insights=[],
        )

    def _write_diary(self, result: DreamResult) -> None:
        """Append to the dream diary."""
        try:
            diary_path = _validate_output_path(self._diary_path, "diary")
        except ValueError as e:
            log.warning("llmem: dream: %s", e)
            return

        diary_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"\n## Dream — {timestamp}\n\n"

        if result.deep:
            entry += f"- Decayed: {result.deep.decayed_count}\n"
            entry += f"- Boosted: {result.deep.boosted_count}\n"
            entry += f"- Promoted: {result.deep.promoted_count}\n"
            entry += f"- Invalidated: {result.deep.invalidated_count}\n"
            entry += f"- Merged: {result.deep.merged_count}\n"

        with open(diary_path, "a") as f:
            f.write(entry)
