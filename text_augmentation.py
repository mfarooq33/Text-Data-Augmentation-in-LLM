"""Deterministic text augmentation for small, scarce-text datasets.

The implementation uses only the Python standard library so it can be used
without downloading model files or adding API credentials.
"""

import random
import re
from typing import Iterable, Mapping, Optional, Sequence


DEFAULT_SYNONYMS = {
    "good": ("great", "positive"),
    "bad": ("poor", "negative"),
    "big": ("large", "substantial"),
    "small": ("little", "compact"),
    "fast": ("quick", "rapid"),
    "slow": ("gradual", "unhurried"),
    "happy": ("glad", "pleased"),
    "sad": ("unhappy", "sorrowful"),
    "important": ("significant", "essential"),
    "useful": ("helpful", "practical"),
    "easy": ("simple", "straightforward"),
    "hard": ("difficult", "challenging"),
    "new": ("recent", "fresh"),
    "old": ("aged", "former"),
    "start": ("begin", "commence"),
    "end": ("finish", "conclude"),
    "help": ("assist", "support"),
    "make": ("create", "produce"),
}

_WORD_RE = re.compile(r"[A-Za-z]+")


class TextAugmenter:
    """Generate label-preserving variants of short text samples."""

    AVAILABLE_METHODS = ("synonym", "swap", "deletion", "insertion")

    def __init__(
        self,
        synonym_map: Optional[Mapping[str, Sequence[str]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if synonym_map is None:
            synonym_map = DEFAULT_SYNONYMS
        self.synonym_map = {
            word.lower(): tuple(options)
            for word, options in synonym_map.items()
        }
        self.seed = seed

    def augment(
        self,
        text: str,
        n: int = 1,
        methods: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Return up to *n* unique variants, excluding the original text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if n < 1:
            return []

        selected = tuple(methods) if methods is not None else self.AVAILABLE_METHODS
        unknown = set(selected) - set(self.AVAILABLE_METHODS)
        if unknown:
            raise ValueError("unknown augmentation method(s): " + ", ".join(sorted(unknown)))

        rng = random.Random(seed if seed is not None else self.seed)
        variants = []
        seen = {text}
        attempts = max(n * 20, 20)

        for _ in range(attempts):
            if len(variants) == n:
                break
            method = rng.choice(selected)
            candidate = getattr(self, "_augment_" + method)(text, rng)
            candidate = " ".join(candidate.split())
            if candidate and candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)

        return variants

    def augment_text(
        self,
        text: str,
        n: int = 1,
        methods: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Alias for :meth:`augment`."""
        return self.augment(text, n=n, methods=methods, seed=seed)

    def _augment_synonym(self, text: str, rng: random.Random) -> str:
        words = text.split()
        choices = [
            index
            for index, word in enumerate(words)
            if _WORD_RE.fullmatch(word.strip(".,!?;:\"'()[]{}"))
            and word.lower().strip(".,!?;:\"'()[]{}") in self.synonym_map
        ]
        if not choices:
            return text
        index = rng.choice(choices)
        original = words[index]
        leading = original[: len(original) - len(original.lstrip(".,!?;:\"'()[]{}"))]
        core = original.strip(".,!?;:\"'()[]{}")
        trailing = original[len(original.rstrip(".,!?;:\"'()[]{}")):]
        replacement = rng.choice(self.synonym_map[core.lower()])
        if core.isupper():
            replacement = replacement.upper()
        elif core[:1].isupper():
            replacement = replacement.capitalize()
        words[index] = leading + replacement + trailing
        return " ".join(words)

    def _augment_swap(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        first = rng.randrange(len(words))
        second = rng.randrange(len(words))
        if first == second:
            second = (first + 1) % len(words)
        words[first], words[second] = words[second], words[first]
        return " ".join(words)

    def _augment_deletion(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        del words[rng.randrange(len(words))]
        return " ".join(words)

    def _augment_insertion(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if not words:
            return text
        candidates = []
        for word in words:
            core = word.strip(".,!?;:\"'()[]{}").lower()
            candidates.extend([core] if core else [])
            candidates.extend(self.synonym_map.get(core, ()))
        if not candidates:
            candidates = words[:]
        position = rng.randrange(len(words) + 1)
        words.insert(position, rng.choice(candidates))
        return " ".join(words)


def augment_text(
    text: str,
    n: int = 1,
    methods: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    synonym_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> list[str]:
    """Convenience function for generating text variants."""
    return TextAugmenter(synonym_map=synonym_map, seed=seed).augment(
        text, n=n, methods=methods, seed=seed
    )


def augment_dataset(
    records: Sequence,
    n_per_sample: int = 1,
    text_key: str = "text",
    include_original: bool = True,
    methods: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    synonym_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> list:
    """Augment string records or mappings while preserving labels and metadata."""
    if n_per_sample < 1:
        raise ValueError("n_per_sample must be at least 1")

    augmenter = TextAugmenter(synonym_map=synonym_map, seed=seed)
    augmented = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            text = record
        elif isinstance(record, Mapping):
            text = record.get(text_key)
            if not isinstance(text, str):
                raise TypeError("mapped text values must be strings")
        else:
            raise TypeError("records must be strings or mappings")

        if include_original:
            augmented.append(record)

        for variant in augmenter.augment(
            text,
            n=n_per_sample,
            methods=methods,
            seed=None if seed is None else seed + index,
        ):
            if isinstance(record, Mapping):
                item = dict(record)
                item[text_key] = variant
                augmented.append(item)
            else:
                augmented.append(variant)
    return augmented
