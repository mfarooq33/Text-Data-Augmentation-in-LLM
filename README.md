# Text-Data-Augmentation-in-LLM

A small, dependency-free Python utility for creating label-preserving text
variations from scarce datasets.  It is useful as a baseline when labeled
examples are limited and avoids sending data to an external API.

## Usage

```python
from text_augmentation import TextAugmenter, augment_dataset

augmenter = TextAugmenter(seed=42)
variants = augmenter.augment(
    "This is a good and useful example.",
    n=3,
    methods=["synonym", "swap", "deletion", "insertion"],
)

records = [
    {"text": "This is good", "label": "positive"},
    {"text": "This is bad", "label": "negative"},
]
augmented = augment_dataset(records, n_per_sample=2, seed=42)
```

The supported methods are:

- `synonym`: replaces an eligible word from a configurable local synonym map
- `swap`: exchanges the positions of two whitespace-separated tokens
- `deletion`: removes one token
- `insertion`: inserts a related local word

Mappings passed to `TextAugmenter` or `augment_dataset` can extend or replace
the default synonym map.  Dataset mappings retain their label and metadata
fields.  Augmentation is deterministic when a seed is supplied.

This library implements conservative local perturbations rather than
ground-truth-generating LLM paraphrases.  Review augmented data when use cases
require semantic or domain-specific precision.

## Tests

```bash
python -m unittest -v test_text_augmentation.py
```
