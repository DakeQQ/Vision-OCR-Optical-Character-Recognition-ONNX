"""Copy tokenizer and processor sidecars into portable OCR ONNX bundles."""

from __future__ import annotations

import shutil
from pathlib import Path


TOKENIZER_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "vocab.txt",
    }
)
TOKENIZER_CODE_PREFIXES = (
    "configuration_",
    "image_processing_",
    "processing_",
    "tokenization_",
)
TOKENIZER_SERIALIZATION_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
)


def _is_tokenizer_asset(path: Path) -> bool:
    return path.name in TOKENIZER_ASSET_NAMES or (
        path.suffix == ".py" and path.name.startswith(TOKENIZER_CODE_PREFIXES)
    )


def copy_tokenizer_assets(source_dir: Path | str, destination_dir: Path | str) -> list[str]:
    """Copy tokenizer, vocabulary, processor, and supporting configuration files."""
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Tokenizer source directory does not exist: {source}")
    if not any((source / name).is_file() for name in TOKENIZER_SERIALIZATION_NAMES):
        expected = ", ".join(TOKENIZER_SERIALIZATION_NAMES)
        raise FileNotFoundError(
            f"No tokenizer serialization found in {source}; expected one of: {expected}."
        )

    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source_path in sorted(source.iterdir()):
        if source_path.is_file() and _is_tokenizer_asset(source_path):
            shutil.copy2(source_path, destination / source_path.name)
            copied.append(source_path.name)
    return copied