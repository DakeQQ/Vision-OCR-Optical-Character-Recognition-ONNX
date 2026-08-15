"""Optimize OvisOCR2's merged ONNX bundle through the shared OCR optimizer."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path("/tmp") / "ovisocr2_numba_cache"),
)
SHARED_OPTIMIZE_ONNX_COMMON = (
    SCRIPT_DIR.parent.parent / "Optimize_ONNX_Common.py"
).resolve()
if not SHARED_OPTIMIZE_ONNX_COMMON.is_file():
    raise FileNotFoundError(
        "Approved Optimize_ONNX_Common.py does not exist: "
        f"{SHARED_OPTIMIZE_ONNX_COMMON}"
    )


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


Shared_Merged = _load_local_module("_ovisocr2_shared_merged", SCRIPT_DIR / "Shared_Merged.py")
_common = _load_local_module("_ovisocr2_optimizer_common", SHARED_OPTIMIZE_ONNX_COMMON)
OptimizerConfig = _common.OptimizerConfig
Plan = _common.Plan
run_optimizer = _common.run_optimizer


# Bundle paths
SOURCE_BUNDLE_DIR    = SCRIPT_DIR / "OvisOCR2_ONNX"
OPTIMIZED_BUNDLE_DIR = SCRIPT_DIR / "OvisOCR2_Optimized"
CHECKPOINT_DIR       = SCRIPT_DIR
ORIGINAL_FOLDER_PATH = SOURCE_BUNDLE_DIR
QUANTED_FOLDER_PATH  = OPTIMIZED_BUNDLE_DIR
DOWNLOAD_PATH        = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q4"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")

# Runtime asset policy
TOKENIZER_ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
REQUIRED_TOKENIZER_ASSET_NAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.json")
KV_HELPER_ROLES = ("kv_slice", "kv_split2", "kv_concat", "rope_shift")

# Per-graph optimization plans
_PRIMARY_MERGED_MODEL = Path(
    Shared_Merged.default_model_file_names()["image_prefill_greedy"]
).stem
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata": Plan(method="F32", optimize=False),
    _PRIMARY_MERGED_MODEL: Plan(
        method=QUANT_METHOD,
        external=True,
        # The generic transformer optimizer can stall on Ovis's nested
        # If/Loop/Sequence recurrent-state graph. Keep the quantized control
        # flow unchanged; shared-bundle validation still checks every strategy.
        optimize=False,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
    ),
    "LLM_Vision": Plan(method=QUANT_METHOD, external=True, optimize=True),
    "LLM_Image_Preprocess": Plan(method="F32", optimize=False),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_BUNDLE_DIR),
    quantized_folder_path=str(OPTIMIZED_BUNDLE_DIR),
    download_path=str(CHECKPOINT_DIR),
    shared_merged=Shared_Merged,
    model_plans=MODEL_PLANS,
    quant_method=QUANT_METHOD,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
)
_common.configure_optimizer(CONFIG)


def __getattr__(name: str):
    """Expose approved shared optimizer helpers without copying them locally."""
    return getattr(_common, name)


def _copy_tokenizer_assets(output_folder: Path) -> list[str]:
    missing = [
        name for name in REQUIRED_TOKENIZER_ASSET_NAMES
        if not (SOURCE_BUNDLE_DIR / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Exported OvisOCR2 tokenizer assets are missing from "
            f"{SOURCE_BUNDLE_DIR}: {missing!r}."
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_ASSET_NAMES:
        source_path = SOURCE_BUNDLE_DIR / name
        if source_path.is_file():
            shutil.copy2(source_path, output_folder / name)
            copied.append(name)
    return copied


def _copy_kv_helper_graphs(output_folder: Path) -> list[str]:
    """Preserve standalone cache helpers that are not part of merged model plans."""
    file_names = Shared_Merged.default_model_file_names()
    helper_paths = {
        role: SOURCE_BUNDLE_DIR / file_names[role]
        for role in KV_HELPER_ROLES
    }
    present = [role for role, path in helper_paths.items() if path.is_file()]
    if not present:
        return []
    missing = [file_names[role] for role, path in helper_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Exported OvisOCR2 KV helper graphs are missing: {missing!r}."
        )
    copied = []
    for role in KV_HELPER_ROLES:
        filename = file_names[role]
        destination = output_folder / filename
        shutil.copy2(helper_paths[role], destination)
        Shared_Merged.validate_onnx_path(destination)
        copied.append(filename)
    return copied


def _restamp_shared_initializer_metadata(output_folder: Path) -> None:
    metadata_path = SOURCE_BUNDLE_DIR / "LLM_Metadata.onnx"
    shared_path = output_folder / Shared_Merged.default_model_file_names()[
        "shared_initializers"
    ]
    if not metadata_path.is_file() or not shared_path.is_file():
        raise FileNotFoundError(
            "Cannot restamp optimized shared-initializer metadata without both "
            f"{metadata_path} and {shared_path}."
        )
    source = onnx.load(str(metadata_path), load_external_data=False)
    shared = onnx.load(str(shared_path), load_external_data=False)
    values = {item.key: item.value for item in shared.metadata_props}
    values.update({item.key: item.value for item in source.metadata_props})
    shared.ClearField("metadata_props")
    for key in sorted(values):
        shared.metadata_props.add(key=key, value=values[key])
    onnx.save_model(shared, str(shared_path))
    Shared_Merged.validate_onnx_path(shared_path)


def _default_output_dir(quant_method: str) -> Path:
    return (
        OPTIMIZED_BUNDLE_DIR
        if quant_method == QUANT_METHOD
        else OPTIMIZED_BUNDLE_DIR.with_name(f"{OPTIMIZED_BUNDLE_DIR.name}_{quant_method}")
    )


def optimize_bundle(
    quant_method: str = QUANT_METHOD,
    output_folder: Path | None = None,
) -> None:
    """Optimize the exported bundle and restore its runtime assets."""
    print(f"Using shared optimizer: {SHARED_OPTIMIZE_ONNX_COMMON}")
    quant_method = quant_method.upper()
    output_folder = (output_folder or _default_output_dir(quant_method)).resolve()
    if output_folder == SOURCE_BUNDLE_DIR.resolve():
        raise ValueError("--output-folder must not overwrite OvisOCR2_ONNX.")
    config = _common.make_affine_v2_variant_config(
        CONFIG, quant_method, output_folder
    )
    run_optimizer(config)
    kv_helpers = _copy_kv_helper_graphs(output_folder)
    tokenizer_assets = _copy_tokenizer_assets(output_folder)
    _restamp_shared_initializer_metadata(output_folder)
    print(f"Copied {len(kv_helpers)} KV helper graphs to {output_folder}.")
    print(f"Copied {len(tokenizer_assets)} tokenizer assets to {output_folder}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize OvisOCR2 with AFFINE_REFINE_V2 Q4 or Q8 weights."
    )
    parser.add_argument(
        "--quant-method", choices=_AFFINE_REFINE_V2_METHODS, default=QUANT_METHOD
    )
    parser.add_argument(
        "--output-folder", type=Path,
        help="Variant output directory; defaults to the canonical Q4 directory or a Q8 sibling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimize_bundle(args.quant_method, args.output_folder)


if __name__ == "__main__":
    main()
