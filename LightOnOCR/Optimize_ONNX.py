"""Optimize LightOnOCR's ONNX bundle through the shared OCR optimizer."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets


def _load_local_shared_merged():
    module_path = SCRIPT_DIR / "Shared_Merged.py"
    spec = importlib.util.spec_from_file_location(
        "_lightonocr_shared_merged", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local merge helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


Shared_Merged = _load_local_shared_merged()


def _load_shared_optimizer():
    module_path = REPO_ROOT / "Optimize_ONNX_Common.py"
    spec = importlib.util.spec_from_file_location(
        "_lightonocr_optimizer_common", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared optimizer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


_common = _load_shared_optimizer()
OptimizerConfig = _common.OptimizerConfig
Plan = _common.Plan
run_optimizer = _common.run_optimizer


# Bundle paths
SOURCE_BUNDLE_DIR    = SCRIPT_DIR / "LightOnOCR_ONNX"
OPTIMIZED_BUNDLE_DIR = SCRIPT_DIR / "LightOnOCR_Optimized"
CHECKPOINT_DIR       = Path.home() / "Downloads" / "LightOnOCR-2-1B"
ORIGINAL_FOLDER_PATH = SOURCE_BUNDLE_DIR
QUANTED_FOLDER_PATH  = OPTIMIZED_BUNDLE_DIR
DOWNLOAD_PATH        = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q4"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"  # AFFINE_REFINE_V2 | k_quant
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")

# Per-graph optimization plans
_PRIMARY_MERGED_MODEL = Path(
    Shared_Merged.default_model_file_names()["image_prefill_greedy"]
).stem
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata": Plan(method="F32", optimize=False),
    _PRIMARY_MERGED_MODEL: Plan(method=QUANT_METHOD, external=True, optimize=True),
    "LLM_Vision": Plan(method=QUANT_METHOD, external=True),
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
    """Preserve direct access to shared helper functions from this legacy path."""
    return getattr(_common, name)


def _copy_f32_bundle() -> None:
    source = SOURCE_BUNDLE_DIR.resolve()
    target = OPTIMIZED_BUNDLE_DIR.resolve()
    if source == target:
        raise ValueError("SOURCE_BUNDLE_DIR and OPTIMIZED_BUNDLE_DIR must differ for F32.")
    if not source.is_dir() or not (source / "LLM_Metadata.onnx").is_file():
        raise FileNotFoundError(f"Missing source LightOnOCR bundle: {source}")
    staging = target.with_name(f"{target.name}.staging")
    previous = target.with_name(f"{target.name}.previous")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    if not (staging / "LLM_SharedInitializers.onnx").is_file():
        raise RuntimeError("F32 bundle copy is missing the shared initializer carrier.")
    if previous.exists():
        shutil.rmtree(previous)
    try:
        if target.exists():
            target.replace(previous)
        staging.replace(target)
    except BaseException:
        if previous.exists() and not target.exists():
            previous.replace(target)
        raise
    print(f"F32 bundle copied: {target}")


def _default_output_dir(quant_method: str) -> Path:
    output = Path(OPTIMIZED_BUNDLE_DIR)
    return output if quant_method == QUANT_METHOD else output.with_name(
        f"{output.name}_{quant_method}"
    )


def optimize_bundle(
    quant_method: str = QUANT_METHOD,
    output_folder: Path | None = None,
) -> None:
    """Optimize the exported bundle and restore its runtime assets."""
    quant_method = quant_method.upper()
    output_folder = (output_folder or _default_output_dir(quant_method)).resolve()
    if output_folder == SOURCE_BUNDLE_DIR.resolve():
        raise ValueError("--output-folder must not overwrite LightOnOCR_ONNX.")
    config = _common.make_affine_v2_variant_config(
        CONFIG, quant_method, output_folder
    )
    run_optimizer(config)
    tokenizer_assets = copy_tokenizer_assets(
        SOURCE_BUNDLE_DIR, output_folder
    )
    print(f"Copied {len(tokenizer_assets)} tokenizer assets to {output_folder}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize LightOnOCR with AFFINE_REFINE_V2 Q4 or Q8 weights."
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