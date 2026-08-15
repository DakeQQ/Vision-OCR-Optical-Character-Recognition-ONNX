"""Optimize PaddleOCR-VL ONNX bundles through the approved shared optimizer."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SHARED_OPTIMIZE_ONNX_COMMON = (
    REPOSITORY_ROOT / "Optimize_ONNX_Common.py"
)
SHARED_OPTIMIZE_ONNX_COMMON = DEFAULT_SHARED_OPTIMIZE_ONNX_COMMON.resolve()


def _load_shared_optimizer():
    """Load the one approved optimizer implementation by its resolved file path."""
    if not SHARED_OPTIMIZE_ONNX_COMMON.is_file():
        raise FileNotFoundError(
            "SHARED_OPTIMIZE_ONNX_COMMON does not exist: "
            f"{SHARED_OPTIMIZE_ONNX_COMMON}"
        )
    spec = importlib.util.spec_from_file_location(
        "Optimize_ONNX_Common", SHARED_OPTIMIZE_ONNX_COMMON
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to import SHARED_OPTIMIZE_ONNX_COMMON: "
            f"{SHARED_OPTIMIZE_ONNX_COMMON}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


import Shared_Merged

_common = _load_shared_optimizer()
OptimizerConfig = _common.OptimizerConfig
Plan = _common.Plan
run_optimizer = _common.run_optimizer

# Bundle paths
SOURCE_BUNDLE_DIR    = SCRIPT_DIR / "PaddleOCRVL_ONNX"
OPTIMIZED_BUNDLE_DIR = SCRIPT_DIR / "PaddleOCRVL_Optimized"
CHECKPOINT_DIR       = Path.home() / "Downloads" / "PaddleOCR-VL-1.6"
ORIGINAL_FOLDER_PATH = SOURCE_BUNDLE_DIR
QUANTED_FOLDER_PATH  = OPTIMIZED_BUNDLE_DIR
DOWNLOAD_PATH        = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q8"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")

# Per-graph optimization plans
_PRIMARY_MERGED_MODEL = Path(
    Shared_Merged.default_model_file_names()["image_prefill_greedy"]
).stem
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata": Plan(method="F32", optimize=False),
    # Export_PaddleOCRVL owns the Q8 cache write/dequant algebra. The shared
    # generic KV surgery recognizes its shapes but attempts to quantize an
    # already-float dequant path, which ORT rejects for DynamicQuantizeMatMul.
    _PRIMARY_MERGED_MODEL: Plan(
        method=QUANT_METHOD,
        external=True,
        optimize=False,
        kv_surgery=False,
        # PaddleOCR-VL's token embedding is untied from lm_head, so keep its
        # semantic Gather in the Q8 plan rather than retaining a float table.
        quantize_embedding=True,
    ),
    "LLM_Vision": Plan(method=QUANT_METHOD, optimize=False, external=True),
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


def _copy_tokenizer_assets(source: Path, destination: Path) -> int:
    """Copy the target checkpoint assets required by the standalone runtime."""
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration_paddleocr_vl.py",
        "generation_config.json",
        "image_processing_paddleocr_vl.py",
        "preprocessor_config.json",
        "processing_paddleocr_vl.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    copied = 0
    for name in names:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
            copied += 1
    return copied


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
    print(f"Using SHARED_OPTIMIZE_ONNX_COMMON: {SHARED_OPTIMIZE_ONNX_COMMON}")
    quant_method = quant_method.upper()
    output_folder = (output_folder or _default_output_dir(quant_method)).resolve()
    if output_folder == SOURCE_BUNDLE_DIR.resolve():
        raise ValueError("--output-folder must not overwrite PaddleOCRVL_ONNX.")
    config = _common.make_affine_v2_variant_config(
        CONFIG, quant_method, output_folder
    )
    run_optimizer(config)
    copied = _copy_tokenizer_assets(CHECKPOINT_DIR, output_folder)
    print(f"Copied {copied} PaddleOCR-VL tokenizer/processor assets.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize PaddleOCR-VL with AFFINE_REFINE_V2 Q4 or Q8 decoder weights."
    )
    parser.add_argument(
        "--quant-method", choices=_AFFINE_REFINE_V2_METHODS, default=QUANT_METHOD
    )
    parser.add_argument(
        "--output-folder", type=Path,
        help="Variant output directory; defaults to the canonical Q8 directory or a Q4 sibling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimize_bundle(args.quant_method, args.output_folder)


if __name__ == "__main__":
    main()