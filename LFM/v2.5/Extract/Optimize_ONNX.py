"""Optimize LFM2.5-VL-450M-Extract through the repository shared optimizer."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import Shared_Merged


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SHARED_OPTIMIZE_ONNX_COMMON = REPO_ROOT / "Optimize_ONNX_Common.py"
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRIPT_DIR / ".numba_cache"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets


def _load_shared_optimizer():
    spec = importlib.util.spec_from_file_location(
        "_lfm25_vl_optimizer_common", SHARED_OPTIMIZE_ONNX_COMMON
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared optimizer: {SHARED_OPTIMIZE_ONNX_COMMON}")
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
SOURCE_BUNDLE_DIR    = SCRIPT_DIR / "LFM25_VL_ONNX"
OPTIMIZED_BUNDLE_DIR = SCRIPT_DIR / "LFM25_VL_Optimized"
CHECKPOINT_DIR       = Path.home() / "Downloads" / "LFM2.5-VL-450M-Extract"
ORIGINAL_FOLDER_PATH = SOURCE_BUNDLE_DIR
QUANTED_FOLDER_PATH  = OPTIMIZED_BUNDLE_DIR
MODEL_FOLDER         = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q4"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")

# Per-graph optimization plans
PRIMARY_MERGED_MODEL = Path(Shared_Merged.default_model_file_names()["image_prefill_greedy"]).stem
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata": Plan(method="F32", optimize=False),
    "LLM_Image_Preprocess": Plan(method="F32", optimize=False),
    "LLM_Vision": Plan(method=QUANT_METHOD, op_types=("MatMul",), axes=(0,), external=True, optimize=True),
    PRIMARY_MERGED_MODEL: Plan(
        method=QUANT_METHOD,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        external=True,
        optimize=True,
        kv_surgery=False,
    ),
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
    """Expose approved shared helpers without copying their implementation."""
    return getattr(_common, name)


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
        raise ValueError("--output-folder must not overwrite LFM25_VL_ONNX.")
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
        description="Optimize LFM2.5-VL Extract with AFFINE_REFINE_V2 Q4 or Q8 weights."
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
