"""Optimize the UnlimitedOCR merged ONNX bundle through Optimize_ONNX_Common."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

SHARED_OPTIMIZE_ONNX_COMMON = REPO_ROOT / "Optimize_ONNX_Common.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


Shared_Merged = _load_module("_unlimitedocr_shared_merged", SCRIPT_DIR / "Shared_Merged.py")
_common = _load_module("_unlimitedocr_optimizer_common", SHARED_OPTIMIZE_ONNX_COMMON)
OptimizerConfig = _common.OptimizerConfig
Plan = _common.Plan
run_optimizer = _common.run_optimizer


# Bundle paths
SOURCE_BUNDLE_DIR    = SCRIPT_DIR / "UnlimitedOCR_ONNX"
OPTIMIZED_BUNDLE_DIR = SCRIPT_DIR / "UnlimitedOCR_Optimized"
CHECKPOINT_DIR       = Path.home() / "Downloads" / "Unlimited-OCR"
ORIGINAL_FOLDER_PATH = SOURCE_BUNDLE_DIR
QUANTED_FOLDER_PATH  = OPTIMIZED_BUNDLE_DIR
DOWNLOAD_PATH        = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q4"
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")

# Per-graph optimization plans
_merged_plan = Plan(
    method=QUANT_METHOD,
    external=True,
    optimize=True,
    op_types=("MatMul", "Gather"),
    axes=(0, 1),
)
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata": Plan(method="F32", optimize=False),
    "LLM_Image_Preprocess": Plan(method="F32", optimize=False),
    "LLM_Vision": Plan(method=QUANT_METHOD, external=True, optimize=True),
}
for file_name, _, _ in Shared_Merged.MERGED_BUILD_PLAN:
    MODEL_PLANS[Path(file_name).stem] = _merged_plan

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


def _stamp_shared_initializer_metadata(output_folder: Path) -> None:
    """Carry bundle facts onto the shared-initializer carrier after rebuilding it."""
    metadata_path = output_folder / Shared_Merged.default_model_file_names()["metadata"]
    shared_path = output_folder / Shared_Merged.default_model_file_names()["shared_initializers"]
    if not metadata_path.is_file() or not shared_path.is_file():
        raise FileNotFoundError("Optimized metadata or shared-initializer carrier is missing.")
    metadata_model = onnx.load(str(metadata_path), load_external_data=False)
    shared_model = onnx.load(str(shared_path), load_external_data=False)
    values = {item.key: item.value for item in shared_model.metadata_props}
    values.update({item.key: item.value for item in metadata_model.metadata_props})
    shared_model.ClearField("metadata_props")
    for key in sorted(values):
        shared_model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(shared_model, str(shared_path))


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
    quant_method = quant_method.upper()
    output_folder = (output_folder or _default_output_dir(quant_method)).resolve()
    if output_folder == SOURCE_BUNDLE_DIR.resolve():
        raise ValueError("--output-folder must not overwrite UnlimitedOCR_ONNX.")
    config = _common.make_affine_v2_variant_config(
        CONFIG, quant_method, output_folder
    )
    run_optimizer(config)
    tokenizer_assets = copy_tokenizer_assets(
        SOURCE_BUNDLE_DIR, output_folder
    )
    _stamp_shared_initializer_metadata(output_folder)
    print(f"Copied {len(tokenizer_assets)} tokenizer assets to {output_folder}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize UnlimitedOCR with AFFINE_REFINE_V2 Q4 or Q8 weights."
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