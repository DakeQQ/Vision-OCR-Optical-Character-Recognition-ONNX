"""Optimize Falcon Perception's merged OCR bundle through the shared optimizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets
from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer
import Shared_Merged


# Bundle paths
SOURCE_BUNDLE_DIR     = SCRIPT_DIR / "Falcon_Perception_ONNX"
OPTIMIZED_BUNDLE_DIR  = SCRIPT_DIR / "Falcon_Perception_Optimized"
CHECKPOINT_DIR        = Path.home() / "Downloads" / "Falcon-Perception-300M"
ORIGINAL_FOLDER_PATH  = SOURCE_BUNDLE_DIR
QUANTIZED_FOLDER_PATH = OPTIMIZED_BUNDLE_DIR
DOWNLOAD_PATH         = CHECKPOINT_DIR

# Quantization policy
QUANT_METHOD          = "Q8"                # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32 (optimize-only).
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"  # AFFINE_REFINE_V2 | DEFAULT | RTN | HQQ | k_quant (Q4-only).
_AFFINE_REFINE_V2_METHODS = ("Q4", "Q8")
_AFFINE_REFINE_V2_OPS = ("MatMul", "Gather")
_AFFINE_REFINE_V2_AXES = (0, 1)

# Per-graph optimization plans
_PRIMARY_MERGED_MODEL = Path(
    Shared_Merged.default_model_file_names()["image_prefill_greedy"]
).stem
def _model_plans(quant_method: str, embedding_only: bool) -> dict[str, Plan]:
    op_types = ("Gather",) if embedding_only else _AFFINE_REFINE_V2_OPS
    axes = (1,) if embedding_only else _AFFINE_REFINE_V2_AXES
    peripheral_method = "F32" if embedding_only else quant_method
    return {
        "LLM_Metadata": Plan(method="F32", optimize=False),
        # Falcon's untied token embedding is detector-sensitive. Quantize its base
        # table, then add a compact residual to preserve detector tokens. Q8 stores
        # that correction as a second Q8 lookup instead of a full F16 table.
        _PRIMARY_MERGED_MODEL: Plan(
            method=quant_method,
            op_types=op_types,
            axes=axes,
            external=True,
            optimize=True,
            quantize_embedding=True,
            embedding_residual_dtype="F16",
            compact_embedding_residual=quant_method == "Q8",
        ),
        "LLM_Vision": Plan(method=peripheral_method, external=True, optimize=False),
        "LLM_Image_Preprocess": Plan(method="F32", optimize=False),
        "LLM_FalconCoordinateFeedback": Plan(method=peripheral_method, optimize=False),
        "LLM_FalconSizeFeedback": Plan(method=peripheral_method, optimize=False),
    }


def _build_config(
    quant_method: str = QUANT_METHOD,
    optimized_bundle_dir: Path = OPTIMIZED_BUNDLE_DIR,
    embedding_only: bool = False,
) -> OptimizerConfig:
    quant_method = quant_method.upper()
    if quant_method not in _AFFINE_REFINE_V2_METHODS:
        raise ValueError(
            f"AFFINE_REFINE_V2 quantization requires one of {_AFFINE_REFINE_V2_METHODS}, got {quant_method!r}."
        )
    return OptimizerConfig(
        original_folder_path=str(SOURCE_BUNDLE_DIR),
        quantized_folder_path=str(optimized_bundle_dir),
        download_path=str(CHECKPOINT_DIR),
        shared_merged=Shared_Merged,
        model_plans=_model_plans(quant_method, embedding_only),
        quant_method=quant_method,
        weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    )


MODEL_PLANS = _model_plans(QUANT_METHOD, embedding_only=False)
CONFIG = _build_config()


def _restamp_bundle_metadata(optimized_bundle_dir: Path) -> None:
    """Preserve the exporter metadata when the shared carrier is regenerated."""
    source = SOURCE_BUNDLE_DIR / Shared_Merged.default_model_file_names()["metadata"]
    metadata_model = onnx.load(str(source), load_external_data=False)
    metadata = {item.key: item.value for item in metadata_model.metadata_props}
    for path in optimized_bundle_dir.glob("*.onnx"):
        model = onnx.load(str(path), load_external_data=False)
        values = {item.key: item.value for item in model.metadata_props}
        values.update(metadata)
        model.ClearField("metadata_props")
        for key in sorted(values):
            model.metadata_props.add(key=key, value=values[key])
        onnx.save_model(model, str(path))


def _node_bits(node: onnx.NodeProto) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "bits":
            return int(attribute.i)
    return None


_FLOAT_MATRIX_TYPES = frozenset({
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.BFLOAT16,
})


def _validate_weight_only_matrix_coverage(
    path: Path,
    model: onnx.ModelProto,
    expected_bits: int,
) -> None:
    """Reject retained constant float matrix weights in a Q8 model."""
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    float_matrix_weights = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm") or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if (
            weight is not None
            and len(weight.dims) == 2
            and weight.data_type in _FLOAT_MATRIX_TYPES
        ):
            float_matrix_weights.append(
                f"{node.name or node.op_type} <- {weight.name}{tuple(weight.dims)}"
            )
    if float_matrix_weights:
        raise RuntimeError(
            f"{path.name} retained float constant matrix weight(s): "
            + "; ".join(float_matrix_weights)
        )

    matmul_bits = {
        _node_bits(node)
        for node in model.graph.node
        if node.op_type == "MatMulNBits"
    }
    if not matmul_bits or matmul_bits != {expected_bits}:
        raise RuntimeError(
            f"{path.name} contains MatMulNBits widths {sorted(matmul_bits)}; "
            f"expected only Q{expected_bits}."
        )


def _validate_affine_refine_v2_bundle(
    optimized_bundle_dir: Path,
    quant_method: str,
    embedding_only: bool,
) -> None:
    expected_bits = int(quant_method[1:])
    compact_residual = _model_plans(
        quant_method, embedding_only
    )[_PRIMARY_MERGED_MODEL].compact_embedding_residual
    file_names = Shared_Merged.default_model_file_names()
    for phase in ("prefill", "decode"):
        for strategy in ("greedy", "penalty_greedy", "sampling"):
            path = optimized_bundle_dir / file_names[f"image_{phase}_{strategy}"]
            model = onnx.load(str(path), load_external_data=False)
            producers = {
                output: node
                for node in model.graph.node
                for output in node.output
                if output
            }
            quantized_outputs = {
                output
                for node in model.graph.node
                if node.op_type == "GatherBlockQuantized"
                and _node_bits(node) == expected_bits
                for output in node.output
                if output
            }
            if compact_residual:
                has_residual_add = any(
                    node.op_type == "Add"
                    and sum(input_name in quantized_outputs for input_name in node.input) >= 2
                    for node in model.graph.node
                )
            else:
                has_residual_add = any(
                    node.op_type == "Add"
                    and any(input_name in quantized_outputs for input_name in node.input)
                    and any(
                        producers.get(input_name, onnx.NodeProto()).op_type in ("Gather", "Cast")
                        for input_name in node.input
                        if input_name not in quantized_outputs
                    )
                    for node in model.graph.node
                )
            if not quantized_outputs or not has_residual_add:
                raise RuntimeError(
                    f"{path.name} is missing a Q{expected_bits} embedding residual correction."
                )
            if compact_residual:
                residual_gathers = [
                    node
                    for node in model.graph.node
                    if node.op_type == "GatherBlockQuantized"
                    and "residual" in node.name.lower()
                ]
                if len(residual_gathers) != 1:
                    raise RuntimeError(
                        f"{path.name} is missing its compact Q8 embedding residual lookup."
                    )
            if not embedding_only:
                _validate_weight_only_matrix_coverage(path, model, expected_bits)

    if not embedding_only:
        for model_name in (
            "LLM_Vision.onnx",
            "LLM_FalconCoordinateFeedback.onnx",
            "LLM_FalconSizeFeedback.onnx",
        ):
            path = optimized_bundle_dir / model_name
            model = onnx.load(str(path), load_external_data=False)
            _validate_weight_only_matrix_coverage(path, model, expected_bits)


def _default_output_dir(quant_method: str, embedding_only: bool) -> Path:
    if quant_method == QUANT_METHOD and not embedding_only:
        return OPTIMIZED_BUNDLE_DIR
    suffix = f"_{quant_method}" + ("_Embed" if embedding_only else "")
    return SCRIPT_DIR / f"Falcon_Perception_Optimized{suffix}"


def optimize_bundle(
    quant_method: str = QUANT_METHOD,
    optimized_bundle_dir: Path = OPTIMIZED_BUNDLE_DIR,
    embedding_only: bool = False,
) -> None:
    """Optimize the exported bundle and restore its runtime assets."""
    quant_method = quant_method.upper()
    optimized_bundle_dir = optimized_bundle_dir.expanduser().resolve()
    run_optimizer(_build_config(quant_method, optimized_bundle_dir, embedding_only))
    _validate_affine_refine_v2_bundle(
        optimized_bundle_dir, quant_method, embedding_only
    )
    _restamp_bundle_metadata(optimized_bundle_dir)
    tokenizer_assets = copy_tokenizer_assets(
        SOURCE_BUNDLE_DIR, optimized_bundle_dir
    )
    print(f"Copied {len(tokenizer_assets)} tokenizer assets to {optimized_bundle_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize Falcon Perception with AFFINE_REFINE_V2 MatMul and residual-corrected embedding quantization."
    )
    parser.add_argument("--quant-method", choices=_AFFINE_REFINE_V2_METHODS, default=QUANT_METHOD)
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Ablation mode: quantize only the untied token embedding and leave Main MatMul weights unquantized.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Output bundle directory; defaults to a method-specific folder for variant runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_folder or _default_output_dir(
        args.quant_method, args.embedding_only
    )
    if output_dir.expanduser().resolve() == SOURCE_BUNDLE_DIR.resolve():
        raise ValueError("--output-folder must not overwrite Falcon_Perception_ONNX.")
    optimize_bundle(args.quant_method, output_dir, args.embedding_only)


if __name__ == "__main__":
    main()
