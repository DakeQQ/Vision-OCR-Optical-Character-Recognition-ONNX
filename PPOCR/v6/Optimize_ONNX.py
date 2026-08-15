import os
import gc
import glob
import shutil
import sys
import onnx
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (
    make_standalone_quantization_plan,
    quantize_dynamic_int8,
    quantize_weight_only,
)


# ==============================================================================
# Path Settings
# ==============================================================================
SOURCE_BUNDLE_DIR   = os.path.join(SCRIPT_DIR, 'PPOCRv6_ONNX')
OPTIMIZED_BUNDLE_DIR = os.path.join(SCRIPT_DIR, 'PPOCRv6_Optimized')
# Backward-compatible configuration aliases.
original_folder_path = SOURCE_BUNDLE_DIR
quanted_folder_path   = OPTIMIZED_BUNDLE_DIR


# ==============================================================================
# Lazy Settings (set one True for auto-select, or both False for manual)
# ==============================================================================
lazy_setting_CPU = True                   # Auto-select CPU settings.
lazy_setting_GPU = False                  # Auto-select GPU settings.
use_openvino     = False                  # Use OpenVINO-specific optimization settings.
SAVE_TWO_PARTS   = False                  # Keep optimized models in split external-data parts.
upgrade_opset    = 0                      # Set a positive opset to upgrade during optimization.


# ==============================================================================
# Model List  (matches the files emitted by Export_PPOCRv6.py)
# ==============================================================================
model_names = [                           # Recommended dtype:
    "PPOCRv6_DocOri",                     # [float32, float16]   conv classifier
    "PPOCRv6_Unwarp",                     # [float32, float16]   conv + GridSample
    "PPOCRv6_Det",                        # [float32, float16]   conv backbone + neck
    "PPOCRv6_DBPost",                     # [float32]            hand-built DB postprocess (Loop/NonZero/Compress)
    "PPOCRv6_Rec",                        # [int4, int8, dynamic, float32, float16]  conv + textline-If + SVTR/CTC MatMul
]


# ==============================================================================
# Manual Quantization Settings
# ==============================================================================
quant_int4         = False
quant_int8         = False
quant_dynamic      = False
quant_float16      = False
keep_io_dtype      = True
fp16_op_block_list = [
    'DynamicQuantizeLinear',
    'DequantizeLinear',
    'DynamicQuantizeMatMul',
    'Range',
    'MatMulIntegerToFloat',
    'GridSample',                               # keep the UVDoc warp in float32
    'Resize',
]


# ==============================================================================
# Recognition Quantization Settings  (recognition MatMuls only)
# ==============================================================================
algorithm        = "AFFINE_REFINE_V2"     # AFFINE_REFINE_V2 | DEFAULT | RTN | HQQ | k_quant (Q4 only).
bits             = 8                      # Q4/Q8 weight-only width; dynamic uses its INT8 controls below.
block_size       = 32
accuracy_level   = 4                      # 0=default, 1=fp32, 2=fp16, 3=bf16, 4=int8.
quant_symmetric  = False
nodes_to_exclude = None
nodes_to_include = None
dynamic_weight_type = "QInt8"             # QInt8 | QUInt8.
dynamic_per_channel = True
dynamic_reduce_range = False


# ==============================================================================
# Per-Model Target Dtype Mapping (CPU)
# ==============================================================================
CPU_MODEL_DTYPE = {
    "PPOCRv6_DocOri":   "float32",
    "PPOCRv6_Unwarp":   "float32",
    "PPOCRv6_Det":      "float32",
    "PPOCRv6_DBPost":   "float32",        # control-flow graph, copied verbatim
    "PPOCRv6_Rec":      "float32",          
}


# ==============================================================================
# Per-Model Target Dtype Mapping (GPU)
# ==============================================================================
GPU_MODEL_DTYPE = {
    "PPOCRv6_DocOri":   "float16",
    "PPOCRv6_Unwarp":   "float16",
    "PPOCRv6_Det":      "float16",
    "PPOCRv6_DBPost":   "float16",        # fp16 overflows the CC labels + integral image
    "PPOCRv6_Rec":      "float16",        
}

if lazy_setting_CPU and lazy_setting_GPU:
    raise ValueError("Only one of lazy_setting_CPU or lazy_setting_GPU can be True.")


# ==============================================================================
# Helper Functions
# ==============================================================================
def _is_matmul_block(name):
    """Only the recognition head carries weight MatMuls worth quantizing."""
    return name == "PPOCRv6_Rec"


def _is_postprocess_graph(name):
    """The DB postprocess graph is hand-built control flow, not a CNN/transformer."""
    return name == "PPOCRv6_DBPost"


def _is_detector_graph(name):
    """The detector has a CUDA-sensitive dynamic image resize."""
    return name == "PPOCRv6_Det"


def _unique_onnx_name(prefix, names):
    candidate = prefix
    suffix = 0
    while candidate in names:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    names.add(candidate)
    return candidate


def _rewrite_detector_resize_for_cuda(model):
    """Cast the detector image to float before its dynamic CUDA Resize node."""
    graph = model.graph
    graph_input_names = {value.name for value in graph.input}
    resize_candidates = [
        (index, node)
        for index, node in enumerate(graph.node)
        if node.op_type == "Resize" and node.input[0] in graph_input_names
    ]
    if not resize_candidates:
        return 0
    if len(resize_candidates) != 1:
        details = ", ".join(
            f"{node.name or '<unnamed>'}@{index}"
            for index, node in resize_candidates
        )
        raise RuntimeError(
            "PPOCRv6_Det contains multiple input-driven Resize nodes for the "
            f"CUDA-safe rewrite: {details}"
        )

    tensor_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    }
    tensor_names.update(value.name for value in graph.input)
    tensor_names.update(value.name for value in graph.output)
    tensor_names.update(initializer.name for initializer in graph.initializer)
    node_names = {node.name for node in graph.node if node.name}

    resize_index, resize_node = resize_candidates[0]
    cast_output = _unique_onnx_name("cuda_safe_detector_resize_input", tensor_names)
    cast_node = onnx.helper.make_node(
        "Cast",
        [resize_node.input[0]],
        [cast_output],
        name=_unique_onnx_name("cuda_safe_detector_resize_cast", node_names),
        to=onnx.TensorProto.FLOAT,
    )
    resize_node.input[0] = cast_output

    rewritten_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(rewritten_nodes[:resize_index])
    graph.node.append(cast_node)
    graph.node.extend(rewritten_nodes[resize_index:])

    # The offline optimizer recorded uint8 intermediate types before this cast.
    # They are optional metadata and would make ORT reject the corrected graph.
    del graph.value_info[:]
    return 1


def _rewrite_dbpost_compress_for_cuda(model):
    """Replace DBPost's empty-input-unsafe Compress filters with Gather nodes."""
    graph = model.graph
    compress_nodes = [node for node in graph.node if node.op_type == "Compress"]
    if not compress_nodes:
        return 0

    for node in compress_nodes:
        axis = next(
            (attribute.i for attribute in node.attribute if attribute.name == "axis"),
            None,
        )
        if axis != 0 or len(node.input) != 2 or len(node.output) != 1:
            raise RuntimeError(
                "PPOCRv6_DBPost contains an unsupported Compress node for the "
                f"CUDA-safe rewrite: {node.name or '<unnamed>'}"
            )

    tensor_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    }
    tensor_names.update(value.name for value in graph.input)
    tensor_names.update(value.name for value in graph.output)
    tensor_names.update(initializer.name for initializer in graph.initializer)
    node_names = {node.name for node in graph.node if node.name}

    axis_name = _unique_onnx_name("cuda_safe_compress_axis", tensor_names)
    graph.initializer.append(onnx.helper.make_tensor(
        axis_name, onnx.TensorProto.INT64, [1], [0]
    ))

    rewritten_nodes = []
    for index, node in enumerate(graph.node, start=1):
        if node.op_type != "Compress":
            rewritten_nodes.append(node)
            continue

        prefix = f"cuda_safe_compress_{index}"
        nonzero_output = _unique_onnx_name(f"{prefix}_nonzero_output", tensor_names)
        indices_output = _unique_onnx_name(f"{prefix}_indices_output", tensor_names)
        rewritten_nodes.extend((
            onnx.helper.make_node(
                "NonZero",
                [node.input[1]],
                [nonzero_output],
                name=_unique_onnx_name(f"{prefix}_nonzero", node_names),
            ),
            onnx.helper.make_node(
                "Squeeze",
                [nonzero_output, axis_name],
                [indices_output],
                name=_unique_onnx_name(f"{prefix}_squeeze", node_names),
            ),
            onnx.helper.make_node(
                "Gather",
                [node.input[0], indices_output],
                list(node.output),
                axis=0,
                name=_unique_onnx_name(f"{prefix}_gather", node_names),
            ),
        ))

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    return len(compress_nodes)


def _validate_target_dtype(model_name, target_dtype):
    if target_dtype in ("int4", "int8", "dynamic") and not _is_matmul_block(model_name):
        raise ValueError(
            f"{model_name} does not support {target_dtype} quantization; "
            "use float16 or float32."
        )


def _opt_level(name):
    # Level 2 emits FusedConv(activation="HardSigmoid"), which CUDA EP cannot load.
    return 1


def _num_heads(name):
    return 6 if name == "PPOCRv6_Rec" else 0    # SVTR: hidden 192 / head_dim 32


def _hidden_size(name):
    return 192 if name == "PPOCRv6_Rec" else 0


def _remaining_float_matrix_weights(model) -> list[str]:
    """Return direct constant matrix operands that escaped weight-only quantization."""
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    remaining = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        for input_index, input_name in enumerate(node.input[:2]):
            weight = initializers.get(input_name)
            if (
                weight is not None
                and len(weight.dims) == 2
                and weight.data_type in float_types
            ):
                remaining.append(
                    f"{node.name or node.op_type}[{input_index}] <- "
                    f"{weight.name}{tuple(weight.dims)}"
                )
    return remaining


def _recognition_quantization_plan(method):
    """Build the shared-helper plan for PP-OCRv6 recognition MatMuls."""
    return make_standalone_quantization_plan(
        method,
        algorithm=algorithm,
        block_size=block_size,
        accuracy_level=accuracy_level,
        symmetric=quant_symmetric,
        quant_format="QOperator",
        dynamic_weight_type=dynamic_weight_type,
        per_channel=dynamic_per_channel,
        reduce_range=dynamic_reduce_range,
        nodes_to_exclude=nodes_to_exclude,
        nodes_to_include=nodes_to_include,
    )


def _matmul_nbits_widths(model) -> set[int]:
    return {
        int(attribute.i)
        for node in model.graph.node
        if node.op_type == "MatMulNBits"
        for attribute in node.attribute
        if attribute.name == "bits"
    }


def _validate_recognition_weight_only_model(model, model_name, selected_bits):
    emitted_bits = _matmul_nbits_widths(model)
    if emitted_bits != {selected_bits}:
        raise RuntimeError(
            f"{model_name} requested Q{selected_bits} MatMulNBits but emitted "
            f"{sorted(emitted_bits) or 'no MatMulNBits nodes'}"
        )
    remaining = _remaining_float_matrix_weights(model)
    if remaining:
        preview = "; ".join(remaining[:8])
        suffix = "" if len(remaining) <= 8 else f"; +{len(remaining) - 8} more"
        raise RuntimeError(
            f"{model_name} retained {len(remaining)} constant float matrix "
            f"operand(s) after Q{selected_bits} quantization: {preview}{suffix}"
        )


def _quantize_recognition_weight_only(model_path, quanted_model_path, model_name, selected_bits):
    plan = _recognition_quantization_plan(f"Q{selected_bits}")
    print(
        f"Quantizing recognition MatMuls with {plan.algo} Q{selected_bits} "
        f"(block={plan.block_size})..."
    )
    quantize_weight_only(
        str(model_path), str(quanted_model_path), plan, selected_bits, external=True
    )
    model = onnx.load(str(quanted_model_path))
    _validate_recognition_weight_only_model(model, model_name, selected_bits)
    del model
    gc.collect()


def _quantize_recognition_dynamic(model_path, quanted_model_path, model_name):
    plan = _recognition_quantization_plan("DYNAMIC")
    print(
        f"Quantizing recognition MatMuls with {plan.algo} dynamic "
        f"{plan.dynamic_weight_type} (per_channel={plan.per_channel})..."
    )
    quantize_dynamic_int8(
        str(model_path), str(quanted_model_path), plan, external=True
    )
    model = onnx.load(str(quanted_model_path))
    if plan.algo == "AFFINE_REFINE_V2" and not any(
        node.op_type == "DynamicQuantizeMatMul" for node in model.graph.node
    ):
        raise RuntimeError(
            f"{model_name} requested AFFINE_REFINE_V2 dynamic quantization but "
            "emitted no DynamicQuantizeMatMul nodes."
        )
    remaining = _remaining_float_matrix_weights(model)
    if remaining:
        preview = "; ".join(remaining[:8])
        suffix = "" if len(remaining) <= 8 else f"; +{len(remaining) - 8} more"
        raise RuntimeError(
            f"{model_name} retained {len(remaining)} constant float matrix "
            f"operand(s) after dynamic quantization: {preview}{suffix}"
        )
    del model
    gc.collect()


# ==============================================================================
# Core Processing Function
# ==============================================================================
def process_single_model(
    model_path,
    quanted_model_path,
    model_name,
    bits,
    block_size,
    quant_int4_flag,
    quant_int8_flag,
    quant_dynamic_flag,
    quant_float16_flag,
    keep_io_flag,
    op_block_list,
):
    """Quantize / optimize / slim a single PP-OCRv6 ONNX graph."""
    be_optimized = False

    # ------------------------------------------------------------------
    # Branch 0: DB postprocess (hand-built control-flow graph)
    # ------------------------------------------------------------------
    # Loop / NonZero / Gather / ScatterElements with outer-scope Loop-body
    # references, plus connected-component labels and an integral image that
    # exceed the float16 range.  It is already minimal and numerically exact, so
    # leave it float32 and avoid generic graph rewrites that could perturb the
    # transformer optimizer, float16 conversion or onnxslim rewriting the control
    # flow.  The only targeted rewrite replaces Compress, whose CUDA kernel cannot
    # accept an empty condition tensor. Returns early so none of the optimization
    # passes below touch it.
    if _is_postprocess_graph(model_name):
        print("Postprocess control-flow graph: applying CUDA-safe empty-filter rewrite...")
        model = onnx.load(model_path)
        rewritten_count = _rewrite_dbpost_compress_for_cuda(model)
        onnx.checker.check_model(model)
        onnx.save(model, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        print(f"Replaced {rewritten_count} Compress node(s) with NonZero + Gather.")
        del model
        gc.collect()
        return

    # ------------------------------------------------------------------
    # Branch 1: weight-only or dynamic quantization (recognition MatMuls)
    # ------------------------------------------------------------------
    if quant_dynamic_flag and _is_matmul_block(model_name):
        _quantize_recognition_dynamic(model_path, quanted_model_path, model_name)

    elif (quant_int4_flag or quant_int8_flag) and _is_matmul_block(model_name):
        _quantize_recognition_weight_only(
            model_path, quanted_model_path, model_name, bits
        )

    # ------------------------------------------------------------------
    # Branch 2: Float16 conversion
    # ------------------------------------------------------------------
    elif quant_float16_flag:
        print("Optimizing model before Float16 conversion...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        print("Converting model to Float16...")
        model.convert_float_to_float16(
            keep_io_types=keep_io_flag,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=op_block_list,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Branch 3: Float32 (optimize only, lossless)
    # ------------------------------------------------------------------
    else:
        print("Target dtype is float32: optimizing without quantization...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Post-quantization optimization pass
    # ------------------------------------------------------------------
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        model = optimize_model(
            quanted_model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Slim pass
    # ------------------------------------------------------------------
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=SAVE_TWO_PARTS,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Optional opset upgrade / final save
    # ------------------------------------------------------------------
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.load(quanted_model_path)
            converted = onnx.version_converter.convert_version(m, upgrade_opset)
            onnx.save(converted, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m, converted
            gc.collect()
        except Exception as exc:
            print(f"Could not upgrade opset: {exc}. Keeping original opset.")
            m = onnx.load(quanted_model_path)
            onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m
            gc.collect()
    else:
        m = onnx.load(quanted_model_path)
        onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del m
        gc.collect()

    if _is_detector_graph(model_name):
        model = onnx.load(quanted_model_path)
        rewritten_count = _rewrite_detector_resize_for_cuda(model)
        if rewritten_count:
            onnx.checker.check_model(model)
            onnx.save(model, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            print(
                "Replaced the detector uint8 Resize input with a float32 cast "
                f"({rewritten_count} node)."
            )
        del model
        gc.collect()


def optimize_bundle() -> None:
    """Optimize the PP-OCRv6 graph set and restore its runtime assets."""
    os.makedirs(OPTIMIZED_BUNDLE_DIR, exist_ok=True)
    selected_bits = bits
    selected_quant_int4 = quant_int4
    selected_quant_int8 = quant_int8
    selected_quant_dynamic = quant_dynamic
    selected_quant_float16 = quant_float16
    selected_keep_io_dtype = keep_io_dtype

    for model_name in model_names:
        print(f"\n--- Processing model: {model_name} ---")

        if lazy_setting_GPU:
            target_dtype = GPU_MODEL_DTYPE.get(model_name, "float16")
            selected_keep_io_dtype = False
        elif lazy_setting_CPU:
            target_dtype = CPU_MODEL_DTYPE.get(model_name, "float32")
            selected_keep_io_dtype = True
        else:
            target_dtype = None

        if target_dtype:
            _validate_target_dtype(model_name, target_dtype)
            selected_quant_int4 = target_dtype == "int4"
            selected_quant_int8 = target_dtype == "int8"
            selected_quant_dynamic = target_dtype == "dynamic"
            selected_quant_float16 = target_dtype == "float16"

        if selected_quant_int4:
            selected_bits = 4
        elif selected_quant_int8:
            selected_bits = 8

        selected_quantization_count = sum((
            selected_quant_int4,
            selected_quant_int8,
            selected_quant_dynamic,
        ))
        if selected_quantization_count > 1:
            raise ValueError(
                "Select only one of quant_int4, quant_int8, or quant_dynamic "
                "for PP-OCRv6 recognition."
            )

        print(f"Selected target dtype for {model_name}: {target_dtype}")
        print(
            f"quant_int4={selected_quant_int4}, quant_int8={selected_quant_int8}, "
            f"quant_dynamic={selected_quant_dynamic}, "
            f"quant_float16={selected_quant_float16}, "
            f"keep_io_dtype={selected_keep_io_dtype}"
        )

        model_path = os.path.join(SOURCE_BUNDLE_DIR, f"{model_name}.onnx")
        quanted_model_path = os.path.join(OPTIMIZED_BUNDLE_DIR, f"{model_name}.onnx")
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue

        process_single_model(
            model_path,
            quanted_model_path,
            model_name,
            selected_bits,
            block_size,
            selected_quant_int4,
            selected_quant_int8,
            selected_quant_dynamic,
            selected_quant_float16,
            selected_keep_io_dtype,
            fp16_op_block_list,
        )

    # Mirror the CTC character list so the optimized directory is self-contained.
    char_list_src = os.path.join(SOURCE_BUNDLE_DIR, 'rec_char_list.npy')
    char_list_dst = os.path.join(OPTIMIZED_BUNDLE_DIR, 'rec_char_list.npy')
    if os.path.exists(char_list_src):
        shutil.copy2(char_list_src, char_list_dst)
        print(f"Copied {char_list_src} -> {char_list_dst}")
    else:
        print(f"Warning: {char_list_src} not found; optimized set will lack the char list.")

    print("Cleaning up temporary *.onnx.data files...")
    for file_path in glob.glob(os.path.join(OPTIMIZED_BUNDLE_DIR, '*.onnx.data')):
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as exc:
            print(f"Error deleting {file_path}: {exc}")
    print("--- All models processed successfully! ---")


def main() -> None:
    optimize_bundle()


if __name__ == "__main__":
    main()
