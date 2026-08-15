"""Run LightOnOCR with image-only merged ONNX Runtime graphs.

The runtime reads its graph names and dimensions from LLM_Metadata.onnx. It
opens only the quantized LightOnOCR bundle and keeps KV and strategy-history
OrtValues on the execution provider between decode steps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnxruntime.capi import _pybind_state as C
from PIL import Image
from tokenizers import Tokenizer


# Runtime paths and demo inputs
SCRIPT_DIR           = Path(__file__).resolve().parent
DEFAULT_MODEL_FOLDER = SCRIPT_DIR / "LightOnOCR_Optimized"

# Inference configuration
INPUT_IMAGES           = [SCRIPT_DIR / "psyduck_2.png"]
QUERY                  = "Transcribe this document exactly."
EXPECT_NONEMPTY_OUTPUT = False

# Generation defaults
STRATEGIES         = ("greedy", "penalty_greedy", "sampling")
STRATEGY           = "greedy"
MAX_NEW_TOKENS     = None
TEMPERATURE        = 0.8
TOP_K              = 20
TOP_P              = 0.95
REPETITION_PENALTY = 1.0

# ONNX Runtime provider and precision controls
ORT_LOG                  = False
ORT_FP16                 = False
ORT_Accelerate_Providers = ["CPUExecutionProvider"]
MAX_THREADS              = 0
DEVICE_ID                = 0

# Bundle tokenizer contract
CHAT_START_TOKEN = "<|im_start|>"
CHAT_END_TOKEN   = "<|im_end|>"
IMAGE_TOKEN      = "<|image_pad|>"

# Graph compatibility constants
_DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "LLM_Metadata.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "vision": "LLM_Vision.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
    "shared_initializers_data": "LLM_SharedInitializers.onnx.data",
    "kv_slice": "LLM_KV_Slice.onnx",
    "kv_split2": "LLM_KV_Split2.onnx",
    "kv_concat": "LLM_KV_Concat.onnx",
    "rope_shift": "LLM_RopeShift.onnx",
    "image_prefill_greedy": "LLM_ImagePrefillGreedy.onnx",
    "image_prefill_penalty_greedy": "LLM_ImagePrefillPenaltyGreedy.onnx",
    "image_prefill_sampling": "LLM_ImagePrefillSampling.onnx",
    "image_decode_greedy": "LLM_ImageDecodeGreedy.onnx",
    "image_decode_penalty_greedy": "LLM_ImageDecodePenaltyGreedy.onnx",
    "image_decode_sampling": "LLM_ImageDecodeSampling.onnx",
}
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_QUANTIZED_OPS = frozenset(
    ("MatMulNBits", "GatherBlockQuantized", "DynamicQuantizeMatMul", "MatMulInteger")
)


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Run LightOnOCR merged ONNX inference.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_MODEL_FOLDER,
        help="Quantized bundle containing merged image graphs and shared initializers.",
    )
    return parser.parse_args().model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise RuntimeError(f"Metadata key {metadata_key!r} must contain a file name, got {value!r}.")
    return value


def _metadata_session(path: Path):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def load_metadata(bundle: Path) -> dict[str, str]:
    path = bundle / _DEFAULT_MODEL_FILE_NAMES["metadata"]
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata carrier: {path}")
    try:
        metadata = dict(_metadata_session(path).get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise RuntimeError("Unable to load LLM_Metadata.onnx before large model sessions.") from error
    required = (
        "image_token_id",
        "image_token_length",
        "input_image_size",
        "input_image_dim",
        "vision_batch_size",
        "max_seq_len",
        "kv_num_tensors",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"LLM_Metadata.onnx is incomplete; missing: {missing!r}.")
    return metadata


def load_model_file_names(metadata: dict[str, str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for role, default in _DEFAULT_MODEL_FILE_NAMES.items():
        key = f"model_file_name_{role}"
        names[role] = _safe_file_name(metadata.get(key, default), key)
    return names


def _parse_providers(value: str) -> list[str]:
    providers = [provider.strip() for provider in value.split(",") if provider.strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    return providers


def _provider_device(providers: list[str]):
    if "CUDAExecutionProvider" in providers:
        return "cuda", C.OrtDevice.cuda()
    if "DmlExecutionProvider" in providers:
        return "dml", C.OrtDevice.dml()
    return "cpu", C.OrtDevice.cpu()


def create_session_options(activations_fp16: bool) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if activations_fp16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if activations_fp16
            else ""
        ),
    }
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    return options


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(
    session_options: onnxruntime.SessionOptions,
    shared_path: Path,
):
    """Attach shared mmap-backed tensor storage and retain its Python lifetimes."""
    shared_model = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    ort_values: list[onnxruntime.OrtValue] = []
    for initializer in shared_model.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {initializer.name!r} has no external-data location.")
        data_path = shared_path.parent / _safe_file_name(location, initializer.name)
        if not data_path.exists():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        except KeyError as error:
            raise RuntimeError(f"Unsupported shared initializer dtype for {initializer.name!r}.") from error
        shape = tuple(int(dimension) for dimension in initializer.dims)
        array = np.memmap(
            data_path,
            dtype=dtype,
            mode="r",
            offset=int(external.get("offset", "0")),
            shape=shape,
        )
        arrays[initializer.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    if not ort_values:
        raise RuntimeError("The shared-initializer carrier has no attachable tensors.")
    return arrays, ort_values


def create_plain_session(
    path: Path,
    providers: list[str],
    activations_fp16: bool,
):
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=create_session_options(activations_fp16),
        providers=providers,
    )


def create_merged_session(
    path: Path,
    shared_path: Path,
    providers: list[str],
    activations_fp16: bool,
):
    options = create_session_options(activations_fp16)
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._lighton_shared_initializers = references
    return session


def _np_dtype(type_name: str):
    for marker, dtype in (
        ("float16", np.float16),
        ("float", np.float32),
        ("uint8", np.uint8),
        ("int8", np.int8),
        ("int64", np.int64),
        ("int32", np.int32),
    ):
        if marker in type_name:
            return dtype
    raise ValueError(f"Unsupported ORT tensor type: {type_name}")


def _input_meta(session) -> dict[str, object]:
    return {item.name: item for item in session.get_inputs()}


def _state_seq_axis(meta) -> int | None:
    axes = [
        index
        for index, dimension in enumerate(meta.shape)
        if index != 0 and not isinstance(dimension, int)
    ]
    return axes[0] if len(axes) == 1 else None


def _zero_from_meta(meta, batch_size: int = 1) -> np.ndarray:
    shape = list(meta.shape)
    sequence_axis = _state_seq_axis(meta)
    for index, dimension in enumerate(shape):
        if index == 0:
            shape[index] = batch_size
        elif sequence_axis is not None and index == sequence_axis:
            shape[index] = 0
        elif not isinstance(dimension, int):
            shape[index] = 1
    return np.zeros(tuple(shape), dtype=_np_dtype(meta.type))


def _ort_value(array: np.ndarray, device_type: str, device_id: int) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device_type, device_id
    )


def _bind_outputs(binding, names: list[str], ort_device) -> None:
    for name in names:
        binding._iobinding.bind_output(name, ort_device)


def _run(session, binding) -> None:
    options = onnxruntime.RunOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    session.run_with_iobinding(binding, run_options=options)


def _strategy_save_id_input(strategy: str, inputs: set[str]) -> str | None:
    if strategy == "greedy":
        return None
    expected = f"{strategy}_previous_ids"
    if expected not in inputs:
        raise RuntimeError(f"Merged graph is missing {expected!r}.")
    return expected


def _vision_input_map(inputs: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for input_name in inputs:
        if input_name.endswith("vision_hidden_states"):
            mapped["vision_hidden_states"] = input_name
        elif "deepstack_feature_" in input_name:
            suffix = input_name.rsplit("deepstack_feature_", 1)[-1]
            if suffix.isdigit():
                mapped[f"deepstack_feature_{suffix}"] = input_name
    return mapped


def plan_merged_io(session, strategy: str, state_count: int, is_decode: bool) -> dict:
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    if len(inputs) < state_count or len(outputs) < state_count:
        raise RuntimeError("Merged graph has fewer state tensors than LLM_Metadata.onnx declares.")
    state_in = inputs[:state_count]
    state_out = outputs[:state_count]
    if any(not name.startswith("in_") for name in state_in):
        raise RuntimeError("Merged graph state inputs must be leading and named in_*.")
    if any(not name.startswith("out_") for name in state_out):
        raise RuntimeError("Merged graph state outputs must be leading and named out_*.")
    tail = outputs[state_count:]
    expected_tail = 2 if strategy == "greedy" else 3
    if len(tail) != expected_tail:
        raise RuntimeError(f"Unexpected {strategy} output tail: {tail!r}.")
    token_input = next((name for name in ("embed_input_ids", "input_ids") if name in inputs), None)
    if token_input is None:
        raise RuntimeError("Merged graph has no token input.")
    kv_seq_input = None
    if is_decode:
        kv_seq_input = next((name for name in inputs if name.startswith("decode_kv_seq_len")), None)
        if kv_seq_input is None:
            raise RuntimeError("Decode graph has no decode KV sequence-length input.")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_in": state_in,
        "state_out": state_out,
        "token_input": token_input,
        "kv_seq_input": kv_seq_input,
        "token_output": tail[0],
        "save_id_output": None if strategy == "greedy" else tail[1],
        "kv_seq_output": tail[-1],
        "save_id_input": _strategy_save_id_input(strategy, set(inputs)),
        "vision_inputs": _vision_input_map(inputs),
    }


def load_tokenizer(bundle: Path) -> Tokenizer:
    path = bundle / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def build_lighton_prompt_tokens(
    tokenizer: Tokenizer,
    query: str,
    image_count: int,
    image_token_id: int,
    image_token_length: int,
    metadata: dict[str, str],
) -> np.ndarray:
    if image_count < 1:
        raise ValueError("LightOnOCR requires at least one input image.")
    if tokenizer.token_to_id(IMAGE_TOKEN) != image_token_id:
        raise RuntimeError("Tokenizer image token ID does not match LLM_Metadata.onnx.")
    prompt = (
        f"{CHAT_START_TOKEN}system{CHAT_END_TOKEN}\n"
        f"{CHAT_START_TOKEN}user\n{IMAGE_TOKEN * image_count}{query}\n{CHAT_END_TOKEN}\n"
        f"{CHAT_START_TOKEN}assistant\n"
    )
    if prompt.count(IMAGE_TOKEN) != image_count:
        raise RuntimeError("Manual LightOnOCR chat prompt has an unexpected image placeholder count.")
    expanded = prompt.replace(IMAGE_TOKEN, IMAGE_TOKEN * image_token_length)
    token_ids = [int(token_id) for token_id in tokenizer.encode(expanded, add_special_tokens=False).ids]
    positions = [index for index, token_id in enumerate(token_ids) if token_id == image_token_id]
    expected_count = image_count * image_token_length
    if len(positions) != expected_count or not positions:
        raise RuntimeError("LightOnOCR chat-template image token expansion does not match metadata.")
    if positions[-1] - positions[0] + 1 != expected_count:
        raise RuntimeError("LightOnOCR image tokens must form one contiguous prompt span.")
    expected_start = metadata.get("image_start")
    expected_end = metadata.get("image_end")
    if expected_start is not None and int(expected_start) != positions[0]:
        raise RuntimeError("LightOnOCR chat-template image start differs from export metadata.")
    if expected_end is not None and int(expected_end) != positions[-1] + 1:
        raise RuntimeError("LightOnOCR chat-template image end differs from export metadata.")
    return np.asarray([token_ids], dtype=np.int64)


def load_images(
    paths: list[Path],
    height: int,
    width: int,
    batch_size: int,
    input_rank: int,
) -> np.ndarray:
    if len(paths) != batch_size:
        raise ValueError(f"Expected {batch_size} image(s), got {len(paths)}.")
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    images = np.empty((batch_size, 3, height, width), dtype=np.uint8)
    for index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            scale = min(width / max(source_width, 1), height / max(source_height, 1))
            resized_width = max(1, min(width, int(round(source_width * scale))))
            resized_height = max(1, min(height, int(round(source_height * scale))))
            if image.size != (resized_width, resized_height):
                image = image.resize((resized_width, resized_height), resampling)
            canvas = Image.new("RGB", (width, height), (127, 127, 127))
            canvas.paste(image, ((width - resized_width) // 2, (height - resized_height) // 2))
            images[index] = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)
    if input_rank == 5:
        images = np.expand_dims(images, axis=1)
    elif input_rank != 4:
        raise ValueError(f"Unsupported LightOnOCR image input rank: {input_rank}.")
    return np.ascontiguousarray(images)


def run_vision(
    preprocess_session,
    vision_session,
    images: np.ndarray,
    device_type: str,
    ort_device,
) -> dict[str, onnxruntime.OrtValue]:
    preprocess_input = preprocess_session.get_inputs()[0]
    value = _ort_value(images.astype(_np_dtype(preprocess_input.type), copy=False), device_type, DEVICE_ID)
    preprocess_binding = preprocess_session.io_binding()
    preprocess_binding.bind_ortvalue_input(preprocess_input.name, value)
    preprocess_names = [item.name for item in preprocess_session.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_names, ort_device)
    _run(preprocess_session, preprocess_binding)
    preprocess_outputs = preprocess_binding.get_outputs()

    vision_binding = vision_session.io_binding()
    vision_values: list[onnxruntime.OrtValue] = []
    for input_meta, output_meta, output_value in zip(
        vision_session.get_inputs(), preprocess_session.get_outputs(), preprocess_outputs
    ):
        if input_meta.type != output_meta.type:
            output_value = _ort_value(
                output_value.numpy().astype(_np_dtype(input_meta.type), copy=False),
                device_type,
                DEVICE_ID,
            )
            vision_values.append(output_value)
        vision_binding.bind_ortvalue_input(input_meta.name, output_value)
    output_names = [item.name for item in vision_session.get_outputs()]
    _bind_outputs(vision_binding, output_names, ort_device)
    _run(vision_session, vision_binding)
    return dict(zip(output_names, vision_binding.get_outputs()))


def _bind_array(
    binding,
    values: list[onnxruntime.OrtValue],
    input_meta: dict[str, object],
    name: str,
    array: np.ndarray,
    device_type: str,
    device_id: int,
) -> None:
    value = _ort_value(np.asarray(array, dtype=_np_dtype(input_meta[name].type)), device_type, device_id)
    values.append(value)
    binding.bind_ortvalue_input(name, value)


def _static_strategy_values(
    strategy: str,
    input_meta: dict[str, object],
    device_type: str,
) -> list[tuple[str, onnxruntime.OrtValue]]:
    if REPETITION_PENALTY <= 0.0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    if strategy == "sampling":
        if TEMPERATURE <= 0.0 or not 0.0 < TOP_P <= 1.0 or TOP_K < 1:
            raise ValueError("Sampling requires TEMPERATURE > 0, TOP_K >= 1, and 0 < TOP_P <= 1.")
        entries = (
            ("sampling_temperature", np.array([TEMPERATURE])),
            ("sampling_top_k", np.array(TOP_K)),
            ("sampling_top_p", np.array([TOP_P])),
            ("sampling_repetition_penalty", np.array([[REPETITION_PENALTY]])),
        )
    elif strategy == "penalty_greedy":
        entries = (("penalty_greedy_repetition_penalty", np.array([[REPETITION_PENALTY]])),)
    else:
        entries = ()
    return [
        (name, _ort_value(array.astype(_np_dtype(input_meta[name].type)), device_type, DEVICE_ID))
        for name, array in entries
    ]


def assert_merged_bundle_graphs(bundle: Path, file_names: dict[str, str]) -> None:
    """Ensure every graph needed by image inference is present and loadable."""
    graph_paths = [bundle / file_names["vision"]]
    for strategy in STRATEGIES:
        graph_paths.extend(
            (
                bundle / file_names[f"image_prefill_{strategy}"],
                bundle / file_names[f"image_decode_{strategy}"],
            )
        )
    for path in graph_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        onnx.load(str(path), load_external_data=False)


def run_generation(
    model_folder: Path,
    metadata: dict[str, str],
    file_names: dict[str, str],
    vision_outputs: dict[str, onnxruntime.OrtValue],
    tokenizer: Tokenizer,
    device_type: str,
    ort_device,
    providers: list[str],
) -> str:
    bundle = model_folder.expanduser().resolve()
    shared_path = bundle / file_names["shared_initializers"]
    shared_data_path = bundle / file_names["shared_initializers_data"]
    if not shared_path.exists() or not shared_data_path.exists():
        raise RuntimeError("Merged runtime requires the shared initializer carrier and data file.")
    activations_fp16 = ORT_FP16
    prefill_path = bundle / file_names[f"image_prefill_{STRATEGY}"]
    decode_path = bundle / file_names[f"image_decode_{STRATEGY}"]
    prefill_session = create_merged_session(prefill_path, shared_path, providers, activations_fp16)
    decode_session = create_merged_session(decode_path, shared_path, providers, activations_fp16)
    print(f"Usable Providers: {decode_session.get_providers()}")

    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill_session, STRATEGY, state_count, is_decode=False)
    decode_plan = plan_merged_io(decode_session, STRATEGY, state_count, is_decode=True)
    prefill_meta = _input_meta(prefill_session)
    decode_meta = _input_meta(decode_session)

    batch_size = int(metadata["vision_batch_size"])
    token_ids = build_lighton_prompt_tokens(
        tokenizer,
        QUERY,
        batch_size,
        int(metadata["image_token_id"]),
        int(metadata["image_token_length"]),
        metadata,
    )
    prefill_tokens = int(token_ids.shape[1])
    limit = max(0, int(metadata["max_seq_len"]) - prefill_tokens)
    if MAX_NEW_TOKENS is not None:
        limit = min(limit, max(0, MAX_NEW_TOKENS))

    prefill_binding = prefill_session.io_binding()
    prefill_values: list[onnxruntime.OrtValue] = []
    _bind_array(
        prefill_binding, prefill_values, prefill_meta, prefill_plan["token_input"],
        token_ids, device_type, DEVICE_ID,
    )
    for name, value in (("prefill_ids_len", [prefill_tokens]), ("prefill_history_len", [0]), ("prefill_cache_len", [0])):
        if name in prefill_plan["inputs"]:
            _bind_array(prefill_binding, prefill_values, prefill_meta, name, np.asarray(value), device_type, DEVICE_ID)
    for vision_name, graph_name in prefill_plan["vision_inputs"].items():
        if vision_name not in vision_outputs:
            raise RuntimeError(f"Image prefill graph requires Vision output {vision_name!r}.")
        prefill_binding.bind_ortvalue_input(graph_name, vision_outputs[vision_name])
    for name in prefill_plan["state_in"]:
        _bind_array(
            prefill_binding, prefill_values, prefill_meta, name,
            _zero_from_meta(prefill_meta[name]), "cpu" if device_type == "dml" else device_type, DEVICE_ID,
        )
    if prefill_plan["save_id_input"] is not None:
        _bind_array(
            prefill_binding, prefill_values, prefill_meta, prefill_plan["save_id_input"],
            np.zeros((1, 0)), device_type, DEVICE_ID,
        )
    for name, value in _static_strategy_values(STRATEGY, prefill_meta, device_type):
        prefill_values.append(value)
        prefill_binding.bind_ortvalue_input(name, value)
    _bind_outputs(prefill_binding, prefill_plan["outputs"], ort_device)

    prefill_start = time.time()
    _run(prefill_session, prefill_binding)
    prefill_elapsed = time.time() - prefill_start
    prefill_outputs = prefill_binding.get_outputs()
    prefill_positions = {name: index for index, name in enumerate(prefill_plan["outputs"])}
    cached_state = prefill_outputs[:state_count]
    kv_seq_len = prefill_outputs[prefill_positions[prefill_plan["kv_seq_output"]]]
    selected_value = prefill_outputs[prefill_positions[prefill_plan["token_output"]]]
    saved_ids = (
        prefill_outputs[prefill_positions[prefill_plan["save_id_output"]]]
        if prefill_plan["save_id_output"] is not None
        else None
    )

    stop_tokens = {
        int(token_id) for token_id in metadata.get("stop_token_ids", metadata.get("eos_token_ids", "")).split(",")
        if token_id
    }
    if not stop_tokens:
        raise RuntimeError("LLM_Metadata.onnx has no stop token IDs.")
    generated: list[int] = []
    selected_id = int(selected_value.numpy().flat[0])
    if selected_id not in stop_tokens and limit > 0:
        generated.append(selected_id)
        print(tokenizer.decode([selected_id]), end="", flush=True)

    decode_static = _static_strategy_values(STRATEGY, decode_meta, device_type)
    decode_start = time.time()
    while len(generated) < limit and selected_id not in stop_tokens:
        binding = decode_session.io_binding()
        decode_values: list[onnxruntime.OrtValue] = []
        token_array = np.asarray(selected_value.numpy(), dtype=_np_dtype(decode_meta[decode_plan["token_input"]].type))
        _bind_array(
            binding, decode_values, decode_meta, decode_plan["token_input"],
            token_array, device_type, DEVICE_ID,
        )
        binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        for name, value in zip(decode_plan["state_in"], cached_state):
            binding.bind_ortvalue_input(name, value)
        if decode_plan["save_id_input"] is not None:
            if saved_ids is None:
                raise RuntimeError("Strategy decode requires generated-ID state missing from prefill output.")
            binding.bind_ortvalue_input(decode_plan["save_id_input"], saved_ids)
        for name, value in decode_static:
            binding.bind_ortvalue_input(name, value)
        _bind_outputs(binding, decode_plan["outputs"], ort_device)
        _run(decode_session, binding)
        outputs = binding.get_outputs()
        positions = {name: index for index, name in enumerate(decode_plan["outputs"])}
        cached_state = outputs[:state_count]
        kv_seq_len = outputs[positions[decode_plan["kv_seq_output"]]]
        selected_value = outputs[positions[decode_plan["token_output"]]]
        if decode_plan["save_id_output"] is not None:
            saved_ids = outputs[positions[decode_plan["save_id_output"]]]
        selected_id = int(selected_value.numpy().flat[0])
        if selected_id not in stop_tokens:
            generated.append(selected_id)
            print(tokenizer.decode([selected_id]), end="", flush=True)
    decode_elapsed = time.time() - decode_start

    result = tokenizer.decode(generated, skip_special_tokens=True)
    total_elapsed = prefill_elapsed + decode_elapsed
    total_tokens = prefill_tokens + len(generated)
    prefill_tokens_per_second = prefill_tokens / prefill_elapsed if prefill_elapsed else 0.0
    decode_tokens_per_second = len(generated) / decode_elapsed if decode_elapsed else 0.0
    overall_tokens_per_second = total_tokens / total_elapsed if total_elapsed else 0.0
    print(
        "\n\nGenerated Output\n"
        f"{result}\n\n"
        "Performance\n"
        f"{'Stage':<10} {'Tokens':>8} {'Time (s)':>10} {'Tokens/s':>12}\n"
        f"{'-' * 10} {'-' * 8:>8} {'-' * 10:>10} {'-' * 12:>12}\n"
        f"{'Prefill':<10} {prefill_tokens:>8} {prefill_elapsed:>10.3f} {prefill_tokens_per_second:>12.2f}\n"
        f"{'Decode':<10} {len(generated):>8} {decode_elapsed:>10.3f} {decode_tokens_per_second:>12.2f}\n"
        f"{'Overall':<10} {total_tokens:>8} {total_elapsed:>10.3f} {overall_tokens_per_second:>12.2f}"
    )
    return result


def run_inference(model_folder: Path) -> None:
    """Run one LightOnOCR request from the selected ONNX bundle."""
    model_folder = model_folder.expanduser().resolve()
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported STRATEGY: {STRATEGY!r}.")
    providers = _parse_providers(",".join(ORT_Accelerate_Providers))
    device_type, ort_device_kind = _provider_device(providers)
    ort_device = C.OrtDevice(ort_device_kind, C.OrtDevice.default_memory(), DEVICE_ID)

    metadata = load_metadata(model_folder)
    file_names = load_model_file_names(metadata)
    assert_merged_bundle_graphs(model_folder, file_names)
    image_paths = [path.expanduser().resolve() for path in INPUT_IMAGES]
    image_size = [int(value) for value in metadata["input_image_size"].split(",")]
    batch_size = int(metadata["vision_batch_size"])
    images = load_images(
        image_paths,
        image_size[0],
        image_size[1],
        batch_size,
        int(metadata["input_image_dim"]),
    )

    activations_fp16 = ORT_FP16
    preprocess_session = create_plain_session(
        model_folder / file_names["image_preprocess"], providers, activations_fp16
    )
    vision_session = create_plain_session(
        model_folder / file_names["vision"], providers, activations_fp16
    )
    tokenizer = load_tokenizer(model_folder)
    vision_outputs = run_vision(
        preprocess_session, vision_session, images, device_type, ort_device
    )
    print(f"\nTest Question: {QUERY}\nLLM Answering:")
    result = run_generation(
        model_folder, metadata, file_names, vision_outputs, tokenizer, device_type, ort_device, providers
    )
    if EXPECT_NONEMPTY_OUTPUT and not result.strip():
        raise RuntimeError("Expected nonempty OCR output, but generation was empty.")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()