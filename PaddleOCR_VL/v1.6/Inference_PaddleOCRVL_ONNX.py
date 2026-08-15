"""Run PaddleOCR-VL image-only merged ONNX Runtime graphs.

The runtime is intentionally self-contained: it discovers all model names and
state layout from ``LLM_Metadata.onnx``, uses only local target assets for the
native chat template, and keeps state OrtValues on the selected provider while
decoding.
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


# Runtime paths and strategy defaults
SCRIPT_DIR          = Path(__file__).resolve().parent
DEFAULT_MODEL_FOLDER = SCRIPT_DIR / "PaddleOCRVL_ONNX"
DEFAULT_IMAGE_PATH  = SCRIPT_DIR / "psyduck_2.png"
STRATEGIES          = ("greedy", "penalty_greedy", "sampling")

# Inference configuration
PROMPT               = "OCR:"
STRATEGY             = "greedy"
MAX_NEW_TOKENS       = 4096
TEMPERATURE          = 0.8
TOP_K                = 20
TOP_P                = 0.95
REPETITION_PENALTY   = 1.0
PROVIDERS            = ("CPUExecutionProvider",)
DEVICE_ID            = 0
EXPECT_NONEMPTY_OUTPUT = False

# Bundle tokenizer contract
CHAT_BOS_TOKEN       = "<|begin_of_sentence|>"
IMAGE_START_TOKEN    = "<|IMAGE_START|>"
IMAGE_TOKEN          = "<|IMAGE_PLACEHOLDER|>"
IMAGE_END_TOKEN      = "<|IMAGE_END|>"

# Graph compatibility constants
_DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "LLM_Metadata.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "vision": "LLM_Vision.onnx",
    "kv_slice": "LLM_KV_Slice.onnx",
    "kv_split2": "LLM_KV_Split2.onnx",
    "kv_concat": "LLM_KV_Concat.onnx",
    "rope_shift": "LLM_RopeShift.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
    "shared_initializers_data": "LLM_SharedInitializers.onnx.data",
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


def parse_args() -> Path:
    global EXPECT_NONEMPTY_OUTPUT

    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL merged ONNX inference from a local bundle."
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_MODEL_FOLDER,
        help="Bundle containing metadata, image graphs, merged graphs, and shared initializers.",
    )
    parser.add_argument(
        "--expect-nonempty-output",
        action="store_true",
        help="Fail when generation produces no visible OCR text.",
    )
    args = parser.parse_args()
    EXPECT_NONEMPTY_OUTPUT = args.expect_nonempty_output
    return args.model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value or ".." in path.parts:
        raise RuntimeError(
            f"Metadata key {metadata_key!r} must contain one safe file name, got {value!r}."
        )
    return value


def _metadata_session(path: Path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def load_metadata(bundle: Path) -> dict[str, str]:
    path = bundle / _DEFAULT_MODEL_FILE_NAMES["metadata"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata carrier: {path}")
    try:
        metadata = dict(_metadata_session(path).get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise RuntimeError(
            "Unable to load LLM_Metadata.onnx before opening large runtime sessions."
        ) from error
    required = (
        "image_token_id",
        "image_token_length",
        "image_start",
        "image_end",
        "max_seq_len",
        "kv_num_tensors",
        "vision_batch_size",
        "input_image_dim",
        "mrope_type",
    )
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise RuntimeError(f"LLM_Metadata.onnx is incomplete; missing: {missing!r}.")
    if metadata["mrope_type"] != "3d":
        raise RuntimeError("This target runtime requires PaddleOCR-VL 3-D mRoPE metadata.")
    return metadata


def load_model_file_names(metadata: dict[str, str]) -> dict[str, str]:
    names = {}
    for role, default in _DEFAULT_MODEL_FILE_NAMES.items():
        metadata_key = f"model_file_name_{role}"
        names[role] = _safe_file_name(metadata.get(metadata_key, default), metadata_key)
    return names


def _parse_providers(value: str) -> list[str]:
    providers = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    return providers


def _provider_device(providers: list[str]):
    if "CUDAExecutionProvider" in providers:
        return "cuda", C.OrtDevice.cuda()
    if "DmlExecutionProvider" in providers:
        return "dml", C.OrtDevice.dml()
    return "cpu", C.OrtDevice.cpu()


def create_session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
    }.items():
        options.add_session_config_entry(key, value)
    return options


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def attach_shared_initializers(
    session_options: onnxruntime.SessionOptions, shared_path: Path
):
    """Attach mmap-backed initializers and retain their owners for session lifetime."""
    shared = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    values: list[onnxruntime.OrtValue] = []
    for initializer in shared.graph.initializer:
        # Logical low-bit tensors cannot be safely represented as NumPy arrays.
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(
                f"Shared initializer {initializer.name!r} has no external-data location."
            )
        data_path = shared_path.parent / _safe_file_name(location, initializer.name)
        if not data_path.is_file():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        except KeyError as error:
            raise RuntimeError(
                f"Unsupported shared initializer dtype for {initializer.name!r}."
            ) from error
        array = np.memmap(
            data_path,
            dtype=dtype,
            mode="r",
            offset=int(external.get("offset", "0")),
            shape=tuple(int(dimension) for dimension in initializer.dims),
        )
        value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        arrays[initializer.name] = array
        values.append(value)
        session_options.add_initializer(initializer.name, value)
    if not values:
        raise RuntimeError("The shared initializer carrier has no attachable tensors.")
    return shared, arrays, values


def create_plain_session(
    path: Path, providers: list[str], conservative_optimization: bool = False
):
    options = create_session_options()
    if conservative_optimization:
        # ORT 1.27's basic reshape rewrite corrupts the target Vision graph;
        # ORT_DISABLE_ALL was validated to preserve correct execution.
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=providers
    )


def create_merged_session(path: Path, shared_path: Path, providers: list[str]):
    options = create_session_options()
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._paddleocr_vl_shared_initializers = references
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
    raise ValueError(f"Unsupported ONNX Runtime tensor type: {type_name}")


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


def _ort_value(
    array: np.ndarray, device_type: str, device_id: int
) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device_type, device_id
    )


def _bind_outputs(binding, names: list[str], ort_device) -> None:
    for name in names:
        binding._iobinding.bind_output(name, ort_device)


def _run(session, binding) -> None:
    options = onnxruntime.RunOptions()
    options.log_severity_level = 4
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
    mapped = {}
    for input_name in inputs:
        if input_name.endswith("vision_hidden_states"):
            mapped["vision_hidden_states"] = input_name
    return mapped


def plan_merged_io(session, strategy: str, state_count: int, is_decode: bool) -> dict:
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    if len(inputs) < state_count or len(outputs) < state_count:
        raise RuntimeError("Merged graph has fewer states than LLM_Metadata.onnx declares.")
    state_in = inputs[:state_count]
    state_out = outputs[:state_count]
    if any(not name.startswith("in_") for name in state_in):
        raise RuntimeError("Merged graph state inputs must lead and use in_* names.")
    if any(not name.startswith("out_") for name in state_out):
        raise RuntimeError("Merged graph state outputs must lead and use out_* names.")
    tail = outputs[state_count:]
    expected_tail = 2 if strategy == "greedy" else 3
    if len(tail) != expected_tail:
        raise RuntimeError(f"Unexpected {strategy} output tail: {tail!r}.")
    token_input = next(
        (name for name in ("embed_input_ids", "input_ids") if name in inputs), None
    )
    if token_input is None:
        raise RuntimeError("Merged graph has no dynamic embedding token input.")
    kv_seq_input = None
    if is_decode:
        kv_seq_input = next(
            (name for name in inputs if name.startswith("decode_kv_seq_len")), None
        )
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


def build_prompt_tokens(
    tokenizer: Tokenizer, query: str, image_token_id: int, metadata: dict[str, str]
) -> np.ndarray:
    if tokenizer.token_to_id(IMAGE_TOKEN) != image_token_id:
        raise RuntimeError("Bundle tokenizer image token does not match LLM_Metadata.onnx.")
    prompt = (
        f"{CHAT_BOS_TOKEN}User: "
        f"{IMAGE_START_TOKEN}{IMAGE_TOKEN}{IMAGE_END_TOKEN}{query}\nAssistant:\n"
    )
    image_token_count = int(metadata["image_token_length"])
    token_ids = [
        int(token_id)
        for token_id in tokenizer.encode(
            prompt.replace(IMAGE_TOKEN, IMAGE_TOKEN * image_token_count),
            add_special_tokens=False,
        ).ids
    ]
    positions = [index for index, token_id in enumerate(token_ids) if token_id == image_token_id]
    if len(positions) != image_token_count or not positions:
        raise RuntimeError("Prompt image-token count does not match export metadata.")
    if positions[-1] - positions[0] + 1 != image_token_count:
        raise RuntimeError("Target image tokens must form one contiguous prompt span.")
    if positions[0] != int(metadata["image_start"]) or positions[-1] + 1 != int(metadata["image_end"]):
        raise RuntimeError("Target prompt image span differs from export metadata.")
    return np.asarray([token_ids], dtype=np.int32)


def load_image(path: Path, input_rank: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
    array = np.ascontiguousarray(array[None, ...])
    if input_rank == 5:
        return array[:, None, ...]
    if input_rank != 4:
        raise ValueError(f"Unsupported image preprocess input rank: {input_rank}.")
    return array


def run_vision(
    preprocess_session,
    vision_session,
    image: np.ndarray,
    device_type: str,
    device_id: int,
    ort_device,
) -> dict[str, onnxruntime.OrtValue]:
    preprocess_input = preprocess_session.get_inputs()[0]
    preprocess_binding = preprocess_session.io_binding()
    preprocess_value = _ort_value(
        image.astype(_np_dtype(preprocess_input.type), copy=False), device_type, device_id
    )
    preprocess_binding.bind_ortvalue_input(preprocess_input.name, preprocess_value)
    preprocess_output_names = [item.name for item in preprocess_session.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_output_names, ort_device)
    _run(preprocess_session, preprocess_binding)
    preprocess_outputs = dict(
        zip(preprocess_output_names, preprocess_binding.get_outputs())
    )

    vision_binding = vision_session.io_binding()
    for input_meta in vision_session.get_inputs():
        value = preprocess_outputs.get(input_meta.name)
        if value is None:
            raise RuntimeError(
                f"Image preprocess graph did not provide vision input {input_meta.name!r}."
            )
        vision_binding.bind_ortvalue_input(input_meta.name, value)
    output_names = [item.name for item in vision_session.get_outputs()]
    _bind_outputs(vision_binding, output_names, ort_device)
    _run(vision_session, vision_binding)
    return dict(zip(output_names, vision_binding.get_outputs()))


def _bind_array(
    binding,
    owners: list[onnxruntime.OrtValue],
    input_meta: dict[str, object],
    name: str,
    array: np.ndarray,
    device_type: str,
    device_id: int,
) -> None:
    value = _ort_value(
        np.asarray(array, dtype=_np_dtype(input_meta[name].type)), device_type, device_id
    )
    owners.append(value)
    binding.bind_ortvalue_input(name, value)


def _static_strategy_values(
    strategy: str,
    input_meta: dict[str, object],
    device_type: str,
) -> list[tuple[str, onnxruntime.OrtValue]]:
    if REPETITION_PENALTY <= 0.0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    if strategy == "sampling":
        if TEMPERATURE <= 0.0 or TOP_K < 1 or not 0.0 < TOP_P <= 1.0:
            raise ValueError("Sampling requires TEMPERATURE > 0, TOP_K >= 1, and 0 < TOP_P <= 1.")
        entries = (
            ("sampling_temperature", np.asarray([TEMPERATURE])),
            ("sampling_top_k", np.asarray(TOP_K)),
            ("sampling_top_p", np.asarray([TOP_P])),
            ("sampling_repetition_penalty", np.asarray([[REPETITION_PENALTY]])),
        )
    elif strategy == "penalty_greedy":
        entries = (
            ("penalty_greedy_repetition_penalty", np.asarray([[REPETITION_PENALTY]])),
        )
    else:
        entries = ()
    return [
        (
            name,
            _ort_value(value.astype(_np_dtype(input_meta[name].type)), device_type, DEVICE_ID),
        )
        for name, value in entries
    ]


def assert_merged_bundle_graphs(bundle: Path, file_names: dict[str, str]) -> None:
    graph_paths = [bundle / file_names["image_preprocess"], bundle / file_names["vision"]]
    for strategy in STRATEGIES:
        graph_paths.extend(
            (
                bundle / file_names[f"image_prefill_{strategy}"],
                bundle / file_names[f"image_decode_{strategy}"],
            )
        )
    for path in graph_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        onnx.load(str(path), load_external_data=False)


def run_generation(
    bundle: Path,
    metadata: dict[str, str],
    file_names: dict[str, str],
    vision_outputs: dict[str, onnxruntime.OrtValue],
    tokenizer: Tokenizer,
    device_type: str,
    ort_device,
    providers: list[str],
) -> str:
    shared_path = bundle / file_names["shared_initializers"]
    shared_data_path = bundle / file_names["shared_initializers_data"]
    if not shared_path.is_file() or not shared_data_path.is_file():
        raise RuntimeError("Merged runtime requires shared initializer metadata and data files.")
    prefill_session = create_merged_session(
        bundle / file_names[f"image_prefill_{STRATEGY}"], shared_path, providers
    )
    decode_session = create_merged_session(
        bundle / file_names[f"image_decode_{STRATEGY}"], shared_path, providers
    )
    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill_session, STRATEGY, state_count, False)
    decode_plan = plan_merged_io(decode_session, STRATEGY, state_count, True)
    prefill_meta = _input_meta(prefill_session)
    decode_meta = _input_meta(decode_session)
    tokens = build_prompt_tokens(
        tokenizer, PROMPT, int(metadata["image_token_id"]), metadata
    )
    prefill_length = int(tokens.shape[1])
    if prefill_length > int(metadata["max_seq_len"]):
        raise ValueError("Prompt is longer than the bundle's configured context length.")
    limit = min(MAX_NEW_TOKENS, int(metadata["max_seq_len"]) - prefill_length)
    if limit < 0:
        raise ValueError("MAX_NEW_TOKENS must be non-negative.")

    prefill_binding = prefill_session.io_binding()
    prefill_owners: list[onnxruntime.OrtValue] = []
    _bind_array(
        prefill_binding,
        prefill_owners,
        prefill_meta,
        prefill_plan["token_input"],
        tokens,
        device_type,
        DEVICE_ID,
    )
    for name, value in (("prefill_ids_len", [prefill_length]), ("prefill_history_len", [0])):
        if name in prefill_plan["inputs"]:
            _bind_array(
                prefill_binding,
                prefill_owners,
                prefill_meta,
                name,
                np.asarray(value),
                device_type,
                DEVICE_ID,
            )
    for vision_name, graph_name in prefill_plan["vision_inputs"].items():
        value = vision_outputs.get(vision_name)
        if value is None:
            raise RuntimeError(f"Image prefill graph requires vision output {vision_name!r}.")
        prefill_binding.bind_ortvalue_input(graph_name, value)
    for name in prefill_plan["state_in"]:
        state_device = "cpu" if device_type == "dml" else device_type
        _bind_array(
            prefill_binding,
            prefill_owners,
            prefill_meta,
            name,
            _zero_from_meta(prefill_meta[name]),
            state_device,
            DEVICE_ID,
        )
    if prefill_plan["save_id_input"] is not None:
        _bind_array(
            prefill_binding,
            prefill_owners,
            prefill_meta,
            prefill_plan["save_id_input"],
            np.zeros((1, 0), dtype=np.int32),
            device_type,
            DEVICE_ID,
        )
    static_prefill = _static_strategy_values(
        STRATEGY,
        prefill_meta,
        device_type,
    )
    for name, value in static_prefill:
        prefill_owners.append(value)
        prefill_binding.bind_ortvalue_input(name, value)
    _bind_outputs(prefill_binding, prefill_plan["outputs"], ort_device)
    prefill_started = time.perf_counter()
    _run(prefill_session, prefill_binding)
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_outputs = prefill_binding.get_outputs()
    prefill_index = {name: index for index, name in enumerate(prefill_plan["outputs"])}
    cached_state = prefill_outputs[:state_count]
    kv_seq_len = prefill_outputs[prefill_index[prefill_plan["kv_seq_output"]]]
    selected_value = prefill_outputs[prefill_index[prefill_plan["token_output"]]]
    saved_ids = (
        prefill_outputs[prefill_index[prefill_plan["save_id_output"]]]
        if prefill_plan["save_id_output"] is not None
        else None
    )
    stop_tokens = {
        int(value)
        for value in metadata.get("stop_token_ids", metadata.get("eos_token_ids", "")).split(",")
        if value
    }
    if not stop_tokens:
        raise RuntimeError("Metadata has no stop/eos token IDs.")
    generated: list[int] = []
    selected_id = int(selected_value.numpy().flat[0])
    if selected_id not in stop_tokens and limit > 0:
        generated.append(selected_id)

    static_decode = _static_strategy_values(
        STRATEGY,
        decode_meta,
        device_type,
    )
    decode_started = time.perf_counter()
    while len(generated) < limit and selected_id not in stop_tokens:
        binding = decode_session.io_binding()
        decode_owners: list[onnxruntime.OrtValue] = []
        # Strategy output token IDs and Embed input IDs are both int32 by contract.
        binding.bind_ortvalue_input(decode_plan["token_input"], selected_value)
        binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        for name, value in zip(decode_plan["state_in"], cached_state):
            binding.bind_ortvalue_input(name, value)
        if decode_plan["save_id_input"] is not None:
            if saved_ids is None:
                raise RuntimeError("Decode strategy requires generated-ID state absent from prefill.")
            binding.bind_ortvalue_input(decode_plan["save_id_input"], saved_ids)
        for name, value in static_decode:
            decode_owners.append(value)
            binding.bind_ortvalue_input(name, value)
        _bind_outputs(binding, decode_plan["outputs"], ort_device)
        _run(decode_session, binding)
        outputs = binding.get_outputs()
        output_index = {name: index for index, name in enumerate(decode_plan["outputs"])}
        cached_state = outputs[:state_count]
        kv_seq_len = outputs[output_index[decode_plan["kv_seq_output"]]]
        selected_value = outputs[output_index[decode_plan["token_output"]]]
        if decode_plan["save_id_output"] is not None:
            saved_ids = outputs[output_index[decode_plan["save_id_output"]]]
        selected_id = int(selected_value.numpy().flat[0])
        if selected_id not in stop_tokens:
            generated.append(selected_id)
    decode_seconds = time.perf_counter() - decode_started
    result = tokenizer.decode(generated, skip_special_tokens=True)
    total_seconds = prefill_seconds + decode_seconds
    print(f"Providers: {decode_session.get_providers()}")
    print(f"Generated Output\n{result}")
    print(
        "Performance\n"
        f"Prefill: {prefill_length} tokens, {prefill_seconds:.3f}s, "
        f"{prefill_length / prefill_seconds if prefill_seconds else 0.0:.2f} tokens/s\n"
        f"Decode: {len(generated)} tokens, {decode_seconds:.3f}s, "
        f"{len(generated) / decode_seconds if decode_seconds else 0.0:.2f} tokens/s\n"
        f"Overall: {prefill_length + len(generated)} tokens, {total_seconds:.3f}s, "
        f"{(prefill_length + len(generated)) / total_seconds if total_seconds else 0.0:.2f} tokens/s"
    )
    return result


def run_inference(model_folder: Path) -> None:
    """Run one PaddleOCR-VL request from the selected ONNX bundle."""
    if MAX_NEW_TOKENS < 0:
        raise ValueError("MAX_NEW_TOKENS must be non-negative.")
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported STRATEGY: {STRATEGY!r}.")
    bundle = model_folder.expanduser().resolve()
    providers = list(PROVIDERS)
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    device_type, device_kind = _provider_device(providers)
    ort_device = C.OrtDevice(device_kind, C.OrtDevice.default_memory(), DEVICE_ID)
    metadata = load_metadata(bundle)
    file_names = load_model_file_names(metadata)
    assert_merged_bundle_graphs(bundle, file_names)
    image = load_image(DEFAULT_IMAGE_PATH.expanduser().resolve(), int(metadata["input_image_dim"]))
    tokenizer = load_tokenizer(bundle)
    preprocess_session = create_plain_session(bundle / file_names["image_preprocess"], providers)
    vision_session = create_plain_session(
        bundle / file_names["vision"], providers, conservative_optimization=True
    )
    vision_outputs = run_vision(
        preprocess_session,
        vision_session,
        image,
        device_type,
        DEVICE_ID,
        ort_device,
    )
    result = run_generation(
        bundle,
        metadata,
        file_names,
        vision_outputs,
        tokenizer,
        device_type,
        ort_device,
        providers,
    )
    if EXPECT_NONEMPTY_OUTPUT and not result.strip():
        raise RuntimeError("Expected nonempty OCR output, but generation was empty.")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()