"""Run the standalone UnlimitedOCR image-only merged ONNX bundle."""

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
DEFAULT_MODEL_FOLDER = SCRIPT_DIR / "UnlimitedOCR_ONNX"
DEFAULT_IMAGE        = SCRIPT_DIR / "psyduck_2.png"

# Inference configuration
QUERY                  = "<image>\nFree OCR."
STRATEGY               = "greedy"
MAX_NEW_TOKENS         = 4096
TEMPERATURE            = 0.8
TOP_K                  = 20
TOP_P                  = 0.95
REPETITION_PENALTY     = 1.0
PROVIDERS              = ("CPUExecutionProvider",)
DEVICE_ID              = 0
ORT_FP16               = False
ORT_LOG                = False
EXPECT_NONEMPTY_OUTPUT = False
STRATEGIES             = ("greedy", "penalty_greedy", "sampling")

# Graph compatibility constants
_DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "LLM_Metadata.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "vision": "LLM_Vision.onnx",
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
    parser = argparse.ArgumentParser(description="Run UnlimitedOCR merged ONNX inference.")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    return parser.parse_args().model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise RuntimeError(f"Metadata key {metadata_key!r} must be a file name, got {value!r}.")
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
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata carrier: {path}")
    try:
        metadata = dict(_metadata_session(path).get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise RuntimeError("Unable to load LLM_Metadata.onnx before large model sessions.") from error
    required = (
        "image_token_id", "image_token_length", "input_image_size", "input_image_dim",
        "vision_batch_size", "max_seq_len", "kv_num_tensors", "stop_token_ids",
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


def _provider_device(providers: list[str]):
    if "CUDAExecutionProvider" in providers:
        return "cuda", C.OrtDevice.cuda()
    if "DmlExecutionProvider" in providers:
        return "dml", C.OrtDevice.dml()
    return "cpu", C.OrtDevice.cpu()


def create_session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16 else ""
        ),
    }
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    return options


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(options: onnxruntime.SessionOptions, shared_path: Path):
    """Attach mmap-backed shared tensors and retain Python owners for session life."""
    shared_model = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    values: list[onnxruntime.OrtValue] = []
    for initializer in shared_model.graph.initializer:
        # Keep logical low-bit tensors in ORT-managed external storage.
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {initializer.name!r} has no external-data location.")
        data_path = shared_path.parent / _safe_file_name(location, initializer.name)
        if not data_path.is_file():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        except KeyError as error:
            raise RuntimeError(f"Unsupported shared dtype for {initializer.name!r}.") from error
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
        options.add_initializer(initializer.name, value)
    if not values:
        raise RuntimeError("The shared-initializer carrier has no attachable tensors.")
    return arrays, values


def create_plain_session(path: Path, providers: list[str]):
    return onnxruntime.InferenceSession(
        str(path), sess_options=create_session_options(), providers=providers
    )


def create_merged_session(path: Path, shared_path: Path, providers: list[str]):
    options = create_session_options()
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._unlimitedocr_shared_initializers = references
    return session


def _np_dtype(type_name: str):
    for marker, dtype in (
        ("float16", np.float16), ("float", np.float32), ("uint8", np.uint8),
        ("int8", np.int8), ("int64", np.int64), ("int32", np.int32),
    ):
        if marker in type_name:
            return dtype
    raise ValueError(f"Unsupported ORT tensor type: {type_name}")


def _input_meta(session) -> dict[str, object]:
    return {item.name: item for item in session.get_inputs()}


def _state_seq_axis(meta) -> int | None:
    dynamic_axes = [
        index for index, dimension in enumerate(meta.shape)
        if index != 0 and not isinstance(dimension, int)
    ]
    return dynamic_axes[0] if len(dynamic_axes) == 1 else None


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


def plan_merged_io(session, strategy: str, state_count: int, is_decode: bool) -> dict:
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    if len(inputs) < state_count or len(outputs) < state_count:
        raise RuntimeError("Merged graph has fewer state tensors than its metadata declares.")
    state_in, state_out = inputs[:state_count], outputs[:state_count]
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
            raise RuntimeError("Decode graph has no KV sequence-length input.")
    save_id_input = None if strategy == "greedy" else f"{strategy}_previous_ids"
    if save_id_input is not None and save_id_input not in inputs:
        raise RuntimeError(f"Merged graph is missing {save_id_input!r}.")
    vision_input = next((name for name in inputs if name.endswith("vision_hidden_states")), None)
    return {
        "inputs": inputs, "outputs": outputs, "state_in": state_in, "state_out": state_out,
        "token_input": token_input, "kv_seq_input": kv_seq_input, "token_output": tail[0],
        "save_id_input": save_id_input, "save_id_output": None if strategy == "greedy" else tail[1],
        "kv_seq_output": tail[-1], "vision_input": vision_input,
    }


def load_tokenizer(bundle: Path) -> Tokenizer:
    """Load the export-normalized Llama backend directly from the ONNX bundle."""
    path = bundle / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle tokenizer: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    if len(tokenizer.encode("", add_special_tokens=True).ids) != 1:
        raise RuntimeError("UnlimitedOCR bundle tokenizer must add exactly one BOS token.")
    return tokenizer


def build_native_prompt_tokens(tokenizer: Tokenizer, query: str, image_token_id: int,
                               image_token_length: int) -> np.ndarray:
    """Apply UnlimitedOCR's native plain prompt contract and expand one image slot."""
    prompt = query.strip()
    image_marker = "<image>"
    if prompt.count(image_marker) != 1:
        raise ValueError("UnlimitedOCR image inference requires exactly one '<image>' marker in QUERY.")
    before, after = prompt.split(image_marker)
    before_ids = tokenizer.encode(before, add_special_tokens=False).ids
    after_ids = tokenizer.encode(after, add_special_tokens=False).ids
    bos_ids = tokenizer.encode("", add_special_tokens=True).ids
    ids = [int(bos_ids[0]), *map(int, before_ids)]
    ids.extend([image_token_id] * image_token_length)
    ids.extend(map(int, after_ids))
    positions = [index for index, token_id in enumerate(ids) if token_id == image_token_id]
    if len(positions) != image_token_length or positions[-1] - positions[0] + 1 != image_token_length:
        raise RuntimeError("Native UnlimitedOCR image tokens are not one contiguous expected span.")
    return np.asarray([ids], dtype=np.int64)


def load_image(path: Path, height: int, width: int, input_rank: int) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
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
        values = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)[None]
    if input_rank == 5:
        values = np.expand_dims(values, axis=1)
    elif input_rank != 4:
        raise ValueError(f"Unsupported image preprocess input rank: {input_rank}.")
    return np.ascontiguousarray(values)


def run_vision(preprocess_session, vision_session, image: np.ndarray, device_type: str,
               device_id: int, ort_device) -> onnxruntime.OrtValue:
    preprocess_input = preprocess_session.get_inputs()[0]
    preprocess_binding = preprocess_session.io_binding()
    preprocess_binding.bind_ortvalue_input(
        preprocess_input.name,
        _ort_value(image.astype(_np_dtype(preprocess_input.type), copy=False), device_type, device_id),
    )
    preprocess_output_names = [item.name for item in preprocess_session.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_output_names, ort_device)
    _run(preprocess_session, preprocess_binding)
    preprocess_outputs = preprocess_binding.get_outputs()
    if len(preprocess_outputs) != len(vision_session.get_inputs()):
        raise RuntimeError("Image preprocess and vision graph I/O counts differ.")
    vision_binding = vision_session.io_binding()
    values: list[onnxruntime.OrtValue] = []
    for input_meta, output_meta, value in zip(
        vision_session.get_inputs(), preprocess_session.get_outputs(), preprocess_outputs
    ):
        if input_meta.type != output_meta.type:
            value = _ort_value(value.numpy().astype(_np_dtype(input_meta.type), copy=False), device_type, device_id)
            values.append(value)
        vision_binding.bind_ortvalue_input(input_meta.name, value)
    output_names = [item.name for item in vision_session.get_outputs()]
    if output_names != ["vision_hidden_states"]:
        raise RuntimeError(f"Unexpected UnlimitedOCR vision outputs: {output_names!r}.")
    _bind_outputs(vision_binding, output_names, ort_device)
    _run(vision_session, vision_binding)
    return vision_binding.get_outputs()[0]


def _bind_array(binding, owners: list[onnxruntime.OrtValue], input_meta: dict[str, object],
                name: str, array: np.ndarray, device_type: str, device_id: int) -> None:
    value = _ort_value(np.asarray(array, dtype=_np_dtype(input_meta[name].type)), device_type, device_id)
    owners.append(value)
    binding.bind_ortvalue_input(name, value)


def _strategy_values(strategy: str, input_meta: dict[str, object], device_type: str):
    if REPETITION_PENALTY <= 0.0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    if strategy == "sampling":
        if TEMPERATURE <= 0.0 or TOP_K < 1 or not 0.0 < TOP_P <= 1.0:
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
        (name, _ort_value(value.astype(_np_dtype(input_meta[name].type)), device_type, DEVICE_ID))
        for name, value in entries
    ]


def assert_bundle_graphs(bundle: Path, names: dict[str, str]) -> None:
    required = [names["image_preprocess"], names["vision"], names["shared_initializers"],
                names["shared_initializers_data"]]
    for strategy in STRATEGIES:
        required.extend((names[f"image_prefill_{strategy}"], names[f"image_decode_{strategy}"]))
    missing = [str(bundle / name) for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError("Bundle is incomplete:\n" + "\n".join(missing))


def run_generation(model_folder: Path, metadata: dict[str, str], names: dict[str, str], tokenizer: Tokenizer,
                   vision_output: onnxruntime.OrtValue, device_type: str, ort_device,
                   providers: list[str]) -> str:
    bundle = model_folder
    shared_path = bundle / names["shared_initializers"]
    prefill = create_merged_session(bundle / names[f"image_prefill_{STRATEGY}"], shared_path, providers)
    decode = create_merged_session(bundle / names[f"image_decode_{STRATEGY}"], shared_path, providers)
    print(f"Usable Providers: {decode.get_providers()}")
    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill, STRATEGY, state_count, is_decode=False)
    decode_plan = plan_merged_io(decode, STRATEGY, state_count, is_decode=True)
    if prefill_plan["vision_input"] is None:
        raise RuntimeError("UnlimitedOCR prefill graph has no vision feature input.")
    prefill_meta, decode_meta = _input_meta(prefill), _input_meta(decode)
    tokens = build_native_prompt_tokens(
        tokenizer, QUERY, int(metadata["image_token_id"]), int(metadata["image_token_length"])
    )
    prefill_tokens = int(tokens.shape[1])
    limit = min(max(0, MAX_NEW_TOKENS), max(0, int(metadata["max_seq_len"]) - prefill_tokens))
    prefill_binding = prefill.io_binding()
    prefill_owners: list[onnxruntime.OrtValue] = []
    _bind_array(prefill_binding, prefill_owners, prefill_meta, prefill_plan["token_input"],
                tokens, device_type, DEVICE_ID)
    for name, value in (("prefill_ids_len", [prefill_tokens]), ("prefill_history_len", [0])):
        if name in prefill_plan["inputs"]:
            _bind_array(prefill_binding, prefill_owners, prefill_meta, name, np.asarray(value),
                        device_type, DEVICE_ID)
    prefill_binding.bind_ortvalue_input(prefill_plan["vision_input"], vision_output)
    state_device = "cpu" if device_type == "dml" else device_type
    for name in prefill_plan["state_in"]:
        _bind_array(prefill_binding, prefill_owners, prefill_meta, name, _zero_from_meta(prefill_meta[name]),
                    state_device, DEVICE_ID)
    if prefill_plan["save_id_input"] is not None:
        _bind_array(prefill_binding, prefill_owners, prefill_meta, prefill_plan["save_id_input"],
                    np.zeros((1, 0)), device_type, DEVICE_ID)
    for name, value in _strategy_values(STRATEGY, prefill_meta, device_type):
        prefill_owners.append(value)
        prefill_binding.bind_ortvalue_input(name, value)
    _bind_outputs(prefill_binding, prefill_plan["outputs"], ort_device)
    prefill_start = time.time()
    _run(prefill, prefill_binding)
    prefill_elapsed = time.time() - prefill_start
    prefill_outputs = prefill_binding.get_outputs()
    prefill_positions = {name: index for index, name in enumerate(prefill_plan["outputs"])}
    cached_state = prefill_outputs[:state_count]
    kv_seq_len = prefill_outputs[prefill_positions[prefill_plan["kv_seq_output"]]]
    selected_value = prefill_outputs[prefill_positions[prefill_plan["token_output"]]]
    saved_ids = (prefill_outputs[prefill_positions[prefill_plan["save_id_output"]]]
                 if prefill_plan["save_id_output"] is not None else None)
    stop_tokens = {int(value) for value in metadata["stop_token_ids"].split(",") if value}
    stop_tokens.update(int(value) for value in metadata.get("eos_token_ids", "").split(",") if value)
    generated: list[int] = []
    selected_id = int(selected_value.numpy().flat[0])
    if selected_id not in stop_tokens and limit:
        generated.append(selected_id)
        print(tokenizer.decode([selected_id]), end="", flush=True)
    decode_static = _strategy_values(STRATEGY, decode_meta, device_type)
    decode_start = time.time()
    while len(generated) < limit and selected_id not in stop_tokens:
        binding = decode.io_binding()
        owners: list[onnxruntime.OrtValue] = []
        _bind_array(binding, owners, decode_meta, decode_plan["token_input"], selected_value.numpy(),
                    device_type, DEVICE_ID)
        binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        for name, value in zip(decode_plan["state_in"], cached_state):
            binding.bind_ortvalue_input(name, value)
        if decode_plan["save_id_input"] is not None:
            if saved_ids is None:
                raise RuntimeError("Decode strategy requires generated-ID state missing from prefill output.")
            binding.bind_ortvalue_input(decode_plan["save_id_input"], saved_ids)
        for name, value in decode_static:
            binding.bind_ortvalue_input(name, value)
        _bind_outputs(binding, decode_plan["outputs"], ort_device)
        _run(decode, binding)
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
    print(
        "\n\nGenerated Output\n" + result + "\n\nPerformance\n"
        f"{'Stage':<10} {'Tokens':>8} {'Time (s)':>10} {'Tokens/s':>12}\n"
        f"{'Prefill':<10} {prefill_tokens:>8} {prefill_elapsed:>10.3f} "
        f"{prefill_tokens / prefill_elapsed if prefill_elapsed else 0.0:>12.2f}\n"
        f"{'Decode':<10} {len(generated):>8} {decode_elapsed:>10.3f} "
        f"{len(generated) / decode_elapsed if decode_elapsed else 0.0:>12.2f}\n"
        f"{'Overall':<10} {prefill_tokens + len(generated):>8} {total_elapsed:>10.3f} "
        f"{(prefill_tokens + len(generated)) / total_elapsed if total_elapsed else 0.0:>12.2f}"
    )
    return result


def run_inference(model_folder: Path) -> None:
    """Run one UnlimitedOCR request from the selected ONNX bundle."""
    if MAX_NEW_TOKENS < 0:
        raise ValueError("MAX_NEW_TOKENS must be non-negative.")
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported STRATEGY: {STRATEGY!r}.")
    model_folder = model_folder.expanduser().resolve()
    providers = list(PROVIDERS)
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    device_type, ort_device_kind = _provider_device(providers)
    ort_device = C.OrtDevice(ort_device_kind, C.OrtDevice.default_memory(), DEVICE_ID)
    metadata = load_metadata(model_folder)
    names = load_model_file_names(metadata)
    assert_bundle_graphs(model_folder, names)
    tokenizer = load_tokenizer(model_folder)
    image_size = [int(value) for value in metadata["input_image_size"].split(",")]
    image = load_image(DEFAULT_IMAGE, image_size[0], image_size[1], int(metadata["input_image_dim"]))
    preprocess = create_plain_session(model_folder / names["image_preprocess"], providers)
    vision = create_plain_session(model_folder / names["vision"], providers)
    vision_output = run_vision(preprocess, vision, image, device_type, DEVICE_ID, ort_device)
    print(f"\nQuestion: {QUERY}\nAnswering:")
    result = run_generation(
        model_folder, metadata, names, tokenizer, vision_output, device_type, ort_device, providers
    )
    if EXPECT_NONEMPTY_OUTPUT and not result.strip():
        raise RuntimeError("Expected nonempty OCR output, but generation was empty.")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()