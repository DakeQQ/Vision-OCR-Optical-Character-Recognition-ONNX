"""Run an LFM2.5-VL-450M-Extract merged ONNX bundle independently.

The runtime owns bundle-local prompt construction, fixed 512px image input,
metadata-driven graph selection, shared initializer attachment, and decode state.
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
SCRIPT_DIR            = Path(__file__).resolve().parent
DEFAULT_MODEL_FOLDER  = SCRIPT_DIR / "LFM25_VL_ONNX"
DEFAULT_IMAGE         = SCRIPT_DIR / "sample_image.png"

# Inference configuration
QUERY              = "Describe the wood color, texture, and pattern."
SYSTEM_PROMPT      = "Respond with only a JSON object. Do not include any text outside the JSON."
STRATEGY           = "sampling"
MAX_NEW_TOKENS     = 4096
TEMPERATURE        = 0.8
TOP_K              = 20
TOP_P              = 0.95
REPETITION_PENALTY = 1.1
PROVIDERS          = ("CPUExecutionProvider",)
DEVICE_ID          = 0
THREADS            = 0

# Bundle tokenizer contract
CHAT_BOS_TOKEN         = "<|startoftext|>"
CHAT_START_TOKEN       = "<|im_start|>"
CHAT_END_TOKEN         = "<|im_end|>"
IMAGE_TOKEN            = "<image>"
IMAGE_START_TOKEN      = "<|image_start|>"
IMAGE_END_TOKEN        = "<|image_end|>"
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
    parser = argparse.ArgumentParser(description="Run LFM2.5-VL-450M-Extract merged ONNX inference.")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    return parser.parse_args().model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or ".." in path.parts:
        raise RuntimeError(f"Metadata key {metadata_key!r} has unsafe file name {value!r}.")
    return value


def _metadata_session(path: Path) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_metadata(bundle: Path) -> dict[str, str]:
    path = bundle / _DEFAULT_MODEL_FILE_NAMES["metadata"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata carrier: {path}")
    metadata = dict(_metadata_session(path).get_modelmeta().custom_metadata_map)
    required = (
        "model_family",
        "input_modality",
        "max_seq_len",
        "stop_token_ids",
        "kv_num_tensors",
        "image_token_id",
        "image_token_length",
        "input_image_size",
        "vision_static_resolution",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"LLM_Metadata.onnx is incomplete; missing: {missing!r}.")
    if metadata["model_family"] != "lfm25_vl_extract" or metadata["input_modality"] != "image_text":
        raise RuntimeError("This runtime requires an LFM2.5-VL image-text bundle.")
    if metadata["vision_static_resolution"] != "1":
        raise RuntimeError("This runtime expects the exporter static-resolution contract.")
    return metadata


def load_model_file_names(metadata: dict[str, str]) -> dict[str, str]:
    return {
        role: _safe_file_name(metadata.get(f"model_file_name_{role}", default), f"model_file_name_{role}")
        for role, default in _DEFAULT_MODEL_FILE_NAMES.items()
    }


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(options: onnxruntime.SessionOptions, shared_path: Path):
    """Attach mmap-backed non-logical low-bit tensors and keep owners alive."""
    shared = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    values: list[onnxruntime.OrtValue] = []
    for initializer in shared.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = _safe_file_name(external.get("location", ""), initializer.name)
        data_path = shared_path.parent / location
        if not data_path.is_file():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        array = np.memmap(
            data_path,
            dtype=onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type),
            mode="r",
            offset=int(external.get("offset", "0")),
            shape=tuple(int(dimension) for dimension in initializer.dims),
        )
        value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        arrays[initializer.name] = array
        values.append(value)
        options.add_initializer(initializer.name, value)
    if not values:
        raise RuntimeError("Shared initializer carrier has no attachable tensors.")
    return shared, arrays, values


def _provider_device(providers: list[str], device_id: int):
    if "CUDAExecutionProvider" in providers:
        kind = C.OrtDevice.cuda()
        return "cuda", C.OrtDevice(kind, C.OrtDevice.default_memory(), device_id)
    if "DmlExecutionProvider" in providers:
        kind = C.OrtDevice.dml()
        return "dml", C.OrtDevice(kind, C.OrtDevice.default_memory(), device_id)
    kind = C.OrtDevice.cpu()
    return "cpu", C.OrtDevice(kind, C.OrtDevice.default_memory(), device_id)


def create_session_options(threads: int) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = threads
    options.intra_op_num_threads = threads
    options.enable_cpu_mem_arena = True
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_cast_chain_elimination": "1",
    }.items():
        options.add_session_config_entry(key, value)
    return options


_session_options = create_session_options


def create_plain_session(path: Path, providers: list[str], threads: int):
    return onnxruntime.InferenceSession(str(path), sess_options=create_session_options(threads), providers=providers)


def create_merged_session(path: Path, shared_path: Path, providers: list[str], threads: int):
    options = create_session_options(threads)
    owners = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._lfm25_shared_initializer_owners = owners
    return session


def _np_dtype(type_name: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
    }
    try:
        return mapping[type_name]
    except KeyError as error:
        raise RuntimeError(f"Unsupported ONNX tensor type: {type_name}") from error


def _zero_from_meta(meta) -> np.ndarray:
    shape = [1 if index == 0 else (dimension if isinstance(dimension, int) else 0) for index, dimension in enumerate(meta.shape)]
    return np.zeros(shape, dtype=_np_dtype(meta.type))


def _ort_value(values: np.ndarray, device_type: str, device_id: int) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(values), device_type, device_id)


def _bind_outputs(binding, names: list[str], ort_device) -> None:
    for name in names:
        binding._iobinding.bind_output(name, ort_device)


def _run(session, binding) -> None:
    options = onnxruntime.RunOptions()
    options.log_severity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    session.run_with_iobinding(binding, run_options=options)


def plan_merged_io(session, strategy: str, state_count: int, is_decode: bool) -> dict:
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    if len(inputs) < state_count or len(outputs) < state_count:
        raise RuntimeError("Merged graph has fewer recurrent state tensors than metadata declares.")
    state_inputs = inputs[:state_count]
    state_outputs = outputs[:state_count]
    if any(not name.startswith("in_") for name in state_inputs):
        raise RuntimeError("Merged state inputs must lead and be named in_*.")
    if any(not name.startswith("out_") for name in state_outputs):
        raise RuntimeError("Merged state outputs must lead and be named out_*.")
    tail = outputs[state_count:]
    expected_tail = 2 if strategy == "greedy" else 3
    if len(tail) != expected_tail:
        raise RuntimeError(f"Unexpected {strategy} output tail: {tail!r}.")
    token_input = next((name for name in ("embed_input_ids", "input_ids") if name in inputs), None)
    if token_input is None:
        raise RuntimeError("Merged graph has no token input.")
    vision_input = next((name for name in inputs if name.endswith("vision_hidden_states")), None)
    kv_seq_input = None
    if is_decode:
        kv_seq_input = next((name for name in inputs if name.startswith("decode_kv_seq_len")), None)
        if kv_seq_input is None:
            raise RuntimeError("Decode graph has no decode KV sequence-length input.")
    save_id_input = None if strategy == "greedy" else f"{strategy}_previous_ids"
    if save_id_input is not None and save_id_input not in inputs:
        raise RuntimeError(f"Merged graph is missing {save_id_input!r}.")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "token_input": token_input,
        "vision_input": vision_input,
        "kv_seq_input": kv_seq_input,
        "token_output": tail[0],
        "save_id_output": None if strategy == "greedy" else tail[1],
        "kv_seq_output": tail[-1],
        "save_id_input": save_id_input,
    }


def _static_strategy_values(strategy: str, input_meta: dict[str, object], device_type: str):
    if REPETITION_PENALTY <= 0.0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    entries = []
    if strategy == "sampling":
        if TEMPERATURE <= 0.0 or TOP_K < 1 or not 0.0 < TOP_P <= 1.0:
            raise ValueError("Sampling requires TEMPERATURE > 0, TOP_K >= 1, and 0 < TOP_P <= 1.")
        entries.extend((
            ("sampling_temperature", np.asarray([TEMPERATURE])),
            ("sampling_top_k", np.asarray(TOP_K)),
            ("sampling_top_p", np.asarray([TOP_P])),
            ("sampling_repetition_penalty", np.asarray([[REPETITION_PENALTY]])),
        ))
    elif strategy == "penalty_greedy":
        entries.append(("penalty_greedy_repetition_penalty", np.asarray([[REPETITION_PENALTY]])))
    return [
        (name, _ort_value(value.astype(_np_dtype(input_meta[name].type)), device_type, DEVICE_ID))
        for name, value in entries
    ]


def load_tokenizer(bundle: Path) -> Tokenizer:
    path = bundle / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def _render_chat_prompt(system_prompt: str, user_content: str) -> str:
    parts = [CHAT_BOS_TOKEN]
    if system_prompt:
        parts.append(f"{CHAT_START_TOKEN}system\n{system_prompt}{CHAT_END_TOKEN}\n")
    parts.append(f"{CHAT_START_TOKEN}user\n{user_content}{CHAT_END_TOKEN}\n")
    parts.append(f"{CHAT_START_TOKEN}assistant\n")
    return "".join(parts)


def build_native_prompt_ids(tokenizer: Tokenizer, metadata: dict[str, str], query: str, system_prompt: str) -> np.ndarray:
    image_token_id = int(metadata["image_token_id"])
    expected = int(metadata["image_token_length"])
    if expected <= 0:
        raise RuntimeError(f"Invalid image token length: {expected}.")
    if tokenizer.token_to_id(IMAGE_TOKEN) != image_token_id:
        raise RuntimeError("Bundle tokenizer image token does not match graph metadata.")
    image_content = f"{IMAGE_START_TOKEN}{IMAGE_TOKEN * expected}{IMAGE_END_TOKEN}{query}"
    token_ids = tokenizer.encode(
        _render_chat_prompt(system_prompt, image_content), add_special_tokens=False
    ).ids
    positions = [index for index, token_id in enumerate(token_ids) if int(token_id) == image_token_id]
    if len(positions) != expected:
        raise RuntimeError(f"Manual prompt has {len(positions)} image tokens, expected {expected}.")
    if positions[-1] - positions[0] + 1 != expected:
        raise RuntimeError("Manual prompt image placeholders must occupy one contiguous span.")
    return np.asarray([token_ids], dtype=np.int32)


def load_static_image(path: Path, metadata: dict[str, str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    height, width = (int(value) for value in metadata["input_image_size"].split(","))
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), resampling)
        values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(values)


def run_vision(bundle: Path, file_names: dict[str, str], image: np.ndarray, device_type: str, ort_device):
    providers = list(PROVIDERS)
    preprocess = create_plain_session(bundle / file_names["image_preprocess"], providers, THREADS)
    vision = create_plain_session(bundle / file_names["vision"], providers, THREADS)
    preprocess_binding = preprocess.io_binding()
    input_meta = preprocess.get_inputs()[0]
    raw_value = _ort_value(image.astype(_np_dtype(input_meta.type), copy=False), device_type, DEVICE_ID)
    preprocess_binding.bind_ortvalue_input(input_meta.name, raw_value)
    preprocess_outputs = [item.name for item in preprocess.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_outputs, ort_device)
    _run(preprocess, preprocess_binding)
    patches = preprocess_binding.get_outputs()[0]
    vision_binding = vision.io_binding()
    vision_binding.bind_ortvalue_input(vision.get_inputs()[0].name, patches)
    vision_outputs = [item.name for item in vision.get_outputs()]
    _bind_outputs(vision_binding, vision_outputs, ort_device)
    _run(vision, vision_binding)
    return vision_binding.get_outputs()[0]


def run_generation(model_folder: Path, metadata: dict[str, str], file_names: dict[str, str]) -> str:
    bundle = model_folder.expanduser().resolve()
    providers = list(PROVIDERS)
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported configured strategy: {STRATEGY!r}.")
    device_type, ort_device = _provider_device(providers, DEVICE_ID)
    tokenizer = load_tokenizer(bundle)
    image = load_static_image(DEFAULT_IMAGE.expanduser().resolve(), metadata)
    vision_features = run_vision(bundle, file_names, image, device_type, ort_device)
    shared_path = bundle / file_names["shared_initializers"]
    shared_data_path = bundle / file_names["shared_initializers_data"]
    if not shared_path.is_file() or not shared_data_path.is_file():
        raise FileNotFoundError("Merged runtime requires both shared initializer artifacts.")
    prefill = create_merged_session(bundle / file_names[f"image_prefill_{STRATEGY}"], shared_path, providers, THREADS)
    decode = create_merged_session(bundle / file_names[f"image_decode_{STRATEGY}"], shared_path, providers, THREADS)
    print(f"Usable Providers: {decode.get_providers()}")
    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill, STRATEGY, state_count, is_decode=False)
    decode_plan = plan_merged_io(decode, STRATEGY, state_count, is_decode=True)
    if prefill_plan["vision_input"] is None:
        raise RuntimeError("Image prefill graph does not accept vision features.")
    prefill_meta = {item.name: item for item in prefill.get_inputs()}
    decode_meta = {item.name: item for item in decode.get_inputs()}
    token_ids = build_native_prompt_ids(tokenizer, metadata, QUERY, SYSTEM_PROMPT)
    prefill_tokens = int(token_ids.shape[1])
    limit = min(MAX_NEW_TOKENS, max(0, int(metadata["max_seq_len"]) - prefill_tokens))

    binding = prefill.io_binding()
    owned_values: list[onnxruntime.OrtValue] = []
    token_value = _ort_value(token_ids.astype(_np_dtype(prefill_meta[prefill_plan["token_input"]].type)), device_type, DEVICE_ID)
    owned_values.append(token_value)
    binding.bind_ortvalue_input(prefill_plan["token_input"], token_value)
    binding.bind_ortvalue_input(prefill_plan["vision_input"], vision_features)
    for name, data in (("prefill_ids_len", [prefill_tokens]), ("prefill_history_len", [0])):
        if name in prefill_plan["inputs"]:
            value = _ort_value(np.asarray(data, dtype=_np_dtype(prefill_meta[name].type)), device_type, DEVICE_ID)
            owned_values.append(value)
            binding.bind_ortvalue_input(name, value)
    state_device = "cpu" if device_type == "dml" else device_type
    for name in prefill_plan["state_inputs"]:
        value = _ort_value(_zero_from_meta(prefill_meta[name]), state_device, DEVICE_ID)
        owned_values.append(value)
        binding.bind_ortvalue_input(name, value)
    if prefill_plan["save_id_input"] is not None:
        name = prefill_plan["save_id_input"]
        value = _ort_value(np.zeros((1, 0), dtype=_np_dtype(prefill_meta[name].type)), device_type, DEVICE_ID)
        owned_values.append(value)
        binding.bind_ortvalue_input(name, value)
    for name, value in _static_strategy_values(STRATEGY, prefill_meta, device_type):
        owned_values.append(value)
        binding.bind_ortvalue_input(name, value)
    _bind_outputs(binding, prefill_plan["outputs"], ort_device)

    prefill_start = time.time()
    _run(prefill, binding)
    prefill_elapsed = time.time() - prefill_start
    outputs = binding.get_outputs()
    positions = {name: index for index, name in enumerate(prefill_plan["outputs"])}
    state = outputs[:state_count]
    kv_seq_len = outputs[positions[prefill_plan["kv_seq_output"]]]
    selected_value = outputs[positions[prefill_plan["token_output"]]]
    save_ids = outputs[positions[prefill_plan["save_id_output"]]] if prefill_plan["save_id_output"] else None

    stop_tokens = {int(token) for token in metadata["stop_token_ids"].split(",") if token}
    generated: list[int] = []
    selected_id = int(selected_value.numpy().flat[0])
    if selected_id not in stop_tokens and limit:
        generated.append(selected_id)
        print(tokenizer.decode([selected_id], skip_special_tokens=False), end="", flush=True)
    decode_static = _static_strategy_values(STRATEGY, decode_meta, device_type)
    decode_start = time.time()
    while len(generated) < limit and selected_id not in stop_tokens:
        binding = decode.io_binding()
        token_array = selected_value.numpy().astype(_np_dtype(decode_meta[decode_plan["token_input"]].type), copy=False)
        token_value = _ort_value(token_array, device_type, DEVICE_ID)
        binding.bind_ortvalue_input(decode_plan["token_input"], token_value)
        binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        for name, value in zip(decode_plan["state_inputs"], state):
            binding.bind_ortvalue_input(name, value)
        if decode_plan["save_id_input"] is not None:
            if save_ids is None:
                raise RuntimeError("Strategy requires generated-ID state missing from prefill output.")
            binding.bind_ortvalue_input(decode_plan["save_id_input"], save_ids)
        for name, value in decode_static:
            binding.bind_ortvalue_input(name, value)
        _bind_outputs(binding, decode_plan["outputs"], ort_device)
        _run(decode, binding)
        outputs = binding.get_outputs()
        positions = {name: index for index, name in enumerate(decode_plan["outputs"])}
        state = outputs[:state_count]
        kv_seq_len = outputs[positions[decode_plan["kv_seq_output"]]]
        selected_value = outputs[positions[decode_plan["token_output"]]]
        if decode_plan["save_id_output"]:
            save_ids = outputs[positions[decode_plan["save_id_output"]]]
        selected_id = int(selected_value.numpy().flat[0])
        if selected_id not in stop_tokens:
            generated.append(selected_id)
            print(tokenizer.decode([selected_id], skip_special_tokens=False), end="", flush=True)
    decode_elapsed = time.time() - decode_start
    result = tokenizer.decode(generated, skip_special_tokens=True)
    prefill_tps = prefill_tokens / prefill_elapsed if prefill_elapsed else 0.0
    decode_tps = len(generated) / decode_elapsed if decode_elapsed else 0.0
    print(
        "\n\nPerformance\n"
        f"Prefill: {prefill_tokens} tokens in {prefill_elapsed:.3f}s ({prefill_tps:.2f} tok/s)\n"
        f"Decode: {len(generated)} tokens in {decode_elapsed:.3f}s ({decode_tps:.2f} tok/s)"
    )
    return result


def run_inference(model_folder: Path) -> None:
    """Run one LFM VL OCR request from the selected ONNX bundle."""
    model_folder = model_folder.expanduser().resolve()
    metadata = load_metadata(model_folder)
    file_names = load_model_file_names(metadata)
    print(run_generation(model_folder, metadata, file_names))


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
