"""Run OvisOCR2's standalone merged ONNX Runtime bundle."""

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
DEFAULT_MODEL_FOLDER = SCRIPT_DIR / "OvisOCR2_Optimized"
DEFAULT_IMAGE        = SCRIPT_DIR / "psyduck_2.png"
DEFAULT_OCR_QUERY    = (
    "Extract all readable content from the image in natural human reading order and "
    "output the result as a single Markdown document. For charts or images, represent "
    "them using an HTML image tag: <img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, "
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe "
    "all other text as standard Markdown. Preserve the original text without translation "
    "or paraphrasing."
)

# Inference configuration
INPUT_IMAGES           = [DEFAULT_IMAGE]
QUERY                  = DEFAULT_OCR_QUERY
STRATEGY               = "greedy"
MAX_NEW_TOKENS         = None
TEMPERATURE            = 0.8
TOP_K                  = 20
TOP_P                  = 0.95
REPETITION_PENALTY     = 1.0
PROVIDERS              = []
DEVICE_ID              = 0
EXPECT_NONEMPTY_OUTPUT = False

# Bundle tokenizer contract
CHAT_START_TOKEN   = "<|im_start|>"
CHAT_END_TOKEN     = "<|im_end|>"
VISION_START_TOKEN = "<|vision_start|>"
IMAGE_TOKEN        = "<|image_pad|>"
VISION_END_TOKEN   = "<|vision_end|>"
THINKING_PREFIX    = "<think>\n\n</think>\n\n"
STRATEGIES         = ("greedy", "penalty_greedy", "sampling")

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
    parser = argparse.ArgumentParser(description="Run OvisOCR2 merged ONNX inference.")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    return parser.parse_args().model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise RuntimeError(
            f"Metadata key {metadata_key!r} must be a safe filename, got {value!r}."
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
    metadata_path = bundle / _DEFAULT_MODEL_FILE_NAMES["metadata"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata carrier: {metadata_path}")
    try:
        metadata = dict(_metadata_session(metadata_path).get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise RuntimeError("Unable to load LLM_Metadata.onnx before large sessions.") from error
    required = (
        "max_seq_len", "input_image_size", "input_image_dim", "vision_batch_size",
        "image_token_id", "image_token_length", "image_spans", "image_grid_height",
        "image_grid_width", "kv_num_tensors", "stop_token_ids",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"LLM_Metadata.onnx is incomplete; missing: {missing!r}.")
    return metadata


def load_model_file_names(metadata: dict[str, str]) -> dict[str, str]:
    names = {}
    for role, default in _DEFAULT_MODEL_FILE_NAMES.items():
        key = f"model_file_name_{role}"
        names[role] = _safe_file_name(metadata.get(key, default), key)
    return names


def _parse_providers(value: str) -> list[str]:
    providers = [item.strip() for item in value.split(",") if item.strip()]
    if not providers:
        raise ValueError("At least one execution provider is required.")
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
        "optimization.enable_cast_chain_elimination": "1",
    }.items():
        options.add_session_config_entry(key, value)
    return options


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(session_options: onnxruntime.SessionOptions, shared_path: Path):
    """Attach shared mmap-backed tensors and retain all owners for session lifetime."""
    shared_model = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    values: list[onnxruntime.OrtValue] = []
    for initializer in shared_model.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {initializer.name!r} has no data location.")
        data_path = shared_path.parent / _safe_file_name(location, initializer.name)
        if not data_path.is_file():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        except KeyError as error:
            raise RuntimeError(
                f"Cannot map shared initializer dtype for {initializer.name!r}."
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
        raise RuntimeError("Shared initializer carrier has no attachable tensors.")
    return shared_model, arrays, values


def create_plain_session(path: Path, providers: list[str]):
    return onnxruntime.InferenceSession(
        str(path), sess_options=create_session_options(), providers=providers
    )


def create_merged_session(path: Path, shared_path: Path, providers: list[str]):
    options = create_session_options()
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._ovis_shared_initializers = references
    return session


def _np_dtype(type_name: str):
    for marker, dtype in (
        ("float16", np.float16), ("float", np.float32), ("uint8", np.uint8),
        ("int8", np.int8), ("int64", np.int64), ("int32", np.int32),
    ):
        if marker in type_name:
            return dtype
    raise ValueError(f"Unsupported ONNX Runtime tensor type: {type_name}")


def _ort_value(array: np.ndarray, device_type: str, device_id: int) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device_type, device_id
    )


def _input_meta(session) -> dict[str, object]:
    return {item.name: item for item in session.get_inputs()}


def _state_seq_axis(meta) -> int | None:
    axes = [
        index for index, dimension in enumerate(meta.shape)
        if index != 0 and not isinstance(dimension, int)
    ]
    return axes[0] if len(axes) == 1 else None


def _zero_from_meta(meta) -> np.ndarray:
    shape = list(meta.shape)
    sequence_axis = _state_seq_axis(meta)
    for index, dimension in enumerate(shape):
        if index == 0:
            shape[index] = 1
        elif index == sequence_axis:
            shape[index] = 0
        elif not isinstance(dimension, int):
            shape[index] = 1
    return np.zeros(tuple(shape), dtype=_np_dtype(meta.type))


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
    if inputs[:state_count] != [name for name in inputs[:state_count] if name.startswith("in_")]:
        raise RuntimeError("Merged OvisOCR2 state inputs must lead and be named in_*.")
    if outputs[:state_count] != [name for name in outputs[:state_count] if name.startswith("out_")]:
        raise RuntimeError("Merged OvisOCR2 state outputs must lead and be named out_*.")
    expected_tail = 2 if strategy == "greedy" else 3
    if len(outputs) != state_count + expected_tail:
        raise RuntimeError(f"Unexpected {strategy} output contract: {outputs!r}.")
    token_input = next((name for name in inputs if name in {"embed_input_ids", "input_ids"}), None)
    if token_input is None:
        raise RuntimeError("Merged graph has no token-ID input.")
    position_input = next((name for name in inputs if name.endswith("position_ids")), None)
    if position_input is None:
        raise RuntimeError("Merged graph has no runtime mRoPE position-ID input.")
    kv_seq_input = None
    if is_decode:
        kv_seq_input = next((name for name in inputs if name.startswith("decode_kv_seq_len")), None)
        if kv_seq_input is None:
            raise RuntimeError("Decode graph has no KV sequence-length input.")
    save_id_input = None if strategy == "greedy" else f"{strategy}_previous_ids"
    if save_id_input is not None and save_id_input not in inputs:
        raise RuntimeError(f"Merged graph is missing {save_id_input!r}.")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_in": inputs[:state_count],
        "state_out": outputs[:state_count],
        "token_input": token_input,
        "position_input": position_input,
        "kv_seq_input": kv_seq_input,
        "token_output": outputs[state_count],
        "save_id_input": save_id_input,
        "save_id_output": None if strategy == "greedy" else outputs[state_count + 1],
        "kv_seq_output": outputs[-1],
    }


def load_tokenizer(bundle: Path) -> Tokenizer:
    path = bundle / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def _parse_spans(metadata: dict[str, str]) -> list[tuple[int, int]]:
    spans = []
    for entry in metadata["image_spans"].split(";"):
        start_text, end_text = entry.split(":", 1)
        start, end = int(start_text), int(end_text)
        if start < 0 or end <= start:
            raise RuntimeError(f"Invalid exported image span {entry!r}.")
        spans.append((start, end))
    return spans


def build_ovis_prompt(
    tokenizer: Tokenizer, query: str, image_count: int, metadata: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    image_token_id = int(metadata["image_token_id"])
    image_token_length = int(metadata["image_token_length"])
    if image_count < 1:
        raise ValueError("OvisOCR2 requires at least one input image.")
    if tokenizer.token_to_id(IMAGE_TOKEN) != image_token_id:
        raise RuntimeError("Bundle tokenizer image token does not match LLM_Metadata.onnx.")
    image_placeholder = f"{VISION_START_TOKEN}{IMAGE_TOKEN}{VISION_END_TOKEN}"
    prompt = (
        f"{CHAT_START_TOKEN}user\n{image_placeholder * image_count}{query}{CHAT_END_TOKEN}\n"
        f"{CHAT_START_TOKEN}assistant\n{THINKING_PREFIX}"
    )
    expanded = prompt.replace(IMAGE_TOKEN, IMAGE_TOKEN * image_token_length)
    token_ids = [int(value) for value in tokenizer.encode(expanded, add_special_tokens=False).ids]
    positions = [index for index, value in enumerate(token_ids) if value == image_token_id]
    expected = image_count * image_token_length
    if len(positions) != expected:
        raise RuntimeError("Expanded OvisOCR2 image token count differs from metadata.")
    expected_spans = _parse_spans(metadata)
    spans = []
    cursor = 0
    while cursor < len(positions):
        start = positions[cursor]
        end = start + 1
        cursor += 1
        while cursor < len(positions) and positions[cursor] == end:
            end += 1
            cursor += 1
        spans.append((start, end))
    if spans != expected_spans:
        raise RuntimeError(
            f"Native chat template image spans {spans!r} differ from export metadata {expected_spans!r}."
        )
    return np.asarray([token_ids], dtype=np.int32), build_ovis_position_ids(token_ids, spans, metadata)


def build_ovis_position_ids(
    token_ids: list[int], spans: list[tuple[int, int]], metadata: dict[str, str]
) -> np.ndarray:
    grid_height = int(metadata["image_grid_height"])
    grid_width = int(metadata["image_grid_width"])
    expected_length = int(metadata["image_token_length"])
    positions = np.zeros((3, len(token_ids)), dtype=np.int64)
    current = 0
    cursor = 0
    for image_start, image_end in spans:
        while cursor < image_start:
            positions[:, cursor] = current
            current += 1
            cursor += 1
        if image_end - image_start != expected_length or expected_length != grid_height * grid_width:
            raise RuntimeError("OvisOCR2 image span does not match exported post-merge vision grid.")
        positions[0, image_start:image_end] = current
        positions[1, image_start:image_end] = (
            np.repeat(np.arange(grid_height, dtype=np.int64), grid_width) + current
        )
        positions[2, image_start:image_end] = (
            np.tile(np.arange(grid_width, dtype=np.int64), grid_height) + current
        )
        current += max(grid_height, grid_width)
        cursor = image_end
    while cursor < len(token_ids):
        positions[:, cursor] = current
        current += 1
        cursor += 1
    return positions


def load_images(paths: list[Path], height: int, width: int, batch_size: int, input_rank: int) -> np.ndarray:
    if len(paths) != batch_size:
        raise ValueError(f"Expected {batch_size} image(s), got {len(paths)}.")
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    images = np.empty((batch_size, 3, height, width), dtype=np.uint8)
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            scale = min(width / max(source_width, 1), height / max(source_height, 1))
            resized_width = max(1, min(width, int(round(source_width * scale))))
            resized_height = max(1, min(height, int(round(source_height * scale))))
            if image.size != (resized_width, resized_height):
                image = image.resize((resized_width, resized_height), resampling)
            canvas = Image.new("RGB", (width, height), (128, 128, 128))
            offset_x = (width - resized_width) // 2
            offset_y = (height - resized_height) // 2
            canvas.paste(image, (offset_x, offset_y))
            images[index] = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)
    if input_rank == 5:
        images = np.expand_dims(images, axis=1)
    elif input_rank != 4:
        raise ValueError(f"Unsupported OvisOCR2 image input rank: {input_rank}.")
    return np.ascontiguousarray(images)


def run_vision(preprocess_session, vision_session, images, device_type, ort_device, device_id):
    preprocess_input = preprocess_session.get_inputs()[0]
    preprocess_binding = preprocess_session.io_binding()
    preprocess_value = _ort_value(
        images.astype(_np_dtype(preprocess_input.type), copy=False), device_type, device_id
    )
    preprocess_binding.bind_ortvalue_input(preprocess_input.name, preprocess_value)
    preprocess_output_names = [item.name for item in preprocess_session.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_output_names, ort_device)
    _run(preprocess_session, preprocess_binding)
    preprocess_outputs = preprocess_binding.get_outputs()

    vision_binding = vision_session.io_binding()
    retained = []
    for input_meta, output_meta, value in zip(
        vision_session.get_inputs(), preprocess_session.get_outputs(), preprocess_outputs
    ):
        if input_meta.type != output_meta.type:
            value = _ort_value(
                value.numpy().astype(_np_dtype(input_meta.type), copy=False), device_type, device_id
            )
            retained.append(value)
        vision_binding.bind_ortvalue_input(input_meta.name, value)
    output_names = [item.name for item in vision_session.get_outputs()]
    _bind_outputs(vision_binding, output_names, ort_device)
    _run(vision_session, vision_binding)
    vision_session._ovis_runtime_values = retained
    return dict(zip(output_names, vision_binding.get_outputs()))


def _bind_array(binding, retained, meta_by_name, name, array, device_type, device_id):
    value = _ort_value(
        np.asarray(array, dtype=_np_dtype(meta_by_name[name].type)), device_type, device_id
    )
    retained.append(value)
    binding.bind_ortvalue_input(name, value)


def _strategy_values(strategy, input_meta, device_type):
    if REPETITION_PENALTY <= 0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    if strategy == "sampling":
        if TEMPERATURE <= 0 or TOP_K < 1 or not 0 < TOP_P <= 1:
            raise ValueError("Sampling requires TEMPERATURE > 0, TOP_K >= 1, and 0 < TOP_P <= 1.")
        values = (
            ("sampling_temperature", np.asarray([TEMPERATURE])),
            ("sampling_top_k", np.asarray(TOP_K)),
            ("sampling_top_p", np.asarray([TOP_P])),
            ("sampling_repetition_penalty", np.asarray([[REPETITION_PENALTY]])),
        )
    elif strategy == "penalty_greedy":
        values = (("penalty_greedy_repetition_penalty", np.asarray([[REPETITION_PENALTY]])),)
    else:
        values = ()
    return [
        (name, _ort_value(array.astype(_np_dtype(input_meta[name].type)), device_type, DEVICE_ID))
        for name, array in values
    ]


def assert_bundle(bundle: Path, file_names: dict[str, str]):
    required = [file_names["image_preprocess"], file_names["vision"], file_names["shared_initializers"]]
    for phase in ("prefill", "decode"):
        for strategy in STRATEGIES:
            required.append(file_names[f"image_{phase}_{strategy}"])
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Merged OvisOCR2 bundle is incomplete: {missing!r}.")


def run_generation(model_folder, metadata, file_names, tokenizer, vision_outputs, device_type, ort_device, providers):
    bundle = model_folder
    shared_path = bundle / file_names["shared_initializers"]
    prefill = create_merged_session(
        bundle / file_names[f"image_prefill_{STRATEGY}"], shared_path, providers
    )
    decode = create_merged_session(
        bundle / file_names[f"image_decode_{STRATEGY}"], shared_path, providers
    )
    print(f"Usable providers: {decode.get_providers()}")
    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill, STRATEGY, state_count, is_decode=False)
    decode_plan = plan_merged_io(decode, STRATEGY, state_count, is_decode=True)
    prefill_meta = _input_meta(prefill)
    decode_meta = _input_meta(decode)

    token_ids, prefill_positions = build_ovis_prompt(
        tokenizer, QUERY, int(metadata["vision_batch_size"]), metadata
    )
    prefill_length = token_ids.shape[1]
    generation_limit = max(0, int(metadata["max_seq_len"]) - prefill_length)
    if MAX_NEW_TOKENS is not None:
        generation_limit = min(generation_limit, max(0, MAX_NEW_TOKENS))
    retained = []
    binding = prefill.io_binding()
    _bind_array(binding, retained, prefill_meta, prefill_plan["token_input"], token_ids, device_type, DEVICE_ID)
    _bind_array(binding, retained, prefill_meta, prefill_plan["position_input"], prefill_positions, device_type, DEVICE_ID)
    for name in prefill_plan["inputs"]:
        if name.startswith("prefill_ids_len"):
            _bind_array(binding, retained, prefill_meta, name, np.asarray([prefill_length]), device_type, DEVICE_ID)
        elif name.startswith("prefill_history_len"):
            _bind_array(binding, retained, prefill_meta, name, np.asarray([0]), device_type, DEVICE_ID)
    vision_input = next((name for name in prefill_plan["inputs"] if name.endswith("vision_hidden_states")), None)
    if vision_input is None or "vision_hidden_states" not in vision_outputs:
        raise RuntimeError("Image prefill graph does not accept the exported vision output.")
    binding.bind_ortvalue_input(vision_input, vision_outputs["vision_hidden_states"])
    for name in prefill_plan["state_in"]:
        _bind_array(binding, retained, prefill_meta, name, _zero_from_meta(prefill_meta[name]), device_type, DEVICE_ID)
    if prefill_plan["save_id_input"]:
        _bind_array(
            binding, retained, prefill_meta, prefill_plan["save_id_input"],
            np.zeros((1, 0), dtype=np.int32), device_type, DEVICE_ID
        )
    for name, value in _strategy_values(STRATEGY, prefill_meta, device_type):
        retained.append(value)
        binding.bind_ortvalue_input(name, value)
    _bind_outputs(binding, prefill_plan["outputs"], ort_device)
    start = time.perf_counter()
    _run(prefill, binding)
    prefill_elapsed = time.perf_counter() - start
    outputs = binding.get_outputs()
    output_index = {name: index for index, name in enumerate(prefill_plan["outputs"])}
    state = outputs[:state_count]
    selected = outputs[output_index[prefill_plan["token_output"]]]
    kv_seq_len = outputs[output_index[prefill_plan["kv_seq_output"]]]
    saved_ids = (
        outputs[output_index[prefill_plan["save_id_output"]]]
        if prefill_plan["save_id_output"] else None
    )

    stop_tokens = {
        int(value) for value in (metadata.get("stop_token_ids", "") + "," + metadata.get("eos_token_ids", "")).split(",")
        if value
    }
    generated = []
    selected_id = int(selected.numpy().flat[0])
    if selected_id not in stop_tokens and generation_limit:
        generated.append(selected_id)
    decode_static = _strategy_values(STRATEGY, decode_meta, device_type)
    decode_start = time.perf_counter()
    current_position = prefill_positions[:, -1] + 1
    while len(generated) < generation_limit and selected_id not in stop_tokens:
        decode_binding = decode.io_binding()
        decode_retained = []
        decode_binding.bind_ortvalue_input(decode_plan["token_input"], selected)
        _bind_array(
            decode_binding, decode_retained, decode_meta, decode_plan["position_input"],
            current_position.reshape(3, 1), device_type, DEVICE_ID,
        )
        decode_binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        for name, value in zip(decode_plan["state_in"], state):
            decode_binding.bind_ortvalue_input(name, value)
        if decode_plan["save_id_input"]:
            if saved_ids is None:
                raise RuntimeError("Strategy needs generated-ID state absent from prefill output.")
            decode_binding.bind_ortvalue_input(decode_plan["save_id_input"], saved_ids)
        for name, value in decode_static:
            decode_binding.bind_ortvalue_input(name, value)
        _bind_outputs(decode_binding, decode_plan["outputs"], ort_device)
        _run(decode, decode_binding)
        outputs = decode_binding.get_outputs()
        output_index = {name: index for index, name in enumerate(decode_plan["outputs"])}
        state = outputs[:state_count]
        selected = outputs[output_index[decode_plan["token_output"]]]
        kv_seq_len = outputs[output_index[decode_plan["kv_seq_output"]]]
        if decode_plan["save_id_output"]:
            saved_ids = outputs[output_index[decode_plan["save_id_output"]]]
        selected_id = int(selected.numpy().flat[0])
        if selected_id not in stop_tokens:
            generated.append(selected_id)
        current_position += 1
    decode_elapsed = time.perf_counter() - decode_start
    result = tokenizer.decode(generated, skip_special_tokens=True)
    termination = "stop token" if selected_id in stop_tokens else "token limit"
    total_elapsed = prefill_elapsed + decode_elapsed
    print(result)
    print(
        "Performance\n"
        f"prefill: {prefill_length} tokens, {prefill_elapsed:.3f}s, "
        f"{prefill_length / prefill_elapsed if prefill_elapsed else 0.0:.2f} tokens/s\n"
        f"decode: {len(generated)} tokens, {decode_elapsed:.3f}s, "
        f"{len(generated) / decode_elapsed if decode_elapsed else 0.0:.2f} tokens/s\n"
        f"overall: {prefill_length + len(generated)} tokens, {total_elapsed:.3f}s\n"
        f"termination: {termination}"
    )
    return result


def run_inference(model_folder: Path) -> None:
    """Run one OvisOCR request from the selected ONNX bundle."""
    model_folder = model_folder.expanduser().resolve()
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported STRATEGY: {STRATEGY!r}.")
    providers = list(PROVIDERS)
    if not providers:
        raise ValueError("At least one execution provider is required.")
    device_type, device_kind = _provider_device(providers)
    ort_device = C.OrtDevice(device_kind, C.OrtDevice.default_memory(), DEVICE_ID)
    metadata = load_metadata(model_folder)
    file_names = load_model_file_names(metadata)
    assert_bundle(model_folder, file_names)
    tokenizer = load_tokenizer(model_folder)
    image_paths = INPUT_IMAGES
    image_paths = [path.expanduser().resolve() for path in image_paths]
    image_size = [int(value) for value in metadata["input_image_size"].split(",")]
    preprocess = create_plain_session(model_folder / file_names["image_preprocess"], providers)
    vision = create_plain_session(model_folder / file_names["vision"], providers)
    images = load_images(
        image_paths,
        image_size[0],
        image_size[1],
        int(metadata["vision_batch_size"]),
        len(preprocess.get_inputs()[0].shape),
    )
    vision_outputs = run_vision(preprocess, vision, images, device_type, ort_device, DEVICE_ID)
    result = run_generation(
        model_folder, metadata, file_names, tokenizer, vision_outputs,
        device_type, ort_device, providers,
    )
    if EXPECT_NONEMPTY_OUTPUT and not result.strip():
        raise RuntimeError("Expected nonempty OCR output, but generation was empty.")


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
