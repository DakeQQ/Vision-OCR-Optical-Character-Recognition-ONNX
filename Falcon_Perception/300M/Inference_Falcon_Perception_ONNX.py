"""Run Falcon Perception's standalone merged ONNX generation bundle.

This runtime intentionally owns metadata, tokenizer, image, shared-initializer,
I/O planning, and decode-state handling. It never loads another model's runtime.
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
from PIL import Image, ImageDraw
from tokenizers import Tokenizer


# Runtime paths and demo inputs
SCRIPT_DIR           = Path(__file__).resolve().parent
DEFAULT_MODEL_FOLDER = SCRIPT_DIR / "Falcon_Perception_Optimized"
DEFAULT_IMAGE        = SCRIPT_DIR / "psyduck.jpg"

# Inference configuration
QUERY                  = "psyduck"
OUTPUT_IMAGE: Path | None = None
STRATEGY               = "greedy"
MAX_NEW_TOKENS         = 4096
EXPECTED_DETECTIONS: int | None = None
TEMPERATURE            = 0.8
TOP_K                  = 20
TOP_P                  = 0.95
REPETITION_PENALTY     = 1.0
PROVIDERS              = ("CPUExecutionProvider",)
DEVICE_ID              = 0

# Bundle tokenizer and feedback contracts
IMAGE_TOKEN        = "<|image|>"
END_OF_QUERY_TOKEN = "<|end_of_query|>"
STRATEGIES         = ("greedy", "penalty_greedy", "sampling")

# Graph compatibility constants
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_DEFAULT_FILE_NAMES = {
    "metadata": "LLM_Metadata.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "vision": "LLM_Vision.onnx",
    "coordinate_feedback": "LLM_FalconCoordinateFeedback.onnx",
    "size_feedback": "LLM_FalconSizeFeedback.onnx",
    "kv_slice": "LLM_KV_Slice.onnx",
    "kv_split2": "LLM_KV_Split2.onnx",
    "kv_concat": "LLM_KV_Concat.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
    "shared_initializers_data": "LLM_SharedInitializers.onnx.data",
    "image_prefill_greedy": "LLM_ImagePrefillGreedy.onnx",
    "image_prefill_penalty_greedy": "LLM_ImagePrefillPenaltyGreedy.onnx",
    "image_prefill_sampling": "LLM_ImagePrefillSampling.onnx",
    "image_decode_greedy": "LLM_ImageDecodeGreedy.onnx",
    "image_decode_penalty_greedy": "LLM_ImageDecodePenaltyGreedy.onnx",
    "image_decode_sampling": "LLM_ImageDecodeSampling.onnx",
}


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Run Falcon Perception with standalone merged ONNX Runtime graphs."
    )
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    return parser.parse_args().model_folder


def _safe_file_name(value: str, metadata_key: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or ".." in path.parts:
        raise RuntimeError(
            f"Metadata key {metadata_key!r} must contain a safe file name, got {value!r}."
        )
    return value


def _metadata_session(path: Path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def load_metadata(bundle: Path) -> dict[str, str]:
    metadata_path = bundle / _DEFAULT_FILE_NAMES["metadata"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata carrier: {metadata_path}")
    try:
        metadata = dict(_metadata_session(metadata_path).get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise RuntimeError("Unable to load LLM_Metadata.onnx before large model sessions.") from error
    required = (
        "model_type",
        "image_token_id",
        "image_token_length",
        "image_start",
        "image_end",
        "input_image_size",
        "input_image_dim",
        "vision_batch_size",
        "max_seq_len",
        "kv_num_tensors",
        "stop_token_ids",
        "coord_token_id",
        "size_token_id",
        "coord_num_bins",
        "size_num_bins",
        "falcon_feedback_mode",
        "falcon_feedback_backend",
        "falcon_feedback_input",
        "falcon_coordinate_history_layout",
        "falcon_coordinate_history_capacity",
        "falcon_coordinate_history_unused_value",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"LLM_Metadata.onnx is incomplete; missing: {missing!r}.")
    if metadata["model_type"] != "falcon_perception":
        raise RuntimeError(f"Expected Falcon Perception metadata, got {metadata['model_type']!r}.")
    if metadata["falcon_feedback_mode"] != "fourier_coord_size":
        raise RuntimeError("Falcon metadata does not declare the required coordinate/size feedback mode.")
    if metadata["falcon_feedback_backend"] != "onnx_graphs":
        raise RuntimeError("Falcon bundle does not declare ONNX feedback postprocessing graphs.")
    if metadata["falcon_feedback_input"] != "final_hidden_states":
        raise RuntimeError("Falcon bundle does not expose final hidden states to its feedback graphs.")
    if metadata["falcon_coordinate_history_layout"] != "capacity,xy,sentinel":
        raise RuntimeError("Falcon feedback coordinate-history layout is unsupported.")
    if int(metadata["falcon_coordinate_history_capacity"]) < 1:
        raise RuntimeError("Falcon feedback coordinate-history capacity must be positive.")
    if float(metadata["falcon_coordinate_history_unused_value"]) >= 0.0:
        raise RuntimeError("Falcon feedback coordinate-history sentinel must be outside the [0, 1] range.")
    return metadata


def load_model_file_names(metadata: dict[str, str]) -> dict[str, str]:
    return {
        role: _safe_file_name(metadata.get(f"model_file_name_{role}", default), f"model_file_name_{role}")
        for role, default in _DEFAULT_FILE_NAMES.items()
    }


load_file_names = load_model_file_names


def _provider_device(providers: list[str]):
    if "CUDAExecutionProvider" in providers:
        return "cuda", C.OrtDevice.cuda()
    if "DmlExecutionProvider" in providers:
        return "dml", C.OrtDevice.dml()
    return "cpu", C.OrtDevice.cpu()


def create_session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
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


_session_options = create_session_options


def _external_data(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(options: onnxruntime.SessionOptions, shared_path: Path):
    """Attach mmap-backed initializers and return owners that must stay alive."""
    shared = onnx.load(str(shared_path), load_external_data=False)
    arrays: dict[str, np.memmap] = {}
    values: list[onnxruntime.OrtValue] = []
    for initializer in shared.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {initializer.name!r} has no external-data location.")
        data_path = shared_path.parent / _safe_file_name(location, initializer.name)
        if not data_path.is_file():
            raise FileNotFoundError(f"Shared initializer data is missing: {data_path}")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        array = np.memmap(
            data_path,
            dtype=dtype,
            mode="r",
            offset=int(external.get("offset", "0")),
            shape=tuple(int(value) for value in initializer.dims),
        )
        value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        arrays[initializer.name] = array
        values.append(value)
        options.add_initializer(initializer.name, value)
    if not values:
        raise RuntimeError("The shared initializer carrier has no attachable tensors.")
    return shared, arrays, values


def create_plain_session(path: Path, providers: list[str]):
    return onnxruntime.InferenceSession(
        str(path), sess_options=create_session_options(), providers=providers
    )


def create_merged_session(path: Path, shared_path: Path, providers: list[str]):
    options = create_session_options()
    owners = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    session._falcon_shared_initializer_owners = owners
    return session


def _np_dtype(type_name: str):
    for marker, dtype in (
        ("bool", np.bool_),
        ("float16", np.float16),
        ("float", np.float32),
        ("uint8", np.uint8),
        ("int8", np.int8),
        ("uint64", np.uint64),
        ("int64", np.int64),
        ("uint32", np.uint32),
        ("int32", np.int32),
    ):
        if marker in type_name:
            return dtype
    raise ValueError(f"Unsupported ONNX Runtime tensor type: {type_name}")


def _ort_value(array: np.ndarray, device_type: str, device_id: int = DEVICE_ID):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device_type, device_id
    )


def _input_meta(session) -> dict[str, object]:
    return {item.name: item for item in session.get_inputs()}


def _sequence_axis(meta) -> int | None:
    dynamic = [
        index
        for index, dimension in enumerate(meta.shape)
        if index != 0 and not isinstance(dimension, int)
    ]
    return dynamic[0] if len(dynamic) == 1 else None


def _zero_from_meta(meta, batch_size: int = 1) -> np.ndarray:
    shape = list(meta.shape)
    sequence_axis = _sequence_axis(meta)
    for index, dimension in enumerate(shape):
        if index == 0:
            shape[index] = batch_size
        elif index == sequence_axis:
            shape[index] = 0
        elif not isinstance(dimension, int):
            shape[index] = 1
    return np.zeros(tuple(shape), dtype=_np_dtype(meta.type))


def _bind_outputs(binding, names: list[str], ort_device) -> None:
    for name in names:
        binding._iobinding.bind_output(name, ort_device)


def _run(session, binding) -> None:
    session.run_with_iobinding(binding)


def load_falcon_tokenizer(bundle: Path) -> Tokenizer:
    path = bundle / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundle tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def build_falcon_prompt_tokens(tokenizer: Tokenizer, query: str, metadata: dict[str, str]) -> np.ndarray:
    """Recreate the checkpoint's native one-image detection prompt layout."""
    image_token_id = int(metadata["image_token_id"])
    if tokenizer.token_to_id(IMAGE_TOKEN) != image_token_id:
        raise RuntimeError("Bundle tokenizer image token does not match LLM_Metadata.onnx.")
    prompt = f"{IMAGE_TOKEN}Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"
    chunks = [tokenizer.encode(part, add_special_tokens=True).ids for part in prompt.split(IMAGE_TOKEN)]
    if len(chunks) != 2:
        raise RuntimeError("Falcon prompt template must contain exactly one image placeholder.")
    prefix = list(chunks[0])
    image_tokens = int(metadata["image_token_length"])
    image_block = [
        int(metadata["image_cls_token_id"]),
        int(metadata["image_reg_1_token_id"]),
        int(metadata["image_reg_2_token_id"]),
        int(metadata["image_reg_3_token_id"]),
        int(metadata["image_reg_4_token_id"]),
        *([image_token_id] * image_tokens),
        int(metadata["image_end_token_id"]),
    ]
    token_ids = prefix + image_block + list(chunks[1])
    positions = [index for index, token_id in enumerate(token_ids) if token_id == image_token_id]
    expected_start = int(metadata["image_start"])
    expected_end = int(metadata["image_end"])
    if (
        len(positions) != image_tokens
        or positions != list(range(expected_start, expected_end))
        or expected_end - expected_start != image_tokens
    ):
        raise RuntimeError("Falcon image token span differs from the exported metadata contract.")
    return np.asarray([token_ids], dtype=np.int64)


def load_image(path: Path, input_size: tuple[int, int], input_rank: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    height, width = input_size
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), resampling)
        values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)[None]
    if input_rank == 5:
        values = values[:, None]
    elif input_rank != 4:
        raise ValueError(f"Unsupported Falcon image input rank: {input_rank}.")
    return np.ascontiguousarray(values)


def resolve_output_image_path(image_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return image_path.with_name(f"{image_path.stem}_detections{image_path.suffix or '.png'}")
    return output_path.expanduser().resolve()


def draw_yolo_boxes(
    image_path: Path, output_path: Path, detections: list[dict[str, list[float]]]
) -> int:
    with Image.open(image_path) as source:
        annotated = source.convert("RGB")
    image_width, image_height = annotated.size
    drawer = ImageDraw.Draw(annotated)
    stroke_width = max(2, round(min(image_width, image_height) * 0.004))
    boxes_drawn = 0
    for detection in detections:
        center_x, center_y = (float(value) for value in detection["xy"])
        box_height, box_width = (float(value) for value in detection["hw"])
        left = round((center_x - box_width / 2.0) * image_width)
        top = round((center_y - box_height / 2.0) * image_height)
        right = round((center_x + box_width / 2.0) * image_width)
        bottom = round((center_y + box_height / 2.0) * image_height)
        left = min(max(left, 0), image_width - 1)
        top = min(max(top, 0), image_height - 1)
        right = min(max(right, 0), image_width - 1)
        bottom = min(max(bottom, 0), image_height - 1)
        if right <= left or bottom <= top:
            continue
        drawer.rectangle((left, top, right, bottom), outline=(0, 255, 0), width=stroke_width)
        boxes_drawn += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return boxes_drawn


def run_vision(preprocess_session, vision_session, image: np.ndarray, device_type: str, ort_device):
    preprocess_input = preprocess_session.get_inputs()[0]
    preprocess_binding = preprocess_session.io_binding()
    preprocess_binding.bind_ortvalue_input(
        preprocess_input.name,
        _ort_value(image.astype(_np_dtype(preprocess_input.type), copy=False), device_type),
    )
    preprocess_outputs = [item.name for item in preprocess_session.get_outputs()]
    _bind_outputs(preprocess_binding, preprocess_outputs, ort_device)
    _run(preprocess_session, preprocess_binding)
    values = preprocess_binding.get_outputs()

    vision_input = vision_session.get_inputs()[0]
    vision_binding = vision_session.io_binding()
    vision_binding.bind_ortvalue_input(vision_input.name, values[0])
    output_names = [item.name for item in vision_session.get_outputs()]
    _bind_outputs(vision_binding, output_names, ort_device)
    _run(vision_session, vision_binding)
    return dict(zip(output_names, vision_binding.get_outputs()))


def _find_one(names: list[str], predicate, description: str) -> str:
    matches = [name for name in names if predicate(name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {matches!r}.")
    return matches[0]


def plan_merged_io(session, strategy: str, state_count: int, is_decode: bool):
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    state_inputs = [name for name in inputs if name.startswith("in_")]
    state_outputs = [name for name in outputs if name.startswith("out_")]
    if len(state_inputs) != state_count or len(state_outputs) != state_count:
        raise RuntimeError(
            f"Merged graph state contract mismatch: inputs={len(state_inputs)}, outputs={len(state_outputs)}, expected={state_count}."
        )
    if inputs[:state_count] != state_inputs or outputs[:state_count] != state_outputs:
        raise RuntimeError("Merged graph does not place all state tensors first.")
    plan = {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "token_input": _find_one(inputs, lambda name: name.endswith("input_ids"), "token input"),
        "token_output": _find_one(
            outputs,
            lambda name: name.endswith("sampled_id") if strategy == "sampling" else name.endswith("max_logits_idx"),
            "strategy token output",
        ),
        "save_id_input": None,
        "save_id_output": None,
        "kv_seq_input": None,
        "kv_seq_output": _find_one(outputs, lambda name: "kv_seq_len" in name, "KV sequence output"),
        "vision_input": None,
        "final_hidden_states_output": _find_one(
            outputs,
            lambda name: name == "final_hidden_states",
            "Falcon final hidden-state output",
        ),
        "feedback_hidden_input": None,
        "feedback_use_input": None,
    }
    if strategy != "greedy":
        plan["save_id_input"] = _find_one(inputs, lambda name: name.endswith("previous_ids"), "strategy previous-ID input")
        plan["save_id_output"] = _find_one(outputs, lambda name: name.endswith("save_id_out"), "strategy previous-ID output")
    if is_decode:
        plan["kv_seq_input"] = _find_one(inputs, lambda name: "kv_seq_len" in name, "decode KV sequence input")
        plan["feedback_hidden_input"] = _find_one(inputs, lambda name: name == "falcon_feedback_hidden_states", "Falcon feedback hidden-state input")
        plan["feedback_use_input"] = _find_one(inputs, lambda name: name == "falcon_use_feedback", "Falcon feedback selector input")
    else:
        plan["vision_input"] = _find_one(inputs, lambda name: name.endswith("vision_hidden_states"), "prefill vision input")
    return plan


def _strategy_values(strategy: str, input_meta: dict[str, object], device_type: str):
    if REPETITION_PENALTY <= 0.0:
        raise ValueError("REPETITION_PENALTY must be positive.")
    values: list[tuple[str, onnxruntime.OrtValue]] = []
    for name, meta in input_meta.items():
        if name.endswith("temperature"):
            if TEMPERATURE <= 0.0:
                raise ValueError("TEMPERATURE must be positive for sampling.")
            array = np.asarray([TEMPERATURE], dtype=_np_dtype(meta.type))
        elif name.endswith("top_k"):
            if TOP_K < 1:
                raise ValueError("TOP_K must be positive for sampling.")
            array = np.asarray(TOP_K, dtype=_np_dtype(meta.type))
        elif name.endswith("top_p"):
            if not 0.0 < TOP_P <= 1.0:
                raise ValueError("TOP_P must be in (0, 1].")
            array = np.asarray([TOP_P], dtype=_np_dtype(meta.type))
        elif name.endswith("repetition_penalty"):
            array = np.asarray([[REPETITION_PENALTY]], dtype=_np_dtype(meta.type))
        else:
            continue
        values.append((name, _ort_value(array, device_type)))
    if strategy == "sampling" and len(values) != 4:
        raise RuntimeError("Sampling graph does not expose the expected four static controls.")
    if strategy == "penalty_greedy" and len(values) != 1:
        raise RuntimeError("Penalty-greedy graph does not expose repetition_penalty.")
    if strategy == "greedy" and values:
        raise RuntimeError("Greedy graph unexpectedly exposes strategy control inputs.")
    return values


def _bind_array(binding, owners, input_meta, name, array, device_type: str):
    value = _ort_value(np.asarray(array, dtype=_np_dtype(input_meta[name].type)), device_type)
    owners.append(value)
    binding.bind_ortvalue_input(name, value)


def _feedback_zeros(meta) -> np.ndarray:
    shape = [1 if not isinstance(dimension, int) else dimension for dimension in meta.shape]
    return np.zeros(tuple(shape), dtype=_np_dtype(meta.type))


def _validate_feedback_graph(session, expected_inputs: tuple[str, ...], expected_outputs: tuple[str, ...], role: str) -> None:
    actual_inputs = tuple(item.name for item in session.get_inputs())
    actual_outputs = tuple(item.name for item in session.get_outputs())
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise RuntimeError(
            f"Unexpected Falcon {role} feedback graph I/O: "
            f"inputs={actual_inputs!r}, outputs={actual_outputs!r}."
        )


def _run_feedback_graph(session, inputs: dict[str, onnxruntime.OrtValue], output_names: tuple[str, ...], ort_device):
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    _bind_outputs(binding, list(output_names), ort_device)
    _run(session, binding)
    return dict(zip(output_names, binding.get_outputs()))


def run_generation(model_folder, metadata, file_names, vision_outputs, tokenizer, image_path, output_image, device_type, ort_device, providers):
    bundle = model_folder
    shared_path = bundle / file_names["shared_initializers"]
    shared_data_path = bundle / file_names["shared_initializers_data"]
    if not shared_path.is_file() or not shared_data_path.is_file():
        raise FileNotFoundError("Merged runtime requires LLM_SharedInitializers.onnx and its .data sidecar.")
    prefill = create_merged_session(
        bundle / file_names[f"image_prefill_{STRATEGY}"], shared_path, providers
    )
    decode = create_merged_session(
        bundle / file_names[f"image_decode_{STRATEGY}"], shared_path, providers
    )
    coordinate_feedback = create_plain_session(
        bundle / file_names["coordinate_feedback"], providers
    )
    size_feedback = create_plain_session(bundle / file_names["size_feedback"], providers)
    coordinate_feedback_inputs = (
        "final_hidden_states",
        "coordinate_history",
    )
    coordinate_feedback_outputs = (
        "feedback_hidden_states",
        "falcon_use_feedback",
        "next_coordinate_history",
        "coordinate_values",
    )
    size_feedback_inputs = ("final_hidden_states",)
    size_feedback_outputs = (
        "feedback_hidden_states",
        "falcon_use_feedback",
        "size_values",
    )
    _validate_feedback_graph(
        coordinate_feedback,
        coordinate_feedback_inputs,
        coordinate_feedback_outputs,
        "coordinate",
    )
    _validate_feedback_graph(
        size_feedback, size_feedback_inputs, size_feedback_outputs, "size"
    )
    state_count = int(metadata["kv_num_tensors"])
    prefill_plan = plan_merged_io(prefill, STRATEGY, state_count, is_decode=False)
    decode_plan = plan_merged_io(decode, STRATEGY, state_count, is_decode=True)
    prefill_meta = _input_meta(prefill)
    decode_meta = _input_meta(decode)
    coordinate_feedback_meta = _input_meta(coordinate_feedback)

    token_ids = build_falcon_prompt_tokens(tokenizer, QUERY, metadata)
    prompt_length = int(token_ids.shape[1])
    if prompt_length >= int(metadata["max_seq_len"]):
        raise ValueError("Falcon prompt length reaches the exported maximum sequence length.")
    token_ids = token_ids.astype(_np_dtype(prefill_meta[prefill_plan["token_input"]].type), copy=False)
    max_new_tokens = min(MAX_NEW_TOKENS, int(metadata["max_seq_len"]) - prompt_length)
    if max_new_tokens < 0:
        raise ValueError("MAX_NEW_TOKENS must be non-negative.")
    coordinate_history_capacity = int(metadata["falcon_coordinate_history_capacity"])
    if coordinate_history_capacity < max_new_tokens:
        raise RuntimeError("Falcon feedback coordinate-history capacity is smaller than the decode limit.")

    binding = prefill.io_binding()
    owners: list[onnxruntime.OrtValue] = []
    bound: set[str] = set()
    _bind_array(binding, owners, prefill_meta, prefill_plan["token_input"], token_ids, device_type)
    bound.add(prefill_plan["token_input"])
    for name, value in (("prefill_ids_len", [prompt_length]), ("prefill_history_len", [0])):
        if name in prefill_meta:
            _bind_array(binding, owners, prefill_meta, name, np.asarray(value), device_type)
            bound.add(name)
    vision_name = prefill_plan["vision_input"]
    if "vision_hidden_states" not in vision_outputs:
        raise RuntimeError("Falcon vision graph did not return vision_hidden_states.")
    binding.bind_ortvalue_input(vision_name, vision_outputs["vision_hidden_states"])
    bound.add(vision_name)
    state_device = "cpu" if device_type == "dml" else device_type
    for name in prefill_plan["state_inputs"]:
        _bind_array(binding, owners, prefill_meta, name, _zero_from_meta(prefill_meta[name]), state_device)
        bound.add(name)
    if prefill_plan["save_id_input"] is not None:
        _bind_array(binding, owners, prefill_meta, prefill_plan["save_id_input"], _zero_from_meta(prefill_meta[prefill_plan["save_id_input"]]), device_type)
        bound.add(prefill_plan["save_id_input"])
    for name, value in _strategy_values(STRATEGY, prefill_meta, device_type):
        owners.append(value)
        binding.bind_ortvalue_input(name, value)
        bound.add(name)
    unbound = [name for name in prefill_plan["inputs"] if name not in bound]
    if unbound:
        raise RuntimeError(f"Prefill graph has unbound inputs: {unbound!r}.")
    _bind_outputs(binding, prefill_plan["outputs"], ort_device)

    prefill_started = time.perf_counter()
    _run(prefill, binding)
    prefill_elapsed = time.perf_counter() - prefill_started
    outputs = dict(zip(prefill_plan["outputs"], binding.get_outputs()))
    cached_state = [outputs[name] for name in prefill_plan["state_outputs"]]
    selected = outputs[prefill_plan["token_output"]]
    kv_seq_len = outputs[prefill_plan["kv_seq_output"]]
    saved_ids = outputs[prefill_plan["save_id_output"]] if prefill_plan["save_id_output"] else None
    final_hidden_states = outputs[prefill_plan["final_hidden_states_output"]]

    stop_tokens = {
        int(value)
        for value in (metadata.get("stop_token_ids") or metadata.get("eos_token_ids", "")).split(",")
        if value
    }
    end_of_query_id = tokenizer.token_to_id(END_OF_QUERY_TOKEN)
    if end_of_query_id is not None:
        stop_tokens.add(int(end_of_query_id))
    if not stop_tokens:
        raise RuntimeError("Falcon metadata has no stop token IDs.")
    generated: list[int] = []
    coordinate_history = _ort_value(
        np.full(
            (coordinate_history_capacity, 2),
            float(metadata["falcon_coordinate_history_unused_value"]),
            dtype=_np_dtype(coordinate_feedback_meta["coordinate_history"].type),
        ),
        device_type,
    )
    pending_coordinate: np.ndarray | None = None
    detections: list[dict[str, list[float]]] = []
    disabled_feedback_hidden = _ort_value(
        _feedback_zeros(decode_meta[decode_plan["feedback_hidden_input"]]), device_type
    )
    disabled_feedback_use = _ort_value(
        np.zeros(
            (1, 1, 1),
            dtype=_np_dtype(decode_meta[decode_plan["feedback_use_input"]].type),
        ),
        device_type,
    )
    selected_id = int(selected.numpy().flat[0])
    if selected_id not in stop_tokens and max_new_tokens:
        generated.append(selected_id)

    decode_static = _strategy_values(STRATEGY, decode_meta, device_type)
    decode_started = time.perf_counter()
    while len(generated) < max_new_tokens and selected_id not in stop_tokens:
        binding = decode.io_binding()
        owners = []
        bound = set()
        binding.bind_ortvalue_input(decode_plan["token_input"], selected)
        bound.add(decode_plan["token_input"])
        binding.bind_ortvalue_input(decode_plan["kv_seq_input"], kv_seq_len)
        bound.add(decode_plan["kv_seq_input"])
        for name, value in zip(decode_plan["state_inputs"], cached_state):
            binding.bind_ortvalue_input(name, value)
            bound.add(name)
        if selected_id == int(metadata["coord_token_id"]):
            feedback_outputs = _run_feedback_graph(
                coordinate_feedback,
                {
                    "final_hidden_states": final_hidden_states,
                    "coordinate_history": coordinate_history,
                },
                coordinate_feedback_outputs,
                ort_device,
            )
            feedback_hidden = feedback_outputs["feedback_hidden_states"]
            feedback_use = feedback_outputs["falcon_use_feedback"]
            coordinate_history = feedback_outputs["next_coordinate_history"]
            pending_coordinate = feedback_outputs["coordinate_values"].numpy().reshape(2)
        elif selected_id == int(metadata["size_token_id"]):
            feedback_outputs = _run_feedback_graph(
                size_feedback,
                {"final_hidden_states": final_hidden_states},
                size_feedback_outputs,
                ort_device,
            )
            feedback_hidden = feedback_outputs["feedback_hidden_states"]
            feedback_use = feedback_outputs["falcon_use_feedback"]
            size = feedback_outputs["size_values"].numpy().reshape(2)
            if pending_coordinate is not None:
                detections.append({"xy": pending_coordinate.tolist(), "hw": size.tolist()})
                pending_coordinate = None
        else:
            feedback_hidden = disabled_feedback_hidden
            feedback_use = disabled_feedback_use
        binding.bind_ortvalue_input(decode_plan["feedback_hidden_input"], feedback_hidden)
        bound.add(decode_plan["feedback_hidden_input"])
        binding.bind_ortvalue_input(decode_plan["feedback_use_input"], feedback_use)
        bound.add(decode_plan["feedback_use_input"])
        if decode_plan["save_id_input"] is not None:
            if saved_ids is None:
                raise RuntimeError("Decode requires generated-ID state missing from prefill.")
            binding.bind_ortvalue_input(decode_plan["save_id_input"], saved_ids)
            bound.add(decode_plan["save_id_input"])
        for name, value in decode_static:
            binding.bind_ortvalue_input(name, value)
            bound.add(name)
        unbound = [name for name in decode_plan["inputs"] if name not in bound]
        if unbound:
            raise RuntimeError(f"Decode graph has unbound inputs: {unbound!r}.")
        _bind_outputs(binding, decode_plan["outputs"], ort_device)
        _run(decode, binding)
        outputs = dict(zip(decode_plan["outputs"], binding.get_outputs()))
        cached_state = [outputs[name] for name in decode_plan["state_outputs"]]
        selected = outputs[decode_plan["token_output"]]
        kv_seq_len = outputs[decode_plan["kv_seq_output"]]
        final_hidden_states = outputs[decode_plan["final_hidden_states_output"]]
        if decode_plan["save_id_output"] is not None:
            saved_ids = outputs[decode_plan["save_id_output"]]
        selected_id = int(selected.numpy().flat[0])
        if selected_id not in stop_tokens:
            generated.append(selected_id)
    decode_elapsed = time.perf_counter() - decode_started

    text = tokenizer.decode(generated, skip_special_tokens=True)
    total_elapsed = prefill_elapsed + decode_elapsed
    stop_reason = "stop_token" if selected_id in stop_tokens else "max_new_tokens"
    print(f"Providers: {decode.get_providers()}")
    print(f"Strategy: {STRATEGY}")
    print(f"Output: {text}")
    detection_preview = detections[:3]
    suffix = " ..." if len(detections) > len(detection_preview) else ""
    print(f"Detections: {len(detections)} {detection_preview}{suffix}")
    if EXPECTED_DETECTIONS is not None and len(detections) != EXPECTED_DETECTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_DETECTIONS} detections, got {len(detections)}."
        )
    boxes_drawn = draw_yolo_boxes(image_path, output_image, detections)
    print(f"Annotated image: {output_image} ({boxes_drawn} boxes)")
    print(f"Stop reason: {stop_reason}")
    print(
        "Performance: "
        f"prefill={prompt_length} tokens/{prefill_elapsed:.3f}s "
        f"decode={len(generated)} tokens/{decode_elapsed:.3f}s "
        f"overall={(prompt_length + len(generated)) / total_elapsed if total_elapsed else 0.0:.2f} tokens/s"
    )
    return text, generated, selected_id in stop_tokens


def validate_bundle_files(
    bundle: Path,
    file_names: dict[str, str],
    metadata: dict[str, str] | None = None,
):
    required = [
        "image_preprocess",
        "vision",
        "coordinate_feedback",
        "size_feedback",
        "shared_initializers",
        "shared_initializers_data",
        *[f"image_{phase}_{strategy}" for phase in ("prefill", "decode") for strategy in STRATEGIES],
    ]
    missing = [file_names[name] for name in required if not (bundle / file_names[name]).is_file()]
    if missing:
        raise FileNotFoundError(f"Falcon merged bundle is incomplete: {missing!r}.")


def run_inference(model_folder: Path) -> None:
    """Run one Falcon OCR request from the selected ONNX bundle."""
    if MAX_NEW_TOKENS < 0:
        raise ValueError("MAX_NEW_TOKENS must be non-negative.")
    if EXPECTED_DETECTIONS is not None and EXPECTED_DETECTIONS < 0:
        raise ValueError("EXPECTED_DETECTIONS must be non-negative.")
    if STRATEGY not in STRATEGIES:
        raise ValueError(f"Unsupported STRATEGY: {STRATEGY!r}.")
    model_folder = model_folder.expanduser().resolve()
    image_path = DEFAULT_IMAGE.expanduser().resolve()
    output_image = resolve_output_image_path(image_path, OUTPUT_IMAGE)
    providers = list(PROVIDERS)
    if not providers:
        raise ValueError("At least one ONNX Runtime execution provider is required.")
    device_type, device_kind = _provider_device(providers)
    ort_device = C.OrtDevice(device_kind, C.OrtDevice.default_memory(), DEVICE_ID)

    metadata = load_metadata(model_folder)
    file_names = load_model_file_names(metadata)
    validate_bundle_files(model_folder, file_names, metadata)
    tokenizer = load_falcon_tokenizer(model_folder)
    input_size = tuple(int(value) for value in metadata["input_image_size"].split(","))
    image = load_image(image_path, input_size, int(metadata["input_image_dim"]))
    preprocess = create_plain_session(model_folder / file_names["image_preprocess"], providers)
    vision = create_plain_session(model_folder / file_names["vision"], providers)
    vision_outputs = run_vision(preprocess, vision, image, device_type, ort_device)
    run_generation(
        model_folder,
        metadata,
        file_names,
        vision_outputs,
        tokenizer,
        image_path,
        output_image,
        device_type,
        ort_device,
        providers,
    )


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
