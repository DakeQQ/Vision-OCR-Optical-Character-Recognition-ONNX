"""Build SuryaOCR's image-prefill and image-decode merged ONNX graphs.

Surya has six full-attention KV caches and eighteen linear-attention state
pairs.  This helper keeps all 48 states leading in merged graph interfaces,
while language weights are externalized exactly once for every strategy graph.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from OCR_Shared_Initializers import (
    add_shareable_constant_initializers,
    add_shared_initializer,
    redirect_shared_constant_nodes,
)


MIN_SHARED_INITIALIZER_ELEMENTS = 1024
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
PHASES = ("prefill", "decode")
SHARED_MODEL_NAME = "LLM_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
SHELL_PREFIXES = (
    "embed_",
    "concat_image_",
    "prefill_",
    "decode_",
    "greedy_",
    "penalty_greedy_",
    "sampling_",
)
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)


def default_model_file_names() -> dict[str, str]:
    """Return SuryaOCR's complete image-only runtime filename contract."""
    names = {
        "metadata": "LLM_Metadata.onnx",
        "embed": "LLM_Embed.onnx",
        "image_preprocess": "LLM_Image_Preprocess.onnx",
        "vision": "LLM_Vision.onnx",
        "concat_image": "LLM_Concat_Image.onnx",
        "rotary_image_prefill": "LLM_Rotary_Image_Prefill.onnx",
        "rotary_image_decode": "LLM_Rotary_Image_Decode.onnx",
        "main": "LLM_Main.onnx",
        "greedy": "LLM_Greedy.onnx",
        "penalty_greedy": "LLM_PenaltyGreedy.onnx",
        "sampling": "LLM_TopKTopPSampling.onnx",
        "kv_slice": "LLM_KV_Slice.onnx",
        "kv_split2": "LLM_KV_Split2.onnx",
        "kv_concat": "LLM_KV_Concat.onnx",
        "rope_shift": "LLM_RopeShift.onnx",
        "shared_initializers": SHARED_MODEL_NAME,
    }
    for phase in PHASES:
        for strategy in STRATEGIES:
            names[f"image_{phase}_{strategy}"] = (
                f"LLM_Image{phase.capitalize()}"
                f"{''.join(piece.capitalize() for piece in strategy.split('_'))}.onnx"
            )
    names["shared_initializers_data"] = names["shared_initializers"] + ".data"
    return names


_DEFAULT_MODEL_FILE_NAMES = default_model_file_names()
MERGED_CONSTITUENT_GRAPHS = (
    "LLM_Embed.onnx",
    "LLM_Concat_Image.onnx",
    "LLM_Rotary_Image_Prefill.onnx",
    "LLM_Rotary_Image_Decode.onnx",
    "LLM_Main.onnx",
    "LLM_Greedy.onnx",
    "LLM_PenaltyGreedy.onnx",
    "LLM_TopKTopPSampling.onnx",
)


def _file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = _DEFAULT_MODEL_FILE_NAMES if model_file_names is None else model_file_names
    try:
        return names[key]
    except KeyError as error:
        raise KeyError(f"Missing SuryaOCR filename role {key!r}.") from error


def load_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save_model(model, str(path))


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def _copy_external_ref(initializer: TensorProto, external: dict[str, str]) -> TensorProto:
    reference = TensorProto()
    reference.name = initializer.name
    reference.data_type = initializer.data_type
    reference.dims.extend(initializer.dims)
    reference.data_location = TensorProto.EXTERNAL
    for key in ("location", "offset", "length", "checksum", "basepath"):
        if key in external:
            entry = reference.external_data.add()
            entry.key = key
            entry.value = external[key]
    if "location" not in external:
        raise RuntimeError(f"Shared initializer {initializer.name!r} has no data location.")
    return reference


def _external_tensor_path(source_folder: Path, tensor: TensorProto) -> tuple[Path, int, int]:
    external = _external_data_map(tensor)
    location = external.get("location")
    if not location:
        raise RuntimeError(f"External initializer {tensor.name!r} has no data location.")
    relative = Path(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe external initializer location {location!r}.")
    data_path = source_folder / relative
    if not data_path.is_file():
        raise FileNotFoundError(f"External initializer data is missing: {data_path}")
    offset = int(external.get("offset", "0"))
    if "length" not in external:
        raise RuntimeError(f"External initializer {tensor.name!r} has no byte length.")
    length = int(external["length"])
    if offset < 0 or length < 0 or offset + length > data_path.stat().st_size:
        raise RuntimeError(f"External initializer {tensor.name!r} points outside its data file.")
    return data_path, offset, length


def _write_initializer_bytes(data_file, tensor: TensorProto, source_folder: Path) -> int:
    if tensor.raw_data:
        data_file.write(tensor.raw_data)
        return len(tensor.raw_data)
    if tensor.data_location != TensorProto.EXTERNAL:
        raw = numpy_helper.to_array(tensor).tobytes()
        data_file.write(raw)
        return len(raw)
    source_path, offset, length = _external_tensor_path(source_folder, tensor)
    remaining = length
    with source_path.open("rb") as source_file:
        source_file.seek(offset)
        while remaining:
            chunk = source_file.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    f"Unexpected EOF while copying initializer {tensor.name!r}."
                )
            data_file.write(chunk)
            remaining -= len(chunk)
    return length


def _save_shared_initializers(
    shared: dict[str, TensorProto], source_folder: Path, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.with_name(path.name + ".data")
    temporary_data_path = data_path.with_name(data_path.name + ".tmp")
    path.unlink(missing_ok=True)
    temporary_data_path.unlink(missing_ok=True)
    references: list[TensorProto] = []
    offset = 0
    try:
        with temporary_data_path.open("wb") as data_file:
            for name, tensor in sorted(shared.items()):
                length = _write_initializer_bytes(data_file, tensor, source_folder)
                reference = TensorProto()
                reference.name = name
                reference.data_type = tensor.data_type
                reference.dims.extend(tensor.dims)
                reference.data_location = TensorProto.EXTERNAL
                for key, value in (
                    ("location", data_path.name),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ):
                    entry = reference.external_data.add()
                    entry.key = key
                    entry.value = value
                references.append(reference)
                offset += length
        temporary_data_path.replace(data_path)
    except BaseException:
        temporary_data_path.unlink(missing_ok=True)
        raise
    graph = helper.make_graph([], "surya_shared_initializers", [], [], initializer=references)
    model = helper.make_model(graph, producer_name="SuryaOCR.Shared_Merged")
    model.ir_version = 10
    model.metadata_props.add(key="native_llm_shared_initializers", value="1")
    model.metadata_props.add(key="initializer_count", value=str(len(references)))
    onnx.save_model(model, str(path))


def _initializer_elements(initializer: TensorProto) -> int:
    count = 1
    for dimension in initializer.dims:
        count *= int(dimension)
    return count


def _shareable_initializers(
    sources: list[onnx.ModelProto], minimum_elements: int, source_folder: Path | None = None
) -> dict[str, TensorProto]:
    shared: dict[str, TensorProto] = {}
    for source in sources:
        for initializer in source.graph.initializer:
            if initializer.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
                continue
            if _initializer_elements(initializer) >= minimum_elements:
                add_shared_initializer(shared, initializer, source_folder)
        add_shareable_constant_initializers(
            shared, source, minimum_elements, source_folder
        )
    return shared


def shared_external_data_map(path: Path) -> dict[str, dict[str, str]]:
    model = onnx.load(str(path), load_external_data=False)
    return {
        initializer.name: _external_data_map(initializer)
        for initializer in model.graph.initializer
    }


def write_shared_initializers(
    sources: list[onnx.ModelProto],
    path: Path,
    minimum_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, dict[str, str]]:
    shared = _shareable_initializers(sources, minimum_elements)
    if not shared:
        raise RuntimeError("No SuryaOCR language initializers were eligible for sharing.")
    temporary_folder = path.parent
    _save_shared_initializers(shared, temporary_folder, path)
    return shared_external_data_map(path)


def write_shared_initializers_from_external_tensors(
    sources: list[onnx.ModelProto],
    source_folder: Path,
    path: Path,
    minimum_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, dict[str, str]]:
    shared = _shareable_initializers(sources, minimum_elements, Path(source_folder))
    for source in sources:
        for initializer in source.graph.initializer:
            if (
                initializer.data_location == TensorProto.EXTERNAL
                and initializer.data_type not in (TensorProto.UNDEFINED, TensorProto.STRING)
            ):
                add_shared_initializer(shared, initializer, Path(source_folder))
    if not shared:
        raise RuntimeError("No SuryaOCR initializers were eligible for streaming sharing.")
    _save_shared_initializers(shared, Path(source_folder), path)
    return shared_external_data_map(path)


def redirect_shared_initializers_to_external(
    model: onnx.ModelProto, external_by_name: dict[str, dict[str, str]]
) -> int:
    rewritten: list[TensorProto] = []
    redirects = 0
    for initializer in model.graph.initializer:
        external = external_by_name.get(initializer.name)
        if external is None:
            rewritten.append(initializer)
        else:
            rewritten.append(_copy_external_ref(initializer, external))
            redirects += 1
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    redirects += redirect_shared_constant_nodes(
        model, external_by_name, _copy_external_ref
    )
    return redirects


def prefixed(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    import onnx.compose

    return onnx.compose.add_prefix(
        model,
        prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=True,
        rename_outputs=True,
        rename_initializers=True,
        rename_value_infos=True,
    )


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


def copy_metadata(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    metadata = {item.key: item.value for item in destination.metadata_props}
    for source in sources:
        metadata.update({item.key: item.value for item in source.metadata_props})
    destination.ClearField("metadata_props")
    for key in sorted(metadata):
        destination.metadata_props.add(key=key, value=metadata[key])


def merge_models_no_check(
    first: onnx.ModelProto, second: onnx.ModelProto, io_map: list[tuple[str, str]]
) -> onnx.ModelProto:
    """Compose two prefixed graphs without hiding initializer collisions."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)
    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "SuryaOCR.Shared_Merged"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"
    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    seen_inputs: set[str] = set()
    for value in list(first.graph.input) + [
        item for item in second.graph.input if item.name not in mapped_targets
    ]:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)
    initializers: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        existing = initializers.get(initializer.name)
        if existing is None:
            initializers[initializer.name] = initializer
        elif existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Initializer collision with incompatible data: {initializer.name!r}."
            )
    merged.graph.initializer.extend(initializers.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, input_name in enumerate(node.input):
            if input_name in source_by_target:
                node.input[index] = source_by_target[input_name]

    known_values = set(seen_inputs) | set(initializers)
    for value in list(first.graph.value_info) + list(second.graph.value_info):
        if value.name not in known_values:
            merged.graph.value_info.append(value)
            known_values.add(value.name)
    seen_outputs: set[str] = set()
    for value in [
        item for item in first.graph.output if item.name not in mapped_sources
    ] + list(second.graph.output):
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)
    copy_metadata(merged, first, second)
    return merged


def _set_graph_outputs(model: onnx.ModelProto, output_names: list[str]) -> None:
    available = value_info_by_name(model)
    missing = [name for name in output_names if name not in available]
    if missing:
        raise RuntimeError(f"Merged graph lacks output value information: {missing!r}.")
    del model.graph.output[:]
    model.graph.output.extend(available[name] for name in output_names)


def _order_state_inputs_first(model: onnx.ModelProto) -> None:
    state = [item for item in model.graph.input if item.name.startswith("in_")]
    other = [item for item in model.graph.input if not item.name.startswith("in_")]
    del model.graph.input[:]
    model.graph.input.extend(state + other)


def _state_outputs(main: onnx.ModelProto) -> list[str]:
    return [item.name for item in main.graph.output if item.name.startswith("out_")]


def _with_embed(
    source_folder: Path, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None
) -> onnx.ModelProto:
    return embed if embed is not None else prefixed(
        load_model(source_folder / _file_name(model_file_names, "embed")), "embed_"
    )


def _with_rotary(
    source_folder: Path, phase: str, model_file_names: dict[str, str] | None
) -> onnx.ModelProto:
    return prefixed(
        load_model(source_folder / _file_name(model_file_names, f"rotary_image_{phase}")),
        f"{phase}_",
    )


def _prefill_shell(
    source_folder: Path,
    main: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
    embed: onnx.ModelProto | None,
) -> tuple[onnx.ModelProto, str, list[onnx.ModelProto]]:
    embed = _with_embed(source_folder, model_file_names, embed)
    concat = prefixed(
        load_model(source_folder / _file_name(model_file_names, "concat_image")),
        "concat_image_",
    )
    hidden = merge_models_no_check(
        embed, concat, [("embed_text_hidden_states", "concat_image_text_hidden_states")]
    )
    rotary = _with_rotary(source_folder, "prefill", model_file_names)
    merged = merge_models_no_check(hidden, rotary, [])
    merged = merge_models_no_check(
        merged,
        main,
        [
            ("concat_image_concat_hidden_states", "hidden_states"),
            ("prefill_rotary_cos", "rotary_cos"),
            ("prefill_rotary_sin", "rotary_sin"),
            ("prefill_attention_mask", "attention_mask"),
        ],
    )
    return merged, "prefill_kv_seq_len", [concat, rotary]


def _decode_shell(
    source_folder: Path,
    main: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
    embed: onnx.ModelProto | None,
) -> tuple[onnx.ModelProto, str, list[onnx.ModelProto]]:
    embed = _with_embed(source_folder, model_file_names, embed)
    rotary = _with_rotary(source_folder, "decode", model_file_names)
    mask_info = next(item for item in main.graph.input if item.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
    rotary.graph.initializer.append(
        numpy_helper.from_array(
            np.zeros((1, 1, 1, 1, 1), dtype=mask_dtype),
            name="decode_zero_attention_mask",
        )
    )
    merged = merge_models_no_check(embed, rotary, [])
    merged = merge_models_no_check(
        merged,
        main,
        [
            ("embed_text_hidden_states", "hidden_states"),
            ("decode_rotary_cos", "rotary_cos"),
            ("decode_rotary_sin", "rotary_sin"),
            ("decode_zero_attention_mask", "attention_mask"),
        ],
    )
    return merged, "decode_kv_seq_len_next", [rotary]


def _finalize(
    merged: onnx.ModelProto, output_names: list[str], *metadata_sources: onnx.ModelProto
) -> onnx.ModelProto:
    _set_graph_outputs(merged, output_names)
    _order_state_inputs_first(merged)
    copy_metadata(merged, *metadata_sources)
    merged.producer_name = "SuryaOCR.Shared_Merged"
    return merged


def _build_strategy_graph(
    source_folder: Path,
    main: onnx.ModelProto,
    phase: str,
    strategy: str,
    model_file_names: dict[str, str] | None,
    embed: onnx.ModelProto | None,
) -> onnx.ModelProto:
    shell, kv_seq_name, shell_sources = (
        _prefill_shell if phase == "prefill" else _decode_shell
    )(source_folder, main, model_file_names, embed)
    strategy_model = prefixed(
        load_model(source_folder / _file_name(model_file_names, strategy)),
        f"{strategy}_",
    )
    merged = merge_models_no_check(shell, strategy_model, [("logits", f"{strategy}_logits")])
    outputs = _state_outputs(main)
    if strategy == "greedy":
        outputs.extend(["greedy_max_logits_idx", kv_seq_name])
    elif strategy == "penalty_greedy":
        outputs.extend([
            "penalty_greedy_max_logits_idx",
            "penalty_greedy_save_id_out",
            kv_seq_name,
        ])
    elif strategy == "sampling":
        outputs.extend(["sampling_sampled_id", "sampling_save_id_out", kv_seq_name])
    else:
        raise ValueError(f"Unknown decoding strategy {strategy!r}.")
    return _finalize(merged, outputs, main, *shell_sources, strategy_model)


def _recipe(phase: str, strategy: str):
    def build(
        source_folder: Path,
        main: onnx.ModelProto,
        model_file_names: dict[str, str] | None = None,
        embed: onnx.ModelProto | None = None,
    ) -> onnx.ModelProto:
        return _build_strategy_graph(
            source_folder, main, phase, strategy, model_file_names, embed
        )

    build.__name__ = f"merge_image_{phase}_{strategy}"
    return build


def make_merged_build_plan(model_file_names: dict[str, str] | None = None):
    plan = []
    for phase in PHASES:
        for strategy in STRATEGIES:
            dependencies = [
                _file_name(model_file_names, "embed"),
                _file_name(model_file_names, "main"),
                _file_name(model_file_names, f"rotary_image_{phase}"),
                _file_name(model_file_names, strategy),
            ]
            if phase == "prefill":
                dependencies.append(_file_name(model_file_names, "concat_image"))
            plan.append((
                _file_name(model_file_names, f"image_{phase}_{strategy}"),
                _recipe(phase, strategy),
                dependencies,
            ))
    return plan


MERGED_BUILD_PLAN = make_merged_build_plan()


def _constant_tensor(node: onnx.NodeProto) -> TensorProto | None:
    if node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.name == "value" and attribute.HasField("t"):
            return attribute.t
    return None


def cleanup_shadowed_constants(model: onnx.ModelProto) -> int:
    """Remove only an initializer exactly duplicated by a Constant output."""
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    remove: set[str] = set()
    for node in model.graph.node:
        if len(node.output) != 1:
            continue
        constant = _constant_tensor(node)
        if constant is None:
            continue
        initializer = initializers.get(node.output[0])
        if initializer is None:
            continue
        comparable = copy.deepcopy(constant)
        comparable.name = node.output[0]
        if initializer.SerializeToString() != comparable.SerializeToString():
            raise RuntimeError(
                f"Constant output {node.output[0]!r} shadows non-identical initializer data."
            )
        remove.add(node.output[0])
    if remove:
        retained = [item for item in model.graph.initializer if item.name not in remove]
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
    return len(remove)


def _node_is_shell(node: onnx.NodeProto) -> bool:
    return any(output.startswith(SHELL_PREFIXES) for output in node.output)


def _embedding_correction_outputs(model: onnx.ModelProto) -> set[str]:
    """Find a quantized embedding path ending at the shell edge."""
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    final = producers.get("embed_text_hidden_states")
    if final is None:
        return set()
    if final.op_type == "GatherBlockQuantized":
        return {output for output in final.output if output}
    if final.op_type != "Add":
        return set()

    outputs: set[str] = set()
    stack = [final]
    found_quantized_gather = False
    while stack:
        node = stack.pop()
        new_outputs = {output for output in node.output if output}
        if new_outputs <= outputs:
            continue
        outputs.update(new_outputs)
        found_quantized_gather |= node.op_type == "GatherBlockQuantized"
        stack.extend(
            producer
            for input_name in node.input
            if (producer := producers.get(input_name)) is not None
        )
    return outputs if found_quantized_gather else set()


def _used_inputs(nodes: list[onnx.NodeProto]) -> set[str]:
    return {name for node in nodes for name in node.input if name}


def _copy_node_with_remap(node: onnx.NodeProto, remap: dict[str, str]) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _merge_opsets(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    versions: dict[str, int] = {}
    for model in (destination, *sources):
        for opset in model.opset_import:
            versions[opset.domain] = max(versions.get(opset.domain, 0), opset.version)
    del destination.opset_import[:]
    for domain, version in sorted(versions.items()):
        destination.opset_import.add(domain=domain, version=version)


def _target_main_remap(target: onnx.ModelProto) -> dict[str, str]:
    names = {item.name for item in target.graph.input}
    names.update(item.name for item in target.graph.initializer)
    names.update(output for node in target.graph.node for output in node.output)
    if "decode_rotary_cos" in names:
        return {
            "prefill_rotary_cos": "decode_rotary_cos",
            "prefill_rotary_sin": "decode_rotary_sin",
            "prefill_attention_mask": "decode_zero_attention_mask",
            "concat_image_concat_hidden_states": "embed_text_hidden_states",
        }
    if "concat_image_concat_hidden_states" in names:
        return {"embed_text_hidden_states": "concat_image_concat_hidden_states"}
    return {}


def transplant_quantized_main(
    target: onnx.ModelProto, quantized_primary: onnx.ModelProto
) -> onnx.ModelProto:
    """Replace a target's Main block with the canonical optimized primary Main."""
    remap = _target_main_remap(target)
    embedding_correction_outputs = _embedding_correction_outputs(quantized_primary)
    all_donor_nodes = [
        _copy_node_with_remap(node, remap)
        for node in quantized_primary.graph.node
        if (
            not _node_is_shell(node)
            or any(output in embedding_correction_outputs for output in node.output)
        )
    ]
    if not all_donor_nodes:
        raise RuntimeError("The canonical SuryaOCR donor has no Main nodes.")
    embedding_correction_nodes = [
        node
        for node in all_donor_nodes
        if any(output in embedding_correction_outputs for output in node.output)
    ]
    donor_nodes = [
        node
        for node in all_donor_nodes
        if not any(output in embedding_correction_outputs for output in node.output)
    ]
    result = copy.deepcopy(target)
    nodes: list[onnx.NodeProto] = list(embedding_correction_nodes)
    inserted = False
    for node in result.graph.node:
        if (
            embedding_correction_outputs
            and "embed_text_hidden_states" in node.output
        ):
            continue
        if _node_is_shell(node):
            nodes.append(node)
        elif not inserted:
            nodes.extend(donor_nodes)
            inserted = True
    if not inserted:
        nodes.extend(donor_nodes)
    donor_initializers = {item.name: item for item in quantized_primary.graph.initializer}
    target_initializers = {item.name: item for item in target.graph.initializer}
    donor_names = _used_inputs(all_donor_nodes) & set(donor_initializers)
    produced = {output for node in nodes for output in node.output if output}
    used = _used_inputs(nodes) - produced
    initializers: dict[str, TensorProto] = {}
    for name in used:
        if name in donor_names:
            initializers[name] = donor_initializers[name]
        elif name in target_initializers:
            initializers[name] = target_initializers[name]
        elif name in donor_initializers:
            initializers[name] = donor_initializers[name]
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.initializer[:]
    result.graph.initializer.extend(initializers[name] for name in sorted(initializers))
    _merge_opsets(result, quantized_primary)
    cleanup_shadowed_constants(result)
    _order_state_inputs_first(result)
    return result


def _validate_graph_model(model: onnx.ModelProto) -> None:
    initializers: dict[str, bytes] = {}
    for initializer in model.graph.initializer:
        serialized = initializer.SerializeToString()
        previous = initializers.setdefault(initializer.name, serialized)
        if previous != serialized:
            raise RuntimeError(f"Conflicting duplicate initializer {initializer.name!r}.")
    known = {item.name for item in model.graph.input} | set(initializers)
    produced = {output for node in model.graph.node for output in node.output if output}
    undefined = sorted({
        name
        for node in model.graph.node
        for name in node.input
        if name and name not in known and name not in produced
    })
    if undefined:
        raise RuntimeError(f"Merged graph has undefined inputs: {undefined!r}.")


def validate_onnx_path(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    _validate_graph_model(model)
    fused_norms = [
        node for node in model.graph.node
        if not node.domain and node.op_type == "SimplifiedLayerNormalization"
    ]
    if fused_norms:
        for node in fused_norms:
            node.domain = "com.suryaocr.runtime"
        model.opset_import.append(helper.make_opsetid("com.suryaocr.runtime", 1))
        checker_path = path.with_name(f".{path.name}.checker.onnx")
        try:
            onnx.save_model(model, str(checker_path))
            onnx.checker.check_model(str(checker_path))
        finally:
            checker_path.unlink(missing_ok=True)
    else:
        onnx.checker.check_model(str(path))


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
    delete_constituents: bool = False,
) -> dict:
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    plan = make_merged_build_plan(model_file_names)
    missing = sorted({
        dependency
        for _, _, dependencies in plan
        for dependency in dependencies
        if not (source_folder / dependency).exists()
    })
    if missing:
        raise FileNotFoundError(
            "Cannot build a complete SuryaOCR merged bundle; missing: " + ", ".join(missing)
        )
    main = load_model(source_folder / _file_name(model_file_names, "main"))
    embed = prefixed(load_model(source_folder / _file_name(model_file_names, "embed")), "embed_")
    shared_path = out_folder / _file_name(model_file_names, "shared_initializers")
    shared_data_path = out_folder / _file_name(model_file_names, "shared_initializers_data")
    if shared_data_path.name != shared_path.name + ".data":
        raise RuntimeError("Shared data filename must be the shared model filename plus '.data'.")
    shared_sources = [main, embed]
    shared_sources.append(prefixed(
        load_model(source_folder / _file_name(model_file_names, "concat_image")),
        "concat_image_",
    ))
    shared_sources.extend(
        _with_rotary(source_folder, phase, model_file_names) for phase in PHASES
    )
    shared_sources.extend(
        prefixed(
            load_model(source_folder / _file_name(model_file_names, strategy)),
            f"{strategy}_",
        )
        for strategy in STRATEGIES
    )
    external_by_name = write_shared_initializers(
        shared_sources, shared_path, min_shared_elements
    )
    del shared_sources
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)
    graphs: dict[str, Path] = {}
    for filename, recipe, _ in plan:
        merged = recipe(source_folder, main, model_file_names, embed=embed)
        redirect_shared_initializers_to_external(merged, external_by_name)
        output_path = out_folder / filename
        save_model(merged, output_path)
        validate_onnx_path(output_path)
        graphs[filename] = output_path
    expected = {
        _file_name(model_file_names, f"image_{phase}_{strategy}")
        for phase in PHASES
        for strategy in STRATEGIES
    }
    if set(graphs) != expected:
        raise RuntimeError(f"Merged graph mismatch: expected {sorted(expected)}, got {sorted(graphs)}.")
    if not shared_path.exists() or not shared_data_path.exists():
        raise RuntimeError("Shared initializer artifacts were not written.")
    result = {"graphs": graphs, "shared_model": shared_path, "shared_data": shared_data_path}
    if delete_constituents and out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder, protected_names=(shared_path.name, shared_data_path.name)
        )
    return result


def _external_locations(path: Path) -> set[str]:
    model = onnx.load(str(path), load_external_data=False)
    return {
        location
        for initializer in model.graph.initializer
        for location in [_external_data_map(initializer).get("location")]
        if initializer.data_location == TensorProto.EXTERNAL and location
    }


def delete_merged_constituents(
    folder: Path, protected_names: tuple[str, ...] | set[str] | None = None
) -> list[str]:
    """Delete split graphs only after all merged graph validators succeeded."""
    folder = Path(folder)
    protected = set(protected_names or (SHARED_MODEL_NAME, SHARED_DATA_NAME))
    removed: list[str] = []
    for filename in MERGED_CONSTITUENT_GRAPHS:
        path = folder / filename
        if not path.exists():
            continue
        for location in _external_locations(path):
            data_path = folder / location
            if data_path.exists() and data_path.name not in protected:
                data_path.unlink()
                removed.append(data_path.name)
        path.unlink()
        removed.append(path.name)
        sidecar = path.with_name(path.name + ".data")
        if sidecar.exists() and sidecar.name not in protected:
            sidecar.unlink()
            removed.append(sidecar.name)
    return removed
