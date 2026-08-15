"""Validate that same-name OCR shared initializers have identical payloads."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Callable, MutableMapping

import onnx
from onnx import TensorProto, numpy_helper


_PACKED_NIBBLE_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4")
    if hasattr(TensorProto, name)
)
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _external_data(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _external_payload_length(tensor: TensorProto, metadata: dict[str, str]) -> int:
    length = metadata.get("length")
    if length is not None:
        return int(length)

    elements = 1
    for dimension in tensor.dims:
        elements *= int(dimension)
    if tensor.data_type in _PACKED_NIBBLE_TYPES:
        return (elements + 1) // 2
    try:
        return elements * onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize
    except KeyError as error:
        raise RuntimeError(
            f"External initializer {tensor.name!r} has no byte length or NumPy dtype."
        ) from error


def _payload_digest(tensor: TensorProto, source_folder: Path | None) -> bytes:
    digest = sha256()
    if tensor.raw_data:
        digest.update(tensor.raw_data)
        return digest.digest()

    if tensor.data_location != TensorProto.EXTERNAL:
        digest.update(numpy_helper.to_array(tensor).tobytes())
        return digest.digest()

    if source_folder is None:
        raise RuntimeError(
            f"Cannot compare external shared initializer {tensor.name!r} without its source folder."
        )
    metadata = _external_data(tensor)
    location = metadata.get("location")
    if not location:
        raise RuntimeError(f"External initializer {tensor.name!r} has no data location.")
    relative = Path(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"External initializer {tensor.name!r} has unsafe data location {location!r}."
        )
    data_path = Path(source_folder) / relative
    if not data_path.is_file():
        raise FileNotFoundError(f"External initializer data is missing: {data_path}")
    offset = int(metadata.get("offset", "0"))
    length = _external_payload_length(tensor, metadata)
    if offset < 0 or length < 0 or offset + length > data_path.stat().st_size:
        raise RuntimeError(f"External initializer {tensor.name!r} points outside {data_path}.")
    with data_path.open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining:
            chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError(
                    f"Unexpected EOF while hashing external initializer {tensor.name!r}."
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.digest()


def add_shared_initializer(
    shared: MutableMapping[str, TensorProto],
    initializer: TensorProto,
    source_folder: Path | None = None,
) -> bool:
    """Add an initializer, rejecting a same-name tensor with different bytes."""
    existing = shared.get(initializer.name)
    if existing is None:
        shared[initializer.name] = initializer
        return True
    if (
        existing.data_type != initializer.data_type
        or tuple(existing.dims) != tuple(initializer.dims)
        or _payload_digest(existing, source_folder) != _payload_digest(initializer, source_folder)
    ):
        raise RuntimeError(
            f"Conflicting shared initializer {initializer.name!r}: same name has different "
            "dtype, shape, or payload."
        )
    return False


def constant_as_initializer(node: onnx.NodeProto) -> TensorProto | None:
    """Return a top-level Constant value as an initializer named after its output."""
    if node.op_type != "Constant" or len(node.output) != 1 or not node.output[0]:
        return None
    for attribute in node.attribute:
        if attribute.name == "value" and attribute.HasField("t"):
            initializer = TensorProto()
            initializer.CopyFrom(attribute.t)
            initializer.name = node.output[0]
            return initializer
    return None


def _tensor_elements(tensor: TensorProto) -> int:
    elements = 1
    for dimension in tensor.dims:
        elements *= int(dimension)
    return elements


def add_shareable_constant_initializers(
    shared: MutableMapping[str, TensorProto],
    model: onnx.ModelProto,
    minimum_elements: int,
    source_folder: Path | None = None,
) -> int:
    """Collect large top-level Constant tensors into the shared initializer map."""
    added = 0
    for node in model.graph.node:
        initializer = constant_as_initializer(node)
        if initializer is None:
            continue
        if initializer.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
            continue
        if _tensor_elements(initializer) >= minimum_elements:
            added += int(add_shared_initializer(shared, initializer, source_folder))
    return added


def redirect_shared_constant_nodes(
    model: onnx.ModelProto,
    external_by_name: dict[str, dict[str, str]],
    copy_external_ref: Callable[[TensorProto, dict[str, str]], TensorProto],
) -> int:
    """Replace shared top-level Constant nodes with external initializer references."""
    existing_names = {initializer.name for initializer in model.graph.initializer}
    retained_nodes: list[onnx.NodeProto] = []
    references: list[TensorProto] = []
    redirects = 0
    for node in model.graph.node:
        initializer = constant_as_initializer(node)
        external = (
            external_by_name.get(initializer.name)
            if initializer is not None
            else None
        )
        if external is None:
            retained_nodes.append(node)
            continue
        if initializer.name in existing_names:
            raise RuntimeError(
                f"Shared Constant output collides with an initializer: {initializer.name!r}."
            )
        references.append(copy_external_ref(initializer, external))
        existing_names.add(initializer.name)
        redirects += 1
    if references:
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        model.graph.initializer.extend(references)
    return redirects