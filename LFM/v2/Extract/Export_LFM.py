"""Export LFM2-350M-Extract into a staged, metadata-driven ONNX bundle.

LFM2-350M-Extract is a text-only hybrid causal model.  Its runtime state is six
attention KV cache pairs plus ten short-convolution states; no image graph is
emitted for this checkpoint family.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import Shared_Merged
from LFM_Text_Components import (
    GREEDY_SEARCH,
    KV_CONCAT,
    KV_ROPE_SHIFT,
    KV_SLICE,
    KV_SPLIT2,
    KVCacheSettings,
    LFM_EMBED,
    LFM_MAIN,
    METADATA_CARRIER,
    PENALTY_GREEDY_SEARCH,
    ROTARY_DECODE,
    ROTARY_PREFILL,
    TOPK_TOPP_SAMPLING,
    build_cache_io,
    cache_state_dynamic_axes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

CHECKPOINT_DIR = Path.home() / "Downloads" / "LFM2-350M-Extract"
# Backward-compatible configuration alias.
MODEL_FOLDER = CHECKPOINT_DIR
EXPORT_DIR = SCRIPT_DIR / "LFM_ONNX"
EXPORT_STAGING_DIR = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")

# Export controls
DO_EXPORT   = True                        # Whether to export the ONNX models.
MAX_SEQ_LEN = 4096                        # Fixed maximum sequence length after export.

# KV cache storage and attention precision
# F16 is the existing fast default; F32 is the reference baseline.
KV_QUANT_DTYPE      = "Q8"               # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 64                  # Quantization group width.
COMPUTE_IN_F32      = False               # F16 KV only: False keeps attention in F16, True upcasts KV reads.

# KV quantization transforms and parameters
USE_HADAMARD           = False            # Apply a Hadamard transform before grouped quantization.
HADAMARD_RANDOM_SEED   = 9527             # Deterministic sign pattern for the Hadamard transform.
USE_CLIP               = False            # Clip outliers before KV quantization.
CLIP_SIGMA             = 3.0              # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False            # Spread channels across grouped quantization blocks.
USE_SYM                = True             # True=signed absmax; False=min-max with bias.
USE_FLOAT16_SCALE_BIAS = True             # Store quantization scales and biases as float16.
USE_QDQ_FRIENDLY_ASYM  = False            # Disable residual bias correction for asymmetric QDQ compatibility.

# Quantization-oriented model reordering
REORDER_DOWNPROJ_FOR_QUANT   = True       # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True       # Reorder LFM2 vision MLP channels before down-projection quantization.
REORDER_KEY                  = "absmean"  # Channel statistic used to build the permutation.

# ONNX graph format
OPSET = 20                                # ONNX opset version.

MODEL_FILE_NAMES = Shared_Merged.default_model_file_names()
MODEL_FILE_NAMES.update({
    "kv_slice": "LLM_KV_Slice.onnx",
    "kv_split2": "LLM_KV_Split2.onnx",
    "kv_concat": "LLM_KV_Concat.onnx",
    "rope_shift": "LLM_RopeShift.onnx",
})
RUNTIME_MODEL_FILE_ROLES = (
    "metadata",
    "kv_slice",
    "kv_split2",
    "kv_concat",
    "rope_shift",
    "shared_initializers",
    "shared_initializers_data",
    "text_prefill_greedy",
    "text_prefill_penalty_greedy",
    "text_prefill_sampling",
    "text_decode_greedy",
    "text_decode_penalty_greedy",
    "text_decode_sampling",
)
MODEL_FILE_NAME_METADATA = {
    f"model_file_name_{role}": MODEL_FILE_NAMES[role]
    for role in RUNTIME_MODEL_FILE_ROLES
}


def _id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _stamp_metadata(path: Path, metadata: dict[str, str]) -> None:
    """Update metadata without resolving large external initializer payloads."""
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(
    path: Path,
    module: torch.nn.Module,
    args: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None,
    metadata: dict[str, str],
) -> None:
    module.eval()
    print(f"Exporting {path.name} ...", flush=True)
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
    )
    _stamp_metadata(path, metadata)
    Shared_Merged.validate_onnx_path(path)
    print(f"Exported {path.name}.", flush=True)


def _prepare_export_staging() -> Path:
    if EXPORT_STAGING_DIR.exists():
        if not EXPORT_STAGING_DIR.is_dir():
            raise NotADirectoryError(f"Export staging path is not a directory: {EXPORT_STAGING_DIR}")
        shutil.rmtree(EXPORT_STAGING_DIR)
    EXPORT_STAGING_DIR.mkdir(parents=True)
    return EXPORT_STAGING_DIR


def _promote_export(staging_dir: Path) -> None:
    """Atomically activate the complete bundle while retaining one rollback."""
    destination = EXPORT_DIR
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    if destination.exists():
        destination.rename(previous)
    staging_dir.rename(destination)


def _cleanup_unreferenced_data(export_dir: Path) -> None:
    referenced = set()
    for model_path in export_dir.glob("*.onnx"):
        model = onnx.load(str(model_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location != onnx.TensorProto.EXTERNAL:
                continue
            external = {item.key: item.value for item in initializer.external_data}
            location = external.get("location")
            if location:
                referenced.add(Path(location).name)
    for candidate in export_dir.iterdir():
        if candidate.is_file() and candidate.suffix != ".onnx" and candidate.name not in referenced:
            candidate.unlink()


def _metadata_values(model, tokenizer, main: LFM_MAIN) -> dict[str, str]:
    config = model.config
    layout = main.kv_layout
    settings = main.kv_settings
    eos_ids = _id_list(getattr(config, "eos_token_id", getattr(tokenizer, "eos_token_id", None)))
    metadata = {
        "model_family": "lfm2_extract_text",
        "input_modality": "text",
        "max_seq_len": str(MAX_SEQ_LEN),
        "stop_token_ids": ",".join(str(item) for item in eos_ids),
        "eos_token_ids": ",".join(str(item) for item in eos_ids),
        "vocab_size": str(int(config.vocab_size)),
        "hidden_size": str(main.hidden_size),
        "num_layers": str(len(main.backbone.layers)),
        "num_attention_layers": str(main.num_attn_layers),
        "num_convolution_layers": str(main.num_conv_layers),
        "num_attention_heads": str(main.num_heads),
        "num_key_value_heads": str(main.num_key_value_heads),
        "head_dim": str(main.head_dim),
        "kv_num_tensors": str(layout.state_count),
        "kv_blocks_per_layer": str(layout.blocks_per_attention_layer),
        "kv_quant_dtype": settings.quant_dtype,
        "kv_quant_group_size": str(settings.quant_group_size),
        "compute_in_f32": str(int(settings.compute_in_f32)),
        "kv_cache_quantization": settings.quant_dtype,
        "kv_cache_tensor_order": ",".join(group.name for group in layout.groups),
        "kv_cache_key_sequence_axis": "4",
        "kv_cache_value_sequence_axis": "3",
        "kv_cache_key_storage_width": str(layout.key_storage_width),
        "kv_cache_value_storage_width": str(layout.value_storage_width),
        "kv_cache_storage_dtype": str(layout.cache_dtype).replace("torch.", ""),
        "kv_cache_scale_bias_dtype": (
            str(layout.scale_bias_dtype).replace("torch.", "")
            if layout.scale_bias_dtype is not None else "none"
        ),
        "kv_cache_quantized": str(int(layout.is_quantized)),
        "kv_cache_symmetric": str(int(layout.is_symmetric)),
        "kv_cache_grouped_6d": str(int(layout.uses_grouped_layout)),
        "kv_cache_group_count": str(layout.group_count if layout.is_quantized else 0),
        "kv_quant_hadamard": str(int(settings.use_hadamard)) if layout.is_quantized else "0",
        "kv_quant_shuffle": str(int(settings.use_shuffle)) if layout.is_quantized else "0",
        "kv_quant_clip": str(int(settings.use_clip)) if layout.is_quantized else "0",
        "conv_state_sequence_axis": "2",
        "conv_state_storage_dtype": str(layout.tensor_dtype("conv")).replace("torch.", ""),
        "fused_simplified_layer_norm_count": str(2 * len(main.backbone.layers) + main.num_attn_layers + 1),
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reorder_key": REORDER_KEY,
        "reordered_language_pair_count": str(main.reorder_summary.language_pairs),
        "reorder_max_equivalence_error": repr(main.reorder_summary.maximum_error),
        "embed_lm_head_tied": "1",
    }
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


def _export_kv_helpers(
    staging_dir: Path,
    main: LFM_MAIN,
    rope_inv_freq: torch.Tensor,
    metadata: dict[str, str],
) -> None:
    """Export standalone cache-window and RoPE-shift utilities for the selected KV layout."""
    layout = main.kv_layout
    kv_groups = layout.kv_groups
    kv_tensors = [
        layout.zeros(group.name, 1, sequence_length=8)
        for group in kv_groups
        for _ in range(group.count)
    ]
    kv_input_names = [
        f"in_{group.name}_{layer_index}"
        for group in kv_groups
        for layer_index in range(group.count)
    ]
    kv_output_names = [
        f"out_{group.name}_{layer_index}"
        for group in kv_groups
        for layer_index in range(group.count)
    ]
    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([4], dtype=torch.int64)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["kv_slice"],
        KV_SLICE(layout),
        tuple(kv_tensors + [slice_start, slice_end]),
        kv_input_names + ["slice_start", "slice_end"],
        kv_output_names,
        {
            **cache_state_dynamic_axes(layout, kv_input_names, "history_len", groups=kv_groups),
            **cache_state_dynamic_axes(layout, kv_output_names, "sliced_len", groups=kv_groups),
        },
        metadata,
    )

    split_at = torch.tensor([4], dtype=torch.int64)
    prefix_names = [f"prefix_{name}" for name in kv_output_names]
    suffix_names = [f"window_{name}" for name in kv_output_names]
    _export_component(
        staging_dir / MODEL_FILE_NAMES["kv_split2"],
        KV_SPLIT2(layout),
        tuple(kv_tensors + [split_at]),
        kv_input_names + ["split_at"],
        prefix_names + suffix_names,
        {
            **cache_state_dynamic_axes(layout, kv_input_names, "history_len", groups=kv_groups),
            **cache_state_dynamic_axes(layout, prefix_names, "prefix_len", groups=kv_groups),
            **cache_state_dynamic_axes(layout, suffix_names, "window_len", groups=kv_groups),
        },
        metadata,
    )

    prefix_inputs = [name.replace("in_", "in_a_", 1) for name in kv_input_names]
    suffix_inputs = [name.replace("in_", "in_b_", 1) for name in kv_input_names]
    _export_component(
        staging_dir / MODEL_FILE_NAMES["kv_concat"],
        KV_CONCAT(layout),
        tuple(kv_tensors + [tensor.clone() for tensor in kv_tensors]),
        prefix_inputs + suffix_inputs,
        kv_output_names,
        {
            **cache_state_dynamic_axes(layout, prefix_inputs, "prefix_len", groups=kv_groups),
            **cache_state_dynamic_axes(layout, suffix_inputs, "suffix_len", groups=kv_groups),
            **cache_state_dynamic_axes(layout, kv_output_names, "concat_len", groups=kv_groups),
        },
        metadata,
    )

    rope_groups = ["key"]
    if layout.is_quantized:
        rope_groups.append("key_scale")
        if not layout.is_symmetric:
            rope_groups.append("key_bias")
    rope_inputs = []
    rope_input_names = []
    rope_output_names = []
    rope_axes = {}
    for group_name in rope_groups:
        group = next(group for group in layout.groups if group.name == group_name)
        for layer_index in range(group.count):
            input_name = f"in_{group_name}_{layer_index}"
            output_name = f"out_{group_name}_{layer_index}"
            rope_inputs.append(layout.zeros(group_name, 1, 4))
            rope_input_names.append(input_name)
            rope_output_names.append(output_name)
            rope_axes[input_name] = {0: "batch", group.sequence_axis: "history_len"}
            rope_axes[output_name] = {0: "batch", group.sequence_axis: "history_len"}
    rope_shift = torch.tensor([1], dtype=torch.int64)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["rope_shift"],
        KV_ROPE_SHIFT(layout, main.num_key_value_groups, rope_inv_freq, MAX_SEQ_LEN),
        tuple(rope_inputs + [rope_shift]),
        rope_input_names + ["shift"],
        rope_output_names,
        rope_axes,
        metadata,
    )


@torch.inference_mode()
def export_lfm() -> Path:
    if not MODEL_FOLDER.is_dir():
        raise FileNotFoundError(f"LFM2-350M checkpoint is missing: {MODEL_FOLDER}")
    if REORDER_KEY not in {"absmean", "L4", "rms", "std"}:
        raise ValueError(f"Unsupported REORDER_KEY: {REORDER_KEY!r}")

    staging_dir = _prepare_export_staging()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_FOLDER,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    backbone = model.model
    if not all(hasattr(backbone, attribute) for attribute in ("embed_tokens", "layers", "rotary_emb", "embedding_norm")):
        raise RuntimeError("Checkpoint does not expose the expected LFM2 text backbone.")
    if model.lm_head.weight.data_ptr() != backbone.embed_tokens.weight.data_ptr():
        raise RuntimeError("LFM2-350M-Extract must retain tied embed_tokens and lm_head weights.")

    kv_settings = KVCacheSettings(
        quant_dtype=KV_QUANT_DTYPE,
        quant_group_size=KV_QUANT_GROUP_SIZE,
        compute_in_f32=COMPUTE_IN_F32,
        use_hadamard=USE_HADAMARD,
        hadamard_random_seed=HADAMARD_RANDOM_SEED,
        use_clip=USE_CLIP,
        clip_sigma=CLIP_SIGMA,
        use_shuffle=USE_SHUFFLE,
        use_sym=USE_SYM,
        use_float16_scale_bias=USE_FLOAT16_SCALE_BIAS,
        use_qdq_friendly_asym=USE_QDQ_FRIENDLY_ASYM,
    )
    rope_inv_freq = backbone.rotary_emb.inv_freq.detach().float().clone()

    main = LFM_MAIN(
        backbone,
        model.config,
        model.lm_head,
        kv_settings=kv_settings,
        reorder_downproj=REORDER_DOWNPROJ_FOR_QUANT,
        reorder_key=REORDER_KEY,
    ).eval()
    for note in main.kv_notes:
        print(f"[KV] {note}")
    metadata = _metadata_values(model, tokenizer, main)
    print(
        "LFM2-350M-Extract: "
        f"layers={metadata['num_layers']} attention={metadata['num_attention_layers']} "
        f"conv={metadata['num_convolution_layers']} tied_embed_lm_head={metadata['embed_lm_head_tied']}"
    )
    print(
        f"Reordered {main.reorder_summary.language_pairs} gated MLP pairs; "
        f"max equivalence error={main.reorder_summary.maximum_error:.3e}"
    )

    marker = torch.zeros((1,), dtype=torch.int32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["metadata"],
        METADATA_CARRIER(),
        (marker,),
        ["metadata_marker"],
        ["metadata_marker_out"],
        None,
        metadata,
    )

    input_ids = torch.ones((1, 8), dtype=torch.int32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["embed"],
        LFM_EMBED(backbone),
        (input_ids,),
        ["input_ids"],
        ["text_hidden_states"],
        {"input_ids": {0: "batch", 1: "ids_len"}, "text_hidden_states": {0: "batch", 1: "ids_len"}},
        metadata,
    )

    ids_len = torch.tensor([8], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["rotary_prefill"],
        ROTARY_PREFILL(backbone, MAX_SEQ_LEN),
        (ids_len, history_len),
        ["ids_len", "history_len"],
        ["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        {
            "rotary_cos": {1: "ids_len"},
            "rotary_sin": {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        },
        metadata,
    )
    _export_component(
        staging_dir / MODEL_FILE_NAMES["rotary_decode"],
        ROTARY_DECODE(backbone, MAX_SEQ_LEN),
        (ids_len,),
        ["kv_seq_len"],
        ["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        None,
        metadata,
    )

    cache_tensors, cache_input_names, cache_output_names, cache_axes = build_cache_io(main, 1)
    hidden_states = torch.ones((1, 8, main.hidden_size), dtype=torch.float32)
    rotary_cos = torch.zeros((1, 8, 1, 1, main.head_dim), dtype=torch.float32)
    rotary_sin = torch.zeros_like(rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, 8, 8), dtype=torch.float32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["main"],
        main,
        tuple(cache_tensors + [hidden_states, rotary_cos, rotary_sin, attention_mask]),
        cache_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
        cache_output_names + ["logits"],
        {
            **cache_axes,
            "hidden_states": {0: "batch", 1: "ids_len"},
            "rotary_cos": {1: "ids_len"},
            "rotary_sin": {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            "logits": {0: "batch"},
        },
        metadata,
    )

    _export_kv_helpers(
        staging_dir,
        main,
        rope_inv_freq,
        metadata,
    )

    logits = torch.ones((1, int(model.config.vocab_size)), dtype=torch.float32)
    previous_ids = torch.zeros((1, 1), dtype=torch.int32)
    repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["greedy"],
        GREEDY_SEARCH(),
        (logits,),
        ["logits"],
        ["token"],
        {"logits": {0: "batch"}, "token": {0: "batch"}},
        metadata,
    )
    _export_component(
        staging_dir / MODEL_FILE_NAMES["penalty_greedy"],
        PENALTY_GREEDY_SEARCH(),
        (logits, repetition_penalty, previous_ids),
        ["logits", "repetition_penalty", "previous_ids"],
        ["token", "save_id_out"],
        {
            "logits": {0: "batch"},
            "repetition_penalty": {0: "batch"},
            "previous_ids": {0: "batch", 1: "history_len"},
            "token": {0: "batch"},
            "save_id_out": {0: "batch", 1: "kv_seq_len"},
        },
        metadata,
    )
    _export_component(
        staging_dir / MODEL_FILE_NAMES["sampling"],
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            torch.ones((1,), dtype=torch.float32),
            torch.tensor(min(50, int(model.config.vocab_size)), dtype=torch.int64),
            torch.ones((1,), dtype=torch.float32),
            repetition_penalty,
            previous_ids,
        ),
        ["logits", "temperature", "top_k", "top_p", "repetition_penalty", "previous_ids"],
        ["token", "save_id_out"],
        {
            "logits": {0: "batch"},
            "temperature": {0: "batch"},
            "top_p": {0: "batch"},
            "repetition_penalty": {0: "batch"},
            "previous_ids": {0: "batch", 1: "history_len"},
            "token": {0: "batch"},
            "save_id_out": {0: "batch", 1: "kv_seq_len"},
        },
        metadata,
    )

    del input_ids, hidden_states, rotary_cos, rotary_sin, attention_mask, cache_tensors
    gc.collect()
    bundle = Shared_Merged.build_shared_merged_bundle(
        staging_dir,
        model_file_names=MODEL_FILE_NAMES,
        delete_constituents=True,
    )
    for path in (*bundle["graphs"].values(), bundle["shared_model"]):
        _stamp_metadata(path, metadata)
        Shared_Merged.validate_onnx_path(path)
    _cleanup_unreferenced_data(staging_dir)
    tokenizer_assets = copy_tokenizer_assets(MODEL_FOLDER, staging_dir)
    _promote_export(staging_dir)
    print(
        f"LFM2-350M text ONNX bundle completed: {EXPORT_DIR} "
        f"({len(tokenizer_assets)} tokenizer assets)."
    )
    return EXPORT_DIR


def export_bundle() -> Path:
    """Export the LFM OCR ONNX bundle."""
    return export_lfm()


def main() -> None:
    if not DO_EXPORT:
        print("DO_EXPORT is False; no ONNX files were written.")
        return
    export_dir = export_bundle()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_LFM_ONNX.py"),
            "--model-folder",
            str(export_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
