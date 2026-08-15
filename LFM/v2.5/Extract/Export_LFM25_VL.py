"""Export LFM2.5-VL-450M-Extract as a staged image ONNX runtime bundle.

Evidence-backed deviation from the native dynamic-resolution processor: this
legacy TorchScript ONNX exporter cannot lower SigLIP2's
``aten::_upsample_bilinear2d_aa`` positional interpolation, even at opset 20.
The export therefore fixes the native single-image contract to 512x512. The
exact 32x32 SigLIP2 positional table is precomputed from checkpoint weights;
raw image patching and the 256 <image> placeholder contract match the processor.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import Shared_Merged
from LFM_Text_Components import (
    GREEDY_SEARCH,
    LFM_EMBED,
    LFM_MAIN,
    METADATA_CARRIER,
    PENALTY_GREEDY_SEARCH,
    ROTARY_DECODE,
    ROTARY_PREFILL,
    TOPK_TOPP_SAMPLING,
    KVCacheSettings,
    build_cache_io,
)
from LFM_VL_Components import (
    LFM_CONCAT_IMAGE,
    LFM_IMAGE_PREPROCESS,
    LFM_VISION,
    STATIC_IMAGE_HEIGHT,
    STATIC_IMAGE_WIDTH,
    STATIC_PATCH_COUNT,
    reorder_vision_mlp_pairs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

CHECKPOINT_DIR = Path.home() / "Downloads" / "LFM2.5-VL-450M-Extract"
# Backward-compatible configuration alias.
MODEL_FOLDER = CHECKPOINT_DIR
EXPORT_DIR = SCRIPT_DIR / "LFM25_VL_ONNX"
EXPORT_STAGING_DIR = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")

# Export controls
DO_EXPORT   = True                        # Whether to export the ONNX models.
MAX_SEQ_LEN = 4096                        # Fixed maximum sequence length after export.

# KV cache storage and attention precision
# F16/F32 are unquantized; Q8/Q4 variants store KV as integer caches.
KV_QUANT_DTYPE      = "Q8"                # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 64                  # Quantization group width; auto-clamped to a divisor.
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
REORDER_VISION_MLP_FOR_QUANT = True       # Reorder vision MLP channels before quantization.
REORDER_KEY                  = "absmean"  # Channel statistic used to build the permutation.

# ONNX graph format
OPSET = 20                                # ONNX opset version.

MODEL_FILE_NAMES = Shared_Merged.default_model_file_names()
RUNTIME_MODEL_FILE_ROLES = (
    "metadata",
    "image_preprocess",
    "vision",
    "shared_initializers",
    "shared_initializers_data",
    "image_prefill_greedy",
    "image_prefill_penalty_greedy",
    "image_prefill_sampling",
    "image_decode_greedy",
    "image_decode_penalty_greedy",
    "image_decode_sampling",
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
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(path: Path, module: torch.nn.Module, args: tuple, input_names: list[str], output_names: list[str], dynamic_axes: dict | None, metadata: dict[str, str]) -> None:
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
    previous = EXPORT_DIR.with_name(EXPORT_DIR.name + ".previous")
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    if EXPORT_DIR.exists():
        EXPORT_DIR.rename(previous)
    staging_dir.rename(EXPORT_DIR)


def _cleanup_unreferenced_data(export_dir: Path) -> None:
    referenced = set()
    for model_path in export_dir.glob("*.onnx"):
        model = onnx.load(str(model_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                external = {item.key: item.value for item in initializer.external_data}
                if external.get("location"):
                    referenced.add(Path(external["location"]).name)
    for candidate in export_dir.iterdir():
        if candidate.is_file() and candidate.suffix != ".onnx" and candidate.name not in referenced:
            candidate.unlink()


def _native_prompt_ids(processor, image_token_id: int) -> list[int]:
    """Build the checkpoint-native static image prompt and validate its placeholders."""
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": ""}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    expanded = processor.expand_text_with_placeholders(
        [prompt],
        [[None]],
        image_rows=[1],
        image_cols=[1],
        image_sizes=[[STATIC_IMAGE_HEIGHT, STATIC_IMAGE_WIDTH]],
        use_image_special_tokens=True,
    )[0]
    token_ids = processor.tokenizer(expanded, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    positions = [index for index, token_id in enumerate(token_ids) if int(token_id) == int(image_token_id)]
    if len(positions) != 256:
        raise RuntimeError(f"Native LFM2.5-VL prompt has {len(positions)} image tokens, expected 256.")
    return [int(token_id) for token_id in token_ids]


def _metadata_values(model, processor, main: LFM_MAIN, vision_reorder) -> dict[str, str]:
    config = model.config
    text_config = config.text_config
    image_token_id = int(processor.image_token_id)
    static_ids = _native_prompt_ids(processor, image_token_id)
    image_positions = [index for index, token_id in enumerate(static_ids) if token_id == image_token_id]
    eos_ids = _id_list(getattr(config, "eos_token_id", getattr(processor.tokenizer, "eos_token_id", None)))
    metadata = {
        "model_family": "lfm25_vl_extract",
        "input_modality": "image_text",
        "max_seq_len": str(MAX_SEQ_LEN),
        "stop_token_ids": ",".join(str(item) for item in eos_ids),
        "eos_token_ids": ",".join(str(item) for item in eos_ids),
        "image_token_id": str(image_token_id),
        "image_token_length": "256",
        "input_image_size": f"{STATIC_IMAGE_HEIGHT},{STATIC_IMAGE_WIDTH}",
        "input_image_dim": "4",
        "vision_batch_size": "1",
        "vision_static_resolution": "1",
        "vision_export_deviation": "static_512_due_to_upsample_bilinear2d_aa_onnx_export_limit",
        "image_start": str(image_positions[0]),
        "image_end": str(image_positions[-1] + 1),
        "vocab_size": str(int(text_config.vocab_size)),
        "hidden_size": str(main.hidden_size),
        "num_layers": str(len(main.backbone.layers)),
        "num_attention_layers": str(main.num_attn_layers),
        "num_convolution_layers": str(main.num_conv_layers),
        "num_attention_heads": str(main.num_heads),
        "num_key_value_heads": str(main.num_key_value_heads),
        "head_dim": str(main.head_dim),
        "kv_num_tensors": str(main.cache_state_count),
        "kv_cache_tensor_order": ",".join(main.cache_state_groups),
        "kv_cache_key_sequence_axis": "4",
        "kv_cache_value_sequence_axis": "3",
        "conv_state_sequence_axis": "2",
        "kv_cache_storage_dtype": main.dtype_name(main.kv_cache_dtype),
        "kv_quant_dtype": main.kv_settings.quant_dtype,
        "kv_quant_group_size": str(main.kv_settings.quant_group_size),
        "kv_blocks_per_layer": str(main.kv_blocks_per_attention),
        "kv_symmetric": str(int(main.kv_sym)),
        "kv_grouped_6d": str(int(main.kv_grouped_layout)),
        "kv_cache_elem_type": main.dtype_name(main.kv_cache_dtype),
        "kv_scale_bias_elem_type": (
            main.dtype_name(main.kv_scale_bias_dtype)
            if main.kv_quantized
            else ""
        ),
        "kv_quant_hadamard": str(int(main.kv_settings.use_hadamard)) if main.kv_quantized else "",
        "kv_quant_shuffle": str(int(main.kv_settings.use_shuffle)) if main.kv_quantized else "",
        "kv_quant_clip": str(int(main.kv_settings.use_clip)) if main.kv_quantized else "",
        "compute_in_f32": str(int(main.compute_in_f32)),
        "fused_simplified_layer_norm_count": str(2 * len(main.backbone.layers) + main.num_attn_layers + 1),
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reorder_key": REORDER_KEY,
        "reordered_language_pair_count": str(main.reorder_summary.language_pairs),
        "reordered_vision_pair_count": str(vision_reorder.pairs),
        "reorder_max_equivalence_error": repr(max(main.reorder_summary.maximum_error, vision_reorder.maximum_error)),
        "embed_lm_head_tied": "1",
    }
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


@torch.inference_mode()
def export_lfm25_vl() -> Path:
    if not MODEL_FOLDER.is_dir():
        raise FileNotFoundError(f"LFM2.5-VL checkpoint is missing: {MODEL_FOLDER}")
    if REORDER_KEY not in {"absmean", "L4", "rms", "std"}:
        raise ValueError(f"Unsupported REORDER_KEY: {REORDER_KEY!r}")

    staging_dir = _prepare_export_staging()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_FOLDER,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_FOLDER)
    backbone = model.model.language_model
    if model.lm_head.weight.data_ptr() != backbone.embed_tokens.weight.data_ptr():
        raise RuntimeError("LFM2.5-VL must retain tied embed_tokens and lm_head weights.")
    vision_reorder = reorder_vision_mlp_pairs(model, REORDER_KEY, REORDER_VISION_MLP_FOR_QUANT)
    kv_cache_settings = KVCacheSettings(
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
    main = LFM_MAIN(
        backbone,
        model.config.text_config,
        model.lm_head,
        kv_cache_settings=kv_cache_settings,
        reorder_downproj=REORDER_DOWNPROJ_FOR_QUANT,
        reorder_key=REORDER_KEY,
    ).eval()
    for note in main.kv_setting_notes:
        print(note)
    metadata = _metadata_values(model, processor, main, vision_reorder)
    print(
        "LFM2.5-VL-450M-Extract: "
        f"image_tokens={metadata['image_token_length']} state_tensors={metadata['kv_num_tensors']} "
        f"language_pairs={main.reorder_summary.language_pairs} vision_pairs={vision_reorder.pairs}"
    )

    _export_component(
        staging_dir / MODEL_FILE_NAMES["metadata"],
        METADATA_CARRIER(),
        (torch.zeros((1,), dtype=torch.int32),),
        ["metadata_marker"],
        ["metadata_marker_out"],
        None,
        metadata,
    )
    input_ids = torch.tensor([_native_prompt_ids(processor, int(processor.image_token_id))], dtype=torch.int32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["embed"],
        LFM_EMBED(backbone),
        (input_ids,),
        ["input_ids"],
        ["text_hidden_states"],
        {"input_ids": {0: "batch", 1: "ids_len"}, "text_hidden_states": {0: "batch", 1: "ids_len"}},
        metadata,
    )

    raw_image = torch.zeros((1, 3, STATIC_IMAGE_HEIGHT, STATIC_IMAGE_WIDTH), dtype=torch.uint8)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["image_preprocess"],
        LFM_IMAGE_PREPROCESS(),
        (raw_image,),
        ["pixel_values"],
        ["patches"],
        None,
        metadata,
    )
    patches = torch.zeros((1, STATIC_PATCH_COUNT, 768), dtype=torch.float32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["vision"],
        LFM_VISION(model),
        (patches,),
        ["patches"],
        ["vision_hidden_states"],
        None,
        metadata,
    )
    vision_hidden_states = torch.zeros((1, 256, main.hidden_size), dtype=torch.float32)
    text_hidden_states = torch.zeros((1, input_ids.shape[1], main.hidden_size), dtype=torch.float32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["concat_image"],
        LFM_CONCAT_IMAGE(int(processor.image_token_id), 256),
        (input_ids, text_hidden_states, vision_hidden_states),
        ["input_ids", "text_hidden_states", "vision_hidden_states"],
        ["concat_hidden_states"],
        {
            "input_ids": {0: "batch", 1: "ids_len"},
            "text_hidden_states": {0: "batch", 1: "ids_len"},
            "concat_hidden_states": {0: "batch", 1: "ids_len"},
        },
        metadata,
    )

    ids_len = torch.tensor([input_ids.shape[1]], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["rotary_image_prefill"],
        ROTARY_PREFILL(backbone, MAX_SEQ_LEN),
        (ids_len, history_len),
        ["ids_len", "history_len"],
        ["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        {"rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"}, "attention_mask": {3: "ids_len", 4: "kv_seq_len"}},
        metadata,
    )
    _export_component(
        staging_dir / MODEL_FILE_NAMES["rotary_image_decode"],
        ROTARY_DECODE(backbone, MAX_SEQ_LEN),
        (ids_len,),
        ["kv_seq_len"],
        ["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        None,
        metadata,
    )

    cache_tensors, cache_input_names, cache_output_names, cache_axes = build_cache_io(main, 1)
    rotary_cos = torch.zeros((1, input_ids.shape[1], 1, 1, main.head_dim), dtype=torch.float32)
    rotary_sin = torch.zeros_like(rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float32)
    _export_component(
        staging_dir / MODEL_FILE_NAMES["main"],
        main,
        tuple(cache_tensors + [text_hidden_states, rotary_cos, rotary_sin, attention_mask]),
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

    logits = torch.ones((1, int(model.config.text_config.vocab_size)), dtype=torch.float32)
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
        {"logits": {0: "batch"}, "repetition_penalty": {0: "batch"}, "previous_ids": {0: "batch", 1: "history_len"}, "token": {0: "batch"}, "save_id_out": {0: "batch", 1: "kv_seq_len"}},
        metadata,
    )
    _export_component(
        staging_dir / MODEL_FILE_NAMES["sampling"],
        TOPK_TOPP_SAMPLING(),
        (logits, torch.ones((1,), dtype=torch.float32), torch.tensor(min(50, logits.shape[1]), dtype=torch.int64), torch.ones((1,), dtype=torch.float32), repetition_penalty, previous_ids),
        ["logits", "temperature", "top_k", "top_p", "repetition_penalty", "previous_ids"],
        ["token", "save_id_out"],
        {"logits": {0: "batch"}, "temperature": {0: "batch"}, "top_p": {0: "batch"}, "repetition_penalty": {0: "batch"}, "previous_ids": {0: "batch", 1: "history_len"}, "token": {0: "batch"}, "save_id_out": {0: "batch", 1: "kv_seq_len"}},
        metadata,
    )

    del raw_image, patches, vision_hidden_states, text_hidden_states, rotary_cos, rotary_sin, attention_mask, cache_tensors
    gc.collect()
    bundle = Shared_Merged.build_shared_merged_bundle(staging_dir, model_file_names=MODEL_FILE_NAMES, delete_constituents=True)
    for path in (*bundle["graphs"].values(), bundle["shared_model"]):
        _stamp_metadata(path, metadata)
        Shared_Merged.validate_onnx_path(path)
    _cleanup_unreferenced_data(staging_dir)
    tokenizer_assets = copy_tokenizer_assets(MODEL_FOLDER, staging_dir)
    _promote_export(staging_dir)
    print(
        f"LFM2.5-VL ONNX bundle completed: {EXPORT_DIR} "
        f"({len(tokenizer_assets)} tokenizer assets)."
    )
    return EXPORT_DIR


def export_bundle() -> Path:
    """Export the LFM VL OCR ONNX bundle."""
    return export_lfm25_vl()


def main() -> None:
    if not DO_EXPORT:
        print("DO_EXPORT is False; no ONNX files were written.")
        return
    export_dir = export_bundle()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_LFM25_VL_ONNX.py"),
            "--model-folder",
            str(export_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
