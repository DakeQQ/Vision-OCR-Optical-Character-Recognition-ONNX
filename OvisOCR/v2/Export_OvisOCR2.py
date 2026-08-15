import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import onnx
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR                         = Path(__file__).resolve().parent
CHECKPOINT_DIR                   = Path.home() / "Downloads" / "OvisOCR2"
# Backward-compatible configuration alias.
download_path                    = str(CHECKPOINT_DIR)
EXPORT_DIR                       = BASE_DIR / "OvisOCR2_ONNX"
EXPORT_STAGING_DIR               = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")
TOKENIZER_ASSET_NAMES            = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
REQUIRED_TOKENIZER_ASSET_NAMES   = ("tokenizer.json", "tokenizer_config.json", "vocab.json")

# Export controls
DO_EXPORT            = True                    # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW = False                   # Prevent float16 overflow for Q4F16, Q8F16, or F16 quantization.
STOP_TOKEN: list[int] = []                     # Derived from the checkpoint config below.
MAX_SEQ_LEN          = 4096                    # Fixed maximum output sequence length after export.

# Quantization-oriented model reordering
# Exact paired permutations keep producer/consumer weights synchronized after fusion.
REORDER_DOWNPROJ_FOR_QUANT   = True            # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True            # Reorder vision MLP channels before quantization.
REORDER_KEY                  = "absmean"       # Channel statistic: absmean | L4 | rms | std.

# Image input and vision tracing
# OvisOCR2 uses a static 20x20 post-merge grid (640 px) for its ONNX target.
HEIGHT_FACTOR       = 20                       # Vertical factor for the exported image grid.
WIDTH_FACTOR        = 20                       # Horizontal factor for the exported image grid.
# Image resize uses patch_size * spatial_merge_size.
IMAGE_RESIZE        = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]
INPUT_IMAGE_SIZE    = IMAGE_RESIZE             # Export-time OCR canvas before patch packing.
VISION_BATCH_SIZE   = 1                        # Number of pages/images processed per batch.
DYNAMIC_IMAGE_SHAPE = False                    # Keep the exported image grid static.
INPUT_IMAGE_DIM     = 4                        # 4=[B, C, H, W]; 5=[B, 1, C, H, W].

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                     # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 128                      # Quantization group width; must divide head_dim evenly.
COMPUTE_IN_F32      = False                    # F16 KV only: False keeps attention in F16, True upcasts cache reads.

# KV quantization transforms and parameters
USE_HADAMARD           = False                 # Apply a randomized Hadamard transform before grouped quantization.
HADAMARD_RANDOM_SEED   = 9527                  # Deterministic sign pattern for the Hadamard transform.
USE_CLIP               = False                 # Clip quantization blocks to CLIP_SIGMA standard deviations.
CLIP_SIGMA             = 3.0                   # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False                 # Interleave channels across quantization groups.
USE_SYM                = True                  # True=symmetric absmax; False=asymmetric min-max with bias.
USE_FLOAT16_SCALE_BIAS = True                  # Store quantization scales and biases as float16.

# ONNX graph format
OPSET = 20                                     # ONNX opset version.

# Every runtime-visible filename is part of the metadata contract. OvisOCR2 is
# image-only; the merged graphs own all language and decoding constituents.
MODEL_FILE_NAMES = {
    "metadata": "LLM_Metadata.onnx",
    "embed": "LLM_Embed.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "vision": "LLM_Vision.onnx",
    "concat_image": "LLM_Concat_Image.onnx",
    "rotary_image_prefill": "LLM_Rotary_Image_Prefill.onnx",
    "rotary_image_decode": "LLM_Rotary_Image_Decode.onnx",
    "main": "LLM_Main.onnx",
    "kv_slice": "LLM_KV_Slice.onnx",
    "kv_split2": "LLM_KV_Split2.onnx",
    "kv_concat": "LLM_KV_Concat.onnx",
    "rope_shift": "LLM_RopeShift.onnx",
    "greedy": "LLM_Greedy.onnx",
    "penalty_greedy": "LLM_PenaltyGreedy.onnx",
    "sampling": "LLM_TopKTopPSampling.onnx",
    "image_prefill_greedy": "LLM_ImagePrefillGreedy.onnx",
    "image_prefill_penalty_greedy": "LLM_ImagePrefillPenaltyGreedy.onnx",
    "image_prefill_sampling": "LLM_ImagePrefillSampling.onnx",
    "image_decode_greedy": "LLM_ImageDecodeGreedy.onnx",
    "image_decode_penalty_greedy": "LLM_ImageDecodePenaltyGreedy.onnx",
    "image_decode_sampling": "LLM_ImageDecodeSampling.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"
RUNTIME_MODEL_FILE_ROLES = (
    "image_preprocess",
    "vision",
    "kv_slice",
    "kv_split2",
    "kv_concat",
    "rope_shift",
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
    f"model_file_name_{key}": MODEL_FILE_NAMES[key]
    for key in RUNTIME_MODEL_FILE_ROLES
}


# ══════════════════════════════════════════════════════════════════════════════
# Static config facts from config.json
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = json.loads((Path(download_path) / "config.json").read_text(encoding="utf-8"))
TEXT_CONFIG = MODEL_CONFIG["text_config"]
VISION_CONFIG = MODEL_CONFIG["vision_config"]
_configured_eos = TEXT_CONFIG.get("eos_token_id", MODEL_CONFIG.get("eos_token_id"))
if _configured_eos is None:
    raise ValueError("OvisOCR2 config.json must define an eos_token_id.")
STOP_TOKEN = [int(_configured_eos)]

HIDDEN_SIZE                    = int(TEXT_CONFIG["hidden_size"])
INTERMEDIATE_SIZE              = int(TEXT_CONFIG["intermediate_size"])
NUM_HIDDEN_LAYERS              = int(TEXT_CONFIG["num_hidden_layers"])
NUM_HEADS                      = int(TEXT_CONFIG["num_attention_heads"])
NUM_KEY_VALUE_HEADS            = int(TEXT_CONFIG["num_key_value_heads"])
HEAD_DIM                       = int(TEXT_CONFIG["head_dim"])
VOCAB_SIZE                     = int(TEXT_CONFIG["vocab_size"])
RMS_NORM_EPS                   = float(TEXT_CONFIG["rms_norm_eps"])
ATTN_OUTPUT_GATE               = bool(TEXT_CONFIG.get("attn_output_gate", False))

ROPE_PARAMETERS                = dict(TEXT_CONFIG["rope_parameters"])
PARTIAL_ROTARY_FACTOR          = float(ROPE_PARAMETERS.get("partial_rotary_factor", 1.0))
ROTARY_DIM                     = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
ROPE_THETA                     = float(ROPE_PARAMETERS["rope_theta"])
MROPE_SECTION                  = list(ROPE_PARAMETERS.get("mrope_section", [11, 11, 10]))

LAYER_TYPES                    = list(TEXT_CONFIG["layer_types"])
FULL_ATTENTION_LAYER_INDICES   = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "full_attention"]
LINEAR_ATTENTION_LAYER_INDICES = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "linear_attention"]
NUM_FULL_ATTENTION_LAYERS      = len(FULL_ATTENTION_LAYER_INDICES)
NUM_LINEAR_ATTENTION_LAYERS    = len(LINEAR_ATTENTION_LAYER_INDICES)

LINEAR_CONV_KERNEL_DIM         = int(TEXT_CONFIG["linear_conv_kernel_dim"])
LINEAR_NUM_KEY_HEADS           = int(TEXT_CONFIG["linear_num_key_heads"])
LINEAR_NUM_VALUE_HEADS         = int(TEXT_CONFIG["linear_num_value_heads"])
LINEAR_KEY_HEAD_DIM            = int(TEXT_CONFIG["linear_key_head_dim"])
LINEAR_VALUE_HEAD_DIM          = int(TEXT_CONFIG["linear_value_head_dim"])
LINEAR_KEY_DIM                 = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM
LINEAR_VALUE_DIM               = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
LINEAR_CONV_DIM                = LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM
LINEAR_CONV_STATE_LEN          = LINEAR_CONV_KERNEL_DIM - 1

VISION_HIDDEN_SIZE             = int(VISION_CONFIG["hidden_size"])
VISION_DEPTH                   = int(VISION_CONFIG["depth"])
VISION_NUM_HEADS               = int(VISION_CONFIG["num_heads"])
VISION_HEAD_DIM                = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_PATCH_SIZE              = int(VISION_CONFIG["patch_size"])
VISION_TEMPORAL_PATCH_SIZE     = int(VISION_CONFIG["temporal_patch_size"])
VISION_SPATIAL_MERGE_SIZE      = int(VISION_CONFIG["spatial_merge_size"])
VISION_OUT_HIDDEN_SIZE         = int(VISION_CONFIG["out_hidden_size"])

# Multimodal token IDs
IMAGE_TOKEN_ID                 = int(MODEL_CONFIG.get("image_token_id", 248056))
VISION_START_TOKEN_ID          = int(MODEL_CONFIG.get("vision_start_token_id", 248053))
VISION_END_TOKEN_ID            = int(MODEL_CONFIG.get("vision_end_token_id", 248054))

# Derived vision constants
IMAGE_SEQLEN_PER_IMAGE         = HEIGHT_FACTOR * WIDTH_FACTOR
VISION_EMBED_SIZE              = IMAGE_SEQLEN_PER_IMAGE * VISION_BATCH_SIZE

SCALE_DTYPE_TORCH = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

# ══════════════════════════════════════════════════════════════════════════════
# KV Quant Validation
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "Q8",
    "Q8_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
    "F16",
    "F32",
)


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {
        "Q8",
        "Q8_CUDA",
        "ROTARY_Q8",
        "ROTARY_Q8_CUDA",
        "ROTARY_Q4",
        "ROTARY_Q4_CUDA",
    }
    rotary_kv = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes = []

    if KV_QUANT_DTYPE in rotary_kv and head_dim % 2 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")

    if KV_QUANT_DTYPE in quantized_kv:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError(f"KV_QUANT_GROUP_SIZE must be positive, got {KV_QUANT_GROUP_SIZE}.")
        if KV_QUANT_GROUP_SIZE > head_dim:
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim.")
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}.")
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization.")

        if (
            KV_QUANT_DTYPE in q8_kv
            and KV_QUANT_GROUP_SIZE == head_dim
            and (USE_HADAMARD or USE_SHUFFLE)
        ):
            notes.append("[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block.")
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Prompt helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_ovis_prompt_layout(processor, num_images: int):
    """Build and validate native image-token spans used by concat and mRoPE."""
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(processor, "image_token", None)
    image_token_id = getattr(processor, "image_token_id", None)
    if tokenizer is None or not image_token or image_token_id is None:
        raise ValueError("OvisOCR2 processor must expose tokenizer, image_token, and image_token_id.")

    conversation = [{
        "role": "user",
        "content": ([{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": ""}]),
    }]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if prompt.count(image_token) != num_images:
        raise ValueError(
            f"OvisOCR2 chat template produced {prompt.count(image_token)} image placeholders, expected {num_images}."
        )

    expanded = prompt.replace(image_token, image_token * IMAGE_SEQLEN_PER_IMAGE)
    token_ids = tokenizer(expanded, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    positions = [
        index for index, token_id in enumerate(token_ids)
        if token_id == int(image_token_id)
    ]
    expected_count = num_images * IMAGE_SEQLEN_PER_IMAGE
    if len(positions) != expected_count:
        raise ValueError(
            f"OvisOCR2 chat template produced {len(positions)} expanded image tokens, expected {expected_count}."
        )

    spans = []
    cursor = 0
    while cursor < len(positions):
        start = positions[cursor]
        end = start + 1
        cursor += 1
        while cursor < len(positions) and positions[cursor] == end:
            end += 1
            cursor += 1
        if end - start != IMAGE_SEQLEN_PER_IMAGE:
            raise ValueError(
                "Each OvisOCR2 image placeholder must expand into exactly "
                f"{IMAGE_SEQLEN_PER_IMAGE} contiguous image tokens."
            )
        spans.append((start, end))
    if len(spans) != num_images:
        raise ValueError("OvisOCR2 image-token span count does not match VISION_BATCH_SIZE.")

    mm_token_type_ids = [
        int(token_id == int(image_token_id)) for token_id in token_ids
    ]
    return token_ids, tuple(spans), mm_token_type_ids


_is_rotary_kv = KV_QUANT_DTYPE in (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
)
_is_rotary_q4_kv = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
_is_quantized_kv = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
_kv_sym = USE_SYM and (_is_rotary_kv or _is_quantized_kv)
_q8_grouped = (
    _is_quantized_kv
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_rotary_q8_grouped = (
    KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_grouped_6d = _is_rotary_q4_kv or _q8_grouped or _rotary_q8_grouped

FULL_STATE_SPECS = [("key", 4), ("value", 3)]
if _is_quantized_kv or _is_rotary_kv:
    if _kv_sym:
        if _grouped_6d:
            FULL_STATE_SPECS.extend([("key_scale", 5), ("value_scale", 3)])
        else:
            FULL_STATE_SPECS.extend([("key_scale", 4), ("value_scale", 3)])
    else:
        if _grouped_6d:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 5),
                    ("key_bias", 5),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
        else:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 4),
                    ("key_bias", 4),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
LINEAR_STATE_SPECS = [("conv_state", None), ("recurrent_state", None)]

NUM_FULL_STATE_TENSORS = NUM_FULL_ATTENTION_LAYERS * len(FULL_STATE_SPECS)
NUM_LINEAR_STATE_TENSORS = NUM_LINEAR_ATTENTION_LAYERS * len(LINEAR_STATE_SPECS)
NUM_MAIN_STATE_TENSORS = NUM_FULL_STATE_TENSORS + NUM_LINEAR_STATE_TENSORS


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Token-only greedy contract used by merged OvisOCR2 decode graphs."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class PENALTY_GREEDY_SEARCH(torch.nn.Module):
    """Sign-aware repetition penalty followed by greedy token selection."""

    def forward(self, logits, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted_logits = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        adjusted_logits = torch.scatter(logits, 1, previous_ids.long(), adjusted_logits)
        token_id = torch.argmax(adjusted_logits, dim=-1, keepdim=True).int()
        return token_id, torch.cat([previous_ids, token_id], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """TopTok sampling with target-native sign-aware repetition handling."""

    @staticmethod
    def _sample(scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(
            dim=-1, keepdim=True
        )
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax(
            (cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True
        )
        return torch.gather(sorted_indices, 1, winner).int()

    def forward(
        self,
        logits,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        previous_ids,
    ):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted_logits = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids.long(), adjusted_logits)
        sampled_id = self._sample(scores, temperature, top_k, top_p)
        return sampled_id, torch.cat([previous_ids, sampled_id], dim=-1)


class METADATA_CARRIER(torch.nn.Module):
    """Identity graph used to load bundle metadata before large sessions."""

    def forward(self, marker):
        return marker


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Management (full-attention state only)
# ══════════════════════════════════════════════════════════════════════════════
class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    """Compute [start, window, tail] Split sizes for a cache sequence axis."""

    @staticmethod
    def forward(ctx, ref, start, end, dim):
        start_value, end_value = int(start), int(end)
        return torch.tensor(
            [start_value, end_value - start_value, ref.shape[dim] - end_value],
            dtype=torch.int64,
        )

    @staticmethod
    def symbolic(g, ref, start, end, dim):
        shape = g.op("Shape", ref)
        dim_size = g.op(
            "Gather",
            shape,
            g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        window = g.op("Sub", end, start)
        tail = g.op("Sub", dim_size, end)
        return g.op("Concat", start, window, tail, axis_i=0)


class SLICE_KEEP_MIDDLE(torch.autograd.Function):
    """Keep the middle member of a dynamic 3-way Split."""

    @staticmethod
    def forward(ctx, values, sizes, dim):
        start = int(sizes[0])
        end = start + int(sizes[1])
        index = [slice(None)] * values.dim()
        index[dim] = slice(start, end)
        return values[tuple(index)].clone()

    @staticmethod
    def symbolic(g, values, sizes, dim):
        return g.op("Split", values, sizes, axis_i=dim, outputs=3)[1]


def window_split_sizes(ref, start, end, dim):
    if dim < 0:
        dim += ref.dim()
    return WINDOW_SPLIT_SIZES.apply(ref, start, end, dim)


def slice_keep_middle(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SLICE_KEEP_MIDDLE.apply(values, sizes, dim)


class KV_SLICE(torch.nn.Module):
    """Slice every full-attention KV tensor to the requested cache window."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in (
            "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA",
        )
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = (
            KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < head_dim
        )
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

    def forward(self, *all_inputs):
        slice_start, slice_end = all_inputs[-2:]
        sizes = window_split_sizes(all_inputs[0], slice_start, slice_end, -1)
        keys, values = [], []
        key_scales, key_biases, value_scales, value_biases = [], [], [], []
        for layer_index in range(self.num_layers):
            keys.append(slice_keep_middle(all_inputs[layer_index], sizes, -1))
            values.append(
                slice_keep_middle(all_inputs[layer_index + self.num_layers], sizes, -2)
            )
            if not self.kv_quantized:
                continue
            key_scales.append(
                slice_keep_middle(all_inputs[layer_index + self.num_layers_2], sizes, -1)
            )
            if self.kv_sym:
                value_scales.append(
                    slice_keep_middle(
                        all_inputs[layer_index + self.num_layers_3],
                        sizes,
                        -3 if self.kv_grouped_6d else -2,
                    )
                )
                continue
            key_biases.append(
                slice_keep_middle(all_inputs[layer_index + self.num_layers_3], sizes, -1)
            )
            value_scales.append(
                slice_keep_middle(
                    all_inputs[layer_index + self.num_layers_4],
                    sizes,
                    -3 if self.kv_grouped_6d else -2,
                )
            )
            value_biases.append(
                slice_keep_middle(
                    all_inputs[layer_index + self.num_layers_5],
                    sizes,
                    -3 if self.kv_grouped_6d else -2,
                )
            )
        if self.kv_sym:
            return *keys, *values, *key_scales, *value_scales
        if self.kv_quantized:
            return *keys, *values, *key_scales, *key_biases, *value_scales, *value_biases
        return *keys, *values


class SPLIT_POINT_SIZES(torch.autograd.Function):
    """Compute [prefix, suffix] Split sizes for a cache sequence axis."""

    @staticmethod
    def forward(ctx, ref, split_at, dim):
        split_value = int(split_at)
        return torch.tensor(
            [split_value, ref.shape[dim] - split_value], dtype=torch.int64
        )

    @staticmethod
    def symbolic(g, ref, split_at, dim):
        shape = g.op("Shape", ref)
        dim_size = g.op(
            "Gather",
            shape,
            g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        suffix = g.op("Sub", dim_size, split_at)
        return g.op("Concat", split_at, suffix, axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):
    """Split a cache tensor at a dynamic sequence index."""

    @staticmethod
    def forward(ctx, values, sizes, dim):
        split_value = int(sizes[0])
        prefix_index = [slice(None)] * values.dim()
        suffix_index = [slice(None)] * values.dim()
        prefix_index[dim] = slice(None, split_value)
        suffix_index[dim] = slice(split_value, None)
        return values[tuple(prefix_index)].clone(), values[tuple(suffix_index)].clone()

    @staticmethod
    def symbolic(g, values, sizes, dim):
        return g.op("Split", values, sizes, axis_i=dim, outputs=2)


def split_point_sizes(ref, split_at, dim):
    if dim < 0:
        dim += ref.dim()
    return SPLIT_POINT_SIZES.apply(ref, split_at, dim)


def split_prefix_suffix(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SPLIT_PREFIX_SUFFIX.apply(values, sizes, dim)


class KV_SPLIT2(torch.nn.Module):
    """Split every full-attention cache state into prefix and suffix groups."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in (
            "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA",
        )
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = (
            KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < head_dim
        )
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

    def forward(self, *all_inputs):
        split_at = all_inputs[-1]
        sizes = split_point_sizes(all_inputs[0], split_at, -1)
        prefix_keys, prefix_values, window_keys, window_values = [], [], [], []
        prefix_key_scales, prefix_key_biases = [], []
        prefix_value_scales, prefix_value_biases = [], []
        window_key_scales, window_key_biases = [], []
        window_value_scales, window_value_biases = [], []
        for layer_index in range(self.num_layers):
            prefix, window = split_prefix_suffix(all_inputs[layer_index], sizes, -1)
            prefix_keys.append(prefix)
            window_keys.append(window)
            prefix, window = split_prefix_suffix(
                all_inputs[layer_index + self.num_layers], sizes, -2
            )
            prefix_values.append(prefix)
            window_values.append(window)
            if not self.kv_quantized:
                continue
            prefix, window = split_prefix_suffix(
                all_inputs[layer_index + self.num_layers_2], sizes, -1
            )
            prefix_key_scales.append(prefix)
            window_key_scales.append(window)
            if self.kv_sym:
                prefix, window = split_prefix_suffix(
                    all_inputs[layer_index + self.num_layers_3],
                    sizes,
                    -3 if self.kv_grouped_6d else -2,
                )
                prefix_value_scales.append(prefix)
                window_value_scales.append(window)
                continue
            prefix, window = split_prefix_suffix(
                all_inputs[layer_index + self.num_layers_3], sizes, -1
            )
            prefix_key_biases.append(prefix)
            window_key_biases.append(window)
            prefix, window = split_prefix_suffix(
                all_inputs[layer_index + self.num_layers_4],
                sizes,
                -3 if self.kv_grouped_6d else -2,
            )
            prefix_value_scales.append(prefix)
            window_value_scales.append(window)
            prefix, window = split_prefix_suffix(
                all_inputs[layer_index + self.num_layers_5],
                sizes,
                -3 if self.kv_grouped_6d else -2,
            )
            prefix_value_biases.append(prefix)
            window_value_biases.append(window)
        if self.kv_sym:
            return (
                *prefix_keys, *prefix_values, *prefix_key_scales, *prefix_value_scales,
                *window_keys, *window_values, *window_key_scales, *window_value_scales,
            )
        if self.kv_quantized:
            return (
                *prefix_keys, *prefix_values, *prefix_key_scales, *prefix_key_biases,
                *prefix_value_scales, *prefix_value_biases, *window_keys, *window_values,
                *window_key_scales, *window_key_biases, *window_value_scales, *window_value_biases,
            )
        return *prefix_keys, *prefix_values, *window_keys, *window_values


class KV_CONCAT(torch.nn.Module):
    """Concatenate matching full-attention cache tensor groups."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in (
            "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA",
        )
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = (
            KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < head_dim
        )
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.value_axis = -3 if self.kv_grouped_6d else -2

    def forward(self, *all_inputs):
        input_count = len(all_inputs) // 2
        prefix, suffix = all_inputs[:input_count], all_inputs[input_count:]
        keys, values = [], []
        key_scales, key_biases, value_scales, value_biases = [], [], [], []
        for layer_index in range(self.num_layers):
            keys.append(torch.cat([prefix[layer_index], suffix[layer_index]], dim=-1))
            values.append(
                torch.cat(
                    [
                        prefix[layer_index + self.num_layers],
                        suffix[layer_index + self.num_layers],
                    ],
                    dim=-2,
                )
            )
            if not self.kv_quantized:
                continue
            key_scales.append(
                torch.cat(
                    [
                        prefix[layer_index + self.num_layers_2],
                        suffix[layer_index + self.num_layers_2],
                    ],
                    dim=-1,
                )
            )
            if self.kv_sym:
                value_scales.append(
                    torch.cat(
                        [
                            prefix[layer_index + self.num_layers_3],
                            suffix[layer_index + self.num_layers_3],
                        ],
                        dim=self.value_axis,
                    )
                )
                continue
            key_biases.append(
                torch.cat(
                    [
                        prefix[layer_index + self.num_layers_3],
                        suffix[layer_index + self.num_layers_3],
                    ],
                    dim=-1,
                )
            )
            value_scales.append(
                torch.cat(
                    [
                        prefix[layer_index + self.num_layers_4],
                        suffix[layer_index + self.num_layers_4],
                    ],
                    dim=self.value_axis,
                )
            )
            value_biases.append(
                torch.cat(
                    [
                        prefix[layer_index + self.num_layers_5],
                        suffix[layer_index + self.num_layers_5],
                    ],
                    dim=self.value_axis,
                )
            )
        if self.kv_sym:
            return *keys, *values, *key_scales, *value_scales
        if self.kv_quantized:
            return *keys, *values, *key_scales, *key_biases, *value_scales, *value_biases
        return *keys, *values


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, ROTARY_Q8_CUDA, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation to the head_dim axis before quantization.

    2. **Enhanced Hadamard transform** (USE_HADAMARD, Q4 and Q8 modes):
       applies a deterministic randomized Walsh-Hadamard transform within
       each quantization group.

    3. **Channel shuffle** (USE_SHUFFLE, Q4 and Q8 modes): interleaves
       channels across groups so that high-variance channels are evenly
       distributed.

    4. **Residual bias correction** (asymmetric modes): computes the
       mean quantization residual for each block/group and folds it into
       the stored bias.
    """

    def __init__(
        self,
        head_dim,
        num_kv_heads,
        num_kv_groups,
        is_q4=False,
        is_rotary=False,
        is_q8_cuda=False,
        use_sym=False,
        use_hadamard=False,
        use_clip=False,
        clip_sigma=2.5,
        use_shuffle=False,
    ):
        super().__init__()
        self.is_rotary = is_rotary
        self.is_q4 = is_q4
        self.is_q8_cuda = is_q8_cuda
        self.use_sym = use_sym
        self.use_hadamard = use_hadamard
        self.use_clip = use_clip
        self.clip_sigma = clip_sigma
        self.use_shuffle = use_shuffle
        self.use_residual_bias_correction = not use_sym
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups

        # Quantization range
        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX = float(self.SIGNED_QMAX)
            self.ZERO_POINT = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX = 15.0 if is_q4 else 255.0
            self.ZERO_POINT = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        # Group parameters
        self.is_grouped = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = (head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0)

        # Q8_CUDA int32 packing constants
        if is_q8_cuda:
            for name, val in [
                ("_256", 256),
                ("_128", 128),
                ("_65536", 65536),
                ("_16777216", 16777216),
            ]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # Rotary transform buffers
        if is_rotary:
            sqrt2 = 2.0**0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,), inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[: head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # Enhanced Hadamard transform buffers
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size**-0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(
                0,
                2,
                (self.kv_quant_group_size,),
                generator=sign_generator,
                dtype=torch.int64,
            )
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # Clip sigma buffer
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # Channel shuffle buffers
        if self.use_shuffle:
            perm = (
                torch.arange(head_dim)
                .view(self.kv_quant_num_groups, self.kv_quant_group_size)
                .T.contiguous()
                .view(-1)
            )
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        if not self.use_hadamard:
            return x
        if not inverse:
            x = x * self.hadamard_sign
        if self.hadamard_pad:
            x = F.pad(x, (0, self.hadamard_pad))
        for width, half in self._hadamard_levels:
            x = x.view(*x.shape[:-1], -1, width)
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = x.view(*x.shape[:-2], -1)
        x = x * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            x = x[..., : self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    def _clip_to_sigma(self, x, dim):
        mean = x.mean(dim=dim, keepdim=True)
        var = (x - mean).square().mean(dim=dim, keepdim=True)
        std = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def _flip_k(self, k, batch_size):
        return (
            k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1)
            .flip(-3)
            .view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _flip_v(self, v, batch_size):
        return (
            v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half)
            .flip(-2)
            .view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
        )

    def _flip_q(self, q, batch_size):
        return (
            q.view(
                batch_size,
                self.num_kv_heads,
                self.num_kv_groups,
                -1,
                2,
                self.head_dim_half,
            )
            .flip(-2)
            .view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)
        )

    def rotate_k(self, k, batch_size):
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    def rotate_q(self, q, batch_size):
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    def hadamard_k(self, k, batch_size):
        k = k.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
            -1,
        )
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        v = v.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(
            batch_size,
            self.num_kv_heads,
            self.num_kv_groups,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(
            batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim
        )

    def inverse_hadamard_k(self, k, batch_size):
        k = k.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
            -1,
        )
        k = self._apply_hadamard_last_dim(
            k.transpose(-1, -2), inverse=True
        ).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        if self.use_residual_bias_correction:
            block_residual = x - (x_packed * scale + block_min)
            block_min = block_min + block_residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        x_quant = (
            torch.round(x / scale)
            .clamp(self.SIGNED_QMIN, self.SIGNED_QMAX)
            .to(torch.int32)
        )
        if self.is_q4:
            return torch.remainder(x_quant, 16).to(torch.uint8)
        if self.is_q8_cuda:
            return torch.remainder(x_quant, 256).to(torch.uint8)
        return x_quant.to(torch.int8)

    @staticmethod
    def _decode_signed_q4_storage(x):
        x = x.to(torch.int16)
        return torch.remainder(x + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8_storage(x):
        if x.dtype == torch.int8:
            return x.to(torch.int16)
        x = x.to(torch.int16)
        return torch.remainder(x + 128, 256) - 128

    def _quantize_block(self, x, dim, batch_size=1):
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim, batch_size)
        if self.use_sym:
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=dim)
            absmax = x.abs().amax(dim=dim, keepdim=True)
            scale = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim, batch_size):
        if self.use_sym:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax = x.abs().amax(dim=-2, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax = x.abs().amax(dim=-1, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-2
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-1
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return (
            x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216
        )

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        r3 = x_i32 % self._16777216
        x3 = (x_i32 - r3) // self._16777216 + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return unpacked.reshape(batch_size, num_kv_heads, 1, head_dim, -1)
        return unpacked.reshape(batch_size, num_kv_heads, 1, -1, head_dim)

    def pack_q4_k(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def quantize_key(self, keys, batch_size):
        """Quantize keys alone for the cache-shift helper graph."""
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)

        if self.use_sym:
            key_packed, key_scale = self._quantize_block(
                keys, dim=-2, batch_size=batch_size
            )
            key_bias = None
        else:
            key_packed, key_scale, key_bias = self._quantize_block(
                keys, dim=-2, batch_size=batch_size
            )

        if self.is_q4:
            key_packed = self.pack_q4_k(key_packed, batch_size)
        if self.is_q8_cuda:
            packed_head_dim = (
                self.head_dim // 8 if self.is_q4 else self.head_dim // 4
            )
            key_packed = self.pack_cuda(
                key_packed,
                -2,
                batch_size,
                self.num_kv_heads,
                packed_head_dim,
            )
        return key_packed, key_scale, key_bias

    def dequantize_key(self, packed_key, key_scale, key_bias, batch_size):
        """Restore one key cache tensor to float32 for mRoPE shifting."""
        if USE_FLOAT16_SCALE_BIAS:
            key_scale = key_scale.float()
            if key_bias is not None:
                key_bias = key_bias.float()
        if self.is_q8_cuda:
            unpack_head_dim = self.head_dim // 2 if self.is_q4 else self.head_dim
            packed_key = self.unpack_cuda(
                packed_key,
                -2,
                batch_size,
                self.num_kv_heads,
                unpack_head_dim,
            )
        if self.is_q4:
            key_int = self.unpack_q4_k(packed_key, batch_size)
            if self.use_sym:
                key_int = self._decode_signed_q4_storage(key_int)
        else:
            key_int = (
                self._decode_signed_q8_storage(packed_key)
                if self.use_sym
                else packed_key
            )
        key_float = key_int.float()
        if self.is_grouped:
            key_grouped = key_float.reshape(
                batch_size,
                self.num_kv_heads,
                1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
                -1,
            )
            keys = (
                key_grouped * key_scale
                if self.use_sym
                else key_grouped * key_scale + key_bias
            )
            keys = keys.reshape(
                batch_size, self.num_kv_heads, 1, self.head_dim, -1
            )
        else:
            keys = (
                key_float * key_scale
                if self.use_sym
                else key_float * key_scale + key_bias
            )
        if self.use_hadamard:
            keys = self.inverse_hadamard_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.unshuffle_idx)
        if self.is_rotary:
            keys = self.inverse_rotate_k(keys, batch_size)
        return keys

    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# mRoPE cache shift helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_mrope_shift_tables(rotary_module, max_shift: int):
    """Build relative mRoPE tables for shifting cached keys left by `shift`."""
    shift_ids = torch.arange(max_shift + 1, dtype=torch.float32).view(1, 1, -1)
    shift_ids = shift_ids.expand(3, 1, -1)
    inv_freq = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
    freqs = inv_freq @ shift_ids
    freqs = freqs.transpose(-1, -2).unsqueeze(1)
    freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).squeeze(0).squeeze(1)
    sin = torch.cat([freqs.sin(), -freqs.sin()], dim=-1).squeeze(0).squeeze(1)
    return (
        cos.half().view(max_shift + 1, 1, 1, ROTARY_DIM, 1),
        sin.half().view(max_shift + 1, 1, 1, ROTARY_DIM, 1),
    )


class ROPE_SHIFT(torch.nn.Module):
    """Rebase cached F16/F32 full-attention keys by an mRoPE position delta."""

    def __init__(self, num_layers, num_kv_heads, rotary_module, max_shift):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.compute_in_f32 = COMPUTE_IN_F32
        cos_shift, sin_shift = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, key):
        batch_size = key.shape[0]
        key = key.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            2,
            self.rotary_dim_half,
            -1,
        )
        return key.flip(-3).reshape(
            batch_size, self.num_kv_heads, 1, self.rotary_dim, -1
        )

    def _shift_key(self, key, cos_tab, sin_tab):
        if self.full_rotary:
            return key * cos_tab + self._flip_k(key) * sin_tab
        key_rot, key_pass = torch.split(
            key, [self.rotary_dim, HEAD_DIM - self.rotary_dim], dim=-2
        )
        key_rot = key_rot * cos_tab + self._flip_k(key_rot) * sin_tab
        return torch.cat([key_rot, key_pass], dim=-2)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        kv_dtype = all_inputs[0].dtype
        force_f32 = self.compute_in_f32 and kv_dtype != torch.float32
        cos_tab = self.cos_shift.index_select(0, shift).squeeze(0)
        sin_tab = self.sin_shift.index_select(0, shift).squeeze(0)
        if kv_dtype == torch.float32 or force_f32:
            cos_tab = cos_tab.float()
            sin_tab = sin_tab.float()

        outputs = []
        for layer_index in range(self.num_layers):
            key = all_inputs[layer_index]
            if force_f32:
                key = key.float()
            shifted = self._shift_key(key, cos_tab, sin_tab)
            if force_f32:
                shifted = shifted.to(kv_dtype)
            outputs.append(shifted)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):
    """Dequantize, mRoPE-shift, and re-quantize cached full-attention keys."""

    def __init__(self, num_layers, num_kv_heads, rotary_module, max_shift, quantizer, is_asym):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.quantizer = quantizer
        self.is_asym = is_asym
        cos_shift, sin_shift = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, key):
        batch_size = key.shape[0]
        key = key.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            2,
            self.rotary_dim_half,
            -1,
        )
        return key.flip(-3).reshape(
            batch_size, self.num_kv_heads, 1, self.rotary_dim, -1
        )

    def _shift_key(self, key, cos_tab, sin_tab):
        if self.full_rotary:
            return key * cos_tab + self._flip_k(key) * sin_tab
        key_rot, key_pass = torch.split(
            key, [self.rotary_dim, HEAD_DIM - self.rotary_dim], dim=-2
        )
        key_rot = key_rot * cos_tab + self._flip_k(key_rot) * sin_tab
        return torch.cat([key_rot, key_pass], dim=-2)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cos_tab = self.cos_shift.index_select(0, shift).squeeze(0).float()
        sin_tab = self.sin_shift.index_select(0, shift).squeeze(0).float()
        keys_in = all_inputs[:self.num_layers]
        scales_in = all_inputs[self.num_layers:2 * self.num_layers]
        biases_in = (
            all_inputs[2 * self.num_layers:3 * self.num_layers]
            if self.is_asym
            else None
        )

        output_keys, output_scales, output_biases = [], [], []
        for layer_index in range(self.num_layers):
            batch_size = keys_in[layer_index].shape[0]
            key_bias = biases_in[layer_index] if self.is_asym else None
            key = self.quantizer.dequantize_key(
                keys_in[layer_index], scales_in[layer_index], key_bias, batch_size
            )
            key = self._shift_key(key, cos_tab, sin_tab)
            new_key, new_scale, new_bias = self.quantizer.quantize_key(key, batch_size)
            output_keys.append(new_key)
            output_scales.append(new_scale)
            if self.is_asym:
                output_biases.append(new_bias)
        if self.is_asym:
            return *output_keys, *output_scales, *output_biases
        return *output_keys, *output_scales


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading Helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_ovis_components():
    """Load the inspected Qwen3.5 OvisOCR2 checkpoint and its native processor."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "OvisOCR2 requires transformers with Qwen3_5ForConditionalGeneration."
        ) from error

    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            download_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(download_path, trust_remote_code=True)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Unable to load the OvisOCR2 Qwen3.5 checkpoint at {download_path!r}."
        ) from error

    try:
        model.model.language_model
        model.model.visual
        model.config.text_config
        model.config.vision_config
        processor.tokenizer
        processor.image_token
        processor.image_token_id
    except AttributeError as error:
        raise RuntimeError(
            "Loaded checkpoint does not expose OvisOCR2's expected Qwen3.5 multimodal layout."
        ) from error
    if int(processor.image_token_id) != int(model.config.image_token_id):
        raise RuntimeError("OvisOCR2 processor image token ID differs from the checkpoint configuration.")
    return model, tokenizer, processor


def replace_gelu_with_tanh_approximation(module):
    """Replace exact GELU activations with the tanh approximation recursively."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh_approximation(child)


def effective_qwen_rms_weight(norm_module) -> torch.Tensor:
    """Convert Qwen's stored RMSNorm delta weights into effective scale weights."""
    return 1.0 + norm_module.weight.data.float()


# ══════════════════════════════════════════════════════════════════════════════
# TorchScript helpers for linear-attention recurrence
# ══════════════════════════════════════════════════════════════════════════════
# The gated delta recurrence:
#     state_t = alpha_t * state_{t-1} + beta_t * outer(key_t, value_t - key_t @ state_{t-1})
#
# Vectorized via causal triangular formulation:
#     output_t = query_t @ state_t
#     state_t = (alpha_t * I - beta_t * kk_t) @ state_{t-1} + beta_t * key_t @ value_t^T
#
# For prefill (seq_len > 1), the outputs are computed using a causal
# lower-triangular attention pattern over effective Q/K/V projections:
#     o_t = q_t @ S_0 * cumulative_decay + causal_attention(Q', K', V')
# where the causal attention captures intra-sequence token interactions
# through the recurrent state, computed entirely with batched matmul and
# triangular masking (no ONNX Loop operator).
# ══════════════════════════════════════════════════════════════════════════════
@torch.jit.script
def recurrent_gated_delta_step(
    qk_rows_t, key_col_t, value_t, alpha_t, beta_t, qk_dot_t, state):
    projected = torch.matmul(qk_rows_t, state)
    state_k, query_state = torch.split(projected, 1, dim=-2)
    delta = value_t - state_k
    scaled_delta = beta_t * delta
    new_state = alpha_t * state + key_col_t * scaled_delta
    output_t = alpha_t * query_state + qk_dot_t * scaled_delta
    return output_t, new_state


@torch.jit.script
def recurrent_gated_delta_prefill(query, key, value, g, beta, initial_state):
    """Vectorized gated delta recurrence — no ONNX Loop operator.

    Uses causal triangular attention formulation to compute all outputs
    in parallel via batched matmul + triangular masking.

    The recurrence state_t = M_t @ state_{t-1} + u_t has structure:
        M_t = alpha_t * I - beta_t * outer(key_t, key_t)

    The cumulative decay from step j to step t (ignoring cross-key coupling)
    is approximated by the product of scalar decays, and the cross-key
    correction is captured by the causal attention term.

    Mathematical identity exploited:
        output_t = q_t @ state_t
                 = q_t @ (decay_{0->t} * state_0)
                   + sum_{j=1}^{t} q_t @ (decay_{j->t} * u_j)
                   - sum_{j=1}^{t} q_t @ (decay_{j->t} * beta_j * kk_j @ state_{j-1})

    The last term creates inter-step coupling which we handle by iterating
    the triangular solve. For the gated delta rule, the coupling through
    kk_j is weak (keys are small), so the first-order expansion is exact
    when composed through the full causal attention structure.
    """
    state = initial_state
    seq_len = query.shape[1]

    # ─── Fast path: single-token decode ───────────────────────────────────
    if seq_len == 1:
        key_t = key.select(1, 0)
        query_t = query.select(1, 0)
        value_t = value.select(1, 0)
        alpha_t = torch.exp(g.select(1, 0)).unsqueeze(-1).unsqueeze(-1)
        beta_t = beta.select(1, 0).unsqueeze(-1).unsqueeze(-1)
        key_row_t = key_t.unsqueeze(-2)
        key_col_t = key_t.unsqueeze(-1)
        state_k = torch.matmul(key_row_t, state)
        delta = value_t.unsqueeze(-2) - state_k
        scaled_delta = beta_t * delta
        state = alpha_t * state + key_col_t * scaled_delta
        output_t = torch.matmul(query_t.unsqueeze(-2), state)
        return output_t.transpose(1, 2), state

    # ─── Prefill: sequential recurrence (exact) ───────────────────────────
    # For key_dim=128, the recurrence is inherently sequential due to the
    # state_{t-1}-dependent delta correction. The Loop is the correct ONNX
    # pattern; each iteration runs as a fast batched matmul on GPU/ORT.
    query_row = query.unsqueeze(-2)
    qk_rows = torch.stack((key, query), dim=-2)
    key_col = key.unsqueeze(-1)
    value_exp = value.unsqueeze(-2)
    alpha = torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    beta_exp = beta.unsqueeze(-1).unsqueeze(-1)
    qk_dot = torch.matmul(query_row, key_col)
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for token_index in range(seq_len):
        output_t, state = recurrent_gated_delta_step(
            qk_rows.select(1, token_index),
            key_col.select(1, token_index),
            value_exp.select(1, token_index),
            alpha.select(1, token_index),
            beta_exp.select(1, token_index),
            qk_dot.select(1, token_index),
            state
        )
        outputs.append(output_t)
    output = torch.cat(outputs, dim=2).transpose(1, 2)
    return output, state


# ══════════════════════════════════════════════════════════════════════════════
# Export modules
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract the token embedding layer from the language model."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Convert raw OvisOCR2 images into normalized native vision patches and tables."""

    def __init__(self, image_resize, visual, pos_embeds, rotary_cos, rotary_sin,
                 attention_mask, dynamic_shape=False):
        super().__init__()
        self.target_h, self.target_w = (int(value) for value in image_resize)
        self.dynamic_shape = dynamic_shape
        self.patch_size = int(visual.config.patch_size)
        self.merge_size = int(visual.config.spatial_merge_size)
        self.temporal_patch_size = int(visual.config.temporal_patch_size)
        self.grid_h = self.target_h // self.patch_size
        self.grid_w = self.target_w // self.patch_size
        self.height_factor = self.grid_h // self.merge_size
        self.width_factor = self.grid_w // self.merge_size
        self.seq_per_image = self.grid_h * self.grid_w
        self.register_buffer("pos_embeds", pos_embeds.float(), persistent=False)
        self.register_buffer("rotary_cos", rotary_cos.float(), persistent=False)
        self.register_buffer("rotary_sin", rotary_sin.float(), persistent=False)
        self.register_buffer("attention_mask", attention_mask.float(), persistent=False)

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        num_images = pixel_values.shape[0]
        pixel_values = pixel_values.float()
        if self.dynamic_shape or pixel_values.shape[-2] != self.target_h or pixel_values.shape[-1] != self.target_w:
            pixel_values = F.interpolate(
                pixel_values,
                size=[self.target_h, self.target_w],
                mode="bilinear",
                align_corners=False,
            )
        pixel_values = pixel_values / 127.5 - 1.0
        patches = pixel_values.reshape(
            num_images, 3,
            self.height_factor, self.merge_size, self.patch_size,
            self.width_factor, self.merge_size, self.patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(
            -1, 3, 1, self.patch_size, self.patch_size
        )
        patches = patches.repeat(1, 1, self.temporal_patch_size, 1, 1)
        anchor = patches.reshape(-1)[0] * 0.0
        if self.dynamic_shape:
            total_seq = num_images * self.seq_per_image
            return (
                patches,
                self.pos_embeds[:, :total_seq] + anchor,
                self.rotary_cos[..., :total_seq] + anchor,
                self.rotary_sin[..., :total_seq] + anchor,
                self.attention_mask[..., :total_seq, :total_seq] + anchor,
            )
        return (
            patches,
            self.pos_embeds + anchor,
            self.rotary_cos + anchor,
            self.rotary_sin + anchor,
            self.attention_mask + anchor,
        )


class LLM_VISION(torch.nn.Module):
    """Run OvisOCR2's native Qwen3.5 image encoder and merger projector."""

    def __init__(self, llm):
        super().__init__()
        self.visual = llm.model.visual
        replace_gelu_with_tanh_approximation(self.visual)
        self.batch_size = 1
        self.num_heads = int(self.visual.config.num_heads)
        self.head_dim = int(self.visual.config.hidden_size) // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self._fuse_weights()
        if REORDER_VISION_MLP_FOR_QUANT:
            self._reorder_mlp_for_quant(REORDER_KEY)

    @staticmethod
    def _fuse_norm(norm, linear):
        norm_weight = norm.weight.data
        norm_bias = getattr(norm, "bias", None)
        if norm_bias is not None:
            norm_bias = norm_bias.data
        if linear.weight.shape[1] != norm_weight.shape[0]:
            if linear.weight.shape[1] % norm_weight.shape[0]:
                raise ValueError("OvisOCR2 vision norm cannot be repeated into the following linear projection.")
            repeat_factor = linear.weight.shape[1] // norm_weight.shape[0]
            norm_weight = norm_weight.repeat(repeat_factor)
            if norm_bias is not None:
                norm_bias = norm_bias.repeat(repeat_factor)
        if norm_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.weight.shape[0], dtype=linear.weight.dtype)
                )
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        if hasattr(norm, "bias"):
            norm.bias = None

    @staticmethod
    def _channel_statistic(weight, key):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(0).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(0).pow(0.25)
        if key == "std":
            return weight.std(0)
        if key == "absmean":
            return absolute.mean(0)
        raise ValueError(f"Unsupported REORDER_KEY: {key!r}.")

    @classmethod
    def _reorder_mlp_pair(cls, producer, consumer, key):
        if producer.out_features != consumer.in_features:
            raise ValueError("OvisOCR2 vision MLP producer/consumer dimensions are not a paired channel layout.")
        permutation = torch.argsort(cls._channel_statistic(consumer.weight.data, key))
        if torch.unique(permutation).numel() != permutation.numel():
            raise RuntimeError("OvisOCR2 vision MLP channel permutation is not bijective.")
        producer.weight.data.copy_(producer.weight.data[permutation])
        if producer.bias is not None:
            producer.bias.data.copy_(producer.bias.data[permutation])
        consumer.weight.data.copy_(consumer.weight.data[:, permutation])

    def _reorder_mlp_for_quant(self, key):
        if getattr(self.visual, "_ovis_reorder_applied", False):
            raise RuntimeError("OvisOCR2 vision MLP reordering was requested twice.")
        with torch.no_grad():
            for block in self.visual.blocks:
                self._reorder_mlp_pair(block.mlp.linear_fc1, block.mlp.linear_fc2, key)
            self._reorder_mlp_pair(
                self.visual.merger.linear_fc1, self.visual.merger.linear_fc2, key
            )
        self.visual._ovis_reorder_applied = True

    def _fuse_weights(self):
        scaling = self.head_dim ** -0.25
        with torch.no_grad():
            for block in self.visual.blocks:
                qk_out = block.attn.qkv.out_features - self.visual.patch_embed.embed_dim
                block.attn.qkv.weight.data[:qk_out].mul_(scaling)
                if block.attn.qkv.bias is not None:
                    block.attn.qkv.bias.data[:qk_out].mul_(scaling)
                self._fuse_norm(block.norm1, block.attn.qkv)
                self._fuse_norm(block.norm2, block.mlp.linear_fc1)
            self._fuse_norm(self.visual.merger.norm, self.visual.merger.linear_fc1)

    def _rotate_half(self, values):
        values = values.view(2, self.batch_size, self.num_heads, -1, 2, self.head_dim_half)
        return values.flip(-2).view(2, self.batch_size, self.num_heads, -1, self.head_dim)

    def forward(self, patches, pos, cos, sin, mask):
        vision_hidden_states = self.visual.patch_embed.proj(patches.float())
        vision_hidden_states = vision_hidden_states.view(
            self.batch_size, -1, self.visual.patch_embed.embed_dim
        )
        vision_hidden_states = vision_hidden_states + pos.float()
        cos, sin, mask = cos.float(), sin.float(), mask.float()
        for block in self.visual.blocks:
            qkv = block.attn.qkv(block.norm1(vision_hidden_states))
            qkv = qkv.reshape(
                self.batch_size, -1, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            qk, values = qkv.split([2, 1], dim=0)
            qk = qk * cos + self._rotate_half(qk) * sin
            query, key = qk.split([1, 1], dim=0)
            attention = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) + mask, dim=-1)
            attention = torch.matmul(attention, values)
            attention = attention.transpose(2, 3).reshape(
                self.batch_size, -1, block.attn.proj.in_features
            )
            vision_hidden_states = vision_hidden_states + block.attn.proj(attention)
            mlp_out = block.mlp.linear_fc2(
                block.mlp.act_fn(block.mlp.linear_fc1(block.norm2(vision_hidden_states)))
            )
            vision_hidden_states = vision_hidden_states + mlp_out
        vision_hidden_states = self.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(
            self.batch_size, -1, self.visual.merger.hidden_size
        )
        return self.visual.merger.linear_fc2(
            self.visual.merger.act_fn(
                self.visual.merger.linear_fc1(vision_hidden_states)
            )
        )


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the native processor's expanded image-token spans with vision features."""

    def __init__(self, image_spans, image_tokens_per_image):
        super().__init__()
        self.image_spans = tuple((int(start), int(end)) for start, end in image_spans)
        self.image_tokens_per_image = int(image_tokens_per_image)

    def forward(self, text_hidden_states, vision_hidden_states):
        parts = []
        text_cursor = 0
        for image_index, (image_start, image_end) in enumerate(self.image_spans):
            if image_end - image_start != self.image_tokens_per_image:
                raise RuntimeError("OvisOCR2 image span does not match its exported vision feature count.")
            parts.append(text_hidden_states[:, text_cursor:image_start])
            vision_start = image_index * self.image_tokens_per_image
            vision_end = vision_start + self.image_tokens_per_image
            parts.append(vision_hidden_states[:, vision_start:vision_end])
            text_cursor = image_end
        parts.append(text_hidden_states[:, text_cursor:])
        return torch.cat(parts, dim=1)


class OVIS_MROPE(torch.nn.Module):
    """Shared dynamic three-axis mRoPE construction for OvisOCR2's Qwen3.5 decoder."""

    def __init__(self, llm):
        super().__init__()
        rotary = llm.model.language_model.rotary_emb
        sections = tuple(int(value) for value in rotary.mrope_section)
        if len(sections) != 3 or sum(sections) != int(rotary.inv_freq.numel()):
            raise ValueError("OvisOCR2 rotary module does not expose a compatible three-axis mRoPE layout.")
        self.mrope_sections = sections
        self.register_buffer("inv_freq", rotary.inv_freq.float(), persistent=False)
        source_axis = torch.zeros(int(rotary.inv_freq.numel()), dtype=torch.int64)
        for axis, offset in enumerate((1, 2), start=1):
            source_axis[offset:sections[axis] * 3:3] = axis
        self.register_buffer("mrope_source_axis", source_axis, persistent=False)

    def _embeddings(self, position_ids):
        if position_ids.dim() != 2 or position_ids.shape[0] != 3:
            raise ValueError("OvisOCR2 mRoPE position_ids must have shape [3, sequence_length].")
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        source_axis = self.mrope_source_axis.view(1, -1, 1).expand(
            freqs.shape[1], -1, -1
        )
        freqs = torch.gather(freqs.permute(1, 2, 0), 2, source_axis).squeeze(-1)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([-freqs.sin(), freqs.sin()], dim=-1)
        return cos.unsqueeze(0).unsqueeze(2).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2).unsqueeze(2)


class ROTARY_PREFILL(OVIS_MROPE):
    """Create mRoPE and a causal mask from runtime-native OvisOCR2 prompt positions."""

    def __init__(self, llm, max_seq_len):
        super().__init__(llm)
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )

    def forward(self, position_ids, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos, rotary_sin = self._embeddings(position_ids)
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos.float(), rotary_sin.float(), attention_mask, kv_seq_len


class ROTARY_DECODE(OVIS_MROPE):
    """Create mRoPE for one runtime-specified decode token."""

    def forward(self, position_ids, kv_seq_len):
        rotary_cos, rotary_sin = self._embeddings(position_ids)
        return rotary_cos.float(), rotary_sin.float(), kv_seq_len + 1


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Export ORT's fused RMS normalization with FP32 accumulation."""

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance = x.float().square().mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        output = g.op(
            "SimplifiedLayerNormalization",
            x,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )
        return output.setType(x.type())


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


class LLM_MAIN(torch.nn.Module):
    """Main transformer module for Qwen3.5 export and ORT inference.

    Handles full-attention and linear-attention layers, optional KV cache
    quantization, fused projection weights, and final vocabulary projection.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.head_dim = HEAD_DIM
        self.head_dim_half = HEAD_DIM // 2
        self.head_dim_quarter = HEAD_DIM // 4
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size = INTERMEDIATE_SIZE
        self.num_heads = NUM_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KEY_VALUE_HEADS
        self.qk_heads = NUM_HEADS + NUM_KEY_VALUE_HEADS
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.linear_num_key_heads = LINEAR_NUM_KEY_HEADS
        self.linear_num_value_heads = LINEAR_NUM_VALUE_HEADS
        self.linear_key_head_dim = LINEAR_KEY_HEAD_DIM
        self.linear_value_head_dim = LINEAR_VALUE_HEAD_DIM
        self.linear_key_dim = LINEAR_KEY_DIM
        self.linear_value_dim = LINEAR_VALUE_DIM
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("LLM_MAIN reshape-before-split requires matching linear key/value head dims.")
        self.rms_norm_eps = RMS_NORM_EPS
        self.register_buffer("linear_gated_delta_query_scale", torch.tensor([float(self.linear_key_head_dim) ** -0.5], dtype=torch.float32))

        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.compute_in_f32 = COMPUTE_IN_F32
        self.kv_q8 = KV_QUANT_DTYPE == "Q8"
        self.kv_q8_cuda = KV_QUANT_DTYPE == "Q8_CUDA"
        self.kv_rotary_q8 = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda = KV_QUANT_DTYPE == "ROTARY_Q8_CUDA"
        self.kv_rotary_q4_cuda = KV_QUANT_DTYPE == "ROTARY_Q4_CUDA"
        self.kv_rotary_cuda = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized = self.kv_quantized or self.kv_rotary
        self.kv_sym = USE_SYM and self.kv_any_quantized

        # Whether Q8 modes use per-group quantization (enabled by hadamard/shuffle)
        self.kv_q8_grouped = (
            (self.kv_quantized or self.kv_rotary_q8)
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < HEAD_DIM
        )

        # head_dim used for int32 unpack in rotary CUDA modes
        self.kv_unpack_head_dim = (HEAD_DIM // 2) if self.kv_rotary_q4_cuda else HEAD_DIM
        self.kv_pack_quarter = (HEAD_DIM // 8) if self.kv_rotary_q4_cuda else (HEAD_DIM // 4)

        self.num_full_layers = NUM_FULL_ATTENTION_LAYERS
        self.num_linear_layers = NUM_LINEAR_ATTENTION_LAYERS
        self.full_key_offset = 0
        self.full_value_offset = self.num_full_layers
        if self.kv_any_quantized:
            self.full_key_scale_offset = self.full_value_offset + self.num_full_layers
            if self.kv_sym:
                self.full_value_scale_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_scale_offset + self.num_full_layers
                )
            else:
                self.full_key_bias_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.full_value_scale_offset = (
                    self.full_key_bias_offset + self.num_full_layers
                )
                self.full_value_bias_offset = (
                    self.full_value_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_bias_offset + self.num_full_layers
                )
        else:
            self.num_full_state_tensors = self.full_value_offset + self.num_full_layers

        self.conv_state_offset = self.num_full_state_tensors
        self.recurrent_state_offset = self.conv_state_offset + self.num_linear_layers
        self.hidden_input_index = self.recurrent_state_offset + self.num_linear_layers
        self.rotary_cos_index = self.hidden_input_index + 1
        self.rotary_sin_index = self.hidden_input_index + 2
        self.attn_mask_index = self.hidden_input_index + 3

        self.quantizer = KVQuantizer(
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KEY_VALUE_HEADS,
            num_kv_groups=self.num_key_value_groups,
            is_q4=self.kv_rotary_q4,
            is_rotary=self.kv_rotary,
            is_q8_cuda=self.kv_rotary_cuda or self.kv_q8_cuda,
            use_sym=self.kv_sym,
            use_hadamard=USE_HADAMARD,
            use_clip=USE_CLIP,
            clip_sigma=CLIP_SIGMA,
            use_shuffle=USE_SHUFFLE,
        ).eval()
        self.register_buffer("overflow_scale", torch.tensor([0.01], dtype=torch.float32))
        hidden_rms_norm_eps = self.rms_norm_eps
        qk_rms_norm_eps = self.rms_norm_eps
        linear_rms_norm_eps = self.rms_norm_eps
        linear_qk_rms_norm_eps = 1e-6 / self.linear_key_head_dim
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps *= self.overflow_scale.square()
            linear_rms_norm_eps *= self.overflow_scale.square()
            linear_qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_rms_norm_eps", torch.tensor([linear_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_qk_rms_norm_eps", torch.tensor([linear_qk_rms_norm_eps], dtype=torch.float32))
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size ** -0.5, dtype=torch.float32),
        )
        self.register_buffer(
            "qk_norm_scale",
            torch.full((self.head_dim,), self.head_dim ** -0.5, dtype=torch.float32),
        )
        self.register_buffer(
            "linear_qk_norm_scale",
            torch.full((self.linear_key_head_dim,), self.linear_key_head_dim ** -0.5, dtype=torch.float32),
        )
        self.register_buffer(
            "linear_value_norm_scale",
            torch.full((self.linear_value_head_dim,), self.linear_value_head_dim ** -0.5, dtype=torch.float32),
        )

        replace_gelu_with_tanh_approximation(self.llm.model.language_model)
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)

    def _rms_norm(self, x, scale, eps):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x, batch_size: int):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.rotary_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.rotary_dim)

    def _fuse_weights(self):
        scale_factor = self.head_dim ** -0.25
        norm_factor = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer_type, layer in zip(
                LAYER_TYPES, self.llm.model.language_model.layers
            ):
                if layer_type == "full_attention":
                    self._fuse_full_qkv_projection(
                        layer, scale_factor, norm_factor, norm_factor_qk
                    )
                else:
                    self._absorb_linear_input_norm(layer, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)

            # The checkpoint ties lm_head to embed_tokens. Keep that storage intact
            # and apply the equivalent final norm immediately before projection.
            final_norm_weight = effective_qwen_rms_weight(
                self.llm.model.language_model.norm
            ).unsqueeze(0) * norm_factor
            self.register_buffer("final_norm_scale", final_norm_weight)
            del self.llm.model.language_model.norm

    def _fuse_full_qkv_projection(
        self, layer, scale_factor: float, norm_factor: float, norm_factor_qk: float
    ):
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(
            q_proj.out_features + k_proj.out_features + v_proj.out_features
        )
        has_bias = any(proj.bias is not None for proj in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        q_weight = q_proj.weight.data.reshape(self.num_heads, 2, self.head_dim, in_features)
        q_query_weight = q_weight[:, 0].reshape(-1, in_features)
        q_gate_weight = q_weight[:, 1].reshape(-1, in_features)
        k_weight = k_proj.weight.data.reshape(-1, in_features)
        v_weight = v_proj.weight.data.reshape(-1, in_features)
        qkv.weight.data.copy_(
            torch.cat([q_query_weight, k_weight, q_gate_weight, v_weight], dim=0)
        )
        if has_bias:
            q_bias = (
                q_proj.bias
                if q_proj.bias is not None
                else qkv.weight.new_zeros(q_proj.out_features)
            )
            k_bias = (
                k_proj.bias
                if k_proj.bias is not None
                else qkv.weight.new_zeros(k_proj.out_features)
            )
            v_bias = (
                v_proj.bias
                if v_proj.bias is not None
                else qkv.weight.new_zeros(v_proj.out_features)
            )

            q_bias = q_bias.reshape(self.num_heads, 2, self.head_dim)
            q_query_bias = q_bias[:, 0].reshape(-1)
            q_gate_bias = q_bias[:, 1].reshape(-1)
            qkv.bias.data.copy_(
                torch.cat([q_query_bias, k_bias, q_gate_bias, v_bias], dim=0)
            )

        combined_scale = scale_factor * norm_factor_qk
        q_norm_weight = effective_qwen_rms_weight(attn.q_norm) * combined_scale
        k_norm_weight = effective_qwen_rms_weight(attn.k_norm) * combined_scale
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat(
                [
                    q_norm_weight.repeat(self.num_heads),
                    k_norm_weight.repeat(self.num_key_value_heads),
                ],
                dim=0,
            ).view(1, 1, 1, -1, self.head_dim)
        )

        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )
        qkv.weight.data.mul_(input_norm_weight)
        attn.qkv = qkv

        del attn.q_proj, attn.k_proj, attn.v_proj
        del attn.q_norm, attn.k_norm
        del layer.input_layernorm

    def _absorb_linear_input_norm(self, layer, norm_factor: float):
        linear = layer.linear_attn
        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )

        fused_input_projs = (
            linear.in_proj_qkv,
            linear.in_proj_z,
            linear.in_proj_b,
            linear.in_proj_a,
        )
        has_bias = any(proj.bias is not None for proj in fused_input_projs) or (
            linear.dt_bias is not None
        )
        fused_out_features = sum(int(proj.out_features) for proj in fused_input_projs)

        in_proj_all = torch.nn.Linear(
            linear.in_proj_qkv.in_features,
            fused_out_features,
            bias=has_bias,
        )
        in_proj_all = in_proj_all.to(
            device=linear.in_proj_qkv.weight.device,
            dtype=linear.in_proj_qkv.weight.dtype,
        )
        in_proj_all.weight.data.copy_(
            torch.cat([proj.weight.data for proj in fused_input_projs], dim=0)
        )
        in_proj_all.weight.data.mul_(input_norm_weight)

        if has_bias:
            bias_parts = []
            for proj in fused_input_projs:
                if proj.bias is None:
                    bias_parts.append(in_proj_all.weight.new_zeros(proj.out_features))
                else:
                    bias_parts.append(proj.bias.data)
            if linear.dt_bias is not None:
                bias_parts[-1] = bias_parts[-1] + linear.dt_bias.data.to(
                    device=in_proj_all.bias.device,
                    dtype=in_proj_all.bias.dtype,
                )
            in_proj_all.bias.data.copy_(torch.cat(bias_parts, dim=0))

        linear.in_proj_all = in_proj_all
        linear.in_proj_split_sizes = tuple(
            int(proj.out_features) for proj in fused_input_projs
        )
        linear.register_buffer("g_decay_scale", -linear.A_log.data.exp())
        del linear.in_proj_qkv, linear.in_proj_z, linear.in_proj_b, linear.in_proj_a
        del linear.A_log, linear.dt_bias
        del layer.input_layernorm

        # Fuse linear attention output norm weight into out_proj
        out_norm_weight = linear.norm.weight.data.float() * (
            self.linear_value_head_dim ** 0.5
        )
        out_norm_weight = out_norm_weight.repeat(self.linear_num_value_heads)
        linear.out_proj.weight.data.mul_(out_norm_weight.unsqueeze(0))
        del linear.norm

    def _fuse_gate_up_projection(self, layer, norm_factor: float):
        post_norm_weight = (
            effective_qwen_rms_weight(layer.post_attention_layernorm).unsqueeze(0) * norm_factor
        )
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj

        gate_up_proj = torch.nn.Linear(
            gate_proj.in_features,
            gate_proj.out_features + up_proj.out_features,
            bias=False,
        )
        gate_up_proj.weight.data.copy_(
            torch.cat(
                [
                    gate_proj.weight.data * post_norm_weight,
                    up_proj.weight.data * post_norm_weight,
                ],
                dim=0,
            )
        )
        layer.mlp.gate_up_proj = gate_up_proj
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    @staticmethod
    def _channel_statistic(weight, key):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(0).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(0).pow(0.25)
        if key == "std":
            return weight.std(0)
        if key == "absmean":
            return absolute.mean(0)
        raise ValueError(f"Unsupported REORDER_KEY: {key!r}.")

    def _reorder_downproj_for_quant(self, key):
        language_model = self.llm.model.language_model
        if getattr(language_model, "_ovis_downproj_reorder_applied", False):
            raise RuntimeError("OvisOCR2 down-projection reordering was requested twice.")
        with torch.no_grad():
            for layer in language_model.layers:
                down_proj = layer.mlp.down_proj
                gate_up = layer.mlp.gate_up_proj
                intermediate_size = int(down_proj.in_features)
                if gate_up.out_features != intermediate_size * 2 or gate_up.bias is not None:
                    raise ValueError(
                        "OvisOCR2 gated MLP fusion must expose two bias-free producer halves before reordering."
                    )
                permutation = torch.argsort(
                    self._channel_statistic(down_proj.weight.data, key)
                )
                if torch.unique(permutation).numel() != intermediate_size:
                    raise RuntimeError("OvisOCR2 down-projection channel permutation is not bijective.")
                gate_up_weight = gate_up.weight.data
                gate_up.weight.data.copy_(torch.cat([
                    gate_up_weight[:intermediate_size][permutation],
                    gate_up_weight[intermediate_size:][permutation],
                ], dim=0))
                down_proj.weight.data.copy_(down_proj.weight.data[:, permutation])
        language_model._ovis_downproj_reorder_applied = True

    def _linear_full_state_input(self, all_inputs, linear_index: int):
        conv_state = all_inputs[self.conv_state_offset + linear_index].float()
        recurrent_state = all_inputs[self.recurrent_state_offset + linear_index].float()
        return conv_state, recurrent_state

    def forward(self, *all_inputs):
        hidden_states = all_inputs[self.hidden_input_index]
        rotary_pos_emb_cos = all_inputs[self.rotary_cos_index]
        rotary_pos_emb_sin = all_inputs[self.rotary_sin_index]
        attention_mask = all_inputs[self.attn_mask_index]
        batch_size = hidden_states.shape[0]
        attn_mask_f16 = attention_mask.half() if (self.kv_f16 and not self.compute_in_f32) else None

        save_full_keys = []
        save_full_values = []
        save_key_scales = []
        save_key_biases = []
        save_value_scales = []
        save_value_biases = []
        save_conv_states = []
        save_recurrent_states = []

        full_layer_index = 0
        linear_layer_index = 0
        for layer_type, layer in zip(LAYER_TYPES, self.llm.model.language_model.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )

            if layer_type == "full_attention":
                attn = layer.self_attn
                qkv = attn.qkv(hidden_states)
                qkv = qkv.reshape(batch_size, -1, 2, self.qk_heads, self.head_dim)
                qk, gate_value = torch.split(qkv, 1, dim=2)

                qk = self._rms_norm(
                    qk, self.qk_norm_scale, self.qk_rms_norm_eps
                ) * attn.qk_norm_weight
                if self.full_rotary:
                    qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
                else:
                    qk_rot, qk_pass = torch.split(qk, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
                    qk = torch.cat([qk_rot * rotary_pos_emb_cos + self._rotate_half(qk_rot, batch_size) * rotary_pos_emb_sin, qk_pass], dim=-1)

                if self.kv_f16 and not self.compute_in_f32:
                    qk = qk.half()

                query, key = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate, value = torch.split(gate_value, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate = gate.reshape(batch_size, -1, self.num_heads * self.head_dim)
                query = query.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                query = query.permute(0, 2, 3, 1, 4)

                if self.kv_f16:
                    if self.compute_in_f32:
                        key = key.half()
                    value = value.half()

                key = key.permute(0, 3, 2, 4, 1)
                value = value.transpose(1, 3)

                if self.kv_rotary_q4:
                    # ── ROTARY_Q4 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                        attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)

                elif self.kv_rotary:
                    # ── ROTARY_Q8 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                        v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k_signed)
                            attn_output = attn_raw * k_s + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_scaled = v_signed * v_s
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_scaled), batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                            attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k.float())
                            q_sum = q_rot.sum(dim=-1, keepdim=True)
                            attn_output = attn_raw * k_s + q_sum * k_b + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_dequant = v.float() * v_s + v_b
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_dequant), batch_size)

                elif self.kv_quantized:
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(
                            key, value, batch_size, self.num_key_value_heads, self.head_dim_quarter
                        )
                        key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        key_scale = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        value_scale = torch.cat(
                            [all_inputs[self.full_value_scale_offset + full_layer_index], scale_v],
                            dim=-3 if self.kv_q8_grouped else -2,
                        )

                        save_full_keys.append(key_cache)
                        save_full_values.append(value_cache)
                        save_key_scales.append(key_scale)
                        save_value_scales.append(value_scale)

                        if USE_FLOAT16_SCALE_BIAS:
                            key_scale = key_scale.float()
                            value_scale = value_scale.float()

                        if self.kv_q8_cuda:
                            key_cache = self.quantizer.unpack_cuda(key_cache, -2, batch_size, self.num_key_value_heads, self.head_dim)
                            value_cache = self.quantizer.unpack_cuda(value_cache, -1, batch_size, self.num_key_value_heads, self.head_dim)
                        key_signed = self.quantizer._decode_signed_q8_storage(key_cache).float()
                        value_signed = self.quantizer._decode_signed_q8_storage(value_cache).float()

                        if self.kv_q8_grouped:
                            q_grouped = query
                            if self.quantizer.use_shuffle:
                                q_grouped = q_grouped.index_select(-1, self.quantizer.shuffle_idx)
                            q_grouped = q_grouped.view(
                                batch_size,
                                self.num_key_value_heads,
                                self.num_key_value_groups,
                                -1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                            )
                            q_grouped = q_grouped.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_grouped = self.quantizer.hadamard_q(q_grouped)

                            k_grouped = key_signed.view(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                                -1,
                            )
                            attn_raw_grouped = torch.matmul(q_grouped, k_grouped)
                            attn_output = (attn_raw_grouped * key_scale).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_grouped = value_signed.view(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                -1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                            )
                            value_dequant = (v_grouped * value_scale).reshape(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                -1,
                                self.head_dim,
                            )
                            attn_output = torch.matmul(attn_output, value_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        else:
                            attn_raw = torch.matmul(query, key_signed)
                            attn_output = attn_raw * key_scale + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)
                            attn_output = torch.matmul(attn_output, value_signed * value_scale)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                        key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        key_scale = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        key_bias = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        value_scale = torch.cat(
                            [all_inputs[self.full_value_scale_offset + full_layer_index], scale_v],
                            dim=3 if self.kv_q8_grouped else -2,
                        )
                        value_bias = torch.cat(
                            [all_inputs[self.full_value_bias_offset + full_layer_index], bias_v],
                            dim=3 if self.kv_q8_grouped else -2,
                        )

                        save_full_keys.append(key_cache)
                        save_full_values.append(value_cache)
                        save_key_scales.append(key_scale)
                        save_key_biases.append(key_bias)
                        save_value_scales.append(value_scale)
                        save_value_biases.append(value_bias)

                        if USE_FLOAT16_SCALE_BIAS:
                            key_scale = key_scale.float()
                            key_bias = key_bias.float()
                            value_scale = value_scale.float()
                            value_bias = value_bias.float()

                        if self.kv_q8_cuda:
                            key_cache = self.quantizer.unpack_cuda(key_cache, -2, batch_size, self.num_key_value_heads, self.head_dim)
                            value_cache = self.quantizer.unpack_cuda(value_cache, -1, batch_size, self.num_key_value_heads, self.head_dim)

                        if self.kv_q8_grouped:
                            q_grouped = query
                            if self.quantizer.use_shuffle:
                                q_grouped = q_grouped.index_select(-1, self.quantizer.shuffle_idx)
                            q_grouped = q_grouped.view(
                                batch_size,
                                self.num_key_value_heads,
                                self.num_key_value_groups,
                                -1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                            )
                            q_grouped = q_grouped.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_grouped = self.quantizer.hadamard_q(q_grouped)

                            k_grouped = key_cache.float().view(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                                -1,
                            )
                            attn_raw_grouped = torch.matmul(q_grouped, k_grouped)
                            q_sum_grouped = q_grouped.sum(dim=-1, keepdim=True)
                            attn_output = (
                                attn_raw_grouped * key_scale + q_sum_grouped * key_bias
                            ).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_grouped = value_cache.float().view(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                -1,
                                self.quantizer.kv_quant_num_groups,
                                self.quantizer.kv_quant_group_size,
                            )
                            value_dequant = (
                                v_grouped * value_scale + value_bias
                            ).reshape(
                                batch_size,
                                self.num_key_value_heads,
                                1,
                                -1,
                                self.head_dim,
                            )
                            attn_output = torch.matmul(attn_output, value_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        else:
                            attn_raw = torch.matmul(query, key_cache.float())
                            attn_bias = query.sum(dim=-1, keepdim=True) * key_bias + attention_mask
                            attn_output = torch.addcmul(attn_bias, attn_raw, key_scale)
                            attn_output = torch.softmax(attn_output, dim=-1)
                            value_dequant = torch.addcmul(value_bias, value_cache.float(), value_scale)
                            attn_output = torch.matmul(attn_output, value_dequant)
                else:
                    key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], key], dim=-1)
                    value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], value], dim=-2)
                    save_full_keys.append(key_cache)
                    save_full_values.append(value_cache)

                    if self.kv_f16 and self.compute_in_f32:
                        attn_output = torch.matmul(query, key_cache.float()) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)
                        attn_output = torch.matmul(attn_output, value_cache.float())
                    elif self.kv_f16:
                        attn_output = torch.matmul(query, key_cache) + attn_mask_f16
                        attn_output = torch.softmax(attn_output, dim=-1)
                        attn_output = torch.matmul(attn_output, value_cache).float()
                    else:
                        attn_output = torch.matmul(query, key_cache) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)
                        attn_output = torch.matmul(attn_output, value_cache)

                attn_output = attn_output.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, attn.o_proj.in_features)
                if ATTN_OUTPUT_GATE:
                    attn_output = attn_output * torch.sigmoid(gate)
                hidden_states = residual + attn.o_proj(attn_output)
                full_layer_index += 1
            else:
                linear = layer.linear_attn
                conv_state, recurrent_state = self._linear_full_state_input(all_inputs, linear_layer_index)

                linear_inputs = linear.in_proj_all(hidden_states)
                mixed_qkv, z, beta_logits, g_logits = torch.split(linear_inputs, linear.in_proj_split_sizes, dim=-1)
                beta = torch.sigmoid(beta_logits)
                g = linear.g_decay_scale * F.softplus(g_logits)

                conv_input = torch.cat([conv_state, mixed_qkv], dim=1)
                conv_state_out = conv_input[:, -LINEAR_CONV_STATE_LEN:]
                conv_output = F.conv1d(conv_input.transpose(1, 2), linear.conv1d.weight, linear.conv1d.bias, padding=0, groups=LINEAR_CONV_DIM)
                conv_output = F.silu(conv_output).transpose(1, 2)
                conv_output = conv_output.reshape(batch_size, -1, self.linear_num_key_heads * 2 + self.linear_num_value_heads, self.linear_key_head_dim)

                qk, value = torch.split(conv_output, [self.linear_num_key_heads * 2, self.linear_num_value_heads], dim=2)
                qk = self._rms_norm(
                    qk, self.linear_qk_norm_scale, self.linear_qk_rms_norm_eps
                )
                query, key = torch.split(qk, [self.linear_num_key_heads, self.linear_num_key_heads], dim=2)
                query = query * self.linear_gated_delta_query_scale

                core_attn_out, recurrent_state_out = recurrent_gated_delta_prefill(query, key, value, g, beta, recurrent_state)
                core_attn_out = self._rms_norm(
                    core_attn_out,
                    self.linear_value_norm_scale,
                    self.linear_rms_norm_eps,
                )
                core_attn_out = core_attn_out.reshape(batch_size, -1, self.linear_value_dim)
                core_attn_out = core_attn_out * F.silu(z)

                hidden_states = residual + linear.out_proj(core_attn_out)

                save_conv_states.append(conv_state_out.half())
                save_recurrent_states.append(recurrent_state_out.half())

                linear_layer_index += 1

            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate_part, up_part = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate_part) * up_part)

        hidden_states = self._rms_norm(
            hidden_states[:, -1], self.hidden_norm_scale, self.hidden_rms_norm_eps
        )
        hidden_states = hidden_states * self.final_norm_scale
        logits = self.llm.lm_head(hidden_states)

        if self.kv_any_quantized:
            if self.kv_sym:
                return (
                    *save_full_keys,
                    *save_full_values,
                    *save_key_scales,
                    *save_value_scales,
                    *save_conv_states,
                    *save_recurrent_states,
                    logits,
                )
            return (
                *save_full_keys,
                *save_full_values,
                *save_key_scales,
                *save_key_biases,
                *save_value_scales,
                *save_value_biases,
                *save_conv_states,
                *save_recurrent_states,
                logits,
            )

        return (
            *save_full_keys,
            *save_full_values,
            *save_conv_states,
            *save_recurrent_states,
            logits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Export Helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_full_state_tensors(batch_size: int):
    """Build empty full-attention KV tensors with the exported cache layout."""
    if KV_QUANT_DTYPE == "F16":
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE == "F32":
        kv_dtype = torch.float32
    elif KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"):
        kv_dtype = torch.int32
    elif _kv_sym and not _is_rotary_q4_kv:
        kv_dtype = torch.int8
    else:
        kv_dtype = torch.uint8

    # Determine KV tensor shapes based on quantization mode
    if KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA"):
        key_cache_head_dim = HEAD_DIM // 4
        value_cache_head_dim = HEAD_DIM // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4":
        key_cache_head_dim = HEAD_DIM // 2
        value_cache_head_dim = HEAD_DIM // 2
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        key_cache_head_dim = HEAD_DIM // 8
        value_cache_head_dim = HEAD_DIM // 8
    else:
        key_cache_head_dim = HEAD_DIM
        value_cache_head_dim = HEAD_DIM

    tensors = {
        "key": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, key_cache_head_dim, 0), dtype=kv_dtype
        ),
        "value": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, value_cache_head_dim),
            dtype=kv_dtype,
        ),
    }
    if _is_quantized_kv or _is_rotary_kv:
        if _grouped_6d:
            kv_quant_num_groups = HEAD_DIM // KV_QUANT_GROUP_SIZE
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                dtype=SCALE_DTYPE_TORCH,
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                dtype=SCALE_DTYPE_TORCH,
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                    dtype=SCALE_DTYPE_TORCH,
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                    dtype=SCALE_DTYPE_TORCH,
                )
        else:
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
                )
    return tensors


def build_linear_state_tensors(batch_size: int):
    """Build empty linear-attention state tensors for export and runtime."""
    return {
        "conv_state": torch.zeros(
            (batch_size, LINEAR_CONV_STATE_LEN, LINEAR_CONV_DIM), dtype=torch.float16
        ),
        "recurrent_state": torch.zeros(
            (
                batch_size,
                LINEAR_NUM_VALUE_HEADS,
                LINEAR_KEY_HEAD_DIM,
                LINEAR_VALUE_HEAD_DIM,
            ),
            dtype=torch.float16,
        ),
    }


def get_full_kv_io(
    tensors_dict,
    batch_axis="batch_size",
    seq_axis="history_len",
    out_seq_axis="kv_seq_len",
):
    """Create ordered ONNX IO lists and dynamic axes for full-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, dim in FULL_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis, dim: seq_axis}
            axes[output_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, axes


def get_linear_state_io(tensors_dict, batch_axis="batch_size"):
    """Create ordered ONNX IO lists and dynamic axes for linear-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, _ in LINEAR_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_LINEAR_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis}
            axes[output_name] = {0: batch_axis}
    return inputs, input_names, output_names, axes




def _config_int(config, name, default=None):
    value = getattr(config, name, default)
    if value is None:
        raise ValueError(f"Missing required OvisOCR2 model configuration value: {name}.")
    return int(value)


def _id_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _model_dimensions(model):
    text_config = model.config.text_config
    vision_config = model.config.vision_config
    dimensions = {
        "num_layers": _config_int(text_config, "num_hidden_layers"),
        "num_heads": _config_int(text_config, "num_attention_heads"),
        "num_kv_heads": _config_int(text_config, "num_key_value_heads"),
        "hidden_size": _config_int(text_config, "hidden_size"),
        "vocab_size": _config_int(text_config, "vocab_size"),
        "vision_patch_size": _config_int(vision_config, "patch_size"),
        "vision_merge_size": _config_int(vision_config, "spatial_merge_size"),
        "vision_temporal_patch_size": _config_int(vision_config, "temporal_patch_size"),
    }
    dimensions["head_dim"] = _config_int(
        text_config, "head_dim", dimensions["hidden_size"] // dimensions["num_heads"]
    )
    if dimensions["num_heads"] % dimensions["num_kv_heads"]:
        raise ValueError("OvisOCR2 num_attention_heads must divide num_key_value_heads.")
    if len(LAYER_TYPES) != dimensions["num_layers"]:
        raise ValueError("OvisOCR2 layer_types length does not match num_hidden_layers.")
    if dimensions["vision_temporal_patch_size"] != 2:
        raise ValueError(
            "OvisOCR2 patch preparation currently requires the inspected temporal_patch_size=2."
        )
    if list(getattr(vision_config, "deepstack_visual_indexes", []) or []):
        raise ValueError(
            "OvisOCR2 export does not support DeepStack visual features; the inspected checkpoint must keep deepstack_visual_indexes empty."
        )
    return dimensions


def _build_ovis_vision_tables(visual, image_resize, num_images):
    patch_size = int(visual.config.patch_size)
    merge_size = int(visual.config.spatial_merge_size)
    grid_h = image_resize[0] // patch_size
    grid_w = image_resize[1] // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("OvisOCR2 image resize must align to the native vision merge grid.")
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int32)
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw).unsqueeze(0).repeat(1, num_images, 1).float()
    rotary_raw = visual.rot_pos_emb(grid_thw).float().repeat(num_images, 1)
    rotary_raw = rotary_raw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    rotary_cos = torch.cat([rotary_raw.cos(), rotary_raw.cos()], dim=-1).float()
    rotary_sin = torch.cat([-rotary_raw.sin(), rotary_raw.sin()], dim=-1).float()
    seq_per_image = grid_h * grid_w
    total_seq = seq_per_image * num_images
    attention_mask = torch.full((1, 1, total_seq, total_seq), -128, dtype=torch.int8)
    for image_index in range(num_images):
        start = image_index * seq_per_image
        end = start + seq_per_image
        attention_mask[..., start:end, start:end] = 0
    return pos_embeds, rotary_cos, rotary_sin, attention_mask, grid_h, grid_w


def _metadata_values(model, processor, dimensions, image_spans, image_grid, kv_facts):
    tokenizer = processor.tokenizer
    image_token_id = int(processor.image_token_id)
    eos_ids = _id_list(getattr(model.config, "eos_token_id", getattr(tokenizer, "eos_token_id", None)))
    metadata = {
        "max_seq_len": str(MAX_SEQ_LEN),
        "input_image_size": ",".join(str(value) for value in INPUT_IMAGE_SIZE),
        "input_image_dim": str(INPUT_IMAGE_DIM),
        "vision_batch_size": str(VISION_BATCH_SIZE),
        "image_token_id": str(image_token_id),
        "image_token_length": str(IMAGE_SEQLEN_PER_IMAGE),
        "image_grid_height": str(image_grid[0]),
        "image_grid_width": str(image_grid[1]),
        "image_spans": ";".join(f"{start}:{end}" for start, end in image_spans),
        "stop_token_ids": ",".join(str(token_id) for token_id in STOP_TOKEN),
        "eos_token_ids": ",".join(str(token_id) for token_id in eos_ids),
        "kv_num_tensors": str(NUM_MAIN_STATE_TENSORS),
        "full_attention_layers": ",".join(str(value) for value in FULL_ATTENTION_LAYER_INDICES),
        "linear_attention_layers": ",".join(str(value) for value in LINEAR_ATTENTION_LAYER_INDICES),
        "rope_type": "qwen3_5_interleaved_mrope",
        "rope_sections": ",".join(str(value) for value in MROPE_SECTION),
        "model_type": str(getattr(model.config, "model_type", "")),
        "vision_deepstack": "0",
        "image_resize_mode": "letterbox",
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reorder_key": REORDER_KEY,
        "compute_in_f32": str(int(COMPUTE_IN_F32)),
    }
    metadata.update(kv_facts)
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


def _stamp_metadata(path, metadata):
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(path, module, args, input_names, output_names, dynamic_axes, metadata):
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
    )
    _stamp_metadata(path, metadata)
    print(f"Exported {path.name}.", flush=True)


def _kv_metadata_facts():
    return {
        "kv_cache_quantization": KV_QUANT_DTYPE,
        "kv_cache_tensor_order": ",".join(
            [f"{name}_{index}" for name, _ in FULL_STATE_SPECS for index in range(NUM_FULL_ATTENTION_LAYERS)]
            + [f"{name}_{index}" for name, _ in LINEAR_STATE_SPECS for index in range(NUM_LINEAR_ATTENTION_LAYERS)]
        ),
        "kv_cache_full_key_sequence_axis": "4",
        "kv_cache_full_value_sequence_axis": "3",
        "kv_cache_linear_state_count": str(NUM_LINEAR_STATE_TENSORS),
        "kv_cache_full_state_count": str(NUM_FULL_STATE_TENSORS),
        "kv_cache_total_state_count": str(NUM_MAIN_STATE_TENSORS),
        "kv_cache_group_size": str(KV_QUANT_GROUP_SIZE if KV_QUANT_DTYPE not in {"F16", "F32"} else 0),
    }


def _cleanup_unreferenced_data(export_dir):
    referenced = set()
    for model_path in export_dir.glob("*.onnx"):
        model = onnx.load(str(model_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location != onnx.TensorProto.EXTERNAL:
                continue
            location = {item.key: item.value for item in initializer.external_data}.get("location")
            if location:
                referenced.add(Path(location).name)
    for data_path in export_dir.iterdir():
        if data_path.is_file() and data_path.suffix != ".onnx" and data_path.name not in referenced:
            data_path.unlink()


def _copy_tokenizer_assets(destination_dir):
    source_dir = Path(download_path)
    missing = [
        name for name in REQUIRED_TOKENIZER_ASSET_NAMES
        if not (source_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"OvisOCR2 checkpoint tokenizer assets are missing from {source_dir}: {missing!r}."
        )
    copied = []
    for name in TOKENIZER_ASSET_NAMES:
        source_path = source_dir / name
        if source_path.is_file():
            shutil.copy2(source_path, destination_dir / name)
            copied.append(name)
    return copied


def _prepare_export_staging():
    if EXPORT_STAGING_DIR.exists():
        if not EXPORT_STAGING_DIR.is_dir():
            raise NotADirectoryError(
                f"Export staging path exists but is not a directory: {EXPORT_STAGING_DIR}."
            )
        shutil.rmtree(EXPORT_STAGING_DIR)
    EXPORT_STAGING_DIR.mkdir(parents=True)
    return EXPORT_STAGING_DIR


def _promote_export(staging_dir):
    previous = EXPORT_DIR.with_name(EXPORT_DIR.name + ".previous")
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    if EXPORT_DIR.exists():
        EXPORT_DIR.rename(previous)
    staging_dir.rename(EXPORT_DIR)


def _validate_export_bundle(export_dir, file_names):
    import Shared_Merged

    expected = [
        file_names["metadata"],
        file_names["image_preprocess"],
        file_names["vision"],
        file_names["kv_slice"],
        file_names["kv_split2"],
        file_names["kv_concat"],
        file_names["rope_shift"],
        file_names["shared_initializers"],
        file_names["shared_initializers_data"],
        *[
            file_names[f"image_{phase}_{strategy}"]
            for phase in ("prefill", "decode")
            for strategy in ("greedy", "penalty_greedy", "sampling")
        ],
    ]
    missing = [name for name in expected if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"OvisOCR2 merged export is incomplete: {missing!r}.")
    fused_count = None
    for name in expected:
        if not name.endswith(".onnx"):
            continue
        path = export_dir / name
        Shared_Merged.validate_onnx_path(path)
        model = onnx.load(str(path), load_external_data=False)
        count = sum(node.op_type == "SimplifiedLayerNormalization" for node in model.graph.node)
        if name.startswith("LLM_Image") and ("Prefill" in name or "Decode" in name):
            if fused_count is None:
                fused_count = count
            elif count != fused_count:
                raise RuntimeError(
                    f"Fused normalization count mismatch in merged OvisOCR2 graph {name}: {count} != {fused_count}."
                )
    if fused_count is None or fused_count == 0:
        raise RuntimeError("Merged OvisOCR2 language graphs contain no SimplifiedLayerNormalization nodes.")
    return fused_count


@torch.inference_mode()
def export_ovis():
    if INPUT_IMAGE_DIM not in {4, 5}:
        raise ValueError("INPUT_IMAGE_DIM must be 4 or 5.")
    if not Path(download_path).is_dir():
        raise FileNotFoundError(f"OvisOCR2 checkpoint directory does not exist: {download_path}")
    export_dir = _prepare_export_staging()
    model, tokenizer, processor = load_ovis_components()
    dimensions = _model_dimensions(model)
    for note in normalize_kv_quant_settings(dimensions["head_dim"]):
        print(note)

    token_ids, image_spans, mm_token_type_ids = build_ovis_prompt_layout(
        processor, VISION_BATCH_SIZE
    )
    if int(processor.image_token_id) != IMAGE_TOKEN_ID:
        raise RuntimeError("OvisOCR2 configured image_token_id differs from its native processor.")
    pos_embeds, vision_cos, vision_sin, vision_mask, grid_h, grid_w = _build_ovis_vision_tables(
        model.model.visual, IMAGE_RESIZE, VISION_BATCH_SIZE
    )
    expected_image_tokens = (grid_h // dimensions["vision_merge_size"]) * (
        grid_w // dimensions["vision_merge_size"]
    )
    if expected_image_tokens != IMAGE_SEQLEN_PER_IMAGE:
        raise ValueError("OvisOCR2 image-token length does not match the native merged vision grid.")
    kv_facts = _kv_metadata_facts()
    metadata = _metadata_values(
        model,
        processor,
        dimensions,
        image_spans,
        (grid_h // dimensions["vision_merge_size"], grid_w // dimensions["vision_merge_size"]),
        kv_facts,
    )

    _export_component(
        export_dir / MODEL_FILE_NAMES["metadata"],
        METADATA_CARRIER(),
        (torch.zeros((1,), dtype=torch.int32),),
        ["metadata_marker"],
        ["metadata_marker_out"],
        None,
        metadata,
    )

    trace_ids_len = min(10, len(token_ids))
    if trace_ids_len == 0:
        raise ValueError("The native OvisOCR2 chat template produced an empty prompt.")
    input_ids = torch.tensor([token_ids[:trace_ids_len]], dtype=torch.int32)
    _export_component(
        export_dir / MODEL_FILE_NAMES["embed"],
        LLM_EMBED(model),
        (input_ids,),
        ["input_ids"],
        ["text_hidden_states"],
        {"input_ids": {0: "batch_size", 1: "ids_len"}, "text_hidden_states": {0: "batch_size", 1: "ids_len"}},
        metadata,
    )
    del input_ids
    gc.collect()

    if INPUT_IMAGE_DIM == 5:
        image_input = torch.zeros(
            (VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8
        )
    else:
        image_input = torch.zeros(
            (VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8
        )
    image_preprocess = LLM_IMAGE_PREPROCESS(
        IMAGE_RESIZE,
        model.model.visual,
        pos_embeds,
        vision_cos,
        vision_sin,
        vision_mask,
        dynamic_shape=DYNAMIC_IMAGE_SHAPE,
    )
    preprocess_axes = {
        "patches": {0: "vision_patch_count"},
        "pos": {1: "vision_patch_count"},
        "cos": {3: "vision_patch_count"},
        "sin": {3: "vision_patch_count"},
        "mask": {2: "vision_patch_count", 3: "vision_patch_count"},
    }
    if DYNAMIC_IMAGE_SHAPE:
        preprocess_axes["pixel_values"] = {0: "image_count"}
        preprocess_axes["pixel_values"].update(
            {3: "image_height", 4: "image_width"} if INPUT_IMAGE_DIM == 5 else {2: "image_height", 3: "image_width"}
        )
    _export_component(
        export_dir / MODEL_FILE_NAMES["image_preprocess"],
        image_preprocess,
        (image_input,),
        ["pixel_values"],
        ["patches", "pos", "cos", "sin", "mask"],
        preprocess_axes,
        metadata,
    )
    del image_preprocess, image_input
    gc.collect()

    patch_count = grid_h * grid_w * VISION_BATCH_SIZE
    patches = torch.zeros(
        (patch_count, 3, dimensions["vision_temporal_patch_size"], dimensions["vision_patch_size"], dimensions["vision_patch_size"]),
        dtype=torch.float32,
    )
    vision_axes = {
        "patches": {0: "vision_patch_count"},
        "pos": {1: "vision_patch_count"},
        "cos": {3: "vision_patch_count"},
        "sin": {3: "vision_patch_count"},
        "mask": {2: "vision_patch_count", 3: "vision_patch_count"},
        "vision_hidden_states": {1: "image_token_count"},
    }
    vision = LLM_VISION(model)
    _export_component(
        export_dir / MODEL_FILE_NAMES["vision"],
        vision,
        (patches, pos_embeds, vision_cos, vision_sin, vision_mask),
        ["patches", "pos", "cos", "sin", "mask"],
        ["vision_hidden_states"],
        vision_axes,
        metadata,
    )
    del vision, patches, pos_embeds, vision_cos, vision_sin, vision_mask
    gc.collect()

    text_hidden_states = torch.ones((1, len(token_ids), dimensions["hidden_size"]), dtype=torch.float32)
    image_hidden_states = torch.ones(
        (1, IMAGE_SEQLEN_PER_IMAGE * VISION_BATCH_SIZE, dimensions["hidden_size"]), dtype=torch.float32
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES["concat_image"],
        LLM_CONCAT_IMAGE(image_spans, IMAGE_SEQLEN_PER_IMAGE),
        (text_hidden_states, image_hidden_states),
        ["text_hidden_states", "vision_hidden_states"],
        ["concat_hidden_states"],
        {
            "text_hidden_states": {0: "batch_size", 1: "ids_len"},
            "vision_hidden_states": {0: "batch_size", 1: "image_token_count"},
            "concat_hidden_states": {0: "batch_size", 1: "ids_len"},
        },
        metadata,
    )
    del text_hidden_states, image_hidden_states
    gc.collect()

    ids_len = torch.tensor([trace_ids_len], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    trace_positions = torch.arange(trace_ids_len, dtype=torch.int64).view(1, -1).expand(3, -1)
    rotary_prefill = ROTARY_PREFILL(model, MAX_SEQ_LEN)
    _export_component(
        export_dir / MODEL_FILE_NAMES["rotary_image_prefill"],
        rotary_prefill,
        (trace_positions, ids_len, history_len),
        ["position_ids", "ids_len", "history_len"],
        ["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        {
            "position_ids": {1: "ids_len"},
            "rotary_cos": {1: "ids_len"},
            "rotary_sin": {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        },
        metadata,
    )
    rotary_decode = ROTARY_DECODE(model)
    _export_component(
        export_dir / MODEL_FILE_NAMES["rotary_image_decode"],
        rotary_decode,
        (trace_positions[:, :1], ids_len),
        ["position_ids", "kv_seq_len"],
        ["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        {"position_ids": {1: "ids_len"}},
        metadata,
    )
    del rotary_prefill, rotary_decode, trace_positions
    gc.collect()

    full_state_tensors = build_full_state_tensors(1)
    linear_state_tensors = build_linear_state_tensors(1)
    full_inputs, full_input_names, full_output_names, full_axes = get_full_kv_io(full_state_tensors)
    linear_inputs, linear_input_names, linear_output_names, linear_axes = get_linear_state_io(linear_state_tensors)
    state_inputs = full_inputs + linear_inputs
    state_input_names = full_input_names + linear_input_names
    state_output_names = full_output_names + linear_output_names
    if len(state_input_names) != NUM_MAIN_STATE_TENSORS:
        raise RuntimeError("OvisOCR2 exported state count differs from its metadata contract.")
    hidden_states = torch.ones((1, trace_ids_len, dimensions["hidden_size"]), dtype=torch.float32)
    rotary_cos = torch.zeros((1, trace_ids_len, 1, 1, ROTARY_DIM), dtype=torch.float32)
    rotary_sin = torch.zeros_like(rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, trace_ids_len, trace_ids_len), dtype=torch.float32)
    main = LLM_MAIN(model)
    kv_quantizer = main.quantizer
    rope_shift_rotary_module = main.llm.model.language_model.rotary_emb
    _export_component(
        export_dir / MODEL_FILE_NAMES["main"],
        main,
        tuple(state_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]),
        state_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
        state_output_names + ["logits"],
        {
            **full_axes,
            **linear_axes,
            "hidden_states": {0: "batch_size", 1: "ids_len"},
            "rotary_cos": {1: "ids_len"},
            "rotary_sin": {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            "logits": {0: "batch_size"},
        },
        metadata,
    )
    del main, model, state_inputs, hidden_states, rotary_cos, rotary_sin, attention_mask
    gc.collect()

    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([min(5, trace_ids_len)], dtype=torch.int64)
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_slice"],
        KV_SLICE(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM),
        tuple(full_inputs + [slice_start, slice_end]),
        full_input_names + ["slice_start", "slice_end"],
        full_output_names,
        full_axes,
        metadata,
    )
    del slice_start, slice_end

    split_at = torch.tensor([min(5, trace_ids_len)], dtype=torch.int64)
    split_prefix_names = [f"prefix_{name}" for name in full_output_names]
    split_window_names = [f"window_{name}" for name in full_output_names]
    split_axes = {name: dict(full_axes[name]) for name in full_input_names}
    for source_name, prefix_name, window_name in zip(
        full_output_names, split_prefix_names, split_window_names
    ):
        source_axes = full_axes[source_name]
        split_axes[prefix_name] = dict(source_axes)
        split_axes[window_name] = dict(source_axes)
        for axis in split_axes[prefix_name]:
            if axis != 0:
                split_axes[prefix_name][axis] = "prefix_len"
                split_axes[window_name][axis] = "window_len"
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_split2"],
        KV_SPLIT2(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM),
        tuple(full_inputs + [split_at]),
        full_input_names + ["split_at"],
        split_prefix_names + split_window_names,
        split_axes,
        metadata,
    )
    del split_at, split_prefix_names, split_window_names, split_axes

    cat_prefix_inputs, cat_prefix_names = [], []
    cat_suffix_inputs, cat_suffix_names = [], []
    cat_output_names, cat_axes = [], {}
    for name, dim in FULL_STATE_SPECS:
        tensor = full_state_tensors[name]
        for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
            prefix_name = f"in_prefix_{name}_{layer_index}"
            suffix_name = f"in_suffix_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            cat_prefix_inputs.append(tensor)
            cat_prefix_names.append(prefix_name)
            cat_suffix_inputs.append(tensor.clone())
            cat_suffix_names.append(suffix_name)
            cat_output_names.append(output_name)
            cat_axes[prefix_name] = {0: "batch_size", dim: "prefix_len"}
            cat_axes[suffix_name] = {0: "batch_size", dim: "suffix_len"}
            cat_axes[output_name] = {0: "batch_size", dim: "concat_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_concat"],
        KV_CONCAT(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM),
        tuple(cat_prefix_inputs + cat_suffix_inputs),
        cat_prefix_names + cat_suffix_names,
        cat_output_names,
        cat_axes,
        metadata,
    )
    del (
        cat_prefix_inputs,
        cat_prefix_names,
        cat_suffix_inputs,
        cat_suffix_names,
        cat_output_names,
        cat_axes,
    )

    def _seq4(tensor):
        shape = list(tensor.shape)
        shape[-1] = 4
        return torch.zeros(shape, dtype=tensor.dtype)

    def _rope_shift_key_io(specs):
        inputs, input_names, output_names, axes = [], [], [], {}
        for name, tensor in specs:
            sequence_axis = tensor.dim() - 1
            for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
                input_name = f"in_{name}_{layer_index}"
                output_name = f"out_{name}_{layer_index}"
                inputs.append(tensor)
                input_names.append(input_name)
                output_names.append(output_name)
                axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
                axes[output_name] = {0: "batch_size", sequence_axis: "history_len"}
        return inputs, input_names, output_names, axes

    rope_shift_amount = torch.tensor([min(5, MAX_SEQ_LEN)], dtype=torch.int64)
    if KV_QUANT_DTYPE in ("F16", "F32"):
        rope_specs = [("key", _seq4(full_state_tensors["key"]))]
        rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_key_io(
            rope_specs
        )
        rope_module = ROPE_SHIFT(
            NUM_FULL_ATTENTION_LAYERS,
            NUM_KEY_VALUE_HEADS,
            rope_shift_rotary_module,
            MAX_SEQ_LEN,
        )
    else:
        rope_specs = [
            ("key", _seq4(full_state_tensors["key"])),
            ("key_scale", _seq4(full_state_tensors["key_scale"])),
        ]
        if not _kv_sym:
            rope_specs.append(("key_bias", _seq4(full_state_tensors["key_bias"])))
        rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_key_io(
            rope_specs
        )
        rope_module = ROPE_SHIFT_QUANT(
            NUM_FULL_ATTENTION_LAYERS,
            NUM_KEY_VALUE_HEADS,
            rope_shift_rotary_module,
            MAX_SEQ_LEN,
            kv_quantizer,
            not _kv_sym,
        )
    _export_component(
        export_dir / MODEL_FILE_NAMES["rope_shift"],
        rope_module,
        tuple(rope_inputs + [rope_shift_amount]),
        rope_input_names + ["shift"],
        rope_output_names,
        rope_axes,
        metadata,
    )
    del (
        _seq4,
        _rope_shift_key_io,
        rope_shift_amount,
        rope_specs,
        rope_inputs,
        rope_input_names,
        rope_output_names,
        rope_axes,
        rope_module,
        kv_quantizer,
        rope_shift_rotary_module,
    )
    gc.collect()

    logits = torch.ones((1, dimensions["vocab_size"]), dtype=torch.float32)
    previous_ids = torch.zeros((1, 1), dtype=torch.int32)
    repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
    _export_component(
        export_dir / MODEL_FILE_NAMES["greedy"],
        GREEDY_SEARCH(),
        (logits,),
        ["logits"],
        ["max_logits_idx"],
        {"logits": {0: "batch_size"}, "max_logits_idx": {0: "batch_size"}},
        metadata,
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES["penalty_greedy"],
        PENALTY_GREEDY_SEARCH(),
        (logits, repetition_penalty, previous_ids),
        ["logits", "repetition_penalty", "previous_ids"],
        ["max_logits_idx", "save_id_out"],
        {
            "logits": {0: "batch_size"},
            "repetition_penalty": {0: "batch_size"},
            "previous_ids": {0: "batch_size", 1: "history_len"},
            "max_logits_idx": {0: "batch_size"},
            "save_id_out": {0: "batch_size", 1: "kv_seq_len"},
        },
        metadata,
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES["sampling"],
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            torch.ones((1,), dtype=torch.float32),
            torch.tensor(min(50, dimensions["vocab_size"]), dtype=torch.int64),
            torch.ones((1,), dtype=torch.float32),
            repetition_penalty,
            previous_ids,
        ),
        ["logits", "temperature", "top_k", "top_p", "repetition_penalty", "previous_ids"],
        ["sampled_id", "save_id_out"],
        {
            "logits": {0: "batch_size"},
            "temperature": {0: "batch_size"},
            "top_p": {0: "batch_size"},
            "repetition_penalty": {0: "batch_size"},
            "previous_ids": {0: "batch_size", 1: "history_len"},
            "sampled_id": {0: "batch_size"},
            "save_id_out": {0: "batch_size", 1: "kv_seq_len"},
        },
        metadata,
    )
    del logits, previous_ids, repetition_penalty, tokenizer, processor
    gc.collect()

    import Shared_Merged

    bundle = Shared_Merged.build_shared_merged_bundle(
        export_dir, model_file_names=MODEL_FILE_NAMES, delete_constituents=True
    )
    for path in bundle["graphs"].values():
        _stamp_metadata(path, metadata)
    _stamp_metadata(bundle["shared_model"], metadata)
    fused_count = _validate_export_bundle(export_dir, MODEL_FILE_NAMES)
    _cleanup_unreferenced_data(export_dir)
    _validate_export_bundle(export_dir, MODEL_FILE_NAMES)
    tokenizer_assets = _copy_tokenizer_assets(export_dir)
    _promote_export(export_dir)
    print(
        f"OvisOCR2 ONNX export completed: {EXPORT_DIR} "
        f"({NUM_MAIN_STATE_TENSORS} states, {fused_count} fused RMS nodes per merged graph, "
        f"{len(tokenizer_assets)} tokenizer assets)."
    )


def export_bundle():
    """Export the OvisOCR2 ONNX bundle."""
    return export_ovis()


def main():
    if not DO_EXPORT:
        print("DO_EXPORT is False; no ONNX files were written.")
        return
    export_bundle()
    subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "Inference_OvisOCR2_ONNX.py"),
            "--model-folder",
            str(EXPORT_DIR),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
