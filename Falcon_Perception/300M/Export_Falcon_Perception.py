"""Export Falcon Perception into a split, merged ONNX OCR bundle.

Falcon Perception is a detector checkpoint rather than a chat OCR model. Its
native image patch projector, hybrid image-prefix mask, temporal-plus-golden
spatial RoPE, and coordinate/size feedback are retained. Greedy,
penalty-greedy, and TopTok sampling remain the only token-selection strategies.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper
from transformers import AutoTokenizer, PreTrainedTokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets
import Shared_Merged


CHECKPOINT_DIR = Path.home() / "Downloads" / "Falcon-Perception-300M"
# Backward-compatible configuration alias.
DOWNLOAD_PATH = CHECKPOINT_DIR
EXPORT_DIR = SCRIPT_DIR / "Falcon_Perception_ONNX"
EXPORT_STAGING_DIR = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")

# Export controls
DO_EXPORT    = True                       # Set False to load and validate this module without writing ONNX files.
OPSET        = 20                         # ONNX opset used for every graph in the bundle.
MAX_SEQ_LEN  = 4096                       # Fixed maximum sequence length for temporal RoPE tables and masks.
COORDINATE_HISTORY_CAPACITY = MAX_SEQ_LEN # Fixed ONNX feedback-history capacity; covers every generated token.
COORDINATE_HISTORY_UNUSED_VALUE = -1.0    # Coordinates occupy [0, 1], so this sentinel cannot be selected.

# Image input and vision tracing
IMAGE_RESIZE        = (448, 448)          # Resize applied before patchification.
INPUT_IMAGE_SIZE    = (448, 448)          # Exported pixel_values height and width.
INPUT_IMAGE_DIM     = 5                   # pixel_values rank: 4=[B, C, H, W], 5=[B, 1, C, H, W].
VISION_BATCH_SIZE   = 1                   # Trace batch size for vision graph inputs.
DYNAMIC_IMAGE_SHAPE = False               # Reserved; the current bundle exports a fixed image size.

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                # F16 | F32 | Q8[_CUDA] | ROTARY_Q8[_CUDA] | ROTARY_Q4[_CUDA].
KV_QUANT_GROUP_SIZE = 64                  # Channel block width used by quantized cache formats.
COMPUTE_IN_F32      = False               # F16 cache only: False keeps KV attention computation in F16.

# KV quantization transforms and parameters
USE_HADAMARD           = False            # Apply a randomized Hadamard transform before grouped quantization.
HADAMARD_RANDOM_SEED   = 9527             # Deterministic sign pattern for the Hadamard transform.
USE_CLIP               = False            # Clip quantization blocks to CLIP_SIGMA standard deviations.
CLIP_SIGMA             = 3.0              # Standard-deviation limit used when clipping is enabled.
USE_SHUFFLE            = False            # Interleave channels across quantization groups.
USE_SYM                = True             # Use symmetric rather than scale-and-bias quantization.
USE_FLOAT16_SCALE_BIAS = True             # Store quantization scales and biases as float16.
USE_QDQ_FRIENDLY_ASYM  = False            # Disable residual bias correction for asymmetric QDQ compatibility.

# Quantization-oriented model reordering
REORDER_DOWNPROJ_FOR_QUANT   = True       # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True       # Record the requested vision MLP reorder policy in bundle metadata.
REORDER_KEY                  = "absmean"  # Channel statistic used to build the language MLP permutation.

# Detector feedback metadata consumed by inference
COORD_DEDUP_THRESHOLD = 0.01              # Coordinate-distance threshold for duplicate detection suppression.
MAX_COORD_ATTEMPTS    = 100               # Maximum coordinate decoding attempts per generated response.

MODEL_FILE_NAMES = {
    **Shared_Merged.default_model_file_names(),
    "coordinate_feedback": "LLM_FalconCoordinateFeedback.onnx",
    "size_feedback": "LLM_FalconSizeFeedback.onnx",
}
MODEL_FILE_NAME_METADATA = {
    f"model_file_name_{key}": value
    for key, value in MODEL_FILE_NAMES.items()
    if key in {
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
        "kv_slice",
        "kv_split2",
        "kv_concat",
        "coordinate_feedback",
        "size_feedback",
    }
}


SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
    "F16", "F32",
)


def normalize_kv_quant_settings(head_dim):
    """Validate the active Qwen-compatible KV cache configuration."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {
        "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
        "ROTARY_Q4", "ROTARY_Q4_CUDA",
    }
    rotary_kv = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes = []

    if KV_QUANT_DTYPE in rotary_kv and head_dim % 2:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")

    if KV_QUANT_DTYPE in quantized_kv:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError(
                f"KV_QUANT_GROUP_SIZE must be positive, got {KV_QUANT_GROUP_SIZE}."
            )
        if KV_QUANT_GROUP_SIZE > head_dim:
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim "
                f"({head_dim}); clamping to head_dim."
            )
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(
                divisor
                for divisor in range(1, KV_QUANT_GROUP_SIZE + 1)
                if head_dim % divisor == 0
            )
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide "
                f"head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}."
            )
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(
                f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim "
                f"({head_dim}); Q8 grouping collapses to per-head quantization."
            )

        if KV_QUANT_DTYPE in q8_kv and KV_QUANT_GROUP_SIZE == head_dim and (
            USE_HADAMARD or USE_SHUFFLE
        ):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when "
                "grouping collapses to one full-head block."
            )
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append(
            "[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32."
        )

    return notes


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, start, end, dim):
        start_value, end_value = int(start), int(end)
        return torch.tensor(
            [start_value, end_value - start_value, reference.shape[dim] - end_value],
            dtype=torch.int64,
        )

    @staticmethod
    def symbolic(graph, reference, start, end, dim):
        shape = graph.op("Shape", reference)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        return graph.op(
            "Concat",
            start,
            graph.op("Sub", end, start),
            graph.op("Sub", dim_size, end),
            axis_i=0,
        )


class SLICE_KEEP_MIDDLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, sizes, dim):
        start = int(sizes[0])
        end = start + int(sizes[1])
        index = [slice(None)] * values.dim()
        index[dim] = slice(start, end)
        return values[tuple(index)].clone()

    @staticmethod
    def symbolic(graph, values, sizes, dim):
        return graph.op("Split", values, sizes, axis_i=dim, outputs=3)[1]


def window_split_sizes(reference, start, end, dim):
    if dim < 0:
        dim += reference.dim()
    return WINDOW_SPLIT_SIZES.apply(reference, start, end, dim)


def slice_keep_middle(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SLICE_KEEP_MIDDLE.apply(values, sizes, dim)


class SPLIT_POINT_SIZES(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, split_at, dim):
        split_value = int(split_at)
        return torch.tensor(
            [split_value, reference.shape[dim] - split_value], dtype=torch.int64
        )

    @staticmethod
    def symbolic(graph, reference, split_at, dim):
        shape = graph.op("Shape", reference)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        return graph.op("Concat", split_at, graph.op("Sub", dim_size, split_at), axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, sizes, dim):
        split_value = int(sizes[0])
        prefix_index = [slice(None)] * values.dim()
        suffix_index = [slice(None)] * values.dim()
        prefix_index[dim] = slice(None, split_value)
        suffix_index[dim] = slice(split_value, None)
        return values[tuple(prefix_index)].clone(), values[tuple(suffix_index)].clone()

    @staticmethod
    def symbolic(graph, values, sizes, dim):
        return graph.op("Split", values, sizes, axis_i=dim, outputs=2)


def split_point_sizes(reference, split_at, dim):
    if dim < 0:
        dim += reference.dim()
    return SPLIT_POINT_SIZES.apply(reference, split_at, dim)


def split_prefix_suffix(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SPLIT_PREFIX_SUFFIX.apply(values, sizes, dim)


class KV_SLICE(torch.nn.Module):
    """Slice every KV state tensor along its declared sequence axis."""

    def __init__(self, sequence_axes):
        super().__init__()
        self.sequence_axes = tuple(sequence_axes)

    def forward(self, *all_inputs):
        states = all_inputs[:-2]
        if len(states) != len(self.sequence_axes):
            raise ValueError("KV_SLICE received an unexpected number of cache tensors.")
        start, end = all_inputs[-2:]
        sizes = window_split_sizes(states[0], start, end, self.sequence_axes[0])
        return tuple(
            slice_keep_middle(values, sizes, axis)
            for values, axis in zip(states, self.sequence_axes)
        )


class KV_SPLIT2(torch.nn.Module):
    """Split every KV state tensor into prefix and suffix cache windows."""

    def __init__(self, sequence_axes):
        super().__init__()
        self.sequence_axes = tuple(sequence_axes)

    def forward(self, *all_inputs):
        states = all_inputs[:-1]
        if len(states) != len(self.sequence_axes):
            raise ValueError("KV_SPLIT2 received an unexpected number of cache tensors.")
        split_at = all_inputs[-1]
        sizes = split_point_sizes(states[0], split_at, self.sequence_axes[0])
        prefixes, suffixes = zip(
            *(
                split_prefix_suffix(values, sizes, axis)
                for values, axis in zip(states, self.sequence_axes)
            )
        )
        return (*prefixes, *suffixes)


class KV_CONCAT(torch.nn.Module):
    """Concatenate matching KV state tensors along their sequence axes."""

    def __init__(self, sequence_axes):
        super().__init__()
        self.sequence_axes = tuple(sequence_axes)

    def forward(self, *all_inputs):
        count = len(self.sequence_axes)
        if len(all_inputs) != count * 2:
            raise ValueError("KV_CONCAT received an unexpected number of cache tensors.")
        prefixes, suffixes = all_inputs[:count], all_inputs[count:]
        return tuple(
            torch.cat([prefix, suffix], dim=axis)
            for prefix, suffix, axis in zip(prefixes, suffixes, self.sequence_axes)
        )


class GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class PENALTY_GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        adjusted_logits = torch.scatter(logits, 1, previous_ids.long(), adjusted)
        token_id = torch.argmax(adjusted_logits, dim=-1, keepdim=True).int()
        return token_id, torch.cat([previous_ids, token_id], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
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

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids.long(), adjusted)
        token_id = self._sample(scores, temperature, top_k, top_p)
        return token_id, torch.cat([previous_ids, token_id], dim=-1)


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Emit ORT's default-domain fused RMS normalization with FP32 accumulation."""

    @staticmethod
    def forward(ctx, values, scale, epsilon, axis):
        variance = values.float().square().mean(dim=axis, keepdim=True)
        normalized = values.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(graph, values, scale, epsilon, axis):
        output = graph.op(
            "SimplifiedLayerNormalization",
            values,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )
        return output.setType(values.type())


def simplified_layer_norm(values, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(values, scale, float(epsilon), axis)


class FalconKVQuantizer(torch.nn.Module):
    """Quantize channel-major Falcon KV tensors with Qwen-compatible formats."""

    def __init__(
        self,
        head_dim,
        num_heads,
        *,
        is_q4=False,
        is_rotary=False,
        is_cuda_storage=False,
        use_sym=False,
        use_hadamard=False,
        use_clip=False,
        clip_sigma=3.0,
        use_shuffle=False,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.head_dim_half = self.head_dim // 2
        self.is_q4 = bool(is_q4)
        self.is_rotary = bool(is_rotary)
        self.is_cuda_storage = bool(is_cuda_storage)
        self.use_sym = bool(use_sym)
        self.use_hadamard = bool(use_hadamard)
        self.use_clip = bool(use_clip)
        self.use_shuffle = bool(use_shuffle)
        self.use_residual_bias_correction = not use_sym and not USE_QDQ_FRIENDLY_ASYM
        self.grouped = self.is_q4 or (
            (self.use_hadamard or self.use_shuffle)
            and KV_QUANT_GROUP_SIZE < self.head_dim
        )
        self.group_size = KV_QUANT_GROUP_SIZE if self.grouped else self.head_dim
        self.group_count = self.head_dim // self.group_size

        if not self.grouped:
            self.use_hadamard = False
            self.use_shuffle = False

        if self.use_sym:
            self.signed_qmin = -8 if self.is_q4 else -128
            self.signed_qmax = 7 if self.is_q4 else 127
            self.qmax = float(self.signed_qmax)
        else:
            self.signed_qmin = None
            self.signed_qmax = None
            self.qmax = 15.0 if self.is_q4 else 255.0
        self.register_buffer(
            "inv_qmax", torch.tensor(1.0 / self.qmax, dtype=torch.float32)
        )

        if self.is_rotary:
            inverse_sqrt_two = 2.0 ** -0.5
            sine = torch.cat([
                torch.full((self.head_dim_half,), -inverse_sqrt_two),
                torch.full((self.head_dim_half,), inverse_sqrt_two),
            ])
            self.register_buffer("rotation_cos", torch.tensor(inverse_sqrt_two))
            self.register_buffer(
                "rotation_sin", sine.view(1, 1, 1, self.head_dim, 1)
            )

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.group_size)
            self.hadamard_pad = self.hadamard_size - self.group_size
            self.register_buffer(
                "hadamard_inv_sqrt",
                torch.tensor(self.hadamard_size ** -0.5, dtype=torch.float32),
            )
            generator = torch.Generator()
            generator.manual_seed(HADAMARD_RANDOM_SEED)
            signs = torch.randint(
                0, 2, (self.group_size,), generator=generator, dtype=torch.int64
            ).float()
            self.register_buffer("hadamard_sign", signs.mul_(2.0).sub_(1.0))
            self.hadamard_levels = []
            width = self.hadamard_size
            while width > 1:
                half = width // 2
                self.hadamard_levels.append((width, half))
                width = half

        if self.use_clip:
            self.register_buffer(
                "clip_sigma", torch.tensor(float(clip_sigma), dtype=torch.float32)
            )

        if self.use_shuffle:
            shuffle = torch.arange(self.head_dim).view(
                self.group_count, self.group_size
            ).transpose(0, 1).contiguous().view(-1)
            unshuffle = torch.empty_like(shuffle)
            unshuffle[shuffle] = torch.arange(self.head_dim)
            self.register_buffer("shuffle_index", shuffle.long())
            self.register_buffer("unshuffle_index", unshuffle.long())

    @staticmethod
    def _next_power_of_two(value):
        result = 1
        while result < value:
            result *= 2
        return result

    def _group(self, values):
        return values.reshape(
            values.shape[0], self.num_heads, 1,
            self.group_count, self.group_size, -1,
        )

    def _ungroup(self, values):
        return values.reshape(values.shape[0], self.num_heads, 1, self.head_dim, -1)

    def _flip_channels(self, values):
        pairs = values.reshape(
            values.shape[0], self.num_heads, 1, 2, self.head_dim_half, -1
        )
        return pairs.flip(-3).reshape(
            values.shape[0], self.num_heads, 1, self.head_dim, -1
        )

    def _rotate_channels(self, values):
        return values * self.rotation_cos + self._flip_channels(values) * self.rotation_sin

    def _inverse_rotate_channels(self, values):
        return values * self.rotation_cos - self._flip_channels(values) * self.rotation_sin

    def _apply_hadamard_last_dim(self, values, inverse=False):
        if not self.use_hadamard:
            return values
        if not inverse:
            values = values * self.hadamard_sign
        if self.hadamard_pad:
            values = F.pad(values, (0, self.hadamard_pad))
        for width, half in self.hadamard_levels:
            values = values.reshape(*values.shape[:-1], -1, width)
            even, odd = torch.split(values, [half, half], dim=-1)
            values = torch.cat([even + odd, even - odd], dim=-1)
            values = values.reshape(*values.shape[:-2], -1)
        values = values * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            values = values[..., :self.group_size]
        if inverse:
            values = values * self.hadamard_sign
        return values

    def _apply_channel_transform(self, values):
        if self.is_rotary:
            values = self._rotate_channels(values)
        if self.use_shuffle:
            values = values.index_select(3, self.shuffle_index)
        if self.use_hadamard:
            groups = self._group(values)
            groups = self._apply_hadamard_last_dim(groups.transpose(-1, -2)).transpose(-1, -2)
            values = self._ungroup(groups)
        return values

    def _invert_channel_transform(self, values):
        if self.use_hadamard:
            groups = self._group(values)
            groups = self._apply_hadamard_last_dim(
                groups.transpose(-1, -2), inverse=True
            ).transpose(-1, -2)
            values = self._ungroup(groups)
        if self.use_shuffle:
            values = values.index_select(3, self.unshuffle_index)
        if self.is_rotary:
            values = self._inverse_rotate_channels(values)
        return values

    def _clip_to_sigma(self, values, dim):
        mean = values.mean(dim=dim, keepdim=True)
        variance = (values - mean).square().mean(dim=dim, keepdim=True)
        bound = self.clip_sigma * variance.sqrt()
        return values.clamp(mean - bound, mean + bound)

    def _quantize_blocks(self, values, dim):
        if self.use_clip:
            values = self._clip_to_sigma(values, dim)
        epsilon = torch.finfo(values.dtype).eps
        if self.use_sym:
            scale = (
                values.abs().amax(dim=dim, keepdim=True) * self.inv_qmax
            ).clamp_min(epsilon)
            packed = torch.round(values / scale).clamp(
                self.signed_qmin, self.signed_qmax
            ).to(torch.int32)
            if self.is_q4:
                packed = torch.remainder(packed, 16).to(torch.uint8)
            elif self.is_cuda_storage:
                packed = torch.remainder(packed, 256).to(torch.uint8)
            else:
                packed = packed.to(torch.int8)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return packed, scale, None

        lower, upper = torch.aminmax(values, dim=dim, keepdim=True)
        scale = ((upper - lower) * self.inv_qmax).clamp_min(epsilon)
        packed = torch.round((values - lower) / scale).clamp(0, self.qmax)
        if self.use_residual_bias_correction:
            lower = lower + (values - (packed * scale + lower)).mean(dim=dim, keepdim=True)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            lower = lower.half()
        return packed.to(torch.uint8), scale, lower

    def _pack_q4(self, values):
        pairs = values.reshape(
            values.shape[0], self.num_heads, 1, self.head_dim_half, 2, -1
        )
        low, high = torch.unbind(pairs, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def _unpack_q4(self, values):
        low = values % 16
        high = values // 16
        return torch.stack([low, high], dim=-2).reshape(
            values.shape[0], self.num_heads, 1, self.head_dim, -1
        )

    def _pack_cuda(self, values):
        storage_width = self.head_dim // (2 if self.is_q4 else 1)
        groups = values.reshape(
            values.shape[0], self.num_heads, 1, storage_width // 4, 4, -1
        ).to(torch.int32)
        value0, value1, value2, value3 = torch.unbind(groups, dim=-2)
        return value0 + value1 * 256 + value2 * 65536 + (value3 - 128) * 16777216

    def _unpack_cuda(self, values):
        storage_width = self.head_dim // (2 if self.is_q4 else 1)
        remainder3 = values % 16777216
        value3 = (values - remainder3) // 16777216 + 128
        value2 = remainder3 // 65536
        remainder2 = remainder3 % 65536
        value1 = remainder2 // 256
        value0 = remainder2 % 256
        return torch.stack([value0, value1, value2, value3], dim=-2).reshape(
            values.shape[0], self.num_heads, 1, storage_width, -1
        )

    @staticmethod
    def _decode_signed_q4(values):
        values = values.to(torch.int16)
        return torch.remainder(values + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8(values):
        if values.dtype == torch.int8:
            return values.to(torch.int16)
        values = values.to(torch.int16)
        return torch.remainder(values + 128, 256) - 128

    def quantize(self, values):
        values = self._apply_channel_transform(values.float())
        if self.grouped:
            packed, scale, bias = self._quantize_blocks(self._group(values), dim=-2)
            packed = self._ungroup(packed)
        else:
            packed, scale, bias = self._quantize_blocks(values, dim=-2)
        if self.is_q4:
            packed = self._pack_q4(packed)
        if self.is_cuda_storage:
            packed = self._pack_cuda(packed)
        return packed, scale, bias

    def dequantize(self, packed, scale, bias):
        if scale.dtype != torch.float32:
            scale = scale.float()
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.float()
        if self.is_cuda_storage:
            packed = self._unpack_cuda(packed)
        if self.is_q4:
            values = self._unpack_q4(packed)
            if self.use_sym:
                values = self._decode_signed_q4(values)
        else:
            values = self._decode_signed_q8(packed) if self.use_sym else packed
        values = values.float()
        if self.grouped:
            values = self._group(values)
            values = values * scale if self.use_sym else values * scale + bias
            values = self._ungroup(values)
        else:
            values = values * scale if self.use_sym else values * scale + bias
        return self._invert_channel_transform(values)


class ONNX_MASK_RESHAPE(torch.autograd.Function):
    """Export a stable [1, 1, 1, query, key] reshape without Unsqueeze chains."""

    @staticmethod
    def forward(ctx, values):
        return values.reshape(1, 1, 1, *values.shape)

    @staticmethod
    def symbolic(graph, values):
        shape = graph.op("Shape", values)
        query = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
            axis_i=0,
        )
        key = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
            axis_i=0,
        )
        target_shape = graph.op(
            "Concat",
            graph.op("Constant", value_t=torch.tensor([1, 1, 1], dtype=torch.int64)),
            query,
            key,
            axis_i=0,
        )
        return graph.op("Reshape", values, target_shape)


def onnx_mask_reshape(values):
    return ONNX_MASK_RESHAPE.apply(values)


def _install_runtime_stubs():
    """Load the checkpoint's custom source on CPU without Triton/FlexAttention."""
    triton_pkg = types.ModuleType("triton")
    triton_pkg.__path__ = []
    triton_lang = types.ModuleType("triton.language")
    triton_pkg.jit = lambda function: function
    triton_pkg.cdiv = lambda left, right: (left + right - 1) // right
    triton_pkg.language = triton_lang
    for name in ("dtype", "program_id", "arange", "load", "store", "where"):
        setattr(triton_lang, name, lambda *args, **kwargs: None)
    triton_lang.constexpr = lambda value: value
    sys.modules["triton"] = triton_pkg
    sys.modules["triton.language"] = triton_lang
    sys.modules["triton.backends"] = types.ModuleType("triton.backends")
    sys.modules["triton.backends.compiler"] = types.ModuleType("triton.backends.compiler")
    pycocotools_pkg = types.ModuleType("pycocotools")
    pycocotools_mask = types.ModuleType("pycocotools.mask")
    pycocotools_pkg.mask = pycocotools_mask
    sys.modules.setdefault("pycocotools", pycocotools_pkg)
    sys.modules.setdefault("pycocotools.mask", pycocotools_mask)
    torch.compile = lambda function=None, *args, **kwargs: (
        function if function is not None else lambda candidate: candidate
    )

    import torch.nn.attention.flex_attention as flex_mod

    class _StubBlockMask:
        BLOCK_SIZE = (128, 128)

        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, index):
            return self

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return None

    flex_mod.BlockMask = _StubBlockMask
    flex_mod.AuxRequest = lambda **kwargs: None
    flex_mod.create_block_mask = lambda *args, **kwargs: _StubBlockMask()
    flex_mod.and_masks = lambda *functions: lambda batch, head, query, key: True
    flex_mod.or_masks = lambda *functions: lambda batch, head, query, key: True
    if not hasattr(flex_mod, "_mask_mod_signature"):
        flex_mod._mask_mod_signature = type(None)


def load_falcon(model_path: Path):
    _install_runtime_stubs()
    parent = str(model_path.parent)
    package = model_path.name
    if parent not in sys.path:
        sys.path.insert(0, parent)
    (model_path / "__init__.py").touch(exist_ok=True)
    config_module = importlib.import_module(f"{package}.configuration_falcon_perception")
    model_module = importlib.import_module(f"{package}.modeling_falcon_perception")
    config = config_module.FalconPerceptionConfig.from_pretrained(str(model_path))
    model = model_module.FalconPerceptionForSegmentation.from_pretrained(
        str(model_path), config=config, torch_dtype=torch.float32, device_map={"": "cpu"}
    ).eval()
    model._weights_fused = False
    model._ensure_device_buffers()
    model._is_compiled = True
    tokenizer = load_falcon_tokenizer(model_path)
    return model, tokenizer, config


def load_falcon_tokenizer(model_path: Path):
    """Load Falcon's tokenizer even when its legacy backend class is absent."""
    try:
        return AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        config = json.loads((model_path / "tokenizer_config.json").read_text(encoding="utf-8"))
        return PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"),
            eos_token=config.get("eos_token"),
            pad_token=config.get("pad_token"),
        )


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


def reorder_language_mlp(model, key, reorder_channels=True):
    """Deinterleave Falcon MLP gates and optionally sort paired channels."""
    if getattr(model, "_falcon_downproj_reordered", False):
        raise RuntimeError("Falcon language MLP preparation was requested twice.")
    pair_count = 0
    max_error = 0.0
    generator = torch.Generator().manual_seed(9527)
    with torch.no_grad():
        for layer in model.layers.values():
            producer = layer.feed_forward.w13
            consumer = layer.feed_forward.w2
            hidden = int(consumer.in_features)
            if producer.out_features != hidden * 2 or producer.bias is not None:
                raise ValueError("Falcon gated MLP must be a bias-free interleaved [gate, up] producer pair.")
            permutation = (
                torch.argsort(_channel_statistic(consumer.weight.data, key))
                if reorder_channels
                else torch.arange(hidden, device=consumer.weight.device)
            )
            if torch.unique(permutation).numel() != hidden:
                raise RuntimeError("Falcon language MLP permutation is not bijective.")
            producer_weight = producer.weight.data.clone()
            consumer_weight = consumer.weight.data.clone()
            probe = torch.randn(
                2,
                producer.in_features,
                generator=generator,
                dtype=producer_weight.dtype,
            )
            original_packed = F.linear(probe, producer_weight)
            original_output = F.linear(
                F.relu(original_packed[:, 0::2]).square() * original_packed[:, 1::2],
                consumer_weight,
            )
            producer.weight.data.copy_(torch.cat([
                producer_weight[0::2][permutation],
                producer_weight[1::2][permutation],
            ], dim=0))
            consumer.weight.data.copy_(consumer_weight[:, permutation])
            reordered_packed = F.linear(probe, producer.weight.data)
            reordered_output = F.linear(
                F.relu(reordered_packed[:, :hidden]).square() * reordered_packed[:, hidden:],
                consumer.weight.data,
            )
            error = float((original_output - reordered_output).abs().max())
            if not torch.allclose(original_output, reordered_output, rtol=1e-5, atol=5e-4):
                raise RuntimeError(
                    "Falcon gated-MLP reordering changed the producer/consumer function "
                    f"beyond FP32 accumulation tolerance (max_abs={error})."
                )
            max_error = max(max_error, error)
            pair_count += 1
    model._falcon_downproj_reordered = True
    return pair_count, max_error


def _temporal_tables(model, max_seq_len):
    freqs = model.freqs_cis[:max_seq_len].to(torch.complex64)
    angles = torch.angle(freqs).float()
    cosine = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sine = torch.cat([-torch.sin(angles), torch.sin(angles)], dim=-1)
    return cosine, sine


def _golden_tables(model, grid_h, grid_w):
    num_heads, num_frequencies, _ = model.freqs_cis_golden.shape
    height = torch.linspace(-float(grid_h / grid_w) ** 0.5, float(grid_h / grid_w) ** 0.5, grid_h)
    width = torch.linspace(-float(grid_w / grid_h) ** 0.5, float(grid_w / grid_h) ** 0.5, grid_w)
    width_grid, height_grid = torch.meshgrid(width, height, indexing="xy")
    positions = torch.stack([height_grid.flatten(), width_grid.flatten()], dim=-1)
    theta = torch.einsum("sp,hfp->shf", positions, model.freqs_cis_golden.float())
    cosine = torch.cat([theta.cos(), theta.cos()], dim=-1)
    sine = torch.cat([-theta.sin(), theta.sin()], dim=-1)
    return cosine, sine


def _make_image_token_ids(config, tokenizer, query, image_tokens):
    image_token = tokenizer.convert_ids_to_tokens(config.img_id)
    prompt = f"<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"
    chunks = [tokenizer.encode(part) for part in prompt.split(image_token)]
    if len(chunks) != 2:
        raise RuntimeError("Falcon prompt template must contain exactly one image placeholder.")
    prefix = chunks[0]
    if getattr(tokenizer, "bos_token_id", None) is not None and prefix and prefix[0] == tokenizer.bos_token_id:
        prefix = prefix[:1] + prefix[1:]
    image_block = [
        config.image_cls_token_id,
        config.image_reg_1_token_id,
        config.image_reg_2_token_id,
        config.image_reg_3_token_id,
        config.image_reg_4_token_id,
        *([config.img_id] * image_tokens),
        config.img_end_id,
    ]
    token_ids = prefix + image_block + chunks[1]
    positions = [index for index, token in enumerate(token_ids) if token == config.img_id]
    if len(positions) != image_tokens or positions != list(range(positions[0], positions[0] + image_tokens)):
        raise RuntimeError("Falcon image token span is not contiguous or does not match the vision grid.")
    return token_ids, positions[0], positions[-1] + 1


class LLM_EMBED(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.tok_embeddings.float()

    def forward(self, input_ids):
        return self.embedding(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Resize raw RGB input and patchify it for Falcon's affine image projector."""

    def __init__(self, config, image_resize):
        super().__init__()
        self.target_h, self.target_w = (int(value) for value in image_resize)
        self.patch_size = int(config.spatial_patch_size)
        self.grid_h = self.target_h // self.patch_size
        self.grid_w = self.target_w // self.patch_size

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        pixel_values = pixel_values.float()
        if pixel_values.shape[-2] != self.target_h or pixel_values.shape[-1] != self.target_w:
            pixel_values = F.interpolate(pixel_values, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        values = pixel_values / 127.5 - 1.0
        patches = values.reshape(-1, 3, self.grid_h, self.patch_size, self.grid_w, self.patch_size)
        return patches.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.grid_h * self.grid_w, self.patch_size * self.patch_size * 3)


class LLM_VISION(torch.nn.Module):
    """Run only the native Falcon image patch projector."""

    def __init__(self, model):
        super().__init__()
        self.projector = model.img_projector.float()

    def forward(self, image_patches):
        return self.projector(image_patches)


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the verified contiguous Falcon img_id span with projected image features."""

    def __init__(self, image_start, image_end):
        super().__init__()
        self.image_start = int(image_start)
        self.image_end = int(image_end)

    def forward(self, text_hidden_states, vision_hidden_states):
        return torch.cat([
            text_hidden_states[:, :self.image_start],
            vision_hidden_states,
            text_hidden_states[:, self.image_end:],
        ], dim=1)


class FALCON_COORDINATE_FEEDBACK(torch.nn.Module):
    """Run Falcon's coordinate head and emit native Fourier feedback in ONNX."""

    def __init__(
        self,
        logit_decoder: torch.nn.Module,
        embed_weight: torch.Tensor,
        transform_weight: torch.Tensor,
        coord_bins: int,
        max_attempts: int,
        dedup_threshold: float,
        history_capacity: int,
        history_unused_value: float,
    ):
        super().__init__()
        self.logit_decoder = logit_decoder.float()
        embed = embed_weight.detach().float()
        transform = transform_weight.detach().float()
        if embed.ndim != 2 or embed.shape[1] != 2:
            raise ValueError("Falcon coordinate feedback embedding must accept two values.")
        if transform.ndim != 2 or transform.shape[1] != embed.shape[0] * 2:
            raise ValueError("Falcon coordinate feedback transform has an invalid feature width.")
        self.coord_bins = int(coord_bins)
        self.candidate_count = min(int(max_attempts), self.coord_bins)
        self.history_capacity = int(history_capacity)
        self.history_unused_value = float(history_unused_value)
        if (
            self.coord_bins < 2
            or self.candidate_count < 1
            or self.history_capacity < 1
            or self.history_unused_value >= 0.0
        ):
            raise ValueError("Falcon coordinate feedback requires valid bin, candidate, and history capacities.")
        self.register_buffer(
            "phase_weight", (embed.t() * math.tau).contiguous()
        )
        self.register_buffer("output_weight", transform.t().contiguous())
        self.register_buffer(
            "bin_inverse", torch.tensor(1.0 / float(self.coord_bins - 1))
        )
        self.register_buffer("dedup_threshold", torch.tensor(float(dedup_threshold)))
        self.register_buffer(
            "history_unused_threshold", torch.tensor(self.history_unused_value * 0.5)
        )
        self.register_buffer(
            "fallback_index", torch.tensor(self.candidate_count - 1, dtype=torch.int64)
        )
        self.register_buffer("use_feedback", torch.ones((1, 1, 1), dtype=torch.bool))

    def _fourier(self, values: torch.Tensor) -> torch.Tensor:
        angles = torch.matmul(values, self.phase_weight)
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return torch.matmul(features, self.output_weight).unsqueeze(0)

    def forward(
        self,
        final_hidden_states: torch.Tensor,
        coordinate_history: torch.Tensor,
    ):
        coord_logits = self.logit_decoder(final_hidden_states).reshape(
            final_hidden_states.shape[0], 2, self.coord_bins
        )
        scores = coord_logits.squeeze(0)
        ordered_bins = torch.topk(
            scores, self.candidate_count, dim=-1, largest=True, sorted=True
        ).indices.t()
        candidates = ordered_bins.to(scores.dtype) * self.bin_inverse
        differences = (candidates.unsqueeze(1) - coordinate_history.unsqueeze(0)).abs()
        same_pair = torch.amax(differences, dim=-1) < self.dedup_threshold
        duplicate = torch.amax(
            same_pair.to(scores.dtype), dim=1
        ) > 0.0
        valid_scores = 1.0 - duplicate.to(scores.dtype)
        first_valid = torch.argmax(valid_scores, dim=0)
        candidate_index = torch.where(
            torch.amax(valid_scores) > 0.0, first_valid, self.fallback_index
        )
        coordinate = torch.index_select(candidates, 0, candidate_index.reshape(1))
        feedback = self._fourier(coordinate)
        write_index = torch.argmax(
            (coordinate_history[:, 0] < self.history_unused_threshold).to(torch.int64),
            dim=0,
        )
        next_history = torch.scatter(
            coordinate_history,
            0,
            write_index.reshape(1, 1).expand_as(coordinate),
            coordinate,
        )
        return feedback, self.use_feedback, next_history, coordinate


class FALCON_SIZE_FEEDBACK(torch.nn.Module):
    """Run Falcon's size head and emit native Fourier feedback in ONNX."""

    def __init__(
        self,
        logit_decoder: torch.nn.Module,
        embed_weight: torch.Tensor,
        transform_weight: torch.Tensor,
        size_bins: int,
    ):
        super().__init__()
        self.logit_decoder = logit_decoder.float()
        embed = embed_weight.detach().float()
        transform = transform_weight.detach().float()
        if embed.ndim != 2 or embed.shape[1] != 2:
            raise ValueError("Falcon size feedback embedding must accept two values.")
        if transform.ndim != 2 or transform.shape[1] != embed.shape[0] * 2:
            raise ValueError("Falcon size feedback transform has an invalid feature width.")
        self.size_bins = int(size_bins)
        if self.size_bins < 2:
            raise ValueError("Falcon size feedback requires at least two bins.")
        self.register_buffer(
            "phase_weight", (embed.t() * math.tau).contiguous()
        )
        self.register_buffer("output_weight", transform.t().contiguous())
        self.register_buffer(
            "size_exponent_scale",
            torch.tensor(math.log(self.size_bins) / float(self.size_bins - 1)),
        )
        self.register_buffer(
            "size_exponent_bias", torch.tensor(-math.log(self.size_bins))
        )
        self.register_buffer("use_feedback", torch.ones((1, 1, 1), dtype=torch.bool))

    def _fourier(self, values: torch.Tensor) -> torch.Tensor:
        angles = torch.matmul(values, self.phase_weight)
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return torch.matmul(features, self.output_weight).unsqueeze(0)

    def forward(self, final_hidden_states: torch.Tensor):
        size_logits = self.logit_decoder(final_hidden_states).reshape(
            final_hidden_states.shape[0], 2, self.size_bins
        )
        selected_bins = torch.argmax(size_logits.squeeze(0), dim=-1)
        size = torch.exp(
            selected_bins.to(size_logits.dtype) * self.size_exponent_scale
            + self.size_exponent_bias
        ).reshape(1, 2)
        return self._fourier(size), self.use_feedback, size


class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Build Falcon temporal and golden image RoPE plus the hybrid image-prefix mask."""

    def __init__(
        self,
        temporal_cos,
        temporal_sin,
        spatial_cos,
        spatial_sin,
        image_start,
        image_end,
        num_kv_heads,
        num_groups,
        max_seq_len,
    ):
        super().__init__()
        self.image_start = int(image_start)
        self.image_end = int(image_end)
        self.image_prefix_start = self.image_start - 5
        self.image_non_increasing_start = self.image_start - 4
        self.image_non_increasing_count = self.image_end - self.image_non_increasing_start + 1
        self.num_kv_heads = int(num_kv_heads)
        self.num_groups = int(num_groups)
        self.register_buffer("temporal_cos", temporal_cos.float().unsqueeze(0), persistent=False)
        self.register_buffer("temporal_sin", temporal_sin.float().unsqueeze(0), persistent=False)
        self.register_buffer("spatial_cos", spatial_cos.float(), persistent=False)
        self.register_buffer("spatial_sin", spatial_sin.float(), persistent=False)
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.int64), persistent=False)
        self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("negative", torch.tensor(-float("inf"), dtype=torch.float32), persistent=False)

    def _temporal_position_ids(self, raw_positions):
        image_anchor = torch.full_like(raw_positions, self.image_prefix_start)
        within_image = (raw_positions >= self.image_non_increasing_start) & (raw_positions <= self.image_end)
        after_image = raw_positions > self.image_end
        shifted = raw_positions - self.image_non_increasing_count
        return torch.where(within_image, image_anchor, torch.where(after_image, shifted, raw_positions))

    def forward(self, ids_len, history_len):
        sequence_length = ids_len.reshape(())
        history = history_len.reshape(())
        total_length = sequence_length + history
        raw_positions = torch.arange(sequence_length, dtype=torch.int64) + history
        temporal_positions = self._temporal_position_ids(raw_positions)
        temporal_cos = self.temporal_cos.index_select(1, temporal_positions).squeeze(0)
        temporal_sin = self.temporal_sin.index_select(1, temporal_positions).squeeze(0)
        temporal_cos = temporal_cos.view(-1, 1, 1, temporal_cos.shape[-1]).expand(-1, self.num_kv_heads, self.num_groups, -1)
        temporal_sin = temporal_sin.view(-1, 1, 1, temporal_sin.shape[-1]).expand(-1, self.num_kv_heads, self.num_groups, -1)
        spatial_cos = self.spatial_cos.index_select(0, raw_positions).view(-1, self.num_kv_heads, self.num_groups, temporal_cos.shape[-1])
        spatial_sin = self.spatial_sin.index_select(0, raw_positions).view(-1, self.num_kv_heads, self.num_groups, temporal_sin.shape[-1])
        rotary_cos = torch.cat([temporal_cos, spatial_cos], dim=-1).unsqueeze(0)
        rotary_sin = torch.cat([temporal_sin, spatial_sin], dim=-1).unsqueeze(0)

        query_positions = raw_positions
        key_positions = torch.arange(total_length, dtype=torch.int64)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        query_image = (query_positions >= self.image_prefix_start) & (query_positions < self.image_end)
        key_image = (key_positions >= self.image_prefix_start) & (key_positions < self.image_end)
        allowed = causal | (query_image.unsqueeze(1) & key_image.unsqueeze(0))
        mask = onnx_mask_reshape(torch.where(allowed, self.zero, self.negative))
        return rotary_cos, rotary_sin, mask, ids_len + history_len


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    """Provide identity spatial RoPE and next temporal RoPE for one decode token."""

    def __init__(self, temporal_cos, temporal_sin, image_start, image_end, num_kv_heads, num_groups):
        super().__init__()
        head_half = temporal_cos.shape[-1]
        self.image_non_increasing_count = int(image_end) - (int(image_start) - 4) + 1
        self.num_kv_heads = int(num_kv_heads)
        self.num_groups = int(num_groups)
        self.register_buffer("temporal_cos", temporal_cos.float().unsqueeze(0), persistent=False)
        self.register_buffer("temporal_sin", temporal_sin.float().unsqueeze(0), persistent=False)
        self.register_buffer("spatial_cos", torch.ones((1, head_half), dtype=torch.float32), persistent=False)
        self.register_buffer("spatial_sin", torch.zeros((1, head_half), dtype=torch.float32), persistent=False)

    def forward(self, kv_seq_len):
        position = kv_seq_len.reshape(-1) - self.image_non_increasing_count
        temporal_cos = self.temporal_cos.index_select(1, position).squeeze(0)
        temporal_sin = self.temporal_sin.index_select(1, position).squeeze(0)
        temporal_cos = temporal_cos.view(-1, 1, 1, temporal_cos.shape[-1]).expand(-1, self.num_kv_heads, self.num_groups, -1)
        temporal_sin = temporal_sin.view(-1, 1, 1, temporal_sin.shape[-1]).expand(-1, self.num_kv_heads, self.num_groups, -1)
        spatial_cos = self.spatial_cos.view(1, 1, 1, -1).expand(temporal_cos.shape[0], self.num_kv_heads, self.num_groups, -1)
        spatial_sin = self.spatial_sin.view(1, 1, 1, -1).expand(temporal_sin.shape[0], self.num_kv_heads, self.num_groups, -1)
        rotary_cos = torch.cat([temporal_cos, spatial_cos], dim=-1).unsqueeze(0)
        rotary_sin = torch.cat([temporal_sin, spatial_sin], dim=-1).unsqueeze(0)
        return rotary_cos, rotary_sin, kv_seq_len + 1


class LLM_MAIN(torch.nn.Module):
    """Falcon's 22-layer decoder with split rotary/mask inputs and compact GQA cache."""

    def __init__(self, model, config):
        super().__init__()
        self.layers = model.layers
        self.output = model.output.float()
        self.num_layers = int(config.n_layers)
        self.num_heads = int(config.n_heads)
        self.num_kv_heads = int(config.n_kv_heads)
        self.num_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(config.head_dim)
        self.hidden_size = int(config.dim)
        self.ffn_size = int(config.ffn_dim)
        self.layer_norm_eps = float(torch.finfo(torch.float32).eps)
        self.final_norm_eps = float(config.norm_eps)
        self.register_buffer("norm_scale", torch.ones((self.hidden_size,), dtype=torch.float32), persistent=False)
        self.register_buffer("qk_scale", torch.full((self.head_dim,), self.head_dim ** -0.25, dtype=torch.float32), persistent=False)
        final_scale = model.norm.weight.detach().float()
        self.register_buffer("final_scale", final_scale, persistent=False)
        self.register_buffer("sinks", torch.stack([layer.attention.sinks.detach().float() for layer in model.layers.values()]).view(self.num_layers, 1, self.num_kv_heads, self.num_groups, 1), persistent=False)

        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.kv_q8 = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
        self.kv_rotary_q8 = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_quantized = self.kv_q8 or self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_symmetric = USE_SYM and self.kv_quantized
        self.compute_in_f32 = COMPUTE_IN_F32
        self.kv_state_names = ["key", "value"]
        if self.kv_quantized:
            self.kv_state_names.append("key_scale")
            if not self.kv_symmetric:
                self.kv_state_names.append("key_bias")
            self.kv_state_names.append("value_scale")
            if not self.kv_symmetric:
                self.kv_state_names.append("value_bias")
        self.kv_state_offsets = {
            name: index * self.num_layers
            for index, name in enumerate(self.kv_state_names)
        }
        self.kv_blocks_per_layer = len(self.kv_state_names)

        if self.kv_quantized:
            quantizer_kwargs = {
                "is_q4": self.kv_rotary_q4,
                "is_rotary": self.kv_rotary_q8 or self.kv_rotary_q4,
                "is_cuda_storage": KV_QUANT_DTYPE.endswith("_CUDA"),
                "use_sym": self.kv_symmetric,
                "use_hadamard": USE_HADAMARD,
                "use_clip": USE_CLIP,
                "clip_sigma": CLIP_SIGMA,
                "use_shuffle": USE_SHUFFLE,
            }
            # Falcon caches keys after GQA expansion because its image RoPE differs
            # for each query head; values remain compact key/value-head tensors.
            self.key_quantizer = FalconKVQuantizer(
                self.head_dim, self.num_heads, **quantizer_kwargs
            ).eval()
            self.value_quantizer = FalconKVQuantizer(
                self.head_dim, self.num_kv_heads, **quantizer_kwargs
            ).eval()
        self._permute_qk_rope_channels()

    def _permute_qk_rope_channels(self):
        """Convert Falcon's interleaved complex pairs to the flip-RoPE half split."""
        if self.head_dim % 4:
            raise ValueError("Falcon flip-RoPE export requires head_dim divisible by four.")
        half_permutation = torch.cat((
            torch.arange(0, self.head_dim // 2, 2),
            torch.arange(1, self.head_dim // 2, 2),
        ))
        head_permutation = torch.cat((
            half_permutation,
            self.head_dim // 2 + half_permutation,
        ))
        qk_width = (self.num_heads + self.num_kv_heads) * self.head_dim
        row_permutation = torch.arange(qk_width + self.num_kv_heads * self.head_dim)
        for head_index in range(self.num_heads):
            start = head_index * self.head_dim
            row_permutation[start:start + self.head_dim] = start + head_permutation
        key_offset = self.num_heads * self.head_dim
        for head_index in range(self.num_kv_heads):
            start = key_offset + head_index * self.head_dim
            row_permutation[start:start + self.head_dim] = start + head_permutation
        with torch.no_grad():
            for layer in self.layers.values():
                attention = layer.attention
                if getattr(attention, "_falcon_rope_channels_permuted", False):
                    raise RuntimeError("Falcon Q/K RoPE channel permutation was requested twice.")
                if attention.wqkv.bias is not None:
                    raise ValueError("Falcon Q/K RoPE export expects a bias-free fused wqkv projection.")
                attention.wqkv.weight.data.copy_(
                    attention.wqkv.weight.data.index_select(0, row_permutation)
                )
                attention._falcon_rope_channels_permuted = True

    def _rotate_half(self, values, groups):
        values = values.reshape(
            values.shape[0],
            values.shape[1],
            self.num_kv_heads,
            groups,
            2,
            2,
            self.head_dim // 4,
        )
        return values.flip(-2).reshape(values.shape[0], values.shape[1], self.num_kv_heads, groups, self.head_dim)

    def _state(self, inputs, name, layer_index):
        return inputs[self.kv_state_offsets[name] + layer_index]

    def _store_value_parameter(self, values):
        if self.value_quantizer.grouped:
            return values.permute(0, 1, 2, 5, 3, 4)
        return values.transpose(-1, -2)

    def _load_value_parameter(self, values):
        if self.value_quantizer.grouped:
            return values.permute(0, 1, 2, 4, 5, 3)
        return values.transpose(-1, -2)

    def forward(self, *inputs):
        state_count = self.num_layers * self.kv_blocks_per_layer
        hidden_states = inputs[state_count]
        rotary_cos = inputs[state_count + 1]
        rotary_sin = inputs[state_count + 2]
        attention_mask = inputs[state_count + 3]
        batch_size = hidden_states.shape[0]
        attention_mask_f16 = (
            attention_mask.half()
            if self.kv_f16 and not self.compute_in_f32
            else None
        )
        saved_states = {name: [] for name in self.kv_state_names}
        for layer_index, layer in enumerate(self.layers.values()):
            residual = hidden_states
            normalized = simplified_layer_norm(hidden_states, self.norm_scale, self.layer_norm_eps)
            qkv = layer.attention.wqkv(normalized)
            qk, value = torch.split(
                qkv,
                [(self.num_heads + self.num_kv_heads) * self.head_dim, self.num_kv_heads * self.head_dim],
                dim=-1,
            )
            qk = qk.reshape(batch_size, -1, self.num_heads + self.num_kv_heads, self.head_dim)
            qk = simplified_layer_norm(qk, self.qk_scale, self.layer_norm_eps)
            query, key = torch.split(qk, [self.num_heads, self.num_kv_heads], dim=2)
            query = query.reshape(batch_size, -1, self.num_kv_heads, self.num_groups, self.head_dim)
            key = key.reshape(batch_size, -1, self.num_kv_heads, 1, self.head_dim)
            value = value.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
            query = query * rotary_cos + self._rotate_half(query, self.num_groups) * rotary_sin
            key = key * rotary_cos + self._rotate_half(key, 1) * rotary_sin

            key = key.permute(0, 2, 3, 4, 1).reshape(
                batch_size, self.num_heads, 1, self.head_dim, -1
            )
            value = value.permute(0, 2, 1, 3).unsqueeze(2)

            if self.kv_quantized:
                packed_key, key_scale, key_bias = self.key_quantizer.quantize(key)
                packed_value, value_scale, value_bias = self.value_quantizer.quantize(
                    value.transpose(-1, -2)
                )
                packed_value = packed_value.transpose(-1, -2)
                value_scale = self._store_value_parameter(value_scale)
                value_bias = self._store_value_parameter(value_bias) if value_bias is not None else None

                key_cache = torch.cat(
                    [self._state(inputs, "key", layer_index), packed_key], dim=-1
                )
                value_cache = torch.cat(
                    [self._state(inputs, "value", layer_index), packed_value], dim=-2
                )
                key_scale_cache = torch.cat(
                    [self._state(inputs, "key_scale", layer_index), key_scale], dim=-1
                )
                value_axis = -3 if self.value_quantizer.grouped else -2
                value_scale_cache = torch.cat(
                    [self._state(inputs, "value_scale", layer_index), value_scale], dim=value_axis
                )
                saved_states["key"].append(key_cache)
                saved_states["value"].append(value_cache)
                saved_states["key_scale"].append(key_scale_cache)
                saved_states["value_scale"].append(value_scale_cache)

                if self.kv_symmetric:
                    key_bias_cache = None
                    value_bias_cache = None
                else:
                    key_bias_cache = torch.cat(
                        [self._state(inputs, "key_bias", layer_index), key_bias], dim=-1
                    )
                    value_bias_cache = torch.cat(
                        [self._state(inputs, "value_bias", layer_index), value_bias], dim=value_axis
                    )
                    saved_states["key_bias"].append(key_bias_cache)
                    saved_states["value_bias"].append(value_bias_cache)

                attention_key = self.key_quantizer.dequantize(
                    key_cache, key_scale_cache, key_bias_cache
                )
                attention_value = self.value_quantizer.dequantize(
                    value_cache.transpose(-1, -2),
                    self._load_value_parameter(value_scale_cache),
                    self._load_value_parameter(value_bias_cache) if value_bias_cache is not None else None,
                ).transpose(-1, -2)
            else:
                if self.kv_f16:
                    key = key.half()
                    value = value.half()
                    if not self.compute_in_f32:
                        query = query.half()
                key_cache = torch.cat(
                    [self._state(inputs, "key", layer_index), key], dim=-1
                )
                value_cache = torch.cat(
                    [self._state(inputs, "value", layer_index), value], dim=-2
                )
                saved_states["key"].append(key_cache)
                saved_states["value"].append(value_cache)
                if self.kv_f16 and self.compute_in_f32:
                    attention_key = key_cache.float()
                    attention_value = value_cache.float()
                else:
                    attention_key = key_cache
                    attention_value = value_cache

            scores = torch.matmul(
                query.permute(0, 2, 3, 1, 4),
                attention_key.reshape(
                    batch_size, self.num_kv_heads, self.num_groups, self.head_dim, -1
                ),
            )
            scores = scores + (attention_mask_f16 if attention_mask_f16 is not None else attention_mask)
            probabilities = torch.softmax(scores, dim=-1)
            attention = torch.matmul(probabilities, attention_value)
            lse = torch.logsumexp(scores, dim=-1)
            sink = self.sinks[layer_index]
            if self.kv_f16 and not self.compute_in_f32:
                sink = sink.half()
            attention = attention * torch.sigmoid(lse - sink).unsqueeze(-1)
            if self.kv_f16 and not self.compute_in_f32:
                attention = attention.float()
            attention = attention.permute(0, 3, 1, 2, 4).reshape(
                batch_size, -1, self.num_heads * self.head_dim
            )
            hidden_states = residual + layer.attention.wo(attention)
            residual = hidden_states
            normalized = simplified_layer_norm(hidden_states, self.norm_scale, self.layer_norm_eps)
            gate_up = layer.feed_forward.w13(normalized)
            gate, up = torch.split(gate_up, self.ffn_size, dim=-1)
            hidden_states = residual + layer.feed_forward.w2(F.relu(gate).square() * up)
        final = simplified_layer_norm(hidden_states[:, -1], self.final_scale, self.final_norm_eps)
        logits = self.output(final)
        state_outputs = []
        for name in self.kv_state_names:
            state_outputs.extend(saved_states[name])
        return (*state_outputs, logits, final)


def _build_kv_layout(config, batch_size=1, history_len=0):
    """Build Falcon's state tensors for the active Qwen-compatible KV mode."""
    rotary_modes = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_modes = {"Q8", "Q8_CUDA"}
    quantized = KV_QUANT_DTYPE in rotary_modes | q8_modes
    rotary_q4 = KV_QUANT_DTYPE in {"ROTARY_Q4", "ROTARY_Q4_CUDA"}
    grouped_q8 = (
        KV_QUANT_DTYPE in q8_modes | {"ROTARY_Q8", "ROTARY_Q8_CUDA"}
        and (USE_HADAMARD or USE_SHUFFLE)
        and KV_QUANT_GROUP_SIZE < config.head_dim
    )
    grouped = rotary_q4 or grouped_q8
    symmetric = USE_SYM and quantized
    specs = [("key", 4), ("value", 3)]
    if quantized:
        specs.append(("key_scale", 5 if grouped else 4))
        if not symmetric:
            specs.append(("key_bias", 5 if grouped else 4))
        specs.append(("value_scale", 3))
        if not symmetric:
            specs.append(("value_bias", 3))

    if KV_QUANT_DTYPE == "F16":
        cache_dtype = torch.float16
    elif KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"}:
        cache_dtype = torch.int32
    elif symmetric and not rotary_q4:
        cache_dtype = torch.int8
    elif quantized:
        cache_dtype = torch.uint8
    else:
        cache_dtype = torch.float32

    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"}:
        key_width = value_width = config.head_dim // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4":
        key_width = value_width = config.head_dim // 2
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        key_width = value_width = config.head_dim // 8
    else:
        key_width = value_width = config.head_dim

    tensors = {
        "key": torch.zeros(
            (batch_size, config.n_heads, 1, key_width, history_len),
            dtype=cache_dtype,
        ),
        "value": torch.zeros(
            (batch_size, config.n_kv_heads, 1, history_len, value_width),
            dtype=cache_dtype,
        ),
    }
    scale_dtype = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32
    group_count = config.head_dim // KV_QUANT_GROUP_SIZE if grouped else 1
    if quantized:
        if grouped:
            key_parameter_shape = (
                batch_size, config.n_heads, 1, group_count, 1, history_len
            )
            value_parameter_shape = (
                batch_size, config.n_kv_heads, 1, history_len, group_count, 1
            )
        else:
            key_parameter_shape = (batch_size, config.n_heads, 1, 1, history_len)
            value_parameter_shape = (batch_size, config.n_kv_heads, 1, history_len, 1)
        tensors["key_scale"] = torch.ones(key_parameter_shape, dtype=scale_dtype)
        tensors["value_scale"] = torch.ones(value_parameter_shape, dtype=scale_dtype)
        if not symmetric:
            tensors["key_bias"] = torch.ones(key_parameter_shape, dtype=scale_dtype)
            tensors["value_bias"] = torch.ones(value_parameter_shape, dtype=scale_dtype)

    state_order = [
        f"{name}_{layer_index}"
        for name, _ in specs
        for layer_index in range(config.n_layers)
    ]
    facts = {
        "kv_cache_quantization": KV_QUANT_DTYPE,
        "kv_cache_tensor_order": ",".join(state_order),
        "kv_cache_blocks_per_layer": str(len(specs)),
        "kv_cache_key_layout": "batch,query_heads,one,key_width,sequence",
        "kv_cache_value_layout": "batch,key_value_heads,one,sequence,value_width",
        "kv_cache_key_sequence_axis": "4",
        "kv_cache_value_sequence_axis": "3",
        "kv_cache_key_storage_width": str(key_width),
        "kv_cache_value_storage_width": str(value_width),
        "kv_cache_quantized": str(int(quantized)),
        "kv_cache_symmetric": str(int(symmetric)),
        "kv_cache_grouped_6d": str(int(grouped)),
        "kv_cache_group_size": str(KV_QUANT_GROUP_SIZE if quantized else 0),
        "kv_cache_group_count": str(group_count if quantized else 0),
        "kv_cache_storage_dtype": str(cache_dtype).replace("torch.", ""),
        "kv_cache_scale_bias_dtype": (
            str(scale_dtype).replace("torch.", "") if quantized else "none"
        ),
    }
    return specs, tensors, facts


def _kv_io(kv_specs, kv_tensors, num_layers):
    inputs, input_names, output_names, axes = [], [], [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
            axes[output_name] = {0: "batch_size", sequence_axis: "kv_seq_len"}
    return inputs, input_names, output_names, axes


def _export_kv_helpers(export_dir, config, kv_specs, metadata):
    """Export cache slice/split/concat helpers for the active state layout."""
    _, helper_tensors, _ = _build_kv_layout(config, history_len=4)
    inputs, input_names, output_names, _ = _kv_io(
        kv_specs, helper_tensors, config.n_layers
    )
    state_specs = [
        (name, sequence_axis)
        for name, sequence_axis in kv_specs
        for _ in range(config.n_layers)
    ]
    sequence_axes = [sequence_axis for _, sequence_axis in state_specs]

    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([1], dtype=torch.int64)
    slice_axes = {}
    for input_name, output_name, (_, sequence_axis) in zip(
        input_names, output_names, state_specs
    ):
        slice_axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
        slice_axes[output_name] = {0: "batch_size", sequence_axis: "sliced_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_slice"],
        KV_SLICE(sequence_axes),
        tuple(inputs + [slice_start, slice_end]),
        input_names + ["slice_start", "slice_end"],
        output_names,
        slice_axes,
        metadata,
    )

    split_at = torch.tensor([1], dtype=torch.int64)
    prefix_names = [f"prefix_{name}" for name in output_names]
    suffix_names = [f"suffix_{name}" for name in output_names]
    split_axes = {}
    for input_name, prefix_name, suffix_name, (_, sequence_axis) in zip(
        input_names, prefix_names, suffix_names, state_specs
    ):
        split_axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
        split_axes[prefix_name] = {0: "batch_size", sequence_axis: "prefix_len"}
        split_axes[suffix_name] = {0: "batch_size", sequence_axis: "suffix_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_split2"],
        KV_SPLIT2(sequence_axes),
        tuple(inputs + [split_at]),
        input_names + ["split_at"],
        prefix_names + suffix_names,
        split_axes,
        metadata,
    )

    prefix_inputs, suffix_inputs = [], []
    prefix_names, suffix_names, concat_names = [], [], []
    concat_axes = {}
    for values, input_name, (name, sequence_axis) in zip(
        inputs, input_names, state_specs
    ):
        state_name = input_name.removeprefix("in_")
        prefix_name = f"in_prefix_{state_name}"
        suffix_name = f"in_suffix_{state_name}"
        output_name = f"out_{state_name}"
        prefix_inputs.append(values)
        suffix_inputs.append(values.clone())
        prefix_names.append(prefix_name)
        suffix_names.append(suffix_name)
        concat_names.append(output_name)
        concat_axes[prefix_name] = {0: "batch_size", sequence_axis: "prefix_len"}
        concat_axes[suffix_name] = {0: "batch_size", sequence_axis: "suffix_len"}
        concat_axes[output_name] = {0: "batch_size", sequence_axis: "concat_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_concat"],
        KV_CONCAT(sequence_axes),
        tuple(prefix_inputs + suffix_inputs),
        prefix_names + suffix_names,
        concat_names,
        concat_axes,
        metadata,
    )


def _stamp_metadata(path, metadata):
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(path, module, args, input_names, output_names, dynamic_axes, metadata, custom_opsets=None):
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        custom_opsets=custom_opsets,
    )
    _stamp_metadata(path, metadata)


def _prepare_export_staging():
    if EXPORT_STAGING_DIR.exists():
        if not EXPORT_STAGING_DIR.is_dir():
            raise NotADirectoryError(EXPORT_STAGING_DIR)
        shutil.rmtree(EXPORT_STAGING_DIR)
    EXPORT_STAGING_DIR.mkdir(parents=True)
    return EXPORT_STAGING_DIR


def _promote_export(staging_dir):
    previous = EXPORT_DIR.with_name(EXPORT_DIR.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if EXPORT_DIR.exists():
        EXPORT_DIR.rename(previous)
    staging_dir.rename(EXPORT_DIR)


def _cleanup_unreferenced_data(folder):
    referenced = set()
    for path in folder.glob("*.onnx"):
        model = onnx.load(str(path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                external = {item.key: item.value for item in initializer.external_data}
                if external.get("location"):
                    referenced.add(Path(external["location"]).name)
    for path in folder.iterdir():
        if path.is_file() and path.suffix != ".onnx" and path.name not in referenced:
            path.unlink()


def _metadata(
    config,
    tokenizer,
    image_tokens,
    image_start,
    image_end,
    language_pairs,
    reorder_error,
    kv_facts,
):
    end_of_query_id = 0
    try:
        candidate = tokenizer.convert_tokens_to_ids("<|end_of_query|>")
        if candidate is not None and candidate != tokenizer.unk_token_id:
            end_of_query_id = int(candidate)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        pass
    stop_token_ids = [int(config.eos_id)]
    if end_of_query_id and end_of_query_id not in stop_token_ids:
        stop_token_ids.append(end_of_query_id)
    return {
        "native_llm_metadata_version": "1",
        "producer": "Export_Falcon_Perception.py",
        "model_type": "falcon_perception",
        "max_seq_len": str(MAX_SEQ_LEN),
        "input_image_size": ",".join(str(value) for value in INPUT_IMAGE_SIZE),
        "input_image_dim": str(INPUT_IMAGE_DIM),
        "vision_batch_size": str(VISION_BATCH_SIZE),
        "image_token_id": str(config.img_id),
        "image_token_length": str(image_tokens),
        "image_start": str(image_start),
        "image_end": str(image_end),
        "image_cls_token_id": str(config.image_cls_token_id),
        "image_reg_1_token_id": str(config.image_reg_1_token_id),
        "image_reg_2_token_id": str(config.image_reg_2_token_id),
        "image_reg_3_token_id": str(config.image_reg_3_token_id),
        "image_reg_4_token_id": str(config.image_reg_4_token_id),
        "image_end_token_id": str(config.img_end_id),
        "coord_token_id": str(config.coord_token_id),
        "size_token_id": str(config.size_token_id),
        "coord_num_bins": str(int(config.coord_out_dim) // 2),
        "size_num_bins": str(int(config.size_out_dim) // 2),
        "coord_dedup_threshold": repr(COORD_DEDUP_THRESHOLD),
        "max_coord_attempts": str(MAX_COORD_ATTEMPTS),
        "falcon_feedback_mode": "fourier_coord_size",
        "falcon_feedback_backend": "onnx_graphs",
        "falcon_feedback_input": "final_hidden_states",
        "falcon_coordinate_history_layout": "capacity,xy,sentinel",
        "falcon_coordinate_history_capacity": str(COORDINATE_HISTORY_CAPACITY),
        "falcon_coordinate_history_unused_value": repr(COORDINATE_HISTORY_UNUSED_VALUE),
        "stop_token_ids": ",".join(str(token_id) for token_id in stop_token_ids),
        "eos_token_ids": str(config.eos_id),
        "num_layers": str(config.n_layers),
        "num_attention_heads": str(config.n_heads),
        "num_key_value_heads": str(config.n_kv_heads),
        "head_dim": str(config.head_dim),
        "hidden_size": str(config.dim),
        "vocab_size": str(config.vocab_size),
        "kv_num_tensors": str(len(kv_facts["kv_cache_tensor_order"].split(","))),
        "kv_blocks_per_layer": kv_facts["kv_cache_blocks_per_layer"],
        "kv_quant_dtype": KV_QUANT_DTYPE,
        "kv_quant_group_size": str(KV_QUANT_GROUP_SIZE if kv_facts["kv_cache_quantized"] == "1" else 0),
        "kv_symmetric": kv_facts["kv_cache_symmetric"],
        "kv_grouped_6d": kv_facts["kv_cache_grouped_6d"],
        "kv_cache_elem_type": kv_facts["kv_cache_storage_dtype"],
        "kv_scale_bias_elem_type": kv_facts["kv_cache_scale_bias_dtype"],
        "compute_in_f32": str(int(COMPUTE_IN_F32)),
        "kv_helper_graphs": "slice,split2,concat",
        "rope_type": "falcon_temporal_plus_golden_spatial",
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": "0",
        "vision_reorder_mlp_requested": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reordered_vision_pairs": "0",
        "reorder_key": REORDER_KEY,
        "reordered_language_pairs": str(language_pairs),
        "reorder_equivalence_max_abs": repr(reorder_error),
        "embed_lm_head_tied": "0",
        "embed_lm_head_equal": "0",
        "fused_simplified_layer_norms": str(config.n_layers * 3 + 1),
        **kv_facts,
        **MODEL_FILE_NAME_METADATA,
    }


@torch.inference_mode()
def export_falcon():
    if not DOWNLOAD_PATH.is_dir():
        raise FileNotFoundError(DOWNLOAD_PATH)
    if INPUT_IMAGE_DIM not in {4, 5}:
        raise ValueError("INPUT_IMAGE_DIM must be 4 or 5.")
    if COORDINATE_HISTORY_CAPACITY < MAX_SEQ_LEN:
        raise ValueError("COORDINATE_HISTORY_CAPACITY must cover MAX_SEQ_LEN.")
    if COORDINATE_HISTORY_UNUSED_VALUE >= 0.0:
        raise ValueError("COORDINATE_HISTORY_UNUSED_VALUE must be outside the [0, 1] coordinate range.")
    model, tokenizer, config = load_falcon(DOWNLOAD_PATH)
    if config.n_heads % config.n_kv_heads:
        raise ValueError("Falcon num_heads must divide num_kv_heads.")
    if config.temporal_patch_size != 1:
        raise ValueError("This Falcon exporter was verified for temporal_patch_size=1.")
    for note in normalize_kv_quant_settings(config.head_dim):
        print(note)
    grid_h = IMAGE_RESIZE[0] // config.spatial_patch_size
    grid_w = IMAGE_RESIZE[1] // config.spatial_patch_size
    image_tokens = grid_h * grid_w
    token_ids, image_start, image_end = _make_image_token_ids(config, tokenizer, "", image_tokens)
    if image_end - image_start != image_tokens:
        raise RuntimeError("Falcon image span does not match projected feature count.")
    prepared_language_pairs, reorder_error = reorder_language_mlp(
        model,
        REORDER_KEY,
        reorder_channels=REORDER_DOWNPROJ_FOR_QUANT,
    )
    language_pairs = prepared_language_pairs if REORDER_DOWNPROJ_FOR_QUANT else 0
    kv_specs, kv_tensors, kv_facts = _build_kv_layout(config)
    metadata = _metadata(
        config,
        tokenizer,
        image_tokens,
        image_start,
        image_end,
        language_pairs,
        reorder_error,
        kv_facts,
    )
    staging = _prepare_export_staging()

    _export_component(
        staging / MODEL_FILE_NAMES["metadata"],
        METADATA_CARRIER(),
        (torch.zeros((1,), dtype=torch.int32),),
        ["metadata_marker"], ["metadata_marker_out"], None, metadata,
    )

    trace_ids = torch.tensor([token_ids[: min(10, len(token_ids))]], dtype=torch.int32)
    _export_component(
        staging / MODEL_FILE_NAMES["embed"], LLM_EMBED(model), (trace_ids,),
        ["input_ids"], ["text_hidden_states"],
        {"input_ids": {0: "batch_size", 1: "ids_len"}, "text_hidden_states": {0: "batch_size", 1: "ids_len"}}, metadata,
    )

    image_shape = (VISION_BATCH_SIZE, 1, 3, *INPUT_IMAGE_SIZE) if INPUT_IMAGE_DIM == 5 else (VISION_BATCH_SIZE, 3, *INPUT_IMAGE_SIZE)
    image_input = torch.zeros(image_shape, dtype=torch.uint8)
    _export_component(
        staging / MODEL_FILE_NAMES["image_preprocess"], LLM_IMAGE_PREPROCESS(config, IMAGE_RESIZE), (image_input,),
        ["pixel_values"], ["image_patches"],
        {"image_patches": {0: "batch_size"}}, metadata,
    )
    patches = torch.zeros((VISION_BATCH_SIZE, image_tokens, config.spatial_patch_size ** 2 * config.channel_size), dtype=torch.float32)
    _export_component(
        staging / MODEL_FILE_NAMES["vision"], LLM_VISION(model), (patches,),
        ["image_patches"], ["vision_hidden_states"],
        {"image_patches": {0: "batch_size"}, "vision_hidden_states": {0: "batch_size", 1: "image_token_count"}}, metadata,
    )
    text_hidden = torch.ones((1, len(token_ids), config.dim), dtype=torch.float32)
    vision_hidden = torch.ones((1, image_tokens, config.dim), dtype=torch.float32)
    _export_component(
        staging / MODEL_FILE_NAMES["concat_image"], LLM_CONCAT_IMAGE(image_start, image_end), (text_hidden, vision_hidden),
        ["text_hidden_states", "vision_hidden_states"], ["concat_hidden_states"],
        {"text_hidden_states": {0: "batch_size", 1: "ids_len"}, "vision_hidden_states": {0: "batch_size", 1: "image_token_count"}, "concat_hidden_states": {0: "batch_size", 1: "ids_len"}}, metadata,
    )

    temporal_cos, temporal_sin = _temporal_tables(model, MAX_SEQ_LEN)
    spatial_image_cos, spatial_image_sin = _golden_tables(model, grid_h, grid_w)
    head_half = config.head_dim // 2
    spatial_cos = torch.ones((MAX_SEQ_LEN, config.n_heads, head_half), dtype=torch.float32)
    spatial_sin = torch.zeros_like(spatial_cos)
    spatial_cos[image_start:image_end] = spatial_image_cos
    spatial_sin[image_start:image_end] = spatial_image_sin
    ids_len = torch.tensor([min(10, len(token_ids))], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    kv_seq_len = torch.tensor([len(token_ids)], dtype=torch.int64)
    _export_component(
        staging / MODEL_FILE_NAMES["rotary_image_prefill"],
        ROTARY_IMAGE_PREFILL(
            temporal_cos,
            temporal_sin,
            spatial_cos,
            spatial_sin,
            image_start,
            image_end,
            config.n_kv_heads,
            config.n_heads // config.n_kv_heads,
            MAX_SEQ_LEN,
        ),
        (ids_len, history_len), ["ids_len", "history_len"], ["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        {"rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"}, "attention_mask": {3: "ids_len", 4: "kv_seq_len"}}, metadata,
    )
    _export_component(
        staging / MODEL_FILE_NAMES["rotary_image_decode"],
        ROTARY_IMAGE_DECODE(
            temporal_cos,
            temporal_sin,
            image_start,
            image_end,
            config.n_kv_heads,
            config.n_heads // config.n_kv_heads,
        ),
        (kv_seq_len,), ["kv_seq_len"], ["rotary_cos", "rotary_sin", "kv_seq_len_next"], None, metadata,
    )

    kv_inputs, kv_input_names, kv_output_names, kv_axes = _kv_io(
        kv_specs, kv_tensors, config.n_layers
    )
    main_length = min(10, len(token_ids))
    hidden = torch.ones((1, main_length, config.dim), dtype=torch.float32)
    rotary_cos = torch.ones(
        (1, main_length, config.n_kv_heads, config.n_heads // config.n_kv_heads, config.head_dim),
        dtype=torch.float32,
    )
    rotary_sin = torch.zeros_like(rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, main_length, main_length), dtype=torch.float32)
    _export_component(
        staging / MODEL_FILE_NAMES["main"], LLM_MAIN(model, config),
        tuple(kv_inputs + [hidden, rotary_cos, rotary_sin, attention_mask]),
        kv_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
        kv_output_names + ["logits", "final_hidden_states"],
        {**kv_axes, "hidden_states": {0: "batch_size", 1: "ids_len"}, "rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"}, "attention_mask": {3: "ids_len", 4: "kv_seq_len"}, "logits": {0: "batch_size"}, "final_hidden_states": {0: "batch_size"}},
        metadata,
    )
    _export_kv_helpers(staging, config, kv_specs, metadata)

    logits = torch.ones((1, config.vocab_size), dtype=torch.float32)
    previous_ids = torch.zeros((1, 1), dtype=torch.int32)
    repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
    _export_component(staging / MODEL_FILE_NAMES["greedy"], GREEDY_SEARCH(), (logits,), ["logits"], ["max_logits_idx"], {"logits": {0: "batch_size"}, "max_logits_idx": {0: "batch_size"}}, metadata)
    _export_component(
        staging / MODEL_FILE_NAMES["penalty_greedy"], PENALTY_GREEDY_SEARCH(), (logits, repetition_penalty, previous_ids),
        ["logits", "repetition_penalty", "previous_ids"], ["max_logits_idx", "save_id_out"],
        {"logits": {0: "batch_size"}, "repetition_penalty": {0: "batch_size"}, "previous_ids": {0: "batch_size", 1: "history_len"}, "max_logits_idx": {0: "batch_size"}, "save_id_out": {0: "batch_size", 1: "kv_seq_len"}}, metadata,
    )
    _export_component(
        staging / MODEL_FILE_NAMES["sampling"], TOPK_TOPP_SAMPLING(),
        (logits, torch.ones((1,), dtype=torch.float32), torch.tensor(min(50, config.vocab_size), dtype=torch.int64), torch.ones((1,), dtype=torch.float32), repetition_penalty, previous_ids),
        ["logits", "temperature", "top_k", "top_p", "repetition_penalty", "previous_ids"], ["sampled_id", "save_id_out"],
        {"logits": {0: "batch_size"}, "temperature": {0: "batch_size"}, "top_p": {0: "batch_size"}, "repetition_penalty": {0: "batch_size"}, "previous_ids": {0: "batch_size", 1: "history_len"}, "sampled_id": {0: "batch_size"}, "save_id_out": {0: "batch_size", 1: "kv_seq_len"}}, metadata,
    )

    coord_bins = int(config.coord_out_dim) // 2
    coordinate_feedback = FALCON_COORDINATE_FEEDBACK(
        model.coord_decoder,
        model.coord_encoder.embed.weight,
        model.coord_encoder.transform.weight,
        coord_bins,
        MAX_COORD_ATTEMPTS,
        COORD_DEDUP_THRESHOLD,
        COORDINATE_HISTORY_CAPACITY,
        COORDINATE_HISTORY_UNUSED_VALUE,
    )
    _export_component(
        staging / MODEL_FILE_NAMES["coordinate_feedback"],
        coordinate_feedback,
        (
            torch.zeros((1, config.dim), dtype=torch.float32),
            torch.full(
                (COORDINATE_HISTORY_CAPACITY, 2),
                COORDINATE_HISTORY_UNUSED_VALUE,
                dtype=torch.float32,
            ),
        ),
        ["final_hidden_states", "coordinate_history"],
        [
            "feedback_hidden_states",
            "falcon_use_feedback",
            "next_coordinate_history",
            "coordinate_values",
        ],
        None,
        metadata,
    )
    size_bins = int(config.size_out_dim) // 2
    size_feedback = FALCON_SIZE_FEEDBACK(
        model.size_decoder,
        model.size_encoder.embed.weight,
        model.size_encoder.transform.weight,
        size_bins,
    )
    _export_component(
        staging / MODEL_FILE_NAMES["size_feedback"],
        size_feedback,
        (torch.zeros((1, config.dim), dtype=torch.float32),),
        ["final_hidden_states"],
        ["feedback_hidden_states", "falcon_use_feedback", "size_values"],
        None,
        metadata,
    )

    bundle = Shared_Merged.build_shared_merged_bundle(staging, model_file_names=MODEL_FILE_NAMES, delete_constituents=True)
    for path in bundle["graphs"].values():
        _stamp_metadata(path, metadata)
    _stamp_metadata(bundle["shared_model"], metadata)
    Shared_Merged.validate_onnx_path(staging / MODEL_FILE_NAMES["image_prefill_greedy"])
    Shared_Merged.validate_onnx_path(staging / MODEL_FILE_NAMES["coordinate_feedback"])
    Shared_Merged.validate_onnx_path(staging / MODEL_FILE_NAMES["size_feedback"])
    _cleanup_unreferenced_data(staging)
    copy_tokenizer_assets(DOWNLOAD_PATH, staging)
    _promote_export(staging)
    print(f"Falcon Perception ONNX export completed: {EXPORT_DIR}")


def export_bundle():
    """Export the Falcon Perception OCR ONNX bundle."""
    return export_falcon()


def main():
    if not DO_EXPORT:
        print("DO_EXPORT is False; no ONNX files were written.")
        return
    export_bundle()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_Falcon_Perception_ONNX.py"),
            "--model-folder",
            str(EXPORT_DIR),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
