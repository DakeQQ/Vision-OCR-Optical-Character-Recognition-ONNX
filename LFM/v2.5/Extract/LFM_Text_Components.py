"""LFM2.5-VL hybrid language export components.

The VL checkpoint's language model is LFM2 with six full-attention layers and
ten short-convolution layers. This local module owns its recurrent-state and
ORT-fused RMSNorm export behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


SUPPORTED_KV_QUANT_DTYPES = frozenset((
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
    "F16", "F32",
))
QUANTIZED_KV_DTYPES = frozenset((
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
))
ROTARY_KV_DTYPES = frozenset((
    "ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
))
Q8_KV_DTYPES = frozenset(("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"))


@dataclass(frozen=True)
class KVCacheSettings:
    """Qwen-compatible storage and accuracy settings for LFM attention KV."""

    quant_dtype: str = "F16"
    quant_group_size: int = 128
    compute_in_f32: bool = False
    use_hadamard: bool = False
    hadamard_random_seed: int = 9527
    use_clip: bool = False
    clip_sigma: float = 3.0
    use_shuffle: bool = False
    use_sym: bool = True
    use_float16_scale_bias: bool = True
    use_qdq_friendly_asym: bool = False

    @property
    def is_quantized(self):
        return self.quant_dtype in QUANTIZED_KV_DTYPES

    @property
    def is_rotary(self):
        return self.quant_dtype in ROTARY_KV_DTYPES

    @property
    def is_rotary_q4(self):
        return self.quant_dtype in {"ROTARY_Q4", "ROTARY_Q4_CUDA"}

    @property
    def is_cuda_packed(self):
        return self.quant_dtype.endswith("_CUDA")

    def uses_grouped_q8(self, head_dim):
        return (
            self.quant_dtype in Q8_KV_DTYPES
            and (self.use_hadamard or self.use_shuffle)
            and self.quant_group_size < head_dim
        )

    def uses_grouped_layout(self, head_dim):
        return self.is_rotary_q4 or self.uses_grouped_q8(head_dim)


def normalize_kv_cache_settings(settings, head_dim):
    """Validate and normalize one Qwen-compatible KV cache configuration."""
    normalized = replace(settings, quant_dtype=str(settings.quant_dtype).upper())
    head_dim = int(head_dim)
    if normalized.quant_dtype not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {settings.quant_dtype!r}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}.")
    if normalized.quant_dtype in ROTARY_KV_DTYPES and head_dim % 2:
        raise ValueError(f"{normalized.quant_dtype} requires an even head_dim, got {head_dim}.")
    if normalized.quant_dtype in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4:
        raise ValueError(f"{normalized.quant_dtype} requires head_dim divisible by 4, got {head_dim}.")
    if normalized.quant_dtype == "ROTARY_Q4_CUDA" and head_dim % 8:
        raise ValueError(f"{normalized.quant_dtype} requires head_dim divisible by 8, got {head_dim}.")

    notes = []
    if normalized.is_quantized:
        if normalized.quant_group_size <= 0:
            raise ValueError(
                f"KV_QUANT_GROUP_SIZE must be positive, got {normalized.quant_group_size}."
            )
        if normalized.quant_group_size > head_dim:
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({normalized.quant_group_size}) > head_dim ({head_dim}); "
                "clamping to head_dim."
            )
            normalized = replace(normalized, quant_group_size=head_dim)
        elif head_dim % normalized.quant_group_size:
            original = normalized.quant_group_size
            group_size = max(
                divisor
                for divisor in range(1, normalized.quant_group_size + 1)
                if head_dim % divisor == 0
            )
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); "
                f"falling back to {group_size}."
            )
            normalized = replace(normalized, quant_group_size=group_size)
        if (
            normalized.quant_dtype in Q8_KV_DTYPES
            and normalized.quant_group_size == head_dim
            and (normalized.use_hadamard or normalized.use_shuffle)
        ):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change full-head Q8 quantization."
            )
    elif normalized.compute_in_f32:
        notes.append("[Info] COMPUTE_IN_F32 applies only to F16 KV storage and is ignored for F32.")
    return normalized, notes


class ONNX_STATIC_RESHAPE(torch.autograd.Function):
    """Export a static reshape while preserving a dynamic leading batch axis."""

    @staticmethod
    def forward(_ctx, values, shape):
        del _ctx
        eager_shape = tuple(
            values.shape[index] if dimension == 0 else dimension
            for index, dimension in enumerate(shape)
        )
        return values.reshape(eager_shape)

    @staticmethod
    def symbolic(graph, values, shape):
        shape_constant = graph.op(
            "Constant", value_t=torch.tensor(shape, dtype=torch.int64)
        )
        return graph.op("Reshape", values, shape_constant)


def onnx_static_reshape(values, shape):
    return ONNX_STATIC_RESHAPE.apply(values, tuple(shape))


def onnx_reshape_batch(values, shape):
    return onnx_static_reshape(values, (0,) + tuple(shape))


class KVQuantizer(torch.nn.Module):
    """Per-token Q4/Q8 KV storage with reversible layout transforms."""

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, settings):
        super().__init__()
        self.settings = settings
        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.num_kv_heads = int(num_kv_heads)
        self.num_kv_groups = int(num_kv_groups)
        self.is_q4 = settings.is_rotary_q4
        self.is_rotary = settings.is_rotary
        self.is_cuda_packed = settings.is_cuda_packed
        self.use_sym = bool(settings.use_sym)
        self.use_hadamard = bool(settings.use_hadamard)
        self.use_clip = bool(settings.use_clip)
        self.use_shuffle = bool(settings.use_shuffle)
        self.use_residual_bias_correction = (
            not self.use_sym and not settings.use_qdq_friendly_asym
        )

        if self.use_sym:
            self.signed_qmin = -8 if self.is_q4 else -128
            self.signed_qmax = 7 if self.is_q4 else 127
            self.qmax = float(self.signed_qmax)
        else:
            self.signed_qmin = None
            self.signed_qmax = None
            self.qmax = 15.0 if self.is_q4 else 255.0
        self.register_buffer(
            "inv_qmax", torch.tensor([1.0 / self.qmax], dtype=torch.float32).view(1, 1, 1, 1, 1)
        )

        self.is_grouped = self.is_q4 or (
            (self.use_hadamard or self.use_shuffle)
            and settings.quant_group_size < self.head_dim
        )
        if not self.is_grouped and not self.is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.group_size = settings.quant_group_size if self.is_grouped else 0
        self.num_groups = self.head_dim // self.group_size if self.is_grouped else 0

        if self.is_cuda_packed:
            for name, value in (
                ("_256", 256),
                ("_128", 128),
                ("_65536", 65536),
                ("_16777216", 16777216),
            ):
                self.register_buffer(
                    name, torch.tensor([value], dtype=torch.int32).view(1, 1, 1, 1, 1)
                )

        if self.is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2], dtype=torch.float32))
            forward_sin = torch.cat((
                torch.full((self.head_dim_half,), -inv_sqrt2),
                torch.full((self.head_dim_half,), inv_sqrt2),
            ))
            self.register_buffer("rot_sin_k", forward_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", forward_sin.view(1, 1, 1, 1, -1))

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.group_size)
            self.hadamard_pad = self.hadamard_size - self.group_size
            self.register_buffer(
                "hadamard_inv_sqrt",
                torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32),
            )
            sign_generator = torch.Generator()
            sign_generator.manual_seed(settings.hadamard_random_seed)
            signs = torch.randint(
                0, 2, (self.group_size,), generator=sign_generator, dtype=torch.int64
            ).float()
            self.register_buffer("hadamard_sign", signs.mul_(2.0).sub_(1.0))
            self._hadamard_levels = []
            width = self.hadamard_size
            while width > 1:
                half = width // 2
                self._hadamard_levels.append((width, half))
                width = half

        if self.use_clip:
            self.register_buffer(
                "clip_sigma", torch.tensor([settings.clip_sigma], dtype=torch.float32)
            )

        if self.use_shuffle:
            permutation = torch.arange(self.head_dim).view(
                self.num_groups, self.group_size
            ).transpose(0, 1).contiguous().view(-1)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(self.head_dim)
            self.register_buffer("shuffle_idx", permutation.int())
            self.register_buffer("unshuffle_idx", inverse.int())

    @staticmethod
    def _next_power_of_two(value):
        result = 1
        while result < value:
            result *= 2
        return result

    @property
    def packed_head_dim(self):
        if self.is_q4:
            return self.head_dim // (8 if self.is_cuda_packed else 2)
        return self.head_dim // 4 if self.is_cuda_packed else self.head_dim

    @property
    def cuda_unpacked_head_dim(self):
        return self.head_dim // 2 if self.is_q4 else self.head_dim

    @property
    def storage_dtype(self):
        if self.is_cuda_packed:
            return torch.int32
        if self.is_q4 or not self.use_sym:
            return torch.uint8
        return torch.int8

    @property
    def scale_bias_dtype(self):
        return torch.float16 if self.settings.use_float16_scale_bias else torch.float32

    def _apply_hadamard_last_dim(self, values, inverse=False):
        if not self.use_hadamard:
            return values
        if not inverse:
            values = values * self.hadamard_sign
        if self.hadamard_pad:
            values = F.pad(values, (0, self.hadamard_pad))
        leading_shape = (0,) * (values.dim() - 1)
        for width, half in self._hadamard_levels:
            values = onnx_static_reshape(values, leading_shape + (-1, width))
            even, odd = torch.split(values, (half, half), dim=-1)
            values = torch.cat((even + odd, even - odd), dim=-1)
            values = onnx_static_reshape(values, leading_shape + (-1,))
        values = values * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            values = values[..., :self.group_size]
        if inverse:
            values = values * self.hadamard_sign
        return values

    def _clip_to_sigma(self, values, dim):
        mean = values.mean(dim=dim, keepdim=True)
        variance = (values - mean).square().mean(dim=dim, keepdim=True)
        bound = self.clip_sigma * variance.sqrt()
        return values.clamp(mean - bound, mean + bound)

    def _flip_k(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        )
        return onnx_reshape_batch(
            values.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _flip_v(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, -1, 2, self.head_dim_half)
        )
        return onnx_reshape_batch(
            values.flip(-2), (self.num_kv_heads, 1, -1, self.head_dim)
        )

    def rotate_k(self, values):
        return values * self.rot_cos + self._flip_k(values) * self.rot_sin_k

    def rotate_v(self, values):
        return values * self.rot_cos + self._flip_v(values) * self.rot_sin_v

    def inverse_rotate_k(self, values):
        return values * self.rot_cos - self._flip_k(values) * self.rot_sin_k

    def inverse_rotate_v(self, values):
        return values * self.rot_cos - self._flip_v(values) * self.rot_sin_v

    def hadamard_k(self, values, inverse=False):
        values = onnx_reshape_batch(
            values,
            (self.num_kv_heads, 1, self.num_groups, self.group_size, -1),
        )
        values = self._apply_hadamard_last_dim(values.transpose(-1, -2), inverse).transpose(-1, -2)
        return onnx_reshape_batch(values, (self.num_kv_heads, 1, self.head_dim, -1))

    def hadamard_v(self, values, inverse=False):
        values = onnx_reshape_batch(
            values,
            (self.num_kv_heads, 1, -1, self.num_groups, self.group_size),
        )
        values = self._apply_hadamard_last_dim(values, inverse)
        return onnx_reshape_batch(values, (self.num_kv_heads, 1, -1, self.head_dim))

    def _finalize_asymmetric_quant(self, values, packed, scale, bias, dim):
        if self.use_residual_bias_correction:
            residual = values - (packed * scale + bias)
            bias = bias + residual.mean(dim=dim, keepdim=True)
        if not self.is_cuda_packed:
            packed = packed.to(torch.uint8)
        if self.settings.use_float16_scale_bias:
            scale = scale.half()
            bias = bias.half()
        return packed, scale, bias

    def _quantize_signed_to_storage(self, values, scale):
        quantized = torch.round(values / scale).clamp(
            self.signed_qmin, self.signed_qmax
        ).to(torch.int32)
        if self.is_q4 or self.is_cuda_packed:
            return torch.remainder(quantized, 16 if self.is_q4 else 256).to(torch.uint8)
        return quantized.to(torch.int8)

    @staticmethod
    def _decode_signed_q4_storage(values):
        values = values.to(torch.int16)
        return torch.remainder(values + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8_storage(values):
        if values.dtype == torch.int8:
            return values.to(torch.int16)
        values = values.to(torch.int16)
        return torch.remainder(values + 128, 256) - 128

    @staticmethod
    def _nonzero_scale(scale):
        return torch.where(scale == 0, torch.ones_like(scale), scale)

    def _quantize_block(self, values, dim):
        if self.is_grouped:
            return self._quantize_block_grouped(values, dim)
        if self.use_sym:
            if self.use_clip:
                values = self._clip_to_sigma(values, dim)
            scale = self._nonzero_scale(values.abs().amax(dim=dim, keepdim=True) * self.inv_qmax)
            packed = self._quantize_signed_to_storage(values, scale)
            if self.settings.use_float16_scale_bias:
                scale = scale.half()
            return packed, scale
        if self.use_clip:
            values = self._clip_to_sigma(values, dim)
        block_min, block_max = torch.aminmax(values, dim=dim, keepdim=True)
        scale = self._nonzero_scale((block_max - block_min) * self.inv_qmax)
        packed = torch.round((values - block_min) / scale)
        return self._finalize_asymmetric_quant(values, packed, scale, block_min, dim)

    def _quantize_block_grouped(self, values, dim):
        if dim == -2:
            values = onnx_reshape_batch(
                values,
                (self.num_kv_heads, 1, self.num_groups, self.group_size, -1),
            )
            quant_dim = -2
            packed_shape = (self.num_kv_heads, 1, self.head_dim, -1)
        else:
            values = onnx_reshape_batch(
                values,
                (self.num_kv_heads, 1, -1, self.num_groups, self.group_size),
            )
            quant_dim = -1
            packed_shape = (self.num_kv_heads, 1, -1, self.head_dim)
        if self.use_sym:
            if self.use_clip:
                values = self._clip_to_sigma(values, quant_dim)
            scale = self._nonzero_scale(
                values.abs().amax(dim=quant_dim, keepdim=True) * self.inv_qmax
            )
            packed = self._quantize_signed_to_storage(values, scale)
            packed = onnx_reshape_batch(packed, packed_shape)
            if self.settings.use_float16_scale_bias:
                scale = scale.half()
            return packed, scale
        if self.use_clip:
            values = self._clip_to_sigma(values, quant_dim)
        block_min, block_max = torch.aminmax(values, dim=quant_dim, keepdim=True)
        scale = self._nonzero_scale((block_max - block_min) * self.inv_qmax)
        packed = torch.round((values - block_min) / scale)
        packed, scale, block_min = self._finalize_asymmetric_quant(
            values, packed, scale, block_min, quant_dim
        )
        return onnx_reshape_batch(packed, packed_shape), scale, block_min

    def _pack_cuda(self, values, dim):
        values = values.to(torch.int32)
        packed_width = self.cuda_unpacked_head_dim // 4
        if dim == -2:
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, packed_width, 4, -1)
            )
        else:
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, -1, packed_width, 4)
            )
        value0, value1, value2, value3 = torch.unbind(values, dim=dim)
        return (
            value0
            + value1 * self._256
            + value2 * self._65536
            + (value3 - self._128) * self._16777216
        )

    def _unpack_cuda(self, values, dim):
        remainder3 = values % self._16777216
        value3 = (values - remainder3) // self._16777216 + self._128
        value2 = remainder3 // self._65536
        remainder2 = remainder3 % self._65536
        value1 = remainder2 // self._256
        value0 = remainder2 % self._256
        unpacked = torch.stack((value0, value1, value2, value3), dim=dim)
        if dim == -2:
            return onnx_reshape_batch(
                unpacked, (self.num_kv_heads, 1, self.cuda_unpacked_head_dim, -1)
            )
        return onnx_reshape_batch(
            unpacked, (self.num_kv_heads, 1, -1, self.cuda_unpacked_head_dim)
        )

    def _pack_q4_k(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        )
        low, high = torch.unbind(values, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def _pack_q4_v(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        )
        low, high = torch.unbind(values, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def _unpack_q4_k(self, values):
        low = values % 16
        high = values // 16
        return onnx_reshape_batch(
            torch.stack((low, high), dim=-2),
            (self.num_kv_heads, 1, self.head_dim, -1),
        )

    def _unpack_q4_v(self, values):
        low = values % 16
        high = values // 16
        return onnx_reshape_batch(
            torch.stack((low, high), dim=-1),
            (self.num_kv_heads, 1, -1, self.head_dim),
        )

    def quantize(self, keys, values):
        if self.is_rotary:
            keys = self.rotate_k(keys)
            values = self.rotate_v(values)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys)
            values = self.hadamard_v(values)

        if self.use_sym:
            packed_k, scale_k = self._quantize_block(keys, -2)
            packed_v, scale_v = self._quantize_block(values, -1)
            bias_k = bias_v = None
        else:
            packed_k, scale_k, bias_k = self._quantize_block(keys, -2)
            packed_v, scale_v, bias_v = self._quantize_block(values, -1)
        if self.is_q4:
            packed_k = self._pack_q4_k(packed_k)
            packed_v = self._pack_q4_v(packed_v)
        if self.is_cuda_packed:
            packed_k = self._pack_cuda(packed_k, -2)
            packed_v = self._pack_cuda(packed_v, -1)
        return packed_k, scale_k, bias_k, packed_v, scale_v, bias_v

    def _dequantize(self, packed, scale, bias, *, key):
        if self.is_cuda_packed:
            packed = self._unpack_cuda(packed, -2 if key else -1)
        if self.is_q4:
            packed = self._unpack_q4_k(packed) if key else self._unpack_q4_v(packed)
            if self.use_sym:
                packed = self._decode_signed_q4_storage(packed)
        elif self.use_sym:
            packed = self._decode_signed_q8_storage(packed)
        values = packed.float()
        scale = scale.float()
        if bias is not None:
            bias = bias.float()
        if self.is_grouped:
            if key:
                values = onnx_reshape_batch(
                    values,
                    (self.num_kv_heads, 1, self.num_groups, self.group_size, -1),
                )
                values = values * scale if self.use_sym else values * scale + bias
                values = onnx_reshape_batch(
                    values, (self.num_kv_heads, 1, self.head_dim, -1)
                )
            else:
                values = onnx_reshape_batch(
                    values,
                    (self.num_kv_heads, 1, -1, self.num_groups, self.group_size),
                )
                values = values * scale if self.use_sym else values * scale + bias
                values = onnx_reshape_batch(
                    values, (self.num_kv_heads, 1, -1, self.head_dim)
                )
        else:
            values = values * scale if self.use_sym else values * scale + bias
        if self.use_hadamard:
            values = self.hadamard_k(values, inverse=True) if key else self.hadamard_v(values, inverse=True)
        if self.use_shuffle:
            values = values.index_select(3 if key else -1, self.unshuffle_idx)
        if self.is_rotary:
            values = self.inverse_rotate_k(values) if key else self.inverse_rotate_v(values)
        return values

    def dequantize_key(self, packed, scale, bias=None):
        return self._dequantize(packed, scale, bias, key=True)

    def dequantize_value(self, packed, scale, bias=None):
        return self._dequantize(packed, scale, bias, key=False)


def _norm_epsilon(module: torch.nn.Module) -> float:
    return float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-5)))


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Export ORT fused RMSNorm with FP32 accumulation."""

    @staticmethod
    def forward(_ctx, values, scale, epsilon, axis):
        del _ctx
        variance = values.float().square().mean(dim=axis, keepdim=True)
        normalized = values.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(values.dtype)

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


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


class LFM_EMBED(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.embed_tokens = backbone.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


def _build_rotary_cos_sin(backbone, max_seq_len):
    rotary = backbone.rotary_emb
    inv_freq = rotary.inv_freq.float()
    attention_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
    freqs = (positions * inv_freq).unsqueeze(0).unsqueeze(2).unsqueeze(2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attention_scaling
    sin = torch.cat([-freqs.sin(), freqs.sin()], dim=-1) * attention_scaling
    return cos, sin


class ROTARY_PREFILL(torch.nn.Module):
    def __init__(self, backbone, max_seq_len):
        super().__init__()
        cos, sin = _build_rotary_cos_sin(backbone, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_DECODE(torch.nn.Module):
    def __init__(self, backbone, max_seq_len):
        super().__init__()
        cos, sin = _build_rotary_cos_sin(backbone, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class PENALTY_GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted_logits = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids.long(), adjusted_logits)
        token = torch.argmax(scores, dim=-1, keepdim=True).int()
        return token, torch.cat([previous_ids, token], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    @staticmethod
    def _sample(scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True)
        return torch.gather(sorted_indices, 1, winner).int()

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted_logits = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids.long(), adjusted_logits)
        token = self._sample(scores, temperature, top_k, top_p)
        return token, torch.cat([previous_ids, token], dim=-1)


def _channel_statistic(weight, key):
    absolute = weight.abs()
    if key == "rms":
        return (weight * weight).mean(0).sqrt()
    if key == "L4":
        return absolute.pow(4).mean(0).pow(0.25)
    if key == "std":
        return weight.std(0)
    if key != "absmean":
        raise ValueError(f"Unsupported reordering statistic: {key!r}.")
    return absolute.mean(0)


def _validate_permutation(permutation, width):
    expected = torch.arange(width, device=permutation.device)
    if permutation.ndim != 1 or permutation.numel() != width:
        raise ValueError("Channel permutation has an unexpected shape.")
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Channel permutation is not bijective.")


@dataclass(frozen=True)
class ReorderSummary:
    language_pairs: int
    maximum_error: float


class LFM_MAIN(torch.nn.Module):
    """Fused LFM2 language decoder with KV and short-convolution state."""

    def __init__(
        self,
        backbone,
        config,
        lm_head,
        *,
        kv_cache_settings=None,
        use_float16_kv=None,
        compute_in_f32=None,
        reorder_downproj=True,
        reorder_key="absmean",
    ):
        super().__init__()
        self.backbone = backbone
        self.config = config
        if kv_cache_settings is None:
            kv_cache_settings = KVCacheSettings(
                quant_dtype="F16" if use_float16_kv else "F32",
                compute_in_f32=bool(compute_in_f32),
            )
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        first_attention = next(layer for layer in backbone.layers if layer.is_attention_layer)
        self.head_dim = int(getattr(first_attention.self_attn, "head_dim", self.hidden_size // self.num_heads))
        if self.head_dim * self.num_heads != self.hidden_size or self.head_dim % 2:
            raise ValueError("Unexpected LFM attention geometry.")
        self.head_dim_half = self.head_dim // 2
        self.kv_settings, self.kv_setting_notes = normalize_kv_cache_settings(
            kv_cache_settings, self.head_dim
        )
        self.kv_quantized = self.kv_settings.is_quantized
        self.kv_f16 = self.kv_settings.quant_dtype == "F16"
        self.use_float16_kv = self.kv_f16
        self.compute_in_f32 = bool(self.kv_f16 and self.kv_settings.compute_in_f32)
        self.kv_sym = bool(self.kv_quantized and self.kv_settings.use_sym)
        self.num_attn_layers = sum(bool(layer.is_attention_layer) for layer in backbone.layers)
        self.num_conv_layers = len(backbone.layers) - self.num_attn_layers
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.total_qkv_heads = self.qk_heads + self.num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, self.num_key_value_heads]
        self.qk_split_sizes = [self.num_heads, self.num_key_value_heads]
        self.o_proj_in_features = self.num_heads * self.head_dim
        self.quantizer = (
            KVQuantizer(
                self.head_dim,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.kv_settings,
            ).eval()
            if self.kv_quantized
            else None
        )
        self.kv_grouped_layout = bool(
            self.quantizer is not None and self.quantizer.is_grouped
        )
        self.kv_cache_dtype = (
            self.quantizer.storage_dtype
            if self.quantizer is not None
            else (torch.float16 if self.kv_f16 else torch.float32)
        )
        self.kv_scale_bias_dtype = (
            self.quantizer.scale_bias_dtype if self.quantizer is not None else None
        )
        self.conv_cache_dtype = torch.float16 if self.kv_f16 else torch.float32
        self.key_offset = 0
        self.value_offset = self.num_attn_layers
        self.key_scale_offset = None
        self.key_bias_offset = None
        self.value_scale_offset = None
        self.value_bias_offset = None
        self.cache_state_groups = ["key", "value"]
        self.kv_blocks_per_attention = 2
        if self.kv_quantized:
            self.key_scale_offset = self.num_attn_layers * 2
            self.cache_state_groups.append("key_scale")
            if self.kv_sym:
                self.value_scale_offset = self.num_attn_layers * 3
                self.cache_state_groups.append("value_scale")
                self.kv_blocks_per_attention = 4
            else:
                self.key_bias_offset = self.num_attn_layers * 3
                self.value_scale_offset = self.num_attn_layers * 4
                self.value_bias_offset = self.num_attn_layers * 5
                self.cache_state_groups.extend(("key_bias", "value_scale", "value_bias"))
                self.kv_blocks_per_attention = 6
        self.conv_state_offset = self.num_attn_layers * self.kv_blocks_per_attention
        self.cache_state_groups.append("conv")
        self.cache_state_groups = tuple(self.cache_state_groups)
        self.register_buffer("hidden_norm_scale", torch.ones(self.hidden_size, dtype=torch.float32))
        self.register_buffer("qk_norm_scale", torch.ones(self.head_dim, dtype=torch.float32))
        self.register_buffer("attention_scale", torch.tensor(self.head_dim ** -0.25, dtype=torch.float32))
        self.lm_head = lm_head
        self.reorder_summary = self._fuse_weights(reorder_downproj, reorder_key)

    @property
    def cache_state_count(self):
        return self.num_attn_layers * self.kv_blocks_per_attention + self.num_conv_layers

    @staticmethod
    def dtype_name(dtype):
        return {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.int32: "int32",
        }.get(dtype, str(dtype))

    @staticmethod
    def _make_linear(weight, bias=None):
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
        return linear

    def _reorder_gated_mlp(self, layer, gate_weight, up_weight, gate_bias, up_bias, key):
        down = layer.feed_forward.w2
        width = int(down.in_features)
        if gate_weight.shape[0] != width or up_weight.shape[0] != width:
            raise ValueError("LFM gated MLP producer widths do not match down projection input.")
        if (gate_bias is None) != (up_bias is None):
            raise ValueError("LFM gated MLP producer biases must be present together.")
        permutation = torch.argsort(_channel_statistic(down.weight.detach(), key))
        _validate_permutation(permutation, width)
        probe = torch.randn((2, gate_weight.shape[1]), dtype=gate_weight.dtype)
        before = F.linear(
            F.silu(F.linear(probe, gate_weight, gate_bias)) * F.linear(probe, up_weight, up_bias),
            down.weight,
            down.bias,
        )
        reordered_gate = gate_weight[permutation].clone()
        reordered_up = up_weight[permutation].clone()
        reordered_gate_bias = gate_bias[permutation].clone() if gate_bias is not None else None
        reordered_up_bias = up_bias[permutation].clone() if up_bias is not None else None
        reordered_down = down.weight.detach()[:, permutation].clone()
        after = F.linear(
            F.silu(F.linear(probe, reordered_gate, reordered_gate_bias))
            * F.linear(probe, reordered_up, reordered_up_bias),
            reordered_down,
            down.bias,
        )
        error = float((before - after).abs().max())
        if error > 2e-5:
            raise RuntimeError(f"LFM gated MLP reordering is not equivalent (max error {error}).")
        down.weight.data.copy_(reordered_down)
        return reordered_gate, reordered_up, reordered_gate_bias, reordered_up_bias, error

    def _fuse_weights(self, reorder_downproj, reorder_key):
        pairs = 0
        maximum_error = 0.0
        with torch.no_grad():
            for layer in self.backbone.layers:
                operator_scale = layer.operator_norm.weight.detach().unsqueeze(0)
                if layer.is_attention_layer:
                    attn = layer.self_attn
                    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
                    qkv_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0) * operator_scale
                    qkv_bias = None
                    if any(projection.bias is not None for projection in (q_proj, k_proj, v_proj)):
                        if not all(projection.bias is not None for projection in (q_proj, k_proj, v_proj)):
                            raise ValueError("LFM QKV projections must either all have biases or none.")
                        qkv_bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
                    attn.qkv = self._make_linear(qkv_weight, qkv_bias)
                    q_weight = attn.q_layernorm.weight.detach() * self.attention_scale
                    k_weight = attn.k_layernorm.weight.detach() * self.attention_scale
                    attn.register_buffer(
                        "qk_norm_weight",
                        torch.cat([q_weight.repeat(self.num_heads), k_weight.repeat(self.num_key_value_heads)]).view(
                            1, 1, 1, self.qk_heads, self.head_dim
                        ),
                    )
                    attn.operator_norm_epsilon = _norm_epsilon(layer.operator_norm)
                    attn.qk_norm_epsilon = _norm_epsilon(attn.q_layernorm)
                    del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_layernorm, attn.k_layernorm
                else:
                    layer.conv.in_proj.weight.data.mul_(operator_scale)
                    layer.conv.operator_norm_epsilon = _norm_epsilon(layer.operator_norm)

                feed_forward = layer.feed_forward
                ffn_scale = layer.ffn_norm.weight.detach().unsqueeze(0)
                gate, up = feed_forward.w1, feed_forward.w3
                gate_weight = gate.weight.detach() * ffn_scale
                up_weight = up.weight.detach() * ffn_scale
                gate_bias = gate.bias.detach() if gate.bias is not None else None
                up_bias = up.bias.detach() if up.bias is not None else None
                if reorder_downproj:
                    gate_weight, up_weight, gate_bias, up_bias, error = self._reorder_gated_mlp(
                        layer, gate_weight, up_weight, gate_bias, up_bias, reorder_key
                    )
                    pairs += 1
                    maximum_error = max(maximum_error, error)
                feed_forward.gate_up = self._make_linear(
                    torch.cat([gate_weight, up_weight], dim=0),
                    torch.cat([gate_bias, up_bias], dim=0) if gate_bias is not None else None,
                )
                feed_forward.mlp_split = [int(gate.out_features), int(up.out_features)]
                feed_forward.norm_epsilon = _norm_epsilon(layer.ffn_norm)
                del feed_forward.w1, feed_forward.w3, layer.operator_norm, layer.ffn_norm

            self.register_buffer("final_norm_scale", self.backbone.embedding_norm.weight.detach().clone())
            self.final_norm_epsilon = _norm_epsilon(self.backbone.embedding_norm)
            del self.backbone.embedding_norm
        return ReorderSummary(pairs, maximum_error)

    @staticmethod
    def _rms_norm(values, epsilon, scale):
        return simplified_layer_norm(values, scale, epsilon)

    def _rotate_half_qk(self, values, batch_size):
        values = values.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        return values.flip(-2).view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states, rotary_cos, rotary_sin, attention_mask = all_inputs[-4:]
        batch_size, ids_len = hidden_states.shape[:2]
        attention_count = 0
        convolution_count = 0
        keys = [None] * self.num_attn_layers
        values = [None] * self.num_attn_layers
        key_scales = [None] * self.num_attn_layers if self.kv_quantized else None
        value_scales = [None] * self.num_attn_layers if self.kv_quantized else None
        key_biases = [None] * self.num_attn_layers if self.kv_quantized and not self.kv_sym else None
        value_biases = [None] * self.num_attn_layers if self.kv_quantized and not self.kv_sym else None
        conv_states = [None] * self.num_conv_layers
        for layer in self.backbone.layers:
            if layer.is_attention_layer:
                attention = layer.self_attn
                normalized = self._rms_norm(hidden_states, attention.operator_norm_epsilon, self.hidden_norm_scale)
                qkv = attention.qkv(normalized).reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
                qk, values_current = torch.split(qkv, self.qkv_split_sizes, dim=-2)
                qk = self._rms_norm(qk, attention.qk_norm_epsilon, self.qk_norm_scale) * attention.qk_norm_weight
                qk = qk * rotary_cos + self._rotate_half_qk(qk, batch_size) * rotary_sin
                queries, keys_current = torch.split(qk, self.qk_split_sizes, dim=-2)
                queries = queries.reshape(
                    batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
                ).permute(0, 2, 3, 1, 4)
                keys_current = keys_current.permute(0, 3, 2, 4, 1)
                values_current = values_current.transpose(1, 3)

                if self.kv_quantized:
                    (
                        packed_key,
                        key_scale,
                        key_bias,
                        packed_value,
                        value_scale,
                        value_bias,
                    ) = self.quantizer.quantize(keys_current, values_current)
                    key_cache = torch.cat(
                        [all_inputs[self.key_offset + attention_count], packed_key], dim=-1
                    )
                    value_cache = torch.cat(
                        [all_inputs[self.value_offset + attention_count], packed_value], dim=-2
                    )
                    key_scale_cache = torch.cat(
                        [all_inputs[self.key_scale_offset + attention_count], key_scale], dim=-1
                    )
                    value_scale_cache = torch.cat(
                        [all_inputs[self.value_scale_offset + attention_count], value_scale],
                        dim=-3 if self.kv_grouped_layout else -2,
                    )
                    if self.kv_sym:
                        key_bias_cache = None
                        value_bias_cache = None
                    else:
                        key_bias_cache = torch.cat(
                            [all_inputs[self.key_bias_offset + attention_count], key_bias], dim=-1
                        )
                        value_bias_cache = torch.cat(
                            [all_inputs[self.value_bias_offset + attention_count], value_bias],
                            dim=-3 if self.kv_grouped_layout else -2,
                        )
                    keys[attention_count] = key_cache
                    values[attention_count] = value_cache
                    key_scales[attention_count] = key_scale_cache
                    value_scales[attention_count] = value_scale_cache
                    if not self.kv_sym:
                        key_biases[attention_count] = key_bias_cache
                        value_biases[attention_count] = value_bias_cache
                    attention_key = self.quantizer.dequantize_key(
                        key_cache, key_scale_cache, key_bias_cache
                    )
                    attention_value = self.quantizer.dequantize_value(
                        value_cache, value_scale_cache, value_bias_cache
                    )
                    attention_mask_current = attention_mask
                else:
                    if self.kv_f16:
                        if not self.compute_in_f32:
                            queries = queries.half()
                        keys_current = keys_current.half()
                        values_current = values_current.half()
                    key_cache = torch.cat(
                        [all_inputs[self.key_offset + attention_count], keys_current], dim=-1
                    )
                    value_cache = torch.cat(
                        [all_inputs[self.value_offset + attention_count], values_current], dim=-2
                    )
                    keys[attention_count] = key_cache
                    values[attention_count] = value_cache
                    if self.kv_f16 and self.compute_in_f32:
                        attention_key = key_cache.float()
                        attention_value = value_cache.float()
                    else:
                        attention_key = key_cache
                        attention_value = value_cache
                    attention_mask_current = (
                        attention_mask.half()
                        if self.kv_f16 and not self.compute_in_f32
                        else attention_mask
                    )

                scores = torch.softmax(
                    torch.matmul(queries, attention_key) + attention_mask_current, dim=-1
                )
                attention_output = torch.matmul(scores, attention_value)
                if self.kv_f16 and not self.compute_in_f32:
                    attention_output = attention_output.float()
                attention_output = attention_output.permute(0, 3, 1, 2, 4).reshape(
                    batch_size, -1, self.o_proj_in_features
                )
                operator_output = attention.out_proj(attention_output)
                attention_count += 1
            else:
                normalized = self._rms_norm(hidden_states, layer.conv.operator_norm_epsilon, self.hidden_norm_scale)
                bcx = layer.conv.in_proj(normalized).transpose(-1, -2)
                b_values, c_values, x_values = bcx.chunk(3, dim=-2)
                state = torch.cat(
                    [all_inputs[self.conv_state_offset + convolution_count].float(), b_values * x_values], dim=-1
                )
                conv_states[convolution_count] = state[..., -2:].to(self.conv_cache_dtype)
                conv_output = layer.conv.conv(state)[..., :state.shape[-1]][..., -ids_len:]
                operator_output = layer.conv.out_proj((c_values * conv_output).transpose(-1, -2).contiguous())
                convolution_count += 1
            hidden_states = hidden_states + operator_output
            feed_forward = layer.feed_forward
            normalized = self._rms_norm(hidden_states, feed_forward.norm_epsilon, self.hidden_norm_scale)
            gate_up = feed_forward.gate_up(normalized)
            gate, up = torch.split(gate_up, feed_forward.mlp_split, dim=-1)
            hidden_states = hidden_states + feed_forward.w2(F.silu(gate) * up)
        last = self._rms_norm(hidden_states[:, -1], self.final_norm_epsilon, self.final_norm_scale)
        if self.kv_quantized:
            if self.kv_sym:
                return (*keys, *values, *key_scales, *value_scales, *conv_states, self.lm_head(last))
            return (
                *keys,
                *values,
                *key_scales,
                *key_biases,
                *value_scales,
                *value_biases,
                *conv_states,
                self.lm_head(last),
            )
        return (*keys, *values, *conv_states, self.lm_head(last))


def build_cache_io(main: LFM_MAIN, batch_size: int, *, cache_dtype=None):
    """Build zero-length recurrent-state IO matching ``main``'s KV storage mode."""
    if cache_dtype is not None and cache_dtype != main.kv_cache_dtype:
        raise ValueError(
            f"cache_dtype ({cache_dtype}) does not match configured KV storage ({main.kv_cache_dtype})."
        )
    packed_head_dim = (
        main.quantizer.packed_head_dim if main.kv_quantized else main.head_dim
    )
    key = torch.zeros(
        (batch_size, main.num_key_value_heads, 1, packed_head_dim, 0),
        dtype=main.kv_cache_dtype,
    )
    value = torch.zeros(
        (batch_size, main.num_key_value_heads, 1, 0, packed_head_dim),
        dtype=main.kv_cache_dtype,
    )
    conv = torch.zeros(
        (batch_size, main.hidden_size, 0), dtype=main.conv_cache_dtype
    )
    tensors, input_names, output_names, dynamic_axes = [], [], [], {}

    def add(group, tensor, sequence_axis, count):
        for index in range(count):
            input_name = f"in_{group}_{index}"
            output_name = f"out_{group}_{index}"
            tensors.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: "batch", sequence_axis: "history_len"}
            dynamic_axes[output_name] = {0: "batch", sequence_axis: "kv_seq_len"}

    add("key", key, 4, main.num_attn_layers)
    add("value", value, 3, main.num_attn_layers)
    if main.kv_quantized:
        if main.kv_grouped_layout:
            key_scale = torch.ones(
                (
                    batch_size,
                    main.num_key_value_heads,
                    1,
                    main.quantizer.num_groups,
                    1,
                    0,
                ),
                dtype=main.kv_scale_bias_dtype,
            )
            value_scale = torch.ones(
                (
                    batch_size,
                    main.num_key_value_heads,
                    1,
                    0,
                    main.quantizer.num_groups,
                    1,
                ),
                dtype=main.kv_scale_bias_dtype,
            )
            key_scale_axis = 5
            value_scale_axis = 3
        else:
            key_scale = torch.ones(
                (batch_size, main.num_key_value_heads, 1, 1, 0),
                dtype=main.kv_scale_bias_dtype,
            )
            value_scale = torch.ones(
                (batch_size, main.num_key_value_heads, 1, 0, 1),
                dtype=main.kv_scale_bias_dtype,
            )
            key_scale_axis = 4
            value_scale_axis = 3
        add("key_scale", key_scale, key_scale_axis, main.num_attn_layers)
        if not main.kv_sym:
            add("key_bias", key_scale, key_scale_axis, main.num_attn_layers)
        add("value_scale", value_scale, value_scale_axis, main.num_attn_layers)
        if not main.kv_sym:
            add("value_bias", value_scale, value_scale_axis, main.num_attn_layers)
    add("conv", conv, 2, main.num_conv_layers)
    return tensors, input_names, output_names, dynamic_axes
