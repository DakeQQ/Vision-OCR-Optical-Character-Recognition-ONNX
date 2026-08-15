"""LFM hybrid language components shared by local LFM exporters.

The LFM2-350M-Extract and LFM2.5-VL-450M-Extract checkpoints share the same
six-attention / ten-short-convolution language backbone.  This module owns the
architecture-specific graph fragments; model-family exporters own filenames,
metadata, vision handling, and staging.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


def normalize_kv_quant_settings(kv_quant_dtype, kv_quant_group_size, head_dim):
    """Validate one Qwen-compatible KV cache configuration for an LFM head."""
    settings = KVCacheSettings(
        quant_dtype=str(kv_quant_dtype).upper(),
        quant_group_size=int(kv_quant_group_size),
    )
    normalized, notes = normalize_kv_cache_settings(settings, int(head_dim))
    return normalized.quant_dtype, normalized.quant_group_size, notes


def _norm_epsilon(module: torch.nn.Module) -> float:
    return float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-5)))


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Export ONNX Runtime's fused RMS normalization with FP32 accumulation."""

    @staticmethod
    def forward(ctx, values, scale, epsilon, axis):
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
    """Small graph used to load bundle metadata before large sessions."""

    def forward(self, marker):
        return marker


class LFM_EMBED(torch.nn.Module):
    """Token IDs to LFM embedding states."""

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
    """Return 1D RoPE tables, causal mask, and advanced sequence length."""

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
    """Return a single 1D RoPE position and advanced sequence length."""

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
    """Select the most likely next token."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class PENALTY_GREEDY_SEARCH(torch.nn.Module):
    """Apply sign-aware repetition penalty and select the most likely token."""

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
    """Top-k/top-p sampling with sign-aware repetition handling."""

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
    quant_dtype: str = "F32"
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
    def is_quantized(self) -> bool:
        return self.quant_dtype in QUANTIZED_KV_DTYPES

    @property
    def is_rotary(self) -> bool:
        return self.quant_dtype in ROTARY_KV_DTYPES

    @property
    def is_rotary_q4(self) -> bool:
        return self.quant_dtype in {"ROTARY_Q4", "ROTARY_Q4_CUDA"}

    @property
    def is_q8_cuda(self) -> bool:
        return self.quant_dtype in {"Q8_CUDA", "ROTARY_Q8_CUDA"}

    @property
    def is_rotary_cuda(self) -> bool:
        return self.quant_dtype in {"ROTARY_Q4_CUDA", "ROTARY_Q8_CUDA"}

    @property
    def uses_cuda_packing(self) -> bool:
        return self.quant_dtype in {"Q8_CUDA", "ROTARY_Q4_CUDA", "ROTARY_Q8_CUDA"}

    def uses_grouped_q8(self, head_dim: int) -> bool:
        return (
            self.quant_dtype in Q8_KV_DTYPES
            and (self.use_hadamard or self.use_shuffle)
            and self.quant_group_size < head_dim
        )

    def uses_grouped_layout(self, head_dim: int) -> bool:
        return self.is_rotary_q4 or self.uses_grouped_q8(head_dim)


def normalize_kv_cache_settings(settings: KVCacheSettings, head_dim: int) -> tuple[KVCacheSettings, list[str]]:
    if settings.quant_dtype not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {settings.quant_dtype}")
    if settings.quant_dtype in ROTARY_KV_DTYPES and head_dim % 2:
        raise ValueError(f"{settings.quant_dtype} requires an even head_dim, got {head_dim}.")
    if settings.quant_dtype in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4:
        raise ValueError(f"{settings.quant_dtype} requires head_dim divisible by 4, got {head_dim}.")
    if settings.quant_dtype == "ROTARY_Q4_CUDA" and head_dim % 8:
        raise ValueError(f"{settings.quant_dtype} requires head_dim divisible by 8, got {head_dim}.")

    notes: list[str] = []
    normalized = settings
    if normalized.is_quantized:
        if normalized.quant_group_size <= 0:
            raise ValueError(f"KV_QUANT_GROUP_SIZE must be positive, got {normalized.quant_group_size}.")
        if normalized.quant_group_size > head_dim:
            normalized = replace(normalized, quant_group_size=head_dim)
            notes.append("KV_QUANT_GROUP_SIZE exceeded head_dim and was clamped.")
        elif head_dim % normalized.quant_group_size:
            group_size = max(value for value in range(1, normalized.quant_group_size + 1) if head_dim % value == 0)
            normalized = replace(normalized, quant_group_size=group_size)
            notes.append("KV_QUANT_GROUP_SIZE was reduced to a divisor of head_dim.")
        if (
            normalized.quant_dtype in Q8_KV_DTYPES
            and normalized.quant_group_size == head_dim
            and (normalized.use_hadamard or normalized.use_shuffle)
        ):
            notes.append("Hadamard and shuffle do not change full-head Q8 quantization.")
    return normalized, notes


@dataclass(frozen=True)
class KVCacheGroup:
    name: str
    sequence_axis: int
    count: int


@dataclass(frozen=True)
class KVCacheLayout:
    settings: KVCacheSettings
    head_dim: int
    num_key_value_heads: int
    hidden_size: int
    num_attention_layers: int
    num_convolution_layers: int
    groups: tuple[KVCacheGroup, ...]
    key_storage_width: int
    value_storage_width: int
    group_count: int
    cache_dtype: torch.dtype
    scale_bias_dtype: torch.dtype | None

    @classmethod
    def create(
        cls,
        settings: KVCacheSettings,
        head_dim: int,
        num_key_value_heads: int,
        hidden_size: int,
        num_attention_layers: int,
        num_convolution_layers: int,
    ) -> "KVCacheLayout":
        settings, _ = normalize_kv_cache_settings(settings, head_dim)
        grouped = settings.uses_grouped_layout(head_dim)
        group_count = head_dim // settings.quant_group_size if grouped else 1
        groups = [
            KVCacheGroup("key", 4, num_attention_layers),
            KVCacheGroup("value", 3, num_attention_layers),
        ]
        if settings.is_quantized:
            key_scale_axis = 5 if grouped else 4
            groups.append(KVCacheGroup("key_scale", key_scale_axis, num_attention_layers))
            if not settings.use_sym:
                groups.append(KVCacheGroup("key_bias", key_scale_axis, num_attention_layers))
            groups.append(KVCacheGroup("value_scale", 3, num_attention_layers))
            if not settings.use_sym:
                groups.append(KVCacheGroup("value_bias", 3, num_attention_layers))
        groups.append(KVCacheGroup("conv", 2, num_convolution_layers))

        if settings.quant_dtype == "F16":
            cache_dtype = torch.float16
        elif settings.uses_cuda_packing:
            cache_dtype = torch.int32
        elif settings.is_quantized and settings.use_sym and not settings.is_rotary_q4:
            cache_dtype = torch.int8
        elif settings.is_quantized:
            cache_dtype = torch.uint8
        else:
            cache_dtype = torch.float32

        if settings.quant_dtype in {"Q8_CUDA", "ROTARY_Q8_CUDA"}:
            storage_width = head_dim // 4
        elif settings.quant_dtype == "ROTARY_Q4":
            storage_width = head_dim // 2
        elif settings.quant_dtype == "ROTARY_Q4_CUDA":
            storage_width = head_dim // 8
        else:
            storage_width = head_dim

        return cls(
            settings=settings,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            num_attention_layers=num_attention_layers,
            num_convolution_layers=num_convolution_layers,
            groups=tuple(groups),
            key_storage_width=storage_width,
            value_storage_width=storage_width,
            group_count=group_count,
            cache_dtype=cache_dtype,
            scale_bias_dtype=torch.float16 if settings.is_quantized and settings.use_float16_scale_bias else (
                torch.float32 if settings.is_quantized else None
            ),
        )

    @property
    def is_quantized(self) -> bool:
        return self.settings.is_quantized

    @property
    def is_symmetric(self) -> bool:
        return self.is_quantized and self.settings.use_sym

    @property
    def uses_grouped_layout(self) -> bool:
        return self.settings.uses_grouped_layout(self.head_dim)

    @property
    def kv_groups(self) -> tuple[KVCacheGroup, ...]:
        return tuple(group for group in self.groups if group.name != "conv")

    @property
    def state_count(self) -> int:
        return sum(group.count for group in self.groups)

    @property
    def kv_state_count(self) -> int:
        return sum(group.count for group in self.kv_groups)

    @property
    def blocks_per_attention_layer(self) -> int:
        return self.kv_state_count // self.num_attention_layers

    @property
    def group_offsets(self) -> dict[str, int]:
        offset = 0
        offsets = {}
        for group in self.groups:
            offsets[group.name] = offset
            offset += group.count
        return offsets

    def state_index(self, group_name: str, layer_index: int) -> int:
        group = next((item for item in self.groups if item.name == group_name), None)
        if group is None:
            raise KeyError(f"Unknown KV cache group: {group_name!r}")
        if not 0 <= layer_index < group.count:
            raise IndexError(f"KV cache layer index {layer_index} is outside {group_name!r}.")
        return self.group_offsets[group_name] + layer_index

    def sequence_axis(self, group_name: str) -> int:
        group = next((item for item in self.groups if item.name == group_name), None)
        if group is None:
            raise KeyError(f"Unknown KV cache group: {group_name!r}")
        return group.sequence_axis

    def tensor_shape(self, group_name: str, batch_size: int, sequence_length: int = 0) -> tuple[int, ...]:
        if group_name == "key":
            return batch_size, self.num_key_value_heads, 1, self.key_storage_width, sequence_length
        if group_name == "value":
            return batch_size, self.num_key_value_heads, 1, sequence_length, self.value_storage_width
        if group_name in {"key_scale", "key_bias"}:
            if self.uses_grouped_layout:
                return batch_size, self.num_key_value_heads, 1, self.group_count, 1, sequence_length
            return batch_size, self.num_key_value_heads, 1, 1, sequence_length
        if group_name in {"value_scale", "value_bias"}:
            if self.uses_grouped_layout:
                return batch_size, self.num_key_value_heads, 1, sequence_length, self.group_count, 1
            return batch_size, self.num_key_value_heads, 1, sequence_length, 1
        if group_name == "conv":
            return batch_size, self.hidden_size, sequence_length
        raise KeyError(f"Unknown KV cache group: {group_name!r}")

    def tensor_dtype(self, group_name: str) -> torch.dtype:
        if group_name in {"key", "value"}:
            return self.cache_dtype
        if group_name == "conv":
            return torch.float16 if self.settings.quant_dtype == "F16" else torch.float32
        if self.scale_bias_dtype is None:
            raise KeyError(f"{group_name!r} is unavailable for an unquantized cache.")
        return self.scale_bias_dtype

    def zeros(self, group_name: str, batch_size: int, sequence_length: int = 0) -> torch.Tensor:
        return torch.zeros(
            self.tensor_shape(group_name, batch_size, sequence_length),
            dtype=self.tensor_dtype(group_name),
        )


class LFMKVQuantizer(torch.nn.Module):
    """Qwen-compatible KV storage quantizer for LFM's attention tensor layout."""

    def __init__(self, layout: KVCacheLayout, num_key_value_groups: int):
        super().__init__()
        if not layout.is_quantized:
            raise ValueError("LFMKVQuantizer requires a quantized KV cache layout.")
        self.layout = layout
        self.settings = layout.settings
        self.head_dim = layout.head_dim
        self.head_dim_half = layout.head_dim // 2
        self.num_key_value_heads = layout.num_key_value_heads
        self.num_key_value_groups = int(num_key_value_groups)
        self.is_q4 = layout.settings.is_rotary_q4
        self.is_rotary = layout.settings.is_rotary
        self.is_cuda_packed = layout.settings.uses_cuda_packing
        self.use_sym = layout.settings.use_sym
        self.is_grouped = layout.uses_grouped_layout
        self.group_size = layout.settings.quant_group_size if self.is_grouped else 0
        self.group_count = layout.group_count if self.is_grouped else 0
        self.use_hadamard = bool(layout.settings.use_hadamard and self.is_grouped)
        self.use_shuffle = bool(layout.settings.use_shuffle and self.is_grouped)
        self.use_clip = bool(layout.settings.use_clip)
        self.use_residual_bias_correction = (
            not self.use_sym and not layout.settings.use_qdq_friendly_asym
        )
        self.qmax = float(7 if self.is_q4 and self.use_sym else 15 if self.is_q4 else 127 if self.use_sym else 255)

        if self.use_sym:
            self.signed_qmin = -8 if self.is_q4 else -128
            self.signed_qmax = 7 if self.is_q4 else 127
        else:
            self.signed_qmin = None
            self.signed_qmax = None

        if self.is_cuda_packed:
            self.register_buffer("_256", torch.tensor(256, dtype=torch.int32))
            self.register_buffer("_128", torch.tensor(128, dtype=torch.int32))
            self.register_buffer("_65536", torch.tensor(65536, dtype=torch.int32))
            self.register_buffer("_16777216", torch.tensor(16777216, dtype=torch.int32))

        if self.is_rotary:
            inverse_sqrt_two = 2.0 ** -0.5
            self.register_buffer("rot_cos", torch.tensor(inverse_sqrt_two, dtype=torch.float32))
            forward_sine = torch.cat([
                torch.full((self.head_dim_half,), -inverse_sqrt_two),
                torch.full((self.head_dim_half,), inverse_sqrt_two),
            ])
            self.register_buffer("rot_sin_key", forward_sine.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_value", forward_sine.view(1, 1, 1, 1, -1))

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.group_size)
            self.hadamard_pad = self.hadamard_size - self.group_size
            self.register_buffer(
                "hadamard_inv_sqrt", torch.tensor(self.hadamard_size ** -0.5, dtype=torch.float32)
            )
            sign_generator = torch.Generator()
            sign_generator.manual_seed(self.settings.hadamard_random_seed)
            signs = torch.randint(0, 2, (self.group_size,), generator=sign_generator, dtype=torch.int64)
            self.register_buffer("hadamard_sign", signs.float().mul_(2.0).sub_(1.0))
            levels = []
            width = self.hadamard_size
            while width > 1:
                levels.append((width, width // 2))
                width //= 2
            self._hadamard_levels = tuple(levels)

        if self.use_clip:
            self.register_buffer("clip_sigma", torch.tensor(float(self.settings.clip_sigma), dtype=torch.float32))

        if self.use_shuffle:
            permutation = torch.arange(self.head_dim).view(self.group_count, self.group_size).T.contiguous().view(-1)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(self.head_dim)
            self.register_buffer("shuffle_index", permutation.int())
            self.register_buffer("unshuffle_index", inverse.int())

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        result = 1
        while result < value:
            result *= 2
        return result

    def _scale_from_absmax(self, absmax: torch.Tensor) -> torch.Tensor:
        scale = absmax / self.qmax
        return torch.where(absmax > 0, scale, torch.ones_like(scale))

    def _clip_to_sigma(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        mean = values.mean(dim=dim, keepdim=True)
        variance = (values - mean).square().mean(dim=dim, keepdim=True)
        bound = self.clip_sigma * variance.sqrt()
        return values.clamp(mean - bound, mean + bound)

    def _flip_key(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(batch_size, self.num_key_value_heads, 1, 2, self.head_dim_half, -1)
        return values.flip(-3).reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)

    def _flip_value(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(batch_size, self.num_key_value_heads, 1, -1, 2, self.head_dim_half)
        return values.flip(-2).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)

    def _rotate_key(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.rot_cos + self._flip_key(values) * self.rot_sin_key

    def _rotate_value(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.rot_cos + self._flip_value(values) * self.rot_sin_value

    def _inverse_rotate_key(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.rot_cos - self._flip_key(values) * self.rot_sin_key

    def _inverse_rotate_value(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.rot_cos - self._flip_value(values) * self.rot_sin_value

    def _apply_hadamard_last_dim(self, values: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not self.use_hadamard:
            return values
        if not inverse:
            values = values * self.hadamard_sign
        if self.hadamard_pad:
            values = F.pad(values, (0, self.hadamard_pad))
        padded = values
        flattened = values.reshape(-1, self.hadamard_size)
        for width, half in self._hadamard_levels:
            flattened = flattened.reshape(-1, width)
            even, odd = torch.split(flattened, [half, half], dim=-1)
            flattened = torch.cat([even + odd, even - odd], dim=-1)
        values = flattened.reshape_as(padded) * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            values = values[..., :self.group_size]
        if inverse:
            values = values * self.hadamard_sign
        return values

    def _hadamard_key(self, values: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(
            batch_size, self.num_key_value_heads, 1, self.group_count, self.group_size, -1
        )
        values = self._apply_hadamard_last_dim(values.transpose(-1, -2), inverse=inverse).transpose(-1, -2)
        return values.reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)

    def _hadamard_value(self, values: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(
            batch_size, self.num_key_value_heads, 1, -1, self.group_count, self.group_size
        )
        values = self._apply_hadamard_last_dim(values, inverse=inverse)
        return values.reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)

    def _quantize_signed(self, values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        quantized = torch.round(values / scale).clamp(self.signed_qmin, self.signed_qmax).to(torch.int32)
        if self.is_q4:
            return torch.remainder(quantized, 16).to(torch.uint8)
        if self.is_cuda_packed:
            return torch.remainder(quantized, 256).to(torch.uint8)
        return quantized.to(torch.int8)

    @staticmethod
    def _decode_signed_q4(values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.int16)
        return torch.remainder(values + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8(values: torch.Tensor) -> torch.Tensor:
        if values.dtype == torch.int8:
            return values.to(torch.int16)
        values = values.to(torch.int16)
        return torch.remainder(values + 128, 256) - 128

    def _finalize_asymmetric(
        self,
        source: torch.Tensor,
        packed: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_residual_bias_correction:
            residual = source - (packed * scale + bias)
            bias = bias + residual.mean(dim=dim, keepdim=True)
        if not self.is_cuda_packed:
            packed = packed.to(torch.uint8)
        if self.settings.use_float16_scale_bias:
            scale = scale.half()
            bias = bias.half()
        return packed, scale, bias

    def _quantize(self, values: torch.Tensor, dim: int):
        if self.is_grouped:
            return self._quantize_grouped(values, dim)
        if self.use_clip:
            values = self._clip_to_sigma(values, dim)
        if self.use_sym:
            scale = self._scale_from_absmax(values.abs().amax(dim=dim, keepdim=True))
            packed = self._quantize_signed(values, scale)
            return packed, scale.half() if self.settings.use_float16_scale_bias else scale, None
        bias, maximum = torch.aminmax(values, dim=dim, keepdim=True)
        scale = self._scale_from_absmax(maximum - bias)
        packed = torch.round((values - bias) / scale)
        packed, scale, bias = self._finalize_asymmetric(values, packed, scale, bias, dim)
        return packed, scale, bias

    def _quantize_grouped(self, values: torch.Tensor, dim: int):
        batch_size = values.shape[0]
        if dim == -2:
            grouped = values.reshape(
                batch_size, self.num_key_value_heads, 1, self.group_count, self.group_size, -1
            )
            reduce_dim = -2
        else:
            grouped = values.reshape(
                batch_size, self.num_key_value_heads, 1, -1, self.group_count, self.group_size
            )
            reduce_dim = -1
        if self.use_clip:
            grouped = self._clip_to_sigma(grouped, reduce_dim)
        if self.use_sym:
            scale = self._scale_from_absmax(grouped.abs().amax(dim=reduce_dim, keepdim=True))
            packed = self._quantize_signed(grouped, scale)
            if dim == -2:
                packed = packed.reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)
            else:
                packed = packed.reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
            return packed, scale.half() if self.settings.use_float16_scale_bias else scale, None
        bias, maximum = torch.aminmax(grouped, dim=reduce_dim, keepdim=True)
        scale = self._scale_from_absmax(maximum - bias)
        packed = torch.round((grouped - bias) / scale)
        packed, scale, bias = self._finalize_asymmetric(grouped, packed, scale, bias, reduce_dim)
        if dim == -2:
            packed = packed.reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)
        else:
            packed = packed.reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
        return packed, scale, bias

    def _pack_cuda(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        batch_size = values.shape[0]
        packed_width = self.head_dim // (8 if self.is_q4 else 4)
        values = values.to(torch.int32)
        if dim == -2:
            values = values.reshape(batch_size, self.num_key_value_heads, 1, packed_width, 4, -1)
        else:
            values = values.reshape(batch_size, self.num_key_value_heads, 1, -1, packed_width, 4)
        first, second, third, fourth = torch.unbind(values, dim=dim)
        return first + second * self._256 + third * self._65536 + (fourth - self._128) * self._16777216

    def _unpack_cuda(self, values: torch.Tensor, dim: int, storage_width: int) -> torch.Tensor:
        remainder_three = values % self._16777216
        fourth = (values - remainder_three) // self._16777216 + self._128
        third = remainder_three // self._65536
        remainder_two = remainder_three % self._65536
        second = remainder_two // self._256
        first = remainder_two % self._256
        unpacked = torch.stack([first, second, third, fourth], dim=dim)
        batch_size = values.shape[0]
        if dim == -2:
            return unpacked.reshape(batch_size, self.num_key_value_heads, 1, storage_width, -1)
        return unpacked.reshape(batch_size, self.num_key_value_heads, 1, -1, storage_width)

    def _pack_q4_key(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(batch_size, self.num_key_value_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(values, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def _pack_q4_value(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(values, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def _unpack_q4_key(self, values: torch.Tensor) -> torch.Tensor:
        low, high = values % 16, values // 16
        batch_size = values.shape[0]
        return torch.stack([low, high], dim=-2).reshape(
            batch_size, self.num_key_value_heads, 1, self.head_dim, -1
        )

    def _unpack_q4_value(self, values: torch.Tensor) -> torch.Tensor:
        low, high = values % 16, values // 16
        batch_size = values.shape[0]
        return torch.stack([low, high], dim=-1).reshape(
            batch_size, self.num_key_value_heads, 1, -1, self.head_dim
        )

    def _prepare_key(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_rotary:
            values = self._rotate_key(values)
        if self.use_shuffle:
            values = values.index_select(3, self.shuffle_index)
        if self.use_hadamard:
            values = self._hadamard_key(values)
        return values

    def _prepare_value(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_rotary:
            values = self._rotate_value(values)
        if self.use_shuffle:
            values = values.index_select(-1, self.shuffle_index)
        if self.use_hadamard:
            values = self._hadamard_value(values)
        return values

    def _pack_key(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_q4:
            values = self._pack_q4_key(values)
        return self._pack_cuda(values, -2) if self.is_cuda_packed else values

    def _pack_value(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_q4:
            values = self._pack_q4_value(values)
        return self._pack_cuda(values, -1) if self.is_cuda_packed else values

    def quantize_key(self, values: torch.Tensor):
        packed, scale, bias = self._quantize(self._prepare_key(values), -2)
        return self._pack_key(packed), scale, bias

    def quantize_pair(self, keys: torch.Tensor, values: torch.Tensor):
        packed_key, key_scale, key_bias = self._quantize(self._prepare_key(keys), -2)
        packed_value, value_scale, value_bias = self._quantize(self._prepare_value(values), -1)
        return (
            self._pack_key(packed_key), key_scale, key_bias,
            self._pack_value(packed_value), value_scale, value_bias,
        )

    def _restore_key_codes(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_cuda_packed:
            values = self._unpack_cuda(values, -2, self.head_dim // 2 if self.is_q4 else self.head_dim)
        if self.is_q4:
            values = self._unpack_q4_key(values)
            return self._decode_signed_q4(values) if self.use_sym else values
        return self._decode_signed_q8(values) if self.use_sym else values

    def _restore_value_codes(self, values: torch.Tensor) -> torch.Tensor:
        if self.is_cuda_packed:
            values = self._unpack_cuda(values, -1, self.head_dim // 2 if self.is_q4 else self.head_dim)
        if self.is_q4:
            values = self._unpack_q4_value(values)
            return self._decode_signed_q4(values) if self.use_sym else values
        return self._decode_signed_q8(values) if self.use_sym else values

    def _dequantize_key(self, values: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        values = self._restore_key_codes(values).float()
        scale = scale.float()
        bias = bias.float() if bias is not None else None
        if self.is_grouped:
            batch_size = values.shape[0]
            values = values.reshape(
                batch_size, self.num_key_value_heads, 1, self.group_count, self.group_size, -1
            )
            values = values * scale if self.use_sym else values * scale + bias
            values = values.reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)
        else:
            values = values * scale if self.use_sym else values * scale + bias
        if self.use_hadamard:
            values = self._hadamard_key(values, inverse=True)
        if self.use_shuffle:
            values = values.index_select(3, self.unshuffle_index)
        if self.is_rotary:
            values = self._inverse_rotate_key(values)
        return values

    def _dequantize_value(self, values: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        values = self._restore_value_codes(values).float()
        scale = scale.float()
        bias = bias.float() if bias is not None else None
        if self.is_grouped:
            batch_size = values.shape[0]
            values = values.reshape(
                batch_size, self.num_key_value_heads, 1, -1, self.group_count, self.group_size
            )
            values = values * scale if self.use_sym else values * scale + bias
            values = values.reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
        else:
            values = values * scale if self.use_sym else values * scale + bias
        if self.use_hadamard:
            values = self._hadamard_value(values, inverse=True)
        if self.use_shuffle:
            values = values.index_select(-1, self.unshuffle_index)
        if self.is_rotary:
            values = self._inverse_rotate_value(values)
        return values

    def dequantize_key(self, values: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self._dequantize_key(values, scale, bias)

    def dequantize_value(self, values: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self._dequantize_value(values, scale, bias)


class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    """Build dynamic three-way split lengths for an ONNX Split node."""

    @staticmethod
    def forward(ctx, values, start, end, dim):
        start_value, end_value = int(start), int(end)
        return torch.tensor(
            [start_value, end_value - start_value, values.shape[dim] - end_value],
            dtype=torch.int64,
        )

    @staticmethod
    def symbolic(graph, values, start, end, dim):
        shape = graph.op("Shape", values)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        return graph.op("Concat", start, graph.op("Sub", end, start), graph.op("Sub", dim_size, end), axis_i=0)


class SLICE_KEEP_MIDDLE(torch.autograd.Function):
    """Select a dynamic middle split while retaining ONNX Split semantics."""

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


def window_split_sizes(values, start, end, dim):
    if dim < 0:
        dim += values.dim()
    return WINDOW_SPLIT_SIZES.apply(values, start, end, dim)


def slice_keep_middle(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SLICE_KEEP_MIDDLE.apply(values, sizes, dim)


class SPLIT_POINT_SIZES(torch.autograd.Function):
    """Build dynamic two-way split lengths for an ONNX Split node."""

    @staticmethod
    def forward(ctx, values, split_at, dim):
        split_value = int(split_at)
        return torch.tensor([split_value, values.shape[dim] - split_value], dtype=torch.int64)

    @staticmethod
    def symbolic(graph, values, split_at, dim):
        shape = graph.op("Shape", values)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        return graph.op("Concat", split_at, graph.op("Sub", dim_size, split_at), axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):
    """Split a dynamic state tensor into immutable prefix and suffix windows."""

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


def split_point_sizes(values, split_at, dim):
    if dim < 0:
        dim += values.dim()
    return SPLIT_POINT_SIZES.apply(values, split_at, dim)


def split_prefix_suffix(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SPLIT_PREFIX_SUFFIX.apply(values, sizes, dim)


class KV_SLICE(torch.nn.Module):
    """Slice LFM attention KV tensors by token range, excluding fixed conv state."""

    def __init__(self, layout: KVCacheLayout):
        super().__init__()
        self.layout = layout

    def forward(self, *all_inputs):
        if len(all_inputs) != self.layout.kv_state_count + 2:
            raise ValueError("KV_SLICE received an unexpected number of state tensors.")
        slice_start, slice_end = all_inputs[-2:]
        outputs = []
        for group in self.layout.kv_groups:
            for layer_index in range(group.count):
                state = all_inputs[self.layout.state_index(group.name, layer_index)]
                sizes = window_split_sizes(state, slice_start, slice_end, group.sequence_axis)
                outputs.append(slice_keep_middle(state, sizes, group.sequence_axis))
        return tuple(outputs)


class KV_SPLIT2(torch.nn.Module):
    """Split LFM attention KV state into prefix and mutable suffix windows."""

    def __init__(self, layout: KVCacheLayout):
        super().__init__()
        self.layout = layout

    def forward(self, *all_inputs):
        if len(all_inputs) != self.layout.kv_state_count + 1:
            raise ValueError("KV_SPLIT2 received an unexpected number of state tensors.")
        split_at = all_inputs[-1]
        prefix, suffix = [], []
        for group in self.layout.kv_groups:
            for layer_index in range(group.count):
                state = all_inputs[self.layout.state_index(group.name, layer_index)]
                sizes = split_point_sizes(state, split_at, group.sequence_axis)
                prefix_state, suffix_state = split_prefix_suffix(state, sizes, group.sequence_axis)
                prefix.append(prefix_state)
                suffix.append(suffix_state)
        return (*prefix, *suffix)


class KV_CONCAT(torch.nn.Module):
    """Concatenate two LFM attention KV windows without touching conv state."""

    def __init__(self, layout: KVCacheLayout):
        super().__init__()
        self.layout = layout

    def forward(self, *all_inputs):
        if len(all_inputs) != self.layout.kv_state_count * 2:
            raise ValueError("KV_CONCAT received an unexpected number of state tensors.")
        prefix = all_inputs[:self.layout.kv_state_count]
        suffix = all_inputs[self.layout.kv_state_count:]
        outputs = []
        for group in self.layout.kv_groups:
            for layer_index in range(group.count):
                state_index = self.layout.state_index(group.name, layer_index)
                outputs.append(torch.cat([prefix[state_index], suffix[state_index]], dim=group.sequence_axis))
        return tuple(outputs)


class KV_ROPE_SHIFT(torch.nn.Module):
    """Shift retained LFM key caches by a text-RoPE offset and requantize if needed."""

    def __init__(self, layout: KVCacheLayout, num_key_value_groups: int, inv_freq, max_seq_len: int):
        super().__init__()
        self.layout = layout
        self.num_key_value_groups = int(num_key_value_groups)
        self.head_dim = layout.head_dim
        self.head_dim_half = layout.head_dim // 2
        self.num_key_value_heads = layout.num_key_value_heads
        self.compute_in_f32 = layout.settings.compute_in_f32
        inv_freq = inv_freq.detach().float().reshape(-1)
        if inv_freq.numel() * 2 != self.head_dim:
            raise ValueError("RoPE inv_freq width is incompatible with LFM head_dim.")
        full_frequency = torch.cat([inv_freq, inv_freq]).view(1, 1, 1, self.head_dim, 1)
        half_sign = torch.cat([
            torch.ones(self.head_dim_half),
            -torch.ones(self.head_dim_half),
        ]).view(1, 1, 1, self.head_dim, 1)
        shifts = torch.arange(max_seq_len + 1, dtype=torch.float32).view(max_seq_len + 1, 1, 1, 1, 1)
        angle = shifts * full_frequency
        angle = angle - 6.283185307179586 * torch.round(angle * (1.0 / 6.283185307179586))
        self.register_buffer("cos_shift", torch.cos(angle).half(), persistent=False)
        self.register_buffer("sin_shift", (torch.sin(angle) * half_sign).half(), persistent=False)
        self.quantizer = (
            LFMKVQuantizer(layout, self.num_key_value_groups).eval()
            if layout.is_quantized
            else None
        )

    def _flip_key(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        values = values.reshape(batch_size, self.num_key_value_heads, 1, 2, self.head_dim_half, -1)
        return values.flip(-3).reshape(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)

    def _shift_key(self, values: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        cache_dtype = values.dtype
        force_f32 = self.compute_in_f32 and cache_dtype != torch.float32
        cosine = self.cos_shift.index_select(0, shift)
        sine = self.sin_shift.index_select(0, shift)
        if cache_dtype == torch.float32 or force_f32:
            cosine = cosine.float()
            sine = sine.float()
        source = values.float() if force_f32 else values
        shifted = source * cosine + self._flip_key(source) * sine
        return shifted.to(cache_dtype) if force_f32 else shifted

    def forward(self, *all_inputs):
        layers = self.layout.num_attention_layers
        if self.layout.is_quantized:
            expected_inputs = layers * (3 if not self.layout.is_symmetric else 2) + 1
        else:
            expected_inputs = layers + 1
        if len(all_inputs) != expected_inputs:
            raise ValueError("KV_ROPE_SHIFT received an unexpected number of key cache tensors.")
        shift = all_inputs[-1].reshape(-1)
        keys = all_inputs[:layers]
        if not self.layout.is_quantized:
            return tuple(self._shift_key(key, shift) for key in keys)

        scales = all_inputs[layers:2 * layers]
        biases = all_inputs[2 * layers:3 * layers] if not self.layout.is_symmetric else None
        output_keys, output_scales, output_biases = [], [], []
        for layer_index, key in enumerate(keys):
            bias = biases[layer_index] if biases is not None else None
            raw_key = self.quantizer.dequantize_key(key, scales[layer_index], bias)
            packed_key, scale, bias = self.quantizer.quantize_key(self._shift_key(raw_key, shift))
            output_keys.append(packed_key)
            output_scales.append(scale)
            if biases is not None:
                output_biases.append(bias)
        return (*output_keys, *output_scales, *output_biases)


def cache_state_dynamic_axes(
    layout: KVCacheLayout,
    state_names: list[str],
    sequence_name: str,
    *,
    batch_name: str = "batch",
    groups: tuple[KVCacheGroup, ...] | None = None,
) -> dict[str, dict[int, str]]:
    """Return dynamic-axis annotations for complete or selected LFM state groups."""
    selected_groups = layout.groups if groups is None else groups
    expected_count = sum(group.count for group in selected_groups)
    if len(state_names) != expected_count:
        raise ValueError("State-name count does not match the selected LFM cache groups.")
    axes = {}
    state_index = 0
    for group in selected_groups:
        for _ in range(group.count):
            name = state_names[state_index]
            axes[name] = {0: batch_name, group.sequence_axis: sequence_name}
            state_index += 1
    return axes


class LFM_MAIN(torch.nn.Module):
    """Fused LFM hybrid decoder with KV and short-convolution recurrent state.

    State order is keys(A), values(A), conv(C), where A is the number of
    full-attention layers and C is the number of short-convolution layers.
    """

    def __init__(
        self,
        backbone,
        config,
        lm_head,
        *,
        kv_settings: KVCacheSettings | None = None,
        use_float16_kv: bool | None = None,
        compute_in_f32: bool | None = None,
        reorder_downproj=True,
        reorder_key="absmean",
    ):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        first_attention = next(layer for layer in backbone.layers if layer.is_attention_layer)
        self.head_dim = int(getattr(first_attention.self_attn, "head_dim", self.hidden_size // self.num_heads))
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("LFM attention geometry does not match hidden_size.")
        self.head_dim_half = self.head_dim // 2
        if self.head_dim % 2:
            raise ValueError("LFM RoPE requires an even attention head dimension.")
        self.num_attn_layers = sum(bool(layer.is_attention_layer) for layer in backbone.layers)
        self.num_conv_layers = len(backbone.layers) - self.num_attn_layers
        if kv_settings is None:
            kv_settings = KVCacheSettings(
                quant_dtype="F16" if use_float16_kv else "F32",
                compute_in_f32=bool(compute_in_f32),
            )
        elif use_float16_kv is not None or compute_in_f32 is not None:
            raise ValueError("Pass either kv_settings or the legacy F16 KV arguments, not both.")
        self.kv_settings, self.kv_notes = normalize_kv_cache_settings(kv_settings, self.head_dim)
        self.kv_layout = KVCacheLayout.create(
            self.kv_settings,
            self.head_dim,
            self.num_key_value_heads,
            self.hidden_size,
            self.num_attn_layers,
            self.num_conv_layers,
        )
        self.use_float16_kv = self.kv_settings.quant_dtype == "F16"
        self.compute_in_f32 = self.kv_settings.compute_in_f32
        self.kv_quantizer = (
            LFMKVQuantizer(self.kv_layout, self.num_key_value_groups).eval()
            if self.kv_layout.is_quantized
            else None
        )
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.total_qkv_heads = self.qk_heads + self.num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, self.num_key_value_heads]
        self.qk_split_sizes = [self.num_heads, self.num_key_value_heads]
        self.o_proj_in_features = self.num_heads * self.head_dim
        self.register_buffer("hidden_norm_scale", torch.ones(self.hidden_size, dtype=torch.float32))
        self.register_buffer("qk_norm_scale", torch.ones(self.head_dim, dtype=torch.float32))
        self.register_buffer("attention_scale", torch.tensor(self.head_dim ** -0.25, dtype=torch.float32))
        self.lm_head = lm_head
        self.reorder_summary = self._fuse_weights(reorder_downproj, reorder_key)

    def _make_linear(self, weight, bias=None):
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
        before = F.linear(F.silu(F.linear(probe, gate_weight, gate_bias)) * F.linear(probe, up_weight, up_bias), down.weight, down.bias)
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
        reordered_pairs = 0
        maximum_error = 0.0
        with torch.no_grad():
            for layer in self.backbone.layers:
                operator_scale = layer.operator_norm.weight.detach().unsqueeze(0)
                if layer.is_attention_layer:
                    attn = layer.self_attn
                    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
                    qkv_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
                    qkv_weight = qkv_weight * operator_scale
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
                        torch.cat([
                            q_weight.repeat(self.num_heads),
                            k_weight.repeat(self.num_key_value_heads),
                        ]).view(1, 1, 1, self.qk_heads, self.head_dim),
                    )
                    attn.operator_norm_epsilon = _norm_epsilon(layer.operator_norm)
                    attn.qk_norm_epsilon = _norm_epsilon(attn.q_layernorm)
                    del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_layernorm, attn.k_layernorm
                else:
                    layer.conv.in_proj.weight.data.mul_(operator_scale)
                    layer.conv.operator_norm_epsilon = _norm_epsilon(layer.operator_norm)

                feed_forward = layer.feed_forward
                ffn_scale = layer.ffn_norm.weight.detach().unsqueeze(0)
                gate = feed_forward.w1
                up = feed_forward.w3
                gate_weight = gate.weight.detach() * ffn_scale
                up_weight = up.weight.detach() * ffn_scale
                gate_bias = gate.bias.detach() if gate.bias is not None else None
                up_bias = up.bias.detach() if up.bias is not None else None
                if reorder_downproj:
                    gate_weight, up_weight, gate_bias, up_bias, error = self._reorder_gated_mlp(
                        layer, gate_weight, up_weight, gate_bias, up_bias, reorder_key
                    )
                    reordered_pairs += 1
                    maximum_error = max(maximum_error, error)
                feed_forward.gate_up = self._make_linear(
                    torch.cat([gate_weight, up_weight], dim=0),
                    torch.cat([gate_bias, up_bias], dim=0) if gate_bias is not None else None,
                )
                feed_forward.mlp_split = [int(gate.out_features), int(up.out_features)]
                feed_forward.norm_epsilon = _norm_epsilon(layer.ffn_norm)
                del feed_forward.w1, feed_forward.w3, layer.operator_norm, layer.ffn_norm

            self.register_buffer(
                "final_norm_scale", self.backbone.embedding_norm.weight.detach().clone()
            )
            self.final_norm_epsilon = _norm_epsilon(self.backbone.embedding_norm)
            del self.backbone.embedding_norm
        return ReorderSummary(reordered_pairs, maximum_error)

    def _rms_norm(self, values, epsilon, scale):
        return simplified_layer_norm(values, scale, epsilon)

    def _rotate_half_qk(self, values, batch_size):
        values = values.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        values = values.flip(-2)
        return values.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        rotary_cos = all_inputs[-3]
        rotary_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0]
        ids_len = hidden_states.shape[1]
        attention_count = 0
        convolution_count = 0
        cache_outputs = {
            group.name: [None] * group.count
            for group in self.kv_layout.groups
        }

        for layer in self.backbone.layers:
            if layer.is_attention_layer:
                attention = layer.self_attn
                normalized = self._rms_norm(
                    hidden_states, attention.operator_norm_epsilon, self.hidden_norm_scale
                )
                qkv = attention.qkv(normalized).reshape(
                    batch_size, -1, 1, self.total_qkv_heads, self.head_dim
                )
                qk, values_current = torch.split(qkv, self.qkv_split_sizes, dim=-2)
                qk = self._rms_norm(qk, attention.qk_norm_epsilon, self.qk_norm_scale)
                qk = qk * attention.qk_norm_weight
                qk = qk * rotary_cos + self._rotate_half_qk(qk, batch_size) * rotary_sin
                queries, keys_current = torch.split(qk, self.qk_split_sizes, dim=-2)
                queries = queries.reshape(
                    batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
                ).permute(0, 2, 3, 1, 4)
                keys_current = keys_current.permute(0, 3, 2, 4, 1)
                values_current = values_current.transpose(1, 3)
                if self.kv_layout.is_quantized:
                    (
                        packed_key, key_scale, key_bias,
                        packed_value, value_scale, value_bias,
                    ) = self.kv_quantizer.quantize_pair(keys_current, values_current)
                    key_cache = torch.cat([
                        all_inputs[self.kv_layout.state_index("key", attention_count)], packed_key
                    ], dim=-1)
                    value_cache = torch.cat([
                        all_inputs[self.kv_layout.state_index("value", attention_count)], packed_value
                    ], dim=-2)
                    key_scale_cache = torch.cat([
                        all_inputs[self.kv_layout.state_index("key_scale", attention_count)], key_scale
                    ], dim=self.kv_layout.sequence_axis("key_scale"))
                    value_scale_cache = torch.cat([
                        all_inputs[self.kv_layout.state_index("value_scale", attention_count)], value_scale
                    ], dim=self.kv_layout.sequence_axis("value_scale"))
                    cache_outputs["key"][attention_count] = key_cache
                    cache_outputs["value"][attention_count] = value_cache
                    cache_outputs["key_scale"][attention_count] = key_scale_cache
                    cache_outputs["value_scale"][attention_count] = value_scale_cache
                    if self.kv_layout.is_symmetric:
                        key_bias_cache = None
                        value_bias_cache = None
                    else:
                        key_bias_cache = torch.cat([
                            all_inputs[self.kv_layout.state_index("key_bias", attention_count)], key_bias
                        ], dim=self.kv_layout.sequence_axis("key_bias"))
                        value_bias_cache = torch.cat([
                            all_inputs[self.kv_layout.state_index("value_bias", attention_count)], value_bias
                        ], dim=self.kv_layout.sequence_axis("value_bias"))
                        cache_outputs["key_bias"][attention_count] = key_bias_cache
                        cache_outputs["value_bias"][attention_count] = value_bias_cache
                    key_cache = self.kv_quantizer.dequantize_key(key_cache, key_scale_cache, key_bias_cache)
                    value_cache = self.kv_quantizer.dequantize_value(value_cache, value_scale_cache, value_bias_cache)
                elif self.use_float16_kv:
                    if not self.compute_in_f32:
                        queries = queries.half()
                    keys_current = keys_current.half()
                    values_current = values_current.half()
                if not self.kv_layout.is_quantized:
                    key_cache = torch.cat([
                        all_inputs[self.kv_layout.state_index("key", attention_count)], keys_current
                    ], dim=-1)
                    value_cache = torch.cat(
                        [all_inputs[self.kv_layout.state_index("value", attention_count)], values_current], dim=-2
                    )
                    cache_outputs["key"][attention_count] = key_cache
                    cache_outputs["value"][attention_count] = value_cache
                    if self.use_float16_kv and self.compute_in_f32:
                        key_cache = key_cache.float()
                        value_cache = value_cache.float()
                attention_mask_current = (
                    attention_mask.half()
                    if self.use_float16_kv and not self.compute_in_f32
                    else attention_mask
                )
                attention_scores = torch.matmul(queries, key_cache) + attention_mask_current
                attention_scores = torch.softmax(attention_scores, dim=-1)
                attention_output = torch.matmul(attention_scores, value_cache)
                if self.use_float16_kv and not self.compute_in_f32:
                    attention_output = attention_output.float()
                attention_output = attention_output.permute(0, 3, 1, 2, 4).reshape(
                    batch_size, -1, self.o_proj_in_features
                )
                operator_output = attention.out_proj(attention_output)
                attention_count += 1
            else:
                normalized = self._rms_norm(
                    hidden_states, layer.conv.operator_norm_epsilon, self.hidden_norm_scale
                )
                bcx = layer.conv.in_proj(normalized).transpose(-1, -2)
                b_values, c_values, x_values = bcx.chunk(3, dim=-2)
                bx = b_values * x_values
                state = torch.cat(
                    [all_inputs[self.kv_layout.state_index("conv", convolution_count)].float(), bx], dim=-1
                )
                cache_outputs["conv"][convolution_count] = state[..., -2:].half() if self.use_float16_kv else state[..., -2:]
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
        logits = self.lm_head(last)
        state_outputs = []
        for group in self.kv_layout.groups:
            state_outputs.extend(cache_outputs[group.name])
        return (*state_outputs, logits)


def build_cache_io(
    main: LFM_MAIN,
    batch_size: int,
    *,
    cache_dtype: torch.dtype | None = None,
):
    """Build typed zero-length recurrent-state inputs and matching ONNX names."""
    if cache_dtype is not None and cache_dtype != main.kv_layout.cache_dtype:
        raise ValueError(
            f"cache_dtype {cache_dtype} conflicts with KV_QUANT_DTYPE {main.kv_settings.quant_dtype}."
        )
    tensors = []
    input_names = []
    output_names = []
    dynamic_axes = {}

    def add(group, tensor, sequence_axis, count):
        for index in range(count):
            input_name = f"in_{group}_{index}"
            output_name = f"out_{group}_{index}"
            tensors.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: "batch", sequence_axis: "history_len"}
            dynamic_axes[output_name] = {0: "batch", sequence_axis: "kv_seq_len"}

    for group in main.kv_layout.groups:
        add(group.name, main.kv_layout.zeros(group.name, batch_size), group.sequence_axis, group.count)
    return tensors, input_names, output_names, dynamic_axes