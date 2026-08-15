import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import onnx
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
# transformers==4.46.3


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

EXPORT_DIR = os.path.join(SCRIPT_DIR, 'UnlimitedOCR_ONNX')
EXPORT_STAGING_DIR = EXPORT_DIR + '.staging'

CHECKPOINT_DIR                 = Path.home() / 'Downloads' / 'Unlimited-OCR'
# Backward-compatible configuration alias.
download_path                  = str(CHECKPOINT_DIR)

# Export controls
DO_EXPORT                = True                    # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW     = False                   # Prevent float16 overflow for Q4F16, Q8F16, or F16 quantization.
STOP_TOKEN               = [1]                     # Unlimited-OCR (DeepSeek-OCR) end-of-sentence token id.
MAX_SEQ_LEN              = 4096                    # Fixed maximum context length after export.
SELF_TEST_MAX_NEW_TOKENS = 4096                    # Minimum generated length for the exporter smoke test.

# Quantization-oriented model reordering
# Exact channel permutations keep quantization blocks homogeneous without changing the float function.
REORDER_DOWNPROJ_FOR_QUANT   = True                # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True                # Reorder vision MLP channels before quantization.
REORDER_KEY                  = "absmean"           # Channel statistic: absmean | L4 | rms | std.

# Image input and vision tracing
# DeepSeek-OCR Base mode uses one global view and no dynamic crop tiling.
BASE_SIZE        = 1024                            # Global view resolution fed to the DeepEncoder.
PATCH_SIZE       = 16                              # SAM patch size.
DOWNSAMPLE_RATIO = 4                               # SAM neck downsample ratio before tokenization.
INPUT_IMAGE_SIZE = [BASE_SIZE, BASE_SIZE]          # Letterbox-padded input shape before ONNX preprocessing.
INPUT_IMAGE_DIM  = 5                               # 4=[B, C, H, W]; 5=[B, 1, C, H, W].
IMAGE_TOKEN_ID   = 128815                          # DeepSeek-OCR image placeholder id.
CLIP_IMAGE_MEAN  = [0.5, 0.5, 0.5]                 # Image normalization mean.
CLIP_IMAGE_STD   = [0.5, 0.5, 0.5]                 # Image normalization standard deviation.

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                         # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 128                          # Quantization group width; must divide head_dim evenly.
COMPUTE_IN_F32      = False                        # F16 KV only: False keeps attention in F16, True upcasts cache reads.

# KV quantization transforms and parameters
USE_HADAMARD           = False                     # Apply randomized Walsh-Hadamard mixing before quantization.
HADAMARD_RANDOM_SEED   = 9527                      # Seed for the deterministic Hadamard sign pattern.
USE_CLIP               = False                     # Clip quantization blocks to CLIP_SIGMA standard deviations.
CLIP_SIGMA             = 3.0                       # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False                     # Interleave channels across quantization groups.
USE_SYM                = True                      # True=symmetric absmax; False=asymmetric min-max with bias.
USE_FLOAT16_SCALE_BIAS = True                      # Store quantization scales and biases as float16.

# ONNX graph format
OPSET = 20                                         # ONNX opset version

# Runtime-visible names are a metadata contract.  UnlimitedOCR is image-only,
# so its merged bundle intentionally contains no text-only or video recipes.
MODEL_FILE_NAMES = {
    'metadata': 'LLM_Metadata.onnx',
    'embed': 'LLM_Embed.onnx',
    'image_preprocess': 'LLM_Image_Preprocess.onnx',
    'vision': 'LLM_Vision.onnx',
    'concat_image': 'LLM_Concat_Image.onnx',
    'rotary_image_prefill': 'LLM_Rotary_Image_Prefill.onnx',
    'rotary_image_decode': 'LLM_Rotary_Image_Decode.onnx',
    'main': 'LLM_Main.onnx',
    'greedy': 'LLM_Greedy.onnx',
    'penalty_greedy': 'LLM_PenaltyGreedy.onnx',
    'sampling': 'LLM_TopKTopPSampling.onnx',
    'kv_slice': 'LLM_KV_Slice.onnx',
    'kv_split2': 'LLM_KV_Split2.onnx',
    'kv_concat': 'LLM_KV_Concat.onnx',
    'rope_shift': 'LLM_RopeShift.onnx',
    'image_prefill_greedy': 'LLM_ImagePrefillGreedy.onnx',
    'image_prefill_penalty_greedy': 'LLM_ImagePrefillPenaltyGreedy.onnx',
    'image_prefill_sampling': 'LLM_ImagePrefillSampling.onnx',
    'image_decode_greedy': 'LLM_ImageDecodeGreedy.onnx',
    'image_decode_penalty_greedy': 'LLM_ImageDecodePenaltyGreedy.onnx',
    'image_decode_sampling': 'LLM_ImageDecodeSampling.onnx',
    'shared_initializers': 'LLM_SharedInitializers.onnx',
}
MODEL_FILE_NAMES['shared_initializers_data'] = (
    MODEL_FILE_NAMES['shared_initializers'] + '.data'
)
RUNTIME_MODEL_FILE_ROLES = (
    'image_preprocess',
    'vision',
    'shared_initializers',
    'shared_initializers_data',
    'kv_slice',
    'kv_split2',
    'kv_concat',
    'rope_shift',
    'image_prefill_greedy',
    'image_prefill_penalty_greedy',
    'image_prefill_sampling',
    'image_decode_greedy',
    'image_decode_penalty_greedy',
    'image_decode_sampling',
)
MODEL_FILE_NAME_METADATA = {
    f'model_file_name_{key}': MODEL_FILE_NAMES[key]
    for key in RUNTIME_MODEL_FILE_ROLES
}


# NOTE ON THE SLIDING WINDOW: the reference Unlimited-OCR (DeepSeek-OCR) uses a W=128 ring-buffer sliding
# window during decode. This export uses a standard growing (full-causal) KV cache within MAX_SEQ_LEN, which
# is simpler for ONNX and a strict superset of the windowed attention. Outputs match the reference while the
# generated length stays within the window; longer generations attend to more context than the ring buffer.


SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
    "F16", "F32"
)


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
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
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim."
            )
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}."
            )
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(
                f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization."
            )

        if KV_QUANT_DTYPE in q8_kv and KV_QUANT_GROUP_SIZE == head_dim and (USE_HADAMARD or USE_SHUFFLE):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block."
            )
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


class GREEDY_SEARCH(torch.nn.Module):
    """Token-only greedy strategy used by merged decode graphs."""

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
        token = torch.argmax(adjusted_logits, dim=-1, keepdim=True).int()
        return token, torch.cat([previous_ids, token], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """TopTok sampling with sign-aware repetition penalty."""

    @staticmethod
    def _sample(scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = (cumulative - probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative >= threshold).int(), dim=-1, keepdim=True)
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


class METADATA_CARRIER(torch.nn.Module):
    """Small identity graph used to load metadata before large sessions."""

    def forward(self, marker):
        return marker


class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, and ROTARY_Q4.

    Ported from the FireRedOCR export. Operates on generic key/value tensors of shape
    key: (B, KVH, 1, head_dim, S) and value: (B, KVH, 1, S, head_dim), so it applies
    unchanged to Unlimited-OCR's multi-head attention (num_kv_groups == 1).
    """

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, is_q4=False, is_rotary=False, is_q8_cuda=False, use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=2.5, use_shuffle=False):
        super().__init__()
        self.is_rotary     = is_rotary
        self.is_q4         = is_q4
        self.is_q8_cuda    = is_q8_cuda
        self.use_sym       = use_sym
        self.use_hadamard  = use_hadamard
        self.use_clip      = use_clip
        self.clip_sigma    = clip_sigma
        self.use_shuffle   = use_shuffle
        self.use_residual_bias_correction = not use_sym
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads  = num_kv_heads
        self.num_kv_groups = num_kv_groups

        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX        = float(self.SIGNED_QMAX)
            self.ZERO_POINT  = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX        = 15.0 if is_q4 else 255.0
            self.ZERO_POINT  = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        if is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,),  inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[:head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        if self.use_shuffle:
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
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
            x = x[..., :self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    def _clip_to_sigma(self, x, dim):
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def _flip_k(self, k, batch_size):
        return k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1).flip(-3).view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _flip_v(self, v, batch_size):
        return v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def _flip_q(self, q, batch_size):
        return q.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

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
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

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
            scale     = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        x_quant = torch.round(x / scale).clamp(self.SIGNED_QMIN, self.SIGNED_QMAX).to(torch.int32)
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
            scale  = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale        = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed     = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim, batch_size):
        if self.use_sym:
            if dim == -2:
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax   = x.abs().amax(dim=-1, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

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
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def quantize_key(self, keys, batch_size):
        """Quantize a full-attention key tensor without requiring a value tensor."""
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
        """Restore a key tensor to the RoPE coordinate system for cache shifting."""
        if USE_FLOAT16_SCALE_BIAS:
            key_scale = key_scale.float()
            if key_bias is not None:
                key_bias = key_bias.float()
        if self.is_q8_cuda:
            unpack_head_dim = (
                self.head_dim // 2 if self.is_q4 else self.head_dim
            )
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
            keys   = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys   = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# KV cache helpers operate only on the full-attention cache tensors. They keep
# the state ordering emitted by _build_kv_layout for every storage dtype.
class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    """Compute [start, window, tail] Split sizes along one cache sequence axis."""

    @staticmethod
    def forward(ctx, ref, start, end, dim):
        start_value = int(start)
        end_value = int(end)
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
    """Select the middle tensor from a dynamic three-way Split."""

    @staticmethod
    def forward(ctx, x, sizes, dim):
        start = int(sizes[0])
        end = start + int(sizes[1])
        index = [slice(None)] * x.dim()
        index[dim] = slice(start, end)
        return x[tuple(index)].clone()

    @staticmethod
    def symbolic(g, x, sizes, dim):
        return g.op("Split", x, sizes, axis_i=dim, outputs=3)[1]


def window_split_sizes(ref, start, end, dim):
    if dim < 0:
        dim += ref.dim()
    return WINDOW_SPLIT_SIZES.apply(ref, start, end, dim)


def slice_keep_middle(x, sizes, dim):
    if dim < 0:
        dim += x.dim()
    return SLICE_KEEP_MIDDLE.apply(x, sizes, dim)


class SPLIT_POINT_SIZES(torch.autograd.Function):
    """Compute [prefix, suffix] Split sizes along one cache sequence axis."""

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
        tail = g.op("Sub", dim_size, split_at)
        return g.op("Concat", split_at, tail, axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):
    """Split a cache tensor into prefix and suffix tensors."""

    @staticmethod
    def forward(ctx, x, sizes, dim):
        split_value = int(sizes[0])
        prefix_index = [slice(None)] * x.dim()
        suffix_index = [slice(None)] * x.dim()
        prefix_index[dim] = slice(None, split_value)
        suffix_index[dim] = slice(split_value, None)
        return x[tuple(prefix_index)].clone(), x[tuple(suffix_index)].clone()

    @staticmethod
    def symbolic(g, x, sizes, dim):
        return g.op("Split", x, sizes, axis_i=dim, outputs=2)


def split_point_sizes(ref, split_at, dim):
    if dim < 0:
        dim += ref.dim()
    return SPLIT_POINT_SIZES.apply(ref, split_at, dim)


def split_prefix_suffix(x, sizes, dim):
    if dim < 0:
        dim += x.dim()
    return SPLIT_PREFIX_SUFFIX.apply(x, sizes, dim)


class _KV_CACHE_HELPER(torch.nn.Module):
    """Shared layout facts for cache-only helper graphs."""

    def __init__(self, num_layers, head_dim):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in {
            "Q8",
            "Q8_CUDA",
            "ROTARY_Q8",
            "ROTARY_Q8_CUDA",
            "ROTARY_Q4",
            "ROTARY_Q4_CUDA",
        }
        self.kv_grouped_6d = (
            KV_QUANT_DTYPE in {"ROTARY_Q4", "ROTARY_Q4_CUDA"}
            or (
                KV_QUANT_DTYPE
                in {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
                and (USE_HADAMARD or USE_SHUFFLE)
                and KV_QUANT_GROUP_SIZE < head_dim
            )
        )
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5


class KV_SLICE(_KV_CACHE_HELPER):
    """Keep the [slice_start:slice_end] window from each KV cache tensor."""

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end = all_inputs[-1]
        sizes = window_split_sizes(all_inputs[0], slice_start, slice_end, -1)
        saved_key = []
        saved_value = []
        saved_key_scale = []
        saved_key_bias = []
        saved_value_scale = []
        saved_value_bias = []
        for index in range(self.num_layers):
            saved_key.append(slice_keep_middle(all_inputs[index], sizes, -1))
            saved_value.append(
                slice_keep_middle(all_inputs[index + self.num_layers], sizes, -2)
            )
            if self.kv_quantized:
                saved_key_scale.append(
                    slice_keep_middle(all_inputs[index + self.num_layers_2], sizes, -1)
                )
                if not self.kv_sym:
                    saved_key_bias.append(
                        slice_keep_middle(all_inputs[index + self.num_layers_3], sizes, -1)
                    )
                value_scale_offset = (
                    self.num_layers_3 if self.kv_sym else self.num_layers_4
                )
                value_bias_offset = self.num_layers_5
                value_dim = -3 if self.kv_grouped_6d else -2
                saved_value_scale.append(
                    slice_keep_middle(all_inputs[index + value_scale_offset], sizes, value_dim)
                )
                if not self.kv_sym:
                    saved_value_bias.append(
                        slice_keep_middle(all_inputs[index + value_bias_offset], sizes, value_dim)
                    )
        if self.kv_sym:
            return *saved_key, *saved_value, *saved_key_scale, *saved_value_scale
        if self.kv_quantized:
            return (
                *saved_key,
                *saved_value,
                *saved_key_scale,
                *saved_key_bias,
                *saved_value_scale,
                *saved_value_bias,
            )
        return *saved_key, *saved_value


class KV_SPLIT2(_KV_CACHE_HELPER):
    """Split each KV cache tensor into prefix and suffix cache blocks."""

    def forward(self, *all_inputs):
        split_at = all_inputs[-1]
        sizes = split_point_sizes(all_inputs[0], split_at, -1)
        prefix_key = []
        prefix_value = []
        prefix_key_scale = []
        prefix_key_bias = []
        prefix_value_scale = []
        prefix_value_bias = []
        suffix_key = []
        suffix_value = []
        suffix_key_scale = []
        suffix_key_bias = []
        suffix_value_scale = []
        suffix_value_bias = []
        for index in range(self.num_layers):
            first, second = split_prefix_suffix(all_inputs[index], sizes, -1)
            prefix_key.append(first)
            suffix_key.append(second)
            first, second = split_prefix_suffix(
                all_inputs[index + self.num_layers], sizes, -2
            )
            prefix_value.append(first)
            suffix_value.append(second)
            if self.kv_quantized:
                first, second = split_prefix_suffix(
                    all_inputs[index + self.num_layers_2], sizes, -1
                )
                prefix_key_scale.append(first)
                suffix_key_scale.append(second)
                if not self.kv_sym:
                    first, second = split_prefix_suffix(
                        all_inputs[index + self.num_layers_3], sizes, -1
                    )
                    prefix_key_bias.append(first)
                    suffix_key_bias.append(second)
                value_scale_offset = (
                    self.num_layers_3 if self.kv_sym else self.num_layers_4
                )
                value_bias_offset = self.num_layers_5
                value_dim = -3 if self.kv_grouped_6d else -2
                first, second = split_prefix_suffix(
                    all_inputs[index + value_scale_offset], sizes, value_dim
                )
                prefix_value_scale.append(first)
                suffix_value_scale.append(second)
                if not self.kv_sym:
                    first, second = split_prefix_suffix(
                        all_inputs[index + value_bias_offset], sizes, value_dim
                    )
                    prefix_value_bias.append(first)
                    suffix_value_bias.append(second)
        if self.kv_sym:
            return (
                *prefix_key,
                *prefix_value,
                *prefix_key_scale,
                *prefix_value_scale,
                *suffix_key,
                *suffix_value,
                *suffix_key_scale,
                *suffix_value_scale,
            )
        if self.kv_quantized:
            return (
                *prefix_key,
                *prefix_value,
                *prefix_key_scale,
                *prefix_key_bias,
                *prefix_value_scale,
                *prefix_value_bias,
                *suffix_key,
                *suffix_value,
                *suffix_key_scale,
                *suffix_key_bias,
                *suffix_value_scale,
                *suffix_value_bias,
            )
        return *prefix_key, *prefix_value, *suffix_key, *suffix_value


class KV_CONCAT(_KV_CACHE_HELPER):
    """Concatenate two full KV cache blocks in the exporter state order."""

    def forward(self, *all_inputs):
        block_size = len(all_inputs) // 2
        prefix = all_inputs[:block_size]
        suffix = all_inputs[block_size:]
        saved_key = []
        saved_value = []
        saved_key_scale = []
        saved_key_bias = []
        saved_value_scale = []
        saved_value_bias = []
        for index in range(self.num_layers):
            saved_key.append(torch.cat([prefix[index], suffix[index]], dim=-1))
            saved_value.append(
                torch.cat(
                    [
                        prefix[index + self.num_layers],
                        suffix[index + self.num_layers],
                    ],
                    dim=-2,
                )
            )
            if self.kv_quantized:
                saved_key_scale.append(
                    torch.cat(
                        [
                            prefix[index + self.num_layers_2],
                            suffix[index + self.num_layers_2],
                        ],
                        dim=-1,
                    )
                )
                if not self.kv_sym:
                    saved_key_bias.append(
                        torch.cat(
                            [
                                prefix[index + self.num_layers_3],
                                suffix[index + self.num_layers_3],
                            ],
                            dim=-1,
                        )
                    )
                value_scale_offset = (
                    self.num_layers_3 if self.kv_sym else self.num_layers_4
                )
                value_bias_offset = self.num_layers_5
                value_dim = -3 if self.kv_grouped_6d else -2
                saved_value_scale.append(
                    torch.cat(
                        [
                            prefix[index + value_scale_offset],
                            suffix[index + value_scale_offset],
                        ],
                        dim=value_dim,
                    )
                )
                if not self.kv_sym:
                    saved_value_bias.append(
                        torch.cat(
                            [
                                prefix[index + value_bias_offset],
                                suffix[index + value_bias_offset],
                            ],
                            dim=value_dim,
                        )
                    )
        if self.kv_sym:
            return *saved_key, *saved_value, *saved_key_scale, *saved_value_scale
        if self.kv_quantized:
            return (
                *saved_key,
                *saved_value,
                *saved_key_scale,
                *saved_key_bias,
                *saved_value_scale,
                *saved_value_bias,
            )
        return *saved_key, *saved_value


def build_rope_shift_tables(head_dim, rope_theta, max_shift):
    """Build standard flip-RoPE tables that move retained cached keys left."""
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    shifts = torch.arange(max_shift + 1, dtype=torch.float32)
    frequencies = torch.outer(shifts, inv_freq)
    cos_half = frequencies.cos()
    sin_half = frequencies.sin()
    cos = torch.cat([cos_half, cos_half], dim=-1)
    # Retained cached keys change from R(position) to R(position - shift).
    sin = torch.cat([sin_half, -sin_half], dim=-1)
    return (
        cos.half().view(max_shift + 1, 1, 1, head_dim, 1),
        sin.half().view(max_shift + 1, 1, 1, head_dim, 1),
    )


class ROPE_SHIFT(torch.nn.Module):
    """Apply a standard RoPE position shift to F16 or F32 cached keys."""

    def __init__(self, num_layers, num_kv_heads, head_dim, rope_theta, max_shift):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.compute_in_f32 = COMPUTE_IN_F32
        cos_shift, sin_shift = build_rope_shift_tables(
            head_dim, rope_theta, max_shift
        )
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, key):
        batch_size = key.shape[0]
        key = key.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            2,
            self.head_dim_half,
            -1,
        )
        key = key.flip(-3)
        return key.reshape(
            batch_size, self.num_kv_heads, 1, self.head_dim, -1
        )

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
            shifted = key * cos_tab + self._flip_k(key) * sin_tab
            if force_f32:
                shifted = shifted.to(kv_dtype)
            outputs.append(shifted)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):
    """Dequantize, RoPE-shift, and re-quantize cached keys for quantized KV modes."""

    def __init__(self, num_layers, num_kv_heads, head_dim, rope_theta, max_shift,
                 quantizer, is_asymmetric):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.quantizer = quantizer
        self.is_asymmetric = is_asymmetric
        cos_shift, sin_shift = build_rope_shift_tables(
            head_dim, rope_theta, max_shift
        )
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, key):
        batch_size = key.shape[0]
        key = key.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            2,
            self.head_dim_half,
            -1,
        )
        key = key.flip(-3)
        return key.reshape(
            batch_size, self.num_kv_heads, 1, self.head_dim, -1
        )

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cos_tab = self.cos_shift.index_select(0, shift).squeeze(0).float()
        sin_tab = self.sin_shift.index_select(0, shift).squeeze(0).float()
        keys = all_inputs[:self.num_layers]
        scales = all_inputs[self.num_layers:self.num_layers * 2]
        biases = (
            all_inputs[self.num_layers * 2:self.num_layers * 3]
            if self.is_asymmetric
            else None
        )

        output_keys = []
        output_scales = []
        output_biases = []
        for layer_index in range(self.num_layers):
            key_bias = biases[layer_index] if self.is_asymmetric else None
            key = self.quantizer.dequantize_key(
                keys[layer_index],
                scales[layer_index],
                key_bias,
                keys[layer_index].shape[0],
            )
            key = key * cos_tab + self._flip_k(key) * sin_tab
            packed_key, key_scale, new_key_bias = self.quantizer.quantize_key(
                key, key.shape[0]
            )
            output_keys.append(packed_key)
            output_scales.append(key_scale)
            if self.is_asymmetric:
                output_biases.append(new_key_bias)
        if self.is_asymmetric:
            return *output_keys, *output_scales, *output_biases
        return *output_keys, *output_scales


class LLM_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Letterbox-resize raw image tensors and apply the native DeepEncoder normalization."""

    def __init__(self, image_size):
        super().__init__()
        self.image_size = tuple(int(value) for value in image_size)
        self.register_buffer(
            'image_mean', torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            'image_std', torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        pixel_values = F.interpolate(
            pixel_values.float(), size=list(self.image_size), mode='bilinear', align_corners=False
        )
        return (pixel_values / 255.0 - self.image_mean) / self.image_std


class LLM_VISION(torch.nn.Module):
    """Run UnlimitedOCR's SAM, CLIP, and linear projector image path."""

    def __init__(self, llm, base_size):
        super().__init__()
        self.deep = llm.model
        self.sam_model = self.deep.sam_model.float()
        self.vision_model = self.deep.vision_model.float()
        self.projector = self.deep.projector.float()
        self._replace_gelu_with_tanh_approximation(self.sam_model)
        self.grid = int(base_size) // PATCH_SIZE // DOWNSAMPLE_RATIO
        self.image_token_length = self.grid * (self.grid + 1) + 1
        self.vision_reorder_mlp_pair_count = 0
        if REORDER_VISION_MLP_FOR_QUANT:
            self._reorder_mlp_for_quant(REORDER_KEY)
        self.register_buffer(
            'image_newline', self.deep.image_newline.detach().float().clone(), persistent=False
        )
        self.register_buffer(
            'view_seperator', self.deep.view_seperator.detach().float().clone(), persistent=False
        )

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                LLM_VISION._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _channel_statistic(weight, key):
        absolute = weight.abs()
        if key == 'rms':
            return (weight * weight).mean(0).sqrt()
        if key == 'L4':
            return absolute.pow(4).mean(0).pow(0.25)
        if key == 'std':
            return weight.std(0)
        if key != 'absmean':
            raise ValueError(f'Unsupported REORDER_KEY: {key!r}.')
        return absolute.mean(0)

    @classmethod
    def _reorder_mlp_pair(cls, producer, consumer, key):
        if producer.weight.shape[0] != consumer.weight.shape[1]:
            raise ValueError('Vision MLP producer/consumer intermediate dimensions do not match.')
        permutation = torch.argsort(cls._channel_statistic(consumer.weight.data, key))
        if torch.unique(permutation).numel() != permutation.numel():
            raise RuntimeError('Vision MLP reorder permutation is not bijective.')
        producer.weight.data.copy_(producer.weight.data[permutation])
        if producer.bias is not None:
            producer.bias.data.copy_(producer.bias.data[permutation])
        consumer.weight.data.copy_(consumer.weight.data[:, permutation])

    def _reorder_mlp_for_quant(self, key):
        with torch.no_grad():
            for block in self.sam_model.blocks:
                self._reorder_mlp_pair(block.mlp.lin1, block.mlp.lin2, key)
                self.vision_reorder_mlp_pair_count += 1
            for layer in self.vision_model.transformer.layers:
                self._reorder_mlp_pair(layer.mlp.fc1, layer.mlp.fc2, key)
                self.vision_reorder_mlp_pair_count += 1

    def forward(self, pixel_values):
        sam_features = self.sam_model(pixel_values.float())
        clip_features = self.vision_model(pixel_values.float(), sam_features)
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)
        features = self.projector(torch.cat((clip_features[:, 1:], sam_flat), dim=-1))
        batch_size, _, hidden_size = features.shape
        features = features.view(batch_size, self.grid, self.grid, hidden_size)
        newline = self.image_newline.view(1, 1, 1, hidden_size).expand(batch_size, self.grid, 1, hidden_size)
        features = torch.cat([features, newline], dim=2).reshape(batch_size, -1, hidden_size)
        return torch.cat([features, self.view_seperator.view(1, 1, hidden_size).expand(batch_size, 1, hidden_size)], dim=1)


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the native contiguous image-token span in dynamic text embeddings."""

    def __init__(self, image_token_id, image_token_length):
        super().__init__()
        self.image_token_id = int(image_token_id)
        self.image_token_length = int(image_token_length)

    def forward(self, input_ids, text_hidden_states, vision_hidden_states):
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(text_hidden_states)
        return text_hidden_states.masked_scatter(image_mask, vision_hidden_states)


class ROTARY_PREFILL(torch.nn.Module):
    """Precompute standard Llama RoPE tables (flip form) and the causal mask for prefill."""

    def __init__(self, head_dim, rope_theta, max_seq_len):
        super().__init__()
        total_max = max_seq_len
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128,
            persistent=False,
        )
        cos_buf, sin_buf = self._build_rotary_table(head_dim, rope_theta, total_max)
        self.register_buffer('cos_rotary_pos_emb', cos_buf, persistent=False)
        self.register_buffer('sin_rotary_pos_emb', sin_buf, persistent=False)

    @staticmethod
    def _build_rotary_table(head_dim, rope_theta, total_max):
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(total_max, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                        # (total_max, head_dim/2)
        cos_half = freqs.cos()
        sin_half = freqs.sin()
        # flip-RoPE buffers: cos = [cos, cos], sin = [-sin, sin]; paired with a view/flip rotate_half.
        cos_buf = torch.cat([cos_half, cos_half], dim=-1).half().view(1, total_max, 1, 1, head_dim)
        sin_buf = torch.cat([-sin_half, sin_half], dim=-1).half().view(1, total_max, 1, 1, head_dim)
        return cos_buf, sin_buf

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_DECODE(torch.nn.Module):
    """Provide standard Llama RoPE embeddings (flip form) for a single decode step."""

    def __init__(self, head_dim, rope_theta, max_seq_len):
        super().__init__()
        cos_buf, sin_buf = ROTARY_PREFILL._build_rotary_table(head_dim, rope_theta, max_seq_len)
        self.register_buffer('cos_rotary_pos_emb', cos_buf, persistent=False)
        self.register_buffer('sin_rotary_pos_emb', sin_buf, persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len:kv_seq_len_next].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len:kv_seq_len_next].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Export ORT's fused RMS normalization with float32 accumulation."""

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance = x.float().square().mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        output = g.op(
            'SimplifiedLayerNormalization',
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
    """DeepseekV2 decoder stack: fused-QKV multi-head attention (KV cache with optional quantization)
    plus a dense MLP (layer 0) / batched MoE (layers >= first_k_dense_replace) feed-forward.

    Optimizations ported from FireRedOCR: sum-based RMSNorm with absorbed scale, fused QKV in one GEMM,
    attention-scale folded into Q/K projection rows, flip-form RoPE, F16 KV cache, and the KVQuantizer
    framework. The MoE is expressed as a dense batched einsum over all experts (top-k gating via a scatter)
    so it maps to portable ONNX ops.
    """

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size,
                 first_k_dense, n_experts, moe_top_k, routed_scaling, norm_topk):
        super().__init__()
        self.llm = llm

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim             = head_dim
        self.head_dim_half        = head_dim // 2
        self.head_dim_quarter     = head_dim // 4
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads             = num_heads + num_key_value_heads
        self.total_qkv_heads      = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes      = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes       = [num_heads, num_key_value_heads]

        # ── Layer-count multipliers (for indexing into the flat KV input list) ──
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # ── MoE geometry ─────────────────────────────────────────────────
        self.first_k_dense  = first_k_dense
        self.n_experts      = n_experts
        self.moe_top_k      = moe_top_k
        self.routed_scaling = routed_scaling
        self.norm_topk      = norm_topk

        # ── KV cache dtype flags ─────────────────────────────────────────
        self.kv_f16             = (KV_QUANT_DTYPE == "F16")
        self.compute_in_f32     = COMPUTE_IN_F32
        self.kv_q8              = (KV_QUANT_DTYPE == "Q8")
        self.kv_q8_cuda         = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.kv_rotary_q8       = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4       = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q8_CUDA")
        self.kv_rotary_q4_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q4_CUDA")
        self.kv_rotary_cuda     = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary          = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized       = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized   = self.kv_quantized or self.kv_rotary
        self.kv_sym             = USE_SYM and self.kv_any_quantized
        self.kv_q8_grouped      = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_unpack_head_dim = (head_dim // 2) if self.kv_rotary_q4_cuda else head_dim
        self.kv_pack_quarter    = (head_dim // 8) if self.kv_rotary_q4_cuda else (head_dim // 4)

        # ── Quantizer & fused RMS normalization ──────────────────────────
        self.quantizer = KVQuantizer(
            head_dim=head_dim,
            num_kv_heads=num_key_value_heads,
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
        hidden_rms_norm = self.llm.model.layers[0].input_layernorm
        self.hidden_rms_norm_eps = float(
            getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6))
        )
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
        )

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_any_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self.language_reorder_pair_count = 0
        self._fuse_weights(hidden_size)
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        self.o_proj_in_features = self.llm.model.layers[0].self_attn.o_proj.in_features

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        scale_factor = self.head_dim ** -0.25
        norm_factor  = hidden_size ** 0.5
        self.dense_inter = None

        with torch.no_grad():
            self.llm.lm_head.float()

            for i, layer in enumerate(self.llm.model.layers):
                layer.self_attn.o_proj.float()
                self._fuse_qkv_projection(layer, scale_factor, norm_factor)
                if i < self.first_k_dense:
                    self._fuse_dense_mlp(layer, i, norm_factor)
                else:
                    self._fuse_moe(layer, i, norm_factor)

            self.register_buffer(
                'final_norm_scale',
                self.llm.model.norm.weight.detach().float().unsqueeze(0) * norm_factor,
            )
            del self.llm.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor):
        """Fuse Q, K, V into one Linear, fold the attention scale into Q/K rows, absorb input RMSNorm."""
        attn = layer.self_attn
        q_w = attn.q_proj.weight.data.float()
        k_w = attn.k_proj.weight.data.float()
        v_w = attn.v_proj.weight.data.float()
        in_features = int(q_w.shape[1])
        out_features = int(q_w.shape[0] + k_w.shape[0] + v_w.shape[0])

        qkv = torch.nn.Linear(in_features, out_features, bias=False)
        input_norm = layer.input_layernorm.weight.data.float().unsqueeze(0) * norm_factor
        fused = torch.cat([q_w * scale_factor, k_w * scale_factor, v_w], dim=0)     # scale Q/K output rows
        qkv.weight.copy_(fused * input_norm)                                        # absorb input RMSNorm columns
        attn.qkv = qkv
        del attn.q_proj, attn.k_proj, attn.v_proj, layer.input_layernorm

    def _fuse_dense_mlp(self, layer, i, norm_factor):
        """Dense layer 0: fuse gate/up into one GEMM, absorb post-attention RMSNorm."""
        mlp = layer.mlp
        post = layer.post_attention_layernorm.weight.data.float().unsqueeze(0) * norm_factor
        gate_up = torch.cat([
            mlp.gate_proj.weight.data.float() * post,
            mlp.up_proj.weight.data.float() * post,
        ], dim=0)
        inter = int(mlp.gate_proj.weight.shape[0])
        self.dense_inter = [inter, inter]
        self.register_buffer(f'dense_gate_up_{i}', gate_up.contiguous())
        self.register_buffer(f'dense_down_{i}', mlp.down_proj.weight.data.float().contiguous())
        del layer.mlp, layer.post_attention_layernorm

    def _fuse_moe(self, layer, i, norm_factor):
        """MoE layer: absorb post-attention RMSNorm into router + expert/shared input projections, stack experts."""
        mlp = layer.mlp
        post = layer.post_attention_layernorm.weight.data.float().unsqueeze(0) * norm_factor

        router = (mlp.gate.weight.data.float() * post).contiguous()                    # (E, hidden)
        self.register_buffer(f'moe_router_{i}', router)

        Wg = torch.stack([e.gate_proj.weight.data.float() for e in mlp.experts], dim=0) * post.unsqueeze(0)
        Wu = torch.stack([e.up_proj.weight.data.float() for e in mlp.experts], dim=0) * post.unsqueeze(0)
        Wd = torch.stack([e.down_proj.weight.data.float() for e in mlp.experts], dim=0)
        self.register_buffer(f'moe_wg_{i}', Wg.contiguous())
        self.register_buffer(f'moe_wu_{i}', Wu.contiguous())
        self.register_buffer(f'moe_wd_{i}', Wd.contiguous())

        sh = mlp.shared_experts
        self.register_buffer(f'moe_sg_{i}', (sh.gate_proj.weight.data.float() * post).contiguous())
        self.register_buffer(f'moe_su_{i}', (sh.up_proj.weight.data.float() * post).contiguous())
        self.register_buffer(f'moe_sd_{i}', sh.down_proj.weight.data.float().contiguous())
        del layer.mlp, layer.post_attention_layernorm

    @staticmethod
    def _channel_statistic(weight, key):
        absolute = weight.abs()
        if key == 'rms':
            return (weight * weight).mean(0).sqrt()
        if key == 'L4':
            return absolute.pow(4).mean(0).pow(0.25)
        if key == 'std':
            return weight.std(0)
        if key != 'absmean':
            raise ValueError(f'Unsupported REORDER_KEY: {key!r}.')
        return absolute.mean(0)

    def _reorder_gated_pair(self, gate, up, down, key):
        if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
            raise ValueError('Expected rank-2 gated-MLP weights for channel reordering.')
        if gate.shape != up.shape or gate.shape[0] != down.shape[1]:
            raise ValueError('Gated-MLP producer/consumer intermediate dimensions do not match.')
        permutation = torch.argsort(self._channel_statistic(down, key))
        if torch.unique(permutation).numel() != permutation.numel():
            raise RuntimeError('Language MLP reorder permutation is not bijective.')
        gate.copy_(gate[permutation])
        up.copy_(up[permutation])
        down.copy_(down[:, permutation])
        self.language_reorder_pair_count += 1

    def _reorder_downproj_for_quant(self, key):
        """Apply paired exact permutations only across gated-MLP intermediates."""
        with torch.no_grad():
            for layer_index in range(self.first_k_dense):
                gate_up = getattr(self, f'dense_gate_up_{layer_index}')
                down = getattr(self, f'dense_down_{layer_index}')
                intermediate = down.shape[1]
                if gate_up.shape[0] != intermediate * 2:
                    raise ValueError('Dense gate/up fusion has an unexpected layout.')
                self._reorder_gated_pair(
                    gate_up[:intermediate], gate_up[intermediate:], down, key
                )
            for layer_index in range(self.first_k_dense, self.num_layers):
                gate = getattr(self, f'moe_wg_{layer_index}')
                up = getattr(self, f'moe_wu_{layer_index}')
                down = getattr(self, f'moe_wd_{layer_index}')
                for expert_index in range(gate.shape[0]):
                    self._reorder_gated_pair(
                        gate[expert_index], up[expert_index], down[expert_index], key
                    )
                self._reorder_gated_pair(
                    getattr(self, f'moe_sg_{layer_index}'),
                    getattr(self, f'moe_su_{layer_index}'),
                    getattr(self, f'moe_sd_{layer_index}'),
                    key,
                )

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    def _rms_norm(self, x):
        """Fused RMS normalization; downstream weights hold the sqrt(hidden) factor."""
        return simplified_layer_norm(x, self.hidden_norm_scale, self.hidden_rms_norm_eps)

    def _rotate_half(self, x, batch_size):
        """Swap-and-negate halves of head_dim (flip form of Llama rotate_half)."""
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def _dense_ffn(self, hidden_states, i):
        gate_up = torch.matmul(hidden_states, getattr(self, f'dense_gate_up_{i}').t())
        gate, up = torch.split(gate_up, self.dense_inter, dim=-1)
        return torch.matmul(F.silu(gate) * up, getattr(self, f'dense_down_{i}').t())

    def _moe_ffn(self, hidden_states, i):
        batch_size, seq_len, hidden = hidden_states.shape
        hf = hidden_states.reshape(-1, hidden)

        router = getattr(self, f'moe_router_{i}')
        logits = torch.matmul(hf, router.t())
        scores = torch.softmax(logits, dim=-1)
        topk_w, topk_i = torch.topk(scores, self.moe_top_k, dim=-1, sorted=False)
        if self.norm_topk and self.moe_top_k > 1:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling
        else:
            topk_w = topk_w * self.routed_scaling
        full_w = torch.zeros(hf.shape[0], self.n_experts, dtype=hf.dtype).scatter_(1, topk_i, topk_w.to(hf.dtype))

        Wg = getattr(self, f'moe_wg_{i}')
        Wu = getattr(self, f'moe_wu_{i}')
        Wd = getattr(self, f'moe_wd_{i}')
        gate = torch.einsum('th,eih->eti', hf, Wg)
        up   = torch.einsum('th,eih->eti', hf, Wu)
        act  = F.silu(gate) * up
        down = torch.einsum('eti,ehi->eth', act, Wd)
        y = torch.einsum('te,eth->th', full_w, down)

        Sg = getattr(self, f'moe_sg_{i}')
        Su = getattr(self, f'moe_su_{i}')
        Sd = getattr(self, f'moe_sd_{i}')
        shared = torch.matmul(F.silu(torch.matmul(hf, Sg.t())) * torch.matmul(hf, Su.t()), Sd.t())
        y = y + shared
        return y.reshape(batch_size, seq_len, hidden)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]
        attn_mask_f16      = attention_mask.half() if (self.kv_f16 and not self.compute_in_f32) else None

        for i, layer in enumerate(self.llm.model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            qkv   = layer.self_attn.qkv(hidden_states)
            qkv   = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Keep the entire F16-KV attention path in F16 unless higher-precision
            # cache compute is explicitly requested.
            if self.kv_f16 and not self.compute_in_f32:
                qk_rot = qk_rot.half()

            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            q    = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q    = q.permute(0, 2, 3, 1, 4)

            if self.kv_f16:
                if self.compute_in_f32:
                    k = k.half()
                v = v.half()

            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            if self.kv_rotary_q4:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()
                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                    q_rot      = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)
                    v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                    v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                    v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],   dim=-3)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()
                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                    q_rot      = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                    attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)
                    v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                    v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)

            elif self.kv_rotary:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()
                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()
                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k_signed)
                        attn          = attn_raw * k_s + attention_mask
                        attn          = torch.softmax(attn, dim=-1)
                        v_scaled  = v_signed * v_s
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-2)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()
                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k.float())
                        q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                        attn_bias     = q_bias_factor * k_b + attention_mask
                        attn          = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn          = torch.softmax(attn, dim=-1)
                        v_scaled  = v.float() * v_s
                        bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size) + bias_term

            elif self.kv_quantized:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()
                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()
                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(q, k_signed)
                        attn     = attn_raw * k_s + attention_mask
                        attn     = torch.softmax(attn, dim=-1)
                        v_scaled  = v_signed * v_s
                        attn      = torch.matmul(attn, v_scaled)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-2)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()
                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        q_sum_g    = q_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw  = torch.matmul(q, k.float())
                        attn_bias = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                        attn      = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn      = torch.softmax(attn, dim=-1)
                        v_dequant = torch.addcmul(v_b, v.float(), v_s)
                        attn      = torch.matmul(attn, v_dequant)

            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i]   = k
                self.save_value[i] = v
                if self.kv_f16 and self.compute_in_f32:
                    attn = torch.matmul(q, k.float()) + attention_mask
                    attn = torch.softmax(attn, dim=-1)
                    attn = torch.matmul(attn, v.float())
                elif self.kv_f16:
                    attn = torch.matmul(q, k) + attn_mask_f16
                    attn = torch.softmax(attn, dim=-1)
                    attn = torch.matmul(attn, v).float()
                else:
                    attn = torch.matmul(q, k) + attention_mask
                    attn = torch.softmax(attn, dim=-1)
                    attn = torch.matmul(attn, v)

            # Output projection & residual
            attn          = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network (dense layer 0, MoE otherwise) ──
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            if i < self.first_k_dense:
                hidden_states = residual + self._dense_ffn(hidden_states, i)
            else:
                hidden_states = residual + self._moe_ffn(hidden_states, i)

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1]) * self.final_norm_scale
        logits        = self.llm.lm_head(hidden_states)

        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale, logits
        elif self.kv_any_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


def _config_int(config, name, default=None):
    value = getattr(config, name, default)
    if value is None:
        raise ValueError(f'Missing required model configuration value: {name}.')
    return int(value)


def _id_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _normalize_unlimited_config():
    """Rehydrate the nested DeepSeek decoder settings on the wrapper config.

    The checkpoint's model wrapper keeps the complete decoder configuration in
    ``language_config``.  Newer Transformers retain only top-level fields when
    resolving this older custom class, so restore its declared defaults and the
    nested values before model construction.
    """
    try:
        from transformers.utils import import_utils
        if not hasattr(import_utils, 'is_torch_fx_available'):
            import_utils.is_torch_fx_available = lambda: False
    except ImportError:
        pass

    checkpoint_path = Path(download_path)
    raw_config = json.loads((checkpoint_path / 'config.json').read_text(encoding='utf-8'))
    config = AutoConfig.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    base_config_class = type(config).__mro__[1]
    for name, parameter in inspect.signature(base_config_class.__init__).parameters.items():
        if name == 'self' or parameter.default is inspect.Parameter.empty:
            continue
        if name not in config.__dict__:
            setattr(config, name, parameter.default)
    for name, value in raw_config['language_config'].items():
        if name not in {'architectures', 'auto_map', 'model_type'} and value is not None:
            setattr(config, name, value)
    config.pad_token_id = getattr(config, 'pad_token_id', None) or 2
    config.sliding_window = raw_config.get(
        'sliding_window', raw_config['language_config'].get('sliding_window_size')
    )
    config.head_dim = config.hidden_size // config.num_attention_heads
    config.rope_parameters = {
        'rope_type': 'default',
        'rope_theta': float(getattr(config, 'rope_theta', 10000.0)),
    }
    config._attn_implementation = 'eager'
    return config, raw_config


def _load_unlimited_components():
    config, raw_config = _normalize_unlimited_config()
    try:
        model = AutoModel.from_pretrained(
            download_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation='eager',
        ).eval()
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            'Unable to load the local UnlimitedOCR checkpoint with its normalized DeepSeek configuration.'
        ) from error

    vision_embeddings = model.model.vision_model.embeddings
    # Transformers 5 reports this registered buffer as a missing checkpoint
    # tensor and leaves its memory uninitialized. The target source constructs
    # it deterministically as arange(num_positions), so restore that exact value.
    vision_embeddings.position_ids = torch.arange(
        vision_embeddings.position_embedding.num_embeddings,
        dtype=torch.long,
        device=vision_embeddings.position_embedding.weight.device,
    ).unsqueeze(0)
    visual_module = sys.modules.get(type(model.model.vision_model).__module__)
    if visual_module is not None and hasattr(visual_module, 'quick_gelu'):
        visual_module.quick_gelu = lambda values: values * torch.sigmoid(1.702 * values)
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=False)
    return model, tokenizer, raw_config


def _metadata_values(dimensions, kv_facts, image_token_length):
    stop_ids = list(dict.fromkeys(STOP_TOKEN + _id_list(dimensions['eos_token_id'])))
    metadata = {
        'model_type': 'unlimited-ocr',
        'max_seq_len': str(MAX_SEQ_LEN),
        'input_image_size': ','.join(str(value) for value in INPUT_IMAGE_SIZE),
        'input_image_dim': str(INPUT_IMAGE_DIM),
        'vision_batch_size': '1',
        'image_token_id': str(IMAGE_TOKEN_ID),
        'image_token_length': str(image_token_length),
        'stop_token_ids': ','.join(str(value) for value in stop_ids),
        'eos_token_ids': ','.join(str(value) for value in _id_list(dimensions['eos_token_id'])),
        'num_layers': str(dimensions['num_layers']),
        'num_attention_heads': str(dimensions['num_heads']),
        'num_key_value_heads': str(dimensions['num_kv_heads']),
        'head_dim': str(dimensions['head_dim']),
        'hidden_size': str(dimensions['hidden_size']),
        'vocab_size': str(dimensions['vocab_size']),
        'first_k_dense_replace': str(dimensions['first_k_dense']),
        'n_routed_experts': str(dimensions['n_experts']),
        'num_experts_per_tok': str(dimensions['moe_top_k']),
        'rope_theta': str(dimensions['rope_theta']),
        'rms_norm_eps': str(dimensions['rms_norm_eps']),
        'compute_in_f32': str(int(COMPUTE_IN_F32)),
        'rope_shift_standard': '1',
        'fused_norm_language_count': str(dimensions['num_layers'] * 2 + 1),
        'reorder_downproj': str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        'vision_reorder_mlp': str(int(REORDER_VISION_MLP_FOR_QUANT)),
        'reorder_key': REORDER_KEY,
        'reorder_equivalence_max_error': '7.62939453125e-06',
        'kv_num_tensors': str(
            dimensions['num_layers'] * len(kv_facts['kv_cache_tensor_order'].split(','))
        ),
    }
    metadata.update(kv_facts)
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


def _stamp_metadata(path, metadata):
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField('metadata_props')
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(path, module, args, input_names, output_names, dynamic_axes, metadata):
    module.eval()
    print(f'Exporting {path.name} ...', flush=True)
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        external_data=True,
    )
    _stamp_metadata(path, metadata)
    print(f'Exported {path.name}.', flush=True)


def _build_kv_layout(batch_size, num_layers, num_kv_heads, head_dim, history_len):
    rotary_modes = {'ROTARY_Q4', 'ROTARY_Q4_CUDA', 'ROTARY_Q8', 'ROTARY_Q8_CUDA'}
    q8_modes = {'Q8', 'Q8_CUDA'}
    rotary_q4 = KV_QUANT_DTYPE in {'ROTARY_Q4', 'ROTARY_Q4_CUDA'}
    q8_grouped = KV_QUANT_DTYPE in q8_modes and (USE_HADAMARD or USE_SHUFFLE) and (
        KV_QUANT_GROUP_SIZE < head_dim
    )
    rotary_q8_grouped = KV_QUANT_DTYPE in {'ROTARY_Q8', 'ROTARY_Q8_CUDA'} and (
        USE_HADAMARD or USE_SHUFFLE
    ) and KV_QUANT_GROUP_SIZE < head_dim
    quantized = KV_QUANT_DTYPE in rotary_modes | q8_modes
    symmetric = USE_SYM and quantized
    grouped_6d = rotary_q4 or q8_grouped or rotary_q8_grouped
    specs = [('key', 4), ('value', 3)]
    if quantized:
        key_scale_axis = 5 if grouped_6d else 4
        specs.append(('key_scale', key_scale_axis))
        if not symmetric:
            specs.append(('key_bias', key_scale_axis))
        specs.append(('value_scale', 3))
        if not symmetric:
            specs.append(('value_bias', 3))

    if KV_QUANT_DTYPE == 'F16':
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE in {'Q8_CUDA', 'ROTARY_Q8_CUDA', 'ROTARY_Q4_CUDA'}:
        kv_dtype = torch.int32
    elif symmetric and not rotary_q4:
        kv_dtype = torch.int8
    elif quantized:
        kv_dtype = torch.uint8
    else:
        kv_dtype = torch.float32
    if KV_QUANT_DTYPE in {'Q8_CUDA', 'ROTARY_Q8_CUDA'}:
        key_width = value_width = head_dim // 4
    elif KV_QUANT_DTYPE == 'ROTARY_Q4':
        key_width = value_width = head_dim // 2
    elif KV_QUANT_DTYPE == 'ROTARY_Q4_CUDA':
        key_width = value_width = head_dim // 8
    else:
        key_width = value_width = head_dim

    tensors = {
        'key': torch.zeros((batch_size, num_kv_heads, 1, key_width, history_len), dtype=kv_dtype),
        'value': torch.zeros((batch_size, num_kv_heads, 1, history_len, value_width), dtype=kv_dtype),
    }
    scale_dtype = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32
    group_count = head_dim // KV_QUANT_GROUP_SIZE if grouped_6d else 1
    if quantized:
        if grouped_6d:
            key_scale_shape = (batch_size, num_kv_heads, 1, group_count, 1, history_len)
            value_scale_shape = (batch_size, num_kv_heads, 1, history_len, group_count, 1)
        else:
            key_scale_shape = (batch_size, num_kv_heads, 1, 1, history_len)
            value_scale_shape = (batch_size, num_kv_heads, 1, history_len, 1)
        tensors['key_scale'] = torch.ones(key_scale_shape, dtype=scale_dtype)
        tensors['value_scale'] = torch.ones(value_scale_shape, dtype=scale_dtype)
        if not symmetric:
            tensors['key_bias'] = torch.ones(key_scale_shape, dtype=scale_dtype)
            tensors['value_bias'] = torch.ones(value_scale_shape, dtype=scale_dtype)
    facts = {
        'kv_cache_quantization': KV_QUANT_DTYPE,
        'kv_cache_tensor_order': ','.join(name for name, _ in specs),
        'kv_cache_key_layout': 'batch,key_value_heads,one,key_width,sequence',
        'kv_cache_value_layout': 'batch,key_value_heads,one,sequence,value_width',
        'kv_cache_key_sequence_axis': '4',
        'kv_cache_value_sequence_axis': '3',
        'kv_cache_key_storage_width': str(key_width),
        'kv_cache_value_storage_width': str(value_width),
        'kv_cache_quantized': str(int(quantized)),
        'kv_cache_symmetric': str(int(symmetric)),
        'kv_cache_grouped_6d': str(int(grouped_6d)),
        'kv_cache_group_size': str(KV_QUANT_GROUP_SIZE if quantized else 0),
        'kv_cache_group_count': str(group_count if quantized else 0),
        'kv_cache_storage_dtype': str(kv_dtype).replace('torch.', ''),
        'kv_cache_scale_bias_dtype': str(scale_dtype).replace('torch.', '') if quantized else 'none',
    }
    return specs, tensors, facts


def _kv_io(kv_specs, kv_tensors, num_layers):
    inputs, input_names, output_names, dynamic_axes = [], [], [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            input_name = f'in_{name}_{layer_index}'
            output_name = f'out_{name}_{layer_index}'
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: 'batch_size', sequence_axis: 'history_len'}
            dynamic_axes[output_name] = {0: 'batch_size', sequence_axis: 'kv_seq_len'}
    return inputs, input_names, output_names, dynamic_axes


def _cleanup_unreferenced_data(export_dir):
    referenced = set()
    for model_path in export_dir.glob('*.onnx'):
        model = onnx.load(str(model_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location != onnx.TensorProto.EXTERNAL:
                continue
            location = {item.key: item.value for item in initializer.external_data}.get('location')
            if location:
                referenced.add(Path(location).name)
        for node in model.graph.node:
            for attribute in node.attribute:
                if attribute.HasField('t') and attribute.t.data_location == onnx.TensorProto.EXTERNAL:
                    location = {item.key: item.value for item in attribute.t.external_data}.get('location')
                    if location:
                        referenced.add(Path(location).name)
    for data_path in export_dir.iterdir():
        if data_path.is_file() and data_path.suffix != '.onnx' and data_path.name not in referenced:
            data_path.unlink()


def _prepare_export_staging():
    staging_dir = Path(EXPORT_STAGING_DIR)
    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise NotADirectoryError(f'Export staging path is not a directory: {staging_dir}.')
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    return staging_dir


def _promote_export(staging_dir):
    destination = Path(EXPORT_DIR)
    previous = destination.with_name(destination.name + '.previous')
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    if destination.exists():
        destination.rename(previous)
    staging_dir.rename(destination)
    return previous


def _rollback_export(previous):
    destination = Path(EXPORT_DIR)
    if destination.exists():
        shutil.rmtree(destination)
    if previous.exists():
        previous.rename(destination)


def _validate_bundle(export_dir):
    import Shared_Merged
    required = [
        MODEL_FILE_NAMES['metadata'], MODEL_FILE_NAMES['image_preprocess'],
        MODEL_FILE_NAMES['vision'], MODEL_FILE_NAMES['shared_initializers'],
        MODEL_FILE_NAMES['shared_initializers_data'],
                MODEL_FILE_NAMES['kv_slice'], MODEL_FILE_NAMES['kv_split2'],
                MODEL_FILE_NAMES['kv_concat'], MODEL_FILE_NAMES['rope_shift'],
        *(MODEL_FILE_NAMES[f'image_{phase}_{strategy}'] for phase in ('prefill', 'decode')
          for strategy in ('greedy', 'penalty_greedy', 'sampling')),
    ]
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'UnlimitedOCR export bundle is incomplete: {missing!r}.')
    for helper_role in ('kv_slice', 'kv_split2', 'kv_concat', 'rope_shift'):
        Shared_Merged.validate_onnx_path(export_dir / MODEL_FILE_NAMES[helper_role])
    for phase in ('prefill', 'decode'):
        for strategy in ('greedy', 'penalty_greedy', 'sampling'):
            Shared_Merged.validate_onnx_path(export_dir / MODEL_FILE_NAMES[f'image_{phase}_{strategy}'])


def _run_self_tests(export_dir):
    runtime = importlib.import_module('Inference_UnlimitedOCR_ONNX')
    original_strategy = runtime.STRATEGY
    original_limit = runtime.MAX_NEW_TOKENS
    try:
        for strategy in ('greedy', 'penalty_greedy', 'sampling'):
            print(
                f'Running standalone {strategy} ONNX self-test '
                f'({SELF_TEST_MAX_NEW_TOKENS} generated tokens) ...',
                flush=True,
            )
            runtime.STRATEGY = strategy
            runtime.MAX_NEW_TOKENS = SELF_TEST_MAX_NEW_TOKENS
            runtime.run_inference(export_dir)
    finally:
        runtime.STRATEGY = original_strategy
        runtime.MAX_NEW_TOKENS = original_limit


def _write_runtime_tokenizer(tokenizer, export_dir: Path) -> None:
    """Write the loaded Llama backend so tokenizers reproduces checkpoint IDs."""
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        raise RuntimeError('UnlimitedOCR tokenizer has no serializable fast backend.')
    (export_dir / 'tokenizer.json').write_text(backend.to_str(), encoding='utf-8')


@torch.inference_mode()
def export_unlimited():
    if INPUT_IMAGE_DIM not in {4, 5}:
        raise ValueError('INPUT_IMAGE_DIM must be 4 or 5.')
    export_dir = _prepare_export_staging()
    previous = None
    try:
        model, tokenizer, raw_config = _load_unlimited_components()
        dimensions = {
            'num_layers': _config_int(model.config, 'num_hidden_layers'),
            'num_heads': _config_int(model.config, 'num_attention_heads'),
            'num_kv_heads': _config_int(model.config, 'num_key_value_heads'),
            'hidden_size': _config_int(model.config, 'hidden_size'),
            'vocab_size': _config_int(model.config, 'vocab_size'),
            'first_k_dense': _config_int(model.config, 'first_k_dense_replace'),
            'n_experts': _config_int(model.config, 'n_routed_experts'),
            'moe_top_k': _config_int(model.config, 'num_experts_per_tok'),
            'eos_token_id': getattr(model.config, 'eos_token_id', tokenizer.eos_token_id),
            'rope_theta': float(getattr(model.config, 'rope_theta', 10000.0)),
            'rms_norm_eps': float(getattr(model.config, 'rms_norm_eps', 1e-6)),
        }
        dimensions['head_dim'] = _config_int(
            model.config, 'head_dim', dimensions['hidden_size'] // dimensions['num_heads']
        )
        dimensions['routed_scaling'] = float(getattr(model.config, 'routed_scaling_factor', 1.0))
        dimensions['norm_topk'] = bool(getattr(model.config, 'norm_topk_prob', False))
        if dimensions['num_heads'] % dimensions['num_kv_heads']:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads.')
        for note in normalize_kv_quant_settings(dimensions['head_dim']):
            print(note)

        image_token_ids = tokenizer.encode('<image>', add_special_tokens=False)
        if image_token_ids != [IMAGE_TOKEN_ID]:
            raise ValueError(f'Checkpoint tokenizer image token mismatch: {image_token_ids!r}.')
        grid = BASE_SIZE // PATCH_SIZE // DOWNSAMPLE_RATIO
        image_token_length = grid * (grid + 1) + 1
        kv_specs, kv_tensors, kv_facts = _build_kv_layout(
            1, dimensions['num_layers'], dimensions['num_kv_heads'], dimensions['head_dim'], 0
        )
        metadata = _metadata_values(dimensions, kv_facts, image_token_length)
        metadata['checkpoint_transformers_version'] = str(raw_config.get('transformers_version', 'unknown'))

        _export_component(
            export_dir / MODEL_FILE_NAMES['metadata'],
            METADATA_CARRIER(),
            (torch.zeros((1,), dtype=torch.int32),),
            ['metadata_marker'], ['metadata_marker_out'], None, metadata,
        )
        embed = LLM_EMBED(model)
        _export_component(
            export_dir / MODEL_FILE_NAMES['embed'], embed,
            (torch.tensor([[int(tokenizer.bos_token_id)]], dtype=torch.int32),),
            ['input_ids'], ['text_hidden_states'],
            {'input_ids': {0: 'batch_size', 1: 'ids_len'},
             'text_hidden_states': {0: 'batch_size', 1: 'ids_len'}}, metadata,
        )
        del embed, model.model.embed_tokens
        gc.collect()

        if INPUT_IMAGE_DIM == 5:
            image_input = torch.zeros((1, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
        else:
            image_input = torch.zeros((1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
        image_preprocess = LLM_IMAGE_PREPROCESS(INPUT_IMAGE_SIZE)
        _export_component(
            export_dir / MODEL_FILE_NAMES['image_preprocess'], image_preprocess, (image_input,),
            ['pixel_values'], ['preprocessed_pixel_values'],
            {'pixel_values': {0: 'image_count'}, 'preprocessed_pixel_values': {0: 'image_count'}}, metadata,
        )
        del image_preprocess, image_input
        gc.collect()

        vision = LLM_VISION(model, BASE_SIZE)
        if vision.image_token_length != image_token_length:
            raise RuntimeError('UnlimitedOCR vision feature count differs from its image-token expansion contract.')
        metadata['vision_reorder_mlp_pair_count'] = str(vision.vision_reorder_mlp_pair_count)
        vision_input = torch.zeros((1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.float32)
        _export_component(
            export_dir / MODEL_FILE_NAMES['vision'], vision, (vision_input,),
            ['preprocessed_pixel_values'], ['vision_hidden_states'],
            {'preprocessed_pixel_values': {0: 'image_count'},
             'vision_hidden_states': {0: 'image_count', 1: 'image_token_count'}}, metadata,
        )
        del vision, vision_input
        del model.model.sam_model, model.model.vision_model, model.model.projector
        del model.model.image_newline, model.model.view_seperator
        gc.collect()

        structural_ids = torch.tensor(
            [[int(tokenizer.bos_token_id)] + [IMAGE_TOKEN_ID] * image_token_length + [int(tokenizer.eos_token_id)]],
            dtype=torch.int32,
        )
        concat = LLM_CONCAT_IMAGE(IMAGE_TOKEN_ID, image_token_length)
        _export_component(
            export_dir / MODEL_FILE_NAMES['concat_image'], concat,
            (
                structural_ids,
                torch.ones((1, structural_ids.shape[1], dimensions['hidden_size']), dtype=torch.float32),
                torch.ones((1, image_token_length, dimensions['hidden_size']), dtype=torch.float32),
            ),
            ['input_ids', 'text_hidden_states', 'vision_hidden_states'], ['concat_hidden_states'],
            {'input_ids': {0: 'batch_size', 1: 'ids_len'},
             'text_hidden_states': {0: 'batch_size', 1: 'ids_len'},
             'concat_hidden_states': {0: 'batch_size', 1: 'ids_len'}}, metadata,
        )
        del concat, structural_ids
        gc.collect()

        trace_ids_len = torch.tensor([2], dtype=torch.int64)
        history_len = torch.zeros((1,), dtype=torch.int64)
        kv_seq_len = trace_ids_len + history_len
        _export_component(
            export_dir / MODEL_FILE_NAMES['rotary_image_prefill'],
            ROTARY_PREFILL(dimensions['head_dim'], dimensions['rope_theta'], MAX_SEQ_LEN),
            (trace_ids_len, history_len), ['ids_len', 'history_len'],
            ['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            {'rotary_cos': {1: 'ids_len'}, 'rotary_sin': {1: 'ids_len'},
             'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}}, metadata,
        )
        _export_component(
            export_dir / MODEL_FILE_NAMES['rotary_image_decode'],
            ROTARY_DECODE(dimensions['head_dim'], dimensions['rope_theta'], MAX_SEQ_LEN),
            (kv_seq_len,), ['kv_seq_len'], ['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            None, metadata,
        )

        kv_inputs, kv_input_names, kv_output_names, kv_axes = _kv_io(
            kv_specs, kv_tensors, dimensions['num_layers']
        )
        hidden_states = torch.ones((1, 2, dimensions['hidden_size']), dtype=torch.float32)
        rotary_cos = torch.zeros((1, 2, 1, 1, dimensions['head_dim']), dtype=torch.float32)
        rotary_sin = torch.zeros_like(rotary_cos)
        attention_mask = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
        main = LLM_MAIN(
            model, dimensions['num_heads'], dimensions['num_kv_heads'], dimensions['head_dim'],
            dimensions['num_layers'], dimensions['hidden_size'], dimensions['first_k_dense'],
            dimensions['n_experts'], dimensions['moe_top_k'], dimensions['routed_scaling'], dimensions['norm_topk'],
        )
        metadata['language_reorder_pair_count'] = str(main.language_reorder_pair_count)
        kv_quantizer = main.quantizer
        _export_component(
            export_dir / MODEL_FILE_NAMES['main'], main,
            tuple(kv_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]),
            kv_input_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask'],
            kv_output_names + ['logits'],
            {**kv_axes, 'hidden_states': {0: 'batch_size', 1: 'ids_len'},
             'logits': {0: 'batch_size'}, 'rotary_cos': {1: 'ids_len'},
             'rotary_sin': {1: 'ids_len'}, 'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}},
            metadata,
        )
        del main, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end = torch.tensor([1], dtype=torch.int64)
        _export_component(
            export_dir / MODEL_FILE_NAMES['kv_slice'],
            KV_SLICE(dimensions['num_layers'], dimensions['head_dim']),
            tuple(kv_inputs + [slice_start, slice_end]),
            kv_input_names + ['slice_start', 'slice_end'],
            kv_output_names,
            kv_axes,
            metadata,
        )
        del slice_start, slice_end

        split_at = torch.tensor([1], dtype=torch.int64)
        split_prefix_names = [f'prefix_{name}' for name in kv_output_names]
        split_suffix_names = [f'suffix_{name}' for name in kv_output_names]
        split_axes = {name: dict(kv_axes[name]) for name in kv_input_names}
        for output_name, prefix_name, suffix_name in zip(
            kv_output_names, split_prefix_names, split_suffix_names
        ):
            prefix_axes = dict(kv_axes[output_name])
            suffix_axes = dict(kv_axes[output_name])
            for axis in prefix_axes:
                if axis != 0:
                    prefix_axes[axis] = 'prefix_len'
                    suffix_axes[axis] = 'suffix_len'
            split_axes[prefix_name] = prefix_axes
            split_axes[suffix_name] = suffix_axes
        _export_component(
            export_dir / MODEL_FILE_NAMES['kv_split2'],
            KV_SPLIT2(dimensions['num_layers'], dimensions['head_dim']),
            tuple(kv_inputs + [split_at]),
            kv_input_names + ['split_at'],
            split_prefix_names + split_suffix_names,
            split_axes,
            metadata,
        )
        del split_at, split_prefix_names, split_suffix_names, split_axes

        concat_prefix_inputs = []
        concat_prefix_names = []
        concat_suffix_inputs = []
        concat_suffix_names = []
        concat_output_names = []
        concat_axes = {}
        for name, sequence_axis in kv_specs:
            tensor = kv_tensors[name]
            for layer_index in range(dimensions['num_layers']):
                prefix_name = f'in_prefix_{name}_{layer_index}'
                suffix_name = f'in_suffix_{name}_{layer_index}'
                output_name = f'out_{name}_{layer_index}'
                concat_prefix_inputs.append(tensor)
                concat_prefix_names.append(prefix_name)
                concat_suffix_inputs.append(tensor.clone())
                concat_suffix_names.append(suffix_name)
                concat_output_names.append(output_name)
                concat_axes[prefix_name] = {0: 'batch_size', sequence_axis: 'prefix_len'}
                concat_axes[suffix_name] = {0: 'batch_size', sequence_axis: 'suffix_len'}
                concat_axes[output_name] = {0: 'batch_size', sequence_axis: 'concat_len'}
        _export_component(
            export_dir / MODEL_FILE_NAMES['kv_concat'],
            KV_CONCAT(dimensions['num_layers'], dimensions['head_dim']),
            tuple(concat_prefix_inputs + concat_suffix_inputs),
            concat_prefix_names + concat_suffix_names,
            concat_output_names,
            concat_axes,
            metadata,
        )
        del (
            concat_prefix_inputs,
            concat_prefix_names,
            concat_suffix_inputs,
            concat_suffix_names,
            concat_output_names,
            concat_axes,
        )

        def _rope_shift_tensor(tensor):
            shape = list(tensor.shape)
            shape[-1] = 4
            return torch.zeros(shape, dtype=tensor.dtype)

        def _rope_shift_io(specifications):
            inputs, input_names, output_names, dynamic_axes = [], [], [], {}
            for name, tensor in specifications:
                sequence_axis = tensor.dim() - 1
                for layer_index in range(dimensions['num_layers']):
                    input_name = f'in_{name}_{layer_index}'
                    output_name = f'out_{name}_{layer_index}'
                    inputs.append(tensor)
                    input_names.append(input_name)
                    output_names.append(output_name)
                    dynamic_axes[input_name] = {0: 'batch_size', sequence_axis: 'history_len'}
                    dynamic_axes[output_name] = {0: 'batch_size', sequence_axis: 'history_len'}
            return inputs, input_names, output_names, dynamic_axes

        rope_shift_amount = torch.tensor([1], dtype=torch.int64)
        if KV_QUANT_DTYPE in {'F16', 'F32'}:
            rope_specs = [('key', _rope_shift_tensor(kv_tensors['key']))]
            rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_io(rope_specs)
            rope_shift_module = ROPE_SHIFT(
                dimensions['num_layers'],
                dimensions['num_kv_heads'],
                dimensions['head_dim'],
                dimensions['rope_theta'],
                MAX_SEQ_LEN,
            )
        else:
            rope_specs = [
                ('key', _rope_shift_tensor(kv_tensors['key'])),
                ('key_scale', _rope_shift_tensor(kv_tensors['key_scale'])),
            ]
            if not USE_SYM:
                rope_specs.append(
                    ('key_bias', _rope_shift_tensor(kv_tensors['key_bias']))
                )
            rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_io(rope_specs)
            rope_shift_module = ROPE_SHIFT_QUANT(
                dimensions['num_layers'],
                dimensions['num_kv_heads'],
                dimensions['head_dim'],
                dimensions['rope_theta'],
                MAX_SEQ_LEN,
                kv_quantizer,
                not USE_SYM,
            )
        _export_component(
            export_dir / MODEL_FILE_NAMES['rope_shift'],
            rope_shift_module,
            tuple(rope_inputs + [rope_shift_amount]),
            rope_input_names + ['shift'],
            rope_output_names,
            rope_axes,
            metadata,
        )
        del (
            _rope_shift_tensor,
            _rope_shift_io,
            rope_shift_amount,
            rope_specs,
            rope_inputs,
            rope_input_names,
            rope_output_names,
            rope_axes,
            rope_shift_module,
            kv_quantizer,
            kv_inputs,
        )
        gc.collect()

        logits = torch.ones((1, dimensions['vocab_size']), dtype=torch.float32)
        previous_ids = torch.zeros((1, 1), dtype=torch.int32)
        repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
        _export_component(
            export_dir / MODEL_FILE_NAMES['greedy'], GREEDY_SEARCH(), (logits,),
            ['logits'], ['max_logits_idx'],
            {'logits': {0: 'batch_size'}, 'max_logits_idx': {0: 'batch_size'}}, metadata,
        )
        _export_component(
            export_dir / MODEL_FILE_NAMES['penalty_greedy'], PENALTY_GREEDY_SEARCH(),
            (logits, repetition_penalty, previous_ids),
            ['logits', 'repetition_penalty', 'previous_ids'], ['max_logits_idx', 'save_id_out'],
            {'logits': {0: 'batch_size'}, 'repetition_penalty': {0: 'batch_size'},
             'previous_ids': {0: 'batch_size', 1: 'history_len'},
             'max_logits_idx': {0: 'batch_size'}, 'save_id_out': {0: 'batch_size', 1: 'kv_seq_len'}}, metadata,
        )
        _export_component(
            export_dir / MODEL_FILE_NAMES['sampling'], TOPK_TOPP_SAMPLING(),
            (logits, torch.ones((1,), dtype=torch.float32),
             torch.tensor(min(50, dimensions['vocab_size']), dtype=torch.int64),
             torch.ones((1,), dtype=torch.float32), repetition_penalty, previous_ids),
            ['logits', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'previous_ids'],
            ['sampled_id', 'save_id_out'],
            {'logits': {0: 'batch_size'}, 'temperature': {0: 'batch_size'},
             'top_p': {0: 'batch_size'}, 'repetition_penalty': {0: 'batch_size'},
             'previous_ids': {0: 'batch_size', 1: 'history_len'}, 'sampled_id': {0: 'batch_size'},
             'save_id_out': {0: 'batch_size', 1: 'kv_seq_len'}}, metadata,
        )
        del logits, previous_ids, repetition_penalty, model
        gc.collect()

        import Shared_Merged
        for constituent in export_dir.glob('*.onnx'):
            _stamp_metadata(constituent, metadata)
        bundle = Shared_Merged.build_shared_merged_bundle(
            export_dir, model_file_names=MODEL_FILE_NAMES, delete_constituents=True
        )
        for path in bundle['graphs'].values():
            _stamp_metadata(path, metadata)
        _stamp_metadata(bundle['shared_model'], metadata)
        _stamp_metadata(export_dir / MODEL_FILE_NAMES['metadata'], metadata)
        _cleanup_unreferenced_data(export_dir)
        _validate_bundle(export_dir)
        tokenizer_assets = copy_tokenizer_assets(download_path, export_dir)
        _write_runtime_tokenizer(tokenizer, export_dir)
        previous = _promote_export(export_dir)
        _run_self_tests(Path(EXPORT_DIR))
        if previous.exists():
            shutil.rmtree(previous)
        print(
            f'UnlimitedOCR ONNX export completed: {EXPORT_DIR} '
            f'({len(tokenizer_assets)} tokenizer assets).'
        )
    except BaseException:
        if previous is not None:
            _rollback_export(previous)
        raise


def export_bundle():
    """Export the UnlimitedOCR ONNX bundle."""
    return export_unlimited()


def main():
    if not DO_EXPORT:
        print('DO_EXPORT is False; no ONNX files were written.')
        return
    export_bundle()


if __name__ == '__main__':
    main()


