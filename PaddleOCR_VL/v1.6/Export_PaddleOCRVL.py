"""Export PaddleOCR-VL-1.6 to the finalized image-only ONNX bundle contract.

The target has an ERNIE-4.5 decoder, a SigLIP-style vision encoder, a 2x2
vision projector, and multimodal 3-D RoPE.  Unlike the Qwen-family reference,
it has no DeepStack side inputs or Q/K RMSNorm modules.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR / "PaddleOCRVL_ONNX"
EXPORT_STAGING_DIR = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")
CHECKPOINT_DIR = Path.home() / "Downloads" / "PaddleOCR-VL-1.6"
# Backward-compatible configuration alias.
MODEL_CHECKPOINT = CHECKPOINT_DIR

DO_EXPORT          = True                     # Whether to export the ONNX models.
MAX_SEQ_LEN        = 4096                     # Fixed maximum context length after export.

# Image input and vision tracing
VISION_IMAGE_HEIGHT = 616                     # Vision encoder input height.
VISION_IMAGE_WIDTH  = 616                     # Vision encoder input width.
INPUT_IMAGE_HEIGHT  = 616                     # Runtime image height before preprocessing.
INPUT_IMAGE_WIDTH   = 616                     # Runtime image width before preprocessing.
VISION_BATCH_SIZE   = 1                       # Number of images supported by the prompt.
INPUT_IMAGE_DIM     = 4                       # pixel_values rank: 4=[B, C, H, W]; 5=[B, 1, C, H, W].

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                    # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 128                     # Quantization group width when grouping is enabled.
COMPUTE_IN_F32      = False                   # F16 cache only: False keeps KV attention in F16, True upcasts reads.

# KV quantization transforms and parameters
USE_SYM                = True                 # Use symmetric rather than scale-and-bias quantization.
USE_FLOAT16_SCALE_BIAS = True                 # Store quantization scales and biases as float16.
USE_HADAMARD           = False                # Apply a randomized Hadamard transform before grouped quantization.
HADAMARD_RANDOM_SEED   = 9527                 # Deterministic sign pattern for the Hadamard transform.
USE_CLIP               = False                # Clip quantization blocks to CLIP_SIGMA standard deviations.
CLIP_SIGMA             = 3.0                  # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False                # Interleave channels across quantization groups.
USE_QDQ_FRIENDLY_ASYM  = False                # Disable residual bias correction for asymmetric QDQ compatibility.

# Quantization-oriented model reordering
REORDER_DOWNPROJ_FOR_QUANT   = True           # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True           # Reorder vision MLP channels before down-projection quantization.
REORDER_KEY                  = "absmean"      # Channel statistic used to build the language MLP permutation.

# ONNX graph format
OPSET = 20                                    # ONNX opset version.

MODEL_FILE_NAMES = {
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
    "image_prefill_greedy": "LLM_ImagePrefillGreedy.onnx",
    "image_prefill_penalty_greedy": "LLM_ImagePrefillPenaltyGreedy.onnx",
    "image_prefill_sampling": "LLM_ImagePrefillSampling.onnx",
    "image_decode_greedy": "LLM_ImageDecodeGreedy.onnx",
    "image_decode_penalty_greedy": "LLM_ImageDecodePenaltyGreedy.onnx",
    "image_decode_sampling": "LLM_ImageDecodeSampling.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = (
    MODEL_FILE_NAMES["shared_initializers"] + ".data"
)
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


def normalize_kv_quant_settings(head_dim: int) -> list[str]:
    """Validate and normalize the Qwen-compatible KV cache settings."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized = {
        "Q8",
        "Q8_CUDA",
        "ROTARY_Q8",
        "ROTARY_Q8_CUDA",
        "ROTARY_Q4",
        "ROTARY_Q4_CUDA",
    }
    rotary = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8 = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes: list[str] = []

    if KV_QUANT_DTYPE in rotary and head_dim % 2:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")

    if KV_QUANT_DTYPE in quantized:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError(
                f"KV_QUANT_GROUP_SIZE must be positive, got {KV_QUANT_GROUP_SIZE}."
            )
        if KV_QUANT_GROUP_SIZE > head_dim:
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim."
            )
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(
                group for group in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % group == 0
            )
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}."
            )
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(
                f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization."
            )

        if KV_QUANT_DTYPE in q8 and KV_QUANT_GROUP_SIZE == head_dim and (
            USE_HADAMARD or USE_SHUFFLE
        ):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block."
            )
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append(
            "[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32."
        )

    return notes



class GREEDY_SEARCH(torch.nn.Module):
    """Token-only greedy contract used by merged decode graphs."""

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
        max_logits_idx = torch.argmax(adjusted_logits, dim=-1, keepdim=True).int()
        return max_logits_idx, torch.cat([previous_ids, max_logits_idx], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """TopTok sampling with sign-aware repetition handling."""

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
    """Identity graph used to load metadata before large runtime sessions."""

    def forward(self, marker):
        return marker


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


class LLM_EMBED(torch.nn.Module):
    """Extract the target's dynamic token embedding lookup in float32."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Resize, normalize, and patchify a raw PaddleOCR-VL image."""

    def __init__(
        self,
        target_height: int,
        target_width: int,
        patch_size: int,
        image_mean: list[float],
        image_std: list[float],
    ):
        super().__init__()
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.patch_size = int(patch_size)
        self.grid_height = self.target_height // self.patch_size
        self.grid_width = self.target_width // self.patch_size
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        pixel_values = F.interpolate(
            pixel_values.float(),
            size=[self.target_height, self.target_width],
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (pixel_values / 255.0 - self.image_mean) / self.image_std
        batch_size = pixel_values.shape[0]
        patches = pixel_values.reshape(
            batch_size,
            3,
            self.grid_height,
            self.patch_size,
            self.grid_width,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            -1, 3, self.patch_size, self.patch_size
        )
        return patches


class STATIC_FUSED_VISION_ENCODER(torch.nn.Module):
    """Run fixed-grid vision layers with precomputed position and RoPE tables."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        post_layernorm: torch.nn.Module,
        position_embedding: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ):
        super().__init__()
        self.layers = layers
        self.post_layernorm = post_layernorm
        self.register_buffer(
            "position_embedding", position_embedding.float(), persistent=False
        )
        self.register_buffer("rotary_cos", rotary_cos.float(), persistent=False)
        self.register_buffer("rotary_sin", rotary_sin.float(), persistent=False)

    def forward(self, patch_embeddings):
        hidden_states = patch_embeddings + self.position_embedding
        rope_emb = (self.rotary_cos, self.rotary_sin)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                cu_seqlens=None,
                rope_emb=rope_emb,
            )[0]
        return self.post_layernorm(hidden_states)


class FUSED_VISION_ATTENTION(torch.nn.Module):
    """Paddle vision attention with fused QKV, folded score scale, and static RoPE."""

    def __init__(
        self,
        attention: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
    ):
        super().__init__()
        self.embed_dim = int(attention.embed_dim)
        self.num_heads = int(attention.num_heads)
        self.head_dim = int(attention.head_dim)
        if self.embed_dim != self.num_heads * self.head_dim or self.head_dim % 2:
            raise ValueError("PaddleOCR-VL vision attention has invalid head geometry.")
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("Static vision attention requires positive batch and sequence lengths.")
        self.head_dim_half = self.head_dim // 2
        self.dropout = float(attention.dropout)
        self.qkv = self._fuse_qkv(attention)
        self.out_proj = attention.out_proj

    @staticmethod
    def _fuse_qkv(attention: torch.nn.Module) -> torch.nn.Linear:
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        if len({projection.in_features for projection in projections}) != 1:
            raise ValueError("Vision QKV projections must share an input width.")
        has_bias = any(projection.bias is not None for projection in projections)
        fused = torch.nn.Linear(
            projections[0].in_features,
            sum(projection.out_features for projection in projections),
            bias=has_bias,
            dtype=projections[0].weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat([projection.weight for projection in projections], dim=0)
            )
            if fused.bias is not None:
                fused.bias.copy_(
                    torch.cat(
                        [
                            projection.bias
                            if projection.bias is not None
                            else torch.zeros_like(projection.weight[:, 0])
                            for projection in projections
                        ],
                        dim=0,
                    )
                )
        return fused

    def fold_score_scale(self) -> None:
        """Fold the native $1 / sqrt(head_dim)$ scale into Q and K."""
        scale = self.head_dim**-0.25
        with torch.no_grad():
            self.qkv.weight[: self.embed_dim * 2].mul_(scale)
            if self.qkv.bias is not None:
                self.qkv.bias[: self.embed_dim * 2].mul_(scale)

    def _rotate_half(self, values):
        values = onnx_static_reshape(
            values,
            (
                2,
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                2,
                self.head_dim_half,
            ),
        )
        return onnx_static_reshape(
            values.flip(-2),
            (
                2,
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                self.head_dim,
            ),
        )

    def _apply_rope(self, qk, cosine, sine):
        return qk * cosine + self._rotate_half(qk) * sine

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        cu_seqlens=None,
        rope_emb=None,
    ):
        del cu_seqlens
        qkv = onnx_static_reshape(
            self.qkv(hidden_states),
            (
                self.batch_size,
                self.sequence_length,
                3,
                self.num_heads,
                self.head_dim,
            ),
        ).permute(2, 0, 3, 1, 4)
        qk, value = torch.split(qkv, [2, 1], dim=0)
        if rope_emb is not None:
            qk = self._apply_rope(qk, *rope_emb)
        query, key = torch.unbind(qk, dim=0)
        value = value.squeeze(0)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = onnx_static_reshape(
            torch.matmul(weights, value).transpose(1, 2),
            (self.batch_size, self.sequence_length, self.embed_dim),
        )
        output = self.out_proj(output)
        return output, weights if output_attentions else None


class STATIC_VISION_PROJECTOR(torch.nn.Module):
    """Project fixed-grid vision features without list or batch-dimension churn."""

    def __init__(
        self,
        projector: torch.nn.Module,
        grid_thw_tuple: tuple[int, int, int],
        batch_size: int,
    ):
        super().__init__()
        self.pre_norm = projector.pre_norm
        self.linear_1 = projector.linear_1
        self.act = projector.act
        self.linear_2 = projector.linear_2
        self.time_count, grid_height, grid_width = (
            int(value) for value in grid_thw_tuple
        )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Static vision projector requires a positive batch size.")
        self.merge_height, self.merge_width = projector.merge_kernel_size
        if grid_height % self.merge_height or grid_width % self.merge_width:
            raise ValueError("Vision grid must be divisible by the projector merge kernel.")
        self.merged_height = grid_height // self.merge_height
        self.merged_width = grid_width // self.merge_width
        self.vision_hidden_size = int(projector.vision_config.hidden_size)
        self.image_token_count = (
            self.time_count * self.merged_height * self.merged_width
        )
        self.merged_hidden_size = (
            self.merge_height * self.merge_width * self.vision_hidden_size
        )

    def forward(self, vision_hidden_states):
        vision_hidden_states = self.pre_norm(vision_hidden_states)
        vision_hidden_states = onnx_static_reshape(
            vision_hidden_states,
            (
                self.batch_size,
                self.time_count,
                self.merged_height,
                self.merge_height,
                self.merged_width,
                self.merge_width,
                self.vision_hidden_size,
            ),
        )
        vision_hidden_states = vision_hidden_states.permute(0, 1, 2, 4, 3, 5, 6)
        vision_hidden_states = onnx_static_reshape(
            vision_hidden_states,
            (self.batch_size, self.image_token_count, self.merged_hidden_size),
        )
        return self.linear_2(self.act(self.linear_1(vision_hidden_states)))


class LLM_VISION(torch.nn.Module):
    """Run PaddleOCR-VL's native encoder and its 2x2 vision projector."""

    def __init__(self, model, grid_thw: torch.Tensor, reorder_key: str):
        super().__init__()
        vision_model = model.visual.vision_model
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.grid_thw_tuple = tuple(int(value) for value in grid_thw[0].tolist())
        self.batch_size = VISION_BATCH_SIZE
        self.projector = STATIC_VISION_PROJECTOR(
            model.mlp_AR, self.grid_thw_tuple, self.batch_size
        )
        self.patch_count = int(torch.prod(grid_thw[0]).item())
        self.vision_hidden_size = int(vision_model.config.hidden_size)
        self.fused_vision_qkv_count = 0
        self.folded_vision_layer_norm_count = 0
        position_embedding, rotary_cos, rotary_sin = self._build_static_tables(
            vision_model.embeddings,
            vision_model.encoder.rotary_pos_emb,
            self.grid_thw_tuple[1],
            self.grid_thw_tuple[2],
        )
        self._fuse_vision_weights(vision_model.encoder.layers)
        self.encoder = STATIC_FUSED_VISION_ENCODER(
            vision_model.encoder.layers,
            vision_model.post_layernorm,
            position_embedding,
            rotary_cos,
            rotary_sin,
        )
        self.reordered_pairs = 0
        self.reorder_equivalence_error = 0.0
        if REORDER_VISION_MLP_FOR_QUANT:
            self._reorder_mlp_pairs(reorder_key)

    @staticmethod
    def _fold_layer_norm_affine(norm, linear) -> None:
        if not isinstance(norm, torch.nn.LayerNorm) or norm.weight is None:
            raise ValueError("Expected an unfused affine LayerNorm before a linear projection.")
        if linear.in_features % norm.weight.numel():
            raise ValueError("LayerNorm width does not divide the following linear input width.")
        repeat_count = linear.in_features // norm.weight.numel()
        norm_scale = norm.weight.detach().repeat(repeat_count)
        norm_bias = (
            norm.bias.detach().repeat(repeat_count)
            if norm.bias is not None
            else torch.zeros_like(norm_scale)
        )
        with torch.no_grad():
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(
                        linear.out_features,
                        dtype=linear.weight.dtype,
                        device=linear.weight.device,
                    )
                )
            linear.bias.add_(torch.matmul(linear.weight, norm_bias))
            linear.weight.mul_(norm_scale.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

    @staticmethod
    def _build_static_tables(embeddings, rotary_embedding, grid_height, grid_width):
        position_weight = embeddings.position_embedding.weight.detach()
        source_count, hidden_size = position_weight.shape
        source_side = int(source_count**0.5)
        if source_side * source_side != source_count:
            raise ValueError("Vision position embedding count must form a square grid.")
        position_embedding = F.interpolate(
            position_weight.view(1, source_side, source_side, hidden_size).permute(0, 3, 1, 2),
            size=(grid_height, grid_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, -1, hidden_size)

        inv_freq = rotary_embedding.inv_freq.detach().float()
        patch_ids = torch.arange(grid_height * grid_width, dtype=torch.int64)
        coordinates = torch.stack(
            [patch_ids // grid_width, patch_ids % grid_width], dim=-1
        )
        frequencies = torch.outer(
            torch.arange(max(grid_height, grid_width), dtype=inv_freq.dtype), inv_freq
        )
        phases = frequencies[coordinates].flatten(1).repeat(1, 2)
        head_dim = int(phases.shape[-1])
        sine_sign = torch.cat(
            [-torch.ones(head_dim // 2), torch.ones(head_dim // 2)]
        )
        return (
            position_embedding,
            phases.cos().view(1, 1, 1, -1, head_dim),
            (phases.sin() * sine_sign).view(1, 1, 1, -1, head_dim),
        )

    def _fuse_vision_weights(self, layers) -> None:
        for layer in layers:
            attention = FUSED_VISION_ATTENTION(
                layer.self_attn, self.batch_size, self.patch_count
            )
            self._fold_layer_norm_affine(layer.layer_norm1, attention.qkv)
            attention.fold_score_scale()
            layer.self_attn = attention
            self._fold_layer_norm_affine(layer.layer_norm2, layer.mlp.fc1)
            self.fused_vision_qkv_count += 1
            self.folded_vision_layer_norm_count += 2
        self._fold_layer_norm_affine(self.projector.pre_norm, self.projector.linear_1)
        self.folded_vision_layer_norm_count += 1

    @staticmethod
    def _channel_statistic(weight: torch.Tensor, key: str) -> torch.Tensor:
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(0).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(0).pow(0.25)
        if key == "std":
            return weight.std(0)
        if key == "absmean":
            return absolute.mean(0)
        raise ValueError("REORDER_KEY must be one of: absmean, L4, rms, std.")

    @classmethod
    def _reorder_pair(cls, first, second, activation, key: str) -> float:
        if first.out_features != second.in_features:
            raise ValueError("Vision MLP producer/consumer dimensions do not match.")
        if getattr(first, "_paddleocr_vl_reordered", False):
            raise RuntimeError("Refusing to apply a vision MLP permutation twice.")
        permutation = torch.argsort(cls._channel_statistic(second.weight.data, key))
        if permutation.numel() != first.out_features or torch.unique(permutation).numel() != permutation.numel():
            raise RuntimeError("Vision MLP permutation is not bijective.")
        with torch.no_grad():
            probe = torch.randn(
                2, first.in_features, dtype=first.weight.dtype, device=first.weight.device
            )
            before = second(activation(first(probe))).float()
            first.weight.data.copy_(first.weight.data[permutation])
            if first.bias is not None:
                first.bias.data.copy_(first.bias.data[permutation])
            second.weight.data.copy_(second.weight.data[:, permutation])
            after = second(activation(first(probe))).float()
            first._paddleocr_vl_reordered = True
        return float((before - after).abs().max())

    def _reorder_mlp_pairs(self, key: str) -> None:
        for layer in self.encoder.layers:
            error = self._reorder_pair(
                layer.mlp.fc1, layer.mlp.fc2, layer.mlp.activation_fn, key
            )
            self.reordered_pairs += 1
            self.reorder_equivalence_error = max(self.reorder_equivalence_error, error)
        error = self._reorder_pair(
            self.projector.linear_1,
            self.projector.linear_2,
            self.projector.act,
            key,
        )
        self.reordered_pairs += 1
        self.reorder_equivalence_error = max(self.reorder_equivalence_error, error)

    def forward(self, patches):
        patch_embeddings = onnx_static_reshape(
            self.patch_embedding(patches),
            (self.batch_size, self.patch_count, self.vision_hidden_size),
        )
        vision_hidden_states = self.encoder(patch_embeddings)
        return self.projector(vision_hidden_states).float()


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the verified contiguous image-token span with vision features."""

    def __init__(self, image_start: int, image_end: int):
        super().__init__()
        self.image_start = int(image_start)
        self.image_end = int(image_end)

    def forward(self, text_hidden_states, vision_hidden_states):
        head = text_hidden_states[:, : self.image_start]
        tail = text_hidden_states[:, self.image_end :]
        return torch.cat([head, vision_hidden_states, tail], dim=1)


class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Provide target mRoPE tables and a causal mask for image prefill."""

    def __init__(self, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor):
        super().__init__()
        total_len = int(rotary_cos.shape[0])
        self.register_buffer(
            "cos_rotary_pos_emb",
            rotary_cos.half().unsqueeze(0).unsqueeze(2).unsqueeze(2),
            persistent=False,
        )
        self.register_buffer(
            "sin_rotary_pos_emb",
            rotary_sin.half().unsqueeze(0).unsqueeze(2).unsqueeze(2),
            persistent=False,
        )
        mask = 1 - torch.tril(torch.ones((1, 1, 1, total_len, total_len), dtype=torch.int8))
        self.register_buffer("attention_mask", mask * -128, persistent=False)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    """Provide the target mRoPE row for a single decode step."""

    def __init__(self, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor):
        super().__init__()
        self.register_buffer(
            "cos_rotary_pos_emb",
            rotary_cos.half().unsqueeze(0).unsqueeze(2).unsqueeze(2),
            persistent=False,
        )
        self.register_buffer(
            "sin_rotary_pos_emb",
            rotary_sin.half().unsqueeze(0).unsqueeze(2).unsqueeze(2),
            persistent=False,
        )

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


KV_QUANTIZED_DTYPES = frozenset(
    {
        "Q8",
        "Q8_CUDA",
        "ROTARY_Q8",
        "ROTARY_Q8_CUDA",
        "ROTARY_Q4",
        "ROTARY_Q4_CUDA",
    }
)
KV_ROTARY_DTYPES = frozenset(
    {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
)
KV_ROTARY_Q4_DTYPES = frozenset({"ROTARY_Q4", "ROTARY_Q4_CUDA"})
KV_Q8_DTYPES = frozenset({"Q8", "Q8_CUDA"})
KV_CUDA_DTYPES = frozenset({"Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"})


def _kv_mode_flags(head_dim: int) -> dict[str, bool | int]:
    """Describe the cache storage layout selected by the current KV settings."""
    quantized = KV_QUANT_DTYPE in KV_QUANTIZED_DTYPES
    rotary = KV_QUANT_DTYPE in KV_ROTARY_DTYPES
    rotary_q4 = KV_QUANT_DTYPE in KV_ROTARY_Q4_DTYPES
    q8 = KV_QUANT_DTYPE in KV_Q8_DTYPES
    q8_cuda = KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"}
    q8_grouped = (
        (q8 or KV_QUANT_DTYPE in {"ROTARY_Q8", "ROTARY_Q8_CUDA"})
        and (USE_HADAMARD or USE_SHUFFLE)
        and KV_QUANT_GROUP_SIZE < head_dim
    )
    grouped_6d = rotary_q4 or q8_grouped
    symmetric = USE_SYM and quantized
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"}:
        storage_width = head_dim // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        storage_width = head_dim // 8
    elif rotary_q4:
        storage_width = head_dim // 2
    else:
        storage_width = head_dim
    return {
        "quantized": quantized,
        "rotary": rotary,
        "rotary_q4": rotary_q4,
        "q8_cuda": q8_cuda,
        "cuda": KV_QUANT_DTYPE in KV_CUDA_DTYPES,
        "symmetric": symmetric,
        "q8_grouped": q8_grouped,
        "grouped_6d": grouped_6d,
        "storage_width": storage_width,
    }


def _full_state_specs(head_dim: int) -> list[tuple[str, int]]:
    """Return field-major full-attention state names and their sequence axes."""
    flags = _kv_mode_flags(head_dim)
    specs = [("key", 4), ("value", 3)]
    if not flags["quantized"]:
        return specs
    key_scale_axis = 5 if flags["grouped_6d"] else 4
    specs.append(("key_scale", key_scale_axis))
    if not flags["symmetric"]:
        specs.append(("key_bias", key_scale_axis))
    specs.append(("value_scale", 3))
    if not flags["symmetric"]:
        specs.append(("value_bias", 3))
    return specs


class ONNX_STATIC_RESHAPE(torch.autograd.Function):
    """Emit a Reshape whose zero entries retain their input dimensions."""

    @staticmethod
    def forward(ctx, values, shape):
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


class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    """Compute [prefix, window, suffix] sizes for a dynamic Split."""

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
        window = graph.op("Sub", end, start)
        suffix = graph.op("Sub", dim_size, end)
        return graph.op("Concat", start, window, suffix, axis_i=0)


class SLICE_KEEP_MIDDLE(torch.autograd.Function):
    """Select the middle member of a dynamic three-way Split."""

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
    """Compute [prefix, suffix] sizes for a dynamic Split."""

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
        suffix = graph.op("Sub", dim_size, split_at)
        return graph.op("Concat", split_at, suffix, axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):
    """Split one cache state at a dynamic sequence index."""

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
    """Slice each cache state to the requested [start:end] window."""

    def __init__(self, state_specs, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        self.sequence_axes = tuple(axis for _, axis in state_specs)
        self.state_count = self.num_layers * len(self.sequence_axes)

    def forward(self, *all_inputs):
        slice_start, slice_end = all_inputs[-2:]
        sizes = window_split_sizes(
            all_inputs[0], slice_start, slice_end, self.sequence_axes[0]
        )
        outputs = []
        for state_index, values in enumerate(all_inputs[: self.state_count]):
            field_index = state_index // self.num_layers
            outputs.append(
                slice_keep_middle(values, sizes, self.sequence_axes[field_index])
            )
        return tuple(outputs)


class KV_SPLIT2(torch.nn.Module):
    """Split each cache state into field-major prefix and suffix groups."""

    def __init__(self, state_specs, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        self.sequence_axes = tuple(axis for _, axis in state_specs)
        self.state_count = self.num_layers * len(self.sequence_axes)

    def forward(self, *all_inputs):
        split_at = all_inputs[-1]
        sizes = split_point_sizes(all_inputs[0], split_at, self.sequence_axes[0])
        prefix, suffix = [], []
        for state_index, values in enumerate(all_inputs[: self.state_count]):
            field_index = state_index // self.num_layers
            before, after = split_prefix_suffix(
                values, sizes, self.sequence_axes[field_index]
            )
            prefix.append(before)
            suffix.append(after)
        return (*prefix, *suffix)


class KV_CONCAT(torch.nn.Module):
    """Concatenate matching field-major cache state groups."""

    def __init__(self, state_specs, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        self.sequence_axes = tuple(axis for _, axis in state_specs)
        self.state_count = self.num_layers * len(self.sequence_axes)

    def forward(self, *all_inputs):
        prefix = all_inputs[: self.state_count]
        suffix = all_inputs[self.state_count :]
        outputs = []
        for state_index, values in enumerate(prefix):
            field_index = state_index // self.num_layers
            outputs.append(
                torch.cat([values, suffix[state_index]], dim=self.sequence_axes[field_index])
            )
        return tuple(outputs)


def build_mrope_shift_tables(rotary_module, max_shift: int):
    """Build inverse mRoPE tables for rebasing text-position cache keys.

    A scalar cache shift is valid once the cached suffix is text-only, where
    all three PaddleOCR-VL mRoPE position channels advance by the same amount.
    """
    shift_ids = torch.arange(max_shift + 1, dtype=torch.float32).view(-1, 1)
    frequencies = shift_ids * rotary_module.inv_freq.float().view(1, -1)
    cosine = torch.cat([frequencies.cos(), frequencies.cos()], dim=-1)
    sine = torch.cat([frequencies.sin(), -frequencies.sin()], dim=-1)
    head_dim = int(cosine.shape[-1])
    return (
        cosine.half().view(max_shift + 1, 1, 1, head_dim, 1),
        sine.half().view(max_shift + 1, 1, 1, head_dim, 1),
    )


class ROPE_SHIFT(torch.nn.Module):
    """Rebase F16/F32 full-attention keys by a text mRoPE position delta."""

    def __init__(self, num_layers, num_kv_heads, rotary_module, max_shift, head_dim):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.compute_in_f32 = COMPUTE_IN_F32
        cosine, sine = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cosine, persistent=False)
        self.register_buffer("sin_shift", sine, persistent=False)

    def _flip_key(self, keys):
        keys = onnx_reshape_batch(
            keys, (self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        )
        return onnx_reshape_batch(
            keys.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _shift_key(self, keys, cosine, sine):
        return keys * cosine + self._flip_key(keys) * sine

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        storage_dtype = all_inputs[0].dtype
        force_f32 = self.compute_in_f32 and storage_dtype != torch.float32
        cosine = self.cos_shift.index_select(0, shift).squeeze(0)
        sine = self.sin_shift.index_select(0, shift).squeeze(0)
        if storage_dtype == torch.float32 or force_f32:
            cosine = cosine.float()
            sine = sine.float()

        outputs = []
        for keys in all_inputs[: self.num_layers]:
            if force_f32:
                keys = keys.float()
            shifted = self._shift_key(keys, cosine, sine)
            if force_f32:
                shifted = shifted.to(storage_dtype)
            outputs.append(shifted)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):
    """Dequantize, rebase, and requantize cached full-attention keys."""

    def __init__(
        self,
        num_layers,
        num_kv_heads,
        rotary_module,
        max_shift,
        head_dim,
        quantizer,
        is_asymmetric,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.quantizer = quantizer
        self.is_asymmetric = bool(is_asymmetric)
        cosine, sine = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cosine, persistent=False)
        self.register_buffer("sin_shift", sine, persistent=False)

    def _flip_key(self, keys):
        keys = onnx_reshape_batch(
            keys, (self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        )
        return onnx_reshape_batch(
            keys.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _shift_key(self, keys, cosine, sine):
        return keys * cosine + self._flip_key(keys) * sine

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cosine = self.cos_shift.index_select(0, shift).squeeze(0).float()
        sine = self.sin_shift.index_select(0, shift).squeeze(0).float()
        keys = all_inputs[: self.num_layers]
        scales = all_inputs[self.num_layers : self.num_layers * 2]
        biases = (
            all_inputs[self.num_layers * 2 : self.num_layers * 3]
            if self.is_asymmetric
            else None
        )

        output_keys, output_scales, output_biases = [], [], []
        for layer_index in range(self.num_layers):
            bias = biases[layer_index] if self.is_asymmetric else None
            dequantized = self.quantizer.dequantize_key(
                keys[layer_index], scales[layer_index], bias
            )
            shifted = self._shift_key(dequantized, cosine, sine)
            packed, scale, new_bias = self.quantizer.quantize_key(shifted)
            output_keys.append(packed)
            output_scales.append(scale)
            if self.is_asymmetric:
                output_biases.append(new_bias)
        if self.is_asymmetric:
            return (*output_keys, *output_scales, *output_biases)
        return (*output_keys, *output_scales)


class KVQuantizer(torch.nn.Module):
    """Qwen-compatible Q8/Q4 KV quantizer for PaddleOCR-VL cache tensors."""

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
        self.is_q4 = bool(is_q4)
        self.is_rotary = bool(is_rotary)
        self.is_q8_cuda = bool(is_q8_cuda)
        self.use_sym = bool(use_sym)
        self.use_hadamard = bool(use_hadamard)
        self.use_clip = bool(use_clip)
        self.use_shuffle = bool(use_shuffle)
        self.use_residual_bias_correction = (
            not self.use_sym and not USE_QDQ_FRIENDLY_ASYM
        )
        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.num_kv_heads = int(num_kv_heads)
        self.num_kv_groups = int(num_kv_groups)

        if self.use_sym:
            self.signed_qmin = -8 if self.is_q4 else -128
            self.signed_qmax = 7 if self.is_q4 else 127
            qmax = float(self.signed_qmax)
        else:
            self.signed_qmin = None
            self.signed_qmax = None
            qmax = 15.0 if self.is_q4 else 255.0
        self.register_buffer("inv_qmax", torch.tensor(1.0 / qmax))
        self.register_buffer("tiny", torch.tensor(1.0e-8))

        self.is_grouped = self.is_q4 or (
            (self.use_hadamard or self.use_shuffle)
            and KV_QUANT_GROUP_SIZE < self.head_dim
        )
        if not self.is_grouped and not self.is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = (
            self.head_dim // self.kv_quant_group_size if self.is_grouped else 0
        )

        if self.is_q8_cuda:
            for name, value in (
                ("_256", 256),
                ("_128", 128),
                ("_65536", 65536),
                ("_16777216", 16777216),
            ):
                self.register_buffer(name, torch.tensor(value, dtype=torch.int32))

        if self.is_rotary:
            inverse_sqrt_two = 2.0**-0.5
            self.register_buffer("rot_cos", torch.tensor(inverse_sqrt_two))
            rotary_sin = torch.cat(
                [
                    torch.full((self.head_dim_half,), -inverse_sqrt_two),
                    torch.full((self.head_dim_half,), inverse_sqrt_two),
                ]
            )
            self.register_buffer("rot_sin_k", rotary_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", rotary_sin.view(1, 1, 1, 1, -1))

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer(
                "hadamard_inv_sqrt",
                torch.tensor(self.hadamard_size**-0.5, dtype=torch.float32),
            )
            generator = torch.Generator()
            generator.manual_seed(HADAMARD_RANDOM_SEED)
            signs = torch.randint(
                0,
                2,
                (self.kv_quant_group_size,),
                generator=generator,
                dtype=torch.int64,
            ).float()
            self.register_buffer("hadamard_sign", signs.mul_(2.0).sub_(1.0))
            self.hadamard_levels = []
            width = self.hadamard_size
            while width > 1:
                half = width // 2
                self.hadamard_levels.append((width, half))
                width = half

        if self.use_clip:
            self.register_buffer("clip_sigma", torch.tensor(float(clip_sigma)))

        if self.use_shuffle:
            permutation = (
                torch.arange(self.head_dim)
                .view(self.kv_quant_num_groups, self.kv_quant_group_size)
                .transpose(0, 1)
                .contiguous()
                .view(-1)
            )
            inverse_permutation = torch.empty_like(permutation)
            inverse_permutation[permutation] = torch.arange(self.head_dim)
            self.register_buffer("shuffle_idx", permutation.long())
            self.register_buffer("unshuffle_idx", inverse_permutation.long())

    @staticmethod
    def _next_power_of_two(value):
        result = 1
        while result < value:
            result *= 2
        return result

    def _apply_hadamard_last_dim(self, values, inverse=False):
        if not self.use_hadamard:
            return values
        if not inverse:
            values = values * self.hadamard_sign
        if self.hadamard_pad:
            values = F.pad(values, (0, self.hadamard_pad))
        leading_zeros = (0,) * (values.dim() - 1)
        for width, half in self.hadamard_levels:
            values = onnx_static_reshape(values, leading_zeros + (-1, width))
            even, odd = torch.split(values, [half, half], dim=-1)
            values = torch.cat([even + odd, even - odd], dim=-1)
            values = onnx_static_reshape(values, leading_zeros + (-1,))
        values = values * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            values = values[..., : self.kv_quant_group_size]
        if inverse:
            values = values * self.hadamard_sign
        return values

    def _clip_to_sigma(self, values, dim):
        mean = values.mean(dim=dim, keepdim=True)
        variance = (values - mean).square().mean(dim=dim, keepdim=True)
        bound = self.clip_sigma * variance.sqrt()
        return values.clamp(mean - bound, mean + bound)

    def _flip_k(self, keys):
        keys = onnx_reshape_batch(
            keys, (self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        )
        return onnx_reshape_batch(
            keys.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _flip_v(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, -1, 2, self.head_dim_half)
        )
        return onnx_reshape_batch(
            values.flip(-2), (self.num_kv_heads, 1, -1, self.head_dim)
        )

    def rotate_k(self, keys):
        return keys * self.rot_cos + self._flip_k(keys) * self.rot_sin_k

    def rotate_v(self, values):
        return values * self.rot_cos + self._flip_v(values) * self.rot_sin_v

    def inverse_rotate_k(self, keys):
        return keys * self.rot_cos - self._flip_k(keys) * self.rot_sin_k

    def inverse_rotate_v(self, values):
        return values * self.rot_cos - self._flip_v(values) * self.rot_sin_v

    def hadamard_k(self, keys):
        keys = onnx_reshape_batch(
            keys,
            (
                self.num_kv_heads,
                1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
                -1,
            ),
        )
        keys = self._apply_hadamard_last_dim(keys.transpose(-1, -2)).transpose(-1, -2)
        return onnx_reshape_batch(keys, (self.num_kv_heads, 1, self.head_dim, -1))

    def hadamard_v(self, values):
        values = onnx_reshape_batch(
            values,
            (
                self.num_kv_heads,
                1,
                -1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
            ),
        )
        values = self._apply_hadamard_last_dim(values)
        return onnx_reshape_batch(values, (self.num_kv_heads, 1, -1, self.head_dim))

    def inverse_hadamard_k(self, keys):
        keys = onnx_reshape_batch(
            keys,
            (
                self.num_kv_heads,
                1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
                -1,
            ),
        )
        keys = self._apply_hadamard_last_dim(
            keys.transpose(-1, -2), inverse=True
        ).transpose(-1, -2)
        return onnx_reshape_batch(keys, (self.num_kv_heads, 1, self.head_dim, -1))

    def inverse_hadamard_v(self, values):
        values = onnx_reshape_batch(
            values,
            (
                self.num_kv_heads,
                1,
                -1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
            ),
        )
        values = self._apply_hadamard_last_dim(values, inverse=True)
        return onnx_reshape_batch(values, (self.num_kv_heads, 1, -1, self.head_dim))

    def _finalize_asymmetric_quant(self, values, packed, scale, bias, dim):
        if self.use_residual_bias_correction:
            residual = values - (packed * scale + bias)
            bias = bias + residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            packed = packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            bias = bias.half()
        return packed, scale, bias

    def _quantize_signed_to_storage(self, values, scale):
        packed = torch.round(values / scale).clamp(
            self.signed_qmin, self.signed_qmax
        ).to(torch.int32)
        if self.is_q4:
            return torch.remainder(packed, 16).to(torch.uint8)
        if self.is_q8_cuda:
            return torch.remainder(packed, 256).to(torch.uint8)
        return packed.to(torch.int8)

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

    def _quantize_block(self, values, dim):
        if self.is_grouped:
            return self._quantize_block_grouped(values, dim)
        if self.use_clip:
            values = self._clip_to_sigma(values, dim)
        if self.use_sym:
            scale = torch.maximum(
                values.abs().amax(dim=dim, keepdim=True) * self.inv_qmax,
                self.tiny,
            )
            packed = self._quantize_signed_to_storage(values, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return packed, scale
        bias, maximum = torch.aminmax(values, dim=dim, keepdim=True)
        scale = torch.maximum((maximum - bias) * self.inv_qmax, self.tiny)
        packed = torch.round((values - bias) / scale).clamp(0, 1.0 / self.inv_qmax)
        return self._finalize_asymmetric_quant(values, packed, scale, bias, dim)

    def _quantize_block_grouped(self, values, dim):
        if dim == -2:
            grouped = onnx_reshape_batch(
                values,
                (
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                ),
            )
            reduce_dim = -2
            restore_shape = (self.num_kv_heads, 1, self.head_dim, -1)
        else:
            grouped = onnx_reshape_batch(
                values,
                (
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                ),
            )
            reduce_dim = -1
            restore_shape = (self.num_kv_heads, 1, -1, self.head_dim)
        if self.use_clip:
            grouped = self._clip_to_sigma(grouped, reduce_dim)
        if self.use_sym:
            scale = torch.maximum(
                grouped.abs().amax(dim=reduce_dim, keepdim=True) * self.inv_qmax,
                self.tiny,
            )
            packed = self._quantize_signed_to_storage(grouped, scale)
            packed = onnx_reshape_batch(packed, restore_shape)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return packed, scale
        bias, maximum = torch.aminmax(grouped, dim=reduce_dim, keepdim=True)
        scale = torch.maximum((maximum - bias) * self.inv_qmax, self.tiny)
        packed = torch.round((grouped - bias) / scale).clamp(0, 1.0 / self.inv_qmax)
        packed, scale, bias = self._finalize_asymmetric_quant(
            grouped, packed, scale, bias, reduce_dim
        )
        return onnx_reshape_batch(packed, restore_shape), scale, bias

    def pack_cuda(self, values, dim, packed_head_dim):
        values = values.to(torch.int32)
        if dim == -2:
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, packed_head_dim, 4, -1)
            )
        else:
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, -1, packed_head_dim, 4)
            )
        x0, x1, x2, x3 = torch.unbind(values, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, packed, dim, head_dim):
        remainder3 = packed % self._16777216
        x3 = (packed - remainder3) // self._16777216 + self._128
        x2 = remainder3 // self._65536
        remainder2 = remainder3 % self._65536
        x1 = remainder2 // self._256
        x0 = remainder2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim == -2:
            return onnx_reshape_batch(
                unpacked, (self.num_kv_heads, 1, head_dim, -1)
            )
        return onnx_reshape_batch(unpacked, (self.num_kv_heads, 1, -1, head_dim))

    def pack_q4_k(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        )
        low, high = torch.unbind(values, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, values):
        values = onnx_reshape_batch(
            values, (self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        )
        low, high = torch.unbind(values, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, packed):
        low = packed % 16
        high = packed // 16
        return onnx_reshape_batch(
            torch.stack([low, high], dim=-2),
            (self.num_kv_heads, 1, self.head_dim, -1),
        )

    def unpack_q4_v(self, packed):
        low = packed % 16
        high = packed // 16
        return onnx_reshape_batch(
            torch.stack([low, high], dim=-1),
            (self.num_kv_heads, 1, -1, self.head_dim),
        )

    def quantize_key(self, keys):
        if self.is_rotary:
            keys = self.rotate_k(keys)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys)
        if self.use_sym:
            packed, scale = self._quantize_block(keys, dim=-2)
            bias = None
        else:
            packed, scale, bias = self._quantize_block(keys, dim=-2)
        if self.is_q4:
            packed = self.pack_q4_k(packed)
        if self.is_q8_cuda:
            packed_width = self.head_dim // (8 if self.is_q4 else 4)
            packed = self.pack_cuda(packed, -2, packed_width)
        return packed, scale, bias

    def quantize_value(self, values):
        if self.is_rotary:
            values = self.rotate_v(values)
        if self.use_shuffle:
            values = values.index_select(-1, self.shuffle_idx)
        if self.use_hadamard:
            values = self.hadamard_v(values)
        if self.use_sym:
            packed, scale = self._quantize_block(values, dim=-1)
            bias = None
        else:
            packed, scale, bias = self._quantize_block(values, dim=-1)
        if self.is_q4:
            packed = self.pack_q4_v(packed)
        if self.is_q8_cuda:
            packed_width = self.head_dim // (8 if self.is_q4 else 4)
            packed = self.pack_cuda(packed, -1, packed_width)
        return packed, scale, bias

    def dequantize_key(self, packed, scale, bias=None):
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.float()
            if bias is not None:
                bias = bias.float()
        if self.is_q8_cuda:
            unpack_width = self.head_dim // 2 if self.is_q4 else self.head_dim
            packed = self.unpack_cuda(packed, -2, unpack_width)
        if self.is_q4:
            values = self.unpack_q4_k(packed)
            if self.use_sym:
                values = self._decode_signed_q4_storage(values)
        else:
            values = self._decode_signed_q8_storage(packed) if self.use_sym else packed
        values = values.float()
        if self.is_grouped:
            values = onnx_reshape_batch(
                values,
                (
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                ),
            )
            values = values * scale if self.use_sym else values * scale + bias
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, self.head_dim, -1)
            )
        else:
            values = values * scale if self.use_sym else values * scale + bias
        if self.use_hadamard:
            values = self.inverse_hadamard_k(values)
        if self.use_shuffle:
            values = values.index_select(3, self.unshuffle_idx)
        if self.is_rotary:
            values = self.inverse_rotate_k(values)
        return values

    def dequantize_value(self, packed, scale, bias=None):
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.float()
            if bias is not None:
                bias = bias.float()
        if self.is_q8_cuda:
            unpack_width = self.head_dim // 2 if self.is_q4 else self.head_dim
            packed = self.unpack_cuda(packed, -1, unpack_width)
        if self.is_q4:
            values = self.unpack_q4_v(packed)
            if self.use_sym:
                values = self._decode_signed_q4_storage(values)
        else:
            values = self._decode_signed_q8_storage(packed) if self.use_sym else packed
        values = values.float()
        if self.is_grouped:
            values = onnx_reshape_batch(
                values,
                (
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                ),
            )
            values = values * scale if self.use_sym else values * scale + bias
            values = onnx_reshape_batch(
                values, (self.num_kv_heads, 1, -1, self.head_dim)
            )
        else:
            values = values * scale if self.use_sym else values * scale + bias
        if self.use_hadamard:
            values = self.inverse_hadamard_v(values)
        if self.use_shuffle:
            values = values.index_select(-1, self.unshuffle_idx)
        if self.is_rotary:
            values = self.inverse_rotate_v(values)
        return values

    def forward(self, keys, values):
        packed_k, scale_k, bias_k = self.quantize_key(keys)
        packed_v, scale_v, bias_v = self.quantize_value(values)
        if self.use_sym:
            return packed_k, scale_k, packed_v, scale_v
        return packed_k, scale_k, bias_k, packed_v, scale_v, bias_v


class LLM_MAIN(torch.nn.Module):
    """Target ERNIE-4.5 decoder with mRoPE and explicit persistent KV state."""

    def __init__(
        self,
        model,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.total_qkv_heads = self.qk_heads + self.num_key_value_heads
        self.query_width = self.num_heads * self.head_dim
        self.qkv_split_sizes = [self.qk_heads, self.num_key_value_heads]
        self.qk_split_sizes = [self.num_heads, self.num_key_value_heads]
        if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
            raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")
        kv_flags = _kv_mode_flags(self.head_dim)
        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.kv_f32 = KV_QUANT_DTYPE == "F32"
        self.compute_in_f32 = COMPUTE_IN_F32
        self.kv_quantized = bool(kv_flags["quantized"])
        self.kv_rotary = bool(kv_flags["rotary"])
        self.kv_rotary_q4 = bool(kv_flags["rotary_q4"])
        self.kv_cuda = bool(kv_flags["cuda"])
        self.kv_sym = bool(kv_flags["symmetric"])
        self.kv_grouped_6d = bool(kv_flags["grouped_6d"])
        self.quantizer = KVQuantizer(
            head_dim=self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            num_kv_groups=self.num_key_value_groups,
            is_q4=self.kv_rotary_q4,
            is_rotary=self.kv_rotary,
            is_q8_cuda=self.kv_cuda,
            use_sym=self.kv_sym,
            use_hadamard=USE_HADAMARD,
            use_clip=USE_CLIP,
            clip_sigma=CLIP_SIGMA,
            use_shuffle=USE_SHUFFLE,
        ).eval()
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size**-0.5, dtype=torch.float32),
        )
        self.input_rms_eps: list[float] = []
        self.post_rms_eps: list[float] = []
        self._fuse_weights(model)
        self.layers = model.model.layers
        self.lm_head = model.lm_head
        self.reordered_pairs = 0
        self.reorder_equivalence_error = 0.0
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        self.save_key = [None] * self.num_layers
        self.save_value = [None] * self.num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * self.num_layers
            self.save_v_scale = [None] * self.num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * self.num_layers
                self.save_v_bias = [None] * self.num_layers

    @staticmethod
    def _channel_statistic(weight: torch.Tensor, key: str) -> torch.Tensor:
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(0).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(0).pow(0.25)
        if key == "std":
            return weight.std(0)
        if key == "absmean":
            return absolute.mean(0)
        raise ValueError("REORDER_KEY must be one of: absmean, L4, rms, std.")

    @staticmethod
    def _fused_linear(first, second) -> torch.nn.Linear:
        if first.in_features != second.in_features:
            raise ValueError("Cannot fuse projections with different input widths.")
        if (first.bias is None) != (second.bias is None):
            raise ValueError("Cannot fuse projections with inconsistent bias layouts.")
        fused = torch.nn.Linear(
            first.in_features,
            first.out_features + second.out_features,
            bias=first.bias is not None,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat([first.weight, second.weight], dim=0))
            if fused.bias is not None:
                fused.bias.copy_(torch.cat([first.bias, second.bias], dim=0))
        return fused

    def _fuse_weights(self, model) -> None:
        norm_factor = self.hidden_size**0.5
        with torch.no_grad():
            for layer in model.model.layers:
                self.input_rms_eps.append(float(layer.input_layernorm.variance_epsilon))
                self.post_rms_eps.append(float(layer.post_attention_layernorm.variance_epsilon))
                attn = layer.self_attn
                qkv = self._fused_linear(attn.q_proj, attn.k_proj)
                qkv = self._fused_linear(qkv, attn.v_proj)
                qkv.weight.mul_(
                    (layer.input_layernorm.weight * norm_factor).unsqueeze(0)
                )
                # Scaling Q here is algebraically identical to scaling QK scores.
                qkv.weight[: self.query_width].mul_(self.head_dim**-0.5)
                if qkv.bias is not None:
                    qkv.bias[: self.query_width].mul_(self.head_dim**-0.5)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj, layer.input_layernorm

                mlp = layer.mlp
                gate_up = self._fused_linear(mlp.gate_proj, mlp.up_proj)
                gate_up.weight.mul_(
                    (layer.post_attention_layernorm.weight * norm_factor).unsqueeze(0)
                )
                mlp.gate_up_proj = gate_up
                del mlp.gate_proj, mlp.up_proj, layer.post_attention_layernorm

            final_weight = model.model.norm.weight.unsqueeze(0) * norm_factor
            self.register_buffer("final_norm_scale", final_weight.float())
            self.final_rms_eps = float(model.model.norm.variance_epsilon)
            del model.model.norm

    def _reorder_downproj_for_quant(self, key: str) -> None:
        with torch.no_grad():
            for layer in self.layers:
                mlp = layer.mlp
                intermediate_size = mlp.down_proj.in_features
                if mlp.gate_up_proj.out_features != intermediate_size * 2:
                    raise ValueError("Unexpected gated MLP producer layout for channel reorder.")
                if getattr(mlp.gate_up_proj, "_paddleocr_vl_reordered", False):
                    raise RuntimeError("Refusing to apply a language MLP permutation twice.")
                permutation = torch.argsort(
                    self._channel_statistic(mlp.down_proj.weight.data, key)
                )
                if (
                    permutation.numel() != intermediate_size
                    or torch.unique(permutation).numel() != permutation.numel()
                ):
                    raise RuntimeError("Language MLP permutation is not bijective.")
                probe = torch.randn(
                    2,
                    mlp.gate_up_proj.in_features,
                    dtype=mlp.gate_up_proj.weight.dtype,
                    device=mlp.gate_up_proj.weight.device,
                )
                original_gate_up = mlp.gate_up_proj(probe)
                original_gate, original_up = torch.split(
                    original_gate_up, [intermediate_size, intermediate_size], dim=-1
                )
                before = mlp.down_proj(mlp.act_fn(original_gate) * original_up).float()
                weight = mlp.gate_up_proj.weight.data
                reordered = torch.cat(
                    [weight[:intermediate_size][permutation], weight[intermediate_size:][permutation]],
                    dim=0,
                )
                mlp.gate_up_proj.weight.data.copy_(reordered)
                if mlp.gate_up_proj.bias is not None:
                    bias = mlp.gate_up_proj.bias.data
                    mlp.gate_up_proj.bias.data.copy_(
                        torch.cat(
                            [bias[:intermediate_size][permutation], bias[intermediate_size:][permutation]],
                            dim=0,
                        )
                    )
                mlp.down_proj.weight.data.copy_(mlp.down_proj.weight.data[:, permutation])
                updated_gate_up = mlp.gate_up_proj(probe)
                updated_gate, updated_up = torch.split(
                    updated_gate_up, [intermediate_size, intermediate_size], dim=-1
                )
                after = mlp.down_proj(mlp.act_fn(updated_gate) * updated_up).float()
                mlp.gate_up_proj._paddleocr_vl_reordered = True
                self.reordered_pairs += 1
                self.reorder_equivalence_error = max(
                    self.reorder_equivalence_error, float((before - after).abs().max())
                )

    @staticmethod
    def _rms_norm(x, scale, eps):
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, values):
        return values.view(*values.shape[:-1], 2, self.head_dim_half).flip(-2).view_as(values)

    def _attention(self, query, key, value, attention_mask):
        scores = torch.matmul(query, key)
        scores = scores + attention_mask.to(dtype=scores.dtype)
        scores = torch.softmax(scores, dim=-1)
        return torch.matmul(scores, value)

    def forward(self, *all_inputs):
        state_types = len(_full_state_specs(self.head_dim))
        state_count = self.num_layers * state_types
        hidden_states = all_inputs[state_count]
        rotary_pos_emb_cos = all_inputs[state_count + 1]
        rotary_pos_emb_sin = all_inputs[state_count + 2]
        attention_mask = all_inputs[state_count + 3]
        attention_mask_f16 = (
            attention_mask.half()
            if self.kv_f16 and not self.compute_in_f32
            else None
        )
        batch_size = hidden_states.shape[0]

        for index, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.input_rms_eps[index]
            )
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, values = torch.split(qkv, self.qkv_split_sizes, dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            if self.kv_f16 and not self.compute_in_f32:
                qk = qk.half()
            query, key = torch.split(qk, self.qk_split_sizes, dim=-2)
            query = query.reshape(
                batch_size,
                -1,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).permute(0, 2, 3, 1, 4)
            key = key.permute(0, 3, 2, 4, 1)
            values = values.transpose(1, 3)

            if self.kv_quantized:
                value_sequence_dim = -3 if self.kv_grouped_6d else -2
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(key, values)
                    key_cache = torch.cat([all_inputs[index], packed_k], dim=-1)
                    value_cache = torch.cat(
                        [all_inputs[index + self.num_layers], packed_v], dim=-2
                    )
                    key_scale = torch.cat(
                        [all_inputs[index + self.num_layers * 2], scale_k], dim=-1
                    )
                    value_scale = torch.cat(
                        [all_inputs[index + self.num_layers * 3], scale_v],
                        dim=value_sequence_dim,
                    )
                    self.save_key[index] = key_cache
                    self.save_value[index] = value_cache
                    self.save_k_scale[index] = key_scale
                    self.save_v_scale[index] = value_scale
                    attention_key = self.quantizer.dequantize_key(key_cache, key_scale)
                    attention_value = self.quantizer.dequantize_value(
                        value_cache, value_scale
                    )
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(
                        key, values
                    )
                    key_cache = torch.cat([all_inputs[index], packed_k], dim=-1)
                    value_cache = torch.cat(
                        [all_inputs[index + self.num_layers], packed_v], dim=-2
                    )
                    key_scale = torch.cat(
                        [all_inputs[index + self.num_layers * 2], scale_k], dim=-1
                    )
                    key_bias = torch.cat(
                        [all_inputs[index + self.num_layers * 3], bias_k], dim=-1
                    )
                    value_scale = torch.cat(
                        [all_inputs[index + self.num_layers * 4], scale_v],
                        dim=value_sequence_dim,
                    )
                    value_bias = torch.cat(
                        [all_inputs[index + self.num_layers * 5], bias_v],
                        dim=value_sequence_dim,
                    )
                    self.save_key[index] = key_cache
                    self.save_value[index] = value_cache
                    self.save_k_scale[index] = key_scale
                    self.save_k_bias[index] = key_bias
                    self.save_v_scale[index] = value_scale
                    self.save_v_bias[index] = value_bias
                    attention_key = self.quantizer.dequantize_key(
                        key_cache, key_scale, key_bias
                    )
                    attention_value = self.quantizer.dequantize_value(
                        value_cache, value_scale, value_bias
                    )
            else:
                if self.kv_f16:
                    key = key.half()
                    values = values.half()
                key_cache = torch.cat([all_inputs[index], key], dim=-1)
                value_cache = torch.cat(
                    [all_inputs[index + self.num_layers], values], dim=-2
                )
                self.save_key[index] = key_cache
                self.save_value[index] = value_cache
                if self.kv_f16 and self.compute_in_f32:
                    attention_key = key_cache.float()
                    attention_value = value_cache.float()
                else:
                    attention_key = key_cache
                    attention_value = value_cache

            attention = self._attention(
                query,
                attention_key,
                attention_value,
                attention_mask_f16
                if self.kv_f16 and not self.compute_in_f32
                else attention_mask,
            )
            if self.kv_f16 and not self.compute_in_f32:
                attention = attention.float()
            attention = attention.permute(0, 3, 1, 2, 4).reshape(
                batch_size, -1, self.num_heads * self.head_dim
            )
            hidden_states = residual + layer.self_attn.o_proj(attention)

            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.post_rms_eps[index]
            )
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(
                gate_up,
                [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features],
                dim=-1,
            )
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self._rms_norm(
            hidden_states[:, -1], self.hidden_norm_scale, self.final_rms_eps
        )
        logits = self.lm_head(hidden_states * self.final_norm_scale)
        if self.kv_quantized:
            if self.kv_sym:
                return (
                    *self.save_key,
                    *self.save_value,
                    *self.save_k_scale,
                    *self.save_v_scale,
                    logits,
                )
            return (
                *self.save_key,
                *self.save_value,
                *self.save_k_scale,
                *self.save_k_bias,
                *self.save_v_scale,
                *self.save_v_bias,
                logits,
            )
        return *self.save_key, *self.save_value, logits


def _legacy_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """Recreate the standard RoPE initializer removed from Transformers 5.x."""
    del seq_len, layer_type
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    dimension = int(head_dim * float(getattr(config, "partial_rotary_factor", 1.0)))
    if dimension <= 0 or dimension % 2:
        raise ValueError("Standard RoPE requires a positive, even rotary dimension.")
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    if rope_theta <= 0.0:
        raise ValueError("Standard RoPE requires a positive rope_theta.")
    exponent = torch.arange(
        0, dimension, 2, dtype=torch.int64, device=device
    ).float() / dimension
    return 1.0 / (rope_theta**exponent), 1.0


def _install_legacy_default_rope_initializer() -> None:
    """Make older remote-code checkpoints work with the Transformers 5.x registry."""
    ROPE_INIT_FUNCTIONS.setdefault("default", _legacy_default_rope_parameters)


def _install_paddleocr_rope_compatibility() -> None:
    """Adapt PaddleOCR-VL's legacy rotary module to Transformers 5.x."""
    _install_legacy_default_rope_initializer()
    model_class = get_class_from_dynamic_module(
        "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration",
        str(MODEL_CHECKPOINT),
        local_files_only=True,
    )
    rotary_embedding = getattr(
        sys.modules[model_class.__module__], "RotaryEmbedding", None
    )
    if rotary_embedding is None:
        raise RuntimeError("PaddleOCR-VL custom code does not expose RotaryEmbedding.")
    if not hasattr(rotary_embedding, "compute_default_rope_parameters"):
        rotary_embedding.compute_default_rope_parameters = staticmethod(
            _legacy_default_rope_parameters
        )


def _restore_rotary_frequency_buffers(model) -> None:
    """Rebuild nonpersistent RoPE buffers omitted by low-memory checkpoint loading."""
    rotary_modules = (
        ("language", model.model.rotary_emb),
        ("vision", model.visual.vision_model.encoder.rotary_pos_emb),
    )
    for name, rotary_module in rotary_modules:
        rope_init = getattr(rotary_module, "rope_init", None)
        if not callable(rope_init):
            raise RuntimeError(f"PaddleOCR-VL {name} rotary module cannot rebuild inv_freq.")
        rope_init()
        inv_freq = getattr(rotary_module, "inv_freq", None)
        if (
            inv_freq is None
            or not bool(torch.isfinite(inv_freq).all())
            or not bool((inv_freq > 0.0).all())
            or float(inv_freq.max()) > 1.0
        ):
            raise RuntimeError(f"PaddleOCR-VL {name} rotary inv_freq is invalid after rebuild.")


def _config_int(config, name: str, default=None) -> int:
    value = getattr(config, name, default)
    if value is None:
        raise ValueError(f"Missing required model configuration value: {name}.")
    return int(value)


def _id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _load_paddleocr_components():
    """Load the target through the checkpoint-declared CausalLM mapping."""
    if not MODEL_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"MODEL_CHECKPOINT does not exist: {MODEL_CHECKPOINT}")
    _install_paddleocr_rope_compatibility()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_CHECKPOINT),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(str(MODEL_CHECKPOINT), trust_remote_code=True)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            "PaddleOCR-VL-1.6 requires its checkpoint-declared custom code and "
            "Transformers 4.55.0 or newer."
        ) from error
    for config in (model.config, model.visual.config):
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    _restore_rotary_frequency_buffers(model)
    required = ("model", "visual", "mlp_AR", "lm_head")
    if any(not hasattr(model, name) for name in required):
        raise RuntimeError("Loaded checkpoint does not expose PaddleOCR-VL components.")
    if not hasattr(processor, "tokenizer") or not getattr(processor, "image_token", None):
        raise RuntimeError("PaddleOCR-VL processor does not expose tokenizer and image_token.")
    return model, processor


def _build_prompt_layout(processor, image_token_id: int, image_token_count: int):
    """Use the native template and verify its one contiguous static image span."""
    tokenizer = processor.tokenizer
    image_token = processor.image_token
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": ""}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    if prompt.count(image_token) != 1:
        raise ValueError("Native chat template must emit exactly one image placeholder.")
    expanded = prompt.replace(image_token, image_token * image_token_count)
    token_ids = tokenizer(expanded, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    positions = [
        index for index, token_id in enumerate(token_ids) if token_id == image_token_id
    ]
    if len(positions) != image_token_count or not positions:
        raise ValueError(
            "Native image-token expansion does not match the static vision feature count."
        )
    if positions[-1] - positions[0] + 1 != image_token_count:
        raise ValueError("PaddleOCR-VL image tokens must occupy one contiguous span.")
    return token_ids, positions[0], positions[-1] + 1


def _build_mrope_tables(model, token_ids: list[int], grid_thw: torch.Tensor):
    """Reproduce PaddleOCRVLForConditionalGeneration.get_rope_index for one image."""
    config = model.config
    image_token_id = int(config.image_token_id)
    vision_start_token_id = int(config.vision_start_token_id)
    merge_size = _config_int(config.vision_config, "spatial_merge_size")
    positions: list[torch.Tensor] = []
    token_cursor = 0
    image_cursor = 0
    image_starts = [
        index for index, token_id in enumerate(token_ids) if token_id == image_token_id
    ]
    if not image_starts:
        raise ValueError("Prompt has no expanded PaddleOCR-VL image tokens.")
    vision_starts = [
        index for index, token_id in enumerate(token_ids) if token_id == vision_start_token_id
    ]
    if not vision_starts or vision_starts[0] + 1 != image_starts[0]:
        raise ValueError("Native chat template does not put image tokens after vision_start.")

    while image_cursor < grid_thw.shape[0]:
        try:
            image_start = token_ids.index(image_token_id, token_cursor)
        except ValueError as error:
            raise ValueError("Prompt has fewer image-token regions than image grids.") from error
        text_length = image_start - token_cursor
        current_pos = int(positions[-1].max().item()) + 1 if positions else 0
        positions.append(
            torch.arange(text_length, dtype=torch.float32).view(1, -1).expand(3, -1)
            + current_pos
        )
        time_count, grid_height, grid_width = [
            int(value) for value in grid_thw[image_cursor].tolist()
        ]
        llm_height = grid_height // merge_size
        llm_width = grid_width // merge_size
        image_length = time_count * llm_height * llm_width
        if token_ids[image_start : image_start + image_length] != [image_token_id] * image_length:
            raise ValueError("Expanded image region does not match the model grid token count.")
        time_index = torch.zeros(image_length, dtype=torch.float32)
        height_index = (
            torch.arange(llm_height, dtype=torch.float32)
            .view(1, -1, 1)
            .expand(time_count, -1, llm_width)
            .flatten()
        )
        width_index = (
            torch.arange(llm_width, dtype=torch.float32)
            .view(1, 1, -1)
            .expand(time_count, llm_height, -1)
            .flatten()
        )
        positions.append(
            torch.stack([time_index, height_index, width_index], dim=0)
            + text_length
            + current_pos
        )
        token_cursor = image_start + image_length
        image_cursor += 1

    if token_cursor < len(token_ids):
        current_pos = int(positions[-1].max().item()) + 1 if positions else 0
        text_length = len(token_ids) - token_cursor
        positions.append(
            torch.arange(text_length, dtype=torch.float32).view(1, -1).expand(3, -1)
            + current_pos
        )
    prefill_positions = torch.cat(positions, dim=1)
    next_position = int(prefill_positions.max().item()) + 1
    decode_positions = (
        torch.arange(MAX_SEQ_LEN, dtype=torch.float32).view(1, -1).expand(3, -1)
        + next_position
    )
    all_positions = torch.cat([prefill_positions, decode_positions], dim=1)

    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.float()
    freqs = all_positions[:, :, None] * inv_freq.view(1, 1, -1)
    embeddings = torch.cat([freqs, freqs], dim=-1)
    sections = [int(value) for value in config.rope_scaling["mrope_section"]] * 2
    if sum(sections) != embeddings.shape[-1]:
        raise ValueError("Target mRoPE section widths do not match head_dim.")
    chunks = torch.split(embeddings, sections, dim=-1)
    embeddings = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1
    )
    scaling = float(getattr(rotary, "attention_scaling", 1.0))
    cos = embeddings.cos() * scaling
    sin = embeddings.sin() * scaling
    half = cos.shape[-1] // 2
    # flip() below provides [x2, x1]; negate the first sine half to reproduce
    # the target's native rotate_half result [-x2, x1].
    return cos, torch.cat([-sin[:, :half], sin[:, half:]], dim=-1), next_position


def _build_kv_layout(
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    history_len: int,
):
    flags = _kv_mode_flags(head_dim)
    quantized = bool(flags["quantized"])
    if KV_QUANT_DTYPE == "F16":
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE == "F32":
        kv_dtype = torch.float32
    elif KV_QUANT_DTYPE in KV_CUDA_DTYPES:
        kv_dtype = torch.int32
    elif bool(flags["symmetric"]) and not bool(flags["rotary_q4"]):
        kv_dtype = torch.int8 if USE_SYM else torch.uint8
    else:
        kv_dtype = torch.uint8
    specs = _full_state_specs(head_dim)
    storage_width = int(flags["storage_width"])
    tensors = {
        "key": torch.zeros(
            (batch_size, num_kv_heads, 1, storage_width, history_len), dtype=kv_dtype
        ),
        "value": torch.zeros(
            (batch_size, num_kv_heads, 1, history_len, storage_width), dtype=kv_dtype
        ),
    }
    scale_dtype = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32
    if quantized:
        if bool(flags["grouped_6d"]):
            group_count = head_dim // KV_QUANT_GROUP_SIZE
            key_scale_shape = (
                batch_size,
                num_kv_heads,
                1,
                group_count,
                1,
                history_len,
            )
            value_scale_shape = (
                batch_size,
                num_kv_heads,
                1,
                history_len,
                group_count,
                1,
            )
        else:
            group_count = 1
            key_scale_shape = (batch_size, num_kv_heads, 1, 1, history_len)
            value_scale_shape = (batch_size, num_kv_heads, 1, history_len, 1)
        tensors["key_scale"] = torch.ones(key_scale_shape, dtype=scale_dtype)
        tensors["value_scale"] = torch.ones(value_scale_shape, dtype=scale_dtype)
        if not bool(flags["symmetric"]):
            tensors["key_bias"] = torch.zeros(key_scale_shape, dtype=scale_dtype)
            tensors["value_bias"] = torch.zeros(value_scale_shape, dtype=scale_dtype)
    actual_group_size = (
        KV_QUANT_GROUP_SIZE if bool(flags["grouped_6d"]) else head_dim
    )
    facts = {
        "kv_cache_quantization": KV_QUANT_DTYPE,
        "kv_cache_tensor_order": ",".join(name for name, _ in specs),
        "kv_cache_key_layout": "batch,key_value_heads,one,head_dim,sequence",
        "kv_cache_value_layout": "batch,key_value_heads,one,sequence,head_dim",
        "kv_cache_key_sequence_axis": "4",
        "kv_cache_value_sequence_axis": "3",
        "kv_cache_key_storage_width": str(storage_width),
        "kv_cache_value_storage_width": str(storage_width),
        "kv_cache_quantized": str(int(quantized)),
        "kv_cache_symmetric": str(int(bool(flags["symmetric"]))),
        "kv_cache_grouped_6d": str(int(bool(flags["grouped_6d"]))),
        "kv_cache_group_size": str(actual_group_size if quantized else 0),
        "kv_cache_group_count": str(group_count if quantized else 0),
        "kv_cache_storage_dtype": str(kv_dtype).replace("torch.", ""),
        "kv_cache_scale_bias_dtype": str(scale_dtype).replace("torch.", "")
        if quantized
        else "none",
        "kv_cache_compute_in_f32": str(int(COMPUTE_IN_F32 and KV_QUANT_DTYPE == "F16")),
        "kv_quant_dtype": KV_QUANT_DTYPE,
        "kv_quant_group_size": str(KV_QUANT_GROUP_SIZE if quantized else 0),
        "kv_symmetric": str(int(bool(flags["symmetric"]))),
        "kv_grouped_6d": str(int(bool(flags["grouped_6d"]))),
        "kv_cache_elem_type": str(kv_dtype).replace("torch.", ""),
        "kv_scale_bias_elem_type": str(scale_dtype).replace("torch.", "")
        if quantized
        else "none",
        "kv_quant_hadamard": str(int(USE_HADAMARD)) if quantized else "0",
        "kv_quant_shuffle": str(int(USE_SHUFFLE)) if quantized else "0",
        "kv_quant_clip": str(int(USE_CLIP)) if quantized else "0",
        "kv_qdq_friendly_asym": str(int(USE_QDQ_FRIENDLY_ASYM))
        if quantized and not bool(flags["symmetric"])
        else "0",
    }
    return specs, tensors, facts


def _kv_io(kv_specs, kv_tensors, num_layers: int):
    inputs, input_names, output_names, dynamic_axes = [], [], [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
            dynamic_axes[output_name] = {0: "batch_size", sequence_axis: "kv_seq_len"}
    return inputs, input_names, output_names, dynamic_axes


def _tensor_equal(left: torch.Tensor, right: torch.Tensor, rows: int = 256) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    for start in range(0, left.shape[0], rows):
        if not torch.equal(left[start : start + rows], right[start : start + rows]):
            return False
    return True


def _metadata_values(
    model,
    processor,
    dimensions: dict[str, int],
    kv_facts: dict[str, str],
    image_start: int,
    image_end: int,
    grid_thw: torch.Tensor,
    vision: LLM_VISION,
    main: LLM_MAIN,
    mrope_decode_start: int,
) -> dict[str, str]:
    tokenizer = processor.tokenizer
    eos_ids = _id_list(getattr(model.config, "eos_token_id", None))
    if not eos_ids:
        eos_ids = _id_list(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    if not eos_ids:
        eos_ids = _id_list(getattr(tokenizer, "eos_token_id", None))
    if not eos_ids:
        raise RuntimeError("PaddleOCR-VL checkpoint exposes no EOS token ID.")
    embed_weight = model.model.embed_tokens.weight
    head_weight = model.lm_head.weight
    same_storage = embed_weight.untyped_storage().data_ptr() == head_weight.untyped_storage().data_ptr()
    weights_equal = _tensor_equal(embed_weight, head_weight)
    metadata = {
        "model_architecture": "PaddleOCR-VL-1.6 ERNIE-4.5 + SigLIP vision",
        "max_seq_len": str(MAX_SEQ_LEN),
        "input_image_size": f"{INPUT_IMAGE_HEIGHT},{INPUT_IMAGE_WIDTH}",
        "vision_image_size": f"{VISION_IMAGE_HEIGHT},{VISION_IMAGE_WIDTH}",
        "input_image_dim": str(INPUT_IMAGE_DIM),
        "vision_batch_size": str(VISION_BATCH_SIZE),
        "image_token_id": str(int(model.config.image_token_id)),
        "vision_start_token_id": str(int(model.config.vision_start_token_id)),
        "stop_token_ids": ",".join(str(token_id) for token_id in eos_ids),
        "eos_token_ids": ",".join(str(token_id) for token_id in eos_ids),
        "image_token_length": str(image_end - image_start),
        "image_start": str(image_start),
        "image_end": str(image_end),
        "image_grid_thw": ",".join(str(int(value)) for value in grid_thw.flatten()),
        "mrope_decode_start": str(mrope_decode_start),
        "mrope_section": ",".join(str(value) for value in model.config.rope_scaling["mrope_section"]),
        "mrope_type": "3d",
        "kv_num_tensors": str(
            dimensions["num_layers"] * len(kv_facts["kv_cache_tensor_order"].split(","))
        ),
        "fused_simplified_layer_norm_count": str(dimensions["num_layers"] * 2 + 1),
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reorder_key": REORDER_KEY,
        "reordered_language_pairs": str(main.reordered_pairs),
        "reordered_vision_pairs": str(vision.reordered_pairs),
        "decoder_attention_scale_fused": "1",
        "fused_vision_qkv_count": str(vision.fused_vision_qkv_count),
        "folded_vision_layer_norm_count": str(
            vision.folded_vision_layer_norm_count
        ),
        "vision_attention_scale_fused": "1",
        "vision_static_tables_precomputed": "1",
        "vision_static_tables_shared": "1",
        "language_reorder_equivalence_error": f"{main.reorder_equivalence_error:.9g}",
        "vision_reorder_equivalence_error": f"{vision.reorder_equivalence_error:.9g}",
        "embed_lm_head_equal": str(int(weights_equal)),
        "embed_lm_head_shared_storage": str(int(same_storage)),
        "compute_in_f32": str(int(COMPUTE_IN_F32)),
    }
    metadata.update(kv_facts)
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


def _stamp_metadata(path: Path, metadata: dict[str, str]) -> None:
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _deduplicate_large_constant_tensors(
    path: Path, minimum_elements: int = 16_384
) -> int:
    """Share repeated static tensors emitted as per-layer ONNX Constant nodes."""
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    existing_names = {initializer.name for initializer in graph.initializer}
    signatures: dict[bytes, str] = {}
    replacements: dict[str, str] = {}
    retained_nodes = []
    converted = 0

    for node in graph.node:
        tensor = None
        if node.op_type == "Constant" and len(node.output) == 1:
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    tensor = attribute.t
                    break
        element_count = 1
        if tensor is not None:
            for dimension in tensor.dims:
                element_count *= int(dimension)
        if tensor is None or element_count < minimum_elements:
            retained_nodes.append(node)
            continue

        shared_tensor = onnx.TensorProto()
        shared_tensor.CopyFrom(tensor)
        shared_tensor.name = ""
        signature = shared_tensor.SerializeToString()
        shared_name = signatures.get(signature)
        if shared_name is None:
            shared_name = f"static_constant_{len(signatures)}"
            while shared_name in existing_names:
                shared_name += "_"
            shared_tensor.name = shared_name
            graph.initializer.append(shared_tensor)
            signatures[signature] = shared_name
            existing_names.add(shared_name)
        replacements[node.output[0]] = shared_name
        converted += 1

    if not replacements:
        return 0
    for node in retained_nodes:
        for index, name in enumerate(node.input):
            node.input[index] = replacements.get(name, name)
    del graph.node[:]
    graph.node.extend(retained_nodes)
    onnx.save_model(model, str(path))
    return converted


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
        external_data=True,
    )
    _stamp_metadata(path, metadata)
    print(f"Exported {path.name}.", flush=True)


def _copy_tokenizer_assets(source: Path, destination: Path) -> list[str]:
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration_paddleocr_vl.py",
        "generation_config.json",
        "image_processing_paddleocr_vl.py",
        "preprocessor_config.json",
        "processing_paddleocr_vl.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in sorted(names):
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, destination / name)
            copied.append(name)
    if not any(name in copied for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError("Target checkpoint has no tokenizer serialization.")
    return copied


def _cleanup_unreferenced_data(export_dir: Path) -> None:
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
        if data_path.is_file() and data_path.name.endswith(".data") and data_path.name not in referenced:
            data_path.unlink()


def _prepare_export_staging() -> Path:
    if EXPORT_STAGING_DIR.exists():
        if not EXPORT_STAGING_DIR.is_dir():
            raise NotADirectoryError(
                f"Export staging path exists but is not a directory: {EXPORT_STAGING_DIR}."
            )
        shutil.rmtree(EXPORT_STAGING_DIR)
    EXPORT_STAGING_DIR.mkdir(parents=True)
    return EXPORT_STAGING_DIR


def _promote_export(staging_dir: Path) -> None:
    """Atomically promote a validated bundle while retaining one rollback bundle."""
    destination = EXPORT_DIR
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    moved_current = False
    try:
        if destination.exists():
            destination.rename(previous)
            moved_current = True
        staging_dir.rename(destination)
    except BaseException:
        if moved_current and previous.exists() and not destination.exists():
            previous.rename(destination)
        raise


def _validate_staging_bundle(
    export_dir: Path, metadata: dict[str, str], expected_fused_norms: int
) -> None:
    import onnxruntime

    import Shared_Merged

    names = {key: value for key, value in MODEL_FILE_NAMES.items()}
    required = [
        names["metadata"],
        names["image_preprocess"],
        names["vision"],
        names["kv_slice"],
        names["kv_split2"],
        names["kv_concat"],
        names["rope_shift"],
        names["shared_initializers"],
        names["shared_initializers_data"],
    ]
    required.extend(names[f"image_{phase}_{strategy}"] for phase in ("prefill", "decode") for strategy in ("greedy", "penalty_greedy", "sampling"))
    missing = [name for name in required if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Validated bundle is incomplete: {missing!r}")
    for path in export_dir.glob("*.onnx"):
        Shared_Merged.validate_onnx_path(path)
    for phase in ("prefill", "decode"):
        for strategy in ("greedy", "penalty_greedy", "sampling"):
            path = export_dir / names[f"image_{phase}_{strategy}"]
            graph = onnx.load(str(path), load_external_data=False)
            count = sum(
                not node.domain and node.op_type == "SimplifiedLayerNormalization"
                for node in graph.graph.node
            )
            if count != expected_fused_norms:
                raise RuntimeError(
                    f"{path.name} has {count} fused RMSNorm nodes, expected {expected_fused_norms}."
                )
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    metadata_session = onnxruntime.InferenceSession(
        str(export_dir / names["metadata"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = dict(metadata_session.get_modelmeta().custom_metadata_map)
    if actual.get("model_architecture") != metadata["model_architecture"]:
        raise RuntimeError("Metadata carrier did not retain target architecture metadata.")
    vision_options = onnxruntime.SessionOptions()
    vision_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    onnxruntime.InferenceSession(
        str(export_dir / names["image_preprocess"]),
        sess_options=vision_options,
        providers=["CPUExecutionProvider"],
    )
    onnxruntime.InferenceSession(
        str(export_dir / names["vision"]),
        sess_options=vision_options,
        providers=["CPUExecutionProvider"],
    )
    for role in ("kv_slice", "kv_split2", "kv_concat", "rope_shift"):
        onnxruntime.InferenceSession(
            str(export_dir / names[role]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )


@torch.inference_mode()
def export_paddleocr_vl() -> None:
    if INPUT_IMAGE_DIM != 4:
        raise ValueError("PaddleOCR-VL exporter currently requires INPUT_IMAGE_DIM=4.")
    export_dir = _prepare_export_staging()
    model, processor = _load_paddleocr_components()
    config = model.config
    vision_config = config.vision_config
    dimensions = {
        "num_layers": _config_int(config, "num_hidden_layers"),
        "num_heads": _config_int(config, "num_attention_heads"),
        "num_kv_heads": _config_int(config, "num_key_value_heads"),
        "hidden_size": _config_int(config, "hidden_size"),
        "vocab_size": _config_int(config, "vocab_size"),
    }
    dimensions["head_dim"] = _config_int(
        config, "head_dim", dimensions["hidden_size"] // dimensions["num_heads"]
    )
    if dimensions["num_heads"] % dimensions["num_kv_heads"]:
        raise ValueError("num_attention_heads must divide num_key_value_heads.")
    for note in normalize_kv_quant_settings(dimensions["head_dim"]):
        print(note)
    if MAX_SEQ_LEN <= 0 or MAX_SEQ_LEN > _config_int(config, "max_position_embeddings"):
        raise ValueError("PADDLEOCR_VL_MAX_SEQ_LEN must be positive and within model context.")
    patch_size = _config_int(vision_config, "patch_size")
    merge_size = _config_int(vision_config, "spatial_merge_size")
    if VISION_IMAGE_HEIGHT % (patch_size * merge_size) or VISION_IMAGE_WIDTH % (patch_size * merge_size):
        raise ValueError("Vision image dimensions must be divisible by patch_size * spatial_merge_size.")
    grid_height = VISION_IMAGE_HEIGHT // patch_size
    grid_width = VISION_IMAGE_WIDTH // patch_size
    grid_thw = torch.tensor([[1, grid_height, grid_width]], dtype=torch.int64)
    image_token_count = (grid_height // merge_size) * (grid_width // merge_size)
    token_ids, image_start, image_end = _build_prompt_layout(
        processor, int(config.image_token_id), image_token_count
    )
    if image_end - image_start != image_token_count:
        raise RuntimeError("Image token span does not match projector feature geometry.")
    rotary_cos, rotary_sin, mrope_decode_start = _build_mrope_tables(
        model, token_ids, grid_thw
    )

    vision = LLM_VISION(model, grid_thw, REORDER_KEY).eval()
    main = LLM_MAIN(
        model,
        dimensions["num_heads"],
        dimensions["num_kv_heads"],
        dimensions["head_dim"],
        dimensions["num_layers"],
        dimensions["hidden_size"],
    ).eval()
    kv_specs, kv_tensors, kv_facts = _build_kv_layout(
        1,
        dimensions["num_layers"],
        dimensions["num_kv_heads"],
        dimensions["head_dim"],
        0,
    )
    metadata = _metadata_values(
        model,
        processor,
        dimensions,
        kv_facts,
        image_start,
        image_end,
        grid_thw,
        vision,
        main,
        mrope_decode_start,
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
    trace_ids_len = min(16, len(token_ids))
    if trace_ids_len <= 0:
        raise ValueError("Native chat template produced an empty prompt.")
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

    image_processor = processor.image_processor
    preprocess = LLM_IMAGE_PREPROCESS(
        VISION_IMAGE_HEIGHT,
        VISION_IMAGE_WIDTH,
        patch_size,
        [float(value) for value in image_processor.image_mean],
        [float(value) for value in image_processor.image_std],
    ).eval()
    raw_image = torch.zeros(
        (VISION_BATCH_SIZE, 3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=torch.uint8
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES["image_preprocess"],
        preprocess,
        (raw_image,),
        ["pixel_values"],
        ["patches"],
        {"pixel_values": {2: "image_height", 3: "image_width"}},
        metadata,
    )
    del preprocess, raw_image
    gc.collect()

    patches = torch.zeros((int(torch.prod(grid_thw[0]).item()), 3, patch_size, patch_size), dtype=torch.float32)
    _export_component(
        export_dir / MODEL_FILE_NAMES["vision"],
        vision,
        (patches,),
        ["patches"],
        ["vision_hidden_states"],
        None,
        metadata,
    )
    _deduplicate_large_constant_tensors(export_dir / MODEL_FILE_NAMES["vision"])
    del patches
    gc.collect()

    text_hidden_states = torch.ones(
        (1, len(token_ids), dimensions["hidden_size"]), dtype=torch.float32
    )
    vision_hidden_states = torch.ones(
        (1, image_token_count, dimensions["hidden_size"]), dtype=torch.float32
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES["concat_image"],
        LLM_CONCAT_IMAGE(image_start, image_end),
        (text_hidden_states, vision_hidden_states),
        ["text_hidden_states", "vision_hidden_states"],
        ["concat_hidden_states"],
        {
            "text_hidden_states": {0: "batch_size", 1: "ids_len"},
            "vision_hidden_states": {0: "batch_size", 1: "image_token_count"},
            "concat_hidden_states": {0: "batch_size", 1: "ids_len"},
        },
        metadata,
    )
    del text_hidden_states, vision_hidden_states
    gc.collect()

    ids_len = torch.tensor([trace_ids_len], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    kv_seq_len = ids_len + history_len
    _export_component(
        export_dir / MODEL_FILE_NAMES["rotary_image_prefill"],
        ROTARY_IMAGE_PREFILL(rotary_cos, rotary_sin),
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
        export_dir / MODEL_FILE_NAMES["rotary_image_decode"],
        ROTARY_IMAGE_DECODE(rotary_cos, rotary_sin),
        (kv_seq_len,),
        ["kv_seq_len"],
        ["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        None,
        metadata,
    )
    del rotary_cos, rotary_sin
    gc.collect()

    kv_inputs, kv_input_names, kv_output_names, kv_axes = _kv_io(
        kv_specs, kv_tensors, dimensions["num_layers"]
    )
    hidden_states = torch.ones(
        (1, trace_ids_len, dimensions["hidden_size"]), dtype=torch.float32
    )
    rotary_shape = (1, trace_ids_len, 1, 1, dimensions["head_dim"])
    main_rotary_cos = torch.zeros(rotary_shape, dtype=torch.float32)
    main_rotary_sin = torch.zeros_like(main_rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, trace_ids_len, trace_ids_len), dtype=torch.float32)
    main_input_names = kv_input_names + [
        "hidden_states",
        "rotary_cos",
        "rotary_sin",
        "attention_mask",
    ]
    main_output_names = kv_output_names + ["logits"]
    main_axes = {
        **kv_axes,
        "hidden_states": {0: "batch_size", 1: "ids_len"},
        "logits": {0: "batch_size"},
        "rotary_cos": {1: "ids_len"},
        "rotary_sin": {1: "ids_len"},
        "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
    }
    _export_component(
        export_dir / MODEL_FILE_NAMES["main"],
        main,
        tuple(kv_inputs + [hidden_states, main_rotary_cos, main_rotary_sin, attention_mask]),
        main_input_names,
        main_output_names,
        main_axes,
        metadata,
    )

    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([min(5, trace_ids_len)], dtype=torch.int64)
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_slice"],
        KV_SLICE(kv_specs, dimensions["num_layers"]),
        tuple(kv_inputs + [slice_start, slice_end]),
        kv_input_names + ["slice_start", "slice_end"],
        kv_output_names,
        kv_axes,
        metadata,
    )
    del slice_start, slice_end

    split_at = torch.tensor([min(5, trace_ids_len)], dtype=torch.int64)
    split_prefix_names = [f"prefix_{name}" for name in kv_output_names]
    split_suffix_names = [f"suffix_{name}" for name in kv_output_names]
    split_axes = {name: dict(kv_axes[name]) for name in kv_input_names}
    for output_name, prefix_name, suffix_name in zip(
        kv_output_names, split_prefix_names, split_suffix_names
    ):
        source_axes = kv_axes[output_name]
        split_axes[prefix_name] = dict(source_axes)
        split_axes[suffix_name] = dict(source_axes)
        for axis in split_axes[prefix_name]:
            if axis != 0:
                split_axes[prefix_name][axis] = "prefix_len"
                split_axes[suffix_name][axis] = "suffix_len"
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_split2"],
        KV_SPLIT2(kv_specs, dimensions["num_layers"]),
        tuple(kv_inputs + [split_at]),
        kv_input_names + ["split_at"],
        split_prefix_names + split_suffix_names,
        split_axes,
        metadata,
    )
    del split_at, split_prefix_names, split_suffix_names, split_axes

    concat_prefix_inputs, concat_prefix_names = [], []
    concat_suffix_inputs, concat_suffix_names = [], []
    concat_output_names, concat_axes = [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(dimensions["num_layers"]):
            prefix_name = f"in_prefix_{name}_{layer_index}"
            suffix_name = f"in_suffix_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            concat_prefix_inputs.append(tensor)
            concat_prefix_names.append(prefix_name)
            concat_suffix_inputs.append(tensor.clone())
            concat_suffix_names.append(suffix_name)
            concat_output_names.append(output_name)
            concat_axes[prefix_name] = {0: "batch_size", sequence_axis: "prefix_len"}
            concat_axes[suffix_name] = {0: "batch_size", sequence_axis: "suffix_len"}
            concat_axes[output_name] = {0: "batch_size", sequence_axis: "concat_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_concat"],
        KV_CONCAT(kv_specs, dimensions["num_layers"]),
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

    def _rope_shift_inputs(state_tensors):
        inputs, input_names, output_names, axes = [], [], [], {}
        for name, tensor in state_tensors:
            sequence_axis = tensor.dim() - 1
            for layer_index in range(dimensions["num_layers"]):
                input_name = f"in_{name}_{layer_index}"
                output_name = f"out_{name}_{layer_index}"
                inputs.append(tensor)
                input_names.append(input_name)
                output_names.append(output_name)
                axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
                axes[output_name] = {0: "batch_size", sequence_axis: "history_len"}
        return inputs, input_names, output_names, axes

    def _with_sequence_length(tensor, length):
        shape = list(tensor.shape)
        shape[-1] = length
        return torch.zeros(shape, dtype=tensor.dtype)

    rope_shift_amount = torch.tensor([min(5, MAX_SEQ_LEN)], dtype=torch.int64)
    flags = _kv_mode_flags(dimensions["head_dim"])
    if not bool(flags["quantized"]):
        rope_specs = [("key", _with_sequence_length(kv_tensors["key"], 4))]
        rope_module = ROPE_SHIFT(
            dimensions["num_layers"],
            dimensions["num_kv_heads"],
            model.model.rotary_emb,
            MAX_SEQ_LEN,
            dimensions["head_dim"],
        )
    else:
        rope_specs = [
            ("key", _with_sequence_length(kv_tensors["key"], 4)),
            ("key_scale", _with_sequence_length(kv_tensors["key_scale"], 4)),
        ]
        if not bool(flags["symmetric"]):
            rope_specs.append(
                ("key_bias", _with_sequence_length(kv_tensors["key_bias"], 4))
            )
        rope_module = ROPE_SHIFT_QUANT(
            dimensions["num_layers"],
            dimensions["num_kv_heads"],
            model.model.rotary_emb,
            MAX_SEQ_LEN,
            dimensions["head_dim"],
            main.quantizer,
            not bool(flags["symmetric"]),
        )
    rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_inputs(
        rope_specs
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
        _rope_shift_inputs,
        _with_sequence_length,
        rope_shift_amount,
        rope_specs,
        rope_module,
        rope_inputs,
        rope_input_names,
        rope_output_names,
        rope_axes,
        flags,
    )

    del kv_inputs, hidden_states, main_rotary_cos, main_rotary_sin, attention_mask
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

    import Shared_Merged

    bundle = Shared_Merged.build_shared_merged_bundle(
        export_dir,
        model_file_names=MODEL_FILE_NAMES,
        delete_constituents=True,
    )
    _stamp_metadata(bundle["shared_model"], metadata)
    for path in bundle["graphs"].values():
        _stamp_metadata(path, metadata)
    _cleanup_unreferenced_data(export_dir)
    tokenizer_assets = _copy_tokenizer_assets(MODEL_CHECKPOINT, export_dir)
    _validate_staging_bundle(
        export_dir, metadata, int(metadata["fused_simplified_layer_norm_count"])
    )
    _promote_export(export_dir)
    print(
        f"PaddleOCR-VL ONNX export completed: {EXPORT_DIR} "
        f"({len(tokenizer_assets)} tokenizer/processor assets)."
    )


def export_bundle() -> None:
    """Export the PaddleOCR-VL ONNX bundle."""
    export_paddleocr_vl()


def main() -> None:
    if not DO_EXPORT:
        print("PADDLEOCR_VL_DO_EXPORT is false; no ONNX files were written.")
        return
    export_bundle()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_PaddleOCRVL_ONNX.py"),
            "--model-folder",
            str(EXPORT_DIR),
            "--expect-nonempty-output",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()