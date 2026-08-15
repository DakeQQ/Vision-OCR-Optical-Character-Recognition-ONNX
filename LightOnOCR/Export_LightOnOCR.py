"""Export LightOnOCR as a metadata-driven, image-only merged ONNX bundle.

LightOnOCR combines a Pixtral vision tower with a tied-embedding Qwen3 decoder.
The vision path is deliberately split into preprocessing, encoder/projector, and
dynamic image-slot replacement graphs; no prompt text is captured in ONNX.
"""

from __future__ import annotations

import gc
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.lighton_ocr.modeling_lighton_ocr import LightOnOcrForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

EXPORT_DIR = SCRIPT_DIR / "LightOnOCR_ONNX"
EXPORT_STAGING_DIR = EXPORT_DIR.with_name(EXPORT_DIR.name + ".staging")
CHECKPOINT_DIR = Path.home() / "Downloads" / "LightOnOCR-2-1B"
# Backward-compatible configuration alias.
DOWNLOAD_PATH = CHECKPOINT_DIR

# Export controls
DO_EXPORT            = True                    # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW = False                   # Prevent float16 overflow for Q4F16, Q8F16, or F16 quantization.
STOP_TOKEN           = [151643, 151645]        # LightOnOCR stop token ids.
MAX_SEQ_LEN          = 4096                    # Fixed maximum context length after export.

# Image input and vision tracing
# The static image grid is intentionally independent from source image size.
HEIGHT_FACTOR       = 25                       # Vertical factor for the exported image grid.
WIDTH_FACTOR        = 25                       # Horizontal factor for the exported image grid.
# Resize is applied before patchification.
IMAGE_RESIZE        = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]
INPUT_IMAGE_SIZE    = [980, 980]               # Input image shape before ONNX preprocessing.
VISION_BATCH_SIZE   = 1                        # Number of images supported by the prompt.
DYNAMIC_IMAGE_SHAPE = False                    # Keep the exported image grid static.
INPUT_IMAGE_DIM     = 5                        # pixel_values rank: 4=[B, C, H, W]; 5=[B, 1, C, H, W].
# Image normalization mean.
CLIP_IMAGE_MEAN     = [0.48145466, 0.4578275, 0.40821073]
# Image normalization standard deviation.
CLIP_IMAGE_STD      = [0.26862954, 0.26130258, 0.27577711]
IMAGE_TOKEN_LENGTH  = HEIGHT_FACTOR * WIDTH_FACTOR

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                     # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 128                      # Quantization group width; auto-clamped to a divisor.
COMPUTE_IN_F32      = False                    # F16 KV only: False=f16 attention, True=upcast KV for f32 math.

# KV quantization transforms and parameters
USE_HADAMARD           = False                 # Apply a randomized Hadamard transform before grouped quantization.
HADAMARD_RANDOM_SEED   = 9527                  # Deterministic sign pattern for the Hadamard transform.
USE_CLIP               = False                 # Clip quantization blocks to CLIP_SIGMA standard deviations.
CLIP_SIGMA             = 3.0                   # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False                 # Interleave channels across quantization groups.
USE_SYM                = True                  # Use symmetric rather than scale-and-bias quantization.
USE_FLOAT16_SCALE_BIAS = True                  # Store quantization scales and biases as float16.

# Quantization-oriented model reordering
REORDER_DOWNPROJ_FOR_QUANT    = True           # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT  = True           # Reorder vision MLP channels before quantization.
REORDER_KEY                   = "absmean"      # Channel statistic used to build the permutation.
_REORDER_KEYS                 = frozenset(("absmean", "L4", "rms", "std"))
# Relative error tolerance for reorder equivalence checks.
REORDER_EQUIVALENCE_RELATIVE_L2 = 1e-5

# ONNX graph format
OPSET = 20                                     # ONNX opset version.

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
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"
RUNTIME_MODEL_FILE_ROLES = (
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
    "rope_shift",
)
MODEL_FILE_NAME_METADATA = {
    f"model_file_name_{key}": MODEL_FILE_NAMES[key]
    for key in RUNTIME_MODEL_FILE_ROLES
}


def _load_reference_export():
    """Load finalized generic Qwen3/KV kernels without executing its exporter."""
    reference_path = SCRIPT_DIR.parent / "FireRedOCR" / "Export_FireRedOCR.py"
    if not reference_path.is_file():
        raise RuntimeError(f"Missing finalized reference exporter: {reference_path}")
    spec = importlib.util.spec_from_file_location("_lighton_reference_export", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load finalized reference exporter: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REFERENCE = _load_reference_export()


def _configure_reference_kv() -> None:
    """Give the shared finalized Qwen3 kernel this target's KV/reorder settings."""
    _REFERENCE.KV_QUANT_DTYPE = KV_QUANT_DTYPE
    _REFERENCE.KV_QUANT_GROUP_SIZE = KV_QUANT_GROUP_SIZE
    _REFERENCE.COMPUTE_IN_F32 = COMPUTE_IN_F32
    _REFERENCE.USE_HADAMARD = USE_HADAMARD
    _REFERENCE.HADAMARD_RANDOM_SEED = HADAMARD_RANDOM_SEED
    _REFERENCE.USE_CLIP = USE_CLIP
    _REFERENCE.CLIP_SIGMA = CLIP_SIGMA
    _REFERENCE.USE_SHUFFLE = USE_SHUFFLE
    _REFERENCE.USE_SYM = USE_SYM
    _REFERENCE.USE_FLOAT16_SCALE_BIAS = USE_FLOAT16_SCALE_BIAS
    _REFERENCE.PREVENT_F16_OVERFLOW = PREVENT_F16_OVERFLOW
    _REFERENCE.REORDER_DOWNPROJ_FOR_QUANT = REORDER_DOWNPROJ_FOR_QUANT
    _REFERENCE.REORDER_KEY = REORDER_KEY


def _require_reorder_key(key: str) -> None:
    if key not in _REORDER_KEYS:
        raise ValueError(f"REORDER_KEY must be one of {sorted(_REORDER_KEYS)}, got {key!r}.")


def _channel_statistic(weight: torch.Tensor, key: str) -> torch.Tensor:
    _require_reorder_key(key)
    absolute = weight.abs()
    if key == "rms":
        return (weight.square().mean(0)).sqrt()
    if key == "L4":
        return absolute.pow(4).mean(0).pow(0.25)
    if key == "std":
        return weight.std(0)
    return absolute.mean(0)


def _validated_permutation(weight: torch.Tensor, key: str) -> torch.Tensor:
    permutation = torch.argsort(_channel_statistic(weight, key))
    if permutation.ndim != 1 or permutation.numel() != weight.shape[1]:
        raise RuntimeError("Channel-reorder permutation has an unexpected shape.")
    if torch.unique(permutation).numel() != permutation.numel():
        raise RuntimeError("Channel-reorder permutation is not bijective.")
    return permutation


def _reorder_error(before: torch.Tensor, after: torch.Tensor, label: str) -> tuple[float, float]:
    delta = before - after
    maximum = float(delta.abs().max())
    relative_l2 = float(delta.norm() / before.norm().clamp_min(1e-12))
    if relative_l2 > REORDER_EQUIVALENCE_RELATIVE_L2:
        raise RuntimeError(
            f"{label} reorder changed output by max_abs={maximum}, relative_l2={relative_l2}."
        )
    return maximum, relative_l2


def normalize_kv_quant_settings(head_dim: int) -> list[str]:
    """Validate and normalize the finalized KV storage settings."""
    global KV_QUANT_GROUP_SIZE
    supported = {
        "ROTARY_Q4", "ROTARY_Q4_CUDA", "Q8", "Q8_CUDA", "ROTARY_Q8",
        "ROTARY_Q8_CUDA", "F16", "F32",
    }
    if KV_QUANT_DTYPE not in supported:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")
    quantized = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
    rotary = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes: list[str] = []
    if KV_QUANT_DTYPE in rotary and head_dim % 2:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")
    if KV_QUANT_DTYPE in quantized:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError("KV_QUANT_GROUP_SIZE must be positive.")
        if KV_QUANT_GROUP_SIZE > head_dim:
            KV_QUANT_GROUP_SIZE = head_dim
            notes.append("KV group size exceeded head_dim and was clamped.")
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE:
            KV_QUANT_GROUP_SIZE = max(value for value in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % value == 0)
            notes.append("KV group size was reduced to a divisor of head_dim.")
    return notes


def _id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        return [int(item) for item in value]
    return [int(value)]


def _config_int(config, name: str, default=None) -> int:
    value = getattr(config, name, default)
    if value is None:
        raise ValueError(f"Model config is missing {name!r}.")
    return int(value)


def _token_ids(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


class LightOnTokenizerProcessor:
    """Fallback adapter for a checkpoint that only exposes AutoTokenizer."""

    def __init__(self, tokenizer, image_token_id: int):
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.image_token = tokenizer.convert_ids_to_tokens(self.image_token_id)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def _load_lighton_components():
    if not DOWNLOAD_PATH.is_dir():
        raise FileNotFoundError(f"LightOnOCR checkpoint directory does not exist: {DOWNLOAD_PATH}")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        str(DOWNLOAD_PATH), torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    try:
        processor = AutoProcessor.from_pretrained(
            str(DOWNLOAD_PATH), trust_remote_code=True, fix_mistral_regex=True
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(str(DOWNLOAD_PATH), trust_remote_code=True)
    if not (
        getattr(processor, "tokenizer", None) is not None
        and getattr(processor, "image_token", None)
        and getattr(processor, "image_token_id", None) is not None
    ):
        tokenizer = AutoTokenizer.from_pretrained(str(DOWNLOAD_PATH), trust_remote_code=True)
        processor = LightOnTokenizerProcessor(tokenizer, int(model.config.image_token_id))
    try:
        model.model.language_model
        model.model.vision_encoder
        model.model.vision_projection
        model.config.text_config
        model.config.vision_config
    except AttributeError as error:
        raise RuntimeError("Loaded checkpoint is not the expected LightOnOCR architecture.") from error
    return model, processor


def build_static_pixtral_tables(
    vision_encoder,
    image_resize: list[int],
    num_images: int,
    image_token_length: int,
    spatial_merge_size: int,
):
    """Build target-specific Pixtral 2D RoPE and independent-image attention blocks."""
    patch_size = int(vision_encoder.patch_size)
    grid_h, grid_w = image_resize[0] // patch_size, image_resize[1] // patch_size
    expected_patch_count = image_token_length * spatial_merge_size ** 2 * num_images
    if grid_h * grid_w * num_images != expected_patch_count:
        raise ValueError("IMAGE_TOKEN_LENGTH does not match the configured Pixtral patch grid after merging.")
    max_width = int(vision_encoder.config.image_size // patch_size)
    rows, columns = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    position_ids = (rows * max_width + columns).reshape(-1).long().repeat(num_images)
    frequencies = vision_encoder.patch_positional_embedding.inv_freq.index_select(0, position_ids).float()
    half = frequencies.shape[-1] // 2
    rotary_cos = frequencies.cos().unsqueeze(0)
    rotary_sin = torch.cat([-frequencies[:, :half].sin(), frequencies[:, half:].sin()], dim=-1).unsqueeze(0)
    total_seq = grid_h * grid_w * num_images
    attention_mask = torch.full((1, 1, total_seq, total_seq), -128, dtype=torch.int8)
    for image_index in range(num_images):
        start = image_index * grid_h * grid_w
        attention_mask[..., start:start + grid_h * grid_w, start:start + grid_h * grid_w] = 0
    return rotary_cos, rotary_sin, attention_mask


def build_lighton_prompt_layout(processor, num_images: int):
    """Validate the fixed image slot span emitted by the native chat template."""
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(processor, "image_token", None)
    image_token_id = getattr(processor, "image_token_id", None)
    if tokenizer is None or not image_token or image_token_id is None:
        raise RuntimeError("LightOnOCR processor must expose tokenizer, image_token, and image_token_id.")
    conversation = [{
        "role": "user",
        "content": [{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": ""}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    if prompt.count(image_token) != num_images:
        raise RuntimeError("LightOnOCR chat template emitted an unexpected number of image placeholders.")
    token_ids = _token_ids(tokenizer, prompt.replace(image_token, image_token * IMAGE_TOKEN_LENGTH))
    positions = [index for index, token_id in enumerate(token_ids) if token_id == int(image_token_id)]
    expected = IMAGE_TOKEN_LENGTH * num_images
    if len(positions) != expected or not positions:
        raise RuntimeError("Expanded image token count does not match the Pixtral projector output.")
    if positions[-1] - positions[0] + 1 != expected:
        raise RuntimeError("LightOnOCR image tokens must occupy one contiguous prompt span.")
    return token_ids, positions[0], positions[-1] + 1


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


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
        adjusted = torch.scatter(logits, 1, previous_ids.long(), adjusted)
        token = torch.argmax(adjusted, dim=-1, keepdim=True).int()
        return token, torch.cat([previous_ids, token], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    @staticmethod
    def _sample(scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
        probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = (cumulative - probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative >= threshold).int(), dim=-1, keepdim=True)
        return torch.gather(sorted_indices, 1, winner).int()

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        previous_logits = torch.gather(logits, 1, previous_ids.long())
        adjusted = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids.long(), adjusted)
        token = self._sample(scores, temperature, top_k, top_p)
        return token, torch.cat([previous_ids, token], dim=-1)


class LLM_EMBED(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Resize and normalize images while carrying fixed Pixtral vision tables."""

    def __init__(self, image_resize, rotary_cos, rotary_sin, attention_mask, dynamic_shape=False):
        super().__init__()
        self.target_h, self.target_w = (int(value) for value in image_resize)
        self.dynamic_shape = bool(dynamic_shape)
        self.register_buffer("rotary_cos", rotary_cos.float(), persistent=False)
        self.register_buffer("rotary_sin", rotary_sin.float(), persistent=False)
        self.register_buffer("attention_mask", attention_mask.float(), persistent=False)
        self.register_buffer("image_mean", torch.tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        pixels = pixel_values.float()
        if self.dynamic_shape or pixels.shape[-2] != self.target_h or pixels.shape[-1] != self.target_w:
            pixels = F.interpolate(pixels, size=[self.target_h, self.target_w], mode="bilinear", align_corners=False)
        pixels = (pixels / 255.0 - self.image_mean) / self.image_std
        anchor = pixels.reshape(-1)[0] * 0.0
        return (
            pixels,
            self.rotary_cos + anchor,
            self.rotary_sin + anchor,
            self.attention_mask + anchor,
        )


class LLM_VISION(torch.nn.Module):
    """Pixtral vision encoder plus LightOnOCR projector, with no text captured."""

    def __init__(self, llm, image_resize, num_images):
        super().__init__()
        self.vision_encoder = llm.model.vision_encoder
        self.layers = self.vision_encoder.transformer.layers
        projector = llm.model.vision_projection
        self.patch_conv = self.vision_encoder.patch_conv.float()
        self.ln_pre = self.vision_encoder.ln_pre.float()
        self.hidden_size = int(llm.config.vision_config.hidden_size)
        self.num_heads = int(llm.config.vision_config.num_attention_heads)
        self.head_dim = int(llm.config.vision_config.head_dim)
        self.head_dim_half = self.head_dim // 2
        self.patch_size = int(llm.config.vision_config.patch_size)
        self.spatial_merge_size = int(llm.config.spatial_merge_size)
        self.grid_h = int(image_resize[0] // self.patch_size)
        self.grid_w = int(image_resize[1] // self.patch_size)
        if self.grid_h % self.spatial_merge_size or self.grid_w % self.spatial_merge_size:
            raise ValueError("Pixtral image grid must divide spatial_merge_size.")
        self.grid_h_merged = self.grid_h // self.spatial_merge_size
        self.grid_w_merged = self.grid_w // self.spatial_merge_size
        self.num_images = int(num_images)
        self.merged_hidden_size = self.hidden_size * self.spatial_merge_size ** 2
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        attention_norm = self.layers[0].attention_norm
        eps = self.hidden_size * float(getattr(attention_norm, "variance_epsilon", getattr(attention_norm, "eps", 1e-6)))
        if PREVENT_F16_OVERFLOW:
            eps *= float(self.overflow_scale.square())
        self.register_buffer("rms_eps", torch.tensor([eps], dtype=torch.float32), persistent=False)
        projection_norm = projector.norm.float()
        projection_eps = self.hidden_size * float(
            getattr(projection_norm, "variance_epsilon", getattr(projection_norm, "eps", 1e-6))
        )
        if PREVENT_F16_OVERFLOW:
            projection_eps *= float(self.overflow_scale.square())
        self.register_buffer("projection_eps", torch.tensor([projection_eps], dtype=torch.float32), persistent=False)
        self.merging_layer = projector.patch_merger.merging_layer.float()
        self.proj_linear_1 = projector.linear_1.float()
        self.proj_linear_2 = projector.linear_2.float()
        self.proj_act = torch.nn.GELU()
        self._fuse_vision_weights(projection_norm)
        self.reordered_mlp_pairs = 0
        self.reorder_equivalence_error = 0.0
        if REORDER_VISION_MLP_FOR_QUANT:
            self._reorder_mlp_pairs(REORDER_KEY)

    def _fuse_vision_weights(self, projection_norm) -> None:
        scaling = self.head_dim ** -0.25
        for layer in self.layers:
            attention = layer.attention
            q_proj, k_proj, v_proj = attention.q_proj, attention.k_proj, attention.v_proj
            if any(projection.bias is not None for projection in (q_proj, k_proj, v_proj)):
                raise RuntimeError("Unexpected Pixtral attention bias layout.")
            qkv = torch.nn.Linear(q_proj.in_features, q_proj.out_features + k_proj.out_features + v_proj.out_features, bias=False)
            qkv.weight.data.copy_(torch.cat([
                q_proj.weight.data * scaling,
                k_proj.weight.data * scaling,
                v_proj.weight.data,
            ], dim=0))
            norm_weight = layer.attention_norm.weight.data * self.hidden_size ** 0.5
            qkv.weight.data.mul_(norm_weight.unsqueeze(0))
            attention.qkv = qkv
            attention._qk_out = q_proj.out_features + k_proj.out_features
            attention._v_out = v_proj.out_features
            del attention.q_proj, attention.k_proj, attention.v_proj

            feed_forward = layer.feed_forward
            gate, up = feed_forward.gate_proj, feed_forward.up_proj
            if gate.bias is not None or up.bias is not None:
                raise RuntimeError("Unexpected Pixtral MLP bias layout.")
            gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
            gate_up.weight.data.copy_(torch.cat([gate.weight.data, up.weight.data], dim=0))
            gate_up.weight.data.mul_((layer.ffn_norm.weight.data * self.hidden_size ** 0.5).unsqueeze(0))
            feed_forward.gate_up = gate_up
            feed_forward._gate_out = gate.out_features
            del feed_forward.gate_proj, feed_forward.up_proj

        norm_weight = projection_norm.weight.data * self.hidden_size ** 0.5
        if self.merging_layer.weight.shape[1] % norm_weight.numel():
            raise RuntimeError("Projector norm and patch-merger dimensions are incompatible.")
        spatial_merge_area = self.merging_layer.weight.shape[1] // norm_weight.numel()
        merged_norm_weight = norm_weight.unsqueeze(1).expand(-1, spatial_merge_area).reshape(-1)
        self.merging_layer.weight.data.mul_(merged_norm_weight.unsqueeze(0))
        first_weight, second_weight = self.merging_layer.weight.data, self.proj_linear_1.weight.data
        fused = torch.nn.Linear(first_weight.shape[1], second_weight.shape[0], bias=False)
        fused.weight.data.copy_(second_weight @ first_weight)
        self.merging_layer = fused
        del self.proj_linear_1

    @staticmethod
    def _assert_pair_shapes(first, second, gated: bool) -> int:
        intermediate = int(second.in_features)
        expected = intermediate * 2 if gated else intermediate
        if first.out_features != expected or first.weight.shape[0] != expected:
            raise RuntimeError("Unexpected producer/consumer MLP layout for channel reordering.")
        if second.weight.shape[1] != intermediate:
            raise RuntimeError("Consumer MLP input dimension disagrees with its weight layout.")
        return intermediate

    def _reorder_gated_pair(self, gate_up, down_proj, key: str) -> float:
        intermediate = self._assert_pair_shapes(gate_up, down_proj, gated=True)
        if getattr(gate_up, "_lighton_reorder_applied", False):
            raise RuntimeError("A Pixtral gated MLP permutation was applied twice.")
        probe = torch.linspace(-1.0, 1.0, steps=gate_up.in_features * 2, dtype=torch.float32).view(2, -1)
        before = down_proj(F.silu(F.linear(probe, gate_up.weight[:intermediate], None)) * F.linear(probe, gate_up.weight[intermediate:], None))
        permutation = _validated_permutation(down_proj.weight.data, key)
        gate_up.weight.data.copy_(torch.cat([
            gate_up.weight.data[:intermediate][permutation],
            gate_up.weight.data[intermediate:][permutation],
        ], dim=0))
        if gate_up.bias is not None:
            gate_up.bias.data.copy_(torch.cat([
                gate_up.bias.data[:intermediate][permutation],
                gate_up.bias.data[intermediate:][permutation],
            ], dim=0))
        down_proj.weight.data.copy_(down_proj.weight.data[:, permutation])
        after = down_proj(F.silu(F.linear(probe, gate_up.weight[:intermediate], None)) * F.linear(probe, gate_up.weight[intermediate:], None))
        error = _reorder_error(before, after, "Pixtral gated MLP")
        gate_up._lighton_reorder_applied = True
        return error

    def _reorder_linear_pair(self, first, second, key: str) -> float:
        intermediate = self._assert_pair_shapes(first, second, gated=False)
        if getattr(first, "_lighton_reorder_applied", False):
            raise RuntimeError("A Pixtral projector permutation was applied twice.")
        probe = torch.linspace(-1.0, 1.0, steps=first.in_features * 2, dtype=torch.float32).view(2, -1)
        before = second(self.proj_act(first(probe)))
        permutation = _validated_permutation(second.weight.data, key)
        first.weight.data.copy_(first.weight.data[permutation])
        if first.bias is not None:
            first.bias.data.copy_(first.bias.data[permutation])
        second.weight.data.copy_(second.weight.data[:, permutation])
        after = second(self.proj_act(first(probe)))
        error = _reorder_error(before, after, "Pixtral projector MLP")
        first._lighton_reorder_applied = True
        return error

    def _reorder_mlp_pairs(self, key: str) -> None:
        with torch.no_grad():
            errors = []
            for layer in self.layers:
                errors.append(self._reorder_gated_pair(layer.feed_forward.gate_up, layer.feed_forward.down_proj, key))
            errors.append(self._reorder_linear_pair(self.merging_layer, self.proj_linear_2, key))
        self.reordered_mlp_pairs = len(errors)
        self.reorder_equivalence_error = max((error[0] for error in errors), default=0.0)
        self.reorder_relative_equivalence_error = max((error[1] for error in errors), default=0.0)

    def _rms_norm(self, values, epsilon):
        if PREVENT_F16_OVERFLOW:
            values = values * self.overflow_scale
        return values * torch.rsqrt(values.square().sum(-1, keepdim=True) + epsilon)

    def _rotate_half(self, values):
        values = values.view(2, self.num_heads, -1, 2, self.head_dim_half)
        return values.flip(-2).view(2, self.num_heads, -1, self.head_dim)

    def forward(self, pixels, rotary_cos, rotary_sin, attention_mask):
        hidden_states = self.patch_conv(pixels.float())
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(1, -1, self.hidden_size)
        hidden_states = self.ln_pre(hidden_states)
        rotary_cos, rotary_sin, attention_mask = rotary_cos.float(), rotary_sin.float(), attention_mask.float()
        for layer in self.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps)
            qkv = layer.attention.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            qk, values = qkv.split([2, 1], dim=0)
            qk = qk * rotary_cos + self._rotate_half(qk) * rotary_sin
            query, key = qk.split(1, dim=0)
            attention = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) + attention_mask, dim=-1)
            attention = torch.matmul(attention, values).transpose(1, 2).reshape(1, -1, self.hidden_size)
            hidden_states = residual + layer.attention.o_proj(attention)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps)
            gate_up = layer.feed_forward.gate_up(hidden_states)
            gate, up = gate_up.split([layer.feed_forward._gate_out, layer.feed_forward._gate_out], dim=-1)
            hidden_states = residual + layer.feed_forward.down_proj(F.silu(gate) * up)
        hidden_states = self._rms_norm(hidden_states, self.projection_eps)
        merged = hidden_states.view(
            self.num_images, self.grid_h_merged, self.spatial_merge_size,
            self.grid_w_merged, self.spatial_merge_size, self.hidden_size,
        ).permute(0, 1, 3, 5, 2, 4).reshape(1, -1, self.merged_hidden_size)
        return self.proj_linear_2(self.proj_act(self.merging_layer(merged)))


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the template's contiguous image token span with Pixtral features."""

    def __init__(self, image_start: int, image_end: int):
        super().__init__()
        self.image_start = int(image_start)
        self.image_end = int(image_end)

    def forward(self, text_hidden_states, vision_hidden_states):
        return torch.cat([
            text_hidden_states[:, :self.image_start],
            vision_hidden_states,
            text_hidden_states[:, self.image_end:],
        ], dim=1)


class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Standard Qwen3 1D RoPE for multimodal text prefill."""

    def __init__(self, llm, max_seq_len: int):
        super().__init__()
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.language_model.rotary_emb.inv_freq.float()
        angles = (positions * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        self.register_buffer("rotary_cos", torch.cat([angles.cos(), angles.cos()], dim=-1).half(), persistent=False)
        self.register_buffer("rotary_sin", torch.cat([-angles.sin(), angles.sin()], dim=-1).half(), persistent=False)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        return (
            self.rotary_cos[:, history_len:kv_seq_len].float(),
            self.rotary_sin[:, history_len:kv_seq_len].float(),
            self.attention_mask[..., :ids_len, :kv_seq_len].float(),
            kv_seq_len,
        )


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    def __init__(self, llm, max_seq_len: int):
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.language_model.rotary_emb.inv_freq.float()
        angles = (positions * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        self.register_buffer("rotary_cos", torch.cat([angles.cos(), angles.cos()], dim=-1).half(), persistent=False)
        self.register_buffer("rotary_sin", torch.cat([-angles.sin(), angles.sin()], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        return self.rotary_cos[:, kv_seq_len].float(), self.rotary_sin[:, kv_seq_len].float(), kv_seq_len + 1


class LLM_MAIN(_REFERENCE.LLM_MAIN):
    """Finalized Qwen3 kernel with LightOnOCR's verified gated-MLP permutation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reordered_mlp_pairs = getattr(self, "reordered_mlp_pairs", 0)
        self.reorder_equivalence_error = getattr(self, "reorder_equivalence_error", 0.0)
        self.reorder_relative_equivalence_error = getattr(self, "reorder_relative_equivalence_error", 0.0)

    def _reorder_downproj_for_quant(self, key: str) -> None:
        _require_reorder_key(key)
        errors = []
        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                gate_up = layer.mlp.gate_up_proj
                down_proj = layer.mlp.down_proj
                if getattr(gate_up, "_lighton_reorder_applied", False):
                    raise RuntimeError("A Qwen3 gated MLP permutation was applied twice.")
                intermediate = int(down_proj.in_features)
                if gate_up.out_features != intermediate * 2 or gate_up.weight.shape[0] != intermediate * 2:
                    raise RuntimeError("Unexpected Qwen3 fused gate/up layout.")
                if down_proj.weight.shape[1] != intermediate:
                    raise RuntimeError("Qwen3 down projection has an incompatible input width.")
                probe = torch.linspace(-1.0, 1.0, steps=gate_up.in_features * 2, dtype=torch.float32).view(2, -1)
                before_gate_up = gate_up(probe)
                before = down_proj(layer.mlp.act_fn(before_gate_up[:, :intermediate]) * before_gate_up[:, intermediate:])
                permutation = _validated_permutation(down_proj.weight.data, key)
                gate_up.weight.data.copy_(torch.cat([
                    gate_up.weight.data[:intermediate][permutation],
                    gate_up.weight.data[intermediate:][permutation],
                ], dim=0))
                if gate_up.bias is not None:
                    gate_up.bias.data.copy_(torch.cat([
                        gate_up.bias.data[:intermediate][permutation],
                        gate_up.bias.data[intermediate:][permutation],
                    ], dim=0))
                down_proj.weight.data.copy_(down_proj.weight.data[:, permutation])
                after_gate_up = gate_up(probe)
                after = down_proj(layer.mlp.act_fn(after_gate_up[:, :intermediate]) * after_gate_up[:, intermediate:])
                error = _reorder_error(before, after, "Qwen3 gated MLP")
                gate_up._lighton_reorder_applied = True
                errors.append(error)
        self.reordered_mlp_pairs = len(errors)
        self.reorder_equivalence_error = max((error[0] for error in errors), default=0.0)
        self.reorder_relative_equivalence_error = max((error[1] for error in errors), default=0.0)


def _build_kv_layout(batch_size: int, num_layers: int, num_kv_heads: int, head_dim: int, history_len: int):
    rotary_modes = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_modes = {"Q8", "Q8_CUDA"}
    rotary_q4 = KV_QUANT_DTYPE in {"ROTARY_Q4", "ROTARY_Q4_CUDA"}
    grouped_q8 = KV_QUANT_DTYPE in q8_modes | {"ROTARY_Q8", "ROTARY_Q8_CUDA"} and (
        USE_HADAMARD or USE_SHUFFLE
    ) and KV_QUANT_GROUP_SIZE < head_dim
    quantized = KV_QUANT_DTYPE in rotary_modes | q8_modes
    symmetric = USE_SYM and quantized
    grouped_6d = rotary_q4 or grouped_q8
    specs = [("key", 4), ("value", 3)]
    if quantized:
        key_scale_axis = 5 if grouped_6d else 4
        specs.append(("key_scale", key_scale_axis))
        if not symmetric:
            specs.append(("key_bias", key_scale_axis))
        specs.append(("value_scale", 3))
        if not symmetric:
            specs.append(("value_bias", 3))
    if KV_QUANT_DTYPE == "F16":
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"}:
        kv_dtype = torch.int32
    elif symmetric and not rotary_q4:
        kv_dtype = torch.int8
    elif quantized:
        kv_dtype = torch.uint8
    else:
        kv_dtype = torch.float32
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"}:
        key_width = value_width = head_dim // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4":
        key_width = value_width = head_dim // 2
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        key_width = value_width = head_dim // 8
    else:
        key_width = value_width = head_dim
    tensors = {
        "key": torch.zeros((batch_size, num_kv_heads, 1, key_width, history_len), dtype=kv_dtype),
        "value": torch.zeros((batch_size, num_kv_heads, 1, history_len, value_width), dtype=kv_dtype),
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
        tensors["key_scale"] = torch.ones(key_scale_shape, dtype=scale_dtype)
        tensors["value_scale"] = torch.ones(value_scale_shape, dtype=scale_dtype)
        if not symmetric:
            tensors["key_bias"] = torch.ones(key_scale_shape, dtype=scale_dtype)
            tensors["value_bias"] = torch.ones(value_scale_shape, dtype=scale_dtype)
    facts = {
        "kv_cache_quantization": KV_QUANT_DTYPE,
        "kv_cache_tensor_order": ",".join(name for name, _ in specs),
        "kv_cache_key_layout": "batch,key_value_heads,one,key_width,sequence",
        "kv_cache_value_layout": "batch,key_value_heads,one,sequence,value_width",
        "kv_cache_key_sequence_axis": "4",
        "kv_cache_value_sequence_axis": "3",
        "kv_cache_key_storage_width": str(key_width),
        "kv_cache_value_storage_width": str(value_width),
        "kv_cache_quantized": str(int(quantized)),
        "kv_cache_symmetric": str(int(symmetric)),
        "kv_cache_grouped_6d": str(int(grouped_6d)),
        "kv_cache_group_size": str(KV_QUANT_GROUP_SIZE if quantized else 0),
        "kv_cache_group_count": str(group_count if quantized else 0),
        "kv_cache_storage_dtype": str(kv_dtype).replace("torch.", ""),
        "kv_cache_scale_bias_dtype": str(scale_dtype).replace("torch.", "") if quantized else "none",
    }
    return specs, tensors, facts


def _kv_io(kv_specs, kv_tensors, num_layers: int):
    inputs, input_names, output_names, dynamic_axes = [], [], [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            input_name, output_name = f"in_{name}_{layer_index}", f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
            dynamic_axes[output_name] = {0: "batch_size", sequence_axis: "kv_seq_len"}
    return inputs, input_names, output_names, dynamic_axes


def _sequence_axes(source_axes, sequence_name: str):
    return {
        axis: ("batch_size" if axis == 0 else sequence_name)
        for axis in source_axes
    }


def _export_kv_helpers(
    export_dir: Path,
    dimensions,
    kv_specs,
    kv_tensors,
    metadata,
    rope_inv_freq: torch.Tensor,
    quantizer,
) -> None:
    """Export Qwen-compatible cache utilities for the selected KV representation."""
    num_layers = dimensions["num_layers"]
    head_dim = dimensions["head_dim"]
    num_kv_heads = dimensions["num_kv_heads"]

    inputs, input_names, output_names, axes = _kv_io(kv_specs, kv_tensors, num_layers)
    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([1], dtype=torch.int64)
    slice_axes = {name: dict(axes[name]) for name in input_names}
    for name in output_names:
        slice_axes[name] = _sequence_axes(axes[name], "sliced_len")
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_slice"],
        _REFERENCE.KV_SLICE(num_layers, head_dim),
        tuple(inputs + [slice_start, slice_end]),
        input_names + ["slice_start", "slice_end"],
        output_names,
        slice_axes,
        metadata,
    )

    split_at = torch.tensor([1], dtype=torch.int64)
    prefix_names = [f"prefix_{name}" for name in output_names]
    window_names = [f"window_{name}" for name in output_names]
    split_axes = {name: dict(axes[name]) for name in input_names}
    for source_name, prefix_name, window_name in zip(output_names, prefix_names, window_names):
        split_axes[prefix_name] = _sequence_axes(axes[source_name], "prefix_len")
        split_axes[window_name] = _sequence_axes(axes[source_name], "window_len")
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_split2"],
        _REFERENCE.KV_SPLIT2(num_layers, head_dim),
        tuple(inputs + [split_at]),
        input_names + ["split_at"],
        prefix_names + window_names,
        split_axes,
        metadata,
    )

    prefix_inputs, suffix_inputs = [], []
    prefix_names, suffix_names, concat_names, concat_axes = [], [], [], {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            prefix_name = f"in_prefix_{name}_{layer_index}"
            suffix_name = f"in_suffix_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            prefix_inputs.append(tensor)
            suffix_inputs.append(tensor.clone())
            prefix_names.append(prefix_name)
            suffix_names.append(suffix_name)
            concat_names.append(output_name)
            concat_axes[prefix_name] = {0: "batch_size", sequence_axis: "prefix_len"}
            concat_axes[suffix_name] = {0: "batch_size", sequence_axis: "suffix_len"}
            concat_axes[output_name] = {0: "batch_size", sequence_axis: "concat_len"}
    _export_component(
        export_dir / MODEL_FILE_NAMES["kv_concat"],
        _REFERENCE.KV_CONCAT(num_layers, head_dim),
        tuple(prefix_inputs + suffix_inputs),
        prefix_names + suffix_names,
        concat_names,
        concat_axes,
        metadata,
    )

    _, rope_tensors, _ = _build_kv_layout(1, num_layers, num_kv_heads, head_dim, 4)
    rope_shift = torch.tensor([1], dtype=torch.int64)
    if KV_QUANT_DTYPE in {"F16", "F32"}:
        rope_inputs = [rope_tensors["key"].clone() for _ in range(num_layers)]
        rope_input_names = [f"in_key_{layer_index}" for layer_index in range(num_layers)]
        rope_output_names = [f"out_key_{layer_index}" for layer_index in range(num_layers)]
        rope_axes = {
            name: {0: "batch_size", 4: "history_len"}
            for name in rope_input_names + rope_output_names
        }
        rope_module = _REFERENCE.ROPE_SHIFT(
            num_layers,
            head_dim,
            num_kv_heads,
            rope_inv_freq,
            MAX_SEQ_LEN,
        )
    else:
        sequence_axes = dict(kv_specs)
        rope_names = ["key", "key_scale"]
        if not USE_SYM:
            rope_names.append("key_bias")
        rope_inputs, rope_input_names, rope_output_names, rope_axes = [], [], [], {}
        for name in rope_names:
            sequence_axis = sequence_axes[name]
            for layer_index in range(num_layers):
                input_name = f"in_{name}_{layer_index}"
                output_name = f"out_{name}_{layer_index}"
                rope_inputs.append(rope_tensors[name].clone())
                rope_input_names.append(input_name)
                rope_output_names.append(output_name)
                rope_axes[input_name] = {0: "batch_size", sequence_axis: "history_len"}
                rope_axes[output_name] = {0: "batch_size", sequence_axis: "history_len"}
        rope_module = _REFERENCE.ROPE_SHIFT_QUANT(
            num_layers,
            head_dim,
            num_kv_heads,
            rope_inv_freq,
            MAX_SEQ_LEN,
            quantizer,
            not USE_SYM,
        )
    _export_component(
        export_dir / MODEL_FILE_NAMES["rope_shift"],
        rope_module,
        tuple(rope_inputs + [rope_shift]),
        rope_input_names + ["shift"],
        rope_output_names,
        rope_axes,
        metadata,
    )


def _metadata_values(model, processor, dimensions, kv_facts, image_start: int, image_end: int):
    expected_fused_norms = dimensions["num_layers"] * 3 + 1
    metadata = {
        "model_family": "LightOnOCR-Qwen3-Pixtral",
        "max_seq_len": str(MAX_SEQ_LEN),
        "input_image_size": ",".join(str(value) for value in INPUT_IMAGE_SIZE),
        "input_image_dim": str(INPUT_IMAGE_DIM),
        "vision_batch_size": str(VISION_BATCH_SIZE),
        "image_token_id": str(int(processor.image_token_id)),
        "image_token_length": str(IMAGE_TOKEN_LENGTH),
        "image_start": str(image_start),
        "image_end": str(image_end),
        "stop_token_ids": ",".join(str(value) for value in STOP_TOKEN),
        "eos_token_ids": ",".join(str(value) for value in _id_list(getattr(model.config, "eos_token_id", None))),
        "kv_num_tensors": str(dimensions["num_layers"] * len(kv_facts["kv_cache_tensor_order"].split(","))),
        "kv_quant_dtype": KV_QUANT_DTYPE,
        "kv_quant_group_size": str(KV_QUANT_GROUP_SIZE),
        "compute_in_f32": str(int(COMPUTE_IN_F32)),
        "text_rope_type": "qwen3_1d",
        "vision_rope_type": "pixtral_2d",
        "deepstack_features": "0",
        "reorder_downproj": str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        "vision_reorder_mlp": str(int(REORDER_VISION_MLP_FOR_QUANT)),
        "reorder_key": REORDER_KEY,
        "language_reorder_pair_count": str(dimensions["num_layers"] if REORDER_DOWNPROJ_FOR_QUANT else 0),
        "vision_reorder_pair_count": str(dimensions["vision_layers"] + 1 if REORDER_VISION_MLP_FOR_QUANT else 0),
        "fused_simplified_layer_norm_count": str(expected_fused_norms),
    }
    metadata.update(kv_facts)
    metadata.update(MODEL_FILE_NAME_METADATA)
    return metadata


def _stamp_metadata(path: Path, metadata: dict) -> None:
    model = onnx.load(str(path), load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    values.update({key: str(value) for key, value in metadata.items()})
    model.ClearField("metadata_props")
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save_model(model, str(path))


def _export_component(path: Path, module, args, input_names, output_names, dynamic_axes, metadata) -> None:
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


def _prepare_export_staging() -> Path:
    if EXPORT_STAGING_DIR.exists():
        if not EXPORT_STAGING_DIR.is_dir():
            raise NotADirectoryError(EXPORT_STAGING_DIR)
        shutil.rmtree(EXPORT_STAGING_DIR)
    EXPORT_STAGING_DIR.mkdir(parents=True)
    return EXPORT_STAGING_DIR


def _promote_export(staging_dir: Path) -> None:
    """Atomically rotate the prior valid bundle and publish the validated staging bundle."""
    destination = EXPORT_DIR
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink()
    if destination.exists():
        destination.rename(previous)
    try:
        staging_dir.rename(destination)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise


def _cleanup_unreferenced_data(export_dir: Path) -> None:
    referenced = set()
    for model_path in export_dir.glob("*.onnx"):
        model = onnx.load(str(model_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                values = {item.key: item.value for item in initializer.external_data}
                if values.get("location"):
                    referenced.add(Path(values["location"]).name)
    for path in export_dir.iterdir():
        if path.is_file() and path.suffix != ".onnx" and path.name not in referenced:
            path.unlink()


def _count_fused_norms(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    return sum(node.op_type == "SimplifiedLayerNormalization" for node in model.graph.node)


def _validate_bundle(export_dir: Path, metadata: dict, expected_norms: int) -> None:
    import Shared_Merged

    required = {MODEL_FILE_NAMES["metadata"]}
    required.update(MODEL_FILE_NAMES[role] for role in RUNTIME_MODEL_FILE_ROLES)
    missing = sorted(name for name in required if not (export_dir / name).exists())
    if missing:
        raise RuntimeError(f"Incomplete LightOnOCR runtime bundle: {missing}")
    merged = [
        MODEL_FILE_NAMES[f"image_{phase}_{strategy}"]
        for phase in ("prefill", "decode")
        for strategy in ("greedy", "penalty_greedy", "sampling")
    ]
    if len(merged) != 6 or any(not (export_dir / name).exists() for name in merged):
        raise RuntimeError("Expected exactly six merged image strategy graphs.")
    for role in ("kv_slice", "kv_split2", "kv_concat", "rope_shift"):
        Shared_Merged.validate_onnx_path(export_dir / MODEL_FILE_NAMES[role])
    state_count = int(metadata["kv_num_tensors"])
    for name in merged:
        path = export_dir / name
        Shared_Merged.validate_onnx_path(path)
        model = onnx.load(str(path), load_external_data=False)
        if any(item.name.startswith("in_") is False for item in model.graph.input[:state_count]):
            raise RuntimeError(f"{name} does not lead with state inputs.")
        if any(item.name.startswith("out_") is False for item in model.graph.output[:state_count]):
            raise RuntimeError(f"{name} does not lead with state outputs.")
        tail = len(model.graph.output) - state_count
        expected_tail = 2 if "Greedy.onnx" in name and "Penalty" not in name else 3
        if tail != expected_tail:
            raise RuntimeError(f"{name} has {tail} tail outputs, expected {expected_tail}.")
        if _count_fused_norms(path) != expected_norms:
            raise RuntimeError(f"{name} has an unexpected SimplifiedLayerNormalization count.")
    shared = onnx.load(str(export_dir / MODEL_FILE_NAMES["shared_initializers"]), load_external_data=False)
    shared_data_size = (export_dir / MODEL_FILE_NAMES["shared_initializers_data"]).stat().st_size
    for initializer in shared.graph.initializer:
        values = {item.key: item.value for item in initializer.external_data}
        if values.get("location") != MODEL_FILE_NAMES["shared_initializers_data"]:
            raise RuntimeError("Shared initializer points outside the bundle data contract.")
        offset, length = int(values.get("offset", "0")), int(values.get("length", "0"))
        if offset < 0 or length < 0 or offset + length > shared_data_size:
            raise RuntimeError("Shared initializer external-data offset is invalid.")
    forbidden = ("Beam", "Argmax", "Apply_Penalty", "Relation")
    present = [path.name for path in export_dir.iterdir() if any(token in path.name for token in forbidden)]
    if present:
        raise RuntimeError(f"Legacy strategy artifacts survived export: {present}")


@torch.inference_mode()
def export_lighton() -> Path:
    if INPUT_IMAGE_DIM not in {4, 5}:
        raise ValueError("INPUT_IMAGE_DIM must be 4 or 5.")
    if IMAGE_TOKEN_LENGTH != HEIGHT_FACTOR * WIDTH_FACTOR:
        raise ValueError("IMAGE_TOKEN_LENGTH must match the merged Pixtral grid.")
    _require_reorder_key(REORDER_KEY)
    _configure_reference_kv()
    staging_dir = _prepare_export_staging()
    try:
        model, processor = _load_lighton_components()
        text_config, vision_config = model.config.text_config, model.config.vision_config
        dimensions = {
            "num_layers": _config_int(text_config, "num_hidden_layers"),
            "num_heads": _config_int(text_config, "num_attention_heads"),
            "num_kv_heads": _config_int(text_config, "num_key_value_heads"),
            "hidden_size": _config_int(text_config, "hidden_size"),
            "vocab_size": _config_int(text_config, "vocab_size"),
            "vision_layers": _config_int(vision_config, "num_hidden_layers"),
        }
        dimensions["head_dim"] = _config_int(text_config, "head_dim", dimensions["hidden_size"] // dimensions["num_heads"])
        if dimensions["num_heads"] % dimensions["num_kv_heads"]:
            raise ValueError("Qwen3 attention heads must divide KV heads.")
        if model.model.language_model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
            raise RuntimeError("LightOnOCR checkpoint no longer ties embed_tokens and lm_head weights.")
        rope_inv_freq = model.model.language_model.rotary_emb.inv_freq.detach().float().clone()
        for note in normalize_kv_quant_settings(dimensions["head_dim"]):
            print(note)
        _configure_reference_kv()
        token_ids, image_start, image_end = build_lighton_prompt_layout(processor, VISION_BATCH_SIZE)
        vision_cos, vision_sin, vision_mask = build_static_pixtral_tables(
            model.model.vision_encoder,
            IMAGE_RESIZE,
            VISION_BATCH_SIZE,
            IMAGE_TOKEN_LENGTH,
            int(model.config.spatial_merge_size),
        )
        image_feature_count = (IMAGE_RESIZE[0] // (vision_config.patch_size * model.config.spatial_merge_size)) * (
            IMAGE_RESIZE[1] // (vision_config.patch_size * model.config.spatial_merge_size)
        ) * VISION_BATCH_SIZE
        if image_feature_count != IMAGE_TOKEN_LENGTH * VISION_BATCH_SIZE:
            raise RuntimeError("Pixtral projector output count does not equal the expanded image token count.")
        kv_specs, kv_tensors, kv_facts = _build_kv_layout(
            1, dimensions["num_layers"], dimensions["num_kv_heads"], dimensions["head_dim"], 0
        )
        metadata = _metadata_values(model, processor, dimensions, kv_facts, image_start, image_end)
        _export_component(
            staging_dir / MODEL_FILE_NAMES["metadata"], METADATA_CARRIER(),
            (torch.zeros((1,), dtype=torch.int32),), ["metadata_marker"], ["metadata_marker_out"], None, metadata,
        )
        trace_ids_len = max(1, min(10, len(token_ids)))
        ids_len, history_len = torch.tensor([trace_ids_len], dtype=torch.int64), torch.zeros((1,), dtype=torch.int64)
        kv_seq_len = ids_len + history_len
        _export_component(
            staging_dir / MODEL_FILE_NAMES["embed"], LLM_EMBED(model),
            (torch.tensor([token_ids[:trace_ids_len]], dtype=torch.int64),),
            ["input_ids"], ["text_hidden_states"],
            {"input_ids": {0: "batch_size", 1: "ids_len"}, "text_hidden_states": {0: "batch_size", 1: "ids_len"}},
            metadata,
        )
        if INPUT_IMAGE_DIM == 5:
            image_input = torch.zeros((VISION_BATCH_SIZE, 1, 3, *INPUT_IMAGE_SIZE), dtype=torch.uint8)
        else:
            image_input = torch.zeros((VISION_BATCH_SIZE, 3, *INPUT_IMAGE_SIZE), dtype=torch.uint8)
        _export_component(
            staging_dir / MODEL_FILE_NAMES["image_preprocess"],
            LLM_IMAGE_PREPROCESS(IMAGE_RESIZE, vision_cos, vision_sin, vision_mask, DYNAMIC_IMAGE_SHAPE),
            (image_input,), ["pixel_values"], ["pixels", "vision_cos", "vision_sin", "vision_mask"],
            {
                "pixels": {0: "image_count"}, "vision_cos": {1: "vision_patch_count"},
                "vision_sin": {1: "vision_patch_count"}, "vision_mask": {2: "vision_patch_count", 3: "vision_patch_count"},
            }, metadata,
        )
        del image_input
        vision = LLM_VISION(model, IMAGE_RESIZE, VISION_BATCH_SIZE)
        if vision.reordered_mlp_pairs != dimensions["vision_layers"] + 1 if REORDER_VISION_MLP_FOR_QUANT else vision.reordered_mlp_pairs != 0:
            raise RuntimeError("Unexpected count of reordered Pixtral MLP pairs.")
        pixels = torch.zeros((VISION_BATCH_SIZE, 3, *IMAGE_RESIZE), dtype=torch.float32)
        _export_component(
            staging_dir / MODEL_FILE_NAMES["vision"], vision,
            (pixels, vision_cos, vision_sin, vision_mask),
            ["pixels", "vision_cos", "vision_sin", "vision_mask"], ["vision_hidden_states"],
            {
                "pixels": {0: "image_count"}, "vision_cos": {1: "vision_patch_count"},
                "vision_sin": {1: "vision_patch_count"}, "vision_mask": {2: "vision_patch_count", 3: "vision_patch_count"},
                "vision_hidden_states": {0: "batch_size", 1: "image_token_count"},
            }, metadata,
        )
        metadata["vision_reorder_equivalence_error"] = repr(vision.reorder_equivalence_error)
        metadata["vision_reorder_relative_l2_error"] = repr(vision.reorder_relative_equivalence_error)
        del vision, pixels, vision_cos, vision_sin, vision_mask
        gc.collect()
        _export_component(
            staging_dir / MODEL_FILE_NAMES["concat_image"], LLM_CONCAT_IMAGE(image_start, image_end),
            (
                torch.ones((1, len(token_ids), dimensions["hidden_size"]), dtype=torch.float32),
                torch.ones((1, image_feature_count, dimensions["hidden_size"]), dtype=torch.float32),
            ), ["text_hidden_states", "vision_hidden_states"], ["concat_hidden_states"],
            {
                "text_hidden_states": {0: "batch_size", 1: "ids_len"},
                "vision_hidden_states": {0: "batch_size", 1: "image_token_count"},
                "concat_hidden_states": {0: "batch_size", 1: "ids_len"},
            }, metadata,
        )
        _export_component(
            staging_dir / MODEL_FILE_NAMES["rotary_image_prefill"], ROTARY_IMAGE_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len), ["ids_len", "history_len"], ["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
            {
                "rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            }, metadata,
        )
        _export_component(
            staging_dir / MODEL_FILE_NAMES["rotary_image_decode"], ROTARY_IMAGE_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,), ["kv_seq_len"], ["rotary_cos", "rotary_sin", "kv_seq_len_next"], None, metadata,
        )
        kv_inputs, kv_input_names, kv_output_names, kv_axes = _kv_io(kv_specs, kv_tensors, dimensions["num_layers"])
        main = LLM_MAIN(
            model, dimensions["num_heads"], dimensions["num_kv_heads"], dimensions["head_dim"],
            dimensions["num_layers"], dimensions["hidden_size"], 0,
        )
        expected_language_pairs = dimensions["num_layers"] if REORDER_DOWNPROJ_FOR_QUANT else 0
        if main.reordered_mlp_pairs != expected_language_pairs:
            raise RuntimeError("Unexpected count of reordered Qwen3 MLP pairs.")
        metadata["language_reorder_equivalence_error"] = repr(main.reorder_equivalence_error)
        metadata["language_reorder_relative_l2_error"] = repr(main.reorder_relative_equivalence_error)
        hidden_states = torch.ones((1, trace_ids_len, dimensions["hidden_size"]), dtype=torch.float32)
        rotary_cos = torch.zeros((1, trace_ids_len, 1, 1, dimensions["head_dim"]), dtype=torch.float32)
        rotary_sin, attention_mask = torch.zeros_like(rotary_cos), torch.zeros((1, 1, 1, trace_ids_len, trace_ids_len), dtype=torch.float32)
        _export_component(
            staging_dir / MODEL_FILE_NAMES["main"], main,
            tuple(kv_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]),
            kv_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"], kv_output_names + ["logits"],
            {
                **kv_axes,
                "hidden_states": {0: "batch_size", 1: "ids_len"}, "logits": {0: "batch_size"},
                "rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            }, metadata,
        )
        expected_norms = int(metadata["fused_simplified_layer_norm_count"])
        if _count_fused_norms(staging_dir / MODEL_FILE_NAMES["main"]) != expected_norms:
            raise RuntimeError("LLM_Main did not preserve the expected fused RMSNorm nodes.")
        _export_kv_helpers(
            staging_dir,
            dimensions,
            kv_specs,
            kv_tensors,
            metadata,
            rope_inv_freq,
            main.quantizer,
        )
        del main, kv_inputs, hidden_states, rotary_cos, rotary_sin, attention_mask, rope_inv_freq, kv_tensors
        gc.collect()
        logits = torch.ones((1, dimensions["vocab_size"]), dtype=torch.float32)
        previous_ids = torch.zeros((1, 1), dtype=torch.int32)
        repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
        _export_component(
            staging_dir / MODEL_FILE_NAMES["greedy"], GREEDY_SEARCH(), (logits,), ["logits"], ["max_logits_idx"],
            {"logits": {0: "batch_size"}, "max_logits_idx": {0: "batch_size"}}, metadata,
        )
        _export_component(
            staging_dir / MODEL_FILE_NAMES["penalty_greedy"], PENALTY_GREEDY_SEARCH(),
            (logits, repetition_penalty, previous_ids), ["logits", "repetition_penalty", "previous_ids"], ["max_logits_idx", "save_id_out"],
            {
                "logits": {0: "batch_size"}, "repetition_penalty": {0: "batch_size"},
                "previous_ids": {0: "batch_size", 1: "history_len"}, "max_logits_idx": {0: "batch_size"},
                "save_id_out": {0: "batch_size", 1: "kv_seq_len"},
            }, metadata,
        )
        _export_component(
            staging_dir / MODEL_FILE_NAMES["sampling"], TOPK_TOPP_SAMPLING(),
            (logits, torch.ones((1,), dtype=torch.float32), torch.tensor(min(50, dimensions["vocab_size"]), dtype=torch.int64), torch.ones((1,), dtype=torch.float32), repetition_penalty, previous_ids),
            ["logits", "temperature", "top_k", "top_p", "repetition_penalty", "previous_ids"], ["sampled_id", "save_id_out"],
            {
                "logits": {0: "batch_size"}, "temperature": {0: "batch_size"}, "top_p": {0: "batch_size"},
                "repetition_penalty": {0: "batch_size"}, "previous_ids": {0: "batch_size", 1: "history_len"},
                "sampled_id": {0: "batch_size"}, "save_id_out": {0: "batch_size", 1: "kv_seq_len"},
            }, metadata,
        )
        del logits, previous_ids, repetition_penalty, model
        gc.collect()
        for constituent_path in staging_dir.glob("*.onnx"):
            _stamp_metadata(constituent_path, metadata)
        import Shared_Merged

        bundle = Shared_Merged.build_shared_merged_bundle(
            staging_dir, model_file_names=MODEL_FILE_NAMES, delete_constituents=True
        )
        for path in [staging_dir / MODEL_FILE_NAMES["metadata"], staging_dir / MODEL_FILE_NAMES["image_preprocess"], staging_dir / MODEL_FILE_NAMES["vision"], bundle["shared_model"], *bundle["graphs"].values()]:
            _stamp_metadata(path, metadata)
        _validate_bundle(staging_dir, metadata, expected_norms)
        _cleanup_unreferenced_data(staging_dir)
        tokenizer_assets = copy_tokenizer_assets(DOWNLOAD_PATH, staging_dir)
        _promote_export(staging_dir)
        print(
            f"LightOnOCR ONNX export completed: {EXPORT_DIR} "
            f"({len(tokenizer_assets)} tokenizer assets)."
        )
        return EXPORT_DIR
    except BaseException:
        # Retain the staging directory for diagnosis; a validated previous bundle stays untouched.
        raise


def export_bundle() -> Path:
    """Export the LightOnOCR ONNX bundle."""
    return export_lighton()


def main() -> None:
    if not DO_EXPORT:
        print("DO_EXPORT is False; no ONNX files were written.")
        return
    export_dir = export_bundle()
    inference_script = SCRIPT_DIR / "Inference_LightOnOCR_ONNX.py"
    subprocess.run(
        [sys.executable, str(inference_script), "--model-folder", str(export_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()