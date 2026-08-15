import gc
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCR_Tokenizer_Assets import copy_tokenizer_assets

EXPORT_DIR = os.path.join(SCRIPT_DIR, 'FireRedOCR_ONNX')
EXPORT_STAGING_DIR = EXPORT_DIR + '.staging'

CHECKPOINT_DIR                 = str(Path.home() / "Downloads" / "FireRed-OCR")
# Backward-compatible configuration alias.
download_path                  = CHECKPOINT_DIR

# Export controls
DO_EXPORT            = True                    # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW = False                   # Prevent float16 overflow for Q4F16, Q8F16, or F16 quantization.
STOP_TOKEN           = [151643, 151645]        # FireRedOCR stop token ids.
MAX_SEQ_LEN          = 4096                    # Fixed maximum context length after export.

# Quantization-oriented model reordering
# Exact channel permutations keep paired producer/consumer weights synchronized.
REORDER_DOWNPROJ_FOR_QUANT   = True            # Reorder language MLP channels before down-projection quantization.
REORDER_VISION_MLP_FOR_QUANT = True            # Reorder vision MLP channels before quantization.
REORDER_KEY                  = "absmean"       # Channel statistic: absmean | L4 | rms | std.

# Image input and vision tracing
HEIGHT_FACTOR       = 25                       # Vertical factor for the exported image grid.
WIDTH_FACTOR        = 25                       # Horizontal factor for the exported image grid.
# Image resize uses patch_size * spatial_merge_size.
IMAGE_RESIZE        = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]
INPUT_IMAGE_SIZE    = [960, 960]               # Input image shape before ONNX preprocessing.
VISION_BATCH_SIZE   = 1                        # Number of images supported by the prompt.
DYNAMIC_IMAGE_SHAPE = False                    # Allow VISION_BATCH_SIZE images at runtime.
INPUT_IMAGE_DIM     = 5                        # 4=[B, C, H, W]; 5=[B, 1, C, H, W].
CLIP_IMAGE_MEAN     = [0.5, 0.5, 0.5]          # Image normalization mean.
CLIP_IMAGE_STD      = [0.5, 0.5, 0.5]          # Image normalization standard deviation.
IMAGE_TOKEN_LENGTH  = HEIGHT_FACTOR * WIDTH_FACTOR
TEMPORAL_PATCH_SIZE = 2
VISION_PATCH_LENGTH = (HEIGHT_FACTOR * 2) * (WIDTH_FACTOR * 2)

# KV cache storage and attention precision
KV_QUANT_DTYPE      = "Q8"                     # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE = 128                      # Quantization group width; must divide head_dim evenly.
COMPUTE_IN_F32      = False                    # F16 KV only: False=f16 attention, True=upcast KV for f32 math.

# KV quantization transforms and parameters
USE_HADAMARD           = False                 # Apply randomized Walsh-Hadamard mixing before quantization.
HADAMARD_RANDOM_SEED   = 9527                  # Seed for the deterministic Hadamard sign pattern.
USE_CLIP               = False                 # Clip outliers to mean +/- CLIP_SIGMA*std before quantization.
CLIP_SIGMA             = 3.0                   # Standard-deviation bound used when clipping is enabled.
USE_SHUFFLE            = False                 # Interleave channels across quantization groups.
USE_SYM                = True                  # True=symmetric absmax; False=asymmetric min-max with bias.
USE_FLOAT16_SCALE_BIAS = True                  # Store quantization scales and biases as float16.

# ONNX graph format
OPSET = 20                                     # ONNX opset version.

# Every runtime-visible filename is part of the metadata contract. Keep this
# image-only: FireRedOCR has no text-only or video-only graph recipes.
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
MODEL_FILE_NAMES['shared_initializers_data'] = MODEL_FILE_NAMES['shared_initializers'] + '.data'
RUNTIME_MODEL_FILE_ROLES = (
    'image_preprocess',
    'vision',
    'shared_initializers',
    'shared_initializers_data',
    'image_prefill_greedy',
    'image_prefill_penalty_greedy',
    'image_prefill_sampling',
    'image_decode_greedy',
    'image_decode_penalty_greedy',
    'image_decode_sampling',
    'kv_slice',
    'kv_split2',
    'kv_concat',
    'rope_shift',
)
MODEL_FILE_NAME_METADATA = {
    f'model_file_name_{key}': MODEL_FILE_NAMES[key]
    for key in RUNTIME_MODEL_FILE_ROLES
}


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


def build_static_firered_image_inputs(visual_model, image_resize, num_images):
    patch_size = int(visual_model.patch_size)
    grid_h = image_resize[0] // patch_size
    grid_w = image_resize[1] // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int32).repeat(num_images, 1)

    pos_embeds = visual_model.fast_pos_embed_interpolate(image_grid_thw).unsqueeze(0).float()
    rotary_raw = visual_model.rot_pos_emb(image_grid_thw).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    rotary_cos = torch.cat([rotary_raw.cos(), rotary_raw.cos()], dim=-1).float()
    rotary_sin = torch.cat([-rotary_raw.sin(), rotary_raw.sin()], dim=-1).float()

    seq_per_image = grid_h * grid_w
    total_seq = seq_per_image * num_images
    attention_mask = torch.full((1, 1, total_seq, total_seq), -128, dtype=torch.int8)
    for image_index in range(num_images):
        start = image_index * seq_per_image
        end = start + seq_per_image
        attention_mask[..., start:end, start:end] = 0

    return image_grid_thw, pos_embeds, rotary_cos, rotary_sin, attention_mask


def build_firered_prompt_layout(processor, num_images):
    """Build the processor-native text/image token layout for static image concat."""
    tokenizer = getattr(processor, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError('FireRedOCR processor does not expose a tokenizer.')
    image_token = getattr(processor, 'image_token', None)
    image_token_id = getattr(processor, 'image_token_id', None)
    if not image_token or image_token_id is None:
        raise ValueError('FireRedOCR processor must expose image_token and image_token_id.')

    content = [{'type': 'image'} for _ in range(num_images)]
    content.append({'type': 'text', 'text': ''})
    conversation = [{'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    image_slots = prompt.count(image_token)
    if image_slots != num_images:
        raise ValueError(f'Chat template produced {image_slots} image slots, expected {num_images}.')

    expanded_image_token = image_token * IMAGE_TOKEN_LENGTH
    token_ids = tokenizer(
        prompt.replace(image_token, expanded_image_token),
        add_special_tokens=False,
    )['input_ids']
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    image_token_positions = [i for i, token_id in enumerate(token_ids) if token_id == int(image_token_id)]
    expected_image_tokens = IMAGE_TOKEN_LENGTH * num_images

    if len(image_token_positions) != expected_image_tokens:
        raise ValueError(
            f'Chat template produced {len(image_token_positions)} expanded image tokens, expected {expected_image_tokens}.'
        )

    image_start = image_token_positions[0]
    image_end = image_token_positions[-1] + 1
    if image_end - image_start != expected_image_tokens:
        raise ValueError('Expanded image tokens must occupy a contiguous prompt span for static concat export.')

    return token_ids, image_start, image_end, [int(token_id == int(image_token_id)) for token_id in token_ids]


class GREEDY_SEARCH(torch.nn.Module):
    """Token-only greedy contract used by merged FireRed decode graphs."""

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
    """Qwen-compatible TopTok sampling with sign-aware repetition handling."""

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


class FIRE_RED_TOKENIZER_PROCESSOR:
    """Expose the checkpoint chat template through the processor interface used here."""

    def __init__(self, tokenizer, image_token_id):
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.image_token = tokenizer.convert_ids_to_tokens(self.image_token_id)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ref, start, end, dim):
        start_value, end_value = int(start), int(end)
        return torch.tensor(
            [start_value, end_value - start_value, ref.shape[dim] - end_value],
            dtype=torch.int64,
        )

    @staticmethod
    def symbolic(graph, ref, start, end, dim):
        shape = graph.op("Shape", ref)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        window = graph.op("Sub", end, start)
        tail = graph.op("Sub", dim_size, end)
        return graph.op("Concat", start, window, tail, axis_i=0)


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


def window_split_sizes(ref, start, end, dim):
    if dim < 0:
        dim += ref.dim()
    return WINDOW_SPLIT_SIZES.apply(ref, start, end, dim)


def slice_keep_middle(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SLICE_KEEP_MIDDLE.apply(values, sizes, dim)


class KV_SLICE(torch.nn.Module):
    """Apply slice to KV cache tensors."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary     = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.save_key     = [None] * num_layers
        self.save_value   = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end   = all_inputs[-1]
        sizes = window_split_sizes(all_inputs[0], slice_start, slice_end, -1)
        for i in range(self.num_layers):
            self.save_key[i]   = slice_keep_middle(all_inputs[i], sizes, -1)
            self.save_value[i] = slice_keep_middle(all_inputs[i + self.num_layers], sizes, -2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    if self.kv_grouped_6d:
                        self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -3)
                    else:
                        self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -2)
                elif self.kv_grouped_6d:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.save_k_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_4], sizes, -3)
                    self.save_v_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_5], sizes, -3)
                else:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.save_k_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_4], sizes, -2)
                    self.save_v_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_5], sizes, -2)
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


class SPLIT_POINT_SIZES(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ref, split_at, dim):
        split_value = int(split_at)
        return torch.tensor([split_value, ref.shape[dim] - split_value], dtype=torch.int64)

    @staticmethod
    def symbolic(graph, ref, split_at, dim):
        shape = graph.op("Shape", ref)
        dim_size = graph.op(
            "Gather",
            shape,
            graph.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            axis_i=0,
        )
        remainder = graph.op("Sub", dim_size, split_at)
        return graph.op("Concat", split_at, remainder, axis_i=0)


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


def split_point_sizes(ref, split_at, dim):
    if dim < 0:
        dim += ref.dim()
    return SPLIT_POINT_SIZES.apply(ref, split_at, dim)


def split_prefix_suffix(values, sizes, dim):
    if dim < 0:
        dim += values.dim()
    return SPLIT_PREFIX_SUFFIX.apply(values, sizes, dim)


class KV_SPLIT2(torch.nn.Module):
    """Split every cache tensor into an immutable prefix and mutable suffix."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.prefix_key   = [None] * num_layers
        self.prefix_value = [None] * num_layers
        self.window_key   = [None] * num_layers
        self.window_value = [None] * num_layers
        if self.kv_quantized:
            self.prefix_k_scale = [None] * num_layers
            self.prefix_v_scale = [None] * num_layers
            self.window_k_scale = [None] * num_layers
            self.window_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.prefix_k_bias = [None] * num_layers
                self.prefix_v_bias = [None] * num_layers
                self.window_k_bias = [None] * num_layers
                self.window_v_bias = [None] * num_layers

    def forward(self, *all_inputs):
        split_at = all_inputs[-1]
        sizes = split_point_sizes(all_inputs[0], split_at, -1)
        for i in range(self.num_layers):
            self.prefix_key[i], self.window_key[i] = split_prefix_suffix(all_inputs[i], sizes, -1)
            self.prefix_value[i], self.window_value[i] = split_prefix_suffix(all_inputs[i + self.num_layers], sizes, -2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    value_scale_axis = -3 if self.kv_grouped_6d else -2
                    self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, value_scale_axis)
                elif self.kv_grouped_6d:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.prefix_k_bias[i], self.window_k_bias[i] = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_4], sizes, -3)
                    self.prefix_v_bias[i], self.window_v_bias[i] = split_prefix_suffix(all_inputs[i + self.num_layers_5], sizes, -3)
                else:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.prefix_k_bias[i], self.window_k_bias[i] = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_4], sizes, -2)
                    self.prefix_v_bias[i], self.window_v_bias[i] = split_prefix_suffix(all_inputs[i + self.num_layers_5], sizes, -2)
        if self.kv_sym:
            return (
                *self.prefix_key, *self.prefix_value, *self.prefix_k_scale, *self.prefix_v_scale,
                *self.window_key, *self.window_value, *self.window_k_scale, *self.window_v_scale,
            )
        if self.kv_quantized:
            return (
                *self.prefix_key, *self.prefix_value, *self.prefix_k_scale, *self.prefix_k_bias, *self.prefix_v_scale, *self.prefix_v_bias,
                *self.window_key, *self.window_value, *self.window_k_scale, *self.window_k_bias, *self.window_v_scale, *self.window_v_bias,
            )
        return *self.prefix_key, *self.prefix_value, *self.window_key, *self.window_value


class KV_CONCAT(torch.nn.Module):
    """Join two cache windows while preserving each tensor's sequence axis."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.value_axis   = -3 if self.kv_grouped_6d else -2
        self.save_key     = [None] * num_layers
        self.save_value   = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_layers
                self.save_v_bias = [None] * num_layers

    @staticmethod
    def _concat(prefix, suffix, dim):
        return torch.cat([prefix, suffix], dim=dim)

    def forward(self, *all_inputs):
        split = len(all_inputs) // 2
        prefix, suffix = all_inputs[:split], all_inputs[split:]
        for i in range(self.num_layers):
            self.save_key[i] = self._concat(prefix[i], suffix[i], dim=-1)
            self.save_value[i] = self._concat(
                prefix[i + self.num_layers], suffix[i + self.num_layers], dim=-2
            )
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = self._concat(
                        prefix[i + self.num_layers_2], suffix[i + self.num_layers_2], dim=-1
                    )
                    self.save_v_scale[i] = self._concat(
                        prefix[i + self.num_layers_3], suffix[i + self.num_layers_3], dim=self.value_axis
                    )
                else:
                    self.save_k_scale[i] = self._concat(
                        prefix[i + self.num_layers_2], suffix[i + self.num_layers_2], dim=-1
                    )
                    self.save_k_bias[i] = self._concat(
                        prefix[i + self.num_layers_3], suffix[i + self.num_layers_3], dim=-1
                    )
                    self.save_v_scale[i] = self._concat(
                        prefix[i + self.num_layers_4], suffix[i + self.num_layers_4], dim=self.value_axis
                    )
                    self.save_v_bias[i] = self._concat(
                        prefix[i + self.num_layers_5], suffix[i + self.num_layers_5], dim=self.value_axis
                    )
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation (θ=π/4) to the head_dim axis *before* quantization.
       The rotation spreads outlier energy across dimension pairs, making the
       value distribution more uniform and reducing quantization error —
       especially at 4-bit.  During attention the inverse rotation is fused
       algebraically so that no full dequant + inverse-rotate is needed.

     2. **Enhanced Hadamard transform** (USE_HADAMARD, Q4 and Q8 modes):
         applies a deterministic randomized Walsh-Hadamard transform within
         each quantization group.  A fixed Rademacher sign pattern is applied
         before the transform, and non-power-of-two groups are zero-padded to
         the next power of two and cropped back.  This keeps the transform
         orthogonal on the active channels while improving energy spreading
         versus a plain fixed Hadamard block.

    3. **Channel shuffle** (USE_SHUFFLE, Q4 and Q8 modes): interleaves
       channels across groups so that high-variance channels are evenly
       distributed.  Like Hadamard, this also enables per-group Q8
       quantization.

     4. **Residual bias correction** (asymmetric modes): computes the
         mean quantization residual for each block/group and folds it into
         the stored bias.  This reduces systematic dequantization drift for
         Q4 without changing the KV cache layout.
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

        # ── Quantization range ───────────────────────────────────────
        # Symmetric: quantize directly into signed integer domains.
        # Q4 uses int4-style codes in [-8, 7] stored as 4-bit two's-complement nibbles.
        # Non-CUDA Q8 stores true int8 tensors; CUDA Q8 keeps byte codes that are packed into int32
        # because the downstream CUDA tile path does not consume uint8/int8 KV tensors directly.
        # Asymmetric: full [0, QMAX] range with per-block min as bias.
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

        # ── Group parameters (ROTARY_Q4 always grouped; Q8/ROTARY_Q8 grouped when hadamard/shuffle enabled) ──
        # When KV_QUANT_GROUP_SIZE >= head_dim, num_groups=1 which is equivalent to per-head quant,
        # so skip the grouped path to avoid unnecessary reshape overhead (Q4 always needs grouping).
        # Also disable hadamard/shuffle when not grouped, since their buffers depend on valid group sizes.
        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        # ── Q8_CUDA int32 packing constants ──────────────────────────
        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # ── Rotary transform buffers ─────────────────────────────────
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

        # ── Enhanced Hadamard transform buffers ───────────────────────
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            # Pre-compute Hadamard butterfly level widths
            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # ── Clip sigma buffer ─────────────────────────────────────────
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # ── Channel shuffle buffers ──────────────────────────────────
        if self.use_shuffle:
            # Interleaved permutation: distributes channel i to group (i % num_groups).
            # This ensures that adjacent channels (which often share similar
            # variance properties) end up in different quantization groups,
            # preventing any single group from accumulating all high-variance
            # channels and dominating the Q4 quantization range.
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Walsh-Hadamard helpers
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        """Apply a deterministic randomized Walsh-Hadamard transform on the last dim.

        Forward path uses D·H for row vectors, where D is a fixed Rademacher
        diagonal.  The inverse path uses H·D.  Non-power-of-two group sizes are
        padded to the next power of two and cropped back after the transform.
        """
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

    # ══════════════════════════════════════════════════════════════════
    # Sigma-based clipping (applied per Q4 group before quantization)
    # ══════════════════════════════════════════════════════════════════
    def _clip_to_sigma(self, x, dim):
        """Clip values to mean ± clip_sigma*std per quantization block.

        Tightens the quantization range (Q4 or Q8) by saturating extreme
        outliers.  For grouped modes clips per-group; otherwise per-head.
        Uses population variance (biased) for stability with small group
        sizes and avoids division or sqrt on zero.

        All ops (mean, square, sqrt, clamp) map to standard ONNX operators.
        """
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    # ══════════════════════════════════════════════════════════════════
    # Rotary flip helpers (view + flip + view)
    # ══════════════════════════════════════════════════════════════════
    def _flip_k(self, k, batch_size):
        """Swap halves along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)"""
        return k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1).flip(-3).view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _flip_v(self, v, batch_size):
        """Swap halves along head_dim (last dim). v: (B, KVH, 1, S, head_dim)"""
        return v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def _flip_q(self, q, batch_size):
        """Swap halves along head_dim (last dim). q: (B, KVH, G, Qlen, head_dim)"""
        return q.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ── Forward rotation (applied during quantization) ───────────────
    def rotate_k(self, k, batch_size):
        """Rotate key pairs along head_dim (dim 3).
        k: (B, KVH, 1, head_dim, S)"""
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        """Rotate value pairs along head_dim (dim -1).
        v: (B, KVH, 1, S, head_dim)"""
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    # ── Inverse rotation (fused into attention computation) ──────────
    def rotate_q(self, q, batch_size):
        """Forward-rotate query along head_dim (last dim) for fused key attention.
        By orthogonality: <Q, R^{-1}(K)> = <R(Q), K>, so we need R(Q).
        q: (B, KVH, G, Qlen, head_dim)"""
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        """Inverse-rotate dequantized V along head_dim (last dim).
        v: (B, KVH, 1, S, head_dim)"""
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        """Inverse-rotate dequantized K along head_dim (dim 3).
        k: (B, KVH, 1, head_dim, S)"""
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        """Inverse-rotate attention output along head_dim (last dim).
        Applied post-matmul instead of pre-matmul on V, since the rotation
        is position-independent: attn @ R^{-1}(V) = R^{-1}(attn @ V).
        x: (B, KVH, G, Qlen, head_dim)"""
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Hadamard transform helpers (within quantization groups, Q4 and Q8)
    # ══════════════════════════════════════════════════════════════════
    def hadamard_k(self, k, batch_size):
        """Apply randomized Walsh-Hadamard mixing within key quantization groups."""
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def inverse_hadamard_k(self, k, batch_size):
        """Restore a grouped key after dequantization."""
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2), inverse=True).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        """Apply randomized Walsh-Hadamard mixing within value quantization groups."""
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        """Apply the forward randomized Walsh-Hadamard transform to grouped queries."""
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        """Apply the inverse randomized Walsh-Hadamard transform to attention output."""
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Block quantization
    # ══════════════════════════════════════════════════════════════════
    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        """Finalize asymmetric quantization with optional residual bias correction."""
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
        """Quantize to signed integers, then encode into the selected storage container."""
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
        """Per-block quantization. Symmetric (absmax) or asymmetric (min-max)."""
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim, batch_size)
        if self.use_sym:
            # Symmetric: absmax-based signed-int quantization.
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
        """Per-group quantization (Q4 or Q8). Symmetric (absmax) or asymmetric (min-max)."""
        if self.use_sym:
            # Symmetric: absmax scaling into signed integer domains.
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
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
            # Asymmetric: min-max scaling, full [0, 15] range
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            return x_packed, scale, block_min

    # ══════════════════════════════════════════════════════════════════
    # CUDA packing / unpacking (4 uint8 → 1 int32)
    # ══════════════════════════════════════════════════════════════════
    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        """Pack 4 uint8 values into a single int32 for CUDA-friendly storage."""
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        """Unpack int32 back into 4 uint8 channels."""
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

    # ══════════════════════════════════════════════════════════════════
    # Q4 packing / unpacking (2 nibbles → 1 byte)
    # ══════════════════════════════════════════════════════════════════
    def pack_q4_k(self, x, batch_size):
        """Pack Q4 keys: (B,KVH,1, D, S) → (B,KVH,1, D//2, S)."""
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        """Pack Q4 values: (B,KVH,1, S, D) → (B,KVH,1, S, D//2)."""
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        """Unpack Q4 keys: (B,KVH,1, D//2, S) → (B,KVH,1, D, S)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        """Unpack Q4 values: (B,KVH,1, S, D//2) → (B,KVH,1, S, D)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def quantize_key(self, keys, batch_size):
        """Encode a key cache tensor without requiring a value-cache tensor."""
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)

        if self.use_sym:
            packed, scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            if self.is_q4:
                packed = self.pack_q4_k(packed, batch_size)
            if self.is_q8_cuda:
                packed = self.pack_cuda(
                    packed,
                    -2,
                    batch_size,
                    self.num_kv_heads,
                    self.head_dim // (8 if self.is_q4 else 4),
                )
            return packed, scale, None

        packed, scale, bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
        if self.is_q4:
            packed = self.pack_q4_k(packed, batch_size)
        if self.is_q8_cuda:
            packed = self.pack_cuda(
                packed,
                -2,
                batch_size,
                self.num_kv_heads,
                self.head_dim // (8 if self.is_q4 else 4),
            )
        return packed, scale, bias

    def dequantize_key(self, packed, scale, bias, batch_size):
        """Decode a key cache tensor into the floating-point attention layout."""
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.float()
            if bias is not None:
                bias = bias.float()

        if self.is_q8_cuda:
            packed = self.unpack_cuda(
                packed,
                -2,
                batch_size,
                self.num_kv_heads,
                self.head_dim // 2 if self.is_q4 else self.head_dim,
            )
        if self.is_q4:
            values = self.unpack_q4_k(packed, batch_size)
            if self.use_sym:
                values = self._decode_signed_q4_storage(values)
        else:
            values = self._decode_signed_q8_storage(packed) if self.use_sym else packed
        values = values.float()

        if self.is_grouped:
            groups = values.reshape(
                batch_size,
                self.num_kv_heads,
                1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
                -1,
            )
            keys = groups * scale if self.use_sym else groups * scale + bias
            keys = keys.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        else:
            keys = values * scale if self.use_sym else values * scale + bias

        if self.use_hadamard:
            keys = self.inverse_hadamard_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.unshuffle_idx)
        if self.is_rotary:
            keys = self.inverse_rotate_k(keys, batch_size)
        return keys

    # ══════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════
    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            # 1. Rotate before quantization
            keys   = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            # 1b. Interleave channels across groups (spreads high-variance channels)
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            # 3. Hadamard within quantization groups (spreads values for better per-group quantization, works for Q4 and Q8)
            keys   = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            # 4a. Symmetric quantize (no bias)
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
            # 4b. Asymmetric min-max quantize (with bias)
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class ROPE_SHIFT(torch.nn.Module):
    """Shift retained floating-point key caches by a standard text RoPE offset."""

    def __init__(self, num_layers, head_dim, num_kv_heads, inv_freq, max_seq_len):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_kv_heads = num_kv_heads
        self.compute_in_f32 = COMPUTE_IN_F32
        inv_freq = inv_freq.detach().float().reshape(-1)
        if inv_freq.numel() * 2 != head_dim:
            raise ValueError(
                f"RoPE inv_freq length {inv_freq.numel()} is incompatible with head_dim {head_dim}."
            )
        inv_freq_full = torch.cat([inv_freq, inv_freq], dim=0).view(1, 1, 1, head_dim, 1)
        half_sign = torch.cat([
            torch.ones(self.head_dim_half),
            -torch.ones(self.head_dim_half),
        ], dim=0).view(1, 1, 1, head_dim, 1)
        shifts = torch.arange(max_seq_len + 1, dtype=torch.float32).view(max_seq_len + 1, 1, 1, 1, 1)
        angle = shifts * inv_freq_full
        angle = angle - 6.283185307179586 * torch.round(angle * (1.0 / 6.283185307179586))
        self.register_buffer("cos_shift", torch.cos(angle).half(), persistent=False)
        self.register_buffer("sin_shift", (torch.sin(angle) * half_sign).half(), persistent=False)

    def _flip_k(self, keys):
        batch_size = keys.shape[0]
        keys = keys.reshape(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        return keys.flip(-3).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cache_dtype = all_inputs[0].dtype
        force_f32 = self.compute_in_f32 and cache_dtype != torch.float32
        cosine = self.cos_shift.index_select(0, shift)
        sine = self.sin_shift.index_select(0, shift)
        if cache_dtype == torch.float32 or force_f32:
            cosine = cosine.float()
            sine = sine.float()

        outputs = []
        for index in range(self.num_layers):
            keys = all_inputs[index].float() if force_f32 else all_inputs[index]
            shifted = keys * cosine + self._flip_k(keys) * sine
            outputs.append(shifted.to(cache_dtype) if force_f32 else shifted)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):
    """Rotate quantized key caches by decode-time text RoPE and requantize them."""

    def __init__(self, num_layers, head_dim, num_kv_heads, inv_freq, max_seq_len, quantizer, is_asymmetric):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_kv_heads = num_kv_heads
        self.quantizer = quantizer
        self.is_asymmetric = is_asymmetric
        inv_freq = inv_freq.detach().float().reshape(-1)
        if inv_freq.numel() * 2 != head_dim:
            raise ValueError(
                f"RoPE inv_freq length {inv_freq.numel()} is incompatible with head_dim {head_dim}."
            )
        inv_freq_full = torch.cat([inv_freq, inv_freq], dim=0).view(1, 1, 1, head_dim, 1)
        half_sign = torch.cat([
            torch.ones(self.head_dim_half),
            -torch.ones(self.head_dim_half),
        ], dim=0).view(1, 1, 1, head_dim, 1)
        shifts = torch.arange(max_seq_len + 1, dtype=torch.float32).view(max_seq_len + 1, 1, 1, 1, 1)
        angle = shifts * inv_freq_full
        angle = angle - 6.283185307179586 * torch.round(angle * (1.0 / 6.283185307179586))
        self.register_buffer("cos_shift", torch.cos(angle).half(), persistent=False)
        self.register_buffer("sin_shift", (torch.sin(angle) * half_sign).half(), persistent=False)

    def _flip_k(self, keys):
        batch_size = keys.shape[0]
        keys = keys.reshape(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1)
        return keys.flip(-3).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cosine = self.cos_shift.index_select(0, shift).float()
        sine = self.sin_shift.index_select(0, shift).float()
        keys = all_inputs[:self.num_layers]
        scales = all_inputs[self.num_layers:2 * self.num_layers]
        biases = all_inputs[2 * self.num_layers:3 * self.num_layers] if self.is_asymmetric else None

        output_keys, output_scales, output_biases = [], [], []
        for index in range(self.num_layers):
            batch_size = keys[index].shape[0]
            bias = biases[index] if self.is_asymmetric else None
            raw_keys = self.quantizer.dequantize_key(keys[index], scales[index], bias, batch_size)
            shifted = raw_keys * cosine + self._flip_k(raw_keys) * sine
            packed, scale, new_bias = self.quantizer.quantize_key(shifted, batch_size)
            output_keys.append(packed)
            output_scales.append(scale)
            if self.is_asymmetric:
                output_biases.append(new_bias)
        if self.is_asymmetric:
            return (*output_keys, *output_scales, *output_biases)
        return (*output_keys, *output_scales)


class LLM_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Convert raw image tensors into normalized FireRed vision patches."""

    def __init__(self, image_resize, visual_model, pos_embeds, rotary_cos, rotary_sin,
                 attention_mask, vision_temporal_patch_size, dynamic_shape=False):
        super().__init__()
        self.target_h, self.target_w = (int(value) for value in image_resize)
        self.dynamic_shape = dynamic_shape
        self.patch_size = int(visual_model.patch_size)
        self.spatial_merge_size = int(visual_model.spatial_merge_size)
        self.vision_temporal_patch_size = int(vision_temporal_patch_size)
        if self.vision_temporal_patch_size <= 0:
            raise ValueError('vision temporal_patch_size must be positive.')
        self.grid_h = self.target_h // self.patch_size
        self.grid_w = self.target_w // self.patch_size
        self.grid_h_merged = self.grid_h // self.spatial_merge_size
        self.grid_w_merged = self.grid_w // self.spatial_merge_size
        self.seq_per_image = self.grid_h * self.grid_w
        self.register_buffer('pos_embeds', pos_embeds.float(), persistent=False)
        self.register_buffer('rotary_cos', rotary_cos.float(), persistent=False)
        self.register_buffer('rotary_sin', rotary_sin.float(), persistent=False)
        self.register_buffer('attention_mask', attention_mask.float(), persistent=False)
        self.register_buffer(
            'image_mean', torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            'image_std', torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        num_images = pixel_values.shape[0]
        pixel_values = pixel_values.float()
        if self.dynamic_shape or pixel_values.shape[-2] != self.target_h or pixel_values.shape[-1] != self.target_w:
            pixel_values = F.interpolate(
                pixel_values,
                size=[self.target_h, self.target_w],
                mode='bilinear',
                align_corners=False,
            )
        pixel_values = (pixel_values / 255.0 - self.image_mean) / self.image_std
        pixel_values = pixel_values.reshape(
            num_images, 3,
            self.grid_h_merged, self.spatial_merge_size, self.patch_size,
            self.grid_w_merged, self.spatial_merge_size, self.patch_size,
        )
        pixel_values = pixel_values.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(
            -1, 3, 1, self.patch_size, self.patch_size
        )
        pixel_values = pixel_values.repeat(1, 1, self.vision_temporal_patch_size, 1, 1)
        output_anchor = pixel_values.reshape(-1)[0] * 0.0
        if self.dynamic_shape:
            total_seq = num_images * self.seq_per_image
            return (
                pixel_values,
                self.pos_embeds[:, :total_seq] + output_anchor,
                self.rotary_cos[..., :total_seq] + output_anchor,
                self.rotary_sin[..., :total_seq] + output_anchor,
                self.attention_mask[..., :total_seq, :total_seq] + output_anchor,
            )
        return (
            pixel_values,
            self.pos_embeds + output_anchor,
            self.rotary_cos + output_anchor,
            self.rotary_sin + output_anchor,
            self.attention_mask + output_anchor,
        )


class LLM_VISION(torch.nn.Module):
    """Run FireRed's image encoder and emit raw DeepStack and vision features."""

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        self.visual_model = llm.model.visual
        self.num_heads = int(llm.config.vision_config.num_heads)
        self.head_dim = int(llm.config.vision_config.hidden_size) // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self.embed_dim = int(self.visual_model.patch_embed.embed_dim)
        self.batch_size = 1
        self.deepstack_modules = self.visual_model.deepstack_merger_list
        self._deepstack_map = {
            layer_num: index
            for index, layer_num in enumerate(self.visual_model.deepstack_visual_indexes)
        }

        scaling = self.head_dim ** -0.25
        embed_dim_patch = self.visual_model.patch_embed.embed_dim
        for block in self.visual_model.blocks:
            block.attn.qkv.weight.data[:-embed_dim_patch] *= scaling
            if block.attn.qkv.bias is not None:
                block.attn.qkv.bias.data[:-embed_dim_patch] *= scaling
            self._fuse_norm(block.norm1, block.attn.qkv)
            self._fuse_norm(block.norm2, block.mlp.linear_fc1)
        for deepstack_layer in self.deepstack_modules:
            self._fuse_norm(deepstack_layer.norm, deepstack_layer.linear_fc1)
        self._fuse_norm(self.visual_model.merger.norm, self.visual_model.merger.linear_fc1)
        if REORDER_VISION_MLP_FOR_QUANT:
            self._reorder_mlp_for_quant(REORDER_KEY)

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                LLM_VISION._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _fuse_norm(norm, linear):
        norm_weight = norm.weight.data
        norm_bias = getattr(norm, 'bias', None)
        if norm_bias is not None:
            norm_bias = norm_bias.data
            if linear.weight.shape[1] != norm_bias.shape[0]:
                repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
                norm_weight = norm_weight.repeat(repeat_factor)
                norm_bias = norm_bias.repeat(repeat_factor)
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.weight.shape[0], dtype=linear.weight.dtype))
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        elif linear.weight.shape[1] != norm_weight.shape[0]:
            norm_weight = norm_weight.repeat(linear.weight.shape[1] // norm_weight.shape[0])
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        if hasattr(norm, 'bias'):
            norm.bias = None

    @staticmethod
    def _channel_permutation(weight, key):
        absolute = weight.abs()
        if key == 'rms':
            statistic = (weight * weight).mean(0).sqrt()
        elif key == 'L4':
            statistic = absolute.pow(4).mean(0).pow(0.25)
        elif key == 'std':
            statistic = weight.std(0)
        else:
            statistic = absolute.mean(0)
        return torch.argsort(statistic)

    @staticmethod
    def _reorder_mlp_pair(first, second, key):
        permutation = LLM_VISION._channel_permutation(second.weight.data, key)
        first.weight.data.copy_(first.weight.data[permutation])
        if first.bias is not None:
            first.bias.data.copy_(first.bias.data[permutation])
        second.weight.data.copy_(second.weight.data[:, permutation])

    def _reorder_mlp_for_quant(self, key):
        with torch.no_grad():
            for block in self.visual_model.blocks:
                self._reorder_mlp_pair(
                    block.mlp.linear_fc1, block.mlp.linear_fc2, key
                )
            for merger in self.deepstack_modules:
                self._reorder_mlp_pair(merger.linear_fc1, merger.linear_fc2, key)
            merger = self.visual_model.merger
            self._reorder_mlp_pair(merger.linear_fc1, merger.linear_fc2, key)

    def _rotate_half(self, values):
        values = values.view(2, self.batch_size, self.num_heads, -1, 2, self.head_dim_half)
        return values.flip(-2).view(2, self.batch_size, self.num_heads, -1, self.head_dim)

    def forward(self, pixel_values, pos_embeds, rotary_cos, rotary_sin, attention_mask):
        vision_hidden_states = self.visual_model.patch_embed.proj(pixel_values.float())
        vision_hidden_states = vision_hidden_states.view(self.batch_size, -1, self.embed_dim)
        vision_hidden_states = vision_hidden_states + pos_embeds.float()
        rotary_cos, rotary_sin, attention_mask = rotary_cos.float(), rotary_sin.float(), attention_mask.float()
        deepstack_features = []
        for layer_num, block in enumerate(self.visual_model.blocks):
            qkv = block.attn.qkv(block.norm1(vision_hidden_states))
            qkv = qkv.reshape(self.batch_size, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            qk, values = qkv.split([2, 1], dim=0)
            qk = qk * rotary_cos + self._rotate_half(qk) * rotary_sin
            query, key = qk.split([1, 1], dim=0)
            attention = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) + attention_mask, dim=-1)
            attention = torch.matmul(attention, values)
            attention = attention.transpose(2, 3).reshape(self.batch_size, -1, block.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + block.attn.proj(attention)
            mlp_out = block.mlp.linear_fc2(block.mlp.act_fn(block.mlp.linear_fc1(block.norm2(vision_hidden_states))))
            vision_hidden_states = vision_hidden_states + mlp_out
            if layer_num in self._deepstack_map:
                deepstack_layer = self.deepstack_modules[self._deepstack_map[layer_num]]
                deepstack_hidden_states = vision_hidden_states.view(
                    self.batch_size, -1, deepstack_layer.hidden_size
                )
                feature = deepstack_layer.linear_fc2(
                    deepstack_layer.act_fn(
                        deepstack_layer.linear_fc1(deepstack_layer.norm(deepstack_hidden_states))
                    )
                )
                deepstack_features.append(feature)
        vision_hidden_states = self.visual_model.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(
            self.batch_size, -1, self.visual_model.merger.hidden_size
        )
        vision_hidden_states = self.visual_model.merger.linear_fc2(
            self.visual_model.merger.act_fn(
                self.visual_model.merger.linear_fc1(
                    vision_hidden_states
                )
            )
        )
        return (*deepstack_features, vision_hidden_states)


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Replace the processor's contiguous image-token span with vision features."""

    def __init__(self, image_start, image_end, deepstack_features_len):
        super().__init__()
        self.image_start = int(image_start)
        self.image_end = int(image_end)
        self.deepstack_features_len = int(deepstack_features_len)

    def forward(self, *all_inputs):
        deepstack_features = all_inputs[:self.deepstack_features_len]
        text_hidden_states = all_inputs[self.deepstack_features_len]
        vision_hidden_states = all_inputs[self.deepstack_features_len + 1]
        head = text_hidden_states[:, :self.image_start]
        tail = text_hidden_states[:, self.image_end:]
        concat_hidden_states = torch.cat([head, vision_hidden_states, tail], dim=1)
        zero_head = torch.zeros_like(head)
        zero_tail = torch.zeros_like(tail)
        aligned_deepstack = [
            torch.cat([zero_head, feature, zero_tail], dim=1)
            for feature in deepstack_features
        ]
        return (*aligned_deepstack, concat_hidden_states)


class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for image+text prefill."""

    def __init__(self, llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        super().__init__()
        total_len = len(mm_token_type_ids)
        total_max = max_seq_len + total_len
        self.register_buffer("attention_mask", (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128)
        cos, sin = self._build_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len)
        self.register_buffer('cos_rotary_pos_emb', torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2), persistent=False)
        self.register_buffer('sin_rotary_pos_emb', torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        """Build mRoPE position table with mm_token_type_ids-driven position assignment."""
        spatial_merge_size = llm.config.vision_config.spatial_merge_size
        input_type_group = []
        for key, group in itertools.groupby(enumerate(mm_token_type_ids), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        image_iter = iter(image_grid_thw)

        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos)
                current_pos += text_len
            else:
                grid_thw = next(image_iter)
                llm_grid_h = grid_thw[1].item() // spatial_merge_size
                llm_grid_w = grid_thw[2].item() // spatial_merge_size
                image_seqlen = llm_grid_h * llm_grid_w
                position_temporal = torch.full((image_seqlen,), float(current_pos))
                position_height = (torch.arange(llm_grid_h, dtype=torch.float32) + current_pos).repeat_interleave(llm_grid_w)
                position_width = (torch.arange(llm_grid_w, dtype=torch.float32) + current_pos).repeat(llm_grid_h)
                llm_pos_ids_list.append(torch.stack([position_temporal, position_height, position_width], dim=0))
                current_pos += max(llm_grid_h, llm_grid_w)

        prefill_positions = torch.cat(llm_pos_ids_list, dim=1)
        tail_positions = torch.arange(max_seq_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos
        position_ids = torch.cat([prefill_positions, tail_positions], dim=-1).unsqueeze(1)

        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (inv_freq_expanded @ position_ids).transpose(-1, -2).unsqueeze(1)
        apply_mrope = getattr(rotary_module, 'apply_interleaved_mrope', None)
        if callable(apply_mrope):
            freqs = apply_mrope(freqs, rotary_module.mrope_section)
        else:
            sections = tuple(int(value) for value in rotary_module.mrope_section)
            if len(sections) != 3 or sum(sections) != freqs.shape[-1]:
                raise ValueError('FireRed rotary module does not provide a compatible mRoPE layout.')
            chunks = torch.split(freqs, sections, dim=-1)
            freqs = torch.cat(
                [chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1
            )
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for a single decode step."""

    def __init__(self, llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_IMAGE_PREFILL._build_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len)
        self.register_buffer('cos_rotary_pos_emb', torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2), persistent=False)
        self.register_buffer('sin_rotary_pos_emb', torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
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
    """
    Main transformer module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE)
      - KV cache management with optional Q8/Q8_CUDA quantization
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
      - Deepstack feature injection from the vision encoder
    """

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size, deepstack_features_len):
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

        # ── Layer count multipliers (for indexing into flat KV input list) ──
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # ── Deepstack features ───────────────────────────────────────────
        self.deepstack_features_len = deepstack_features_len
        self._ds_offset = 3 + deepstack_features_len

        # ── KV cache dtype flags ─────────────────────────────────────────
        self.kv_f16             = (KV_QUANT_DTYPE == "F16")
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
        self.compute_in_f32     = COMPUTE_IN_F32

        # Whether Q8 modes use per-group quantization (enabled by hadamard/shuffle)
        # When KV_QUANT_GROUP_SIZE >= head_dim, per-group is equivalent to per-head, so skip grouping.
        self.kv_q8_grouped      = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim

        # head_dim used for int32 unpack in rotary CUDA modes
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
        hidden_rms_norm = self.llm.model.language_model.layers[0].input_layernorm
        qk_rms_norm = self.llm.model.language_model.layers[0].self_attn.q_norm
        self.hidden_rms_norm_eps = float(
            getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6))
        )
        self.qk_rms_norm_eps = float(
            getattr(qk_rms_norm, "variance_epsilon", getattr(qk_rms_norm, "eps", self.hidden_rms_norm_eps))
        )
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
        )
        self.register_buffer(
            "qk_norm_scale",
            torch.full((self.head_dim,), self.head_dim ** -0.5, dtype=torch.float32),
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
        self._replace_gelu_with_tanh_approximation(self.llm)
        self._fuse_weights(hidden_size)
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)

        # ── Pre-computed per-layer constants (uniform across all layers) ──
        self.o_proj_in_features = self.llm.model.language_model.layers[0].self_attn.o_proj.in_features
        self.mlp_split          = [self.llm.model.language_model.layers[0].mlp.down_proj.in_features] * 2

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Do not mutate lm_head: FireRed ties it to embed_tokens. The equivalent
            # RMSNorm scale is applied immediately before the projection in forward().
            final_norm_scale = self.llm.model.language_model.norm.weight.unsqueeze(0) * norm_factor
            self.register_buffer('final_norm_scale', final_norm_scale)
            del self.llm.model.language_model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features  = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias     = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:

            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)

            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features

        del attn.q_proj, attn.k_proj, attn.v_proj

        # ── Fuse QK norms (absorb scale factors) ────────────────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)

        q_norm_repeated     = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated     = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up         = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    @staticmethod
    def _channel_statistic(weight, key):
        absolute = weight.abs()
        if key == 'rms':
            return (weight * weight).mean(0).sqrt()
        if key == 'L4':
            return absolute.pow(4).mean(0).pow(0.25)
        if key == 'std':
            return weight.std(0)
        return absolute.mean(0)

    def _reorder_downproj_for_quant(self, key):
        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                down_weight = layer.mlp.down_proj.weight
                permutation = torch.argsort(
                    self._channel_statistic(down_weight, key)
                )
                intermediate_size = layer.mlp.down_proj.in_features
                gate_up_weight = layer.mlp.gate_up_proj.weight
                reordered_gate_up = torch.cat(
                    [
                        gate_up_weight[:intermediate_size][permutation],
                        gate_up_weight[intermediate_size:][permutation],
                    ],
                    dim=0,
                )
                layer.mlp.gate_up_proj.weight.data.copy_(reordered_gate_up)
                layer.mlp.down_proj.weight.data.copy_(down_weight[:, permutation])

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                LLM_MAIN._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _rms_norm(x, scale, eps):
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-(self._ds_offset + 1)]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]
        attn_mask_f16      = attention_mask.half() if (self.kv_f16 and not self.compute_in_f32) else None

        for i, layer in enumerate(self.llm.model.language_model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )

            # Fused QKV projection & reshape
            qkv   = layer.self_attn.qkv(hidden_states)
            qkv   = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            # QK normalization & rotary embedding
            qk     = self._rms_norm(qk, self.qk_norm_scale, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
            if self.kv_f16 and not self.compute_in_f32:
                qk_rot = qk_rot.half()

            # Split into query and key, reshape query for GQA
            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            q    = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q    = q.permute(0, 2, 3, 1, 4)

            # Optional FP16 cast for KV
            if self.kv_f16:
                if self.compute_in_f32:
                    k = k.half()
                v = v.half()

            # Transpose K and V into cache layout
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            if self.kv_rotary_q4:
                # ── ROTARY_Q4 ────────────────────────────────────────
                if self.kv_sym:
                    # Symmetric: no stored bias, derived on-the-fly as -zp * scale
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)

                    # Save updated caches (4 types)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    # Fused rotary-dequant attention (symmetric signed-int):
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

                    # Value dequant (symmetric signed-int):
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
                    # Asymmetric: min-max with stored bias
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                    v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],   dim=-3)

                    # Save updated caches (6 types)
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

                    # Fused rotary-dequant attention (asymmetric):
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

                    # Value dequant (asymmetric):
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
                # ── ROTARY_Q8 ────────────────────────────────────────
                if self.kv_sym:
                    # Symmetric: no stored bias, derived on-the-fly as -zp * scale
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)

                    # Save updated caches (4 types)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    # Fused rotary-dequant attention (symmetric signed-int):
                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        # Per-group attention path (with shuffle/hadamard)
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

                        # Value dequant (symmetric signed-int, grouped):
                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        # Per-head attention path (no grouping)
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k_signed)
                        attn          = attn_raw * k_s + attention_mask
                        attn          = torch.softmax(attn, dim=-1)

                        # Value dequant (symmetric signed-int):
                        v_scaled  = v_signed * v_s
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size)
                else:
                    # Asymmetric: min-max with stored bias
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

                    # Save updated caches (6 types)
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

                    # Fused rotary-dequant attention (asymmetric):
                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                    if self.kv_q8_grouped:
                        # Per-group attention path (with shuffle/hadamard)
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

                        # Value dequant (asymmetric, grouped):
                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        # Per-head attention path (no grouping)
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k.float())
                        q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                        attn_bias     = q_bias_factor * k_b + attention_mask
                        attn          = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn          = torch.softmax(attn, dim=-1)

                        # Value dequant with post-matmul inverse rotation:
                        v_scaled  = v.float() * v_s
                        bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size) + bias_term

            elif self.kv_quantized:
                if self.kv_sym:
                    # Symmetric Q8: signed-int quantization, no stored bias
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)

                    # Save updated caches (4 types)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    # Unpack int32-packed Q8 for CUDA path
                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        # Per-group Q8 attention (with shuffle/hadamard)
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

                        # Value dequant (symmetric signed-int, grouped):
                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        # Per-head Q8 attention (no grouping)
                        attn_raw = torch.matmul(q, k_signed)
                        attn     = attn_raw * k_s + attention_mask
                        attn     = torch.softmax(attn, dim=-1)

                        # Value dequant (symmetric signed-int):
                        v_scaled  = v_signed * v_s
                        attn      = torch.matmul(attn, v_scaled)
                else:
                    # Asymmetric Q8: min-max with stored bias
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

                    # Save updated caches (6 types)
                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b

                    # Upcast scale/bias if stored as FP16
                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    # Unpack int32-packed Q8 for CUDA path
                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        # Per-group Q8 attention (with shuffle/hadamard)
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

                        # Value dequant (asymmetric, grouped):
                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        # Per-head Q8 attention (no grouping)
                        attn_raw  = torch.matmul(q, k.float())
                        attn_bias = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                        attn      = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn      = torch.softmax(attn, dim=-1)
                        v_dequant = torch.addcmul(v_b, v.float(), v_s)
                        attn      = torch.matmul(attn, v_dequant)

            else:
                # Concatenate with cached K/V (F16 or F32)
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i]   = k
                self.save_value[i] = v

                if self.kv_f16:
                    if self.compute_in_f32:
                        attn = torch.matmul(q, k.float()) + attention_mask
                        attn = torch.softmax(attn, dim=-1)
                        attn = torch.matmul(attn, v.float())
                    else:
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

            # ── Feed-Forward Network ─────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )

            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, self.mlp_split, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

            if i < self.deepstack_features_len:
                hidden_states = all_inputs[i - self._ds_offset] + hidden_states

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(
            hidden_states[:, -1], self.hidden_norm_scale, self.hidden_rms_norm_eps
        )
        hidden_states = hidden_states * self.final_norm_scale
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


def _load_firered_components():
    if AutoTokenizer is None:
        raise RuntimeError(
            'transformers must provide AutoTokenizer and a Qwen3-VL model implementation. '
            'Use Transformers 4.57 or newer for this FireRedOCR checkpoint.'
        )

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    except (ImportError, ModuleNotFoundError):
        Qwen3VLForConditionalGeneration = None

    try:
        if Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                download_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).eval()
        elif AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(
                download_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).eval()
        else:
            raise RuntimeError('No compatible Qwen3-VL model loader is available.')
        tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            'FireRedOCR requires a Transformers installation with Qwen3VLForConditionalGeneration. '
            'The checkpoint declares Transformers 4.57.0 compatibility.'
        ) from error

    image_token_id = int(getattr(model.config, 'image_token_id'))
    processor = None
    if AutoProcessor is not None:
        try:
            candidate = AutoProcessor.from_pretrained(download_path, trust_remote_code=True)
            if (
                getattr(candidate, 'tokenizer', None) is not None
                and getattr(candidate, 'image_token', None)
                and getattr(candidate, 'image_token_id', None) is not None
            ):
                processor = candidate
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            pass
    if processor is None:
        processor = FIRE_RED_TOKENIZER_PROCESSOR(tokenizer, image_token_id)

    try:
        model.model.language_model
        model.model.visual
        model.config.text_config
        model.config.vision_config
    except AttributeError as error:
        raise RuntimeError('Loaded model does not expose the FireRedOCR language and vision modules.') from error
    return model, processor


def _metadata_values(model, processor, dimensions, kv_facts, image_start, image_end):
    tokenizer = processor.tokenizer
    image_token_id = int(getattr(processor, 'image_token_id'))
    eos_ids = _id_list(getattr(model.config, 'eos_token_id', getattr(tokenizer, 'eos_token_id', None)))
    metadata = {
        'max_seq_len': str(MAX_SEQ_LEN),
        'input_image_size': ','.join(str(value) for value in INPUT_IMAGE_SIZE),
        'image_token_id': str(image_token_id),
        'stop_token_ids': ','.join(str(token_id) for token_id in STOP_TOKEN),
        'eos_token_ids': ','.join(str(token_id) for token_id in eos_ids),
        'input_image_dim': str(INPUT_IMAGE_DIM),
        'vision_batch_size': str(VISION_BATCH_SIZE),
        'image_token_length': str(IMAGE_TOKEN_LENGTH),
        'image_start': str(image_start),
        'image_end': str(image_end),
        'kv_num_tensors': str(dimensions['num_layers'] * len(kv_facts['kv_cache_tensor_order'].split(','))),
        'kv_quant_dtype': KV_QUANT_DTYPE,
        'kv_quant_group_size': str(KV_QUANT_GROUP_SIZE),
        'compute_in_f32': str(int(COMPUTE_IN_F32)),
        'reorder_downproj': str(int(REORDER_DOWNPROJ_FOR_QUANT)),
        'vision_reorder_mlp': str(int(REORDER_VISION_MLP_FOR_QUANT)),
        'reorder_key': REORDER_KEY,
    }
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
    )
    _stamp_metadata(path, metadata)
    print(f'Exported {path.name}.', flush=True)


def _build_kv_layout(batch_size, num_layers, num_kv_heads, head_dim, history_len):
    rotary_modes = {'ROTARY_Q4', 'ROTARY_Q4_CUDA', 'ROTARY_Q8', 'ROTARY_Q8_CUDA'}
    q8_modes = {'Q8', 'Q8_CUDA'}
    rotary_q4 = KV_QUANT_DTYPE in {'ROTARY_Q4', 'ROTARY_Q4_CUDA'}
    rotary_q8_grouped = KV_QUANT_DTYPE in {'ROTARY_Q8', 'ROTARY_Q8_CUDA'} and (
        USE_HADAMARD or USE_SHUFFLE
    ) and KV_QUANT_GROUP_SIZE < head_dim
    q8_grouped = KV_QUANT_DTYPE in q8_modes and (USE_HADAMARD or USE_SHUFFLE) and (
        KV_QUANT_GROUP_SIZE < head_dim
    )
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


def _sequence_axes(source_axes, sequence_name):
    return {
        axis: ('batch_size' if axis == 0 else sequence_name)
        for axis in source_axes
    }


def _export_kv_helpers(
    export_dir,
    dimensions,
    kv_specs,
    kv_tensors,
    metadata,
    rope_inv_freq,
    quantizer,
):
    """Export optional cache-management graphs matching the active KV layout."""
    num_layers = dimensions['num_layers']
    head_dim = dimensions['head_dim']
    num_kv_heads = dimensions['num_kv_heads']

    inputs, input_names, output_names, axes = _kv_io(kv_specs, kv_tensors, num_layers)
    slice_start = torch.tensor([0], dtype=torch.int64)
    slice_end = torch.tensor([1], dtype=torch.int64)
    slice_axes = {name: dict(axes[name]) for name in input_names}
    for name in output_names:
        slice_axes[name] = _sequence_axes(axes[name], 'sliced_len')
    _export_component(
        export_dir / MODEL_FILE_NAMES['kv_slice'],
        KV_SLICE(num_layers, head_dim),
        tuple(inputs + [slice_start, slice_end]),
        input_names + ['slice_start', 'slice_end'],
        output_names,
        slice_axes,
        metadata,
    )

    split_at = torch.tensor([1], dtype=torch.int64)
    prefix_names = [f'prefix_{name}' for name in output_names]
    window_names = [f'window_{name}' for name in output_names]
    split_axes = {name: dict(axes[name]) for name in input_names}
    for source_name, prefix_name, window_name in zip(output_names, prefix_names, window_names):
        split_axes[prefix_name] = _sequence_axes(axes[source_name], 'prefix_len')
        split_axes[window_name] = _sequence_axes(axes[source_name], 'window_len')
    _export_component(
        export_dir / MODEL_FILE_NAMES['kv_split2'],
        KV_SPLIT2(num_layers, head_dim),
        tuple(inputs + [split_at]),
        input_names + ['split_at'],
        prefix_names + window_names,
        split_axes,
        metadata,
    )

    prefix_inputs, suffix_inputs = [], []
    prefix_names, suffix_names, concat_names = [], [], []
    concat_axes = {}
    for name, sequence_axis in kv_specs:
        tensor = kv_tensors[name]
        for layer_index in range(num_layers):
            prefix_name = f'in_prefix_{name}_{layer_index}'
            suffix_name = f'in_suffix_{name}_{layer_index}'
            output_name = f'out_{name}_{layer_index}'
            prefix_inputs.append(tensor)
            suffix_inputs.append(tensor.clone())
            prefix_names.append(prefix_name)
            suffix_names.append(suffix_name)
            concat_names.append(output_name)
            concat_axes[prefix_name] = {0: 'batch_size', sequence_axis: 'prefix_len'}
            concat_axes[suffix_name] = {0: 'batch_size', sequence_axis: 'suffix_len'}
            concat_axes[output_name] = {0: 'batch_size', sequence_axis: 'concat_len'}
    _export_component(
        export_dir / MODEL_FILE_NAMES['kv_concat'],
        KV_CONCAT(num_layers, head_dim),
        tuple(prefix_inputs + suffix_inputs),
        prefix_names + suffix_names,
        concat_names,
        concat_axes,
        metadata,
    )

    _, rope_tensors, _ = _build_kv_layout(
        1,
        num_layers,
        num_kv_heads,
        head_dim,
        4,
    )
    rope_shift = torch.tensor([1], dtype=torch.int64)
    if KV_QUANT_DTYPE in {'F16', 'F32'}:
        rope_inputs = [rope_tensors['key'].clone() for _ in range(num_layers)]
        rope_input_names = [f'in_key_{layer_index}' for layer_index in range(num_layers)]
        rope_output_names = [f'out_key_{layer_index}' for layer_index in range(num_layers)]
        rope_axes = {
            name: {0: 'batch_size', 4: 'history_len'}
            for name in rope_input_names
        }
        rope_axes.update({
            name: {0: 'batch_size', 4: 'history_len'}
            for name in rope_output_names
        })
        rope_module = ROPE_SHIFT(
            num_layers,
            head_dim,
            num_kv_heads,
            rope_inv_freq,
            MAX_SEQ_LEN,
        )
    else:
        sequence_axes = dict(kv_specs)
        rope_names = ['key', 'key_scale']
        if not USE_SYM:
            rope_names.append('key_bias')
        rope_inputs, rope_input_names, rope_output_names, rope_axes = [], [], [], {}
        for name in rope_names:
            sequence_axis = sequence_axes[name]
            for layer_index in range(num_layers):
                input_name = f'in_{name}_{layer_index}'
                output_name = f'out_{name}_{layer_index}'
                rope_inputs.append(rope_tensors[name].clone())
                rope_input_names.append(input_name)
                rope_output_names.append(output_name)
                rope_axes[input_name] = {0: 'batch_size', sequence_axis: 'history_len'}
                rope_axes[output_name] = {0: 'batch_size', sequence_axis: 'history_len'}
        rope_module = ROPE_SHIFT_QUANT(
            num_layers,
            head_dim,
            num_kv_heads,
            rope_inv_freq,
            MAX_SEQ_LEN,
            quantizer,
            not USE_SYM,
        )
    _export_component(
        export_dir / MODEL_FILE_NAMES['rope_shift'],
        rope_module,
        tuple(rope_inputs + [rope_shift]),
        rope_input_names + ['shift'],
        rope_output_names,
        rope_axes,
        metadata,
    )


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
    for data_path in export_dir.iterdir():
        if data_path.is_file() and data_path.suffix != '.onnx' and data_path.name not in referenced:
            data_path.unlink()


def _prepare_export_staging():
    staging_dir = Path(EXPORT_STAGING_DIR)
    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise NotADirectoryError(
                f'Export staging path exists but is not a directory: {staging_dir}.'
            )
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
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    staging_dir.rename(destination)


@torch.inference_mode()
def export_firered():
    if INPUT_IMAGE_DIM not in {4, 5}:
        raise ValueError('INPUT_IMAGE_DIM must be 4 or 5.')

    export_dir = _prepare_export_staging()
    model, processor = _load_firered_components()
    text_config = model.config.text_config
    vision_config = model.config.vision_config
    dimensions = {
        'num_layers': _config_int(text_config, 'num_hidden_layers'),
        'num_heads': _config_int(text_config, 'num_attention_heads'),
        'num_kv_heads': _config_int(text_config, 'num_key_value_heads'),
        'hidden_size': _config_int(text_config, 'hidden_size'),
        'vocab_size': _config_int(text_config, 'vocab_size'),
    }
    dimensions['head_dim'] = _config_int(
        text_config,
        'head_dim',
        dimensions['hidden_size'] // dimensions['num_heads'],
    )
    if dimensions['num_heads'] % dimensions['num_kv_heads']:
        raise ValueError('num_attention_heads must divide num_key_value_heads for FireRed GQA export.')

    for note in normalize_kv_quant_settings(dimensions['head_dim']):
        print(note)

    rope_inv_freq = model.model.language_model.rotary_emb.inv_freq.detach().float().clone()

    token_ids, image_start, image_end, mm_token_type_ids = build_firered_prompt_layout(
        processor,
        VISION_BATCH_SIZE,
    )
    image_grid_thw, pos_embeds, vision_cos, vision_sin, vision_mask = build_static_firered_image_inputs(
        model.model.visual,
        IMAGE_RESIZE,
        VISION_BATCH_SIZE,
    )
    patch_size = _config_int(vision_config, 'patch_size')
    spatial_merge_size = _config_int(vision_config, 'spatial_merge_size')
    temporal_patch_size = _config_int(vision_config, 'temporal_patch_size')
    grid_h = IMAGE_RESIZE[0] // patch_size
    grid_w = IMAGE_RESIZE[1] // patch_size
    image_feature_count = (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size) * VISION_BATCH_SIZE
    if image_feature_count != IMAGE_TOKEN_LENGTH * VISION_BATCH_SIZE:
        raise ValueError('IMAGE_TOKEN_LENGTH does not match the configured FireRed vision grid.')

    kv_specs, kv_tensors, kv_facts = _build_kv_layout(
        1,
        dimensions['num_layers'],
        dimensions['num_kv_heads'],
        dimensions['head_dim'],
        0,
    )
    metadata = _metadata_values(model, processor, dimensions, kv_facts, image_start, image_end)
    metadata_path = export_dir / MODEL_FILE_NAMES['metadata']
    _export_component(
        metadata_path,
        METADATA_CARRIER(),
        (torch.zeros((1,), dtype=torch.int32),),
        ['metadata_marker'],
        ['metadata_marker_out'],
        None,
        metadata,
    )

    trace_ids_len = min(10, len(token_ids))
    if trace_ids_len == 0:
        raise ValueError('The processor chat template produced an empty prompt.')
    ids_len = torch.tensor([trace_ids_len], dtype=torch.int64)
    history_len = torch.zeros((1,), dtype=torch.int64)
    kv_seq_len = ids_len + history_len
    input_ids = torch.tensor([token_ids[:trace_ids_len]], dtype=torch.int32)
    embed = LLM_EMBED(model)
    _export_component(
        export_dir / MODEL_FILE_NAMES['embed'],
        embed,
        (input_ids,),
        ['input_ids'],
        ['text_hidden_states'],
        {
            'input_ids': {0: 'batch_size', 1: 'ids_len'},
            'text_hidden_states': {0: 'batch_size', 1: 'ids_len'},
        },
        metadata,
    )
    del embed, input_ids
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
        temporal_patch_size,
        dynamic_shape=DYNAMIC_IMAGE_SHAPE,
    )
    preprocess_axes = {
        'patches': {0: 'vision_patch_count'},
        'pos': {1: 'vision_patch_count'},
        'cos': {3: 'vision_patch_count'},
        'sin': {3: 'vision_patch_count'},
        'mask': {2: 'vision_patch_count', 3: 'vision_patch_count'},
    }
    if DYNAMIC_IMAGE_SHAPE:
        preprocess_axes['pixel_values'] = {0: 'image_count'}
        if INPUT_IMAGE_DIM == 5:
            preprocess_axes['pixel_values'].update({3: 'image_height', 4: 'image_width'})
        else:
            preprocess_axes['pixel_values'].update({2: 'image_height', 3: 'image_width'})
    _export_component(
        export_dir / MODEL_FILE_NAMES['image_preprocess'],
        image_preprocess,
        (image_input,),
        ['pixel_values'],
        ['patches', 'pos', 'cos', 'sin', 'mask'],
        preprocess_axes,
        metadata,
    )
    del image_preprocess, image_input
    gc.collect()

    patch_count = grid_h * grid_w * VISION_BATCH_SIZE
    patches = torch.zeros(
        (patch_count, 3, temporal_patch_size, patch_size, patch_size), dtype=torch.float32
    )
    deepstack_count = len(model.model.visual.deepstack_visual_indexes)
    vision_output_names = [f'deepstack_feature_{index}' for index in range(deepstack_count)]
    vision_output_names.append('vision_hidden_states')
    vision_axes = {
        'patches': {0: 'vision_patch_count'},
        'pos': {1: 'vision_patch_count'},
        'cos': {3: 'vision_patch_count'},
        'sin': {3: 'vision_patch_count'},
        'mask': {2: 'vision_patch_count', 3: 'vision_patch_count'},
    }
    for output_name in vision_output_names:
        vision_axes[output_name] = {1: 'image_token_count'}
    vision = LLM_VISION(model)
    _export_component(
        export_dir / MODEL_FILE_NAMES['vision'],
        vision,
        (patches, pos_embeds, vision_cos, vision_sin, vision_mask),
        ['patches', 'pos', 'cos', 'sin', 'mask'],
        vision_output_names,
        vision_axes,
        metadata,
    )
    del vision, patches, pos_embeds, vision_cos, vision_sin, vision_mask
    gc.collect()

    text_hidden_states = torch.ones((1, len(token_ids), dimensions['hidden_size']), dtype=torch.float32)
    image_hidden_states = torch.ones(
        (1, image_feature_count, dimensions['hidden_size']), dtype=torch.float32
    )
    deepstack_features = [
        torch.ones((1, image_feature_count, dimensions['hidden_size']), dtype=torch.float32)
        for _ in range(deepstack_count)
    ]
    concat_output_names = [f'out_deepstack_feature_{index}' for index in range(deepstack_count)]
    concat_output_names.append('concat_hidden_states')
    concat_axes = {
        'text_hidden_states': {0: 'batch_size', 1: 'ids_len'},
        'vision_hidden_states': {0: 'batch_size', 1: 'image_token_count'},
        'concat_hidden_states': {0: 'batch_size', 1: 'ids_len'},
    }
    for index in range(deepstack_count):
        concat_axes[f'deepstack_feature_{index}'] = {0: 'batch_size', 1: 'image_token_count'}
        concat_axes[f'out_deepstack_feature_{index}'] = {0: 'batch_size', 1: 'ids_len'}
    concat_image = LLM_CONCAT_IMAGE(image_start, image_end, deepstack_count)
    _export_component(
        export_dir / MODEL_FILE_NAMES['concat_image'],
        concat_image,
        tuple(deepstack_features + [text_hidden_states, image_hidden_states]),
        [f'deepstack_feature_{index}' for index in range(deepstack_count)]
        + ['text_hidden_states', 'vision_hidden_states'],
        concat_output_names,
        concat_axes,
        metadata,
    )
    del concat_image, deepstack_features, text_hidden_states, image_hidden_states
    gc.collect()

    rotary_prefill_axes = {
        'rotary_cos': {1: 'ids_len'},
        'rotary_sin': {1: 'ids_len'},
        'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
    }
    rotary_prefill = ROTARY_IMAGE_PREFILL(
        model, mm_token_type_ids, image_grid_thw.to(torch.int64), MAX_SEQ_LEN
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES['rotary_image_prefill'],
        rotary_prefill,
        (ids_len, history_len),
        ['ids_len', 'history_len'],
        ['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
        rotary_prefill_axes,
        metadata,
    )
    del rotary_prefill
    gc.collect()
    rotary_decode = ROTARY_IMAGE_DECODE(
        model, mm_token_type_ids, image_grid_thw.to(torch.int64), MAX_SEQ_LEN
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES['rotary_image_decode'],
        rotary_decode,
        (kv_seq_len,),
        ['kv_seq_len'],
        ['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
        None,
        metadata,
    )
    del rotary_decode, image_grid_thw
    gc.collect()

    kv_inputs, kv_input_names, kv_output_names, kv_axes = _kv_io(
        kv_specs,
        kv_tensors,
        dimensions['num_layers'],
    )
    hidden_states = torch.ones((1, trace_ids_len, dimensions['hidden_size']), dtype=torch.float32)
    main_deepstack = [
        torch.ones((1, trace_ids_len, dimensions['hidden_size']), dtype=torch.float32)
        for _ in range(deepstack_count)
    ]
    rotary_cos = torch.zeros((1, trace_ids_len, 1, 1, dimensions['head_dim']), dtype=torch.float32)
    rotary_sin = torch.zeros_like(rotary_cos)
    attention_mask = torch.zeros((1, 1, 1, trace_ids_len, trace_ids_len), dtype=torch.float32)
    main_input_names = (
        kv_input_names
        + ['hidden_states']
        + [f'deepstack_feature_{index}' for index in range(deepstack_count)]
        + ['rotary_cos', 'rotary_sin', 'attention_mask']
    )
    main_output_names = kv_output_names + ['logits']
    main_axes = {
        **kv_axes,
        'hidden_states': {0: 'batch_size', 1: 'ids_len'},
        'logits': {0: 'batch_size'},
        'rotary_cos': {1: 'ids_len'},
        'rotary_sin': {1: 'ids_len'},
        'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
    }
    for index in range(deepstack_count):
        main_axes[f'deepstack_feature_{index}'] = {0: 'batch_size', 1: 'ids_len'}
    main = LLM_MAIN(
        model,
        dimensions['num_heads'],
        dimensions['num_kv_heads'],
        dimensions['head_dim'],
        dimensions['num_layers'],
        dimensions['hidden_size'],
        deepstack_count,
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES['main'],
        main,
        tuple(kv_inputs + [hidden_states] + main_deepstack + [rotary_cos, rotary_sin, attention_mask]),
        main_input_names,
        main_output_names,
        main_axes,
        metadata,
    )
    _export_kv_helpers(
        export_dir,
        dimensions,
        kv_specs,
        kv_tensors,
        metadata,
        rope_inv_freq,
        main.quantizer,
    )
    del main, kv_inputs, hidden_states, main_deepstack, rotary_cos, rotary_sin, attention_mask, rope_inv_freq
    gc.collect()

    logits = torch.ones((1, dimensions['vocab_size']), dtype=torch.float32)
    previous_ids = torch.zeros((1, 1), dtype=torch.int32)
    repetition_penalty = torch.ones((1, 1), dtype=torch.float32)
    _export_component(
        export_dir / MODEL_FILE_NAMES['greedy'],
        GREEDY_SEARCH(),
        (logits,),
        ['logits'],
        ['max_logits_idx'],
        {'logits': {0: 'batch_size'}, 'max_logits_idx': {0: 'batch_size'}},
        metadata,
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES['penalty_greedy'],
        PENALTY_GREEDY_SEARCH(),
        (logits, repetition_penalty, previous_ids),
        ['logits', 'repetition_penalty', 'previous_ids'],
        ['max_logits_idx', 'save_id_out'],
        {
            'logits': {0: 'batch_size'},
            'repetition_penalty': {0: 'batch_size'},
            'previous_ids': {0: 'batch_size', 1: 'history_len'},
            'max_logits_idx': {0: 'batch_size'},
            'save_id_out': {0: 'batch_size', 1: 'kv_seq_len'},
        },
        metadata,
    )
    _export_component(
        export_dir / MODEL_FILE_NAMES['sampling'],
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            torch.ones((1,), dtype=torch.float32),
            torch.tensor(min(50, dimensions['vocab_size']), dtype=torch.int64),
            torch.ones((1,), dtype=torch.float32),
            repetition_penalty,
            previous_ids,
        ),
        ['logits', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'previous_ids'],
        ['sampled_id', 'save_id_out'],
        {
            'logits': {0: 'batch_size'},
            'temperature': {0: 'batch_size'},
            'top_p': {0: 'batch_size'},
            'repetition_penalty': {0: 'batch_size'},
            'previous_ids': {0: 'batch_size', 1: 'history_len'},
            'sampled_id': {0: 'batch_size'},
            'save_id_out': {0: 'batch_size', 1: 'kv_seq_len'},
        },
        metadata,
    )

    import Shared_Merged

    bundle = Shared_Merged.build_shared_merged_bundle(
        export_dir,
        model_file_names=MODEL_FILE_NAMES,
        delete_constituents=True,
    )
    for path in bundle['graphs'].values():
        _stamp_metadata(path, metadata)
    _cleanup_unreferenced_data(export_dir)
    tokenizer_assets = copy_tokenizer_assets(download_path, export_dir)
    _promote_export(export_dir)
    print(
        f'FireRedOCR ONNX export completed: {EXPORT_DIR} '
        f'({len(tokenizer_assets)} tokenizer assets).'
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(SCRIPT_DIR) / 'Inference_FireRedOCR_ONNX.py'),
            '--model-folder',
            str(Path(EXPORT_DIR)),
        ],
        check=True,
    )


def export_bundle():
    """Export the FireRedOCR ONNX bundle."""
    return export_firered()


def main():
    if not DO_EXPORT:
        print('DO_EXPORT is False; no ONNX files were written.')
        return
    export_bundle()


if __name__ == '__main__':
    main()


