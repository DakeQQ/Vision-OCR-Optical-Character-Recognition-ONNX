"""Native LFM2.5-VL-450M static image-front-end export components.

The installed legacy ONNX exporter cannot lower SigLIP2's antialiased positional
interpolation.  This module precomputes the checkpoint's exact 32x32 positional
table for the supported 512x512 single-image contract, then retains the native
SigLIP2 encoder and LFM projector mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


STATIC_IMAGE_HEIGHT = 512
STATIC_IMAGE_WIDTH = 512
STATIC_PATCH_SIZE = 16
STATIC_PATCH_GRID = STATIC_IMAGE_HEIGHT // STATIC_PATCH_SIZE
STATIC_PATCH_COUNT = STATIC_PATCH_GRID * STATIC_PATCH_GRID


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
        raise ValueError("Vision channel permutation has an unexpected shape.")
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Vision channel permutation is not bijective.")


@dataclass(frozen=True)
class VisionReorderSummary:
    pairs: int
    maximum_error: float


def _reorder_pair(first, second, activation, key) -> float:
    """Permute one elementwise MLP producer/consumer pair exactly once."""
    width = int(second.in_features)
    if int(first.out_features) != width:
        raise ValueError("Vision MLP producer width does not match consumer input width.")
    permutation = torch.argsort(_channel_statistic(second.weight.detach(), key))
    _validate_permutation(permutation, width)
    probe = torch.randn((2, int(first.in_features)), dtype=first.weight.dtype)
    before = second(activation(first(probe)))
    reordered_weight = first.weight.detach()[permutation].clone()
    reordered_bias = first.bias.detach()[permutation].clone() if first.bias is not None else None
    reordered_consumer = second.weight.detach()[:, permutation].clone()
    after = torch.nn.functional.linear(
        activation(torch.nn.functional.linear(probe, reordered_weight, reordered_bias)),
        reordered_consumer,
        second.bias,
    )
    error = float((before - after).abs().max())
    # Permuting a wide GEMM changes FP32 accumulation order while preserving
    # the algebra exactly. The observed projector bound is below 1e-4.
    if error > 1e-4:
        raise RuntimeError(f"Vision MLP reordering is not equivalent (max error {error}).")
    first.weight.data.copy_(reordered_weight)
    if first.bias is not None:
        first.bias.data.copy_(reordered_bias)
    second.weight.data.copy_(reordered_consumer)
    return error


def reorder_vision_mlp_pairs(model, key: str, enabled: bool) -> VisionReorderSummary:
    """Reorder only proven elementwise SigLIP2 and projector MLP pairs."""
    if not enabled:
        return VisionReorderSummary(0, 0.0)
    maximum_error = 0.0
    pairs = 0
    with torch.no_grad():
        vision = model.model.vision_tower
        for layer in vision.encoder.layers:
            maximum_error = max(
                maximum_error,
                _reorder_pair(layer.mlp.fc1, layer.mlp.fc2, layer.mlp.activation_fn, key),
            )
            pairs += 1
        projector = model.model.multi_modal_projector
        maximum_error = max(
            maximum_error,
            _reorder_pair(projector.linear_1, projector.linear_2, projector.act, key),
        )
        pairs += 1
    return VisionReorderSummary(pairs, maximum_error)


class LFM_IMAGE_PREPROCESS(torch.nn.Module):
    """Convert a static 512x512 RGB uint8 image into SigLIP2 patch vectors."""

    def __init__(self):
        super().__init__()
        # TorchvisionBackend fuses rescale 1/255 and normalization 0.5/0.5
        # into normalize(raw_uint8, mean=127.5, std=127.5).
        self.register_buffer("image_mean", torch.full((1, 3, 1, 1), 127.5, dtype=torch.float32))
        self.register_buffer("image_std", torch.full((1, 3, 1, 1), 127.5, dtype=torch.float32))

    def forward(self, pixel_values):
        values = (pixel_values.float() - self.image_mean) / self.image_std
        values = values.reshape(
            1,
            3,
            STATIC_PATCH_GRID,
            STATIC_PATCH_SIZE,
            STATIC_PATCH_GRID,
            STATIC_PATCH_SIZE,
        )
        values = values.permute(0, 2, 4, 3, 5, 1)
        return values.reshape(1, STATIC_PATCH_COUNT, 3 * STATIC_PATCH_SIZE * STATIC_PATCH_SIZE)


class LFM_VISION(torch.nn.Module):
    """Run the native static SigLIP2 encoder and LFM multimodal projector."""

    def __init__(self, model):
        super().__init__()
        vision = model.model.vision_tower
        if int(vision.config.patch_size) != STATIC_PATCH_SIZE:
            raise ValueError("This exporter supports only the checkpoint's 16px SigLIP2 patches.")
        self.patch_embedding = vision.embeddings.patch_embedding
        self.encoder = vision.encoder
        self.post_layernorm = vision.post_layernorm
        self.projector = model.model.multi_modal_projector
        with torch.no_grad():
            source = vision.embeddings.position_embedding.weight.reshape(
                vision.embeddings.position_embedding_size,
                vision.embeddings.position_embedding_size,
                -1,
            )
            static_shape = torch.tensor(
                [[STATIC_PATCH_GRID, STATIC_PATCH_GRID]], dtype=torch.int64
            )
            table = vision.embeddings.resize_positional_embeddings(
                source,
                static_shape,
                STATIC_PATCH_COUNT,
            )
        self.register_buffer("position_embeddings", table, persistent=False)

    def forward(self, patches):
        hidden_states = self.patch_embedding(patches.float()) + self.position_embeddings
        hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
        ).last_hidden_state
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(
            1,
            STATIC_PATCH_GRID,
            STATIC_PATCH_GRID,
            -1,
        )
        hidden_states = self.projector(hidden_states)
        return hidden_states.reshape(1, -1, hidden_states.shape[-1])


class LFM_CONCAT_IMAGE(torch.nn.Module):
    """Replace native <image> token embeddings with projected visual features."""

    def __init__(self, image_token_id: int, image_token_count: int):
        super().__init__()
        self.image_token_id = int(image_token_id)
        self.image_token_count = int(image_token_count)

    def forward(self, input_ids, text_hidden_states, vision_hidden_states):
        image_mask = input_ids == self.image_token_id
        # The exporter and standalone runtime validate this 256-token contract.
        return text_hidden_states.masked_scatter(image_mask.unsqueeze(-1), vision_hidden_states)
