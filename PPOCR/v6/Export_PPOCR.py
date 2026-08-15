import gc
import os
import shutil
import subprocess
import sys

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'PPOCRv6_ONNX')

# --- Local model directories (this workspace layout) -----------------------------
# Recognition is the SEPARATE top-level workspace folder, not under PPOCR/v6/models.
# PPOCR URL: https://www.paddleocr.ai/main/version3.x/pipeline_usage/OCR.html
_WS_ROOT           = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODELS_DIR         = os.path.join(SCRIPT_DIR, 'models')
DOC_ORI_MODEL_DIR  = os.path.join(MODELS_DIR, 'PP-LCNet_x1_0_doc_ori_safetensors')
UNWARP_MODEL_DIR   = os.path.join(MODELS_DIR, 'UVDoc_safetensors')
DET_MODEL_DIR      = os.path.join(MODELS_DIR, 'PP-OCRv6_medium_det_safetensors')
TEXTLINE_MODEL_DIR = os.path.join(MODELS_DIR, 'PP-LCNet_x1_0_textline_ori_safetensors')
REC_MODEL_DIR      = os.path.join(_WS_ROOT,   'PP-OCRv6_medium_rec_safetensors')

# --- Official sub-model auto-download (mirrors PPOCR/v6/demo.py BOS mechanism) ----
# Rec is bundled locally; the other sub-models auto-download once from the BOS mirror
# into MODELS_DIR. Each entry maps the PaddleX name to its local *_safetensors dir.
MODEL_SOURCE               = 'BOS'        # Official PaddleOCR model mirror.
DISABLE_MODEL_SOURCE_CHECK = True         # Skip the model-hoster connectivity probe.
PADDLEX_CACHE_DIR          = os.path.join(SCRIPT_DIR, '.paddlex_cache')
LEGACY_OFFICIAL_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.paddlex', 'official_models')
OFFICIAL_SUBMODELS = (
    ('PP-LCNet_x1_0_doc_ori',      DOC_ORI_MODEL_DIR),
    ('UVDoc',                      UNWARP_MODEL_DIR),
    ('PP-OCRv6_medium_det',        DET_MODEL_DIR),
    ('PP-LCNet_x1_0_textline_ori', TEXTLINE_MODEL_DIR),
)

# --- ONNX output paths -----------------------------------------------------------
onnx_model_DocOri = os.path.join(EXPORT_DIR, 'PPOCRv6_DocOri.onnx')
onnx_model_Unwarp = os.path.join(EXPORT_DIR, 'PPOCRv6_Unwarp.onnx')
onnx_model_Det    = os.path.join(EXPORT_DIR, 'PPOCRv6_Det.onnx')
onnx_model_Rec    = os.path.join(EXPORT_DIR, 'PPOCRv6_Rec.onnx')
onnx_model_DBPost = os.path.join(EXPORT_DIR, 'PPOCRv6_DBPost.onnx')

# --- ONNX export -----------------------------------------------------------------
OPSET = 20                              # >=20 for GridSample and modern Resize behavior.

# --- Detection geometry ----------------------------------------------------------
DET_LIMIT_SIDE_LEN = 736
DET_MAX_SIDE_LIMIT = 4000
DET_SIDE_MULTIPLE  = 32

# --- DB postprocess (from PP-OCRv6_medium_det inference.yml) ----------------------
DB_THRESH       = 0.2
DB_BOX_THRESH   = 0.45
DB_UNCLIP_RATIO = 1.4
DB_MIN_SIZE     = 3
DB_CC_MAX_ITERS = 4096                    # Connected-component max-propagation cap.
DB_CC_UNROLL    = 4                       # Exact 3x3 max-propagation steps per Loop trip.

# --- Recognition geometry --------------------------------------------------------
REC_HEIGHT      = 48
REC_MAX_WIDTH   = 3200
REC_WIDTH_ALIGN = 32                      # Round each batch crop width up to this multiple.
REC_PAD_VALUE   = 128                     # Gray (~normalised 0), matches the rec processor's pad.

# --- Classifier input geometry ---------------------------------------------------
DOC_ORI_RESIZE_SHORT = 256
DOC_ORI_CROP         = 224
TEXTLINE_SIZE        = (80, 160)          # (height, width).

# --- Empirically-traced per-stage preprocessing (pv[c] = scale[c]*x_bgr[c]+base[c])
# Channels are BGR-ordered at the model input; x_bgr is the uint8 (0-255) value.
PRE_CLS  = {'scale': [0.0171204, 0.0175121, 0.0174301], 'base': [-2.117904, -2.035714, -1.804444]}
PRE_DET  = PRE_CLS
PRE_REC  = {'scale': [1.0 / 127.5] * 3, 'base': [-1.0] * 3}
RGB_TO_BGR = [2, 1, 0]                    # Input-channel reorder applied in-graph at each model feed.


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT-FOLDING HELPERS  (Conv+BN fusion, preprocessing fold, classifier scale)
# ══════════════════════════════════════════════════════════════════════════════
def fuse_conv_bn_recursive(module):
    """Fold every Conv2d/ConvTranspose2d + BatchNorm2d pair into a single conv.

    A fold target is any submodule with a conv attribute (``convolution``/``conv``)
    paired with a BatchNorm attribute (``normalization``/``norm``): PP-LCNet(/V4),
    the PP-OCRv6 det head and UVDoc ``*ConvLayer`` blocks.
    """
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for _, child in list(module.named_children()):
        conv = bn = conv_attr = bn_attr = None
        for attr in ('convolution', 'conv'):
            candidate = getattr(child, attr, None)
            if isinstance(candidate, (nn.Conv2d, nn.ConvTranspose2d)):
                conv, conv_attr = candidate, attr
        for attr in ('normalization', 'norm'):
            candidate = getattr(child, attr, None)
            if isinstance(candidate, nn.BatchNorm2d):
                bn, bn_attr = candidate, attr
        if conv is not None and bn is not None:
            # fuse_conv_bn_eval does NOT auto-detect transposed convs; the det head's
            # ConvTranspose2d needs transpose=True or BN folds on the wrong axis
            # (silently wrong when in_channels == out_channels).
            setattr(child, conv_attr, fuse_conv_bn_eval(conv, bn, transpose=conv.transposed))
            setattr(child, bn_attr, nn.Identity())
        else:
            fuse_conv_bn_recursive(child)
    return module


def fold_classifier_dropout_scale(model):
    """Fold the PP-LCNet inference-time ``*(1 - hidden_dropout_prob)`` quirk.

    The constant scale folds exactly into ``head.weight``; the runtime multiply is
    then neutralised by setting the probability to zero.
    """
    p = float(getattr(model, 'hidden_dropout_prob', 0.0) or 0.0)
    if p and hasattr(model, 'head') and hasattr(model.head, 'weight'):
        model.head.weight.data = model.head.weight.data * (1.0 - p)
        model.hidden_dropout_prob = 0.0
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT WRAPPERS  (one torch.nn.Module per stage, fully ONNX-optimised forward)
# ══════════════════════════════════════════════════════════════════════════════
def _build_wrappers():
    """Import torch lazily and build the fused / folded export wrappers."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    @torch.jit.script
    def _rotate_upright(image: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        # Re-orient a (1, 3, H, W) image upright from the doc-orientation class
        # (0/1/2/3 -> 0/90/180/270 deg). Scripted so the exporter emits a real
        # data-dependent If (90/270 swap H<->W) instead of baking the dummy branch.
        out = image
        if bool(class_id == 1):
            out = image.transpose(2, 3).flip([2])
        elif bool(class_id == 2):
            out = image.flip([2, 3])
        elif bool(class_id == 3):
            out = image.transpose(2, 3).flip([3])
        return out

    class FoldedPreprocessStem(nn.Module):
        """Encoder stem with uint8->normalised BGR preprocessing folded inside.

        Scale folds into the first conv weights.  Fixed-size classifiers fold the
        additive base into a constant output map; dynamic-size stages keep it as a
        raw-input offset before the scaled conv so padded borders stay exact.
        """

        def __init__(self, stem, pre, input_hw=None):
            super().__init__()
            self.convolution = _stem_conv(stem)
            self.normalization = getattr(stem, 'normalization', nn.Identity())
            self.activation = getattr(stem, 'activation', nn.Identity())
            self.lab = getattr(stem, 'lab', nn.Identity())

            if self.convolution is None:
                raise ValueError('preprocess folding expects a stem layer with a Conv2d')
            if self.convolution.groups != 1 or self.convolution.in_channels != len(pre['scale']):
                raise ValueError('preprocess folding expects an ungrouped 3-channel stem convolution')

            weight = self.convolution.weight.detach().clone()
            weight_dtype = weight.dtype
            weight_device = weight.device
            scale = torch.tensor(pre['scale'], dtype=weight_dtype, device=weight_device).view(1, -1, 1, 1)
            base = torch.tensor(pre['base'], dtype=weight_dtype, device=weight_device).view(1, -1, 1, 1)

            with torch.no_grad():
                self.convolution.weight.data = weight * scale

            if input_hw is None:
                if torch.any(scale == 0):
                    raise ValueError('preprocess scale must be non-zero for dynamic stem folding')
                self.register_buffer('input_offset', base / scale)
                self.register_buffer('base_map', None)
            else:
                input_height, input_width = input_hw
                with torch.no_grad():
                    base_image = base.expand(1, -1, input_height, input_width)
                    base_map = F.conv2d(
                        base_image, weight, bias=None, stride=self.convolution.stride,
                        padding=self.convolution.padding, dilation=self.convolution.dilation,
                        groups=self.convolution.groups)
                self.register_buffer('input_offset', None)
                self.register_buffer('base_map', base_map)

        def forward(self, image_bgr_u8):
            image = image_bgr_u8.float()
            if self.input_offset is not None:
                image = image + self.input_offset
            hidden_state = self.convolution(image)
            if self.base_map is not None:
                hidden_state = hidden_state + self.base_map
            hidden_state = self.normalization(hidden_state)
            hidden_state = self.activation(hidden_state)
            hidden_state = self.lab(hidden_state)
            return hidden_state

    def _stem_conv(stem):
        conv = getattr(stem, 'convolution', None)
        if isinstance(conv, nn.Conv2d):
            return conv
        conv = getattr(stem, 'conv', None)
        if isinstance(conv, nn.Conv2d):
            return conv
        return None

    def fold_preprocess_into_encoder_stem(module, pre, input_hw=None):
        """Replace the first 3-channel stem layer with a folded preprocess stem."""
        root = module.encoder if hasattr(module, 'encoder') else module
        stem = getattr(root, 'convolution', None)
        if stem is None:
            raise ValueError('could not find encoder convolution stem for preprocess folding')

        if _stem_conv(stem) is not None:
            root.convolution = FoldedPreprocessStem(stem, pre, input_hw)
            return

        for name, child in stem.named_children():
            conv = _stem_conv(child)
            if conv is not None and conv.in_channels == len(pre['scale']) and conv.groups == 1:
                setattr(stem, name, FoldedPreprocessStem(child, pre, input_hw))
                return

        raise ValueError('could not find an ungrouped 3-channel stem convolution for preprocess folding')

    def fold_svtr_attention_scale(model):
        """Fold each SVTR attention scale into the query slice of fused QKV."""
        for block in model.head.encoder.svtr_block:
            attention = block.self_attn
            scale = float(getattr(attention, 'scale', 1.0) or 1.0)
            if scale == 1.0:
                continue
            query_width = attention.num_heads * attention.head_dim
            with torch.no_grad():
                attention.qkv.weight[:query_width].mul_(scale)
                if attention.qkv.bias is not None:
                    attention.qkv.bias[:query_width].mul_(scale)
            attention.scale = 1.0
        return model

    def _fold_layer_norm_affine_into_linear(norm, linear):
        """Absorb LayerNorm gamma/beta into a directly-following Linear layer."""
        if not isinstance(norm, nn.LayerNorm):
            raise ValueError('SVTR norm affine fold expects a LayerNorm')
        if not norm.elementwise_affine or norm.weight is None:
            return norm
        if linear.weight.shape[1] != norm.weight.numel():
            raise ValueError('LayerNorm width does not match Linear input width')

        weight = linear.weight.detach().clone()
        gamma = norm.weight.detach().to(dtype=weight.dtype, device=weight.device)
        if norm.bias is None:
            beta = torch.zeros_like(gamma)
        else:
            beta = norm.bias.detach().to(dtype=weight.dtype, device=weight.device)

        with torch.no_grad():
            bias_offset = torch.matmul(weight, beta)
            linear.weight.mul_(gamma.unsqueeze(0))
            if linear.bias is None:
                linear.bias = nn.Parameter(bias_offset)
            else:
                linear.bias.add_(bias_offset)

        return nn.LayerNorm(
            norm.normalized_shape, eps=norm.eps, elementwise_affine=False,
            device=weight.device, dtype=weight.dtype)

    def fold_svtr_layer_norm_affines(model):
        """Fold SVTR pre-projection LayerNorm affine terms into QKV/MLP weights."""
        for block in model.head.encoder.svtr_block:
            block.layer_norm1 = _fold_layer_norm_affine_into_linear(block.layer_norm1, block.self_attn.qkv)
            block.layer_norm2 = _fold_layer_norm_affine_into_linear(block.layer_norm2, block.mlp.fc1)
        return model

    def fold_rec_final_norm_bias_into_head(model):
        """Absorb final SVTR LayerNorm beta through the residual add into the CTC head."""
        norm = model.head.encoder.norm
        linear = model.head.head
        if not isinstance(norm, nn.LayerNorm) or norm.bias is None:
            return model
        if linear.weight.shape[1] != norm.bias.numel():
            raise ValueError('final LayerNorm width does not match recognition head input width')

        weight = linear.weight.detach()
        beta = norm.bias.detach().to(dtype=weight.dtype, device=weight.device)
        with torch.no_grad():
            bias_offset = torch.matmul(weight, beta)
            if linear.bias is None:
                linear.bias = nn.Parameter(bias_offset)
            else:
                linear.bias.add_(bias_offset)
            norm.bias = None
        return model

    def fold_rec_avg_pool_into_head(model, pool_kernel=(3, 2)):
        """Fold the pre-head AvgPool2d into the two parallel 1x1 head convs."""
        for layer in model.head.encoder.conv_block[:2]:
            conv = layer.convolution
            if (conv.kernel_size != (1, 1) or conv.stride != (1, 1) or conv.padding != (0, 0)
                    or conv.dilation != (1, 1)):
                raise ValueError('recognition avg-pool fold expects 1x1 stride-1 convs')

            folded = nn.Conv2d(
                conv.in_channels, conv.out_channels, pool_kernel,
                stride=pool_kernel, padding=0, dilation=1, groups=conv.groups,
                bias=conv.bias is not None, padding_mode=conv.padding_mode,
                device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                folded.weight.copy_(conv.weight.expand(-1, -1, pool_kernel[0], pool_kernel[1]))
                folded.weight.div_(float(pool_kernel[0] * pool_kernel[1]))
                if conv.bias is not None:
                    folded.bias.copy_(conv.bias)
            layer.convolution = folded
        return model

    class ClsExport(nn.Module):
        """PP-LCNet orientation classifier: uint8 BGR -> (int8 class_id, score).

        Conv+BN fused, dropout-scale + preprocessing folded into the stem; argmax
        and top-1 softmax score are computed in-graph.
        """

        def __init__(self, model, pre, input_hw):
            super().__init__()
            fold_classifier_dropout_scale(model)
            self.model = model
            fold_preprocess_into_encoder_stem(self.model.encoder, pre, input_hw)

        def forward(self, image_bgr_u8):                   # (1, 3, H, W) uint8 BGR
            hidden_state = self.model.encoder.convolution(image_bgr_u8)
            for block in self.model.encoder.blocks:
                hidden_state = block(hidden_state)
            logits = self.model.avg_pool(hidden_state)
            logits = self.model.last_convolution(logits)
            logits = self.model.act_fn(logits)
            logits = self.model.flatten(logits)
            logits = self.model.head(logits)
            probs = logits.softmax(-1)
            score, idx = probs.max(-1)
            return idx.to(torch.int8), score

    class DocOriExport(nn.Module):
        """Fused doc-orientation preprocess + PP-LCNet classifier + in-graph rotate.

        Takes the full-image uint8 RGB tensor; computes resize-short(256),
        centre-crop(224), RGB->BGR and the classifier, then re-orients the FULL
        image upright via a data-dependent ONNX If.  The orientation class never
        crosses to the host.  Outputs the upright image + angle (0/90/180/270).
        """

        def __init__(self, model, pre):
            super().__init__()
            self.classifier = ClsExport(model, pre, (DOC_ORI_CROP, DOC_ORI_CROP))

        def forward(self, image_nchw):                  # (1, 3, H, W) uint8 RGB
            shape = torch._shape_as_tensor(image_nchw)
            hw = shape[2:4].to(torch.float32)
            scale = float(DOC_ORI_RESIZE_SHORT) / torch.min(hw)
            resized_hw = torch.round(hw * scale).to(torch.int64)

            image = image_nchw.float()
            resized = F.interpolate(
                image, size=(resized_hw[0], resized_hw[1]),
                mode='bilinear', align_corners=False)

            start_hw = ((resized_hw - DOC_ORI_CROP) // 2).to(torch.int64)
            cropped = resized[:, :,
                              start_hw[0]:start_hw[0] + DOC_ORI_CROP,
                              start_hw[1]:start_hw[1] + DOC_ORI_CROP]
            cropped_bgr = cropped.round().clamp(0, 255).to(torch.uint8)[:, RGB_TO_BGR]
            class_id, _ = self.classifier(cropped_bgr)
            class_id = class_id.reshape(()).to(torch.int64)
            image_oriented = _rotate_upright(image_nchw, class_id)
            angle = (class_id * 90).to(torch.int64)
            return image_oriented, angle

    class DetExport(nn.Module):
        """PP-OCRv6 detection: resized uint8 BGR -> probability map."""

        def __init__(self, model, pre):
            super().__init__()
            self.model = model
            fold_preprocess_into_encoder_stem(self.model.model.backbone, pre)

        def forward(self, image_bgr_u8):                   # (1, 3, H, W) uint8 BGR
            backbone_outputs = self.model.model.backbone(image_bgr_u8)
            hidden_state = self.model.model.neck(backbone_outputs.feature_maps)
            return self.model.head(hidden_state)

    class DetFullExport(nn.Module):
        """Fused detection resize + PP-OCRv6 detector.

        Applies the native PP-OCR resize policy and RGB->BGR reorder, then runs the
        detector.  ``source_hw`` is returned for DBPost scaling.
        """

        def __init__(self, model, pre):
            super().__init__()
            self.detector = DetExport(model, pre)

        def forward(self, image_nchw):                  # (1, 3, H, W) uint8 RGB
            shape = torch._shape_as_tensor(image_nchw)
            source_hw = shape[2:4]
            hw = source_hw.to(torch.float32)

            min_hw = torch.min(hw)
            ratio0 = torch.where(
                min_hw < float(DET_LIMIT_SIDE_LEN),
                float(DET_LIMIT_SIDE_LEN) / min_hw,
                torch.ones_like(min_hw))
            resized_hw = hw * ratio0

            max_resized = torch.max(resized_hw)
            ratio1 = torch.where(
                max_resized > float(DET_MAX_SIDE_LIMIT),
                float(DET_MAX_SIDE_LIMIT) / max_resized,
                torch.ones_like(max_resized))
            resized_hw = resized_hw * ratio1
            resized_hw = torch.round(resized_hw / float(DET_SIDE_MULTIPLE)) * float(DET_SIDE_MULTIPLE)
            resized_hw_i64 = resized_hw.to(torch.int64)
            min_side = torch.full_like(resized_hw_i64, DET_SIDE_MULTIPLE)
            resized_hw_i64 = torch.maximum(resized_hw_i64, min_side)

            # CUDA Resize corrupts this dynamic bilinear path for uint8 tensors.
            resized_rgb = F.interpolate(
                image_nchw.float(), size=(resized_hw_i64[0], resized_hw_i64[1]),
                mode='bilinear', align_corners=False)
            resized_bgr = resized_rgb[:, RGB_TO_BGR]
            prob_map = self.detector(resized_bgr)
            return prob_map, source_hw

    class RecExport(nn.Module):
        """PP-OCRv6 recognition: uint8 RGB image + boxes -> CTC-decoded ids + score.

        The per-box crop (``RecCropExport``) is fused in, so the crop tensor stays
        inside the graph -- no separate crop session.  Greedy CTC decode (argmax,
        repeat collapse, blank removal, mean confidence) runs in-graph; the host
        only maps the decoded ids through ``rec_char_list``.
        """

        def __init__(self, model, pre):
            super().__init__()
            self.crop = RecCropExport()
            self.model = model
            fold_preprocess_into_encoder_stem(self.model.model.backbone, pre)
            fold_rec_avg_pool_into_head(self.model)
            fold_svtr_layer_norm_affines(self.model)
            fold_svtr_attention_scale(self.model)
            fold_rec_final_norm_bias_into_head(self.model)

        def _forward_conv_layer(self, layer, hidden_state):
            if hasattr(layer, 'input_offset'):
                hidden_state = hidden_state.float()
                if layer.input_offset is not None:
                    hidden_state = hidden_state + layer.input_offset
            hidden_state = layer.convolution(hidden_state)
            if hasattr(layer, 'base_map') and layer.base_map is not None:
                hidden_state = hidden_state + layer.base_map
            hidden_state = layer.normalization(hidden_state)
            hidden_state = layer.activation(hidden_state)
            lab = getattr(layer, 'lab', None)
            if lab is not None:
                hidden_state = lab(hidden_state)
            return hidden_state

        def _forward_token_conv(self, layer, hidden_state):
            if hasattr(layer, 'convolution') and hasattr(layer, 'normalization'):
                return self._forward_conv_layer(layer, hidden_state)
            return layer(hidden_state)

        def _forward_squeeze_excitation(self, layer, hidden_state):
            if not hasattr(layer, 'avg_pool'):
                return layer(hidden_state)
            residual = hidden_state
            hidden_state = layer.avg_pool(hidden_state)
            for sublayer in layer.convolutions:
                hidden_state = sublayer(hidden_state)
            return residual * hidden_state

        def _forward_lcnet_depthwise_layer(self, layer, hidden_state):
            hidden_state = self._forward_token_conv(layer.token_conv, hidden_state)
            hidden_state = self._forward_squeeze_excitation(layer.token_squeeze_excitation, hidden_state)
            residual = hidden_state

            hidden_state = self._forward_conv_layer(layer.channel_conv1, hidden_state)
            hidden_state = layer.channel_act_fn(hidden_state)
            hidden_state = self._forward_conv_layer(layer.channel_conv2, hidden_state)

            if layer.has_residual:
                hidden_state = residual + hidden_state
            return hidden_state

        def _forward_backbone_stem(self, image_bgr_u8):
            stem = self.model.model.backbone.encoder.convolution
            if hasattr(stem, 'stem1'):
                embedding = self._forward_conv_layer(stem.stem1, image_bgr_u8)
                embedding = F.pad(embedding, (0, 1, 0, 1))

                branch_embedding = self._forward_conv_layer(stem.stem2a, embedding)
                branch_embedding = F.pad(branch_embedding, (0, 1, 0, 1))
                branch_embedding = self._forward_conv_layer(stem.stem2b, branch_embedding)

                pooled_embedding = stem.pool(embedding)
                embedding = torch.cat([pooled_embedding, branch_embedding], dim=1)
                embedding = self._forward_conv_layer(stem.stem3, embedding)
                embedding = self._forward_conv_layer(stem.stem4, embedding)
                return embedding

            hidden_state = self._forward_conv_layer(stem.conv1, image_bgr_u8)
            hidden_state = stem.act_fn(hidden_state)
            hidden_state = self._forward_conv_layer(stem.conv2, hidden_state)
            return hidden_state

        def _forward_backbone(self, image_bgr_u8):
            hidden_state = self._forward_backbone_stem(image_bgr_u8)
            for stage in self.model.model.backbone.encoder.blocks:
                for layer in stage.blocks:
                    hidden_state = self._forward_lcnet_depthwise_layer(layer, hidden_state)
            return hidden_state

        def _forward_svtr_attention(self, attention, hidden_state, batch_size):
            mixed_qkv = attention.qkv(hidden_state)
            mixed_qkv = mixed_qkv.reshape(batch_size, -1, 3, attention.num_heads, attention.head_dim)
            mixed_qkv = mixed_qkv.permute(2, 0, 3, 1, 4)
            query_state, key_state, value_state = mixed_qkv.split(1, dim=0)

            attn_weights = torch.matmul(query_state, key_state.transpose(-1, -2))
            attn_weights = F.softmax(attn_weights, dim=-1)

            hidden_state = torch.matmul(attn_weights, value_state).transpose(2, 3)
            hidden_state = hidden_state.reshape(batch_size, -1, attention.projection.in_features).contiguous()
            hidden_state = attention.projection(hidden_state)
            return hidden_state

        def _forward_svtr_mlp(self, mlp, hidden_state):
            hidden_state = mlp.fc1(hidden_state)
            hidden_state = mlp.activation(hidden_state)
            hidden_state = mlp.fc2(hidden_state)
            return hidden_state

        def _forward_svtr_block(self, block, hidden_state, batch_size):
            residual = hidden_state
            hidden_state = block.layer_norm1(hidden_state)
            hidden_state = self._forward_svtr_attention(block.self_attn, hidden_state, batch_size)
            hidden_state = residual + hidden_state

            residual = hidden_state
            hidden_state = block.layer_norm2(hidden_state)
            hidden_state = self._forward_svtr_mlp(block.mlp, hidden_state)
            hidden_state = residual + hidden_state
            return hidden_state

        def _forward_rec_head(self, hidden_state):
            rec_head = self.model.head
            encoder = rec_head.encoder
            batch_size = hidden_state.shape[0]

            residual = self._forward_conv_layer(encoder.conv_block[0], hidden_state)
            hidden_state = self._forward_conv_layer(encoder.conv_block[1], hidden_state)
            hidden_state = hidden_state + self._forward_conv_layer(encoder.conv_block[2], hidden_state)

            # Index the single backbone row (H collapses to 1) instead of squeeze(2):
            # torch.jit.trace renders squeeze(dim) as a data-dependent If of unknown
            # rank, which breaks the following transpose in the traced fused graph.
            residual = residual[:, :, 0].transpose(1, 2)
            hidden_state = hidden_state[:, :, 0].transpose(1, 2)
            for block in encoder.svtr_block:
                hidden_state = self._forward_svtr_block(block, hidden_state, batch_size)

            hidden_state = encoder.norm(hidden_state)
            hidden_state = hidden_state + residual

            logits = rec_head.head(hidden_state)
            probs = logits.softmax(dim=-1)
            return probs

        def forward(self, image_nchw, polys, flip_flags, target_widths):  # RGB image + boxes
            image_bgr_u8 = self.crop(image_nchw, polys, flip_flags, target_widths)
            hidden_state = self._forward_backbone(image_bgr_u8)
            probs = self._forward_rec_head(hidden_state)
            score, idx = probs.max(-1)                      # (B, T)

            first = torch.ones((idx.shape[0], 1), dtype=torch.bool, device=idx.device)
            not_repeat = torch.cat([first, idx[:, 1:] != idx[:, :-1]], dim=1)
            keep = (idx != 0) & not_repeat                  # drop CTC blank(0)

            keep_i32 = keep.to(torch.int32)
            decoded_lengths = keep_i32.sum(dim=1)           # (B,)
            compact_pos = torch.cumsum(keep_i32, dim=1) - 1 # (B, T), kept chars -> 0..L-1
            out_pos = torch.arange(idx.shape[1], dtype=torch.int32, device=idx.device)
            out_pos = out_pos.view(1, 1, -1)                # (1, 1, T)

            compact_match = compact_pos.unsqueeze(-1) == out_pos
            compact_match = compact_match & keep.unsqueeze(-1)
            decoded_ids = (idx.to(torch.int32).unsqueeze(-1) * compact_match.to(torch.int32)).sum(dim=1)

            keep_f = keep.to(score.dtype)
            selected_count = keep_f.sum(dim=1)
            text_scores = (score * keep_f).sum(dim=1) / torch.clamp(selected_count, min=1.0)
            return decoded_ids, decoded_lengths.to(torch.int32), text_scores

    class UnwarpExport(nn.Module):
        """UVDoc unwarp: uint8 RGB -> rectified uint8 RGB.

        Conv+BN fused.  RGB->BGR is folded into the graph (channel-axis flip) and
        the 1/255 rescale into the stem conv, so the BGR tensor stays in 0-255
        space for both the mesh predictor and the grid_sample source.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model
            # 1/255 rescale folds into the stem conv: a uniform scalar commutes
            # through bilinear interpolation and the zero-padded conv, so scaling
            # only the weight (never the BN-folded bias) is exact.
            stem_conv = model.backbone.resnet.resnet_head[0].convolution
            if not isinstance(stem_conv, nn.Conv2d):
                raise TypeError('UVDoc backbone stem is not a Conv2d; cannot fold rescale')
            with torch.no_grad():
                stem_conv.weight.mul_(1.0 / 255.0)

        def forward(self, image_rgb_u8):                   # (1, 3, H, W) uint8 RGB
            bgr = image_rgb_u8.float().flip(dims=[1])           # RGB -> BGR (rescale folded into stem conv)
            model_in = F.interpolate(bgr, size=(712, 488), mode='bilinear', align_corners=True)
            backbone_outputs = self.model.backbone(pixel_values=model_in)
            fused_outputs = torch.cat(backbone_outputs.feature_maps, dim=1)
            mesh = self.model.head(fused_outputs)          # (1, 2, gh, gw)
            height = image_rgb_u8.shape[2]
            width = image_rgb_u8.shape[3]
            mesh_up = F.interpolate(mesh, size=(height, width), mode='bilinear', align_corners=True)
            mesh_up = mesh_up.permute(0, 2, 3, 1)                                # (1, H, W, 2)
            warped = F.grid_sample(bgr, mesh_up, align_corners=True)            # (1, 3, H, W) BGR (0-255)
            out = warped.flip(dims=[1])                                          # BGR -> RGB (0-255)
            return out.round().clamp(0, 255).to(torch.uint8)

    class TextlineCropExport(nn.Module):
        """Perspective-ish textline crop to the fixed orientation classifier size.

        The four bilinear corner weights are pre-computed as buffers (static grid).
        The per-axis [-1, 1] normalisation is folded into one (sx, sy) scale-shift,
        and 1/255 is dropped since grid_sample is linear in the source pixels.
        """

        def __init__(self):
            super().__init__()
            crop_h, crop_w = TEXTLINE_SIZE
            ys = torch.linspace(0.0, 1.0, crop_h, dtype=torch.float32)
            xs = torch.linspace(0.0, 1.0, crop_w, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid_x = grid_x.reshape(1, crop_h, crop_w, 1)
            grid_y = grid_y.reshape(1, crop_h, crop_w, 1)
            one_minus_grid_x = 1.0 - grid_x
            one_minus_grid_y = 1.0 - grid_y
            # Pre-computed bilinear corner weights (tl, tr, br, bl) for p0..p3.
            self.register_buffer('weight_tl', one_minus_grid_x * one_minus_grid_y)
            self.register_buffer('weight_tr', grid_x * one_minus_grid_y)
            self.register_buffer('weight_br', grid_x * grid_y)
            self.register_buffer('weight_bl', one_minus_grid_x * grid_y)

        def forward(self, image_nchw, polys):           # RGB image, polys (N, 4, 2)
            image = image_nchw.float()
            height = torch._shape_as_tensor(image_nchw)[2].to(torch.float32)
            width = torch._shape_as_tensor(image_nchw)[3].to(torch.float32)

            polys = polys.view(-1, 4, 1, 2)
            p0, p1, p2, p3 = polys.split(1, dim=1)
            points = (p0 * self.weight_tl + p1 * self.weight_tr
                      + p2 * self.weight_br + p3 * self.weight_bl)
            inv_scale = torch.stack([
                2.0 / torch.clamp(width - 1.0, min=1.0),
                2.0 / torch.clamp(height - 1.0, min=1.0)])
            grid = points * inv_scale - 1.0

            batch_image = image.expand(polys.shape[0], -1, -1, -1)
            crops = F.grid_sample(batch_image, grid, align_corners=True, padding_mode='border')
            return crops.round().clamp(0, 255).to(torch.uint8)[:, RGB_TO_BGR]

    class RecCropExport(nn.Module):
        """Crop text boxes to a per-batch dynamic recognition width.

        The output width is the batch's max target width rounded up to
        ``REC_WIDTH_ALIGN`` (clamped to ``REC_MAX_WIDTH``), so each batch pads only
        to its own widest crop.  The text region (cols ``< target_width``) is
        sampled independently of the padded width, so crop content is unchanged.
        1/255 is dropped since grid_sample is linear in the source pixels.
        """

        def __init__(self):
            super().__init__()
            ys = torch.linspace(0.0, 1.0, REC_HEIGHT, dtype=torch.float32)
            self.register_buffer('ys', ys.reshape(1, REC_HEIGHT, 1))
            # Pre-shaped sample axis (1, 1, REC_MAX_WIDTH); each batch slices the
            # last dim to out_width with no runtime view/reshape.
            self.register_buffer('xs_full', torch.arange(REC_MAX_WIDTH, dtype=torch.float32).view(1, 1, -1))

        def forward(self, image_nchw, polys, flip_flags, target_widths):
            image = image_nchw.float()
            height = torch._shape_as_tensor(image_nchw)[2].to(torch.float32)
            width = torch._shape_as_tensor(image_nchw)[3].to(torch.float32)
            num_boxes = polys.shape[0]

            target_f = target_widths.to(torch.float32)
            out_width = torch.clamp(
                torch.ceil(target_f.max() / REC_WIDTH_ALIGN) * REC_WIDTH_ALIGN,
                min=float(REC_WIDTH_ALIGN), max=float(REC_MAX_WIDTH)).to(torch.int64)
            xs_axis = self.xs_full[..., :out_width]

            target_w = target_f.view(-1, 1, 1)
            active = xs_axis < target_w
            xs = xs_axis / torch.clamp(target_w - 1.0, min=1.0)
            grid_x = xs.expand(num_boxes, REC_HEIGHT, -1)
            grid_y = self.ys.expand_as(grid_x)
            flags = flip_flags.to(torch.bool).view(-1, 1, 1)
            grid_x = torch.where(flags, 1.0 - grid_x, grid_x)
            grid_y = torch.where(flags, 1.0 - grid_y, grid_y)
            grid_x = grid_x[..., None]
            grid_y = grid_y[..., None]
            one_minus_grid_x = 1.0 - grid_x
            one_minus_grid_y = 1.0 - grid_y

            polys = polys.view(-1, 4, 1, 2)
            p0, p1, p2, p3 = polys.split(1, dim=1)
            top = p0 * one_minus_grid_x + p1 * grid_x
            bottom = p3 * one_minus_grid_x + p2 * grid_x
            points = top * one_minus_grid_y + bottom * grid_y
            inv_scale = torch.stack([
                2.0 / torch.clamp(width - 1.0, min=1.0),
                2.0 / torch.clamp(height - 1.0, min=1.0)])
            grid = points * inv_scale - 1.0

            batch_image = image.expand(num_boxes, -1, -1, -1)
            crops = F.grid_sample(batch_image, grid, align_corners=True, padding_mode='border')
            crops = crops.round().clamp(0, 255).to(torch.uint8)
            pad = torch.full_like(crops, REC_PAD_VALUE)
            crops = torch.where(active.view(num_boxes, 1, 1, -1), crops, pad)
            return crops[:, RGB_TO_BGR]

    class TextlineFlipExport(nn.Module):
        """Per-box textline-orientation crop + classifier -> int8 180-flip flags.

        Emits ``1`` where the line is upside-down (class 1) so the rec crop can
        180-rotate it.  Traced as a leaf; runs only inside the ``use_textline``
        ONNX ``If`` then-branch.
        """

        def __init__(self, model, pre):
            super().__init__()
            self.crop = TextlineCropExport()
            self.classifier = ClsExport(model, pre, TEXTLINE_SIZE)

        def forward(self, image_nchw, polys):           # RGB image, polys (N, 4, 2)
            crops_bgr = self.crop(image_nchw, polys)
            class_id, _ = self.classifier(crops_bgr)
            return (class_id.reshape(-1) == 1).to(torch.int8)

    class FusedRecExport(nn.Module):
        """Textline-orientation + recognition fused behind an int8 switch input.

        ``use_textline`` (0/1) drives a scripted ONNX ``If``: when 1 the textline
        classifier runs and produces per-box 180-flip flags; when 0 it is skipped
        (flags all zero).  The two heavy halves are traced leaves and only this
        thin dispatch is scripted, so the exporter emits a genuine ``If`` around
        the classifier.
        """

        def __init__(self, textline_flip, rec):
            super().__init__()
            self.textline_flip = textline_flip
            self.rec = rec

        def forward(self, image_nchw, polys, target_widths, use_textline):
            if bool(use_textline.reshape(()) == 1):
                flip_flags = self.textline_flip(image_nchw, polys)
            else:
                flip_flags = torch.zeros([polys.size(0)], dtype=torch.int8)
            return self.rec(image_nchw, polys, flip_flags, target_widths)

    return (ClsExport, DocOriExport, DetFullExport, RecExport, UnwarpExport,
            TextlineFlipExport, FusedRecExport)


class _OnnxBuilder:
    """Tiny ONNX node/initializer accumulator with automatic naming.

    Keeps the hand-built DB postprocess graph compact instead of a flat wall of
    ``helper.make_node`` calls.
    """

    def __init__(self, helper, tensor_proto):
        self._helper = helper
        self._tp = tensor_proto
        self.nodes = []
        self.inits = []
        self._counter = 0

    def name(self, prefix):
        self._counter += 1
        return f'{prefix}_{self._counter}'

    def const(self, values, dtype, name=None):
        array = np.asarray(values, dtype=dtype)
        const_name = name or self.name('c')
        proto_type = {np.dtype('int64'): self._tp.INT64,
                      np.dtype('int32'): self._tp.INT32,
                      np.dtype('float32'): self._tp.FLOAT,
                      np.dtype('bool'): self._tp.BOOL}[array.dtype]
        self.inits.append(self._helper.make_tensor(
            const_name, proto_type, array.shape, array.reshape(-1).tolist()))
        return const_name

    def node(self, op_type, inputs, outputs=1, out=None, **attrs):
        if out is not None:
            names = [out] if isinstance(out, str) else list(out)
        else:
            names = [self.name(op_type.lower()) for _ in range(outputs)]
        self.nodes.append(self._helper.make_node(op_type, inputs, names, **attrs))
        return names[0] if len(names) == 1 else names

def export_db_postprocess_onnx(path):
    """Pure-ONNX DBPostProcess: prob map -> text polygons + scores + rec widths.

    Tensor-only replacement for the cv2 contour pipeline: connectivity via an
    iterative 3x3 max-propagation ``Loop`` (exact 8-connected components),
    per-component axis-aligned boxes via ``ScatterElements`` min/max, box score =
    mean prob over an integral image, DB unclip = axis-aligned expansion.  Outputs
    polygons (tl, tr, br, bl) in source coords, tall boxes rolled so the long side
    maps to the recognition width.
    """
    import onnx
    from onnx import TensorProto, helper

    g = _OnnxBuilder(helper, TensorProto)
    f0 = g.const([0.0], np.float32)
    f1 = g.const([1.0], np.float32)
    i1 = g.const([1], np.int64)
    i1_32 = g.const([1], np.int32)        # int32 unit for index arithmetic
    flat = g.const([-1], np.int64)        # shared 1-D flatten shape
    thresh = g.const([DB_THRESH], np.float32)
    box_thresh = g.const([DB_BOX_THRESH], np.float32)
    unclip = g.const([DB_UNCLIP_RATIO], np.float32)
    min_size = g.const([float(DB_MIN_SIZE)], np.float32)
    min_size2 = g.const([float(DB_MIN_SIZE + 2)], np.float32)
    rec_h = g.const([float(REC_HEIGHT)], np.float32)
    rec_maxw = g.const([float(REC_MAX_WIDTH)], np.float32)
    tall_ratio = g.const([1.5], np.float32)
    two = g.const([2.0], np.float32)
    eps = g.const([1e-6], np.float32)
    max_iters = g.const(DB_CC_MAX_ITERS // DB_CC_UNROLL, np.int64)
    true_b = g.const(True, np.bool_)
    ax0 = g.const(0, np.int64)            # scalar axis for CumSum
    ax1 = g.const(1, np.int64)            # scalar axis for CumSum
    ax2 = g.const([2], np.int64)          # 1-D axis for the polygon Unsqueeze
    sq0 = g.const([0], np.int64)          # 1-D axes for Squeeze

    shp = g.node('Shape', ['prob_map'])
    Hh = g.node('Gather', [shp, g.const([2], np.int64)], axis=0)
    Ww = g.node('Gather', [shp, g.const([3], np.int64)], axis=0)
    Hs = g.node('Squeeze', [Hh, sq0])
    Ws = g.node('Squeeze', [Ww, sq0])
    Hf = g.node('Cast', [Hh], to=TensorProto.FLOAT)
    Wf = g.node('Cast', [Ww], to=TensorProto.FLOAT)
    hw2d = g.node('Concat', [Hh, Ww], axis=0)         # [H, W] shape (reused)
    HW = g.node('Mul', [Hh, Ww])
    HWs = g.node('Squeeze', [HW, sq0])
    HWp1 = g.node('Add', [HW, i1])

    pred = g.node('Squeeze', ['prob_map', g.const([0, 1], np.int64)])
    # 4-D bitmap built directly (no 2-D copy + reshape round-trip)
    bm4d = g.node('Cast', [g.node('Greater', ['prob_map', thresh])], to=TensorProto.FLOAT)

    rng = g.node('Range', [g.const(0, np.int64), HWs, g.const(1, np.int64)])
    idx_flat = g.node('Add', [g.node('Cast', [rng], to=TensorProto.FLOAT), f1])
    idx4d = g.node('Reshape', [idx_flat, g.node('Concat', [i1, i1, Hh, Ww], axis=0)])
    label0 = g.node('Mul', [idx4d, bm4d])

    # Connected components via iterative 3x3 max-propagation (Loop, early-exit).
    # The 3x3 kernel is REQUIRED for exact 8-connectivity (a larger kernel jumps
    # labels across background gaps and merges components).  DB_CC_UNROLL exact 3x3
    # steps run per Loop trip; max-prop is monotonic so the change test is a plain
    # Sub and "no change across a block" means fully converged.
    b = _OnnxBuilder(helper, TensorProto)
    state = 'label_in'
    for _ in range(DB_CC_UNROLL):
        pooled = b.node('MaxPool', [state], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        state = b.node('Mul', [pooled, bm4d])
    lab_out = state
    md = b.node('ReduceMax', [b.node('Sub', [lab_out, 'label_in'])], keepdims=0)
    cond_out = b.node('Greater', [md, g.const(0.0, np.float32, name='loop_zero')])
    body = helper.make_graph(
        b.nodes, 'cc_body',
        [helper.make_tensor_value_info('iter', TensorProto.INT64, []),
         helper.make_tensor_value_info('cond_in', TensorProto.BOOL, []),
         helper.make_tensor_value_info('label_in', TensorProto.FLOAT, [1, 1, 'H', 'W'])],
        [helper.make_tensor_value_info(cond_out, TensorProto.BOOL, []),
         helper.make_tensor_value_info(lab_out, TensorProto.FLOAT, [1, 1, 'H', 'W'])],
        b.inits)
    label = g.node('Loop', [max_iters, true_b, label0], body=body)

    lab_flat = g.node('Reshape', [label, flat])
    lab_int = g.node('Cast', [lab_flat], to=TensorProto.INT32)   # int32 scatter indices

    # per-pixel x/y coordinate grids, flattened (no H*W zero buffer)
    xs_row = g.node('Cast', [g.node('Range', [g.const(0, np.int64), Ws, g.const(1, np.int64)])], to=TensorProto.FLOAT)
    ys_col = g.node('Cast', [g.node('Range', [g.const(0, np.int64), Hs, g.const(1, np.int64)])], to=TensorProto.FLOAT)
    xs_f = g.node('Tile', [xs_row, Hh])
    ys_f = g.node('Reshape', [g.node('Expand', [g.node('Reshape', [ys_col, g.const([-1, 1], np.int64)]), hw2d]), flat])

    # representative pixel per component: label == own seed index implies in-bitmap,
    # so the extra bitmap check is redundant.
    rep_pos = g.node('Squeeze', [g.node('NonZero', [g.node('Equal', [lab_flat, idx_flat])]), sq0])
    rep_labels = g.node('Add', [g.node('Cast', [rep_pos], to=TensorProto.INT32), i1_32])

    init_W = g.node('Expand', [Wf, HWp1])
    init_H = g.node('Expand', [Hf, HWp1])
    init_0 = g.node('Expand', [f0, HWp1])
    xmin_t = g.node('ScatterElements', [init_W, lab_int, xs_f], axis=0, reduction='min')
    xmax_t = g.node('ScatterElements', [init_0, lab_int, xs_f], axis=0, reduction='max')
    ymin_t = g.node('ScatterElements', [init_H, lab_int, ys_f], axis=0, reduction='min')
    ymax_t = g.node('ScatterElements', [init_0, lab_int, ys_f], axis=0, reduction='max')
    xmin = g.node('Gather', [xmin_t, rep_labels], axis=0)
    xmax = g.node('Gather', [xmax_t, rep_labels], axis=0)
    ymin = g.node('Gather', [ymin_t, rep_labels], axis=0)
    ymax = g.node('Gather', [ymax_t, rep_labels], axis=0)

    integ = g.node('CumSum', [g.node('CumSum', [pred, ax0]), ax1])
    Jp_flat = g.node('Reshape', [g.node('Pad', [integ, g.const([1, 1, 0, 0], np.int64), f0]), flat])
    Wp1 = g.node('Add', [g.node('Cast', [Ww], to=TensorProto.INT32), i1_32])

    Wm1 = g.node('Sub', [Wf, f1])
    Hm1 = g.node('Sub', [Hf, f1])
    x0i = g.node('Cast', [g.node('Clip', [g.node('Floor', [xmin]), f0, Wm1])], to=TensorProto.INT32)
    x1i = g.node('Cast', [g.node('Clip', [g.node('Ceil', [xmax]), f0, Wm1])], to=TensorProto.INT32)
    y0i = g.node('Cast', [g.node('Clip', [g.node('Floor', [ymin]), f0, Hm1])], to=TensorProto.INT32)
    y1i = g.node('Cast', [g.node('Clip', [g.node('Ceil', [ymax]), f0, Hm1])], to=TensorProto.INT32)
    x1p = g.node('Add', [x1i, i1_32])
    y1p = g.node('Add', [y1i, i1_32])

    def corner(yy, xx):
        return g.node('Gather', [Jp_flat, g.node('Add', [g.node('Mul', [yy, Wp1]), xx])], axis=0)

    A = corner(y1p, x1p)
    Bc = corner(y0i, x1p)
    Cc = corner(y1p, x0i)
    Dc = corner(y0i, x0i)
    ssum = g.node('Sub', [g.node('Add', [g.node('Sub', [A, Bc]), Dc]), Cc])
    aref = g.node('Cast', [g.node('Mul', [g.node('Add', [g.node('Sub', [x1i, x0i]), i1_32]),
                                          g.node('Add', [g.node('Sub', [y1i, y0i]), i1_32])])], to=TensorProto.FLOAT)
    score = g.node('Div', [ssum, aref])

    def gather_selected(values, condition):
        indices = g.node('Squeeze', [g.node('NonZero', [condition]), sq0])
        return g.node('Gather', [values, indices], axis=0)

    bw = g.node('Add', [g.node('Sub', [xmax, xmin]), f1])
    bh = g.node('Add', [g.node('Sub', [ymax, ymin]), f1])
    keep = g.node('And', [g.node('GreaterOrEqual', [score, box_thresh]),
                          g.node('GreaterOrEqual', [g.node('Min', [bw, bh]), min_size])])
    # Gather from nonzero filter indices remains valid when no candidate survives.
    xmin = gather_selected(xmin, keep)
    xmax = gather_selected(xmax, keep)
    ymin = gather_selected(ymin, keep)
    ymax = gather_selected(ymax, keep)
    score = gather_selected(score, keep)

    bw = g.node('Add', [g.node('Sub', [xmax, xmin]), f1])
    bh = g.node('Add', [g.node('Sub', [ymax, ymin]), f1])
    perim = g.node('Max', [g.node('Mul', [two, g.node('Add', [bw, bh])]), eps])
    off = g.node('Div', [g.node('Mul', [g.node('Mul', [bw, bh]), unclip]), perim])
    xmin = g.node('Sub', [xmin, off]); xmax = g.node('Add', [xmax, off])
    ymin = g.node('Sub', [ymin, off]); ymax = g.node('Add', [ymax, off])

    keep2 = g.node('GreaterOrEqual', [g.node('Min', [g.node('Sub', [xmax, xmin]), g.node('Sub', [ymax, ymin])]), min_size2])
    xmin = gather_selected(xmin, keep2)
    xmax = gather_selected(xmax, keep2)
    ymin = gather_selected(ymin, keep2)
    ymax = gather_selected(ymax, keep2)
    score = gather_selected(score, keep2)

    srcf = g.node('Cast', ['source_hw'], to=TensorProto.FLOAT)
    src_h = g.node('Gather', [srcf, g.const([0], np.int64)], axis=0)
    src_w = g.node('Gather', [srcf, g.const([1], np.int64)], axis=0)
    sx = g.node('Div', [src_w, Wf])
    sy = g.node('Div', [src_h, Hf])
    xmin = g.node('Clip', [g.node('Round', [g.node('Mul', [xmin, sx])]), f0, src_w])
    xmax = g.node('Clip', [g.node('Round', [g.node('Mul', [xmax, sx])]), f0, src_w])
    ymin = g.node('Clip', [g.node('Round', [g.node('Mul', [ymin, sy])]), f0, src_h])
    ymax = g.node('Clip', [g.node('Round', [g.node('Mul', [ymax, sy])]), f0, src_h])

    cw = g.node('Sub', [xmax, xmin])
    ch = g.node('Sub', [ymax, ymin])
    tall = g.node('GreaterOrEqual', [ch, g.node('Mul', [tall_ratio, cw])])

    # Build corners as two (N, 4) coordinate rows and pick the tall-rolled ordering
    # with one Where per row (avoids per-corner Unsqueeze/Concat stacking).
    ux_min = g.node('Unsqueeze', [xmin, i1])
    ux_max = g.node('Unsqueeze', [xmax, i1])
    uy_min = g.node('Unsqueeze', [ymin, i1])
    uy_max = g.node('Unsqueeze', [ymax, i1])
    tall_c = g.node('Unsqueeze', [tall, i1])
    xs = g.node('Where', [tall_c,
                          g.node('Concat', [ux_min, ux_min, ux_max, ux_max], axis=1),
                          g.node('Concat', [ux_min, ux_max, ux_max, ux_min], axis=1)])
    ys = g.node('Where', [tall_c,
                          g.node('Concat', [uy_max, uy_min, uy_min, uy_max], axis=1),
                          g.node('Concat', [uy_min, uy_min, uy_max, uy_max], axis=1)])
    g.node('Concat', [g.node('Unsqueeze', [xs, ax2]), g.node('Unsqueeze', [ys, ax2])], axis=2, out='polys')
    g.node('Identity', [score], out='det_scores')

    cw_eff = g.node('Where', [tall, ch, cw])
    ch_eff = g.node('Where', [tall, cw, ch])
    tw = g.node('Round', [g.node('Div', [g.node('Mul', [rec_h, cw_eff]), g.node('Max', [ch_eff, f1])])])
    g.node('Cast', [g.node('Clip', [tw, f1, rec_maxw])], to=TensorProto.INT64, out='target_widths')

    graph = helper.make_graph(
        g.nodes, 'PPOCRv6_DBPost',
        [helper.make_tensor_value_info('prob_map', TensorProto.FLOAT, [1, 1, 'H', 'W']),
         helper.make_tensor_value_info('source_hw', TensorProto.INT64, [2])],
        [helper.make_tensor_value_info('polys', TensorProto.FLOAT, ['N', 4, 2]),
         helper.make_tensor_value_info('det_scores', TensorProto.FLOAT, ['N']),
         helper.make_tensor_value_info('target_widths', TensorProto.INT64, ['N'])],
        g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', OPSET)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


# ══════════════════════════════════════════════════════════════════════════════
# OFFICIAL SUB-MODEL AUTO-DOWNLOAD  (mirrors PPOCR/v6/demo.py BOS mechanism)
# ══════════════════════════════════════════════════════════════════════════════
def _migrate_legacy_official_models():
    """Reuse already-downloaded official sub-models from the legacy ~/.paddlex cache."""
    if not os.path.isdir(LEGACY_OFFICIAL_MODELS_DIR):
        return
    for _name, local_dir in OFFICIAL_SUBMODELS:
        legacy_dir = os.path.join(LEGACY_OFFICIAL_MODELS_DIR, os.path.basename(local_dir))
        if os.path.isdir(local_dir) or not os.path.isdir(legacy_dir):
            continue
        print(f'[models] migrating cached model: {legacy_dir} -> {local_dir}')
        shutil.copytree(legacy_dir, local_dir)


def ensure_official_submodels():
    """Download the official PP-OCRv6 sub-models (safetensors) on first run.

    Mirrors demo.py: point PaddleX at the BOS mirror, redirect storage into the
    local ``models/`` dir, reuse any legacy ``~/.paddlex`` cache, then fetch the
    missing packages.  Present packages are used as-is; rec is never downloaded.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, path in OFFICIAL_SUBMODELS:
        if os.path.isdir(path):
            print(f'[models] {name}: using cached package')
    missing = [(name, path) for name, path in OFFICIAL_SUBMODELS if not os.path.isdir(path)]
    if not missing:
        return

    # Configure the official download source BEFORE importing paddlex: the model
    # source / connectivity-check flags are read from the environment at import time.
    os.environ['PADDLE_PDX_MODEL_SOURCE'] = MODEL_SOURCE
    if DISABLE_MODEL_SOURCE_CHECK:
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['PADDLE_PDX_CACHE_HOME'] = PADDLEX_CACHE_DIR

    try:
        from paddlex.inference.utils.official_models import official_models
    except ImportError as exc:
        names = ', '.join(name for name, _ in missing)
        raise RuntimeError(
            f'paddlex is required to auto-download the official sub-models ({names}); '
            f'install paddlex/paddleocr, or place the *_safetensors packages under '
            f'{MODELS_DIR}.') from exc

    from pathlib import Path

    # Redirect downloads into the local models directory and force the hosters to
    # rebuild against it (each hoster captures the save dir when first constructed).
    official_models._save_dir = Path(MODELS_DIR)
    official_models._hosters = None

    _migrate_legacy_official_models()

    for name, path in missing:
        if os.path.isdir(path):                          # filled in by the legacy migration
            print(f'[models] {name}: using cached package')
            continue
        print(f'[models] {name}: downloading from {MODEL_SOURCE} ...')
        official_models.get_model_path(name, model_formats=['safetensors'])


def export_bundle():
    """Export the PP-OCRv6 ONNX model set."""
    return export_all()


def export_all():
    ensure_official_submodels()

    import torch
    from transformers import (
        PPLCNetForImageClassification, UVDocModel,
        PPOCRV6MediumDetForObjectDetection, PPOCRV6SmallRecForTextRecognition,
        AutoImageProcessor,
    )

    os.makedirs(EXPORT_DIR, exist_ok=True)
    (ClsExport, DocOriExport, DetFullExport, RecExport, UnwarpExport,
     TextlineFlipExport, FusedRecExport) = _build_wrappers()

    # ----- bundle the recognition character list for the standalone runtime -----
    rec_proc = AutoImageProcessor.from_pretrained(REC_MODEL_DIR)
    rec_char_list = list(rec_proc.character_list)
    np.save(os.path.join(EXPORT_DIR, 'rec_char_list.npy'), np.array(rec_char_list, dtype=object), allow_pickle=True)

    torch.manual_seed(0)

    # ══════════════════════════════════════════════════════════════════════════
    # Doc-orientation classifier
    # ══════════════════════════════════════════════════════════════════════════
    print('[export] doc-orientation ...')
    model = PPLCNetForImageClassification.from_pretrained(DOC_ORI_MODEL_DIR, dtype=torch.float32).eval()
    fold_classifier_dropout_scale(model)
    fuse_conv_bn_recursive(model)
    wrapper = DocOriExport(model, PRE_CLS).eval()
    dummy = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    torch.onnx.export(
        wrapper, (dummy,), onnx_model_DocOri,
        input_names=['image'], output_names=['image_oriented', 'angle'],
        dynamic_axes={'image': {2: 'H', 3: 'W'}, 'image_oriented': {2: 'OH', 3: 'OW'}},
        opset_version=OPSET, dynamo=False)
    del model, wrapper
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # UVDoc unwarp
    # ══════════════════════════════════════════════════════════════════════════
    print('[export] unwarp ...')
    model = UVDocModel.from_pretrained(UNWARP_MODEL_DIR, dtype=torch.float32).eval()
    fuse_conv_bn_recursive(model)
    wrapper = UnwarpExport(model).eval()
    dummy = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    torch.onnx.export(
        wrapper, (dummy,), onnx_model_Unwarp,
        input_names=['image'], output_names=['rectified'],
        dynamic_axes={'image': {2: 'H', 3: 'W'}, 'rectified': {2: 'H', 3: 'W'}},
        opset_version=OPSET, dynamo=False)
    del model, wrapper
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # Detection
    # ══════════════════════════════════════════════════════════════════════════
    print('[export] detection ...')
    model = PPOCRV6MediumDetForObjectDetection.from_pretrained(DET_MODEL_DIR, dtype=torch.float32).eval()
    fuse_conv_bn_recursive(model)
    wrapper = DetFullExport(model, PRE_DET).eval()
    dummy = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    torch.onnx.export(
        wrapper, (dummy,), onnx_model_Det,
        input_names=['image'], output_names=['prob_map', 'source_hw'],
        dynamic_axes={'image': {2: 'H', 3: 'W'}, 'prob_map': {2: 'H', 3: 'W'}},
        opset_version=OPSET, dynamo=False)
    del model, wrapper
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # Recognition  (textline-orientation classifier fused in behind a switch input)
    # ══════════════════════════════════════════════════════════════════════════
    # Textline classifier folded into the rec graph behind the int8 `use_textline`
    # If (see FusedRecExport): the heavy halves are traced leaves, only the switch
    # dispatch is scripted.
    print('[export] recognition (textline + crop + rec fused, switch-gated) ...')
    textline_model = PPLCNetForImageClassification.from_pretrained(
        TEXTLINE_MODEL_DIR, dtype=torch.float32).eval()
    fold_classifier_dropout_scale(textline_model)
    fuse_conv_bn_recursive(textline_model)
    textline_flip = TextlineFlipExport(textline_model, PRE_CLS).eval()

    rec_model = PPOCRV6SmallRecForTextRecognition.from_pretrained(
        REC_MODEL_DIR, dtype=torch.float32, attn_implementation='eager').eval()
    fuse_conv_bn_recursive(rec_model)
    rec_core = RecExport(rec_model, PRE_REC).eval()

    dummy_image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    dummy_polys = torch.tensor(
        [[[20.0, 20.0], [260.0, 20.0], [260.0, 60.0], [20.0, 60.0]],
         [[30.0, 90.0], [190.0, 90.0], [190.0, 130.0], [30.0, 130.0]]], dtype=torch.float32)
    dummy_flip = torch.tensor([0, 1], dtype=torch.int8)
    dummy_target_widths = torch.tensor([288, 192], dtype=torch.int64)
    dummy_use_textline = torch.tensor([1], dtype=torch.int8)

    # Trace the heavy halves into ScriptModule leaves, then script ONLY the int8
    # switch dispatch (the leaves are called as compiled graphs, not re-scripted).
    textline_flip_traced = torch.jit.trace(
        textline_flip, (dummy_image, dummy_polys), check_trace=False)
    rec_core_traced = torch.jit.trace(
        rec_core, (dummy_image, dummy_polys, dummy_flip, dummy_target_widths), check_trace=False)
    wrapper = torch.jit.script(FusedRecExport(textline_flip_traced, rec_core_traced).eval())

    torch.onnx.export(
        wrapper, (dummy_image, dummy_polys, dummy_target_widths, dummy_use_textline), onnx_model_Rec,
        input_names=['image', 'polys', 'target_widths', 'use_textline'],
        output_names=['decoded_ids', 'decoded_lengths', 'text_scores'],
        dynamic_axes={'image': {2: 'H', 3: 'W'}, 'polys': {0: 'N'},
                      'target_widths': {0: 'N'},
                      'decoded_ids': {0: 'N', 1: 'T'},
                      'decoded_lengths': {0: 'N'}, 'text_scores': {0: 'N'}},
        opset_version=OPSET, dynamo=False)
    del textline_model, textline_flip, textline_flip_traced
    del rec_model, rec_core, rec_core_traced, wrapper
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # Tensor-only ONNXRuntime postprocess stages (no cv2 / numpy in the hot path)
    # ══════════════════════════════════════════════════════════════════════════
    print('[export] tensor-only onnx-runtime stages ...')
    export_db_postprocess_onnx(onnx_model_DBPost)

    print('[export] done ->', EXPORT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    export_bundle()
    return subprocess.call(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, 'Inference_PPOCR_ONNX.py'),
            '--model-folder',
            EXPORT_DIR,
        ]
    )


if __name__ == '__main__':
    raise SystemExit(main())
