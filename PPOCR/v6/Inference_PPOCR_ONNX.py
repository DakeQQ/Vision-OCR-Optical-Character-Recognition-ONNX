import argparse
import os
import time

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FOLDER = os.path.join(SCRIPT_DIR, 'PPOCRv6_Optimized')
# Backward-compatible configuration alias.
EXPORT_DIR = DEFAULT_MODEL_FOLDER

# --- ONNX model paths ------------------------------------------------------------
onnx_model_DocOri = os.path.join(EXPORT_DIR, 'PPOCRv6_DocOri.onnx')
onnx_model_Unwarp = os.path.join(EXPORT_DIR, 'PPOCRv6_Unwarp.onnx')
onnx_model_Det    = os.path.join(EXPORT_DIR, 'PPOCRv6_Det.onnx')
onnx_model_Rec    = os.path.join(EXPORT_DIR, 'PPOCRv6_Rec.onnx')
onnx_model_DBPost = os.path.join(EXPORT_DIR, 'PPOCRv6_DBPost.onnx')

# --- Pipeline stage toggles ------------------------------------------------------
USE_DOC_ORI      = True            # Document orientation classification (runtime stage).
USE_UNWARP       = True            # UVDoc unwarping (runtime stage).
USE_TEXTLINE_ORI = True            # Text-line orientation and recognition switch.

# --- ONNX / runtime --------------------------------------------------------------
ORT_Accelerate_Providers = []      # e.g. ['CUDAExecutionProvider'], ['DmlExecutionProvider'].
MAX_THREADS              = 0       # 0 = auto.
DEVICE_ID                = 0
ORT_LOG                  = False
ORT_FP16                 = False

# --- Recognition batching --------------------------------------------------------
REC_MAX_WIDTH  = 3200
REC_BATCH_SIZE = 8                 # Maximum crops per recognition batch.

# --- Demo image ------------------------------------------------------------------
DEFAULT_IMAGE      = os.path.join(SCRIPT_DIR, 'general_ocr_002.png')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
# Backward-compatible configuration aliases.
DEMO_IMAGE_PATH     = DEFAULT_IMAGE
OUTPUT_DIR          = DEFAULT_OUTPUT_DIR
VISUALIZE_OUTPUT    = True
VIS_LABEL_FONT_SIZE = 15
# Visualization font candidates, tried in order (first existing file wins; see
# _load_visualization_font).
if os.name == 'nt':
    _SYS_FONTS_DIR = os.path.join(r'C:\Windows', 'Fonts')
    _PLATFORM_FONT_CANDIDATES = (
        os.path.join(_SYS_FONTS_DIR, 'msyh.ttc'),    # Microsoft YaHei (Simplified Chinese)
        os.path.join(_SYS_FONTS_DIR, 'msyh.ttf'),    # Microsoft YaHei (older single-face)
        os.path.join(_SYS_FONTS_DIR, 'simsun.ttc'),  # SimSun (always present)
        os.path.join(_SYS_FONTS_DIR, 'simhei.ttf'),  # SimHei
        os.path.join(_SYS_FONTS_DIR, 'msjh.ttc'),    # Microsoft JhengHei (Traditional Chinese)
        os.path.join(_SYS_FONTS_DIR, 'arial.ttf'),   # Latin-only fallback
    )
else:
    _PLATFORM_FONT_CANDIDATES = (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    )
VIS_FONT_CANDIDATES = ('',) + _PLATFORM_FONT_CANDIDATES

def parse_args():
    parser = argparse.ArgumentParser(description='Run PP-OCRv6 ONNX inference.')
    parser.add_argument('--model-folder', default=DEFAULT_MODEL_FOLDER)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ONNXRUNTIME HELPERS  (OrtValue + IOBinding, LightOnOCR naming/behaviour)
# ══════════════════════════════════════════════════════════════════════════════
import onnxruntime


def bind_ort_out(binding, names, device):
    """Bind outputs by name, letting ORT allocate on `device`."""
    for name in names:
        binding.bind_output(name, device)


def create_ort_from_numpy(array, device, device_id):
    """Create an OrtValue from an existing numpy array."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()

for _opt in (session_opts, run_options):
    _opt.log_severity_level = 0 if ORT_LOG else 4
    _opt.log_verbosity_level = 4

session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.enable_cast_chain_elimination':    '1',
}
for _k, _v in _session_configs.items():
    session_opts.add_session_config_entry(_k, _v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')
disabled_optimizers = None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if 'CUDAExecutionProvider' in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                    DEVICE_ID,
        'arena_extend_strategy':        'kNextPowerOfTwo',
        'cudnn_conv_algo_search':       'EXHAUSTIVE',
        'cudnn_conv_use_max_workspace': '1',
        'do_copy_in_default_stream':    '1',
    }]
    device_type = 'cuda'
elif 'DmlExecutionProvider' in ORT_Accelerate_Providers:
    provider_options = [{'device_id': DEVICE_ID, 'performance_preference': 'high_performance', 'device_filter': 'gpu'}]
    device_type = 'dml'
else:
    provider_options = None
    device_type = 'cpu'

packed_settings = {
    '_session_opts':        session_opts,
    '_providers':           ORT_Accelerate_Providers,
    '_provider_options':    provider_options,
    '_disabled_optimizers': disabled_optimizers,
}
RUN_DEVICE = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# HOST PREPROCESSING  (tiny integer shape arithmetic + unavoidable cv2 work)
# ══════════════════════════════════════════════════════════════════════════════
import cv2


def load_rgb_uint8(path):
    """Load an image as a contiguous (1, 3, H, W) uint8 RGB NCHW tensor."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def _load_visualization_font(size):
    from PIL import ImageFont

    for font_path in VIS_FONT_CANDIDATES:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _fit_label(draw, label, font, max_width):
    if draw.textbbox((0, 0), label, font=font)[2] <= max_width:
        return label
    ellipsis = '...'
    low, high = 0, len(label)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = label[:mid].rstrip() + ellipsis
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            low = mid
        else:
            high = mid - 1
    return label[:low].rstrip() + ellipsis


def save_ocr_visualization(image_hwc, result, save_path):
    """Save a YOLO-style overlay in the same coordinate space as dt_polys."""
    canvas = np.ascontiguousarray(image_hwc.copy())
    overlay = canvas.copy()
    height, width = canvas.shape[:2]
    palette = (
        (42, 157, 143), (233, 196, 106), (231, 111, 81), (38, 70, 83),
        (46, 134, 193), (155, 89, 182), (39, 174, 96), (214, 48, 49),
    )

    for index, poly in enumerate(result.get('dt_polys', [])):
        points = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = palette[index % len(palette)]
        cv2.fillPoly(overlay, [points], color)

    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)

    for index, poly in enumerate(result.get('dt_polys', [])):
        points = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = palette[index % len(palette)]
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)

    from PIL import Image, ImageDraw

    pil_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_image)
    font = _load_visualization_font(VIS_LABEL_FONT_SIZE)

    for index, poly in enumerate(result.get('dt_polys', [])):
        points = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = palette[index % len(palette)]

        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        text = rec_texts[index] if index < len(rec_texts) else ''
        score = rec_scores[index] if index < len(rec_scores) else 0.0
        label = f'{index:02d} {score:.2f} {text}'

        x = int(np.clip(points[:, 0, 0].min(), 0, max(width - 1, 0)))
        y = int(np.clip(points[:, 0, 1].min(), 0, max(height - 1, 0)))
        label = _fit_label(draw, label, font, max(width - x - 8, 1))
        left, upper, right_text, lower = draw.textbbox((0, 0), label, font=font)
        label_w = right_text - left
        label_h = lower - upper
        top = y - label_h - 8
        if top < 0:
            top = min(y + 2, max(height - label_h - 8, 0))
        bottom = min(top + label_h + 8, height)
        right = min(x + label_w + 8, width)
        draw.rectangle((x, top, right, bottom), fill=color)
        draw.text((x + 4, top + 3 - upper), label, fill=(255, 255, 255), font=font)

    canvas = np.asarray(pil_image)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME PIPELINE  (persistent sessions, OrtValue + IOBinding, zero host churn)
# ══════════════════════════════════════════════════════════════════════════════
class PPOCRv6Pipeline:
    """End-to-end PP-OCRv6 OCR over persistent ONNXRuntime sessions."""

    def __init__(self):
        char_path = os.path.join(EXPORT_DIR, 'rec_char_list.npy')
        if os.path.exists(char_path):
            self.char_list = list(np.load(char_path, allow_pickle=True))
        else:
            raise FileNotFoundError(
                f'Missing bundled recognition character list: {char_path}. '
                'Re-export the complete PP-OCRv6 ONNX bundle.'
            )

        self.sessions, self.bindings, self.io = {}, {}, {}
        stages = (
            ('doc_ori', onnx_model_DocOri, lambda: USE_DOC_ORI),
            ('unwarp',  onnx_model_Unwarp, lambda: USE_UNWARP),
            ('det',     onnx_model_Det,    lambda: True),
            ('rec',     onnx_model_Rec,    lambda: True),
        )
        for tag, path, enabled in stages:
            if enabled() and os.path.exists(path):
                session = create_session(path, **packed_settings)
                self.sessions[tag] = session
                self.bindings[tag] = session.io_binding()
                self.io[tag] = (get_in_names(session), get_out_names(session))
        for tag, path in (('db_post', onnx_model_DBPost),):
            if os.path.exists(path):
                session = create_session(path, **packed_settings)
                self.sessions[tag] = session
                self.bindings[tag] = session.io_binding()
                self.io[tag] = (get_in_names(session), get_out_names(session))
        if 'det' in self.sessions:
            print(f'Usable Providers: {self.sessions["det"].get_providers()}')

    # --- generic IOBinding run -------------------------------------------------
    def _to_ort(self, value):
        """Wrap a numpy array as an OrtValue; pass an existing OrtValue through."""
        if isinstance(value, onnxruntime.OrtValue):
            return value
        return create_ort_from_numpy(value, RUN_DEVICE, DEVICE_ID)

    def _infer_ort(self, tag, *inputs):
        """Run one stage with OrtValue I/O, leaving outputs on the execution device.

        Inputs may be OrtValues (bound through, no re-upload) or numpy arrays
        (wrapped once); outputs stay device-resident so consecutive stages chain
        without a host round-trip. A binding is never re-run before its outputs
        are consumed, so each handoff stays valid.
        """
        session, binding = self.sessions[tag], self.bindings[tag]
        in_names, out_names = self.io[tag]
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        for name, value in zip(in_names, inputs):
            binding.bind_ortvalue_input(name, self._to_ort(value))
        bind_ort_out(binding, out_names, RUN_DEVICE)
        run(session, binding)
        return binding.get_outputs()

    # --- stages ----------------------------------------------------------------
    def doc_orientation(self, image_ort):
        """Classify orientation and re-orient the image upright, both in-graph.

        Returns the upright image as a device-resident OrtValue plus the detected
        angle, which is pulled back to the host only for reporting.
        """
        image_oriented, angle = self._infer_ort('doc_ori', image_ort)
        angle_value = int(angle.numpy().flat[0])
        return image_oriented, angle_value

    def _aux_ready(self):
        required = {'db_post', 'rec'}
        return required.issubset(self.sessions)

    def unwarp(self, image_ort):
        return self._infer_ort('unwarp', image_ort)[0]

    def detect(self, image_ort):
        """Detect text boxes fully in ONNXRuntime: resize -> det -> DB postprocess.

        Returns polygons ordered (tl, tr, br, bl) in processed-image coordinates
        (tall boxes pre-rolled so their long side maps to the recognition width),
        per-box detection scores and the recognition target widths. Large
        intermediates stay on the execution device; only the small polygon/score/
        width OrtValues are sliced on the host by the caller.
        """
        prob_map, source_hw = self._infer_ort('det', image_ort)
        return self._infer_ort('db_post', prob_map, source_hw)

    def _decode_rec_outputs(self, outputs):
        decoded_ids, decoded_lengths, text_scores = outputs
        results = []
        for row in range(decoded_ids.shape[0]):
            length = int(decoded_lengths[row])
            token_ids = decoded_ids[row, :length]
            text = ''.join(self.char_list[int(token_id)] for token_id in token_ids)
            results.append((text, float(text_scores[row])))
        return results

    def recognize(self, image_ort, polys, target_widths, use_textline):
        results = [('', 0.0)] * len(polys)
        if len(polys) == 0:
            return results
        # Sort boxes by target width so each batch pads close to its own width,
        # minimising recognition padding. image_ort is bound once and reused per
        # batch (no re-upload); textline classification + 180 flip run inside the
        # rec graph (gated by use_textline), so only per-chunk polygon/width slices
        # and decoded ids cross the host boundary.
        order = np.argsort(np.minimum(target_widths, REC_MAX_WIDTH), kind='stable')
        for start in range(0, len(order), REC_BATCH_SIZE):
            chunk = order[start:start + REC_BATCH_SIZE].astype(np.int64)
            rec_outputs = [ov.numpy() for ov in self._infer_ort(
                'rec', image_ort, polys[chunk], target_widths[chunk], use_textline)]
            for offset, result in enumerate(self._decode_rec_outputs(rec_outputs)):
                results[int(chunk[offset])] = result
        return results

    # --- full pipeline ---------------------------------------------------------
    def predict(self, image_nchw):
        if not self._aux_ready():
            raise RuntimeError(
                'ONNX postprocess stages are unavailable; re-run export to build '
                'the db_post / rec stages.')

        original_shape = image_nchw.shape[2:]
        image_ort = self._to_ort(np.ascontiguousarray(image_nchw))

        angle = 0
        if 'doc_ori' in self.sessions:
            image_ort, angle = self.doc_orientation(image_ort)
        unwarp_used = 'unwarp' in self.sessions
        if unwarp_used:
            image_ort = self.unwarp(image_ort)

        # Detection/recognition feed off the same on-device image OrtValue; only the
        # small per-box tensors are pulled to the host for bucketing and output.
        polys_ort, box_scores_ort, target_widths_ort = self.detect(image_ort)
        polys = polys_ort.numpy().astype(np.float32, copy=False)
        box_scores = box_scores_ort.numpy().astype(np.float32, copy=False)
        target_widths = target_widths_ort.numpy().astype(np.int64, copy=False)

        # Textline-orientation is fused into the rec graph behind the int8
        # use_textline switch; when disabled the rec If skips the classifier
        # sub-graph entirely instead of loading a separate session.
        use_textline = np.array([1 if USE_TEXTLINE_ORI else 0], dtype=np.int8)
        rec_results = self.recognize(image_ort, polys, target_widths, use_textline)

        # Polygons stay in the preprocessed-image coordinate space, mirroring native
        # PP-OCRv6: back-projecting through the UVDoc grid (not an exact inverse)
        # would compress boxes toward the image centre. OrtValue.shape() reports the
        # processed H/W with no device->host copy; the full image is pulled back only
        # when a visualization is requested.
        _, _, proc_h, proc_w = image_ort.shape()
        dt_polys = [np.asarray(poly, dtype=np.float32).reshape(-1, 2).round().astype(int).tolist()
                    for poly in polys]

        result = {
            'angle': angle,
            'dt_polys': dt_polys,
            'bbox_coordinate_space': 'processed_image',
            'processed_image_shape': [int(proc_h), int(proc_w)],
            'original_image_shape': list(original_shape),
            'doc_unwarping_used': unwarp_used,
            'onnx_postprocess': True,
            'rec_texts': [text for text, _ in rec_results],
            'rec_scores': [score for _, score in rec_results],
            'det_scores': [float(score) for score in np.asarray(box_scores).reshape(-1)],
        }
        if VISUALIZE_OUTPUT:
            result['_visual_image'] = np.ascontiguousarray(image_ort.numpy()[0].transpose(1, 2, 0))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# DEMO ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(image_path=None, output_dir=None):
    """Run the PP-OCRv6 ONNX pipeline and save its result artifacts."""
    import json

    image_path = DEMO_IMAGE_PATH if image_path is None else os.fspath(image_path)
    output_dir = OUTPUT_DIR if output_dir is None else os.fspath(output_dir)
    image_nchw = load_rgb_uint8(image_path)
    original_image_hwc = np.ascontiguousarray(image_nchw[0].transpose(1, 2, 0))

    pipeline = PPOCRv6Pipeline()

    start = time.time()
    result = pipeline.predict(image_nchw)
    elapsed = time.time() - start

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, 'general_ocr_002_onnx_res.json')
    out_vis = os.path.join(output_dir, 'general_ocr_002_onnx_ocr_res_img.png')

    json_result = result.copy()
    visual_image = json_result.pop('_visual_image', None)
    if visual_image is None:
        visual_image = original_image_hwc

    if VISUALIZE_OUTPUT:
        save_ocr_visualization(visual_image, json_result, out_vis)

    if VISUALIZE_OUTPUT:
        json_result['visualization_path'] = out_vis
        json_result['visualization_source_path'] = image_path
    with open(out_json, 'w', encoding='utf-8') as handle:
        json.dump({'input_path': image_path, **json_result}, handle, ensure_ascii=False, indent=2)

    print(f'\nPP-OCRv6 ONNX pipeline: {len(result["rec_texts"])} text lines in {elapsed * 1000:.1f} ms')
    print(f'  doc-orientation angle: {result["angle"]}')
    for text, score in zip(result['rec_texts'], result['rec_scores']):
        if text:
            print(f'   {score:.3f}  {text}')
    print(f'\nResult JSON -> {out_json}')
    if VISUALIZE_OUTPUT:
        print(f'Visualization -> {out_vis}')
    return json_result


def configure_model_folder(model_folder):
    """Point the pipeline at one complete PP-OCRv6 ONNX bundle."""
    global EXPORT_DIR, onnx_model_DocOri, onnx_model_Unwarp, onnx_model_Det, onnx_model_Rec, onnx_model_DBPost
    EXPORT_DIR = os.path.abspath(os.fspath(model_folder))
    onnx_model_DocOri = os.path.join(EXPORT_DIR, 'PPOCRv6_DocOri.onnx')
    onnx_model_Unwarp = os.path.join(EXPORT_DIR, 'PPOCRv6_Unwarp.onnx')
    onnx_model_Det = os.path.join(EXPORT_DIR, 'PPOCRv6_Det.onnx')
    onnx_model_Rec = os.path.join(EXPORT_DIR, 'PPOCRv6_Rec.onnx')
    onnx_model_DBPost = os.path.join(EXPORT_DIR, 'PPOCRv6_DBPost.onnx')


def main():
    args = parse_args()
    configure_model_folder(args.model_folder)
    run_inference()


if __name__ == '__main__':
    main()
