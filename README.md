# Vision-OCR-Optical-Character-Recognition-ONNX

### 功能 Features  
1. **支持的模型 Supported Models**:
   - [LightOnOCR-2](https://modelscope.cn/models/lightonai/LightOnOCR-2-1B)
   - [FireRedOCR](https://modelscope.cn/models/FireRedTeam/FireRed-OCR)
   - [Surya-OCR-2](https://huggingface.co/datalab-to/surya-ocr-2)
   - [LFM2-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract)
   - [LFM2.5-VL-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract)
   - [Falcon-Perception-300M](https://huggingface.co/tiiuae/Falcon-Perception-300M)
   - [PP-OCR-v6](https://modelscope.cn/models/PaddlePaddle/PP-OCRv6_medium_rec)
   - [PaddleOCR-VL-1.6](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6)
   - [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)
   - [Unlimited-OCR](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR/summary)

2. **Unified Workflow Script Entry Points**:
   - Exporters expose `export_bundle()`.
   - Optimizers expose `optimize_bundle()`.
   - Merged generative workflows expose `build_shared_merged_bundle()`.
   - Inference scripts expose `run_inference(...)`.


---

### 性能 Performance  
| OS           | Device       | Backend           | Model                     | Tokens per Second |
|:------------:|:------------:|:-----------------:|:-------------------------:|:-----------------:|
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | LightOnOCR-2-1B<br>Q4F32 |        24         |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | FireRedOCR-2B<br>Q4F32   |        15         |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | SuryaOCR-2-0.65B<br>Q4F32 |       40         |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | LFM2-Extract-0.35B<br>Q4F32 |       85       |





---

### To-Do List  
- [ ] [Qianfan-OCR](https://modelscope.cn/models/baidu-qianfan/Qianfan-OCR)
- [ ] [NuExtract3](https://github.com/numindai/nuextract)
---
