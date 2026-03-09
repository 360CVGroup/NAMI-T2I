# NAMI: Efficient Image Generation via Bridged Progressive Rectified Flow Transformers

  

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-NAMI-ff9900?style=flat)](https://huggingface.co/qihoo360/NAMI-T2I) [![arXiv](https://img.shields.io/badge/arXiv-2503.09242-B31B1B?style=flat)](https://arxiv.org/pdf/2503.09242)

  

![examples](examples.png)

  

  

The proposed NAMI architecture reduces the inference time required to generate 1024
resolution images by 64%, while still maintaining a high
level of image quality.

  

  

  

## 💡 Update

  

  

- [x] [2026.03.06] Release the NAMI-2B inference code and weights.

- [x] [2026.03.01] Paper was accepted by CVPR2026.
  

  

## 🧩 Environment Setup

  

  

  

```
1、pip install -r requirements.txt
2、Directly use our diffusers folder, or replace the corresponding files in the installed diffusers package in your Python environment with transformer_flux.py and pipeline_flux.py from the src directory.
```

  

  

  

## 📂 Preparation of Model Weights
  

  

We provide model weights for evaluation and deployment. Please download files from [NAMI](https://huggingface.co/qihoo360/NAMI-T2I), [mclip](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder), [mt5-xxl]( https://huggingface.co/google/mt5-xxl/tree/main) and place them in the `weights` directory. 

  

  

## ⏳ Inference Pipeline

  

  

  

Here we provide the inference demo for our NAMI.

  

  

```
cd src
python infer.py
```

  

## 🌸 Acknowledgement

  

  

This code is mainly built upon [Diffusers](https://github.com/huggingface/diffusers/tree/main), [Flux](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux), and [PyramidFlow](https://github.com/jy0205/Pyramid-Flow) repositories. Thanks so much for their solid work!

  

## 💖 Citation


If you find this repository useful, please consider citing our paper:
```
@article{ma2025nami,
  title={NAMI: Efficient Image Generation via Bridged Progressive Rectified Flow Transformers},
  author={Ma, Yuhang and Cheng, Bo and Liu, Shanyuan and Zhou, Hongyi and Wu, Liebucha and Leng, Dawei and Yin, Yuhui},
  journal={arXiv preprint arXiv:2503.09242},
  year={2025}
}
```
