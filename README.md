<div align="center">
<h1>ArtCrafter: Text-Image Aligning Artistic Attribute Transfer via Embedding Reframing</h1>

[**Nisha Huang**](https://scholar.google.com/citations?user=wTmPkSsAAAAJ&hl=en) · [**Kaer Huang**](https://carlhuangnuc.github.io/) · [**Yifan Pu**](https://scholar.google.com/citations?user=oM9rnYQAAAAJ&hl=en&authuser=2) · [**Jiangshan Wang**](https://github.com/wangjiangshan0725) · [**Jie Guo**](https://scholar.google.com.hk/citations?user=hqLytuEAAAAJ&hl=zh-CN) · [**Yiqiang Yan**](https://scholar.google.com/citations?user=p6q5mOkAAAAJ&hl=en)· [**Xiu Li**](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ)<sup>*</sup>· [**Tong-Yee Lee**](http://graphics.csie.ncku.edu.tw/Tony/tony.htm)

Tsinghua University, Pengcheng Laboratory, Lenovo, National Cheng Kung University

<sup>*</sup>corresponding authors

<a href='https://arxiv.org/abs/2501.02064'><img src='https://img.shields.io/badge/Arxiv-Paper-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red)](https://huggingface.co/lisalisalisa/artcrafter)
[![GitHub](https://img.shields.io/github/stars/haha-lisa/ArtCrafter?style=social)](https://github.com/haha-lisa/ArtCrafter)

</div>


## Results
![MAIN3_e2-min](https://github.com/haha-lisa/ArtCrafter/blob/main/images/teaser.png)

## Abstract
Recent years have witnessed significant advancements in text-guided style transfer, primarily attributed to innovations in diffusion models. These models excel in conditional guidance, utilizing text or images to direct the sampling process. Traditional style transfer focuses on low-level visual features, such as brushstroke textures and color distributions, and appears more like applying an artistic filter to an image. Artistic attribute transfer, however, transcends the limitations of traditional style transfer by achieving the transfer of visual concepts from color and brushstrokes to high level aesthetic attributes such as composition, pose, and key semantic elements, resulting in more natural outcomes. Therefore, we propose an innovative text-to-image artistic attribute transfer framework named ArtCrafter. Specifically, we introduce an attention-based style extraction module, meticulously engineered to capture the subtle artistic attribute elements within an image. This module features a multi-layer architecture that leverages the capabilities of perceiver attention mechanisms to integrate fine-grained information. Additionally, we present a novel text-image aligning augmentation component that adeptly balances control over both modalities, enabling the model to efficiently map image and text embeddings into a shared feature space. We achieve this through attention operations that enable smooth information flow between modalities. Lastly, we incorporate an explicit modulation that seamlessly blends multimodal enhanced embeddings with original embeddings through an embedding reframing design, empowering the model to generate diverse outputs. Extensive experiments demonstrate that ArtCrafter yields impressive results in visual stylization, exhibiting exceptional levels of artistic attribute intensity, controllability, and diversity. 

## Framework
![MAIN3_e2-min](https://github.com/haha-lisa/ArtCrafter/blob/main/images/3pipeline_00.png)

## Install dependencies

Follow [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file#download-models) to download pre-trained checkpoints from [here](https://huggingface.co/h94/IP-Adapter).

```
git clone https://github.com/haha-lisa/ArtCrafter.git
cd ArtCrafter

# download the models
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models

# download the ckpt.bin
https://drive.google.com/file/d/1QAG2gWZ2NFKyJx6ZhcpQNhYyLM9ABWQr/view?usp=sharing
or
https://huggingface.co/lisalisalisa/artcrafter/blob/main/ckpt.bin
```


## Usage
```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from adapter import IPAdapterPlus


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder"
ip_ckpt = "ckpt.bin"
device = "cuda"


def save_image(image, path):
    image.save(path)


def main(image_path, prompt, save_path, scale=0.8):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    image = Image.open(image_path)
    image = image.resize((512, 512))

    images = ip_model.generate(
        pil_image=image,
        num_samples=10,
        num_inference_steps=50,
        seed=42,
        prompt=prompt,
        scale=scale  
    )

    save_image(image, f"{save_path}_input_with_prompt.png")


    for i, img in enumerate(images):
        save_image(img, f"{save_path}_result_{i+1}.png")

    with open(f"{save_path}_prompt.txt", 'w') as f:
        f.write(prompt)

if __name__ == "__main__":
    image_path = "path/to/your/input/image.jpg"  # "assets/1.jpg"
    prompt = "Your prompt here"  # "Cat"
    save_path = "path/to/save/results"  # "result.png"
    scale = 0.8  

    main(image_path, prompt, save_path, scale)
```

For quick command
```python
python main.py --image_path "assets/1.jpg" --prompt "Cat" --save_path "result.png"
```


## Cite
If you find InstantStyle useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{huang2025artcrafter,
  title={Artcrafter: Text-image aligning style transfer via embedding reframing},
  author={Huang, Nisha and Huang, Kaer and Pu, Yifan and Wang, Jiangshan and Guo, Jie and Yan, Yiqiang and Li, Xiu and Lee, Tong-Yee},
  journal={arXiv preprint arXiv:2501.02064},
  year={2025}
}

```