import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from ip_adapter import IPAdapterPlus,IPAdapter

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/data1/hns/2024code/zest_code/models/image_encoder"
ip_ckpt = "/data1/hns/trained_model/20241024/20241024-15plus-10000.bin"
device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Image generation script using IPAdapterPlus")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for image generation')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated images')
    parser.add_argument('--scale', type=float, default=0.8, help='Scaling factor for image generation')
    # parser.add_argument('--ip_ckpt', type=str, required=True, help='Path to the IPAdapterPlus checkpoint')
    return parser.parse_args()

def save_image(image, path):
    image.save(path)

def main(args):
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

    # ip_ckpt = args.ip_ckpt
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    # ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    image = Image.open(args.image_path)
    image = image.resize((512, 512))

    images = ip_model.generate(
        pil_image=image,
        num_samples=10,
        num_inference_steps=50,
        seed=42,
        prompt=args.prompt,
        scale=args.scale  # Use the scale argument
    )

    save_image(image, f"{args.save_path}_input_with_prompt.png")

    # Save each result image separately
    for i, img in enumerate(images):
        save_image(img, f"{args.save_path}_result_{i+1}.png")

    # Save the prompt to a text file
    with open(f"{args.save_path}_prompt.txt", 'w') as f:
        f.write(args.prompt)

if __name__ == "__main__":
    args = parse_args()
    main(args)
