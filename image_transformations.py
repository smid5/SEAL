from PIL import Image, ImageFilter
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import zlib
import wandb
import random
import io
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from optim_utils import image_distortion
from utils import *
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline


def compute_l2(image, sentence_model, k, b, seed, pipe, device):
    caption = generate_caption(image, cap_processor, cap_model, device=device)
    embed = sentence_model.encode(caption, convert_to_tensor=True).to(device)
    embed = embed / torch.norm(embed)
    noise = generate_initial_noise(embed, k, b, seed, device).to(dtype=pipe.vae.dtype)
    tensor = transform_img(image).unsqueeze(0).to(device, dtype=pipe.vae.dtype)
    latents = pipe.get_image_latents(tensor, sample=False)
    recon_noise = pipe.forward_diffusion(
        latents=latents,
        text_embeddings=pipe.get_text_embedding(''),
        guidance_scale=1,
        num_inference_steps=50,
    )
    return calculate_patch_l2(recon_noise, noise, k)

def detect_watermark_from_l2(l2_list, tau, m_match):
    """
    Returns (detected_bool, num_matches)
    """
    m = sum(1 for l2 in l2_list if l2 < tau)
    detected = (m >= m_match)
    return detected, m

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Noise patch detection with distortions')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='noise-detection')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--model_id', default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=2.3)
    parser.add_argument('--m_match', type=int, default=12)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project=args.wandb_project, name="exp_distortions", entity=args.wandb_entity, config=vars(args))

    # Load models
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16).to(device)
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)
    sentence_model = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)

    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)

    # Define distortion types based on distorted_image_list
    distortions = [
        {"name": "Clean", "params": {}},
        {"name": "Rot75", "params": {"r_degree": 75}},
        {"name": "JPEG25", "params": {"jpeg_ratio": 25}},
        {"name": "CropScale75", "params": {"crop_scale": 0.75, "crop_ratio": 0.75}},
        {"name": "Blur8", "params": {"gaussian_blur_r": 8}},
        {"name": "Noise0.1", "params": {"gaussian_std": 0.1}},
        {"name": "Brightness6", "params": {"brightness_factor": 6}},
    ]

    all_l2 = {}

    for k in args.k_values:
        for b in args.b_values:
            combo_key = f"{k}_{b}"
            all_l2[combo_key] = {
                "watermarked": {dist["name"]: [] for dist in distortions},
                "random": []
            }
            for img_idx in tqdm(range(args.start, args.end), desc=f'k={k}, b={b}'):
                prompt_1 = dataset[img_idx]['Prompt']
                prompt_2 = dataset[(img_idx + 1) % len(dataset)]['Prompt']

                # Generate watermarked image
                first_image = pipe(prompt_1).images[0]
                image_caption = generate_caption(first_image, cap_processor, cap_model)
                embed = sentence_model.encode(image_caption, convert_to_tensor=True).to(device)
                embed = embed / torch.norm(embed)

                image_noise = generate_initial_noise(embed, k, b, 42, device).to(dtype=pipe.vae.dtype)

                org_img = pipe(prompt_1, latents=image_noise).images[0]

                # Apply distortions and compute min_l2
                for dist in distortions:
                    torch.manual_seed(img_idx)
                    np.random.seed(img_idx)
                    random.seed(img_idx)
                    if dist["name"] == "Clean":
                        distorted_img = org_img
                    else:
                        distorted_img = image_distortion(None, org_img, seed=img_idx, **dist["params"])[1] 
                    l2 = compute_l2(distorted_img, sentence_model, k, b, 42, pipe, device)
                    detected, num_matches = detect_watermark_from_l2(l2, tau=args.tau, m_match=args.m_match)
                    all_l2[combo_key]["watermarked"][dist["name"]].append(int(detected))

                random_image = pipe(prompt_2).images[0]
                l2_random = compute_l2(random_image, sentence_model, k, b, 42, pipe, device)
                detected_random, num_matches_random = detect_watermark_from_l2(l2_random, tau=args.tau, m_match=args.m_match)
                all_l2[combo_key]["random"].append(int(detected_random))

            for dist in distortions:
                name = dist["name"]
                acc = np.mean(all_l2[combo_key]["watermarked"][name])
                print(name, acc)

            names = [d["name"] for d in distortions]
            accs = [np.mean(all_l2[combo_key]["watermarked"][n]) for n in names]

            plt.figure(figsize=(10,5))
            plt.bar(names, accs)
            plt.ylim(0, 1.05)
            plt.ylabel("Detection Accuracy")
            plt.xlabel("Image Transformation")
            plt.title(f"Robustness (k={k}, b={b}, tau={args.tau}, m_match={args.m_match})")

            for i, v in enumerate(accs):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"robustness_{combo_key}.png"))
            plt.show()

    wandb.finish()