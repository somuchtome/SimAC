import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from PIL import Image
import numpy as np
from einops import rearrange

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--v",
    type=str,
    default="sks",
    help="The output directory where predictions are saved",
)

args = parser.parse_args()

if __name__ == "__main__":
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # define prompts
    prompts = [
        f"a photo of {args.v} person",
        f"a dslr portrait of {args.v} person",
        f"a photo of {args.v} person looking at the mirror",
        f"a photo of {args.v} person in front of eiffel tower",
    ]


    # create & load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True,
    ).to("cuda")

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        all_samples = list()
        for i in range(5):
            images = pipe([prompt] * 6, num_inference_steps=100, guidance_scale=7.5,).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")
                image = np.array(image, dtype=np.float32)  
                image /= 255.0  
                image = np.transpose(image, (2, 0, 1))  
                image = torch.from_numpy(image)  # numpy->tensor
                all_samples.append(image)
        grid = torch.stack(all_samples, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w') 
        grid = make_grid(grid, nrow=8)
        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img.save(f"{args.output_dir}/{prompt}.png")
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()
