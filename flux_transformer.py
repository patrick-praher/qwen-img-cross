import torch
from diffusers import Flux2Pipeline

# FLUX.2-klein-4b is highly optimized for local Mac execution
repo_id = "black-forest-labs/FLUX.2-klein-9b" 

# Use float16 or bfloat16 depending on your M-series generation support
dtype = torch.float16 

# 1. Load the pipeline directly
pipe = Flux2Pipeline.from_pretrained(
    repo_id, 
    torch_dtype=dtype
)

# 2. CRITICAL FOR MAC: Enable sequential CPU offloading to save Unified Memory
# This keeps only the active layer in MPS VRAM at any given second.
pipe.enable_sequential_cpu_offload()

# 3. Alternative option: If you have a high-spec Mac (64GB+ RAM / M-Max/Ultra),
# you can bypass offloading and push the full pipeline straight to MPS:
# pipe.to("mps")

# 4. Generate the image
prompt = "A cinematic shot of a futuristic neon city in rain, cyberpunk aesthetic"
image = pipe(
    prompt=prompt,
    num_inference_steps=28, # Adjust down to 4 if using a distilled 'Turbo' version
    guidance_scale=3.5,
    height=1024,
    width=1024,
).images[0]

image.save("flux2_mac_output.png")

