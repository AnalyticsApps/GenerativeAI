#!/usr/bin/env bash
set -e
apt-get update

mkdir /opt/AIArtGenerator
cd /opt/AIArtGenerator

echo "Setup the required packages for Stability Diffusion Pipeline"

python -m venv .env
source .env/bin/activate
pip install diffusers["torch"] transformers
pip install diffusers["flax"] transformers
pip install accelerate
pip install git+https://github.com/huggingface/diffusers
pip install diffusers --upgrade
pip install invisible_watermark transformers accelerate safetensors
pip install gradio

# Create SSL Keys
public_ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

# Generate a private key
openssl genpkey -algorithm RSA -out private_key.pem
# Generate a certificate signing request (CSR)
openssl req -new -key private_key.pem -out certificate.csr -subj "/C=AU/ST=NSW/L=Sydney/O=AIArt/CN=$public_ip"
# Create a self-signed certificate using the CSR
openssl x509 -req -days 365 -in certificate.csr -signkey private_key.pem -out certificate.crt

# Code for AI Art Generator
cat << EOF > gradio_prompt_image.py
import gradio as gr
from gradio.themes.base import Base
import time
import gc

from diffusers import DiffusionPipeline
import torch

class Seafoam(Base):
    pass

seafoam = Seafoam()

example = """

A majestic lion jumping from a big stone at night.

Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.

"""

base = None
refiner = None

def loadBaseLLM():
    global base
    if base is None:
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    return base

def loadRefinerLLM():
    global refiner
    if refiner is None:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    return refiner


def generate_Image(prompt):

    global base
    global refiner

    loadBaseLLM()
    base.to("cuda")
    loadRefinerLLM()
    refiner.to("cuda")

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        ).images[0]

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)
    return image

with gr.Blocks(theme=seafoam) as demo:

    history_num = gr.State(value=0)
    history_prompt = gr.State(" ")

    gr.Markdown(
    """
    # AI Artist

    ## Image Generator based on prompt.

    """)

    prompt = gr.Textbox(label="Prompt", placeholder="Describe Image")

    with gr.Row():
        button = gr.Button("Generate Image", variant="primary")
        clear = gr.Button("Clear")
    with gr.Row():
        output = gr.Textbox(label="Prompt History")
        image_output = gr.Image(label="genImage", interactive=False)
    with gr.Accordion("Example Prompt"):
        gr.Markdown(example)

    def prompt_processing(prompt, history_num, history_prompt):
        history_num += 1
        history_prompt += "\n" + "Prompt " + str(history_num) + " :" + prompt + "\n"
        genImg = generate_Image(str(prompt))
        return history_prompt, genImg, history_num, history_prompt

    def clear_fun(history_num, history_prompt):
        history_num = 0
        history_prompt = " "
        return None, None, history_num, history_prompt

    button.click(fn=prompt_processing, inputs=[prompt, history_num, history_prompt], outputs=[output, image_output, history_num, history_prompt])
    clear.click(fn=clear_fun, inputs=[history_num, history_prompt], outputs=[prompt, output, history_num, history_prompt])

demo.launch(auth=("admin", "admin@123"), server_name="0.0.0.0", server_port=443, ssl_verify=False, ssl_certfile="certificate.crt",ssl_keyfile="private_key.pem")

EOF

nohup python gradio_prompt_image.py > output.log &
#python gradio_prompt_image.py

# Logs avaialble under /var/log/cloud-init-output.log
