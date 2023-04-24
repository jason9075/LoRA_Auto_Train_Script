#!/bin/bash

mkdir ~/ckpt_model/
cd ~/ckpt_model/
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

accelerate config
