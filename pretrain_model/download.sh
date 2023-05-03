#!/bin/bash

# BLIP Model
if [ ! -f model_large_caption.pth ]; then
    wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth .
fi

# Real-ESRGAN
if [ ! -f RealESRGAN_x4plus.pth ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth .
fi
