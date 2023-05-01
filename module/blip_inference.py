import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder


model_path = os.path.join("pretrain_model", "model_large_caption.pth")


def init_blip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 384
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    model = blip_decoder(pretrained=model_path, image_size=384, vit="large")
    model.eval()
    model = model.to(device)

    return model, transform, device


def blip_inference(model, transform, device, image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image).to(device).unsqueeze(0)
    caption = model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5)

    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="tmp/test.png")
    args = parser.parse_args()

    model, transform, device = init_blip()
    caption = blip_inference(model, transform, device, args.image_path)
    print(caption)
