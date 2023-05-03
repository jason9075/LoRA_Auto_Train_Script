import argparse
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


OUT_SCALE = 4
MODEL_PATH = os.path.join("pretrain_model", "RealESRGAN_x4plus.pth")


def init_esrgan():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    netscale = 4

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=MODEL_PATH,
        model=model,
        half=True,
    )

    return upsampler


def esrgan_infernece(model, image):
    output, _ = model.enhance(image, outscale=OUT_SCALE)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="tmp/test.png")

    args = parser.parse_args()
    model = init_esrgan()
    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    image_output = esrgan_infernece(model, img)
    cv2.imwrite("./tmp/esrgan.jpg", image_output)
