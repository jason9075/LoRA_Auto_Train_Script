from safetensors.torch import load_file, save_file
import os
import torch
import argparse
import logging

logger = logging.getLogger()


prefix_keys = [
    #    "lora_te_text_model_encoder_layers",
    "lora_unet_down_blocks_0_attentions_0_transformer",
    "lora_unet_down_blocks_0_attentions_1_transformer",
    "lora_unet_down_blocks_1_attentions_0_transformer",
    "lora_unet_down_blocks_1_attentions_1_transformer",
    "lora_unet_down_blocks_2_attentions_0_transformer",
    "lora_unet_down_blocks_2_attentions_1_transformer",
    #    "lora_unet_mid_block_attentions_0_transformer",
    #    "lora_unet_up_blocks_1_attentions_0_transformer",
    #    "lora_unet_up_blocks_1_attentions_1_transformer",
    #    "lora_unet_up_blocks_1_attentions_2_transformer",
    #    "lora_unet_up_blocks_2_attentions_0_transformer",
    #    "lora_unet_up_blocks_2_attentions_1_transformer",
    #    "lora_unet_up_blocks_2_attentions_2_transformer",
    #    "lora_unet_up_blocks_3_attentions_0_transformer",
    #    "lora_unet_up_blocks_3_attentions_1_transformer",
    #    "lora_unet_up_blocks_3_attentions_2_transformer",
]


def remove_face_weight(target_path, log_path="log"):
    file_list = os.listdir(target_path)
    file_list = [f for f in file_list if f.endswith(".safetensors")]

    # remove weights of down blocks
    logger.info(f"Remove Blocks Weight for {file_list}.")
    for file in file_list:
        model_file = os.path.join(target_path, file)
        weights = load_file(model_file)
        for key in weights.keys():
            if any([key.startswith(prefix_key) for prefix_key in prefix_keys]):
                weights[key] = torch.zeros_like(weights[key])
        save_file(weights, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for model folder
    parser.add_argument("model_dir", type=str, default="~/lora_auto_train/model/")
    args = parser.parse_args()
    remove_face_weight(args.model_dir)
