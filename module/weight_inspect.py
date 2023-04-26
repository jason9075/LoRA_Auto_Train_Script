from safetensors.torch import load_file
import os
import numpy as np
import torch
import logging


logger = logging.getLogger()
# 設定模型檔資料夾和要計算的 prefix key
prefix_keys = [
    "lora_te_text_model_encoder_layers",
    "lora_unet_down_blocks_0_attentions_0_transformer",
    "lora_unet_down_blocks_0_attentions_1_transformer",
    "lora_unet_down_blocks_1_attentions_0_transformer",
    "lora_unet_down_blocks_1_attentions_1_transformer",
    "lora_unet_down_blocks_2_attentions_0_transformer",
    "lora_unet_down_blocks_2_attentions_1_transformer",
    "lora_unet_mid_block_attentions_0_transformer",
    "lora_unet_up_blocks_1_attentions_0_transformer",
    "lora_unet_up_blocks_1_attentions_1_transformer",
    "lora_unet_up_blocks_1_attentions_2_transformer",
    "lora_unet_up_blocks_2_attentions_0_transformer",
    "lora_unet_up_blocks_2_attentions_1_transformer",
    "lora_unet_up_blocks_2_attentions_2_transformer",
    "lora_unet_up_blocks_3_attentions_0_transformer",
    "lora_unet_up_blocks_3_attentions_1_transformer",
    "lora_unet_up_blocks_3_attentions_2_transformer",
]


def write_mean_var_file(model_dir, output_dir):
    report_name = "weight_mean_var.txt"
    # 設定 numpy 輸出格式
    np.set_printoptions(linewidth=np.inf, formatter={"float": "{: 0.5f}".format})

    # 遍歷每個模型檔並計算權重平均值和變異數
    # Collect .safetenors models and sorting
    file_list = os.listdir(model_dir)
    file_list = [f for f in file_list if f.endswith(".safetensors")]
    file_list = sorted(file_list)

    with open(os.path.join(output_dir, report_name), "w") as file:
        logger.info(f"Writing mean and var to {file_list}")
        for model_name in file_list:
            model_file = os.path.join(model_dir, model_name)
            results = calculate_weights_stats(model_file, prefix_keys)

            # 輸出計算結果
            file.write(f"Results for model '{model_file}':\n")
            if not np.isnan(results).any():
                file.write(f"avg {results[0, :]}\n")
                file.write(f"var {results[1, :]}\n")
            else:
                file.write("avg nan\n")
                file.write("var nan\n")


def calculate_weights_stats(model_file, prefix_keys):
    # 讀取 .safetensors 模型檔
    weights = load_file(model_file)

    # 計算每個 prefix key 底下權重的平均值和變異數
    results = np.zeros((2, len(prefix_keys)))
    for idx, prefix_key in enumerate(prefix_keys):
        weights_subset = [w for k, w in weights.items() if k.startswith(prefix_key)]
        weights_subset = [
            w.double() if w.dtype == torch.bfloat16 or w.dtype == torch.float16 else w
            for w in weights_subset
        ]
        weights_subset = [w.reshape(-1) for w in weights_subset]
        if weights_subset == []:
            continue
        weights_subset = np.concatenate(weights_subset)

        results[0, idx] = np.mean(weights_subset)
        results[1, idx] = np.var(weights_subset)

    return results


if __name__ == "__main__":
    model_dir = "output/model/"
    output_dir = "log/"
    write_mean_var_file(model_dir, output_dir)
