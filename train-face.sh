#!/bin/bash

max_train_steps=1250
num_ckpts=5

image_count=$(ls ./train_image/1_face/*.jpg 2>/dev/null | wc -l)
# 讀取JSON文件中的設置
pretrained_model_name_or_path=$(jq -r '.pretrained_model_name_or_path' config.json)
logging_dir=$(jq -r '.logging_dir' config.json)
train_data_dir=$(jq -r '.train_data_dir' config.json)
reg_data_dir=$(jq -r '.reg_data_dir' config.json)
output_dir=$(jq -r '.output_dir' config.json)
max_resolution=$(jq -r '.max_resolution' config.json)
learning_rate=$(jq -r '.learning_rate' config.json)
lr_scheduler=$(jq -r '.lr_scheduler' config.json)
lr_warmup=$(jq -r '.lr_warmup' config.json)
train_batch_size=$(jq -r '.train_batch_size' config.json)
mixed_precision=$(jq -r '.mixed_precision' config.json)
save_precision=$(jq -r '.save_precision' config.json)
seed=$(jq -r '.seed' config.json)
num_cpu_threads_per_process=$(jq -r '.num_cpu_threads_per_process' config.json)
caption_extension=$(jq -r '.caption_extension' config.json)
full_fp16=$(jq -r '.full_fp16' config.json)
no_token_padding=$(jq -r '.no_token_padding' config.json)
stop_text_encoder_training=$(jq -r '.stop_text_encoder_training' config.json)
save_model_as=$(jq -r '.save_model_as' config.json)
save_state=$(jq -r '.save_state' config.json)
resume=$(jq -r '.resume' config.json)
text_encoder_lr=$(jq -r '.text_encoder_lr' config.json)
unet_lr=$(jq -r '.unet_lr' config.json)
network_dim=$(jq -r '.network_dim' config.json)
output_name=$(jq -r '.output_name' config.json)
max_data_loader_n_workers=$(jq -r '.max_data_loader_n_workers' config.json)
network_alpha=$(jq -r '.network_alpha' config.json)
training_comment=$(jq -r '.training_comment' config.json)
lr_scheduler_num_cycles=$(jq -r '.lr_scheduler_num_cycles' config.json)
lr_scheduler_power=$(jq -r '.lr_scheduler_power' config.json)
persistent_data_loader_workers=$(jq -r '.persistent_data_loader_workers' config.json)
optimizer=$(jq -r '.optimizer' config.json)
optimizer_args=$(jq -r '.optimizer_args' config.json)
noise_offset=$(jq -r '.noise_offset' config.json)

# calculate save_every_n_epochs 
save_every_n_epochs=$((max_train_steps / image_count / num_ckpts))

cd ~/projects/kohya_ss-linux/

accelerate launch --num_cpu_threads_per_process="$num_cpu_threads_per_process" \
--num_processes=1 \
--num_machines=1 \
--mixed_precision="no" \
"train_network.py" \
--pretrained_model_name_or_path="$pretrained_model_name_or_path" \
--train_data_dir="$train_data_dir" \
--resolution="$max_resolution" \
--output_dir="$output_dir" \
--logging_dir="$logging_dir" \
--network_alpha="$network_alpha" \
--save_model_as="$save_model_as" \
--network_module="networks.lora" \
--text_encoder_lr="$text_encoder_lr" \
--unet_lr="$unet_lr" \
--network_dim="$network_dim" \
--output_name="$output_name" \
--learning_rate="$learning_rate" \
--lr_scheduler="$lr_scheduler" \
--train_batch_size="$train_batch_size" \
--max_train_steps="$max_train_steps" \
--save_every_n_epochs="$save_every_n_epochs" \
--mixed_precision="$mixed_precision" \
--save_precision="$save_precision" \
--seed="$seed" \
--caption_extension="$caption_extension" \
--optimizer_type=$optimizer \
--max_data_loader_n_workers="$max_data_loader_n_workers" \
--training_comment="$training_comment" \
--bucket_reso_steps=64 \
--mem_eff_attn \
--xformers \
--bucket_no_upscale

# remove down blocks weights
cd -
python remove_down_blocks_weights.py "$output_dir"

# scp model/AutoTrainFace.safetensors pop:/home/jason/stable-diffusion-webui/models/Lora/ 
