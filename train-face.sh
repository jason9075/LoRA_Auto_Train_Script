#!/bin/bash

TRAIN_SCRIPT_PATH=~/projects/LoRA_Easy_Training_Scripts/sd_scripts/
# TRAIN_SCRIPT_PATH=~/projects/kohya_ss-linux/
TRAIN_DATA_PATH=~/projects/LoRA_Auto_Train_Script
MODEL_NAME="ccc-tune"
GENDER=male
max_train_steps=1250
num_ckpts=1

# check training exist
image_count=$(ls ./train_image/1_face/*.jpg 2>/dev/null | wc -l)
if [ "$image_count" -eq 0 ]; then
    echo "Gen Faces."
    python module/gen_face.py "$GENDER"
fi

image_count=$(ls ./train_image/1_face/*.jpg 2>/dev/null | wc -l)

CONFIG_PATH=./config/face.json
# 讀取JSON文件中的設置
pretrained_model_name_or_path=$(jq -r '.pretrained_model_name_or_path' $CONFIG_PATH)
logging_dir=$(jq -r '.logging_dir' $CONFIG_PATH)
train_data_dir=$(jq -r '.train_data_dir' $CONFIG_PATH)
reg_data_dir=$(jq -r '.reg_data_dir' $CONFIG_PATH)
output_dir=$(jq -r '.output_dir' $CONFIG_PATH)
resolution=$(jq -r '.resolution' $CONFIG_PATH)
learning_rate=$(jq -r '.learning_rate' $CONFIG_PATH)
lr_scheduler=$(jq -r '.lr_scheduler' $CONFIG_PATH)
lr_warmup=$(jq -r '.lr_warmup' $CONFIG_PATH)
train_batch_size=$(jq -r '.train_batch_size' $CONFIG_PATH)
mixed_precision=$(jq -r '.mixed_precision' $CONFIG_PATH)
save_precision=$(jq -r '.save_precision' $CONFIG_PATH)
seed=$(jq -r '.seed' $CONFIG_PATH)
num_cpu_threads_per_process=$(jq -r '.num_cpu_threads_per_process' $CONFIG_PATH)
caption_extension=$(jq -r '.caption_extension' $CONFIG_PATH)
full_fp16=$(jq -r '.full_fp16' $CONFIG_PATH)
no_token_padding=$(jq -r '.no_token_padding' $CONFIG_PATH)
stop_text_encoder_training=$(jq -r '.stop_text_encoder_training' $CONFIG_PATH)
save_model_as=$(jq -r '.save_model_as' $CONFIG_PATH)
save_state=$(jq -r '.save_state' $CONFIG_PATH)
resume=$(jq -r '.resume' $CONFIG_PATH)
text_encoder_lr=$(jq -r '.text_encoder_lr' $CONFIG_PATH)
unet_lr=$(jq -r '.unet_lr' $CONFIG_PATH)
network_dim=$(jq -r '.network_dim' $CONFIG_PATH)
max_data_loader_n_workers=$(jq -r '.max_data_loader_n_workers' $CONFIG_PATH)
network_alpha=$(jq -r '.network_alpha' $CONFIG_PATH)
training_comment=$(jq -r '.training_comment' $CONFIG_PATH)
lr_scheduler_num_cycles=$(jq -r '.lr_scheduler_num_cycles' $CONFIG_PATH)
lr_scheduler_power=$(jq -r '.lr_scheduler_power' $CONFIG_PATH)
persistent_data_loader_workers=$(jq -r '.persistent_data_loader_workers' $CONFIG_PATH)
optimizer_type=$(jq -r '.optimizer_type' $CONFIG_PATH)
optimizer_args=$(jq -r '.optimizer_args' $CONFIG_PATH)
noise_offset=$(jq -r '.noise_offset' $CONFIG_PATH)
keep_tokens=$(jq -r '.keep_tokens' $CONFIG_PATH)
rules=$(jq -r '.rules' $CONFIG_PATH)

# calculate save_every_n_epochs 
max_train_steps=$((max_train_steps / train_batch_size))
image_per_steps=$((image_count / train_batch_size))
# if num_ckpt > 1, save_every_n_epochs = max_train_steps / image_per_steps / num_ckpt else 0
if [ "$num_ckpts" -gt 1 ]; then
    save_every_n_epochs=$((max_train_steps / image_per_steps / num_ckpts))
else
    save_every_n_epochs=0
fi

cd $TRAIN_SCRIPT_PATH


# if save_n_epochs = 0, dont't use save_every_n_epochs
if [ "$save_every_n_epochs" -eq 0 ]; then
    accelerate launch \
    --num_cpu_threads_per_process=6 \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision="no" \
    "train_network.py" \
    --pretrained_model_name_or_path="$pretrained_model_name_or_path" \
    --train_data_dir="$TRAIN_DATA_PATH/$train_data_dir" \
    --resolution="$resolution" \
    --output_dir="$TRAIN_DATA_PATH/$output_dir" \
    --logging_dir="$TRAIN_DATA_PATH/$logging_dir" \
    --network_alpha="$network_alpha" \
    --save_model_as="$save_model_as" \
    --network_module="networks.lora" \
    --text_encoder_lr="$text_encoder_lr" \
    --unet_lr="$unet_lr" \
    --network_dim="$network_dim" \
    --output_name="$MODEL_NAME" \
    --learning_rate="$learning_rate" \
    --lr_scheduler="$lr_scheduler" \
    --rules="$rules" \
    --train_batch_size="$train_batch_size" \
    --max_train_steps="$max_train_steps" \
    --mixed_precision="$mixed_precision" \
    --save_precision="$save_precision" \
    --seed="$seed" \
    --caption_extension="$caption_extension" \
    --optimizer_type=$optimizer_type \
    --max_data_loader_n_workers="$max_data_loader_n_workers" \
    --training_comment="$training_comment" \
    --keep_tokens="$keep_tokens" \
    --bucket_reso_steps=64 \
    --mem_eff_attn \
    --xformers \
    --bucket_no_upscale
else
    accelerate launch \
    --num_cpu_threads_per_process=6 \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision="no" \
    "train_network.py" \
    --pretrained_model_name_or_path="$pretrained_model_name_or_path" \
    --train_data_dir="$TRAIN_DATA_PATH/$train_data_dir" \
    --resolution="$resolution" \
    --output_dir="$TRAIN_DATA_PATH/$output_dir" \
    --logging_dir="$TRAIN_DATA_PATH/$logging_dir" \
    --network_alpha="$network_alpha" \
    --save_model_as="$save_model_as" \
    --network_module="networks.lora" \
    --text_encoder_lr="$text_encoder_lr" \
    --unet_lr="$unet_lr" \
    --network_dim="$network_dim" \
    --output_name="$MODEL_NAME" \
    --learning_rate="$learning_rate" \
    --lr_scheduler="$lr_scheduler" \
    --rules="$rules" \
    --train_batch_size="$train_batch_size" \
    --max_train_steps="$max_train_steps" \
    --save_every_n_epochs="$save_every_n_epochs" \
    --mixed_precision="$mixed_precision" \
    --save_precision="$save_precision" \
    --seed="$seed" \
    --caption_extension="$caption_extension" \
    --optimizer_type=$optimizer_type \
    --max_data_loader_n_workers="$max_data_loader_n_workers" \
    --training_comment="$training_comment" \
    --keep_tokens="$keep_tokens" \
    --bucket_reso_steps=64 \
    --mem_eff_attn \
    --xformers \
    --bucket_no_upscale
fi

# remove down blocks weights
cd -
python module/remove_down_blocks_weights.py "$output_dir"

# scp model/AutoTrainFace.safetensors pop:/home/jason/stable-diffusion-webui/models/Lora/ 
