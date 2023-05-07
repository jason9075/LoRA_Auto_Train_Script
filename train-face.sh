#!/bin/bash

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

TRAIN_SCRIPT_PATH=~/projects/LoRA_Easy_Training_Scripts/sd_scripts/
TRAIN_DATA_PATH=~/projects/LoRA_Auto_Train_Script
MODEL_NAME="AliceOu"
CONFIG_PATH=./config/face.json
GENDER=female
train_batch_size=$BATCH_SIZE
num_ckpts=1

# check training exist
image_count=$(ls ./train_image/1_face/*.jpg 2>/dev/null | wc -l)
if [ "$image_count" -eq 0 ]; then
    echo "Gen aug image and txt."
    python module/aug_gen_face.py "$GENDER"
fi

image_count=$(ls ./train_image/1_face/*.jpg 2>/dev/null | wc -l)

# for loop json config file key and variable to bash dict
declare -A config_dict
while IFS=':' read -r key value; do
    config_dict[$key]=$value
done < <(jq -r 'to_entries | .[] | .key + ":" + (.value | tostring)' $CONFIG_PATH)

# filter empty value
for i in "${!config_dict[@]}"; do
    if [ "${config_dict[$i]}" = "" ]; then
        unset config_dict[$i]
    fi
done

# filter config_dict with false value and convert true value to empty string
for i in "${!config_dict[@]}"; do
    if [ "${config_dict[$i]}" = "false" ]; then
        unset config_dict[$i]
    elif [ "${config_dict[$i]}" = "true" ]; then
        config_dict[$i]=""
    fi
done

# overwrite some config_dict
config_dict["train_data_dir"]=$TRAIN_DATA_PATH/"${config_dict["train_data_dir"]}"
config_dict["output_dir"]=$TRAIN_DATA_PATH/"${config_dict["output_dir"]}"
config_dict["logging_dir"]=$TRAIN_DATA_PATH/"${config_dict["logging_dir"]}"
config_dict["log_prefix"]="$MODEL_NAME-"
config_dict["network_module"]="networks.lora"
config_dict["output_name"]="$MODEL_NAME"
config_dict["train_batch_size"]="$train_batch_size"
# divide max_train_steps by train_batch_size
max_train_steps="${config_dict["max_train_steps"]}"
max_train_steps=$((max_train_steps / train_batch_size))
config_dict["max_train_steps"]=$max_train_steps
# unset if save_every_n_epochs equal 1
if [ "$num_ckpts" -eq 1 ]; then
    unset config_dict["save_every_n_epochs"]
else
    image_count=$(ls ./train_image/1_$SUBJECT_NAME/*.jpg 2>/dev/null | wc -l)
    image_per_steps=$((image_count / train_batch_size))
    save_every_n_epochs=$((max_train_steps / image_per_steps / num_ckpts))
    config_dict["save_every_n_epochs"]="$save_every_n_epochs"
    echo "save_every_n_epochs: $save_every_n_epochs"
fi

# format config_dict to string
config_dict_string=""
for i in "${!config_dict[@]}"; do
    config_dict_string="$config_dict_string --$i ${config_dict[$i]}"
done

cd $TRAIN_SCRIPT_PATH

accelerate launch \
--num_cpu_threads_per_process=6 \
--num_processes=1 \
--num_machines=1 \
--mixed_precision="no" \
"train_network.py" \
$config_dict_string

# remove down blocks weights
cd -
python module/remove_down_blocks_weights.py ${config_dict["output_dir"]}

# scp model/AutoTrainFace.safetensors pop:/home/jason/stable-diffusion-webui/models/Lora/ 
