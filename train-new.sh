#!/bin/bash

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

TRAIN_SCRIPT_PATH=~/projects/LoRA_Easy_Training_Scripts/sd_scripts/
TRAIN_DATA_PATH=~/projects/LoRA_Auto_Train_Script
TARGET_LORA_DIR=~/projects/stable-diffusion-webui/models/Lora/
CONFIG_PATH=./config/face.json

IS_CP_FILE=true

folder="/home/jason9075/projects/LoRA_Auto_Train_Script/person/new"
train_batch_size=$BATCH_SIZE
num_ckpts=1

# train func
train(){
    echo "Train $1 $2"

    #splite $2 text to name and gender
    IFS='_' read -r MODEL_NAME GENDER <<< "$2"

    # clean origin_image and train_image jpg png jpeg
    rm -rf $TRAIN_DATA_PATH/origin_image/*.jpg
    rm -rf $TRAIN_DATA_PATH/origin_image/*.png
    rm -rf $TRAIN_DATA_PATH/origin_image/*.jpeg
    rm -rf $TRAIN_DATA_PATH/train_image/1_face/*.jpg
    rm -rf $TRAIN_DATA_PATH/train_image/1_face/*.png
    rm -rf $TRAIN_DATA_PATH/train_image/1_face/*.jpeg
    rm -rf $TRAIN_DATA_PATH/train_image/1_face/*.txt

    # cp folder_image to origin_image
    cp -r $1/*.jpg $TRAIN_DATA_PATH/origin_image/
    cp -r $1/*.png $TRAIN_DATA_PATH/origin_image/
    cp -r $1/*.jpeg $TRAIN_DATA_PATH/origin_image/

    echo "Gen Faces."
    python module/aug_gen_face.py "$GENDER"

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
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision="no" \
    "train_network.py" \
    $config_dict_string

    # remove down blocks weights
    cd -
    python module/remove_down_blocks_weights.py ${config_dict["output_dir"]}

    # copy model to webui
    if [ "$IS_CP_FILE" = true ]; then
        cp ${config_dict["output_dir"]}/$MODEL_NAME.safetensors $TARGET_LORA_DIR
    fi

}
# scp model/AutoTrainFace.safetensors pop:/home/jason/stable-diffusion-webui/models/Lora/ 

# Loop through each file in the folder
for item in "$folder"/*
do
  if [ -d "$item" ]
  then
    # get folder name
    folder_name=$(basename "$item")

    # train folder data
    train $item $folder_name
  fi
done

