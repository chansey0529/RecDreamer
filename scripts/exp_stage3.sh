#!/bin/bash

gpu=$1



# Read variables
read_variable() {
    local dict_name=$1
    local key=$2

    # Extract variables
    value=$(jq -r ".${dict_name}.${key}" classifier_configs.json | sed "s/'/'\"'\"'/g")

    echo "$value"
}

dict_names=("redpanda")


for dict_name in "${dict_names[@]}"; do
    echo "Reading variables for dictionary: $dict_name"
    
    declare -A keys=(
        ["text"]="default text"
        ["text_aux"]="default auxilary text"
        ["cfg"]="7.5"
        ["stage2_pth"]=""
        ["dth"]="0.3"
    )
    
    for key in "${!keys[@]}"; do
        default_value="${keys[$key]}"
        value=$(read_variable "$dict_name" "$key" "$default_value")
        if [ "$value" == "null" ]; then
            value="$default_value"
        fi
        eval "${key}='${value}'"
        echo "Value of $key in $dict_name: $value"
    done

    echo $text
    echo $text_aux
    echo "$text, $text_aux"
    echo $stage2_pth
    
    TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu python main.py --text "$text $text_aux" --iters 30000 --scale 7.5 --dmtet --mesh_idx 0 --init_ckpt "$stage2_pth" --density_thresh $dth --finetune True --lambda_normal 5000 --workspace exp-stage3-usd/ --seed 0 --simple_dir --note "$dict_name" --name "pose_3d" --cls_ckpt "exp-classifier/$dict_name" --usd --seed 0
done