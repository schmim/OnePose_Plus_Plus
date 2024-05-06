#!/bin/bash
set -e

for obj_id in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
    do
        echo "obj_id:$obj_id"
        python parse_lm_real_data.py \
            --obj_id $obj_id \
            --assign_onepose_id 08${obj_id} \
            --data_base_dir "data/lm" \
            --split train

        python parse_lm_real_data.py \
            --obj_id $obj_id \
            --assign_onepose_id 08${obj_id} \
            --data_base_dir "data/lm" \
            --split val \
            # --use_yolo_box
    done