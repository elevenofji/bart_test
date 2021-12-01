# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# you need to specify data_dir, output_dir and model_name_or_path
export DATA_DIR=/data/jixin/query_aug_model/debatepedia
export OUTPUT_DIR=/data/jixin/query_aug_model/output_4

CUDA_VISIBLE_DEVICES=1 python train_qfs.py \
    --data_dir $DATA_DIR \
    --baseline \
    --raw \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path /home/jx/pretrained_models/bart-large-xsum \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --max_source_length 142 \
    --max_target_length 48 \
    --val_max_target_length 48 \
    --test_max_target_length 48 \
    --freeze_embeds \
    --train_batch_size 60 \
    --eval_batch_size 60 \
    --num_workers 8 \
    --gradient_accumulation_steps 4
