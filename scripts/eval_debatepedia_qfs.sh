# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# you need to specify data_dir, output_dir and model_name_or_path
export OUTPUT_DIR=/data/jixin/query_aug_model/output_4
export DATA_DIR=/data/jixin/query_aug_model/debatepedia

CUDA_VISIBLE_DEVICES=1 python eval_qfs.py --model_name $OUTPUT_DIR/best_tfmr \
    --input_dir $DATA_DIR \
    --save_path debatepedia_test_generations.txt \
    --reference_path $DATA_DIR/test_summary \
    --score_path debatepedia_rouge.json \
    --task summarization \
    --n_obs 100 \
    --device cuda \
    --bs 110 \
    --raw \
    --baseline
