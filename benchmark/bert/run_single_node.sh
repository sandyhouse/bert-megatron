DATA_DIR='/ssd1/bert_data/data'
export PYTHONPATH=$PYTHONPATH:/workspace/issue_test/ernie_3_mp
export PADDLE_WITH_GLOO=0

rm -rf log
python3 -m paddle.distributed.launch \
    --gpus 0,1,2,3 \
    ./run_pretrain.py \
    --num_partitions=4 \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 128 \
    --use_amp False \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir $DATA_DIR \
    --output_dir ./tmp2/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 140
