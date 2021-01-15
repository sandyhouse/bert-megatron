DATA_DIR='./data'
PYTHONPATH=$PYTHONPATH:../../PaddleNLP

# Set to ip addresses of your machines
IPS="10.0.0.1,10.0.0.2"

python3 -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --ips=$IPS \
    ./run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32  \
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
