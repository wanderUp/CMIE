export CUDA_VISIBLE_DEVICES=4

seq_len=96
model=SEHS

for pred_len in 192
do
python run.py \
    --model $model \
    --patch_size 8 \
    --d_layers 1 \
    --top_k 5 \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.3 \
    --limit_len 128 \
    --num_blocks 3 \
    --dropout 0.1 \
    --gpt_layers 4 \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 64 \
    --d_model 768 \
    --n_heads 2 \
    --output_loss mse \
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 100 \
    --patience 5 \
    --data ETTh1 \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \

echo '====================================================================================================================='
done

