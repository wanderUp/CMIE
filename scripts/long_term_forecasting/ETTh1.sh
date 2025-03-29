export CUDA_VISIBLE_DEVICES=1

seq_len=96
model=SEHS

for pred_len in 96
do
python run.py \
    --model $model \
    --patch_size 16 \
    --d_layers 1 \
    --top_k 5 \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_blocks 3 \
    --dropout 0.1 \
    --gpt_layers 6 \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 256 \
    --d_model 768 \
    --n_heads 4 \
    --output_loss smooth_l1 \
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

