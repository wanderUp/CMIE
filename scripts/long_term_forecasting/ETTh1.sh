export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=SEHS

for pred_len in 96
do
python run.py \
    --model $model \
    --patch_size 16 \
    --d_layers 1 \
    --is_training 1 \
    --gpt_layers 4 \
    --dropout 0.3 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 256 \
    --d_model 768 \
    --n_heads 2 \
    --output_loss mse \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 100 \
    --patience 10 \
    --data ETTh1 \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \

echo '====================================================================================================================='
done

