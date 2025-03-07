export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=CALF


for percent in 10 20 30 40
do
for pred_len in 96 192 336 720
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name few_shot_forecast \
    --model_id ETTh2_$seq_len'_'$pred_len \
    --data ETTh2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --patience 10 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --timesnet_k 2 \
    --percent $percent \
    --task_w 1 \
    --feature_w 0.2 \
    --output_w 0.2

echo '====================================================================================================================='
done
done
