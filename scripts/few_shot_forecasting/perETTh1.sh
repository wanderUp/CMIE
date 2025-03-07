export CUDA_VISIBLE_DEVICES=1

seq_len=96
model=CALF


for percent in 10 20 30 40
do
for pred_len in 96 192 336 720
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name few_shot_forecast \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --timesnet_k 4 \
    --patience 10 \
    --percent $percent \
    --task_w 1 \
    --feature_w 0.2 \
    --output_w 0.2 \
    --task_loss l1

echo '====================================================================================================================='
done
done