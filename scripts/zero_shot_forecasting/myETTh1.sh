export CUDA_VISIBLE_DEVICES=9

seq_len=96
model=CALF

for target_data in ETTm1 ETTm2
do
for k in 2 4
do
for pair in "(1,0)" "(1,0.5)" "(1,0.2)"
do
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
for lora_alpha in 16 32
do
for lora_dropout in 0.1 0.2
do
for pred_len in 96 192 336 720
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.001 \
    --lradj type1 \
    --train_epochs 5 \
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
    --timesnet_k $k \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --task_w $task_w \
    --feature_w $output_w \
    --output_w $output_w \
    --patience 1 \
    --zero_shot 1 \
    --target_data $target_data \

echo '====================================================================================================================='
done
done
done
done
done
done