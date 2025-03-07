export CUDA_VISIBLE_DEVICES=0

model_name=CALF

for k in 2 4
do
for lora_alpha in 16 32
do
for lora_dropout in 0.1 0.2
do
for pair in "(1,0)" "(1,0.5)" "(1,0.2)"
do
# 解析元组字符串，将括号去掉，并按逗号分割
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
for interval in 'Monthly' 'Quarterly' 'Yearly' 'Daily' 'Hourly' 'Weekly'
do
python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./datasets/m4 \
    --seasonal_patterns $interval \
    --model_id m4_$interval \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --train_epochs 200 \
    --batch_size 256 \
    --d_model 768 \
    --d_ff 768 \
    --n_heads 4 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --timesnet_k $k \
    --patience 10 \
    --gpt_layers 6 \
    --task_loss smape \
    --output_loss mase \
    --feature_loss smooth_l1 \
    --task_w $task_w \
    --feature_w $output_w \
    --output_w $output_w \

echo '====================================================================================================================='
done
done
done
done
done
