export CUDA_VISIBLE_DEVICES=6

seq_len=96
model=CALF


for k in 1 8 16
do
for pair in "(1,0)" "(1,0.2)"
do
for pred_len in 96 192 336 720
do
for lora_alpha in 16 32
do
for lora_dropout in 0.1 0.2
do
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
python run.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id weather_$model'_'$seq_len'_'$pred_len \
    --data Weather \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --lradj type3 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha $lora_alpha\
    --lora_dropout $lora_dropout \
    --patience 5 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --timesnet_k $k \
    --task_w $task_w \
    --feature_w $output_w \
    --output_w $output_w

echo '====================================================================================================================='
done
done
done
done
done

