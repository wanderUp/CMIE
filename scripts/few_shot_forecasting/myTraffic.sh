export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=CALF

for k in 2 4
do
for lora_alpha in 16 32
do
for lora_dropout in 0.1 0.2
do
for pair in "(1,1)" "(1,0)" "(1,0.5)" "(1,0.2)" "(0.5,1)"
do
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
for percent in 10
do
for pred_len in 96 192 336 720
do
python run.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --is_training 1 \
    --task_name few_shot_forecast \
    --model_id traffic_$model'_'$seq_len'_'$pred_len \
    --data Traffic \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
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
    --output_w $output_w \
    --percent $percent
echo '====================================================================================================================='
done
done
done
done
done
done
