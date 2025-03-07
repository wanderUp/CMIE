export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=CALF


for percent in 10 20 30 40
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
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
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

