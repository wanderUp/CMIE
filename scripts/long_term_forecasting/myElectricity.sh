export CUDA_VISIBLE_DEVICES=9

seq_len=96
model=CALF

for pair in "(1,0)" "(1,0.5)" "(1,0.2)"
do
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
for lora in "(16,0.2)" "(32,0.1)" "(32,0.2)"
do
lora_alpha=$(echo $lora | cut -d',' -f1 | tr -d '()')
lora_dropout=$(echo $lora | cut -d',' -f2 | tr -d '()')
for pred_len in 96 192 336 720
do
python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ECL_$model'_'$seq_len'_'$pred_len \
    --data ECL \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --patience 5 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --timesnet_k $k \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --task_w $task_w \
    --feature_w $output_w \
    --output_w $output_w

echo '====================================================================================================================='
done
done
done

