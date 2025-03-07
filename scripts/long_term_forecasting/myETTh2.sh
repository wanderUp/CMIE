export CUDA_VISIBLE_DEVICES=1

seq_len=96
model=CALF


for lora_alpha in 16 32
do
for lora_dropout in 0.1 0.2
do
for pair in "(1,1)" "(1,0)" "(1,0.5)" "(1,0.2)" "(0.5,1)"
do
for pred_len in 96 192 336 720
do
# 解析元组字符串，将括号去掉，并按逗号分割
task_w=$(echo $pair | cut -d',' -f1 | tr -d '()')
output_w=$(echo $pair | cut -d',' -f2 | tr -d '()')
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len \
    --data ETTh2 \
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
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --patience 10 \
    --timesnet_k 4 \
    --task_w $task_w \
    --feature_w $output_w \
    --output_w $output_w

echo '====================================================================================================================='
done
done
done
done


