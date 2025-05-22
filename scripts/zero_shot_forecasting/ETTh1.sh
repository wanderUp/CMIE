export CUDA_VISIBLE_DEVICES=2

seq_len=96
model=SEHS
for pred_len in 96
do
for target_data in ETTm1
do
for prompt in  256 512 700
do
for epsilon in 0.1 0.2 0.3
do
for decay in 0. 0.2 0.3 0.4
do
for drop in 0.1 0.2 0.3
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name zero_shot_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --prompt_length $prompt \
    --dropout $drop \
    --epsilon $epsilon \
    --weight_decay $decay \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.001 \
    --lradj type1 \
    --train_epochs 5 \
    --d_model 768 \
    --n_heads 4 \
    --dropout 0.3 \
    --gpt_layer 4 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --zero_shot 1 \
    --target_data $target_data \

echo '====================================================================================================================='
done
done
done
done
done
done
