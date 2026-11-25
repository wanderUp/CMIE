gpu=5

seq_len=96
model=CMIE
for pred_len in 96
do
for target_data in ETTm2
do
for epsilon in 0.3
do
for decay in 0.1
do
for drop in 0.1
do
python run.py \
    --gpu $gpu \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name zero_shot_forecast \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len \
    --d_layers 1 \
    --data ETTh2 \
    --seq_len $seq_len \
    --merge_layers 1 \
    --dropout $drop \
    --epsilon $epsilon \
    --weight_decay $decay \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.001 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --gpt_layers 2 \
    --itr 1 \
    --model $model \
    --top_k 5 \
    --r 4 \
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
