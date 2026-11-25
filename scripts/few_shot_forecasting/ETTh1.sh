gpu=7

seq_len=96
model=CMIE

for pred_len in 96 192 336 720
do
for epsilon in 0.3
do
for decay in 0
do
for drop in 0.1
do
python run.py \
    --gpu $gpu \
    --model $model \
    --merge_threshold 0.5 \
    --d_layers 1 \
    --top_k 5 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --gpt_layers 6 \
    --merge_layers 1 \
    --dropout $drop \
    --epsilon $epsilon \
    --weight_decay $decay \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 32 \
    --d_model 768 \
    --n_heads 4 \
    --output_loss mse\
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 100 \
    --patience 5 \
    --data ETTh1 \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --percent 10

echo '====================================================================================================================='
done
done
done
done
