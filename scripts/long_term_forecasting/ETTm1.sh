gpu=6

seq_len=96
model=SEHS

for pred_len in 336
do
for prompt in 512
do
for epsilon in 0.3
do
for decay in 0.3
do
for drop in 0.1
do
python run.py \
    --gpu $gpu \
    --model $model \
    --d_layers 1 \
    --top_k 5 \
    --r 4 \
    --merge_layers 2 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --prompt_length $prompt \
    --gpt_layers 2 \
    --dropout $drop \
    --epsilon $epsilon \
    --weight_decay $decay \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 128 \
    --d_model 768 \
    --n_heads 4 \
    --output_loss mse \
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 100 \
    --patience 5 \
    --data ETTm1 \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \

echo '====================================================================================================================='
done
done
done
done
done