gpu=3

seq_len=96
model=CMIE
for pred_len in 96
do
for epsilon in 0.3
do
for decay in 0.1
do
for drop in 0.1
do
python run.py \
    --gpu $gpu \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model $model \
    --merge_layers 2 \
    --d_layers 1 \
    --top_k 5 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --gpt_layers 2 \
    --dropout $drop \
    --epsilon $epsilon \
    --weight_decay $decay \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 32 \
    --d_model 768 \
    --n_heads 4 \
    --output_loss mse \
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 100 \
    --patience 5 \
    --data Traffic \
    --model_id Traffic_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \

echo '====================================================================================================================='
done
done
done
done
