export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=SEHS

for pred_len in 96
do
for promot in  700
do
for epsilon in 0.3
do
for decay in 0.4
do
for drop in 0.1
do
python run.py \
    --model $model \
    --patch_size 16 \
    --d_layers 1 \
    --top_k 5 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --prompt_length $promot \
    --num_blocks 3 \
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
    --output_loss smooth_l1 \
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 100 \
    --patience 5 \
    --data ETTh1 \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \

echo '====================================================================================================================='
done
done
done
done
done
