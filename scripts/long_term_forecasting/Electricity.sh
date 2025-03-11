

seq_len=96
model=SEHS

for pred_len in 96
do

python test.py \
    --gpu 3 \
    --itr 1 \
    --patch_size 16\
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --data custom \
    --is_training 1 \
    --d_layers 2 \
    --task_name long_term_forecast \
    --model_id ECL_$model'_'$seq_len'_'$pred_len \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --train_epochs 1 \
    --d_model 768 \
    --n_heads 4 \
    --model $model \
    --patience 5 \
    --output_loss smooth_l1

echo '====================================================================================================================='
done

