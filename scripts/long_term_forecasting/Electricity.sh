

seq_len=96
model=SEHS

for pred_len in 96
do
python run.py \
    --gpu 3 \
    --model $model \
    --patch_size 16 \
    --d_layers 2 \
    --is_training 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 64 \
    --d_model 256 \
    --n_heads 4 \
    --output_loss smooth_l1 \
    --learning_rate 0.0005 \
    --itr 1 \
    --train_epochs 1 \
    --patience 5 \
    --data custom \
    --model_id ECL_$model'_'$seq_len'_'$pred_len \
    --task_name long_term_forecast \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv

echo '====================================================================================================================='
done

