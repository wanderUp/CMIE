# CALF efficiency benchmark — 7 datasets, pred_len=336, batch_size=32
# Usage: gpu=0 bash scripts/efficiency/run_calf_all.sh

common_args="\
    --is_training 1 \
    --task_name long_term_forecast \
    --model CALF \
    --seq_len 96 \
    --pred_len 336 \
    --label_len 0 \
    --d_model 768 \
    --n_heads 4 \
    --d_layers 1 \
    --gpt_layers 3 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_epochs 5 \
    --patience 5 \
    --dropout 0.1 \
    --epsilon 0.3 \
    --weight_decay 0 \
    --output_loss mse \
    --itr 1 \
    --gpu $gpu"

# ==================== ETTh1 ====================
echo "==================== CALF: ETTh1 ===================="
python -u run.py $common_args --data ETTh1 --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --model_id CALFeff_ETTh1

# ==================== ETTh2 ====================
echo "==================== CALF: ETTh2 ===================="
python -u run.py $common_args --data ETTh2 --root_path ./datasets/ETT-small/ --data_path ETTh2.csv --model_id CALFeff_ETTh2

# ==================== ETTm1 ====================
echo "==================== CALF: ETTm1 ===================="
python -u run.py $common_args --data ETTm1 --root_path ./datasets/ETT-small/ --data_path ETTm1.csv --model_id CALFeff_ETTm1

# ==================== ETTm2 ====================
echo "==================== CALF: ETTm2 ===================="
python -u run.py $common_args --data ETTm2 --root_path ./datasets/ETT-small/ --data_path ETTm2.csv --model_id CALFeff_ETTm2

# ==================== Weather ====================
echo "==================== CALF: Weather ===================="
python -u run.py $common_args --data Weather --root_path ./datasets/weather/ --data_path weather.csv --model_id CALFeff_Weather

# ==================== ECL ====================
echo "==================== CALF: ECL ===================="
python -u run.py $common_args --data ECL --root_path ./datasets/electricity/ --data_path electricity.csv --model_id CALFeff_ECL

# ==================== Traffic ====================
echo "==================== CALF: Traffic ===================="
python -u run.py $common_args --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv --model_id CALFeff_Traffic

echo "CALF efficiency done!"
