# CMIE efficiency benchmark — 7 datasets x 6 merge_layers, pred_len=336, batch_size=32
# Usage: gpu=0 bash scripts/efficiency/run_cmie_all.sh

common_args="\
    --is_training 1 \
    --task_name long_term_forecast \
    --model CMIE \
    --seq_len 96 \
    --pred_len 336 \
    --label_len 0 \
    --d_model 768 \
    --n_heads 4 \
    --d_layers 1 \
    --gpt_layers 3 \
    --use_ffc 1 \
    --use_channel_aware_prompt 1 \
    --use_llm 1 \
    --lambda_sim 0.1 \
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

for merge in 1 2 3 4 5 6; do

    # ==================== ETTh1 ====================
    echo "==================== CMIE merge${merge}: ETTh1 ===================="
    python -u run.py $common_args --merge_layers $merge --data ETTh1 --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --model_id CMIEeff_merge${merge}_ETTh1

    # ==================== ETTh2 ====================
    echo "==================== CMIE merge${merge}: ETTh2 ===================="
    python -u run.py $common_args --merge_layers $merge --data ETTh2 --root_path ./datasets/ETT-small/ --data_path ETTh2.csv --model_id CMIEeff_merge${merge}_ETTh2

    # ==================== ETTm1 ====================
    echo "==================== CMIE merge${merge}: ETTm1 ===================="
    python -u run.py $common_args --merge_layers $merge --data ETTm1 --root_path ./datasets/ETT-small/ --data_path ETTm1.csv --model_id CMIEeff_merge${merge}_ETTm1

    # ==================== ETTm2 ====================
    echo "==================== CMIE merge${merge}: ETTm2 ===================="
    python -u run.py $common_args --merge_layers $merge --data ETTm2 --root_path ./datasets/ETT-small/ --data_path ETTm2.csv --model_id CMIEeff_merge${merge}_ETTm2

    # ==================== Weather ====================
    echo "==================== CMIE merge${merge}: Weather ===================="
    python -u run.py $common_args --merge_layers $merge --data Weather --root_path ./datasets/weather/ --data_path weather.csv --model_id CMIEeff_merge${merge}_Weather

    # ==================== ECL ====================
    echo "==================== CMIE merge${merge}: ECL ===================="
    python -u run.py $common_args --merge_layers $merge --data ECL --root_path ./datasets/electricity/ --data_path electricity.csv --model_id CMIEeff_merge${merge}_ECL

    # ==================== Traffic ====================
    echo "==================== CMIE merge${merge}: Traffic ===================="
    python -u run.py $common_args --merge_layers $merge --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv --model_id CMIEeff_merge${merge}_Traffic

done

echo "CMIE efficiency done!"
