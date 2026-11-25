#!/usr/bin/env bash
# =============================================================================
# 效率实验：Traffic (862 通道)，batch_size=24（24G 显存内），全部重跑
#   1. CALF  2. CMIE无CM  3. CM无patch  4. CM+patch
#
# 用法: gpu=0 bash scripts/efficiency/run_cm_cost_traffic.sh
# =============================================================================

GPU=${gpu:-0}
echo "GPU = $GPU"

common_args="\
    --is_training 1 \
    --task_name long_term_forecast \
    --seq_len 96 \
    --pred_len 336 \
    --label_len 0 \
    --d_model 768 \
    --n_heads 4 \
    --d_layers 1 \
    --gpt_layers 6 \
    --use_ffc 1 \
    --use_channel_aware_prompt 1 \
    --use_llm 1 \
    --lambda_sim 0.1 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --batch_size 24 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 5 \
    --dropout 0.1 \
    --epsilon 0.3 \
    --weight_decay 0 \
    --output_loss mse \
    --itr 1 \
    --gpu $GPU"

# ---------- 1. CALF ----------
echo "==================== CALF: Traffic (batch_size=24) ===================="
python -u run.py $common_args \
    --model CALF \
    --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv \
    --model_id CALFeff_Traffic

# ---------- 2. CMIE 无 ChannelMerge ----------
echo "==================== CMIE w/o CM: Traffic (batch_size=24) ===================="
python -u run.py $common_args \
    --model CMIE \
    --merge_layers 0 \
    --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv \
    --model_id CMIEeff_nocm_Traffic

# ---------- 3. CMIE CM 无 patch ----------
echo "==================== CMIE CM no-patch: Traffic (batch_size=24) ===================="
python -u run.py $common_args \
    --model CMIE \
    --merge_layers 1 \
    --merge_threshold 0 \
    --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv \
    --model_id CMIEeff_nopatch_Traffic

# ---------- 4. CMIE CM + patch ----------
echo "==================== CMIE CM + patch: Traffic (batch_size=24) ===================="
python -u run.py $common_args \
    --model CMIE \
    --merge_layers 1 \
    --merge_threshold 0.5 \
    --data Traffic --root_path ./datasets/traffic/ --data_path traffic.csv \
    --model_id CMIEeff_patch_Traffic

echo "Traffic efficiency runs done!"
