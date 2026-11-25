#!/usr/bin/env bash
# =============================================================================
# CMIE 效率实验：三档 ChannelMerge 开销对比 + CALF 对照
# -----------------------------------------------------------------------------
# 目标：验证 ChannelMerge 引入的计算开销，以及 patch 化（ACP）能否将其
#       降低到可接受水平，实证 O(C^2) vs O(C*M)。
#
# 4 数据集: ETTh2 + ECL + Weather + Traffic  (pred_len=336, batch_size=32; Traffic=24, 100 epochs)
# 4 配置:
#   1. CALF（对照）
#   2. CMIE 无 ChannelMerge      (--merge_layers 0)
#   3. CMIE CM 无 patch          (--merge_layers 1 --merge_threshold 0)  O(C^2)
#   4. CMIE CM + patch           (--merge_layers 1 --merge_threshold 0.5) O(C*M)
#
# 用法: gpu=0 bash scripts/efficiency/run_cm_cost.sh
# 结果: result_CMIE.txt / result_CALF.txt (自动附加)
# =============================================================================

GPU=${gpu:-0}
echo "GPU = $GPU"

base_args="\
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
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 5 \
    --dropout 0.1 \
    --epsilon 0.3 \
    --weight_decay 0 \
    --output_loss mse \
    --itr 1 \
    --gpu $GPU"

# 数据集 -> (data, root_path, data_path)
declare -A DATA
DATA[ETTh2]="ETTh2|./datasets/ETT-small/|ETTh2.csv"
DATA[ECL]="ECL|./datasets/electricity/|electricity.csv"
DATA[Weather]="Weather|./datasets/weather/|weather.csv"
DATA[Traffic]="Traffic|./datasets/traffic/|traffic.csv"

# 单独用小 batch_size 控制显存（24G 上限）；其他数据集用 base_args 默认 32。
declare -A BATCH
BATCH[Traffic]=24

# ---------- 1. CALF 对照 ----------
for ds in ETTh2 ECL Weather Traffic; do
    IFS='|' read -r data root path <<< "${DATA[$ds]}"
    echo "======================================================"
    echo "CALF: $ds (batch_size=${BATCH[$ds]:-32})"
    echo "======================================================"
    python -u run.py $base_args \
        --model CALF \
        ${BATCH[$ds]:+--batch_size ${BATCH[$ds]}} \
        --data "$data" --root_path "$root" --data_path "$path" \
        --model_id CALFeff_${ds}
done

# ---------- 2. CMIE 无 ChannelMerge (merge_layers 0) ----------
for ds in ETTh2 ECL Weather Traffic; do
    IFS='|' read -r data root path <<< "${DATA[$ds]}"
    echo "======================================================"
    echo "CMIE w/o ChannelMerge: $ds (batch_size=${BATCH[$ds]:-32})"
    echo "======================================================"
    python -u run.py $base_args \
        --model CMIE \
        --merge_layers 0 \
        ${BATCH[$ds]:+--batch_size ${BATCH[$ds]}} \
        --data "$data" --root_path "$root" --data_path "$path" \
        --model_id CMIEeff_nocm_${ds}
done

# ---------- 3. CMIE CM 无 patch (merge_threshold 0) ----------
for ds in ETTh2 ECL Weather Traffic; do
    IFS='|' read -r data root path <<< "${DATA[$ds]}"
    echo "======================================================"
    echo "CMIE CM no-patch (O(C^2)): $ds (batch_size=${BATCH[$ds]:-32})"
    echo "======================================================"
    python -u run.py $base_args \
        --model CMIE \
        --merge_layers 1 \
        --merge_threshold 0 \
        ${BATCH[$ds]:+--batch_size ${BATCH[$ds]}} \
        --data "$data" --root_path "$root" --data_path "$path" \
        --model_id CMIEeff_nopatch_${ds}
done

# ---------- 4. CMIE CM + patch (merge_threshold 0.5) ----------
for ds in ETTh2 ECL Weather Traffic; do
    IFS='|' read -r data root path <<< "${DATA[$ds]}"
    echo "======================================================"
    echo "CMIE CM + patch (O(C*M)): $ds (batch_size=${BATCH[$ds]:-32})"
    echo "======================================================"
    python -u run.py $base_args \
        --model CMIE \
        --merge_layers 1 \
        --merge_threshold 0.5 \
        ${BATCH[$ds]:+--batch_size ${BATCH[$ds]}} \
        --data "$data" --root_path "$root" --data_path "$path" \
        --model_id CMIEeff_patch_${ds}
done

echo ""
echo "All CM-cost efficiency runs done!"
echo "Results in: result_CMIE.txt / result_CALF.txt"
