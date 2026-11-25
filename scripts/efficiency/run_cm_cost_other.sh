#!/usr/bin/env bash
# =============================================================================
# 效率实验（补跑）：非 Traffic 数据集 (ETTh2 / ECL / Weather)，batch_size=32
# 只需补跑: 4. CMIE CM + patch (merge_threshold 0.5)
#   CALF / CMIE无CM / CM无patch 已完成，不再运行
#
# 用法: gpu=0 bash scripts/efficiency/run_cm_cost_other.sh
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

# ---------- 4. CMIE CM + patch (merge_threshold 0.5) ----------
for ds in ETTh2 ECL Weather; do
    case $ds in
        ETTh2)  data=ETTh2;   root=./datasets/ETT-small/;    path=ETTh2.csv ;;
        ECL)    data=ECL;     root=./datasets/electricity/;  path=electricity.csv ;;
        Weather)data=Weather; root=./datasets/weather/;      path=weather.csv ;;
    esac
    echo "==================== CMIE CM + patch: $ds (batch_size=32) ===================="
    python -u run.py $common_args \
        --model CMIE \
        --merge_layers 1 \
        --merge_threshold 0.5 \
        --data "$data" --root_path "$root" --data_path "$path" \
        --model_id CMIEeff_patch_${ds}
done

echo "Non-Traffic CM+patch runs done!"
