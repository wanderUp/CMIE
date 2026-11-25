# Variant 8: Replace LLM with Transformer Encoder (llm_backbone=transformer)
# Usage: gpu=0 bash scripts/run_ablation_transformer.sh

common_args="\
    --is_training 1 \
    --task_name long_term_forecast \
    --model CMIE \
    --root_path ./datasets/ETT-small/ \
    --seq_len 96 \
    --pred_len 96 \
    --label_len 0 \
    --d_model 512 \
    --n_heads 8 \
    --d_layers 1 \
    --gpt_layers 6 \
    --merge_layers 1 \
    --use_ffc 1 \
    --use_channel_aware_prompt 1 \
    --use_llm 1 \
    --llm_backbone transformer \
    --lambda_sim 0.1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --dropout 0.1 \
    --gpu $gpu"

# ETTh1
echo "==================== Transformer Encoder: ETTh1 ===================="
python -u run.py $common_args --data ETTh1 --data_path ETTh1.csv --model_id CMIE_ablate_transformer_ETTh1

# ETTh2
echo "==================== Transformer Encoder: ETTh2 ===================="
python -u run.py $common_args --data ETTh2 --data_path ETTh2.csv --model_id CMIE_ablate_transformer_ETTh2

# ETTm1
echo "==================== Transformer Encoder: ETTm1 ===================="
python -u run.py $common_args --data ETTm1 --data_path ETTm1.csv --model_id CMIE_ablate_transformer_ETTm1

# ETTm2
echo "==================== Transformer Encoder: ETTm2 ===================="
python -u run.py $common_args --data ETTm2 --data_path ETTm2.csv --model_id CMIE_ablate_transformer_ETTm2

echo "Replace LLM with Transformer Encoder done!"