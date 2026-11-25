# Variant 6: w/o Similarity Regularization (lambda_sim=0)
# Usage: gpu=0 bash scripts/run_ablation_wosim.sh

common_args="\
    --is_training 1 \
    --task_name long_term_forecast \
    --model CMIE \
    --root_path ./datasets/ETT-small/ \
    --seq_len 96 \
    --pred_len 96 \
    --label_len 0 \
    --d_model 768 \
    --n_heads 8 \
    --d_layers 1 \
    --gpt_layers 6 \
    --merge_threshold 0.5 \
    --merge_layers 1 \
    --use_ffc 1 \
    --use_channel_aware_prompt 1 \
    --use_llm 1 \
    --llm_backbone gpt2 \
    --lambda_sim 0 \
    --r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 5 \
    --dropout 0.1 \
    --gpu $gpu"

# ETTh1
echo "==================== w/o SimReg: ETTh1 ===================="
python -u run.py $common_args --data ETTh1 --data_path ETTh1.csv --model_id CMIE_ablate_wosim_ETTh1

# ETTh2
echo "==================== w/o SimReg: ETTh2 ===================="
python -u run.py $common_args --merge_layers 2 --data ETTh2 --data_path ETTh2.csv --model_id CMIE_ablate_wosim_ETTh2

# ETTm1
echo "==================== w/o SimReg: ETTm1 ===================="
python -u run.py $common_args --data ETTm1 --data_path ETTm1.csv --model_id CMIE_ablate_wosim_ETTm1

# ETTm2
echo "==================== w/o SimReg: ETTm2 ===================="
python -u run.py $common_args --merge_layers 2 --data ETTm2 --data_path ETTm2.csv --model_id CMIE_ablate_wosim_ETTm2

echo "w/o Similarity Regularization done!"
