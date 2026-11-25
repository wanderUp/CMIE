export CUDA_VISIBLE_DEVICES=0

model_name=CMIE


for interval in 'Monthly' 'Quarterly' 'Yearly' 'Daily' 'Hourly' 'Weekly'
do
python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./datasets/m4 \
    --seasonal_patterns $interval \
    --model_id m4_$interval \
    --model $model_name \
    --data m4 \
    --features M \
    --d_layers 1 \
    --factor 3 \
    --train_epochs 200 \
    --batch_size 256 \
    --d_model 768 \
    --n_heads 4 \
    --itr 1 \
    --learning_rate 0.001 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --gpt_layers 4 \
    --output_loss smooth_l1 \

echo '====================================================================================================================='
done
done
done
done
done
