torchrun --nproc_per_node=2 LongLoRA/supervised-fine-tune-qlora.py  \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --bf16 True \
        --model_max_length 16384 \
        --use_flash_attn True \
        --low_rank_training True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 16     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 200     \
        --save_total_limit 1     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 20     \
        --deepspeed "LongLoRA/ds_configs/stage2.json" \
        --tf32 True \
        --use_flash_attn True \
        --output_dir "" \
        --data_path ""