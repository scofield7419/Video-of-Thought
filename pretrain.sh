
#!/bin/bash


# =================== Training ======================
DATASET_NAME_LIST=(
    ""
)
DATASET_NAME_LIST="${DATASET_NAME_LIST[@]}"

LLM_MODEL_NAME="./pretrain_ckpt/vicuna-7b-v1.5"
MM_MODEL_NAME="./pretrain_ckpt/clip"

echo "DATASET_NAME_LIST: $DATASET_NAME_LIST"
echo "LLM_MODEL_NAME: $LLM_MODEL_NAME"
echo "MM_MODEL_NAME: $MM_MODEL_NAME"

deepspeed  --include localhost:0,2 --master_addr 127.0.0.1 --master_port 28460 train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_MODEL_NAME \
    --version v1 \
    --dataset_name_list $DATASET_NAME_LIST \
    --multimodal_tower $MM_MODEL_NAME \
    --group_by_modality_length True \
    --group_by_modality_type False \
    --tune_mm_input_adapter True \
    --freeze_mm_input_adapter False \
    --mm_input_projector_type mlp \
    --mm_use_vid_start_end False \
    --mm_use_vid_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/pretrain \
    --num_train_epochs 6 \
    --per_device_train_batch_size 120 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
