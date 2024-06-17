#!/bin/bash
export EXPERIMENT_NAME=103
export MODEL_PATH="/public/huangqidong/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
export CLASS_DIR="data/class-person"
export INSTANCE_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/clean/CelebA-HQ/$EXPERIMENT_NAME"

accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_PATH  \
    --enable_xformers_memory_efficient_attention \
    --train_text_encoder \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$DREAMBOOTH_OUTPUT_DIR \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks person" \
    --class_prompt="a photo of person" \
    --inference_prompt="a photo of sks person" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-7 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=1000 \
    --center_crop \
    --mixed_precision=bf16 \
    --prior_generation_precision=bf16 \
    --sample_batch_size=1 \
    --seed=0
  
python infer.py \
      --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
      --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer

