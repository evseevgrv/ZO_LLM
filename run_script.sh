#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 # Указываем, какие GPU использовать
export WANDB_DISABLED="true"        # Включаем логирование в Weights & Biases
export WANDB_PROJECT="zo-llm-ft"    # Название проекта в W&B
# export WANDB_API_KEY=$(cat ~/.wandb_api_key)

# добавил новые флаги в self.args: zo_tau, zo_use_smoothing
# вместо --model_name=facebook/opt-13b сейчас --model_name=microsoft/deberta-v3-base
# вместо --task_name=MultiRC --output_dir=result/MultiRC-ft-$TAG сейчас --task_name=MultiRC --output_dir=result/MultiRC-ft-$TAG
# module_wise_perturbation = False вместо  True
# убрал --lora

# result/Copa-ft-$TAG

python run.py --model_name="meta-llama/Llama-2-7b-hf" \
            --task_name=WSC --output_dir=None --num_train_epochs=5  \
            --per_device_train_batch_size=64 --evaluation_strategy=steps  \
            --save_strategy=no --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10  \
            --num_eval=1000 --num_train=1000 --num_dev=100 --perturbation_mode=two_side  \
            --trainer=zo_muon_sampling_v2 --train_set_seed=0 --lr_scheduler_type=constant --save_steps=1000  \
            --learning_rate=1e-3 --zo_eps=0.01 --momentum=0 --weight_decay=0 --module_wise_perturbation=False \
            --zo_tau=0.003 --zo_use_smoothing=true --zo_beta=0.8 --overwrite_output_dir --load_float16 --warmup_steps=0 \
            --sampling_type='Householder_reflection'