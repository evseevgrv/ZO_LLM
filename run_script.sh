#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,5 # Указываем, какие GPU использовать
export WANDB_DISABLED="false"        # Включаем логирование в Weights & Biases
export WANDB_PROJECT="zo-llm-ft"    # Название проекта в W&B
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

# добавил новые флаги в self.args: zo_tau, zo_use_smoothing
# вместо --model_name=facebook/opt-13b сейчас --model_name=microsoft/deberta-v3-base
# вместо --task_name=MultiRC --output_dir=result/MultiRC-ft-$TAG сейчас --task_name=MultiRC --output_dir=result/MultiRC-ft-$TAG
# module_wise_perturbation = False вместо  True
# убрал --lora

python run.py --model_name=facebook/opt-1.3b \
            --task_name=SST2 --output_dir=result/SST2-ft-$TAG --num_train_epochs=5  \
            --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps  \
            --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10  \
            --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side  \
            --trainer=zo_muon --train_set_seed=0 --lr_scheduler_type=constant --save_steps=1000 --load_float16  \
            --learning_rate=5e-1 --zo_eps=0.001 --momentum=0.9 --weight_decay=0 --module_wise_perturbation=False \
            --zo_tau=0.1 --zo_use_smoothing=true --zo_beta=1e-2 --overwrite_output_dir

# python run.py --model_name=facebook/opt-1.3b \
#             --task_name=SST2 --output_dir=result/SST2-ft-$TAG --num_train_epochs=5 \
#             --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps \
#             --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 \
#             --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_sgd --train_set_seed=0 \
#             --lr_scheduler_type=constant --save_steps=1000 --load_float16 --learning_rate=1e-8 --zo_eps=0.001 --momentum=0.9 
#             --weight_decay=0 --module_wise_perturbation=True
