#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2
export WANDB_DISABLED="false"        
export WANDB_ENTITY="andrey"
export WANDB_PROJECT="zo-llm-ft"    
export WANDB_API_KEY=""
export HF_TOKEN="" # for llama 

python run.py --model_name="facebook/opt-1.3b" \
            --task_name=SST2 --output_dir=result/SST2-FT-$TAG --num_train_epochs=5  \
            --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps  \
            --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10  \
            --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side  \
            --trainer=zo_ns_jaguar --train_set_seed=0 --lr_scheduler_type=constant --save_steps=1000  \
            --learning_rate=1e-4 --zo_eps=1e-3 --momentum=0.0 --weight_decay=0 --module_wise_perturbation=False \
            --zo_tau=5e-3 --zo_use_smoothing=true --zo_beta=0.1 --overwrite_output_dir --report_to="wandb"
