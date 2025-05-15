# Implementation of JAGUAR and ZO-MUON
Here you can see the current state of our research in JAGUAR and ZO-MUON implementation

The code is based on the [benchmark](https://github.com/ZO-Bench)

На данный момент методы реализованы в функциях ```zo_jaguar_step``` и ```zo_muon_step``` соответственно

## Добавление своего метода в общий ZO-Benchmark

Статья: [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark](https://arxiv.org/pdf/2402.11592)

Работа предлагает несколько методов оптимизации нулевого и первого порядков, при желании сравниться с которыми хочется добавить свой метод

Как это сделать?

1. В файле ```trainer.py``` в строке ```~315``` необходимо добавить ваш метод в ```if```'е:
   
    то есть написать:
  
    ```
    elif args.trainer == "zo_your_method_name":
  
      self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    ```

    где во второй строчке вы выбираете оптимизатор, схожий вашему (чаще всего ```SGD```, в самой работе предлагается также ```Adam```)

2. Далее в ```~515``` строке необходимо снова добавить ваш метод в ```if```'е:
   
    то есть написать:
  
    ```
    elif args.trainer == "zo_your_method_name":
  
      tr_loss_step = self.zo_your_method_step(model, inputs)
    ```

    В этой части мы уже непосредственно ссылаемся на функцию, которую предстоит написать

3. В коде также фигурирует функция, которая не была корректно реализована в самой работе, но на будущее сможем добавить наш метод и в неё тоже
   
   То есть в строке ```~554``` добавляем наш метод в список на update параметров: ```if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt", "zo_conserv", "zo_your_method_name"]```

4. Далее остается реализовать наш метод, это удобно делать рядом с уже готовыми реализациями:
   
   В строке ```~1000``` добавляем функцию шага вашего метода, которая должна иметь следующую сигнатуру:

    ```
    @torch.no_grad()
    def zo_your_method_step(self, model, inputs) -> loss
    ```

    В процессе реализации пользуйтесь готовыми функциями, например, вычилением ```loss```'а, используя метод  ```self.zo_forward(model, inputs)```

    Готовую реализацию можно посмотреть по поиску в коде ```zo_step```


Остается в запускаемом скрипте (т.е. в ```.sh``` файле) добавить ваш метод в ```--trainer=zo_your_method_name```

Пример организации запускаемого скрипта:

```
#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,5  # Указываем, какие GPU использовать
export WANDB_DISABLED="false"        # Включаем логирование в Weights & Biases
export WANDB_PROJECT="zo-llm-ft"    # Название проекта в W&B
export WANDB_API_KEY=$(cat ~/.wandb_api_key)

python run.py --model_name=facebook/opt-13b \
            --lora --task_name=MultiRC --output_dir=result/MultiRC-ft-$TAG --num_train_epochs=5  \
            --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps  \
            --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10  \
            --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side  \
            --trainer=zo_cheetah --train_set_seed=0 --lr_scheduler_type=constant --save_steps=1000 --load_float16  \
            --learning_rate=1e-8 --zo_eps=0.001 --momentum=0.9 --weight_decay=0 --module_wise_perturbation=True \
            --tau=1e-6, --zo_use_smoothing=false
```

Сюда же стоит добавлять нужные вам гиперпараметры, которые вы хотите видеть в коде в ```self.args```, соответственно после добавление флага ```--new_hyperparam=<init value, e.g. 0.001>``` вы можете использовать его в вашем коде так ```self.args.new_hyperparam```, предварительно указав их в файле ```run.py```, например в строках ```40-170``` при определении класса

