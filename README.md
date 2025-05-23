# Leveraging Coordinate Momentum in SignSGD and Muon: Memory-Optimized Zero-Order LLM Fine-Tuning

This repository contains the code for experiments applying ZO JAGUAR SignSGD and ZO JAGUAR Muon methods for different LLM Fine-Tuning tasks.

The code is based on the [benchmark](https://github.com/ZO-Bench)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train and evaluate the model in the paper, run this command:

```
./run_script.sh
```

## Methods 

* `zo_ns_jaguar` is Jaguar Muon
* `zo_jaguar` is Jaguar SignSGD
* `zo_muon` is ZO-Muon
