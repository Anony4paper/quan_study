
## Requirements
    - Tensorflow 2.3
    - Keras 2.4.3

## project structure
```
├── datasets                      # put the data here
├── motivating_examples           # motivating examples of our work
├── model_evaluation              
    ├── coreml_evaluation.py      # coreml model evaluation
    ├── ori_model_eval.py         # original model evaluation
    ├── tflite_evaluation.py      # tensorflowlite model evaluation
├── model_prepare     
    ├── model_training.py         # train the original model and regression model for RQ3
├── results                       # results for RQ1,2,3,4
├── retrain                       # retrain the model with disagreements
├── training_strategy             # training strategy
```

## Others
- All the models: https://zenodo.org/record/5916315#.YfT7oBNKhQI
- WILDs data: https://wilds.stanford.edu/
- MNIST-C data: https://github.com/google-research/mnist-c
- CIFAR10-C data: https://zenodo.org/record/2535967#.YfRacRNKhQI

