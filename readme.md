# bert-classification-tf-serving

## Introduction
Use [BERT](https://github.com/google-research/bert/) to train a classification model and deploy the model by [tensorflow serving](https://www.tensorflow.org/tfx/serving/serving_basic). 
Then we can use REST API to do online prediction.

## Get Started

The whole experiment is based on `Chnsenticorp` data set, which is a two-class data set of positive and negative emotions.

### 0. Prepare Pre-train model

Download the Chinese bert model `chinese_L-12_H-768_A-12`, then unzip and move to the `models` dir.

### 1. Train a model

```bash
sh fine-tuning.sh
```

### 2. Do prediction and export model

We need to change the checkpoint to the tensorflow serving's format.

```bash
sh export-model.sh
```

Then the structure of `export-model` will be:
```bash
.
└── 1569141360
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

### 3. Deploy model
We use tensorflow serving docker version to deploy the model on CPU OR GPU.

a. On CPU

`docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=$(pwd)/models/export-model,target=/models/chnsenticorp -e MODEL_NAME=chnsenticorp -t tensorflow/serving`

b. On GPU

You need to install nvidia-docker environment first!

`docker run --runtime=nvidia -e CUDA_VISIBLE_DEVICES=0 -p 8500:8500 -p 8501:8501 --mount type=bind,source=$(pwd)/models/export-model,target=/models/chnsenticorp -e MODEL_NAME=chnsenticorp -t tensorflow/serving:1.12.0-gpu`

### 4. Predict online

```
export PYTHONPATH=$(pwd)/google-bert
python predict_online.py
```

a. On CPU
```
Input test sentence:
跟住招待所没什么太大区别。 绝对不会再住第2次的酒店！
negative pro:0.96344769 positive pro:0.0365523435 time consuming:198ms
Input test sentence:
LED屏就是爽，基本硬件配置都很均衡，镜面考漆不错，小黑，我喜欢。
negative pro:0.0291572567 positive pro:0.970842779 time consuming:195ms
```

b. On GPU
```
Input test sentence:
跟住招待所没什么太大区别。 绝对不会再住第2次的酒店！
negative pro:0.963448 positive pro:0.0365524 time consuming:92ms
Input test sentence:
LED屏就是爽，基本硬件配置都很均衡，镜面考漆不错，小黑，我喜欢。
negative pro:0.0291573 positive pro:0.970843 time consuming:87ms
```

The probability on GPU and GPU is the same, but the speed on GPU is faster **50%+** than on CPU!

The max token length is 128 and the CPU and GPU configuration is as follows:

|CPU|GPU|
|:---:|:---:|
|Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz|Tesla V100|

## Conclusion
This is a demo project for classification by BERT and deploy the model by tensorflow serving.

We use tensorflow serving docker version, so we can use docker stack technology to easily expand for load balance in production environment.

