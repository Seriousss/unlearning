## Over-unleanring

```shell
$ cd Over-unlearning
```

```shell
$ conda create --name Over-unlearning python=3.7
$ pip install torch==1.13.1
$ pip install torchvision==0.14.1
$ pip install tqdm==4.65.0
$ pip install numpy==1.21.6
```

更多环境参考environment.yml

### Train: 

现在已有的process过的数据是cifar10：

```shell
$ python train.py --dataset cifar10 --network vgg16 --model_folder VGG16
```

查看 VGG16/train_log.csv 选举表现最好的model 重命名为best_model.hdf5

### Unlearning:

```shell
$ python unlearn.py --model_folder VGG16 --dataset cifar10 --network vgg16  --unlearn_class airplane --    log_path VGG16/log
```

## UnlearningLeaks

```shell
$ cd UnlearningLeaks
$ conda create --name unlearningleaks python=3.9
$ conda activate unlearningleaks
$ pip3 install scikit-learn pandas opacus tqdm psutil
$ pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

###### Step 1: Train Original and Unlearned Models ######
$ python main.py --exp model_train

###### Step 2: Membership Inference Attack under Different Settings ######

###### UnlearningLeaks in 'Retraining from scratch' ######
$ python main.py --exp mem_inf --unlearning_method scratch

###### UnlearningLeaks in 'SISA'
$ python main.py --exp model_train --unlearning_method sisa
$ python main.py --exp mem_inf --unlearning_method sisa
```

