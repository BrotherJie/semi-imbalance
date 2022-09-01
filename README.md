# Dependencies

* `python3`
* `pytorch`
* `torchvision`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`



# Stage 1 training

label sampler: random

unlabel sampler: random

imbalanced ratio: 100

unlabeled data ratio: 2

```shell
nohup python fix_train.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 --sampler random --semi-sampler random --out result/cifar10_fix_100_2_random_random_stage1 >logfile_100_2_random_random_stage1.txt 2>&1 &
```


# Stage 2 training

## Resampling way

fix feature extractor and retrain classifer by using mean sampler for both labeled and unlabeled data

stage 1 training strategy

labeled sampler: random

unlabeled sampler: random

```shell
nohup python fix_finetune.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 --sampler mean --semi-sampler mean --resume result/cifar10_fix_100_2_random_random_stage1/202X-MM-DD-HH-MM-SS/best_model.pth.tar --out result/cifar10_fix_100_2_random_random_stage2 >result/cifar10_fix_100_2_random_random_stage2/log/logfile_exp1.txt 2>&1 &
```

## Logit adjustment way

fix feature extractor and use logit adjustment

stage 1 training strategy

labeled sampler: random

unlabeled sampler: random

```shell

nohup python fix_finetune_LA_post-hoc.py --gpu 0 --ratio 2 --imb_ratio 100 --imb_ratio_u 100 --resume result/stage1/cifar10_fix_100_2_random_random_stage1/2021-11-28-18-05-21/best_model.pth.tar --out result/LA-post-hoc/cifar10/100_2 >result/LA-post-hoc/cifar10/100_2/all_repredict_exp1.txt 2>&1 &
```