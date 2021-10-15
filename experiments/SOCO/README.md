# Unofficial-SOCO
Unofficial application of SOCO:[Aligning Pretraining for Detection via Object-Level Contrastive Learning](https://arxiv.org/abs/2106.02637). This application includes all tricks used in SOCO.

## Preparation

Setup Env: [requirements.txt](requirements.txt).

Use [./selective-search](selective-search) to generate selective-search proposals.

## How to run
### Config
```
waiting
```

### Training SOCO
```
cd ./model_zoo_exp/imagenet1k_scratch_sgd_224_p4_r50_512/
./slurm_train.sh spring_scheduler 32
```

### Transferring to Object Detection
First, use `soco2detectron.py` to convert the checkpoint to detectron.

Then, train faster-rcnn-fpn on detectron2 with the same setting with [detco](https://github.com/xieenze/DetCo).

## Experiments

| Method | Setting | VOC-AP50 | VOC-AP50 |
| - | - | - | - | 
| SOCO| bs512, syncBN, sgd, lr0.4, epoch100, weight decay$1e^{-4}$, v3:224 | 55.7656| 82.03 |
| SOCO| bs2048, syncBN, lars, lr8, epoch400, weight decay$1e^{-5}$, v3:96| - | - |
