# domain-adaptation-cloud-detection

This repository is based on the paper, "Domain Adaptation for Satellite-Borne Hyperspectral Cloud Detection" - https://arxiv.org/abs/2309.02150

```
@article{du2023domain,
  title={Domain Adaptation for Satellite-Borne Hyperspectral Cloud Detection},
  author={Du, Andrew and Doan, Anh-Dzung and Law, Yee Wei and Chin, Tat-Jun},
  journal={arXiv preprint arXiv:2309.02150},
  year={2023}
}
```
## Getting started

### Installations
We used Python 3.7 to write/run our code and Anaconda to install the following libraries:

```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install tensorboardX tensorboard
pip install torch-summary
conda install matplotlib 
conda install -c anaconda pandas
```

### Datasets
You can download the datasets (```Sentinel-2-Cloud-Mask-Catalogue``` and ```Landsat-9-Level-1```) at: ...

NOTE: Make sure you add the two folders into the ```cloud-detection/datasets/``` directory.

## Training a source model
To train a source model, run the following python script:

```
python train-source.py --MODEL_ARCH cloudscout --DATASET S2-2018 --NUM_BANDS 3 --GPU 0 --NUM_EPOCHS 300 --ROOT [directory of `domain-adaptation-cloud-detection` folder]
                                    cloudscout8          L9-2023             8
                                    resnet50
```

For example, to train the CloudScout architecture on the 3 bands of Sentinel-2:
```
python train-source.py --MODEL_ARCH cloudscout --DATASET S2-2018 --NUM_BANDS 3 --GPU 0 --NUM_EPOCHS 300 --ROOT /home/andrew/domain-adaptation-cloud-detection
```
Once training has completed, the results and checkpoint files are saved in ```cloud-detector/checkpoints/source-models/cloudscout-128a-S2-2018```.


## Updating the source model to the target domain via offline adaptation (bandwidth efficient SDA)
There are three python scripts used to update a source model in the offline adaptation setting:

* ```generate_mask.py``` - defines functions used to calculate the FISH Mask.
* ```fish-mask-cloudscout.py``` - used to update the CloudScout or CloudScout8 model.
* ```fish-mask-resnet50.py``` - used to update the ResNet50 model.

For example, to update only 25% of the weights of CloudScout (trained on Sentinel-2) to the 3 bands of Landsat-9: 
```
python fish-mask-cloudscout.py \
    --MODEL cloudscout-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --TRAIN_EPOCH 300 \
    --TRAIN_BATCH_SIZE 2 \
    --TEST_BATCH_SIZE 2 \
    --FISH_NUM_SAMPLES 2000 \
    --FISH_KEEP_RATIO 0.25 \
    --FISH_SAMPLE_TYPE label \
    --FISH_GRAD_TYPE square \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```
or to update only 1% of the weights of resnet-50 (trained on Landsat-9) to the 3 bands of Sentinel-2:
```
python fish-mask-resnet50.py \
    --MODEL resnet50-8-L9-2023 \
    --NUM_BANDS 8 \
    --DATASET S2-2018 \
    --TRAIN_EPOCH 300 \
    --TRAIN_BATCH_SIZE 2 \
    --TEST_BATCH_SIZE 2 \
    --FISH_NUM_SAMPLES 2000 \
    --FISH_KEEP_RATIO 0.01 \
    --FISH_SAMPLE_TYPE label \
    --FISH_GRAD_TYPE square \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```

## Updating the source model to the target domain via online adaptation (test-time adaptation)
Two test-time adaptation appoaches were used to update a source model to the target domain: (1) Dynamic Unsupervised Adaptation (DUA), and (2) Test Entropy Minimisation (TENT).


### DUA
For example, to update the CloudScout model (trained on Sentinel-2) to the 3 bands of Landsat-9:
```
python tta-dua-cloudscout.py \
    --MODEL cloudscout-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --ADAPTATION_BATCH_SIZE 16 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 16 \
    --ADAPTATION_AUGMENTATION False \
    --ADAPTATION_DECAY_FACTOR 0.94 \
    --ADAPTATION_MIN_MOMENTUM_CONSTANT 0.005 \
    --ADAPTATION_MOM_PRE 0.1 \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```
or to update the ResNet50 model (trained on Landsat-9) to the 3 bands of Sentinel-2:
```
python tta-dua-resnet50.py \
    --MODEL resnet50-125-L9-2018 \
    --NUM_BANDS 3 \
    --DATASET S2-2018 \
    --ADAPTATION_BATCH_SIZE 16 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 16 \
    --ADAPTATION_AUGMENTATION False \
    --ADAPTATION_DECAY_FACTOR 0.94 \
    --ADAPTATION_MIN_MOMENTUM_CONSTANT 0.005 \
    --ADAPTATION_MOM_PRE 0.1 \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```

### TENT
For example, to update the CloudScout model (trained on Sentinel-2) to the 3 bands of Landsat-9:
```
python tta-tent-cloudscout.py \
    --MODEL cloudscout-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --ADAPTATION_EPOCH 1 \
    --ADAPTATION_BATCH_SIZE 6 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 9999 \
    --ADAPTATION_RESET_STATS True \
    --ADAPTATION_LEARNING_RATE 0.001 \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```
or to update the ResNet50 model (trained on Landsat-9) to the 3 bands of Sentinel-2:
```
python tta-dua-resnet50.py \
    --MODEL resnet50-128a-L9-2023 \
    --NUM_BANDS 3 \
    --DATASET S2-2018 \
    --ADAPTATION_BATCH_SIZE 16 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 16 \
    --ADAPTATION_AUGMENTATION False \
    --ADAPTATION_DECAY_FACTOR 0.94 \
    --ADAPTATION_MIN_MOMENTUM_CONSTANT 0.005 \
    --ADAPTATION_MOM_PRE 0.1 \
    --GPU 0 \
    --LOG False \
    --ROOT /home/andrew/domain-adaptation-cloud-detection
```


## Help and updates
When time permits, I will update the repository based on the issues others may have at setting up and/or running the code.


