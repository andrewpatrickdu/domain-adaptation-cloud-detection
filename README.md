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
You can download the datasets at:

NOTE: Make sure you add the two folders into the main directory (```cloud-detection```)

## Training a source model
To train a source model, run the following python script:

```
python train-source.py --MODEL_ARCH cloudscout --DATASET S2-2018 --NUM_BANDS 3 --GPU 0 --NUM_EPOCHS 300 --ROOT [directory of `cloud-detection` folder]
                                    cloudscout8          L9-2023             8
                                    resnet50
```

For example, to train the CloudScout architecture on Sentinel-2 data using 3 bands:
```
python train-source.py --MODEL_ARCH cloudscout --DATASET S2-2018 --NUM_BANDS 3 --GPU 0 --NUM_EPOCHS 300 --ROOT /home/andrew/cloud-detector
```

## Updating the source model to the target domain via offline adaptation (bandwidth efficient SDA)
There are three main python scripts used to update a source model in the offline adaptation setting:

* ```generate_mask.py``` - defines functions used to calculate the FISH Mask.
* ```fish-mask-cloudscout.py``` - used to update the CloudScout or CloudScout8 model.
* ```fish-mask-resnet50.py``` - used to update the resnet50 model.

For example, to update only 25% of the weights of CloudScout (trained on Sentinel-2) to Landsat-9 using 3 bands: 
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
    --LOG True
```
or to update only 1% of the weights of resnet-50 (trained on Landsat-9) to Sentinel-2 using 8 bands:
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
    --LOG True
```








