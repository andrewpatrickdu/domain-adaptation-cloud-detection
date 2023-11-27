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
