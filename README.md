# Spatial Attention Generative Adversarial Network

This repository contains the PyTorch implementation of the ECCV 2018 paper "Generative Adversarial Network with Spatial Attention for Face Attribute Editing" ([pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf)).

## Requirements

* Python 3.5
* PyTorch 1.0.0

```bash
pip3 install -r requirements.txt
```

The training procedure described in paper takes 5.5GB memory on a single GPU.

## Usage

Train a model with a target attribute

```bash
python3 train.py --experiment-name celeba_128_eyeglasses --target-attr Eyeglasses --gpu
```

Generate images from trained models

```bash
python3 generate.py --experiment-name celeba_128_eyeglasses --gpu
```