# Spatial Attention Generative Adversarial Network

This repository contains the PyTorch implementation of the ECCV 2018 paper "Generative Adversarial Network with Spatial Attention for Face Attribute Editing" ([pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf)).

My results with images and attention masks on CelebA 128 _(original, eyeglasses, mouth_slightly_open, no_beard, smiling)_

![Results](https://github.com/elvisyjlin/SpatialAttentionGAN/blob/master/pics/4_attr_results.jpg)


## Requirements

* Python 3.5
* PyTorch 1.0.0

```bash
pip3 install -r requirements.txt
```

The training procedure described in paper takes 5.5GB memory on a single GPU.

* Datasets
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    * Put _Align&Cropped Images_ in `./data/celeba/*.jpg`
    * Put _Attributes Annotations_ in `./data/list_attr_celeba.txt`

* Pretrained models (download from https://goo.gl/EMj4VL and decompress the zips to `./results`)
  ```
    results
    ├── celeba_128_eyeglasses
    ├── celeba_128_mouth_slightly_open
    ├── celeba_128_no_beard
    └── celeba_128_smiling
  ```

## Usage

Train a model with a target attribute

```bash
python3 train.py --experiment-name celeba_128_eyeglasses --target-attr Eyeglasses --gpu
```

Generate images from trained models

```bash
python3 generate.py --experiment-name celeba_128_eyeglasses --gpu
```