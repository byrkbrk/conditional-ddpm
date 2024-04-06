# Conditional DDPM

## Introduction

We implement a simple conditional form of *Diffusion Model* described in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), in PyTorch. Preparing this repository, we inspired by the course [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work) and the repository [minDiffusion](https://github.com/cloneofsimo/minDiffusion). While training, we use [MNIST](http://yann.lecun.com/exdb/mnist/), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), and Sprite (see [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels) and [kyrise](https://kyrise.itch.io/)) datasets.

## Setting Up the Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/diffusion-model.git
    ~~~
3. In the directory diffusion-model, if you're mac user run:
    ~~~
    conda env create -f diffusion-env_macos.yaml
    ~~~
    If you're linux or windows user run:
    ~~~
    conda env create -f diffusion-env_linux_or_windows.yaml
    ~~~
4. Activate the environment:
    ~~~
    conda activate diffusion-env
    ~~~

## Training and Sampling

### MNIST
To train the model on MNIST dataset from scratch,
~~~
python3 train.py --dataset-name mnist --device your-device
~~~
Above, please change `your-device` input, as either `cuda` or `mps`.

In order to sample from a (pre-trained) checkpoint, run:
~~~
python3 sample.py pre_trained_mnist_checkpoint_49.pth --n-samples 400 --n-images-per-row 20 --device your-device
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each two rows represents a class label (in total 20 rows and 10 classes).


<div style="display: flex;">
    <img src="files-for-readme/mnist_ddpm_images.jpeg" alt="mnist-jpeg" style="width: 45%; margin-right: 5%;">
    <img src="files-for-readme/mnist_ani.gif" alt="mnist-ani" style="width: 45%; margin-left: 5%;">
</div>

### Fashion-MNIST

To train the model from scratch on Fashion-MNIST dataset,
~~~
python3 train.py --dataset-name fashion_mnist --device your-device
~~~
Above, please change `your-device` input, as either `cuda` or `mps`.

In order to sample from a (pre-trained) checkpoint, run:
~~~
python3 sample.py pre_trained_fashion_mnist_checkpoint_49.pth --n-samples 400 --n-images-per-row 20 --device your-device
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each two rows represents a class label (in total 20 rows and 10 classes).

<div style="display: flex;">
    <img src="files-for-readme/fashion_mnist_ddpm_images.jpeg" alt="mnist-jpeg" style="width: 45%; margin-right: 5%;">
    <img src="files-for-readme/fashion_mnist_ani.gif" alt="mnist-ani" style="width: 45%; margin-left: 5%;">
</div>

### Sprite
To train the model from scratch on Sprite dataset,
~~~
python3 train.py --dataset-name sprite --device your-device
~~~
Above, please change `your-device` input, as either `cuda` or `mps`.

In order to sample from a (pre-trained) checkpoint, run:
~~~
python3 sample.py pre_trained_sprite_checkpoint_49.pth --n-samples 225 --n-images-per-row 15 --device your-device
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each three rows represents a class label (in total 15 rows and 5 classes).

<div style="display: flex;">
    <img src="files-for-readme/sprite_ddpm_images.jpeg" alt="mnist-jpeg" style="width: 45%; margin-right: 5%;">
    <img src="files-for-readme/sprite_ani.gif" alt="mnist-ani" style="width: 45%; margin-left: 5%;">
</div>
