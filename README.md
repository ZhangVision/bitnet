# Convolutional Networks with Binary Tree Architectures

This repository is the torch implementation for the paper Truncating Wide Networks using Binary Tree Architectures.

## Installation and Usage

To run our code,

- Clone and install [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
- Put ```bitnet.lua``` into the folder ```models/```.
- Replace ```opts.lua``` of fb.resnet.torch with our ```opts.lua```.
- (Optional) To use the same learning rate schedule for cifar classification task as in the paper, one can change line 180/182 of ```train.lua``` to ```decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0```.

To train on ImageNet with bitnet-26 model,

```th main.lua -dataset imagenet -netType bitnet -depth 26 -nEpochs 90 ```.
 
To train on cifar,

```th main.lua -dataset cifar100 -netType bitnet -width 10 -bdepth 2 -nblock 3 -LR 0.1 -nEpochs 200 ```

where ```width bdepth nblock``` implement the ```d k n ``` in the Table 1 of the paper.
 For other traing options e.g. batchSize and nGPU, one can check [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
