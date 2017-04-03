# Convolutional Networks with Binary Tree Architectures

This repository is the torch implementation for the paper Truncating Wide Networks using Binary Tree Architectures.

## Installation and Usage

To run our code,

- Clone and install [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
- Put ```bitnet.lua``` into the folder ```models/```.
- Replace ```opts.lua``` of fb.resnet.torch with our ```opts.lua```.
- (Optional) To use the same learning rate schedule for cifar classification task as in the paper, one can change line 180/182 of ```train.lua``` to ```decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0```.

To train on ImageNet with bitnet-26 model,

```th main.lua -dataset imagenet -netType bitnet -depth 26 -nEpochs 90 -data /path/to/ILSVRC2012/```.
 
To train on cifar,

```th main.lua -dataset cifar100 -netType bitnet -width 10 -bdepth 2 -nblock 3 -LR 0.1 -nEpochs 200 ```

where ```width bdepth nblock``` implement the ```d k n ``` in the Table 1 of the paper.
 For other traing options e.g. batchSize and nGPU, one can check [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Pretrained Models
Pretrained bitnet models for ImageNet are available for download (Google Drive): 
[bitnet-26](https://drive.google.com/file/d/0B49TImF4hCTfdkJ0aUZwNnNiRTg/view?usp=sharing),
[bitnet-34](https://drive.google.com/file/d/0B49TImF4hCTfemFCSXBDM0hvdjA/view?usp=sharing)

How to use pretrained models? See instructions [here](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).

## Citation

If you use our code in your research and would like to cite our paper:

	@article{zhang2017bitnet,
	  title={Truncating Wide Networks using Binary Tree Architectures},
	  author={Yan Zhang and Mete Ozay and Shuohao Li and Takayuki Okatani},
	  journal={arXiv preprint arXiv:--},
	  year={2017}
	}
 
 If you have any further questions or suggestion, feel free to contact us at ```zhang at vision.is.tohoku.ac.jp```

