
Torch implementation of the pixel CNN architecture proposed in the paper
 [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759
).


#### Usage

```sh
luarocks install cudnn
luarocks install torchnet
luarocks install mnist
```

Note that only `torch.cudnn` binding support 4D CrossEntropyLoss,
or it will be horribly slowing down if you do the unroll on Lua level.

* MNIST training script:

    `th train_mnist.lua -usegpu -crit softmax`


####TODOs

* :white_check_mark: tarining on `Torchnet`
* :white_check_mark: `MaskConv()` module
* :white_check_mark:`Sigmoid()` criterion (on MNIST)
* :white_small_square: sampling on `Torchnet`
* :white_check_mark: 256-way `Softmax` support
* :white_small_square: CIFAR dataset (?)
* :white_small_square: Conditional Image Generation with PixelCNN Decoders



#### Reference

Thanks @linmx0130, @liuzhuang13 for their helpful discussions.

* Lasagne + Theano: https://github.com/kundan2510/pixelCNN
* Tenserflow: https://github.com/carpedm20/pixel-rnn-tensorflow


#### Author

@taineleau