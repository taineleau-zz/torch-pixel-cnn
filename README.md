
Torch implementation of the pixel CNN architecture proposed in the paper
 [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759
) and other variants.

~~Will be back to update & commit after the final exam week at NUS :D.~~

So busy these days, I am unlikely to contribute to this repo recently. But I'm happy to accept new PR on TODOs!



#### Usage

```sh
luarocks install cudnn
luarocks install torchnet
luarocks install mnist
```

Note that only the`torch.cudnn` binding supports 4D CrossEntropyLoss,
or it will be horribly slowing down if you do the unroll on Lua level.

* MNIST training script:

    `th train_mnist.lua -usegpu -crit softmax`


####TODOs

* :white_check_mark: tarining on `Torchnet`
* :white_check_mark: `MaskConv()` module
* :white_check_mark:`Sigmoid()` criterion (on MNIST)
* :white_small_square: sampling on `Torchnet`
* :white_check_mark: 256-way `Softmax` support
* :white_small_square: CIFAR dataset (it's easy but let me first finish the
sampling part)
* :white_small_square: Conditional Image Generation with PixelCNN Decoders



### Reference

Thanks [@Shiyi Lan](https://github.com/voidrank) and
 [@liuzhuang13](https://github.com/liuzhuang13) for their helpful discussions on the implementation.

Implementation in other languages/frameworks:

* Lasagne + Theano: https://github.com/kundan2510/pixelCNN
* Tenserflow: https://github.com/carpedm20/pixel-rnn-tensorflow


### Author

[@taineleau](https://taineleau.me)