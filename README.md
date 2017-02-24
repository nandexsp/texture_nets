# Texture Networks + Instance normalization: Feed-forward Synthesis of Textures and Stylized Images

In the paper [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417) we describe a faster way to generate textures and stylize images. It requires learning a feedforward generator with a loss function proposed by [Gatys et. al.](http://arxiv.org/abs/1505.07376). When the model is trained, a texture sample or stylized image of any size can be generated instantly.

[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) presents a better architectural design for the generator network. By switching `batch_norm` to `instance norm` we facilitate the learning process resulting in much better quality.

This also implements the stylization part from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

# Prerequisites
- [Torch7](http://torch.ch/docs/getting-started.html) + [loadcaffe](https://github.com/szagoruyko/loadcaffe)
- cudnn + torch.cudnn (optionally)
- [display](https://github.com/szym/display) (optionally)

Download VGG-19.
```
cd data/pretrained && bash download_models.sh && cd ../..
```

# Stylization
<!-- 
Content image|  Dalaunay | Modern 
:-------------------------:|:-------------------------:|:------------------------------:
![](data/readme_pics/karya.jpg " ") | ![](data/readme_pics/karya512.jpg  " ")| ![](data/readme_pics/karya_s_mo.jpg  " ")
 -->
![](data/readme_pics/all.jpg " ")

### Training

#### Preparing image dataset

I recommend to download the MSCOCO: 
Training Images: http://msvocds.blob.core.windows.net/coco2014/train2014.zip
Validation Images: http://msvocds.blob.core.windows.net/coco2014/val2014.zip

Unzip the Images into dataset:
```
dataset/train
dataset/val
```
Download and create Symbolic Link to dummy: 
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip train2014.zip
unzip val2014.zip
mkdir -p dataset/train
mkdir -p dataset/val
ln -s `pwd`/val2014 dataset/val/dummy
ln -s `pwd`/train2014 dataset/train/dummy
```

#### Training a network

```
th train.lua -data <path to any image dataset>  -style_image path/to/img.jpg
```
I recommend this params for test:
```
th train.lua -data datasets/ -style_image $style -style_size 1024  -model johnson -batch_size 4 -learning_rate 1e-2 -style_weight 10 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -backend cudnn -save_every 10000
```
### Testing

```
th test.lua -input_image path/to/image.jpg -model data/checkpoints/model.t7
```

### Process

Stylize an image.
```
th stylization_process.lua -model data/out/model.t7 -input_image data/readme_pics/kitty.jpg -noise_depth 3
```
Again, `noise_depth` should be consistent with training setting.

### Example

![Cezanne](data/textures/cezanne.jpg)

![Original](data/readme_pics/kitty.jpg)

![Processed](data/readme_pics/kitty_cezanne.jpg)

# Hardware
- The code was tested with 12GB NVIDIA Titan X GPU and Ubuntu 14.04.
- You may decrease `batch_size`, `image_size` if the model do not fit your GPU memory.
- The pretrained models do not need much memory to sample.

# Credits

The code is based on [Justin Johnson's great code](https://github.com/jcjohnson/neural-style) for artistic style.

The work was supported by [Yandex](https://www.yandex.ru/) and [Skoltech](http://sites.skoltech.ru/compvision/).
