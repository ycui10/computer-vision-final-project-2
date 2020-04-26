#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import image
import model
import ssl

content_path = 'input/content/james.jpg'
style_path = 'input/style/Vincent_van_Gogh_604.jpg'

# content = image.loadimg(content_path).astype('uint8')
# style = image.loadimg(style_path).astype('uint8')

# Content layer where will pull our feature maps
content_layers = ['block4_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

if __name__ == "__main__":
    print(f'GPU AVAILABLE :{tf.test.is_gpu_available()}')
    best, best_loss = model.run_nst(content_path, style_path, iteration=1000)
    image.saveimg(best, 'output/output10.jpg')
