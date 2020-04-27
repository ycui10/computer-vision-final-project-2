#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import image
import model
import ssl

content_path = 'input/content/james.jpg'
style_path = 'input/style/Vincent_van_Gogh_604.jpg'

#image.py用的github上的原码，我过几天再更新一版

if __name__ == "__main__":
    best, best_loss = model.run(content_path, style_path, iteration=1000)
    image.saveimg(best, 'output/best.jpg')
