#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import image
from datetime import datetime
from tensorflow.keras.models import Model
# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )
                               

content_layers = ['block4_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def model_init():
    """
    this function will load pretrained(imagenet) vgg19 model and give access to output of intermedia layer
    then it will initialize a new model which take a picture as a input and output a list of vgg19 layer output.
    
    Return:
    return a model that input a picture and output the content feature and style feature
    """
    
    vgg19 = tf.keras.applications.VGG19(include_top = False,weights = 'imagenet')
    vgg19.trainable = False
    
    content_outputs = [vgg19.get_layer(layername).output for layername in content_layers]
    style_outputs = [vgg19.get_layer(layername).output for layername in style_layers]
    model_outputs = content_outputs + style_outputs
   
    
    model = Model(vgg19.input,model_outputs)
    
    return model    


def content_loss(base_content,target):
    c_loss = tf.reduce_mean(tf.square(base_content - target))/2
    return c_loss

def gram_matrix(input_tensor):
    channel = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor,[-1,channel]) # reshape img into dims of [H*W,C]
    gram = tf.matmul(a,a,transpose_a = True)
    return gram

def style_loss(base_style,target):
    a = gram_matrix(base_style)
    b = gram_matrix(target)
    h,w,c = base_style.get_shape().as_list()
    s_loss = tf.reduce_mean(tf.square(a - b))/(4*(h**2)*(w**2)*(c**2))
    return s_loss


def get_feature_representation(model,path,mode):
    """
    This helper function will load  picture and the model, 
    then produce different layer output according to the need
    
    Arguments:
     model : the model we initialized
     path : path to the picture
     mode : 'content' feature or 'style' feature
     
    Returns:
     return the content features or the style features
    """
    img = image.pre_process_img(path)
    feature_outputs = model(img)
    if mode == 'style':
        return [feature[0] for feature in feature_outputs[num_content_layers:]] # get the last several outputs
    if mode =='content':
        return [feature[0] for feature in feature_outputs[:num_content_layers]] # get the first several outputs
    

def loss(model,loss_weights,init_image,content_features,style_features):
    """
    The loss function
    
    Arguments:
    
     model : the model we are using
     loss_weights : the weights of each contribution in the loss function
                     (conten tloss weight,style loss weight,vatiation weight)
     init_image : the generated image upon which we would impose Gradient Descent
     content_features : the precomputed content picture feature
     style_features : the precomputed style picture feature
     
    Returns:
    
     the total loss we are going to optimize
    """
    style_weight,content_weight = loss_weights
    
    # feed the init image in the model,then we would get the 
    # content feature and the style feature from the layers 
    # we desire
    
    features = model(init_image)
    gen_style_feature = features[num_content_layers:]
    gen_content_feature = features[:num_content_layers]
    
    total_style_loss = 0
    total_content_loss = 0
    
    # equal seperate the layer weight of style loss
    # and accumulate the style loss from desired layers
    weight_per_style_layer = 1.0/ float(num_style_layers)
    for style_pic_features,gen_pic_stylefeatures in zip(style_features,gen_style_feature):
        total_style_loss += weight_per_style_layer * style_loss(style_pic_features,gen_pic_stylefeatures)
    
    # equal seperate the layer weight of content loss
    # and accumulate the content loss from desired layers
    weight_per_content_layer = 1.0/ float(num_content_layers)
    for content_pic_features,gen_pic_contentfeatures in zip(content_features,gen_content_feature):
        total_content_loss += weight_per_content_layer * content_loss(content_pic_features,gen_pic_contentfeatures)
    
    total_style_loss *= style_weight
    total_content_loss *= content_weight
    total_loss = total_style_loss + total_content_loss
    return total_loss,total_content_loss,total_style_loss
    

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        allloss = loss(**cfg)
    #Compute Gradient with respect to the generated image
    total_loss = allloss[0]
    return tape.gradient(total_loss,cfg['init_image']),allloss


def run_nst(content_path,style_path,iteration = 1000,content_weight = 1e3,style_weight = 1):
    model = model_init()
    for layer in model.layers:
        layer.trainable = False
        
    content_features = get_feature_representation(model,content_path,mode = 'content')
    style_features = get_feature_representation(model,style_path,mode = 'style')
    
    init_image = image.pre_process_img(content_path) # initialize the generated image with content image
    init_image = tf.Variable(init_image,dtype = tf.float32)
    
    opt = tf.keras.optimizers.Adam(5,beta_1 = 0.99,epsilon = 1e-1)
    
    epoch = 1
    loss_weights = (content_weight,style_weight)
    
    cfg = {
        'model':model,
        'loss_weights':loss_weights,
        'init_image':init_image,
        'content_features':content_features,
        'style_features':style_features
    }
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    #store the loss and the img
    best_loss, best_img = float('inf'), None
    imgs = []
    start = datetime.now()
    for i in range(iteration):
        
        grads, all_loss = compute_grads(cfg)
        losss, content_losss, style_losss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        

        if losss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = losss
            best_img = image.deprocess_img(init_image.numpy())

        if i % 100 == 0:
            end = datetime.now()
            print('[INFO]Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}'
                  .format(losss, style_losss, content_losss))
            print(f'100 iters takes {end -start}')
            start = datetime.now()
        if i % 500 == 0:
            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = image.deprocess_img(plot_img)
            path = 'output/output_' + str(i) + '.jpg'
            image.saveimg(plot_img, path)
            imgs.append(plot_img)


    return best_img, best_loss
    
    

