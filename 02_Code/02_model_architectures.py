#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:06:58 2019

@author: Manita
"""
#******************************************************************************
# Basic
#******************************************************************************
from keras.layers import BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def basic_model(img_size, n_classes=16, opt='adam', metric=f1):
    
    in_shape=(img_size,img_size, 3)

    cnn = models.Sequential()
    # conv 1 
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu',
                            input_shape = in_shape))
    # pool 1
    cnn.add(layers.MaxPooling2D((2, 2)))
    # conv 2
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    # pool 2
    cnn.add(layers.MaxPooling2D((2, 2)))
    # flat
    cnn.add(layers.Flatten())
    # dense
    cnn.add(layers.Dense(128, activation = 'relu'))
    # output
    cnn.add(layers.Dense(n_classes, activation = 'softmax'))
    
    # Compile
    cnn.compile(loss='categorical_crossentropy', optimizer = opt, metrics=[metric])
    
    cnn.summary()
    return cnn


def cnn_sigmoid(img_size, n_classes=16, opt='adam', metric=f1):
    
    in_shape=(img_size,img_size, 3)

    cnn = models.Sequential()
    # conv 1 
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'sigmoid',
                            input_shape = in_shape))
    # pool 1
    cnn.add(layers.MaxPooling2D((2, 2)))
    # conv 2
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'sigmoid'))
    # pool 2
    cnn.add(layers.MaxPooling2D((2, 2)))
    # flat
    cnn.add(layers.Flatten())
    # dense
    cnn.add(layers.Dense(128, activation = 'sigmoid'))
    # output
    cnn.add(layers.Dense(n_classes, activation = 'softmax'))
    
    # Compile
    cnn.compile(loss='categorical_crossentropy', optimizer = opt, metrics=[metric])
    
    cnn.summary()
    return cnn


def cnn_2(img_size, n_classes=16, opt='adam', metric=f1):
    
    in_shape=(img_size,img_size, 3)

    cnn = models.Sequential()
    # conv 1 
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu',
                            input_shape = in_shape))
    # pool 1
    cnn.add(layers.MaxPooling2D((2, 2)))
    # conv 2
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    # pool 2
    cnn.add(layers.MaxPooling2D((2, 2)))
    # flat
    cnn.add(layers.Flatten())
    # dropout
    cnn.add(layers.Dropout(0.5))
    # dense
    cnn.add(layers.Dense(128, activation = 'relu'))
    # output
    cnn.add(layers.Dense(n_classes, activation = 'softmax'))
    
    # Compile
    cnn.compile(loss='categorical_crossentropy', optimizer = opt, metrics=[metric])
    
    cnn.summary()
    return cnn



def cnn_3(img_size, n_classes=16, opt='adam', metric=f1):
    
    in_shape=(img_size,img_size, 3)

    cnn = models.Sequential()
    # conv 1 
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu',
                            input_shape = in_shape))
    # pool 1
    cnn.add(layers.MaxPooling2D((2, 2)))
    
    # batcnormalization
    cnn.add(layers.BatchNormalization())
    # conv 2
    cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    # pool 2
    cnn.add(layers.MaxPooling2D((2, 2)))
    # flat
    cnn.add(layers.Flatten())
    # dense
    cnn.add(layers.Dense(128, activation = 'relu'))
    # output
    cnn.add(layers.Dense(n_classes, activation = 'softmax'))
    # Compile
    cnn.compile(loss='categorical_crossentropy', optimizer = opt, metrics=[metric])
    
    cnn.summary()
    return cnn


# LeNet-5 *****************************************************************
def lenet_5(img_size, n_classes=16, opt='sgd', metric = f1):
    
    in_shape=(img_size,img_size, 3)
    
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(filters=20, kernel_size=5,
                          padding='same', activation='relu')(in_layer)
    pool1 = layers.MaxPool2D()(conv1)
    conv2 = layers.Conv2D(filters=50, kernel_size=5,
                          padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D()(conv2)
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(500, activation='relu')(flatten)
    preds = layers.Dense(n_classes, activation='softmax')(dense1)

    model = models.Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=[metric])
    model.summary()
    return model


# AlexNet *****************************************************************

def alexnet(img_size, n_classes=16, opt='sgd', metric = f1):
    
    in_shape=(img_size,img_size, 3)
    
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)
    pool1 = layers.MaxPool2D(2, 2)(conv1)
    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(2, 2)(conv2)
    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D(2, 2)(conv4)
    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(128, activation='relu')(flattened)
    drop1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(128, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)
    preds = layers.Dense(n_classes, activation='softmax')(drop2)

    model = models.Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=[f1])
    return model


# VGG16 *****************************************************************
'''
from functools import partial

conv3 = partial(layers.Conv2D,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu')

def block(in_tensor, filters, n_convs):
    conv_block = in_tensor
    for _ in range(n_convs):
        conv_block = conv3(filters=filters)(conv_block)
    return conv_block

def _vgg(img_size,n_classes=16,opt='sgd',n_stages_per_blocks=[2, 2, 3, 3, 3], metric = f1):
    
    in_shape=(img_size,img_size, 3)
    in_layer = layers.Input(in_shape)

    block1 = block(in_layer, 64, n_stages_per_blocks[0])
    pool1 = layers.MaxPool2D()(block1)
    block2 = block(pool1, 128, n_stages_per_blocks[1])
    pool2 = layers.MaxPool2D()(block2)
    block3 = block(pool2, 256, n_stages_per_blocks[2])
    pool3 = layers.MaxPool2D()(block3)
    block4 = block(pool3, 512, n_stages_per_blocks[3])
    pool4 = layers.MaxPool2D()(block4)
    block5 = block(pool4, 512, n_stages_per_blocks[4])
    pool5 = layers.MaxPool2D()(block5)
    flattened = layers.GlobalAvgPool2D()(pool5)

    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(128, activation='relu')(dense1)
    preds = layers.Dense(128, activation='softmax')(dense2)

    model = models.Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=[f1])
    model.summary()
    return model

def vgg16(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _vgg(in_shape, n_classes, opt, metric = f1)

'''

# VGG16 V2*****************************************************************

def vgg16(img_size, n_classes=16, opt='adam', metric=f1):
    #Instantiate an empty model
    in_shape=(img_size,img_size, 3)
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=in_shape, padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
    ])
    
    model.summary()
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[f1])
    return model

# ResNet *****************************************************************
    
def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)

def conv1(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1)(in_tensor)
    return _after_conv(conv)

def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2)(in_tensor)
    return _after_conv(conv)

def conv3(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(in_tensor)
    return _after_conv(conv)

def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)

def resnet_block_wo_bottlneck(in_tensor, filters, downsample=False):
    if downsample:
        conv1_rb = conv3_downsample(in_tensor, filters)
    else:
        conv1_rb = conv3(in_tensor, filters)
    conv2_rb = conv3(conv1_rb, filters)

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    result = layers.Add()([conv2_rb, in_tensor])

    return layers.Activation('relu')(result)

def resnet_block_w_bottlneck(in_tensor,
                             filters,
                             downsample=False,
                             change_channels=False):
    if downsample:
        conv1_rb = conv1_downsample(in_tensor, int(filters/4))
    else:
        conv1_rb = conv1(in_tensor, int(filters/4))
    conv2_rb = conv3(conv1_rb, int(filters/4))
    conv3_rb = conv1(conv2_rb, filters)

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    elif change_channels:
        in_tensor = conv1(in_tensor, filters)
    result = layers.Add()([conv3_rb, in_tensor])

    return result

def _pre_res_blocks(in_tensor):
    conv = layers.Conv2D(64, 7, strides=2, padding='same')(in_tensor)
    conv = _after_conv(conv)
    pool = layers.MaxPool2D(3, 2, padding='same')(conv)
    return pool

def _post_res_blocks(in_tensor, n_classes):
    pool = layers.GlobalAvgPool2D()(in_tensor)
    preds = layers.Dense(n_classes, activation='softmax')(pool)
    return preds

def convx_wo_bottleneck(in_tensor, filters, n_times, downsample_1=False):
    res = in_tensor
    for i in range(n_times):
        if i == 0:
            res = resnet_block_wo_bottlneck(res, filters, downsample_1)
        else:
            res = resnet_block_wo_bottlneck(res, filters)
    return res

def convx_w_bottleneck(in_tensor, filters, n_times, downsample_1=False):
    res = in_tensor
    for i in range(n_times):
        if i == 0:
            res = resnet_block_w_bottlneck(res, filters, downsample_1, not downsample_1)
        else:
            res = resnet_block_w_bottlneck(res, filters)
    return res

def _resnet(in_shape=(50,50,3),
            n_classes=16,
            opt='sgd',
            #convx=[64, 128, 256, 512],
            convx=[64, 128, 200, 300],
            n_convx=[2, 2, 2, 2],
            convx_fn=convx_wo_bottleneck,
            metric = f1):
    in_layer = layers.Input(in_shape)

    downsampled = _pre_res_blocks(in_layer)

    conv2x = convx_fn(downsampled, convx[0], n_convx[0])
    conv3x = convx_fn(conv2x, convx[1], n_convx[1], True)
    conv4x = convx_fn(conv3x, convx[2], n_convx[2], True)
    conv5x = convx_fn(conv4x, convx[3], n_convx[3], True)

    preds = _post_res_blocks(conv5x, n_classes)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=[f1])
    model.summary()
    return model


def resnet18(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _resnet(in_shape=in_shape, n_classes = n_classes, opt = opt)

def resnet34(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _resnet(in_shape,
                  n_classes,
                  opt,
                  n_convx=[3, 4, 6, 3])

def resnet50(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _resnet(in_shape,
                  n_classes,
                  opt,
                  [256, 512, 1024, 2048],
                  [3, 4, 6, 3],
                  convx_w_bottleneck)

def resnet101(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _resnet(in_shape,
                  n_classes,
                  opt,
                  [256, 512, 1024, 2048],
                  [3, 4, 23, 3],
                  convx_w_bottleneck)

def resnet152(img_size, n_classes=16, opt='sgd'):
    in_shape=(img_size,img_size, 3)
    return _resnet(in_shape,
                  n_classes,
                  opt,
                  [256, 512, 1024, 2048],
                  [3, 8, 36, 3],
                  convx_w_bottleneck)

