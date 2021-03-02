# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
import os
import numpy as np


EPS = 1e-9


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=EPS):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut', kernel_size=1, stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Deeplabv3(input_shape=(None, None, 3), classes=9, OS=16, activation='softmax', drop_pro=0.5, middle_repeat=16):
    img_input = Input(input_shape)
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)

    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(middle_repeat):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
    # end of feature extractor Xception_71

    # branching for Atrous Spatial Pyramid Pooling
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x1: K.expand_dims(x1, 1))(b4)
    b4 = Lambda(lambda x2: K.expand_dims(x2, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=EPS)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    # size_before = tf.keras.backend.int_shape(x)
    size_before = tf.shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size_before[1:3], align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=EPS)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    # rate = (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=EPS)
    # rate = (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=EPS)
    # rate = (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=EPS)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # 1X1 conv
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=EPS)(x)
    x = Activation('relu')(x)
    x = Dropout(drop_pro)(x)
    # DeepLab v.3+ decoder
    # size_before2 = tf.keras.backend.int_shape(x)
    size_before2 = tf.shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize_bilinear(xx, size_before2[1:3], align_corners=True))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=EPS)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])

    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=EPS)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=EPS)
    # you can use it with arbitary number of classes
    if classes == 2:
        last_layer_name = 'logits_semantic_2'
    elif classes == 9:
        last_layer_name = 'custom_logits_semantic_9'
    elif classes == 8:
        last_layer_name = 'custom_logits_semantic_8'
    elif classes == 7:
        last_layer_name = 'custom_logits_semantic_7'  # custom_logits_semantic_7
    else:
        last_layer_name = 'custom_logits_semantic'
    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    # size_before3 = tf.keras.backend.int_shape(img_input)
    size_before3 = tf.shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize_bilinear(xx, size_before3[1:3], align_corners=True))(x)
    inputs = img_input
    # if activation in {'softmax', 'sigmoid'}:
    x = tf.keras.layers.Activation(activation)(x)
    model = Model(inputs, x, name='deeplabv3plus')

    return model


def load_weights_from_dirs(weights_dir, save_model_path):
    model = Deeplabv3(input_shape=(None, None, 3), classes=9)
    for layer in model.layers[0:394]:   # 358, 394 index
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(weights_dir, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    filenames = 'deeplabv3plus_xception_71_imagenet.h5'
    model.save_weights(os.path.join(save_model_path, filenames))


if __name__ == '__main__':
    weights_dir = './weights_keras_71'
    save_model_path = './models_deeplabv3plus_xception'
    # load_weights_from_dirs(weights_dir, save_model_path)
    model = Deeplabv3(input_shape=(None, None, 3), classes=9)
    model.summary()

