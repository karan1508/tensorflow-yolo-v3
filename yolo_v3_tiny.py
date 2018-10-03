# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo_v3 import _conv2d_fixed_padding, _fixed_padding, _get_size, _detection_layer, _upsample
from tensorflow import keras

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10., 14.), (23., 27.), (37., 58.), (81., 82.), (135., 169.), (344., 319.)]


def yolo_v3_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 tiny model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    predictions = model_fn(inputs, mode='inference', num_classes=num_classes,
                           img_size=img_size, data_format=data_format)

    return predictions
    # with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
    #     with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
    #         with slim.arg_scope([slim.conv2d],
    #                         normalizer_fn=slim.batch_norm,
    #                         normalizer_params=batch_norm_params,
    #                         biases_initializer=None,
    #                         activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
    #
    #             with tf.variable_scope('yolo-v3-tiny'):
    #                 for i in range(6):
    #                     inputs = _conv2d_fixed_padding(inputs, 16 * pow(2, i), 3)
    #
    #                     if i == 4:
    #                         route_1 = inputs
    #
    #                     if i < 5:
    #                         inputs = slim.max_pool2d(inputs, [2, 2], scope='pool2')
    #                     else:
    #                         inputs = slim.max_pool2d(inputs, [2, 2], stride=1, padding="SAME",
    #                                                  scope='pool2')
    #
    #                     # inputs = slim.max_pool2d(inputs, [2, 2], scope='pool2')
    #
    #                 inputs = _conv2d_fixed_padding(inputs, 1024, 3)
    #                 inputs = _conv2d_fixed_padding(inputs, 256, 1)
    #                 route_2 = inputs
    #
    #                 inputs = _conv2d_fixed_padding(inputs, 512, 3)
    #                 # inputs = _conv2d_fixed_padding(inputs, 255, 1)
    #
    #                 detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
    #                 detect_1 = tf.identity(detect_1, name='detect_1')
    #
    #                 inputs = _conv2d_fixed_padding(route_2, 128, 1)
    #                 upsample_size = route_1.get_shape().as_list()
    #                 inputs = _upsample(inputs, upsample_size, data_format)
    #
    #                 inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)
    #
    #                 inputs = _conv2d_fixed_padding(inputs, 256, 3)
    #                 # inputs = _conv2d_fixed_padding(inputs, 255, 1)
    #
    #                 detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
    #                 detect_2 = tf.identity(detect_2, name='detect_2')
    #
    #                 detections = tf.concat([detect_1, detect_2], axis=1)
    #                 detections = tf.identity(detections, name='detections')
    #                 return detections


def conv_block(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs)
    padding = 'same' if strides == 1 else 'valid'
    inputs = keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False)(inputs)
    inputs = keras.layers.BatchNormalization(epsilon=_BATCH_NORM_EPSILON,
                                             scale=True, )(inputs)
    inputs = keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
    return inputs


def yolo_layers(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = keras.layers.Conv2D(
        num_anchors * (5 + num_classes),
        kernel_size=1,
        strides=1,
        activation=None,
        bias_initializer=tf.zeros_initializer()
    )(inputs)

    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes],
                                                           axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions


def model_fn(inputs, mode, **params):
    num_classes = params.get('num_classes')
    data_format = params.get('data_format')
    img_size = params.get('img_size')
    for i in range(6):
        inputs = conv_block(inputs, 16 * pow(2, i), 3)

        if i == 4:
            route_1 = inputs

        if i < 5:
            inputs = keras.layers.MaxPooling2D(pool_size=[2, 2])(inputs)
        else:
            inputs = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=1)(inputs)

    inputs = conv_block(inputs, 1024, 3)
    inputs = conv_block(inputs, 256, 1)
    route_2 = inputs

    inputs = conv_block(inputs, 512, 3)
    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

    detect_1 = yolo_layers(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
    detect_1 = tf.identity(detect_1, name='detect_1')

    inputs = conv_block(route_2, 128, 1)
    upsample_size = route_1.get_shape().as_list()
    inputs = _upsample(inputs, upsample_size, data_format)

    inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)

    inputs = conv_block(inputs, 256, 3)
    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

    detect_2 = yolo_layers(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
    detect_2 = tf.identity(detect_2, name='detect_2')

    detections = tf.concat([detect_1, detect_2], axis=1)
    detections = tf.identity(detections, name='detections')
    return detections
