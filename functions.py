# TT
# Code for ResNet copied from
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn, nn_ops, array_ops
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import pickle
import keras.backend as K
import six
import functools


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    # if epoch > 180 * 5:
    #     lr *= 0.5e-3
    # elif epoch > 160 * 5:
    #     lr *= 1e-3
    if epoch > 120 * 5:
        lr *= 0.5e-3
    elif epoch > 80 * 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 conv_layer=Conv2D,
                 **kwargs):
    """2D Convolution-Batch Normalization-Activation stack builder
    Arguments:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string or None): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
        conv_layer (keras.Layer): which layer to use as the basis for ResNet
            (usually either Conv2D or ConvDecomposed2D)
    Returns:
        x (tensor): tensor as input to the next layer
    """
    if conv_layer != Conv2D:
        str_parameter = "decomposition_rank"
        if conv_layer == CircConv2D:
            str_parameter = "n"
        if kwargs[str_parameter] == -1:
            conv_layer = Conv2D
            kwargs = {}
    conv = conv_layer(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4),
                      **kwargs)
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, conv_layer=Conv2D,
              compress_first=True,
              compress_schedule=None, **kwargs):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved
    (downsampled) by a convolutional layer with strides=2, while
    the number of filters is doubled. Within each stage,
    the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
        conv_layer (keras.Layer): which layer to use as the basis for ResNet
            (usually either Conv2D or ConvDecomposed2D)
        compress_schedule (list): schedule of decomposition_rank/n
    Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, in [a])')
    # start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    str_parameter = "decomposition_rank"
    if conv_layer == CircConv2D:
        str_parameter = "n"
    cnt = 0
    if compress_schedule is not None:
        assert len(compress_schedule) == 3 * num_res_blocks + 1
    inputs = Input(shape=input_shape)
    if compress_schedule is not None:
        kwargs[str_parameter] = compress_schedule[cnt]
        cnt += 1
        x = resnet_layer(inputs=inputs, conv_layer=conv_layer, **kwargs)
    elif compress_first:
        x = resnet_layer(inputs=inputs, conv_layer=conv_layer, **kwargs)
    else:
        x = resnet_layer(inputs=inputs, conv_layer=Conv2D)
    # instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                strides = 2  # downsample
            if compress_schedule is not None:
                kwargs[str_parameter] = compress_schedule[cnt]
                cnt += 1
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             conv_layer=conv_layer,
                             **kwargs)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             conv_layer=conv_layer,
                             **kwargs)
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # linear projection residual shortcut
                # connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 conv_layer=conv_layer,
                                 **kwargs)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def define_and_compile_ResNet_model(input_shape, depth=32, compress_first=True,
                                    compress_schedule=None,
                                    conv_layer=Conv2D, **kwargs):
    # model name, depth and version
    model_type = 'ResNet%dv1' % depth
    model = resnet_v1(input_shape=input_shape, depth=depth,
                      conv_layer=conv_layer, compress_first=compress_first,
                      compress_schedule=compress_schedule,
                      **kwargs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['acc'])
    return model, model_type


# prepare model saving directory.
def prepare_model_saving_directory(model_type):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    return filepath


# prepare callbacks for model saving and for learning rate adjustment.
def standard_callbacks(model_type):
    checkpoint = ModelCheckpoint(
        filepath=prepare_model_saving_directory(model_type),
        monitor='val_acc',
        verbose=1,
        save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    return [checkpoint, lr_reducer, lr_scheduler]


# run training, with or without data augmentation.
def run_training(model, model_type, x_train, y_train, x_test, y_test,
                 history_file_dump_name,
                 steps_per_epoch=None,
                 batch_size=128, epochs=250, data_augmentation=True,
                 callbacks=None):
    if steps_per_epoch is None:
        steps_per_epoch = math.ceil(len(x_train) / batch_size)
    if callbacks is None:
        callbacks = standard_callbacks(model_type)
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train, seed=0)

        # fit the model on the batches generated by datagen.flow().
        history = model.fit(x=datagen.flow(x_train, y_train,
                                           batch_size=batch_size),
                            verbose=1,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)
    file_pi = open(history_file_dump_name, 'wb')
    pickle.dump(history.history, file_pi)
    file_pi.close()
    return history


def Clip_OperatorNorm(conv, inp_shape, clip_to):
    conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex128)
    conv_shape = conv.get_shape().as_list()
    padding = tf.constant([[0, 0], [0, 0],
                           [0, inp_shape[0] - conv_shape[0]],
                           [0, inp_shape[1] - conv_shape[1]]])
    transform_coeff = tf.signal.fft2d(tf.pad(conv_tr, padding))
    D, U, V = tf.linalg.svd(tf.transpose(transform_coeff, perm=[2, 3, 0, 1]))
    norm = tf.reduce_max(D)
    D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex128)
    clipped_coeff = tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped),
                                           V, adjoint_b=True))
    clipped_conv_padded = tf.math.real(tf.signal.ifft2d(
        tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
    return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
                    [0] * len(conv_shape), conv_shape), norm


def circulant_sinvals(kernel, input_shape):
    kernel_shape = kernel.shape
    kernel_tr = tf.transpose(tf.cast(kernel, tf.complex128), [2, 3, 4, 0, 1])

    padding = tf.constant([[0, 0], [0, 0], [0, 0],
                           [0, input_shape[0] - kernel_shape[0]],
                           [0, input_shape[1] - kernel_shape[1]]])

    ps = tf.transpose(tf.signal.fft2d(tf.pad(kernel_tr, padding)),
                      [3, 4, 0, 1, 2])  # puv from Sedghi
    ps_fft = tf.signal.fft(ps)  # diagonilizing puv-s
    tensorok = tf.transpose(ps_fft, [0, 1, 4, 2,
                                     3])  # corresponds to moving into
    # blocks in theory
    sinval = tf.linalg.svd(tensorok)  # final values
    return sinval


def svd_reconstruction(svds):
    u, sigma, v = svds
    sigma = tf.cast(sigma, tf.complex128)

    # reconstructing svd
    rec0 = tf.matmul(u,
                     tf.matmul(tf.linalg.diag(tf.cast(sigma, tf.complex128)),
                               v, adjoint_b=True))

    detensorok = tf.transpose(rec0, [0, 1, 3, 4, 2])
    ps_ifft = tf.signal.ifft(detensorok)
    res = tf.transpose(
        tf.signal.ifft2d(tf.transpose(ps_ifft, [2, 3, 4, 0, 1])),
        [3, 4, 0, 1, 2])
    return res


class Clipping(tf.keras.callbacks.Callback):
    def __init__(self, clip_to, mode="decomposed", compress_first=True):
        tf.keras.callbacks.Callback.__init__(self)
        self.clip_to = clip_to
        if mode not in ("decomposed", "circulant", "simple"):
            raise ValueError("Unsupported mode")
        self.mode = mode
        self.compress_first = compress_first

    @staticmethod
    def get_new_K(K1, K2, K3):
        # (m, R) -> (m, min(m, R)) -- q: q.T @ q = I, (min(m, R), R) -- r
        q1, r1 = tf.linalg.qr(K1, full_matrices=False)
        q3, r3 = tf.linalg.qr(tf.transpose(K3), full_matrices=False)
        middle_k = full_tt(r1, K2, tf.transpose(r3))
        return q1, middle_k, tf.transpose(q3)

    def on_epoch_end(self, epochs, logs=None):
        if self.mode == "decomposed":
            for layer in self.model.layers:
                if layer.name[:17] == "conv_decomposed2d":
                    K1, K2, K3 = self.get_new_K(layer.K1, layer.K2, layer.K3)
                    K2 = tf.transpose(Clip_OperatorNorm(K2,
                                                        layer.input_shape[1:3],
                                                        self.clip_to)[0],
                                      perm=[2, 0, 1, 3])
                    K.set_value(layer.K1, K1)
                    K.set_value(layer.K3, K3)
                    K.set_value(layer.K2, K2)
        elif self.mode == "circulant":
            for layer in self.model.layers:
                if layer.name.startswith("circ_conv2d"):
                    sinvals, us, vs = circulant_sinvals(layer.K,
                                                        layer.input_shape[1:3])
                    sinvals_clipped = tf.cast(
                        tf.minimum(sinvals, self.clip_to), tf.complex128)
                    new_K = svd_reconstruction((us, sinvals_clipped, vs))
                    k = layer.K.shape[0]
                    K.set_value(layer.K, new_K[:k, :k])
        else:
            is_first = self.compress_first
            for layer in self.model.layers:
                if layer.name.startswith("conv2d"):
                    if not is_first:
                        is_first = True
                        continue
                    K.set_value(
                        layer.kernel,
                        Clip_OperatorNorm(layer.kernel, layer.input_shape[1:3],
                                          self.clip_to)[0]
                    )


def plot_loss_acc(history, ub_loss=2, ub_error=0.4, max_len=None):
    if max_len is None:
        max_len = len(history['loss'])
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].plot(history['loss'][4:max_len:5], label='train')
    axs[0].plot(history['val_loss'][4:max_len:5], label='val')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylim(0, ub_loss)
    axs[1].plot(1 - np.array(history['acc'][4:max_len:5]), label='train')
    axs[1].plot(1 - np.array(history['val_acc'][4:max_len:5]), label='val')
    axs[1].set_title('Error')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylim(0, ub_error)
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')


def complex_convolution(inputs, kernel):
    """performs convolution for complex inputs and kernel"""
    inputs_real = tf.dtypes.cast(tf.math.real(inputs), tf.float32)
    inputs_imag = tf.dtypes.cast(tf.math.imag(inputs), tf.float32)
    kernel_real = tf.dtypes.cast(tf.math.real(kernel), tf.float32)
    kernel_imag = tf.dtypes.cast(tf.math.imag(kernel), tf.float32)

    outputs_real = tf.keras.backend.conv2d(inputs_real, kernel_real,
                                           [1] * len(inputs.shape), 'same')
    outputs_real -= tf.keras.backend.conv2d(inputs_imag, kernel_imag,
                                            [1] * len(inputs.shape), 'same')
    outputs_imag = tf.keras.backend.conv2d(inputs_real, kernel_imag,
                                           [1] * len(inputs.shape), 'same')
    outputs_imag += tf.keras.backend.conv2d(inputs_imag, kernel_real,
                                            [1] * len(inputs.shape), 'same')

    return tf.complex(outputs_real, outputs_imag)


# @tf.custom_gradient
# TODO: if we are adding custom gradient calculation, this is to be used
def fft_convolution(K, inputs, filters):  # TODO: add stride, padding and so on
    # getting shapes
    k, _, r, s, n = K.shape
    batch_size, w, h, c0 = inputs.shape

    # padding input to match kernel
    inputs_padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, r * n - c0]])
    input = tf.pad(inputs, inputs_padding)

    # fft along the last dimension
    input_fft_reshape = tf.signal.fft(
        tf.dtypes.cast(tf.reshape(input, (-1, w, h, r, n)), tf.complex128))
    kernel_fft = tf.signal.fft(tf.dtypes.cast(K, tf.complex128))

    # n (* 4) independent convolutions for complex inputs and kernel
    outputs = []
    # input_fft_reshape = tf.reshape(input_fft, (-1, w, h, r, n))
    for i in range(n):
        Y = complex_convolution(input_fft_reshape[:, :, :, :, i],
                                kernel_fft[:, :, :, :, i])
        outputs.append(Y)

    # stacking outputs
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 2, 3, 4, 0])

    # ifft & final reshape
    outputs = tf.math.real(tf.signal.ifft(outputs))
    outputs = tf.reshape(outputs, (-1, w, h, s * n))

    # truncating outputs to the filter shape
    outputs = outputs[:, :, :, :filters]

    # TODO: backprop
    # def gradient(dL):

    return outputs


def full_tt(K1, K2, K3):
    """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor)."""
    res = K1
    K2_reshaped = tf.reshape(K2, (K2.shape[0], -1))
    res = tf.matmul(res, K2_reshaped)
    res = tf.reshape(res, (-1, K3.shape[0]))
    res = tf.matmul(res, K3)
    res = tf.reshape(res, (K1.shape[0],) + K2.shape[1:-1] + (K3.shape[-1],))
    num_dims = len(K2.shape[1:-1])
    return tf.transpose(res, list(range(1, num_dims + 1)) + [0, num_dims + 1])


def full_circ(kernel, input_shape):
    """Converts the truncated circular version of kernel with size k k rn s
    into the full version with size k k rn sn. """

    m, _, r, s, n = kernel.shape

    full_kernel = tf.stack([tf.roll(kernel, i, -1) for i in range(n)], axis=-1)
    full_kernel = tf.reshape(tf.transpose(full_kernel, [0, 1, 2, 4, 3, 5]),
                             (m, m, r * n, s * n))

    return full_kernel[:, :, :input_shape[0], :input_shape[1]]


def faster_memory(convolution_op, inputs, K1, K2, K3):
    return convolution_op(inputs, full_tt(K1, K2, K3))


def slower_without_memory(convolution_op, inputs, K1, K2, K3):
    inputs1 = convolution_op(inputs,
                             tf.reshape(K1, (1, 1, K1.shape[0], K1.shape[1])))
    inputs2 = convolution_op(inputs1, tf.transpose(K2, perm=[1, 2, 0, 3]))
    return convolution_op(inputs2,
                          tf.reshape(K3, (1, 1, K3.shape[0], K3.shape[1])))


class CircConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 n,  # c0 = R * n, c2 = S * n
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CircConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,  # doesn't support strides
            padding=padding,  # as well as padding
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=1,  # does not support groups!
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.n = n
        self.K = None
        self.bias = None
        self._convolution_op = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)

        if input_channel < self.n:
            print("input_channel number is less than n, shrinking n forcibly")
            self.n = input_channel
        if self.filters < self.n:
            print("output_channel number is less than n, shrinking n forcibly")
            self.n = self.filters
        r, s = int(np.ceil(input_channel / self.n)), int(
            np.ceil(self.filters / self.n))

        self.K = self.add_weight(
            name='K',
            shape=(self.kernel_size[0], self.kernel_size[0], r, s, self.n),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        # the rest is copied from Conv build function
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        # outputs = fft_convolution(self.K, inputs, self.filters) # not used
        # because of for loop that slows down the calculation
        outputs = self._convolution_op(inputs, full_circ(self.K, (
        inputs.shape[-1], self.filters)))

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias,
                                           data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ConvDecomposed2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 decomposition_rank,
                 use_memory=True,
                 use_memory_test=True,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvDecomposed2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=1,  # does not support groups!
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.decomposition_rank = decomposition_rank
        self.K1 = None
        self.K2 = None
        self.K3 = None
        self.bias = None
        self._convolution_op = None
        self.use_memory = use_memory  # can be changed if we use too much
        # memory in singular clipping anyway
        self.use_memory_test = use_memory_test

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            input_smth = input_shape[1:-1]
        else:
            input_smth = input_shape[2:]
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        r1 = min(input_channel, self.decomposition_rank)
        r2 = min(self.filters, self.decomposition_rank)

        # if not self.use_memory:
        #     big_kernel = input_channel * self.filters * self.kernel_size[0]
        #     big_kernel *= self.kernel_size[1]
        #     padded_kernel = r1 * r2 * input_smth[0] * input_smth[1]
        #     self.use_memory = (big_kernel < padded_kernel)

        self.K1 = self.add_weight(
            name='K1',
            shape=(input_channel, r1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.K2 = self.add_weight(
            name='K2',
            shape=(r1,) + self.kernel_size + (r2,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.K3 = self.add_weight(
            name='K3',
            shape=(r2, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        # the rest is copied from Conv build function
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs, training=None):
        flag_training = training and self.use_memory
        flag_not_training = not training and self.use_memory_test
        if flag_training or flag_not_training:
            outputs = faster_memory(self._convolution_op, inputs, self.K1,
                                    self.K2, self.K3)
        else:
            outputs = slower_without_memory(self._convolution_op, inputs,
                                            self.K1, self.K2, self.K3)
        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias,
                                           data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
