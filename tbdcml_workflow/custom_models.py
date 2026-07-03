from __future__ import annotations

import tensorflow as tf

from .architectures import get_padding_shape


def build_tbdcnet_model_cnn(inputShape, outputShape, params, normalizer):
    """Build the custom benchmark CNN used by the HPC and local scripts."""
    if params['L1kernel_regularizer'] > 0 and params['L2kernel_regularizer'] > 0:
        regularizer = tf.keras.regularizers.L1L2(l1=params['L1kernel_regularizer'], l2=params['L2kernel_regularizer'])
    elif params['L1kernel_regularizer'] > 0:
        regularizer = tf.keras.regularizers.L1(params['L1kernel_regularizer'])
    elif params['L2kernel_regularizer'] > 0:
        regularizer = tf.keras.regularizers.L2(params['L2kernel_regularizer'])
    else:
        regularizer = None

    input = tf.keras.layers.Input(shape=inputShape)
    x = normalizer(input)

    if params['downSample'] == 1 and params['pooling'] == 1:
        pad = get_padding_shape(inputShape[0], inputShape[1])
        x = tf.keras.layers.ZeroPadding2D(padding=(pad))(x)

    x = tf.keras.layers.Conv2D(
        filters=int(params['filterScale'] * 32),
        kernel_size=(int(params['layer1Kernel']), int(params['layer1Kernel'])),
        activation=params['conv1Activation'],
        data_format='channels_last',
        padding='same',
        kernel_regularizer=regularizer,
    )(x)
    if params['batchNorm'] == 1:
        x = tf.keras.layers.BatchNormalization()(x)
    if params['pooling'] == 1:
        if params['downSample'] == 1:
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
        else:
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
    if params['dropout'] > 0:
        x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
    encoder1 = x

    if params['layer2'] == 1:
        x = tf.keras.layers.Conv2D(
            filters=int(params['filterScale'] * 64),
            kernel_size=(int(params['layer2Kernel']), int(params['layer2Kernel'])),
            activation=params['conv2Activation'],
            data_format='channels_last',
            padding='same',
            kernel_regularizer=regularizer,
        )(x)
        if params['batchNorm'] == 1:
            x = tf.keras.layers.BatchNormalization()(x)
        if params['pooling'] == 1:
            if params['downSample'] == 1:
                x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
            else:
                x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
        if params['dropout'] > 0:
            x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
        encoder2 = x

        if params['layer3'] == 1:
            x = tf.keras.layers.Conv2D(
                filters=int(params['filterScale'] * 128),
                kernel_size=(int(params['layer3Kernel']), int(params['layer3Kernel'])),
                activation=params['conv3Activation'],
                data_format='channels_last',
                padding='same',
                kernel_regularizer=regularizer,
            )(x)
            if params['batchNorm'] == 1:
                x = tf.keras.layers.BatchNormalization()(x)
            if params['pooling'] == 1:
                if params['downSample'] == 1:
                    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
                else:
                    x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
            if params['dropout'] > 0:
                x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
            encoder3 = x

            if params['layer4'] == 1:
                x = tf.keras.layers.Conv2D(
                    filters=int(params['filterScale'] * 256),
                    kernel_size=(int(params['layer4Kernel']), int(params['layer4Kernel'])),
                    activation=params['conv4Activation'],
                    data_format='channels_last',
                    padding='same',
                    kernel_regularizer=regularizer,
                )(x)
                if params['batchNorm'] == 1:
                    x = tf.keras.layers.BatchNormalization()(x)
                if params['pooling'] == 1:
                    if params['downSample'] == 1:
                        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
                    else:
                        x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
                if params['dropout'] > 0:
                    x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
                encoder4 = x

                if params['layer5'] == 1:
                    x = tf.keras.layers.Conv2D(
                        filters=int(params['filterScale'] * 512),
                        kernel_size=(int(params['layer5Kernel']), int(params['layer5Kernel'])),
                        activation=params['conv5Activation'],
                        data_format='channels_last',
                        padding='same',
                        kernel_regularizer=regularizer,
                    )(x)
                    if params['batchNorm'] == 1:
                        x = tf.keras.layers.BatchNormalization()(x)
                    if params['pooling'] == 1:
                        if params['downSample'] == 1:
                            x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
                        else:
                            x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
                    if params['dropout'] > 0:
                        x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
                    encoder5 = x

                    if params['layer6'] == 1:
                        x = tf.keras.layers.Conv2D(
                            filters=int(params['filterScale'] * 1024),
                            kernel_size=(int(params['layer6Kernel']), int(params['layer6Kernel'])),
                            activation=params['conv6Activation'],
                            data_format='channels_last',
                            padding='same',
                            kernel_regularizer=regularizer,
                        )(x)
                        if params['batchNorm'] == 1:
                            x = tf.keras.layers.BatchNormalization()(x)
                        if params['pooling'] == 1:
                            if params['downSample'] == 1:
                                x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
                            else:
                                x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
                        if params['dropout'] > 0:
                            x = tf.keras.layers.SpatialDropout2D(rate=params['dropout'])(x)
                        encoder6 = x

                        if params['ActivationUp'] == 0:
                            temp_activation = 'linear'
                        else:
                            temp_activation = params['conv6Activation']

                        if params['downSample'] == 1 and params['pooling'] == 1:
                            x = tf.keras.layers.Conv2DTranspose(
                                filters=int(params['filterScale'] * 512),
                                kernel_size=(int(params['layer6Kernel']), int(params['layer6Kernel'])),
                                strides=2,
                                padding='same',
                                activation=temp_activation,
                            )(x)
                        else:
                            x = tf.keras.layers.Conv2DTranspose(
                                filters=int(params['filterScale'] * 512),
                                kernel_size=(int(params['layer6Kernel']), int(params['layer6Kernel'])),
                                padding='same',
                                activation=temp_activation,
                            )(x)
                        if params['skipConnections'] == 1:
                            x = tf.keras.layers.Concatenate()([x, encoder5])

                    if params['ActivationUp'] == 0:
                        temp_activation = 'linear'
                    else:
                        temp_activation = params['conv5Activation']

                    if params['downSample'] == 1 and params['pooling'] == 1:
                        x = tf.keras.layers.Conv2DTranspose(
                            filters=int(params['filterScale'] * 256),
                            kernel_size=(int(params['layer5Kernel']), int(params['layer5Kernel'])),
                            strides=2,
                            padding='same',
                            activation=temp_activation,
                        )(x)
                    else:
                        x = tf.keras.layers.Conv2DTranspose(
                            filters=int(params['filterScale'] * 256),
                            kernel_size=(int(params['layer5Kernel']), int(params['layer5Kernel'])),
                            padding='same',
                            activation=temp_activation,
                        )(x)
                    if params['skipConnections'] == 1:
                        x = tf.keras.layers.Concatenate()([x, encoder4])

                if params['type'] == 'dense':
                    y = tf.keras.layers.Flatten()(x)
                    y = tf.keras.layers.Dense(64, activation='relu')(y)
                    y = tf.keras.layers.Dense(outputShape[0] * outputShape[1])(y)
                    y = tf.keras.layers.Reshape(outputShape)(y)

                if params['ActivationUp'] == 0:
                    temp_activation = 'linear'
                else:
                    temp_activation = params['conv4Activation']

                if params['downSample'] == 1 and params['pooling'] == 1:
                    x = tf.keras.layers.Conv2DTranspose(
                        filters=int(params['filterScale'] * 128),
                        kernel_size=(int(params['layer4Kernel']), int(params['layer4Kernel'])),
                        strides=2,
                        padding='same',
                        activation=temp_activation,
                    )(x)
                else:
                    x = tf.keras.layers.Conv2DTranspose(
                        filters=int(params['filterScale'] * 128),
                        kernel_size=(int(params['layer4Kernel']), int(params['layer4Kernel'])),
                        padding='same',
                        activation=temp_activation,
                    )(x)
                if params['skipConnections'] == 1:
                    x = tf.keras.layers.Concatenate()([x, encoder3])

        if params['ActivationUp'] == 0:
            temp_activation = 'linear'
        else:
            temp_activation = params['conv3Activation']

        if params['downSample'] == 1 and params['pooling'] == 1:
            x = tf.keras.layers.Conv2DTranspose(
                filters=int(params['filterScale'] * 64),
                kernel_size=(int(params['layer3Kernel']), int(params['layer3Kernel'])),
                strides=2,
                padding='same',
                activation=temp_activation,
            )(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(
                filters=int(params['filterScale'] * 64),
                kernel_size=(int(params['layer3Kernel']), int(params['layer3Kernel'])),
                padding='same',
                activation=temp_activation,
            )(x)
        if params['skipConnections'] == 1:
            x = tf.keras.layers.Concatenate()([x, encoder2])

    if params['ActivationUp'] == 0:
        temp_activation = 'linear'
    else:
        temp_activation = params['conv2Activation']

    if params['downSample'] == 1 and params['pooling'] == 1:
        x = tf.keras.layers.Conv2DTranspose(
            filters=int(params['filterScale'] * 32),
            kernel_size=(int(params['layer2Kernel']), int(params['layer2Kernel'])),
            strides=2,
            padding='same',
            activation=temp_activation,
        )(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(
            filters=int(params['filterScale'] * 32),
            kernel_size=(int(params['layer2Kernel']), int(params['layer2Kernel'])),
            padding='same',
            activation=temp_activation,
        )(x)
    if params['skipConnections'] == 1:
        x = tf.keras.layers.Concatenate()([x, encoder1])

    temp_activation = 'linear'

    if params['downSample'] == 1 and params['pooling'] == 1:
        x = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(int(params['layer1Kernel']), int(params['layer1Kernel'])),
            strides=2,
            padding='same',
            activation=temp_activation,
        )(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(int(params['layer1Kernel']), int(params['layer1Kernel'])),
            padding='same',
            activation=temp_activation,
        )(x)

    if params['downSample'] == 1 and params['pooling'] == 1:
        x = tf.keras.layers.Cropping2D(cropping=(pad))(x)

    output = y if params['type'] == 'dense' else x
    return tf.keras.Model(inputs=input, outputs=output)


def build_tbdcnet_unet(inputShape, outputShape, params, normalizer, seed):
    """Build the custom U-Net variant used in the local test workflow."""

    def double_convBlock(x, filters):
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        return x

    def downSamplingBlock(x, filters):
        skip = double_convBlock(x, filters)
        x = tf.keras.layers.MaxPool2D(2)(skip)
        x = tf.keras.layers.Dropout(params['dropout'])(x)
        return skip, x

    def upSamplingBlock(x, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.concatenate([x, skip])
        x = tf.keras.layers.Dropout(params['dropout'])(x)
        x = double_convBlock(x, filters)
        return x

    input = tf.keras.layers.Input(shape=inputShape)
    x = normalizer(input)
    if params['dsAugmentation'] == 1:
        x = tf.keras.layers.RandomFlip(mode='horizontal_and_vertical', seed=seed)(x)

    skip1, x1 = downSamplingBlock(x, 64)
    skip2, x2 = downSamplingBlock(x1, 128)
    skip3, x3 = downSamplingBlock(x2, 256)
    skip4, x4 = downSamplingBlock(x3, 512)

    bottleneck = double_convBlock(x4, 1024)

    u6 = upSamplingBlock(bottleneck, skip4, 512)
    u7 = upSamplingBlock(u6, skip3, 256)
    u8 = upSamplingBlock(u7, skip2, 128)
    u9 = upSamplingBlock(u8, skip1, 64)

    outputs = tf.keras.layers.Conv2D(1, 3, padding='same', activation='linear')(u9)
    return tf.keras.Model(input, outputs, name='U-Net')
