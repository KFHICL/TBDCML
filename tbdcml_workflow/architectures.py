from __future__ import annotations

import tensorflow as tf


def get_padding_shape(height, width, multiple=32):
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return ((0, pad_height), (0, pad_width))


def _application_backbone(factory, input_shape, **kwargs):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    base_model = factory(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        **kwargs,
    )
    output_layer = base_model(inputs=input_layer)
    return input_layer, output_layer


def Xception_Model(inputShape):
    return _application_backbone(tf.keras.applications.Xception, inputShape, name="xception")


def mobileNetV2_Model(inputShape):
    return _application_backbone(tf.keras.applications.MobileNetV2, inputShape, name="mobilenetV2")


def VGG16_Model(inputShape):
    return _application_backbone(tf.keras.applications.VGG16, inputShape, name="vgg16")


def ResNet50_Model(inputShape):
    return _application_backbone(tf.keras.applications.ResNet50, inputShape, name="ResNet50")


def ResNet50V2_Model(inputShape):
    return _application_backbone(tf.keras.applications.ResNet50V2, inputShape, name="ResNet50V2")


def InceptionV3_Model(inputShape):
    return _application_backbone(tf.keras.applications.InceptionV3, inputShape, name="InceptionV3")


def InceptionResNetV2_Model(inputShape):
    return _application_backbone(tf.keras.applications.InceptionResNetV2, inputShape, name="InceptionResNetV2")


def DenseNet121_Model(inputShape):
    return _application_backbone(tf.keras.applications.DenseNet121, inputShape, name="DenseNet121")


def NASNetMobile_Model(inputShape):
    return _application_backbone(tf.keras.applications.NASNetMobile, inputShape, name="NASNetMobile")


def EfficientNetV2S_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.EfficientNetV2S,
        inputShape,
        include_preprocessing=False,
        name="efficientnetv2-s",
    )


def EfficientNetV2M_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.EfficientNetV2M,
        inputShape,
        include_preprocessing=False,
        name="efficientnetv2-m",
    )


def EfficientNetV2L_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.EfficientNetV2L,
        inputShape,
        include_preprocessing=False,
        name="efficientnetv2-l",
    )


def ConvNeXtTiny_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.ConvNeXtTiny,
        inputShape,
        include_preprocessing=False,
        name="ConvNeXtTiny",
    )


def ConvNeXtSmall_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.ConvNeXtSmall,
        inputShape,
        include_preprocessing=False,
        name="ConvNeXtSmall",
    )


def ConvNeXtLarge_Model(inputShape):
    return _application_backbone(
        tf.keras.applications.ConvNeXtLarge,
        inputShape,
        include_preprocessing=False,
        name="ConvNeXtLarge",
    )


def applyDecoder(input_layer, x, outputShape, params):
    bn_shape = x.shape
    in_shape = bn_shape[1]
    out_shape = outputShape[1]
    s = int(out_shape / in_shape)
    pd = "same"
    if in_shape == 5:
        x = tf.keras.layers.Resizing(
            height=7,
            width=7,
            interpolation="bilinear",
            crop_to_aspect_ratio=False,
        )(x)
        s = int(out_shape / 7)

    x = tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="linear")(x)
    model = tf.keras.Model(input_layer, outputs)
    return model
