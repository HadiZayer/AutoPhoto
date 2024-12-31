import tensorflow as tf
import numpy as np

from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm


def mixed_cnn(scaledimages, **kwargs):
    print(scaledimages.get_shape())

    image_tensor = scaledimages[:, :, :, :3]

    features_left = 512 - 2*224

    feats1 = scaledimages[:, :, 0, 3]
    feats2 = scaledimages[:, :, 1, 3]
    feats3 = scaledimages[:, :features_left, 2, 3]
    aesthetic_features = tf.concat([feats1, feats2, feats3], axis=1)
    print(aesthetic_features.get_shape())

    activ = tf.nn.relu
    cnn = nature_cnn

    cnn_feats = cnn(image_tensor)
    processed_aes_feats = activ(linear(aesthetic_features, 'aes_fc', n_hidden=512))

    combined_feats = tf.concat([cnn_feats, processed_aes_feats], axis=1)

    # temp added
    # combined_feats = activ(linear(combined_feats, 'mid_fc', n_hidden=512))
    print(combined_feats.get_shape())

    return activ(linear(combined_feats, 'final_fc', n_hidden=512))
    # return activ(linear(combined_feats, 'final_fc', n_hidden=256))

def mixed_cnn_only_aes(scaledimages, **kwargs):
    print(scaledimages.get_shape())


    features_left = 512 - 2*224

    feats1 = scaledimages[:, :, 0, 3]
    feats2 = scaledimages[:, :, 1, 3]
    feats3 = scaledimages[:, :features_left, 2, 3]
    aesthetic_features = tf.concat([feats1, feats2, feats3], axis=1)

    activ = tf.nn.relu

    processed_aes_feats = activ(linear(aesthetic_features, 'aes_fc', n_hidden=512))
    combined_feats = processed_aes_feats
    print(combined_feats.get_shape())

    return activ(linear(combined_feats, 'final_fc', n_hidden=512))


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # scaled_images = tf.keras.applications.mobilenet.preprocess_input(scaled_images)

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def big_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    scaled_images = tf.keras.applications.mobilenet.preprocess_input(scaled_images)

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=128, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_5 = conv_to_fc(layer_5)
    return activ(linear(layer_5, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def nasnet(scaled_images, **kwargs):
    print('--------------- loading nasnet --------------')
    activ = tf.nn.relu
    scaled_images = tf.keras.applications.nasnet.preprocess_input(scaled_images)
    extracted_features = tf.keras.applications.NASNetMobile(include_top=False, weights='imagenet',
                                                            input_tensor=scaled_images).output
    extracted_features = conv_to_fc(extracted_features)
    extracted_features = linear(extracted_features, 'fc', n_hidden=512, init_scale=np.sqrt(2))
    extracted_features = activ(extracted_features)
    print('-----------finished loading nasnet-------------')
    return extracted_features

def mobilenet(scaled_images, **kwargs):
    print('-----------loading mobile net-------------')
    activ = tf.nn.relu
    scaled_images = tf.keras.applications.mobilenet.preprocess_input(scaled_images)
    extracted_features = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                      input_tensor=scaled_images).output
    extracted_features = conv_to_fc(extracted_features)
    extracted_features = linear(extracted_features, 'fc', n_hidden=512, init_scale=np.sqrt(2))
    extracted_features = activ(extracted_features)
    print('-----------finished loading mobile net-------------')
    return extracted_features

def resnet(scaled_images, **kwargs):
    print('-----------loading resnet-------------')
    activ = tf.nn.relu
    scaled_images = tf.keras.applications.resnet50.preprocess_input(scaled_images)
    extracted_features = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                                         input_tensor=scaled_images).output
    extracted_features = conv_to_fc(extracted_features)
    extracted_features = linear(extracted_features, 'fc', n_hidden=512, init_scale=np.sqrt(2))
    extracted_features = activ(extracted_features)
    print('-----------finished loading resnet-------------')
    return extracted_features