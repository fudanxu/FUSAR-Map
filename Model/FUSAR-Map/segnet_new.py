import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
# from .vgg16 import get_vgg_encoder
# from .mobilenet import get_mobilenet_encoder
# from .basic_models import vanilla_encoder
# from .resnet50 import get_resnet50_encoder


pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.1/" \
                 "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


def vanilla_encoder(input_height=256,  input_width=256):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(filter_size, (kernel, kernel), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad)))(x)
    x = (Conv2D(128, (kernel, kernel),
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size)))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad)))(x)
        x = (Conv2D(256, (kernel, kernel), padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size)))(x)
        levels.append(x)

    return img_input, levels


def get_vgg_encoder(input_height=256,  input_width=256, pretrained='imagenet'):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    if pretrained == 'imagenet':
        VGG_Weights_path = keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path)

    return img_input, [f1, f2, f3, f4, f5]


def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    # o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width

    return model


def segnet_decoder(f, n_classes, n_up=4):

    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2)))(o)
        o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(128, (3, 3), padding='valid'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    return o


def _segnet(n_classes, encoder,  input_height=256, input_width=256,
            encoder_level=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=4)
    model = get_segmentation_model(img_input, o)

    return model


def segnet(n_classes, input_height=256, input_width=256, encoder_level=3):

    model = _segnet(n_classes, vanilla_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "segnet"
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def vgg_segnet(n_classes, input_height=256, input_width=256, encoder_level=3):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "vgg_segnet"
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#
# def resnet50_segnet(n_classes, input_height=416, input_width=608,
#                     encoder_level=3):
#
#     model = _segnet(n_classes, get_resnet50_encoder, input_height=input_height,
#                     input_width=input_width, encoder_level=encoder_level)
#     model.model_name = "resnet50_segnet"
#     return model


# def mobilenet_segnet(n_classes, input_height=224, input_width=224,
#                      encoder_level=3):
#
#     model = _segnet(n_classes, get_mobilenet_encoder,
#                     input_height=input_height,
#                     input_width=input_width, encoder_level=encoder_level)
#     model.model_name = "mobilenet_segnet"
#     return model


if __name__ == '__main__':
    # m0 = vgg_segnet(4)
    m1 = segnet(4)
    m1.summary()
    # m = mobilenet_segnet( 101 )
    # from keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')