"""
Author: xzshi20
Date: 2020.07.03
Aims: Network training with tensorflow1.15.3
"""
import tensorflow as tf
import os
import random
import numpy as np
import scipy.io as io
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from deeplab_weights_xception_65 import Deeplabv3

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

NUMBER = 7
HEIGHT = 256
WIDTH = 256
EPS = 1e-6


def create_filenames(image_dir_path, label_dir_path):
    file_names = []
    train_img_filenames = []
    train_lab_filenames = []
    for path, subdirs, files in os.walk(image_dir_path):
        for name in files:
            file_names.append(name)
    random.seed(66)
    random.shuffle(file_names)
    for i in range(len(file_names)):
        # char_name = file_names[i].split('.')
        train_img_filenames.append(os.path.join(image_dir_path, file_names[i]))
        # train_lab_filenames.append(os.path.join(label_dir_path, char_name[0] + '_gt.png'))
        train_lab_filenames.append(os.path.join(label_dir_path, file_names[i]))
    return train_img_filenames, train_lab_filenames


def load_and_preprocess_image(train_img, height=HEIGHT, width=WIDTH):
    tr_img = tf.io.read_file(train_img)
    tr_image = tf.image.decode_png(tr_img, channels=3, dtype=tf.uint8)  # [0, 255]
    tr_image = tf.reshape(tr_image, (height, width, 3))
    # tr_image = tf.tile(tr_image, [1, 1, 3])
    # tr_image = tf.image.resize(tr_image, (256, 256))
    # tr_image = tf.cast(tr_image, tf.float32) / 127.5 - 1.0   # [-1, 1]
    tr_image = tf.cast(tr_image, tf.float32) / 127.5 - 1.0

    return tr_image


def load_and_preprocess_label(train_lab, height=HEIGHT, width=WIDTH, num=NUMBER):
    tr_lab = tf.io.read_file(train_lab)
    tr_lab = tf.image.decode_png(tr_lab, channels=0, dtype=tf.uint8)  # [0 1 2 3 4 5 6]
    tr_lab -= 1
    one_hot_lab = tf.one_hot(tr_lab, depth=num, on_value=1, off_value=0, axis=-1)  # [0, 0, 0, 0, 0, 1]
    one_hot_lab = tf.squeeze(one_hot_lab)
    one_hot_lab = tf.reshape(one_hot_lab, (height, width, num))
    # one_hot_lab = tf.image.resize(one_hot_lab, (256, 256))

    return one_hot_lab


# def image_and_label_process(train_img, train_lab):
#     tr_img = tf.io.read_file(train_img)
#     tr_image = tf.image.decode_png(tr_img, channels=3, dtype=tf.uint8)  # [0, 255]
#     tr_image = tf.reshape(tr_image, (HEIGHT, WIDTH, 3))
#
#     tr_lab = tf.io.read_file(train_lab)
#     tr_lab = tf.image.decode_png(tr_lab, channels=0, dtype=tf.uint8)  # [0 1 2 3 4 5 6]
#     tr_lab -= 1
#     one_hot_lab = tf.one_hot(tr_lab, depth=NUMBER, on_value=1, off_value=0, axis=-1)  # [0, 0, 0, 0, 0, 1]
#     one_hot_lab = tf.squeeze(one_hot_lab)
#     one_hot_lab = tf.reshape(one_hot_lab, (HEIGHT, WIDTH, NUMBER))
#
#     scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.]
#     int_random = random.randint(1, 6)
#     # print(int_random, '********************')
#     tr_image = tf.image.resize_images(tr_image,
#                                       [int(HEIGHT * scale[(int_random - 1)]), int(WIDTH * scale[(int_random - 1)])])
#     one_hot_lab = tf.image.resize_images(one_hot_lab,
#                                          [int(HEIGHT * scale[(int_random - 1)]), int(WIDTH * scale[(int_random - 1)])],
#                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#
#     tr_image = tf.cast(tr_image, tf.float32) / 127.5 - 1.0
#     # one_hot_lab = tf.one_hot(tr_lab, depth=NUMBER, on_value=1, off_value=0, axis=-1)  # [0, 0, 0, 0, 0, 1]
#     # one_hot_lab = tf.squeeze(one_hot_lab)
#     # one_hot_lab = tf.reshape(one_hot_lab, (int(HEIGHT*scale[(int_random - 1)]), int(WIDTH*scale[(int_random - 1)]), NUMBER))
#
#     return tr_image, one_hot_lab


def create_dataset(tr_img_path, tr_lab_path):
    tr_img_ds = tf.data.Dataset.from_tensor_slices(tr_img_path)
    tr_lab_ds = tf.data.Dataset.from_tensor_slices(tr_lab_path)
    image_ds = tr_img_ds.map(load_and_preprocess_image)
    label_ds = tr_lab_ds.map(load_and_preprocess_label)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds


# def create_dataset_ds(tr_img_path, tr_lab_path):
#     tr_img_ds = tf.data.Dataset.from_tensor_slices(tr_img_path)
#     tr_lab_ds = tf.data.Dataset.from_tensor_slices(tr_lab_path)
#     image_label_ds = tf.data.Dataset.zip((tr_img_ds, tr_lab_ds))
#
#     return image_label_ds


def weighted_categorical_crossentropy(y_true, y_pred):
    # weights_all = tf.reduce_sum(y_true)
    # weights_1 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 0]), weights_all))
    # weights_2 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 1]), weights_all))
    # weights_3 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 2]), weights_all))
    # weights_4 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 3]), weights_all))
    # weights_5 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 4]), weights_all))
    # weights_6 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 5]), weights_all))
    # weights_7 = tf.subtract(tf.constant(1.0), tf.divide(tf.reduce_sum(y_true[..., 6]), weights_all))
    # weights = [weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7]
    weights = [100 / 5.5 + EPS, 100 / 2.8 + EPS, 100 / 22 + EPS, 100 / 37.6 + EPS, 100 / 20.4 + EPS, 100 / 9.7 + EPS, 100 / 2.1 + EPS]
    weights = tf.reshape(weights, [1, 1, 1, 7])
    y_true_shape = tf.shape(y_true)
    weights = tf.tile(weights, [y_true_shape[0], y_true_shape[1], y_true_shape[2], 1])
    y_true = tf.multiply(y_true, weights)
    return K.categorical_crossentropy(y_true, y_pred)
#
#
# def categorical_focal_loss(gamma=2., alpha=.25):
#     def categorical_focal_loss_fixed(y_true, y_pred):
#         # Scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # Clip the prediction value to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         # Calculate Cross Entropy
#         cross_entropy = -y_true * K.log(y_pred)
#         # Calculate Focal Loss
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
#         # Compute mean loss in mini_batch
#         return K.mean(K.sum(loss, axis=-1))
#
#     return categorical_focal_loss_fixed


# def fwiou(pred, true):
#     y_true = tf.cast(tf.equal())

def main():
    image_tr_256 = '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/MS_to_half/data_resize_visual_t0_train'
    label_tr_256 = '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/MS_to_half/mask_resize_aug_256_train'
    image_val_256 = '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/MS_to_half/data_m3_256'
    label_val_256 = '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/MS_to_half/mask_256'

    tr_img_file, tr_lab_file = create_filenames(image_tr_256, label_tr_256)
    train_ds_256 = create_dataset(tr_img_file, tr_lab_file)

    # print('training datasets perpared...')

    epochs = 30
    batch_size = 10
    buffer_size = int(len(tr_img_file))
    steps_per_epoch = buffer_size // batch_size
    augmented_train_batches = (train_ds_256.shuffle(500).repeat().batch(batch_size).prefetch(30))

    val_img_file, val_lab_file = create_filenames(image_val_256, label_val_256)
    valid_ds_256 = create_dataset(val_img_file, val_lab_file)
    total_valid = len(val_img_file)
    valid_steps = total_valid // batch_size
    valid_ds = (valid_ds_256.batch(batch_size).prefetch(30))

    # model = Deeplabv3(input_shape=(None, None, 3), classes=NUMBER, OS=16, drop_pro=0.5, middle_repeat=16)  # drop_pro
    model = Deeplabv3(input_shape=(None, None, 3), classes=7, OS=16, middle_repeat=8)
    model.load_weights(
        '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/V4/weights_IM41_t0_all_MS/deeplabv3plus_27-0.98407--0.71516.hdf5',
        by_name=True)
    # model.load_weights(
    #     '/emwuser/xianzhengshi/2020_compete/xzshi_code/uesful_weights/deeplabv3plus_xception_41_imagenet.h5',
    #     by_name=True)
    # model.load_weights('/emwuser/xianzhengshi/2020_compete/xzshi_code/uesful_weights/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5', by_name=True)
    # model.load_weights(
    #     '/emwuser/xianzhengshi/2020_compete/xzshi_code/uesful_weights/deeplabv3plus_xception_65_imagenet_coco_decoder.h5',
    #     by_name=True)

    init = 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['acc'])  # tf.keras.metrics.MeanIoU(num_classes=6)
    # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    # tf.keras.optimizers.SGD(learning_rate=init, momentum=0.9, nesterov=False)
    # loss=weighted_categorical_crossentropy()
    # tf.keras.metrics.CategoricalAccuracy()
    # model.summary()

    filepath = '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/V_hold/MS_to_half/weights_IM41_t0_all_MS_pre27'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = filepath + '/deeplabv3plus_{epoch:02d}-{acc:.5f}--{val_acc:.5f}.hdf5'
    model_checkpoint = ModelCheckpoint(filepath, monitor='acc',
                                       verbose=1,
                                       save_best_only=False, save_weights_only=True)  # , save_weights_only=True
    # model_checkpoint1 = ModelCheckpoint(
    #     '/emwuser/xianzhengshi/2020_compete/GF-3-polSAR/valid8/train_loss_best_COCO.hdf5',
    #     monitor='loss', verbose=1,
    #     save_best_only=True, save_weights_only=True)
    # model_checkpoint2 = ModelCheckpoint('/emwusr/xianzhengshi/2020_compete/GF-3-polSAR/trainval7/X41_IM/deeplabv3_val_loss_p3_41_IM.hdf5',
    #                                     monitor='val_loss', verbose=1,
    #                                     save_best_only=True, save_weights_only=True)
    model_earlystop = EarlyStopping(patience=2, monitor='loss')

    # model_earlystop1 = EarlyStopping(patience=6, monitor='val_acc')

    def schedule(epoch):
        return init * pow(0.92, epoch)

    # def schedule(epoch):
    #     init = 1e-2
    #     decay = 0.95
    #     return init / (1 + decay * epoch)

    # poly lr
    # def schedule(epoch):
    #     init = 7e-3
    #     decay = 0.95
    #     return init * pow((1 - epoch / epochs), decay)

    learningrate_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)

    history = model.fit(augmented_train_batches, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, verbose=1, validation_data=valid_ds, validation_steps=valid_steps // 2,
                        callbacks=[model_earlystop, model_checkpoint,
                                   learningrate_schedule])  # , learningrate_schedule

    # io.savemat('./2020_compete/20200812-deeplab-polSAR/history', {'name': history})


if __name__ == '__main__':
    main()
