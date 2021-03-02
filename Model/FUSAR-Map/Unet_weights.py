from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping


def get_unet(input_shape=(256, 256, 3), drop_rate=0.3, classes=4):
    # encode
    inputs = Input(input_shape)

    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    batch_normal1_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv1_1)
    relu1_1 = Activation('relu')(batch_normal1_1)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu1_1)
    batch_normal1_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv1_2)
    relu1_2 = Activation('relu')(batch_normal1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu1_2)
    print("pool1 shape:", pool1.shape)

    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    batch_normal2_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv2_1)
    relu2_1 = Activation('relu')(batch_normal2_1)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu2_1)
    batch_normal2_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv2_2)
    relu2_2 = Activation('relu')(batch_normal2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(relu2_2)
    print("pool2 shape:", pool2.shape)

    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    batch_normal3_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv3_1)
    relu3_1 = Activation('relu')(batch_normal3_1)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu3_1)
    batch_normal3_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv3_2)
    relu3_2 = Activation('relu')(batch_normal3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(relu3_2)
    print("pool3 shape:", pool3.shape)

    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    batch_normal4_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv4_1)
    relu4_1 = Activation('relu')(batch_normal4_1)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu4_1)
    batch_normal4_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv4_2)
    relu4_2 = Activation('relu')(batch_normal4_2)
    drop4 = Dropout(drop_rate)(relu4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(drop4)
    print("pool4 shape:", pool4.shape)

    conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    batch_normal5_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv5_1)
    relu5_1 = Activation('relu')(batch_normal5_1)
    conv5_2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu5_1)
    batch_normal5_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv5_2)
    relu5_2 = Activation('relu')(batch_normal5_2)
    drop5 = Dropout(drop_rate)(relu5_2)

    # decode
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    # up6 = Cropping2D(cropping=((1, 0), (0, 0)))(up6)  # ?
    print("up6 shape:", up6.shape)
    # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    merge6 = concatenate([drop4, up6])  # axis=-1
    conv6_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    batch_normal6_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv6_1)
    relu6_1 = Activation('relu')(batch_normal6_1)
    conv6_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu6_1)
    batch_normal6_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv6_2)
    relu6_2 = Activation('relu')(batch_normal6_2)
    print('relu6_2 shape:', relu6_2.shape)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(relu6_2))
    # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    merge7 = concatenate([relu3_2, up7])
    conv7_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    batch_normal7_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv7_1)
    relu7_1 = Activation('relu')(batch_normal7_1)
    conv7_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu7_1)
    batch_normal7_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv7_2)
    relu7_2 = Activation('relu')(batch_normal7_2)
    print("reu7_2 shape:", relu7_2.shape)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(relu7_2))
    # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    merge8 = concatenate([relu2_2, up8])
    conv8_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    batch_normal8_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv8_1)
    relu8_1 = Activation('relu')(batch_normal8_1)
    conv8_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu8_1)
    batch_normal8_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv8_2)
    relu8_2 = Activation('relu')(batch_normal8_2)
    print("relu8_2 shape:", relu8_2.shape)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(relu8_2))
    # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    merge9 = concatenate([relu1_2, up9])
    conv9_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    batch_normal9_1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv9_1)
    relu9_1 = Activation('relu')(batch_normal9_1)
    conv9_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(relu9_1)
    batch_normal9_2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-12,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(conv9_2)
    relu9_2 = Activation('relu')(batch_normal9_2)
    conv9_3 = Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(relu9_2)
    output = Activation('softmax')(conv9_3)

    model = Model(input=inputs, output=output)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = get_unet()

