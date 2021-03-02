from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import cv2
import os
import numpy as np
from keras import utils
import scipy.io as io
from deeplab_weights_xception_65 import Deeplabv3


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class DeepLab(object):

    def __init__(self, img_rows=256, img_cols=256, drop_rate=0.5, n_labels=4, num_class=4, pretrained_weights=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.drop_rate = drop_rate
        self.pretrained_weights = pretrained_weights
        # self.input_shape = (self.img_rows, self.img_cols, 1)
        self.n_labels = n_labels
        self.num_class = num_class

    def one_hot_lab(self, labels):
        lab_nd = np.zeros([self.img_cols, self.img_rows, self.num_class])
        building = np.array(labels[:, :, :] == [255, 0, 0], dtype='uint8')
        vegetation = np.array(labels[:, :, :] == [0, 255, 0], dtype='uint8')
        water = np.array(labels[:, :, :] == [0, 0, 255], dtype='uint8')
        road = np.array(labels[:, :, :] == [255, 255, 0], dtype='uint8')
        bare_land = np.array(labels[:, :, :] == [0, 0, 0], dtype='uint8')
        other = road + bare_land
        # building_1 = building.sum(axis=2)
        # building = building.sum(axis=2) / 3
        lab_nd[:, :, 0] = np.floor(building.sum(axis=2) / 3)
        lab_nd[:, :, 1] = np.floor(vegetation.sum(axis=2) / 3)
        lab_nd[:, :, 2] = np.floor(water.sum(axis=2) / 3)
        lab_nd[:, :, 3] = np.floor(other.sum(axis=2) / 3)
        # lab_nd[:, :, 4] = np.floor(bare_land.sum(axis=2) / 3)

        return lab_nd

    def load_sample(self, sar_dir, lab_dir):
        sar_filename = []
        lab_filename = []
        for path, subdirs, files in os.walk(sar_dir):
            for name in files:
                sar_filename.append(os.path.join(path, name))
        for path, subdirs, files in os.walk(lab_dir):
            for name in files:
                lab_filename.append(os.path.join(path, name))

        return sar_filename, lab_filename

    def create_train_data(self):
        sar_filename, lab_filename = self.load_sample(sar_dir='/emwusr/xianzhengshi/2020_test/20200404/train_256/sar',
                                                      lab_dir='/emwusr/xianzhengshi/2020_test/20200404/train_256/lab')
        train_data = np.ndarray((len(sar_filename), self.img_cols, self.img_rows, 3))
        train_lab = np.ndarray((len(lab_filename), self.img_cols, self.img_rows, 4))
        for num, name in enumerate(sar_filename):
            img = load_img(name)
            img = img_to_array(img)
            train_data[num] = img
            # train_data[num] = (img[:, :, 0]).reshape((self.img_rows, self.img_cols, 1))
        for num1, name1 in enumerate(lab_filename):
            lab = load_img(name1)
            lab = img_to_array(lab)
            train_lab[num1] = self.one_hot_lab(lab)
            # lab_one = lab[:, :, 0] - np.ones((128, 128))
        return train_data, train_lab

    def create_test_data(self):
        sar_filename, lab_filename = self.load_sample(sar_dir='/emwusr/xianzhengshi/2020_test/20200404/test_256/sar',
                                                      lab_dir='/emwusr/xianzhengshi/2020_test/20200404/test_256/lab')
        test_data = np.ndarray((len(sar_filename), self.img_cols, self.img_rows, 3))
        test_lab = np.ndarray((len(lab_filename), self.img_cols, self.img_rows, 4))
        for num, name in enumerate(sar_filename):
            img = load_img(name)
            img = img_to_array(img)
            test_data[num] = img
            # test_data[num] = (img[:, :, 0]).reshape((self.img_rows, self.img_cols, 1))
        for num1, name1 in enumerate(lab_filename):
            lab = load_img(name1)
            lab = img_to_array(lab)
            test_lab[num1] = self.one_hot_lab(lab)
        return test_data, test_lab

    def train(self):
        print('loading data...')
        train_sar, train_lab = self.create_train_data()
        train_sar = train_sar / 127.5 - 1
        test_sar, test_lab = self.create_test_data()
        test_sar = test_sar / 127.5 - 1
        print('loading data done!')

        model = Deeplabv3(input_shape=(256, 256, 3), classes=4)  # 256, 256, 3 -> 256 256 5
        # path = './deeplab_result/weights2/deeplabv3_val_acc.hdf5'
        # model.load_weights(path, by_name=True)
        # model.load_weights('./network/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5', by_name=True)
        # model.summary()
        #
        model.load_weights('./network/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5', by_name=True)
        model.summary()
        # weights_file = './deeplab_result/weights_part/'
        # model.load_weights(weights_file + 'deeplabv3_val_acc.hdf5')
        model_checkpoint = ModelCheckpoint('./2020_test/weights0404/deeplabv3_val_acc.hdf5', monitor='val_acc', verbose=1,
                                           save_best_only=True, save_weights_only=True)
        model_checkpoint1 = ModelCheckpoint('./2020_test/weights0404/deeplabv3_train_loss.hdf5', monitor='loss', verbose=1,
                                            save_best_only=True, save_weights_only=True)
        model_checkpoint2 = ModelCheckpoint('./2020_test/weights0404/deeplabv3_val_loss.hdf5', monitor='val_loss', verbose=1,
                                            save_best_only=True, save_weights_only=True)
        model_earlystop = EarlyStopping(patience=4, monitor='loss')
        print('Fitting model...')
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sar, train_lab, batch_size=10, epochs=150, verbose=1, shuffle=True,
                  validation_data=(test_sar, test_lab),
                  callbacks=[model_checkpoint, model_earlystop, model_checkpoint1, model_checkpoint2])

        # print('train data...')
        # imgs_pre = train_sar[15:30]
        # imgs_train_result = model.predict(imgs_pre, batch_size=1, verbose=1)
        # num = imgs_train_result.shape
        # train_pre = np.zeros((num[0], num[1], num[2]))
        # for i in range(num[0]):
        #     pro = imgs_train_result[i, :, :, :]
        #     lab_num = np.argmax(pro, axis=2)
        #     train_pre[i, :, :] = lab_num
        #
        # train_mask = train_lab[15:30]
        # num = train_mask.shape
        # train_mask2 = np.zeros((num[0], num[1], num[2]))
        # for i in range(num[0]):
        #     pro = train_mask[i, :, :, :]
        #     lab_num = np.argmax(pro, axis=2)
        #     train_mask2[i, :, :] = lab_num
        #
        # test_img = test_sar[30:45]
        # imgs_test_result = model.predict(test_img, batch_size=1, verbose=1)
        # num = imgs_test_result.shape
        # test_pre = np.zeros((num[0], num[1], num[2]))
        # for i in range(num[0]):
        #     pro = imgs_test_result[i, :, :, :]
        #     lab_num = np.argmax(pro, axis=2)
        #     test_pre[i, :, :] = lab_num
        # # np.save(weights_file + 'test_pre.npy', imgs_test_result)
        #
        # test_mask = test_lab[30:45]
        # num = test_mask.shape
        # test_mask2 = np.zeros((num[0], num[1], num[2]))
        # for i in range(num[0]):
        #     pro = test_mask[i, :, :, :]
        #     lab_num = np.argmax(pro, axis=2)
        #     test_mask2[i, :, :] = lab_num
        #
        # io.savemat(weights_file + 'train_sar', {'name': ((train_sar[15:30]) + 1) * 127.5})
        # io.savemat(weights_file + 'test_sar', {'name': ((test_sar[30:45]) + 1) * 127.5})
        # io.savemat(weights_file + 'train_predict', {'name': train_pre})
        # io.savemat(weights_file + 'test_predict', {'name': test_pre})
        # io.savemat(weights_file + 'train_lab', {'name': train_mask2})
        # io.savemat(weights_file + 'test_lab', {'name': test_mask2})

        # print('train data...')
        # imgs_pre = imgs_train[15:30]
        # imgs_mask_train_result = model.predict(imgs_pre, batch_size=1, verbose=1)
        # np.save('imgs_mask_train_result', imgs_mask_train_result)
        #
        # print('predict train data...')
        # imgs_mask_train_predict = model.predict(imgs_train_predict, batch_size=1, verbose=1)
        # np.save('imgs_mask_train_predict', imgs_mask_train_predict)
        #
        # print('predict test data...')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    myunet = DeepLab()
    myunet.train()


