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
from keras import backend as K
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

    def load_sample(self, sar_dir):
        sar_filename = []
        path_list = os.listdir(sar_dir)
        path_list.sort()
        for filename in path_list:
            new_path = os.path.join(sar_dir, filename)
            print(new_path)
            sar_filename.append(new_path)
        # for path, subdirs, files in os.walk(sar_dir):
        #     for name in files:
        #         path_new = os.path.join(path, name)
        #         print(path_new)
        #         sar_filename.append(path_new)

        return sar_filename

    def create_test_data(self):
        sar_filename = self.load_sample(sar_dir='/emwusr/xianzhengshi/2020_test/20200404/tianjin_256')
        test_data = np.ndarray((len(sar_filename), self.img_cols, self.img_rows, 3))
        for num, name in enumerate(sar_filename):
            img = load_img(name)
            img = img_to_array(img)
            test_data[num] = img
        return test_data

    def train(self):
        test_sar = self.create_test_data()
        test_sar = test_sar / 127.5 - 1
        print('loading data done!')

        model = Deeplabv3(input_shape=(256, 256, 3), classes=self.num_class)  # 256, 256, 3 -> 256 256 5
        path = '/emwusr/xianzhengshi/2020_test/weights0404/deeplabv3_val_acc.hdf5'
        model.load_weights(path, by_name=True)
        # model.load_weights('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5', by_name=True)
        # model.summary()

        # model.load_weights('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5', by_name=True)
        # model.summary()
        weights_file = '/emwusr/xianzhengshi/2020_test/20200418/sar_seg/test_acc_pre/'
        if not os.path.exists(weights_file):
            os.mkdir(weights_file)

        imgs_test_result = model.predict(test_sar, batch_size=1, verbose=1)

        K.clear_session()

        num = imgs_test_result.shape
        test_pre = np.zeros((num[0], num[1], num[2]))
        for i in range(num[0]):
            pro = imgs_test_result[i, :, :, :]
            lab_num = np.argmax(pro, axis=2)
            test_pre[i, :, :] = lab_num

        # io.savemat(weights_file + 'test_sar_jiujiang', {'name': (test_sar + 1) * 127.5})
        io.savemat(weights_file + 'test_predict_val_acc_lab_tianjin', {'name': test_pre})

        # index_list = [308*4, 144*4, 144*4, 76*4, 500*4, 680*4, 540*4, 48*4]
        # start = 0
        # end = 0
        # for i, index in enumerate(index_list):
        #     end += index
        #     io.savemat(weights_file + 'test_sar_{}'.format(i + 1), {'name': (test_sar[start:end] + 1) * 127.5})
        #     io.savemat(weights_file + 'test_predict_{}'.format(i + 1), {'name': test_pre[start:end]})
        #     start = end


if __name__ == '__main__':
    myunet = DeepLab()
    myunet.train()


