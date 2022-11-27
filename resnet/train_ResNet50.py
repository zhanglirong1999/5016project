# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass

import os
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import gc; gc.enable() 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras.applications
import numpy as np
from keras import layers

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import Activation,BatchNormalization,Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
# import tensorflow.keras.backend as K
# from tensorflow.keras.utils.data_utils import get_file
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import optimizers
from sklearn.preprocessing import OneHotEncoder


print(os.listdir("/home/msbd5016/lirong/data"))

masks = pd.read_csv(os.path.join('/home/msbd5016/lirong/data', 'train_ship_segmentations_v2.csv'))
not_empty = pd.notna(masks.EncodedPixels)
print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')#非空图片中的mask数量
print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')#所有图片中非空图片
# masks.head()

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

ship_dir = '/home/msbd5016/lirong/data'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 
                                                               os.stat(os.path.join(train_image_dir, 
                                                                                    c_img_id)).st_size/1024)

train_0 = unique_img_ids[unique_img_ids['ships']==1].sample(1800)
train_1 = unique_img_ids[unique_img_ids['ships']==2].sample(1800)
train_2 = unique_img_ids[unique_img_ids['ships']==3].sample(1800)  

train_3 = unique_img_ids[unique_img_ids['ships']!=3]
train_3 = train_3[unique_img_ids['ships']!=2]
train_3 = train_3[unique_img_ids['ships']!=1]

unique_img_ids=pd.concat([train_0,train_1,train_2,train_3])

SAMPLES_PER_GROUP = 10000
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

balanced_train_df=balanced_train_df.reset_index(drop = True)#删除原来的索引。
balanced_train_df=balanced_train_df.sample(frac=1.0)

x = np.empty(shape=(15000, 256,256,3),dtype=np.uint8)
y = np.empty(shape=15000,dtype=np.uint8)
for index, image in enumerate(balanced_train_df['ImageId']):
    if index > 14999:
        break
    image_array= Image.open('/home/msbd5016/lirong/data/train_v2/' + image).resize((256,256)).convert('RGB')
    x[index] = image_array
    y[index]=balanced_train_df[balanced_train_df['ImageId']==image]['has_ship'].iloc[0]

y_targets =y.reshape(len(y),-1)
enc = OneHotEncoder()
enc.fit(y_targets)
y = enc.transform(y_targets).toarray()

x_train, x_temp, y_train, y_temp  = train_test_split(x,y,test_size = 0.2,random_state=1,stratify=y)

x_test, x_val, y_test, y_val  = train_test_split(x_temp,y_temp,test_size = 0.1,random_state=1,stratify=y_temp)
# x_train.shape, x_val.shape, y_train.shape, y_val.shape
print(x_train.shape)
print(y_train.shape)

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[256,256,3],classes=2):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='sigmoid', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

#     model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

model_final = ResNet50()

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []

#     def on_epoch_end(self, epoch, logs={}):
# #         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
# #         val_targ = self.validation_data[1]
#         val_targ = np.argmax(self.validation_data[1], axis=1)
#         _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
# #         print(' — val_f1:' ,_val_f1)
#         return

# metrics1 = Metrics()

# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# weight_path="{}_weights.best.hdf5".format('boat_detector')

# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
#                              save_best_only=True, mode='min', save_weights_only = True)

# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=0.0001)
# early = EarlyStopping(monitor="val_loss", 
#                       mode="min", 
#                       patience=3) # probably needs to be more patient, but kaggle time is limited
# callbacks_list = [checkpoint, early, reduceLROnPlat,metrics1]
def fit():
    epochs = 5
    lrate = 0.01
    decay = lrate/epochs
    #adam = optimizers.Adam(learning_rate=lrate,beta_1=0.9, beta_2=0.999, decay=decay)
    sgd = optimizers.SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
    model_final.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])
    loss_history=[model_final.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=40, batch_size=32)]
    
    return loss_history
num=0
# def evaluate():


while True:
    num=num+1
#     prefix='%d'%(num)
    loss_history = fit()
    model_final.save_weights('2_my_model_weights%d.h5'% num)
    model_final.save('2_model.h5')
    if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.3:
        model_final.save('2_best.h5')
        break
    if num==1:
        break
