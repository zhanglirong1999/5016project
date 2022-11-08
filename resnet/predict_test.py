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
from PIL import Image

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import Activation,BatchNormalization,Flatten
from tensorflow.keras.models import Model

from keras.models import load_model
model = load_model('my_best.h5')

ship_dir = '/home/msbd5016/lirong/data'
# train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
masks = pd.read_csv(os.path.join('/home/msbd5016/lirong/data', 'sample_submission_v2.csv'))

x = np.empty(shape=(15606, 256,256,3),dtype=np.uint8)
y = np.empty(shape=15606,dtype=np.uint8)
images = []
for index, image in enumerate(masks['ImageId']):
    if index > 15605:
        break
    image_array= Image.open('/home/msbd5016/lirong/data/test_v2/' + image).resize((256,256)).convert('RGB')
    images.append(image)
    x[index] = image_array

print(x.shape)
predict_test = model.predict(x)
preds = np.argmax(predict_test, axis=1)
print(preds)
np_array  = [preds,images]
df = pd.DataFrame(np_array).transpose()
df.columns = ['preds', 'images']
name = '/home/msbd5016/lirong/data/result.csv'
df.to_csv(name)