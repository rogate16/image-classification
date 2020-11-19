import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model("model/model.h5")

path = "kue-indonesia/train"
name = pd.DataFrame(os.listdir(path), columns=["name"])
name = name.sort_values(by="name").reset_index(drop=True)
name.name = [re.sub("_"," ",text) for text in name.name]
name.name = [text.title() for text in name.name]

img_path = "example/"
img_path = img_path + os.listdir(img_path)[0]
test_img = load_img(img_path, target_size=(200,200))
test_img = img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)

result = model.predict(test_img)
result = np.where(result==1)[1][0]
result = name.iloc[result].values[0]

plt.text(10,-20,result, color="blue")
plt.imshow(load_img(img_path))
plt.show()