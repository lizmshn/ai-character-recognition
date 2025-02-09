from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
images = []
labels = []
path_to_data = 'C:/Users/779/Desktop/6sem/IskIN/пп/lb5/Dataset/data/training_data' # Путь к папке с изображениями
def create_dataset(path):
 """
 Creates a dataset from the images in the given directory.
 Parameters:
 path (str): The path to the directory containing the images.
 Returns:
 tuple: A tuple containing the features (X) and labels (y) of the dataset.
 This function iterates over the directories and files in the given path, reads each
image, resizes it to 64x64,
 converts it to a numpy array, normalizes it, and appends it to the `images` list. It
also appends the corresponding
 directory name to the `labels` list. After iterating over all the images, it converts
the `images` and `labels`
 lists to numpy arrays and applies label encoding to the `labels` array. Finally, it
returns the features (X) and
 labels (y) as a tuple.
 """
 dir_list = os.listdir(path)
 for i in dir_list:
   dirs = os.path.join(path, i)
   file_list = os.listdir(dirs)
   for j in file_list:
     files = os.path.join(dirs, j)
     img = cv2.imread(files)
     img = cv2.resize(img, (64, 64))
     img = np.array(img, dtype=np.float32)
     img = img / 255
     images.append(img)
     labels.append(i)
 X = np.array(images)
 y = np.array(labels)
 y = le.fit_transform(y)
 return X, y

def create_model():
 """
 Creates a deep learning model for image recognition using Convolutional Neural
Networks (CNN).
 The model architecture consists of several convolutional layers with max pooling,
followed by dense layers.
 Returns:
 Sequential: A Sequential model representing the CNN architecture.
 """
 model = models.Sequential()
 model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
input_shape=(64, 64, 3)))
 model.add(layers.MaxPooling2D())
 model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
 model.add(layers.MaxPooling2D())
 model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
 model.add(layers.MaxPooling2D())
 model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
 model.add(layers.Flatten())
 model.add(layers.Dense(units=256, activation='relu'))
 model.add(layers.Dense(units=64, activation='relu'))
 model.add(layers.Dense(units=36, activation='softmax'))
 return model
X, y = create_dataset(path_to_data)
X_sh, y_sh = shuffle(X, y, random_state=42)
12
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy']) # Настройка модели
history = model.fit(X_sh, y_sh, validation_split=0.2, batch_size=16, epochs=10) #
Обучение
model.save('model2.h5') # Сохранение модели
