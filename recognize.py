import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
model = keras.models.load_model('model1.h5') # Загрузка сохраненной модели
img = cv2.imread("input.jpg") # Загрузка изображения
# Обработка изображения
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, im_th = cv2.threshold(blur, 180, 400, cv2.THRESH_BINARY_INV)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
# Выборка контуров
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
path_to_data = 'C:/Users/779/Desktop/6sem/IskIN/пп/lb5/Dataset/data/training_data'
# Путь к папке с изображениями
le = LabelEncoder()
def create_labels(path):
 """
 Creates a list of labels from the directories in the given path.
 Parameters:
 path (str): The path to the directory containing the directories to be processed.
 Returns:
 None
 This function iterates over the directories in the given path and appends each
directory name to the `labels` list.
 It then converts the `labels` list into a NumPy array and fits the labels using the `le`
object.
 Note:
 The `le` object is assumed to be defined and initialized before calling this
function.
 """
 labels = []
 dir_list = os.listdir(path)
 for i in dir_list:
 labels.append(i) # Перебор папок по пути
 y = np.array(labels)
 le.fit_transform(y) # Сохранение лейблов
def recognize(img):
  """
 Recognizes and predicts labels for objects in an image using a pre-trained model.
 """
 for x, y, w, h in rects:
 if y >= 3:
 y -= 3
 else:
 y = 0
 if x >= 3:
 x -= 3
 else:
 x = 0
 w += 3
 h += 3
 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
 sliced = img[y:y + h, x:x + w] # Вырезание области изображения
 if sliced.shape[0] == 0 or sliced.shape[1] == 0: # Проверка на пустую область
 continue
 sliced = cv2.resize(sliced, (64, 64)) # Изменение размера
 sliced = np.array(sliced, dtype=np.float32) # Преобразование в массив
 sliced = sliced / 255 # Нормализация
 sliced = np.expand_dims(sliced, axis=-1) # Добавление оси
 sliced = np.expand_dims(sliced, axis=0) # Добавление оси
 prediction = model.predict(sliced) # Предсказание
 prediction = np.argmax(prediction, axis=1) # Выбор лейбла
 predicted_labels = le.inverse_transform(prediction) # Перевод лейбла в
название
 cv2.putText(img, str(predicted_labels[0]), (x + w, y + int(h / 2)), 
             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
 cv2.LINE_AA) # Вывод
 img=cv2.resize(img, (800, 800))
 cv2.imshow("Output", img) # Вывод изображения
 cv2.waitKey(0)
 cv2.destroyAllWindows()
create_labels(path_to_data)
recognize(img)
