from os.path import join
import numpy as np
import sys
from IPython.display import Image, display
from utils.decode_predictions import decode_predictions
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = 224

class predictor:
    def __init__(self):
        hot_dog_image_dir = './train/hot_dog'
        hot_dog_paths = [join(hot_dog_image_dir, filename) for filename in 
                         ['1000288.jpg',
                          '127117.jpg']]
        
        not_hot_dog_image_dir = './train/not_hot_dog'
        not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                                    ['823536.jpg',
                                     'IMG_20180526_194347.jpg',
                                     '99890.jpg']]
        
        self.image_paths = hot_dog_paths + not_hot_dog_paths
        self.model = \
        ResNet50(weights='./resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    def read_and_prep_images(self, img_paths, img_height=IMAGE_SIZE,
                             img_width=IMAGE_SIZE):
        imgs = [load_img(img_path, target_size=(img_height, img_width)) for
                img_path in self.image_paths]
        print(imgs)
        img_array = np.array([img_to_array(img) for img in imgs])
        return preprocess_input(img_array)

    def predict(self, img_array):
        predictions = self.model.predict(img_array) 
        return predictions




if __name__ == '__main__' :
    predict = predictor()
    img_array = predict.read_and_prep_images("")
    predictions = predict.predict(img_array)
    most_likely_labels = decode_predictions(predictions, top=3,
                                            class_list_path='imagenet_class_index.json')

    for i , img_path in enumerate(predict.image_paths):
        print(most_likely_labels[i])
