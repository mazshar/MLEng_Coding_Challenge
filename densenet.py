import tensorflow as tf
import numpy as np
import pickle
from urllib import request

# ImageNet labels taken from github user yrevar
IMAGENET_LABELS_PATH = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'

WEIGHT_PATH = 'densenet121_weights_tf_dim_ordering_tf_kernels.h5'


class DenseNet:

    def __init__(self):
        self.labels = pickle.load(request.urlopen(IMAGENET_LABELS_PATH))
        self.model = tf.keras.applications.DenseNet121(include_top=True, weights=WEIGHT_PATH)

    def process_image(self, img_path: str) -> np.ndarray:
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(image)

        # Expand to (1,244,244,3)
        x = np.expand_dims(x, axis=0)
        
        return x

    def make_prediction(self, x: np.ndarray) -> str:
        preprocessed = tf.keras.applications.densenet.preprocess_input(x)

        # Returns probabilities. Take max to get correct class.
        prediction = np.argmax(self.model.predict(preprocessed))

        # Return the label with the given class index.
        return self.labels[prediction]
