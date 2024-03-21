import numpy as np
import tensorflow as tf
import os
from PIL import Image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = tf.keras.models.load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        img = Image.open(imagename)
        img = img.resize((244, 244), resample=Image.LANCZOS)
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        test_image = np.expand_dims(img, axis=0)
        # test_image = tf.keras.preprocessing.image.load_img(imagename)
        # test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{"image": prediction}]
        else:
            prediction = 'Normal'
            return [{"image": prediction}]