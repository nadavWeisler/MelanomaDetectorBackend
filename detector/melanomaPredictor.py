import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
from keras.layers import Dropout


class MelanomaPredictor:
    RESIZE_FACTOR = 128

    def __init__(self, modelPath):
        """
        Initialize an object from this class.
        """

        # Thaw the frozen models
        self.model = tf.keras.models.load_model(modelPath, compile=False, custom_objects={'FixedDropout': Dropout})
        self.model.summary()

    def predict(self, imagesPath):
        """
        predict beauty marks - Melanoma or not
        :param imagesPath: a path to a directory of squared proportional imagess
        """
        X = self._prepare_data(imagesPath)
        return self.model.predict(X)

    def _prepare_data(self, imagesPath) -> np.ndarray:
        """
        Prepare the test sick for evaluation.
        :param imagesPath: a path to a directory of squared proportional imagess
        :return:
        """

        X = []

        readImg = lambda imname: np.asarray(
            MelanomaPredictor.crop_max_square(Image.open(imname).convert("RGB")))

        for imageName in os.listdir(imagesPath):
            path = os.path.join(imagesPath, imageName)
            img = cv2.resize(readImg(path), (self.RESIZE_FACTOR, self.RESIZE_FACTOR))
            X.append(np.asarray(img, dtype="float32") / 255.)
        return np.asarray(X)

    @staticmethod
    def crop_max_square(pil_img):
        return MelanomaPredictor.crop_center(pil_img, min(pil_img.size), min(pil_img.size))

    @staticmethod
    def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                             (img_height - crop_height) // 2,
                             (img_width + crop_width) // 2,
                             (img_height + crop_height) // 2))
