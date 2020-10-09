import io
import random
import string

from PIL import Image
from flask import Flask, request, Request
from flask_cors import CORS, cross_origin
import os
import base64

from detector.melanomaPredictor import MelanomaPredictor

request: Request

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

if not os.path.exists("image_store"):
    os.mkdir("image_store")

melanomaModel = MelanomaPredictor("detector/model20201001-0625.h5")
melanomaModel: MelanomaPredictor


@app.route('/', methods=['POST'])
@cross_origin()
def _add_pic():
    if request.form and 'image' in request.form:
        imageHeaderData = request.form['image'].split(',')
        if len(imageHeaderData) > 1:
            imageDataBase64 = imageHeaderData[1]
            imageDataBytesEncoded = bytes(imageDataBase64, 'ascii')
            imageDataBytes = base64.decodebytes(imageDataBytesEncoded)

            img = Image.open(io.BytesIO(imageDataBytes)).convert("RGB")
            img: Image.Image
            img = MelanomaPredictor.crop_max_square(img)
            img = img.resize((MelanomaPredictor.RESIZE_FACTOR, MelanomaPredictor.RESIZE_FACTOR))

            imageFolderPath = "image_store/"
            while os.path.exists(imageFolderPath):
                imageFolderName = _get_random_string(15)
                imageFolderPath = f"image_store/{imageFolderName}"

            os.mkdir(imageFolderPath)
            img.save(f"{imageFolderPath}/image.png")

            prediction = melanomaModel.predict(f"{imageFolderPath}")[0][1] * 100
            return "{num:.2f}".format(num=prediction), 200

    res = "must contain form-data with 'image' variable inside"
    return res, 400


def _get_random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))


def getApp():
    return app
