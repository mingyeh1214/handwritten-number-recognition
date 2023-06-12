from flask import Flask, render_template, request
import base64
import re
import cv2
import tensorflow as tf
import numpy as np
import json
from train import image_grey2black
import pandas as pd


def init_models():
    model_NN = tf.keras.models.load_model("./models/NN.h5")
    model_CNN = tf.keras.models.load_model("./models/CNN.h5")
    model_NN2 = tf.keras.models.load_model("./models/NN2.h5")
    model_CNN2 = tf.keras.models.load_model("./models/CNN2.h5")
    return model_NN, model_CNN, model_NN2, model_CNN2


global model_NN, model_CNN, model_NN2, model_CNN2
model_NN, model_CNN, model_NN2, model_CNN2 = init_models()


def parseImg(imgData):
    img_df = pd.read_csv("./static/img_idx.csv",
                         index_col=None, encoding='UTF-8', sep=',')
    img_idx = np.max(img_df["index"]) + 1
    img_df = pd.concat([img_df, pd.DataFrame(
        {'index': [img_idx]})], ignore_index=True)
    img_df.to_csv("./static/img_idx.csv", index=False)
    canvas_img_path = f'./static/images/img/canvas_img_{img_idx}.png'
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open(canvas_img_path, 'wb') as output:
        output.write(base64.decodebytes(imgstr))
    image = cv2.imread(canvas_img_path)
    image = 255 - cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    image = image_grey2black(image, 255 / 8)
    contours, _ = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        z = max(w, h)
        cv2.rectangle(image, (x, y), (x+z, y+z),
                      color=(0, 255, 0), thickness=2)
        if (w < h):
            x = max(int(x - (h - w) / 2), 0)
        else:
            y = max(int(y - (w - h) / 2), 0)
        digit = image[y:y+z, x:x+z]
        padded_digit = np.pad(digit, ((10, 10), (10, 10)),
                              "constant", constant_values=0)
        preprocessed_digits.append(padded_digit)
    inp = np.array(preprocessed_digits)
    image = cv2.resize(inp[0], (28, 28))
    image = image_grey2black(image, 255 / 8)
    process_img_path = f"./static/images/img/process_img_{img_idx}.png"
    cv2.imwrite(process_img_path, image)
    img = image.reshape(1, 28, 28, 1) / 255.0
    return canvas_img_path, process_img_path, img


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    canvas_img_path, process_img_path, img = parseImg(request.get_data())

    NN_pred = model_NN.predict(img)
    NN_result = str(np.argmax(NN_pred, axis=1)[0])
    value = np.round(np.ndarray.tolist(NN_pred[0]), 4)
    key = range(0, 10)
    NN_pred_dict = dict(sorted(dict(zip(key, value)).items(),
                        key=lambda item: item[1], reverse=True))

    CNN_pred = model_CNN.predict(img)
    CNN_result = str(np.argmax(CNN_pred, axis=1)[0])
    value = np.round(np.ndarray.tolist(CNN_pred[0]), 4)
    key = range(0, 10)
    CNN_pred_dict = dict(sorted(dict(zip(key, value)).items(),
                         key=lambda item: item[1], reverse=True))

    NN2_pred = model_NN2.predict(img)
    NN2_result = str(np.argmax(NN2_pred, axis=1)[0])
    value = np.round(np.ndarray.tolist(NN2_pred[0]), 4)
    key = range(0, 10)
    NN2_pred_dict = dict(sorted(dict(zip(key, value)).items(),
                         key=lambda item: item[1], reverse=True))

    CNN2_pred = model_CNN2.predict(img)
    CNN2_result = str(np.argmax(CNN2_pred, axis=1)[0])
    value = np.round(np.ndarray.tolist(CNN2_pred[0]), 4)
    key = range(0, 10)
    CNN2_pred_dict = dict(
        sorted(dict(zip(key, value)).items(), key=lambda item: item[1], reverse=True))

    return {
        "NN_result": NN_result,
        "NN2_result": NN2_result,
        "CNN_result": CNN_result,
        "CNN2_result": CNN2_result,
        "NN_pred": json.dumps(NN_pred_dict),
        "CNN_pred": json.dumps(CNN_pred_dict),
        "NN2_pred": json.dumps(NN2_pred_dict),
        "CNN2_pred": json.dumps(CNN2_pred_dict),
        "canvas_img_path": canvas_img_path,
        "process_img_path": process_img_path
    }


if __name__ == "__main__":
    app.run(debug=True)
