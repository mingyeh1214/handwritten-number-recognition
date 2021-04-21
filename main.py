from flask import Flask, render_template, request
import base64
import re
import cv2
import tensorflow as tf
import numpy as np
import json
from train import image_grey2black

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Predict function
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImg(request.get_data())

    img = cv2.imread("./static/images/process_img.png", cv2.IMREAD_GRAYSCALE)
    img = img.reshape(1,28,28,1) / 255.0

    model_NN = tf.keras.models.load_model("./models/NN.h5")
    NN_pred = model_NN.predict(img)
    NN_result = str(np.argmax(NN_pred, axis = 1)[0])
    value = np.round(np.ndarray.tolist(NN_pred[0]), 4)
    key = range(0,10)
    NN_pred_dict = dict(sorted(dict(zip(key, value)).items(), key=lambda item: item[1], reverse = True))
    
    model_CNN = tf.keras.models.load_model("./models/CNN.h5")
    CNN_pred = model_CNN.predict(img)
    CNN_result = str(np.argmax(CNN_pred, axis = 1)[0])
    value = np.round(np.ndarray.tolist(CNN_pred[0]), 4)
    key = range(0,10)
    CNN_pred_dict = dict(sorted(dict(zip(key, value)).items(), key=lambda item: item[1], reverse = True))
    
    model_NN2 = tf.keras.models.load_model("./models/NN2.h5")
    NN2_pred = model_NN2.predict(img)
    NN2_result = str(np.argmax(NN2_pred, axis = 1)[0])
    value = np.round(np.ndarray.tolist(NN2_pred[0]), 4)
    key = range(0,10)
    NN2_pred_dict = dict(sorted(dict(zip(key, value)).items(), key=lambda item: item[1], reverse = True))

    model_CNN2 = tf.keras.models.load_model("./models/CNN2.h5")
    CNN2_pred = model_CNN2.predict(img)
    CNN2_result = str(np.argmax(CNN2_pred, axis = 1)[0])
    value = np.round(np.ndarray.tolist(CNN2_pred[0]), 4)
    key = range(0,10)
    CNN2_pred_dict = dict(sorted(dict(zip(key, value)).items(), key=lambda item: item[1], reverse = True))
    
    return {"NN_result": NN_result, "NN2_result": NN2_result, "CNN_result": CNN_result, "CNN2_result": CNN2_result, 
    "NN_pred": json.dumps(NN_pred_dict), "CNN_pred": json.dumps(CNN_pred_dict), "NN2_pred": json.dumps(NN2_pred_dict), "CNN2_pred": json.dumps(CNN2_pred_dict)}

def parseImg(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open("./static/images/canvas_img.png", 'wb') as output:
        output.write(base64.decodebytes(imgstr))
    img_preprocess()

def img_preprocess():
    image = cv2.imread('./static/images/canvas_img.png')
    image = 255 - cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    image = image_grey2black(image, 255 / 8)
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        z = max(w, h)
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+z, y+z), color=(0, 255, 0), thickness=2)
        if(w < h):
            x = max(int(x - (h - w) / 2), 0)
        else:
            y = max(int(y - (w - h) / 2), 0)
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = image[y:y+z, x:x+z]

        # Resizing that digit to (18, 18)
        #resized_digit = cv2.resize(digit, (18,18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(digit, ((10,10),(10,10)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)

    inp = np.array(preprocessed_digits)
    image = cv2.resize(inp[0], (28, 28))
    image = image_grey2black(image, 255 / 8)
    cv2.imwrite("./static/images/process_img.png", image)

if __name__ == "__main__":
    app.run(debug=True)
