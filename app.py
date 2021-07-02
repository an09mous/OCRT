from flask import Flask, render_template, request, redirect
import base64
import numpy as np
import cv2
import time
import pytesseract
from imutils.object_detection import non_max_suppression
from googletrans import Translator
from detect import *
from google_trans_new import google_translator  

app = Flask(__name__)
translator= Translator()

words=[]
corpus=[]
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload-image/result', methods=["POST"]) 
def nfun():
    val = request.form["Language"]
    Dict = {"Hindi": 'hi', "Japanese": 'ja', "French": 'fr', "German": 'de', "Italian": 'it', 
    "Korean": 'ko', "Arabic": 'ar', "Urdu": 'ur', "Irish":  'ga', "Spanish": 'es'}
    if request.method == "POST":
        translations=  translator.translate(" ".join(map(str, corpus)), dest=Dict[val])
        return render_template("Result.html",  words=[translations])
    return redirect('/upload-image')
    

        


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]      # file storage object
            npimg = np.fromstring(image.read(), np.uint8)    # array
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # ndarray

            global words
            words=[]
            image = east_detect(img)      # ndarray
            lst=[]
            for x in words:
                lst.append(x[2])
            global corpus
            corpus=lst
            _, buffer = cv2.imencode('.png', image)
            
            image_string = base64.b64encode(buffer)
            image_string = image_string.decode('utf-8')
            return render_template("output.html", filestring=image_string, words=lst)
    return redirect('/')

@app.route("/upload-image-hw", methods=["GET", "POST"])
def upload_image_hw():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]      # file storage object
            npimg = np.fromstring(image.read(), np.uint8)    # array
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # ndarray

            res = solve(cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE))[0]
            # image = east_detect(img)
            lst=res.split()
            global corpus
            corpus=lst
            image = east_detect(img)
            _, buffer = cv2.imencode('.png', image)
            
            image_string = base64.b64encode(buffer)
            image_string = image_string.decode('utf-8')
            return render_template("output.html", filestring=image_string, words=lst)
    return redirect('/')



def east_detect(image):
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    orig = image.copy()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    (H, W) = image.shape[:2]


    (newW, newH) = (320, 320)

    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))

    (H, W) = image.shape[:2]

    net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        global words
        word= []
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

   
    for (startX, startY, endX, endY) in boxes:
        startX = max(int(startX * rW) - 5, 0)
        startY = max(int(startY * rH) - 5, 0)
        endX = int(endX * rW) + 5
        endY = int(endY * rH) + 5

        print("./././.", startX, startY, endX, endY)
        roi = orig[startY:endY, startX:endX]
        words.append((startX, startY, tesseract(roi)))

        cv2.rectangle(orig, (startX, startY),
                      (endX, endY), (0, 255, 0), 2)

    words.sort(key=lambda x: x[1])
    id = 1

    for startX, startY, _ in words:
        orig = cv2.putText(orig, str(id), (startX, startY - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        id += 1

    print(words)
    print("Time taken", time.time() - start)
    return orig 


def tesseract(image):
    text = pytesseract.image_to_string(
        image, config=("-l eng --oem 1 --psm 8"))
    text = text.split('\n')[0]
    return text

if __name__ == "__main__":
    app.run()
