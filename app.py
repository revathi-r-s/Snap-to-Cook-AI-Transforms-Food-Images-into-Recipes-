from flask import render_template, url_for, flash, redirect, request
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from Foodimg2Ing import app
#from output import output
import os
import sqlite3
import numpy as np
import pandas as pd
from Recipe import *
import os
import cv2
import shutil
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

from flask import send_file
from gtts import gTTS

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

import ast

recipe_list = []

recipe_list.clear()
recipe_list=[]
indian = np.load('index.txt.npy', allow_pickle=True)
for i in range(len(indian)):
    recipe_list.append(indian[i])

obj = recipe_list

global classifier
if os.path.exists('model/cnn_model.json'):
    with open('model/cnn_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/cnn_model_weights.h5")
    #classifier._make_predict_function()  
    print(classifier.summary())

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/logon')
def logon():
    return render_template('signup.html')


@app.route('/login')
def login():
    return render_template('signin.html')


@app.route("/signup")
def signup():

    username = request.args.get('user', '')
    name = request.args.get('name', '')
    email = request.args.get('email', '')
    number = request.args.get('mobile', '')
    password = request.args.get('password', '')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",
                (username, email, password, number, name))
    con.commit()
    con.close()
    return render_template("signin.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get('user', '')
    password1 = request.args.get('password', '')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute(
        "select `user`, `password` from info where `user` = ? AND `password` = ?", (mail1, password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index1.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index1.html")
    else:
        return render_template("signup.html")


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(
        app.root_path, 'static\\images\\demo_imgs', imagefile.filename)
    print(image_path,'@@@@@@@@@@@@@@@@@')
    imagefile.save(image_path)
    print(imagefile,'#########################')
    img = "/images/demo_imgs/"+imagefile.filename
    title, ingredients, recipe = output(image_path)

    return render_template('predict.html', title=title, ingredients=ingredients, recipe=recipe, img=img)


@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    print("Entered")

    print("Entered here")
    file = request.files['imagefile']  # fet input
    filename = file.filename
    print("@@ Input posted = ", file, UPLOAD_FOLDER + file.filename)
    #os.remove(UPLOAD_FOLDER)
    
    fileList = os.listdir(UPLOAD_FOLDER)
    for fileName in fileList:  
        os.remove(UPLOAD_FOLDER + "/" + fileName)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(file_path)
    print(file_path,'$$$$$$$$$$$$$$$$$$$$$$')
    image = cv2.imread(file_path)
    cv2.imwrite('static/img/1.jpg',image)
    
    #image = cv2.imread(file_path)
    img = cv2.resize(image, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)
    if predict > 0:
        predict = predict - 1
    print(predict)

    obj = recipe_list[predict]
    name = obj.getName()
    ingre = obj.getIngredients()
    cook = obj.getCooking()
    #nutri = obj.getNutritions()
    print(name)
    print(ingre)
    print(cook)
    #print(nutri)
    return render_template('after.html', name=name, ingre=ingre, cook=cook,  img_src=UPLOAD_FOLDER + file.filename)#UPLOAD_FOLDER + file.filename)


@app.route('/notebook', methods=['GET'])
def notebook():
    return render_template('NOtebook.html')


@app.route('/index1', methods=['GET'])
def index1():
    return render_template('index1.html')


@app.route('/home1', methods=['GET'])
def home1():
    return render_template('home.html')

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form.get('text', 'Hello, welcome!')
    
    # Convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    audio_path = "C:/Users/hp/Desktop/project/static/speech"
    tts.save(audio_path)
# Return the audio file
    return send_file(audio_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
