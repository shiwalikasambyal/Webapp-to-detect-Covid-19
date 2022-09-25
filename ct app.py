from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import matplotlib.pyplot as plt
from builtins import range, input
# before
from keras.preprocessing.image import ImageDataGenerator, load_img
# after
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
img_size=250

app = Flask(__name__) 

model=load_model('C:\\Users\\computer world\\Desktop\\link webapp to html\\vgg_ct_trail_50_epoch.h5')

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

def preprocess(img):
 #imgr=cv2.imread('imgr')
 #imgr  = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)      # arrange format as per deep learning libraries
 #imgr  = cv2.resize(imgr,(250,250))                 # resize as per model
 #x = img_to_array(imgr)  # Numpy array with shape (250, 250, 3)
 #x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 250, 250, 3)
 #x /= 255
 #return x
	img=np.array(img)

	#if(img.ndim==2):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	#else:
		#gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=img_to_array(resized)
	reshaped=resized.reshape((1,) + reshaped.shape)
	return reshaped

		
@app.route("/")
def index():
	return(render_template("CT index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=False, port=8000)