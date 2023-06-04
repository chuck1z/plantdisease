import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import shutil
import os
from PIL import Image
from keras.models import load_model

# Load the trained model
model = tf.keras.models.load_model('CNNmodif.h5')

# Define the image size and preprocessing function
img_size = (224, 224)
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(input_file):
    img = preprocess_image(input_file)
    pred = model.predict(img)
    result = np.argmax(pred)
    return (result)

app = FastAPI()

@app.get("/")
def hello_world():
    return ("hello world")

@app.post("/predict")
def classify(input: UploadFile = File(...)):
    print(input.filename)
    print(type(input.filename))
    savefile = input.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)
    result = predict(savefile)
    os.remove(savefile)
    return str(result)