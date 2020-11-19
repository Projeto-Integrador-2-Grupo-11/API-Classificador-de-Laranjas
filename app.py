# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import cv2
import numpy as np
from keras.models import Model, load_model
from pymongo import MongoClient

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEVICE = "cpu"
MODEL = None


def connect_and_save_mongo(classification):
    
    print('conectando mongo')
    #Conecta mongo local
	cliente_prod = MongoClient('mongodb://admin:admin@localhost:27017/admin')

	db_prod   = cliente_prod.admin
	coll_prod = db_prod.new_collection
	mydict = { "classification": classification, "image": "image 1" }
	coll_prod.insert_one(mydict)
    print('insert realizado')

def predict(image_path, model):
    I = []
    print(image_path)
    img=cv2.imread(image_path)
    img=cv2.resize(img,(100,100))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    I.append(img)
    NP_ARRAY = np.array(I)
    X_VAL = NP_ARRAY/255.0
    R = MODEL.predict_classes(X_VAL)
    print(model.predict(X_VAL) > 0.5).astype("int32"))
    #Salva mongo
    connect_and_save_mongo(R[0])

    if(R[0]==0):
        return 'BOA SEM MANCHAS'
    if(R[0]==1):
        return 'RUIM'
    if(R[0]==2):
        return 'BOA COM MANCHAS'
    
    #Manda p/ eletronica


    #Faz o reduce dos resultados (pega varias fotos da mesma laranja e retorna um resultado)

    return MODEL.predict_classes(np.array(I))


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            return render_template("index.html", prediction=pred, image_loc='./static/'+image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    
    MODEL=load_model('./h5/rottenvsfresh.h5')   
    
    app.run(host='0.0.0.0')
