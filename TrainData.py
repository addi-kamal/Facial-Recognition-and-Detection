# import libraries
from PIL import Image
import cv2
import numpy as np
import os

def train_data(data_directory):     # this function take in parameter the directory of data
    path = [os.path.join(data_directory, file) for file in os.listdir(data_directory)]
    list_faces = []
    list_ids = []

    for image in path:
        img = Image.open(image).convert("L")
        imgArray = np.array(img, "uint8")
        id = int(os.path.split(image)[1].split("-")[1][:4])
        list_faces.append(imgArray)
        list_ids.append(id)
    list_ids = np.array(list_ids)
    classifier = cv2.face.LBPHFaceRecognizer_create()
    classifier.train(list_faces, list_ids)
    classifier.write("MyClassifier.yml")
train_data("database/Kamal")