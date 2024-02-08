#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Wed Jun 23 10:06:05 2021
@author: ngbla 
"""
import os 
import cv2 as cv 
import numpy as np 
import pandas as pd 
from subprocess import PIPE, Popen 
import matplotlib.pyplot as plt 
import PIL
from convertisseur import detect_face

#Initialize names and path to empty list 
names = [] 
path = []
img_training = "memoire\img_training"
modelTrain = "memoire/trainingEigen.yml"

(im_width, im_height) = (160, 160) 
# Get the names of all the users 
# #img_modif_train #img_training"
for users in os.listdir(img_training): 
    names.append(users)

# Get the path to all the images 
for name in names: 
    for image in os.listdir(img_training+"\{}".format(name)):
        path_string = os.path.join(img_training+"\{}".format(name), image)
        path.append(path_string)

print('[PATH]')
print(path_string)
faces = []
ids = []

# For each image create a numpy array and add it to faces list
for img_path in path: 
    # Load image in grayscale 
    # image = Image.open(img_path).convert("L") 
    image = cv.imread(img_path) 
    # Resize like other images in dataset 
    image = cv.resize(image,(im_width, im_height) ) 
    # image = image.resize((im_width, im_height) , Image.ANTIALIAS) 
    # cv.imshow("Trainning image ...", image) 
    # cv.waitkey(100)
    # image = np.array(image, "uint8")
    #print('[INFO] - Face detection ... ')
    face, gray_image, cv = detect_face(image,cv)
    if face is not None :
        print('[INFO] Face ok')
        print('[INFO] img_path', img_path)
        #face = cv.resize(face,(im_width, im_height) )
        (x, y, w, h) = face[0]
        facedet = gray_image[y : y + h, x : x + w]
        face_resize =facedet
        face = cv.resize(face_resize, (im_width, im_height))
        faces.append(face)
        id = int(img_path.split("\\")[2].split("_")[0])
        print("id =", id)
        ids.append(id)

# Convert the ids to numpy array and add it to ids list
ids = np.array(ids)
print("ids =", ids)
print("[INFO] Created faces and names Numpy Arrays")
print("[INFO] Initializing the Classifier")
# Make sure contrib is installed
# The command is pip install opencv-contrib-python
# Call the recognizer
#trainer = cv2.face.LBPHFaceRecognizer_create()
#or use EigenFaceRecognizer by replacing above line with
trainer = cv.face.EigenFaceRecognizer_create()
#or use FisherFaceRecognizer by replacing above line with
#trainer = cv2.face.FisherFaceRecognizer_create() names Numpy Arrays")
# Give the faces and ids numpy arrays
trainer.train(faces, ids)
# Write the generated model to a yml file
#trainer.write("training.yml")
trainer.write(modelTrain)
print("[INFO] Training Done")
print("[INFO] Total faces :",len(faces))
print("[INFO] Total Labels :",len(ids))



