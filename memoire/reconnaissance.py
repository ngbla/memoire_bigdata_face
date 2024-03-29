#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 03:48:40 2021
@author: ngbla
"""
from __future__ import print_function
#import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from convertisseur import detect_face
import os

import pyscreenshot as ImageGrab
import time
from PIL import Image as ImagePil

# Begin fonction
def recongition(frame,ret,names,cv):
    modelTrain = "memoire/trainingEigen.yml"
    newImgDir = "memoire/save_file"
    
    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print(current_working_directory)
    
    # Call the recognizer
    # Call the trained model yml file to recognize faces
    recognizer_LBPH = cv.face.LBPHFaceRecognizer_create()
    #or use EigenFaceRecognizer by replacing above line with
    recognizer_Eigen = cv.face.EigenFaceRecognizer_create()
    #or use FisherFaceRecognizer by replacing above line with 
    recognizer_Fisher = cv.face.FisherFaceRecognizer_create()
    
    recognizer = recognizer_LBPH
    print('--(!)recongition read : trainingEigen')
    #recognizer.read("training.yml")
    recognizer.read(modelTrain)
    #img = cv.imread("test/chris.jpeg")
    #while True:
    #_, img = video_capture.read()
    (im_width, im_height) = (160, 160)

    print('[INFO] - Face detection ... ')
    faces, gray_image, cv =detect_face(frame,cv)
    
    if faces is None :
        print('[INFO] Face Empty !')
        return "empty"
    
    print('[INFO] Face Detect ...')
    # Try to predict the face and get the id
    # Then check if id == 1 or id == 2
    # Accordingly add the names
    reps ="{0} - {1} "

    for (x, y, w, h) in faces:
        print('-- face detecter --')
        #Localisation des yeux
        #faceROI = gray_image[y:y+h,x:x+w]
        #-- In each face, detect eyes
        #eyes = eyes_cascade.detectMultiScale(faceROI)
        #for (x2,y2,w2,h2) in eyes:
        #eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #radius = int(round((w2 + h2)*0.25))
        #frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        #Localisation des yeux

        # Coordinates of face after scaling back by `size`
        facedet = gray_image[y : y + h, x : x + w]
        face_resize = facedet
        face_resize = cv.resize(face_resize, (im_width, im_height))
        #Convert to array
        #imgNp = np.array(face_resize, "uint8")
        #Affichage
        #plt.imshow(imgNp, cmap=plt.get_cmap('gray'))

        # Try to recognize the face
        id, conf = recognizer.predict(face_resize)
        #taux = round(conf/10000, 3)+"%"
        #taux_verif = 5000
        #LBPHF
        taux = str(round(conf, 3))+"%"
        taux_verif = 50
        

        print('[USER ID] = ', id)
        print('[PRECISION] = ', conf)
        if ((id> 1 or 5> id) and conf > taux_verif):
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(
                frame,
                reps.format(names[id],taux),
                #names[id],
                (x, y - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

            print('-- id user=', id)
        else:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv.putText(
                frame,
                reps.format("Unknown",taux),
                #"0-Inconnu",
                (x, y - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                1,
                cv.LINE_AA,
            )
            print('-- Unknown')
            # Save image
            t1 = time.time()
            #imgScreen = ImageGrab.grab(backend="mss", childprocess=False)
            #img = imgScreen.resize((640,480))
            #img.save(newImgDir+"/"+str(t1)+"screen.png")
            
            #img.save(newImgDir+"/"+"screen.png")
            #cv.imshow("Resized image", frame)
            im = ImagePil.fromarray(facedet)
            #im.save(newImgDir+"/"+str(id)+"_"+str(t1)+".png")
            
    cv.imshow("Recognize", frame)
# End fonction 