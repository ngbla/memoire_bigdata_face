#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 03:48:40 2021

@author: ngbla
"""
from __future__ import print_function

import cv2
import os
import argparse
    
# Begin fonction  
def recongition(frame,ret,names):
    
    # Call the recognizer
    # Call the trained model yml file to recognize faces
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #or use EigenFaceRecognizer by replacing above line with 
    #recognizer = cv2.face.EigenFaceRecognizer_create()
    #or use FisherFaceRecognizer by replacing above line with 
    #recognizer = cv2.face.FisherFaceRecognizer_create() names Numpy Arrays")
    
    recognizer.read("training.yml")

    #img = cv2.imread("test/chris.jpeg")
    #while True:
    #_, img = video_capture.read()

    
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='webcammer-master/haarcascade_frontalface_alt.xml')
    
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    
    face_cascade_name = args.face_cascade
    face_cascade = cv2.CascadeClassifier()

    #-- 1. Load the cascades (verification de l'existences des fichier xml)
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    """
    """
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
    )
    
    #faces = face_cascade.detectMultiScale(gray_image)

    # Try to predict the face and get the id
    # Then check if id == 1 or id == 2
    # Accordingly add the names
    reps ="{0} - {1} "   
    

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray_image[y : y + h, x : x + w])
        if id:
            cv2.putText(
                frame,
                reps.format(names[id - 1],conf),
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                reps.format("Unknown",conf),
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Recognize", frame)

# End fonction  