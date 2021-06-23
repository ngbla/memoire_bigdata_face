# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 05:39:14 2021

@author: ngbla
"""
# import the necessary packages
import cv2 as cv
print(cv.__version__)
import numpy as np
print(np.__version__)

#import matplotlib

#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('webcammer-master/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('webcammer-master/haarcascade_eye.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR5652GRAY )
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Display the resulting frame
    cv.imshow('frame', gray)


    # Detect the faces
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw the rectangle around each face   
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("Un Visage détecter !")
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display
    cv.imshow('img', frame)
    # Stop if escape key is pressed
    #k = cv.waitKey(30) & 0xff


    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()