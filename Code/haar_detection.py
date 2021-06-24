#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 06:15:43 2021

@author: ngbla
"""
from __future__ import print_function
# import the necessary packages
import cv2 as cv
print(cv.__version__)
import numpy as np
print(np.__version__)
#-- Import de la fonction dans le meme dossier
from detectAndDisplay import *
from reconnaissance import *
#-- 2. Read the video stream
#camera_device = args.camera
#cap = cv.VideoCapture(camera_device)
cap = cv.VideoCapture(0)

# Names corresponding to each id
names = []
for users in os.listdir("img_training"):
    if users == "1" :
        txt ="{0} - {1} "
        #print("users")
        #print(users)
        names.append(txt.format(users,"NGBLA") )
#-- END Infos User for recongnition 


if not cap.isOpened:
    print('--(!) Caméra vidéo en cour d utilisation !')
    cap.release()
    cv.destroyAllWindows()
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) Image Absente -- Exit !')
        cap.release()
        cv.destroyAllWindows()
        break
    
    #detectAndDisplay(frame)
    recongition(frame,ret,names)
    
    if cv.waitKey(10) == 27 :
        break