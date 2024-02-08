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
#from detectAndDisplay import *
#from reconnaissance import *
from reconnaissance import recongition
#-- 2. Read the video stream
#camera_device = args.camera
#cap = cv.VideoCapture(camera_device)
cap = cv.VideoCapture(0)
# Names corresponding to each id
#names = ["0-Inconnu","1-Ngbla","2-Toure","3-Assie","4-Amael"]
names = ["0-Inconnu","1-Ngbla"]
#txt ="{0} - {1} "
users= []
"""
for users in os.listdir("img_training"):
 if (users == "1" or users == 1):
 #print(users)
 names.append(txt.format(users,"Ngbla") )
 if (users == "2" or users == 2) :
 #print(users)
 names.append(txt.format(users,"Elvis") )

"""
#-- END Infos User for recongnition 
count = 0
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
    
    if (count == 0):    
        cv.imshow("Recognize", frame)
    
    #detectAndDisplay(frame)
    recongition(frame,ret,names,cv)

    if cv.waitKey(10) == 27 :
        count = 0
        cap.release()
        cv.destroyAllWindows()
        exit(0)
        break





