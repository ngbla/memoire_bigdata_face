#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:56:57 2021
@author: ngbla
"""
from __future__ import print_function
#import cv2
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
from random import randrange

# Begin fonction
def detect_face(image,cv):
    parser = argparse.ArgumentParser(description='Classifier Import')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='memoire/webcammer-master/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='memoire/webcammer-master/haarcascade_eye_tree_eyeglasses.xml')
    args = parser.parse_args()

    face_cascade_name = args.face_cascade
    face_cascade = cv.CascadeClassifier()

    eyes_cascade_name = args.eyes_cascade
    eyes_cascade = cv.CascadeClassifier()
    
    #-- 1. Load the cascades (verification de l'existences des fichier xml)
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image)
    #faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)

    #print('[INFO] len face', len(faces))
    if( len(faces) == 0) :
        return None,None, cv

    #print('[INFO) End detection face')
    #return gray_image[y:y+w, x:x+h], faces[0]
    #(im_width, im_height) = (160, 160)
    #faces = cv.resize(faces,(im_width, im_height) )
    return faces, gray_image, cv
    #End