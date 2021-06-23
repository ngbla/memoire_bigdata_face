#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 05:30:05 2021

@author: ngbla
"""
import cv2 as cv
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print( flags )