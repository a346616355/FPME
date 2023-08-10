# -*- coding: utf-8 -*-
"""
Main content: main function
@author: HaolongHong
"""
import tools as tl
import cv2
import converting as ct
import rendering as rd
import picking as pk
import numpy as np
'''read data'''
dataMat = tl.readData('data\..') #load data

normDataMat = tl.autoNorm(dataMat)

'''Step 1. Converting'''
binnaryImage = ct.converting(abs(normDataMat),19)

'''Step 2. Rendering'''
structuralElement = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))
# subroutine 1. dilation
dilatedImage = rd.dilate(binnaryImage,structuralElement,20)
# subroutine 2. dilation
erodedImage = rd.erode(dilatedImage,structuralElement,20)
# subroutine 3. delimitation
boundaryIndices = rd.getBoundary(erodedImage,normDataMat,2)

'''Step 3. Picking'''
# subroutine 1. preparation
greyscaleImage = pk.prepare(normDataMat,boundaryIndices)
# subroutine 2. detection
edgePointIndices = []
sampleValues = [] #detection results
# Roberts
edgePoints,values = pk.Roberts(greyscaleImage,normDataMat,65)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# Sobel
edgePoints,values = pk.Sobel(greyscaleImage,normDataMat,185)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# Prewitt
edgePoints,values = pk.Prewitt(greyscaleImage,normDataMat,140)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# Scharr
edgePoints,values= pk.Scharr(greyscaleImage,normDataMat,255)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# LoG
edgePoints,values = pk.LoG(greyscaleImage,normDataMat,255)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# Canny
edgePoints,values = pk.Canny(greyscaleImage,normDataMat,5)
edgePointIndices.append(edgePoints)
sampleValues.append(values)
# subroutine 3. evaluation
firstArrivalIndices,firstArrivalValues = pk.evaluate(edgePointIndices,sampleValues)


# plot all detection results
tl.plotAllResults(normDataMat,edgePointIndices,sampleValues,2)
# plot first arrivals
tl.plotFirstArrival(normDataMat,firstArrivalIndices,firstArrivalValues,2)
#compute accuracy
accuracy = tl.computeAccuracy(firstArrivalIndices,10)
print ("The accuracy is：" + str(accuracy))
