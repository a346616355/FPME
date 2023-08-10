# -*- coding: utf-8 -*-
"""
Main content: step 3. Picking
@author: HaolongHong
"""

import cv2
import numpy as np
import tools as tl

'''subroutine 1. preparation'''
def prepare(dataMat,boundaryIndices):
    preparationMat = np.zeros(dataMat.shape)
    for i in range(dataMat.shape[1]):
        preparationMat[int(boundaryIndices[i]):,i] = dataMat[int(boundaryIndices[i]):,i]
    preparationMat = abs(np.rint(preparationMat*255))
    retval, im_at_fixed = cv2.threshold(preparationMat,210,250,cv2.THRESH_TRUNC)
    cv2.imwrite('greyscaleImage.jpg',im_at_fixed)
    greyscaleImage = (cv2.imread('greyscaleImage.jpg'))[:,:,0]
    return greyscaleImage

'''subroutine 2. detection'''
#Roberts
def Roberts(greyscaleImage,dataMat,gamma):
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    x = cv2.filter2D(greyscaleImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(greyscaleImage, cv2.CV_16S, kernely)
    # trans to uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    robertsImg = cv2.addWeighted(absX,0.5,absY,0.5,0)
    #cv2.imshow("canny_data400.jpg", robertsImg)
    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,robertsImg,gamma)
    return selectedIndexArray,selectedValueArray

#Sobel
def Sobel(greyscaleImage,dataMat):
    x = cv2.Sobel(greyscaleImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(greyscaleImage, cv2.CV_16S, 0, 1)
    xy = cv2.Sobel(greyscaleImage,cv2.CV_16S, 1 , 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    xy = cv2.convertScaleAbs(xy)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,dst)
    return selectedIndexArray,selectedValueArray

#Prewitt
def Prewitt(greyscaleImage,dataMat):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
    x = cv2.filter2D(greyscaleImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(greyscaleImage, cv2.CV_16S, kernely)
    #trans to uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    prewittImg = cv2.addWeighted(absX,0.5,absY,0.5,0)
    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,prewittImg)
    return selectedIndexArray,selectedValueArray

# Scharr
def Scharr(greyscaleImage,dataMat):
    scharrx=cv2.Scharr(greyscaleImage,cv2.CV_64F,1,0)
    scharry=cv2.Scharr(greyscaleImage,cv2.CV_64F,0,1)
    scharrx=cv2.convertScaleAbs(scharrx)
    scharry=cv2.convertScaleAbs(scharry)
    xy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,xy)
    return selectedIndexArray,selectedValueArray

#LoG
def LoG(greyscaleImage,dataMat):
    imgGau = cv2.GaussianBlur(greyscaleImage,(3,3),0)
    gray_lap = cv2.Laplacian(greyscaleImage, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap)
    bond = np.hstack((greyscaleImage,dst))
    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,dst)
    return selectedIndexArray,selectedValueArray

#Canny
def Canny(greyscaleImage,dataMat):
    cannyImg = cv2.Canny(greyscaleImage, 230, 400)
    selectedIndexArray,selectedValueArray = tl.selectFirstPoint(dataMat,cannyImg)
    return selectedIndexArray,selectedValueArray

'''subroutine 3. evaluation'''
def evaluate(indices,values):
    indices = np.array(indices)
    values = np.array(values)
    m = indices.shape[0]
    n = indices.shape[1]
    evas = []
    for i in range(m):
        copy_i = np.tile(indices[i,:],(m,1))
        slice_i = np.exp(np.delete((-(copy_i - indices)**2),i,0))
        eva_i = np.sum(slice_i,0)
        evas.append(eva_i)
    evas = np.array(evas)
    selectIndices = evas.argmax(axis = 0)
    idx = []
    vlu = []
    for i in range(n):
        idx.append(indices[int(selectIndices[i]),i])
        vlu.append(values[int(selectIndices[i]),i])

    for i in range(n):
        if idx[i] == -1:
            for j in range(m):
                if indices[j,i] != -1:
                   idx[i] = indices[j,i]
                   vlu[i] = values[j,i]
                   break
    return idx,vlu