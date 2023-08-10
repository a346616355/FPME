# -*- coding: utf-8 -*-
"""
Main content: step 2. Rendering
@author: HaolongHong
"""
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
'''subroutine 1. dilation'''
def dilate(binnaryImage,structuralElement,nums):
    dilatedImage = binnaryImage
    for i in range(nums):
        tempDilated = cv2.dilate(dilatedImage,structuralElement)
        dilatedImage = tempDilated
    cv2.imwrite("dilatedImage.jpg",dilatedImage*255);
    return dilatedImage

'''subroutine 2. erosion'''
def erode(dilatedImage,structuralElement,nums):
    erodedImage = dilatedImage
    for i in range(nums):
        tempEroded = cv2.erode(erodedImage,structuralElement)
        erodedImage = tempEroded
    cv2.imwrite("erodedImage.jpg",erodedImage*255);
    return erodedImage

'''subroutine 3. delimitation'''
def getBoundary(erodedMat,dataMat,k):
    m, n = dataMat.shape
    sf = []
    df = []
    default_sf_value = 0  # Choose an appropriate default value for sf
    default_df_value = 0  # Choose an appropriate default value for df

    for i in range(n):
        found_boundary = False
        for j in reversed(range(m)):
            if erodedMat[j, i] == 0:
                sf.append(int(j))
                df.append(dataMat[int(j), i] + i * k)
                found_boundary = True
                break

        # If a boundary point is not found for a column, insert the default value
        if not found_boundary:
            sf.append(default_sf_value)
            df.append(default_df_value)
    sf = np.array(sf)
    df = np.array(df)

    '''plot upper boundary of the signal zone'''
    sample=np.arange(m)    #sample coordinate array
    boundLine=np.zeros(m)    #boundary coordinate array
    datalist=np.zeros(m)
    fig=plt.figure(figsize=(10,10))    #set canvas size
    ax=plt.gca()
    if n%8 == 0:
        a = range(0,n*22//8,n//5)
        b = range(0,n*11//8,n//10)
    else:
        a = range(0,n*2+60,60)
        b = range(0,n+30,30)
    plt.xticks(a,b)
    plt.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.ylabel('sample',size = 20)
    plt.title('trace number',size = 20,pad=37)
    sf=sf.tolist()
    df=df.tolist()
    while -1 in sf:     #remove points marked as errors
        sf.remove(-1)
    while -2 in df:
        df.remove(-2)
    for i in range(n):
        datalist=np.array(dataMat[0:m,i])+i*k
        plt.plot(boundLine+i*k,sample,color='black',linewidth=0.3)
        plt.plot(datalist,sample,color='black',linewidth=0.7)
        plt.fill_betweenx(sample,datalist,boundLine+i*k,where=(datalist>i*k) & (datalist<i*k+1),facecolor='black',alpha=1)
        plt.scatter(df[i],sf[i],s=30,marker='o',color='r',edgecolors='r')

    plt.show()
    # plt.close()
    return sf