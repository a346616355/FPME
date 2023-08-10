# -*- coding: utf-8 -*-
"""
Main function:Provide tool functions
@author: HaolongHong
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

"""
    Extract data from the source dataset and store it in a matrix
"""


def readData(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numOfLines = len(arrayOLines)
    numOfRows = len((arrayOLines[0].strip()).split('  '))
    dataMat = zeros((numOfRows, numOfLines))
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('  ')
        dataMat[:, index] = listFromLine[0:numOfRows]
        index += 1
    return dataMat


"""
    Extract data and store it in a matrix
"""


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numOfLines = len(arrayOLines)
    numOfRows = len((arrayOLines[0].strip()).split(' '))
    dataMat = zeros((numOfLines, numOfRows))
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = np.array(line.split(' '))
        dataMat[index, :] = listFromLine[0:numOfRows]
        index += 1
    return dataMat


"""
    Store the matrix in a file
"""


def matrix2file(filename, dataMat):
    f = open(filename, 'w+')
    for i in range(dataMat.shape[0]):
        dataList = dataMat[i, :]
        for j in range(dataMat.shape[1]):
            label = str(int(dataList[j]))
            f.write(label)
            f.write(' ')
        f.write('\n')
    f.close()


"""
    Standard normalization method to map data between -1 and 1
"""


def autoNorm(dataMat):
    m = dataMat.shape[0]  # row number
    absMat = np.zeros(np.shape(dataMat))  # initialize the absolute value matrix
    signMat = np.zeros(np.shape(dataMat))  # initialize the positive and negative matrix
    normDataMat = np.zeros(np.shape(dataMat))
    for i in range(m):
        array = dataMat[i, :]
        absMat[i, :] = np.abs(array)
        signMat[i, :] = np.sign(array)
    minVals = absMat.min(0)  # get the minimum value of each column
    maxVals = absMat.max(0)  # get the maximum value of each column
    ranges = maxVals - minVals
    normDataMat = (absMat - np.tile(minVals, (m, 1)))
    normDataMat = normDataMat / np.tile(ranges, (m, 1))
    normMat = np.multiply(normDataMat, signMat)
    return normMat


"""
    Draw a seismic waveform
"""


def plotWave(dataMat):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    sample = []  # sample coordinate array
    boundLine = []  # boundary coordinate array
    datalist = np.zeros(m)
    for i in range(m):
        sample.append(i)  # constructing sample axis coordinates
    for i in range(m):
        boundLine.append(0)  # constructing boundary coordinates
    boundLine = np.array(boundLine[:])
    fig = plt.figure(figsize=(10, 10))  # set canvas size
    ax = plt.gca()
    if n % 8 == 0:
        a = range(0, n * 22 // 8, n // 5)
        b = range(0, n * 11 // 8, n // 10)
    else:
        a = range(0, n * 2 + 60, 60)
        b = range(0, n + 30, 30)
    plt.xticks(a, b)
    plt.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.ylabel('sample', size=20)
    plt.title('trace number', size=20, pad=37)
    for i in range(n):
        datalist = np.array(dataMat[0:m, i]) + i * 2
        plt.plot(boundLine + i * 2, sample, color='black', linewidth=0.3)
        plt.plot(datalist, sample, color='black', linewidth=0.7)
        plt.fill_betweenx(sample, datalist, boundLine + i * 2, where=(datalist > i * 2) & (datalist < i * 2 + 1),
                          facecolor='black', alpha=1)
    plt.show()


"""
   Choose the first point of the class on each trace
"""


def selectFirstPoint(dataMat, imgMat, a, b):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    d_first = []  # energy axis coordinate of the point
    s_first = []  # Sample axis coordinate of the point
    for i in range(n):
        imgList = imgMat[:, i]  # find the point where the clustering label in the data set is equal to cls
        singleTrace = dataMat[:, i]
        haveFlag = 0  # the exist flag of class
        for j in range(m):
            if imgList[j] >= a and imgList[
                j] <= b:  # store all point coordinates in the data set clustering label equal to cls
                haveFlag = 1
                s_first.append(int(j))
                d_first.append(singleTrace[j] + i * 2)
                break
        if haveFlag == 0:  # if there is no such class on the trace
            s_first.append(-1)
            d_first.append(-2)
    s_first = np.array(s_first)
    d_first = np.array(d_first)
    return s_first, d_first


"""
    Choose the first point of the class on each trace
"""


def selectFirstPoint(dataMat, imgMat, a):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    d_first = []  # energy axis coordinate of the point
    s_first = []  # Sample axis coordinate of the point
    for i in range(n):
        imgList = imgMat[:, i]  # find the point where the clustering label in the data set is equal to cls
        singleTrace = dataMat[:, i]
        haveFlag = 0  # the exist flag of class
        for j in range(m - 1):
            if imgList[j] >= imgList[j - 1] and imgList[j] >= imgList[j + 1] and imgList[
                j] >= a:  # store all point coordinates in the data set clustering label equal to cls
                haveFlag = 1
                s_first.append(int(j))
                d_first.append(singleTrace[j] + i * 2)
                break
        if haveFlag == 0:  # if there is no such class on the trace
            s_first.append(-1)
            d_first.append(-2)
    s_first = np.array(s_first)
    d_first = np.array(d_first)
    return s_first, d_first


"""
    Draw the first-arrival of the seismic data
"""


def plotFirstArrival(dataMat, sf, df, k):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    sample = []  # sample coordinate array
    boundLine = []  # boundary coordinate array
    datalist = np.zeros(m)
    for i in range(m):
        sample.append(i)  # constructing sample axis coordinates
    for i in range(m):
        boundLine.append(0)  # constructing boundary coordinates
    boundLine = np.array(boundLine[:])
    fig = plt.figure(figsize=(10, 10))  # set canvas size
    ax = plt.gca()
    if n % 8 == 0:
        a = range(0, n * 22 // 8, n // 5)
        b = range(0, n * 11 // 8, n // 10)
    else:
        a = range(0, n * 2 + 60, 60)
        b = range(0, n + 30, 30)
    plt.xticks(a, b)
    plt.tick_params(labelsize=15)  # 刻度字体大小13
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.ylabel('sample', size=20)
    plt.title('trace number', size=20, pad=37)
    for i in range(n):
        datalist = np.array(dataMat[0:m, i]) + i * k
        plt.plot(boundLine + i * k, sample, color='black', linewidth=0.3)
        plt.plot(datalist, sample, color='black', linewidth=0.7)
        plt.fill_betweenx(sample, datalist, boundLine + i * k, where=(datalist > i * k) & (datalist < i * k + 1),
                          facecolor='black', alpha=1)
    plt.scatter(df, sf, s=30, marker='o', color='r', edgecolors='r')

    plt.show()


"""
    Draw the first-arrival of the seismic data
"""


def plotAllResults(dataMat, sf, df, k):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    sample = []  # sample coordinate array
    boundLine = []  # boundary coordinate array
    datalist = np.zeros(m)
    for i in range(m):
        sample.append(i)  # constructing sample axis coordinates
    for i in range(m):
        boundLine.append(0)  # constructing boundary coordinates
    boundLine = np.array(boundLine[:])
    fig = plt.figure(figsize=(10, 10))  # set canvas size
    ax = plt.gca()
    if n % 8 == 0:
        a = range(0, n * 22 // 8, n // 5)
        b = range(0, n * 11 // 8, n // 10)
    else:
        a = range(0, n * 2 + 60, 60)
        b = range(0, n + 30, 30)
    plt.xticks(a, b)
    plt.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.ylabel('sample', size=20)
    plt.title('trace number', size=20, pad=37)

    for i in range(n):
        datalist = np.array(dataMat[0:m, i]) + i * k
        plt.plot(boundLine + i * k, sample, color='black', linewidth=0.3)
        plt.plot(datalist, sample, color='black', linewidth=0.7)
        plt.fill_betweenx(sample, datalist, boundLine + i * k, where=(datalist > i * k) & (datalist < i * k + 1),
                          facecolor='black', alpha=1)
    colors = ['brown', 'g', 'b', 'darkorange', 'deeppink', 'purple']
    markers = ['x', 'p', 'v', 's', '*', 'o']
    labels = ['Roberts', 'Sobel', 'Prewitt', 'Scharr', 'Laplacian', 'Canny']
    for i in range(len(sf)):
        s = sf[i].tolist()
        d = df[i].tolist()
        while -1 in s:  # remove points marked as errors
            s.remove(-1)
        while -2 in d:
            d.remove(-2)
        plt.scatter(d, s, s=35 - i * 5, marker=markers[i], color=colors[i], label='$' + labels[i] + '$')
    plt.legend(loc='upper right')
    plt.show()


def plotComResults(dataMat, idx, k):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    sample = []  # sample coordinate array
    boundLine = []  # boundary coordinate array
    datalist = np.zeros(m)
    vlus = []
    for i in range(4):
        vlu = []
        for j in range(len(idx[i])):
            vlu.append(dataMat[int(idx[i][j]), j] + k * j)
        vlus.append(vlu)

    for i in range(m):
        sample.append(i)  # constructing sample axis coordinates
    for i in range(m):
        boundLine.append(0)  # constructing boundary coordinates
    boundLine = np.array(boundLine[:])
    fig = plt.figure(figsize=(10, 10))  # set canvas size
    ax = plt.gca()
    if n % 8 == 0:
        a = range(0, n * 22 // 8, n // 5)
        b = range(0, n * 11 // 8, n // 10)
    else:
        a = range(0, n * 2 + 60, 60)
        b = range(0, n + 30, 30)
    plt.xticks(a, b)
    plt.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.ylabel('sample', size=20)
    plt.title('trace number', size=20, pad=37)

    for i in range(n):
        #        if i%2==0:
        datalist = np.array(dataMat[0:m, i]) + i * k
        plt.plot(boundLine + i * k, sample, color='black', linewidth=0.3)
        plt.plot(datalist, sample, color='black', linewidth=0.7)
        plt.fill_betweenx(sample, datalist, boundLine + i * k, where=(datalist > i * k) & (datalist < i * k + 1),
                          facecolor='black', alpha=1)
    colors = ['brown', 'g', 'b', 'darkorange', 'deeppink', 'purple']
    markers = ['x', 'p', 'v', 's', '*', 'o']
    labels = ['DC', 'MCM', 'APF', 'FPME', 'Laplacian', 'Canny']
    for i in range(len(idx)):
        plt.scatter(vlus[i], idx[i], s=90 - i * 10, marker=markers[i], color=colors[i], label='$' + labels[i] + '$')
    plt.legend(loc='upper right')
    plt.show()


def computeAccuracy(indices, cutDist):
    dataSource_1 = np.array([1, 2, 3, ...])  # Array of manually labeled initial arrival points
    dataSource_2 = np.array([1, 2, 3, ...])  # Array of manually labeled initial arrival points
    dataSource_3 = np.array([1, 2, 3, ...])  # Array of manually labeled initial arrival points
    dataSource_4 = np.array([1, 2, 3, ...])  # Array of manually labeled initial arrival points

    if (len(indices) == 200):
        diff = abs(indices - dataSource_1)
        acNum = len(diff[diff <= cutDist])
        accuracy = acNum * 1.0 / len(dataSource_1)
    if (len(indices) == 300):
        diff = abs(indices - dataSource_2)
        acNum = len(diff[diff <= cutDist])
        accuracy = acNum * 1.0 / len(dataSource_2)
    if (len(indices) == 400):
        diff = abs(indices - dataSource_3)
        acNum = len(diff[diff <= cutDist])
        accuracy = acNum * 1.0 / len(dataSource_3)
    if (len(indices) == 320):
        diff = abs(indices - dataSource_4)
        acNum = len(diff[diff <= cutDist])
        accuracy = acNum * 1.0 / len(dataSource_4)
    return accuracy
