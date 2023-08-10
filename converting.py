# -*- coding: utf-8 -*-
"""
Main content: step 1. Converting
@author: HaolongHong
"""

import numpy as np
import cv2
from scipy import ndimage

def converting(dataMat,kernelSize):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    binnaryImage = np.zeros((dataMat.shape))
    '''subroutine 1. labeling'''
    for i in range(n):
        for j in range(m):
            if dataMat[j,i] >= 0.055:
                binnaryImage[j,i] = 1
    cv2.imwrite('labelMat.jpg',binnaryImage*255)
    '''subroutine 2. filtering'''
    binnaryImage = ndimage.median_filter(binnaryImage, size=kernelSize, mode="mirror")
    cv2.imwrite('updatedLabelMat.jpg',binnaryImage*255)
    return np.rint(binnaryImage)