# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:07:13 2018

@author: Alexander
"""

import numpy as np

## try to import tqdm to give DSC cuts nice progressbar
try:
    from tqdm import tqdm
    haveTQDM = True
except:
    haveTQDM = False

def permutationTest(doseData, statuses, nperm=1000):
    """
    Perform a permutation test to get the global p-value and t-thresholds
    Inputs:
        - doseData: the dose data, should be structured such that the number of patients in it is along the last axis
        - statuses: the outcome labels. 1 indicates an event, 0 indicates no event.
        - nperm: The number of permutations to calculate. Defaults to 1000 which is the minimum for reasonable accuracy
    Returns:
        - globalPNeg: the global significance of the test for negative t-values
        - globalPPos: the global significance of the test for positive t-values
        - tThreshNeg: the list of minT from all the permutations, use it to set a significance threshold.
        - tThreshPos: the list of maxT from all the permutations, use it to set a significance threshold.
    """
    tthresh = []
    gtCount = 0
    ltCount = 0
    trueT = imagesTTest(doseData, statuses)
    trueMaxT = np.max(trueT)
    trueMinT = np.min(trueT)
    if haveTQDM:
        for perm in tqdm(range(nperm)):
            tthresh.append(doPermutation(doseData, statuses))
            if tthresh[-1][1] > trueMaxT:
                gtCount += 1.0
            if tthresh[-1][0] < trueMinT:
                ltCount += 1.0
    else:
        for perm in range(nperm):
            tthresh.append(doPermutation(doseData, statuses))
            if tthresh[-1][1] > trueMaxT:
                gtCount += 1.0
            if tthresh[-1][0] < trueMinT:
                ltCount += 1.0

    globalpPos = gtCount / float(nperm)
    globalpNeg = ltCount / float(nperm)
    tthresh = np.array(tthresh)
    return globalpNeg, globalpPos, sorted(tthresh[:, 0]), sorted(tthresh[:, 1])
    

def doPermutation(doseData, statuses):
    """
    Permute the statuses and return the maximum t value for this permutation
    Inputs:
        - doseData: the dose data, should be structured such that the number of patients in it is along the last axis
        - statuses: the outcome labels. 1 indicates an event, 0 indicates no event. These will be permuted in this function to
                    assess the null hypothesis of no dose interaction
    Returns:
        - (tMin, tMax): the extreme values of the whole t-value map for this permutation
    """
    pstatuses = np.random.permutation(statuses)
    permT = imagesTTest(doseData, pstatuses)


    return (np.min(permT), np.max(permT))

def imagesTTest(doseData, statuses):
    """
    Calculate per-voxel t statistic between two images. Uses Welford's method to calculate mean and variance.
    NB: there is DSC cuts tricky little bit at the end. Welford's method requires dividing by N to get variance, the T-test
    also requires dividing by N, so we just divide by N^2.
    Inputs:
        - doseData: the dose data, should be structured such that the number of patients in it is along the last axis
        - statuses: the outcome labels. 1 indicates an event, 0 indicates no event
    Returns:
        - Tvalues: an array of the same size as one of the images which contains the per-voxel t values
    """
    ## expand non 3D data
    while len(doseData.shape) <  4:
        doseData = np.expand_dims(doseData, axis=0)

    

    noEventMean = np.zeros_like(doseData[:,:,:,0])
    eventMean = np.zeros_like(doseData[:,:,:,0])# + np.finfo(float).eps

    noEventStd = np.zeros_like(doseData[:,:,:,0])
    eventStd = np.zeros_like(doseData[:,:,:,0])# + np.finfo(float).eps

    eventCount = 0
    nonEventCount = 0
    for n, stat in enumerate(statuses):
        if stat == 1 and eventCount == 0:
            eventCount += 1.0
            eventMean = doseData[:,:,:,n]
        elif stat == 1:
            eventCount += 1.0
            om = eventMean.copy()
            eventMean = om + (doseData[:,:,:,n] - om)/eventCount
            eventStd = eventStd + ((doseData[:,:,:,n] - om)*(doseData[:,:,:,n] - eventMean))


        elif stat == 0 and nonEventCount == 0:
            nonEventCount += 1.0
            noEventMean = doseData[:,:,:,n]
        elif stat == 0:
            nonEventCount += 1.0
            om = noEventMean.copy()
            noEventMean = om + (doseData[:,:,:,n] - om)/nonEventCount
            noEventStd = noEventStd + ((doseData[:,:,:,n] - om)*(doseData[:,:,:,n] - noEventMean))

    eventStd /= (eventCount**2)      
    noEventStd /= (nonEventCount**2)

    Tvalues = np.divide((eventMean - noEventMean), np.sqrt(eventStd + noEventStd))
    
    return Tvalues