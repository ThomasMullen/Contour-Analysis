#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns;
import collections
from Code.AllPatients import AllPatients

sns.set()
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# =============================================================================
# Load patients list data for all fractions
# =============================================================================

SaveDirect = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/"

# =============================================================================
# Specify the corrupt patients to be filtered out of analysis
# =============================================================================

# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
         '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}
# corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
# corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}

allPatients = AllPatients(r"../Data/OnlyProstateResults/Global",
                          ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
allPatients.removePatients(atlas)

# =============================================================================
# Group the patients by fractions, and recurrence
# =============================================================================

(PatientsWhoRecur, PatientsWhoDontRecur) = allPatients.recurrenceGroups()

# Group patients with fractions
PatientRecurrencew19Frac = PatientsWhoRecur.groupby('Fractions').get_group(19)
PatientRecurrencew16Frac = PatientsWhoRecur.groupby('Fractions').get_group(16)
PatientNonRecurrencew20Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(20)
PatientNonRecurrencew19Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(19)
PatientNonRecurrencew16Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(16)

# =============================================================================
# # Read in the patients map and store in correct container
# =============================================================================
# Patient map containers
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

corLocal = {'200801658', '200606193', '200610929', '200701370'}

'''class to hold calculated data for a givient patient'''


class PatientCalcResults:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix


'''
returns sdValue for a given patientMap
'''

RadialStats = collections.namedtuple('RadialStats', 'mean sd')

def calcPatientMapSD(patientMap):
    sxx = 0
    mapMean = (sum(patientMap.flatten())) / patientMap.size
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff - mapMean) ** 2
    sdValue = np.sqrt(sxx / (patientMap.size - 1))
    return RadialStats(mean=mapMean, sd=sdValue)

#
# def calcPatientMapMean(patientMap):
#     return (sum(patientMap.flatten())) / patientMap.size
#
# #
#
# def loadPatientMap(dataDirectory, patientId):
#     file = r"%s/%s.csv" % (dataDirectory, patientId)
#     matrix = pd.read_csv(file, header=None).as_matrix()
#     return PatientCalcResults(patientId, matrix)
#
# #
# # # Read in map
# class SDVAlResult:
#     def __init__(self, patientId, mean, sd):
#         self.patientId = patientId
#         self.mean = mean
#         self.sd = sd

#
# def extractPatientSDVals(dataDir, allPatientsDF):
#     meanCell = []
#     sdCell = []
#     totalPatients = len(allPatientsDF)
#     for x in range(0, totalPatients):
#         name = str(allPatientsDF["patientList"].iloc[x])
#         if name not in corLocal:
#             patientMap = loadPatientMap(dataDir, name)
#             (mean, sdValue) = calcPatientMapSD(patientMap.matrix)
#             meanCell.append(mean)
#             sdCell.append(sdValue)
#     return (meanCell, sdCell)
#

def calcPatientMapSD2(dataDir, patientId):
    file = r"%s/%s.csv" % (dataDir, patientId)
    matrix = pd.read_csv(file, header=None).as_matrix()
    result =  calcPatientMapSD(matrix)
    return result


def addCalcCols(dataDir, allPatientsDF):

    df = allPatientsDF.assign(radMapStats=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2(dataDir, x)))
    df2 = df.apply(mean=lambda df: calcPatientMapSD2(dataDir, df.radMapStats._1))
    df3 = df2.apply(sd=lambda df: calcPatientMapSD2(dataDir, df["radMapStats"])._2))
    return df3


# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================
# Create Arrays
# phi = []; theta =[]
# for i in range(0,120):
#    phi.append('')
# for i in range(0,60):
#    theta.append('')
## Define ticks
# phi[0] = 0; phi[30] = 90; phi[60] = 180; phi[90] = 270; phi[119] = 360;
# theta[0] = -90; theta[30] = 0; theta[59] = 90
def plotHist(data, colour, bin, name="Single Value"):
    result = plt.hist(data, bins=bin, alpha=0.5, label='map mean', color=colour)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.xlim((min(data), max(data)))
    # mean = np.mean(meanCell)
    # variance = np.var(meanCell)
    # sigma = np.sqrt(variance)
    # x = np.linspace(min(meanCell), max(meanCell), 100)
    # plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.show()


# Note: patients out of range +-10: 200801658, 200606193, 200610929, 200701370
#
# plt.hist(sdCell, 50, alpha=0.5, label='map spread',normed=True,color='green')
# plt.xlabel('single value')
# plt.ylabel('Frequency')
# plt.legend(loc='upper left')
# plt.show()

# Note: patients above 5: 200801658 21.701085922156444, 200606193 25.6532265835603, 200610929 19.887989619324294, 200701370 22.627171920841946


# mapCor=pd.read_csv(r"../Data/120x60 Data/200700427.csv",header=None).as_matrix()
# corruptMap = sns.heatmap(mapCor, center=0,xticklabels=phi,yticklabels=theta)
# corruptMap.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# plt.show()
#
## Print patient number of corrupt maps
# print(str(AllPatients.query("patientList == 200701370").patientNumber))
# print(str(AllPatients.query("patientList == 200700427").patientNumber))
# print(str(AllPatients.query("patientList == 200610929").patientNumber))
# print(str(AllPatients.query("patientList == 200606193").patientNumber))
# print(str(AllPatients.query("patientList == 200600383").patientNumber))
# print(str(AllPatients.query("patientList == 200511824").patientNumber))


def main():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)

    enhancedDF = addCalcCols(dataDirectory, allPatients.allPatients)
    plotHist(enhancedDF['mean'], 'red', 75, "Mean Radial Difference")
    plotHist(enhancedDF['sd'], 'blue', 75, "Standard Deviation of Radial Difference")
    #


if __name__ == '__main__':
    main()
