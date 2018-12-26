#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns;

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

'''we have identified these corrupted from previous contour. we need to check '''
expected_corrupt_to_check = {'200701370','200700427','200610929','200606193','200600383','200511824', '196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}


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


def calcPatientMapSD(patientMap):
    sxx = 0
    mapMean = (sum(patientMap.flatten())) / patientMap.size
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff - mapMean) ** 2
    sdValue = np.sqrt(sxx / (patientMap.size - 1))
    return (mapMean, sdValue)

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

def calcPatientMapSD2_mean(dataDir, patientId):
    (m, s) =  calcPatientMapSD2(dataDir, patientId)
    return m

def calcPatientMapSD2_sd(dataDir, patientId):
    (m, s) = calcPatientMapSD2(dataDir, patientId)
    return s


def addCalcCols(dataDir, allPatientsDF):
    df = allPatientsDF.assign(mean=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_mean(dataDir, x)))
    df2 = df.assign(sd=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_sd(dataDir, x)))
    return df2


# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================

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


'''
plots a heat map of patients radial map: expect df of local field passed in.
'''
def plot_heat_map(data,title):
    phi = [];
    theta = []
    for i in range(0, 120):
        phi.append('')
    for i in range(0, 60):
        theta.append('')
    # Define ticks
    phi[0] = 0;
    phi[30] = 90;
    phi[60] = 180;
    phi[90] = 270;
    phi[119] = 360;
    theta[0] = -90;
    theta[30] = 0;
    theta[59] = 90
    map = data.as_matrix()
    heat_map = sns.heatmap(map, center=0, xticklabels=phi, yticklabels=theta)
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title = title)
    plt.show()



def main():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)

    enhancedDF = addCalcCols(dataDirectory, allPatients.allPatients)
    lower_cut_off = np.percentile(enhancedDF['sd'],10)
    upper_cut_off = np.percentile(enhancedDF['sd'], 90)
    print("%s, %s" % (lower_cut_off,upper_cut_off))
    selected_patients = enhancedDF[enhancedDF.sd.between(0.224015772555, 0.656248627237)]
    lower_patients_outliers = enhancedDF[enhancedDF.sd < 0.224015772555].to_csv('%s/lower_patients_outliers.csv'%outputDirectory)
    upper_patients_outliers = enhancedDF[enhancedDF.sd > 0.656248627237].to_csv('%s/upper_patients_outliers.csv'%outputDirectory)



    plotHist(enhancedDF['sd'], 'blue', 75, "Standard Deviation of Radial Difference with full data")
    plotHist(selected_patients['sd'], 'green', 75, "Standard Deviation of Radial Difference with outliers removed ")
    # print(upper_patients.head())
if __name__ == '__main__':
    main()
# TODO Cut off upper 10 percentile for local sd
# TODO Cut off ContourVolumeDifference Upper and Lower
# TODO Print Histograms
# TODO Use Local Combined Map to produce an average and variance Radial Patient for recurrence and non-recurrence
