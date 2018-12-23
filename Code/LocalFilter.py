#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# =============================================================================
# Load patients list data for all fractions
# =============================================================================

Patients1920Frac = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/AllData19Frac.csv")
PatientsOld16Frac = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/AllDataOld16Frac.csv")
SaveDirect = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/"
#PatientsNew16Frac = pd.read_csv(r"C:/Users/Alexander/Documents/4th_year/Lab/Data/60x60Data/GlobalDifference/AllDataNew16Frac.csv")

# Concatonate all the patients into a singular dataframe
AllPatients = pd.concat([Patients1920Frac,PatientsOld16Frac])

# =============================================================================
# Specify the corrupt patients to be filtered out of analysis
# =============================================================================

# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}

# Join two sets together
listRemovedPatients = corrupt.union(atlas)

# Filter corrupt or biased patients from list
FilteredPatients = AllPatients[~AllPatients['patientList'].isin(listRemovedPatients)]

# =============================================================================
# Group the patients by fractions, and recurrence
# =============================================================================

# Access patient ID numbers and recurrence
PatientID = AllPatients["patientList"]
Recurrence = AllPatients["reccurrence"]

# Get total number of patients
totalPatients = PatientID.size

# Group patients by recurrence
AllPatientsGrouped = FilteredPatients.groupby('reccurrence')
PatientsWhoRecur = pd.concat([AllPatientsGrouped.get_group('1'),AllPatientsGrouped.get_group('YES')])
PatientsWhoDontRecur = pd.concat([AllPatientsGrouped.get_group('0'),AllPatientsGrouped.get_group('censor'),AllPatientsGrouped.get_group('NO')])
 
# Group patients with fractions
#PatientRecurrencew20Frac = PatientsWhoRecur.groupby('fractions').get_group(20)
PatientRecurrencew19Frac = PatientsWhoRecur.groupby('fractions').get_group(19)
PatientRecurrencew16Frac = PatientsWhoRecur.groupby('fractions').get_group(16)
PatientNonRecurrencew20Frac = PatientsWhoDontRecur.groupby('fractions').get_group(20)
PatientNonRecurrencew19Frac = PatientsWhoDontRecur.groupby('fractions').get_group(19)
PatientNonRecurrencew16Frac = PatientsWhoDontRecur.groupby('fractions').get_group(16)

# =============================================================================
# # Read in the patients map and store in correct container
# =============================================================================
# Patient map containers
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []


corLocal = {'200801658','200606193','200610929','200701370'}

'''
returns sdValue for a given patientMap
'''
def calcPatientMapSD(patientMap):
    sxx=0
    mean = calcPatientMapMean(patientMap)
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff-mean)**2
    sdValue = np.sqrt(sxx/(patientMap.size-1))
    return (mean, sdValue)
  
def calcPatientMapMean(patientMap):
    return (sum(patientMap.flatten()))/patientMap.size

def loadPatientMap(dataDirectory, patientId): 
    file = r"%s/%s.csv"%(dataDirectory,patientId)
    return pd.read_csv(file,header=None).as_matrix()


   
      
# Read in map
dataDirectory = r"/Users/Tom/Documents/University/ProstateCode/Data/120x60 Data"   

def extractPatientSDVals(dataDir, allPatientsDF):
    meanCell = []
    sdCell = []
    totalPatients = len(allPatientsDF)
    for x in range(0, totalPatients):
        name = str(allPatientsDF["patientList"].iloc[x])
        if name not in corLocal:
            patientMap = loadPatientMap(dataDirectory, name)
            (mean, sdValue) = calcPatientMapSD(patientMap) 
            meanCell.append(mean)
            sdCell.append(sdValue)
    return (meanCell,sdCell)

# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================
# Create Arrays
#phi = []; theta =[]
#for i in range(0,120):
#    phi.append('')
#for i in range(0,60):
#    theta.append('')
## Define ticks
#phi[0] = 0; phi[30] = 90; phi[60] = 180; phi[90] = 270; phi[119] = 360;
#theta[0] = -90; theta[30] = 0; theta[59] = 90
def plotHist(data, colour):
    result=plt.hist(data,bins=50, alpha=0.5, label='map mean',color=colour)
    plt.xlabel('single value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.xlim((min(data), max(data)))
    #mean = np.mean(meanCell)
    #variance = np.var(meanCell)
    #sigma = np.sqrt(variance)
    #x = np.linspace(min(meanCell), max(meanCell), 100)
    #plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.show()



# Note: patients out of range +-10: 200801658, 200606193, 200610929, 200701370
#
#plt.hist(sdCell, 50, alpha=0.5, label='map spread',normed=True,color='green')
#plt.xlabel('single value')
#plt.ylabel('Frequency')
#plt.legend(loc='upper left')
#plt.show()

# Note: patients above 5: 200801658 21.701085922156444, 200606193 25.6532265835603, 200610929 19.887989619324294, 200701370 22.627171920841946


#mapCor=pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/120x60 Data/200700427.csv",header=None).as_matrix()
#corruptMap = sns.heatmap(mapCor, center=0,xticklabels=phi,yticklabels=theta)
#corruptMap.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
#plt.show()
#
## Print patient number of corrupt maps
#print(str(AllPatients.query("patientList == 200701370").patientNumber))
#print(str(AllPatients.query("patientList == 200700427").patientNumber))
#print(str(AllPatients.query("patientList == 200610929").patientNumber))
#print(str(AllPatients.query("patientList == 200606193").patientNumber))
#print(str(AllPatients.query("patientList == 200600383").patientNumber))
#print(str(AllPatients.query("patientList == 200511824").patientNumber))
    
    
    
def main():
    (meanVals, sdVals) =  extractPatientSDVals(dataDirectory,AllPatients)
    plotHist(meanVals, 'red')
    plotHist(sdVals, 'blue')
    
    
main()