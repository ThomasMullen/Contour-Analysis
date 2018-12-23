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

# Read in map
for x in range(0, totalPatients):
    name = str(PatientID.iloc[x])
    patientMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/120x60 Data/"+name+".csv",header=None)
#    plt.imshow(patientMap, cmap='hot', interpolation='nearest')
#    plt.show()
#    print(name)
    if name in atlas or name in corrupt:
        print("Not including patient: " + name)
        # Reacurrence
    elif Recurrence.iloc[x] == '1':
        patientMapRecurrenceContainer.append(patientMap)
    elif Recurrence.iloc[x] == 'YES':
        patientMapRecurrenceContainer.append(patientMap)
        # Non Recurrence
    else:
        patientMapNonRecurrenceContainer.append(patientMap)
        # print

# =============================================================================
#  Create Mean Patient Map
# =============================================================================
        
# Calculate Mean and Variance Heat map for patient recurrence
totalRecurrencePatients = pd.concat(patientMapRecurrenceContainer)
by_row_indexRec = totalRecurrencePatients.groupby(totalRecurrencePatients.index)
meanRecurrence = by_row_indexRec.mean()
varRecurrence = by_row_indexRec.var()
stdRecurrence = by_row_indexRec.std()

# Calculate Mean and Variance Heat map for patient non-recurrence
totalNonRecurrencePatients = pd.concat(patientMapNonRecurrenceContainer)
by_row_indexNonRec = totalNonRecurrencePatients.groupby(totalNonRecurrencePatients.index)
meanNonRecurrence = by_row_indexNonRec.mean()
varNonRecurrence = by_row_indexNonRec.var()

# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================
# Create Arrays
phi = []; theta =[]
for i in range(0,120):
    phi.append('')
for i in range(0,60):
    theta.append('')
# Define ticks
phi[0] = 0; phi[30] = 90; phi[60] = 180; phi[90] = 270; phi[119] = 360;
theta[0] = -90; theta[30] = 0; theta[59] = 90

mapCor=pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/120x60 Data/200700427.csv",header=None)
corruptMap = sns.heatmap(mapCor, center=0,xticklabels=phi,yticklabels=theta)
corruptMap.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
plt.show()
# =============================================================================
# # Display 2D Heat maps
# =============================================================================

# f, (recurrenceMean, recurrenceVar) = plt.subplots(1, 2)
recurrenceMean = sns.heatmap(meanRecurrence,vmax = 1, vmin = -1, center=0,xticklabels=phi,yticklabels=theta)
recurrenceMean.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig = recurrenceMean.get_figure()
# fig.savefig(tomLocalAnalysis + "16Frac" + "60x60meanRecurrenceMap.png")
# plt.show()
# fig.clear()


recurrenceVar = sns.heatmap(varRecurrence,vmax = 0.5, vmin = 0, center=0,xticklabels=phi,yticklabels=theta)
recurrenceVar.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig2 = recurrenceVar.get_figure()
# fig2.savefig(tomLocalAnalysis + "16Frac" +  "60x60varRecurrenceMap.png")
# plt.show()
# fig2.clear()

nonRecurrenceMean = sns.heatmap(meanNonRecurrence,vmax = 0.5, vmin = -0.5, center=0,xticklabels=phi,yticklabels=theta)
nonRecurrenceMean.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig3 = nonRecurrenceMean.get_figure()
#fig3.savefig(tomLocalAnalysis + "16Frac" +  "60x60meanNonRecurrenceMap.png")
# plt.show()
# fig3.clear()

nonRecurrenceVar = sns.heatmap(varNonRecurrence, vmax = 1, vmin = 0, center=0,xticklabels=phi,yticklabels=theta)
nonRecurrenceVar.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig4 = nonRecurrenceVar.get_figure()
# fig4.savefig(tomLocalAnalysis + "16Frac" +  "60x60varNonRecurrenceMap.png")
# plt.show()
# fig4.clear()

DifferenceInMean = meanRecurrence - meanNonRecurrence
DifferenceInMeanGraph = sns.heatmap(DifferenceInMean, center=0,xticklabels=phi,yticklabels=theta)
DifferenceInMeanGraph.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
fig5 = DifferenceInMeanGraph.get_figure()
# fig5.savefig(tomLocalAnalysis + "16Frac" +  "60x60differenceMeanMap.png")
# plt.show()
# fig5.clear()

# Print patient number of corrupt maps
print(str(AllPatients.query("patientList == 200701370").patientNumber))
print(str(AllPatients.query("patientList == 200700427").patientNumber))
print(str(AllPatients.query("patientList == 200610929").patientNumber))
print(str(AllPatients.query("patientList == 200606193").patientNumber))
print(str(AllPatients.query("patientList == 200600383").patientNumber))
print(str(AllPatients.query("patientList == 200511824").patientNumber))