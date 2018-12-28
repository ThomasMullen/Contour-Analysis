#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns;

from Code.AllPatients import AllPatients

sns.set()
import matplotlib.pyplot as plt
import pandas as pd



SaveDirect = "/Users/Tom/Documents/University/ProstateCode/AllDataResults/2018_12_22/"

# =============================================================================
# Load patients list data for all fractions
# =============================================================================
AllData19Frac = pd.read_csv(r"../../Data/OnlyProstateResults/Global/AllData19Frac_old.csv")
AllData16Frac_old = pd.read_csv(r"../../Data/OnlyProstateResults/Global/AllData16Frac_old.csv")
AllData16Frac = pd.read_csv(r"../../Data/OnlyProstateResults/Global/AllData16Frac.csv")

AllPatients = pd.concat([AllData19Frac,AllData16Frac_old,AllData16Frac])

# Access patient ID numbers and recurrence
PatientID = AllPatients["patientList"]
Recurrence = AllPatients["Recurrence"]

# Atlas or Corrupt
atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
corrupt19Frac = {}#'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'
corrupt16Frac = {}#'200701370','200700427','200610929','200606193','200600383','200511824'

# Join two sets together
listRemovedPatients = atlas
#listRemovedPatients = corrupt19Frac.union(atlas)

# Filter corrupt or biased patients from list
FilteredPatients = AllPatients[~AllPatients['patientList'].isin(listRemovedPatients)]

# Group patients by recurrence
AllPatientsGrouped = FilteredPatients.groupby('Recurrence')
patientsWithRecurrence = pd.concat([AllPatientsGrouped.get_group('1'),AllPatientsGrouped.get_group('YES')])
patientsWithoutRecurrence = pd.concat([AllPatientsGrouped.get_group('0'),AllPatientsGrouped.get_group('censor'),AllPatientsGrouped.get_group('NO')])


# =============================================================================
# Global Analysis
# =============================================================================

# Unity graph without cut
fig = plt.figure()
x = np.linspace(-10, 180, 1000) #Plot straight line
y = x
#plt.scatter(patientsWithRecurrence["volumeContour"], patientsWithRecurrence["volumeContourAuto"],c='r',label='Recurrence')
plt.scatter(patientsWithoutRecurrence["volumeContour"], patientsWithoutRecurrence["volumeContourAuto"],c='b',label='Non-Recurrence')
plt.plot(x,y,linestyle = 'solid') # y = x line
#plt.plot(x,x,'yo', AllPatients["volumeContour"], fit_fn(AllPatients["volumeContour"]), '--k') # linear fit
plt.xlim(0, 240)
plt.ylim(0, 180)
plt.xlabel('Manual contour volume [cm$^3$]')
plt.ylabel('Automatic contour volume [cm$^3$]')
plt.legend(loc='upper left');
plt.grid(True)
#fig.savefig(SaveDirect + 'AllPatientsContourtoAutoContourCheckNonRecurrence.png')
#plt.show()
fig.clear()
# Plot LGobal Histogram with no cuts
DSCbins = 50
VolBins = 50


fig1 = plt.figure()
plt.hist(patientsWithRecurrence['DSC'], DSCbins, alpha=0.5, label='Recurrence',normed=True,color='red')
plt.hist(patientsWithoutRecurrence['DSC'], DSCbins, alpha=0.5, label='Non-recurrence',normed=True,color='green')
plt.xlabel('Dice Coefficient')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
#fig1.savefig(SaveDirect + 'DicePatientAllPatients.png')
#plt.show()
fig1.clear()

# Potential Rogue Patients based on DSC
rogueDSCPAtient = AllPatients.query("DSC < 0.4")

#new_path = SaveDirect + 'Outliers/PotentialAnomoliesBasedOnDSC.txt'
#stats = open(new_path,'w')
#stats.write("\n Patients with DCS < 0.4: \n")
#stats.write(str(AllPatients.query("DSC < 0.4")))
#stats.write("\n Patients with DCS < 0.2: \n")
#stats.write(str(AllPatients.query("DSC < 0.2")))
#stats.close()

fig2 = plt.figure()
plt.hist(AllPatients['volumeContourDifference'], DSCbins, alpha=0.5, label='Recurrence',normed=True,color='blue')
plt.xlabel('Volume difference between contour and auto-contour, $\Delta V$')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.xlim((min(AllPatients['volumeContourDifference']), max(AllPatients['volumeContourDifference'])))
#fig2.savefig(SaveDirect + 'VolumeDifferenwAllPatientsNoDiscrim.png')
#plt.show()

fig1 = plt.figure()
plt.hist(patientsWithRecurrence['volumeContourDifference'], DSCbins, alpha=0.5, label='Recurrence',normed=True,color='red')
plt.hist(patientsWithoutRecurrence['volumeContourDifference'], 50, alpha=0.5, label='Non-recurrence',normed=True,color='green')
plt.xlabel('Volume difference between contour and auto-contour, $\Delta V$')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.xlim((min(patientsWithRecurrence['volumeContourDifference']), max(patientsWithRecurrence['volumeContourDifference'])))
# Plot Gaussian
#mean = np.mean(patientsWithRecurrence['volumeContourDifference'])
#variance = np.var(patientsWithRecurrence['volumeContourDifference'])
#sigma = np.sqrt(variance)
#x = np.linspace(min(patientsWithRecurrence['volumeContourDifference']), max(patientsWithRecurrence['volumeContourDifference']), 100)
#plt.plot(x, mlab.normpdf(x, mean, sigma))
#fig1.savefig(SaveDirect + 'VolumeDifferenwAllRecPatients.png')
#plt.show()

#new_path = SaveDirect + 'Outliers/title.txt'
#stats.write("\n Patients with Recurrence Volume Difference Distribution: \n")
#stats.write(str(patientsWithRecurrence['volumeContourDifference'].describe()))
#stats.write("\n Patients with Volume Differenc > 25: \n")
#stats.write(patientsWithRecurrence.query("volumeContourDifference > 25").to_string())
#stats.write("\n Patients with Volume Differenc < -25: \n")
#stats.write(patientsWithRecurrence.query("volumeContourDifference < -25").to_string())
#stats.write(patientsWithRecurrence.sort_values(by=['volumeContourDifference']).tail(n=20).to_string())
#stats.write(patientsWithRecurrence.sort_values(by=['volumeContourDifference']).head(n=20).to_string())
#stats.close()

# Potential Rogue Patients based on DSC
rogueDSCPatients = AllPatients.query("volumeContourDifference < 0.4")
rogueVolDiffPatients = AllPatients.query("volumeContourDifference < -14.873734283447202" or "volumeContourDifference > 31.72608489990239")
# =============================================================================
# Local Analysis
# =============================================================================

