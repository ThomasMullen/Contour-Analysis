#Created: 27/11/2018

#import librarys
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#def allbut(*names):
#    names = set(names)
#    return [item for item in levels if item not in names]

# Load patients list data
AllPatients = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/AllData16Frac.csv")

# Access patient ID numbers and recurrence
PatientID = AllPatients["patientList"]
Recurrence = AllPatients["reccurrence"]

# Get total number of patients
totalPatients = PatientID.size

# List the patient ID's of those who are contained in our ATLAS and have corrupted packs
atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
corrupt = {'196708754','200801658','201201119'}

# Join two sets together
listRemovedPatients = corrupt.union(atlas)

#Filter corrupt or biased patients from list
FilteredPatients = AllPatients[~AllPatients['patientList'].isin(listRemovedPatients)]

# Group patients by recurrence
#AllPatientsGrouped = FilteredPatients.groupby('reccurrence')
#PatientsWhoRecur = AllPatientsGrouped.get_group('1')
#PatientsWhoDontRecur = pd.concat([AllPatientsGrouped.get_group('0'),AllPatientsGrouped.get_group('censor')])
 
bins = 50

# =============================================================================
# Plot a scatter graph for the volume of contour versus auto-contour
# =============================================================================

# Fitting a linear regression

# Fitting a linear regression
#fit = np.polyfit(FilteredPatients["volumeContour"],FilteredPatients["volumeContourAuto"],1)
#fit_fn = np.poly1d(fit) 


fig = plt.figure()
x = np.linspace(0, 200, 1000) #Plot straight line
#plt.scatter(PatientsWhoRecur["volumeContour"], PatientsWhoRecur["volumeContourAuto"],c='r',label='Recur')
#plt.scatter(PatientsWhoDontRecur["volumeContour"], PatientsWhoDontRecur["volumeContourAuto"],c='b',label='Non-Recur')
plt.scatter(FilteredPatients["volumeContour"], FilteredPatients["volumeContourAuto"],c='b',label='Non-Recur')
plt.plot(x,x,linestyle = 'solid')
#plt.xlim(0, 150)
#plt.ylim(0, 130)
plt.xlabel('Manual contour volume [cm$^3$]')
plt.ylabel('Automatic contour volume [cm$^3$]')
plt.legend(loc='upper left');
plt.grid(True)
fig.savefig('/Users/Tom/Documents/University/ProstateCode/LinearRegressionAutoToContVol.png')
plt.show()


# =============================================================================
# Plot histograms for these patients volume difference & DSC
# =============================================================================

#    # Plot the dice coefficient
#fig1 = plt.figure()
#plt.hist(PatientsWhoRecur['DiceCoef'], bins, alpha=0.5, label='Recurrence',normed=True,color='red')
#plt.hist(PatientsWhoDontRecur['DiceCoef'], bins, alpha=0.5, label='Non-recurrence',normed=True,color='green')
#plt.xlabel('Dice Coefficient')
#plt.ylabel('Frequency')
#plt.legend(loc='upper left')
#fig1.savefig('/Users/Tom/Documents/University/ProstateCode/corruptDiceCoeff.png')
#plt.show()
#
#    # Plot the volume difference
#fig2 = plt.figure()
#plt.hist(PatientsWhoRecur['volumeContourDifference'], bins, alpha=0.5, label='Recurrence',normed=True,color='red')
#plt.hist(PatientsWhoDontRecur['volumeContourDifference'], bins, alpha=0.5, label='Non-recurrence',normed=True,color='green')
#plt.xlabel('Volume difference between contour and auto-contour, $\Delta V$')
#plt.ylabel('Frequency')
#plt.legend(loc='upper right')
#fig2.savefig('/Users/Tom/Documents/University/ProstateCode/corruptGlobalVolDiff.png')
#plt.show()
