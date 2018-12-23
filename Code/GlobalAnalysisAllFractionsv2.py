#Created: 27/11/2018

#import librarys
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#import seaborn as sns

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
corrupt = {'196708754','200801658','201201119','200911702'}

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
# Plot a scatter graph for the volume of contour versus auto-contour
# =============================================================================

# Fitting a linear regression for comparison
fit = np.polyfit(AllPatients["volumeContour"],AllPatients["volumeContourAuto"],1)
fit_fn = np.poly1d(fit) 

# Fitting the graph
fig = plt.figure()
x = np.linspace(0, 160, 1000) #Plot straight line
y = x
plt.scatter(PatientRecurrencew16Frac["volumeContour"], PatientRecurrencew16Frac["volumeContourAuto"],c='r',label='Recur')
plt.scatter(PatientNonRecurrencew16Frac["volumeContour"], PatientNonRecurrencew16Frac["volumeContourAuto"],c='b',label='Non-Recur')
plt.plot(x,y,linestyle = 'solid') # y = x line
#plt.plot(x,x,'yo', AllPatients["volumeContour"], fit_fn(AllPatients["volumeContour"]), '--k') # linear fit
#plt.xlim(0, 150)
#plt.ylim(0, 130)
plt.xlabel('Manual contour volume [cm$^3$]')
plt.ylabel('Automatic contour volume [cm$^3$]')
plt.legend(loc='upper left');
plt.grid(True)
#fig.savefig(SaveDirect + '16Frac/VolumeScatterPlots16Frac.png')
#plt.show()


# =============================================================================
# Plot histograms for these patients volume difference & DSC
# =============================================================================

DSCbins = [0,0.5,0.6,0.7,0.8,0.9,1]
VolBins = [-40,-16,-10,-2.5,2.5,10,16,40]

#DSCbins = 50
#VolBins = 50

    # Plot the dice coefficient WITHOUT the corrupted data
fig1 = plt.figure()
plt.hist(PatientRecurrencew16Frac['DiceCoef'], DSCbins, alpha=0.5, label='Recurrence',normed=True,color='red')
plt.hist(PatientNonRecurrencew16Frac['DiceCoef'], DSCbins, alpha=0.5, label='Non-recurrence',normed=True,color='green')
plt.xlabel('Dice Coefficient')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
#fig1.savefig(SaveDirect + '16Frac/DicePatientRecurrencew16Frac.png')
#plt.show()

#new_path = SaveDirect + '16Frac/PatientRecurrencew16FracStats.txt'
#stats = open(new_path,'w')
#stats.write("Recur patients with 16 Fraction data DSC: \n")
#stats.write(str(PatientRecurrencew16Frac['DiceCoef'].describe()))
#stats.write("\n Non-recur patients with 16 Fraction data DSC: \n")
#stats.write(str(PatientNonRecurrencew16Frac['DiceCoef'].describe()))
#stats.write("Recur patients with 16 Fraction data VolDiff: \n")
#stats.write(str(PatientNonRecurrencew16Frac['volumeContourDifference'].describe()))
#stats.write("\n Non-recur patients with 16 Fraction data VolDiff: \n")
#stats.write(str(PatientNonRecurrencew16Frac['volumeContourDifference'].describe()))
#stats.close()

    # Plot the volume difference WITHOUT corrupted data
fig2 = plt.figure()
plt.hist(PatientRecurrencew16Frac['volumeContourDifference'], 50, alpha=0.5, label='Recurrence',normed=True,color='red')
plt.hist(PatientNonRecurrencew16Frac['volumeContourDifference'], 50, alpha=0.5, label='Non-recurrence',normed=True,color='green')
plt.xlabel('Volume difference between contour and auto-contour, $Delta V$')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
#fig2.savefig(SaveDirect + '16Frac/VolumeDifferencePatientRecurrencew16FracHighBin.png')
#plt.show()

print(str(AllPatients.query("volumeContourDifference < -50").risk))
print(str(AllPatients.query("volumeContourDifference > 50").risk))

#new_path = SaveDirect + 'General/PotentialAnomoliesPatientNumber16Frac.txt'
#stats = open(new_path,'w')
#stats.write(str(AllPatients.query("volumeContourDifference > 50").risk))
#stats.close()


#new_path = SaveDirect + 'General/PotentialAnomoliesPatientNumber16Frac.txt'
#stats = open(new_path,'w')
#stats.write("Recur patients > 50: \n")
#stats.write(str(PatientRecurrencew16Frac.query("volumeContourDifference > 50").patientNumber))
#stats.write("\n Recur patients < -50: \n")
#stats.write(str(PatientRecurrencew16Frac.query("volumeContourDifference < -50").patientNumber))
#stats.write("\n Non-Recur patients > 50: \n")
#stats.write(str(PatientNonRecurrencew16Frac.query("volumeContourDifference > 50").patientNumber))
#stats.write("\n Non-Recur patients < -50: \n")
#stats.write(str(PatientNonRecurrencew16Frac.query("volumeContourDifference < -50").patientNumber))
#stats.close()