# Alex and Tom
# Date 11/12.2018
# Data mining analysis code for mean local prostate map
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from scipy.stats import ttest_ind

from pymining import imagesTTest, permutationTest


#Directory Main
tomLocalAnalysis = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/"
tomRadialMap = "/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/RadialDifference/"
tomProstateLink = "Tom/Documents/University/ProstateCode/Data/60x60 Data/GlobalDifference/"
#AllDataOld16Frac

# Load patients list data
AdmirePatientList = pd.read_csv(r"/Users/"+tomProstateLink+"AllData19Frac.csv")
# Access patient ID numbers and recurrence

PatientID = AdmirePatientList["patientList"]
Recurrence = AdmirePatientList["reccurrence"]

# Create containers to store patients prostate maps
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

# Get total number of patients
totalNumberPatients = PatientID.size

atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
corrupt19Frac = {'196708754','200801658','201201119','20091702'}

#create (60,60,91) np array
patientMapNonRecurrenceContainer = np.zeros((60,60,60)) # Non-Recurrence containter
patientMapRecurrenceContainer = np.zeros((60,60,31)) # Recurrence containter

# Read in the patients map and store in correct container
i = 0
j = 0

for x in range(0, totalNumberPatients):
    name = str(PatientID.iloc[x])
    patientMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/Data/60x60 Data/RadialDifference/19Fractions/"+name+".csv",header=None).as_matrix()

    if name in atlas or name in corrupt19Frac:
        print("Not including patient: " + name)

    elif Recurrence.loc[x] == '1':    	
    	patientMapRecurrenceContainer[:,:,i] = patientMap[:,:]
    	i += 1#x/x
    else:    	
    	patientMapNonRecurrenceContainer[:,:,j] = patientMap[:,:]
    	j += 1#x/x,

# Concatenate the two
totalPatients = np.concatenate((patientMapRecurrenceContainer,patientMapNonRecurrenceContainer), axis=-1)

# Label first 31 recurring as 1
labels = np.concatenate((np.ones((31,)), np.zeros((60,))))

# ## Use scipy ttest, we should get the same result later
# print(ttest_ind(patientMapRecurrenceContainer, patientMapNonRecurrenceContainer, equal_var=False, axis=-1))


# ## Use pymining to get the t statistic
print(imagesTTest(totalPatients, labels).flatten()[0])
tStat = imagesTTest(totalPatients, labels)


# ## Now use pymining to get DSC cuts global p value. It should be similar to that from scipy
globalp, tthresh = permutationTest(totalPatients, labels)
print(globalp)

# Plot Threshold histogram

fig = plt.figure()
plt.hist(tthresh, 20, alpha=0.5, label='t-Test Threshold',normed=True,color='green')
plt.xlabel('t value')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
# fig.savefig('/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/T-testHist.png')
plt.show()

# Plot Threshhold Map
tThreshMap = np.zeros((60,60))
tThreshMap = tStat[0,:,:]


# fig = plt.figure()
# plt.imshow(tThreshMap, cmap = 'hot', interpolation='nearest')
# plt.xlabel('Phi')
# plt.ylabel('Theta')
# fig.savefig('/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/tTestHeatMap.png')
# plt.show()

tThresh = sns.heatmap(tThreshMap,  center=0)
tThresh.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
sns.set_style("ticks")
plt.show()

