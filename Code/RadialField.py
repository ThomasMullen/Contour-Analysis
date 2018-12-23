#Created: 27/11/2018

#import librarys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
#import seaborn as sns

# Load patients list data
AdmirePatientList = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/AllData19Frac.csv")

# Access patient ID numbers and recurrence
PatientID = AdmirePatientList["patientList"]
Recurrence = AdmirePatientList["reccurrence"]

# Create containers to store patients prostate maps
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

# Get total number of patients
totalPatients = PatientID.size

# Read in the patients map and store in correct container
for x in range(0, totalPatients):
    name = str(PatientID.iloc[x])
    patientMap = pd.read_csv(r"/Users/Tom/Documents/University/ProstateCode/RadialDifference/"+name+".csv",header=None)
    
    # Reacurrence
    if Recurrence.loc[x] == '1':
        patientMapRecurrenceContainer.append(patientMap)
    #    plt.imshow(patientMap, cmap='hot', interpolation='nearest')
    #    plt.show()
    
    # Non Recurrence
    else:
        patientMapNonRecurrenceContainer.append(patientMap)
        
# Calculate Mean and Variance Heat map for patient recurrence
totalRecurrencePatients = pd.concat(patientMapRecurrenceContainer)
by_row_indexRec = totalRecurrencePatients.groupby(totalRecurrencePatients.index)
meanRecurrence = by_row_indexRec.mean()
varRecurrence = by_row_indexRec.var()

# Calculate Mean and Variance Heat map for patient non-recurrence
totalNonRecurrencePatients = pd.concat(patientMapNonRecurrenceContainer)
by_row_indexNonRec = totalNonRecurrencePatients.groupby(totalNonRecurrencePatients.index)
meanNonRecurrence = by_row_indexNonRec.mean()
varNonRecurrence = by_row_indexNonRec.var()

# Display 2D Heat maps
#f, (recurrenceMean, recurrenceVar) = plt.subplots(2, sharey=True)
recurrenceMean = sns.heatmap(meanRecurrence, center=0)
recurrenceMean.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
recurrenceVar = sns.heatmap(varRecurrence, center=0)
recurrenceVar.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
#f.subplots_adjust(hspace=0.3)
plt.show()

nonRecurrenceMean = sns.heatmap(meanNonRecurrence, center=0)
nonRecurrenceMean.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
nonRecurrenceVar = sns.heatmap(varNonRecurrence, center=0)
nonRecurrenceVar.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')


#f, (axrMean, axrVar) = plt.subplots(1, 2, sharey=True)
#
#axrMean = sns.heatmap(df_meansRec, center=0)
#axrMean.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
#axrVar = sns.heatmap(df_varRec, center=0)
#axrVar.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
#
#g, (axnrMean, axnrVar) = plt.subplots(1, 2, sharey=True)
#axnrMean = sns.heatmap(df_meansNonRec, center=0)
#axnrMean.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
#axnrVar = sns.heatmap(df_meansNonRec, center=0)
#axnrVar.set(ylabel='Theta $\dot{\Theta}$', xlabel='Azimutal $\phi$')
#plt.show()


# Plot Heat map of Mean and Variance
#plt.imshow(df_meansRec, cmap='hot', interpolation='nearest')
#plt.imshow(df_meansNonRec, cmap='hot', interpolation='nearest')