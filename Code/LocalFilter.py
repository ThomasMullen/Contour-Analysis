
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns;

from Code.AllPatients import AllPatients, recurrenceGroups

sns.set()
import matplotlib.pyplot as plt


# =============================================================================
# Load patients list data for all fractions
# =============================================================================

SaveDirect = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/"




# =============================================================================
# Group the patients by fractions, and recurrence
# =============================================================================
def patientRecurrenceFracs():
    (PatientsWhoRecur, PatientsWhoDontRecur) = load_global_patients().recurrenceGroups()

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

def calcPatientMapSD(patientMap):
    sxx = 0
    mapMean = (sum(patientMap.flatten())) / patientMap.size
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff - mapMean) ** 2
    sdValue = np.sqrt(sxx / (patientMap.size - 1))
    return (mapMean, sdValue)


'''
Specify the corrupt patients to be filtered out of analysis
# ===================================================================
'''
def load_global_patients():
    # List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
    atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
             '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}

    '''we have identified these corrupted from previous contour. we need to check '''
    expected_corrupt_to_check = {'200701370', '200700427', '200610929', '200606193', '200600383', '200511824',
                                 '196708754', '200801658', '201201119', '200911702', '200701370', '200700427',
                                 '200610929', '200606193', '200600383', '200511824'}

    allPatients = AllPatients(r"../Data/OnlyProstateResults/Global",
                              ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    allPatients.removePatients(atlas)
    return allPatients

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

'''
Finds the Mean and the SD of the radial difference at each solid angle for each patient. And appends data to global patient df
'''
def radial_mean_sd_for_patients(dataDir, allPatientsDF):
    df = allPatientsDF.assign(mean=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_mean(dataDir, x)))
    df2 = df.assign(sd=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_sd(dataDir, x)))
    return df2


# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================

def plotHist(data, colour, bin, name="Single Value"):
    result = plt.hist(data, bins=bin, alpha=0.5, label='map sd value', color=colour)
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

def partition_patient_data_with_outliers( data, lower_bound, upper_bound):

    lower_cut_off = np.percentile(data['sd'], lower_bound)
    upper_cut_off = np.percentile(data['sd'], upper_bound)
    print("%s, %s" % (lower_cut_off, upper_cut_off))
    selected_patients = data[data.sd.between(lower_cut_off, upper_cut_off)]
    lower_patients_outliers = data[data.sd < lower_cut_off]
    upper_patients_outliers = data[data.sd > upper_cut_off]
    return (selected_patients, lower_patients_outliers, upper_patients_outliers)



def test_filtered():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        enhancedDF, 10, 90)
    lower_patients_outliers.to_csv('%s/lower_patients_outliers.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers.csv' % outputDirectory)

    patient_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)

    plotHist(patient_who_recur['sd'], 'blue', 75, "Standard Deviation of Radial Difference recurrence")
    plotHist(patients_who_dont_recur['sd'], 'green', 75, "Standard Deviation of Radial Difference no recurrence ")


def main():
    test_filtered

if __name__ == '__main__':
    main()
# TODO Cut off upper 10 percentile for local sd
# TODO Cut off ContourVolumeDifference Upper and Lower
# TODO Print Histograms
# TODO Use Local Combined Map to produce an average and variance Radial Patient for recurrence and non-recurrence
