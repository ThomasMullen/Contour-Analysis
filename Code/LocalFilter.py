#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom and Alex
"""

import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
from AllPatients import AllPatients, separate_by_recurrence

sns.set()

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
    mapMax = patientMap.max()

    return (mapMean, sdValue, mapMax)


'''
Specify the corrupt patients to be filtered out of analysis
'''


def load_global_patients():
    """
    Loads all data frames, removes atlas and corrupt patients and patients that have PC >= T3
    :return: All global data of all patients in DSC cuts single df
    """
    # List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
    atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
             '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}

    '''we have identified these corrupted from previous contour. we need to check '''
    expected_corrupt_to_check = {200710358, 200705181, 200807021, 200502036, 200606193, 200303191, \
                                 200708782, 200503218, 200511135, 200301067, 200501574, 200502944, \
                                 201001322, 201007576, 201009663, 200502133, 200502319, 200502439, 200502794, \
                                 200503516, 200508940, 200511480, 199301039, 200405868, 200504358, 200701300, 200502136, \
                                 200609194, 200704603}
    # {'200701370', '200700427', '200610929', '200606193', '200600383', '200511824','196708754', '200801658',
    # '201201119', '200911702', '200701370', '200700427','200610929', '200606193', '200600383', '200511824'}

    patients_ID_to_exclude = atlas.union(expected_corrupt_to_check)
    all_patients = AllPatients(r"../Data/OnlyProstateResults/Global",
                               ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    all_patients.removePatients(patients_ID_to_exclude)
    all_patients.remove_stageT3()
    return all_patients


def patients_mean_sd_maxvalue(dataDir, patientId):
    file = r"%s/%s.csv" % (dataDir, patientId)
    matrix = pd.read_csv(file, header=None).values
    result = calcPatientMapSD(matrix)
    return result


'''
Finds the Mean and the SD of the radial difference at each solid angle for each patient. And appends data to global patient df
'''


def radial_mean_sd_for_patients(dataDir, allPatientsDF):
    df = allPatientsDF.assign(
        mean_sd_maxV=lambda df: df["patientList"].map(lambda x: patients_mean_sd_maxvalue(dataDir, x))) \
        .assign(mean=lambda df: df["mean_sd_maxV"].map(lambda x: x[0])) \
        .assign(sd=lambda df: df["mean_sd_maxV"].map(lambda x: x[1])) \
        .assign(maxval=lambda df: df["mean_sd_maxV"].map(lambda x: x[2]))
    return df


def partition_patient_data_with_outliers(data, lower_bound, upper_bound, discriminator_fieldname="sd"):
    lower_cut_off = np.percentile(data[discriminator_fieldname], lower_bound)
    upper_cut_off = np.percentile(data[discriminator_fieldname], upper_bound)
    print("%s, %s" % (lower_cut_off, upper_cut_off))
    selected_patients = data[data[discriminator_fieldname].between(lower_cut_off, upper_cut_off)]
    lower_patients_outliers = data[data[discriminator_fieldname] < lower_cut_off]
    upper_patients_outliers = data[data[discriminator_fieldname] > upper_cut_off]

    selected_patients.sort_values([discriminator_fieldname])

    return selected_patients, lower_patients_outliers, upper_patients_outliers


def partition_patient_data_with_range(data, lower_bound, upper_bound, discriminator_fieldname):
    ''' A function to return DSC cuts dataframe of all patients within DSC cuts certain range '''
    lower_cut_off = data.loc[data[discriminator_fieldname] < lower_bound]
    upper_cut_off = data.loc[data[discriminator_fieldname] > upper_bound]
    selected_patients = data[data[discriminator_fieldname].between(lower_cut_off, upper_cut_off)]
    return selected_patients


def return_patient_sample_range(data, size, lower_bound, upper_bound):
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(data,
                                                                                                               lower_bound,
                                                                                                               upper_bound)
    return (selected_patients.sample(n=size), lower_patients_outliers, upper_patients_outliers)


def test_filtered():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()  # returns the data cleaned for atlas and corruption
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        enhancedDF, 10, 90)
    lower_patients_outliers.to_csv('%s/lower_patients_outliers.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers.csv' % outputDirectory)

    #     Output sample of patients within certain percentiles
    sample1 = return_patient_sample_range(enhancedDF, 5, 10, 20)
    sample1[0].to_csv('%s/sample_set1.csv' % outputDirectory)
    sample2 = return_patient_sample_range(enhancedDF, 5, 30, 50)
    sample2[0].to_csv('%s/sample_set2.csv' % outputDirectory)
    sample3 = return_patient_sample_range(enhancedDF, 5, 50, 80)
    sample3[0].to_csv('%s/sample_set3.csv' % outputDirectory)
    sample4 = return_patient_sample_range(enhancedDF, 5, 80, 90)
    sample4[0].to_csv('%s/sample_set4.csv' % outputDirectory)


def main():
    test_filtered()
    # test_on_single_map()


if __name__ == '__main__':
    main()
