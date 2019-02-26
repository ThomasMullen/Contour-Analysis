#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom and Alex
"""

import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()
from AllPatients import AllPatients, separate_by_recurrence

SaveDirect = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/"


# corLocal = {'200801658', '200606193', '200610929', '200701370'}

def calculate_mean_sd_max_of__patient_map(patientMap):
    '''
    This function calculates the mean, sd and maximum magnitude values for each patient map. This is important for identifying
    local anomolies
    :param patientMap: Is the radial difference map of the patient
    :return: returns the patients mean, sd, and max value
    '''
    sxx = 0
    mapMean = (sum(patientMap.flatten())) / patientMap.size
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff - mapMean) ** 2
    sdValue = np.sqrt(sxx / (patientMap.size - 1))
    # Take the magnitude of the map to identify maximum value magnitude
    patientMap = np.sqrt(pow(patientMap, 2))
    mapMax = patientMap.max()
    return mapMean, sdValue, mapMax


'''
Specify the corrupt patients to be filtered out of analysis
'''


def load_global_patients():
    """
    Concats all global df. Loads all data frames, removes atlas and corrupt patients and patients that have PC >= T3.
    Corrupt patients were identified by loading the maps of the upper/lower bounds of the selection cuts.
    :return: All global data of all patients in DSC cuts single df
    """
    # List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
    atlas = {'200511319', '200603037', '200606282',
             '200604499', '197100826', '200805129',
             '200504430', '200701790', '200512187',
             '200806697', '200512262', '200502368',
             '200504914', '200502882', '200805103',
             '201002333'}

    '''we have identified these corrupted from previous contour. we need to check '''
    # expected_corrupt_to_check = {200710358, 200705181, 200807021, 200502036, 200606193, 200303191, \ 200708782,
    # 200503218, 200511135, 200301067, 200501574, 200502944, \ 201001322, 201007576, 201009663, 200502133, 200502319,
    # 200502439, 200502794, \ 200503516, 200508940, 200511480, 199301039, 200405868, 200504358, 200701300, 200502136,
    # \ 200609194, 200704603}
    expected_corrupt_to_check = {}
    patients_ID_to_exclude = atlas.union(expected_corrupt_to_check)
    all_patients = AllPatients(r"../Data/OnlyProstateResults/16Fractions_Golden_Atlas",
                               ['All_16frac_golden_atlas_data'])
    all_patients.removePatients(patients_ID_to_exclude)
    # all_patients.remove_stageT3()
    return all_patients

def patients_mean_sd_max_value(dataDir, patientId):
    file = r"%s/%s.csv" % (dataDir, patientId)
    radial_map = pd.read_csv(file, header=None).values
    result = calculate_mean_sd_max_of__patient_map(radial_map)
    return result


'''
Finds the Mean and the SD of the radial difference at each solid angle for each patient. And appends data to global patient df
'''


def radial_mean_sd_for_patients(dataDir, allPatientsDF):
    '''
    Adds 3 additional parameters to the global dataset: the patients mean map value, the patient sd map value and the patients maximum value
    :param dataDir: The folder that contains the local radial fields
    :param allPatientsDF: The global patient dataset table
    :return: Returns three additional parameters
    '''
    df = allPatientsDF.assign(
        mean_sd_maxV=lambda df: df["patientList"].map(lambda x: patients_mean_sd_max_value(dataDir, x))) \
        .assign(mean=lambda df: df["mean_sd_maxV"].map(lambda x: x[0])) \
        .assign(sd=lambda df: df["mean_sd_maxV"].map(lambda x: x[1])) \
        .assign(maxval=lambda df: df["mean_sd_maxV"].map(lambda x: x[2]))
    return df


def partition_patient_data_with_outliers(data, lower_bound, upper_bound, discriminator_fieldname="sd"):
    '''
    Divides the data set using an upper and lower bound percentile. parameter used to partition the dataset can be any
    quantitive fieldname form the global dataset table e.g. map standard deviation, map mean, map max, volume difference,
    dsc. Note, default is set as the map standard deviation.
    :param data: The global dataset contain all the patients
    :param lower_bound: lower percentile cut
    :param upper_bound: upper percentile cut
    :param discriminator_fieldname: parameter used to partition the dataset
    :return: returns the global dataset for patient within the bounds, the upp-bound patients and the lower-bound
    patients, respectively.
    '''
    lower_cut_off = np.percentile(data[discriminator_fieldname], lower_bound)
    upper_cut_off = np.percentile(data[discriminator_fieldname], upper_bound)
    print("%s, %s" % (lower_cut_off, upper_cut_off))
    selected_patients = data[data[discriminator_fieldname].between(lower_cut_off, upper_cut_off)]
    lower_patients_outliers = data[data[discriminator_fieldname] < lower_cut_off]
    upper_patients_outliers = data[data[discriminator_fieldname] > upper_cut_off]

    selected_patients.sort_values([discriminator_fieldname])

    return selected_patients, lower_patients_outliers, upper_patients_outliers


def partition_patient_data_with_range(data, lower_bound, upper_bound, discriminator_fieldname):
    ''' A function to return DSC cuts dataframe of all patients within DSC cuts certain range. Very similiar to the
    function partition_patient_data_with_outliers but it does not include the extreme bound patients
    '''
    lower_cut_off = data.loc[data[discriminator_fieldname] < lower_bound]
    upper_cut_off = data.loc[data[discriminator_fieldname] > upper_bound]
    selected_patients = data[data[discriminator_fieldname].between(lower_cut_off, upper_cut_off)]
    return selected_patients


def return_patient_sample_range(data, size, lower_bound, upper_bound):
    '''
    same as partition_patient_data_with_outliers but returns a sample of patients within to the two bounds
    :param data:
    :param size: sample sise of selection for patients within range
    :param lower_bound:
    :param upper_bound:
    :return:
    '''
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(data,
                                                                                                               lower_bound,
                                                                                                               upper_bound)
    return (selected_patients.sample(n=size), lower_patients_outliers, upper_patients_outliers)


def select_atlas(seed=23403485, save_name='untitled', file_name='allData19Frac'):
    total_dataset = pd.read_csv('../Data/OnlyProstateResults/Global/'+file_name+'.csv')
    atlas_patients = total_dataset.sample(n=15, random_state=seed)
    print(atlas_patients['patientList'])
    atlas_patients['patientList'].to_csv('../' + save_name + '.csv')


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
